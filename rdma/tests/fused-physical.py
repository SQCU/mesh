"""Cluster correctness for physical fused collectives; no throughput claim.

Launch on two or three ranks with the usual mesh environment. Every batch
queues changing inputs before one host synchronization. Expected results are
derived locally; there are no network receipts or verification collectives
inside a batch. GPU snapshots in the persistent-output cases are test-only.
"""
import argparse
import json
import math
import time

import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition

import mesh
from routing import QueueSample


def values(source, iteration, shape, dtype):
    count = math.prod(shape)
    pattern = torch.arange(count, dtype=torch.float32).reshape(shape).remainder_(31).div_(16)
    return (pattern + source * 8 + iteration * 0.25).mul_(-1 if iteration % 2 else 1).to(dtype)


def failure(group):
    status = mesh.ffi.new('ncclResult_t *')
    mesh.check(mesh.lib.ncclCommGetAsyncError(group.comm, status))
    if status[0]:
        raise RuntimeError(mesh.ffi.string(mesh.lib.ncclGetLastError(group.comm)).decode())


def traffic(before, after):
    return [dict(cable=b['cable'], wire=b['wire'], sent_bytes=a['sent'] - b['sent'],
                 landed_bytes=a['landed'] - b['landed']) for b, a in zip(before, after)]


def payload_traffic(group, sample, before, sent, landed):
    """Exact two-rank accounting after execution; larger graphs report relay bytes.

    A local SEND CQ can trail the consumer's GPU completion slightly. This
    diagnostic poll is outside the function and does not gate payload progress.
    """
    deadline = time.monotonic() + 5
    while True:
        wires = traffic(before, sample.read())
        if group.size() != 2:
            return wires
        measured = (sum(row['sent_bytes'] for row in wires), sum(row['landed_bytes'] for row in wires))
        if measured == (sent, landed):
            return wires
        assert measured[0] <= sent and measured[1] <= landed, ('wire padding or duplicate traffic', measured, sent, landed)
        if time.monotonic() >= deadline:
            raise AssertionError(('exact payload accounting incomplete', measured, sent, landed))
        time.sleep(0.001)


def gathered(group, sample, name, sizes, dtype, strided, persistent, repeats, iterations, require_all):
    rank, pre, post = group.rank(), 2, 257 if require_all else 5
    shape = (pre, sizes[rank], post)
    storage = torch_mesh.empty(pre, sizes[rank], post * 2 + 3 if strided else post, dtype=dtype)
    source = storage[..., 1:1 + 2 * post:2] if strided else storage
    output = torch_mesh.empty(pre, sum(sizes), post, dtype=dtype) if persistent else None
    group.plan([partition.gathered(source, 1, sizes, group)])
    before, snapshots, expected = sample.read(), [], []
    for iteration in range(iterations):
        source.copy_(values(rank, iteration, shape, dtype))
        if persistent:
            torch.ops.torch_mesh.allgatherv_out(source, output, 1, sizes, group.group_name, repeats)
            snapshot = torch_mesh.empty(output.shape, dtype=dtype)
            snapshot.copy_(output)
        else:
            snapshot = torch.ops.torch_mesh.allgatherv(source, 1, sizes, group.group_name, repeats)
        snapshots.append(snapshot)
        expected.append(torch.cat([values(r, iteration, (pre, size, post), dtype) for r, size in enumerate(sizes)], 1))
    torch.mps.synchronize()
    failure(group)
    for iteration, (actual, reference) in enumerate(zip(snapshots, expected)):
        actual = actual.cpu()
        if not torch.equal(actual, reference):
            errors = (actual != reference).nonzero()
            at = tuple(errors[0].tolist())
            raise AssertionError((name, rank, iteration, len(errors), at, actual[at].item(), reference[at].item()))
    useful_sent = source.numel() * source.element_size() * iterations
    useful_landed = pre * (sum(sizes) - sizes[rank]) * post * source.element_size() * iterations
    wires = payload_traffic(group, sample, before, useful_sent, useful_landed)
    if require_all:
        assert all(row['sent_bytes'] > 0 and row['landed_bytes'] > 0 for row in wires), (name, rank, wires)
    group.plan([])
    return dict(case=name, rank=rank, dtype=str(dtype), sizes=sizes, strided=strided, persistent_output=persistent,
                repeats=repeats, calls=iterations, exact=True, channels=len(wires), traffic=wires,
                payload_padding_bytes=0, common_write_plans=True,
                two_rank_exact_payload_verified=group.size() == 2)


def summed(group, sample, dtype, iterations):
    source = torch_mesh.empty(8193, dtype=dtype)
    group.plan([partition.summed(source, group)])
    before, snapshots = sample.read(), []
    for iteration in range(iterations):
        source.copy_(values(group.rank(), iteration, source.shape, dtype))
        snapshots.append(partition._fused(source, 0, [source.numel()] * group.size(), group, True))
    torch.mps.synchronize()
    failure(group)
    for iteration, actual in enumerate(snapshots):
        expected = sum(values(rank, iteration, source.shape, dtype).float() for rank in range(group.size())).to(dtype)
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(('sum', group.rank(), dtype, iteration))
    useful = source.numel() * source.element_size() * iterations
    wires = payload_traffic(group, sample, before, useful, useful)
    group.plan([])
    return dict(case='sum', rank=group.rank(), dtype=str(dtype), calls=iterations, exact=True, traffic=wires,
                payload_padding_bytes=0, common_write_plans=True,
                two_rank_exact_payload_verified=group.size() == 2)


def migrated(group, iterations):
    """Exercise asymmetric source retirement without an application receipt."""
    from unittest.mock import patch

    topology = DeviceMesh.from_group(group, 'mps', mesh_dim_names=('migration',))
    old, new = ([4, 3, 1], [4, 2, 2]) if group.size() == 3 else ([3, 1], [2, 2])
    rank, snapshots, expected = group.rank(), [], []
    for dim in (0, 1):
        shape = (sum(old), 65536) if dim == 0 else (3, sum(old), 65536)
        global_stride = torch.empty(shape, device='meta').stride()
        for iteration in range(iterations):
            partition._attach(topology, 'migration', old)
            operands = []
            for tensor in range(4):
                full = values(0, iteration * 4 + tensor, shape, torch.float32)
                local = full.narrow(dim, sum(old[:rank]), old[rank])
                source = torch_mesh.empty(local.shape, dtype=torch.float32)
                source.copy_(local)
                operands.append(partition.from_local(source, topology, [Shard(dim)], shape=shape, stride=global_stride))
                expected.append(full.narrow(dim, sum(new[:rank]), new[rank]).contiguous())
            partition._attach(topology, 'migration', new)
            # This case must exercise indexed original-to-final writes, not
            # merely obtain correct numbers through the staging fallback.
            with patch.object(dist, 'all_to_all_single', side_effect=AssertionError('aligned migration staged')):
                snapshots.extend(t.to_local() for t in partition.follow(*operands))
    torch.mps.synchronize()
    failure(group)
    for i, (actual, reference) in enumerate(zip(snapshots, expected)):
        if not torch.equal(actual.cpu(), reference):
            raise AssertionError(('asymmetric migration', rank, i))
    return dict(case='asymmetric-migration', rank=rank, old_parts=old, new_parts=new,
                calls=len(snapshots), axes=[0, 1], exact=True, receipt_collectives=0,
                staging_collectives=0,
                note='Affine original-to-final exact-byte writes; distinct relay leases, native deferred retirement')


def byte_migrations(group, sample, iterations):
    """Exact-byte affine runs, poisoned neighbors, and more rows than descriptor entries."""
    from unittest.mock import patch
    from migration import compile_rows

    topology = DeviceMesh.from_group(group, 'mps', mesh_dim_names=('bytes',))
    old, new = ([4, 3, 1], [3, 3, 2]) if group.size() == 3 else ([3, 1], [2, 2])
    rank, snapshots, references, guards, sources, metadata = group.rank(), [], [], [], [], []

    def guarded_empty(shape, like, group=None, dtype=None):
        count = math.prod(shape)
        backing = torch_mesh.empty(count + 34, dtype=dtype or like.dtype)
        # Exact receives can be preposted immediately. Initialize guards only;
        # no outstanding GPU store may overwrite an incoming payload interval.
        backing[:17].fill_(173)
        backing[count + 17:].fill_(173)
        guards.append((backing, count, 173))
        return backing.narrow(0, 17, count).view(shape)

    before = sample.read()
    for dim, shape in ((0, (sum(old), 3)), (1, (7, sum(old), 32771)), (1, (5003, sum(old), 3))):
        plan = compile_rows(shape, dim, old, new, 1)
        trains = []
        for run in plan.runs:
            if run.source == run.destination:
                continue
            trains.append(dict(source=run.source, destination=run.destination, logical_messages=1,
                               row_bytes=run.row_bytes, rows=run.rows, source_stride=run.source_stride,
                               destination_stride=run.destination_stride, useful_bytes=run.bytes, padding_bytes=0))
        metadata.append(dict(axis=dim, shape=shape, trains=trains))
        for iteration in range(iterations):
            partition._attach(topology, 'bytes', old)
            full = (torch.arange(math.prod(shape), dtype=torch.int64) + 53 * iteration).remainder_(256).to(torch.uint8).view(shape)
            expected_source = full.narrow(dim, sum(old[:rank]), old[rank]).contiguous()
            count = expected_source.numel()
            backing = torch_mesh.empty(count + 34, dtype=torch.uint8).fill_(231)
            source = backing.narrow(0, 17, count).view(expected_source.shape)
            source.copy_(expected_source)
            guards.append((backing, count, 231))
            sources.append((source, expected_source))
            operand = partition.from_local(source, topology, [Shard(dim)], shape=shape,
                                           stride=torch.empty(shape, device='meta').stride())
            partition._attach(topology, 'bytes', new)
            with patch.object(dist, 'all_to_all_single', side_effect=AssertionError('byte migration used whole-operand staging')), \
                    patch.object(partition, '_empty', side_effect=guarded_empty):
                snapshots.append(partition.follow(operand)[0].to_local())
            references.append(full.narrow(dim, sum(new[:rank]), new[rank]).contiguous())
    torch.mps.synchronize()
    failure(group)
    for i, (actual, expected) in enumerate(zip(snapshots, references)):
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(('exact migration bytes', rank, i))
    for source, expected in sources:
        assert torch.equal(source.cpu(), expected), ('byte source overwritten', rank)
    for backing, count, poison in guards:
        actual = backing.cpu()
        assert torch.all(actual[:17] == poison) and torch.all(actual[count + 17:] == poison), ('byte neighbor poison', rank)
    return dict(case='byte-migration', rank=rank, calls=len(snapshots), exact=True, original_bits_preserved=True,
                neighboring_poison_preserved=True, whole_operand_staging_collectives=0, receipt_collectives=0,
                geometry=metadata, traffic=traffic(before, sample.read()))


def indexed_span_writes(group, sample, iterations):
    """Real Torch SpanWrites: original/final strided views and distinct fresh leases."""
    from torch_mesh import _stream
    from torch_mesh.backend import Call, WRITE

    rank, size, shape = group.rank(), group.size(), (3, 5, 19)
    count = math.prod(shape)
    pairs = [(source, (source + 1) % size) for source in range(size)]
    group.plan([[Call(WRITE, count, torch.uint8, root=source, destination=destination)
                 for source, destination in pairs]])
    prepared, originals, finals, relays = [], [], [], []
    for iteration in range(iterations):
        deliveries = []
        for source, destination in pairs:
            pattern = ((torch.arange(count, dtype=torch.int64) * 37 + source * 53 + iteration * 71)
                       .remainder_(256).to(torch.uint8).view(shape))
            if rank == source:
                storage = torch_mesh.empty(3, 7, 64, dtype=torch.uint8).fill_(231)
                view = storage[:, 1:6, 2:59:3]
                view.copy_(pattern)
                expected = torch.full(storage.shape, 231, dtype=torch.uint8)
                expected[:, 1:6, 2:59:3].copy_(pattern)
                originals.append((storage, expected))
            elif rank == destination:
                storage = torch_mesh.empty(5, 3, 49, dtype=torch.uint8)
                view = storage[..., 3:41:2].permute(1, 0, 2)
                # Only holes/borders are initialized. No outstanding producer
                # store can overwrite an immediately preposted payload byte.
                storage[..., :3].fill_(173)
                storage[..., 4:40:2].fill_(173)
                storage[..., 40:].fill_(173)
                expected = torch.full(storage.shape, 173, dtype=torch.uint8)
                expected[..., 3:41:2].permute(1, 0, 2).copy_(pattern)
                finals.append((storage, expected))
            else:
                storage = torch_mesh.empty(count + 23, dtype=torch.uint8)
                storage[:7].fill_(149)
                storage[7 + count:].fill_(149)
                view = storage.narrow(0, 7, count)
                relays.append(storage)
            deliveries.append((view, source, destination))
        # This public binding always constructs _stream.SpanWrites; it cannot
        # take affine_rows, padded gather, or a staging collective instead.
        prepared.append(group.prepare_writes(deliveries))
    _stream.publish_boundary()
    before = sample.read()
    for plan in prepared:
        plan.execute()
    torch.mps.synchronize()
    failure(group)
    for iteration, (storage, expected) in enumerate(finals):
        assert torch.equal(storage.cpu(), expected), ('indexed final bytes/holes/borders', rank, iteration)
    for iteration, (storage, expected) in enumerate(originals):
        assert torch.equal(storage.cpu(), expected), ('indexed original bytes/holes/borders', rank, iteration)
    for storage in relays:
        actual = storage.cpu()
        assert torch.all(actual[:7] == 149) and torch.all(actual[7 + count:] == 149), ('indexed relay guards', rank)
    wires = payload_traffic(group, sample, before, count * iterations, count * iterations)
    group.plan([])
    return dict(case='indexed-span-writes', rank=rank, calls=iterations, logical_writes=size * iterations,
                shape=shape, bytes_per_write=count, exact=True, original_bits_preserved=True,
                neighboring_poison_preserved=True, payload_padding_bytes=0, torch_span_writes=True,
                fresh_retained_generations=True, receipt_collectives=0, traffic=wires,
                two_rank_exact_payload_verified=size == 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=12)
    args = parser.parse_args()
    if args.iterations < 6 or args.iterations % 3:
        parser.error('--iterations must be a multiple of three and at least six')
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    group = dist.group.WORLD
    if group.size() not in (2, 3):
        raise ValueError('this cluster check is for two or three ranks')
    # Native communicator realization supplies every physical default route.
    # No test-only route installation may conceal a missing production default.
    sample = QueueSample(group.comm, group.rank())
    cases = []
    for dtype in (torch.float16, torch.float32):
        for strided, persistent, repeats in ((False, False, 1), (True, True, 3)):
            cases.append(gathered(group, sample, 'tiny', [r + 1 for r in range(group.size())], dtype,
                                  strided, persistent, repeats, args.iterations, False))
            cases.append(gathered(group, sample, 'all-cables', [256 * (r + 1) for r in range(group.size())], dtype,
                                  strided, persistent, repeats, args.iterations, True))
        zero = [0 if r == group.size() - 1 else 4 * (r + 1) for r in range(group.size())]
        cases.append(gathered(group, sample, 'zero-share', zero, dtype, True, True, 3, args.iterations, False))
        cases.append(summed(group, sample, dtype, args.iterations))
        for case in cases:
            if case['dtype'] == str(dtype):
                print(json.dumps(case), flush=True)
    moved = migrated(group, args.iterations // 3)
    cases.append(moved)
    print(json.dumps(moved), flush=True)
    exact = byte_migrations(group, sample, args.iterations // 3)
    cases.append(exact)
    print(json.dumps(exact), flush=True)
    indexed = indexed_span_writes(group, sample, args.iterations // 3)
    cases.append(indexed)
    print(json.dumps(indexed), flush=True)
    print(json.dumps(dict(rank=group.rank(), cases=len(cases), calls=sum(case['calls'] for case in cases),
                          passed=True, scope='Correctness and physical-cable usage; diagnostic output snapshots, no bandwidth claim')), flush=True)
    # Control-only teardown, after every tested function and numerical check.
    store = dist.distributed_c10d._get_default_store()
    store.set(f'physical-finished/{group.rank()}', b'1')
    for rank in range(group.size()):
        store.get(f'physical-finished/{rank}')
    store.set(f'physical-drained/{group.rank()}', b'1')
    if group.rank() == 0:
        for rank in range(group.size()):
            store.get(f'physical-drained/{rank}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
