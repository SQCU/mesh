"""Live exact-byte DMA correctness and complete three-byte function timing.

Run on two/three ranks with the normal mesh environment. Preparation, shader
compilation, allocation, initialization, and diagnostics are outside GPU timing.
The returned payload is a numerical result, never a receipt acknowledgment.
"""
import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist
import torch_mesh
from torch_mesh import _stream, fused

import mesh
from routing import QueueSample

EXTENT = 1 << 30
POISON = 173
SOURCE = r'''
struct Words {
    uint wire_bytes, row_bytes, rows, stride, rounds, seed, unused0, unused1;
    mesh_step_binding land[CHANNELS], publish[CHANNELS];
};
inline void arrive(constant Words& w, uint tid, uint j=0) {
    if (tid == 0) for (uint c=0; c<CHANNELS; c++) if (w.land[c].p) {
        mesh_step_enter(*w.land[c].p, w.land[c].v, w.land[c].progress, true);
        mesh_step_counts(*w.land[c].p, w.land[c].a, w.land[c].v, j);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
}
inline void release(constant Words& w, uint tid, uint j=0) {
    for (uint c=0; c<CHANNELS; c++) if (w.publish[c].p) {
        mesh_step_enter(*w.publish[c].p, w.publish[c].v, w.publish[c].progress, tid == 0);
        mesh_step_release(*w.publish[c].p, w.publish[c].a, tid, j);
    }
}
kernel void release_existing(constant Words& w [[buffer(3)]], uint tid [[thread_index_in_threadgroup]]) {
    release(w, tid);
}
kernel void await_payload(constant Words& w [[buffer(3)]], uint tid [[thread_index_in_threadgroup]]) {
    arrive(w, tid);
}
// Complete function: x_j -> remote (3*x_j+7) -> local xor 0x5a.
// Each call uses a fresh slot. The useful return value determines its output.
kernel void function_root(device uchar* forward [[buffer(0)]], device const uchar* backward [[buffer(1)]],
                          device uchar* output [[buffer(2)]], constant Words& w [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]]) {
    for (uint j=0; j<w.rounds; j++) {
        if (tid<3) mesh_step_write<uchar>(forward, ulong(j)*w.stride+tid, uchar((w.seed+7*j+tid)%251));
        release(w, tid, j);
        arrive(w, tid, j);
        if (tid<3) output[ulong(j)*3+tid] = mesh_step_read<uchar>(backward, ulong(j)*w.stride+tid) ^ uchar(0x5a);
        threadgroup_barrier(mem_flags::mem_device);
    }
}
kernel void function_worker(device const uchar* forward [[buffer(0)]], device uchar* backward [[buffer(1)]],
                            constant Words& w [[buffer(3)]], uint tid [[thread_index_in_threadgroup]]) {
    for (uint j=0; j<w.rounds; j++) {
        arrive(w, tid, j);
        if (tid<3) {
            const uchar x = mesh_step_read<uchar>(forward, ulong(j)*w.stride+tid);
            mesh_step_write<uchar>(backward, ulong(j)*w.stride+tid, uchar(3*uint(x)+7));
        }
        release(w, tid, j);
    }
}
'''


def exchange(group, key, value):
    """Configuration/diagnostic exchange, never inside a tested data function."""
    store = dist.distributed_c10d._get_default_store()
    store.set(f'byte-writes/{key}/{group.rank()}', json.dumps(value))
    return [json.loads(store.get(f'byte-writes/{key}/{r}').decode()) for r in range(group.size())]


def async_error(group):
    status = mesh.ffi.new('ncclResult_t *')
    mesh.check(mesh.lib.ncclCommGetAsyncError(group.comm, status))
    if status[0]:
        raise RuntimeError(mesh.ffi.string(mesh.lib.ncclGetLastError(group.comm)).decode())


def dma_retired(buffers):
    """Local bridge-owned DMA completion, after timing, for exact diagnostics."""
    ready, deadline = mesh.ffi.new('int *'), time.monotonic() + 5
    for buffer in buffers:
        while True:
            mesh.check(mesh.lib.ncclMeshMemReusable(mesh.ffi.cast('void *', torch_mesh.address(buffer)), ready))
            if ready[0]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('local source DMA retirement did not complete')
            time.sleep(0.001)


def traffic(before, after):
    return [dict(cable=b['cable'], wire=b['wire'], peer=b['peer'], sent_bytes=a['sent']-b['sent'],
                 landed_bytes=a['landed']-b['landed']) for b, a in zip(before, after)]


def guarded(row_bytes, rows, stride, offset):
    extent = (rows-1)*stride + row_bytes
    backing = torch_mesh.empty(offset+extent+31, dtype=torch.uint8).fill_(POISON)
    return backing, backing.narrow(0, offset, extent)


def verify_guards(backing, offset, row_bytes, rows, stride):
    assert torch.all(backing[:offset].cpu() == POISON), 'leading neighbor overwritten'
    assert torch.all(backing[offset+(rows-1)*stride+row_bytes:].cpu() == POISON), 'trailing neighbor overwritten'
    if rows > 1 and stride > row_bytes:
        gaps = backing.as_strided((rows-1, stride-row_bytes), (stride, 1), offset+row_bytes)
        assert torch.all(gaps.cpu() == POISON), 'row gap overwritten'


def accounting(reports, source, destination, expected, require_all, fabric):
    sent = [sum(t['sent_bytes'] for t in r['traffic']) for r in reports]
    landed = [sum(t['landed_bytes'] for t in r['traffic']) for r in reports]
    assert sent[source] == expected, ('source payload bytes', sent, expected)
    assert landed[destination] == expected, ('destination payload bytes', landed, expected)
    assert sum(sent) == sum(landed), ('wire conservation', sent, landed)
    active = sorted({t['cable'] for r in reports for t in r['traffic'] if t['sent_bytes']})
    if require_all:
        assert active == list(range(len(fabric.cables))), ('not every physical cable carried data', active)
    return dict(source_payload_bytes=sent[source], destination_payload_bytes=landed[destination],
                physical_hop_bytes=sum(sent), relay_bytes=sum(sent)-expected, active_cables=active)


def one_way(group, kernels, sample, name, row_bytes, rows=1, all_links=False, mr=False):
    rank, channels = group.rank(), len(group.fabric.adjacent[group.rank()])
    stride, offset = row_bytes+7+2*rank, 17+12*rank
    backing = view = None
    error = None
    try:
        backing, view = guarded(row_bytes, rows, stride, offset)
    except RuntimeError as exc:
        error = str(exc)
    allocation = exchange(group, name+'/allocation', dict(error=error))
    if any(r['error'] for r in allocation):
        if mr:
            return dict(case=name, skipped=True, reason='contiguous registered lease unavailable', allocation=allocation)
        raise RuntimeError(allocation)
    reference = None
    if mr:
        if rank == 0:
            view.fill_(91)
    else:
        reference = (torch.arange(row_bytes*rows, dtype=torch.int64)*3+19).remainder_(251).to(torch.uint8).view(rows, row_bytes)
        if rank == 0:
            view.as_strided((rows, row_bytes), (stride, 1)).copy_(reference)
    # Initialization finishes BEFORE immediate receive preposting; no guard
    # fill can later overwrite an incoming valid byte.
    torch.mps.synchronize()
    before = sample.read()
    crossing = fused.write_rows(view[:row_bytes], [view[:row_bytes]], [(0, 1)], group,
                                row_bytes=row_bytes, rows=rows, stride=stride)
    words = [row_bytes*rows, row_bytes, rows, stride, 1, 0, 0, 0]
    if rank == 0:
        fused.encode_many([(kernels['release_existing'], [], words, 3, 1, 1, 256, channels, None, crossing)])
    elif rank == 1:
        fused.encode_many([(kernels['await_payload'], [], words, 3, 1, 1, 256, channels, crossing, None)])
    torch.mps.synchronize()
    async_error(group)
    dma_retired([view])
    if rank == 1:
        if mr:
            for at in range(0, row_bytes, 1 << 20):
                assert torch.all(view[at:min(at+(1 << 20), row_bytes)].cpu() == 91), ('MR payload', at)
        else:
            assert torch.equal(view.as_strided((rows, row_bytes), (stride, 1)).cpu(), reference), name
    verify_guards(backing, offset, row_bytes, rows, stride)
    report = dict(rank=rank, traffic=traffic(before, sample.read()))
    reports = exchange(group, name+'/complete', report)
    counted = accounting(reports, 0, 1, row_bytes*rows, all_links, group.fabric)
    return dict(case=name, passed=True, row_bytes=row_bytes, rows=rows, unaligned_offsets=True,
                payload_padding_bytes=0, metadata_prepared_before_execution=True, neighboring_bytes_preserved=True,
                mr_boundary_crossed=mr, **counted, ranks=reports)


def min_cut_bytes_per_second(fabric, source, destination):
    cuts = []
    for mask in range(1 << fabric.nodes):
        if not (mask & (1 << source)) or mask & (1 << destination):
            continue
        cuts.append(sum(c['bits_per_second']/8 for c in fabric.cables
                        if bool(mask & (1 << c['nodes'][0])) != bool(mask & (1 << c['nodes'][1]))))
    return min(cuts)


def function_batch(group, kernels, sample, mode, rounds, trial):
    rank, channels = group.rank(), len(group.fabric.adjacent[group.rank()])
    wire = 3 if mode == 'exact' else group.page
    stride, offset = (wire+17+2*rank, 17+12*rank) if mode == 'exact' else (wire, wire)
    forward_store, forward = guarded(wire, rounds, stride, offset)
    backward_store, backward = guarded(wire, rounds, stride, offset)
    output = torch.empty(rounds*3, dtype=torch.uint8, device='mps') if rank == 0 else None
    # Padding is initialized AOT for the control case. Both timed functions
    # compute the same three useful bytes; the baseline sends a full page.
    torch.mps.synchronize()
    before = sample.read()
    outward = fused.writes(forward[:wire], [forward[:wire]], [(0, 1)], group, repeats=rounds, stride=stride)
    returned = fused.writes(backward[:wire], [backward[:wire]], [(1, 0)], group, repeats=rounds, stride=stride)
    exchange(group, f'function/{mode}/{trial}/prepared', True)
    words = [wire, 3, 1, stride, rounds, 19+trial*13, 0, 0]
    timer = _stream.GPUTimings() if rank == 0 else None
    if rank == 0:
        timer.mark()
        fused.encode_many([(kernels['function_root'], [forward, backward, output], words, 3, 1, 1, 256,
                            channels, returned, outward)])
        timer.mark()
    elif rank == 1:
        fused.encode_many([(kernels['function_worker'], [forward, backward], words, 3, 1, 1, 256,
                            channels, outward, returned)])
    torch.mps.synchronize()
    async_error(group)
    elapsed = None
    if rank == 0:
        start, end = timer.read()
        elapsed = end-start
        expected = [((3*((words[5]+7*j+i)%251)+7) & 255) ^ 0x5a for j in range(rounds) for i in range(3)]
        assert torch.equal(output.cpu(), torch.tensor(expected, dtype=torch.uint8)), ('complete function', mode, trial)
    dma_retired([forward, backward])
    verify_guards(forward_store, offset, wire, rounds, stride)
    verify_guards(backward_store, offset, wire, rounds, stride)
    reports = exchange(group, f'function/{mode}/{trial}/complete',
                       dict(rank=rank, traffic=traffic(before, sample.read()), gpu_seconds=elapsed))
    sent = [sum(t['sent_bytes'] for t in r['traffic']) for r in reports]
    landed = [sum(t['landed_bytes'] for t in r['traffic']) for r in reports]
    assert sent[0] == sent[1] == landed[0] == landed[1] == wire*rounds, (sent, landed, wire, rounds)
    assert sum(sent) == sum(landed), (sent, landed)
    duration = reports[0]['gpu_seconds']
    capacity = min_cut_bytes_per_second(group.fabric, 0, 1)
    return dict(mode=mode, trial=trial, functions=rounds, useful_output_bytes_per_function=3,
                logical_transaction_bytes_per_function=6, wire_payload_bytes_per_direction=wire,
                physical_hop_bytes=sum(sent), payload_padding_bytes_per_function=2*(wire-3),
                gpu_batch_seconds=duration, amortized_function_seconds=duration/rounds,
                useful_bytes_per_second=3*rounds/duration,
                all_link_cut_capacity_bytes_per_second=capacity,
                useful_function_capacity_lower_bound_seconds=6/capacity if capacity > 0 else None,
                configured_wire_capacity_lower_bound_seconds=2*wire/capacity if capacity > 0 else None,
                useful_bandwidth_fraction_of_capacity_bound=6*rounds/(duration*capacity) if capacity > 0 else None,
                pure_nic_latency_seconds=None,
                timing_scope='Complete function GPU timeline; includes fixed marker/launch gaps and actual remote computation/transport',
                ranks=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=128)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--large-bytes', type=int, default=4*1024*1024+13)
    parser.add_argument('--mr-crossing', action='store_true', help='Try a >1GiB registered lease per rank; skip collectively if unavailable')
    args = parser.parse_args()
    if not 1 <= args.rounds < (1 << 32) or args.trials < 1 or not 1 <= args.large_bytes < (1 << 30):
        parser.error('rounds must be a positive uint32, trials positive, and large-bytes between 1 and 1GiB-1')
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    group = dist.group.WORLD
    if group.size() not in (2, 3):
        raise ValueError('this live check requires two or three ranks')
    channels = len(group.fabric.adjacent[group.rank()])
    code = fused.steps() + f'\n#define CHANNELS {channels}\n' + SOURCE
    kernels = {name: _stream.pipeline(code, name)
               for name in ('release_existing', 'await_payload', 'function_root', 'function_worker')}
    sample = QueueSample(group.comm, group.rank())
    checks = []
    for name, width, rows, all_links in (('three-bytes', 3, 1, False), ('odd-rows', 257, 7, False),
                                       ('many-rows', 3, 5003, False), ('all-cables', args.large_bytes, 1, True)):
        result = one_way(group, kernels, sample, name, width, rows, all_links)
        checks.append(result)
        if group.rank() == 0:
            print(json.dumps(result), flush=True)
    mr = (one_way(group, kernels, sample, 'registration-crossing', EXTENT+19, mr=True)
          if args.mr_crossing else dict(case='registration-crossing', skipped=True,
                                       reason='not requested; requires >1GiB contiguous registered lease per rank'))
    checks.append(mr)
    if group.rank() == 0:
        print(json.dumps(mr), flush=True)
    measured = []
    # Warm-up uses fresh slots and the same full function; it is not retained as
    # a timing result. Alternate the tested order to reduce thermal/order bias.
    for mode in ('exact', 'padded'):
        function_batch(group, kernels, sample, mode, min(args.rounds, 8), -1)
    for trial in range(args.trials):
        for mode in (('exact', 'padded') if trial % 2 == 0 else ('padded', 'exact')):
            result = function_batch(group, kernels, sample, mode, args.rounds, trial)
            measured.append(result)
            if group.rank() == 0:
                print(json.dumps(result), flush=True)
    if group.rank() == 0:
        summary = {}
        for mode in ('exact', 'padded'):
            durations = [x['amortized_function_seconds'] for x in measured if x['mode'] == mode]
            summary[mode] = dict(best_function_seconds=min(durations), median_function_seconds=statistics.median(durations))
        print(json.dumps(dict(passed=True, useful_function='seed -> remote affine uint8 -> local xor',
                              raw_nic_latency_seconds=None, results=summary,
                              page_to_exact_payload_ratio=group.page/3,
                              observed_median_function_speedup=summary['padded']['median_function_seconds']/summary['exact']['median_function_seconds'],
                              qualification='Whole-function observations; no claim that byte ratio predicts latency improvement')), flush=True)
    exchange(group, 'finished', True)
    # Ensure rank0's TCPStore survives until other ranks finish reading results.
    store = dist.distributed_c10d._get_default_store()
    store.set(f'byte-writes/drained/{group.rank()}', b'1')
    if group.rank() == 0:
        for rank in range(group.size()):
            store.get(f'byte-writes/drained/{rank}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
