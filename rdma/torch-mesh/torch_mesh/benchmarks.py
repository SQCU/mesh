"""Shared collective measurements over the production mesh backend.

Useful output bytes describe the operation. Physical wire payload comes only from
native per-cable completed-byte counters; it includes relays, excludes framing and
is never inferred from world size. All initialization and validation is untimed.
"""
import json
from pathlib import Path
import sys
import time


def traffic_delta(before, after):
    key = lambda row: (row['cable'], row['peer'], row['wire'])
    old, new = {key(row): row for row in before}, {key(row): row for row in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError('native sample cable identity changed')
    result = []
    for identity, row in new.items():
        delta = {name: row[name] - old[identity][name] for name in ('sent', 'landed')}
        if min(delta.values()) < 0:
            raise ValueError('native completion byte counter moved backwards')
        result.append(dict(zip(('cable', 'peer', 'wire'), identity), **delta))
    return result


def realize_inputs(group):
    """Finish initial GPU fills before publishing that these bindings exist.

    This is configuration construction, once per new operand set. It is never
    called between repetitions of a realized function.
    """
    import torch
    import torch.distributed as dist
    torch.mps.synchronize()
    store = dist.distributed_c10d._get_default_store()
    serial = getattr(group, '_measurement_inputs', 0)
    group._measurement_inputs = serial + 1
    key = f'mesh/measurement/inputs/{group.group_name}/{serial}'
    store.set(f'{key}/{group.rank()}', b'1')
    for peer in range(group.size()):
        store.get(f'{key}/{peer}')


def sequence(call, window_calls, windows, sample=None):
    """Submit every window before draining once; no peer rendezvous or logging.

    Host duration covers the entire submission and drain, including dependency
    stalls. No timing markers split the command buffers. A window is an
    observation interval, not a maximum in-flight call count.
    """
    import torch
    if window_calls <= 0 or windows <= 0:
        raise ValueError('measurement windows and call counts must be positive')
    before = sample.read() if sample is not None else None
    began = time.perf_counter_ns()
    for _ in range(windows):
        for _ in range(window_calls):
            call()
    torch.mps.synchronize()
    host_ns = time.perf_counter_ns() - began
    after = sample.read() if sample is not None else None
    return {'host_ns': host_ns, 'window_calls': window_calls, 'windows': windows,
            'submitted_calls': window_calls * windows,
            'traffic': traffic_delta(before, after) if sample is not None else None}


def timed(call, calls, sample, group):
    result = sequence(call, calls, 1, sample)
    return result['host_ns'] * 1e-9, result['traffic']


def latency(call, calls, group):
    """Amortized host interval; this is not an isolated call latency."""
    return sequence(call, 1, calls)['host_ns'] * 1e-9 / calls


def exchange_counts(rank, size, count):
    """A directed cyclic useful delivery; all physical ranks remain relays."""
    matrix = [count if destination == (source + 1) % size else 0
              for source in range(size) for destination in range(size)]
    return (matrix[rank * size:(rank + 1) * size],
            [matrix[source * size + rank] for source in range(size)], matrix)


def bandwidth(config):
    import torch
    import torch.distributed as dist
    import torch_mesh
    from .backend import Call, ALLREDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL
    from routing import QueueSample
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
        sample = QueueSample(group.comm, rank)
        for requested in config['sizes']:
            if requested <= 0 or requested % 2:
                raise ValueError('float16 bandwidth operand bytes must be positive and even')
            n = requested // 2
            per = max(1, n // size)
            a, b = (torch_mesh.empty(n, dtype=torch.half) for _ in range(2))
            c, part = torch_mesh.empty(per * size, dtype=torch.half), torch_mesh.empty(per, dtype=torch.half)
            calls = min(config['most_calls'], max(config['least_calls'], config['bytes_per_size'] // requested))
            # Idempotent MAX prevents repeated in-place reductions overflowing half precision.
            sends, receives, matrix = exchange_counts(rank, size, n)
            figures = [('all_reduce_max', lambda: dist.all_reduce(a, op=dist.ReduceOp.MAX), a, n * 2,
                        Call(ALLREDUCE, n, a.dtype, op=dist.ReduceOp.MAX)),
                       ('broadcast', lambda: dist.broadcast(a, 0), a, n * 2 if rank else 0,
                        Call(BROADCAST, n, a.dtype, root=0)),
                       ('all_gather', lambda: dist.all_gather_into_tensor(c, part), c, 2 * per * (size - 1),
                        Call(ALLGATHER, per * size, a.dtype, counts=[per] * size)),
                       ('reduce_scatter_max', lambda: dist.reduce_scatter_tensor(part, c, op=dist.ReduceOp.MAX), part, 2 * per,
                        Call(REDUCE_SCATTER, per * size, a.dtype, op=dist.ReduceOp.MAX, counts=[per] * size)),
                       ('exchange', lambda: dist.all_to_all_single(b, a, receives, sends), b, n * 2 if size > 1 else 0,
                        Call(ALLTOALL, n, a.dtype, counts=sends + receives, matrix=matrix))]
            for what, call, output, useful, declaration in figures:
                group.plan([[declaration]])
                a.fill_(rank + 1)
                b.fill_(-1)
                c.fill_(rank + 1)
                part.fill_(rank + 1)
                realize_inputs(group)
                call()
                torch.mps.synchronize()
                seconds, traffic = timed(call, calls, sample, group)
                once = latency(call, config['latency_calls'], group)
                expected = (torch.arange(1, size + 1).repeat_interleave(per) if what == 'all_gather'
                            else torch.full((output.numel(),), size if what.endswith('_max')
                                            else 1 if what == 'broadcast' else (rank - 1) % size + 1))
                error = float((output.cpu().float() - expected.float()).abs().max())
                sent, landed = (sum(row[name] for row in traffic) for name in ('sent', 'landed'))
                print(json.dumps({'rank': rank, 'size': size, 'what': what, 'calls': calls,
                    'requested_bytes': requested, 'microseconds_per_call': seconds * 1e6 / calls,
                    'useful_output_bytes_per_call_rank': useful,
                    'useful_output_gigabytes_per_second_rank': useful * calls / seconds / 1e9,
                    'wire_payload_bytes_sent_rank': sent, 'wire_payload_bytes_landed_rank': landed,
                    'wire_payload_scope': 'exact completed-byte counter delta across the measurement window; framing unknown',
                    'wire_payload_call_attribution': 'unestablished: warmup/final SEND completions may cross sample boundaries',
                    'traffic': traffic, 'streaming_interval_microseconds': once * 1e6,
                    'timing_method': 'continuous submission; one local drain after all windows',
                    'max_abs_error': error}), flush=True)
            group.plan([])
    finally:
        dist.destroy_process_group()


def bandwidth_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bandwidth(json.loads(Path(argv[0]).read_text()))


if __name__ == '__main__':
    bandwidth_main()
