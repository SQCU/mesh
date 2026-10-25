"""Shared collective measurements over the production mesh backend.

Useful output bytes describe the operation. Physical wire payload comes only from
native per-cable completed-byte counters; it includes relays, excludes framing and
is never inferred from world size. All initialization and validation is untimed.
"""
import json
from pathlib import Path
import sys
import time

TIMING_METHOD = 'continuous-host-submission-and-final-completion'


class HostTimings:
    def __init__(self):
        self.times = []

    def mark(self):
        self.times.append(time.perf_counter())

    def ready(self):
        return len(self.times) >= 2

    def read(self):
        return self.times


def function_timings(device):
    if device == 'mps':
        from . import _stream
        return _stream.CompletionTimings()
    return HostTimings()


def emit(record, *, file=None):
    """One structured record format for examples and the hourly reader."""
    print(json.dumps(record, sort_keys=True), file=file, flush=True)


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
    submitted_ns = time.perf_counter_ns() - began
    torch.mps.synchronize()
    host_ns = time.perf_counter_ns() - began
    after = sample.read() if sample is not None else None
    return {'host_ns': host_ns, 'submitted_ns': submitted_ns, 'timing_method': TIMING_METHOD,
            'window_calls': window_calls, 'windows': windows,
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


def collective_operands(what, n, rank, size):
    """Realize fresh bindings; native lease retirement protects earlier DMA.

    Reinitializing the previous operation's source could overwrite unfinished
    SENDs. Distinct generations let ranks advance without a peer rendezvous.
    Each returned callable retains its inputs for its entire submitted sequence.
    """
    import torch
    import torch.distributed as dist
    from . import empty
    from .backend import Call, ALLREDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL
    per = max(1, n // size)
    tensor = lambda count, value: empty(count, dtype=torch.half).fill_(value)
    if what == 'all_reduce_max':
        a = tensor(n, rank + 1)
        # MAX is idempotent under repeated in-place application.
        return (lambda: dist.all_reduce(a, op=dist.ReduceOp.MAX), a, n * 2,
                Call(ALLREDUCE, n, a.dtype, op=dist.ReduceOp.MAX))
    if what == 'broadcast':
        a = tensor(n, rank + 1)
        return lambda: dist.broadcast(a, 0), a, n * 2 if rank else 0, Call(BROADCAST, n, a.dtype, root=0)
    if what == 'all_gather':
        source, output = tensor(per, rank + 1), tensor(per * size, -1)
        return (lambda: dist.all_gather_into_tensor(output, source), output, 2 * per * (size - 1),
                Call(ALLGATHER, per * size, source.dtype, counts=[per] * size))
    if what == 'reduce_scatter_max':
        source, output = tensor(per * size, rank + 1), tensor(per, -1)
        return (lambda: dist.reduce_scatter_tensor(output, source, op=dist.ReduceOp.MAX), output, 2 * per,
                Call(REDUCE_SCATTER, per * size, source.dtype, op=dist.ReduceOp.MAX, counts=[per] * size))
    if what == 'exchange':
        source, output = tensor(n, rank + 1), tensor(n, -1)
        sends, receives, matrix = exchange_counts(rank, size, n)
        return (lambda: dist.all_to_all_single(output, source, receives, sends), output, n * 2 if size > 1 else 0,
                Call(ALLTOALL, n, source.dtype, counts=sends + receives, matrix=matrix))
    raise ValueError(f'unknown collective measurement: {what}')


def bandwidth(config):
    import torch
    import torch.distributed as dist
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
            calls = min(config['most_calls'], max(config['least_calls'], config['bytes_per_size'] // requested))
            for what in ('all_reduce_max', 'broadcast', 'all_gather', 'reduce_scatter_max', 'exchange'):
                call, output, useful, declaration = collective_operands(what, n, rank, size)
                group.plan([[declaration]])
                # Finish local input construction before the observation.
                # There is no message aligning this rank with its peers.
                torch.mps.synchronize()
                call()
                torch.mps.synchronize()
                seconds, traffic = timed(call, calls, sample, group)
                once = latency(call, config['latency_calls'], group)
                expected = (torch.arange(1, size + 1).repeat_interleave(per) if what == 'all_gather'
                            else torch.full((output.numel(),), size if what.endswith('_max')
                                            else 1 if what == 'broadcast' else (rank - 1) % size + 1))
                error = float((output.cpu().float() - expected.float()).abs().max())
                sent, landed = (sum(row[name] for row in traffic) for name in ('sent', 'landed'))
                emit({'rank': rank, 'size': size, 'what': what, 'calls': calls,
                    'requested_bytes': requested, 'microseconds_per_call': seconds * 1e6 / calls,
                    'useful_output_bytes_per_call_rank': useful,
                    'useful_output_gigabytes_per_second_rank': useful * calls / seconds / 1e9,
                    'wire_payload_bytes_sent_rank': sent, 'wire_payload_bytes_landed_rank': landed,
                    'wire_payload_scope': 'exact completed-byte counter delta across the measurement window; framing unknown',
                    'wire_payload_call_attribution': 'unestablished: warmup/final SEND completions may cross sample boundaries',
                    'traffic': traffic, 'streaming_interval_microseconds': once * 1e6,
                    'timing_method': TIMING_METHOD,
                    'max_abs_error': error})
            group.plan([])
    finally:
        dist.destroy_process_group()


def bandwidth_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bandwidth(json.loads(Path(argv[0]).read_text()))


if __name__ == '__main__':
    bandwidth_main()
