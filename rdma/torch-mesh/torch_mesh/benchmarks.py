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


def completed(record, seconds):
    # design/whole-function.md#observations
    result = {**record['report'], 'rank': record['rank'], 'size': record['size'], 'seconds': seconds,
              'useful_output_bytes': record['useful_work_bytes'],
              'useful_gigabytes_per_second': record['useful_work_bytes'] / seconds / 1e9,
              'timing_method': 'host-clock-poll-of-ordinary-completion', 'function_feedback': record}
    if 'steps' in result:
        result.update(step_microseconds=seconds / result['steps'] * 1e6,
                      calls_per_second=result['steps'] / seconds)
        if 'rows' in result:
            result['aggregate_rows_per_second'] = result['steps'] * result['rows'] / seconds
    if 'calls' in result:
        result['microseconds_per_call'] = seconds * 1e6 / result['calls']
        result['useful_output_gigabytes_per_second_rank'] = record['useful_work_bytes'] / seconds / 1e9
    return result


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
    def tensor(count, value):
        from .engines import host
        result = empty(count, dtype=torch.half)
        host(result).fill_(value)
        return result
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
    # design/whole-function.md#observations
    import torch
    import torch.distributed as dist
    import rollout
    from . import _stream
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    cases = []
    for requested in config['sizes']:
        n = requested // 2
        calls = min(config['most_calls'], max(config['least_calls'], config['bytes_per_size'] // requested))
        for what in ('all_reduce_max', 'broadcast', 'all_gather', 'reduce_scatter_max', 'exchange'):
            call, output, useful, declaration = collective_operands(what, n, rank, size)
            cases.append((call, output, useful, declaration, calls, what, requested))
    group.plan([([declaration], calls) for _, _, _, declaration, calls, _, _ in cases])
    observer = rollout.start(dist.distributed_c10d._get_default_store(), 'mesh/bandwidth', rank, size,
                             Path('~/.cache/mesh/observations/bandwidth.jsonl').expanduser())
    group._observers = [observer]
    for index, (call, output, useful, declaration, calls, what, requested) in enumerate(cases):
        started = time.monotonic()
        for _ in range(calls):
            call()
        completion = _stream.completion()
        _stream.publish_boundary()
        observer[0].put((completion, {'episode': str(index), 'parts': {}, 'graphs': {}, 'costs': {},
            'workload': {'collective': what, 'bytes': requested, 'calls': calls}, 'implementation': 'loaded-program',
            'started': started, 'rank': rank, 'size': size, 'useful_work_bytes': useful * calls,
            'report': {'what': what, 'requested_bytes': requested, 'calls': calls,
                       'useful_output_bytes_per_call_rank': useful}}))
    dist.destroy_process_group()


def bandwidth_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bandwidth(json.loads(Path(argv[0]).read_text()))


if __name__ == '__main__':
    bandwidth_main()
