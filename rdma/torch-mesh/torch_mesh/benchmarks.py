"""Shared collective measurements over the production mesh backend.

Useful output bytes and whole-function completion time describe the work.
Preparation occurs before submission; observations do not change execution.
"""
import json
from pathlib import Path
import sys

from mesh_statistics import now, Observations


def exchange_counts(rank, size, count):
    """A directed cyclic useful delivery; all physical ranks remain relays."""
    matrix = [count if destination == (source + 1) % size else 0
              for source in range(size) for destination in range(size)]
    return (matrix[rank * size:(rank + 1) * size],
            [matrix[source * size + rank] for source in range(size)], matrix)


def collective_operands(what, n, rank, size):
    """Declare source and final output storage before the whole call."""
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
        return (a, a, n * 2,
                Call(ALLREDUCE, n, a.dtype, op=dist.ReduceOp.MAX))
    if what == 'broadcast':
        a = tensor(n, rank + 1)
        return a, a, n * 2 if rank else 0, Call(BROADCAST, n, a.dtype, root=0)
    if what == 'all_gather':
        source, output = tensor(per, rank + 1), tensor(per * size, -1)
        return (source, output, 2 * per * (size - 1),
                Call(ALLGATHER, per * size, source.dtype, counts=[per] * size))
    if what == 'reduce_scatter_max':
        source, output = tensor(per * size, rank + 1), tensor(per, -1)
        return (source, output, 2 * per,
                Call(REDUCE_SCATTER, per * size, source.dtype, op=dist.ReduceOp.MAX, counts=[per] * size))
    if what == 'exchange':
        source, output = tensor(n, rank + 1), tensor(n, -1)
        sends, receives, matrix = exchange_counts(rank, size, n)
        return (source, output, n * 2 if size > 1 else 0,
                Call(ALLTOALL, n, source.dtype, counts=sends + receives, matrix=matrix))
    raise ValueError(f'unknown collective measurement: {what}')


def expected(what, rank, size, count):
    """Each case's output once its calls are done, from its operands (collective_operands): the max of every rank's
    rank + 1, the root's 1, the all-gather's segment r r + 1, the exchange's its predecessor's"""
    import torch
    if what in ('all_reduce_max', 'reduce_scatter_max'):
        return torch.full((count,), float(size))
    if what == 'broadcast':
        return torch.full((count,), 1.0)
    if what == 'all_gather':
        return torch.arange(size, dtype=torch.float32).repeat_interleave(count // size) + 1
    return torch.full((count,), float((rank - 1) % size + 1))


def checked(rank, outputs):
    """Every case's output against what its operands give, read once every call is done: the library's contract,
    outside every measured interval. A case whose output differs fails the run, naming it."""
    from .engines import host
    wrong = []
    for what, requested, output, want in outputs:
        got = host(output).float()
        bad = int((got != want).sum())
        if bad:
            first = int((got != want).nonzero()[0])
            wrong.append(f'{what} at {requested} bytes: {bad} of {got.numel()} elements, the first at {first} '
                         f'{float(got[first])} against {float(want[first])}')
    print(json.dumps({'rank': rank, 'checked': len(outputs), 'wrong': wrong}), flush=True)
    if wrong:
        raise RuntimeError(f'rank {rank}: ' + '; '.join(wrong))


def bandwidth(config):
    # design/whole-function.md#observations
    import torch
    import torch.distributed as dist
    from .backend import prepare_fragments
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    cases, outputs = [], []
    for requested in config['sizes']:
        n = requested // 2
        calls = min(config['most_calls'], max(config['least_calls'], config['bytes_per_size'] // requested))
        for what in ('all_reduce_max', 'broadcast', 'all_gather', 'reduce_scatter_max', 'exchange'):
            source, output, useful, declaration = collective_operands(what, n, rank, size)
            binding, declarations = prepare_fragments([(declaration, source, output)], group)
            cases.append((binding, useful, declarations, calls, what, requested))
            outputs.append((what, requested, output, expected(what, rank, size, output.numel())))
    group.plan([(declarations, calls) for _, _, declarations, calls, _, _ in cases])
    prepared = []
    for binding, useful, _, calls, what, requested in cases:
        binding.realize(calls)
        prepared.append((binding, useful, calls, what, requested))
    observations = Observations()
    for index, (binding, useful, calls, what, requested) in enumerate(prepared):
        # each case submitted once this rank's earlier ones have ended: its submission-to-completion time is then its
        # own, where submitted behind every case still queued it included all of theirs (d4916d1e's "Cumulative",
        # design/measurement-regressions-2026-10-30.md); a rank whose part of the case before ended early (a
        # broadcast's root) still starts its next case early
        observations.close()
        started = now()
        binding.enqueue()
        completion = binding.completion()
        observations.put(completion, {'episode': str(index), 'parts': {}, 'graphs': {}, 'proposal': 0,
            'workload': {'collective': what, 'bytes': requested, 'calls': calls}, 'implementation': 'loaded-program',
            'started': started, 'rank': rank, 'size': size, 'useful_work_bytes': useful * calls,
            'report': {'what': what, 'requested_bytes': requested, 'calls': calls,
                       'useful_output_bytes_per_call_rank': useful,
                       'compiled_microseconds_per_call': group._plan.seconds[index] * 1e6}})
    observations.close()
    checked(rank, outputs)
    dist.destroy_process_group()


def bandwidth_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bandwidth(json.load(sys.stdin) if argv[0] == '-' else json.loads(Path(argv[0]).read_text()))


if __name__ == '__main__':
    bandwidth_main()
