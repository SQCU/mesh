"""Shared collective measurements over the production mesh backend.

Useful output bytes and whole-function completion time describe the work.
Preparation occurs before submission; observations do not change execution.
"""
import json
import time
from pathlib import Path
import sys

from mesh_statistics import now


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


def expected(what, rank, size, per):
    """A collective's output on `rank` by its definition, over collective_operands' values (rank r holds r + 1)."""
    import torch
    if what in ('all_reduce_max', 'reduce_scatter_max'):
        return torch.full((1,), float(size))
    if what == 'broadcast':
        return torch.full((1,), 1.0)
    if what == 'all_gather':
        return torch.arange(1, size + 1, dtype=torch.float32).repeat_interleave(per)
    return torch.full((1,), float((rank - 1) % size + 1))


def bandwidth(config):
    # design/whole-function.md#observations
    import torch
    import torch.distributed as dist
    import rollout
    from . import _stream
    from .backend import prepare_fragments
    from .engines import host
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    # design/replay-templates-2026-10-29.md: {'directory', 'phase'}: 'save' prepares the program and saves it as one
    # template; 'load' makes it from that template on new operands, with no lowering or planning
    template = config.get('template')
    loading = template is not None and template['phase'] == 'load'
    started, cases, bound = now(), [], []
    for requested in config['sizes']:
        n = requested // 2
        calls = min(config['most_calls'], max(config['least_calls'], config['bytes_per_size'] // requested))
        for what in ('all_reduce_max', 'broadcast', 'all_gather', 'reduce_scatter_max', 'exchange'):
            source, output, useful, declaration = collective_operands(what, n, rank, size)
            bound += [source, output]
            binding, declarations = (None, None) if loading else prepare_fragments([(declaration, source, output)], group)
            cases.append((binding, useful, declarations, calls, what, requested, output, max(1, n // size)))
    if loading:
        functions = _stream.load(template['directory'], bound, communicators=[group.handle])
        for function in functions:
            function.finish()
    else:
        group.plan([(declarations, calls) for _, _, declarations, calls, _, _, _, _ in cases])
        functions = []
        for binding, _, _, calls, _, _, _, _ in cases:
            function = _stream.Schedule()
            binding.append(function, calls)
            function.finish()
            functions.append(function)
    preparation = now() - started
    if template is not None and not loading:
        _stream.save(template['directory'], functions, bound)
    implementation = 'loaded-program' if template is None else f"template-{template['phase']}"
    observer = rollout.start(dist.distributed_c10d._get_default_store(), 'mesh/bandwidth', rank, size,
                             'bandwidth')
    group._observers = [observer]
    completions = []
    for index, (function, (_, useful, _, calls, what, requested, _, _)) in enumerate(zip(functions, cases)):
        started = now()
        function.enqueue()
        completion = function.completion()
        completions.append(completion)
        observer[0].put((completion, {'episode': str(index), 'parts': {}, 'graphs': {}, 'proposal': 0,
            'workload': {'collective': what, 'bytes': requested, 'calls': calls}, 'implementation': implementation,
            'started': started, 'rank': rank, 'size': size, 'useful_work_bytes': useful * calls,
            'report': {'what': what, 'requested_bytes': requested, 'calls': calls,
                       'useful_output_bytes_per_call_rank': useful, 'preparation_seconds': preparation}}))
    if template is not None:
        # the outputs as each collective defines them, on both sides of the template
        while not _stream.completed(tuple(completions)):
            time.sleep(0.001)
        wrong = []
        for _, _, _, _, what, requested, output, per in cases:
            values, want = host(output).float().flatten(), expected(what, rank, size, per)
            if not torch.equal(values, want if what == 'all_gather' else want.expand_as(values)):
                wrong.append(f'{what} at {requested} bytes')
        print(json.dumps({'template': template['phase'], 'rank': rank, 'functions': len(functions),
                          'preparation_seconds': preparation, 'outputs_as_defined': not wrong, 'wrong': wrong}), flush=True)
        if wrong:
            raise RuntimeError(f"template {template['phase']}: outputs not as their collectives define them: {wrong}")
    dist.destroy_process_group()


def bandwidth_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bandwidth(json.load(sys.stdin) if argv[0] == '-' else json.loads(Path(argv[0]).read_text()))


if __name__ == '__main__':
    bandwidth_main()
