"""Matched workload measurements using production collective/function APIs.

This module contains measurement coordination and numerical workloads, no
transport implementation. Each window contains complete function calls. All
windows are submitted continuously, then drained once. Total host duration
includes submission and drain. Individual window latencies are not inferred.
"""
import json
import math
from pathlib import Path
import sys


def workload(spec, group):
    import torch
    import torch.distributed as dist
    import torch_mesh
    from . import partition
    from .backend import Call, ALLREDUCE, ALLGATHER, REDUCE_SCATTER, BROADCAST

    rank, size = group.rank(), group.size()
    dtype = getattr(torch, spec.get('dtype', 'float32'))
    width = torch.empty((), dtype=dtype).element_size()
    kind, box = spec['kind'], {}
    allocate = lambda *shape: torch_mesh.empty(*shape, dtype=dtype)
    if kind in ('allreduce', 'broadcast', 'allgather', 'reduce_scatter'):
        count = spec['elements']
        a = allocate(count).fill_(rank + 1)
        if kind == 'allreduce':
            calls = [[Call(ALLREDUCE, count, dtype, op=dist.ReduceOp.MAX)]]
            def run():
                dist.all_reduce(a, op=dist.ReduceOp.MAX)
                return a
            expected, useful = torch.full((count,), size, dtype=dtype), count * width
        elif kind == 'broadcast':
            calls = [[Call(BROADCAST, count, dtype, root=0)]]
            def run():
                dist.broadcast(a, 0)
                return a
            expected, useful = torch.ones(count, dtype=dtype), count * width if rank else 0
        else:
            counts = [count] * size
            b = allocate(count * size).fill_(rank + 1)
            if kind == 'allgather':
                calls = [[Call(ALLGATHER, count * size, dtype, counts=counts)]]
                def run():
                    dist.all_gather_into_tensor(b, a)
                    return b
                expected = torch.arange(1, size + 1, dtype=dtype).repeat_interleave(count)
                useful = count * width * (size - 1)
            else:
                calls = [[Call(REDUCE_SCATTER, count * size, dtype, counts=counts, op=dist.ReduceOp.MAX)]]
                def run():
                    dist.reduce_scatter_tensor(a, b, op=dist.ReduceOp.MAX)
                    return a
                expected, useful = torch.full((count,), size, dtype=dtype), count * width
    elif kind == 'fused_gather':
        sizes = spec['sizes'][:size]
        if len(sizes) != size:
            raise ValueError('gather configuration must name every rank')
        rows, columns = spec.get('rows', 2), spec.get('columns', 17)
        stride = spec.get('stride', 1)
        backing = allocate(rows, sizes[rank], columns * stride).fill_(rank + 1)
        a = backing[..., ::stride]
        calls = [partition.gathered(a, 1, sizes, group)]
        def run():
            return torch.ops.torch_mesh.allgatherv(a, 1, sizes, group.group_name)
        expected = torch.cat([torch.full((rows, n, columns), r + 1, dtype=dtype)
                              for r, n in enumerate(sizes)], 1)
        useful = rows * columns * (sum(sizes) - sizes[rank]) * width
    elif kind == 'mlp':
        rows, hidden, layers = spec['rows'], spec['hidden'], spec['layers']
        weights = spec.get('shares', [1] * size)[:size]
        edges = [round(hidden * sum(weights[:r]) / sum(weights)) for r in range(size + 1)]
        begin, end = edges[rank:rank + 2]
        generator = torch.Generator().manual_seed(7103)
        reference = torch.randn(rows, hidden, generator=generator) * .2
        a = allocate(rows, hidden).copy_(reference)
        model = []
        for _ in range(layers):
            first = torch.randn(hidden, hidden, generator=generator) / math.sqrt(hidden)
            second = torch.randn(hidden, hidden, generator=generator) / math.sqrt(hidden)
            model.append((allocate(hidden, end - begin).copy_(first[:, begin:end]),
                          allocate(end - begin, hidden).copy_(second[begin:end])))
            reference = torch.tanh(torch.tanh(reference @ first) @ second)
        fused = spec.get('fused', False)
        prototype = allocate(rows, hidden)
        calls = [partition.summed(prototype, group) if fused else
                 [Call(ALLREDUCE, rows * hidden, dtype)] for _ in range(layers)]
        def run():
            value = a
            for first, second in model:
                local = torch.tanh(value @ first)
                partial = allocate(rows, hidden)
                torch.mm(local, second, out=partial)
                if fused:
                    combined = partition._fused(partial, 0, [rows] * size, group, True)
                else:
                    dist.all_reduce(partial)
                    combined = partial
                value = torch.tanh(combined)
            return value
        expected, useful = reference.to(dtype), layers * rows * hidden * width
    else:
        raise ValueError(f'unknown numerical workload {kind}')
    group.plan(calls)
    def invoke():
        box['output'] = run()
    def verify():
        actual = box['output'].cpu()
        error = float((actual.float() - expected.float()).abs().max()) if actual.numel() else 0.0
        torch.testing.assert_close(actual, expected, rtol=.02 if kind == 'mlp' else 0,
                                   atol=.003 if kind == 'mlp' else 0)
        return error
    return invoke, verify, useful


def measure(config):
    if config.get('optimization'):
        raise ValueError('serialized benchmark optimization was removed; supply prepared controller states in configurations')
    import torch
    import torch.distributed as dist
    from routing import Fabric, QueueSample
    from .benchmarks import realize_inputs, sequence
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    store = dist.distributed_c10d._get_default_store()
    rank, size = group.rank(), group.size()
    fabric, sample = Fabric.of(group.comm), QueueSample(group.comm, rank)
    serial = 0
    def realize():
        nonlocal serial
        realize_inputs(group)
        serial += 1
    def check(verify, spec, phase):
        """Record invalid outputs without stranding peers at the next boundary."""
        nonlocal serial
        try:
            result = {'error': verify(), 'valid': True}
        except AssertionError as exc:
            result = {'valid': False, 'detail': str(exc)}
        key = f'pareto/verification/{serial}'
        serial += 1
        store.set(f'{key}/{rank}', json.dumps(result))
        peers = [json.loads(store.get(f'{key}/{peer}')) for peer in range(size)]
        valid = all(p['valid'] for p in peers)
        print(json.dumps({'record': 'verification', 'rank': rank, 'case': spec['name'],
                          'phase': phase, 'valid_all_ranks': valid, 'local': result}), flush=True)
        return valid, result.get('error')
    print(json.dumps({'record': 'fabric', 'rank': rank, 'nodes': size, 'cables': fabric.cables,
                      'configuration': config}), flush=True)
    try:
        with torch.inference_mode():
            for spec in config['cases']:
                invoke, verify, useful = workload(spec, group)
                # Setup fills must finish before any rank writes their landing
                # storage. This boundary is outside every timed function.
                realize()
                for _ in range(config.get('warmup', 3)):
                    invoke()
                torch.mps.synchronize()
                valid, error = check(verify, spec, 'warmup')
                if not valid:
                    group.plan([])
                    continue
                for depth in spec.get('depths', config['depths']):
                    state = config.get('configurations', {}).get(f'{spec["name"]}/{depth}')
                    if state is not None:
                        from controller import WholeFunctionController
                        controller = WholeFunctionController.of(state)
                        group.reset_fused_configuration()
                        controller.install_routes(group.comm)
                        group.plan(group._declaration)
                        realize()
                    measured = sequence(invoke, depth, config['trials'], sample)
                    print(json.dumps({'record': 'sequence', 'rank': rank, 'case': spec['name'],
                        **measured, 'useful_bytes_per_call_rank': useful,
                        'timing_method': 'continuous-sequence-host-completion',
                        'wire_attribution': 'whole sequence counter delta; boundary completions may cross sequences'}), flush=True)
                torch.mps.synchronize()
                valid, error = check(verify, spec, 'final')
                if valid:
                    print(json.dumps({'record': 'verified', 'rank': rank, 'case': spec['name'],
                                      'max_abs_error': error}), flush=True)
                group.plan([])
            # Keep the TCPStore server alive until every non-root has finished
            # reading the final measurement boundary.
            store.set(f'pareto/exit/{rank}', b'1')
            if rank == 0:
                for peer in range(1, size):
                    store.get(f'pareto/exit/{peer}')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    measure(json.loads(Path(sys.argv[1]).read_text()))
