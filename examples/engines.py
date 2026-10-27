#!/usr/bin/env python3
"""A stack of projections and explicitly assigned CPU/GPU/ANE shares.

Shapes, engine options and shares are configuration. Every call uses the
prepared spans; observations arrive through the ordinary whole-function module.
"""
import os
import json

import torch
from torch_mesh import engines
from torch_mesh.compilation import parameters
import rollout
from mesh_statistics import now

env = lambda key, default: int(os.environ.get(key, default))
rows, k, n, steps, every, layers = env('ROWS', 1024), env('K', 3840), env('N', 3456), env('STEPS', 40), env('EVERY', 1), env('LAYERS', 4)
names, kind = os.environ.get('ENGINES', 'mps,cpu,ane'), os.environ.get('OP', 'ffn')
rates = json.loads(os.environ['ENGINE_CAPACITIES'])
capacities = [rates[name] for name in names.split(',')]
codes, first = os.environ.get('CODES'), env('FIRST', 0)
if codes:
    from torch_mesh import streamed
    export = streamed.Streamed(codes)
    coded = [[export[f'model.language_model.layers.{L}.mlp.{p}_proj'] for p in ('gate', 'up', 'down')]
             for L in range(first, first + layers)]
    n, k = coded[0][0].rows, coded[0][0].columns
    assert kind == 'ffn' and all(g.shape == (n, k) for g, _, _ in coded), 'CODES: FFN layers of one shape'
torch.manual_seed(0)
x = torch.randn(rows, k).half().to('mps') / 4
pool = engines.Engines(names.split(','), compilation=parameters(), ane={'tile': env('ANE_TILE', 128), 'block': env('ANE_BLOCK', 4096)})
projections, shared = [], []
for layer in range(layers):
    projections.append((torch.randn(k, k) / k ** 0.5).half().to('mps'))
    if kind == 'linear':
        w = (torch.randn(k, k) / k ** 0.5).half().to('mps')
        shared.append(engines.Linear(w, engines=pool, name=kind, grain=128))
    else:
        if codes:
            gate, up, down = coded[layer]
        else:
            gate, up = ((torch.randn(n, k) / k ** 0.5).half().to('mps') for _ in range(2))
            down = (torch.randn(k, n) / n ** 0.5).half().to('mps')
        shared.append(engines.FFN(gate, up, down, engines=pool, name=kind, grain=128))
key = shared[0].declare(rows, x.dtype, capacities=capacities)
for operation in shared[1:]:
    operation.declare(rows, x.dtype, capacities=capacities)
observations = []
observer = rollout.start(None, 'engines', 0, 1, observations)
from torch_mesh.declarations import Function, execution_sequence
for start in range(0, steps, every):
    assigned = engines.placement(pool, observations, step=start // every)
    calls = min(every, steps - start)
    spans, source = [], x
    for projection, operation in zip(projections, shared):
        normalized = torch.empty_like(x)
        def project(source=source, projection=projection, normalized=normalized):
            return normalized.copy_(torch.nn.functional.rms_norm(torch.nn.functional.linear(source, projection), (k,), eps=1e-6))
        project = Function(project, configuration=assigned['operands'])
        project.declare(None)
        work = operation.prepare(normalized, parts=assigned['parts'][key])
        spans.extend((project, work))
        source = work.output
    program = execution_sequence(spans * calls, source)
    program.realize()
    began = now()
    program.forward()
    completion = program.completion()
    observer[0].put((completion, {'episode': str(start), 'parts': assigned['parts'], 'proposal': assigned['proposal'], 'graphs': {},
        'workload': {'layers': layers, 'kind': kind, 'rows': rows, 'k': k, 'n': n},
        'implementation': 'loaded-program', 'started': began, 'rank': 0, 'size': 1,
        'useful_work_bytes': rows * k * calls * 2,
        'report': {'program': 'engines', 'steps': calls, 'rows': rows, 'parts': assigned['parts']}}))
rollout.close(observer)
