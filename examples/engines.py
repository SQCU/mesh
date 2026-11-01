#!/usr/bin/env python3
"""A stack of projections and explicitly assigned CPU/GPU/ANE shares.

Shapes, engine options and shares are configuration. Every call uses the
prepared spans; observations arrive through the ordinary whole-function module.
"""
import os
import time
import json
from pathlib import Path

import torch
from torch_mesh import engines, empty
from torch_mesh.compilation import parameters
from mesh_statistics import now, Observations
from assignment import window, domains, choice_domains, source_axes, source_operands, layout_operands
from mesh_source import revision
from mesh_relation import Algebra

env = lambda key, default: int(os.environ.get(key, default))
rows, k, n, steps, every, layers = env('ROWS', 1024), env('K', 3840), env('N', 3456), env('STEPS', 40), env('EVERY', 1), env('LAYERS', 4)
names, kind = os.environ.get('ENGINES', 'mps,cpu,ane'), os.environ.get('OP', 'ffn')
rates = json.loads(os.environ['ENGINE_CAPACITIES'])
source_scope = json.loads(os.environ['SOURCE_SCOPE'])
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
coreml = json.loads(Path(os.environ.get('COREML_CONFIGURATION', Path(__file__).resolve().parents[1] / 'etc/compute/coreml.json')).read_text())
pool = engines.Engines(names.split(','), compilation=parameters(), source_scope=source_scope,
                       ane={**coreml, 'tile': env('ANE_TILE', 128), 'block': env('ANE_BLOCK', 4096)})
projections, shared = [], []
for layer in range(layers):
    projections.append((torch.randn(k, k) / k ** 0.5).half().to('mps'))
    if kind == 'linear':
        w = (torch.randn(k, k) / k ** 0.5).half().to('mps')
        shared.append(engines.Linear(w, engines=pool, name=f'{kind}/{layer}', grain=128))
    else:
        if codes:
            gate, up, down = coded[layer]
        else:
            gate, up = ((torch.randn(n, k) / k ** 0.5).half().to('mps') for _ in range(2))
            down = (torch.randn(k, n) / n ** 0.5).half().to('mps')
        shared.append(engines.FFN(gate, up, down, engines=pool, name=f'{kind}/{layer}', grain=128))
keys = [operation.declare(rows, x.dtype, capacities=capacities) for operation in shared]
observations = Observations()
from torch_mesh.declarations import Function, execution_sequence
history, choices, layouts = [], [], []
implementation = revision(Path(__file__).resolve().parents[1])
for proposal, start in enumerate(range(0, steps, every)):
    calls = min(every, steps - start)
    family = {'program': 'engines', 'engines': names.split(','), 'hardware': pool.hardware,
              'kind': kind, 'rows': rows, 'k': k, 'n': n, 'layers': layers, 'calls': calls}
    history = window(history + observations.poll())
    assigned = engines.placement(pool, history, step=proposal, family=family, choices=choices, layouts=layouts)
    spans, source, algebra = [], x, Algebra()
    for layer, (projection, operation, key) in enumerate(zip(projections, shared, keys, strict=True)):
        normalized = empty(x.shape, dtype=x.dtype, device=x.device)
        def project(source=source, projection=projection, normalized=normalized):
            return normalized.copy_(torch.nn.functional.rms_norm(torch.nn.functional.linear(source, projection), (k,), eps=1e-6))
        project = Function(project, configuration=assigned['operands'], algebra=algebra,
                           source_scope=dict(scope=source_scope, operation=['projection', layer]))
        project.declare(None)
        output = empty((rows, operation.width), dtype=x.dtype)
        work = operation.function(normalized, parts=assigned['parts'][key], output=output, algebra=algebra,
                                  configuration=assigned['operands'])
        spans.extend((project, work))
        source = output
    program = execution_sequence(spans * calls, source, algebra=algebra)
    program.realize()
    choices = source_axes([choice for function in spans for choice in function.choices])
    layouts = [choice for function in spans for choice in function.layout_choices]
    executed = domains(pool.decisions, {}, pool.parameters, choices=choices, layouts=layouts)
    assigned.update(parts={key: value['parts'] for key, value in executed.items()}, domains=choice_domains(executed))
    assigned['operands']['source_preferences'] = source_operands(assigned['parts'], choices)
    assigned['operands']['layout_preferences'] = layout_operands(assigned['parts'], layouts)
    began = now()
    program.forward()
    completion = program.completion()
    observations.put(completion, {'episode': str(start), 'parts': assigned['parts'], 'proposal': assigned['proposal'], 'graphs': {},
        'family': family, 'domains': assigned['domains'], 'choices': choices, 'layouts': layouts, 'operands': assigned['operands'],
        'workload': {'layers': layers, 'kind': kind, 'rows': rows, 'k': k, 'n': n},
        'implementation': implementation, 'started': began, 'rank': 0, 'size': 1,
        'useful_work_bytes': rows * k * calls * 2,
        'report': {'program': 'engines', 'steps': calls, 'rows': rows, 'parts': assigned['parts']}})
while not observations.close():
    time.sleep(0)
