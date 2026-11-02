import json
import time
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition
from torch_mesh.declarations import Function, execution_sequence
from torch_mesh.configuration import placement, bind, record
from torch_mesh import compute
from mesh_statistics import now, next_configuration

import chain


# docs/writing-a-service.md#rotations
def rotated(w, block, axis):
    n = w.shape[axis]
    if block < 1 or block & (block - 1) or n % block:
        raise ValueError(f'A rotation block is a power of two that divides its {n} columns, not {block}')
    v, spare, s = chain.region(w.shape, torch.float64), None, 1
    v[...] = w
    half = lambda t, i: t[(slice(None),) * (axis + 1) + (i,)]
    while s < block:
        spare = chain.region(w.shape, torch.float64) if spare is None else spare
        x, y = (t.reshape(*w.shape[:axis], n // (2 * s), 2, s, *w.shape[axis + 1:]) for t in (v, spare))
        np.add(half(x, 0), half(x, 1), out=half(y, 0))
        np.subtract(half(x, 0), half(x, 1), out=half(y, 1))
        np.divide(spare, np.sqrt(2.0), out=spare)
        v, spare, s = spare, v, s * 2
    return v


# docs/writing-a-service.md#how-the-example-moves-its-data
def folded(config, full):
    blocks = config['rotation_blocks']
    held = tuple(chain.region(w.shape, torch.half) for w in full)
    for h, w, block, axis in zip(held, full, blocks, (1, 0)):
        h[...] = rotated(w, block, axis)
    return held


# docs/writing-a-service.md#the-configuration
def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    capabilities = compute.metal(config['capabilities'])
    steps, rows = config['steps'], config['rows']
    # every rank draws the same inputs from the configuration's seed: no broadcast of E into E (design/page-maps-
    # 2026-11-02.md, Construction 7); rank 0's folds are the ones distributed
    inputs, full = chain.inputs(config, steps, rows)
    folds = folded(config, full)
    del full
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size, width = dist.get_rank(), dist.get_world_size(), config['width']
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=width)
    workload = {'function': 'chain_sequence', 'width': width, 'steps': steps, 'rows': rows}
    configuration = placement(mesh, 'tp', workload, config.get('placement'), parameters=config['parameters'],
        observations=config.get('observations', []), proposal=config.get('proposal', 0),
        family=config.get('family', workload), forms=config.get('forms', ()))
    chosen = configuration['assignment']['operands']
    k = chosen['steps_per_replay']
    a, b = chain.weights(mesh, folds, width)
    del folds
    E, O = (torch_mesh.empty(steps, rows, width, dtype=torch.half) for _ in range(2))
    E.copy_(torch.from_numpy(inputs[:]))
    del inputs
    group = mesh.get_group('tp')
    # the chain's step through the frontend the LSTM uses: one declaration for each distinct chunk length, replayed for
    # every chunk of that length, the step counter t on the GPU selecting its steps (examples/lstm-service.py)
    tick = chain.Tick(mesh, a, b, rows, width)
    t = torch.zeros(1, dtype=torch.int64, device='mps')
    full, rest = divmod(steps, k)
    chunks = [k] * full + [rest] * bool(rest)
    from mesh_relation import Algebra
    algebra = Algebra()
    unit = Function(tick.stepper(E, O, 1), t, algebra=algebra,
                    source_scope={'program': 'chain_sequence', 'phase': 'step'}).capture()
    functions = {}
    for count in dict.fromkeys(chunks):
        function = unit.repeated(count)
        function.declare(group)
        functions[count] = function
    ordered = [functions[count] for count in chunks]
    groups = []
    for function in ordered:
        own = function.groups.get(group.group_name, [])
        if len(own) == 1 and groups and isinstance(groups[-1], tuple) and groups[-1][0] is own[0]:
            groups[-1] = (own[0], groups[-1][1] + 1)
        elif len(own) == 1:
            groups.append((own[0], 1))
        else:
            groups.extend(own)
    torch_mesh.check_memory()
    program = execution_sequence(ordered, O, algebra=algebra)
    def prepare():
        program.realize(1)
        return {'layouts': [choice for function in ordered for choice in function.layout_choices],
                'choices': [choice for function in ordered for choice in function.choices]}
    bind(configuration, groups, prepare)
    started = now()
    program.forward()
    completion = program.completion()
    summary = dict(blocks=partition.split(mesh, 'tp', width), rows=rows, width=width, steps=steps,
                   steps_per_replay=k)
    record(configuration, completion, started, O.numel() * O.element_size(), summary)
    while not configuration['observations'].close():
        time.sleep(0)
    dist.destroy_process_group()
    state = next_configuration(configuration['observations'].declaration,
        [*configuration['group'].configuration_observations, *configuration['observations'].poll()],
        family=configuration['family']['source'])
    return {'output': O, 'configuration': state}


if __name__ == '__main__':
    main()
