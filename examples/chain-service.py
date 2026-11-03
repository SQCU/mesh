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
from torch_mesh.compiler import Program
from mesh_statistics import now, measure, emit

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
    steps, rows, width = config['steps'], config['rows'], config['width']
    k = config['parameters']['steps_per_replay']['value']
    # every rank draws the same inputs from the configuration's seed: no broadcast of E into E (design/page-maps-
    # 2026-11-02.md, Construction 7)
    inputs, full = chain.inputs(config, steps, rows)
    folds = folded(config, full)
    del full
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=width)
    a, b = chain.weights(mesh, folds, width)
    del folds
    E, O = (torch_mesh.empty(steps, rows, width, dtype=torch.half) for _ in range(2))
    E.copy_(torch.from_numpy(inputs[:]))
    del inputs
    tick = chain.Tick(mesh, a, b, rows, width)
    t = torch.zeros(1, dtype=torch.int64, device='mps')
    # a chunk of k steps compiled once and instanced for every chunk of that length, the step counter t on the GPU
    # selecting its steps (torch_mesh/compiler.py)
    whole, rest = divmod(steps, k)
    stepper = {n: tick.stepper(E, O, n) for n in {k, rest} if n}
    program = Program([(stepper[k], (t,))] * whole + ([(stepper[rest], (t,))] if rest else []))
    torch_mesh.check_memory()
    program.prepare(1)
    started = now()
    program.forward()
    while (ended := program.completion()) is None:
        if now() - started > 8.0:
            raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
        time.sleep(0)
    summary = dict(blocks=partition.split(mesh, 'tp', width), rows=rows, width=width, steps=steps, steps_per_replay=k)
    emit({**measure(started, ended, O.numel() * O.element_size(), steps), 'workload': 'chain_sequence',
          'rank': rank, 'size': size, 'report': summary})
    dist.destroy_process_group()
    return {'output': O, 'configuration': config.get('placement')}


if __name__ == '__main__':
    main()
