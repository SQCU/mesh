import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition
from torch_mesh import chain as sequence
from torch_mesh.configuration import placement, bind, record
from torch_mesh import _stream, compute
from mesh_statistics import now

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
def main():
    config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    capabilities = compute.metal(config['capabilities'])
    steps, rows = config['steps'], config['rows']
    inputs, folds = None, None
    if os.environ['RANK'] == '0':
        inputs, full = chain.inputs(config, steps, rows)
        folds = folded(config, full)
        del full
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size, width = dist.get_rank(), dist.get_world_size(), config['width']
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=width)
    configuration = placement(mesh, 'tp', {'function': 'chain_sequence', 'width': width, 'steps': steps, 'rows': rows},
                            config['observations'])
    k = config['steps_per_replay']
    a, b = chain.weights(mesh, folds, width)
    del folds
    E, O = (torch_mesh.empty(steps, rows, width, dtype=torch.half) for _ in range(2))
    group = mesh.get_group('tp')
    groups, operands = sequence.operands(a, b, E, O, group, k, config['resident_tiles'], config['input_frames'], capabilities)
    if rank == 0:
        E.copy_(torch.from_numpy(inputs[:]))
    torch_mesh.check_memory()
    bind(configuration, groups, lambda: sequence.prepare(operands))
    completion = _stream.chain_completion(operands[0])
    started = now()
    _stream.chain_run(operands[0])
    summary = dict(blocks=partition.split(mesh, 'tp', width), rows=rows, width=width, steps=steps,
                   steps_per_replay=k)
    record(configuration, completion, started, O.numel() * O.element_size(), summary)
    dist.destroy_process_group()
    return O


if __name__ == '__main__':
    main()
