import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition

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
def folded(config):
    full = tuple(np.load(f'{config["weights"]}/{name}.npy', mmap_mode='r') for name in ('a', 'b'))
    if any(w.shape != (config['width'],) * 2 for w in full):
        raise ValueError('The weights a and b are [width, width] each')
    blocks = config['rotation_blocks']
    held = tuple(chain.region(w.shape, torch.half) for w in full)
    for h, w, block, axis in zip(held, full, blocks, (1, 0)):
        h[...] = rotated(w, block, axis)
    return held


# docs/writing-a-service.md#the-configuration
def main():
    config = json.loads(open(sys.argv[1]).read())
    folds = folded(config) if os.environ['RANK'] == '0' else None
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size, width = dist.get_rank(), dist.get_world_size(), config['width']
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=width)
    steps, rows, inputs = chain.operand(config['input'], rank, width)
    at, b = chain.weights(mesh, folds, width)
    del folds
    E, O = (torch_mesh.empty(steps, rows, width, dtype=torch.half) for _ in range(2))
    o, t = torch_mesh.empty(rows, width, dtype=torch.half).zero_(), torch.zeros(1, dtype=torch.int64, device='mps')
    recording = torch_mesh.record(chain.stepper(E, O, o, t, at, b))
    torch_mesh.replay(recording)
    o.zero_()
    t.zero_()
    if rank == 0:
        E.copy_(torch.from_numpy(inputs[:]))
    torch_mesh.check_memory()
    torch.mps.synchronize()
    dist.barrier()
    half = steps // 2
    middle, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    began = time.perf_counter()
    dist.broadcast(E, src=0)
    for s in range(steps):
        recording.compiled()
        if s == half:
            middle.record()
    end.record()
    hosted = time.perf_counter() - began
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    steady = middle.elapsed_time(end) / 1e3 / (steps - 1 - half)
    if rank == 0:
        torch.from_numpy(np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, width))).copy_(O)
    p = torch_mesh.empty(rows, width, dtype=torch.half).zero_()
    dist.barrier()
    torch.mps.synchronize()
    began = time.perf_counter()
    for _ in range(steps):
        dist.all_reduce(p)
    torch.mps.synchronize()
    crossing = time.perf_counter() - began
    summary = dict(rank=rank, size=size, blocks=partition.split(mesh, 'tp', width), rows=rows, width=width, steps=steps,
                   seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   steady_aggregate_rows_per_second=rows / steady, per_stream_steps_per_second=steps / seconds,
                   step_microseconds=seconds / steps * 1e6, steady_step_microseconds=steady * 1e6,
                   host_microseconds=hosted / steps * 1e6, crossing_microseconds=crossing / steps * 1e6)
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
