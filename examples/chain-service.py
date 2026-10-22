import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh

import chain


# docs/writing-a-service.md#rotations
def rotated(w, block, axis):
    v = np.moveaxis(np.asarray(w, np.float64), axis, -1)
    shape, n = v.shape, v.shape[-1]
    if block < 1 or block & (block - 1) or n % block:
        raise ValueError(f'A rotation block is a power of two that divides its {n} columns, not {block}')
    s = 1
    while s < block:
        v = v.reshape(*shape[:-1], n // (2 * s), 2, s)
        v = np.stack((v[..., 0, :] + v[..., 1, :], v[..., 0, :] - v[..., 1, :]), axis=-2) / np.sqrt(2.0)
        s *= 2
    return np.moveaxis(v.reshape(shape), -1, axis)


# docs/writing-a-service.md#how-the-example-moves-its-data
def folded(config):
    weights = np.load(config['weights'])
    full = tuple(weights[name] for name in ('a', 'b'))
    if any(w.shape != (sum(config['parts']), config['width']) for w in full):
        raise ValueError('The weights a and b are [sum of the parts, width] each')
    blocks = config['rotation_blocks']
    return tuple(np.ascontiguousarray(rotated(w, block, axis), dtype=np.float16)
                 for w, block, axis in zip(full, blocks, (1, 0)))


# docs/writing-a-service.md#how-the-example-moves-its-data
def shares(config, rank, size, folds):
    parts, width = config['parts'], config['width']
    a, b = (torch_mesh.empty(parts[rank], width, dtype=torch.half) for _ in range(2))
    if rank:
        dist.recv(a, src=0)
        dist.recv(b, src=0)
        return a, b
    for r in range(size):
        begin = sum(parts[:r])
        for held, w in zip((a, b), folds):
            part = held if r == 0 else torch_mesh.empty(parts[r], width, dtype=torch.half)
            part.copy_(torch.from_numpy(w[begin:begin + parts[r]]))
            if r:
                dist.send(part, dst=r)
    return a, b


# docs/writing-a-service.md#how-the-example-moves-its-data
def chain_stream(step, rank, steps, block, E, O, inputs, outputs):
    e, pace = [s for b in E for s in b.unbind(0)], chain.Pace()
    o = (None,) * block if O is None else O.unbind(0)
    deep = len(e)
    landing = chain.fill(E, rank, inputs, 0, steps, block)
    for t in range(steps):
        if t % block == 0:
            pace.wait()
            ahead = chain.fill(E, rank, inputs, t + block, steps, block) if t + block < steps else None
            landing.wait()
            landing = ahead
        step(e[t % deep], o[(t - 1) % block])
        if t % block == 0 and t:
            chain.drain(t - block, steps, block, O, outputs)
            pace.mark()
    step.land(e[(steps - 1) % deep], o[(steps - 1) % block])
    chain.drain((steps - 1) // block * block, steps, block, O, outputs)
    return pace.first


# docs/writing-a-service.md#how-the-example-moves-its-data
def crossings(steps, block, P):
    pace = chain.Pace()
    for t in range(steps):
        if t % block == 0:
            pace.wait()
        dist.all_reduce(P[t % len(P)])
        if (t + 1) % block == 0 and t + 1 < steps:
            pace.mark()


# docs/writing-a-service.md#the-configuration
def main():
    config = json.loads(open(sys.argv[1]).read())
    folds = folded(config) if os.environ['RANK'] == '0' else None
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if len(config['parts']) != size:
        raise ValueError('The configuration gives one part a rank')
    width = config['width']
    steps, rows, inputs = chain.operand(config['input'], rank, width)
    block, blocks, sums = chain.ring(config['ring_bytes'], steps, rows * width * torch.half.itemsize)
    a, b = shares(config, rank, size, folds)
    del folds
    step = chain.Step(a, b, rows, config['row_capacity'], size - 1, chain.capability(config['capabilities']))
    del a, b
    E = [torch_mesh.empty(block, rows, width, dtype=torch.half) for _ in range(blocks)]
    P = [torch_mesh.empty(rows, width, dtype=torch.half) for _ in range(sums)]
    O = torch.empty(block, rows, width, dtype=torch.half, device='mps') if rank == 0 else None
    outputs = np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, width)) if rank == 0 else None
    start, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    torch.mps.synchronize()
    dist.barrier()
    began = time.perf_counter()
    start.record()
    first = chain_stream(step, rank, steps, block, E, O, inputs, outputs)
    end.record()
    hosted = time.perf_counter() - began
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    steady = first.elapsed_time(end) / 1e3 / (steps - block - 1) if first is not None else None
    opened = start.elapsed_time(first) / 1e3 if first is not None else None
    dist.barrier()
    torch.mps.synchronize()
    began = time.perf_counter()
    crossings(steps, block, P)
    torch.mps.synchronize()
    crossing = time.perf_counter() - began
    moved = 2 * (size - 1) * rows * width * torch.half.itemsize
    summary = dict(rank=rank, size=size, parts=config['parts'], rows=rows, width=width, steps=steps, block=block,
                   input_ring=blocks * block, sum_ring=sums, seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   per_stream_steps_per_second=steps / seconds, step_microseconds=seconds / steps * 1e6,
                   host_microseconds=hosted / steps * 1e6,
                   first_block_milliseconds=opened * 1e3 if first is not None else None,
                   steady_step_microseconds=steady * 1e6 if steady else None,
                   steady_aggregate_rows_per_second=rows / steady if steady else None,
                   crossing_microseconds=crossing / steps * 1e6, collective_bytes_per_step=moved,
                   collective_gigabytes_per_second=moved * steps / seconds / 1e9,
                   kernel={key: step.args[key] for key in chain.ARGS})
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
