import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh

import chain


# docs/writing-a-service.md#how-the-example-moves-its-data
def loaded(config):
    weights = np.load(config['weights'])
    full = tuple(weights[name] for name in ('a', 'b'))
    if any(w.shape != (sum(config['parts']), config['width']) for w in full):
        raise ValueError('The weights a and b are [sum of the parts, width] each')
    return tuple(np.ascontiguousarray(w, dtype=np.float16) for w in full)


# docs/writing-a-service.md#how-the-example-moves-its-data
def shares(config, rank, size, full):
    parts, width = config['parts'], config['width']
    a, b = (torch_mesh.empty(parts[rank], width, dtype=torch.half) for _ in range(2))
    matrix = [parts[q] * width if p == 0 else 0 for p in range(size) for q in range(size)]
    for held, w in zip((a, b), full or (None, None)):
        whole = torch_mesh.empty(sum(parts), width, dtype=torch.half) if rank == 0 else None
        if whole is not None:
            whole.copy_(torch.from_numpy(w))
        torch_mesh.all_to_all(held, whole, matrix).wait()
    return a, b


# docs/writing-a-service.md#how-the-example-moves-its-data
def chain_stream(step, steps, E, O):
    first = torch.mps.Event(enable_timing=True)
    for t in range(steps):
        step(E[t], O[t - 1] if O is not None and t else None)
        if t == 0:
            first.record()
    step.land(E[steps - 1], O[steps - 1] if O is not None else None)
    return first


# docs/writing-a-service.md#how-the-example-moves-its-data
def crossings(steps, P):
    for t in range(steps):
        dist.all_reduce(P)


# docs/writing-a-service.md#the-configuration
def main():
    config = json.loads(open(sys.argv[1]).read())
    full = loaded(config) if os.environ['RANK'] == '0' else None
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if len(config['parts']) != size:
        raise ValueError('The configuration gives one part a rank')
    width = config['width']
    steps, rows, inputs = chain.operand(config['input'], rank, width)
    a, b = shares(config, rank, size, full)
    del full
    step = chain.Step(a, b, rows, config['row_capacity'], size - 1, chain.capability(config['capabilities']))
    del a, b
    P = torch_mesh.empty(rows, width, dtype=torch.half)
    O = torch_mesh.empty(steps, rows, width, dtype=torch.half) if rank == 0 else None
    E = chain.landed(rank, inputs, steps, rows, width)
    start, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    began = time.perf_counter()
    start.record()
    first = chain_stream(step, steps, E, O)
    end.record()
    hosted = time.perf_counter() - began
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    steady = first.elapsed_time(end) / 1e3 / (steps - 1) if steps > 1 else None
    opened = start.elapsed_time(first) / 1e3
    if O is not None:
        torch.from_numpy(np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, width))).copy_(O)
    began = time.perf_counter()
    crossings(steps, P)
    torch.mps.synchronize()
    crossing = time.perf_counter() - began
    moved = 2 * (size - 1) * rows * width * torch.half.itemsize
    summary = dict(rank=rank, size=size, parts=config['parts'], rows=rows, width=width, steps=steps,
                   seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   per_stream_steps_per_second=steps / seconds, step_microseconds=seconds / steps * 1e6,
                   host_microseconds=hosted / steps * 1e6,
                   first_step_milliseconds=opened * 1e3,
                   steady_step_microseconds=steady * 1e6 if steady else None,
                   steady_aggregate_rows_per_second=rows / steady if steady else None,
                   crossing_microseconds=crossing / steps * 1e6, collective_bytes_per_step=moved,
                   collective_gigabytes_per_second=moved * steps / seconds / 1e9,
                   kernel={key: step.args[key] for key in chain.ARGS})
    print(json.dumps(summary), flush=True)
    step.exchange.free()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
