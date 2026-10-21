import json
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh


# docs/writing-a-service.md#the-function
def step(x, a, b, z):
    torch.matmul(torch.tanh(x @ a.T), b, out=z)
    dist.all_reduce(z)
    return torch.tanh(z)


# docs/writing-a-service.md#the-function
def reference(inputs, a, b, steps):
    x, out = inputs[0], []
    for t in range(steps):
        out.append(np.tanh(np.tanh(x @ a.T) @ b))
        if t + 1 < steps:
            x = inputs[t + 1] + out[-1]
    return np.stack(out)


# docs/writing-a-service.md#how-the-example-moves-its-data
def shares(config, rank, size):
    parts, width = config['parts'], config['width']
    a, b = (torch_mesh.empty(parts[rank], width, dtype=torch.half) for _ in range(2))
    if rank:
        dist.recv(a, src=0)
        dist.recv(b, src=0)
        return a, b, None
    weights = np.load(config['weights'])
    full = tuple(weights[name] for name in ('a', 'b'))
    if any(w.shape != (sum(parts), width) for w in full):
        raise ValueError('The weights a and b are [sum of the parts, width] each')
    for r in range(size):
        begin = sum(parts[:r])
        for held, w in zip((a, b), full):
            part = torch.from_numpy(np.ascontiguousarray(w[begin:begin + parts[r]])).to('mps', torch.half)
            if r:
                dist.send(part, dst=r)
            else:
                held.copy_(part)
    return a, b, full


# docs/writing-a-service.md#the-configuration
def main():
    config = json.loads(open(sys.argv[1]).read())
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if len(config['parts']) != size:
        raise ValueError('The configuration gives one part a rank')
    rows, width, steps, chunk = (config[key] for key in ('rows', 'width', 'steps', 'chunk'))
    a, b, full = shares(config, rank, size)
    e = torch_mesh.empty(chunk, rows, width, dtype=torch.half)
    z = torch_mesh.empty(rows, width, dtype=torch.half)
    if rank == 0:
        inputs = np.load(config['input'], mmap_mode='r')
        if inputs.shape != (steps, rows, width):
            raise ValueError('The input is [steps, rows, width]')
        outputs = np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, width))
        held = torch.empty(chunk, rows, width, dtype=torch.half, device='mps')
    torch.mps.synchronize()
    dist.barrier()
    began = time.perf_counter()
    previous = None
    for first in range(0, steps, chunk):
        count = min(chunk, steps - first)
        if rank == 0:
            e[:count].copy_(torch.from_numpy(np.array(inputs[first:first + count])))
        dist.broadcast(e[:count], src=0)
        for i in range(count):
            previous = step(e[i] if previous is None else e[i] + previous, a, b, z)
            if rank == 0:
                held[i].copy_(previous)
        if rank == 0:
            outputs[first:first + count] = held[:count].cpu().numpy()
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    summary = dict(rank=rank, size=size, parts=config['parts'], rows=rows, width=width, steps=steps, seconds=seconds,
                   aggregate_rows_per_second=steps * rows / seconds, per_stream_steps_per_second=steps / seconds,
                   step_microseconds=seconds / steps * 1e6)
    if rank == 0 and config['check']:
        count = config['check']
        expected = reference(np.asarray(inputs[:count], np.float64), *(w.astype(np.float64) for w in full), count)
        summary.update(check_steps=count, check_max_difference=float(np.max(np.abs(expected - outputs[:count]))))
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
