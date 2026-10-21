import collections
import json
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh

import chain


# docs/writing-a-service.md#the-function
def reference(inputs, a, b, steps):
    x, out = inputs[0], []
    for t in range(steps):
        out.append(np.tanh(np.tanh(x @ a.T) @ b))
        if t + 1 < steps:
            x = inputs[t + 1] + out[-1]
    return np.stack(out)


# docs/writing-a-service.md#how-the-example-moves-its-data
def operand(config, rank):
    shape, inputs = torch_mesh.empty(2, dtype=torch.int64, device='cpu'), None
    if rank == 0:
        inputs = np.load(config['input'], mmap_mode='r')
        if (inputs.ndim != 3 or inputs.shape[2] != config['width'] or inputs.dtype != np.float16
                or not inputs.flags.c_contiguous):
            raise ValueError('The input is C-ordered float16 [steps, rows, width]')
        shape[0], shape[1] = inputs.shape[:2]
    dist.broadcast(shape, src=0)
    steps, rows = shape.tolist()
    return steps, rows, inputs


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


# docs/writing-a-service.md#the-rings
def ring(config, steps, rows):
    slot = rows * config['width'] * 2
    slots = config['ring_bytes'] // (2 * slot)
    if slots < 2:
        raise ValueError(f'ring_bytes {config["ring_bytes"]} holds fewer than two slots of {slot} bytes in each of the '
                         f'two rings; two slots each are {4 * slot} bytes')
    if slots >= steps:
        return steps, steps
    return 2 * (slots // 2), slots // 2


# docs/writing-a-service.md#the-rings
def span(first, steps, block, depth):
    last = min(steps, first + block)
    return slice(first, last), slice(first % depth, first % depth + last - first)


class Pace:
    # docs/writing-a-service.md#the-rings
    def __init__(self, depth, block):
        self.blocks, self.marks, self.waited = depth // block, collections.deque(), 0.0

    # docs/writing-a-service.md#the-rings
    def wait(self):
        if len(self.marks) == self.blocks:
            began = time.perf_counter()
            self.marks.popleft().synchronize()
            self.waited += time.perf_counter() - began

    # docs/writing-a-service.md#the-rings
    def mark(self):
        self.marks.append(torch.mps.Event())
        self.marks[-1].record()


# docs/writing-a-service.md#how-the-example-moves-its-data
def chain_stream(step, rank, steps, block, depth, E, P, inputs, outputs):
    e, p, pace = E.unbind(0), P.unbind(0), Pace(depth, block)
    if rank == 0:
        held, slots = span(0, steps, block, depth)
        E[slots].copy_(torch.from_numpy(inputs[held]), non_blocking=True)
    landing = {0: dist.broadcast(e[0], src=0, async_op=True)}
    for t in range(steps):
        if t % block == 0:
            pace.wait()
            if rank == 0 and t + block < steps:
                held, slots = span(t + block, steps, block, depth)
                E[slots].copy_(torch.from_numpy(inputs[held]), non_blocking=True)
        if t + 1 < steps:
            landing[t + 1] = dist.broadcast(e[(t + 1) % depth], src=0, async_op=True)
        landing.pop(t).wait()
        step(e[t % depth], p[(t - 1) % depth], p[t % depth], t == 0)
        dist.all_reduce(p[t % depth])
        if (t + 1) % block == 0 or t + 1 == steps:
            if rank == 0:
                held, slots = span(t - t % block, steps, block, depth)
                torch.from_numpy(outputs[held]).copy_(torch.tanh(P[slots]), non_blocking=True)
            if t + 1 < steps:
                pace.mark()
    return pace.waited


# docs/writing-a-service.md#how-the-example-moves-its-data
def crossings(steps, block, depth, P):
    p, pace = P.unbind(0), Pace(depth, block)
    for t in range(steps):
        if t % block == 0:
            pace.wait()
        dist.all_reduce(p[t % depth])
        if (t + 1) % block == 0 and t + 1 < steps:
            pace.mark()


# docs/writing-a-service.md#the-configuration
def main():
    config = json.loads(open(sys.argv[1]).read())
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if len(config['parts']) != size:
        raise ValueError('The configuration gives one part a rank')
    width = config['width']
    steps, rows, inputs = operand(config, rank)
    depth, block = ring(config, steps, rows)
    a, b, full = shares(config, rank, size)
    step = chain.Step(a, b, rows, config['row_capacity'], config['threads'], config['patience'], config['trials'])
    del a, b
    E = torch_mesh.empty(depth, rows, width, dtype=torch.half)
    P = torch_mesh.empty(depth, rows, width, dtype=torch.half)
    outputs = np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, width)) if rank == 0 else None
    torch.mps.synchronize()
    dist.barrier()
    began = time.perf_counter()
    paced = chain_stream(step, rank, steps, block, depth, E, P, inputs, outputs)
    issuing = time.perf_counter() - began - paced
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    step.check()
    dist.barrier()
    torch.mps.synchronize()
    began = time.perf_counter()
    crossings(steps, block, depth, P)
    torch.mps.synchronize()
    crossing = time.perf_counter() - began
    summary = dict(rank=rank, size=size, parts=config['parts'], rows=rows, width=width, steps=steps, ring_depth=depth,
                   block=block, seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   per_stream_steps_per_second=steps / seconds, step_microseconds=seconds / steps * 1e6,
                   issue_microseconds=issuing / steps * 1e6, paced_microseconds=paced / steps * 1e6,
                   crossing_microseconds=crossing / steps * 1e6, kernel={key: step.args[key] for key in chain.ARGS})
    if rank == 0 and config['check']:
        count = config['check']
        expected = reference(np.asarray(inputs[:count], np.float64), *(w.astype(np.float64) for w in full), count)
        summary.update(check_steps=count, check_max_difference=float(np.max(np.abs(expected - outputs[:count]))))
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
