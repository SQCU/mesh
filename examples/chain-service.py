import collections
import json
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh

import chain

RING_BLOCKS = 2


# docs/writing-a-service.md#the-function
def reference(inputs, a, b, steps, carried):
    x, out = inputs[0] + carried, []
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
        if (inputs.ndim != 3 or 0 in inputs.shape or inputs.shape[2] != config['width'] or inputs.dtype != np.float16
                or not inputs.flags.c_contiguous):
            raise ValueError('The input is C-ordered float16 [steps, rows, width], none of them 0')
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
    slot = rows * config['width'] * torch.half.itemsize
    slots = config['ring_bytes'] // slot
    if slots >= 2 * steps:
        return steps, steps, steps
    block = slots // (RING_BLOCKS + 1)
    if block < 1:
        raise ValueError(f'ring_bytes {config["ring_bytes"]} holds {slots} slots of {slot} bytes; the input ring of '
                         f'{RING_BLOCKS} blocks and the sum ring of one take {RING_BLOCKS + 1} slots at least, '
                         f'{(RING_BLOCKS + 1) * slot} bytes')
    return block, min(steps, RING_BLOCKS * block), block


# docs/writing-a-service.md#the-rings
def span(first, steps, block, depth):
    last = min(steps, first + block)
    return slice(first, last), slice(first % depth, first % depth + last - first)


class Pace:
    # docs/writing-a-service.md#the-rings
    def __init__(self):
        self.marks, self.done, self.waited = collections.deque(), [], 0.0

    # docs/writing-a-service.md#the-rings
    def wait(self):
        if len(self.marks) == RING_BLOCKS - 1:
            began = time.perf_counter()
            self.marks.popleft().synchronize()
            self.done.append(time.perf_counter())
            self.waited += self.done[-1] - began

    # docs/writing-a-service.md#the-rings
    def mark(self):
        self.marks.append(torch.mps.Event())
        self.marks[-1].record()


# docs/writing-a-service.md#how-the-example-moves-its-data
def fill(E, rank, inputs, first, steps, block):
    held, slots = span(first, steps, block, len(E))
    if rank == 0:
        E[slots].copy_(torch.from_numpy(inputs[held]), non_blocking=True)
    return dist.broadcast(E[slots], src=0, async_op=True)


# docs/writing-a-service.md#how-the-example-moves-its-data
def drain(first, steps, block, O, outputs):
    if O is not None:
        held = span(first, steps, block, len(O))[0]
        torch.from_numpy(outputs[held]).copy_(O[:held.stop - held.start], non_blocking=True)


# docs/writing-a-service.md#how-the-example-moves-its-data
def chain_stream(step, rank, steps, block, E, P, O, inputs, outputs):
    e, p, pace = E.unbind(0), P.unbind(0), Pace()
    deep, sums = len(e), len(p)
    landing, crossing = fill(E, rank, inputs, 0, steps, block), None
    for t in range(steps):
        if t % block == 0:
            pace.wait()
            ahead = fill(E, rank, inputs, t + block, steps, block) if t + block < steps else None
            landing.wait()
            landing = ahead
        issued = time.perf_counter()
        crossing = step(e[t % deep], p[t % sums], None if O is None else O[(t - 1) % block], crossing)
        if t == 0:
            first = time.perf_counter() - issued
        if t % block == 0 and t:
            drain(t - block, steps, block, O, outputs)
            pace.mark()
    step.land(e[(steps - 1) % deep], None if O is None else O[(steps - 1) % block], crossing)
    drain((steps - 1) // block * block, steps, block, O, outputs)
    return pace.waited, pace.done, first


# docs/writing-a-service.md#how-the-example-moves-its-data
def crossings(steps, block, P):
    p, pace = P.unbind(0), Pace()
    for t in range(steps):
        if t % block == 0:
            pace.wait()
        dist.all_reduce(p[t % len(p)])
        if (t + 1) % block == 0 and t + 1 < steps:
            pace.mark()


# docs/writing-a-service.md#running-it
def checked(config, inputs, outputs, full, steps, block):
    count, weights, found = config['check'], [w.astype(np.float64) for w in full], {}
    for first in sorted({0, (steps - 1) // block * block}):
        last = min(steps, first + count)
        carried = np.asarray(outputs[first - 1], np.float64) if first else 0.0
        expected = reference(np.asarray(inputs[first:last], np.float64), *weights, last - first, carried)
        found[first] = float(np.max(np.abs(expected - outputs[first:last])))
    return found


# docs/writing-a-service.md#the-configuration
def main():
    config = json.loads(open(sys.argv[1]).read())
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if len(config['parts']) != size:
        raise ValueError('The configuration gives one part a rank')
    width = config['width']
    steps, rows, inputs = operand(config, rank)
    block, deep, sums = ring(config, steps, rows)
    a, b, full = shares(config, rank, size)
    step = chain.Step(a, b, rows, config['row_capacity'], chain.capability(config['capability']))
    del a, b
    E = torch_mesh.empty(deep, rows, width, dtype=torch.half)
    P = torch_mesh.empty(sums, rows, width, dtype=torch.half)
    O = torch.empty(block, rows, width, dtype=torch.half, device='mps') if rank == 0 else None
    outputs = np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, width)) if rank == 0 else None
    torch.mps.synchronize()
    dist.barrier()
    began = time.perf_counter()
    paced, done, first = chain_stream(step, rank, steps, block, E, P, O, inputs, outputs)
    hosted = time.perf_counter() - began - paced
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    step.check()
    steady = (began + seconds - done[0]) / (steps - block - 1) if done else None
    opened = done[0] - began if done else None
    dist.barrier()
    torch.mps.synchronize()
    began = time.perf_counter()
    crossings(steps, block, P)
    torch.mps.synchronize()
    crossing = time.perf_counter() - began
    moved = 2 * rows * width * torch.half.itemsize
    summary = dict(rank=rank, size=size, parts=config['parts'], rows=rows, width=width, steps=steps, block=block,
                   input_ring=deep, sum_ring=sums, seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   per_stream_steps_per_second=steps / seconds, step_microseconds=seconds / steps * 1e6,
                   host_microseconds=hosted / steps * 1e6, paced_microseconds=paced / steps * 1e6,
                   first_issue_milliseconds=first * 1e3, first_block_milliseconds=opened * 1e3 if done else None,
                   steady_step_microseconds=steady * 1e6 if steady else None,
                   steady_aggregate_rows_per_second=rows / steady if steady else None,
                   crossing_microseconds=crossing / steps * 1e6, collective_bytes_per_step=moved,
                   collective_gigabytes_per_second=moved * steps / seconds / 1e9,
                   kernel={key: step.args[key] for key in chain.ARGS})
    if rank == 0 and config['check']:
        summary.update(check_steps=config['check'],
                       check_max_difference=checked(config, inputs, outputs, full, steps, block))
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
