import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh

import chain
import lstm


# docs/writing-a-service.md#the-lstm
def sizes(config, h):
    KIN, H, L = config['input_width'], sum(config['shares']), config['layers']
    return ((KIN + H) + 2 * H * (L - 1)) * 4 * h, L * 4 * h


# docs/writing-a-service.md#the-lstm
def prepared(config):
    weights, shares, L = np.load(config['weights']), config['shares'], config['layers']
    KIN, H = config['input_width'], sum(shares)
    layers = []
    for l in range(L):
        w_ih, w_hh, b = weights[f'w_ih_{l}'], weights[f'w_hh_{l}'], weights[f'b_{l}']
        if w_ih.shape != (4 * H, KIN if l == 0 else H) or w_hh.shape != (4 * H, H) or b.shape != (4 * H,):
            raise ValueError(f'Layer {l} is w_ih_{l} [4H, its input], w_hh_{l} [4H, H] and b_{l} [4H], H the sum of '
                             f'the shares')
        layers.append((np.concatenate([w_ih, w_hh], 1), b))
    held = []
    for r, h in enumerate(shares):
        units = np.arange(sum(shares[:r]), sum(shares[:r]) + h)
        columns = np.stack([units, H + units, 2 * H + units, 3 * H + units], 1).reshape(-1)
        held.append((np.concatenate([np.ascontiguousarray(w[columns], dtype=np.float16).reshape(-1) for w, _ in layers]),
                     np.concatenate([b[columns] for _, b in layers]).astype(np.float32)))
    return held


# docs/writing-a-service.md#the-lstm
def shares(config, rank, size, held):
    w_size, b_size = sizes(config, config['shares'][rank])
    W = torch_mesh.empty(w_size, dtype=torch.half)
    bias = torch_mesh.empty(b_size, dtype=torch.float32)
    if rank:
        dist.recv(W, src=0)
        dist.recv(bias, src=0)
        return W, bias
    for r in range(size):
        w, b = held[r]
        parts = (W, bias) if r == 0 else (torch_mesh.empty(w.size, dtype=torch.half), torch_mesh.empty(b.size, dtype=torch.float32))
        for part, array in zip(parts, (w, b)):
            part.copy_(torch.from_numpy(array))
            if r:
                dist.send(part, dst=r)
    return W, bias


# docs/writing-a-service.md#the-lstm
def lstm_stream(tick, rank, steps, block, E, O, inputs, outputs):
    e, pace, L = E.unbind(0), chain.Pace(), tick.L
    landing, first = chain.fill(E, rank, inputs, 0, steps, block), None
    for s in range(steps + L - 1):
        if s < steps and s % block == 0:
            pace.wait()
            ahead = chain.fill(E, rank, inputs, s + block, steps, block) if s + block < steps else None
            landing.wait()
            landing = ahead
        t = s - L
        issued = time.perf_counter()
        tick(steps, e[min(s, steps - 1) % len(e)], O[t % block] if O is not None and t >= 0 else None)
        if s == 0:
            first = time.perf_counter() - issued
        if O is not None and t >= 0 and (t + 1) % block == 0:
            chain.drain(t + 1 - block, steps, block, O, outputs)
        if s < steps and s % block == 0 and s:
            pace.mark()
    tick.land(e[(steps - 1) % len(e)], O[(steps - 1) % block] if O is not None else None)
    chain.drain((steps - 1) // block * block, steps, block, O, outputs)
    return pace.waited, pace.done, first


# docs/writing-a-service.md#the-lstm
def main():
    config = json.loads(open(sys.argv[1]).read())
    held = prepared(config) if os.environ['RANK'] == '0' else None
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if len(config['shares']) != size:
        raise ValueError('The configuration gives each rank a share')
    KIN, H, L = config['input_width'], sum(config['shares']), config['layers']
    steps, rows, inputs = chain.operand(config['input'], rank, KIN)
    block, deep, _ = chain.ring(config['ring_bytes'], steps, rows * KIN * torch.half.itemsize)
    W, bias = shares(config, rank, size, held)
    del held
    tick = lstm.Tick(W, bias, rows, config['shares'], rank, L, KIN, chain.capability(config['capabilities']))
    del W, bias
    E = torch_mesh.empty(deep, rows, KIN, dtype=torch.half)
    O = torch.empty(block, rows, H, dtype=torch.half, device='mps') if rank == 0 else None
    outputs = np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, H)) if rank == 0 else None
    torch.mps.synchronize()
    dist.barrier()
    began = time.perf_counter()
    paced, done, first = lstm_stream(tick, rank, steps, block, E, O, inputs, outputs)
    hosted = time.perf_counter() - began - paced
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    ticks = steps + L - 1
    steady = (began + seconds - done[0]) / (ticks - block - 1) if done else None
    moved = (size - 1) * rows * L * config['shares'][rank] * torch.half.itemsize
    summary = dict(rank=rank, size=size, shares=config['shares'], layers=L, rows=rows, input_width=KIN, width=H,
                   steps=steps, ticks=ticks, block=block, seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   per_stream_steps_per_second=steps / seconds, tick_microseconds=seconds / ticks * 1e6,
                   latency_ticks=L, host_microseconds=hosted / ticks * 1e6, paced_microseconds=paced / ticks * 1e6,
                   first_issue_milliseconds=first * 1e3, steady_tick_microseconds=steady * 1e6 if steady else None,
                   sent_bytes_per_tick=moved, kernel={key: tick.args[key] for key in lstm.ARGS})
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
