import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh

import chain
import lstm

CHUNK = 64


# docs/writing-a-service.md#the-lstm
def sizes(config, h):
    KIN, H, L = config['input_width'], sum(config['shares']), config['layers']
    return ((KIN + H) + 2 * H * (L - 1)) * 4 * h, L * 4 * h


# docs/writing-a-service.md#the-lstm
def prepared(config, tile):
    shares, L, KIN = config['shares'], config['layers'], config['input_width']
    H, folder = sum(shares), Path(config['weights'])
    load = lambda name: np.load(folder / f'{name}.npy', mmap_mode='r')
    for l in range(L):
        if (load(f'w_ih_{l}').shape != (4 * H, KIN if l == 0 else H) or load(f'w_hh_{l}').shape != (4 * H, H)
                or load(f'b_{l}').shape != (4 * H,)):
            raise ValueError(f'Layer {l} is w_ih_{l} [4H, its input], w_hh_{l} [4H, H] and b_{l} [4H], H the sum of '
                             f'the shares')
    held = []
    for r, h in enumerate(shares):
        w_size, b_size = sizes(config, h)
        W = torch_mesh.empty(w_size, dtype=torch.half, device='cpu')
        bias = torch_mesh.empty(b_size, dtype=torch.float32, device='cpu')
        units = np.arange(sum(shares[:r]), sum(shares[:r]) + h)
        rows = np.stack([units, H + units, 2 * H + units, 3 * H + units], 1).reshape(-1)
        w, b, at = W.numpy(), bias.numpy(), 0
        for l in range(L):
            w_ih, w_hh = load(f'w_ih_{l}'), load(f'w_hh_{l}')
            K, N = w_ih.shape[1] + H, 4 * h
            tiles = w[at:at + N * K].reshape(N // tile, K // tile, tile, tile)
            for c in range(0, N // tile, CHUNK):
                picked = rows[c * tile:(c + CHUNK) * tile]
                block = np.concatenate([w_ih[picked], w_hh[picked]], 1)
                tiles[c:c + CHUNK] = block.reshape(-1, tile, K // tile, tile).transpose(0, 2, 3, 1)
            b[l * N:(l + 1) * N] = load(f'b_{l}')[rows]
            at += N * K
        held.append((W, bias))
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
    for r in range(1, size):
        for part in held[r]:
            dist.send(part, dst=r)
    W.copy_(held[0][0])
    bias.copy_(held[0][1])
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
    held = prepared(config, chain.capability(config['capabilities'])['simdgroup_matrix_edge']) if os.environ['RANK'] == '0' else None
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
