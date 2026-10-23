import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import init_device_mesh
from torch_mesh import partition

import chain
import lstm

CHUNK = 512


# docs/writing-a-service.md#the-lstm
def weights(config, blocks, rank, size):
    """Each layer's gate rows of this rank's block of units, [4 H_r, K] and [4 H_r], in MPS region tensors: rank 0
    reads the files memory-mapped and sends each rank its rows."""
    L, KIN, H, folder = config['layers'], config['input_width'], sum(blocks), Path(config['weights'])
    load = lambda name: np.load(folder / f'{name}.npy', mmap_mode='r')
    made = []
    for l in range(L):
        K = (KIN if l == 0 else H) + H
        W = torch_mesh.empty(4 * blocks[rank], K, dtype=torch.half)
        b = torch_mesh.empty(4 * blocks[rank], dtype=torch.half)
        if rank:
            dist.recv(W, src=0)
            dist.recv(b, src=0)
        else:
            w_ih, w_hh, bias = load(f'w_ih_{l}'), load(f'w_hh_{l}'), load(f'b_{l}')
            for r in reversed(range(size)):
                rows = lstm.gate_rows(blocks, r)
                Wr = torch_mesh.empty(len(rows), K, dtype=torch.half, device='cpu')
                for c in range(0, len(rows), CHUNK):
                    picked = rows[c:c + CHUNK]
                    Wr[c:c + CHUNK] = torch.from_numpy(np.concatenate([w_ih[picked], w_hh[picked]], 1))
                br = torch_mesh.empty(len(rows), dtype=torch.half, device='cpu').copy_(torch.from_numpy(bias[rows].copy()))
                if r:
                    dist.send(Wr, dst=r)
                    dist.send(br, dst=r)
                else:
                    W.copy_(Wr)
                    b.copy_(br)
        made.append((W, b))
    return [w for w, _ in made], [b for _, b in made]


# docs/writing-a-service.md#the-lstm
def main():
    config = json.loads(open(sys.argv[1]).read())
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    L, KIN, H = config['layers'], config['input_width'], config['width']
    mesh = init_device_mesh('mps', (size,), mesh_dim_names=('tp',))
    even = [H // size + (r < H % size) for r in range(size)]
    partition.attach(mesh, tp=(even, even))
    blocks = partition.split(mesh, 'tp', H)
    steps, rows, inputs = chain.operand(config['input'], rank, KIN)
    W, b = weights(config, blocks, rank, size)
    tick = lstm.Tick(mesh, W, b, rows, steps)
    X = torch_mesh.empty(steps, rows, KIN, dtype=torch.half)
    if rank == 0:
        X.copy_(torch.from_numpy(inputs[:]))
    O = torch.empty(steps, rows, H, dtype=torch.half, device='mps') if rank == 0 else None
    torch_mesh.check_memory()
    torch.mps.synchronize()
    dist.barrier()
    ticks, half = steps + L - 1, (steps + L - 1) // 2
    middle, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    began = time.perf_counter()
    dist.broadcast(X, src=0)
    for s in range(ticks):
        top = tick(X[min(s, steps - 1)])
        if O is not None and s >= L - 1:
            O[s - L + 1].copy_(top)
        if s == half:
            middle.record()
    end.record()
    hosted = time.perf_counter() - began
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    steady = middle.elapsed_time(end) / 1e3 / (ticks - 1 - half)
    if O is not None:
        np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, H))[:] = O.cpu().numpy()
    summary = dict(rank=rank, size=size, blocks=blocks, layers=L, rows=rows, input_width=KIN, width=H, steps=steps,
                   ticks=ticks, seconds=seconds, aggregate_rows_per_second=steps * rows / seconds,
                   steady_aggregate_rows_per_second=rows / steady, per_stream_steps_per_second=steps / seconds,
                   tick_microseconds=seconds / ticks * 1e6, steady_tick_microseconds=steady * 1e6,
                   host_microseconds=hosted / ticks * 1e6, latency_ticks=L,
                   sent_bytes_per_tick=(size - 1) * rows * L * blocks[rank] * torch.half.itemsize)
    print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
