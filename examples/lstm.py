import os

import numpy as np
import torch
import torch_mesh
from torch.distributed.tensor import Replicate, Shard
from torch_mesh import partition

import chain


# docs/writing-a-service.md#the-lstm
def gate_rows(blocks, k):
    """Block k's rows of a layer's [4H, K] gate weights, H the blocks' sum, unit-major: (i, f, g, o) of each of its
    units in turn."""
    H, start = sum(blocks), sum(blocks[:k])
    units = np.arange(start, start + blocks[k])
    return np.stack([units, H + units, 2 * H + units, 3 * H + units], 1).reshape(-1)


class Tick:
    # docs/writing-a-service.md#the-lstm
    def __init__(self, mesh, weights, biases, rows, steps):
        self.mesh, self.rows, self.steps, self.L = mesh, rows, steps, len(weights)
        self.W = [partition.from_local(w, mesh, [Shard(0)]) for w in weights]
        self.b = [partition.from_local(b, mesh, [Shard(0)]) for b in biases]
        H, HR = self.W[0].shape[0] // 4, biases[0].numel() // 4
        self.c = torch_mesh.empty(self.L, rows, HR, dtype=torch.float32).zero_()
        self.mine = [torch_mesh.empty(self.L, rows, HR, dtype=torch.half).zero_() for _ in range(2)]
        self.h = torch.zeros(self.L, rows, H, dtype=torch.half, device='mps')
        self.tick = 0

    # docs/writing-a-service.md#the-lstm
    def __call__(self, x):
        s = self.tick
        mine, last = self.mine[s % 2], self.mine[(s + 1) % 2]
        for l in range(self.L):
            if not 0 <= s - l < self.steps:
                mine[l] = last[l]
                continue
            u = partition.from_local(torch.cat([x if l == 0 else self.h[l - 1], self.h[l]], 1), self.mesh, [Replicate()])
            i, f, g, o = torch.nn.functional.linear(u, self.W[l], self.b[l]).to_local().float().view(self.rows, -1, 4).unbind(-1)
            self.c[l] = torch.sigmoid(f) * self.c[l] + torch.sigmoid(i) * torch.tanh(g)
            mine[l] = torch.sigmoid(o) * torch.tanh(self.c[l])
        self.h = partition.from_local(mine, self.mesh, [Shard(2)]).full_tensor()
        self.tick = s + 1
        return self.h[self.L - 1]


# docs/writing-a-service.md#the-lstm
def inputs(config, steps, rows):
    g, H, KIN = np.random.default_rng(0), config['width'], config['input_width']
    k, part, scratch = 1 / np.sqrt(H), config['weights'] + '.part', chain.region(chain.CHUNK, torch.float64)
    normal = lambda b: g.standard_normal(out=b)
    uniform = lambda b: np.add(np.multiply(g.random(out=b), 2 * k, out=b), -k, out=b)
    np.save(config['input'], chain._drawn(chain.region((steps, rows, KIN), torch.half), normal, scratch))
    os.makedirs(part, exist_ok=True)
    for l in range(config['layers']):
        for name, columns in ((f'w_ih_{l}', KIN if l == 0 else H), (f'w_hh_{l}', H)):
            np.save(f'{part}/{name}.npy', chain._drawn(chain.region((4 * H, columns), torch.half), uniform, scratch))
        np.save(f'{part}/b_{l}.npy', (g.uniform(-k, k, 4 * H) + g.uniform(-k, k, 4 * H)).astype(np.float16))
    os.replace(part, config['weights'])
