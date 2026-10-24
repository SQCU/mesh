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
        """Every state unit-major, [units, L, rows], so a tick's gather crosses its outermost dimension."""
        self.mesh, self.rows, self.steps, self.L = mesh, rows, steps, len(weights)
        self.W = [partition.from_local(w, mesh, [Shard(0)]) for w in weights]
        self.b = [partition.from_local(b, mesh, [Shard(0)]) for b in biases]
        H, HR = self.W[0].shape[0] // 4, biases[0].numel() // 4
        self.c = torch_mesh.empty(HR, self.L, rows, dtype=torch.float32).zero_()
        self.mine = [torch_mesh.empty(HR, self.L, rows, dtype=torch.half).zero_() for _ in range(2)]
        self.h = torch.zeros(H, self.L, rows, dtype=torch.half, device='mps')
        self.tick = 0

    # docs/writing-a-service.md#the-lstm
    def __call__(self, x):
        """x [rows, input]: the tick's input; returns the top layer's h [H, rows] of the step L - 1 ticks before."""
        s, L = self.tick, self.L
        mine, last = self.mine[s % 2], self.mine[(s + 1) % 2]
        for l in range(L):
            if not 0 <= s - l < self.steps:
                mine[:, l] = last[:, l]
                continue
            u = partition.from_local(torch.cat([x.T if l == 0 else self.h[:, l - 1], self.h[:, l]]), self.mesh, [Replicate()])
            z = torch.mm(self.W[l], u).to_local() + self.b[l].to_local().view(-1, 1)
            i, f, g, o = z.float().view(-1, 4, self.rows).unbind(1)
            self.c[:, l] = torch.sigmoid(f) * self.c[:, l] + torch.sigmoid(i) * torch.tanh(g)
            mine[:, l] = torch.sigmoid(o) * torch.tanh(self.c[:, l])
        self.h = partition.from_local(mine, self.mesh, [Shard(0)]).full_tensor()
        self.tick = s + 1
        return self.h[:, L - 1]

    # docs/writing-a-service.md#the-lstm
    def follow(self):
        split = [partition.RaggedShard(0, self.W[0].placements[0].key)]
        self.W, self.b = [partition.follow(w) for w in self.W], [partition.follow(b) for b in self.b]
        held = lambda t: partition.follow(partition.from_local(t, self.mesh, split)).to_local()
        self.c, self.mine = held(self.c), [held(m) for m in self.mine]


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
