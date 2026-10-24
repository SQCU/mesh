import os

import numpy as np
import torch
import torch_mesh
from torch.distributed.tensor import Shard
from torch_mesh import partition

import chain


# docs/writing-a-service.md#the-lstm
@torch.compile(fullgraph=True)
def cell(z, c, h):
    """One layer's cell update from its gates z [rows, 4 x units] (torch's order: i, f, g and o, each over the units):
    its units' c and h written in place, one compiled kernel."""
    i, f, g, o = z.float().view(z.shape[0], 4, -1).unbind(1)
    c.copy_(torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g))
    h.copy_(torch.sigmoid(o) * torch.tanh(c))


class Tick:
    # docs/writing-a-service.md#the-lstm
    def __init__(self, mesh, weights, biases, rows, steps):
        """weights and biases: each layer's [4, H, K] and [4, H] split along the units (the backend's ragged shards),
        each rank's held from here as [4 x its units, K] and [4 x its units] in region tensors."""
        self.mesh, self.rows, self.steps, self.L = mesh, rows, steps, len(weights)
        held = lambda t: torch_mesh.empty(t.to_local().shape, dtype=t.dtype).copy_(t.to_local()).view(-1, *t.shape[2:])
        H, HR = weights[0].shape[1], weights[0].to_local().shape[1]
        self.W, self.b = [held(w) for w in weights], [held(b) for b in biases]
        self.c = torch_mesh.empty(self.L, rows, HR, dtype=torch.float32).zero_()
        self.mine = [torch_mesh.empty(self.L, rows, HR, dtype=torch.half).zero_() for _ in range(2)]
        self.h = torch.zeros(self.L, rows, H, dtype=torch.half, device='mps')
        self.tick = 0
        for l in range(self.L):
            cell(torch.zeros(rows, 4 * HR, dtype=torch.half, device='mps'), self.c[l], self.mine[0][l])

    # docs/writing-a-service.md#the-lstm
    def __call__(self, x):
        s = self.tick
        mine, last = self.mine[s % 2], self.mine[(s + 1) % 2]
        for l in range(self.L):
            if not 0 <= s - l < self.steps:
                mine[l] = last[l]
                continue
            u = torch.cat([x if l == 0 else self.h[l - 1], self.h[l]], 1)
            cell(torch.nn.functional.linear(u, self.W[l], self.b[l]), self.c[l], mine[l])
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
