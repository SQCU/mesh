import numpy as np
import torch
import torch_mesh
from torch_mesh import partition

import chain


# docs/writing-a-service.md#the-lstm
def cell(z, c, h):
    """The cell update of layers from their gates z [layers, rows, 4 x units] (torch's order: i, f, g and o, each over
    the units): their units' c and h written in place."""
    i, f, g, o = z.float().unflatten(-1, (4, -1)).unbind(-2)
    c.copy_(torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g))
    h.copy_(torch.sigmoid(o) * torch.tanh(c))


class Tick:
    # docs/writing-a-service.md#the-lstm
    def __init__(self, mesh, weights, biases, rows, steps):
        """weights and biases: each layer's [4, H, K] and [4, H] split along the units (the backend's ragged shards),
        each rank's held from here as [4 x its units, K] and [4 x its units] in region tensors."""
        self.rows, self.steps, self.L = rows, steps, len(weights)
        held = lambda t: torch_mesh.empty(t.to_local().shape, dtype=t.dtype).copy_(t.to_local()).view(-1, *t.shape[2:])
        H, HR = weights[0].shape[1], weights[0].to_local().shape[1]
        self.W, self.b = [held(w) for w in weights], [held(b) for b in biases]
        self.group, self.sizes = mesh.get_group('tp').group_name, partition.split(mesh, 'tp', H)
        self.c = torch_mesh.empty(self.L, rows, HR, dtype=torch.float32)
        self.mine = torch_mesh.empty(self.L, rows, HR, dtype=torch.half)
        self.h = torch_mesh.empty(self.L, rows, H, dtype=torch.half)
        self.reset()

    # docs/writing-a-service.md#the-lstm
    def reset(self):
        """Every layer's state h(-1) and c(-1): zeros."""
        for state in (self.c, self.mine, self.h):
            state.zero_()

    # docs/writing-a-service.md#the-lstm
    def layers(self, n, x):
        """Layers 0 to n - 1 of a tick: each one's gates from its input and its own h of the tick before, then their
        cells at once."""
        z = torch.stack([torch.nn.functional.linear(torch.cat([x if l == 0 else self.h[l - 1], self.h[l]], 1), self.W[l],
                                                    self.b[l]) for l in range(n)])
        cell(z, self.c[:n], self.mine[:n])

    # docs/writing-a-service.md#the-lstm
    def gather(self):
        """Every rank's h slices of every layer, gathered along the units (partition's allgatherv), into h."""
        torch.ops.torch_mesh.allgatherv_out(self.mine, self.h, 2, self.sizes, self.group)

    # docs/writing-a-service.md#the-lstm
    def starting(self, s, x):
        """Tick s < L - 1, while the layers above s have not begun (their h stays h(-1))."""
        self.layers(s + 1, x)
        self.gather()

    # docs/writing-a-service.md#the-lstm
    def stepper(self, X, O, t):
        """Every later tick with no arguments (torch_mesh.record replays it): the input at the step counter t held on
        the GPU (the last one past the input's end), every layer, the gather, the top layer's h of step t - L + 1 into
        the output O (where this rank holds it), t advanced."""
        def one():
            self.layers(self.L, X.index_select(0, t.clamp(max=self.steps - 1))[0])
            self.gather()
            if O is not None:
                O.index_copy_(0, t - (self.L - 1), self.h[self.L - 1][None])
            t.add_(1)
        return one


# docs/writing-a-service.md#the-lstm
def inputs(config, steps, rows):
    g, H, KIN = np.random.default_rng(config['seed']), config['width'], config['input_width']
    k, scratch = 1 / np.sqrt(H), chain.region(chain.CHUNK, torch.float64)
    normal = lambda b: g.standard_normal(out=b)
    uniform = lambda b: np.add(np.multiply(g.random(out=b), 2 * k, out=b), -k, out=b)
    x = chain._drawn(chain.region((steps, rows, KIN), torch.half), normal, scratch)
    weights = {}
    for l in range(config['layers']):
        for name, columns in ((f'w_ih_{l}', KIN if l == 0 else H), (f'w_hh_{l}', H)):
            weights[name] = chain._drawn(chain.region((4 * H, columns), torch.half), uniform, scratch)
        bias = chain.region((4 * H,), torch.half)
        bias[...] = (g.uniform(-k, k, 4 * H) + g.uniform(-k, k, 4 * H)).astype(np.float16)
        weights[f'b_{l}'] = bias
    return x, weights
