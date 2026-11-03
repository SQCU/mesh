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
        # stored units first (H, L, rows permuted): each rank's block of every layer and row is one contiguous run, so a
        # tick's gather is one call, one unit a peer (torch_mesh/compiler.py, an outermost gathered axis); row-major it
        # was a call a (layer, row) frame, 16 SENDs a peer a tick at two layers and 8 rows (docs/facts.md, run-20)
        self.h = torch_mesh.empty(H, self.L, rows, dtype=torch.half).permute(1, 2, 0)
        first = sum(self.sizes[:mesh.get_group('tp').rank()])
        self.mine = self.h.narrow(2, first, HR)
        self.reset()

    # docs/writing-a-service.md#the-lstm
    def reset(self):
        """Every layer's state h(-1) and c(-1): zeros."""
        for state in (self.c, self.h):
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
    def starting(self, s, x, gather=True):
        """Tick s < L - 1, while the layers above s have not begun (their h stays h(-1)); without the gather (`gather`
        False) a rank's compute alone (balance-service's observation)."""
        self.layers(s + 1, x)
        if gather:
            self.gather()

    # docs/writing-a-service.md#the-lstm
    def stepper(self, X, O, steps=1, gather=True):
        """A fixed number of later ticks for AOT declaration: the input at the step counter t held on
        the GPU (the last one past the input's end), every layer, the gather, the top layer's h of step t - L + 1 into
        the output O (where this rank holds it), t advanced."""
        def chunk(t):
            for _ in range(steps):
                self.layers(self.L, X.index_select(0, t.clamp(max=self.steps - 1))[0])
                if gather:
                    self.gather()
                if O is not None:
                    O.index_copy_(0, t - (self.L - 1), self.h[self.L - 1][None])
                t.add_(1)
        return chunk


# docs/writing-a-service.md#the-lstm
def reference(config, x, values, steps):
    """The stepper's outputs computed on the CPU in float32 from the same inputs and weights: the wavefront Tick runs
    (each tick every started layer's gates from the tick before's h, layer l > 0 reading layer l - 1's previous h; ticks
    0 to L - 2 start the layers one by one; tick t reads input min(t, steps - 1)), its gates and h rounded to fp16 as
    the GPU stores them; output step s the top layer's h after tick s + L - 1."""
    L, H = config['layers'], config['width']
    rows = x.shape[1]
    W = [np.concatenate([values[f'w_ih_{l}'], values[f'w_hh_{l}']], 1).astype(np.float32) for l in range(L)]
    b = [values[f'b_{l}'].astype(np.float32) for l in range(L)]
    h, c = np.zeros((L, rows, H), np.float32), np.zeros((L, rows, H), np.float32)
    out = np.zeros((steps, rows, H), np.float32)
    half = lambda v: v.astype(np.float16).astype(np.float32)
    sigmoid = lambda v: 1 / (1 + np.exp(-v))
    for t in range(steps + L - 1):
        n = min(t + 1, L)
        step = half(np.asarray(x[min(t, steps - 1)]))
        z = [half(np.concatenate([step if l == 0 else h[l - 1], h[l]], 1) @ W[l].T + b[l]) for l in range(n)]
        for l in range(n):
            i, f, g, o = np.split(z[l], 4, axis=1)
            c[l] = sigmoid(f) * c[l] + sigmoid(i) * np.tanh(g)
            h[l] = half(sigmoid(o) * np.tanh(c[l]))
        if t >= L - 1:
            out[t - (L - 1)] = h[L - 1]
    return out


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
