import numpy as np
import torch
import torch_mesh
from torch.distributed.tensor import Shard
from torch_mesh import partition

CHUNK = 1 << 24


# docs/writing-a-service.md#the-step
def weights(mesh, folds, width):
    if folds is None:
        folds = [torch_mesh.empty(1, width, dtype=torch.half).expand(width, width)] * 2
    else:
        folds = [torch_mesh.empty(width, width, dtype=torch.half).copy_(torch.from_numpy(w)) for w in folds]
    return tuple(partition.distribute_tensor(w, mesh, [Shard(0)], src_data_rank=0).to_local() for w in folds)


# docs/writing-a-service.md#the-configuration
def region(shape, dtype):
    return torch_mesh.empty(shape, dtype=dtype, device='cpu').numpy()


# docs/writing-a-service.md#the-configuration
def _drawn(out, draw, scratch):
    flat = out.reshape(len(out), -1)
    n = max(1, len(scratch) // flat.shape[1])
    for i in range(0, len(flat), n):
        part = flat[i:i + n]
        part[...] = draw(scratch[:part.size]).reshape(part.shape)
    return out


# docs/writing-a-service.md#the-configuration
def inputs(config, steps, rows):
    g, w, scratch = np.random.default_rng(config['seed']), config['width'], region(CHUNK, torch.float64)
    x = _drawn(region((steps, rows, w), torch.half), lambda b: g.standard_normal(out=b), scratch)
    scaled = lambda b: np.divide(g.standard_normal(out=b), np.sqrt(w), out=b)
    return x, tuple(_drawn(region((w, w), torch.half), scaled, scratch) for _ in range(2))


class Tick:
    # docs/writing-a-service.md#the-step
    def __init__(self, mesh, a, b, rows, width):
        """The chain's step as a torch program through the frontend the LSTM uses (design/page-maps-2026-11-02.md: the
        normal path, its values placed by the run's solve): a and b each rank's rows of the two folds (weights' Shard(0),
        [its rows, width]). A step: y = tanh((E[t] + z) a^T) over its rows, its partial p = y b over every column, every
        rank's partial gathered (partition's allgatherv) and summed into the state z = tanh(sum), the output at t."""
        group = mesh.get_group('tp')
        self.rows, self.width, self.a, self.b = rows, width, a, b
        self.group, self.sizes = group.group_name, [rows] * group.size()
        self.z = torch_mesh.empty(rows, width, dtype=torch.half)
        self.p = torch_mesh.empty(rows, width, dtype=torch.half)
        self.P = torch_mesh.empty(group.size() * rows, width, dtype=torch.half)
        self.reset()

    # docs/writing-a-service.md#the-step
    def reset(self):
        """The state before the first step: zeros."""
        self.z.zero_()

    # docs/writing-a-service.md#the-step
    def stepper(self, E, O, steps=1):
        """A fixed number of steps for AOT declaration, the input and output at the step counter t held on the GPU."""
        def chunk(t):
            for _ in range(steps):
                x = E.index_select(0, t)[0] + self.z
                self.p.copy_(torch.tanh(torch.nn.functional.linear(x, self.a)) @ self.b)
                torch.ops.torch_mesh.allgatherv_out(self.p, self.P, 0, self.sizes, self.group)
                self.z.copy_(torch.tanh(self.P.view(-1, self.rows, self.width).float().sum(0)))
                O.index_copy_(0, t, self.z[None])
                t.add_(1)
        return chunk
