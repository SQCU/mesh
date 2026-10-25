import numpy as np
import torch
import torch_mesh
from torch.distributed.tensor import Replicate, Shard
from torch_mesh import partition

CHUNK = 1 << 24


# docs/writing-a-service.md#the-step
def weights(mesh, folds, width):
    if folds is None:
        folds = [torch_mesh.empty(1, width, dtype=torch.half).expand(width, width)] * 2
    else:
        folds = [torch_mesh.empty(width, width, dtype=torch.half).copy_(torch.from_numpy(w)) for w in folds]
    a, b = (partition.distribute_tensor(w, mesh, [Shard(0)], src_data_rank=0) for w in folds)
    rows = b.to_local().shape[0]
    bt = torch_mesh.empty(width, -(-rows // 4) * 4, dtype=torch.half).zero_()
    bt[:, :rows].copy_(b.to_local().T)
    return a.T, partition.from_local(bt[:, :rows].T, mesh, [Shard(0)], shape=(width, width), stride=(1, width))


# docs/writing-a-service.md#the-step
def step(x, at, b, out):
    y = torch.tanh(partition.from_local(x, at.device_mesh, [Replicate()]) @ at)
    return torch.tanh((y @ b).redistribute(placements=[Replicate()]).to_local(), out=out)


# docs/writing-a-service.md#the-step
def stepper(E, O, o, t, at, b):
    def one():
        step(E.index_select(0, t)[0] + o, at, b, o)
        O.index_copy_(0, t, o[None])
        t.add_(1)
    return one


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
