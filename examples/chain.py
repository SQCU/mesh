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
