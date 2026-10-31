import numpy as np
import torch
import torch_mesh
from torch_mesh.engines import host


class TensorStorage:
    """57628982^:storage.TensorStorage on torch: host arrays a frame is packed into in place (`host`), and their
    values on the program's device (`values`), which share those same pages."""

    def __init__(self, shapes=(), device='cpu'):
        self.device = device
        self.host, self.values = {}, {}
        self.allocations = self.nbytes = 0
        self.realize(dict(shapes))

    def realize(self, shapes):
        for name, (shape, dtype) in shapes.items():
            if name in self.host and self.host[name].shape == tuple(shape) and self.host[name].dtype == np.dtype(dtype):
                continue
            scalar = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            value = torch_mesh.empty(tuple(shape), dtype=scalar, device=self.device)
            self.values[name], self.host[name] = value, host(value).numpy()
            self.host[name].fill(0)
            self.allocations += 1
        self.nbytes = sum(value.nbytes for value in self.host.values())

    def load(self, values):
        for name, value in values.items():
            np.copyto(self.host[name], value)
        return {name: self.values[name] for name in values}

    def report(self):
        return {'allocations': self.allocations, 'bytes': self.nbytes, 'arrays': len(self.host)}
