import numpy as np
import torch


class TensorStorage:
    """57628982^:storage.TensorStorage on torch: host arrays a frame is packed into in place (`host`), and their
    values on the program's device (`values`, copied from the host arrays at each read)."""

    def __init__(self, shapes=(), device='cpu'):
        self.device = device
        self.host = {}
        self.allocations = self.nbytes = 0
        self.realize(dict(shapes))

    def realize(self, shapes):
        changed = {name: np.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in shapes.items()
            if name not in self.host or self.host[name].shape != tuple(shape)
            or self.host[name].dtype != np.dtype(dtype)}
        self.host.update(changed)
        self.allocations += len(changed)
        self.nbytes = sum(value.nbytes for value in self.host.values())

    @property
    def values(self):
        return {name: torch.from_numpy(value).to(self.device) for name, value in self.host.items()}

    def load(self, values):
        for name, value in values.items():
            np.copyto(self.host[name], value)
        return {name: torch.from_numpy(self.host[name]).to(self.device) for name in values}

    def report(self):
        return {'allocations': self.allocations, 'bytes': self.nbytes, 'arrays': len(self.host)}
