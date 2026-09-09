import mlx.core as mx
import numpy as np


class TensorStorage:
    def __init__(self, shapes=()):
        self.values, self.host = {}, {}
        self.allocations = self.nbytes = 0
        self.realize(dict(shapes))

    def realize(self, shapes):
        changed = {name: mx.array(np.zeros(shape, dtype=dtype))
            for name, (shape, dtype) in shapes.items()
            if name not in self.host or self.host[name].shape != tuple(shape)
            or self.host[name].dtype != np.dtype(dtype)}
        mx.eval(changed)
        self.values.update(changed)
        self.host.update({name: np.array(value, copy=False) for name, value in changed.items()})
        self.allocations += len(changed)
        self.nbytes = sum(value.nbytes for value in self.host.values())

    def load(self, values):
        for name, value in values.items():
            np.copyto(self.host[name], value)
        return {name: self.values[name] for name in values}

    def report(self):
        return {'allocations': self.allocations, 'bytes': self.nbytes, 'arrays': len(self.values)}
