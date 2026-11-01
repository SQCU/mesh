"""Policy product declarations and mesh membership.

Administrative messages carry call names and operand schemas. Numerical
operands, products, gradients, and replication enter the same prepared Function.
Configured product shares select local matrix views at preparation; autograd
records their collective derivatives in that function's graph.
"""
import os
import time

import numpy as np
import torch
import torch.distributed._functional_collectives as collective
import torch_mesh
from torch_mesh.engines import host


def split(total, parts):
    """Each rank's [lo, hi) of `total` units given relative `parts`: largest remainder, every rank alike."""
    weights = np.asarray(parts, np.float64)
    if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f'parts {list(parts)}: one nonnegative part a rank, not all zero')
    raw = total * weights / weights.sum()
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts), kind='stable')[:total - int(counts.sum())]:
        counts[index] += 1
    bounds = np.concatenate(([0], np.cumsum(counts)))
    return [(int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:])]


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, world):
        ctx.world = world
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.world.allreduce(gradient), None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, world):
        return world.allreduce(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class World:
    """This process's place in the learner's world.  One process (WORLD_SIZE unset or 1) is a world of one, whose
    products are whole and whose calls are announced to no one."""

    def __init__(self, device, products=None):
        self.device = device
        self.products = {name: list(parts) for name, parts in (products or {}).items()}
        size = int(os.environ.get('WORLD_SIZE', '1'))
        self.dist, self.group, self.native = None, None, None
        self.rank, self.size = 0, 1
        if size > 1:
            import torch.distributed as dist
            import torch_mesh
            self.dist, self.torch_mesh = dist, torch_mesh
            dist.init_process_group(backend='mesh', **({'device_id': torch.device('mps', 0)} if device == 'mps' else {}))
            self.rank, self.size = dist.get_rank(), dist.get_world_size()
            from torch_mesh import backend
            self.group = backend._mesh(dist.distributed_c10d._get_default_group().group_name, 'policy function')
            self.native = self.group.comm
        for name, parts in self.products.items():
            if len(parts) != self.size:
                raise ValueError(f'products.{name} {parts}: one part a rank of {self.size}')
        self.calls = 0
        self.inputs = {}

    def buffer(self, shape, dtype, device):
        return torch_mesh.empty(tuple(shape), dtype=dtype, device=device)

    def parts(self, name):
        return self.products.get(name) if self.size > 1 else None

    def span(self, name, total):
        parts = self.parts(name)
        return (0, total) if parts is None else split(total, parts)[self.rank]

    def allreduce(self, value, op='sum'):
        return collective.wait_tensor(collective.all_reduce(value, op, self.group.group_name)) if self.group else value

    def copy(self, value):
        return _Copy.apply(value, self) if self.size > 1 else value

    def reduce(self, value):
        return _Reduce.apply(value, self) if self.size > 1 else value

    def reduce_norm(self, op, value):
        return self.allreduce(value, op)

    def broadcast_(self, tensors):
        """Source declaration: replicated values written by the compiled graph."""
        if self.group is not None:
            for tensor in tensors:
                tensor.copy_(collective.wait_tensor(collective.broadcast(tensor, 0, self.group.group_name)))

    def _encode(self, value):
        if isinstance(value, np.generic):
            return {'value': value.item()}
        if value is None or isinstance(value, (bool, int, float, str)):
            return {'value': value}
        if isinstance(value, np.ndarray):
            return {'numpy': value.dtype.str, 'shape': list(value.shape)}
        if isinstance(value, torch.Tensor):
            return {'torch': str(value.dtype).removeprefix('torch.'), 'shape': list(value.shape), 'device': value.device.type}
        if isinstance(value, dict):
            return {'dict': {key: self._encode(part) for key, part in value.items()}}
        if isinstance(value, (tuple, list)):
            return {'list': [self._encode(part) for part in value]}
        raise TypeError(f'undeclared call operand {type(value).__name__}')

    def _decode(self, entry, path=()):
        if 'value' in entry:
            return entry['value']
        if 'dict' in entry:
            return {key: self._decode(value, (*path, key)) for key, value in entry['dict'].items()}
        if 'list' in entry:
            return [self._decode(value, (*path, index)) for index, value in enumerate(entry['list'])]
        descriptor = (entry.get('numpy'), entry.get('torch'), tuple(entry['shape']), entry.get('device'))
        previous = self.inputs.get(path)
        if previous is None or previous[0] != descriptor:
            if 'numpy' in entry:
                value = np.zeros(entry['shape'], dtype=entry['numpy'])
            else:
                value = self.buffer(entry['shape'], getattr(torch, entry['torch']), entry['device'])
                host(value).numpy().fill(0)
            self.inputs[path] = descriptor, value
        return self.inputs[path][1]

    def materialize(self, operands):
        """Initialization's explicit state import, through an ordinary Function."""
        from torch.utils._pytree import tree_flatten, tree_unflatten
        from torch_mesh.declarations import Function
        from torch_mesh import _stream
        from prepared import pages
        leaves, spec = tree_flatten(operands)
        slots, tensors = [], []
        for i, value in enumerate(leaves):
            if isinstance(value, (np.ndarray, torch.Tensor)):
                slots.append(i)
                tensors.append(pages(value if isinstance(value, np.ndarray) else host(value).numpy(),
                                     device='cpu' if isinstance(value, np.ndarray) else value.device.type))
        def source(*values):
            return tuple(collective.wait_tensor(collective.broadcast(value, 0, self.group.group_name)) for value in values)
        if tensors:
            function = Function(source, *tensors, source_scope={'function': 'policy-state-import'})
            destinations = [self.buffer(value.shape, value.dtype, value.device.type) for value in tensors]
            self.group.plan(function.declare(self.group, destinations=destinations))
            function.prepare(); function.run()
            completion = function.completion()
            while _stream.completion_observe(completion) is None:
                time.sleep(0)
            for i, value in zip(slots, destinations, strict=True):
                leaves[i] = host(value).numpy() if isinstance(leaves[i], np.ndarray) else value
        return tree_unflatten(leaves, spec)

    def announce(self, name, *operands, materialize=False):
        """Administrative kind/type metadata; numerical values enter the Function."""
        if self.size == 1 or self.rank:
            return operands
        import mesh
        mesh.broadcast_configuration(self.native, {'name': name, 'operands': self._encode(list(operands)), 'materialize': materialize})
        self.calls += 1
        return self.materialize(list(operands)) if materialize else operands

    def receive(self):
        import mesh
        message = mesh.broadcast_configuration(self.native, None)
        if message is None:
            return None
        operands = self._decode(message['operands'], (message['name'],))
        if message['materialize']:
            operands = self.materialize(operands)
        self.calls += 1
        return message['name'], operands

    def stop(self):
        if self.size > 1 and self.rank == 0:
            import mesh
            mesh.broadcast_configuration(self.native, None)

    def close(self):
        if self.dist is not None:
            self.dist.destroy_process_group()
            self.dist = None

    def report(self):
        return {'rank': self.rank, 'size': self.size, 'products': self.products, 'calls': self.calls}
