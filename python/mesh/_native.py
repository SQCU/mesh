"""Bindings from partial declarations to the common native transport.

Preparation fixes operand views, collective metadata and physical write plans.
The compiled graph enqueues them on the numerical stream, including the native
consumer dependencies. No Work objects or host waits are used by this frontend.
"""
import os

import numpy as np
import torch
import torch.distributed as dist

_TORCH = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16,
          'int8': torch.int8, 'uint8': torch.uint8, 'int32': torch.int32, 'uint32': torch.uint32, 'int64': torch.int64,
          'uint64': torch.uint64, 'bool': torch.bool}
_NUMPY = {value: np.dtype(name) for name, value in _TORCH.items() if name != 'bfloat16'}
DEVICES = {'cpu': 'cpu', 'metal': 'mps', 'mps': 'mps'}
OPS = {'sum': dist.ReduceOp.SUM, 'max': dist.ReduceOp.MAX, 'min': dist.ReduceOp.MIN}


# design/algorithm-sources.md#programtensor
def dtype(value):
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str) and value in _TORCH:
        return _TORCH[value]
    return _TORCH[np.dtype(value).name]


# design/algorithm-sources.md#programtensor
def numpy_dtype(value):
    return _NUMPY[value]


# design/algorithm-sources.md#program
class World:
    """This process's rank among the program's: `group` (a torch.distributed group of the mesh backend), else the
    world torch already holds, else the one torchrun's variables name (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT;
    ncclCommInitRank through torch_mesh), else a lone rank with no communicator. Logical participant sets and queues
    retain the same full physical communicator, so every machine remains available for forwarding."""

    def __init__(self, device, group):
        self.device, self.owned = device, False
        if group is None and dist.is_available() and dist.is_initialized():
            group = dist.group.WORLD
        if group is None and int(os.environ.get('WORLD_SIZE', '1')) > 1:
            from torch_mesh import backend  # noqa: F401
            dist.init_process_group(backend='mesh', **({'device_id': torch.device('mps', 0)} if device == 'mps' else {}))
            group, self.owned = dist.group.WORLD, True
        self.group = group
        self.rank = dist.get_rank(group) if group is not None else 0
        self.size = dist.get_world_size(group) if group is not None else 1
        self.groups = {}

    def union_cuts(self, task, cuts):
        """Agree logical alias-fragment boundaries once during realization."""
        cuts = sorted(set(cuts))
        if self.group is None or self.size == 1:
            return cuts
        return self.group._get_backend(torch.device(self.device)).union_cuts(cuts)

    # design/algorithm-sources.md#independent-native-queues
    def prepare(self, keys):
        for queue, members in sorted(set(keys)):
            self.groups[(queue, tuple(members))] = self.group

    # design/algorithm-sources.md#independent-native-queues
    def of(self, queue, members):
        return self.groups[(queue, tuple(sorted(members)))]

    def declare(self, tasks):
        """Realize the common physical communicator's complete ordered graph."""
        calls = [task.prepared[1] for task in tasks if task.crossing and self.rank in task.members]
        if self.group is not None:
            from torch_mesh import plan
            plan(self.group, calls)

    # design/algorithm-sources.md#program
    def close(self):
        for group in set(g for g in self.groups.values() if g is not None and g is not self.group):
            dist.destroy_process_group(group)
        self.groups.clear()
        if self.owned:
            dist.destroy_process_group()
            self.owned = False



def call(kind, count, scalar, *, counts=None, op='sum', root=0, contributors=None, matrix=None, recipients=None,
         destination=0):
    """One declaration in the common Torch/native collective metadata type."""
    from torch_mesh import backend
    return backend.Call(getattr(backend, kind), count, scalar, root=root,
                        op=OPS[op] if isinstance(op, str) else op, counts=counts,
                        contributors=contributors, matrix=matrix, recipients=recipients, destination=destination)


def prepare_writes(deliveries, group):
    """Use the production backend's one exact indexed WRITE preparation."""
    return group._get_backend(deliveries[0][0].device).prepare_writes(deliveries)


# design/algorithm-sources.md#programtensor
def allocate(numel, scalar, device, in_region):
    if in_region:
        from torch_mesh import empty
        return empty(max(1, numel), dtype=scalar, device=device)
    return torch.empty(max(1, numel), dtype=scalar, device=device)


def prepare_fragments(bindings, group):
    from torch_mesh.backend import prepare_fragments
    return prepare_fragments(bindings, group)
