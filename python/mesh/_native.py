"""The partial-tensor library's binding to the mesh's own (design/algorithm-sources.md#program): operands are torch
tensors, the window's where they cross (torch_mesh.backend.blank over mesh.lease: rdma/NCCL.md "Memory"); every
crossing and collective is libnccl-mesh's, through torch.distributed's "mesh" backend (rdma/torch-mesh/README.md):
host-path calls complete where they return, MPS ones are encoded on torch's stream and waited where their consumer
runs; contractions are torch_mesh.engines' Linear over a program's Engines operand, else torch's own product."""
import math
import os

import numpy as np
import torch
import torch.distributed as dist

_TORCH = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16,
          'int8': torch.int8, 'uint8': torch.uint8, 'int32': torch.int32, 'uint32': torch.uint32, 'int64': torch.int64,
          'uint64': torch.uint64, 'bool': torch.bool}
_NUMPY = {value: np.dtype(name) for name, value in _TORCH.items() if name != 'bfloat16'}
DEVICES = {'cpu': 'cpu', 'metal': 'mps', 'mps': 'mps'}


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


# design/algorithm-sources.md#programtensor
def floating(value):
    return value.is_floating_point if isinstance(value, torch.dtype) else value.dtype.is_floating_point


# design/algorithm-sources.md#program
class World:
    """This process's rank among the program's: `group` (a torch.distributed group of the mesh backend), else the
    world torch already holds, else the one torchrun's variables name (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT;
    ncclCommInitRank through torch_mesh), else a lone rank with no communicator.  Each queue is a communicator of its
    own and each set of members a collective spans another, made in one order on every member at realize."""

    def __init__(self, device, group):
        self.owned = False
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

    # design/algorithm-sources.md#independent-native-queues
    def prepare(self, keys):
        for queue, members in sorted(set(keys)):
            if (queue, members) in self.groups:
                continue
            if queue == 0 and len(members) == self.size:
                self.groups[(queue, members)] = self.group
            elif self.rank in members:
                ranks = [dist.get_global_rank(self.group, r) for r in members]
                self.groups[(queue, members)] = dist.new_group(ranks=ranks, use_local_synchronization=True)
            else:
                self.groups[(queue, members)] = None

    # design/algorithm-sources.md#independent-native-queues
    def of(self, queue, members):
        return self.groups[(queue, tuple(sorted(members)))]

    # design/algorithm-sources.md#program
    def close(self):
        for group in set(g for g in self.groups.values() if g is not None and g is not self.group):
            dist.destroy_process_group(group)
        self.groups.clear()
        if self.owned:
            dist.destroy_process_group()
            self.owned = False


# design/algorithm-sources.md#programtensor
def allocate(numel, scalar, device, window):
    if window:
        from torch_mesh import backend
        return backend.blank((max(1, numel),), scalar, device)
    return torch.empty(max(1, numel), dtype=scalar, device=device)


# design/algorithm-sources.md#programcopy
def send(tensor, peer, group):
    return dist.isend(tensor, group=group, group_dst=peer)


# design/algorithm-sources.md#programcopy
def recv(tensor, peer, group):
    return dist.irecv(tensor, group=group, group_src=peer)


# design/algorithm-sources.md#collective-movement
def broadcast(tensor, root, group):
    return dist.broadcast(tensor, group=group, group_src=root, async_op=True)


# design/algorithm-sources.md#collectivereduce_scatter
def reduce(tensor, root, op, group):
    return dist.reduce(tensor, group=group, group_dst=root, op=op, async_op=True)


# design/algorithm-sources.md#collectivereduce_scatter
def all_reduce(tensor, op, group):
    return dist.all_reduce(tensor, op=op, group=group, async_op=True)


# design/algorithm-sources.md#kernelsdot
def contract(engines, operations, left, right, key=None):
    if left.dtype != right.dtype or not left.dtype.is_floating_point:
        left, right = left.float(), right.float()
    if engines is None or key is None:
        return torch.mm(left, right)
    operation = operations.get(key)
    if operation is None:
        from torch_mesh import engines as shared
        operation = operations[key] = shared.Linear(right.T, engines=engines, name=f'mesh{key}',
                                                    grain=math.gcd(right.shape[1], 64))
    return operation(left)
