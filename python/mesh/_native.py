"""The partial-tensor library's binding to the mesh's own (design/algorithm-sources.md#program): operands are torch
tensors, the region's where they cross (torch_mesh.empty: rdma/NCCL.md "Memory"); every
crossing and collective is libnccl-mesh's, through torch.distributed's "mesh" backend (rdma/torch-mesh/README.md):
exact indexed WRITE plans and native reductions, including the split-KV partials merge
(ncclMeshRedOpCreatePartials's operator). Host-path calls
complete where they return, MPS ones are encoded on torch's stream and waited where their consumer runs.  It computes
nothing: a program's numerics are its supplied kernels'."""
import os
import json

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

    serial = 0

    def __init__(self, device, group):
        World.serial += 1
        self.serial = World.serial
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
        store = dist.distributed_c10d._get_default_store()
        key = f'mesh/program/{self.group.group_name}/{self.serial}/cuts/{task}'
        store.set(f'{key}/{self.rank}', json.dumps(cuts).encode())
        rows = [json.loads(store.get(f'{key}/{rank}')) for rank in range(self.size)]
        if any(row[0] != 0 or row[-1] != cuts[-1] for row in rows):
            raise ValueError('Program collective operands have different logical extents across ranks')
        return sorted(set(cut for row in rows for cut in row))

    # design/algorithm-sources.md#independent-native-queues
    def prepare(self, keys):
        for queue, members in sorted(set(keys)):
            self.groups[(queue, tuple(members))] = self.group

    # design/algorithm-sources.md#independent-native-queues
    def of(self, queue, members):
        return self.groups[(queue, tuple(sorted(members)))]

    def declare(self, tasks):
        """Realize the common physical communicator's complete ordered graph."""
        calls = []
        for task in tasks:
            declaration = getattr(task.run[0], 'declaration', None) if task.crossing else None
            if declaration is not None:
                if self.rank in task.members:
                    calls.extend(declaration(task))
        if self.group is not None:
            from torch_mesh import plan
            plan(self.group, calls)

    # design/algorithm-sources.md#mesherror
    def revoke(self):
        """Every communicator's work in flight here ends with an error, and none is issued after it (torch_mesh
        ProcessGroupMesh.revoke: ncclCommRevoke); its peers' waits end at their own revoke."""
        groups = {g for g in self.groups.values() if g is not None} | ({self.group} if self.group is not None else set())
        for group in groups:
            group._get_backend(torch.device(self.device)).revoke()

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


def fragments(bindings, group):
    """One group of canonical native collectives over retained alias slices."""
    from torch_mesh import backend
    calls, outputs = [], []
    for declaration, source, destination in bindings:
        code = declaration.op if isinstance(declaration.op, int) else backend.op(declaration.op)
        calls.append((declaration.kind, [source], [destination], declaration.counts or [], code, declaration.root))
        if destination is not None:
            outputs.append(destination)
    device = next(t.device for _, source, destination in bindings for t in (source, destination) if t is not None)
    return group._get_backend(device)._issue(calls, outputs, True)
