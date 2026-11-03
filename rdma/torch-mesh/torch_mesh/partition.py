"""A mesh dimension's work split over its ranks: attach(mesh, tp=units) the units, split(mesh, 'tp', N) each rank's
block of an extent N, distribute_tensor(t, mesh, [Shard(d)]) this rank's block of a tensor every rank holds (no
communication), and allgatherv_out, the all-gather a stepper declares (torch_mesh/compiler.py lowers it; it has no
eager form)."""
import torch
from torch.distributed.tensor import Shard

_PARTS = {}


def attach(mesh, **dims):
    """each named mesh dimension's units over its ranks in rank order: a count split evenly (the first units mod n one
    more), or a list, each rank's own (a balancer's partition, the same list on every rank)"""
    for name, units in dims.items():
        n = mesh.size(mesh.mesh_dim_names.index(name))
        if isinstance(units, (list, tuple)):
            if len(units) != n or min(units) < 0 or sum(units) <= 0:
                raise ValueError(f'mesh: partition {name}: {list(units)} over a mesh dimension of {n} ranks')
            _PARTS[id(mesh), name] = [int(u) for u in units]
            continue
        if n <= 0 or units <= 0:
            raise ValueError(f'mesh: partition {name}: {units} units over a mesh dimension of {n} ranks')
        _PARTS[id(mesh), name] = [units // n + (r < units % n) for r in range(n)]


def split(mesh, name, N):
    """each rank's block of an extent N along mesh dimension `name`, in proportion to its units, in rank order"""
    parts = _PARTS[id(mesh), name]
    total, cuts, at = sum(parts), [0], 0
    for p in parts:
        at += p
        cuts.append(N * at // total)
    return [b - a for a, b in zip(cuts, cuts[1:])]


class Block:
    """this rank's block of a distributed tensor: to_local() its tensor; shape the whole tensor's"""
    def __init__(self, local, shape):
        self._local, self.shape = local, torch.Size(shape)
        self.dtype, self.device = local.dtype, local.device

    def to_local(self):
        return self._local


def distribute_tensor(tensor, device_mesh, placements):
    """this rank's block of `tensor`, which every rank holds: a Shard(d) on an attached dimension its split block"""
    local = tensor
    names = device_mesh.mesh_dim_names or ()
    for i, p in enumerate(placements):
        if isinstance(p, Shard):
            blocks = split(device_mesh, names[i], tensor.shape[p.dim])
            rank = device_mesh.get_local_rank(i)
            local = local.narrow(p.dim, sum(blocks[:rank]), blocks[rank])
    return Block(local, tensor.shape)


@torch.library.custom_op('torch_mesh::allgatherv_out', mutates_args=('out',))
def allgatherv_out(x: torch.Tensor, out: torch.Tensor, dim: int, sizes: list[int], group_name: str) -> None:
    """every rank's `x` gathered along `dim` into `out` (each rank's block `sizes[r]`): declared in a stepper"""
    raise RuntimeError('an all-gather is declared in a stepper and runs compiled (torch_mesh/compiler.py)')


@allgatherv_out.register_fake
def _(x, out, dim, sizes, group_name):
    return None


@torch.library.custom_op('torch_mesh::allreduce_', mutates_args=('x',))
def allreduce_(x: torch.Tensor, group_name: str) -> None:
    """`x` summed over every rank, in place (a reduce-scatter and an all-gather along the communicator's trees):
    declared in a stepper"""
    raise RuntimeError('an all-reduce is declared in a stepper and runs compiled (torch_mesh/compiler.py)')


@allreduce_.register_fake
def _(x, group_name):
    return None
