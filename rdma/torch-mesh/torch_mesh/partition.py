"""A partition operand for DeviceMesh: DTensor's Shard, tensor parallelism and context parallelism
split each mesh dimension by it instead of equally.

Stock PyTorch splits every sharded extent into equal chunks (Shard follows torch.chunk), a uniform
prior over the ranks that nothing in a program can change.  Here a mesh dimension carries integer
parts p_r >= 1, one per coordinate, of P = sum(p) units: an extent N that P divides is split into
N p_r / P per coordinate r, in rank order; any other extent, and a dimension whose parts are all
equal, is split as stock splits it.  The grain is N / P (a head, an expert, a tile), and a program
chooses it by choosing P.  Without an operand every replaced function below calls PyTorch's own
with the same arguments and returns its result.

The operand, keyed by mesh dimension name:
  MESH_PARTITION="tp=5,11 cp=21,43"  attached when a DeviceMesh with that dimension name is made (a
                                     sub-mesh keeps its root's parts); the same on every rank.
  attach(mesh, tp=(5, 11))           the same, before the mesh is first used.
  sizes(mesh, "tp", N)               the per-coordinate sizes of N along that dimension.
The parts come from the nodes' rates (mesh rdma/allocate.py min_max; design/heterogeneity.md).

What follows the parts: DeviceMesh.size() and .shape return, for a partitioned dimension, an int
subclass carrying them (the only way the parts reach Shard's mesh-less size function); Shard's split,
local sizes and offsets, so distribute_tensor, from_local, compute_local_shape_and_global_offset
and every view's and factory's local shape; redistribution by MPI's v-collectives on torch's own
spellings, each block at its own size and never padded to the largest (Allgatherv: all_gather of
a list of per-rank sizes; Reduce_scatter: reduce_scatter of such a list; Scatterv: scatter;
Alltoallv: all_to_all_single with splits) [MPI 4.1 §6.5-6.10]; context_parallel's head-tail split
(rank r's head and tail both (S/2) p_r / P positions) and its ring (_mps.py), each source's keys and
values at their own length, and context_parallel_unshard.

Contracts: DTensor.from_local without shape= on a partitioned dimension means "split by the
parts" (global = local P / p_r, which must be exact); a tensor split any other way passes shape=
and stride=, as stock asks of uneven shards.  A program takes its own splits from
distribute_tensor(x, mesh, [Shard(d)], src_data_rank=None).to_local() or sizes(), never from
x.chunk(world) (both are stock's without an operand).  This module is imported by the backend's
first process group; a program that imports context_parallel_unshard by name imports this module
before it.  Eager mode only.

Where uneven shards cannot go, stock's own fallback, or the same error on every rank: a view keeps a
partitioned shard only where P divides the dimension it splits or the first it flattens (stock's
rule with P for the mesh size), else Replicate (reshape) or an error (view); mean and avg over a
partitioned dimension replicate it first; convolution's last-dimension sharding is not offered;
DTensor's RNG offsets step by the largest shard.  _StridedShard (FSDP2 with TP, nested views),
flattening or unflattening a partitioned dimension, flex-attention context parallelism and the
per-document and PTRR balancers raise NotImplementedError.  Left as they are: the strategy costs
(MeshTopoInfo, redistribute_cost), which choose among collectives, not splits."""
import functools
import math
import os

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, _api, _random, _redistribute, _utils
from torch.distributed.tensor._ops import _conv_ops, _math_ops, _view_ops  # noqa: F401 (registers conv)
from torch.distributed.tensor.experimental import _attention as _attention_stub
from torch.distributed.tensor.experimental import _context_parallel as _context_parallel
from torch.distributed.tensor.experimental._context_parallel import _attention as _cp, _load_balancer as _lb
from torch.distributed.tensor.placement_types import Shard, _StridedShard

aten = torch.ops.aten


class _Chunks(int):
    """A partitioned mesh dimension's size n: an int to all of PyTorch, carrying its parts."""

    def __new__(cls, parts):
        self = super().__new__(cls, len(parts))
        self.parts, self.units = parts, sum(parts)
        return self

    def __reduce__(self):
        return _Chunks, (self.parts,)


def _weighted(n, N):
    return isinstance(n, _Chunks) and N % n.units == 0


def _split(n, N):
    """(sizes, offsets) of an extent N over a mesh dimension of size n: the parts' where P divides
    N, else torch.chunk's."""
    if _weighted(n, N):
        sizes = [N // n.units * p for p in n.parts]
    else:
        step = -(-N // n)
        sizes = [max(0, min(step, N - step * r)) for r in range(n)]
    return sizes, [sum(sizes[:r]) for r in range(len(sizes))]


def sizes(mesh, name, N):
    """The per-coordinate sizes of an extent N along mesh dimension `name`."""
    return tuple(_split(mesh.size(mesh._get_mesh_dim_by_name(name)), N)[0])


def attach(mesh, **parts):
    """Partition mesh dimensions by name, before the mesh is first used: attach(mesh, tp=(5, 11))."""
    if mesh.__dict__.get('_hash'):
        raise RuntimeError('mesh: a partition is attached before the mesh is first used')
    for name in parts:
        if name not in (mesh.mesh_dim_names or ()):
            raise ValueError(f'mesh: no mesh dimension {name!r} to partition in {mesh.mesh_dim_names}')
    _attach(mesh, {**_named(mesh), **parts})


def _attach(mesh, named):
    partition = {}
    for name, parts in named.items():
        d, parts = mesh.mesh_dim_names.index(name), tuple(int(p) for p in parts)
        if len(parts) != _size(mesh, d) or min(parts) < 1:
            raise ValueError(f'mesh: partition {name}={parts}: {_size(mesh, d)} whole parts, each at least 1')
        if len(set(parts)) > 1:
            partition[d] = parts
    mesh._partition = partition
    mesh._chunks = {d: _Chunks(p) for d, p in partition.items()}


def _named(mesh):
    return {mesh.mesh_dim_names[d]: p for d, p in (mesh.__dict__.get('_partition') or {}).items()}


def _environment():
    named = {}
    for item in os.environ.get('MESH_PARTITION', '').split():
        name, _, parts = item.partition('=')
        named[name] = tuple(int(p) for p in parts.split(','))
    return named


# DeviceMesh (torch/distributed/device_mesh.py): the parts at construction, in size and shape, hash
# and equality; flattening and unflattening a partitioned dimension refused.
_init, _size, _shape = DeviceMesh.__init__, DeviceMesh.size, DeviceMesh.shape.fget
_hash_key, _eq, _flatten, _unflatten = DeviceMesh._hash_key, DeviceMesh.__eq__, DeviceMesh._create_flatten_mesh, DeviceMesh._unflatten


def _mesh_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    root = kwargs.get('_root_mesh')
    source = _named(root) if root is not None else _environment()
    named = {name: parts for name, parts in source.items() if name in (self.mesh_dim_names or ())}
    if named:
        _attach(self, named)


def _mesh_size(self, mesh_dim=None):
    chunks = self.__dict__.get('_chunks')
    if chunks and (mesh_dim is not None or len(self._layout) == 1):
        n = chunks.get((mesh_dim or 0) % len(self._layout))
        if n is not None:
            return n
    return _size(self, mesh_dim)


def _mesh_shape(self):
    chunks = self.__dict__.get('_chunks')
    shape = _shape(self)
    return tuple(chunks.get(d, n) for d, n in enumerate(shape)) if chunks else shape


def _mesh_hash_key(self):
    partition = self.__dict__.get('_partition')
    return _hash_key(self) + (tuple(sorted(partition.items())),) if partition else _hash_key(self)


def _mesh_eq(self, other):
    same = _eq(self, other)
    if same and self is not other:
        return (self.__dict__.get('_partition') or None) == (other.__dict__.get('_partition') or None)
    return same


def _mesh_flatten(self, *args, **kwargs):
    if self.__dict__.get('_partition'):
        raise NotImplementedError(f'mesh: flattening partitioned mesh dimensions {_named(self)}')
    return _flatten(self, *args, **kwargs)


def _mesh_unflatten(self, dim, *args, **kwargs):
    if (self.mesh_dim_names.index(dim) if isinstance(dim, str) else dim) in (self.__dict__.get('_partition') or {}):
        raise NotImplementedError(f'mesh: unflattening partitioned mesh dimension {dim!r}')
    return _unflatten(self, dim, *args, **kwargs)


# The v-collectives, rank order packed, the split dimension moved to the front.
def _empty(shape, like):
    return torch.empty(shape, dtype=like.dtype, device=like.device)  # an MPS window tensor (torch_mesh)


def _allgatherv(x, dim, sizes, group):
    y = x.movedim(dim, 0).contiguous()
    out = _empty((sum(sizes),) + y.shape[1:], y)
    dist.all_gather(list(out.split(sizes)), y, group=group)
    return out.movedim(0, dim).contiguous()


def _reduce_scatterv(x, dim, sizes, rank, op, group):
    y = x.movedim(dim, 0).contiguous()
    out = _empty((sizes[rank],) + y.shape[1:], y)
    dist.reduce_scatter(out, list(y.split(sizes)), op=getattr(dist.ReduceOp, op.upper()), group=group)
    return out.movedim(0, dim).contiguous()


# Shard (torch/distributed/tensor/placement_types.py).
_size_and_offset, _split_tensor, _select_split_tensor = Shard.local_shard_size_and_offset, Shard._split_tensor, Shard._select_split_tensor
_shard_tensor, _reduce_shard_tensor = Shard._shard_tensor, Shard._reduce_shard_tensor
_to_replicate_tensor, _to_new_shard_dim = Shard._to_replicate_tensor, Shard._to_new_shard_dim


@maybe_run_for_local_tensor
def _shard_size_and_offset(curr_local_size, num_chunks, rank):
    if not _weighted(num_chunks, curr_local_size):
        return _size_and_offset(curr_local_size, num_chunks, rank)
    sizes, offsets = _split(num_chunks, curr_local_size)
    return sizes[rank], offsets[rank]


def _shard_split_tensor(self, tensor, num_chunks, *, with_padding=True, contiguous=True):
    if not _weighted(num_chunks, tensor.size(self.dim)):
        return _split_tensor(self, tensor, num_chunks, with_padding=with_padding, contiguous=contiguous)
    shards = tensor.split(_split(num_chunks, tensor.size(self.dim))[0], self.dim)
    return [s.contiguous() if contiguous else s for s in shards], [0] * len(shards)


@maybe_run_for_local_tensor
def _shard_select_split_tensor(self, tensor, num_chunks, index, *, with_padding=True, contiguous=True, clone=True):
    if not _weighted(num_chunks, tensor.size(self.dim)):
        return _select_split_tensor(self, tensor, num_chunks, index, with_padding=with_padding, contiguous=contiguous,
                                    clone=clone)
    sizes, offsets = _split(num_chunks, tensor.size(self.dim))
    shard = tensor.narrow(self.dim, offsets[index], sizes[index])
    return shard.clone() if clone else shard.contiguous() if contiguous else shard


def _shard_shard_tensor(self, tensor, mesh, mesh_dim, src_data_rank=0):
    """Scatterv from src_data_rank (src_data_rank None: this rank's own split, _select_split_tensor)."""
    n, coordinate = mesh.size(mesh_dim), mesh.get_coordinate()
    if src_data_rank is None or coordinate is None or not _weighted(n, tensor.size(self.dim)):
        return _shard_tensor(self, tensor, mesh, mesh_dim, src_data_rank)
    (sizes, offsets), rank = _split(n, tensor.size(self.dim)), coordinate[mesh_dim]
    out = _empty(tensor.shape[:self.dim] + (sizes[rank],) + tensor.shape[self.dim + 1:], tensor)
    blocks = [tensor.narrow(self.dim, o, s).contiguous() for s, o in zip(sizes, offsets)] if rank == src_data_rank else None
    dist.scatter(out, blocks, group=mesh.get_group(mesh_dim), group_src=src_data_rank)
    return out


def _shard_reduce_shard_tensor(self, tensor, mesh, reduce_op, mesh_dim):
    n, coordinate = mesh.size(mesh_dim), mesh.get_coordinate()
    if coordinate is None or not _weighted(n, tensor.size(self.dim)):
        return _reduce_shard_tensor(self, tensor, mesh, reduce_op, mesh_dim)
    return _reduce_scatterv(tensor, self.dim, _split(n, tensor.size(self.dim))[0], coordinate[mesh_dim], reduce_op,
                            mesh.get_group(mesh_dim))


def _shard_to_replicate_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
    n, N = mesh.size(mesh_dim), current_logical_shape[self.dim]
    if not _weighted(n, N):
        return _to_replicate_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape)
    return _allgatherv(local_tensor, self.dim, _split(n, N)[0], mesh.get_group(mesh_dim))


def _shard_to_new_shard_dim(self, local_tensor, mesh, mesh_dim, current_logical_shape, new_shard_dim):
    """Alltoallv: to rank q this rank's part of q's new shard; from q, q's rows of this rank's."""
    n, coordinate = mesh.size(mesh_dim), mesh.get_coordinate()
    old, new = self.dim, new_shard_dim
    if coordinate is None or not (_weighted(n, current_logical_shape[old]) or _weighted(n, current_logical_shape[new])):
        return _to_new_shard_dim(self, local_tensor, mesh, mesh_dim, current_logical_shape, new_shard_dim)
    rank, had, (get, at) = coordinate[mesh_dim], _split(n, current_logical_shape[old])[0], _split(n, current_logical_shape[new])
    send = torch.cat([local_tensor.narrow(new, o, s).flatten() for s, o in zip(get, at)])
    shapes = [[had[q] if d == old else get[rank] if d == new else s for d, s in enumerate(local_tensor.shape)] for q in range(n)]
    counts = [math.prod(s) for s in shapes]
    out = _empty((sum(counts),), local_tensor)
    dist.all_to_all_single(out, send, counts, [s * local_tensor.numel() // local_tensor.size(new) for s in get],
                           group=mesh.get_group(mesh_dim))
    return torch.cat([piece.view(s) for piece, s in zip(out.split(counts), shapes)], dim=old)


def _strided(stock):
    """_StridedShard's sizes are right-to-left strided chunks, which the parts do not describe."""
    @functools.wraps(stock)
    def refuse(self, *args, **kwargs):
        if isinstance(kwargs.get('num_chunks', args[1] if len(args) > 1 else None), _Chunks):
            raise NotImplementedError('mesh: _StridedShard (FSDP2 with TP, nested views) on a partitioned mesh dimension')
        return stock(self, *args, **kwargs)
    return refuse


# from_local's inverse (torch._C._DTensor_compute_global_tensor_info, bound in _api and _utils).
_global_tensor_info = _utils.compute_global_tensor_info


def _compute_global_tensor_info(tensor, mesh, placements):
    coordinate = mesh.get_coordinate()
    if coordinate is None or not any(p.is_shard() and isinstance(mesh.size(i), _Chunks) for i, p in enumerate(placements)):
        return _global_tensor_info(tensor, mesh, placements)
    shape, stride = list(tensor.size()), list(tensor.stride())
    for i in reversed(range(len(placements))):
        p, n = placements[i], mesh.size(i)
        if isinstance(p, _StridedShard):
            raise NotImplementedError('mesh: _StridedShard with a partitioned mesh dimension')
        if not p.is_shard():
            continue
        d, local = p.dim, shape[p.dim]
        size, rest = divmod(local * n.units, n.parts[coordinate[i]]) if isinstance(n, _Chunks) else (local * n, 0)
        for j in range(len(stride)):
            if j != d and stride[j] >= stride[d] and local:
                stride[j], left = divmod(stride[j] * size, local)
                rest += left
        if rest:
            raise ValueError(f'mesh: a local extent {local} on mesh dimension {i} is no share of its parts '
                             f'{n.parts}: from_local takes shape= and stride=')
        shape[d] = size
    return shape, stride


# Rules that assume equal shards.
def _partitioned(spec, dim):
    dim %= spec.ndim
    return any(p.is_shard(dim) and isinstance(spec.mesh.size(i), _Chunks) for i, p in enumerate(spec.placements))


_evenly_on_dim, _spec_evenly_on_dim = _math_ops.is_tensor_evenly_shardable_on_dim, _math_ops._is_spec_evenly_sharded_on_dim


def _is_tensor_evenly_shardable_on_dim(shape, spec, dim):
    """mean/avg: Partial("avg") of unequal local means is not their mean; replicate first."""
    return not _partitioned(spec, dim) and _evenly_on_dim(shape, spec, dim)


def _is_spec_evenly_sharded_on_dim(spec, dim):
    return not _partitioned(spec, dim) and _spec_evenly_on_dim(spec, dim)


_propagate = _view_ops.propagate_shape_and_sharding


def _propagate_shape_and_sharding(input_src_placements, global_input_shape, rule, mesh_sizes, strict_view=False):
    """Views: stock's divisibility by the mesh size, by P on a partitioned dimension (P | the dimension
    split or first flattened is exactly when every share maps onto whole output rows)."""
    if not any(isinstance(n, _Chunks) for n in mesh_sizes):
        return _propagate(input_src_placements, global_input_shape, rule, mesh_sizes, strict_view)
    units = tuple(n.units if isinstance(n, _Chunks) else n for n in mesh_sizes)
    target, output = _propagate(input_src_placements, global_input_shape, rule, units, strict_view)
    if any(isinstance(p, _StridedShard) for p in output) and any(
            p.is_shard() and isinstance(n, _Chunks) for p, n in zip(target, mesh_sizes)):
        raise NotImplementedError('mesh: a view making a _StridedShard of a partitioned mesh dimension')
    return target, output


def _convolution_filter(stock):
    """Convolution's last-dimension sharding assumes equal widths (stride-divisible shards)."""
    def last_dim_unpartitioned(mesh, op_schema, input_specs, output_specs):
        spec = input_specs[1 if op_schema.op == aten.convolution_backward.default else 0]
        return not _partitioned(spec, -1) and stock(mesh, op_schema, input_specs, output_specs)
    return last_dim_unpartitioned


def _calc_first_shard_size(spec):
    """DTensor RNG's per-shard stride: the largest shard (rank 0's under torch.chunk), so no two
    shards' offsets overlap."""
    size = list(spec.shape)
    for i, p in enumerate(spec.placements):
        if isinstance(p, (Shard, _StridedShard)):
            n, N = spec.mesh.size(i), spec.shape[p.dim]
            size[p.dim] = max(_split(n, N)[0]) if _weighted(n, N) else p._local_shard_size_and_offset(N, n, 0)[0]
    return size


_optimize = _redistribute._optimize_transform_infos


def _optimize_transform_infos(transform_infos, device_mesh, src_placements, dst_placements):
    """Merging collectives needs a flattened mesh, which a partitioned dimension has none of."""
    if any(isinstance(device_mesh.size(info.mesh_dim), _Chunks) for info in transform_infos):
        return transform_infos
    return _optimize(transform_infos, device_mesh, src_placements, dst_placements)


# Context parallelism (torch/distributed/tensor/experimental/_context_parallel).
_default_load_balancer, _head_tail_indices = _lb._create_default_load_balancer, _lb._HeadTailLoadBalancer._generate_indices
_cp_block_mask, _unshard = _cp._create_cp_block_mask, _cp.context_parallel_unshard


def _create_default_load_balancer(seq_length, world_size, device):
    """Head-tail over the parts where 2P divides the sequence; else none (contiguous shares)."""
    if not isinstance(world_size, _Chunks):
        return _default_load_balancer(seq_length, world_size, device)
    if _cp._cp_options.enable_load_balance and seq_length % (2 * world_size.units) == 0:
        return _lb._HeadTailLoadBalancer(seq_length, world_size, device)
    return None


def _generate_head_tail_indices(self, restore=False):
    """torch's head-tail with shares: rank r holds positions [H_r, H_r + a_r) and their mirror
    [S - H_r - a_r, S - H_r), a_r = (S/2) p_r / P, H_r the heads of the ranks before it; its causal
    work is a_r S, linear in its share, as each rank's is S^2 / 2n at equal shares."""
    n, S = self.world_size, self.seq_length
    if not isinstance(n, _Chunks):
        return _head_tail_indices(self, restore)
    if S % (2 * n.units):
        raise ValueError(f'mesh: head-tail balancing of {S} positions over {n.units} units needs a multiple of {2 * n.units}')
    a, h = _split(n, S // 2)
    spans = [span for r in range(n) for span in ((h[r], h[r] + a[r]), (S - h[r] - a[r], S - h[r]))]
    indices = torch.cat([torch.arange(*span, dtype=torch.int, device=self.device) for span in spans])
    return (torch.argsort(indices) if restore else indices).unsqueeze(0)


def _refuse_balancer(stock):
    @functools.wraps(stock)
    def refuse(self, restore=False):
        if isinstance(self.world_size, _Chunks):
            raise NotImplementedError(f'mesh: {type(self).__name__} on a partitioned mesh dimension')
        return stock(self, restore)
    return refuse


def _create_cp_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device_mesh, load_balancer=None):
    if isinstance(device_mesh.size(), _Chunks):
        raise NotImplementedError('mesh: flex-attention context parallelism on a partitioned mesh dimension')
    return _cp_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device_mesh, load_balancer)


@torch.no_grad()
def context_parallel_unshard(mesh, buffers, seq_dims, load_balancer=None):
    """torch's context_parallel_unshard: the sequence P / p_r times this rank's share, gathered by
    Allgatherv, restored by the head-tail balancer's indices."""
    n = mesh.size()
    if not isinstance(n, _Chunks):
        return _unshard(mesh, buffers, seq_dims, load_balancer)
    local, rank = buffers[0].shape[seq_dims[0]], mesh.get_local_rank()
    seq_length, rest = divmod(local * n.units, n.parts[rank])
    if rest:
        raise ValueError(f'mesh: {local} positions on rank {rank} are no share of the parts {n.parts}')
    load_balancer = load_balancer or _cp._create_default_load_balancer(seq_length, n, buffers[0].device)
    restore = load_balancer._generate_indices(restore=True) if load_balancer else None
    unsharded = []
    for b, dim in zip(buffers, seq_dims):
        b = _allgatherv(b, dim, _split(n, seq_length)[0], mesh.get_group())
        if restore is not None:
            for i in range(b.size(0)):
                b[i] = torch.index_select(b[i], dim - 1, restore[0 if restore.size(0) == 1 else i])
        unsharded.append(b)
    return unsharded


def _ring_attention(group, op, query, key, value, lengths, is_causal=False, **kwargs):
    """torch's _templated_ring_attention forward (_attention.py) with each source's block at its own
    length (lengths[j], its share of the sequence): the rotation one Allgatherv of every rank's keys
    and values, the block of source j = (rank - i) mod n viewed at lengths[j].  The rest is torch's:
    _is_causal_behavior, the head-tail halves (a rank's head and tail are equal), _SDPAMerger."""
    if is_causal and query.size(2) != key.size(2):
        raise NotImplementedError('is_causal requires the same query and context sequence lengths')
    balanced = _cp._cp_options.enable_load_balance
    if not is_causal and balanced:
        raise RuntimeError('Load balancing requires `is_causal=True`.')
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    key, value = key.contiguous(), value.contiguous()
    row_k, row_v = key.numel() // key.size(2), value.numel() // value.size(2)
    counts = [s * (row_k + row_v) for s in lengths]
    blocks = list(_empty((sum(counts),), key).split(counts))
    gathered = dist.all_gather(blocks, torch.cat([key.flatten(), value.flatten()]), group=group, async_op=True)
    merger = _cp._SDPAMerger(_cp._cp_options.convert_to_f32, seq_dim=2)
    saved_rest = None
    for i in range(size):
        if i > 0:
            if i == 1:
                gathered.wait()
            j = (rank - i) % size
            key = blocks[j][:lengths[j] * row_k].view(key.shape[:2] + (lengths[j],) + key.shape[3:])
            value = blocks[j][lengths[j] * row_k:].view(value.shape[:2] + (lengths[j],) + value.shape[3:])
        behavior = _cp._is_causal_behavior(rank=rank, world_size=size, i=i, is_causal=is_causal)
        if behavior == _cp._CausalBehavior.SKIP:
            continue
        if i == 0 or not balanced or not is_causal:
            q, k, v, partial = query, key, value, False
        elif i <= rank:
            q, k, v, partial = query, key.chunk(2, dim=2)[0], value.chunk(2, dim=2)[0], False
        else:
            q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True
        out, logsumexp, *rest = op(q, k, v, is_causal=behavior.value, **kwargs)
        if saved_rest is None:
            saved_rest = rest
        merger.step(out, logsumexp, partial)
    return *merger.results(), *saved_rest


def _install():
    DeviceMesh.__init__, DeviceMesh.size, DeviceMesh.shape = _mesh_init, _mesh_size, property(_mesh_shape)
    DeviceMesh._hash_key, DeviceMesh.__eq__ = _mesh_hash_key, _mesh_eq
    DeviceMesh._create_flatten_mesh, DeviceMesh._unflatten = _mesh_flatten, _mesh_unflatten
    Shard.local_shard_size_and_offset = staticmethod(_shard_size_and_offset)
    Shard._split_tensor, Shard._select_split_tensor = _shard_split_tensor, _shard_select_split_tensor
    Shard._shard_tensor, Shard._reduce_shard_tensor = _shard_shard_tensor, _shard_reduce_shard_tensor
    Shard._to_replicate_tensor, Shard._to_new_shard_dim = _shard_to_replicate_tensor, _shard_to_new_shard_dim
    _StridedShard.local_shard_size_and_offset = _strided(_StridedShard.local_shard_size_and_offset)
    _StridedShard._split_tensor = _strided(_StridedShard._split_tensor)
    _api.compute_global_tensor_info = _utils.compute_global_tensor_info = _compute_global_tensor_info
    _math_ops.is_tensor_evenly_shardable_on_dim = _is_tensor_evenly_shardable_on_dim
    _math_ops._is_spec_evenly_sharded_on_dim = _is_spec_evenly_sharded_on_dim
    _view_ops.propagate_shape_and_sharding = _propagate_shape_and_sharding
    for op in (aten.convolution.default, aten.convolution_backward.default):
        info = DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs[op]
        info.full_mesh_strategy_filter = _convolution_filter(info.full_mesh_strategy_filter)
    _random._calc_first_shard_size = _calc_first_shard_size
    _redistribute._optimize_transform_infos = _optimize_transform_infos
    _lb._create_default_load_balancer = _cp._create_default_load_balancer = _create_default_load_balancer
    _lb._HeadTailLoadBalancer._generate_indices = _generate_head_tail_indices
    for balancer in (_lb._PerDocumentHeadTailLoadBalancer, _lb._PTRRLoadBalancer):
        balancer._generate_indices = _refuse_balancer(balancer._generate_indices)
    _cp._create_cp_block_mask = _create_cp_block_mask
    for module in (_cp, _context_parallel, _attention_stub):
        module.context_parallel_unshard = context_parallel_unshard


_install()
