"""A capacity-shaped partition operand for DeviceMesh: DTensor's Shard, tensor parallelism and context
parallelism split a mesh dimension by parts that vary between calls, in local buffers whose shapes do
not (design/heterogeneity.md §4 and R14).

Stock PyTorch splits every sharded extent equally (Shard follows torch.chunk).  Here a mesh dimension
carries a capacity c_r >= 1 per coordinate and parts p_r (1 <= p_r <= c_r) of P = sum(p) units,
attached with the mesh: attach(mesh, tp=((15, 15), (4, 12))).  An extent N that P divides (any other
raises ValueError, on every rank alike) is split N p_r / P per coordinate in rank order, as the logical
tensor reads, and rank r holds its share in a local buffer of c_r N / P along that dimension: its
N p_r / P valid elements first, padding after.  Every shape follows the capacities, fixed for the
attach (a coordinate's capacity is the largest share the configuration allows it: rdma/allocate.py
capacity).  The parts are a tensor operand (operand(mesh, "tp").parts), written between calls by
write(mesh, tp=(5, 11)) and read by the ops below when they run, so a program recorded or compiled once
is replayed across parts with no shape change, recompile or reallocation.  relay(t, full) writes a
sharded tensor's local buffer from its logical value under the current parts (a program's weights,
after each write).

Uneven partitioning as GSPMD does it [Xu et al. 2021, "GSPMD: general and scalable parallelization for
ML computation graphs", arXiv:2105.04663, §3.3: pad a dimension to the shards' size, and mask the padding
to the identity of the next operation wherever its data could leak into valid results]:
- Padding holds the identity of its next consumer, or garbage that nothing reads unmasked.  Sources keep
  the identity: a local buffer is written with zeros past its valid extent (distribute_tensor, relay),
  and the collectives below write the identity past `valid`.
- Before an op that reduces or contracts over a partitioned dimension of a local shard (its DTensor
  sharding has a Partial output on that mesh dimension: mm, bmm, addmm, baddbmm, dot, mv, sum, amax,
  amin), each input sharded there is re-masked to the Partial's identity (0 for sums and contractions,
  -inf for max, +inf for min): where(valid, x, identity), the op torch_mesh::remask.
- Count-dependent: mean over a partitioned dimension is the masked sum divided by its logical count.
  Softmax, var, the norms, sort, top-k, scans and indexing need the dimension whole, which DTensor gives
  them by the gather below, where only the valid elements remain.
- Attention: context parallelism's ring masks each source's keys past its valid extents (score -inf).
No kernel changes: stock kernels compute garbage in padding that nothing reads unmasked.

Redistribution, each block at its capacity (static counts, never padded to a common size): Shard to
Replicate all-gathers the blocks and packs the valid elements into the logical tensor (torch_mesh::pack,
a fixed-shape index from the parts); Replicate to Shard takes this rank's valid elements into its block
(torch_mesh::unpack, then its static block); Partial to Shard unpacks with the identity and
reduce-scatters the blocks; Shard to Partial places this rank's valid elements among zeros; Shard(d) to
Shard(d') goes through Replicate.  DeviceMesh.size() and .shape carry the capacities as an int subclass
(the only way they reach Shard's mesh-less size function), so local sizes, from_local's inverse
(global = local P / c_r), views (divisibility by P) and factories follow.  Context parallelism's
head-tail split keeps each half at its capacity: rank r's block is its head's valid positions, padding,
its tail's, padding.

Without an attach every replaced function calls PyTorch's own.  Refused (on every rank alike): a mesh
dimension of more than one coordinate before a partitioned one (ValueError at attach), _StridedShard,
flattening or unflattening a partitioned dimension, flex-attention context parallelism, the
per-document and PTRR balancers and LocalTensor's RNG tracker (NotImplementedError).  Under torch.compile
context parallelism's SDPA on a partitioned mesh is a graph break (_distribute_function).  Left as they are:
the strategy costs (MeshTopoInfo, redistribute_cost), which choose among collectives, not splits.  This
module is imported by the backend's first process group; a program that imports
context_parallel_unshard by name imports this module before it."""
import functools
import itertools
import math

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, _api, _random, _redistribute, _utils
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._ops import _conv_ops, _math_ops, _view_ops  # noqa: F401 (registers conv)
from torch.distributed.tensor.experimental import _attention as _attention_stub
from torch.distributed.tensor.experimental import _context_parallel as _context_parallel
from torch.distributed.tensor.experimental._context_parallel import _attention as _cp, _load_balancer as _lb
from torch.distributed.tensor.placement_types import Shard, _StridedShard
from torch.utils import _pytree as pytree

aten = torch.ops.aten


class _Operand:
    """A partitioned dimension's capacities (fixed) and parts (a CPU tensor, written between calls)."""

    def __init__(self, key, capacity, parts):
        self.key, self.capacity, self.units = key, tuple(int(c) for c in capacity), sum(int(p) for p in parts)
        self.parts, self.epoch, self.cache = torch.zeros(len(self.capacity), dtype=torch.int64), 0, {}
        self.write(parts)

    def write(self, parts):
        parts = tuple(int(p) for p in parts)
        if len(parts) != len(self.capacity) or sum(parts) != self.units or not all(1 <= p <= c for p, c in zip(parts, self.capacity)):
            raise ValueError(f'mesh: parts {parts} of {self.units} units within the capacities {self.capacity}: one a '
                             f'coordinate, each at least 1')
        self.parts.copy_(torch.tensor(parts))
        self.epoch, self.cache = self.epoch + 1, {}

    def cached(self, what, build):
        if what not in self.cache:
            self.cache[what] = build(self.parts.tolist())
        return self.cache[what]


_OPERANDS = {}


class _Chunks(int):
    """A partitioned mesh dimension's size n: an int to all of PyTorch, carrying its operand's key,
    capacities and units."""

    def __new__(cls, key):
        o = _OPERANDS[key]
        self = super().__new__(cls, len(o.capacity))
        self.key, self.capacity, self.units = key, o.capacity, o.units
        return self

    def __reduce__(self):
        return _Chunks, (self.key,)


def _weighted(n, N):
    """Whether an extent N is split by a partition (n a partitioned dimension's size): P must divide
    N, so that every rank infers the same global extent from its local one (global = local P / c_r)."""
    if not isinstance(n, _Chunks):
        return False
    if N % n.units:
        raise ValueError(f'mesh: an extent of {N} split over {n.units} units is no multiple of them')
    return True


def _split(n, N):
    """(sizes, offsets) of the local buffers of an extent N over a mesh dimension of size n: the
    capacities' on a partitioned dimension (static), else torch.chunk's."""
    if _weighted(n, N):
        sizes = [N // n.units * c for c in n.capacity]
    else:
        step = -(-N // n)
        sizes = [max(0, min(step, N - step * r)) for r in range(n)]
    return sizes, [sum(sizes[:r]) for r in range(len(sizes))]


# The operand read when an op runs: masks and indices from the parts, cached per write.
def _valid(key, rank, length, device):
    """A local block of `length` of rank `rank`: which of its elements are valid (its first length p/c)."""
    o = _OPERANDS[key]
    return o.cached(('valid', rank, length, str(device)),
                    lambda p: torch.arange(length, device=device) < length // o.capacity[rank] * p[rank])


def _index(key, length, device):
    """For blocks of every rank laid end to end (`length` in all): where each logical element lies."""
    o = _OPERANDS[key]

    def build(p):
        g = length // sum(o.capacity)
        starts = itertools.accumulate((c * g for c in o.capacity), initial=0)
        return torch.cat([torch.arange(s, s + q * g) for s, q in zip(starts, p)]).to(device)
    return o.cached(('index', length, str(device)), build)


def _along(mask, dim, ndim):
    shape = [1] * ndim
    shape[dim] = -1
    return mask.view(shape)


@torch.library.custom_op('torch_mesh::remask', mutates_args=())
def remask(x: torch.Tensor, dim: int, key: str, rank: int, identity: float) -> torch.Tensor:
    """x, rank `rank`'s local block along `dim`, with every element past its valid extent `identity`."""
    return torch.where(_along(_valid(key, rank, x.size(dim), x.device), dim, x.dim()), x, identity)


@torch.library.custom_op('torch_mesh::pack', mutates_args=())
def pack(x: torch.Tensor, dim: int, key: str) -> torch.Tensor:
    """Every rank's block along `dim`, laid end to end in x, as the logical tensor: the valid elements."""
    return x.index_select(dim, _index(key, x.size(dim), x.device))


@torch.library.custom_op('torch_mesh::unpack', mutates_args=())
def unpack(x: torch.Tensor, dim: int, key: str, identity: float) -> torch.Tensor:
    """The logical tensor x as every rank's block laid end to end along `dim`, `identity` past each
    block's valid extent."""
    o = _OPERANDS[key]
    shape = list(x.shape)
    shape[dim] = x.size(dim) * sum(o.capacity) // o.units
    return x.new_full(shape, identity).index_copy(dim, _index(key, shape[dim], x.device), x)


@remask.register_fake
def _(x, dim, key, rank, identity):
    return torch.empty_like(x)


@pack.register_fake
def _(x, dim, key):
    o, shape = _OPERANDS[key], list(x.shape)
    shape[dim] = x.size(dim) * o.units // sum(o.capacity)
    return x.new_empty(shape)


@unpack.register_fake
def _(x, dim, key, identity):
    o, shape = _OPERANDS[key], list(x.shape)
    shape[dim] = x.size(dim) * sum(o.capacity) // o.units
    return x.new_empty(shape)


def _keep(names):
    def setup(ctx, inputs, output):
        ctx.saved = dict(zip(names, inputs[1:]))
    return setup


remask.register_autograd(lambda ctx, g: (remask(g, ctx.saved['dim'], ctx.saved['key'], ctx.saved['rank'], 0.0), None, None, None, None),
                         setup_context=_keep(('dim', 'key', 'rank')))
pack.register_autograd(lambda ctx, g: (unpack(g, ctx.saved['dim'], ctx.saved['key'], 0.0), None, None),
                       setup_context=_keep(('dim', 'key')))
unpack.register_autograd(lambda ctx, g: (pack(g, ctx.saved['dim'], ctx.saved['key']), None, None, None),
                         setup_context=_keep(('dim', 'key')))


# The operand's API.
def attach(mesh, **dims):
    """Partition mesh dimensions by name, before the mesh is first used: attach(mesh, tp=(capacity,
    parts)), a capacity and a part a coordinate."""
    if mesh.__dict__.get('_hash'):
        raise RuntimeError('mesh: a partition is attached before the mesh is first used')
    for name in dims:
        if name not in (mesh.mesh_dim_names or ()):
            raise ValueError(f'mesh: no mesh dimension {name!r} to partition in {mesh.mesh_dim_names}')
    keys = dict(_named(mesh))
    for name, (capacity, parts) in dims.items():
        key = f'{name}#{len(_OPERANDS)}'
        _OPERANDS[key] = _Operand(key, capacity, parts)
        keys[name] = key
    _handlers()
    _attach(mesh, keys)


def write(mesh, **dims):
    """New parts for partitioned dimensions, between calls: write(mesh, tp=(5, 11)).  Shapes do not change;
    the tensors a program holds keep their old parts' layout until relay rewrites them."""
    named = _named(mesh)
    for name, parts in dims.items():
        if name not in named:
            raise ValueError(f'mesh: mesh dimension {name!r} is not partitioned')
        _OPERANDS[named[name]].write(parts)


def operand(mesh, name):
    """The partition of mesh dimension `name`: .capacity, .units, .parts (the tensor), .key."""
    return _OPERANDS[_named(mesh)[name]]


def sizes(mesh, name, N):
    """The local buffers' sizes of an extent N along mesh dimension `name` (capacities; stock's without)."""
    return tuple(_split(mesh.size(mesh._get_mesh_dim_by_name(name)), N)[0])


@torch.no_grad()
def relay(tensor, full):
    """Writes DTensor `tensor`'s local buffer from its logical value `full` under the current parts: on a
    partitioned dimension its valid elements, zeros past them (a plain tensor: all of `full`).  In place:
    no reallocation."""
    if not isinstance(tensor, DTensor):
        return tensor.copy_(full)
    mesh, value, coordinate = tensor.device_mesh, full, tensor.device_mesh.get_coordinate()
    for i, p in enumerate(tensor.placements):
        if p.is_partial():
            raise NotImplementedError('mesh: relay of a Partial placement')
        if p.is_shard():
            value = p._select_split_tensor(value, mesh.size(i), coordinate[i], with_padding=False, clone=False)
    tensor._local_tensor.copy_(value)


def _attach(mesh, keys):
    partition = {}
    for name, key in keys.items():
        d, o = mesh.mesh_dim_names.index(name), _OPERANDS[key]
        if len(o.capacity) != _size(mesh, d):
            raise ValueError(f'mesh: partition {name}: {len(o.capacity)} capacities for {_size(mesh, d)} coordinates')
        outer = [mesh.mesh_dim_names[j] for j in range(d) if _size(mesh, j) > 1]
        if outer:
            raise ValueError(f'mesh: partition {name}: mesh dimensions {outer} precede it; a partitioned dimension is '
                             f'its mesh\'s first of more than one coordinate')
        partition[d] = key
    mesh._partition = partition
    mesh._chunks = {d: _Chunks(key) for d, key in partition.items()}


def _named(mesh):
    return {mesh.mesh_dim_names[d]: key for d, key in (mesh.__dict__.get('_partition') or {}).items()}


# DeviceMesh (torch/distributed/device_mesh.py): the partition at construction (a sub-mesh keeps its
# root's), in size and shape, hash and equality; flattening and unflattening a partitioned dimension
# refused.
_init, _size, _shape = DeviceMesh.__init__, DeviceMesh.size, DeviceMesh.shape.fget
_hash_key, _eq, _flatten, _unflatten = DeviceMesh._hash_key, DeviceMesh.__eq__, DeviceMesh._create_flatten_mesh, DeviceMesh._unflatten


def _mesh_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    root = kwargs.get('_root_mesh')
    named = {name: key for name, key in (_named(root) if root is not None else {}).items() if name in (self.mesh_dim_names or ())}
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
    other = getattr(other, 'real_obj', other)  # the mesh a compiled trace holds opaque (FakeScriptObject)
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


# The collectives, each block at its capacity, the split dimension moved to the front.
_IDENTITY = {'sum': 0.0, 'avg': 0.0, 'max': -math.inf, 'min': math.inf}


def _empty(shape, like):
    """An MPS window tensor where the mesh backend has a group (torch_mesh.empty: its collectives run on it in
    place), else torch.empty.  Named explicitly: under DTensor's dispatch (the ring's handler) torch
    function modes do not run, so the backend's factory mode would not see a torch.empty there."""
    if like.device.type == 'mps':
        import torch_mesh
        if torch_mesh._world is not None:
            return torch_mesh.empty(tuple(shape), dtype=like.dtype)
    return torch.empty(shape, dtype=like.dtype, device=like.device)


def _allgatherv(x, dim, sizes, group):
    """The blocks gathered, this rank's copied into its place in the output first (the backend then runs
    the all-gather in place)."""
    y = x.movedim(dim, 0)
    out = _empty((sum(sizes),) + y.shape[1:], y)
    blocks = list(out.split(sizes))
    mine = blocks[dist.get_rank(group)].copy_(y)
    dist.all_gather(blocks, mine, group=group)
    return out.movedim(0, dim)


def _reduce_scatterv(x, dim, sizes, rank, op, group):
    """The blocks reduce-scattered in place in a contiguous copy of x (the backend's in-place form: the
    output is this rank's block of the input)."""
    y = x.movedim(dim, 0)
    blocks = list(_empty(y.shape, y).copy_(y).split(sizes))
    dist.reduce_scatter(blocks[rank], blocks, op=getattr(dist.ReduceOp, op.upper()), group=group)
    return blocks[rank].movedim(0, dim).contiguous()


# Shard (torch/distributed/tensor/placement_types.py).
_size_and_offset, _split_tensor, _select_split_tensor = Shard.local_shard_size_and_offset, Shard._split_tensor, Shard._select_split_tensor
_shard_tensor, _reduce_shard_tensor, _to_partial_tensor = Shard._shard_tensor, Shard._reduce_shard_tensor, Shard._to_partial_tensor
_to_replicate_tensor, _to_new_shard_dim = Shard._to_replicate_tensor, Shard._to_new_shard_dim


@maybe_run_for_local_tensor
def _shard_size_and_offset(curr_local_size, num_chunks, rank):
    if not _weighted(num_chunks, curr_local_size):
        return _size_and_offset(curr_local_size, num_chunks, rank)
    sizes, offsets = _split(num_chunks, curr_local_size)
    return sizes[rank], offsets[rank]


def _shard_split_tensor(self, tensor, num_chunks, *, with_padding=True, contiguous=True):
    N = tensor.size(self.dim)
    if not _weighted(num_chunks, N):
        return _split_tensor(self, tensor, num_chunks, with_padding=with_padding, contiguous=contiguous)
    blocks = unpack(tensor, self.dim, num_chunks.key, 0.0).split(_split(num_chunks, N)[0], self.dim)
    return [b.contiguous() if contiguous else b for b in blocks], [0] * len(blocks)


@maybe_run_for_local_tensor
def _shard_select_split_tensor(self, tensor, num_chunks, index, *, with_padding=True, contiguous=True, clone=True):
    N = tensor.size(self.dim)
    if not _weighted(num_chunks, N):
        return _select_split_tensor(self, tensor, num_chunks, index, with_padding=with_padding, contiguous=contiguous,
                                    clone=clone)
    sizes, offsets = _split(num_chunks, N)
    block = unpack(tensor, self.dim, num_chunks.key, 0.0).narrow(self.dim, offsets[index], sizes[index])
    return block.contiguous() if contiguous or clone else block


def _shard_shard_tensor(self, tensor, mesh, mesh_dim, src_data_rank=0):
    """Scatterv of the blocks from src_data_rank (src_data_rank None: this rank's own block)."""
    n, coordinate = mesh.size(mesh_dim), mesh.get_coordinate()
    if src_data_rank is None or coordinate is None or not _weighted(n, tensor.size(self.dim)):
        return _shard_tensor(self, tensor, mesh, mesh_dim, src_data_rank)
    sizes, rank = _split(n, tensor.size(self.dim))[0], coordinate[mesh_dim]
    out = _empty(tensor.shape[:self.dim] + (sizes[rank],) + tensor.shape[self.dim + 1:], tensor)
    blocks = self._split_tensor(tensor, n)[0] if rank == src_data_rank else None
    dist.scatter(out, blocks, group=mesh.get_group(mesh_dim), group_src=src_data_rank)
    return out


def _shard_reduce_shard_tensor(self, tensor, mesh, reduce_op, mesh_dim):
    """Partial to Shard: the logical partial unpacked with the reduction's identity, reduce-scattered."""
    n, coordinate = mesh.size(mesh_dim), mesh.get_coordinate()
    if coordinate is None or not _weighted(n, tensor.size(self.dim)):
        return _reduce_shard_tensor(self, tensor, mesh, reduce_op, mesh_dim)
    padded = unpack(tensor, self.dim, n.key, _IDENTITY[reduce_op])
    return _reduce_scatterv(padded, self.dim, _split(n, tensor.size(self.dim))[0], coordinate[mesh_dim], reduce_op,
                            mesh.get_group(mesh_dim))


def _shard_to_replicate_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
    n, N = mesh.size(mesh_dim), current_logical_shape[self.dim]
    if not _weighted(n, N):
        return _to_replicate_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape)
    return pack(_allgatherv(local_tensor, self.dim, _split(n, N)[0], mesh.get_group(mesh_dim)), self.dim, n.key).contiguous()


def _shard_to_partial_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
    """Shard to Partial("sum"): this rank's valid elements at their logical places, zeros elsewhere."""
    n, N = mesh.size(mesh_dim), current_logical_shape[self.dim]
    if not _weighted(n, N):
        return _to_partial_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape)
    sizes, offsets = _split(n, N)
    rank, total = mesh.get_coordinate()[mesh_dim], sum(sizes)
    parts = [local_tensor.new_zeros(local_tensor.shape[:self.dim] + (s,) + local_tensor.shape[self.dim + 1:])
             for s in (offsets[rank], total - offsets[rank] - sizes[rank])]
    return pack(torch.cat([parts[0], remask(local_tensor, self.dim, n.key, rank, 0.0), parts[1]], self.dim), self.dim, n.key)


def _shard_to_new_shard_dim(self, local_tensor, mesh, mesh_dim, current_logical_shape, new_shard_dim):
    """Shard(d) to Shard(d') on a partitioned mesh dimension: through Replicate."""
    n = mesh.size(mesh_dim)
    if not (_weighted(n, current_logical_shape[self.dim]) or _weighted(n, current_logical_shape[new_shard_dim])):
        return _to_new_shard_dim(self, local_tensor, mesh, mesh_dim, current_logical_shape, new_shard_dim)
    full = self._to_replicate_tensor(local_tensor, mesh, mesh_dim, current_logical_shape)
    return Shard(new_shard_dim)._replicate_to_shard(full, mesh, mesh_dim, mesh.get_coordinate()[mesh_dim])


def _strided(stock):
    """_StridedShard's sizes are right-to-left strided chunks, which a partition does not describe."""
    @functools.wraps(stock)
    def refuse(self, *args, **kwargs):
        if isinstance(kwargs.get('num_chunks', args[1] if len(args) > 1 else None), _Chunks):
            raise NotImplementedError('mesh: _StridedShard (FSDP2 with TP, nested views) on a partitioned mesh dimension')
        return stock(self, *args, **kwargs)
    return refuse


# from_local's inverse (torch._C._DTensor_compute_global_tensor_info, bound in _api and _utils).
_global_tensor_info = _utils.compute_global_tensor_info


def _by_parts(mesh, placements):
    """Whether placements shard a tensor dimension over a partitioned mesh dimension."""
    return any(p.is_shard() and isinstance(mesh.size(i), _Chunks) for i, p in enumerate(placements))


def _compute_global_tensor_info(tensor, mesh, placements):
    coordinate = mesh.get_coordinate()
    if coordinate is None or not _by_parts(mesh, placements):
        return _global_tensor_info(tensor, mesh, placements)
    shape, stride = list(tensor.size()), list(tensor.stride())
    for i in reversed(range(len(placements))):
        p, n = placements[i], mesh.size(i)
        if isinstance(p, _StridedShard):
            raise NotImplementedError('mesh: _StridedShard with a partitioned mesh dimension')
        if not p.is_shard():
            continue
        d, local = p.dim, shape[p.dim]
        size, rest = divmod(local * n.units, n.capacity[coordinate[i]]) if isinstance(n, _Chunks) else (local * n, 0)
        for j in range(len(stride)):
            if j != d and stride[j] >= stride[d] and local:
                stride[j], left = divmod(stride[j] * size, local)
                rest += left
        if rest:
            raise ValueError(f'mesh: a local extent {local} on mesh dimension {i} is no capacity block of '
                             f'{n.capacity}: from_local takes shape= and stride=')
        shape[d] = size
    return shape, stride


# Rules that assume equal shards.
def _partitioned(spec, dim):
    dim %= spec.ndim
    return any(p.is_shard(dim) and isinstance(spec.mesh.size(i), _Chunks) for i, p in enumerate(spec.placements))


_evenly_on_dim, _spec_evenly_on_dim = _math_ops.is_tensor_evenly_shardable_on_dim, _math_ops._is_spec_evenly_sharded_on_dim


def _is_tensor_evenly_shardable_on_dim(shape, spec, dim):
    """Reductions whose Partial of local results assumes equal shards (avg, the norms): replicate first."""
    return not _partitioned(spec, dim) and _evenly_on_dim(shape, spec, dim)


def _is_spec_evenly_sharded_on_dim(spec, dim):
    return not _partitioned(spec, dim) and _spec_evenly_on_dim(spec, dim)


_propagate = _view_ops.propagate_shape_and_sharding


def _propagate_shape_and_sharding(input_src_placements, global_input_shape, rule, mesh_sizes, strict_view=False):
    """Views: stock's divisibility by the mesh size, by P on a partitioned dimension (P | the dimension
    split or first flattened is exactly when every block maps onto whole output rows)."""
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


# The re-mask before a reduction or contraction over a partitioned dimension (DTensor's op handlers).
def _remasked(op_call, args, kwargs):
    """The op as DTensor dispatches it (sharding, redistribution, the local op, the wrap), each input
    sharded on a partitioned mesh dimension on which the output is Partial first re-masked to the
    Partial's identity."""
    d = DTensor._op_dispatcher
    info = d.unwrap_to_op_info(op_call, args, kwargs)
    d.sharding_propagator.propagate(info)
    out = info.output_sharding
    specs = info.flat_args_schema
    if out.needs_redistribute:
        d.redistribute_local_args(info, out.redistribute_schema, out.use_val_from_redistribute_schema)
        schema = out.redistribute_schema.args_schema
        specs = pytree.tree_leaves(schema) if info.args_tree_spec is not None else schema
    mesh, spec, local = info.compute_mesh, out.output_spec, list(info.local_args)
    for i, placement in enumerate(spec.placements if isinstance(spec, DTensorSpec) else ()):
        n = mesh.size(i)
        if placement.is_partial() and isinstance(n, _Chunks):
            for j, s in enumerate(specs):
                if isinstance(s, DTensorSpec) and s.placements[i].is_shard():
                    local[j] = remask(local[j], s.placements[i].dim % local[j].dim(), n.key, mesh.get_coordinate()[i],
                                      _IDENTITY[placement.reduce_op])
    args = pytree.tree_unflatten(local, info.args_tree_spec) if info.args_tree_spec is not None else local
    op = out.redistribute_schema.op if out.needs_redistribute and out.redistribute_schema.op != op_call else op_call
    return d.wrap(op(*args, **info.local_kwargs), spec)


def _mean(op_call, args, kwargs):
    """mean over a partitioned dimension: the masked sum over the logical count."""
    named = dict(zip((a.name for a in op_call._schema.arguments), args), **kwargs)
    x, dims, keepdim, dtype = named['self'], named.get('dim'), named.get('keepdim', False), named.get('dtype')
    dims = sorted({d % x.ndim for d in dims} if dims else range(x.ndim))
    if not any(p.is_shard() and p.dim % x.ndim in dims and isinstance(x.device_mesh.size(i), _Chunks)
               for i, p in enumerate(x.placements)):
        return _remasked(op_call, args, kwargs)
    return torch.sum(x, dims, keepdim=keepdim, dtype=dtype) / math.prod(x.shape[d] for d in dims)


_REMASKED = (aten.mm.default, aten.bmm.default, aten.addmm.default, aten.baddbmm.default, aten.dot.default, aten.mv.default,
             aten.sum.default, aten.sum.dim_IntList, aten.amax.default, aten.amin.default)


def _handlers():
    """Installed with the first partition: without one, DTensor dispatches these ops as it does."""
    handlers = DTensor._op_dispatcher._custom_op_handlers
    for op in _REMASKED:
        handlers.setdefault(op, _remasked)
    handlers.setdefault(aten.mean.dim, _mean)
    handlers.setdefault(aten.mean.default, _mean)


_rng_offsets, _first_shard_size = _random.OffsetBasedRNGTracker._compute_rng_offsets, _random._calc_first_shard_size


def _compute_rng_offsets(self, spec):
    """DTensor RNG's (start, end) offset increments: stock's are k times the first shard's elements for
    shard k and the tensor's elements, which past equal shards would run the larger blocks into the next
    op's offsets.  Here block k (stock's row-major order over the shard grid) starts past the blocks
    before it and the op ends past the last, each rounded up to 4 as stock rounds."""
    mesh = spec.mesh
    if not _by_parts(mesh, spec.placements):
        return _rng_offsets(self, spec)
    extents = [[N] for N in spec.shape]
    for i, p in enumerate(spec.placements):
        if isinstance(p, _StridedShard):
            raise NotImplementedError('mesh: DTensor RNG over _StridedShard with a partitioned mesh dimension')
        if p.is_shard():
            n = mesh.size(i)
            extents[p.dim] = [_shard_size_and_offset(e, n, r)[0] for e in extents[p.dim] for r in range(n)]
    starts, end = [], 0
    for index in itertools.product(*(range(len(e)) for e in extents)):
        starts.append(end)
        end = (end + math.prod(e[j] for e, j in zip(extents, index)) + 3) // 4 * 4
    coordinate = [mesh._sym_get_coordinate(i) for i in range(mesh.ndim)]
    return starts[_random._calc_shard_linear_idx(*_random._calc_shard_info(coordinate, spec))], end


def _calc_first_shard_size(spec):
    """Stock's per-shard stride, which no longer serves a partitioned spec (above); LocalTensor's RNG
    tracker still calls it."""
    if _by_parts(spec.mesh, spec.placements):
        raise NotImplementedError("mesh: LocalTensor's DTensor RNG on a partitioned mesh dimension")
    return _first_shard_size(spec)


_optimize = _redistribute._optimize_transform_infos


def _optimize_transform_infos(transform_infos, device_mesh, src_placements, dst_placements):
    """Merging collectives needs a flattened mesh, which a partitioned dimension has none of."""
    if any(isinstance(device_mesh.size(info.mesh_dim), _Chunks) for info in transform_infos):
        return transform_infos
    return _optimize(transform_infos, device_mesh, src_placements, dst_placements)


# Context parallelism (torch/distributed/tensor/experimental/_context_parallel): each half at its capacity.
_default_load_balancer, _buffers, _cp_block_mask, _unshard = (_lb._create_default_load_balancer, _cp._context_parallel_buffers,
                                                              _cp._create_cp_block_mask, _cp.context_parallel_unshard)


class _HeadTail:
    """The balancer context_parallel holds on a partitioned mesh (its presence turns on the ring's
    head-tail halves); the positions are _positions', from the parts."""

    def __init__(self, world_size):
        self.world_size = world_size

    def _generate_indices(self, restore=False):
        raise NotImplementedError('mesh: head-tail indices on a partitioned mesh are _positions\' (capacity blocks)')


def _create_default_load_balancer(seq_length, world_size, device):
    """Head-tail where 2P divides the sequence; else none (contiguous shares)."""
    if not isinstance(world_size, _Chunks):
        return _default_load_balancer(seq_length, world_size, device)
    return _HeadTail(world_size) if _cp._cp_options.enable_load_balance and seq_length % (2 * world_size.units) == 0 else None


def _positions(key, rank, S, halves, device):
    """Rank `rank`'s block of a sequence of S split over the partition in `halves` (2: head-tail, rank r
    holding positions [H_r, H_r + a_r) and their mirror [S - H_r - a_r, S - H_r), a_r = (S/2) p_r / P, H_r
    the heads of the ranks before it; 1: contiguous shares): (the logical position of each element of
    its capacity block, 0 in padding; which are valid).  Its causal work is a_r S, linear in its share."""
    o = _OPERANDS[key]

    def build(p):
        g, before = S // halves // o.units, sum(p[:rank])
        a, h, H = g * p[rank], g * o.capacity[rank], g * before
        spans = [(H, H + a)] if halves == 1 else [(H, H + a), (S - H - a, S - H)]
        pad = torch.zeros(h - a, dtype=torch.int64)
        index = torch.cat([t for lo, hi in spans for t in (torch.arange(lo, hi), pad)])
        valid = torch.cat([t for _ in spans for t in (torch.ones(a, dtype=torch.bool), torch.zeros(h - a, dtype=torch.bool))])
        return index.to(device), valid.to(device)
    return o.cached(('positions', rank, S, halves, str(device)), build)


def _context_parallel_buffers(mesh, buffers, buffer_seq_dims, load_balancer=None):
    """Each buffer's capacity block of this rank: its valid positions (head-tail where load_balancer),
    zeros past them."""
    n = mesh.size()
    if not isinstance(n, _Chunks):
        return _buffers(mesh, buffers, buffer_seq_dims, load_balancer)
    rank, shards = mesh.get_local_rank(), []
    for b, dim in zip(buffers, buffer_seq_dims):
        if not isinstance(b, torch.Tensor):
            raise NotImplementedError('mesh: a BlockMask buffer (flex attention) on a partitioned mesh dimension')
        _weighted(n, b.size(dim))
        index, valid = _positions(n.key, rank, b.size(dim), 2 if load_balancer else 1, b.device)
        shards.append(torch.where(_along(valid, dim, b.dim()), b.index_select(dim, index), 0))
    return shards


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
    """torch's context_parallel_unshard: every rank's capacity block gathered (Allgatherv) and the valid
    positions put back in sequence order."""
    n = mesh.size()
    if not isinstance(n, _Chunks):
        return _unshard(mesh, buffers, seq_dims, load_balancer)
    local, rank = buffers[0].shape[seq_dims[0]], mesh.get_local_rank()
    S, rest = divmod(local * n.units, n.capacity[rank])
    if rest:
        raise ValueError(f'mesh: {local} positions on rank {rank} are no capacity block of {n.capacity}')
    halves = 2 if (load_balancer or _create_default_load_balancer(S, n, buffers[0].device)) else 1
    sizes, offsets = _split(n, S)

    def build(parts):
        """Where each logical position lies in the gathered blocks, from the parts on the host (no device
        sync per call)."""
        order = torch.empty(S, dtype=torch.int64)
        for r in range(len(sizes)):
            index, valid = _positions(n.key, r, S, halves, 'cpu')
            order[index[valid]] = offsets[r] + valid.nonzero().squeeze(1)
        return order.to(buffers[0].device)
    order = _OPERANDS[n.key].cached(('order', S, halves, str(buffers[0].device)), build)
    return [_allgatherv(b, dim, sizes, mesh.get_group()).index_select(dim, order) for b, dim in zip(buffers, seq_dims)]


_distribute = _cp._distribute_function


def _distribute_function(fn, fn_module, device_mesh, input_fn, output_fn):
    """context_parallel's SDPA on a partitioned mesh dimension, left out of torch.compile's graphs: its
    input conversion (DTensor.from_local over a partitioned mesh, whose size dynamo cannot build) and the
    ring run eagerly (torch.compiler.disable on the function it installs: a graph break at the call)."""
    _distribute(fn, fn_module, device_mesh, input_fn, output_fn)
    wrapper = getattr(fn_module, fn.__name__)
    if isinstance(device_mesh.size(), _Chunks) and wrapper in _cp._replaced_functions:
        disabled = torch.compiler.disable(wrapper)
        _cp._replaced_functions[disabled] = _cp._replaced_functions.pop(wrapper)
        setattr(fn_module, fn.__name__, disabled)


def _ring_attention(group, n, query, key, value, is_causal=False, **kwargs):
    """torch's _templated_ring_attention forward (_attention.py) over capacity blocks: the rotation one
    Allgatherv of every rank's keys and values (static counts), the block of source j = (rank - i) mod
    n viewed at its capacity and its keys past their valid extents masked (-inf scores); the rest is
    torch's: _is_causal_behavior, the head-tail halves (a block's halves are its head's and its tail's),
    _SDPAMerger.  Padded query rows compute garbage that the unshard drops."""
    from ._mps import _mps_block
    if is_causal and query.size(2) != key.size(2):
        raise NotImplementedError('is_causal requires the same query and context sequence lengths')
    balanced = _cp._cp_options.enable_load_balance
    if not is_causal and balanced:
        raise RuntimeError('Load balancing requires `is_causal=True`.')
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    S = query.size(2) * n.units // n.capacity[rank]
    lengths = _split(n, S)[0]
    valid = [_positions(n.key, j, S, 2 if balanced else 1, query.device)[1] for j in range(size)]
    key, value = key.contiguous(), value.contiguous()
    row_k, row_v = key.numel() // key.size(2), value.numel() // value.size(2)
    counts = [s * (row_k + row_v) for s in lengths]
    blocks = list(_empty((sum(counts),), key).split(counts))
    mine = torch.cat([key.flatten(), value.flatten()], out=blocks[rank])
    gathered = dist.all_gather(blocks, mine, group=group, async_op=True)
    merger = _cp._SDPAMerger(_cp._cp_options.convert_to_f32, seq_dim=2)
    for i in range(size):
        j = (rank - i) % size
        if i > 0:
            if i == 1:
                gathered.wait()
            key = blocks[j][:lengths[j] * row_k].view(key.shape[:2] + (lengths[j],) + key.shape[3:])
            value = blocks[j][lengths[j] * row_k:].view(value.shape[:2] + (lengths[j],) + value.shape[3:])
        behavior = _cp._is_causal_behavior(rank=rank, world_size=size, i=i, is_causal=is_causal)
        if behavior == _cp._CausalBehavior.SKIP:
            continue
        if i == 0 or not balanced or not is_causal:
            q, k, v, m, partial = query, key, value, valid[j], False
        elif i <= rank:
            q, k, v, m, partial = query, key.chunk(2, dim=2)[0], value.chunk(2, dim=2)[0], valid[j].chunk(2)[0], False
        else:
            q, k, v, m, partial = query.chunk(2, dim=2)[1], key, value, valid[j], True
        out, logsumexp = _mps_block(q, k, v, is_causal=behavior.value, key_valid=m, **kwargs)
        merger.step(out, logsumexp, partial)
    return merger.results()


def _install():
    DeviceMesh.__init__, DeviceMesh.size, DeviceMesh.shape = _mesh_init, _mesh_size, property(_mesh_shape)
    DeviceMesh._hash_key, DeviceMesh.__eq__ = _mesh_hash_key, _mesh_eq
    DeviceMesh._create_flatten_mesh, DeviceMesh._unflatten = _mesh_flatten, _mesh_unflatten
    Shard.local_shard_size_and_offset = staticmethod(_shard_size_and_offset)
    Shard._split_tensor, Shard._select_split_tensor = _shard_split_tensor, _shard_select_split_tensor
    Shard._shard_tensor, Shard._reduce_shard_tensor = _shard_shard_tensor, _shard_reduce_shard_tensor
    Shard._to_replicate_tensor, Shard._to_new_shard_dim = _shard_to_replicate_tensor, _shard_to_new_shard_dim
    Shard._to_partial_tensor = _shard_to_partial_tensor
    _StridedShard.local_shard_size_and_offset = _strided(_StridedShard.local_shard_size_and_offset)
    _StridedShard._split_tensor = _strided(_StridedShard._split_tensor)
    _api.compute_global_tensor_info = _utils.compute_global_tensor_info = _compute_global_tensor_info
    _math_ops.is_tensor_evenly_shardable_on_dim = _is_tensor_evenly_shardable_on_dim
    _math_ops._is_spec_evenly_sharded_on_dim = _is_spec_evenly_sharded_on_dim
    _view_ops.propagate_shape_and_sharding = _propagate_shape_and_sharding
    for op in (aten.convolution.default, aten.convolution_backward.default):
        info = DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs[op]
        info.full_mesh_strategy_filter = _convolution_filter(info.full_mesh_strategy_filter)
    _random.OffsetBasedRNGTracker._compute_rng_offsets = _compute_rng_offsets
    _random._calc_first_shard_size = _calc_first_shard_size
    _redistribute._optimize_transform_infos = _optimize_transform_infos
    _lb._create_default_load_balancer = _cp._create_default_load_balancer = _create_default_load_balancer
    _cp._context_parallel_buffers = _context_parallel_buffers
    for balancer in (_lb._PerDocumentHeadTailLoadBalancer, _lb._PTRRLoadBalancer):
        balancer._generate_indices = _refuse_balancer(balancer._generate_indices)
    _cp._create_cp_block_mask = _create_cp_block_mask
    _cp._distribute_function = _distribute_function
    for module in (_cp, _context_parallel, _attention_stub):
        module.context_parallel_unshard = context_parallel_unshard


_install()
