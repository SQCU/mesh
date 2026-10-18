"""A capacity-shaped partition operand: DTensor's Shard, tensor parallelism and context parallelism split a mesh
dimension by parts that vary between calls, in local buffers whose shapes do not (design/heterogeneity.md §4 and R14).

Stock PyTorch splits every sharded extent equally (Shard follows torch.chunk).  Here a mesh dimension carries a
capacity c_r >= 1 per coordinate and parts p_r (1 <= p_r <= c_r) of P = sum(p) units, attached with the mesh:
attach(mesh, tp=((15, 15), (4, 12))), the same on every rank.  An extent N that P divides (any other raises ValueError,
on every rank alike) is split N p_r / P per coordinate in rank order, as the logical tensor reads, and rank r holds its
share in a local buffer of c_r N / P along that dimension: its N p_r / P valid elements first, padding after.  Every
shape follows the capacities, fixed for the attach (a coordinate's capacity is the largest share the configuration
allows it, stated with the parts: design/heterogeneity.md §4).  The parts are a tensor operand (operand(mesh,
"tp").parts), written between calls by write(mesh, tp=(5, 11)) and read by the ops below when they run, so a program
recorded or compiled once is replayed across parts with no shape change, recompile or reallocation.  relay(t, full)
writes a sharded tensor's local buffer from its logical value under the current parts (a program's weights, after each
write).  On raw tensors split(mesh, "tp", N) gives the valid blocks in rank order, the ones the list collectives take
(dist.all_gather into a list of unequal outputs, dist.reduce_scatter of a list of unequal inputs: MPI's v-collectives
in torch's spelling), and sizes(mesh, "tp", N) the buffers'.

On DTensor the partition is a placement, RaggedShard(dim, key) [veScale's RaggedShard(local_units), pytorch#169320],
held by the tensors it splits; shard(mesh, "tp", dim) is the attached dimension's.  A DTensor holding one is a
RaggedTensor, a DTensor subclass whose own dispatch keeps it ragged through every op.  distribute_tensor, from_local and
redistribute below are torch's, with a Shard on an attached mesh dimension that dimension's RaggedShard; from_local
without shape= means "a capacity block" (global = local P / c_r).  ColwiseParallel, RowwiseParallel and
SequenceParallel below are torch's parallel styles over them.  A tensor that meets a RaggedShard on its mesh dimension
holds the same one or none there (R6: a plain Shard or another partition raises ValueError).

Uneven partitioning as GSPMD does it [Xu et al. 2021, "GSPMD: general and scalable parallelization for ML computation
graphs", arXiv:2105.04663, §3.3: pad a dimension to the shards' size, and mask the padding to the identity of the next
operation wherever its data could leak into valid results]:
- Padding holds the identity of its next consumer, or garbage that nothing reads unmasked.  Sources keep the identity:
  a local buffer is written with zeros past its valid extent (distribute_tensor, relay), and the collectives below
  write the identity past `valid`.
- Before an op that reduces or contracts over a partitioned dimension of a local shard (its DTensor sharding has a
  Partial output on that mesh dimension: mm, bmm, addmm, baddbmm, dot, mv, sum, amax, amin and every other), each
  input sharded there is re-masked to the Partial's identity (0 for sums and contractions, -inf for max, +inf for
  min): where(valid, x, identity), the op torch_mesh::remask.
- Count-dependent: mean over a partitioned dimension is the masked sum divided by its logical count.  An op whose
  sharding gives any other Partial there (avg, the norms, embedding's mask), or a ragged output extent P does not
  divide (a view), and convolution, argmax, argmin and equal, get the dimension whole first, by the gather below, where
  only the valid elements remain; softmax, var, sort, top-k, scans and indexing need it whole by DTensor's own rules.
- Attention: SDPA on query, key and value split along the sequence by one RaggedShard (on any mesh dimension) is ring
  attention, each source's keys past its valid extents masked (score -inf); context parallelism's ring
  (ring_attention) masks the same way over head-tail halves.  The softmax over the split is the library's partials
  merge (below).
No kernel changes: stock kernels compute garbage in padding that nothing reads unmasked.

Redistribution, each block at its capacity (static counts, never padded to a common size): Shard to Replicate
all-gathers the blocks and packs the valid elements into the logical tensor (torch_mesh::pack, a fixed-shape index from
the parts); Replicate to Shard takes this rank's valid elements into its block (torch_mesh::unpack, then its static
block); Partial to Shard unpacks with the identity and reduce-scatters the blocks; Shard to Partial places this rank's
valid elements among zeros; Shard(d) to Shard(d') goes through Replicate; Shard's scatter from a source rank is the
rooted V (the backend's exchange, MPI_Scatterv).  Context parallelism's head-tail split keeps each half at its capacity:
rank r's block is its head's valid positions, padding, its tail's, padding (context_parallel_buffers,
context_parallel_unshard, ring_attention: torch's context_parallel, its unshard and its ring, called in their place).
The ring is split-KV attention [Dao et al. 2023, Flash-Decoding]: every rank's query block gathered (one Allgatherv),
each block's attention over this rank's own keys and values its partials (the row's logsumexp and output), and the
partials reduce-scattered by the library's partials operator (ncclMeshRedOpCreatePartials in ncclMeshReduceScatterV),
each block's merged at its owner by their log-sum-exp, so the crossing follows from the role of the split axis, a
softmax over it, as a contraction's is a reduction.  On a group
of the mesh backend the MPS buffers these collectives send from and land in are window tensors (torch_mesh.empty, mesh
5ff5d699 and f9898875): the all-gather's output, this rank's block copied into its place there and the call run in
place; the reduce-scatter's input (where it does not lie in the window already, its copy) and its output; the ring's
gathered queries and its partials; so the library reads what they send where it lies.  Elsewhere, and where the window
refuses a lease, torch's own memory.

A routed partition (routed): a mixture of experts' experts as a partition's units, each token's rows sent to the ranks
holding their experts and back by the counted all-to-all (torch_mesh.all_to_all_counted: the counts the GPU wrote move
with the rows), the slots past a rank's part receiving none.

Torch entry points (torch 2.14; nothing of torch is assigned, so a tensor with no RaggedShard runs as torch runs it):
- torch.library.custom_op with register_fake and register_autograd: torch_mesh::remask, pack, unpack, allgatherv,
  reduce_scatterv.
- RaggedShard subclasses torch.distributed.tensor.placement_types.Shard and overrides the methods DTensor calls on a
  placement instance: local_shard_size_and_offset and _local_shard_size_and_offset (compute_local_shape_and_global_
  offset, the redistribute planner, the RNG), _split_tensor and _select_split_tensor (and through them the inherited
  _replicate_to_shard), _shard_tensor, _reduce_shard_tensor (Partial._reduce_shard_value calls the target's),
  _to_replicate_tensor, _to_partial_tensor and _to_new_shard_dim (the redistribution calls the source's).  It equals
  Shard(dim), as DTensorSpec requires, so torch's cached planner (_gen_transform_infos) would serve a stock spec's or
  another partition's placements for its own: redistribution here (_local_redistribute) runs torch's planner uncached
  (_gen_transform_infos_non_cached) and calls each step's placement method as redistribute_local_tensor does, one mesh
  dimension a step (never merged into a flattened collective).
- RaggedTensor subclasses torch.distributed.tensor.DTensor with its own __torch_dispatch__ (DTensor.__torch_dispatch__
  exists for subclasses; the C++ fast path serves the base class alone), to_local, redistribute, full_tensor and
  __tensor_unflatten__, whose autograd functions keep gradients ragged.  The dispatch calls, never assigns:
  DTensor._op_dispatcher.unwrap_to_op_info, sharding_propagator.propagate (on the specs with each RaggedShard a Shard,
  so strategies, costs and its cache are stock) and _adjust_shape_and_stride_args (local shapes of views and
  factories from the RaggedShard), the dispatcher's _custom_op_handlers and _random_ops (read);
  torch.utils._python_dispatch.return_and_correct_aliasing.
- DTensor RNG: a subclass of torch.distributed.tensor._random.OffsetBasedRNGTracker whose _compute_rng_offsets follows
  capacity blocks, held here per mesh (torch's own tracker is never replaced).
- ColwiseParallel, RowwiseParallel and SequenceParallel subclass torch.distributed.tensor.parallel's, overriding
  _partition_linear_fn, _partition_embedding_fn, _prepare_input_fn and _prepare_output_fn.
- Context parallelism: the blocks are _block's (MPS's SDPA returns no logsumexp), merged by the backend's partials
  merge (the library's partials operator).
All but custom_op, DTensor, Shard and the parallel styles are underscored: torch 2.14's internals, read again on an
upgrade.

A partitioned dimension may be any of its mesh's (a RaggedShard on any mesh dimension: each placement's methods take
their mesh dimension's group and coordinate, and the redistribution moves one mesh dimension a step).  Refused (on
every rank alike): _StridedShard (FSDP2 with TP, nested views), a flattened or unflattened partitioned dimension, a
BlockMask (flex attention), from_local's run_check and LocalTensor's RNG tracker on a RaggedShard (NotImplementedError).
Left as they are: the strategy costs (MeshTopoInfo, redistribute_cost), which choose among collectives, not splits.
Under torch.compile the SDPA ring is a graph break (UnsupportedOperatorException on fake tensors).

rebalance(mesh, name, every) moves the parts by the evidence of the program's steps (allocate.Balancer)."""
import contextlib
import itertools
import math

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor, UnsupportedOperatorException
from torch.distributed._local_tensor import maybe_enable_local_tracker, maybe_run_for_local_tensor
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, _random
from torch.distributed.tensor import distribute_tensor as _distribute_tensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._redistribute import _gen_transform_infos_non_cached
from torch.distributed.tensor.parallel import ColwiseParallel as _Colwise
from torch.distributed.tensor.parallel import RowwiseParallel as _Rowwise
from torch.distributed.tensor.parallel import SequenceParallel as _Sequence
from torch.distributed.tensor.placement_types import _StridedShard
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

aten = torch.ops.aten


class _Operand:
    """A partitioned dimension's capacities (fixed) and parts (a CPU tensor, written between calls).  The masks and
    indices ops read from the parts are cached per what they are, and a write rebuilds each in place (the same tensors,
    their shapes fixed by the capacities), so a step recorded before it (torch_mesh.record) reads the new parts where it
    reads the old ones: a parts change records nothing again."""

    def __init__(self, key, capacity, parts):
        self.key, self.capacity, self.units = key, tuple(int(c) for c in capacity), sum(int(p) for p in parts)
        self.parts, self.epoch, self.cache, self.builds = torch.zeros(len(self.capacity), dtype=torch.int64), 0, {}, {}
        self.write(parts)

    def write(self, parts):
        parts = tuple(int(p) for p in parts)
        if len(parts) != len(self.capacity) or sum(parts) != self.units or not all(1 <= p <= c for p, c in zip(parts, self.capacity)):
            raise ValueError(f'mesh: parts {parts} of {self.units} units within the capacities {self.capacity}: one a '
                             f'coordinate, each at least 1')
        self.parts.copy_(torch.tensor(parts))
        self.epoch += 1
        for what, held in self.cache.items():
            fresh, held = self.builds[what](parts), held if isinstance(held, tuple) else (held,)
            for old, new in zip(held, fresh if isinstance(fresh, tuple) else (fresh,)):
                old.copy_(new)

    def cached(self, what, build):
        if what not in self.cache:
            self.cache[what], self.builds[what] = build(self.parts.tolist()), build
        return self.cache[what]


_OPERANDS = {}
_NAMED = {}
_KEYS = itertools.count()


def _units(o, N):
    """An extent N's grain over a partition: P must divide N, so that every rank infers the same global extent from its
    local one (global = local P / c_r)."""
    if N % o.units:
        raise ValueError(f'mesh: an extent of {N} split over {o.units} units is no multiple of them')
    return N // o.units


def _blocks(o, N):
    """(sizes, offsets) of the local buffers of an extent N over a partition: the capacities' (static)."""
    g = _units(o, N)
    sizes = [g * c for c in o.capacity]
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
    """The logical tensor x as every rank's block laid end to end along `dim`, `identity` past each block's valid
    extent."""
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


# The collectives, each block at its capacity, the split dimension moved to the front.
_IDENTITY = {'sum': 0.0, 'avg': 0.0, 'max': -math.inf, 'min': math.inf}


def _windowed(like, group):
    """Whether the collectives' buffers like `like` on `group` are window tensors: an MPS tensor on a group of the mesh
    backend (its communicator's handle: rebalance's test), whose node's bridge leases the window (mesh 5ff5d699)."""
    return group is not None and like.device.type == 'mps' and bool(getattr(group, 'handle', 0))


def _lease(shape, like):
    """An MPS window tensor like `like` (torch_mesh.empty: a lease of the node's bridge's window, which the library reads
    and writes where it lies: rdma/NCCL.md "Memory"), made by name as 5ff5d699 made it (DTensor's dispatch runs no
    factory mode, and torch documents no MPS allocator to place it: mesh 7e6f0335); None where the window refuses the
    lease (full, or its bridge stopped: the library's error, the collective's own to raise), where aee2829b's window
    allocator took the device's own buffer."""
    from . import empty
    try:
        return empty(tuple(shape), dtype=like.dtype, device='mps')
    except RuntimeError:
        return None


def _empty(shape, like, group=None, dtype=None):
    """A window tensor (_lease) where the buffer is _windowed and the window holds it, else torch.empty; like `like`,
    of `dtype` where given."""
    like = like if dtype is None else like.new_empty(0, dtype=dtype)
    made = _lease(shape, like) if _windowed(like, group) else None
    return made if made is not None else torch.empty(shape, dtype=like.dtype, device=like.device)


def _in_window(y):
    """Whether contiguous `y` lies in the window already (torch_mesh.address), so a collective reads it where it lies."""
    from . import address
    return y.is_contiguous() and address(y) != 0


@torch.library.custom_op('torch_mesh::allgatherv', mutates_args=())
def allgatherv(x: torch.Tensor, dim: int, sizes: list[int], group_name: str) -> torch.Tensor:
    """The blocks gathered from x's block (contiguous along dim), an opaque op, so a compiled graph calls it as torch's
    functional collectives are called.  On the mesh backend the list is a split of `out`, so it is one MPI_Allgatherv
    into `out` in place (ncclMeshAllGatherV: backend.py allgather).  Where the output is a window tensor (_windowed),
    this rank's block is copied into its place in it first and gathered from there, the call in place (nccl.h: sendbuff
    rank's segment of recvbuff; mesh f9898875), so what it sends lies in the window; else out of place from x's block."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if not _windowed(x, group):
        y = x.movedim(dim, 0).contiguous()
        out = _empty((sum(sizes),) + y.shape[1:], y)
        dist.all_gather(list(out.split(sizes)), y, group=group)
        return out.movedim(0, dim).contiguous()
    y = x.movedim(dim, 0)
    out = _empty((sum(sizes),) + y.shape[1:], y, group)
    blocks = list(out.split(sizes))
    mine = blocks[dist.get_rank(group)].copy_(y)
    dist.all_gather(blocks, mine, group=group)
    return out.movedim(0, dim).contiguous()


@torch.library.custom_op('torch_mesh::reduce_scatterv', mutates_args=())
def reduce_scatterv(x: torch.Tensor, dim: int, sizes: list[int], rank: int, op: str, group_name: str) -> torch.Tensor:
    """The blocks reduce-scattered out of place from x (contiguous along dim), this rank's block, an opaque op.  On
    the mesh backend the list is a split of `y`, so it is one MPI_Reduce_scatter reading `y` in place
    (ncclMeshReduceScatterV: backend.py reduce_scatter).  Where the buffers are window tensors (_windowed), `y` is x's
    block where it lies in the window already, else its contiguous copy into a window tensor (mesh f9898875), and the
    output a window tensor of its own, so what it sends lies in the window and the output holds no lease of the input."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    y = x.movedim(dim, 0)
    held = _lease(y.shape, y) if _windowed(x, group) and not _in_window(y) else None
    y = held.copy_(y) if held is not None else y.contiguous()
    out = _empty((sizes[rank],) + y.shape[1:], y, group)
    dist.reduce_scatter(out, list(y.split(sizes)), op=getattr(dist.ReduceOp, op.upper()), group=group)
    return out.movedim(0, dim).contiguous()


@allgatherv.register_fake
def _(x, dim, sizes, group_name):
    shape = list(x.shape)
    shape[dim] = sum(sizes)
    return x.new_empty(shape)


@reduce_scatterv.register_fake
def _(x, dim, sizes, rank, op, group_name):
    shape = list(x.shape)
    shape[dim] = sizes[rank]
    return x.new_empty(shape)


def _allgatherv(x, dim, sizes, group):
    return allgatherv(x, dim, list(sizes), group.group_name)


def _reduce_scatterv(x, dim, sizes, rank, op, group):
    return reduce_scatterv(x, dim, list(sizes), rank, op, group.group_name)


class RaggedShard(Shard):
    """Shard(dim) over a partitioned mesh dimension, `key` its operand (attach): each rank's local buffer its capacity
    block, its valid elements (its part's) first.  DTensor calls these methods on the placement instance.  It equals
    Shard(dim), as DTensorSpec requires of a shard placement (_verify_shard_order); this module compares the key too
    (_same)."""

    def __init__(self, dim, key):
        super().__init__(dim)
        self.key = key

    def __reduce__(self):
        return RaggedShard, (self.dim, self.key)

    def __repr__(self):
        return f'RaggedShard(dim={self.dim}, key={self.key!r})'

    def __str__(self):
        return f'RS({self.dim}, {self.key})'

    def _operand(self, num_chunks):
        o = _OPERANDS[self.key]
        if num_chunks != len(o.capacity):
            raise NotImplementedError(f'mesh: {self!r} over {num_chunks} coordinates, a flattened or unflattened '
                                      f'partitioned mesh dimension of {len(o.capacity)}')
        return o

    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(self, curr_local_size, num_chunks, rank):
        sizes, offsets = _blocks(self._operand(num_chunks), curr_local_size)
        return sizes[rank], offsets[rank]

    def _local_shard_size_and_offset(self, curr_local_size, num_chunks, rank):
        return self.local_shard_size_and_offset(curr_local_size, num_chunks, rank)

    def _split_tensor(self, tensor, num_chunks, *, with_padding=True, contiguous=True):
        sizes = _blocks(self._operand(num_chunks), tensor.size(self.dim))[0]
        blocks = unpack(tensor, self.dim, self.key, 0.0).split(sizes, self.dim)
        return [b.contiguous() if contiguous else b for b in blocks], [0] * len(blocks)

    @maybe_run_for_local_tensor
    def _select_split_tensor(self, tensor, num_chunks, index, *, with_padding=True, contiguous=True, clone=True):
        sizes, offsets = _blocks(self._operand(num_chunks), tensor.size(self.dim))
        block = unpack(tensor, self.dim, self.key, 0.0).narrow(self.dim, offsets[index], sizes[index])
        return block.contiguous() if contiguous or clone else block

    def _shard_tensor(self, tensor, mesh, mesh_dim, src_data_rank=0):
        """Scatterv of the blocks from src_data_rank (src_data_rank None: this rank's own block)."""
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)
        n, rank = mesh.size(mesh_dim), coordinate[mesh_dim]
        if src_data_rank is None:
            return self._select_split_tensor(tensor, n, rank, with_padding=False, contiguous=True)
        sizes = _blocks(self._operand(n), tensor.size(self.dim))[0]
        out = _empty(tensor.shape[:self.dim] + (sizes[rank],) + tensor.shape[self.dim + 1:], tensor)
        blocks = self._split_tensor(tensor, n)[0] if rank == src_data_rank else None
        dist.scatter(out, blocks, group=mesh.get_group(mesh_dim), group_src=src_data_rank)
        return out

    def _reduce_shard_tensor(self, tensor, mesh, reduce_op, mesh_dim):
        """Partial to Shard: the logical partial unpacked with the reduction's identity, reduce-scattered."""
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor
        sizes = _blocks(self._operand(mesh.size(mesh_dim)), tensor.size(self.dim))[0]
        padded = unpack(tensor, self.dim, self.key, _IDENTITY[reduce_op])
        return _reduce_scatterv(padded, self.dim, sizes, coordinate[mesh_dim], reduce_op, mesh.get_group(mesh_dim))

    def _to_replicate_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
        sizes = _blocks(self._operand(mesh.size(mesh_dim)), current_logical_shape[self.dim])[0]
        return pack(_allgatherv(local_tensor, self.dim, sizes, mesh.get_group(mesh_dim)), self.dim, self.key).contiguous()

    def _to_partial_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
        """Shard to Partial("sum"): this rank's valid elements at their logical places, zeros elsewhere."""
        sizes, offsets = _blocks(self._operand(mesh.size(mesh_dim)), current_logical_shape[self.dim])
        rank, total = mesh.get_coordinate()[mesh_dim], sum(sizes)
        zeros = [local_tensor.new_zeros(local_tensor.shape[:self.dim] + (s,) + local_tensor.shape[self.dim + 1:])
                 for s in (offsets[rank], total - offsets[rank] - sizes[rank])]
        valid = remask(local_tensor, self.dim, self.key, rank, 0.0)
        return pack(torch.cat([zeros[0], valid, zeros[1]], self.dim), self.dim, self.key)

    def _to_new_shard_dim(self, local_tensor, mesh, mesh_dim, current_logical_shape, new_shard_dim):
        """Shard(d) to Shard(d') on a partitioned mesh dimension: through Replicate."""
        full = self._to_replicate_tensor(local_tensor, mesh, mesh_dim, current_logical_shape)
        return RaggedShard(new_shard_dim, self.key)._replicate_to_shard(full, mesh, mesh_dim, mesh.get_coordinate()[mesh_dim])


# The operand's API.
def attach(mesh, **dims):
    """Partition mesh dimensions by name: attach(mesh, tp=(capacity, parts)), a capacity and a part a coordinate, the
    same on every rank.  Attaching a dimension again makes a new operand: tensors split by the old one keep it."""
    for name, (capacity, parts) in dims.items():
        if name not in (mesh.mesh_dim_names or ()) or len(capacity) != mesh.size(mesh.mesh_dim_names.index(name)):
            raise ValueError(f'mesh: partition {name}: {len(capacity)} capacities, one a coordinate of a mesh dimension of '
                             f'{mesh.mesh_dim_names}')
        key = f'{name}#{next(_KEYS)}'
        _OPERANDS[key] = _Operand(key, capacity, parts)
        _NAMED[mesh, name] = key


def operand(mesh, name):
    """The partition of mesh dimension `name`: .capacity, .units, .parts (the tensor), .key."""
    return _OPERANDS[_NAMED[mesh, name]]


def write(mesh, **dims):
    """New parts for partitioned dimensions, between calls: write(mesh, tp=(5, 11)).  Shapes do not change; the
    tensors a program holds keep their old parts' layout until relay rewrites them."""
    for name, parts in dims.items():
        operand(mesh, name).write(parts)


def split(mesh, name, N):
    """The valid blocks of an extent N along partitioned mesh dimension `name` under the current parts, in rank order."""
    o = operand(mesh, name)
    return [_units(o, N) * p for p in o.parts.tolist()]


def sizes(mesh, name, N):
    """The local buffers' sizes of an extent N along partitioned mesh dimension `name` (its capacities')."""
    return tuple(_blocks(operand(mesh, name), N)[0])


def shard(mesh, name, dim):
    """The placement of tensor dimension `dim` split along mesh dimension `name`: its RaggedShard where `name` is
    attached, else torch's Shard."""
    key = _NAMED.get((mesh, name))
    return RaggedShard(dim, key) if key else Shard(dim)


@torch.no_grad()
def relay(tensor, full):
    """Writes DTensor `tensor`'s local buffer from its logical value `full` under the current parts: on a partitioned
    dimension its valid elements, zeros past them (a plain tensor: all of `full`).  In place: no reallocation."""
    if not isinstance(tensor, DTensor):
        return tensor.copy_(full)
    mesh, value, coordinate = tensor.device_mesh, full, tensor.device_mesh.get_coordinate()
    for i, p in enumerate(tensor.placements):
        if p.is_partial():
            raise NotImplementedError('mesh: relay of a Partial placement')
        if p.is_shard():
            value = p._select_split_tensor(value, mesh.size(i), coordinate[i], with_padding=False, clone=False)
    tensor._local_tensor.copy_(value)


def _ragged(placements):
    return any(isinstance(p, RaggedShard) for p in placements)


def _same(a, b):
    """Placements alike, a RaggedShard's key and type included (torch's equality takes a RaggedShard for Shard(dim))."""
    a, b = tuple(a), tuple(b)
    return len(a) == len(b) and all(type(p) is type(q) and p == q and getattr(p, 'key', None) == getattr(q, 'key', None)
                                    for p, q in zip(a, b))


def _local_redistribute(local, current, target, async_op=False):
    """torch's redistribute_local_tensor: its planner's steps (_gen_transform_infos_non_cached; its cache takes a
    RaggedShard for the Shard it equals, one partition's for another's), each the method of the placement torch calls
    there.  A shard of the same dimension under another partition (or none) goes through Replicate."""
    mesh = current.mesh
    if not mesh._is_current_rank_part_of_mesh():
        return local
    mid = tuple(Replicate() if p.is_shard() and q.is_shard() and p.dim == q.dim and not _same((p,), (q,)) else p
                for p, q in zip(current.placements, target.placements))
    if not _same(mid, current.placements):
        local, current = _local_redistribute(local, current, _with(current, mid), async_op), _with(current, mid)
    for info in _gen_transform_infos_non_cached(current, target):
        i, (p, q) = info.mesh_dim, info.src_dst_placements
        if mesh.size(i) == 1 or _same((p,), (q,)):
            continue
        if isinstance(p, _StridedShard) or isinstance(q, _StridedShard):
            raise NotImplementedError('mesh: _StridedShard with a partitioned mesh dimension')
        if q.is_replicate():
            local = p._reduce_value(local, mesh, i) if p.is_partial() else p._to_replicate_tensor(local, mesh, i, info.logical_shape)
        elif q.is_shard():
            if p.is_partial():
                local = p._reduce_shard_value(local, mesh, i, q)
            elif p.is_replicate():
                local = q._replicate_to_shard(local, mesh, i, mesh._sym_get_coordinate(i))
            elif p.dim != q.dim:
                local = p._to_new_shard_dim(local, mesh, i, info.logical_shape, q.dim)
        elif p.is_replicate():
            local = q._partition_value(local, mesh, i)
        elif p.is_shard() and type(q) is Partial and q.reduce_op == 'sum':
            local = p._to_partial_tensor(local, mesh, i, info.logical_shape)
        else:
            raise RuntimeError(f'redistribute from {p} to {q} not supported yet')
        if not async_op and isinstance(local, funcol.AsyncCollectiveTensor):
            local = local.wait()
    return local


def _placements(mesh, placements, ndim, held=None):
    """placements with every dimension non-negative and each Shard on an attached mesh dimension that dimension's
    RaggedShard: the one `held` (a tensor's own placements) has there, else the attached one."""
    out, names = [], mesh.mesh_dim_names
    for i, p in enumerate(placements):
        if isinstance(p, Shard):
            d = p.dim + ndim if p.dim < 0 else p.dim
            if isinstance(p, RaggedShard):
                key = p.key
            elif held is not None and isinstance(held[i], RaggedShard):
                key = held[i].key
            else:
                key = _NAMED.get((mesh, names[i])) if names else None
            p = RaggedShard(d, key) if key else Shard(d)
        out.append(p)
    return tuple(out)


def _wrap(local, spec, requires_grad=None):
    """local as a DTensor of spec: a RaggedTensor where it holds a RaggedShard."""
    cls = RaggedTensor if _ragged(spec.placements) else DTensor
    return cls(local, spec, requires_grad=local.requires_grad if requires_grad is None else requires_grad)


def _global_info(tensor, mesh, placements):
    """from_local's inverse: the global shape and stride of a local tensor, global = local P / c_r on a RaggedShard,
    local times the coordinates on a Shard."""
    coordinate = mesh.get_coordinate()
    shape, stride = list(tensor.size()), list(tensor.stride())
    for i in reversed(range(len(placements))):
        p = placements[i]
        if isinstance(p, _StridedShard):
            raise NotImplementedError('mesh: _StridedShard with a partitioned mesh dimension')
        if not p.is_shard():
            continue
        d, local = p.dim, shape[p.dim]
        if isinstance(p, RaggedShard):
            o = p._operand(mesh.size(i))
            size, rest = divmod(local * o.units, o.capacity[coordinate[i]])
        else:
            size, rest = local * mesh.size(i), 0
        for j in range(len(stride)):
            if j != d and stride[j] >= stride[d] and local:
                stride[j], left = divmod(stride[j] * size, local)
                rest += left
        if rest:
            raise ValueError(f'mesh: a local extent {local} on mesh dimension {i} is no capacity block of {p!r}: '
                             f'from_local takes shape= and stride=')
        shape[d] = size
    return torch.Size(shape), tuple(stride)


class _FromLocal(torch.autograd.Function):
    """torch's _FromTorchTensor whose output is a RaggedTensor of the global shape given."""

    @staticmethod
    def forward(ctx, local, mesh, placements, shape, stride, grad_placements):
        ctx.mesh, ctx.placements, ctx.grad_placements = mesh, placements, grad_placements
        ctx.set_materialize_grads(False)
        spec = DTensorSpec(mesh, placements, tensor_meta=TensorMeta(shape, stride, local.dtype))
        return _wrap(local.view_as(local), spec, local.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None, None, None, None
        placements = ctx.grad_placements or tuple(Replicate() if p.is_partial() else p for p in ctx.placements)
        placements = _placements(ctx.mesh, placements, grad.ndim, ctx.placements)
        local = grad._local_tensor
        if not _same(grad.placements, placements):
            local = _local_redistribute(local, grad._spec, DTensorSpec(ctx.mesh, placements, tensor_meta=grad._spec.tensor_meta))
        return local, None, None, None, None, None


class _ToLocal(torch.autograd.Function):
    """torch's _ToTorchTensor whose gradient is a RaggedTensor where the tensor was one."""

    @staticmethod
    def forward(ctx, input, grad_placements):
        ctx.spec, ctx.grad_placements = input._spec, grad_placements
        return input._local_tensor.view_as(input._local_tensor)

    @staticmethod
    def backward(ctx, grad):
        spec = ctx.spec
        placements = spec.placements if ctx.grad_placements is None else _placements(spec.mesh, ctx.grad_placements, grad.ndim, spec.placements)
        meta = TensorMeta(spec.shape, _global_info(grad, spec.mesh, placements)[1], grad.dtype)
        return _wrap(grad, DTensorSpec(spec.mesh, placements, tensor_meta=meta), grad.requires_grad), None


class _Redistribute(torch.autograd.Function):
    """torch's Redistribute over RaggedShards: its backward redistributes the gradient to the input's placements."""

    @staticmethod
    def forward(ctx, input, placements, async_op):
        ctx.spec, ctx.async_op = input._spec, async_op
        target = DTensorSpec(input.device_mesh, placements, tensor_meta=input._spec.tensor_meta)
        return _wrap(_local_redistribute(input._local_tensor, input._spec, target, async_op), target, input.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        mesh = ctx.spec.mesh
        placements = tuple(Replicate() if q.is_partial() and (p.is_shard() or p.is_replicate()) else q
                           for p, q in zip(grad.placements, ctx.spec.placements))
        local = _local_redistribute(grad._local_tensor, grad._spec, DTensorSpec(mesh, placements, tensor_meta=grad._spec.tensor_meta),
                                    ctx.async_op)
        meta = TensorMeta(grad.shape, grad.stride(), local.dtype)
        return _wrap(local, DTensorSpec(mesh, placements, tensor_meta=meta), local.requires_grad), None, None


def distribute_tensor(tensor, device_mesh, placements, *, src_data_rank=0):
    """torch's distribute_tensor, a Shard on an attached mesh dimension its RaggedShard: each rank's capacity block,
    zeros past its valid elements (scattered from src_data_rank, or this rank's own block where it is None)."""
    placements = _placements(device_mesh, placements, tensor.ndim)
    if not _ragged(placements) or isinstance(tensor, DTensor):
        return _distribute_tensor(tensor, device_mesh, placements, src_data_rank=src_data_rank)
    if not tensor.is_leaf:
        raise RuntimeError('`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!')
    local = tensor.detach()
    if local.device.type != device_mesh.device_type and not local.is_meta:
        local = local.to(device_mesh.device_type)
    for i, p in enumerate(placements):
        if isinstance(p, Shard):
            local = p._shard_tensor(local, device_mesh, i, src_data_rank)
        elif isinstance(p, (Replicate, Partial)):
            local = Replicate._make_replicate_tensor(local, device_mesh, i, src_data_rank)
            if isinstance(p, Partial):
                local = p._partition_value(local, device_mesh, i)
        else:
            raise NotImplementedError(f'mesh: distributing over {p}')
    spec = DTensorSpec(device_mesh, placements, tensor_meta=TensorMeta(tensor.size(), tensor.stride(), tensor.dtype))
    return RaggedTensor(local.requires_grad_(tensor.requires_grad), spec, requires_grad=tensor.requires_grad)


def from_local(local_tensor, device_mesh, placements, *, run_check=False, shape=None, stride=None, grad_placements=None):
    """torch's DTensor.from_local, a Shard on an attached mesh dimension its RaggedShard; without shape= and stride=
    the local tensor is a capacity block (global = local P / c_r)."""
    placements = _placements(device_mesh, placements, local_tensor.ndim)
    if not _ragged(placements):
        return DTensor.from_local(local_tensor, device_mesh, placements, run_check=run_check, shape=shape, stride=stride,
                                  grad_placements=grad_placements)
    if run_check:
        raise NotImplementedError('mesh: from_local\'s run_check on a RaggedShard')
    if local_tensor.device.type != device_mesh.device_type and not local_tensor.is_meta:
        local_tensor = local_tensor.to(device_mesh.device_type)
    if shape is None or stride is None:
        shape, stride = _global_info(local_tensor, device_mesh, placements)
    return _FromLocal.apply(local_tensor, device_mesh, placements, torch.Size(shape), tuple(stride),
                            None if grad_placements is None else tuple(grad_placements))


def redistribute(tensor, placements, *, async_op=False):
    """tensor.redistribute(placements=...), a Shard on an attached mesh dimension its RaggedShard (the tensor's own
    there first)."""
    placements = _placements(tensor.device_mesh, placements, tensor.ndim, tensor.placements)
    if _same(tensor.placements, placements):
        return tensor
    if not (_ragged(placements) or _ragged(tensor.placements)):
        return DTensor.redistribute(tensor, placements=placements, async_op=async_op)
    for current, target in zip(tensor.placements, placements):
        if target.is_partial() and current != target and not (isinstance(current, Shard) and type(target) is Partial
                                                               and target.reduce_op == 'sum'):
            raise RuntimeError(f'Can not redistribute from {current} to {target}, only Shard to Partial(sum) '
                               f'redistribution is supported!')
    return _Redistribute.apply(tensor, placements, async_op)


class RaggedTensor(DTensor):
    """A DTensor holding a RaggedShard.  Its ops are dispatched here (_dispatch) and its to_local, redistribute and
    full_tensor keep its gradients ragged; every DTensor one makes that holds a RaggedShard is one."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return _dispatch(func, args, kwargs or {})

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        t = DTensor.__tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride)
        return _wrap(t._local_tensor, t._spec, t.requires_grad)

    from_local = staticmethod(from_local)

    def to_local(self, *, grad_placements=None):
        if not torch.is_grad_enabled():
            result = self._local_tensor
        else:
            result = _ToLocal.apply(self, None if grad_placements is None else tuple(grad_placements))
        if getattr(self, '_is_param', False) and not getattr(result, '_is_param', False):
            if result is self._local_tensor:
                result = result.view_as(result)
            result._is_param = True
        return result

    def redistribute(self, device_mesh=None, placements=None, *, async_op=False, forward_dtype=None, backward_dtype=None):
        if device_mesh not in (None, self.device_mesh):
            raise NotImplementedError('Cross device mesh comm not supported yet!')
        if placements is None:
            raise RuntimeError('placements is needed for redistribute!')
        if forward_dtype is not None or backward_dtype is not None:
            raise NotImplementedError('mesh: redistribute\'s forward_dtype and backward_dtype on a RaggedShard')
        return redistribute(self, placements, async_op=async_op)

    def full_tensor(self, *, grad_placements=None):
        whole = self.redistribute(placements=[Replicate()] * self.device_mesh.ndim)
        return whole.to_local(grad_placements=grad_placements)


# The dispatch: DTensor's (sharding propagation, redistribution, the local op, the wrap), each Shard on a partitioned
# mesh dimension its RaggedShard, each input sharded on a partitioned mesh dimension on which the output is Partial
# first re-masked to the Partial's identity.
_REMASKED = ('sum', 'max', 'min')
_WHOLE = (aten.convolution.default, aten.convolution_backward.default, aten.argmax.default, aten.argmin.default,
          aten.equal.default)


def _map(f, x):
    return pytree.tree_map_only(DTensorSpec, f, x)


def _specs(x):
    if isinstance(x, DTensorSpec):
        return [x]
    if isinstance(x, (list, tuple)):
        return [s for y in x for s in _specs(y)]
    if isinstance(x, dict):
        return [s for y in x.values() for s in _specs(y)]
    return []


def _with(spec, placements):
    placements = tuple(placements)
    if _same(placements, spec.placements):
        return spec
    return DTensorSpec(spec.mesh, placements, tensor_meta=spec.tensor_meta)


def _plain(spec):
    return _with(spec, (Shard(p.dim) if isinstance(p, RaggedShard) else p for p in spec.placements))


def _retag(spec, keys):
    placements = []
    for i, p in enumerate(spec.placements):
        if i in keys:
            if isinstance(p, _StridedShard):
                raise NotImplementedError('mesh: _StridedShard (FSDP2 with TP, nested views) on a partitioned mesh dimension')
            if isinstance(p, Shard) and not isinstance(p, RaggedShard):
                p = RaggedShard(p.dim, keys[i])
        placements.append(p)
    return _with(spec, placements)


def _schema(schema, f):
    return OpSchema(schema.op, _map(f, schema.args_schema), _map(f, schema.kwargs_schema), schema_info=schema.schema_info)


def _keys(specs):
    """{mesh dimension: the key of its RaggedShard} over the inputs; ValueError where inputs meeting there differ."""
    keys = {}
    for s in specs:
        for i, p in enumerate(s.placements):
            if isinstance(p, RaggedShard) and keys.setdefault(i, p.key) != p.key:
                raise ValueError(f'mesh: tensors partitioned by {keys[i]} and {p.key} meet on mesh dimension {i}')
    for s in specs:
        for i, p in enumerate(s.placements):
            if i in keys and (isinstance(p, (Shard, _StridedShard)) and not isinstance(p, RaggedShard)):
                raise ValueError(f'mesh: a {p} meets {keys[i]}\'s RaggedShard on mesh dimension {i} (design/heterogeneity.md '
                                 f'R6: tensors that meet split their shared extent alike)')
    return keys


def _unfit(specs, keys):
    """The partitioned mesh dimensions the op's result cannot hold ragged: a Partial other than a sum, max or min
    there, or a RaggedShard of an extent P does not divide."""
    dims = set()
    for s in specs:
        for i, p in enumerate(s.placements):
            if i not in keys:
                continue
            if p.is_partial() and not (type(p) is Partial and p.reduce_op in _REMASKED):
                dims.add(i)
            elif isinstance(p, RaggedShard) and s.shape[p.dim] % _OPERANDS[p.key].units:
                dims.add(i)
    return dims


def _whole(tree, dims):
    """Each DTensor in tree replicated on the partitioned mesh dimensions `dims`: its valid elements, gathered."""
    def whole(t):
        if not _ragged(t.placements[i] for i in dims):
            return t
        return redistribute(t, [Replicate() if i in dims and isinstance(p, RaggedShard) else p for i, p in enumerate(t.placements)])
    with torch.no_grad():
        return pytree.tree_map_only(DTensor, whole, tree)


def _dispatch(op_call, args, kwargs):
    handler = _HANDLERS.get(op_call)
    if handler is not None:
        return handler(op_call, args, kwargs)
    return _propagated(op_call, args, kwargs)


def _propagated(op_call, args, kwargs):
    d = DTensor._op_dispatcher
    sp = d.sharding_propagator
    info = d.unwrap_to_op_info(op_call, args, kwargs)
    keys = _keys(_specs(info.flat_args_schema) + _specs(info.schema.kwargs_schema))
    if not keys:
        return DTensor.__torch_dispatch__(op_call, (), args, kwargs)
    if op_call in _WHOLE:
        return op_call(*_whole(args, set(keys)), **_whole(kwargs, set(keys)))
    if op_call in d._custom_op_handlers:
        return d._custom_op_handlers[op_call](op_call, args, kwargs)
    schema = info.schema
    info.schema = _schema(schema, _plain)
    try:
        sp.propagate(info)
    except NotImplementedError:
        if torch._C._dispatch_has_kernel_for_dispatch_key(op_call.name(), torch._C.DispatchKey.CompositeImplicitAutograd):
            return op_call.decompose(*args, **kwargs)
        raise
    out = info.output_sharding
    spec = _map(lambda s: _retag(s, keys), out.output_spec)
    whole = _unfit(_specs(spec), keys) | (set(keys) if op_call in sp.op_to_scalar_shape_adjuster else set())
    if whole:
        return op_call(*_whole(args, whole), **_whole(kwargs, whole))
    flat, local = info.flat_args_schema, list(info.local_args)
    if out.needs_redistribute:
        target = _schema(out.redistribute_schema, lambda s: _retag(s, keys))
        if out.use_val_from_redistribute_schema and target.op in sp.op_to_shape_and_stride_idx and _ragged(spec.placements):
            target = sp._adjust_shape_and_stride_args(spec.tensor_meta, target, spec)
        flat = pytree.tree_leaves(target.args_schema) if info.args_tree_spec is not None else target.args_schema
        local = [(t if _same(s.placements, w.placements) else _local_redistribute(t, s, w)) if isinstance(s, DTensorSpec)
                 else (w if out.use_val_from_redistribute_schema else s)
                 for s, w, t in zip(info.flat_args_schema, flat, local)]
        if out.use_val_from_redistribute_schema:
            local.extend(flat[len(info.flat_args_schema):])
    mesh = info.compute_mesh
    coordinate = mesh.get_coordinate()
    for i, key in keys.items():
        partial = next((s.placements[i] for s in _specs(spec) if s.placements[i].is_partial()), None)
        if partial is None:
            continue
        for j, s in enumerate(flat):
            if isinstance(s, DTensorSpec) and isinstance(s.placements[i], RaggedShard):
                local[j] = remask(local[j], s.placements[i].dim % local[j].dim(), key, coordinate[i], _IDENTITY[partial.reduce_op])
    local_args = pytree.tree_unflatten(local, info.args_tree_spec) if info.args_tree_spec is not None else local
    op = out.redistribute_schema.op if out.needs_redistribute and out.redistribute_schema.op != op_call else op_call
    if op_call in d._random_ops and _random.is_rng_supported_mesh(mesh) and not local_args[0].is_meta:
        generator = info.local_kwargs.pop('generator', None)
        with _tracker(mesh)._distribute_region(args[0]._spec, generator):
            result = op(*local_args, **info.local_kwargs)
    else:
        result = op(*local_args, **info.local_kwargs)
    if spec is None:
        return result
    if schema.is_inplace_op():
        if not _same(args[0]._spec.placements, spec.placements):
            args[0]._spec = spec
        return args[0]
    if schema.is_out_variant_op():
        outs = [kwargs[a.name] for a in op_call._schema.arguments if a.is_out]
        for t, s in zip(outs, spec if isinstance(spec, tuple) else (spec,)):
            t._spec = s
        return tuple(outs) if len(outs) > 1 else outs[0]
    ret = _wrapped(result, spec)
    return return_and_correct_aliasing(op_call, args, kwargs, ret) if op_call._schema._is_view_op() else ret


def _wrapped(result, spec):
    if isinstance(result, torch.Tensor):
        if spec is None:
            return result
        return _wrap(result, spec, result.requires_grad)
    if isinstance(result, (list, tuple)):
        out = [_wrapped(r, s) for r, s in zip(result, spec)]
        return tuple(out) if isinstance(result, tuple) else out
    return result


def _named(op_call, args, kwargs):
    return dict(zip((a.name for a in op_call._schema.arguments), args), **kwargs)


def _mean(op_call, args, kwargs):
    """mean over a partitioned dimension: the masked sum over the logical count."""
    named = _named(op_call, args, kwargs)
    x, dims, keepdim, dtype = named['self'], named.get('dim'), named.get('keepdim', False), named.get('dtype')
    dims = sorted({d % x.ndim for d in dims} if dims else range(x.ndim))
    if not isinstance(x, DTensor) or not any(isinstance(p, RaggedShard) and p.dim in dims for p in x.placements):
        return _propagated(op_call, args, kwargs)
    return torch.sum(x, dims, keepdim=keepdim, dtype=dtype) / math.prod(x.shape[d] for d in dims)


def _linear_backward(op_call, args, kwargs):
    """aten.linear_backward (MPS's linear, which DTensor has no sharding of) as the matmuls it is: grad @ weight,
    grad^T @ input, grad summed."""
    named = _named(op_call, args, kwargs)
    x, grad, weight, mask = named['self'], named['grad_output'], named['weight'], named['output_mask']
    rows = grad.flatten(0, -2)
    return (grad @ weight if mask[0] else None, rows.t() @ x.flatten(0, -2) if mask[1] else None,
            rows.sum(0) if mask[2] else None)


def _spec_like(spec, shape, dtype):
    stride = torch.empty(shape, device='meta').stride()
    return DTensorSpec(spec.mesh, spec.placements, tensor_meta=TensorMeta(torch.Size(shape), stride, dtype))


def _attention(op_call, args, kwargs):
    """SDPA on query, key and value split along the sequence (dimension 2) by one RaggedShard, on any mesh dimension
    (its others replicated or splitting the batch or the heads, the three tensors placed alike): ring attention over
    their capacity blocks, contiguous shares, along that dimension's group (_ring_attention).  torch.compile's tracing
    runs ops on fake tensors, and the ring (its collectives, the operand's host-side tables) runs only on real ones:
    under fake tensors the op is refused as unsupported, which dynamo takes as a graph break."""
    named = _named(op_call, args, kwargs)
    query, key, value = named['query'], named['key'], named['value']
    spec = query._spec
    ragged = [i for i, p in enumerate(query.placements) if isinstance(p, RaggedShard) and p.dim % query.ndim == 2]
    if len(ragged) != 1 or not all(_same(t.placements, query.placements) for t in (key, value)) or any(
            p.is_partial() or (p.is_shard() and p.dim % query.ndim > 1) for i, p in enumerate(query.placements) if i != ragged[0]):
        return _propagated(op_call, args, kwargs)
    held = query.placements[ragged[0]]
    if any(isinstance(t._local_tensor, FakeTensor) for t in (query, key, value)):
        raise UnsupportedOperatorException(op_call)
    if (named.get('attn_mask') is not None or named.get('dropout_p', 0.0) or named.get('dropout_mask') is not None
            or named.get('enable_gqa', False)):
        raise NotImplementedError('mesh: ragged SDPA takes no attn_mask, dropout or enable_gqa')
    out, lse = _ring_attention(spec.mesh.get_group(ragged[0]), held.key, query._local_tensor, key._local_tensor,
                               value._local_tensor, named.get('is_causal', False), named.get('scale'), 1)
    out = RaggedTensor(out, _spec_like(spec, query.shape[:-1] + value.shape[-1:], out.dtype), requires_grad=False)
    if op_call is aten._scaled_dot_product_attention_math_for_mps.default:
        return out, None
    return out, RaggedTensor(lse, _spec_like(spec, query.shape[:-1], lse.dtype), requires_grad=False)


_HANDLERS = {aten.mean.default: _mean, aten.mean.dim: _mean, aten.linear_backward.default: _linear_backward,
             aten._scaled_dot_product_flash_attention_for_cpu.default: _attention,
             aten._scaled_dot_product_attention_math_for_mps.default: _attention}


class _Offsets(_random.OffsetBasedRNGTracker):
    """torch's offset-based DTensor RNG tracker over capacity blocks: stock's (start, end) offset increments are k times
    the first shard's elements for shard k and the tensor's elements, which past equal shards would run the larger
    blocks into the next op's offsets.  Here block k (stock's row-major order over the shard grid) starts past the
    blocks before it and the op ends past the last, each rounded up to 4 as stock rounds."""

    def _compute_rng_offsets(self, spec):
        mesh = spec.mesh
        extents = [[N] for N in spec.shape]
        for i, p in enumerate(spec.placements):
            if isinstance(p, _StridedShard):
                raise NotImplementedError('mesh: DTensor RNG over _StridedShard with a partitioned mesh dimension')
            if p.is_shard():
                n = mesh.size(i)
                extents[p.dim] = [p._local_shard_size_and_offset(e, n, r)[0] for e in extents[p.dim] for r in range(n)]
        starts, end = [], 0
        for index in itertools.product(*(range(len(e)) for e in extents)):
            starts.append(end)
            end = (end + math.prod(e[j] for e, j in zip(extents, index)) + 3) // 4 * 4
        coordinate = [mesh._sym_get_coordinate(i) for i in range(mesh.ndim)]
        return starts[_random._calc_shard_linear_idx(*_random._calc_shard_info(coordinate, spec))], end

    @contextlib.contextmanager
    def _distribute_region(self, spec, generator=None):
        if maybe_enable_local_tracker(self._device.type, self.distribute_region_enabled, spec, generator) is not None:
            raise NotImplementedError("mesh: LocalTensor's DTensor RNG on a RaggedShard")
        with super()._distribute_region(spec, generator):
            yield


_TRACKERS = {}


def _tracker(mesh):
    """This module's RNG tracker of mesh (its device states synchronized from rank 0 when made, as torch's is), on or
    off as torch's own is."""
    tracker = _TRACKERS.get(mesh)
    if tracker is None:
        tracker = _TRACKERS[mesh] = _Offsets(mesh)
    if _random._rng_tracker is not None:
        tracker.distribute_region_enabled = _random._rng_tracker.distribute_region_enabled
    return tracker


# torch's tensor-parallel styles over the partition.
def _prepare(x, layouts, desired, mesh):
    if not isinstance(x, DTensor):
        x = from_local(x, mesh, layouts)
    return redistribute(x, desired, async_op=True) if layouts != desired else x


def _parameter(param, mesh, placement, src_data_rank):
    return nn.Parameter(distribute_tensor(param, mesh, [placement], src_data_rank=src_data_rank),
                        requires_grad=param.requires_grad)


class ColwiseParallel(_Colwise):
    """torch's ColwiseParallel; on a partitioned mesh dimension the weight's rows (and the bias) are RaggedShard(0),
    the output's last dimension each rank's capacity block of columns."""

    def _partition_linear_fn(self, name, module, device_mesh):
        for n, param in module.named_parameters():
            module.register_parameter(n, _parameter(param, device_mesh, Shard(0), self.src_data_rank))

    def _partition_embedding_fn(self, name, module, device_mesh):
        for n, param in module.named_parameters():
            module.register_parameter(n, _parameter(param, device_mesh, Shard(1), self.src_data_rank))

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        return _prepare(inputs[0], input_layouts, desired_input_layouts, device_mesh)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = redistribute(outputs, output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs


class RowwiseParallel(_Rowwise):
    """torch's RowwiseParallel; on a partitioned mesh dimension the weight's columns are RaggedShard(1) and the input's
    last dimension each rank's capacity block (the bias replicated)."""

    def _partition_linear_fn(self, name, module, device_mesh):
        module.register_parameter('weight', _parameter(module.weight, device_mesh, Shard(1), self.src_data_rank))
        if getattr(module, 'bias', None) is not None:
            module.register_parameter('bias', _parameter(module.bias, device_mesh, Replicate(), self.src_data_rank))

    def _partition_embedding_fn(self, name, module, device_mesh):
        for n, param in module.named_parameters():
            module.register_parameter(n, _parameter(param, device_mesh, Shard(0), self.src_data_rank))

    _prepare_input_fn = staticmethod(ColwiseParallel._prepare_input_fn)
    _prepare_output_fn = staticmethod(ColwiseParallel._prepare_output_fn)


class SequenceParallel(_Sequence):
    """torch's SequenceParallel; on a partitioned mesh dimension the sequence is RaggedShard(sequence_dim)."""

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        x = inputs[0]
        if isinstance(x, DTensor):
            return redistribute(x, sequence_sharding, async_op=True)
        if isinstance(x, torch.Tensor):
            return from_local(x, device_mesh, sequence_sharding)
        raise ValueError(f'expecting input of {mod} to be a torch.Tensor or DTensor, but got {x}')


# Context parallelism (torch/distributed/tensor/experimental/_context_parallel): each half at its capacity.
def _halves(o, S, load_balance):
    """Head-tail (2) where load_balance and 2P divides the sequence; else contiguous shares (1)."""
    return 2 if load_balance and S % (2 * o.units) == 0 else 1


def _positions(key, rank, S, halves, device):
    """Rank `rank`'s block of a sequence of S split over the partition in `halves` (2: head-tail, rank r holding
    positions [H_r, H_r + a_r) and their mirror [S - H_r - a_r, S - H_r), a_r = (S/2) p_r / P, H_r the heads of the
    ranks before it; 1: contiguous shares): (the logical position of each element of its capacity block, 0 in padding;
    which are valid).  Its causal work is a_r S, linear in its share."""
    o = _OPERANDS[key]

    def build(p):
        h = S // halves // o.units * o.capacity[rank]
        spans = _spans(o, rank, S, halves, p)
        index = torch.cat([t for lo, hi in spans for t in (torch.arange(lo, hi), torch.zeros(h - hi + lo, dtype=torch.int64))])
        valid = torch.cat([torch.arange(h) < hi - lo for lo, hi in spans])
        return index.to(device), valid.to(device)
    return o.cached(('positions', rank, S, halves, str(device)), build)


def _spans(o, rank, S, halves, parts=None):
    """The logical positions [lo, hi) rank `rank` holds in each half of its block (_positions), from `parts` (the
    operand's own where None)."""
    p = o.parts.tolist() if parts is None else parts
    g = S // halves // o.units
    H, a = g * sum(p[:rank]), g * p[rank]
    return [(H, H + a)] if halves == 1 else [(H, H + a), (S - H - a, S - H)]


def context_parallel_buffers(mesh, name, buffers, seq_dims, load_balance=True):
    """torch's context_parallel's sharding of its buffers over partitioned mesh dimension `name`: each buffer's capacity
    block of this rank, its valid positions (head-tail where load_balance and 2P divides the sequence), zeros past
    them."""
    o, rank, shards = operand(mesh, name), mesh.get_local_rank(name), []
    for b, dim in zip(buffers, seq_dims):
        if not isinstance(b, torch.Tensor):
            raise NotImplementedError('mesh: a BlockMask buffer (flex attention) on a partitioned mesh dimension')
        S = b.size(dim)
        _units(o, S)
        index, valid = _positions(o.key, rank, S, _halves(o, S, load_balance), b.device)
        shards.append(torch.where(_along(valid, dim, b.dim()), b.index_select(dim, index), 0))
    return shards


@torch.no_grad()
def context_parallel_unshard(mesh, name, buffers, seq_dims, load_balance=True):
    """torch's context_parallel_unshard over partitioned mesh dimension `name`: every rank's capacity block gathered
    (Allgatherv) and the valid positions put back in sequence order."""
    o, rank = operand(mesh, name), mesh.get_local_rank(name)
    local = buffers[0].shape[seq_dims[0]]
    S, rest = divmod(local * o.units, o.capacity[rank])
    if rest:
        raise ValueError(f'mesh: {local} positions on rank {rank} are no capacity block of {o.capacity}')
    halves = _halves(o, S, load_balance)
    lengths, offsets = _blocks(o, S)

    def build(parts):
        """Where each logical position lies in the gathered blocks, from the parts on the host (no device sync per
        call)."""
        order = torch.empty(S, dtype=torch.int64)
        for r in range(len(lengths)):
            index, valid = _positions(o.key, r, S, halves, 'cpu')
            order[index[valid]] = offsets[r] + valid.nonzero().squeeze(1)
        return order.to(buffers[0].device)
    order = o.cached(('order', S, halves, str(buffers[0].device)), build)
    group = mesh.get_group(name)
    return [_allgatherv(b, dim, lengths, group).index_select(dim, order) for b, dim in zip(buffers, seq_dims)]


def ring_attention(mesh, name, query, key, value, is_causal=False, scale=None, load_balance=True):
    """This rank's query block's attention over every rank's keys and values, the blocks along dimension 2 (the
    sequence) as context_parallel_buffers made them over partitioned mesh dimension `name`: (output, each query row's
    logsumexp), padded query rows garbage that the unshard drops."""
    o = operand(mesh, name)
    S = query.size(2) * o.units // o.capacity[mesh.get_local_rank(name)]
    return _ring_attention(mesh.get_group(name), o.key, query, key, value, is_causal, scale, _halves(o, S, load_balance))


SCORES = 1 << 26


def _block(query, key, value, scale=None, queries=None, keys=None, valid=None):
    """softmax(q k^T scale) v and each query row's logsumexp (MPS's SDPA returns none), the scores SCORES at a time: keys
    where `valid` (a bool a key) is False masked, and where `queries` (each query's logical position) is given each key
    past its query (`keys`: each key's)."""
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    rows, n = query.shape[-2], key.shape[-2]
    step = max(1, SCORES // (query[..., 0, 0].numel() * n))
    out = query.new_empty(query.shape[:-1] + value.shape[-1:])
    lse = query.new_empty(query.shape[:-1], dtype=torch.float32)
    kt = key.transpose(-2, -1)
    for at in range(0, rows, step):
        scores = (query[..., at:at + step, :] @ kt).float() * scale
        if queries is not None:
            scores.masked_fill_(keys > queries[at:at + step, None], -math.inf)
        if valid is not None:
            scores.masked_fill_(~valid, -math.inf)
        top = torch.logsumexp(scores, -1, keepdim=True)
        lse[..., at:at + step] = top.squeeze(-1)
        out[..., at:at + step, :] = torch.exp(scores - top).to(value.dtype) @ value
    return out, lse


def _ring_attention(group, key_name, query, key, value, is_causal=False, scale=None, halves=1):
    """Ring attention over capacity blocks as split-KV partials: every rank's query block gathered (one Allgatherv,
    static counts, into a window tensor on the mesh backend: _empty, this rank's copied into its place there and the
    call run in place, mesh f9898875); each block's attention over this rank's own keys and values written as its
    owner's rows of the partials (backend.partial; keys past their valid extents masked, a causal one by the logical
    positions of its queries and keys, a half of the block attending only to the halves of this rank's keys it can see,
    none where it sees none); and the partials reduce-scattered by the library's partials operator, each owner's rows
    merged at it by their log-sum-exp (ProcessGroupMesh.merge_partials, ncclMeshReduceScatterV).  This rank's own block
    is computed while the gather is in flight.  Padded query rows compute garbage that the unshard drops.  Forward
    only."""
    from .backend import ProcessGroupMesh, attention, partial
    if not isinstance(group, ProcessGroupMesh):
        raise TypeError('mesh: the ring\'s merge is the mesh backend\'s partials operator (torch_mesh::merge_partials)')
    o = _OPERANDS[key_name]
    if is_causal and query.size(2) != key.size(2):
        raise NotImplementedError('is_causal requires the same query and context sequence lengths')
    if not is_causal and halves == 2:
        raise RuntimeError('Load balancing requires `is_causal=True`.')
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    S = query.size(2) * o.units // o.capacity[rank]
    lengths, lead, width = _blocks(o, S)[0], tuple(query.shape[:2]), 2 + value.size(-1)
    row = query.numel() // query.size(2)
    blocks = list(_empty((sum(lengths) * row,), query, group).split([s * row for s in lengths]))
    blocks[rank].view(query.shape).copy_(query)
    gathered = dist.all_gather(blocks, blocks[rank], group=group, async_op=True)
    rows = [math.prod(lead) * s for s in lengths]
    packed = _empty((sum(rows), width), query, group, torch.float32).zero_()
    owners = list(packed.split(rows))
    keys, valid = _positions(key_name, rank, S, halves, query.device)
    mine, step = _spans(o, rank, S, halves), lengths[rank] // halves

    def owned(j):
        """Block j's queries' attention over this rank's keys, as owner j's rows of the partials."""
        into, q = owners[j].view(lead + (lengths[j], width)), blocks[j].view(lead + (lengths[j],) + query.shape[3:])
        queries, height = _positions(key_name, j, S, halves, query.device)[0], lengths[j] // halves
        for a, (_, hi) in enumerate(_spans(o, j, S, halves)):
            seen = [b for b, (lo, _) in enumerate(mine) if not is_causal or lo < hi]
            if seen:
                at, cols = slice(a * height, (a + 1) * height), slice(seen[0] * step, (seen[-1] + 1) * step)
                into[:, :, at] = partial(*_block(q[:, :, at], key[:, :, cols], value[:, :, cols], scale,
                                                 queries[at] if is_causal else None, keys[cols], valid[cols]))
    owned(rank)
    gathered.wait()
    for j in range(size):
        if j != rank:
            owned(j)
    merged = _empty((rows[rank], width), query, group, torch.float32)
    group.merge_partials(merged, packed, rows)
    out, lse = attention(merged.view(lead + (lengths[rank], width)))
    return out.to(query.dtype), lse


def routed(mesh, name, x, experts, slots):
    """A routed range partition (a mixture of experts' expert parallelism): partitioned mesh dimension `name`'s units are
    experts, rank r holding capacity[r] slots, the first parts[r] of them experts off_r.. (off_r the parts before it)
    and the rest receiving no rows; `x` this rank's rows (T x D), `experts` each row's experts (T x K, int64), `slots`
    this rank's capacity[r] callables, slot j's rows to it.  Each (row, k) crosses to its expert's rank and back by the
    counted all-to-all (torch_mesh.all_to_all_counted: the counts the GPU wrote move with the rows, the received counts
    read on the host as they land, no host read of the splits), every rank passing the same T K.  Returns each (row,
    k)'s expert output (T x K x D), to be weighted and summed by the caller.  The rows it sends lie in window tensors
    (torch_mesh.empty), made at their first use and kept (their shapes the capacities' and T K's)."""
    from . import all_to_all_counted, counted
    o, group = operand(mesh, name), mesh.get_group(name)
    rank, world, capacity = mesh.get_local_rank(name), len(o.capacity), list(o.capacity)
    n, flat = capacity[rank], experts.flatten()
    order = flat.argsort(stable=True)
    counts = torch.zeros(o.units, dtype=torch.int64, device=x.device).scatter_add_(0, flat, torch.ones_like(flat))
    sent = torch.index_select(x, 0, order // experts.shape[1], out=_routing(o.key, 'dispatched', flat.numel(), x))
    received, landed, handle, work = all_to_all_counted(sent, unpack(counts, 0, o.key, 0.0), capacity, [n] * world, world * flat.numel(), group)
    theirs = counted(handle)
    work.wait()
    sizes = [sum(theirs[r * n + e] for r in range(world)) for e in range(n)]
    regroup = torch.repeat_interleave(torch.arange(n, device=x.device).repeat(world), landed, output_size=sum(theirs)).argsort(stable=True)
    results = _routing(o.key, 'results', received.shape[0], received)
    for slot, rows in zip(slots, regroup.split(sizes)):
        results[rows] = slot(received[rows])
    back, _, _, work = all_to_all_counted(results, theirs, [n] * world, capacity, flat.numel(), group)
    work.wait()
    out = torch.empty_like(back)
    out[order] = back
    return out.view(*experts.shape, -1)


_ROUTING = {}


def _routing(key, what, rows, like):
    """A routed partition's window tensor of `rows` rows like `like`, made at its first use (torch_mesh.empty)."""
    from . import empty
    at = (key, what, rows, like.shape[1:], like.dtype, like.device.type)
    if at not in _ROUTING:
        _ROUTING[at] = empty(rows, *like.shape[1:], dtype=like.dtype, device=like.device.type)
    return _ROUTING[at]


_BALANCING = {}


def rebalance(mesh, name='tp', every=8):
    """The parts of partitioned mesh dimension `name` that the evidence of its last `every` steps gives
    (design/heterogeneity.md R3, R5, R10, R11): call it once a step, after the step is issued, on every rank of the
    mesh alike.  The first call watches the dimension's group (torch_mesh.evidence: each rank's own work between its
    collectives there, one stretch a collective); every `every` steps after it the ranks' stretches are gathered to
    every rank and one step of the mesh repo's rdma/allocate.py Balancer (the same on every rank: R4) gives the parts,
    each at least 1, at most the units less one a peer, minimising the steps' time: the sum over their stretches of
    the slowest rank's, every stretch scaling with this dimension's parts.

    A rank computes its whole buffer, its capacity, padding and all (R14: stock kernels), so its work follows its
    capacity: the evidence measures parts where they are the capacities, and parts that move within capacities set
    only what is valid.  Returns (parts, written): written where the parts moved and fit the capacities (then written
    here, and the program relays its tensors, relay); parts that moved past a capacity are the program's to attach
    again (capacities and parts both: the shapes change, which R11 weighs), whose dimension's group keeps this
    balancer.  The parts stand once the evidence promises no gain past its resolution."""
    import allocate
    from . import evidence
    o = operand(mesh, name)
    group = mesh.get_group(name)
    if mesh.device_type == 'mps' and not getattr(group, 'handle', 0):
        raise ValueError(f'mesh: rebalance {name}: its group is not the mesh backend\'s (no communicator to watch)')
    where = (id(group), name)
    held = _BALANCING.get(where)
    n = len(o.capacity)
    if held is None:
        _BALANCING[where] = {'balancer': allocate.Balancer({name: {'parts': o.parts.tolist(), 'low': [1] * n,
                                                                   'high': [o.units - n + 1] * n}}), 'steps': 0}
        evidence.watch(group, mesh.device_type)
        return tuple(o.parts.tolist()), False
    evidence.step(group)
    held['steps'] += 1
    if held['steps'] < every:
        return tuple(held['balancer'].parts[name]), False
    held['steps'] = 0
    everyone = [None] * dist.get_world_size(group)
    dist.all_gather_object(everyone, evidence.work(group), group=group)
    held['work'] = [sorted(sum(step) for step in steps)[len(steps) // 2] if steps else None for steps in everyone]
    positions = max((len(step) for steps in everyone if steps for step in steps), default=0)
    stretches = {f'step:{k}': {'scales': {name: [1.0] * n},
                               'times': [[step[k] for step in steps if len(step) > k] if steps else None for steps in everyone]}
                 for k in range(positions)}
    held['balancer'].decisions[name]['parts'] = o.parts.tolist()
    parts = held['balancer'].observe(stretches)[name]
    written = parts != o.parts.tolist() and all(p <= c for p, c in zip(parts, o.capacity))
    if written:
        o.write(parts)
    return tuple(parts), written


def balancing(mesh, name='tp'):
    """What rebalance last saw of partitioned dimension `name`: {'parts', 'stands', 'work': each rank's median
    step work (seconds, its stretches summed) in the last window}, None before its first window."""
    held = _BALANCING.get((id(mesh.get_group(name)), name))
    if not held or 'work' not in held:
        return None
    return {'parts': held['balancer'].parts[name], 'stands': held['balancer'].stands(name), 'work': held['work']}
