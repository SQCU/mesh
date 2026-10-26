"""A partition operand the backend sets: DTensor's Shard, tensor parallelism and context parallelism split a mesh
dimension by parts the backend moves between calls by the measured residual (design/heterogeneity.md §4, R3, R14).

Stock PyTorch splits every sharded extent equally (Shard follows torch.chunk).  Here a program declares a mesh
dimension's P units, attach(mesh, tp=P), the same on every rank, and the backend gives coordinate r a part p_r >= 1 of
them, an even split, and a capacity c_r, its part.  An extent N that P divides (any
other raises ValueError, on every rank alike) is split N p_r / P per coordinate in rank order, as the logical tensor
reads, and rank r holds its share, N p_r / P elements along that dimension, with no padding: the blocks laid end to
end in rank order are the logical tensor.  The parts are a tensor operand (operand(mesh, "tp").parts) read by the ops
below when they run.  follow(*ts) moves sharded tensors onto the parts the Balancer attached, one group a move.  On raw
tensors split(mesh, "tp", N) gives the valid blocks in rank order, the ones the list collectives take
(dist.all_gather into a list of unequal outputs, dist.reduce_scatter of a list of unequal inputs: MPI's v-collectives
in torch's spelling).

On DTensor the partition is a placement, RaggedShard(dim, key) [veScale's RaggedShard(local_units), pytorch#169320],
held by the tensors it splits.  A DTensor holding one is a
RaggedTensor, a DTensor subclass whose own dispatch keeps it ragged through every op.  distribute_tensor, from_local and
redistribute below are torch's, with a Shard on an attached mesh dimension that dimension's RaggedShard; from_local
without shape= means "a capacity block" (global = local P / c_r).  ColwiseParallel, RowwiseParallel and
SequenceParallel below are torch's parallel styles over them.  A tensor that meets a RaggedShard on its mesh dimension
holds the same one or none there (R6: a plain Shard or another partition raises ValueError).

No block holds padding (each capacity is its part), so an op on a local shard runs as torch runs it, a contraction
or reduction over a partitioned dimension giving the Partial its sharding gives, with nothing masked:
- Count-dependent: mean over a partitioned dimension is the sum divided by its logical count.  An op whose sharding
  gives any other Partial there (avg, the norms, embedding's mask), or a ragged output extent P does not divide (a
  view), and convolution, argmax, argmin and equal, get the dimension whole first, by the gather below; softmax, var,
  sort, top-k, scans and indexing need it whole by DTensor's own rules.
- Attention: SDPA on query, key and value split along the sequence by one RaggedShard (on any mesh dimension) is ring
  attention, each source's keys past its valid extents masked (score -inf); context parallelism's ring
  (ring_attention) masks the same way over head-tail halves.  The softmax over the split is the library's partials
  merge (below).

Redistribution, each block its part (static counts, never padded to a common size): Shard to Replicate all-gathers
the blocks, which laid end to end are the logical tensor; Replicate to Shard takes this rank's block of it; Partial to
Shard reduce-scatters the blocks; Shard to Partial places this rank's block among zeros; Shard(d) to Shard(d') goes
through Replicate; Shard's scatter from a source rank is the rooted V (the backend's scatter, MPI_Scatterv through
ncclMeshAlltoAllV), each segment carrying its exact useful element count. Context parallelism's head-tail split keeps
each half at its capacity: rank r's block is its head's valid positions, padding, its tail's, padding
(context_parallel_buffers, context_parallel_unshard, ring_attention: torch's context_parallel, its unshard and its
ring, called in their place).
The ring is split-KV attention [Dao et al. 2023, Flash-Decoding]: every rank's query block gathered (one Allgatherv),
each block's attention over this rank's own keys and values its partials (the row's logsumexp and output), and the
partials reduce-scattered by the library's partials operator (ncclMeshRedOpCreatePartials in ncclMeshReduceScatterV),
each block's merged at its owner by their log-sum-exp, so the crossing follows from the role of the split axis, a
softmax over it, as a contraction's is a reduction.  On a group
of the mesh backend the MPS buffers these collectives send from and land in are region tensors (torch_mesh.empty, mesh
5ff5d699 and f9898875): the all-gather's output, this rank's block copied into its place there and the call run in
place; the reduce-scatter's input (where it does not lie in the region already, its copy) and its output; the ring's
gathered queries and its partials; so the library reads what they send where it lies.  Elsewhere, and where the region
refuses a lease, torch's own memory.

Torch entry points (torch 2.14; nothing of torch is assigned, so a tensor with no RaggedShard runs as torch runs it):
- torch.library.custom_op with register_fake: torch_mesh::allgatherv and reduce_scatterv.
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
their mesh dimension's group and coordinate, and the redistribution moves one mesh dimension a step).  Not implemented
(NotImplementedError on every rank alike; the code past it computes even splits or indexes another dimension's
capacities): _StridedShard (FSDP2 with TP, nested views), a flattened or unflattened partitioned dimension, a BlockMask
(flex attention) and LocalTensor's RNG tracker on a RaggedShard.  from_local's run_check on a RaggedShard broadcasts its
Replicate mesh dimensions, as torch's does, and compares no shape (a ragged one's differ by rank).
Left as they are: the strategy costs (MeshTopoInfo, redistribute_cost), which choose among collectives, not splits.
Under torch.compile the SDPA ring is a graph break (UnsupportedOperatorException on fake tensors).

The initial partition is even. evaluations() keeps a realized generation immutable while submitting functions.
Asynchronous rank-max evidence updates a proposed vector without blocking that sequence; configuration construction
applies a subsequent generation's vector.
A declared native graph supplies exact useful-delivery demands for the physical routing vectors."""
import contextlib
import itertools
import json
import math
import resource
import weakref
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor, UnsupportedOperatorException
from torch.distributed._local_tensor import maybe_enable_local_tracker, maybe_run_for_local_tensor
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, _random
from torch.distributed.tensor import distribute_tensor as _distribute_tensor
from torch.distributed.tensor._collective_utils import mesh_broadcast
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
_LINEAR_SOURCE = Path(__file__).with_name('linear.metal').read_text()


class _Operand:
    """A partitioned dimension's parts (a CPU tensor), each its coordinate's capacity, as the backend attached them;
    the masks and indices ops read from them cached per what they are."""

    def __init__(self, key, parts):
        self.key, self.capacity, self.units = key, tuple(int(p) for p in parts), sum(int(p) for p in parts)
        self.parts, self.cache = torch.tensor(self.capacity, dtype=torch.int64), {}

    def cached(self, what, build):
        if what not in self.cache:
            self.cache[what] = build(self.parts.tolist())
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


def _along(mask, dim, ndim):
    shape = [1] * ndim
    shape[dim] = -1
    return mask.view(shape)


# The collectives, each block its part, the split dimension moved to the front.


def _region_backed(like, group):
    """Whether the collectives' buffers like `like` on `group` are region tensors: an MPS tensor on a group of the mesh
    backend (its communicator's handle), whose node's bridge leases the region (mesh 5ff5d699)."""
    return group is not None and like.device.type == 'mps' and bool(getattr(group, 'handle', 0))


def _empty(shape, like, group=None, dtype=None):
    """A region tensor (torch_mesh.empty) on `like`'s device, of its dtype or `dtype`: every buffer a collective here
    sends from or lands in is a lease, and a region that refuses one is the library's error, raised."""
    from . import empty
    return empty(tuple(shape), dtype=dtype or like.dtype, device=like.device.type)


def _in_region(y):
    """Whether contiguous `y` lies in the region already (torch_mesh.address), so a collective reads it where it lies."""
    from . import address
    return y.is_contiguous() and address(y) != 0


@torch.library.custom_op('torch_mesh::allgatherv', mutates_args=())
def allgatherv(x: torch.Tensor, dim: int, sizes: list[int], group_name: str, repeats: int = 1) -> torch.Tensor:
    """A graph declaration; the compiler binds each multi-rank occurrence."""
    if len(sizes) == 1:
        return x.clone()
    raise RuntimeError('allgatherv must be lowered to an indexed crossing before execution')


@torch.library.custom_op('torch_mesh::allgatherv_out', mutates_args=('out',))
def allgatherv_out(x: torch.Tensor, out: torch.Tensor, dim: int, sizes: list[int], group_name: str,
                  repeats: int = 1) -> None:
    """Gather directly into a caller's declared destination.

    This is the same physical write program as allgatherv. The destination's
    previous readers must precede this call on the MPS stream; its next readers
    follow it. No intermediate output lease and copy into `out` are encoded.
    """
    if len(sizes) == 1:
        out.copy_(x)
    else:
        raise RuntimeError('allgatherv_out must be lowered to an indexed crossing before execution')


@allgatherv_out.register_fake
def _(x, out, dim, sizes, group_name, repeats=1):
    return None


# The generic gather and sum are data functions over the compiler's physical
# one-sided all-gather writes. Rank-sized arrays describe tensor layout; channel-
# sized arrays describe the transport, including parallel cables and relays.
# The producer writes one region segment and publishes its physical bindings.
# A single ready dispatch observes those landings, then parallel gather/fold
# readers load coherent payloads without repeating waits. The out variant binds
# a declared destination directly, avoiding a second output copy.
# Two slots may be reused only for the recurrent all-gather: consuming every
# peer's preceding contribution before publishing the next supplies a natural
# descendant observation of each old reader. It is not a general lifetime rule
# for asymmetric writes (follow uses transfer-owned deferred retirement).
_GATHER_SOURCE = r"""
template <uint N, uint CHANNELS>
struct gather_words { uint count, pre, post, rank, j; uint sizes[N], starts[N]; mesh_step_binding land[CHANNELS], publish[CHANNELS]; };

template <uint AXES, uint CHANNELS>
struct gather_publish_words {
    uint count, j, shape[AXES], strides[AXES];
    mesh_step_binding land[CHANNELS], publish[CHANNELS];
};

template <uint AXES, uint UNITS, uint COPY_BYTES, uint CHANNELS, bool CONTIGUOUS, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void gather_publish(device const metal::conditional_t<COPY_BYTES == 1, uchar, ushort>* x [[buffer(0)]], device uchar* send [[buffer(1)]],
                           device atomic_uint* arrivals [[buffer(2)]], constant gather_publish_words<AXES, CHANNELS>& w [[buffer(15)]],
                           uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]]) {
    using T = metal::conditional_t<COPY_BYTES == 1, uchar, ushort>;
    threadgroup uint last;
    for (uint q = 0; q < CHANNELS; q++) if (w.publish[q].p)
        mesh_step_enter(*w.publish[q].p, w.publish[q].v, w.publish[q].progress, g == 0 && tid == 0);
    for (uint k = g * THREADS + tid; k < w.count; k += groups * THREADS) {
        ulong at = k;
        if (!CONTIGUOUS) {
            uint element = k / UNITS;
            at = k % UNITS;
            for (uint axis = AXES; axis-- > 0;) {
                at += ulong(element % w.shape[axis]) * w.strides[axis] * UNITS;
                element /= w.shape[axis];
            }
        }
        mesh_step_write<T>(send, k, x[at]);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, groups, tid, &last)) {
        for (uint q = 0; q < CHANNELS; q++) if (w.publish[q].p)
            mesh_step_release(*w.publish[q].p, w.publish[q].a, tid, w.j);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}

// Exactly one threadgroup observes the compiled physical landing prefixes.
// The serial encoder orders the parallel consumers after this dispatch; they
// use system-coherent payload loads, without repeating these waits or fences.
template <uint N, uint CHANNELS>
inline void gather_observe(constant gather_words<N, CHANNELS>& w, uint tid) {
    if (tid == 0)
        for (uint q = 0; q < CHANNELS; q++) if (w.land[q].p) {
            mesh_step_enter(*w.land[q].p, w.land[q].v, w.land[q].progress, true);
            mesh_step_counts(*w.land[q].p, w.land[q].a, w.land[q].v, w.j);
        }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
}

template <uint N, uint CHANNELS, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void gather_ready(constant gather_words<N, CHANNELS>& w [[buffer(15)]],
                         uint tid [[thread_index_in_threadgroup]]) {
    gather_observe(w, tid);
}

template <uint COPY_BYTES, uint N, uint CHANNELS, bool WAIT, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void gather_land(device const uchar* lease [[buffer(0)]], device metal::conditional_t<COPY_BYTES == 1, uchar, ushort>* out [[buffer(1)]],
                        constant gather_words<N, CHANNELS>& w [[buffer(15)]], uint tid [[thread_index_in_threadgroup]],
                        uint g [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
    using T = metal::conditional_t<COPY_BYTES == 1, uchar, ushort>;
    if (WAIT) gather_observe(w, tid);
    uint total = 0;
    for (uint r = 0; r < N; r++) total += w.sizes[r];
    const uint row = total * w.post;
    for (uint e = g * THREADS + tid; e < w.pre * row; e += groups * THREADS) {
        const uint i = e / row, k = e % w.post;
        uint r = 0, at = e / w.post % total;
        while (at >= w.sizes[r]) at -= w.sizes[r++];
        const ulong from = w.starts[r] + (ulong(i) * w.sizes[r] + at) * w.post + k;
        out[e] = r == w.rank ? ((device const T*)lease)[from] : mesh_step_read<T>(lease, from);
    }
}

template <uint BYTES, uint N, uint CHANNELS, bool WAIT, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void sum_land(device const uchar* lease [[buffer(0)]], device metal::conditional_t<BYTES == 2, half, float>* out [[buffer(1)]],
                     constant gather_words<N, CHANNELS>& w [[buffer(15)]], uint tid [[thread_index_in_threadgroup]],
                     uint g [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
    using T = metal::conditional_t<BYTES == 2, half, float>;
    if (WAIT) gather_observe(w, tid);
    for (uint k = g * THREADS + tid; k < w.count * 2 / BYTES; k += groups * THREADS) {
        float sum = 0.0f;
        for (uint r = 0; r < N; r++)
            sum += float(r == w.rank ? ((device const T*)lease)[w.starts[r] + k] : mesh_step_read<T>(lease, w.starts[r] + k));
        out[k] = T(sum);
    }
}
"""
_GATHER_KERNELS, _GATHER_THREADS = {}, 256


def _gather_kernel(kernel, *arguments):
    """A gather/fold kernel at `arguments` (ranks and physical channels, sum's element bytes first), compiled
    once after the library's steps (fused.steps)."""
    from . import _stream, fused
    from .argmax import SOURCE
    from .streamed import instance
    name, line = instance(kernel, *arguments, _GATHER_THREADS)
    if name not in _GATHER_KERNELS:
        _GATHER_KERNELS[name] = _stream.pipeline(fused.steps() + SOURCE + '\n' + _GATHER_SOURCE + '\n' + line + '\n', name)
    return _GATHER_KERNELS[name]


def _linear_kernel(*arguments):
    """linear.metal's linear_publish at `arguments`, compiled once after the library's steps (fused.steps) and
    argmax.metal's arrival count."""
    from . import _stream, fused
    from .argmax import SOURCE
    from .streamed import instance
    name, line = instance('linear_publish', *arguments)
    if name not in _GATHER_KERNELS:
        _GATHER_KERNELS[name] = _stream.pipeline(fused.steps() + SOURCE + '\n' + _LINEAR_SOURCE + '\n' + line + '\n', name)
    return _GATHER_KERNELS[name]


def _gather_paths(fabric):
    """Node paths for the realized physical fabric, retaining parallel fragments."""
    routes = {}
    for pair, paths in fabric.paths.items():
        routes[pair] = []
        for path in paths:
            nodes = [pair[0]]
            for cable in path:
                a, b = fabric.cables[cable]['nodes']
                if nodes[-1] not in (a, b):
                    raise ValueError('gather lifetime path is disconnected from its physical cable')
                nodes.append(b if nodes[-1] == a else a)
            routes[pair].append(tuple(nodes))
    return routes


def _fused_form(x, w, dim, sizes, group_name, adds, channels):
    """A fused crossing's form for torch_mesh::fused_step, made during preparation (_stream.mm fused_maker): the
    communicator and the waits' bound; this rank and the ranks; the publication's kernel (gather_publish, or
    linear_publish where `w`), constant words, its repeat's word and grid; the landing's (sum_land or gather_land, and
    this rank's segment's start and each rank's sizes in its words); every rank's exact segment start and count
    and the output's shape; the publication's arrival count; the separate landing-prefix
    observer and physical channel extent."""
    from . import _stream
    from lifetimes import prove_recurrent_gather
    group = dist.distributed_c10d._resolve_process_group(group_name)
    n, rank = len(sizes), group.rank()
    operand = tuple(x.shape) if w is None else (x.shape[0], w.shape[0])
    dim %= len(operand)
    copy_bytes = 1 if x.element_size() == 1 else 2
    row, half = math.prod(operand[:dim] + operand[dim + 1:]), x.element_size() // copy_bytes
    counts = [s * row for s in sizes]
    starts = list(itertools.accumulate(counts, initial=0))
    # Positive contributions provide the returning dependency that proves the
    # two-slot recurrence. Empty participants have no such publication; native
    # realization must retain fresh generations for those shapes instead.
    if all(counts):
        prove_recurrent_gather(n, _gather_paths(group.fabric))
    words = ([math.prod(operand) * half, math.prod(operand[:dim]), math.prod(operand[dim + 1:]) * half, rank, 0] + list(sizes) +
             [a * (1 if adds else half) for a in starts[:-1]])
    shape = operand if adds else operand[:dim] + (sum(sizes),) + operand[dim + 1:]
    spread = lambda count: max(1, min(16, -(-count // (_GATHER_THREADS * 8))))
    groups = spread(math.prod(shape) * half)
    inline_wait = groups == 1
    land = (_gather_kernel('sum_land', x.element_size(), n, channels, inline_wait) if adds else
            _gather_kernel('gather_land', copy_bytes, n, channels, inline_wait))
    if w is None:
        publish_words = [math.prod(operand) * half, 0, *x.shape, *x.stride()]
        publish = (_gather_kernel('gather_publish', x.ndim, half, copy_bytes, channels, x.is_contiguous()), publish_words, 1,
                   spread(words[0]), _GATHER_THREADS)
    else:
        K, N, ldw = x.shape[1], w.shape[0], w.stride(0)
        if x.stride(1) != 1:
            raise ValueError('mesh: linear publication requires contiguous columns; realize a strided producer kernel')
        split = _stream.linear_split(K, N)
        publish = (_linear_kernel(4 if ldw % 4 == 0 else 1, 4 if x.stride(0) % 4 == 0 else 1, 2, split, 2, channels),
                   [x.shape[0], K, N, 0, ldw, K, x.stride(0), x.stride(0), 0], 8, -(-N // 2), split * 32)
    return (group.handle, _stream.polls, n, rank, *publish, land, words, 4, groups, _GATHER_THREADS,
            starts, counts, list(shape), torch.zeros(1, dtype=torch.int32, device='mps'),
            -1 if inline_wait else _gather_kernel('gather_ready', n, channels), channels)


from . import _stream as _native  # noqa: E402  (torch_mesh::fused_step's library)

_native.fused_maker(_fused_form)


@torch.library.register_fake('torch_mesh::indexed_step')
def _(x, w, index):
    from .declarations import _INDEXED_SHAPES
    return x.new_empty(_INDEXED_SHAPES[index])


@torch.library.register_fake('torch_mesh::indexed_step_out')
def _(x, w, out, index):
    return None


@torch.library.register_fake('torch_mesh::fused_step')
def _(x, w, dim, sizes, group_name, adds, repeats):
    operand = tuple(x.shape) if w is None else (x.shape[0], w.shape[0])
    dim %= len(operand)
    return x.new_empty(operand if adds else operand[:dim] + (sum(sizes),) + operand[dim + 1:])


@torch.library.register_fake('torch_mesh::fused_step_out')
def _(x, w, out, dim, sizes, group_name, adds, repeats):
    return None


@torch.library.custom_op('torch_mesh::reduce_scatterv', mutates_args=())
def reduce_scatterv(x: torch.Tensor, dim: int, sizes: list[int], rank: int, op: str, group_name: str) -> torch.Tensor:
    """The blocks reduce-scattered out of place from x (contiguous along dim), this rank's block, an opaque op.  On
    the mesh backend the list is a split of `y`, so it is one MPI_Reduce_scatter reading `y` in place
    (ncclMeshReduceScatterV: backend.py reduce_scatter).  `y` is x's block where it lies in the region already, else its
    contiguous copy into a region tensor (mesh f9898875), and the output a region tensor of its own, so what it sends
    lies in the region and the output holds no lease of the input."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    y = x.movedim(dim, 0)
    held = _empty(y.shape, y) if not _in_region(y) else None
    y = held.copy_(y) if held is not None else y.contiguous()
    out = _empty((sizes[rank],) + y.shape[1:], y, group)
    dist.reduce_scatter(out, list(y.split(sizes)), op=getattr(dist.ReduceOp, op.upper()), group=group)
    return out.movedim(0, dim).contiguous()


@allgatherv.register_fake
def _(x, dim, sizes, group_name, repeats=1):
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


def _row(x, dim):
    """The elements of x a step along `dim` holds (its shape's product but that dimension's)."""
    dim %= len(x.shape)
    return math.prod(s for k, s in enumerate(x.shape) if k != dim)


def _gather_writes(counts, dtype):
    """One exact routed WRITE per nonempty source/destination pair, in native order.

    Source rank segments are immutable publications. Each final segment and
    each relay generation has its own landing interval; the common physical
    write planner chooses all stripes and relay paths.
    """
    from .backend import WRITE, Call
    return [Call(WRITE, count, dtype, root=source, destination=destination, fused=True)
            for source, count in enumerate(counts) if count
            for destination in range(len(counts)) if source != destination]


def gathered(x, dim, sizes, group):
    """The exact writes made by a generic gather, declared before execution.

    No rank segment is padded to a transport page. Non-mesh backends retain
    their own collective declaration.
    """
    from .backend import ALLGATHER, Call
    counts = [s * _row(x, dim) for s in sizes]
    if not _region_backed(x, group):
        return [Call(ALLGATHER, x.numel(), x.dtype, counts=counts)]
    return _gather_writes(counts, x.dtype)


def summed(x, group):
    """Exact routed partial writes followed by the generic data function's sum."""
    return _gather_writes([x.numel()] * group.size(), x.dtype)


def scattered(x, dim, sizes, op='sum'):
    """reduce_scatterv's group of an operand like `x` along `dim`, each rank's block sizes[r] long: one reduce-scatter
    (torch_mesh.plan)."""
    from .backend import REDUCE_SCATTER, Call
    return [Call(REDUCE_SCATTER, x.numel(), x.dtype, op=getattr(dist.ReduceOp, op.upper()), counts=[s * _row(x, dim) for s in sizes])]


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
        blocks = tensor.split(_blocks(self._operand(num_chunks), tensor.size(self.dim))[0], self.dim)
        return [b.contiguous() if contiguous else b for b in blocks], [0] * len(blocks)

    @maybe_run_for_local_tensor
    def _select_split_tensor(self, tensor, num_chunks, index, *, with_padding=True, contiguous=True, clone=True):
        sizes, offsets = _blocks(self._operand(num_chunks), tensor.size(self.dim))
        block = tensor.narrow(self.dim, offsets[index], sizes[index])
        return block.contiguous() if contiguous or clone else block

    def _shard_tensor(self, tensor, mesh, mesh_dim, src_data_rank=0):
        """Realize an exact-count scatter of the declared blocks at configuration.

        Mesh ranks bind exact original/final strided views. A foreign initial
        allocation is imported once into registered storage; regional inputs
        are never packed. Other distributed backends retain their scatter API.
        """
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)
        n, rank, group = mesh.size(mesh_dim), coordinate[mesh_dim], mesh.get_group(mesh_dim)
        if src_data_rank is None:
            return self._select_split_tensor(tensor, n, rank, with_padding=False, contiguous=True)
        sizes, rest = _blocks(self._operand(n), tensor.size(self.dim))[0], tensor.movedim(self.dim, 0).shape[1:]
        direct = _scatter_spans(tensor, self.dim, sizes, rank, src_data_rank, group)
        if direct is not None:
            return direct
        counts = [s * math.prod(rest) for s in sizes]
        at = list(itertools.accumulate(counts, initial=0))
        out, blocks = _empty((counts[rank],), tensor, group), None
        if rank == src_data_rank:
            pieces = self._split_tensor(tensor, n, contiguous=False)[0]
            if self.dim == 0 and tensor.is_contiguous() and _in_region(tensor):
                blocks = list(tensor.reshape(-1).split(counts))
            else:
                staged = _empty((at[-1],), tensor, group)
                for q, piece in enumerate(pieces):
                    staged[at[q]:at[q] + counts[q]].view((sizes[q],) + rest).copy_(piece.movedim(self.dim, 0))
                blocks = list(staged.split(counts))
        outgoing = counts if rank == src_data_rank else [0] * n
        incoming = [counts[rank] if r == src_data_rank else 0 for r in range(n)]
        matrix = [counts[b] if a == src_data_rank else 0 for a in range(n) for b in range(n)]
        with _configuration_exchange(group, outgoing, incoming, tensor.dtype, matrix):
            dist.scatter(out, blocks, group=group, group_src=src_data_rank)
        return out[:counts[rank]].view((sizes[rank],) + rest).movedim(0, self.dim)

    def _reduce_shard_tensor(self, tensor, mesh, reduce_op, mesh_dim):
        """Partial to Shard: the blocks of the logical partial reduce-scattered."""
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor
        sizes = _blocks(self._operand(mesh.size(mesh_dim)), tensor.size(self.dim))[0]
        return _reduce_scatterv(tensor, self.dim, sizes, coordinate[mesh_dim], reduce_op, mesh.get_group(mesh_dim))

    def _to_replicate_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
        """The blocks gathered (allgatherv), which laid end to end are the logical tensor."""
        sizes = _blocks(self._operand(mesh.size(mesh_dim)), current_logical_shape[self.dim])[0]
        return _allgatherv(local_tensor, self.dim, sizes, mesh.get_group(mesh_dim))

    def _to_partial_tensor(self, local_tensor, mesh, mesh_dim, current_logical_shape):
        """Shard to Partial("sum"): this rank's block at its logical place, zeros elsewhere."""
        sizes, offsets = _blocks(self._operand(mesh.size(mesh_dim)), current_logical_shape[self.dim])
        rank, total = mesh.get_coordinate()[mesh_dim], sum(sizes)
        zeros = [local_tensor.new_zeros(local_tensor.shape[:self.dim] + (s,) + local_tensor.shape[self.dim + 1:])
                 for s in (offsets[rank], total - offsets[rank] - sizes[rank])]
        return torch.cat([zeros[0], local_tensor, zeros[1]], self.dim)

    def _to_new_shard_dim(self, local_tensor, mesh, mesh_dim, current_logical_shape, new_shard_dim):
        """Shard(d) to Shard(d') on a partitioned mesh dimension: through Replicate."""
        full = self._to_replicate_tensor(local_tensor, mesh, mesh_dim, current_logical_shape)
        return RaggedShard(new_shard_dim, self.key)._replicate_to_shard(full, mesh, mesh_dim, mesh.get_coordinate()[mesh_dim])


# The operand's API.
def attach(mesh, **dims):
    """Partition mesh dimensions by name: attach(mesh, tp=units), the units every extent the program splits along the
    dimension is a multiple of, the same on every rank.  The backend splits them evenly over the coordinates in rank
    order (the first units mod n one more)."""
    for name, units in dims.items():
        n = mesh.size(mesh.mesh_dim_names.index(name)) if name in (mesh.mesh_dim_names or ()) else 0
        if not 0 < n <= units:
            raise ValueError(f'mesh: partition {name}: {units} units over a mesh dimension of {mesh.mesh_dim_names}, '
                             f'at least one a coordinate')
        _attach(mesh, name, [units // n + (r < units % n) for r in range(n)])


def _attach(mesh, name, parts):
    """`parts` attached to mesh dimension `name` as a new operand: tensors split by the old one keep it (follow)."""
    key = f'{name}#{next(_KEYS)}'
    _OPERANDS[key] = _Operand(key, parts)
    _NAMED[mesh, name] = key


def operand(mesh, name):
    """The partition of mesh dimension `name`: .capacity, .units, .parts (the tensor), .key."""
    return _OPERANDS[_NAMED[mesh, name]]


@contextlib.contextmanager
def _configuration_exchange(group, outgoing, incoming, dtype, matrix=None):
    """A counted transfer belonging to configuration, preserving the function declaration."""
    from .backend import ALLTOALL, Call
    if hasattr(group, 'configuration'):
        call = Call(ALLTOALL, sum(outgoing), dtype, counts=outgoing + incoming, matrix=matrix)
        with group.configuration([[call]]):
            yield
    else:
        yield


_MIGRATION_SOURCE = r"""
template <uint CHANNELS>
struct migration_words {
    ulong source, destination, row, rows, source_stride, destination_stride, frame;
    uint j, unused;
    mesh_step_binding land[CHANNELS], publish[CHANNELS];
};
template <uint BYTES> using migration_bits = metal::conditional_t<BYTES == 16, uint4,
    metal::conditional_t<BYTES == 8, uint2, metal::conditional_t<BYTES == 4, uint,
    metal::conditional_t<BYTES == 2, ushort, uchar>>>>;

template <uint CHANNELS>
inline void migration_observe(constant migration_words<CHANNELS>& w, uint tid) {
    if (tid == 0)
        for (uint q = 0; q < CHANNELS; q++) if (w.land[q].p) {
            mesh_step_enter(*w.land[q].p, w.land[q].v, w.land[q].progress, true);
            mesh_step_counts(*w.land[q].p, w.land[q].a, w.land[q].v, w.j);
        }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
}
template <uint CHANNELS, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void migration_ready(constant migration_words<CHANNELS>& w [[buffer(15)]],
                            uint tid [[thread_index_in_threadgroup]]) { migration_observe(w, tid); }

template <uint CHANNELS, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void migration_publish_existing(constant migration_words<CHANNELS>& w [[buffer(15)]],
                                       uint tid [[thread_index_in_threadgroup]]) {
    for (uint q = 0; q < CHANNELS; q++) if (w.publish[q].p) {
        mesh_step_enter(*w.publish[q].p, w.publish[q].v, w.publish[q].progress, tid == 0);
        mesh_step_release(*w.publish[q].p, w.publish[q].a, tid);
    }
}

template <uint BYTES, uint CHANNELS, ulong ROW, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void migration_local(device const migration_bits<BYTES>* original [[buffer(0)]],
                            device migration_bits<BYTES>* destination [[buffer(1)]],
                            constant migration_words<CHANNELS>& w [[buffer(15)]],
                            uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]],
                            uint groups [[threadgroups_per_grid]]) {
    for (ulong i = ulong(g) * THREADS + tid; i < w.rows * ROW; i += ulong(groups) * THREADS)
        destination[w.destination + (i / ROW) * w.destination_stride + i % ROW] =
            original[w.source + (i / ROW) * w.source_stride + i % ROW];
}

template <uint AXES> struct migration_view_words {
    ulong count, shape[AXES], source_stride[AXES], destination_stride[AXES];
};
template <uint AXES, uint UNITS, uint BYTES, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void migration_local_views(device const migration_bits<BYTES>* source [[buffer(0)]],
                                  device migration_bits<BYTES>* destination [[buffer(1)]],
                                  constant migration_view_words<AXES>& w [[buffer(15)]],
                                  uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]],
                                  uint groups [[threadgroups_per_grid]]) {
    for (ulong k = ulong(g) * THREADS + tid; k < w.count; k += ulong(groups) * THREADS) {
        ulong element = k / UNITS, a = k % UNITS, b = k % UNITS;
        for (uint axis = AXES; axis-- > 0;) {
            const ulong index = element % w.shape[axis];
            a += index * w.source_stride[axis] * UNITS;
            b += index * w.destination_stride[axis] * UNITS;
            element /= w.shape[axis];
        }
        destination[b] = source[a];
    }
}
"""


def _migration_kernel(kernel, *arguments):
    from . import _stream, fused
    from .argmax import SOURCE
    from .streamed import instance
    name, line = instance(kernel, *arguments, _GATHER_THREADS)
    if name not in _GATHER_KERNELS:
        _GATHER_KERNELS[name] = _stream.pipeline(fused.steps() + SOURCE + '\n' + _MIGRATION_SOURCE + '\n' + line, name)
    return _GATHER_KERNELS[name]


def _migration_words(run, frame_bytes, vector, j=0):
    values = (run.source_offset // vector, run.destination_offset // vector, run.row_bytes // vector,
              run.rows, run.source_stride // vector, run.destination_stride // vector, frame_bytes // vector)
    return [word for value in values for word in (value & 0xFFFFFFFF, value >> 32)] + [j, 0]


def _follow_bytes(placement, local, logical_shape, logical_stride, key, rank, group, publish_boundary):
    """AOT affine byte writes from original rows into final destination rows.

    Each RowWrite is one compiled logical message, independently of row count.
    The native plan gathers registration phases and local pitches once, fixes
    equal exact-byte units along every selected path, and owns its immutable
    span tables. No payload is packed, padded, or unpacked. Relays have distinct
    compact leases; their receives directly supply their outgoing DMA reads.
    """
    from migration import compile_rows
    from . import address, fused

    if (not _region_backed(local, group) or
            tuple(logical_stride) != tuple(torch.empty(tuple(logical_shape), device='meta').stride())):
        return None
    dim, size = placement.dim % local.ndim, local.element_size()
    shape = list(local.shape)
    shape[dim] = logical_shape[dim]
    program = compile_rows(shape, dim, _OPERANDS[placement.key].capacity, _OPERANDS[key].capacity, size)
    if not local.is_contiguous():
        return None
    if tuple(local.shape) != program.source_shapes[rank] or (local.numel() and not address(local)):
        raise ValueError('mesh: byte migration requires the declared regional source shape')
    original = local.view(-1)
    incoming = [run for run in program.runs if run.destination == rank]
    extent = math.prod(program.destination_shapes[rank])
    alias = (len(incoming) == 1 and incoming[0].source == rank and incoming[0].destination_offset == 0 and
             incoming[0].bytes == extent * size and
             (incoming[0].rows == 1 or incoming[0].source_stride == incoming[0].row_bytes))
    result = (original.narrow(0, incoming[0].source_offset // size, extent).view(program.destination_shapes[rank])
              if alias else _empty(program.destination_shapes[rank], local, group))
    source_bytes, destination_bytes = original.view(torch.uint8), result.view(-1).view(torch.uint8)
    channels = len(group.fabric.adjacent[rank])
    jobs = []
    # Allocate leases and compile kernels before any plan can present receives.
    for run in program.runs:
        if run.source == run.destination:
            if run.source != rank or alias:
                continue
            vector = math.gcd(16, run.source_offset, run.destination_offset, run.row_bytes,
                              run.source_stride, run.destination_stride, address(local), address(result))
            words = _migration_words(run, run.bytes, vector)
            groups = max(1, -(-(run.bytes // vector) // (_GATHER_THREADS * 8)))
            jobs.append((run, _migration_kernel('migration_local', vector, channels, run.row_bytes // vector),
                         words, groups, None, 0))
            continue
        if rank == run.source:
            first = source_bytes.narrow(0, run.source_offset, run.row_bytes)
            stride = run.source_stride
            kernel = _migration_kernel('migration_publish_existing', channels)
        elif rank == run.destination:
            first = destination_bytes.narrow(0, run.destination_offset, run.row_bytes)
            stride = run.destination_stride
            kernel = _migration_kernel('migration_ready', channels)
        else:
            relay = _empty((run.bytes,), local, group, torch.uint8)
            first, stride, kernel = relay.narrow(0, 0, run.row_bytes), run.row_bytes, None
        jobs.append((run, kernel, _migration_words(run, run.row_bytes, 1), 1, first, stride))
    schedule = _stream.Schedule(mps=True)
    consumers = []
    for run, kernel, words, groups, first, stride in jobs:
        if run.source == run.destination:
            schedule.compute(kernel, [local, result], fused._encode_words(words, channels, None, None),
                             15, groups, 1, _GATHER_THREADS, 1 | (2 << 32))
            continue
        crossing = fused.write_rows(schedule, first, [first], [(run.source, run.destination)], group,
                                    row_bytes=run.row_bytes, rows=run.rows, stride=stride)
        if rank == run.source:
            schedule.compute(kernel, [], fused._encode_words(words, channels, None, crossing),
                             15, 1, 1, _GATHER_THREADS, 0)
        elif rank == run.destination:
            consumers.append((kernel, words, crossing))
    for kernel, words, crossing in consumers:
        schedule.compute(kernel, [], fused._encode_words(words, channels, crossing, None),
                         15, 1, 1, _GATHER_THREADS, 0)
    schedule.finish()
    schedule.enqueue()
    return result


def _copy_view(source, destination):
    """Necessary local relocation, one indexed bit-preserving data kernel."""
    if not source.numel():
        return
    if source.device.type != 'mps':
        destination.copy_(source)
        return
    from . import _stream
    width = math.gcd(source.element_size(), 4)
    units = source.element_size() // width
    values = [source.numel() * units, *source.shape, *source.stride(), *destination.stride()]
    words = [word for value in values for word in (value & 0xFFFFFFFF, value >> 32)]
    kernel = _migration_kernel('migration_local_views', source.ndim, units, width)
    groups = max(1, -(-(source.numel() * units) // (_GATHER_THREADS * 8)))
    _stream.encode(kernel, [source, destination], words, 15, groups, 1, _GATHER_THREADS)


def _scatter_spans(tensor, dim, sizes, rank, source, group):
    """Initial input-domain import, then the same exact native span WRITE plan."""
    if not hasattr(group, 'prepare_writes'):
        return None
    from . import _stream
    from .backend import Call, WRITE
    dim %= tensor.ndim
    offsets = list(itertools.accumulate(sizes, initial=0))
    original = tensor
    if rank == source and not _in_region(tensor):
        # Native DMA addresses currently belong to the registered region. This
        # is one input-domain import, independent of destination count/layout.
        original = _empty(tensor.shape, tensor, group)
        _copy_view(tensor, original)
    shape = list(tensor.shape)
    shape[dim] = sizes[rank]
    result = (original.narrow(dim, offsets[rank], sizes[rank]) if rank == source
              else _empty(shape, tensor, group))
    deliveries, calls = [], []
    row = _row(tensor, dim)
    for destination, count in enumerate(sizes):
        if destination == source or not count * row:
            continue
        view = (original.narrow(dim, offsets[destination], count) if rank == source
                else result if rank == destination else _empty((count * row,), tensor, group))
        deliveries.append((view, source, destination))
        calls.append(Call(WRITE, count * row * tensor.element_size(), torch.uint8, root=source, destination=destination))
    if deliveries:
        prepared = group.prepare_writes(deliveries)
        if tensor.device.type == 'mps':
            _stream.publish_boundary()
        with group.configuration([calls]):
            prepared.realize()
            prepared.enqueue()
    return result


def _follow_spans(placement, local, logical_shape, key, rank, group, publish_boundary):
    """Arbitrary regional strides use exact AOT span WRITE bindings."""
    if not hasattr(group, 'prepare_writes'):
        return None
    dim = placement.dim % local.ndim
    old, new = _OPERANDS[placement.key], _OPERANDS[key]
    before_sizes, before = _blocks(old, logical_shape[dim])
    after_sizes, after = _blocks(new, logical_shape[dim])
    if local.shape[dim] != before_sizes[rank]:
        raise ValueError('strided migration source does not match its attached partition')
    shape = list(local.shape)
    shape[dim] = after_sizes[rank]
    result = _empty(shape, local, group)
    deliveries = []
    for source in range(len(before)):
        for destination in range(len(after)):
            first = max(before[source], after[destination])
            last = min(before[source] + before_sizes[source], after[destination] + after_sizes[destination])
            if first >= last:
                continue
            count = last - first
            if source == destination:
                if source == rank:
                    _copy_view(local.narrow(dim, first - before[source], count),
                               result.narrow(dim, first - after[destination], count))
                continue
            if rank == source:
                view = local.narrow(dim, first - before[source], count)
            elif rank == destination:
                view = result.narrow(dim, first - after[destination], count)
            else:
                extent = math.prod(local.shape[:dim] + local.shape[dim + 1:]) * count
                view = _empty((extent,), local, group)
            deliveries.append((view, source, destination))
    if deliveries:
        prepared = group.prepare_writes(deliveries)
        if local.device.type == 'mps':
            publish_boundary()
        # Repartition is a configuration transition. Ordinary native stream
        # completion orders next consumers; no application receipt is emitted.
        prepared.realize()
        prepared.enqueue()
    return result


@torch.no_grad()
def follow(*tensors):
    """DTensors `tensors` (of one mesh), each split along each partitioned mesh dimension by the operand attached there
    now (after the Balancer moved it), as a list. Contiguous region layouts use compact affine exact-byte writes:
    original source rows to final destination rows, with distinct compact relay leases. A RowWrite remains one
    logical message regardless of its row count. Configuration-time metadata fixes common packet cuts at row and
    registration boundaries; execution performs no metadata exchange, payload packing, padding, or unpacking.
    An unchanged contiguous local interval is aliased; other retained local ranges are copied directly once.

    Other mesh layouts bind original strided source views and final destination views through exact indexed WRITE
    plans. Their span descriptors are agreed once at configuration, without materializing payloads. Local retained
    values use one indexed bit-preserving relocation kernel; every relay has its own compact lease.

    Other distributed backends use one declared MPI_Alltoallv per tensor of the rows that
    change rank alone, each piece an exact compact segment of a staging lease, sent once and copied once into its
    place in the tensor's new block, a region tensor contiguous in the tensor's own layout; the rows that stay are
    copied into their place there, crossing nothing. Each tensor has fresh transfer leases. Dropping their Python
    references returns them through native deferred retirement: their local GPU users and every DMA reader must
    finish before the region can allocate the bytes again. No receipt all-gather or other application acknowledgment
    is emitted, and no unrelated incoming transfer is used as evidence of consumption. The rows that stay pass
    through the staging too, so no copy has both sides strided (torch's MPS copy makes such a copy through a temporary
    of its size, from 10 MiB a 1 GiB heap).  Each is returned at its logical shape and that shape's contiguous strides
    (a moved block's leading extents of one may carry any stride, which from_local's inference would refuse).

    This other-backend compatibility path still materializes whole-operand staging.
    The exact-byte path writes only the declared final intervals, preserving neighboring live bytes."""
    from . import _stream
    mesh, held = tensors[0].device_mesh, [[list(t.placements), t.to_local(), t.shape, t.stride()] for t in tensors]
    for i in range(mesh.ndim):
        key = _NAMED.get((mesh, mesh.mesh_dim_names[i])) if mesh.mesh_dim_names else None
        moving = [h for h in held if isinstance(h[0][i], RaggedShard) and key not in (None, h[0][i].key)]
        rank, group = mesh.get_coordinate()[i], mesh.get_group(i)
        published = False

        def publish_boundary():
            nonlocal published
            if not published:
                _stream.publish_boundary()
                published = True

        for h in moving:
            p, d = h[0][i], h[0][i].dim % h[1].ndim
            direct = _follow_bytes(p, h[1], h[2], h[3], key, rank, group, publish_boundary)
            if direct is None:
                direct = _follow_spans(p, h[1], h[2], key, rank, group, publish_boundary)
            if direct is not None:
                h[0][i], h[1] = RaggedShard(p.dim, key), direct
                continue
            rows = h[1].movedim(d, 0)
            old, rest = _OPERANDS[p.key], rows.shape[1:]
            a, b = (list(itertools.accumulate(o.parts.tolist(), initial=0)) for o in (old, _OPERANDS[key]))
            g, n, row = _units(old, h[2][p.dim]), len(old.capacity), math.prod(rest)
            span = lambda r, q: g * max(0, min(a[r + 1], b[q + 1]) - max(a[r], b[q]))
            sends, receives = [span(rank, q) for q in range(n)], [span(r, rank) for r in range(n)]
            first, into = list(itertools.accumulate(sends, initial=0)), list(itertools.accumulate(receives, initial=0))
            out = [0 if q == rank else c * row for q, c in enumerate(sends)]
            land = [0 if r == rank else c * row for r, c in enumerate(receives)]
            out_at, land_at = (list(itertools.accumulate(counts, initial=0)) for counts in (out, land))
            sizes = ((out_at[-1] + sends[rank] * row) * rows.element_size(), land_at[-1] * rows.element_size())
            pair = [_empty((max(need, 1),), rows, group, torch.uint8) for need in sizes]
            x, y = (pair[s][:need].view(rows.dtype) for s, need in enumerate(sizes))
            x, stay = x[:out_at[-1]], x[out_at[-1]:].view((sends[rank],) + rest)
            new = _empty(h[1].shape[:d] + (into[-1],) + h[1].shape[d + 1:], rows, group)
            block = new.movedim(d, 0)
            block[into[rank]:into[rank + 1]].copy_(stay.copy_(rows[first[rank]:first[rank + 1]]))
            for q in range(n):
                if q != rank:
                    x[out_at[q]:out_at[q] + sends[q] * row].view((sends[q],) + rest).copy_(rows[first[q]:first[q + 1]])
            matrix = [0 if r == q else span(r, q) * row for r in range(n) for q in range(n)]
            with _configuration_exchange(group, out, land, rows.dtype, matrix):
                dist.all_to_all_single(y, x, land, out, group=group)
            for r in range(n):
                if r != rank:
                    block[into[r]:into[r + 1]].copy_(y[land_at[r]:land_at[r] + receives[r] * row].view((receives[r],) + rest))
            h[0][i], h[1] = RaggedShard(h[0][i].dim, key), new
    contiguous = lambda shape: torch.empty(shape, device='meta').stride()
    return [from_local(local, mesh, placements, shape=shape, stride=contiguous(shape)) for placements, local, shape, _ in held]


# The tensors partition split along each partitioned mesh dimension, re-laid in place where the balancer moves its
# shares between evaluations (evaluations): weak references by id (a tensor's == is elementwise), so the registry holds
# nothing alive.
_SPLIT = {}


def _register(t):
    """`t` (a RaggedTensor) registered with each partitioned mesh dimension it is split along; `t` returned."""
    names = t.device_mesh.mesh_dim_names or ()
    for i, p in enumerate(t.placements):
        if isinstance(p, RaggedShard) and i < len(names):
            held = _SPLIT.setdefault((t.device_mesh, names[i]), {})
            held[id(t)] = weakref.ref(t, lambda _, held=held, k=id(t): held.pop(k, None))
    return t


def _registered(t):
    return any((r := held.get(id(t))) is not None and r() is t for held in _SPLIT.values())


def empty(mesh, name, shape, dim, dtype=None):
    """A program's buffer declared by its whole extent: a RaggedTensor of logical `shape` split along `dim` by
    partitioned mesh dimension `name`, its local block a region lease of this rank's share (_empty), registered, so
    it is re-laid in place where the shares move between evaluations.  The program takes its local view (to_local, no
    copy) inside its step."""
    placements = [Replicate()] * mesh.ndim
    placements[mesh.mesh_dim_names.index(name)] = Shard(dim)
    local = distribute_tensor(torch.empty(shape, dtype=dtype, device='meta'), mesh, placements, src_data_rank=None).to_local()
    held = _empty(local.shape, torch.empty(0, dtype=local.dtype, device=mesh.device_type), mesh.get_group(name))
    return from_local(held, mesh, placements, shape=shape, stride=torch.empty(shape, device='meta').stride())


def _relay(mesh, name):
    """Every live tensor split along mesh dimension `name` moved onto the parts attached now, in place (its local
    block and spec replaced, the object kept, so a module's parameters move under it): one follow, one coalesced
    group."""
    held = [t for r in list(_SPLIT.get((mesh, name), {}).values()) if (t := r()) is not None]
    if held:
        for t, moved in zip(held, follow(*held)):
            t._local_tensor, t._spec = moved._local_tensor, moved._spec


def calls(tensor):
    """The groups full_tensor() of RaggedTensor `tensor` issues, one all-gather of its blocks a partitioned mesh
    dimension (RaggedShard._to_replicate_tensor), each from its shape and the operand attached now (torch_mesh.plan)."""
    mesh, local = tensor.device_mesh, tensor.to_local()
    return [gathered(local, p.dim, _blocks(_OPERANDS[p.key], tensor.shape[p.dim])[0], mesh.get_group(i))
            for i, p in enumerate(tensor.placements) if isinstance(p, RaggedShard) and mesh.size(i) > 1]


def plan(mesh, name, groups):
    """The program's collectives on mesh dimension `name`'s group declared once, in their order (torch_mesh.plan):
    `groups` from the ops' own (calls, gathered, scattered), as a step issues them, or a function of no arguments
    giving them from the parts attached now, called again when the controller moves the parts. A list declares
    invariant counts; use a callable for every declaration whose counts depend on the current partition."""
    from . import plan as declare
    _PLANS[mesh, name] = groups
    return declare(mesh.get_group(name), groups() if callable(groups) else groups)


_PLANS = {}


def _replan(mesh, name):
    """Re-realize the declared graph, including unchanged-count graphs after route updates."""
    groups = _PLANS.get((mesh, name))
    if groups is not None:
        from . import plan as declare
        declare(mesh.get_group(name), groups() if callable(groups) else groups)


def split(mesh, name, N):
    """The valid blocks of an extent N along partitioned mesh dimension `name` under the current parts, in rank order."""
    o = operand(mesh, name)
    return [_units(o, N) * p for p in o.parts.tolist()]


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
    for info in _planned(current, target):
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


_PLANNED = {}


def _planned(current, target):
    """torch's planner's steps from `current` to `target` (_gen_transform_infos_non_cached), planned once for their
    placements, each told apart by its type and a RaggedShard's key as _same tells them, and their shape, which the
    program fixes when it is compiled (design/data-model.md section 9: no planning per call)."""
    typed = lambda placements: tuple((type(p), p, getattr(p, 'key', None)) for p in placements)
    key = (current.mesh, typed(current.placements), typed(target.placements), tuple(current.shape))
    if key not in _PLANNED:
        _PLANNED[key] = _gen_transform_infos_non_cached(current, target)
    return _PLANNED[key]


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
            if not o.capacity[coordinate[i]]:
                raise ValueError('a zero-share local tensor needs declared shape= and stride= in from_local')
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
        # An empty local gradient cannot reconstruct the global stride. Its
        # enclosing DTensor already retains that declared metadata.
        stride = spec.tensor_meta.stride if not grad.numel() else _global_info(grad, spec.mesh, placements)[1]
        meta = TensorMeta(spec.shape, stride, grad.dtype)
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
    return _register(RaggedTensor(local.requires_grad_(tensor.requires_grad), spec, requires_grad=tensor.requires_grad))


def from_local(local_tensor, device_mesh, placements, *, run_check=False, shape=None, stride=None, grad_placements=None):
    """torch's DTensor.from_local, a Shard on an attached mesh dimension its RaggedShard; without shape= and stride=
    the local tensor is a capacity block (global = local P / c_r). A zero-share rank must retain the explicit global
    shape and stride because its empty local shape cannot determine the distributed extent."""
    placements = _placements(device_mesh, placements, local_tensor.ndim)
    if not _ragged(placements):
        return DTensor.from_local(local_tensor, device_mesh, placements, run_check=run_check, shape=shape, stride=stride,
                                  grad_placements=grad_placements)
    if local_tensor.device.type != device_mesh.device_type and not local_tensor.is_meta:
        local_tensor = local_tensor.to(device_mesh.device_type)
    if run_check and device_mesh._is_current_rank_part_of_mesh():
        with torch.no_grad():
            for i, p in enumerate(placements):
                if p.is_replicate():
                    local_tensor = local_tensor.contiguous()
                    mesh_broadcast(local_tensor, device_mesh, mesh_dim=i)
    if shape is None or stride is None:
        shape, stride = _global_info(local_tensor, device_mesh, placements)
    return _register(_FromLocal.apply(local_tensor, device_mesh, placements, torch.Size(shape), tuple(stride),
                                      None if grad_placements is None else tuple(grad_placements)))


def redistribute(tensor, placements, *, async_op=False):
    """tensor.redistribute(placements=...), a Shard on an attached mesh dimension its RaggedShard (the tensor's own
    there first); a ragged one to Partial from Replicate takes each coordinate's term (Partial._partition_value), and
    from a Shard to Partial(sum) places its valid elements among zeros (_local_redistribute)."""
    placements = _placements(tensor.device_mesh, placements, tensor.ndim, tensor.placements)
    if _same(tensor.placements, placements):
        return tensor
    if not (_ragged(placements) or _ragged(tensor.placements)):
        return DTensor.redistribute(tensor, placements=placements, async_op=async_op)
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
# mesh dimension its RaggedShard.
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
                               value._local_tensor, named.get('is_causal', False), named.get('scale'), 1, query.shape[2])
    out = RaggedTensor(out, _spec_like(spec, query.shape[:-1] + value.shape[-1:], out.dtype), requires_grad=False)
    if op_call is aten._scaled_dot_product_attention_math_for_mps.default:
        return out, None
    return out, RaggedTensor(lse, _spec_like(spec, query.shape[:-1], lse.dtype), requires_grad=False)


def _detached(op_call, args, kwargs):
    """detach (nn.Parameter makes a parameter by it) of a tensor split along a partitioned dimension, registered as
    its source is (_register), so the parameter is the one re-laid."""
    out = _propagated(op_call, args, kwargs)
    return _register(out) if _registered(args[0]) else out


_HANDLERS = {aten.mean.default: _mean, aten.mean.dim: _mean, aten.linear_backward.default: _linear_backward,
             aten._scaled_dot_product_flash_attention_for_cpu.default: _attention,
             aten._scaled_dot_product_attention_math_for_mps.default: _attention, aten.detach.default: _detached}


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

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        x = inputs[0]
        if (not isinstance(x, DTensor) and isinstance(mod, nn.Linear) and x.shape[-1] == 0
                and any(p.is_shard() and p.dim % x.ndim == x.ndim - 1 for p in input_layouts)):
            shape = x.shape[:-1] + (mod.in_features,)
            x = from_local(x, device_mesh, input_layouts, shape=shape,
                           stride=torch.empty(shape, device='meta').stride())
        return _prepare(x, input_layouts, desired_input_layouts, device_mesh)

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


def _sequence_extent(o, rank, local, sequence_length=None):
    """Preserve declared logical extent even on a rank with no sequence tiles."""
    if sequence_length is None:
        if not o.capacity[rank]:
            raise ValueError('a zero-share rank needs the declared sequence_length')
        sequence_length, remainder = divmod(local * o.units, o.capacity[rank])
        if remainder:
            raise ValueError('local positions are not a capacity block of this partition')
    if local != _units(o, sequence_length) * o.capacity[rank]:
        raise ValueError('local positions do not match the declared sequence_length')
    return sequence_length


@torch.no_grad()
def context_parallel_unshard(mesh, name, buffers, seq_dims, load_balance=True, *, sequence_length=None):
    """Gather context blocks through the common exact physical WRITE program.

    The ordinary contiguous sequence layout is the declared final destination.
    A head/tail partition gathers the same exact contributions, then permutes
    its two logical halves locally; no page-padded transport buffer is used.
    Original input strides are read by the shared gather publisher. Supply the
    declared sequence_length on zero-share ranks, whose local extent is empty.
    """
    o, rank, group = operand(mesh, name), mesh.get_local_rank(name), mesh.get_group(name)
    local = buffers[0].shape[seq_dims[0]]
    S = _sequence_extent(o, rank, local, sequence_length)
    halves, lengths, joined = _halves(o, S, load_balance), _blocks(o, S)[0], []
    for b, dim in zip(buffers, seq_dims):
        dim %= b.ndim
        shape = b.shape[:dim] + (S,) + b.shape[dim + 1:]
        gathered = allgatherv(b, dim, lengths, group.group_name)
        whole = gathered
        if halves != 1:
            pieces = []
            for r, (at, size) in enumerate(zip(itertools.accumulate(lengths, initial=0), lengths)):
                for half, (lo, hi) in enumerate(_spans(o, r, S, halves)):
                    pieces.append((lo, gathered.narrow(dim, at + half * (size // halves), hi - lo)))
            whole = torch.cat([value for _, value in sorted(pieces, key=lambda item: item[0])], dim)
        joined.append(whole)
    return joined


def ring_attention(mesh, name, query, key, value, is_causal=False, scale=None, load_balance=True, *, sequence_length=None):
    """This rank's query block's attention over every rank's keys and values, the blocks along dimension 2 (the
    sequence) as context_parallel_buffers made them over partitioned mesh dimension `name`: (output, each query row's
    logsumexp), padded query rows garbage that the unshard drops. sequence_length retains the AOT logical extent
    on a zero-share rank; that rank supplies no partial input but remains in the physical communicator."""
    o = operand(mesh, name)
    S = _sequence_extent(o, mesh.get_local_rank(name), query.size(2), sequence_length)
    return _ring_attention(mesh.get_group(name), o.key, query, key, value, is_causal, scale, _halves(o, S, load_balance), S)


def unshard_calls(mesh, name, shapes, dims, dtype, *, sequence_length=None):
    """The static gathered blocks used by context_parallel_unshard."""
    group, o, rank = mesh.get_group(name), operand(mesh, name), mesh.get_local_rank(name)
    groups = []
    for shape, dim in zip(shapes, dims):
        dim %= len(shape)
        total = _sequence_extent(o, rank, shape[dim], sequence_length)
        row = math.prod(shape[:dim] + shape[dim + 1:])
        counts = [length * row for length in _blocks(o, total)[0]]
        groups.append(_gather_writes(counts, dtype))
    return groups


SCORES = 1 << 26
def _block(query, key, value, scale=None, queries=None, keys=None, valid=None):
    """Declare bounded score tiles and their [m,l,O] rows for the common compiler."""
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    lead, rows, n = query.shape[:-2], query.shape[-2], key.shape[-2]
    if not n:
        return query.new_zeros((*lead, rows, value.shape[-1] + 2), dtype=torch.float32)
    step = max(1, SCORES // max(1, math.prod(lead) * n))
    parts = []
    for at in range(0, rows, step):
        q = query[..., at:at + step, :]
        scores = (q @ key.transpose(-2, -1)).float() * scale
        if queries is not None:
            scores = scores.masked_fill(keys > queries[at:at + step, None], -math.inf)
        if valid is not None:
            scores = scores.masked_fill(~valid, -math.inf)
        top = scores.amax(-1, keepdim=True)
        top = torch.where(torch.isfinite(top), top, 0)
        probabilities = (scores - top).exp()
        parts.append(torch.cat((top, probabilities.sum(-1, keepdim=True),
                                (probabilities.to(value.dtype) @ value).float()), -1))
    return torch.cat(parts, -2) if parts else query.new_empty((*lead, 0, value.shape[-1] + 2), dtype=torch.float32)


def _ring_attention(group, key_name, query, key, value, is_causal=False, scale=None, halves=1, sequence_length=None):
    """Declare query movement, split-KV work and the configured partial-row reduction.

    The whole function is traced before execution. Its tensor operations and
    crossings share the same compiler; there is no eager ring scheduler.
    """
    from .backend import merge_partials, attention
    o = _OPERANDS[key_name]
    rank = dist.get_rank(group)
    S = _sequence_extent(o, rank, query.size(2), sequence_length)
    lengths, lead = _blocks(o, S)[0], tuple(query.shape[:2])
    rows = [math.prod(lead) * length for length in lengths]
    gathered = allgatherv(query, 2, lengths, group.group_name)
    blocks = gathered.split(lengths, dim=2)
    keys, valid = _positions(key_name, rank, S, halves, query.device)
    mine, step = _spans(o, rank, S, halves), lengths[rank] // halves
    partials = []
    for owner, q in enumerate(blocks):
        positions = _positions(key_name, owner, S, halves, query.device)[0]
        height = lengths[owner] // halves
        local = []
        for half, (_, hi) in enumerate(_spans(o, owner, S, halves)):
            seen = [b for b, (lo, _) in enumerate(mine) if not is_causal or lo < hi]
            at = slice(half * height, (half + 1) * height)
            first, end = (seen[0] * step, (seen[-1] + 1) * step) if seen else (0, 0)
            local.append(_block(q[:, :, at], key[:, :, first:end], value[:, :, first:end], scale,
                                positions[at] if is_causal else None, keys[first:end], valid[first:end]))
        block = torch.cat(local, 2)
        partials.append(block.transpose(1, 2).reshape(rows[owner], value.shape[-1] + 2))
    merged = merge_partials(torch.cat(partials, 0), -1, group.group_name, rows)
    output, lse = attention(merged)
    return (output.view(lead[0], lengths[rank], lead[1], value.shape[-1]).transpose(1, 2),
            lse.view(lead[0], lengths[rank], lead[1]).transpose(1, 2))
