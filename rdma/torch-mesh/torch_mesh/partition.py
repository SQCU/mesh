"""A partition operand the backend sets: DTensor's Shard, tensor parallelism and context parallelism split a mesh
dimension by parts the backend moves between calls by the measured residual (design/heterogeneity.md §4, R3, R14).

Stock PyTorch splits every sharded extent equally (Shard follows torch.chunk).  Here a program declares a mesh
dimension's P relative units, attach(mesh, tp=P), the same on every rank. Each rank holds the interval between
two integer-rounded cumulative shares of the declared extent. These intervals cover the axis exactly, including
odds, tails and empty shares. Divisible extents retain their previous boundaries. The capacity and index operands
are fixed during preparation; a later assignment constructs a new generation.  follow(*ts) moves sharded tensors onto the parts the Balancer attached, one group a move.  On raw
tensors split(mesh, "tp", N) gives the valid blocks in rank order, the ones the list collectives take
(dist.all_gather into a list of unequal outputs, dist.reduce_scatter of a list of unequal inputs: MPI's v-collectives
in torch's spelling).

On DTensor the partition is a placement, RaggedShard(dim, key) [veScale's RaggedShard(local_units), pytorch#169320],
held by the tensors it splits.  A DTensor holding one is a
RaggedTensor, a DTensor subclass whose own dispatch keeps it ragged through every op.  distribute_tensor, from_local and
redistribute below are torch's, with a Shard on an attached mesh dimension that dimension's RaggedShard; from_local
requires the declared global shape and stride for a ragged local operand.  ColwiseParallel, RowwiseParallel and
SequenceParallel below are torch's parallel styles over them.  A tensor that meets a RaggedShard on its mesh dimension
holds the same one or none there (R6: a plain Shard or another partition raises ValueError).

No block holds padding (each capacity is its part), so an op on a local shard runs as torch runs it, a contraction
or reduction over a partitioned dimension giving the Partial its sharding gives, with nothing masked:
- Count-dependent: mean over a partitioned dimension is the sum divided by its logical count. Vector norms,
  softmax, log-softmax and their gradients use the shared algebra before sharding propagation. Their sum/max/min
  summaries are reduced across the split dimensions; their input rows remain sharded. An op whose sharding gives
  another Partial there (avg, embedding's mask), a reshape that changes rank ownership,
  and equal still get the dimension whole first. Convolution consumes explicit result ownership
  through Function's source graph declaration. Argmax/argmin reduce finite value/position summaries;
  owner-split index_select combines integer representations so value bits are unchanged. Static window,
  slice and select views move only their declared page intersections. Sort and scans retain DTensor's
  own whole-dimension rules until their lowering is extended.
- Attention: SDPA on query, key and value split along the sequence by one RaggedShard (on any mesh dimension) is ring
  attention, each source's keys past its valid extents masked (score -inf); context parallelism's ring
  (ring_attention) masks the same way over head-tail halves.  The softmax over the split is the library's partials
  merge (below).

Redistribution, each block its part (static counts, never padded to a common size): Shard to Replicate all-gathers
the blocks, which laid end to end are the logical tensor; Replicate to Shard takes this rank's block of it; Partial to
Shard reduce-scatters the blocks; Shard to Partial places this rank's block among zeros; Shard(d) to Shard(d') goes
through Replicate; Shard's scatter from a source rank is the rooted V (the backend's scatter, MPI_Scatterv through
ncclMeshAlltoAllV), each segment carrying its exact useful element count. Context parallelism's head-tail split keeps
each half at its exact weighted boundaries: rank r holds its head positions followed by its tail positions
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
from contextvars import ContextVar
import itertools
import json
import math
import resource
import weakref
from pathlib import Path
from allocate import partition_bounds

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
            # These are placement operands, not work in the declared tensor
            # graph. A trace must not leave FakeTensors in the live page map.
            from torch._subclasses.fake_tensor import unset_fake_temporarily
            from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
            with disable_proxy_modes_tracing(), unset_fake_temporarily():
                self.cache[what] = build(self.parts.tolist())
        return self.cache[what]


_OPERANDS = {}
_NAMED = {}
_KEYS = itertools.count()
_SOURCE_CAPTURE = ContextVar('mesh_source_capture', default=None)


def _blocks(o, N):
    """Exact local sizes and origins of the declared logical extent."""
    bounds = partition_bounds(o.capacity, N)
    return [b - a for a, b in zip(bounds, bounds[1:])], list(bounds[:-1])


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
# A reduction's rank-local partials, fused producer publication and numerical
# fold. Gathers use the common indexed WRITE planner and final tensor views.
_GATHER_SOURCE = r"""
template <uint N, uint CHANNELS>
struct gather_words { uint count, pre, post, rank, j; uint sizes[N], starts[N]; mesh_step_binding land[CHANNELS], publish[CHANNELS]; };

template <uint AXES, uint CHANNELS>
struct partial_publish_words {
    uint count, j, shape[AXES], strides[AXES];
    mesh_step_binding land[CHANNELS], publish[CHANNELS];
};

template <uint AXES, uint UNITS, uint COPY_BYTES, uint CHANNELS, bool CONTIGUOUS, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void partial_publish(device const metal::conditional_t<COPY_BYTES == 1, uchar, ushort>* x [[buffer(0)]], device uchar* send [[buffer(1)]],
                           device atomic_uint* arrivals [[buffer(2)]], constant partial_publish_words<AXES, CHANNELS>& w [[buffer(15)]],
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
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
}

template <uint N, uint CHANNELS, uint THREADS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void gather_ready(constant gather_words<N, CHANNELS>& w [[buffer(15)]],
                         uint tid [[thread_index_in_threadgroup]]) {
    gather_observe(w, tid);
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
from . import _stream
_GATHER_KERNELS, _GATHER_THREADS = {}, _stream.compilation['GATHER_THREADS']


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
    communicator and the waits' bound; this rank and the ranks; the publication's kernel (partial_publish, or
    linear_publish where `w`), constant words, its repeat's word and grid; the reduction's (sum_land, and
    this rank's segment's start and each rank's sizes in its words); every rank's exact segment start and count
    and the output's shape; the publication's arrival count; the separate landing-prefix
    observer and physical channel extent."""
    from . import _stream
    from lifetimes import prove_recurrent_gather
    if not adds:
        raise ValueError('gathers use prepared final views, not fused reduction storage')
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
             starts[:-1])
    shape = operand
    from .streamed import node
    geometry = _stream.compilation
    capacity = node()[1] * geometry['GATHER_GROUPS_PER_CORE']
    grain = _GATHER_THREADS * geometry['GATHER_ELEMENTS_PER_THREAD']
    spread = lambda count: max(1, min(capacity, -(-count // grain)))
    groups = spread(math.prod(shape) * half)
    inline_wait = groups == 1
    land = _gather_kernel('sum_land', x.element_size(), n, channels, inline_wait)
    if w is None:
        publish_words = [math.prod(operand) * half, 0, *x.shape, *x.stride()]
        publish = (_gather_kernel('partial_publish', x.ndim, half, copy_bytes, channels, x.is_contiguous()), publish_words, 1,
                   spread(words[0]), _GATHER_THREADS)
    else:
        K, N, ldw = x.shape[1], w.shape[0], w.stride(0)
        if x.stride(1) != 1:
            raise ValueError('mesh: linear publication requires contiguous columns; realize a strided producer kernel')
        split = _stream.linear_split(K, N, x.shape[0])
        vector, columns, loads = (geometry[k] for k in ('LINEAR_VECTOR', 'LINEAR_COLUMNS', 'LINEAR_LOADS'))
        publish = (_linear_kernel(vector if ldw % vector == 0 else 1, vector if x.stride(0) % vector == 0 else 1, columns, split, loads, channels),
                   [x.shape[0], K, N, 0, ldw, K, x.stride(0), x.stride(0), 0], 8, _stream.linear_groups(x.shape[0], N), split * geometry['SIMD_WIDTH'])
    return (group.handle, _stream.polls, n, rank, *publish, land, words, 4, groups, _GATHER_THREADS,
            starts, counts, list(shape), torch.zeros(1, dtype=torch.int32, device='mps'),
            -1 if inline_wait else _gather_kernel('gather_ready', n, channels), channels)


from . import _stream as _native  # noqa: E402  (torch_mesh::fused_step's library)

_native.fused_maker(_fused_form)


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
    if hasattr(group, 'prepare_writes'):
        raise RuntimeError('torch_mesh::reduce_scatterv on a mesh group runs inside a prepared Function declaration; '
                           'an eager collective is not executed (design/eager-execution.md)')
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
    from .backend import ALLGATHER, WRITE, Call
    counts = [s * _row(x, dim) for s in sizes]
    if not _region_backed(x, group):
        return [Call(ALLGATHER, x.numel(), x.dtype, counts=counts)]
    return [Call(WRITE, count * x.element_size(), torch.uint8, root=source, destination=destination)
            for source, count in enumerate(counts) if count
            for destination in range(len(counts)) if source != destination]


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
    """Partition mesh dimensions by name: attach(mesh, tp=units), the relative work units, the same on every rank.
    The backend splits them evenly over the coordinates in rank
    order (the first units mod n one more)."""
    for name, units in dims.items():
        n = mesh.size(mesh.mesh_dim_names.index(name)) if name in (mesh.mesh_dim_names or ()) else 0
        if n <= 0 or units <= 0:
            raise ValueError(f'mesh: partition {name}: {units} units over a mesh dimension of {mesh.mesh_dim_names}, '
                             f'positive units and a named mesh dimension are required')
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


def _scatter_spans(tensor, dim, sizes, rank, source, group):
    """Initial input-domain import, then the same exact native span WRITE plan."""
    if not hasattr(group, 'prepare_writes'):
        return None
    from . import _stream
    imports = []
    dim %= tensor.ndim
    offsets = list(itertools.accumulate(sizes, initial=0))
    original = tensor
    if rank == source and not _in_region(tensor):
        # Native DMA addresses currently belong to the registered region. This
        # is one input-domain import, independent of destination count/layout.
        imported, imports = _stream.import_attributes([tensor])
        original = imported[0]
    shape = list(tensor.shape)
    shape[dim] = sizes[rank]
    result = (original.narrow(dim, offsets[rank], sizes[rank]) if rank == source
              else _empty(shape, tensor, group))
    deliveries = []
    row = _row(tensor, dim)
    for destination, count in enumerate(sizes):
        if destination == source or not count * row:
            continue
        # a relay is its shape and type alone: the library leases it where a route crosses this rank
        view = (original.narrow(dim, offsets[destination], count) if rank == source
                else result if rank == destination else torch.empty((count * row,), dtype=tensor.dtype, device='meta'))
        deliveries.append((view, source, destination))
    prepared, calls = group.prepare_writes(deliveries) if deliveries else (None, [])
    with group.configuration([calls]) if calls else contextlib.nullcontext():
        schedule = _stream.Schedule()
        for source_bytes, destination_bytes in imports:
            schedule.copy(source_bytes, destination_bytes)
        if prepared is not None:
            prepared.append(schedule)
        schedule.finish()
        schedule.enqueue()
    return result


def _follow_spans(placement, local, logical_shape, key, rank, group):
    """Arbitrary regional strides use exact AOT span WRITE bindings."""
    if not hasattr(group, 'prepare_writes'):
        return None
    from .layout import copies
    commands = []
    dim = placement.dim % local.ndim
    old, new = _OPERANDS[placement.key], _OPERANDS[key]
    before_sizes, before = _blocks(old, logical_shape[dim])
    after_sizes, after = _blocks(new, logical_shape[dim])
    if local.shape[dim] != before_sizes[rank]:
        raise ValueError('strided migration source does not match its attached partition')
    shape = list(local.shape)
    shape[dim] = after_sizes[rank]
    start = after[rank] - before[rank]
    result = (local.narrow(dim, start, after_sizes[rank])
              if 0 <= start and start + after_sizes[rank] <= before_sizes[rank]
              else _empty(shape, local, group))
    deliveries = []
    row = math.prod(local.shape[:dim] + local.shape[dim + 1:])
    for source in range(len(before)):
        for destination in range(len(after)):
            first = max(before[source], after[destination])
            last = min(before[source] + before_sizes[source], after[destination] + after_sizes[destination])
            if first >= last:
                continue
            count = last - first
            if source == destination:
                if source == rank:
                    commands.extend(copies(local.narrow(dim, first - before[source], count),
                                           result.narrow(dim, first - after[destination], count)))
                continue
            if rank == source:
                view = local.narrow(dim, first - before[source], count)
            elif rank == destination:
                view = result.narrow(dim, first - after[destination], count)
            else:   # a relay is its shape and type alone: the library leases it where a route crosses this rank
                view = torch.empty((row * count,), dtype=local.dtype, device='meta')
            deliveries.append((view, source, destination))
    prepared, calls = group.prepare_writes(deliveries) if deliveries else (None, [])
    with group.configuration([calls]) if calls else contextlib.nullcontext():
        schedule = _stream.Schedule()
        schedule.append(_stream.Commands(commands))
        if prepared is not None:
            prepared.append(schedule)
        schedule.finish()
        schedule.enqueue()
    return result


@torch.no_grad()
def follow(*tensors):
    """Realize changed shard assignments through typed final page views.

    Mesh deliveries become bound communicator Calls; local relocations use the
    shared layout-copy commands in the same prepared schedule. Original input
    imports are exact byte copies in that schedule. Other distributed backends
    retain their existing staging adapter until their function compiler is
    connected to the same page declarations.
    """
    from . import _stream
    mesh, held = tensors[0].device_mesh, [[list(t.placements), t.to_local(), t.shape, t.stride()] for t in tensors]
    for i in range(mesh.ndim):
        key = _NAMED.get((mesh, mesh.mesh_dim_names[i])) if mesh.mesh_dim_names else None
        moving = [h for h in held if isinstance(h[0][i], RaggedShard) and key not in (None, h[0][i].key)]
        rank, group = mesh.get_coordinate()[i], mesh.get_group(i)
        for h in moving:
            p, d = h[0][i], h[0][i].dim % h[1].ndim
            direct = _follow_spans(p, h[1], h[2], key, rank, group)
            if direct is not None:
                h[0][i], h[1] = RaggedShard(p.dim, key), direct
                continue
            rows = h[1].movedim(d, 0)
            old, rest = _OPERANDS[p.key], rows.shape[1:]
            a, b = (partition_bounds(o.capacity, h[2][p.dim]) for o in (old, _OPERANDS[key]))
            n, row = len(old.capacity), math.prod(rest)
            span = lambda r, q: max(0, min(a[r + 1], b[q + 1]) - max(a[r], b[q]))
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


def plan(mesh, name, groups, *, production_seconds=None):
    """The program's collectives on mesh dimension `name`'s group declared once, in their order (torch_mesh.plan):
    `groups` from the ops' own (calls, gathered, scattered), as a step issues them, or a function of no arguments
    giving them from the parts attached now, called again when the controller moves the parts. A list declares
    invariant counts; use a callable for every declaration whose counts depend on the current partition."""
    from . import plan as declare
    _PLANS[mesh, name] = (groups, production_seconds)
    return declare(mesh.get_group(name), groups() if callable(groups) else groups, production_seconds=production_seconds)


_PLANS = {}


def _replan(mesh, name):
    """Re-realize the declared graph, including unchanged-count graphs after route updates."""
    held = _PLANS.get((mesh, name))
    if held is not None:
        groups, production_seconds = held
        from . import plan as declare
        declare(mesh.get_group(name), groups() if callable(groups) else groups, production_seconds=production_seconds)


def split(mesh, name, N):
    """The valid blocks of an extent N along partitioned mesh dimension `name` under the current parts, in rank order."""
    o = operand(mesh, name)
    return _blocks(o, N)[0]


def _ragged(placements):
    return any(isinstance(p, RaggedShard) for p in placements)


def _same(a, b):
    """Placements alike, a RaggedShard's key and type included (torch's equality takes a RaggedShard for Shard(dim))."""
    a, b = tuple(a), tuple(b)
    return len(a) == len(b) and all(type(p) is type(q) and p == q and getattr(p, 'key', None) == getattr(q, 'key', None)
                                    for p, q in zip(a, b))


def _placement_key(placements):
    return tuple((type(p), p, getattr(p, 'key', None)) for p in placements)


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
    key = (current.mesh, current.mesh.get_coordinate(), _placement_key(current.placements),
           _placement_key(target.placements), tuple(current.shape))
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
        meta = TensorMeta(spec.shape, spec.tensor_meta.stride, grad.dtype)
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
    scattered from src_data_rank, or this rank's own exact block where it is None."""
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
    """Bind a local page view to its declared global shape, stride and placement.

    Ragged cuts cannot determine a global extent from one rank's local size.
    The frontend supplies that metadata, including on empty ranks.
    """
    placements = _placements(device_mesh, placements, local_tensor.ndim)
    if not _ragged(placements):
        return DTensor.from_local(local_tensor, device_mesh, placements, run_check=run_check, shape=shape, stride=stride,
                                  grad_placements=grad_placements)
    if shape is None or stride is None:
        raise ValueError('a ragged local operand requires its declared global shape and stride')
    if local_tensor.device.type != device_mesh.device_type and not local_tensor.is_meta:
        local_tensor = local_tensor.to(device_mesh.device_type)
    if run_check and device_mesh._is_current_rank_part_of_mesh():
        with torch.no_grad():
            for i, p in enumerate(placements):
                if p.is_replicate():
                    local_tensor = local_tensor.contiguous()
                    mesh_broadcast(local_tensor, device_mesh, mesh_dim=i)
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
        source = _SOURCE_CAPTURE.get()
        if source is not None:
            return source.local(self)
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
_WHOLE = (aten.equal.default,)


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


def _view_region(spec, coordinates):
    """The ordered page selection held by a rank, in the common layout algebra."""
    from . import _stream
    view = _stream.TensorLayout(tuple(spec.shape), tuple(spec.tensor_meta.stride))
    for i, rank in enumerate(coordinates):
        p = spec.placements[i]
        if p.is_shard():
            count, first = p._local_shard_size_and_offset(view.shape[p.dim], spec.mesh.size(i), rank)
            view = view.slice(p.dim, first, count, 1)
    return view.reshape((view.count,))


def _rank_selection(spec, coordinate, selection=None):
    """One rank's ordered logical pages, retaining all preceding axis cuts."""
    declaration = {'shape': list(spec.shape), 'steps': []} if selection is None else {
        'shape': selection['shape'], 'steps': list(selection['steps'])}
    shape = list(spec.shape)
    for i, p in enumerate(spec.placements):
        if p.is_shard():
            count, first = p._local_shard_size_and_offset(shape[p.dim], spec.mesh.size(i), coordinate[i])
            declaration['steps'].append(['select', p.dim, [[first, count, 0, 1]]])
            shape[p.dim] = count
    return declaration, tuple(shape)


def exchange_steps(mesh, dimensions, sources, destinations):
    """Compile the shared mesh-axis relay geometry from declared page domains.

    All ranks' selections are operands. A step's group and peer row name this
    process's ordinary mesh group; the complete maps also retain the other rows
    for demand propagation. Repeated uses remain in the consumer declaration.
    """
    from mesh_relation import Algebra
    algebra, dimensions = Algebra(), tuple(dimensions)
    coordinates = tuple(sources)
    semantic_shape = next(iter(sources.values()))['shape']
    extent = math.prod(semantic_shape)
    original = {rank: algebra.selection(selection) for rank, selection in sources.items()}
    wanted = {rank: algebra.selection(selection) for rank, selection in destinations.items()}
    fixed = tuple(axis for axis in range(mesh.ndim) if axis not in dimensions)
    def union(values):
        result = algebra.set('{ [i] : false }', 1)
        for value in values:
            result = result.binary(value, 'union')
        return result
    def agrees(left, right, axes):
        return all(left[axis] == right[axis] for axis in axes)
    held, current, steps = tuple(mesh.get_coordinate()), sources, []
    for position, axis in enumerate(dimensions):
        if position + 1 == len(dimensions):
            selected = wanted
        else:
            selected = {rank: union(value for peer, value in wanted.items()
                if agrees(peer, rank, (*fixed, *dimensions[:position + 1]))).binary(
                union(value for peer, value in original.items()
                      if agrees(peer, rank, (*fixed, *dimensions[position + 1:]))), 'intersect')
                for rank in coordinates}
        following = {rank: {'shape': list(semantic_shape), 'steps': [
            ['reshape', [extent]], ['select', 0, domain.spans()]]} for rank, domain in selected.items()}
        peers = tuple(tuple(q if dim == axis else held[dim] for dim in range(mesh.ndim))
                      for q in range(mesh.size(axis)))
        steps.append(dict(axis=axis, sources=current, destinations=following,
                          peers=peers, group=mesh.get_group(axis)))
        current = following
    return tuple(steps)


def _delivery_domains(section, source, requested, provided, mesh, algebra, *, routing=None):
    """Exact producer demands and the relay views that deliver those pages."""
    described = pytree.key_get(source[0].meta['val'], source[1])
    shape, coordinates = tuple(described.shape), _coordinates(mesh)
    empty = algebra.set('{ [i] : false }', 1)
    def selection(domain):
        return dict(shape=list(shape), steps=[['reshape', [math.prod(shape)]], ['select', 0, domain.spans()]])
    destinations = {rank: selection(requested.get(rank, empty)) for rank in coordinates}
    pending = []
    if provided is None:
        # An unallocated internal value is produced where its declared uses
        # demand it. External values already require an explicit owner record.
        sources, dimensions = destinations, ()
        relay_sources, relay_destinations = sources, destinations
    else:
        dimensions = tuple(provided['dimensions'])
        local = {rank: (requested.get(rank, empty).binary(
            algebra.selection(provided['owners'][rank]['selection']), 'intersect')
            if routing is None else empty) for rank in coordinates}
        remote = {rank: requested.get(rank, empty).binary(local[rank], 'subtract') for rank in coordinates}
        relay_destinations = {rank: selection(domain) for rank, domain in remote.items()}
        fixed = tuple(axis for axis in range(mesh.ndim) if axis not in dimensions)
        cohorts = {}
        for rank in coordinates:
            cohorts.setdefault(tuple(rank[axis] for axis in fixed), []).append(rank)
        demands = {rank: empty for rank in coordinates}
        for ranks in cohorts.values():
            incoming = [provided['owners'][rank]['selection'] if routing is None else routing[rank]
                        for rank in ranks]
            outgoing = [relay_destinations[rank] for rank in ranks]
            orders = [algebra.selection_relation(value) for value in incoming]
            need = empty
            for rank in ranks:
                need = need.binary(remote[rank], 'union')
            domains = [order.image(order.domain()).binary(need, 'intersect').binary(
                algebra.selection(provided['owners'][rank]['selection']), 'intersect')
                for rank, order in zip(ranks, orders, strict=True)]
            ambiguous = any(not domain.binary(previous, 'intersect').empty()
                            for at, domain in enumerate(domains) for previous in domains[:at])
            if ambiguous:
                # Replicated computation is legal. Selecting which equivalent
                # producer supplies a remote copy is a separate routing input.
                pending.append(dict(source=source, section=section, coordinates=tuple(ranks),
                                    destinations=destinations, requirement='delivery_sources'))
                for rank, domain in zip(ranks, domains, strict=True):
                    demands[rank] = domain
                continue
            incoming = [selection(domain) for domain in domains]
            orders = [algebra.selection_relation(value) for value in incoming]
            for origin, _, read, _, _, _ in algebra.exchanges(incoming, outgoing):
                positions = orders[origin].image(read.image(read.domain()))
                rank = ranks[origin]
                demands[rank] = demands[rank].binary(positions, 'union')
        relay_sources = {rank: selection(domain) for rank, domain in demands.items()}
        sources = {rank: selection(domain.binary(local[rank], 'union')) for rank, domain in demands.items()}
    steps = (() if pending or all(algebra.selection(value).empty() for value in relay_destinations.values())
             else exchange_steps(mesh, dimensions, relay_sources, relay_destinations))
    domains = (sources, destinations, *(step[side] for step in steps for side in ('sources', 'destinations')))
    participating = tuple(rank for rank in coordinates
                          if any(not algebra.selection(domain[rank]).empty() for domain in domains))
    return dict(section=section, source=source, provided=provided, sources=sources,
                destinations=destinations, steps=steps, participants=participating,
                pending=tuple(pending),
                type=TensorMeta(described.shape, described.stride(), described.dtype))


def _project_owned_boundaries(section, needs, algebra, mesh):
    """Use declared group membership to transduce every peer's source demand."""
    from .declarations import boundary_groups, boundary_projection
    from .ownership import port_requests
    original_needs = needs
    sections = {rank: section.get('rank_sections', {}).get(rank, section) for rank in needs}
    needs = {rank: port_requests(values, sections[rank].get('ports', {}), algebra, coordinate=rank)
             for rank, values in needs.items()}
    members = {int(mesh.mesh[rank]): rank for rank in _coordinates(mesh)}
    records = {}
    for rank, requested in needs.items():
        member = sections[rank]
        ports = member.get('ports', {})
        groups = boundary_groups(member)
        scopes = {}
        for name, group in groups.items():
            axis = member.get('group_axes', {}).get(name)
            scopes[name] = ({i: tuple(i if dim == axis else rank[dim] for dim in range(mesh.ndim))
                             for i in range(mesh.size(axis))} if axis is not None else
                            {i: members[global_rank] for i, global_rank in
                             enumerate(dist.get_process_group_ranks(group))})
        peers = {name: {i: needs[peer] for i, peer in ranks.items()} for name, ranks in scopes.items()}
        local = {name: i for name, ranks in scopes.items() for i, peer in ranks.items() if peer == rank}
        nodes = tuple(node for node in member['boundaries']
                      if not boundary_groups(dict(member, boundaries=(node,))) or
                      any(name in local for name in boundary_groups(dict(member, boundaries=(node,)))))
        scoped = dict(member, boundaries=nodes, peer_outputs=peers)
        record = boundary_projection(scoped, requested, algebra, ranks=local)
        record['section'] = member
        if ports:
            record['inputs'] = port_requests(record['inputs'], ports, algebra, coordinate=rank, incoming=True)
            record['outputs'] = original_needs[rank]
        if not nodes and any(not value.empty() for value in requested.values()):
            record['pending'] = (*record['pending'], dict(section=section, rank=rank,
                outputs=requested, requirement='collective_source_scope'))
        records[rank] = record
    return records


def _exchange_domains(x, desired, mesh, ownership=None):
    """Bind ordered page uses through the shared prepared mesh-axis steps."""
    from mesh_relation import Algebra
    from .declarations import select_exchange
    from .operand import TensorView, relation_view
    if isinstance(x, DTensor) and any(p.is_partial() for p in x.placements):
        x = redistribute(x, [Replicate() if p.is_partial() else p for p in x.placements])
    dimensions = (tuple(ownership['dimensions']) if ownership is not None else
                  tuple(i for i, p in enumerate(x.placements) if p.is_shard()) if isinstance(x, DTensor) else ())
    coordinates, held = _coordinates(mesh), tuple(mesh.get_coordinate())
    sources = {rank: ownership['owners'][rank]['selection'] if ownership is not None else
               _rank_selection(x._spec, rank)[0] if isinstance(x, DTensor) else
               {'shape': list(x.shape), 'steps': []} for rank in coordinates}
    wanted = {rank: desired(rank) for rank in coordinates}
    steps = exchange_steps(mesh, dimensions, sources, {rank: value[0] for rank, value in wanted.items()})
    local = _page_tensor(x._local_tensor if isinstance(x, DTensor) else x)
    current = sources
    algebra = Algebra()
    for step in steps:
        peers = step['peers']
        declarations = {side: [step[side][rank] for rank in peers] for side in ('sources', 'destinations')}
        shape = algebra.selection_relation(step['destinations'][held]).shape
        local = select_exchange(local, list(shape), json.dumps(declarations, separators=(',', ':')),
                                step['group'].group_name)
        current = step['destinations']
    selected, shape = wanted[held]
    requested = algebra.selection_relation(selected)
    provided = algebra.selection_relation(current[held])
    return relation_view(TensorView.of(local), requested.then(provided.inverse((math.prod(selected['shape']),))).lexmin(), shape)


def _page_tensor(value):
    """Expose a declared page view to tensor syntax without a numerical copy."""
    from .operand import TensorView
    if not isinstance(value, TensorView):
        return value
    if value.layout.affine:
        return value.tensor()
    from .host import assemble
    target = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    return assemble([(value, TensorView.of(target))])


def _exchange_view(x, target, selection):
    selected = _exchange_domains(x, lambda coordinate: _rank_selection(target, coordinate, selection), x.device_mesh)
    return _wrap(selected.tensor(), target, selected.storage.requires_grad)


def _coordinates(mesh):
    return tuple(itertools.product(*(range(mesh.size(i)) for i in range(mesh.ndim))))


def _value_ownership(value, mesh):
    """Normalize a frontend tensor type into the shared ordered page domains."""
    from .ownership import from_spec
    tensor = value._local_tensor if isinstance(value, DTensor) else value
    meta = TensorMeta(torch.Size(value.shape), tuple(value.stride()), value.dtype)
    spec = value._spec if isinstance(value, DTensor) else DTensorSpec(mesh, (Replicate(),) * mesh.ndim, tensor_meta=meta)
    return from_spec(spec), tensor


def declare_result(graph, value, ownership, *, mesh):
    """Attach an explicit allocation operand to an existing typed graph value.

    ``owners`` maps mesh coordinates to ordered page selections and local
    shapes. ``replicated_axes`` declares equivalent holders, when present.
    Stable graph identities survive the subsequent numerical decomposition.
    """
    name = value.name if isinstance(value, torch.fx.Node) else value
    node = next(n for n in graph.graph.nodes if n.name == name or n.meta.get('mesh_source') == name)
    described = node.meta['val']
    owners = ownership['owners'] if 'owners' in ownership else ownership
    replicated = ownership.get('replicated_axes', ()) if 'owners' in ownership else ()
    spec = ownership.get('spec') if 'owners' in ownership else None
    from .ownership import from_spec, owner_layouts
    typed = from_spec(spec) if spec is not None else {}
    typ = ownership.get('type', typed.get('type', TensorMeta(described.shape, described.stride(), described.dtype)))
    declaration = dict(type=typ, owners=owner_layouts(typ, owners),
                       dimensions=ownership.get('dimensions', typed.get('dimensions',
                           tuple(i for i in range(mesh.ndim) if i not in replicated))),
                       reductions=ownership.get('reductions', typed.get('reductions', ())), mesh=mesh, spec=spec)
    node.meta['mesh_ownership'] = declaration
    return declaration


def _same_ownership(left, right):
    from mesh_relation import Algebra
    algebra = Algebra()
    return left['owners'].keys() == right['owners'].keys() and all(
        tuple(owner['shape']) == tuple(right['owners'][rank]['shape']) and
        algebra.selection_relation(owner['selection']).equal(
            algebra.selection_relation(right['owners'][rank]['selection']))
        for rank, owner in left['owners'].items())


def _transition_sources(crossings, values, mesh):
    """Capture existing placement expressions as distinct typed value ports."""
    from .compilation import source_expression
    from .ownership import from_spec
    coordinate = tuple(mesh.get_coordinate())
    declared, replacements, result_versions, pending = [], {}, {}, []
    metadata = {}
    versions = torch.fx.Graph()
    order = {node: index for transition in crossings for index, node in enumerate(transition['consumer'].graph.nodes)}
    def semantic_spec(record, other):
        if record.get('spec') is not None:
            return record['spec']
        base = other.get('spec')
        reductions = dict(record.get('reductions', ()))
        placements = tuple(reductions.get(axis, Replicate() if base is None or base.placements[axis].is_partial()
                                          else base.placements[axis]) for axis in range(mesh.ndim))
        candidate = DTensorSpec(mesh, placements, tensor_meta=record['type'])
        return candidate if _same_ownership(from_spec(candidate), record) else None
    for transition in sorted(crossings, key=lambda item: (order[item['consumer']], item['side'] == 'results', item['order'])):
        before, after = transition['provided'], transition['required']
        if not (before.get('reductions') or after.get('reductions')):
            continue
        current, target = semantic_spec(before, after), semantic_spec(after, before)
        if current is None or target is None:
            pending.append(dict(transition=transition, requirement='placement_source_declaration'))
            continue
        source, path = transition['source']
        described = pytree.key_get(source.meta['val'], path)
        def version(owner):
            made = versions.placeholder(f'{source.name}_value_{len(versions.nodes)}')
            with described.fake_mode:
                made.meta['val'] = torch.empty_strided(described.shape, described.stride(),
                    dtype=described.dtype, device=described.device)
            key = made, ()
            values[key] = owner
            return key
        source_key = source, path
        if transition['side'] == 'results':
            source_key = version(before)
            result_versions[source, path] = source_key
            output_key = source, path
        else:
            output_key = version(after)
            replacements.setdefault((transition['consumer'], transition['side'], transition['path']), {})[path] = output_key
        if source_key not in metadata:
            with described.fake_mode:
                owner = before['owners'][coordinate]
                if 'stride' not in owner:
                    raise ValueError('placement source requires its declared per-rank stride for a gapped layout')
                metadata[source_key] = torch.empty_strided(owner['shape'], owner['stride'],
                    dtype=described.dtype, device=described.device)
        value = metadata[source_key]
        cache = source.meta.setdefault('mesh_placement_sources', {})
        identity = (path, _placement_key(current.placements), _placement_key(target.placements),
                    tuple(value.shape), coordinate, transition['consumer'], transition['side'], transition['path'])
        if identity not in cache:
            module, operands = source_expression(lambda tensor: _local_redistribute(tensor, current, target), value)
            output = next(node for node in module.graph.nodes if node.op == 'output')
            if isinstance(output.args[0], torch.fx.Node) and output.args[0].op == 'placeholder':
                with module.graph.inserting_before(output):
                    result = module.graph.call_function(torch.ops.aten.alias.default, (output.args[0],))
                    result.meta = output.args[0].meta.copy()
                output.args = (result,)
                module.recompile()
            cache[identity] = dict(graph=module, operands=tuple(operands))
        expression = cache[identity]
        result = next(node.args[0] for node in expression['graph'].graph.nodes if node.op == 'output')
        expression = dict(expression, result=result)
        input_node = next(node for node in expression['graph'].graph.nodes if node.op == 'placeholder')
        ports = {(input_node, ()): dict(field=source_key,
            selection=before['owners'][coordinate]['selection'],
            selections={rank: owner['selection'] for rank, owner in before['owners'].items()}),
            (result, ()): dict(field=output_key, selection=after['owners'][coordinate]['selection'],
            selections={rank: owner['selection'] for rank, owner in after['owners'].items()})}
        declared.append(dict(expression=expression, ports=ports, source=source_key, result=output_key,
            transition=transition, before=transition['side'] != 'results'))
    return tuple(declared), replacements, result_versions, tuple(pending)


def lower_owned_graph(graph, arguments, configuration):
    """Declare owners and demands over one retained global source program.

    Source schemas describe placements and required uses. They never execute
    local tensor operations to construct another numerical graph. The retained
    numerical relations select work and pages for every declared rank.
    """
    import operator
    from .ownership import (from_spec, propagate, prepare_expression, compile_assignments, contribution_domains,
                            source_uses, bind_ports, prepare_sources, source_ports)
    from .compilation import normalize_source
    from .algebra import partition_decompositions
    mesh = configuration['mesh']
    def declared_node(value):
        return value if isinstance(value, torch.fx.Node) else next(
            node for node in graph.graph.nodes if node.name == value or node.meta.get('mesh_source') == value)
    routes = []
    for route in configuration.get('deliveries', ()):
        source = route['source']
        node, path = source if isinstance(source, tuple) else (source, ())
        routes.append(dict(route, source=(declared_node(node), tuple(path)),
                           consumer=None if route['consumer'] is None else declared_node(route['consumer'])))
    for name, declaration in configuration.get('values', {}).items():
        declare_result(graph, name, declaration, mesh=mesh)
    local_nodes = set()
    for node in graph.graph.nodes:
        if node.op != 'output' and ('mesh_local_source' in node.meta or
                                   any(source in local_nodes for source in node.all_input_nodes)):
            local_nodes.add(node)
            node.meta['mesh_local_scope'] = True
    graph = normalize_source(graph, partition_decompositions, preserve=local_nodes)
    for route in routes:
        node, path = route['source']
        mapped = graph.meta.get('mesh_source_results', {}).get(node, node)
        route['source'] = (mapped, path) if isinstance(mapped, torch.fx.Node) else (pytree.key_get(mapped, path), ())
    nodes = list(graph.graph.nodes)
    placeholders = [node for node in nodes if node.op == 'placeholder']
    arguments = tuple(graph.graph.process_inputs(*arguments))
    leaf = lambda value: isinstance(value, dict) and 'owners' in value
    records = {node: node.meta['mesh_ownership'] for node in nodes if 'mesh_ownership' in node.meta}
    allocated = set(records)
    materialized = set(allocated)
    uses, crossings, cuts, scopes = {}, [], set(), {}
    pending = []

    def record_map(function, value):
        return pytree.tree_map(function, value, is_leaf=leaf)

    def fields(node):
        return [(tuple(path), value) for path, value in pytree.tree_flatten_with_path(node.meta['val'])[0]
                if isinstance(value, torch.Tensor)]

    def owners(node):
        record = records.get(node)
        return {} if record is None else {tuple(path): value for path, value in
            pytree.tree_flatten_with_path(record, is_leaf=leaf)[0] if leaf(value)}

    def types(value):
        return pytree.tree_map(lambda item: TensorMeta(item.shape, item.stride(), item.dtype)
                               if isinstance(item, torch.Tensor) else None, value)

    for node, value in zip(placeholders, arguments, strict=True):
        if isinstance(value, DTensor) and node not in records:
            records[node] = from_spec(value._spec)
        if isinstance(value, torch.Tensor) and node not in records:
            raise ValueError(f'{node.name}: distributed input requires an explicit owner declaration')
    external = {node: records.get(node) for node in placeholders}

    def same(left, right):
        if not _same_ownership(left, right):
            return False
        a, b = left.get('spec'), right.get('spec')
        if a is not None and b is not None:
            return _same(a.placements, b.placements)
        first, second = left.get('reductions', ()), right.get('reductions', ())
        return tuple(axis for axis, _ in first) == tuple(axis for axis, _ in second) and _same(
            tuple(placement for _, placement in first), tuple(placement for _, placement in second))

    def schema_value(value):
        if not isinstance(value, torch.fx.Node):
            return value
        if value in records:
            return record_map(lambda item: item.get('spec') if leaf(item) else item, records[value])
        return pytree.tree_map(lambda item: None if isinstance(item, torch.Tensor) else item, value.meta['val'])

    def complete_value(value):
        if not leaf(value) or not value.get('reductions'):
            return value
        spec = value.get('spec')
        if spec is not None:
            spec = _with(spec, tuple(Replicate() if placement.is_partial() else placement
                                     for placement in spec.placements))
        return dict(value, reductions=(), spec=spec)

    def complete_inputs(value):
        if not isinstance(value, torch.fx.Node):
            return value
        if value in records:
            return record_map(complete_value, records[value])
        return pytree.tree_map(lambda item: None if isinstance(item, torch.Tensor) else item, value.meta['val'])

    def same_value(left, right):
        first, second = left.get('reductions', ()), right.get('reductions', ())
        return tuple(axis for axis, _ in first) == tuple(axis for axis, _ in second) and _same(
            tuple(placement for _, placement in first), tuple(placement for _, placement in second))

    def declare_uses(consumer, side, original, required, path=()):
        if isinstance(original, torch.fx.Node):
            source_records = owners(original)
            for field, value in fields(original):
                source = source_records.get(field)
                spec = pytree.key_get(required, field) if required is not None else None
                target = from_spec(spec) if isinstance(spec, DTensorSpec) else spec if leaf(spec) else source
                key = (consumer, side, path, field)
                uses[key] = dict(source=(original, field), provided=source, required=target,
                                 schema=spec.get('spec') if leaf(spec) else spec,
                                 consumer=consumer, side=side, path=path)
                if source is not None and target is not None and not same(source, target):
                    transition = dict(uses[key], mesh=mesh, order=len(crossings))
                    crossings.append(transition)
                    materialized.add(original)
                    cuts.add((consumer, 'before'))
                    # Placement value expressions preserve masks, scale factors
                    # and postprocessing. Their source edges must be declared
                    # before a selected crossing can consume their results.
            return
        if isinstance(original, (tuple, list)):
            for index, value in enumerate(original):
                item = required[index] if required is not None and index < len(required) else None
                declare_uses(consumer, side, value, item, (*path, pytree.SequenceKey(index)))
        elif isinstance(original, dict):
            for name, value in original.items():
                declare_uses(consumer, side, value, None if required is None else required.get(name),
                             (*path, pytree.MappingKey(name)))

    for node in nodes:
        node.meta['mesh_source_nodes'] = tuple(dict.fromkeys((node, *node.meta.get('mesh_source_nodes', ()))))
        if node.op in ('placeholder', 'get_attr', 'output'):
            continue
        if node.meta.get('mesh_local_scope'):
            declaration = node.meta.get('mesh_local_source')
            source = None
            if declaration is not None:
                original, path = declaration['source']
                mapped = graph.meta.get('mesh_source_results', {}).get(original, original)
                source = (mapped, path) if isinstance(mapped, torch.fx.Node) else (pytree.key_get(mapped, path), ())
                records.setdefault(node, declaration)
            output_owners = owners(node)
            if output_owners:
                # The value's public coordinates come from the explicit type;
                # source_uses supplies each rank's local numerical type.
                value = node.meta['val']
                def global_value(path, item):
                    owner = output_owners.get(tuple(path))
                    if owner is None:
                        return item
                    with item.fake_mode:
                        return torch.empty_strided(owner['type'].shape, owner['type'].stride,
                            dtype=owner['type'].dtype, device=item.device, requires_grad=item.requires_grad)
                node.meta['val'] = pytree.tree_map_with_path(global_value, value)
            scopes[node] = dict(uses=(), outputs=output_owners, sharding=None, source=source)
            continue
        required = None
        assigned = records.get(node) if node in allocated else None
        complete_work = (assigned is not None and len(owners(node)) == len(fields(node))
                         and all(not owner.get('reductions') for owner in owners(node).values()))
        if complete_work:
            # These are the captured global equations with an explicit output
            # assignment. Local sharding strategies cannot replace that work.
            # Partial-valued inputs still need their complete source expression.
            args, kwargs = pytree.tree_map(complete_inputs, (node.args, node.kwargs))
            if node.target is operator.getitem and node.args[0] in records and not leaf(records[node.args[0]]):
                # A structural result-field selection consumes only that field.
                source = records[node.args[0]]
                selected = node.args[1]
                values_ = dict(source) if isinstance(source, dict) else list(source)
                values_[selected] = record_map(complete_value, source[selected])
                args = (tuple(values_) if isinstance(source, tuple) else values_, *args[1:])
            required = OpSchema(node.target, args, kwargs)
        elif node.target is operator.getitem and node.args[0] in records and not leaf(records[node.args[0]]):
            records.setdefault(node, records[node.args[0]][node.args[1]])
        else:
            inputs = [record for source in node.all_input_nodes for record in owners(source).values()]
            complete = (inputs and all(record.get('spec') is not None for record in inputs)
                        and all(len(owners(source)) == len(fields(source)) for source in node.all_input_nodes))
            if complete:
                args, kwargs = pytree.tree_map(schema_value, (node.args, node.kwargs))
                sharding = propagate(node.target, args, kwargs, mesh, result_types=types(node.meta['val']))
                node.meta['mesh_sharding'] = sharding
                produced = pytree.tree_map_only(DTensorSpec, from_spec, sharding.output_spec)
                records[node] = assigned if assigned is not None else produced
                if assigned is not None:
                    for path, target in pytree.tree_flatten_with_path(assigned, is_leaf=leaf)[0]:
                        source = pytree.key_get(produced, path)
                        if leaf(target) and leaf(source) and not same_value(source, target):
                            source = dict(source, owners=target['owners'], dimensions=target['dimensions'])
                            transition = dict(source=(node, tuple(path)), provided=source, required=target,
                                consumer=node, side='results', path=tuple(path), mesh=mesh, order=len(crossings))
                            crossings.append(transition)
                            materialized.add(node)
                            cuts.add((node, 'after'))
                required = sharding.redistribute_schema
            elif assigned is not None:
                pending.append(dict(source=node, declaration=assigned, requirement='placement_value_relation'))
        declare_uses(node, 'args', node.args, None if required is None else required.args_schema)
        declare_uses(node, 'kwargs', node.kwargs, None if required is None else required.kwargs_schema)
        if not complete_work and 'mesh_sharding' in node.meta:
            declared_uses = [use for use in uses.values() if use['consumer'] is node
                             and use['provided'] is not None and use['required'] is not None]
            local_types = tuple(use for use in declared_uses if
                tuple(use['required']['owners'][tuple(mesh.get_coordinate())]['shape']) !=
                tuple(pytree.key_get(use['source'][0].meta['val'], use['source'][1]).shape))
            output_owners = {tuple(path): owner for path, owner in
                pytree.tree_flatten_with_path(produced, is_leaf=leaf)[0] if leaf(owner)}
            local_outputs = any(tuple(owner['owners'][tuple(mesh.get_coordinate())]['shape']) !=
                tuple(pytree.key_get(node.meta['val'], path).shape) for path, owner in output_owners.items())
            if local_types or local_outputs:
                # The existing source rule declares a local numerical scope.
                # In particular, placement expressions may change index values;
                # their consumer must see the required local tensor type.
                scopes[node] = dict(uses=tuple(declared_uses), outputs=output_owners,
                    sharding=node.meta['mesh_sharding'])

    result = next(node.args[0] for node in nodes if node.op == 'output')
    result_nodes = tuple(dict.fromkeys(item for item in pytree.tree_leaves(result) if isinstance(item, torch.fx.Node)))
    observable = materialized.difference(placeholders).union(result_nodes)
    requested_nodes = allocated.difference(placeholders).union(result_nodes)
    values = {(node, path): owner for node in nodes for path, owner in owners(node).items()}
    programs, replacements, result_versions, unresolved = _transition_sources(crossings, values, mesh)
    pending.extend(unresolved)
    requests = {rank: {} for rank in _coordinates(mesh)}
    for node in requested_nodes:
        requested = records.get(node)
        declared = {} if requested is None else {tuple(path): owner for path, owner in
            pytree.tree_flatten_with_path(requested, is_leaf=leaf)[0] if leaf(owner)}
        for path, value in fields(node):
            if path not in declared:
                raise ValueError(f'{node.name}{path}: distributed result requires an explicit owner declaration')
            for rank, owner in declared[path]['owners'].items():
                requests[rank][(node, path)] = owner
    metadata = tuple(node.meta['val'] for node in placeholders)
    variants = {}
    for rank in (_coordinates(mesh) if scopes else (tuple(mesh.get_coordinate()),)):
        expression, ports, source_nodes = source_uses(graph, replacements, result_versions, scopes=scopes, coordinate=rank)
        expression.update(materializations=(*expression['materializations'],
                              *(source_nodes[node] for node in nodes if node in observable)),
                          cuts=(*expression['cuts'], *((source_nodes[node], side) for node, side in
                              sorted(cuts, key=lambda item: nodes.index(item[0])))))
        if variants:
            expression['algebra'] = next(iter(variants.values()))['algebra']
        prepared = prepare_expression(expression)
        normalized_ports = source_ports(prepared, ports)
        variants[rank] = dict(prepared, sections=tuple(bind_ports(section, normalized_ports, prepared['algebra'])
                                                 for section in prepared['sections']),
                        external_fields=tuple((node, path) for node in placeholders for path, _ in fields(node)))
    prepared = variants[tuple(mesh.get_coordinate())]
    expression = prepared['expression']
    if scopes:
        shared = []
        for members in zip(*(value['sections'] for value in variants.values()), strict=True):
            wrapper = dict(members[0])
            wrapper['rank_sections'] = {rank: dict(section, source_section=wrapper)
                for rank, section in zip(variants, members, strict=True)}
            shared.append(wrapper)
        prepared = dict(prepared, sections=tuple(shared))
    programs = prepare_sources(programs, prepared['algebra']) if programs else ()
    for route in routes:
        node, path = route['source']
        while node in prepared['aliases']:
            node = prepared['aliases'][node]
        route['source'] = node, path
        while route['consumer'] in prepared['aliases']:
            route['consumer'] = prepared['aliases'][route['consumer']]
    def delivery(section, key, needs, algebra):
        candidates = [route for route in routes if route['source'] == key and
            (route['consumer'] is None if section is None else
             route['consumer'] in section['graph'].meta.get('mesh_source_results', {}))]
        if len(candidates) > 1:
            raise ValueError(f'{key[0].name}: multiple declared delivery-source relations for one consumer section')
        record = _delivery_domains(section, key, needs, values.get(key), mesh, algebra,
                                   routing=candidates[0]['sources'] if candidates else None)
        pending.extend(record['pending'])
        return record
    def read_transition(site, ports, algebra):
        from .compilation import source_expression
        from mesh_relation import flat
        described = site['fields'][0].meta['val']
        shape = tuple(described.shape)
        flattened = algebra.map(shape, (described.numel(),),
            lambda x, y: [f'{y[0]} = {flat(shape, x)}'])
        owners = {}
        for rank, port in ports.items():
            boundary = port['boundary']
            positions = (algebra.set('{ [i] : false }', 1) if boundary is None else
                         flattened.image(boundary['work']))
            selection = dict(shape=list(shape), steps=[['reshape', [described.numel()]],
                                                       ['select', 0, positions.spans()]])
            owners[rank] = dict(selection=selection, shape=algebra.selection_relation(selection).shape)
        source, result = ((node, ()) for node in site['fields'])
        dimensions = site['dimensions']
        partial = tuple(Partial('sum') if axis in dimensions else Replicate() for axis in range(mesh.ndim))
        complete = (Replicate(),) * mesh.ndim
        selecting = {rank: owner['selection'] for rank, owner in owners.items()}
        fixed = tuple(axis for axis in range(mesh.ndim) if axis not in dimensions)
        cohorts = {}
        for rank in owners:
            cohorts.setdefault(tuple(rank[axis] for axis in fixed), []).append(rank)
        variants = {}
        group_axes = {mesh.get_group(axis).group_name: axis for axis in dimensions}
        for cohort in cohorts.values():
            coordinate = cohort[0]
            local_shape = owners[coordinate]['shape']
            if any(owners[rank]['shape'] != local_shape for rank in cohort):
                raise ValueError('a selected-read cohort must share its declared union-work ordinals')
            with described.fake_mode:
                local = torch.empty(local_shape, dtype=described.dtype, device=described.device)
            width = math.prod(local_shape) * described.dtype.itemsize
            meta = TensorMeta(torch.Size((width,)), (1,), torch.uint8)
            before, after = DTensorSpec(mesh, partial, tensor_meta=meta), DTensorSpec(mesh, complete, tensor_meta=meta)
            def declaration(value):
                bits = value.reshape(-1).view(torch.uint8)
                received = _local_redistribute(bits, before, after)
                return received.view(described.dtype).reshape(local_shape)
            module, operands = source_expression(declaration, local)
            argument = next(node for node in module.graph.nodes if node.op == 'placeholder')
            output = next(node.args[0] for node in module.graph.nodes if node.op == 'output')
            bindings = {(argument, ()): dict(field=source, selection=owners[coordinate]['selection'], selections=selecting),
                        (output, ()): dict(field=result, selection=owners[coordinate]['selection'], selections=selecting)}
            expression = dict(graph=module, operands=tuple(operands), algebra=algebra)
            captured = prepare_expression(expression)
            normalized = source_ports(captured, bindings)
            captured = dict(captured, sections=tuple(dict(bind_ports(span, normalized, algebra), group_axes=group_axes)
                                                     for span in captured['sections']))
            variants.update((rank, captured) for rank in cohort)
        captured = variants[tuple(mesh.get_coordinate())]
        shared = []
        for members in zip(*(value['sections'] for value in variants.values()), strict=True):
            wrapper = dict(members[0])
            wrapper['rank_sections'] = {rank: dict(member, source_section=wrapper)
                for rank, member in zip(variants, members, strict=True)}
            shared.append(wrapper)
        captured = dict(captured, sections=tuple(shared))
        typ = TensorMeta(described.shape, described.stride(), described.dtype)
        # The typed prefix returns masked values. Its byte view, inside the
        # captured declaration, carries the SUM semantics; the float/int source
        # field itself is never mislabeled as a numerical Partial(sum).
        values[source] = dict(type=typ, owners=owners, dimensions=dimensions,
                             reductions=(), mesh=mesh, spec=None)
        values[result] = dict(type=typ, owners=owners, dimensions=(), reductions=(), mesh=mesh, spec=None)
        return dict(source=source, result=result, prepared=captured, state_outputs=())
    constraints = {}
    for rank in requests:
        limits, unresolved = contribution_domains(prepared, values, rank, uses=uses)
        constraints[rank] = limits
        pending.extend(dict(item, coordinate=rank) for item in unresolved)
    selected = compile_assignments(prepared,
        {rank: {key: owner['owners'][rank]['selection'] for key, owner in values.items()} for rank in requests},
        {rank: {key: owner['selection'] for key, owner in requested.items()} for rank, requested in requests.items()},
        constraints=constraints,
        delivery=delivery,
        boundary_projection=lambda section, needs, algebra: _project_owned_boundaries(section, needs, algebra, mesh),
        transitions=programs, read_transition=read_transition)
    projections, deliveries = selected['projections'], selected['deliveries']
    pending.extend(dict(item, coordinate=rank) for rank, projection in projections.items() for item in projection['pending'])
    local = pytree.tree_map(lambda value: value._local_tensor if isinstance(value, DTensor) else value, arguments)
    return dict(graph=graph, arguments=local, metadata=metadata, mesh=mesh, values=values, uses=uses,
                requests=requests, expression=expression, prepared=prepared, projections=projections,
                crossings=tuple(crossings), deliveries=tuple(deliveries), pending=tuple(pending), external=external, result=result)


def _unfold(op, args, kwargs):
    """Declare the window's output ownership and exact input-page relation."""
    from mesh_relation import Algebra, flat
    x, dimension, size, step = args
    dimension %= x.ndim
    if not any(p.is_shard() and p.dim == dimension for p in x.placements):
        return _propagated(op, args, kwargs)
    shape = list(x.shape)
    shape[dimension] = (shape[dimension] - size) // step + 1
    shape.append(size)
    described = op(torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device='meta'), dimension, size, step)
    target = DTensorSpec(x.device_mesh, x.placements, tensor_meta=TensorMeta(described.shape, described.stride(), x.dtype))
    algebra = Algebra()
    access = algebra.window(shape, tuple(x.shape), axes=[(dimension, dimension, x.ndim)],
        strides=[step], dilations=[1], padding=[0], transposed=False,
        equations=lambda a, b: [f'{a[i]} = {b[i]}' for i in range(x.ndim) if i != dimension])
    source = algebra.map(tuple(x.shape), (x.numel(),), lambda a, b: [f'{flat(x.shape, a)} = {b[0]}'])
    order = algebra.map((math.prod(shape),), shape, lambda a, b: [f'{a[0]} = {flat(shape, b)}'])
    selection = {'shape': list(x.shape), 'steps': [['reindex', {'shape': shape, 'index': order.then(access).then(source).terms()}]]}
    return _exchange_view(x, target, selection)


def _slice_view(op, args, kwargs):
    """A static view cut uses the same selected-page crossing as a window."""
    from mesh_relation import Algebra, flat
    named = _named(op, args, kwargs)
    x = named['self']
    dim = named['dim'] % x.ndim
    if not any(p.is_shard() and p.dim == dim for p in x.placements):
        return _propagated(op, args, kwargs)
    described = op(torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device='meta'), *args[1:], **kwargs)
    selecting = op is aten.select.int
    if selecting:
        first = named['index'] % x.shape[dim]
        placements = [Replicate() if p.is_shard() and p.dim == dim else
                      RaggedShard(p.dim - int(p.dim > dim), p.key) if isinstance(p, RaggedShard) else
                      Shard(p.dim - int(p.dim > dim)) if p.is_shard() else p for p in x.placements]
        equations = lambda a, b: [f'{b[dim]} = {first}', *[f'{a[i - int(i > dim)]} = {b[i]}' for i in range(x.ndim) if i != dim]]
    else:
        first, _, step = slice(named.get('start'), named.get('end'), named.get('step', 1)).indices(x.shape[dim])
        placements = x.placements
        equations = lambda a, b: [f'{b[i]} = {first} + {a[i]} * {step}' if i == dim else f'{a[i]} = {b[i]}' for i in range(x.ndim)]
    target = DTensorSpec(x.device_mesh, tuple(placements), tensor_meta=TensorMeta(described.shape, described.stride(), x.dtype))
    algebra = Algebra()
    access = algebra.map(tuple(described.shape), tuple(x.shape), equations)
    source = algebra.map(tuple(x.shape), (x.numel(),), lambda a, b: [f'{flat(x.shape, a)} = {b[0]}'])
    order = algebra.map((described.numel(),), tuple(described.shape), lambda a, b: [f'{a[0]} = {flat(described.shape, b)}'])
    selection = {'shape': list(x.shape), 'steps': [['reindex', {'shape': list(described.shape), 'index': order.then(access).then(source).terms()}]]}
    return _exchange_view(x, target, selection)


def _reshape(op, args, kwargs):
    """Compose declared view regions before applying a rank's local shape.

    Ordinary Shard cannot express every mapped page selection. A changed
    ownership relation remains an explicit collective in the declared graph.
    """
    x = args[0]
    described = op(torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device='meta'), *args[1:], **kwargs)
    meta = TensorMeta(described.shape, described.stride(), x.dtype)
    placements = list(x.placements)
    for i, p in enumerate(x.placements):
        if not p.is_shard():
            continue
        for dim in range(described.ndim):
            candidate = RaggedShard(dim, p.key) if isinstance(p, RaggedShard) else Shard(dim)
            target = DTensorSpec(x.device_mesh, (*placements[:i], candidate, *placements[i + 1:]), tensor_meta=meta)
            coordinates = itertools.product(*(range(x.device_mesh.size(axis)) for axis in range(i + 1)))
            if all(_view_region(x._spec, rank).equivalent(_view_region(target, rank)) for rank in coordinates):
                placements[i] = candidate
                break
        else:
            source = redistribute(x, [Replicate() if axis == i else p for axis, p in enumerate(x.placements)])
            return op(source, *args[1:], **kwargs)
    shape = list(described.shape)
    for i, p in enumerate(placements):
        if p.is_shard():
            shape[p.dim] = p._local_shard_size_and_offset(shape[p.dim], x.device_mesh.size(i), x.device_mesh.get_local_rank(i))[0]
    local = op(x._local_tensor, shape)
    result = _wrap(local, DTensorSpec(x.device_mesh, tuple(placements), tensor_meta=meta), local.requires_grad)
    return return_and_correct_aliasing(op, args, kwargs, result) if op._schema._is_view_op() else result


def _unfit(specs, keys):
    """Collective dimensions with a partial operation not yet lowered here."""
    dims = set()
    for s in specs:
        for i, p in enumerate(s.placements):
            if i not in keys:
                continue
            if p.is_partial() and not (type(p) is Partial and p.reduce_op in _REMASKED):
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
    if op_call in (aten.convolution.default, aten.convolution_backward.default):
        raise ValueError('distributed convolution requires result ownership on the Function source graph')
    if op_call is aten.index_select.default:
        return _index_select(op_call, args, kwargs)
    from .algebra import partition_decompositions
    if op_call in partition_decompositions:
        return partition_decompositions[op_call](*args, **kwargs)
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
    whole = _unfit(_specs(spec), keys)
    whole |= set(keys) if op_call in sp.op_to_scalar_shape_adjuster else set()
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
    elif op in (aten.amax.default, aten.amin.default) and local_args[0].numel() == 0:
        named = _named(op, local_args, info.local_kwargs)
        x, dims = named['self'], named.get('dim')
        dims = {d % x.ndim for d in dims} if dims else set(range(x.ndim))
        shape = tuple(1 if axis in dims else size for axis, size in enumerate(x.shape)) if named.get('keepdim', False) else tuple(
            size for axis, size in enumerate(x.shape) if axis not in dims)
        low, high = ((-math.inf, math.inf) if x.is_floating_point() else
                     (False, True) if x.dtype == torch.bool else (torch.iinfo(x.dtype).min, torch.iinfo(x.dtype).max))
        result = x.new_full(shape, low if op is aten.amax.default else high)
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
    return {a.name: args[i] if i < len(args) else kwargs.get(a.name, a.default_value)
            for i, a in enumerate(op_call._schema.arguments)}


def _index_select(op_call, args, kwargs):
    """Select owned values and combine their bytes, never their numeric values.

    Exactly one owner supplies each selected value. Integer-byte SUM therefore
    moves its representation unchanged, including signed zero and NaN payloads.
    The index tensor remains ordinary runtime data; only its page relationships
    and output width are fixed during preparation.
    """
    named = _named(op_call, args, kwargs)
    x, dim, index = named['self'], named['dim'] % named['self'].ndim, named['index']
    if not isinstance(x, DTensor) or not any(p.is_shard() and p.dim == dim for p in x.placements):
        return _propagated(op_call, args, kwargs)
    reduced = [i for i, p in enumerate(x.placements) if p.is_shard() and p.dim == dim]
    requested = list(x.placements)
    index_placements = list(index.placements) if isinstance(index, DTensor) else [Replicate()] * x.device_mesh.ndim
    for i, p in enumerate(requested):
        if i in reduced:
            q = index_placements[i]
            requested[i] = RaggedShard(dim, q.key) if isinstance(q, RaggedShard) else Shard(dim) if q.is_shard() else Replicate()
        elif p.is_replicate() and index_placements[i].is_shard():
            q = index_placements[i]
            requested[i] = RaggedShard(dim, q.key) if isinstance(q, RaggedShard) else Shard(dim)
    if isinstance(index, DTensor):
        index = redistribute(index, [Replicate() if i in reduced or x.placements[i].is_shard() else p
                                     for i, p in enumerate(index.placements)])._local_tensor
    first, extent = 0, x.shape[dim]
    for i, p in enumerate(x.placements):
        if p.is_shard() and p.dim == dim:
            extent, offset = p._local_shard_size_and_offset(extent, x.device_mesh.size(i), x.device_mesh.get_local_rank(i))
            first += offset
    shape = list(x.shape)
    shape[dim] = named['index'].numel()
    local_shape = list(x._local_tensor.shape)
    local_shape[dim] = index.numel()
    if extent:
        selected = torch.index_select(x._local_tensor, dim, (index - first).clamp(0, extent - 1))
        raw = selected.contiguous().view(torch.uint8).reshape((*local_shape, x.element_size()))
        mask_shape = [1] * (x.ndim + 1)
        mask_shape[dim] = index.numel()
        owned = ((index >= first) & (index < first + extent)).reshape(mask_shape)
        raw = torch.where(owned, raw, 0)
    else:
        raw = x._local_tensor.new_zeros((*local_shape, x.element_size()), dtype=torch.uint8)
    for i in reduced:
        raw = funcol.all_reduce(raw, 'sum', x.device_mesh.get_group(i))
    local = raw.reshape((*local_shape[:-1], local_shape[-1] * x.element_size())).view(x.dtype).reshape(local_shape)
    held = [Replicate() if i in reduced else p for i, p in enumerate(requested)]
    result = _wrap(local, DTensorSpec(x.device_mesh, tuple(held),
                   tensor_meta=TensorMeta(torch.Size(shape), torch.empty(shape, device='meta').stride(), x.dtype)))
    return redistribute(result, requested)


def _arg_extreme(op_call, args, kwargs):
    """Reduce values and first global positions without moving the input pages."""
    named = _named(op_call, args, kwargs)
    x, dim, keep = named['self'], named.get('dim'), named.get('keepdim', False)
    if dim is None:
        result = _arg_extreme(op_call, (x.reshape(-1), 0, False), {})
        return result.reshape((1,) * x.ndim) if keep else result
    dim %= x.ndim
    if not any(isinstance(p, RaggedShard) and p.dim == dim for p in x.placements):
        return _propagated(op_call, args, kwargs)
    x = redistribute(x, [Replicate() if p.is_partial() else p for p in x.placements])
    reduced = [i for i, p in enumerate(x.placements) if p.is_shard() and p.dim == dim]
    first, extent = 0, x.shape[dim]
    for i in reduced:
        extent, offset = x.placements[i]._local_shard_size_and_offset(extent, x.device_mesh.size(i), x.device_mesh.get_local_rank(i))
        first += offset
    maximum = op_call is aten.argmax.default
    local = x._local_tensor
    shape = list(local.shape)
    shape[dim] = 1
    neutral = (-math.inf if maximum else math.inf) if x.is_floating_point() else (
        (False if maximum else True) if x.dtype == torch.bool else
        (torch.iinfo(x.dtype).min if maximum else torch.iinfo(x.dtype).max))
    nan = torch.isnan(local) if x.is_floating_point() else None
    values = torch.where(nan, neutral, local) if nan is not None else local
    best = (values.amax if maximum else values.amin)(dim, keepdim=True) if extent else local.new_full(shape, neutral)
    def combine(value, operation):
        for i in reduced:
            value = funcol.all_reduce(value, operation, x.device_mesh.get_group(i))
        return value
    best = combine(best, 'max' if maximum else 'min')
    matches = local == best
    if nan is not None:
        seen = nan.to(torch.uint8).amax(dim, keepdim=True) if extent else local.new_zeros(shape, dtype=torch.uint8)
        matches = torch.where(combine(seen, 'max').bool(), nan, matches)
    index_shape = [1] * local.ndim
    index_shape[dim] = extent
    positions = torch.arange(first, first + extent, dtype=torch.int64, device=local.device).reshape(index_shape)
    candidate = torch.where(matches, positions, x.shape[dim]).amin(dim, keepdim=keep) if extent else torch.full(
        shape if keep else shape[:dim] + shape[dim + 1:], x.shape[dim], dtype=torch.int64, device=local.device)
    result = combine(candidate, 'min')
    placements = []
    for i, p in enumerate(x.placements):
        if i in reduced:
            p = Replicate()
        elif p.is_shard() and not keep and p.dim > dim:
            p = RaggedShard(p.dim - 1, p.key) if isinstance(p, RaggedShard) else Shard(p.dim - 1)
        placements.append(p)
    logical = list(x.shape)
    logical[dim] = 1
    if not keep:
        logical.pop(dim)
    return _wrap(result, DTensorSpec(x.device_mesh, tuple(placements), tensor_meta=TensorMeta(
        torch.Size(logical), torch.empty(logical, device='meta').stride(), torch.int64)))


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
                               value._local_tensor, named.get('is_causal', False), named.get('scale'), 1, sequence_length=query.shape[2])
    out = RaggedTensor(out, _spec_like(spec, query.shape[:-1] + value.shape[-1:], out.dtype), requires_grad=False)
    if op_call is aten._scaled_dot_product_attention_math_for_mps.default:
        return out, None
    return out, RaggedTensor(lse, _spec_like(spec, query.shape[:-1], lse.dtype), requires_grad=False)


def _detached(op_call, args, kwargs):
    """detach (nn.Parameter makes a parameter by it) of a tensor split along a partitioned dimension, registered as
    its source is (_register), so the parameter is the one re-laid."""
    out = _propagated(op_call, args, kwargs)
    return _register(out) if _registered(args[0]) else out


_HANDLERS = {aten.argmax.default: _arg_extreme, aten.argmin.default: _arg_extreme,
             aten.unfold.default: _unfold, aten.slice.Tensor: _slice_view, aten.select.int: _slice_view,
             aten.view.default: _reshape, aten.reshape.default: _reshape, aten._unsafe_view.default: _reshape,
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
        if (not isinstance(x, DTensor) and isinstance(mod, nn.Linear)
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
def _halves(load_balance):
    """The declared context placement, independent of sequence divisibility."""
    return 2 if load_balance else 1


def _positions(key, rank, S, halves, device):
    """Prepared logical positions, with no padding or invented valid lanes."""
    o = _OPERANDS[key]

    def build(p):
        spans = _spans(o, rank, S, halves, p)
        return torch.cat([torch.arange(lo, hi) for lo, hi in spans]).to(device)
    return o.cached(('positions', rank, S, halves, str(device)), build)


def _spans(o, rank, S, halves, parts=None):
    """The logical positions [lo, hi) rank `rank` holds in each half of its block (_positions), from `parts` (the
    operand's own where None)."""
    p = o.capacity if parts is None else parts
    head = partition_bounds(p, S // halves)
    spans = [(head[rank], head[rank + 1])]
    if halves == 2:
        tail = partition_bounds(p, S - S // halves)
        spans.append((S - tail[rank + 1], S - tail[rank]))
    return spans


def _context_lengths(o, S, halves):
    return [sum(hi - lo for lo, hi in _spans(o, rank, S, halves)) for rank in range(len(o.capacity))]


def context_parallel_buffers(mesh, name, buffers, seq_dims, load_balance=True):
    """Select exact contiguous or head-tail spans from each declared buffer."""
    o, rank, shards = operand(mesh, name), mesh.get_local_rank(name), []
    for b, dim in zip(buffers, seq_dims):
        if not isinstance(b, torch.Tensor):
            raise NotImplementedError('mesh: a BlockMask buffer (flex attention) on a partitioned mesh dimension')
        S = b.size(dim)
        index = _positions(o.key, rank, S, _halves(load_balance), b.device)
        shards.append(b.index_select(dim, index))
    return shards


@torch.no_grad()
def context_parallel_unshard(mesh, name, buffers, seq_dims, load_balance=True, *, sequence_length):
    """Gather context blocks through the common exact physical WRITE program.

    The ordinary contiguous sequence layout is the declared final destination.
    A head/tail partition gathers the same exact contributions, then permutes
    its two logical halves locally; no page-padded transport buffer is used.
    Original input strides bind the common indexed WRITE plan. Supply the
    declared sequence_length on every rank; rounded local extents do not determine it.
    """
    o, rank, group = operand(mesh, name), mesh.get_local_rank(name), mesh.get_group(name)
    S = sequence_length
    halves = _halves(load_balance)
    lengths, joined = _context_lengths(o, S, halves), []
    for b, dim in zip(buffers, seq_dims):
        dim %= b.ndim
        gathered = allgatherv(b, dim, lengths, group.group_name)
        whole = gathered
        if halves != 1:
            pieces = []
            for r, at in enumerate(itertools.accumulate(lengths[:-1], initial=0)):
                for lo, hi in _spans(o, r, S, halves):
                    pieces.append((lo, gathered.narrow(dim, at, hi - lo)))
                    at += hi - lo
            whole = torch.cat([value for _, value in sorted(pieces, key=lambda item: item[0])], dim)
        joined.append(whole)
    return joined


def ring_attention(mesh, name, query, key, value, is_causal=False, scale=None, load_balance=True, *, sequence_length):
    """This rank's query block's attention over every rank's keys and values, the blocks along dimension 2 (the
    sequence) as context_parallel_buffers made them over partitioned mesh dimension `name`: (output, each query row's
    logsumexp). sequence_length supplies the AOT logical extent on every rank. A zero-share rank supplies
    no partial input but remains in the physical communicator."""
    o = operand(mesh, name)
    return _ring_attention(mesh.get_group(name), o.key, query, key, value, is_causal, scale, _halves(load_balance),
                           sequence_length=sequence_length)



SCORES = 1 << 26
def _block(query, key, value, scale=None, queries=None, keys=None):
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
        top = scores.amax(-1, keepdim=True)
        top = torch.where(torch.isfinite(top), top, 0)
        probabilities = (scores - top).exp()
        parts.append(torch.cat((top, probabilities.sum(-1, keepdim=True),
                                (probabilities.to(value.dtype) @ value).float()), -1))
    return torch.cat(parts, -2) if parts else query.new_empty((*lead, 0, value.shape[-1] + 2), dtype=torch.float32)


def _ring_attention(group, key_name, query, key, value, is_causal=False, scale=None, halves=1, *, sequence_length):
    """Declare query movement, split-KV work and the configured partial-row reduction.

    The whole function is traced before execution. Its tensor operations and
    crossings share the same compiler; there is no eager ring scheduler.
    """
    from .backend import merge_partials, attention
    o = _OPERANDS[key_name]
    rank = dist.get_rank(group)
    S = sequence_length
    lengths, lead = _context_lengths(o, S, halves), tuple(query.shape[:2])
    rows = [math.prod(lead) * length for length in lengths]
    gathered = allgatherv(query, 2, lengths, group.group_name)
    blocks = gathered.split(lengths, dim=2)
    keys = _positions(key_name, rank, S, halves, query.device)
    mine = _spans(o, rank, S, halves)
    key_offsets = list(itertools.accumulate((hi - lo for lo, hi in mine), initial=0))
    partials = []
    for owner, q in enumerate(blocks):
        positions = _positions(key_name, owner, S, halves, query.device)
        offset = 0
        local = []
        for lo, hi in _spans(o, owner, S, halves):
            seen = [b for b, (lo, _) in enumerate(mine) if not is_causal or lo < hi]
            at = slice(offset, offset + hi - lo)
            offset += hi - lo
            first, end = (key_offsets[seen[0]], key_offsets[seen[-1] + 1]) if seen else (0, 0)
            local.append(_block(q[:, :, at], key[:, :, first:end], value[:, :, first:end], scale,
                                positions[at] if is_causal else None, keys[first:end]))
        block = torch.cat(local, 2)
        partials.append(block.transpose(1, 2).reshape(rows[owner], value.shape[-1] + 2))
    merged = merge_partials(torch.cat(partials, 0), -1, group.group_name, rows)
    output, lse = attention(merged)
    return (output.view(lead[0], lengths[rank], lead[1], value.shape[-1]).transpose(1, 2),
            lse.view(lead[0], lengths[rank], lead[1]).transpose(1, 2))
