"""Distributed source operands and prepared page ownership.

A mesh dimension has configured relative shares. RaggedShard names those
shares; every distributed value retains its global type and rank-local pages.
Function captures numerical source once, including its backward. The ownership
pass carries exact page selections through the retained graph and compiles
only the intersections required by each consumer. A reshape or repartition
never requires whole replication merely because a single-axis placement cannot
name the resulting selection.

RaggedTensor is the common source operand for regular and ragged placements.
Its tensor operations are captured into Function; it has no eager numerical
executor. A source redistribute annotates a value with its chosen destination
owners. to_local declares a local view while preserving the global source edge.
Partial placements retain their mathematical expression, including masks and
scales, until their contributions and reductions are lowered together.

Initial distribution and changed configurations prepare ordinary declared
writes. Their storage and collective bindings are fixed before processing.
Execution consumes the same page operands as other Function frontends.
"""
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
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor import distribute_tensor as _distribute_tensor
from torch.distributed.tensor._collective_utils import mesh_broadcast
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.parallel import ColwiseParallel as _Colwise
from torch.distributed.tensor.parallel import RowwiseParallel as _Rowwise
from torch.distributed.tensor.parallel import SequenceParallel as _Sequence
from torch.distributed.tensor.placement_types import _StridedShard
from torch.utils import _pytree as pytree

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


def _empty(shape, like, group=None, dtype=None):
    """A region tensor (torch_mesh.empty) on `like`'s device, of its dtype or `dtype`: every buffer a collective here
    sends from or lands in is a lease, and a region that refuses one is the library's error, raised."""
    from . import empty
    return empty(tuple(shape), dtype=dtype or like.dtype, device=like.device.type)


# Restored (ebd8bd24 removed this definition and kept its two uses, at RaggedShard's local view and _scatter_spans:
# the chain service's weight scatter failed with NameError on main). What it says must stay: a contiguous tensor
# in the region is read where it lies.
def _in_region(y):
    """Whether contiguous `y` lies in the region already (torch_mesh.address), so a collective reads it where it lies."""
    from . import address
    return y.is_contiguous() and address(y) != 0


@torch.library.custom_op('torch_mesh::allgatherv', mutates_args=())
def allgatherv(x: torch.Tensor, dim: int, sizes: list[int], group_name: str, repeats: int = 1) -> torch.Tensor:
    """A graph declaration; the compiler binds each multi-rank occurrence."""
    raise RuntimeError('allgatherv must be lowered to an indexed crossing before execution')


@torch.library.custom_op('torch_mesh::allgatherv_out', mutates_args=('out',))
def allgatherv_out(x: torch.Tensor, out: torch.Tensor, dim: int, sizes: list[int], group_name: str,
                  repeats: int = 1) -> None:
    """The indexed gather declaration with an explicit output page operand."""
    raise RuntimeError('allgatherv_out requires whole-function preparation')


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


def _fused_form(x, dim, sizes, group_name, channels):
    """A chain crossing's prepared physical declarations: the communicator;
    this rank and its peers; the partial publication's kernel and grid; the reduction's (sum_land, and
    this rank's segment's start and each rank's sizes in its words); every rank's exact segment start and count
    and the output's shape; the publication's arrival count; the separate landing-prefix
    observer and physical channel extent."""
    from . import _stream
    group = dist.distributed_c10d._resolve_process_group(group_name)
    n, rank = len(sizes), group.rank()
    operand = tuple(x.shape)
    dim %= len(operand)
    copy_bytes = 1 if x.element_size() == 1 else 2
    row, half = math.prod(operand[:dim] + operand[dim + 1:]), x.element_size() // copy_bytes
    counts = [s * row for s in sizes]
    starts = list(itertools.accumulate(counts, initial=0))
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
    publish_words = [math.prod(operand) * half, 0, *x.shape, *x.stride()]
    publish = (_gather_kernel('partial_publish', x.ndim, half, copy_bytes, channels, x.is_contiguous()), publish_words, 1,
               spread(words[0]), _GATHER_THREADS)
    return (group.handle, _stream.polls, n, rank, *publish, land, words, 4, groups, _GATHER_THREADS,
            starts, counts, list(shape), torch.zeros(1, dtype=torch.int32, device='mps'),
            -1 if inline_wait else _gather_kernel('gather_ready', n, channels), channels)


from . import _stream as _native  # noqa: E402

_native.fused_maker(_fused_form)


@torch.library.custom_op('torch_mesh::reduce_scatterv', mutates_args=())
def reduce_scatterv(x: torch.Tensor, dim: int, sizes: list[int], rank: int, op: str, group_name: str) -> torch.Tensor:
    """Declare the selected output pages of a reduction across the group."""
    raise RuntimeError('reduce_scatterv requires whole-function preparation')


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


def gathered(x, dim, sizes, group, *, declaration=None):
    """The common collective compiler owns every gather's direct placement."""
    from .backend import ALLGATHER, Call
    counts = [s * _row(x, dim) for s in sizes]
    return [declaration if declaration is not None else Call(ALLGATHER, x.numel(), x.dtype, counts=counts)]


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


def plan(mesh, name, groups, *, production_seconds=None, routing=None):
    """The program's collectives on mesh dimension `name`'s group declared once, in their order (torch_mesh.plan):
    `groups` from the ops' own (calls, gathered, scattered), as a step issues them, or a function of no arguments
    giving them from the parts attached now, called when preparation supplies new parts. A list declares
    invariant counts; use a callable for every declaration whose counts depend on the current partition."""
    from . import plan as declare
    return declare(mesh.get_group(name), groups() if callable(groups) else groups,
                   production_seconds=production_seconds, routing=routing)


def split(mesh, name, N):
    """The valid blocks of an extent N along partitioned mesh dimension `name` under the current parts, in rank order."""
    o = operand(mesh, name)
    return _blocks(o, N)[0]


def _ragged(placements):
    return any(isinstance(p, RaggedShard) for p in placements)


def _is_shard(placement):
    return isinstance(placement, (Shard, _StridedShard))


def _same(a, b):
    """Placements alike, a RaggedShard's key and type included (torch's equality takes a RaggedShard for Shard(dim))."""
    a, b = tuple(a), tuple(b)
    return len(a) == len(b) and all(type(p) is type(q) and p == q and getattr(p, 'key', None) == getattr(q, 'key', None)
                                    for p, q in zip(a, b))


def _placement_key(placements):
    return tuple((type(p), p, getattr(p, 'key', None)) for p in placements)


def _local_redistribute(local, current, target, async_op=False):
    """Declare exact page exchanges, plus explicitly typed partial expressions.

    Ordinary placement changes compose all source and target page selections
    directly. Partial values retain their placement's numerical expression;
    capture lowers that expression and its crossings into the same function.
    """
    mesh = current.mesh
    if not mesh._is_current_rank_part_of_mesh() or _same(current.placements, target.placements):
        return local
    from .ownership import from_spec
    sources, destinations = list(current.placements), list(target.placements)
    sparse = {}
    for axis, (source, destination) in enumerate(zip(sources, destinations, strict=True)):
        if _same((source,), (destination,)):
            continue
        if source.is_partial():
            local = source._reduce_value(local, mesh, axis)
            sources[axis] = Replicate()
        if destination.is_partial():
            if _is_shard(source) and type(destination) is Partial and destination.reduce_op == 'sum':
                sparse[axis] = source
                destinations[axis] = source
            else:
                destinations[axis] = Replicate()
    before = DTensorSpec(mesh, tuple(sources), tensor_meta=current.tensor_meta)
    after = DTensorSpec(mesh, tuple(destinations), tensor_meta=target.tensor_meta)
    # Declared reductions remain ordinary graph equations. Backward page
    # demand selects only the needed fibers, including reduce-scatter outputs.
    local = _page_tensor(_exchange_domains(local, lambda rank: _rank_selection(after, rank), mesh,
                                          ownership=from_spec(before)))
    if sparse:
        from . import _stream
        from mesh_relation import Algebra
        from .host import assemble
        from .operand import TensorView, relation_view
        algebra, coordinate = Algebra(), tuple(mesh.get_coordinate())
        expanded = tuple(Replicate() if axis in sparse else placement
                         for axis, placement in enumerate(destinations))
        complete = DTensorSpec(mesh, expanded, tensor_meta=target.tensor_meta)
        source_order = algebra.selection_relation(_rank_selection(after, coordinate)[0])
        wanted, shape = _rank_selection(complete, coordinate)
        destination_order = algebra.selection_relation(wanted)
        positions = source_order.then(destination_order.inverse((math.prod(current.shape),)))
        out = torch.empty(shape, dtype=local.dtype, device=local.device)
        selected = relation_view(out, positions, local.shape)
        missing = destination_order.domain().binary(positions.image(positions.domain()), 'subtract')
        absent = _stream.TensorLayout.runs(missing.spans())
        zero = torch.zeros((), dtype=local.dtype, device=local.device)
        zero_view = TensorView(zero, _stream.TensorLayout((absent.count,), (0,)))
        flat = TensorView.of(out).reshape((math.prod(shape),))
        target_view = TensorView(out, absent.rebase(_stream.TensorLayout((math.prod(shape),), (1,)), flat.layout))
        local = assemble(((local, selected), (zero_view, target_view)))
    for axis, (source, destination) in enumerate(zip(current.placements, target.placements, strict=True)):
        if axis not in sparse and destination.is_partial() and not _same((source,), (destination,)):
            local = destination._partition_value(local, mesh, axis)
    return local


def _placements(mesh, placements, ndim, held=None):
    """placements with every dimension non-negative and each Shard on an attached mesh dimension that dimension's
    RaggedShard: the one `held` (a tensor's own placements) has there, else the attached one."""
    out, names = [], mesh.mesh_dim_names
    for i, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            p = _StridedShard(p.dim % ndim, split_factor=p.split_factor)
        elif isinstance(p, Shard):
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
    """One source operand type for every configured ownership relation."""
    return RaggedTensor(local, spec, requires_grad=local.requires_grad if requires_grad is None else requires_grad)


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



def distribute_tensor(tensor, device_mesh, placements, *, src_data_rank=0):
    """torch's distribute_tensor, a Shard on an attached mesh dimension its RaggedShard: each rank's capacity block,
    scattered from src_data_rank, or this rank's own exact block where it is None."""
    placements = _placements(device_mesh, placements, tensor.ndim)
    if not _ragged(placements) or isinstance(tensor, DTensor):
        made = _distribute_tensor(tensor, device_mesh, placements, src_data_rank=src_data_rank)
        return _register(_wrap(made._local_tensor, made._spec, made.requires_grad))
    if not tensor.is_leaf:
        raise RuntimeError('`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!')
    local = tensor.detach()
    if local.device.type != device_mesh.device_type and not local.is_meta:
        local = local.to(device_mesh.device_type)
    for i, p in enumerate(placements):
        if _is_shard(p):
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
        made = DTensor.from_local(local_tensor, device_mesh, placements, run_check=run_check, shape=shape, stride=stride,
                                 grad_placements=grad_placements)
        return _register(_wrap(made._local_tensor, made._spec, made.requires_grad))
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


class _OwnershipCast(torch.autograd.Function):
    """Capture configured forward/backward wire types as ordinary source ops."""
    @staticmethod
    def forward(ctx, value, forward_dtype, backward_dtype, source, spec):
        ctx.input_dtype, ctx.backward_dtype, ctx.source, ctx.spec = value.dtype, backward_dtype, source, spec
        return value.to(forward_dtype)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.to(ctx.backward_dtype)
        typ = TensorMeta(ctx.spec.shape, ctx.spec.tensor_meta.stride, ctx.backward_dtype)
        gradient = ctx.source.value(ctx.source.placed(gradient,
            DTensorSpec(ctx.spec.mesh, ctx.spec.placements, tensor_meta=typ)))
        return gradient.to(ctx.input_dtype), None, None, None, None


def redistribute(tensor, placements, *, async_op=False, forward_dtype=None, backward_dtype=None):
    """Declare changed ownership inside the complete function source."""
    placements = _placements(tensor.device_mesh, placements, tensor.ndim, tensor.placements)
    if _same(tensor.placements, placements) and forward_dtype is None and backward_dtype is None:
        return tensor
    source = _SOURCE_CAPTURE.get()
    if source is None:
        raise RuntimeError('redistribute is a Function source declaration; prepare the complete function before execution')
    value, meta = tensor, tensor._spec.tensor_meta
    if forward_dtype is not None or backward_dtype is not None:
        value = _OwnershipCast.apply(source.value(tensor), forward_dtype or tensor.dtype,
                                     backward_dtype or tensor.dtype, source, tensor._spec)
        meta = TensorMeta(meta.shape, meta.stride, forward_dtype or tensor.dtype)
    return source.placed(value, DTensorSpec(tensor.device_mesh, placements, tensor_meta=meta))


class RaggedTensor(DTensor):
    """A typed source operand, whose numerical uses belong to Function capture."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        source = _SOURCE_CAPTURE.get()
        if source is not None:
            arguments, keywords = pytree.tree_map(source.value, (args, kwargs or {}))
            return func(*arguments, **keywords)
        if func in (aten.detach.default, aten.alias.default):
            value = args[0]
            local = value._local_tensor.detach() if func is aten.detach.default else value._local_tensor
            made = _wrap(local, value._spec, local.requires_grad)
            return _register(made) if _registered(value) else made
        raise RuntimeError('distributed numerical operations are Function source; prepare the complete function before execution')

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
        return redistribute(self, placements, async_op=async_op,
                            forward_dtype=forward_dtype, backward_dtype=backward_dtype)

    def full_tensor(self, *, grad_placements=None):
        whole = self.redistribute(placements=[Replicate()] * self.device_mesh.ndim)
        return whole.to_local(grad_placements=grad_placements)


def _with(spec, placements):
    placements = tuple(placements)
    if _same(placements, spec.placements):
        return spec
    return DTensorSpec(spec.mesh, placements, tensor_meta=spec.tensor_meta)


def _rank_selection(spec, coordinate, selection=None):
    """One rank's ordered logical pages, retaining all preceding axis cuts."""
    declaration = {'shape': list(spec.shape), 'steps': []} if selection is None else {
        'shape': selection['shape'], 'steps': list(selection['steps'])}
    shape = list(spec.shape)
    for i, p in enumerate(spec.placements):
        if _is_shard(p):
            width = shape[p.dim]
            if isinstance(p, _StridedShard):
                block = -(-width // p.split_factor)
                runs = []
                for start in range(0, width, max(1, block)):
                    count, first = Shard(p.dim)._local_shard_size_and_offset(min(block, width - start),
                                                                            spec.mesh.size(i), coordinate[i])
                    if count:
                        runs.append([start + first, count, 0, 1])
            else:
                count, first = p._local_shard_size_and_offset(width, spec.mesh.size(i), coordinate[i])
                runs = [[first, count, 0, 1]] if count else []
            declaration['steps'].append(['select', p.dim, runs])
            shape[p.dim] = sum(run[1] for run in runs)
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
                  tuple(i for i, p in enumerate(x.placements) if _is_shard(p)) if isinstance(x, DTensor) else ())
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
    from .ownership import (from_spec, result_owners, prepare_expression, compile_assignments, contribution_domains,
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
    from .compilation import implementation_assignment
    policy = configuration.get('result_ownership', implementation_assignment()['result_ownership'])
    results = next(node.args[0] for node in nodes if node.op == 'output')
    for node in dict.fromkeys(value for value in pytree.tree_leaves(results) if isinstance(value, torch.fx.Node)):
        if node not in records:
            records[node] = pytree.tree_map(lambda value: result_owners(
                TensorMeta(value.shape, value.stride(), value.dtype), mesh, policy)
                if isinstance(value, torch.Tensor) else None, node.meta['val'])
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
        # All ordinary nodes keep the captured global equation. Page demand
        # and the configured owners assign its numerical work; no operator's
        # local sharding rule may substitute a replicated implementation.
        args, kwargs = pytree.tree_map(complete_inputs, (node.args, node.kwargs))
        if node.target is operator.getitem and node.args[0] in records and not leaf(records[node.args[0]]):
            source = records[node.args[0]]
            selected = node.args[1]
            values_ = dict(source) if isinstance(source, dict) else list(source)
            values_[selected] = record_map(complete_value, source[selected])
            args = (tuple(values_) if isinstance(source, tuple) else values_, *args[1:])
        declare_uses(node, 'args', node.args, args)
        declare_uses(node, 'kwargs', node.kwargs, kwargs)
        for path, target in owners(node).items():
            if target.get('reductions'):
                before = complete_value(target)
                crossings.append(dict(source=(node, path), provided=before, required=target,
                    consumer=node, side='results', path=path, mesh=mesh, order=len(crossings)))
                materialized.add(node)
                cuts.add((node, 'after'))

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
                and any(_is_shard(p) and p.dim % x.ndim == x.ndim - 1 for p in input_layouts)):
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
