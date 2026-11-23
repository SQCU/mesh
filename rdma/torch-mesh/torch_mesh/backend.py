"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h over the bridges' page tables), issued by one path, _stream.mm's `issue`: a list of
calls, each a kind, its tensors, counts, operator and root.  Every call is complete when it returns (the library's calls
block, on host memory): an async op's Work is complete when it is made.  A group on MPS tensors drains torch's stream
first, and its tensors are read where the host reaches them (torch_mesh.empty's) or staged through such a tensor by
torch's copies; CPU tensors are read where they lie.  A non-contiguous tensor fails as a value (_stream.mm Group):
nothing is copied to make it one.

The collectives: all-reduce, reduce, broadcast, all-gather, reduce-scatter, all-to-all, gather and scatter, each of a
single tensor or a list.  A list of segments, one a rank, is read and written in place where the segments lie end to end
in one contiguous tensor (a split's pieces), else staged through one flat tensor (packed).  An all-gather's and a
reduce-scatter's segments may be of any sizes (MPI's v-collectives in torch's spelling): the library takes equal
segments as ncclAllGather and ncclReduceScatter, others as a broadcast or a reduce a rank in one group.  An all-to-all's,
a gather's and a scatter's are of one size, as ncclAlltoAll, ncclGather and ncclScatter move them (others raise
NotImplementedError).  The partials merge (merge_partials) is an all-gather of every member's partials and their merge
by log-sum-exp in rank order on each member.

A group's communicator is NCCL's bootstrap's (ncclCommInitRank, rank 0's ncclGetUniqueId passed through the group's
store, as ProcessGroupNCCL does); its pg_options are not read.  shrink, grow and revoke are ncclCommShrink,
ncclCommGrow and ncclCommRevoke."""
import math

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp, _create_work_from_future
from torch.futures import Future

from . import _stream, empty, evidence  # _stream's kernels for torch's functional collectives on MPS register as it loads

import mesh  # noqa: E402  (rdma/mesh.py, on the path the mesh's install gives the environment: its _mesh_c)

ffi, LIB, check = mesh.ffi, mesh.lib, mesh.check
OPS = ((ReduceOp.SUM, LIB.ncclSum), (ReduceOp.PRODUCT, LIB.ncclProd), (ReduceOp.MAX, LIB.ncclMax), (ReduceOp.MIN, LIB.ncclMin),
       (ReduceOp.AVG, LIB.ncclAvg))
# a call's kind (_stream.mm Kind)
ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, GATHER, SCATTER = range(8)
# the partials merge's row with no key seen: [the least float32, 0, 0, ...]
PARTIAL_NONE = -3.4028234663852886e38


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def span(tensors):
    """The one contiguous tensor `tensors` lie in end to end, in their order (the pieces of a split: out.split(sizes)),
    as a flat view of it; None where they do not."""
    first, at = tensors[0], tensors[0].storage_offset()
    for t in tensors:
        if (t.dtype != first.dtype or t.device != first.device or not t.is_contiguous() or t.storage_offset() != at
                or t.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()):
            return None
        at += t.numel()
    return first.as_strided((at - first.storage_offset(),), (1,))


def packed(tensors, read):
    """The one tensor `tensors` lie in end to end (span), else a flat one they are copied into where it is `read`, with
    the copy back to them where it is written: (it, that copy or None); (None, None) where there are none."""
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None, None
    whole = span(tensors)
    if whole is not None:
        return whole, None
    whole = empty(sum(t.numel() for t in tensors), dtype=tensors[0].dtype, device=tensors[0].device)
    if read:
        return torch.cat([t.detach().reshape(-1) for t in tensors], out=whole), None
    return whole, lambda: [t.copy_(p.view(t.shape)) for t, p in zip(tensors, whole.split([t.numel() for t in tensors]))]


def done(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


def _flat(value):
    return [value] if isinstance(value, torch.Tensor) else [t for v in value for t in _flat(v)]


def merged(stack):
    """Partials stacked in rank order (rank x rows x (2 + dim), rows of [m, l, O]) merged by their log-sum-exp: a row
    with l > 0 rescaled to the greatest m and summed, one with none counting for none, a row none saw [PARTIAL_NONE, 0,
    0, ...]."""
    m, l = stack[..., 0], stack[..., 1]
    seen = l > 0
    top = torch.where(seen, m, -math.inf).amax(0)
    any_seen = seen.any(0)
    scale = torch.where(seen, torch.exp(m - torch.where(any_seen, top, 0)), 0)
    rest = (stack[..., 1:] * scale[..., None]).sum(0)
    return torch.cat([torch.where(any_seen, top, PARTIAL_NONE)[..., None], rest], -1)


class ProcessGroupMesh(dist.ProcessGroup):
    supports_coalescing = True
    supports_splitting = False

    def _get_backend(self, device):
        """The group itself, its one backend on every device it runs (torch asks a group's backend by device:
        init_process_group's device_id reads it)."""
        return self

    def __init__(self, rank, size, store):
        super().__init__(rank, size)
        made, unique = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if rank == 0:
            check(LIB.ncclGetUniqueId(unique))
            store.set('mesh/unique', bytes(ffi.buffer(unique)))
        else:
            ffi.memmove(unique, store.get('mesh/unique'), ffi.sizeof('ncclUniqueId'))
        check(LIB.ncclCommInitRank(made, size, unique[0], rank))
        self._held(rank, size, made[0])

    def _held(self, rank, size, comm):
        self._rank, self._size, self._pending = rank, size, None
        self.comm = comm
        self.handle = int(ffi.cast('uintptr_t', self.comm))
        self._met = torch.zeros(1, dtype=torch.int32)
        _stream.attach(self, self.handle)

    @classmethod
    def _of(cls, comm):
        """A group over a communicator libnccl-mesh made (a shrink's or a grow's), with no bootstrap of its own."""
        count, rank = ffi.new('int *'), ffi.new('int *')
        check(LIB.ncclCommCount(comm, count))
        check(LIB.ncclCommUserRank(comm, rank))
        made = cls.__new__(cls)
        dist.ProcessGroup.__init__(made, rank[0], count[0])
        made._held(rank[0], count[0], comm)
        return made

    def shrink(self, exclude, abort=False):
        """This group without the ranks `exclude`, the kept ranks in their order (ncclCommShrink; `abort` revokes this
        group first); None on a rank among those excluded."""
        made, ranks = ffi.new('ncclComm_t *'), ffi.new('int[]', list(exclude) or [0])
        check(LIB.ncclCommShrink(self.comm, ranks, len(exclude), made, ffi.NULL, 1 if abort else 0))
        return ProcessGroupMesh._of(made[0]) if made[0] != ffi.NULL else None

    def grow(self, size, unique=None, rank=-1):
        """This group's ranks and `size` - its size more (ncclCommGrow), met at the id this group's root gives
        (ncclCommGetUniqueId, `unique` None on its ranks); a joining process passes the root's id and its `rank` on a
        None group: ProcessGroupMesh.grow(None, size, unique, rank)."""
        made, held = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if self is not None and self._rank == 0:
            check(LIB.ncclCommGetUniqueId(self.comm, held))
        elif unique is not None:
            ffi.memmove(held, unique, ffi.sizeof('ncclUniqueId'))
        check(LIB.ncclCommGrow(ffi.NULL if self is None else self.comm, size, held, rank, made, ffi.NULL))
        return ProcessGroupMesh._of(made[0]), bytes(ffi.buffer(held))

    def revoke(self):
        """This group's calls under way end as a value and every later call on it is refused (ncclCommRevoke)."""
        check(LIB.ncclCommRevoke(self.comm, 0))

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    @property
    def group_name(self):
        """The name torch registered this group under (a creator's ProcessGroup is never given its name)."""
        return dist.distributed_c10d._world.pg_names[self]

    pg_name = group_name

    def _issue(self, calls, result, backs=()):
        """One NCCL group of `calls` (_stream.mm issue), complete where it returns, `result` what its Work holds;
        `backs` then run.  Between start_coalescing and end_coalescing the calls join the coalesced group instead.  The
        evidence's marks of a host group are taken here (an MPS group's on torch's stream, by _stream.mm)."""
        if self._pending is not None:
            self._pending.append((calls, result, backs))
            return done(result)
        host = not any(t is not None and t.device.type == 'mps' for call in calls for t in (*call[1], *call[2]))
        if host:
            evidence.mark(self, True)
        _stream.issue(calls, self.handle)
        for back in backs:
            back()
        if host:
            evidence.mark(self, False)
        return done(result)

    def start_coalescing(self, device):
        self._pending = []

    def end_coalescing(self, device):
        pending, self._pending = self._pending, None
        return self._issue([c for calls, _, _ in pending for c in calls], [r for _, r, _ in pending],
                           backs=[b for _, _, backs in pending for b in backs])

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        return self._issue([(ALLREDUCE, [t], [], [], op(opts.reduceOp), 0) for t in tensors], tensors)

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        return self._issue([(REDUCE, [t], [], [], op(opts.reduceOp), opts.rootRank) for t in tensors], tensors)

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        return self._issue([(BROADCAST, [t], [], [], 0, opts.rootRank) for t in tensors], tensors)

    def barrier(self, opts=None):
        """Every rank met, torch's stream drained first so the work before it has run: a one-element all-reduce,
        complete on a rank only once every rank has called it."""
        torch.mps.synchronize()
        return self._issue([(ALLREDUCE, [self._met], [], [], LIB.ncclSum, 0)], [])

    def all_gather_single_coalesced(self, outputs, inputs, opts=None):
        return self._issue([(ALLGATHER, [i], [o], [i.numel()] * self._size, 0, 0) for o, i in zip(outputs, inputs)], outputs)

    def all_gather_single(self, output, input, opts=None):
        return self.all_gather_single_coalesced([output], [input], opts)

    allgather_into_tensor_coalesced = all_gather_single_coalesced

    def allgather(self, outputs, inputs, opts=None):
        """Each input gathered into its list of outputs, of any sizes (MPI_Allgatherv in torch's spelling): into the
        outputs in place where they lie end to end in one contiguous tensor (a split's pieces: torch_mesh.partition's
        allgatherv), else into one flat tensor copied out to them (packed)."""
        wholes = [packed(out, False) for out in outputs]
        return self._issue([(ALLGATHER, [i], [whole], [t.numel() for t in out], 0, 0) for (whole, _), out, i in zip(wholes, outputs, inputs)],
                           outputs, [back for _, back in wholes if back])

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        return self._issue([(REDUCE_SCATTER, [i], [o], [o.numel()] * self._size, op(opts.reduceOp), 0) for o, i in zip(outputs, inputs)],
                           outputs)

    def reduce_scatter_single(self, output, input, opts=dist.ReduceScatterOptions()):
        return self.reduce_scatter_single_coalesced([output], [input], opts)

    reduce_scatter_tensor_coalesced = reduce_scatter_single_coalesced

    def reduce_scatter(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        """Each list of inputs reduced, rank r's block left at rank r, of any sizes (MPI_Reduce_scatter in torch's
        spelling): read in place where the inputs lie end to end in one contiguous tensor (a split's pieces:
        torch_mesh.partition's reduce_scatterv), else from one flat tensor they are copied into (packed)."""
        return self._issue([(REDUCE_SCATTER, [packed(parts, True)[0]], [o], [x.numel() for x in parts], op(opts.reduceOp), 0)
                            for o, parts in zip(outputs, inputs)], outputs)

    def _equal(self, tensors, what):
        """The one size of `tensors`' segments; NotImplementedError where they differ (the library's `what` moves equal
        segments)."""
        sizes = {t.numel() for t in tensors}
        if len(sizes) != 1:
            raise NotImplementedError(f'the mesh backend\'s {what} moves segments of one size (nccl.h), not {sorted(sizes)}')
        return sizes.pop()

    def _alltoall(self, sends, receives, result):
        """MPI_Alltoall (ncclAlltoAll): sends[q] to rank q and receives[q] from it, every segment of one size."""
        count = self._equal([*sends, *receives], 'all-to-all')
        (send, _), (receive, back) = packed(sends, True), packed(receives, False)
        return self._issue([(ALLTOALL, [send], [receive], [count] * self._size, 0, 0)], result, [back] if back else ())

    def alltoall(self, outputs, inputs, opts=None):
        return self._alltoall(inputs, outputs, outputs)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        def pieces(t, sizes):
            return list(t.split(list(sizes) or [t.shape[0] // self._size] * self._size))
        return self._alltoall(pieces(input, input_split_sizes), pieces(output, output_split_sizes), output)

    def gather(self, outputs, inputs, opts=dist.GatherOptions()):
        """Every rank's input to the root's list of outputs, each of the input's size (ncclGather)."""
        root, receive, back = opts.rootRank, None, None
        if self._rank == root:
            self._equal([*outputs[0], inputs[0]], 'gather')
            receive, back = packed(outputs[0], False)
        return self._issue([(GATHER, [inputs[0]], [receive], [], 0, root)], outputs, [back] if back else ())

    def scatter(self, outputs, inputs, opts=dist.ScatterOptions()):
        """The root's list of inputs, one to each rank, each of the output's size (ncclScatter)."""
        root, send = opts.rootRank, None
        if self._rank == root:
            self._equal([*inputs[0], outputs[0]], 'scatter')
            send = packed(inputs[0], True)[0]
        return self._issue([(SCATTER, [send], [outputs[0]], [], 0, root)], outputs)

    def merge_partials(self, out, partials, rows=None, root=-1):
        """Split-KV attention partials merged by their log-sum-exp: `partials` this member's float32 rows of [m, l, O]
        (partial), the same shape on every member, all-gathered and merged in rank order (merged) into `out` (it may be
        `partials`) on every member (root -1) or at `root` alone; where `rows` gives each rank's rows of them in rank
        order, rank r's rows[r] merged into its `out`.  It is not coalesced: its merge reads what the gather wrote."""
        if self._pending is not None:
            raise ValueError('the mesh backend merges partials outside a coalesced group')
        partials = partials.contiguous()
        gathered = empty((self._size, *partials.shape), dtype=partials.dtype, device=partials.device.type)
        self._issue([(ALLGATHER, [partials], [gathered], [partials.numel()] * self._size, 0, 0)], [gathered])
        if rows is not None:
            gathered = gathered.narrow(1, sum(rows[:self._rank]), rows[self._rank])
        if root < 0 or root == self._rank:
            out.copy_(merged(gathered))
        return done([out])


def partial(out, lse):
    """An attention's output over some keys (shape x dim) and each row's logsumexp (shape) as the library's partials,
    float32 rows (shape x (2 + dim)) of [m, l, O]: m the logsumexp, l 1, O the output; a row that saw no key (its
    logsumexp -inf) l 0, so it counts for none."""
    seen = torch.isfinite(lse)[..., None]
    return torch.cat([torch.where(seen, lse[..., None], 0), seen.float(), torch.where(seen, out, 0).float()], -1)


def attention(merged):
    """Merged partials (rows of [M, L, O]) as the attention over every member's keys, O / L (0 where L is 0), and each
    row's logsumexp, M + log L (-inf where L is 0)."""
    m, l, o = merged[..., 0], merged[..., 1], merged[..., 2:]
    seen = l > 0
    l = torch.where(seen, l, 1)
    return torch.where(seen[..., None], o / l[..., None], 0), torch.where(seen, m + l.log(), -math.inf)


def _mesh(group_name, what):
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if not isinstance(group, ProcessGroupMesh):
        raise TypeError(f'{what} runs on a group of the mesh backend, not {group_name}\'s')
    return group


@torch.library.custom_op('torch_mesh::merge_partials', mutates_args=())
def merge_partials(partials: torch.Tensor, root: int, group_name: str) -> torch.Tensor:
    """The partials merge on the mesh group named `group_name` as a torch operator (torch_mesh.merge_partials):
    float32 rows of [m, l, O] (partial) merged by their log-sum-exp, on every member (`root` -1) or at `root`."""
    out = empty(partials.shape, dtype=partials.dtype, device=partials.device.type)
    _mesh(group_name, 'merge_partials').merge_partials(out, partials, root=root)
    return out


@merge_partials.register_fake
def _(partials, root, group_name):
    return torch.empty_like(partials)


@torch.library.custom_op('torch_mesh::all_reduce', mutates_args=())
def all_reduce(input: torch.Tensor, reduce_op: str, group_name: str) -> torch.Tensor:
    """torch's functional all_reduce as an op a compiled graph calls (torch_mesh.compiler), so Inductor calls it rather
    than lowering it to a copy into a buffer of its own and an all_reduce_ there: its result a host tensor of the
    functional kernel's (_stream.mm functional), complete when it returns."""
    return torch.ops._c10d_functional.all_reduce.default(input, reduce_op, group_name)


@all_reduce.register_fake
def _(input, reduce_op, group_name):
    return torch.empty_like(input)


@torch.library.register_fake('torch_mesh::linears')
def _(x, w, bias, act, x1):
    return [a.new_empty(a.shape[0], b.shape[0]) for a, b in zip(x, w)]


@torch.library.register_fake('torch_mesh::linear')
def _(x, w, bias=None, act=0, x1=None):
    return x.new_empty(x.shape[0], w.shape[0])


def create(opts, options):
    """The group (torch's extended creator: `opts` its store, rank and size; `options`, the caller's pg_options, not
    read)."""
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'  # init_process_group() without a backend: MPS's default is mesh
