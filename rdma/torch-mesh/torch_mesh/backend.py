"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h on the bridges' prepared transfers), issued by one path, _stream.mm's `issue`: a list
of calls, each a kind, its tensors, counts, operator and root.  MPS tensors take the library's Metal path: the group is
encoded on torch's MPS stream, each tensor passed as its storage's MTLBuffer and byte offset (ncclMeshBuffer), so the
work is ordered with torch's own and the call returns at once (an async op issued and not completed: a Work whose wait
completes it on torch's stream).  CPU tensors take the host path, complete where the call returns.  A non-contiguous
tensor fails as a value (_stream.mm Group): nothing is copied to make it one.

One form a collective, the library's V call where it has one (the library takes equal segments as its equal
program): an all-gather and a reduce-scatter are ncclMeshAllGatherV and ncclMeshReduceScatterV; all_to_all,
all_to_all_single, gather and scatter are ncclMeshAlltoAllV, MPI_Alltoallv with its counts on the host (a gather and a
scatter of any sizes the rooted V, MPI's Gatherv and Scatterv: every count 0 but toward or from the root, as NCCL's
gather and scatter are its forms); a list that is not the pieces of one tensor is staged through one flat tensor
(packed).  send and recv are the library's point to point.  The partials merge (merge_partials) is an all-reduce,
reduce or V reduce-scatter by the library's partials operator.

A group's communicator is NCCL's bootstrap's (ncclCommInitRank, rank 0's ncclGetUniqueId passed through the group's
store, as ProcessGroupNCCL does), its sessions each on the map the bridges keep when it opens; its pg_options are not
read.
The silent-link bound is the library's (nccl.h)."""
import collections
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
ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, SEND, RECV, WRITE = range(9)
# a declared call (ProcessGroupMesh.plan): its kind, its elements, its dtype, its root or peer, its operator and, for an
# all-gather or a reduce-scatter, each rank's elements (an all-to-all's to each rank, then from each)
Call = collections.namedtuple('Call', 'kind count dtype root op counts destination fused', defaults=(0, ReduceOp.SUM, None, 0, False))
WHAT = {ALLREDUCE: LIB.MESH_ALLREDUCE, REDUCE: LIB.MESH_REDUCE, BROADCAST: LIB.MESH_BROADCAST, ALLGATHER: LIB.MESH_ALLGATHER,
        REDUCE_SCATTER: LIB.MESH_REDUCE_SCATTER, ALLTOALL: LIB.ncclMeshCallAlltoAllV, SEND: LIB.ncclMeshCallSend,
        RECV: LIB.ncclMeshCallRecv, WRITE: LIB.ncclMeshCallWrite}
TYPES = {torch.int8: LIB.ncclInt8, torch.uint8: LIB.ncclUint8, torch.bool: LIB.ncclUint8, torch.int32: LIB.ncclInt32,
         torch.uint32: LIB.ncclUint32, torch.int64: LIB.ncclInt64, torch.uint64: LIB.ncclUint64, torch.float16: LIB.ncclFloat16,
         torch.float32: LIB.ncclFloat32, torch.float64: LIB.ncclFloat64, torch.bfloat16: LIB.ncclBfloat16,
         torch.float8_e4m3fn: LIB.ncclFloat8e4m3, torch.float8_e5m2: LIB.ncclFloat8e5m2}


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def typed(dtype, reduces):
    """A dtype as _stream.mm's Group passes a tensor of it (moving, reducing): its NCCL type and that type's elements an
    element (a complex reduction's two; a type NCCL lacks moved as its bytes)."""
    if reduces and dtype.is_complex:
        return TYPES[dtype.to_real()], 2
    return (TYPES[dtype], 1) if dtype in TYPES else (LIB.ncclUint8, torch.empty(0, dtype=dtype).element_size())


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


def asynchronous(opts):
    """An async op (torch sets the option's asyncOp from async_op): issued now, completed where it is waited."""
    return bool(getattr(opts, 'asyncOp', False))


def done(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


def _flat(value):
    return [value] if isinstance(value, torch.Tensor) else [t for v in value for t in _flat(v)]


class ProcessGroupMesh(dist.ProcessGroup):
    supports_coalescing = True
    supports_splitting = False

    def _get_backend(self, device):
        """The group itself, its one backend on every device it runs (torch asks a group's backend by device:
        batch_isend_irecv coalesces through it, init_process_group's device_id reads it)."""
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
        self._rank, self._size, self._pending, self._operators = rank, size, None, {}
        self.comm = comm
        self.handle = int(ffi.cast('uintptr_t', self.comm))
        _stream.attach(self, self.handle)

    @property
    def page(self):
        """The bytes of a page of this communicator's region, on which each segment of a V collective starts (nccl.h
        ncclMeshCommPage)."""
        held = ffi.new('size_t *')
        check(LIB.ncclMeshCommPage(self.comm, held))
        return held[0]

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
        group's work in flight first); None on a rank among those excluded."""
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
        """This group's work in flight is revoked and none is issued after it (ncclCommRevoke): its bridge sends what
        it published and nothing else, and its own GPU waits end at their landings or their bounds."""
        check(LIB.ncclCommRevoke(self.comm, 0))

    def shutdown(self):
        """torch's end of this group (destroy_process_group): the fused crossings' forms on its communicator released,
        their leases back to the region once the stream has passed them (_stream.mm fused_release)."""
        _stream.fused_release(self.handle)
        super().shutdown()

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    @property
    def group_name(self):
        """The name torch registered this group under (a creator's ProcessGroup is never given its name)."""
        return dist.distributed_c10d._world.pg_names[self]

    pg_name = group_name

    def _issue(self, calls, result, split=False, backs=()):
        """One NCCL group of `calls` (_stream.mm issue), `result` what its Work holds: issued and not completed where
        `split` (an async op) and nothing is to be written back after it, else complete; `backs` then run.  Between
        start_coalescing and end_coalescing the calls join the coalesced group instead.  The evidence's marks of a
        host-path group are taken here (a Metal-path group's on torch's stream, by _stream.mm)."""
        if self._pending is not None:
            self._pending.append((calls, result, backs))
            return done(result)
        host = not any(t is not None and t.device.type == 'mps' for call in calls for t in (*call[1], *call[2]))
        if host:
            evidence.mark(self, True)
        work = _stream.issue(calls, self.handle, bool(split) and not backs, _flat(result))
        for back in backs:
            back()
        if host:
            evidence.mark(self, False)
        if work is None:
            return done(result)
        return work

    def plan(self, groups):
        """This group's collectives declared once, in their order (nccl.h ncclMeshCommPlan): `groups` the groups its
        calls issue, each a list of Calls (one torch collective is one group), or (that list, repeat) issued repeat
        times in a row, the order repeated from this group's next collective on.  Each is compiled now, so none is
        planned where it is issued, and none is compared with its declaration.  [] drops the plan.  Returns each
        declared group's (send entries, landing entries) on this rank's lists."""
        rows = [(g, *(group if isinstance(group, tuple) else (group, 1))) for g, group in enumerate(groups)]
        rows = [(g, repeat, call) for g, calls, repeat in rows for call in calls]
        made, kept = [], []
        for g, repeat, c in rows:
            reduces = c.kind in (ALLREDUCE, REDUCE, REDUCE_SCATTER)
            nccl, per = typed(c.dtype, reduces)
            code = (c.op if isinstance(c.op, int) else op(c.op)) if reduces else LIB.ncclSum
            if c.dtype == torch.bool and reduces:
                code = LIB.ncclMax if code == LIB.ncclSum else LIB.ncclMin if code == LIB.ncclProd else code
            kept.append(ffi.new('size_t[]', [n * per for n in c.counts]) if c.counts is not None else ffi.NULL)
            made.append({'what': WHAT[c.kind], 'peer': c.root, 'count': c.count * per, 'counts': kept[-1], 'type': nccl, 'op': code,
                         'group': g, 'repeat': repeat, 'destination': c.destination, 'fused': int(c.fused)})
        view = ffi.new('ncclMeshPlan **')
        check(LIB.ncclMeshCommPlan(self.comm, ffi.new('ncclMeshCall[]', made or 1), len(made), view))
        return [] if view[0] == ffi.NULL else [tuple(view[0].entries[g]) for g in range(view[0].groups)]

    def plan_active(self, active):
        """Suspend the function declaration for feedback; resume at its first group."""
        check(LIB.ncclMeshCommPlanActive(self.comm, bool(active)))

    def write(self, source_buffer, landing_buffer, source, destination, async_op=False):
        """design/whole-function.md#one-way-writes"""
        return self._issue([(WRITE, [source_buffer], [landing_buffer], [destination], 0, source)], landing_buffer, async_op)

    def _alltoall(self, sends, receives, result, opts):
        """MPI_Alltoallv (ncclMeshAlltoAllV): sends[q] to rank q and receives[q] from it, None moving none."""
        (send, _), (receive, back) = packed(sends, True), packed(receives, False)
        counts = [0 if t is None else t.numel() for t in (*sends, *receives)]
        return self._issue([(ALLTOALL, [send], [receive], counts, 0, 0)], result, asynchronous(opts), [back] if back else ())

    def _rooted(self, root, tensor):
        """A rooted all-to-all's list: `tensor` to or from `root` alone."""
        return [tensor if q == root else None for q in range(self._size)]

    def start_coalescing(self, device):
        self._pending = []

    def end_coalescing(self, device):
        pending, self._pending = self._pending, None
        return self._issue([c for calls, _, _ in pending for c in calls], [r for _, r, _ in pending],
                           backs=[b for _, _, backs in pending for b in backs])

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        return self._issue([(ALLREDUCE, [t], [], [], op(opts.reduceOp), 0) for t in tensors], tensors, asynchronous(opts))

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        return self._issue([(REDUCE, [t], [], [], op(opts.reduceOp), opts.rootRank) for t in tensors], tensors, asynchronous(opts))

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        return self._issue([(BROADCAST, [t], [], [], 0, opts.rootRank) for t in tensors], tensors, asynchronous(opts))

    def barrier(self, opts=None):
        return self.allreduce([empty(1, dtype=torch.float32, device='cpu').fill_(1)])

    def all_gather_single_coalesced(self, outputs, inputs, opts=None):
        return self._issue([(ALLGATHER, [i], [o], [i.numel()] * self._size, 0, 0) for o, i in zip(outputs, inputs)], outputs,
                           asynchronous(opts))

    def all_gather_single(self, output, input, opts=None):
        return self.all_gather_single_coalesced([output], [input], opts)

    allgather_into_tensor_coalesced = all_gather_single_coalesced

    def allgather(self, outputs, inputs, opts=None):
        """Each input gathered into its list of outputs, of any sizes (MPI_Allgatherv in torch's spelling): into the
        outputs in place where they lie end to end in one contiguous tensor (a split's pieces: torch_mesh.partition's
        allgatherv), else into one flat tensor copied out to them (packed)."""
        wholes = [packed(out, False) for out in outputs]
        return self._issue([(ALLGATHER, [i], [whole], [t.numel() for t in out], 0, 0) for (whole, _), out, i in zip(wholes, outputs, inputs)],
                           outputs, asynchronous(opts), [back for _, back in wholes if back])

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        return self._issue([(REDUCE_SCATTER, [i], [o], [o.numel()] * self._size, op(opts.reduceOp), 0) for o, i in zip(outputs, inputs)],
                           outputs, asynchronous(opts))

    def reduce_scatter_single(self, output, input, opts=dist.ReduceScatterOptions()):
        return self.reduce_scatter_single_coalesced([output], [input], opts)

    reduce_scatter_tensor_coalesced = reduce_scatter_single_coalesced

    def reduce_scatter(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        """Each list of inputs reduced, rank r's block left at rank r, of any sizes (MPI_Reduce_scatter in torch's
        spelling): read in place where the inputs lie end to end in one contiguous tensor (a split's pieces:
        torch_mesh.partition's reduce_scatterv), else from one flat tensor they are copied into (packed)."""
        return self._issue([(REDUCE_SCATTER, [packed(parts, True)[0]], [o], [x.numel() for x in parts], op(opts.reduceOp), 0)
                            for o, parts in zip(outputs, inputs)], outputs, asynchronous(opts))

    def alltoall(self, outputs, inputs, opts=None):
        return self._alltoall(inputs, outputs, outputs, opts)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        def pieces(t, sizes):
            return list(t.split(list(sizes) or [t.shape[0] // self._size] * self._size))
        return self._alltoall(pieces(input, input_split_sizes), pieces(output, output_split_sizes), output, opts)

    def gather(self, outputs, inputs, opts=dist.GatherOptions()):
        """Every rank's input to the root's list of outputs, any sizes (MPI_Gatherv): the rooted all-to-all."""
        receives = list(outputs[0]) if self._rank == opts.rootRank else [None] * self._size
        return self._alltoall(self._rooted(opts.rootRank, inputs[0]), receives, outputs, opts)

    def scatter(self, outputs, inputs, opts=dist.ScatterOptions()):
        """The root's list of inputs, one to each rank, any sizes (MPI_Scatterv): the rooted all-to-all."""
        sends = list(inputs[0]) if self._rank == opts.rootRank else [None] * self._size
        return self._alltoall(sends, self._rooted(opts.rootRank, outputs[0]), outputs, opts)

    def send(self, tensors, dst, tag):
        return self._issue([(SEND, [t], [], [], 0, dst) for t in tensors], tensors, True)

    def recv(self, tensors, src, tag):
        return self._issue([(RECV, [], [t], [], 0, src) for t in tensors], tensors, True)

    def partials_operator(self, width):
        """Realize a partial-row reduction operator before declaring a function."""
        if width not in self._operators:
            made = ffi.new('ncclRedOp_t *')
            check(LIB.ncclMeshRedOpCreatePartials(made, width - 2, LIB.ncclFloat32, self.comm))
            self._operators[width] = made[0]
        return self._operators[width]

    def merge_partials(self, out, partials, rows=None, root=-1, async_op=False):
        """Split-KV attention partials merged by their log-sum-exp, by the library's partials operator (nccl.h
        ncclMeshRedOpCreatePartials, one an operand width, made at its first use): `partials` this member's float32 rows
        of [m, l, O] (partial), merged into `out` (it may be `partials`) on every member (an all-reduce), at `root` (a
        reduce), or, where `rows` gives each rank's rows of them in rank order, rank r's rows[r] merged into its `out`
        (ncclMeshReduceScatterV).  One call of the one path, coalesced like any other."""
        width = partials.shape[-1]
        merge = self.partials_operator(width)
        call = ((REDUCE_SCATTER, [partials], [out], [r * width for r in rows], merge, 0) if rows is not None
                else (ALLREDUCE, [partials], [out], [], merge, 0) if root < 0 else (REDUCE, [partials], [out], [], merge, root))
        return self._issue([call], [out], async_op)


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
    out = empty(partials.shape, dtype=partials.dtype, device=partials.device)
    _mesh(group_name, 'merge_partials').merge_partials(out, partials, root=root)
    return out


@merge_partials.register_fake
def _(partials, root, group_name):
    return torch.empty_like(partials)


@torch.library.custom_op('torch_mesh::all_reduce', mutates_args=())
def all_reduce(input: torch.Tensor, reduce_op: str, group_name: str, repeats: int = 1) -> torch.Tensor:
    """torch's functional all_reduce as an op a compiled graph calls (torch_mesh.compiler).  A sum of MPS tensors on a
    mesh group is a fused crossing (partition._fused, as allgatherv's): this rank's part published from a held lease by
    a kernel of partition's own, every peer's awaited and all of them added in rank order by another, the host issuing
    the crossing (once for `repeats` of them: the graph's, torch_mesh.compiler) and two encodes; any other its
    functional kernel's, its work waited where the graph waits on it (wait_tensor)."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if reduce_op == 'sum' and isinstance(group, ProcessGroupMesh) and input.device.type == 'mps':
        from .partition import _fused
        x = input.contiguous().view(-1)
        return _fused(x, 0, [x.numel()] * group.size(), group, True, None, repeats).view(input.shape)
    return torch.ops._c10d_functional.all_reduce.default(input, reduce_op, group_name)


@all_reduce.register_fake
def _(input, reduce_op, group_name, repeats=1):
    return torch.empty_like(input)


@torch.library.custom_op('torch_mesh::linear_sum', mutates_args=())
def linear_sum(x: torch.Tensor, w: torch.Tensor, group_name: str, repeats: int = 1) -> torch.Tensor:
    """x w^T (torch_mesh::linear's product, x [R, K], w [N, K]) summed over the mesh group named `group_name` as one
    fused crossing (partition._fused), issued once for `repeats` of them (torch_mesh.compiler): linear.metal's
    linear_publish computes this rank's product into its segment and publishes it through the collective's steps,
    sum_land adds every rank's in rank order (torch_mesh.compiler)."""
    from .partition import _fused
    group = dist.distributed_c10d._resolve_process_group(group_name)
    x = x.contiguous()
    return _fused(x, 0, [x.shape[0]] * group.size(), group, True, w, repeats)


@linear_sum.register_fake
def _(x, w, group_name, repeats=1):
    return x.new_empty(x.shape[0], w.shape[0])


@torch.library.register_fake('torch_mesh::linears')
def _(x, w, bias, act, x1):
    return [a.new_empty(a.shape[0], b.shape[0]) for a, b in zip(x, w)]


@torch.library.register_fake('torch_mesh::linear')
def _(x, w, bias=None, act=0, x1=None):
    return x.new_empty(x.shape[0], w.shape[0])


def create(opts, options):
    """The group (torch's extended creator: `opts` its store, rank and size; `options`, the caller's pg_options, not
    read: a group's map is its bridges')."""
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'  # init_process_group() without a backend: MPS's default is mesh
