"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h on the bridges' prepared transfers), called through _mesh_c (the headers' own
declarations, compiled against them: ../../mesh_c_build.py).  MPS tensors take the library's Metal path: the group
is encoded on torch's MPS stream (_stream.mm) and committed, each tensor passed as its storage's MTLBuffer and byte
offset (ncclMeshBuffer), so the work is ordered with torch's own and the call returns at once.  CPU tensors take the
host path, a non-contiguous one staged through a contiguous copy.

A group's communicator is NCCL's bootstrap's (ncclCommInitRank, rank 0's ncclGetUniqueId passed through the group's
store, as ProcessGroupNCCL does), its sessions each on the map the bridges keep when it opens; it takes no pg_options.
The silent-link bound is the library's (nccl.h)."""
import math
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp, _create_work_from_future
from torch.futures import Future

from . import evidence

_stream = None
if torch.backends.mps.is_available():
    from . import _stream  # its kernels for torch's functional collectives on MPS register as it loads

import mesh  # noqa: E402  (rdma/mesh.py, on the path the mesh's install gives the environment: its _mesh_c)

ffi, LIB = mesh.ffi, mesh.lib
TYPES = {torch.int8: LIB.ncclInt8, torch.uint8: LIB.ncclUint8, torch.bool: LIB.ncclUint8, torch.int32: LIB.ncclInt32,
         torch.uint32: LIB.ncclUint32, torch.int64: LIB.ncclInt64, torch.uint64: LIB.ncclUint64, torch.float16: LIB.ncclFloat16,
         torch.float32: LIB.ncclFloat32, torch.float64: LIB.ncclFloat64, torch.bfloat16: LIB.ncclBfloat16,
         torch.float8_e4m3fn: LIB.ncclFloat8e4m3, torch.float8_e5m2: LIB.ncclFloat8e5m2}
OPS = ((ReduceOp.SUM, LIB.ncclSum), (ReduceOp.PRODUCT, LIB.ncclProd), (ReduceOp.MAX, LIB.ncclMax), (ReduceOp.MIN, LIB.ncclMin),
       (ReduceOp.AVG, LIB.ncclAvg))


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(LIB.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(LIB.ncclGetLastError(ffi.NULL)).decode()}')


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def kind(tensor):
    if tensor.dtype not in TYPES:
        raise NotImplementedError(f'the mesh backend moves {sorted(map(str, TYPES))}, not {tensor.dtype}')
    return TYPES[tensor.dtype]


def blank(shape, dtype, device):
    """An uninitialized contiguous tensor the backend hands a collective: a CPU one in this node's bridge's window (a
    lease, mesh.lease: rdma/NCCL.md "Memory"); an MPS one is torch's (the Metal path copies it into the window's
    rings)."""
    if torch.device(device).type != 'cpu':
        return torch.empty(shape, dtype=dtype, device=device)
    count, element = math.prod(shape), torch.empty(0, dtype=dtype).element_size()
    return torch.frombuffer(mesh.lease(max(count, 1) * element), dtype=dtype, count=max(count, 1))[:count].view(shape)


class Host:
    """A tensor's host image: the tensor itself where it is a contiguous CPU tensor, else a contiguous copy in the window
    (blank; `read` its contents, else uninitialized) that `back` writes into it."""

    def __init__(self, tensor, read=True):
        self.tensor = tensor
        self.direct = tensor.device.type == 'cpu' and tensor.is_contiguous()
        self.host = tensor if self.direct else blank(tensor.shape, tensor.dtype, 'cpu')
        if read and not self.direct:
            self.host.copy_(tensor.detach())

    def at(self, elements=0):
        return ffi.cast('void *', self.host.data_ptr() + elements * self.host.element_size())

    def back(self):
        if not self.direct:
            self.tensor.copy_(self.host)


class Device:
    """An MPS tensor where it lies: its storage's MTLBuffer and byte offset, `at` an ncclMeshBuffer kept alive
    until the group ends; a non-contiguous tensor through a contiguous copy (`read`) that `back` writes back."""

    def __init__(self, tensor, read=True):
        self.tensor = tensor
        self.direct = tensor.is_contiguous()
        self.host = tensor if self.direct else (tensor.contiguous() if read else torch.empty(tensor.shape, dtype=tensor.dtype,
                                                                                              device=tensor.device))
        self.made = []

    def at(self, elements=0):
        made = ffi.new('ncclMeshBuffer *')
        made.buffer = ffi.cast('void *', self.host.untyped_storage().data_ptr())
        made.offset = (self.host.storage_offset() + elements) * self.host.element_size()
        self.made.append(made)
        return made

    def back(self):
        if not self.direct:
            self.tensor.copy_(self.host)


def operand(tensor, read=True):
    return Device(tensor, read) if tensor.device.type == 'mps' else Host(tensor, read)


def on_mps(value):
    if isinstance(value, torch.Tensor):
        return value.device.type == 'mps'
    return isinstance(value, (list, tuple)) and any(on_mps(v) for v in value)


class After:
    """A write-back that is a function: run where the staged outputs are written back."""

    def __init__(self, back):
        self.back = back


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


def segments(values):
    """A call's segment counts (nccl.h ncclMeshAllGatherV, ncclMeshReduceScatterV) as the size_t array it reads."""
    return ffi.new('size_t[]', [int(v) for v in values])


def issued(works, result):
    """The last of a call's groups issued and not completed (completing it completes the groups before it: _stream.mm
    MeshWork), else its result done."""
    last = None
    for work in works:
        last = work if work is not None else last
    return last if last is not None else done(result)


def asynchronous(opts):
    """An async op (torch sets the option's asyncOp from async_op): issued now, completed where it is waited."""
    return bool(getattr(opts, 'asyncOp', False))


def done(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class ProcessGroupMesh(dist.ProcessGroup):
    _streamed = False
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
        self._rank, self._size, self._pending, self._stream = rank, size, None, ffi.NULL
        self.comm = comm
        self.handle = int(ffi.cast('uintptr_t', self.comm))
        if _stream is not None:
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
        group's work in flight first)."""
        made, ranks = ffi.new('ncclComm_t *'), ffi.new('int[]', list(exclude) or [0])
        check(LIB.ncclCommShrink(self.comm, ranks, len(exclude), made, ffi.NULL, 1 if abort else 0))
        return ProcessGroupMesh._of(made[0])

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
        """This group's work in flight ends with an error and none is issued after it (ncclCommRevoke)."""
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

    def _group(self, issue, outputs, result=None, inputs=None):
        """One NCCL group: `issue` makes its calls (their stream self._stream: torch's MPS stream, its open encoder
        and allocator, where the operands, `result` and `inputs`, are MPS tensors, else none), then every staged
        output is written back.
        An MPS group is committed at once, so its publications reach the peer while it computes on; a host group
        after MPS groups first completes every group issued and not completed and waits for the stream, as the
        session's positions run in its order.  Between
        start_coalescing and end_coalescing the calls join the coalesced group instead."""
        if self._pending is not None:
            self._pending.append((issue, outputs, result, inputs))
            return done(result)
        mps = on_mps(result) or on_mps(inputs)
        evidence.mark(self, True)
        if mps:
            self._stream = ffi.cast('void *', _stream.begin())
            ProcessGroupMesh._streamed = True
        else:
            self._stream = ffi.NULL
            if ProcessGroupMesh._streamed:
                _stream.complete_all()
                torch.mps.synchronize()
                ProcessGroupMesh._streamed = False
        check(LIB.ncclGroupStart())
        try:
            issue()
        finally:
            ended = LIB.ncclGroupEnd()
            if mps:
                _stream.commit()
            check(ended)
        for output in outputs:
            output.back()
        evidence.mark(self, False)
        return done(result)

    def _direct(self, *tensors):
        """torch's MPS stream, through the one-turn C++ calls (_stream.mm), where every tensor is a contiguous MPS
        tensor and no coalesced group is open; else None."""
        if self._pending is not None or _stream is None or not all(t.device.type == 'mps' and t.is_contiguous() for t in tensors):
            return None
        ProcessGroupMesh._streamed = True
        return _stream

    def start_coalescing(self, device):
        self._pending = []

    def end_coalescing(self, device):
        pending, self._pending = self._pending, None
        return self._group(lambda: [issue() for issue, _, _, _ in pending], [o for _, outputs, _, _ in pending for o in outputs],
                           [r for _, _, r, _ in pending], [i for _, _, _, i in pending])

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.allreduce(t, op(opts.reduceOp), self.handle, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclAllReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                            self.comm, self._stream)) for h in hosts], hosts, tensors)

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.reduce(t, op(opts.reduceOp), opts.rootRank, self.handle, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                         opts.rootRank, self.comm, self._stream)) for h in hosts], hosts, tensors)

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.broadcast(t, opts.rootRank, self.handle, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclBroadcast(h.at(), h.at(), h.host.numel(), kind(h.host), opts.rootRank,
                                                            self.comm, self._stream)) for h in hosts], hosts, tensors)

    def barrier(self, opts=None):
        return self.allreduce([blank((1,), torch.float32, 'cpu').fill_(1)])

    def all_gather_single(self, output, input, opts=None):
        if (s := self._direct(output, input)) is not None:
            return issued([s.allgather(output, input, self.handle, asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        return self._group(lambda: check(LIB.ncclAllGather(source.at(), target.at(), source.host.numel(), kind(source.host),
                                                           self.comm, self._stream)), [target], output)

    def all_gather_single_coalesced(self, outputs, inputs, opts=None):
        pairs = [(operand(i), operand(o, read=False)) for o, i in zip(outputs, inputs)]
        return self._group(lambda: [check(LIB.ncclAllGather(s.at(), t.at(), s.host.numel(), kind(s.host), self.comm, self._stream))
                                    for s, t in pairs], [t for _, t in pairs], outputs)

    allgather_into_tensor_coalesced = all_gather_single_coalesced

    def allgather(self, outputs, inputs, opts=None):
        """Each input gathered into its list of outputs; outputs of unequal sizes (an Allgatherv in torch's
        spelling) are MPI_Allgatherv, one program over the unequal segments (ncclMeshAllGatherV): into the
        outputs in place where they lie end to end in one contiguous tensor (a split's pieces: torch_mesh.partition's
        allgatherv), else into one flat operand copied out to them."""
        if all(len({t.numel() for t in out}) == 1 for out in outputs):
            sources = [operand(i) for i in inputs]
            flats = [operand(blank((self._size * s.host.numel(),), s.host.dtype, s.host.device), read=False)
                     for s in sources]

            def back():
                for out, flat in zip(outputs, flats):
                    for r, piece in enumerate(flat.host.chunk(self._size)):
                        out[r].copy_(piece.view(out[r].shape))
            return self._group(lambda: [check(LIB.ncclAllGather(s.at(), f.at(), s.host.numel(), kind(s.host), self.comm, self._stream))
                                        for s, f in zip(sources, flats)], [After(back)], outputs)
        sizes, spans = [[t.numel() for t in out] for out in outputs], [span(out) for out in outputs]
        if all(w is not None for w in spans) and (s := self._direct(*inputs, *spans)) is not None:
            return issued([s.allgatherv(w, i, c, self.handle, asynchronous(opts)) for w, i, c in zip(spans, inputs, sizes)], outputs)
        sources = [operand(i) for i in inputs]
        wholes = [operand(w if w is not None else blank((sum(c),), i.dtype, i.device), read=False)
                  for w, i, c in zip(spans, inputs, sizes)]

        def back():
            for out, whole, c, w in zip(outputs, wholes, sizes, spans):
                whole.back()
                if w is None:
                    for t, piece in zip(out, whole.host.split(c)):
                        t.copy_(piece.view(t.shape))
        return self._group(lambda: [check(LIB.ncclMeshAllGatherV(s.at(), w.at(), segments(c), kind(s.host), self.comm, self._stream))
                                    for s, w, c in zip(sources, wholes, sizes)], [After(back)], outputs)

    def reduce_scatter_single(self, output, input, opts=dist.ReduceScatterOptions()):
        if (s := self._direct(output, input)) is not None:
            return issued([s.reduce_scatter(output, input, op(opts.reduceOp), self.handle, asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        return self._group(lambda: check(LIB.ncclReduceScatter(source.at(), target.at(), target.host.numel(), kind(source.host),
                                                               op(opts.reduceOp), self.comm, self._stream)), [target], output)

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        pairs = [(operand(i), operand(o, read=False)) for o, i in zip(outputs, inputs)]
        return self._group(lambda: [check(LIB.ncclReduceScatter(s.at(), t.at(), t.host.numel(), kind(s.host), op(opts.reduceOp),
                                                                 self.comm, self._stream)) for s, t in pairs], [t for _, t in pairs], outputs)

    reduce_scatter_tensor_coalesced = reduce_scatter_single_coalesced

    def reduce_scatter(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        """Each list of inputs reduced, rank r's block left at rank r; blocks of unequal sizes (a
        Reduce_scatter of a list in torch's spelling) are MPI_Reduce_scatter, one program over the unequal segments
        (ncclMeshReduceScatterV): read in place where the inputs lie end to end in one contiguous tensor (a split's
        pieces: torch_mesh.partition's reduce_scatterv), else from one flat operand they are copied into."""
        if all(len({t.numel() for t in parts}) == 1 for parts in inputs):
            flats = [operand(torch.cat([x.detach().reshape(-1) for x in parts],
                                       out=blank((sum(x.numel() for x in parts),), parts[0].dtype, parts[0].device))) for parts in inputs]
            targets = [operand(o, read=False) for o in outputs]
            return self._group(lambda: [check(LIB.ncclReduceScatter(f.at(), t.at(), t.host.numel(), kind(t.host),
                                                                    op(opts.reduceOp), self.comm, self._stream))
                                        for f, t in zip(flats, targets)], targets, outputs)
        sizes, spans = [[x.numel() for x in parts] for parts in inputs], [span(parts) for parts in inputs]
        if all(w is not None for w in spans) and (s := self._direct(*outputs, *spans)) is not None:
            return issued([s.reduce_scatterv(o, w, c, op(opts.reduceOp), self.handle, asynchronous(opts))
                           for o, w, c in zip(outputs, spans, sizes)], outputs)
        wholes = [operand(w if w is not None else torch.cat([x.detach().reshape(-1) for x in parts],
                                                             out=blank((sum(c),), parts[0].dtype, parts[0].device)))
                  for w, parts, c in zip(spans, inputs, sizes)]
        targets = [operand(o, read=False) for o in outputs]
        return self._group(lambda: [check(LIB.ncclMeshReduceScatterV(w.at(), t.at(), segments(c), kind(t.host), op(opts.reduceOp),
                                                                     self.comm, self._stream))
                                    for w, t, c in zip(wholes, targets, sizes)], targets, outputs)

    def alltoall(self, outputs, inputs, opts=None):
        sources, targets = [operand(i) for i in inputs], [operand(o, read=False) for o in outputs]

        def issue():
            for r in range(self._size):
                check(LIB.ncclSend(sources[r].at(), sources[r].host.numel(), kind(sources[r].host), r, self.comm, self._stream))
                check(LIB.ncclRecv(targets[r].at(), targets[r].host.numel(), kind(targets[r].host), r, self.comm, self._stream))
        return self._group(issue, targets, outputs)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        if (s := self._direct(output, input)) is not None:
            width = math.prod(input.shape[1:])
            rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
            rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
            return issued([s.alltoall(output, input, [r * width for r in rows_in], [r * width for r in rows_out], self.handle,
                                      asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        if not output_split_sizes and not input_split_sizes:
            return self._group(lambda: check(LIB.ncclAlltoAll(source.at(), target.at(), source.host.numel() // self._size,
                                                              kind(source.host), self.comm, self._stream)), [target], output)
        rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
        rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
        width = math.prod(input.shape[1:])

        def issue():
            at_in = at_out = 0
            for r in range(self._size):
                check(LIB.ncclSend(source.at(at_in * width), rows_in[r] * width, kind(source.host), r, self.comm, self._stream))
                check(LIB.ncclRecv(target.at(at_out * width), rows_out[r] * width, kind(target.host), r, self.comm, self._stream))
                at_in, at_out = at_in + rows_in[r], at_out + rows_out[r]
        return self._group(issue, [target], output)

    def gather(self, outputs, inputs, opts=dist.GatherOptions()):
        """Every rank's input to the root's list of outputs, any sizes: grouped sends and receives, as
        ProcessGroupNCCL's gather."""
        source = operand(inputs[0])
        targets = [operand(t, read=False) for t in outputs[0]] if self._rank == opts.rootRank else []

        def issue():
            check(LIB.ncclSend(source.at(), source.host.numel(), kind(source.host), opts.rootRank, self.comm, self._stream))
            for r, target in enumerate(targets):
                check(LIB.ncclRecv(target.at(), target.host.numel(), kind(target.host), r, self.comm, self._stream))
        return self._group(issue, targets, outputs, inputs)

    def scatter(self, outputs, inputs, opts=dist.ScatterOptions()):
        """The root's list of inputs, one to each rank, any sizes (a Scatterv in torch's spelling): grouped sends
        and receives, as ProcessGroupNCCL's scatter."""
        target = operand(outputs[0], read=False)
        sources = [operand(t) for t in inputs[0]] if self._rank == opts.rootRank else []

        def issue():
            for r, source in enumerate(sources):
                check(LIB.ncclSend(source.at(), source.host.numel(), kind(source.host), r, self.comm, self._stream))
            check(LIB.ncclRecv(target.at(), target.host.numel(), kind(target.host), opts.rootRank, self.comm, self._stream))
        return self._group(issue, [target], outputs)

    def all_to_all_counted(self, out, input, counts, sends, receives, host, landed):
        """MPI_Alltoallv whose counts the GPU wrote (nccl.h ncclMeshAlltoAllCounted; torch_mesh.all_to_all_counted):
        `input`'s rows grouped by destination, as many to rank q as segment q of `counts` sums to, received packed in
        rank order into `out`, the received counts into host[1:] (host[0] set once they landed) and `landed`.  On MPS
        issued: its rows and `landed` complete on torch's stream where wait_tensor(out) waits; on the host path at once."""
        if (s := self._direct(out, input, counts, landed)) is not None:
            s.alltoall_counted(out, input, counts, list(sends), list(receives), host, landed, self.handle)
            return
        source, number = operand(input), operand(counts)
        target, arrived = operand(out, read=False), operand(landed, read=False)
        words = ffi.cast('int64_t *', host.data_ptr())
        self._group(lambda: check(LIB.ncclMeshAlltoAllCounted(
            source.at(), input.shape[0], ffi.cast('const int64_t *', number.at()), segments(sends), target.at(), out.shape[0],
            words + 1, ffi.cast('int64_t *', arrived.at()), segments(receives), ffi.cast('uint64_t *', words),
            math.prod(input.shape[1:]), kind(source.host), self.comm, self._stream)), [target, arrived], [out, landed], [input])

    def send(self, tensors, dst, tag):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.send(t, dst, self.handle, True) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclSend(h.at(), h.host.numel(), kind(h.host), dst, self.comm, self._stream))
                                    for h in hosts], [], tensors)

    def recv(self, tensors, src, tag):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.recv(t, src, self.handle, True) for t in tensors], tensors)
        hosts = [operand(t, read=False) for t in tensors]
        return self._group(lambda: [check(LIB.ncclRecv(h.at(), h.host.numel(), kind(h.host), src, self.comm, self._stream))
                                    for h in hosts], hosts, tensors)


@torch.library.custom_op('torch_mesh::all_to_all_counted', mutates_args=('host',))
def all_to_all_counted(input: torch.Tensor, counts: torch.Tensor, send_segments: list[int], recv_segments: list[int], capacity: int,
                       host: torch.Tensor, group_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The counted all-to-all on the mesh group named `group_name` as a torch operator (torch_mesh.all_to_all_counted):
    (the rows received, `capacity` of them; the received counts on the input's device), host[1:] the received counts
    on the host once host[0] is set (counted)."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if not isinstance(group, ProcessGroupMesh):
        raise TypeError(f'all_to_all_counted runs on a group of the mesh backend, not {group_name}\'s')
    out = torch.empty((capacity,) + tuple(input.shape[1:]), dtype=input.dtype, device=input.device)
    landed = torch.empty(sum(recv_segments), dtype=torch.int64, device=input.device)
    group.all_to_all_counted(out, input.contiguous(), counts.contiguous(), send_segments, recv_segments, host, landed)
    return out, landed


@all_to_all_counted.register_fake
def _(input, counts, send_segments, recv_segments, capacity, host, group_name):
    return input.new_empty((capacity,) + tuple(input.shape[1:])), counts.new_empty(sum(recv_segments))


def counted(host, group):
    """A counted all-to-all's received counts once they have landed (host[0] set), read on the host with no wait on
    the GPU: the wait ends at their arrival or at the call's failure (host[0] 2, or the communicator's asynchronous
    error: a lost, silent or revoked link's cancellation, nccl.h), never at a time."""
    word, error = ffi.cast('uint64_t *', host.data_ptr()), ffi.new('ncclResult_t *')
    while not word[0]:
        check(LIB.ncclCommGetAsyncError(group.comm, error))
        check(error[0])
        os.sched_yield()
    if word[0] != 1:
        check(LIB.ncclCommGetAsyncError(group.comm, error))
        check(error[0])
        raise RuntimeError('libnccl-mesh: all_to_all_counted failed before its counts arrived (its wait raises why)')
    return host[1:].tolist()


def create(opts, options):
    """The group (torch's extended creator: `opts` its store, rank and size, `options` the caller's pg_options)."""
    if options is not None:
        raise TypeError("the mesh backend takes no pg_options: a group's map is its bridges'")
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'  # init_process_group() without a backend: MPS tensors on the mesh
