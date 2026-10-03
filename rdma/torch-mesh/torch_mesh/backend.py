"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h on the bridges' prepared transfers).  MPS tensors take the library's Metal
path: the group is encoded on torch's MPS stream (_stream.mm) and committed, each tensor passed as its
storage's MTLBuffer and byte offset (ncclMeshBuffer), so the work is ordered with torch's own and the call
returns at once.  CPU tensors take the host path, a non-contiguous one staged through a contiguous copy.

A rank's communicator is the clique's explicit configuration (MESH_LINKS, MESH_REGION; ncclGetUniqueId) over
the bridges' nodes; each rank reads its node from its own bridge (mesh_observe on MESH_REGION) and the ranks
exchange theirs through the group's store, so the communicator is that one split with torch's rank as key
(ncclCommSplit), its ranks torch's."""
import ctypes as C
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp, _create_work_from_future
from torch.futures import Future

_stream = None
if torch.backends.mps.is_available():
    from . import _stream  # its kernels for torch's functional collectives on MPS register as it loads

RDMA = os.environ.get('MESH_RDMA') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = C.CDLL(os.path.join(RDMA, 'libnccl-mesh.dylib'))
MESH = C.CDLL(os.path.join(RDMA, 'libmesh.dylib'))
MESH.mesh_observe.restype, MESH.mesh_observe.argtypes = C.c_int, [C.c_char_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_uint32)]
P, I, Z = C.c_void_p, C.c_int, C.c_size_t


class UniqueId(C.Structure):
    _fields_ = [('internal', C.c_char * 128)]


class MeshBuffer(C.Structure):
    _fields_ = [('buffer', C.c_void_p), ('offset', C.c_size_t)]


for name, arguments in (
        ('ncclGetUniqueId', [C.POINTER(UniqueId)]), ('ncclCommInitRank', [C.POINTER(P), I, UniqueId, I]),
        ('ncclCommInitAll', [C.POINTER(P), I, P]), ('ncclCommSplit', [P, I, I, C.POINTER(P), P]),
        ('ncclCommDestroy', [P]), ('ncclGroupStart', []), ('ncclGroupEnd', []),
        ('ncclAllReduce', [P, P, Z, I, I, P, P]), ('ncclReduce', [P, P, Z, I, I, I, P, P]),
        ('ncclBroadcast', [P, P, Z, I, I, P, P]), ('ncclAllGather', [P, P, Z, I, P, P]),
        ('ncclReduceScatter', [P, P, Z, I, I, P, P]), ('ncclSend', [P, Z, I, I, P, P]), ('ncclRecv', [P, Z, I, I, P, P]),
        ('ncclAlltoAll', [P, P, Z, I, P, P]), ('ncclGather', [P, P, Z, I, I, P, P]), ('ncclScatter', [P, P, Z, I, I, P, P])):
    getattr(LIB, name).restype, getattr(LIB, name).argtypes = I, arguments
LIB.ncclGetErrorString.restype, LIB.ncclGetErrorString.argtypes = C.c_char_p, [I]
LIB.ncclGetLastError.restype, LIB.ncclGetLastError.argtypes = C.c_char_p, [P]

TYPES = {torch.int8: 0, torch.uint8: 1, torch.bool: 1, torch.int32: 2, torch.uint32: 3, torch.int64: 4,
         torch.uint64: 5, torch.float16: 6, torch.float32: 7, torch.float64: 8, torch.bfloat16: 9,
         torch.float8_e4m3fn: 10, torch.float8_e5m2: 11}
OPS = ((ReduceOp.SUM, 0), (ReduceOp.PRODUCT, 1), (ReduceOp.MAX, 2), (ReduceOp.MIN, 3), (ReduceOp.AVG, 4))


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {LIB.ncclGetErrorString(result).decode()}: {LIB.ncclGetLastError(None).decode()}')


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def kind(tensor):
    if tensor.dtype not in TYPES:
        raise NotImplementedError(f'the mesh backend moves {sorted(map(str, TYPES))}, not {tensor.dtype}')
    return TYPES[tensor.dtype]


class Host:
    """A tensor's host image: the tensor itself where it is a contiguous CPU tensor, else a contiguous CPU copy
    (`read` its contents, else uninitialized) that `back` writes into it."""

    def __init__(self, tensor, read=True):
        self.tensor = tensor
        self.direct = tensor.device.type == 'cpu' and tensor.is_contiguous()
        self.host = tensor if self.direct else (tensor.detach().to('cpu').contiguous() if read
                                                else torch.empty(tensor.shape, dtype=tensor.dtype))

    def at(self, elements=0):
        return self.host.data_ptr() + elements * self.host.element_size()

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
        made = MeshBuffer(self.host.untyped_storage().data_ptr(), (self.host.storage_offset() + elements) * self.host.element_size())
        self.made.append(made)
        return C.addressof(made)

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

    def __init__(self, rank, size, store=None):
        super().__init__(rank, size)
        self._rank, self._size, self._pending = rank, size, None
        self._stream = None
        self.comm = P()
        if size == 1:
            check(LIB.ncclCommInitAll(C.byref(self.comm), 1, None))
        else:
            self._join(rank, size, store)
        if _stream is not None:
            _stream.attach(self, self.comm.value)

    def _join(self, rank, size, store):
        node = C.c_uint32()
        observed = MESH.mesh_observe((os.environ.get('MESH_REGION') or '/mesh0').encode(), None, 0, C.byref(node))
        if observed < 0:
            raise RuntimeError(f'libmesh: the bridge of region {os.environ.get("MESH_REGION") or "/mesh0"}: '
                               f'{os.strerror(-observed)}')
        store.set(f'mesh/node/{rank}', str(node.value))
        nodes = [int(store.get(f'mesh/node/{r}')) for r in range(size)]
        unique, base = UniqueId(), P()
        check(LIB.ncclGetUniqueId(C.byref(unique)))
        check(LIB.ncclCommInitRank(C.byref(base), size, unique, nodes[rank]))
        check(LIB.ncclCommSplit(base, 0, rank, C.byref(self.comm), None))
        check(LIB.ncclCommDestroy(base))

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    @property
    def group_name(self):
        """The name torch registered this group under (a creator's ProcessGroup is never given its name)."""
        return dist.distributed_c10d._world.pg_names[self]

    pg_name = group_name

    def _group(self, issue, outputs, result=None):
        """One NCCL group: `issue` makes its calls (their stream self._stream: torch's MPS stream, its open encoder
        and allocator, where the operands are MPS tensors, else none), then every staged output is written back.
        An MPS group is committed at once, so its publications reach the peer while it computes on; a host group
        after MPS groups first completes every group issued and not completed and waits for the stream, as the
        session's positions run in its order.  Between
        start_coalescing and end_coalescing the calls join the coalesced group instead."""
        if self._pending is not None:
            self._pending.append((issue, outputs, result))
            return done(result)
        mps = on_mps(result)
        if mps:
            self._stream = _stream.begin()
            ProcessGroupMesh._streamed = True
        else:
            self._stream = None
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
        return self._group(lambda: [issue() for issue, _, _ in pending], [o for _, outputs, _ in pending for o in outputs],
                           [r for _, _, r in pending])

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.allreduce(t, op(opts.reduceOp), self.comm.value, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclAllReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                            self.comm, self._stream)) for h in hosts], hosts, tensors)

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.reduce(t, op(opts.reduceOp), opts.rootRank, self.comm.value, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                         opts.rootRank, self.comm, self._stream)) for h in hosts], hosts, tensors)

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.broadcast(t, opts.rootRank, self.comm.value, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclBroadcast(h.at(), h.at(), h.host.numel(), kind(h.host), opts.rootRank,
                                                            self.comm, self._stream)) for h in hosts], hosts, tensors)

    def barrier(self, opts=None):
        return self.allreduce([torch.ones(1)])

    def all_gather_single(self, output, input, opts=None):
        if (s := self._direct(output, input)) is not None:
            return issued([s.allgather(output, input, self.comm.value, asynchronous(opts))], output)
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
        spelling) as ProcessGroupNCCL takes them: one broadcast rooted at each rank, in one group."""
        if all(len({t.numel() for t in out}) == 1 for out in outputs):
            sources = [operand(i) for i in inputs]
            flats = [operand(torch.empty(self._size * s.host.numel(), dtype=s.host.dtype, device=s.host.device), read=False)
                     for s in sources]

            def back():
                for out, flat in zip(outputs, flats):
                    for r, piece in enumerate(flat.host.chunk(self._size)):
                        out[r].copy_(piece.view(out[r].shape))
            return self._group(lambda: [check(LIB.ncclAllGather(s.at(), f.at(), s.host.numel(), kind(s.host), self.comm, self._stream))
                                        for s, f in zip(sources, flats)], [After(back)], outputs)
        sources, targets = [operand(i) for i in inputs], [[operand(t, read=False) for t in out] for out in outputs]

        def issue():
            for source, out in zip(sources, targets):
                for r, target in enumerate(out):
                    sent = source if r == self._rank else target
                    check(LIB.ncclBroadcast(sent.at(), target.at(), target.host.numel(), kind(target.host), r, self.comm, self._stream))
        return self._group(issue, [t for out in targets for t in out], outputs)

    def reduce_scatter_single(self, output, input, opts=dist.ReduceScatterOptions()):
        if (s := self._direct(output, input)) is not None:
            return issued([s.reduce_scatter(output, input, op(opts.reduceOp), self.comm.value, asynchronous(opts))], output)
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
        Reduce_scatter of a list in torch's spelling) as ProcessGroupNCCL takes them: one reduce rooted at
        each rank, in one group."""
        if all(len({t.numel() for t in parts}) == 1 for parts in inputs):
            flats = [operand(torch.cat([x.detach().reshape(-1) for x in parts])) for parts in inputs]
            targets = [operand(o, read=False) for o in outputs]
            return self._group(lambda: [check(LIB.ncclReduceScatter(f.at(), t.at(), t.host.numel(), kind(t.host),
                                                                    op(opts.reduceOp), self.comm, self._stream))
                                        for f, t in zip(flats, targets)], targets, outputs)
        sources, targets = [[operand(x) for x in parts] for parts in inputs], [operand(o, read=False) for o in outputs]

        def issue():
            for parts, target in zip(sources, targets):
                for r, source in enumerate(parts):
                    received = target if r == self._rank else source
                    check(LIB.ncclReduce(source.at(), received.at(), source.host.numel(), kind(source.host), op(opts.reduceOp),
                                         r, self.comm, self._stream))
        return self._group(issue, targets, outputs)

    def alltoall(self, outputs, inputs, opts=None):
        sources, targets = [operand(i) for i in inputs], [operand(o, read=False) for o in outputs]

        def issue():
            for r in range(self._size):
                check(LIB.ncclSend(sources[r].at(), sources[r].host.numel(), kind(sources[r].host), r, self.comm, self._stream))
                check(LIB.ncclRecv(targets[r].at(), targets[r].host.numel(), kind(targets[r].host), r, self.comm, self._stream))
        return self._group(issue, targets, outputs)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        if (s := self._direct(output, input)) is not None:
            width = input[0].numel() if input.dim() and input.shape[0] else 1
            rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
            rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
            return issued([s.alltoall(output, input, [r * width for r in rows_in], [r * width for r in rows_out], self.comm.value,
                                      asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        if not output_split_sizes and not input_split_sizes:
            return self._group(lambda: check(LIB.ncclAlltoAll(source.at(), target.at(), source.host.numel() // self._size,
                                                              kind(source.host), self.comm, self._stream)), [target], output)
        rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
        rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
        width = input[0].numel() if input.dim() and input.shape[0] else 1

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
        return self._group(issue, targets, outputs)

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

    def send(self, tensors, dst, tag):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.send(t, dst, self.comm.value, True) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclSend(h.at(), h.host.numel(), kind(h.host), dst, self.comm, self._stream))
                                    for h in hosts], [], tensors)

    def recv(self, tensors, src, tag):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.recv(t, src, self.comm.value, True) for t in tensors], tensors)
        hosts = [operand(t, read=False) for t in tensors]
        return self._group(lambda: [check(LIB.ncclRecv(h.at(), h.host.numel(), kind(h.host), src, self.comm, self._stream))
                                    for h in hosts], hosts, tensors)


_installed = False


def create(store, rank, size, timeout):
    """The group; the first one also completes what PyTorch's parallelism APIs need of MPS tensors (_mps.py)
    and installs the partition header (partition.py)."""
    global _installed
    if not _installed:
        from . import _mps, partition  # noqa: F401 (partition installs itself)
        _mps.register()
        _installed = True
    return ProcessGroupMesh(rank, size, store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'])
