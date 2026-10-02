"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h on the bridges' prepared transfers).  A tensor that is not a contiguous CPU
tensor is staged through a contiguous CPU copy and written back; every call returns once complete.

A rank's communicator is the clique's explicit configuration (MESH_LINKS, MESH_REGION; ncclGetUniqueId) over
the bridges' nodes, and NODES (comma-separated, default 0,1,...) names the node of each torch rank: the
communicator is that one split with torch's rank as key (ncclCommSplit), so ranks are torch's."""
import ctypes as C
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp, _create_work_from_future
from torch.futures import Future

RDMA = os.environ.get('MESH_RDMA') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = C.CDLL(os.path.join(RDMA, 'libnccl-mesh.dylib'))
P, I, Z = C.c_void_p, C.c_int, C.c_size_t


class UniqueId(C.Structure):
    _fields_ = [('internal', C.c_char * 128)]


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


class After:
    """A write-back that is a function: run where the staged outputs are written back."""

    def __init__(self, back):
        self.back = back


def done(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class ProcessGroupMesh(dist.ProcessGroup):
    def __init__(self, rank, size):
        super().__init__(rank, size)
        self._rank, self._size, self._pending = rank, size, None
        self.comm = P()
        if size == 1:
            check(LIB.ncclCommInitAll(C.byref(self.comm), 1, None))
            return
        nodes = [int(v) for v in os.environ.get('NODES', ','.join(map(str, range(size)))).split(',')]
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
        """One NCCL group: `issue` makes its calls on host pointers, then every staged output is written back.
        Between start_coalescing and end_coalescing the calls join the coalesced group instead."""
        if self._pending is not None:
            self._pending.append((issue, outputs))
            return done(result)
        check(LIB.ncclGroupStart())
        try:
            issue()
        finally:
            check(LIB.ncclGroupEnd())
        for output in outputs:
            output.back()
        return done(result)

    def start_coalescing(self, device):
        self._pending = []

    def end_coalescing(self, device):
        pending, self._pending = self._pending, None
        return self._group(lambda: [issue() for issue, _ in pending], [o for _, outputs in pending for o in outputs])

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        hosts = [Host(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclAllReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                            self.comm, None)) for h in hosts], hosts, tensors)

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        hosts = [Host(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                         opts.rootRank, self.comm, None)) for h in hosts], hosts, tensors)

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        hosts = [Host(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclBroadcast(h.at(), h.at(), h.host.numel(), kind(h.host), opts.rootRank,
                                                            self.comm, None)) for h in hosts], hosts, tensors)

    def barrier(self, opts=None):
        return self.allreduce([torch.ones(1)])

    def all_gather_single(self, output, input, opts=None):
        source, target = Host(input), Host(output, read=False)
        return self._group(lambda: check(LIB.ncclAllGather(source.at(), target.at(), source.host.numel(), kind(source.host),
                                                           self.comm, None)), [target], output)

    def all_gather_single_coalesced(self, outputs, inputs, opts=None):
        pairs = [(Host(i), Host(o, read=False)) for o, i in zip(outputs, inputs)]
        return self._group(lambda: [check(LIB.ncclAllGather(s.at(), t.at(), s.host.numel(), kind(s.host), self.comm, None))
                                    for s, t in pairs], [t for _, t in pairs], outputs)

    allgather_into_tensor_coalesced = all_gather_single_coalesced

    def allgather(self, outputs, inputs, opts=None):
        sources = [Host(i) for i in inputs]
        flats = [torch.empty(self._size * s.host.numel(), dtype=s.host.dtype) for s in sources]

        def back():
            for out, flat in zip(outputs, flats):
                for r, piece in enumerate(flat.chunk(self._size)):
                    out[r].copy_(piece.view(out[r].shape))
        return self._group(lambda: [check(LIB.ncclAllGather(s.at(), f.data_ptr(), s.host.numel(), kind(s.host), self.comm, None))
                                    for s, f in zip(sources, flats)], [After(back)], outputs)

    def reduce_scatter_single(self, output, input, opts=dist.ReduceScatterOptions()):
        source, target = Host(input), Host(output, read=False)
        return self._group(lambda: check(LIB.ncclReduceScatter(source.at(), target.at(), target.host.numel(), kind(source.host),
                                                               op(opts.reduceOp), self.comm, None)), [target], output)

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        pairs = [(Host(i), Host(o, read=False)) for o, i in zip(outputs, inputs)]
        return self._group(lambda: [check(LIB.ncclReduceScatter(s.at(), t.at(), t.host.numel(), kind(s.host), op(opts.reduceOp),
                                                                 self.comm, None)) for s, t in pairs], [t for _, t in pairs], outputs)

    reduce_scatter_tensor_coalesced = reduce_scatter_single_coalesced

    def reduce_scatter(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        flats = [torch.cat([x.detach().to('cpu').reshape(-1) for x in parts]) for parts in inputs]
        targets = [Host(o, read=False) for o in outputs]
        return self._group(lambda: [check(LIB.ncclReduceScatter(f.data_ptr(), t.at(), t.host.numel(), kind(t.host),
                                                                op(opts.reduceOp), self.comm, None))
                                    for f, t in zip(flats, targets)], targets, outputs)

    def alltoall(self, outputs, inputs, opts=None):
        sources, targets = [Host(i) for i in inputs], [Host(o, read=False) for o in outputs]

        def issue():
            for r in range(self._size):
                check(LIB.ncclSend(sources[r].at(), sources[r].host.numel(), kind(sources[r].host), r, self.comm, None))
                check(LIB.ncclRecv(targets[r].at(), targets[r].host.numel(), kind(targets[r].host), r, self.comm, None))
        return self._group(issue, targets, outputs)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        source, target = Host(input), Host(output, read=False)
        if not output_split_sizes and not input_split_sizes:
            return self._group(lambda: check(LIB.ncclAlltoAll(source.at(), target.at(), source.host.numel() // self._size,
                                                              kind(source.host), self.comm, None)), [target], output)
        rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
        rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
        width = input[0].numel() if input.dim() and input.shape[0] else 1

        def issue():
            at_in = at_out = 0
            for r in range(self._size):
                check(LIB.ncclSend(source.at(at_in * width), rows_in[r] * width, kind(source.host), r, self.comm, None))
                check(LIB.ncclRecv(target.at(at_out * width), rows_out[r] * width, kind(target.host), r, self.comm, None))
                at_in, at_out = at_in + rows_in[r], at_out + rows_out[r]
        return self._group(issue, [target], output)

    def gather(self, outputs, inputs, opts=dist.GatherOptions()):
        source = Host(inputs[0])
        flat = torch.empty(self._size * source.host.numel(), dtype=source.host.dtype)

        def back():
            if self._rank == opts.rootRank:
                for r, piece in enumerate(flat.chunk(self._size)):
                    outputs[0][r].copy_(piece.view(outputs[0][r].shape))
        return self._group(lambda: check(LIB.ncclGather(source.at(), flat.data_ptr(), source.host.numel(), kind(source.host),
                                                        opts.rootRank, self.comm, None)), [After(back)], outputs)

    def scatter(self, outputs, inputs, opts=dist.ScatterOptions()):
        target = Host(outputs[0], read=False)
        flat = (torch.cat([x.detach().to('cpu').reshape(-1) for x in inputs[0]]) if self._rank == opts.rootRank
                else torch.empty(self._size * target.host.numel(), dtype=target.host.dtype))
        return self._group(lambda: check(LIB.ncclScatter(flat.data_ptr(), target.at(), target.host.numel(), kind(target.host),
                                                         opts.rootRank, self.comm, None)), [target], outputs)

    def send(self, tensors, dst, tag):
        hosts = [Host(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclSend(h.at(), h.host.numel(), kind(h.host), dst, self.comm, None))
                                    for h in hosts], [], tensors)

    def recv(self, tensors, src, tag):
        hosts = [Host(t, read=False) for t in tensors]
        return self._group(lambda: [check(LIB.ncclRecv(h.at(), h.host.numel(), kind(h.host), src, self.comm, None))
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
    return ProcessGroupMesh(rank, size)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'])
