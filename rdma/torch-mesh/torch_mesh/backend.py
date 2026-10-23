import math
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp, _create_work_from_future, _register_work
from torch.futures import Future

from . import _stream, evidence

import mesh

ffi, LIB = mesh.ffi, mesh.lib
OPS = ((ReduceOp.SUM, LIB.ncclSum), (ReduceOp.PRODUCT, LIB.ncclProd), (ReduceOp.MAX, LIB.ncclMax), (ReduceOp.MIN, LIB.ncclMin),
       (ReduceOp.AVG, LIB.ncclAvg))
ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, SEND, RECV, COUNTED, PAIRS = range(10)


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(LIB.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(LIB.ncclGetLastError(ffi.NULL)).decode()}')


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def blank(shape, dtype, device):
    kind = torch.device(device).type
    if kind == 'mps':
        return _stream.region_empty(list(shape), torch.empty(0, dtype=dtype), True)
    if kind != 'cpu':
        return torch.empty(shape, dtype=dtype, device=device)
    count, element = math.prod(shape), torch.empty(0, dtype=dtype).element_size()
    return torch.frombuffer(mesh.lease(max(count, 1) * element), dtype=dtype, count=max(count, 1))[:count].view(shape)


def span(tensors):
    first, at = tensors[0], tensors[0].storage_offset()
    for t in tensors:
        if (t.dtype != first.dtype or t.device != first.device or not t.is_contiguous() or t.storage_offset() != at
                or t.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()):
            return None
        at += t.numel()
    return first.as_strided((at - first.storage_offset(),), (1,))


def packed(tensors, read):
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None, None
    whole = span(tensors)
    if whole is not None:
        return whole, None
    whole = blank((sum(t.numel() for t in tensors),), tensors[0].dtype, tensors[0].device)
    if read:
        return torch.cat([t.detach().reshape(-1) for t in tensors], out=whole), None
    return whole, lambda: [t.copy_(p.view(t.shape)) for t, p in zip(tensors, whole.split([t.numel() for t in tensors]))]


def asynchronous(opts):
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

    @classmethod
    def _of(cls, comm):
        count, rank = ffi.new('int *'), ffi.new('int *')
        check(LIB.ncclCommCount(comm, count))
        check(LIB.ncclCommUserRank(comm, rank))
        made = cls.__new__(cls)
        dist.ProcessGroup.__init__(made, rank[0], count[0])
        made._held(rank[0], count[0], comm)
        return made

    def shrink(self, exclude, abort=False):
        made, ranks = ffi.new('ncclComm_t *'), ffi.new('int[]', list(exclude) or [0])
        check(LIB.ncclCommShrink(self.comm, ranks, len(exclude), made, ffi.NULL, 1 if abort else 0))
        return ProcessGroupMesh._of(made[0]) if made[0] != ffi.NULL else None

    def grow(self, size, unique=None, rank=-1):
        made, held = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if self is not None and self._rank == 0:
            check(LIB.ncclCommGetUniqueId(self.comm, held))
        elif unique is not None:
            ffi.memmove(held, unique, ffi.sizeof('ncclUniqueId'))
        check(LIB.ncclCommGrow(ffi.NULL if self is None else self.comm, size, held, rank, made, ffi.NULL))
        return ProcessGroupMesh._of(made[0]), bytes(ffi.buffer(held))

    def revoke(self):
        check(LIB.ncclCommRevoke(self.comm, 0))

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    @property
    def group_name(self):
        return dist.distributed_c10d._world.pg_names[self]

    pg_name = group_name

    def _issue(self, calls, result, split=False, backs=(), registered=None):
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
        if registered is not None:
            _register_work(registered, work)
        return work

    def _alltoall(self, sends, receives, result, opts, matrix):
        (send, _), (receive, back) = packed(sends, True), packed(receives, False)
        return self._issue([(ALLTOALL, [send], [receive], matrix, 0, 0)], result, asynchronous(opts), [back] if back else ())

    def _rooted(self, root, tensor):
        return [tensor if q == root else None for q in range(self._size)]

    def start_coalescing(self, device):
        self._pending = []

    def end_coalescing(self, device):
        pending, self._pending = self._pending, None
        return self._issue([c for calls, _, _ in pending for c in calls], [r for _, r, _ in pending],
                           backs=[b for _, _, backs in pending for b in backs])

    def all_to_all_matrix(self, output, input, matrix):
        n, me = self._size, self._rank
        row, column = matrix[me * n:(me + 1) * n], matrix[me::n]
        sends = list(input.reshape(-1).split(row)) if input is not None else [None] * n
        return self._alltoall(sends, list(output.reshape(-1).split(column)), output, None, list(matrix))

    def pairs(self, ops, pairs):
        self.start_coalescing(None)
        for op in ops:
            op.op(op.tensor, op.peer, op.group, op.tag)
        self._pending.append(([(PAIRS, [], [], [x for p in pairs for x in p], 0, 0)], [], ()))
        return self.end_coalescing(None)

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        return self._issue([(ALLREDUCE, [t], [], [], op(opts.reduceOp), 0) for t in tensors], tensors, asynchronous(opts))

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        return self._issue([(REDUCE, [t], [], [], op(opts.reduceOp), opts.rootRank) for t in tensors], tensors, asynchronous(opts))

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        return self._issue([(BROADCAST, [t], [], [], 0, opts.rootRank) for t in tensors], tensors, asynchronous(opts))

    def barrier(self, opts=None):
        return self.allreduce([blank((1,), torch.float32, 'cpu').fill_(1)])

    def all_gather_single_coalesced(self, outputs, inputs, opts=None):
        return self._issue([(ALLGATHER, [i], [o], [i.numel()] * self._size, 0, 0) for o, i in zip(outputs, inputs)], outputs,
                           asynchronous(opts))

    def all_gather_single(self, output, input, opts=None):
        return self.all_gather_single_coalesced([output], [input], opts)

    allgather_into_tensor_coalesced = all_gather_single_coalesced

    def allgather(self, outputs, inputs, opts=None):
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
        return self._issue([(REDUCE_SCATTER, [packed(parts, True)[0]], [o], [x.numel() for x in parts], op(opts.reduceOp), 0)
                            for o, parts in zip(outputs, inputs)], outputs, asynchronous(opts))

    def alltoall(self, outputs, inputs, opts=None):
        if len({t.numel() for t in (*inputs, *outputs)}) != 1:
            raise RuntimeError('libnccl-mesh: an all_to_all of uneven sizes gives only this rank\'s row; torch_mesh.all_to_all takes the '
                               'whole count matrix every rank holds')
        return self._alltoall(inputs, outputs, outputs, opts, [inputs[0].numel()] * self._size ** 2)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        def pieces(t, sizes):
            return list(t.split(list(sizes) or [t.shape[0] // self._size] * self._size))
        if input_split_sizes or output_split_sizes:
            raise RuntimeError('libnccl-mesh: an all_to_all_single of uneven splits gives only this rank\'s row; torch_mesh.all_to_all '
                               'takes the whole count matrix every rank holds')
        sends = pieces(input, input_split_sizes)
        return self._alltoall(sends, pieces(output, output_split_sizes), output, opts, [sends[0].numel()] * self._size ** 2)

    def gather(self, outputs, inputs, opts=dist.GatherOptions()):
        receives = list(outputs[0]) if self._rank == opts.rootRank else [None] * self._size
        n, root = self._size, opts.rootRank
        matrix = [inputs[0].numel() if b == root else 0 for a in range(n) for b in range(n)]
        return self._alltoall(self._rooted(root, inputs[0]), receives, outputs, opts, matrix)

    def scatter(self, outputs, inputs, opts=dist.ScatterOptions()):
        sends = list(inputs[0]) if self._rank == opts.rootRank else [None] * self._size
        n, root = self._size, opts.rootRank
        matrix = [outputs[0].numel() if a == root else 0 for a in range(n) for b in range(n)]
        return self._alltoall(sends, self._rooted(root, outputs[0]), outputs, opts, matrix)

    def send(self, tensors, dst, tag):
        return self._issue([(SEND, [t], [], [], 0, dst) for t in tensors], tensors, True)

    def recv(self, tensors, src, tag):
        return self._issue([(RECV, [], [t], [], 0, src) for t in tensors], tensors, True)

    def all_to_all_counted(self, out, input, counts, sends, receives, host, landed):
        return self._issue([(COUNTED, [input, counts], [out, landed, host], [*sends, *receives], 0, 0)], [out, landed], True,
                           registered=out)

    def merge_partials(self, out, partials, rows=None, root=-1, async_op=False):
        width = partials.shape[-1]
        if width not in self._operators:
            made = ffi.new('ncclRedOp_t *')
            check(LIB.ncclMeshRedOpCreatePartials(made, width - 2, LIB.ncclFloat32, self.comm))
            self._operators[width] = made[0]
        merge = self._operators[width]
        call = ((REDUCE_SCATTER, [partials], [out], [r * width for r in rows], merge, 0) if rows is not None
                else (ALLREDUCE, [partials], [out], [], merge, 0) if root < 0 else (REDUCE, [partials], [out], [], merge, root))
        return self._issue([call], [out], async_op)


def partial(out, lse):
    seen = torch.isfinite(lse)[..., None]
    return torch.cat([torch.where(seen, lse[..., None], 0), seen.float(), torch.where(seen, out, 0).float()], -1)


def attention(merged):
    m, l, o = merged[..., 0], merged[..., 1], merged[..., 2:]
    seen = l > 0
    l = torch.where(seen, l, 1)
    return torch.where(seen[..., None], o / l[..., None], 0), torch.where(seen, m + l.log(), -math.inf)


def _mesh(group_name, what):
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if not isinstance(group, ProcessGroupMesh):
        raise TypeError(f'{what} runs on a group of the mesh backend, not {group_name}\'s')
    return group


@torch.library.custom_op('torch_mesh::all_to_all_counted', mutates_args=('host',))
def all_to_all_counted(input: torch.Tensor, counts: torch.Tensor, send_segments: list[int], recv_segments: list[int], capacity: int,
                       host: torch.Tensor, group_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    group = _mesh(group_name, 'all_to_all_counted')
    out = torch.empty((capacity,) + tuple(input.shape[1:]), dtype=input.dtype, device=input.device)
    landed = torch.empty(sum(recv_segments), dtype=torch.int64, device=input.device)
    group.all_to_all_counted(out, input.contiguous(), counts.contiguous(), send_segments, recv_segments, host, landed)
    return out, landed


@all_to_all_counted.register_fake
def _(input, counts, send_segments, recv_segments, capacity, host, group_name):
    return input.new_empty((capacity,) + tuple(input.shape[1:])), counts.new_empty(sum(recv_segments))


@torch.library.custom_op('torch_mesh::merge_partials', mutates_args=())
def merge_partials(partials: torch.Tensor, root: int, group_name: str) -> torch.Tensor:
    out = blank(partials.shape, partials.dtype, partials.device)
    _mesh(group_name, 'merge_partials').merge_partials(out, partials, root=root)
    return out


@merge_partials.register_fake
def _(partials, root, group_name):
    return torch.empty_like(partials)


def counted(host, group):
    _stream.commit()
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
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'
