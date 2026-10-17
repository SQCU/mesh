"""NumPy arrays on libnccl-mesh, spelled as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import (rdma/NCCL.md "NumPy"):

    from mesh_mpi import Comm, SUM, MAX
    comm = Comm()
    comm.Allreduce(local, total, op=SUM)

Comm() is the world: this process is rank RANK of WORLD_SIZE (the variables torchrun and the launchers set), its
communicator made by NCCL's bootstrap at NCCL_COMM_ID, else MASTER_ADDR:MASTER_PORT (ncclCommInitRank: the map is the
links among the ranks' nodes their bridges hold paired), its collectives the library's compiled defaults.  Every call is libnccl-mesh's host path: contiguous arrays at host pointers, a NULL stream, the call
complete when it returns.  Methods: Get_rank, Get_size, Allreduce, Reduce, Bcast, Allgather, Allgatherv,
Reduce_scatter_block, Reduce_scatter, Alltoall, Gather, Scatter, Send, Recv, Sendrecv (one group, so both directions
pair), Barrier, allreduce (a Python number), Free; IN_PLACE as a send buffer, None for a buffer a rank does not use
(Gather's off the root, Scatter's send off it).  Allgatherv and Reduce_scatter take a count a rank, as mpi4py spells
them (a receive buffer [array, counts], recvcounts): one program over unequal segments (ncclMeshAllGatherV,
ncclMeshReduceScatterV), every rank passing the same counts.  COMM_WORLD is Comm(), made at first use.  Subcommunicators: Split(color, key) (ncclCommSplit; UNDEFINED
for none), Shrink(exclude) (ncclCommShrink), Grow(size) and Comm.joining (ncclCommGrow), Revoke() (ncclCommRevoke), and
a Cartesian process grid, Create_cart(dims) with Get_coords, Get_cart_rank, Shift and Sub(remain_dims),
each sub-grid a Split of it (a 2-D mesh's rows and columns), whose members the cables need not join.  Not here:
nonblocking calls, Alltoallv, Gatherv and Scatterv, object (pickled) collectives; separately issued Send and Recv that do not pair
are reported (rdma/NCCL.md "Rules").  A message to a rank the
map does not link is forwarded by the nodes between (rdma/NCCL.md "Routes between unlinked ranks").  The host path works on any node whose bridge runs (a communicator, a lone rank's too, is a client of its node's bridge: rdma/NCCL.md "Communicators"); it is not the measured path (GPU tensors take the Metal
path through torch-mesh).  Memory: empty(shape, dtype) is an array in this node's bridge's window (ncclMemAlloc, the
role mpi4py's MPI.Alloc_mem plays), the memory a program hands the collectives (rdma/NCCL.md "Memory"); the module's
own temporaries are such arrays."""
import math
import os

import numpy as np

import mesh

ffi, lib = mesh.ffi, mesh.lib
SUM, PROD, MAX, MIN, AVG = lib.ncclSum, lib.ncclProd, lib.ncclMax, lib.ncclMin, lib.ncclAvg
IN_PLACE, UNDEFINED = object(), -32766
TYPES = {np.dtype(np.int8): lib.ncclInt8, np.dtype(np.uint8): lib.ncclUint8, np.dtype(np.int32): lib.ncclInt32,
         np.dtype(np.uint32): lib.ncclUint32, np.dtype(np.int64): lib.ncclInt64, np.dtype(np.uint64): lib.ncclUint64,
         np.dtype(np.float16): lib.ncclFloat16, np.dtype(np.float32): lib.ncclFloat32, np.dtype(np.float64): lib.ncclFloat64}
try:
    import ml_dtypes
    TYPES.update({np.dtype(ml_dtypes.bfloat16): lib.ncclBfloat16, np.dtype(ml_dtypes.float8_e4m3fn): lib.ncclFloat8e4m3,
                  np.dtype(ml_dtypes.float8_e5m2): lib.ncclFloat8e5m2})
except ImportError:
    pass


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(lib.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(lib.ncclGetLastError(ffi.NULL)).decode()}')


def empty(shape, dtype=np.float64):
    """An uninitialized C-contiguous array of `shape` and `dtype` in this node's bridge's window (mesh.lease), its pages
    returned when it and every view of it are gone."""
    dtype = np.dtype(dtype)
    count = math.prod(shape) if isinstance(shape, (tuple, list)) else int(shape)
    return np.frombuffer(mesh.lease(count * dtype.itemsize), dtype, count=count).reshape(shape)


def window(array):
    """`array` where it is C-contiguous, else its copy in the window (empty)."""
    if array.flags.c_contiguous:
        return array
    made = empty(array.shape, array.dtype)
    made[...] = array
    return made


def at(array):
    if array is None:
        return ffi.NULL
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise ValueError('a buffer is a C-contiguous numpy array')
    if array.dtype not in TYPES:
        raise ValueError(f'a buffer of {array.dtype}: the library moves {sorted(map(str, TYPES))}')
    return ffi.cast('void *', array.ctypes.data)


def segments(spec, size):
    """A v-collective's receive buffer as mpi4py gives it, [array, counts] or [array, (counts, displacements)]: the
    array and a count a rank, summing to its size, the displacements (where given) the counts' running sums."""
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError('a v-collective\'s buffer is [array, counts]')
    array, layout = spec
    counts, displacements = (layout if isinstance(layout, tuple) and len(layout) == 2 and not np.isscalar(layout[0]) else (layout, None))
    counts = [int(c) for c in counts]
    packed = [sum(counts[:r]) for r in range(len(counts))]
    if len(counts) != size or sum(counts) != array.size or (displacements is not None and [int(d) for d in displacements] != packed):
        raise ValueError(f'counts {counts}: one a rank of {size}, summing to the array\'s {array.size}, packed in rank order')
    return array, counts


class Comm:
    """The world (the module's docstring)."""

    def __init__(self):
        rank, size = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
        address = os.environ.get('NCCL_COMM_ID') or f"{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}"
        unique, made = ffi.new('ncclUniqueId *'), ffi.new('ncclComm_t *')
        ffi.memmove(unique.internal, address.encode(), len(address.encode()))
        check(lib.ncclCommInitRank(made, size, unique[0], rank))
        self.rank, self.size, self.comm = rank, size, made[0]

    @classmethod
    def _of(cls, handle, parent):
        made = cls.__new__(cls)
        count, rank = ffi.new('int *'), ffi.new('int *')
        check(lib.ncclCommCount(handle, count))
        check(lib.ncclCommUserRank(handle, rank))
        made.rank, made.size, made.comm = rank[0], count[0], handle
        return made

    def Split(self, color=0, key=0):
        """The ranks of `color`, ordered by `key` then rank; None for UNDEFINED (ncclCommSplit)."""
        made = ffi.new('ncclComm_t *')
        check(lib.ncclCommSplit(self.comm, -1 if color == UNDEFINED else color, key, made, ffi.NULL))
        return Comm._of(made[0], self) if made[0] != ffi.NULL else None

    def Shrink(self, exclude, abort=False):
        """This communicator without the ranks `exclude`, each kept rank's order kept (ncclCommShrink; `abort` revokes
        this one's groups in flight first, NCCL_SHRINK_ABORT)."""
        made, ranks = ffi.new('ncclComm_t *'), ffi.new('int[]', list(exclude) or [0])
        check(lib.ncclCommShrink(self.comm, ranks, len(exclude), made, ffi.NULL, 1 if abort else 0))
        return Comm._of(made[0], self)

    def Grow(self, size):
        """This communicator's ranks and `size` - its size more, which join with Comm.joining at the id its root
        gives (ncclCommGetUniqueId), returned on the root to hand to them (ncclCommGrow)."""
        made, unique = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if self.rank == 0:
            check(lib.ncclCommGetUniqueId(self.comm, unique))
        check(lib.ncclCommGrow(self.comm, size, unique, -1, made, ffi.NULL))
        return Comm._of(made[0], self), unique[0] if self.rank == 0 else None

    @classmethod
    def joining(cls, size, unique, rank):
        """A process joining a grown communicator of `size` ranks as `rank`, at the root's id `unique` (ncclCommGrow)."""
        made = ffi.new('ncclComm_t *')
        check(lib.ncclCommGrow(ffi.NULL, size, ffi.new('ncclUniqueId *', unique), rank, made, ffi.NULL))
        return cls._of(made[0], None)

    def Revoke(self):
        """This communicator's groups in flight end with an error and none is issued after it (ncclCommRevoke)."""
        check(lib.ncclCommRevoke(self.comm, 0))

    def Create_cart(self, dims, periods=None, reorder=False):
        """This communicator's ranks as a process grid of shape `dims` (row-major: the last dimension varies
        fastest); `periods` (default every dimension) where Shift wraps."""
        if math.prod(dims) != self.size:
            raise ValueError(f'a grid {list(dims)} of {math.prod(dims)} ranks over {self.size}')
        return Cartcomm(self, list(dims), list(periods) if periods is not None else [True] * len(dims))

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Allreduce(self, sendbuf, recvbuf, op=SUM):
        send = recvbuf if sendbuf is IN_PLACE else sendbuf
        check(lib.ncclAllReduce(at(send), at(recvbuf), recvbuf.size, TYPES[recvbuf.dtype], op, self.comm, ffi.NULL))

    def Reduce(self, sendbuf, recvbuf, op=SUM, root=0):
        send = recvbuf if sendbuf is IN_PLACE else sendbuf
        target = recvbuf if recvbuf is not None else empty(send.shape, send.dtype)
        check(lib.ncclReduce(at(send), at(target), send.size, TYPES[send.dtype], op, root, self.comm, ffi.NULL))

    def Bcast(self, buf, root=0):
        check(lib.ncclBroadcast(at(buf), at(buf), buf.size, TYPES[buf.dtype], root, self.comm, ffi.NULL))

    def Allgather(self, sendbuf, recvbuf):
        send = window(recvbuf[self.rank * (recvbuf.size // self.size):][:recvbuf.size // self.size] if sendbuf is IN_PLACE else sendbuf)
        check(lib.ncclAllGather(at(send), at(recvbuf), recvbuf.size // self.size, TYPES[recvbuf.dtype], self.comm, ffi.NULL))

    def Allgatherv(self, sendbuf, recvbuf):
        """Every rank's segment into recvbuf, given as [array, counts] (or [array, (counts, displacements)], the
        displacements the counts' running sums: segments packed in rank order), rank r's counts[r] elements
        (ncclMeshAllGatherV); IN_PLACE takes this rank's segment from the array."""
        array, counts = segments(recvbuf, self.size)
        before = sum(counts[:self.rank])
        send = window(array[before:before + counts[self.rank]] if sendbuf is IN_PLACE else sendbuf)
        if send.size != counts[self.rank]:
            raise ValueError(f'a send buffer of {send.size} elements: counts[{self.rank}] is {counts[self.rank]}')
        check(lib.ncclMeshAllGatherV(at(send), at(array), ffi.new('size_t[]', counts), TYPES[array.dtype], self.comm, ffi.NULL))

    def Reduce_scatter_block(self, sendbuf, recvbuf, op=SUM):
        check(lib.ncclReduceScatter(at(sendbuf), at(recvbuf), recvbuf.size, TYPES[recvbuf.dtype], op, self.comm, ffi.NULL))

    def Reduce_scatter(self, sendbuf, recvbuf, recvcounts=None, op=SUM):
        """sendbuf's segments reduced by op, rank r's recvcounts[r] elements left in its recvbuf (packed in rank order in
        sendbuf; ncclMeshReduceScatterV); recvcounts None: recvbuf's size each, Reduce_scatter_block."""
        if recvcounts is None:
            return self.Reduce_scatter_block(sendbuf, recvbuf, op)
        counts = [int(c) for c in recvcounts]
        if len(counts) != self.size or sum(counts) != sendbuf.size or recvbuf.size != counts[self.rank]:
            raise ValueError(f'recvcounts {counts}: one a rank of {self.size}, summing to the send buffer\'s {sendbuf.size}, '
                             f'this rank\'s its receive buffer\'s {recvbuf.size}')
        check(lib.ncclMeshReduceScatterV(at(sendbuf), at(recvbuf), ffi.new('size_t[]', counts), TYPES[recvbuf.dtype], op, self.comm,
                                         ffi.NULL))

    def Alltoall(self, sendbuf, recvbuf):
        check(lib.ncclAlltoAll(at(sendbuf), at(recvbuf), sendbuf.size // self.size, TYPES[sendbuf.dtype], self.comm, ffi.NULL))

    def Gather(self, sendbuf, recvbuf, root=0):
        count = (recvbuf.size // self.size) if sendbuf is IN_PLACE else sendbuf.size
        dtype = (recvbuf if sendbuf is IN_PLACE else sendbuf).dtype
        send = recvbuf[self.rank * count:][:count] if sendbuf is IN_PLACE else sendbuf
        check(lib.ncclGather(at(send), at(recvbuf if self.rank == root else None), count, TYPES[dtype], root, self.comm, ffi.NULL))

    def Scatter(self, sendbuf, recvbuf, root=0):
        count = (sendbuf.size // self.size) if recvbuf is IN_PLACE else recvbuf.size
        dtype = (sendbuf if recvbuf is IN_PLACE else recvbuf).dtype
        target = sendbuf[self.rank * count:][:count] if recvbuf is IN_PLACE else recvbuf
        check(lib.ncclScatter(at(sendbuf if self.rank == root else None), at(target), count, TYPES[dtype], root, self.comm, ffi.NULL))

    def Sendrecv(self, sendbuf, dest, recvbuf, source):
        check(lib.ncclGroupStart())
        try:
            check(lib.ncclSend(at(sendbuf), sendbuf.size, TYPES[sendbuf.dtype], dest, self.comm, ffi.NULL))
            check(lib.ncclRecv(at(recvbuf), recvbuf.size, TYPES[recvbuf.dtype], source, self.comm, ffi.NULL))
        finally:
            check(lib.ncclGroupEnd())

    def Send(self, buf, dest):
        check(lib.ncclSend(at(buf), buf.size, TYPES[buf.dtype], dest, self.comm, ffi.NULL))

    def Recv(self, buf, source):
        check(lib.ncclRecv(at(buf), buf.size, TYPES[buf.dtype], source, self.comm, ffi.NULL))

    def Barrier(self):
        box = empty(1, np.int32)
        box[0] = 0
        self.Allreduce(IN_PLACE, box)

    def allreduce(self, value, op=SUM):
        box = empty(1, value.dtype if isinstance(value, np.generic) else np.float64 if isinstance(value, float) else np.int64)
        box[0] = value
        self.Allreduce(IN_PLACE, box, op)
        return box[0].item()

    def Free(self):
        check(lib.ncclCommDestroy(self.comm))


class Cartcomm(Comm):
    """A process grid over a communicator's ranks (Comm.Create_cart)."""

    def __init__(self, base, dims, periods):
        self.rank, self.size, self.comm = base.rank, base.size, base.comm
        self.dims, self.periods = dims, periods

    def Get_coords(self, rank):
        coords = []
        for d in reversed(self.dims):
            coords.append(rank % d)
            rank //= d
        return coords[::-1]

    def Get_cart_rank(self, coords):
        rank = 0
        for c, d in zip(coords, self.dims):
            rank = rank * d + c
        return rank

    def Shift(self, direction, disp):
        """(source, dest): the ranks `disp` back and on along `direction`, None past a non-periodic edge."""
        coords, d = self.Get_coords(self.rank), self.dims[direction]

        def at(offset):
            c = coords[direction] + offset
            if not self.periods[direction] and not 0 <= c < d:
                return None
            moved = list(coords)
            moved[direction] = c % d
            return self.Get_cart_rank(moved)
        return at(-disp), at(disp)

    def Sub(self, remain_dims):
        """The sub-grid through this rank that keeps the dimensions where `remain_dims` is true (a row of a 2-D
        grid: [False, True]); a Split of this grid by the coordinates it drops."""
        coords = self.Get_coords(self.rank)
        color = self.Get_cart_rank([0 if keep else c for keep, c in zip(remain_dims, coords)])
        sub = self.Split(color, self.rank)
        return Cartcomm(sub, [d for keep, d in zip(remain_dims, self.dims) if keep],
                        [p for keep, p in zip(remain_dims, self.periods) if keep])


_world = None


def __getattr__(name):
    global _world
    if name == 'COMM_WORLD':
        _world = _world or Comm()
        return _world
    raise AttributeError(name)
