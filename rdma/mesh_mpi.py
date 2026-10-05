"""NumPy arrays on libnccl-mesh, spelled as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import (rdma/NCCL.md "NumPy"):

    from mesh_mpi import Comm, SUM, MAX
    comm = Comm()
    comm.Allreduce(local, total, op=SUM)

Comm() is the world: this process's rank is its node in the link map MESH_LINK_MAP names (its bridge, at MESH_REGION,
reports which: one rank a node), the communicator over that map (ncclMeshCommInitRank), its collectives the library's
compiled defaults.  Every call is libnccl-mesh's host path: contiguous arrays at host pointers, a NULL stream, the call
complete when it returns.  Methods: Get_rank, Get_size, Allreduce, Reduce, Bcast, Allgather, Reduce_scatter_block,
Alltoall, Gather, Scatter, Send, Recv, Sendrecv (one group, so both directions pair), Barrier, allreduce (a Python
number), Free; IN_PLACE as a send buffer, None for a buffer a rank does not use (Gather's off the root, Scatter's send
off it).  COMM_WORLD is Comm(), made at first use.  Subcommunicators: Split(color, key) (ncclCommSplit; UNDEFINED
for none), and a Cartesian process grid, Create_cart(dims) with Get_coords, Get_cart_rank, Shift and Sub(remain_dims),
each sub-grid a Split of it (a 2-D mesh's rows and columns), whose members the cables need not join.  Not here:
nonblocking calls, the v-collectives, object (pickled) collectives; separately issued Send and Recv that do not pair
are reported (rdma/NCCL.md "Rules").  A message to a rank the
map does not link is forwarded by the nodes between (rdma/NCCL.md "Routes between unlinked ranks").  The host path works on any node; it is not the measured path (GPU tensors take the Metal
path through torch-mesh)."""
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


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(lib.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(lib.ncclGetLastError(ffi.NULL)).decode()}')


def at(array):
    if array is None:
        return ffi.NULL
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise ValueError('a buffer is a C-contiguous numpy array')
    if array.dtype not in TYPES:
        raise ValueError(f'a buffer of {array.dtype}: the library moves {sorted(map(str, TYPES))}')
    return ffi.cast('void *', array.ctypes.data)


class Comm:
    """The world over the link map `links` (default MESH_LINK_MAP), this rank's bridge at `region` (default
    MESH_REGION, else /mesh0)."""

    def __init__(self, links=None, region=None):
        region = region or os.environ.get('MESH_REGION') or '/mesh0'
        path = links or os.environ.get('MESH_LINK_MAP')
        if not path:
            raise RuntimeError('no link map: MESH_LINK_MAP names it (metal-microbench tools/mesh/grid.py run sets it)')
        topology = mesh.read_link_map(path)
        views, node = ffi.new('struct mesh_link_view[]', 64), ffi.new('uint32_t *')
        if lib.mesh_observe(region.encode(), views, 64, node) < 0:
            raise RuntimeError(f'no bridge at region {region}')
        made = ffi.new('ncclComm_t *')
        check(lib.ncclMeshCommInitRank(made, node[0], topology.c, ffi.NULL, ffi.NULL, 0, ffi.NULL, region.encode()))
        self.rank, self.size, self.comm, self.topology = int(node[0]), topology.nodes, made[0], topology

    @classmethod
    def _of(cls, handle, parent):
        made = cls.__new__(cls)
        count, rank = ffi.new('int *'), ffi.new('int *')
        check(lib.ncclCommCount(handle, count))
        check(lib.ncclCommUserRank(handle, rank))
        made.rank, made.size, made.comm, made.topology = rank[0], count[0], handle, parent.topology
        return made

    def Split(self, color=0, key=0):
        """The ranks of `color`, ordered by `key` then rank; None for UNDEFINED (ncclCommSplit)."""
        made = ffi.new('ncclComm_t *')
        check(lib.ncclCommSplit(self.comm, -1 if color == UNDEFINED else color, key, made, ffi.NULL))
        return Comm._of(made[0], self) if made[0] != ffi.NULL else None

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
        target = recvbuf if recvbuf is not None else np.empty_like(send)
        check(lib.ncclReduce(at(send), at(target), send.size, TYPES[send.dtype], op, root, self.comm, ffi.NULL))

    def Bcast(self, buf, root=0):
        check(lib.ncclBroadcast(at(buf), at(buf), buf.size, TYPES[buf.dtype], root, self.comm, ffi.NULL))

    def Allgather(self, sendbuf, recvbuf):
        send = recvbuf[self.rank * (recvbuf.size // self.size):][:recvbuf.size // self.size] if sendbuf is IN_PLACE else sendbuf
        check(lib.ncclAllGather(at(np.ascontiguousarray(send)), at(recvbuf), recvbuf.size // self.size, TYPES[recvbuf.dtype],
                                self.comm, ffi.NULL))

    def Reduce_scatter_block(self, sendbuf, recvbuf, op=SUM):
        check(lib.ncclReduceScatter(at(sendbuf), at(recvbuf), recvbuf.size, TYPES[recvbuf.dtype], op, self.comm, ffi.NULL))

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
        self.Allreduce(IN_PLACE, np.zeros(1, np.int32))

    def allreduce(self, value, op=SUM):
        box = np.array([value], value.dtype if isinstance(value, np.generic) else np.float64 if isinstance(value, float) else np.int64)
        self.Allreduce(IN_PLACE, box, op)
        return box[0].item()

    def Free(self):
        check(lib.ncclCommDestroy(self.comm))


class Cartcomm(Comm):
    """A process grid over a communicator's ranks (Comm.Create_cart)."""

    def __init__(self, base, dims, periods):
        self.rank, self.size, self.comm, self.topology = base.rank, base.size, base.comm, base.topology
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
