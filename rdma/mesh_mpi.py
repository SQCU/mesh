"""NumPy arrays on libnccl-mesh, spelled as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import (rdma/NCCL.md "NumPy"):

    from mesh_mpi import Comm, SUM, MAX
    comm = Comm()
    comm.Allreduce(local, total, op=SUM)

Comm() is the world: this process's rank is its node in the link map MESH_LINK_MAP names (its bridge, at MESH_REGION,
reports which: one rank a node), the communicator over that map (ncclMeshCommInitRank), its collectives the library's
compiled defaults.  Every call is libnccl-mesh's host path: contiguous arrays at host pointers, a NULL stream, the call
complete when it returns.  Methods: Get_rank, Get_size, Allreduce, Reduce, Bcast, Allgather, Reduce_scatter_block,
Alltoall, Send, Recv, Barrier, allreduce (a Python number), Free; IN_PLACE as a send buffer.  A message to a rank the
map does not link takes routes through intermediates, and is refused where the bridges' queue pairs do not carry them
(rdma/NCCL.md "Routes").  The host path works on any node; it is not the measured path (GPU tensors take the Metal
path through torch-mesh)."""
import os

import numpy as np

import mesh

ffi, lib = mesh.ffi, mesh.lib
SUM, PROD, MAX, MIN, AVG = lib.ncclSum, lib.ncclProd, lib.ncclMax, lib.ncclMin, lib.ncclAvg
IN_PLACE = object()
TYPES = {np.dtype(np.int8): lib.ncclInt8, np.dtype(np.uint8): lib.ncclUint8, np.dtype(np.int32): lib.ncclInt32,
         np.dtype(np.uint32): lib.ncclUint32, np.dtype(np.int64): lib.ncclInt64, np.dtype(np.uint64): lib.ncclUint64,
         np.dtype(np.float16): lib.ncclFloat16, np.dtype(np.float32): lib.ncclFloat32, np.dtype(np.float64): lib.ncclFloat64}


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(lib.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(lib.ncclGetLastError(ffi.NULL)).decode()}')


def at(array):
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
