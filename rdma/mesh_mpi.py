"""NumPy arrays on libnccl-mesh, spelled as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import (rdma/NCCL.md "NumPy"):

    from mesh_mpi import Comm, SUM, MAX
    comm = Comm()
    comm.Allreduce(local, total, op=SUM)

Comm() is the world: this process is rank RANK of WORLD_SIZE (the variables torchrun and the launchers set), its
communicator made by NCCL's bootstrap at NCCL_COMM_ID, else MASTER_ADDR:MASTER_PORT (ncclCommInitRank: the map is the
links among the ranks' nodes their bridges hold paired), its collectives the library's compiled defaults.  Every call is libnccl-mesh's host path: contiguous arrays at host pointers, a NULL stream, the call
complete when it returns.  Methods: Get_rank, Get_size, Allreduce, Reduce, Bcast, Allgather, Allgatherv,
Reduce_scatter_block, Reduce_scatter, Alltoall, Gather, Scatter, Barrier, allreduce (a Python number), Free;
IN_PLACE as a send buffer, None for a buffer a rank does not use
(Gather's off the root, Scatter's send off it); an array of a dtype the library has no type of moves as its bytes, and
a reduction takes the library's types alone.  Allgatherv and Reduce_scatter take a count a rank, as mpi4py spells
them (a receive buffer [array, counts], recvcounts): one program over the segments (ncclMeshAllGatherV,
ncclMeshReduceScatterV), every rank passing the same counts; Allgather and Reduce_scatter_block are the same calls with
equal counts (the library takes equal segments as its equal program).  COMM_WORLD is Comm(), made at first use.  Subcommunicators: Split(color, key) (ncclCommSplit; UNDEFINED
for none), Shrink(exclude) (ncclCommShrink), Grow(size) and Comm.joining (ncclCommGrow), Revoke() (ncclCommRevoke), and
a Cartesian process grid, Create_cart(dims) with Get_coords, Get_cart_rank, Shift and Sub(remain_dims),
each sub-grid a Split of it (a 2-D mesh's rows and columns), whose members the cables need not join.  Not here:
nonblocking calls, Alltoallv, Gatherv and Scatterv, object (pickled) collectives, or paired Send/Recv.
Directed transfers use the prepared WRITE interface below, with every physical rank admitted for forwarding.
The host path works on any node whose bridge runs (a communicator, a lone rank's too, is a client of its node's bridge: rdma/NCCL.md "Communicators"); it is not the measured path (GPU tensors take the Metal
path through torch-mesh).  Memory: empty(shape, dtype) is an array in this node's bridge's region (ncclMemAlloc, the
role mpi4py's MPI.Alloc_mem plays), the memory a program hands the collectives (rdma/NCCL.md "Memory"); the module's
own temporaries are such arrays. Write_prepare, Write_rows_prepare, Write_spans_prepare and Write_view_prepare expose
the same native AOT exact-byte WRITE bindings as other front ends; their WritePlan.Execute reuses the fixed bindings
without exchanging layout metadata. Explicit spans and metadata-derived ndarray views preserve arbitrary byte
strides and original array owners without packing values. This adds no alternative collective planner."""
import math
import operator
import os
import weakref

import numpy as np

import mesh

ffi, lib, check = mesh.ffi, mesh.lib, mesh.check
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


def empty(shape, dtype=np.float64):
    """An uninitialized C-contiguous array of `shape` and `dtype` in this node's bridge's region (mesh.lease), its pages
    logically freed when it and every view/plan retaining it are gone; native future and posted DMA obligations defer reuse."""
    dtype = np.dtype(dtype)
    count = math.prod(shape) if isinstance(shape, (tuple, list)) else int(shape)
    return np.frombuffer(mesh.lease(count * dtype.itemsize), dtype, count=count).reshape(shape)


def to_region(array):
    """`array` where it is C-contiguous, else its copy in the region (empty)."""
    if array.flags.c_contiguous:
        return array
    made = empty(array.shape, array.dtype)
    made[...] = array
    return made


def at(array):
    """A buffer's address: a C-contiguous array's own (another would cross only through a copy: rdma/NCCL.md "Memory")."""
    if array is None:
        return ffi.NULL
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise ValueError('a buffer is a C-contiguous numpy array')
    return ffi.cast('void *', array.ctypes.data)


def kind(dtype):
    """The library's type a call that moves `dtype` takes and the count of it in one element: (its type, 1), or, a dtype
    the library has no type of, (ncclUint8, its itemsize), moved as its bytes.  A reduction takes TYPES alone."""
    dtype = np.dtype(dtype)
    return (TYPES[dtype], 1) if dtype in TYPES else (lib.ncclUint8, dtype.itemsize)


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


def _row_pitch(array, row_bytes, rows):
    """Prove the array's logical C-order bytes equal one positive affine row map.

    This only reads shape/strides, never array contents. General indexed views
    belong in the shared native view compiler; this binding does not pack them.
    """
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ValueError('a WRITE binding is a numpy array without Python objects')
    if row_bytes <= 0 or rows <= 0 or row_bytes * rows != array.nbytes or row_bytes % array.itemsize:
        raise ValueError('row_bytes * rows must cover exactly the array, in whole dtype elements')
    axes = [(int(n), int(stride)) for n, stride in zip(array.shape, array.strides) if n > 1]
    contiguous = array.itemsize
    while axes and axes[-1][1] == contiguous:
        n, _ = axes.pop()
        contiguous *= n
    if not axes:
        return row_bytes
    if contiguous != row_bytes:
        raise ValueError('the requested rows are not contiguous intervals of this view')
    pitch, extent = axes[-1][1], 1
    if pitch < row_bytes:
        raise ValueError('affine WRITE rows must have positive, nonoverlapping pitch')
    for n, stride in reversed(axes):
        if stride != pitch * extent:
            raise ValueError('this view needs the shared indexed descriptor API; it is not one affine row map')
        extent *= n
    return pitch


class WritePlan:
    """Fixed native WRITE bindings; preparation exchanges layout metadata once.

    Execute is the synchronous host execution of that native plan. Original
    arrays and their owners remain retained until Free. Source mutation requires
    every scheduled DMA read to retire; a source-only Execute return alone does
    not prove it. Destination reuse additionally requires its own consumers to
    finish. Bindings and shape remain fixed. Every
    participating rank prepares and executes the same write sequence, including
    relay ranks with landing storage. This class contains no collective planner.
    """
    def __init__(self, comm, handle, arrays):
        self.comm, self.handle, self.arrays = comm, handle, arrays
        if not hasattr(comm, '_writes'):
            comm._writes = weakref.WeakSet()
        comm._writes.add(self)

    def Execute(self):
        if self.handle == ffi.NULL:
            raise ValueError('WRITE plan is already freed')
        check(lib.ncclMeshWriteExecute(self.handle, ffi.NULL))

    __call__ = Execute

    def Free(self):
        if self.handle != ffi.NULL:
            check(lib.ncclMeshWritePlanDestroy(self.handle))
            self.handle, self.arrays = ffi.NULL, ()
            self.comm._writes.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.Free()

    def __del__(self):
        try:
            self.Free()
        except Exception:
            pass


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
        this one's groups in flight first, NCCL_SHRINK_ABORT); None on a rank among those excluded."""
        made, ranks = ffi.new('ncclComm_t *'), ffi.new('int[]', list(exclude) or [0])
        check(lib.ncclCommShrink(self.comm, ranks, len(exclude), made, ffi.NULL, 1 if abort else 0))
        return Comm._of(made[0], self) if made[0] != ffi.NULL else None

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
        fastest); `periods` (default every dimension) where Shift wraps.  A grid of fewer ranks than the communicator
        holds its first ones (a Split of them), the others None, as MPI_Cart_create gives them MPI_COMM_NULL; one of
        more has no rank for a coordinate."""
        cells = math.prod(dims)
        if cells > self.size:
            raise ValueError(f'a grid {list(dims)} of {cells} ranks over {self.size}')
        base = self if cells == self.size else self.Split(0 if self.rank < cells else UNDEFINED, self.rank)
        return Cartcomm(base, list(dims), list(periods) if periods is not None else [True] * len(dims)) if base is not None else None

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Plan(self, calls):
        """Declare the complete function's native calls through the shared controller.

        Calls are dictionaries of ncclMeshCall fields, with ordinary count lists
        where required. This is configuration metadata, not a Python schedule or
        a collective decomposition. Every rank declares the same function.
        """
        from controller import declare
        return declare(self.comm, calls)

    def PlanDemands(self):
        """Exact compiler-reported directed bytes for the declared complete function."""
        from controller import function_demands
        return function_demands(self.comm)

    def Write_prepare(self, sendbuf, landingbuf, source, destination, *, max_packet=0):
        """Bind one exact contiguous native WRITE, without copying or routing in Python.

        Only source's sendbuf and other ranks' landingbuf are used. All ranks
        provide the same element count/type/endpoints; unused buffers may be
        None. Every used array must belong to a bridge lease (empty or a view
        thereof); native preparation validates actual registration and bounds.
        """
        own = sendbuf if self.rank == source else landingbuf
        if not isinstance(own, np.ndarray) or own.dtype.hasobject:
            raise ValueError('a WRITE binding is a numpy array without Python objects')
        pointer, (datatype, multiple) = at(own), kind(own.dtype)
        made = ffi.new('ncclMeshWritePlan_t *')
        check(lib.ncclMeshWritePrepare(pointer if self.rank == source else ffi.NULL,
            ffi.NULL if self.rank == source else pointer, own.size * multiple, datatype,
            source, destination, self.comm, 0, 1, 0, max_packet, made))
        return WritePlan(self, made[0], (own,))

    def Write_rows_prepare(self, sendbuf, landingbuf, source, destination, *, row_bytes, rows, max_packet=0):
        """Bind the same native WRITE as exact rows with local pitches.

        row_bytes and rows agree across ranks; source, destination and relay
        pitches may differ. The original NumPy view must describe exactly this
        positive affine row map. No padding, staging allocation or per-row
        execution call is introduced. General non-affine views need the shared
        indexed native descriptor, not a private packing implementation here.
        """
        own = sendbuf if self.rank == source else landingbuf
        row_bytes, rows = int(row_bytes), int(rows)
        pitch = _row_pitch(own, row_bytes, rows)
        pointer, made = ffi.cast('void *', own.ctypes.data), ffi.new('ncclMeshWritePlan_t *')
        check(lib.ncclMeshWriteRowsPrepare(pointer if self.rank == source else ffi.NULL,
            ffi.NULL if self.rank == source else pointer, row_bytes, rows, pitch,
            source, destination, self.comm, 0, max_packet, made))
        return WritePlan(self, made[0], (own,))

    def _write_spans_prepare(self, own, spans, source, destination, max_packet, base_shift=0):
        if not isinstance(own, np.ndarray) or own.dtype.hasobject:
            raise ValueError('a WRITE binding is a numpy array without Python objects')
        spans = [(operator.index(offset), operator.index(length)) for offset, length in spans]
        metadata = ffi.new('ncclMeshSpan[]', spans) if spans else ffi.NULL
        pointer = ffi.cast('void *', own.ctypes.data + base_shift)
        made = ffi.new('ncclMeshWritePlan_t *')
        check(lib.ncclMeshWriteSpansPrepare(pointer if self.rank == source else ffi.NULL,
            ffi.NULL if self.rank == source else pointer, metadata, len(spans), source, destination,
            self.comm, 0, max_packet, made))
        return WritePlan(self, made[0], (own,))

    def Write_spans_prepare(self, sendbuf, landingbuf, spans, source, destination, *, max_packet=0):
        """Bind logical bytes to explicit (offset, length) ranges in one lease.

        Offsets are relative to the used local array's first element address.
        Every rank may supply a different ordered list, but logical byte totals
        agree. Native preparation validates lease bounds and landing overlap.
        Source ranges may repeat. This copies metadata only, never values.
        """
        own = sendbuf if self.rank == source else landingbuf
        return self._write_spans_prepare(own, spans, source, destination, max_packet)

    def Write_view_prepare(self, sendbuf, landingbuf, source, destination, *, max_packet=0):
        """Bind an arbitrary ndarray view in logical row-major element order.

        The shared metadata mapper handles transpose, gaps, reversed strides
        and broadcast source views. The native plan rejects overlapping landing
        views and spans outside one live lease. The original view retains its
        storage; no contiguous temporary or numerical copy is constructed.
        """
        from views import byte_spans
        own = sendbuf if self.rank == source else landingbuf
        if not isinstance(own, np.ndarray) or own.dtype.hasobject:
            raise ValueError('a WRITE binding is a numpy array without Python objects')
        shift, spans = byte_spans(own.shape, own.strides, own.dtype.itemsize)
        return self._write_spans_prepare(own, spans, source, destination, max_packet, shift)

    def Allreduce(self, sendbuf, recvbuf, op=SUM):
        send = recvbuf if sendbuf is IN_PLACE else sendbuf
        check(lib.ncclAllReduce(at(send), at(recvbuf), recvbuf.size, TYPES[recvbuf.dtype], op, self.comm, ffi.NULL))

    def Reduce(self, sendbuf, recvbuf, op=SUM, root=0):
        send = recvbuf if sendbuf is IN_PLACE else sendbuf
        target = recvbuf if recvbuf is not None else empty(send.shape, send.dtype)
        check(lib.ncclReduce(at(send), at(target), send.size, TYPES[send.dtype], op, root, self.comm, ffi.NULL))

    def Bcast(self, buf, root=0):
        t, k = kind(buf.dtype)
        check(lib.ncclBroadcast(at(buf), at(buf), buf.size * k, t, root, self.comm, ffi.NULL))

    def Allgather(self, sendbuf, recvbuf):
        self.Allgatherv(sendbuf, [recvbuf, [recvbuf.size // self.size] * self.size])

    def Allgatherv(self, sendbuf, recvbuf):
        """Every rank's segment into recvbuf, given as [array, counts] (or [array, (counts, displacements)], the
        displacements the counts' running sums: segments packed in rank order), rank r's counts[r] elements
        (ncclMeshAllGatherV); IN_PLACE takes this rank's segment from the array."""
        array, counts = segments(recvbuf, self.size)
        before = sum(counts[:self.rank])
        send = to_region(array[before:before + counts[self.rank]] if sendbuf is IN_PLACE else sendbuf)
        if send.size != counts[self.rank]:
            raise ValueError(f'a send buffer of {send.size} elements: counts[{self.rank}] is {counts[self.rank]}')
        t, k = kind(array.dtype)
        check(lib.ncclMeshAllGatherV(at(send), at(array), ffi.new('size_t[]', [c * k for c in counts]), t, self.comm, ffi.NULL))

    def Reduce_scatter_block(self, sendbuf, recvbuf, op=SUM):
        self.Reduce_scatter(sendbuf, recvbuf, None, op)

    def Reduce_scatter(self, sendbuf, recvbuf, recvcounts=None, op=SUM):
        """sendbuf's segments reduced by op, rank r's recvcounts[r] elements left in its recvbuf (packed in rank order in
        sendbuf; ncclMeshReduceScatterV); recvcounts None: recvbuf's size each (Reduce_scatter_block)."""
        counts = [int(c) for c in recvcounts] if recvcounts is not None else [recvbuf.size] * self.size
        if len(counts) != self.size or sum(counts) != sendbuf.size or recvbuf.size != counts[self.rank]:
            raise ValueError(f'recvcounts {counts}: one a rank of {self.size}, summing to the send buffer\'s {sendbuf.size}, '
                             f'this rank\'s its receive buffer\'s {recvbuf.size}')
        check(lib.ncclMeshReduceScatterV(at(sendbuf), at(recvbuf), ffi.new('size_t[]', counts), TYPES[recvbuf.dtype], op, self.comm,
                                         ffi.NULL))

    def Alltoall(self, sendbuf, recvbuf):
        if sendbuf.size % self.size or recvbuf.size != sendbuf.size or recvbuf.dtype != sendbuf.dtype:
            raise ValueError('Alltoall requires equal typed send/receive extents, divisible by the communicator size')
        t, k = kind(sendbuf.dtype)
        check(lib.ncclAlltoAll(at(sendbuf), at(recvbuf), sendbuf.size // self.size * k, t, self.comm, ffi.NULL))

    def Gather(self, sendbuf, recvbuf, root=0):
        count = (recvbuf.size // self.size) if sendbuf is IN_PLACE else sendbuf.size
        dtype = (recvbuf if sendbuf is IN_PLACE else sendbuf).dtype
        send = recvbuf[self.rank * count:][:count] if sendbuf is IN_PLACE else sendbuf
        t, k = kind(dtype)
        check(lib.ncclGather(at(send), at(recvbuf if self.rank == root else None), count * k, t, root, self.comm, ffi.NULL))

    def Scatter(self, sendbuf, recvbuf, root=0):
        count = (sendbuf.size // self.size) if recvbuf is IN_PLACE else recvbuf.size
        dtype = (sendbuf if recvbuf is IN_PLACE else recvbuf).dtype
        target = sendbuf[self.rank * count:][:count] if recvbuf is IN_PLACE else recvbuf
        t, k = kind(dtype)
        check(lib.ncclScatter(at(sendbuf if self.rank == root else None), at(target), count * k, t, root, self.comm, ffi.NULL))

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
        for plan in tuple(getattr(self, '_writes', ())):
            plan.Free()
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
