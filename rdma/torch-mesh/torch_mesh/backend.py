"""torch.distributed's "mesh" backend: a process group is a libnccl-mesh communicator (nccl.h), made over the store
torch gives it, its handle attached for the compiler (_stream.attach, compiler.Template's declarations). It issues no
collective: a stepper declares its collectives and runs as a compiled program (torch_mesh/compiler.py)."""
import torch
import torch.distributed as dist

from . import _stream

import mesh  # noqa: E402  (rdma/mesh.py: the cffi binding of nccl.h and the planner's headers)

ffi, LIB, check = mesh.ffi, mesh.lib, mesh.check
TYPES = {torch.int8: LIB.ncclInt8, torch.uint8: LIB.ncclUint8, torch.bool: LIB.ncclUint8, torch.int32: LIB.ncclInt32,
         torch.uint32: LIB.ncclUint32, torch.int64: LIB.ncclInt64, torch.uint64: LIB.ncclUint64, torch.float16: LIB.ncclFloat16,
         torch.float32: LIB.ncclFloat32, torch.float64: LIB.ncclFloat64, torch.bfloat16: LIB.ncclBfloat16,
         torch.float8_e4m3fn: LIB.ncclFloat8e4m3, torch.float8_e5m2: LIB.ncclFloat8e5m2}


class ProcessGroupMesh(dist.ProcessGroup):
    def __init__(self, rank, size, store):
        super().__init__(rank, size)
        made, unique = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if rank == 0:
            check(LIB.ncclGetUniqueId(unique))
            store.set('mesh/unique', bytes(ffi.buffer(unique)))
        else:
            ffi.memmove(unique, store.get('mesh/unique'), ffi.sizeof('ncclUniqueId'))
        check(LIB.ncclCommInitRank(made, size, unique[0], rank))
        self._rank, self._size, self.comm = rank, size, made[0]
        self.handle = int(ffi.cast('uintptr_t', self.comm))
        _stream.attach(self, self.handle)

    def _get_backend(self, device):
        """the group itself, its one backend on every device it runs"""
        return self

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    def shutdown(self):
        if self.comm is not None:
            check(LIB.ncclCommDestroy(self.comm))
            self.comm = None


def create(opts, options):
    """the group (torch's extended creator: `opts` its store, rank and size)"""
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'
