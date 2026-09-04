import ctypes, os, numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmesh.dylib"))
_lib.mesh_open.restype = ctypes.c_void_p
_lib.mesh_open.argtypes = [ctypes.POINTER(ctypes.c_size_t)]*3
_lib.mesh_write.restype = ctypes.c_size_t
_lib.mesh_write.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.mesh_write_copy.restype = ctypes.c_size_t
_lib.mesh_write_copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_int]
_lib.mesh_queue_copy.restype = ctypes.c_size_t
_lib.mesh_queue_copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_int]
_lib.mesh_pump.restype = ctypes.c_size_t
_lib.mesh_queued.restype = ctypes.c_size_t
_lib.mesh_inflight.restype = ctypes.c_size_t
_lib.mesh_read.restype = ctypes.c_size_t
_lib.mesh_read.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]
_lib.mesh_readv.restype = ctypes.c_size_t
_lib.mesh_readv.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int),
                            ctypes.c_size_t]

READ_DEPTH = 1024

class Mesh:
    def __init__(self, nbytes=None):
        ns, stride, usable = (ctypes.c_size_t() for _ in range(3))
        p = _lib.mesh_open(ctypes.byref(ns), ctypes.byref(stride), ctypes.byref(usable))
        if not p:
            raise RuntimeError("no bridge")
        self.region = os.environ.get("MESH_REGION", "/mesh0")
        self.base, self.stride, self.usable = p, stride.value, usable.value
        self.slots = ns.value if nbytes is None else min(ns.value, int(nbytes) // usable.value)
        self._read_data = np.empty((READ_DEPTH, self.usable), np.uint8)
        self._read_sizes = np.empty(READ_DEPTH, np.uint32)
        self._read_sources = np.empty(READ_DEPTH, np.int32)

    def _view(self, first, n, dtype):
        return np.ctypeslib.as_array(
            ctypes.cast(self.base + first * self.stride,
                        ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))), (n,))

    def slot(self, i, dtype=np.float32, count=None):
        return self._view(i, count or self.usable // np.dtype(dtype).itemsize, dtype)

    def block(self, first, n):
        return self._view(first, n * self.stride, np.uint8) \
                   .reshape(n, self.stride)[:, :self.usable]

    def write(self, first, nslots, node):
        took = _lib.mesh_write(ctypes.c_void_p(self.base + first * self.stride),
                               nslots * self.usable, node)
        return took // self.usable

    def send(self, frames, node):
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        if frames.ndim != 2 or frames.shape[1] > self.usable:
            raise ValueError("frames must be a two-dimensional byte array within mesh payload width")
        return _lib.mesh_write_copy(
            ctypes.c_void_p(frames.ctypes.data), frames.strides[0], frames.shape[1],
            frames.shape[0], node,
        )

    def queue(self, frames, node):
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        if frames.ndim != 2 or frames.shape[1] > self.usable:
            raise ValueError("frames must be a two-dimensional byte array within mesh payload width")
        return _lib.mesh_queue_copy(
            ctypes.c_void_p(frames.ctypes.data), frames.strides[0], frames.shape[1],
            frames.shape[0], node,
        )

    def pump(self):
        return _lib.mesh_pump()

    def queued(self):
        return _lib.mesh_queued()

    def inflight(self):
        return _lib.mesh_inflight()

    def read(self, dtype=np.float32):
        while True:
            count = _lib.mesh_readv(
                ctypes.c_void_p(self._read_data.ctypes.data), self.usable,
                self._read_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                self._read_sources.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                READ_DEPTH,
            )
            if not count:
                return
            itemsize = np.dtype(dtype).itemsize
            for i in range(count):
                n = int(self._read_sizes[i])
                yield np.frombuffer(self._read_data[i], dtype, n // itemsize), int(self._read_sources[i])
