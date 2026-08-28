"""The mesh, from Python. The same three functions.

    m = Mesh(22e9)                 # map a share of this node
    a = m.slot(i)                  # a numpy view of slot i, zero copy
    m.write(0, n)                  # send slots [0,n) to the other node
    for buf, src in m.read(): ...  # slots that arrived, as numpy views
"""
import ctypes, os, numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmesh.dylib"))
_lib.mesh_open.restype = ctypes.c_void_p
_lib.mesh_open.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
                           ctypes.POINTER(ctypes.c_size_t)]
_lib.mesh_write.restype = ctypes.c_size_t
_lib.mesh_write.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.mesh_read.restype = ctypes.c_size_t
_lib.mesh_read.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]


class Mesh:
    def __init__(self, nbytes):
        stride, usable = ctypes.c_size_t(), ctypes.c_size_t()
        p = _lib.mesh_open(int(nbytes), ctypes.byref(stride), ctypes.byref(usable))
        if not p:
            raise RuntimeError("no bridge, or it is holding less than %.2f GB" % (nbytes / 1e9))
        self.base, self.stride, self.usable = p, stride.value, usable.value
        self.slots = int(nbytes) // self.usable

    def slot(self, i, dtype=np.float32, count=None):
        """A numpy view of one slot. No copy: this is mesh memory."""
        n = count if count is not None else self.usable // np.dtype(dtype).itemsize
        return np.ctypeslib.as_array(
            ctypes.cast(self.base + i * self.stride,
                        ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))), (n,))

    def block(self, first, n):
        """A (n, usable) uint8 view of n consecutive slots. Strided, so it is
        not contiguous, but assignment into it is one vectorised copy instead
        of a Python loop per slot."""
        raw = np.ctypeslib.as_array(
            ctypes.cast(self.base + first * self.stride, ctypes.POINTER(ctypes.c_uint8)),
            (n * self.stride,))
        return raw.reshape(n, self.stride)[:, :self.usable]

    def write(self, first, nslots, node):
        """Send slots [first, first+nslots). Returns slots taken; call again
        with the rest. Any number of bytes, no size at which it refuses."""
        took = _lib.mesh_write(ctypes.c_void_p(self.base + first * self.stride),
                               nslots * self.usable, node)
        return took // self.usable

    def read(self, dtype=np.float32):
        """Yield (view, from_node) for each slot that has arrived."""
        p, src = ctypes.c_void_p(), ctypes.c_int()
        while True:
            n = _lib.mesh_read(ctypes.byref(p), ctypes.byref(src))
            if not n:
                return
            yield (np.ctypeslib.as_array(
                ctypes.cast(p, ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))),
                (n // np.dtype(dtype).itemsize,)), src.value)
