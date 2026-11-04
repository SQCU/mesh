import ctypes as C
import os

U, Q, Z = C.c_uint32, C.c_uint64, C.c_size_t


# mesh-dataflow.h struct mesh_ctx: the region mapped and the client word; no claim (a plan's claims are mesh_plan's)
class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), ('client', Q), ('fd', C.c_int)]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


# mesh-collective.h mesh_host_arrived's error value: a phase ahead, or a range cancelled
FAILED = 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_transfers_start', C.c_int, [CONTEXT, Q]),
        ('mesh_host_publish', None, [CONTEXT, Section, Q]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, Q])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))
