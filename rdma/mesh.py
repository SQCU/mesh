import ctypes as C
import os

U, Q, Z = C.c_uint32, C.c_uint64, C.c_size_t


class Range(C.Structure):
    _fields_ = [('first', U), ('count', U)]


# mesh-dataflow.h struct mesh_ctx: the program's claim (`rows`, `window`, `local`) and its cursors
class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), ('client', Q), *((k, Range) for k in ('rows', 'window', 'local')),
                *((k, U) for k in ('row_cursor', 'local_cursor')), ('fd', C.c_int)]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


CANCELLED = 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_section_create', C.c_int, [CONTEXT, Z, U, C.c_int, C.POINTER(Section)]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_host_inputs', C.c_int, [CONTEXT]),
        ('mesh_transfers_start', C.c_int, [CONTEXT]),
        ('mesh_host_publish', None, [CONTEXT, Section, U]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, U])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))
