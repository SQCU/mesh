# design/the-compiler.md, order 4; mesh-metal4.m
import ctypes as C
import os

LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh-metal4.dylib'))
P, Q, U = C.c_void_p, C.c_uint64, C.c_uint32


class Dispatch(C.Structure):
    _fields_ = [('pipeline', P), ('table', P), ('address', C.POINTER(Q)), ('addresses', U), ('barrier', U),
                ('groups', U), ('padding', U), ('grid', Q * 3), ('group', Q * 3)]


for name, result, arguments in (
        ('mesh_metal4_pipeline', P, [C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]),
        ('mesh_metal4_width', Q, [P]),
        ('mesh_metal4_table', P, [U]),
        ('mesh_metal4_buffer', P, [Q]),
        ('mesh_metal4_view', P, [P, Q]),
        ('mesh_metal4_release', None, [P]),
        ('mesh_metal4_address', Q, [P]),
        ('mesh_metal4_contents', P, [P]),
        ('mesh_metal4_after', None, [P, P, P, P]),
        ('mesh_metal4_fence', P, []),
        ('mesh_metal4_stream', P, [C.POINTER(P), U]),
        ('mesh_metal4_record', P, []),
        ('mesh_metal4_times', None, [P, C.POINTER(C.c_double)]),
        ('mesh_metal4_commit', None, [P, C.POINTER(Dispatch), U, C.POINTER(P), U, C.POINTER(P), U, P, P]),
        ('mesh_metal4_wait', C.c_int, [C.POINTER(P), U, C.c_char_p, C.c_size_t]),
        ('mesh_metal4_reset', None, [P])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments
MESSAGE = 4096


def pipeline(source, name):
    message = C.create_string_buffer(MESSAGE)
    made = LIB.mesh_metal4_pipeline(source.encode(), name.encode(), message, MESSAGE)
    if not made:
        raise RuntimeError(message.value.decode())
    return made


def wait(records):
    message = C.create_string_buffer(MESSAGE)
    if LIB.mesh_metal4_wait(records, len(records), message, MESSAGE):
        raise RuntimeError(message.value.decode())


def times(record):
    found = (C.c_double * 2)()
    LIB.mesh_metal4_times(record, found)
    return tuple(found)
