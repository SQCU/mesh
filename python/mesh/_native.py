import ctypes as C
from pathlib import Path

P, U, Z = C.c_void_p, C.c_uint32, C.c_size_t


class Shape(C.Structure):
    _fields_ = [('rows', Z), ('columns', Z), ('scalar', C.c_int)]


class View(C.Structure):
    _fields_ = [('tensor', P), ('extent', U)] + [(k, Z) for k in
        ('offset', 'rows', 'columns', 'row_stride', 'column_stride')]


class Endpoint(C.Structure):
    _fields_ = [('tensor', P), ('peer', U), ('first', U), ('stride', U)]


class MetalDispatch(C.Structure):
    _fields_ = [("name", C.c_char_p), ("grid", Z * 3), ("group", Z * 3), ("argument_buffer", Z), ("argument_offset", Z)]


class MetalConstant(C.Structure):
    _fields_ = [("bytes", P), ("length", Z)]


class Report(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in ('submitted', 'completed',
        'native_submitted', 'native_backings', 'ne_planned_operations')]
    _fields_ += [('code', C.c_int64), ('gpu_seconds', C.c_double),
                 ('cpu_submitted', C.c_uint64)]


Completion = C.CFUNCTYPE(None, P, C.c_int64)
Submission = C.CFUNCTYPE(None, P, Completion, P)


# design/algorithm-sources.md#indexed-library-functions
class Native:
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self):
        directory = Path(__file__).parent / 'lib'
        self.runtime = C.CDLL(str(directory / 'libmesh.dylib'), use_errno=True)
        self.algebra = C.CDLL(str(directory / 'libmesh-algebra.dylib'), use_errno=True)
        signatures = {
            'mesh_context': (P, []),
            'mesh_attach': (C.c_int, [P, C.c_char_p]),
            'mesh_detach': (C.c_int, [P]),
            'mesh_algebra_create': (P, [P]),
            'mesh_algebra_create_cpu': (P, [P]),
            'mesh_algebra_destroy': (None, [P]),
            'mesh_algebra_publication_bytes': (Z, [P]),
            'mesh_algebra_node': (U, [P]),
            'mesh_algebra_coreml': (C.c_int, [P, C.c_char_p, C.c_char_p, C.c_char_p]),
            'mesh_tensor_create': (P, [P, C.POINTER(Shape), Z, C.c_int]),
            'mesh_tensor_view': (View, [P, U]),
            'mesh_tensor_data': (P, [P, U]),
            'mesh_view_slice': (View, [View, Z, Z, Z, Z]),
            'mesh_view_transpose': (View, [View]),
            'mesh_view_broadcast': (View, [View, Z, Z]),
            'mesh_tensor_present': (C.c_int, [P, U]),
            'mesh_tensor_constant': (C.c_int, [P, U]),
            'mesh_tensor_writable': (C.c_int, [P, U]),
            'mesh_tensor_issue': (C.c_int, [P, U]),
            'mesh_tensor_complete': (None, [P, U]),
            'mesh_algebra_function': (C.c_int, [P, C.POINTER(View), Z,
                C.POINTER(View), Z, Submission, P]),
            'mesh_algebra_metal': (C.c_int, [P, C.c_char_p, C.POINTER(MetalDispatch), Z,
                C.POINTER(MetalConstant), Z, C.POINTER(View), Z, C.POINTER(View), Z]),
            'mesh_algebra_bind': (C.c_int, [P, C.c_int, View, View, View, C.c_float, C.c_float]),
            'mesh_algebra_copy': (C.c_int, [P, Endpoint, Endpoint, Z, C.c_uint16]),
            'mesh_algebra_export': (C.c_int, [P, P, U, C.POINTER(Z)]),
            'mesh_algebra_realize': (C.c_int, [P]),
            'mesh_algebra_available': (C.c_int, [P, Z]),
            'mesh_algebra_consume': (None, [P, Z]),
            'mesh_algebra_report': (Report, [P]),
        }
        for name, (result, arguments) in signatures.items():
            library = self.runtime if name in ('mesh_context', 'mesh_attach', 'mesh_detach') else self.algebra
            function = getattr(library, name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name.removeprefix('mesh_'), function)
