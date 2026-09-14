import ctypes as C
from pathlib import Path

P, U, Z = C.c_void_p, C.c_uint32, C.c_size_t


class Shape(C.Structure):
    _fields_ = [('rows', Z), ('columns', Z), ('scalar', C.c_int)]


class View(C.Structure):
    _fields_ = [('tensor', P), ('extent', U)] + [(k, Z) for k in
        ('offset', 'rows', 'columns', 'row_stride', 'column_stride')]


class CopyRegion(C.Structure):
    _fields_ = [('source', View), ('row', Z), ('column', Z)]


class Report(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in ('submitted', 'completed',
        'native_submitted', 'native_backings')]
    _fields_ += [('code', C.c_int64), ('gpu_seconds', C.c_double),
                 ('cpu_submitted', C.c_uint64)]


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
            'mesh_algebra_kernel': (C.c_int, [P]),
            'mesh_algebra_destroy': (None, [P]),
            'mesh_algebra_publication_bytes': (Z, [P]),
            'mesh_algebra_page_bytes': (Z, [P]),
            'mesh_algebra_node': (U, [P]),
            'mesh_algebra_coreml': (C.c_int, [P, C.c_char_p, C.c_char_p, C.c_char_p]),
            'mesh_tensor_create': (P, [P, C.POINTER(Shape), Z, C.c_int, C.c_int]),
            'mesh_algebra_materialize': (C.c_int, [P, C.POINTER(CopyRegion), Z, View]),
            'mesh_tensor_view': (View, [P, U]),
            'mesh_tensor_data': (P, [P, U]),
            'mesh_tensor_publication_bytes': (Z, [P, U]),
            'mesh_view_slice': (View, [View, Z, Z, Z, Z]),
            'mesh_view_transpose': (View, [View]),
            'mesh_view_broadcast': (View, [View, Z, Z]),
            'mesh_tensor_constant': (C.c_int, [P, U]),
            'mesh_algebra_writer': (C.c_int, [P, View, C.POINTER(P)]),
            'mesh_writer_writable': (C.c_int, [P]),
            'mesh_writer_issue': (C.c_int, [P]),
            'mesh_writer_complete': (None, [P, C.c_int]),
            'mesh_algebra_source': (C.c_int, [P, C.c_char_p, C.c_char_p, C.POINTER(View), Z, View, C.POINTER(C.c_uint8), Z, Z, Z, Z]),
            'mesh_algebra_indexed': (C.c_int, [P, Z, View, C.POINTER(Z), Z, C.POINTER(View), Z]),
            'mesh_algebra_indexed_range': (C.c_int, [P, Z, View, View, C.POINTER(Z), Z, C.POINTER(View), Z]),
            'mesh_algebra_route_create': (P, [P, View, View, View, C.POINTER(View), Z, Z]),
            'mesh_algebra_route_table': (View, [P, P]),
            'mesh_algebra_route_attach': (C.c_int, [P, Z, P, Z]),
            'mesh_algebra_route_hold': (C.c_int, [P, P, C.POINTER(View), Z]),
            'mesh_algebra_active': (C.c_int, [P, Z, View, Z]),
            'mesh_algebra_route_producers': (C.c_int, [P, P, C.POINTER(Z), Z]),
            'mesh_algebra_bind': (C.c_int, [P, C.c_int, View, View, View, C.c_float, C.c_float]),
            'mesh_algebra_view_pages': (C.c_int, [P, View, C.POINTER(View), Z, C.POINTER(Z)]),
            'mesh_algebra_contract_select': (C.c_int, [P, View, C.POINTER(View), C.POINTER(View), Z, C.POINTER(View), Z, View, C.c_float, C.POINTER(Z)]),
            'mesh_algebra_copy': (C.c_int, [P, View, U, View, U, C.c_uint16]),
            'mesh_algebra_observe': (C.c_int, [P, View, C.POINTER(Z), C.POINTER(Z)]),
            'mesh_algebra_present': (C.c_int, [P, Z, Z]),
            'mesh_algebra_export': (C.c_int, [P, View, C.POINTER(Z), C.POINTER(Z)]),
            'mesh_algebra_realize': (C.c_int, [P]),
            'mesh_algebra_available': (C.c_int, [P, Z]),
            'mesh_algebra_consume': (None, [P, Z]),
            'mesh_algebra_function_count': (Z, [P]),
            'mesh_algebra_report': (Report, [P]),
        }
        for name, (result, arguments) in signatures.items():
            library = self.runtime if name in ('mesh_context', 'mesh_attach', 'mesh_detach') else self.algebra
            function = getattr(library, name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name.removeprefix('mesh_'), function)
