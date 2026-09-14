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


class Endpoint(C.Structure):
    _fields_ = [('tensor', P), ('peer', U), ('first', U), ('stride', U)]


class MetalDispatch(C.Structure):
    _fields_ = [("name", C.c_char_p), ("grid", Z * 3), ("group", Z * 3), ("argument_buffer", Z), ("argument_offset", Z)]


class MetalConstant(C.Structure):
    _fields_ = [("bytes", P), ("length", Z)]


class RowRange(C.Structure):
    _fields_ = [("first", U), ("count", U)]


class RowMap(C.Structure):
    _fields_ = [(name, U) for name in ('first', 'count', 'stride', 'plane')] + [
        ('ranges', C.POINTER(RowRange)), ('members', C.POINTER(U)), ('member_offsets', C.POINTER(Z))]


class RowFunction(C.Structure):
    _fields_ = [('input', C.POINTER(RowMap)), ('output', C.POINTER(RowMap))] + [
        (name, U) for name in ('inputs', 'outputs', 'rows')] + [
        (name, P) for name in ('indexed', 'routes', 'active')]


class Writer(C.Structure):
    _fields_ = [('context', P), ('output', RowMap), ('function', RowFunction)]


class ReaderEvent(C.Structure):
    _fields_ = [(name, U) for name in ('source', 'member', 'plane', 'completed', 'flags')]


class IndexedEvent(C.Structure):
    _fields_ = [('input', C.c_uint64)] + [(name, U) for name in
        ('descriptor', 'role', 'candidate', 'first', 'count', 'plane', 'retired',
         'selected', 'completed', 'mapped', 'flags')]


class RouteEvent(C.Structure):
    _fields_ = [('function', C.c_uint64)] + [(name, U) for name in
        ('domain', 'role', 'index', 'first', 'count', 'plane', 'retired',
         'completed', 'prepared', 'consumer', 'flags')]


class ActiveEvent(C.Structure):
    _fields_ = [('function', C.c_uint64), ('omissions', C.c_uint64)] + [(name, U) for name in
        ('slot', 'count_first', 'count_maps', 'disposition', 'omitted', 'retired', 'inputs', 'flags')]


class Event(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
        ('ready_ns', 'start_ns', 'complete_ns', 'gpu_start_ns', 'gpu_end_ns', 'submissions')]
    _fields_ += [(name, U) for name in ('first_output', 'output_maps', 'kind', 'input_maps')]


# design/algorithm-sources.md#function-cost-profiles
class Profile(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ('successful', 'failed', 'gpu_samples')]
    _fields_ += [(name, C.c_double) for name in ('dispatch_mean_ns', 'dispatch_m2_ns2',
        'execution_mean_ns', 'execution_m2_ns2', 'gpu_mean_ns', 'gpu_m2_ns2')]
    _fields_ += [(name, U) for name in ('kind', 'backend')]


# design/algorithm-sources.md#function-cost-profiles
class Plan(C.Structure):
    _fields_ = [(name, View) for name in ('left', 'right', 'output')]
    _fields_ += [(name, C.c_uint64) for name in ('first', 'count')]
    _fields_ += [(name, U) for name in ('backend', 'operation', 'left_scalar', 'right_scalar', 'output_scalar', 'rectangles')]
    _fields_ += [(name, C.c_float) for name in ('alpha', 'beta')]


class Transfer(C.Structure):
    _fields_ = [(name, U) for name in ('local_row', 'local_page', 'peer_row', 'peer_page',
        'binding', 'offset', 'plane', 'index', 'bytes', 'peer_index')]


class TransferEvent(C.Structure):
    _fields_ = [('queue', U), ('direction', U), ('transfer', Transfer)]
    _fields_ += [(name, C.c_uint64) for name in ('ready_ns', 'post_ns', 'cq_ns', 'occurrences')]


class Report(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in ('submitted', 'completed',
        'native_submitted', 'native_backings', 'ne_planned_operations')]
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
            'mesh_algebra_environment': (C.c_char_p, [P]),
            'mesh_algebra_create': (P, [P]),
            'mesh_algebra_create_cpu': (P, [P]),
            'mesh_algebra_destroy': (None, [P]),
            'mesh_algebra_publication_bytes': (Z, [P]),
            'mesh_algebra_node': (U, [P]),
            'mesh_algebra_coreml': (C.c_int, [P, C.c_char_p, C.c_char_p, C.c_char_p]),
            'mesh_tensor_create': (P, [P, C.POINTER(Shape), Z, C.c_int, C.c_int]),
            'mesh_algebra_materialize': (C.c_int, [P, C.POINTER(CopyRegion), Z, View]),
            'mesh_tensor_view': (View, [P, U]),
            'mesh_tensor_data': (P, [P, U]),
            'mesh_tensor_publication_bytes': (Z, [P, U]),
            'mesh_tensor_rows': (RowMap, [P, U]),
            'mesh_view_slice': (View, [View, Z, Z, Z, Z]),
            'mesh_view_transpose': (View, [View]),
            'mesh_view_broadcast': (View, [View, Z, Z]),
            'mesh_tensor_constant': (C.c_int, [P, U]),
            'mesh_algebra_writer': (C.c_int, [P, View, C.POINTER(Writer)]),
            'mesh_writer_writable': (C.c_int, [C.POINTER(Writer)]),
            'mesh_writer_issue': (C.c_int, [C.POINTER(Writer)]),
            'mesh_writer_complete': (None, [C.POINTER(Writer)]),
            'mesh_algebra_metal': (C.c_int, [P, C.c_char_p, C.POINTER(MetalDispatch), Z,
                C.POINTER(MetalConstant), Z, C.POINTER(View), Z, C.POINTER(View), Z]),
            'mesh_algebra_source': (C.c_int, [P, C.c_char_p, C.c_char_p, C.POINTER(View), Z, View, C.POINTER(C.c_uint8), Z, Z, Z, Z]),
            'mesh_algebra_specialization': (C.c_char_p, [P, Z]),
            'mesh_algebra_source_text': (C.c_char_p, [P, Z, U]),
            'mesh_algebra_indexed': (C.c_int, [P, Z, View, C.POINTER(Z), Z]),
            'mesh_algebra_indexed_range': (C.c_int, [P, Z, View, View, C.POINTER(Z), Z]),
            'mesh_algebra_route_create': (P, [P, View, View, View, C.POINTER(View), Z, Z]),
            'mesh_algebra_route_table': (View, [P, P]),
            'mesh_algebra_route_attach': (C.c_int, [P, Z, P, Z]),
            'mesh_algebra_route_hold': (C.c_int, [P, P, C.POINTER(View), Z]),
            'mesh_algebra_active': (C.c_int, [P, Z, View, Z]),
            'mesh_algebra_route_producers': (C.c_int, [P, P, C.POINTER(Z), Z]),
            'mesh_algebra_trace_active': (ActiveEvent, [P, Z]),
            'mesh_algebra_trace_active_count': (RowRange, [P, Z, Z]),
            'mesh_algebra_trace_active_reader': (ReaderEvent, [P, Z, Z, U]),
            'mesh_algebra_trace_route_producer': (ActiveEvent, [P, Z]),
            'mesh_algebra_trace_route_count': (Z, [P]),
            'mesh_algebra_trace_route': (RouteEvent, [P, Z]),
            'mesh_algebra_trace_route_reader': (ReaderEvent, [P, Z, U]),
            'mesh_algebra_bind': (C.c_int, [P, C.c_int, View, View, View, C.c_float, C.c_float]),
            'mesh_algebra_view_pages': (C.c_int, [P, View, C.POINTER(View), Z, C.POINTER(Z)]),
            'mesh_algebra_contract_select': (C.c_int, [P, View, C.POINTER(View), C.POINTER(View), Z, C.POINTER(View), Z, View, C.c_float, C.POINTER(Z)]),
            'mesh_algebra_copy': (C.c_int, [P, Endpoint, Endpoint, Z, C.c_uint16]),
            'mesh_algebra_present': (C.c_int, [P, View]),
            'mesh_algebra_export': (C.c_int, [P, View, C.POINTER(Z), C.POINTER(Z)]),
            'mesh_algebra_realize': (C.c_int, [P]),
            'mesh_algebra_available': (C.c_int, [P, Z]),
            'mesh_algebra_consume': (None, [P, Z]),
            'mesh_algebra_trace_count': (Z, [P]),
            'mesh_algebra_trace': (Event, [P, Z]),
            'mesh_algebra_profile': (Profile, [P, Z]),
            'mesh_algebra_plan_count': (Z, [P, Z]),
            'mesh_algebra_plan': (Plan, [P, Z, Z]),
            'mesh_algebra_trace_input': (RowRange, [P, Z, Z]),
            'mesh_algebra_trace_output': (RowRange, [P, Z, Z]),
            'mesh_algebra_trace_indexed_count': (Z, [P, Z]),
            'mesh_algebra_trace_indexed': (IndexedEvent, [P, Z, Z]),
            'mesh_algebra_trace_input_reader': (ReaderEvent, [P, Z, Z, U]),
            'mesh_algebra_trace_indexed_reader': (ReaderEvent, [P, Z, Z, U]),
            'mesh_algebra_report': (Report, [P]),
            'mesh_transfer_trace_count': (Z, [P]),
            'mesh_transfer_trace': (TransferEvent, [P, Z]),
        }
        for name, (result, arguments) in signatures.items():
            library = self.runtime if name in ('mesh_context', 'mesh_attach', 'mesh_detach', 'mesh_transfer_trace_count', 'mesh_transfer_trace') else self.algebra
            function = getattr(library, name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name.removeprefix('mesh_'), function)
