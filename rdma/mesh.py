import ctypes as C
import os

U, Q, Z, D = C.c_uint32, C.c_uint64, C.c_size_t, C.c_double


# mesh-dataflow.h struct mesh_ctx: the region mapped, the client word, `epoch` (its mesh_plan calls) and `state` (its
# standing plan's state section, 0 none); no claim (a plan's claims are mesh_plan's)
# 2026-11-04, owner T (design/compilation-coordination.md, "The region" and "The torch backend"): `padding`, `epoch` and
# `state` appended, as the header's 48 bytes.  What must stay: the fields in the header's order
class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), ('client', Q), ('fd', C.c_int), ('padding', U), ('epoch', Q),
                ('state', Q)]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


# mesh-collective.h struct mesh_value, struct mesh_declared, struct mesh_weight and struct mesh_dependence, and their
# constants as the header defines them (design/compilation-coordination.md, "The plan call")
class Value(C.Structure):
    _fields_ = [('section', Section), ('kind', U), ('wire', U), ('state', U), ('begin', U), ('end', U), ('lag', U)]


class Declared(C.Structure):
    _fields_ = [('source', U), ('destination', U), ('identity', U), ('kind', U), ('value', U), ('padding', U),
                ('bytes', Z)]


class Weight(C.Structure):
    _fields_ = [('produced', D), ('consumed', D), ('read', D), ('page', D)]


class Dependence(C.Structure):
    _fields_ = [('from', U), ('to', U), ('lag', U), ('kinds', U)]


INVARIANT, EXECUTION = 0, 1
JOIN, PAGES = 0, 1
FROM_PRODUCED, TO_PRODUCED, FROM_SOURCE_READ, TO_SOURCE_READ = 1, 2, 4, 8
# mesh.h MESH_ABSENT, MESH_NAME (the node's one region) and MESH_PLATFORM_NS
ABSENT = 2 ** 32 - 1
NAME = b'/mesh0'
PLATFORM_NS = 5000000000
# mesh-dataflow.h struct mesh_nodes, the map reader's (mesh_nodes_read)
NODES_MOST = 32


class Nodes(C.Structure):
    _fields_ = [('serial', Q), ('count', U), ('self', U), ('node', U * NODES_MOST),
                ('capacity', (Q * NODES_MOST) * NODES_MOST)]


# mesh-collective.h mesh_host_arrived's error value: a phase ahead, or a range cancelled
FAILED = 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
# mesh-graph.h mesh_graph_pack's `rooted`
ROOTED = C.CFUNCTYPE(C.c_int, C.c_void_p, U)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_nodes_read', C.c_int, [CONTEXT, C.POINTER(Nodes)]),
        ('mesh_plan', C.c_int, [CONTEXT, Q, C.POINTER(Declared), C.POINTER(Weight), U, C.POINTER(Value), U,
                                C.POINTER(Dependence), U]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_transfers_start', C.c_int, [CONTEXT, Q]),
        ('mesh_host_publish', None, [CONTEXT, Section, Q]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, Q]),
        ('mesh_graph_pack', U, [U, U, C.POINTER(U), C.POINTER(U), C.POINTER(U), C.POINTER(U), U, C.POINTER(D), ROOTED,
                                C.c_void_p, D, U, C.c_int, C.POINTER(U), C.POINTER(U), C.POINTER(D)])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))
