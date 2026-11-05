import ctypes as C
import os

U, Q, Z, D = C.c_uint32, C.c_uint64, C.c_size_t, C.c_double


# mesh-dataflow.h struct mesh_ctx: the region mapped and the client word; no claim and no plan (a plan's claims are
# mesh_plan's, each plan named by its `*plan`)
# 2026-11-04, owner T2 (design/compilation-coordination.md, "The appendix's revisions", "The torch bindings" and N5):
# `epoch` and `state` go, as the header's 32 bytes: the context holds no plan, since a program holds any number of
# compiled functions at once (design/aot-compilation-change-request.md, appendix item 6).  What must stay: the fields
# in the header's order
class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), ('client', Q), ('fd', C.c_int), ('padding', U)]


# mesh.h struct hdr, its first words (magic, version, pgsz, block, in the header's order): the region's page size, the
# length unit of a placed value's copies (mesh-call.h struct mesh_section `pages`)
class Header(C.Structure):
    _fields_ = [('magic', U), ('version', U), ('pgsz', U), ('block', U)]


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


# mesh.h struct mesh_state, its words to `entries` (offsets 0 to 104, as the header asserts): the plan's state section
# at the region offset mesh_plan's `*plan` gives, whose `serial` is the map serial the plan was made on; `group` and
# `epoch` the plan's name, `epoch` the program's ordinal of the compiled function (design/compilation-coordination.md,
# "The appendix's revisions", N1)
# 2026-11-04, owner T2 (N1, N5): `epoch` reads as the plan's function, no longer the planning context's count of its
# mesh_plan calls (C-epoch goes), and the section is found by the plan's handle, no longer by context.state.  What
# must stay: the words in the header's order
class State(C.Structure):
    _fields_ = [('copies', U), ('receives', U), ('ranges', U), ('releases', U), ('phases', Q), ('counts', Q),
                ('supplied', Q), ('status', Q), ('executions', Q), ('padding', Q), ('group', Q), ('epoch', Q),
                ('serial', Q), ('hops', Q), ('entries', Q)]


# mesh-dataflow.h struct mesh_nodes, the map reader's (mesh_nodes_read)
NODES_MOST = 32


class Nodes(C.Structure):
    _fields_ = [('serial', Q), ('count', U), ('self', U), ('node', U * NODES_MOST),
                ('capacity', (Q * NODES_MOST) * NODES_MOST)]


# mesh-collective.h mesh_host_arrived's error value: a phase ahead, or a range cancelled
FAILED = 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
# 2026-11-04, owner T (design/compilation-coordination.md, A-T3'): mesh_graph_pack's and mesh_graph_tree_depth's
# bindings are gone: the lowering picks no tree, and mesh_plan's demands and trees route the collectives.  What must
# stay: the plan call's bindings
# 2026-11-04, owner T2 (design/compilation-coordination.md, "The appendix's revisions", "The plan call", "The region and
# the start", N1 to N4): mesh_plan takes the plan's `function` after `group` and gives the plan's handle in `*plan` (its
# state section's region offset); mesh_transfers_start takes that handle; mesh_plan_retire, the program's retirement of
# one compiled function, is bound.  What must stay: each binding the header's declaration
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_nodes_read', C.c_int, [CONTEXT, C.POINTER(Nodes)]),
        ('mesh_plan', C.c_int, [CONTEXT, Q, Q, C.POINTER(Declared), C.POINTER(Weight), U, C.POINTER(Value), U,
                                C.POINTER(Dependence), U, C.POINTER(Q)]),
        ('mesh_plan_retire', C.c_int, [CONTEXT, Q]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_transfers_start', C.c_int, [CONTEXT, Q, Q]),
        ('mesh_host_publish', None, [CONTEXT, Section, Q]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, Q])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))
