import ctypes as C
import os

U, Q, Z, D = C.c_uint32, C.c_uint64, C.c_size_t, C.c_double


# mesh-dataflow.h struct mesh_ctx: the region mapped and the client word; no claim and no plan (a plan's claims are
# mesh_plan's, each plan named by its `*plan`)
# 2026-11-04, owner T2 (design/compilation-coordination.md, "The appendix's revisions", "The torch bindings" and N5):
# `epoch` and `state` go, as the header's 32 bytes: the context holds no plan, since a program holds any number of
# compiled functions at once (design/aot-compilation-change-request.md, appendix item 6).  What must stay: the fields
# in the header's order
# 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS): `clock` appended, the program's
# Lamport clock (mesh-dataflow.h struct mesh_ctx), which mesh_transfers_start increments.  What must stay: the fields
# in the header's order
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3 and G4; "The
# revisions of 55b9cbad"): `start` appended, the process's start mesh_attach reads, which mesh_plan copies into the
# operand's holder (48 bytes).  What must stay: the fields in the header's order
class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), ('client', Q), ('fd', C.c_int), ('padding', U), ('clock', Q),
                ('start', Q)]


# mesh.h struct hdr, its first words (magic, version, pgsz, block, in the header's order): the region's page size, the
# length unit of a placed value's copies (mesh-call.h struct mesh_section `pages`), which the MPS binding's buffer
# over a value's copies needs, whole pages long as metal-microbench tools/e2b_replay.m placed_buffer makes it
# (design/compilation-coordination.md, "The appendix's revisions", K1; rdma/mesh_compile.py _placed_buffer)
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 and 7, and "The
# revisions of 4e78f6f4"): read for a copy's place in the program's mapping (mesh-collective.h struct mesh_value
# `view`: copy c at view + c x section.pages x pgsz) and for a block's bytes (block x pgsz), which a part of a value
# must begin at (struct mesh_declared `first`).  What must stay: the header's first words in its order
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3; "The
# revisions of 55b9cbad"): the page size places a copy in the program's own mapping of its rows (rdma/mesh_compile.py
# _mapped, mesh-dataflow.h mesh_view_compose), struct mesh_value `view` being gone.  What must stay: the header's first
# words in its order
class Header(C.Structure):
    _fields_ = [('magic', U), ('version', U), ('pgsz', U), ('block', U)]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


# mesh-collective.h struct mesh_value, struct mesh_declared, struct mesh_weight and struct mesh_dependence, and their
# constants as the header defines them (design/compilation-coordination.md, "The plan call")
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 to 8, and "The
# revisions of 4e78f6f4"; mesh dd215cbe): Value's `run`, `padding` and `view` appended (64 bytes, mesh-collective.h's
# assert), Declared's `first` where `padding` was (32 bytes), and struct mesh_function (Function), the plan call's one
# compiled function, and mesh-dataflow.h struct mesh_view (View), the program's one mapping, mirrored.  What must stay:
# the fields in the headers' order and of their types
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3, K4 and V2;
# "The revisions of 55b9cbad"): Value's `view` gone (`padding1`) and `shares` out where `padding` was; struct
# mesh_share (Share) mirrored, and Function's `share` and `shares` (72 bytes).  What must stay: the fields in the
# headers' order and of their types
class Value(C.Structure):
    _fields_ = [('section', Section), ('kind', U), ('wire', U), ('state', U), ('begin', U), ('end', U), ('lag', U),
                ('run', U), ('shares', U), ('padding1', Q)]


# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"): mesh-collective.h struct mesh_strided (Strided), a view of a value's bytes, offset and strides
# in bytes, and Declared's `from` and `to` views where `first` and `bytes` were (88 bytes); struct mesh_share (Share)
# and Function's `share` and `shares` gone (64 bytes).  What must stay: the fields in the headers' order and of their
# types
# MPI 4.1 section 6.1.3, MPI_TYPE_CREATE_SUBARRAY: "The ndims parameter specifies the number of dimensions in the full
# data array and gives the number of elements in array_of_sizes, array_of_subsizes, and array_of_starts."
class Strided(C.Structure):
    _fields_ = [('offset', Q), ('element', Q), ('dims', U), ('padding', U), ('pairs', C.POINTER(Q))]


class Declared(C.Structure):
    _fields_ = [('source', U), ('destination', U), ('identity', U), ('kind', U), ('value', U), ('padding', U),
                ('from_', Strided), ('to', Strided)]


class Weight(C.Structure):
    _fields_ = [('produced', D), ('consumed', D), ('read', D), ('page', D)]


class Dependence(C.Structure):
    _fields_ = [('from', U), ('to', U), ('lag', U), ('kinds', U)]


class Function(C.Structure):
    _fields_ = [('declared', C.POINTER(Declared)), ('weights', C.POINTER(Weight)), ('value', C.POINTER(Value)),
                ('dependence', C.POINTER(Dependence)), ('count', U), ('values', U), ('dependences', U),
                ('padding', U), ('outstanding', Q), ('plan', Q)]


class View(C.Structure):
    _fields_ = [('address', C.c_void_p), ('bytes', Z)]


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
# 2026-11-05 (117; design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 4, and "The
# revisions of 4e78f6f4"): the whole section, 320 bytes, its tail from byte 232 the plan's synchronization edges
# (`waits`, `tells`, `syncs`), `calls` and `closed`, where it ended at `entries` (mesh.h struct mesh_state and its
# assert).  What must stay: the words in the header's order and at its offsets
# JGroups ViewId.java: "Lamport time of the view" (`serial`), "Address of the creator of this view" (`sequencer`,
# `boot`), here and in Nodes; UUID.java: "Logical address which is unique over space and time."  libfabric
# man/fi_cntr.3.md: "Counters record the number of requested operations that have completed." (`completions`)
class State(C.Structure):
    _fields_ = [('copies', U), ('receives', U), ('ranges', U), ('releases', U), ('phases', Q), ('counts', Q),
                ('supplied', Q), ('status', Q), ('executions', Q), ('completions', Q), ('group', Q), ('epoch', Q),
                ('serial', Q), ('hops', Q), ('entries', Q), ('held', Q), ('refused', C.c_int32 * 8), ('nodes', Q),
                ('sender', U), ('outstanding', U), ('submitted', Q), ('delivered', Q), ('submissions', Q),
                ('broadcast', Q), ('arrived', Q), ('room', Q), ('deliveries', Q), ('submits', Q), ('announced', Q),
                ('waits', U), ('tells', U), ('syncs', Q), ('calls', U), ('closed', U), ('rows', Q),
                ('submit_room', Q), ('transfers', Q), ('sequencer', U), ('padding4', U), ('boot', Q),
                ('padding3', Q * 3)]


# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-7: "Every
# wait ends at MESH_PLATFORM_NS without progress ... The operator's bound is judged by time without progress"; "The
# revisions of 14b9a757"): mesh.h struct mesh_publication, its fields before `targets`, and its size (64 bytes, the
# header's assert), the stride of struct mesh_state `transfers` (mesh.h mesh_transfer_at), so a host reader finds a
# received range's count word, copy c's at `count` + 8 c, whose rises are its progress (rdma/mesh_compile.py _arrived).
# What must stay: the fields in the header's order and at its offsets
# rdma-core libibverbs/man/ibv_post_send.3, NOTES: "The buffers used by a WR can only be safely reused after WR the
# request is fully executed and a work completion has been retrieved from the corresponding completion queue (CQ)."
# libfabric man/fi_cntr.3.md: "Counters record the number of requested operations that have completed." (`completion`)
class Target(C.Structure):
    _fields_ = [('word', Q), ('count', U), ('stride', U)]


class Publication(C.Structure):
    _fields_ = [('sends', U), ('pages', U), ('phases', Q), ('count', Q), ('state', Q), ('targets', Target * 1),
                ('completion', Q)]


PUBLICATION_BYTES = 64


# mesh-dataflow.h struct mesh_nodes, the map reader's (mesh_nodes_read)
NODES_MOST = 32


class Nodes(C.Structure):
    _fields_ = [('serial', Q), ('count', U), ('self', U), ('node', U * NODES_MOST),
                ('capacity', (Q * NODES_MOST) * NODES_MOST), ('sequencer', U), ('padding', U), ('boot', Q)]


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
# 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS and PQ): mesh_plan takes the
# program's nodes `program` and A10's bound `outstanding` after `function` (mesh-collective.h).  What must stay: as
# above
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 5 and 6, and "The
# revisions of 4e78f6f4"; mesh dd215cbe): mesh_plan takes the whole program, its compiled functions (Function) and
# its calls of them in program order, and gives its one mapping (View); mesh_plan_retire is void; mesh_view_release
# releases the mapping; mesh_retire, the program's retirement (mesh-dataflow.h: its root closed where its `owner` is
# the client), takes the region and the client word.  mesh_section_address's binding goes: a value's copies are bound
# at their places in the mapping (mesh-collective.h struct mesh_value `view`), and nothing here reads one row.  What
# must stay: each binding the header's declaration
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3 and B1; "The
# revisions of 55b9cbad"): mesh_plan gives no mapping (the program maps its `run` values' rows by mesh_view_compose);
# mesh_transfers_start takes the execution's one absolute deadline, CLOCK_MONOTONIC nanoseconds
# (time.clock_gettime_ns(time.CLOCK_MONOTONIC)).  What must stay: each binding the header's declaration
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
# revisions of 55b9cbad"; mesh-dataflow.h mesh_retire at mesh b35234c3), reversing "its root closed where its `owner`
# is the client" above: mesh_retire retires the program's operand (mesh.h struct mesh_algo, A9's `<algo>`) where hdr
# `client`, its holder, is the client and its `published` is not 0 (`retired` = `published`), taking the region and the
# client word as before.  What must stay: each binding the header's declaration
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1, R-7
# and R-10; "The revisions of 14b9a757"): mesh_plan takes the preparation's one absolute deadline, CLOCK_MONOTONIC
# nanoseconds, last; mesh_transfers_start takes none (each wait bounded by MESH_PLATFORM_NS without progress);
# mesh_host_publish and mesh_host_arrived name a range by its plan and its declared transfer's index, where they took
# its first row's section; mesh_view_values maps every `run` value of the program and gives each value's offset in the
# mapping.  What must stay: each binding the header's declaration
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_nodes_read', C.c_int, [CONTEXT, C.POINTER(Nodes)]),
        ('mesh_plan', C.c_int, [CONTEXT, Q, Q, C.POINTER(Function), U, C.POINTER(U), U, Q]),
        ('mesh_plan_retire', None, [CONTEXT, Q]),
        ('mesh_view_values', None, [CONTEXT, C.POINTER(Function), U, C.POINTER(Q), C.POINTER(Q), C.POINTER(View)]),
        ('mesh_retire', None, [C.c_void_p, Q]),
        ('mesh_transfers_start', C.c_int, [CONTEXT, Q, Q]),
        ('mesh_host_publish', None, [CONTEXT, Q, U, Q]),
        ('mesh_host_started', None, [CONTEXT, Q, U, Q]),
        ('mesh_host_arrived', Q, [CONTEXT, Q, U, Q]),
        ('mesh_host_sent', Q, [CONTEXT, Q, U, Q])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))
