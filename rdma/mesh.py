import ctypes as C
import os

import mesh_graph

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


# mesh-dataflow.h struct mesh_range
class Range(C.Structure):
    _fields_ = [('first', U), ('count', U)]


class PlanGeometry(C.Structure):
    _fields_ = [('pgsz', U), ('block', U), ('rows', U), ('padding', U), ('arena', Range * 2)]


# mesh.h struct hdr
class Header(C.Structure):
    _fields_ = [('magic', U), ('version', U), ('pgsz', U), ('block', U), ('rows', U), ('node', U), ('vacant0', U),
                ('links', U), ('vacant1', U), ('wire_pages', U), ('vacant7', U), ('vacant14', U), ('vacant2', Q),
                ('planes_off', Q), ('arena_off', Q), ('page_off', Q), ('buffer_off', Q), ('link_off', Q),
                ('generation', Q), ('target_off', Q), ('preparing_client', Q), ('preparing_start', Q), ('data_off', Q),
                ('length', Q), ('vacant6', Q), ('target_stride', Q), ('vacant15', Q), ('bridge_pid', Q),
                ('device_client', Q),
                ('serial', Q), ('control', Q), ('port_phase', U), ('port_domain', U), ('port_code', C.c_int64),
                ('port_prepared', Q), ('view_map', Q), ('view', Q), ('vacant8', Q), ('view_off', Q),
                ('view_nodes', Q), ('map', Q), ('map_off', Q), ('map_nodes', Q), ('map_serial', Q),
                ('bridge_off', Q), ('vacant16', Q), ('boot', Q), ('health', Q), ('busy', Q), ('idle_since', Q),
                ('stopping', Q), ('vacant9', Q), ('vacant10', Q), ('vacant11', Q), ('vacant12', Q), ('vacant13', Q),
                ('map_room', Q), ('vacant17', Q), ('plans', Q), ('deliveries_off', Q), ('deliveries_room', Q),
                ('delivered', Q), ('delivery_read', Q * 8), ('view_sequencer', U), ('vacant22', U), ('view_boot', Q)]


# mesh.h struct mesh_hop, MESH_SEND, MESH_RECEIVE
class Hop(C.Structure):
    _fields_ = [('page', U), ('node', U), ('direction', U), ('field', U), ('word', Q), ('count', Q), ('pages', U),
                ('bytes', U), ('serial', Q), ('set', U), ('offset', U)]


SEND, RECEIVE = 0, 1


# mesh.h struct mesh_state, struct mesh_delivery, struct mesh_submission
STATE_BYTES, DELIVERY_BYTES, SUBMISSION_BYTES = 320, 32, 16


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
# mesh-collective.h struct mesh_value; design/joint-plan-results.md
class Value(C.Structure):
    _fields_ = [('section', Section), ('kind', U), ('wire', U), ('state', U), ('begin', U), ('end', U), ('lag', U),
                ('run', U), ('owned', U)]


# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"): mesh-graph.h struct mesh_strided (Strided), a view of a value's bytes, offset and strides
# in bytes, and Declared's `from` and `to` views where `first` and `bytes` were (88 bytes); struct mesh_share (Share)
# and Function's `share` and `shares` gone (64 bytes).  What must stay: the fields in the headers' order and of their
# types
# MPI 4.1 section 6.1.3, MPI_TYPE_CREATE_SUBARRAY: "The ndims parameter specifies the number of dimensions in the full
# data array and gives the number of elements in array_of_sizes, array_of_subsizes, and array_of_starts."
class Strided(C.Structure):
    _fields_ = [('offset', Q), ('element', Q), ('dims', U), ('padding', U), ('pairs', C.POINTER(Q))]


class Declared(C.Structure):
    _fields_ = [('source', U), ('destination', U), ('identity', U), ('kind', U), ('value', U), ('root', U),
                ('from_', Strided), ('to', Strided)]


class Weight(C.Structure):
    _fields_ = [('produced', D), ('consumed', D), ('read', D), ('page', D)]


class Dependence(C.Structure):
    _fields_ = [('from', U), ('to', U), ('lag', U), ('kinds', U)]


# mesh-collective.h struct mesh_task
class Task(C.Structure):
    _fields_ = [('length', D), ('resource', U), ('padding', U)]


class Function(C.Structure):
    _fields_ = [('declared', C.POINTER(Declared)), ('weights', C.POINTER(Weight)), ('value', C.POINTER(Value)),
                ('dependence', C.POINTER(Dependence)), ('count', U), ('values', U), ('dependences', U),
                ('padding', U), ('outstanding', Q), ('host', C.POINTER(C.c_double)),
                ('work', C.POINTER(C.c_double)), ('edge', C.POINTER(mesh_graph.Dependence)), ('parts', U), ('edges', U),
                ('task', C.POINTER(Task)), ('before', C.POINTER(mesh_graph.Dependence)), ('sender', C.POINTER(U)),
                ('receiver', C.POINTER(U)), ('tasks', U), ('resources', U),
                ('befores', U), ('padding2', U), ('relation', C.c_void_p)]


class View(C.Structure):
    _fields_ = [('address', C.c_void_p), ('bytes', Z)]


# design/compilation-composition-change-request.md item 6, unit 6D; mesh-collective.h struct mesh_planned_call, struct
# mesh_planned; design/algorithm-sources.md#the-plans-results-for-native-preparation
class PlannedCall(C.Structure):
    _fields_ = [('lead', Q), ('function', U), ('ordinal', U), ('every', U), ('parts', U), ('waits', U),
                ('padding', U)]


class PlannedCause(C.Structure):
    _fields_ = [('kind', U), ('pair', U)]


# mesh-collective.h struct mesh_planned_transfer
class PlannedTransfer(C.Structure):
    _fields_ = [('received', Z), ('sending', Z), ('end', Z)]


class PlanWorkspace(C.Structure):
    _fields_ = [('function', U), ('lifetime', U), ('copies', U), ('reusable', U),
                ('body', U), ('value', U), ('shares', U), ('zeroed', U), ('bytes', Q), ('first', Z), ('section', Section)]


class PlanValue(C.Structure):
    _fields_ = [('section', Section), ('shares', U)]


class PlanActors(C.Structure):
    _fields_ = [('host', C.POINTER(D)), ('work', C.POINTER(D)), ('edge', C.POINTER(mesh_graph.Dependence)),
                ('parts', U), ('edges', U), ('relation', C.c_void_p)]


class PlanScheduleResult(C.Structure):
    pass


class PlanFunction(C.Structure):
    _fields_ = [('value', C.POINTER(PlanValue)), ('completion', C.POINTER(D)),
                ('order', C.POINTER(U)), ('actors', C.POINTER(PlanActors)), ('schedule', C.POINTER(PlanScheduleResult))]


class PlanAnalysis(C.Structure):
    pass


class Planned(C.Structure):
    _fields_ = [('call', C.POINTER(PlannedCall)), ('signature', C.POINTER(mesh_graph.Signature)),
                ('order', C.POINTER(mesh_graph.Dependence)), ('calls', U), ('padding', U), ('orders', Z),
                ('cause', C.POINTER(PlannedCause)), ('horizon', Q), ('functions', U), ('extents', Z),
                ('first', C.POINTER(Z)), ('transfer', C.POINTER(PlannedTransfer)),
                ('extent', C.POINTER(mesh_graph.Access)), ('owned', C.c_void_p), ('used', Q * 2),
                ('arena', Range * 2), ('rows', Range),
                ('result', C.POINTER(PlanFunction)), ('root', C.POINTER(PlanValue)), ('results', U), ('roots', U), ('analysis', C.POINTER(PlanAnalysis))]


# mesh-collective.h struct mesh_demand, struct mesh_grouped and mesh_declared_group: the one canonical grouping of a
# function's declared rows into demands on one node (the caller's arrays demand, demand_of, release_of)
class Demand(C.Structure):
    _fields_ = [('source', U), ('identity', U), ('first', U), ('release', U), ('leaves', Q), ('bytes', Q),
                ('pages', U), ('padding', U)]


class Grouped(C.Structure):
    _fields_ = [('demand', C.POINTER(Demand)), ('demand_of', C.POINTER(U)), ('release_of', C.POINTER(U)),
                ('transfers', U), ('demands', U), ('releases', U), ('nodes', U), ('self', U), ('padding', U)]


# mesh-collective.h struct mesh_local_use: call `call` binds a program root to its function's value `value`
class LocalUse(C.Structure):
    _fields_ = [('call', U), ('value', U)]


class PlanItem(C.Structure):
    _fields_ = [('element', Q), ('blocks', Q), ('kind', U), ('function', U), ('index', U), ('copy', U), ('space', U)]


class PlanPlacement(C.Structure):
    _fields_ = [('item', C.POINTER(PlanItem)), ('count', C.POINTER(Q)), ('chain', C.POINTER(Q)),
                ('items', Q), ('elements', Q), ('split', U), ('table', U), ('range', Range * 2),
                ('at', C.POINTER(Q)), ('blocks', Q), ('allocations', Z), ('run', U), ('space', U),
                ('used', Q * 2), ('size', C.POINTER(Q)), ('start', C.POINTER(Q)), ('end', C.POINTER(Q)),
                ('offset', C.POINTER(Q)), ('operations', C.POINTER(Q)), ('function', C.POINTER(U)),
                ('index', C.POINTER(U)), ('copy', C.POINTER(U))]


# mesh-collective.h struct mesh_local, mesh_plan_local, mesh_local_save and mesh_local_load: the whole solved result
# with source operands retained in planned.analysis
class Local(C.Structure):
    _fields_ = [('planned', Planned), ('bytes', Q), ('values', Z), ('offsets', Z), ('first', C.POINTER(Z)),
                ('offset', C.POINTER(Q)), ('value_first', C.POINTER(Z)), ('demand_first', C.POINTER(Z)),
                ('demand_of', C.POINTER(U)), ('release_of', C.POINTER(U)),
                ('quantum', Q), ('placement', C.POINTER(PlanPlacement))]


class PlanStep(C.Structure):
    pass


class PlanStorageFunction(C.Structure):
    _fields_ = [('schedule', C.POINTER(PlanScheduleResult)), ('function', C.POINTER(Function)), ('lifetimes', Z),
                ('first', C.POINTER(Z)), ('last', C.POINTER(Z)), ('index', C.POINTER(U)),
                ('copies', U), ('value_lag', C.POINTER(U)), ('before', mesh_graph.Closed)]


class PlanStorage(C.Structure):
    _fields_ = [('references', Z), ('analysis', C.POINTER(PlanAnalysis)),
                ('bounds', C.POINTER(mesh_graph.BufferResult)), ('bound_function', C.POINTER(U)),
                ('horizon', U), ('function_lag', C.POINTER(Q)),
                ('function', C.POINTER(PlanStorageFunction)), ('mapped', C.c_void_p)]


class PlanCompilation(C.Structure):
    _fields_ = [('references', Z), ('result', Local), ('step', C.POINTER(PlanStep)),
                ('placement', C.POINTER(PlanPlacement)), ('root_item', C.POINTER(U)),
                ('page', C.POINTER(U)), ('adjacency', C.POINTER(U)), ('toward', C.POINTER(U)),
                ('place', C.POINTER(U)), ('rows_by', C.POINTER(Z)), ('listed', C.POINTER(Z)),
                ('places', Q), ('state', C.POINTER(C.c_void_p)), ('state_bytes', C.POINTER(Q)),
                ('owned', C.c_void_p), ('storage', C.POINTER(PlanStorage)), ('workspaces', Z),
                ('workspace', C.POINTER(PlanWorkspace))]


class PlanBinding(C.Structure):
    _fields_ = [('compilation', C.POINTER(PlanCompilation)), ('rows', Range), ('arena', Range * 2),
                ('generation', Q), ('plan', C.POINTER(Q)), ('page', C.POINTER(U))]


# mesh-collective.h: an order pair's Bernstein condition (mesh_plan `causes`) and the other causes of a planned edge
(ORDER_FLOW, ORDER_CONFLICT, PLANNED_PROGRAM, PLANNED_SUBMIT, PLANNED_OUTSTANDING, PLANNED_CHAIN, PLANNED_TRANSFER,
 PLANNED_PART) = range(8)
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
# 2026-11-09 (design/compilation-composition-change-request.md item 6, unit 6D): `signature` and `signature_bytes` go
# with the plan record's region copy (the plan's signature is mesh_planned `signature` in the compilation; the exported
# record is the composed program's). receive_wakes is relative to this state (the section stays 320 bytes by its
# alignment).  What must stay: the words in the header's order and at its offsets
class State(C.Structure):
    _fields_ = [('copies', U), ('receives', U), ('ranges', U), ('releases', U), ('phases', Q), ('counts', Q),
                ('supplied', Q), ('status', Q), ('executions', Q), ('completions', Q), ('group', Q), ('epoch', Q),
                ('serial', Q), ('hops', Q), ('entries', Q), ('held', Q), ('refused', C.c_int32 * 8), ('nodes', Q),
                ('sender', U), ('outstanding', U), ('submitted', Q), ('delivered', Q), ('submissions', Q),
                ('broadcast', Q), ('arrived', Q), ('room', Q), ('deliveries', Q), ('submits', Q), ('announced', Q),
                ('waits', U), ('tells', U), ('syncs', Q), ('calls', U), ('closed', U), ('rows', Q),
                ('submit_room', Q), ('transfers', Q), ('sequencer', U), ('padding4', U), ('boot', Q),
                ('times', Q), ('receive_wakes', Q)]


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
                ('completion', Q), ('origin', Q)]


PUBLICATION_BYTES = 64


# mesh-dataflow.h struct mesh_nodes, the map reader's (mesh_nodes_read)
NODES_MOST = 32


class Nodes(C.Structure):
    _fields_ = [('serial', Q), ('count', U), ('self', U), ('node', U * NODES_MOST),
                ('capacity', (Q * NODES_MOST) * NODES_MOST), ('sequencer', U), ('padding', U), ('boot', Q)]


class PlanSlot(C.Structure):
    _fields_ = [('root', U), ('local', U)]


class PlanFlow(C.Structure):
    pass


class PlanEndpoint(C.Structure):
    _fields_ = [('transfer', U), ('role', U), ('view', C.POINTER(Strided))]


PLAN_FROM, PLAN_TO = 0, 1


class PlanCuts(C.Structure):
    pass


PlanCuts._fields_ = [('geometry', C.POINTER(PlanGeometry)), ('source', C.POINTER(PlanFlow)),
                    ('demand', U), ('views', U), ('endpoint', C.POINTER(PlanEndpoint)),
                    ('spans', U), ('padding', U), ('span', C.POINTER(mesh_graph.Interval)),
                    ('previous', C.POINTER(PlanCuts)), ('instructions', Q), ('cut', C.POINTER(Q)),
                    ('at', C.POINTER(Q)), ('route', C.POINTER(U)), ('count', C.POINTER(Q)), ('previous_first', C.POINTER(Q)),
                    ('previous_last', C.POINTER(Q)), ('mapped', C.c_void_p)]


class PlanDemand(C.Structure):
    _fields_ = [('source', U), ('identity', U), ('first', U), ('mine', U), ('root', U), ('view', C.POINTER(Strided)),
                ('tree', U), ('trees', U), ('release', U), ('leaves', Q), ('bytes', Q),
                ('relay', U), ('on', Q), ('forwards', U), ('own', U), ('set', U), ('vertex', U), ('value', U)]


class PlanVertex(C.Structure):
    _fields_ = [('demand', U), ('node', U), ('write', U), ('read', U), ('reads', U), ('source', U)]


class PlanEntry(C.Structure):
    _fields_ = [('node', U), ('direction', U), ('demand', U), ('field', U), ('communication', U), ('set', U),
                ('instruction', Q)]


class PlanDemandGeometry(C.Structure):
    _fields_ = [('range', U), ('received', U), ('consumed', U), ('counted', U),
                ('sending', U), ('sends', U), ('group', Q)]


class PlanFlowGeometry(C.Structure):
    _fields_ = [('references', Z), ('source', C.POINTER(PlanFlow)),
                ('cuts', C.POINTER(C.POINTER(PlanCuts))), ('count', C.POINTER(C.POINTER(Q))),
                ('local', C.POINTER(U)), ('lead', C.POINTER(Q)), ('comm', C.POINTER(Q)), ('comms', Q),
                ('demand', C.POINTER(PlanDemandGeometry)), ('entry', C.POINTER(PlanEntry)),
                ('entries', Q), ('receives', Q), ('toward', Q), ('group_bytes', Q), ('forwarded', Q),
                ('ranges', U), ('consumed', U), ('sending', U), ('page', C.POINTER(D)),
                ('time', D * NODES_MOST), ('mapped', C.c_void_p)]


class ViewId(C.Structure):
    _fields_ = [('id', Q), ('sequencer', U), ('padding', U), ('boot', Q)]


PlanFlow._fields_ = [('geometry', C.POINTER(PlanGeometry)), ('declared', C.POINTER(Declared)), ('transfers', U),
                ('given', C.POINTER(Value)), ('value_lag', C.POINTER(U)), ('givens', U),
                ('named', C.POINTER(C.c_ubyte)), ('uncovered', C.POINTER(U)),
                ('endpoint', C.POINTER(PlanEndpoint)), ('transfer_rows', C.POINTER(mesh_graph.Grouping)),
                ('cuts', C.POINTER(C.POINTER(PlanCuts))), ('counts', C.POINTER(C.POINTER(Q))),
                ('cut_local', C.POINTER(U)), ('physical', C.POINTER(PlanFlowGeometry)),
                ('sets', U), ('set_first', C.POINTER(U)),
                ('member', C.POINTER(U)), ('set_blocks', C.POINTER(U)), ('set_far', C.POINTER(U)),
                ('block_first', C.POINTER(U)), ('block_set', C.POINTER(U)), ('block_element', C.POINTER(U)),
                ('weights', C.POINTER(Weight)), ('dependence', C.POINTER(Dependence)), ('dependences', U),
                ('demand', C.POINTER(PlanDemand)), ('demands', U), ('n', U), ('self', U), ('copies', U),
                ('start_lead', U), ('view', ViewId), ('of', C.POINTER(U)), ('own', Grouped), ('pairs', U),
                ('from', U * (NODES_MOST * (NODES_MOST - 1) // 2 + 1)),
                ('to', U * (NODES_MOST * (NODES_MOST - 1) // 2 + 1)),
                ('capacity', D * (NODES_MOST * (NODES_MOST - 1) // 2 + 1)),
                ('tree', C.POINTER(Z)), ('parent', C.POINTER(U)), ('run', C.POINTER(Q)), ('made', U),
                ('members', C.POINTER(Q)), ('span', C.POINTER(mesh_graph.Interval)),
                ('releases', Q), ('blocks', Q), ('child', C.POINTER(Q)), ('roots', C.POINTER(Value)),
                ('nroots', U), ('carried', C.POINTER(C.c_ubyte)), ('bound', C.POINTER(U)),
                ('value', C.POINTER(PlanVertex)), ('values', U), ('read', C.POINTER(U)), ('here', C.POINTER(U)),
                ('local', U), ('vertices', U), ('edges', Z), ('edge', C.POINTER(mesh_graph.Dependence)),
                ('rows', C.POINTER(mesh_graph.Grouping)), ('between', C.POINTER(U)), ('farthest', C.POINTER(U)),
                ('outstanding', Q), ('spanned', Q), ('nodes', Q), ('calls', U), ('horizon', U),
                ('adjacency', C.POINTER(C.POINTER(mesh_graph.Tree))),
                ('acyclic_indegree', C.POINTER(U)), ('acyclic_ordinal', C.POINTER(U)),
                ('lag_source', C.POINTER(U)), ('lag_target', C.POINTER(U)),
                ('lag_graph', C.POINTER(mesh_graph.LagGraph)), ('lags', C.POINTER(mesh_graph.LagMatrix)),
                ('covers', C.POINTER(mesh_graph.CoverResult)), ('touch', C.POINTER(mesh_graph.Touch)),
                ('relay', C.POINTER(U)), ('slot', C.POINTER(PlanSlot))]


PlanAnalysis._fields_ = [
    ('references', Z), ('geometry', C.POINTER(PlanGeometry)), ('program', Q), ('function', C.POINTER(Function)),
    ('functions', U), ('calls', C.POINTER(U)), ('chains', C.POINTER(U)), ('ordered', C.POINTER(U)),
    ('source_calls', C.POINTER(U)), ('orders', C.POINTER(U)), ('causes', C.POINTER(U)), ('ncalls', U), ('norders', U),
    ('sources', U), ('horizon', U), ('source_outstanding', C.POINTER(Q)), ('relation', C.c_void_p),
    ('closure', C.POINTER(mesh_graph.Closed)), ('root', C.POINTER(Value)), ('roots', U), ('use_first', C.POINTER(Z)),
    ('use', C.POINTER(LocalUse)), ('grouped', C.POINTER(Grouped)), ('flow', C.POINTER(PlanFlow)),
    ('call_group', C.POINTER(mesh_graph.Grouping)), ('source_group', C.POINTER(mesh_graph.Grouping)),
    ('chain_pairs', C.POINTER(mesh_graph.ChainResult)), ('bounds', C.POINTER(mesh_graph.BufferResult)),
    ('actor_function', C.POINTER(U)), ('function_lag', C.POINTER(Q)),
    ('bound_host', C.POINTER(D)), ('bound_work', C.POINTER(D)),
    ('bound_edge', C.POINTER(mesh_graph.Dependence)), ('bound_function', C.POINTER(U)),
    ('carried', C.POINTER(C.c_ubyte)),
    ('nodes', C.POINTER(Nodes)), ('owned', C.c_void_p), ('environment', C.c_void_p)]


class PlanSync(C.Structure):
    _fields_ = [('node', U), ('function', U), ('entry', U), ('every', U), ('base', C.c_int64)]


class StepEntry(C.Structure):
    _fields_ = [('source', C.POINTER(PlanEntry)), ('place', U)]


class PlanSpan(C.Structure):
    _fields_ = [('first', U), ('count', U)]


class PlanAccessQuery(C.Structure):
    _fields_ = [('task', U), ('source', U), ('sink', U), ('dependency', Z)]


PlanScheduleResult._fields_ = [('references', Z), ('geometry', C.POINTER(PlanFlowGeometry)), ('function', C.POINTER(Function)),
                ('first', U), ('tasks', U), ('instructions', U), ('edges', Z), ('entries', Z),
                ('dependencies', Z), ('reader', C.POINTER(U)), ('consumer', C.POINTER(PlanSpan)),
                ('w', C.POINTER(D)), ('cost', C.POINTER(D)), ('rate', C.POINTER(D)),
                ('completion', C.POINTER(D)), ('priority', C.POINTER(D)), ('completed', C.POINTER(D)), ('resource', C.POINTER(U)),
                ('from', C.POINTER(U)), ('to', C.POINTER(U)), ('link', C.POINTER(U)),
                ('communication', C.POINTER(U)), ('before', C.POINTER(U)), ('after', C.POINTER(U)),
                ('place', C.POINTER(U)), ('route', C.POINTER(Z)), ('owned', C.POINTER(PlanAnalysis)),
                ('accesses', Z), ('access', C.POINTER(mesh_graph.Viewed)), ('access_relation', C.c_void_p),
                ('page_first', C.POINTER(U)), ('page_edge', C.POINTER(U)), ('access_role', C.POINTER(C.c_ubyte)),
                ('event_first', C.POINTER(Z)), ('event_member', C.POINTER(U)),
                ('task_edge', C.POINTER(Z)), ('access_edge', C.POINTER(Z)), ('ordering', C.c_void_p),
                ('order', C.POINTER(U)), ('resource_cycle', C.POINTER(mesh_graph.Dependence)),
                ('resource_edges', Z), ('access_queries', Z), ('access_query', C.POINTER(PlanAccessQuery))]


class PlanGraphResult(C.Structure):
    _fields_ = [('analysis', C.POINTER(PlanAnalysis)), ('vertices', U), ('added', U), ('built', Z),
                ('lambda', D), ('time', C.POINTER(D)), ('actor', C.POINTER(U)), ('later', C.POINTER(U)),
                ('from', C.POINTER(U)), ('to', C.POINTER(U)), ('call', C.POINTER(U)), ('node', C.POINTER(U)),
                ('observer', C.POINTER(U)), ('key', C.POINTER(U)), ('added_from', C.POINTER(U)),
                ('added_to', C.POINTER(U)), ('entry', C.POINTER(U)), ('lag', C.POINTER(Q)),
                ('lagged', C.POINTER(Q)), ('delay', C.POINTER(Q)), ('added_delay', C.POINTER(Q)),
                ('first', C.POINTER(Z)), ('rows', C.POINTER(Z)), ('wait_first', C.POINTER(Z)),
                ('wait_order', C.POINTER(Z)), ('tell_first', C.POINTER(Z)), ('tell_order', C.POINTER(Z)),
                ('task', C.POINTER(mesh_graph.Task)), ('cause', C.POINTER(PlannedCause)),
                ('room_rows', C.POINTER(mesh_graph.Grouping)), ('room_graph', C.POINTER(mesh_graph.LagGraph)),
                ('room_query', C.POINTER(mesh_graph.Dependence)), ('room_lags', C.POINTER(mesh_graph.LagResult))]


PlanStep._fields_ = [
    ('flow', C.POINTER(PlanFlow)), ('geometry', C.POINTER(PlanFlowGeometry)),
    ('function', C.POINTER(Function)), ('result', C.POINTER(PlanFunction)),
    ('root_result', C.POINTER(PlanValue)), ('item', C.POINTER(U)), ('set_item', U), ('state', U), ('base', Q),
    ('entry', C.POINTER(StepEntry)), ('phases', Q), ('counts', Q),
    ('completions', Q), ('supplied', Q), ('publications', Q), ('status', Q), ('hops', Q), ('times', Q),
    ('words', Q), ('deliveries', Q), ('submits', Q), ('syncs', Q), ('rows', Q), ('room', Q), ('submit_room', Q),
    ('waits', U), ('tells', U), ('wait_row', C.POINTER(U)), ('sync', C.POINTER(PlanSync)),
    ('schedule', C.POINTER(PlanScheduleResult)), ('graph', C.POINTER(PlanGraphResult)),
    ('storage', C.POINTER(PlanStorageFunction))]


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
# the client), takes the region and the client word.  What must stay: each binding the header's declaration
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3 and B1; "The
# revisions of 55b9cbad"): mesh_plan gives no mapping (the program maps its `run` values' rows by mesh_view_compose);
# mesh_transfers_start takes the execution's one absolute deadline, CLOCK_MONOTONIC nanoseconds
# (time.clock_gettime_ns(time.CLOCK_MONOTONIC)).  What must stay: each binding the header's declaration
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
# revisions of 55b9cbad"; mesh-dataflow.h mesh_retire at mesh b35234c3), reversing "its root closed where its `owner`
# is the client" above: mesh_retire retires the program's operand (mesh.h struct mesh_algo, A9's `<algo>`) where hdr
# `client`, its holder, is the client and its `published` is not 0 (`retired` = `published`), taking the region and the
# client word as before.  What must stay: each binding the header's declaration
# 121 (design/plan-to-bridge-change-request.md R8; output_data/compiler-workgroup-20261114/compiler/publication-4.md):
# mesh_retire takes the region and the planned result whose publication it closes; mesh_publications_clear empties a
# private copy's publication slots and preparation identity.
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1, R-7
# and R-10; "The revisions of 14b9a757"): mesh_plan takes the preparation's one absolute deadline, CLOCK_MONOTONIC
# nanoseconds, last; mesh_transfers_start takes none (each wait bounded by MESH_PLATFORM_NS without progress);
# mesh_host_publish and mesh_host_arrived name a range by its plan and its declared transfer's index, where they took
# its first row's section; mesh_view_values maps every `run` value of the program and gives each value's offset in the
# mapping.  What must stay: each binding the header's declaration
# 2026-11-09 (design/compilation-composition-change-request.md items 1 and 2; design/aot-compilation-change-request.md
# A10, "Its output"): mesh_plan takes, after the calls, A10's order constraints between calls (`orders`, pairs of call
# indices) and each call's commuting chain (`chains`, its chain's first call), before the deadline.  What must stay:
# each binding the header's declaration
# 2026-11-09 (design/compilation-composition-change-request.md item 6, unit 6D, root's postcommit review): mesh_plan
# takes the program's issue `horizon` (block executions) before the deadline, the copies' lifetime operand apart from
# A10's `outstanding`.  What must stay: each binding the header's declaration
# 2026-11-09 (design/compilation-composition-change-request.md item 6, unit 6D): mesh_plan gives, last, the plan's
# results for native preparation (Planned), which mesh_planned_free releases.  What must stay: each binding the header's
# declaration
# 2026-11-09 (design/compilation-composition-change-request.md, "Complete replacement workgroup", program relations):
# mesh_plan takes each order pair's cause after `orders` (ORDER_FLOW, ORDER_CONFLICT) and gives each planned edge's
# cause (Planned `cause`).  What must stay: each binding the header's declaration
# 2026-11-09 (design/torch-capture-change-request.md item 2 and its appendix): mesh_plan takes no horizon; it derives
# it from each Function's `host` and `work` estimates and gives it as Planned `horizon`.  What must stay: each binding
# the header's declaration
# mesh-collective.h mesh_plan_compilation_load and mesh_local_load: the archive bytes' owner, released once the loaded result and
# its relations are; mesh_local_load's and mesh_plan_compilation_load's `analysis` the loading compilation's one analysis context
# (mesh-graph.h struct mesh_graph_context, rdma/mesh_graph.py Context.pointer)
RELEASE = C.CFUNCTYPE(None, C.c_void_p)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_nodes_read', C.c_int, [CONTEXT, C.POINTER(Nodes)]),
        ('mesh_view_whole', C.c_int, [C.c_void_p, Q]),
        ('mesh_plan', C.c_int, [CONTEXT, Q, Q, C.POINTER(Function), U, C.POINTER(U), U, C.POINTER(U), C.POINTER(U), U,
                                C.POINTER(U), Q, C.POINTER(PlanBinding)]),
        ('mesh_planned_free', None, [C.POINTER(Planned)]),
        ('mesh_plan_local', C.c_int, [C.POINTER(Function), U, C.POINTER(U), U, C.c_void_p, C.POINTER(U), U,
                                      C.c_void_p, C.POINTER(U), C.POINTER(Value), U, C.POINTER(Z),
                                      C.POINTER(LocalUse), Q, C.POINTER(Local)]),
        ('mesh_plan_analysis_retain', None, [C.POINTER(PlanAnalysis)]),
        ('mesh_plan_analysis_free', None, [C.POINTER(PlanAnalysis)]),
        ('mesh_plan_local_analyze', C.c_int, [C.POINTER(Function), U, C.POINTER(U), U, C.c_void_p, C.POINTER(U), U,
                                              C.c_void_p, C.POINTER(U), C.POINTER(Value), U, C.POINTER(Z),
                                              C.POINTER(LocalUse), C.POINTER(C.POINTER(PlanAnalysis))]),
        ('mesh_plan_environment', C.c_int, [CONTEXT, C.POINTER(Nodes), C.POINTER(PlanGeometry)]),
        ('mesh_plan_program_analyze', C.c_int, [C.POINTER(Nodes), C.POINTER(PlanGeometry), Q,
                                                C.POINTER(mesh_graph.BufferResult), C.POINTER(U), C.POINTER(Function), U, C.POINTER(Grouped), C.POINTER(U), U,
                                                C.POINTER(U), U, C.POINTER(Q), C.c_void_p, C.POINTER(U), U, C.c_void_p,
                                                C.POINTER(U), C.POINTER(Value), U, C.POINTER(Z), C.POINTER(LocalUse),
                                                C.POINTER(C.POINTER(PlanAnalysis))]),
        ('mesh_plan_local_complete', C.c_int, [C.POINTER(PlanAnalysis), Q, C.POINTER(Local)]),
        ('mesh_plan_cut', C.c_int, [C.POINTER(PlanGeometry), C.POINTER(PlanFlow), U,
                                  U, C.POINTER(PlanEndpoint), U, C.POINTER(mesh_graph.Interval),
                                  C.POINTER(PlanCuts), C.POINTER(C.POINTER(PlanCuts))]),
        ('mesh_plan_cut_free', None, [C.POINTER(PlanCuts)]),
        ('mesh_plan_flow_geometry', C.c_int, [C.POINTER(PlanFlow), C.POINTER(C.POINTER(PlanCuts)),
                                             C.POINTER(C.POINTER(Q)), C.POINTER(U),
                                             C.POINTER(C.POINTER(PlanFlowGeometry))]),
        ('mesh_plan_flow_geometry_retain', None, [C.POINTER(PlanFlowGeometry)]),
        ('mesh_plan_flow_geometry_free', None, [C.POINTER(PlanFlowGeometry)]),
        ('mesh_plan_schedule', C.c_int, [C.POINTER(PlanFlowGeometry), C.POINTER(Function), Z, C.POINTER(mesh_graph.Viewed),
                                           C.POINTER(Z), C.POINTER(U),
                                           C.POINTER(C.POINTER(PlanScheduleResult))]),
        ('mesh_plan_schedule_retain', None, [C.POINTER(PlanScheduleResult)]),
        ('mesh_plan_schedule_free', None, [C.POINTER(PlanScheduleResult)]),
        ('mesh_plan_storage', C.c_int, [C.POINTER(PlanAnalysis), C.POINTER(C.POINTER(PlanScheduleResult)),
                                       C.POINTER(mesh_graph.BufferResult), C.POINTER(U), C.POINTER(Z),
                                       C.POINTER(C.POINTER(Z)), C.POINTER(C.POINTER(Z)), C.POINTER(C.POINTER(U)),
                                       C.POINTER(C.POINTER(PlanStorage))]),
        ('mesh_plan_storage_ordered', C.c_int, [C.POINTER(PlanAnalysis), C.POINTER(C.POINTER(PlanScheduleResult)),
                                       C.POINTER(mesh_graph.BufferResult), C.POINTER(U), C.POINTER(Z),
                                       C.POINTER(C.POINTER(Z)), C.POINTER(C.POINTER(Z)), C.POINTER(C.POINTER(U)),
                                       C.POINTER(C.POINTER(Q)), C.POINTER(C.POINTER(PlanStorage))]),
        ('mesh_plan_storage_retain', None, [C.POINTER(PlanStorage)]),
        ('mesh_plan_storage_free', None, [C.POINTER(PlanStorage)]),
        ('mesh_plan_call_slot', C.c_int, [C.POINTER(PlanCompilation), U, U, Q, C.POINTER(PlanSlot)]),
        ('mesh_plan_program_compile_workspaces', C.c_int, [C.POINTER(PlanAnalysis), C.POINTER(C.POINTER(PlanScheduleResult)),
            C.POINTER(PlanStorage), C.POINTER(C.POINTER(PlanActors)), Z, C.POINTER(PlanWorkspace), C.POINTER(C.POINTER(PlanCompilation))]),
        ('mesh_plan_program_compile', C.c_int, [C.POINTER(PlanAnalysis), C.POINTER(C.POINTER(PlanScheduleResult)),
                                               C.POINTER(PlanStorage), C.POINTER(C.POINTER(PlanCompilation))]),
        ('mesh_plan_program_install', C.c_int, [CONTEXT, Q, C.POINTER(PlanCompilation), Q, C.POINTER(PlanBinding)]),
        ('mesh_plan_binding_free', None, [C.POINTER(PlanBinding)]),
        ('mesh_plan_compilation_retain', None, [C.POINTER(PlanCompilation)]),
        ('mesh_plan_compilation_free', None, [C.POINTER(PlanCompilation)]),
        ('mesh_plan_compilation_write', C.c_int, [C.POINTER(PlanCompilation), C.c_void_p, Z, C.POINTER(Z)]),
        ('mesh_plan_compilation_save', C.c_int, [C.POINTER(PlanCompilation), C.POINTER(C.c_void_p), C.POINTER(Z)]),
        ('mesh_plan_compilation_load', C.c_int, [C.c_void_p, C.c_void_p, Z, RELEASE, C.c_void_p,
                                                C.POINTER(C.POINTER(PlanCompilation))]),
        ('mesh_local_free', None, [C.POINTER(Local)]),
        ('mesh_declared_group', C.c_int, [C.POINTER(Declared), C.POINTER(Weight), U, U, U, C.POINTER(Grouped)]),
        ('mesh_plan_program', C.c_int, [CONTEXT, Q, Q, C.POINTER(Function), U, C.POINTER(Grouped), C.POINTER(U), U,
                                        C.POINTER(U), U, C.POINTER(Q), C.c_void_p, C.POINTER(U), U, C.c_void_p,
                                        C.POINTER(U), C.POINTER(Value), U, C.POINTER(Z), C.POINTER(LocalUse), Q,
                                        C.POINTER(PlanBinding)]),
        ('mesh_local_save', C.c_int, [C.POINTER(Local), C.POINTER(C.c_void_p), C.POINTER(Z)]),
        ('mesh_local_load', C.c_int, [C.c_void_p, C.c_void_p, Z, RELEASE, C.c_void_p, C.POINTER(Local)]),
        ('mesh_view_program', None, [CONTEXT, C.POINTER(PlanBinding), C.POINTER(Q), C.POINTER(Q), C.POINTER(View)]),
        ('mesh_plan_retire', None, [CONTEXT, Q]),
        ('mesh_view_values', None, [CONTEXT, C.POINTER(Function), C.POINTER(PlanBinding), U, C.POINTER(Q), C.POINTER(Q), C.POINTER(View)]),
        ('mesh_retire', None, [C.c_void_p, C.POINTER(PlanBinding)]),
        ('mesh_publications_clear', None, [C.c_void_p]),
        ('mesh_backing_bind', None, [CONTEXT, U, U, U, U]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_transfers_start', C.c_int, [CONTEXT, Q, Q]),
        ('mesh_host_publish', None, [CONTEXT, Q, U, Q]),
        ('mesh_host_publish_cut', C.c_int, [CONTEXT, Q, U, U, Q]),
        ('mesh_host_started', None, [CONTEXT, Q, U, Q]),
        ('mesh_host_arrived', Q, [CONTEXT, Q, U, Q]),
        ('mesh_host_sent', Q, [CONTEXT, Q, U, Q]),
        ('mesh_host_wait', C.c_int, [CONTEXT, Q, U, Q, U]),
        ('mesh_host_wait_fields', C.c_int, [CONTEXT, Q, U, Q, Z, C.POINTER(Q)]),
        ('mesh_plan_receive_wakes', C.c_int, [C.POINTER(PlanCompilation), U, U, Z, C.POINTER(Q)])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments

LIB.mesh_plan_program_analyze_routed.restype = C.c_int
LIB.mesh_plan_program_analyze_routed.argtypes = [*LIB.mesh_plan_program_analyze.argtypes[:-1],
                                                C.POINTER(D), C.POINTER(C.POINTER(PlanAnalysis))]
LIB.mesh_plan_program_analyze_chains.restype = C.c_int
LIB.mesh_plan_program_analyze_chains.argtypes = [*LIB.mesh_plan_program_analyze_routed.argtypes[:-1],
                                                C.POINTER(mesh_graph.ChainResult), C.POINTER(C.POINTER(PlanAnalysis))]
# mesh-collective.h mesh_strided_extents (design/plan-to-bridge-change-request.md R1; MPI 4.1 §6.1.3): a view's maximal
# contiguous byte ranges, the owning module's traversal.  What must stay: the header's declaration
EXTENT = C.CFUNCTYPE(None, C.c_void_p, Q, Q)
LIB.mesh_strided_extents.restype, LIB.mesh_strided_extents.argtypes = None, [C.POINTER(Strided), EXTENT, C.c_void_p]


def extents(view):
    found = []
    LIB.mesh_strided_extents(C.byref(view), EXTENT(lambda context, offset, length: found.append((offset, length))),
                             None)
    return found


def check(status):
    if status:
        raise OSError(status, os.strerror(status))


def strided(offset, element, pairs):
    words = (Q * max(2 * len(pairs), 1))(*(word for pair in pairs for word in pair))
    return Strided(offset, element, len(pairs), 0, words)


def function_tasks(made, tasks, before, senders, receivers, resources):
    encoded = tasks if isinstance(tasks, C.Array) and tasks._type_ is Task else (
        (Task * len(tasks))(*(Task(length, resource, 0) for length, resource in tasks)))
    dependencies = before if isinstance(before, C.Array) and before._type_ is mesh_graph.Dependence else (
        (mesh_graph.Dependence * max(len(before), 1))(*(mesh_graph.Dependence(a, b, 0) for a, b in before)))
    sender = senders if isinstance(senders, C.Array) and senders._type_ is U else (
        (U * max(len(senders), 1))(*senders))
    receiver = receivers if isinstance(receivers, C.Array) and receivers._type_ is U else (
        (U * max(len(receivers), 1))(*receivers))
    made.task, made.before, made.sender, made.receiver = encoded, dependencies, sender, receiver
    made.tasks, made.resources, made.befores = len(tasks), resources, len(before)
    return encoded, dependencies, sender, receiver


# mesh-collective.h struct mesh_function, struct mesh_declared, struct mesh_weight, struct mesh_value, struct
# mesh_dependence, struct mesh_task; mesh-graph.h struct mesh_strided; compiler/mesh_compiler/schedule.py dependences:
# a Dependence array is held as it is, its count its length; (from, to, lag, kinds) records are converted once
def function(declared, weights, values, dependences, outstanding, tasks=(), before=(), senders=(), receivers=(),
             resources=0):
    held = []

    held += [(Declared * max(len(declared), 1))(*[
                 Declared(source, destination, identity, kind, value, 0, strided(*sent), strided(*landed))
                 for source, destination, identity, kind, value, sent, landed in declared]),
             (Weight * max(len(weights), 1))(*[Weight(*weight) for weight in weights]),
             (Value * max(len(values), 1))(*[Value(Section(0, 0, size, 0, 0), *fields, 0)
                                             for size, *fields in values]),
             dependences if isinstance(dependences, C.Array) and dependences._type_ is Dependence else
             (Dependence * max(len(dependences), 1))(*[Dependence(*edge) for edge in dependences])]
    made = (Function * 1)(Function(*held[-4:], len(declared), len(values), len(dependences), 0, outstanding))
    if tasks:
        held += function_tasks(made[0], tasks, before, senders, receivers, resources)
    return made, held
