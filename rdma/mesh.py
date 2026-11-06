import ctypes as C
import os

U, Q, Z, D = C.c_uint32, C.c_uint64, C.c_size_t, C.c_double


class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), ('client', Q), ('fd', C.c_int), ('padding', U), ('clock', Q),
                ('start', Q)]


class Header(C.Structure):
    _fields_ = [('magic', U), ('version', U), ('pgsz', U), ('block', U)]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


class Value(C.Structure):
    _fields_ = [('section', Section), ('kind', U), ('wire', U), ('state', U), ('begin', U), ('end', U), ('lag', U),
                ('run', U), ('shares', U), ('padding1', Q)]


# MPI 4.1 section 6.1.3, MPI_TYPE_CREATE_SUBARRAY's ndims: Strided's `dims` (size, stride) pairs at `pairs`
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
                ('lead', U), ('outstanding', Q), ('plan', Q)]


class View(C.Structure):
    _fields_ = [('address', C.c_void_p), ('bytes', Z)]


INVARIANT, EXECUTION = 0, 1
JOIN, PAGES = 0, 1
FROM_PRODUCED, TO_PRODUCED, FROM_SOURCE_READ, TO_SOURCE_READ = 1, 2, 4, 8
ABSENT = 2 ** 32 - 1
NAME = b'/mesh0'
PLATFORM_NS = 5000000000


# JGroups ViewId.java, "Lamport time of the view" and "Address of the creator of this view": a plan's view (`serial`,
# `sequencer`, `boot`), and the map read's (Nodes `serial`, `sequencer`, `boot`)
class State(C.Structure):
    _fields_ = [('copies', U), ('receives', U), ('ranges', U), ('releases', U), ('phases', Q), ('counts', Q),
                ('supplied', Q), ('status', Q), ('executions', Q), ('completions', Q), ('group', Q), ('epoch', Q),
                ('serial', Q), ('hops', Q), ('entries', Q), ('held', Q), ('refused', C.c_int32 * 8), ('nodes', Q),
                ('sender', U), ('outstanding', U), ('submitted', Q), ('delivered', Q), ('submissions', Q),
                ('broadcast', Q), ('arrived', Q), ('room', Q), ('deliveries', Q), ('submits', Q), ('announced', Q),
                ('waits', U), ('tells', U), ('syncs', Q), ('calls', U), ('closed', U), ('rows', Q),
                ('submit_room', Q), ('transfers', Q), ('sequencer', U), ('padding4', U), ('boot', Q),
                ('padding3', Q * 3)]


class Target(C.Structure):
    _fields_ = [('word', Q), ('count', U), ('stride', U)]


class Publication(C.Structure):
    _fields_ = [('sends', U), ('pages', U), ('phases', Q), ('count', Q), ('state', Q), ('targets', Target * 1),
                ('completion', Q)]


PUBLICATION_BYTES = 64
NODES_MOST = 32


class Nodes(C.Structure):
    _fields_ = [('serial', Q), ('count', U), ('self', U), ('node', U * NODES_MOST),
                ('capacity', (Q * NODES_MOST) * NODES_MOST), ('sequencer', U), ('padding', U), ('boot', Q)]


FAILED = 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_nodes_read', C.c_int, [CONTEXT, C.POINTER(Nodes)]),
        ('mesh_plan', C.c_int, [CONTEXT, Q, Q, C.POINTER(Function), U, C.POINTER(U), U, Q]),
        ('mesh_view_values', C.c_int, [CONTEXT, C.POINTER(Function), U, C.POINTER(Q), C.POINTER(View)]),
        ('mesh_view_release', None, [View]),
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
