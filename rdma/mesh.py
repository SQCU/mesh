import ctypes as C
import os

U, Q = C.c_uint32, C.c_uint64


class Operand(C.Structure):
    _fields_ = [('type', U), ('element_bytes', U), ('elements', Q)]


class Step(C.Structure):
    _fields_ = [('op', U), ('peer', U), ('round', U), ('padding', U), ('first', Q), ('piece', Operand)]


class LinkMap(C.Structure):
    """mesh-collective.h struct mesh_link_map: any number of nodes; `link` its links' storage (link_map
    builds one the map keeps; mesh_link_map_read allocates one mesh_link_map_free releases)."""
    _fields_ = [('kind', U), ('nodes', U), ('links', U), ('link', C.POINTER(U * 2))]


def link_map(kind, nodes, pairs):
    """A LinkMap of `kind` (KINDS) over `nodes` with the links `pairs`, its storage held by the map."""
    storage = ((U * 2) * max(1, len(pairs)))(*[(U * 2)(a, b) for a, b in pairs])
    made = LinkMap(kind=KINDS.index(kind), nodes=nodes, links=len(pairs), link=C.cast(storage, C.POINTER(U * 2)))
    made._storage = storage
    return made


class Collective(C.Structure):
    """mesh-collective.h struct mesh_collective: what (ALLREDUCE, BROADCAST), how (an algorithm; on
    mesh_collective_choose's entry the bit set of the algorithms it may take, 0 every one), root,
    accumulator_bytes, contributors (a bit set of nodes, 64 a word; NULL every one: see bits)."""
    _fields_ = [*((k, U) for k in ('what', 'how', 'root', 'accumulator_bytes')), ('contributors', C.POINTER(Q))]


def bits(nodes, members):
    """A contributors bit set of `members` among `nodes`, for Collective.contributors (keep it alive)."""
    words = (Q * max(1, -(-nodes // 64)))()
    for m in members:
        words[m // 64] |= 1 << (m % 64)
    return words


SEND, REDUCE, COPY = range(3)
KINDS = ('mesh', 'ring', 'tree')  # MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE
ALLREDUCE, BROADCAST = range(2)
ALGORITHMS = ('direct', 'ring', 'tree', 'binomial')  # MESH_DIRECT .. MESH_BINOMIAL; MESH_UNAVAILABLE after them
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
for name, result, arguments in (
        ('mesh_link_map_read', C.c_int, [C.c_char_p, C.POINTER(LinkMap)]),
        ('mesh_link_map_free', None, [C.POINTER(LinkMap)]),
        ('mesh_collective_plan', U, [C.POINTER(LinkMap), U, Collective, Operand, C.POINTER(Step)]),
        ('mesh_collective_time', C.c_double, [C.POINTER(LinkMap), Collective, Operand, C.c_double, C.c_double]),
        ('mesh_collective_choose', Collective, [C.POINTER(LinkMap), Collective, Operand, C.c_double, C.c_double])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments
