"""The mesh from Python: mesh-plan.h through _mesh_c (mesh_c_build.py compiles the header's declarations against it, so
no layout or prototype here is written by hand; `make mesh-c` builds it for this interpreter).  The read side is the
node's bridge's: current() is the map it keeps (mesh_map) and wait() the map's serial (mesh_map_wait).  The planner is
the header's own: link maps, the trees every collective and message take, operands, collectives, steps, their
planning and pricing.  Collectives run through libnccl-mesh (mesh_mpi for NumPy, torch-mesh for torch)."""
import errno
import os
import sys
from types import SimpleNamespace

RDMA = os.path.dirname(os.path.abspath(__file__))
if RDMA not in sys.path:
    sys.path.insert(0, RDMA)
try:
    from _mesh_c import ffi, lib  # noqa: E402
except ImportError as missing:
    raise ImportError(f'_mesh_c is not built for {sys.executable} (Python {sys.version.split()[0]}): '
                      f'make -C {RDMA} mesh-c PYTHON={sys.executable}') from missing


def _named(prefix, names):
    """The header's constants `prefix`NAME for `names`, as a tuple in their values' order (each value its index)."""
    values = {name: getattr(lib, prefix + name.upper()) for name in names}
    ordered = tuple(sorted(names, key=values.get))
    assert [values[n] for n in ordered] == list(range(len(ordered))), prefix
    return ordered


KINDS = _named('MESH_LINKS_', ('mesh', 'ring', 'tree', 'graph'))
WHATS = _named('MESH_', ('allreduce', 'broadcast', 'reduce', 'reduce_scatter', 'allgather'))
SEND, REDUCE = lib.MESH_STEP_SEND, lib.MESH_STEP_REDUCE
ALLREDUCE, BROADCAST = lib.MESH_ALLREDUCE, lib.MESH_BROADCAST


class LinkMap:
    """mesh-plan.h's struct mesh_link_map (`c`, a pointer to it) of `kind` (KINDS) over `nodes`, the links `pairs`
    and each one's (alpha us, beta ns a byte) in `cost` (None: the caller's alpha and beta for every link), the
    storage held here; refused where mesh_link_map_check refuses it."""

    def __init__(self, kind, nodes, pairs, cost=None):
        pairs = [tuple(p) for p in pairs]
        self.kind, self.nodes, self.pairs, self.cost = kind, nodes, pairs, None if cost is None else [tuple(c) for c in cost]
        self._link = ffi.new('uint32_t[][2]', [list(p) for p in pairs] or [[0, 0]])
        self._cost = ffi.new('double[][2]', [list(c) for c in self.cost] or [[0.0, 0.0]]) if cost is not None else ffi.NULL
        self.c = ffi.new('struct mesh_link_map *')
        self.c.kind, self.c.nodes, self.c.links, self.c.link, self.c.cost = KINDS.index(kind), nodes, len(pairs), self._link, self._cost
        status = lib.mesh_link_map_check(self.c)
        if status:
            raise ValueError(f'a {kind} of {nodes} nodes with links {pairs} is not one its algorithms run on: {os.strerror(status)}')


def link_map(kind, nodes, pairs, cost=None):
    """A LinkMap (above)."""
    return LinkMap(kind, nodes, pairs, cost)


def wait(serial=0, seconds=0.0):
    """mesh_map_wait: the serial of the map this node's bridge keeps once it differs from `serial` (0: at once), else
    `serial` when `seconds` pass first.  OSError naming the bridge where it is unreachable."""
    made = ffi.new('uint64_t *', serial)
    status = lib.mesh_map_wait(made, int(seconds * 1e9))
    if status and status != errno.ETIMEDOUT:
        raise OSError(status, f"this node's bridge is unreachable: {os.strerror(status)}")
    return made[0]


def current(members=None):
    """mesh_map: the map this node's bridge keeps as of `serial` (wait()'s), over `members` (bridge node numbers, map
    node v the bridge node members[v]; None: every node its records hold): `map` the LinkMap, `nodes` and `names` each
    map node's bridge node and name, `node` this node's bridge's (mesh_observe).  OSError naming the bridge where it is
    unreachable; ValueError (LinkMap's) where no route reaches a member."""
    serial, asked, own = wait(), list(members or ()), ffi.new('uint32_t *')
    lib.mesh_observe(ffi.NULL, 0, own)
    room = ffi.new('struct mesh_link_map *', {'nodes': len(asked), 'links': len(asked) * (len(asked) - 1) // 2})
    while True:
        node, name = ffi.new('uint32_t[]', max(1, room.nodes)), ffi.new(f'char[][{lib.MESH_NAME_BYTES}]', max(1, room.nodes))
        link, cost = ffi.new('uint32_t[][2]', max(1, room.links)), ffi.new('double[][2]', max(1, room.links))
        made = ffi.new('struct mesh_link_map *', {'nodes': room.nodes, 'links': room.links, 'link': link, 'cost': cost})
        count = lib.mesh_map(ffi.new('uint32_t[]', asked) if asked else ffi.NULL, len(asked), node, name, made)
        if count != -errno.ENOSPC:
            break
        room = made
    if count < 0:
        raise OSError(-count, f"this node's bridge is unreachable: {os.strerror(-count)}")
    return SimpleNamespace(map=LinkMap(KINDS[made.kind], count, link[0:made.links], cost[0:made.links]), nodes=list(node[0:count]),
                           names=[ffi.string(n).decode() for n in name[0:count]], node=own[0], serial=serial)


def operand(type, element_bytes, elements):
    """mesh-plan.h's struct mesh_operand (a pointer that owns it; [0] is the struct a function takes)."""
    made = ffi.new('struct mesh_operand *')
    made.type, made.element_bytes, made.elements = type, element_bytes, elements
    return made


def trees(nodes, root, parent, log_weight):
    """mesh-plan.h's struct mesh_trees over the given trees (root[t], parent[t][v], log_weight[t]): plain data, `c` the
    struct's pointer and the arrays it points at kept beside it."""
    count = len(root)
    roots, parents = ffi.new('uint32_t[]', list(root) or [0]), ffi.new('uint32_t[]', [int(v) for p in parent for v in p] or [0])
    weights = ffi.new('double[]', [float(w) for w in log_weight] or [0.0])
    made = ffi.new('struct mesh_trees *', {'nodes': nodes, 'count': count, 'root': roots, 'parent': parents, 'log_weight': weights})
    return SimpleNamespace(c=made, nodes=nodes, count=count, root=list(root), parent=[list(p) for p in parent],
                           log_weight=list(log_weight), keep=(roots, parents, weights))


def pack(lmap, eps=0.1, most=4):
    """mesh_trees_pack: every node's trees packed into lmap's links (Garg-Konemann), at most `most` a root, as trees()."""
    n = lmap.nodes
    root, parent, weight = ffi.new('uint32_t[]', n * most), ffi.new('uint32_t[]', n * n * most), ffi.new('double[]', n * most)
    count = lib.mesh_trees_pack(lmap.c, eps, most, root, parent, weight)
    if not count:
        raise ValueError(f'mesh_trees_pack: no trees on this map ({lmap.kind} of {n})')
    return trees(n, [root[t] for t in range(count)], [[parent[t * n + v] for v in range(n)] for t in range(count)],
                 [weight[t] for t in range(count)])


def path(routes, a, b):
    """mesh_trees_path: the nodes a message from a to b crosses, a and b included ([] where b roots no tree)."""
    made = ffi.new('uint32_t[]', routes.nodes)
    return [made[i] for i in range(lib.mesh_trees_path(routes.c, a, b, made))]


def shortest(lmap):
    """mesh_trees_shortest: each node's in-tree on fewest links, as trees()."""
    n = lmap.nodes
    root, parent, weight = ffi.new('uint32_t[]', n), ffi.new('uint32_t[]', n * n), ffi.new('double[]', n)
    if not lib.mesh_trees_shortest(lmap.c, root, parent, weight):
        raise ValueError(f'mesh_trees_shortest: the map ({lmap.kind} of {n}) is not connected')
    return trees(n, [root[t] for t in range(n)], [[parent[t * n + v] for v in range(n)] for t in range(n)], [weight[t] for t in range(n)])


def compile(routes, what, root=0, whole=False, contributors=None):
    """mesh_compile: the collective `what` (WHATS index) along `routes` (trees()), a program: plain data, `c` the struct's
    pointer and the arrays it points at, `moves` every rank's moves as tuples (op, peer, round, buffer, part, parts)."""
    n, parts, moves = routes.nodes, ffi.new('uint32_t *'), ffi.new('uint32_t *')
    lib.mesh_compile_room(routes.c, parts, moves)
    segment, weight = ffi.new('uint32_t[]', parts[0]), ffi.new('double[]', parts[0])
    first, move = ffi.new('uint32_t[]', n + 1), ffi.new('struct mesh_move[]', moves[0])
    words = None
    if contributors is not None:
        words = ffi.new('uint64_t[]', max(1, -(-n // 64)))
        for m in contributors:
            words[m // 64] |= 1 << (m % 64)
    made = ffi.new('struct mesh_program *')
    if lib.mesh_compile(routes.c, what, root, int(whole), words if words is not None else ffi.NULL, made, segment, weight, first, move):
        raise ValueError(f'mesh_compile refused {WHATS[what]} root {root} whole {whole}')
    return SimpleNamespace(c=made, what=what, root=root, nodes=n, scratch=made.scratch,
                           moves=[[(m.op, m.peer, m.round, m.buffer, m.part, m.parts) for m in (move[k] for k in range(first[r], first[r + 1]))]
                                  for r in range(n)],
                           keep=(segment, weight, first, move, words, routes))


def steps(program, rank, piece):
    """mesh_program_steps: rank's steps of `program` for the operand `piece`, in its order."""
    made = ffi.new('struct mesh_step[]', max(1, program.c.first[program.nodes]))
    count = lib.mesh_program_steps(program.c, rank, piece[0], made)
    return [ffi.new('struct mesh_step *', made[i]) for i in range(count)]


def depth(program):
    """The program's dependent crossings: its last round plus one (a round's receives wait on lower rounds' SENDs)."""
    return 1 + max((m[2] for moves in program.moves for m in moves), default=-1)


def program_time(lmap, program, piece, alpha=0.0, beta=0.0):
    """mesh_program_time: every rank's steps run in the alpha-beta model, in us (negative where it stops or mismatches)."""
    return lib.mesh_program_time(lmap.c, program.c, piece[0], alpha, beta)
