"""The mesh from Python.  The planner (mesh-plan.h: link maps, the trees every collective and message take, operands,
collectives, steps, their planning and pricing) is the header's own, through _mesh_c (mesh_c_build.py compiles the header's declarations against it, so
no layout or prototype here is written by hand; `make mesh-c` builds it for this interpreter).  The host transfers
(Mesh, Steps, AllReduce: mesh.h's attach, sections and prepared transfers) are bound below by ctypes, mesh.h not
being a header a binder takes as written."""
import ctypes as C
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

RDMA = os.path.dirname(os.path.abspath(__file__))
if RDMA not in sys.path:
    sys.path.insert(0, RDMA)
try:
    from _mesh_c import ffi, lib  # noqa: E402
except ImportError as missing:
    raise ImportError(f'_mesh_c is not built for {sys.executable} (Python {sys.version.split()[0]}): '
                      f'make -C {RDMA} mesh-c PYTHON={sys.executable}') from missing

U, Q, Z = C.c_uint32, C.c_uint64, C.c_size_t


def _named(prefix, names):
    """The header's constants `prefix`NAME for `names`, as a tuple in their values' order (each value its index)."""
    values = {name: getattr(lib, prefix + name.upper()) for name in names}
    ordered = tuple(sorted(names, key=values.get))
    assert [values[n] for n in ordered] == list(range(len(ordered))), prefix
    return ordered


KINDS = _named('MESH_LINKS_', ('mesh', 'ring', 'tree', 'graph'))
WHATS = _named('MESH_', ('allreduce', 'broadcast', 'reduce', 'reduce_scatter', 'allgather'))
SEND, REDUCE, COPY = lib.MESH_STEP_SEND, lib.MESH_STEP_REDUCE, lib.MESH_STEP_COPY
ALLREDUCE, BROADCAST = lib.MESH_ALLREDUCE, lib.MESH_BROADCAST
CANCELLED = 2 ** 64 - 1


class LinkMap:
    """mesh-plan.h's struct mesh_link_map (`c`, a pointer to it) of `kind` (KINDS) over `nodes`, the links `pairs`
    and each one's (alpha us, beta ns a byte) in `cost` (None: the caller's alpha and beta for every link), the
    storage held here; refused where mesh_link_map_check refuses it, except that with `apart` a graph whose links
    leave its nodes apart is kept (a subgroup's: the library joins it along the paths of the widest communicator alive
    over its ranks, nccl.h)."""

    def __init__(self, kind, nodes, pairs, cost=None, apart=False):
        pairs = [tuple(p) for p in pairs]
        self.kind, self.nodes, self.pairs, self.cost = kind, nodes, pairs, None if cost is None else [tuple(c) for c in cost]
        self._link = ffi.new('uint32_t[][2]', [list(p) for p in pairs] or [[0, 0]])
        self._cost = ffi.new('double[][2]', [list(c) for c in self.cost] or [[0.0, 0.0]]) if cost is not None else ffi.NULL
        self.c = ffi.new('struct mesh_link_map *')
        self.c.kind, self.c.nodes, self.c.links, self.c.link, self.c.cost = KINDS.index(kind), nodes, len(pairs), self._link, self._cost
        status = lib.mesh_link_map_check(self.c)
        if status and apart and kind == 'graph' and all(a != b and max(a, b) < nodes for a, b in pairs):
            status = 0
        if status:
            raise ValueError(f'a {kind} of {nodes} nodes with links {pairs} is not one its algorithms run on: {os.strerror(status)}')


def link_map(kind, nodes, pairs, cost=None, apart=False):
    """A LinkMap (above)."""
    return LinkMap(kind, nodes, pairs, cost, apart)


def read_link_map(path):
    """The link map a text file holds (mesh-plan.h's text form), as a LinkMap."""
    made = ffi.new('struct mesh_link_map *')
    status = lib.mesh_link_map_read(os.fsencode(str(path)), made)
    if status:
        raise ValueError(f'{path}: not a link map: {os.strerror(status)}')
    try:
        pairs = [(made.link[i][0], made.link[i][1]) for i in range(made.links)]
        cost = [(made.cost[i][0], made.cost[i][1]) for i in range(made.links)] if made.cost != ffi.NULL else None
        return LinkMap(KINDS[made.kind], made.nodes, pairs, cost)
    finally:
        lib.mesh_link_map_free(made)


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


class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), *((k, Q) for k in ('client', 'send_off', 'send_bytes')),
                *((k, U) for k in ('rows', 'arena', 'wire', 'shared_pages', 'row_cursor', 'wire_cursor', 'bulk_cursor')),
                ('fd', C.c_int)]


class Header(C.Structure):
    _fields_ = [(k, U) for k in ('magic', 'version', 'pgsz', 'block', 'rows', 'node', 'qps', 'links')]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


LIB = C.CDLL(os.path.join(RDMA, 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_section_create', C.c_int, [CONTEXT, Z, U, C.c_int, C.POINTER(Section)]),
        ('mesh_section_slice', C.c_int, [CONTEXT, Section, Z, Z, U, U, C.POINTER(Section)]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_collective_bind', C.c_int, [CONTEXT, C.c_void_p, U, U, Section, C.POINTER(Section), U, U, C.POINTER(Section)]),
        ('mesh_transfers_prepare', C.c_int, [CONTEXT, U, U, U]),
        ('mesh_host_inputs', C.c_int, [CONTEXT]),
        ('mesh_transfers_start', C.c_int, [CONTEXT]),
        ('mesh_host_publish', None, [CONTEXT, Section, U]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, U])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))


class Mesh:
    """One attach to this node's bridge region and the one program prepared on it: every collective
    and exchange bound on it (collective, bind), in the same order on every rank, prepared for
    `invocations` calls, storage ringing over `depth` slots, and started by `start` (an NCCL group is
    one).  `links` is a LinkMap (or a link map file's path, read_link_map); its node v is the bridge's node
    members[v] (default v), and this rank is the map node of this bridge's.  `routes` are the trees its collectives
    are compiled along (trees(); None: pack(links), made once here).  Each collective takes the transfer identities
    from `identity` on, as many as its program has moves."""

    def __init__(self, links, invocations=1, depth=4, region=None, identity=1, members=None, routes=None):
        self.context = Context()
        check(LIB.mesh_attach(C.byref(self.context), os.fsencode(region) if region else None))
        self.header = Header.from_address(self.context.M)
        self.map = links if isinstance(links, LinkMap) else read_link_map(links)
        self.routes = routes if routes is not None else pack(self.map)
        self.nodes, self.members = self.map.nodes, list(members or range(self.map.nodes))
        self.rank = self.members.index(self.header.node)
        self.invocations, self.depth, self.identity = invocations, min(depth, invocations), identity
        self.block = self.header.pgsz * self.header.block

    def ring(self, nbytes, offset=0):
        """A registered section of `depth` slots of `offset` + `nbytes` in whole blocks: the section, a
        slot's pages, and the slots as bytes."""
        pages = -(-(offset + nbytes) // self.block) * self.header.block
        section = Section()
        check(LIB.mesh_section_create(C.byref(self.context), self.depth * pages * self.header.pgsz, 1, 1, C.byref(section)))
        memory = (C.c_char * (self.depth * pages * self.header.pgsz)).from_address(
            LIB.mesh_section_address(C.byref(self.context), section, 0))
        return section, pages, np.frombuffer(memory, np.uint8).reshape(self.depth, -1)

    def collective(self, what, shape, dtype, root=0, op=np.add):
        """This rank's part of the collective `what` (WHATS) of a `shape` array of `dtype`, compiled along this Mesh's
        trees (compile; reduce-scatter + all-gather for an all-reduce, which binds one operand), bound; `op` combines
        what a REDUCE brings in (any binary ufunc of the dtype)."""
        dtype = np.dtype(dtype)
        piece = operand(ord(dtype.char), dtype.itemsize, int(np.prod(shape)))
        program = compile(self.routes, WHATS.index(what), root)
        return Steps(self, steps(program, self.rank, piece), shape, dtype, op, identity=None, moves=program.c.first[self.nodes])

    def bind(self, steps, shape, dtype, identity=None):
        """The caller's own steps over one `shape` array of `dtype` (a point-to-point SEND or COPY),
        bound under transfer identity `identity` + each step's round (default the next collective's)."""
        return Steps(self, steps, shape, np.dtype(dtype), np.add, identity)

    def start(self):
        """Prepares every bound transfer for the invocations, gives each receive its completion words
        and starts the transfers: the peers' bridges pair."""
        context = C.byref(self.context)
        check(LIB.mesh_transfers_prepare(context, 1, self.invocations, self.depth))
        check(LIB.mesh_host_inputs(context))
        check(LIB.mesh_transfers_start(context))

    def close(self, linger=0.5):
        """Holds the pair open while the peer's last receive lands, then retires this client."""
        time.sleep(linger)
        return LIB.mesh_detach(C.byref(self.context))


class Steps:
    """One rank's steps of one collective or exchange over one typed operand, bound on its Mesh
    (mesh_collective_bind): the operand's ring of slots, and one received section a REDUCE, placed at
    its SEND piece's offset within a block so that both ends cut the piece in the same chunks."""

    def __init__(self, mesh, steps, shape, dtype, op, identity=None, moves=None):
        self.mesh, self.rank, self.shape, self.dtype, self.op = mesh, mesh.rank, tuple(shape), dtype, op
        self.steps = list(steps)
        if identity is None:
            identity, mesh.identity = mesh.identity, mesh.identity + (moves or len(self.steps)) + 1
        context = C.byref(mesh.context)
        self.bytes = int(np.prod(shape)) * dtype.itemsize
        self.operand, pages, self.slots = mesh.ring(self.bytes)
        received, self.received = [], []
        for step in self.steps:
            if step.op == REDUCE:
                size, offset = step.piece.elements * step.piece.element_bytes, step.first * step.piece.element_bytes % mesh.block
                section, step_pages, slots = mesh.ring(size, offset)
                received.append(Section())
                check(LIB.mesh_section_slice(context, section, offset, size, mesh.depth, step_pages, C.byref(received[-1])))
                self.received.append((slots, offset, size))
        bound = ffi.new('struct mesh_step[]', max(1, len(self.steps)))
        for i, s in enumerate(self.steps):
            bound[i] = s[0]
            bound[i].peer = mesh.members[s.peer]
        self.pieces = (Section * max(1, len(self.steps)))()
        check(LIB.mesh_collective_bind(context, int(ffi.cast('uintptr_t', bound)), len(self.steps), identity, self.operand,
                                      (Section * max(1, len(received)))(*received), mesh.depth, pages, self.pieces))

    def slot(self, invocation):
        """Where this rank writes its contribution to `invocation`, in registered pages."""
        return self.slots[invocation % self.mesh.depth, :self.bytes].view(self.dtype).reshape(self.shape)

    def __call__(self, invocation, deadline=float('inf')):
        """Runs this rank's steps in plan order and returns the result, in place: each SEND published, each
        receive awaited, a REDUCE's piece combined into the result by `op`."""
        total = self.slot(invocation)
        flat, received = total.reshape(-1), iter(self.received)
        context, row = C.byref(self.mesh.context), invocation % self.mesh.depth
        for step, piece in zip(self.steps, self.pieces):
            if step.op == SEND:
                LIB.mesh_host_publish(context, piece, invocation)
                continue
            while not (arrived := LIB.mesh_host_arrived(context, piece, invocation)):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'invocation {invocation} step {step.op} from {step.peer}')
            if arrived == CANCELLED:
                raise ConnectionAbortedError(f'invocation {invocation} cancelled on the link to {step.peer}')
            span = slice(step.first, step.first + step.piece.elements)
            if step.op == REDUCE:
                slots, offset, size = next(received)
                self.op(flat[span], slots[row, offset:offset + size].view(self.dtype), out=flat[span])
        return total

    def close(self, linger=0.5):
        """Closes this collective's Mesh (the one of AllReduce)."""
        return self.mesh.close(linger)


def AllReduce(shape, dtype, links, invocations, depth=4, identity=1, region=None):
    """One rank's all-reduce of a typed array over an explicit link map (the Xonotic planner's), along the map's
    packed trees, prepared once for `invocations` calls and started: a Mesh of one collective."""
    mesh = Mesh(links, invocations, depth, region, identity)
    steps = mesh.collective('allreduce', shape, dtype)
    mesh.start()
    return steps
