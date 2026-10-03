"""The mesh from Python.  The planner (mesh-plan.h: link maps, operands, collectives, steps, their planning and
pricing) is the header's own, through _mesh_c (mesh_c_build.py compiles the header's declarations against it, so
no layout or prototype here is written by hand; `make mesh-c` builds it for this interpreter).  The host transfers
(Mesh, Steps, AllReduce: mesh.h's attach, sections and prepared transfers) are bound below by ctypes, mesh.h not
being a header a binder takes as written."""
import ctypes as C
import os
import sys
import time

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
ALGORITHMS = _named('MESH_', ('direct', 'ring', 'tree', 'binomial'))  # MESH_UNAVAILABLE after them
SEND, REDUCE, COPY = lib.MESH_STEP_SEND, lib.MESH_STEP_REDUCE, lib.MESH_STEP_COPY
ALLREDUCE, BROADCAST = lib.MESH_ALLREDUCE, lib.MESH_BROADCAST
CANCELLED = 2 ** 64 - 1


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


class Choice:
    """A collective (mesh-plan.h's struct mesh_collective, `c`) and the contributors' bit set it points at."""

    def __init__(self, c, keep=None):
        self.c, self._keep = c, keep
        self.what, self.how, self.root, self.accumulator_bytes = c.what, c.how, c.root, c.accumulator_bytes


def choose(lmap, what, piece, alpha=0.0, beta=0.0, how=0, root=0, accumulator_bytes=0, contributors=None):
    """mesh_collective_choose: the collective `what` (WHATS index) of least time for `piece` on `lmap` among the
    algorithms of the bit set `how` (0: every one), every link at its own cost or alpha and beta; `contributors`
    the nodes an all-reduce combines (None: every one).  Its `how` is len(ALGORITHMS) where the map carries none."""
    template = ffi.new('struct mesh_collective *')
    template.what, template.how, template.root, template.accumulator_bytes = what, how, root, accumulator_bytes
    words = None
    if contributors is not None:
        words = ffi.new('uint64_t[]', max(1, -(-lmap.nodes // 64)))
        for m in contributors:
            words[m // 64] |= 1 << (m % 64)
        template.contributors = words
    return Choice(lib.mesh_collective_choose(lmap.c, template[0], piece[0], alpha, beta), words)


def collective(what, how, root=0, accumulator_bytes=0):
    """A collective as given, for plan and time (no choice made)."""
    made = ffi.new('struct mesh_collective *')
    made.what, made.how, made.root, made.accumulator_bytes = what, how, root, accumulator_bytes
    return Choice(made[0], made)


def plan(lmap, rank, chosen, piece):
    """mesh_collective_plan: rank's steps of `chosen` (a Choice), in its order."""
    steps = ffi.new('struct mesh_step[]', lib.mesh_collective_steps(lmap.nodes))
    count = lib.mesh_collective_plan(lmap.c, rank, chosen.c, piece[0], steps)
    return [ffi.new('struct mesh_step *', steps[i]) for i in range(count)]


def time_of(lmap, chosen, piece, alpha=0.0, beta=0.0):
    """mesh_collective_time: every node's plan of `chosen` run in the alpha-beta model, in us (negative where it
    stops or mismatches)."""
    return lib.mesh_collective_time(lmap.c, chosen.c, piece[0], alpha, beta)


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
    members[v] (default v), and this rank is the map node of this bridge's.  Each collective takes the transfer
    identities from `identity` on, mesh_collective_steps(nodes) of them."""

    def __init__(self, links, invocations=1, depth=4, region=None, identity=1, members=None):
        self.context = Context()
        check(LIB.mesh_attach(C.byref(self.context), os.fsencode(region) if region else None))
        self.header = Header.from_address(self.context.M)
        self.map = links if isinstance(links, LinkMap) else read_link_map(links)
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

    def collective(self, what, shape, dtype, root=0, op=np.add, how=0, alpha=0.0, beta=0.0):
        """This rank's part of the collective `what` (WHATS) of a `shape` array of `dtype`, planned by
        mesh_collective_choose among the algorithms of the bit set `how` (0: every one) that the map
        carries, at a link cost of alpha us + beta ns a byte (0, 0: the first carried), and bound;
        `op` combines what a REDUCE brings in (any binary ufunc of the dtype)."""
        dtype = np.dtype(dtype)
        piece = operand(ord(dtype.char), dtype.itemsize, int(np.prod(shape)))
        chosen = choose(self.map, WHATS.index(what), piece, alpha, beta, how=how, root=root)
        if chosen.how >= len(ALGORITHMS):
            raise ValueError(f'no algorithm of {how:#x} carries {what} of {piece.elements} elements on this map')
        return Steps(self, plan(self.map, self.rank, chosen, piece), shape, dtype, op, chosen)

    def bind(self, steps, shape, dtype, identity=None):
        """The caller's own steps over one `shape` array of `dtype` (a point-to-point SEND or COPY),
        bound under transfer identity `identity` + each step's round (default the next collective's)."""
        return Steps(self, steps, shape, np.dtype(dtype), np.add, None, identity)

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

    def __init__(self, mesh, steps, shape, dtype, op, chosen, identity=None):
        self.mesh, self.rank, self.shape, self.dtype, self.op = mesh, mesh.rank, tuple(shape), dtype, op
        self.steps, self.chosen = list(steps), chosen
        self.direct = chosen is not None and chosen.how == ALGORITHMS.index('direct')
        if identity is None:
            identity, mesh.identity = mesh.identity, mesh.identity + lib.mesh_collective_steps(mesh.nodes)
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
        """Runs this rank's steps in plan order and returns the result: each SEND published, each
        receive awaited, a REDUCE's piece combined into the result by `op`.  A direct exchange
        combines into a copy, because its SEND may still be reading the slot, and copies each COPY's
        piece into it; a ring or tree works in place."""
        own = self.slot(invocation)
        total = own.copy() if self.direct else own
        flat, mine, received = total.reshape(-1), own.reshape(-1), iter(self.received)
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
            elif total is not own:
                flat[span] = mine[span]
        return total

    def close(self, linger=0.5):
        """Closes this collective's Mesh (the one of AllReduce)."""
        return self.mesh.close(linger)


def AllReduce(shape, dtype, links, invocations, depth=4, identity=1, region=None):
    """One rank's all-reduce of a typed array over an explicit link map (the Xonotic planner's), by
    the first algorithm the map carries (mesh_collective_choose at no cost: the direct exchange, a
    ring, a tree), prepared once for `invocations` calls and started: a Mesh of one collective."""
    mesh = Mesh(links, invocations, depth, region, identity)
    steps = mesh.collective('allreduce', shape, dtype)
    mesh.start()
    return steps
