import ctypes as C
import os
import struct
import time

import numpy as np

U, Q, Z = C.c_uint32, C.c_uint64, C.c_size_t


class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), *((k, Q) for k in ('client', 'send_off', 'send_bytes')),
                *((k, U) for k in ('rows', 'arena', 'wire', 'shared_pages', 'row_cursor', 'wire_cursor', 'bulk_cursor')),
                ('fd', C.c_int)]


class Header(C.Structure):
    _fields_ = [(k, U) for k in ('magic', 'version', 'pgsz', 'block', 'rows', 'node', 'qps', 'links')]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


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
    """mesh-collective.h struct mesh_collective: what (an index of WHATS), how (an algorithm; on
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
WHATS = ('allreduce', 'broadcast', 'reduce', 'reduce_scatter', 'allgather')  # MESH_ALLREDUCE .. MESH_ALLGATHER
ALGORITHMS = ('direct', 'ring', 'tree', 'binomial')  # MESH_DIRECT .. MESH_BINOMIAL; MESH_UNAVAILABLE after them
CANCELLED = 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_link_map_read', C.c_int, [C.c_char_p, C.POINTER(LinkMap)]),
        ('mesh_link_map_free', None, [C.POINTER(LinkMap)]),
        ('mesh_collective_plan', U, [C.POINTER(LinkMap), U, Collective, Operand, C.POINTER(Step)]),
        ('mesh_collective_time', C.c_double, [C.POINTER(LinkMap), Collective, Operand, C.c_double, C.c_double]),
        ('mesh_collective_choose', Collective, [C.POINTER(LinkMap), Collective, Operand, C.c_double, C.c_double]),
        ('mesh_section_create', C.c_int, [CONTEXT, Z, U, C.c_int, C.POINTER(Section)]),
        ('mesh_section_slice', C.c_int, [CONTEXT, Section, Z, Z, U, U, C.POINTER(Section)]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_collective_bind', C.c_int, [CONTEXT, C.POINTER(Step), U, U, Section, C.POINTER(Section), U, U,
                                           C.POINTER(Section)]),
        ('mesh_transfers_prepare', C.c_int, [CONTEXT, U, U, U]),
        ('mesh_host_inputs', C.c_int, [CONTEXT]),
        ('mesh_transfers_start', C.c_int, [CONTEXT]),
        ('mesh_host_publish', None, [CONTEXT, Section, U]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, U])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


class NetProperties(C.Structure):
    """mesh-net.h struct mesh_net_properties, ncclNetProperties_v12_t member for member."""
    _fields_ = [('name', C.c_char_p), ('pciPath', C.c_char_p), ('guid', Q), *((k, C.c_int) for k in (
        'ptrSupport', 'regIsGlobal', 'forceFlush', 'speed', 'port')), ('latency', C.c_float),
        *((k, C.c_int) for k in ('maxComms', 'maxRecvs', 'netDeviceType', 'netDeviceVersion')),
        ('ndevs', C.c_int), ('devs', C.c_int * 8), ('maxP2pBytes', Z), ('maxCollBytes', Z),
        ('maxMultiRequestSize', C.c_int), ('railId', C.c_int16), ('planeId', C.c_int16)]


P = C.c_void_p
PP = C.POINTER(C.c_void_p)
for name, arguments in (
        ('mesh_net_init', [PP, Q, P, P, P]), ('mesh_net_devices', [C.POINTER(C.c_int)]),
        ('mesh_net_get_properties', [C.c_int, C.POINTER(NetProperties)]),
        ('mesh_net_listen', [P, C.c_int, P, PP]), ('mesh_net_connect', [P, C.c_int, P, PP, PP]),
        ('mesh_net_accept', [P, PP, PP]), ('mesh_net_reg_mr', [P, P, Z, C.c_int, PP]), ('mesh_net_dereg_mr', [P, P]),
        ('mesh_net_isend', [P, P, Z, C.c_int, P, P, PP]),
        ('mesh_net_irecv', [P, C.c_int, PP, C.POINTER(Z), C.POINTER(C.c_int), PP, PP, PP]),
        ('mesh_net_iflush', [P, C.c_int, PP, C.POINTER(C.c_int), PP, PP]),
        ('mesh_net_test', [P, C.POINTER(C.c_int), C.POINTER(C.c_int)]),
        ('mesh_net_close_send', [P]), ('mesh_net_close_recv', [P]), ('mesh_net_close_listen', [P]),
        ('mesh_net_finalize', [P]), ('mesh_net_mem_alloc', [PP, Z]), ('mesh_net_mem_free', [P]), ('mesh_net_error', [])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = C.c_int, arguments
LIB.mesh_net_comm_counts.restype, LIB.mesh_net_comm_counts.argtypes = None, [P, C.POINTER(Q)]
NET_RESULTS = {0: 'ncclSuccess', 2: 'ncclSystemError', 3: 'ncclInternalError', 4: 'ncclInvalidArgument',
               5: 'ncclInvalidUsage', 6: 'ncclRemoteError', 7: 'ncclInProgress'}
NET_HANDLE_MAGIC, NET_HANDLE_BYTES = 0x4d4e4844, 128


class NetError(OSError):
    pass


def net_check(result):
    if result:
        error = LIB.mesh_net_error()
        raise NetError(error, f'{NET_RESULTS.get(result, result)}: {os.strerror(error)}')


class Net:
    """NCCL's network plugin (ncclNet_v12_t) over this node's bridge (mesh-net.h): init on `region`
    (MESH_REGION), the links as devices, listen/connect/accept by 128-byte handles, registrations of
    shared memory in place, and isend/irecv/iflush/test.  `handle(node, key)` is the handle a listen
    of that key on bridge node `node` gives, so peers deriving keys from one commId need no bootstrap."""

    def __init__(self, comm_id=0, region=None):
        if region:
            os.environ['MESH_REGION'] = region
        self.ctx, self.memory = C.c_void_p(), {}
        net_check(LIB.mesh_net_init(C.byref(self.ctx), comm_id, None, None, None))

    def devices(self):
        count = C.c_int()
        net_check(LIB.mesh_net_devices(C.byref(count)))
        return count.value

    def properties(self, dev):
        props = NetProperties()
        net_check(LIB.mesh_net_get_properties(dev, C.byref(props)))
        return props

    @staticmethod
    def handle(node, key):
        return C.create_string_buffer(struct.pack('<IIQ', NET_HANDLE_MAGIC, node, key), NET_HANDLE_BYTES)

    def listen(self, dev, key=0):
        """(handle, listen comm); a nonzero `key` names the listen."""
        handle, comm = self.handle(0, key) if key else C.create_string_buffer(NET_HANDLE_BYTES), C.c_void_p()
        net_check(LIB.mesh_net_listen(self.ctx, dev, handle, C.byref(comm)))
        return handle, comm

    def connect(self, dev, handle):
        comm = C.c_void_p()
        net_check(LIB.mesh_net_connect(self.ctx, dev, handle, C.byref(comm), None))
        return comm if comm.value else None

    def accept(self, listen):
        comm = C.c_void_p()
        net_check(LIB.mesh_net_accept(listen, C.byref(comm), None))
        return comm if comm.value else None

    def alloc(self, nbytes):
        """A uint8 array over a new shared-memory object a registration takes in place."""
        pointer = C.c_void_p()
        net_check(LIB.mesh_net_mem_alloc(C.byref(pointer), nbytes))
        array = np.frombuffer((C.c_char * nbytes).from_address(pointer.value), np.uint8)
        self.memory[array.ctypes.data] = pointer
        return array

    def free(self, array):
        net_check(LIB.mesh_net_mem_free(self.memory.pop(array.ctypes.data)))

    def regmr(self, comm, array):
        mhandle = C.c_void_p()
        net_check(LIB.mesh_net_reg_mr(comm, array.ctypes.data, array.nbytes, 1, C.byref(mhandle)))
        return mhandle

    def deregmr(self, comm, mhandle):
        net_check(LIB.mesh_net_dereg_mr(comm, mhandle))

    def isend(self, comm, array, mhandle, tag=0):
        """A request, or None when the comm's ring is full (post again)."""
        request = C.c_void_p()
        net_check(LIB.mesh_net_isend(comm, array.ctypes.data, array.nbytes, tag, mhandle, None, C.byref(request)))
        return request if request.value else None

    def irecv(self, comm, array, mhandle, tag=0):
        request, data, size, tags, handles = C.c_void_p(), (P * 1)(array.ctypes.data), (Z * 1)(array.nbytes), \
            (C.c_int * 1)(tag), (P * 1)(mhandle.value if mhandle else None)
        net_check(LIB.mesh_net_irecv(comm, 1, data, size, tags, handles, None, C.byref(request)))
        return request if request.value else None

    def iflush(self, comm):
        request = C.c_void_p()
        net_check(LIB.mesh_net_iflush(comm, 1, None, None, None, C.byref(request)))
        return request if request.value else None

    def test(self, request):
        """(done, bytes)."""
        done, size = C.c_int(), C.c_int()
        net_check(LIB.mesh_net_test(request, C.byref(done), C.byref(size)))
        return bool(done.value), size.value

    def wait(self, request, deadline=float('inf')):
        while True:
            done, size = self.test(request)
            if done:
                return size
            if time.monotonic() > deadline:
                raise TimeoutError('request not done by its deadline')

    @staticmethod
    def counts(comm):
        """The comm's bytes moved, requests completed, sends that waited for credit, requests posted."""
        out = (Q * 4)()
        LIB.mesh_net_comm_counts(comm, out)
        return dict(zip(('bytes', 'completions', 'credit_waits', 'posted'), out))

    def close(self, comm, kind):
        net_check(getattr(LIB, f'mesh_net_close_{kind}')(comm))

    def finalize(self):
        for pointer in self.memory.values():
            LIB.mesh_net_mem_free(pointer)
        self.memory.clear()
        net_check(LIB.mesh_net_finalize(self.ctx))


class LinkView(C.Structure):
    _fields_ = [('peer', U), ('phase', U), ('device', C.c_char * 32), ('bandwidth', Q)]


LIB.mesh_observe.argtypes = [C.c_char_p, C.POINTER(LinkView), U, C.POINTER(U)]


def links_of(region=None):
    """(this bridge's node, [(link, peer node, device)]) from its region (mesh_observe)."""
    views, node = (LinkView * 64)(), U()
    count = LIB.mesh_observe(os.fsencode(region) if region else None, views, 64, C.byref(node))
    if count < 0:
        raise OSError(-count, os.strerror(-count))
    return node.value, [(i, views[i].peer, views[i].device.decode()) for i in range(min(count, 64))]


def check(status):
    if status:
        raise OSError(status, os.strerror(status))


class Mesh:
    """One attach to this node's bridge region and the one program prepared on it: every collective
    and exchange bound on it (collective, bind), in the same order on every rank, prepared for
    `invocations` calls, storage ringing over `depth` slots, and started by `start` (an NCCL group is
    one).  `links` is an explicit link map (mesh-collective.h), its path or a LinkMap; its node v is
    the bridge's node members[v] (default v), and this rank is the map node of this bridge's.  Each
    collective takes the transfer identities from `identity` on, MESH_COLLECTIVE_STEPS(nodes) of them."""

    def __init__(self, links, invocations=1, depth=4, region=None, identity=1, members=None):
        self.context = Context()
        check(LIB.mesh_attach(C.byref(self.context), os.fsencode(region) if region else None))
        self.header = Header.from_address(self.context.M)
        self.owned = not isinstance(links, LinkMap)
        self.map = LinkMap() if self.owned else links
        if self.owned:
            check(LIB.mesh_link_map_read(os.fsencode(links), C.byref(self.map)))
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
        operand = Operand(ord(dtype.char), dtype.itemsize, int(np.prod(shape)))
        chosen = LIB.mesh_collective_choose(C.byref(self.map), Collective(what=WHATS.index(what), how=how, root=root),
                                            operand, alpha, beta)
        if chosen.how >= len(ALGORITHMS):
            raise ValueError(f'no algorithm of {how:#x} carries {what} of {operand.elements} elements on this map')
        steps = (Step * (4 * self.nodes))()
        count = LIB.mesh_collective_plan(C.byref(self.map), self.rank, chosen, operand, steps)
        return Steps(self, steps[:count], shape, dtype, op, chosen)

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
        if self.owned:
            LIB.mesh_link_map_free(C.byref(self.map))
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
            identity, mesh.identity = mesh.identity, mesh.identity + 4 * mesh.nodes
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
        bound = [Step(op=s.op, peer=mesh.members[s.peer], round=s.round, first=s.first, piece=s.piece) for s in self.steps]
        self.pieces = (Section * max(1, len(bound)))()
        check(LIB.mesh_collective_bind(context, (Step * max(1, len(bound)))(*bound), len(bound), identity, self.operand,
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
