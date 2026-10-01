import ctypes as C
import hashlib
import os
import struct
import time

import numpy as np

U, Q, Z = C.c_uint32, C.c_uint64, C.c_size_t


class Operand(C.Structure):
    _fields_ = [('type', U), ('element_bytes', U), ('elements', Q)]


class Step(C.Structure):
    _fields_ = [('op', U), ('peer', U), ('round', U), ('padding', U), ('first', Q), ('piece', Operand)]


class LinkMap(C.Structure):
    """mesh-collective.h struct mesh_link_map: any number of nodes; `link` its links' storage (link_map
    builds one the map keeps; mesh_link_map_read allocates one mesh_link_map_free releases); `cost`, where
    not NULL, each directed pair's (alpha us, beta ns a byte) at [a * nodes + b]."""
    _fields_ = [('kind', U), ('nodes', U), ('links', U), ('link', C.POINTER(U * 2)), ('cost', C.POINTER(C.c_float * 2))]


def link_map(kind, nodes, pairs, cost=None):
    """A LinkMap of `kind` (KINDS) over `nodes` with the links `pairs`, its storage held by the map;
    `cost` {(a, b): (alpha, beta)} each directed pair's, where given (every pair's: the planner weighs it
    instead of the alpha and beta it is passed)."""
    storage = ((U * 2) * max(1, len(pairs)))(*[(U * 2)(a, b) for a, b in pairs])
    made = LinkMap(kind=KINDS.index(kind), nodes=nodes, links=len(pairs), link=C.cast(storage, C.POINTER(U * 2)))
    made._storage = storage
    if cost is not None:
        costs = ((C.c_float * 2) * (nodes * nodes))(*[(C.c_float * 2)(*cost.get((a, b), (0.0, 0.0)))
                                                       for a in range(nodes) for b in range(nodes)])
        made.cost, made._costs = C.cast(costs, C.POINTER(C.c_float * 2)), costs
    return made


class LinkState(C.Structure):
    """mesh-dataflow.h struct mesh_link_state: one directed link's stated alpha and beta, stated, up."""
    _fields_ = [('alpha', C.c_float), ('beta', C.c_float), ('stated', U), ('up', U)]


def contents_bytes(nodes):
    """mesh-dataflow.h mesh_link_contents_bytes: a snapshot of a table of `nodes` nodes."""
    return (16 + nodes * nodes * C.sizeof(LinkState) + nodes * 4 + 15) & ~15


class LinkTable:
    """The link table of a bridge region (mesh-dataflow.h: the link map as an operand of fixed shape),
    mapped: state(path) writes a link-map file's stated configuration into it, read() a snapshot and its
    epoch, map(snapshot, nodes) the planner's LinkMap of it over those nodes.  A table made here (create,
    as node `node`) takes what a bridge would write: observe(peer, up) its node's own link, report(origin,
    sequence, up) another node's report (the set of nodes its links reach up)."""

    def __init__(self, region=None, create=0, node=0):
        """The bridge's table of `region`, or with create=N one made here of N nodes as node `node`'s."""
        self.table, self.region, self.created = C.c_void_p(), region, create
        check(LIB.mesh_link_table_open(os.fsencode(region) if region else None, create, node, C.byref(self.table)))
        self.node, self.nodes = C.cast(self.table, C.POINTER(U))[1], C.cast(self.table, C.POINTER(U))[2]

    def observe(self, peer, up):
        return LIB.mesh_link_table_observe(self.table, peer, int(up))

    def report(self, origin, sequence, up):
        words = (Q * ((self.nodes + 63) // 64))()
        for v in up:
            words[v // 64] |= 1 << (v % 64)
        return LIB.mesh_link_table_report(self.table, origin, sequence, words)

    def state(self, path):
        check(LIB.mesh_link_table_state(self.table, os.fsencode(str(path))))

    def read(self):
        """A snapshot (mesh_link_contents, bytes) and its epoch; at(contents, a, b) its link a -> b."""
        contents = C.create_string_buffer(contents_bytes(self.nodes))
        return contents, LIB.mesh_link_table_read(self.table, contents)

    @staticmethod
    def at(contents, a, b):
        nodes = C.cast(contents, C.POINTER(U))[0]
        return LinkState.from_buffer(contents, 16 + (a * nodes + b) * C.sizeof(LinkState))

    @staticmethod
    def map(contents, nodes):
        n = len(nodes)
        made, pairs, cost = LinkMap(), ((U * 2) * max(1, n * n))(), ((C.c_float * 2) * max(1, n * n))()
        LIB.mesh_link_table_map(contents, (U * n)(*nodes), n, C.byref(made), pairs, cost)
        made._storage = (pairs, cost)
        return made

    def close(self):
        LIB.mesh_link_table_close(self.table)
        if self.created:
            C.CDLL(None).shm_unlink(os.fsencode(f'{self.region}.links'))


class Collective(C.Structure):
    """mesh-collective.h struct mesh_collective: what (an index of WHATS), how (an algorithm; on
    mesh_collective_choose's entry the bit set of the algorithms it may take, 0 every one), root,
    accumulator_bytes, contributors (a bit set of nodes, 64 a word; NULL every one: see bits), segments
    (a reduce-scatter's or all-gather's count a node, (Q * nodes)(...); NULL the equal cut)."""
    _fields_ = [*((k, U) for k in ('what', 'how', 'root', 'accumulator_bytes')), ('contributors', C.POINTER(Q)),
                ('segments', C.POINTER(Q))]


def bits(nodes, members):
    """A contributors bit set of `members` among `nodes`, for Collective.contributors (keep it alive)."""
    words = (Q * max(1, -(-nodes // 64)))()
    for m in members:
        words[m // 64] |= 1 << (m % 64)
    return words


SEND, REDUCE, COPY = range(3)
KINDS = ('mesh', 'ring', 'tree', 'graph')  # MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH
ALLREDUCE, BROADCAST = range(2)
WHATS = ('allreduce', 'broadcast', 'reduce', 'reduce_scatter', 'allgather')  # MESH_ALLREDUCE .. MESH_ALLGATHER
ALGORITHMS = ('direct', 'ring', 'tree', 'binomial')  # MESH_DIRECT .. MESH_BINOMIAL; MESH_UNAVAILABLE after them
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
for name, result, arguments in (
        ('mesh_link_map_read', C.c_int, [C.c_char_p, C.POINTER(LinkMap)]),
        ('mesh_link_map_free', None, [C.POINTER(LinkMap)]),
        ('mesh_link_table_open', C.c_int, [C.c_char_p, U, U, C.POINTER(C.c_void_p)]),
        ('mesh_link_table_close', None, [C.c_void_p]),
        ('mesh_link_table_state', C.c_int, [C.c_void_p, C.c_char_p]),
        ('mesh_link_table_read', Q, [C.c_void_p, C.c_void_p]),
        ('mesh_link_table_map', None, [C.c_void_p, C.POINTER(U), U, C.POINTER(LinkMap), C.c_void_p, C.c_void_p]),
        ('mesh_link_table_observe', C.c_int, [C.c_void_p, U, U]),
        ('mesh_link_table_report', C.c_int, [C.c_void_p, U, Q, C.POINTER(Q)]),
        ('mesh_collective_plan', U, [C.POINTER(LinkMap), U, Collective, Operand, C.POINTER(Step)]),
        ('mesh_collective_time', C.c_double, [C.POINTER(LinkMap), Collective, Operand, C.c_double, C.c_double]),
        ('mesh_collective_choose', Collective, [C.POINTER(LinkMap), Collective, Operand, C.c_double, C.c_double])):
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
    the bridge's registered window in place (alloc), and isend/irecv/iflush/test.  `handle(node, key)` is the handle a listen
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
        """A uint8 array over whole blocks of the bridge's registered window, which a registration takes
        in place (mesh_net_mem_alloc)."""
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


class UniqueId(C.Structure):
    """nccl.h ncclUniqueId."""
    _fields_ = [('internal', C.c_char * 128)]


class MeshConfig(C.Structure):
    """nccl.h ncclMeshConfig_t: ncclConfig_t member for member (size, magic, version, then its ints and its two
    names), the link table and nodes[r], rank r's node in it."""
    _fields_ = [('size', Z), ('magic', C.c_uint), ('version', C.c_uint), *((f'int{i}', C.c_int) for i in range(4)),
                ('netName', C.c_char_p), *((f'int{i}', C.c_int) for i in range(4, 6)), ('commName', C.c_char_p),
                *((f'int{i}', C.c_int) for i in range(6, 21)), ('links', P), ('nodes', C.POINTER(C.c_int))]


NCCL = None
TYPES = ('int8', 'uint8', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64')  # ncclDataType_t's first


def nccl():
    """libnccl-mesh beside this file, loaded at its first use and typed."""
    global NCCL
    if NCCL is None:
        NCCL = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libnccl-mesh.dylib'))
        for name, arguments in (
                ('ncclMeshLinksAttach', [C.c_char_p, PP]), ('ncclMeshLinksState', [P, C.c_char_p]),
                ('ncclMeshLinksRead', [P, P, P, P, P, C.POINTER(U), P]), ('ncclMeshLinksDetach', [P]),
                ('ncclMeshUniqueIdOf', [Q, C.POINTER(UniqueId)]),
                ('ncclCommInitRankConfig', [PP, C.c_int, UniqueId, C.c_int, C.POINTER(MeshConfig)]),
                ('ncclCommDestroy', [P]), ('ncclMemAlloc', [PP, Z]), ('ncclMemFree', [P]),
                ('ncclAllReduce', [P, P, Z, C.c_int, C.c_int, P, P])):
            getattr(NCCL, name).restype, getattr(NCCL, name).argtypes = C.c_int, arguments
        NCCL.ncclGetErrorString.restype, NCCL.ncclGetErrorString.argtypes = C.c_char_p, [C.c_int]
        NCCL.ncclGetLastError.restype, NCCL.ncclGetLastError.argtypes = C.c_char_p, [P]
    return NCCL


def nccl_check(result, comm=None):
    if result:
        raise RuntimeError(f'{NCCL.ncclGetErrorString(result).decode()}: {NCCL.ncclGetLastError(comm).decode()}')


class AllReduce:
    """One rank's all-reduce (a sum) of a `shape` array of `dtype` over the link map file `links` (the Xonotic
    planner's), on libnccl-mesh: the file stated into the link table of this node's bridge (`region`, else
    MESH_REGION), map node v the bridge node v, this rank the bridge's own node, the ranks' communicator named by
    the unique id of a key each derives from the file's bytes and `identity` (ncclMeshUniqueIdOf: no exchange).
    slot(t) is where this rank writes its contribution, one window allocation (ncclMemAlloc) every call's; a call
    is one ncclAllReduce in place on the NULL stream (synchronous), returning the sum, a copy.  `steps` lists what
    one call runs, one ncclAllReduce (the library plans it).  `invocations`, `depth` and a call's `deadline` are
    accepted and unused: nothing is prepared ahead, and no time ends a call."""

    def __init__(self, shape, dtype, links, invocations=1, depth=4, identity=1, region=None):
        if region:
            os.environ['MESH_REGION'] = region
        lib, self.shape, self.dtype, self.steps = nccl(), tuple(shape), np.dtype(dtype), ['ncclAllReduce']
        self.table, self.comm, self.buffer, node, read, unique = P(), P(), P(), U(), LinkMap(), UniqueId()
        nccl_check(lib.ncclMeshLinksAttach(os.fsencode(region) if region else None, C.byref(self.table)))
        nccl_check(lib.ncclMeshLinksState(self.table, os.fsencode(links)))
        nccl_check(lib.ncclMeshLinksRead(self.table, None, None, None, None, C.byref(node), None))
        check(LIB.mesh_link_map_read(os.fsencode(links), C.byref(read)))
        nodes, self.rank = read.nodes, node.value
        LIB.mesh_link_map_free(C.byref(read))
        with open(links, 'rb') as file:
            key = hashlib.blake2b(file.read() + identity.to_bytes(8, 'little'), digest_size=8).digest()
        nccl_check(lib.ncclMeshUniqueIdOf(int.from_bytes(key, 'little') or 1, C.byref(unique)))
        # NCCL_MESH_CONFIG_INITIALIZER: NCCL_API_MAGIC, NCCL_VERSION_CODE, every int NCCL_CONFIG_UNDEF_INT
        config = MeshConfig(size=C.sizeof(MeshConfig), magic=0xcafebeef, version=23203, links=self.table,
                            nodes=(C.c_int * nodes)(*range(nodes)),
                            **{k: -2 ** 31 for k, t in MeshConfig._fields_ if t is C.c_int})
        nccl_check(lib.ncclCommInitRankConfig(C.byref(self.comm), nodes, unique, self.rank, C.byref(config)))
        self.count = int(np.prod(self.shape))
        nccl_check(lib.ncclMemAlloc(C.byref(self.buffer), self.count * self.dtype.itemsize))
        self.array = np.frombuffer((C.c_char * (self.count * self.dtype.itemsize)).from_address(self.buffer.value),
                                   self.dtype).reshape(self.shape)

    def slot(self, invocation):
        return self.array

    def __call__(self, invocation, deadline=None):
        nccl_check(NCCL.ncclAllReduce(self.buffer, self.buffer, self.count, TYPES.index(self.dtype.name), 0,  # ncclSum
                                      self.comm, None), self.comm)
        return self.array.copy()

    def close(self):
        NCCL.ncclCommDestroy(self.comm)
        NCCL.ncclMemFree(self.buffer)
        NCCL.ncclMeshLinksDetach(self.table)
