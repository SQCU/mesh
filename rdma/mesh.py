"""The mesh from Python: mesh-plan.h through _mesh_c (mesh_c_build.py compiles the header's declarations against it, so
no layout or prototype here is written by hand; `make mesh-c` builds it for this interpreter).  The read side is the
node's bridge's: current() is the map it keeps (mesh_map) and wait() the map's serial (mesh_map_wait).  The planner is
the header's own: link maps, the trees every collective and message take, operands, collectives, steps, their
planning and pricing on the wire libnccl-mesh's sessions run (wire(), mesh_program_time: as the library runs a program,
every position a whole slot), and the window a session takes on each member and on each relay its routes cross
(session_bytes, relay_bytes).  Collectives run through libnccl-mesh (mesh_mpi for NumPy, torch-mesh for torch); lease() is its
ncclMemAlloc, the memory they move taken from the node's bridge's window (rdma/NCCL.md "Memory")."""
import errno
import os
import re
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
    and each one's (alpha us, beta ns a byte) in `cost` (None: the caller's alpha and beta for every link; a NaN alpha:
    that link's), with `relays` more nodes (numbered from `nodes`, holding no rank), each node's `mark` (None: its
    number; the bridges mark a node by its node number) and the routing table `next` (each node b's in-tree, next[v * N
    + b] v's parent in it, for N = nodes + relays; None: none), the storage held here; refused where
    mesh_link_map_check refuses it (no map).  A graph need not join its nodes (the bridges' records while a cable is
    out); joined() says whether it does, and the trees and collectives run only where it does."""

    def __init__(self, kind, nodes, pairs, cost=None, relays=0, next=None, mark=None):
        pairs = [tuple(p) for p in pairs]
        self.kind, self.nodes, self.pairs, self.cost = kind, nodes, pairs, None if cost is None else [tuple(c) for c in cost]
        self.relays = relays
        self.next, self.mark = (None if x is None else [int(v) for v in x] for x in (next, mark))
        self._link = ffi.new('uint32_t[][2]', [list(p) for p in pairs] or [[0, 0]])
        self._cost = ffi.new('double[][2]', [list(c) for c in self.cost] or [[0.0, 0.0]]) if cost is not None else ffi.NULL
        self._next = ffi.new('uint32_t[]', self.next) if self.next else ffi.NULL
        self._mark = ffi.new('uint32_t[]', self.mark) if self.mark else ffi.NULL
        self.c = ffi.new('struct mesh_link_map *')
        self.c.kind, self.c.nodes, self.c.links, self.c.link, self.c.cost = KINDS.index(kind), nodes, len(pairs), self._link, self._cost
        self.c.relays, self.c.next, self.c.mark = relays, self._next, self._mark
        status = lib.mesh_link_map_check(self.c)
        if status:
            raise ValueError(f'a {kind} of {nodes} nodes and {relays} relays with links {pairs} is not one its algorithms run on: '
                             f'{os.strerror(status)}')


def link_map(kind, nodes, pairs, cost=None, relays=0, next=None, mark=None):
    """A LinkMap (above)."""
    return LinkMap(kind, nodes, pairs, cost, relays, next, mark)


def joined(lmap):
    """mesh_link_map_joined: None where lmap joins every two of its nodes (a graph's links connect them; with a table
    every route between two of them reaches), else the first two it does not join, (a, b), a the lower."""
    apart = ffi.new('uint32_t[2]')
    status = lib.mesh_link_map_joined(lmap.c, apart)
    if status == errno.ENOTCONN:
        return apart[0], apart[1]
    if status:
        raise ValueError(f'mesh_link_map_joined: {os.strerror(status)}')
    return None


def _apart(lmap, named, nodes):
    """ValueError naming the first two of `nodes` (map node v's name nodes[v]) lmap does not join, where it does not:
    the map a communicator of them reads, which no collective runs on."""
    pair = joined(lmap)
    if pair is not None:
        raise ValueError(f'{named}: no route of the map joins {nodes[pair[0]]} and {nodes[pair[1]]}')


def routes(lmap):
    """mesh_routes: the routing table of lmap's links by its marks (each node's in-tree of shortest routes, a parent
    among equals the one of the least hash of its mark, the parent's and the root's: a route moves only where the
    shortest routes between its ends change), as a flat list N x N (LinkMap's `next`)."""
    N = lmap.nodes + lmap.relays
    made = ffi.new('uint32_t[]', max(1, N * N))
    status = lib.mesh_routes(lmap.c, made)
    if status:
        raise ValueError(f'mesh_routes: {os.strerror(status)}')
    return [made[k] for k in range(N * N)]


def route(lmap, a, b):
    """mesh_route_path: the nodes a message between a and b crosses (the one from b to a reversed), a and b included
    ([] where none)."""
    made = ffi.new('uint32_t[]', lmap.nodes + lmap.relays + 1)
    return [made[i] for i in range(lib.mesh_route_path(lmap.c, a, b, made))]


def _members(whole, members, current):
    """mesh_map_members (current False) or mesh_map (current True): the room grown until it fits."""
    asked = list(members or ())
    room = ffi.new('struct mesh_link_map *', {'nodes': max(1, len(asked)), 'links': max(1, len(asked) * (len(asked) - 1) // 2)})
    while True:
        node, name = ffi.new('uint32_t[]', max(1, room.nodes)), ffi.new(f'char[][{lib.MESH_NAME_BYTES}]', max(1, room.nodes))
        link, cost = ffi.new('uint32_t[][2]', max(1, room.links)), ffi.new('double[][2]', max(1, room.links))
        table, mark = ffi.new('uint32_t[]', max(1, room.nodes * room.nodes)), ffi.new('uint32_t[]', max(1, room.nodes))
        made = ffi.new('struct mesh_link_map *', {'nodes': room.nodes, 'links': room.links, 'link': link, 'cost': cost,
                                                  'next': table, 'mark': mark})
        given = ffi.new('uint32_t[]', asked) if asked else ffi.NULL
        if current:
            count = lib.mesh_map(given, len(asked), node, name, made)
        else:
            count = lib.mesh_map_members(whole.c, given, len(asked), node, made)
        if count != -errno.ENOSPC:
            break
        room = made
    if count < 0:
        return count, None, None, None
    N = count + made.relays
    lmap = LinkMap(KINDS[made.kind], count, link[0:made.links], [tuple(c) for c in cost[0:made.links]], made.relays,
                   table[0:N * N], mark[0:N])
    return count, lmap, list(node[0:N]), [ffi.string(n).decode() for n in name[0:N]]


def members(whole, among, marks=None):
    """mesh_map_members: the map a communicator of the whole map's nodes `among` (map node i the whole map's among[i])
    reads, the whole map's other nodes its relays, its marks and table the whole map's (routes(whole) where it has
    none), so two members are linked where their route crosses no other member, at that route's cost: the rule mesh_map
    reads the bridges' records by, for a planner pricing a record.  `marks` (None: the whole map's own) each whole-map
    node's mark, as the bridges mark each node by its node number, so the table is the bridges' (a whole map given
    links and costs alone, link_map('graph', N, links, costs), takes them here; its table, made by other marks, is made
    again by these)."""
    if marks is not None:
        whole = LinkMap(whole.kind, whole.nodes, whole.pairs, whole.cost, whole.relays, None, marks)
    among = None if among is None else list(among)
    count, lmap, nodes, _ = _members(whole, among, False)
    if count < 0:
        raise ValueError(f'mesh_map_members over {among}: {os.strerror(-count)}')
    _apart(lmap, f'mesh_map_members over {among}', [f'whole-map node {v}' for v in nodes])
    return lmap


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
    node v the bridge node members[v], the other nodes its relays; None: every node its records hold, no relays):
    `map` the members' map as libnccl-mesh reads it, the one map to price a program on (its relays, a link a cable, at
    its cost, `mark` each node's bridge node number, `next` the routing table the bridges forward along; members(map,
    ...) of the whole map is the map over any of them), `nodes` and `names` each map node's (then each relay's) bridge
    node and name, `node` this node's bridge's (mesh_observe).  The whole map (members None) is the records as they are,
    a node no cable joins among its nodes (joined() names two it leaves apart).  OSError naming the bridge where it is
    unreachable; ValueError naming two members no route joins, where `members` are given and the map leaves them apart
    (no session runs on it: nccl-mesh.c comm_read)."""
    serial, own = wait(), ffi.new('uint32_t *')
    lib.mesh_observe(ffi.NULL, 0, own)
    count, lmap, nodes, names = _members(None, members, True)
    if count < 0:
        raise OSError(-count, f"this node's bridge is unreachable: {os.strerror(-count)}")
    if members:
        _apart(lmap, f"the bridges' map over {list(members)}", [f'node {v} ({names[i] or "no record"})' for i, v in enumerate(nodes)])
    return SimpleNamespace(map=lmap, nodes=nodes, names=names, node=own[0], serial=serial)


def nodes():
    """mesh_nodes_health: every node's record this node's bridge holds, its own among them (`own`): its bridge node and
    name, its bridge's window and its memory (bytes), its model, and its health as its bridge last sampled it (`sampled`
    and `booted`, ns since the epoch; `free` and `wired` bytes; the kernel's memory `pressure`, 1 normal, 2 warning, 4
    critical; `thermal`; why it last restarted, `reset` and `shutdown`; its `power`, `charge` and `minutes`; its boot
    `session`).  What a client asks its own daemon about its node and its neighbours, carried over the fabric by the
    bridges, never probed over another network.  OSError naming the bridge where it is unreachable."""
    held, own = ffi.new('struct mesh_node_health[64]'), ffi.new('uint32_t *')
    count = lib.mesh_nodes_health(held, 64)
    if count < 0:
        raise OSError(-count, f"this node's bridge is unreachable: {os.strerror(-count)}")
    lib.mesh_observe(ffi.NULL, 0, own)
    text = lambda chars: ffi.string(chars).decode('utf-8', 'replace')
    return [{'node': r.node, 'name': text(r.name), 'window': r.window, 'memory': r.memory, 'model': text(r.model),
             'sampled': r.health.sampled, 'booted': r.health.booted, 'free': r.health.free, 'wired': r.health.wired,
             'reset': r.health.reset, 'power': r.health.power, 'pressure': r.health.pressure, 'thermal': r.health.thermal,
             'charge': r.health.charge, 'minutes': r.health.minutes, 'session': text(r.health.session),
             'shutdown': text(r.health.shutdown), 'own': r.node == own[0]} for r in held[0:count]]


def request(node, service):
    """design/bridge-and-ipc.md#requests: a channel to `service` at `node` (its bridge number; this node's own too),
    opened through this node's bridge and carried by the bridges along the map's route: a connected socket whose bytes
    reach the service and whose reads are the service's, each direction ended by a half close (shutdown(SHUT_WR)); a
    channel that fails resets it.  OSError naming the node and service where the bridge refuses it (EHOSTUNREACH: no
    path to the node; ENOENT: no such service there; ESRCH: this node's bridge is not running)."""
    import socket
    fd = lib.mesh_request(int(node), service.encode())
    if fd < 0:
        raise OSError(-fd, f"node {node}'s {service}: {os.strerror(-fd)}")
    return socket.socket(fileno=fd)


def ask(node, service, data, timeout=None):
    """request(node, service) given `data` whole and half closed, the service's answer read to its end: bytes."""
    with request(node, service) as channel:
        channel.settimeout(timeout)
        channel.sendall(data)
        channel.shutdown(1)
        out = bytearray()
        while chunk := channel.recv(1 << 16):
            out += chunk
        return bytes(out)


def serve(service, path):
    """`service` registered with this node's bridge as the Unix socket at `path` (a channel to it is a connection
    there): the socket whose life the registration lasts (closing it ends it; it reads end of file once the bridge
    stops, and is registered again by serving again)."""
    import socket
    fd = lib.mesh_serve(service.encode(), str(path).encode())
    if fd < 0:
        raise OSError(-fd, f'{service} at {path}: {os.strerror(-fd)}')
    return socket.socket(fileno=fd)


def footprint(pid=0):
    """The physical footprint of a process (this one where pid is 0), bytes: what the kernel charges it for (its
    ri_phys_footprint, proc_pid_rusage RUSAGE_INFO_V0), the measure Activity Monitor's Memory column and jetsam use."""
    import ctypes
    proc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    info = (ctypes.c_uint64 * 14)()
    if proc.proc_pid_rusage(pid or os.getpid(), 0, ctypes.byref(info)):
        raise OSError(ctypes.get_errno(), f'proc_pid_rusage({pid or os.getpid()})')
    return info[2 + 7]


def wire(block=None, buffsize=None):
    """mesh_wire_of: the rings a session of libnccl-mesh runs on (mesh-plan.h struct mesh_wire: `slot`, `small`, `lag`),
    as this process's sessions make them: `block` (None: this machine's page, the block every bridge's region takes) and
    `buffsize` (None: NCCL_BUFFSIZE as the library reads it, a positive integer, else NCCL's default MESH_BUFFSIZE)."""
    if buffsize is None:
        given = re.match(r'\s*([-+]?\d+)', os.environ.get('NCCL_BUFFSIZE', ''))
        buffsize = int(given.group(1)) if given and int(given.group(1)) > 0 else lib.MESH_BUFFSIZE
    made = ffi.new('struct mesh_wire *')
    lib.mesh_wire_of(block if block is not None else os.sysconf('SC_PAGE_SIZE'), buffsize, made)
    return made


def session_bytes(peers, wire_=None):
    """ncclMeshSessionBytes: the bytes of its node's window one communicator's session takes toward `peers` other members
    on `wire_` (wire(): None, this process's): its rings, transfer lists, SEND cells, completion words and violation
    page.  A session binds every other member of its communicator; a split's is its own."""
    made = ffi.new('uint64_t *')
    if lib.ncclMeshSessionBytes(peers, wire_ if wire_ is not None else wire(), made):
        raise ValueError(f'ncclMeshSessionBytes of {peers} peers')
    return made[0]


def relay_bytes(wire_=None):
    """ncclMeshRelayBytes: the bytes of its node's window a relay's bridge leases for one direction of a session on
    `wire_` (wire(): None, this process's) that its route crosses (mesh-flow.c mirror): one such lease, and one client
    slot of its bridge (mesh.h MESH_CLIENTS), a direction of two members of a session whose route crosses the node,
    whatever its stripes; session_bytes (above) is what each member's own bridge leases."""
    made = ffi.new('uint64_t *')
    if lib.ncclMeshRelayBytes(wire_ if wire_ is not None else wire(), made):
        raise ValueError('ncclMeshRelayBytes')
    return made[0]


def lease(nbytes):
    """ncclMemAlloc: `nbytes` (at least one) of this node's bridge's window, whole pages already registered on its
    devices (design/prepared-machine.md#M09), as a writable buffer (numpy.frombuffer and torch.frombuffer take it),
    returned to the window (ncclMemFree) once it and every object over it are gone.  RuntimeError with the library's
    reason where the window cannot meet it (the bytes asked and the pages free), the node runs no bridge, or this
    process still holds a lease of a bridge that stopped (rdma/NCCL.md "Memory")."""
    held, size = ffi.new('void **'), max(1, nbytes)
    result = lib.ncclMemAlloc(held, size)
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(lib.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(lib.ncclGetLastError(ffi.NULL)).decode()}')
    return ffi.buffer(ffi.gc(ffi.cast('char *', held[0]), lib.ncclMemFree), size)


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
    """mesh_trees_shortest: each node's in-tree of the map's routes (its table, else routes()), a node's parent the first
    of the map's nodes on its route to the root, as trees()."""
    n = lmap.nodes
    root, parent, weight = ffi.new('uint32_t[]', n), ffi.new('uint32_t[]', n * n), ffi.new('double[]', n)
    if not lib.mesh_trees_shortest(lmap.c, root, parent, weight):
        raise ValueError(f'mesh_trees_shortest: the map ({lmap.kind} of {n}) is not connected')
    return trees(n, [root[t] for t in range(n)], [[parent[t * n + v] for v in range(n)] for t in range(n)], [weight[t] for t in range(n)])


def compile(trees, what, root=0, whole=False, contributors=None, chunks=1):
    """mesh_compile: the collective `what` (WHATS index) along `trees` (trees()), each tree's part cut into `chunks`
    (pipelined), a program: plain data, `c` the struct's pointer and the arrays it points at, `moves` every rank's moves
    as tuples (op, peer, round, buffer, part, parts)."""
    n, parts, moves = trees.nodes, ffi.new('uint32_t *'), ffi.new('uint32_t *')
    lib.mesh_compile_room(trees.c, chunks, parts, moves)
    segment, weight = ffi.new('uint32_t[]', parts[0]), ffi.new('double[]', parts[0])
    first, move = ffi.new('uint32_t[]', n + 1), ffi.new('struct mesh_move[]', moves[0])
    words = None
    if contributors is not None:
        words = ffi.new('uint64_t[]', max(1, -(-n // 64)))
        for m in contributors:
            words[m // 64] |= 1 << (m % 64)
    made = ffi.new('struct mesh_program *')
    if lib.mesh_compile(trees.c, what, root, int(whole), chunks, words if words is not None else ffi.NULL, made, segment, weight, first, move):
        raise ValueError(f'mesh_compile refused {WHATS[what]} root {root} whole {whole} chunks {chunks}')
    return SimpleNamespace(c=made, what=what, root=root, nodes=n, scratch=made.scratch, chunks=chunks,
                           moves=[[(m.op, m.peer, m.round, m.buffer, m.part, m.parts) for m in (move[k] for k in range(first[r], first[r + 1]))]
                                  for r in range(n)],
                           keep=(segment, weight, first, move, words, trees))


def choose(lmap, sets, what, root, piece, alpha=lib.MESH_ALPHA_US, beta=lib.MESH_BETA_NS, wire_=None):
    """mesh_choose: the fastest of `what` (at `root`) along each tree set of `sets` (trees()), an all-reduce also whole,
    cut into 1, 2, 4, ... 64 chunks (every count timed), timed for `piece` (operand()) on `wire_` (wire(); None: this
    process's): set (its index), whole, chunks and time (us)."""
    made = ffi.new('struct mesh_choice *')
    held = ffi.new('struct mesh_trees[]', [t.c[0] for t in sets])
    status = lib.mesh_choose(lmap.c, held, len(sets), what, root, piece[0], wire_ if wire_ is not None else wire(), alpha, beta, made)
    if status:
        raise ValueError(f'mesh_choose of {WHATS[what]} root {root}: {os.strerror(status)}')
    return SimpleNamespace(set=made.set, whole=bool(made.whole), chunks=made.chunks, time=made.time)


def chosen(lmap, what, root, nbytes, sets=None, wire_=None):
    """The program libnccl-mesh's default table runs for a call of `nbytes` on lmap (nccl-mesh.c program_of): the
    choice (choose) at its size class, the least 2^k of at least nbytes (mesh-plan.h MESH_CLASS_LEAST ..
    MESH_CLASS_MOST), among `sets` (None: the shortest-path and the packed trees), on `wire_` (None: this process's),
    compiled (compile); price it with program_time on the same wire."""
    sets = sets or (shortest(lmap), pack(lmap))
    k = lib.MESH_CLASS_LEAST
    while k < lib.MESH_CLASS_MOST and (1 << k) < nbytes:
        k += 1
    picked = choose(lmap, sets, what, root, operand(0, 4, (1 << k) // 4), wire_=wire_)
    return compile(sets[picked.set], what, root, picked.whole, chunks=picked.chunks)


def steps(program, rank, piece):
    """mesh_program_steps: rank's steps of `program` for the operand `piece`, in its order."""
    made = ffi.new('struct mesh_step[]', max(1, program.c.first[program.nodes]))
    count = lib.mesh_program_steps(program.c, rank, piece[0], made)
    return [ffi.new('struct mesh_step *', made[i]) for i in range(count)]


def depth(program):
    """The program's dependent crossings: its last round plus one (a round's receives wait on lower rounds' SENDs)."""
    return 1 + max((m[2] for moves in program.moves for m in moves), default=-1)


def program_time(lmap, program, piece, alpha=0.0, beta=0.0, wire_=None):
    """mesh_program_time: the program as libnccl-mesh runs it on `wire_` (wire(); None: this process's) over lmap's ports,
    every rank's steps laid on its channels, every position a whole slot, in us (negative where it stops or
    mismatches); alpha and beta those of a link lmap gives no cost."""
    return lib.mesh_program_time(lmap.c, program.c, piece[0], wire_ if wire_ is not None else wire(), alpha, beta)
