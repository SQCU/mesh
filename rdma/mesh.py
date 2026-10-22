"""The mesh from Python: mesh-plan.h through _mesh_c (mesh_c_build.py compiles the header's declarations against it, so
no layout or prototype here is written by hand; `make mesh-c` builds it for this interpreter).  The read side is the
node's bridge's: current() is the map it keeps (mesh_map) and wait() the map's serial (mesh_map_wait); link maps,
their routes and a communicator's members' map are the header's own.
Collectives run through libnccl-mesh (mesh_mpi for NumPy, torch-mesh for torch); lease() is its ncclMemAlloc, the
memory they move taken from the node's bridge's region (rdma/NCCL.md "Memory")."""
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


class LinkMap:
    """mesh-plan.h's struct mesh_link_map (`c`, a pointer to it) of `kind` (KINDS) over `nodes`, the links `pairs`
    and each one's bandwidth in bits a second in `bandwidth` (None: none given; 0: that link's unknown), the hardware
    fact the routes and the packed trees read, with `relays` more nodes (numbered from `nodes`, holding no rank), each
    node's `mark` (None: its number; the bridges mark a node by its node number) and the routing table `next` (each
    node b's in-tree, next[v * N + b] v's parent in it, for N = nodes + relays; None: none), the storage held here;
    refused where mesh_link_map_check refuses it (no map).  A graph need not join its nodes (the bridges' records while
    a cable is out); mesh_link_map_joined says whether it does, and the trees and collectives run only where it does."""

    def __init__(self, kind, nodes, pairs, bandwidth=None, relays=0, next=None, mark=None):
        pairs = [tuple(p) for p in pairs]
        self.kind, self.nodes, self.pairs = kind, nodes, pairs
        self.bandwidth = None if bandwidth is None else [int(b) for b in bandwidth]
        self.relays = relays
        self.next, self.mark = (None if x is None else [int(v) for v in x] for x in (next, mark))
        self._link = ffi.new('uint32_t[][2]', [list(p) for p in pairs] or [[0, 0]])
        self._bandwidth = ffi.new('uint64_t[]', self.bandwidth or [0]) if bandwidth is not None else ffi.NULL
        self._next = ffi.new('uint32_t[]', self.next) if self.next else ffi.NULL
        self._mark = ffi.new('uint32_t[]', self.mark) if self.mark else ffi.NULL
        self.c = ffi.new('struct mesh_link_map *')
        self.c.kind, self.c.nodes, self.c.links, self.c.link = KINDS.index(kind), nodes, len(pairs), self._link
        self.c.bandwidth = self._bandwidth
        self.c.relays, self.c.next, self.c.mark = relays, self._next, self._mark
        status = lib.mesh_link_map_check(self.c)
        if status:
            raise ValueError(f'a {kind} of {nodes} nodes and {relays} relays with links {pairs} is not one its algorithms run on: '
                             f'{os.strerror(status)}')


def link_map(kind, nodes, pairs, bandwidth=None, relays=0, next=None, mark=None):
    """A LinkMap (above)."""
    return LinkMap(kind, nodes, pairs, bandwidth, relays, next, mark)


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
        node = ffi.new('uint32_t[]', max(1, room.nodes))
        link, bandwidth = ffi.new('uint32_t[][2]', max(1, room.links)), ffi.new('uint64_t[]', max(1, room.links))
        table, mark = ffi.new('uint32_t[]', max(1, room.nodes * room.nodes)), ffi.new('uint32_t[]', max(1, room.nodes))
        made = ffi.new('struct mesh_link_map *', {'nodes': room.nodes, 'links': room.links, 'link': link, 'bandwidth': bandwidth,
                                                  'next': table, 'mark': mark})
        given = ffi.new('uint32_t[]', asked) if asked else ffi.NULL
        if current:
            count = lib.mesh_map(given, len(asked), node, made)
        else:
            count = lib.mesh_map_members(whole.c, given, len(asked), node, made)
        if count != -errno.ENOSPC:
            break
        room = made
    if count < 0:
        return count, None, None
    N = count + made.relays
    lmap = LinkMap(KINDS[made.kind], count, link[0:made.links], bandwidth[0:made.links], made.relays, table[0:N * N],
                   mark[0:N])
    return count, lmap, list(node[0:N])


def members(whole, among, marks=None):
    """mesh_map_members: the map a communicator of the whole map's nodes `among` (map node i the whole map's among[i])
    reads, the whole map's other nodes its relays, its marks and table the whole map's (routes(whole) where it has
    none), so two members are linked where their route crosses no other member: the rule mesh_map reads the bridges'
    records by.  `marks` (None: the whole map's own) each whole-map node's mark, as the bridges mark each node by its
    node number, so the table is the bridges' (a whole map given links and bandwidths alone, link_map('graph', N, links,
    bandwidths), takes them here; its table, made by other marks, is made again by these).  The map as it is, whether
    or not it joins its members (mesh_link_map_joined names two it leaves apart)."""
    if marks is not None:
        whole = LinkMap(whole.kind, whole.nodes, whole.pairs, whole.bandwidth, whole.relays, None, marks)
    among = None if among is None else list(among)
    count, lmap, _ = _members(whole, among, False)
    if count < 0:
        raise ValueError(f'mesh_map_members over {among}: {os.strerror(-count)}')
    return lmap


def wait(serial=0, seconds=None):
    """mesh_map_wait: the serial of the map this node's bridge keeps once it differs from `serial` (0: at once), waited
    for on the serial's word, which the bridge wakes as it moves the map; `seconds` None waits with no bound (8e716999's
    form), a number bounds it, `serial` returned where it passes first.  OSError naming the bridge where it is
    unreachable, ESRCH where it exits while waited on."""
    made = ffi.new('uint64_t *', serial)
    status = lib.mesh_map_wait(made, (1 << 64) - 1 if seconds is None else max(0, min(int(seconds * 1e9), (1 << 64) - 1)))
    if status and status != errno.ETIMEDOUT:
        raise OSError(status, f"this node's bridge is unreachable: {os.strerror(status)}")
    return made[0]


def current(members=None):
    """mesh_map: the map this node's bridge keeps as of `serial` (wait()'s), over `members` (bridge node numbers, map
    node v the bridge node members[v], the other nodes its relays; None: every node its records hold, no relays):
    `map` the members' map as libnccl-mesh reads it (its relays, a link where two nodes' bridges reach each other, at its bandwidth, `mark` each node's
    bridge node number, `next` the routing table; members(map, ...) of the whole map is the
    map over any of them), `nodes` each map node's (then each relay's) bridge node (a node is its number;
    a name a person reads is peers.label's, from the name table), `node` this node's bridge's (mesh_reached).  The whole map (members None) is the records as they are,
    a node no bridge reaches among its nodes, and the members' map as it is, joined or not (mesh_link_map_joined names
    two it leaves apart).  OSError naming the bridge where it is unreachable."""
    serial, own = wait(), ffi.new('uint32_t *')
    lib.mesh_reached(ffi.NULL, 0, own)
    count, lmap, nodes = _members(None, members, True)
    if count < 0:
        raise OSError(-count, f"this node's bridge is unreachable: {os.strerror(-count)}")
    return SimpleNamespace(map=lmap, nodes=nodes, node=own[0], serial=serial)


def nodes():
    """mesh_nodes_health: every node's record this node's bridge holds, its own among them (`own`): its bridge node (a
    name a person reads is peers.label's), its bridge's region and its memory (bytes), its model, its boot (`booted`, ns since the epoch) and its
    `available` memory as its bridge last sampled it (what the kernel can give without pressure, its reclaimable cache
    counted).  What a client asks its own daemon about its node and its neighbours, carried over the fabric by the
    bridges, never probed over another network.  OSError naming the bridge where it is unreachable."""
    held, own = ffi.new('struct mesh_node_health[64]'), ffi.new('uint32_t *')
    count = lib.mesh_nodes_health(held, 64)
    if count < 0:
        raise OSError(-count, f"this node's bridge is unreachable: {os.strerror(-count)}")
    lib.mesh_reached(ffi.NULL, 0, own)
    text = lambda chars: ffi.string(chars).decode('utf-8', 'replace')
    return [{'node': r.node, 'region': r.region, 'memory': r.memory, 'model': text(r.model),
             'booted': r.health.booted, 'available': r.health.available, 'own': r.node == own[0]} for r in held[0:count]]


def footprint(pid=0):
    """The physical footprint of a process (this one where pid is 0), bytes: what the kernel charges it for (its
    ri_phys_footprint, proc_pid_rusage RUSAGE_INFO_V0), the measure Activity Monitor's Memory column and jetsam use."""
    import ctypes
    proc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    info = (ctypes.c_uint64 * 14)()
    if proc.proc_pid_rusage(pid or os.getpid(), 0, ctypes.byref(info)):
        raise OSError(ctypes.get_errno(), f'proc_pid_rusage({pid or os.getpid()})')
    return info[2 + 7]


def lease(nbytes):
    """ncclMemAlloc: `nbytes` (at least one) of this node's bridge's region, whole pages already registered on its
    devices (design/transport.md), as a writable buffer (numpy.frombuffer and torch.frombuffer take it),
    returned to the region (ncclMemFree) once it and every object over it are gone.  RuntimeError with the library's
    reason where the region cannot meet it (the bytes asked and the pages free), the node runs no bridge, or this
    process still holds a lease of a bridge that stopped (rdma/NCCL.md "Memory")."""
    held, size = ffi.new('void **'), max(1, nbytes)
    result = lib.ncclMemAlloc(held, size)
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(lib.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(lib.ncclGetLastError(ffi.NULL)).decode()}')
    return ffi.buffer(ffi.gc(ffi.cast('char *', held[0]), lib.ncclMemFree), size)
