"""The mesh from Python: mesh-plan.h through _mesh_c (mesh_c_build.py compiles the header's declarations against it, so
no layout or prototype here is written by hand; `make mesh-c` builds it for this interpreter).  The read side is the
node's bridge's: current() is the map it keeps (mesh_map); link maps,
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


class LinkMap:
    """mesh-plan.h's struct mesh_link_map (`c`, a pointer to it) over `nodes`: its `edges`, one a cable, each (node a,
    node b, cable, capacity in bytes a second), the hardware fact the routes and the packed trees read, with `relays`
    more nodes (numbered from `nodes`, holding no rank), each node's `mark` (None: its number; the bridges mark a node
    by its node number) and the routing table `next` (each node b's in-tree, next[v * N + b] v's parent in it, for N =
    nodes + relays; None: none), the storage held here.  A graph need not join its nodes (the bridges' records while a
    cable is out)."""

    def __init__(self, nodes, edges, mark, relays=0, next=None):
        edges = [tuple(int(v) for v in e) for e in edges]
        self.nodes, self.edges, self.relays = nodes, edges, relays
        self.mark, self.next = [int(v) for v in mark], None if next is None else [int(v) for v in next]
        self._edge = ffi.new('struct mesh_edge[]', [{'a': a, 'b': b, 'cable': c, 'capacity': k} for a, b, c, k in edges] or 1)
        self._next = ffi.new('uint32_t[]', self.next) if self.next else ffi.NULL
        self._mark = ffi.new('uint32_t[]', self.mark or [0])
        self.c = ffi.new('struct mesh_link_map *')
        self.c.nodes, self.c.edges, self.c.edge = nodes, len(edges), self._edge
        self.c.relays, self.c.next, self.c.mark = relays, self._next, self._mark


def link_map(nodes, edges, mark, relays=0, next=None):
    """A LinkMap (above)."""
    return LinkMap(nodes, edges, mark, relays, next)


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
    room = ffi.new('struct mesh_link_map *', {'nodes': max(1, len(asked)), 'edges': max(1, len(asked) * (len(asked) - 1) // 2)})
    while True:
        node = ffi.new('uint32_t[]', max(1, room.nodes))
        edge = ffi.new('struct mesh_edge[]', max(1, room.edges))
        table, mark = ffi.new('uint32_t[]', max(1, room.nodes * room.nodes)), ffi.new('uint32_t[]', max(1, room.nodes))
        made = ffi.new('struct mesh_link_map *', {'nodes': room.nodes, 'edges': room.edges, 'edge': edge, 'next': table, 'mark': mark})
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
    lmap = LinkMap(count, [(e.a, e.b, e.cable, e.capacity) for e in edge[0:made.edges]], mark[0:N], made.relays, table[0:N * N])
    return count, lmap, list(node[0:N])


def members(whole, among):
    """mesh_map_members: the map a communicator of the whole map's nodes `among` (map node i the whole map's among[i])
    reads, the whole map's other nodes its relays, its marks and table the whole map's, so two members are linked where
    their route crosses no other member: the rule mesh_map reads the bridges' records by, with every cable and capacity
    of the whole map kept.  The whole map carries its routing table (routes) and its marks.  The map as it is, whether
    or not it joins its members."""
    among = None if among is None else list(among)
    count, lmap, _ = _members(whole, among, False)
    if count < 0:
        raise ValueError(f'mesh_map_members over {among}: {os.strerror(-count)}')
    return lmap


def current(members=None):
    """mesh_map: the map this node's bridge keeps, over `members` (bridge node numbers, map
    node v the bridge node members[v], the other nodes its relays; None: every node its records hold, no relays):
    `map` the members' map as libnccl-mesh reads it (its relays, an edge for each cable two nodes' bridges reach each other over, at its capacity, `mark` each node's
    bridge node number, `next` the routing table; members(map, ...) of the whole map is the
    map over any of them), `nodes` each map node's (then each relay's) bridge node (a node is its number;
    a name a person reads is peers.label's, from the name table), `node` this node's bridge's (mesh_reached).  The whole map (members None) is the records as they are,
    a node no bridge reaches among its nodes, and the members' map as it is, joined or not.  OSError naming the bridge
    where it is unreachable."""
    own = ffi.new('uint32_t *')
    lib.mesh_reached(ffi.NULL, 0, own)
    count, lmap, nodes = _members(None, members, True)
    if count < 0:
        raise OSError(-count, f"this node's bridge is unreachable: {os.strerror(-count)}")
    return SimpleNamespace(map=lmap, nodes=nodes, node=own[0])


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
