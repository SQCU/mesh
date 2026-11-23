"""The mesh from Python: mesh-view.h, mesh-call.h and nccl.h through _mesh_c (mesh_c_build.py compiles the headers'
declarations against them, so no layout or prototype here is written by hand; `make mesh-c` builds it for this
interpreter).  The read side is the node's bridge's: current() is the map it keeps (mesh_map) and wait() the map's
serial (mesh_map_wait); nodes() is every node's record it holds (mesh_nodes_health).
Collectives run through libnccl-mesh (mesh_mpi for NumPy, torch-mesh for torch) on ordinary host memory; lease() is its
ncclMemAlloc."""
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
    """mesh-view.h's struct mesh_link_map (`c`, a pointer to it) of `kind` (KINDS) over `nodes`, the links `pairs`
    and each one's bandwidth in bits a second in `bandwidth` (None: none given; 0: that link's unknown), the storage
    held here; refused where mesh_link_map_check refuses it (no map).  A graph need not join its nodes (the bridges'
    records while a cable is out)."""

    def __init__(self, kind, nodes, pairs, bandwidth=None):
        pairs = [tuple(p) for p in pairs]
        self.kind, self.nodes, self.pairs = kind, nodes, pairs
        self.bandwidth = None if bandwidth is None else [int(b) for b in bandwidth]
        self._link = ffi.new('uint32_t[][2]', [list(p) for p in pairs] or [[0, 0]])
        self._bandwidth = ffi.new('uint64_t[]', self.bandwidth or [0]) if bandwidth is not None else ffi.NULL
        self.c = ffi.new('struct mesh_link_map *')
        self.c.kind, self.c.nodes, self.c.links, self.c.link = KINDS.index(kind), nodes, len(pairs), self._link
        self.c.bandwidth = self._bandwidth
        status = lib.mesh_link_map_check(self.c)
        if status:
            raise ValueError(f'a {kind} of {nodes} nodes with links {pairs} is no map: {os.strerror(status)}')


def _map():
    """mesh_map, the room grown until it fits: its count of nodes (or -errno), the LinkMap and each map node's bridge
    node."""
    room = ffi.new('struct mesh_link_map *', {'nodes': 1, 'links': 1})
    while True:
        node = ffi.new('uint32_t[]', max(1, room.nodes))
        link, bandwidth = ffi.new('uint32_t[][2]', max(1, room.links)), ffi.new('uint64_t[]', max(1, room.links))
        made = ffi.new('struct mesh_link_map *', {'nodes': room.nodes, 'links': room.links, 'link': link,
                                                  'bandwidth': bandwidth})
        count = lib.mesh_map(node, made)
        if count != -errno.ENOSPC:
            break
        room = made
    if count < 0:
        return count, None, None
    return count, LinkMap(KINDS[made.kind], count, link[0:made.links], bandwidth[0:made.links]), list(node[0:count])


def wait(serial=0):
    """mesh_map_wait: the serial of the map this node's bridge keeps once it differs from `serial` (0: at once), waited
    for on the serial's word, which the bridge wakes as it moves the map; the wait is the caller's own.  OSError naming
    the bridge where it is unreachable, ESRCH where it exits while waited on."""
    made = ffi.new('uint64_t *', serial)
    status = lib.mesh_map_wait(made)
    if status:
        raise OSError(status, f"this node's bridge is unreachable: {os.strerror(status)}")
    return made[0]


def current():
    """mesh_map: the map this node's bridge keeps as of `serial` (wait()'s): `map` its LinkMap over every node its
    records hold (a link where two nodes' bridges reach each other, at its bandwidth), `nodes` each map node's bridge
    node (a node is its number; a name a person reads is peers.label's, from the name table), `node` this node's
    bridge's (mesh_reached).  The map is the records as they are, joined or not.  OSError naming the bridge where it is
    unreachable."""
    serial, own = wait(), ffi.new('uint32_t *')
    lib.mesh_reached(ffi.NULL, 0, own)
    count, lmap, nodes = _map()
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


def check(result):
    """A libnccl-mesh result: RuntimeError with the library's reason where it is not ncclSuccess."""
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(lib.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(lib.ncclGetLastError(ffi.NULL)).decode()}')


def lease(nbytes):
    """ncclMemAlloc: `nbytes` (at least one) of memory the library allocates, as a writable buffer (numpy.frombuffer
    and torch.frombuffer take it), returned (ncclMemFree) once it and every object over it are gone.  RuntimeError with
    the library's reason where it cannot allocate them."""
    held, size = ffi.new('void **'), max(1, nbytes)
    check(lib.ncclMemAlloc(held, size))
    return ffi.buffer(ffi.gc(ffi.cast('char *', held[0]), lib.ncclMemFree), size)
