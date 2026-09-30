"""torch.distributed's "mesh" backend: libnccl-mesh's collectives for CPU and MPS tensors
(ProcessGroupMesh.mm).  Importing it, or importing torch with this package installed (its
torch.backends entry point), registers the backend; a rank's bridge is MESH_REGION.  The group's link
map is its options: init_process_group(backend="mesh", pg_options=torch_mesh.Options(nodes, links)),
rank r's node nodes[r] in the bridge's link table, `links` a link-map file stated into it first (none:
the table as it stands); without them there is no group (ValueError on every rank alike).  A group made
later (new_group, a DeviceMesh's) without options takes the world's, restricted to its ranks.
links() is a snapshot of the table (epoch, the bridge's node, present, alpha/beta/stated/up per
directed link, each node's last report's sequence).  A failed call (a link lost, a bridge stalled, the link map's epoch moved during it:
revoked) revokes the group's communicator; the backend raises it only once every rank has agreed on it
(ncclMeshCommAgree, ULFM's MPI_Comm_agree), when the call is issued or at a later Work's wait.  agree()
is that agreement on the default group, every rank calling it: (the first call that failed on any rank
since the previous agreement, or None; this rank's link-table epoch); agreed() the last agreement made,
by agree() or the error path: (how many so far, the failed call or None, the epoch).  Once a process
group of the backend exists, every MPS tensor PyTorch's MPS allocator makes (a factory's, an op's
output) is window memory, which the backend sends and receives in place: the allocator's heaps place
each buffer in a window allocation of its own (ProcessGroupMesh.mm window_heaps).  empty() makes a
tensor (an MPS one, or a CPU one of a window allocation); counts() is what libnccl-mesh and the backend
copied, sent and waited for so far; records() the window allocator's records, and address(t) where a
tensor's bytes lie in them (0: outside the window).
  The backend also completes what PyTorch's parallelism APIs need of MPS tensors (_mps.py): DeviceMesh
on "mps", DTensor's backward through nn.Linear, and context_parallel's SDPA (CPU tensors too).  Its
first process group also installs partition.py: a mesh dimension's capacity-shaped parts (attach, write),
by which DTensor's Shard, tensor and context parallelism split it instead of equally.
MESH_TRACE=<file> writes each call's trace there (ProcessGroupMesh.mm) when the group is destroyed or the
process exits."""
import atexit
import os


class Options:
    """The "mesh" backend's pg_options: the link map as an operand whose shape is fixed and whose
    contents vary, the link table of the bridge of `region` (None: MESH_REGION), with rank r of the
    group at node nodes[r]; `links`, a link-map file (mesh rdma/mesh-collective.h), is stated into the
    table when the group is made (its nodes and each link's alpha and beta)."""

    def __init__(self, nodes, links=None, region=None):
        self.nodes, self.links, self.region = tuple(int(v) for v in nodes), links, region


_world = None


def _autoload():
    import torch.distributed as dist
    if 'mesh' in dist.Backend.backend_list:
        return
    dist.Backend.register_backend('mesh', _create, extended_api=True, devices=['cpu', 'mps'])


def _create(opts, options):
    """The backend (its first makes MPS tensors window memory from then on)."""
    from . import _C, _mps, partition  # noqa: F401 (partition installs itself)
    global _world
    ranks = list(opts.global_ranks_in_group)
    if options is None and _world is not None:
        options = Options([_world.nodes[r] for r in ranks], region=_world.region)
    if not isinstance(options, Options):
        raise ValueError('mesh: no link map: init_process_group(backend="mesh", pg_options=torch_mesh.Options(nodes, links)) '
                         'names each rank\'s node in the bridge\'s link table and the link-map file stated into it')
    if _world is None:
        _world = options
    backend = _C.createProcessGroupMesh(opts.store, opts.group_rank, opts.group_size, opts.timeout, options.region or '',
                                        str(options.links or ''), list(options.nodes))
    global _registered
    if not _registered:
        _registered = True
        _mps.register()
        if os.environ.get('MESH_TRACE'):
            atexit.register(_C.trace_dump)
    return backend


_registered = False
_autoload()


def empty(*size, dtype=None, device='mps'):
    """A tensor of window memory (for 'mps' the MPS allocator's): collectives read and write it in place."""
    import torch
    from . import _C
    return _C.empty(list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size),
                    torch.empty(0, dtype=dtype or torch.get_default_dtype()), device)


def counts():
    from . import _C
    return _C.counts()


def links(region=None):
    """The link table of N nodes: (epoch, the bridge's node, present [N], links [N, N, 4]: alpha, beta, stated,
    up of a to b, reported [N]: the sequence of each node's last report the table holds, 0 none)."""
    from . import _C
    return _C.links(region or '')


def agree():
    """ULFM's MPI_Comm_agree on the default group, every rank calling it: (the first call that failed on
    any rank since the previous agreement, or None; this rank's link-table epoch)."""
    from . import _C
    return _C.agree()


def agreed():
    """The last agreement this process made (by agree() or a call's error path): (how many so far, the
    failed call or None, the epoch)."""
    from . import _C
    return _C.agreed()


def records():
    from . import _C
    return _C.records()


def address(tensor):
    from . import _C
    return _C.address(tensor)
