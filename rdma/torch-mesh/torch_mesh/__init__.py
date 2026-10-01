"""torch.distributed's "mesh" backend: libnccl-mesh's collectives for CPU and MPS tensors
(ProcessGroupMesh.mm).  Importing it, or importing torch with this package installed (its
torch.backends entry point), registers the backend; a rank's bridge is MESH_REGION.  The group's link
map is its options: init_process_group(backend="mesh", pg_options=torch_mesh.Options(nodes, links)),
rank r's node nodes[r] in the bridge's link table, `links` a link-map file stated into it first (none:
the table as it stands); without them there is no group (ValueError on every rank alike).  A group made
later (new_group, a DeviceMesh's) without options takes the world's, restricted to its ranks.
links() is a snapshot of the table (epoch, the bridge's node, present, alpha/beta/stated/up per
directed link, each node's last report's sequence).  Nothing fails a call in time (a late peer, a bridge stopped, a link whose
session resumes make it late); a call that fails (its bridge observed a peer's exit, a peer's bridge leaving the mesh, or a
peer's bridge pairing again as another instance) revokes the group's communicator; the backend raises it only once the
ranks that stay have agreed on it (ncclMeshCommAgree, ULFM's MPI_Comm_agree: a rank departed as this rank's bridge observes
it, its node's bridge or this one having left the mesh, this node's region replaced or its process exited, does not vote; a
rank whose node returned with its process alive votes again and is connected again), when the call is issued or at a later
Work's wait, its message naming the ranks that voted; members() says which ranks are members now.  agree() is that agreement on the default group, every rank that stays calling it: (the first call that
failed on any voting rank since the previous agreement, or None; this rank's link-table epoch); agreed() the last agreement made,
by agree() or the error path: (how many so far, the failed call or None, the epoch).  Once a process
group of the backend exists, every MPS tensor PyTorch's MPS allocator makes (a factory's, an op's
output) is window memory, which the backend sends and receives in place: the allocator's heaps place
each buffer in a window allocation of its own (ProcessGroupMesh.mm window_heaps).  empty() makes a
tensor (an MPS one, or a CPU one of a window allocation); counts() is what libnccl-mesh (lagged, as stats())
and the backend copied, sent and waited for so far; stats() the transport's statistics, lagged; records() the
window allocator's records, and address(t) where a tensor's bytes lie in them (0: outside the window).
  The backend also completes what PyTorch's parallelism APIs need of MPS tensors (_mps.py): DeviceMesh
on "mps", DTensor's backward through nn.Linear, and context_parallel's SDPA (CPU tensors too).  Its
first process group also installs partition.py: a mesh dimension's capacity-shaped parts (attach, write),
by which DTensor's Shard, tensor and context parallelism split it instead of equally.
MESH_TRACE=<file> traces each call and replay (ProcessGroupMesh.mm; off without it): held in a bounded ring (at
most 8192 calls and 128 replays, about 12 MB), appended to the file when a group is destroyed, when the process
exits and on trace_dump(), each record written once, under the mesh's disk floor (rdma/mesh-disk.h);
trace_count() says how many calls and replays were traced so far (a call's records are the sequence numbers
between two counts), how many the ring dropped, and how many it holds unwritten."""
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
    """A tensor of window memory (for 'mps' the MPS allocator's): collectives read and write it in place.  While
    a step is record()ed, an 'mps' one is in pages no command of the step binds after its last cut so far, so a
    collective's receives into it are posted ahead from that cut (the backend's collective outputs are so too)."""
    import torch
    from . import _C
    return _C.empty(list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size),
                    torch.empty(0, dtype=dtype or torch.get_default_dtype()), device)


def all_to_all_counted(input, counts, send_segments, recv_segments, capacity, group=None):
    """MPI_Alltoallv whose counts the GPU wrote (libnccl-mesh ncclMeshAlltoAllCounted): input's rows grouped by
    destination rank, as many to rank q as segment q of `counts` sums to (an int64 MPS tensor, or a list of
    ints; send_segments[q] entries for rank q); each rank's segment for this rank (recv_segments[q] entries)
    and its rows move in one exchange, the rows received packed in rank order into a tensor of `capacity` rows.  Returns
    (that tensor, the received counts on the input's device, their handle, the Work): counted(handle) is the
    received counts on the host, waited for as they land (no wait on the GPU); the Work's wait() orders the
    rows and the device counts on the MPS stream."""
    import torch
    import torch.distributed as dist
    from . import _C
    group = group or dist.distributed_c10d._get_default_group()
    if not isinstance(counts, torch.Tensor):
        made = empty(len(counts), dtype=torch.int64, device='cpu')
        made.copy_(torch.tensor(counts, dtype=torch.int64))
        counts = made
    out = torch.empty((capacity,) + tuple(input.shape[1:]), dtype=input.dtype, device=input.device)
    host = empty(1 + sum(recv_segments), dtype=torch.int64, device='cpu').zero_()
    landed = torch.empty(sum(recv_segments), dtype=torch.int64, device=input.device)
    work = _C.all_to_all_counted(group.group_name, out, input.contiguous(), counts, list(send_segments), list(recv_segments), host, landed)
    return out, landed, host, work


def counted(handle):
    """An all_to_all_counted's received counts (each rank's segment for this rank, in rank order), once they have
    landed: waited for on the host, not the GPU."""
    from . import _C
    return _C.counted(handle)


def record(step):
    """`step` (a function of no arguments that runs one step on MPS tensors) recorded, the CUDA graph's
    analogue: every MPS command it encodes, from any thread, recorded as one invocation by metal-microbench's
    recorder (the program started with DYLD_INSERT_LIBRARIES=<metal-microbench>/.build/libmetal_recording.dylib,
    so every Metal object is its interposed one), each call's library call persistent (ncclMeshPersistentBegin:
    planned and placed, its GPU work in the recording, its requests posted ahead at each replay's start and
    started as the replay passes its cut) and its fence a cut there, or, where the recorder makes PyTorch's own
    copy kernels that store its send buffer publish it by range (MetalRecordPublish), no cut: its sends start
    as the ranges are published.  Nothing runs while it records: its tensors keep what they held,
    and each replay writes them; the MPS allocator's cached buffers are given back afterwards, so the
    recording's intermediates (retained by it) are never another tensor's.  Every rank records the same calls
    in the same order.  A Recording (commands, cuts, calls)."""
    import torch
    from . import _C
    made = _C.record(step)
    torch.mps.empty_cache()
    return made


def replay(recording, steps=1):
    """`steps` replays of a record()ed step, each waited for; every rank replays alike.  The last one's command
    buffers on the GPU, in seconds: (their GPU times summed, the first's start to the last's end)."""
    from . import _C
    return _C.replay(recording, steps)


def trace_dump():
    """The trace's records not yet written, appended to MESH_TRACE now (nothing without it)."""
    from . import _C
    _C.trace_dump()


def trace_count():
    """(calls traced, replays traced, calls dropped, replays dropped, calls held, replays held) so far."""
    from . import _C
    return _C.trace_count()


def counts():
    from . import _C
    return _C.counts()


def stats(first=0):
    """The transport's statistics as a reader may read them now, lagged (libnccl-mesh ncclMeshStats: a statistic
    reaches a reader K function evaluations after the one it describes, K the bridge's -K): the evaluations ended,
    K, and the entries of evaluations `first` (0: the last readable only) up to the evaluations ended less K."""
    from . import _C
    return _C.stats(first)


def links(region=None):
    """The link table of N nodes: (epoch, the bridge's node, present [N], links [N, N, 4]: alpha, beta, stated,
    up of a to b, reported [N]: the sequence of each node's last report the table holds, 0 none)."""
    from . import _C
    return _C.links(region or '')


def agree():
    """ULFM's MPI_Comm_agree on the default group, every rank that stays calling it (a rank departed as this rank's
    bridge sees it, its bridge or this rank's having left the mesh, does not vote): (the first call that failed on any voting rank since the
    previous agreement, or None; this rank's link-table epoch)."""
    from . import _C
    return _C.agree()


def members():
    """(the ranks of the default group that are members as this rank's bridge observes them now, the ranks that voted in
    its last agreement): a rank departs when its node's bridge leaves the mesh, this node's region is replaced, or its
    process exits; it is a member again once its node's bridge pairs again with its process alive."""
    from . import _C
    return _C.members()


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
