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
group of the backend exists, torch's MPS factories (torch.empty, zeros, ones, full, rand, randn,
tensor, their *_like forms, and Tensor.to onto "mps") make window tensors on that thread: tensors of
the bridge's registered window, which the backend sends and receives in place, and whose release
returns their pages only once every use recorded on them is done (window(False) turns it off for a
block).  empty() makes one directly; counts() is what libnccl-mesh and the backend copied, sent and
waited for so far; records() the window allocator's records, and address(t) where a tensor's bytes
lie in them (0: outside the window).
  The backend also completes what PyTorch's parallelism APIs need of MPS tensors (_mps.py): DeviceMesh
on "mps", DTensor's backward through nn.Linear, and context_parallel's SDPA (CPU tensors too).  Its
first process group also installs partition.py: a mesh dimension's capacity-shaped parts (attach, write),
by which DTensor's Shard, tensor and context parallelism split it instead of equally.
MESH_TRACE=<file> writes each call's trace there (ProcessGroupMesh.mm) when the group is destroyed or the
process exits."""
import atexit
import contextlib
import os
import threading


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
    """The backend, and from then on this thread's MPS factories making window tensors."""
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
    global _mode
    if _mode is None:
        _mode = _window_mode()
        _mode.__enter__()
        _mps.register()
        if os.environ.get('MESH_TRACE'):
            atexit.register(_C.trace_dump)
    return backend


_mode = None
_autoload()
_state = threading.local()


@contextlib.contextmanager
def window(enabled=True):
    """A block in which torch's MPS factories make window tensors (enabled) or the MPS allocator's."""
    before = getattr(_state, 'enabled', True)
    _state.enabled = enabled
    try:
        yield
    finally:
        _state.enabled = before


def empty(*size, dtype=None, device='mps'):
    """A tensor of window memory (an MTLBuffer over it for 'mps'): collectives read and write it in place."""
    import torch
    from . import _C
    return _C.empty(list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size),
                    torch.empty(0, dtype=dtype or torch.get_default_dtype()), device)


def counts():
    from . import _C
    return _C.counts()


def links(region=None):
    """(epoch, the bridge's node, present [16], links [16, 16, 4]: alpha, beta, stated, up of a to b, reported
    [16]: the sequence of each node's last report the table holds, 0 none)."""
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


def _window_mode():
    import torch
    from torch.overrides import TorchFunctionMode

    filled = {torch.empty: None, torch.zeros: 0, torch.ones: 1, torch.full: 'full', torch.rand: 'uniform', torch.randn: 'normal',
              torch.empty_like: None, torch.zeros_like: 0, torch.ones_like: 1, torch.full_like: 'full',
              torch.rand_like: 'uniform', torch.randn_like: 'normal'}

    def on_mps(device):
        return device is not None and torch.device(device).type == 'mps'

    def plain(kwargs):
        return (kwargs.get('out') is None and kwargs.get('layout') in (None, torch.strided) and not kwargs.get('pin_memory')
                and kwargs.get('memory_format') in (None, torch.contiguous_format, torch.preserve_format))

    def made(size, dtype):
        return empty(list(size), dtype=dtype or torch.get_default_dtype())

    def make(func, args, kwargs):
        """A window tensor for an MPS factory call, filled on the MPS stream; None for any other call."""
        if not plain(kwargs):
            return None
        if func is torch.Tensor.to:
            device, dtype, _, _ = torch._C._nn._parse_to(*args[1:], **kwargs)
            source = args[0]
            if not on_mps(device) or source.device.type == 'mps':
                return None
            return None if source.requires_grad else made(source.shape, dtype or source.dtype).copy_(source)
        if func is torch.tensor:
            if not on_mps(kwargs.get('device')):
                return None
            if kwargs.get('requires_grad'):
                return None
            host = func(*args, **{**kwargs, 'device': 'cpu'})
            return made(host.shape, host.dtype).copy_(host)
        if func not in filled:
            return None
        how = filled[func]
        sequence = (list, tuple, torch.Size)
        if func in (torch.empty_like, torch.zeros_like, torch.ones_like, torch.full_like, torch.rand_like, torch.randn_like):
            source = args[0]
            if not on_mps(kwargs.get('device', source.device)):
                return None
            size, dtype, value = source.shape, kwargs.get('dtype') or source.dtype, (args[1:] or [kwargs.get('fill_value')])[0]
        else:
            if not on_mps(kwargs.get('device')):
                return None
            if how == 'full':
                size, value = kwargs.get('size', args[0] if args else None), (args[1:] or [kwargs.get('fill_value')])[0]
            else:
                size, value = kwargs.get('size', args[0] if len(args) == 1 and isinstance(args[0], sequence) else args), None
            dtype = kwargs.get('dtype')
        if kwargs.get('requires_grad') or size is None:
            return None
        if how == 'full' and dtype is None:
            dtype = torch.tensor(value).dtype if not isinstance(value, float) else torch.get_default_dtype()
        t = made(size, dtype)
        if how is None:
            return t
        if how == 'full':
            return t.fill_(value)
        if how in ('uniform', 'normal'):
            generator = kwargs.get('generator')
            return t.uniform_(generator=generator) if how == 'uniform' else t.normal_(generator=generator)
        return t.fill_(how)

    class Window(TorchFunctionMode):
        """torch's MPS factories making window tensors (above), while window() leaves it enabled."""

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if getattr(_state, 'enabled', True):
                t = make(func, args, kwargs)
                if t is not None:
                    return t
            return func(*args, **kwargs)

    return Window()
