"""torch.distributed's "mesh" backend: libnccl-mesh's collectives for CPU and MPS tensors
(ProcessGroupMesh.mm).  Importing it, or importing torch with this package installed (its
torch.backends entry point), registers the backend; a rank's bridge is MESH_REGION.  Once a process
group of the backend exists, torch's MPS factories (torch.empty, zeros, ones, full, rand, randn,
tensor, their *_like forms, and Tensor.to onto "mps") make window tensors on that thread: tensors of
the bridge's registered window, which the backend sends and receives in place, and whose release
returns their pages only once every use recorded on them is done (window(False) turns it off for a
block).  empty() makes one directly; counts() is what libnccl-mesh and the backend copied, sent and
waited for so far; records() the window allocator's records, and address(t) where a tensor's bytes
lie in them (0: outside the window).
  The backend also completes what PyTorch's parallelism APIs need of an MPS device: DeviceMesh("mps")
asks torch.mps whether its device is set up (_mps_device), and context_parallel's SDPA dispatch covers
CPU and MPS tensors (_context_parallel.py).  MESH_TRACE=<file> writes each call's trace there
(ProcessGroupMesh.mm) when the group is destroyed or the process exits."""
import atexit
import contextlib
import os
import threading


def _autoload():
    import torch.distributed as dist
    if 'mesh' in dist.Backend.backend_list:
        return
    dist.Backend.register_backend('mesh', _create, devices=['cpu', 'mps'])


def _create(store, rank, size, timeout):
    """The backend, and from then on this thread's MPS factories making window tensors."""
    from . import _C, _context_parallel
    backend = _C.createProcessGroupMesh(store, rank, size, timeout)
    global _mode
    if _mode is None:
        _mode = _window_mode()
        _mode.__enter__()
        _mps_device()
        _context_parallel.register()
        if os.environ.get('MESH_TRACE'):
            atexit.register(_C.trace_dump)
    return backend


def _mps_device():
    """DeviceMesh(device_type="mps") asks the device module whether a device is already selected and, if
    not, selects one: torch/distributed/device_mesh.py, DeviceMesh._setup_world_group_and_device calls
    device_handle.is_initialized() and then set_device() (torch 2.14, device_mesh.py:496, 505, 530), and
    torch.mps has neither.  A process has one Metal device, in use once MPS is available."""
    import torch
    if not hasattr(torch.mps, 'is_initialized'):
        torch.mps.is_initialized = torch.backends.mps.is_available


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
