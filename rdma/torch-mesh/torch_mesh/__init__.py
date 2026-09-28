"""torch.distributed's "mesh" backend: libnccl-mesh's collectives for CPU and MPS tensors
(ProcessGroupMesh.mm).  Importing it, or importing torch with this package installed (its
torch.backends entry point), registers the backend; a rank's bridge is MESH_REGION.  empty() makes a
tensor of the bridge's registered window, which the backend sends and receives in place; counts() is
what libnccl-mesh and the backend copied and waited for so far."""


def _autoload():
    import torch.distributed as dist
    if 'mesh' in dist.Backend.backend_list:
        return
    from . import _C
    dist.Backend.register_backend('mesh', _C.createProcessGroupMesh, devices=['cpu', 'mps'])


_autoload()


def empty(*size, dtype=None, device='mps'):
    """A tensor of window memory (an MTLBuffer over it for 'mps'): collectives read and write it in place."""
    import torch
    from . import _C
    return _C.empty(list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple)) else size),
                    torch.empty(0, dtype=dtype or torch.get_default_dtype()), device)


def counts():
    from . import _C
    return _C.counts()
