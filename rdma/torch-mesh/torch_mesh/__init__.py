"""torch.distributed's "mesh" backend: libnccl-mesh's collectives for CPU and MPS tensors
(ProcessGroupMesh.cpp).  Importing it, or importing torch with this package installed (its
torch.backends entry point), registers the backend; a rank's bridge is MESH_REGION."""


def _autoload():
    import torch.distributed as dist
    if 'mesh' in dist.Backend.backend_list:
        return
    from . import _C
    dist.Backend.register_backend('mesh', _C.createProcessGroupMesh, devices=['cpu', 'mps'])


_autoload()
