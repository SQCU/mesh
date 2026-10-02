"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  The torch.backends
entry point loads it when torch is imported, so a program only names it: init_process_group(backend="mesh")."""


def _autoload():
    from . import backend  # noqa: F401
