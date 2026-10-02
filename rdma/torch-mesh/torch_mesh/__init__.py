"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  The torch.backends
entry point loads it when torch is imported, so a program only names it: init_process_group(backend="mesh")."""


def _autoload():
    from . import _mps, backend, partition  # noqa: F401 (partition before a program imports context_parallel_unshard by name)
    _mps.install()
