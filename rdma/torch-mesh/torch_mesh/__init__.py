"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py) and its compiler
(torch_mesh/compiler.py). Importing torch_mesh registers the backend, and so does the torch.backends entry point, so a
program only names it: init_process_group(backend="mesh").

empty() makes a region tensor, whose bytes are a lease of this node's bridge's region, which a collective reads and
writes where it lies; every MPS tensor is one (allocator.mm). address(t) is where a tensor's bytes lie in the region
(0: outside it); check_memory() warns where this process's MPS tensors outside it come to 1 GiB or more. A stepper
declares its collectives (partition.allgatherv_out) and runs as a compiled program (compiler.Program); no collective
runs eagerly."""


def _autoload():
    from . import backend  # noqa: F401


def empty(*size, dtype=None, device='mps'):
    """A tensor of region memory (rdma/NCCL.md "Memory"), uninitialized: collectives read and write it in place where
    the library reads its operands in place.  An 'mps' one is torch's, whose allocator is the region's: its storage is
    the lease's Metal buffer (ncclMeshMemBuffer), and it returns its lease once the GPU's work before its release is
    done."""
    import torch
    from torch._guards import detect_fake_mode

    from . import _stream
    sizes = list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    kind = torch.device(device).type
    if kind not in ('cpu', 'mps'):
        raise ValueError(f'mesh: empty() makes cpu or mps tensors, not {device}')
    fake = detect_fake_mode()
    if fake is not None:
        with fake:
            return torch.empty(sizes, dtype=dtype or torch.get_default_dtype(), device=kind)
    return _stream.region_empty(sizes, torch.empty(0, dtype=dtype or torch.get_default_dtype()), kind == 'mps')


def address(tensor):
    """Where a tensor's bytes lie in the region (its first byte's address), 0 where they lie outside it."""
    from . import _stream
    return _stream.address(tensor)


def check_memory():
    """Bytes of this process's MPS tensors outside the region (torch.mps.current_allocated_memory: torch's own heaps,
    which hold only pinned CPU memory and what was made before the region allocator was installed), with a warning
    where they come to 1 GiB or more (docs/writing-a-service.md#memory-outside-the-region)."""
    import warnings

    import torch
    outside = torch.mps.current_allocated_memory()
    if outside >= 2**30:
        warnings.warn(f'mesh: {outside / 2**30:.2f} GiB of MPS tensors lie outside the region: no peer can be passed them '
                      'zero copy, and the GPU wires them beside the region, which risks the node. Make them with '
                      'torch_mesh.empty (docs/writing-a-service.md#memory-outside-the-region)', stacklevel=2)
    return outside



from . import backend  # noqa: E402,F401  (registers the backend)
