"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  The torch.backends
entry point loads it when torch is imported, so a program only names it: init_process_group(backend="mesh").

compiler is the torch.compile backend "mesh" (setup.py's torch_dynamo_backends entry point, a backend torch documents
for a package to register): torch.compile(f, backend="mesh") compiles f with Inductor and batches the collectives of
each compiled graph's run into torch's command buffers (_stream.mm batch).

empty() makes a window tensor, whose bytes are a lease of this node's bridge's window (an MPS one through torch's
DLPack interop over the lease's Metal buffer: _stream.mm window_empty), which a collective reads and writes where it
lies; records() is the window tensors' leases live now and address(t) where a tensor's bytes lie in the window (0:
outside it)."""


def _autoload():
    from . import backend  # noqa: F401


def compiler(gm, example_inputs):
    """Inductor's compilation of a graph (torch._dynamo.lookup_backend("inductor"), as torch's composable backends
    compose it), each run of it a batch (_stream.mm batch): while it runs, a collective's publication joins torch's
    open command buffer while two of its buffers are in flight, and its end commits what it left open, so a compiled
    collective pays no command buffer of its own (mesh 3c00d37: 12.7 us a crossing compiled against 17.4 eager).
    Inductor's wrapper is straight-line code, so no host wait inside it waits on a buffer left open.  The graph the
    backend is handed is a forward (a training graph's backward runs from autograd, unbatched)."""
    import functools

    from torch._dynamo import lookup_backend

    from . import backend
    compiled = lookup_backend('inductor')(gm, example_inputs)
    if backend._stream is None:
        return compiled
    batch = backend._stream.batch

    @functools.wraps(compiled)
    def run(*args):
        batch(1)
        try:
            return compiled(*args)
        finally:
            batch(-1)
    return run


def empty(*size, dtype=None, device='mps'):
    """A tensor of window memory (rdma/NCCL.md "Memory"), uninitialized: collectives read and write it in place where
    the library reads its operands in place.  An 'mps' one is an MPS tensor whose storage is the lease's Metal buffer
    (ncclMeshMemBuffer), made through DLPack; it returns its lease once the GPU's work before its release is done."""
    import torch

    from . import backend
    sizes = list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    dtype = dtype or torch.get_default_dtype()
    kind = torch.device(device).type
    if kind not in ('cpu', 'mps'):
        raise ValueError(f'mesh: empty() makes cpu or mps tensors, not {device}')
    if backend._stream is None:
        if kind == 'mps':
            raise RuntimeError('mesh: empty(device="mps") needs MPS')
        return backend.blank(sizes, dtype, 'cpu')
    return backend._stream.window_empty(sizes, torch.empty(0, dtype=dtype), kind == 'mps')


def records():
    """The window tensors empty() made that are live now: each {address, bytes} of its lease."""
    from . import backend
    return [{'address': a, 'bytes': b} for a, b in backend._stream.records()] if backend._stream is not None else []


def address(tensor):
    """Where a tensor's bytes lie in the window (its first byte's address), 0 where they lie outside it."""
    from . import backend
    if backend._stream is not None:
        return backend._stream.address(tensor)
    held, offset = backend.ffi.new('void **'), backend.ffi.new('size_t *')
    pointer = backend.ffi.cast('void *', tensor.data_ptr())
    return tensor.data_ptr() if backend.LIB.ncclMeshMemBuffer(pointer, held, offset) == 0 else 0
