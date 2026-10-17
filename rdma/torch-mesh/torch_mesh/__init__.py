"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  The torch.backends
entry point loads it when torch is imported, so a program only names it: init_process_group(backend="mesh").

compiler is the torch.compile backend "mesh" (setup.py's torch_dynamo_backends entry point, a backend torch documents
for a package to register): torch.compile(f, backend="mesh") compiles f with Inductor and batches the collectives of
each compiled graph's run into torch's command buffers (_stream.mm batch)."""


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
