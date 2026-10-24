"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  The torch.backends
entry point loads it when torch is imported, so a program only names it: init_process_group(backend="mesh").

compiler is the torch.compile backend "mesh" (setup.py's torch_dynamo_backends entry point, a backend torch documents
for a package to register): torch.compile(f, backend="mesh") compiles f with Inductor, its collectives the library's,
with nothing wrapped around a compiled graph's run.  Every collective, compiled or eager, is committed by _stream.mm's
one commit rule (README, Performance).

empty() makes a region tensor, whose bytes are a lease of this node's bridge's region (an MPS one through torch's
DLPack interop over the lease's Metal buffer: _stream.mm region_empty), which a collective reads and writes where it
lies; records() is the region tensors' leases live now and address(t) where a tensor's bytes lie in the region (0:
outside it); check_memory() warns where this process's MPS tensors outside it come to 1 GiB or more.

merge_partials() is the softmax over a split as the library's collective (the torch operator torch_mesh::merge_partials:
an all-reduce or reduce by libnccl-mesh's partials operator, ncclMeshRedOpCreatePartials): each member's attention over
its own keys merged by the rows' logsumexp into the attention over all of them; partition's ring attention merges by it
(a reduce-scatter by the operator).

plan() declares a program's collectives on a group once, in their order (libnccl-mesh's ncclMeshCommPlan), each
compiled then, so none is planned where it is issued.

record() and replay() are a step recorded once and replayed (the CUDA graph's analogue) through torch's documented
compile path: the step one graph (torch.compile, fullgraph) of the "mesh" backend, its replays issued back to back."""


def _autoload():
    from . import backend  # noqa: F401


def compiler(gm, example_inputs):
    """Inductor's compilation of a graph (torch._dynamo.lookup_backend("inductor"), as torch's composable backends
    compose it), its collectives the library's (_stream.mm's kernels for torch's functional collectives on MPS), with
    nothing wrapped around its run.  A compiled collective's publication is committed by the same rule as an eager
    one's (_stream.mm, the commit rule; README, Performance), so how a program is written does not decide how fast its
    collectives are.  The graph the backend is handed is a forward (a training graph's backward runs from autograd,
    under the same rule).  Its all-reduces are torch_mesh::all_reduce (backend.all_reduce), which Inductor calls rather
    than lowers: it would lower one to a copy into a buffer of its own and an all_reduce_ in place there
    (torch/_inductor/comm_lowering.py), memory the library does not send from (rdma/nccl.h Memory).  Its products of a
    few fp16 rows by a weight laid out as nn.Linear's (aten.mm(x, w^T), aten.addmm(b, x, w^T): w^T the transposed view of
    a row-major [N, K], x at most 8 rows) are torch_mesh::linear, one plain dispatch of torch-mesh's own (linear.metal),
    where MPS's matmul costs its host about 12 us a call.  Inductor's run-time size, alignment and scalar asserts are
    off: each is a check at every call of what the compile fixed (design/data-model.md section 6)."""
    import torch
    from torch._dynamo import lookup_backend
    from torch._inductor import config
    from torch._inductor.custom_graph_pass import CustomGraphPass

    from . import backend  # noqa: F401  (torch_mesh::all_reduce)

    aten = torch.ops.aten

    def rows(x, w):
        """Whether x @ w is linear's: fp16, x [R <= 8, K] contiguous, w the transposed view of a row-major [N, K]."""
        x, w = x.meta.get('val'), w.meta.get('val')
        return (x is not None and w is not None and x.dtype == w.dtype == torch.half and x.dim() == w.dim() == 2
                and x.shape[0] <= 8 and x.is_contiguous() and w.stride() == (1, w.shape[0]))

    class Leased(CustomGraphPass):
        def __call__(self, graph):
            for node in list(graph.nodes):
                if node.target is torch.ops._c10d_functional.all_reduce.default:
                    node.target = torch.ops.torch_mesh.all_reduce.default
                    continue
                mm = node.target is aten.mm.default and rows(*node.args)
                addmm = (node.target is aten.addmm.default and not node.kwargs and len(node.args) == 3 and rows(*node.args[1:])
                         and node.args[0].meta['val'].dim() == 1)
                if mm or addmm:
                    x, w = node.args[-2:]
                    with graph.inserting_before(node):
                        weight = graph.call_function(aten.permute.default, (w, [1, 0]))
                        weight.meta['val'] = w.meta['val'].permute(1, 0)
                        made = graph.call_function(torch.ops.torch_mesh.linear.default, (x, weight, node.args[0] if addmm else None))
                    made.meta['val'] = node.meta['val']
                    node.replace_all_uses_with(made)
                    graph.erase_node(node)

        def uuid(self):
            return 'torch_mesh::all_reduce torch_mesh::linear'
    with config.patch(post_grad_custom_pre_pass=Leased(), size_asserts=False, alignment_asserts=False, scalar_asserts=False):
        return lookup_backend('inductor')(gm, example_inputs)


def empty(*size, dtype=None, device='mps'):
    """A tensor of region memory (rdma/NCCL.md "Memory"), uninitialized: collectives read and write it in place where
    the library reads its operands in place.  An 'mps' one is an MPS tensor whose storage is the lease's Metal buffer
    (ncclMeshMemBuffer), made through DLPack; it returns its lease once the GPU's work before its release is done."""
    import torch

    from . import _stream
    sizes = list(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    kind = torch.device(device).type
    if kind not in ('cpu', 'mps'):
        raise ValueError(f'mesh: empty() makes cpu or mps tensors, not {device}')
    return _stream.region_empty(sizes, torch.empty(0, dtype=dtype or torch.get_default_dtype()), kind == 'mps')


def records():
    """The region tensors empty() made that are live now: each {address, bytes} of its lease."""
    from . import _stream
    return [{'address': a, 'bytes': b} for a, b in _stream.records()]


def address(tensor):
    """Where a tensor's bytes lie in the region (its first byte's address), 0 where they lie outside it."""
    from . import _stream
    return _stream.address(tensor)


def check_memory():
    """Bytes of this process's MPS tensors outside the region (torch.mps.current_allocated_memory: torch's allocator
    makes no region tensor), with a warning where they come to 1 GiB or more
    (docs/writing-a-service.md#memory-outside-the-region)."""
    import warnings

    import torch
    outside = torch.mps.current_allocated_memory()
    if outside >= 2**30:
        warnings.warn(f'mesh: {outside / 2**30:.2f} GiB of MPS tensors lie outside the region: no peer can be passed them '
                      'zero copy, and the GPU wires them beside the region, which risks the node. Make them with '
                      'torch_mesh.empty (docs/writing-a-service.md#memory-outside-the-region)', stacklevel=2)
    return outside


def plan(group, calls):
    """The collectives a program issues on `group` (None: the default group) declared once, in their order: `calls`
    the groups, each a list of torch_mesh.backend.Call (one torch collective, or one coalesced batch, a group) or (that
    list, repeat) issued repeat times in a row, the order repeated from the group's next collective on (backend
    ProcessGroupMesh.plan; partition's ops give their own, partition.calls).  Every rank declares its own calls, the same
    groups in the same order, at the same point.  Returns each group's (send entries, landing entries) on this rank."""
    import torch.distributed as dist

    from . import backend
    group = group or dist.distributed_c10d._get_default_group()
    return backend._mesh(group.group_name, 'plan').plan(calls)


def merge_partials(out, lse, group=None, root=-1):
    """Every member's attention over its own keys merged into the attention over all of them (split-KV attention's
    merge: libnccl-mesh's partials operator, ncclMeshRedOpCreatePartials, through the torch operator
    torch_mesh::merge_partials): `out` (rows x dim, any leading shape) and `lse` (each row's logsumexp, -inf where it saw
    no key), the same rows on every member.  Returns (the attention in out's dtype, each row's logsumexp), on every member
    (`root` -1) or at `root`."""
    import torch
    import torch.distributed as dist

    from . import backend
    group = group or dist.distributed_c10d._get_default_group()
    attended, lse = backend.attention(torch.ops.torch_mesh.merge_partials(backend.partial(out, lse), root, group.group_name))
    return attended.to(out.dtype), lse


class Recording:
    """A record()ed step: `k` steps unrolled in one graph compiled whole by the "mesh" backend at its first issue, and
    the fewer a count k does not divide in a graph of their own, compiled at its first issue; each rank's compilation
    met by every rank's before any of its collectives runs; `graphs` the graphs compiled so far."""

    def __init__(self, step, name, k):
        self.step, self.name, self.k, self.graphs, self.replays, self.unrolled = step, name, k, 0, 0, {}

    def issue(self, steps):
        """`steps` steps issued back to back on torch's stream with no wait: steps // k replays of the k-step graph,
        then one of the rest's."""
        full, rest = divmod(steps, self.k)
        for n in [self.k] * full + [rest] * bool(rest):
            if n not in self.unrolled:
                self.unrolled[n] = self._graph(n)
            self.unrolled[n]()
        self.replays += steps

    def _graph(self, n):
        """n steps unrolled in one graph (torch.compile with fullgraph), compiled at its first call."""
        import torch

        def steps():
            for _ in range(n):
                self.step()
        return torch.compile(steps, backend=self._backend, fullgraph=True, dynamic=False)

    def _backend(self, gm, example_inputs):
        """compiler's graph, then every rank's compilation of it met at the default group's store (a rank's compile
        outlasting the library's silent-link bound would cancel a peer's first crossings): bounded by the store's
        timeout.  The meeting is a host wait, so a publication waiting in torch's open buffer is committed first."""
        import torch.distributed as dist

        from . import _stream
        made = compiler(gm, example_inputs)
        if dist.is_initialized():
            _stream.commit()
            store, key = dist.distributed_c10d._get_default_store(), f'mesh/recording/{self.name}/{self.graphs}'
            store.set(f'{key}/{dist.get_rank()}', b'1')
            store.wait([f'{key}/{r}' for r in range(dist.get_world_size())])
        self.graphs += 1
        return made


_recordings = 0


def record(step, k=1):
    """`step` (a function of no arguments that runs one step on MPS tensors, its collectives torch's) recorded, the
    CUDA graph's analogue, through torch's documented compile path (A53: no interposition): one graph of `k` steps
    unrolled (torch.compile with fullgraph, its collectives functional ones), so a replay's host cost (the graph's
    guards, its subclass wrapping, Inductor's wrapper) is paid once for k steps, compiled at its first replay by the "mesh"
    backend (compiler: Inductor's kernels, the graph's collectives each issued on torch's stream under the one commit
    rule and completed where it is waited) after every rank has compiled it.  Nothing runs while it records:
    its tensors keep what they held, and each replay writes them.  Every rank records the same steps in the same
    order.  What the 10-01 form (mesh 4cbe704, metal-microbench 35c30a9 metal_recording.m) did beyond this needs an
    interposed recorder of torch's Metal commands, which has no documented extension point (README, Record and
    replay)."""
    global _recordings
    _recordings += 1
    return Recording(step, _recordings, k)


def replay(recording, steps=1):
    """`steps` steps of a record()ed step issued back to back on torch's stream (Recording.issue), with no wait between
    them: the steps are an operand's leading dimension, each reading what the last wrote in the stream's order; every
    rank replays alike.  The one wait is where a consumer reads, the time this returns: the last graph's interval in
    the stream over its steps, the span between two MPS timing events torch records before and after it (each event's
    time taken as its listener runs, equally late for both), read once the end event has passed.  With the graphs back
    to back the span runs from the end of the graph before it to its own end, the GPU's time or the host's issue,
    whichever is slower; as before, a first replay's compile falls inside it.  The steps' outputs are read where the
    caller reads them (torch commits and waits at .item(), .cpu() and torch.mps.synchronize())."""
    import torch
    if steps < 1:
        return 0.0
    last = steps % recording.k or min(recording.k, steps)
    recording.issue(steps - last)
    begin, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    begin.record()
    recording.issue(last)
    end.record()
    return begin.elapsed_time(end) / 1e3 / last
