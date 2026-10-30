"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  Importing torch_mesh
registers it, and so does the torch.backends entry point when torch is imported where this package is installed, so
a program only names it: init_process_group(backend="mesh").  A run that brings its commit's torch_mesh by path
has no installed entry point, so the import is what registers it there.

declarations.Function compiles a complete PyTorch graph into the common native
command records, with buffers and dispatches bound before execution.

empty() makes a region tensor, whose bytes are a lease of this node's bridge's region, which a collective reads and
writes where it lies.  Every MPS tensor is one: torch's MPS allocator is the region's (allocator.mm, installed as
_stream loads).  records() is the region tensors' leases live now and address(t) where a tensor's bytes lie in the
region (0: outside it); check_memory() warns where this process's MPS tensors outside it come to 1 GiB or more.

merge_partials() is the softmax over a split as the library's collective (the torch operator torch_mesh::merge_partials:
an all-reduce or reduce by libnccl-mesh's partials operator, ncclMeshRedOpCreatePartials): each member's attention over
its own keys merged by the rows' logsumexp into the attention over all of them; partition's ring attention merges by it
(a reduce-scatter by the operator).

plan() declares a program's collectives on a group once, in their order (libnccl-mesh's ncclMeshCommPlan), each
compiled then, so none is planned where it is issued.

declarations.Function traces a complete numerical function with fake tensors and compiles it before execution.
Its run attribute is the prepared callable; declaration and preparation do not execute the numerical payload."""


def _autoload():
    from . import backend  # noqa: F401


def compiler(gm, example_inputs, *, configuration=None, destinations=None):
    from .compilation import bind
    from .native import compile_graph
    with bind(configuration):
        lowered, metadata, operands = prepare_source(gm, example_inputs)
        return compile_graph(lowered, metadata, operands, destinations=destinations)


def prepare_source(gm, example_inputs):
    """Prepare the common numerical source before logical IR or page binding.

    Ownership and ordinary compilation consume these same functionalization,
    tensor equations and numerical rewrites. The returned operands retain their
    page identities; no numerical function is invoked.
    """
    from pathlib import Path
    import torch
    from . import _stream, backend, partition
    from .streamed import node
    from .compilation import operands as compiler_operands
    _stream.linear_source((Path(__file__).parent / 'linear.metal').read_text(), node()[1])
    aten = torch.ops.aten

    def replace(module, before, after):
        """Carry result identities through an equivalent numerical rewrite."""
        after.meta = before.meta.copy()
        before.replace_all_uses_with(after)
        module.meta['mesh_source_results'] = tree_map(
            lambda value: after if value is before else value, module.meta.get('mesh_source_results', {}))

    def remove_unobserved(module, value):
        # A fused intermediate has no separately emitted result. Its source
        # origins remain on the fused operation; it cannot name the final value.
        module.meta['mesh_source_results'] = {source: result for source, result in
            module.meta.get('mesh_source_results', {}).items()
            if not any(leaf is value for leaf in tree_flatten(result)[0])}

    def rows(x, w):
        """Whether x @ w is linear's: on MPS (its one implementation), fp16, x [R, K] in row views, w the transposed
        view of a [N, K] whose rows lie apart by K or more."""
        x, w = x.meta.get('val'), w.meta.get('val')
        return (x is not None and w is not None and x.device.type == 'mps' and x.dtype == w.dtype == torch.half and x.dim() == w.dim() == 2
                and x.stride(1) == 1 and w.stride(0) == 1 and w.stride(1) >= w.shape[0])

    def joined(x):
        """The two fp16 rows-segments x is the cat of along its last dimension, where that cat has no other use."""
        if x.target is not aten.cat.default or len(x.users) != 1 or len(x.args[0]) != 2 or (x.args[1:] or (0,))[0] not in (1, -1):
            return None
        parts = [p.meta.get('val') for p in x.args[0]]
        if any(p is None or p.dtype != torch.half or p.dim() != 2 or p.stride(1) != 1 for p in parts):
            return None
        return x.args[0]

    def grouped(module):
        """Consecutive products of torch_mesh::linear, within the manifest binding capacity, none reading another's output nor any of their outputs
        used before the last of them, as one torch_mesh::linears: one dispatch."""
        import operator
        graph = module.graph
        order = {n: i for i, n in enumerate(graph.nodes)}
        products = [n for n in graph.nodes if n.target is torch.ops.torch_mesh.linear.default]
        above = {}

        def ancestors(n):
            if n not in above:
                seen, stack = set(), list(n.all_input_nodes)
                while stack:
                    m = stack.pop()
                    if m not in seen:
                        seen.add(m)
                        stack.extend(m.all_input_nodes)
                above[n] = seen
            return above[n]
        i = 0
        while i < len(products):
            group = [products[i]]
            for m in products[i + 1:compiler_operands()['LINEAR_PRODUCTS'] + i]:
                if any(g in ancestors(m) for g in group) or any(order[u] < order[m] for g in group for u in g.users):
                    break
                group.append(m)
            if len(group) > 1:
                args = [list(m.args) + [None, 0, None][len(m.args) - 2:] for m in group]
                with graph.inserting_before(group[-1]):
                    made = graph.call_function(torch.ops.torch_mesh.linears.default, tuple([a[k] for a in args] for k in range(5)))
                    made.meta['val'] = [m.meta['val'] for m in group]
                    for j, m in enumerate(group):
                        part = graph.call_function(operator.getitem, (made, j))
                        replace(module, m, part)
                    made.meta['mesh_source_nodes'] = tuple(dict.fromkeys(
                        origin for item in group for origin in item.meta.get('mesh_source_nodes', (item,))))
                for m in group:
                    graph.erase_node(m)
            i += len(group)

    def rewrite(module):
        graph = module.graph
        for node in list(graph.nodes):
            mm = node.target is aten.mm.default and rows(*node.args)
            addmm = (node.target is aten.addmm.default and not node.kwargs and len(node.args) == 3 and rows(*node.args[1:])
                     and node.args[0].meta['val'].dim() == 1)
            if mm or addmm:
                x, w = node.args[-2:]
                parts = joined(x)
                with graph.inserting_before(node):
                    weight = graph.call_function(aten.permute.default, (w, [1, 0]))
                    weight.meta['val'] = w.meta['val'].permute(1, 0)
                    made = graph.call_function(torch.ops.torch_mesh.linear.default,
                                               (parts[0] if parts else x, weight, node.args[0] if addmm else None, 0) +
                                               ((parts[1],) if parts else ()))
                replace(module, node, made)
                graph.erase_node(node)
                if parts:
                    remove_unobserved(module, x)
                    graph.erase_node(x)
                users = list(made.users)
                if len(users) == 1 and users[0].target is aten.tanh.default:
                    made.args = made.args[:3] + (1,) + made.args[4:]
                    origins = tuple(dict.fromkeys((*made.meta.get('mesh_source_nodes', ()),
                                                   *users[0].meta.get('mesh_source_nodes', (users[0],)))))
                    remove_unobserved(module, made)
                    replace(module, users[0], made)
                    made.meta['mesh_source_nodes'] = origins
                    graph.erase_node(users[0])
        grouped(module)

    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._inductor.decomposition import select_decomp_table
    from .algebra import decompositions
    from torch.utils._pytree import tree_map, tree_flatten
    # Functionalization receives every tensor as an operand, including
    # weights and prebound outputs. It can then lower out= operations
    # while retaining their writes to those exact destinations.
    from .compilation import functionalize_source, lift_attributes
    lifted, operands = lift_attributes(gm, example_inputs)
    described = [node.meta['val'] for node in gm.graph.nodes if 'val' in node.meta]
    mode = detect_fake_mode(described) or FakeTensorMode(allow_non_fake_inputs=True)
    from .operand import TensorView
    def described_input(value, declared):
        if isinstance(value, TensorView):
            return declared
        return mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
    inputs = [tree_map(described_input, value, node.meta['val'])
              for node, value in zip((n for n in lifted.graph.nodes if n.op == 'placeholder'), operands, strict=True)]
    with mode:
        lowered = functionalize_source(lifted, inputs, select_decomp_table() | decompositions)
    output = next(node for node in lowered.graph.nodes if node.op == 'output')
    if not isinstance(output.args[0], (tuple, list)):
        output.args = ((output.args[0],),)
        lowered.recompile()
    rewrite(lowered)
    lowered.recompile()
    return lowered, tree_flatten(inputs)[0], tree_flatten(operands)[0]


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


def records():
    """The region tensors empty() made that are live now: each {address, bytes} of its lease."""
    from . import _stream
    return [{'address': a, 'bytes': b} for a, b in _stream.records()]


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


def plan(group, calls, *, production_seconds=None):
    """The collectives a program issues on `group` (None: the default group) declared once, in their order: `calls`
    the groups, each a list of torch_mesh.backend.Call (one torch collective, or one coalesced batch, a group) or (that
    list, repeat) issued repeat times in a row, the order repeated from the group's next collective on (backend
    ProcessGroupMesh.plan; partition's ops give their own, partition.calls).  Every rank declares its own calls, the same
    groups in the same order, at the same point.  Returns each group's (send entries, landing entries) on this rank."""
    import torch.distributed as dist

    from . import backend
    group = group or dist.distributed_c10d._get_default_group()
    return backend._mesh(group.group_name, 'plan').plan(calls, production_seconds=production_seconds)


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
    attended, lse = backend.attention(torch.ops.torch_mesh.merge_partials(backend.partial(out, lse), root, group.group_name, []))
    return attended.to(out.dtype), lse


# the backend registered by the import itself, not by install metadata (an entry point a by-path run does not have)
_autoload()
