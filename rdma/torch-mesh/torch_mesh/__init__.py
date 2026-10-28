"""torch.distributed's "mesh" backend over ../libnccl-mesh.dylib (torch_mesh/backend.py).  The torch.backends
entry point loads it when torch is imported, so a program only names it: init_process_group(backend="mesh").

declarations.Function compiles a complete PyTorch graph into the common native
command records, with buffers and dispatches bound before execution.

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

declarations.Function traces a complete numerical function with fake tensors and compiles it before execution.
Its run attribute is the prepared callable; declaration and preparation do not execute the numerical payload."""


def _autoload():
    from . import backend  # noqa: F401


def compiler(gm, example_inputs, *, configuration=None, destinations=None):
    from .compilation import bind
    with bind(configuration):
        return _compile(gm, example_inputs, destinations)


def _compile(gm, example_inputs, destinations):
    """Lower a declared whole function to the shared native command records.

    Torch supplies the graph and tensor metadata. Mathematical fusion and Metal
    source generation happen here; framework launchers are not an executor.
    """
    from pathlib import Path
    import torch
    from . import _stream, backend, partition
    from .streamed import node
    from .native import compile_graph
    from .compilation import operands as compiler_operands
    _stream.linear_source((Path(__file__).parent / 'linear.metal').read_text(), node()[1])
    aten = torch.ops.aten

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

    def grouped(graph):
        """Consecutive products of torch_mesh::linear, within the manifest binding capacity, none reading another's output nor any of their outputs
        used before the last of them, as one torch_mesh::linears: one dispatch."""
        import operator
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
                        part.meta['val'] = m.meta['val']
                        m.replace_all_uses_with(part)
                for m in group:
                    graph.erase_node(m)
            i += len(group)

    def rewrite(graph):
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
                made.meta['val'] = node.meta['val']
                node.replace_all_uses_with(made)
                graph.erase_node(node)
                if parts:
                    graph.erase_node(x)
                users = list(made.users)
                if len(users) == 1 and users[0].target is aten.tanh.default:
                    made.args = made.args[:3] + (1,) + made.args[4:]
                    users[0].replace_all_uses_with(made)
                    graph.erase_node(users[0])
        grouped(graph)

    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._inductor.decomposition import select_decomp_table
    from .algebra import decompositions
    from torch._subclasses.functional_tensor import dispatch_functionalize
    from torch.utils._pytree import tree_map, tree_flatten
    # Functionalization receives every tensor as an operand, including
    # weights and prebound outputs. It can then lower out= operations
    # while retaining their writes to those exact destinations.
    copied = torch.fx.Graph()
    copied.output(copied.graph_copy(gm.graph, {}))
    lifted = torch.fx.GraphModule(gm, copied)
    constants = []
    first = next(node for node in copied.nodes if node.op != 'placeholder')
    for node in list(copied.nodes):
        if node.op != 'get_attr':
            continue
        value = lifted
        for name in node.target.split('.'):
            value = getattr(value, name)
        if isinstance(value, torch.Tensor):
            with copied.inserting_before(first):
                operand = copied.placeholder(f'_mesh_constant_{len(constants)}')
            operand.meta = node.meta.copy()
            constants.append(value)
            node.replace_all_uses_with(operand)
            if node is first:
                first = node.next
            copied.erase_node(node)
    lifted.recompile()
    described = [node.meta['val'] for node in gm.graph.nodes if 'val' in node.meta]
    mode = detect_fake_mode(described) or FakeTensorMode(allow_non_fake_inputs=True)
    operands = [*example_inputs, *constants]
    from .operand import TensorView
    def described_input(value, node):
        if isinstance(value, TensorView):
            return node.meta['val']
        return mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
    inputs = [tree_map(lambda value: described_input(value, node), value)
              for node, value in zip((n for n in copied.nodes if n.op == 'placeholder'), operands, strict=True)]
    with mode:
        lowered = make_fx(dispatch_functionalize(lifted, propagate_input_mutations=True), decomposition_table=select_decomp_table() | decompositions,
                          tracing_mode='fake', _allow_non_fake_inputs=True)(*inputs)
    output = next(node for node in lowered.graph.nodes if node.op == 'output')
    if not isinstance(output.args[0], (tuple, list)):
        output.args = ((output.args[0],),)
        lowered.recompile()
    rewrite(lowered.graph)
    lowered.recompile()
    return compile_graph(lowered, tree_flatten(inputs)[0], tree_flatten(operands)[0], destinations=destinations)


def empty(*size, dtype=None, device='mps'):
    """A tensor of region memory (rdma/NCCL.md "Memory"), uninitialized: collectives read and write it in place where
    the library reads its operands in place.  An 'mps' one is an MPS tensor whose storage is the lease's Metal buffer
    (ncclMeshMemBuffer), made through DLPack; it returns its lease once the GPU's work before its release is done."""
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
