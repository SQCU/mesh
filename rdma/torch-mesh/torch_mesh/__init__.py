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

declarations.Function traces a complete numerical function with fake tensors and compiles it before execution.
Its run attribute is the prepared callable; declaration and preparation do not execute the numerical payload."""


def _autoload():
    from . import backend  # noqa: F401


def compiler(gm, example_inputs, *, whole_graph=False):
    """Inductor's compilation of a graph (torch._dynamo.lookup_backend("inductor"), as torch's composable backends
    compose it), its collectives the library's (_stream.mm's kernels for torch's functional collectives on MPS), with
    nothing wrapped around its run.  A compiled collective's publication is committed by the same rule as an eager
    one's (_stream.mm, the commit rule; README, Performance), so how a program is written does not decide how fast its
    collectives are.  The graph the backend is handed is a forward (a training graph's backward runs from autograd,
    under the same rule).  Its all-reduces are torch_mesh::all_reduce (backend.all_reduce), which Inductor calls rather
    than lowers: it would lower one to a copy into a buffer of its own and an all_reduce_ in place there
    (torch/_inductor/comm_lowering.py), memory the library does not send from (rdma/nccl.h Memory).  Its products of a
    few fp16 rows by a weight laid out as nn.Linear's (aten.mm(x, w^T), aten.addmm(b, x, w^T): w^T the transposed view of
    a [N, K] whose rows lie apart by K or more, x at most 8 rows) are torch_mesh::linear, one plain dispatch of
    torch-mesh's own (linear.metal) where MPS's matmul costs its host about 12 us a call, with a tanh that is a
    product's one use folded into it, a cat of two row segments that is its x read in place, and up to four consecutive
    independent products one dispatch (torch_mesh::linears).  A graph's sums on a group, where every collective it
    names that group in is one of them and all of one form, are one issue's repeats (each node told their count:
    rdma/nccl.h ncclMeshFused repeats), so the host issues their crossing once a graph.  Inductor's run-time size,
    alignment and scalar asserts are off: each is a check at every call of what the compile fixed (design/data-model.md
    section 6)."""
    from pathlib import Path

    import torch
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner
    from torch._functorch import config as autograd_config
    from .declarations import operation, compiled_graph, Preparations, _CAPTURE

    from . import _stream, backend, partition  # noqa: F401  (torch_mesh::all_reduce, torch_mesh::allgatherv)
    from .streamed import node
    _stream.linear_source((Path(__file__).parent / 'linear.metal').read_text(), node()[1])

    aten = torch.ops.aten
    # each fused crossing's op and where its group's name lies among its arguments, its repeats the next
    CROSSINGS = {torch.ops.torch_mesh.all_reduce.default: 2, torch.ops.torch_mesh.linear_sum.default: 2,
                 torch.ops.torch_mesh.allgatherv.default: 3, torch.ops.torch_mesh.allgatherv_out.default: 4}

    def rows(x, w):
        """Whether x @ w is linear's: on MPS (its one implementation), fp16, x [R <= 8, K] contiguous, w the transposed
        view of a [N, K] whose rows lie apart by K or more."""
        x, w = x.meta.get('val'), w.meta.get('val')
        return (x is not None and w is not None and x.device.type == 'mps' and x.dtype == w.dtype == torch.half and x.dim() == w.dim() == 2
                and x.shape[0] <= 8 and x.is_contiguous() and w.stride(0) == 1 and w.stride(1) >= w.shape[0])

    def told(node, repeats):
        """A crossing's node told its repeats (the argument after its group's name), the fake values Inductor records
        for its call alike."""
        if node.target not in CROSSINGS:
            node.kwargs = dict(node.kwargs, repeats=repeats)
            node.meta.pop('eager_input_vals', None)
            return
        i = CROSSINGS[node.target] + 1
        node.args = (*node.args[:i], repeats)
        if 'eager_input_vals' in node.meta:
            values, named = node.meta['eager_input_vals']
            node.meta['eager_input_vals'] = ((*values[:i], repeats), named)

    def joined(x):
        """The two fp16 rows-segments x is the cat of along its last dimension, where that cat has no other use."""
        if x.target is not aten.cat.default or len(x.users) != 1 or len(x.args[0]) != 2 or (x.args[1:] or (0,))[0] not in (1, -1):
            return None
        parts = [p.meta.get('val') for p in x.args[0]]
        if any(p is None or p.dtype != torch.half or p.dim() != 2 or p.stride(1) != 1 for p in parts):
            return None
        return x.args[0]

    def grouped(graph):
        """Consecutive products of torch_mesh::linear, up to four, none reading another's output nor any of their outputs
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
            for m in products[i + 1:4 + i]:
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

    def native(node):
        """Lower both a product's sum and a plain sum to the same crossing."""
        import torch.distributed as dist
        group = dist.distributed_c10d._resolve_process_group(node.args[2])
        if not (isinstance(group, backend.ProcessGroupMesh) and group.size() > 1):
            return
        x, w = node.args[:2]
        plain = node.target is torch.ops.torch_mesh.all_reduce.default
        if plain:
            if w != 'sum' or x.meta['val'].device.type != 'mps' or x.meta['val'].dtype not in (torch.float16, torch.float32):
                return
            w = None
        scalar = x.meta['val'].ndim == 0
        output = node.meta['val']
        if scalar:
            with node.graph.inserting_before(node):
                source = node.graph.call_function(aten.view.default, (x, [1]))
                source.meta['val'] = x.meta['val'].view(1)
            x = source
            node.meta['val'] = output.view(1)
        repeats = node.args[3] if len(node.args) > 3 else 1
        node.target = torch.ops.torch_mesh.fused_step.default
        node.args = (x, w, 0, [x.meta['val'].shape[0]] * group.size(), node.args[2], True, repeats)
        node.meta.pop('eager_input_vals', None)
        if scalar:
            with node.graph.inserting_after(node):
                restored = node.graph.call_function(aten.view.default, (node, []))
                restored.meta['val'] = output
            node.replace_all_uses_with(restored)
            restored.args = (node, [])

    def rewrite(graph):
        for node in list(graph.nodes):
            if node.target is torch.ops._c10d_functional.all_reduce.default:
                node.target = torch.ops.torch_mesh.all_reduce.default
                told(node, 1)
                made = node.args[0]
                if (node.args[1] == 'sum' and made.target is torch.ops.torch_mesh.linear.default and len(made.users) == 1
                        and made.args[2:] in ((), (None,), (None, 0))):
                    with graph.inserting_before(node):
                        summed = graph.call_function(torch.ops.torch_mesh.linear_sum.default, (*made.args[:2], node.args[2], 1))
                    summed.meta['val'] = node.meta['val']
                    node.replace_all_uses_with(summed)
                    graph.erase_node(node)
                    graph.erase_node(made)
                continue
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

        def form(value):
            if isinstance(value, torch.Tensor):
                return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device.type)
            if isinstance(value, (list, tuple)):
                return tuple(map(form, value))
            return value

        crossings = {}
        for item in graph.nodes:
            target, args = operation(item)
            name = args.get('group_name')
            if name is not None:
                crossings.setdefault(name, []).append((item, target, args))
        for entries in crossings.values():
            forms = {(target, tuple((key, form(value)) for key, value in args.items() if key != 'repeats'))
                     for _, target, args in entries}
            if len(entries) > 1 and len(forms) == 1 and entries[0][1] in CROSSINGS:
                for item, _, _ in entries:
                    told(item, len(entries))
        for item in [n for n in graph.nodes if n.target in (torch.ops.torch_mesh.linear_sum.default,
                                                           torch.ops.torch_mesh.all_reduce.default)]:
            native(item)
        for item in graph.nodes:
            target, values = operation(item)
            if target not in (torch.ops.torch_mesh.allgatherv.default, torch.ops.torch_mesh.allgatherv_out.default):
                continue
            import torch.distributed as dist
            group = dist.distributed_c10d._resolve_process_group(values['group_name'])
            if not (partition._region_backed(values['x'], group) and group.size() > 1):
                continue
            into = target is torch.ops.torch_mesh.allgatherv_out.default
            native_target = torch.ops.torch_mesh.fused_step_out.default if into else torch.ops.torch_mesh.fused_step.default
            if item.target is target:
                args = {a.name: item.args[i] if i < len(item.args) else item.kwargs.get(a.name, a.default_value)
                        for i, a in enumerate(target._schema.arguments)}
                item.target = native_target
                item.args = (args['x'], None, *((args['out'],) if into else ()), args['dim'], args['sizes'],
                             args['group_name'], False, args['repeats'])
                item.kwargs = {}
            else:
                # Keep PyTorch's alias/view metadata and its reinplacing pass.
                # Only the underlying call changes to the existing native op.
                item.args = (native_target, *item.args[1:])
                item.kwargs = dict(item.kwargs, w=None, adds=False)
            item.meta.pop('eager_input_vals', None)

    def realize(graph, inputs, **options):
        # Declaration is configuration output, needed even on a numerical code
        # cache hit. Rewrite and capture before Inductor's FX cache lookup; the
        # cache then keys the actual rewritten operators and repeat operands.
        rewrite(graph.graph)
        for node in graph.graph.nodes:
            target, args = operation(node)
            if target is torch.ops.torch_mesh.linear.default:
                _stream.prepare_linear([args['x']], [args['w']], [args['x1']], args['act'])
            elif target is torch.ops.torch_mesh.linears.default:
                _stream.prepare_linear(args['x'], args['w'], args['x1'], -1)
            elif target is torch.ops.torch_mesh.linear_sum.default:
                _stream.prepare_linear([args['x']], [args['w']], [None], 0)
        forms = compiled_graph(graph)
        graph.recompile()
        made = compile_fx_inner(graph, inputs, **options)
        if _CAPTURE.get() is None:
            Preparations(forms)()
        return made

    # AOTAutograd's cache stores executable code without this communicator's
    # declarations. Always realize that configuration; Inductor still caches
    # the resulting numerical executable and kernels in compile_fx_inner.
    with autograd_config.patch(enable_autograd_cache=False), config.patch(
            size_asserts=False, alignment_asserts=False, scalar_asserts=False,
            max_fusion_unique_io_buffers=31):  # Metal's buffer-argument capacity
        if whole_graph:
            # Function has already declared forward and backward together.
            # Preserve that graph's shape environment and lower it directly;
            # a second AOTAutograd pass would derive its gradients again.
            from torch._guards import detect_fake_mode
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.proxy_tensor import make_fx
            from torch._inductor.decomposition import select_decomp_table
            from torch._subclasses.functional_tensor import dispatch_functionalize
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
            inputs = [mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                      for value in [*example_inputs, *constants]]
            with mode:
                lowered = make_fx(dispatch_functionalize(lifted, propagate_input_mutations=True), decomposition_table=select_decomp_table(),
                                  tracing_mode='fake', _allow_non_fake_inputs=True)(*inputs)
            output = next(node for node in lowered.graph.nodes if node.op == 'output')
            if not isinstance(output.args[0], (tuple, list)):
                output.args = ((output.args[0],),)
                lowered.recompile()
            made = realize(lowered, inputs, is_inference=True)
            return lambda *values: made([*values, *constants])
        return compile_fx(gm, example_inputs, inner_compile=realize)


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
    attended, lse = backend.attention(torch.ops.torch_mesh.merge_partials(backend.partial(out, lse), root, group.group_name, []))
    return attended.to(out.dtype), lse
