"""AOT collective declarations from the framework's complete function graph.

The frontend records operators, shapes and groups. The common native compiler
chooses reduction trees, physical paths and packet cuts; this module contains
no network topology or collective algorithm of its own.
"""
import math
import json

import torch
import torch.distributed as dist
from torch.overrides import TorchFunctionMode


class SourceCapture(TorchFunctionMode):
    """Fake source values paired with their original typed page operands.

    This mode exists only while FX records source. Captured parameters and
    explicit arguments share the same identity table, including gradient leaves.
    The generated graph contains ordinary operators and lifted state values.
    """
    def __init__(self, mode):
        super().__init__()
        self.mode, self.values, self.originals, self.local_sources = mode, {}, {}, {}
        self.meshes = []

    def value(self, value):
        from torch.distributed.tensor import DTensor
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
        if not isinstance(value, torch.Tensor):
            return value
        if id(value) in self.values:
            return self.values[id(value)]
        if isinstance(value, FakeTensor):
            return value
        with disable_proxy_modes_tracing(), self.mode:
            if isinstance(value, DTensor):
                made = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                           device=value.device, requires_grad=value.requires_grad)
                self.local_sources[id(value._local_tensor)] = value
                if value.device_mesh not in self.meshes:
                    self.meshes.append(value.device_mesh)
            else:
                made = self.mode.from_tensor(value)
        self.values[id(value)], self.originals[id(made)] = made, value
        return made

    def local(self, value):
        """An explicit local view keeps its global source identity as metadata."""
        from .partition import _rank_selection, _value_ownership
        from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot
        made = self.value(value)
        original = self.originals.get(id(made), value)
        selection, _ = _rank_selection(original._spec, original.device_mesh.get_coordinate())
        for _, axis, runs in selection['steps']:
            first, count, _, _ = runs[0]
            made = made.narrow(axis, first, count)
        made = torch.ops.aten.alias.default(made)
        declaration, _ = _value_ownership(original, original.device_mesh)
        proxy = get_proxy_mode()
        get_proxy_slot(made, proxy.tracer).proxy.node.meta['mesh_local_source'] = dict(
            declaration, mesh=original.device_mesh, selection=selection)
        return made

    def __torch_function__(self, operation, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_map
        return operation(*tree_map(self.value, args), **tree_map(self.value, kwargs or {}))

    def __enter__(self):
        from .partition import _SOURCE_CAPTURE
        self.token = _SOURCE_CAPTURE.set(self)
        return super().__enter__()

    def __exit__(self, *error):
        from .partition import _SOURCE_CAPTURE
        try:
            return super().__exit__(*error)
        finally:
            _SOURCE_CAPTURE.reset(self.token)

    def bind_arguments(self, described, originals):
        from torch.utils._pytree import tree_flatten
        from torch.distributed.tensor import DTensor
        from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot, set_proxy_slot
        from .partition import _value_ownership
        proxy = get_proxy_mode()
        for traced, original in zip(tree_flatten(described)[0], tree_flatten(originals)[0], strict=True):
            if not isinstance(original, torch.Tensor):
                continue
            if isinstance(original, DTensor):
                made = self.value(original)
                self.values[id(traced)] = made
                self.originals[id(traced._local_tensor)] = original._local_tensor
                self.local_sources[id(original._local_tensor)] = original
                self.local_sources[id(traced._local_tensor)] = original
                slot = get_proxy_slot(traced, proxy.tracer)
                set_proxy_slot(made, proxy.tracer, slot)
                slot.proxy.node.meta['val'] = made
                declaration, _ = _value_ownership(original, original.device_mesh)
                slot.proxy.node.meta['mesh_ownership'] = dict(declaration, mesh=original.device_mesh)
            else:
                self.values[id(original)], self.originals[id(traced)] = traced, original

    def restore(self, graph):
        from .partition import _value_ownership
        from torch.distributed.tensor import DTensor
        for node in graph.graph.nodes:
            if node.op != 'get_attr':
                continue
            owner = graph
            names = node.target.split('.')
            for name in names[:-1]:
                owner = getattr(owner, name)
            captured = getattr(owner, names[-1])
            original = self.originals.get(id(captured))
            if original is None:
                continue
            setattr(owner, names[-1], original)
            if isinstance(original, DTensor):
                declaration, _ = _value_ownership(original, original.device_mesh)
                node.meta['mesh_ownership'] = dict(declaration, mesh=original.device_mesh)
            local_source = self.local_sources.get(id(original))
            if local_source is not None:
                declaration, _ = _value_ownership(local_source, local_source.device_mesh)
                node.meta['mesh_local_source'] = dict(declaration, mesh=local_source.device_mesh)
        graph.meta['mesh_scopes'] = tuple(self.meshes)

    def gradients(self, parameters):
        from torch.distributed.tensor import DTensor
        from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot
        from .partition import _value_ownership
        proxy = get_proxy_mode()
        for ordinal, parameter in enumerate(parameters):
            gradient = self.value(parameter).grad
            if gradient is None:
                continue
            node = get_proxy_slot(gradient, proxy.tracer).proxy.node
            node.meta['mesh_gradient'] = ordinal
            if isinstance(parameter, DTensor):
                declaration, _ = _value_ownership(parameter, parameter.device_mesh)
                node.meta['mesh_ownership'] = dict(declaration, mesh=parameter.device_mesh)




@torch.library.custom_op('torch_mesh::select_exchange', mutates_args=())
def select_exchange(x: torch.Tensor, shape: list[int], selections: str, group_name: str) -> torch.Tensor:
    """design/compiler-followups-2026-10-30.md#selected-page-crossings"""
    raise RuntimeError('select_exchange must be bound to its declared pages before execution')


@select_exchange.register_fake
def _select_exchange_fake(x, shape, selections, group_name):
    return x.new_empty(shape)


@torch.library.custom_op('torch_mesh::select_reduce', mutates_args=())
def select_reduce(x: torch.Tensor, zero: torch.Tensor | None, shape: list[int], selections: str,
                  reduce_op: str, group_name: str) -> torch.Tensor:
    """Combine the declared contributors directly into selected output pages."""
    raise RuntimeError('select_reduce must be bound to its declared pages before execution')


@select_reduce.register_fake
def _select_reduce_fake(x, zero, shape, selections, reduce_op, group_name):
    return x.new_empty(shape)


def _reduce_plan(node, named):
    """Partition exact contributor sets with the shared native relation cover."""
    if 'mesh_selection_reductions' not in node.meta:
        from mesh_relation import Algebra
        from . import _stream
        declaration, algebra = json.loads(named['selections']), Algebra()
        sources = [algebra.selection_relation(value) for value in declaration['sources']]
        targets = [algebra.selection_relation(value) for value in declaration['destinations']]
        sets = [relation.image(relation.domain()) for relation in [*sources, *targets]]
        count = len(sources)
        extent = math.prod(declaration['sources'][0]['shape'])
        plans, missing = [], []
        for domain, members in algebra.cover(sets):
            owners = [rank - count for rank in members if rank >= count]
            contributors = tuple(rank for rank in members if rank < count)
            selected = algebra.selection_relation({'shape': [extent], 'steps': [['select', 0, domain.spans()]]})
            def local(relation):
                positions = selected.then(relation.inverse((extent,)))
                return _stream.TensorLayout(positions.shape, (1,), index_shape=relation.shape,
                    index_coordinates=json.dumps([positions.terms()]))
            for rank in owners:
                write = local(targets[rank])
                if contributors:
                    plans.append((rank, contributors, {source: local(sources[source]) for source in contributors}, write))
                else:
                    missing.append((rank, write))
        node.meta['mesh_selection_reductions'] = tuple(plans), tuple(missing)
    return node.meta['mesh_selection_reductions']


def _bind_reduce(source, destination, plan, group, reduction):
    from . import backend
    from .operand import TensorView
    bindings = []
    source = TensorView.of(source)
    for root, contributors, reads, write in plan:
        if contributors == (root,):
            continue  # The common placement already aliases this identity edge.
        read = (TensorView(source.storage, _selected_layout(source, reads[group.rank()]))
                if group.rank() in reads else source.reshape((source.numel(),)).slice(0, 0, 0))
        output = TensorView(TensorView.of(destination).storage, _selected_layout(destination, write)) if root == group.rank() else None
        call = backend.Call(backend.REDUCE, write.count, source.dtype, root=root,
                            op=_operation(reduction), contributors=contributors)
        bindings.append((call, read, output))
    return backend.prepare_fragments(bindings, group) if bindings else (None, [])


def _select_exchange_context(ctx, inputs, output):
    x, shape, selections, group_name = inputs
    ctx.shape, ctx.selections, ctx.group_name = list(x.shape), selections, group_name


def _select_exchange_backward(ctx, gradient):
    from mesh_relation import Algebra
    from .algebra import index_values
    algebra, declaration = Algebra(), json.loads(ctx.selections)
    rank = dist.distributed_c10d._resolve_process_group(ctx.group_name).rank()
    selected = [algebra.selection_relation(value) for value in declaration['destinations']]
    domains = [relation.image(relation.domain()) for relation in selected]
    extent = math.prod(declaration['sources'][0]['shape'])
    compact = [{'shape': [extent], 'steps': [['select', 0, domain.spans()]]} for domain in domains]
    placed = algebra.selection_relation(compact[rank])
    indices = selected[rank].then(placed.inverse((extent,)))
    local = torch.zeros(placed.shape, dtype=gradient.dtype, device=gradient.device)
    if gradient.numel():
        local = local.scatter_add(0, index_values(indices.terms(), gradient.numel(), gradient.device), gradient.reshape(-1))
    covered = algebra.set('{ [i] : false }', 1)
    for domain in domains:
        covered = covered.binary(domain, 'union')
    holes = any(not algebra.selection(value).binary(covered, 'subtract').empty() for value in declaration['sources'])
    zero = gradient.new_zeros((1,)) if holes else None
    result = select_reduce(local, zero, ctx.shape,
        json.dumps({'sources': compact, 'destinations': declaration['sources']}, separators=(',', ':')), 'sum', ctx.group_name)
    return result, None, None, None


select_exchange.register_autograd(_select_exchange_backward, setup_context=_select_exchange_context)


def _exchange_plan(node, named):
    """design/compiler-followups-2026-10-30.md#selected-page-crossings"""
    if 'mesh_selection_exchanges' not in node.meta:
        from .collective_views import selection_exchanges
        declaration = json.loads(named['selections'])
        node.meta['mesh_selection_exchanges'] = selection_exchanges(
            declaration['sources'], declaration['destinations'])
    return node.meta['mesh_selection_exchanges']


def _selected_layout(value, selection):
    from . import _stream
    from .operand import descriptor
    logical = _stream.TensorLayout((value.numel(),), (1,))
    return selection.rebase(logical, descriptor(value).reshape((value.numel(),)))


def _bind_exchange(source, destination, plan, group):
    from .operand import TensorView
    deliveries = []
    for a, b, read, write in plan:
        if a == b:
            continue
        if group.rank() == a:
            value = TensorView(TensorView.of(source).storage, _selected_layout(source, read))
        elif group.rank() == b:
            value = TensorView(TensorView.of(destination).storage, _selected_layout(destination, write))
        else:
            value = torch.empty(read.count, dtype=source.dtype, device='meta')
        deliveries.append((value, a, b))
    return group.prepare_writes(deliveries) if deliveries else (None, [])


def effects(nodes):
    """FX's declared mutation/random effects, shared by selection and partition."""
    random_effects = True
    if torch._guards.TracingContext.try_get():
        from torch._inductor import config
        random_effects = config.fallback_random
    return [i for i, node in enumerate(nodes) if node.is_impure(random_effects)]


def spans(graph, cuts=()):
    """Adapt FX values and effects to the common ordered dependency slicer."""
    from mesh_graph import needed
    graph.lint()
    nodes = list(graph.nodes)
    selected = needed([node.all_input_nodes for node in nodes], [(node,) for node in nodes], (), cuts,
                      effects(nodes))
    live = {i for span in selected for i in span}
    for i in range(len(nodes) - 1, -1, -1):
        if i not in live:
            graph.erase_node(nodes[i])
    if graph.owning_module is not None:
        attributes = {node.target for node in graph.nodes if node.op == 'get_attr'}
        for name, child in graph.owning_module.named_children():
            if name in attributes and isinstance(child, torch.fx.GraphModule):
                spans(child.graph)
                child.recompile()
    return [[nodes[i] for i in span] for span in selected]


def _value(argument):
    """A node's traced value; None for an operator that returns nothing (make_fx records no value for one, such as
    allgatherv_out, which writes its `out` operand)."""
    if isinstance(argument, torch.fx.Node):
        if 'val' in argument.meta:
            return argument.meta['val']
        if argument.op == 'call_function' and not getattr(getattr(argument.target, '_schema', None), 'returns', True):
            return None
        raise ValueError(f'collective operand {argument.name} has no AOT tensor metadata')
    return argument


def operation(node):
    """The underlying operator and metadata, including functionalized writes.

    PyTorch wraps mutating custom operators until its final reinplacing pass.
    Read those wrappers through PyTorch's own view metadata so an out-parameter
    write has the same declaration as its eager call, without executing it.
    """
    from torch.utils._pytree import tree_map
    from torch._higher_order_ops.auto_functionalize import (
        auto_functionalized, auto_functionalized_v2, _generate_new_op_kwargs_from_bases)
    if node.op != 'call_function':
        return None, {}
    wrapped = node.target in (auto_functionalized, auto_functionalized_v2)
    target = node.args[0] if wrapped else node.target
    schema = getattr(target, '_schema', None)
    if schema is None:
        return None, {}
    if schema.name.split('::', 1)[0] not in ('torch_mesh', '_c10d_functional', '_c10d_functional_autograd'):
        return target, {}
    if wrapped:
        values = tree_map(_value, dict(node.kwargs))
        if node.target is auto_functionalized_v2:
            bases = values.pop('_all_bases', [])
            values, _ = _generate_new_op_kwargs_from_bases(
                schema, values, bases, (), torch._library.utils.is_out(target))
        args = {a.name: values.get(a.name, a.default_value) for a in schema.arguments}
    else:
        args = {a.name: tree_map(_value, node.args[i] if i < len(node.args)
                                else node.kwargs.get(a.name, a.default_value))
                for i, a in enumerate(schema.arguments)}
    return target, args


def _operation(name):
    names = {'sum': 'SUM', 'product': 'PRODUCT', 'max': 'MAX', 'min': 'MIN', 'avg': 'AVG'}
    if name not in names:
        raise ValueError(f'unknown framework reduction {name}')
    return getattr(dist.ReduceOp, names[name])


def _collective_results(node, target, arguments):
    """Use functionalization's declared result views and untouched base fields."""
    from torch.utils._pytree import tree_map, tree_leaves
    from torch._higher_order_ops.auto_functionalize import (
        auto_functionalized, auto_functionalized_v2, get_mutable_args,
        _generate_new_op_kwargs_from_bases)
    values, preserved = _value(node), ()
    result_arguments = arguments
    if node.target is auto_functionalized:
        names, _ = get_mutable_args(target)
        result_arguments = dict(arguments, **dict(zip(names, values[-len(names):], strict=True)))
    elif node.target is auto_functionalized_v2 and not torch._library.utils.is_out(target):
        keywords = tree_map(_value, dict(node.kwargs))
        old_bases = keywords.pop('_all_bases', [])
        new_bases = values[-len(old_bases):] if old_bases else ()
        result_arguments, _ = _generate_new_op_kwargs_from_bases(target._schema, keywords, new_bases, (), False)
        names, _ = get_mutable_args(target)
        writes = [value for name in names for value in tree_leaves(result_arguments[name])
                  if isinstance(value, torch.Tensor)]
        preserved = tuple(dict(source=before, result=after,
                               writes=tuple(value for value in writes
                                            if torch._C._is_alias_of(value, after)))
                          for before, after in zip(old_bases, new_bases, strict=True))
    result = result_arguments.get('out')
    if result is None or node.target is auto_functionalized_v2 and torch._library.utils.is_out(target):
        result = values
    return [value for value in tree_leaves(result) if isinstance(value, torch.Tensor)], preserved


def _collective_declaration(node):
    """Normalize source syntax once, retaining math before transport expansion."""
    from . import backend, partition, _stream
    if 'mesh_collective_declaration' in node.meta:
        return node.meta['mesh_collective_declaration']
    target, args = operation(node)
    schema = getattr(target, '_schema', None)
    if schema is None:
        return None
    namespace, name = schema.name.split('::', 1)
    if namespace not in ('_c10d_functional', '_c10d_functional_autograd', 'torch_mesh') or name == 'wait_tensor':
        return None
    group_name = args.get('group_name')
    if group_name is None:
        if namespace == 'torch_mesh':
            return None
        raise ValueError(f'communication operator {schema.name} has no declared group')
    group = dist.distributed_c10d._resolve_process_group(group_name)
    n, calls, mathematical = group.size(), [], []
    x = args.get('input', args.get('x', args.get('self', args.get('partials'))))
    results, preserved = _collective_results(node, target, args)
    reduction = args.get('reduce_op', args.get('op', 'sum'))

    def declare(call, source, result, *, source_axis=None, result_axis=None, reads=None, writes=None, initialization=False):
        mathematical.append(dict(call=call, source=source, result=result, source_axis=source_axis,
                                 result_axis=result_axis, reads=reads, writes=writes, initialization=initialization))
        return call

    def gather(source, result, dim, sizes):
        counts = [int(size) * partition._row(source, dim) for size in sizes]
        declare(backend.Call(backend.ALLGATHER, source.numel(), source.dtype, counts=counts), source, result,
                source_axis=dim, result_axis=dim)
        return partition.gathered(source, dim, sizes, group)

    if namespace == 'torch_mesh' and name == 'select_exchange':
        for source, destination, read, write in _exchange_plan(node, args):
            call = declare(backend.Call(backend.WRITE, read.count, x.dtype, root=source,
                                       destination=destination, fused=True), x, results[0],
                           reads={source: read}, writes={destination: write})
            if source != destination:
                calls.append(call)
    elif namespace == 'torch_mesh' and name == 'select_reduce':
        plans, missing = _reduce_plan(node, args)
        for root, contributors, reads, write in plans:
            call = declare(backend.Call(backend.REDUCE, write.count, x.dtype, root=root,
                                       op=_operation(reduction), contributors=contributors), x, results[0],
                           reads=reads, writes={root: write})
            if contributors != (root,):
                calls.append(call)
        for root, write in missing:
            declare(backend.Call(backend.REDUCE, write.count, x.dtype, root=root,
                                 op=_operation(reduction), contributors=()), args['zero'], results[0],
                    reads={root: _stream.TensorLayout((write.count,), (0,))}, writes={root: write}, initialization=True)
    elif namespace == 'torch_mesh' and name in ('allgatherv', 'allgatherv_out'):
        calls = gather(x, results[0], args['dim'], args['sizes'])
    elif namespace == 'torch_mesh' and name in ('all_reduce', 'linear_sum', 'fused_step', 'fused_step_out'):
        if name == 'all_reduce':
            calls = [declare(backend.Call(backend.ALLREDUCE, x.numel(), x.dtype, op=_operation(reduction)), x, results[0])]
        elif name == 'linear_sum' or args.get('adds'):
            output = _value(node) if name != 'fused_step_out' else args['out']
            calls = partition._gather_writes([output.numel()] * n, output.dtype) if n > 1 else []
        else:
            calls = partition.gathered(x, args['dim'], args['sizes'], group)
    elif namespace == 'torch_mesh' and name == 'reduce_scatterv':
        counts = [int(size) * partition._row(x, args['dim']) for size in args['sizes']]
        calls = [declare(backend.Call(backend.REDUCE_SCATTER, x.numel(), x.dtype, counts=counts, op=_operation(reduction)),
                         x, results[0], source_axis=args['dim'], result_axis=args['dim'])]
    elif namespace == 'torch_mesh' and name == 'merge_partials':
        operator = group.partials_operator(x.shape[-1])
        root, rows = args['root'], args.get('rows', [])
        calls = [declare(backend.Call(backend.REDUCE_SCATTER if rows else backend.ALLREDUCE if root < 0 else backend.REDUCE,
                                     x.numel(), x.dtype, root=max(0, root), op=operator, element_shape=(x.shape[-1],),
                                     counts=[r * x.shape[-1] for r in rows] if rows else None), x, results[0])]
    elif name in ('all_reduce', 'all_reduce_'):
        calls = [declare(backend.Call(backend.ALLREDUCE, x.numel(), x.dtype, op=_operation(reduction)), x, results[0])]
    elif name in ('broadcast', 'broadcast_'):
        calls = [declare(backend.Call(backend.BROADCAST, x.numel(), x.dtype, root=args['src']), x, results[0])]
    elif name in ('all_gather_into_tensor', 'all_gather_into_tensor_out'):
        calls = gather(x, results[0], 0, [x.shape[0]] * n)
    elif name == 'reduce_scatter_tensor':
        if x.numel() % n:
            raise ValueError('reduce-scatter metadata is not divisible by the group size')
        calls = [declare(backend.Call(backend.REDUCE_SCATTER, x.numel(), x.dtype,
                                     counts=[x.numel() // n] * n, op=_operation(reduction)), x, results[0])]
    elif name == 'all_to_all_single':
        width = math.prod(x.shape[1:])
        incoming = args['output_split_sizes'] or [x.shape[0] // n] * n
        outgoing = args['input_split_sizes'] or [x.shape[0] // n] * n
        calls = [declare(backend.Call(backend.ALLTOALL, x.numel(), x.dtype,
                                     counts=[int(rows) * width for rows in (*outgoing, *incoming)]), x, results[0])]
    elif name in ('all_reduce_coalesced', 'all_gather_into_tensor_coalesced', 'reduce_scatter_tensor_coalesced'):
        for tensor, result in zip(args['inputs'], results, strict=True):
            if name == 'all_reduce_coalesced':
                calls.append(declare(backend.Call(backend.ALLREDUCE, tensor.numel(), tensor.dtype,
                                                 op=_operation(reduction)), tensor, result))
            elif name == 'all_gather_into_tensor_coalesced':
                calls.extend(gather(tensor, result, 0, [tensor.shape[0]] * n))
            else:
                calls.append(declare(backend.Call(backend.REDUCE_SCATTER, tensor.numel(), tensor.dtype,
                                                 counts=[tensor.numel() // n] * n, op=_operation(reduction)), tensor, result))
    else:
        raise NotImplementedError(f'AOT collective declaration for {schema.name}')
    declaration = dict(node=node, operation=target, arguments=args, group_name=group_name, group=group,
                       calls=calls, mathematical=tuple(mathematical), preserved=preserved)
    node.meta['mesh_collective_declaration'] = declaration
    return declaration


def lower(graph, *, forms=None):
    """Transport Calls from the source's retained mathematical declarations."""
    from . import backend
    groups, trains = {}, {}
    for node in (graph.graph if hasattr(graph, 'graph') else graph).nodes:
        declaration = _collective_declaration(node)
        if declaration is None:
            continue
        target, args = declaration['operation'], declaration['arguments']
        group_name, group, calls = declaration['group_name'], declaration['group'], declaration['calls']
        name, n = target._schema.name.split('::', 1)[1], group.size()
        x = args.get('input', args.get('x', args.get('self', args.get('partials'))))
        if calls:
            fused_calls = all(call.kind == backend.WRITE and call.fused for call in calls)
            repeats = int(args.get('repeats', 1)) if fused_calls else 1
            if repeats < 1:
                raise ValueError('compiled collective repeat count must be positive')
            if group_name in trains:
                expected, remaining = trains[group_name]
                if calls != expected:
                    raise ValueError('another collective interrupts a compiled fused repeat train')
                if remaining == 1:
                    del trains[group_name]
                else:
                    trains[group_name] = expected, remaining - 1
                continue
            groups.setdefault(group_name, []).append((calls, repeats) if repeats > 1 else calls)
            if forms is not None and fused_calls:
                weight, adds = args.get('w'), name in ('all_reduce', 'linear_sum') or args.get('adds', False)
                source = x.view(1) if x.ndim == 0 else x
                dim = args.get('dim', 0)
                sizes = args.get('sizes', [source.shape[dim]] * n)
                forms.append((source, weight, dim, tuple(sizes), group_name, adds, repeats))
            if repeats > 1:
                trains[group_name] = calls, repeats - 1
    if trains:
        raise ValueError('the function graph ends inside a fused repeat train')
    return groups


def _collective_relations(declaration, mathematical, algebra):
    """Typed logical views of every rank, before storage and routing cuts."""
    from .collective_views import _domains, domain_relations, complete_calls
    from .operand import layout_configuration
    call, group = mathematical['call'], declaration['group']
    made = complete_calls([call], group)[0]
    if made is not call:
        mathematical['call'] = made
        declaration['calls'][:] = [made if value is call else value for value in declaration['calls']]
        call = made
    reads, writes = {}, {}
    if mathematical['reads'] is not None:
        return ({rank: algebra.layout(layout_configuration(layout)) for rank, layout in mathematical['reads'].items()},
                {rank: algebra.layout(layout_configuration(layout)) for rank, layout in mathematical['writes'].items()})

    def order(value, count, axis):
        shape, steps = [count], []
        if axis is not None and count:
            shape = list(map(int, value.shape))
            axis %= len(shape)
            shape[axis] = count // math.prod(shape[:axis] + shape[axis + 1:])
            steps = [['permute', [axis, *(i for i in range(len(shape)) if i != axis)]], ['reshape', [count]]]
        return algebra.selection_relation(dict(shape=shape, steps=steps))

    for rank in range(group.size()):
        _, sends, receives = _domains(call, rank, group.size())
        source, result = domain_relations(call, rank, group.size(), algebra)
        if sends:
            reads[rank] = source.then(order(mathematical['source'], sum(last - first for first, last in sends),
                                            mathematical['source_axis']))
        if receives:
            writes[rank] = result.then(order(mathematical['result'], sum(last - first for first, last in receives),
                                             mathematical['result_axis']))
    return reads, writes


def boundary_groups(section):
    """The existing mathematical declarations supply their group operands."""
    return {declared['group_name']: declared['group'] for node in section['boundaries']
            if (declared := _collective_declaration(node)) is not None}


def boundary_projection(section, outputs, algebra, *, ranks=None):
    """Project declared collective math, retaining all peer demands explicitly.

    ``outputs`` requests this rank's original source fields. When other ranks'
    requests are known, ``section['peer_outputs'][group_name][rank]`` supplies
    them in the same field coordinates. An absent peer request remains pending;
    it is never presumed equal to the local request or to the entire result.
    ``ranks`` supplies an explicitly selected rank per group when projecting
    another peer's declared program. Returned peer inputs and call records retain contributors and reduction
    meaning independently of the routing compiler's transport Call expansion.
    """
    from torch.utils import _pytree
    from . import backend
    from .operand import TensorView, byte_relation, layout_configuration
    from .collective_views import _domains

    def fields(nodes):
        return [((node, tuple(path)), value) for node in nodes
                for path, value in _pytree.tree_flatten_with_path(_value(node))[0]
                if isinstance(value, torch.Tensor)]

    input_fields, output_fields = fields(section['inputs']), fields(section['outputs'])
    source_nodes = {value: key for key, value in section['graph'].meta['mesh_source_results'].items()
                    if isinstance(value, torch.fx.Node)}
    inputs, peer_inputs, records, pending = {}, {}, [], []

    def local_rank(declaration):
        return declaration['group'].rank() if ranks is None else ranks[declaration['group_name']]

    def join(mapping, key, selected):
        mapping[key] = selected if key not in mapping else mapping[key].binary(selected, 'union')

    def bindings(value, candidates):
        direct = [(key, None) for key, tensor in candidates if tensor is value]
        if direct:
            return direct
        if value is None:
            return []
        actual = TensorView.of(value)
        result = []
        def bytes_(view):
            return byte_relation(dict(dtype=view.dtype, storage_bytes=view.storage.untyped_storage().nbytes(),
                                      relation=algebra.layout(layout_configuration(view.layout))))
        for key, tensor in candidates:
            original = TensorView.of(tensor)
            if actual.storage.untyped_storage()._cdata == original.storage.untyped_storage()._cdata:
                result.append((key, bytes_(actual).then(bytes_(original).inverse((original.numel(),)))))
        return result

    def preserved_fields(declaration, requests, group_inputs):
        for field in declaration['preserved']:
            result = field['result']
            unchanged = algebra.selection(dict(shape=[result.numel()], steps=[]))
            for written in field['writes']:
                for _, relation in bindings(written, [(None, result)]):
                    covered = unchanged if relation is None else relation.image(relation.domain())
                    unchanged = unchanged.binary(covered, 'subtract')
            for rank, requested in requests.items():
                needed = algebra.set('{ [i] : false }', 1)
                for key, relation in bindings(result, output_fields):
                    if key in requested:
                        positions = requested[key] if relation is None else relation.preimage(requested[key])
                        needed = needed.binary(positions, 'union')
                needed = needed.binary(unchanged, 'intersect')
                if needed.empty():
                    continue
                for key, relation in bindings(field['source'], input_fields):
                    positions = needed if relation is None else relation.image(needed)
                    join(group_inputs.setdefault(rank, {}), key, positions)
                    if rank == local_rank(declaration):
                        join(inputs, key, positions)
                records.append(dict(boundary=source_nodes.get(declaration['node'], declaration['node']),
                    operation=declaration['operation'], group=declaration['group'],
                    group_name=declaration['group_name'], source=field['source'], result=result,
                    preserved=True, rank=rank, positions=needed,
                    source_fields=tuple(bindings(field['source'], input_fields)),
                    result_fields=tuple(bindings(result, output_fields))))

    for node in section['boundaries']:
        declaration = _collective_declaration(node)
        if declaration is None or not declaration['mathematical']:
            peers = section.get('peer_outputs', {}).get(declaration['group_name'], {}) if declaration else {}
            if any(not value.empty() for request in (outputs, *peers.values()) for value in request.values()):
                pending.append(dict(boundary=source_nodes.get(node, node), outputs=outputs))
            continue
        group, group_name = declaration['group'], declaration['group_name']
        requests = {rank: dict(values) for rank, values in section.get('peer_outputs', {}).get(group_name, {}).items()}
        at = local_rank(declaration)
        local = requests.setdefault(at, {})
        for key, selected in outputs.items():
            join(local, key, selected)
        group_inputs = peer_inputs.setdefault(group_name, {})
        preserved_fields(declaration, requests, group_inputs)
        for mathematical in declaration['mathematical']:
            read_relations, write_relations = _collective_relations(declaration, mathematical, algebra)
            call = mathematical['call']
            result_fields = bindings(mathematical['result'], output_fields)
            source_fields = bindings(mathematical['source'], input_fields)
            total = _domains(call, at, group.size())[0]
            required = algebra.set('{ [i] : false }', 1)
            requested = {}
            for rank, relation in write_relations.items():
                if rank not in requests:
                    pending.append(dict(boundary=source_nodes.get(node, node), group=group_name, rank=rank,
                                        result=mathematical['result'], relation=relation, requirement='peer_outputs'))
                    continue
                for key, view in result_fields:
                    if key not in requests[rank]:
                        continue
                    selected = requests[rank][key]
                    selected = selected if view is None else view.preimage(selected)
                    terms = relation.preimage(selected)
                    required = required.binary(terms, 'union')
                    join(requested, rank, terms)
            unit = math.prod(call.element_shape)
            if unit != 1:
                fiber = algebra.map((total,), (total,), lambda x, y: [f'floor({x[0]} / {unit}) = floor({y[0]} / {unit})'])
                required = fiber.image(required)
                requested = {rank: fiber.image(terms) for rank, terms in requested.items()}
            reads = {rank: relation.image(required) for rank, relation in read_relations.items()}
            for rank, selected in reads.items():
                if selected.empty():
                    continue
                if not source_fields:
                    pending.append(dict(boundary=source_nodes.get(node, node), group=group_name, rank=rank,
                                        source=mathematical['source'], positions=selected, requirement='source_field'))
                for key, view in source_fields:
                    positions = selected if view is None else view.image(selected)
                    join(group_inputs.setdefault(rank, {}), key, positions)
                    if rank == at:
                        join(inputs, key, positions)
            if not result_fields:
                pending.append(dict(boundary=source_nodes.get(node, node), result=mathematical['result'],
                                    outputs=outputs, requirement='result_field'))
            contributors = tuple(read_relations) if call.contributors is None else tuple(call.contributors)
            divisor = (len(contributors) if call.op in (dist.ReduceOp.AVG, backend.LIB.ncclAvg)
                       and not mathematical['initialization'] else None)
            records.append(dict(boundary=source_nodes.get(node, node), operation=declaration['operation'],
                group=group, group_name=group_name, call=call, contributors=contributors,
                recipients=tuple(write_relations), element_shape=call.element_shape, divisor=divisor,
                source=mathematical['source'], result=mathematical['result'], initialization=mathematical['initialization'],
                source_fields=tuple(source_fields), result_fields=tuple(result_fields),
                read_relations=read_relations, write_relations=write_relations, requested=requested,
                work=required, reads=reads))
    return dict(inputs=inputs, peer_inputs=peer_inputs, records=tuple(records), pending=tuple(pending))


def trace(function, *arguments, parameters=()):
    """Trace one complete function, including its backward, using FakeTensor.

    Parameters' real gradient references are saved and restored; fake backward
    neither accumulates into an existing real gradient nor becomes model state.
    The caller supplies a static function. Data-dependent Python branches are
    configuration errors, never an excuse to run a hidden measurement or probe.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    parameters = tuple(parameters)
    gradients = [parameter.grad for parameter in parameters]
    mode = detect_fake_mode(arguments) or FakeTensorMode(allow_non_fake_inputs=True)
    source = SourceCapture(mode)

    def recorded(*values):
        source.bind_arguments(values, arguments)
        with source:
            output = function(*values)
            source.gradients(parameters)
            return output

    try:
        with mode:
            for parameter in parameters:
                source.value(parameter).grad = None
            graph = make_fx(recorded, decomposition_table={}, tracing_mode='fake',
                            _allow_non_fake_inputs=True)(*arguments)
        source.restore(graph)
    finally:
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
    # Framework layout queries can leave unused tensor operations in make_fx's
    # graph (including a mesh-coordinate nonzero). They are configuration work,
    # not part of the declared function or its explicit gradient outputs.
    spans(graph.graph)
    graph.recompile()
    return graph


class Execution:
    """Configuration which appends every numerical span and crossing to one native function."""
    def __init__(self, steps, output=None, pages=(), initializations=()):
        from . import _stream
        self.steps, self.output, self.pages = tuple(steps), output, tuple(pages)
        self.initializations = tuple(initializations)
        self._initialized = False
        self.executions = _stream.Executions(output)
        self.forward = self.executions.run
        self.completion = self.executions.completion

    def append(self, schedule):
        # This is declaration-time state: only the first prepared invocation
        # contains imports. Runtime submits the already recorded schedules.
        if not self._initialized:
            for source, destination in self.initializations:
                schedule.copy(source, destination)
        for step in self.steps:
            step.append(schedule)
        self._initialized = True

    def realize(self, repetitions=1):
        from . import _stream
        for _ in range(repetitions):
            schedule = _stream.Schedule()
            self.append(schedule)
            schedule.finish()
            self.executions.add(schedule)


def execution_sequence(steps, output=None, pages=(), initializations=()):
    return Execution(steps, output, pages, initializations)


_BOUND = {'all_to_all_single', 'merge_partials', 'reduce_scatterv', 'allgatherv', 'allgatherv_out', 'select_exchange', 'select_reduce',
          'reduce_scatter_tensor', 'all_gather_into_tensor', 'all_gather_into_tensor_out',
          'broadcast', 'broadcast_', 'all_reduce_coalesced', 'all_gather_into_tensor_coalesced',
          'reduce_scatter_tensor_coalesced'}


def _bound(node):
    target, args = operation(node)
    schema = getattr(target, '_schema', None)
    if schema is None or args.get('group_name') is None:
        return False
    name = schema.name.split('::')[-1]
    if name in ('all_reduce', 'all_reduce_'):
        return True
    return name in _BOUND


def normalize_edges(module):
    """Retain result dependencies without functional collective wait nodes.

    Return removed-node aliases so preparation retains source result identities.
    The native collective binding supplies the actual page dependency.
    """
    attributes, aliases = {}, {}
    for node in list(module.graph.nodes):
        replacement = None
        if node.op == 'get_attr':
            replacement = attributes.get(node.target)
            attributes.setdefault(node.target, node)
        target, _ = operation(node)
        if getattr(getattr(target, '_schema', None), 'name', '') == '_c10d_functional::wait_tensor':
            replacement = node.args[0]
        if replacement is not None:
            aliases[node] = replacement
            node.replace_all_uses_with(replacement)
            module.graph.erase_node(node)
    if aliases:
        from torch.fx.node import map_arg
        def original(node):
            while node in aliases:
                node = aliases[node]
            return node
        for field in ('mesh_source_results', 'mesh_source_fields'):
            if field in module.meta:
                module.meta[field] = {key: map_arg(tree, original) for key, tree in module.meta[field].items()}
        module.recompile()
    return aliases


def sections(module, cuts, requested_outputs=()):
    """Materialize the common graph partition as FX fragments, retaining metadata."""
    from mesh_graph import partition
    nodes = list(module.graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    output = next(node for node in nodes if node.op == 'output')
    constants = [node for node in nodes if node.op == 'get_attr']
    pieces = partition([node.all_input_nodes for node in nodes], [(node,) for node in nodes],
                       [*output.all_input_nodes, *requested_outputs], cuts,
                       effects=effects(nodes),
                       constants=constants,
                       external=[i for i, node in enumerate(nodes) if node.op in ('placeholder', 'get_attr', 'output')])
    for indices, inputs, outputs in pieces:
        if not indices:
            continue
        inputs, outputs = sorted(inputs, key=order.__getitem__), sorted(outputs, key=order.__getitem__)
        graph, bound = torch.fx.Graph(), {}
        for node in inputs:
            bound[node] = graph.placeholder(node.name, type_expr=node.type)
            bound[node].meta = node.meta.copy()
        def operand(node):
            if node not in bound:
                if node.op != 'get_attr':
                    raise ValueError(f'undeclared fragment input: {node.name}')
                bound[node] = graph.node_copy(node, operand)
            return bound[node]
        for index in indices:
            node = nodes[index]
            bound[node] = graph.node_copy(node, operand)
        graph.output(tuple(operand(node) for node in outputs))
        graph.lint()
        fragment = torch.fx.GraphModule(module, graph)
        from torch.fx.node import map_arg
        from torch.utils import _pytree
        results, fields = dict(bound), {}
        for source, tree in module.meta.get('mesh_source_results', {}).items():
            leaves = _pytree.tree_flatten_with_path(tree)[0]
            if all(not isinstance(value, torch.fx.Node) or value in bound for _, value in leaves):
                results[source] = map_arg(tree, bound.__getitem__)
            for path, value in leaves:
                if isinstance(value, torch.fx.Node) and value in bound:
                    fields[source, tuple(path)] = bound[value]
        for key, tree in module.meta.get('mesh_source_fields', {}).items():
            if all(not isinstance(value, torch.fx.Node) or value in bound for value in _pytree.tree_leaves(tree)):
                fields[key] = map_arg(tree, bound.__getitem__)
        fragment.meta = dict(module.meta, mesh_source_results=results, mesh_source_fields=fields)
        yield fragment, tuple(inputs), tuple(outputs)


def _gathers(target, named, outputs, group):
    """Gather sources and final views, independent of the frontend spelling."""
    name = target._schema.name.split('::', 1)[1]
    from torch.utils._pytree import tree_flatten
    destinations = [named['out']] if named.get('out') is not None else tree_flatten(outputs)[0]
    if name in ('allgatherv', 'allgatherv_out'):
        return [(named['x'], destinations[0], named['dim'], named['sizes'])]
    if name in ('all_gather_into_tensor', 'all_gather_into_tensor_out', 'all_gather_into_tensor_coalesced'):
        sources = named['inputs'] if name.endswith('_coalesced') else [named['input']]
        return [(source, destination, 0, [source.shape[0]] * group.size())
                for source, destination in zip(sources, destinations, strict=True)]
    return None


def _boundary_work(section, projection, algebra):
    """The same selected storage operands for declared collective work."""
    from .ownership import _source_leaves
    from .operand import TensorView, byte_relation, layout_configuration
    from .collective_views import selected_calls

    def binding(value):
        view = TensorView.of(value)
        return dict(name=value.untyped_storage()._cdata, dtype=value.dtype,
                    storage_bytes=value.untyped_storage().nbytes(),
                    relation=algebra.layout(layout_configuration(view.layout)))

    inputs = {key: binding(value) for key, value in _source_leaves(section['inputs'])
              if isinstance(value, torch.Tensor)}
    outputs = {key: binding(value) for key, value in _source_leaves(section['outputs'])
               if isinstance(value, torch.Tensor)}
    domains, records, equalities, commands = {}, [], [], []

    def retain(access, work):
        name = access['name']
        positions = byte_relation(access).image(work)
        domains[name] = positions if name not in domains else domains[name].binary(positions, 'union')
        return dict(access=access, relation=byte_relation(access), positions=positions)

    def use(value, relation):
        described = binding(value)
        return dict(binding=described, relation=relation.then(described['relation']), shape=relation.shape)

    for record in projection.get('records', ()):
        group, rank = record['group'], record['group'].rank()
        if record.get('preserved'):
            if record['rank'] == rank:
                order = algebra.selection_relation(dict(shape=[record['result'].numel()],
                    steps=[['select', 0, record['positions'].spans()]]))
                equalities.append((use(record['source'], order), use(record['result'], order)))
            continue
        for selected in selected_calls(record, algebra):
            roles, accesses = {}, {'reads': {}, 'writes': {}}
            work = None
            for side, value in (('reads', record['source']), ('writes', record['result'])):
                relation = selected[side].get(rank)
                if relation is None:
                    roles[side] = None
                    continue
                described = use(value, relation)
                work = algebra.selection(dict(shape=list(relation.shape), steps=[]))
                access = dict(described['binding'], relation=described['relation'])
                accesses[side][side] = retain(access, work)
                roles[side] = described
            if work is not None:
                records.append(dict(work=work, **accesses))
            if selected['identity'] and roles['reads'] is not None and roles['writes'] is not None:
                equalities.append((roles['reads'], roles['writes']))
            if (selected['identity'] and selected['call'].contributors == selected['call'].recipients):
                continue
            commands.append(dict(group=group, group_name=record['group_name'], call=selected['call'], **roles))
    return dict(inputs=inputs, outputs=outputs, projection=dict(storage_bytes=domains, records=tuple(records)),
                equalities=tuple(equalities), commands=tuple(commands))


class BoundGraph:
    """Configuration-time binding of ordinary collective graph boundaries.

    The producer span writes the source region; the native collective binds
    that region and its final output. There is no eager transport call in the
    generated execution sequence. Numerical spans use the existing compiler.
    """
    def __init__(self, graph, arguments, group, configuration, destinations=None, *, prepared=None, ownership=None):
        if ownership is not None:
            self.prepared = ownership['prepared']
            self.groups, self.steps = {}, []
            self._bind_owned(ownership, destinations)
            self.execution.initializations = tuple(graph.meta.get('mesh_initializations', ()))
            return
        from .compilation import lift_attributes
        if prepared is None:
            graph, arguments = lift_attributes(graph, arguments)
            from .ownership import prepare_expression
            prepared = prepare_expression(dict(graph=graph, operands=tuple(arguments)))
        else:
            graph = prepared['expression']['graph']
        self.prepared = prepared
        from torch.fx.node import map_arg
        from torch.utils._pytree import tree_map, tree_flatten
        from . import backend, host
        from .native import empty_strided
        from .placement import Placements, alias
        values, arguments = {}, iter(arguments)
        self.groups, self.steps = {}, []
        for node in graph.graph.nodes:
            if node.op == 'placeholder':
                values[node] = next(arguments)
            elif node.op == 'get_attr':
                value = graph
                for name in node.target.split('.'):
                    value = getattr(value, name)
                values[node] = value
        placements = Placements()
        for node, value in values.items():
            for described, actual in zip(tree_flatten(_value(node))[0], tree_flatten(value)[0], strict=True):
                if isinstance(described, torch.Tensor):
                    placements.bind(described, actual)
        result = next(n.args[0] for n in graph.graph.nodes if n.op == 'output')
        if destinations is not None:
            for described, destination in zip(tree_flatten(map_arg(result, _value))[0], destinations, strict=True):
                if destination is not None:
                    placements.bind(described, destination)
        from .layout import place
        # Collect view equalities before assigning any physical page. Full
        # storage families retain the producer's untouched sibling values.
        for node in graph.graph.nodes:
            if not _bound(node):
                continue
            target, named = operation(node)
            native = dist.distributed_c10d._resolve_process_group(named['group_name'])
            if target is torch.ops.torch_mesh.select_exchange.default:
                for source, destination, read, write in _exchange_plan(node, named):
                    if source == destination == native.rank():
                        placements.equate(named['x'], _value(node),
                            _selected_layout(named['x'], read), _selected_layout(_value(node), write))
            if target is torch.ops.torch_mesh.select_reduce.default:
                from . import _stream
                plans, missing = _reduce_plan(node, named)
                for rank, write in missing:
                    if rank == native.rank():
                        placements.equate(named['zero'], _value(node),
                            _stream.TensorLayout((write.count,), (0,)), _selected_layout(_value(node), write))
                for root, contributors, reads, write in plans:
                    if contributors == (root,) and root == native.rank():
                        placements.equate(named['x'], _value(node),
                            _selected_layout(named['x'], reads[root]), _selected_layout(_value(node), write))
            for source, described, dim, sizes in _gathers(target, named, _value(node), native) or ():
                own = described.narrow(dim, sum(sizes[:native.rank()]), sizes[native.rank()])
                placements.equate(source, own)
        place(graph.graph, placements)
        self.placements = placements
        for node in values:
            values[node] = tree_map(lambda declared, actual: placements.get(declared)
                if isinstance(declared, torch.Tensor) else actual, _value(node), values[node])
        for section in prepared['sections']:
            module, inputs, exported = section['graph'], section['inputs'], section['outputs']
            args = tuple(values[node] for node in inputs)
            live = {n: v for n, v in zip((n for n in module.graph.nodes if n.op == 'placeholder'), args)}
            for n in module.graph.nodes:
                if n.op == 'get_attr':
                    value = module
                    for name in n.target.split('.'):
                        value = getattr(value, name)
                    live[n] = value
            outputs = next(n.args[0] for n in module.graph.nodes if n.op == 'output')
            aliases = [(described, value) for dependency, operand in live.items()
                       for described, value in zip(tree_flatten(_value(dependency))[0], tree_flatten(operand)[0])
                       if isinstance(described, torch.Tensor)]
            def storage(n):
                def bind(metadata):
                    if not isinstance(metadata, torch.Tensor):
                        return metadata
                    destination = placements.get(metadata)
                    if destination is not None:
                        return destination
                    for described, value in aliases:
                        if torch._C._is_alias_of(metadata, described):
                            return alias(described, value, metadata)
                    return empty_strided(tuple(metadata.shape), tuple(metadata.stride()), metadata.dtype, metadata.device.type)
                return tree_map(bind, _value(n))
            host_call = next((n for n in module.graph.nodes if host.bound(n)), None)
            crossing = next((n for n in module.graph.nodes if _bound(n)), None)
            if host_call is not None:
                targets = map_arg(outputs, storage)
                actual = map_arg(host_call.args, live.__getitem__)
                keywords = map_arg(host_call.kwargs, live.__getitem__)
                prepared = host.prepare(*actual, **keywords, outputs=tree_flatten(targets)[0])
                if prepared is not None:
                    self.steps.append(prepared)
                values.update(zip(exported, targets))
                continue
            if crossing is None:
                from torch.utils._pytree import tree_unflatten
                output_spec = tree_flatten(tree_map(_value, outputs))[1]
                destinations = [placements.get(value) for value in tree_flatten(tree_map(_value, outputs))[0]]
                made = section['retained'].bind(tree_flatten(args)[0], destinations=destinations)
                targets = tree_unflatten(tree_flatten(made.output)[0], output_spec)
                self.steps.append(made)
                declarations = lower(module)
            else:
                targets = map_arg(outputs, storage)
                target, metadata = operation(crossing)
                if crossing.target is not target:
                    raise NotImplementedError('bound collectives require functional graph operands')
                actual = map_arg(crossing.args, live.__getitem__)
                keywords = map_arg(crossing.kwargs, live.__getitem__)
                named = {a.name: actual[i] if i < len(actual) else keywords.get(a.name, a.default_value)
                         for i, a in enumerate(target._schema.arguments)}
                source = named.get('input', named.get('x', named.get('self', named.get('partials'))))
                native = dist.distributed_c10d._resolve_process_group(metadata['group_name'])
                declarations = lower(module)
                if target is torch.ops.torch_mesh.select_exchange.default:
                    prepared, calls = _bind_exchange(named['x'], tree_flatten(targets)[0][0],
                                              _exchange_plan(crossing, metadata), native)
                    if prepared is not None:
                        self.steps.append(prepared)
                    if calls:
                        self.groups.setdefault(metadata['group_name'], []).append(calls)
                    values.update(zip(exported, targets))
                    continue
                if target is torch.ops.torch_mesh.select_reduce.default:
                    prepared, calls = _bind_reduce(named['x'], tree_flatten(targets)[0][0],
                                                   _reduce_plan(crossing, metadata)[0], native, metadata['reduce_op'])
                    if prepared is not None:
                        self.steps.append(prepared)
                    if calls:
                        self.groups.setdefault(metadata['group_name'], []).append(calls)
                    values.update(zip(exported, targets))
                    continue
                gathers = _gathers(target, named, targets, native)
                if gathers is not None:
                    prepared, calls = backend.bind_gather_views(gathers, native)
                    if prepared is not None:
                        self.steps.append(prepared)
                    if calls:
                        self.groups.setdefault(metadata['group_name'], []).append(calls)
                    values.update(zip(exported, targets))
                    continue
                if target is torch.ops.torch_mesh.reduce_scatterv.default:
                    destination = tree_flatten(targets)[0][0]
                    made, calls = backend.bind_scatter_views(source, destination, named['dim'], named['sizes'],
                                                             _operation(named['op']), native)
                    self.steps.append(made)
                    self.groups.setdefault(metadata['group_name'], []).append(calls)
                    values.update(zip(exported, targets))
                    continue
                calls = declarations[metadata['group_name']]
                if len(calls) != 1:
                    raise ValueError('a bound crossing declares one collective group')
                outputs_flat = tree_flatten(targets)[0]
                sources = named.get('inputs', [source])
                if len(outputs_flat) != len(calls[0]) or len(sources) != len(calls[0]):
                    raise ValueError('each declared collective needs its source and final output')
                made, lowered = backend.prepare_fragments(list(zip(calls[0], sources, outputs_flat)), native)
                self.steps.append(made)
                declarations[metadata['group_name']] = [lowered]
            for name, calls in declarations.items():
                self.groups.setdefault(name, []).extend(calls)
            values.update(zip(exported, targets))
        output = next(node for node in graph.graph.nodes if node.op == 'output')
        self.output = map_arg(output.args[0], values.__getitem__)
        self.execution = execution_sequence(self.steps, self.output, placements.owners,
                                           graph.meta.get('mesh_initializations', ()))

    def _bind_owned(self, ownership, destinations):
        """Bind explicit work and delivery assignments to the existing recorder."""
        from torch.utils import _pytree as tree
        from torch.fx.node import map_arg
        from . import backend
        from .ownership import _source_leaves
        from .operand import TensorView, layout_configuration, relation_view
        from .placement import selected, selected_view
        from .collective_views import selection_exchanges
        from .native import empty_strided

        prepared, mesh = self.prepared, ownership['mesh']
        algebra, coordinate = prepared['algebra'], tuple(mesh.get_coordinate())
        if ownership['pending']:
            requirements = tuple(dict.fromkeys(item.get('requirement', 'access relation') for item in ownership['pending']))
            raise ValueError('function preparation needs declared ' + ', '.join(requirements))
        available, owners = {}, []

        def original(key):
            node, path = key
            while node in prepared['aliases']:
                node = prepared['aliases'][node]
            return node, path

        def described(key):
            return tree.key_get(key[0].meta['val'], key[1])

        def selection(key, positions):
            value = described(key)
            return dict(shape=list(value.shape), steps=[['reshape', [value.numel()]], ['select', 0, positions.spans()]])

        def view(held, wanted, shape):
            source = algebra.selection_relation(held['selection'])
            destination = algebra.selection_relation(wanted)
            access = destination.then(source.inverse((math.prod(wanted['shape']),))).lexmin()
            return relation_view(TensorView.of(held['operand']), access, shape)

        placeholders = [node for node in ownership['graph'].graph.nodes if node.op == 'placeholder']
        for node, argument in zip(placeholders, ownership['arguments'], strict=True):
            for key, value in _source_leaves((node,)):
                if isinstance(value, torch.Tensor):
                    owner = ownership['values'][key]['owners'][coordinate]
                    available[original(key)] = dict(selection=owner['selection'], operand=tree.key_get(argument, key[1]))

        field = lambda value: isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], torch.fx.Node)
        results = map_arg(ownership['result'], lambda node: tree.tree_map_with_path(
            lambda path, value: (node, tuple(path)) if isinstance(value, torch.Tensor) else value, node.meta['val']))
        final_targets = {}
        output_uses = tree.tree_leaves(results, is_leaf=field)
        targets = [None] * len(output_uses) if destinations is None else list(destinations)
        for key, destination in zip(output_uses, targets, strict=True):
            if field(key) and destination is not None:
                final_targets.setdefault(original(key), []).append((key, destination))

        def bind_section(section, projection, input_bindings, output_bindings, *, equalities=()):
            delivered = [record for record in ownership['deliveries'] if record['section'] is section]
            fixed, requirements = [], []
            wanted = projection.get('outputs', {})
            requested = []
            empty = algebra.set('{ [i] : false }', 1)
            for key, binding in output_bindings.items():
                order = selection(key, wanted.get(key, empty))
                requested.append(dict(binding=binding, selection=order,
                                      shape=algebra.selection_relation(order).shape))
                key = original(key)
                if key in final_targets:
                    requested_key, destination = final_targets[key][0]
                    owner = ownership['requests'][coordinate][requested_key]
                    held = available.get(key)
                    relocatable = binding['name'] not in {value['name'] for value in input_bindings.values()}
                    if held is not None:
                        from .placement import same_view
                        overlap = algebra.selection(held['selection']).binary(algebra.selection(owner['selection']), 'intersect')
                        common = selection(key, overlap)
                        shape = algebra.selection_relation(common).shape
                        relocatable = overlap.empty() or same_view(view(held, common, shape),
                            view(dict(selection=owner['selection'], operand=destination), common, shape))
                    if relocatable:
                        fixed.append(dict(binding=binding, selection=owner['selection'], operand=destination))
            for key, binding in input_bindings.items():
                held = available.get(original(key))
                if held is not None:
                    fixed.append(dict(binding=binding, **held))
            for delivery in delivered:
                key = delivery['source']
                binding = input_bindings[key]
                # Relay pages and numerical inputs share the same original
                # value coordinates. Local edges therefore name the same pages.
                selections = [delivery['sources'][coordinate], delivery['destinations'][coordinate]]
                selections.extend(step[side][coordinate] for step in delivery['steps'] for side in ('sources', 'destinations'))
                for at, order in enumerate(selections):
                    relation = algebra.selection_relation(order).then(binding['relation'])
                    requirements.append(dict(key=('delivery', key, at), binding=binding, relation=relation,
                        shape=relation.shape, requirements=(), access='readwrite'))
            retained = None if section is None else section.get('retained')
            if retained is not None:
                retained.select_domains(projection['records'])
                requirements.extend(retained.consumer_uses())
            types = [described(key) for key in (*input_bindings, *output_bindings)]
            device = types[0].device.type
            plan = selected(projection, algebra=algebra, device=device, results=requested,
                            fixed=fixed, requirements=requirements, equalities=equalities)
            owners.extend(plan['pages'])

            def placed(binding, order):
                return selected_view(plan['storage'], dict(binding=binding, selection=order), algebra=algebra)

            for delivery in delivered:
                binding = input_bindings[delivery['source']]
                for step in delivery['steps']:
                    peers, group = step['peers'], step['group']
                    transfers = selection_exchanges([step['sources'][peer] for peer in peers],
                                                    [step['destinations'][peer] for peer in peers], algebra=algebra)
                    source = placed(binding, step['sources'][coordinate])
                    destination = placed(binding, step['destinations'][coordinate])
                    command, calls = _bind_exchange(source, destination, transfers, group)
                    if command is not None:
                        self.steps.append(command)
                        self.groups.setdefault(group.group_name, []).append(calls)
            for (key, binding), operand, request in zip(output_bindings.items(), plan['results'], requested, strict=True):
                available[original(key)] = dict(selection=request['selection'], operand=operand)
            return plan

        local = ownership['projections'][coordinate]
        for section, projection in zip(prepared['sections'], local['sections'], strict=True):
            if not section['boundaries']:
                input_actuals = {binding['name']: available[original(key)]['operand']
                                for key, binding in section['input_bindings'].items() if original(key) in available}
                plan = bind_section(section, projection, section['input_bindings'], section['output_bindings'])
                actuals = [plan['storage'][name]['operand'] if name in plan['storage'] else
                           input_actuals.get(name, value)
                           for name, value in zip(section['retained'].graph_inputs, section['operands'], strict=True)]
                self.steps.append(section['retained'].bind(actuals, **plan))
            else:
                boundary = _boundary_work(section, projection, algebra)
                projected = dict(boundary['projection'], outputs=projection['outputs'])
                plan = bind_section(section, projected, boundary['inputs'], boundary['outputs'],
                                    equalities=boundary['equalities'])
                for command in boundary['commands']:
                    def operand(role):
                        use = command[role]
                        return None if use is None else selected_view(plan['storage'], use, algebra=algebra)
                    source, destination = operand('reads'), operand('writes')
                    if source is None and destination is None:
                        value = next(iter(boundary['inputs']), next(iter(boundary['outputs']), None))
                        metadata = described(value)
                        source = empty_strided((0,), (1,), metadata.dtype, metadata.device.type)
                    command_, calls = backend.prepare_fragments([(command['call'], source, destination)], command['group'])
                    self.steps.append(command_)
                    self.groups.setdefault(command['group_name'], []).append(calls)

        final = [record for record in ownership['deliveries'] if record['section'] is None]
        if final:
            bindings, needs = {}, {}
            for record in final:
                key = record['source']
                value = described(key)
                bindings[key] = dict(name=value.untyped_storage()._cdata, dtype=value.dtype,
                    storage_bytes=value.untyped_storage().nbytes(),
                    relation=algebra.layout(layout_configuration(TensorView.of(value).layout)))
                needs[key] = algebra.selection(record['destinations'][coordinate])
            bind_section(None, dict(storage_bytes={}, records=(), outputs=needs), bindings, bindings)

        def result(key):
            if not field(key):
                return key
            owner = ownership['requests'][coordinate][key]
            return view(available[original(key)], owner['selection'], owner['shape'])
        from . import _stream
        from .layout import copies
        from .native import Program
        values, spec = tree.tree_flatten(tree.tree_map(result, results, is_leaf=field))
        copied = []
        for index, (value, destination) in enumerate(zip(values, targets, strict=True)):
            if destination is not None:
                copied.extend(copies(value, destination))
                values[index] = destination
        self.output = tree.tree_unflatten(values, spec)
        if copied:
            self.steps.append(Program(_stream.Commands(copied), self.output))
        self.execution = execution_sequence(self.steps, self.output, owners)

class Function:
    """One numerical callable and its framework-derived AOT declaration.

    Tensor arguments are fake inputs during tracing and are bound once to the
    realized callable. ``declare`` retraces after placement changes. Compilation operates on the
    complete explicit forward/backward graph, with parameter gradients retained
    as outputs; a compiler cannot discard the backward as unused work. No payload
    function is executed to discover communication. Execution submits the
    bound native commands; compilation failures propagate to the caller.

    ``ownership`` supplies a mesh and explicit source-value owner selections.
    Its source graph is captured before numerical decomposition, then rewritten
    into the same page operands and commands as the other declarations.
    """
    def __init__(self, function, *arguments, parameters=(), configuration=None, ownership=None):
        self.function, self.parameters = function, tuple(parameters)
        self.arguments = arguments
        self.configuration = configuration
        self.ownership = ownership
        self.source_graph = None
        self.graph, self.groups, self.run, self.output_spec = None, {}, None, None
        self.execution = None

    def declare(self, group, *, destinations=None):
        from functools import partial
        from torch.utils._pytree import tree_flatten
        from torch.distributed.tensor import DTensor
        from .compilation import bind

        def recorded(*arguments):
            output = self.function(*arguments)
            values, self.output_spec = tree_flatten(output)
            gradients = [None if p.grad is None else p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                         for p in self.parameters]
            return (*values, *gradients)

        with bind(self.configuration):
            graph = trace(recorded, *self.arguments, parameters=self.parameters)
        return self.declare_graph(graph, group, destinations=destinations)

    def declare_graph(self, graph, group, *, destinations=None):
        """Lower an existing typed graph, retaining its source identities."""
        from torch.distributed.tensor import DTensor
        from torch.utils._pytree import tree_leaves
        from .compilation import lift_attributes, bind
        graph, arguments = lift_attributes(graph, self.arguments)
        self.source_graph = graph
        ownership = self.ownership
        if ownership is None:
            scopes = list(graph.meta.get('mesh_scopes', ()))
            for value in tree_leaves(arguments):
                if isinstance(value, DTensor) and value.device_mesh not in scopes:
                    scopes.append(value.device_mesh)
            if len(scopes) > 1:
                raise ValueError('source values use multiple meshes; supply the enclosing ownership mesh and domains')
            if scopes:
                ownership = {'mesh': scopes[0], 'values': {}}
        assigned = None
        if ownership is not None:
            from .partition import lower_owned_graph
            with bind(self.configuration):
                assigned = lower_owned_graph(graph, arguments, ownership)
            graph, arguments = assigned['graph'], assigned['arguments']
        with bind(self.configuration):
            prepared = BoundGraph(graph, arguments, group, self.configuration, destinations, ownership=assigned)
        self.graph, self.groups, self.execution = graph, prepared.groups, prepared.execution
        self.run = self.execution.forward
        self.completion = self.execution.completion
        return self.groups.get(group.group_name, []) if group is not None else []

    def append(self, schedule):
        self.execution.append(schedule)

    def prepare(self, repetitions=1):
        """Encode declared invocations before execution."""
        self.execution.realize(repetitions)

    def __call__(self):
        if self.run is None:
            raise RuntimeError('realize the function declaration before execution')
        from torch.utils._pytree import tree_unflatten
        from torch.distributed.tensor import DTensor
        from . import partition
        with torch.no_grad():
            values = self.run()
        output_count = self.output_spec.num_leaves
        for parameter, gradient in zip(self.parameters, values[output_count:]):
            parameter.grad = (partition.from_local(gradient, parameter.device_mesh, parameter.placements,
                                                   shape=parameter.shape, stride=parameter.stride())
                              if gradient is not None and isinstance(parameter, DTensor) else gradient)
        return tree_unflatten(values[:output_count], self.output_spec)
