"""AOT collective declarations from the framework's complete function graph.

The frontend records operators, shapes and groups. The common native compiler
chooses reduction trees, physical paths and packet cuts; this module contains
no network topology or collective algorithm of its own.
"""
import math

import torch
import torch.distributed as dist


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
    if isinstance(argument, torch.fx.Node):
        if 'val' not in argument.meta:
            raise ValueError(f'collective operand {argument.name} has no AOT tensor metadata')
        return argument.meta['val']
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


def lower(graph, *, forms=None):
    """Return group-name -> ordered common Calls from an actual FX graph.

    Unknown communication operators fail during configuration. Numerical ops,
    views and dependency waits do not create additional transport declarations.
    """
    from . import backend, partition
    groups, trains = {}, {}
    for node in (graph.graph if hasattr(graph, 'graph') else graph).nodes:
        target, args = operation(node)
        schema = getattr(target, '_schema', None)
        if schema is None:
            continue
        namespace, name = schema.name.split('::', 1)
        if namespace not in ('_c10d_functional', '_c10d_functional_autograd', 'torch_mesh'):
            continue
        if name == 'wait_tensor':
            continue
        group_name = args.get('group_name')
        if group_name is None:
            if namespace == 'torch_mesh':
                continue  # native numerical kernels such as linear
            raise ValueError(f'communication operator {schema.name} has no declared group')
        group = dist.distributed_c10d._resolve_process_group(group_name)
        n, calls = group.size(), []
        x = args.get('input', args.get('x', args.get('self', args.get('partials'))))
        reduction = args.get('reduce_op', args.get('op', 'sum'))
        if namespace == 'torch_mesh' and name in ('allgatherv', 'allgatherv_out'):
            calls = partition.gathered(x, args['dim'], args['sizes'], group)
        elif namespace == 'torch_mesh' and name in ('all_reduce', 'linear_sum', 'fused_step', 'fused_step_out'):
            if name == 'all_reduce':
                calls = [backend.Call(backend.ALLREDUCE, x.numel(), x.dtype, op=_operation(reduction))]
            elif name == 'linear_sum' or args.get('adds'):
                output = _value(node) if name != 'fused_step_out' else args['out']
                calls = partition._gather_writes([output.numel()] * n, output.dtype) if n > 1 else []
            else:
                calls = partition.gathered(x, args['dim'], args['sizes'], group)
        elif namespace == 'torch_mesh' and name == 'reduce_scatterv':
            counts = [int(size) * partition._row(x, args['dim']) for size in args['sizes']]
            calls = [backend.Call(backend.REDUCE_SCATTER, x.numel(), x.dtype, counts=counts, op=_operation(reduction))]
        elif namespace == 'torch_mesh' and name == 'merge_partials':
            operator = group.partials_operator(x.shape[-1])
            root = args['root']
            rows = args.get('rows', [])
            calls = [backend.Call(backend.REDUCE_SCATTER if rows else backend.ALLREDUCE if root < 0 else backend.REDUCE,
                                  x.numel(), x.dtype, root=max(0, root), op=operator,
                                  element_shape=(x.shape[-1],),
                                  counts=[r * x.shape[-1] for r in rows] if rows else None)]
        elif name in ('all_reduce', 'all_reduce_'):
            calls = [backend.Call(backend.ALLREDUCE, x.numel(), x.dtype, op=_operation(reduction))]
        elif name in ('broadcast', 'broadcast_'):
            calls = [backend.Call(backend.BROADCAST, x.numel(), x.dtype, root=args['src'])]
        elif name in ('all_gather_into_tensor', 'all_gather_into_tensor_out'):
            calls = partition.gathered(x, 0, [x.shape[0]] * n, group)
        elif name == 'reduce_scatter_tensor':
            if x.numel() % n:
                raise ValueError('reduce-scatter metadata is not divisible by the group size')
            calls = [backend.Call(backend.REDUCE_SCATTER, x.numel(), x.dtype,
                                  counts=[x.numel() // n] * n, op=_operation(reduction))]
        elif name == 'all_to_all_single':
            width = math.prod(x.shape[1:])
            incoming = args['output_split_sizes'] or [x.shape[0] // n] * n
            outgoing = args['input_split_sizes'] or [x.shape[0] // n] * n
            calls = [backend.Call(backend.ALLTOALL, x.numel(), x.dtype,
                                  counts=[int(rows) * width for rows in (*outgoing, *incoming)])]
        elif name in ('all_reduce_coalesced', 'all_gather_into_tensor_coalesced', 'reduce_scatter_tensor_coalesced'):
            inputs = [_value(tensor) for tensor in args['inputs']]
            for tensor in inputs:
                if name == 'all_reduce_coalesced':
                    calls.append(backend.Call(backend.ALLREDUCE, tensor.numel(), tensor.dtype, op=_operation(reduction)))
                elif name == 'all_gather_into_tensor_coalesced':
                    calls.extend(partition.gathered(tensor, 0, [tensor.shape[0]] * n, group))
                else:
                    calls.append(backend.Call(backend.REDUCE_SCATTER, tensor.numel(), tensor.dtype,
                                               counts=[tensor.numel() // n] * n, op=_operation(reduction)))
        else:
            raise NotImplementedError(f'AOT collective declaration for {schema.name}')
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


def trace(function, *arguments, parameters=()):
    """Trace one complete function, including its backward, using FakeTensor.

    Parameters' real gradient references are saved and restored; fake backward
    neither accumulates into an existing real gradient nor becomes model state.
    The caller supplies a static function. Data-dependent Python branches are
    configuration errors, never an excuse to run a hidden measurement or probe.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    parameters = tuple(parameters)
    gradients = [parameter.grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.grad = None
        graph = make_fx(function, tracing_mode='fake', _allow_non_fake_inputs=True)(*arguments)
    finally:
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
    # Framework layout queries can leave unused tensor operations in make_fx's
    # graph (including a mesh-coordinate nonzero). They are configuration work,
    # not part of the declared function or its explicit gradient outputs.
    spans(graph.graph)
    graph.recompile()
    return graph, lower(graph)


class Execution:
    """Configuration which appends every numerical span and crossing to one native function."""
    def __init__(self, steps, output=None):
        from . import _stream
        self.steps, self.output = tuple(steps), output
        self.executions = _stream.Executions(output)
        self.forward = self.executions.run
        self.completion = self.executions.completion

    def append(self, schedule):
        for step in self.steps:
            step.append(schedule)

    def realize(self, repetitions=1):
        from . import _stream
        for _ in range(repetitions):
            schedule = _stream.Schedule()
            self.append(schedule)
            schedule.finish()
            self.executions.add(schedule)


def execution_sequence(steps, output=None):
    return Execution(steps, output)


_BOUND = {'all_to_all_single', 'merge_partials', 'reduce_scatterv', 'allgatherv', 'allgatherv_out',
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


def sections(module, cuts):
    """Materialize the common graph partition as FX fragments, retaining metadata."""
    from mesh_graph import partition
    nodes = list(module.graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    output = next(node for node in nodes if node.op == 'output')
    constants = [node for node in nodes if node.op == 'get_attr']
    pieces = partition([node.all_input_nodes for node in nodes], [(node,) for node in nodes],
                       output.all_input_nodes, cuts,
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
        yield torch.fx.GraphModule(module, graph), tuple(inputs), tuple(outputs)


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


class BoundGraph:
    """Configuration-time binding of ordinary collective graph boundaries.

    The producer span writes the source region; the native collective binds
    that region and its final output. There is no eager transport call in the
    generated execution sequence. Numerical spans use the existing compiler.
    """
    def __init__(self, graph, arguments, group, configuration, destinations=None):
        from .compilation import lift_attributes
        graph, arguments = lift_attributes(graph, arguments)
        from torch.fx.node import map_arg
        from torch.utils._pytree import tree_map, tree_flatten
        from . import backend, host
        from .native import empty_strided
        from .placement import Placements, alias, same_view
        # Waiting on a functional collective describes its result edge. The
        # prepared native binding already supplies that stream dependency.
        attributes = {}
        for node in list(graph.graph.nodes):
            if node.op == 'get_attr':
                if node.target in attributes:
                    node.replace_all_uses_with(attributes[node.target]);graph.graph.erase_node(node)
                    continue
                attributes[node.target] = node
            target, _ = operation(node)
            if getattr(getattr(target, '_schema', None), 'name', '') == '_c10d_functional::wait_tensor':
                node.replace_all_uses_with(node.args[0]);graph.graph.erase_node(node)
        cuts = [at for i, node in enumerate(graph.graph.nodes) if _bound(node) or host.bound(node) for at in (i, i + 1)]
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
        placements, inputs_placed = Placements(), Placements()
        for node, value in values.items():
            for described, actual in zip(tree_flatten(_value(node))[0], tree_flatten(value)[0], strict=True):
                if isinstance(described, torch.Tensor):
                    inputs_placed.bindings.append((described, actual))
        result = next(n.args[0] for n in graph.graph.nodes if n.op == 'output')
        if destinations is not None:
            for described, destination in zip(tree_flatten(map_arg(result, _value))[0], destinations, strict=True):
                if destination is not None:
                    placements.bind(described, destination)
        from .layout import place
        # Solve from recipients back to producers before compiling any span.
        # A gather which feeds another gather inherits that gather's local view.
        for node in reversed(list(graph.graph.nodes)):
            if not _bound(node):
                continue
            target, named = operation(node)
            native = dist.distributed_c10d._resolve_process_group(named['group_name'])
            for source, described, dim, sizes in _gathers(target, named, _value(node), native) or ():
                destination = placements.get(described)
                if destination is None:
                    destination = inputs_placed.get(described)
                if destination is None:
                    destination = empty_strided(tuple(described.shape), tuple(described.stride()),
                                                described.dtype, described.device.type)
                placements.bind(described, destination)
                own = destination.narrow(dim, sum(sizes[:native.rank()]), sizes[native.rank()])
                supplied = inputs_placed.get(source)
                if supplied is not None and not same_view(supplied, own):
                    raise ValueError('a gather input supplied by the caller must already occupy its final local slice')
                placements.bind(source, own)
        # Collective destinations are storage assignments. Layout equations
        # consume those assignments before placing still-unbound producers.
        place(graph.graph, placements, [logical for logical, _ in inputs_placed.bindings])
        for module, inputs, exported in sections(graph, cuts):
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
                self.steps.append(host.prepare(*actual, **keywords, outputs=tree_flatten(targets)[0]))
                values.update(zip(exported, targets))
                continue
            if crossing is None:
                # Keep the graph and its one FakeTensor shape environment.
                # Retracing a split graph creates a second set of symbols
                # for the same internal values and loses their bindings.
                from torch.utils._pytree import tree_unflatten
                made = Function(module, *args, configuration=configuration)
                made.output_spec = tree_flatten(tree_map(_value, outputs))[1]
                destinations = [placements.get(value) for value in tree_flatten(tree_map(_value, outputs))[0]]
                made.declare_graph(module, group, destinations=destinations)
                targets = tree_unflatten(tree_flatten(made.execution.output)[0], made.output_spec)
                self.steps.append(made)
                declarations = made.groups
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
                gathers = _gathers(target, named, targets, native)
                if gathers is not None:
                    prepared = backend.bind_gather_views(gathers, native)
                    if prepared is not None:
                        self.steps.append(prepared)
                    for name, calls in declarations.items():
                        self.groups.setdefault(name, []).extend(calls)
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
        self.execution = execution_sequence(self.steps, self.output)

class Function:
    """One numerical callable and its framework-derived AOT declaration.

    Tensor arguments are fake inputs during tracing and are bound once to the
    realized callable. ``declare`` retraces after placement changes. Compilation operates on the
    complete explicit forward/backward graph, with parameter gradients retained
    as outputs; a compiler cannot discard the backward as unused work. No payload
    function is executed to discover communication. Execution submits the
    bound native commands; compilation failures propagate to the caller.
    """
    def __init__(self, function, *arguments, parameters=(), configuration=None):
        self.function, self.parameters = function, tuple(parameters)
        self.arguments = arguments
        self.configuration = configuration
        self.graph, self.groups, self.run, self.output_spec = None, {}, None, None
        self.execution = None

    def declare(self, group, *, destinations=None):
        from functools import partial
        from torch.utils._pytree import tree_flatten
        from torch.distributed.tensor import DTensor

        def recorded(*arguments):
            output = self.function(*arguments)
            values, self.output_spec = tree_flatten(output)
            gradients = [None if p.grad is None else p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                         for p in self.parameters]
            return (*values, *gradients)

        graph, declarations = trace(recorded, *self.arguments, parameters=self.parameters)
        return self.declare_graph(graph, group, declarations, destinations=destinations)

    def declare_graph(self, graph, group, declarations=None, *, destinations=None):
        """Compile an existing complete graph without retracing its operators."""
        from functools import partial
        declarations = lower(graph) if declarations is None else declarations
        from . import host
        if any(_bound(node) or host.bound(node) for node in graph.graph.nodes):
            from .compilation import bind
            with bind(self.configuration):
                prepared = BoundGraph(graph, self.arguments, group, self.configuration, destinations)
            self.graph, self.groups, self.execution = graph, prepared.groups, prepared.execution
        else:
            from . import compiler
            with torch.no_grad():
                program = compiler(graph, list(self.arguments), configuration=self.configuration, destinations=destinations)
            self.graph, self.groups = graph, declarations
            self.execution = execution_sequence([program], program.output)
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
