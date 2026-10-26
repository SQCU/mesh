"""AOT collective declarations from the framework's complete function graph.

The frontend records operators, shapes and groups. The common native compiler
chooses reduction trees, physical paths and packet cuts; this module contains
no network topology or collective algorithm of its own.
"""
import math
from contextvars import ContextVar

import torch
import torch.distributed as dist

_CAPTURE = ContextVar('mesh_compiled_declarations', default=None)


_INDEXED_SHAPES = {}


def tensor_metadata(x):
    return None if x is None else (tuple(x.shape), tuple(x.stride()), x.dtype)


def crossing_key(form):
    x, w, dim, sizes, name, adds, repeats = form[:7]
    return tensor_metadata(x), tensor_metadata(w), dim, tuple(sizes), name, adds, repeats


def crossing_slot(form, destination):
    from . import _stream
    index = _stream.crossing_slot()
    x, w, dim, sizes, _, adds, _ = form[:7]
    operand = tuple(x.shape) if w is None else (x.shape[0], w.shape[0])
    dim %= len(operand)
    _INDEXED_SHAPES[index] = operand if adds else operand[:dim] + (sum(sizes),) + operand[dim + 1:]
    return index, destination


def compiled_graph(graph):
    """Lower configuration, then replace each crossing by its prepared slot."""
    forms = []
    declarations = lower(graph, forms=forms)
    slots = {}
    fx = graph.graph if hasattr(graph, 'graph') else graph
    for node in fx.nodes:
        target, args = operation(node)
        if target not in (torch.ops.torch_mesh.fused_step.default, torch.ops.torch_mesh.fused_step_out.default):
            continue
        into = target is torch.ops.torch_mesh.fused_step_out.default
        form = (args['x'], args['w'], args['dim'], tuple(args['sizes']), args['group_name'], args['adds'], args['repeats'])
        index, destination = crossing_slot(form, into)
        slots.setdefault(crossing_key(form), []).append((index, destination))
        replacement = torch.ops.torch_mesh.indexed_step_out.default if into else torch.ops.torch_mesh.indexed_step.default
        if node.target is target:
            values = {a.name: node.args[i] if i < len(node.args) else node.kwargs.get(a.name, a.default_value)
                      for i, a in enumerate(target._schema.arguments)}
            node.target = replacement
            node.args = (values['x'], values['w'], *((values['out'],) if into else ()), index)
            node.kwargs = {}
        else:
            node.args = (replacement, *node.args[1:])
            node.kwargs = {**{key: value for key, value in node.kwargs.items()
                              if key not in ('dim', 'sizes', 'group_name', 'adds', 'repeats')},
                           'index': index}
        node.meta.pop('eager_input_vals', None)
    ordered = []
    for form in forms:
        key = crossing_key(form)
        if key in slots:
            count = form[6]
            ordered.append((*form, tuple(slots[key][:count])))
            del slots[key][:count]
        else:
            ordered.append(form)
    forms = ordered
    capture = _CAPTURE.get()
    if capture is not None:
        capture.append((declarations, forms))
    return forms


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
                if reduction == 'sum' and x.device.type == 'mps' and x.dtype in (torch.float16, torch.float32):
                    calls = partition.summed(x, group)
                else:
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
                                  counts=[r * x.shape[-1] for r in rows] if rows else None)]
        elif name in ('all_reduce', 'all_reduce_'):
            calls = [backend.Call(backend.ALLREDUCE, x.numel(), x.dtype, op=_operation(reduction))]
        elif name in ('broadcast', 'broadcast_'):
            calls = [backend.Call(backend.BROADCAST, x.numel(), x.dtype, root=args['src'])]
        elif name in ('all_gather_into_tensor', 'all_gather_into_tensor_out'):
            calls = [backend.Call(backend.ALLGATHER, x.numel() * n, x.dtype, counts=[x.numel()] * n)]
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
                    calls.append(backend.Call(backend.ALLGATHER, tensor.numel() * n, tensor.dtype, counts=[tensor.numel()] * n))
                else:
                    calls.append(backend.Call(backend.REDUCE_SCATTER, tensor.numel(), tensor.dtype,
                                               counts=[tensor.numel() // n] * n, op=_operation(reduction)))
        else:
            raise NotImplementedError(f'AOT collective declaration for {schema.name}')
        if calls:
            repeats = int(args.get('repeats', 1)) if all(call.kind == backend.WRITE for call in calls) else 1
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
            if forms is not None and all(call.kind == backend.WRITE for call in calls):
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
    return graph, lower(graph)


class Preparations:
    """Page bindings and dispatch operands realized before a complete function."""
    def __init__(self, forms):
        from collections import Counter
        batches = Counter(crossing_key(form) for form in forms)
        prototype = lambda m: None if m is None else torch.empty_strided(m[0], m[1], dtype=m[2], device='meta')
        self.forms, self.bindings, self.order = [], {}, []
        for key, count in batches.items():
            x, w, dim, sizes, name, adds, repeats = key
            group = dist.distributed_c10d._resolve_process_group(name)
            if len(sizes) > 1 and getattr(group, 'handle', 0):
                form = (prototype(x), prototype(w), dim, list(sizes), name, adds, repeats)
                self.forms.append((*form, count))
        for source in forms:
            key = crossing_key(source)
            x, w, dim, sizes, name, adds, repeats = key
            if len(sizes) > 1 and getattr(dist.distributed_c10d._resolve_process_group(name), 'handle', 0) and len(source) != 8:
                raise ValueError('crossings must have compiler-assigned indices before preparation')
            for index, destination in source[7] if len(source) == 8 else ():
                self.bindings[index] = (prototype(x), prototype(w), dim, list(sizes), name, adds, repeats, destination)
                self.order.append(index)

    def bind(self, repetitions=1):
        from . import _stream
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily():
            for form in self.forms:
                _stream.fused_prepare(*form[:-1], form[-1] * repetitions)
            for index, operands in self.bindings.items():
                _stream.prepare_crossing(index, *operands)

    def __call__(self, repetitions=1):
        from . import _stream
        self.bind(repetitions)
        for _ in range(repetitions):
            for index in self.order:
                _stream.compile_crossing(index)


def execution_sequence(steps, output=None):
    """One fixed FX call sequence for partial blocks and whole-function spans."""
    from torch.utils._pytree import tree_map
    graph, bindings = torch.fx.Graph(), {}
    for value, method in steps:
        name = f'step_{len(bindings)}'
        bindings[name] = value
        graph.call_method(method, (graph.get_attr(name),))
    def result(value):
        if not isinstance(value, torch.Tensor):
            return value
        name = f'result_{len(bindings)}'
        bindings[name] = value
        return graph.get_attr(name)
    graph.output(tree_map(result, output))
    return torch.fx.GraphModule(bindings, graph, 'PreparedFunction')


_BOUND = {'all_to_all_single', 'merge_partials', 'reduce_scatterv',
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
        x = args.get('input', args.get('x', args.get('self')))
        reduction = args.get('reduce_op', args.get('op', 'sum'))
        return not (reduction == 'sum' and x.device.type == 'mps' and x.dtype in (torch.float16, torch.float32))
    return name in _BOUND


class BoundGraph:
    """Configuration-time binding of ordinary collective graph boundaries.

    The producer span writes the source region; the native collective binds
    that region and its final output. There is no eager transport call in the
    generated execution sequence. Numerical spans use the existing compiler.
    """
    def __init__(self, graph, arguments, group):
        import operator
        from torch.fx.passes.split_module import split_module
        from torch.fx.node import map_arg
        from torch.utils._pytree import tree_map, tree_flatten
        from . import backend, empty
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
        parts, phase = {}, 0
        for node in graph.graph.nodes:
            if _bound(node):
                phase += 1;parts[node] = phase;phase += 1
            else:
                parts[node] = phase
        divided = split_module(graph, graph, lambda node: parts[node], keep_original_order=True)
        values, arguments = {}, iter(arguments)
        self.groups, self.forms, self.steps = {}, [], []
        self.order = []
        for node in divided.graph.nodes:
            if node.op == 'placeholder':
                values[node] = next(arguments)
            elif node.op == 'get_attr':
                value = divided
                for name in node.target.split('.'):
                    value = getattr(value, name)
                values[node] = value
            elif node.op == 'call_function' and node.target is operator.getitem:
                args = map_arg(node.args, values.__getitem__)
                values[node] = operator.getitem(*args)
            elif node.op == 'call_module':
                module = divided.get_submodule(node.target)
                args = map_arg(node.args, values.__getitem__)
                live = {n: v for n, v in zip((n for n in module.graph.nodes if n.op == 'placeholder'), args)}
                for n in module.graph.nodes:
                    if n.op == 'get_attr':
                        value = module
                        for name in n.target.split('.'):
                            value = getattr(value, name)
                        live[n] = value
                outputs = next(n.args[0] for n in module.graph.nodes if n.op == 'output')
                def storage(n):
                    metadata = _value(n)
                    if not isinstance(metadata, torch.Tensor):
                        return metadata
                    for dependency, value in live.items():
                        described = _value(dependency)
                        if isinstance(described, torch.Tensor) and torch._C._is_alias_of(metadata, described):
                            return value.as_strided(tuple(metadata.shape), tuple(metadata.stride()),
                                value.storage_offset() + metadata.storage_offset() - described.storage_offset())
                    return empty(tuple(metadata.shape), dtype=metadata.dtype, device=metadata.device.type)
                targets = map_arg(outputs, storage)
                crossing = next((n for n in module.graph.nodes if _bound(n)), None)
                if crossing is None:
                    # Keep the graph and its one FakeTensor shape environment.
                    # Retracing a split graph creates a second set of symbols
                    # for the same internal values and loses their bindings.
                    output_node = next(n for n in module.graph.nodes if n.op == 'output')
                    leaves, specification = tree_flatten(outputs)
                    destinations = tree_flatten(targets)[0]
                    bound = []
                    with module.graph.inserting_before(output_node):
                        for k, (result, destination) in enumerate(zip(leaves, destinations)):
                            if not isinstance(destination, torch.Tensor):
                                bound.append(result)
                                continue
                            name = f'_mesh_output_{k}'
                            module.register_buffer(name, destination)
                            into = module.graph.get_attr(name)
                            bound.append(module.graph.call_function(torch.ops.aten.copy_.default, (into, result)))
                    from torch.utils._pytree import tree_unflatten
                    output_node.args = (tree_unflatten(bound, specification),)
                    module.recompile()
                    made = Function(module, *args)
                    made.output_spec = tree_flatten(targets)[1]
                    made.declare_graph(module, group)
                    self.steps.append((made, 'run'));self.order.append(made.preparations.order)
                    self.forms.extend(made.forms)
                    declarations = made.groups
                else:
                    target, metadata = operation(crossing)
                    if crossing.target is not target:
                        raise NotImplementedError('bound collectives require functional graph operands')
                    actual = map_arg(crossing.args, live.__getitem__)
                    keywords = map_arg(crossing.kwargs, live.__getitem__)
                    named = {a.name: actual[i] if i < len(actual) else keywords.get(a.name, a.default_value)
                             for i, a in enumerate(target._schema.arguments)}
                    source = named.get('input', named.get('x', named.get('self', named.get('partials'))))
                    if metadata.get('dim', 0) != 0:
                        raise NotImplementedError('nonleading reduction axes require partial tensor views')
                    native = dist.distributed_c10d._resolve_process_group(metadata['group_name'])
                    declarations = lower(module)
                    calls = declarations[metadata['group_name']]
                    if len(calls) != 1:
                        raise ValueError('a bound crossing declares one collective group')
                    outputs_flat = tree_flatten(targets)[0]
                    sources = named.get('inputs', [source])
                    if len(outputs_flat) != len(calls[0]) or len(sources) != len(calls[0]):
                        raise ValueError('each declared collective needs its source and final output')
                    made, _ = backend.prepare_fragments(list(zip(calls[0], sources, outputs_flat)), native)
                    self.steps.append((made, 'enqueue'));self.order.append(made)
                for name, calls in declarations.items():
                    self.groups.setdefault(name, []).extend(calls)
                values[node] = targets
            elif node.op == 'output':
                self.output = map_arg(node.args[0], values.__getitem__)
            else:
                raise NotImplementedError(f'bound graph configuration node {node.op}: {node.target}')
        self.preparations = Preparations(self.forms)
        self.graph = execution_sequence(self.steps, self.output)
        self.run = self.graph.forward

    def __call__(self, repetitions=1):
        from . import _stream
        self.preparations.bind(repetitions)
        for _ in range(repetitions):
            for step in self.order:
                if isinstance(step, list):
                    for index in step:
                        _stream.compile_crossing(index)
                else:
                    step.realize()


class Function:
    """One numerical callable and its framework-derived AOT declaration.

    Tensor arguments are fake inputs during tracing and are bound once to the
    realized callable. ``declare`` retraces after placement changes. Compilation operates on the
    complete explicit forward/backward graph, with parameter gradients retained
    as outputs; a compiler cannot discard the backward as unused work. No payload
    function is executed to discover communication. Execution uses only the
    realized FX callable; compilation failures propagate to the caller.
    """
    def __init__(self, function, *arguments, parameters=()):
        self.function, self.parameters = function, tuple(parameters)
        self.arguments = arguments
        self.graph, self.groups, self.run, self.output_spec = None, {}, None, None
        self.forms, self.preparations = [], None

    def declare(self, group):
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
        return self.declare_graph(graph, group, declarations)

    def declare_graph(self, graph, group, declarations=None):
        """Compile an existing complete graph without retracing its operators."""
        from functools import partial
        declarations = lower(graph) if declarations is None else declarations
        if any(_bound(node) for node in graph.graph.nodes):
            prepared = BoundGraph(graph, self.arguments, group)
            self.graph, self.groups, self.run, self.forms = prepared.graph, prepared.groups, prepared.run, prepared.forms
            self.preparations = prepared
            return self.groups.get(group.group_name, []) if group is not None else []
        from . import compiler
        captured = []
        token = _CAPTURE.set(captured)
        try:
            # Backward is explicit in this graph. No numerical execution or
            # alternative eager function is used to discover its declarations.
            with torch.no_grad():
                run = partial(compiler(graph, list(self.arguments), whole_graph=True), *self.arguments)
        finally:
            _CAPTURE.reset(token)
        if declarations and not captured:
            raise RuntimeError('the compiler did not publish the complete communication graph')
        declarations, forms = {}, []
        for lowered, prepared in captured:
            forms.extend(prepared)
            for name, calls in lowered.items():
                declarations.setdefault(name, []).extend(calls)
        self.graph, self.groups, self.run, self.forms = graph, declarations, run, forms
        self.preparations = Preparations(forms)
        return declarations.get(group.group_name, []) if group is not None else []

    def prepare(self, repetitions=1):
        """Bind source/landing generations before entering the numerical call."""
        self.preparations(repetitions)

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
