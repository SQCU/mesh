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
_EAGER_CROSSINGS = {}


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
    forms = [(*form, tuple(slots[crossing_key(form)])) if crossing_key(form) in slots else form for form in forms]
    capture = _CAPTURE.get()
    if capture is not None:
        capture.append((declarations, forms))
    return forms


def eager_index(x, w, dim, sizes, name, adds, repeats, destination):
    form = (x, w, dim, tuple(sizes), name, adds, repeats)
    key = (*crossing_key(form), destination)
    slots, cursor = _EAGER_CROSSINGS[key]
    index = slots[cursor]
    _EAGER_CROSSINGS[key][1] = (cursor + 1) % len(slots)
    return index


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
            calls = [backend.Call(backend.ALLREDUCE if root < 0 else backend.REDUCE,
                                  x.numel(), x.dtype, root=max(0, root), op=operator)]
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
        self.forms, self.bindings, self.eager = [], {}, {}
        for key, count in batches.items():
            x, w, dim, sizes, name, adds, repeats = key
            group = dist.distributed_c10d._resolve_process_group(name)
            if len(sizes) > 1 and getattr(group, 'handle', 0):
                form = (prototype(x), prototype(w), dim, list(sizes), name, adds, repeats)
                self.forms.append((*form, count))
        # Eager declarations retain one output per occurrence in the complete
        # function. Compiled declarations already carry distinct graph slots.
        for key, count in Counter(crossing_key(form) for form in forms if len(form) == 7).items():
            x, w, dim, sizes, name, adds, repeats = key
            group = dist.distributed_c10d._resolve_process_group(name)
            if len(sizes) <= 1 or not getattr(group, 'handle', 0):
                continue
            form = (prototype(x), prototype(w), dim, list(sizes), name, adds, repeats)
            for destination in (False, True):
                eager_key = (*key, destination)
                slots = list(_EAGER_CROSSINGS.get(eager_key, ([], 0))[0][:count])
                slots.extend(crossing_slot(form, destination)[0] for _ in range(count - len(slots)))
                self.eager[eager_key] = slots
                for index in slots:
                    self.bindings[index] = (*form, destination)
        for source in forms:
            key = crossing_key(source)
            x, w, dim, sizes, name, adds, repeats = key
            for index, destination in source[7] if len(source) > 7 else ():
                self.bindings[index] = (prototype(x), prototype(w), dim, list(sizes), name, adds, repeats, destination)

    def __call__(self, repetitions=1):
        from . import _stream
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily():
            for form in self.forms:
                _stream.fused_prepare(*form[:-1], form[-1] * repetitions)
            for index, operands in self.bindings.items():
                _stream.prepare_crossing(index, *operands)
            for key, slots in self.eager.items():
                _EAGER_CROSSINGS[key] = [slots, 0]


class Function:
    """One numerical callable and its framework-derived AOT declaration.

    ``declare`` retraces after placement changes. Compilation operates on the
    complete explicit forward/backward graph, with parameter gradients retained
    as outputs; a compiler cannot discard the backward as unused work. No payload
    function is executed to discover communication. Eager execution keeps its
    ordinary Python callable; compiled execution uses the realized FX callable.
    """
    def __init__(self, function, *, parameters=(), compiler=None):
        self.function, self.parameters, self.compiler = function, tuple(parameters), compiler
        self.graph, self.groups, self.run, self.output_spec = None, {}, None, None
        self.forms, self.preparations = [], None

    def declare(self, group):
        from torch.utils._pytree import tree_flatten
        from torch.distributed.tensor import DTensor

        def recorded():
            output = self.function()
            values, self.output_spec = tree_flatten(output)
            gradients = [None if p.grad is None else p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                         for p in self.parameters]
            return (*values, *gradients)

        graph, declarations = trace(recorded, parameters=self.parameters)
        forms = []
        declarations = lower(graph, forms=forms)
        if self.compiler is None:
            run = self.function
        else:
            from torch._dynamo import lookup_backend
            captured = []
            token = _CAPTURE.set(captured)
            try:
                # Backward is already explicit in this graph. A second autograd
                # layer would add unrelated saved tensors and gradient work.
                with torch.no_grad():
                    run = lookup_backend(self.compiler)(graph, [])
            finally:
                _CAPTURE.reset(token)
            if captured:
                declarations, forms = {}, []
                for lowered, prepared in captured:
                    forms.extend(prepared)
                    for name, calls in lowered.items():
                        declarations.setdefault(name, []).extend(calls)
            elif self.compiler not in ('eager', 'aot_eager'):
                raise ValueError('this compiler must publish its final collective graph at configuration')
        self.graph, self.groups, self.run, self.forms = graph, declarations, run, forms
        self.preparations = Preparations(forms)
        return declarations.get(group.group_name, [])

    def prepare(self, repetitions=1):
        """Bind source/landing generations before entering the numerical call."""
        self.preparations(repetitions)

    def __call__(self):
        if self.run is None:
            raise RuntimeError('realize the function declaration before execution')
        if self.compiler is None:
            return self.run()
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
