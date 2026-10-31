"""FX numerical lowering into the native recorder's command descriptions.

Inductor supplies tensor IR, fusion and Metal source. Its Python wrapper is a
preparation program: allocations, views and kernel bindings run once here.
No generated wrapper, ATen fallback or shader launcher runs with the payload.
"""
from contextvars import ContextVar
from functools import lru_cache
import math
import struct

import torch

from . import _stream
from .compilation import operands as compiler_operands
from .operand import TensorView, descriptor, rebase_typed

_records = ContextVar('mesh_native_commands')
_ALLOC_ADAPTERS = {'torch.ops.torch_mesh.linear.default': 'linear',
                   'torch.ops.torch_mesh.linears.default': 'linears',
                   'torch.ops.aten.topk.default': 'topk',
                   'torch.ops.aten.sort.default': 'sort',
                   'torch.ops.aten.sort.stable': 'sort',
                   'torch.ops.aten.cumsum.default': 'cumsum',
                   'torch.ops.torch_mesh.draw.default': 'draw',
                   'torch.ops.torch_mesh.retain.default': 'retain',
                   'torch.ops.torch_mesh.fwht.default': 'fwht',
                   'torch.ops.torch_mesh.fwht_stage.default': 'fwht_stage',
                   'torch.ops.torch_mesh.native_kernel.default': 'kernel'}
_OUT_ADAPTERS = {'extern_kernels.mm': 'mm', 'extern_kernels.bmm': 'mm', 'extern_kernels.addmm': 'addmm'}


def empty_strided(shape, stride, dtype, device):
    """Bind a compiler-assigned view to bridge storage during preparation."""
    from . import empty
    extent = 0 if math.prod(shape) == 0 else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    return empty((extent,), dtype=dtype, device=device).as_strided(shape, stride)


@lru_cache(None)
def shader_source(source):
    from pathlib import Path
    from torch.utils._cpp_embed_headers import _embed_headers
    return _embed_headers(source.splitlines(keepends=True), [Path(torch.__file__).parent / 'include'], set())


def shader(source, name, access, names=(), types=(), accesses=()):
    """Retain the emitted command declaration before final pipeline binding."""
    source = shader_source(source)

    def bind(*arguments, threads, group_size=None):
        from .operand import byte_operand
        tensors, constants = [], {}
        for index, value in enumerate(arguments):
            tensors.append(byte_operand(value) if isinstance(value, (torch.Tensor, TensorView)) else None)
            if not isinstance(value, (torch.Tensor, TensorView)):
                constants[index] = struct.pack('q', int(value))
        _records.get().append(dict(kind='shader', source=source, function=name, access=access,
            accesses=accesses,
            operands=tensors, names=tuple(names), types=tuple(types), index_dtype=torch.uint8, constants=constants,
            binding_origins=tuple(None if value is None else value.binding_origin for value in tensors),
            grid=tuple(map(int, threads)) + (1,) * (3 - len(threads)),
            threadgroup=None if group_size is None else tuple(map(int, group_size)) + (1,) * (3 - len(group_size)),
            thread_grid=True))
    return bind


def command_records(declarations, *, with_reaches=False):
    """Realize the shared preparation declarations into ordinary command records."""
    from . import layout
    for declaration in declarations:
        if not isinstance(declaration, dict):
            yield (declaration, None) if with_reaches else declaration
            continue
        if declaration['kind'] == 'product':
            from .matrix import product as prepare
            yield from command_records(declaration['before'], with_reaches=with_reaches)
            if declaration['operands']['out'].numel():
                yield from command_records(prepare(**declaration['operands'], **declaration['parameters']),
                                           with_reaches=with_reaches)
            yield from command_records(declaration['after'], with_reaches=with_reaches)
            continue
        operands = declaration['operands']
        handle = layout.pipeline('', {i: value for i, value in enumerate(operands) if value is not None},
                                 declaration['source'], declaration['function'])
        grid, group = declaration['grid'], declaration['threadgroup']
        if group is None:
            remaining, _, _ = _stream.pipeline_facts(handle)
            group = []
            for extent in grid:
                part = min(remaining, extent)
                group.append(part)
                remaining //= part
            group = tuple(group)
        record = (handle, operands, declaration['constants'], grid, group, declaration['thread_grid'], declaration['access'])
        yield (record, declaration['reaches']) if with_reaches else record


def linear(x, w, bias=None, act=0, x1=None, *, out=None):
    return linears([x], [w], [bias], [act], [x1], outputs=None if out is None else [out])[0]


def linears(xs, ws, biases, acts, x1s, *, outputs=None):
    from .placement import output
    outputs = [output((x.shape[0], w.shape[0]), x.dtype, x.device, out)
               for x, w, out in zip(xs, ws, outputs if outputs is not None else [None] * len(xs), strict=True)]
    if (all(descriptor(value).affine for value in (*xs, *ws, *biases, *x1s, *outputs) if value is not None)
        and all(x.dtype == w.dtype == torch.half and x.shape[0] <= compiler_operands()['LINEAR_ROW_CROSSOVER'] and x.stride(1) == w.stride(1) == 1
           and (part is None or part.stride(1) == 1) and (bias is None or bias.is_contiguous())
           for x, w, part, bias, out in zip(xs, ws, x1s, biases, outputs))):
        handles = lambda values: [None if v is None else TensorView.of(v).tensor() for v in values]
        _records.get().extend(_stream.linear_commands(handles(xs), handles(ws), handles(biases), acts, handles(x1s), handles(outputs)))
    else:
        from .matrix import product
        for x, w, part, bias, act, out in zip(xs, ws, x1s, biases, acts, outputs):
            if out.numel():
                _records.get().extend(product(x, w.t(), out, x1=part, bias=bias, act=bool(act)))
    return outputs


def product(x, w, out, *, before=(), after=(), **parameters):
    roles = dict(x=x, w=w, out=out)
    roles.update((name, parameters.pop(name)) for name in ('x1', 'bias') if name in parameters)
    _records.get().append(dict(kind='product', operands=roles, parameters=parameters,
                              before=tuple(before), after=tuple(after)))


def mm(x, w, *, out):
    product(x, w, out)


def addmm(bias, x, w, *, out, alpha=1, beta=1):
    product(x, w, out, bias=bias, alpha=alpha, beta=beta)


def topk(x, k, dim=-1, largest=True, sorted=True, *, outputs=None):
    from .selection import topk as prepare
    output, commands = prepare(x, k, dim, largest, sorted, outputs=outputs)
    _records.get().extend(commands)
    return output


def sort(x, dim=-1, descending=False, *, stable=None, outputs=None):
    """Sorting is the shared ordered selection of the complete axis.

    The selector orders equal values by their original index, so it also
    implements the stable declaration without another numerical kernel.
    """
    shape = x.shape or (1,)
    source = TensorView.of(x).reshape(shape)
    destinations = None if outputs is None else [None if value is None else TensorView.of(value).reshape(shape)
                                                for value in outputs]
    values = topk(source, shape[dim], dim, descending, True, outputs=destinations)
    return tuple(TensorView.of(value).reshape(x.shape) for value in values)


def cumsum(x, dim, *, dtype=None, out=None):
    from .scan import cumsum as prepare
    output, commands = prepare(x, dim, dtype=dtype, out=out)
    _records.get().extend(commands)
    return output


def draw(*args, out=None):
    from .argmax import prepare_draw
    output, commands = prepare_draw(*args, out=out)
    _records.get().extend(commands)
    return output


def retain(*args, outputs=None):
    from .argmax import prepare_retain
    output, commands = prepare_retain(*args, outputs=outputs)
    _records.get().extend(commands)
    return output


def fwht(*args, out=None):
    from .structured import prepare
    output, commands = prepare(*args, out=out)
    _records.get().extend(commands)
    return output


def fwht_stage(*args, out=None):
    from .structured import prepare_stage
    output, commands = prepare_stage(*args, out=out)
    _records.get().extend(commands)
    return output


def kernel(*args, outputs=None):
    from .kernel import prepare
    output, commands = prepare(*args, outputs=outputs)
    _records.get().extend(commands)
    return output


class Program:
    """Bound numerical command records; composition happens during preparation."""
    def __init__(self, commands, output, pages=()):
        self.commands, self.output, self.pages = commands, output, tuple(pages)

    def append(self, schedule):
        schedule.append(self.commands)


def body_accesses(body):
    """Read the scheduler's existing scalar graph, including guarded blocks.

    Nodes and indirect producers are retained by identity. This describes the
    captured scalar program; it neither interprets it nor captures it again.
    """
    import sympy
    from torch._inductor.loop_body import LoopBody

    reads, writes, producers = [], [], {}

    def index(value):
        return (body.indexing_exprs[value.args[0]] if isinstance(value, torch.fx.Node)
                else sympy.Integer(value))

    def visit(block, predicates):
        for node in block.graph.nodes:
            if node.op == 'call_module':
                if node.target in body.subblocks:
                    visit(body.subblocks[node.target], (*predicates, node.args[0]))
                clone = getattr(body.submodules[node.target], 'clone', None)
                if clone is not None and clone.func is LoopBody.bind_set_indirect_shim:
                    declaration = dict(clone.keywords)
                    symbol = declaration.pop('var')
                    producers[symbol] = dict(declaration, node=node, value=node.args[0], predicates=predicates)
            elif node.op == 'call_method':
                if node.target in ('load', 'load_seed', 'store', 'store_reduction'):
                    item = dict(name=node.args[1], index=index(node.args[2]), node=node,
                                predicates=predicates, index_producers=producers, body=body)
                    (reads if node.target in ('load', 'load_seed') else writes).append(item)
                elif node.target == 'bucketize':
                    # This scalar primitive performs dependent search reads.
                    # Retain its declared address bounds and exact IR operands;
                    # a buffer-wide StarDep is not its executed read relation.
                    boundaries = node.args[2]
                    sorter = node.args[6] if len(node.args) > 6 else node.kwargs.get('sorter')
                    for argument in (boundaries, sorter):
                        if argument is not None:
                            reads.append(dict(name=argument[0], index=None, bounds=tuple(map(index, argument[1:])),
                                              node=node, predicates=predicates, index_producers=producers, body=body))

    visit(body.root_block, ())
    return dict(body=body, reads=tuple(reads), writes=tuple(writes), index_producers=producers)


def prepare_graph(module, metadata, *, materializations=()):
    """Lower once to logical IR, before assigning physical page operands.

    Materializations name values that later consumers must read from storage.
    Declare those boundaries here; ``bind`` later chooses their actual pages
    and emits commands from this same graph.
    """
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._inductor import config
    from torch._inductor.codegen.common import get_device_op_overrides
    from torch._inductor.codegen.mps import MetalScheduling, MetalKernel
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.choices import InductorChoices
    from torch._inductor.graph import GraphLowering
    from torch._inductor.scheduler import Scheduler, BaseSchedulerNode, SchedulerNode
    from torch._inductor.virtualized import V
    from torch._inductor.debug import DebugContext
    from torch._inductor import ir
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from torch.fx.node import map_arg
    from .placement import Placements, same_view
    from contextlib import contextmanager
    from mesh_compute import hardware

    placements, bound_storage, views, storage_layouts = Placements(), {}, [], {}
    assigned_results, assigned_owners = None, None
    product_arguments, external_outputs, command_accesses = [], [], []
    page_values = []
    configuration = compiler_operands()
    reduction_groups = hardware()['gpu']['cores'] * configuration['REDUCTION_GROUPS_PER_CORE']

    class NativeChoices(InductorChoices):
        def reduction_split_factor(self, device, reduction_numel_hint, numel_hint, inner_reduction):
            # Split the numerical IR, not execution. Both stages are ordinary
            # page producers; the recorder retains their data dependencies.
            tiles = (reduction_numel_hint + configuration['REDUCTION_TILE_ELEMENTS'] - 1) // configuration['REDUCTION_TILE_ELEMENTS']
            rows = max(1, numel_hint)
            groups = (reduction_groups + rows - 1) // rows
            return max(1, min(tiles, groups))

    class PageKernel(MetalKernel):
        # These objects exist during source emission only. The numerical body
        # receives page bindings and the resulting ordinary address expression.
        max_threadgroup_size = min(configuration['ROW_THREADS'], _stream.device_facts()[1])
        simd_group_size = configuration['SIMD_WIDTH']

        @contextmanager
        def access(self, name):
            previous = getattr(self, 'page_access', None)
            self.page_access = name
            try:
                yield
            finally:
                self.page_access = previous

        def index_to_str(self, index):
            expression = super().index_to_str(index)
            name = getattr(self, 'page_access', None)
            if name is not None and assigned_owners is not None and name not in storage_layouts:
                raise ValueError(f'owned numerical access {name} has no assigned page operand')
            if name is not None:
                bindings = self.__dict__.setdefault('page_bindings', {})
                slot = bindings.setdefault(name, len(bindings))
                return f'mesh_storage_{slot}({expression})'
            return expression

        def load(self, name, index):
            with self.access(name):
                return super().load(name, index)

        def store(self, name, index, value, mode=None):
            with self.access(name):
                return super().store(name, index, value, mode)

        def store_reduction(self, name, index, value):
            with self.access(name):
                return super().store_reduction(name, index, value)

        def reduction(self, dtype, src_dtype, reduction_type, value):
            # A broadcast value still covers the entire reduction domain.
            # Establish its loop before choosing accumulator scope, even when
            # no load has yet needed the reduction index. A later fused load
            # must not move that value into a loop without its accumulator.
            for tree in self.range_trees:
                if tree.is_reduction:
                    tree.full_range().codegen()
            return super().reduction(dtype, src_dtype, reduction_type, value)

    class PageView(ir.ReinterpretView):
        def codegen_reference(self, writer=None):
            if hasattr(self, 'selected_binding'):
                original, unit, physical = (self.selected_binding[key] for key in ('source', 'source_dtype', 'operand'))
                mapped = rebase_typed(ir_layout(self.layout), self.layout.dtype.itemsize,
                                      original, unit.itemsize, physical.layout)
                operand = TensorView(physical.storage_as(self.layout.dtype), mapped)
            else:
                operand = self.operand
            index = len(views)
            views.append(operand)
            return f'mesh_views[{index}]'

    def page_view(storage, layout, operand):
        result = PageView(data=storage, layout=layout)
        result.operand = operand
        return ir.TensorBox(result)

    def ir_layout(layout):
        return _stream.TensorLayout(tuple(map(int, layout.size)), tuple(map(int, layout.stride)), int(layout.offset))

    def ir_operand(value):
        if value is None:
            return None
        storage, layout = ir.as_storage_and_layout(value, freeze=False)
        return dict(name=storage.get_name(), dtype=layout.dtype, storage_dtype=storage.get_dtype(),
                    storage_bytes=storage.get_layout().storage_size() * storage.get_dtype().itemsize,
                    view=ir_layout(layout))

    def external_domains(operation, algebra):
        """Bind shared mathematical declarations to the retained IR's views."""
        import json
        import mesh_matrix
        from .operand import binding_layout, layout_configuration

        operand = ir_operand

        def product(values, parameters):
            declared = {name: None if value is None else dict(shape=tuple(value['view'].shape), type=value['dtype'])
                        for name, value in values.items()}
            domains = mesh_matrix.product_domains(algebra, **declared)
            def accesses(relations):
                return {name: dict(values[name], relation=relation.then(
                    algebra.layout(layout_configuration(values[name]['view']))))
                        for name, relation in relations.items()}
            finish = domains['finish']
            return dict(work=domains['work'], reads=accesses(domains['reads']), writes=accesses(domains['writes']),
                        reduction=domains['reduction'], parameters=parameters,
                        finish=dict(work=finish['work'], reads=accesses(finish['reads']), writes=accesses(finish['writes'])))

        if isinstance(operation, ir.ExternKernelOut):
            adapter = _OUT_ADAPTERS.get(operation.get_kernel_name())
            if adapter in ('mm', 'addmm'):
                inputs = list(operation.inputs)
                bias = operand(inputs.pop(0)) if adapter == 'addmm' else None
                x, w = map(operand, inputs)
                return (product(dict(x=x, w=w, out=operand(operation.output_view or operation), bias=bias),
                                dict(alpha=operation.kwargs.get('alpha', 1), beta=operation.kwargs.get('beta', 1), act=False)),)
        if not isinstance(operation, ir.FallbackKernel):
            return ()
        adapter = _ALLOC_ADAPTERS.get(operation.get_kernel_name())
        if adapter not in ('linear', 'linears', 'kernel'):
            return ()
        args, keywords = operation.unflatten_args(operation.inputs, operation.constant_args)
        named = {field.name: args[i] if i < len(args) else keywords.get(field.name, field.default_value)
                 for i, field in enumerate(operation.op_overload._schema.arguments)}
        outputs = tree_flatten(operation.codegen_outputs())[0]
        if adapter in ('linear', 'linears'):
            fields = ('x', 'w', 'bias', 'act', 'x1')
            arguments = ([named[field]] for field in fields) if adapter == 'linear' else (named[field] for field in fields)
            result = []
            for x, w, bias, act, x1, out in zip(*arguments, outputs, strict=True):
                weight = operand(w)
                weight['view'] = weight['view'].permute((1, 0))
                result.append(product(dict(x=operand(x), w=weight, out=operand(out), bias=operand(bias), x1=operand(x1)),
                                      dict(alpha=1, beta=1, act=act)))
            return tuple(result)
        declarations = json.loads(named['operands'])
        uses = {}
        def use(slot, value):
            selected = operand(value)
            declaration = declarations[str(slot)]
            selected['view'] = binding_layout(selected['view'], declaration)
            uses[slot] = selected
        for slot, value in zip(named['input_slots'], named['inputs'], strict=True):
            use(slot, value)
        for slots, value in zip(named['output_slots'], outputs, strict=True):
            for slot in slots:
                use(slot, value)
        result = []
        for declaration in declarations.get('domains', ()):
            if declaration['kind'] != 'product':
                raise ValueError(f"undeclared native mathematical domain {declaration['kind']}")
            result.append(product({name: uses[slot] for name, slot in declaration['operands'].items()}, declaration['parameters']))
        return tuple(result)

    def computed_domains(operation, body, algebra):
        """Address relations and scalar operands from the same prepared body."""
        import sympy
        from .operand import index_relation, scalar_expression
        data = operation.data
        iterations, reductions = map(tuple, body.sizes)
        variables = (*body.iter_vars, *body.reduce_vars)
        work = (*iterations, *reductions)
        declared = body_accesses(body)
        scalar_cache = {}

        def accesses(items, shape, axes):
            known = set(axes) - {0}
            result = {}
            for slot, item in enumerate(items):
                index = item['index']
                relation = (index_relation(algebra, index, axes, shape)
                            if index is not None and index.free_symbols <= known else None)
                conditions = tuple(scalar_expression(body, value, scalar_cache) for value in item['predicates'])
                condition = sympy.And(*(value.expr for value in conditions if value is not None))
                predicate_domain = (index_relation(algebra, condition, axes, shape).preimage(algebra.set('{ [i] : i = 1 }', 1))
                                    if conditions and condition.free_symbols <= known else None)
                if relation is not None and predicate_domain is not None:
                    relation = relation.subtract_domain(relation.domain().binary(predicate_domain, 'subtract'))
                dtype = V.graph.get_dtype(item['name'])
                storage = V.graph.get_buffer(item['name'])
                # A changing address has ordinary scalar producers. Retain
                # their declared value dimensions alongside the work axes in
                # the same relation, rather than substituting the whole input
                # or treating a changing selection as another tensor kind.
                symbols = tuple(symbol for symbol in declared['index_producers']
                                if index is not None and symbol in index.free_symbols)
                index_axes = (*axes, *symbols)
                index_shape = (*shape, *(body.indirect_var_ranges[symbol] for symbol in symbols))
                index_known = set(index_axes) - {0}
                index_map = relation if not symbols else (
                    index_relation(algebra, index, index_axes, index_shape) if index.free_symbols <= index_known else None)
                index_predicate = predicate_domain if not symbols else (
                    index_relation(algebra, condition, index_axes, index_shape).preimage(algebra.set('{ [i] : i = 1 }', 1))
                    if conditions and condition.free_symbols <= index_known else None)
                if symbols and index_map is not None and index_predicate is not None:
                    index_map = index_map.subtract_domain(index_map.domain().binary(index_predicate, 'subtract'))
                result[slot] = dict(item, dtype=dtype, storage_dtype=dtype, variables=axes,
                                    storage_bytes=storage.get_layout().storage_size() * dtype.itemsize,
                                    operation=operation, relation=relation, predicate_domain=predicate_domain,
                                    index_space=dict(shape=index_shape, variables=index_axes, work_dimensions=len(axes),
                                        values=tuple(dict(symbol=symbol, **declared['index_producers'][symbol]) for symbol in symbols),
                                        relation=index_map, predicate_domain=index_predicate),
                                    unresolved_predicates=tuple(node for node, value in zip(item['predicates'], conditions, strict=True)
                                                                if value is None or not sympy.sympify(value.expr).free_symbols <= known))
            return result

        reduction = None
        if isinstance(data, ir.Reduction):
            kind = data.get_reduction_type()
            reduction = dict(operation=kind, axes=tuple(range(len(iterations), len(work))),
                             identity=data.default_value(kind, data.get_dtype()),
                             dtype=data.get_dtype(), source_dtype=data.src_dtype)
        finish_shape = iterations if reduction is not None else work
        finish_axes = variables[:len(iterations)] if reduction is not None else variables
        return (dict(work=work, variables=variables, body=body, index_producers=declared['index_producers'],
                     reads=accesses(declared['reads'], work, variables),
                     writes=accesses(declared['writes'], work, variables), reduction=reduction,
                     finish=dict(work=finish_shape, variables=finish_axes, reads={},
                                 writes=accesses(declared['writes'], finish_shape, finish_axes))),)

    from .layout import declare
    placeholders = [node for node in module.graph.nodes if node.op == 'placeholder']
    declare(module.graph, placements, [*(node.meta.get('val') for node in placeholders), *materializations])

    class Preparation(PythonWrapperCodegen):
        def get_output_refs(self):
            if assigned_results is not None:
                return [f'mesh_results[{i}]' for i in range(len(assigned_results))]
            return super().get_output_refs()

        def codegen_reinterpret_view(self, data, size, stride, offset, writeline, dtype=None):
            if data.get_name() not in storage_layouts:
                return super().codegen_reinterpret_view(data, size, stride, offset, writeline, dtype)
            binding = storage_layouts[data.get_name()]
            original, unit, physical = (binding[key] for key in ('source', 'source_dtype', 'operand'))
            dtype = data.get_dtype() if dtype is None else dtype
            described = _stream.TensorLayout(tuple(map(int, size)), tuple(map(int, stride)), int(offset))
            mapped = rebase_typed(described, dtype.itemsize, original, unit.itemsize, physical.layout)
            views.append(TensorView(physical.storage_as(dtype), mapped))
            return f'mesh_views[{len(views) - 1}]'

        def make_allocation(self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False):
            if name in bound_storage:
                return f'{name} = mesh_destinations[{name!r}]'
            if assigned_owners is not None:
                raise ValueError(f'owned numerical storage {name} has no assigned page operand')
            if is_pinned:
                raise NotImplementedError('native graph storage uses the bridge region, not a separate pinned allocator')
            allocation_shape = shape if allocation_shape is None else allocation_shape
            sizes = self.codegen_python_shape_tuple(allocation_shape)
            strides = self.codegen_python_shape_tuple(stride)
            made = f'{name} = mesh_native.empty_strided({sizes}, {strides}, {dtype}, {str(device)!r})'
            if tuple(shape) != tuple(allocation_shape):
                made += f'.as_strided({self.codegen_python_shape_tuple(shape)}, {strides})'
            return made

        def can_reuse(self, input_buffer, output_buffer=None):
            # Logical landing declarations constrain reuse before pages exist.
            outputs = (output_buffer.get_buffer_names() if isinstance(output_buffer, BaseSchedulerNode)
                       else () if output_buffer is None else (output_buffer.get_name(),))
            if any(name in V.graph.never_reuse_buffers or name in bound_storage for name in outputs):
                return False
            return super().can_reuse(input_buffer, output_buffer)

        def selected_external(self, node):
            selected = getattr(self, 'current_external_products', None)
            if selected is None:
                return False
            for plan in selected:
                product_arguments.append(plan)
                self.writeline(f'mesh_native.product(**mesh_products[{len(product_arguments) - 1}])')
            from torch.utils._pytree import tree_map
            declared = node.codegen_outputs() if isinstance(node, ir.FallbackKernel) else node
            outputs = tree_map(lambda leaf: bound_storage[leaf.get_name()] if isinstance(leaf, ir.IRNode) else leaf,
                               declared)
            external_outputs.append(outputs)
            self.writeline(f'{node.get_name()} = mesh_external_outputs[{len(external_outputs) - 1}]')
            return True

        def _generate_extern_kernel_alloc_helper(self, node, args):
            names = _ALLOC_ADAPTERS
            name = node.get_kernel_name()
            if name not in names:
                raise NotImplementedError(f'no native preparation for {name}; a framework fallback is not an executor')
            outputs = node.codegen_outputs()
            output_refs = [f'mesh_destinations[{leaf.get_name()!r}]' if isinstance(leaf, ir.IRNode)
                           and leaf.get_name() in bound_storage else 'None' for leaf in tree_flatten(outputs)[0]]
            keyword = 'outputs' if names[name] in ('linears', 'topk', 'sort', 'retain', 'kernel') else 'out'
            destination = '[' + ', '.join(output_refs) + ']' if keyword == 'outputs' else output_refs[0]
            self.writeline(f'{node.get_name()} = mesh_native.{names[name]}({", ".join(args)}, {keyword}={destination})')

        def _generate_extern_kernel_out_helper(self, kernel, out, out_view, args, device, stack_traces=None):
            names = _OUT_ADAPTERS
            if kernel not in names or device != 'mps':
                raise NotImplementedError(f'no native preparation for {kernel} on {device}')
            self.writeline(f'mesh_native.{names[kernel]}({", ".join(args)}, out={out_view or out})')

    class MetalSource(MetalScheduling):
        kernel_type = PageKernel

        def codegen_node(self, node):
            pieces = getattr(node, 'page_pieces', None)
            if pieces is None:
                return super().codegen_node(node)
            pending = self.scheduler.buffer_names_to_free.copy()
            self.scheduler.buffer_names_to_free.clear()
            for at, piece in enumerate(pieces):
                future = {self.scheduler.mutation_real_name.get(name, name)
                          for later in pieces[at + 1:] for name in later.used_or_aliased_buffer_names()}
                ready = pending.difference(future)
                pending.difference_update(ready)
                self.scheduler.buffer_names_to_free.update(ready)
                piece.outputs, piece.outputs_by_name = node.outputs, node.outputs_by_name
                piece.ancestors, piece.last_usage = node.ancestors, node.last_usage
                piece.mutation_renames = node.mutation_renames
                super().codegen_node(piece)
            self.scheduler.buffer_names_to_free.update(pending)
            if not pieces:
                node.mark_run()

        def define_kernel(self, source, node_schedule, kernel):
            wrapper = V.graph.wrapper_code
            start = source.index("compile_mps_shader('''") + len("compile_mps_shader('''")
            source_text = source[start:source.rindex("''')")]
            outputs = [name for name in kernel.args.output_buffers if name not in kernel.removed_buffers]
            inputs = list(kernel.args.input_buffers)
            arguments = [*outputs, *inputs]
            declarations = []
            from .layout import strides
            for name, slot in getattr(kernel, 'page_bindings', {}).items():
                buffer = V.graph.get_buffer(name)
                binding = storage_layouts.get(name, dict(source=ir_layout(buffer.get_layout()),
                                                        source_dtype=buffer.get_dtype()))
                original, unit = binding['source'], binding['source_dtype']
                byte_source = original.components(unit.itemsize)
                local = _stream.TensorLayout(tuple(original.shape), strides(original.shape)).components(unit.itemsize)
                text = local.storage_declaration(slot, byte_source)
                text = text[text.index(f'inline long mesh_storage_{slot}('):]
                text = text.replace(f'mesh_storage_{slot}(', f'mesh_logical_storage_{slot}(')
                width = V.graph.get_dtype(name).itemsize
                declarations.append(text + f'inline long mesh_storage_{slot}(long i) '
                    f'{{ return mesh_index<{arguments.index(name)}>(mesh_logical_storage_{slot}(i * {width}l)) / {width}l; }}\n')
            source_text = ''.join(declarations) + source_text
            if 'error' in kernel.headers:
                raise NotImplementedError('runtime index assertions must be resolved in the declared graph')
            access = (1 << (len(outputs) + len(inputs))) - 1 | ((1 << len(outputs)) - 1) << _stream.compilation['ACCESS_WRITE_SHIFT']
            name = f'mesh_kernel_{wrapper.next_kernel_suffix()}'
            types = tuple(V.graph.get_dtype(argument) for argument in arguments)
            from mesh_relation import Algebra
            algebra, accesses = Algebra(), []
            for node in node_schedule:
                if not isinstance(node, SchedulerNode):
                    continue
                for domain in computed_domains(node.node, node._body, algebra):
                    work = algebra.map(domain['work'], (), lambda source, target: []).domain()
                    for side in ('reads', 'writes'):
                        for item in domain[side].values():
                            if item['name'] in arguments:
                                accesses.append(dict(side=side, slot=arguments.index(item['name']), access=item, work=work,
                                                     origins=tuple(node.node.get_origins())))
            index = len(command_accesses)
            command_accesses.append(tuple(accesses))
            wrapper.header.writeline(f'{name} = mesh_native.shader({source_text!r}, "generated_kernel", {access}, {tuple(arguments)!r}, {types!r}, mesh_accesses[{index}])')
            return name

    class NativeScheduler(Scheduler):
        def codegen_extern_call(self, node):
            if not hasattr(node, 'page_products'):
                return super().codegen_extern_call(node)
            # This is the ordinary external declaration's preparation boundary.
            # Its selected views are already bound, so no full-view references
            # are reconstructed before invoking the shared product declaration.
            from torch._inductor.codegen.common import Kernel
            with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
                node.decide_inplace_update()
                node.mark_run()
            wrapper = V.graph.wrapper_code
            wrapper.current_external_products = node.page_products
            try:
                wrapper.selected_external(node.node)
            finally:
                wrapper.current_external_products = None
            self.free_buffers()

        def create_scheduler_node(self, node):
            prepared = getattr(self, 'prepared_nodes', None)
            if prepared is None:
                result = super().create_scheduler_node(node)
                result.declared_read_writes = result.read_writes
                return result
            result, dependencies = prepared[node.get_operation_name()]
            result._init_from_node(node)
            result.set_read_writes(dependencies)
            return result

        def can_fuse(self, left, right, *args, **kwargs):
            # Distinct reduction fibers have distinct launch domains. They
            # remain ordinary commands; equal single domains use normal fusion.
            if any(hasattr(node, 'page_pieces') for group in (left, right) for node in group.get_nodes()):
                return False
            return super().can_fuse(left, right, *args, **kwargs)

        def merge_loops(self):
            super().merge_loops()
            for group in self.nodes:
                for node in group.get_nodes():
                    for piece in getattr(node, 'page_pieces', ()):
                        piece.merge_loops()

        def create_backend(self, device):
            if device.type != 'mps':
                raise NotImplementedError(f'FX numerical commands require a declared native backend for {device}')
            return MetalSource(self)

    class NativeGraph(GraphLowering):
        def run_node(self, node):
            result = super().run_node(node)
            if node.op == 'output' or 'val' not in node.meta:
                self.source_values[node] = self.graph_outputs if node.op == 'output' else result
                return result
            leaves, spec = tree_flatten(result)
            described = tree_flatten(node.meta['val'])[0]
            for i, (value, logical) in enumerate(zip(leaves, described, strict=True)):
                if not placements.materializes(logical):
                    continue
                # This is a logical dependency boundary. Keep it before later
                # consumers lower, independently of which pages will hold it.
                if isinstance(value, (ir.TensorBox, ir.StorageBox, ir.BaseView)):
                    value.realize()
                while isinstance(value, (ir.TensorBox, ir.StorageBox)):
                    value = value.data
                if isinstance(value, ir.BaseView) and not isinstance(value, ir.ReinterpretView):
                    value = ir.ExternKernel.convert_to_reinterpret_view(value)
                storage, layout = ir.as_storage_and_layout(value)
                view = page_view(storage, layout, None)
                page_values.append((view.data, logical, storage.data, layout))
                leaves[i] = view
            result = tree_unflatten(leaves, spec)
            self.source_values[node] = result
            return result

        def value_binding(self, value, algebra):
            """A realized source IR value's exact view; never realize it here."""
            from .operand import layout_configuration
            with environment(), V.set_graph_handler(self), mode:
                binding = ir_operand(value)
                return dict(binding, value=value, relation=algebra.layout(layout_configuration(binding['view'])))

        def loop_body(self, operation):
            """The scheduler's current cached body, without retracing or freezing."""
            scheduled = self.scheduler.name_to_node[operation.get_operation_name()]
            return scheduled._body if isinstance(scheduled, SchedulerNode) else None

        def consumer_constraints(self):
            """Typed backend eligibility, separate from storage interpretation."""
            from .compilation import api_layouts
            allowed = tuple(getattr(torch, name.lower()) for name in api_layouts('scalar_emitter')['arithmetic_types'])
            declarations = []
            for operation in self.operations:
                body = self.loop_body(operation)
                if body is None:
                    continue
                required = {operation.get_dtype()}
                for block in (body.root_block, *body.subblocks.values()):
                    for node in block.graph.nodes:
                        required.update(value for value in tree_flatten((node.args, node.kwargs))[0] if isinstance(value, torch.dtype))
                accesses = body_accesses(body)
                for access in (*accesses['reads'], *accesses['writes']):
                    required.add(self.get_dtype(access['name']))
                declarations.append(dict(consumer='scalar_emitter', operation=operation,
                    required_types=tuple(sorted(required, key=str)), arithmetic_types=allowed,
                    missing_types=tuple(sorted(required.difference(allowed), key=str))))
            return tuple(declarations)

        def select_domains(self, records):
            """Apply owner-selected work to this IR's existing scalar bodies."""
            import copy
            from torch._inductor import dependencies
            from torch._inductor.loop_body import LoopBody
            from .layout import strides
            from .operand import iteration_order, layout_configuration, symbolic_index

            records = tuple(records)
            selected = {}
            for record in records:
                operation = record['effect']['operation']
                selected.setdefault(operation.get_operation_name(), []).append(record)

            def piece_body(body, domain):
                shapes = tuple(map(tuple, body.sizes))
                boundary = len(shapes[0])
                domains = (domain.project(0, boundary), domain.project(boundary, len(shapes[1])))
                orders = [iteration_order(part, shape) for part, shape in zip(domains, shapes, strict=True)]
                sizes = tuple(shape if whole else (order.shape[0],)
                              for shape, (order, whole) in zip(shapes, orders, strict=True))
                if all(whole for _, whole in orders):
                    return body
                variables, ranges = dependencies.index_vars_no_squeeze(*sizes, prefix='q')
                arguments = []
                for shape, (order, whole), values in zip(shapes, orders, variables, strict=True):
                    if whole:
                        arguments.append(values)
                    else:
                        ordinal = symbolic_index(order.terms(), values[0])
                        coordinates = layout_configuration(_stream.TensorLayout(shape, strides(shape)))['index_coordinates']
                        arguments.append([symbolic_index(term, ordinal) for term in coordinates])
                return LoopBody(body, arguments, ranges, *variables, allow_same_symbol_in_index=True)

            with environment(), V.set_graph_handler(self), V.set_extern_kernel_nodes([]), mode:
                prepared, undeclared = {}, set()
                declarations = {effect['operation'].get_operation_name(): effect for effect in self.declared_effects}
                for operation in self.operations:
                    node = self.scheduler.name_to_node[operation.get_operation_name()]
                    requests = selected.get(operation.get_operation_name(), ())
                    if isinstance(node, SchedulerNode):
                        pieces = []
                        def append(body, domain):
                            piece = copy.copy(node)
                            piece.__dict__.pop('page_pieces', None)
                            piece._body = piece_body(body, domain)
                            piece._sizes = piece._body.sizes
                            device = operation.get_device_or_error()
                            piece.group = device, self.scheduler.get_backend(device).group_fn(piece._sizes)
                            piece.mutation_renames = {}
                            piece.read_writes = node.declared_read_writes
                            piece.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)
                            pieces.append(piece)
                        for request in requests:
                            body = request['domain']['body']
                            boundary = len(body.iter_vars)
                            cuts = (boundary,) if 0 < boundary < len(request['domain']['work']) else ()
                            for domain in request['work'].products(cuts):
                                append(body, domain)
                            if body.reduce_vars and request['finish'] is not None:
                                missing = request['finish']['work'].binary(request['work'].project(0, boundary), 'subtract')
                                if not missing.empty():
                                    # The existing reduction lowering supplies
                                    # its typed identity for an empty fiber.
                                    # This derives only a constant finish body;
                                    # the source expression is never retraced.
                                    if not hasattr(node, 'page_empty_finish'):
                                        from torch._inductor.virtualized import ops
                                        reduction = operation.data
                                        neutral = request['domain']['reduction']['identity']
                                        if isinstance(neutral, (tuple, list)):
                                            neutral = neutral[reduction.output_index]
                                        dtype = reduction.get_dtype()
                                        constant = ir.Pointwise(device=reduction.get_device(), dtype=dtype,
                                            inner_fn=lambda index, value=neutral, dtype=dtype: ops.constant(value, dtype),
                                            ranges=reduction.ranges)
                                        identity = ir.ComputedBuffer(name=operation.get_name(), layout=operation.layout, data=constant)
                                        _, finish_body, _ = identity.get_default_sizes_body()
                                        finish_domain, = computed_domains(identity, finish_body, missing.algebra)
                                        node.page_empty_finish = finish_body, finish_domain
                                    finish_body, finish_domain = node.page_empty_finish
                                    produced = request['domain']['finish']['writes']
                                    selected_finish = None
                                    for write in finish_domain['writes'].values():
                                        for source in produced.values():
                                            if write['name'] == source['name']:
                                                selected = write['relation'].preimage(source['relation'].image(missing))
                                                selected_finish = selected if selected_finish is None else selected_finish.binary(selected, 'union')
                                    append(finish_body, selected_finish)
                        if len(pieces) == 1:
                            node._body, node._sizes, node.group = pieces[0]._body, pieces[0]._sizes, pieces[0].group
                            node.__dict__.pop('page_pieces', None)
                        else:
                            node.page_pieces = tuple(pieces)
                        rw = dependencies.ReadWrites.merge_list([piece.read_writes for piece in pieces])
                    else:
                        import mesh_matrix
                        if declarations[operation.get_operation_name()]['domains']:
                            node.selected_products = tuple(plan for request in requests for plan in mesh_matrix.selected_products(
                                request['domain'], request['work'], request['finish']['work']))
                        else:
                            undeclared.add(operation.get_operation_name())
                        rw = node.declared_read_writes
                    prepared[operation.get_operation_name()] = node, rw
                self.selected_domains = records
                # Reuse the scheduler and its normal dependency/fusion passes.
                # create_scheduler_node retains these bodies and operation
                # objects; no GraphLowering or scalar capture runs again.
                self.scheduler.prepared_nodes = prepared
                self.scheduler._init(self.operations)
                self.pending_declarations = tuple(node.node for group in self.scheduler.nodes for node in group.get_nodes()
                                                  if node.get_name() in undeclared)
            return self

        def consumer_uses(self):
            """Selected external API uses, before the common page-placement solve."""
            from .compilation import api_layouts
            layouts = api_layouts('matrix')
            roles = dict(x='input', x1='input', w='weight', out='output', bias='bias')
            uses = []
            for operation in self.operations:
                name = operation.get_operation_name()
                scheduled = self.scheduler.name_to_node[name]
                for index, plan in enumerate(getattr(scheduled, 'selected_products', ())):
                    for role, use in plan['operands'].items():
                        uses.append(dict(key=(name, index, role), binding=use['access'], relation=use['relation'],
                                         shape=use['shape'], requirements=(layouts[roles[role]],),
                                         access='write' if role == 'out' else 'read'))
            return tuple(uses)

        def emit(self, operands, *, destinations=None, storage=None, results=None, pages=(), uses=None):
            """Bind already selected scalar domains to their assigned pages.

            ``storage`` maps each retained IR storage name to ``source`` (its
            selected original scalar positions), ``source_dtype`` (their explicit
            scalar unit), and ``operand`` (the assigned compact page view in
            that same unit and order). ``results`` retains requested
            output order and repetitions. Placement owns allocation; this mode
            only composes those relations into the existing emitted bindings.
            """
            nonlocal assigned_results, assigned_owners
            missing = [declaration for declaration in self.consumer_constraints() if declaration['missing_types']]
            if missing:
                names = ', '.join(f"{declaration['operation'].get_operation_name()} ({', '.join(map(str, declaration['missing_types']))})"
                                  for declaration in missing)
                raise ValueError(f'scalar_emitter has no type-compatible backend assignment: {names}')
            if getattr(self, 'pending_declarations', ()):
                names = ', '.join(operation.get_operation_name() for operation in self.pending_declarations)
                raise ValueError(f'selected numerical work is missing external access declarations: {names}')
            with environment(), V.set_graph_handler(self), V.set_extern_kernel_nodes([]), mode:
                with unset_fake_temporarily():
                    if storage is not None:
                        if results is None:
                            raise ValueError('selected page binding requires explicit ordered result operands')
                        assigned_results, assigned_owners = tuple(results), tuple(pages)
                        for name, binding in storage.items():
                            physical = TensorView.of(binding['operand'])
                            storage_layouts[name] = dict(source=binding['source'], source_dtype=binding['source_dtype'], operand=physical)
                            if not isinstance(self.get_buffer(name), ir.InputBuffer):
                                bound_storage[name] = physical
                        actuals = [storage_layouts[name]['operand'] if name in storage_layouts else actual
                                   for (name, value), actual in zip(self.graph_inputs.items(), operands, strict=True)]
                        for view, _, buffer, _ in page_values:
                            if buffer.get_name() in storage_layouts:
                                view.selected_binding = storage_layouts[buffer.get_name()]
                    else:
                        for node, actual in zip(placeholders, operands, strict=True):
                            if isinstance(node.meta.get('val'), torch.Tensor):
                                placements.bind(node.meta['val'], actual)
                        if destinations is not None:
                            output_nodes = next(n.args[0] for n in module.graph.nodes if n.op == 'output')
                            described_outputs = tree_flatten(map_arg(output_nodes, lambda n: n.meta['val']))[0]
                            for described, destination in zip(described_outputs, destinations, strict=True):
                                if destination is not None:
                                    placements.bind(described, destination)
                        placements.solve()
                        input_storage = {node.name: placements.get(node.meta['val']) if isinstance(node.meta.get('val'), torch.Tensor) else actual
                                         for node, actual in zip(placeholders, operands, strict=True)}
                        actuals = [input_storage[node.name] for node in placeholders]
                        for view, logical, buffer, layout in page_values:
                            destination = placements.get(logical)
                            name = buffer.get_name()
                            if isinstance(buffer, ir.InputBuffer):
                                from .placement import alias
                                original = next(n.meta['val'] for n in placeholders if n.name == name)
                                actual = alias(original, input_storage[name], logical)
                                if not same_view(actual, destination):
                                    raise ValueError('a supplied graph input cannot be relocated by producer placement')
                                storage_layouts[name] = dict(source=ir_layout(buffer.get_layout()), source_dtype=buffer.get_dtype(),
                                                             operand=TensorView.of(input_storage[name]))
                            elif name not in bound_storage:
                                base = buffer.get_layout()
                                source_layout, base_layout = ir_layout(layout), ir_layout(base)
                                mapped = rebase_typed(base_layout, base.dtype.itemsize, source_layout, layout.dtype.itemsize, destination.layout)
                                bound_storage[name] = TensorView(destination.storage_as(base.dtype), mapped)
                                storage_layouts[name] = dict(source=base_layout, source_dtype=base.dtype, operand=bound_storage[name])
                            view.operand = destination
                    for operation in self.operations:
                        scheduled = self.scheduler.name_to_node[operation.get_operation_name()]
                        if not hasattr(scheduled, 'selected_products'):
                            continue
                        plans = []
                        for index, plan in enumerate(scheduled.selected_products):
                            arguments = dict(plan['parameters'], before=[], after=[])
                            for role in ('out', *[name for name in plan['operands'] if name != 'out']):
                                use = plan['operands'][role]
                                from .operand import relation_layout
                                access = use['access']
                                assigned = None if uses is None else uses.get((operation.get_operation_name(), index, role))
                                if assigned is not None:
                                    arguments[role] = assigned['operand']
                                    arguments['before'].extend(assigned['before'])
                                    arguments['after'].extend(assigned['after'])
                                    continue
                                if not math.prod(use['shape']) and access['name'] not in storage_layouts:
                                    # An empty declared read has no storage demand.
                                    # The API's unused slot names an empty view of
                                    # the already assigned result pages.
                                    logical = relation_layout(use['relation'], use['shape'], 0)
                                    arguments[role] = TensorView(arguments['out'].storage_as(access['dtype']), logical)
                                    continue
                                binding = storage_layouts[access['name']]
                                original, unit, physical = (binding[key] for key in ('source', 'source_dtype', 'operand'))
                                logical = relation_layout(use['relation'], use['shape'],
                                    int(access['storage_bytes']) // access['dtype'].itemsize)
                                mapped = rebase_typed(logical, access['dtype'].itemsize,
                                                      original, unit.itemsize, physical.layout)
                                arguments[role] = TensorView(physical.storage_as(access['dtype']), mapped)
                            plans.append(arguments)
                        scheduled.page_products = tuple(plans)
                source, _ = self.codegen()
                namespace = {'__name__': 'mesh_preparation', 'mesh_destinations': bound_storage, 'mesh_views': views,
                             'mesh_results': assigned_results, 'mesh_products': product_arguments,
                             'mesh_external_outputs': external_outputs, 'mesh_accesses': command_accesses}
                records = []
                token = _records.set(records)
                try:
                    with unset_fake_temporarily():
                        exec(compile(source.value, '<mesh native preparation>', 'exec'), namespace)
                        output = namespace['call'](actuals)
                finally:
                    _records.reset(token)
            for declaration in records:
                if not isinstance(declaration, dict) or declaration['kind'] != 'shader':
                    continue
                for slot, name in enumerate(declaration['names']):
                    if name not in storage_layouts:
                        buffer = self.get_buffer(name)
                        storage_layouts[name] = dict(source=ir_layout(buffer.get_layout()).components(buffer.get_dtype().itemsize),
                                                    source_dtype=torch.uint8, operand=declaration['operands'][slot])
                # Shader slots retain their source identity and coordinate
                # units. A portable caller composes this declared relation
                # with its own pages; it never identifies an input by address.
                declaration['arguments'] = tuple(
                    dict(slot=slot, name=name, dtype=dtype, binding_origin=declaration['binding_origins'][slot],
                         storage_dtype=self.get_buffer(name).get_dtype(),
                         storage_bytes=self.get_buffer(name).get_layout().storage_size()
                             * self.get_buffer(name).get_dtype().itemsize,
                         source=storage_layouts[name]['source'],
                         source_dtype=storage_layouts[name]['source_dtype'])
                    for slot, (name, dtype) in enumerate(zip(declaration['names'], declaration['types'], strict=True)))
                from .operand import bound_access
                reaches = []
                for item in declaration['accesses']:
                    slot = item['slot']
                    argument, operand = declaration['arguments'][slot], declaration['operands'][slot]
                    selected = bound_access(item['access'], item['work'], argument['source'],
                                            argument['source_dtype'], operand.layout)
                    reaches.append((slot, 1 if item['side'] == 'reads' else 2,
                                    TensorView(operand.storage, selected['layout'])))
                declaration['reaches'] = tuple(reaches)
            return dict(commands=tuple(records), output=output,
                        pages=tuple(placements.owners if assigned_owners is None else assigned_owners),
                        storage=dict(storage_layouts))

        def bind(self, operands, **placement):
            declared = self.emit(operands, **placement)
            recorded = tuple(command_records(declared['commands'], with_reaches=True))
            return Program(_stream.Commands([record for record, _ in recorded], [reach for _, reach in recorded]),
                           declared['output'], declared['pages'])

        def effects(self, algebra=None):
            """Logical operations with one shared iteration domain per body.

            Domains retain the cached body's symbols, guards and scalar nodes.
            Scheduler dependencies retain their existing scheduling role.
            Aggregate StarDeps are not scalar address relations.
            Declared external domains compose their logical uses with the same
            selected storage views; each access carries its scalar type.
            """
            from torch.utils._pytree import tree_map
            from mesh_relation import Algebra
            algebra = Algebra() if algebra is None else algebra

            def value_type(name):
                value = self.get_buffer(name)
                if isinstance(value, ir.Buffer) and isinstance(value.layout, ir.MultiOutputLayout):
                    return tree_map(lambda child: (child.get_name(), value_type(child.get_name()))
                                    if isinstance(child, ir.IRNode) else child, value.codegen_outputs())
                if isinstance(value, ir.Buffer) and isinstance(value.layout, ir.NoneLayout):
                    return value.layout
                return self.get_dtype(name)

            with environment(), V.set_graph_handler(self), V.set_extern_kernel_nodes([]), mode:
                records = []
                requirements = {record['operation'].get_operation_name(): record for record in self.consumer_constraints()}
                for operation in self.operations:
                    data = operation.data if isinstance(operation, ir.ComputedBuffer) else None
                    body = self.loop_body(operation)
                    dependencies = self.scheduler.name_to_node[operation.get_operation_name()].read_writes
                    records.append(dict(operation=operation, body=body, scheduling_dependencies=dependencies,
                        scheduling_reads=tuple((dep, value_type(dep.name)) for dep in dependencies.reads),
                        scheduling_writes=tuple((dep, value_type(dep.name)) for dep in dependencies.writes),
                        domains=(external_domains(operation, algebra) if data is None
                                 else computed_domains(operation, body, algebra) if body is not None else ()),
                        consumer_constraints=tuple([requirements[operation.get_operation_name()]])
                            if operation.get_operation_name() in requirements else (),
                        origins=tuple(operation.get_origins())))
                self.declared_effects = tuple(records)
                return self.declared_effects

        def init_wrapper_code(self, *args, **kwargs):
            if self.wrapper_code is not None:
                return
            self.device_type = 'mps'
            self.device_ops = get_device_op_overrides('mps')
            self.wrapper_code = Preparation()
            self.wrapper_code.header.writeline('from torch_mesh import native as mesh_native')

        def _update_scheduler(self):
            if self.scheduler is None:
                self.scheduler = NativeScheduler(self.operations)

    mode = detect_fake_mode((metadata, [node.meta.get('val') for node in module.graph.nodes]))
    if mode is None:
        raise ValueError('native lowering requires the declaration\'s fake tensor metadata')
    @contextmanager
    def environment():
        with config.patch(size_asserts=False, alignment_asserts=False, scalar_asserts=False, split_reductions=True,
                          max_fusion_unique_io_buffers=configuration['BUFFER_BINDINGS'], max_autotune=False,
                          max_autotune_gemm=False, distributed_max_autotune_gemm=False,
                          benchmark_fusion=False, benchmark_combo_kernel=False, assert_indirect_indexing=False), \
                V.set_fake_mode(mode), V.set_choices_handler(NativeChoices()), DebugContext():
            yield

    with environment():
        graph = NativeGraph(module, metadata, shape_env=mode.shape_env, is_inference=True)
        graph.source_values = {}
        with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]), mode:
            # Keep reduction fibers while ownership still selects work. This
            # existing compiler setting applies only to logical construction;
            # selected loops then reach the ordinary Metal/ISA optimizer.
            with config.patch(unroll_reductions_threshold=0):
                graph.run(*metadata)
            graph.never_reuse_buffers.update(buffer.get_name() for _, _, buffer, _ in page_values)
            # This is the scheduler's normal body/layout preparation. Ownership
            # and placement inspect these same cached bodies; codegen retains
            # this wrapper and scheduler after physical page bindings arrive.
            graph.init_wrapper_code()
            graph._update_scheduler()
    return graph


def compile_graph(module, metadata, operands, *, destinations=None):
    """Build one logical IR, then assign its pages and emit native commands."""
    from torch.fx.node import map_arg
    from torch.utils._pytree import tree_flatten
    materializations = []
    if destinations is not None:
        outputs = next(node.args[0] for node in module.graph.nodes if node.op == 'output')
        described = tree_flatten(map_arg(outputs, lambda node: node.meta['val']))[0]
        materializations = [value for value, destination in zip(described, destinations, strict=True) if destination is not None]
    return prepare_graph(module, metadata, materializations=materializations).bind(operands, destinations=destinations)
