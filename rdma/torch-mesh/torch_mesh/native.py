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


def command_declaration(declaration, configuration=None):
    """Expand one numerical preparation declaration, preserving named roles."""
    from . import command_declarations as commands
    configuration = commands.configuration() if configuration is None else configuration
    original = declaration.get('declaration')
    operands = ({role: dict(shape=use['shape'], dtype=use['access']['dtype'],
                           requirements=use.get('requirements', ()))
                 for role, use in original['operands'].items()} if original is not None else
                commands.descriptions(declaration['operands']))
    kind, parameters = declaration['kind'], declaration['parameters']
    if kind == 'product':
        from .matrix import declaration as prepare
    elif kind == 'topk':
        from .selection import declaration as prepare
    elif kind == 'cumsum':
        from .scan import declaration as prepare
    elif kind in ('draw', 'retain'):
        from .argmax import declaration as prepare_sampling
        return prepare_sampling(kind, operands, parameters, configuration)
    elif kind == 'fwht_stage':
        from .structured import declaration as prepare
    else:
        raise ValueError(f'{kind}: missing numerical command declaration')
    return prepare(operands, parameters, configuration)


def command_records(declarations, *, with_reaches=False):
    """Realize the shared preparation declarations into ordinary command records."""
    from . import layout
    for declaration in declarations:
        if not isinstance(declaration, dict):
            yield (declaration, None) if with_reaches else declaration
            continue
        if declaration['kind'] in ('product', 'topk', 'cumsum', 'draw', 'retain', 'fwht_stage'):
            yield from command_records(declaration['before'], with_reaches=with_reaches)
            from .command_declarations import bind
            commands = bind(command_declaration(declaration), declaration['operands'])
            yield from command_records(commands, with_reaches=with_reaches)
            yield from command_records(declaration['after'], with_reaches=with_reaches)
            continue
        operands = declaration['operands']
        views = {i: value for i, value in enumerate(operands) if value is not None}
        origins = {i: None if value.binding_origin is None else value.binding_origin // value.element_size()
                   for i, value in views.items()}
        views.update(declaration.get('index_layouts', {}))
        origins.update((i, 0) for i in declaration.get('index_layouts', {}))
        handle = layout.pipeline(_stream.metal_prelude(), views,
                                 declaration['source'], declaration['function'], origins=origins)
        grid, group = declaration['grid'], declaration['threadgroup']
        maximum, simd, memory = _stream.pipeline_facts(handle)
        policy = declaration.get('threadgroup_policy')
        if policy is not None:
            preferred = max(simd if policy['minimum_simd'] else 1, policy['preferred'])
            group = (min(maximum, preferred), 1, 1)
        elif group is None:
            remaining = maximum
            group = []
            for extent in grid:
                part = min(remaining, max(1, extent))
                group.append(part)
                remaining //= part
            group = tuple(group)
        limits = declaration.get('limits')
        if math.prod(group) > maximum or (limits and memory > limits['threadgroup_memory_bytes']):
            raise ValueError('declared command geometry exceeds the configured device/pipeline capability')
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


def product(x, w, out, *, before=(), after=(), declaration=None, **parameters):
    roles = dict(x=x, w=w, out=out)
    roles.update((name, parameters.pop(name)) for name in ('x1', 'bias') if name in parameters)
    _records.get().append(dict(kind='product', operands=roles, parameters=parameters,
                              before=tuple(before), after=tuple(after), declaration=declaration))


def external(adapter, operands, parameters, *, before=(), after=(), declaration=None):
    """The selected operands of an existing numerical preparation declaration."""
    _records.get().append(dict(kind=adapter, operands=operands, parameters=parameters,
                              before=tuple(before), after=tuple(after), declaration=declaration))


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

    @classmethod
    def prepare(cls, declaration):
        recorded = tuple(command_records(declaration['commands'], with_reaches=True))
        return cls(_stream.Commands([record for record, _ in recorded], [reach for _, reach in recorded]),
                   declaration['output'], declaration['pages'])


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


def clone_body(body):
    """Copy the cached compiler graph and retain its original node identities."""
    import copy
    from collections import Counter, defaultdict
    from torch._inductor.loop_body import LoopBodyBlock
    result = copy.copy(body)
    result.indexing_exprs = dict(body.indexing_exprs)
    result.indexing_exprs_name = {expr: name for name, expr in result.indexing_exprs.items()}
    result.indirect_vars = list(body.indirect_vars)
    result.indirect_var_ranges = dict(body.indirect_var_ranges)
    result.memory_usage = defaultdict(list)
    result.op_counts = Counter()
    result.indexing = None
    nodes = {}
    def block(value):
        made = LoopBodyBlock.__new__(LoopBodyBlock)
        made.body, made.graph = result, torch.fx.Graph()
        for node in value.graph.nodes:
            nodes[node] = made.graph.node_copy(node, lambda value: nodes[value])
            nodes[node].meta['mesh_source_node'] = node.meta.get('mesh_source_node', node)
        return made
    result.root_block = block(body.root_block)
    result.subblocks = {name: block(value) for name, value in body.subblocks.items()}
    result.submodules = {'get_index': result.get_index,
        **{name: value.clone(result) for name, value in body.submodules.items() if name != 'get_index'}}
    return result, nodes


def body_index(result, graph, expression):
    name = result.indexing_exprs_name.get(expression)
    if name is None:
        ordinal = len(result.indexing_exprs)
        while (name := f'index{ordinal}') in result.indexing_exprs:
            ordinal += 1
        result.indexing_exprs[name] = expression
        result.indexing_exprs_name[expression] = name
    return graph.call_module('get_index', (name,))

def finish_body(result, *, writes=False):
    from collections import Counter, defaultdict
    from torch._inductor.loop_body import MemoryEntry, MemoryUsageType
    # get_index names its indirect scalar producer through a symbolic
    # variable, not an FX argument. Include those exact existing edges.
    producers = body_accesses(result)['index_producers']
    live, blocks = set(), set()
    def keep(node):
        if node in live:
            return
        live.add(node)
        for value in node.all_input_nodes:
            keep(value)
        if node.op == 'call_module' and node.target == 'get_index':
            for symbol in result.indexing_exprs[node.args[0]].free_symbols:
                if symbol in producers:
                    keep(producers[symbol]['node'])
        elif node.op == 'call_module' and node.target in result.subblocks:
            keep_block(result.subblocks[node.target])
    def keep_block(block):
        if block in blocks:
            return
        blocks.add(block)
        keep(next(node for node in block.graph.nodes if node.op == 'output'))
        if writes:
            for node in block.graph.nodes:
                if node.op == 'call_method' and node.target in ('store', 'store_reduction', 'partial_accumulate'):
                    keep(node)
    keep_block(result.root_block)
    result.subblocks = {name: block for name, block in result.subblocks.items() if block in blocks}
    for block in blocks:
        for node in reversed(tuple(block.graph.nodes)):
            if node not in live and node.op != 'placeholder':
                block.graph.erase_node(node)
        block.graph.lint()
    used_modules = {node.target for node in live if node.op == 'call_module'}
    result.submodules = {name: value for name, value in result.submodules.items() if name in used_modules or name == 'get_index'}
    expressions = {node.args[0] for node in live if node.op == 'call_module' and node.target == 'get_index'}
    result.indexing_exprs = {name: expr for name, expr in result.indexing_exprs.items() if name in expressions}
    result.indexing_exprs_name = {expr: name for name, expr in result.indexing_exprs.items()}
    result.memory_usage, result.op_counts = defaultdict(list), Counter()
    kinds = {'load': MemoryUsageType.LOAD, 'load_seed': MemoryUsageType.LOAD_SEED,
             'store': MemoryUsageType.STORE, 'store_reduction': MemoryUsageType.STORE_REDUCTION}
    for node in live:
        if node.op != 'call_method':
            continue
        result.op_counts[node.target] += 1
        if node.target in kinds:
            expression = node.args[2]
            name = expression.args[0] if isinstance(expression, torch.fx.Node) else result.add_index_expr(expression, kinds[node.target])
            mode = node.args[4] if node.target == 'store' and len(node.args) > 4 else None
            result.memory_usage[kinds[node.target]].append(MemoryEntry(name, node.args[1], mode))
        elif node.target in ('index_expr', 'value_expr', 'check_bounds'):
            kind = MemoryUsageType.CHECK_BOUNDS if node.target == 'check_bounds' else MemoryUsageType.INDEX_EXPR
            for value in node.all_input_nodes:
                if value.op == 'call_module' and value.target == 'get_index':
                    result.memory_usage[kind].append(MemoryEntry(value.args[0], None, None))
    result.has_partial_accumulate = bool(result.op_counts['partial_accumulate'])
    return result


def compose_body_read(body, access, producer, coordinates, dtype):
    """Substitute a cached scalar producer at one of its declared reads.

    Coordinates come from the producer's write relation. Compiler graphs,
    masks and indirect-index shims are copied without evaluating source code.
    The original storage rounding and load arithmetic type remain explicit.
    """
    from torch._inductor.loop_body import LoopBody, LoopBodyBlock
    from torch._inductor.utils import sympy_subs, upcast_compute_type
    result, copied = clone_body(body)
    selected = copied[access['node']]
    replacements = dict(zip(producer.iter_vars, coordinates, strict=True))
    for var in producer.indirect_vars:
        replacements[var] = result.add_indirect(sympy_subs(producer.indirect_var_ranges[var], replacements))

    names = {}
    for name, shim in producer.submodules.items():
        if name == 'get_index':
            continue
        if shim.clone.func is LoopBody.bind_set_indirect_shim:
            # Upstream index analysis uses this established ordinal spelling.
            names[name] = f"set_indirect{result.indirect_vars.index(replacements[shim.clone.keywords['var']])}"
            result.submodules[names[name]] = None
        else:
            names[name] = result.add_submodule(None, name)
    for name, shim in producer.submodules.items():
        if name == 'get_index':
            continue
        arguments = dict(shim.clone.keywords)
        if shim.clone.func is LoopBody.bind_set_indirect_shim:
            arguments['var'] = replacements[arguments['var']]
            arguments['size'] = sympy_subs(arguments['size'], replacements)
        elif shim.clone.func is LoopBody.bind_masked_shim:
            arguments['name'] = names[arguments['name']]
        result.submodules[names[name]] = shim.clone.func(result, **arguments)

    def append(source, graph, ops, *, storing=False):
        mapped, returned = {}, None
        for node in source.graph.nodes:
            if node.op == 'placeholder':
                made = ops
            elif node.op == 'output':
                returned = torch.fx.node.map_arg(node.args[0], mapped.__getitem__)
                continue
            elif node.op == 'call_module' and node.target == 'get_index':
                made = body_index(result, graph, sympy_subs(producer.indexing_exprs[node.args[0]], replacements))
            elif storing and node.op == 'call_method' and node.target == 'store':
                made = graph.call_method('to_dtype', (ops, mapped[node.args[3]], dtype), {'use_compute_types': False})
                if not selected.meta.get('mesh_read_bits') and upcast_compute_type(dtype) != dtype:
                    made = graph.call_method('to_dtype', (ops, made, upcast_compute_type(dtype)))
            else:
                made = graph.node_copy(node, mapped.__getitem__)
                if made.op == 'call_module':
                    made.target = names[made.target]
                made.meta['mesh_source_node'] = node.meta.get('mesh_source_node', node)
            mapped[node] = made
        return returned

    for name, block in producer.subblocks.items():
        made = LoopBodyBlock.__new__(LoopBodyBlock)
        made.body, made.graph = result, torch.fx.Graph()
        ops = made.graph.placeholder('ops')
        made.graph.output(append(block, made.graph, ops))
        result.subblocks[names[name]] = made
    with selected.graph.inserting_before(selected):
        value = append(producer.root_block, selected.graph, selected.args[0], storing=True)
    selected.replace_all_uses_with(value)
    selected.graph.erase_node(selected)
    return finish_body(result, writes=True)


def cut_body_read(body, access, output, incoming, ordinal, owner):
    """Split one cached scalar load into a page producer and its continuation.

    This copies compiler graphs, including their index-producer shims. It never
    calls the source function or captures another scalar body. Implicit index
    dependencies participate in graph liveness alongside ordinary FX edges.
    """
    import copy
    from collections import defaultdict
    from torch._inductor.loop_body import LoopBodyBlock, MemoryUsageType
    from .operand import access_owner

    def cloned():
        result, nodes = clone_body(body)
        selected = next(value for old, value in nodes.items()
                        if old is access['node'] or old.meta.get('mesh_source_node') is access['node'])
        return result, selected

    prefix, selected = cloned()
    path = []
    def find(block):
        if any(node is selected for node in block.graph.nodes):
            return True
        for node in block.graph.nodes:
            if node.op == 'call_module' and node.target in prefix.subblocks and find(prefix.subblocks[node.target]):
                path.append((block, node))
                return True
        return False
    if not find(prefix.root_block):
        raise ValueError('the read does not belong to the retained scalar body')
    graph = selected.graph
    ops = next(node for node in graph.nodes if node.op == 'placeholder')
    masked = LoopBodyBlock.__new__(LoopBodyBlock)
    masked.body, masked.graph = prefix, torch.fx.Graph()
    mask_ops = masked.graph.placeholder('ops')
    loaded_index = masked.graph.call_module('get_index', (selected.args[2].args[0],))
    loaded = masked.graph.call_method('load', (mask_ops, access['name'], loaded_index))
    loaded.meta['mesh_read_bits'] = True
    masked.graph.output(loaded)
    name = prefix.add_submodule(None, 'masked_subblock')
    prefix.subblocks[name] = masked
    prefix.submodules[name] = prefix.bind_masked_shim(name)
    with graph.inserting_before(selected):
        declared_owner = access_owner(access, owner)
        if declared_owner in (True, False):
            predicate = graph.call_method('constant', (ops, bool(declared_owner), torch.bool))
        else:
            predicate = graph.call_method('index_expr', (ops, body_index(prefix, graph, declared_owner), torch.bool))
        value = graph.call_module(name, (predicate, 0))
    selected.replace_all_uses_with(value)
    graph.erase_node(selected)
    next(node for node in graph.nodes if node.op == 'output').args = (value,)
    for block, node in path:
        node.args = (node.args[0], 0)
        next(item for item in block.graph.nodes if item.op == 'output').args = (node,)
    graph = prefix.root_block.graph
    returned = next(node for node in graph.nodes if node.op == 'output')
    ops = next(node for node in graph.nodes if node.op == 'placeholder')
    with graph.inserting_before(returned):
        if declared_owner == False:
            # An empty owner contributes literal zero bits and has no source
            # or index read. Erase those dead scalar dependencies now, rather
            # than requiring unused argument pages in the prepared command.
            returned.args = (graph.call_method('constant', (ops, 0, access['dtype'])),)
        stored = graph.call_method('store', (ops, output, body_index(prefix, graph, ordinal), returned.args[0], None))
    returned.args = (stored,)
    prefix.iter_vars = [*body.iter_vars, *body.reduce_vars]
    prefix.reduce_vars = []
    prefix.sizes = ([*body.sizes[0], *body.sizes[1]], [])

    continuation, selected = cloned()
    with selected.graph.inserting_before(selected):
        replacement = body_index(continuation, selected.graph, ordinal)
    selected.args = (selected.args[0], incoming, replacement)
    prefix, continuation = finish_body(prefix), finish_body(continuation, writes=True)
    value_body = copy.copy(prefix)
    value_body.root_block = prefix.root_block.clone(value_body)
    value_body.root_block.graph = torch.fx.Graph()
    values = {}
    for node in prefix.root_block.graph.nodes:
        if node.op == 'call_method' and node.target == 'store':
            values[node] = values[node.args[3]]
        else:
            values[node] = value_body.root_block.graph.node_copy(node, values.__getitem__)
    value_body.subblocks = {name: block.clone(value_body) for name, block in prefix.subblocks.items()}
    value_body.submodules = {'get_index': value_body.get_index,
        **{name: shim.clone(value_body) for name, shim in prefix.submodules.items() if name != 'get_index'}}
    value_body.memory_usage = defaultdict(list, {kind: entries for kind, entries in prefix.memory_usage.items()
                                               if kind != MemoryUsageType.STORE})
    prefix.value_body = value_body
    return prefix, continuation


def prepare_graph(module, metadata, *, materializations=()):
    """Lower once to logical IR, before assigning physical page operands.

    Materializations name values that later consumers must read from storage.
    Declare those boundaries here; ``bind`` later chooses their actual pages
    and emits commands from this same graph.
    """
    import copy
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._inductor import config
    from torch._inductor.codegen.common import get_device_op_overrides
    from torch._inductor.codegen.mps import MetalScheduling, MetalKernel
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.choices import InductorChoices
    from torch._inductor.graph import GraphLowering
    from torch._inductor.scheduler import Scheduler, BaseSchedulerNode, SchedulerNode, NopKernelSchedulerNode
    from torch._inductor.virtualized import V
    from torch._inductor.debug import DebugContext
    from torch._inductor import ir
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from torch.fx.node import map_arg
    from .placement import Placements, same_view
    from .compilation import implementations
    from contextlib import contextmanager
    from mesh_compute import hardware

    placements, bound_storage, views, storage_layouts = Placements(), {}, [], {}
    assigned_results = ()
    external_arguments, external_outputs, command_accesses = [], [], []
    materialized_buffers = set()
    configuration = compiler_operands()
    implementation_catalog = copy.deepcopy(implementations())

    class NativeChoices(InductorChoices):
        def reduction_split_factor(self, device, reduction_numel_hint, numel_hint, inner_reduction):
            # Split the numerical IR, not execution. Both stages are ordinary
            # page producers; the recorder retains their data dependencies.
            reduction_groups = hardware()['gpu']['cores'] * configuration['REDUCTION_GROUPS_PER_CORE']
            tiles = (reduction_numel_hint + configuration['REDUCTION_TILE_ELEMENTS'] - 1) // configuration['REDUCTION_TILE_ELEMENTS']
            rows = max(1, numel_hint)
            groups = (reduction_groups + rows - 1) // rows
            return max(1, min(tiles, groups))

    class LogicalChoices(NativeChoices):
        def reduction_split_factor(self, device, reduction_numel_hint, numel_hint, inner_reduction):
            # Assignment consumes the original reduction fiber. Backend work
            # splitting belongs to preparation of the selected implementation.
            return 1

        def reduction_unroll_threshold(self):
            return 0

    class PageKernel(MetalKernel):
        # These objects exist during source emission only. The numerical body
        # receives page bindings and the resulting ordinary address expression.
        max_threadgroup_size = configuration['ROW_THREADS']
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
            if name is not None:
                name = V.graph.scheduler.mutation_real_name.get(name, name)
            if name is not None and name not in storage_layouts:
                raise ValueError(f'numerical access {name} has no assigned page operand')
            if name is not None:
                bindings = self.__dict__.setdefault('page_bindings', {})
                slot = bindings.setdefault(name, len(bindings))
                return f'mesh_storage_{slot}({expression})'
            return expression

        def load(self, name, index):
            with self.access(name):
                if getattr(V.interpreter, 'current_node', None) is not None and V.interpreter.current_node.meta.get('mesh_read_bits'):
                    # This declared value is transported, not evaluated in a
                    # wider arithmetic type. In particular a half/bfloat load
                    # must preserve NaN payloads through its fixed-width port.
                    pointer = self.args.input(name)
                    index = self.prepare_indexing(index)
                    return self.cse.generate(self.loads, f'{pointer}[{self.index_to_str(index)}]', dtype=V.graph.get_dtype(name))
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
            binding = storage_layouts[self.data.get_name()]
            original, unit, physical = (binding[key] for key in ('source', 'source_dtype', 'operand'))
            mapped = rebase_typed(ir_layout(self.layout), self.layout.dtype.itemsize,
                                  original, unit.itemsize, physical.layout)
            operand = TensorView(physical.storage_as(self.layout.dtype), mapped)
            index = len(views)
            views.append(operand)
            return f'mesh_views[{index}]'

    def page_view(storage, layout):
        return ir.TensorBox(PageView(data=storage, layout=layout))

    def ir_layout(layout):
        return _stream.TensorLayout(tuple(map(int, layout.size)), tuple(map(int, layout.stride)), int(layout.offset))

    def ir_operand(value):
        if value is None:
            return None
        storage, layout = ir.as_storage_and_layout(value, freeze=False)
        name = storage.get_name()
        name = V.graph.storage_aliases.get(name, name)
        storage = V.graph.get_buffer(name)
        return dict(name=name, dtype=layout.dtype, storage_dtype=storage.get_dtype(),
                    storage_bytes=storage.get_layout().storage_size() * storage.get_dtype().itemsize,
                    view=ir_layout(layout))

    def external_domains(operation, algebra):
        """Bind shared mathematical declarations to the retained IR's views."""
        import json
        import mesh_matrix
        from .operand import binding_layout, layout_configuration

        operand = ir_operand

        if isinstance(operation, ir.MultiOutput):
            # Selecting a declared result from an external call introduces no
            # numerical work or new storage. The producer already writes this
            # named result; preserve its exact typed view through projection.
            value = operand(operation)
            work = tuple(value['view'].shape)
            use = dict(value, relation=algebra.layout(layout_configuration(value['view'])).reshape(work))
            return (dict(kind='alias', work=work, reads={0: use}, writes={0: use}, reduction=None,
                         finish=None, source=operation.inputs[0], path=tuple(operation.indices)),)

        def atomic(reads, writes):
            # A registered command is an indivisible work item until its
            # numerical declaration supplies finer fibers. These are its
            # actual typed argument views, not scheduler StarDep envelopes.
            work = (int(any(value['view'].count for value in writes.values())),)
            def accesses(values):
                return {slot: dict(value, relation=algebra.map(work, (value['view'].count,),
                    lambda x, y: []).then(algebra.layout(layout_configuration(value['view']))))
                    for slot, value in values.items()}
            return (dict(kind='atomic', work=work, reads=accesses(reads), writes=accesses(writes),
                         reduction=None, finish=None),)

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
        if adapter not in ('linear', 'linears', 'kernel', 'topk', 'sort', 'cumsum', 'draw', 'retain', 'fwht_stage'):
            return ()
        args, keywords = operation.unflatten_args(operation.inputs, operation.constant_args)
        named = {field.name: args[i] if i < len(args) else keywords.get(field.name, field.default_value)
                 for i, field in enumerate(operation.op_overload._schema.arguments)}
        outputs = tree_flatten(operation.codegen_outputs())[0]
        if adapter == 'fwht_stage':
            from .structured import factor_domain
            return (factor_domain(algebra, operand(named['x']), operand(named['signs']), operand(outputs[0]),
                **{key: named[key] for key in ('width', 'step', 'normalization', 'signed', 'sources', 'starts')}),)
        if adapter in ('draw', 'retain'):
            from .selection import row_domain
            inputs = {role: operand(named[role]) for role in (('z', 'keys', 'cutoff') if adapter == 'draw' else ('z', 'keys'))}
            roles = ('ids',) if adapter == 'draw' else ('ids', 'kept', 'log_z')
            parameters = {name: named[name] for name in (('seed', 'first') if adapter == 'draw' else ('seed', 'head', 'tail'))}
            width = 1 if adapter == 'draw' else named['head'] + named['tail']
            return (row_domain(algebra, inputs, dict(zip(roles, map(operand, outputs), strict=True)),
                               width=width, adapter=adapter, parameters=parameters),)
        if adapter in ('topk', 'sort', 'cumsum'):
            from .selection import row_domain
            source = operand(named['self'])
            def rows(value):
                value = dict(value)
                shape = tuple(value['view'].shape) or (1,)
                axis = named.get('dim', -1) % len(shape)
                axes = (*[d for d in range(len(shape)) if d != axis], axis)
                value['view'] = value['view'].reshape(shape).permute(axes).reshape(
                    (math.prod(shape[d] for d in axes[:-1]), shape[axis]))
                return value
            source = rows(source)
            targets = dict(zip(('out',) if adapter == 'cumsum' else ('values', 'indices'),
                               (rows(operand(value)) for value in outputs), strict=True))
            width = next(iter(targets.values()))['view'].shape[-1]
            parameters = (dict(dim=1, dtype=targets['out']['dtype']) if adapter == 'cumsum' else
                          dict(k=width, dim=1, largest=named.get('largest', named.get('descending', False)),
                               sorted=named.get('sorted', True)))
            return (row_domain(algebra, dict(x=source), targets, width=width,
                               adapter='cumsum' if adapter == 'cumsum' else 'topk',
                               parameters=parameters, prefix=adapter == 'cumsum'),)
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
        if result:
            return tuple(result)
        return atomic({slot: uses[slot] for slot in named['input_slots']},
                      {slot: uses[slot] for slots in named['output_slots'] for slot in slots})

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
                name = V.graph.storage_aliases.get(item['name'], item['name'])
                dtype = V.graph.get_dtype(item['name'])
                storage = V.graph.get_buffer(name)
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
                result[slot] = dict(item, name=name, dtype=dtype, storage_dtype=storage.get_dtype(), variables=axes,
                                    storage_bytes=storage.get_layout().storage_size() * storage.get_dtype().itemsize,
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
            return [f'mesh_results[{i}]' for i in range(len(assigned_results))]

        def codegen_reinterpret_view(self, data, size, stride, offset, writeline, dtype=None):
            if data.get_name() not in storage_layouts:
                return super().codegen_reinterpret_view(data, size, stride, offset, writeline, dtype)
            binding = storage_layouts[data.get_name()]
            original, unit, physical = (binding[key] for key in ('source', 'source_dtype', 'operand'))
            dtype = data.get_dtype() if dtype is None else dtype
            described = _stream.TensorLayout(tuple(map(int, size)), tuple(map(int, stride)), int(offset))
            mapped = rebase_typed(described, dtype.itemsize, original, unit.itemsize, physical.layout)
            with unset_fake_temporarily():
                views.append(TensorView(physical.storage_as(dtype), mapped))
            return f'mesh_views[{len(views) - 1}]'

        def make_allocation(self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False):
            if name in bound_storage:
                return f'{name} = mesh_destinations[{name!r}]'
            raise ValueError(f'numerical storage {name} has no assigned page operand')

        def can_reuse(self, input_buffer, output_buffer=None):
            # Logical landing declarations constrain reuse before pages exist.
            outputs = (output_buffer.get_buffer_names() if isinstance(output_buffer, BaseSchedulerNode)
                       else () if output_buffer is None else (output_buffer.get_name(),))
            if any(name in V.graph.never_reuse_buffers or name in bound_storage for name in outputs):
                return False
            return super().can_reuse(input_buffer, output_buffer)

        def selected_external(self, node):
            selected = getattr(self, 'current_external_plans', None)
            if selected is None:
                return False
            for plan in selected:
                external_arguments.append(plan)
                self.writeline(f'mesh_native.external(**mesh_externals[{len(external_arguments) - 1}])')
            self.bound_external_results(node)
            return True

        def bound_external_results(self, node):
            from torch.utils._pytree import tree_map
            declared = node.codegen_outputs() if isinstance(node, ir.FallbackKernel) else node
            outputs = tree_map(lambda leaf: bound_storage[leaf.get_name()] if isinstance(leaf, ir.IRNode) else leaf,
                               declared)
            external_outputs.append(outputs)
            self.writeline(f'{node.get_name()} = mesh_external_outputs[{len(external_outputs) - 1}]')

        def _generate_extern_kernel_alloc_helper(self, node, args):
            names = _ALLOC_ADAPTERS
            name = node.get_kernel_name()
            if name not in names:
                raise NotImplementedError(f'no native preparation for {name}; a framework fallback is not an executor')
            outputs = node.codegen_outputs()
            output_refs = [self.codegen_reinterpret_view(leaf, leaf.get_size(), leaf.get_stride(),
                           leaf.get_layout().offset, False) if isinstance(leaf, ir.IRNode)
                           and leaf.get_name() in storage_layouts else 'None' for leaf in tree_flatten(outputs)[0]]
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
            for child in node.get_nodes():
                if child.get_name() in self.scheduler.removed_ops:
                    child.mark_run()
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
                if name in kernel.removed_buffers:
                    continue
                binding = storage_layouts[name]
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
                            argument_name = V.graph.scheduler.mutation_real_name.get(item['name'], item['name'])
                            if argument_name in arguments:
                                accesses.append(dict(side=side, slot=arguments.index(argument_name), access=item, work=work,
                                                     origins=tuple(node.node.get_origins())))
            index = len(command_accesses)
            command_accesses.append(tuple(accesses))
            wrapper.header.writeline(f'{name} = mesh_native.shader({source_text!r}, "generated_kernel", {access}, {tuple(arguments)!r}, {types!r}, mesh_accesses[{index}])')
            return name

    scalar_compilers = {'metal_scalar': (MetalSource, 'mps')}

    class DeclaredSchedulerNode(SchedulerNode):
        """Start backend scheduling from the retained, assigned scalar body."""
        def __init__(self, scheduler, operation, declaration):
            self.declaration = declaration
            super().__init__(scheduler, operation)

        def _compute_attrs(self, *args, **kwargs):
            declaration = self.__dict__.pop('declaration', None)
            if declaration is None:
                return super()._compute_attrs(*args, **kwargs)
            self._body, self._sizes = declaration._body, declaration._sizes
            device = self.node.get_device_or_error()
            self.group = device, self.scheduler.get_backend(device).group_fn(self._sizes)
            self.set_read_writes(declaration.read_writes)

    class NativeScheduler(Scheduler):
        def codegen_extern_call(self, node):
            if getattr(node, 'page_atomic', None) is False:
                node.mark_run()
                V.graph.wrapper_code.bound_external_results(node.node)
                self.free_buffers()
                return
            if not hasattr(node, 'page_plans'):
                return super().codegen_extern_call(node)
            # This is the ordinary external declaration's preparation boundary.
            # Its selected views are already bound, so no full-view references
            # are reconstructed before invoking the shared product declaration.
            from torch._inductor.codegen.common import Kernel
            with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
                node.decide_inplace_update()
                node.mark_run()
            wrapper = V.graph.wrapper_code
            wrapper.current_external_plans = node.page_plans
            try:
                wrapper.selected_external(node.node)
            finally:
                wrapper.current_external_plans = None
            self.free_buffers()

        def create_scheduler_node(self, node):
            prepared = getattr(self, 'prepared_nodes', None)
            if prepared is None:
                declaration = V.graph.declaration_nodes[node.get_operation_name()]
                result = (DeclaredSchedulerNode(self, node, declaration) if declaration._body is not None
                          else super().create_scheduler_node(node))
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
            # The selected compiler owns code generation. A source tensor's
            # storage device is not a numerical implementation assignment.
            if V.graph.selected_implementations:
                compiler = V.graph.selected_implementations[0]['implementation']['compiler']
                return scalar_compilers[compiler][0](self)
            # Externals already name their current native numerical recipe.
            return MetalSource(self)

    class NativeSection:
        """A compiler operation slice with the same retained declaration bodies."""
        def __init__(self, graph, operations, inputs, outputs, nodes, algebra):
            self.graph, self.operations = graph, tuple(operations)
            self.graph_inputs_original, self.outputs = dict(inputs), tuple(outputs)
            self.graph_inputs = {name: ir.TensorBox.create(value) for name, value in inputs.items()}
            self.nodes = {op.get_operation_name(): nodes[op.get_operation_name()] for op in operations}
            self.algebra, self.declared_effects, self.selected_domains = algebra, None, None
            self.scheduler = None
            self.pending_declarations = ()
            self.source_values = graph.source_values

        @contextmanager
        def active(self):
            graph = self.graph
            names = ('operations', 'graph_inputs', 'graph_inputs_original', 'graph_input_names',
                     'graph_outputs', 'example_inputs', 'wrapper_code', 'removed_operations',
                     'removed_buffers', 'inplaced_to_remove', 'declared_effects', 'selected_domains',
                     'pending_declarations', 'declaration_nodes', 'scheduler', 'selected_implementations', 'reduction_uses')
            absent = object()
            saved = {name: getattr(graph, name, absent) for name in names}
            with environment(), V.set_graph_handler(graph), V.set_extern_kernel_nodes([]), mode:
                try:
                    graph.operations = list(self.operations)
                    graph.graph_inputs = dict(self.graph_inputs)
                    graph.graph_inputs_original, graph.graph_input_names = dict(self.graph_inputs_original), list(self.graph_inputs)
                    graph.graph_outputs, graph.example_inputs = list(self.outputs), None
                    graph.wrapper_code = None
                    for name in ('removed_operations', 'removed_buffers', 'inplaced_to_remove'):
                        setattr(graph, name, type(saved[name])())
                    graph.declared_effects = self.declared_effects
                    graph.selected_domains = self.selected_domains
                    graph.pending_declarations = self.pending_declarations
                    graph.declaration_nodes, graph.scheduler = self.nodes, self.scheduler
                    graph.selected_implementations = getattr(self, 'selected_implementations', None)
                    graph.reduction_uses = getattr(self, 'reduction_uses', ())
                    yield graph
                finally:
                    self.scheduler = graph.scheduler
                    self.selected_implementations = graph.selected_implementations
                    self.reduction_uses = graph.reduction_uses
                    self.pending_declarations = graph.pending_declarations
                    for name, value in saved.items():
                        if value is absent:
                            graph.__dict__.pop(name, None)
                        else:
                            setattr(graph, name, value)

        def effects(self, algebra=None):
            if self.declared_effects is None:
                with self.active() as graph:
                    self.declared_effects = graph.effects(self.algebra if algebra is None else algebra)
            return self.declared_effects

        def value_binding(self, value, algebra):
            return self.graph.value_binding(value, algebra)

        def select_domains(self, records):
            self.effects()
            with self.active() as graph:
                graph.select_domains(records)
                self.selected_domains = tuple(records)
            return self

        def consumer_uses(self):
            with self.active() as graph:
                return graph.consumer_uses()

        def page_aliases(self):
            with self.active() as graph:
                return graph.page_aliases()

        def emit(self, operands, *, storage, results, pages, uses):
            with self.active() as graph:
                # The ordinary binder consumes the same named storage records
                # for original operands, carried values and received ports.
                bound = [storage[name]['operand'] for name in self.graph_inputs]
                return graph.emit(bound, storage=storage, results=results, pages=pages, uses=uses)

        def bind(self, operands, **placement):
            return Program.prepare(self.emit(operands, **placement))

    class NativeGraph(GraphLowering):
        def declaration(self, operation, body=None):
            """Keep original scalar coordinates without choosing a backend."""
            from types import SimpleNamespace
            from torch._inductor import dependencies
            if body is None and isinstance(operation, ir.ComputedBuffer) and not operation.is_no_op():
                sizes, original, variables = operation.get_default_sizes_body()
                body, _ = clone_body(original)
                # The default cache retains literal-zero argument positions
                # for singleton tensor axes. Its returned variable/range
                # lists are the corresponding independent work coordinates.
                body.sizes = tuple(map(tuple, sizes))
                body.iter_vars, body.reduce_vars = variables
                body = finish_body(body, writes=True)
            sizes = None if body is None else body.sizes
            reads_writes = (operation.get_read_writes() if body is None else
                           dependencies.extract_read_writes(body, *sizes, normalize=False))
            return SimpleNamespace(node=operation, _body=body, _sizes=sizes, read_writes=reads_writes)

        def retain_declarations(self):
            self.declaration_nodes = {operation.get_operation_name(): self.declaration(operation)
                                      for operation in self.operations}
            # These are the IR's declared storage aliases. The selected
            # scheduler still constructs versioned ordering dependencies.
            self.storage_aliases = {}
            for operation in self.operations:
                for output in operation.get_outputs():
                    for source in output.get_mutation_names():
                        self.storage_aliases[output.get_name()] = self.storage_aliases.get(source, source)

        def compose_reads(self):
            """Fold live reads of cached pointwise values into their consumers.

            Explicit materializations remain page boundaries. Reductions and
            external commands keep their own declarations. Pure scalar values
            use their exact write inverse; no intermediate table is allocated
            solely because an index into that value changes at execution.
            """
            from mesh_relation import Algebra, flat
            from .layout import strides
            from .operand import access_domain, byte_relation, layout_configuration, symbolic_index
            algebra = Algebra()
            nodes = self.declaration_nodes
            producers = {value.get_name(): op for op in self.operations for value in op.get_outputs()}
            protected = self.never_reuse_buffers | {value.get_name() for value in self.graph_outputs}
            substituted = set()
            for operation in self.operations:
                node = nodes[operation.get_operation_name()]
                if node._body is None:
                    continue
                while True:
                    domain, = computed_domains(operation, node._body, algebra)
                    changed = False
                    for access in domain['reads'].values():
                        producer = producers.get(access['name'])
                        if (producer is None or producer is operation or access['name'] in protected or
                            access['relation'] is not None or access['index'] is None or
                            not isinstance(producer, ir.ComputedBuffer) or not isinstance(producer.data, ir.Pointwise)):
                            continue
                        source = nodes[producer.get_operation_name()]
                        if source._body is None or source._body.reduce_vars:
                            continue
                        source_domain, = computed_domains(producer, source._body, algebra)
                        if len(source_domain['writes']) != 1:
                            continue
                        write, = source_domain['writes'].values()
                        if (write['name'] != access['name'] or write['relation'] is None or write['predicates'] or
                            write['node'].target != 'store' or write['node'].args[4] is not None):
                            continue
                        inverse = write['relation'].inverse((int(write['storage_bytes']) // write['dtype'].itemsize,))
                        if not inverse.single_valued():
                            continue
                        work = algebra.map(domain['work'], (), lambda x, y: []).domain()
                        read = access_domain(access, work)['positions']
                        written = byte_relation(write).image(write['relation'].domain())
                        if read is None or not read.binary(written, 'subtract').empty():
                            continue
                        shape = source_domain['work']
                        flatten = algebra.map(shape, (math.prod(shape),),
                            lambda x, y: [f'{y[0]} = {flat(shape, x)}'])
                        ordinal = symbolic_index(inverse.then(flatten).terms(), access['index'])
                        coordinates = layout_configuration(_stream.TensorLayout(shape, strides(shape)))['index_coordinates']
                        body = compose_body_read(node._body, access, source._body,
                            [symbolic_index(term, ordinal) for term in coordinates], access['dtype'])
                        node = nodes[operation.get_operation_name()] = self.declaration(operation, body)
                        operation.origins = operation.get_origins() | producer.get_origins()
                        substituted.add(producer.get_operation_name())
                        changed = True
                        break
                    if not changed:
                        break
            if not substituted:
                return self
            retained = list(self.operations)
            for operation in reversed(self.operations):
                if operation.get_operation_name() not in substituted:
                    continue
                live = protected | {dep.name for other in retained if other is not operation
                                    for dep in nodes[other.get_operation_name()].read_writes.reads}
                if not any(value.get_name() in live for value in operation.get_outputs()):
                    retained.remove(operation)
            self.operations = retained
            self.declaration_nodes = {op.get_operation_name(): nodes[op.get_operation_name()] for op in retained}
            return self

        def cut_reads(self, requests):
            """Expose ordered page-result cuts in this retained scalar graph.

            Requests carry immutable work and byte-owner operands. The result
            contains ordinary numerical sections over this graph and explicit
            output/input ports between them. No possible table envelope becomes
            a transfer request, and no collective is chosen by this compiler.
            """
            import copy
            import sympy
            from torch._inductor import dependencies
            from torch._inductor.loop_body import LoopBody
            from mesh_relation import flat
            from .layout import strides
            from .operand import iteration_order, symbolic_index, layout_configuration
            requests = tuple(request for request in requests if not request['work'].empty())
            if not requests:
                return dict(sections=(self,), boundaries=())
            algebra = requests[0]['work'].algebra
            with environment(), V.set_graph_handler(self), V.set_extern_kernel_nodes([]), mode:
                original_operations = tuple(self.operations)
                original_inputs, original_outputs = dict(self.graph_inputs), tuple(self.graph_outputs)
                nodes = {operation.get_operation_name(): copy.copy(self.declaration_nodes[operation.get_operation_name()])
                         for operation in original_operations}
                order = {operation.get_operation_name(): index for index, operation in enumerate(original_operations)}
                read_order = {operation.get_operation_name(): {id(access['node']): index
                    for index, access in enumerate(body_accesses(nodes[operation.get_operation_name()]._body)['reads'])}
                    for operation in original_operations if nodes[operation.get_operation_name()]._body is not None}
                requests = sorted(enumerate(requests), key=lambda pair: (
                    order[pair[1]['effect']['operation'].get_operation_name()],
                    read_order[pair[1]['effect']['operation'].get_operation_name()][id(pair[1]['access']['node'])]))
                stages, boundaries, cursor, pending = [], [], 0, []
                fields, ports = torch.fx.Graph(), {}

                def port(value):
                    import json
                    name = value.get_name()
                    binding = self.value_binding(value, algebra)
                    key = name, binding['dtype'], json.dumps(layout_configuration(binding['view']), sort_keys=True)
                    if key in ports:
                        return ports[key]
                    field = fields.placeholder(name)
                    field.meta['val'] = torch.empty_strided(tuple(value.get_size()), tuple(value.get_stride()),
                                                           dtype=value.get_dtype(), device=value.get_device())
                    ports[key] = dict(field=(field, ()), binding=binding, value=value)
                    return ports[key]

                for _, request in requests:
                    operation, access, work = request['effect']['operation'], request['access'], request['work']
                    at = order[operation.get_operation_name()]
                    pending.extend(original_operations[cursor:at])
                    cursor = at
                    template = nodes[operation.get_operation_name()]
                    body = template._body
                    order_relation, _ = iteration_order(work, request['domain']['work'])
                    count = order_relation.shape[0]
                    flatten = algebra.map(request['domain']['work'], (math.prod(request['domain']['work']),),
                        lambda x, y: [f'{y[0]} = {flat(request["domain"]["work"], x)}'])
                    work_ordinals = order_relation.then(flatten.inverse((math.prod(request['domain']['work']),)))
                    flat_index = sum(var * math.prod(request['domain']['work'][axis + 1:])
                                     for axis, var in enumerate(request['domain']['variables']))
                    ordinal = symbolic_index(order_relation.inverse((math.prod(request['domain']['work']),)).terms(), flat_index)
                    device, dtype = operation.get_device_or_error(), access['dtype']
                    layout = ir.FixedLayout(device, dtype, (sympy.Integer(count),), (sympy.Integer(1),))
                    # The ordinary Pointwise owns the cached value body. Its
                    # scalar callable is installed below from that same body;
                    # the source function is not invoked during construction.
                    produced = ir.ComputedBuffer(name=None, layout=layout, data=None)
                    self.register_buffer(produced, set_name=True)
                    # Different rank-local cuts retain distinct compiler
                    # operations even though each activates the same source
                    # graph's operation slice.
                    produced.operation_name = self.qualify_name(f'op{len(self.name_to_op)}')
                    self.name_to_op[produced.operation_name] = produced
                    incoming = ir.InputBuffer(name=f'{produced.get_name()}_received', layout=layout)
                    self.name_to_buffer[incoming.get_name()] = incoming
                    prefix, continuation = cut_body_read(body, access, produced.get_name(), incoming.get_name(), ordinal, request['owner'])
                    # A prefix has one output per *assigned* work point. Its
                    # compact loop enumerates precisely that supplied set,
                    # including disconnected selections and reduction terms.
                    variables, ranges = dependencies.index_vars_no_squeeze((count,), (), prefix='c')
                    logical = symbolic_index(order_relation.terms(), variables[0][0])
                    coordinates = layout_configuration(_stream.TensorLayout(request['domain']['work'], strides(request['domain']['work'])))['index_coordinates']
                    arguments = ([symbolic_index(term, logical) for term in coordinates], [])
                    value_body = LoopBody(prefix.value_body, arguments, ranges, *variables, allow_same_symbol_in_index=True)
                    prefix = LoopBody(prefix, arguments, ranges, *variables, allow_same_symbol_in_index=True)
                    # The body is already cached at the scheduler's normal
                    # compiler boundary. Keep that cache on the new ordinary IR
                    # buffer so all subsequent compiler queries use it.
                    setattr(produced, '__get_default_sizes_body_cache', (prefix.sizes, prefix, (prefix.iter_vars, prefix.reduce_vars)))
                    produced.data = ir.Pointwise(device=device, dtype=dtype,
                        inner_fn=lambda indices, body=value_body: body(indices), ranges=(count,))
                    produced.origins = operation.get_origins()
                    nodes[produced.get_operation_name()] = self.declaration(produced, prefix)
                    nodes[operation.get_operation_name()] = self.declaration(operation, continuation)
                    pending.append(produced)
                    stages.append(tuple(pending)); pending = []
                    output_port, input_port = port(produced), port(incoming)
                    boundaries.append(dict(key=request['key'], access=access, work=work, owner=request['owner'],
                        source_order=(at,read_order[operation.get_operation_name()][id(access['node'])]),
                        output=output_port, input=input_port, ordinals=work_ordinals,
                        before=len(stages) - 1, after=len(stages)))
                stages.append((*pending, *original_operations[cursor:]))
                self.operations = list(original_operations)
                # Ordinary intermediate values crossing a section cut keep the
                # same named storage; only the transferred read result has a
                # distinct receiver port.
                received = {item['input']['value'].get_name(): item['input']['value'] for item in boundaries}
                buffers = {**original_inputs, **self.name_to_buffer, **received}
                produced_at = {buffer.get_name(): at for at, stage in enumerate(stages)
                               for operation in stage for buffer in operation.get_outputs()}
                read_names = [{dep.name for operation in stage for dep in nodes[operation.get_operation_name()].read_writes.reads}
                              for stage in stages]
                final_names = {value.get_name() for value in original_outputs}
                sections = []
                for at, stage in enumerate(stages):
                    own = {buffer.get_name() for operation in stage for buffer in operation.get_outputs()}
                    required_inputs = read_names[at] | (final_names if at == len(stages) - 1 else set())
                    inputs = {name: value if isinstance(value, ir.InputBuffer) else ir.InputBuffer(name=name, layout=value.get_layout())
                              for name, value in buffers.items() if name in required_inputs - own}
                    needed = set().union(*read_names[at + 1:], final_names)
                    outputs = [buffers[name] for name in produced_at if produced_at[name] == at and name in needed]
                    outputs.extend(item['output']['value'] for item in boundaries if item['before'] == at)
                    if at == len(stages) - 1:
                        outputs = list(original_outputs)
                    section = NativeSection(self, stage, inputs, outputs, nodes, algebra)
                    section.input_ports = {name: port(value) for name, value in inputs.items()}
                    section.output_ports = tuple(port(value) for value in outputs)
                    sections.append(section)
                return dict(sections=tuple(sections), boundaries=tuple(boundaries))

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
                view = page_view(storage, layout)
                materialized_buffers.add(storage.get_name())
                leaves[i] = view
            result = tree_unflatten(leaves, spec)
            # Inductor mutates the live TensorBox while lowering an in-place
            # write. Input page declarations retain its original InputBuffer,
            # which Inductor already stores separately for writeback binding.
            self.source_values[node] = (self.graph_inputs_original.get(self.qualify_name(str(node.target)), result)
                                        if node.op == 'placeholder' else result)
            return result

        def value_binding(self, value, algebra):
            """A realized source IR value's exact view; never realize it here."""
            from .operand import layout_configuration
            with environment(), V.set_graph_handler(self), mode:
                binding = ir_operand(value)
                return dict(binding, value=value, relation=algebra.layout(layout_configuration(binding['view'])))

        def loop_body(self, operation, bodies=None):
            """Original declared work coordinates, independent of scheduling."""
            if bodies is not None:
                return bodies.get(operation.get_operation_name())
            return self.declaration_nodes[operation.get_operation_name()]._body

        def consumer_constraints(self, bodies=None):
            """Typed backend eligibility, separate from storage interpretation."""
            from mesh_compute import eligibility
            declarations = []
            for operation in self.operations:
                body = self.loop_body(operation, bodies)
                if body is None:
                    continue
                required = {operation.get_dtype()}
                for block in (body.root_block, *body.subblocks.values()):
                    for node in block.graph.nodes:
                        required.update(value for value in tree_flatten((node.args, node.kwargs))[0] if isinstance(value, torch.dtype))
                accesses = body_accesses(body)
                for access in (*accesses['reads'], *accesses['writes']):
                    required.add(self.get_dtype(access['name']))
                requirement = dict(interfaces=['scalar'], input_types=[],
                    arithmetic_types=sorted(str(dtype).removeprefix('torch.') for dtype in required), constants=[])
                declarations.append(dict(operation=operation, required=requirement,
                    candidates=tuple(eligibility(requirement, candidate)
                                     for candidate in implementation_catalog.values())))
            return tuple(declarations)

        def select_implementations(self, records=None):
            """Resolve explicit choices against the retained typed requirements."""
            from mesh_compute import select_implementation
            choices = []
            for declaration in self.consumer_constraints():
                operation = declaration['operation']
                requests = ((None,) if records is None else tuple(record for record in records
                    if record['effect']['operation'] is operation and
                    (not record['work'].empty() or record['finish'] is not None and not record['finish']['work'].empty())))
                for record in requests:
                    implementation = select_implementation(declaration['required'], implementation_catalog.values(),
                        None if record is None else record.get('implementation'))
                    if implementation['compiler'] not in scalar_compilers:
                        raise NotImplementedError(f"retained scalar emission has no compiler for {implementation['compiler']}")
                    choices.append(dict(operation=operation, domain=None if record is None else record['domain'],
                        work=None if record is None else record['work'],
                        required=declaration['required'], implementation=implementation))
            self.selected_implementations = tuple(choices)

        def split_reduction(self, node):
            """Apply the selected compiler's existing split to a cached body."""
            import copy
            import sympy
            from torch._inductor.utils import reduction_num_outputs
            from mesh_relation import Algebra
            from .operand import layout_configuration
            operation, body = node.node, node._body
            if not body.reduce_vars or not isinstance(operation.data, ir.Reduction):
                return (), node
            value, _ = clone_body(body)
            reduction = next(item for item in value.root_block.graph.nodes
                             if item.op == 'call_method' and item.target == 'reduction')
            payload = reduction.args[4]
            next(item for item in value.root_block.graph.nodes if item.op == 'output').args = (payload,)
            value = finish_body(value)
            def inner(indices, reductions):
                return value(indices, reductions, allow_same_symbol_in_index=True)
            data = operation.data
            ranges, reduction_ranges = map(list, body.sizes)
            hint, split = ir.Reduction.num_splits(data.device, data.dtype, data.src_dtype, inner,
                ranges, reduction_ranges, data.reduction_type, sympy.prod(reduction_ranges))
            if split <= 1:
                return (), node
            # GraphLowering assigns operation names from its registration
            # sequence. A selected section is only a slice of that sequence;
            # retain the full existing declarations while adding stages.
            operations = self.operations
            self.operations = list(self.name_to_op.values())
            start = len(self.operations)
            try:
                if isinstance(data, ir.WelfordReduction):
                    values = payload if isinstance(payload, (tuple, list)) else (payload,)
                    loaders = tuple((lambda i, r, at=at: inner(i, r)[at]) for at in range(len(values))) if isinstance(payload, (tuple, list)) else (inner,)
                    made = type(data).create_multilayer(data.device, data.dtype, loaders, ranges,
                        reduction_ranges, data.reduction_type, split, hint)[data.output_index]
                elif isinstance(data, ir.MultiOutputReduction):
                    made = type(data).create_multilayer(data.device, data.dtype, data.src_dtype, inner,
                        ranges, reduction_ranges, reduction_num_outputs(data.reduction_type), split, hint)[data.output_index]
                else:
                    made = type(data).create_multilayer(data.device, data.dtype, data.src_dtype, inner,
                        ranges, reduction_ranges, data.reduction_type, split, hint)
                while isinstance(made, (ir.TensorBox, ir.StorageBox)):
                    made = made.data
                for stage in self.operations[start:]:
                    stage.decide_layout()
                final = copy.copy(operation)
                final.data = made.data if isinstance(made, ir.ComputedBuffer) else made
                # The backend's final fiber keeps the original selected store
                # relation rather than publishing a dense result temporary.
                layout = ir.FixedLayout(data.device, data.dtype, ranges, ir.FlexibleLayout.contiguous_strides(ranges))
                described = ir.ComputedBuffer(name=operation.get_name(), layout=layout, data=final.data)
                _, final_body, _ = described.get_default_sizes_body()
                final_body, _ = clone_body(final_body)
                original_write, = body_accesses(body)['writes']
                substitutions = dict(zip(body.iter_vars, final_body.iter_vars, strict=True))
                target = original_write['index'].xreplace(substitutions)
                write, = body_accesses(final_body)['writes']
                with write['node'].graph.inserting_before(write['node']):
                    index = body_index(final_body, write['node'].graph, target)
                write['node'].args = (write['node'].args[0], original_write['name'], index, *write['node'].args[3:])
                final_body = finish_body(final_body, writes=True)
                # Other ordinary IR queries (free symbols and dependencies)
                # use the same retained store, including singleton axes and
                # compact selections whose shape differs from the storage.
                final.get_store_function = lambda: lambda *indices: final_body(*indices, allow_same_symbol_in_index=True)
                setattr(final, '__get_default_sizes_body_cache',
                    (final_body.sizes, final_body, (final_body.iter_vars, final_body.reduce_vars)))
                stages = tuple(stage for stage in self.operations[start:] if stage is not made)
            finally:
                self.operations = operations
            physical = []
            algebra = Algebra()
            for stage in stages:
                stage.origins = operation.get_origins()
                declaration = self.declaration(stage)
                prepared = DeclaredSchedulerNode(self.scheduler, stage, declaration)
                prepared.declared_read_writes = declaration.read_writes
                physical.append(prepared)
                binding = self.value_binding(stage, algebra)
                relation = algebra.layout(layout_configuration(binding['view']))
                self.reduction_uses.append(dict(key=('reduction', stage.get_operation_name()),
                    binding=binding, relation=relation, shape=relation.shape, requirements=(), access='readwrite'))
            final_node = DeclaredSchedulerNode(self.scheduler, final, self.declaration(final, final_body))
            final_node.declared_read_writes = final_node.read_writes
            return tuple(physical), final_node

        def select_domains(self, records):
            """Apply owner-selected work to this IR's existing scalar bodies."""
            import copy
            from torch._inductor import dependencies
            from torch._inductor.loop_body import LoopBody
            from .compilation import api_layouts
            from .layout import strides
            from .operand import iteration_order, layout_configuration, symbolic_index

            records = tuple(records)
            layouts = api_layouts('matrix')
            product_requirements = {name: (layouts[role],) for name, role in
                dict(x='input', x1='input', w='weight', out='output', bias='bias').items()}
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
                self.select_implementations(records)
                self._update_scheduler()
                prepared, undeclared, backend_operations = {}, set(), []
                self.reduction_uses = []
                declarations = {effect['operation'].get_operation_name(): effect for effect in self.declared_effects}
                for operation in tuple(self.operations):
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
                            stages, piece = self.split_reduction(piece)
                            for stage in stages:
                                prepared[stage.get_name()] = stage, stage.read_writes
                                backend_operations.append(stage.node)
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
                            node = pieces[0]
                            node.__dict__.pop('page_pieces', None)
                        else:
                            node.page_pieces = tuple(pieces)
                        rw = dependencies.ReadWrites.merge_list([piece.read_writes for piece in pieces])
                    elif isinstance(node, NopKernelSchedulerNode):
                        # The retained IR declares no command. Its result
                        # storage still passes through the ordinary binder.
                        rw = node.declared_read_writes
                    else:
                        import mesh_matrix
                        domains = declarations[operation.get_operation_name()]['domains']
                        if domains and all(domain.get('kind') == 'alias' for domain in domains):
                            pass
                        elif domains and all(domain.get('kind') == 'atomic' for domain in domains):
                            node.page_atomic = any(not request['work'].empty() for request in requests)
                        elif domains and all(domain.get('kind') == 'rows' for domain in domains):
                            from .selection import selected_rows
                            node.selected_plans = tuple(plan for request in requests for plan in selected_rows(request))
                        elif domains:
                            node.selected_plans = tuple(dict(plan, adapter='product') for request in requests for plan in mesh_matrix.selected_products(
                                request['domain'], request['work'], request['finish']['work'], requirements=product_requirements))
                        else:
                            undeclared.add(operation.get_operation_name())
                        rw = node.declared_read_writes
                    prepared[operation.get_operation_name()] = node, rw
                    backend_operations.append(node.node)
                self.selected_domains = records
                # Reuse the scheduler and its normal dependency/fusion passes.
                # create_scheduler_node retains these bodies and operation
                # objects; no GraphLowering or scalar capture runs again.
                self.scheduler.prepared_nodes = prepared
                self.scheduler._init(backend_operations)
                self.pending_declarations = tuple(node.node for group in self.scheduler.nodes for node in group.get_nodes()
                                                  if node.get_name() in undeclared)
            return self

        def consumer_uses(self):
            """Selected external API uses, before the common page-placement solve."""
            live = {output.get_name() for group in self.scheduler.nodes for node in group.get_nodes()
                    for output in node.get_outputs()}
            uses = [use for use in getattr(self, 'reduction_uses', ()) if use['binding']['name'] in live]
            for operation in self.operations:
                name = operation.get_operation_name()
                scheduled = self.scheduler.name_to_node[name]
                for index, plan in enumerate(getattr(scheduled, 'selected_plans', ())):
                    for role, use in plan['operands'].items():
                        uses.append(dict(key=(name, index, role), binding=use['access'], relation=use['relation'],
                                         shape=use['shape'], requirements=use['requirements'], access=use['access_mode']))
            return tuple(uses)

        def page_aliases(self):
            """Exact selected value equalities from cached load/store bodies."""
            from .operand import iteration_order
            aliases = []
            for record in getattr(self, 'selected_domains', None) or ():
                domain, work = record['domain'], record['work']
                body = domain.get('body')
                if body is None or not body.is_memory_copy() or work.empty():
                    continue
                read, = domain['reads'].values()
                write, = domain['writes'].values()
                if (read['dtype'] != write['dtype'] or read['relation'] is None or write['relation'] is None or
                    write['node'].args[4] is not None):
                    continue
                order, _ = iteration_order(work, domain['work'])
                pair = tuple(dict(binding=access, relation=order.then(
                    access['relation'].reshape((math.prod(domain['work']),))), shape=order.shape)
                             for access in (read, write))
                aliases.append(dict(operation=record['effect']['operation'], domain=domain, work=work, pair=pair))
            return tuple(aliases)

        def emit(self, operands, *, storage, results, pages, uses):
            """Bind already selected scalar domains to their assigned pages.

            ``storage`` maps each retained IR storage name to ``source`` (its
            selected original scalar positions), ``source_dtype`` (their explicit
            scalar unit), and ``operand`` (the assigned compact page view in
            that same unit and order). ``results`` retains requested
            output order and repetitions. Placement owns allocation; emission
            only composes those relations into the existing emitted bindings.
            """
            nonlocal assigned_results
            if getattr(self, 'pending_declarations', ()):
                names = ', '.join(operation.get_operation_name() for operation in self.pending_declarations)
                raise ValueError(f'selected numerical work is missing external access declarations: {names}')
            bound_storage.clear(); storage_layouts.clear()
            views.clear(); external_arguments.clear(); external_outputs.clear(); command_accesses.clear()
            with environment(), V.set_graph_handler(self), V.set_extern_kernel_nodes([]), mode:
                self._update_scheduler()
                with unset_fake_temporarily():
                    assigned_results = tuple(results)
                    for name, binding in storage.items():
                        physical = TensorView.of(binding['operand'])
                        storage_layouts[name] = dict(source=binding['source'], source_dtype=binding['source_dtype'], operand=physical)
                        if not isinstance(self.get_buffer(name), ir.InputBuffer):
                            bound_storage[name] = physical
                    actuals = [storage_layouts[name]['operand'] if name in storage_layouts else actual
                               for (name, value), actual in zip(self.graph_inputs.items(), operands, strict=True)]
                    for operation in self.operations:
                        scheduled = self.scheduler.name_to_node[operation.get_operation_name()]
                        if not hasattr(scheduled, 'selected_plans'):
                            continue
                        plans = []
                        for index, plan in enumerate(scheduled.selected_plans):
                            arguments = dict(adapter=plan['adapter'], operands={}, parameters=plan['parameters'], before=[], after=[],
                                declaration=dict(operation=operation, index=index,
                                                 operands=plan['operands'], parameters=plan['parameters']))
                            for role in plan['operands']:
                                assigned = uses[operation.get_operation_name(), index, role]
                                arguments['operands'][role] = assigned['operand']
                                arguments['before'].extend(assigned['before'])
                                arguments['after'].extend(assigned['after'])
                            plans.append(arguments)
                        scheduled.page_plans = tuple(plans)
                # Identity continuations disappear only after the supplied
                # pages actually satisfy their declared selected equalities.
                # The ordinary scheduler's removed-op mechanism also handles
                # a copy embedded in an otherwise fused scheduling group.
                removed = self.scheduler.removed_ops.copy()
                aliased = {}
                with unset_fake_temporarily():
                    for alias in self.page_aliases():
                        actual = []
                        for use in alias['pair']:
                            from .operand import relation_layout
                            access = use['binding']
                            binding = storage_layouts.get(access['name'])
                            if binding is None:
                                break
                            original, unit, physical = (binding[key] for key in ('source', 'source_dtype', 'operand'))
                            logical = relation_layout(use['relation'], use['shape'],
                                int(access['storage_bytes']) // access['dtype'].itemsize)
                            mapped = rebase_typed(logical, access['dtype'].itemsize, original, unit.itemsize, physical.layout)
                            actual.append(TensorView(physical.storage_as(access['dtype']), mapped))
                        name = alias['operation'].get_operation_name()
                        aliased[name] = aliased.get(name, True) and len(actual) == 2 and same_view(*actual)
                for name, same in aliased.items():
                    if same:
                        self.scheduler.removed_ops.add(name)
                try:
                    source, _ = self.codegen()
                finally:
                    self.scheduler.removed_ops = removed
                namespace = {'__name__': 'mesh_preparation', 'mesh_destinations': bound_storage, 'mesh_views': views,
                             'mesh_results': assigned_results, 'mesh_externals': external_arguments,
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
                from .command_declarations import dense, bind_accesses
                accesses = []
                for item in declaration['accesses']:
                    slot = item['slot']
                    argument = declaration['arguments'][slot]
                    count = argument['source'].count * argument['source_dtype'].itemsize
                    selected = bound_access(item['access'], item['work'], argument['source'],
                                            argument['source_dtype'], dense((count,)))
                    accesses.append(dict(item, argument_layout=selected['layout']))
                declaration['accesses'] = tuple(accesses)
                declaration['reaches'] = bind_accesses(declaration)
            return dict(commands=tuple(records), output=output,
                        pages=tuple(pages),
                        storage=dict(storage_layouts))

        def bind(self, operands, **placement):
            return Program.prepare(self.emit(operands, **placement))

        def effects(self, algebra=None, *, bodies=None):
            """Logical operations with one shared iteration domain per body.

            Domains retain the cached body's symbols, guards and scalar nodes.
            IR dependencies retain their declared role; the selected scheduler
            later adds its versioned ordering dependencies.
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
                requirements = {record['operation'].get_operation_name(): record for record in self.consumer_constraints(bodies)}
                for operation in self.operations:
                    data = operation.data if isinstance(operation, ir.ComputedBuffer) else None
                    body = self.loop_body(operation, bodies)
                    dependencies = self.declaration_nodes[operation.get_operation_name()].read_writes
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
            self.device_type = (scalar_compilers[self.selected_implementations[0]['implementation']['compiler']][1]
                                if self.selected_implementations else 'mps')
            self.device_ops = get_device_op_overrides(self.device_type)
            self.wrapper_code = Preparation()
            self.wrapper_code.header.writeline('from torch_mesh import native as mesh_native')

        def _update_scheduler(self):
            if getattr(self, 'selected_implementations', None) is None:
                self.select_implementations()
            self.init_wrapper_code()
            if self.scheduler is None:
                PageKernel.max_threadgroup_size = min(configuration['ROW_THREADS'], _stream.device_facts()[1])
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
        graph.storage_aliases = {}
        with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]), mode:
            # Keep reduction axes until ownership selects work. The pinned
            # source hook applies even when an operator has its own policy.
            # Later preparation uses NativeChoices and the upstream policy.
            if not hasattr(InductorChoices, 'reduction_unroll_threshold'):
                raise RuntimeError('Torch compiler source extension is missing; install with bin/mesh-user.sh')
            with V.set_choices_handler(LogicalChoices()):
                graph.run(*metadata)
            graph.never_reuse_buffers.update(materialized_buffers)
            graph.retain_declarations()
            graph.compose_reads()
    return graph
