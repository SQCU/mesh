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


def empty_strided(shape, stride, dtype, device):
    """Bind a compiler-assigned view to bridge storage during preparation."""
    from . import empty
    extent = 0 if math.prod(shape) == 0 else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    return empty((extent,), dtype=dtype, device=device).as_strided(shape, stride)


@lru_cache(None)
def shader(source, name, access):
    from pathlib import Path
    from torch.utils._cpp_embed_headers import _embed_headers
    source = _embed_headers(source.splitlines(keepends=True), [Path(torch.__file__).parent / 'include'], set())
    handle = _stream.pipeline(source, name)
    maximum, _, _ = _stream.pipeline_facts(handle)

    def bind(*arguments, threads, group_size=None):
        grid = tuple(threads) + (1,) * (3 - len(threads))
        if group_size is None:
            remaining, group_size = maximum, []
            for extent in grid:
                part = min(remaining, extent)
                group_size.append(part)
                remaining //= part
        group = tuple(group_size) + (1,) * (3 - len(group_size))
        tensors, constants = [], {}
        for index, value in enumerate(arguments):
            tensors.append(value if isinstance(value, (torch.Tensor, TensorView)) else None)
            if not isinstance(value, (torch.Tensor, TensorView)):
                constants[index] = struct.pack('q', int(value))
        _records.get().append((handle, tensors, constants, grid, group, True, access))
    return bind


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


def mm(x, w, *, out):
    from .matrix import product
    if out.numel():
        _records.get().extend(product(x, w, out))


def addmm(bias, x, w, *, out, alpha=1, beta=1):
    from .matrix import product
    if out.numel():
        _records.get().extend(product(x, w, out, bias=bias, alpha=alpha, beta=beta))


def topk(x, k, dim=-1, largest=True, sorted=True, *, outputs=None):
    from .selection import topk as prepare
    output, commands = prepare(x, k, dim, largest, sorted, outputs=outputs)
    _records.get().extend(commands)
    return output


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
    def __init__(self, commands, output):
        self.commands, self.output = commands, output

    def append(self, schedule):
        schedule.append(self.commands)


def compile_graph(module, metadata, operands, *, destinations=None):
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._inductor import config
    from torch._inductor.codegen.common import get_device_op_overrides
    from torch._inductor.codegen.mps import MetalScheduling, MetalKernel
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.choices import InductorChoices
    from torch._inductor.graph import GraphLowering
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.virtualized import V
    from torch._inductor.debug import DebugContext
    from torch._inductor import ir
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from torch.fx.node import map_arg
    from .placement import Placements, same_view
    from contextlib import contextmanager
    from mesh_compute import hardware

    placements, bound_storage, views, storage_layouts = Placements(), {}, [], {}
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
            if name in storage_layouts:
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
            index = len(views)
            views.append(self.operand)
            return f'mesh_views[{index}]'

    def page_view(storage, layout, operand):
        result = PageView(data=storage, layout=layout)
        result.operand = operand
        return ir.TensorBox(result)

    def ir_layout(layout):
        return _stream.TensorLayout(tuple(map(int, layout.size)), tuple(map(int, layout.stride)), int(layout.offset))
    output_nodes = next(n.args[0] for n in module.graph.nodes if n.op == 'output')
    described_outputs = tree_flatten(map_arg(output_nodes, lambda n: n.meta['val']))[0]
    if destinations is not None:
        for described, destination in zip(described_outputs, destinations, strict=True):
            if destination is not None:
                placements.bind(described, destination)
    input_storage = dict(zip((n.name for n in module.graph.nodes if n.op == 'placeholder'), operands, strict=True))
    for node in module.graph.nodes:
        if node.op == 'placeholder' and isinstance(node.meta.get('val'), torch.Tensor):
            placements.bind(node.meta['val'], input_storage[node.name])
    from .layout import place
    place(module.graph, placements, [v for n in module.graph.nodes if n.op in ('placeholder', 'get_attr')
                                    for v in tree_flatten(n.meta.get('val'))[0]])

    class Preparation(PythonWrapperCodegen):
        def codegen_reinterpret_view(self, data, size, stride, offset, writeline, dtype=None):
            if data.get_name() not in storage_layouts:
                return super().codegen_reinterpret_view(data, size, stride, offset, writeline, dtype)
            original, physical = storage_layouts[data.get_name()]
            dtype = physical.dtype if dtype is None else dtype
            described = _stream.TensorLayout(tuple(map(int, size)), tuple(map(int, stride)), int(offset))
            mapped = rebase_typed(described, dtype.itemsize, original, physical.element_size(), physical.layout)
            views.append(TensorView(physical.storage_as(dtype), mapped))
            return f'mesh_views[{len(views) - 1}]'

        def make_allocation(self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False):
            if name in bound_storage:
                return f'{name} = mesh_destinations[{name!r}]'
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
            # Caller-owned landing storage is fixed, not a scratch pool slot.
            if input_buffer.get_name() in bound_storage or (output_buffer is not None and output_buffer.get_name() in bound_storage):
                return False
            return super().can_reuse(input_buffer, output_buffer)

        def _generate_extern_kernel_alloc_helper(self, node, args):
            names = {'torch.ops.torch_mesh.linear.default': 'linear',
                     'torch.ops.torch_mesh.linears.default': 'linears',
                     'torch.ops.aten.topk.default': 'topk',
                     'torch.ops.aten.cumsum.default': 'cumsum',
                     'torch.ops.torch_mesh.draw.default': 'draw',
                     'torch.ops.torch_mesh.retain.default': 'retain',
                     'torch.ops.torch_mesh.fwht.default': 'fwht',
                     'torch.ops.torch_mesh.fwht_stage.default': 'fwht_stage',
                     'torch.ops.torch_mesh.native_kernel.default': 'kernel'}
            name = node.get_kernel_name()
            if name not in names:
                raise NotImplementedError(f'no native preparation for {name}; a framework fallback is not an executor')
            outputs = node.codegen_outputs()
            output_refs = [f'mesh_destinations[{leaf.get_name()!r}]' if isinstance(leaf, ir.IRNode)
                           and leaf.get_name() in bound_storage else 'None' for leaf in tree_flatten(outputs)[0]]
            keyword = 'outputs' if names[name] in ('linears', 'topk', 'retain', 'kernel') else 'out'
            destination = '[' + ', '.join(output_refs) + ']' if keyword == 'outputs' else output_refs[0]
            self.writeline(f'{node.get_name()} = mesh_native.{names[name]}({", ".join(args)}, {keyword}={destination})')

        def _generate_extern_kernel_out_helper(self, kernel, out, out_view, args, device, stack_traces=None):
            names = {'extern_kernels.mm': 'mm', 'extern_kernels.bmm': 'mm', 'extern_kernels.addmm': 'addmm'}
            if kernel not in names or device != 'mps':
                raise NotImplementedError(f'no native preparation for {kernel} on {device}')
            self.writeline(f'mesh_native.{names[kernel]}({", ".join(args)}, out={out_view or out})')

    class MetalSource(MetalScheduling):
        kernel_type = PageKernel

        def define_kernel(self, source, node_schedule, kernel):
            wrapper = V.graph.wrapper_code
            start = source.index("compile_mps_shader('''") + len("compile_mps_shader('''")
            source_text = source[start:source.rindex("''')")]
            declarations = []
            for name, slot in getattr(kernel, 'page_bindings', {}).items():
                original, operand = storage_layouts[name]
                declarations.append(operand.layout.storage_declaration(slot, original))
            source_text = _stream.view_prelude() + ''.join(declarations) + source_text
            if source_text in wrapper.src_to_kernel:
                return wrapper.src_to_kernel[source_text]
            if 'error' in kernel.headers:
                raise NotImplementedError('runtime index assertions must be resolved in the declared graph')
            outputs = [name for name in kernel.args.output_buffers if name not in kernel.removed_buffers]
            inputs = list(kernel.args.input_buffers)
            access = (1 << (len(outputs) + len(inputs))) - 1 | ((1 << len(outputs)) - 1) << _stream.compilation['ACCESS_WRITE_SHIFT']
            name = f'mesh_kernel_{wrapper.next_kernel_suffix()}'
            wrapper.src_to_kernel[source_text] = name
            wrapper.header.writeline(f'{name} = mesh_native.shader({source_text!r}, "generated_kernel", {access})')
            return name

    class NativeScheduler(Scheduler):
        def create_backend(self, device):
            if device.type != 'mps':
                raise NotImplementedError(f'FX numerical commands require a declared native backend for {device}')
            return MetalSource(self)

    class NativeGraph(GraphLowering):
        def run_node(self, node):
            result = super().run_node(node)
            if not placements.bindings or node.op == 'output' or 'val' not in node.meta:
                return result
            leaves, spec = tree_flatten(result)
            described = tree_flatten(node.meta['val'])[0]
            for i, (value, logical) in enumerate(zip(leaves, described, strict=True)):
                with unset_fake_temporarily():
                    destination = placements.get(logical)
                if destination is None:
                    continue
                # Realize the existing producer. realize_input(Buffer) in the
                # upstream IR can create a copy, so it is not a placement API.
                if isinstance(value, (ir.TensorBox, ir.StorageBox, ir.BaseView)):
                    value.realize()
                while isinstance(value, (ir.TensorBox, ir.StorageBox)):
                    value = value.data
                if isinstance(value, ir.BaseView) and not isinstance(value, ir.ReinterpretView):
                    value = ir.ExternKernel.convert_to_reinterpret_view(value)
                storage, layout = ir.as_storage_and_layout(value)
                buffer = storage.data
                name = buffer.get_name()
                if isinstance(buffer, ir.InputBuffer):
                    actual = input_storage[node.name] if node.op == 'placeholder' else input_storage[buffer.get_name()]
                    with unset_fake_temporarily():
                        from .placement import alias
                        # Inputs keep their supplied pages. Their aliases use
                        # the same declaration-to-placement relation as outputs.
                        original = next(n.meta['val'] for n in module.graph.nodes
                                        if n.op == 'placeholder' and n.name == buffer.get_name())
                        actual = alias(original, actual, logical)
                        if not same_view(actual, destination):
                            raise ValueError('a supplied graph input cannot be relocated by producer placement')
                    physical = TensorView.of(input_storage[buffer.get_name()])
                    storage_layouts[name] = (ir_layout(buffer.get_layout()), physical)
                    leaves[i] = page_view(storage, layout, destination)
                    continue
                if name not in bound_storage:
                    base = buffer.get_layout()
                    source_layout, base_layout = ir_layout(layout), ir_layout(base)
                    with unset_fake_temporarily():
                        mapped = rebase_typed(base_layout, base.dtype.itemsize, source_layout, layout.dtype.itemsize, destination.layout)
                        bound_storage[name] = TensorView(destination.storage_as(base.dtype), mapped)
                    storage_layouts[name] = (base_layout, bound_storage[name])
                    self.never_reuse_buffers.add(name)
                # Logical IR remains unchanged. Emission and command binding
                # use the producer's one physical page relation.
                leaves[i] = page_view(storage, layout, destination)
            return tree_unflatten(leaves, spec)

        def init_wrapper_code(self, *args, **kwargs):
            self.device_type = 'mps'
            self.device_ops = get_device_op_overrides('mps')
            self.wrapper_code = Preparation()
            self.wrapper_code.header.writeline('from torch_mesh import native as mesh_native')

        def _update_scheduler(self):
            self.scheduler = NativeScheduler(self.operations)

    mode = detect_fake_mode(metadata)
    if mode is None:
        raise ValueError('native lowering requires the declaration\'s fake tensor metadata')
    with config.patch(size_asserts=False, alignment_asserts=False, scalar_asserts=False, split_reductions=True,
                      max_fusion_unique_io_buffers=configuration['BUFFER_BINDINGS'], max_autotune=False,
                      max_autotune_gemm=False, assert_indirect_indexing=False), \
            V.set_fake_mode(mode), V.set_choices_handler(NativeChoices()), DebugContext():
        graph = NativeGraph(module, metadata, shape_env=mode.shape_env, is_inference=True)
        with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]), mode:
            graph.run(*metadata)
            source, _ = graph.codegen()
        namespace = {'__name__': 'mesh_preparation', 'mesh_destinations': bound_storage, 'mesh_views': views}
        records = []
        token = _records.set(records)
        try:
            with unset_fake_temporarily():
                exec(compile(source.value, '<mesh native preparation>', 'exec'), namespace)
                output = namespace['call'](list(operands))
                commands = _stream.Commands(records)
        finally:
            _records.reset(token)
    return Program(commands, output)
