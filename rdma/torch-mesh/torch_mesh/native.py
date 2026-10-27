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

_records = ContextVar('mesh_native_commands')


def empty_strided(shape, stride, dtype, device):
    """Bind a compiler-assigned view to bridge storage during preparation."""
    from . import empty
    extent = 0 if math.prod(shape) == 0 else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    return empty((extent,), dtype=dtype, device=device).as_strided(shape, stride)


def row_offset(name, shape, strides, axis):
    """Emit the base of a tensor view's rows along one declared axis."""
    terms, divisor = [], 1
    for dim in range(len(shape) - 1, -1, -1):
        if dim != axis:
            terms.append(f'(ulong(row) / {divisor} % {shape[dim]}) * {strides[dim]}')
            divisor *= shape[dim]
    return f'inline ulong {name}(uint row) {{ return ' + (' + '.join(terms) or '0') + '; }\n'


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
            tensors.append(value if isinstance(value, torch.Tensor) else None)
            if not isinstance(value, torch.Tensor):
                constants[index] = struct.pack('q', int(value))
        _records.get().append((handle, tensors, constants, grid, group, True, access))
    return bind


def linear(x, w, bias=None, act=0, x1=None, *, out=None):
    return linears([x], [w], [bias], [act], [x1], outputs=None if out is None else [out])[0]


def linears(xs, ws, biases, acts, x1s, *, outputs=None):
    from .placement import output
    outputs = [output((x.shape[0], w.shape[0]), x.dtype, x.device, out)
               for x, w, out in zip(xs, ws, outputs if outputs is not None else [None] * len(xs), strict=True)]
    if all(x.dtype == w.dtype == torch.half and x.shape[0] <= compiler_operands()['LINEAR_ROW_CROSSOVER'] and x.stride(1) == w.stride(1) == 1
           and (part is None or part.stride(1) == 1) and (bias is None or bias.is_contiguous())
           for x, w, part, bias, out in zip(xs, ws, x1s, biases, outputs)):
        _records.get().extend(_stream.linear_commands(xs, ws, biases, acts, x1s, outputs))
    else:
        from .matrix import product
        for x, w, part, bias, act, out in zip(xs, ws, x1s, biases, acts, outputs):
            if out.numel():
                _records.get().append(product(x, w.t(), out, x1=part, bias=bias, act=bool(act)))
    return outputs


def mm(x, w, *, out):
    from .matrix import product
    if out.numel():
        _records.get().append(product(x, w, out))


def addmm(bias, x, w, *, out, alpha=1, beta=1):
    from .matrix import product
    if out.numel():
        _records.get().append(product(x, w, out, bias=bias, alpha=alpha, beta=beta))


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
    from torch._inductor.codegen.mps import MetalScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.graph import GraphLowering
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.virtualized import V
    from torch._inductor.debug import DebugContext
    from torch._inductor import ir
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from torch.fx.node import map_arg
    from .placement import Placements, same_view
    from views import rebase

    placements, bound_storage, placed_values = Placements(), {}, []
    output_nodes = next(n.args[0] for n in module.graph.nodes if n.op == 'output')
    described_outputs = tree_flatten(map_arg(output_nodes, lambda n: n.meta['val']))[0]
    if destinations is not None:
        for described, destination in zip(described_outputs, destinations, strict=True):
            if destination is not None:
                placements.bind(described, destination)
    input_storage = dict(zip((n.name for n in module.graph.nodes if n.op == 'placeholder'), operands, strict=True))

    class Preparation(PythonWrapperCodegen):
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
        def define_kernel(self, source, node_schedule, kernel):
            wrapper = V.graph.wrapper_code
            if source in wrapper.src_to_kernel:
                return wrapper.src_to_kernel[source]
            start = source.index("compile_mps_shader('''") + len("compile_mps_shader('''")
            source_text = source[start:source.rindex("''')")]
            if 'error' in kernel.headers:
                raise NotImplementedError('runtime index assertions must be resolved in the declared graph')
            outputs = [name for name in kernel.args.output_buffers if name not in kernel.removed_buffers]
            inputs = list(kernel.args.input_buffers)
            access = (1 << (len(outputs) + len(inputs))) - 1 | ((1 << len(outputs)) - 1) << _stream.compilation['ACCESS_WRITE_SHIFT']
            name = f'mesh_kernel_{wrapper.next_kernel_suffix()}'
            wrapper.src_to_kernel[source] = name
            wrapper.header.writeline(f'{name} = mesh_native.shader({source_text!r}, "generated_kernel", {access})')
            return name

    class NativeScheduler(Scheduler):
        def create_backend(self, device):
            if device.type != 'mps':
                raise NotImplementedError(f'FX numerical commands require a declared native backend for {device}')
            return MetalSource(self)

    class NativeGraph(GraphLowering):
        def run_node(self, node):
            logical = node.meta.get('val')
            schema = getattr(node.target, '_schema', None)
            # A declared alias of an already placed producer is just a view.
            # Lower it here, before an upstream reshape helper can pack it.
            if (node.op == 'call_function' and isinstance(logical, torch.Tensor)
                    and schema is not None and not schema.is_mutable
                    and len(schema.returns) == 1 and schema.returns[0].alias_info is not None):
                for previous, storage, physical in reversed(placed_values):
                    if previous.dtype == logical.dtype and torch._C._is_alias_of(previous, logical):
                        with unset_fake_temporarily():
                            destination = placements.get(logical)
                        layout = ir.FixedLayout(logical.device, logical.dtype, list(destination.shape),
                                                list(destination.stride()), destination.storage_offset() - physical.storage_offset())
                        return ir.TensorBox(ir.ReinterpretView(data=storage, layout=layout))
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
                        actual = actual.as_strided(tuple(layout.size), tuple(layout.stride), actual.storage_offset() + int(layout.offset))
                        if not same_view(actual, destination):
                            raise ValueError('a supplied graph input cannot be relocated by producer placement')
                    placed_values.append((logical, storage, input_storage[buffer.get_name()]))
                    continue
                if buffer.get_dtype() != destination.dtype:
                    raise ValueError('native producer placement requires a matching storage element type')
                if name not in bound_storage:
                    base = buffer.get_layout()
                    offset, strides = rebase(layout.size, layout.stride, layout.offset,
                                              destination.stride(), destination.storage_offset(),
                                              base.size, base.stride, base.offset)
                    with unset_fake_temporarily():
                        bound_storage[name] = destination.as_strided(tuple(base.size), strides, offset)
                    buffer.layout = ir.FixedLayout(base.device, base.dtype, list(base.size), list(strides))
                    self.never_reuse_buffers.add(name)
                placed = bound_storage[name]
                placed_values.append((logical, storage, placed))
                relative = destination.storage_offset() - placed.storage_offset()
                view = ir.FixedLayout(layout.device, layout.dtype, list(layout.size), list(destination.stride()), relative)
                # Views retain their physical relation after the producer's
                # layout changes. No tensor copy is introduced to freeze it.
                leaves[i] = ir.TensorBox(ir.ReinterpretView(data=storage, layout=view))
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
    with config.patch(size_asserts=False, alignment_asserts=False, scalar_asserts=False,
                      max_fusion_unique_io_buffers=compiler_operands()['BUFFER_BINDINGS'], max_autotune=False,
                      max_autotune_gemm=False, assert_indirect_indexing=False), \
            V.set_fake_mode(mode), DebugContext():
        graph = NativeGraph(module, metadata, shape_env=mode.shape_env, is_inference=True)
        with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]), mode:
            graph.run(*metadata)
            source, _ = graph.codegen()
        namespace = {'__name__': 'mesh_preparation', 'mesh_destinations': bound_storage}
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
