"""FX numerical lowering into the native recorder's command descriptions.

Inductor supplies tensor IR, fusion and Metal source. Its Python wrapper is a
preparation program: allocations, views and kernel bindings run once here.
No generated wrapper, ATen fallback or shader launcher runs with the payload.
"""
from contextvars import ContextVar
from functools import lru_cache
import struct

import torch

from . import _stream

_records = ContextVar('mesh_native_commands')


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
    outputs = outputs if outputs is not None else [torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
                                                   for x, w in zip(xs, ws)]
    if all(x.dtype == w.dtype == torch.half and x.shape[0] <= 8 and x.stride(1) == w.stride(1) == 1
           and (part is None or part.stride(1) == 1) and (bias is None or bias.is_contiguous()) and out.is_contiguous()
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


class Program:
    """Bound numerical command records; composition happens during preparation."""
    def __init__(self, commands, output):
        self.commands, self.output = commands, output

    def append(self, schedule):
        schedule.append(self.commands)


def compile_graph(module, metadata, operands):
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

    class Preparation(PythonWrapperCodegen):
        def _generate_extern_kernel_alloc_helper(self, node, args):
            names = {'torch.ops.torch_mesh.linear.default': 'linear',
                     'torch.ops.torch_mesh.linears.default': 'linears'}
            name = node.get_kernel_name()
            if name not in names:
                raise NotImplementedError(f'no native preparation for {name}; a framework fallback is not an executor')
            self.writeline(f'{node.get_name()} = mesh_native.{names[name]}({", ".join(args)})')

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
            access = (1 << (len(outputs) + len(inputs))) - 1 | ((1 << len(outputs)) - 1) << 32
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
                      max_fusion_unique_io_buffers=31, max_autotune=False,
                      max_autotune_gemm=False, assert_indirect_indexing=False), \
            V.set_fake_mode(mode), DebugContext():
        graph = NativeGraph(module, metadata, shape_env=mode.shape_env, is_inference=True)
        with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]), mode:
            graph.run(*metadata)
            source, _ = graph.codegen()
        namespace = {'__name__': 'mesh_preparation'}
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
