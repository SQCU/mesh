"""Compute-share declarations for the common whole-function compiler.

Declare shapes and choose shares before execution. ``Operation.prepare`` binds
weights, outputs and host views through one ``declarations.Function`` graph.
Host shares lower to the native recorder's host producer/consumer records. Only result
consumers depend on host completion. No operation times itself, chooses shares,
loads a model, or joins a future while it runs.
"""
import ctypes
import hashlib
import math
import json
import os
import shutil
from pathlib import Path

import torch

_objc = None


def _send(pointer, selector, restype=ctypes.c_void_p):
    global _objc
    if _objc is None:
        _objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        _objc.sel_registerName.restype, _objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    call = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', _objc))
    return call(pointer, _objc.sel_registerName(selector))


def host(t):
    """An MPS tensor's memory as a CPU tensor of the same shape and dtype, in place (its storage's MTLBuffer is
    shared), or None where its storage is not CPU-visible.  The caller orders the GPU's and the host's accesses."""
    if t.device.type == 'cpu':
        return t
    if t.device.type != 'mps':
        return None
    buffer = t.untyped_storage().data_ptr()
    if _send(buffer, b'storageMode', ctypes.c_ulong) != 0:
        return None
    base = _send(buffer, b'contents')
    elements = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride())) if t.numel() else 0
    nbytes = elements * t.element_size()
    raw = (ctypes.c_uint8 * nbytes).from_address(base + t.storage_offset() * t.element_size())
    view = torch.frombuffer(raw, dtype=t.dtype, count=elements).as_strided(t.shape, t.stride()) if nbytes else torch.empty(t.shape, dtype=t.dtype)
    view._mesh_storage = t
    return view


def _host_address(value):
    """Bind the shared allocation and its declared offset once at preparation."""
    from .operand import TensorView
    view = TensorView.of(value)
    storage = host(view.storage)
    return storage.data_ptr() + (view.layout.offset - view.storage.storage_offset()) * view.element_size()


def _host_bindings(source, target, input_layouts, output_layouts, binding_layout):
    """CPU and Core ML consume the same resolved operands and copy records."""
    from . import layout
    from .operand import TensorView
    prepared, before, after = [], [], []
    for values, requirements, writing in ((source, input_layouts, False), (target, output_layouts, True)):
        devices = {name: 'cpu' if host(TensorView.of(value).storage) is None else value.device.type for name, value in values.items()}
        placed, commands = layout.bindings(values, requirements, writing=writing, binding=binding_layout, devices=devices)
        (after if writing else before).extend(commands)
        prepared.append(placed)
    return *prepared, before, after


_CORE_ML = None


def _coreml():
    """The native Core ML binding (rdma/mesh-coreml.m: predictions on caller memory, no copies on the CPU), the mesh's
    installed library beside mesh.py; None where it cannot load."""
    global _CORE_ML
    if _CORE_ML is None:
        from mesh import RDMA
        try:
            lib = ctypes.CDLL(os.path.join(RDMA, 'libmesh-coreml.dylib'))
            lib.mesh_coreml_load.restype = ctypes.c_long
            lib.mesh_coreml_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
            lib.mesh_coreml_release.argtypes = [ctypes.c_long]
            lib.mesh_coreml_bind.restype = ctypes.c_void_p
            lib.mesh_coreml_bind.argtypes = [ctypes.c_long, ctypes.POINTER(ModelView), ctypes.c_size_t,
                                             ctypes.POINTER(ModelView), ctypes.c_size_t]
            lib.mesh_coreml_run.restype, lib.mesh_coreml_run.argtypes = ctypes.c_char_p, [ctypes.c_void_p]
            lib.mesh_coreml_retain.argtypes = [ctypes.c_void_p]
            lib.mesh_coreml_unbind.argtypes = [ctypes.c_void_p]
            lib.mesh_coreml_error.restype = ctypes.c_char_p
            lib.mesh_coreml_placement.restype = ctypes.c_long
            lib.mesh_coreml_placement.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_size_t]
            lib.mesh_coreml_compile.restype = ctypes.c_int
            lib.mesh_coreml_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.mesh_cpu_program_create.argtypes = [ctypes.POINTER(CpuCommand), ctypes.c_size_t]
            lib.mesh_cpu_program_create.restype = ctypes.c_void_p
            lib.mesh_cpu_program_run.argtypes = [ctypes.c_void_p]
            lib.mesh_cpu_program_run.restype = ctypes.c_char_p
            lib.mesh_cpu_program_retain.argtypes = [ctypes.c_void_p]
            lib.mesh_cpu_program_release.argtypes = [ctypes.c_void_p]
            lib.mesh_cpu_product_bind.argtypes = [ctypes.POINTER(CpuCommand)]
            lib.mesh_cpu_product_bind.restype = ctypes.c_int
            lib.mesh_cpu_command_release.argtypes = [ctypes.POINTER(CpuCommand)]
            _CORE_ML = lib
        except OSError:
            _CORE_ML = False
    return _CORE_ML or None


def _gelu(x):
    return torch.nn.functional.gelu(x, approximate='tanh')


def ffn(x, gate, up, down, out=None):
    """down(gelu_tanh(x gate^T) * (x up^T)) for x [n, H], whatever the operands are: where their type has a fused
    couple (`ffn`, as a coded matrix's) and no gradient is due, that; else torch's own products, which autograd
    differentiates."""
    fused = getattr(type(gate), 'ffn', None)
    if fused is not None and not (torch.is_grad_enabled() and x.requires_grad):
        y = fused(x, gate, up, down)
        return y if out is None else out.copy_(y)
    h = (_gelu(torch.mm(x, gate.T).float()) * torch.mm(x, up.T).float()).to(x.dtype)
    return torch.mm(h, down.T) if out is None else torch.mm(h, down.T, out=out)


class Engine:
    """One engine of the node.  `limits`: dtypes (the operand dtypes it takes), constants (its weights must be
    constants).  `occupies`: the physical units it runs on (the configuration's `occupies` option, '+'-separated, in
    place of the engine's own).  It lowers each operation kind by its prepare_<kind> (its resident shard of units
    [start, end)) and run_<kind> (its share into `out`)."""
    name, limits = None, {'dtypes': (torch.float16, torch.bfloat16, torch.float32), 'constants': False}
    host_side, occupies = True, frozenset()

    def __init__(self, **options):
        self.options = options
        if 'occupies' in options:
            self.occupies = frozenset(u for u in str(options['occupies']).split('+') if u)

    @classmethod
    def missing(cls):
        """Why the node lacks this engine, or None."""
        return None

    def takes(self, op, dtype):
        return hasattr(self, 'prepare_' + op.kind) and dtype in self.limits['dtypes']

    def input_dtype(self, dtype):
        return dtype

    def out_dtype(self, dtype):
        return dtype

    def sum_dtype(self):
        return torch.float32

    def prepare(self, op, start, end):
        return getattr(self, 'prepare_' + op.kind)(op, start, end)

    def run(self, op, shard, x, out):
        return getattr(self, 'run_' + op.kind)(op, shard, x, out)


class Mps(Engine):
    """GPU products declared to the whole-function compiler."""
    name, host_side, occupies = 'mps', False, frozenset({'gpu'})

    @classmethod
    def missing(cls):
        return None if torch.backends.mps.is_available() else 'no MPS device'

    def prepare_linear(self, op, start, end):
        return op.weight[start:end]

    def run_linear(self, op, shard, x, out):
        torch.mm(x, shard.T, out=out)

    def prepare_ffn(self, op, start, end):
        return op.gate[start:end], op.up[start:end], op.down[:, start:end]

    def run_ffn(self, op, shard, x, out):
        ffn(x, *shard, out=out)


class Cpu(Engine):
    """The CPU's matrix units through prepared Accelerate commands: the input read in place as fp32 (the
    GPU's prologue converts it), the fp32 shard resident in host memory, the output written in place in fp32."""
    name, occupies = 'cpu', frozenset({'cpu'})

    def input_dtype(self, dtype):
        return torch.float32

    def out_dtype(self, dtype):
        return torch.float32

    def prepare_linear(self, op, start, end):
        from . import empty
        weight = op.weight[start:end].detach()
        return empty(weight.shape, dtype=torch.float32, device='cpu').copy_(weight)

    def prepare_ffn(self, op, start, end):
        from . import empty
        def cpu(t):
            return empty(t.shape, dtype=torch.float32, device='cpu').copy_(t.detach())
        return cpu(op.gate[start:end]), cpu(op.up[start:end]), cpu(op.down[:, start:end])

    def declaration(self, op, shard, rows):
        from .compilation import api_layouts
        layouts = api_layouts('cpu_matrix')
        weights = (shard,) if op.kind == 'linear' else shard
        weights = {f'weight{i}': w for i, w in enumerate(weights)}
        return dict(definition=dict(provider='cpu', operation=op.kind, weights=list(weights)), weights=weights,
                    input_layouts=dict(input=layouts['input'], **{name: layouts['weight'] for name in weights}),
                    output_layouts={'output': layouts['output']}, binding_layout='indexed')

    def bind(self, op, shard, x, out, *, input_layouts, output_layouts, binding_layout):
        weights = (shard,) if op.kind == 'linear' else shard
        inputs = dict(input=x, **{f'weight{i}': w for i, w in enumerate(weights)})
        source, target, before, after = _host_bindings(inputs, {'output': out},
            input_layouts, output_layouts, binding_layout)
        x, out = source['input'], target['output']
        weights = tuple(source[f'weight{i}'] for i in range(len(weights)))
        native = _coreml()
        if native is None:
            raise RuntimeError('CPU commands require the native engine library')
        def product(x, w, y):
            command = CpuCommand(None,
                              x.shape[0], w.shape[0], x.shape[1], x.stride(0), w.stride(0), y.stride(0),
                              x.stride(1), w.stride(1), y.stride(1),
                              _host_address(x), _host_address(w), _host_address(y), None)
            if native.mesh_cpu_product_bind(ctypes.byref(command)):
                raise ValueError('CPU matrix views could not be bound to a bulk contraction')
            return command
        if op.kind == 'linear':
            return CpuProgram([product(x, weights[0], out)], tuple(source.values()), (out,), before, after)
        gate, up, down = weights
        from . import empty
        g, u, scratch = (empty(x.shape[0], gate.shape[0], dtype=torch.float32, device='cpu') for _ in range(3))
        activation = CpuCommand(ctypes.cast(native.mesh_cpu_gelu, ctypes.c_void_p), g.shape[0], g.shape[1], 0, 0, 0, 0, 0, 0, 0,
                                 g.data_ptr(), u.data_ptr(), g.data_ptr(), scratch.data_ptr())
        return CpuProgram([product(x, gate, g), product(x, up, u), activation, product(g, down, out)],
                          tuple(source.values()), (out, g, u, scratch), before, after)


class CpuCommand(ctypes.Structure):
    _fields_ = [('run', ctypes.c_void_p)] + [(name, ctypes.c_int) for name in ('rows', 'columns', 'inner', 'input_stride', 'weight_stride', 'output_stride',
                                                                          'input_column', 'weight_column', 'output_column')] \
             + [(name, ctypes.c_void_p) for name in ('input', 'weight', 'output', 'scratch')]
    _fields_ += [(name, ctypes.c_int) for name in ('input_transpose', 'weight_transpose')]


def _native_entry(run, binding, retain, release):
    return (ctypes.cast(run, ctypes.c_void_p).value, binding,
            ctypes.cast(retain, ctypes.c_void_p).value, ctypes.cast(release, ctypes.c_void_p).value)


class CpuProgram:
    def __init__(self, commands, inputs, outputs, before, after):
        from . import _stream
        self.native = _coreml()
        if self.native is None:
            raise RuntimeError('CPU commands require the native engine library')
        records = (CpuCommand * len(commands))(*commands)
        self.binding = self.native.mesh_cpu_program_create(records, len(records))
        if not self.binding:
            for command in records:
                self.native.mesh_cpu_command_release(ctypes.byref(command))
            raise MemoryError('preparing CPU commands')
        self.inputs, self.outputs = inputs, outputs
        self.before, self.after = _stream.Commands(before), _stream.Commands(after)
        self.entry = _native_entry(self.native.mesh_cpu_program_run, self.binding,
                                   self.native.mesh_cpu_program_retain, self.native.mesh_cpu_program_release)

    def append(self, schedule):
        schedule.append(self.before)
        schedule.host(self.entry, self.inputs, self.outputs)
        schedule.append(self.after)

    def __del__(self):
        if getattr(self, 'binding', None):
            self.native.mesh_cpu_program_release(self.binding)


class _Compiled:
    """A model share: its declared operation kind and typed constant arrays."""

    def __init__(self, kind, arrays):
        self.kind, self.arrays = kind, arrays


class Ane(Engine):
    """Core ML graph shares with explicit tile and channel block configuration.

    Compilation and model loading happen during preparation. Predictions bind
    shared input/output views and fail if Core ML does not honor output backing.
    """
    name, limits, occupies = 'ane', {'dtypes': (torch.float16, torch.bfloat16, torch.float32), 'constants': True}, frozenset({'ane'})

    def __init__(self, **options):
        super().__init__(**options)
        self.tile, self.block = int(options['tile']), int(options['block'])
        self.model = options['model']

    def input_dtype(self, dtype):
        return getattr(torch, self.model['input']['dtype'])

    def out_dtype(self, dtype):
        return getattr(torch, self.model['output']['dtype'])

    def sum_dtype(self):
        return getattr(torch, self.model['sum_dtype'])

    @classmethod
    def missing(cls):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            return 'coremltools is not installed (rdma/requirements.txt)'
        return None

    def prepare_linear(self, op, start, end):
        return _Compiled('linear', (op.weight[start:end].detach().to('cpu', getattr(torch, self.model['weight_dtype'])).numpy(),))

    def prepare_ffn(self, op, start, end):
        numpy = lambda t: t.detach().to('cpu', getattr(torch, self.model['weight_dtype'])).numpy()
        return _Compiled('ffn', (numpy(op.gate[start:end]), numpy(op.up[start:end]), numpy(op.down[:, start:end])))

    def declaration(self, op, shard, rows):
        compiled, _, _ = neural_engine_share(shard.kind, shard.arrays, rows, self.tile, self.block, self.model)
        source, target = self.model['input'], self.model['output']
        return dict(definition=dict(provider='coreml', compiled=str(compiled), units=self.model['units']), weights={},
                    input_layouts={source['name']: source['layout']}, output_layouts={target['name']: target['layout']},
                    binding_layout=self.model['binding_layout'])


class ModelView(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p), ('data', ctypes.c_void_p), ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)), ('rank', ctypes.c_int), ('dtype', ctypes.c_int)]


class Prediction:
    """One Core ML invocation's views, bound before the numerical function."""
    def __init__(self, native, model, source, target, *, input_layouts, output_layouts, binding_layout):
        from . import _stream
        from .operand import TensorView
        self.native = native
        source, target, before, after = _host_bindings(source, target, input_layouts, output_layouts, binding_layout)
        self.before, self.after = _stream.Commands(before), _stream.Commands(after)
        self.inputs, self.outputs = tuple(source.values()), tuple(target.values())
        def views(values):
            types = (torch.float16, torch.float32, torch.int32, torch.float64)
            records = []
            for name, original in values.items():
                operand = TensorView.of(original)
                if operand.dtype not in types:
                    raise ValueError('Core ML views require shared fp16, fp32, int32 or fp64 storage')
                rank = len(operand.shape)
                records.append(ModelView(name.encode(), _host_address(operand), (ctypes.c_int64 * rank)(*operand.shape),
                                         (ctypes.c_int64 * rank)(*operand.layout.strides), rank, types.index(operand.dtype)))
            return (ModelView * len(records))(*records)
        inputs, outputs = views(source), views(target)
        self.binding = native.mesh_coreml_bind(model, inputs, len(inputs), outputs, len(outputs))
        if not self.binding:
            raise RuntimeError(native.mesh_coreml_error().decode())

        self.entry = _native_entry(native.mesh_coreml_run, self.binding,
                                   native.mesh_coreml_retain, native.mesh_coreml_unbind)

    def __del__(self):
        if getattr(self, 'binding', None):
            self.native.mesh_coreml_unbind(self.binding)

    def append(self, schedule):
        schedule.append(self.before)
        schedule.host(self.entry, self.inputs, self.outputs)
        schedule.append(self.after)


def coreml_graph(compiled, inputs, outputs, *, units, input_layouts, output_layouts, binding_layout):
    """Bind any compiled Core ML graph to named shared views at preparation.

    ``units`` is the caller's backend operand (0 CPU/ANE, 1 all, 2 CPU, 3 CPU/GPU).
    Graph conversion belongs to preparation; no operation-kind dispatch or
    model loading occurs in the returned host record. The common schedule
    derives dependencies from the declared input and output views. Per-name
    ``input_layouts`` and ``output_layouts`` declare strided, contiguous, or
    exact element strides. ``binding_layout`` supplies the API representation
    requirement from the backend configuration; it is intersected with each
    use requirement. Necessary copies are recorded at preparation.
    """
    native = _coreml()
    if native is None:
        raise RuntimeError('Core ML graph requires the native shared-memory binding')
    handle = native.mesh_coreml_load(os.fsencode(compiled), units)
    if handle < 0:
        raise RuntimeError(native.mesh_coreml_error().decode())
    try:
        return Prediction(native, handle, inputs, outputs, input_layouts=input_layouts, output_layouts=output_layouts, binding_layout=binding_layout)
    finally:
        native.mesh_coreml_release(handle)


def export_coreml_graph(program, inputs, outputs, *, directory, units, deployment, precision, input_layouts, output_layouts, binding_layout):
    """Lower an exported tensor graph once, then bind the common host record.

    ``program`` is a torch.export.ExportedProgram (its graph and constants).
    The caller supplies names/views, CPU or ANE placement, target and precision.
    Conversion emits compiler artifacts in the supplied directory. It does not
    invoke the numerical graph, and the returned record does no file access.
    """
    import coremltools as ct
    native = _coreml()
    if native is None:
        raise RuntimeError('Core ML graph requires the native shared-memory binding')
    compute = coreml_units(units)
    model = ct.convert(program, convert_to='mlprogram', compute_units=compute,
                       minimum_deployment_target=getattr(ct.target, deployment),
                       compute_precision=getattr(ct.precision, precision), skip_model_load=True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    package, compiled = directory / 'graph.mlpackage', directory / 'graph.mlmodelc'
    model.save(str(package))
    try:
        if native.mesh_coreml_compile(os.fsencode(package), os.fsencode(compiled)):
            raise RuntimeError(native.mesh_coreml_error().decode())
    finally:
        shutil.rmtree(package)
    return coreml_graph(compiled, inputs, outputs, units=units, input_layouts=input_layouts, output_layouts=output_layouts, binding_layout=binding_layout)


def coreml_units(units):
    import coremltools as ct
    return {0: ct.ComputeUnit.CPU_AND_NE, 1: ct.ComputeUnit.ALL, 2: ct.ComputeUnit.CPU_ONLY, 3: ct.ComputeUnit.CPU_AND_GPU}[units]


def neural_engine_model(kind, arrays, rows, tile, block, configuration):
    """Prepare a declared linear/FFN share using configured Core ML operands.

    Tile and channel blocks choose convolution domains, including their tails.
    Input/output names and types, weight type, precision, deployment and compute
    units come from the same configuration consumed by the binding frontend.
    """
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types, Program, Function
    width = arrays[0].shape[1]
    units = arrays[0].shape[0]
    block = int(block)
    spans = [(b, min(b + block, units)) for b in range(0, units, block)]
    conv = lambda array: mb.const(val=np.ascontiguousarray(array, dtype=configuration['weight_dtype']).reshape(array.shape[0], array.shape[1], 1, 1))

    source, target = configuration['input'], configuration['output']
    kinds = {'float16': types.fp16, 'float32': types.fp32, 'int32': types.int32, 'float64': types.fp64}
    casts = {'float16': 'fp16', 'float32': 'fp32', 'int32': 'int32', 'float64': 'fp64'}
    program = Program()
    with Function({source['name']: mb.TensorSpec(shape=(rows, width), dtype=kinds[source['dtype']])},
                  opset_version=getattr(ct.target, configuration['deployment'])) as function:
        x = function.inputs[source['name']]
        if source['dtype'] != configuration['weight_dtype']:
            x = mb.cast(x=x, dtype=casts[configuration['weight_dtype']])
        if kind == 'linear':
            weights = [(conv(arrays[0][a:b]),) for a, b in spans]
        else:
            weights = [(conv(arrays[0][a:b]), conv(arrays[1][a:b]), conv(arrays[2][:, a:b])) for a, b in spans]
        channels = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, width, 1, rows])
        if rows <= tile:
            pieces = [channels]
        elif rows % tile:
            pieces = mb.split(x=channels, split_sizes=[tile] * (rows // tile) + [rows % tile], axis=3)
        else:
            pieces = mb.split(x=channels, num_splits=rows // tile, axis=3)
        outputs = []
        for p in pieces:
            if kind == 'linear':
                parts = [mb.conv(x=p, weight=w[0]) for w in weights]
                outputs.append(mb.concat(values=parts, axis=1) if len(parts) > 1 else parts[0])
            else:
                total = None
                for g, u, d in weights:
                    h = mb.mul(x=mb.gelu(x=mb.conv(x=p, weight=g), mode='TANH_APPROXIMATION'), y=mb.conv(x=p, weight=u))
                    part = mb.conv(x=h, weight=d)
                    total = part if total is None else mb.add(x=total, y=part)
                outputs.append(total)
        y = mb.concat(values=outputs, axis=3) if len(outputs) > 1 else outputs[0]
        n = arrays[0].shape[0] if kind == 'linear' else arrays[2].shape[0]
        y = mb.transpose(x=mb.reshape(x=y, shape=[n, rows]), perm=[1, 0])
        y = mb.cast(x=y, dtype=casts[target['dtype']], name=target['name']) if configuration['weight_dtype'] != target['dtype'] else mb.identity(x=y, name=target['name'])
        function.set_outputs([y])
    program.add_function('main', function)
    return ct.convert(program, convert_to='mlprogram', compute_units=coreml_units(configuration['units']),
                      compute_precision=getattr(ct.precision, configuration['precision']),
                      minimum_deployment_target=getattr(ct.target, configuration['deployment']), skip_model_load=True)


def neural_engine_share(kind, arrays, rows, tile, block, configuration):
    """Compile a configured model share and retain its provider placement.

    Names, types, row/channel tiles, precision, target, compute units and allowed
    devices are compiler operands. Artifacts are reused by those operands and
    constants. Provider placement is checked during preparation, before binding.
    """
    import collections
    import coremltools as ct
    import numpy as np
    from coremltools.models.compute_plan import MLComputePlan
    cache = Path(os.path.expanduser('~/.cache/mesh-engines'))
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays) + json.dumps([kind, rows, tile, block, configuration], sort_keys=True).encode()).hexdigest()
    native = _coreml()
    if native is None:
        raise RuntimeError('Neural Engine share: rdma/libmesh-coreml.dylib does not load')
    compiled = cache / f'{digest}-b{block}.mlmodelc'
    if not compiled.exists():
        package = cache / f'{digest}-b{block}.mlpackage'
        neural_engine_model(kind, arrays, rows, tile, block, configuration).save(str(package))
        failed = native.mesh_coreml_compile(str(package).encode(), str(compiled).encode())
        shutil.rmtree(package, ignore_errors=True)
        if failed:
            raise RuntimeError(f'Neural Engine share: {native.mesh_coreml_error().decode()}')
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=coreml_units(configuration['units']))
    devices = collections.Counter()
    for op in plan.model_structure.program.functions['main'].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            name = type(usage.preferred_compute_device).__name__
            devices['ane' if 'NeuralEngine' in name else 'gpu' if 'GPU' in name else 'cpu' if 'CPU' in name else 'unknown'] += 1
    outside = set(devices) - set(configuration['devices'])
    if outside:
        raise ValueError(f"model placement {dict(devices)} exceeds declared devices {configuration['devices']}")
    return compiled, block, dict(devices)


ENGINES = {e.name: e for e in (Mps, Cpu, Ane)}


class Engines:
    """The engines one node's operations may share their units among, in order (the order of their shares along an
    operation's units), each with its options; the evidence of the operations using them; each decision's search."""

    def __init__(self, names, **options):
        from mesh_compute import hardware
        self.hardware = hardware()
        names = [names] if isinstance(names, str) else list(names)
        unknown = [n for n in names if n not in ENGINES]
        if unknown:
            raise ValueError(f'engines: {unknown} are none of {sorted(ENGINES)}')
        lacking = {n: ENGINES[n].missing() for n in names if ENGINES[n].missing()}
        if lacking:
            raise RuntimeError(f'engines: this node lacks {lacking}')
        self.names = names
        self.engines = [ENGINES[n](**options.get(n, {})) for n in names]
        self.operations, self.decisions = [], {}
        self.parameters = options.get('compilation', {})


class Operation:
    """An operation declaration. Shares and shapes are configuration operands.

    ``prepare`` binds an input, its final output and every engine's scratch and
    model. The returned call uses those bindings; it never chooses or loads a
    shard. Placement changes require preparing the next whole function.
    """
    kind = combine = None

    def __init__(self, units, width, engines, name, grain, constant):
        self.units, self.width, self.engines, self.grain, self.constant = units, width, engines, grain, constant
        self.name = name or f'{self.kind}{id(self):x}'
        if engines is not None:
            engines.operations.append(self)

    def declare(self, rows, dtype, *, capacities):
        """Work rates for this operation and shape, in the configured engine order.

        Rates are supplied operands, not CPU/GPU/ANE core counts. Whole-call
        observations refine them through the same assignment solver as routes.
        A zero explicitly excludes an engine from this declaration.
        """
        key = f'{self.name}@{rows}'
        total = sum(capacities)
        if (len(capacities) != len(self.engines.engines) or not all(math.isfinite(c) and c >= 0 for c in capacities)
                or not math.isfinite(total) or total <= 0):
            raise ValueError(f'{key}: supply a nonnegative work rate per engine, with a positive total')
        able = [c > 0 for c in capacities]
        for engine, active in zip(self.engines.engines, able):
            if active and (not engine.takes(self, dtype) or (engine.limits['constants'] and not self.constant)):
                raise ValueError(f'{key}: configured engine {engine.name} cannot take this operation at {dtype}')
        from assignment import project
        parts = project([self.units * (c / total) for c in capacities], self.units, self.grain,
                        [0] * len(able), [self.units if a else 0 for a in able])
        self.engines.decisions[key] = {'parts': parts, 'grain': self.grain, 'low': [0] * len(able),
                                      'high': [self.units if a else 0 for a in able], 'units': self.units,
                                      'capacity': list(capacities)}
        return key

    def prepare(self, x, *, parts, output=None):
        return Prepared(self, x, parts, output)

    def finish(self, y, first=0):
        return y


class Prepared:
    """One declared graph for all engine shares, casts and their consumers."""
    def __init__(self, op, x, parts, output=None):
        self.op, self.x = op, x
        rows, dtype = x.shape[0], x.dtype
        if len(parts) != len(op.engines.engines) or sum(parts) != op.units or min(parts) < 0:
            raise ValueError('engine shares must partition the declared units')
        from . import empty
        self.output = empty((rows, op.width), dtype=dtype) if output is None else output
        if self.output.shape != (rows, op.width) or self.output.dtype != dtype:
            raise ValueError('output must match the declared shape and dtype')
        output_device = self.output.device
        configuration = {k: p['value'] for k, p in op.engines.parameters.items()}
        shares, start = [], 0
        sum_types = [engine.sum_dtype() for engine, count in zip(op.engines.engines, parts) if count]
        from functools import reduce
        accumulator = reduce(torch.promote_types, sum_types, dtype)
        for engine, count in zip(op.engines.engines, parts):
            end = start + count
            if count:
                if not engine.takes(op, dtype) or (engine.limits['constants'] and not op.constant):
                    raise ValueError(f'{engine.name} cannot execute the declared operation')
                shard = engine.prepare(op, start, end)
                declaration = engine.declaration(op, shard, rows) if engine.host_side else None
                shares.append((engine, shard, start, end, declaration))
            start = end

        from .host import declare
        def function(value):
            sources, pieces = {dtype: value}, []
            for engine, shard, first, end, declaration in shares:
                needed = engine.input_dtype(dtype)
                if needed not in sources:
                    sources[needed] = value.to(dtype=needed, device=output_device)
                source = sources[needed]
                shape = (rows, end - first if op.combine == 'concat' else op.width)
                out = torch.empty(shape, dtype=engine.out_dtype(dtype), device=output_device)
                if declaration is None:
                    engine.run(op, shard, source, out)
                else:
                    source_name, = [name for name in declaration['input_layouts'] if name not in declaration['weights']]
                    target_name, = declaration['output_layouts']
                    out = declare({source_name: source, **declaration['weights']}, {target_name: out}, declaration['definition'],
                                  input_layouts=declaration['input_layouts'], output_layouts=declaration['output_layouts'],
                                  binding_layout=declaration['binding_layout'])[target_name]
                pieces.append(out)
            if op.combine == 'concat':
                return tuple(op.finish(piece.to(dtype), first) for piece, (_, _, first, _, _) in zip(pieces, shares))
            y = pieces[0].to(accumulator) if pieces else torch.zeros((rows, op.width), dtype=accumulator, device=output_device)
            for piece in pieces[1:]:
                y = y + piece.to(accumulator)
            return op.finish(y.to(dtype))

        from .declarations import Function
        from .operand import TensorView
        destinations = ([TensorView.of(self.output).slice(1, first, end - first) for _, _, first, end, _ in shares]
                        if op.combine == 'concat' else [self.output])
        self.function = Function(function, x, configuration=configuration)
        self.function.declare(None, destinations=destinations)

    def append(self, schedule):
        self.function.append(schedule)


class Linear(Operation):
    """y = x W^T (+ b), W [N, K]: its N outputs shared (they concatenate)."""
    kind, combine = 'linear', 'concat'

    def __init__(self, weight, bias=None, engines=None, name=None, grain=64, constant=True):
        self.weight, self.bias = weight, bias
        super().__init__(weight.shape[0], weight.shape[0], engines, name, grain, constant)

    def torch(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def finish(self, y, first=0):
        return y if self.bias is None else y + self.bias.narrow(0, first, y.shape[-1])


class FFN(Operation):
    """y = down(gelu_tanh(x Wg^T) * (x Wu^T)), Wg and Wu [I, H], Wd [H, I]: its I intermediate neurons shared, each
    engine's share a partial of the whole output (they sum): Megatron's column-then-row split [Shoeybi et al. 2019].
    The operands are any tensors: coded matrices (torch_mesh.streamed) share their neurons as their codes' rows and
    columns (grain a multiple of 32), the GPU's share their fused couple, another engine's share decoded."""
    kind, combine = 'ffn', 'sum'

    def __init__(self, gate, up, down, engines=None, name=None, grain=128, constant=True):
        self.gate, self.up, self.down = gate, up, down
        super().__init__(gate.shape[0], down.shape[0], engines, name, grain, constant)

    def torch(self, x):
        return ffn(x.reshape(-1, x.shape[-1]), self.gate, self.up, self.down).reshape(*x.shape[:-1], -1)


def placement(engines, observations, *, step):
    # design/whole-function.md#numerical-operands
    from assignment import assign, domains
    decisions = {k: {field: d[field] for field in ('parts', 'grain', 'low', 'high', 'capacity')}
                 for k, d in engines.decisions.items()}
    assigned = assign(domains(decisions, {}, engines.parameters), observations, step=step)
    for key, parts in assigned['parts'].items():
        if key in engines.decisions:
            engines.decisions[key]['parts'] = parts
    for key, value in assigned['operands'].items():
        engines.parameters[key]['value'] = value
    return assigned
