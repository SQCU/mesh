"""Compute-share declarations for the common whole-function compiler.

Declare shapes and choose shares before execution. ``Operation.prepare`` binds
weights, outputs and host views. GPU spans use ``declarations.Function``; host
shares use the native recorder's host producer/consumer records. Only result
consumers depend on host completion. No operation times itself, chooses shares,
loads a model, or joins a future while it runs.
"""
import ctypes
import hashlib
import math
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
            lib.mesh_coreml_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
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

    def out_dtype(self, dtype):
        return dtype

    def prepare(self, op, start, end):
        return getattr(self, 'prepare_' + op.kind)(op, start, end)

    def run(self, op, shard, x, prologue, out):
        return getattr(self, 'run_' + op.kind)(op, shard, x, prologue, out)


class Mps(Engine):
    """GPU products declared to the whole-function compiler."""
    name, host_side, occupies = 'mps', False, frozenset({'gpu'})

    @classmethod
    def missing(cls):
        return None if torch.backends.mps.is_available() else 'no MPS device'

    def prepare_linear(self, op, start, end):
        return op.weight[start:end]

    def run_linear(self, op, shard, x, prologue, out):
        torch.mm(x, shard.T, out=out)

    def prepare_ffn(self, op, start, end):
        return op.gate[start:end], op.up[start:end], op.down[:, start:end].contiguous()

    def run_ffn(self, op, shard, x, prologue, out):
        ffn(x, *shard, out=out)


class Cpu(Engine):
    """The CPU's matrix units through prepared Accelerate commands: the input read in place as fp32 (the
    GPU's prologue converts it), the fp32 shard resident in host memory, the output written in place in fp32."""
    name, occupies = 'cpu', frozenset({'cpu'})

    def out_dtype(self, dtype):
        return torch.float32

    def prepare_linear(self, op, start, end):
        return op.weight[start:end].detach().float().cpu().contiguous()

    def prepare_ffn(self, op, start, end):
        cpu = lambda t: t.detach().float().cpu().contiguous()
        return cpu(op.gate[start:end]), cpu(op.up[start:end]), cpu(op.down[:, start:end])

    def bind(self, op, shard, x, prologue, out):
        source, target = host(prologue['fp32']), host(out)
        native = _coreml()
        if native is None:
            raise RuntimeError('CPU commands require the native engine library')
        def product(x, w, y):
            command = CpuCommand(None,
                              x.shape[0], w.shape[0], x.shape[1], x.stride(0), w.stride(0), y.stride(0),
                              x.stride(1), w.stride(1), y.stride(1),
                              x.data_ptr(), w.data_ptr(), y.data_ptr(), None)
            if native.mesh_cpu_product_bind(ctypes.byref(command)):
                raise ValueError('CPU matrix views could not be bound to a bulk contraction')
            return command
        if op.kind == 'linear':
            return CpuProgram([product(source, shard, target)], (prologue['fp32'], shard), (out,))
        gate, up, down = shard
        g = torch.empty(source.shape[0], gate.shape[0], dtype=torch.float32)
        u, scratch = torch.empty_like(g), torch.empty_like(g)
        activation = CpuCommand(ctypes.cast(native.mesh_cpu_gelu, ctypes.c_void_p), g.shape[0], g.shape[1], 0, 0, 0, 0, 0, 0, 0,
                                 g.data_ptr(), u.data_ptr(), g.data_ptr(), scratch.data_ptr())
        return CpuProgram([product(source, gate, g), product(source, up, u), activation, product(g, down, target)],
                          (prologue['fp32'], *shard), (out, g, u, scratch))


class CpuCommand(ctypes.Structure):
    _fields_ = [('run', ctypes.c_void_p)] + [(name, ctypes.c_int) for name in ('rows', 'columns', 'inner', 'input_stride', 'weight_stride', 'output_stride',
                                                                          'input_column', 'weight_column', 'output_column')] \
             + [(name, ctypes.c_void_p) for name in ('input', 'weight', 'output', 'scratch')]
    _fields_ += [(name, ctypes.c_int) for name in ('input_transpose', 'weight_transpose')]


def _native_entry(run, binding, retain, release):
    return (ctypes.cast(run, ctypes.c_void_p).value, binding,
            ctypes.cast(retain, ctypes.c_void_p).value, ctypes.cast(release, ctypes.c_void_p).value)


class CpuProgram:
    def __init__(self, commands, inputs, outputs):
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
        self.entry = _native_entry(self.native.mesh_cpu_program_run, self.binding,
                                   self.native.mesh_cpu_program_retain, self.native.mesh_cpu_program_release)

    def append(self, schedule):
        schedule.host(self.entry, self.inputs, self.outputs)

    def __del__(self):
        if getattr(self, 'binding', None):
            self.native.mesh_cpu_program_release(self.binding)


class _Compiled:
    """A Neural Engine share: its operation kind, its fp16 weights and their compiled models by row count."""

    def __init__(self, kind, arrays):
        self.kind, self.arrays = kind, arrays


class Ane(Engine):
    """Core ML graph shares with explicit tile and channel block configuration.

    Compilation and model loading happen during preparation. Predictions bind
    shared input/output views and fail if Core ML does not honor output backing.
    """
    name, limits, occupies = 'ane', {'dtypes': (torch.float16,), 'constants': True}, frozenset({'ane'})

    def __init__(self, **options):
        super().__init__(**options)
        self.tile, self.block = int(options['tile']), int(options['block'])

    @classmethod
    def missing(cls):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            return 'coremltools is not installed (rdma/requirements.txt)'
        return None

    def prepare_linear(self, op, start, end):
        return _Compiled('linear', (op.weight[start:end].detach().to('cpu', torch.float16).numpy(),))

    def prepare_ffn(self, op, start, end):
        numpy = lambda t: t.detach().to('cpu', torch.float16).contiguous().numpy()
        return _Compiled('ffn', (numpy(op.gate[start:end]), numpy(op.up[start:end]), numpy(op.down[:, start:end])))

    def bind(self, op, shard, x, prologue, out):
        native = _coreml()
        if native is None:
            raise RuntimeError('Neural Engine requires the native shared-memory binding')
        rows = x.shape[0]
        compiled, shard.block, _ = neural_engine_share(shard.kind, shard.arrays, rows, self.tile, self.block)
        handle = native.mesh_coreml_load(str(compiled).encode(), b'', b'', 0)
        if handle < 0:
            raise RuntimeError(native.mesh_coreml_error().decode())
        return Prediction(native, handle, x, out)


class ModelView(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p), ('data', ctypes.c_void_p), ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)), ('rank', ctypes.c_int), ('dtype', ctypes.c_int)]


class Prediction:
    """One Core ML invocation's views, bound before the numerical function."""
    def __init__(self, native, model, source, target):
        self.native = native
        source = source if isinstance(source, dict) else {'': source}
        target = target if isinstance(target, dict) else {'': target}
        self.inputs, self.outputs = tuple(source.values()), tuple(target.values())
        # Empty names select the load declaration's names; general graphs pass
        # an explicit mapping for every input and output.
        def views(values):
            types = (torch.float16, torch.float32, torch.int32, torch.float64)
            records = []
            for name, original in values.items():
                tensor = host(original)
                if tensor is None or tensor.dtype not in types:
                    raise ValueError('Core ML views require shared fp16, fp32, int32 or fp64 storage')
                rank = tensor.ndim
                records.append(ModelView(name.encode(), tensor.data_ptr(), (ctypes.c_int64 * rank)(*tensor.shape),
                                         (ctypes.c_int64 * rank)(*tensor.stride()), rank, types.index(tensor.dtype)))
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
        schedule.host(self.entry, self.inputs, self.outputs)


def coreml_graph(compiled, inputs, outputs, *, units):
    """Bind any compiled Core ML graph to named shared views at preparation.

    ``units`` is the caller's backend operand (0 CPU/ANE, 1 all, 2 CPU).
    Graph conversion belongs to preparation; no operation-kind dispatch or
    model loading occurs in the returned host record. The common schedule
    derives dependencies from the declared input and output views.
    """
    native = _coreml()
    if native is None:
        raise RuntimeError('Core ML graph requires the native shared-memory binding')
    handle = native.mesh_coreml_load(os.fsencode(compiled), b'', b'', units)
    if handle < 0:
        raise RuntimeError(native.mesh_coreml_error().decode())
    return Prediction(native, handle, inputs, outputs)


def export_coreml_graph(program, inputs, outputs, *, directory, units, deployment, precision):
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
    compute = {0: ct.ComputeUnit.CPU_AND_NE, 1: ct.ComputeUnit.ALL, 2: ct.ComputeUnit.CPU_ONLY}[units]
    model = ct.convert(program, convert_to='mlprogram', compute_units=compute,
                       minimum_deployment_target=getattr(ct.target, deployment),
                       compute_precision=getattr(ct.precision, precision))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    package, compiled = directory / 'graph.mlpackage', directory / 'graph.mlmodelc'
    model.save(str(package))
    try:
        if native.mesh_coreml_compile(os.fsencode(package), os.fsencode(compiled)):
            raise RuntimeError(native.mesh_coreml_error().decode())
    finally:
        shutil.rmtree(package)
    return coreml_graph(compiled, inputs, outputs, units=units)


def neural_engine_model(kind, arrays, rows, tile=128, block=None):
    """A Neural Engine share as a Core ML program (coremltools; fp16): `kind` 'linear' (arrays: W [N, K]) or 'ffn'
    (arrays: Wg, Wu [I, K], Wd [K, I]: down(gelu_tanh(x Wg^T) * (x Wu^T))), its input x [rows, K] and output [rows, N or
    K] row-major, the transposes to and from channels-first inside the graph, rows in tiles of `tile` (the last the
    rows' remainder), each tile one 1x1 convolution a matrix with its weights constant, the units (N outputs, I
    neurons) in blocks of at most `block` (each block its own convolutions; a linear's blocks concatenated, an FFN's
    down partials summed), so no convolution's weights pass what the Neural Engine's compiler takes
    (neural_engine_block).  The one statement of the graph: this
    module's Ane engine and the metal-microbench engine's prepared shares (tools/mesh/neural_engine.py) both build it
    here."""
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    width = arrays[0].shape[1]
    units = arrays[0].shape[0]
    block = units if not block or block >= units else int(block)
    spans = [(b, min(b + block, units)) for b in range(0, units, block)]
    conv = lambda array: mb.const(val=np.ascontiguousarray(array).reshape(array.shape[0], array.shape[1], 1, 1))

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, width), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def program(x):
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
        return mb.transpose(x=mb.reshape(x=y, shape=[n, rows]), perm=[1, 0])
    return ct.convert(program, convert_to='mlprogram', compute_units=ct.ComputeUnit.CPU_AND_NE,
                      compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)


def neural_engine_share(kind, arrays, rows, tile, block):
    """A Neural Engine share in convolutions of `block` units (the caller's configuration): built (neural_engine_model),
    compiled once into ~/.cache/mesh-engines by content (the arrays, kind, rows, tile and block), and its compute plan
    read (coremltools MLComputePlan).  The Neural Engine's compiler refuses a convolution past a weight size (on an M4 Pro
    between 7680 and 8192 units of a 1536-wide input in fp16: 22.5 to 24 MiB), and Core ML then runs the whole program
    on the CPU, 20 to 60 times slower: the devices returned say where its operations run, and a caller whose share
    places elsewhere fails it.  The search that halved the block and built again until it placed, and its store of the
    largest block a node placed (~/.cache/mesh-engines/blocks.json), were deleted 2026-10-22 (no retry: the block is a
    stated fact of the configuration).  (compiled .mlmodelc path, block, its operations' devices)."""
    import collections
    import coremltools as ct
    import numpy as np
    from coremltools.models.compute_plan import MLComputePlan
    cache = Path(os.path.expanduser('~/.cache/mesh-engines'))
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays) + f'{kind} {rows} {tile}'.encode()).hexdigest()
    native = _coreml()
    if native is None:
        raise RuntimeError('Neural Engine share: rdma/libmesh-coreml.dylib does not load')
    compiled = cache / f'{digest}-b{block}.mlmodelc'
    if not compiled.exists():
        package = cache / f'{digest}-b{block}.mlpackage'
        neural_engine_model(kind, arrays, rows, tile, block).save(str(package))
        failed = native.mesh_coreml_compile(str(package).encode(), str(compiled).encode())
        shutil.rmtree(package, ignore_errors=True)
        if failed:
            raise RuntimeError(f'Neural Engine share: {native.mesh_coreml_error().decode()}')
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    devices = collections.Counter()
    for op in plan.model_structure.program.functions['main'].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            name = type(usage.preferred_compute_device).__name__
            devices['ane' if 'NeuralEngine' in name else 'gpu' if 'GPU' in name else 'cpu' if 'CPU' in name else 'unknown'] += 1
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


def _gpu(function, configuration=None):
    from .declarations import Function
    made = Function(function, configuration=configuration)
    made.declare(None)
    return made


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

    def finish(self, y):
        return y


class Prepared:
    """Fixed engine operands and data events for the declared invocations."""
    def __init__(self, op, x, parts, output=None):
        self.op, self.x = op, x
        rows, dtype = x.shape[0], x.dtype
        if len(parts) != len(op.engines.engines) or sum(parts) != op.units or min(parts) < 0:
            raise ValueError('engine shares must partition the declared units')
        from . import empty
        self.output = empty((rows, op.width), dtype=dtype) if output is None else output
        if self.output.shape != (rows, op.width) or self.output.dtype != dtype:
            raise ValueError('output must match the declared shape and dtype')
        prologue = {}
        configuration = {k: p['value'] for k, p in op.engines.parameters.items()}
        self.prologue = None
        if any(count and engine.name == 'cpu' for engine, count in zip(op.engines.engines, parts)):
            prologue['fp32'] = x if dtype == torch.float32 else empty(tuple(x.shape), dtype=torch.float32)
            if dtype != torch.float32:
                self.prologue = _gpu(lambda: prologue['fp32'].copy_(x), configuration)
        self.gpu, self.host = [], []
        pieces, conversions, start = [], [], 0
        for engine, count in zip(op.engines.engines, parts):
            end = start + count
            if count:
                if not engine.takes(op, dtype) or (engine.limits['constants'] and not op.constant):
                    raise ValueError(f'{engine.name} cannot execute the declared operation')
                shard = engine.prepare(op, start, end)
                shape = (rows, count if op.combine == 'concat' else op.width)
                destination = self.output[:, start:end] if op.combine == 'concat' else self.output
                direct = engine.out_dtype(dtype) == dtype and (op.combine == 'concat' or not pieces)
                out = destination if direct else empty(shape, dtype=engine.out_dtype(dtype))
                pieces.append(out)
                if op.combine == 'concat' and not direct:
                    conversions.append((destination, out))
                if engine.host_side:
                    bound = engine.bind(op, shard, x, prologue, out)
                    self.host.append(bound)
                else:
                    self.gpu.append(_gpu(lambda e=engine, w=shard, y=out: e.run(op, w, x, prologue, y), configuration))
            start = end
        def combine():
            if op.combine == 'concat':
                for destination, source in conversions:
                    destination.copy_(source)
                y = self.output
            else:
                y = pieces[0].float()
                for p in pieces[1:]:
                    y = y + p.float()
                y = y.to(dtype)
            return self.output.copy_(op.finish(y))
        needs_combine = (op.combine == 'sum' and (len(pieces) > 1 or pieces[0] is not self.output)
                         or bool(conversions) or getattr(op, 'bias', None) is not None)
        self.combine = _gpu(combine, configuration) if needs_combine else None

    def append(self, schedule):
        if self.prologue is not None:
            self.prologue.append(schedule)
        for function in self.host:
            function.append(schedule)
        for function in self.gpu:
            function.append(schedule)
        if self.combine is not None:
            self.combine.append(schedule)


class Linear(Operation):
    """y = x W^T (+ b), W [N, K]: its N outputs shared (they concatenate)."""
    kind, combine = 'linear', 'concat'

    def __init__(self, weight, bias=None, engines=None, name=None, grain=64, constant=True):
        self.weight, self.bias = weight, bias
        super().__init__(weight.shape[0], weight.shape[0], engines, name, grain, constant)

    def torch(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def finish(self, y):
        return y if self.bias is None else y + self.bias


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


def placement(engines, observations):
    # design/whole-function.md#numerical-operands
    from assignment import assign, domains
    decisions = {k: {field: d[field] for field in ('parts', 'grain', 'low', 'high', 'capacity')}
                 for k, d in engines.decisions.items()}
    assigned = assign(domains(decisions, {}, engines.parameters), observations)
    for key, parts in assigned['parts'].items():
        if key in engines.decisions:
            engines.decisions[key]['parts'] = parts
    for key, value in assigned['operands'].items():
        engines.parameters[key]['value'] = value
    return assigned
