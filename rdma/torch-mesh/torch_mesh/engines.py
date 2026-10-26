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
    if t.device.type != 'mps' or not t.is_contiguous():
        return None
    buffer = t.untyped_storage().data_ptr()
    if _send(buffer, b'storageMode', ctypes.c_ulong) != 0:
        return None
    base = _send(buffer, b'contents')
    nbytes = t.numel() * t.element_size()
    raw = (ctypes.c_uint8 * nbytes).from_address(base + t.storage_offset() * t.element_size())
    return torch.frombuffer(raw, dtype=t.dtype, count=t.numel()).view(t.shape) if nbytes else torch.empty(t.shape, dtype=t.dtype)


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
            lib.mesh_coreml_bind.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
                                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
            lib.mesh_coreml_run.restype, lib.mesh_coreml_run.argtypes = ctypes.c_int, [ctypes.c_void_p]
            lib.mesh_coreml_unbind.argtypes = [ctypes.c_void_p]
            lib.mesh_coreml_error.restype = ctypes.c_char_p
            lib.mesh_coreml_placement.restype = ctypes.c_long
            lib.mesh_coreml_placement.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_size_t]
            lib.mesh_coreml_compile.restype = ctypes.c_int
            lib.mesh_coreml_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
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
    """The CPU's matrix units through torch's fp32 CPU products (Accelerate): the input read in place as fp32 (the
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
        if op.kind == 'linear':
            weight = shard.T
            return lambda: torch.mm(source, weight, out=target)
        gate, up, down = (w.T for w in shard)
        g = torch.empty(source.shape[0], gate.shape[1], dtype=torch.float32)
        u = torch.empty_like(g)
        def run():
            torch.mm(source, gate, out=g)
            torch.mm(source, up, out=u)
            torch.ops.aten.gelu.out(g, approximate='tanh', out=g)
            torch.mul(g, u, out=g)
            torch.mm(g, down, out=target)
        return run


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
        return Prediction(native, handle, host(x), host(out))


class Prediction:
    """One Core ML invocation's views, bound before the numerical function."""
    def __init__(self, native, model, source, target):
        self.native, self.source, self.target = native, source, target
        shape_in, shape_out = (ctypes.c_int64 * 2)(*source.shape), (ctypes.c_int64 * 2)(*target.shape)
        self.binding = native.mesh_coreml_bind(model, source.data_ptr(), shape_in, 2, target.data_ptr(), shape_out, 2)
        if not self.binding:
            raise RuntimeError(native.mesh_coreml_error().decode())

    def __call__(self):
        if self.native.mesh_coreml_run(self.binding):
            raise RuntimeError(self.native.mesh_coreml_error().decode())

    def __del__(self):
        if getattr(self, 'binding', None):
            self.native.mesh_coreml_unbind(self.binding)


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


def _gpu(function):
    from .declarations import Function
    made = Function(function)
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

    def prepare(self, x, *, parts):
        return Prepared(self, x, parts)

    def finish(self, y):
        return y


class Prepared:
    """Fixed engine operands and data events for the declared invocations."""
    def __init__(self, op, x, parts):
        self.op, self.x = op, x
        rows, dtype = x.shape[0], x.dtype
        if len(parts) != len(op.engines.engines) or sum(parts) != op.units or min(parts) < 0:
            raise ValueError('engine shares must partition the declared units')
        self.output = torch.empty(rows, op.width, device=x.device, dtype=dtype)
        prologue = {}
        self.prologue = None
        if any(count and engine.name == 'cpu' for engine, count in zip(op.engines.engines, parts)):
            prologue['fp32'] = torch.empty_like(x, dtype=torch.float32)
            self.prologue = _gpu(lambda: prologue['fp32'].copy_(x))
        self.gpu, self.host = [], []
        pieces, start = [], 0
        for engine, count in zip(op.engines.engines, parts):
            end = start + count
            if count:
                if not engine.takes(op, dtype) or (engine.limits['constants'] and not op.constant):
                    raise ValueError(f'{engine.name} cannot execute the declared operation')
                shard = engine.prepare(op, start, end)
                shape = (rows, count if op.combine == 'concat' else op.width)
                out = torch.empty(shape, device=x.device, dtype=engine.out_dtype(dtype))
                pieces.append(out)
                if engine.host_side:
                    bound = engine.bind(op, shard, x, prologue, out)
                    self.host.append(bound)
                else:
                    self.gpu.append(_gpu(lambda e=engine, w=shard, y=out: e.run(op, w, x, prologue, y)))
            start = end
        def combine():
            if op.combine == 'concat':
                y = torch.cat([p.to(dtype) for p in pieces], dim=1)
            else:
                y = pieces[0].float()
                for p in pieces[1:]:
                    y = y + p.float()
                y = y.to(dtype)
            return self.output.copy_(op.finish(y))
        self.combine = _gpu(combine)

    def append(self, schedule):
        if self.prologue is not None:
            self.prologue.append(schedule)
        results = [schedule.host(function) for function in self.host]
        for function in self.gpu:
            function.append(schedule)
        for result in results:
            schedule.consume(result)
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
    from assignment import assign
    decisions = {k: {field: d[field] for field in ('parts', 'grain', 'low', 'high', 'capacity')}
                 for k, d in engines.decisions.items()}
    assigned = assign(decisions, observations)
    for key, parts in assigned['parts'].items():
        engines.decisions[key]['parts'] = parts
    return assigned
