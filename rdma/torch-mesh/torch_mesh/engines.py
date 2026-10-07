"""A node's compute engines as members of one tensor operation: its units shared among them, solved jointly on the
calls' own evidence (design/heterogeneity.md: raggedness within a node is the raggedness between nodes; metal-microbench
docs/soc_compute_backends.md for the engines' measured envelopes).

An Apple SoC holds several matrix engines with their own rates by shape, dtype and operand form: the GPU (torch's MPS
queue; on an M4 Pro about 6 TFLOP/s at fp16 and fp32 alike, on an M5 its neural accelerators), the CPU's matrix units
(SME/AMX, one a cluster, reached through Accelerate by torch's fp32 CPU products: about 3 TFLOP/s on an M4 Pro's two
performance clusters) and the Neural Engine (Core ML's static graphs of 1x1 convolutions with constant weights, rows
in tiles).  No one engine is fastest for every operation, the fastest set changes with the shape, and engines running
at once slow each other (power, performance cores and memory shared), so an operation's final backend is resolved
here, per operation and row count, as one joint decision: its units shared among the configured engines like a mesh
dimension among ranks, each engine holding its shard of the weights resident, the shares running at once on unified
memory, and rdma/allocate.py `Coupled` choosing the allocation that minimises the call's time under a model of every
engine's time as a function of every engine's share (its own work, slowed by its co-runners'), fitted to every
allocation measured and minimised over the whole lattice of allocations.  A share of nothing leaves that engine out:
the resolution can be one engine.

An operation states how its shares combine, the mesh's crossing rule for its split axis (tools/mesh/programs.py
points): `Linear` (y = x W^T + b) shares its output features, which concatenate; `FFN` (y = down(act(x Wg^T) *
(x Wu^T)), Megatron's column-then-row split) shares its intermediate neurons, each engine giving a partial of the whole
output, which sum.  Each engine lowers each kind its own way (the Neural Engine an FFN share as one fused graph).

A call that prepares a share (copies a shard, compiles and loads a Neural Engine model) or first runs it at its row
count (torch's MPS builds a graph a shape; Core ML's first predictions after a load are slow) is no evidence.  A call is
measured by timing events on torch's stream (its span, and the GPU's share where others run beside it), each about 0.1
ms of host time, so once a decision's allocation stands one call in `sample` (default 16) is measured, enough to see its
time move.

Engines never time-share a physical unit.  Each states what it occupies (`Engine.occupies`: the GPU; the CPU's cores;
the Neural Engine, and whatever units Core ML places a Neural Engine share's operations on, read from its compute
plan), and two engines occupying one unit never hold units of a decision together (allocate.Coupled.forbid): running
both at once would time-share it (two APIs on the same arithmetic units, contending for registers and switching
between them) rather than add capacity.

Nothing is implicit.  An operation takes its engines as an operand (`Engines`), from the program or from the
configuration the program names (`Engines.configured()`: MESH_ENGINES, e.g. "mps,cpu,ane:tile=128"); without
one it is torch's own operation on its input's device.  An engine the configuration names and the node lacks is an
error, not a silent omission.  An engine's limits are data (`Engine.limits`): its operand dtypes, whether its
weights must be constants (the Neural Engine's: a share there is compiled), and the row counts it takes (the Neural
Engine's whole tiles); a row count an engine cannot take gives it no units for that decision.

The data path is zero-copy where the memory allows: torch's MPS tensors live in shared Metal memory, so the CPU and
Neural Engine shares read their inputs in place (the GPU's prologue converts the input to fp32 for the CPU's products;
the Neural Engine's graph reads it row-major as it is) and write their outputs into MPS tensors in place; the GPU
combines the shares.  The Neural Engine's binding through coremltools copies its input and output once each (counted
in its time).

  pool = engines.Engines.configured()
  ffn = engines.FFN(gate, up, down, engines=pool, name='layer3.ffn')
  y = ffn(x)                # x [..., H] on MPS: every configured engine its share of the neurons
  engines.rebalance(pool)   # the joint solver on the calls since the last: each (operation, rows)'s allocation"""
import concurrent.futures
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import time
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
    """The native Core ML binding (rdma/mesh-coreml.m: predictions on caller memory, no copies on the CPU), built where
    missing; None where it cannot load."""
    global _CORE_ML
    if _CORE_ML is None:
        rdma = Path(__file__).resolve().parents[2]
        path = rdma / 'libmesh-coreml.dylib'
        try:
            if not path.exists():
                subprocess.run(['make', '-s', '-C', str(rdma), 'libmesh-coreml.dylib'], check=True)
            lib = ctypes.CDLL(str(path))
            lib.mesh_coreml_load.restype = ctypes.c_long
            lib.mesh_coreml_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            lib.mesh_coreml_predict.restype = ctypes.c_int
            lib.mesh_coreml_predict.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
                                                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
            lib.mesh_coreml_error.restype = ctypes.c_char_p
            lib.mesh_coreml_placement.restype = ctypes.c_long
            lib.mesh_coreml_placement.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_size_t]
            lib.mesh_coreml_compile.restype = ctypes.c_int
            lib.mesh_coreml_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            _CORE_ML = lib
        except (OSError, subprocess.CalledProcessError):
            _CORE_ML = False
    return _CORE_ML or None


def _gelu(x):
    return torch.nn.functional.gelu(x, approximate='tanh')


class Engine:
    """One engine of the node.  `limits`: dtypes (the operand dtypes it takes), constants (its weights must be
    constants), tile (the row counts it takes are whole multiples of it).  It lowers each operation kind by its
    prepare_<kind> (its resident shard of units [start, end)) and run_<kind> (its share into `out`, its seconds or
    None where the GPU's events time it)."""
    name, limits = None, {'dtypes': (torch.float16, torch.bfloat16, torch.float32), 'constants': False, 'tile': 1}
    host_side, occupies = True, frozenset()

    def __init__(self, **options):
        self.options = options

    @classmethod
    def missing(cls):
        """Why the node lacks this engine, or None."""
        return None

    def takes(self, op, rows, dtype):
        return (hasattr(self, 'run_' + op.kind) and dtype in self.limits['dtypes'] and rows % self.limits['tile'] == 0
                and rows >= self.limits['tile'])

    def out_dtype(self, dtype):
        return dtype

    def prepare(self, op, start, end):
        return getattr(self, 'prepare_' + op.kind)(op, start, end)

    def run(self, op, shard, x, prologue, out):
        return getattr(self, 'run_' + op.kind)(op, shard, x, prologue, out)


class Mps(Engine):
    """torch's MPS queue: the share enqueued as the GPU's own products, timed by events on its stream."""
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
        gate, up, down = shard
        h = (_gelu(torch.mm(x, gate.T).float()) * torch.mm(x, up.T).float()).to(x.dtype)
        torch.mm(h, down.T, out=out)


class Cpu(Engine):
    """The CPU's matrix units through torch's fp32 CPU products (Accelerate): the input read in place as fp32 (the
    GPU's prologue converts it), the fp32 shard resident in host memory, the output written in place in fp32."""
    name, occupies = 'cpu', frozenset({'cpu'})

    def out_dtype(self, dtype):
        return torch.float32

    def prepare_linear(self, op, start, end):
        return op.weight[start:end].detach().float().cpu().contiguous()

    def run_linear(self, op, shard, x, prologue, out):
        began = time.perf_counter()
        torch.mm(host(prologue['fp32']), shard.T, out=host(out))
        return time.perf_counter() - began

    def prepare_ffn(self, op, start, end):
        cpu = lambda t: t.detach().float().cpu().contiguous()
        return cpu(op.gate[start:end]), cpu(op.up[start:end]), cpu(op.down[:, start:end])

    def run_ffn(self, op, shard, x, prologue, out):
        began = time.perf_counter()
        gate, up, down = shard
        x32 = host(prologue['fp32'])
        h = _gelu(torch.mm(x32, gate.T)).mul_(torch.mm(x32, up.T))
        torch.mm(h, down.T, out=host(out))
        return time.perf_counter() - began


class _Compiled:
    """A Neural Engine share: its operation kind, its fp16 weights and their compiled models by row count."""

    def __init__(self, kind, arrays):
        self.kind, self.arrays, self.models = kind, arrays, {}


class Ane(Engine):
    """The Neural Engine through Core ML: a share a graph of 1x1 convolutions with its weights constant (a linear's
    one; an FFN's gate and up, the activation, their product and down, fused), rows in tiles of `tile` (default 128,
    metal-microbench docs/soc_compute_backends.md: the tiled graph's measured geometry), its input and output
    row-major (the transposes to and from channels-first inside the graph: on an M4 Pro 4.5 ms against 5.9 with them
    on the GPU), compiled and loaded once a share and row count, its package kept in ~/.cache/mesh-engines; its
    predictions through rdma/mesh-coreml.m on the operands' own memory (no CPU copy on its critical path: through
    coremltools' copies a 3 ms prediction took 9 once the CPU clocked down beside a busy GPU), coremltools' own where
    that binding cannot load; fp16 operands and activation.  Its outputs carry an
    absolute error floor (on an M4 Pro and an M5 Max about 3e-4 RMS for a 3840-wide contraction, whatever the input's
    scale; above unit-scale activations its error is fp16's relative 2e-4): accurate on normalized activations,
    inaccurate on small ones (a sixteenth-scale input's linear 5e-3 relative), which an operation's `tolerance`
    checks.  (Its compiler folds a constant scaling of the weights into the convolution, so no rescaling in the
    graph moves the floor.)"""
    name, limits = 'ane', {'dtypes': (torch.float16,), 'constants': True, 'tile': 128}

    def __init__(self, **options):
        super().__init__(**options)
        self.limits = {**self.limits, 'tile': int(options.get('tile', 128))}
        self.occupies = frozenset({'ane'})

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

    def ready(self, shard, rows):
        return rows in shard.models

    def load(self, shard, rows):
        import numpy as np
        tile = self.limits['tile']
        width = shard.arrays[0].shape[1]
        n = shard.arrays[0].shape[0] if shard.kind == 'linear' else shard.arrays[2].shape[0]
        native = _coreml()
        handle, model = None, None
        if native:
            compiled, shard.block, _ = neural_engine_share(shard.kind, shard.arrays, rows, tile)
            inputs, outputs = '', ''
            handle = native.mesh_coreml_load(str(compiled).encode(), b'', b'', 0)
            if handle < 0:
                raise RuntimeError(f'Neural Engine share: {native.mesh_coreml_error().decode()}')
            placed = ctypes.create_string_buffer(1 << 20)
            if native.mesh_coreml_placement(handle, placed, len(placed)) >= 0:
                devices = {line.split('=')[1] for line in placed.value.decode().split('\n') if '=' in line}
                shard.placement = sorted(devices - {'unknown'})
                self.occupies = self.occupies | frozenset(devices - {'unknown'})
        else:
            model = neural_engine_model(shard.kind, shard.arrays, rows, tile)
            spec = model.get_spec().description
            inputs, outputs = spec.input[0].name, spec.output[0].name
        shard.models[rows] = (model, inputs, outputs, handle, n)
        warm = torch.zeros(rows, width, dtype=torch.float16)
        self._predict(shard, warm, torch.empty(rows, n, dtype=torch.float16))

    def _predict(self, shard, source, target):
        """One prediction from `source` into `target` (CPU tensors over the operands' memory): natively on that memory
        where the binding loads, else through coremltools (a copy each way)."""
        model, name, output, handle, n = shard.models[source.shape[0]]
        if handle is None:
            target.copy_(torch.from_numpy(model.predict({name: source.numpy()})[output]))
            return
        shape_in = (ctypes.c_int64 * 2)(*source.shape)
        shape_out = (ctypes.c_int64 * 2)(*target.shape)
        native = _coreml()
        backed = native.mesh_coreml_predict(handle, source.data_ptr(), shape_in, 2, target.data_ptr(), shape_out, 2)
        if backed < 0:
            raise RuntimeError(f'Neural Engine share: {native.mesh_coreml_error().decode()}')
        self.backed = backed

    def _run(self, shard, x, out):
        began = time.perf_counter()
        self._predict(shard, host(x), host(out))
        return time.perf_counter() - began

    def run_linear(self, op, shard, x, prologue, out):
        return self._run(shard, x, out)

    def run_ffn(self, op, shard, x, prologue, out):
        return self._run(shard, x, out)


def neural_engine_model(kind, arrays, rows, tile=128, block=None):
    """A Neural Engine share as a Core ML program (coremltools; fp16): `kind` 'linear' (arrays: W [N, K]) or 'ffn'
    (arrays: Wg, Wu [I, K], Wd [K, I]: down(gelu_tanh(x Wg^T) * (x Wu^T))), its input x [rows, K] and output [rows, N or
    K] row-major, the transposes to and from channels-first inside the graph, rows in tiles of `tile`, each tile one 1x1
    convolution a matrix with its weights constant, the units (N outputs, I neurons) in blocks of at most `block` (each
    block its own convolutions; a linear's blocks concatenated, an FFN's down partials summed), so no convolution's
    weights pass what the Neural Engine's compiler takes (neural_engine_block).  The one statement of the graph: this
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
        pieces = mb.split(x=channels, num_splits=rows // tile, axis=3) if rows > tile else [channels]
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


def neural_engine_share(kind, arrays, rows, tile=128, grain=1):
    """A Neural Engine share compiled where its every operation runs on the Neural Engine: built
    (neural_engine_model), compiled once into ~/.cache/mesh-engines by content (the arrays, kind, rows, tile and block),
    its compute plan read (coremltools MLComputePlan), and built again in halved blocks of `grain` units while any
    operation is placed elsewhere: the Neural Engine's compiler refuses a convolution past a weight size (on an M4 Pro
    between 7680 and 8192 units of a 1536-wide input in fp16: 22.5 to 24 MiB), and Core ML then runs the whole program
    on the CPU, 20 to 60 times slower.  The largest block that placed is kept for the node by input width
    (~/.cache/mesh-engines/blocks.json) and tried first.  (compiled .mlmodelc path, block, its operations' devices)."""
    import collections
    import coremltools as ct
    import numpy as np
    from coremltools.models.compute_plan import MLComputePlan
    cache = Path(os.path.expanduser('~/.cache/mesh-engines'))
    cache.mkdir(parents=True, exist_ok=True)
    known_path = cache / 'blocks.json'
    try:
        known = json.loads(known_path.read_text())
    except (OSError, ValueError):
        known = {}
    units, width = arrays[0].shape[0], arrays[0].shape[1]
    digest = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays) + f'{kind} {rows} {tile}'.encode()).hexdigest()
    native = _coreml()
    if native is None:
        raise RuntimeError('Neural Engine share: rdma/libmesh-coreml.dylib does not load')
    block = min(units, int(known.get(str(width), units)))
    while True:
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
        if set(devices) <= {'ane'} or block <= grain:
            if set(devices) <= {'ane'} and block > int(known.get(str(width), 0)) and block < units:
                known[str(width)] = block
                known_path.write_text(json.dumps(known) + '\n')
            return compiled, block, dict(devices)
        shutil.rmtree(compiled, ignore_errors=True)
        known[str(width)] = min(int(known.get(str(width), units)), max(grain, (block // 2) // grain * grain))
        block = known[str(width)]


ENGINES = {e.name: e for e in (Mps, Cpu, Ane)}


def _interactive():
    """A worker thread at user-interactive QoS: the scheduler keeps it, Core ML's host work and Accelerate's
    products on the performance cores (at the default QoS a Core ML prediction took 5.2 ms against 4.7 on an M4
    Pro)."""
    try:
        ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0)
    except (OSError, AttributeError):
        pass


class Engines:
    """The engines one node's operations may share their units among, in order (the order of their shares along an
    operation's units), each with its options; the evidence of the operations using them; their joint solvers."""

    def __init__(self, names, window=3, sample=16, **options):
        names = [names] if isinstance(names, str) else list(names)
        unknown = [n for n in names if n not in ENGINES]
        if unknown:
            raise ValueError(f'engines: {unknown} are none of {sorted(ENGINES)}')
        lacking = {n: ENGINES[n].missing() for n in names if ENGINES[n].missing()}
        if lacking:
            raise RuntimeError(f'engines: this node lacks {lacking}')
        self.names, self.window, self.sample = names, int(window), int(sample)
        self.engines = [ENGINES[n](**options.get(n, {})) for n in names]
        self.workers = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(names) - 1), initializer=_interactive)
        self.operations, self.decisions, self.evidence, self.solvers, self.last, self.excluded = [], {}, {}, {}, {}, {}
        self.separated = {}

    @classmethod
    def configured(cls, variable='MESH_ENGINES'):
        """The engines the configuration names ("mps,cpu,ane:tile=128"), or None where it names none."""
        given = os.environ.get(variable, '').strip()
        if not given:
            return None
        names, options = [], {}
        for item in given.split(','):
            name, *rest = item.strip().split(':')
            names.append(name)
            options[name] = dict(kv.split('=', 1) for kv in rest[0].split(';')) if rest else {}
        return cls(names, **options)


class Operation:
    """A column-separable operation: `units` shared among `engines` (Engines; None: torch's own) in grains of `grain`,
    a decision a name and row count (operations of one name, a stack's layers of one shape, share it and pool their
    evidence: the allocation is measured where the program runs it, between its other operations); its shares concatenate (`combine` 'concat', the shared axis the output's) or sum ('sum',
    the shared axis contracted); `constant` states that its weights do not change (an engine whose weights are
    constants takes a share only then); `tolerance` (relative), where given, is checked on each allocation's first
    call: every share an engine other than the GPU computes against the same share on the GPU, an engine off by more
    holding none of the decision after (an engine's accuracy is part of choosing it: the Neural Engine's absolute
    error floor)."""
    kind = combine = None

    def __init__(self, units, width, engines, name, grain, constant, tolerance=None):
        self.units, self.width, self.engines, self.grain, self.constant = units, width, engines, grain, constant
        self.tolerance = tolerance
        self.name = name or f'{self.kind}{id(self):x}'
        self.shards, self.warm, self.calls = {}, set(), {}
        if engines is not None:
            if units % grain:
                raise ValueError(f'{self.name}: {units} units in grains of {grain}')
            engines.operations.append(self)

    def torch(self, x):
        raise NotImplementedError

    @property
    def decisions(self):
        """The decisions this operation's name holds (shared by every operation of its name: its layers')."""
        prefix = self.name + '@'
        return {k: d for k, d in (self.engines.decisions if self.engines else {}).items() if k.startswith(prefix)}

    def _decision(self, rows, dtype):
        key = f'{self.name}@{rows}'
        if key not in self.engines.decisions:
            able = [e.takes(self, rows, dtype) and (self.constant or not e.limits['constants']) for e in self.engines.engines]
            if not any(able):
                raise ValueError(f'{key}: no configured engine takes {rows} rows of {dtype}')
            share = (self.units // self.grain) // sum(able)
            parts = [share * self.grain if a else 0 for a in able]
            parts[able.index(True)] += self.units - sum(parts)
            self.engines.decisions[key] = {'parts': parts, 'grain': self.grain, 'low': [0] * len(able),
                                           'high': [self.units if a else 0 for a in able], 'units': self.units}
        return key

    def _shard(self, engine, start, end, rows):
        key = (engine.name, start, end)
        prepared = key not in self.shards
        if prepared:
            self.shards = {k: v for k, v in self.shards.items() if k[0] != engine.name}
            self.shards[key] = engine.prepare(self, start, end)
        shard = self.shards[key]
        if hasattr(engine, 'ready') and not engine.ready(shard, rows):
            engine.load(shard, rows)
            prepared = True
        return shard, prepared

    def __call__(self, x):
        if self.engines is None:
            return self.torch(x)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        rows, dtype, engines = x2.shape[0], x2.dtype, self.engines.engines
        key = self._decision(rows, dtype)
        parts = self.engines.decisions[key]['parts']
        bounds = [sum(parts[:i]) for i in range(len(parts) + 1)]
        held = [i for i, p in enumerate(parts) if p]
        shards = {i: self._shard(engines[i], bounds[i], bounds[i + 1], rows) for i in held}
        first = {(engines[i].name, bounds[i], bounds[i + 1], rows) for i in held} - self.warm
        prepared = any(p for _, p in shards.values()) or bool(first)
        self.warm |= first
        solver = self.engines.solvers.get(key)
        self.calls[key] = self.calls.get(key, 0) + 1
        measured = not prepared and (solver is None or not solver.stands or self.calls[key] % self.engines.sample == 0)
        alone = len(held) == 1 and not engines[held[0]].host_side
        if measured:
            span = (torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True))
            span[0].record()
        prologue = {'fp32': x2.float()} if any(engines[i].name == 'cpu' for i in held) else {}
        outs = {i: torch.empty(rows, parts[i] if self.combine == 'concat' else self.width, device=x2.device,
                               dtype=engines[i].out_dtype(dtype)) for i in held}
        hosted = [i for i in held if engines[i].host_side]
        if hosted:
            ready = torch.mps.Event(enable_timing=False)
            ready.record()
        marks = {}
        for i in (i for i in held if not engines[i].host_side):
            timing = measured and not alone
            if timing:
                begin, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
                begin.record()
            engines[i].run(self, shards[i][0], x2, prologue, outs[i])
            if timing:
                end.record()
                marks[i] = (begin, end)
        if hosted:
            ready.synchronize()
        futures = {i: self.engines.workers.submit(engines[i].run, self, shards[i][0], x2, prologue, outs[i]) for i in hosted}
        seconds = {i: f.result() for i, f in futures.items()}
        pieces = [outs[i] for i in held]
        if self.combine == 'concat':
            y = torch.cat([p.to(dtype) for p in pieces], dim=1) if len(pieces) > 1 else pieces[0].to(dtype)
        else:
            y = pieces[0].float() if len(pieces) > 1 else pieces[0].to(dtype)
            for p in pieces[1:]:
                y = y + p.float()
            y = y.to(dtype)
        if self.tolerance is not None and first:
            self._verify(key, held, bounds, shards, x2, prologue, outs)
        self._separate(key)
        y = self.finish(y)
        if measured:
            span[1].record()
            if alone:
                marks[held[0]] = span
            self.engines.evidence.setdefault(key, []).append((marks, seconds, span))
        return y.reshape(*lead, -1)

    def finish(self, y):
        return y

    def _solver(self, key):
        solver = self.engines.solvers.get(key)
        if solver is None:
            import allocate
            d = self.engines.decisions[key]
            solver = self.engines.solvers[key] = allocate.Coupled(d['units'], d['grain'], d['low'], d['high'], d['parts'],
                                                                  window=self.engines.window)
        return solver

    def _separate(self, key):
        """No two engines occupying one physical unit (`Engine.occupies`, a Neural Engine share's from Core ML's
        placement of its operations) hold units of this decision together: they would time-share it."""
        engines = self.engines.engines
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                pair = (i, j)
                if engines[i].occupies & engines[j].occupies and pair not in self.engines.separated.setdefault(key, set()):
                    self.engines.separated[key].add(pair)
                    self.engines.decisions[key]['parts'] = self._solver(key).forbid(i, j)

    def _verify(self, key, held, bounds, shards, x2, prologue, outs):
        """An allocation's first call: each share an engine other than the GPU computed, against the same share on the
        GPU (torch's MPS, its own products); an engine whose share is off by more than the operation's `tolerance`
        (relative) holds none of this decision from now on."""
        engines = self.engines.engines
        gpu = next((e for e in engines if e.name == 'mps'), None) or Mps()
        for i in held:
            if engines[i].name == 'mps':
                continue
            reference = torch.empty(outs[i].shape, device=x2.device, dtype=x2.dtype)
            gpu.run(self, gpu.prepare(self, bounds[i], bounds[i + 1]), x2, prologue, reference)
            got, want = outs[i].float(), reference.float()
            error = float((got - want).norm() / want.norm().clamp_min(1e-30))
            if error > self.tolerance:
                self.engines.decisions[key]['parts'] = self._solver(key).exclude(i)
                self.engines.decisions[key]['high'][i] = 0
                self.engines.excluded.setdefault(key, {})[engines[i].name] = round(error, 6)


class Linear(Operation):
    """y = x W^T (+ b), W [N, K]: its N outputs shared (they concatenate)."""
    kind, combine = 'linear', 'concat'

    def __init__(self, weight, bias=None, engines=None, name=None, grain=64, constant=True, tolerance=None):
        self.weight, self.bias = weight, bias
        super().__init__(weight.shape[0], weight.shape[0], engines, name, grain, constant, tolerance)

    def torch(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def finish(self, y):
        return y if self.bias is None else y + self.bias


class FFN(Operation):
    """y = down(gelu_tanh(x Wg^T) * (x Wu^T)), Wg and Wu [I, H], Wd [H, I]: its I intermediate neurons shared, each
    engine's share a partial of the whole output (they sum): Megatron's column-then-row split [Shoeybi et al. 2019]."""
    kind, combine = 'ffn', 'sum'

    def __init__(self, gate, up, down, engines=None, name=None, grain=128, constant=True, tolerance=None):
        self.gate, self.up, self.down = gate, up, down
        super().__init__(gate.shape[0], down.shape[0], engines, name, grain, constant, tolerance)

    def torch(self, x):
        h = (_gelu(torch.nn.functional.linear(x, self.gate).float()) * torch.nn.functional.linear(x, self.up).float())
        return torch.nn.functional.linear(h.to(x.dtype), self.down)


def rebalance(engines):
    """The joint solver (allocate.Coupled) of each (operation, rows) decision on its calls since the last: each
    engine's share times (the GPU's from its events) and the calls' spans on the GPU's timeline; the next allocation,
    written into the operations.  {decision: parts}."""
    import allocate
    torch.mps.synchronize()
    out, engines.last = {}, {}
    for key, d in engines.decisions.items():
        calls = engines.evidence.pop(key, [])
        solver = engines.solvers.get(key)
        if solver is None:
            solver = engines.solvers[key] = allocate.Coupled(d['units'], d['grain'], d['low'], d['high'], d['parts'],
                                                             window=engines.window)
        if not calls:
            continue
        width = len(engines.engines)
        times, spans = [[] for _ in range(width)], []
        for marks, seconds, span in calls:
            whole = span[0].elapsed_time(span[1]) / 1e3
            for i, pair in marks.items():
                times[i].append(whole if pair is span else pair[0].elapsed_time(pair[1]) / 1e3)
            for i, s in seconds.items():
                times[i].append(s)
            spans.append(whole)
        engines.last[key] = {'ms': [round(sorted(t)[len(t) // 2] * 1e3, 3) if t else None for t in times],
                             'call_ms': round(sorted(spans)[len(spans) // 2] * 1e3, 3)}
        d['parts'] = solver.observe(times, spans)
        out[key] = list(d['parts'])
    return out


def balancing(engines):
    """Each decision's parts, whether they stand, and the last window's share and call times."""
    return {k: {'parts': list(s.parts), 'stands': s.stands, **engines.last.get(k, {})} for k, s in engines.solvers.items()}
