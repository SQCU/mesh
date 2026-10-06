"""A node's compute engines as members of one tensor operation: its units shared among them by evidence
(design/heterogeneity.md: raggedness within a node is the raggedness between nodes; metal-microbench
docs/soc_compute_backends.md for the engines' measured envelopes).

An Apple SoC holds several matrix engines with their own rates by shape, dtype and operand form: the GPU (torch's MPS
queue; on an M4 Pro about 6 TFLOP/s at fp16 and fp32 alike, on an M5 its neural accelerators), the CPU's matrix units
(SME/AMX, one a cluster, reached through Accelerate by torch's fp32 CPU products: about 3 TFLOP/s on an M4 Pro's two
performance clusters) and the Neural Engine (Core ML's static graphs; a 1x1 convolution with constant weights, rows in
tiles, about 6 TFLOP/s with Python's copies on an M4 Pro and faster where bound natively).  No one engine is fastest
for every operation, and the fastest set changes with the shape, so the operation's final backend is resolved here,
per operation and row count, from what each engine measures: a column-separable operation's output units (a
linear's output features) are shared among the configured engines like a mesh dimension among ranks, each engine
holding its shard of the weights resident, the shares running at once on unified memory, and one
rdma/allocate.py Balancer moving the shares on the calls' own times (an engine's time is its share's; the
operation's is the slowest engine's, as a collective's is its slowest rank's).  A share of nothing leaves that engine
out: the resolution can be one engine.

Nothing is implicit.  An operation takes its engines as an operand (`Engines`), from the program or from the
configuration the program names (`Engines.configured()`: MESH_ENGINES, e.g. "mps,cpu,ane:tile=128"); without
one it is torch's own operation on its input's device.  An engine the configuration names and the node lacks is an
error, not a silent omission.  An engine's limits are data (`Engine.limits`): its operand dtypes, whether its
weights must be constants (the Neural Engine's: a share there is compiled), and the row counts it takes (the Neural
Engine's whole tiles); a row count an engine cannot take gives it no units for that decision.

The data path is zero-copy where the memory allows: torch's MPS tensors live in shared Metal memory, so the CPU and
Neural Engine shares read the GPU's prologue (the input as fp32 for the CPU's products, transposed channels-first
for the Neural Engine's convolution) in place and write their outputs into MPS tensors in place; the GPU combines the
shares.  The Neural Engine's binding through coremltools copies its input and output once each (counted in its
time).

  lin = engines.Linear(weight, engines=engines.Engines.configured(), name='ffn.up')
  y = lin(x)                      # x [..., K] on MPS: every configured engine its share of the N outputs
  engines.rebalance(lin.engines)  # a Balancer step on the calls since the last: each (operation, rows)'s parts"""
import concurrent.futures
import ctypes
import hashlib
import os
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


class Engine:
    """One engine of the node.  `limits`: dtypes (the operand dtypes it takes), constants (its weights must be
    constants), tile (the row counts it takes are whole multiples of it)."""
    name, limits = None, {'dtypes': (torch.float16, torch.bfloat16, torch.float32), 'constants': False, 'tile': 1}

    def __init__(self, **options):
        self.options = options

    @classmethod
    def missing(cls):
        """Why the node lacks this engine, or None."""
        return None

    def takes(self, rows, dtype):
        return dtype in self.limits['dtypes'] and rows % self.limits['tile'] == 0 and rows >= self.limits['tile']

    def prepare(self, weight, start, end):
        """The engine's resident shard of `weight`'s output rows [start, end)."""
        raise NotImplementedError

    def run(self, shard, x, prologue, out):
        """Its share: `out` [rows, end - start] (an MPS tensor) from `x` [rows, K] and the prologue's operands; its
        seconds, or None where the GPU's events time it."""
        raise NotImplementedError


class Mps(Engine):
    """torch's MPS queue: the share enqueued as the GPU's own product, timed by events on its stream."""
    name = 'mps'

    @classmethod
    def missing(cls):
        return None if torch.backends.mps.is_available() else 'no MPS device'

    def prepare(self, weight, start, end):
        return weight[start:end]

    def run(self, shard, x, prologue, out):
        torch.mm(x, shard.T, out=out)
        return None


class Cpu(Engine):
    """The CPU's matrix units through torch's fp32 CPU products (Accelerate): the input read in place as fp32 (the
    GPU's prologue converts it), the fp32 shard resident in host memory, the output written in place."""
    name, limits = 'cpu', {'dtypes': (torch.float16, torch.bfloat16, torch.float32), 'constants': False, 'tile': 1}

    def prepare(self, weight, start, end):
        return weight[start:end].detach().float().cpu().contiguous()

    def run(self, shard, x, prologue, out):
        began = time.perf_counter()
        source, target = host(prologue['fp32']), host(out)
        result = torch.mm(source, shard.T)
        target.copy_(result)
        return time.perf_counter() - began


class Ane(Engine):
    """The Neural Engine through Core ML: the shard a 1x1 convolution with its weights constant, the input
    channels-first [1, K, 1, rows] (the GPU's prologue transposes it) split into row tiles of `tile` (default 128,
    metal-microbench docs/soc_compute_backends.md: the tiled graph's measured geometry), compiled once a (weight
    shard, rows) and cached; fp16 operands."""
    name, limits = 'ane', {'dtypes': (torch.float16,), 'constants': True, 'tile': 128}

    def __init__(self, **options):
        super().__init__(**options)
        self.limits = {**self.limits, 'tile': int(options.get('tile', 128))}
        self.models = {}

    @classmethod
    def missing(cls):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            return 'coremltools is not installed (rdma/requirements.txt)'
        return None

    def prepare(self, weight, start, end):
        return weight[start:end].detach().to('cpu', torch.float16).numpy()

    def _model(self, shard, rows):
        key = (hashlib.sha1(shard.tobytes()).hexdigest(), rows)
        if key not in self.models:
            import coremltools as ct
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil.mil import types
            n, k = shard.shape
            tile = self.limits['tile']

            @mb.program(input_specs=[mb.TensorSpec(shape=(1, k, 1, rows), dtype=types.fp16)], opset_version=ct.target.macOS15)
            def program(x):
                w = mb.const(val=shard.reshape(n, k, 1, 1))
                pieces = mb.split(x=x, num_splits=rows // tile, axis=3) if rows > tile else [x]
                outputs = [mb.conv(x=p, weight=w) for p in pieces]
                return mb.concat(values=outputs, axis=3) if len(outputs) > 1 else outputs[0]
            model = ct.convert(program, convert_to='mlprogram', compute_units=ct.ComputeUnit.CPU_AND_NE,
                               compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
            spec = model.get_spec().description
            self.models[key] = (model, spec.input[0].name, spec.output[0].name)
        return self.models[key]

    def run(self, shard, x, prologue, out):
        began = time.perf_counter()
        rows, n = x.shape[0], shard.shape[0]
        model, name, output = self._model(shard, rows)
        channels = host(prologue['channels'])
        result = model.predict({name: channels.numpy().reshape(1, -1, 1, rows)})[output]
        host(out).copy_(torch.from_numpy(result.reshape(n, rows)))
        return time.perf_counter() - began


ENGINES = {e.name: e for e in (Mps, Cpu, Ane)}


class Engines:
    """The engines one node's operations may share their units among, in order (the order of their shares along an
    operation's units), each with its options; the evidence of the operations using them; their Balancer."""

    def __init__(self, names, **options):
        names = [names] if isinstance(names, str) else list(names)
        unknown = [n for n in names if n not in ENGINES]
        if unknown:
            raise ValueError(f'engines: {unknown} are none of {sorted(ENGINES)}')
        lacking = {n: ENGINES[n].missing() for n in names if ENGINES[n].missing()}
        if lacking:
            raise RuntimeError(f'engines: this node lacks {lacking}')
        self.names = names
        self.engines = [ENGINES[n](**options.get(n, {})) for n in names]
        self.workers = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(names) - 1))
        self.operations, self.evidence, self.balancer = [], {}, None

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


class Linear:
    """y = x W^T (+ b), W [N, K]: its N outputs shared among `engines` (Engines; None: torch's own linear), in grains
    of `grain` outputs, a decision a row count (the shares a prefill chunk and a decode step take differ); `constant`
    states that W does not change (an engine whose weights are constants takes a share only then)."""

    def __init__(self, weight, bias=None, engines=None, name=None, grain=64, constant=True):
        self.weight, self.bias, self.engines, self.grain, self.constant = weight, bias, engines, grain, constant
        self.name = name or f'linear{id(self):x}'
        self.decisions, self.shards = {}, {}
        if engines is not None:
            if weight.shape[0] % grain:
                raise ValueError(f'{self.name}: {weight.shape[0]} outputs in grains of {grain}')
            engines.operations.append(self)

    def _decision(self, rows, dtype):
        """The (operation, rows) decision: its parts, its engines' bounds (an engine that cannot take these rows or
        this dtype, or needs constant weights the operation does not promise, holds nothing)."""
        key = f'{self.name}@{rows}'
        if key not in self.decisions:
            n = self.weight.shape[0]
            able = [e.takes(rows, dtype) and (self.constant or not e.limits['constants']) for e in self.engines.engines]
            if not any(able):
                raise ValueError(f'{key}: no configured engine takes {rows} rows of {dtype}')
            share = (n // self.grain) // sum(able)
            parts = [share * self.grain if a else 0 for a in able]
            parts[able.index(True)] += n - sum(parts)
            self.decisions[key] = {'parts': parts, 'grain': self.grain, 'low': [0] * len(able),
                                   'high': [n if a else 0 for a in able]}
        return key

    def _shard(self, engine, start, end):
        key = (engine.name, start, end)
        if key not in self.shards:
            self.shards = {k: v for k, v in self.shards.items() if k[0] != engine.name}
            self.shards[key] = engine.prepare(self.weight, start, end)
        return self.shards[key]

    def __call__(self, x):
        if self.engines is None:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        rows = x2.shape[0]
        key = self._decision(rows, x2.dtype)
        parts = self.decisions[key]['parts']
        engines = self.engines.engines
        bounds = [sum(parts[:i]) for i in range(len(parts) + 1)]
        held = [i for i, p in enumerate(parts) if p]
        prologue = {}
        if any(engines[i].name == 'cpu' for i in held):
            prologue['fp32'] = x2.float()
        if any(engines[i].name == 'ane' for i in held):
            prologue['channels'] = x2.T.contiguous()
        shapes = {'ane': lambda p: (p, rows)}
        outs = {i: torch.empty(*shapes.get(engines[i].name, lambda p: (rows, p))(parts[i]), device=x2.device,
                               dtype=torch.float32 if engines[i].name == 'cpu' else x2.dtype) for i in held}
        ready = torch.mps.Event(enable_timing=False)
        ready.record()
        marks = {}
        for i in (i for i in held if engines[i].name == 'mps'):
            begin, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
            begin.record()
            engines[i].run(self._shard(engines[i], bounds[i], bounds[i + 1]), x2, prologue, outs[i])
            end.record()
            marks[i] = (begin, end)
        ready.synchronize()
        futures = {i: self.engines.workers.submit(engines[i].run, self._shard(engines[i], bounds[i], bounds[i + 1]),
                                                   x2, prologue, outs[i]) for i in held if engines[i].name != 'mps'}
        seconds = {i: f.result() for i, f in futures.items()}
        pieces = [outs[i].T if engines[i].name == 'ane' else outs[i] for i in held]
        y = torch.cat([p.to(x2.dtype) for p in pieces], dim=1) if len(pieces) > 1 else pieces[0].to(x2.dtype)
        if self.bias is not None:
            y = y + self.bias
        self.engines.evidence.setdefault(key, []).append((marks, seconds, len(engines)))
        return y.reshape(*lead, -1)


def rebalance(engines):
    """One Balancer step over the operations' decisions on their calls since the last (each call's engines' times,
    the GPU's from its events): the next parts, written into the operations.  {decision: parts}."""
    import allocate
    torch.mps.synchronize()
    decisions = {k: d for op in engines.operations for k, d in op.decisions.items()}
    if engines.balancer is None or set(engines.balancer.decisions) != set(decisions):
        engines.balancer = allocate.Balancer({k: dict(d) for k, d in decisions.items()})
    stretches = {}
    for key, calls in engines.evidence.items():
        width = calls[0][2]
        times = [[] for _ in range(width)]
        for marks, seconds, _ in calls:
            for i, (begin, end) in marks.items():
                times[i].append(begin.elapsed_time(end) / 1e3)
            for i, s in seconds.items():
                times[i].append(s)
        stretches[key] = {'scales': {key: [1.0] * width}, 'times': [t or None for t in times]}
    engines.evidence = {}
    if stretches:
        engines.balancer.observe(stretches)
    parts = engines.balancer.parts
    for op in engines.operations:
        for key in op.decisions:
            if key in parts:
                op.decisions[key]['parts'] = list(parts[key])
    return parts


def balancing(engines):
    """Each decision's parts and whether they stand."""
    return {k: {'parts': list(v), 'stands': engines.balancer.stands(k)} for k, v in (engines.balancer.parts if engines.balancer else {}).items()}
