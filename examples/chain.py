import json
from pathlib import Path

import resource

import torch
import torch_mesh
from torch_mesh import _stream, argmax, fused
from torch_mesh.streamed import instance, node

# docs/writing-a-service.md#the-step-kernel
SOURCE = fused.steps() + argmax.SOURCE + '\n' + Path(__file__).with_name('chain.metal').read_text()
ARGS = ('TILE', 'K', 'S', 'G', 'RG', 'RB', 'THREADS', 'SIMD', 'PATIENCE')
PAGE = resource.getpagesize()
_PIPELINES = {}


# docs/writing-a-service.md#the-capability
def capability(path):
    held = json.loads(Path(path).read_text())
    threads, memory = _stream.device_facts()
    if (threads, memory) != (held['max_threads_per_threadgroup'], held['threadgroup_memory_bytes']):
        raise ValueError(f'{path}: {held["family"]} documents {held["max_threads_per_threadgroup"]} threads and '
                         f'{held["threadgroup_memory_bytes"]} bytes of threadgroup memory a threadgroup, and this device '
                         f'reports {threads} and {memory}: the file is not this device\'s capability')
    return held


# docs/writing-a-service.md#the-capability
def _width():
    if 'width' not in _PIPELINES:
        probe = _stream.pipeline('kernel void width_probe(device uint* x [[buffer(0)]]) { x[0] = 0; }', 'width_probe')
        _PIPELINES['width'] = _stream.pipeline_facts(probe)[1]
    return _PIPELINES['width']


# docs/writing-a-service.md#the-crossing
def paged(count, dtype=torch.half):
    size = torch.tensor([], dtype=dtype).element_size()
    return -(-count * size // PAGE) * PAGE // size


# docs/writing-a-service.md#the-step-kernel
def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


# docs/writing-a-service.md#the-step-kernel
def _floor2(n):
    return 1 << (n.bit_length() - 1)


# docs/writing-a-service.md#the-step-kernel
def form(K, S, rows, held):
    tile = held['simdgroup_matrix_edge']
    wrong = [f'{name} {n}' for name, n in (('width', K), ('share', S), ('rows', rows)) if n < tile or n % tile]
    if wrong:
        raise ValueError(f'chain step: the {held["family"]} matrix unit takes {tile} x {tile} tiles, so a width, a share '
                         f'and a row count are positive multiples of {tile}, not {", ".join(wrong)}')
    simd, cores = _width(), node()[1]
    kt, st, rt = K // tile, S // tile, rows // tile
    G = min(cores, max(kt, st))
    most = held['max_threads_per_threadgroup'] // simd

    def simdgroups(rg):
        groups = rt // rg
        return min(most, _floor2(max(-(-st // G) * groups * kt, -(-kt // G) * groups * st)))

    def splits(rg, sg):
        groups = rt // rg
        return _ksplit(-(-st // G) * groups, kt, sg), _ksplit(-(-kt // G) * groups, st, sg)

    SG = simdgroups(1)
    RG = max(rg for rg in (1 << i for i in range(rt.bit_length()))
             if rt % rg == 0 and simdgroups(rg) == SG and splits(rg, SG) == splits(1, SG))
    while True:
        RB = RG
        while SG * RB * tile * tile * 4 + 2 * held['threadgroup_memory_alignment_bytes'] > held['threadgroup_memory_bytes']:
            RB //= 2
        if not RB:
            raise ValueError(f'chain step: {SG} simdgroups hold no {tile} x {tile} float partial each in '
                             f'{held["threadgroup_memory_bytes"]} bytes of threadgroup memory')
        args = dict(TILE=tile, K=K, S=S, G=G, RG=RG, RB=RB, THREADS=SG * simd, SIMD=simd, PATIENCE=_stream.polls)
        handle = _pipeline(args)
        threads, width, memory = _stream.pipeline_facts(handle)
        if width != simd or memory > held['threadgroup_memory_bytes']:
            raise RuntimeError(f'chain step {args}: its pipeline runs {width} threads a simdgroup and {memory} bytes of '
                               f'threadgroup memory')
        if threads >= args['THREADS']:
            break
        if RG == 1:
            raise RuntimeError(f'chain step {args}: its pipeline holds {threads} threads a threadgroup, fewer than its '
                               f'{SG} simdgroups')
        RG //= 2
    return handle, args, [rows, *splits(RG, SG)]


# docs/writing-a-service.md#the-step-kernel
def _pipeline(args):
    name, line = instance('chain_step', *(args[a] for a in ARGS))
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


class Step:
    # docs/writing-a-service.md#the-step-kernel
    def __init__(self, a, b, rows, capacity, held):
        S, K = a.shape
        tile = held['simdgroup_matrix_edge']
        tiled = lambda w: w.reshape(w.shape[0] // tile, tile, w.shape[1] // tile, tile).permute(2, 0, 1, 3).contiguous()
        self.At, self.Bt = tiled(a.T), tiled(b)
        self.K, self.S, self.held, self.capacity = K, S, held, capacity
        self.X = torch.empty(capacity, K, dtype=torch.half, device='mps')
        self.Y = torch.empty(capacity, S, dtype=torch.half, device='mps')
        self.landing = torch_mesh.empty(paged(capacity * K), dtype=torch.half)
        self.sent = None
        self.forms = {}
        self.plan(rows)
        self.args = self.forms[rows][1]
        self.stamps = torch.zeros(self.args['G'], dtype=torch.int32, device='mps')
        self.err = torch.zeros(1, dtype=torch.int32, device='mps')
        self.arrivals = torch.zeros(1, dtype=torch.int32, device='mps')
        self.phase = 0

    # docs/writing-a-service.md#the-step-kernel
    def plan(self, rows):
        if rows > self.capacity:
            raise ValueError(f'chain step: {rows} rows, past the capacity {self.capacity}')
        if rows not in self.forms:
            self.forms[rows] = form(self.K, self.S, rows, self.held)

    # docs/writing-a-service.md#the-step-kernel
    def __call__(self, e, p, out, previous):
        crossing = fused.all_reduce(p, self.landing[:p.numel()].view_as(p), _stream.polls)
        self._encode(e, p, out, previous, crossing, 0)
        self.sent = p
        self.phase += 2
        return crossing

    # docs/writing-a-service.md#the-step-kernel
    def land(self, e, out, previous):
        self._encode(e, None, out, previous, None, 1)

    # docs/writing-a-service.md#the-step-kernel
    def _encode(self, e, p, out, previous, crossing, last):
        handle, args, words = self.forms[e.shape[0]]
        fused.encode(handle, [self.At, self.Bt, e, out, self.X, self.Y, p, self.stamps, self.err, self.arrivals,
                              self.sent if self.sent is not None else self.X],
                     [int(previous is None), self.phase, *words, last, int(out is not None), 0], 15,
                     args['G'], 1, args['THREADS'], land=previous, publish=crossing)

    # docs/writing-a-service.md#the-step-kernel
    def check(self):
        cause = int(self.err.item())
        if cause:
            raise RuntimeError(f'chain step {self.args}: a wait ran past its patience (causes {cause:#x})')
