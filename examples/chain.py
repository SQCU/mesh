import collections
import json
import math
import os
import time
from pathlib import Path

import resource

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch_mesh import _stream, argmax, fused
from torch_mesh.streamed import instance, node

# docs/writing-a-service.md#the-step-kernel
SOURCE = fused.steps() + argmax.SOURCE + '\n' + Path(__file__).with_name('chain.metal').read_text()
ARGS = ('TILE', 'K', 'S', 'G', 'RG', 'RB', 'THREADS', 'SIMD', 'PATIENCE', 'PEERS')
PAGE = resource.getpagesize()
RING_BLOCKS = 2
CHUNK = 1 << 24
_PIPELINES = {}


# docs/writing-a-service.md#the-capability
def capability(directory):
    family, threads, memory = _stream.device_facts()
    held = json.loads((Path(directory) / f'apple{family}.json').read_text())
    held['max_threads_per_threadgroup'], held['threadgroup_memory_bytes'] = threads, memory
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
def form(K, S, rows, peers, held):
    tile = held['simdgroup_matrix_edge']
    wrong = [f'{name} {n}' for name, n in (('width', K), ('share', S), ('rows', rows)) if n < tile or n % tile]
    if wrong:
        raise ValueError(f'chain step: the Apple{held["family"]} matrix unit takes {tile} x {tile} tiles, so a width, '
                         f'a share and a row count are positive multiples of {tile}, not {", ".join(wrong)}')
    simd, cores = _width(), node()[1]
    kt, st, rt = K // tile, S // tile, rows // tile
    G = min(cores // 2, max(kt, st))
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
        args = dict(TILE=tile, K=K, S=S, G=G, RG=RG, RB=RB, THREADS=SG * simd, SIMD=simd, PATIENCE=_stream.polls, PEERS=peers)
        handle = _pipeline(args)
        threads = _stream.pipeline_facts(handle)[0]
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
    def __init__(self, a, b, rows, capacity, peers, held):
        S, K = a.shape
        tile = held['simdgroup_matrix_edge']
        tiled = lambda w: w.reshape(w.shape[0] // tile, tile, w.shape[1] // tile, tile).permute(2, 0, 1, 3).contiguous()
        self.At, self.Bt = tiled(a.T), tiled(b)
        self.K, self.S, self.held, self.capacity, self.peers = K, S, held, capacity, peers
        self.X = torch.empty(capacity, K, dtype=torch.half, device='mps')
        self.Y = torch.empty(capacity, S, dtype=torch.half, device='mps')
        self.slot = paged(capacity * K)
        self.landing = torch_mesh.empty(peers * self.slot, dtype=torch.half)
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
            self.forms[rows] = form(self.K, self.S, rows, self.peers, self.held)

    # docs/writing-a-service.md#the-step-kernel
    def __call__(self, e, p, out, previous):
        landings = [self.landing[q * self.slot:q * self.slot + p.numel()].view_as(p) for q in range(self.peers)]
        crossing = fused.exchange(p, landings, _stream.polls)
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
                     args['G'], 1, args['THREADS'], self.peers, land=previous, publish=crossing)

    # docs/writing-a-service.md#the-step-kernel
    def check(self):
        cause = int(self.err.item())
        if cause:
            raise RuntimeError(f'chain step {self.args}: a wait ran past its patience (causes {cause:#x})')


# docs/writing-a-service.md#how-the-example-moves-its-data
def operand(path, rank, width):
    shape, inputs = torch_mesh.empty(2, dtype=torch.int64, device='cpu'), None
    if rank == 0:
        inputs = np.load(path, mmap_mode='r')
        if (inputs.ndim != 3 or 0 in inputs.shape or inputs.shape[2] != width or inputs.dtype != np.float16
                or not inputs.flags.c_contiguous):
            raise ValueError('The input is C-ordered float16 [steps, rows, width], none of them 0')
        shape[0], shape[1] = inputs.shape[:2]
    dist.broadcast(shape, src=0)
    steps, rows = shape.tolist()
    return steps, rows, inputs


# docs/writing-a-service.md#the-rings
def ring(ring_bytes, steps, slot):
    slots = ring_bytes // slot
    if slots >= 2 * steps:
        return steps, steps, steps
    whole = PAGE // math.gcd(slot, PAGE)
    block = slots // (RING_BLOCKS + 1) // whole * whole
    if block < 1:
        raise ValueError(f'ring_bytes {ring_bytes} holds {slots} slots of {slot} bytes; the input ring of '
                         f'{RING_BLOCKS} blocks and the crossing ring of one take {RING_BLOCKS + 1} slots at least, '
                         f'{(RING_BLOCKS + 1) * slot} bytes')
    return block, min(steps, RING_BLOCKS * block), block


# docs/writing-a-service.md#the-rings
def span(first, steps, block, depth):
    last = min(steps, first + block)
    return slice(first, last), slice(first % depth, first % depth + last - first)


class Pace:
    # docs/writing-a-service.md#the-rings
    def __init__(self):
        self.marks, self.done, self.waited = collections.deque(), [], 0.0

    # docs/writing-a-service.md#the-rings
    def wait(self):
        if len(self.marks) == RING_BLOCKS - 1:
            began = time.perf_counter()
            self.marks.popleft().synchronize()
            self.done.append(time.perf_counter())
            self.waited += self.done[-1] - began

    # docs/writing-a-service.md#the-rings
    def mark(self):
        self.marks.append(torch.mps.Event())
        self.marks[-1].record()


# docs/writing-a-service.md#how-the-example-moves-its-data
def fill(E, rank, inputs, first, steps, block):
    held, slots = span(first, steps, block, len(E))
    if rank == 0:
        E[slots].copy_(torch.from_numpy(inputs[held]))
    return dist.broadcast(E[slots], src=0, async_op=True)


# docs/writing-a-service.md#how-the-example-moves-its-data
def drain(first, steps, block, O, outputs):
    if O is not None:
        held = span(first, steps, block, len(O))[0]
        torch.from_numpy(outputs[held]).copy_(O[:held.stop - held.start], non_blocking=True)


# docs/writing-a-service.md#the-configuration
def _drawn(out, draw):
    flat = out.reshape(len(out), -1)
    n = max(1, CHUNK // flat.shape[1])
    for i in range(0, len(flat), n):
        flat[i:i + n] = draw(flat[i:i + n].shape)
    return out


# docs/writing-a-service.md#the-configuration
def _saved(path, shape, draw):
    _drawn(np.lib.format.open_memmap(path, 'w+', np.float16, shape), draw).flush()


# docs/writing-a-service.md#the-configuration
def inputs(config, steps, rows):
    g, w, n = np.random.default_rng(0), config['width'], sum(config['parts'])
    _saved(config['input'], (steps, rows, w), g.standard_normal)
    with open(config['weights'] + '.part', 'wb') as f:
        np.savez(f, **{k: _drawn(np.empty((n, w), np.float16), lambda s: g.standard_normal(s) / np.sqrt(w)) for k in 'ab'})
    os.replace(config['weights'] + '.part', config['weights'])


# docs/writing-a-service.md#the-function
def reference(e, a, b, carried):
    x, out = e[0] + carried, []
    for t in range(len(e)):
        out.append(np.tanh(np.tanh(x @ a.T) @ b))
        if t + 1 < len(e):
            x = e[t + 1] + out[-1]
    return np.stack(out)
