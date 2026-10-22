import collections
import json
import math
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
ARGS = ('TILE', 'K', 'S', 'G', 'RG', 'RB', 'THREADS', 'SIMD', 'PEERS')
STAGES = 3
PAGE = resource.getpagesize()
RING_BLOCKS = 2
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
def publications(crossing, peers):
    records = crossing.publish if crossing else ()
    return [w for k in range(peers) for w in fused._words(records[k] if k < len(records) else (0,) * 6)]


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
        args = dict(TILE=tile, K=K, S=S, G=G, RG=RG, RB=RB, THREADS=SG * simd, SIMD=simd, PEERS=peers)
        handles = [_pipeline(stage, args) for stage in range(STAGES)]
        threads = min(_stream.pipeline_facts(handle)[0] for handle in handles)
        if threads >= args['THREADS']:
            break
        if RG == 1:
            raise RuntimeError(f'chain step {args}: its pipeline holds {threads} threads a threadgroup, fewer than its '
                               f'{SG} simdgroups')
        RG //= 2
    return handles, args, [rows, *splits(RG, SG)]


# docs/writing-a-service.md#the-step-kernel
def _pipeline(stage, args):
    name, line = instance('chain_step', stage, *(args[a] for a in ARGS))
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
        self.forms, self.landings, self.published = {}, {}, {}
        self.plan(rows)
        self.args = self.forms[rows][1]
        self.arrivals = torch.zeros(1, dtype=torch.int32, device='mps')

    # docs/writing-a-service.md#the-step-kernel
    def plan(self, rows):
        if rows > self.capacity:
            raise ValueError(f'chain step: {rows} rows, past the capacity {self.capacity}')
        if rows not in self.forms:
            self.forms[rows] = form(self.K, self.S, rows, self.peers, self.held)
            self.landings[rows] = [self.landing[q * self.slot:q * self.slot + rows * self.K].view(rows, self.K)
                                   for q in range(self.peers)]

    # docs/writing-a-service.md#the-step-kernel
    def __call__(self, e, p, out, previous):
        crossing = fused.exchange(p, self.landings[e.shape[0]], _stream.polls)
        self._encode(e, p, out, previous, crossing, self.published.get(p.data_ptr()))
        self.published[p.data_ptr()], self.sent = crossing, p
        return crossing

    # docs/writing-a-service.md#the-step-kernel
    def land(self, e, out, previous):
        self._encode(e, None, out, previous, None, None)

    # docs/writing-a-service.md#the-step-kernel
    def _encode(self, e, p, out, previous, crossing, earlier):
        handles, args, words = self.forms[e.shape[0]]
        buffers = [self.At, self.Bt, e, out, self.X, self.Y, p, self.arrivals, self.sent if self.sent is not None else self.X]
        constants = [int(previous is None), int(out is not None), *words, 0, *publications(earlier, self.peers)]
        G, threads = args['G'], args['THREADS']
        fused.encode(handles[0], buffers, constants, 15, G, 1, threads, self.peers, land=previous)
        if crossing:
            fused.encode(handles[1], buffers, constants, 15, G, 1, threads, self.peers)
            fused.encode(handles[2], buffers, constants, 15, G, 1, threads, self.peers, publish=crossing)


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
