from pathlib import Path

import torch
import torch_mesh
from torch_mesh import _stream, fused
from torch_mesh.streamed import instance, node

import chain

# docs/writing-a-service.md#the-lstm
SOURCE = chain.SOURCE + '\n' + Path(__file__).with_name('lstm.metal').read_text()
ARGS = ('TILE', 'KIN', 'H', 'HR', 'L', 'G', 'RG', 'RB', 'THREADS', 'SIMD', 'PEERS', 'NSEG')
SLOTS = 2
_PIPELINES = {}


# docs/writing-a-service.md#the-lstm
def form(KIN, H, HR, L, rows, ranks, held):
    tile = held['simdgroup_matrix_edge']
    wrong = [f'{name} {n}' for name, n in (('input width', KIN), ('width', H), ('share', HR), ('rows', rows))
             if n < tile or n % tile]
    if wrong:
        raise ValueError(f'lstm tick: the Apple{held["family"]} matrix unit takes {tile} x {tile} tiles, so an input '
                         f'width, a width, a share and a row count are positive multiples of {tile}, not '
                         f'{", ".join(wrong)}')
    simd, cores = chain._width(), node()[1]
    nt, rt = 4 * HR // tile, rows // tile
    kts = [(KIN + H) // tile] + [2 * H // tile] * (L - 1)
    G = min(cores, nt)
    most = held['max_threads_per_threadgroup'] // simd

    def items(rg):
        return -(-nt // G) * (rt // rg)

    def simdgroups(rg):
        return min(most, chain._floor2(items(rg) * max(kts)))

    def splits(rg, sg):
        return [chain._ksplit(items(rg), kt, sg) for kt in kts]

    SG = simdgroups(1)
    RG = max(rg for rg in (1 << i for i in range(rt.bit_length()))
             if rt % rg == 0 and simdgroups(rg) == SG and splits(rg, SG) == splits(1, SG))
    while True:
        RB = RG
        while SG * RB * tile * tile * 4 + 2 * held['threadgroup_memory_alignment_bytes'] > held['threadgroup_memory_bytes']:
            RB //= 2
        if not RB:
            raise ValueError(f'lstm tick: {SG} simdgroups hold no {tile} x {tile} float partial each in '
                             f'{held["threadgroup_memory_bytes"]} bytes of threadgroup memory')
        args = dict(TILE=tile, KIN=KIN, H=H, HR=HR, L=L, G=G, RG=RG, RB=RB, THREADS=SG * simd, SIMD=simd,
                    PEERS=ranks - 1, NSEG=max(1 + ranks, 2 * ranks))
        handle = _pipeline(args)
        if _stream.pipeline_facts(handle)[0] >= args['THREADS']:
            break
        if RG == 1:
            raise RuntimeError(f'lstm tick {args}: its pipeline holds fewer threads a threadgroup than its {SG} '
                               f'simdgroups')
        RG //= 2
    return handle, args, splits(RG, SG)


# docs/writing-a-service.md#the-lstm
def _pipeline(args):
    name, line = instance('lstm_tick', *(args[a] for a in ARGS))
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


# docs/writing-a-service.md#the-lstm
def tiled(w, tile):
    K, N = w.shape
    return w.reshape(K // tile, tile, N // tile, tile).permute(2, 0, 1, 3).reshape(-1)


class Tick:
    # docs/writing-a-service.md#the-lstm
    def __init__(self, weights, bias, rows, shares, rank, L, KIN, held):
        H, HR, tile = sum(shares), shares[rank], held['simdgroup_matrix_edge']
        self.rows, self.shares, self.rank, self.L, self.HR = rows, shares, rank, L, HR
        self.handle, self.args, self.ks = form(KIN, H, HR, L, rows, len(shares), held)
        sizes = [(KIN + H) * 4 * HR] + [2 * H * 4 * HR] * (L - 1)
        layers = torch.split(weights, sizes)
        self.W = torch.cat([tiled(w.view(-1, 4 * HR), tile) for w in layers]).contiguous()
        self.bias = bias.to('mps')
        self.C = torch.zeros(L, rows, HR, dtype=torch.float32, device='mps')
        self.arrivals = torch.zeros(1, dtype=torch.int32, device='mps')
        self.S = torch_mesh.empty(SLOTS, chain.paged(L * rows * HR), dtype=torch.half)
        peers = [q for q in range(len(shares)) if q != rank]
        offsets, at = {}, 0
        for q in peers:
            offsets[q], at = at, at + chain.paged(L * rows * shares[q])
        self.landing = torch_mesh.empty(max(at, chain.paged(1)), dtype=torch.half)
        self.landings = [self.landing[offsets[q]:offsets[q] + L * rows * shares[q]] for q in peers]
        self.segments = self._segments(KIN, H, offsets)
        self.sent, self.previous, self.tick = None, None, 0

    # docs/writing-a-service.md#the-lstm
    def _segments(self, KIN, H, offsets):
        starts = [sum(self.shares[:q]) for q in range(len(self.shares))]

        def slices(layer, base):
            return [(1, layer * self.rows * h, h, (base + starts[q]) // 8, h // 8) if q == self.rank else
                    (2, offsets[q] + layer * self.rows * h, h, (base + starts[q]) // 8, h // 8)
                    for q, h in enumerate(self.shares)]

        return [[(0, 0, KIN, 0, KIN // 8)] + slices(0, KIN)] + [slices(l - 1, 0) + slices(l, H) for l in range(1, self.L)]

    # docs/writing-a-service.md#the-lstm
    def _words(self, last, drains, zeroed, active, outs):
        even = lambda xs: list(xs) + [0] * (len(xs) % 2)
        words = [int(self.previous is None), last, drains, self.rows, zeroed, active, outs, 0]
        words += even(self.ks) + even([len(t) for t in self.segments])
        for t in self.segments:
            for segment in t + [(0, 0, 0, 0, 0)] * (self.args['NSEG'] - len(t)):
                words += list(segment)
        return words

    # docs/writing-a-service.md#the-lstm
    def __call__(self, steps, e, out):
        s, L = self.tick, self.L
        send = self.S[s % SLOTS][:L * self.rows * self.HR]
        crossing = fused.exchange(send, self.landings, _stream.polls)
        active = sum(1 << l for l in range(L) if 0 <= s - l < steps)
        zeroed = sum(1 << l for l in range(L) if s < l)
        self._encode(e, out, send, crossing, self._words(0, int(out is not None), zeroed, active, int(out is not None)))
        self.sent, self.previous, self.tick = send, crossing, s + 1

    # docs/writing-a-service.md#the-lstm
    def land(self, e, out):
        self._encode(e, out, self.sent, None, self._words(1, int(out is not None), 0, 0, int(out is not None)))

    # docs/writing-a-service.md#the-lstm
    def _encode(self, e, out, send, crossing, words):
        prev = self.sent if self.sent is not None else send
        fused.encode(self.handle, [self.W, self.bias, e, out if out is not None else self.C, self.C, prev, send,
                                   self.arrivals, self.landing],
                     words, 15, self.args['G'], 1, self.args['THREADS'], len(self.shares) - 1,
                     land=self.previous, publish=crossing)
