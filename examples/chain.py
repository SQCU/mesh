import time
from pathlib import Path

import torch
from torch_mesh import _stream
from torch_mesh.streamed import instance, node

SOURCE = Path(__file__).with_name('chain.metal').read_text()
TILE = 8
ARGS = ('TILE', 'K', 'S', 'G', 'RG', 'THREADS', 'PATIENCE')
_PIPELINES = {}


# docs/writing-a-service.md#the-step-kernel
def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


# docs/writing-a-service.md#the-step-kernel
def workers(K, S):
    cores = node()[1]
    G = 1 << (cores.bit_length() - 1)
    unit = TILE * G
    wrong = [(name, n) for name, n in (('width', K), ('share', S)) if n % unit]
    if wrong:
        nearest = '; '.join(f'{name} {n}: {" or ".join(str(m) for m in (n // unit * unit, (n // unit + 1) * unit) if m)}'
                            for name, n in wrong)
        raise ValueError(f'chain step: {cores} GPU cores run {G} workers, which take a width and a share that are multiples '
                         f'of {unit}; the nearest that are, {nearest}')
    return G


# docs/writing-a-service.md#the-step-kernel
def plans(K, S, rows, threads, patience):
    if rows < TILE or rows % TILE or threads % 32:
        raise ValueError(f'chain step: rows {rows} are a multiple of {TILE}, threads {threads} of 32')
    G, RT = workers(K, S), rows // TILE
    return [dict(TILE=TILE, K=K, S=S, G=G, RG=RG, THREADS=threads, PATIENCE=patience)
            for RG in (1 << i for i in range(RT.bit_length())) if RT % RG == 0]


# docs/writing-a-service.md#the-step-kernel
def words(args, rows, capacity):
    if rows % (TILE * args['RG']) or rows > capacity:
        raise ValueError(f'chain step: instance RG {args["RG"]} takes rows that are multiples of {TILE * args["RG"]} up '
                         f'to the capacity {capacity}, not {rows}')
    groups, G, SG = rows // (TILE * args['RG']), args['G'], args['THREADS'] // 32
    K, S = args['K'], args['S']
    return [rows, _ksplit((S // G // TILE) * groups, K // TILE, SG), _ksplit((K // G // TILE) * groups, S // TILE, SG)]


# docs/writing-a-service.md#the-step-kernel
def _pipeline(args):
    name, line = instance('chain_step', *(args[a] for a in ARGS))
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


class Step:
    # docs/writing-a-service.md#the-step-kernel
    def __init__(self, a, b, rows, capacity, threads, patience, trials):
        S, K = a.shape
        tiled = lambda w: w.reshape(w.shape[0] // TILE, TILE, w.shape[1] // TILE, TILE).permute(2, 0, 1, 3).contiguous()
        self.At, self.Bt = tiled(a.T), tiled(b)
        self.capacity = capacity
        self.X = torch.empty(capacity, K, dtype=torch.half, device='mps')
        self.Y = torch.empty(capacity, S, dtype=torch.half, device='mps')
        candidates = plans(K, S, rows, threads, patience)
        self.stamps = torch.zeros(candidates[0]['G'], dtype=torch.int32, device='mps')
        self.err = torch.zeros(1, dtype=torch.int32, device='mps')
        self.phase = 0
        e, p = torch.zeros(2, rows, K, dtype=torch.half, device='mps')
        self.kernel_alone = {}
        for args in candidates:
            self.use(args)
            self.plan(rows)
            self(e, p, p, True)
            torch.mps.synchronize()
            began = time.perf_counter()
            for _ in range(trials):
                self(e, p, p, False)
            torch.mps.synchronize()
            self.kernel_alone[args['RG']] = (time.perf_counter() - began) / trials
        self.check()
        self.use(min(candidates, key=lambda args: self.kernel_alone[args['RG']]))
        self.plan(rows)

    # docs/writing-a-service.md#the-step-kernel
    def use(self, args):
        self.args, self.handle, self.planned = args, _pipeline(args), {}

    # docs/writing-a-service.md#the-step-kernel
    def plan(self, rows):
        self.planned[rows] = words(self.args, rows, self.capacity)

    # docs/writing-a-service.md#the-step-kernel
    def __call__(self, e, z, p, first):
        _stream.encode(self.handle, [self.At, self.Bt, e, z, self.X, self.Y, p, self.stamps, self.err],
                       [int(first), self.phase, *self.planned[e.shape[0]]], 15, self.args['G'], 1, self.args['THREADS'])
        self.phase += 2

    # docs/writing-a-service.md#the-step-kernel
    def check(self):
        cause = int(self.err.item())
        if cause:
            raise RuntimeError(f'chain step {self.args}: a wait ran past its patience (causes {cause:#x})')
