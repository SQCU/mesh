from pathlib import Path

import torch
from torch_mesh import _stream
from torch_mesh.streamed import instance, node

SOURCE = Path(__file__).with_name('chain.metal').read_text()
TILE = 8
RG = 1
ARGS = ('TILE', 'K', 'S', 'G', 'RG', 'THREADS', 'PATIENCE')
_PIPELINES = {}


# docs/writing-a-service.md#the-step-kernel
def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


# docs/writing-a-service.md#the-step-kernel
def _nearest(n, least):
    return sorted({max(least, n // TILE * TILE), max(least, -(-n // TILE) * TILE)})


# docs/writing-a-service.md#the-step-kernel
def workers(K, S):
    G = node()[1]
    least = TILE * G
    wrong = [(name, n) for name, n in (('width', K), ('share', S)) if n % TILE or n < least]
    if wrong:
        nearest = '; '.join(f'{name} {n}: {" or ".join(map(str, _nearest(n, least)))}' for name, n in wrong)
        raise ValueError(f'chain step: each of the {G} GPU cores runs a worker of at least one tile of {TILE} columns, so a '
                         f'width and a share are multiples of {TILE} and at least {least}; the nearest that run, {nearest}')
    return G


# docs/writing-a-service.md#the-step-kernel
def widest(K, S):
    args = dict(TILE=TILE, K=K, S=S, G=workers(K, S), RG=RG, THREADS=32, PATIENCE=_stream.polls)
    _pipeline(args)
    while True:
        wider = dict(args, THREADS=2 * args['THREADS'])
        try:
            _pipeline(wider)
        except RuntimeError:
            return args
        args = wider


# docs/writing-a-service.md#the-step-kernel
def words(args, rows, capacity):
    if rows < TILE * args['RG'] or rows % (TILE * args['RG']) or rows > capacity:
        raise ValueError(f'chain step: instance RG {args["RG"]} takes rows that are multiples of {TILE * args["RG"]} from '
                         f'{TILE * args["RG"]} up to the capacity {capacity}, not {rows}')
    groups, G, SG = rows // (TILE * args['RG']), args['G'], args['THREADS'] // 32
    kt, st = args['K'] // TILE, args['S'] // TILE
    return [rows, _ksplit(-(-st // G) * groups, kt, SG), _ksplit(-(-kt // G) * groups, st, SG)]


# docs/writing-a-service.md#the-step-kernel
def _pipeline(args):
    name, line = instance('chain_step', *(args[a] for a in ARGS))
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


class Step:
    # docs/writing-a-service.md#the-step-kernel
    def __init__(self, a, b, rows, capacity):
        S, K = a.shape
        tiled = lambda w: w.reshape(w.shape[0] // TILE, TILE, w.shape[1] // TILE, TILE).permute(2, 0, 1, 3).contiguous()
        self.At, self.Bt = tiled(a.T), tiled(b)
        self.capacity = capacity
        self.X = torch.empty(capacity, K, dtype=torch.half, device='mps')
        self.Y = torch.empty(capacity, S, dtype=torch.half, device='mps')
        self.args = widest(K, S)
        self.handle, self.planned = _pipeline(self.args), {}
        self.stamps = torch.zeros(self.args['G'], dtype=torch.int32, device='mps')
        self.err = torch.zeros(1, dtype=torch.int32, device='mps')
        self.phase = 0
        self.plan(rows)

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
