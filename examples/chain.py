import time
from pathlib import Path

import torch
from torch_mesh import _stream
from torch_mesh.streamed import instance, node

SOURCE = Path(__file__).with_name('chain.metal').read_text()
TILE = 8
ARGS = ('TILE', 'K', 'S', 'G', 'KS0', 'KS1', 'RT', 'RG', 'THREADS', 'PATIENCE')
_PIPELINES = {}


# docs/writing-a-service.md#the-step-kernel
def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


# docs/writing-a-service.md#the-step-kernel
def plans(K, S, rows, threads, patience):
    G = 1 << (node()[1].bit_length() - 1)
    while G > 1 and ((K // TILE) % G or (S // TILE) % G):
        G //= 2
    if K % TILE or S % TILE or rows % TILE or threads % 32:
        raise ValueError(f'chain step: width {K}, share {S} and rows {rows} are multiples of {TILE}, threads {threads} of 32')
    RT, SG = rows // TILE, threads // 32
    return [dict(TILE=TILE, K=K, S=S, G=G, RT=RT, RG=RG, THREADS=threads, PATIENCE=patience,
                 KS0=_ksplit((S // G // TILE) * (RT // RG), K // TILE, SG), KS1=_ksplit((K // G // TILE) * (RT // RG), S // TILE, SG))
            for RG in (1 << i for i in range(RT.bit_length())) if RT % RG == 0]


# docs/writing-a-service.md#the-step-kernel
def _pipeline(args):
    name, line = instance('chain_step', *(args[a] for a in ARGS))
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


class Step:
    # docs/writing-a-service.md#the-step-kernel
    def __init__(self, a, b, rows, threads, patience, trials):
        S, K = a.shape
        tiled = lambda w: w.reshape(w.shape[0] // TILE, TILE, w.shape[1] // TILE, TILE).permute(2, 0, 1, 3).contiguous()
        self.At, self.Bt = tiled(a.T), tiled(b)
        self.X = torch.empty(rows, K, dtype=torch.half, device='mps')
        self.Y = torch.empty(rows, S, dtype=torch.half, device='mps')
        candidates = plans(K, S, rows, threads, patience)
        self.stamps = torch.zeros(candidates[0]['G'], dtype=torch.int32, device='mps')
        self.err = torch.zeros(1, dtype=torch.int32, device='mps')
        self.phase = 0
        e, p = torch.zeros(2, rows, K, dtype=torch.half, device='mps')
        timed = []
        for args in candidates:
            self.args, self.handle = args, _pipeline(args)
            self(e, p, p, True)
            torch.mps.synchronize()
            began = time.perf_counter()
            for _ in range(trials):
                self(e, p, p, False)
            torch.mps.synchronize()
            timed.append((time.perf_counter() - began, args))
        self.check()
        self.args = min(timed, key=lambda t: t[0])[1]
        self.handle = _pipeline(self.args)

    # docs/writing-a-service.md#the-step-kernel
    def __call__(self, e, z, p, first):
        _stream.encode(self.handle, [self.At, self.Bt, e, z, self.X, self.Y, p, self.stamps, self.err],
                       [int(first), self.phase], 15, self.args['G'], 1, self.args['THREADS'])
        self.phase += 2

    # docs/writing-a-service.md#the-step-kernel
    def check(self):
        cause = int(self.err.item())
        if cause:
            raise RuntimeError(f'chain step {self.args}: a wait ran past its patience (causes {cause:#x})')
