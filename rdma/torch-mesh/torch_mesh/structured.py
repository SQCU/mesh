"""Structured random orthogonal transforms on Apple GPUs (structured.metal): a seeded rotation of a matrix's rows or
columns in O(n log n) a row, exactly orthogonal, from fast Walsh-Hadamard transforms (the randomized Hadamard
transform of structured random projections [Ailon & Chazelle 2006; Le, Sarlos & Smola 2013, Fastfood]).

A width n = B * N, N its largest power-of-two factor within the configured block width and device memory: a round is
diag(signs) then H_N / sqrt N on each of the B blocks (one prepared threadgroup per block) then a random orthogonal B x B mixing the
blocks; two rounds, each its own signs and mixing, make the rotation.  Every factor is orthogonal, so the rotation
preserves norms, singular values and the other side's singular vectors exactly: R . E keeps E's input-side principal
components and spectrum with its output directions random, E . R^T the converse, R1 . E . R2^T its spectrum alone,
none of them decomposing E.  The same seed gives the same rotation, so a kernel can regenerate it rather than store it.

  Rotation(n, seed); .rows(x) rotates each row of x [m, n] (x R^T); .columns(x) each column of x [n, k] (R x);
  .rows_inverse(y) undoes .rows (y R)"""
from functools import lru_cache
from pathlib import Path

import torch

from . import _stream
from .compilation import operands

SOURCE = Path(__file__).with_name('structured.metal').read_text()


@torch.library.custom_op('torch_mesh::fwht', mutates_args=())
def fwht(x: torch.Tensor, signs: torch.Tensor, width: int) -> torch.Tensor:
    raise RuntimeError('structured transforms must be prepared as part of a declarations.Function')


@fwht.register_fake
def _shape(x, signs, width):
    return torch.empty_like(x)


def _context(ctx, inputs, output):
    x, signs, ctx.width = inputs
    ctx.save_for_backward(signs)


def _backward(ctx, grad):
    signs, = ctx.saved_tensors
    return fwht(grad, torch.ones_like(signs), ctx.width) * signs, None, None


fwht.register_autograd(_backward, setup_context=_context)


def prepare(x, signs, width, *, out=None):
    from .placement import output
    import struct
    if x.ndim != 2 or signs.shape != x.shape[-1:] or x.dtype != torch.float32 or signs.dtype != x.dtype:
        raise ValueError('fwht binds fp32 rows and one fp32 sign per column')
    if width <= 0 or width & (width - 1) or x.shape[-1] % width:
        raise ValueError('fwht blocks must be power-of-two factors of the row width')
    y = output(x.shape, x.dtype, x.device, out)
    from .layout import index_source
    handle = _kernel(width, index_source({0: x, 1: y, 2: signs}))
    maximum, lanes, _ = _stream.pipeline_facts(handle)
    threads = min(maximum, max(lanes, width // 2))
    blocks = x.shape[-1] // width
    return y, [(handle, [x, y, signs], {3: struct.pack('2I', width, blocks)},
                (x.shape[0] * blocks, 1, 1), (threads, 1, 1), False,
                5 | (2 << _stream.compilation['ACCESS_WRITE_SHIFT']))] if x.numel() else []


@lru_cache(None)
def _kernel(width, indices):
    _, _, memory = _stream.device_facts()
    if width * torch.float32.itemsize > memory:
        raise ValueError('fwht block exceeds the device threadgroup memory; select a smaller block during graph construction')
    handle = _stream.pipeline(indices + f'#define MESH_FWHT_WIDTH {width}\n' + SOURCE, 'fwht')
    if _stream.pipeline_facts(handle)[2] > memory:
        raise ValueError('fwht specialized workspace exceeds the device threadgroup memory')
    return handle


def _factor(n, block_width):
    if n <= 0 or block_width <= 0:
        raise ValueError('rotation width and block width must be positive')
    _, _, memory = _stream.device_facts()
    limit = min(block_width, memory // torch.float32.itemsize)
    if limit < 1:
        raise ValueError('rotation requires a device with fp32 threadgroup storage')
    N = 1
    while n % (N * 2) == 0 and N * 2 <= limit:
        N *= 2
    return N, n // N


class Rotation:
    """A seeded random orthogonal transform of width n (module docstring)."""

    def __init__(self, n, seed, rounds=2, device='mps', *, block_width=None):
        self.n, self.N, self.B = n, *_factor(n, operands()['FWHT_BLOCK_WIDTH'] if block_width is None else block_width)
        g = torch.Generator().manual_seed(int(seed))
        self.signs = [(torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device) for _ in range(rounds)]
        self.mixes = [torch.linalg.qr(torch.randn(self.B, self.B, generator=g, dtype=torch.float64))[0].float().to(device)
                      for _ in range(rounds)]
        self.ones = torch.ones(self.n, device=device)

    def rows(self, x):
        """Each row of x [m, n] rotated (x R^T)."""
        x = x.float()
        m = x.shape[0]
        for sign, mix in zip(self.signs, self.mixes):
            y = fwht(x, sign, self.N)
            x = torch.einsum('bc,mcn->mbn', mix, y.view(m, self.B, self.N)).reshape(m, self.n) if self.B > 1 else y
        return x

    def columns(self, x):
        """Each column of x [n, k] rotated (R x)."""
        return self.rows(x.T).T

    def rows_inverse(self, y):
        """Each row of y [m, n] rotated back (y R): the rounds undone in reverse, each block's transform its own
        inverse (H_N / sqrt N is symmetric and orthogonal), then the signs."""
        y = y.float()
        m = y.shape[0]
        for sign, mix in reversed(list(zip(self.signs, self.mixes))):
            if self.B > 1:
                y = torch.einsum('bc,mbn->mcn', mix, y.view(m, self.B, self.N)).reshape(m, self.n)
            x = fwht(y, self.ones, self.N)
            y = x * sign
        return y
