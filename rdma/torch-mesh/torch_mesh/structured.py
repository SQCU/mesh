"""Structured random orthogonal transforms on Apple GPUs (structured.metal): a seeded rotation of a matrix's rows or
columns in O(n log n) a row, exactly orthogonal, from fast Walsh-Hadamard transforms (the randomized Hadamard
transform of structured random projections [Ailon & Chazelle 2006; Le, Sarlos & Smola 2013, Fastfood]).

A width n = B * N, N its largest power-of-two factor up to 4096: a round is diag(signs) then H_N / sqrt N on each of
the B blocks (structured.metal, a threadgroup a block, in torch's stream) then a random orthogonal B x B mixing the
blocks; two rounds, each its own signs and mixing, make the rotation.  Every factor is orthogonal, so the rotation
preserves norms, singular values and the other side's singular vectors exactly: R . E keeps E's input-side principal
components and spectrum with its output directions random, E . R^T the converse, R1 . E . R2^T its spectrum alone,
none of them decomposing E.  The same seed gives the same rotation, so a kernel can regenerate it rather than store it.

  Rotation(n, seed); .rows(x) rotates each row of x [m, n] (x R^T); .columns(x) each column of x [n, k] (R x);
  .rows_inverse(y) undoes .rows (y R)"""
from pathlib import Path

import torch

from . import _stream

SOURCE = Path(__file__).with_name('structured.metal').read_text()
_KERNEL = None


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
    return fwht(grad.contiguous(), torch.ones_like(signs), ctx.width) * signs, None, None


fwht.register_autograd(_backward, setup_context=_context)


def prepare(x, signs, width, *, out=None):
    from .placement import output
    import struct
    y = output(x.shape, x.dtype, x.device, out)
    maximum, _, _ = _stream.pipeline_facts(_kernel())
    threads = min(maximum, max(_stream.compilation['SIMD_WIDTH'], width // 2))
    blocks = x.shape[-1] // width
    return y, [(_kernel(), [x, y, signs], {3: struct.pack('7I', width, blocks, *x.stride(), *y.stride(), signs.stride(0))},
                (x.shape[0] * blocks, 1, 1), (threads, 1, 1), False,
                5 | (2 << _stream.compilation['ACCESS_WRITE_SHIFT']))] if x.numel() else []


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _stream.pipeline(SOURCE, 'fwht')
    return _KERNEL


def _factor(n):
    N = 1
    while n % (N * 2) == 0 and N * 2 <= 4096:
        N *= 2
    return N, n // N


class Rotation:
    """A seeded random orthogonal transform of width n (module docstring)."""

    def __init__(self, n, seed, rounds=2, device='mps'):
        self.n, self.N, self.B = n, *_factor(n)
        g = torch.Generator().manual_seed(int(seed))
        self.signs = [(torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device) for _ in range(rounds)]
        self.mixes = [torch.linalg.qr(torch.randn(self.B, self.B, generator=g, dtype=torch.float64))[0].float().to(device)
                      for _ in range(rounds)]
        self.ones = torch.ones(self.n, device=device)

    def rows(self, x):
        """Each row of x [m, n] rotated (x R^T)."""
        x = x.float().contiguous()
        m = x.shape[0]
        for sign, mix in zip(self.signs, self.mixes):
            y = fwht(x, sign, self.N)
            x = torch.einsum('bc,mcn->mbn', mix, y.view(m, self.B, self.N)).reshape(m, self.n) if self.B > 1 else y
        return x

    def columns(self, x):
        """Each column of x [n, k] rotated (R x)."""
        return self.rows(x.T.contiguous()).T

    def rows_inverse(self, y):
        """Each row of y [m, n] rotated back (y R): the rounds undone in reverse, each block's transform its own
        inverse (H_N / sqrt N is symmetric and orthogonal), then the signs."""
        y = y.float().contiguous()
        m = y.shape[0]
        for sign, mix in reversed(list(zip(self.signs, self.mixes))):
            if self.B > 1:
                y = torch.einsum('bc,mbn->mcn', mix, y.view(m, self.B, self.N)).reshape(m, self.n).contiguous()
            x = fwht(y, self.ones, self.N)
            y = x * sign
        return y
