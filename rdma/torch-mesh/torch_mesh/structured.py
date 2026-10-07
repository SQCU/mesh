"""Structured random orthogonal transforms on Apple GPUs (structured.metal): a seeded rotation of a matrix's rows or
columns in O(n log n) a row, exactly orthogonal, from fast Walsh-Hadamard transforms (the randomized Hadamard
transform of structured random projections [Ailon & Chazelle 2006; Le, Sarlos & Smola 2013, Fastfood]).

A width n = B * N, N its largest power-of-two factor up to 4096: a round is diag(signs) then H_N / sqrt N on each of
the B blocks (structured.metal, a threadgroup a block, in torch's stream) then a random orthogonal B x B mixing the
blocks; two rounds, each its own signs and mixing, make the rotation.  Every factor is orthogonal, so the rotation
preserves norms, singular values and the other side's singular vectors exactly: R . E keeps E's input-side principal
components and spectrum with its output directions random, E . R^T the converse, R1 . E . R2^T its spectrum alone,
none of them decomposing E.  The same seed gives the same rotation, so a kernel can regenerate it rather than store it.

  rotation(n, seed) -> Rotation; .rows(x) rotates each row of x [m, n]; .columns(x) each column of x [n, k]"""
from pathlib import Path

import torch

from . import _stream

SOURCE = Path(__file__).with_name('structured.metal').read_text()
_KERNEL = None


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

    def rows(self, x):
        """Each row of x [m, n] rotated (x R^T)."""
        x = x.float().contiguous()
        m = x.shape[0]
        for sign, mix in zip(self.signs, self.mixes):
            y = torch.empty_like(x)
            threads = min(1024, max(32, self.N // 2))
            _stream.encode(_kernel(), [x, y, sign], [self.N, self.B, 0, 0], 3, m * self.B, 1, threads)
            x = torch.einsum('bc,mcn->mbn', mix, y.view(m, self.B, self.N)).reshape(m, self.n) if self.B > 1 else y
        return x

    def columns(self, x):
        """Each column of x [n, k] rotated (R x)."""
        return self.rows(x.T.contiguous()).T


def _check():
    torch.manual_seed(0)
    for n in (1536, 6144, 12288, 4096):
        R = Rotation(n, seed=7)
        x = torch.randn(64, n, device='mps')
        y = R.rows(x)
        eye = R.rows(torch.eye(n, device='mps')[:256])
        gram = eye @ eye.T
        print(n, 'norms kept', float((y.norm(dim=1) / x.norm(dim=1) - 1).abs().max()),
              'orthogonality', float((gram - torch.eye(256, device='mps')).abs().max()),
              'mixed (max |R_ij| * sqrt n)', round(float(eye.abs().max()) * n ** 0.5, 2))


if __name__ == '__main__':
    _check()
