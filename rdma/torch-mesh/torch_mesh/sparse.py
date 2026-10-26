"""Tile-sparse linear products in the mesh library (metal-microbench docs/kernels.md "Tile sparsity"), operand gated: a
weight carries a closure over its tiles and an explicitly configured lowering.

The mask is a closure: a predicate KEPT(k, n) over a weight's tiles (k its output tile, n its input tile in torch's
[out, in] layout), compiled into every kernel (sparse.metal), its environment captured as constants (an arithmetic
pattern: block-diagonal, banded) or a bitmap (one bit a tile).  The kernels are the dense tensor-operation products
(MetalPerformancePrimitives matmul2d), each block's loop visiting the tiles its closure keeps, encoded on torch's own
stream (_stream.encode: torch's open compute encoder, no host wait).  One closure serves the three products of
y = x . (W (.) M)^T, its arguments' roles exchanged, and the derivative is exact and as sparse as the product.

What a tensor is decides, not what it is said to be: `survey` measures a weight's tile energy and, given the rows it
multiplies, each tile's contribution to the product and the product's error at each density; `sparsify` induces a
closure as a quantizer induces a lattice, the fewest tiles whose product on held-out rows stays within a tolerance
(None where none fewer than all does).  A trained model's dense weights are dense at tile granularity (E2B's FFN:
docs/kernels.md); a model trained toward a tile support (a projection distance, metal-microbench docs/kernels.md
"Learning the support") is not.

`SparseLinear` takes its lowering as an explicit configuration operand. The
whole-function observations can inform the next configuration. Numerical calls
neither time candidates nor change their lowering.
"""
import math
from pathlib import Path

import torch

from . import _stream

SOURCE = Path(__file__).with_name('sparse.metal').read_text()
BITMAP = '(bits[((n) * KB + (k)) >> 5] >> (((n) * KB + (k)) & 31)) & 1'
_EMPTY = None


class Closure:
    """A kept-tile predicate KEPT(k, n) of tile size `tile` (a multiple of 64): an MSL expression in k, n, KB, NB and,
    where it captures one, the bitmap `bits` (bit n * KB + k)."""

    def __init__(self, expression, tile, bits=None, name=None):
        if tile % 64:
            raise ValueError(f'closure: tile {tile} is no multiple of 64')
        self.expression, self.tile, self.name = expression, tile, name or expression
        self.bits = bits if bits is not None else torch.zeros(1, dtype=torch.int32, device='mps')
        self.kernels, self.evaluated = {}, {}

    def kernel(self, name, out):
        key = (name, out)
        if key not in self.kernels:
            text = SOURCE.replace('(TILE_MOD)', f'({self.expression})').replace('(OUT_TYPE)', 'half' if out == torch.float16 else 'float')
            self.kernels[key] = _stream.pipeline(text, name)
        return self.kernels[key]

    def evaluate(self, k, n):
        """The closure over the k/tile x n/tile grid (rows: k), evaluated once on the GPU (FlexAttention's
        create_block_mask): its bitmap words and its kept tiles' count."""
        if (k, n) not in self.evaluated:
            kb, nb = k // self.tile, n // self.tile
            words = torch.zeros((kb * nb + 31) // 32, dtype=torch.int32, device='mps')
            _encode(self, 'evaluate', [None, None, words], (0, k, n), (words.numel() + 63) // 64, 1, 64, torch.float32)
            self.evaluated[k, n] = words, int(unpack(words, kb, nb).sum())
        return self.evaluated[k, n]

    def mask(self, k, n):
        """The closure's [k/tile, n/tile] booleans."""
        return unpack(self.evaluate(k, n)[0], k // self.tile, n // self.tile)


def unpack(words, kb, nb):
    bit = torch.arange(kb * nb)
    on = ((words.cpu().to(torch.int64) & 0xffffffff)[bit // 32] >> (bit % 32)) & 1
    return on.reshape(nb, kb).T.bool()


def bitmap(mask, tile):
    """The closure of a mask ([K/tile, N/tile] booleans: rows the weight's output tiles): its bitmap captured."""
    kb, nb = mask.shape
    flat = mask.to('cpu', torch.bool).T.reshape(-1).to(torch.int64)
    words = torch.zeros((kb * nb + 31) // 32, dtype=torch.int64)
    words.index_add_(0, torch.arange(kb * nb) // 32, flat << (torch.arange(kb * nb) % 32))
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return Closure(BITMAP, tile, words.to('mps'), name=f'bitmap {float(mask.float().mean()):.3f}')


def named(spec, tile):
    """A closure by name: all, blockdiag:G (G diagonal groups of tiles), band:W (|k - n| <= W)."""
    kind, _, value = spec.partition(':')
    expression = {'all': 'true', 'blockdiag': f'(k) * {value or 1} / KB == (n) * {value or 1} / NB',
                  'band': f'abs((k) - (n)) <= {value or 0}'}[kind]
    return Closure(expression, tile, name=spec)


def _encode(closure, name, tensors, shape, gx, gy, threads, out, bits=None):
    """Kernel `name` of `closure` on `tensors` (buffers 0 to 2, None for none) and its bitmap (or `bits`), `shape`
    (M, K, N), encoded on torch's stream."""
    _stream.encode(closure.kernel(name, out), [*tensors, None, closure.bits if bits is None else bits],
                   [*shape, closure.tile], 3, gx, gy, threads)


def _rows(x):
    return x.reshape(-1, x.shape[-1]).contiguous()


def _padded(x2):
    """Rows [M, K] in whole blocks of 64, the padding zeros: the tensor operations read whole blocks (a slice past an
    operand's extent is not clipped), so a partial block is padded rather than read past its storage."""
    m = x2.shape[0]
    if m % 64 == 0:
        return x2
    out = torch.zeros(-(-m // 64) * 64, x2.shape[1], dtype=x2.dtype, device=x2.device)
    out[:m] = x2
    return out


def linear(x, w, closure, out=torch.float16):
    """x . (w (.) M)^T, w [N, K] (torch's layout), x [..., K] fp16: the closure's kernel (its input_grad form, w as
    its [K, N] operand)."""
    x2 = _rows(x)
    m, (n, k) = x2.shape[0], w.shape
    y = torch.empty(m, n, dtype=out, device='mps')
    _encode(closure, 'input_grad', [_padded(x2), w, y], (m, n, k), -(-m // 64), n // 64, 128, out)
    return y.reshape(*x.shape[:-1], n)


def input_grad(dy, w, closure, out=torch.float16):
    """dy . (w (.) M): the closure's kernel (its forward form)."""
    dy2 = _rows(dy)
    m, (n, k) = dy2.shape[0], w.shape
    dx = torch.empty(m, k, dtype=out, device='mps')
    _encode(closure, 'forward', [_padded(dy2), w, dx], (m, n, k), -(-m // 64), k // 64, 128, out)
    return dx.reshape(*dy.shape[:-1], k)


def weight_grad(x, dy, closure, n, k, out=torch.float32):
    """M (.) (dy^T . x), [n, k]: a block of a kept tile over every row, only the kept tiles' blocks dispatched."""
    words, kept = closure.evaluate(n, k)
    dw = torch.zeros(n, k, dtype=out, device='mps')
    if kept:
        side = closure.tile // 64
        dy2, x2 = _padded(_rows(dy)), _padded(_rows(x))
        _encode(closure, 'weight_grad_kept', [dy2, x2, dw], (x2.shape[0], n, k), kept * side * side, 1, 128, out, bits=words)
    return dw


class LinearFunction(torch.autograd.Function):
    """y = x . (w (.) M)^T with its exact backward, each product the closure's."""

    @staticmethod
    def forward(ctx, x, w, closure):
        ctx.save_for_backward(x, w)
        ctx.closure = closure
        return linear(x, w, closure)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy = dy.to(torch.float16)
        n, k = w.shape
        return (input_grad(dy, w, ctx.closure) if ctx.needs_input_grad[0] else None,
                weight_grad(x, dy, ctx.closure, n, k).to(w.dtype) if ctx.needs_input_grad[1] else None, None)


def expand(mask, tile, shape):
    """A tile mask [N/tile, K/tile] as an element mask of `shape`."""
    return mask.to('cpu').repeat_interleave(tile, 0).repeat_interleave(tile, 1)[:shape[0], :shape[1]]


def contributions(w, x, tile):
    """Each tile's contribution to the product x . w^T on the rows x: the squared norm of its own term, [N/tile, K/tile]
    (float32, on w's device)."""
    n, k = w.shape
    xs = x.float().reshape(x.shape[0], k // tile, tile)
    out = torch.zeros(n // tile, k // tile, device=w.device)
    for a in range(n // tile):
        block = w[a * tile:(a + 1) * tile].float().reshape(tile, k // tile, tile)
        out[a] = torch.einsum('rkt,ckt->krc', xs, block).pow(2).sum((1, 2))
    return out


def _keep(scores, count):
    flat = torch.zeros(scores.numel(), dtype=torch.bool)
    flat[scores.flatten().cpu().topk(count).indices] = True
    return flat.reshape(scores.shape)


def survey(w, x=None, tile=64, densities=(0.75, 0.5, 0.25)):
    """What weight w [N, K] is at tile granularity: the share of its energy its top tiles hold (by density), and, given
    rows x [M, K] it multiplies, the product's relative error when only each density's most contributing tiles are kept
    (the rows' own contributions: the tensor as it is used)."""
    n, k = w.shape
    energy = w.float().reshape(n // tile, tile, k // tile, tile).pow(2).sum((1, 3)).flatten().sort(descending=True).values
    out = {'tiles': energy.numel(), 'energy': {d: float(energy[:max(1, int(energy.numel() * d))].sum() / energy.sum()) for d in densities}}
    if x is not None:
        scores, y = contributions(w, x, tile), x.float() @ w.float().T
        out['error'] = {}
        for d in densities:
            m = expand(_keep(scores, max(1, int(round(scores.numel() * d)))), tile, w.shape).to(w.device)
            out['error'][d] = float((x.float() @ (w.float() * m).T - y).norm() / y.norm())
    return out


def sparsify(w, x, tolerance, tile=64, held=0.4):
    """The closure a tolerance induces on weight w [N, K] used on rows x [M, K]: the fewest tiles, by their
    contributions on the first rows, whose product on the last `held` of the rows stays within `tolerance` of the
    dense product's relative error (bisection over the kept count).  (closure, density, error), the closure None
    where no fewer than all tiles hold."""
    cut = max(1, int(x.shape[0] * (1 - held)))
    fit, test = x[:cut], x[cut:]
    scores, y = contributions(w, fit, tile), test.float() @ w.float().T
    total = scores.numel()

    def error(count):
        m = expand(_keep(scores, count), tile, w.shape).to(w.device)
        return float((test.float() @ (w.float() * m).T - y).norm() / y.norm())

    low, high = 1, total
    if error(total - 1) > tolerance:
        return None, 1.0, 0.0
    while low < high:
        mid = (low + high) // 2
        if error(mid) <= tolerance:
            high = mid
        else:
            low = mid + 1
    return bitmap(_keep(scores, low), tile), low / total, error(low)


class SparseLinear(torch.nn.Module):
    """A configured dense or tile lowering; whole-function observations choose operands."""
    def __init__(self, weight, closure, bias=None, tile=64, *, lowering):
        super().__init__()
        n, k = weight.shape
        if n % tile or k % tile:
            raise ValueError(f'SparseLinear: [{n}, {k}] is not in whole {tile}-tiles (pad the weight)')
        self.closure = closure if isinstance(closure, Closure) else bitmap(closure, tile)
        mask = self.closure.mask(n, k)
        self.weight = torch.nn.Parameter((weight.float() * expand(mask, tile, weight.shape).to(weight.device)).half(), requires_grad=False)
        self.bias = bias

        if lowering not in ('dense', 'tiles'):
            raise ValueError('SparseLinear lowering must be dense or tiles')
        self.lowering = lowering

    def forward(self, x):
        x16 = x.to(torch.float16)
        y = torch.nn.functional.linear(x16, self.weight) if self.lowering == 'dense' else linear(x16, self.weight, self.closure)
        return y if self.bias is None else y + self.bias
