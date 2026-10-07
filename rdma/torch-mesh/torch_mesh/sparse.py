"""Tile-sparse linear products in the mesh library (metal-microbench docs/kernels.md "Tile sparsity"), operand gated: a
weight carries a closure over its tiles, and its products route between the dense reference and the closure's kernels
by their measured prices.

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

Routing.  `SparseLinear` holds one copy of the masked weight; each call runs on the lowering its prices give the
least time: `dense` (torch's product of the masked weight, the reference) or `tiles` (the closure's kernel).  A
lowering's price is a line in its work in units of 64 x 64 x 64 multiply-adds, padded to whole tiles (dense: every
tile, rows in blocks of 64; tiles: the kept tiles), fitted to every observation of every SparseLinear of this process
(`Prices`, one per process: the node's evidence, no identity of operation or shape); a shape's own measurements
outrank the line: the two lowerings explored in alternation (each its first call untimed, then `window` timed, so a
GPU's clocks ramping weigh on neither), the measured faster runs, sampled again every `sample` calls; where the closure's kernel leads within the resolution,
the alternation runs on (up to four windows) before the reference is kept.  A lowering the line prices
past the other by more than `margin` at a shape is not explored there, and where a shape's two medians lie within
their resolution (twice the larger relative standard error, at least 2 %) the reference runs: the closure's kernel
replaces it only where it measures faster.  So a call is never slower than the reference beyond the exploration of a
lowering the line could not rule out and the measurement's resolution.

  python -m torch_mesh.sparse check    every closure's three products against float64, rows not a multiple of 64
  python -m torch_mesh.sparse time     the lowerings' times by density and the routing's choice"""
import argparse
import math
import statistics
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


class Prices:
    """Each lowering's price in its work (units of 64 x 64 x 64 multiply-adds, padded to whole tiles): a least-squares
    line through every observation (seconds) of this process, by lowering."""

    def __init__(self):
        self.seen = {}

    def observe(self, lowering, work, seconds):
        self.seen.setdefault(lowering, []).append((float(work), float(seconds)))

    def predict(self, lowering, work):
        points = self.seen.get(lowering) or []
        if len({w for w, _ in points}) < 2:
            return None
        n = len(points)
        mw, ms = sum(w for w, _ in points) / n, sum(s for _, s in points) / n
        slope = sum((w - mw) * (s - ms) for w, s in points) / sum((w - mw) ** 2 for w, _ in points)
        return max(0.0, ms + slope * (work - mw))


PRICES = Prices()


class SparseLinear(torch.nn.Module):
    """y = x . (W (.) M)^T + b, W [out, in] fp16 on MPS and its closure (or a tile mask [out/tile, in/tile]), each call
    routed to its cheaper lowering (module docstring)."""

    def __init__(self, weight, closure, bias=None, tile=64, prices=None, window=5, sample=64, margin=0.25):
        super().__init__()
        n, k = weight.shape
        if n % tile or k % tile:
            raise ValueError(f'SparseLinear: [{n}, {k}] is not in whole {tile}-tiles (pad the weight)')
        self.closure = closure if isinstance(closure, Closure) else bitmap(closure, tile)
        mask = self.closure.mask(n, k)
        self.weight = torch.nn.Parameter((weight.float() * expand(mask, tile, weight.shape).to(weight.device)).half(), requires_grad=False)
        self.bias = bias
        self.kept, self.tile = int(mask.sum()), tile
        self.prices, self.window, self.sample, self.margin = prices or PRICES, window, sample, margin
        self.seen, self.calls, self.pending, self.decided, self.rounds, self.rounds_most = {}, {}, [], {}, {}, 4

    def work(self, lowering, rows):
        n, k = self.weight.shape
        blocks = -(-rows // 64)
        return blocks * (self.kept * (self.tile // 64) ** 2 if lowering == 'tiles' else (n // 64) * (k // 64))

    def route(self, rows):
        """The lowering of a call of `rows` rows: the price's where it rules the other out, else the two explored in
        alternation (each its first call untimed, then `window` timed), then the measured faster beyond the resolution,
        ties to the reference."""
        issued = {l: self.calls.get((rows, l), 0) for l in ('dense', 'tiles')}
        priced = {l: self.prices.predict(l, self.work(l, rows)) for l in ('dense', 'tiles')}
        if all(p is not None for p in priced.values()) and min(issued.values()) == 0:
            best = min(priced, key=priced.get)
            if priced['tiles' if best == 'dense' else 'dense'] > priced[best] * (1 + self.margin):
                self.decided[rows] = best
                return best
        rounds = self.rounds.get(rows, 1)
        if min(issued.values()) <= self.window * rounds:
            return 'dense' if issued['dense'] <= issued['tiles'] else 'tiles'
        if self.pending:
            torch.mps.synchronize()
            self._settle()
        seen = self.seen.get(rows, {})
        latest = {l: seen.get(l, [])[1 - self.window * rounds - 1:] for l in ('dense', 'tiles')}
        if any(len(v) < 2 for v in latest.values()):
            return 'dense'
        median = {l: statistics.median(v) for l, v in latest.items()}
        error = lambda v: statistics.stdev(v) / math.sqrt(len(v)) / statistics.mean(v)
        resolution = max(0.02, 2 * max(error(latest['dense']), error(latest['tiles'])))
        if median['tiles'] < median['dense'] * (1 - resolution):
            self.decided[rows] = 'tiles'
        elif median['tiles'] < median['dense'] and rounds < self.rounds_most:
            self.rounds[rows] = rounds + 1
            return 'dense' if issued['dense'] <= issued['tiles'] else 'tiles'
        else:
            self.decided[rows] = 'dense'
        return self.decided[rows]

    def _settle(self):
        still = []
        for rows, lowering, begin, end in self.pending:
            if end.query():
                seconds = begin.elapsed_time(end) / 1e3
                self.seen.setdefault(rows, {}).setdefault(lowering, []).append(seconds)
                self.prices.observe(lowering, self.work(lowering, rows), seconds)
            else:
                still.append((rows, lowering, begin, end))
        self.pending = still

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        lowering = self.decided.get(rows)
        if lowering is None:
            lowering = self.route(rows)
        count = self.calls[rows, lowering] = self.calls.get((rows, lowering), 0) + 1
        timed = count > 1 and (rows not in self.decided or count % self.sample == 0)
        if self.pending and count % self.sample == 0:
            self._settle()
        if timed:
            begin, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
            begin.record()
        x16 = x.to(torch.float16)
        y = torch.nn.functional.linear(x16, self.weight) if lowering == 'dense' else linear(x16, self.weight, self.closure)
        if timed:
            end.record()
            self.pending.append((rows, lowering, begin, end))
        return y if self.bias is None else y + self.bias


def _check(args):
    torch.manual_seed(0)
    ok = True
    for m, n, k, tile in ((711, 1024, 768, 64), (256, 512, 1024, 128), (33, 256, 256, 64)):
        x = (torch.randn(m, k) / 4).half().to('mps')
        w = (torch.randn(n, k) / k ** 0.5).half().to('mps')
        dy = (torch.randn(m, n) / 4).half().to('mps')
        for closure in (named('all', tile), named('blockdiag:4', tile), named('band:1', tile),
                        bitmap(torch.rand(n // tile, k // tile) < 0.4, tile)):
            mask = expand(closure.mask(n, k), tile, w.shape).double()
            wm = w.cpu().double() * mask
            X, DY = x.cpu().double(), dy.cpu().double()
            got = [linear(x, w, closure).cpu().double(), input_grad(dy, w, closure).cpu().double(),
                   weight_grad(x, dy, closure, n, k).cpu().double()]
            want = [X @ wm.T, DY @ wm, (DY.T @ X) * mask]
            errors = [float((g - r).norm() / max(r.norm(), 1e-30)) for g, r in zip(got, want)]
            fine = all(e < 2e-3 for e in errors)
            ok &= fine
            print(f'{m}x{n}x{k} tile {tile} {closure.name}: linear {errors[0]:.1e} input_grad {errors[1]:.1e} weight_grad {errors[2]:.1e}'
                  f' {"ok" if fine else "FAILED"}')
    return ok


def _time(args):
    torch.manual_seed(0)
    m, n, k, tile = args.m, args.n, args.k, args.tile
    x = (torch.randn(m, k) / 4).half().to('mps')
    w = (torch.randn(n, k) / k ** 0.5).half().to('mps')

    def ms(fn, reps=20):
        for _ in range(3):
            fn()
        torch.mps.synchronize()
        times = []
        for _ in range(reps):
            b, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
            b.record()
            fn()
            e.record()
            torch.mps.synchronize()
            times.append(b.elapsed_time(e))
        return statistics.median(times)

    ms(lambda: torch.nn.functional.linear(x, w), reps=100)
    reference = ms(lambda: torch.nn.functional.linear(x, w))
    print(f'{m} x {n} x {k}, tile {tile}: the dense reference (the unmasked weight, after a warm-up) {reference:.3f} ms')
    for density in (1.0, 0.9, 0.75, 0.5, 0.25, 0.125):
        mask = torch.rand(n // tile, k // tile) < density if density < 1 else torch.ones(n // tile, k // tile, dtype=torch.bool)
        layer = SparseLinear(w, mask, tile=tile)
        tiles = ms(lambda: linear(x, layer.weight, layer.closure))
        dense = ms(lambda: torch.nn.functional.linear(x, layer.weight))
        for _ in range(12):
            layer(x)
        torch.mps.synchronize()
        layer._settle()
        routed = ms(lambda: layer(x))
        print(f'density {float(mask.float().mean()):.3f}: tiles {tiles:.3f} ms, dense {dense:.3f}, routed {routed:.3f} '
              f'({layer.route(m)}), against the reference {reference / routed:.2f}x')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['check', 'time'])
    parser.add_argument('--m', type=int, default=1024)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--tile', type=int, default=64)
    args = parser.parse_args()
    if args.action == 'check':
        raise SystemExit(0 if _check(args) else 1)
    _time(args)


if __name__ == '__main__':
    main()
