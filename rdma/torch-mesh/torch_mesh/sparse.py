"""Tile-sparse products declared to the common whole-function compiler.

A closure runs during preparation over weight-tile coordinates. Its fixed output
is a schedule of output tiles and reduction intervals. The three products share
one TensorOps kernel; tensor shapes and sparse tile boundaries supply exact tail
extents. No numerical call allocates padding, evaluates a predicate or scans a
bitmap. Layout requirements and output placement use the common kernel interface.
"""
from functools import lru_cache
import math

import torch

from .compilation import operands, api_layouts


class Closure:
    """A preparation function (output tile, input tile, output count, input count).

    Capture arithmetic parameters or a CPU bitmap in the function. The returned
    booleans describe immutable support for one compiled function execution.
    """
    def __init__(self, predicate, tile, *, name=None):
        if tile <= 0:
            raise ValueError('sparse tile extent must be positive')
        self.predicate, self.tile = predicate, tile
        self.name = name or getattr(predicate, '__name__', 'tile predicate')

    @lru_cache(None)
    def support(self, outputs, inputs):
        nr, nc = -(-outputs // self.tile), -(-inputs // self.tile)
        return tuple(tuple(bool(self.predicate(r, c, nr, nc)) for c in range(nc)) for r in range(nr))

    def mask(self, outputs, inputs):
        return torch.tensor(self.support(outputs, inputs), dtype=torch.bool).reshape(
            -(-outputs // self.tile), -(-inputs // self.tile))


def bitmap(mask, tile):
    """Capture a CPU mask at preparation; its partial edge tiles are retained."""
    nr, nc = mask.shape
    held = tuple(tuple(row) for row in mask.to('cpu', torch.bool).tolist())
    def predicate(r, c, rows, columns):
        if (rows, columns) != (nr, nc):
            raise ValueError('the sparse bitmap must describe the declared weight tile grid')
        return held[r][c]
    return Closure(predicate, tile, name=f'bitmap {nr}x{nc}')


def named(spec, tile):
    """Arithmetic support: all, blockdiag:G or band:W, evaluated at preparation."""
    kind, _, value = spec.partition(':')
    if kind == 'all':
        predicate = lambda r, c, nr, nc: True
    elif kind == 'blockdiag':
        groups = int(value or 1)
        predicate = lambda r, c, nr, nc: r * groups // nr == c * groups // nc
    elif kind == 'band':
        width = int(value or 0)
        predicate = lambda r, c, nr, nc: abs(r - c) <= width
    else:
        raise ValueError(f'unknown sparse support {spec}')
    return Closure(predicate, tile, name=spec)


@lru_cache(None)
def _schedule(m, k, n, support, tile, weight, geometry):
    """Resolve output domains and reduction intervals; all coordinates are AOT.

    Products mask columns versus inner intervals. Weight gradients mask the
    output domain instead. Equal reduction lists share the same immutable range.
    Empty ranges write zeros in the same product, without a separate clearing op.
    """
    rm, cn, inner, groups = geometry
    pieces, records, ranges = [], [], {}
    def reduction(intervals):
        merged = []
        for first, end in intervals:
            if merged and merged[-1][1] == first:
                merged[-1] = (merged[-1][0], end)
            elif first < end:
                merged.append((first, end))
        key = tuple(merged)
        if key not in ranges:
            first = len(pieces)
            for start, end in key:
                pieces.extend((at, min(inner, end - at)) for at in range(start, end, inner))
            ranges[key] = (first, len(pieces))
        return ranges[key]
    def axis(length, block, boundary):
        for start in range(0, length, boundary):
            for at in range(start, min(start + boundary, length), block):
                yield at, min(block, length - at, start + boundary - at)
    for row, height in axis(m, rm, tile if weight else max(1, m)):
        for col, width in axis(n, cn, tile):
            if weight:
                intervals = ((0, k),) if support[row // tile][col // tile] else ()
            else:
                intervals = tuple((i * tile, min((i + 1) * tile, k))
                                  for i, kept in enumerate(support[col // tile]) if kept)
            first, end = reduction(intervals)
            records.extend(((row, col, height, width), (first, end, 0, 0)))
    count = len(records) // 2
    for _ in range(-count % groups):
        records.extend(((0, 0, 0, 0), (0, 0, 0, 0)))
    return tuple(records), tuple(pieces), -(-count // groups)


def _product(a, b, support, tile, out, *, weight=False):
    from .kernel import declare
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError('sparse contraction operands must agree on inner dimension')
    n = b.shape[1]
    types = {torch.float16: 'half', torch.float32: 'float', torch.bfloat16: 'bfloat'}
    configuration = operands()
    geometry = tuple(configuration['MATRIX_' + key] for key in ('ROWS', 'COLUMNS', 'INNER')) + (
        configuration['MATRIX_ROW_GROUPS'] * configuration['MATRIX_COLUMN_GROUPS'],)
    records, pieces, groups = _schedule(m, k, n, support, tile, weight, geometry)
    # Schedule construction is preparation. Capture the completed values as
    # attributes; tracing torch.tensor itself would add per-call CPU copies.
    with disable_proxy_modes_tracing(), unset_fake_temporarily():
        tiles = torch.tensor(records, dtype=torch.int32).reshape(-1, 4)
        spans = torch.tensor(pieces, dtype=torch.int32).reshape(-1, 2)
    result = torch.empty((m, n), dtype=out, device=a.device)
    rm, cn, inner, _ = geometry
    full = (int(m % rm == 0 and (not weight or tile % rm == 0)), int(n % cn == 0 and tile % cn == 0),
            int(all(length == inner for _, length in pieces)))
    args = ', '.join(map(str, (types[a.dtype], types[b.dtype], types[out], *geometry, *full)))
    instance = f'template [[host_name("sparse_product_bound")]] [[kernel]] decltype(sparse_product<{args}>) sparse_product<{args}>;'
    buffers = [a, b, result, tiles, spans]
    layouts = api_layouts('matrix')
    requirements = {0: layouts['input'], 1: layouts['weight'], 2: layouts['output'],
                    3: tiles.stride(), 4: spans.stride()}
    result, = declare('sparse.metal', 'sparse_product_bound', buffers, (2,), (m, k, n), 5,
                      (groups, 1, 1), (geometry[-1] * configuration['SIMD_WIDTH'], 1, 1), instance,
                      indexed=(), requirements=requirements)
    return result


def _rows(x):
    return x.reshape(math.prod(x.shape[:-1]), x.shape[-1])


def linear(x, w, closure, out=torch.float16):
    """x @ masked(w).T, with output placement supplied by graph preparation."""
    n, k = w.shape
    result = _product(_rows(x), w.T, closure.support(n, k), closure.tile, out)
    return result.reshape(*x.shape[:-1], n)


def input_grad(dy, w, closure, out=torch.float16):
    n, k = w.shape
    support = closure.support(n, k)
    transposed = tuple(tuple(support[r][c] for r in range(len(support))) for c in range(-(-k // closure.tile)))
    result = _product(_rows(dy), w, transposed, closure.tile, out)
    return result.reshape(*dy.shape[:-1], k)


def weight_grad(x, dy, closure, n, k, out=torch.float32):
    return _product(_rows(dy).T, _rows(x), closure.support(n, k), closure.tile, out, weight=True)


class LinearFunction(torch.autograd.Function):
    """y = x . (w (.) M)^T with its exact backward, each product the closure's."""

    @staticmethod
    def forward(ctx, x, w, closure):
        ctx.save_for_backward(x, w)
        ctx.closure = closure
        return linear(x, w, closure, out=x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        n, k = w.shape
        return (input_grad(dy, w, ctx.closure, out=x.dtype) if ctx.needs_input_grad[0] else None,
                weight_grad(x, dy, ctx.closure, n, k, out=w.dtype) if ctx.needs_input_grad[1] else None, None)


def expand(mask, tile, shape):
    """A tile mask [N/tile, K/tile] as an element mask of `shape`."""
    return mask.to('cpu').repeat_interleave(tile, 0).repeat_interleave(tile, 1)[:shape[0], :shape[1]]


def contributions(w, x, tile):
    """Each tile's contribution to the product x . w^T on the rows x: the squared norm of its own term, [N/tile, K/tile]
    (float32, on w's device)."""
    n, k = w.shape
    out = torch.empty(-(-n // tile), -(-k // tile), device=w.device)
    for a in range(0, n, tile):
        for b in range(0, k, tile):
            out[a // tile, b // tile] = (x[:, b:b + tile].float() @ w[a:a + tile, b:b + tile].float().T).square().sum()
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
    energy = torch.stack([w[a:a + tile, b:b + tile].float().square().sum()
                          for a in range(0, n, tile) for b in range(0, k, tile)]).sort(descending=True).values
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
        self.closure = closure if isinstance(closure, Closure) else bitmap(closure, tile)
        if lowering not in ('dense', 'tiles'):
            raise ValueError('SparseLinear lowering must be dense or tiles')
        if lowering == 'dense':
            weight = weight * expand(self.closure.mask(n, k), self.closure.tile, weight.shape).to(weight.device)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = bias
        self.lowering = lowering

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight) if self.lowering == 'dense' else linear(x, self.weight, self.closure, out=x.dtype)
        return y if self.bias is None else y + self.bias
