"""Structured random orthogonal transforms on Apple GPUs (structured.metal): a seeded rotation of a matrix's rows or
columns in O(n log n) a row, exactly orthogonal, from fast Walsh-Hadamard transforms (the randomized Hadamard
transform of structured random projections [Ailon & Chazelle 2006; Le, Sarlos & Smola 2013, Fastfood]).

A width n = B * N, N its largest power-of-two factor within the declared block width: a round is
diag(signs) then H_N / sqrt N on each of the B blocks (split into local factors when needed) then a random orthogonal B x B mixing the
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
    return torch.empty_like(x, dtype=torch.float32)


def _context(ctx, inputs, output):
    x, signs, ctx.width = inputs
    ctx.dtype = x.dtype
    ctx.save_for_backward(signs)


def _backward(ctx, grad):
    signs, = ctx.saved_tensors
    return (expand(grad, signs, ctx.width, signed=False) * signs).to(ctx.dtype), None, None


fwht.register_autograd(_backward, setup_context=_context)


@torch.library.custom_op('torch_mesh::fwht_stage', mutates_args=())
def stage(x: torch.Tensor, signs: torch.Tensor, width: int, step: int, normalization: int, signed: bool, domain: list[int]) -> torch.Tensor:
    raise RuntimeError('butterfly stages require whole-function preparation')


@stage.register_fake
def _stage_shape(x, signs, width, step, normalization, signed, domain):
    return torch.empty((x.shape[0], signs.shape[0]), dtype=torch.float32, device=x.device)


def _stage_context(ctx, inputs, output):
    x, signs, ctx.width, ctx.step, ctx.normalization, ctx.signed, ctx.domain = inputs
    ctx.shape, ctx.dtype = x.shape, x.dtype
    ctx.save_for_backward(signs)


def _stage_backward(ctx, grad):
    signs, = ctx.saved_tensors
    result = stage(grad, signs, ctx.width, ctx.step, ctx.normalization, False, [])
    if ctx.signed:
        result = result * signs
    if ctx.domain:
        kept = [i for i, column in enumerate(ctx.domain) if column >= 0]
        index = torch.tensor([ctx.domain[i] for i in kept], device=result.device)
        result = torch.zeros(ctx.shape, dtype=result.dtype, device=result.device).index_add(1, index, result[:, kept])
    return result.to(ctx.dtype), None, None, None, None, None, None


stage.register_autograd(_stage_backward, setup_context=_stage_context)


def plan(width):
    # The mathematical width is supplied by the graph. Device memory only
    # selects how many butterfly levels fit in each local factor.
    _, _, memory = _stream.device_facts()
    return _stream.butterfly_plan(width, min(operands()['FWHT_BLOCK_WIDTH'], memory // torch.float32.itemsize))


def _validate(x, signs, width):
    if x.ndim != 2 or signs.shape != x.shape[-1:] or x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or signs.dtype not in (torch.int8, torch.float32):
        raise ValueError('fwht binds floating point rows and one int8 or fp32 sign per column')
    if width <= 0 or width & (width - 1) or x.shape[-1] % width:
        raise ValueError('fwht blocks must be power-of-two factors of the row width')


def expand(x, signs, width, *, domain=(), normalization=None, signed=True):
    """Expose local factors before graph partition and collective placement."""
    if not domain:
        _validate(x, signs, width)
    for local, step in plan(width):
        x = stage(x, signs, local, step, (width if normalization is None else normalization) if local * step == width else 1, signed and step == 1, list(domain) if step == 1 else [])
    return x


def prepare(x, signs, width, *, out=None):
    _validate(x, signs, width)
    commands = []
    for local, step in plan(width):
        last = local * step == width
        x, made = prepare_stage(x, signs, local, step, width if last else 1, step == 1, [], out=out if last else None)
        commands.extend(made)
    return x, commands


def stage_views(value, width, step):
    """The coordinates changed by one factor, with its local axis last."""
    from .operand import TensorView
    value = TensorView.of(value)
    rows, columns = value.shape
    return value.reshape((rows, columns // (width * step), width, step)).permute((0, 1, 3, 2))


def prepare_stage(x, signs, width, step, normalization, signed, domain, *, out=None):
    from .placement import output
    from .operand import TensorView
    from .layout import index_source
    import math
    import struct
    y = output((x.shape[0], signs.shape[0]), torch.float32, x.device, out)
    target = stage_views(y, width, step)
    # Domain-addressed reads are bound per tile below. The target supplies
    # their logical coordinates even when the source has no columns.
    source = target if domain else stage_views(x, width, step)
    sign = stage_views(TensorView.of(signs).reshape((1, signs.shape[0])), width, step)
    kinds = {torch.float32: 'float', torch.float16: 'half', torch.bfloat16: 'bfloat', torch.int8: 'char'}
    commands = []
    # Each record names only its pages. Completion of one piece can release
    # its consumer or a one-sided send without joining the rest of the row.
    for origin, size in _stream.view_tiles(source.shape, max(width, operands()['FWHT_TILE_ELEMENTS'])):
        def cut(view, broadcast=False):
            for axis, (first, count) in enumerate(zip(origin, size)):
                view = view.slice(axis, 0 if broadcast and axis == 0 else first, 1 if broadcast and axis == 0 else count)
            return view
        reads, writes, signs_view = cut(source), cut(target), cut(sign, True)
        mask = None
        if domain:
            positions = [domain[(origin[1] + outer) * width * step + lane * step + origin[2] + inner]
                         for outer in range(size[1]) for inner in range(size[2]) for lane in range(width)]
            mask = {4: [int(column >= 0) for column in positions]} if -1 in positions else None
            valid = next((column for column in positions if column >= 0), None)
            # Masked lanes name an already needed page for address emission;
            # wholly padded pieces have no input read or input dependency.
            reads = (TensorView.of(x).slice(0, origin[0], size[0]).select(1,
                     [column if column >= 0 else valid for column in positions]).reshape(size)) if valid is not None else None
        views = {1: writes, **({0: reads} if reads is not None else {}), **({2: signs_view} if signed else {})}
        handle = _kernel(width, normalization, signed, kinds[x.dtype], kinds[signs.dtype], bool(mask), index_source(views, mask))
        maximum, lanes, _ = _stream.pipeline_facts(handle)
        threads = min(maximum, max(lanes, width // 2))
        groups = math.prod(size) // width
        commands.append((handle, [reads, writes, signs_view if signed else None],
                         {3: struct.pack('I', math.prod(size[1:]) // width)},
                         (groups, 1, 1), (threads, 1, 1), False,
                         ((1 if reads is not None else 0) | (4 if signed else 0)) | (2 << _stream.compilation['ACCESS_WRITE_SHIFT'])))
    return y, commands


@lru_cache(None)
def _kernel(width, normalization, signed, input_type, sign_type, masked, indices):
    instance = f'structured_butterfly<{width},{normalization},{str(signed).lower()},{input_type},{sign_type},{str(masked).lower()}>'
    declaration = f'\ntemplate [[host_name("butterfly")]] [[kernel]] decltype({instance}) {instance};'
    return _stream.pipeline(_stream.metal_prelude() + indices + SOURCE + declaration, 'butterfly')


def _factor(n, block_width):
    if n <= 0 or block_width <= 0:
        raise ValueError('rotation width and block width must be positive')
    N = 1
    while n % (N * 2) == 0 and N * 2 <= block_width:
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
