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
from pathlib import Path
import math
import struct

import torch

from . import _stream
from .compilation import operands
from .command_declarations import Commands, dense, descriptions, configuration, bind

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
def stage(x: torch.Tensor, signs: torch.Tensor, width: int, step: int, normalization: int, signed: bool, sources: list[int], starts: list[int]) -> torch.Tensor:
    raise RuntimeError('butterfly stages require whole-function preparation')


@stage.register_fake
def _stage_shape(x, signs, width, step, normalization, signed, sources, starts):
    return torch.empty((x.shape[0], signs.shape[0]), dtype=torch.float32, device=x.device)


def _stage_context(ctx, inputs, output):
    x, signs, ctx.width, ctx.step, ctx.normalization, ctx.signed, ctx.sources, ctx.starts = inputs
    ctx.shape, ctx.dtype = x.shape, x.dtype
    ctx.save_for_backward(signs)


def _stage_backward(ctx, grad):
    signs, = ctx.saved_tensors
    result = stage(grad, signs, ctx.width, ctx.step, ctx.normalization, False, [], [])
    if ctx.signed:
        result = result * signs
    if ctx.starts:
        # each coordinate's gradient returns to its own sources: x[:, sources[k]] += result[:, i] for k in its group
        coordinates = [i for i in range(len(ctx.starts) - 1) for _ in range(ctx.starts[i + 1] - ctx.starts[i])]
        index = torch.tensor(ctx.sources, dtype=torch.long, device=result.device)
        result = torch.zeros(ctx.shape, dtype=result.dtype, device=result.device).index_add(1, index, result[:, coordinates])
    return result.to(ctx.dtype), None, None, None, None, None, None, None


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


def expand(x, signs, width, *, sources=(), starts=(), normalization=None, signed=True):
    """Expose local factors before graph partition and collective placement.

    Where `starts` is given, coordinate i of the first factor's input is the sum of x's columns
    sources[starts[i]:starts[i + 1]], each coordinate its own extent (an empty one is zero); otherwise x is
    already in coordinates."""
    if not starts:
        _validate(x, signs, width)
    for local, step in plan(width):
        x = stage(x, signs, local, step, (width if normalization is None else normalization) if local * step == width else 1, signed and step == 1,
                  list(sources) if step == 1 else [], list(starts) if step == 1 else [])
    return x


def prepare(x, signs, width, *, out=None):
    _validate(x, signs, width)
    commands = []
    for local, step in plan(width):
        last = local * step == width
        x, made = prepare_stage(x, signs, local, step, width if last else 1, step == 1, [], [], out=out if last else None)
        commands.extend(made)
    return x, commands


def _stage_layout(layout, width, step):
    """The coordinates changed by one factor, with its local axis last."""
    rows, columns = layout.shape
    return layout.reshape((rows, columns // (width * step), width, step)).permute((0, 1, 3, 2))


def factor_domain(algebra, source, signs, target, *, width, step, normalization, signed, sources, starts):
    """One existing factor's typed reads, including ordered source occurrences."""
    from .operand import layout_configuration
    from .selection import row_domain
    rows, columns = target['view'].shape
    groups = rows * columns // width
    def grouped(value):
        value = dict(value)
        value['view'] = _stage_layout(value['view'], width, step).reshape((groups, width))
        return value
    inputs = {} if starts else dict(x=grouped(source))
    if signed:
        repeated = _stream.TensorLayout((rows, columns), (0, 1))
        repeated = repeated.rebase(_stream.TensorLayout((columns,), (1,)), signs['view'])
        inputs['signs'] = grouped(dict(signs, view=repeated))
    parameters = dict(width=width, normalization=normalization, signed=signed,
                      sign_type={torch.int8: 'char', torch.float32: 'float'}[signs['dtype']])
    domain = row_domain(algebra, inputs, dict(out=grouped(target)), width=width,
                        adapter='fwht_stage', parameters=parameters)
    if starts:
        coordinates = [outer * width * step + lane * step + inner
                       for outer in range(columns // (width * step)) for inner in range(step) for lane in range(width)]
        present = [column for coordinate in coordinates for column in sources[starts[coordinate]:starts[coordinate + 1]]]
        occurrence_coordinates = [ordinal for ordinal, coordinate in enumerate(coordinates)
                                  for _ in range(starts[coordinate], starts[coordinate + 1])]
        # The declaration already supplies these finite occurrence lists. Only
        # rows repeat; their relation stays symbolic rather than expanding it.
        occurrences = source['view'].select(1, present).reshape((rows * len(present),))
        coordinate_view = _stream.TensorLayout((rows, columns), (columns, 1)).select(
            1, occurrence_coordinates).reshape((rows * len(present),))
        segments = algebra.layout(layout_configuration(coordinate_view)).then(algebra.map(
            (rows * columns,), (groups, width), lambda i, o: [f'{i[0]} = {o[0]} * {width} + {o[1]}']))
        logical = algebra.map(domain['work'], domain['work'], lambda i, o: [f'{o[0]} = {i[0]}']).then(
            segments.inverse(domain['work']))
        storage = algebra.layout(layout_configuration(occurrences))
        access = dict(source, relation=logical.then(storage))
        domain['reads']['x'] = access
        domain['uses']['x'] = dict(access=access, logical=logical, storage=storage, shape=occurrences.shape,
                                   side='read', segments=segments)
    return domain


def declaration(operands, parameters, configuration):
    """The factor's unchanged shader and source-occurrence maps, before pages."""
    commands = Commands(operands, **configuration)
    width, normalization, signed = (parameters[key] for key in ('width', 'normalization', 'signed'))
    source, target = commands.view('x'), commands.view('out')
    signs = commands.view('signs') if signed else None
    kinds = {torch.float32: 'float', torch.float16: 'half', torch.bfloat16: 'bfloat', torch.int8: 'char'}
    input_type = kinds[source['dtype']]
    sign_type = parameters.get('sign_type') or (kinds[signs['dtype']] if signed else 'float')
    maps = parameters.get('write_maps')
    if maps is not None:
        groups = maps['out'].count // width
        extents = (dense((groups * width + 1,)) if parameters.get('read_extents') is None else
                   parameters['read_extents']['x'])
        _piece(commands, source if source['layout'].count else None, target, signs, extents,
               width=width, normalization=normalization, signed=signed, groups=groups, sign_groups=groups,
               input_type=input_type, sign_type=sign_type, write_map=maps['out'],
               write_complete=parameters['write_complete']['out'])
    else:
        step, starts, sources = parameters['step'], parameters['starts'], parameters['sources']
        target = dict(target, layout=_stage_layout(target['layout'], width, step))
        # Ragged reads get their coordinates from the output even when no
        # source columns exist. Each piece declares only its own exact pages.
        grouped = target if starts else dict(source, layout=_stage_layout(source['layout'], width, step))
        if signed:
            signs = dict(signs, layout=_stage_layout(signs['layout'].reshape((1, signs['layout'].count)), width, step))
        for origin, size in _stream.view_tiles(grouped['layout'].shape, max(width, commands.geometry['FWHT_TILE_ELEMENTS'])):
            def cut(view, broadcast=False):
                if view is None:
                    return None
                layout = view['layout']
                for axis, (first, count) in enumerate(zip(origin, size)):
                    layout = layout.slice(axis, 0 if broadcast and axis == 0 else first,
                                          1 if broadcast and axis == 0 else count, 1)
                return dict(view, layout=layout)
            reads, writes, selected_signs = cut(grouped), cut(target), cut(signs, True)
            elements = math.prod(size[1:])
            extents = dense((elements + 1,))
            if starts:
                coordinates = [(origin[1] + outer) * width * step + lane * step + origin[2] + inner
                               for outer in range(size[1]) for inner in range(size[2]) for lane in range(width)]
                present = [column for c in coordinates for column in sources[starts[c]:starts[c + 1]]]
                first = [0]
                for c in coordinates:
                    first.append(first[-1] + starts[c + 1] - starts[c])
                extents = extents.select(0, first)
                reads = (dict(source, layout=source['layout'].slice(0, origin[0], size[0], 1).select(1, present))
                         if present else None)
            _piece(commands, reads, writes, selected_signs, extents, width=width, normalization=normalization,
                   signed=signed, groups=math.prod(size) // width, sign_groups=math.prod(size[1:]) // width,
                   input_type=input_type, sign_type=sign_type)
    return commands.declaration()


def _piece(commands, reads, writes, signs, extents, *, width, normalization, signed, groups, sign_groups,
           input_type, sign_type, write_map=None, write_complete=True):
    if not groups:
        return
    indices, definitions = {4: extents}, {}
    if write_map is not None:
        indices[8] = write_map
        definitions = {'MESH_FWHT_RESULT(i)': 'mesh_index<8>(i)', 'MESH_FWHT_RESULTS': writes['layout'].count,
                       'MESH_FWHT_SELECTED': int(not write_complete)}
    instance = f'structured_butterfly<{width},{normalization},{str(signed).lower()},{input_type},{sign_type}>'
    instantiated = f'\ntemplate [[host_name("butterfly")]] [[kernel]] decltype({instance}) {instance};'
    commands.shader(SOURCE, 'butterfly', [reads, writes, signs if signed else None],
        definitions=definitions, indices=indices, instance=instantiated, constants={3: struct.pack('I', sign_groups)},
        grid=(groups, 1, 1), threadgroup=None, threadgroup_policy={'preferred': width // 2, 'minimum_simd': True},
        limits={'threadgroup_memory_bytes': commands.capabilities['threadgroup_memory_bytes']},
        access=((1 if reads is not None else 0) | (4 if signed else 0)) | (2 << commands.geometry['ACCESS_WRITE_SHIFT']))


def prepare_stage(x, signs, width, step, normalization, signed, sources, starts, *, out=None):
    from .placement import output
    from .native import command_records
    y = output((x.shape[0], signs.shape[0]), torch.float32, x.device, out)
    values = dict(x=x, signs=signs, out=y)
    prepared = declaration(descriptions(values),
        dict(width=width, step=step, normalization=normalization, signed=signed, sources=sources, starts=starts),
        configuration())
    return y, list(command_records(bind(prepared, values)))


def prepare_factor(x, *, signs=None, out, width, normalization, signed, sign_type, write_maps, write_complete,
                   read_extents=None):
    """The same factor recipe over selected, already grouped page operands."""
    from .operand import TensorView
    from .native import command_records
    out = TensorView.of(out)
    values = dict(x=x, signs=signs, out=out)
    prepared = declaration(descriptions(values),
        dict(width=width, normalization=normalization, signed=signed, sign_type=sign_type,
             write_maps=write_maps, write_complete=write_complete, read_extents=read_extents), configuration())
    return out, list(command_records(bind(prepared, values)))


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
        self.n, self.N, self.B = n, *_factor(n, n if block_width is None else block_width)
        g = torch.Generator().manual_seed(int(seed))
        self.signs = [(torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device) for _ in range(rounds)]
        self.mixes = [torch.linalg.qr(torch.randn(self.B, self.B, generator=g, dtype=torch.float64))[0].float().to(device)
                      for _ in range(rounds)]

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
            x = expand(y, sign, self.N, signed=False)
            y = x * sign
        return y
