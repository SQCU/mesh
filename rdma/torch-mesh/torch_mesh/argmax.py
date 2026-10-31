"""Selection declarations lowered to the common prepared native commands in argmax.metal
(metal-microbench docs/kernels.md#speculative-decoding, sampling). The noise is argmax.metal's mesh_gumbel
alone; nothing here computes it.

  draw(z, seed, keys, first=0, min_p=0)        each row's token at temperature 1 (mesh_draw): over its candidates c the
                                               greatest z(c) + g(seed, key, first + c), its id first + c; with min_p > 0
                                               a candidate below the row's greatest z plus log min_p is never taken
  retain(z, seed, positions, head, tail)       each row's distribution as a retained set (mesh_retain): ids
                                               [..., head + tail], their z, and the row's log Z

z is [..., V], rows of logits (on the GPU in float32: another device's or dtype's copied there); a row's key is the
position it predicts from, one for every row or one a row, so a row computed again draws the same token.

The retained set (the operator, 2026-10-14: "you have to choose a power of 2 and then accept the compromises that come
from the power of 2 thus chosen"): a head, the row's 2^k greatest z, and a tail, 2^m of the rest sampled without
replacement in proportion to their probabilities (the greatest z + g keyed as the draw keys the row: the Gumbel-top-k,
Kool, van Hoof & Welling 2019), each in ascending id order, so a record names no greatest entry, and the token
draw(z, seed, positions) takes is in it.  A divergence over a record takes what it excludes as one bucket, its mass
1 - sum exp(z - log Z) over the ids (the other row's at the same ids, over its own log Z): the coarse-grained KL is at
most the rows' (the log-sum inequality), and equal to it where nothing is excluded."""
import math
from pathlib import Path

import torch

from . import _stream
from .streamed import instance

SOURCE = Path(__file__).with_name('argmax.metal').read_text()
def _rows(z, keys):
    z2 = z.to('mps', torch.float32).reshape(-1, z.shape[-1])
    return z2, torch.as_tensor(keys, dtype=torch.int32, device=z2.device).reshape(-1).expand(z2.shape[0])


@torch.library.custom_op('torch_mesh::draw', mutates_args=())
def _draw(z: torch.Tensor, keys: torch.Tensor, cutoff: torch.Tensor, seed: int, first: int) -> torch.Tensor:
    raise RuntimeError('sampling must be prepared as part of a declarations.Function')


@_draw.register_fake
def _draw_shape(z, keys, cutoff, seed, first):
    return z.new_empty((z.shape[0],), dtype=torch.int64)


@torch.library.custom_op('torch_mesh::retain', mutates_args=())
def _retain(z: torch.Tensor, keys: torch.Tensor, seed: int, head: int, tail: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError('retained sampling must be prepared as part of a declarations.Function')


@_retain.register_fake
def _retain_shape(z, keys, seed, head, tail):
    return (z.new_empty((z.shape[0], head + tail), dtype=torch.int64),
            z.new_empty((z.shape[0], head + tail)), z.new_empty((z.shape[0],)))


def draw(z, seed, keys, first=0, min_p=0):
    """Each row's token, int64 [...] (module docstring)."""
    z2, keys = _rows(z, keys)
    rows, v = z2.shape
    cutoff = z2.amax(1) + math.log(min_p) if min_p > 0 else torch.full((rows,), -math.inf, device=z2.device)
    return _draw(z2, keys, cutoff, seed, first).reshape(z.shape[:-1])


def retain(z, seed, positions, head=_stream.compilation['KEPT_HEAD'], tail=_stream.compilation['KEPT_TAIL']):
    """Each row's retained set (module docstring): ids int64 [..., head + tail], their z float32 [..., head + tail], and
    log Z float32 [...]."""
    if not all(n > 0 and n & (n - 1) == 0 for n in (head, tail)) or head + tail >= z.shape[-1]:
        raise ValueError(f'retain: head {head} and tail {tail} are each a power of two, together fewer than a row\'s {z.shape[-1]}')
    z2, positions = _rows(z, positions)
    ids, kept, log_z = _retain(z2, positions, seed, head, tail)
    return ids.reshape(*z.shape[:-1], -1), kept.reshape(*z.shape[:-1], -1), log_z.reshape(z.shape[:-1])


def declaration(kind, operands, parameters, configuration):
    """Retain the existing sampling kernels as named preparation commands."""
    import struct
    from .command_declarations import Commands
    commands = Commands(operands, **configuration)
    geometry = commands.geometry
    rows, width = operands['z']['shape']
    if not rows:
        return commands.declaration()
    maps, complete = parameters.get('write_maps'), parameters.get('write_complete')
    definitions, indices = {}, {}
    threads = geometry['DRAW_THREADS']
    shift = geometry['ACCESS_WRITE_SHIFT']
    if kind == 'draw':
        blocks = max(1, min(geometry['DRAW_MAX_BLOCKS'], width // geometry['DRAW_COLUMNS_PER_BLOCK']))
        parts = commands.temporary('draw/parts', (2 * rows * blocks,), torch.float32, requirements=('contiguous',))
        arrivals = commands.temporary('draw/arrivals', (rows,), torch.int32, initializer=0, requirements=('contiguous',))
        if maps is not None:
            indices[8] = maps['ids']
            definitions = {'MESH_DRAW_RESULT(i)': 'mesh_index<8>(i)',
                           'MESH_DRAW_RESULTS': math.prod(operands['ids']['shape']),
                           'MESH_DRAW_SELECTED': int(not complete['ids'])}
        function, instantiated = instance('mesh_draw_rows', threads)
        commands.shader(SOURCE + '\n', function,
            [commands.view(role) for role in ('z', 'keys', 'cutoff', 'ids')] + [parts, arrivals],
            constants={6: struct.pack('4I', width, parameters['seed'] & 0xffffffff,
                                     parameters['first'] & 0xffffffff, blocks)},
            definitions=definitions, indices=indices, instance=instantiated + '\n',
            grid=(blocks, rows, 1), threadgroup=(threads, 1, 1), access=7 | (56 << shift))
    else:
        if maps is not None:
            for slot, role in enumerate(('ids', 'kept', 'log_z'), 8):
                indices[slot] = maps[role]
                definitions[f'MESH_RETAIN_{role.upper()}(i)'] = f'mesh_index<{slot}>(i)'
                definitions[f'MESH_RETAIN_{role.upper()}_COUNT'] = math.prod(operands[role]['shape'])
                definitions[f'MESH_RETAIN_{role.upper()}_SELECTED'] = int(not complete[role])
        function, instantiated = instance('mesh_retain_rows', threads, parameters['head'], parameters['tail'])
        commands.shader(SOURCE + '\n', function, [commands.view(role) for role in ('z', 'keys', 'ids', 'kept', 'log_z')],
            constants={5: struct.pack('2I', width, parameters['seed'] & 0xffffffff)},
            definitions=definitions, indices=indices, instance=instantiated + '\n',
            grid=(1, rows, 1), threadgroup=(threads, 1, 1), access=3 | (28 << shift))
    return commands.declaration()


def prepare_draw(z, keys, cutoff, seed, first, *, out=None, write_maps=None, write_complete=None):
    from .placement import output
    from .operand import TensorView
    from .command_declarations import descriptions, configuration, bind
    from .native import command_records
    ids = TensorView.of(out) if write_maps is not None else output((z.shape[0],), torch.int64, z.device, out)
    values = dict(z=z, keys=keys, cutoff=cutoff, ids=ids)
    prepared = declaration('draw', descriptions(values),
        dict(seed=seed, first=first, write_maps=write_maps, write_complete=write_complete), configuration())
    return ids, list(command_records(bind(prepared, values)))


def prepare_retain(z, keys, seed, head, tail, *, outputs=None, write_maps=None, write_complete=None):
    from .placement import output
    from .operand import TensorView
    from .command_declarations import descriptions, configuration, bind
    from .native import command_records
    rows = z.shape[0]
    ids, kept, log_z = ([TensorView.of(value) for value in outputs] if write_maps is not None else
                       [output(shape, dtype, z.device, out) for shape, dtype, out in
                       zip(((rows, head + tail), (rows, head + tail), (rows,)),
                           (torch.int64, torch.float32, torch.float32),
                           outputs if outputs is not None else (None, None, None), strict=True)])
    values = dict(z=z, keys=keys, ids=ids, kept=kept, log_z=log_z)
    prepared = declaration('retain', descriptions(values),
        dict(seed=seed, head=head, tail=tail, write_maps=write_maps, write_complete=write_complete), configuration())
    return (ids, kept, log_z), list(command_records(bind(prepared, values)))
