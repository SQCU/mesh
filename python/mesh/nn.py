from math import gcd

import numpy as np

from . import BlockSpec, ShapeDtypeStruct
from . import kernels


# design/algorithm-sources.md#streaming-ffn
def _rows(i):
    return (i, 0)


# design/algorithm-sources.md#pallas-panel-composition
def _tile(size, *boundaries):
    for boundary in boundaries:
        size = gcd(size, boundary)
    return size


# design/algorithm-sources.md#pallas-panel-composition
def _block(i, j):
    return (i, j)


# design/algorithm-sources.md#pallas-panel-composition
def _pointwise(program, kernel, operands, tile_rows, *, peer=None, output_dtype=None):
    shape = operands[0].shape
    block = (_tile(min(tile_rows, shape[0]), *(value.block_shape[0] if value.grid[0] > 1 else 0 for value in operands)),
             _tile(min(value.block_shape[1] for value in operands),
                   *(value.block_shape[1] if value.grid[1] > 1 else 0 for value in operands)))
    return program.kernel_call(kernel,
        grid=tuple((size + tile - 1) // tile for size, tile in zip(shape, block)),
        in_specs=tuple(BlockSpec(block, _block) for _ in operands),
        out_specs=BlockSpec(block, _block),
        out_shape=ShapeDtypeStruct(shape, operands[0].dtype if output_dtype is None else output_dtype), peer=peer)(*operands)


# design/algorithm-sources.md#pallas-panel-composition
def _sum(program, values, tile_rows, *, peer=None):
    values = tuple(values)
    if not values:
        raise ValueError('A linear reduction requires contributions')
    while len(values) > 1:
        values = tuple(_pointwise(program, kernels.add, values[i:i+2], tile_rows, peer=peer)
            if i+1 < len(values) else values[i] for i in range(0, len(values), 2))
    return values[0]


# design/algorithm-sources.md#contraction-accumulation
def _cast(program, x, dtype, tile_rows, *, peer=None):
    return x if x.dtype == np.dtype(dtype) else _pointwise(
        program, kernels.affine(), (x,), tile_rows, peer=peer, output_dtype=dtype)


# design/algorithm-sources.md#pallas-panel-composition
def linear(program, x, w, *, tile_rows, tile_k=128, tile_columns=128, peer=None, output_dtype=None):
    rows, inner = x.shape
    if inner != w.shape[0]:
        raise ValueError('Contraction dimensions differ')
    mr = _tile(min(tile_rows, rows), x.block_shape[0] if x.grid[0] > 1 else 0)
    kr = _tile(min(tile_k, inner), x.block_shape[1] if x.grid[1] > 1 else 0,
               w.block_shape[0] if w.grid[0] > 1 else 0)
    nr = _tile(min(tile_columns, w.shape[1]), w.block_shape[1] if w.grid[1] > 1 else 0)
    grid = ((rows + mr - 1) // mr, (w.shape[1] + nr - 1) // nr)
    parts = []
    for panel in range((inner + kr - 1) // kr):
        parts.append(program.kernel_call(kernels.matmul, grid=grid,
            in_specs=(BlockSpec((mr, kr), lambda i, j, panel=panel: (i, panel)),
                      BlockSpec((kr, nr), lambda i, j, panel=panel: (panel, j))),
            out_specs=BlockSpec((mr, nr), _block),
            out_shape=ShapeDtypeStruct((rows, w.shape[1]), "float32"), peer=peer)(x, w))
    return _cast(program, _sum(program, parts, mr, peer=peer),
                 x.dtype if output_dtype is None else output_dtype, mr, peer=peer)


# design/algorithm-sources.md#pallas-panel-composition
def ffn(program, inputs, up_weights, down_weights, *, tile_rows, tile_k=128,
        tile_columns=128, exchange=None, peer=None):
    inputs, up_weights, down_weights = tuple(inputs), tuple(up_weights), tuple(down_weights)
    if not inputs or not up_weights or len(up_weights) != len(down_weights) or any(len(group) != len(inputs) for group in up_weights):
        raise ValueError('Weights must cover every input partition and hidden section')
    outputs = []
    for up, down in zip(up_weights, down_weights):
        terms = tuple(linear(program, x, w, tile_rows=tile_rows, tile_k=tile_k,
                             tile_columns=tile_columns, peer=peer, output_dtype="float32") for x, w in zip(inputs, up))
        hidden = _sum(program, terms, tile_rows, peer=peer)
        activated = _pointwise(program, kernels.swish, (hidden,), tile_rows, peer=peer, output_dtype=inputs[0].dtype)
        operand = exchange(activated) if exchange is not None else activated
        outputs.append(linear(program, operand, down, tile_rows=tile_rows, tile_k=tile_k,
                              tile_columns=tile_columns, peer=peer, output_dtype="float32"))
    return _cast(program, _sum(program, outputs, tile_rows, peer=peer), inputs[0].dtype, tile_rows, peer=peer)


# design/algorithm-sources.md#pallas-panel-composition
def _row_reduce(program, kernel, x, *, tile_rows, peer=None, output_dtype=None):
    rows = x.shape[0]
    mr = _tile(min(tile_rows, rows), x.block_shape[0] if x.grid[0] > 1 else 0)
    nr = x.block_shape[1]
    parts = tuple(program.kernel_call(kernel,
        grid=((rows + mr - 1) // mr,),
        in_specs=(BlockSpec((mr, nr), lambda i, panel=panel: (i, panel)),),
        out_specs=BlockSpec((mr, 1), _rows),
        out_shape=ShapeDtypeStruct((rows, 1), x.dtype if output_dtype is None else output_dtype), peer=peer)(x)
        for panel in range(x.grid[1]))
    return _sum(program, parts, mr, peer=peer)


# design/algorithm-sources.md#pallas-panel-composition
def rmsnorm(program, x, gamma, *, tile_rows, epsilon=1e-6):
    value, statistic, weight = kernels.arguments(3)
    total = _row_reduce(program, kernels.expression((value * value).sum()), x,
                        tile_rows=tile_rows, output_dtype="float32")
    normalize = kernels.expression(value * (statistic / x.shape[1] + epsilon).rsqrt() * weight)
    return _pointwise(program, normalize,
        (x, total.broadcast_to(x.shape), gamma.broadcast_to(x.shape)), tile_rows)


# design/algorithm-sources.md#streamed-normalization-and-embedding
def embedding(program, table, indices, *, tile_rows, tile_columns=128):
    rows, width = indices.shape[0], table.shape[1]
    mr = _tile(min(tile_rows, rows), indices.block_shape[0] if indices.grid[0] > 1 else 0)
    nr = _tile(min(tile_columns, width), table.block_shape[1] if table.grid[1] > 1 else 0)
    table_value, selected = kernels.arguments(2)
    _, column = kernels.indices()
    selected = kernels.select(selected < 0, selected + table.shape[0], selected)
    return program.kernel_call(kernels.expression(table_value.at(selected, kernels.program_id(1) * nr + column)),
        grid=((rows + mr - 1) // mr, (width + nr - 1) // nr),
        in_specs=(BlockSpec(None),
                  BlockSpec((mr, 1), lambda i, j: (i, 0))),
        out_specs=BlockSpec((mr, nr), _block),
        out_shape=ShapeDtypeStruct((rows, width), table.dtype))(table, indices)


# design/algorithm-sources.md#streamed-normalization-and-embedding
def summed_embedding(program, x, tables, indices, *, tile_rows):
    tables, indices = tuple(tables), tuple(indices)
    if len(tables) != len(indices):
        raise ValueError('Each embedding table requires its index tensor')
    values = tuple(embedding(program, table, index, tile_rows=tile_rows, tile_columns=x.block_shape[1])
                   for table, index in zip(tables, indices))
    return _sum(program, (x, *values), tile_rows)
