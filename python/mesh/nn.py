from math import gcd

from . import BlockSpec, ShapeDtypeStruct
from . import kernels
from .collective import reduce_scatter, all_gather


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
def linear(program, x, w, *, tile_rows, tile_k=128, tile_columns=128, peer=None, output_dtype=None):
    rows, inner = x.shape
    if inner != w.shape[0]:
        raise ValueError('Contraction dimensions differ')
    mr = _tile(min(tile_rows, rows), x.block_shape[0] if x.grid[0] > 1 else 0)
    nr = _tile(min(tile_columns, w.shape[1]), w.block_shape[1] if w.grid[1] > 1 else 0)
    left, right = kernels.arguments(2)
    return program.kernel_call(kernels.expression(kernels.dot(left, right, tile_k=tile_k)),
        grid=((rows + mr - 1) // mr, (w.shape[1] + nr - 1) // nr),
        in_specs=(BlockSpec(None), BlockSpec(None)),
        out_specs=BlockSpec((mr, nr), _block),
        out_shape=ShapeDtypeStruct((rows, w.shape[1]), x.dtype if output_dtype is None else output_dtype),
        peer=peer)(x, w)



# design/algorithm-sources.md#typed-ffn-expression-composition
def _expression_sum(terms):
    terms = tuple(terms)
    while len(terms) > 1:
        terms = tuple(terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
                      for i in range(0, len(terms), 2))
    return terms[0]


# design/algorithm-sources.md#typed-ffn-expression-composition
def ffn(program, inputs, up_weights, down_weights, *, tile_rows, tile_k=128,
        tile_columns=128, peers, owners):
    peers = tuple(peers)
    inputs, up_weights, down_weights = tuple(inputs), tuple(map(tuple, up_weights)), tuple(down_weights)
    if not inputs or not up_weights or len(up_weights) != len(down_weights) or any(len(group) != len(inputs) for group in up_weights):
        raise ValueError('Weights must cover every input partition and hidden section')
    rows, columns = inputs[0].shape[0], down_weights[0].shape[1]
    operands = (*inputs, *(w for group in up_weights for w in group), *down_weights)
    arguments = kernels.arguments(len(operands))
    row_tiles = tuple(_tile(min(tile_rows, rows), x.block_shape[0] if x.grid[0] > 1 else 0) for x in inputs)
    hidden_rows = _tile(min(tile_rows, rows), *(tile if tile < rows else 0 for tile in row_tiles))
    hidden, layouts = [], []
    for index, (up, down) in enumerate(zip(up_weights, down_weights)):
        width = up[0].shape[1]
        if any(x.shape[0] != rows or w.shape[1] != width or x.shape[1] != w.shape[0]
               for x, w in zip(inputs, up)) or down.shape != (width, columns):
            raise ValueError('FFN contractions must share their output domains')
        column_tiles = tuple(_tile(min(tile_columns, width), w.block_shape[1] if w.grid[1] > 1 else 0) for w in up)
        block = (hidden_rows, _tile(min(column_tiles), *(tile if tile < width else 0 for tile in column_tiles)))
        weights = arguments[len(inputs) * (index + 1):len(inputs) * (index + 2)]
        projected = _expression_sum(kernels.dot(x, w, tile_k=tile_k) for x, w in zip(arguments, weights))
        hidden.append((projected / (1 + (0 - projected).exp())).astype(inputs[0].dtype))
        layouts.append(((rows, width), block))
    projections, output_tiles = [], []
    for value, (shape, block), weight, down in zip(hidden, layouts, arguments[-len(down_weights):], down_weights):
        projections.append(kernels.dot(value, weight,
            tile_k=_tile(min(tile_k, shape[1]), block[1] if block[1] < shape[1] else 0)))
        output_tiles.append((_tile(min(tile_rows, rows), block[0] if block[0] < rows else 0),
                             _tile(min(tile_columns, columns), down.block_shape[1] if down.grid[1] > 1 else 0)))
    block = (_tile(min(tile_rows, rows), *(tile[0] if tile[0] < rows else 0 for tile in output_tiles)),
             _tile(min(tile[1] for tile in output_tiles), *(tile[1] if tile[1] < columns else 0 for tile in output_tiles)))
    partials = program.kernel_call(kernels.expression(_expression_sum(projections).astype(inputs[0].dtype)),
        grid=tuple((size + tile - 1) // tile for size, tile in zip((rows, columns), block)),
        in_specs=(BlockSpec(None),) * len(operands),
        out_specs=BlockSpec(block, _block),
        out_shape=ShapeDtypeStruct((rows, columns), inputs[0].dtype))(*operands)

    return all_gather(program, reduce_scatter(program, partials, peers=peers, owners=owners),
                      peers=peers, owners=owners)


# design/algorithm-sources.md#rmsnorm-shared-expression-composition
def rmsnorm(program, x, gamma, *, tile_rows, epsilon=1e-6):
    gamma = gamma.broadcast_to(x.shape)
    value, weight = kernels.arguments(2)
    normalize = kernels.expression(value * ((value * value).sum() / x.shape[1] + epsilon).rsqrt() * weight)
    block = (_tile(min(tile_rows, x.shape[0]),
                   *(operand.block_shape[0] if operand.grid[0] > 1 else 0 for operand in (x, gamma))),
             _tile(min(x.shape[1], x.block_shape[1], gamma.block_shape[1]),
                   *(operand.block_shape[1] if operand.grid[1] > 1 else 0 for operand in (x, gamma))))
    return program.kernel_call(normalize,
        grid=tuple((size + tile - 1) // tile for size, tile in zip(x.shape, block)),
        in_specs=(BlockSpec(None), BlockSpec(None)),
        out_specs=BlockSpec(block, _block),
        out_shape=ShapeDtypeStruct(x.shape, x.dtype))(x, gamma)


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
