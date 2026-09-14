from math import gcd

from . import BlockSpec, ShapeDtypeStruct
from . import kernels
from .collective import reduce_scatter, all_gather


# design/algorithm-sources.md#nnffn
def _tile(size, *boundaries):
    for boundary in boundaries:
        size = gcd(size, boundary)
    return size


# design/algorithm-sources.md#nnffn
def _block(i, j):
    return (i, j)


# design/algorithm-sources.md#nnffn
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




# design/algorithm-sources.md#nnffn
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



# design/algorithm-sources.md#nnffn
def _sum(program, terms, tile_rows):
    terms = tuple(terms)
    while len(terms) > 1:
        terms = tuple(_pointwise(program, kernels.add, terms[i:i + 2], tile_rows)
                      if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2))
    return terms[0]


# design/algorithm-sources.md#nnffn
def ffn(program, inputs, up_weights, down_weights, *, tile_rows, tile_k=128,
        tile_columns=128, peers, owners):
    peers = tuple(peers)
    inputs, up_weights, down_weights = tuple(inputs), tuple(map(tuple, up_weights)), tuple(down_weights)
    if not inputs or not up_weights or len(up_weights) != len(down_weights) or any(len(group) != len(inputs) for group in up_weights):
        raise ValueError('Weights must cover every input partition and hidden section')
    rows, columns = inputs[0].shape[0], down_weights[0].shape[1]
    projections = []
    for up, down in zip(up_weights, down_weights):
        width = up[0].shape[1]
        if any(x.shape[0] != rows or w.shape[1] != width or x.shape[1] != w.shape[0]
               for x, w in zip(inputs, up)) or down.shape != (width, columns):
            raise ValueError('FFN contractions must share their output domains')
        projected = _sum(program, (linear(program, x, w, tile_rows=tile_rows,
            tile_k=tile_k, tile_columns=tile_columns, output_dtype='float32')
            for x, w in zip(inputs, up)), tile_rows)
        hidden = _pointwise(program, kernels.swish, (projected,), tile_rows, output_dtype=inputs[0].dtype)
        projections.append(linear(program, hidden, down, tile_rows=tile_rows,
            tile_k=tile_k, tile_columns=tile_columns, output_dtype='float32'))
    partials = _sum(program, projections, tile_rows)
    value, = kernels.arguments(1)
    partials = _pointwise(program, kernels.expression(value.astype(inputs[0].dtype)),
                         (partials,), tile_rows, output_dtype=inputs[0].dtype)
    return all_gather(program, reduce_scatter(program, partials, peers=peers, owners=owners),
                      peers=peers, owners=owners)


# design/algorithm-sources.md#nnrmsnorm
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


# design/algorithm-sources.md#nnembedding
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
