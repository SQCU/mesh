from . import BlockSpec, ShapeDtypeStruct
from . import kernels


# design/algorithm-sources.md#streaming-ffn
def _rows(i):
    return (i, 0)


# design/algorithm-sources.md#streaming-ffn
def _weight(i):
    return (0, 0)


# design/algorithm-sources.md#streaming-ffn
def _pointwise(program, kernel, operands, tile_rows):
    shape = operands[0].shape
    block = (tile_rows, shape[1])
    return program.kernel_call(kernel, grid=((shape[0] + tile_rows - 1) // tile_rows,),
        in_specs=tuple(BlockSpec(block, _rows) for _ in operands),
        out_specs=BlockSpec(block, _rows),
        out_shape=ShapeDtypeStruct(shape, operands[0].dtype))(*operands)


# design/algorithm-sources.md#streaming-ffn
def _sum(program, values, tile_rows):
    values = tuple(values)
    if not values:
        raise ValueError('A linear reduction requires contributions')
    while len(values) > 1:
        values = tuple(_pointwise(program, kernels.add, values[i:i+2], tile_rows)
            if i+1 < len(values) else values[i] for i in range(0, len(values), 2))
    return values[0]


# design/algorithm-sources.md#streaming-ffn
def _linear(program, x, w, tile_rows):
    rows, k = x.shape
    return program.kernel_call(kernels.matmul,
        grid=((rows + tile_rows - 1) // tile_rows,),
        in_specs=(BlockSpec((tile_rows, k), _rows), BlockSpec(w.shape, _weight)),
        out_specs=BlockSpec((tile_rows, w.shape[1]), _rows),
        out_shape=ShapeDtypeStruct((rows, w.shape[1]), x.dtype))(x, w)


# design/algorithm-sources.md#streaming-ffn
def ffn(program, inputs, up_weights, down_weights, *, tile_rows, exchange=None):
    inputs, up_weights, down_weights = tuple(inputs), tuple(up_weights), tuple(down_weights)
    if not inputs or not up_weights or len(up_weights) != len(down_weights) or any(len(group) != len(inputs) for group in up_weights):
        raise ValueError('Weights must cover every input partition and hidden section')
    outputs = []
    for up, down in zip(up_weights, down_weights):
        terms = tuple(_linear(program, x, w, tile_rows) for x, w in zip(inputs, up))
        hidden = _sum(program, terms, tile_rows)
        activated = _pointwise(program, kernels.swish, (hidden,), tile_rows)
        operand = exchange(activated) if exchange is not None else activated
        outputs.append(_linear(program, operand, down, tile_rows))
    return _sum(program, outputs, tile_rows)
