"""Compositions of kernel_call (design/algorithm-sources.md#nnffn, #nnrmsnorm, #nnembedding; design/deliverables.md
section 5, program type C: a layer API over the library, never in it): each contraction cut into K panels whose
products are partials (Partial: a kernel that declares no combine refuses them until their sum completes), summed by
a tree of adds, every product and sum a grid of row and column tiles, so each tile is published as it finishes and its
consumers fire on it.  The numerics are the kernels' (kernels.py: torch's); placement is the caller's (`peer`,
`exchange`)."""
from math import gcd

import kernels
from mesh import BlockSpec, Partial, ShapeDtypeStruct
from mesh._native import dtype as _dtype


# design/algorithm-sources.md#nnffn
def _tile(size, *boundaries):
    for boundary in boundaries:
        size = gcd(size, boundary)
    return size


# design/algorithm-sources.md#nnffn
def _block(i, j):
    return (i, j)


# design/algorithm-sources.md#nnrmsnorm
def _rows(i):
    return (i, 0)


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
def _sum(program, terms, tile_rows, *, peer=None, output_dtype=None):
    terms = tuple(terms)
    if not terms:
        raise ValueError('A linear reduction requires contributions')
    if len(terms) == 1:
        if output_dtype is None or terms[0].dtype == _dtype(output_dtype):
            return terms[0]
        return _pointwise(program, kernels.identity, terms, tile_rows, peer=peer, output_dtype=output_dtype)
    while len(terms) > 1:
        last = len(terms) == 2
        terms = tuple(_pointwise(program, kernels.add, terms[i:i + 2], tile_rows, peer=peer,
                                 output_dtype=output_dtype if last else None)
                      if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2))
    return terms[0]


# design/algorithm-sources.md#kernelsdot
def _partials(tensors):
    contributions = tuple(object() for _ in tensors)
    if len(tensors) > 1:
        required = frozenset(contributions)
        for tensor, contribution in zip(tensors, contributions):
            for ref in tensor.blocks.values():
                ref.partial = Partial(required, frozenset((contribution,)))
    return tensors


# design/algorithm-sources.md#nnffn
def linear(program, x, w, *, tile_rows, tile_k=128, tile_columns=128, peer=None, output_dtype=None):
    rows, inner = x.shape
    if inner != w.shape[0]:
        raise ValueError('Contraction dimensions differ')
    dtype = x.dtype if output_dtype is None else output_dtype
    mr = _tile(min(tile_rows, rows), x.block_shape[0] if x.grid[0] > 1 else 0)
    kr = _tile(min(tile_k, inner), x.block_shape[1] if x.grid[1] > 1 else 0, w.block_shape[0] if w.grid[0] > 1 else 0)
    nr = _tile(min(tile_columns, w.shape[1]), w.block_shape[1] if w.grid[1] > 1 else 0)
    grid = ((rows + mr - 1) // mr, (w.shape[1] + nr - 1) // nr)
    panels = (inner + kr - 1) // kr
    parts = _partials(tuple(program.kernel_call(kernels.matmul, grid=grid,
        in_specs=(BlockSpec((mr, kr), lambda i, j, panel=panel: (i, panel)),
                  BlockSpec((kr, nr), lambda i, j, panel=panel: (panel, j))),
        out_specs=BlockSpec((mr, nr), _block),
        out_shape=ShapeDtypeStruct((rows, w.shape[1]), dtype if panels == 1 else 'float32'), peer=peer)(x, w)
        for panel in range(panels)))
    return _sum(program, parts, mr, peer=peer, output_dtype=dtype)


# design/algorithm-sources.md#nnffn
def ffn(program, inputs, up_weights, down_weights, *, tile_rows, tile_k=128, tile_columns=128, exchange=None, peer=None):
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
        projected = _sum(program, (linear(program, x, w, tile_rows=tile_rows, tile_k=tile_k, tile_columns=tile_columns,
                                          output_dtype='float32', peer=peer) for x, w in zip(inputs, up)),
                         tile_rows, peer=peer)
        hidden = _pointwise(program, kernels.swish, (projected,), tile_rows, peer=peer, output_dtype=inputs[0].dtype)
        operand = exchange(hidden) if exchange is not None else hidden
        projections.append(linear(program, operand, down, tile_rows=tile_rows, tile_k=tile_k,
                                  tile_columns=tile_columns, output_dtype='float32', peer=peer))
    return _sum(program, projections, tile_rows, peer=peer, output_dtype=inputs[0].dtype)


# design/algorithm-sources.md#nnrmsnorm
def _row_reduce(program, kernel, x, *, tile_rows, peer=None, output_dtype=None):
    rows = x.shape[0]
    mr = _tile(min(tile_rows, rows), x.block_shape[0] if x.grid[0] > 1 else 0)
    nr = x.block_shape[1]
    parts = _partials(tuple(program.kernel_call(kernel,
        grid=((rows + mr - 1) // mr,),
        in_specs=(BlockSpec((mr, nr), lambda i, panel=panel: (i, panel)),),
        out_specs=BlockSpec((mr, 1), _rows),
        out_shape=ShapeDtypeStruct((rows, 1), x.dtype if output_dtype is None else output_dtype), peer=peer)(x)
        for panel in range(x.grid[1])))
    return _sum(program, parts, mr, peer=peer)


# design/algorithm-sources.md#nnrmsnorm
def rmsnorm(program, x, gamma, *, tile_rows, epsilon=1e-6, peer=None):
    value, statistic, weight = kernels.arguments(3)
    total = _row_reduce(program, kernels.expression((value * value).sum()), x, tile_rows=tile_rows,
                        output_dtype='float32', peer=peer)
    normalize = kernels.expression(value * (statistic / x.shape[1] + epsilon).rsqrt() * weight)
    return _pointwise(program, normalize, (x, total.broadcast_to(x.shape), gamma.broadcast_to(x.shape)), tile_rows,
                      peer=peer)


# design/algorithm-sources.md#nnembedding
def embedding(program, table, indices, *, tile_rows, tile_columns=128, peer=None):
    rows, width = indices.shape[0], table.shape[1]
    mr = _tile(min(tile_rows, rows), indices.block_shape[0] if indices.grid[0] > 1 else 0)
    nr = _tile(min(tile_columns, width), table.block_shape[1] if table.grid[1] > 1 else 0)
    return program.kernel_call(kernels.gather,
        grid=((rows + mr - 1) // mr, (width + nr - 1) // nr),
        in_specs=(BlockSpec((table.shape[0], nr), lambda i, j: (0, j)),
                  BlockSpec((mr, 1), lambda i, j: (i, 0))),
        out_specs=BlockSpec((mr, nr), _block),
        out_shape=ShapeDtypeStruct((rows, width), table.dtype), peer=peer)(table, indices)


# design/algorithm-sources.md#nnembedding
def summed_embedding(program, x, tables, indices, *, tile_rows, peer=None):
    tables, indices = tuple(tables), tuple(indices)
    if len(tables) != len(indices):
        raise ValueError('Each embedding table requires its index tensor')
    values = tuple(embedding(program, table, index, tile_rows=tile_rows, tile_columns=x.block_shape[1], peer=peer)
                   for table, index in zip(tables, indices))
    return _sum(program, (x, *values), tile_rows, peer=peer)
