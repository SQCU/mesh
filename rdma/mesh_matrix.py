"""Matrix shader declarations from scalar, shape and capability operands.

The importer and FX frontend supply metadata to this module. Page placement,
shader compilation and command recording consume its result in their shared
lowerings; this module neither allocates numerical storage nor executes work.
"""
from math import prod


LIBRARIES = ('argmax.metal', 'matrix_operand.metal', 'matrix.metal')


def row_tiles(rows, geometry):
    """The product's compiled row groups, also used by tile publications."""
    width = geometry['MATRIX_ROWS'] * geometry['MATRIX_ROW_GROUPS']
    return tuple((first, min(width, rows - first)) for first in range(0, rows, width))


def product(x, w, out, geometry, *, x1=None, bias=None, act=False, alpha=1, beta=1,
            ready=None, arrivals=None):
    """Declare the shared product from ``{shape, type}`` scalar views.

    ``type`` is the caller's declared Metal scalar spelling. Shapes are the
    resolved logical matrix shapes; physical layout stays in the page operands.
    ``bindings`` names each occupied shader slot, preserving optional operands.
    """
    def shape(value):
        return tuple(value['shape'])

    xs, ws, ys = map(shape, (x, w, out))
    if len(xs) not in (2, 3) or len(ws) != len(xs) or len(ys) != len(xs):
        raise ValueError('matrix operands require two-dimensional or batched matrix views')
    rows, first, columns = xs[-2], xs[-1], ws[-1]
    if x1 is not None and (x1['type'] != x['type'] or len(shape(x1)) != len(xs) or shape(x1)[-2] != rows):
        raise ValueError('matrix input segments require the same dtype, rank and row count')
    inner = first + (shape(x1)[-1] if x1 is not None else 0)
    if ws[-2] != inner or ys[-2:] != (rows, columns):
        raise ValueError('matrix views disagree on product dimensions')
    batches = ys[0] if len(ys) == 3 else 1
    if len(xs) == 3 and (xs[0] not in (1, batches) or ws[0] not in (1, batches)):
        raise ValueError('matrix batch views must match or broadcast')
    if x1 is not None and len(shape(x1)) == 3 and shape(x1)[0] not in (1, batches):
        raise ValueError('matrix input segment batch views must match or broadcast')
    if bias is not None and (bias['type'] != out['type'] or len(shape(bias)) not in (1, 2) or
                            shape(bias)[-1] not in (1, columns) or
                            (len(shape(bias)) == 2 and shape(bias)[0] not in (1, rows))):
        raise ValueError('matrix bias requires the output dtype and a broadcastable row view')
    if ready is not None:
        tiles = len(row_tiles(rows, geometry)) * batches
        if (ready['type'] != 'long' or prod(shape(ready)) != tiles or arrivals is None or
                arrivals['type'] != 'int' or prod(shape(arrivals)) != tiles):
            raise ValueError('tile publication needs one ready word and arrival count per batch row tile')

    def step(binding, value, axis):
        return f'mesh_matrix_stride<{binding},{axis % len(shape(value))}>()' if value is not None else 0

    def affine(binding, value):
        return f'mesh_affine<{binding}>()' if value is not None else 0

    definitions = dict(INPUT=x['type'], WEIGHT=w['type'], OUTPUT=out['type'],
                       AM=step(0, x, -2), AK=step(0, x, -1), BK=step(1, w, -2), BN=step(1, w, -1),
                       AFFINE_A=affine(0, x), AFFINE_B=affine(1, w),
                       BATCH_A=int(len(xs) == 3 and xs[0] > 1), BATCH_B=int(len(ws) == 3 and ws[0] > 1),
                       BIAS_ROWS=shape(bias)[0] if bias is not None and len(shape(bias)) == 2 else 1,
                       BIAS_COLUMNS=shape(bias)[-1] if bias is not None else 1,
                       ACTIVATE=int(act), BIAS=int(bias is not None), ALPHA=repr(float(alpha))+'f',
                       BETA=repr(float(beta))+'f', PRODUCED=int(ready is not None),
                       SEGMENTED=int(x1 is not None), K0=first, PRODUCT_ROWS=rows,
                       PRODUCT_COLUMNS=columns, PRODUCT_INNER=inner,
                       A1M=step(6, x1, -2), A1K=step(6, x1, -1), AFFINE_A1=affine(6, x1),
                       BATCH_A1=int(x1 is not None and len(shape(x1)) == 3 and shape(x1)[0] > 1))
    definitions.update((key, geometry['MATRIX_' + key])
                       for key in ('ROWS', 'COLUMNS', 'INNER', 'ROW_GROUPS', 'COLUMN_GROUPS'))
    row_tile = definitions['ROWS'] * definitions['ROW_GROUPS']
    column_tile = definitions['COLUMNS'] * definitions['COLUMN_GROUPS']
    threads = definitions['ROW_GROUPS'] * definitions['COLUMN_GROUPS'] * geometry['SIMD_WIDTH']
    bindings = ('x', 'w', 'out', 'arrivals' if arrivals is not None else None, None,
                'bias' if bias is not None else None, 'x1' if x1 is not None else None,
                'ready' if ready is not None else None)
    return dict(definitions=definitions, bindings=bindings, libraries=LIBRARIES, function='product',
                grid=(-(-rows // row_tile), -(-columns // column_tile), batches), threads=(threads, 1, 1))
