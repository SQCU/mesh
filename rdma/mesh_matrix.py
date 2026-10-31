"""Matrix shader declarations from scalar, shape and capability operands.

The importer and FX frontend supply metadata to this module. Page placement,
shader compilation and command recording consume its result in their shared
lowerings; this module neither allocates numerical storage nor executes work.
"""
from math import prod


LIBRARIES = ('argmax.metal', 'matrix_operand.metal', 'linear.metal', 'matrix_rows.metal', 'matrix.metal')


def row_tiles(rows, geometry):
    """The product's compiled row groups, also used by tile publications."""
    width = (geometry['LINEAR_ROWS'] if rows <= geometry['LINEAR_ROW_CROSSOVER']
             else geometry['MATRIX_ROWS'] * geometry['MATRIX_ROW_GROUPS'])
    return tuple((first, min(width, rows - first)) for first in range(0, rows, width))


def row_split(groups, inner, geometry, capabilities):
    """SIMD groups per row tile from declared work and compute capacity."""
    split = 1
    maximum = min(geometry['LINEAR_SIMDGROUPS'], capabilities['max_threads_per_threadgroup'] // geometry['SIMD_WIDTH'])
    while (split * 2 <= maximum and split * groups < capabilities['compute_units'] * geometry['LINEAR_GROUPS_PER_CORE']
           and inner // (split * 2) >= geometry['LINEAR_MIN_INNER']):
        split *= 2
    return split


def product_domains(algebra, x, w, out, *, x1=None, bias=None):
    """The product's work, operand reads and result positions before placement.

    Each relation maps the declared work coordinates to one operand's logical
    ordinal. Physical views compose with these relations in the shared binder.
    The finish domain has no reduction axis, including for an empty contraction.
    """
    from mesh_relation import bounds, flat
    xs, ws, ys = (tuple(value['shape']) for value in (x, w, out))
    first = xs[-1]
    batches = ys[0] if len(ys) == 3 else 1
    iterations = (batches, ys[-2], ys[-1])
    work = (*iterations, ws[-2])

    def access(domain, value, positions):
        shape = tuple(value['shape'])
        def equations(i, o):
            coordinates = positions(i)
            return [f'{o[0]} = {flat(shape, coordinates)}', *bounds(shape, coordinates)]
        return algebra.map(domain, (prod(shape),), equations)

    def matrix_positions(shape, i, row, column):
        batch = [i[0] if shape[0] != 1 else '0'] if len(shape) == 3 else []
        return [*batch, row, column]

    reads = {
        'x': access(work, x, lambda i: matrix_positions(xs, i, i[1], i[3])),
        'w': access(work, w, lambda i: matrix_positions(ws, i, i[3], i[2])),
    }
    if x1 is not None:
        shape = tuple(x1['shape'])
        reads['x1'] = access(work, x1, lambda i: matrix_positions(shape, i, i[1], f'({i[3]} - {first})'))
    output = lambda i: matrix_positions(ys, i, i[1], i[2])
    finish_reads = {}
    if bias is not None:
        shape = tuple(bias['shape'])
        finish_reads['bias'] = access(iterations, bias, lambda i:
            ([i[1] if shape[0] != 1 else '0'] if len(shape) == 2 else []) +
            [i[2] if shape[-1] != 1 else '0'])
    return dict(work=work, reads=reads, writes={'out': access(work, out, output)},
                reduction=dict(operation='sum', axes=(len(iterations),), identity=0),
                finish=dict(work=iterations, reads=finish_reads,
                            writes={'out': access(iterations, out, output)}))


def product(x, w, out, geometry, capabilities, *, x1=None, bias=None, act=False, alpha=1, beta=1,
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
        return f'(mesh_affine<{binding}>() && !mesh_observed<{binding}>())' if value is not None else 0

    definitions = dict(INPUT=x['type'], WEIGHT=w['type'], OUTPUT=out['type'],
                       AM=step(0, x, -2), AK=step(0, x, -1), BK=step(1, w, -2), BN=step(1, w, -1),
                       AFFINE_A=affine(0, x), AFFINE_B=affine(1, w),
                       BATCH_A=int(len(xs) == 3 and xs[0] > 1), BATCH_B=int(len(ws) == 3 and ws[0] > 1),
                       BIAS_ROWS=shape(bias)[0] if bias is not None and len(shape(bias)) == 2 else 1,
                       BIAS_COLUMNS=shape(bias)[-1] if bias is not None else 1,
                       ACTIVATE=int(act), BIAS=int(bias is not None), ALPHA=repr(float(alpha))+'f',
                       BETA=repr(float(beta))+'f', PRODUCED=int(ready is not None),
                       SEGMENTED=int(x1 is not None), PRODUCT_FIRST=first, PRODUCT_ROWS=rows,
                       PRODUCT_COLUMNS=columns, PRODUCT_INNER=inner,
                       A1M=step(6, x1, -2), A1K=step(6, x1, -1), AFFINE_A1=affine(6, x1),
                       BATCH_A1=int(x1 is not None and len(shape(x1)) == 3 and shape(x1)[0] > 1))
    definitions.update((key, geometry['MATRIX_' + key])
                       for key in ('ROWS', 'COLUMNS', 'INNER', 'ROW_GROUPS', 'COLUMN_GROUPS'))
    row_kernel = rows <= geometry['LINEAR_ROW_CROSSOVER']
    row_tile = geometry['LINEAR_ROWS'] if row_kernel else definitions['ROWS'] * definitions['ROW_GROUPS']
    column_tile = geometry['LINEAR_COLUMNS'] if row_kernel else definitions['COLUMNS'] * definitions['COLUMN_GROUPS']
    groups = -(-rows // row_tile) * -(-columns // column_tile) * batches
    split = row_split(groups, inner, geometry, capabilities) if row_kernel else 1
    threads = (split if row_kernel else definitions['ROW_GROUPS'] * definitions['COLUMN_GROUPS']) * geometry['SIMD_WIDTH']
    definitions.update(PRODUCT_ROW_KERNEL=int(row_kernel), ROW_SPLIT=split,
        AB=step(0, x, 0) if len(xs) == 3 else 0,
        BB=step(1, w, 0) if len(ws) == 3 else 0,
        A1B=step(6, x1, 0) if x1 is not None and len(shape(x1)) == 3 else 0)
    bindings = ('x', 'w', 'out', 'arrivals' if arrivals is not None else None, None,
                'bias' if bias is not None else None, 'x1' if x1 is not None else None,
                'ready' if ready is not None else None)
    return dict(definitions=definitions, bindings=bindings, libraries=LIBRARIES, function='product',
                grid=(-(-rows // row_tile), -(-columns // column_tile), batches), threads=(threads, 1, 1),
                tile=(row_tile, column_tile), group_order=(1, 2, 0))


def product_dispatch(algebra, recipe, operands, *, offset=0, total=None, group_order=None):
    """Project the declared product's work onto its emitted threadgroup grid.

    The same tile operands that formed the launch now group its retained access
    relations. Page binding composes these logical ordinals with actual storage.
    """
    domain = product_domains(algebra, **operands)
    rows, columns, batches = recipe['grid']
    row_width, column_width = recipe['tile']
    total = prod(recipe['grid']) if total is None else total
    order = recipe['group_order'] if group_order is None else group_order
    widths = {0: batches, 1: rows, 2: columns}

    def grouped(shape):
        def equations(i, o):
            coordinates = {0: i[0], 1: f'floor({i[1]} / {row_width})',
                           2: f'floor({i[2]} / {column_width})'}
            stride, terms = 1, [str(offset)]
            for axis in order:
                terms.append(f'{stride} * ({coordinates[axis]})')
                stride *= widths[axis]
            return [f'{o[0]} = ' + ' + '.join(terms)]
        return algebra.map(shape, (total,), equations).inverse((total,))

    result = []
    for stage, sides in ((domain, ('reads',)), (domain['finish'], ('reads', 'writes'))):
        groups = grouped(stage['work'])
        for side in sides:
            for role, relation in stage[side].items():
                result.append(dict(side=side, role=role, group_relation=groups.then(relation)))
    return tuple(result)


def selected_products(domain, work, finish, *, requirements):
    """Factor selected product work into ordinary mapped product declarations.

    Equal output fibers share their selected reduction set. Each output is
    written once, including outputs whose reduction set is empty. Role relations
    stay in the named storage's declared scalar coordinates; page binding is a
    later composition. No tensor values or per-element tables are constructed.
    """
    from mesh_relation import flat
    algebra = work.algebra
    original = tuple(domain['work'])
    output_shape = original[:3]
    finish_domain = domain['finish']

    def selected(shape, axes):
        local = tuple(sum(count * repeats for _, count, _, repeats in runs) for runs in axes)
        order = algebra.selection_relation(dict(shape=list(shape),
            steps=[['select', axis, runs] for axis, runs in enumerate(axes)]))
        unflatten = algebra.map((prod(shape),), shape,
            lambda i, o: [f'{i[0]} = {flat(shape, o)}'])
        return local, order.reshape(local).then(unflatten)

    covered = work.project(0, 3)
    empty_outputs = finish.binary(covered, 'subtract')
    fibers = [(part.project(0, 3), part.project(3)) for part in work.products((3,))]
    if not empty_outputs.empty():
        fibers.append((empty_outputs, algebra.set('{ [k] : false }', 1)))
    plans = []
    for outputs, inner in fibers:
        phases = ((outputs.binary(finish, 'intersect'), True), (outputs.binary(finish, 'subtract'), False))
        for rectangle, finished in ((rectangle, finished) for positions, finished in phases
                                     for rectangle in positions.products()):
            axes = rectangle.cartesian()
            full_shape, to_work = selected(original, (*axes, inner.spans()))
            final_shape, to_finish = selected(output_shape, axes)
            batches, rows, columns, size = full_shape
            operands = {}

            def role(name, shape, positions, source, relation):
                embed = algebra.map(shape, source.shape,
                    lambda i, o: [f'{right} = {left}' for left, right in zip(positions(i), o, strict=True)])
                access = relation['reads' if name != 'out' else 'writes'][name]
                operands[name] = dict(access=access, shape=shape,
                                      relation=embed.then(source).then(access['relation']),
                                      requirements=requirements[name],
                                      access_mode='write' if name in relation['writes'] else 'read')

            for name in ('x', 'x1'):
                if name not in domain['reads']:
                    continue
                valid = domain['reads'][name]['relation'].domain().project(3)
                part = inner.binary(valid, 'intersect')
                local, mapping = selected(original, (*axes, part.spans()))
                role(name, (batches, rows, local[3]), lambda i: (i[0], i[1], '0', i[2]), mapping, domain)
            role('w', (batches, size, columns), lambda i: (i[0], '0', i[2], i[1]), to_work, domain)
            role('out', final_shape, lambda i: i, to_finish, finish_domain)
            if finished and 'bias' in finish_domain['reads']:
                role('bias', (rows, columns), lambda i: ('0', i[0], i[1]), to_finish, finish_domain)
            parameters = dict(domain['parameters'])
            if not finished:
                parameters['act'] = False
            plans.append(dict(operands=operands, parameters=parameters))
    return tuple(plans)


def row_product_group(operands, requests, geometry, capabilities):
    """Emit independent row products from the same typed product contract."""
    recipes = {}
    for prefix, options in requests:
        declared = {name[len(prefix):]: value for name, value in operands.items() if name.startswith(prefix)}
        recipes[prefix] = product(**declared, geometry=geometry, capabilities=capabilities, **options)
    slots, bindings, parameters, accesses = {}, [], [], 0
    for name, value in operands.items():
        slot = len(bindings);slots[name] = slot;bindings.append(name)
        scalar = value['type']
        output = name.endswith('_out')
        parameters.append(f'device {"" if output else "const "}{scalar}* value{slot} [[buffer({slot})]]')
        accesses |= 1 << (slot + (geometry['ACCESS_WRITE_SHIFT'] if output else 0))
    tiles, total, least = [], 0, None
    for prefix, options in requests:
        rows, first = operands[prefix + 'x']['shape']
        inner, columns = operands[prefix + 'w']['shape']
        groups = -(-rows // geometry['LINEAR_ROWS']) * -(-columns // geometry['LINEAR_COLUMNS'])
        tiles.append((prefix, options, total, rows, first, inner, columns));total += groups
        least = inner if least is None else min(least, inner)
    split = row_split(total, least, geometry, capabilities)
    lines = ['kernel void products(' + ','.join(parameters) + ',uint tg [[threadgroup_position_in_grid]],'
             'uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],'
             'uint tid [[thread_index_in_threadgroup]]) {',
             f'threadgroup float part[{split}*MESH_LINEAR_ROWS*MESH_LINEAR_COLUMNS];']
    for index, (prefix, options, start, rows, first, inner, columns) in enumerate(reversed(tiles)):
        lines.append(('if' if index == 0 else 'else if') + f'(tg>={start}) {{')
        def reader(role, width, transpose=False):
            slot = slots[prefix + role];dtype = operands[prefix + role]['type']
            row, column = ((1, 0) if transpose else (0, 1))
            read = f'mesh_page_read<{slot},{dtype}>'
            return (f'matrix_row_read<{slot},{dtype},{str(transpose).lower()},{width},mesh_affine<{slot}>(),'
                    f'mesh_matrix_stride<{slot},{row}>(),mesh_matrix_stride<{slot},{column}>(),0,{read}>'
                    f'{{{read}{{value{slot}}},0}}')
        lines.append(f'const auto x={reader("x", first)};const auto w={reader("w", columns, True)};')
        lines.append(f'const auto x1={reader("x1", inner-first) if prefix+"x1" in operands else "x"};')
        bias = prefix + 'bias'
        bias_shape = operands[bias]['shape'] if bias in operands else (0,)
        br, bc = (bias_shape[-2] if len(bias_shape) == 2 else 1), bias_shape[-1]
        out = slots[prefix + 'out'];dtype = operands[prefix + 'out']['type']
        if bias in slots:
            lines.append(f'const mesh_page_read<{slots[bias]},{dtype}> bias{{value{slots[bias]}}};')
        else:
            lines.append(f'const mesh_page_read<{out},{dtype}> bias{{value{out}}};')
        lines.append(f'const linear_dims dims={{{rows},{inner},{columns},0,0,{first},0,0,0,0}};')
        act = str(bool(options.get('act', False))).lower()
        alpha, beta = (repr(float(options.get(key, 1))) + 'f' for key in ('alpha', 'beta'))
        lines.append(f'const matrix_row_store<{out},{dtype},decltype(bias),{rows},{columns},{br},{bc},{act}>'
                     f' store{{(device uchar*)value{out},bias,0,{alpha},{beta}}};')
        lines.append(f'linear_body_read<MESH_LINEAR_COLUMNS,{split},MESH_LINEAR_LOADS>'
                     f'(x,x1,w,dims,store,part,sg,lane,tg-{start},tid);}}')
    lines.append('}')
    return dict(source='\n'.join(lines), function='products', bindings=tuple(bindings), access=accesses,
                grid=(total, 1, 1), threads=(split * geometry['SIMD_WIDTH'], 1, 1),
                products=tuple((prefix, recipes[prefix], start) for prefix, _, start, *_ in tiles))
