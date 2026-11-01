"""Dispatch fibers of the retained streamed source, in argument scalar units.

The encoding, transform and decomposition operands below are the same values
that form each source instantiation. No tensor payload or kernel name is used
when constructing these relations.
"""
import math


def relations(buffers, outputs, grid, threads, dims, indices, declaration, geometry):
    from mesh_relation import Algebra
    algebra = Algebra()
    count = math.prod(grid)
    extents = {slot: math.prod(value.shape) for slot, value in enumerate(buffers) if value is not None}
    result = {slot: algebra.map((count,), (extent,), lambda i, o: ['false']) for slot, extent in extents.items()}
    d = dict(zip(('tiles', 'outputs', 'rows', 'per', 'stride', 'shares', 'split', 'finish', 'count', 'columns',
                  'inputs', 'cap', 'eps', 'flags', 'layer', 'pitch'), dims, strict=True))
    kind = declaration['kind']
    block = geometry['WEIGHT_ROW_GROUP']
    gx = lambda g: f'({g} mod {grid[0]})'
    gy = lambda g: f'floor({g}/{grid[0]})'
    def add(slot, equations):
        if slot in result and extents[slot]:
            result[slot] = result[slot].union(algebra.map((count,), (extents[slot],),
                lambda i, o: equations(i[0], o[0])))
    def interval(slot, low, high, group=None):
        if high <= low:
            return
        add(slot, lambda g, j: [f'{low} <= {j} < {high}', *(group(g) if group else [])])
    def rectangle(slot, columns, row_first, row_end, col_first, col_end, group=None):
        if not columns or row_end <= row_first or col_end <= col_first:
            return
        add(slot, lambda g, j: [f'{row_first} <= floor({j}/{columns}) < {row_end}',
            f'{col_first} <= {j} mod {columns} < {col_end}', *(group(g) if group else [])])
    def members(expression, values):
        values = sorted(set(map(int, values)))
        spans = []
        for value in values:
            if spans and value == spans[-1][1]:
                spans[-1][1] += 1
            else:
                spans.append([value, value + 1])
        return '(' + ' or '.join(f'{lo} <= {expression} < {hi}' for lo, hi in spans) + ')' if spans else 'false'

    if kind == 'finish':
        width, size = d['outputs'], d['count']
        add(1, lambda g, j: [f'floor({j}/{threads}) = {g}'])
        add(0, lambda g, j: [f'floor(({j} mod {d["split"]})/{threads}) = {g}'])
        add(3, lambda g, j: [f'exists (i : {threads}*{g} <= i < {threads}*({g}+1) and i < {size} and {j} = i mod {width})'])
        add(4, lambda g, j: [f'exists (i : {threads}*{g} <= i < {threads}*({g}+1) and i < {size} and {j} = floor(i/{width}))'])
    elif kind == 'scale':
        for slot in (0, 1):
            add(slot, lambda g, j: [f'floor({j}/{d["columns"]}) = {g}'])
        add(2, lambda g, j: [f'{j} = {g}'])
    elif kind in ('gather', 'hadamard', 'rotate'):
        width = d['columns'];inner = declaration['input_columns']
        stores = declaration['stores'];scales = declaration['scales']
        if kind == 'gather':
            sources, places = indices[2], indices[4]
            add(0, lambda g, j: [f'floor({j}/{inner}) = {g}', members(f'({j} mod {inner})', sources)])
            stride = declaration['stride']
            for slot in stores:
                add(slot, lambda g, j: [f'floor({j}/{stride}) mod {d["rows"]} = {g}'] if stride else [f'floor({j}/{width}) = {g}'])
            for slot in scales:
                add(slot, lambda g, j: [members(j, places[:len(sources)])])
        elif kind == 'hadamard':
            add(0, lambda g, j: [f'floor({j}/{inner}) = {g}', members(f'({j} mod {inner})', indices[12])])
            for slot in stores:
                add(slot, lambda g, j: [f'floor({j}/{width}) = {g}'])
            for slot in (*scales, 6, 7):
                interval(slot, 0, extents.get(slot, 0))
        else:
            columns = 4 * threads // geometry['SIMD_WIDTH']
            interval(0, 0, extents[0])
            add(6, lambda g, j: [f'floor(floor({j}/{inner})/{columns}) = {g}'])
            for slot in (*scales, 7):
                add(slot, lambda g, j: [f'floor({j}/{columns}) = {g}'])
            for slot in stores:
                add(slot, lambda g, j: [f'floor(({j} mod {width})/{columns}) = {g}'])
            if declaration['tables']:
                entries = geometry['STREAMED_LUT_ENTRIES']
                add(8, lambda g, j: [f'floor(({j} mod {width//block*entries})/{entries}) = {g}'])
    elif kind == 'gelu':
        width = d['outputs'];chunk = 4 * threads
        for slot, shares, split in ((0, d['shares'], d['split']), (1, d['inputs'], d['per'])):
            add(slot, lambda g, j, split=split: [f'floor(({j} mod {split})/{width}) = {gy(g)}',
                f'floor(({j} mod {width})/{chunk}) = {gx(g)}'])
        for slot in (2, 3, 6):
            add(slot, lambda g, j: [f'floor({j}/{chunk}) = {gx(g)}'])
        add(4, lambda g, j: [f'floor({j}/{width}) = {gy(g)}', f'floor(({j} mod {width})/{chunk}) = {gx(g)}'])
        if d['count']:
            add(7, lambda g, j: [f'floor({j}/{d["count"]}) = {gy(g)}', f'floor(({j} mod {d["count"]})/{chunk}) = {gx(g)}'])
    elif kind in ('dense', 'alu', 'coop', 'panel', 'panel_t', 'couple'):
        # Every matrix's immutable encoding declares exact code words. The
        # same groups name its tile records, numerical input and destination.
        spec = declaration.get('spec', {})
        tiles = d['tiles'];columns = tiles * block;rows = d['rows']
        panels = geometry['STREAMED_PANEL_TILES']
        for matrix_number, metadata in enumerate(declaration['matrices']):
            codes, tile_words, band_words, source = ((0, 1, 2, 4) if matrix_number == 0 else (6, 7, 8, 9))
            bands = len(metadata['bands'])
            for band, band_word in enumerate(metadata['bands']):
                slab = int(band_word) >> geometry['STREAMED_SLAB_SHIFT']
                base = int(band_word) & geometry['STREAMED_BAND_BASE_MASK']
                for tile in range(tiles):
                    word = int(metadata['tiles'][band * tiles + tile]) & 0xffffffff
                    bits = word >> geometry['STREAMED_TILE_WIDTH_SHIFT'] & geometry['STREAMED_TILE_WIDTH_MASK']
                    offset = word & geometry['STREAMED_TILE_OFFSET_MASK']
                    if kind == 'dense':
                        group = lambda g, band=band, tile=tile: [f'{gx(g)} = {band}', f'{gy(g)} = {tile}']
                    elif kind == 'couple':
                        group = lambda g, band=band: [f'{gx(g)} = {band}']
                    elif kind in ('alu', 'coop'):
                        per = -(-tiles // spec['J'])
                        group = lambda g, band=band, tile=tile: [f'{gx(g)} = {band//spec["BANDS"]}', f'{gy(g)} = {tile//per}']
                    elif kind == 'panel':
                        blocks = -(-rows // spec['M']);share = tile // (d['per'] * panels)
                        group = lambda g, band=band, share=share: [f'{gx(g)} = {band}', f'floor({gy(g)}/{blocks}) = {share}']
                    elif kind == 'panel_t':  # the transposed panel's band share.
                        blocks = -(-rows // spec['M'])
                        group = lambda g, band=band, tile=tile: [f'{gx(g)} = {tile//panels}', f'floor({gy(g)}/{blocks}) = {band//d["per"]}']
                    interval(tile_words, band * tiles + tile, band * tiles + tile + 1, group)
                    interval(band_words, band, band + 1, group)
                    plane = geometry['STREAMED_PLANE_WORDS']
                    if bits:
                        interval(codes, (base + offset) * plane, (base + offset + bits) * plane, group)
                    if kind == 'dense':
                        rectangle(4, columns, band * block, (band + 1) * block, tile * block, (tile + 1) * block, group)
                        interval(6, band * block, (band + 1) * block, group)
                        interval(9, tile * block, (tile + 1) * block, group)
                    elif kind == 'panel_t':
                        add(source, lambda g, j, band=band, group=group: [*group(g),
                            f'floor({j}/{rows*block}) = {band}',
                            f'floor((floor({j}/{block}) mod {rows})/{spec["M"]}) = {gy(g)} mod {blocks}'])
                    elif kind == 'panel':
                        add(source, lambda g, j, slab=slab, tile=tile, group=group: [*group(g),
                            f'floor({j}/{rows*columns}) = {slab}',
                            f'floor((floor({j}/{columns}) mod {rows})/{spec["M"]}) = {gy(g)} mod {blocks}',
                            f'{tile*block} <= {j} mod {columns} < {(tile+1)*block}'])
                    elif bits:
                        if kind == 'alu' and spec.get('LUT') and bits == 1:
                            entries = geometry['STREAMED_LUT_ENTRIES']
                            rectangle(3, tiles * entries, 0, rows, tile * entries, (tile + 1) * entries, group)
                        else:
                            height = min(rows, spec.get('R', spec.get('M', rows)))
                            rectangle(source, columns, slab * rows, slab * rows + height,
                                      tile * block, (tile + 1) * block, group)
            if kind == 'dense':
                continue
        if kind in ('alu', 'coop', 'couple'):
            bands_per = spec.get('BANDS', 1)
            height = min(rows, spec.get('R', spec.get('M', rows)))
            if kind == 'couple':
                add(11, lambda g, j: [f'floor(({j} mod {d["outputs"]})/{block}) = {gx(g)}', f'floor({j}/{d["outputs"]}) < {height}'])
                for slot in (10, 13, 14):
                    add(slot, lambda g, j: [f'floor({j}/{block}) = {gx(g)}'])
            else:
                add(5, lambda g, j: [f'floor({j}/{d["split"]}) = {gy(g)}',
                    f'floor(({j} mod {d["outputs"]})/{block*bands_per}) = {gx(g)}',
                    f'floor(({j} mod {d["split"]})/{d["outputs"]}) < {height}'])
        elif kind == 'panel_t':
            blocks = -(-rows // spec['M'])
            add(5, lambda g, j: [f'floor({j}/{d["split"]}) = floor({gy(g)}/{blocks})',
                f'floor((floor({j}/{columns}) mod {rows})/{spec["M"]}) = {gy(g)} mod {blocks}',
                f'floor(({j} mod {columns})/{panels*block}) = {gx(g)}'])
        elif kind == 'panel':
            blocks = -(-rows // spec['M'])
            if d['finish']:
                # Row order is a declaration table; compose it with its group
                # assignment rather than rediscover it from stored output.
                for row, logical in enumerate(declaration['matrices'][0]['row_order']):
                    add(11, lambda g, j, row=row, logical=logical: [f'{gx(g)} = {row//block}',
                        f'{j} mod {d["stride"]} = {logical}', f'floor(floor({j}/{d["stride"]})/{spec["M"]}) = {gy(g)} mod {blocks}'])
                add(13, lambda g, j: [f'floor({j}/{block}) = {gx(g)}'])
            else:
                add(5, lambda g, j: [f'floor({j}/{d["split"]}) = floor({gy(g)}/{blocks})',
                    f'floor((floor({j}/{d["outputs"]}) mod {rows})/{spec["M"]}) = {gy(g)} mod {blocks}',
                    f'floor(({j} mod {d["outputs"]})/{block}) = {gx(g)}'])
    else:
        raise ValueError(f'{kind}: missing source dispatch declaration')
    return tuple(dict(side='writes' if slot in outputs else 'reads', slot=slot, group_relation=relation)
                 for slot, relation in result.items())
