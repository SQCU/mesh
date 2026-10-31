"""Prepare tensor selection using the shared native radix-selection source."""
from functools import lru_cache
import math
from pathlib import Path
import struct

import torch
from . import _stream, empty
from .operand import TensorView
from .layout import index_source


def row_domain(algebra, inputs, outputs, *, width, adapter, parameters, prefix=False):
    """Independent row results and their declared source fibers.

    Operand views have already put the numerical axis last. Relations retain
    original storage coordinates; neither rows nor output positions prescribe
    physical packing. A prefix read is the scan's mathematical input domain.
    """
    from .operand import layout_configuration
    rows = next(iter(inputs.values()))['view'].shape[0]
    work, uses, reads, writes = (rows, width), {}, {}, {}
    for side, values, accesses in (('read', inputs, reads), ('write', outputs, writes)):
        for role, value in values.items():
            shape = tuple(value['view'].shape)
            columns = shape[1] if len(shape) == 2 else 1
            def equations(i, o, side=side, columns=columns, shape=shape):
                if not math.prod(shape):
                    return ['false']
                result = [f'floor({o[0]}/{columns}) = {i[0]}']
                if side == 'write' and len(shape) == 2:
                    result.append(f'{o[0]} mod {columns} = {i[1]}')
                elif side == 'read' and prefix and len(shape) == 2:
                    result.append(f'{o[0]} mod {columns} <= {i[1]}')
                return result
            logical = algebra.map(work, (math.prod(shape),), equations)
            storage = algebra.layout(layout_configuration(value['view']))
            access = dict(value, relation=logical.then(storage))
            accesses[role] = access
            uses[role] = dict(access=access, logical=logical, storage=storage, shape=shape, side=side)
    return dict(kind='rows', work=work, reads=reads, writes=writes, reduction=None, finish=None,
                adapter=adapter, parameters=parameters, uses=uses, prefix=prefix)


def selected_rows(request):
    """Bind selected row fibers to the existing numerical adapter operands."""
    from .operand import byte_relation, iteration_order, relation_layout
    domain, work = request['domain'], request['work']
    algebra = work.algebra
    plans = []
    for part in work.products((1,)):
        row_order, _ = iteration_order(part.project(0), domain['work'][:1])
        rows = row_order.shape[0]
        parameters, operands, maps, complete = dict(domain['parameters']), {}, {}, {}
        ordered_prefix = domain['adapter'] == 'topk' and parameters['sorted']
        extent = part.project(1).edge(True) + 1 if domain['prefix'] or ordered_prefix else domain['work'][1]
        if ordered_prefix:
            parameters['k'] = extent
        # The adapter's local row ordinal names the exact original row set;
        # its numerical axis retains the declared rank/prefix coordinates.
        local = algebra.map((rows, extent), (rows,), lambda i, o: [f'{o[0]} = {i[0]}']).then(row_order)
        coordinates = algebra.map((rows, extent), domain['work'], lambda i, o: [f'{o[1]} = {i[1]}'])
        row_coordinates = local.then(algebra.map(domain['work'][:1], domain['work'], lambda i, o: [f'{o[0]} = {i[0]}']))
        to_work = coordinates.intersect(row_coordinates)
        for role, use in domain['uses'].items():
            selected = part
            if use['side'] == 'write':
                requested = request['writes'][role]['requested']
                selected = selected.binary(byte_relation(use['access']).preimage(requested), 'intersect')
            positions = use['logical'].image(selected)
            order, _ = iteration_order(positions, (math.prod(use['shape']),))
            count = order.shape[0]
            shape = ((rows, count // rows) if len(use['shape']) == 2 else (rows,)) if use['side'] == 'read' else (count,)
            operands[role] = dict(access=use['access'], relation=order.then(use['storage']), shape=shape,
                                  access_mode=use['side'], requirements=())
            if use['side'] == 'write':
                inverse = order.inverse((math.prod(use['shape']),))
                mapped = to_work.then(use['logical']).then(inverse).reshape((rows * extent,))
                absent = algebra.map((rows * extent,), (count + 1,), lambda i, o: [f'{o[0]} = {count}'])
                total = mapped.union(absent.subtract_domain(mapped.domain()))
                maps[role] = relation_layout(total, (rows * extent,), count + 1)
                complete[role] = mapped.domain().equal(absent.domain())
        parameters['write_maps'] = maps
        parameters['write_complete'] = complete
        plans.append(dict(adapter=domain['adapter'], operands=operands, parameters=parameters))
    return tuple(plans)


@lru_cache(None)
def pipelines(definitions, offsets, names):
    directory = Path(__file__).parent
    source = ''.join(f'#define {k} {v}\n' for k, v in definitions)
    source += offsets + (directory / 'argmax.metal').read_text() + '\n'
    source += (directory / 'topk.metal').read_text()
    return tuple(_stream.pipeline(source, name) for name in names)


def topk(x, k, dim=-1, largest=True, sorted=True, *, outputs=None, write_maps=None, write_complete=None):
    """Values, indices and prepared commands. Views bind without packing."""
    kinds = {torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
    if x.dtype not in kinds or not x.ndim:
        raise NotImplementedError('native topk requires a floating tensor with a selection dimension')
    dim %= x.ndim
    shape = list(x.shape)
    shape[dim] = k
    from .placement import output
    values, indices = ([TensorView.of(value) for value in outputs] if write_maps is not None else
                      [output(shape, dtype, x.device, out) for dtype, out in
                       zip((x.dtype, torch.int64), outputs if outputs is not None else (None, None), strict=True)])
    rows = math.prod(x.shape[:dim] + x.shape[dim + 1:])
    if not k or not rows:
        return (values, indices), []
    geometry = _stream.compilation
    _, limit, memory = _stream.device_facts()
    threads = min(limit, geometry['SELECTION_THREADS'])
    threads -= threads % geometry['SIMD_WIDTH']
    pad = 1 << (k - 1).bit_length()
    words = (1 << geometry['SELECTION_RADIX_BITS']) + geometry['SELECTION_STATE_WORDS'] + threads // geometry['SIMD_WIDTH']
    local = sorted and (pad + words) * struct.calcsize('I') <= memory
    definitions = dict(INPUT=kinds[x.dtype],
                       EXTENT=x.shape[dim], COUNT=k, PAD=pad, THREADS=threads,
                       LARGEST=int(largest), SORTED=int(sorted), LOCAL=int(local),
                       VALUE_COUNT=values.numel(), INDEX_COUNT=indices.numel(),
                       VALUE_SELECTED=int(write_maps is not None and not write_complete['values']),
                       INDEX_SELECTED=int(write_maps is not None and not write_complete['indices']))

    source = TensorView.of(x).rows(dim)
    target, positions = ((values, indices) if write_maps is not None else
                         tuple(TensorView.of(value).rows(dim) for value in (values, indices)))
    maps = ({8: write_maps['values'], 9: write_maps['indices']} if write_maps is not None else
            {8: _stream.TensorLayout((rows * k,), (1,)), 9: _stream.TensorLayout((rows * k,), (1,))})
    offsets = index_source({0: source, 1: target, 2: positions, **maps})
    external = sorted and not local
    names = ('select_top', 'order_top', 'write_top') if external else ('select_top',)
    handles = pipelines(tuple(definitions.items()), offsets, names)
    for handle in handles:
        maximum, _, fixed = _stream.pipeline_facts(handle)
        if threads <= 0 or threads > maximum or fixed > memory:
            raise ValueError('selection geometry exceeds the configured device/pipeline capability')
    scratch = empty((rows, pad), dtype=torch.uint32) if external else None
    grid, group = (rows, 1, 1), (threads, 1, 1)
    shift = geometry['ACCESS_WRITE_SHIFT']
    records = [(handles[0], [source, target, positions, scratch], {}, grid, group, False,
                1 | ((8 if external else 6) << shift))]
    if external:
        width = 2
        while width <= pad:
            stride = width // 2
            while stride:
                records.append((handles[1], [source, scratch], {2: struct.pack('2I', width, stride)},
                                grid, group, False, 3 | (2 << shift)))
                stride //= 2
            width *= 2
        records.append((handles[2], [source, target, positions, scratch], {}, grid, group, False, 9 | (6 << shift)))
    return (values, indices), records
