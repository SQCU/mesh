"""Prepare tensor selection using the shared native radix-selection source."""
import math
from pathlib import Path
import struct

import torch
from . import _stream
from .operand import TensorView
from .command_declarations import Commands, dense, rows, descriptions, configuration, bind


def row_domain(algebra, inputs, outputs, *, width, adapter, parameters, prefix=False):
    """Independent row results and their declared source fibers.

    Operand views have already put the numerical axis last. Relations retain
    original storage coordinates; neither rows nor output positions prescribe
    physical packing. A prefix read is the scan's mathematical input domain.
    """
    from .operand import layout_configuration
    rows = next(iter(outputs.values()))['view'].shape[0]
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
            if 'segments' in use:
                # Ordered occurrences retain multiplicity even when several
                # summands name the same storage scalar. Segment boundaries
                # rank those occurrences in the selected coordinate domain;
                # empty segments get the next boundary, including the end.
                shape = (count,)
                coordinate = order.then(use['segments']).then(
                    to_work.reshape((rows * extent,)).inverse(domain['work']))
                candidates = algebra.map((rows * extent + 1,), (rows * extent,),
                    lambda i, o: [f'{o[0]} >= {i[0]}']).then(coordinate.inverse((rows * extent,)))
                terminal = algebra.map((rows * extent + 1,), (count + 1,),
                    lambda i, o: [f'{o[0]} = {count}'])
                boundaries = candidates.union(terminal).lexmin()
                parameters.setdefault('read_extents', {})[role] = relation_layout(
                    boundaries, (rows * extent + 1,), count + 1)
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


_KINDS = {torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
_SOURCE = Path(__file__).with_name('argmax.metal').read_text() + '\n' + Path(__file__).with_name('topk.metal').read_text()


def declaration(operands, parameters, configuration):
    """The existing selection and ordering passes before page binding."""
    commands = Commands(operands, **configuration)
    source = commands.view('x')
    if source['dtype'] not in _KINDS or not source['layout'].shape:
        raise NotImplementedError('native topk requires a floating tensor with a selection dimension')
    k, dim = parameters['k'], parameters.get('dim', -1)
    largest, ordered = parameters.get('largest', True), parameters.get('sorted', True)
    source = rows(source, dim)
    count, extent = source['layout'].shape
    if not k or not count:
        return commands.declaration()
    maps, complete = parameters.get('write_maps'), parameters.get('write_complete')
    target, positions = (commands.view(role) for role in ('values', 'indices'))
    if maps is None:
        target, positions = rows(target, dim), rows(positions, dim)
    geometry, capabilities = commands.geometry, commands.capabilities
    threads = min(capabilities['max_threads_per_threadgroup'], geometry['SELECTION_THREADS'])
    threads -= threads % geometry['SIMD_WIDTH']
    if threads <= 0:
        raise ValueError('selection geometry requires at least one configured SIMD group')
    memory = capabilities['threadgroup_memory_bytes']
    pad = 1 << (k - 1).bit_length()
    words = (1 << geometry['SELECTION_RADIX_BITS']) + geometry['SELECTION_STATE_WORDS'] + threads // geometry['SIMD_WIDTH']
    local = ordered and (pad + words) * struct.calcsize('I') <= memory
    definitions = dict(INPUT=_KINDS[source['dtype']], EXTENT=extent, COUNT=k, PAD=pad, THREADS=threads,
                       LARGEST=int(largest), SORTED=int(ordered), LOCAL=int(local),
                       VALUE_COUNT=target['layout'].count, INDEX_COUNT=positions['layout'].count,
                       VALUE_SELECTED=int(maps is not None and not complete['values']),
                       INDEX_SELECTED=int(maps is not None and not complete['indices']))
    indices = ({8: maps['values'], 9: maps['indices']} if maps is not None else
               {8: dense((count * k,)), 9: dense((count * k,))})
    external = ordered and not local
    scratch = commands.temporary('topk/scratch', (count, pad), torch.uint32,
                                 requirements=('contiguous',)) if external else None
    grid, group = (count, 1, 1), (threads, 1, 1)
    shift = geometry['ACCESS_WRITE_SHIFT']
    options = dict(definitions=definitions, indices=indices, grid=grid, threadgroup=group,
                   limits={'threadgroup_memory_bytes': memory})
    commands.shader(_SOURCE, 'select_top', [source, target, positions, scratch],
                    access=1 | ((8 if external else 6) << shift), **options)
    if external:
        width = 2
        while width <= pad:
            stride = width // 2
            while stride:
                commands.shader(_SOURCE, 'order_top', [source, scratch],
                    constants={2: struct.pack('2I', width, stride)}, access=3 | (2 << shift), **options)
                stride //= 2
            width *= 2
        commands.shader(_SOURCE, 'write_top', [source, target, positions, scratch],
                        access=9 | (6 << shift), **options)
    return commands.declaration()


def topk(x, k, dim=-1, largest=True, sorted=True, *, outputs=None, write_maps=None, write_complete=None):
    """Values, indices and prepared commands. Views bind without packing."""
    from .placement import output
    from .native import command_records
    if x.dtype not in _KINDS or not x.ndim:
        raise NotImplementedError('native topk requires a floating tensor with a selection dimension')
    dim %= x.ndim
    shape = list(x.shape)
    shape[dim] = k
    values, indices = ([TensorView.of(value) for value in outputs] if write_maps is not None else
                      [output(shape, dtype, x.device, out) for dtype, out in
                       zip((x.dtype, torch.int64), outputs if outputs is not None else (None, None), strict=True)])
    bound = dict(x=x, values=values, indices=indices)
    prepared = declaration(descriptions(bound),
        dict(k=k, dim=dim, largest=largest, sorted=sorted, write_maps=write_maps, write_complete=write_complete),
        configuration())
    return (values, indices), list(command_records(bind(prepared, bound)))
