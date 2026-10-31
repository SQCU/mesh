"""Prepare hierarchical tensor scans with the shared threadgroup prefix sum."""
import math
from pathlib import Path

import torch
from .operand import TensorView
from .command_declarations import Commands, dense, rows, descriptions, configuration, bind


_KINDS = {torch.bool: 'bool', torch.uint8: 'uchar', torch.int8: 'char', torch.int16: 'short',
          torch.int32: 'int', torch.int64: 'long', torch.uint32: 'uint', torch.uint64: 'ulong',
          torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
_SOURCE = Path(__file__).with_name('argmax.metal').read_text() + '\n' + Path(__file__).with_suffix('.metal').read_text()


def declaration(operands, parameters, configuration):
    """The scan hierarchy, typed slots and selected writes before page binding."""
    commands = Commands(operands, **configuration)
    geometry, capabilities = commands.geometry, commands.capabilities
    threads = min(capabilities['max_threads_per_threadgroup'], geometry['SCAN_THREADS'])
    threads -= threads % geometry['SIMD_WIDTH']
    if threads <= 0:
        raise ValueError('scan geometry requires at least one configured SIMD group')
    limits = {'threadgroup_memory_bytes': capabilities['threadgroup_memory_bytes']}
    shift = geometry['ACCESS_WRITE_SHIFT']

    def scan(source, target, dim, maps=None, complete=None, prefix='scan'):
        dtype = target['dtype']
        if source['dtype'] not in _KINDS or dtype not in _KINDS or dtype == torch.bool:
            raise NotImplementedError('native cumsum requires a supported numeric output type')
        if not source['layout'].count:
            return
        if not source['layout'].shape:
            source, target = (dict(value, layout=value['layout'].reshape((1,))) for value in (source, target))
        source = rows(source, dim)
        target = target if maps is not None else rows(target, dim)
        count, extent = source['layout'].shape
        accumulator = torch.float32 if dtype.is_floating_point else (torch.uint64 if dtype.itemsize > 4 else torch.uint32)
        blocks = -(-extent // threads)
        totals = commands.temporary(prefix + '/totals', (count, blocks), accumulator,
                                    requirements=('contiguous',)) if blocks > 1 else None
        partial = target
        if blocks > 1:
            if (maps is None or complete['out']) and dtype.itemsize == accumulator.itemsize:
                mapping = target['layout'] if maps is None else maps['out'].rebase(
                    dense((target['layout'].count,)), target['layout'].reshape((target['layout'].count,)))
                partial = dict(target, layout=mapping.reshape(source['layout'].shape), dtype=accumulator)
            else:
                partial = commands.temporary(prefix + '/partial', source['layout'].shape, accumulator)
        definitions = dict(INPUT=_KINDS[source['dtype']], OUTPUT=_KINDS[dtype], ACCUM=_KINDS[accumulator],
                           EXTENT=extent, BLOCKS=blocks, THREADS=threads, OUTPUT_COUNT=target['layout'].count,
                           OUTPUT_SELECTED=int(maps is not None and not complete['out']))
        selected = maps['out'] if maps is not None else dense((count * extent,))
        grid, group = (blocks, count, 1), (threads, 1, 1)
        commands.shader(_SOURCE, 'scan_blocks', [source, target, totals, partial],
            definitions=definitions, indices={8: selected}, grid=grid, threadgroup=group,
            access=1 | ((12 if blocks > 1 else 2) << shift), limits=limits)
        if blocks > 1:
            result = commands.temporary(prefix + '/prefix', (count, blocks), accumulator,
                                        requirements=('contiguous',))
            scan(totals, result, 1, prefix=prefix + '/nested')
            commands.shader(_SOURCE, 'add_prefix', [target, result, partial], definitions=definitions,
                indices={8: selected}, grid=grid, threadgroup=group, access=6 | (1 << shift), limits=limits)

    scan(commands.view('x'), commands.view('out'), parameters['dim'], parameters.get('write_maps'),
         parameters.get('write_complete'))
    return commands.declaration()


def cumsum(x, dim, *, dtype=None, out=None, write_maps=None, write_complete=None):
    from .placement import output as bind_output
    from .native import command_records
    dtype = dtype or (x.dtype if x.is_floating_point() else torch.int64)
    output = TensorView.of(out) if write_maps is not None else bind_output(x.shape, dtype, x.device, out)
    values = dict(x=x, out=output)
    prepared = declaration(descriptions(values), dict(dim=dim, write_maps=write_maps, write_complete=write_complete), configuration())
    return output, list(command_records(bind(prepared, values)))
