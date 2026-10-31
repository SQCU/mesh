"""Prepare hierarchical tensor scans with the shared threadgroup prefix sum."""
from functools import lru_cache
import math
from pathlib import Path

import torch
from . import _stream, empty
from .operand import TensorView
from .layout import index_source


@lru_cache(None)
def pipelines(definitions, offsets, finishing):
    directory = Path(__file__).parent
    definitions = ''.join(f'#define {k} {v}\n' for k, v in definitions)
    source = (directory / 'argmax.metal').read_text() + '\n' + (directory / 'scan.metal').read_text()
    return tuple(_stream.pipeline(definitions + indices + source, name)
                 for indices, name in ((offsets, 'scan_blocks'), (finishing, 'add_prefix')))


def cumsum(x, dim, *, dtype=None, out=None, write_maps=None, write_complete=None):
    kinds = {torch.bool: 'bool', torch.uint8: 'uchar', torch.int8: 'char', torch.int16: 'short',
             torch.int32: 'int', torch.int64: 'long', torch.uint32: 'uint', torch.uint64: 'ulong',
             torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
    dtype = dtype or (x.dtype if x.is_floating_point() else torch.int64)
    if x.dtype not in kinds or dtype not in kinds or dtype == torch.bool:
        raise NotImplementedError('native cumsum requires a supported numeric output type')
    from .placement import output as bind_output
    output = TensorView.of(out) if write_maps is not None else bind_output(x.shape, dtype, x.device, out)
    if not x.numel():
        return output, []
    source, target = TensorView.of(x), TensorView.of(output)
    if not source.ndim:
        source, target = source.reshape((1,)), target.reshape((1,))
    dim %= source.ndim
    rows = math.prod(source.shape[:dim] + source.shape[dim + 1:])
    accumulator = torch.float32 if output.is_floating_point() else (torch.uint64 if output.element_size() > 4 else torch.uint32)
    geometry = _stream.compilation
    _, limit, memory = _stream.device_facts()
    threads = min(limit, geometry['SCAN_THREADS'])
    threads -= threads % geometry['SIMD_WIDTH']
    if threads <= 0:
        raise ValueError('scan geometry requires at least one configured SIMD group')
    extent = source.shape[dim]
    blocks = -(-extent // threads)
    totals = empty((rows, blocks), dtype=accumulator) if blocks > 1 else None
    # Partial sums use their own layout when accumulation changes the width.
    source = source.rows(dim)
    target = target if write_maps is not None else target.rows(dim)
    partial = target
    if blocks > 1:
        complete = write_maps is None or write_complete['out']
        if complete and target.element_size() == accumulator.itemsize:
            if write_maps is not None:
                physical = write_maps['out'].rebase(_stream.TensorLayout((target.numel(),), (1,)),
                                                  target.layout.reshape((target.numel(),)))
                partial = TensorView(target.storage, physical.reshape(source.shape), target.binding_origin)
            partial = partial.typed(accumulator)
        else:
            partial = empty(source.shape, dtype=accumulator)
    definitions = dict(INPUT=kinds[source.dtype], OUTPUT=kinds[dtype], ACCUM=kinds[accumulator],
                       EXTENT=extent, BLOCKS=blocks, THREADS=threads, OUTPUT_COUNT=target.numel(),
                       OUTPUT_SELECTED=int(write_maps is not None and not write_complete['out']))

    selected = write_maps['out'] if write_maps is not None else _stream.TensorLayout((rows * extent,), (1,))
    offsets = index_source({0: source, 1: target, 2: partial, 8: selected})
    finishing = index_source({0: target, 1: target, 2: partial, 8: selected})
    handles = pipelines(tuple(definitions.items()), offsets, finishing)
    for handle in handles:
        maximum, _, fixed = _stream.pipeline_facts(handle)
        if threads > maximum or fixed > memory:
            raise ValueError('scan geometry exceeds the configured device/pipeline capability')
    grid, group = (blocks, rows, 1), (threads, 1, 1)
    shift = geometry['ACCESS_WRITE_SHIFT']
    records = [(handles[0], [source, target, totals, partial], {}, grid, group, False,
                1 | ((12 if blocks > 1 else 2) << shift))]
    if blocks > 1:
        prefix, nested = cumsum(totals, 1, dtype=accumulator)
        records.extend(nested)
        records.append((handles[1], [target, prefix, partial], {}, grid, group, False, 6 | (1 << shift)))
    return output, records
