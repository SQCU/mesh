"""Prepare hierarchical tensor scans with the shared threadgroup prefix sum."""
from functools import lru_cache
import math
from pathlib import Path

import torch
from . import _stream, empty
from .native import row_offset


@lru_cache(None)
def pipelines(definitions, offsets):
    directory = Path(__file__).parent
    source = ''.join(f'#define {k} {v}\n' for k, v in definitions)
    source += (directory / 'argmax.metal').read_text() + '\n' + offsets
    source += (directory / 'scan.metal').read_text()
    return tuple(_stream.pipeline(source, name) for name in ('scan_blocks', 'add_prefix'))


def cumsum(x, dim, *, dtype=None):
    kinds = {torch.bool: 'bool', torch.uint8: 'uchar', torch.int8: 'char', torch.int16: 'short',
             torch.int32: 'int', torch.int64: 'long', torch.uint32: 'uint', torch.uint64: 'ulong',
             torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
    dtype = dtype or (x.dtype if x.is_floating_point() else torch.int64)
    if x.dtype not in kinds or dtype not in kinds or dtype == torch.bool:
        raise NotImplementedError('native cumsum requires a supported numeric output type')
    output = empty(x.shape, dtype=dtype)
    if not x.numel():
        return output, []
    source = x if x.ndim else x.view(1)
    target = output if x.ndim else output.view(1)
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
    definitions = dict(INPUT=kinds[source.dtype], OUTPUT=kinds[dtype], ACCUM=kinds[accumulator],
                       INPUT_STRIDE=source.stride(dim), OUTPUT_STRIDE=target.stride(dim),
                       EXTENT=extent, BLOCKS=blocks, THREADS=threads)

    offsets = row_offset('source_base', source.shape, source.stride(), dim) + row_offset('output_base', source.shape, target.stride(), dim)
    handles = pipelines(tuple(definitions.items()), offsets)
    for handle in handles:
        maximum, _, fixed = _stream.pipeline_facts(handle)
        if threads > maximum or fixed > memory:
            raise ValueError('scan geometry exceeds the configured device/pipeline capability')
    totals = empty((rows, blocks), dtype=accumulator) if blocks > 1 else None
    # Keep floating partials at accumulation precision until the final store.
    partial = (target.view(accumulator) if target.element_size() == accumulator.itemsize
               else empty(source.shape, dtype=accumulator)) if blocks > 1 else None
    grid, group = (blocks, rows, 1), (threads, 1, 1)
    shift = geometry['ACCESS_WRITE_SHIFT']
    records = [(handles[0], [source, target, totals, partial], {}, grid, group, False,
                1 | ((12 if blocks > 1 else 2) << shift))]
    if blocks > 1:
        prefix, nested = cumsum(totals, 1, dtype=accumulator)
        records.extend(nested)
        records.append((handles[1], [target, prefix, partial], {}, grid, group, False, 6 | (1 << shift)))
    return output, records
