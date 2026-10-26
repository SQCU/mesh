"""Prepare tensor selection using the shared native radix-selection source."""
from functools import lru_cache
import math
from pathlib import Path
import struct

import torch
from . import _stream, empty


@lru_cache(None)
def pipelines(definitions, offsets, names):
    directory = Path(__file__).parent
    source = ''.join(f'#define {k} {v}\n' for k, v in definitions)
    source += (directory / 'argmax.metal').read_text() + '\n' + offsets
    source += (directory / 'topk.metal').read_text()
    return tuple(_stream.pipeline(source, name) for name in names)


def topk(x, k, dim=-1, largest=True, sorted=True):
    """Values, indices and prepared commands. Views bind without packing."""
    kinds = {torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
    if x.dtype not in kinds or not x.ndim:
        raise NotImplementedError('native topk requires a floating tensor with a selection dimension')
    dim %= x.ndim
    shape = list(x.shape)
    shape[dim] = k
    values, indices = empty(shape, dtype=x.dtype), empty(shape, dtype=torch.int64)
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
    definitions = dict(INPUT=kinds[x.dtype], INPUT_STRIDE=x.stride(dim), OUTPUT_STRIDE=values.stride(dim),
                       EXTENT=x.shape[dim], COUNT=k, PAD=pad, THREADS=threads,
                       LARGEST=int(largest), SORTED=int(sorted), LOCAL=int(local))

    def offset(name, strides):
        terms, divisor = [], 1
        for axis in range(x.ndim - 1, -1, -1):
            if axis != dim:
                terms.append(f'(ulong(row) / {divisor} % {x.shape[axis]}) * {strides[axis]}')
                divisor *= x.shape[axis]
        return f'inline ulong {name}(uint row) {{ return ' + (' + '.join(terms) or '0') + '; }\n'

    offsets = offset('source_base', x.stride()) + offset('output_base', values.stride())
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
    records = [(handles[0], [x, values, indices, scratch], {}, grid, group, False,
                1 | ((8 if external else 6) << shift))]
    if external:
        width = 2
        while width <= pad:
            stride = width // 2
            while stride:
                records.append((handles[1], [x, scratch], {2: struct.pack('2I', width, stride)},
                                grid, group, False, 3 | (2 << shift)))
                stride //= 2
            width *= 2
        records.append((handles[2], [x, values, indices, scratch], {}, grid, group, False, 9 | (6 << shift)))
    return (values, indices), records
