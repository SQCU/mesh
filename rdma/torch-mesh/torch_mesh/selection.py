"""Prepare tensor selection using the shared native radix-selection source."""
from functools import lru_cache
import math
from pathlib import Path
import struct

import torch
from . import _stream, empty
from .operand import TensorView
from .layout import index_source


@lru_cache(None)
def pipelines(definitions, offsets, names):
    directory = Path(__file__).parent
    source = ''.join(f'#define {k} {v}\n' for k, v in definitions)
    source += offsets + (directory / 'argmax.metal').read_text() + '\n'
    source += (directory / 'topk.metal').read_text()
    return tuple(_stream.pipeline(source, name) for name in names)


def topk(x, k, dim=-1, largest=True, sorted=True, *, outputs=None):
    """Values, indices and prepared commands. Views bind without packing."""
    kinds = {torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}
    if x.dtype not in kinds or not x.ndim:
        raise NotImplementedError('native topk requires a floating tensor with a selection dimension')
    dim %= x.ndim
    shape = list(x.shape)
    shape[dim] = k
    from .placement import output
    values, indices = [output(shape, dtype, x.device, out) for dtype, out in
                       zip((x.dtype, torch.int64), outputs if outputs is not None else (None, None), strict=True)]
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
                       LARGEST=int(largest), SORTED=int(sorted), LOCAL=int(local))

    source, target, positions = (TensorView.of(value).rows(dim) for value in (x, values, indices))
    offsets = index_source({0: source, 1: target, 2: positions})
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
