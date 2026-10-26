"""Native matrix command operands shared by FX and the streamed DNN program."""
from functools import lru_cache
from pathlib import Path
import struct

import torch
from . import _stream, fused


@lru_cache(None)
def pipeline(definitions):
    source = fused.steps() + '\n' + ''.join(f'#define {key} {value}\n' for key, value in definitions)
    return _stream.pipeline(source + Path(__file__).with_suffix('.metal').read_text(), 'product')


def product(x, w, out, *, x1=None, bias=None, act=False, alpha=1, beta=1, publication=None, arrivals=None):
    """Bind matrix views, including batch strides, without packing or execution."""
    kinds = {torch.float16: 'half', torch.float32: 'float', torch.bfloat16: 'bfloat'}
    if x.dtype != w.dtype or x.dtype not in kinds or out.dtype not in kinds:
        raise ValueError('matrix operands require supported, explicit input and output types')
    if x.ndim not in (2, 3) or w.ndim != x.ndim or out.ndim != x.ndim:
        raise ValueError('matrix operands require two-dimensional or batched matrix views')
    rows, first, columns = x.shape[-2], x.shape[-1], w.shape[-1]
    if x1 is not None and (x1.dtype != x.dtype or x1.ndim != x.ndim or x1.shape[-2] != rows):
        raise ValueError('matrix input segments require the same dtype, rank and row count')
    inner = first + (x1.shape[-1] if x1 is not None else 0)
    if w.shape[-2] != inner or out.shape[-2:] != (rows, columns):
        raise ValueError('matrix views disagree on product dimensions')
    batches = out.shape[0] if out.ndim == 3 else 1
    if x.ndim == 3 and (x.shape[0] not in (1, batches) or w.shape[0] not in (1, batches)):
        raise ValueError('matrix batch views must match or broadcast')
    if x1 is not None and x1.ndim == 3 and x1.shape[0] not in (1, batches):
        raise ValueError('matrix input segment batch views must match or broadcast')
    if bias is not None and (bias.dtype != out.dtype or bias.ndim not in (1, 2) or
                            bias.shape[-1] not in (1, columns) or
                            (bias.ndim == 2 and bias.shape[0] not in (1, rows))):
        raise ValueError('matrix bias requires the output dtype and a broadcastable row view')
    channels = len(publication.publish) if publication is not None else 0
    stride = lambda t: t.stride(0) if t.ndim == 3 and t.shape[0] > 1 else 0
    bias_row = bias.stride(0) if bias is not None and bias.ndim == 2 and bias.shape[0] > 1 else 0
    bias_col = bias.stride(-1) if bias is not None and bias.shape[-1] > 1 else 0
    definitions = dict(INPUT=kinds[x.dtype], OUTPUT=kinds[out.dtype], AM=x.stride(-2), AK=x.stride(-1),
                       BK=w.stride(-2), BN=w.stride(-1), YM=out.stride(-2), YN=out.stride(-1),
                       ABATCH=stride(x), BBATCH=stride(w), YBATCH=stride(out), CM=bias_row, CN=bias_col,
                       ACTIVATE=int(act), BIAS=int(bias is not None), ALPHA=repr(float(alpha))+'f',
                       BETA=repr(float(beta))+'f', CHANNELS=channels, FUSED=int(channels > 0),
                       SEGMENTED=int(x1 is not None), K0=first,
                       A1M=x1.stride(-2) if x1 is not None else 0,
                       A1K=x1.stride(-1) if x1 is not None else 0,
                       A1BATCH=stride(x1) if x1 is not None else 0)
    geometry = _stream.compilation
    definitions.update((key, geometry['MATRIX_' + key])
                       for key in ('ROWS', 'COLUMNS', 'INNER', 'ROW_GROUPS', 'COLUMN_GROUPS'))
    row_tile = definitions['ROWS'] * definitions['ROW_GROUPS']
    column_tile = definitions['COLUMNS'] * definitions['COLUMN_GROUPS']
    threads = definitions['ROW_GROUPS'] * definitions['COLUMN_GROUPS'] * geometry['SIMD_WIDTH']
    words = [rows, inner, columns, batches]
    if publication is not None:
        words = fused._encode_words(words, channels, None, publication)
    return (pipeline(tuple(definitions.items())), [x, w, out, arrivals, None, bias, x1],
            {4: struct.pack(f'{len(words)}I', *words)}, (-(-rows // row_tile), -(-columns // column_tile), batches),
            (threads, 1, 1), False, 3 | (32 if bias is not None else 0) | (64 if x1 is not None else 0)
            | ((12 if publication is not None else 4) << 32))
