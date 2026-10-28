"""Metadata-only logical byte views shared by array front ends.

The native WRITE preparer validates lease bounds, destination aliases and
common DMA cuts. This module never reads, allocates or packs tensor values.
"""
import itertools
import operator


def _byte_geometry(shape, strides, itemsize):
    shape = tuple(map(operator.index, shape))
    strides = tuple(map(operator.index, strides))
    itemsize = operator.index(itemsize)
    if len(shape) != len(strides) or itemsize < 0 or any(n < 0 for n in shape):
        raise ValueError('a byte view needs matching nonnegative shape and element width')
    if not itemsize or any(n == 0 for n in shape):
        return shape, strides, 0, 0
    width = itemsize
    outer = len(shape)
    while outer and (shape[outer - 1] == 1 or strides[outer - 1] == width):
        outer -= 1
        width *= shape[outer]
    return shape, strides, width, outer


def _row_geometry(shape, strides, width, outer):
    if not width:
        return 0, 0, 0
    axes = [(n, step) for n, step in zip(shape[:outer], strides[:outer]) if n > 1]
    pitch = axes[-1][1] if axes else width
    rows = 1
    affine = pitch >= width
    for n, step in reversed(axes):
        affine = affine and step == pitch * rows
        rows *= n
    if affine:
        return width, rows, pitch
    return 0, 0, 0


def byte_rows(shape, strides, itemsize):
    """Return affine row operands, or zeros, in time bounded by dimensions."""
    return _row_geometry(*_byte_geometry(shape, strides, itemsize))


def byte_layout(shape, strides, itemsize):
    """Return ``(base_shift, (row_bytes, rows, pitch), spans)`` in C order.

    Offsets are relative to the lowest addressed byte, at ``base_shift``.
    Affine views have constant-size row operands and an empty span list.
    Other views have zero row operands and explicit spans. Both describe the
    same page-bound WRITE, including reversed and broadcast source views.
    """
    shape, strides, width, outer = _byte_geometry(shape, strides, itemsize)
    geometry = _row_geometry(shape, strides, width, outer)
    if geometry[1] or not width:
        return 0, geometry, []
    lowest = sum((n - 1) * step for n, step in zip(shape, strides) if step < 0)
    spans = []
    for coordinate in itertools.product(*(range(n) for n in shape[:outer])):
        offset = sum(i * step for i, step in zip(coordinate, strides)) - lowest
        if spans and spans[-1][0] + spans[-1][1] == offset:
            spans[-1] = (spans[-1][0], spans[-1][1] + width)
        else:
            spans.append((offset, width))
    return lowest, (0, 0, 0), spans


def byte_spans(shape, strides, itemsize):
    """Expand layout intervals only for callers that need explicit alias runs.

    Transport frontends consume ``byte_layout`` directly, retaining affine rows.
    """
    shift, (width, rows, pitch), spans = byte_layout(shape, strides, itemsize)
    if rows:
        spans = ([(0, width * rows)] if pitch == width else
                 [(row * pitch, width) for row in range(rows)])
    return shift, spans


def alias_runs(segments):
    """Concatenate described views in logical order, without reading their data.

    Each segment is (identity, shape, strides, base offset). Units may be bytes
    or typed elements; strides and offsets use the same unit throughout.
    """
    runs, cursor = [], 0
    for identity, shape, strides, base in segments:
        shift, spans = byte_spans(shape, strides, 1)
        for offset, width in spans:
            runs.append((cursor, cursor + width, identity, base + shift + offset))
            cursor += width
    return runs, cursor


def cut_runs(runs, cuts):
    """Resolve agreed logical cuts to contiguous aliases of the original views."""
    pieces, index = [], 0
    for first, last in zip(cuts, cuts[1:]):
        while runs[index][1] <= first:
            index += 1
        lo, hi, identity, offset = runs[index]
        if first < lo or last > hi:
            raise ValueError('common fragment crosses a local contiguous alias')
        pieces.append((identity, offset + first - lo, last - first))
    return pieces
