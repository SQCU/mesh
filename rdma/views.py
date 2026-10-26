"""Metadata-only logical byte views shared by array front ends.

The native WRITE preparer validates lease bounds, destination aliases and
common DMA cuts. This module never reads, allocates or packs tensor values.
"""
import itertools
import operator


def byte_spans(shape, strides, itemsize):
    """Return ``(base_shift, [(offset, bytes), ...])`` in logical C order.

    Offsets are nonnegative relative to the lowest addressed byte, whose
    displacement from the array's first logical element is ``base_shift``.
    Contiguous suffix dimensions collapse into one span; reversed/broadcast
    dimensions retain their actual mapping. Native preparation decides whether
    overlaps are valid for a source or invalid for a destination.
    """
    shape = tuple(map(operator.index, shape))
    strides = tuple(map(operator.index, strides))
    itemsize = operator.index(itemsize)
    if len(shape) != len(strides) or itemsize < 0 or any(n < 0 for n in shape):
        raise ValueError('a byte view needs matching nonnegative shape and element width')
    if not itemsize or any(n == 0 for n in shape):
        return 0, []
    lowest = sum((n - 1) * step for n, step in zip(shape, strides) if step < 0)
    width = itemsize
    outer = len(shape)
    while outer and (shape[outer - 1] == 1 or strides[outer - 1] == width):
        outer -= 1
        width *= shape[outer]
    spans = []
    for coordinate in itertools.product(*(range(n) for n in shape[:outer])):
        offset = sum(i * step for i, step in zip(coordinate, strides)) - lowest
        if spans and spans[-1][0] + spans[-1][1] == offset:
            spans[-1] = (spans[-1][0], spans[-1][1] + width)
        else:
            spans.append((offset, width))
    return lowest, spans


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
