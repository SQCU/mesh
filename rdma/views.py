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
