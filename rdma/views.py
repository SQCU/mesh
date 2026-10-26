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


def access_spans(shape, strides, itemsize):
    """The bytes a view touches, independent of its logical traversal order.

    Transfer lists preserve logical order through ``byte_spans``. Dependency
    ordering needs the union instead: broadcasts name one range and a dense
    transpose still names one range. All work here happens during preparation.
    """
    shape, strides = tuple(shape), tuple(strides)
    if len(shape) != len(strides):
        raise ValueError('an access view needs one stride per dimension')
    axes = sorted(range(len(shape)), key=lambda i: abs(strides[i]), reverse=True)
    shift, spans = byte_spans([min(shape[i], 1) if not strides[i] else shape[i] for i in axes],
                             [abs(strides[i]) for i in axes], itemsize)
    if not spans:
        return 0, []
    shift += sum((n - 1) * step for n, step in zip(shape, strides) if n and step < 0)
    union = []
    for offset, width in sorted(spans):
        if union and offset <= union[-1][0] + union[-1][1]:
            lo, size = union[-1]
            union[-1] = lo, max(lo + size, offset + width) - lo
        else:
            union.append((offset, width))
    return shift, union


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
