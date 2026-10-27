"""Metadata-only logical byte views shared by array front ends.

The native WRITE preparer validates lease bounds, destination aliases and
common DMA cuts. This module never reads, allocates or packs tensor values.
"""
import itertools
import operator


def rebase(shape, strides, offset, placed_strides, placed_offset,
           view_shape, view_strides, view_offset):
    """Express a logical alias in a placed view, using bounded integer relations.

    All offsets and strides have the same unit. This is a preparation operation;
    its cost depends on the number of dimensions, not the number of elements.
    A non-affine alias needs a different declared placement, not a packing copy.
    """
    shape, strides, placed_strides, view_shape, view_strides = (
        tuple(map(operator.index, values))
        for values in (shape, strides, placed_strides, view_shape, view_strides))
    offset, placed_offset, view_offset = map(operator.index, (offset, placed_offset, view_offset))
    if len(shape) != len(strides) or len(shape) != len(placed_strides) or len(view_shape) != len(view_strides):
        raise ValueError('placement dimensions must match their strides')
    if any(n < 0 for n in (*shape, *view_shape)):
        raise ValueError('placement dimensions must be nonnegative')
    if shape == view_shape and strides == view_strides and offset == view_offset:
        return placed_offset, placed_strides
    if 0 in view_shape:
        return placed_offset, view_strides
    axes = sorted(((s, p, n) for n, s, p in zip(shape, strides, placed_strides) if n > 1), reverse=True)
    merged = []
    for stride, placed, count in axes:
        if stride <= 0 or placed < 0:
            raise ValueError('a placed source needs positive logical strides')
        if merged and merged[-1][0] == stride * count and merged[-1][1] == placed * count:
            _, _, previous = merged.pop()
            count *= previous
        merged.append((stride, placed, count))

    def bounds(base, steps):
        spans = [(n - 1) * step for n, step in zip(view_shape, steps)]
        return base + sum(min(0, span) for span in spans), base + sum(max(0, span) for span in spans)

    remainder, steps = view_offset - offset, list(view_strides)
    result, result_steps = placed_offset, [0] * len(view_shape)
    for stride, placed, count in merged:
        quotient = remainder // stride
        deltas = [(remainder + step) // stride - quotient if n > 1 else 0
                  for n, step in zip(view_shape, steps)]
        remainder -= quotient * stride
        steps = [step - delta * stride if n > 1 else 0
                 for n, step, delta in zip(view_shape, steps, deltas)]
        lo, hi = bounds(remainder, steps)
        first, last = bounds(quotient, deltas)
        if lo < 0 or hi >= stride or first < 0 or last >= count:
            raise ValueError('alias cannot be represented by this affine placement')
        result += quotient * placed
        result_steps = [step + delta * placed for step, delta in zip(result_steps, deltas)]
    if bounds(remainder, steps) != (0, 0):
        raise ValueError('alias names elements outside its placed source')
    # Singleton strides have no effect on addresses; retain a nonzero stride.
    return result, tuple(step if n > 1 else max(1, step) for n, step in zip(view_shape, result_steps))


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
