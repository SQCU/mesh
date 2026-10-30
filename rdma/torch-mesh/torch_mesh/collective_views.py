"""Lower ordered page selections to the existing collective compiler's operands.

Only preparation calls these functions. Logical cuts, operator records and rank
roles are retained in the declarations passed to the native routing compiler.
"""
from bisect import bisect_right
from itertools import accumulate
import math

from .operand import TensorView, ordered_runs
from . import _stream


def _domains(call, rank, size):
    from .backend import REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL
    if call.kind == ALLTOALL:
        edges = list(accumulate([0, *call.matrix]))
        sends = [(edges[rank * size + q], edges[rank * size + q + 1]) for q in range(size)]
        receives = [(edges[q * size + rank], edges[q * size + rank + 1]) for q in range(size)]
        return edges[-1], sends, receives
    if call.kind in (ALLGATHER, REDUCE_SCATTER):
        edges = list(accumulate([0, *call.counts]))
        own, whole = [(edges[rank], edges[rank + 1])], [(0, edges[-1])]
        return (edges[-1], own, whole) if call.kind == ALLGATHER else (edges[-1], whole, own)
    whole = [(0, call.count)]
    sends = whole if call.kind != BROADCAST or rank == call.root else []
    receives = whole if call.kind != REDUCE or rank == call.root else []
    return call.count, sends, receives


def _binding(value, domain, unit, writing, layouts):
    """Resolve the numerical consumer's field constraint before cutting rows."""
    from .layout import resolve
    value = TensorView.of(value)
    runs, count = ordered_runs([(value.storage, value)])
    if count != sum(end - start for start, end in domain):
        raise ValueError('collective page selection must cover its declared local domain')
    commands, spans = [], [(value.storage, value)]
    # A scalar is one field. A record is indivisible numerical work, even when
    # its rows are gathered or separated by gaps in the original allocation.
    split = sorted({end // unit for _, end, _, _ in runs if end % unit})
    if split:
        rows = value.reshape((count // unit, unit))
        spans, first, at = [], 0, 0
        while at < len(split):
            start, end = split[at], split[at] + 1
            at += 1
            while at < len(split) and split[at] == end:
                end += 1
                at += 1
            if first < start:
                spans.append((value.storage, rows.slice(0, first, start - first)))
            placed, copies = resolve(rows.slice(0, start, end - start),
                                     (layouts['output' if writing else 'input'],), writing=writing)
            placed = TensorView.of(placed)
            spans.append((placed.storage, placed))
            commands.extend(copies)
            first = end
        if first < count // unit:
            spans.append((value.storage, rows.slice(0, first, count // unit - first)))
    selection = _stream.TensorLayout.runs([[first, end - first, 0, 1] for first, end in domain])
    requested = _stream.TensorLayout.runs([[0, max((end for _, end in domain), default=0), 0, 1]])
    traversal = _stream.TensorTraversal([view.layout for _, view in spans], selection, requested)
    return spans, traversal, commands


def _alias(spans, traversal, first, last):
    part, offset, count = traversal.fragment(first, last)
    return spans[part][0].as_strided((count,), (1,), offset)


def lower(bindings, group):
    """Return native bindings, matching declarations, and required copy edges."""
    from .backend import ALLTOALL, ALLGATHER, REDUCE_SCATTER
    from .compilation import api_layouts
    bindings = list(bindings)
    rank, size = group.rank(), group.size()
    # The graph supplies per-peer counts. Exchange this configuration once to
    # complete an all-to-all's matrix, before physical routing is compiled.
    missing = [i for i, (c, _, _) in enumerate(bindings) if c.kind == ALLTOALL and c.matrix is None]
    if missing:
        rows = group.exchange_configuration([bindings[i][0].counts for i in missing])
        for k, i in enumerate(missing):
            call, source, destination = bindings[i]
            matrix = [rows[r][k][q] for r in range(size) for q in range(size)]
            if any(rows[r][k][size + q] != matrix[q * size + r] for r in range(size) for q in range(size)):
                raise ValueError('all-to-all declarations must pair send and receive counts')
            bindings[i] = call._replace(matrix=matrix), source, destination
    cuts, cursor, prepared, before, after = [0], 0, [], [], []
    layouts = api_layouts('reduction')
    for call, source, destination in bindings:
        total, sends, receives = _domains(call, rank, size)
        if call.contributors is not None and rank not in call.contributors:
            sends = []
        if call.recipients is not None and rank not in call.recipients:
            receives = []
        unit = math.prod(call.element_shape)
        if total % unit or any(count % unit for count in call.counts or ()):
            raise ValueError('collective counts must contain complete declared operator elements')
        values = []
        exemplar = TensorView.of(source if source is not None else destination)
        for value, domain, writing in ((source, sends, False), (destination, receives, True)):
            if not domain or not any(end > start for start, end in domain):
                value = exemplar.storage.as_strided((0,), (1,), 0)
            spans, traversal, commands = _binding(value, domain, unit, writing, layouts)
            (after if writing else before).extend(commands)
            cuts.extend(cursor + end for end in traversal.cuts)
            values.append((spans, traversal))
        prepared.append((call, cursor, total, values))
        cursor += total
        cuts.append(cursor)
    cuts = group.union_cuts(cuts)
    lowered, declarations = [], []
    for call, origin, total, values in prepared:
        begin = bisect_right(cuts, origin) - 1
        end = bisect_right(cuts, origin + total)
        edges = [cut - origin for cut in cuts[begin:end]]
        intervals = zip(edges, edges[1:]) if total else [(0, 0)]
        counts = call.matrix if call.kind == ALLTOALL else call.counts
        bounds = list(accumulate([0, *counts])) if counts is not None else []
        for first, last in intervals:
            fragment = call._replace(count=last - first)
            if call.kind in (ALLGATHER, REDUCE_SCATTER, ALLTOALL):
                counts = [max(0, min(last, end) - max(first, start)) for start, end in zip(bounds, bounds[1:])]
                if call.kind == ALLTOALL:
                    own = counts[rank * size:(rank + 1) * size]
                    fragment = fragment._replace(count=sum(own), matrix=counts,
                        counts=[*own, *(counts[q * size + rank] for q in range(size))])
                else:
                    fragment = fragment._replace(counts=counts)
            lowered.append((fragment, *(_alias(*value, first, last) for value in values)))
            declarations.append(fragment)
    return lowered, declarations, before, after
