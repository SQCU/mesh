"""Lower ordered page selections to the existing collective compiler's operands.

Only preparation calls these functions. Logical cuts, operator records and rank
roles are retained in the declarations passed to the native routing compiler.
"""
from bisect import bisect_right
from itertools import accumulate
import math

from .operand import TensorView, ordered_runs
from . import _stream


def selection_exchanges(sources, destinations, *, algebra=None):
    """Paired local page positions for explicitly declared owner selections."""
    import json
    from mesh_relation import Algebra
    algebra = Algebra() if algebra is None else algebra
    return tuple((source, destination, *(
        _stream.TensorLayout(relation.shape, (1,), index_shape=(count,),
                             index_coordinates=json.dumps([relation.terms()]))
        for relation, count in ((read, source_count), (write, destination_count))))
        for source, destination, read, write, source_count, destination_count in
        algebra.exchanges(sources, destinations))


def selected_calls(record, algebra):
    """Project declared math onto exact contributor and recipient domains.

    The common relation cover supplies cells with equal roles. Routing and
    reduction ordering remain the existing collective compiler's work.
    """
    from .backend import ALLREDUCE, REDUCE, BROADCAST
    source = record['call']
    roles = [('read', rank, relation.domain().binary(record['work'], 'intersect'))
             for rank, relation in record['read_relations'].items()]
    roles.extend(('write', rank, domain) for rank, domain in record['requested'].items())
    extent = _domains(source, 0, record['group'].size())[0]
    for domain, members in algebra.cover([domain for _, _, domain in roles]):
        contributors = tuple(roles[i][1] for i in members if roles[i][0] == 'read')
        recipients = tuple(roles[i][1] for i in members if roles[i][0] == 'write')
        if not recipients:
            continue
        order = algebra.selection_relation(dict(shape=[extent], steps=[['select', 0, domain.spans()]]))
        identity = len(contributors) == 1
        kind = BROADCAST if identity else REDUCE if len(recipients) == 1 else ALLREDUCE
        root = contributors[0] if identity else recipients[0]
        call = source._replace(kind=kind, count=order.shape[0], root=root, counts=None, matrix=None,
                               contributors=contributors, recipients=recipients)
        yield dict(call=call, reads={rank: order.then(record['read_relations'][rank]) for rank in contributors},
                   writes={rank: order.then(record['write_relations'][rank]) for rank in recipients},
                   identity=identity)


def _domains(call, rank, size):
    from .backend import REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, WRITE
    if call.kind == ALLTOALL:
        edges = list(accumulate([0, *call.matrix]))
        sends = [(edges[rank * size + q], edges[rank * size + q + 1]) for q in range(size)]
        receives = [(edges[q * size + rank], edges[q * size + rank + 1]) for q in range(size)]
        total = edges[-1]
    elif call.kind in (ALLGATHER, REDUCE_SCATTER):
        edges = list(accumulate([0, *call.counts]))
        own, whole = [(edges[rank], edges[rank + 1])], [(0, edges[-1])]
        total, sends, receives = (edges[-1], own, whole) if call.kind == ALLGATHER else (edges[-1], whole, own)
    else:
        total, whole = call.count, [(0, call.count)]
        sends = whole if call.kind not in (BROADCAST, WRITE) or rank == call.root else []
        receives = whole if (call.kind != REDUCE or rank == call.root) and (call.kind != WRITE or rank == call.destination) else []
    if call.contributors is not None and rank not in call.contributors:
        sends = []
    if call.recipients is not None and rank not in call.recipients:
        receives = []
    return total, sends, receives


def domain_relations(call, rank, size, algebra):
    """The same logical call positions used by projection and physical binding.

    Relations map common call positions to this rank's compact scalar operands.
    Counts and contributor roles are declarations; routing never enters them.
    """
    total, sends, receives = _domains(call, rank, size)
    def relation(spans):
        selection = dict(shape=[total], steps=[['select', 0, [(first, last - first, 0, 1) for first, last in spans]]])
        return algebra.selection_relation(selection).inverse((total,))
    return relation(sends), relation(receives)


def complete_calls(calls, group):
    """Resolve declared peer counts through the existing configuration exchange."""
    from .backend import ALLTOALL
    calls = list(calls)
    missing = [i for i, call in enumerate(calls) if call.kind == ALLTOALL and call.matrix is None]
    if missing:
        for i, matrix in zip(missing, group.count_matrices([calls[i] for i in missing]), strict=True):
            calls[i] = calls[i]._replace(matrix=matrix)
    return calls


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
    # The graph supplies this rank's counts. The library completes an all-to-all's
    # matrix from every rank's, and pairs them, before physical routing is compiled.
    bindings = [(call, source, destination) for call, (_, source, destination) in zip(
        complete_calls([call for call, _, _ in bindings], group), bindings, strict=True)]
    cuts, cursor, prepared, before, after = [0], 0, [], [], []
    layouts = api_layouts('reduction')
    for call, source, destination in bindings:
        total, sends, receives = _domains(call, rank, size)
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
            # a cut where this rank's storage breaks inside its domain, never at the domain's own ends: a
            # contiguous binding is one call, its segments the library's (mesh_program_steps_cut), so every
            # root's pieces interleave on every cable instead of one root's call after another's
            cuts.extend(cursor + end for end in traversal.cuts if any(start < end < stop for start, stop in domain))
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
