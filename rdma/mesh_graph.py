"""Ordered graph slicing shared by tensor frontends, during preparation only.

Operands are value identities, not tensors or storage. Frontends supply reads,
writes, result roots and required effects. Selection preserves the given order;
cuts mark existing boundaries and introduce no operation or dependency.
"""
from bisect import bisect_right


def view_source(value, views=()):
    """The defining value behind a declared chain of read views."""
    while value in views:
        value = views[value]
    return value


def dependencies(values, views=()):
    """The defining values read by these operands, during preparation."""
    return {view_source(value, views) for value in values}


def needed(reads, writes, roots, cuts=(), effects=(), views=()):
    """Indices needed by roots/effects, grouped before and after each cut.

    ``reads[i]`` and ``writes[i]`` name operation i's values. An effect names an
    operation that must remain even when none of its results is returned.
    Duplicate cuts retain empty spans, so adjacent boundaries keep their slots.

    Read views have no operator of their own. Their source producers remain in
    the dependency cone, including when the requested result is itself a view.
    """
    wanted, effects = dependencies(roots, views), set(effects)
    spans = [[] for _ in range(len(cuts) + 1)]
    for i in range(len(writes) - 1, -1, -1):
        if i in effects or wanted.intersection(writes[i]):
            wanted.difference_update(writes[i])
            wanted.update(dependencies(reads[i], views))
            spans[bisect_right(cuts, i)].append(i)
    return [span[::-1] for span in spans]


def frontiers(reads, writes, spans, roots, constants=(), views=()):
    """Input and output value sets for each selected span, without copies.

    Constants stay with their frontend's constant table. An output is exposed
    only when a later span or a result root needs it. Values retain their
    identities across boundaries; the frontend binds their original storage.
    """
    owner, constants = {}, set(constants)
    result = [(set(), set()) for _ in spans]
    for stage, span in enumerate(spans):
        for i in span:
            for value in reads[i]:
                source = view_source(value, views)
                previous = owner.get(source)
                if previous != stage and source not in constants:
                    result[stage][0].add(value)
                    if previous is not None:
                        result[previous][1].add(source)
            owner.update((value, stage) for value in writes[i])
    for value in dependencies(roots, views):
        if value in owner:
            result[owner[value]][1].add(value)
    return result


def partition(reads, writes, roots, cuts=(), effects=(), constants=(), external=(), views=()):
    """Selected operations and their boundary operands, in declaration order.

    External operations declare inputs/constants/results in frontends such as
    FX. They participate in dependency selection but emit no numerical command.
    Rewritten value identities refer to their most recent preceding definition.
    """
    external, roots = set(external), tuple(roots)
    spans = [[i for i in span if i not in external] for span in needed(reads, writes, roots, cuts, effects, views)]
    return [(span, inputs, outputs) for span, (inputs, outputs) in
            zip(spans, frontiers(reads, writes, spans, roots, constants, views))]
