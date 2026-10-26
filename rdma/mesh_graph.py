"""Ordered graph slicing shared by tensor frontends, during preparation only.

Operands are value identities, not tensors or storage. Frontends supply reads,
writes, result roots and required effects. Selection preserves the given order;
cuts mark existing boundaries and introduce no operation or dependency.
"""
from bisect import bisect_right


def needed(reads, writes, roots, cuts=(), effects=()):
    """Indices needed by roots/effects, grouped before and after each cut.

    ``reads[i]`` and ``writes[i]`` name operation i's values. An effect names an
    operation that must remain even when none of its results is returned.
    Duplicate cuts retain empty spans, so adjacent boundaries keep their slots.
    """
    wanted, effects = set(roots), set(effects)
    spans = [[] for _ in range(len(cuts) + 1)]
    for i in range(len(writes) - 1, -1, -1):
        if i in effects or wanted.intersection(writes[i]):
            wanted.update(reads[i])
            spans[bisect_right(cuts, i)].append(i)
    return [span[::-1] for span in spans]


def frontiers(reads, writes, spans, roots, constants=()):
    """Input and output value sets for each selected span, without copies.

    Constants stay with their frontend's constant table. An output is exposed
    only when a later span or a result root needs it. Values retain their
    identities across boundaries; the frontend binds their original storage.
    """
    owner = {value: stage for stage, span in enumerate(spans) for i in span for value in writes[i]}
    required = [{value for i in span for value in reads[i]} for span in spans]
    later, constants = set(roots), set(constants)
    result = []
    for stage in range(len(spans) - 1, -1, -1):
        inputs = {value for value in required[stage] if owner.get(value) != stage and value not in constants}
        outputs = {value for value in later if owner.get(value) == stage}
        result.append((inputs, outputs))
        later.update(required[stage])
    return result[::-1]
