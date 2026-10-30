"""Ordered graph slicing shared by tensor frontends, during preparation only.

Frontends supply value identities, reads, writes, result roots and effects.
Views name their shared storage root; regions describe the positions each value
reads or writes there. Selection preserves the given order. Cuts introduce no
operation or dependency. None of these objects reaches numerical execution.
"""
from bisect import bisect_right


def alias_roots(pairs):
    """Canonical storage identities from declared alias equalities.

    Frontends may name either side as a source. Joining the equalities once
    avoids interpreting those names as a directed descriptor chain.
    """
    parent = {}

    def root(value):
        parent.setdefault(value, value)
        first = value
        while parent[value] != value:
            value = parent[value]
        while first != value:
            following = parent[first]
            parent[first] = value
            first = following
        return value

    for left, right in pairs:
        left, right = root(left), root(right)
        if left != right:
            parent[left] = right
    return {value: root(value) for value in parent if root(value) != value}


def view_source(value, views=()):
    """The shared storage identity behind a declared chain of views."""
    while value in views:
        value = views[value]
    return value


def dependencies(values, views=()):
    """The storage roots named by these operands, during preparation."""
    return {view_source(value, views) for value in values}


class _Atomic(frozenset):
    """A whole identity is one position in the same region algebra."""
    def binary(self, other, operation):
        name = {'union': 'union', 'intersect': 'intersection', 'subtract': 'difference'}[operation]
        return _Atomic(getattr(self, name)(other))

    def empty(self):
        return not self


_ATOM = _Atomic((0,))


def _accesses(reads, writes, roots, views, regions, constants=()):
    values = set(roots) | set(constants)
    values.update(value for entries in (reads, writes) for row in entries for value in row)
    sources = {value: view_source(value, views) for value in values | set(regions)}
    regional = {sources[value] for value in regions}
    result = {}
    for value in values:
        source = sources[value]
        if source in regional:
            if value not in regions:
                raise ValueError(f'{value!r}: missing domain in shared root {source!r}')
            domain = regions[value]
        else:
            domain = _ATOM
        result[value] = source, domain
    return result


def _include(regions, source, domain):
    if not domain.empty():
        regions[source] = regions[source].binary(domain, 'union') if source in regions else domain


def _subtract(regions, source, domain):
    if source in regions:
        left = regions[source].binary(domain, 'subtract')
        if left.empty():
            del regions[source]
        else:
            regions[source] = left


def _needed(reads, writes, roots, cuts, effects, accesses):
    wanted, effects = {}, set(effects)
    for value in roots:
        _include(wanted, *accesses[value])
    spans = [[] for _ in range(len(cuts) + 1)]
    for i in range(len(writes) - 1, -1, -1):
        if i in effects or any(source in wanted and not wanted[source].binary(domain, 'intersect').empty()
                               for source, domain in (accesses[value] for value in writes[i])):
            for value in writes[i]:
                _subtract(wanted, *accesses[value])
            for value in reads[i]:
                _include(wanted, *accesses[value])
            spans[bisect_right(cuts, i)].append(i)
    return [span[::-1] for span in spans]


def needed(reads, writes, roots, cuts=(), effects=(), views=(), regions=()):
    """Indices needed by roots/effects, grouped before and after each cut.

    ``reads[i]`` and ``writes[i]`` name operation i's values. An effect names an
    operation that must remain even when none of its results is returned.
    Duplicate cuts retain empty spans, so adjacent boundaries keep their slots.

    ``regions[value]`` is its symbolic position set in the coordinates of its
    storage root. Every referenced value sharing that root supplies its domain.
    Identity-only roots use atomic domains. A write to a view defines exactly
    that region; a later read retains only the preceding writes it intersects.
    Sets implement ``binary(other, union/intersect/subtract)`` and ``empty()``;
    mesh_relation.IndexSet provides the exact integer algebra.
    """
    roots = tuple(roots)
    return _needed(reads, writes, roots, cuts, effects, _accesses(reads, writes, roots, views, regions))


def _sources(owners, source, remaining):
    found = []
    for owner, written, domain in owners.get(source, ()):
        if not remaining.binary(domain, 'intersect').empty():
            found.append((owner, written))
            remaining = remaining.binary(domain, 'subtract')
    return found, remaining


def _define(owners, owner, value, source, domain):
    previous = []
    for at, written, held in owners.get(source, ()):
        left = held.binary(domain, 'subtract')
        if not left.empty():
            previous.append((at, written, left))
    if not domain.empty():
        previous.append((owner, value, domain))
    owners[source] = previous


def binding_cuts(reads, writes, cuts=(), views=(), regions=()):
    """Additional cuts where a provider must expose a changed intermediate view.

    Same-ID reads remain inside the provider. A different ID reading positions
    written in this span must bind the producer's exposed pages. This includes
    assembled-root reads of output views, and excludes disjoint positions.
    Supplied cuts already expose their preceding span; duplicates stay with the
    caller's original boundary list rather than being returned again here.
    """
    accesses = _accesses(reads, writes, (), views, regions)
    declared, added, owners = set(cuts), [], {}
    for i, inputs in enumerate(reads):
        if i in declared:
            owners.clear()
        if any(written != value for value in inputs
               for _, written in _sources(owners, *accesses[value])[0]):
            added.append(i)
            owners.clear()
        for value in writes[i]:
            _define(owners, i, value, *accesses[value])
    return added


def _frontiers(reads, writes, spans, roots, constants, accesses):
    owners, constant_regions = {}, {}
    for value in constants:
        _include(constant_regions, *accesses[value])
    result = [(set(), set()) for _ in spans]
    for stage, span in enumerate(spans):
        for i in span:
            for value in reads[i]:
                source, remaining = accesses[value]
                found, remaining = _sources(owners, source, remaining)
                for previous, written in found:
                    if previous != stage:
                        result[stage][0].add(value)
                        result[previous][1].add(written)
                if source in constant_regions:
                    remaining = remaining.binary(constant_regions[source], 'subtract')
                if not remaining.empty():
                    result[stage][0].add(value)
            for value in writes[i]:
                _define(owners, stage, value, *accesses[value])
    for value in roots:
        if accesses[value][1].empty():
            if result:
                result[-1][1].add(value)
            continue
        for stage, written in _sources(owners, *accesses[value])[0]:
            result[stage][1].add(written)
    return result


def frontiers(reads, writes, spans, roots, constants=(), views=(), regions=()):
    """Input and output value sets for each selected span, without copies.

    Constants stay with their frontend's constant table. An output is exposed
    only when a later span or a result root needs its written region. The
    consumer keeps its read ID; the producer exports its actual output ID.
    Disjoint writes share a root without becoming a whole-result join. Later
    writes replace only their own positions in the preceding definitions.
    Empty result domains retain their declared identity on the final span and
    require no producer or source operand.
    """
    roots, constants = tuple(roots), tuple(constants)
    return _frontiers(reads, writes, spans, roots, constants,
                      _accesses(reads, writes, roots, views, regions, constants))


def partition(reads, writes, roots, cuts=(), effects=(), constants=(), external=(), views=(), regions=()):
    """Selected operations and their boundary operands, in declaration order.

    External operations declare inputs/constants/results in frontends such as
    FX. They participate in dependency selection but emit no numerical command.
    Rewritten value identities refer to their most recent preceding definition.
    """
    external, roots, constants = set(external), tuple(roots), tuple(constants)
    accesses = _accesses(reads, writes, roots, views, regions, constants)
    spans = [[i for i in span if i not in external] for span in _needed(reads, writes, roots, cuts, effects, accesses)]
    return [(span, inputs, outputs) for span, (inputs, outputs) in
            zip(spans, _frontiers(reads, writes, spans, roots, constants, accesses))]
