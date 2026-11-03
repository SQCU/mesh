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


def cycles(graph):
    """Every cycle a depth-first walk of ``graph`` ({vertex: [successor, ...]}) closes, each the vertices on the walk
    from the one it returns to, in order (one a back edge; a vertex walked from once)."""
    seen, found = set(), []

    def visit(v, path):
        if v in path:
            found.append(path[path.index(v):])
            return
        if v in seen:
            return
        seen.add(v)
        for w in graph.get(v, ()):
            visit(w, path + [v])
    for v in graph:
        visit(v, [])
    return found


def ancestors(inputs, value, known):
    """Every vertex ``value`` reads, directly or through others: ``inputs(v)`` gives v's direct inputs, and ``known``
    (a dict) keeps each answer for later queries of the same graph."""
    if value not in known:
        seen, stack = set(), list(inputs(value))
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(inputs(v))
        known[value] = seen
    return known[value]


def cyclic(successors, predecessors, value, forward, backward):
    """The vertices on a cycle through ``value``, itself among them: those it reaches that reach it (its strongly
    connected component), empty where it lies on none (a basic block outside every loop). ``successors(v)`` and
    ``predecessors(v)`` give v's direct ones; ``forward`` and ``backward`` keep each direction's answers, as ``known``
    does for ancestors."""
    return ancestors(successors, value, forward) & ancestors(predecessors, value, backward)


def recorded_slack(order, touches, k, end=None):
    """How much earlier item ``order[k]`` of a recorded order could have been recorded, and what follows it:
    ``touches(a, b)`` whether two items reach the same pages (a dependency, either way). The index of the last earlier
    item it touches (None where none does), the indices recorded between that one and k (it could precede them), and
    the later indices before ``end`` (the order's end where None) of the items it touches."""
    last = next((j for j in range(k - 1, -1, -1) if touches(order[j], order[k])), None)
    later = [j for j in range(k + 1, len(order) if end is None else end) if touches(order[k], order[j])]
    return last, list(range(0 if last is None else last + 1, k)), later


def independent(inputs, users, order, group, value, known):
    """Whether ``value`` can join ``group`` in one dispatch at ``value``'s place in ``order``: it reads no member's
    result, directly or through others, and no member's result is used before it (``users(v)`` gives v's direct
    users)."""
    return (not any(member in ancestors(inputs, value, known) for member in group)
            and not any(order[u] < order[value] for member in group for u in users(member)))


def descendants(root, children):
    """Every vertex beneath ``root`` in a tree, in preorder: ``children(v)`` gives v's direct children (an IORegistry
    entry's IORegistryEntryChildren, for one). ``root`` itself is not among them."""
    stack = list(reversed(children(root)))
    while stack:
        v = stack.pop()
        yield v
        stack.extend(reversed(children(v)))


def preorder_parents(depths):
    """Each row's parent row in a tree written in preorder with each row's depth (an indented outline, such as a
    sample(1) call graph): its index, or None for a root."""
    parents, stack = [], []
    for i, depth in enumerate(depths):
        while stack and depths[stack[-1]] >= depth:
            stack.pop()
        parents.append(stack[-1] if stack else None)
        stack.append(i)
    return parents


def retained(items, removable, writes, reads, protected=()):
    """``items`` in order, less every removable one whose values no other kept item reads and none of which is
    protected, the later ones considered first (a dead-code sweep): ``removable(item)`` says whether it may go,
    ``writes(item)`` and ``reads(item)`` name the values it writes and reads. Each value's readers are counted once and
    an item's removal takes its reads off them, so the sweep is one pass over the items' reads."""
    items, protected = list(items), set(protected)
    own = [set(reads(item)) for item in items]
    readers = {}
    for values in own:
        for value in values:
            readers[value] = readers.get(value, 0) + 1
    kept = [True] * len(items)
    for k in reversed(range(len(items))):
        if not removable(items[k]):
            continue
        if any(value in protected or readers.get(value, 0) - (value in own[k]) > 0 for value in writes(items[k])):
            continue
        kept[k] = False
        for value in own[k]:
            readers[value] -= 1
    return [item for item, keep in zip(items, kept) if keep]


def sources(nodes, edges):
    """type: nodes, edges (earlier, later) -> the nodes no edge enters, in the nodes' order"""
    entered = {later for _, later in edges}
    return [node for node in nodes if node not in entered]


def sinks(nodes, edges):
    """type: nodes, edges (earlier, later) -> the nodes no edge leaves, in the nodes' order"""
    left = {earlier for earlier, _ in edges}
    return [node for node in nodes if node not in left]


def through(edges, kept):
    """type: edges (earlier, later), kept nodes -> the edges between kept nodes, each a path whose interior nodes are
    none of them (a run's event with no node passes its edges through: torch_mesh/compiler.py)"""
    successors = {}
    for earlier, later in edges:
        successors.setdefault(earlier, []).append(later)
    out = set()
    for start in kept:
        seen, stack = set(), list(successors.get(start, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in kept:
                out.add((start, node))
            else:
                stack.extend(successors.get(node, ()))
    return sorted(out)


def joined(edges, label, earlier, later):
    """type: edges (a, b), label(node) -> (kind, group, item), the kinds `earlier` and `later` -> for each edge from an
    `earlier` node to a `later` node of the same group, (later's item, earlier's item): a COPY's landing to a SEND of
    its occurrence (torch_mesh/compiler.py's forwards), by occurrence"""
    out = {}
    for a, b in edges:
        x, y = label(a), label(b)
        if x[0] == earlier and y[0] == later and x[1] == y[1]:
            out.setdefault(x[1], []).append((y[2], x[2]))
    return out


def contract(edges, label):
    """type: edges (a, b), label(node) -> the edges between labels, each pair once, none from a label to itself (a run's
    events merged into their channel's node: torch_mesh/compiler.py)"""
    return sorted({(label(a), label(b)) for a, b in edges if label(a) != label(b)})


def absorbed(nodes, users, joins):
    """type: nodes, users(n) -> n's consumers, joins(n, u) -> whether u can take n in -> the nodes their one consumer
    absorbs (a pointwise chain fused: torch_mesh/compiler.py)"""
    return {n for n in nodes if len(users(n)) == 1 and joins(n, users(n)[0])}


def serialized(at, node, place):
    """type: at, node -> its place in one order; a node to place at `place` -> the edges putting it after every node
    before that place and before every node after it"""
    return [(other, node) for other, where in at.items() if where < place] + \
           [(node, other) for other, where in at.items() if where > place]


def held_before(held, members, after, target):
    """type: held items in order, members(item) -> its nodes, after (node -> the nodes that follow it), a target node ->
    the held items that must come before `target`: those a member of which `target` follows, and those a member of which
    one of theirs follows, transitively, in `held`'s order (torch_mesh/compiler.py: an observation goes just before the
    first node that follows it, each after the held ones it follows)"""
    def reaches(item, nodes):
        return any(x in after.get(m, ()) for m in members(item) for x in nodes)
    needed = [k for k in held if reaches(k, [target])]
    grew = True
    while grew:
        grew = False
        for k in held:
            if k not in needed and reaches(k, [x for n in needed for x in members(n)]):
                needed.append(k)
                grew = True
    return [k for k in held if k in needed]


def backward(edges, order):
    """type: edges (a, b), order (nodes as emitted, a node repeated for each of its parts) -> the edges whose head is first
    emitted before their tail is last: a run's order its realize refuses (torch_mesh/compiler.py names them)"""
    first, last = {}, {}
    for k, node in enumerate(order):
        first.setdefault(node, k)
        last[node] = k
    return [(a, b) for a, b in edges if a in last and b in first and first[b] < last[a]]


def carried(history, accesses):
    """type: history [(end, writes, lo, hi)], accesses [(begin, writes, lo, hi)] -> the edges (end, begin) from each
    earlier access to each later one over overlapping bytes [lo, hi) where either writes: a run's chunks ordered by the
    bytes they share, where every sink of one chunk before every source of the next held a chunk's receives until its
    last SEND had completed (torch_mesh/compiler.py; metal-microbench docs/facts.md, the LSTM's chunk boundaries)"""
    return sorted({(end, begin) for end, w0, lo0, hi0 in history for begin, w1, lo1, hi1 in accesses
                   if (w0 or w1) and lo0 < hi1 and lo1 < hi0 and end != begin})


def largest(items):
    """type: items [(node, words [(word, count)])] -> {node: the node it is implied by}: of the nodes that each watch one
    word, every one but the largest count's for that word, mapped to it (a count that only rises: a run's observations
    of a channel's `sent` or `landed`, torch_mesh/compiler.py)"""
    best = {}
    for node, words in items:
        if len(words) == 1:
            word, count = words[0]
            if word not in best or count > best[word][1]:
                best[word] = (node, count)
    return {node: best[words[0][0]][0] for node, words in items if len(words) == 1 and best[words[0][0]][0] != node}
