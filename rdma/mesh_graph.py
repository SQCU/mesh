"""Ordered graph slicing shared by tensor frontends, during preparation only.

Frontends supply value identities, reads, writes, result roots and effects.
Views name their shared storage root; regions describe the positions each value
reads or writes there. Selection preserves the given order. Cuts introduce no
operation or dependency. None of these objects reaches numerical execution.
"""
import itertools
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


def held_before(held, members, after, targets):
    """type: held items in order, members(item) -> its nodes, after (node -> the nodes that follow it), target nodes ->
    the held items that must come before the first of `targets`: those a member of which a target follows, and those a
    member of which one of theirs follows, transitively, in `held`'s order (torch_mesh/compiler.py: an observation goes
    just before the first node that follows it, each after the held ones it follows; a stretch of folds, consecutive in
    the emitted order, takes every one's observations before its first, so the folds meet in one dispatch).
    Changed from one target: a round's folds each took their own observations just before them (observation, fold,
    observation, fold), so no two folds were adjacent and each was a dispatch of its own (metal-microbench
    docs/facts.md, "The all-reduce's all-gather lands in pages of its own": 41-45 a rank). What must stay: every held
    item a target follows, and every held item one of those follows, comes before the targets, in `held`'s order."""
    def reaches(item, nodes):
        return any(x in after.get(m, ()) for m in members(item) for x in nodes)
    needed = [k for k in held if reaches(k, targets)]
    grew = True
    while grew:
        grew = False
        for k in held:
            if k not in needed and reaches(k, [x for n in needed for x in members(n)]):
                needed.append(k)
                grew = True
    return [k for k in held if k in needed]


def stretches(order, member):
    """type: order [node], member(node) -> whether it is one of a kind -> {first: [node]}: each maximal stretch of
    members consecutive in `order`, by its first (torch_mesh/compiler.py: a stretch of folds in the emitted order takes
    every one's held observations before its first, held_before, so its folds meet in one dispatch)"""
    out = {}
    for of_members, stretch in itertools.groupby(order, member):
        stretch = list(stretch)
        if of_members:
            out[stretch[0]] = stretch
    return out


def backward(edges, order):
    """type: edges (a, b), order (nodes as emitted, a node repeated for each of its parts) -> the edges whose head is first
    emitted before their tail is last: a run's order its realize refuses (torch_mesh/compiler.py names them)"""
    first, last = {}, {}
    for k, node in enumerate(order):
        first.setdefault(node, k)
        last[node] = k
    return [(a, b) for a, b in edges if a in last and b in first and first[b] < last[a]]


def carried(history, accesses, quantum=16384):
    """type: history [(end, writes, lo, hi)], accesses [(begin, writes, lo, hi)] -> the edges (end, begin) from each
    earlier access to each later one over overlapping bytes [lo, hi) where either writes: a run's chunks ordered by the
    bytes they share, where every sink of one chunk before every source of the next held a chunk's receives until its
    last SEND had completed (torch_mesh/compiler.py; metal-microbench docs/facts.md, the LSTM's chunk boundaries). The
    earlier accesses found through buckets of `quantum` bytes (each in every bucket it touches), not every pair: every
    pair was 1.95 of a 256-step LSTM's 3.6 s preparation, its history every chunk before (lstm-2026-11-03 profile-2)"""
    buckets = {}
    for k, (_, _, lo, hi) in enumerate(history):
        for q in range(lo // quantum, (hi - 1) // quantum + 1):
            buckets.setdefault(q, []).append(k)
    out = set()
    for begin, w1, lo1, hi1 in accesses:
        seen = set()
        for q in range(lo1 // quantum, (hi1 - 1) // quantum + 1):
            for k in buckets.get(q, ()):
                if k not in seen:
                    seen.add(k)
                    end, w0, lo0, hi0 = history[k]
                    if (w0 or w1) and lo0 < hi1 and lo1 < hi0 and end != begin:
                        out.add((end, begin))
    return sorted(out)


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


def unfollowed(order, waits, copies, edges):
    """type: order [node] (as emitted, one part), waits {node}, copies {node}, edges (a, b) -> [(wait, copy)] each copy
    after a wait it does not follow: on an M5 Ultra a copy runs only once every earlier dispatch of its encoder has ended
    (metal-microbench visible.m behind), so in one part each pair holds the copy behind the wait (torch_mesh/compiler.py,
    a run with no part ends)"""
    before = {}
    for a, b in edges:
        before.setdefault(b, set()).add(a)
    ancestors = {}

    def of(node):
        if node not in ancestors:
            seen, stack = set(), list(before.get(node, ()))
            while stack:
                n = stack.pop()
                if n not in seen:
                    seen.add(n)
                    stack.extend(before.get(n, ()))
            ancestors[node] = seen
        return ancestors[node]
    out, seen_waits = [], []
    for node in order:
        if node in copies:
            out += [(w, node) for w in seen_waits if w not in of(node) and w != node]
        if node in waits:
            seen_waits.append(node)
    return out


def unordered(order, edges, accesses):
    """type: order [event] (a run's, as emitted, each once), edges [(from, to)] forward in it, accesses [(befores, afters,
    writes, lo, hi)] -> the index pairs (j, i) of accesses over overlapping bytes [lo, hi), one a write, that no path of
    edges orders either way. An access starts after any event at or before one of its `befores` and ends before any at
    or after one of its `afters` (an event with no node of its own stands for the nodes it passes through). In one part
    the edges are the run's only order (its barriers), so each pair is a hazard the run does not hold (torch_mesh
    compiler.py MESH_RUN_ORDER_CHECK, the one-part reading; metal-microbench docs/facts.md)"""
    position = {e: k for k, e in enumerate(order)}
    into = {}
    for a, b in edges:
        into.setdefault(b, []).append(a)
    reach = {}  # each event's set of events at or before it, as bits by position
    for e in order:
        bits = 1 << position[e]
        for a in into.get(e, ()):
            bits |= reach.get(a, 0)
        reach[e] = bits

    def precedes(afters, befores):
        return any(a in position and b in reach and reach[b] >> position[a] & 1 for a in afters for b in befores)
    active, found = [], []
    for i in sorted(range(len(accesses)), key=lambda k: accesses[k][3]):
        befores, afters, writes, lo, hi = accesses[i]
        active = [j for j in active if accesses[j][4] > lo]
        for j in active:
            if (writes or accesses[j][2]) and not precedes(accesses[j][1], befores) and not precedes(afters, accesses[j][0]):
                found.append((j, i))
        active.append(i)
    return found


def labelling(names, adjacent, labels):
    """type: names [member], adjacent {member: {label}} (the labels each member's edges reach), labels [label] ->
    {member: label}: an assignment of distinct labels under which no member reaches its own and each edge is reached
    from both its ends (a member's streams toward peer nodes, named by node identity: metal-microbench
    tools/mesh/tick_words.py); {} where none is. A search over the assignments: the members are few"""
    import itertools
    for chosen in itertools.permutations(labels, len(names)):
        label = dict(zip(names, chosen))
        if all(label[n] not in adjacent[n] for n in names) and \
                all((label[m] in adjacent[n]) == (label[n] in adjacent[m]) for n in names for m in names if n != m):
            return label
    return {}


def waits_late(order, edges, waits, fences):
    """type: order [event] (topological over edges), edges [(from, to)], waits {event}, fences {event} -> the order with,
    in each segment between fences (a fence a segment's last), the events that follow none of its waits (over the edges,
    within the segment) first and those that do after them, each kind in its given order. A Metal 4 barrier waits for
    every dispatch before it, so a node emitted after an observation behind a barrier waits for the observation though it
    does not follow it: on the LSTM's rank 0 the step counter's update and the next tick's input selection went after
    the tick's landings with the output's copy, three barriers more on the relay cycle that sets the period
    (metal-microbench docs/facts.md, the tick on the Mini). Every edge stays forward: no event that follows a wait moves
    before one that does not, and segments keep their order (torch_mesh compiler.py, the emitted order)"""
    after = {}
    for a, b in edges:
        after.setdefault(a, []).append(b)
    out, segment = [], []

    def close():
        inside = set(segment)
        late, stack = set(), [w for w in segment if w in waits]
        while stack:
            for b in after.get(stack.pop(), ()):
                if b in inside and b not in late:
                    late.add(b)
                    stack.append(b)
        out.extend(e for e in segment if e not in late)
        out.extend(e for e in segment if e in late)
        segment.clear()
    for e in order:
        segment.append(e)
        if e in fences:
            close()
    close()
    return out


def superseded(history, newer):
    """type: history [(end, writes, lo, hi)], newer [(node, writes, lo, hi)] (accesses carried() has just ordered after
    `history`) -> history without the accesses whose bytes [lo, hi) the newer writes cover (their union): such an
    access precedes a write over its bytes (carried's edge), which precedes any later access over them that conflicts
    with it, so it orders nothing more. The history of every earlier chunk instance, on the same pages, made carried
    quadratic in a run's chunks (a 512-step LSTM's preparation 7.8 s, lstm-2026-11-03 profile-3)"""
    spans = sorted((lo, hi) for _, w, lo, hi in newer if w and hi > lo)
    merged = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    starts = [lo for lo, _ in merged]
    import bisect

    def covered(lo, hi):
        k = bisect.bisect_right(starts, lo) - 1
        return k >= 0 and merged[k][1] >= hi
    return [h for h in history if not covered(h[2], h[3])]


def versions(moves):
    """type: page moves [(source, destination, kind, link, first, pages)] (mesh-plan.h struct mesh_page_move: kind
    'transfer', 'forward' or 'reduce', link a move's index or None, operand pages [first, first + pages)) -> (send,
    land, folds): send[k] and land[k], for each page of move k, the version its SEND reads at the source and the one
    its landing writes at the destination, (version, index) each; folds {(rank, target): {operand page: (index,
    inputs)}}, each target page's one fold, `index` the target's page it writes and `inputs` the (version, index) pages
    it adds, the rank's own first.
    Renaming: a REDUCE lands in pages of its own, ('landing', k), and a sum carried on is a version of its own,
    ('sum', L) by the pages of L, the move carrying it, so no landing or fold writes a page a SEND reads and each
    version page has one writer. 'own' (the rank's value before the collective) and 'result' (its result) go by operand
    page, a tuple's version by its own pages. A TRANSFER or a FORWARD lands in 'result'. Every REDUCE landing on a
    rank's page into one target (its link's sum, or 'result' where it has none) is added with the rank's own value in
    one fold (mesh-plan.h MESH_MOVE_REDUCE: one fold over all its inputs, never a fold a landing), so a sum is its last
    arrival plus one fold, whatever order the landings arrive in. A FORWARD sends what its link landed; a REDUCE the sum
    where a fold writes that page of it, else the rank's own; a TRANSFER the result where something lands it at the
    source, else the rank's own. No order is walked: each read names its version and each version page its one writer,
    so every rank states every rank's alike (page_schedule's operand).
    Changed from versions walked in a move order (move_order, a Kahn order of dependences derived again from links and
    page overlap) with a fold a REDUCE landing, each later fold into a target reading and writing it in place: k
    dispatches and about 3k page streams for k children, a chain fixed by move order, so a child first in that order
    that landed last held every fold behind it. What must stay: renaming, and one fold a target page."""
    folds, written = {}, set()
    for k, (_, destination, kind, link, first, pages) in enumerate(moves):
        if kind != 'reduce':
            written.update((destination, 'result', p) for p in range(first, first + pages))
            continue
        target, base = (('sum', link), moves[link][4]) if link is not None else ('result', 0)
        into = folds.setdefault((destination, target), {})
        for i in range(pages):
            into.setdefault(first + i, (first + i - base, [('own', first + i)]))[1].append((('landing', k), i))
            written.add((destination, target, first + i - base))
    send, land = [], []
    for k, (source, _, kind, link, first, pages) in enumerate(moves):
        span = range(first, first + pages)
        if kind == 'forward':
            landed = moves[link]
            send.append([(('landing', link), p - landed[4]) if landed[2] == 'reduce' else ('result', p) for p in span])
        elif kind == 'reduce':
            send.append([(('sum', k), i) if (source, ('sum', k), i) in written else ('own', first + i)
                         for i in range(pages)])
        else:
            send.append([('result', p) if (source, 'result', p) in written else ('own', p) for p in span])
        land.append([(('landing', k), i) for i in range(pages)] if kind == 'reduce' else [('result', p) for p in span])
    return send, land, folds


def page_schedule(moves, flow, cables, hop, fold):
    """type: page moves (versions'), their versions (versions), the fabric's cables [(node, node, rate)] in its order
    (mesh-fabric.h struct mesh_cable, rate in pages a second), hop (seconds from a page's wire time's end to its
    landing's presence), fold (a fold's own seconds, its page streams a second) -> (units, grains): units[k][i] for page
    i of move k, (cable, unit, start, arrival): the cable that carries it, its place among the units that cable carries
    toward the move's destination, and the seconds from the collective's start at which its SEND starts and its landing
    is present; grains [(rank, start, end, [(target, operand page)])] in start order, a grain one fold at a rank (one
    dispatch) over every target page whose landings had all arrived when the rank's GPU took it.
    A list schedule at page granularity, computed alike at every rank from the same moves and cables, so a cable's
    units are in one order at both ends. Its resources are each cable one way and each rank's GPU; its tasks a page's
    SEND (1 / rate on one of its pair's cables, its landing present `hop` after) and a grain (fold[0] and its pages'
    streams over fold[1], a page's inputs read and its sum written). No resource idles while a task is ready for it
    (R. L. Graham, "Bounds on multiprocessing timing anomalies", SIAM J. Appl. Math. 17(2), 1969). A page is ready once
    the version it reads is: at once for the rank's own, at its landing's arrival for a relay, at its grain's end for a
    sum. A cable that frees, or is free when a page of its pair becomes ready, takes of the pair's ready pages the first,
    highest level first (the longest path to the end: T. L. Adam, K. M. Chandy and J. R. Dickson, "A comparison of list
    schedules for parallel processing systems", CACM 17(12), 1974; ties by move, then page), that would finish soonest
    on it of all the pair's cables (earliest finish, mesh-graph.h mesh_graph_fill's rule for uniform machines), so each
    cable carries a share by its rate and no ready page waits behind a move that is not ready: the pages of moves
    interleave on a cable as they become ready. A rank's GPU, once free, folds every page whose inputs have landed, so a
    grain's length follows from the fold's cost against the landings' rate, no size chosen: the first a page, each
    later one what landed while the last folded, and the first summed page leaves while later pages still land. A cable
    of no rate carries nothing. ValueError where a page never becomes ready (the moves' dependences hold a cycle: the
    planner's defect, at compile time).
    Changed from a move order (move_order: Kahn's, least index first) and a spread of a pair's pages over its cables by
    page k mod n (spread): move-major units, so a move ready late took a cable's earlier units and held ready pages
    behind it, and a relayed FORWARD paced its whole outbound cable at its inbound rate; unequal parallel cables finishing
    at n times the slowest one's share; and a REDUCE's fold one event over all its move's pages, so each hop was store
    and forward over the whole move. What must stay: one schedule alike at every rank, a cable's units in one order."""
    import heapq
    import math
    send, land, folds = flow
    seconds = [1.0 / rate if rate > 0 else math.inf for _, _, rate in cables]
    pair = {}
    for c, (a, b, rate) in enumerate(cables):
        if rate > 0:
            pair.setdefault((a, b), []).append(c)
            pair.setdefault((b, a), []).append(c)
    # each version page's one writer at its rank (a page's landing, a target page's fold), and what reads it there
    writer, readers, waiting = {}, {}, {}
    for k, m in enumerate(moves):
        for i, version in enumerate(land[k]):
            writer[m[1], version] = ('unit', k, i)
    for (rank, target), pages in folds.items():
        for p, (index, inputs) in pages.items():
            writer[rank, (target, index)] = ('fold', rank, target, p)
            waiting['fold', rank, target, p] = len(inputs) - 1
            for version in inputs[1:]:
                readers.setdefault(writer[rank, version], []).append(('fold', rank, target, p))
    for k, m in enumerate(moves):
        for i, version in enumerate(send[k]):
            if (m[0], version) in writer:
                readers.setdefault(writer[m[0], version], []).append(('unit', k, i))
    level = {}

    def height(node):
        # the longest path from a task's start to the end, its own time among it
        if node not in level:
            if node[0] == 'unit':
                own = min(seconds[c] for c in pair[moves[node[1]][0], moves[node[1]][1]]) + hop
            else:
                own = fold[0] + (len(folds[node[1], node[2]][node[3]][1]) + 1) / fold[1]
            level[node] = own + max((height(r) for r in readers.get(node, ())), default=0.0)
        return level[node]
    units = [[None] * m[5] for m in moves]
    grains, free, taken, pool, gpu, foldable, queue, sequence = [], {}, {}, {}, {}, {}, [], itertools.count()

    def at(t, kind, x):
        heapq.heappush(queue, (t, next(sequence), kind, x))

    def take(ends, t):
        ready = pool.get(ends)
        for c in pair[ends]:
            if not ready or free.get((c, ends[1]), 0.0) > t:
                continue
            finish = {j: max(t, free.get((j, ends[1]), 0.0)) + seconds[j] for j in pair[ends]}
            passed, chosen = [], None
            while ready:
                page = heapq.heappop(ready)
                j = min(pair[ends], key=lambda j: (finish[j], j != c, j))
                if j == c:
                    chosen = page
                    break
                finish[j] += seconds[j]
                passed.append(page)
            for page in passed:
                heapq.heappush(ready, page)
            if chosen is not None:
                _, k, i = chosen
                unit = taken.get((c, ends[1]), 0)
                taken[c, ends[1]], free[c, ends[1]] = unit + 1, t + seconds[c]
                units[k][i] = (c, unit, t, t + seconds[c] + hop)
                at(t + seconds[c], 'free', ends)
                at(t + seconds[c] + hop, 'landed', ('unit', k, i))

    def take_fold(rank, t):
        if gpu.get(rank, 0.0) > t or not foldable.get(rank):
            return
        grain = foldable.pop(rank)
        end = t + fold[0] + sum(len(folds[rank, target][p][1]) + 1 for target, p in grain) / fold[1]
        gpu[rank] = end
        grains.append((rank, t, end, grain))
        at(end, 'folded', len(grains) - 1)
    for k, m in enumerate(moves):
        for i, version in enumerate(send[k]):
            if (m[0], version) not in writer:
                at(0.0, 'ready', ('unit', k, i))
    while queue:
        t, _, kind, x = heapq.heappop(queue)
        if kind == 'ready':
            ends = moves[x[1]][0], moves[x[1]][1]
            heapq.heappush(pool.setdefault(ends, []), (-height(x), x[1], x[2]))
            take(ends, t)
        elif kind == 'free':
            take(x, t)
        elif kind == 'landed':
            for node in readers.get(x, ()):
                if node[0] == 'unit':
                    at(t, 'ready', node)
                    continue
                waiting[node] -= 1
                if not waiting[node]:
                    foldable.setdefault(node[1], []).append(node[2:])
                    take_fold(node[1], t)
        else:
            rank, _, _, grain = grains[x]
            for target, p in grain:
                for node in readers.get(('fold', rank, target, p), ()):
                    at(t, 'ready', node)
            take_fold(rank, t)
    if any(u is None for move in units for u in move):
        raise ValueError('page moves whose dependences hold a cycle (mesh-plan.h mesh_page_moves)')
    return units, grains


def consecutive(rows):
    """type: [(same, ascending)] a pair of tuples a row -> [(first, end)]: the maximal runs of rows [first, end) in which
    each row's `same` is its predecessor's and each place of its `ascending` its predecessor's plus one (a grain's fold
    pages contiguous in every buffer it binds, a piece of its dispatch each: torch_mesh/compiler.py)"""
    out = []
    for k, (same, ascending) in enumerate(rows):
        if out and rows[k - 1][0] == same and all(b == a + 1 for a, b in zip(rows[k - 1][1], ascending)):
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1])
    return [tuple(r) for r in out]


def arborescences(nodes, root, capacity, depth=None):
    """every spanning arborescence of range(nodes) from `root` over the pairs capacity[u][v] > 0 joins (u v's parent),
    as {v: u}, each within `depth` edges of the root where it is given; by enumeration of each node's parent, so for
    a collective of a few nodes (5: at most 4^4 assignments a root)"""
    others = [v for v in range(nodes) if v != root]
    for chosen in itertools.product(range(nodes), repeat=len(others)):
        parent = dict(zip(others, chosen))
        if any(u == v or capacity[u][v] <= 0 for v, u in parent.items()):
            continue
        deepest = 0
        for v in others:
            walked, u = 0, v
            while u != root and walked <= nodes:
                u, walked = parent[u], walked + 1
            deepest = max(deepest, walked)
        if deepest <= (nodes - 1 if depth is None else depth):
            yield parent


def trees_bound(nodes, capacity, depth=None):
    """the most every root can broadcast at once over spanning arborescences (an all-gather's per-root rate, in
    capacity's units), the trees within `depth` where given: the linear program over every arborescence's rate,
    each directed pair's summed load at most its capacity (scipy's HiGHS, the workstation's analysis venv). Returns
    (rate, [(root, parent, rate)] of the trees it uses). Against the planner's packing (mesh-graph.c
    mesh_graph_pack): the bound a packing's trees can reach on the same cables."""
    import numpy
    from scipy.optimize import linprog
    trees = [(r, t) for r in range(nodes) for t in arborescences(nodes, r, capacity, depth)]
    pairs = [(u, v) for u in range(nodes) for v in range(nodes) if capacity[u][v] > 0]
    count = len(trees)
    cost = numpy.zeros(count + 1)
    cost[-1] = -1
    loads = numpy.array([[1.0 if t.get(v) == u else 0.0 for (_, t) in trees] + [0.0] for (u, v) in pairs])
    shares = numpy.array([[1.0 if r == root else 0.0 for (r, _) in trees] + [-1.0] for root in range(nodes)])
    out = linprog(cost, A_ub=loads, b_ub=[capacity[u][v] for (u, v) in pairs], A_eq=shares, b_eq=numpy.zeros(nodes),
                  bounds=[(0, None)] * (count + 1), method='highs')
    return out.x[-1], [(r, t, x) for (r, t), x in zip(trees, out.x[:-1]) if x > 1e-9]


def release_gaps(releases, arrivals):
    """one rank's ticks' critical path: each release's gap from the last arrival of the latest tick whose arrivals all
    precede it (a tick's last arrival the latest of each stream's arrivals[s][j]), so a release is paired with the
    data it followed whichever counts first (the Mini's first landing came a tick before its first release, which
    paired by index read every gap a period long, 55-79 us: metal-microbench balance-2026-11-03 words-3)"""
    lasts = sorted(max(a[j] for a in arrivals) for j in range(min(map(len, arrivals), default=0))
                   if arrivals and all(a[j] is not None for a in arrivals))
    out, k = [], 0
    for r in releases[1:]:
        while k < len(lasts) and lasts[k] < r:
            k += 1
        if k:
            out.append(r - lasts[k - 1])
    return out
