"""The graph module (rubric C9): the graph algorithms mesh's Python code runs, with their consumers as operands.

A caller supplies its vertices and a neighbours function. Nothing here knows what a vertex or an edge stands for.
"""
import collections
import math


def walk(starts, neighbours):
    """The vertices reachable from ``starts``, themselves included, in the order a breadth-first walk first reaches
    them: ``starts`` in their order, then the vertices ``neighbours(v)`` gives, in its order, for each v so reached.
    Each vertex is visited once, so a graph with cycles (a fabric's cables) ends."""
    order, seen, queue = [], set(), collections.deque(starts)
    while queue:
        v = queue.popleft()
        if v not in seen:
            seen.add(v)
            order.append(v)
            queue.extend(neighbours(v))
    return order


def reach(starts, neighbours):
    """The set of vertices reachable from ``starts``, themselves included: walk's vertices, unordered."""
    return set(walk(starts, neighbours))


def closure(order, neighbours):
    """The transitive closure of an acyclic graph: for each vertex v of ``order``, the set of vertices reachable from v
    by one or more steps of ``neighbours``, each of whose vertices precedes v in ``order`` (so ``order`` is a
    topological order of the graph reversed).  Goralčíková and Koubek's reduct-and-closure: the vertices are taken in
    that order and each vertex's neighbours nearest first, a neighbour not yet in the vertex's set adding itself and its
    own set, made before, and a neighbour already in it adding nothing (A. Goralčíková and V. Koubek, A reduct-and-
    closure algorithm for graphs, Mathematical Foundations of Computer Science 1979, LNCS 74; K. Simon, An improved
    algorithm for transitive closure on acyclic digraphs, Theoretical Computer Science 58, 1988).
    (design/aot-compilation-change-request.md A10, "Its graph operations are the graph module's": "The order's
    transitive closure".)"""
    place, reached = {}, {}
    for v in order:
        place[v], reached[v] = len(place), set()
        for u in sorted(set(neighbours(v)), key=place.__getitem__, reverse=True):
            if u not in reached[v]:
                reached[v] |= {u} | reached[u]
    return reached


def propagate(order, neighbours, forward, backward):
    """Propagation to a fixed point along an acyclic graph's edges, in rounds over ``order`` (a topological order): a
    forward pass over ``order`` calls forward(v, e) for each edge e of ``neighbours(v)`` (v's edges from the vertices
    before it), then a backward pass over its reverse calls backward(v, e) for each, until a whole round changes
    nothing, each call returning whether it changed anything.  The round-robin iterative algorithm of global data flow
    analysis (G. A. Kildall, A unified approach to global program optimization, POPL 1973; J. B. Kam and J. D. Ullman,
    Global data flow analysis and iterative algorithms, J. ACM 23(1), 1976), with GSPMD's alternation of directions: it
    "runs the propagation pass on the entire graph in multiple iterations, and alternates between forward propagation
    (input to output) and backward propagation (output to input)" (Y. Xu et al., GSPMD, arXiv 2105.04663, 2021, section
    3.5).  It ends where every call changes its values only toward a finite bound, as GSPMD's refinement does ("This
    property guarantees the pass could reach a fixed point after finite iterations")."""
    changed = True
    while changed:
        changed = False
        for visit, vertices in ((forward, order), (backward, order[::-1])):
            for v in vertices:
                for e in neighbours(v):
                    changed |= visit(v, e)


def paths(vertices, neighbours):
    """The paths of a linear forest, a forest whose components are paths (F. Harary, Graph Theory, 1969): a graph in
    which ``neighbours(v)`` gives each vertex at most one vertex and each vertex is given by at most one.  From each of
    ``vertices``, in their order, that has a neighbour and is no vertex's neighbour, the walk along ``neighbours``
    (walk), which visits its path's vertices in the path's order.  (design/aot-compilation-change-request.md A10: "the
    walk of commuting chains".)"""
    entered = {u for v in vertices for u in neighbours(v)}
    return [walk([v], neighbours) for v in vertices if v not in entered and list(neighbours(v))]


def matching(left, neighbours):
    """A maximum matching of the bipartite graph with an edge from each vertex a of ``left`` to each vertex b that
    ``neighbours(a)`` gives, b on the graph's other side even where it is also of ``left``: {a: b} for each matched a.
    Hopcroft and Karp's phases: a breadth-first search layers the graph from the unmatched vertices of ``left`` to the
    length of the shortest augmenting paths, then depth-first searches augment along vertex-disjoint paths of that
    length, until none is left (J. E. Hopcroft and R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite
    graphs, SIAM J. Comput. 2(4), 1973)."""
    nil = object()
    mate, partner, depth = {a: nil for a in left}, {}, {}

    def layer():
        queue = [a for a in left if mate[a] is nil]
        for a in left:
            depth[a] = 0 if mate[a] is nil else math.inf
        depth[nil] = math.inf
        for a in queue:
            if depth[a] < depth[nil]:
                for b in neighbours(a):
                    c = partner.get(b, nil)
                    if depth[c] == math.inf:
                        depth[c] = depth[a] + 1
                        if c is not nil:
                            queue.append(c)
        return depth[nil] != math.inf

    def augment(a):
        if a is nil:
            return True
        for b in neighbours(a):
            c = partner.get(b, nil)
            if depth[c] == depth[a] + 1 and augment(c):
                mate[a], partner[b] = b, a
                return True
        depth[a] = math.inf
        return False
    while layer():
        for a in left:
            if mate[a] is nil:
                augment(a)
    return {a: b for a, b in mate.items() if b is not nil}


def width(vertices, neighbours):
    """The width of a partial order on ``vertices``, the most of them no two of which are ordered, ``neighbours(a)``
    giving the vertices ordered after a (the order whole, as closure gives it).  By Dilworth's theorem it is the fewest
    chains covering them (R. P. Dilworth, A decomposition theorem for partially ordered sets, Annals of Mathematics 51,
    1950), their number less a maximum matching of the bipartite graph with an edge from a to b for each a ordered
    before b (D. R. Fulkerson, Note on Dilworth's decomposition theorem for partially ordered sets, Proc. AMS 7, 1956),
    matched by Hopcroft and Karp's phases (matching).  (design/aot-compilation-change-request.md A10: "the width by
    Fulkerson's matching in Hopcroft and Karp's phases".)"""
    return len(vertices) - len(matching(vertices, neighbours))
