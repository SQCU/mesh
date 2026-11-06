import collections
import math


def walk(starts, neighbours):
    order, seen, queue = [], set(), collections.deque(starts)
    while queue:
        v = queue.popleft()
        if v not in seen:
            seen.add(v)
            order.append(v)
            queue.extend(neighbours(v))
    return order


def reach(starts, neighbours):
    return set(walk(starts, neighbours))


def closure(order, neighbours):
    """A. Goralčíková and V. Koubek, A reduct-and-closure algorithm for graphs, Mathematical Foundations of
    Computer Science 1979, LNCS 74; K. Simon, An improved algorithm for transitive closure on acyclic digraphs,
    Theoretical Computer Science 58, 1988."""
    place, reached = {}, {}
    for v in order:
        place[v], reached[v] = len(place), set()
        for u in sorted(set(neighbours(v)), key=place.__getitem__, reverse=True):
            if u not in reached[v]:
                reached[v] |= {u} | reached[u]
    return reached


def propagate(order, neighbours, forward, backward):
    """G. A. Kildall, A unified approach to global program optimization, POPL 1973; J. B. Kam and J. D. Ullman,
    Global data flow analysis and iterative algorithms, J. ACM 23(1), 1976; Y. Xu et al., GSPMD, arXiv 2105.04663,
    2021, section 3.5: it "runs the propagation pass on the entire graph in multiple iterations, and alternates between
    forward propagation (input to output) and backward propagation (output to input)"; "This property guarantees the
    pass could reach a fixed point after finite iterations"."""
    changed = True
    while changed:
        changed = False
        for visit, vertices in ((forward, order), (backward, order[::-1])):
            for v in vertices:
                for e in neighbours(v):
                    changed |= visit(v, e)


def forward(order, neighbours, transfer):
    """G. A. Kildall, A unified approach to global program optimization, POPL 1973; J. B. Kam and J. D. Ullman,
    Global data flow analysis and iterative algorithms, J. ACM 23(1), 1976."""
    values = {}
    for v in order:
        values[v] = transfer(v, [values[u] for u in neighbours(v)])
    return values


def ends(order, uses):
    """M. Poletto and V. Sarkar, Linear scan register allocation, ACM TOPLAS 21(5), 1999."""
    last = {}
    for v in order:
        for u in uses(v):
            last[u] = v
    return last


def paths(vertices, neighbours):
    """F. Harary, Graph Theory, 1969: a linear forest."""
    entered = {u for v in vertices for u in neighbours(v)}
    return [walk([v], neighbours) for v in vertices if v not in entered and list(neighbours(v))]


def matching(left, neighbours):
    """J. E. Hopcroft and R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite graphs, SIAM J.
    Comput. 2(4), 1973."""
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
    """R. P. Dilworth, A decomposition theorem for partially ordered sets, Annals of Mathematics 51, 1950; D. R.
    Fulkerson, Note on Dilworth's decomposition theorem for partially ordered sets, Proc. AMS 7, 1956."""
    return len(vertices) - len(matching(vertices, neighbours))
