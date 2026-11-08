# design/rubrics.md C9
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


# A. Goralčíková and V. Koubek, LNCS 74, 1979; K. Simon, TCS 58, 1988; design/aot-compilation-change-request.md A10
def closure(order, neighbours):
    place, reached = {}, {}
    for v in order:
        place[v], reached[v] = len(place), set()
        for u in sorted(set(neighbours(v)), key=place.__getitem__, reverse=True):
            if u not in reached[v]:
                reached[v] |= {u} | reached[u]
    return reached


# G. A. Kildall, POPL 1973; J. B. Kam and J. D. Ullman, J. ACM 1976; GSPMD, Xu et al., arXiv 2105.04663, section 3.5
def propagate(order, neighbours, forward, backward):
    changed = True
    while changed:
        changed = False
        for visit, vertices in ((forward, order), (backward, order[::-1])):
            for v in vertices:
                for e in neighbours(v):
                    changed |= visit(v, e)


# G. A. Kildall, POPL 1973; J. B. Kam and J. D. Ullman, J. ACM 1976
def forward(order, neighbours, transfer):
    values = {}
    for v in order:
        values[v] = transfer(v, [values[u] for u in neighbours(v)])
    return values


# M. Poletto and V. Sarkar, Linear scan register allocation, ACM TOPLAS 21(5), 1999
def ends(order, uses):
    last = {}
    for v in order:
        for u in uses(v):
            last[u] = v
    return last


# F. Harary, Graph Theory, 1969; design/aot-compilation-change-request.md A10
def paths(vertices, neighbours):
    entered = {u for v in vertices for u in neighbours(v)}
    return [walk([v], neighbours) for v in vertices if v not in entered and list(neighbours(v))]


# J. E. Hopcroft and R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite graphs, SIAM J. Comput. 2, 1973
def matching(left, neighbours):
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


# design/joint-compilation.md §3, row 2
def width(vertices, neighbours):
    return len(vertices) - len(matching(vertices, neighbours))
