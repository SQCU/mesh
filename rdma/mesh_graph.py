# design/rubrics.md C9
import collections.abc
import ctypes as C
import errno
import math
import os
import weakref


# mesh-graph.h struct mesh_graph_search and struct mesh_graph_closure: the owned results
class Searched(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('count', C.c_uint32), ('preorder', C.POINTER(C.c_uint32)),
                ('postorder', C.POINTER(C.c_uint32)), ('parent', C.POINTER(C.c_uint32)),
                ('place', C.POINTER(C.c_uint32)), ('entry', C.POINTER(C.c_size_t)), ('end', C.POINTER(C.c_uint32))]


class Closed(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('words', C.c_size_t), ('reached', C.POINTER(C.c_uint64))]


# one owned native result, the graph whose vertices its IDs name and the operands it was made from (`held`), freed with
# its last view
class Owned:
    def __init__(self, made, free, graph, held=()):
        self.made, self.graph, self.held, self.freed = made, graph, held, weakref.finalize(self, free, made)


# a borrowed native array over a span of its indices, each element read through `at`; slices are views of the same
# array and keep its owner
class View(collections.abc.Sequence):
    def __init__(self, owner, items, span, at):
        self.owner, self.items, self.span, self.at = owner, items, span, at

    def __len__(self):
        return len(self.span)

    def __getitem__(self, k):
        span = self.span[k]
        return View(self.owner, self.items, span, self.at) if isinstance(span, range) else self.at(self.items[span])

    def __iter__(self):
        return (self.at(self.items[k]) for k in self.span)


# a search's order of its owned result, its native vertex IDs mapped to the graph's objects
def ordered(owned, ids, span):
    return View(owned, ids, span, owned.graph.vertices.__getitem__)


# the vertices a search reached: membership by its place array
class Reached(collections.abc.Set):
    def __init__(self, owned):
        self.owned, self.searched = owned, owned.made.contents

    def __contains__(self, v):
        k = self.owned.graph.at.get(v)
        return k is not None and self.searched.place[k] != NONE

    def __iter__(self):
        return iter(ordered(self.owned, self.searched.preorder, range(self.searched.count)))

    def __len__(self):
        return self.searched.count

    @classmethod
    def _from_iterable(cls, items):
        return set(items)

    # mesh-graph.h mesh_graph_search_save: its search's arrays as the library wrote them (that allocation freed once
    # copied, or should the copy fail)
    def save(self):
        made, length = C.c_void_p(), C.c_size_t()
        _check(LIB.mesh_graph_search_save(self.owned.made, C.byref(made), C.byref(length)))
        try:
            return C.string_at(made, length.value)
        finally:
            LIB.free(made)

    # mesh-graph.h mesh_graph_search_load: a search of its own made from `data` without searching, its vertex IDs
    # `graph`'s
    @classmethod
    def load(cls, data, graph):
        made = C.POINTER(Searched)()
        _check(LIB.mesh_graph_search_load(C.c_char_p(data), len(data), C.byref(made)))
        return cls(Owned(made, LIB.mesh_graph_search_free, graph))


# one vertex k of the owned closure bitset: its row (the vertices k's rows lead to) or, transposed, its column (the
# vertices whose rows lead to k); a row's columns the graph's vertices, or `columns` (objects, their index)
class Closure(collections.abc.Set):
    def __init__(self, owned, k, transposed, columns=None):
        closed = owned.made.contents
        self.owned, self.bits, self.words, self.k, self.transposed = owned, closed.reached, closed.words, k, transposed
        self.columns = columns or (owned.graph.vertices, owned.graph.at)

    def _bit(self, row, column):
        return bool(self.bits[row * self.words + column // 64] >> (column % 64) & 1)

    def __contains__(self, u):
        j = (self.owned.graph.at if self.transposed else self.columns[1]).get(u)
        return j is not None and (self._bit(j, self.k) if self.transposed else self._bit(self.k, j))

    def __iter__(self):
        if self.transposed:
            vertices = self.owned.graph.vertices
            return (vertices[j] for j in range(len(vertices)) if self._bit(j, self.k))
        return (self.columns[0][64 * w + b] for w in range(self.words) for b in range(64)
                if self.bits[self.k * self.words + w] >> b & 1)

    def __len__(self):
        return sum(1 for _ in self)

    @classmethod
    def _from_iterable(cls, items):
        return set(items)


# the closure by vertex
class Closures(collections.abc.Mapping):
    def __init__(self, owned, transposed, columns=None):
        self.owned, self.transposed, self.columns = owned, transposed, columns

    def __getitem__(self, v):
        return Closure(self.owned, self.owned.graph.at[v], self.transposed, self.columns)

    def __iter__(self):
        return iter(self.owned.graph.vertices)

    def __len__(self):
        return len(self.owned.graph.vertices)

    # mesh-graph.h mesh_graph_closure_save: its closure's arrays as the library wrote them (that allocation freed once
    # copied, or should the copy fail)
    def save(self):
        made, length = C.c_void_p(), C.c_size_t()
        _check(LIB.mesh_graph_closure_save(self.owned.made, C.byref(made), C.byref(length)))
        try:
            return C.string_at(made, length.value)
        finally:
            LIB.free(made)

    # mesh-graph.h mesh_graph_closure_load: a closure of its own made from `data` without closing, its vertex IDs
    # `graph`'s, its rows or (`transposed`) columns and `columns` as saved, holding `held` (its processing order) as
    # ancestors' and labels' results hold theirs
    @classmethod
    def load(cls, data, graph, transposed, columns=None, held=()):
        made = C.POINTER(Closed)()
        _check(LIB.mesh_graph_closure_load(C.c_char_p(data), len(data), C.byref(made)))
        return cls(Owned(made, LIB.mesh_graph_closure_free, graph, held), transposed, columns)


# mesh-graph.h struct mesh_graph_relation_data, mesh_graph_search, mesh_graph_breadth(_masked), mesh_graph_closure,
# mesh_graph_closure_labels, mesh_graph_relation_head, mesh_graph_relation_incoming_head, mesh_graph_relation_contract
# and mesh_graph_relation_quotient: one retained relation and its vertices' objects (vertex k is vertices[k]); a search
# runs in place over its outgoing rows (row_first, row_edge, each edge's head) or, `incoming`, its incoming rows
# (in_first, in_edge, each edge's tail), `active` by vertex the rows followed (None every row), and every result is a
# view of its owned native result, retaining the relation and its operands
class Graph:
    def __init__(self, vertices, relation):
        self.vertices, self.relation = list(vertices), relation
        self.at = {v: k for k, v in enumerate(self.vertices)}
        self.data, self.context = relation.data, C.cast(relation.descriptor, C.c_void_p)

    # the relation dependences_supplied returned, its vertex k the caller's vertices[k]
    @classmethod
    def from_conflicts(cls, vertices, relation):
        return cls(vertices, relation)

    # a source adjacency constructed once as GIVEN conflict records, edge v -> u for each u of neighbours(v), `given` the
    # record's index, no view
    @classmethod
    def adjacency(cls, universe, neighbours):
        vertices = list(dict.fromkeys(universe))
        at = {v: k for k, v in enumerate(vertices)}
        records = [(at[v], at[u]) for v in vertices for u in neighbours(v)]
        conflict = (Conflict * max(len(records), 1))(*[Conflict(a, b, GIVEN, k, 0, SIZE, SIZE)
                                                        for k, (a, b) in enumerate(records)])
        return cls(vertices, Relation.create(len(vertices), 0, None, len(records), conflict))

    # the rows a search follows: outgoing (row_first, mesh_graph_relation_head) or incoming (in_first,
    # mesh_graph_relation_incoming_head, resolved on use)
    def rows(self, incoming=False):
        return (self.data.in_first, C.cast(LIB.mesh_graph_relation_incoming_head, C.c_void_p)) if incoming else (
            self.data.row_first, RELATION_HEAD)

    def _mask(self, active):
        return None if active is None else (C.c_ubyte * max(self.data.vertices, 1))(*[bool(a) for a in active])

    def _search(self, call, roots, *operands, incoming=False):
        start, count = _array(C.c_uint32, [self.at[v] for v in roots])
        made, (first, head) = C.POINTER(Searched)(), self.rows(incoming)
        _check(call(self.data.vertices, first, head, self.context, count, start, *operands, C.byref(made)))
        return Owned(made, LIB.mesh_graph_search_free, self, operands)

    # mesh_graph_breadth_masked (resolved on use) over the `active` rows
    def _breadth(self, starts, incoming, active):
        call = LIB.mesh_graph_breadth_masked
        call.restype, call.argtypes = C.c_int, [C.c_uint32, _Z, C.c_void_p, C.c_void_p, C.c_uint32, _U,
                                                C.POINTER(C.c_ubyte), C.POINTER(C.POINTER(Searched))]
        return self._search(call, starts, self._mask(active), incoming=incoming)

    # T. H. Cormen et al., Introduction to Algorithms, 3rd ed., §22.3
    def search(self, roots, reverse=False, incoming=False):
        return self._search(LIB.mesh_graph_search, roots, int(reverse), incoming=incoming)

    def preorder(self, roots, incoming=False):
        owned = self.search(roots, incoming=incoming)
        return ordered(owned, owned.made.contents.preorder, range(owned.made.contents.count))

    # R. E. Tarjan, SIAM J. Comput. 1(2), 1972: the finishing order of the search from the roots in reverse order over
    # rows in reverse order, the order of a stack each root and row is pushed on
    def postorder(self, roots, incoming=False):
        owned = self.search(list(roots)[::-1], True, incoming)
        return ordered(owned, owned.made.contents.postorder, range(owned.made.contents.count))

    # T. H. Cormen et al., 3rd ed., §22.4: the reverse finishing order of the search from every vertex, a topological
    # order of the rows' edges
    def order(self, incoming=False):
        owned = self.search(self.vertices, incoming=incoming)
        return ordered(owned, owned.made.contents.postorder, range(owned.made.contents.count - 1, -1, -1))

    # T. H. Cormen et al., 3rd ed., §22.2
    def walk(self, starts, incoming=False, active=None):
        owned = self._search(LIB.mesh_graph_breadth, starts, incoming=incoming) if active is None else self._breadth(
            starts, incoming, active)
        return ordered(owned, owned.made.contents.preorder, range(owned.made.contents.count))

    def reach(self, starts, incoming=False, active=None):
        return Reached(self._search(LIB.mesh_graph_breadth, starts, incoming=incoming) if active is None else
                       self._breadth(starts, incoming, active))

    # mesh_graph_closure over the rows (`incoming` as the searches') processed in `order`, a View of one search's native
    # vertex IDs covering every vertex (its postorder over these rows: each head before its row), read in place from its
    # first element by its step: v's set the vertices whose rows lead to v (rows of predecessors: v's own closure row;
    # `successors`: the transposed column)
    def ancestors(self, order, successors=False, incoming=False):
        made, (first, head) = C.POINTER(Closed)(), self.rows(incoming)
        _check(LIB.mesh_graph_closure(self.data.vertices, first, head, self.context, *self._order(order),
                                      C.byref(made)))
        return Closures(Owned(made, LIB.mesh_graph_closure_free, self, (order,)), successors)

    # mesh_graph_closure_labels (resolved on use) over the rows (`incoming`) processed in `order` (as ancestors'): v's
    # set the `classes` of its active rows' heads (`label` by vertex a class or None) and of their sets, an inactive
    # vertex's (`active` by vertex, None every vertex) empty; vertices rows of (len(classes) + 63) // 64 words
    def labels(self, order, classes, label, incoming=False, active=None):
        classes = list(classes)
        index = {c: k for k, c in enumerate(classes)}
        labelled, mask = _array(C.c_uint32, [NONE if c is None else index[c] for c in label])[0], self._mask(active)
        made, (first, head) = C.POINTER(Closed)(), self.rows(incoming)
        call = LIB.mesh_graph_closure_labels
        call.restype, call.argtypes = C.c_int, [C.c_uint32, _Z, C.c_void_p, C.c_void_p, _U, C.c_ssize_t,
                                                C.POINTER(C.c_ubyte), C.c_uint32, _U, C.POINTER(C.POINTER(Closed))]
        _check(call(self.data.vertices, first, head, self.context, *self._order(order), mask, len(classes), labelled,
                    C.byref(made)))
        return Closures(Owned(made, LIB.mesh_graph_closure_free, self, (order, mask, labelled)), False,
                        (classes, index))

    # mesh_graph_longest over the prerequisite rows (`incoming`: in_first/in_edge, else row_first/row_edge) processed in
    # `order` (as ancestors'), `weight[e]` for the relation's own edge e: a View of the caller-owned lengths, by vertex
    def longest(self, order, weight, incoming=False):
        weights = (C.c_double * max(self.data.edges, 1))(*weight)
        length = (C.c_double * max(self.data.vertices, 1))()
        _check(LIB.mesh_graph_longest(self.relation.pointer, int(incoming), *self._order(order), weights, length))
        return View((self, order, weights), length, range(self.data.vertices), float)

    # mesh_graph_boundaries over this relation's edges not in `excluded` (the View of its own redundant mask) within one
    # class of `class_of` (dense IDs by vertex): Views of the caller-owned source and sink flags, by vertex
    def boundaries(self, class_of, excluded):
        classes = _array(C.c_uint32, class_of)[0]
        source, sink = ((C.c_ubyte * max(self.data.vertices, 1))() for _ in range(2))
        _check(LIB.mesh_graph_boundaries(self.relation.pointer, classes, excluded.items, source, sink))
        owner = (self, classes, excluded)
        return View(owner, source, range(self.data.vertices), bool), View(owner, sink, range(self.data.vertices), bool)

    # the native start and step of an order View covering every vertex once; its length guards the native read
    def _order(self, order):
        if len(order) != self.data.vertices:
            raise ValueError("an order covers every vertex once")
        return C.cast(C.addressof(order.items.contents) + order.span.start * C.sizeof(C.c_uint32),
                      C.POINTER(C.c_uint32)), order.span.step

    # the distinct edges in place, (from, to) objects
    def edges(self):
        return View(self.relation, self.data.edge, range(self.data.edges),
                    lambda e: (self.vertices[e.source], self.vertices[e.sink]))

    # derived relations: each retains this relation natively and in Python, with its operand array
    def contract(self, kept):
        marks = (C.c_ubyte * max(self.data.vertices, 1))(*[bool(k) for k in kept])
        return self._derived(self.vertices, LIB.mesh_graph_relation_contract, marks, marks)

    # `classes` the class objects, class k the quotient's vertex k
    def quotient(self, classes, class_of):
        held = _array(C.c_uint32, class_of)[0]
        return self._derived(classes, LIB.mesh_graph_relation_quotient, held, len(classes), held)

    def _derived(self, vertices, call, held, *operands):
        made = C.c_void_p()
        _check(call(self.relation.pointer, *operands, None, None, C.byref(made)))
        return Graph(vertices, Relation(made.value, (held,), self.relation))


def walk(universe, starts, neighbours):
    return Graph.adjacency(universe, neighbours).walk(starts)


def reach(universe, starts, neighbours):
    return Graph.adjacency(universe, neighbours).reach(starts)


# mesh-graph.h mesh_graph_ancestor (resolved on use): whether b is in a's subtree of the owned depth-first search
# (Graph.search), the objects mapped through its graph
def ancestor(owned, a, b):
    call = LIB.mesh_graph_ancestor
    call.restype, call.argtypes = C.c_int, [C.POINTER(Searched), C.c_uint32, C.c_uint32]
    at = owned.graph.at
    return a in at and b in at and bool(call(owned.made, at[a], at[b]))


# A. Goralčíková and V. Koubek, LNCS 74, 1979; K. Simon, TCS 58, 1988; design/aot-compilation-change-request.md A10
def closure(order, neighbours):
    graph = Graph.adjacency(order, neighbours)
    owned = graph.search(graph.vertices)
    return graph.ancestors(ordered(owned, owned.made.contents.postorder, range(owned.made.contents.count)))


# G. A. Kildall, POPL 1973; J. B. Kam and J. D. Ullman, J. ACM 1976; GSPMD, Xu et al., arXiv 2105.04663, section 3.5;
# design/joint-compilation.md §3, row 1; mesh-graph.h struct mesh_graph_projection, struct mesh_graph_sharding and
# mesh_graph_sharding (resolved on use): `state` solved in place, the descriptor retaining its operand arrays
class Projection(C.Structure):
    _fields_ = [('operand', C.c_uint32), ('level', C.c_uint32), ('forward', C.c_size_t), ('backward', C.c_size_t)]


class Sharding(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('atoms', C.c_uint32), ('levels', C.c_uint32), ('padding', C.c_uint32),
                ('order', C.POINTER(C.c_uint32)), ('axis', C.POINTER(C.c_uint32)), ('first', C.POINTER(C.c_size_t)),
                ('edge_first', C.POINTER(C.c_size_t)), ('map', C.POINTER(C.c_size_t)),
                ('frozen', C.POINTER(C.c_ubyte)), ('edge', C.POINTER(Projection)), ('state', C.POINTER(C.c_uint32))]


def sharding(order, axis, first, edge_first, maps, frozen, edges, state, levels):
    call = LIB.mesh_graph_sharding
    call.restype, call.argtypes = C.c_int, [C.POINTER(Sharding)]
    arrays = [_array(kind, items)[0] for kind, items in ((C.c_uint32, order), (C.c_uint32, axis), (C.c_size_t, first),
                                                           (C.c_size_t, edge_first), (C.c_size_t, maps),
                                                           (C.c_ubyte, frozen), (Projection, edges), (C.c_uint32, state))]
    made = Sharding(len(order), len(axis), levels, 0, *arrays)
    _check(call(C.byref(made)))
    return made


# G. A. Kildall, POPL 1973; J. B. Kam and J. D. Ullman, J. ACM 1976
def forward(order, neighbours, transfer):
    values = {}
    for v in order:
        values[v] = transfer(v, [values[u] for u in neighbours(v)])
    return values


def postorder(universe, roots, neighbours):
    return Graph.adjacency(universe, neighbours).postorder(roots)


# one end of each key's lifetime, read through the key's value ID from its native result array
class Endpoint(collections.abc.Mapping):
    def __init__(self, lifetimes, array):
        self.lifetimes, self.array = lifetimes, array

    def __getitem__(self, key):
        return self.array[self.lifetimes.at[key]]

    def __iter__(self):
        return iter(self.lifetimes.at)

    def __len__(self):
        return len(self.lifetimes.at)


# ExecuTorch exir/memory_planning.py update_tensor_lifetime; mesh-graph.h mesh_graph_lifetimes: over use events (key,
# position), `at` each used key's value ID in first-use order, the native use arrays and the first and last position
# of each key's uses, read through `first` and `last`
class Lifetimes:
    def __init__(self, uses):
        self.at, value, position = {}, [], []
        for key, g in uses:
            value.append(self.at.setdefault(key, len(self.at)))
            position.append(g)
        n, count = len(self.at), len(value)
        self.value, self.position = (C.c_uint32 * max(count, 1))(*value), (C.c_size_t * max(count, 1))(*position)
        self.firsts, self.lasts = (C.c_size_t * max(n, 1))(), (C.c_size_t * max(n, 1))()
        call = LIB.mesh_graph_lifetimes
        call.restype, call.argtypes = C.c_int, [C.c_uint32, C.c_size_t, _U, _Z, _Z, _Z]
        _check(call(n, count, self.value, self.position, self.firsts, self.lasts))

    @property
    def first(self):
        return Endpoint(self, self.firsts)

    @property
    def last(self):
        return Endpoint(self, self.lasts)


# F. Harary, Graph Theory, 1969; design/aot-compilation-change-request.md A10: the paths of a relation whose vertices
# have one successor and one predecessor at most (disjoint chains), each from a vertex with a successor and no
# predecessor, in the vertices' order; one depth-first search (mesh_graph_search) from those roots over the one retained
# graph, each path the borrowed preorder slice [place[root], end[root]) of its root's subtree, its vertices' objects:
# a View over the roots owning that search (no roots: an empty View, the search still retained)
def paths(vertices, neighbours):
    graph = Graph.adjacency(vertices, neighbours)
    into, out = graph.data.in_first, graph.data.row_first
    roots = [k for k in range(len(graph.vertices)) if into[k + 1] == into[k] and out[k + 1] > out[k]]
    owned = graph.search([graph.vertices[k] for k in roots])
    found = owned.made.contents
    return View(owned, roots, range(len(roots)),
                lambda k: ordered(owned, found.preorder, range(found.place[k], found.end[k])))


# design/aot-compilation-change-request.md A10, "Its bound"; mesh-graph.h mesh_graph_closure_chains (resolved on use;
# R. P. Dilworth, 1950; D. R. Fulkerson, 1956): the fewest chains covering `members`, vertices of the closure `closed`
# (a closure of rows as `closure` returns it, each vertex's set its predecessors), each counted once, a before b where a
# is in b's set: their count, and a View of each member's chain retaining the closure
def closure_chains(closed, members):
    call = LIB.mesh_graph_closure_chains
    call.restype, call.argtypes = C.c_int, [C.POINTER(Closed), C.c_uint32, _U, C.POINTER(C.c_uint64),
                                            C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    owned = closed.owned
    member, items = _array(C.c_uint32, [owned.graph.at[m] for m in members])
    counts = (C.c_uint64 * max(items, 1))(*[1] * items)
    chain, chains = (C.c_uint64 * max(items, 1))(), C.c_uint64()
    _check(call(owned.made, items, member, counts, chain, C.byref(chains)))
    return chains.value, View((owned, member, counts, chains), chain, range(items), int)


LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
NONE = 2 ** 32 - 1
KINDS = ('queue', 'host', 'bridge', 'kernel')
PRIMITIVES = ('submit', 'queue_signal', 'queue_wait', 'fence_update', 'fence_wait', 'pass_barrier', 'queue_barrier',
              'host_wait', 'host_notify', 'bridge_start', 'threadgroup_barrier', 'simdgroup_barrier', 'atomic',
              'producer_barrier', 'bridge_raise', 'initial', 'host_publish', 'kernel_fence', 'host_signal', 'bridge_read')
NODES, PLAN_COUNT, PLAN_SUBMISSION, LOWERED, PLAN_SUPPLIED = 64, 0x80000000, 0xC0000000, 0x40000000, 0x20000000
STATES = ('image', 'stronger', 'false', 'unmatched', 'satisfied', 'transformed', 'scoped')

# mesh-graph.h mesh_graph_search, mesh_graph_breadth, mesh_graph_search_free, mesh_graph_relation_head,
# mesh_graph_closure and mesh_graph_closure_free; MESH_GRAPH_GIVEN and SIZE_MAX
GIVEN, SIZE = 3, 2 ** 64 - 1
_U, _Z = C.POINTER(C.c_uint32), C.POINTER(C.c_size_t)
LIB.mesh_graph_search.restype = LIB.mesh_graph_breadth.restype = LIB.mesh_graph_closure.restype = C.c_int
LIB.mesh_graph_search.argtypes = [C.c_uint32, _Z, C.c_void_p, C.c_void_p, C.c_uint32, _U, C.c_int,
                                  C.POINTER(C.POINTER(Searched))]
LIB.mesh_graph_breadth.argtypes = [C.c_uint32, _Z, C.c_void_p, C.c_void_p, C.c_uint32, _U,
                                   C.POINTER(C.POINTER(Searched))]
LIB.mesh_graph_closure.argtypes = [C.c_uint32, _Z, C.c_void_p, C.c_void_p, _U, C.c_ssize_t,
                                   C.POINTER(C.POINTER(Closed))]
LIB.mesh_graph_longest.restype = C.c_int
LIB.mesh_graph_longest.argtypes = [C.c_void_p, C.c_int, _U, C.c_ssize_t, C.POINTER(C.c_double), C.POINTER(C.c_double)]
LIB.mesh_graph_boundaries.restype = C.c_int
LIB.mesh_graph_boundaries.argtypes = [C.c_void_p, _U, C.POINTER(C.c_ubyte), C.POINTER(C.c_ubyte), C.POINTER(C.c_ubyte)]
LIB.mesh_graph_search_free.restype = LIB.mesh_graph_closure_free.restype = None
LIB.mesh_graph_search_free.argtypes = [C.POINTER(Searched)]
LIB.mesh_graph_closure_free.argtypes = [C.POINTER(Closed)]
LIB.mesh_graph_search_save.restype = LIB.mesh_graph_search_load.restype = C.c_int
LIB.mesh_graph_search_save.argtypes = [C.POINTER(Searched), C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
LIB.mesh_graph_search_load.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.POINTER(Searched))]
LIB.mesh_graph_closure_save.restype = LIB.mesh_graph_closure_load.restype = C.c_int
LIB.mesh_graph_closure_save.argtypes = [C.POINTER(Closed), C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
LIB.mesh_graph_closure_load.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.POINTER(Closed))]
RELATION_HEAD = C.cast(LIB.mesh_graph_relation_head, C.c_void_p)


# mesh-graph.h struct mesh_graph_access
class Access(C.Structure):
    _fields_ = [('key', C.c_uint64), ('start', C.c_uint64), ('end', C.c_uint64), ('vertex', C.c_uint32),
                ('write', C.c_uint32)]


# mesh-graph.h struct mesh_graph_machine, mesh_graph_read and mesh_graph_actor (mesh_graph_actor_time; design/torch-
# capture-change-request.md, "Source for item 2's execution-time estimates")
class Rate(C.Structure):
    _fields_ = [('peak', C.c_double), ('lanes', C.c_uint32), ('padding', C.c_uint32)]


class Work(C.Structure):
    _fields_ = [('operations', (C.c_double * 3) * 3), ('unknown', C.c_uint64)]


class Machine(C.Structure):
    _fields_ = [('rate', (Rate * 3) * 3)] + [(name, C.c_double) for name in (
        'bandwidth', 'block_bandwidth', 'l2', 'l1', 'launch', 'l2_speedup', 'l1_speedup', 'overlap')] + [
        ('cores', C.c_uint32), ('padding', C.c_uint32), ('direct', C.c_double), ('blit', C.c_double)]


class Read(C.Structure):
    _fields_ = [('net', C.c_uint64), ('total', C.c_uint64), ('utilization', C.c_double)]


class Actor(C.Structure):
    _fields_ = [('work', Work), ('blocks', C.c_uint64), ('threads', C.c_uint64), ('written', C.c_uint64),
                ('launches', C.c_uint64), ('reads', C.c_size_t), ('read', C.POINTER(Read)), ('direct', C.c_uint64),
                ('blits', C.c_uint64)]


LIB.mesh_graph_actor_time.restype = C.c_int
LIB.mesh_graph_actor_time.argtypes = [C.POINTER(Machine), C.POINTER(Actor), C.POINTER(C.c_double)]
LIB.mesh_graph_buffer_bound.restype = C.c_int
LIB.mesh_graph_buffer_bound.argtypes = [C.c_uint32, C.POINTER(C.c_double), C.POINTER(C.c_double), C.c_size_t,
                                        C.c_void_p, C.POINTER(C.c_uint64)]


# mesh-graph.h struct mesh_graph_processor, mesh_graph_task, mesh_graph_dependence, mesh_graph_operation,
# mesh_graph_synchronized, mesh_graph_signature
class Processor(C.Structure):
    _fields_ = [('kind', C.c_uint32), ('index', C.c_uint32)]


class Task(C.Structure):
    _fields_ = [('processor', C.c_uint32), ('place', C.c_uint32), ('label', C.c_uint32), ('copy', C.c_uint32)]


class Dependence(C.Structure):
    _fields_ = [('source', C.c_uint32), ('sink', C.c_uint32), ('delay', C.c_uint64)]


class Grouping(C.Structure):
    _fields_ = [('keys', C.c_uint32), ('items', C.c_size_t), ('key', C.c_void_p), ('context', C.c_void_p),
                ('first', _Z), ('order', _Z), ('ordinal', _Z)]


class Tree(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('root', C.c_uint32), ('parent', _U),
                ('children', C.POINTER(Grouping))]


class LagGraph(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('first', _Z), ('head', C.c_void_p), ('lag', C.c_void_p),
                ('context', C.c_void_p)]


class LagResult(C.Structure):
    _fields_ = [('graph', C.POINTER(LagGraph)), ('queries', C.c_size_t), ('query', C.POINTER(Dependence)),
                ('sources', C.POINTER(Grouping)), ('value', C.POINTER(C.c_uint64))]


class LagMatrix(C.Structure):
    _fields_ = [('graph', C.POINTER(LagGraph)), ('sources', C.c_uint32), ('targets', C.c_uint32),
                ('source', _U), ('target', _U), ('value', C.POINTER(C.c_uint64))]


class EventLagResult(C.Structure):
    _fields_ = [('graph', C.POINTER(LagGraph)), ('events', C.c_uint32), ('first', _Z), ('member', _U),
                ('queries', C.c_size_t), ('query', C.POINTER(Dependence)), ('expanded_first', _Z),
                ('expanded', C.POINTER(Dependence)), ('lags', C.POINTER(LagResult)),
                ('value', C.POINTER(C.c_uint64))]


class BufferResult(C.Structure):
    _fields_ = [('references', C.c_size_t), ('vertices', C.c_uint32), ('actors', C.c_uint32),
                ('time', C.POINTER(C.c_double)), ('host', C.POINTER(C.c_double)), ('work', C.POINTER(C.c_double)),
                ('edges', C.c_size_t), ('part_edges', C.c_size_t), ('queries', C.c_size_t), ('from', _U), ('to', _U),
                ('delay', C.POINTER(C.c_uint64)), ('edge', C.POINTER(Dependence)), ('query', _Z),
                ('lambda', C.c_double), ('bound', C.POINTER(C.c_uint64)),
                ('release', C.c_void_p), ('owner', C.c_void_p)]


class ChainResult(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('chain', _U), ('edges', C.c_size_t), ('edge', C.POINTER(Dependence))]


class Touch(C.Structure):
    _fields_ = [('start', C.c_uint64), ('end', C.c_uint64), ('value', C.c_uint32), ('member', C.c_uint32)]


class Cover(C.Structure):
    _fields_ = [('member', _U), ('members', C.c_uint32), ('value', C.c_uint32), ('block', C.c_uint32)]


class CoverResult(C.Structure):
    _fields_ = [('quantum', C.c_uint64), ('values', C.c_uint32), ('blocks', C.c_uint32),
                ('extras', C.c_uint32), ('sets', C.c_uint32), ('touches', C.c_size_t),
                ('covers', C.c_size_t), ('listed', C.c_size_t), ('held_count', C.c_size_t), ('block_first', _U),
                ('bytes', C.c_void_p), ('context', C.c_void_p), ('touch', C.POINTER(Touch)),
                ('extra_member', C.c_void_p), ('extra_blocks', C.c_void_p), ('extra_context', C.c_void_p),
                ('order', C.POINTER(C.POINTER(Touch))), ('cover', C.POINTER(Cover))] + [
                    (name, _U) for name in ('held', 'first', 'member', 'set_blocks', 'block_set',
                                           'block_element', 'extra_set', 'uncovered')]


for name, result, arguments in (
        ('mesh_graph_group', C.c_int, [C.c_uint32, C.c_size_t, C.c_void_p, C.c_void_p,
                                      C.POINTER(C.POINTER(Grouping))]),
        ('mesh_graph_group_free', None, [C.POINTER(Grouping)]),
        ('mesh_graph_tree_create', C.c_int, [C.c_uint32, C.c_uint32, _U, C.POINTER(C.POINTER(Tree))]),
        ('mesh_graph_tree_free', None, [C.POINTER(Tree)]),
        ('mesh_graph_tree_parent', C.c_uint32, [C.c_void_p, C.c_size_t]),
        ('mesh_graph_lag_prepare', C.c_int, [C.c_uint32, _Z, C.c_void_p, C.c_void_p, C.c_void_p,
                                            C.POINTER(C.POINTER(LagGraph))]),
        ('mesh_graph_lag_graph_free', None, [C.POINTER(LagGraph)]),
        ('mesh_graph_lag_queries', C.c_int, [C.POINTER(LagGraph), C.c_size_t, C.POINTER(Dependence),
                                            C.POINTER(C.POINTER(LagResult))]),
        ('mesh_graph_lag_result_free', None, [C.POINTER(LagResult)]),
        ('mesh_graph_query_source', C.c_uint32, [C.c_void_p, C.c_size_t]),
        ('mesh_graph_buffer_source', C.c_uint32, [C.c_void_p, C.c_size_t]),
        ('mesh_graph_buffer_head', C.c_uint32, [C.c_void_p, C.c_size_t]),
        ('mesh_graph_buffer_lag', C.c_uint64, [C.c_void_p, C.c_size_t]),
        ('mesh_graph_lag_matrix', C.c_int, [C.POINTER(LagGraph), C.c_uint32, _U, C.c_uint32, _U,
                                           C.POINTER(C.POINTER(LagMatrix))]),
        ('mesh_graph_lag_matrix_free', None, [C.POINTER(LagMatrix)]),
        ('mesh_graph_event_lags', C.c_int, [C.POINTER(LagGraph), C.c_uint32, _Z, _U,
                                           C.c_size_t, C.POINTER(Dependence),
                                           C.POINTER(C.POINTER(EventLagResult))]),
        ('mesh_graph_event_lag_result_free', None, [C.POINTER(EventLagResult)]),
        ('mesh_graph_buffer_bounds', C.c_int, [C.c_uint32, C.POINTER(C.c_double), C.POINTER(C.c_double),
                                              C.c_size_t, C.POINTER(Dependence), C.POINTER(C.POINTER(BufferResult))]),
        ('mesh_graph_buffer_queries', C.c_int, [C.c_uint32, C.POINTER(C.c_double), C.c_size_t, _U, _U,
                                               C.POINTER(C.c_uint64), C.c_size_t, _Z,
                                               C.POINTER(C.POINTER(BufferResult))]),
        ('mesh_graph_buffer_result_retain', None, [C.POINTER(BufferResult)]),
        ('mesh_graph_buffer_result_free', None, [C.POINTER(BufferResult)]),
        ('mesh_graph_chain_pairs', C.c_int, [C.c_uint32, _U, C.POINTER(C.POINTER(ChainResult))]),
        ('mesh_graph_chain_result_free', None, [C.POINTER(ChainResult)]),
        ('mesh_graph_cover_sets', C.c_int, [C.c_uint64, C.c_uint32, C.c_uint32, _U, C.c_void_p, C.c_void_p,
                                           C.c_size_t, C.POINTER(Touch), C.c_uint32, C.c_void_p, C.c_void_p,
                                           C.c_void_p, C.POINTER(C.POINTER(CoverResult))]),
        ('mesh_graph_cover_result_free', None, [C.POINTER(CoverResult)]),
        ('mesh_graph_relation_edge_cause', C.c_int, [C.c_void_p, C.c_size_t, _U])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


class Operation(C.Structure):
    _fields_ = [('processor', C.c_uint32), ('primitive', C.c_uint32), ('vertex', C.c_uint32), ('object', C.c_uint32),
                ('execution', C.c_uint64), ('value', C.c_uint64)]


class Synchronized(C.Structure):
    _fields_ = [('signal', C.c_uint32), ('wait', C.c_uint32), ('state', C.c_uint32), ('redundant', C.c_uint32),
                ('dependence', C.c_uint32), ('witness', C.c_uint32), ('delay', C.c_uint64)]


class Event(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('object', 'role', 'argument', 'padding')] + [('lookback', C.c_uint64)]


class Span(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('first', 'count', 'begin', 'end')]


# mesh-graph.h struct mesh_graph_repeated and struct mesh_graph_proof
class Repeated(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('processor', 'primitive', 'vertex', 'object')] + [
        (n, C.c_uint64) for n in ('every', 'ordinal', 'lookback')] + [('startup', C.c_uint32), ('padding', C.c_uint32)]


class Proof(C.Structure):
    _fields_ = [('source', C.c_uint32), ('sink', C.c_uint32), ('total_delay', C.c_uint64), ('witness', C.c_uint32),
                ('proven', C.c_uint32)]


# mesh-graph.h struct mesh_graph_completion, mesh_graph_completion_writer, mesh_graph_release, mesh_graph_scopes_data,
# mesh_graph_occurrence, mesh_graph_site and mesh_graph_answered: completion scopes and their realized events
class Completion(C.Structure):
    _fields_ = [('value', C.c_uint32), ('counter', C.c_uint32), ('writer', C.c_uint32), ('writers', C.c_uint32),
                ('total', C.c_uint64)]


class CompletionWriter(C.Structure):
    _fields_ = [('dispatch', C.c_uint32), ('completion', C.c_uint32), ('later', C.c_uint32), ('padding', C.c_uint32),
                ('count', C.c_uint64)]


class Release(C.Structure):
    _fields_ = [('owner', C.c_uint32), ('transfer', C.c_uint32)]


# mesh-graph.h struct mesh_graph_scope_source: a scope row's original artifact row, native node and structural identity
class ScopeSource(C.Structure):
    _fields_ = [('row', C.c_uint32), ('node', C.c_uint32), ('identity', C.c_ubyte * 32)]


class ScopesData(C.Structure):
    _fields_ = [('completions', C.c_size_t), ('writers', C.c_size_t), ('releases', C.c_size_t),
                ('completion', C.POINTER(Completion)), ('writer', C.POINTER(CompletionWriter)),
                ('release_first', C.POINTER(C.c_size_t)), ('release', C.POINTER(Release)),
                ('source', C.POINTER(ScopeSource))]


class Occurrence(C.Structure):
    _fields_ = [('completion', C.c_uint32), ('site', C.c_uint32), ('event', C.c_uint32), ('row', C.c_uint32),
                ('execution', C.c_uint64), ('generation', C.c_uint64), ('copy', C.c_uint64)]


class Site(C.Structure):
    _fields_ = [('first', C.c_uint32), ('later', C.c_uint32), ('first_site', C.c_uint32), ('later_site', C.c_uint32)]


class Answered(C.Structure):
    _fields_ = [('event', C.c_uint32), ('operation', C.c_uint32)]


class Signature(C.Structure):
    _fields_ = [('processors', C.c_uint32), ('tasks', C.c_uint32), ('operations', C.c_size_t),
                ('dependences', C.c_size_t), ('synchronized', C.c_size_t), ('uncovered', C.c_size_t),
                ('redundant', C.c_size_t), ('unwaited', C.c_size_t), ('states', C.c_size_t * len(STATES)),
                ('processor', C.POINTER(Processor)), ('task', C.POINTER(Task)), ('operation', C.POINTER(Operation)),
                ('dependence', C.POINTER(Dependence)), ('edge', C.POINTER(Synchronized)),
                ('position', C.POINTER(C.c_uint64)), ('count', C.POINTER(C.c_uint64)),
                ('least', C.POINTER(C.c_uint64)), ('previous', C.POINTER(C.c_size_t)),
                ('waited', C.POINTER(C.c_size_t)), ('first', C.POINTER(C.c_size_t)), ('order', C.POINTER(C.c_size_t)),
                ('unknown', C.POINTER(C.c_uint64)), ('alias', C.POINTER(C.c_uint32)), ('digest', C.c_ubyte * 32), ('events', C.c_size_t),
                ('event', C.POINTER(Event)), ('spans', C.c_size_t), ('span', C.POINTER(Span)),
                ('repeats', C.c_size_t), ('repeated', C.POINTER(Repeated)), ('proofs', C.c_size_t),
                ('proof', C.POINTER(Proof)), ('scopes', C.c_void_p), ('realized', C.c_void_p),
                ('occurrences', C.c_size_t), ('sites', C.c_size_t), ('publications', C.c_size_t),
                ('answers', C.c_size_t), ('occurrence', C.POINTER(Occurrence)), ('site', C.POINTER(Site)),
                ('published', C.POINTER(C.c_uint64)), ('answer', C.POINTER(Answered)), ('projected', C.POINTER(Site)),
                ('archived', C.c_void_p)]
    # the Python buffers whose bytes a scoped signature's views borrow (from_record, from_archive; a composed
    # signature's its operands'), kept as long as it is; None: it borrows none
    borrowed = None


PRINTERS = ('mesh_graph_matrix_market', 'mesh_graph_dot', 'mesh_graph_space_time')
LIB.mesh_graph_signature.restype = C.c_int
LIB.mesh_graph_signature.argtypes = [C.c_uint32, C.POINTER(Processor), C.c_uint32, C.POINTER(Task), C.c_size_t,
                                     C.POINTER(Dependence), C.c_size_t, C.POINTER(Operation), C.POINTER(Signature)]
LIB.mesh_graph_signature_free.restype = None
LIB.mesh_graph_signature_free.argtypes = [C.POINTER(Signature)]
for name in PRINTERS:
    getattr(LIB, name).restype = C.c_int
    getattr(LIB, name).argtypes = [C.POINTER(Signature), C.c_char_p, C.c_size_t, C.POINTER(C.c_size_t)]


def _check(status):
    if status:
        raise OSError(status, os.strerror(status))


def _array(kind, items):
    items = list(items)
    return (kind * max(len(items), 1))(*items), len(items)


# mesh-graph.h mesh_graph_actor_time: (t(v) in ns over the recorded operands, whether every operand it needs is recorded)
def actor_time(machine, actor):
    time = C.c_double()
    status = LIB.mesh_graph_actor_time(C.byref(machine), C.byref(actor), C.byref(time))
    if status not in (0, errno.ENODATA):
        _check(status)
    return time.value, status == 0


def buffer_queries(source, *operands):
    made = C.POINTER(BufferResult)()
    _check(LIB.mesh_graph_buffer_queries(*operands, C.byref(made)))
    return Owned(made, LIB.mesh_graph_buffer_result_free, source, operands)


def event_lags(bounds, first, member, queries):
    original = bounds.made.contents
    groups = C.POINTER(Grouping)()
    _check(LIB.mesh_graph_group(original.vertices, original.edges,
                                C.cast(LIB.mesh_graph_buffer_source, C.c_void_p),
                                C.cast(bounds.made, C.c_void_p), C.byref(groups)))
    grouping = Owned(groups, LIB.mesh_graph_group_free, bounds)
    prepared = C.POINTER(LagGraph)()
    _check(LIB.mesh_graph_lag_prepare(original.vertices, groups.contents.first,
                                      C.cast(LIB.mesh_graph_buffer_head, C.c_void_p),
                                      C.cast(LIB.mesh_graph_buffer_lag, C.c_void_p),
                                      C.cast(groups, C.c_void_p), C.byref(prepared)))
    graph = Owned(prepared, LIB.mesh_graph_lag_graph_free, bounds, grouping)
    made = C.POINTER(EventLagResult)()
    _check(LIB.mesh_graph_event_lags(prepared, len(first) - 1, first, member, len(queries), queries, C.byref(made)))
    return Owned(made, LIB.mesh_graph_event_lag_result_free, bounds, (graph, first, member, queries))


# BSL §4.2-4.3, (6); mesh-graph.h mesh_graph_buffer_bound: Bff of each actor's host -> device edge from its t(v),
# `edges` (from, to, delay) between parts (host part p vertex p, device part p len(host) + p)
def buffer_bound(host, work, edges=()):
    count = len(host)
    bound = (C.c_uint64 * max(count, 1))()
    edge = (Dependence * max(len(edges), 1))(*[Dependence(a, b, d) for a, b, d in edges])
    _check(LIB.mesh_graph_buffer_bound(count, (C.c_double * max(count, 1))(*host), (C.c_double * max(count, 1))(*work),
                                       len(edges), C.cast(edge, C.c_void_p), bound))
    return list(bound[:count])


# design/unequal-shards-change-request.md items 6 and 14; S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3);
# mesh-graph.h mesh_graph_shares: `comm[i]` stage i's collective time a unit of the largest share, `comp[i][j]` rank j's
# computation time in stage i as (fixed, a unit of its share), `upper[j]` the most rank j holds
LIB.mesh_graph_shares.restype = C.c_int
LIB.mesh_graph_shares.argtypes = [C.c_uint32, C.c_uint32] + [C.POINTER(C.c_double)] * 5


def shares(comm, comp, upper):
    ranks, stages = len(upper), len(comm)
    fixed, slope = ([time[k] for stage in comp for time in stage] for k in range(2))
    share = (C.c_double * max(ranks, 1))()
    _check(LIB.mesh_graph_shares(ranks, stages, *(_array(C.c_double, v)[0] for v in (comm, fixed, slope, upper)),
                                 share))
    return list(share[:ranks])


# mesh-graph.h mesh_graph_shares_coupled (resolved on use): `comm[i][p]` stage i's collective time a unit of part p's
# largest share; each of `rows` (stage, fixed, {(part, rank): coefficient}) one row of its stage, its coefficients over
# the full share variables part * ranks + rank; share[p][rank] part p's share of that rank
def shares_coupled(comm, rows, upper, parts):
    ranks, stages, first = len(upper), len(comm), [0]
    for _, _, coefficients in rows:
        first.append(first[-1] + len(coefficients))
    call = LIB.mesh_graph_shares_coupled
    call.restype = C.c_int
    call.argtypes = [C.c_uint32] * 4 + [C.POINTER(C.c_double), C.POINTER(C.c_uint32), C.POINTER(C.c_double),
                                        C.POINTER(C.c_size_t), C.POINTER(C.c_uint32)] + [C.POINTER(C.c_double)] * 3
    share = (C.c_double * max(parts * ranks, 1))()
    _check(call(ranks, parts, stages, len(rows), _array(C.c_double, [c for stage in comm for c in stage])[0],
                _array(C.c_uint32, [stage for stage, _, _ in rows])[0],
                _array(C.c_double, [fixed for _, fixed, _ in rows])[0], _array(C.c_size_t, first)[0],
                _array(C.c_uint32, [p * ranks + j for _, _, coefficients in rows for p, j in coefficients])[0],
                _array(C.c_double, [c for _, _, coefficients in rows for c in coefficients.values()])[0],
                _array(C.c_double, upper)[0], share))
    return [list(share[p * ranks:(p + 1) * ranks]) for p in range(parts)]


# BSL §3; R. E. Tarjan, SIAM J. Comput. 1(2), 1972; mesh-graph.h mesh_graph_components: each vertex's strongly
# connected component, a View of the caller-owned native array (its edge operands its owner), by vertex
LIB.mesh_graph_components.restype = C.c_int
LIB.mesh_graph_components.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
                                      C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]


def components(vertices, edges):
    tails, heads = _array(C.c_uint32, [a for a, _ in edges]), _array(C.c_uint32, [b for _, b in edges])
    component, count = (C.c_uint32 * max(vertices, 1))(), C.c_uint32()
    _check(LIB.mesh_graph_components(vertices, len(edges), tails[0], heads[0], component, C.byref(count)))
    return View((tails, heads), component, range(vertices), int)


# mesh-graph.h mesh_graph_rows: items bucketed by key, stable; first[keys + 1], order the items grouped by key
ROW_KEY = C.CFUNCTYPE(C.c_uint32, C.c_void_p, C.c_size_t)
LIB.mesh_graph_rows.restype = None
LIB.mesh_graph_rows.argtypes = [C.c_uint32, C.c_size_t, ROW_KEY, C.c_void_p, C.POINTER(C.c_size_t),
                                C.POINTER(C.c_size_t)]


def rows(keys, key):
    first, order = (C.c_size_t * (keys + 1))(), (C.c_size_t * max(len(key), 1))()
    LIB.mesh_graph_rows(keys, len(key), ROW_KEY(lambda context, k: key[k]), None, first, order)
    return list(first), list(order[:len(key)])


# design/aot-compilation-change-request.md A10: the successive pairs of each row of `rows`, a chain labelling's order
def successive(keys, key):
    first, order = rows(keys, key)
    return [(order[i], order[i + 1]) for k in range(keys) for i in range(first[k], first[k + 1] - 1)]


def _text(printer, signature):
    made = C.c_size_t()
    status = printer(C.byref(signature), None, 0, C.byref(made))
    if status != errno.ERANGE:
        _check(status or errno.EINVAL)
    text = C.create_string_buffer(made.value)
    _check(printer(C.byref(signature), text, made.value, C.byref(made)))
    return text.value.decode()


# mesh-graph.h mesh_graph_matrix_market, mesh_graph_dot, mesh_graph_space_time and the signature's counts
def forms(made):
    texts = [_text(getattr(LIB, name), made) for name in PRINTERS]
    return {'digest': bytes(made.digest).hex(), 'matrix_market': texts[0], 'dot': texts[1], 'space_time': texts[2],
            'synchronized': [(e.signal, e.wait, STATES[e.state], bool(e.redundant), e.dependence, e.witness, e.delay)
                             for e in made.edge[:made.synchronized]],
            'counts': {(q, PRIMITIVES[k]): made.count[q * len(PRIMITIVES) + k]
                       for q in range(made.processors) for k in range(len(PRIMITIVES)) if made.count[q * len(PRIMITIVES) + k]},
            'unknown': {(q, PRIMITIVES[k]): made.unknown[q * len(PRIMITIVES) + k]
                        for q in range(made.processors) for k in range(len(PRIMITIVES))
                        if made.unknown[q * len(PRIMITIVES) + k]},
            'states': dict(zip(STATES, made.states)), 'redundant': made.redundant, 'uncovered': made.uncovered,
            'unwaited': made.unwaited}


def free(made):
    LIB.mesh_graph_signature_free(C.byref(made))


# design/synchronization-signature-change-request.md items 1, 2, 4 and 6; mesh-graph.h mesh_graph_signature;
# design/compilation-composition-change-request.md items 3 and 6
def build_signature(processors, tasks, dependences, operations):
    made = Signature()
    processor, p = _array(Processor, processors)
    task, t = _array(Task, tasks)
    dependence, d = _array(Dependence, dependences)
    operation, o = _array(Operation, operations)
    _check(LIB.mesh_graph_signature(p, processor, t, task, d, dependence, o, operation, C.byref(made)))
    return made


def signature(processors, tasks, dependences, operations):
    made = build_signature(processors, tasks, dependences, operations)
    try:
        return dict(forms(made), record=record(made))
    finally:
        free(made)


# mesh-graph.h struct mesh_graph_record, mesh_graph_record, mesh_graph_record_read: the compiled program's signature
# at its external boundary (the frontend's and E2B's record, the monitor's input)
LIB.mesh_graph_record.restype = C.c_int
LIB.mesh_graph_record.argtypes = [C.POINTER(Signature), C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
LIB.mesh_graph_record_read.restype = C.c_int
LIB.mesh_graph_record_read.argtypes = [C.c_void_p, C.POINTER(Signature)]


def record(made):
    size = C.c_size_t()
    status = LIB.mesh_graph_record(C.byref(made), None, 0, C.byref(size))
    if status != errno.ERANGE:
        _check(status or errno.EINVAL)
    buffer = C.create_string_buffer(size.value)
    _check(LIB.mesh_graph_record(C.byref(made), buffer, size.value, C.byref(size)))
    return buffer.raw[:size.value]


# the caller's immutable bytes viewed in place (other input made bytes once, as its normalization) through a ctypes
# pointer into them: an archive's restored signature (every array) and a record's scopes borrow them, so the returned
# signature keeps both
def _immutable(data):
    data = data if isinstance(data, bytes) else bytes(data)
    return data, C.c_char_p(data)


def from_record(data):
    data, pointer = _immutable(data)
    made = Signature()
    _check(LIB.mesh_graph_record_read(pointer, C.byref(made)))
    made.borrowed = (data, pointer)
    return made


# mesh-graph.h mesh_graph_archive, mesh_graph_archive_read and mesh_graph_compose;
# design/compilation-composition-change-request.md items 3 and 6
LIB.mesh_graph_archive.restype = LIB.mesh_graph_archive_read.restype = LIB.mesh_graph_compose.restype = C.c_int
LIB.mesh_graph_archive.argtypes = [C.POINTER(Signature), C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
LIB.mesh_graph_archive_read.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p, C.POINTER(Signature)]
LIB.mesh_graph_compose.argtypes = [C.POINTER(Signature), C.POINTER(Signature), C.POINTER(Signature)]


def archive(made):
    size = C.c_size_t()
    status = LIB.mesh_graph_archive(C.byref(made), None, 0, C.byref(size))
    if status != errno.ERANGE:
        _check(status or errno.EINVAL)
    buffer = C.create_string_buffer(size.value)
    _check(LIB.mesh_graph_archive(C.byref(made), buffer, size.value, C.byref(size)))
    return buffer.raw[:size.value]


def from_archive(data):
    data, pointer = _immutable(data)
    made = Signature()
    _check(LIB.mesh_graph_archive_read(pointer, len(data), None, None, C.byref(made)))
    made.borrowed = (data, pointer)
    return made


# the composed signature retains both operands' scopes and realized holders, so it keeps their borrowed buffers
def compose(a, b):
    made = Signature()
    _check(LIB.mesh_graph_compose(C.byref(a), C.byref(b), C.byref(made)))
    made.borrowed = (a.borrowed, b.borrowed)
    return made


# mesh-graph.h mesh_graph_signature_span; design/synchronization-signature-change-request.md item 1
LIB.mesh_graph_signature_span.restype = C.c_int
LIB.mesh_graph_signature_span.argtypes = [C.POINTER(Signature), C.c_uint64, C.POINTER(Signature)]


def signature_span(made, executions):
    spanned = Signature()
    _check(LIB.mesh_graph_signature_span(C.byref(made), executions, C.byref(spanned)))
    return spanned


# ISO/IEC 9899:2018 7.22.3.3: the release of a conflict vector the library allocated
LIB.free.restype = None
LIB.free.argtypes = [C.c_void_p]


# mesh-graph.h struct mesh_graph_view (its struct mesh_strided in place) and struct mesh_graph_conflict,
# mesh_graph_dependences_views and mesh_graph_synchronization; design/compilation-composition-change-request.md
# item 6 and "Compact effects"
class Viewed(C.Structure):
    # `access` its access relation (mesh-graph.h struct mesh_graph_access_map), NULL for a strided view
    _fields_ = [('key', C.c_uint64), ('offset', C.c_uint64), ('element', C.c_uint64), ('dims', C.c_uint32),
                ('padding', C.c_uint32), ('pairs', C.POINTER(C.c_uint64)), ('vertex', C.c_uint32), ('write', C.c_uint32),
                ('access', C.c_void_p)]


class Conflict(C.Structure):
    _fields_ = [('source', C.c_uint32), ('sink', C.c_uint32), ('kind', C.c_uint32), ('given', C.c_uint32),
                ('key', C.c_uint64), ('from_view', C.c_size_t), ('to_view', C.c_size_t)]


HAZARDS = ('read after write', 'write after read', 'write after write', 'given')


# mesh-graph.h struct mesh_strided; torch 2.14.0 c10/core/TensorImpl.h, storage_offset_ and sizes_and_strides_
def strided(offset, sizes, strides, element):
    return offset * element, element, [(n if s else min(n, 1), s * element) for n, s in zip(sizes, strides)]


def extent(sizes, strides):
    return (1 + sum((n - 1) * s for n, s in zip(sizes, strides))) * all(sizes)


# MPI 4.1 section 5.1.2, MPI_TYPE_VECTOR; output_data/compiler-workgroup-20261113/compiler/placement.replacement.md:
# x = lo_a + p j, (p/g) j = (lo_b - lo_a)/g (mod q/g), g = gcd(p, q), period = lcm(p, q); a[left[k]] = common[k] =
# b[right[k]]; Python 3.12 pow(x, -1, m)
def intersection(a, b):
    if not a or not b:
        return range(0), range(0), range(0)
    p, q = abs(a.step), abs(b.step)
    (low_a, high_a), (low_b, high_b) = sorted((a[0], a[-1])), sorted((b[0], b[-1]))
    low, high, g, d = max(low_a, low_b), min(high_a, high_b), math.gcd(p, q), low_b - low_a
    if low > high or d % g:
        return range(0), range(0), range(0)
    base, period = low_a + p * (d // g * pow(p // g, -1, q // g) % (q // g)), p // g * q
    first = base - (base - low) // period * period
    count = max(0, 1 + (high - first) // period)
    return range(first, first + count * period, period), *(
        range((first - r.start) // r.step, (first - r.start) // r.step + count * (period // r.step), period // r.step)
        for r in (a, b))


# mesh-graph.h struct mesh_graph_witness (16 bytes) and struct mesh_graph_origin (24 bytes)
class Witness(C.Structure):
    _fields_ = [('record', C.c_size_t), ('reverse', C.c_uint32), ('known', C.c_uint32)]


class Origin(C.Structure):
    _fields_ = [('relation', C.c_void_p), ('witness', Witness)]


# mesh-graph.h struct mesh_graph_relation_data, mesh_graph_relation_create, mesh_graph_relation_data and
# mesh_graph_relation_free; design/compilation-composition-change-request.md items 6, 6.4 and 7
class RelationData(C.Structure):
    _fields_ = [('vertices', C.c_uint32), ('views', C.c_size_t), ('conflicts', C.c_size_t), ('edges', C.c_size_t),
                ('view', C.POINTER(Viewed)), ('conflict', C.POINTER(Conflict)), ('edge', C.POINTER(Dependence)),
                ('first', C.POINTER(C.c_size_t)), ('member', C.POINTER(C.c_size_t)),
                ('row_first', C.POINTER(C.c_size_t)), ('row_edge', C.POINTER(C.c_size_t)),
                ('in_first', C.POINTER(C.c_size_t)), ('in_edge', C.POINTER(C.c_size_t)),
                ('parent', C.c_void_p), ('source_first', C.POINTER(C.c_size_t)), ('source', C.POINTER(C.c_size_t)),
                ('class_of', C.POINTER(C.c_uint32)), ('kept', C.POINTER(C.c_ubyte)), ('query', C.c_uint32),
                ('origin', C.POINTER(Origin))]


LIB.mesh_graph_relation_create.restype = C.c_int
LIB.mesh_graph_relation_create.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(Viewed), C.c_size_t, C.POINTER(Conflict),
                                           C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)]
LIB.mesh_graph_relation_data.restype = C.POINTER(RelationData)
LIB.mesh_graph_relation_data.argtypes = [C.c_void_p]
LIB.mesh_graph_relation_free.restype = None
LIB.mesh_graph_relation_free.argtypes = [C.c_void_p]
LIB.mesh_graph_relation_contract.restype = LIB.mesh_graph_relation_quotient.restype = C.c_int
LIB.mesh_graph_relation_contract.argtypes = [C.c_void_p, C.POINTER(C.c_ubyte), C.c_void_p, C.c_void_p,
                                             C.POINTER(C.c_void_p)]
LIB.mesh_graph_relation_quotient.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_uint32), C.c_void_p, C.c_void_p,
                                             C.POINTER(C.c_void_p)]
LIB.mesh_graph_relation_synchronization.restype = C.c_int
LIB.mesh_graph_relation_synchronization.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_ubyte)),
                                                    C.POINTER(C.c_size_t)]
LIB.mesh_graph_relation_retain.restype = None
LIB.mesh_graph_relation_retain.argtypes = [C.c_void_p]
LIB.mesh_graph_relation_save.restype = LIB.mesh_graph_relation_load.restype = C.c_int
LIB.mesh_graph_relation_save.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
# mesh-graph.h mesh_graph_relation_load: its first operand the loading compilation's one analysis context (Context)
LIB.mesh_graph_relation_load.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p,
                                         C.POINTER(C.c_void_p)]
LIB.mesh_graph_context_create.restype = C.c_int
LIB.mesh_graph_context_create.argtypes = [C.POINTER(C.c_void_p)]
LIB.mesh_graph_context_retain.restype = LIB.mesh_graph_context_free.restype = None
LIB.mesh_graph_context_retain.argtypes = LIB.mesh_graph_context_free.argtypes = [C.c_void_p]
# ISO/IEC 9899:2018 7.22.3.4 and 7.22.3.3: a loaded relation's backing (malloc's alignment, at least the 8 bytes
# mesh_graph_relation_load requires) and its release, free, the owner callback the relations adopt
LIB.malloc.restype = C.c_void_p
LIB.malloc.argtypes = [C.c_size_t]
FREE = C.cast(LIB.free, C.c_void_p)
# the release of one native reference, then of the conflict vector the library allocated for it (None: free(NULL))
def _freed(pointer, allocated):
    LIB.mesh_graph_relation_free(pointer)
    LIB.free(allocated)


# mesh-graph.h mesh_graph_context_create, mesh_graph_context_retain and mesh_graph_context_free: one compilation's
# analysis context (one isl_ctx and its lock, refcounted), this object one reference to it, released with it
class Context:
    def __init__(self, pointer):
        self.pointer = pointer
        self.freed = weakref.finalize(self, LIB.mesh_graph_context_free, pointer)

    @classmethod
    def create(cls):
        made = C.c_void_p()
        _check(LIB.mesh_graph_context_create(C.byref(made)))
        return cls(made.value)

    # a context another owner holds (a compilation's or a loaded native result's), one more reference this Context
    # releases
    @classmethod
    def borrowed(cls, pointer):
        LIB.mesh_graph_context_retain(pointer)
        return cls(pointer)


# one retained relation: the native reference this object holds, the arrays it borrows (`held`: views, their pairs,
# conflict records, operands) and its parent relation, alive with it; the native reference released first, then the
# library's conflict vector `allocated`; `descriptor`/`data` the borrowed descriptor
class Relation:
    def __init__(self, pointer, held=(), parent=None, allocated=None):
        self.pointer, self.held, self.parent = pointer, held, parent
        self.descriptor = LIB.mesh_graph_relation_data(pointer)
        self.data = self.descriptor.contents
        self.freed = weakref.finalize(self, _freed, pointer, allocated)

    # a root over the caller's views and conflict records, borrowed for the Relation's lifetime
    @classmethod
    def create(cls, vertices, views, viewed, count, conflict, held=(), allocated=None):
        made = C.c_void_p()
        status = LIB.mesh_graph_relation_create(vertices, views, viewed, count, conflict, None, None, C.byref(made))
        if status:
            LIB.free(allocated)
            _check(status)
        return cls(made.value, (viewed, conflict, *held), None, allocated)

    # mesh_graph_relation_synchronization: the redundant mask over the edges, computed once natively
    def redundant(self):
        mask, kept = C.POINTER(C.c_ubyte)(), C.c_size_t()
        _check(LIB.mesh_graph_relation_synchronization(self.pointer, C.byref(mask), C.byref(kept)))
        return View(self, mask, range(self.data.edges), bool)

    # the distinct edges, (from, to) vertex IDs, in place
    def edges(self):
        return View(self, self.data.edge, range(self.data.edges), lambda e: (e.source, e.sink))

    # edge e's conflict indices, in place
    def members(self, e):
        return View(self, self.data.member, range(self.data.first[e], self.data.first[e + 1]), int)

    def conflict(self, k):
        return self.data.conflict[k]

    # mesh_graph_relation_save: this relation and every relation it references, as the library wrote them (its
    # allocation released with free once copied, or should the copy fail)
    def save(self):
        made, length = C.c_void_p(), C.c_size_t()
        _check(LIB.mesh_graph_relation_save(self.pointer, C.byref(made), C.byref(length)))
        try:
            return C.string_at(made, length.value)
        finally:
            LIB.free(made)

    # mesh_graph_relation_retain: a relation another owner holds (a loaded mesh_local's `relation`), one more reference
    # this Relation releases
    @classmethod
    def borrowed(cls, pointer):
        LIB.mesh_graph_relation_retain(pointer)
        return cls(pointer)

    # mesh_graph_relation_load: the same relations over one malloc backing holding `data`, adopted as their owner
    # (release free, called once the last of them, this one or another retaining it, is freed), their access maps
    # loaded into `context`, the loading compilation's one analysis Context (the loaded relations retain it natively);
    # a failed load keeps nothing and its backing is freed here once
    @classmethod
    def load(cls, data, context):
        backing = LIB.malloc(max(len(data), 1))
        if not backing:
            raise MemoryError("the loaded relation's backing")
        C.memmove(backing, data, len(data))
        made = C.c_void_p()
        status = LIB.mesh_graph_relation_load(context.pointer, backing, len(data), FREE, backing, C.byref(made))
        if status:
            LIB.free(backing)
            _check(status)
        return cls(made.value)


LIB.mesh_graph_relation_views.restype = C.c_int
LIB.mesh_graph_relation_views.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(Viewed), C.c_size_t, C.POINTER(C.c_uint32),
                                          C.POINTER(C.c_uint32), C.c_int, C.c_void_p, C.c_void_p,
                                          C.POINTER(C.c_void_p)]


# mesh-graph.h mesh_graph_relation_views (self 0, query DIRECTED): the one conflict solve over the supplied views and
# the supplied causes (`supplied` their (from, to) function endpoints, GIVEN's `given` their index), its conflict
# records the relation's own; the views, their pairs and the endpoints borrowed for the Relation's lifetime
def dependences_supplied(vertices, views, supplied):
    viewed, held = (Viewed * max(len(views), 1))(), []
    for made, (key, offset, element, pairs, vertex, write) in zip(viewed, views):
        held.append((C.c_uint64 * max(2 * len(pairs), 1))(*[x for pair in pairs for x in pair]))
        made.key, made.offset, made.element, made.dims, made.pairs = key, offset, element, len(pairs), held[-1]
        made.vertex, made.write = vertex, write
    given, made = [_array(C.c_uint32, [edge[end] for edge in supplied])[0] for end in range(2)], C.c_void_p()
    _check(LIB.mesh_graph_relation_views(vertices, len(views), viewed, len(supplied), *given, 0, None, None,
                                         C.byref(made)))
    return Relation(made.value, (viewed, held, given))


def dependences_views(vertices, views):
    return dependences_supplied(vertices, views, ())


# design/torch-source-lowering-change-request.md item 3; Apple, MTL4CommandEncoder.h: `commit[v]` the commit of
# vertex v in issue order, None a host vertex
def orderings(edges, commit):
    barred, fences, waits, last = set(), {}, {}, {}
    for a, b in sorted(edges, key=lambda edge: edge[::-1]):
        if commit[b] is None:
            waits.setdefault(b, {}).setdefault(commit[a], []).append((a, b))
        elif commit[a] != commit[b]:
            fences.setdefault((commit[a], commit[b]), []).append((a, b))
        elif a >= last.get(commit[b], 0):
            barred.add(b)
            last[commit[b]] = b
    return barred, fences, waits


# design/torch-source-lowering-change-request.md item 3
def arrivals(edges, commit):
    found = {}
    for a, b in sorted(edges, key=lambda edge: edge[::-1]):
        if commit[a] is None and commit[b] is not None:
            found.setdefault(commit[b], {}).setdefault(a, []).append((a, b))
    return found


# design/torch-source-lowering-change-request.md item 3; torch/_inductor/scheduler.py additional_buffer_deps
def adjacent(positions, written, read):
    order = sorted(range(len(positions)), key=positions.__getitem__)
    return [(a, b) for a, b in zip(order, order[1:]) if written[a] and not set(written[a]) & set(read[b])]


# design/synchronization-signature-change-request.md item 1; E. A. Lee and D. G. Messerschmitt, Proc. IEEE 75(9), 1987;
# A. J. Bernstein, IEEE Trans. Electronic Computers EC-15(5), 1966
def handled(vertices, commit, accesses):
    handler = {c: vertices + k for k, c in enumerate(dict.fromkeys(c for c in commit if c is not None))}
    edges = [(v, handler[c], 0) for v, c in enumerate(commit) if c is not None]
    edges += [(a, b, 1) for held, a, first in accesses for other, b, second in accesses
              if held == other and (first or second)]
    return handler, edges
