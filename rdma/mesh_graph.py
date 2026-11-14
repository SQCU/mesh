# design/rubrics.md C9
import bisect
import collections
import ctypes as C
import errno
import fractions
import math
import os


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


# design/unequal-shards-change-request.md item 8
def replicas(order, users, replicated, place):
    onto = {}
    for v in reversed(order):
        if replicated(v):
            onto[v] = {q for u in users(v) for q in {place(u)} | onto.get(u, set())} - {place(v)}
    return {v: sorted(found) for v, found in onto.items() if found}


# design/unequal-shards-change-request.md item 8
def chain(start, following):
    path = [start]
    while (after := following(path[-1])) is not None:
        path.append(after)
    return path


# A. Goralčíková and V. Koubek, LNCS 74, 1979; K. Simon, TCS 58, 1988; design/aot-compilation-change-request.md A10
def closure(order, neighbours):
    place, reached = {}, {}
    for v in order:
        place[v], reached[v] = len(place), set()
        for u in sorted(set(neighbours(v)), key=place.__getitem__, reverse=True):
            if u not in reached[v]:
                reached[v] |= {u} | reached[u]
    return reached


# A. V. Aho, M. R. Garey and J. D. Ullman, The transitive reduction of a directed graph, SIAM J. Comput. 1(2), 1972
def reduction(order, neighbours):
    reached = closure(order, neighbours)
    return {v: [u for u in dict.fromkeys(neighbours(v)) if not any(u in reached[w] for w in neighbours(v))]
            for v in order}


# F. Harary, Graph Theory, 1969
def extremes(vertices, neighbours):
    entered = {u for v in vertices for u in neighbours(v)}
    return [v for v in vertices if v not in entered], [v for v in vertices if not neighbours(v)]


# L. Lamport, CACM 21(7), 1978; M. Lam, PLDI 1988, section 2.3; design/disjoint-pages-change-request.md C3.3 and C3.5
def precedence(events, place, thread, reached):
    def precedes(a, b):
        return a != b and place[a] == place[b] and all(x in reached[y] for x in events[a] for y in events[b])

    after = {a: [b for b in range(len(events)) if precedes(a, b)] for a in range(len(events))}
    covering = reduction(postorder(list(after), after.__getitem__), after.__getitem__)
    found = [(a, b, 0) for a in covering for b in covering[a]]
    for one in dict.fromkeys(thread):
        held = [a for a in covering if thread[a] == one]
        firsts, lasts = extremes(held, lambda a: [b for b in covering[a] if b in held])
        found += [(a, b, 1) for a in lasts for b in firsts]
    return found


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


# R. E. Tarjan, SIAM J. Comput. 1(2), 1972
def postorder(vertices, neighbours):
    order, seen, stack = [], set(), list(vertices)
    while stack:
        v = stack[-1]
        unseen = [u for u in neighbours(v) if u not in seen]
        if unseen:
            stack.extend(unseen)
        elif stack.pop() not in seen:
            seen.add(v)
            order.append(v)
    return order


# S. Zhang et al., HAP, EuroSys 2024, section 3.2; design/unequal-shards-change-request.md item 6
def longest(vertices, predecessors, weight):
    def length(v, lengths):
        return max((at + weight(u, v) for u, at in zip(predecessors(v), lengths)), default=0)
    return forward(postorder(vertices, predecessors), predecessors, length)


# A. Pınar, E. K. Tabak and C. Aykanat, JPDC 68(11), 2008, section 4, LR-PROBE;
# design/unequal-shards-change-request.md item 13
def probe(offsets, speeds, load):
    cuts = [0]
    for speed in speeds:
        cuts.append(bisect.bisect_right(offsets, offsets[cuts[-1]] + load * speed) - 1)
    return cuts


# A. Pınar, E. K. Tabak and C. Aykanat, JPDC 68(11), 2008, sections 2, 3.3 and 3.3.1;
# design/unequal-shards-change-request.md item 13
def separators(offsets, speeds):
    rows, speeds = len(offsets) - 1, [fractions.Fraction(speed) for speed in speeds]
    succeeded, failed, first = (offsets[rows] - offsets[0]) / speeds[0], -1, 0
    for speed in speeds:
        low, high = first + 1, rows + 1
        while low < high:
            end = (low + high) // 2
            load = (offsets[end] - offsets[first]) / speed
            if failed < load < succeeded:
                if probe(offsets, speeds, load)[-1] == rows:
                    succeeded = load
                else:
                    failed = load
            low, high = (low, end) if load >= succeeded else (end + 1, high)
        first = low - 1
    return probe(offsets, speeds, succeeded)


# L. Lamport, CACM 21(7), 1978; design/disjoint-pages-change-request.md A3 item 2
def ends(order, reached, uses):
    readers = {}
    for v in order:
        for u in uses(v):
            readers.setdefault(u, []).append(v)
    return {u: next(x for x in reversed(order) if not all(r in reached[x] for r in found))
            for u, found in readers.items()}


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


LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
NONE = 2 ** 32 - 1
KINDS = ('queue', 'host', 'bridge', 'kernel')
PRIMITIVES = ('submit', 'queue_signal', 'queue_wait', 'fence_update', 'fence_wait', 'pass_barrier', 'queue_barrier',
              'host_wait', 'host_notify', 'bridge_start', 'threadgroup_barrier', 'simdgroup_barrier', 'atomic',
              'producer_barrier', 'bridge_raise', 'initial', 'host_publish', 'kernel_fence', 'host_signal')
NODES, PLAN_COUNT, PLAN_SUBMISSION, LOWERED = 64, 0x80000000, 0xC0000000, 0x40000000
STATES = ('image', 'stronger', 'false', 'unmatched', 'satisfied', 'transformed')


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
                ('proof', C.POINTER(Proof))]


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


# BSL §3; R. E. Tarjan, SIAM J. Comput. 1(2), 1972; mesh-graph.h mesh_graph_components: each vertex's strongly
# connected component
LIB.mesh_graph_components.restype = C.c_int
LIB.mesh_graph_components.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
                                      C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]


def components(vertices, edges):
    tails, heads = _array(C.c_uint32, [a for a, _ in edges]), _array(C.c_uint32, [b for _, b in edges])
    component, count = (C.c_uint32 * max(vertices, 1))(), C.c_uint32()
    _check(LIB.mesh_graph_components(vertices, len(edges), tails[0], heads[0], component, C.byref(count)))
    return list(component[:vertices])


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


def from_record(data):
    buffer, made = C.create_string_buffer(bytes(data), len(data)), Signature()
    _check(LIB.mesh_graph_record_read(C.addressof(buffer), C.byref(made)))
    return made


# mesh-graph.h mesh_graph_archive, mesh_graph_archive_read and mesh_graph_compose;
# design/compilation-composition-change-request.md items 3 and 6
LIB.mesh_graph_archive.restype = LIB.mesh_graph_archive_read.restype = LIB.mesh_graph_compose.restype = C.c_int
LIB.mesh_graph_archive.argtypes = [C.POINTER(Signature), C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
LIB.mesh_graph_archive_read.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Signature)]
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
    buffer, made = C.create_string_buffer(bytes(data), len(data)), Signature()
    _check(LIB.mesh_graph_archive_read(C.addressof(buffer), len(data), C.byref(made)))
    return made


def compose(a, b):
    made = Signature()
    _check(LIB.mesh_graph_compose(C.byref(a), C.byref(b), C.byref(made)))
    return made


# mesh-graph.h mesh_graph_signature_span; design/synchronization-signature-change-request.md item 1
LIB.mesh_graph_signature_span.restype = C.c_int
LIB.mesh_graph_signature_span.argtypes = [C.POINTER(Signature), C.c_uint64, C.POINTER(Signature)]


def signature_span(made, executions):
    spanned = Signature()
    _check(LIB.mesh_graph_signature_span(C.byref(made), executions, C.byref(spanned)))
    return spanned


# design/compilation-composition-change-request.md item 6.2; mesh-graph.h mesh_graph_contract, mesh_graph_quotient
LIB.mesh_graph_contract.restype = LIB.mesh_graph_quotient.restype = C.c_int
LIB.mesh_graph_contract.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(Dependence), C.POINTER(C.c_ubyte),
                                    C.POINTER(C.POINTER(Dependence)), C.POINTER(C.c_size_t)]
LIB.mesh_graph_quotient.argtypes = [C.c_uint32, C.POINTER(C.c_uint32), C.c_size_t, C.POINTER(C.c_uint32),
                                    C.POINTER(C.c_uint32), C.POINTER(C.POINTER(Dependence)), C.POINTER(C.c_size_t)]
# ISO/IEC 9899:2018 7.22.3.3: the release of a relation the library allocated
LIB.free.restype = None
LIB.free.argtypes = [C.c_void_p]


def _relation(call):
    made, count = C.POINTER(Dependence)(), C.c_size_t()
    _check(call(C.byref(made), C.byref(count)))
    try:
        return [(e.source, e.sink) for e in made[:count.value]]
    finally:
        LIB.free(made)


def contract(kept, edges):
    edge, count = _array(Dependence, [Dependence(a, b, 0) for a, b in edges])
    marks = (C.c_ubyte * max(len(kept), 1))(*kept)
    return _relation(lambda made, size: LIB.mesh_graph_contract(len(kept), count, edge, marks, made, size))


def quotient(class_of, edges):
    tails, heads = _array(C.c_uint32, [a for a, _ in edges]), _array(C.c_uint32, [b for _, b in edges])
    classes = (C.c_uint32 * max(len(class_of), 1))(*class_of)
    return _relation(lambda made, size: LIB.mesh_graph_quotient(len(class_of), classes, tails[1], tails[0], heads[0],
                                                                made, size))


# mesh-graph.h struct mesh_graph_view (its struct mesh_strided in place) and struct mesh_graph_conflict,
# mesh_graph_dependences_views and mesh_graph_synchronization; design/compilation-composition-change-request.md
# item 6 and "Compact effects"
class Viewed(C.Structure):
    _fields_ = [('key', C.c_uint64), ('offset', C.c_uint64), ('element', C.c_uint64), ('dims', C.c_uint32),
                ('padding', C.c_uint32), ('pairs', C.POINTER(C.c_uint64)), ('vertex', C.c_uint32), ('write', C.c_uint32)]


class Conflict(C.Structure):
    _fields_ = [('source', C.c_uint32), ('sink', C.c_uint32), ('kind', C.c_uint32), ('given', C.c_uint32),
                ('key', C.c_uint64), ('from_view', C.c_size_t), ('to_view', C.c_size_t)]


HAZARDS = ('read after write', 'write after read', 'write after write', 'given')
LIB.mesh_graph_dependences_views.restype = LIB.mesh_graph_synchronization.restype = C.c_int
LIB.mesh_graph_dependences_views.argtypes = [
    C.c_size_t, C.POINTER(Viewed), C.c_size_t, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
    C.POINTER(C.POINTER(Conflict)), C.POINTER(C.c_size_t)]
LIB.mesh_graph_synchronization.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
                                           C.POINTER(C.c_ubyte), C.POINTER(C.c_size_t)]


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


def dependences_supplied(vertices, views, supplied):
    viewed, held = (Viewed * max(len(views), 1))(), []
    for made, (key, offset, element, pairs, vertex, write) in zip(viewed, views):
        held.append((C.c_uint64 * max(2 * len(pairs), 1))(*[x for pair in pairs for x in pair]))
        made.key, made.offset, made.element, made.dims, made.pairs = key, offset, element, len(pairs), held[-1]
        made.vertex, made.write = vertex, write
    given = [_array(C.c_uint32, [edge[end] for edge in supplied])[0] for end in range(2)]
    conflict, count, found = C.POINTER(Conflict)(), C.c_size_t(), {}
    _check(LIB.mesh_graph_dependences_views(len(views), viewed, len(supplied), *given, C.byref(conflict),
                                            C.byref(count)))
    try:
        for cause in conflict[:count.value]:
            found.setdefault((cause.source, cause.sink), {})[cause.kind, cause.key, cause.given] = (
                HAZARDS[cause.kind], cause.key)
    finally:
        LIB.free(conflict)
    found = {edge: list(held.values()) for edge, held in found.items()}
    edges, room = list(found), max(len(found), 1)
    redundant, kept = (C.c_ubyte * room)(), C.c_size_t()
    _check(LIB.mesh_graph_synchronization(vertices, len(edges), (C.c_uint32 * room)(*[a for a, _ in edges]),
                                          (C.c_uint32 * room)(*[b for _, b in edges]), redundant, C.byref(kept)))
    return edges, found, [bool(r) for r in redundant[:len(edges)]]


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


# M. Lam, PLDI 1988, section 2.3; mesh-graph.h mesh_graph_contract_unrolled
LIB.mesh_graph_contract_unrolled.restype = C.c_int
LIB.mesh_graph_contract_unrolled.argtypes = [C.c_uint32, C.c_size_t, C.POINTER(Dependence), C.POINTER(C.c_ubyte),
                                             C.c_uint32, C.POINTER(C.POINTER(Dependence)), C.POINTER(C.c_size_t)]


def unrolled(vertices, edges, copies):
    edge, count = _array(Dependence, [Dependence(a, b, delay) for a, b, delay in edges])
    kept = (C.c_ubyte * max(vertices, 1))(*[1] * vertices)
    return _relation(lambda made, size: LIB.mesh_graph_contract_unrolled(vertices, count, edge, kept, copies, made,
                                                                         size))


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
