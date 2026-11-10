# design/rubrics.md C9
import bisect
import collections
import ctypes as C
import errno
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


# Rabin and Scott, IBM JRD 3(2), 1959, Definition 11 and Theorem 11; design/synchronization-signature-change-request.md 3–4.
def trace_machine(operations, program, metadata, objects, lanes):
    spans = metadata.get('spans') or [Span(0, 1 + max((operations[i].execution for i in program), default=0), 0, len(operations))]
    code, at = [], 0
    for span in range(len(spans)):
        first, end = (bisect.bisect_left(program, getattr(spans[span], name)) for name in ('begin', 'end'))
        code += [('op', k) for k in range(at, first)] + [('enter', span)] + [('op', k) for k in range(first, end)] + \
                [('leave', span)]
        at = end
    code += [('op', k) for k in range(at, len(program))]
    object_ids, lane_ids = sorted(set(objects.values())), sorted(set(lanes.values()))
    object_at, lane_at = {v: k for k, v in enumerate(object_ids)}, {v: k for k, v in enumerate(lane_ids)}
    bound, queues, transitions = [0] * len(object_ids), [0] * len(lane_ids), []
    for source, (kind, value) in enumerate(code):
        ob, ln = 0, 0
        if kind == 'op' and value in objects:
            slot = object_at[objects[value]]
            bound[slot] = bound[slot] or max(bound, default=0) + 1
            ob = bound[slot]
        if kind == 'op' and value in lanes:
            slot = lane_at[lanes[value]]
            queues[slot] = queues[slot] or max(queues, default=0) + 1
            ln = queues[slot]
        transitions.append((source, source + 1, kind, value, ob, ln))
    return {'transitions': transitions, 'accepting': [len(code)], 'objects': object_ids, 'spans': spans}


def trace_groups(machine, labels):
    groups = {}
    for source, target, kind, index, ob, lane in machine['transitions']:
        key = (kind, labels[(source, kind, index)], ob, lane, index - source, target - source, source // 64, target // 64)
        if key not in groups:
            groups[key] = [source, target, kind, index, ob, lane, 0]
        groups[key][-1] |= 1 << (source % 64)
    return list(groups.values())


def trace_matches(machine, labels):
    groups = collections.defaultdict(list)
    for row in trace_groups(machine, labels):
        source, _, kind, index, ob, lane, _ = row
        groups[(kind, labels[source, kind, index], ob, lane)].append(row)
    result = []
    for key, rows in groups.items():
        words = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        for source, target, _, _, _, _, mask in rows:
            words[source // 64][target // 64][target % 64 - source % 64] |= mask
        result.append((key, rows, words))
    return result


def trace_paths(records, accepting):
    at = collections.defaultdict(list)
    for record in records:
        at[record[0]].append(record)
    steps = max(at, default=-1) + 1
    finals = {state for count, state in accepting if count == steps}
    if not finals:
        raise ValueError('no accepting compiled state at observation end')
    targets, layers = finals, [None] * steps
    for step in reversed(range(steps)):
        layers[step] = [row for row in at[step] if row[2] in targets]
        targets = {row[1] for row in layers[step]}
    if 0 not in targets:
        raise ValueError('trace does not start in the compiled initial state')
    reached = {0}
    for step in range(steps):
        layers[step] = [row for row in layers[step] if row[1] in reached]
        reached = {row[2] for row in layers[step]}
    finals &= reached
    target, witness = min(finals), []
    for layer in reversed(layers):
        row = next(row for row in layer if row[2] == target)
        witness.append(row)
        target = row[1]
    return list(reversed(witness)), layers, finals


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


LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
NONE = 2 ** 32 - 1
KINDS = ('queue', 'host', 'bridge', 'kernel')
PRIMITIVES = ('submit', 'queue_signal', 'queue_wait', 'fence_update', 'fence_wait', 'pass_barrier', 'queue_barrier',
              'host_wait', 'host_notify', 'bridge_start', 'threadgroup_barrier', 'simdgroup_barrier', 'atomic',
              'producer_barrier', 'bridge_raise', 'initial', 'host_publish', 'kernel_fence', 'host_signal')
NODES, PLAN_COUNT, PLAN_SUBMISSION = 64, 0x80000000, 0xC0000000
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
                ('event', C.POINTER(Event)), ('spans', C.c_size_t), ('span', C.POINTER(Span))]


PRINTERS = ('mesh_graph_matrix_market', 'mesh_graph_dot', 'mesh_graph_space_time')
LIB.mesh_graph_signature.restype = C.c_int
LIB.mesh_graph_signature.argtypes = [C.c_uint32, C.POINTER(Processor), C.c_uint32, C.POINTER(Task), C.c_size_t,
                                     C.POINTER(Dependence), C.c_size_t, C.POINTER(Operation), C.POINTER(Signature)]
LIB.mesh_graph_signature_free.restype = None
LIB.mesh_graph_signature_free.argtypes = [C.POINTER(Signature)]
for name in PRINTERS:
    getattr(LIB, name).restype = C.c_int
    getattr(LIB, name).argtypes = [C.POINTER(Signature), C.c_char_p, C.c_size_t, C.POINTER(C.c_size_t)]
LIB.mesh_graph_heap_order.restype = C.c_int
LIB.mesh_graph_heap_order.argtypes = [C.c_uint32, C.POINTER(C.c_double), C.c_size_t, C.POINTER(C.c_uint32),
                                      C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]


def _check(status):
    if status:
        raise OSError(status, os.strerror(status))


def _array(kind, items):
    items = list(items)
    return (kind * max(len(items), 1))(*items), len(items)


# M. Cowan et al., MSCCLang, ASPLOS 2023, section 5.2, step 3; mesh-graph.h mesh_graph_heap_order
def heap_order(priority, edges):
    count = len(priority)
    tails, heads = _array(C.c_uint32, [a for a, _ in edges]), _array(C.c_uint32, [b for _, b in edges])
    place = (C.c_uint32 * max(count, 1))()
    _check(LIB.mesh_graph_heap_order(count, (C.c_double * max(count, 1))(*priority), len(edges), tails[0], heads[0],
                                     place))
    return list(place[:count])


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


# design/synchronization-signature-change-request.md items 1, 2, 4 and 6; mesh-graph.h mesh_graph_signature
def signature(processors, tasks, dependences, operations):
    made = Signature()
    processor, p = _array(Processor, processors)
    task, t = _array(Task, tasks)
    dependence, d = _array(Dependence, dependences)
    operation, o = _array(Operation, operations)
    _check(LIB.mesh_graph_signature(p, processor, t, task, d, dependence, o, operation, C.byref(made)))
    try:
        return forms(made)
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
