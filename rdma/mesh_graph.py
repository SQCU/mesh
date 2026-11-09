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
def trace_machine(operations, program, metadata, objects, lanes, action_owner=None):
    controls, choices = metadata.get('controls', ()), metadata.get('choices', ())
    instructions, scopes = metadata.get('instructions', ()), metadata.get('scopes', ())
    if metadata.get('selections') and not choices:
        raise ValueError('signature has obsolete address-based choices; regenerate it from the retained compiled program')
    if instructions and action_owner is None:
        raise ValueError('retained submission instructions require their compiled owner')
    spans = metadata.get('spans') or [Span(0, 1 + max((operations[i].execution for i in program), default=0), 0, len(operations))]
    groups, last, code, children, bounds, loop_conditions = {}, {}, [], collections.defaultdict(list), {}, {}
    def token(kind, value, conditions=(), owner=0, execution=0):
        group = groups.setdefault((execution, owner), len(groups)) if owner else None
        if group is not None:
            last[group] = len(code)
        code.append((kind, value, tuple(conditions), group))
    if instructions:
        at = {i: k for k, i in enumerate(program)}
        points = collections.defaultdict(dict)
        for index, instruction in enumerate(instructions):
            conditions = [(g.choice, g.taken) for g in metadata['conditions'][instruction.condition:instruction.condition + instruction.conditions]]
            kind = instruction.kind
            if kind == 0:
                token('op' if instruction.index in at else 'skip', at.get(instruction.index), conditions,
                      instruction.once, instruction.execution)
            elif kind == 1:
                token('action', instruction.index, conditions, instruction.once, instruction.execution)
            elif kind in (6, 7):
                token('enter' if kind == 6 else 'leave', instruction.index)
            elif kind == 8:
                token('task', instruction.index, conditions, instruction.once, instruction.execution)
            else:
                points[instruction.index][kind] = index
                if kind == 3:
                    loop_conditions[instruction.index] = conditions
                token('skip', None)
        positions = [instruction.position for instruction in instructions]
        for c, control in enumerate(controls):
            bounds[c] = (tuple(points[c][kind] for kind in (2, 3, 4, 5)) if c in points else
                         tuple(bisect.bisect_left(positions, getattr(control, name)) for name in ('entry', 'body', 'retest', 'end')))
    else:
        guards = collections.defaultdict(list)
        for guard in metadata.get('guards', ()):
            k = bisect.bisect_left(program, guard.operation)
            if k < len(program) and program[k] == guard.operation:
                guards[k].append((guard.selection, guard.taken))
        for k, i in enumerate(program):
            token('op', k, guards[k], scopes[i].once if scopes else 0, operations[i].execution)
        for c, control in enumerate(controls):
            bounds[c] = tuple(bisect.bisect_left(program, getattr(control, name)) for name in ('entry', 'body', 'retest', 'end'))
    for c, control in enumerate(controls):
        children[control.parent].append(c)
    for rows in children.values():
        rows.sort(key=lambda c: bounds[c][0])
    descendants = {c: reach([c], children.__getitem__) for c in range(len(controls))}
    reset = {c: tuple(i for i, choice in enumerate(choices) if choice.parent in descendants[c]) for c in descendants}
    nodes, requirements, once, once_last = [], {}, {}, set()
    def node(kind, value=None, targets=()):
        made = len(nodes)
        nodes.append((kind, value, tuple(targets)))
        return made
    def leaf(index, tail):
        kind, value, conditions, group = code[index]
        if kind == 'skip':
            return tail
        if kind == 'task':
            tail = node('mark', value, (tail,))
        made = node(kind, value, (tail,))
        requirements[made] = conditions
        if group is not None:
            once[made] = group
            if last[group] == index:
                once_last.add(made)
        return made
    def sequence(first, end, tail, parent, selected=None):
        cursor = end
        for c in reversed(children[parent] if selected is None else selected):
            a, b, r, z = bounds[c]
            if a < first or z > cursor:
                raise ValueError('control %d is outside its retained parent interval' % c)
            for k in reversed(range(z, cursor)):
                tail = leaf(k, tail)
            if controls[c].kind == 1 and b != z:
                decision = node('decide', c)
                requirements[decision] = loop_conditions.get(c, ())
                retest = sequence(r, z, decision, c, [])
                exited = node('exit', c, (tail,))
                after = node('bound', c, (exited, retest))
                body = sequence(b, r, after, c)
                entered = node('trip', c, (body,))
                nodes[decision] = ('decide', c, (exited, entered))
                initial = sequence(a, b, decision, c, [])
                tail = node('init', c, (initial,))
            elif controls[c].kind == 2:
                tail = node('optional', c, (tail, sequence(a, z, tail, c)))
            cursor = a
        for k in reversed(range(first, cursor)):
            tail = leaf(k, tail)
        return tail
    start = node('final')
    if instructions:
        start = sequence(0, len(code), start, NONE)
    else:
        for span in reversed(range(len(spans))):
            first, end = (bisect.bisect_left(program, getattr(spans[span], name)) for name in ('begin', 'end'))
            root = [c for c in children[NONE] if first <= bounds[c][0] and bounds[c][3] <= end]
            leave = node('leave', span, (start,))
            start = node('enter', span, (sequence(first, end, leave, NONE, root),))
    uses = [set(g[0] for g in requirements.get(k, ())) for k in range(len(nodes))]
    live = [set(use) for use in uses]
    predecessors = collections.defaultdict(set)
    for n, (_, _, targets) in enumerate(nodes):
        for target in targets:
            predecessors[target].add(n)
    queue, queued = collections.deque(range(len(nodes))), set(range(len(nodes)))
    while queue:
        n = queue.popleft()
        queued.remove(n)
        kind, value, targets = nodes[n]
        future = set().union(*(live[t] for t in targets)) if targets else set()
        if kind == 'trip':
            future.difference_update(reset[value])
        made = uses[n] | future
        if made != live[n]:
            live[n] = made
            for before in predecessors[n] - queued:
                queue.append(before)
                queued.add(before)
    def normalized(state):
        cursor, seen, selected, trips, bound, queues, native = state
        return cursor, seen, tuple(v if i in live[cursor] else -1 for i, v in enumerate(selected)), trips, bound, queues, native
    def move(state, target, selected=None, seen=None, trips=None, native=None):
        return normalized((target, state[1] if seen is None else seen, state[2] if selected is None else tuple(selected),
                           state[3] if trips is None else tuple(trips), state[4], state[5], state[6] if native is None else native))
    def assign(state, needed):
        selected = list(state[2])
        for choice, taken in needed:
            if selected[choice] not in (-1, taken):
                return None
            selected[choice] = taken
        return selected
    expanded = {}
    def expansion(cursor, native):
        key = cursor, native
        if key not in expanded:
            _, action, targets = nodes[cursor]
            machine = metadata['actions'][action].submission
            changed, outputs = action_owner.step(action, native[machine])
            updated = list(native)
            updated[machine] = changed
            mask = (1 << once[cursor]) if cursor in once_last else 0
            tail = node('install', (tuple(updated), mask), targets)
            live.append(set(live[targets[0]]))
            for operation in reversed(outputs):
                tail = node('op', operation, (tail,))
                live.append(set(live[targets[0]]))
            expanded[key] = tail
        return expanded[key]
    def epsilon(state):
        cursor, seen, selected, trips, _, _, native = state
        kind, value, targets = nodes[cursor]
        if kind in ('op', 'action', 'task'):
            if cursor in once and seen & (1 << once[cursor]):
                return [move(state, targets[0])]
            completed = seen | ((1 << once[cursor]) if cursor in once_last else 0)
            result = [move(state, targets[0], selected=made, seen=completed)
                      for choice, taken in requirements.get(cursor, ())
                      if (made := assign(state, [(choice, 1 - taken)])) is not None]
            if kind in ('action', 'task') and (made := assign(state, requirements.get(cursor, ()))) is not None:
                result.append(move(state, expansion(cursor, native) if kind == 'action' else targets[0],
                                   selected=made, seen=completed if kind == 'task' else seen))
            return result
        if kind in ('mark', 'exit'):
            return [move(state, targets[0])]
        if kind == 'install':
            return [move(state, targets[0], native=value[0], seen=seen | value[1])]
        if kind == 'decide':
            selected = assign(state, requirements.get(cursor, ()))
            return [move(state, targets[0])] + ([] if selected is None else [move(state, targets[1], selected=selected)])
        if kind == 'optional':
            return [move(state, target) for target in targets]
        if kind == 'init':
            changed = list(trips)
            changed[value] = 0
            return [move(state, targets[0], trips=changed)]
        if kind == 'bound':
            reached = controls[value].bound and trips[value] >= controls[value].bound
            return [move(state, targets[0 if reached else 1])]
        if kind == 'trip':
            changed, selected = list(trips), list(selected)
            if controls[value].bound:
                changed[value] += 1
            for choice in reset[value]:
                selected[choice] = -1
            return [move(state, targets[0], selected=selected, trips=changed)]
        return []
    object_ids = sorted(set(objects.values()) | (action_owner.objects if action_owner else set()))
    lane_ids = sorted(set(lanes.values()) | (action_owner.lanes if action_owner else set()))
    object_at, lane_at = {v: k for k, v in enumerate(object_ids)}, {v: k for k, v in enumerate(lane_ids)}
    initial = normalized((start, 0, (-1,) * len(choices), (0,) * len(controls), (0,) * len(object_ids), (0,) * len(lane_ids),
                          action_owner.initial if action_owner else ()))
    states, indexed, transitions, accepting = [initial], {initial: 0}, [], []
    epsilon_graphs, endpoints, finals = {}, collections.defaultdict(set), collections.defaultdict(set)
    for source, state in enumerate(states):
        graph = {}
        def successors(current):
            if current not in graph:
                graph[current] = tuple(epsilon(current))
            return graph[current]
        for current in walk([state], successors):
            cursor, seen, selected, trips, bound, queues, native = current
            kind, value, targets = nodes[cursor]
            if kind == 'final':
                accepting.append(source)
                finals[source].add(current)
                continue
            if kind not in ('op', 'enter', 'leave'):
                continue
            if kind == 'op':
                if cursor in once and seen & (1 << once[cursor]):
                    continue
                selected = assign(current, requirements.get(cursor, ()))
                if selected is None:
                    continue
                if cursor in once_last:
                    seen |= 1 << once[cursor]
            ob, ln = 0, 0
            bound, queues = list(bound), list(queues)
            if kind == 'op' and value in objects:
                slot = object_at[objects[value]]
                bound[slot] = bound[slot] or max(bound, default=0) + 1
                ob = bound[slot]
            if kind == 'op' and value in lanes:
                slot = lane_at[lanes[value]]
                queues[slot] = queues[slot] or max(queues, default=0) + 1
                ln = queues[slot]
            following = normalized((targets[0], seen, tuple(selected), trips, tuple(bound), tuple(queues), native))
            if following not in indexed:
                indexed[following] = len(states)
                states.append(following)
            transition = (source, indexed[following], kind, value, ob, ln)
            transitions.append(transition)
            endpoints[transition[:4]].add(current)
        epsilon_graphs[source] = graph
    activations, final_tasks = {}, {}
    by_source = collections.defaultdict(list)
    for key, ends in endpoints.items():
        by_source[key[0]].append((key, ends))
    for source, graph in epsilon_graphs.items():
        order = list(graph)
        at = {v: k for k, v in enumerate(order)}
        component = components(len(order), [(at[v], at[w]) for v in order for w in graph[v]])
        counts = collections.Counter(component)
        repeated = {v for v in order if counts[component[at[v]]] > 1 or v in graph[v]}
        dom = dominators(states[source], graph.__getitem__)
        previous = collections.defaultdict(list)
        for v in order:
            for w in graph[v]:
                previous[w].append(v)
        def summary(ends):
            relevant = reach(ends, previous.__getitem__)
            required = set.intersection(*(dom[v] for v in ends))
            markers = {v for v in relevant if nodes[v[0]][0] in ('mark', 'init', 'trip', 'exit')}
            exact = sorted((markers & required) - repeated, key=lambda v: len(dom[v]))
            segments = [set() for _ in range(len(exact) + 1)]
            for v in markers - set(exact):
                before = sum(x in dom[v] for x in exact)
                segments[before].add(nodes[v[0]][:2])
            return tuple((frozenset(segments[k]), nodes[v[0]][:2]) for k, v in enumerate(exact)) + \
                   ((frozenset(segments[-1]), None),)
        for key, ends in by_source[source]:
            activations[key] = summary(ends)
        if finals[source]:
            final_tasks[source] = summary(finals[source])
    return {'states': states, 'transitions': sorted(set(transitions)), 'accepting': sorted(set(accepting)),
            'objects': object_ids, 'spans': spans, 'activations': activations, 'final_tasks': final_tasks,
            'epsilon_graphs': epsilon_graphs, 'endpoints': endpoints, 'descendants': descendants,
            'feedbacks': metadata.get('feedbacks', ()), 'controls': controls,
            'marked_tasks': {x.index for x in instructions if x.kind == 8}}


def trace_groups(machine, labels):
    groups = {}
    for source, target, kind, index, ob, lane in machine['transitions']:
        key = (kind, labels[(source, kind, index)], ob, lane, index - source, target - source, source // 64, target // 64)
        if key not in groups:
            groups[key] = [source, target, kind, index, ob, lane, 0]
        groups[key][-1] |= 1 << (source % 64)
    return list(groups.values())


def trace_witness(records, accepting):
    at = collections.defaultdict(list)
    for record in records:
        at[record[0]].append(record)
    steps = max(at, default=-1) + 1
    target = next((state for count, state in accepting if count == steps), None)
    if target is None:
        raise ValueError('no accepting compiled control state at observation end')
    result = []
    for step in reversed(range(steps)):
        predecessor = next((row for row in at[step] if row[2] == target), None)
        if predecessor is None:
            raise ValueError('missing trace predecessor at step %d state %d' % (step, target))
        result.append(predecessor)
        target = predecessor[1]
    if target != 0:
        raise ValueError('trace does not start in the compiled initial state')
    return list(reversed(result))


def trace_occurrences(tasks, dependences, operations, program, machine, witness):
    marked = machine.get('marked_tasks', set())
    if not marked:
        return tasks, dependences, [operations[program[row[3]]].vertex if row[3] >= 0 else NONE for row in witness], set(), dict(enumerate(range(len(tasks))))
    made, edges, latest, vertices, symbolic, uncertain = [], [], {}, [], set(), set()
    activations = collections.Counter()
    frames, contexts, serial = {}, {}, 0
    incoming = collections.defaultdict(list)
    feedback, sources, distances, carry = collections.defaultdict(list), collections.defaultdict(set), {}, {}
    for edge in machine.get('feedbacks', ()):
        feedback[edge.sink].append(edge)
        sources[edge.control].add(edge.source)
        distances[edge.control] = max(distances.get(edge.control, 0), edge.distance)
    def boundary(control):
        frame = frames.get(control)
        snapshot = {}
        for template in sources[control]:
            source = latest.get(template)
            context = contexts.get(source, {})
            previous = context.get(control)
            if template in symbolic or (frame and frame[2]) or (previous and previous[2]):
                snapshot[template] = None
            elif frame and previous and previous[0] == frame[0]:
                parents = set(context) - machine['descendants'][control]
                if all(context[p] == frames.get(p) for p in parents):
                    snapshot[template] = source
        carry.setdefault(control, collections.deque(maxlen=distances.get(control, 1))).append(snapshot)
    for edge in dependences:
        incoming[edge.sink].append(edge)
    def compatible(vertex):
        for control, source in contexts.get(vertex, {}).items():
            current = frames.get(control)
            if current is None or source == current:
                continue
            if source[2] or current[2]:
                return None
            return False
        return True
    def activate(template, family=False):
        vertex = len(made)
        task = Task.from_buffer_copy(tasks[template])
        task.place = vertex
        made.append(task)
        if family:
            uncertain.add(vertex)
        for edge in incoming[template]:
            if edge.source in symbolic:
                uncertain.add(vertex)
            elif edge.source in latest:
                source = latest[edge.source]
                matching = compatible(source)
                if matching is None:
                    uncertain.add(vertex)
                elif matching:
                    edges.append(Dependence(source, vertex, edge.delay))
        for edge in feedback[template]:
            history = carry.get(edge.control, ())
            if edge.control not in frames or frames[edge.control][2]:
                uncertain.add(vertex)
            elif len(history) >= edge.distance and edge.source in history[-edge.distance]:
                source = history[-edge.distance][edge.source]
                if source is None:
                    uncertain.add(vertex)
                else:
                    edges.append(Dependence(source, vertex, 0))
        latest[template] = vertex
        contexts[vertex] = dict(frames)
        activations[template] += 1
        symbolic.discard(template)
    def event(kind, value, possible=False):
        nonlocal serial
        if kind == 'mark':
            if possible:
                symbolic.add(value)
            else:
                activate(value)
            return
        serial += 1
        if possible:
            frames[value] = (serial, serial, True)
            for child in machine['descendants'][value] - {value}:
                if child in frames:
                    frames[child] = (serial, serial, True)
        elif kind == 'init':
            for child in machine['descendants'][value]:
                frames.pop(child, None)
                carry.pop(child, None)
            frames[value] = (serial, 0, False)
        elif kind == 'trip':
            entry = frames.get(value)
            if entry is None:
                raise ValueError('trip has no retained control entry %d' % value)
            boundary(value)
            frames[value] = (entry[0], serial, entry[2])
            for child in machine['descendants'][value] - {value}:
                frames.pop(child, None)
                carry.pop(child, None)
        elif kind == 'exit':
            for child in machine['descendants'][value]:
                frames.pop(child, None)
                carry.pop(child, None)
    def apply(summary):
        for possible, exact in summary:
            for kind, value in possible:
                event(kind, value, possible=True)
            if exact is not None:
                event(*exact)
    for template in range(len(tasks)):
        if template not in marked:
            task = Task.from_buffer_copy(tasks[template])
            task.place = len(made)
            latest[template] = len(made)
            made.append(task)
    for row in witness:
        _, source, target, index, *_ = row
        kind, token = ('op', index) if index >= 0 else ('leave' if (-index - 1) % 2 else 'enter', (-index - 1) // 2)
        apply(machine['activations'][(source, target, kind, token)])
        template = operations[program[index]].vertex if index >= 0 else NONE
        if template in symbolic:
            activate(template, family=True)
        if template != NONE and template not in latest:
            raise ValueError('operation references task %d before any retained activation' % template)
        vertices.append(latest.get(template, NONE))
    if witness:
        apply(machine['final_tasks'].get(witness[-1][2], ()))
    for edge in dependences:
        if edge.sink not in marked:
            if edge.source in symbolic or activations[edge.source] > 1:
                uncertain.add(latest[edge.sink])
            elif edge.source in latest:
                edges.append(Dependence(latest[edge.source], latest[edge.sink], edge.delay))
    downstream = collections.defaultdict(list)
    for edge in edges:
        downstream[edge.source].append(edge.sink)
    return made, edges, vertices, reach(uncertain, downstream.__getitem__), latest


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
def dominators(start, neighbours):
    order = walk([start], neighbours)
    previous = collections.defaultdict(list)
    for v in order:
        for w in neighbours(v):
            previous[w].append(v)
    result = {v: {start} if v == start else set(order) for v in order}
    def update(_, v):
        if v == start:
            return False
        value = {v} | set.intersection(*(result[u] for u in previous[v]))
        if value == result[v]:
            return False
        result[v] = value
        return True
    propagate(order, neighbours, update, lambda _v, _w: False)
    return result


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
              'producer_barrier', 'bridge_raise', 'initial', 'host_publish', 'kernel_fence', 'host_signal', 'trip_test')
NODES, PLAN_COUNT, LOOP, PLAN_SUBMISSION = 64, 0x80000000, 0xA0000000, 0xC0000000
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


class Control(C.Structure):
    _fields_ = [('kind', C.c_uint32), ('parent', C.c_uint32)] + [(n, C.c_uint64) for n in ('entry', 'body', 'retest', 'end', 'execution', 'bound')]


class Event(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('object', 'role', 'counter', 'argument')] + [('lookback', C.c_uint64)]


class Scope(C.Structure):
    _fields_ = [('once', C.c_uint32), ('counter', C.c_uint32), ('delta', C.c_int64)]


class Span(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('first', 'count', 'begin', 'end')]


class Selection(C.Structure):
    _fields_ = [('argument', C.c_uint32), ('bytes', C.c_uint32), ('table', C.c_uint64), ('index', C.c_uint64)]


class Guard(C.Structure):
    _fields_ = [('operation', C.c_uint64), ('selection', C.c_uint32), ('taken', C.c_uint32)]


class Choice(C.Structure):
    _fields_ = [('parent', C.c_uint32), ('control', C.c_uint32), ('execution', C.c_uint64), ('call', C.c_uint64),
                ('entry', C.c_uint32), ('trip', C.c_uint32)]


class Submission(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('lag', 'positions', 'window', 'flags', 'queue', 'host', 'drained', 'ended', 'progress', 'padding')] + [('policy', C.c_ubyte * 16)]


class Action(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('kind', 'submission', 'object', 'primitive', 'vertex', 'provenance')] + \
               [(n, C.c_uint64) for n in ('execution', 'value', 'begin', 'end')]


LIB.mesh_graph_action_owner.argtypes = [C.POINTER(Action), C.POINTER(C.c_uint32), C.POINTER(C.c_uint64), C.POINTER(Action)]
LIB.mesh_graph_action_owner.restype = None


class Instruction(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('kind', 'index', 'once', 'condition', 'conditions', 'padding')] + \
               [(n, C.c_uint64) for n in ('execution', 'position')]


class Condition(C.Structure):
    _fields_ = [('choice', C.c_uint32), ('taken', C.c_uint32)]


class Feedback(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('control', 'source', 'sink', 'distance')]


class Signature(C.Structure):
    _fields_ = [('processors', C.c_uint32), ('tasks', C.c_uint32), ('operations', C.c_size_t),
                ('dependences', C.c_size_t), ('synchronized', C.c_size_t), ('uncovered', C.c_size_t),
                ('redundant', C.c_size_t), ('unwaited', C.c_size_t), ('states', C.c_size_t * len(STATES)),
                ('processor', C.POINTER(Processor)), ('task', C.POINTER(Task)), ('operation', C.POINTER(Operation)),
                ('dependence', C.POINTER(Dependence)), ('edge', C.POINTER(Synchronized)),
                ('position', C.POINTER(C.c_uint64)), ('count', C.POINTER(C.c_uint64)),
                ('least', C.POINTER(C.c_uint64)), ('previous', C.POINTER(C.c_size_t)),
                ('waited', C.POINTER(C.c_size_t)), ('first', C.POINTER(C.c_size_t)), ('order', C.POINTER(C.c_size_t)),
                ('unknown', C.POINTER(C.c_uint64)), ('alias', C.POINTER(C.c_uint32)), ('digest', C.c_ubyte * 32), ('controls', C.c_size_t), ('events', C.c_size_t),
                ('control', C.POINTER(Control)), ('event', C.POINTER(Event)), ('scope', C.POINTER(Scope)),
                ('spans', C.c_size_t), ('span', C.POINTER(Span)), ('selections', C.c_size_t), ('guards', C.c_size_t),
                ('selection', C.POINTER(Selection)), ('guard', C.POINTER(Guard)), ('choices', C.c_size_t),
                ('choice', C.POINTER(Choice)), ('submissions', C.c_size_t), ('actions', C.c_size_t),
                ('instructions', C.c_size_t), ('conditions', C.c_size_t), ('submission', C.POINTER(Submission)),
                ('action', C.POINTER(Action)), ('instruction', C.POINTER(Instruction)), ('condition', C.POINTER(Condition)),
                ('feedbacks', C.c_size_t), ('feedback', C.POINTER(Feedback))]


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
