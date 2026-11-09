#!/usr/bin/env mesh-python
"""design/synchronization-signature-change-request.md items 3, 4 and 5"""
import bisect
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rdma'))
import mesh_graph

REVIEWS = os.path.join(HERE, '..', '..', '..', 'metal-microbench', 'output_data', 'change-request-reviews.jsonl')
KINDS, PRIMITIVES = mesh_graph.KINDS, mesh_graph.PRIMITIVES
NONE = 0xFFFFFFFF
ROLES = {'submit': 'queue', 'queue_signal': 'queue', 'queue_wait': 'queue', 'fence_update': 'queue', 'fence_wait': 'queue',
         'pass_barrier': 'queue', 'queue_barrier': 'queue', 'producer_barrier': 'queue', 'host_notify': 'listener'}
# the program thread's synchronization calls: primitive, probe, the call's object, its value, its queue (D expressions)
PROBES = [
    ('submit', 'objc$target::-commit?count?options?:', None, 'arg3', 'arg0'),
    ('queue_signal', 'objc$target::-signalEvent?value?:', 'arg2', 'arg3', 'arg0'),
    ('queue_wait', 'objc$target::-waitForEvent?value?:', 'arg2', 'arg3', 'arg0'),
    ('fence_update', 'objc$target::-updateFence?afterEncoderStages?:', 'arg2', None, None),
    ('fence_wait', 'objc$target::-waitForFence?beforeEncoderStages?:', 'arg2', None, None),
    ('pass_barrier', 'objc$target::-barrierAfterEncoderStages?beforeEncoderStages?visibilityOptions?:', None, None, None),
    ('queue_barrier', 'objc$target::-barrierAfterQueueStages?beforeStages?visibilityOptions?:', None, None, None),
    ('producer_barrier', 'objc$target::-barrierAfterStages?beforeQueueStages?visibilityOptions?:', None, None, None),
    ('host_wait', 'pid$target::MetalEventWait:', 'arg0', 'arg1', None),
    ('host_notify', 'objc$target::-notifyListener?atValue?block?:', 'arg0', 'arg3', None),
    ('host_publish', 'pid$target::mesh_host_publish:', '(arg1 << 20) + arg2', 'arg3 + 1', None),
    ('host_signal', 'objc$target::-setSignaledValue?:', 'arg0', 'arg2', None),
    ('host_wait', 'pid$target::MetalNativeWait:', '(arg2 << 20) + *(uint32_t *)copyin(arg3 + 4, 4)', 'arg4 + 1', None),
    ('host_wait', 'pid$target::MetalSentWait:', 'arg0', 'arg1 + 1', None),
]
# mesh-dataflow.c mesh_transfers_start: the submission, checked at its return (after the plan's count waits it holds)
START = ('bridge_start', 'pid$target::mesh_transfers_start:', 'arg1', 'arg2')
UNLISTED = [
    'objc$target::-waitUntilCompleted:entry',
    'objc$target::-waitUntilScheduled:entry',
    'objc$target::-waitUntilSignaledValue?timeoutMS?:entry',
    'pid$target::dispatch_semaphore_wait:entry',
    'pid$target::os_sync_wait_on_address:entry',
    'pid$target::os_sync_wait_on_address_with_timeout:entry',
    'pid$target::pthread_cond_wait:entry',
    'pid$target::pthread_cond_timedwait:entry',
    'pid$target::nanosleep:entry',
    'pid$target::usleep:entry',
]
# mesh-bridge.h enum COUNT, struct mesh_count (group at 0, count at 40, node/sink/function/entry at 48..60)
COUNT_KIND = 9
# where each compiled operation is established: `program` by this node's program-thread probes in the record's issue
# order, `static` by the compiled kernels' IR (item 6), `bridge` by this node's bridge raises (read-only, --bridge),
# `implied` the plan's count waits held inside mesh_transfers_start, `initial` the plan's cleared count words,
# `elsewhere` another node's processors (that node's monitor)
CLASSES = ('program', 'static', 'bridge', 'implied', 'initial', 'elsewhere')


LIBMESH = os.path.join(HERE, '..', 'rdma', 'libmesh.dylib')


# mesh-dataflow.c mesh_transfers_start's wait for each edge of its row, as the built libmesh compiled it: the block an
# edge's test branches to once its count has reached its need (`cmp seen, need`; `ngcs xzr, need_high`; `b.ge`), whose
# first instruction steps the edge index (`add edge, edge, #0x1`). A pid probe at that instruction's offset observes
# each satisfied wait with its edge, count seen and need in registers, before the submission's release store
def count_wait_point(library):
    listing = subprocess.run(['otool', '-tV', '-p', '_mesh_transfers_start', library], capture_output=True, text=True).stdout
    code, started = [], False
    for line in listing.splitlines():
        if line.startswith('_mesh_transfers_start:'):
            started = True
            continue
        if started and re.match(r'^_\w+:', line):
            break
        m = re.match(r'^([0-9a-f]+)\t(\S+)\t?(.*)$', line) if started else None
        if m:
            code.append((int(m.group(1), 16), m.group(2), m.group(3).split(';')[0].strip()))
    at = {address: (op, args) for address, op, args in code}
    found = set()
    for k in range(len(code) - 2):
        a, b, c = code[k], code[k + 1], code[k + 2]
        test = re.match(r'^x(\d+), x(\d+)$', a[2]) if a[1] == 'cmp' else None
        high = re.match(r'^xzr, x(\d+)$', b[2]) if b[1] == 'ngcs' else None
        if not (test and high and c[1] in ('b.ge', 'b.hs')):
            continue
        target = int(c[2], 16)
        step = re.match(r'^x(\d+), x(\d+), #0x1$', at.get(target, ('', ''))[1]) if at.get(target, ('',))[0] == 'add' else None
        if step and step.group(1) == step.group(2):
            found.add((target - code[0][0], int(step.group(1)), int(test.group(1)), int(test.group(2)), int(high.group(1))))
    if len(found) != 1:
        return None, '%d candidate wait points in %s mesh_transfers_start' % (len(found), library)
    offset, edge, seen, need, high = found.pop()
    digest = hashlib.sha256(open(library, 'rb').read()).hexdigest()
    return {'offset': offset, 'edge': edge, 'seen': seen, 'need': need, 'high': high, 'library': library,
            'sha256': digest}, None


def load(path):
    made = mesh_graph.from_record(open(path, 'rb').read())
    try:
        compiled = mesh_graph.forms(made)
        processors = [mesh_graph.Processor.from_buffer_copy(p) for p in made.processor[:made.processors]]
        tasks = [mesh_graph.Task.from_buffer_copy(t) for t in made.task[:made.tasks]]
        dependences = [mesh_graph.Dependence.from_buffer_copy(d) for d in made.dependence[:made.dependences]]
        operations = [mesh_graph.Operation.from_buffer_copy(o) for o in made.operation[:made.operations]]
        lane = {p: [made.order[k] for k in range(made.first[p], made.first[p + 1])] for p in range(made.processors)}
        metadata = {'controls': [mesh_graph.Control.from_buffer_copy(c) for c in made.control[:made.controls]],
                    'events': [mesh_graph.Event.from_buffer_copy(e) for e in made.event[:made.events]],
                    'spans': [mesh_graph.Span.from_buffer_copy(x) for x in made.span[:made.spans]],
                    'choices': [mesh_graph.Choice.from_buffer_copy(x) for x in made.choice[:made.choices]],
                    'selections': [mesh_graph.Selection.from_buffer_copy(x) for x in made.selection[:made.selections]],
                    'guards': [mesh_graph.Guard.from_buffer_copy(x) for x in made.guard[:made.guards]],
                    'scopes': [mesh_graph.Scope.from_buffer_copy(x) for x in made.scope[:made.operations]] if made.scope else []}
    finally:
        mesh_graph.free(made)
    return compiled, processors, tasks, dependences, operations, lane, metadata


def classify(processors, operations, waits=True):
    queues = {p.index % mesh_graph.NODES for p in processors if KINDS[p.kind] == 'queue'}
    if len(queues) > 1:
        raise ValueError('the record holds queues of nodes %s: one node\'s program a record' % sorted(queues))
    node = queues.pop() if queues else None

    def which(o):
        p, name = processors[o.processor], PRIMITIVES[o.primitive]
        if KINDS[p.kind] == 'kernel':
            return 'static'
        if name == 'initial':
            return 'initial'
        if p.index % mesh_graph.NODES != node:
            return 'elsewhere'
        if KINDS[p.kind] == 'bridge':
            return 'bridge'
        if name == 'host_wait' and mesh_graph.PLAN_COUNT <= o.object < mesh_graph.PLAN_SUBMISSION and not waits:
            return 'implied'
        return 'program'
    return node, [which(o) for o in operations]


def role(processors, node, name):
    want = ROLES.get(name, 'host')
    for q, p in enumerate(processors):
        if p.index % mesh_graph.NODES != node:
            continue
        if want == 'queue' and KINDS[p.kind] == 'queue' or want == 'host' and KINDS[p.kind] == 'host' and \
                p.index < mesh_graph.NODES or want == 'listener' and KINDS[p.kind] == 'host' and p.index >= mesh_graph.NODES:
            return q
    raise ValueError('no %s processor of node %s in the record' % (want, node))


def counter_rules(operations, program, scopes=()):
    return [(scopes[i].counter, scopes[i].delta) if scopes else (0, 0) for i in program]


def generate(operations, program, observe, point=None, metadata=None):
    metadata = metadata or {}
    completion = next((e for e in metadata.get('events', ()) if e.role == 1), None)
    slots = max(1, completion.lookback) if completion else 1
    object_primitives = {PRIMITIVES.index(name) for name, _, obj, _, _ in PROBES if obj is not None}
    lane_primitives = {PRIMITIVES.index(name) for name, _, _, _, lane in PROBES if lane is not None}
    object_primitives.update((PRIMITIVES.index('trip_test'), PRIMITIVES.index('bridge_start')))
    objects = {k: operations[i].object for k, i in enumerate(program)
               if operations[i].primitive in object_primitives and (not completion or operations[i].object != completion.object)}
    lanes = {k: operations[i].processor for k, i in enumerate(program) if operations[i].primitive in lane_primitives}
    machine = mesh_graph.trace_machine(operations, program, metadata, objects, lanes)
    rules = counter_rules(operations, program, metadata.get('scopes', ()))
    family_objects = collections.defaultdict(set)
    for event in metadata.get('events', ()):
        if event.counter:
            family_objects[event.counter].add(event.object)
    spans, transitions = machine['spans'], machine['transitions']
    act = 'printf("mesh-sync-monitor: failure target pid %d\\n", pid); ustack(); ' + (
        'exit(1);' if observe else 'stop(); exit(1);')
    lines = ['#pragma D option quiet', '#pragma D option strsize=512', '#pragma D option bufsize=64m',
             '#pragma D option dynvarsize=64m'] + ([] if observe else ['#pragma D option destructive'])
    lines += ['self int inrun, inapi, bs;', 'self uint64_t exec, ended, bsob, bsva, spanfirst, spanend;',
              'uint64_t epoch, invocation, objectcount, lanecount;',
              'uint64_t active[uint64_t, uint64_t], om[uint64_t], lm[uint64_t], signals[uint64_t];',
              'uint64_t spanEvent[uint64_t], spanFirst[uint64_t], spanEnd[uint64_t];',
              'BEGIN { epoch = 1; active[1, 0] = 1; }']
    trip, wait, signal = (PRIMITIVES.index(name) for name in ('trip_test', 'host_wait', 'queue_signal'))

    def check(probe, point, predicate, kind, name=None, obj=None, val=None, lane=None):
        at = probe + point
        primitive = PRIMITIVES.index(name) if name else -1
        lines.append('%s { this->checked = 0; }' % at)
        lines.append('%s %s { this->checked = 1; this->ok = 0; this->ordinary = 0; '
                     'this->ob = %s; this->va = %s; this->ln = %s; this->completion = 0; }' %
                     (at, predicate, obj or '0', val or '0', lane or '0'))
        if obj is not None:
            lines.append('%s /this->checked/ { this->obid = om[this->ob] ? om[this->ob] : objectcount + 1; }' % at)
            if completion:
                lines.append('%s /this->checked/ { this->completion = this->ob == self->ended; }' % at)
                for slot in range(slots):
                    lines.append('%s /this->checked && spanEvent[%d] == this->ob/ { this->completion = 1; }' % (at, slot))
        if lane is not None:
            lines.append('%s /this->checked/ { this->lnid = lm[this->ln] ? lm[this->ln] : lanecount + 1; }' % at)
        for source, target, token, index, ob, ln in transitions:
            if token != kind:
                continue
            clause = ['active[epoch %% 2, %d] == epoch' % source]
            if kind == 'op':
                operation = operations[program[index]]
                if operation.primitive != primitive and not (operation.primitive == trip and primitive == wait):
                    continue
                if name == 'bridge_start':
                    clause.append('arg1 == 0')
                if obj is not None:
                    if completion and operation.object == completion.object:
                        slot = '(this->va - 1) %% %d' % slots
                        expected = ('(this->va && this->va <= self->spanfirst ? '
                            '(this->va > spanFirst[%s] && this->va <= spanEnd[%s] ? spanEvent[%s] : 0) : self->ended)' %
                            (slot, slot, slot))
                        clause.append('this->ob == %s' % expected)
                    else:
                        clause.append('!this->completion && this->obid == %d' % ob)
                if lane is not None:
                    clause.append('this->lnid == %d' % ln)
                family, delta = rules[index]
                value = 'this->tally + (%d)' % delta if family else str(operation.value)
                if val is not None:
                    value_clause = 'this->va == (%s)' % value
                else:
                    value_clause = '1'
                printed = (index, operation.primitive, operation.object if obj is not None else -1, operation.execution)
            else:
                span, family = spans[index], 0
                clause.append('invocation == %d' % index)
                if kind == 'enter':
                    clause.extend(('arg2 == %d' % span.first, 'arg3 == %d' % span.count))
                else:
                    clause.append('arg1 == 0')
                value_clause, printed = '1', (-1 - index * 2 - (kind == 'leave'), -1, -1, span.first)
            lines.append('%s /this->checked/ { this->matching = %s; this->tally = 0; }' % (at, ' && '.join(clause)))
            if family:
                bindings = machine['states'][source][4]
                for logical, ordinal in zip(machine['objects'], bindings):
                    if logical in family_objects[family] and ordinal:
                        lines.append('%s /this->checked && this->matching && signals[%d] > this->tally/ '
                                     '{ this->tally = signals[%d]; }' % (at, ordinal, ordinal))
            lines.append('%s /this->checked && this->matching && (%s)/ { this->ok = 1; '
                'active[(epoch + 1) %% 2, %d] = epoch + 1; this->ordinary = this->ordinary || %d; '
                'printf("mesh-sync-monitor: transition %%d %d %d %d %d %d %%d %d %%d\\n", epoch - 1, %s, tid); }' %
                (at, value_clause, target, bool(ob), source, target, *printed[:3], printed[3],
                 'this->va' if val is not None else '-1'))
        if obj is not None:
            lines.append('%s /this->checked && this->ok && this->ordinary/ '
                '{ om[this->ob] = this->obid; objectcount = this->obid > objectcount ? this->obid : objectcount; }' % at)
            if primitive == signal:
                lines.append('%s /this->checked && this->ok && this->ordinary/ { signals[this->obid] = this->va; }' % at)
        if lane is not None:
            lines.append('%s /this->checked && this->ok/ '
                '{ lm[this->ln] = this->lnid; lanecount = this->lnid > lanecount ? this->lnid : lanecount; }' % at)
        lines.append('%s /this->checked && !this->ok/ '
            '{ printf("mesh-sync-monitor: unexpected %s at trace step %%d invocation %%d value %%d thread %%d\\n", '
            'epoch - 1, invocation, this->va, tid); %s }' % (at, name or kind, act))
        lines.append('%s /this->checked && this->ok/ { epoch++; }' % at)

    check('pid$target::MetalProgramRun:', 'entry', '/1/', 'enter')
    argument = completion.argument if completion else 4
    lines.append('pid$target::MetalProgramRun:entry { self->inrun = 1; self->exec = arg2; self->ended = arg%d; '
                 'self->spanfirst = arg2; self->spanend = arg2 + arg3; }' % argument)
    check('pid$target::MetalProgramRun:', 'return', '/self->inrun/', 'leave')
    if completion:
        for back in range(slots):
            lines.append('pid$target::MetalProgramRun:return /self->inrun && arg1 == 0 && self->spanend - self->spanfirst > %d/ '
                '{ this->slot = (self->spanend - 1 - %d) %% %d; spanEvent[this->slot] = self->ended; '
                'spanFirst[this->slot] = self->spanfirst; spanEnd[this->slot] = self->spanend; }' % (back, back, slots))
    lines.append('pid$target::MetalProgramRun:return /self->inrun/ { self->inrun = 0; invocation++; '
                 'printf("mesh-sync-monitor: run %d\\n", (int)arg1); }')
    for name, probe, obj, val, lane in PROBES:
        check(probe, 'entry', '/self->inrun && self->inapi == 0/', 'op', name, obj, val, lane)
        lines.append('%sentry /self->inrun/ { self->inapi++; }' % probe)
        lines.append('%sreturn /self->inrun && self->inapi/ { self->inapi--; }' % probe)
    name, probe, obj, val = START
    lines.append('%sentry /self->inrun && self->inapi == 0/ { self->bs = 1; self->bsob = %s; '
                 'self->bsva = %s; }' % (probe, obj, val))
    lines.append('%sentry /self->inrun/ { self->inapi++; }' % probe)
    lines.append('%sreturn /self->inrun && self->inapi/ { self->inapi--; }' % probe)
    if point:
        need = 'uregs[R_X%d]' % point['need']
        check('pid$target:libmesh.dylib:mesh_transfers_start:', '%x' % point['offset'], '/self->bs/', 'op', 'host_wait',
              '(self->bsob << 20) + uregs[R_X%d]' % point['edge'],
              '((int64_t)uregs[R_X%d] < 0 ? 0 : uregs[R_X%d] ? -1 : %s)' % (point['high'], point['high'], need))
    check(probe, 'return', '/self->bs/', 'op', name, 'self->bsob', 'self->bsva')
    lines.append('%sreturn /self->bs/ { self->bs = 0; }' % probe)
    lines.append('%s /self->inrun && self->inapi == 0/ { printf("mesh-sync-monitor: unexpected %%s %%s '
                 'execution %%d thread %%d\\n", probeprov, probefunc, self->exec, tid); %s }' %
                 (',\n'.join(UNLISTED), act))
    lines.append('END { this->accepted = 0; }')
    for state in machine['accepting']:
        lines.append('END /active[epoch %% 2, %d] == epoch/ { this->accepted = 1; '
                     'printf("mesh-sync-monitor: accepting %%d %d\\n", epoch - 1); }' % (state, state))
    lines.append('END /!this->accepted/ { printf("mesh-sync-monitor: unexpected incomplete trace at %d\\n", epoch - 1); exit(1); }')
    return '\n'.join(lines) + '\n', sum(span.count for span in spans)


def bridge_program(group):
    return '\n'.join([
        '#pragma D option quiet', '#pragma D option bufsize=16m',
        'pid$1::tree_toward:entry /arg0 == %d/ { this->m = (uint64_t *)copyin(arg1, 64); }' % COUNT_KIND,
        'pid$1::tree_toward:entry /arg0 == %d && this->m[0] == %d/ { this->w = (uint32_t *)this->m; '
        'printf("mesh-sync-monitor: raise %%d %%d %%d %%d %%d\\n", this->m[5], this->w[12], this->w[13], this->w[14], '
        'this->w[15]); }' % (COUNT_KIND, group)]) + '\n'


# the bridge's raises against its compiled order: a count word is monotone (mesh-records.c counts_tell sends the
# plan's submissions), so one raise of count c answers every pending raise of its word up to c; a raise's word is its
# (node, sink, function, entry), bound one to one to the compiled object
def bridge_match(expected, operations, raises):
    bound, back, pending, made, missing = {}, {}, {}, [], []
    for i in expected:
        pending.setdefault(operations[i].object, []).append(i)
    seen = set()
    for count, *word in raises:
        key = tuple(word)
        if (key, count) in seen:
            continue
        seen.add((key, count))
        candidates = [obj for obj, rows in pending.items() if rows and operations[rows[0]].value <= count]
        obj = bound.get(key)
        if obj is None:
            free = [c for c in candidates if c not in back]
            if len(free) != 1:
                continue
            obj = free[0]
            bound[key], back[obj] = obj, key
        rows = pending.get(obj, [])
        while rows and operations[rows[0]].value <= count:
            made.append((rows.pop(0), count))
    for rows in pending.values():
        missing += rows
    return made, missing


def verbs_owners():
    owners = set(subprocess.run(['pgrep', '-f', 'mesh-flow'], capture_output=True, text=True).stdout.split())
    listed = subprocess.run(['launchctl', 'print', 'system/io.mesh.bridge'], capture_output=True, text=True).stdout
    owners.update(re.findall(r'^\s*pid = (\d+)', listed, re.M))
    return owners


def cover(path):
    compiled, processors, _, _, operations, _, metadata = load(path)
    nodes = {p.index % mesh_graph.NODES for p in processors if KINDS[p.kind] != 'kernel'}
    runs = [json.loads(line) for line in open(REVIEWS)] if os.path.exists(REVIEWS) else []
    runs = [r for r in runs if r.get('kind') == 'synchronization-monitor' and r.get('sha256') == compiled['digest'] and
            r.get('result') in ('pass', 'pass-scoped')]
    covered = {r.get('node') for r in runs}
    blind = {}
    for r in runs:
        for name, n in (r.get('blind') or {}).items():
            if name != 'elsewhere':
                blind['node %s %s' % (r.get('node'), name)] = n
    remaining = sorted(nodes - covered)
    print('mesh-sync-monitor: %s: nodes %s, observed on %s, not observed on %s, unobserved classes %s' % (
        compiled['digest'], sorted(nodes), sorted(covered), remaining, blind or 'none'))
    full = not remaining and not blind
    print('mesh-sync-monitor: %s' % ('full coverage' if full else 'scoped coverage only'))
    return 0 if full else 1


def main(argv):
    if len(argv) < 3 or argv[1] not in ('generate', 'run', 'cover'):
        sys.stderr.write('usage: mesh-sync-monitor.py generate SIGNATURE.record [--observe]\n'
                         '       mesh-sync-monitor.py run SIGNATURE.record [--observe] [--bridge PID --group GROUP] '
                         '[--env NAME=VALUE ...] (-p PID | -- COMMAND ...)\n'
                         '       (pid probes are made when dtrace attaches: a program that loads libmetal_recording and '
                         'libmesh after it starts is attached with -p once loaded)\n'
                         '       mesh-sync-monitor.py cover SIGNATURE.record\n')
        return 2
    mode, path, rest = argv[1], argv[2], argv[3:]
    if mode == 'cover':
        return cover(path)
    pid, command, observe, bridge, group = None, None, False, None, None
    environment = []
    while rest:
        if rest[0] == '-p': pid, rest = rest[1], rest[2:]
        elif rest[0] == '--observe': observe, rest = True, rest[1:]
        elif rest[0] == '--bridge': bridge, rest = rest[1], rest[2:]
        elif rest[0] == '--group': group, rest = int(rest[1], 0), rest[2:]
        elif rest[0] == '--env':
            if len(rest) < 2 or not re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', rest[1]):
                sys.stderr.write('mesh-sync-monitor: --env requires NAME=VALUE\n')
                return 2
            environment.append(rest[1])
            rest = rest[2:]
        elif rest[0] == '--': command, rest = rest[1:], []
        else:
            sys.stderr.write('mesh-sync-monitor: unknown argument %s\n' % rest[0])
            return 2
    if mode == 'run' and (not (pid or command) or (pid and environment)):
        sys.stderr.write('mesh-sync-monitor: run needs -p PID or -- COMMAND; --env applies only to COMMAND\n')
        return 2
    compiled, processors, tasks, dependences, operations, lane, metadata = load(path)
    point, unpointed = count_wait_point(LIBMESH)
    if unpointed:
        sys.stderr.write('mesh-sync-monitor: the plan\'s count waits stay unobserved (implied): %s\n' % unpointed)
    node, which = classify(processors, operations, point is not None)
    program = [i for i, c in enumerate(which) if c == 'program']
    coverage = {c: which.count(c) for c in CLASSES if c in which}
    script, executions = generate(operations, program, observe, point, metadata)
    target_path = os.path.splitext(path)[0] + '.monitor.d'
    open(target_path, 'w').write(script)
    # item 4: every compiled edge the image of a dependence and none redundant, before anything runs; item 6: every
    # kernel's synchronization known from its compiled IR
    before = []
    if compiled['states'].get('false'):
        before.append('%d synchronization edges with no dependence under them' % compiled['states']['false'])
    if compiled['redundant']:
        before.append('%d redundant synchronization edges' % compiled['redundant'])
    if compiled['unknown']:
        before.append('kernel synchronization not known from the compiled IR: %s' % sorted(compiled['unknown'].items()))
    sys.stderr.write('mesh-sync-monitor: %s written, digest %s, node %s, %d executions; operations by class %s; edge '
                     'states %s\n' % (target_path, compiled['digest'], node, executions, coverage, compiled['states']))
    for reason in before:
        sys.stderr.write('mesh-sync-monitor: fails before running: %s\n' % reason)
    if mode == 'generate':
        return 1 if before else 0
    record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
              'kind': 'synchronization-monitor', 'signature': os.path.abspath(path), 'sha256': compiled['digest'],
              'node': node, 'executions': executions, 'coverage': coverage, 'destructive': not observe,
              'count_wait_point': point}
    if before:
        record.update(result='fail', reasons=before)
        with open(REVIEWS, 'a') as reviews:
            reviews.write(json.dumps(record) + '\n')
        return 1
    owners = verbs_owners()
    if (pid and pid in owners) or (command and any('mesh-flow' in c for c in command)):
        sys.stderr.write('mesh-sync-monitor: the target holds the verbs device; a stopped verbs owner wedges the node '
                         'until a person pulls its power (AGENTS.md, The kill guard). Observe the program, not the '
                         'bridge.\n')
        return 2
    if bridge and (bridge not in owners or group is None):
        sys.stderr.write('mesh-sync-monitor: --bridge names this node\'s running bridge (%s) and needs --group, the '
                         'program\'s group\n' % sorted(owners))
        return 2
    # dtrace(1) -c splits its command on blanks with no quoting, so an argument holding one is not representable
    if command and any(not c or re.search(r'\s', c) for c in command):
        sys.stderr.write('mesh-sync-monitor: dtrace -c cannot pass an empty argument or one holding whitespace: %r\n'
                         % [c for c in command if not c or re.search(r'\s', c)])
        return 2
    watcher = None
    if bridge:
        bridge_path = os.path.splitext(path)[0] + '.bridge.d'
        open(bridge_path, 'w').write(bridge_program(group))
        watcher = subprocess.Popen(['sudo', 'dtrace', '-q', '-s', bridge_path, bridge], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
    target = ['-p', pid] if pid else ['-c', ' '.join(command)]
    done = subprocess.run(['sudo'] + environment + ['dtrace'] + ([] if observe else ['-w']) + ['-Z', '-s', target_path] + target,
                          capture_output=True, text=True)
    raised, bridge_text = [], ''
    if watcher:
        watcher.send_signal(2)
        bridge_text = watcher.communicate()[0]
        raised = [tuple(map(int, m.groups())) for m in
                  (re.match(r'mesh-sync-monitor: raise (\d+) (\d+) (\d+) (\d+) (\d+)$', line)
                   for line in bridge_text.splitlines()) if m]
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    observed, failures, walk, records, accepting = [], [], [], [], []
    for line in done.stdout.splitlines():
        match = re.fullmatch(r'mesh-sync-monitor: transition ' + r'(-?\d+) ' * 8 + r'(-?\d+)', line)
        if match:
            records.append(tuple(map(int, match.groups())))
        match = re.fullmatch(r'mesh-sync-monitor: accepting (\d+) (\d+)', line)
        if match:
            accepting.append(tuple(map(int, match.groups())))
        if 'mesh-sync-monitor: unexpected' in line:
            failures.append(line)
        match = re.fullmatch(r'mesh-sync-monitor: run (-?\d+)', line)
        if match and int(match.group(1)):
            failures.append('MetalProgramRun returned %s' % match.group(1))
    if done.returncode:
        failures.append('dtrace exited %d' % done.returncode)
    try:
        witness = mesh_graph.trace_witness(records, accepting)
    except ValueError as failure:
        failures.append(str(failure))
        witness = []
    for _, _, _, k, primitive, obj, value, execution, _ in witness:
        if k < 0:
            continue
        compiled_op = operations[program[k]]
        walk.append(k)
        observed.append(mesh_graph.Operation(role(processors, node, PRIMITIVES[primitive]), primitive, compiled_op.vertex,
                                             NONE if obj < 0 else obj, execution,
                                             compiled_op.value if value < 0 else value))
    blind = {c: n for c, n in coverage.items() if c not in ('program', 'static')}
    if bridge:
        made, missing = bridge_match([i for i, c in enumerate(which) if c == 'bridge'], operations, raised)
        if missing:
            failures.append('%d compiled bridge raises not observed (%d observed)' % (len(missing), len(made)))
        blind.pop('bridge', None)
        observed += [mesh_graph.Operation(operations[i].processor, operations[i].primitive, operations[i].vertex,
                                          operations[i].object, operations[i].execution, count) for i, count in made]
    expected_ops, counters = [], [0, 0, 0]
    rules = counter_rules(operations, program, metadata['scopes'])
    for k in walk:
        source = operations[program[k]]
        family, delta = rules[k]
        value = counters[family] + delta if family else source.value
        expected_ops.append(mesh_graph.Operation(source.processor, source.primitive, source.vertex, source.object,
                                                 source.execution, value))
        if family and source.primitive == PRIMITIVES.index('queue_signal'):
            counters[family] = value
    if bridge:
        expected_ops += [operations[i] for i, c in enumerate(which) if c == 'bridge']
    expected = mesh_graph.signature(processors, tasks, dependences, expected_ops)
    watched = mesh_graph.signature(processors, tasks, dependences, observed)
    if expected['digest'] != watched['digest']:
        failures.append('observed graph %s differs from the compiled graph %s over the observed scope'
                        % (watched['digest'], expected['digest']))
    differing = sorted(k for k in set(expected['counts']) | set(watched['counts'])
                       if expected['counts'].get(k, 0) != watched['counts'].get(k, 0))
    for key in differing:
        print('mesh-sync-monitor: count %s %s compiled %d observed %d' % (key[0], key[1], expected['counts'].get(key, 0),
                                                                          watched['counts'].get(key, 0)))
    for name in ('matrix_market', 'dot', 'space_time'):
        print('mesh-sync-monitor: compiled %s\n%s' % (name, compiled[name]))
        print('mesh-sync-monitor: observed %s\n%s' % (name, watched[name]))
    if observe:
        sys.stderr.write('mesh-sync-monitor: --observe: no destructive action; an unexpected operation ends the trace '
                         'and fails the run, the process is not stopped at the call (item 4 not exercised)\n')
    result = 'fail' if failures else 'pass-scoped' if blind else 'pass'
    for reason in failures:
        sys.stderr.write('mesh-sync-monitor: fail: %s\n' % reason)
    runtime = {'program': len(walk), 'bridge': len(observed) - len(walk)}
    static = coverage.get('static', 0)
    sys.stderr.write('mesh-sync-monitor: %s; runtime-observed operations %s; %d kernel records from compiled IR '
                     '(not runtime-observed); other unobserved classes %s\n' % (result, runtime, static, blind or 'none'))
    record.update(result=result, reasons=failures[:16], blind=blind, runtime_observed=runtime, compiled_kernel_records=static,
                  observed_sha256=watched['digest'],
                  expected_sha256=expected['digest'], counts_differing=len(differing), dtrace_status=done.returncode,
                  bridge=bool(bridge), bridge_raises=len(raised))
    with open(REVIEWS, 'a') as reviews:
        reviews.write(json.dumps(record) + '\n')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
