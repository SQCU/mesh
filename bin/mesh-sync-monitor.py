#!/usr/bin/env mesh-python
"""design/synchronization-signature-change-request.md items 3, 4 and 5"""
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
    finally:
        mesh_graph.free(made)
    return compiled, processors, tasks, dependences, operations, lane


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


# metal_recording.m MetalDrain/MetalSubmit: global test and commit eventcounts, advanced by their queue signals.
def counter_rules(operations, program):
    counters, rules = [0, 0, 0], []
    signal = PRIMITIVES.index('queue_signal')
    for i in program:
        o = operations[i]
        family = 1 if mesh_graph.LOOP <= o.object < mesh_graph.PLAN_SUBMISSION else 2 if o.object == 0x3FFFFFFF else 0
        rules.append((family, o.value - counters[family] if family else 0))
        if family and o.primitive == signal:
            counters[family] = o.value
    return rules


def generate(operations, program, observe, point=None):
    total, executions = len(program), 1 + max((operations[i].execution for i in program), default=0)
    rules = counter_rules(operations, program)
    lines = ['#pragma D option quiet', '#pragma D option strsize=512', '#pragma D option bufsize=64m',
             '#pragma D option dynvarsize=64m'] + ([] if observe else ['#pragma D option destructive'])
    lines += ['self int inrun, inapi, bs;', 'self uint64_t exec, ended, bsob, bsva;', 'int64_t cur;',
              'int64_t e[int64_t], o[int64_t], v[int64_t], x[int64_t], q[int64_t];',
              'int64_t om[uint64_t], ko[int64_t], lm[uint64_t], kl[int64_t];',
              'int64_t family[int64_t], delta[int64_t], eventcount[int64_t];', 'BEGIN { cur = 0; }']
    for start in range(0, total, 64):
        lines.append('BEGIN { %s }' % ' '.join(
            'e[%d] = %d; o[%d] = %d; v[%d] = %d; x[%d] = %d; q[%d] = %d;' % (
                k, operations[i].primitive, k, operations[i].object, k, operations[i].value, k, operations[i].execution, k,
                operations[i].processor) for k, i in enumerate(program[start:start + 64], start)))
    for start in range(0, total, 64):
        lines.append('BEGIN { %s }' % ' '.join('family[%d] = %d; delta[%d] = %d;' % (k, rules[k][0], k, rules[k][1])
                                             for k in range(start, min(start + 64, total))))
    lines.append('pid$target::MetalProgramRun:entry { self->inrun = 1; self->exec = arg2; self->ended = arg4; }')
    lines.append('pid$target::MetalProgramRun:return /self->inrun/ { self->inrun = 0; '
                 'printf("mesh-sync-monitor: run %d\\n", (int)arg1); }')
    act = 'ustack(); exit(1);' if observe else 'ustack(); stop(); exit(1);'
    ended = (PRIMITIVES.index('queue_signal'), PRIMITIVES.index('host_signal'))

    # a WHILE of no bound (mesh-graph.h MESH_GRAPH_TRIP_TEST, object MESH_GRAPH_LOOP + loop): its compiled signature takes
    # one trip, between the loop's two tests in an execution; a run takes the trips its data gives, so after the second
    # test the walk may return to after the first (another trip) and after the first it may go on after the second (no
    # trip). Test/commit eventcounts retain their compiled recurrence under that walk (counter_rules).
    trip, wait = PRIMITIVES.index('trip_test'), PRIMITIVES.index('host_wait')
    tests = {}
    for k, i in enumerate(program):
        if operations[i].primitive == trip:
            tests.setdefault((operations[i].execution, operations[i].object), []).append(k)
    alt = {}
    for ks in tests.values():
        for a, b in zip(ks[0::2], ks[1::2]):
            alt[b + 1], alt[a + 1] = a + 1, b + 1
    lines.append('int64_t alt[int64_t]; int shifted;')
    for start in range(0, len(alt), 64):
        lines.append('BEGIN { %s }' % ' '.join('alt[%d] = %d;' % (k, t + 1) for k, t in sorted(alt.items())[start:start + 64]))

    def match(index):
        return ('%s < %d && (e[%s] == this->c || (this->c == %d && e[%s] == %d)) && x[%s] >= self->exec && (!this->hasva || '
                '%s) && (!this->hasob || ((om[this->ob] == 0 || om[this->ob] == o[%s] + 1) && (ko[o[%s]] == 0 || '
                'ko[o[%s]] == this->ob + 1))) && (!this->hasln || ((lm[this->ln] == 0 || lm[this->ln] == q[%s] + 1) && '
                '(kl[q[%s]] == 0 || kl[q[%s]] == this->ln + 1)))' % (
                    index, total, index, wait, index, trip, index,
                    '(family[%s] ? eventcount[family[%s]] + delta[%s] : v[%s]) == this->va' % (index, index, index, index), index, index, index, index, index, index))

    def check(probe, name, obj, val, lane, point, predicate):
        guard = '/this->checked && %s/'
        lines.append('%s%s { this->checked = 0; }' % (probe, point))
        lines.append('%s%s %s { this->checked = 1; this->c = %d; this->ob = %s; this->hasob = %d; this->va = %s; '
                     'this->hasva = %d; this->ln = %s; this->hasln = %d; this->i = cur; this->ok = %s%s; }' % (
                         probe, point, predicate, PRIMITIVES.index(name), obj or '0', obj is not None, val or '0', val is not None,
                         lane or '0', lane is not None, match('this->i'),
                         ' && arg1 == 0' if name == 'bridge_start' else ''))
        lines.append('%s %s { this->i = alt[cur] - 1; this->ok = %s%s; shifted = this->ok ? 1 : shifted; }' % (
            probe + point, guard % '!this->ok && alt[cur]', match('this->i'),
            ' && arg1 == 0' if name == 'bridge_start' else ''))
        lines.append('%s %s { om[this->hasob ? this->ob : 0] = this->hasob ? o[this->i] + 1 : om[0]; '
                     'ko[this->hasob ? o[this->i] : -1] = this->hasob ? this->ob + 1 : ko[-1]; '
                     'eventcount[family[this->i]] = family[this->i] && this->c == 1 ? this->va : eventcount[family[this->i]]; '
                     'lm[this->hasln ? this->ln : 0] = this->hasln ? q[this->i] + 1 : lm[0]; '
                     'kl[this->hasln ? q[this->i] : -1] = this->hasln ? this->ln + 1 : kl[-1]; cur = this->i + 1; '
                     'printf("mesh-sync-monitor: observed %%d %%d %%d %%d %%d %%d\\n", this->i, this->c, this->hasob ? '
                     'o[this->i] : -1, this->hasva ? (int64_t)this->va : -1, self->exec, tid); '
                     'self->exec = x[this->i]; self->exec = this->hasob && this->ob == self->ended && (this->c == %d || '
                     'this->c == %d) && this->va > self->exec ? this->va : self->exec; }'
                     % (probe + point, guard % 'this->ok', ended[0], ended[1]))
        lines.append('%s %s { printf("mesh-sync-monitor: unexpected %s at operation %%d of %d, execution %%d (compiled '
                     'primitive %%d value %%d execution %%d; called value %%d) thread %%d\\n", this->i, self->exec, '
                     'this->i < %d ? e[this->i] : -1, this->i < %d ? v[this->i] : -1, this->i < %d ? x[this->i] : -1, '
                     'this->hasva ? (int64_t)this->va : -1, tid); %s }' % (probe + point, guard % '!this->ok', name, total,
                                                                          total, total, total, act))

    for name, probe, obj, val, lane in PROBES:
        check(probe, name, obj, val, lane, 'entry', '/self->inrun && self->inapi == 0/')
        lines.append('%sentry /self->inrun/ { self->inapi++; }' % probe)
        lines.append('%sreturn /self->inrun && self->inapi/ { self->inapi--; }' % probe)
    name, probe, obj, val = START
    lines.append('%sentry /self->inrun && self->inapi == 0/ { self->bs = 1; self->bsob = %s; '
                 'self->bsva = %s; }' % (probe, obj, val))
    lines.append('%sentry /self->inrun/ { self->inapi++; }' % probe)
    lines.append('%sreturn /self->inrun && self->inapi/ { self->inapi--; }' % probe)
    if point:
        need = 'uregs[R_X%d]' % point['need']
        check('pid$target:libmesh.dylib:mesh_transfers_start:', 'host_wait', '(self->bsob << 20) + uregs[R_X%d]' %
              point['edge'], '((int64_t)uregs[R_X%d] < 0 ? 0 : uregs[R_X%d] ? -1 : %s)' % (point['high'], point['high'], need),
              None, '%x' % point['offset'], '/self->bs/')
    check(probe, name, 'self->bsob', 'self->bsva', None, 'return', '/self->bs/')
    lines.append('%sreturn /self->bs/ { self->bs = 0; }' % probe)
    lines.append('%s /self->inrun && self->inapi == 0/ { printf("mesh-sync-monitor: unexpected %%s %%s '
                 'execution %%d thread %%d\\n", probeprov, probefunc, self->exec, tid); %s }' % (
                     ',\n'.join(UNLISTED), act))
    lines.append('END { printf("mesh-sync-monitor: program %%d of %d\\n", cur); printf("mesh-sync-monitor: shifted %%d\\n", '
                 'shifted); }' % total)
    return '\n'.join(lines) + '\n', executions


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
    compiled, processors, _, _, operations, _ = load(path)
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
    compiled, processors, tasks, dependences, operations, lane = load(path)
    point, unpointed = count_wait_point(LIBMESH)
    if unpointed:
        sys.stderr.write('mesh-sync-monitor: the plan\'s count waits stay unobserved (implied): %s\n' % unpointed)
    node, which = classify(processors, operations, point is not None)
    program = [i for i, c in enumerate(which) if c == 'program']
    coverage = {c: which.count(c) for c in CLASSES if c in which}
    script, executions = generate(operations, program, observe, point)
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
    # the walk: the compiled operation each observed one matched, in order (a WHILE's trips repeat or skip a span of
    # it). A trip test is observed as the host wait it is; eventcount operands stay observed values.
    observed, failures, walk, shifted = [], [], [], 0
    trip = PRIMITIVES.index('trip_test')
    for line in done.stdout.splitlines():
        m = re.match(r'mesh-sync-monitor: shifted (\d+)$', line)
        if m:
            shifted = int(m.group(1))
        m = re.match(r'mesh-sync-monitor: observed (\d+) (\d+) (-?\d+) (-?\d+) (\d+) \d+$', line)
        if m:
            k, c, obj, value, execution = map(int, m.groups())
            compiled_op = operations[program[k]]
            walk.append(k)
            c = trip if compiled_op.primitive == trip else c
            observed.append(mesh_graph.Operation(role(processors, node, PRIMITIVES[c]), c, compiled_op.vertex,
                                                 NONE if obj < 0 else obj, execution,
                                                 compiled_op.value if value < 0 else value))
        if 'mesh-sync-monitor: unexpected' in line:
            failures.append(line)
        m = re.match(r'mesh-sync-monitor: run (-?\d+)$', line)
        if m and int(m.group(1)):
            failures.append('MetalProgramRun returned %s' % m.group(1))
    if done.returncode:
        failures.append('dtrace exited %d' % done.returncode)
    looped = any(operations[i].primitive == trip for i in program)
    if not looped and len(observed) != len(program):
        failures.append('%d of %d compiled program operations observed' % (len(observed), len(program)))
    if looped and program and (not walk or walk[-1] != len(program) - 1):
        failures.append('the walk of %d observed operations ended at compiled operation %s of %d'
                        % (len(walk), walk[-1] if walk else None, len(program)))
    blind = {c: n for c, n in coverage.items() if c not in ('program', 'static')}
    if bridge:
        made, missing = bridge_match([i for i, c in enumerate(which) if c == 'bridge'], operations, raised)
        if missing:
            failures.append('%d compiled bridge raises not observed (%d observed)' % (len(missing), len(made)))
        blind.pop('bridge', None)
        observed += [mesh_graph.Operation(operations[i].processor, operations[i].primitive, operations[i].vertex,
                                          operations[i].object, operations[i].execution, count) for i, count in made]
    expected_ops, counters = [], [0, 0, 0]
    rules = counter_rules(operations, program)
    for k in (walk if looped else range(len(program))):
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
