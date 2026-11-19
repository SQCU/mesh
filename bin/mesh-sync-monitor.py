#!/usr/bin/env mesh-python
"""design/synchronization-signature-change-request.md items 3, 4 and 5"""
import datetime
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
LOWERED, COUNT, SUBMISSION = mesh_graph.LOWERED, mesh_graph.PLAN_COUNT, mesh_graph.PLAN_SUBMISSION
PROBES = [
    ('submit', 'objc$target::-commit?count?options?:', 'entry', None, 'arg3', None),
    ('queue_signal', 'objc$target::-signalEvent?value?:', 'entry', 'arg2', 'arg3', None),
    ('queue_wait', 'objc$target::-waitForEvent?value?:', 'entry', 'arg2', 'arg3', None),
    ('fence_update', 'objc$target::-updateFence?afterEncoderStages?:', 'entry', 'arg2', None, None),
    ('fence_wait', 'objc$target::-waitForFence?beforeEncoderStages?:', 'entry', 'arg2', None, None),
    ('pass_barrier', 'objc$target::-barrierAfterEncoderStages?beforeEncoderStages?visibilityOptions?:', 'entry', None,
     None, None, 'queue'),
    ('pass_barrier', 'pid$target::dispatch_apply_f:', 'return', None, None, None, 'host'),
    ('queue_barrier', 'objc$target::-barrierAfterQueueStages?beforeStages?visibilityOptions?:', 'entry', None, None,
     None),
    ('producer_barrier', 'objc$target::-barrierAfterStages?beforeQueueStages?visibilityOptions?:', 'entry', None, None,
     None),
    ('host_wait', 'objc$target::-waitUntilSignaledValue?timeoutMS?:', 'entry', 'arg0', 'arg2', (LOWERED, COUNT)),
    ('host_wait', 'objc$target::-waitUntilCompleted:', 'entry', None, None, (NONE, NONE + 1)),
    ('host_wait', 'pid$target::mesh_host_arrived:', 'return', 'self->plan', 'self->arrives', (COUNT, SUBMISSION)),
    ('host_wait', 'pid$target::_queue_SimpleQueue_get:', 'return', 'self->queue', None, (0, LOWERED)),
    ('host_signal', 'pid$target::_queue_SimpleQueue_put:', 'entry', 'arg0', None, (0, LOWERED)),
    ('host_signal', 'objc$target::-setSignaledValue?:', 'entry', 'arg0', 'arg2', (LOWERED, COUNT)),
    ('host_notify', 'objc$target::-notifyListener?atValue?block?:', 'entry', 'arg0', 'arg3', None),
    ('host_publish', 'pid$target::mesh_host_publish:', 'entry', '(arg1 << 20) + arg2', 'arg3 + 1', None),
    ('bridge_start', 'pid$target::mesh_transfers_start:', 'entry', 'arg1', 'arg2', None),
]
FEEDBACK = ('host_signal', 'objc$target::-setSignaledValue?:', 'arg0', 'arg2')
UNLISTED = [
    'objc$target::-waitUntilScheduled:entry',
    'pid$target::dispatch_semaphore_wait:entry',
    'pid$target::os_sync_wait_on_address:entry',
    'pid$target::os_sync_wait_on_address_with_timeout:entry',
    'pid$target::pthread_cond_wait:entry',
    'pid$target::pthread_cond_timedwait:entry',
    'pid$target::nanosleep:entry',
    'pid$target::usleep:entry',
]
SAVED = ['pid$target::mesh_host_arrived:entry { self->plan = (arg1 << 20) + arg2, self->arrives = arg3 + 1; }',
         'pid$target::_queue_SimpleQueue_get:entry { self->queue = arg0; }']
INTERPRETER = 'pid$target::take_gil*:'
APPLY = ['pid$target::dispatch_apply_f:entry { self->applying++; }',
         'pid$target::dispatch_apply_f:return /self->applying/ { self->applying--; }',
         'pid$target::_dispatch_client_callout2:entry /self->applying && self->inapi/ { self->inapi--, self->called++; }',
         'pid$target::_dispatch_client_callout2:return /self->called/ { self->inapi++, self->called--; }']
COUNT_KIND = 9
CLASSES = ('program', 'feedback', 'static', 'bridge', 'elsewhere')


def load(path):
    with open(path, 'rb') as f:
        made = mesh_graph.from_record(f.read())
    owner = mesh_graph.Owned(made, mesh_graph.free, None)
    def view(rows, count):
        return mesh_graph.View(owner, rows, range(count), lambda row: row)
    return (mesh_graph.forms(made), view(made.processor, made.processors), view(made.task, made.tasks),
            view(made.dependence, made.dependences), view(made.operation, made.operations))


def classify(processors, operations, node):
    def which(o):
        p, name = processors[o.processor], PRIMITIVES[o.primitive]
        if KINDS[p.kind] == 'kernel':
            return 'static'
        if p.index % mesh_graph.NODES != node:
            return 'elsewhere'
        if KINDS[p.kind] == 'bridge':
            return 'bridge'
        return 'feedback' if name == 'host_signal' and KINDS[p.kind] == 'queue' else 'program'
    return [which(o) for o in operations]


def probed(operation, processors):
    return next(row for row in PROBES if row[0] == PRIMITIVES[operation.primitive] and (
        not row[5] or row[5][0] <= operation.object < row[5][1]) and (
        not row[6:] or KINDS[processors[operation.processor].kind] in row[6:]))


def sequences(processors, operations, which, node):
    found = []
    for p in processors:
        if KINDS[p.kind] != 'host' or p.index % mesh_graph.NODES != node:
            continue
        lanes = {q for q, other in enumerate(processors) if other.index == p.index and KINDS[other.kind] != 'bridge'}
        order = [i for i, o in enumerate(operations) if o.processor in lanes and which[i] == 'program']
        if order:
            found.append(order)
    return found


def periodic(operations, order):
    rows = {}
    for i in order:
        rows.setdefault(operations[i].execution, []).append(i)
    rows = [rows[e] for e in sorted(rows)]
    period = len(rows[0])

    def step(k):
        return operations[rows[1][k]].value - operations[rows[0][k]].value if len(rows) > 1 else 0
    uniform = all(len(row) == period for row in rows[:-1]) and len(rows[-1]) >= period and all(
        (operations[row[k]].primitive, operations[row[k]].object, operations[row[k]].value) ==
        (operations[rows[0][k]].primitive, operations[rows[0][k]].object, operations[rows[0][k]].value + e * step(k))
        for e, row in enumerate(rows) for k in range(period))
    if not uniform:
        return len(order), 1, [0] * len(order)
    return period, len(rows), [step(k) for k in range(period)]


def generate(processors, operations, orders, observe, addresses):
    act = 'printf("mesh-sync-monitor: failure target pid %d\\n", pid); ustack(); ' + (
        'exit(1);' if observe else 'stop(); exit(1);')
    lines = ['#pragma D option quiet', '#pragma D option strsize=512', '#pragma D option bufsize=64m',
             '#pragma D option dynvarsize=64m'] + ([] if observe else ['#pragma D option destructive'])
    lines += ['self int inapi, bound, applying, called;', 'self uint64_t at, cand, objects, plan, arrives, queue;',
              'self uint64_t om[uint64_t];', 'BEGIN { printf("mesh-sync-monitor: attached\\n"); }'] + SAVED + APPLY
    lines += ['%sentry { self->inapi++; }' % INTERPRETER, '%sreturn /self->inapi/ { self->inapi--; }' % INTERPRETER]
    shapes = [periodic(operations, order) for order in orders]
    local = []
    for order in orders:
        seen = {}
        local.append({i: seen.setdefault(operations[i].object, len(seen) + 1) for i in order
                      if probed(operations[i], processors)[3]})
    for row in PROBES:
        name, probe, point, obj, val = row[:5]
        at = probe + point
        satisfied = ' && arg1 != 0' if point == 'return' and obj else ''
        if point == 'return':
            lines.append('%sentry { self->inapi++; }' % probe)
            lines.append('%sreturn /self->inapi/ { self->inapi--; }' % probe)
        lines.append('%s { this->checked = 0; }' % at)
        lines.append('%s /self->inapi == 0%s/ { this->checked = 1, this->m = 0, this->ob = (uint64_t)(%s), this->va = (uint64_t)(%s), '
                     'this->cand = self->bound ? self->cand : %d, this->lid = self->om[(uint64_t)(%s)] ? self->om[(uint64_t)(%s)] : '
                     'self->objects + 1; }' % (at, satisfied, obj or '0', val or '0', (1 << len(orders)) - 1,
                                               obj or '0', obj or '0'))
        for s, (order, (period, executions, steps)) in enumerate(zip(orders, shapes)):
            for k, i in enumerate(order[:period] + order[period * executions:]):
                o = operations[i]
                if PRIMITIVES[o.primitive] != name or probed(o, processors) is not row:
                    continue
                where = 'self->at < %d && self->at %% %d == %d' % (period * executions, period, k) if k < period else \
                    'self->at == %d' % (period * executions + k - period)
                match = ['this->lid == %d' % local[s].get(i, 0)] * (obj is not None)
                match += ['this->va == %d + (self->at / %d) * %d' % (o.value, period, steps[k] if k < period else 0)] * \
                    (val is not None)
                lines.append('%s /this->checked && (this->cand & %d) && %s/ { this->m = this->m | (%s ? %d : 0); }' %
                             (at, 1 << s, where, ' && '.join(match) or '1', 1 << s))
        lines.append('%s /this->checked && !this->m%s/ { printf("mesh-sync-monitor: unexpected %s at step %%d of thread '
                     '%%d value %%d object %%d of %%d\\n", self->at, tid, this->va, this->lid, self->objects); %s }' %
                     (at, ' && self->bound' * (probe == FEEDBACK[1]), name, act))
        if obj is not None:
            lines.append('%s /this->checked && this->m && !self->om[(uint64_t)this->ob]/ { self->om[(uint64_t)this->ob] = this->lid, '
                         'self->objects = this->lid; }' % at)
        lines.append('%s /this->checked && this->m/ { printf("mesh-sync-monitor: op %%d %%d %%d %d %%d\\n", tid, '
                     'self->at, this->m, this->va); self->cand = this->m, self->bound = 1, self->at++; }' %
                     (at, PRIMITIVES.index(name)))
        if point == 'entry':
            lines.append('%sentry { self->inapi++; }' % probe)
            lines.append('%sreturn /self->inapi/ { self->inapi--; }' % probe)
    name, probe, obj, val = FEEDBACK
    lines.append('%sentry /!this->checked || !this->m/ { printf("mesh-sync-monitor: signal %%d %%d\\n", %s, %s); }' %
                 (probe, obj, val))
    lines.append('%s /self->bound && self->inapi == 0/ { printf("mesh-sync-monitor: unexpected %%s %%s at step %%d of '
                 'thread %%d\\n", probeprov, probefunc, self->at, tid); %s }' %
                 (',\n'.join(u for u in UNLISTED if addresses or 'os_sync' not in u), act))
    return '\n'.join(lines) + '\n', shapes


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
    compiled, processors, _, _, operations = load(path)
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


def trace_log(text, operations=(), orders=()):
    threads, signals, failures = {}, {}, []
    for line in text.splitlines():
        match = re.fullmatch(r'mesh-sync-monitor: op (\d+) (\d+) (\d+) (\d+) (-?\d+)', line)
        if match:
            thread, at, mask, primitive, value = map(int, match.groups())
            threads.setdefault(thread, []).append((at, mask, primitive, value))
        match = re.fullmatch(r'mesh-sync-monitor: signal (\d+) (\d+)', line)
        if match:
            signals.setdefault(int(match.group(1)), []).append(int(match.group(2)))
        if 'mesh-sync-monitor: unexpected' in line:
            position = re.search(r' at step (\d+) of thread (\d+)', line)
            if position:
                at, tid = map(int, position.groups())
                prior = threads.get(tid)
                mask = prior[-1][1] if prior else (1 << len(orders)) - 1
                candidates = (operations[row[at]] for i, row in enumerate(orders) if mask >> i & 1 and at < len(row))
                sites = sorted({o.native_site for o in candidates if o.native_site})
                if sites:
                    line += ' (expected compiled native_site %s)' % ', '.join(map(str, sites))
            failures.append(line)
    return threads, signals, failures


def observed_operation(operation, primitive, value):
    if (operation.primitive, operation.value) == (primitive, value):
        return operation
    observed = mesh_graph.Operation.from_buffer_copy(operation)
    observed.primitive, observed.value = primitive, value
    return observed


def observed_operations(processors, operations, which, orders, threads, signals, failures):
    seen, taken = {}, set()
    for thread, steps in sorted(threads.items()):
        steps.sort()
        free = [s for s in range(len(orders)) if steps[-1][1] >> s & 1 and s not in taken]
        if not free:
            failures.append('thread %d ran a compiled order another thread ran' % thread)
            continue
        taken.add(free[0])
        for at, _, primitive, value in steps:
            i = orders[free[0]][at]
            operation = operations[i]
            seen[i] = observed_operation(operation, primitive, value if probed(operation, processors)[4] else operation.value)
    compiled, unbound = {}, sorted(sorted(values) for values in signals.values())
    for i, c in enumerate(which):
        if c == 'feedback':
            compiled.setdefault(operations[i].object, []).append(i)
    for rows in compiled.values():
        values = sorted(operations[i].value for i in rows)
        if values in unbound:
            unbound.remove(values)
            seen.update((i, operations[i]) for i in rows)
    if unbound:
        failures.append('%d events signalled with values the compiled signature does not have' % len(unbound))
    missing_sites = sorted({o.native_site for i, o in enumerate(operations)
                            if which[i] in ('program', 'feedback') and i not in seen and o.native_site})
    if missing_sites:
        failures.append('unobserved operations at compiled native_site %s' % ', '.join(map(str, missing_sites)))
    return [seen[i] for i in sorted(seen)]


def main(argv):
    if len(argv) < 3 or argv[1] not in ('generate', 'run', 'cover'):
        sys.stderr.write('usage: mesh-sync-monitor.py generate SIGNATURE.record [--node N] [--observe]\n'
                         '       mesh-sync-monitor.py run SIGNATURE.record [--node N] [--observe] '
                         '[--bridge PID --group GROUP] -p PID\n'
                         '       mesh-sync-monitor.py cover SIGNATURE.record\n')
        return 2
    mode, path, rest = argv[1], argv[2], argv[3:]
    if mode == 'cover':
        return cover(path)
    pid, observe, bridge, group, node = None, False, None, None, None
    while rest:
        if rest[0] == '-p': pid, rest = rest[1], rest[2:]
        elif rest[0] == '--observe': observe, rest = True, rest[1:]
        elif rest[0] == '--bridge': bridge, rest = rest[1], rest[2:]
        elif rest[0] == '--group': group, rest = int(rest[1], 0), rest[2:]
        elif rest[0] == '--node': node, rest = int(rest[1]), rest[2:]
        else:
            sys.stderr.write('mesh-sync-monitor: unknown argument %s\n' % rest[0])
            return 2
    if mode == 'run' and not pid:
        sys.stderr.write('mesh-sync-monitor: run needs -p PID\n')
        return 2
    compiled, processors, tasks, dependences, operations = load(path)
    nodes = sorted({p.index % mesh_graph.NODES for p in processors if KINDS[p.kind] == 'host'})
    if node is None and len(nodes) != 1:
        sys.stderr.write('mesh-sync-monitor: the record holds the threads of nodes %s: --node N\n' % nodes)
        return 2
    node = nodes[0] if node is None else node
    which = classify(processors, operations, node)
    orders = sequences(processors, operations, which, node)
    coverage = {c: which.count(c) for c in CLASSES if c in which}
    if len(orders) > 64:
        sys.stderr.write('mesh-sync-monitor: %d threads on node %d\n' % (len(orders), node))
        return 2
    addresses = not any(COUNT <= operations[i].object < SUBMISSION for order in orders for i in order)
    script, shapes = generate(processors, operations, orders, observe, addresses)
    target_path = os.path.splitext(path)[0] + '.monitor.d'
    open(target_path, 'w').write(script)
    before = []
    if compiled['states'].get('false'):
        before.append('%d synchronization edges with no dependence under them' % compiled['states']['false'])
    if compiled['redundant']:
        before.append('%d redundant synchronization edges' % compiled['redundant'])
    if compiled['unknown']:
        before.append('kernel synchronization not known from the compiled IR: %s' % sorted(compiled['unknown'].items()))
    sys.stderr.write('mesh-sync-monitor: %s written, digest %s, node %s, threads %s (operations an execution, '
                     'executions); operations by class %s; edge states %s\n' % (
                         target_path, compiled['digest'], node, [shape[:2] for shape in shapes], coverage,
                         compiled['states']))
    for reason in before:
        sys.stderr.write('mesh-sync-monitor: fails before running: %s\n' % reason)
    if mode == 'generate':
        return 1 if before else 0
    record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
              'kind': 'synchronization-monitor', 'signature': os.path.abspath(path), 'sha256': compiled['digest'],
              'node': node, 'coverage': coverage, 'destructive': not observe}
    if before and not observe:
        record.update(result='fail', reasons=before)
        with open(REVIEWS, 'a') as reviews:
            reviews.write(json.dumps(record) + '\n')
        return 1
    owners = verbs_owners()
    if pid in owners:
        sys.stderr.write('mesh-sync-monitor: the target holds the verbs device; a stopped verbs owner wedges the node '
                         'until a person pulls its power (AGENTS.md, The kill guard). Observe the program, not the '
                         'bridge.\n')
        return 2
    if bridge and (bridge not in owners or group is None):
        sys.stderr.write('mesh-sync-monitor: --bridge names this node\'s running bridge (%s) and needs --group, the '
                         'program\'s group\n' % sorted(owners))
        return 2
    watcher = None
    if bridge:
        bridge_path = os.path.splitext(path)[0] + '.bridge.d'
        open(bridge_path, 'w').write(bridge_program(group))
        watcher = subprocess.Popen(['sudo', 'dtrace', '-q', '-s', bridge_path, bridge], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
    done = subprocess.Popen(['sudo', 'dtrace'] + ([] if observe else ['-w']) + ['-Z', '-s', target_path, '-p', pid],
                            stdout=subprocess.PIPE, text=True)
    text = []
    for line in done.stdout:
        text.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    status = done.wait()
    raised = []
    if watcher:
        watcher.send_signal(2)
        raised = [tuple(map(int, m.groups())) for m in
                  (re.match(r'mesh-sync-monitor: raise (\d+) (\d+) (\d+) (\d+) (\d+)$', line)
                   for line in watcher.communicate()[0].splitlines()) if m]
    threads, signals, failures = trace_log(''.join(text), operations, orders)
    failures[:0] = before
    if status:
        failures.append('dtrace exited %d' % status)
    observed = observed_operations(processors, operations, which, orders, threads, signals, failures)
    blind = {c: n for c, n in coverage.items() if c not in ('program', 'feedback', 'static')}
    scope = {'program', 'feedback'}
    if bridge:
        made, missing = bridge_match([i for i, c in enumerate(which) if c == 'bridge'], operations, raised)
        if missing:
            failures.append('%d compiled bridge raises not observed (%d observed)' % (len(missing), len(made)))
        blind.pop('bridge', None)
        scope.add('bridge')
        observed += [observed_operation(operations[i], operations[i].primitive, count) for i, count in made]
    expected = mesh_graph.signature(processors, tasks, dependences, [o for o, c in zip(operations, which) if c in scope])
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
    sys.stderr.write('mesh-sync-monitor: %s; %d operations observed at run time; %d kernel records from compiled IR '
                     '(not runtime-observed); other unobserved classes %s\n' % (
                         result, len(observed), coverage.get('static', 0), blind or 'none'))
    record.update(result=result, reasons=failures[:16], blind=blind, runtime_observed=len(observed),
                  compiled_kernel_records=coverage.get('static', 0), observed_sha256=watched['digest'],
                  expected_sha256=expected['digest'], counts_differing=len(differing), dtrace_status=status,
                  bridge=bool(bridge), bridge_raises=len(raised))
    with open(REVIEWS, 'a') as reviews:
        reviews.write(json.dumps(record) + '\n')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
