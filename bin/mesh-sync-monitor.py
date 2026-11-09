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
OBJC = [
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
    ('bridge_start', 'pid$target::mesh_transfers_start:', None, 'arg2', None),
    ('host_publish', 'pid$target::mesh_host_publish:', None, 'arg3 + 1', None),
    ('host_signal', 'objc$target::-setSignaledValue?:', 'arg0', 'arg2', None),
    ('host_wait', 'pid$target::MetalNativeWait:', None, 'arg4 + 1', None),
    ('host_wait', 'pid$target::MetalSentWait:', None, 'arg1 + 1', None),
]
UNLISTED = [
    'objc$target::-waitUntilCompleted:entry',
    'objc$target::-waitUntilScheduled:entry',
    'pid$target::dispatch_semaphore_wait:entry',
    'pid$target::os_sync_wait_on_address:entry',
    'pid$target::os_sync_wait_on_address_with_timeout:entry',
    'pid$target::pthread_cond_wait:entry',
    'pid$target::pthread_cond_timedwait:entry',
]
VALUED = {'submit', 'queue_signal', 'queue_wait', 'host_wait', 'host_notify', 'bridge_start', 'host_publish', 'host_signal'}


def observed_here(processors, o):
    node = {p.index % mesh_graph.NODES for p in processors if KINDS[p.kind] == 'queue'}
    kind = KINDS[processors[o.processor].kind]
    return kind in ('queue', 'host') and processors[o.processor].index % mesh_graph.NODES in node and \
        PRIMITIVES[o.primitive] != 'initial' and not mesh_graph.PLAN_COUNT <= o.object < mesh_graph.PLAN_SUBMISSION


# mesh-graph.h struct mesh_graph_signature `first`/`order`: each processor's operations in its compiled order
def lanes(processors, rows):
    lane, order = {}, {}
    for p, ops in sorted(rows.items()):
        mine = [o for o in ops if observed_here(processors, o)]
        if mine:
            order[len(order)] = mine
    for p, ops in order.items():
        for o in ops:
            lane.setdefault(PRIMITIVES[o.primitive], set()).add(p)
    return order, lane


def generate(processors, rows, window, runs, observe):
    order, home = lanes(processors, rows)
    notes = ['%s is on lanes %s: the monitor orders it on %d only' % (name, sorted(places), min(places))
             for name, places in sorted(home.items()) if len(places) > 1]
    lines = ['#pragma D option quiet', '#pragma D option strsize=256'] + ([] if observe else ['#pragma D option destructive'])
    lines.append('BEGIN { armed = 0; runs = 0; }')
    for p, ops in sorted(order.items()):
        lines.append('BEGIN { n[%d] = %d; cur[%d] = 0; }' % (p, len(ops), p))
        for start in range(0, len(ops), 96):
            body = ['e[%d, %d] = %d; o[%d, %d] = %d; v[%d, %d] = %d; q[%d, %d] = %d;' % (
                p, i, ops[i].primitive, p, i, ops[i].object, p, i, ops[i].value, p, i, ops[i].processor)
                for i in range(start, min(start + 96, len(ops)))]
            lines.append('BEGIN { %s }' % ' '.join(body))
    lines.append('pid$target::%s:entry /runs < %d/ { armed = 1; self->inrun = 1; }' % (window, runs))
    lines.append('pid$target::%s:return /self->inrun/ { self->inrun = 0; runs++; armed = runs < %d; }' % (window, runs))
    entries = []
    for name, probe, obj, val, queue in OBJC:
        p = min(home[name]) if name in home else -1
        entries.append(probe + 'entry')
        lines.append('%sentry /armed/ { self->inapi++; this->p = %d; this->c = %d; this->ob = %s; this->hasob = %d; '
                     'this->va = %s; this->hasva = %d; this->qu = %s; this->hasqu = %d; }' % (
                         probe, p, PRIMITIVES.index(name), obj or '0', obj is not None, val or '0', name in VALUED,
                         queue or '0', queue is not None))
        lines.append('%sreturn /self->inapi/ { self->inapi--; }' % probe)
    shared = ',\n'.join(entries)
    lines.append('%s /armed/ { this->i = this->p < 0 ? 0 : cur[this->p]; this->ok = this->p >= 0 && this->i < n[this->p] && '
                 'e[this->p, this->i] == this->c && (!this->hasva || v[this->p, this->i] == this->va) && (!this->hasob || '
                 'om[this->ob] == 0 || om[this->ob] == o[this->p, this->i] + 1) && (!this->hasqu || qm[this->qu] == 0 || '
                 'qm[this->qu] == q[this->p, this->i] + 1); }' % shared)
    lines.append('%s /armed && this->ok/ { om[this->hasob ? this->ob : 0] = this->hasob ? o[this->p, this->i] + 1 : om[0]; '
                 'qm[this->hasqu ? this->qu : 0] = this->hasqu ? q[this->p, this->i] + 1 : qm[0]; cur[this->p]++; '
                 'printf("mesh-sync-monitor: observed %%d %%d %%d %%d %%d %%d\\n", q[this->p, this->i], this->c, this->i, '
                 'this->hasob ? o[this->p, this->i] : -1, this->va, tid); }' % shared)
    act = 'ustack(); exit(1);' if observe else 'ustack(); stop(); exit(1);'
    lines.append('%s /armed && !this->ok/ { printf("mesh-sync-monitor: unexpected primitive %%d on lane %%d at position %%d '
                 '(expected primitive %%d value %%d, given value %%d) thread %%d\\n", this->c, this->p, this->i, '
                 'this->p >= 0 ? e[this->p, this->i] : -1, this->p >= 0 ? v[this->p, this->i] : -1, this->va, tid); %s }'
                 % (shared, act))
    lines.append('%s /armed && self->inrun && !self->inapi/ { printf("mesh-sync-monitor: unexpected %%s %%s thread %%d\\n", '
                 'probeprov, probefunc, tid); %s }' % (',\n'.join(UNLISTED), act))
    lines.append('END { %s }' % ' '.join('printf("mesh-sync-monitor: lane %d observed %%d of %d\\n", cur[%d]);'
                                         % (p, len(ops), p) for p, ops in sorted(order.items())))
    return '\n'.join(lines) + '\n', notes


def verbs_owners():
    owners = set(subprocess.run(['pgrep', '-f', 'mesh-flow'], capture_output=True, text=True).stdout.split())
    listed = subprocess.run(['launchctl', 'print', 'system/io.mesh.bridge'], capture_output=True, text=True).stdout
    owners.update(re.findall(r'^\s*pid = (\d+)', listed, re.M))
    return owners


def main(argv):
    if len(argv) < 3 or argv[1] not in ('generate', 'run'):
        sys.stderr.write('usage: mesh-sync-monitor.py generate SIGNATURE.record [--window FUNCTION] [--runs N] [--observe]\n'
                         '       mesh-sync-monitor.py run SIGNATURE.record [--window FUNCTION] [--runs N] [--observe] '
                         '(-p PID | -- COMMAND ...)\n')
        return 2
    mode, path, rest = argv[1], argv[2], argv[3:]
    window, runs, pid, command, observe = 'MetalProgramRun', None, None, None, False
    while rest:
        if rest[0] == '--window': window, rest = rest[1], rest[2:]
        elif rest[0] == '--runs': runs, rest = int(rest[1]), rest[2:]
        elif rest[0] == '-p': pid, rest = rest[1], rest[2:]
        elif rest[0] == '--observe': observe, rest = True, rest[1:]
        elif rest[0] == '--': command, rest = rest[1:], []
        else:
            sys.stderr.write('mesh-sync-monitor: unknown argument %s\n' % rest[0])
            return 2
    signature = mesh_graph.from_record(open(path, 'rb').read())
    rebuilt = mesh_graph.forms(signature)
    compiled = rebuilt['digest']
    processors = [mesh_graph.Processor.from_buffer_copy(p) for p in signature.processor[:signature.processors]]
    operations = [mesh_graph.Operation.from_buffer_copy(o) for o in signature.operation[:signature.operations]]
    rows = {p: [operations[signature.order[k]] for k in range(signature.first[p], signature.first[p + 1])]
            for p in range(signature.processors)}
    mesh_graph.free(signature)
    if rebuilt.get('unknown'):
        sys.stderr.write('mesh-sync-monitor: unknown kernel synchronization %s (no recorded source or entry)\n'
                         % (rebuilt['unknown'],))
    executions = 1 + max((o.execution for o in operations), default=0)
    runs = runs or 1
    program, notes = generate(processors, rows, window, runs, observe)
    for note in notes:
        sys.stderr.write('mesh-sync-monitor: %s\n' % note)
    kernel = [o for o in operations if KINDS[processors[o.processor].kind] == 'kernel']
    sys.stderr.write('mesh-sync-monitor: %d kernel counts are static (item 6) and not observed by DTrace\n' % len(kernel))
    unobserved = {}
    for o in operations:
        if KINDS[processors[o.processor].kind] != 'kernel' and not observed_here(processors, o):
            unobserved[(o.processor, PRIMITIVES[o.primitive])] = unobserved.get((o.processor, PRIMITIVES[o.primitive]), 0) + 1
    if unobserved:
        sys.stderr.write('mesh-sync-monitor: compiled, outside these probes (other nodes, the bridge, initial counts, the '
                         'plan\'s count waits inside mesh_transfers_start): %s\n' % sorted(unobserved.items()))
    script = os.path.splitext(path)[0] + '.monitor.d'
    open(script, 'w').write(program)
    sys.stderr.write('mesh-sync-monitor: %s written, digest %s, %d executions in %d calls of %s\n' % (
        script, compiled, executions, runs, window))
    if mode == 'generate':
        return 0
    owners = verbs_owners()
    if (pid and pid in owners) or (command and any('mesh-flow' in c for c in command)):
        sys.stderr.write('mesh-sync-monitor: the target holds the verbs device; a stopped verbs owner wedges the node '
                         'until a person pulls its power (AGENTS.md, The kill guard). Observe the program, not the '
                         'bridge.\n')
        return 2
    target = ['-p', pid] if pid else ['-c', ' '.join(command)]
    done = subprocess.run(['sudo', 'dtrace'] + ([] if observe else ['-w']) + ['-Z', '-s', script] + target,
                          capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    observed = []
    for line in done.stdout.splitlines():
        m = re.match(r'mesh-sync-monitor: observed (-?\d+) (\d+) \d+ (-?\d+) (\d+) \d+$', line)
        if m:
            p, c, obj, value = map(int, m.groups())
            observed.append(mesh_graph.Operation(p, c, NONE, NONE if obj < 0 else obj, 0, value))
    if observe:
        sys.stderr.write('mesh-sync-monitor: --observe: no destructive action; an unexpected operation ends the trace '
                         '(exit) and fails the run, but the process is not stopped at the call (item 4 not exercised)\n')
    unexpected = [line for line in done.stdout.splitlines() if 'unexpected' in line]
    watched = mesh_graph.signature(processors, [], [], observed) if observed else None
    expected = {key: n - unobserved.get(key, 0) for key, n in rebuilt['counts'].items()
                if KINDS[processors[key[0]].kind] != 'kernel' and n > unobserved.get(key, 0)}
    seen = watched['counts'] if watched else {}
    differing = sorted(k for k in set(expected) | set(seen) if expected.get(k, 0) != seen.get(k, 0))
    for key in differing:
        print('mesh-sync-monitor: count %s %s compiled %d observed %d' % (key[0], key[1], expected.get(key, 0),
                                                                          seen.get(key, 0)))
    for name in ('matrix_market', 'dot', 'space_time'):
        print('mesh-sync-monitor: compiled %s\n%s' % (name, rebuilt[name]))
        print('mesh-sync-monitor: observed %s\n%s' % (name, watched[name] if watched else '(no operation observed)'))
    failed = bool(done.returncode or unexpected or differing)
    record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
              'kind': 'synchronization-monitor', 'signature': os.path.abspath(path), 'sha256': compiled,
              'observed_sha256': watched['digest'] if watched else None, 'executions': executions, 'window': window,
              'result': 'fail' if failed else 'pass', 'unexpected': unexpected[:8], 'counts_differing': len(differing),
              'dtrace_status': done.returncode, 'destructive': not observe}
    with open(REVIEWS, 'a') as reviews:
        reviews.write(json.dumps(record) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
