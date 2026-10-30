"""The hourly, run by the collective itself (design/workstation-and-collective-2026-10-29.md, "What a run on the
collective needs"). Every node's launchd job (bin/mesh-hourly-agent.sh) starts `node` at the configured minute; no
machine launches, holds or records another's run.
- `node` takes the run's minute and both repositories' main commits, moves this node's mesh checkout to the mesh
  commit, and runs the committed `collective`.
- `collective` holds the election, a top-1 (the operator, 2026-10-29). Each linked part's representative, its smallest
  node number (every member reads it from the same bridge map), announces the part's capacity over mDNS. Every node
  takes the largest capacity it heard, ties to the smaller representative. Only the winning part's members prepare
  and run, each its own rank, each keeping its own record (~/.cache/mesh/hourly). A node that hears no other node by
  any means closes the run with a warning.
- `read` is any reader's: it gathers the members' records into one ledger. It prepares, starts and restarts nothing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import zlib

from mesh_source import checkout, git

HOURLY = Path.home() / '.cache/mesh/hourly'
# seconds after the run's minute: a node that starts later than LATE cannot hear this hour's election. Until
# BROWSE_UNTIL every node announces itself (_meshrank) and each part's representative its part (_meshpart), and every
# node listens; from then until LIST_UNTIL the winning part's representative announces the run's members and commits
# (_meshrun), which its part's members hear until LISTEN_UNTIL. dns-sd writes what it heard to a pipe only as it
# exits, so each phase is a fixed window.
LATE, BROWSE_UNTIL, LISTEN_UNTIL, LIST_UNTIL = 30, 40, 55, 60
SPEAKERS = []
PAIRING_SECONDS = 30


def workload_configuration(microbench_root):
    """The hourly's workloads (schema 3): each an end-to-end program the elected part runs, on the part its paired
    cables link at the workload's start (run_rank). None names, counts or selects nodes."""
    raw = json.loads((Path(microbench_root) / 'tools/mesh/hourly/workloads.json').read_text())
    if raw.get('schema') != 3 or not raw.get('workloads'):
        raise ValueError(f'tools/mesh/hourly/workloads.json is schema {raw.get("schema")}; mesh reads schema 3 `workloads`')
    return raw['workloads']


def compact(record):
    """Completion time and useful rate, in the committed configuration's shape order."""
    def rounded(value):
        if value is None:
            return None
        number = float(format(value, '.3g'))
        return int(number) if number.is_integer() else number
    rows, incomplete, errors = {}, {}, []
    for item in record['workloads']:
        values = [
            [rounded(None if (t := m.get('function_seconds', m.get('amortized_seconds_per_call'))) is None else t * 1e6),
             rounded(m['useful_gigabytes_per_second']),
             *([round(m['aggregate_rows_per_second'])] if 'aggregate_rows_per_second' in m else [])]
            for m in item['metrics']]
        if item['status'] == 'completed':
            rows[item['tag']] = values
        else:
            # each rank's phases (configuration.py: placed, prepared), wall-clock seconds after the first rank's placement
            phases = {rank: value.get('phases', {}) for rank, value in item.get('ranks', {}).items()}
            first = min((t for p in phases.values() for t in p.values()), default=0)
            incomplete[item['tag']] = {'status': item['status'], 'us,GB/s': values,
                                       **({'phases': {r: {k: round(t - first, 2) for k, t in p.items()} for r, p in phases.items()}}
                                          if first else {})}
            failures = [[item['tag'] + '/' + rank, value['error'][:160]]
                        for rank, value in item.get('ranks', {}).items() if value.get('error')]
            errors.extend(failures or [[item['tag'], item.get('error', 'incomplete')[:160]]])
    errors.extend([e.get('stage', 'driver'), (e.get('error') or e.get('stderr') or f'exit {e.get("exit", "unknown")}')[-160:]]
                  for e in record['errors'])
    errors.extend([name, node.get('error', 'installation failed')[-160:]]
                  for name, node in record['nodes'].items() if node['status'] != 'installed')
    result = {'schema': 3, 'commits': record['commits'], 'unix': int(record['started_unix']),
              'nodes': [r['number'] for r in record['nodes'].values() if r['status'] == 'installed'],
              # each node's preparation: its whole and its build steps, seconds
              'prepared': {r['number']: r['seconds'] for r in record['nodes'].values() if r.get('seconds')},
              'us,GB/s': rows, **({'incomplete': incomplete} if incomplete else {}),
              **({'absent': record['absent']} if record.get('absent') else {}),
              'loaded': sorted({commit for item in record['workloads'] for commit in item.get('loaded', ())}),
              **({'overlapped': overlapped} if (overlapped := sorted({'@'.join(token.split(':')[:2]) for item in record['workloads']
                                                                       for token in item.get('overlapped', ())})) else {})}
    if record['status'] != 'completed':
        result['errors'] = errors or [['run', 'incomplete']]
    return result


class Stopped(BaseException):
    """The run's bound (SIGALRM, etc/mesh-nodes.json run_seconds) or a stop: SIGTERM, SIGHUP or SIGINT."""


def quiet():
    """No further stop: the bound is cancelled and the stop signals ignored, so at most one Stopped is raised."""
    signal.alarm(0)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGALRM):
        signal.signal(s, signal.SIG_IGN)


def stop(signum, _):
    quiet()
    raise Stopped('the run reached run_seconds' if signum == signal.SIGALRM else f'stopped by {signal.Signals(signum).name}')



def rank_results(outputs):
    result = {}
    for name, output in outputs.items():
        lines = output['diagnostics']
        code = output['exit']
        values, observations, phases = [], [], {}
        for value in output['records']:
            if 'observations' in value:
                observations.extend(value['observations'])
            elif 'phase' in value:
                phases[value['phase']] = value['unix']
            else:
                values.append(value)
        failures = [line.strip() for line in lines if any(word in line.lower()
                    for word in ('error', 'exception', 'expired', 'deadline', 'abort', 'killed', 'segmentation'))
                    and not line.lstrip().startswith(('raise ', 'except ', 'frame #', 'Exception raised from '))]
        detail = failures[-1] if failures else '; '.join(line.strip() for line in lines[-12:]
                 if line.strip() and 'resource_tracker' not in line and 'warnings.warn' not in line)
        for index, line in enumerate(lines):
            if line.startswith('Exception raised from '):
                detail = next((prior.strip() for prior in reversed(lines[:index]) if prior.strip()), detail)
                break
        result[name] = {'exit': code, 'measurements': values, 'observations': observations, 'phases': phases,
                        'error': f'exit {code}: {detail}' if code else ''}

    return result




def write(record):
    """This node's record of the run, replaced whole as it grows; its line in log.jsonl once."""
    out = HOURLY / record['run']
    out.mkdir(parents=True, exist_ok=True)
    first = not (out / 'result.json').exists()
    (out / 'result.json.new').write_text(json.dumps(record, separators=(',', ':')) + '\n')
    os.replace(out / 'result.json.new', out / 'result.json')
    if first:
        with (HOURLY / 'log.jsonl').open('a') as log:
            log.write(json.dumps({'run': record['run'], 'node': record['node']}, separators=(',', ':')) + '\n')


def node():
    """launchd's start (bin/mesh-hourly-agent.sh): the run's minute, both repositories' main commits as this node reads
    them then, and this node's mesh checkout moved to the mesh commit; then the committed code's `collective`."""
    from mesh_nodes import load
    cluster = load()
    now = int(time.time())
    minute = now - (now - 60 * cluster.hourly_minute) % 3600
    # bounded from the start: a fetch that stalls ends this run (SIGALRM's default) rather than holding the job, which
    # launchd never starts twice; the alarm survives the exec, where collective sets its own
    signal.alarm(cluster.run_seconds)
    record = {'schema': 4, 'run': time.strftime('%Y%m%dT%H%MZ', time.gmtime(minute)), 'minute': minute,
              'node': socket.gethostname(), 'commits': {}, 'status': 'starting', 'errors': []}
    try:
        record['commits'] = {name: git(Path.home(), 'ls-remote', url, 'refs/heads/main').split()[0].decode()
                             for name, url in cluster.upstreams.items()}
        mesh = Path.home() / cluster.checkouts / 'mesh/current'
        checkout(cluster.upstreams['mesh'], record['commits']['mesh'], mesh, 60)
    except Exception as error:
        record.update(status='failed', errors=[{'stage': 'checkout', 'error': str(error)}])
        write(record)
        return 1
    os.execv(sys.executable, [sys.executable, str(mesh / 'rdma/mesh_hourly.py'), 'collective', json.dumps(record)])


def part_of(report, own):
    """This node's linked part in its bridge's map, by node number, and its capacity: the members' region bytes, from
    the bridge's own records of them."""
    from mesh_nodes import parts
    cables = {tuple(sorted(map(str, link[:2]))) for link in report['map']['links']}
    part = next(p for p in parts({str(x) for x in report['map']['nodes']} | {own}, cables) if own in p)
    regions = {str(row['node']): int(row.get('region', 0)) for row in report.get('nodes', [])}
    return sorted(part, key=int), sum(regions.get(number, 0) for number in part)


def speak(instance, kind, until, **txt):
    """An mDNS announcement of `kind` until `until` (unix seconds), ended by a timer or at this run's end (silence)."""
    import threading
    p = subprocess.Popen(['dns-sd', '-R', instance, kind, 'local', '9', *(f'{k}={v}' for k, v in txt.items())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    SPEAKERS.append(p)
    timer = threading.Timer(max(0.0, until - time.time()), lambda: silence([p]))
    timer.daemon = True
    timer.start()
    return p


def silence(speakers=None):
    for p in list(SPEAKERS if speakers is None else speakers):
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        p.wait()
        if p in SPEAKERS:
            SPEAKERS.remove(p)


def browse(kind, until):
    """Every TXT record of the mDNS service `kind` heard on any network until `until` (unix seconds), by instance
    (dns-sd -Z's zone lines)."""
    p = subprocess.Popen(['dns-sd', '-Z', kind, 'local'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         start_new_session=True)
    try:
        output = p.communicate(timeout=max(0.0, until - time.time()))[0]
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        output = p.communicate()[0]
    heard = {}
    for line in output.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[1] == 'TXT':
            heard[fields[0].split('.')[0]] = dict(item.split('=', 1) for item in shlex.split(fields[2]) if '=' in item)
    return heard


def elect(record, report, own):
    """The top-1 (module docstring), then the winning part's members and commits from its representative. Records what
    this node heard and returns whether it is one of the run's members."""
    from mesh_nodes import parallel
    part, capacity = part_of(report, own)
    start, run = record['minute'], record['run']
    commits = record['commits']
    speak(own, '_meshrank._tcp', start + BROWSE_UNTIL, run=run, part=part[0], mesh=commits['mesh'],
          bench=commits['metal-microbench'])
    if own == part[0]:
        speak(own, '_meshpart._tcp', start + BROWSE_UNTIL, run=run, capacity=capacity, members=len(part))
    parts, beacons, ranks = parallel([lambda: browse('_meshpart._tcp', start + BROWSE_UNTIL),
                                      lambda: browse('_meshnode._tcp', start + BROWSE_UNTIL),
                                      lambda: browse('_meshrank._tcp', start + BROWSE_UNTIL)])
    silence()
    heard = {rep: {'capacity': int(t['capacity']), 'members': int(t['members'])} for rep, t in parts.items()
             if t.get('run') == run and t.get('capacity', '').isdigit() and t.get('members', '').isdigit()}
    heard[part[0]] = {'capacity': capacity, 'members': len(part)}
    record['beacons'] = {t['node']: t.get('node_id', '') for t in beacons.values() if t.get('node')}
    taking_part = {n: t for n, t in ranks.items() if t.get('run') == run}
    winner = max(heard, key=lambda rep: (heard[rep]['capacity'], -int(rep)))
    record['election'] = {'part': part, 'capacity': capacity, 'heard': heard, 'winner': winner,
                          'taking_part': sorted(taking_part, key=int)}
    if len(part) == 1 and not set(record['beacons']) - {own}:
        record['status'] = 'unknown'
        record['warning'] = ("heard no other node by any means: its bridge's map holds only itself and no beacon "
                             "answered, so it does not know what is going on")
        return False
    if winner != part[0]:
        record['status'] = 'not elected'
        record['warning'] = (f'the part of {winner} ({heard[winner]["members"]} nodes, {heard[winner]["capacity"]} bytes) '
                             f'won over its own ({len(part)} nodes, {capacity} bytes)')
        return False
    # nodes whose beacons answered but whose parts announced nothing (their jobs did not start, or started late) could
    # make a part larger than this one; a part outnumbered by them does not know it won
    unaccounted = len(set(record['beacons']) | set(part)) - sum(v['members'] for v in heard.values())
    if unaccounted >= len(part):
        record['status'] = 'unknown'
        record['warning'] = (f'{unaccounted} nodes answered by beacon but no part announced them, as many as its own '
                             f'part\'s {len(part)}, so it does not know what is going on')
        return False
    if own == part[0]:
        # the run's members: the part's nodes heard taking part in this run (a node late, without a checkout or without
        # its job never announced itself), and the representative's commits, the same for every member
        from mesh_nodes import parts
        present = {n for n, t in taking_part.items() if t.get('part') == own and n in part} | {own}
        # of those, the largest set the cables still link without the absent ones (ties to the one holding this node)
        cables = {tuple(sorted(map(str, link[:2]))) for link in report['map']['links']}
        members = sorted(max(parts(present, cables), key=lambda g: (len(g), own in g)), key=int)
        chunks = [','.join(members[i:i + 20]) for i in range(0, len(members), 20)]
        speak(run, '_meshrun._tcp', start + LIST_UNTIL, run=run, rep=own, mesh=commits['mesh'],
              bench=commits['metal-microbench'], **{f'm{i}': chunk for i, chunk in enumerate(chunks)})
        if own not in members:
            time.sleep(max(0.0, start + LIST_UNTIL - time.time()))  # the members hear the list before it is silenced
    else:
        listed = next((t for t in browse('_meshrun._tcp', start + LISTEN_UNTIL).values()
                       if t.get('run') == run and t.get('rep') == part[0]), None)
        if not listed:
            record['status'] = 'failed'
            record['warning'] = f'its part\'s representative {part[0]} announced no members for this run'
            return False
        members = [n for i in range(len(listed)) if f'm{i}' in listed for n in listed[f'm{i}'].split(',') if n]
        record['commits'] = {'mesh': listed.get('mesh', ''), 'metal-microbench': listed.get('bench', '')}
    record['members'] = members
    if own not in members:
        record['status'] = 'not a member'
        record['warning'] = f'its part\'s representative {part[0]} did not hear it take part'
        return False
    return True


def fabric_address(number, record, cluster):
    """A node's fabric address: bin/mesh-router-init.sh's from the host UUID its beacon announced, else its configured
    one."""
    uuid = record.get('beacons', {}).get(number)
    if uuid:
        h = hashlib.sha256(uuid.encode()).hexdigest()[:20]
        return 'fd6d:6573:68:' + ':'.join(f'{int(h[i:i + 4], 16) | 0x1000:x}' for i in range(0, 20, 4))
    if number in cluster.members:
        return cluster.members[number].fabric
    raise RuntimeError(f'node {number}: no beacon heard and no configured fabric address')


def run_rank(item, me, record, ordinal, cluster, bench, known):
    """This node's rank of one workload, on the elected part as its paired cables link it at the workload's start."""
    import mesh_nodes as mesh
    tag = item['tag']
    end = min(time.time() + item.get('timeout', 300), record['deadline_unix'])
    result = {'tag': tag, 'status': 'preparing'}
    try:
        # a failed workload's ranks leave links re-pairing for a few seconds; a node gone never pairs, so the wait is
        # that long and no longer, and the case runs on the part linked by then
        mesh.wait_paired([me], None, min(end, time.time() + PAIRING_SECONDS))
        part, _ = part_of(me.stat()[1], me.number)
        members = record['members']
        k = members.index(me.number)
        result.update(members=members, rank=k)
        unlinked = [number for number in members if number not in part]
        if unlinked:
            # every member's map shows the same loss, so every member records this case unrun
            raise RuntimeError(f'members {unlinked} not linked at the workload\'s start; not run')
        source = me.source
        env = (f'PYTHONPATH={mesh.home_path(source + "/rdma")}:{mesh.home_path(source + "/rdma/torch-mesh")} '
               'PYTHONDONTWRITEBYTECODE=1 ')
        command = f'{cluster.python} {mesh.home_path(source + "/examples/" + item["program"] + ".py")}'
        history = known.get(tag, [])[-64:]
        if item.get('configuration'):
            operands = json.loads((bench / item['configuration']).read_text())
            operands.update({key: item[key] for key in ('steps', 'rows') if key in item})
            operands['observations'] = history
            command += ' -'
        else:
            operands = history
            env += 'MESH_OBSERVATIONS_STDIN=1 '
        result['status'] = 'running'
        port = cluster.rank_port + zlib.crc32(record['run'].encode()) % 20 * 20 + ordinal
        output = mesh.rank(me, k, len(members), fabric_address(members[0], record, cluster), tag, '.', command, port,
                           alarm=max(1, int(end - time.time())), env=env, input=json.dumps(operands).encode(),
                           run=record['token'])
        rank = rank_results({f'rank{k}': output})[f'rank{k}']
        known[tag] = (known.get(tag, []) + rank.pop('observations'))[-64:]
        result.update(result=rank, loaded=sorted({v['loaded'] for v in output['records'] if 'loaded' in v}),
                      overlapped=sorted({t for v in output['records'] for t in v.get('overlap', ())}),
                      status='completed' if rank['exit'] == 0 else 'failed')
        if rank['exit']:
            result['error'] = rank['error']
    except Exception as error:
        result.update(status='failed', error=str(error))
        print(f'{tag}: {result["error"]}', file=sys.stderr, flush=True)
    return result


def collective(record):
    """The committed code's run on this node (module docstring): the election, then, in the winning part, this node's
    preparation and its rank of every workload. Its record is written as it grows."""
    import mesh_nodes as mesh
    cluster = mesh.load()
    for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, stop)
    record['deadline_unix'] = record['minute'] + cluster.run_seconds - 60
    signal.alarm(max(1, int(record['minute'] + cluster.run_seconds - time.time())))
    stage = 'election'
    try:
        local = mesh.Node(cluster, '-', 'localhost', local=True, deadline=record['deadline_unix'])
        local.source = f'{cluster.checkouts}/mesh/current'
        _, report = local.stat()
        if not report.get('up'):
            raise RuntimeError(f'its bridge is not up: {report}')
        own = str(report['node'])
        me = mesh.Node(cluster, own, 'localhost', local=True, deadline=record['deadline_unix'])
        me.source = local.source
        record.update(node=own, token=f'{record["run"]}:{own}:{os.getpid()}:{mesh.started(os.getpid())}')
        if 'members' not in record:
            if time.time() > record['minute'] + LATE:
                record.update(status='late', warning=f'started {int(time.time() - record["minute"])} s after its '
                                                      f'minute, past the election ({LATE} s)')
                return 0
            mesh_commit = record['commits']['mesh']
            if not elect(record, report, own):
                return 0
            if record['commits']['mesh'] != mesh_commit:
                # the representative read another mesh commit: this node runs it, by its committed code
                checkout(cluster.upstreams['mesh'], record['commits']['mesh'], Path.home() / me.source, 120)
                silence()
                os.execv(sys.executable, [sys.executable, str(Path.home() / me.source / 'rdma/mesh_hourly.py'),
                                          'collective', json.dumps(record)])
        write(record)
        stage = 'preparation'
        record['status'] = 'preparing'
        bench = Path.home() / cluster.checkouts / 'metal-microbench/current'
        # only the hourly's own directory: its workloads and their configurations
        checkout(cluster.upstreams['metal-microbench'], record['commits']['metal-microbench'], bench,
                 max(1, record['deadline_unix'] - time.time()), paths=('tools/mesh/hourly',))
        began = time.time()
        code, output = me.py('prepare', timeout=max(1, record['deadline_unix'] - time.time()))
        if code:
            raise RuntimeError(f'preparation failed ({code}): {output[-600:]}')
        record['prepared'] = next((json.loads(line) for line in output.splitlines() if line.startswith('{"bridge"')), {})
        record['prepared']['seconds'] = {'whole': round(time.time() - began, 1), **record['prepared'].get('seconds', {})}
        # a preparation that restarted this node's bridge, or a peer's, leaves links pairing again: each node waits for
        # its own (at most 300 s, wait_paired), so the members' maps agree when the first workload reads them
        record['paired'] = mesh.wait_paired([me], None, record['deadline_unix'])
        # and until its bridge's map links every member (at most 300 s), so no workload starts on a part still forming
        for _ in range(60):
            if set(record['members']) <= set(part_of(me.stat()[1], own)[0]):
                break
            time.sleep(5)
        items = workload_configuration(bench)
        history = HOURLY / 'history.json'
        known = json.loads(history.read_text()) if history.exists() else {}
        record.update(status='running', workloads=[])
        for ordinal, item in enumerate(items):
            stage = item['tag']
            record['workloads'].append(run_rank(item, me, record, ordinal, cluster, bench, known))
            write(record)
        history.write_text(json.dumps(known))
        record['status'] = 'completed' if all(w['status'] == 'completed' for w in record['workloads']) else 'failed'
    except (Exception, Stopped) as error:
        quiet()
        mesh.release()
        record['status'] = 'stopped' if isinstance(error, Stopped) else 'failed'
        record['errors'].append({'stage': stage, 'error': str(error), 'traceback': traceback.format_exc()})
    finally:
        quiet()
        silence()
        record['finished_unix'] = time.time()
        write(record)
    return 0 if record['status'] in ('completed', 'not elected') else 1


def merged(records, cluster):
    """One run's ledger row (compact) from its nodes' own records: the elected part's members, each workload's ranks
    gathered across them, and every other configured node's reason for being absent."""
    from mesh_statistics import aggregate
    # the parts that ran, by representative: one, unless two parts heard each other differently; then the top-1 of
    # them is the run and the other part's nodes are named as having run apart
    groups = {}
    for n, r in records.items():
        if 'election' in r and r.get('status') not in ('unknown', 'late', 'not elected', 'not a member'):
            groups.setdefault(r['election']['part'][0], {})[n] = r
    chosen = max(groups, key=lambda rep: (next(iter(groups[rep].values()))['election']['capacity'], -int(rep)), default=None)
    ran = groups.get(chosen, {})
    first = next(iter(ran.values()), next(iter(records.values())))
    absent = {n: r.get('warning') or (f'ran apart, in the part of {r["election"]["part"][0]}: it did not hear the part '
                                      f'of {chosen}' if r.get('election') and r.get('status') not in ('unknown', 'late', 'not elected', 'not a member')
                                      else f'{r.get("status")}: ' + '; '.join(e.get('error', '')[-160:] for e in r.get('errors', [])))
              for n, r in records.items() if n not in ran}
    absent.update({n: 'no record of this run (unreachable, or its job did not start)'
                   for n in cluster.members if n not in records})
    nodes = {n: {'number': n, 'status': 'installed' if r.get('prepared') else 'failed', 'seconds': r.get('prepared', {}).get('seconds'),
                 'error': '; '.join(e.get('error', '') for e in r.get('errors', []))[-160:] or 'preparation incomplete'}
             for n, r in ran.items()}
    workloads = []
    tags = list(dict.fromkeys(w['tag'] for r in ran.values() for w in r.get('workloads', [])))
    for tag in tags:
        item = {'tag': tag}
        rows = [w for r in ran.values() for w in r.get('workloads', []) if w['tag'] == item['tag']]
        ranks = {f'rank{w["rank"]}': w['result'] for w in rows if 'result' in w}
        members = {m for w in rows for m in w.get('members', [])}
        done = {m for r in ran.values() for w in r.get('workloads', []) if w['tag'] == item['tag'] and w['status'] == 'completed'
                for m in [r['node']]}
        status = 'completed' if members and members <= done else 'failed'
        workloads.append({'tag': item['tag'], 'status': status, 'ranks': ranks,
                          'metrics': aggregate({k: v for k, v in ranks.items() if 'measurements' in v}),
                          'loaded': sorted({c for w in rows for c in w.get('loaded', ())}),
                          'overlapped': sorted({t for w in rows for t in w.get('overlapped', ())}),
                          'error': next((w['error'] for w in rows if w.get('error')), 'incomplete')})
    errors = [{'stage': f'{n}/{e.get("stage")}', 'error': e.get('error', '')} for n, r in ran.items() for e in r.get('errors', [])]
    status = 'completed' if ran and not errors and all(w['status'] == 'completed' for w in workloads) else 'failed'
    return compact({'commits': first.get('commits', {}), 'started_unix': first['minute'], 'status': status, 'errors': errors,
                    'nodes': nodes, 'workloads': workloads, 'absent': absent})


def read(output_root):
    """Any reader's (read only): every reachable configured node's kept records, each beside its run's ledger row
    (output_root/<run>/nodes/<number>.json), the row (output_root/<run>/result.json) and its line in log.jsonl."""
    import mesh_nodes as mesh
    cluster = mesh.load()
    output_root = Path(output_root).resolve()
    here = None
    try:
        local = mesh.Node(cluster, '-', 'localhost', local=True, deadline=time.time() + 30)
        local.source = f'{cluster.checkouts}/mesh/current'
        here = str(local.stat()[1].get('node'))
    except Exception:
        pass
    listing = ('cd ~/.cache/mesh/hourly 2>/dev/null || exit 0; for f in $(ls -dt */result.json | head -48); '
               'do printf "%s\\t" "${f%/result.json}"; cat "$f"; done')

    def fetch(member):
        node = mesh.Node(cluster, member.number, 'localhost' if member.number == here else member.management,
                         local=member.number == here, deadline=time.time() + 60)
        try:
            return member.number, node.run(listing, timeout=60)[1]
        except Exception as error:
            print(f'{member.name}: {error}', file=sys.stderr)
            return member.number, ''
    runs = {}
    for number, text in mesh.parallel([lambda m=m: fetch(m) for m in cluster.members.values()]):
        for line in text.splitlines():
            run, _, body = line.partition('\t')
            if body.startswith('{'):
                runs.setdefault(run, {})[number] = json.loads(body)
    logged = (output_root / 'log.jsonl').read_text() if (output_root / 'log.jsonl').exists() else ''
    for run, fetched in sorted(runs.items()):
        out = output_root / run
        (out / 'nodes').mkdir(parents=True, exist_ok=True)
        # a node that does not answer this time keeps the record an earlier read kept of it, so its run is not
        # re-merged without it
        records = {path.stem: json.loads(path.read_text()) for path in (out / 'nodes').glob('*.json')}
        records.update(fetched)
        for number, record in fetched.items():
            (out / 'nodes' / f'{number}.json').write_text(json.dumps(record, separators=(',', ':')) + '\n')
        (out / 'result.json').write_text(json.dumps(merged(records, cluster), separators=(',', ':')) + '\n')
        if str(out / 'result.json') not in logged:
            with (output_root / 'log.jsonl').open('a') as log:
                log.write(json.dumps({'result': str(out / 'result.json')}, separators=(',', ':')) + '\n')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('node', 'collective', 'read'))
    parser.add_argument('operand', nargs='?')
    args = parser.parse_args(argv)
    if args.action == 'node':
        return node()
    if args.action == 'collective':
        return collective(json.loads(args.operand))
    return read(args.operand)


if __name__ == '__main__':
    sys.exit(main())
