"""Run committed mesh/microbench Git checkouts fetched from each repository's origin."""
import argparse
import contextlib
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

from mesh_source import checkout, prune, revision, upstream, verify

REQUIRED = ('rdma/mesh_hourly.py', 'rdma/mesh_nodes.py', 'rdma/mesh_source.py',
            'rdma/mesh_install.py', 'rdma/mesh-install-state.c')
FIGURES = ('steady_step_microseconds', 'step_microseconds', 'crossing_microseconds',
           'steady_tick_microseconds', 'tick_microseconds', 'aggregate_rows_per_second',
           'steady_aggregate_rows_per_second', 'per_stream_steps_per_second')


def workload_configuration(microbench_root):
    """One canonical configuration for hourly and interactive comparison entrypoints: schema 2, each workload placed
    by etc/mesh-nodes.json roles (`place`), none by node (`nodes`, `rank_limit`)."""
    raw = json.loads((Path(microbench_root) / 'tools/mesh/hourly/workloads.json').read_text())
    items = raw.get('placed_workloads') or []
    unplaced = [item.get('tag') for item in items if 'place' not in item or {'nodes', 'rank_limit'} & item.keys()]
    if raw.get('schema') != 2 or not items or unplaced:
        raise ValueError(f'tools/mesh/hourly/workloads.json is schema {raw.get("schema")}; mesh needs schema 2, its '
                         f'placed_workloads each with a `place` and no `nodes` or `rank_limit` (unplaced: {unplaced})')
    return items


def measurement_values(record, tag):
    """Read both historical ledger rows and current structured workload results."""
    if 'result' in record:
        record = json.loads(Path(record['result']).read_text())
    if record.get('schema') == 3:
        item = record['us,GB/s'].get(tag)
        if not item or len(item) != 1 or item[0] is None:
            return {}
        us, rate, *rows = item[0]
        return {'function_seconds': [us / 1e6, None],
                'useful_gigabytes_per_second': [rate, None],
                **({'aggregate_rows_per_second': [rows[0], None]} if rows else {})}
    if tag in record:
        return record[tag]
    workload = next((item for item in record.get('workloads', []) if item['tag'] == tag), {})
    ranks = workload.get('ranks', {})
    figures = {}
    for row in ranks.get('rank0', {}).get('measurements', []):
        figures.update({key: [row[key], None] for key in FIGURES if key in row})
    return dict(figures, ranks={name: f'exit {value["exit"]}' for name, value in ranks.items()})


def compact(record):
    """Completion time and useful rate, in the committed configuration's shape order."""
    def rounded(value):
        number = float(format(value, '.3g'))
        return int(number) if number.is_integer() else number
    rows, incomplete, errors = {}, {}, []
    for item in record['workloads']:
        values = [
            [rounded(m.get('function_seconds', m.get('amortized_seconds_per_call')) * 1e6),
             rounded(m['useful_gigabytes_per_second']),
             *([round(m['aggregate_rows_per_second'])] if 'aggregate_rows_per_second' in m else [])]
            for m in item['metrics']]
        if item['status'] == 'completed':
            rows[item['tag']] = values
        else:
            incomplete[item['tag']] = {'status': item['status'], 'us,GB/s': values}
            failures = [[item['tag'] + '/' + rank, value['error'][:160]]
                        for rank, value in item.get('ranks', {}).items() if value.get('error')]
            errors.extend(failures or [[item['tag'], item.get('error', 'incomplete')[:160]]])
    errors.extend([e.get('stage', 'driver'), (e.get('error') or e.get('stderr') or f'exit {e.get("exit", "unknown")}')[-160:]]
                  for e in record['errors'])
    errors.extend([name, node.get('error', 'installation failed')[-160:]]
                  for name, node in record['nodes'].items() if node['status'] != 'installed')
    result = {'schema': 3, 'commits': record['commits'], 'unix': int(record['started_unix']),
              'nodes': [r['number'] for r in record['nodes'].values() if r['status'] == 'installed'],
              'us,GB/s': rows, **({'incomplete': incomplete} if incomplete else {}),
              **({'absent': record['absent']} if record.get('absent') else {}),
              'loaded': sorted({commit for item in record['workloads'] for commit in item.get('loaded', ())}),
              **({'overlapped': overlapped} if (overlapped := sorted({'@'.join(token.split(':')[:2]) for item in record['workloads']
                                                                       for token in item.get('overlapped', ())})) else {})}
    if record['status'] != 'completed':
        result['errors'] = errors or [['run', 'incomplete']]
    return result


class Stopped(Exception):
    """The run's bound (SIGALRM, etc/mesh-nodes.json run_seconds) or a stop: SIGTERM, SIGHUP or SIGINT."""


def quiet():
    """No further stop: the bound is cancelled and the stop signals ignored, so at most one Stopped is raised."""
    signal.alarm(0)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGALRM):
        signal.signal(s, signal.SIG_IGN)


def stop(signum, _):
    quiet()
    raise Stopped('the run reached run_seconds' if signum == signal.SIGALRM else f'stopped by {signal.Signals(signum).name}')


def ledger(output_root, tag, result):
    """The run's result.json and its one line in log.jsonl."""
    out = Path(output_root).resolve() / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
    with (out.parent / 'log.jsonl').open('a') as log:
        log.write(json.dumps({'result': str(out / 'result.json')}, separators=(',', ':')) + '\n')


def launcher(mesh_root, microbench_root, output_root):
    """Every hourly leaves one ledger line, a run that cannot start too."""
    from mesh_nodes import load
    began, code = time.time(), None
    try:
        code = _launch(mesh_root, microbench_root, output_root, load())
    except Exception as error:
        ledger(output_root, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(began)) + '-launch',
               {'schema': 3, 'commits': {}, 'unix': int(began), 'nodes': [], 'us,GB/s': {}, 'errors': [['launch', str(error)]]})
    return 1 if code is None else code


def _launch(mesh_root, microbench_root, output_root, cluster):
    """Select main commits and launch the driver from its upstream Git checkout."""
    from mesh_nodes import started
    output_root = Path(output_root).resolve()
    began = time.time()
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(began)) + f'-{time.time_ns() % 1000000000:09d}'
    out = output_root / tag
    out.mkdir(parents=True)
    for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, stop)
    signal.alarm(cluster.run_seconds)
    phase = 'checkout'
    record = {'schema': 2, 'run': tag, 'started_unix': began, 'deadline_unix': began + cluster.run_seconds - 60,
              'token': f'{tag}:{socket.gethostname()}:{os.getpid()}:{started(os.getpid())}', 'commits': {}, 'checkouts': {},
              'status': 'failed', 'errors': [], 'nodes': {}, 'workloads': [],
              'launcher': {'file': str(Path(__file__).resolve()),
                           'role': 'select commits; target helpers execute from upstream Git checkouts'}}
    try:
        roots = {'mesh': Path(mesh_root).resolve(), 'metal-microbench': Path(microbench_root).resolve()}
        # Resolve both main commits before fetching either. Later commits or
        # an interactive branch checkout cannot change this invocation's inputs.
        record['commits'] = {name: revision(root, 'refs/heads/main') for name, root in roots.items()}
        record['upstreams'] = {name: upstream(root) for name, root in roots.items()}
        for name, root in roots.items():
            target = Path.home() / cluster.checkouts / name / record['commits'][name]
            checkout(record['upstreams'][name], record['commits'][name], target, max(1, record['deadline_unix'] - time.time()))
            target.with_name(target.name + '.used').touch()
            prune(target.parent, cluster.kept)
            record['checkouts'][name] = {'path': str(target)}
        committed_mesh = Path(record['checkouts']['mesh']['path'])
        committed_bench = Path(record['checkouts']['metal-microbench']['path'])
        absent = [name for name in REQUIRED if not (committed_mesh / name).is_file()]
        if absent:
            raise RuntimeError(f'pinned mesh commit lacks shared hourly prerequisites: {absent}')
        config = committed_bench / 'tools/mesh/hourly/workloads.json'
        record['configuration'] = {'path': str(config.relative_to(committed_bench))}
        history = output_root / 'history.json'
        record['history'] = known = json.loads(history.read_text()) if history.exists() else {}
        phase = 'build'
        subprocess.run(['make', '-s', '-C', str(committed_mesh / 'rdma'), 'statistics'],
                       check=True, stdout=subprocess.DEVNULL)
        phase = 'driver'
        child = subprocess.Popen([sys.executable, str(committed_mesh / 'rdma/mesh_hourly.py'), 'execute'], cwd=committed_mesh,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
        stopped = None
        try:
            stdout = child.communicate(json.dumps(record))[0]
        except Stopped as error:
            # The driver ends its ranks, each guard confirming, and returns its record; past a minute its session ends.
            stopped = error
            os.kill(child.pid, signal.SIGTERM)
            try:
                stdout = child.communicate(timeout=60)[0]
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                stdout = child.communicate()[0]
        if stdout:
            record = json.loads(stdout)
        if stopped:
            raise stopped
        record.pop('history', None)
        record['driver_exit'] = child.returncode
        if child.returncode:
            record['status'] = 'failed'
            if not record['errors'] and not any(item['status'] != 'completed' for item in record['workloads']):
                record['errors'].append({'stage': 'committed-driver', 'exit': child.returncode})
        quiet()
    except Exception as error:
        quiet()
        record['errors'].append({'stage': phase, 'error': str(error), 'traceback': traceback.format_exc()})
    record['finished_unix'] = time.time()
    if record['workloads']:
        for item in record['workloads']:
            known[item['tag']] = (known.get(item['tag'], []) + item.get('observations', []))[-64:]
        history.write_text(json.dumps(known))
    ledger(output_root, tag, compact(record))
    return 0 if record['status'] == 'completed' else 1


def prepare_node(node, record):
    """This commit's mesh checkout on the node, built there (node prepare) for ranks that load it by path; the
    `checkouts_kept` most recently used kept."""
    from mesh_nodes import home_path
    from mesh_source import checkout_command
    cluster, commit = node.cluster, record['commits']['mesh']
    node.source = f'{cluster.checkouts}/mesh/{commit}'
    result = {'number': node.number, 'status': 'installing', 'commit': commit, 'commands': node.commands}
    record['nodes'][f'{node.number}@{node.addr}'] = result
    try:
        code, output = node.run('\n'.join((
            f'export MESH_RUN={shlex.quote(record["token"])}',
            checkout_command(record['upstreams']['mesh'], commit, '~/' + node.source),
            f'touch {home_path(node.source)}.used',
            f'PYTHONPATH={home_path(node.source + "/rdma")} {cluster.python} -m mesh_nodes node prepare',
            f'{cluster.python} {home_path(node.source + "/rdma/mesh_source.py")} prune '
            f'{home_path(cluster.checkouts + "/mesh")} {cluster.kept}')))
        result['install_exit'] = code
        if code:
            raise RuntimeError(f'preparation failed ({code}): {output}')
        result.update(status='installed', bridge=next((json.loads(line)['bridge'] for line in output.splitlines()
                                                        if line.startswith('{"bridge"')), None))
    except Exception as error:
        result.update(status='failed', error=str(error))
    return result


def rank_results(outputs):
    result = {}
    for name, output in outputs.items():
        lines = output['diagnostics']
        code = output['exit']
        values, observations = [], []
        for value in output['records']:
            if 'observations' in value:
                observations.extend(value['observations'])
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
        result[name] = {'exit': code, 'measurements': values, 'observations': observations,
                        'error': f'exit {code}: {detail}' if code else ''}

    return result



def run_workload(item, nodes, paths, record, ordinal, cluster):
    import mesh_nodes as mesh
    from mesh_statistics import aggregate
    tag = item['tag']
    end = min(time.time() + item.get('timeout', 300), record['deadline_unix'])
    selected = [node for node in (mesh.at(nodes, m.number) for m in cluster.place(item['place'])) if node]
    result = {'tag': tag, 'configuration': item, 'nodes': [node.number for node in selected], 'status': 'preparing',
              'metrics': []}
    record['workloads'].append(result)
    try:
        if not selected:
            raise RuntimeError(f'placement unmet: no live, installed member meets {item["place"]}')
        if len(mesh.parts([node.number for node in selected], mesh.paired(selected))) > 1:
            raise RuntimeError('placement unmet: its members are not linked by paired cables')
        history = record.get('history', {}).get(tag, [])[-64:]
        env = (f'PYTHONPATH={mesh.home_path(selected[0].source + "/rdma")}:{mesh.home_path(selected[0].source + "/rdma/torch-mesh")} '
               'PYTHONDONTWRITEBYTECODE=1 ')
        command = f'{cluster.python} {mesh.home_path(selected[0].source + "/examples/" + item["program"] + ".py")}'
        if item.get('configuration'):
            operands = json.loads((paths['metal-microbench'] / item['configuration']).read_text())
            operands.update({key: item[key] for key in ('steps', 'rows') if key in item})
            operands['observations'] = history
            command += ' -'
        else:
            operands = history
            env += 'MESH_OBSERVATIONS_STDIN=1 '
        result.update(status='running', ranks={f'rank{k}': {} for k in range(len(selected))})
        port = cluster.rank_port + zlib.crc32(record['run'].encode()) % 20 * 20 + ordinal
        outputs = mesh.ranks(selected, tag, '.', command, port, env=env,
                             alarm=max(1, int(end - time.time())), input=json.dumps(operands).encode(), states=result['ranks'],
                             run=record['token'])
        result['loaded'] = sorted({v['loaded'] for row in outputs.values() for v in row['records'] if 'loaded' in v})
        result['overlapped'] = sorted({t for row in outputs.values() for v in row['records'] for t in v.get('overlap', ())})
        result['ranks'] = rank_results(outputs)
        result['observations'] = [row for rank in result['ranks'].values() for row in rank.pop('observations')]
        if len(result['ranks']) != len(selected) or any(v['exit'] != 0 for v in result['ranks'].values()):
            failures = list(dict.fromkeys(value['error'] for value in result['ranks'].values() if value['exit']))
            raise RuntimeError('; '.join(failures) or 'missing rank completion')
        result['status'] = 'completed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = str(error)
        print(f'{tag}: {result["error"]}', file=sys.stderr, flush=True)
    # A later fault does not erase earlier completed observations. Only rows
    # present for every participating rank can form an aggregate interval.
    result['metrics'] = aggregate({k: v for k, v in result.get('ranks', {}).items() if 'measurements' in v})
    return result


def _execute_locked(record):
    import mesh_nodes as mesh
    record.update(status='running', stage='configuration')
    try:
        paths = {name: Path(value['path']) for name, value in record['checkouts'].items()}
        for name, root in paths.items():
            if verify(root) != record['commits'][name]:
                raise ValueError(f'{name}: checkout differs from the selected commit')
        cluster = mesh.load()
        items = workload_configuration(paths['metal-microbench'])
        record.update(stage='discovery', discovery={})
        found, record['absent'] = mesh.discover(cluster, state=record['discovery'], deadline=record['deadline_unix'])
        record['stage'] = 'installation'
        installs = mesh.parallel([lambda n=n: prepare_node(n, record)
                                  for n in found])
        record['nodes'] = {f'{node.number}@{node.addr}': result for node, result in zip(found, installs)}
        ready = [node for node, result in zip(found, installs)
                 if result['status'] == 'installed' and node.number not in record['absent']]
        record['stage'] = 'pairing'
        if ready and not mesh.wait_paired(ready, None, record['deadline_unix']):
            record['errors'].append({'stage': 'pairing', 'error': 'not all links paired; each workload needs its own members paired'})
        for ordinal, item in enumerate(items):
            record['stage'] = item['tag']
            run_workload(item, ready, paths, record, ordinal, cluster)
        record['status'] = 'completed' if (ready and not record['errors'] and
            all(value['status'] == 'completed' for value in record['workloads'])) else 'failed'
    except Exception as error:
        record['status'] = 'failed'
        record['errors'].append({'stage': record['stage'], 'error': str(error), 'traceback': traceback.format_exc()})
    record.pop('history', None)
    record['stage'] = 'finished'
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'execute'))
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args(argv)
    if args.action == 'execute':
        import mesh_nodes
        from mesh_statistics import serving

        def halt(signum, _):
            mesh_nodes.release()
            raise Stopped(f'the driver received {signal.Signals(signum).name}')
        for s in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, halt)
        record = json.load(sys.stdin)
        host, port = json.loads((Path(__file__).resolve().parents[1] / 'etc/statistics.json').read_text())['listen']
        for port in range(port, port + 10):  # another run's driver may serve the first
            with contextlib.suppress(OSError), serving(record, (host, port)):
                record['statistics'] = {'listen': [host, port], 'pid': os.getpid()}
                print(f'Live state: http://{host}:{port}/STATS', file=sys.stderr, flush=True)
                record = _execute_locked(record)
                break
        else:
            raise OSError(f'no statistics port free from {port - 9}')
        print(json.dumps(record, separators=(',', ':')))
        return 0 if record['status'] == 'completed' else 1
    return launcher(*args.paths)


if __name__ == '__main__':
    sys.exit(main())
