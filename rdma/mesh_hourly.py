"""Run committed mesh/microbench Git checkouts fetched from each repository's origin."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from mesh_nodes import fetch
from mesh_source import checkout, revision, upstream, verify

REQUIRED = ('rdma/mesh_hourly.py', 'rdma/mesh_nodes.py', 'rdma/mesh_source.py',
            'rdma/mesh_install.py', 'rdma/mesh-install-state.c')
# Observations belong to the application lifetime, never the measurement ledger.
HISTORY = {}

FIGURES = ('steady_step_microseconds', 'step_microseconds', 'crossing_microseconds',
           'steady_tick_microseconds', 'tick_microseconds', 'aggregate_rows_per_second',
           'steady_aggregate_rows_per_second', 'per_stream_steps_per_second')


def workload_configuration(microbench_root):
    """One canonical configuration for hourly and interactive comparison entrypoints."""
    root = Path(microbench_root)
    return json.loads((root / 'tools/mesh/hourly/workloads.json').read_text())['workloads']


def comparison_configuration(microbench_root):
    items = workload_configuration(microbench_root)
    runs = tuple((item['tag'], item['program'], item.get('reference', 'no record yet'),
                  item['steps'], item['rows'], tuple(item['nodes'])) for item in items if 'steps' in item and 'nodes' in item)
    scripts = tuple(item['program'] + '.py' for item in items if 'steps' not in item or 'nodes' not in item)
    return runs, scripts, Path(microbench_root) / 'output_data/hourly/log.jsonl'


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
    rows, errors = {}, []
    for item in record['workloads']:
        rows[item['tag']] = [
            [rounded(m.get('function_seconds', m.get('amortized_seconds_per_call')) * 1e6),
             rounded(m['useful_gigabytes_per_second']),
             *([round(m['aggregate_rows_per_second'])] if 'aggregate_rows_per_second' in m else [])]
            for m in item['metrics']]
        if item['status'] != 'completed':
            failures = [[item['tag'] + '/' + rank, value['error'][:160]]
                        for rank, value in item.get('ranks', {}).items() if value.get('error')]
            errors.extend(failures or [[item['tag'], item.get('error', 'incomplete')[:160]]])
    errors.extend([e.get('stage', 'driver'), (e.get('error') or e.get('stderr') or f'exit {e.get("exit", "unknown")}')[-160:]]
                  for e in record['errors'])
    errors.extend([name, node.get('error', 'installation failed')[-160:]]
                  for name, node in record['nodes'].items() if node['status'] != 'installed')
    result = {'schema': 3, 'commits': record['commits'], 'unix': int(record['started_unix']),
              'nodes': [r['number'] for r in record['nodes'].values() if r['status'] == 'installed'],
              'us,GB/s': rows}
    if record['status'] != 'completed':
        result['errors'] = errors or [['run', 'incomplete']]
    return result


def launcher(mesh_root, microbench_root, output_root):
    from mesh_nodes import exclusive_lock
    with exclusive_lock('committed hourly'):
        return _launch(mesh_root, microbench_root, output_root)


def _launch(mesh_root, microbench_root, output_root):
    """Select main commits and launch the driver from its upstream Git checkout."""
    output_root = Path(output_root).resolve()
    began = time.time()
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(began)) + f'-{time.time_ns() % 1000000000:09d}'
    out = output_root / tag
    out.mkdir(parents=True)
    record = {'schema': 2, 'run': tag, 'started_unix': began, 'commits': {}, 'checkouts': {},
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
            target = Path.home() / '.cache/mesh-checkouts' / name / record['commits'][name]
            checkout(record['upstreams'][name], record['commits'][name], target)
            record['checkouts'][name] = {'path': str(target)}
        committed_mesh = Path(record['checkouts']['mesh']['path'])
        committed_bench = Path(record['checkouts']['metal-microbench']['path'])
        absent = [name for name in REQUIRED if not (committed_mesh / name).is_file()]
        if absent:
            raise RuntimeError(f'pinned mesh commit lacks shared hourly prerequisites: {absent}')
        config = committed_bench / 'tools/mesh/hourly/workloads.json'
        record['configuration'] = {'path': str(config.relative_to(committed_bench))}
        record['history'] = HISTORY
        subprocess.run(['make', '-s', '-C', str(committed_mesh / 'rdma'), 'statistics'],
                       check=True, stdout=subprocess.DEVNULL)
        child = subprocess.run([sys.executable, str(committed_mesh / 'rdma/mesh_hourly.py'), 'execute'],
                               cwd=committed_mesh, input=json.dumps(record), stdout=subprocess.PIPE, text=True)
        if child.stdout:
            record = json.loads(child.stdout)
        record.pop('history', None)
        record['driver_exit'] = child.returncode
        if child.returncode:
            record['status'] = 'failed'
            if not record['errors'] and not any(item['status'] != 'completed' for item in record['workloads']):
                record['errors'].append({'stage': 'committed-driver', 'exit': child.returncode})
    except Exception as error:
        record['errors'].append({'stage': 'git-checkout', 'error': str(error), 'traceback': traceback.format_exc()})
    record['finished_unix'] = time.time()
    for item in record['workloads']:
        HISTORY[item['tag']] = (HISTORY.get(item['tag'], []) + item.get('observations', []))[-64:]
    (out / 'result.json').write_text(json.dumps(compact(record), separators=(',', ':')) + '\n')
    with (output_root / 'log.jsonl').open('a') as log:
        log.write(json.dumps({'result': str(out / 'result.json')}, separators=(',', ':')) + '\n')
    return 0 if record['status'] == 'completed' else 1


def prepare_node(node, paths, record, out, since):
    """Only successfully verified installs enter measurements; stale packages are never used."""
    result = {'number': node.number, 'status': 'installing', 'sources': {}, 'commands': node.commands}
    record['nodes'][f'{node.number}@{node.addr}'] = result
    try:
        for name in paths:
            destination = f'.cache/mesh-checkouts/{name}/{record["commits"][name]}'
            result['sources'][name] = fetch(node, record['upstreams'][name], record['commits'][name], destination)
            if name == 'mesh':
                node.source = destination
        code, resets = node.py('resets', str(int(since)))
        result['resets_exit'] = code
        result['resets'] = resets
        code, _ = node.py('installed', record['commits']['mesh'], '1')
        if code:
            code, output = node.py('install', record['commits']['mesh'], '1')
            result['installation'] = output
        result['install_exit'] = code
        if code:
            raise RuntimeError(f'install failed ({code}): {output.splitlines()[-1] if output else "no diagnostic"}')
        code, _ = node.py('installed', record['commits']['mesh'], '1')
        result['verify_exit'] = code
        if code:
            raise RuntimeError(f'installed provenance verification failed ({code}); node excluded')
        result['status'] = 'installed'
        from mesh_nodes import PY, home_path
        for name in paths:
            code, output = node.run(f'{PY} {home_path(node.source + "/rdma/mesh_source.py")} prune '
                                    f'{home_path(".cache/mesh-checkouts/" + name)} {record["commits"][name]}')
            if code:
                raise RuntimeError(output)
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = str(error)
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



def run_workload(item, nodes, paths, record, out, ordinal):
    import mesh_nodes as mesh
    from mesh_statistics import aggregate
    tag = item['tag']
    selected = ([mesh.at(nodes, str(n)) for n in item['nodes']] if item.get('nodes')
                else nodes[:item.get('rank_limit', len(nodes))])
    result = {'tag': tag, 'configuration': item, 'status': 'preparing', 'metrics': []}
    record['workloads'].append(result)
    try:
        if not selected or None in selected:
            raise RuntimeError('required nodes are absent or did not install the pinned source successfully')
        if not mesh.wait_paired(nodes, None):
            raise RuntimeError('the discovered fabric has not paired before workload configuration')
        history = record.get('history', {}).get(tag, [])[-64:]
        env = f'PYTHONPATH={mesh.home_path(selected[0].source + "/rdma")} PYTHONDONTWRITEBYTECODE=1 '
        command = f'{mesh.PY} {mesh.home_path(selected[0].source + "/examples/" + item["program"] + ".py")}'
        if item.get('configuration'):
            operands = json.loads((paths['metal-microbench'] / item['configuration']).read_text())
            operands.update({key: item[key] for key in ('steps', 'rows') if key in item})
            operands['observations'] = history
            command += ' -'
        else:
            operands = history
            env += 'MESH_OBSERVATIONS_STDIN=1 '
        result.update(status='running', ranks={f'rank{k}': {} for k in range(len(selected))})
        outputs = mesh.ranks(selected, None, tag, '.', command, 29600 + ordinal,
                             env=env, alarm=item.get('timeout', 300), input=json.dumps(operands).encode(), states=result['ranks'])
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
    result['metrics'] = aggregate(result.get('ranks', {}))
    return result


def _execute_locked(record):
    import mesh_nodes as mesh
    out = None
    record['status'] = 'running'
    try:
        paths = {name: Path(value['path']) for name, value in record['checkouts'].items()}
        for name, root in paths.items():
            if verify(root) != record['commits'][name]:
                raise ValueError(f'{name}: checkout differs from the selected commit')
        mesh.configure(paths['mesh'], paths['metal-microbench'])
        items = workload_configuration(paths['metal-microbench'])
        record.update(stage='discovery', discovery={})
        # Discovery observes the currently running bridge, explicitly before installation.
        # It is not evidence that the tested revision has been installed.
        def bootstrap(candidate):
            record['discovery'][candidate.addr] = candidate.commands
            destination = f'.cache/mesh-checkouts/mesh/{record["commits"]["mesh"]}'
            fetch(candidate, record['upstreams']['mesh'], record['commits']['mesh'], destination)
            candidate.source = destination

        found, candidates = mesh.discover(None, prepare=bootstrap)
        if not found:
            found = [mesh.Node('-', address) for address in candidates]
        record['stage'] = 'installation'
        installs = mesh.parallel([lambda n=n: prepare_node(n, paths, record, out, record['started_unix'] - 3600)
                                  for n in found])
        record['nodes'] = {f'{node.number}@{node.addr}': result for node, result in zip(found, installs)}
        if any(node.number == '-' for node in found):
            # Read the newly installed topology using the selected checkout's helpers.
            found, _ = mesh.discover(None, prepare=bootstrap)
            found = found or []
            installs = mesh.parallel([lambda n=n: prepare_node(n, paths, record, out, record['started_unix'] - 3600)
                                      for n in found])
            record['nodes'].update({f'{node.number}@{node.addr}': result for node, result in zip(found, installs)})
        ready = [node for node, result in zip(found, installs) if result['status'] == 'installed']
        required = {str(number) for item in items for number in item.get('nodes', ())}
        missing = required - {node.number for node in ready}
        if missing:
            raise RuntimeError(f'configured cluster is incomplete; no workloads submitted: missing {sorted(missing)}')
        if ready and not mesh.wait_paired(ready, None):
            record['errors'].append({'stage': 'pairing', 'error': 'not all current topology links paired'})
            ready = []
        for ordinal, item in enumerate(items):
            record['stage'] = item['tag']
            result = run_workload(item, ready, paths, record, out, ordinal)
            if result['status'] != 'completed':
                raise RuntimeError(f'{item["tag"]} did not complete; no further workloads submitted')
        record['status'] = 'completed' if (ready and not record['errors'] and
            all(value['status'] == 'completed' for value in record['workloads'])) else 'failed'
    except Exception as error:
        record['status'] = 'failed'
        record['errors'].append({'stage': 'execute', 'error': str(error), 'traceback': traceback.format_exc()})
    record.pop('history', None)
    record['stage'] = 'finished'
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'execute'))
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args(argv)
    if args.action == 'execute':
        from mesh_statistics import serving
        record = json.load(sys.stdin)
        address = json.loads((Path(__file__).resolve().parents[1] / 'etc/statistics.json').read_text())['listen']
        record['statistics'] = {'listen': address, 'pid': os.getpid()}
        with serving(record, address):
            print(f'Live state: http://{address[0]}:{address[1]}/STATS', file=sys.stderr, flush=True)
            record = _execute_locked(record)
        print(json.dumps(record, separators=(',', ':')))
        return 0 if record['status'] == 'completed' else 1
    return launcher(*args.paths)


if __name__ == '__main__':
    sys.exit(main())
