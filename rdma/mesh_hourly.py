"""Run committed mesh/microbench snapshots and retain failures without changing Git state.

The live launcher only pins/exports and records its own provenance. Execution is a
fresh interpreter importing this driver from the pinned mesh snapshot. A helper
missing in that commit is a recorded prerequisite failure, never a source fallback.
"""
import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import traceback

from mesh_source import MANIFEST, export, pin, verify

REQUIRED = ('rdma/mesh_hourly.py', 'rdma/mesh_nodes.py', 'rdma/mesh_source.py',
            'rdma/mesh_install.py', 'rdma/mesh_workloads.py', 'rdma/mesh-install-state.c')
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
                  item['steps'], item['rows'], tuple(item['nodes'])) for item in items if item['kind'] == 'example')
    scripts = tuple(Path(item['script']).name for item in items if item['kind'] == 'script')
    return runs, scripts, Path(microbench_root) / 'output_data/hourly/log.jsonl'


def measurement_values(record, tag):
    """Read both historical ledger rows and current structured workload results."""
    if tag in record:
        return record[tag]
    workload = next((item for item in record.get('workloads', []) if item['tag'] == tag), {})
    ranks = workload.get('ranks', {})
    figures = {}
    for row in ranks.get('rank0', {}).get('measurements', []):
        figures.update({key: [row[key], None] for key in FIGURES if key in row})
    return dict(figures, ranks={name: f'exit {value["exit"]}' for name, value in ranks.items()})


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def launcher(mesh_root, microbench_root, output_root):
    """The only working-tree code used is this explicitly identified export launcher."""
    output_root = Path(output_root).resolve()
    began = time.time()
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(began)) + f'-{time.time_ns() % 1000000000:09d}'
    out = output_root / tag
    out.mkdir(parents=True)
    record = {'schema': 2, 'run': tag, 'started_unix': began, 'commits': {}, 'snapshots': {},
              'status': 'failed', 'errors': [], 'nodes': {}, 'workloads': [],
              'launcher': {'file': str(Path(__file__).resolve()),
                           'role': 'pin/export only; target helpers execute from the committed snapshot'}}
    try:
        roots = {'mesh': Path(mesh_root).resolve(), 'metal-microbench': Path(microbench_root).resolve()}
        # Resolve both before exporting either. Later HEAD changes cannot change these inputs.
        record['commits'] = {name: pin(root) for name, root in roots.items()}
        (out / 'commits').write_text(''.join(f'{name} {sha}\n' for name, sha in record['commits'].items()))
        for name, root in roots.items():
            target = out / 'sources' / name
            export(root, record['commits'][name], target)
            record['snapshots'][name] = {'path': str(target)}
        committed_mesh = Path(record['snapshots']['mesh']['path'])
        committed_bench = Path(record['snapshots']['metal-microbench']['path'])
        absent = [name for name in REQUIRED if not (committed_mesh / name).is_file()]
        if absent:
            raise RuntimeError(f'pinned mesh commit lacks shared hourly prerequisites: {absent}')
        config = committed_bench / 'tools/mesh/hourly/workloads.json'
        record['configuration'] = {'path': str(config.relative_to(committed_bench))}
        save(out / 'request.json', record)
        with (out / 'hourly.log').open('wb') as log:
            child = subprocess.run([sys.executable, str(committed_mesh / 'rdma/mesh_hourly.py'),
                                    'execute', str(out / 'request.json')], cwd=committed_mesh,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if (out / 'result.json').exists():
            record = json.loads((out / 'result.json').read_text())
        record['driver_exit'] = child.returncode
        if child.returncode:
            record['status'] = 'failed'
            record['errors'].append({'stage': 'committed-driver', 'exit': child.returncode})
    except Exception as error:
        record['errors'].append({'stage': 'snapshot/export', 'error': str(error), 'traceback': traceback.format_exc()})
    record['finished_unix'] = time.time()
    save(out / 'result.json', record)
    with (output_root / 'log.jsonl').open('a') as log:
        log.write(json.dumps(record, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record['status'] == 'completed' else 1


def archive_snapshot(source):
    """Include only verified committed files and their manifest, never build leftovers."""
    source = Path(source)
    document = verify(source)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as tar:
        for name in [*document['files'], MANIFEST]:
            tar.add(source / name, arcname=name, recursive=False)
    return stream.getvalue()


def stage(node, source, destination):
    from mesh_nodes import home_path
    payload = archive_snapshot(source)
    # The directory is unique per run; no agent checkout or earlier build is overwritten.
    folder, bundle = home_path(destination), home_path(destination + '.tar')
    command = (f'mkdir -p {folder} && cat > {bundle} && '
               f'tar -xf {bundle} -C {folder} && rm {bundle}')
    code, output = node.run(command, input=payload)
    if code:
        raise RuntimeError(f'{node.name}: committed snapshot upload failed ({code}): {output}')
    return {'commit': verify(source)['commit'], 'remote_root': destination}


def prepare_node(node, paths, record, out, since):
    """Only successfully verified installs enter measurements; stale packages are never used."""
    result = {'number': node.number, 'status': 'failed', 'sources': {}}
    try:
        for name, source in paths.items():
            destination = f'.cache/mesh-hourly/{record["run"]}/{name}'
            result['sources'][name] = stage(node, source, destination)
            if name == 'mesh':
                node.source = destination
        code, _ = node.py('resets', str(int(since)), out=out / f'resets-{node.number}.log')
        result['resets_exit'] = code
        code, _ = node.py('install', record['commits']['mesh'], '1', out=out / f'install-{node.number}.log')
        result['install_exit'] = code
        if code:
            raise RuntimeError(f'install failed ({code}); node excluded')
        code, _ = node.py('installed', record['commits']['mesh'], '1', out=out / f'verify-{node.number}.log')
        result['verify_exit'] = code
        if code:
            raise RuntimeError(f'installed provenance verification failed ({code}); node excluded')
        result['status'] = 'installed'
    except Exception as error:
        result['error'] = str(error)
    return result


def rank_results(out, tag):
    result = {}
    for path in sorted(out.glob(f'{tag}-rank*.log')):
        lines = path.read_text(errors='replace').splitlines()
        code = next((int(line[5:]) for line in reversed(lines) if line.startswith('exit ')), None)
        values = []
        for line in lines:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                values.append(value)
        result[path.stem.rsplit('-', 1)[-1]] = {'exit': code, 'measurements': values,
                                               'last_lines': lines[-6:] if code else []}
    return result


def run_workload(item, nodes, paths, record, out, ordinal):
    import mesh_nodes as mesh
    tag = item['tag']
    selected = ([mesh.at(nodes, str(n)) for n in item['nodes']] if item.get('nodes')
                else nodes[:item.get('rank_limit', len(nodes))])
    result = {'tag': tag, 'configuration': item, 'status': 'failed'}
    try:
        if not selected or None in selected:
            raise RuntimeError('required nodes are absent or did not install the pinned source successfully')
        if not mesh.wait_paired(nodes, out / f'{tag}-pairing.log'):
            raise RuntimeError('the discovered fabric has not paired before workload configuration')
        # Workload configuration is committed; generated tensors live outside the source archives.
        folder = f'mesh-runs/hourly/{record["run"]}/{tag}'
        if item['kind'] == 'example':
            configuration = paths['metal-microbench'] / item['configuration']
            operands = json.loads(configuration.read_text())
            operands['controller_state'] = f'~/.cache/mesh/controllers/{tag}.json'
            mesh.send(selected, folder, {'configuration.json': json.dumps(operands).encode()})
            build = (f'cd {mesh.home_path(folder)} && PYTHONPATH={mesh.home_path(selected[0].source + "/rdma")} '
                     f'{mesh.PY} -m mesh_workloads build {mesh.home_path(selected[0].source + "/examples")} '
                     f'{shlex.quote(item["program"].removesuffix("-service"))} configuration.json '
                     f'{int(item["steps"])} {int(item["rows"])}')
            code, _ = selected[0].run(build, out=out / f'{tag}-build.log')
            result['build_exit'] = code
            if code:
                raise RuntimeError(f'input preparation failed ({code})')
            command = f'{mesh.PY} {mesh.home_path(selected[0].source + "/examples/" + item["program"] + ".py")} configuration.json'
        else:
            bench = f'.cache/mesh-hourly/{record["run"]}/metal-microbench'
            for node in selected:
                code, message = node.run(f'mkdir -p {mesh.home_path(folder)}')
                if code:
                    raise RuntimeError(message)
            command = f'{mesh.PY} {mesh.home_path(bench + "/" + item["script"])}'
            if item.get('configuration'):
                command += ' ' + mesh.home_path(bench + '/' + item['configuration'])
        mesh.ranks(selected, out, tag, folder, command, 29600 + ordinal,
                   env=f'PYTHONPATH={mesh.home_path(selected[0].source + "/rdma")} ', alarm=item.get('timeout', 300))
        result['ranks'] = rank_results(out, tag)
        if len(result['ranks']) != len(selected) or any(v['exit'] != 0 for v in result['ranks'].values()):
            raise RuntimeError('one or more rank processes failed or lack a completion record')
        result['status'] = 'completed'
    except Exception as error:
        result['error'] = str(error)
        result['ranks'] = rank_results(out, tag)
    return result


def _execute_locked(request):
    import mesh_nodes as mesh
    request = Path(request)
    out = request.parent
    record = json.loads(request.read_text())
    try:
        paths = {name: Path(value['path']) for name, value in record['snapshots'].items()}
        for name, root in paths.items():
            if verify(root)['commit'] != record['commits'][name]:
                raise ValueError(f'{name}: commit does not match manifest')
        mesh.configure(paths['mesh'], paths['metal-microbench'])
        items = workload_configuration(paths['metal-microbench'])
        # Discovery observes the currently running bridge, explicitly before installation.
        # It is not evidence that the tested revision has been installed.
        def bootstrap(candidate):
            destination = f'.cache/mesh-hourly/{record["run"]}/mesh'
            stage(candidate, paths['mesh'], destination)
            candidate.source = destination

        found, candidates = mesh.discover(out / 'discovery.log', prepare=bootstrap)
        if not found:
            found = [mesh.Node('-', address) for address in candidates]
        installs = mesh.parallel([lambda n=n: prepare_node(n, paths, record, out, record['started_unix'] - 3600)
                                  for n in found])
        record['nodes'] = {f'{node.number}@{node.addr}': result for node, result in zip(found, installs)}
        if any(node.number == '-' for node in found):
            # Read the newly installed topology; staged helpers stay at the same per-run path.
            found, _ = mesh.discover(out / 'discovery.log', prepare=bootstrap)
            found = found or []
            installs = mesh.parallel([lambda n=n: prepare_node(n, paths, record, out, record['started_unix'] - 3600)
                                      for n in found])
            record['nodes'].update({f'{node.number}@{node.addr}': result for node, result in zip(found, installs)})
        ready = [node for node, result in zip(found, installs) if result['status'] == 'installed']
        if ready and not mesh.wait_paired(ready, out / 'pairing.log'):
            record['errors'].append({'stage': 'pairing', 'error': 'not all current topology links paired'})
            ready = []
        for ordinal, item in enumerate(items):
            record['workloads'].append(run_workload(item, ready, paths, record, out, ordinal))
            save(out / 'result.json', record)
        record['status'] = 'completed' if (ready and not record['errors'] and
            all(value['status'] == 'completed' for value in record['workloads'])) else 'failed'
    except Exception as error:
        record['errors'].append({'stage': 'execute', 'error': str(error), 'traceback': traceback.format_exc()})
    finally:
        save(out / 'result.json', record)
    return 0 if record['status'] == 'completed' else 1


def execute(request):
    import mesh_nodes as mesh
    request = Path(request)
    try:
        with mesh.exclusive_lock('hourly ' + request.parent.name):
            return _execute_locked(request)
    except Exception as error:
        record = json.loads(request.read_text())
        record['errors'].append({'stage': 'cluster-lease', 'error': str(error)})
        record['status'] = 'failed'
        save(request.parent / 'result.json', record)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'execute'))
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args(argv)
    if args.action == 'execute':
        return execute(*args.paths)
    return launcher(*args.paths)


if __name__ == '__main__':
    sys.exit(main())
