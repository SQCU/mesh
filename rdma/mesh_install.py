"""Explicit source-root transition for committed snapshot installation.

Normal mesh-user.sh installs retain their existing behavior. The hourly driver calls
this entrypoint under the cluster lease; a node-local flock serializes installers.
No client is killed. A failed transition excludes that node from measurement.
"""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import time

from mesh_source import verify


def run(*args, check=True):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}): {result.stderr.strip()} {result.stdout.strip()}')
    return result


def wait_for(read, accept, seconds, description):
    end = time.monotonic() + seconds
    while True:
        value = read()
        if accept(value):
            return value
        if time.monotonic() >= end:
            raise RuntimeError(f'{description}: {value}')
        time.sleep(0.25)


def disabled(domain):
    result = run('launchctl', 'print-disabled', domain)
    return bool(re.search(r'"io\.mesh\.bridge"\s*=>\s*(true|disabled)', result.stdout))


def active_program(service):
    result = run('launchctl', 'print', service, check=False)
    if result.returncode:
        return None
    match = re.search(r'^\s*program = (.*)$', result.stdout, re.M)
    if not match:
        raise RuntimeError('loaded bridge has no observable executable path')
    return match.group(1).strip().strip('"')


def loaded_program_arguments(service):
    """Read the running job's argv; its on-disk plist may name a different source."""
    text = run('launchctl', 'print', service).stdout
    program = re.search(r'^\s*program = (.*)$', text, re.M)
    arguments = re.search(r'^\s*arguments = \{\s*\n(.*?)^\s*\}', text, re.M | re.S)
    if not program or not arguments:
        raise RuntimeError('cannot observe loaded bridge ProgramArguments for rollback')
    argv = [line.strip().strip('"') for line in arguments.group(1).splitlines() if line.strip()]
    executable = program.group(1).strip().strip('"')
    if len(argv) != 3 or argv[0] != executable or Path(argv[0]).name != 'mesh-heartbeat' or Path(argv[1]).name != 'mesh-flow':
        raise RuntimeError(f'unrecognized loaded bridge arguments: {argv}')
    return argv


def verify_active(source, commit):
    source = Path(source).resolve()
    service = f'user/{os.getuid()}/io.mesh.bridge'
    expected = source / 'rdma/mesh-heartbeat'
    if active_program(service) != str(expected):
        raise RuntimeError('loaded heartbeat executable is not the requested committed snapshot')
    job = run('launchctl', 'print', service).stdout
    heartbeat = re.search(r'^\s*pid = ([0-9]+)$', job, re.M)
    if heartbeat is None:
        raise RuntimeError('loaded heartbeat has no running pid')
    # The bridge forks a mesh-flow child per physical port. Its direct parent
    # is the heartbeat; those link workers are children of the bridge itself.
    pids = run('pgrep', '-x', '-P', heartbeat.group(1), 'mesh-flow').stdout.split()
    if len(pids) != 1:
        raise RuntimeError(f'expected exactly one bridge process, found {pids}')
    actual = run('ps', '-o', 'comm=', '-p', pids[0]).stdout.strip()
    if actual != str(source / 'rdma/mesh-flow'):
        raise RuntimeError(f'active bridge runs {actual}, not {source}')
    check = json.loads(run(source / 'rdma/mesh-stat').stdout.splitlines()[-1])
    if not check.get('up'):
        raise RuntimeError(f'new bridge is not up: {check}')
    return {'commit': commit, 'active_heartbeat': str(expected), 'active_bridge': actual,
            'topology': check}


def install_snapshot(source, allow_source_transition=False):
    source = Path(source).resolve()
    document = verify(source)
    cache = Path.home() / '.cache/mesh'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'install.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _install_locked(source, document['commit'], allow_source_transition)


def _install_locked(source, commit, allow_source_transition):
    domain, service = f'user/{os.getuid()}', f'user/{os.getuid()}/io.mesh.bridge'
    if disabled(domain):
        raise RuntimeError('operator-disabled bridge retained; snapshot not installed')
    before = active_program(service)
    target = str(source / 'rdma/mesh-heartbeat')
    changing = before is not None and before != target
    if changing and not allow_source_transition:
        raise RuntimeError('active source-root transition requires its explicit configuration operand')
    plist = Path.home() / 'Library/LaunchAgents/io.mesh.bridge.plist'
    old = plist.read_bytes() if plist.exists() else None
    rollback = None
    if changing:
        rollback = plistlib.loads(old) if old is not None else {'Label': 'io.mesh.bridge', 'RunAtLoad': True}
        rollback['ProgramArguments'] = loaded_program_arguments(service)
        if 'Program' in rollback:
            rollback['Program'] = rollback['ProgramArguments'][0]
    config = Path((Path.home() / 'Library/Application Support/io.mesh.bridge/document').read_text().strip())
    observer = source / 'rdma/.build/mesh-install-state'
    observer.parent.mkdir(exist_ok=True)
    run('cc', '-O2', '-Wall', '-Wextra', '-Werror', '-o', observer, source / 'rdma/mesh-install-state.c')
    read_state = lambda: json.loads(run(observer).stdout)
    initial = read_state()
    if initial['clients']:
        raise RuntimeError(f'live client holders prevent install: {initial["clients"]}')
    released = unloaded = False
    try:
        if changing:
            # Heartbeat owns expiration and unlink. Do not unlink a live region ourselves.
            run('launchctl', 'kill', 'SIGUSR1', service)
            released = True
            wait_for(read_state, lambda v: not v['present'], 60, 'old region still present')
            run('launchctl', 'bootout', service)
            unloaded = True
        output = run(source / 'bin/mesh-user.sh', config, source)
        print(output.stdout, end='', flush=True)
        wait_for(lambda: json.loads(run(source / 'rdma/mesh-stat').stdout.splitlines()[-1]),
                 lambda v: v.get('up') is True, 60, 'installed bridge did not start')
        # A module loaded from this immutable source must verify the installed package too.
        check_package(source)
        result = verify_active(source, commit)
        (source / '.mesh-install.json').write_text(json.dumps(result, sort_keys=True) + '\n')
        active = Path.home() / '.cache/mesh/active-source'
        active.parent.mkdir(parents=True, exist_ok=True)
        active.write_text(str(source) + '\n')
        print(json.dumps({'installed': result}), flush=True)
        return result
    except BaseException:
        if unloaded:
            # Roll back the old executable roots, which were never overwritten by this snapshot.
            with contextlib.suppress(Exception):
                if active_program(service):
                    run('launchctl', 'kill', 'SIGUSR1', service)
                    wait_for(read_state, lambda v: not v['present'], 60, 'new region held during rollback')
                    run('launchctl', 'bootout', service)
                if rollback is not None:
                    plist.write_bytes(plistlib.dumps(rollback))
                    try:
                        if not disabled(domain):
                            run('launchctl', 'bootstrap', domain, plist)
                    finally:
                        if old is not None:
                            plist.write_bytes(old)
        elif released:
            with contextlib.suppress(Exception):
                run('launchctl', 'kill', 'SIGHUP', service)
        raise


def check_package(source):
    import glob
    package = Path(next(iter(glob.glob(str(Path.home() / '.venv-mesh-uv/lib/python*/site-packages/torch_mesh'))), '-'))
    wanted = verify(source)['files']
    prefix = 'rdma/torch-mesh/torch_mesh/'
    for name, record in wanted.items():
        if name.startswith(prefix) and name.endswith(('.py', '.metal')):
            installed = package / name[len(prefix):]
            if not installed.is_file() or installed.read_bytes() != (Path(source) / name).read_bytes():
                raise RuntimeError(f'installed torch_mesh differs from committed source: {name}')
    if not list(package.glob('_stream*.so')):
        raise RuntimeError('installed torch_mesh has no native extension')
