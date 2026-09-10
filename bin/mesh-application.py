#!/usr/bin/env mesh-python
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
ROOT = Path(__file__).resolve().parents[1]


# ../design/algorithm-sources.md#configuration-storage-layout
def command(values, host=None, **kwargs):
    values = list(map(str, values))
    return subprocess.run(SSH + [host, shlex.join(values)] if host else values, check=True, **kwargs)


# ../design/algorithm-sources.md#configuration-storage-layout
def checkout(root, host=None):
    dirty = command(['git', '-C', root, 'status', '--porcelain'], host, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f'{host or "local"}:{root}: commit source changes before deployment')
    command(['git', '-C', root, 'checkout', 'main'], host, stdout=sys.stderr)
    return command(['git', '-C', root, 'rev-parse', 'HEAD'], host, capture_output=True, text=True).stdout.strip()


# ../design/algorithm-sources.md#configuration-storage-layout
def activate(root, target):
    root = Path(root).resolve()
    revision = checkout(root)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    link = target / '.current'
    link.unlink(missing_ok=True)
    link.symlink_to(root)
    os.replace(link, target / 'current')
    print(json.dumps({'event': 'application_activated', 'root': str(root), 'id': revision}), file=sys.stderr)


# ../design/algorithm-sources.md#configuration-storage-layout
def deploy(args):
    source = Path(args.source).resolve()
    revision = checkout(source)
    destination = Path(args.checkout) if args.host else source
    if args.host:
        checkout(destination, args.host)
        command(['git', '-C', source, 'push', f'{args.host}:{destination}', 'main'],
            env={**os.environ, 'GIT_SSH_COMMAND': shlex.join(SSH)})
        remote = checkout(destination, args.host)
        if remote != revision:
            raise RuntimeError(f'git revision mismatch: local={revision}, remote={remote}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds = [pool.submit(command, ['make', '-C', source / 'rdma', 'all'], stdout=sys.stderr)]
        if args.host:
            builds.append(pool.submit(command, ['make', '-C', destination / 'rdma', 'all'], args.host, stdout=sys.stderr))
        for build in builds: build.result()
    if args.host:
        command([destination / 'bin/mesh-python', destination / 'bin/mesh-application.py', 'activate',
            '--source', destination, '--target', args.target], args.host)
    else:
        activate(source, args.target)


# ../design/algorithm-sources.md#configuration-storage-layout
def run(args):
    root = (Path(args.target) / 'current').resolve()
    revision = checkout(root)
    provenance = {'id': revision, 'root': str(root)}
    for label, relative in [('native_client', 'rdma/libmesh.dylib'), ('wire', 'rdma/xonwire.def')]:
        provenance[label] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    paths = [str(root / suffix) for suffix in ('xonotic', 'rdma', 'xonotic/payload/tools')]
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(paths),
        'MESH_APPLICATION_ROOT': str(root), 'MESH_APPLICATION_IDENTITY': json.dumps(provenance)}
    values = [args.python or str(root / 'bin/mesh-python'), '-B', '-m', *args.arguments]
    print(json.dumps({'event': 'application_launch', **provenance, 'module': args.arguments[0]}), file=sys.stderr)
    if args.log:
        log = Path(args.log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as output:
            process = subprocess.Popen(values, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'event': 'application_started', 'pid': process.pid, 'log': str(log)}), file=sys.stderr)
    else:
        os.execvpe(values[0], values, environment)


# ../design/algorithm-sources.md#configuration-storage-layout
def main():
    global SSH
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('deploy', 'activate', 'launch', 'run'))
    parser.add_argument('--source', default=str(ROOT))
    parser.add_argument('--checkout', default='/Users/mdot/mesh')
    parser.add_argument('--target', required=True)
    parser.add_argument('--host')
    parser.add_argument('--ssh-command', default=shlex.join(SSH))
    parser.add_argument('--python')
    parser.add_argument('--log')
    values = sys.argv[1:]
    boundary = values.index('--') if '--' in values else len(values)
    args = parser.parse_args(values[:boundary])
    SSH = shlex.split(args.ssh_command)
    args.arguments = values[boundary + 1:]
    if args.operation == 'activate': activate(args.source, args.target)
    elif args.operation == 'run': run(args)
    else:
        deploy(args)
        if args.operation == 'launch':
            root = Path(args.checkout if args.host else args.source)
            values = [root / 'bin/mesh-python', root / 'bin/mesh-application.py', 'run', '--target', args.target]
            if args.python: values += ['--python', args.python]
            if args.log: values += ['--log', args.log]
            values = list(map(str, values + ['--', *args.arguments]))
            values = SSH + [args.host, shlex.join(values)] if args.host else values
            os.execvp(values[0], values)


if __name__ == '__main__':
    main()
