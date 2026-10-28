"""Shared node discovery, deployment and rank execution; runners supply operands only."""
import concurrent.futures
import contextlib
import codecs
import dataclasses
import fcntl
import functools
import getpass
import glob
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT, HOME = Path(__file__).resolve().parents[1], Path.home()
SSHO = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=4', '-o', 'ControlPath=none']
PY = '~/.venv-mesh-uv/bin/python'


@dataclasses.dataclass(frozen=True)
class Member:
    number: str
    name: str
    management: str
    fabric: str
    roles: frozenset = frozenset()
    memory_gib: float = 0
    guard_free_pages: int = 0


@dataclasses.dataclass(frozen=True)
class Cluster:
    """etc/mesh-nodes.json with the environment: members by bridge node number, the administrator's user and key, the
    lease file and a run's bound in seconds."""
    members: dict
    user: str
    key: str | None
    lock: Path
    run_seconds: int

    @property
    def ssh(self):
        return [*SSHO, *(['-i', self.key] if self.key else [])]

    def place(self, need=None):
        """The members holding `need`'s role with at least its other values, in rank order: the root first, the rest
        by bridge node number (metal-microbench tools/mesh/grid.py order)."""
        need = dict(need or {})
        role = need.pop('role', None)
        chosen = [m for m in self.members.values()
                  if (role is None or role in m.roles) and all(getattr(m, k) >= v for k, v in need.items())]
        return sorted(chosen, key=lambda m: ('root' not in m.roles, int(m.number)))


@functools.cache
def load(path=ROOT / 'etc/mesh-nodes.json'):
    raw = json.loads(Path(path).read_text())
    access = raw.get('access', {})
    members = {str(row['node']): Member(str(row['node']), row['name'], row['management'], row['fabric'],
                                        frozenset(row.get('roles', ())), row.get('memory_gib', 0),
                                        row.get('guard_free_pages', 0)) for row in raw['nodes'].values()}
    return Cluster(members, access.get('user') or getpass.getuser(),
                   str(Path(access['key']).expanduser()) if 'key' in access else None,
                   Path(raw.get('lock', '~/.cache/mesh/nodes.lock')).expanduser(), raw['run_seconds'])


def home_path(relative):
    return '"$HOME"/' + shlex.quote(str(relative).removeprefix('~/'))



def sh(argv, input=None, out=None, append=False, timeout=None, state=None):
    state = {} if state is None else state
    state.clear()
    state.update(command=argv, status='starting', started_unix=time.time(), output=[], records=[], diagnostics=[])
    try:
        p = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as error:
        state.update(status='failed', finished_unix=time.time())
        state['diagnostics'].append(str(error))
        raise
    state.update(pid=p.pid, status='running')

    def read():
        decode = codecs.getincrementaldecoder('utf-8')('replace')
        pending = ''

        def line_record(line):
            try:
                value = json.loads(line)
            except ValueError:
                if line.strip():
                    state['diagnostics'].append(line)
            else:
                if isinstance(value, dict):
                    state['records'].append(value)

        while raw := p.stdout.read1(io.DEFAULT_BUFFER_SIZE):
            text = decode.decode(raw)
            state['output'].append(text)
            state['updated_unix'] = time.time()
            lines = (pending + text).split('\n')
            pending = lines.pop()
            for line in lines:
                line_record(line)
        tail = decode.decode(b'', final=True)
        state['output'].append(tail)
        line_record(pending + tail)
        p.stdout.close()

    def write():
        try:
            p.stdin.write(input)
            p.stdin.close()
        except BrokenPipeError:
            pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    writer = threading.Thread(target=write, daemon=True) if input is not None else None
    if writer:
        writer.start()
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        state['status'] = 'timed-out'
    else:
        state['status'] = 'exited'
    state.update(exit=p.returncode, finished_unix=time.time())
    reader.join()
    if writer:
        writer.join()
    output = ''.join(state['output'])
    if state['status'] == 'timed-out':
        raise subprocess.TimeoutExpired(argv, timeout, output=output.encode())
    if out is not None:
        with open(out, 'a' if append else 'w') as f:
            f.write(output)
        return p.returncode, ''
    return p.returncode, output


def loads(text):
    with contextlib.suppress(ValueError):
        return json.loads(text)
    return {}


def note(path, *lines):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(''.join(f'{line}\n' for line in lines))


def parallel(jobs):
    with concurrent.futures.ThreadPoolExecutor(max(1, len(jobs))) as pool:
        return [f.result() for f in [pool.submit(job) for job in jobs]]


class Node:
    def __init__(self, cluster, number, addr, via=None, ula='', *, local=False):
        self.cluster, self.number, self.addr, self.via, self.ula = cluster, number, addr, via, ula
        self.local = local
        member = cluster.members.get(number)
        self.name, self.guard = (member.name, member.guard_free_pages) if member else (number, 0)
        self.source = 'mesh'
        self.discovered_links = 0
        self.commands = []

    def argv(self, command):
        """The shared administration launcher, also used for resident-rank configuration."""
        if self.local:
            return ['/bin/sh', '-c', command]
        ssh, user = self.cluster.ssh, self.cluster.user
        proxy = ['-o', f"ProxyCommand=ssh {' '.join(ssh)} -W '[%h]:%p' {user}@{self.via}"] if self.via else []
        return ['ssh', *ssh, *proxy, f'{user}@{self.addr}', command]

    def run(self, command, state=None, **kw):
        state = {} if state is None else state
        self.commands.append(state)
        return sh(self.argv(command), state=state, **kw)

    def py(self, *args, **kw):
        return self.run(f'PYTHONPATH={home_path(self.source + "/rdma")} {PY} -m mesh_nodes node '
                        + ' '.join(shlex.quote(str(a)) for a in args), **kw)

    def stat(self):
        line = self.run(home_path(self.source + '/bin/mesh-stat'))[1].strip().split('\n')[-1]
        return line, loads(line)


def discover(cluster, log=None, state=None):
    """Read the local bridge's existing cluster map and bind configured endpoints."""
    local = Node(cluster, '-', 'localhost', local=True)
    if state is not None:
        state['localhost'] = local.commands
    report = local.stat()[1]
    if not report.get('up'):
        raise RuntimeError(f'local bridge topology is unavailable: {report}')
    expected = {int(number) for number in cluster.members}
    found = set(report['map']['nodes'])
    if found != expected:
        raise RuntimeError(f'live topology differs from configured nodes: missing {expected - found}, unconfigured {found - expected}')
    nodes = [Node(cluster, m.number, 'localhost' if int(m.number) == report['node'] else m.management,
                  ula=m.fabric, local=int(m.number) == report['node']) for m in cluster.members.values()]
    for node in nodes:
        node.discovered_links = sum(int(node.number) in link[:2] for link in report['map']['links'])
    note(log, 'map ' + ' '.join(node.number for node in nodes))
    return nodes, [node.addr for node in nodes]


def up(nodes, log=None):
    def one(n):
        if n.stat()[1].get('up') is True:
            return n
        note(log, f'not up {n.number}')
    return [n for n in parallel([lambda n=n: one(n) for n in nodes]) if n]


def at(nodes, number):
    return next((n for n in nodes if n.number == number), None)


def origin_main():
    subprocess.run(['git', '-C', ROOT, 'fetch', '-q', 'origin'])
    return resolve('origin/main')


def resolve(ref, repo=ROOT):
    return subprocess.run(['git', '-C', repo, 'rev-parse', f'{ref}^{{commit}}'], check=True, capture_output=True,
                          text=True).stdout.strip()


def fetch(node, origin, commit, destination):
    from mesh_source import checkout_command
    code, output = node.run(checkout_command(origin, commit, '~/' + destination))
    if code:
        raise RuntimeError(f'{node.name}: Git checkout failed ({code}): {output}')
    return {'commit': commit, 'remote_root': destination, 'upstream': origin}


def install_one(n, sha, log):
    from mesh_source import upstream
    n.source = '.cache/mesh-checkouts/mesh/current'
    fetch(n, upstream(ROOT), sha, n.source)
    code = n.py('install', sha, '1', out=log)[0]
    note(log, f'exit {code}')
    return code


def install(nodes, sha, out):
    out.mkdir(parents=True, exist_ok=True)
    codes = parallel([lambda n=n: install_one(n, sha, out / f'install-{n.name}.log') for n in nodes])
    bad = [f'{n.name} (exit {code})' for n, code in zip(nodes, codes) if code]
    if bad:
        raise RuntimeError(f'mesh {sha[:8]} is not installed on {", ".join(bad)}; see {out}/install-*.log')


def ports(n):
    d = n.stat()[1]
    peers, links = d.get('peers', []), d.get('map', {}).get('links', [])
    return (n.name, sum(p.get('phase') == 'PAIRED' for p in peers),
            max(n.discovered_links, sum(int(n.number) in x[:2] for x in links)),
            sum(p.get('node', 0) != 0 or p.get('phase') != 'STOPPED' for p in peers))


def wait_paired(nodes, log):
    for k in range(1, 61):
        state = parallel([lambda n=n: ports(n) for n in nodes])
        ok = all(paired >= links and (links > 0 or len(nodes) == 1) for _, paired, links, _ in state)
        if ok or k == 60:
            break
        time.sleep(5)
    note(log, f'{f"paired after {k * 5} s" if ok else "not paired after 300 s"}: ' + ', '.join(
        f'{name} {paired} of {links} discovered links PAIRED' + (f', {cabled} ports cabled: degraded' if links < cabled else '')
        for name, paired, links, cabled in state))
    return ok


def ranks(nodes, out, tag, folder, command, port, append=False, alarm=300, env='', input=None, states=None):
    master = nodes[0].ula
    states = {} if states is None else states

    def rank(k):
        state = states.setdefault(f'rank{k}', {})
        log = out / f'{tag}-rank{k}.log' if out is not None else None
        invoke = (f'RANK={k} WORLD_SIZE={len(nodes)} MASTER_ADDR={shlex.quote(master)} MASTER_PORT={port} {env}{PY} '
                  f'{home_path(nodes[k].source + "/rdma/mesh_nodes.py")} node guard {alarm} {nodes[k].guard} {command}')
        remote = f'cd {home_path(folder)} && {invoke}'
        try:
            code, output = nodes[k].run(remote, input=input, timeout=alarm + 30, state=state)
        except subprocess.TimeoutExpired as error:
            code = 124
            reason = 'remote process did not exit after its deadline'
            state['diagnostics'].append(reason)
            output = (error.output or b'').decode(errors='replace') + '\n' + reason + '\n'
        if code:
            print(f'{tag} rank{k} (exit {code}):\n' + '\n'.join(state['diagnostics']), file=sys.stderr, flush=True)
        if log is not None:
            with log.open('a' if append else 'w') as destination:
                destination.write(output + f'\nexit {code}\n')
        return {'exit': code, 'output': output, 'records': state['records'], 'diagnostics': state['diagnostics']}

    results = parallel([lambda k=k: rank(k) for k in range(len(nodes))])
    if out is None:
        return {f'rank{k}': row for k, row in enumerate(results)}
    return [line for row in results for line in row['diagnostics'] if 'GUARD' in line]


def hourly_running():
    return sh(['pgrep', '-f', r'tools/mesh/hourly/hourly\.(sh|py)'])[0] == 0


def gate(last=52, skip=None, first=12):
    while skip is None or time.gmtime().tm_min < skip:
        if first <= time.gmtime().tm_min < last and not hourly_running():
            return True
        time.sleep(10)
    return False


def clock(hhmm=None):
    return int(hhmm[:2]) * 60 + int(hhmm[-2:]) if hhmm else time.gmtime().tm_hour * 60 + time.gmtime().tm_min


def lease(path, owner):
    """The cluster lease file, held for `owner` and this process, else None while a live process holds it: the kernel
    releases a flock when its holder exits, so no lease outlives its process."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    held = open(path, 'a+')
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return held.close()
    held.truncate(0)
    held.write(f'{owner} (pid {os.getpid()})\n')
    held.flush()
    return held


@contextlib.contextmanager
def exclusive_lock(owner, path):
    """Shared cluster lease, independent of process-name heuristics and safe for hourly itself."""
    held = lease(path, owner)
    if held is None:
        raise RuntimeError(f'cluster is already leased: {Path(path).read_text().strip()}')
    with held:
        try:
            yield
        finally:
            held.truncate(0)


@contextlib.contextmanager
def lock(owner, path, start=None, after=None, latest=None, by=None, log=None):
    seen, path = False, Path(path)
    while True:
        now = clock()
        if start is None or now >= clock(start):
            text = path.read_text() if path.exists() else ''
            seen = seen or bool(after and text.lower().startswith(after.lower()))
            if (not after or seen or (latest and now >= clock(latest))) and not hourly_running() and (held := lease(path, owner)):
                break
        if by and now >= clock(by):
            note(log, f'no slot by {by} {time.ctime()}')
            raise SystemExit(f'nodes.lock not taken by {by}')
        time.sleep(2)
    with held:
        try:
            note(log, f'lock {time.ctime()}')
            for s in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(s, lambda signum, _: sys.exit(128 + signum))
            yield
        finally:
            held.truncate(0)


def run(*argv, **kw):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)


def node_install(commit, transition='0'):
    from mesh_install import install_checkout
    from mesh_source import revision
    source = Path(__file__).resolve().parent.parent
    if revision(source) != commit:
        raise ValueError('requested install commit differs from the imported checkout')
    install_checkout(source, allow_source_transition=transition == '1')


def node_prepare(commit, number):
    from mesh_install import environment, install_checkout
    from mesh_source import verify
    source = Path(__file__).resolve().parent.parent
    if verify(source) != commit:
        raise ValueError('preparation source differs from the requested commit')
    if str(loads(run(source / 'bin/mesh-stat').stdout).get('node')) != number:
        raise ValueError('the administration endpoint does not own the configured bridge node')
    receipt = source / 'rdma/.build/installed.json'
    wanted = {'commit': commit, 'environment': environment()}
    if not receipt.exists() or json.loads(receipt.read_text()) != wanted:
        install_checkout(source, allow_source_transition=True)
    else:
        node_installed(commit, '1')


def node_installed(commit, exact_imported_source='0'):
    from mesh_install import check_package, verify_active, environment
    from mesh_source import verify
    active = HOME / '.cache/mesh/active-source'
    source = (Path(active.read_text().strip()) if exact_imported_source != '1' and active.exists()
              else Path(__file__).resolve().parent.parent)
    if verify(source) != commit:
        raise ValueError('requested commit differs from the Git checkout')
    receipt = source / 'rdma/.build/installed.json'
    if not receipt.exists() or json.loads(receipt.read_text()) != {'commit': commit, 'environment': environment()}:
        raise ValueError('requested source and build environment have not been installed')
    check_package(source)
    print(json.dumps({'installed': verify_active(source, commit)}))
    return 0


def node_resets(since):
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(since)))
    print('boot ' + re.sub(r'.*} ', '', run('sysctl', '-n', 'kern.boottime').stdout.strip()))
    found = run('sudo', '-n', 'find', '/Library/Logs/DiagnosticReports', '-type', 'f', '-newermt', t, '(', '-name', 'panic-full*',
                '-o', '-name', 'ResetCounter*', '-o', '-name', 'shutdown_stall*', ')').stdout.split()
    for f in found:
        if f.endswith('.panic'):
            b = loads(run('sudo', '-n', 'tail', '-n', '+2', f).stdout)
            pages = b.get('memoryStatus', {}).get('memoryPages', {})
            print(f'{os.path.basename(f)}: {(str(b.get("panicString", "")).splitlines() or [""])[0]}; '
                  f'wired {pages.get("wired", "")} free {pages.get("free", "")}')
        elif f.endswith('.diag'):
            print(f'{os.path.basename(f)}: ' + run('sudo', '-n', 'grep', '-h', 'Boot faults', f).stdout.rstrip('\n'))
        else:
            print(os.path.basename(f))
    shown = run('/usr/bin/log', 'show', '--start', t, '--style', 'compact', '--predicate',
                'process == "kernel" AND eventMessage CONTAINS "pmu fault log:"').stdout
    print(''.join(line + '\n' for line in shown.splitlines() if 'pmu fault log:' in line), end='')


def node_guard(seconds, floor, *argv):
    """Run one rank in its own process group; kill the group at the deadline, or below `floor` free pages (0: no
    floor), and confirm that none of it remains."""
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda signum, _: sys.exit(128 + signum))
    rank, end, why = subprocess.Popen(argv, start_new_session=True), time.monotonic() + float(seconds), None
    free = lambda: int(next(line for line in run('vm_stat').stdout.splitlines()
                            if line.startswith('Pages free')).split()[-1].rstrip('.'))
    try:
        while why is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                return rank.wait(timeout=1)
            pages = free() if int(floor) else 0
            why = (f'deadline {seconds} s' if time.monotonic() >= end
                   else f'GUARD: free pages {pages} below {floor}' if pages < int(floor) else None)
        print(f'{why}: rank {rank.pid} killed', file=sys.stderr, flush=True)
        return 124
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(rank.pid, signal.SIGKILL)
        rank.wait()
        for _ in range(40):
            try:
                os.killpg(rank.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.25)
        else:
            print(f'rank group {rank.pid} remains after SIGKILL', file=sys.stderr, flush=True)


def installed(sha, *names):
    subprocess.run(['git', '-C', ROOT, 'fetch', '-q', 'origin'])
    nodes, sha = [n for n in discover(load())[0] or [] if not names or n.name in names or n.number in names], resolve(sha)
    lines = parallel([lambda n=n: (n.name, *n.py('installed', sha)) for n in nodes])
    print(''.join(f'{name}: {text}' for name, _, text in lines), end='')
    sys.exit(1 if any(code for _, code, _ in lines) or not lines else 0)


def show():
    nodes = discover(load())[0] or []
    on = {n.number for n in up(nodes)}
    for n in nodes:
        print(n.number, n.name, n.addr, f'via {n.via}' if n.via else 'lan', n.ula, 'up' if n.number in on else 'not up')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    remote = bool(argv and argv[0] == 'node')
    if remote:
        argv.pop(0)
    commands = {'install': node_install, 'prepare': node_prepare, 'installed': node_installed, 'resets': node_resets, 'guard': node_guard} \
        if remote else {'installed': installed, 'discover': show}
    return commands[argv[0]](*argv[1:])


if __name__ == '__main__':
    sys.exit(main())
