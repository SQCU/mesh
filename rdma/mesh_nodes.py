"""Shared node discovery, deployment and rank execution; runners supply operands only."""
import concurrent.futures
import contextlib
import codecs
import ctypes
import dataclasses
import fcntl
import functools
import getpass
import io
import errno
import json
import math
import os
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT, HOME = Path(__file__).resolve().parents[1], Path.home()
SSHO = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=4', '-o', 'ControlPath=none']
RUNNING, RELEASED = set(), []
LIBC = ctypes.CDLL(None, use_errno=True)


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
    lease and its ownerless grace, a run's bound in seconds, the nodes' Python, their checkouts under home and the first
    rank port."""
    members: dict
    user: str
    key: 'str | None'
    lock: Path
    lock_grace: int
    run_seconds: int
    python: str
    checkouts: str
    rank_port: int

    @property
    def ssh(self):
        return [*SSHO, *(['-i', self.key] if self.key else [])]

    def place(self, need):
        """The `compute` members also holding `need`'s role, with at least its other values, in rank order: the root
        first, the rest by bridge node number (metal-microbench tools/mesh/grid.py order)."""
        roles = {'compute', need.get('role', 'compute')}
        chosen = [m for m in self.members.values()
                  if roles <= m.roles and all(getattr(m, k) >= v for k, v in need.items() if k != 'role')]
        return sorted(chosen, key=lambda m: ('root' not in m.roles, int(m.number)))


@functools.cache
def load(path=ROOT / 'etc/mesh-nodes.json'):
    raw = json.loads(Path(path).read_text())
    access = raw.get('access', {})
    members = {str(row['node']): Member(str(row['node']), row['name'], row['management'], row['fabric'],
                                        frozenset(row.get('roles', ())), row.get('memory_gib', 0),
                                        row.get('guard_free_pages', 0)) for row in raw['nodes'].values()}
    return Cluster(members, access.get('user') or getpass.getuser(),
                   str(Path(access['key']).expanduser()) if 'key' in access else None, Path(raw['lock']).expanduser(),
                   raw['lock_grace_seconds'], raw['run_seconds'], raw['python'], raw['checkouts'], raw['rank_port'])


def home_path(relative):
    return '"$HOME"/' + shlex.quote(str(relative).removeprefix('~/'))



def sh(argv, input=None, timeout=None, state=None, hold=False):
    if RELEASED:
        raise RuntimeError('this process released its commands and starts no more')
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
    RUNNING.add(p)

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
            p.stdin.flush()
            if not hold:
                p.stdin.close()
        except (BrokenPipeError, ValueError):
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
    finally:
        RUNNING.discard(p)
        if p.stdin:
            with contextlib.suppress(OSError, ValueError):
                p.stdin.close()
    state.update(exit=p.returncode, finished_unix=time.time())
    # A descendant that outlives its command may hold the output pipe.
    reader.join(10)
    if writer:
        writer.join(10)
    output = ''.join(state['output'])
    if state['status'] == 'timed-out':
        raise subprocess.TimeoutExpired(argv, timeout, output=output.encode())
    return p.returncode, output


def release():
    """End every running command: a guarded rank's held stdin is closed (its guard then ends the rank and exits), and
    the rest are terminated. No command starts after."""
    RELEASED.append(True)
    for p in list(RUNNING):
        with contextlib.suppress(OSError, ValueError):
            if p.stdin and not p.stdin.closed:
                p.stdin.close()
            else:
                p.terminate()


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
    def __init__(self, cluster, number, addr, via=None, ula='', *, local=False, deadline=None):
        self.cluster, self.number, self.addr, self.via, self.ula = cluster, number, addr, via, ula
        self.local, self.deadline = local, deadline
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

    def run(self, command, state=None, timeout=None, **kw):
        if self.deadline is not None:
            left = self.deadline - time.time()
            if left <= 0:
                raise TimeoutError(f'{self.name}: the run deadline has passed')
            timeout = left if timeout is None else min(timeout, left)
        state = {} if state is None else state
        self.commands.append(state)
        return sh(self.argv(command), state=state, timeout=timeout, **kw)

    def py(self, *args, **kw):
        return self.run(f'PYTHONPATH={home_path(self.source + "/rdma")} {self.cluster.python} -m mesh_nodes node '
                        + ' '.join(shlex.quote(str(a)) for a in args), **kw)

    def stat(self):
        line = self.run(home_path(self.source + '/bin/mesh-stat'))[1].strip().split('\n')[-1]
        return line, loads(line)


def discover(cluster, log=None, state=None, deadline=None):
    """Read the local bridge's existing cluster map and bind configured endpoints."""
    local = Node(cluster, '-', 'localhost', local=True, deadline=deadline)
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
                  ula=m.fabric, local=int(m.number) == report['node'], deadline=deadline) for m in cluster.members.values()]
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


def resolve(ref, repo=ROOT):
    return subprocess.run(['git', '-C', repo, 'rev-parse', f'{ref}^{{commit}}'], check=True, capture_output=True,
                          text=True).stdout.strip()


def ports(n):
    d = n.stat()[1]
    peers, links = d.get('peers', []), d.get('map', {}).get('links', [])
    return (n.name, sum(p.get('phase') == 'PAIRED' for p in peers),
            max(n.discovered_links, sum(int(n.number) in x[:2] for x in links)),
            sum(p.get('node', 0) != 0 or p.get('phase') != 'STOPPED' for p in peers))


def wait_paired(nodes, log, end=math.inf):
    for k in range(1, 61):
        state = parallel([lambda n=n: ports(n) for n in nodes])
        ok = all(paired >= links and (links > 0 or len(nodes) == 1) for _, paired, links, _ in state)
        if ok or k == 60 or time.time() + 5 >= end:
            break
        time.sleep(5)
    note(log, f'{"paired" if ok else "not paired"} after {k * 5} s: ' + ', '.join(
        f'{name} {paired} of {links} discovered links PAIRED' + (f', {cabled} ports cabled: degraded' if links < cabled else '')
        for name, paired, links, cabled in state))
    return ok


def ranks(nodes, tag, folder, command, port, alarm=300, env='', input=None, states=None, run=''):
    """One rank a node under `node guard` with the run's token (MESH_RUN), its operands the first line of a stdin held
    open until it ends: closing that stdin (release) has its guard end it."""
    master = nodes[0].ula
    states = {} if states is None else states

    def rank(k):
        state, node = states.setdefault(f'rank{k}', {}), nodes[k]
        invoke = (f'MESH_RUN={shlex.quote(run)} RANK={k} WORLD_SIZE={len(nodes)} MASTER_ADDR={shlex.quote(master)} '
                  f'MASTER_PORT={port} {env}'
                  f'{node.cluster.python} {home_path(node.source + "/rdma/mesh_nodes.py")} node guard {alarm} {node.guard} '
                  f'{command}')
        try:
            code, output = node.run(f'cd {home_path(folder)} && {invoke}', input=(input or b'') + b'\n', hold=True,
                                    timeout=alarm + 30, state=state)
        except subprocess.TimeoutExpired as error:
            code = 124
            reason = 'remote process did not exit after its deadline'
            state['diagnostics'].append(reason)
            output = (error.output or b'').decode(errors='replace') + '\n' + reason + '\n'
        if code:
            print(f'{tag} rank{k} (exit {code}):\n' + '\n'.join(state['diagnostics']), file=sys.stderr, flush=True)
        return {'exit': code, 'output': output, 'records': state['records'], 'diagnostics': state['diagnostics']}

    return {f'rank{k}': row for k, row in enumerate(parallel([lambda k=k: rank(k) for k in range(len(nodes))]))}


def started(pid):
    """`pid`'s start time while it lives (not a zombie), else None."""
    seen = process(pid)
    return f'{seen[0][0]}.{seen[0][1]:06d}' if seen and seen[1] != 5 else None


def holder(pid):
    return f'pid {pid} start {started(pid)} host {socket.gethostname()}\n'


def stale(path, grace):
    """No holder the lease's owner file names (`pid P start S host H`) lives on this host, or it has no owner file and
    its directory was made (st_birthtime) more than `grace` seconds ago."""
    try:
        text = (path / 'owner').read_text()
    except FileNotFoundError:
        with contextlib.suppress(FileNotFoundError):
            return time.time() - path.stat().st_birthtime > grace
        return False
    except OSError:
        return False
    holders = re.findall(r'^pid (\d+) start (\S+) host (\S+)$', text, re.M)
    return bool(holders) and all(host == socket.gethostname() and started(int(pid)) != start
                                 for pid, start, host in holders)


def remove(path):
    """Take a lease away in one rename into a private directory, then delete it: an interruption leaves none."""
    tomb = Path(tempfile.mkdtemp(prefix=path.name + '.removed.', dir=path.parent))
    try:
        os.rename(path, tomb / 'lease')
    finally:
        shutil.rmtree(tomb)


@contextlib.contextmanager
def exclusive_lock(owner, path, grace):
    """Shared cluster lease: the directory at `path`, which every session's tools take by mkdir, here renamed into place
    whole (renamex_np RENAME_EXCL, which fails while it exists) with its owner file naming the holder's pid, start time
    and host. `hold` names another holder. A stale lease (stale) is removed under a flock that serializes the sessions
    removing one; it is released by one rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    made = Path(tempfile.mkdtemp(prefix=path.name + '.made.', dir=path.parent))
    (made / 'owner').write_text(f'{owner}\n{holder(os.getpid())}')
    with open(path.with_name(path.name + '.reclaim'), 'a') as serial:
        fcntl.flock(serial, fcntl.LOCK_EX)
        if stale(path, grace):
            with contextlib.suppress(FileNotFoundError):
                remove(path)
        if LIBC.renamex_np(os.fsencode(made), os.fsencode(path), 4):
            failure = ctypes.get_errno()
            shutil.rmtree(made)
            if failure != errno.EEXIST:
                raise OSError(failure, os.strerror(failure), str(path))
            held = 'no owner file'
            with contextlib.suppress(OSError):
                held = (path / 'owner').read_text().strip()
            raise RuntimeError(f'cluster is already leased: {held}')

    def hold(pid):
        (path / 'owner.new').write_text((path / 'owner').read_text() + holder(pid))
        os.replace(path / 'owner.new', path / 'owner')
    try:
        yield hold
    finally:
        with contextlib.suppress(FileNotFoundError):
            remove(path)


def run(*argv, **kw):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)


def node_install(commit, transition='0'):
    from mesh_install import install_checkout
    from mesh_source import revision
    source = Path(__file__).resolve().parent.parent
    if revision(source) != commit:
        raise ValueError('requested install commit differs from the imported checkout')
    install_checkout(source, allow_source_transition=transition == '1')


def node_prepare(commit, number, run):
    from mesh_install import environment, install_checkout
    from mesh_source import verify
    if left := sweep(lambda token: token != run, time.monotonic() + 30):
        raise RuntimeError(f'processes {left} of an earlier run remain after SIGKILL')
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


class Process(ctypes.Structure):
    """libproc's struct proc_bsdinfo (PROC_PIDTBSDINFO)."""
    _fields_ = [('flags', ctypes.c_uint32), ('status', ctypes.c_uint32), ('xstatus', ctypes.c_uint32),
                ('pid', ctypes.c_uint32), ('ppid', ctypes.c_uint32), ('ids', ctypes.c_uint32 * 7),
                ('comm', ctypes.c_char * 16), ('name', ctypes.c_char * 32), ('nfiles', ctypes.c_uint32),
                ('pgid', ctypes.c_uint32), ('tty', ctypes.c_uint32 * 4), ('start', ctypes.c_uint64 * 2)]


def process(pid):
    """`pid`'s (start time, status), reading a process through its exit as docs/facts.md has it, else None."""
    info = Process()
    if LIBC.proc_pidinfo(pid, 3, 1, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        return None
    return tuple(info.start), info.status


def carrying(match):
    """This user's processes but this one whose environment's MESH_RUN `match` accepts (sysctl KERN_PROCARGS2). It does
    not read a zombie, another user's process, or a platform binary, whose environment macOS hides."""
    pids, args, size, found = (ctypes.c_int * 16384)(), ctypes.create_string_buffer(1 << 20), ctypes.c_size_t(), []
    for pid in pids[:max(0, LIBC.proc_listpids(4, os.getuid(), pids, ctypes.sizeof(pids))) // 4]:
        size.value = len(args)
        if pid in (0, os.getpid()) or LIBC.sysctl((ctypes.c_int * 3)(1, 49, pid), 3, args, ctypes.byref(size), None, 0):
            continue
        raw = ctypes.string_at(args, size.value)
        words = raw[4:].split(b'\0')[1:]
        first = next((i for i, word in enumerate(words) if word), len(words))
        token = next((w[9:].decode(errors='replace') for w in words[first + int.from_bytes(raw[:4], sys.byteorder):]
                      if w.startswith(b'MESH_RUN=')), None)
        if token is not None and match(token):
            found.append(pid)
    return found


def sweep(match, last):
    """SIGKILL the processes `carrying(match)` names until none is left or `last` (monotonic); those left."""
    while (left := carrying(match)) and time.monotonic() < last:
        for pid in left:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
    return left


def node_guard(seconds, floor, *argv):
    """Run one rank in its own session, its operands the first line of stdin. End it at its deadline, below `floor`
    free pages (0: none), when stdin closes, or on SIGTERM or SIGHUP; then SIGKILL it and each descendant tracked by
    parent pid (sampled every 0.1 s) until none remains, and then every process still carrying this guard's MESH_RUN
    (carrying), each for at most 30 s (exit 125 naming survivors). Should this guard die, the rank's alarm ends it 10 s
    late. Still escaping: a descendant that leaves its parent between samples and runs a platform binary or with
    MESH_RUN scrubbed, and such a descendant of a dead guard."""
    stop, end, floor, zombie = [], time.monotonic() + float(seconds), int(floor), 5
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda signum, _: stop.append(signal.Signals(signum).name))
    while not (stop or time.monotonic() >= end or select.select([0], [], [], 0.1)[0]):
        pass
    line = b'' if stop or time.monotonic() >= end else sys.stdin.buffer.readline()
    if not line.endswith(b'\n'):
        print(f'{stop[0] if stop else "no operands"}: rank not started', file=sys.stderr, flush=True)
        return 124
    operands = tempfile.TemporaryFile()
    operands.write(line[:-1])
    operands.seek(0)
    rank = subprocess.Popen(argv, stdin=operands, start_new_session=True,
                            preexec_fn=lambda: signal.alarm(math.ceil(float(seconds)) + 10))
    known, why, pages, size = {rank.pid: process(rank.pid)[0]}, None, ctypes.c_uint32(), ctypes.c_size_t(4)
    live = lambda pid, start: (seen := process(pid)) is not None and seen[0] == start and seen[1] != zombie
    while why is None and process(rank.pid)[1] != zombie:
        for pid, start in list(known.items()):
            children = (ctypes.c_int * 4096)()
            count = LIBC.proc_listchildpids(pid, children, ctypes.sizeof(children)) if live(pid, start) else 0
            for child in children[:max(0, count)]:
                if child not in known and (seen := process(child)):
                    known[child] = seen[0]
        if floor and LIBC.sysctlbyname(b'vm.page_free_count', ctypes.byref(pages), ctypes.byref(size), None, 0):
            why = f'GUARD: vm.page_free_count unreadable: {os.strerror(ctypes.get_errno())}'
        why = why or (f'received {stop[0]}' if stop else f'deadline {seconds} s' if time.monotonic() >= end
                      else f'GUARD: free pages {pages.value} below {floor}' if pages.value < floor
                      else 'its caller closed stdin' if select.select([0], [], [], 0)[0] and not os.read(0, 1 << 16)
                      else None)
        time.sleep(0 if why else 0.1)
    last = time.monotonic() + 30
    while (left := [pid for pid, start in known.items() if live(pid, start)]) and time.monotonic() < last:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(rank.pid, signal.SIGKILL)
        for pid in left:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
    token = os.environ.get('MESH_RUN')
    left += sweep(token.__eq__, time.monotonic() + 30) if token else []
    code = rank.poll()
    if why:
        print(f'{why}: rank {rank.pid} killed', file=sys.stderr, flush=True)
    if left:
        print(f'processes {left} of rank {rank.pid} remain after SIGKILL', file=sys.stderr, flush=True)
    return 125 if left else 124 if why else code if code >= 0 else 128 - code


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
