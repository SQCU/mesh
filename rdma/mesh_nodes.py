"""Shared node discovery, deployment and rank execution; runners supply operands only."""
import concurrent.futures
import contextlib
import codecs
import collections
import ctypes
import dataclasses
import functools
import getpass
import io
import json
import math
import os
import re
import select
import shlex
import signal
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
    guard_headroom_gib: float = 0


@dataclasses.dataclass(frozen=True)
class Cluster:
    """etc/mesh-nodes.json with the environment: members by bridge node number, the administrator's user and key, a
    run's bound in seconds, the nodes' Python, the directory under home of each repository's one checkout
    (`<repository>/current`), and the first rank port."""
    members: dict
    user: str
    key: 'str | None'
    run_seconds: int
    python: str
    checkouts: str
    rank_port: int

    @property
    def ssh(self):
        return [*SSHO, *(['-i', self.key] if self.key else [])]


@functools.cache
def load(path=ROOT / 'etc/mesh-nodes.json'):
    raw = json.loads(Path(path).read_text())
    access = raw.get('access', {})
    members = {str(row['node']): Member(str(row['node']), row['name'], row['management'], row['fabric'],
                                        frozenset(row.get('roles', ())), row.get('guard_headroom_gib', 0)) for row in raw['nodes'].values()}
    return Cluster(members, access.get('user') or getpass.getuser(),
                   str(Path(access['key']).expanduser()) if 'key' in access else None,
                   raw['run_seconds'], raw['python'], raw['checkouts'], raw['rank_port'])


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
        self.name, self.guard = ((member.name, f'{member.guard_headroom_gib} {int("workstation" in member.roles)}')
                                 if member else (number, '0 0'))
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
    """Read each configured node's live map over its administration path. The fabric, the membership, is the largest
    linked part of the maps' union; the configuration says only how to reach a node. A member with no configured entry
    is reached at its fabric-routed address through a reachable configured member (bin/mesh-peers.sh), as before
    e4868b3d. Returns the reachable nodes (their mesh-stat answered) and the absent ones, each with why: unreachable,
    its bridge not up, or outside the fabric."""
    here = str(Node(cluster, '-', 'localhost', local=True, deadline=deadline).stat()[1].get('node'))
    nodes = [Node(cluster, m.number, 'localhost' if m.number == here else m.management, ula=m.fabric,
                  local=m.number == here, deadline=deadline) for m in cluster.members.values()]
    if state is not None:
        state.update({n.name: n.commands for n in nodes})

    def read(node):
        try:
            line, report = node.stat()
        except Exception as error:
            return None, f'unreachable: {error}'
        return (report, None) if report.get('up') else (report or None, f'its bridge is not up: {line[-200:]}')
    reads = dict(zip((n.number for n in nodes), parallel([lambda n=n: read(n) for n in nodes])))
    up = {number: report for number, (report, why) in reads.items() if not why}
    cables = collections.Counter()
    for report in up.values():
        cables |= collections.Counter(tuple(sorted(map(str, link[:2]))) for link in report['map']['links'])
    fabric = max(parts({str(x) for report in up.values() for x in report['map']['nodes']}, cables), key=len, default=set())
    guide = next((n for n in nodes if n.number in fabric and reads[n.number][0] is not None), None)
    unconfigured = sorted(fabric - set(cluster.members), key=int)
    # mesh-peers.sh browses until stopped: 10 s, then SIGTERM to its group, whose trap ends its dns-sd children.
    browse = (f"{cluster.python} -c 'import os, signal, subprocess, sys, time; p = subprocess.Popen(sys.argv[1:], "
              f"start_new_session=True); time.sleep(10); os.killpg(p.pid, signal.SIGTERM); p.wait()' "
              f"{home_path(guide.source + '/bin/mesh-peers.sh')}") if guide else ''
    text = guide.run(browse, timeout=30)[1] if guide and unconfigured else ''
    routes = {f[1]: f[3][5:] for f in map(str.split, text.splitlines()) if f[:1] == ['path'] and f[2:3] == ['kind=fabric-routed']}
    for number in unconfigured:
        if number in routes:
            nodes.append(Node(cluster, number, routes[number], None if guide.local else guide.addr, routes[number],
                              deadline=deadline))
            reads[number] = read(nodes[-1])
        else:
            reads[number] = None, f'unreachable: no fabric-routed path to it through a configured member ({guide and guide.name})'
    absent = {number: why or f'outside the linked fabric {sorted(fabric, key=int)}: its map holds '
                             f'{sorted(map(str, report["map"]["nodes"]), key=int)}'
              for number, (report, why) in reads.items() if why or number not in fabric}
    reachable = [n for n in nodes if reads[n.number][0] is not None]
    for node in reachable:
        node.discovered_links = sum(count for pair, count in cables.items() if node.number in pair)
    note(log, 'fabric ' + ' '.join(sorted(fabric, key=int)), *(f'absent {k}: {v}' for k, v in absent.items()))
    if state is not None:
        state.update(fabric=sorted(fabric, key=int), absent=absent,
                     unconfigured=unconfigured)
    return reachable, absent


def parts(names, pairs):
    """The linked parts of `names` under `pairs`, each a set."""
    linked = {x: set() for x in names}
    for a, b in pairs:
        if a in linked and b in linked:
            linked[a].add(b)
            linked[b].add(a)
    found, seen = [], set()
    for start in sorted(linked, key=int):
        part, todo = set(), [start] if start not in seen else []
        while todo:
            if (x := todo.pop()) not in part:
                part.add(x)
                todo.extend(linked[x] - part)
        seen |= part
        found += [part] if part else []
    return found


def paired(nodes):
    """The node pairs a PAIRED cable joins, as the nodes' bridges report them."""
    return {tuple(sorted((n.number, str(p['node'])))) for n, (_, d) in zip(nodes, parallel([lambda n=n: n.stat() for n in nodes]))
            for p in d.get('peers', []) if p.get('phase') == 'PAIRED'}


def up(nodes, log=None):
    def one(n):
        if n.stat()[1].get('up') is True:
            return n
        note(log, f'not up {n.number}')
    return [n for n in parallel([lambda n=n: one(n) for n in nodes]) if n]


def at(nodes, number):
    return next((n for n in nodes if n.number == number), None)


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


def run(*argv, **kw):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)


def node_prepare():
    """Build the node's one checkout, `current` (the bridge's and the ranks', moved to this commit before this runs),
    as nothing but make's and ninja's staleness decide: bin/mesh-user.sh's `make all install` (in parallel), the one
    build of the libraries and the bridge, which restarts the bridge (SIGHUP, its region kept) only where its images
    changed; then _mesh_c and torch_mesh's extension in place. A node's preparations take turns."""
    import fcntl
    jobs = {**os.environ, 'MESH_RDMA': str(ROOT / 'rdma'), 'MAKEFLAGS': f'-j{os.cpu_count() or 1}'}
    with open(ROOT.parent / 'prepare.lock', 'a') as turn:
        fcntl.flock(turn, fcntl.LOCK_EX)
        installed, seconds = None, {}
        # the bridge's restart is asked within the first step (bin/mesh-user.sh's SIGHUP), so a bridge started at or
        # after this time answered it, however long the later steps take
        asked = time.time()
        for where, *argv in (('.', ROOT / 'bin/mesh-user.sh', (HOME / 'Library/Application Support/io.mesh.bridge/'
                                                               'document').read_text().strip(), ROOT),
                             ('rdma', sys.executable, 'mesh_c_build.py'),
                             ('rdma/torch-mesh', sys.executable, 'setup.py', '-q', 'build_ext', '--inplace',
                              '--build-temp', '../.build/torch-mesh')):
            began = time.time()
            done = run(*argv, cwd=ROOT / where, env=jobs)
            seconds[os.path.basename(str(argv[1] if argv[0] == sys.executable else argv[0]))] = round(time.time() - began, 2)
            if done.returncode:
                raise RuntimeError(f'{where}: {" ".join(map(str, argv[:3]))} failed: {(done.stderr or done.stdout)[-600:]}')
            installed = installed or done
    services = services_refresh(ROOT)
    told = {'services': services, 'seconds': seconds}
    # standup (the operator, 2026-10-29): a machine whose bridge does not answer its restart signal, or whose running
    # heartbeat is not this checkout's installed one (a loaded job takes a new definition only at its next load, a
    # boot: bin/mesh-user.sh), is restarted where this boot has no kernel fault; one with a fault is left as it is, named
    if 'restarts the bridge' in installed.stdout:
        deadline, began = time.time() + BRIDGE_RESTART_SECONDS, bridge_began()
        while not (began and began >= int(asked)) and time.time() < deadline:
            time.sleep(2)
            began = bridge_began()
        if not (began and began >= int(asked)):
            restart_unless_faulted(f'the bridge did not restart within {BRIDGE_RESTART_SECONDS} s of its signal')
        told['bridge'] = 'restarted'
    else:
        told['bridge'] = 'unchanged'
    stale = heartbeat_stale(ROOT)
    if stale:
        restart_unless_faulted(stale)
    print(json.dumps({'bridge': told.pop('bridge'), **told}))


def kernel_faults():
    """This boot's kernel faults: a bridge or heartbeat process in an uninterruptible kernel wait (ps state U, as in a
    driver call that never returns), each named. A running machine has had no panic this boot (a panic reboots it):
    a panic report is written at the boot after its panic and describes the boot before (node resets reports those)."""
    rows = [line.split(None, 2) for line in run('ps', '-axo', 'pid=,stat=,comm=').stdout.splitlines()]
    return [f'{os.path.basename(r[2])} {r[0]} in state {r[1]}' for r in rows
            if len(r) == 3 and os.path.basename(r[2]) in ('mesh-flow', 'mesh-heartbeat') and 'U' in r[1]]


def restart_unless_faulted(why):
    faults = kernel_faults()
    if faults:
        raise RuntimeError(f'{why}, and this boot has a kernel fault, so the machine is left as it is: {faults}')
    run('sudo', '-n', 'shutdown', '-r', 'now')
    raise RuntimeError(f'{why}; no kernel fault this boot, so the machine is restarting')


def heartbeat_stale(root):
    """Why the running heartbeat is not this checkout's installed one (its program another's, or started before this
    image was installed), or None where it is."""
    image = root / 'rdma/mesh-heartbeat'
    pid = next(iter(run('pgrep', '-x', 'mesh-heartbeat').stdout.split()), None)
    if not pid:
        return 'no heartbeat runs'
    program = run('ps', '-o', 'comm=', '-p', pid).stdout.strip()
    if program != str(image):
        return f'the heartbeat runs {program}, not {image}'
    # a heartbeat runs its image again on its restart signal (mesh-heartbeat.c: execv, its pid and start time kept), so
    # whether it runs the installed image is the file it executes (its first text, lsof), not when it started
    running = next((line[1:] for line in run('lsof', '-a', '-p', pid, '-d', 'txt', '-Fi').stdout.splitlines()
                    if line.startswith('i')), None)
    if running and image.is_file() and int(running) != image.stat().st_ino:
        return f'the heartbeat runs an earlier {image} (file {running}; the installed one is {image.stat().st_ino})'
    return None


BRIDGE_RESTART_SECONDS = 180

# the node's services that read it beside every run, each its launchd job and its files (the checkout's, the installed
# one's under /usr/local/mesh, its mode), as install.sh installs them; the router and beacon, which carry the fabric's
# routes, stay install.sh's
SERVICES = (('io.mesh.telemetry', (('user/mesh-telemetry.py', 'bin/mesh-telemetry.py', '755'),
                                   ('rdma/peers.py', 'bin/peers.py', '644'),
                                   ('etc/mesh-capacity.json', 'etc/mesh-capacity.json', '644'))),
            ('io.mesh.observer', (('viz/serve.py', 'viz/serve.py', '755'), ('viz/index.html', 'viz/index.html', '644'),
                                  ('rdma/peers.py', 'bin/peers.py', '644'),
                                  ('etc/mesh-capacity.json', 'etc/mesh-capacity.json', '644'),
                                  ('etc/mesh-nodes.json', 'etc/mesh-nodes.json', '644'))))


def services_refresh(checkout, installed=Path('/usr/local/mesh')):
    """Each loaded service whose files differ from this commit's, installed as install.sh installs them and restarted on
    them (launchctl kickstart -k): {label: 'restarted', 'unchanged' or the failure}."""
    import filecmp
    told = {}
    for label, files in SERVICES:
        if run('launchctl', 'print', f'system/{label}').returncode:
            told[label] = 'not loaded'
            continue
        changed = [(source, target, mode) for source, target, mode in files
                   if not (installed / target).is_file() or not filecmp.cmp(checkout / source, installed / target, shallow=False)]
        if not changed:
            told[label] = 'unchanged'
            continue
        failed = next((done.stderr.strip() or f'install {target} failed' for source, target, mode in changed
                       if (done := run('sudo', '-n', 'install', '-m', mode, '-o', 'root', '-g', 'wheel', checkout / source,
                                       installed / target)).returncode), None)
        if not failed and (done := run('sudo', '-n', 'launchctl', 'kickstart', '-k', f'system/{label}')).returncode:
            failed = done.stderr.strip() or 'kickstart -k failed'
        told[label] = failed or 'restarted'
    return told


def bridge_began():
    """The bridge's start (unix seconds): its first mesh-flow process, the one whose parent is no mesh-flow; None where
    none runs."""
    rows = [line.split(None, 2) for line in run('ps', '-axo', 'pid=,ppid=,comm=').stdout.splitlines()]
    flows = {r[0]: r[1] for r in rows if len(r) == 3 and os.path.basename(r[2]) == 'mesh-flow'}
    main = next((pid for pid, parent in flows.items() if parent not in flows), None)
    if not main:
        return None
    started = run('ps', '-o', 'lstart=', '-p', main).stdout.strip()
    return time.mktime(time.strptime(started, '%a %b %d %H:%M:%S %Y')) if started else None


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
    pids, args, size, found = (ctypes.c_int * 16384)(), ctypes.create_string_buffer(1 << 20), ctypes.c_size_t(), {}
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
            found[pid] = token
    return found


def sweep(match, last):
    """SIGKILL the processes `carrying(match)` names until none is left or `last` (monotonic); those left."""
    while (left := carrying(match)) and time.monotonic() < last:
        for pid in left:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
    return left


def node_guard(seconds, floor, workstation, *argv):
    """Run one rank in its own session, its operands the first line of stdin. End it at its deadline, when the node's
    headroom falls below `floor` GiB (0: none), when stdin closes, or on SIGTERM or SIGHUP; then SIGKILL it and each
    descendant tracked by parent pid (sampled every 0.1 s) until none remains, and then every process still carrying
    this guard's MESH_RUN (carrying), each for at most 30 s (exit 125 naming survivors). Should this guard die, the
    rank's alarm ends it 10 s late. Still escaping: a descendant that leaves its parent between samples and runs a
    platform binary or with MESH_RUN scrubbed, and such a descendant of a dead guard."""
    stop, end, floor, zombie = [], time.monotonic() + float(seconds), float(floor) * 2 ** 30, 5
    # The headroom is memorystatus's numerator A, the pages neither wired nor compressed (free, speculative, pageable),
    # read live by sysctl (host_statistics64 is rate-limited); on a workstation, A less the compressor's bytes in use.
    reads = [(b'vm.page_free_count', page := os.sysconf('SC_PAGE_SIZE')), (b'vm.page_speculative_count', page),
             (b'vm.page_pageable_internal_count', page), (b'vm.page_pageable_external_count', page),
             *([(b'vm.compressor_bytes_used', -1)] if workstation == '1' else [])]
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
    # PYTHONFAULTHANDLER: a rank's fatal signal, and the SIGABRT that ends it past its deadline, writes every thread's
    # traceback to its stderr, so a rank that hangs names where (the guard pauses no rank to look)
    rank = subprocess.Popen(argv, stdin=operands, start_new_session=True, env={**os.environ, 'PYTHONFAULTHANDLER': '1'},
                            preexec_fn=lambda: signal.alarm(math.ceil(float(seconds)) + 10))
    known, why, value, size = {rank.pid: process(rank.pid)[0]}, None, ctypes.c_uint64(), ctypes.c_size_t()
    own = os.environ.get('MESH_RUN')
    others = lambda: sorted(set(carrying(lambda token: token != own).values()))
    from mesh_source import revision
    print(json.dumps({'loaded': revision(ROOT), 'overlap': others()}), flush=True)
    live = lambda pid, start: (seen := process(pid)) is not None and seen[0] == start and seen[1] != zombie
    while why is None and process(rank.pid)[1] != zombie:
        for pid, start in list(known.items()):
            children = (ctypes.c_int * 4096)()
            count = LIBC.proc_listchildpids(pid, children, ctypes.sizeof(children)) if live(pid, start) else 0
            for child in children[:max(0, count)]:
                if child not in known and (seen := process(child)):
                    known[child] = seen[0]
        headroom = 0
        for name, scale in reads if floor else ():
            value.value, size.value = 0, 8
            if LIBC.sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0):
                why = f'GUARD: {name.decode()} unreadable: {os.strerror(ctypes.get_errno())}'
            headroom += value.value * scale
        why = why or (f'received {stop[0]}' if stop else f'deadline {seconds} s' if time.monotonic() >= end
                      else f'GUARD: headroom {headroom / 2 ** 30:.2f} GiB below {floor / 2 ** 30:.2f}' if headroom < floor
                      else 'its caller closed stdin' if select.select([0], [], [], 0)[0] and not os.read(0, 1 << 16)
                      else None)
        time.sleep(0 if why else 0.1)
    print(json.dumps({'overlap': others()}), flush=True)
    if why and why.startswith('deadline') and live(rank.pid, known[rank.pid]):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(rank.pid, signal.SIGABRT)
        with contextlib.suppress(subprocess.TimeoutExpired):
            rank.wait(5)
    last = time.monotonic() + 30
    # EPERM from killpg is a group of zombies or of a member this user cannot signal (macOS), not a failure of the
    # rank: each tracked pid is killed alone, and those still live after 30 s are named (exit 125)
    while (left := [pid for pid, start in known.items() if live(pid, start)]) and time.monotonic() < last:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(rank.pid, signal.SIGKILL)
        for pid in left:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
    token = os.environ.get('MESH_RUN')
    left += list(sweep(token.__eq__, time.monotonic() + 30)) if token else []
    code = rank.poll()
    if why:
        print(f'{why}: rank {rank.pid} killed', file=sys.stderr, flush=True)
    if left:
        print(f'processes {left} of rank {rank.pid} remain after SIGKILL', file=sys.stderr, flush=True)
    return 125 if left else 124 if why else code if code >= 0 else 128 - code


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
    commands = {'prepare': node_prepare, 'resets': node_resets, 'guard': node_guard} if remote else {'discover': show}
    return commands[argv[0]](*argv[1:])


if __name__ == '__main__':
    sys.exit(main())
