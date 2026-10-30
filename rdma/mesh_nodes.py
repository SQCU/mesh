"""Shared node discovery, deployment and rank execution; runners supply operands only."""
import concurrent.futures
import contextlib
import codecs
import collections
import dataclasses
import functools
import getpass
import io
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT, HOME = Path(__file__).resolve().parents[1], Path.home()
SSHO = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=4', '-o', 'ControlPath=none']


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
    """etc/mesh-nodes.json with the environment: members by bridge node number, the administrator's user and key, the
    nodes' Python and the first rank port."""
    members: dict
    user: str
    key: 'str | None'
    python: str
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
                   raw['python'], raw['rank_port'])


def home_path(relative):
    return '"$HOME"/' + shlex.quote(str(relative).removeprefix('~/'))



def sh(argv, input=None, timeout=None, state=None, hold=False):
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
        self.source = 'mesh'  # the node's one mesh checkout, ~/mesh
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


def up(nodes, log=None):
    def one(n):
        if n.stat()[1].get('up') is True:
            return n
        note(log, f'not up {n.number}')
    return [n for n in parallel([lambda n=n: one(n) for n in nodes]) if n]


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


def run(*argv, **kw):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)


def node_prepare():
    """Build the node's one checkout, ~/mesh (the bridge's and the ranks', moved to this commit before this runs),
    as nothing but make's and ninja's staleness decide: bin/mesh-user.sh's `make all install` (in parallel), the one
    build of the libraries and the bridge, which restarts the bridge (SIGHUP, its region kept) only where its images
    changed; then _mesh_c and torch_mesh's extension in place."""
    jobs = {**os.environ, 'MESH_RDMA': str(ROOT / 'rdma'), 'MAKEFLAGS': f'-j{os.cpu_count() or 1}'}
    installed, seconds = None, {}
    # the bridge's restart is asked within the first step (bin/mesh-user.sh's SIGHUP), so a bridge started at or
    # after this time answered it, however long the later steps take
    asked = time.time()
    # the bridge's configuration document as bin/mesh-user.sh recorded it; one of a repository's (etc/bridge/<name>) is
    # read from this checkout, the node's one, never from a second checkout nothing updates (every node's named
    # ~/mesh's until 2026-10-29)
    recorded = Path((HOME / 'Library/Application Support/io.mesh.bridge/document').read_text().strip())
    document = ROOT / 'etc/bridge' / recorded.name
    if recorded.parent.parts[-2:] != ('etc', 'bridge') or not document.is_file():
        document = recorded
    for where, *argv in (('.', ROOT / 'bin/mesh-user.sh', document, ROOT),
                         ('rdma', sys.executable, 'mesh_c_build.py'),
                         ('rdma/torch-mesh', sys.executable, 'setup.py', '-q', 'build_ext', '--inplace',
                          '--build-temp', '../.build/torch-mesh')):
        began = time.time()
        done = run(*argv, cwd=ROOT / where, env=jobs)
        seconds[os.path.basename(str(argv[1] if argv[0] == sys.executable else argv[0]))] = round(time.time() - began, 2)
        if done.returncode:
            raise RuntimeError(f'{where}: {" ".join(map(str, argv[:3]))} failed: {(done.stderr or done.stdout)[-600:]}')
        installed = installed or done
    told = {'seconds': seconds}
    # No machine is restarted here (the operator, 2026-10-29: the administrative process kills no machine, by a fault
    # or by a restart that did not need to happen). A new bridge or heartbeat image takes effect by the heartbeat's own
    # re-exec on mesh-user.sh's SIGHUP; a bridge that does not stop, or a heartbeat not running the installed image, is
    # a defect this record names, with any process stuck in the kernel.
    if 'restarts the bridge' in installed.stdout:
        deadline, began = time.time() + BRIDGE_RESTART_SECONDS, bridge_began()
        while not (began and began >= int(asked)) and time.time() < deadline:
            time.sleep(2)
            began = bridge_began()
        faults = kernel_faults()
        told['bridge'] = ('restarted' if began and began >= int(asked) else
                          f'not restarted within {BRIDGE_RESTART_SECONDS} s of its signal' + (f'; {faults}' if faults else ''))
    else:
        told['bridge'] = 'unchanged'
    stale = heartbeat_stale(ROOT)
    if stale:
        told['heartbeat'] = stale
    print(json.dumps({'bridge': told.pop('bridge'), **told}))


def kernel_faults():
    """This boot's kernel faults: a bridge or heartbeat process in an uninterruptible kernel wait (ps state U, as in a
    driver call that never returns), each named for the record. A running machine has had no panic this boot (a panic
    reboots it): a panic report is written at the boot after its panic and describes the boot before (node resets
    reports those)."""
    rows = [line.split(None, 2) for line in run('ps', '-axo', 'pid=,stat=,comm=').stdout.splitlines()]
    return [f'{os.path.basename(r[2])} {r[0]} in state {r[1]}' for r in rows
            if len(r) == 3 and os.path.basename(r[2]) in ('mesh-flow', 'mesh-heartbeat') and 'U' in r[1]]


def heartbeat_stale(root):
    """Why the running heartbeat is not this checkout's installed one (its program another's, or started before this
    image was installed), or None where it is."""
    image = root / 'rdma/mesh-heartbeat'
    pid = next(iter(run('pgrep', '-x', 'mesh-heartbeat').stdout.split()), None)
    if not pid:
        return 'no heartbeat runs'
    program = run('ps', '-o', 'comm=', '-p', pid).stdout.strip()
    if os.path.realpath(program) != os.path.realpath(image):
        return f'the heartbeat runs {program}, not {image}'
    # a heartbeat runs its image again on its restart signal (mesh-heartbeat.c: execv, its pid and start time kept), so
    # whether it runs the installed image is the file it executes (its first text, lsof), not when it started
    running = next((line[1:] for line in run('lsof', '-a', '-p', pid, '-d', 'txt', '-Fi').stdout.splitlines()
                    if line.startswith('i')), None)
    if running and image.is_file() and int(running) != image.stat().st_ino:
        return f'the heartbeat runs an earlier {image} (file {running}; the installed one is {image.stat().st_ino})'
    return None


BRIDGE_RESTART_SECONDS = 180

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
    commands = {'prepare': node_prepare} if remote else {'discover': show}
    return commands[argv[0]](*argv[1:])


if __name__ == '__main__':
    sys.exit(main())
