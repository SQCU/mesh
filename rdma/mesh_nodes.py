"""Shared node discovery, deployment and rank execution; runners supply operands only."""
import concurrent.futures
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

MB, MESH, HOME = Path('/Users/mdot/metal-microbench'), Path('/Users/mdot/dox/mesh'), Path.home()
HOURLY, KEY = MB / 'tools/mesh/hourly', '/Users/mdot/.ssh/primeintellect_ed25519'
LOCK = Path('/Users/mdot/.claude-personal/jobs/15c469fa/tmp/nodes.lock')
SSHO = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=4', '-o', 'ControlPath=none', '-i', KEY]
PY = '~/.venv-mesh-uv/bin/python'
M5ULT1, M5ULT2, MINI = '521750041', '455770905', '1974797816'
NAMES = {M5ULT1: 'm5ult1', M5ULT2: 'm5ult2', MINI: 'mini'}


def configure(mesh_root, microbench_root, key=None, lock_path=None):
    global MESH, MB, HOURLY, KEY, LOCK, SSHO
    MESH, MB = Path(mesh_root), Path(microbench_root)
    HOURLY = MB / 'tools/mesh/hourly'
    if key is not None:
        KEY = str(key)
        SSHO[-1] = KEY
    if lock_path is not None:
        LOCK = Path(lock_path)


def home_path(relative):
    return '"$HOME"/' + shlex.quote(str(relative).removeprefix('~/'))



def sh(argv, input=None, out=None, append=False, timeout=None):
    io_ = {'input': input} if input is not None else {'stdin': subprocess.DEVNULL}
    if out is not None:
        with open(out, 'ab' if append else 'wb') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, **io_).returncode, ''
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, **io_)
    return p.returncode, p.stdout.decode(errors='replace')


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
    def __init__(self, number, addr, via=None, ula=''):
        self.number, self.addr, self.via, self.ula = number, addr, via, ula
        self.name = NAMES.get(number, number)
        self.source = 'mesh'

    def argv(self, command):
        """The shared administration launcher, also used for resident-rank configuration."""
        proxy = ['-o', f"ProxyCommand=ssh {' '.join(SSHO)} -W '[%h]:%p' mdot@{self.via}"] if self.via else []
        return ['ssh', *SSHO, *proxy, f'mdot@{self.addr}', command]

    def run(self, command, **kw):
        return sh(self.argv(command), **kw)

    def py(self, *args, **kw):
        return self.run(f'PYTHONPATH={home_path(self.source + "/rdma")} {PY} -m mesh_nodes node '
                        + ' '.join(shlex.quote(str(a)) for a in args), **kw)

    def stat(self):
        line = self.run(home_path(self.source + '/bin/mesh-stat'))[1].strip().split('\n')[-1]
        return line, loads(line)


def routed(text):
    paths = {}
    for f in (line.split() for line in text.splitlines()):
        if len(f) > 3 and f[0] == 'path' and f[2] == 'kind=fabric-routed':
            paths.setdefault(f[1], f[3][5:])
    return paths


def discover(log=None, prepare=None):
    with tempfile.TemporaryFile() as f:
        p = subprocess.Popen([MESH / 'bin/mesh-peers.sh'], stdout=f, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(p.pid, signal.SIGTERM)
            p.kill()
            p.wait()
        f.seek(0)
        text = f.read()
    lines = [f.split() for f in text.decode(errors='replace').splitlines()]
    me = next((f[1] for f in lines if len(f) > 2 and f[2] == 'self'), None)
    heard = {f[1]: f[3][5:] for f in lines if len(f) > 3 and f[0] == 'path' and f[2] == 'kind=lan' and f[1] != me}
    tailnet = loads(sh(['/Applications/Tailscale.app/Contents/MacOS/Tailscale', 'status', '--json'])[1])
    candidates, entry, management = [], None, {}
    for n, a in [*heard.items(), *(('-', p['TailscaleIPs'][0]) for p in tailnet.get('Peer', {}).values() if p.get('Online'))]:
        candidates.append(a)
        candidate = Node(n, a)
        if prepare:
            try:
                prepare(candidate)
            except Exception as error:
                note(log, f'candidate {n} {a}: committed discovery helper unavailable: {error}')
                continue
        line, d = candidate.stat()
        if d.get('node'):
            management[str(d['node'])] = a
        note(log, f'candidate {n} {a}: {line or "no reply"}')
        if len(d.get('map', {}).get('nodes', [])) < 2:
            continue
        paths = routed(candidate.run(f"perl -e 'alarm 10; exec @ARGV' {home_path(candidate.source + '/bin/mesh-peers.sh')}")[1])
        entry = entry or (a, d, paths)
        if all(str(m) in paths for m in d['map']['nodes']):
            entry = (a, d, paths)
    note(log, f'entry {entry[0] if entry else "none"}')
    if not entry:
        return None, candidates
    a, d, paths = entry
    nodes = [Node(str(m), heard.get(str(m)) or management.get(str(m)) or paths.get(str(m), ''),
                  None if str(m) in heard or str(m) in management else a, paths.get(str(m), ''))
             for m in d['map']['nodes']]
    note(log, 'map ' + ' '.join(n.number for n in nodes))
    return nodes, candidates


def up(nodes, log=None):
    def one(n):
        if n.stat()[1].get('up') is True:
            return n
        note(log, f'not up {n.number}')
    return [n for n in parallel([lambda n=n: one(n) for n in nodes]) if n]


def at(nodes, number):
    return next((n for n in nodes if n.number == number), None)


def origin_main():
    subprocess.run(['git', '-C', MESH, 'fetch', '-q', 'origin'])
    return resolve('origin/main')


def resolve(ref, repo=MESH):
    return subprocess.run(['git', '-C', repo, 'rev-parse', f'{ref}^{{commit}}'], check=True, capture_output=True,
                          text=True).stdout.strip()


def install_one(n, sha, log):
    from mesh_hourly import stage
    from mesh_source import export
    # Historical callers use the same immutable deployment; they never checkout a target repo.
    with tempfile.TemporaryDirectory(prefix='mesh-install-') as temporary:
        source = Path(temporary) / 'source'
        export(MESH, sha, source)
        for helper in ('mesh_nodes.py', 'mesh_source.py', 'mesh_install.py'):
            if not (source / 'rdma' / helper).exists():
                raise RuntimeError(f'{sha}: shared install helper {helper} is absent; no uncommitted helper injected')
        n.source = f'.cache/mesh-installs/{sha}'
        stage(n, source, n.source)
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
    return (n.name, sum(p.get('phase') == 'PAIRED' for p in peers), sum(int(n.number) in x[:2] for x in links),
            sum(p.get('node', 0) != 0 or p.get('phase') != 'STOPPED' for p in peers))


def wait_paired(nodes, log):
    for k in range(1, 61):
        state = parallel([lambda n=n: ports(n) for n in nodes])
        ok = all(paired >= links for _, paired, links, _ in state)
        if ok or k == 60:
            break
        time.sleep(5)
    note(log, f'{f"paired after {k * 5} s" if ok else "not paired after 300 s"}: ' + ', '.join(
        f'{name} {paired} of {links} map links PAIRED' + (f', {cabled} ports cabled: degraded' if links < cabled else '')
        for name, paired, links, cabled in state))
    return ok


def send(nodes, folder, files):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(body), 0o644, time.time()
            tar.addfile(info, io.BytesIO(body))
    for n in nodes:
        code, output = n.run(f'mkdir -p {home_path(folder)} && tar -C {home_path(folder)} -xf -', input=data.getvalue())
        if code:
            raise RuntimeError(f'{n.name}: source/config upload failed ({code}): {output}')


def ranks(nodes, out, tag, folder, command, port, append=False, alarm=300, env='', timed=False, guard=None):
    master, time_ = nodes[0].ula, '/usr/bin/time -p ' if timed else ''

    def rank(k):
        log = out / f'{tag}-rank{k}.log'
        # Each surviving node keeps its own durable rank log, including its actual exit code.
        invoke = (f'RANK={k} WORLD_SIZE={len(nodes)} MASTER_ADDR={shlex.quote(master)} MASTER_PORT={port} '
                  f"{env}{time_}perl -e 'alarm {alarm}; exec @ARGV' {command}")
        remote = (f'cd {home_path(folder)} && set -o pipefail && {{ {invoke}; mesh_rank_status=$?; '
                  f'printf "exit %s\\n" "$mesh_rank_status"; exit "$mesh_rank_status"; }} '
                  f'2>&1 | tee rank{k}.log')
        code = nodes[k].run(remote, out=log, append=append)[0]
        note(log, f'exit {code}')

    jobs = [lambda k=k: rank(k) for k in range(len(nodes))]
    pattern = f'^[^ ]*python [^ ]*{guard}'
    jobs += [lambda n=n: n.py('guard', pattern, out=out / f'guard-{tag}-{n.name}.log') for n in nodes] if guard else []
    parallel(jobs)
    return [line for n in nodes if guard for line in (out / f'guard-{tag}-{n.name}.log').read_text().splitlines()
            if 'GUARD' in line]


def demo(up_nodes, out, tag, program, config, steps, rows, numbers, port, build_log=None, script=None, args='', **kw):
    folder, nodes = f'mesh-runs/{config}', [at(up_nodes, n) for n in numbers]
    if None in nodes:
        return note(out / f'{tag}-rank0.log', f'not run: node {numbers[nodes.index(None)]} has no bridge up')
    for n in nodes:
        n.run(f'mkdir -p ~/{folder} && cat > ~/{folder}/{program}.json', input=(HOURLY / f'{config}.json').read_bytes())
    nodes[0].run(f'cd ~/{folder} && PYTHONPATH={home_path(nodes[0].source + "/rdma")} {PY} -m mesh_workloads build '
                 f'{home_path(nodes[0].source + "/examples")} {program.removesuffix("-service")} {program}.json {steps} {rows}',
                 out=build_log or out / f'{tag}-rank0.log')
    command = f'{PY} {home_path(nodes[0].source + "/examples/" + program + ".py")} {program}.json'
    if script:
        send(nodes, folder, {script.name: script.read_bytes()})
        command = f'{PY} {script.name} {home_path(nodes[0].source + "/examples")} {program}.json {args}'
    return ranks(nodes, out, tag, folder, command, port, append=build_log is None, **kw)


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


@contextlib.contextmanager
def exclusive_lock(owner, path=None):
    """Shared cluster lease, independent of process-name heuristics and safe for hourly itself."""
    path = Path(path or LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        held = (path / 'owner').read_text() if (path / 'owner').exists() else 'owner registration in progress'
        raise RuntimeError(f'cluster is already leased: {held.strip()}') from None
    try:
        (path / 'owner').write_text(owner + '\n')
        yield
    finally:
        (path / 'owner').unlink(missing_ok=True)
        path.rmdir()


@contextlib.contextmanager
def lock(owner, start=None, after=None, latest=None, by=None, log=None):
    seen = False
    while True:
        now = clock()
        if start is None or now >= clock(start):
            held = (LOCK / 'owner').read_text() if (LOCK / 'owner').exists() else ''
            seen = seen or bool(after and held.lower().startswith(after.lower()))
            if (not after or seen or (latest and now >= clock(latest))) and not hourly_running():
                with contextlib.suppress(FileExistsError):
                    LOCK.mkdir()
                    break
        if by and now >= clock(by):
            note(log, f'no slot by {by} {time.ctime()}')
            raise SystemExit(f'nodes.lock not taken by {by}')
        time.sleep(2)
    try:
        (LOCK / 'owner').write_text(owner + '\n')
        note(log, f'lock {time.ctime()}')
        for s in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, lambda signum, _: sys.exit(128 + signum))
        yield
    finally:
        (LOCK / 'owner').unlink(missing_ok=True)
        LOCK.rmdir()


def run(*argv, **kw):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)


def lanes(text):
    opened, nodes = False, set()
    for line in text.decode(errors='replace').splitlines():
        if re.search(r'heartbeat: bridge [0-9]+ opened', line):
            opened, nodes = True, set()
        f = line.split()
        if 'lanes up toward node' in line:
            nodes |= {f[i + 1] for i in range(1, len(f) - 1) if f[i] == 'node' and f[i - 1] == 'toward' and f[i + 1][:1].isdigit()}
    return opened, nodes


def node_install(commit, transition='0'):
    from mesh_install import install_snapshot
    from mesh_source import revision
    source = Path(__file__).resolve().parent.parent
    if revision(source) != commit:
        raise ValueError('requested install commit differs from the imported snapshot')
    install_snapshot(source, allow_source_transition=transition == '1')


def node_installed(commit, exact_imported_source='0'):
    from mesh_install import check_package, verify_active
    from mesh_source import verify
    active = HOME / '.cache/mesh/active-source'
    source = (Path(active.read_text().strip()) if exact_imported_source != '1' and active.exists()
              else Path(__file__).resolve().parent.parent)
    if verify(source)['commit'] != commit:
        raise ValueError('requested commit differs from source manifest')
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


def node_guard(pattern, floor=65536):
    def pages():
        out = run('vm_stat').stdout.splitlines()
        get = lambda key: int(next(line for line in out if line.startswith(key)).split()[-1].rstrip('.'))
        return get('Pages free'), get('Pages wired')
    say = lambda text: print(time.strftime('%H:%M:%S'), text, flush=True)
    say('before the rank: free %d wired %d' % pages())
    for _ in range(240):
        pid = (run('pgrep', '-f', pattern).stdout.split() or [None])[0]
        if pid:
            break
        time.sleep(0.25)
    else:
        return say('no rank')
    with contextlib.suppress(ProcessLookupError):
        while not os.kill(int(pid), 0):
            free, wired = pages()
            say(f'free {free} wired {wired}')
            if free < int(floor):
                os.kill(int(pid), signal.SIGKILL)
                return say(f'GUARD: free pages {free} below {floor}, rank {pid} killed')
            time.sleep(1)


def installed(sha, *names):
    subprocess.run(['git', '-C', MESH, 'fetch', '-q', 'origin'])
    nodes, sha = [n for n in discover()[0] or [] if not names or n.name in names or n.number in names], resolve(sha)
    lines = parallel([lambda n=n: (n.name, *n.py('installed', sha)) for n in nodes])
    print(''.join(f'{name}: {text}' for name, _, text in lines), end='')
    sys.exit(1 if any(code for _, code, _ in lines) or not lines else 0)


def show():
    nodes = discover()[0] or []
    on = {n.number for n in up(nodes)}
    for n in nodes:
        print(n.number, n.name, n.addr, f'via {n.via}' if n.via else 'lan', n.ula, 'up' if n.number in on else 'not up')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    remote = bool(argv and argv[0] == 'node')
    if remote:
        argv.pop(0)
    commands = {'install': node_install, 'installed': node_installed, 'resets': node_resets, 'guard': node_guard} \
        if remote else {'installed': installed, 'discover': show}
    return commands[argv[0]](*argv[1:])


if __name__ == '__main__':
    sys.exit(main())
