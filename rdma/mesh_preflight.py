#!/usr/bin/env python3
"""Before a launch: is each node in a collective, and is anything running there that the launch would collide with.

It only observes. Nothing is locked, claimed or changed, and no node list is configured:
- The candidates are the nodes whose beacon answers (_meshnode._tcp, bin/mesh-beacon.sh, instance = IOPlatformUUID)
  within BROWSE_S, plus the launch's --target names.
- Each candidate is read over ssh in parallel, with this file sent as the program, so its own checkout does not matter.
  Each reading is bounded to READ_S seconds. A node is its IOPlatformUUID; a node reached by two names is one node.
- The cables come from the Thunderbolt hardware. Each node's own Domain UUIDs (IOThunderboltLocalNode) and the far-end
  Domain UUID of each of its host-to-host links (IOThunderboltXDomainLink) are read. A link whose far end a probed node
  owns is a cable between the two. A link whose far end no probed node owns is a cable to an unseen node.
- A node is in a collective when any of its RDMA ports is PORT_ACTIVE (ibv_devinfo). The targets' collective is every
  probed node reachable from them over cables. An unseen cable in it blocks the targets, since what runs there is
  unknown.

Busy, anywhere in the targets' collective:
- a process other than a bridge with a mesh library loaded (a mesh client or program, whatever the bridge's version);
- the hourly (mesh_hourly.py);
- the GPU at --gpu-busy percent or more.

Blocked, on a target, by what the launch says it will use:
- --bridge: a bridge already runs there, so the launch's bridge would be a second region;
- --bytes N: N bytes the launch places in memory there, at or above the node's unpinned memory (hw.memsize - wired);
- --port P: a process already listens on TCP port P;
- --region NAME: a POSIX shared-memory object NAME already exists (the launch's bridge would unlink it);
- --device NODE=DEV: RDMA device DEV is missing on NODE, or its port is not PORT_ACTIVE.

ssh uses --ssh-identity, or $MESH_SSH_IDENTITY, as the key the nodes accept. Exit 0 when every target is clear.
"""
import argparse, concurrent.futures, ctypes, json, os, re, subprocess, sys
from pathlib import Path

BROWSE_S, READ_S = 1.0, 4
TAILSCALE = (['/usr/local/mesh/bin/tailscale', '--socket=/var/run/tailscaled.socket'],
             ['/Applications/Tailscale.app/Contents/MacOS/Tailscale'], ['tailscale'])
LIBRARIES = re.compile(r'/(libmesh[^/]*|libnccl-mesh[^/]*|libmesh_metal[^/]*|libmetal_recording[^/]*)\.dylib$'
                       r'|/torch_mesh/_stream[^/]*\.so$')
BRIDGES = ('mesh-flow', 'mesh-heartbeat')
UUID = r'"%s" = "([0-9A-F-]{36})"'


def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=READ_S - 2).stdout
    except (OSError, subprocess.SubprocessError):
        return ''


def listen(cmd, seconds):
    """What a never-ending command (dns-sd) prints within `seconds`; it is then stopped by SIGTERM."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return ''
    try:
        return p.communicate(timeout=seconds)[0]
    except subprocess.TimeoutExpired:
        p.terminate()
        return p.communicate()[0]


def probe(region=None):
    """This node's facts, all read-only."""
    page = int(run('sysctl', '-n', 'hw.pagesize') or 16384)
    memsize = int(run('sysctl', '-n', 'hw.memsize') or 0)
    wired = re.search(r'Pages wired down:\s+(\d+)', run('vm_stat'))
    wired = int(wired.group(1)) * page if wired else 0
    ports = {}
    for block in re.split(r'\n(?=hca_id:)', run('ibv_devinfo')):
        name, state = re.search(r'hca_id:\s*(\S+)', block), re.search(r'state:\s*(\w+)', block)
        if name:
            ports[name.group(1)] = state.group(1) if state else 'UNKNOWN'
    uuid = re.search(UUID % 'IOPlatformUUID', run('ioreg', '-r', '-d', '1', '-c', 'IOPlatformExpertDevice'))
    domains = re.findall(UUID % 'Domain UUID', run('ioreg', '-r', '-d', '1', '-l', '-w', '0', '-c',
                                                   'IOThunderboltLocalNode'))
    far = re.findall(UUID % 'Domain UUID', run('ioreg', '-r', '-d', '1', '-l', '-w', '0', '-c',
                                               'IOThunderboltXDomainLink'))
    processes = {}
    for line in run('ps', '-axo', 'pid=,command=').splitlines():
        pid, _, command = line.strip().partition(' ')
        if pid.isdigit():
            processes[int(pid)] = command.strip()
    bridges = {pid: c for pid, c in processes.items() if os.path.basename(c.split(' ')[0]) in BRIDGES}
    loaded, pid = set(), None
    for line in run('lsof', '-n', '-w', '-d', 'txt', '-Fpn').splitlines():
        if line.startswith('p'):
            pid = int(line[1:])
        elif line.startswith('n') and pid is not None and LIBRARIES.search(line):
            loaded.add(pid)
    clients = {pid: processes.get(pid, '') for pid in loaded if pid not in bridges and pid != os.getpid()}
    listening, pid, command = {}, None, ''
    for line in run('lsof', '-nP', '-w', '-iTCP', '-sTCP:LISTEN', '-Fpcn').splitlines():
        if line.startswith('p'):
            pid = int(line[1:])
        elif line.startswith('c'):
            command = line[1:]
        elif line.startswith('n') and ':' in line:
            listening.setdefault(line.rsplit(':', 1)[1], '%s (pid %s)' % (command, pid))
    gpu = [int(v) for v in re.findall(r'"Device Utilization %"=(\d+)',
                                      run('ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'))]
    tailnet = {}
    for cli in TAILSCALE:
        try:
            status = json.loads(run(*cli, 'status', '--json') or 'null')
        except ValueError:
            status = None
        if status:
            for peer in [status.get('Self') or {}] + list((status.get('Peer') or {}).values()):
                ips = [ip for ip in peer.get('TailscaleIPs') or [] if '.' in ip]
                if peer.get('HostName') and ips:
                    tailnet[peer['HostName'].lower()] = ips[0]
            break
    region_exists = None
    if region:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.shm_open.argtypes, libc.shm_open.restype = [ctypes.c_char_p, ctypes.c_int], ctypes.c_int
        fd = libc.shm_open(region.encode(), os.O_RDONLY)
        region_exists = fd >= 0
        if fd >= 0:
            os.close(fd)
    return {'host': run('hostname', '-s').strip(), 'uuid': uuid.group(1) if uuid else None,
            'memsize': memsize, 'wired': wired, 'unpinned': memsize - wired, 'ports': ports,
            'domains': sorted(set(domains)), 'far': far,
            'bridges': {str(k): v for k, v in bridges.items()}, 'clients': {str(k): v for k, v in clients.items()},
            'hourly': [c for c in processes.values() if 'mesh_hourly.py' in c],
            'listening': listening, 'gpu': max(gpu) if gpu else None, 'region_exists': region_exists,
            'tailnet': tailnet}


def beacons():
    """{IOPlatformUUID: (its .local host, its serial)} for every node whose beacon answers within BROWSE_S."""
    found = re.findall(r'\s_meshnode\._tcp\.\s+([0-9A-F-]{36})\s*$', listen(['dns-sd', '-B', '_meshnode._tcp',
                                                                             'local.'], BROWSE_S), re.M)

    def resolve(instance):
        text = listen(['dns-sd', '-L', instance, '_meshnode._tcp', 'local.'], BROWSE_S)
        at, serial = re.search(r'can be reached at ([^:\s]+?)\.?:\d+', text), re.search(r'\bserial=(\S+)', text)
        return instance, (at.group(1) if at else None, serial.group(1) if serial else None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(set(found)))) as pool:
        return {i: hs for i, hs in pool.map(resolve, sorted(set(found))) if hs[0]}


def read(via, identity, region):
    """One node's probe over ssh; None if it did not answer within READ_S seconds."""
    key = ['-i', identity] if identity else []
    extra = ['--region', region] if region else []
    try:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=2', *key, via, 'python3', '-I',
                              '-', '--probe', *extra], input=Path(__file__).read_text(), capture_output=True,
                             text=True, timeout=READ_S)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--target', action='append', default=[], help='local, a host name, or an ssh destination')
    ap.add_argument('--bridge', action='store_true')
    ap.add_argument('--bytes', type=int)
    ap.add_argument('--port')
    ap.add_argument('--region')
    ap.add_argument('--device', action='append', default=[], help='NODE=DEV')
    ap.add_argument('--gpu-busy', type=int, default=90)
    ap.add_argument('--ssh-identity', default=os.environ.get('MESH_SSH_IDENTITY'))
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    if a.probe:
        print(json.dumps(probe(a.region)))
        return 0

    here = probe(a.region)
    facts, name_of = {here['host']: here}, {'local': here['host'], here['host']: here['host']}
    by_uuid, answers = {here['uuid']: here['host']}, {}

    def gather(vias):
        """Read every via in parallel; a node reached by two names is one node (its IOPlatformUUID)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(vias))) as pool:
            got = dict(zip(vias, pool.map(lambda v: read(v, a.ssh_identity, a.region), vias)))
        for via, f in got.items():
            answers[via] = f
            if f and f['uuid'] in by_uuid:
                name_of[via] = by_uuid[f['uuid']]
            elif f:
                facts[f['host']], by_uuid[f['uuid']] = f, f['host']
                name_of[via] = name_of[f['host']] = name_of[f['host'] + '.local'] = f['host']

    # a beacon's node at the tailnet address of its serial (the tailnet names each node by its serial, and its host
    # key is known there), else at its .local host
    gather(sorted({here['tailnet'].get((serial or '').lower(), host) for i, (host, serial) in beacons().items()
                   if i != here['uuid']}))
    gather(sorted({t for t in a.target if t not in name_of}))
    silent = sorted(v for v, f in answers.items() if not f)
    owner = {d: n for n, f in facts.items() for d in f['domains']}

    def cables(n):
        return [(d, owner.get(d)) for d in facts[n]['far']]

    targets = [name_of.get(t, t) for t in (a.target or ['local'])]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mesh_graph  # the graph module (C9); the launcher's side only, after --probe has returned
    collective = mesh_graph.reach([t for t in targets if t in facts], lambda n: [o for _, o in cables(n) if o])

    busy, blocked = [], {t: [] for t in targets}
    for n in sorted(collective):
        f = facts[n]
        busy += ['%s: mesh program pid %s: %s' % (n, p, c[:100]) for p, c in f['clients'].items()]
        busy += ['%s: the hourly is running' % n for _ in f['hourly'][:1]]
        if f['gpu'] is not None and f['gpu'] >= a.gpu_busy:
            busy.append('%s: GPU %s%% busy' % (n, f['gpu']))
        for d, o in cables(n):
            if o is None:
                for t in targets:
                    blocked[t].append('%s has a cable to an unseen node (domain %s): its collective is not all seen'
                                      % (n, d))
    devices = {}
    for d in a.device:
        node, _, dev = d.partition('=')
        devices.setdefault(name_of.get(node, node), []).append(dev)
    for t in targets:
        f = facts.get(t)
        if not f:
            blocked[t].append('%s: did not answer within %s s%s' % (t, READ_S, '' if a.ssh_identity else
                              ' (no --ssh-identity or MESH_SSH_IDENTITY given)'))
            continue
        if a.bridge and f['bridges']:
            blocked[t].append('%s: a bridge already runs (pids %s); the launch would be a second region'
                              % (t, ', '.join(sorted(f['bridges'], key=int))))
        if a.bytes is not None and a.bytes >= f['unpinned']:
            blocked[t].append('%s: the launch places %.1f GiB, at or above its unpinned %.1f GiB (%.1f of %.0f GiB '
                              'wired)' % (t, a.bytes / 2**30, f['unpinned'] / 2**30, f['wired'] / 2**30,
                                          f['memsize'] / 2**30))
        if a.port and a.port in f['listening']:
            blocked[t].append('%s: port %s is taken by %s' % (t, a.port, f['listening'][a.port]))
        if f['region_exists']:
            blocked[t].append('%s: shared memory %s already exists' % (t, a.region))
        for dev in devices.get(t, []):
            if f['ports'].get(dev) != 'PORT_ACTIVE':
                blocked[t].append('%s: %s is %s' % (t, dev, f['ports'].get(dev, 'not an RDMA device here')))

    if a.json:
        print(json.dumps({'facts': facts, 'silent': silent, 'targets': targets, 'collective': sorted(collective),
                          'busy': busy, 'blocked': blocked}, indent=1))
    else:
        print('%-20s %-6s %-7s %-10s %-9s %-5s %s' % ('node', 'ports', 'cables', 'bridge', 'programs', 'gpu',
                                                      'unpinned'))
        for n in sorted(facts):
            f = facts[n]
            up = sum(s == 'PORT_ACTIVE' for s in f['ports'].values())
            seen = sum(1 for _, o in cables(n) if o)
            print('%-20s %-6s %-7s %-10s %-9s %-5s %.1f GiB of %.0f' % (
                n, '%d/%d' % (up, len(f['ports'])), '%d/%d' % (seen, len(f['far'])),
                'pid ' + min(f['bridges'], key=int) if f['bridges'] else '-', len(f['clients']),
                '' if f['gpu'] is None else f['gpu'], f['unpinned'] / 2**30, f['memsize'] / 2**30))
        for v in silent:
            print('%-20s did not answer%s' % (v, '' if a.ssh_identity else
                                              ' (no --ssh-identity or MESH_SSH_IDENTITY given)'))
        print('collective of %s: %s' % (', '.join(targets), ', '.join(sorted(collective)) or 'none'))
        for t in targets:
            reasons = blocked[t] + busy
            print('%s: %s' % (t, 'clear' if not reasons else 'not clear\n  ' + '\n  '.join(reasons)))
    return 0 if not busy and not any(blocked.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
