"""python -m route cluster COMMAND: rank i on cluster.hosts[i], over ssh.

  build              the sources packed here, unpacked into cluster.dir on every host and built there (build.sh)
  train [--set ...]  a flush of any dead run's exchange programs (every rank's xbench 0), then every rank's training,
                     detached (nohup); each rank's records into cluster.dir/NAME-RANK.jsonl; NAME the run.name setting
  xbench N           the exchange alone on every rank
  stat               every rank's telemetry record, now
  logs NAME          the last record of each kind from every rank's NAME log

The bootstrap root's port is one rank 0's host hands out (bound to port 0 there), fresh every launch."""
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading

from .config import HERE

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # the checkout
SOURCES = ['rdma/mesh_c_build.py', 'rdma/tests/switched-world.h', 'rdma/tests/fabric-20261119.txt']   # and rdma/*.h (the bridge's switch: the environment)
ROUTE = ['*.c', '*.h', '*.swift', '*.metal', 'build.sh']


def ssh(host, cmd, capture=True):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd], capture_output=capture, text=True)


def each(hosts, fn):
    out = [None] * len(hosts)
    ts = [threading.Thread(target=lambda i=i, h=h: out.__setitem__(i, fn(i, h))) for i, h in enumerate(hosts)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def pack():
    import glob
    f = tempfile.NamedTemporaryFile(suffix='.tgz', delete=False)
    with tarfile.open(fileobj=f, mode='w:gz') as tar:
        for s in SOURCES + [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, 'rdma/*.h'))]:
            tar.add(os.path.join(REPO, s), s)
        for pat in ROUTE:
            for p in glob.glob(os.path.join(REPO, 'rdma/tests/route', pat)):
                tar.add(p, os.path.relpath(p, REPO))
        tar.add(os.path.join(REPO, 'rdma/tests/route/py'), 'rdma/tests/route/py', filter=lambda t: None if '__pycache__' in t.name or t.name.endswith(('.o', '.cmi', '.cmx')) else t)
    f.close()
    return f.name


def build(cfg):
    tgz, d = pack(), cfg['cluster']['dir']
    def one(i, h):
        if subprocess.run(['scp', '-q', tgz, f'{h}:/tmp/route-src.tgz']).returncode:
            return {'host': h, 'built': False, 'errors': ['scp failed']}
        r = ssh(h, f'mkdir -p {d} && cd {d} && rm -rf rdma && tar -xzf /tmp/route-src.tgz && cd rdma/tests/route && ./build.sh 2>&1 | grep -E " error|error:" -A3 | head -20; test -f libroute.dylib')
        return {'host': h, 'built': r.returncode == 0, 'errors': [l for l in r.stdout.splitlines() if l.strip()]}
    for rec in each(cfg['cluster']['hosts'], one):
        print(json.dumps({'record': 'build', **rec}))
    os.unlink(tgz)


def port(cfg):
    py = cfg['cluster']['python']
    r = ssh(cfg['cluster']['hosts'][0], f"{py} -c 'import socket; s = socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])'")
    return int(r.stdout.strip())


def rank_cmd(cfg, i, what, args, detach=None):
    c = cfg['cluster']; world = len(c['hosts'])
    sets = ' '.join(f'--set {shlex.quote(s)}' for s in [f'run.rank={i}', f'run.world={world}'] + args)
    run = f"{c['python']} -W ignore -m route {what} {sets}"
    return f"cd {c['dir']}/rdma/tests/route/py && " + (f"nohup {run} > {detach} 2>&1 < /dev/null &" if detach else run)


def launch(cfg, what, sets, name=None):
    mesh = f"{cfg['cluster']['root']}:{port(cfg)}"
    def one(i, h):
        log = f"{cfg['cluster']['dir']}/{name}-{i}.jsonl" if name else None
        return ssh(h, rank_cmd(cfg, i, what, [f'run.mesh_id={mesh}'] + sets, log)).stdout
    return mesh, each(cfg['cluster']['hosts'], one)


def main(cfg, argv, sets):
    """sets: the command line's --set overrides, passed on to every rank"""
    cmd, plain = argv[0], argv[1:]
    if cmd == 'build':
        build(cfg)
    elif cmd == 'xbench':
        _, out = launch(cfg, f'xbench {plain[0]}', sets)
        for o in out:
            sys.stdout.write(o)
    elif cmd == 'train':
        name = cfg['run']['name']
        _, out = launch(cfg, 'xbench 0', [])   # every rank declares and closes: a dead run's programs flushed
        for o in out:
            sys.stdout.write(o)
        mesh, _ = launch(cfg, 'train', sets, name)
        print(json.dumps({'record': 'launched', 'name': name, 'mesh_id': mesh, 'logs': [f"{h}:{cfg['cluster']['dir']}/{name}-{i}.jsonl" for i, h in enumerate(cfg['cluster']['hosts'])]}))
    elif cmd == 'stat':
        for o in each(cfg['cluster']['hosts'], lambda i, h: ssh(h, f"cd {cfg['cluster']['dir']}/rdma/tests/route/py && {cfg['cluster']['python']} -W ignore -m route stat /tmp/route-telemetry-{i}").stdout):
            sys.stdout.write(o)
    elif cmd == 'logs':
        def last(i, h):
            r = ssh(h, f"tail -n 400 {cfg['cluster']['dir']}/{plain[0]}-{i}.jsonl")
            kinds = {}
            for line in r.stdout.splitlines():
                try:
                    rec = json.loads(line); kinds[rec.get('record')] = rec
                except ValueError:
                    kinds.setdefault('text', []).append(line)
            return kinds
        for i, k in enumerate(each(cfg['cluster']['hosts'], last)):
            print(json.dumps({'rank': i, **k}))
