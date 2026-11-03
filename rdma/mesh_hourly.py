"""The hourly (bin/mesh-hourly-agent.sh runs it on every node at etc/mesh-nodes.json's hourly_minute): the git state
this node holds, exactly (~/mesh and ~/metal-microbench as they are checked out, built), each program metal-microbench's
tools/mesh/hourly/workloads.json lists, run as anyone runs it, one rank on each member of this node's linked part of the
fabric (its bridge's map, mesh-stat). A node etc/mesh-nodes.json names a workstation is no member and runs nothing.

Its report, REPORT on each node: one JSON line per workload per run (the run's start, this node, both checkouts'
commits, the workload, its nodes in rank order and this node's rank, its exit status, how it ended, its seconds, the
records it measured (mesh_statistics.emit's lines) and, on a failure, its error output's tail), and one line a run
(`stage`: run) with its preparation, pairing, nodes and absent members. Each program's whole output goes to this job's
log, and its last run's to ~/.cache/mesh/hourly/<workload>.out and .err. A stage's failure or a program's crash is
recorded and the run goes on to the next. metal-microbench tools/mesh/hourly/ledger.py gathers every member's report.

Workload k is ended at DEADLINE: the run's start + PREPARE_S + the slots of workloads 0..k (SLOT_S, or its `timeout`),
the same instant on every node, so a rank that ends early waits at the next program's rendezvous for its peers and no
rank is ended before a late peer arrives (a bound from each node's own start let one hung rank's peers move on and
miss every later rendezvous). A run is bounded by PREPARE_S plus its slots, under the hour.

It moves no checkout. A fetch and checkout of main here (d830075c, 10-30) rewrote every node's git state at the hour,
under any run on it, and its records named no commit: the operator, 2026-10-31, "run what is on the current machines at
current main head", not "some other arbitrary thing to tamper with records". Keeping the nodes at main is the job of
whoever moves them (metal-microbench tools/mesh/demonstrate.py --deploy)."""
import json, os, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
home = Path.home()
sys.path.insert(0, str(home / 'mesh/rdma'))
import mesh_nodes
REPORT, OUTPUT, PREPARE_S, SLOT_S = home / '.cache/mesh/hourly.jsonl', home / '.cache/mesh/hourly', 120, 120
began = time.time()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a stop ends the running program's group too (finally)
cluster = mesh_nodes.load()
members = {n for n, m in cluster.members.items() if 'workstation' not in m.roles}
run = {'schema': 1, 'run': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(began)),
       'commits': {repo: subprocess.run(['git', '-C', home / repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
                   for repo in ('mesh', 'metal-microbench')}}


def bridge():
    """This node's bridge's report (mesh-stat): its node number and its map; {} where it does not answer."""
    report = mesh_nodes.Node(cluster, '-', 'localhost', local=True).stat()[1]
    report = report if isinstance(report, dict) else {}
    run['number'] = report.get('node')
    run['node'] = cluster.members[str(run['number'])].name if str(run['number']) in cluster.members else socket.gethostname()
    return report


def record(**row):
    line = json.dumps({**run, **row}, separators=(',', ':'))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def failed():
    return traceback.format_exc()[-4000:]


if str(bridge().get('node')) in cluster.members.keys() - members:
    print(json.dumps({'hourly': f'{run["node"]} is a workstation (etc/mesh-nodes.json): it runs no hourly'}), flush=True)
    sys.exit(0)
stage = {'stage': 'run'}
try:
    done = subprocess.run([sys.executable, '-m', 'mesh_nodes', 'node', 'prepare'], cwd=home / 'mesh/rdma',
                          capture_output=True, text=True, timeout=1800)
    print(done.stdout + done.stderr, flush=True)
    stage['prepare'] = {'exit': done.returncode, 'seconds': round(time.time() - began, 2),
                        'said': mesh_nodes.loads((done.stdout.strip().splitlines() or [''])[-1]) or done.stdout[-400:],
                        **({'error': (done.stderr or done.stdout)[-4000:]} if done.returncode else {})}
except Exception:
    stage['prepare'] = {'exit': None, 'seconds': round(time.time() - began, 2), 'error': failed()}
nodes, absent = [], {}
try:
    # the node's number from its own bridge's report: a local Node numbered '-' failed ports()'s int(n.number), and the
    # hourly ended there before its first program (94 runs in m5ult4's log to 2026-11-03)
    number = str(bridge().get('node'))
    if run['number'] is None:
        raise RuntimeError("this node's bridge did not answer (mesh-stat)")
    stage['paired'] = mesh_nodes.wait_paired([mesh_nodes.Node(cluster, number, 'localhost', local=True)], None, time.time() + 60)
    held = bridge().get('map', {})
    linked = next(part for part in mesh_nodes.parts({str(x) for x in held['nodes']} | {number},
                                                    [tuple(map(str, link[:2])) for link in held['links']]) if number in part)
    nodes = sorted(linked & members, key=int)
    absent = {cluster.members[n].name: 'in its bridge\'s map, not linked to this node' if int(n) in held['nodes'] else
              'not in its bridge\'s map' for n in sorted(members - linked, key=int)}
except Exception:
    stage['error'] = failed()
# alone where its bridge did not answer or it is no member: its programs then say why they cannot run
nodes = nodes if run['number'] is not None and str(run['number']) in nodes else [str(run['number'])]
names, rank = [cluster.members[n].name if n in cluster.members else n for n in nodes], nodes.index(str(run['number']))
record(**stage, nodes=names, absent=absent)
try:
    workloads = json.loads((home / 'metal-microbench/tools/mesh/hourly/workloads.json').read_text())['workloads']
except Exception:
    workloads = []
    record(stage='workloads', error=failed())
OUTPUT.mkdir(parents=True, exist_ok=True)
deadline = began + PREPARE_S
for ordinal, item in enumerate(workloads):
    deadline += item.get('timeout', SLOT_S)
    started, ended, program = time.time(), 'exited', None
    try:
        # the rendezvous over the tailnet (each member's management address): a fabric address is a host route over
        # one Thunderbolt link, and one stopped port left the other ranks finding rank 0 unreachable (2026-11-01 17:06)
        env = {**os.environ, 'RANK': str(rank), 'WORLD_SIZE': str(len(nodes)), 'MASTER_PORT': str(cluster.rank_port + ordinal),
               'MASTER_ADDR': cluster.members[nodes[0]].management if nodes[0] in cluster.members else '127.0.0.1',
               'PYTHONPATH': f'{home}/mesh/rdma:{home}/mesh/rdma/torch-mesh'}
        out, err = OUTPUT / f"{item['tag']}.out", OUTPUT / f"{item['tag']}.err"
        print(json.dumps({'hourly': item['tag'], 'nodes': names, 'rank': rank, 'deadline': round(deadline)}), flush=True)
        with out.open('w') as o, err.open('w') as e:
            program = subprocess.Popen([sys.executable, home / 'mesh/examples' / f"{item['program']}.py",
                                        *([home / 'metal-microbench' / item['configuration']] if 'configuration' in item else [])],
                                       env=env, cwd=home, stdout=o, stderr=e, stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                program.wait(timeout=max(deadline - time.time(), 10))
            except subprocess.TimeoutExpired:
                ended = 'at its deadline'
            finally:
                # the program's whole group, its own descendants included, so none holds a port or a lease into the next
                for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, None)):
                    try:
                        os.killpg(program.pid, sig)
                        program.wait(timeout=grace)
                        break
                    except (ProcessLookupError, PermissionError):
                        break
                    except subprocess.TimeoutExpired:
                        pass
        stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
        print(stdout + stderr, flush=True)
        measured = [v for v in map(mesh_nodes.loads, stdout.splitlines()) if isinstance(v, dict) and v and 'phase' not in v]
        failure = {'error': (stderr.strip() or stdout.strip())[-4000:]} if program.returncode or ended != 'exited' else {}
    except Exception:
        measured, failure = [], {'error': failed()}
    record(workload=item['tag'], program=item.get('program'), configuration=item.get('configuration'), nodes=names,
           rank=rank, exit=program and program.returncode, ended=ended, started=round(started, 3),
           seconds=round(time.time() - started, 2), measured=measured, **failure)
