"""The hourly (bin/mesh-hourly-agent.sh runs it on every node at etc/mesh-nodes.json's hourly_minute): the git state
this node holds, exactly (~/mesh and ~/metal-microbench as they are checked out, built), each program metal-microbench's
tools/mesh/hourly/workloads.json lists, run as anyone runs it, one rank on each node this node's bridge links. Each
program's output goes to this job's log, after a line naming both checkouts' commits.

It moves no checkout. A fetch and checkout of main here (d830075c, 10-30) rewrote every node's git state at the hour,
under any run on it, and its records named no commit: the operator, 2026-10-31, "run what is on the current machines at
current main head", not "some other arbitrary thing to tamper with records". Keeping the nodes at main is the job of
whoever moves them."""
import json, os, subprocess, sys, time
from pathlib import Path
home = Path.home()
commits = {repo: subprocess.run(['git', '-C', home / repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
           for repo in ('mesh', 'metal-microbench')}
subprocess.run([sys.executable, '-m', 'mesh_nodes', 'node', 'prepare'], cwd=home / 'mesh/rdma', timeout=1800)
sys.path.insert(0, str(home / 'mesh/rdma'))
import mesh, mesh_nodes
cluster = mesh_nodes.load()
mesh_nodes.wait_paired([mesh_nodes.Node(cluster, '-', 'localhost', local=True)], None, time.time() + 300)
here = mesh.current()
nodes = sorted(map(str, here.nodes), key=int)
# the rendezvous over the tailnet (each member's management address): a fabric address is a host route over one
# Thunderbolt link, and one stopped port left the other ranks finding rank 0 unreachable (2026-11-01 17:06)
for ordinal, item in enumerate(json.loads((home / 'metal-microbench/tools/mesh/hourly/workloads.json').read_text())['workloads']):
    env = {**os.environ, 'RANK': str(nodes.index(str(here.node))), 'WORLD_SIZE': str(len(nodes)), 'MASTER_ADDR': cluster.members[nodes[0]].management,
           'MASTER_PORT': str(cluster.rank_port + ordinal), 'PYTHONPATH': f'{home}/mesh/rdma:{home}/mesh/rdma/torch-mesh'}
    print(json.dumps({'hourly': item['tag'], 'nodes': nodes, 'commits': commits, 'unix': time.time()}), flush=True)
    program = subprocess.Popen([sys.executable, home / 'mesh/examples' / f"{item['program']}.py",
                                *([home / 'metal-microbench' / item['configuration']] if 'configuration' in item else [])], env=env, cwd=home)
    try:
        program.wait(timeout=item.get('timeout', 300))
    except subprocess.TimeoutExpired:
        program.terminate()
        try:
            program.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    print(json.dumps({'hourly': item['tag'], 'exit': program.returncode, 'unix': time.time()}), flush=True)
