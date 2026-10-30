"""The hourly (bin/mesh-hourly-agent.sh runs it on every node at etc/mesh-nodes.json's hourly_minute): this node's ~/mesh
and ~/metal-microbench at main and built, then each program metal-microbench's tools/mesh/hourly/workloads.json lists,
run as anyone runs it, one rank on each node this node's bridge links. Each program's output goes to this job's log."""
import json, os, subprocess, sys, time
from pathlib import Path
home = Path.home()
for repo in ('mesh', 'metal-microbench'):
    subprocess.run(f'git -C {home / repo} fetch -q origin main && git -C {home / repo} checkout -q --detach FETCH_HEAD', shell=True, timeout=300)
subprocess.run([sys.executable, '-m', 'mesh_nodes', 'node', 'prepare'], cwd=home / 'mesh/rdma', timeout=1800)
sys.path.insert(0, str(home / 'mesh/rdma'))
import mesh, mesh_nodes
cluster = mesh_nodes.load()
mesh_nodes.wait_paired([mesh_nodes.Node(cluster, '-', 'localhost', local=True)], None, time.time() + 300)
here = mesh.current()
nodes = sorted(map(str, here.nodes), key=int)
for ordinal, item in enumerate(json.loads((home / 'metal-microbench/tools/mesh/hourly/workloads.json').read_text())['workloads']):
    env = {**os.environ, 'RANK': str(nodes.index(str(here.node))), 'WORLD_SIZE': str(len(nodes)), 'MASTER_ADDR': cluster.members[nodes[0]].fabric,
           'MASTER_PORT': str(cluster.rank_port + ordinal), 'PYTHONPATH': f'{home}/mesh/rdma:{home}/mesh/rdma/torch-mesh'}
    print(json.dumps({'hourly': item['tag'], 'nodes': nodes, 'unix': time.time()}), flush=True)
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
