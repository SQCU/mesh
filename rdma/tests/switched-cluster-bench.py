"""The compiled routing against the switched bridges on the whole cluster (design/switched-multipath-dma.md "Deploy by
git, then measure"): the all-reduce and the all-gather timed at each size (tests/collectives-bench.c), microseconds a
call and GB/s, under each branch in turn, every node converged to it by the repo's own deployment (mesh_nodes.install:
each node's install of the branch's newest commit, its bridge restarted by the install), its links awaited PAIRED.

- compiled (OLD, default switched-dma-20261119): the declaration-time route tables; the Mac mini a relay-only rank
  (ncclMeshCall contributors and recipients), so its cables carry routes;
- switched (NEW, default this worktree's branch): the bridges switch every piece; four compute ranks, the Mac mini's
  bridge relaying by name with no client.

Rank 0 is the bootstrap root's node (10.0.0.14, Studio-3), then A (m5ult2), B (Studio), C (Studio-2), the mini last.
It takes the shared cluster lease (mesh_nodes.exclusive_lock) before it installs anything and holds it to the end, and
reinstalls RESTORE (default OLD) at the end.  Results: OUT/results.json and a table on stdout.

usage: uv run python rdma/tests/switched-cluster-bench.py OUT [--old BRANCH] [--new BRANCH] [--restore BRANCH]
       [--sizes BYTES,...]   (run from the repo root, on the machine whose /Users/mdot/dox/mesh holds the branches) """
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mesh_nodes as nodes_  # noqa: E402

ROOT_ADDRESS = '10.0.0.14'
SIZES = [6144, 24576, 98304, 393216, 1572864, 3145728, 12582912]


ROOT_NUMBER = 371057962   # Studio-3, the bootstrap root's node (reached at ROOT_ADDRESS on the mesh)


def ordered(found):
    """rank 0 the bootstrap root's node (Studio-3 by its node number, whatever address discovery reports; else the
    lowest number), the mini last, the rest by node number. Always an order: every discovered node takes part."""
    by = sorted(found, key=lambda n: n.number)
    root = [n for n in by if n.number == ROOT_NUMBER or n.addr == ROOT_ADDRESS or ROOT_ADDRESS in (n.ula or '')][:1] or by[:1]
    mini = [n for n in by if n.number == nodes_.MINI and n not in root]
    return root + [n for n in by if n not in root and n not in mini] + mini


def phase(found, out, tag, branch, sizes, relay_mini, port):
    sha = nodes_.resolve(branch)
    log = out / f'{tag}.log'
    nodes_.note(log, f'{tag}: branch {branch} at {sha}')
    nodes_.install(found, sha, out / tag)
    if not nodes_.wait_paired(found, log):
        raise SystemExit(f'{tag}: the links did not pair after the install of {branch}; see {log}')
    # the benchmark built on every node against its installed libnccl-mesh
    for n in found:
        src = nodes_.home_path(n.source + '/rdma')
        code, text = n.run(f'cd {src} && mkdir -p .build/tests && cc -O2 -Wall -Wextra -Werror -I. -o .build/tests/collectives-bench '
                           f'tests/collectives-bench.c -L.build -lnccl-mesh -Wl,-rpath,"$PWD/.build"')
        if code:
            raise SystemExit(f'{tag}: {n.name}: collectives-bench does not build: {text}')
    ranks = found if relay_mini else found[:-1]
    relay = len(ranks) - 1 if relay_mini else -1
    folder = f'mesh-runs/switched-bench-{tag}'
    for n in ranks:
        n.run(f'mkdir -p {nodes_.home_path(folder)}')
    bench = nodes_.home_path(ranks[0].source + '/rdma/.build/tests/collectives-bench')
    command = f'sh -c \'exec {bench} "$RANK" "$WORLD_SIZE" "$MASTER_ADDR:$MASTER_PORT" {relay} {" ".join(map(str, sizes))}\''
    nodes_.ranks(ranks, out, tag, folder, command, port, alarm=1200)
    rows = [json.loads(line) for line in (out / f'{tag}-rank0.log').read_text().splitlines() if line.startswith('{')]
    if not rows or not all(r['verified'] for r in rows):
        raise SystemExit(f'{tag}: rank 0 reported {len(rows)} rows, not all verified; see {out}/{tag}-rank*.log')
    return {'branch': branch, 'commit': sha, 'ranks': len(ranks), 'relay_rank': relay, 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--old', default='switched-dma-20261119')
    p.add_argument('--new', default='worktree-agent-a6f7829072d3f09f4')
    p.add_argument('--restore', default=None)
    p.add_argument('--sizes', default=','.join(map(str, SIZES)))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in a.sizes.split(',')]
    with nodes_.exclusive_lock('switched-cluster-bench'):
        found, _ = nodes_.discover(out / 'discover.log')
        if not found:
            raise SystemExit('no cluster discovered; see discover.log')
        found = ordered(nodes_.up(found, out / 'discover.log'))
        results = {'old': phase(found, out, 'compiled', a.old, sizes, True, 29711),
                   'new': phase(found, out, 'switched', a.new, sizes, False, 29723)}
        restore = a.restore or a.old
        nodes_.install(found, nodes_.resolve(restore), out / 'restore')
        nodes_.wait_paired(found, out / 'restore.log')
    (out / 'results.json').write_text(json.dumps(results, indent=1))
    print(f'{"collective":<11} {"bytes":>9} {"compiled us":>12} {"switched us":>12} {"compiled GB/s":>14} {"switched GB/s":>14}')
    old = {(r['what'], r['bytes']): r for r in results['old']['rows']}
    for r in results['new']['rows']:
        o = old.get((r['what'], r['bytes']))
        print(f'{r["what"]:<11} {r["bytes"]:>9} {o["us_per_call"] if o else float("nan"):>12.1f} {r["us_per_call"]:>12.1f} '
              f'{o["algorithm_gbps"] if o else float("nan"):>14.2f} {r["algorithm_gbps"]:>14.2f}')


if __name__ == '__main__':
    main()
