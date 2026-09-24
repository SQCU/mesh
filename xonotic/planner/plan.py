import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rdma'))
from mesh import AllReduce

TEAMS = 5
EXPERTS = 8
FF = 2048
SEED = 20260828
SPLIT = Path(__file__).with_name('split.json')


# Megatron-LM tensor parallelism (Shoeybi et al. 2019) over the planner's routed experts: rank r
# holds its block of every expert's up-projection columns and the same rows of its down projection,
# so its expert output is a partial sum that one all-reduce completes.  Router and objective basis
# are small and replicated.
def model(width, columns, rank):
    rng = np.random.default_rng(SEED)
    route, up, down, objective = (rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-2])
        for shape in ((width, EXPERTS), (EXPERTS, width, FF), (EXPERTS, FF, width), (width, TEAMS)))
    share = slice(sum(columns[:rank]), sum(columns[:rank + 1]))
    return route, np.ascontiguousarray(up[:, :, share]), np.ascontiguousarray(down[:, share]), objective


# Route each bot to an expert and apply this rank's share of it, grouping rows per expert rather
# than gathering a weight matrix per row.
def solve(positions, route, up, down, out):
    selected = np.argmax(positions @ route, axis=1)
    out[:] = 0
    for expert in range(EXPERTS):
        rows = np.nonzero(selected == expert)[0]
        if rows.size:
            out[rows] = np.maximum(positions[rows] @ up[expert], 0) @ down[expert]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticks', type=int, nargs='?', default=2000)
    parser.add_argument('bots', type=int, nargs='?', default=480)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--columns', default=','.join(map(str, json.loads(SPLIT.read_text())['columns'])),
                        help="each rank's expert columns, in rank order (default: split.json)")
    parser.add_argument('--links', default=str(ROOT / 'examples/links-pair.conf'))
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=150)
    args = parser.parse_args()
    columns = [int(value) for value in args.columns.split(',')]
    if sum(columns) != FF:
        parser.error(f'--columns must sum to {FF}')
    deadline = time.monotonic() + args.seconds
    reduce = AllReduce((args.bots, args.width), np.float32, args.links, args.ticks, args.depth)
    route, up, down, objective = model(args.width, columns, reduce.rank)
    positions = np.random.default_rng(7).standard_normal((args.bots, args.width)).astype(np.float32) * .1
    positions[:, 0] = np.arange(args.bots)
    objectives = np.full(args.bots, -1)
    switches = collective = solving = 0
    print(json.dumps({'event': 'planner_started', 'rank': reduce.rank, 'bots': args.bots, 'width': args.width,
                      'experts': EXPERTS, 'columns': columns[reduce.rank], 'steps': len(reduce.steps)}), flush=True)
    start = time.monotonic()
    for tick in range(args.ticks):
        began = time.monotonic()
        solve(positions, route, up, down, reduce.slot(tick))
        solving += time.monotonic() - began
        began = time.monotonic()
        experts = reduce(tick, deadline)
        collective += time.monotonic() - began
        picks = np.argmax(experts @ objective, axis=1)
        switches += int(np.count_nonzero(objectives != picks))
        objectives = picks
        positions[:, 1:] += .35 * objective[1:, picks].T
        positions[:, 1:] *= .98
    elapsed = time.monotonic() - start
    reduce.close()
    print(json.dumps({'event': 'planner_finished', 'rank': reduce.rank, 'ticks': args.ticks,
                      'bot_plans': args.ticks * args.bots, 'seconds': elapsed, 'tick_ms': 1e3 * elapsed / args.ticks,
                      'solve_ms': 1e3 * solving / args.ticks, 'collective_ms': 1e3 * collective / args.ticks,
                      'switches': switches,
                      'objective_split': np.bincount(objectives, minlength=TEAMS).tolist()}), flush=True)


if __name__ == '__main__':
    main()
