import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rdma'))
from allocate import Allocator
from mesh import AllReduce

TEAMS = 5
EXPERTS = 8
FF = 2048
SEED = 20260828
SPLIT = Path(__file__).with_name('split.json')


# Megatron-LM tensor parallelism (Shoeybi et al. 2019) over the planner's routed experts: rank r
# applies its block of every expert's up-projection columns and the same rows of its down
# projection, so its expert output is a partial sum that one all-reduce completes.  Router and
# objective basis are small and replicated.  Every rank holds every column, so the blocks are a
# run-time operand: each tick slices them by the shares the allocator wrote on the tick before.
def model(width):
    rng = np.random.default_rng(SEED)
    return tuple(rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-2])
                 for shape in ((width, EXPERTS), (EXPERTS, width, FF), (EXPERTS, FF, width), (width, TEAMS)))


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
    split = json.loads(SPLIT.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('ticks', type=int, nargs='?', default=2000)
    parser.add_argument('bots', type=int, nargs='?', default=480)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--columns', default=','.join(map(str, split['columns'])),
                        help="config_t0: each rank's expert columns, in rank order (default: split.json)")
    parser.add_argument('--links', default=str(ROOT / 'examples/links-pair.conf'))
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=150)
    args = parser.parse_args()
    columns = [int(value) for value in args.columns.split(',')]
    if sum(columns) != FF:
        parser.error(f'--columns must sum to {FF}')
    if args.width < len(columns):
        parser.error('--width must hold one segment time per rank')
    deadline = time.monotonic() + args.seconds
    # The operand's last row carries each rank's segment time in its own column and zeros
    # elsewhere, so the reduction sums it exactly and every rank reads every rank's time from the
    # tick's result: the allocator (rdma/allocate.py) is duplicated on every rank and steps on
    # identical inputs, and its shares are the next tick's configuration operand.
    reduce = AllReduce((args.bots + 1, args.width), np.float32, args.links, args.ticks, args.depth)
    allocate = Allocator(columns, split['grain'], split['low'], split['prior'], split['floor'],
                         split['forget'], split['step'])
    route, up, down, objective = model(args.width)
    positions = np.random.default_rng(7).standard_normal((args.bots, args.width)).astype(np.float32) * .1
    positions[:, 0] = np.arange(args.bots)
    objectives = np.full(args.bots, -1)
    switches = collective = solving = 0
    print(json.dumps({'event': 'planner_started', 'rank': reduce.rank, 'bots': args.bots, 'width': args.width,
                      'experts': EXPERTS, 'columns': columns[reduce.rank], 'steps': len(reduce.steps)}), flush=True)
    start = ready = time.monotonic()
    for tick in range(args.ticks):
        began = time.monotonic()
        low = sum(allocate.shares[:reduce.rank])
        share = slice(low, low + allocate.shares[reduce.rank])
        slot = reduce.slot(tick)
        solve(positions, route, up[:, :, share], down[:, share], slot[:args.bots])
        solving += time.monotonic() - began
        # this rank's segment: from the last reduction's return to entering this one
        slot[args.bots] = 0
        slot[args.bots, reduce.rank] = 1e3 * (time.monotonic() - ready)
        began = time.monotonic()
        total = reduce(tick, deadline)
        ready = time.monotonic()
        collective += ready - began
        if tick >= split['warmup']:
            allocate([float(value) for value in total[args.bots, :len(columns)]])
        experts = total[:args.bots]
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
                      'objective_split': np.bincount(objectives, minlength=TEAMS).tolist(),
                      'shares': allocate.shares, 'b_ms_per_column': allocate.b}), flush=True)


if __name__ == '__main__':
    main()
