import argparse
from pathlib import Path
import sys
import time

import numpy as np

from mesh import Program

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.strat import tensor as mx
from solver.strat.tensor_metal import kernel_calls

TEAMS = 5
EXPERTS = 8
FF = 2048
SEED = 20260828


# ../../design/algorithm-sources.md#xonotic-planner-migration
def model(width):
    rng = np.random.default_rng(SEED)
    return tuple(rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-2])
                 for shape in ((width, EXPERTS), (EXPERTS, width, FF), (EXPERTS, FF, width), (width, TEAMS)))


# ../../design/algorithm-sources.md#xonotic-planner-migration
def solve(program, source, weights):
    graph = mx.Graph()
    with graph:
        x = graph.input('position', source.shape)
        r, w1, w2, o = (graph.input(name, shape) for name, shape in
            zip(('route', 'up', 'down', 'objective'),
                ((source.shape[1], EXPERTS), (EXPERTS, source.shape[1], FF),
                 (EXPERTS, FF, source.shape[1]), (source.shape[1], TEAMS))))
        logits = mx.matmul(x, r)
        selected = mx.min(mx.where(logits == mx.max(logits, axis=1, keepdims=True), mx.arange(EXPERTS), EXPERTS), axis=1)
        hidden = mx.maximum(mx.expert_matmul(x, w1, selected), 0)
        y = mx.matmul(mx.expert_matmul(hidden, w2, selected), o)
    inputs = {value.index: tensor for value, tensor in zip((x, r, w1, w2, o), (source, *weights))}
    return kernel_calls(program, graph, (), inputs)[y.index]


# ../../design/algorithm-sources.md#xonotic-planner-migration
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=('play', 'solve'))
    parser.add_argument('peer', type=int)
    parser.add_argument('seconds', type=float, nargs='?', default=15)
    parser.add_argument('bots', type=int, nargs='?', default=480)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--tile-rows', type=int, default=64)
    args = parser.parse_args()
    data = model(args.width)
    with Program(backend='metal') as program:
        player, solver = (program.node, args.peer) if args.role == 'play' else (args.peer, program.node)
        weights = []
        if args.role == 'solve':
            for value in data:
                tensor = program.tensor((int(np.prod(value.shape[:-1])), value.shape[-1]))
                program.constant(tensor[0, 0], value.reshape(tensor.shape))
                weights.append(tensor)
        sources, returned = [], []
        for first in range(0, args.bots, args.tile_rows):
            rows = min(args.tile_rows, args.bots - first)
            tx, rx = (program.tensor((rows, args.width)) for _ in range(2))
            program.copy(tx.on(player), rx.on(solver))
            y = solve(program, rx, weights) if args.role == 'solve' else program.tensor((rows, TEAMS))
            result = program.tensor((rows, TEAMS))
            program.copy(y.on(solver), result.on(player))
            sources.append((first, tx[0, 0]))
            if args.role == 'play':
                returned.append((first, program.export(result[0, 0])))
        program.realize()
        positions = np.random.default_rng(7).standard_normal((args.bots, args.width)).astype(np.float32) * .1
        positions[:, 0] = np.arange(args.bots)
        objectives = np.full(args.bots, -1, np.int32)
        planned = switches = 0
        start = time.monotonic()
        print(f'{args.role}: D={args.width} bots={args.bots} teams={TEAMS} experts={EXPERTS}', flush=True)
        while time.monotonic() - start < args.seconds:
            for first, result in returned:
                if not result.ready:
                    continue
                picks = np.argmax(result.array, axis=1)
                last = first + len(picks)
                switches += int(np.count_nonzero(objectives[first:last] != picks))
                objectives[first:last] = picks
                positions[first:last, 1:] += .35 * data[3][1:, picks].T
                positions[first:last, 1:] *= .98
                planned += len(picks)
                result.consume()
            if args.role == 'play':
                for first, ref in sources:
                    try:
                        with program.write(ref) as output:
                            output[:] = positions[first:first + ref.shape[0]]
                    except BlockingIOError:
                        pass
        elapsed = time.monotonic() - start
        print(f'{args.role}: {planned} returned plans in {elapsed:.3f}s; switches={switches}; '
              f'objective split={np.bincount(objectives[objectives >= 0], minlength=TEAMS).tolist()}', flush=True)


if __name__ == '__main__':
    main()
