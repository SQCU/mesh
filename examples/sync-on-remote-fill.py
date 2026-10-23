import argparse
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import numpy as np  # noqa: E402
import kernels  # noqa: E402
from mesh import BlockSpec, Program, ShapeDtypeStruct  # noqa: E402
from mesh.collective import send, sync_on_remote_fill, sync_on_some_remote_fill  # noqa: E402

STOP = (signal.SIGTERM, signal.SIGINT)


class Stopped(Exception):
    pass


# design/algorithm-sources.md#collectivesync_on_remote_fill
def stop(*_):
    raise Stopped


# design/algorithm-sources.md#collectivesync_on_remote_fill
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('stream', 'sync', 'independent'), default='stream')
    parser.add_argument('--root', type=int, default=0)
    parser.add_argument('--peer', type=int, default=1)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--elements', type=int, default=16384)
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    args = parser.parse_args()
    if min(args.blocks, args.rounds, args.elements) < 1 or args.root == args.peer:
        parser.error('Use positive dimensions and two distinct participants')
    for number in STOP:
        signal.signal(number, stop)
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP)
    with Program(backend=args.backend) as program:
        if program.size != 2:
            parser.error('two ranks (RANK, WORLD_SIZE=2, MASTER_ADDR, MASTER_PORT)')
        shape = (1, args.elements)
        count = 1 + args.blocks * args.rounds
        source = program.tensor((count, args.elements), shape)
        received = program.tensor(source.shape, shape)
        replies = program.tensor(source.shape, shape)
        one = program.tensor(shape)
        program.constant(one[0, 0], np.ones(shape, np.float32))
        spec = BlockSpec(shape, lambda i, j: (i, j))
        transformed = program.kernel_call(kernels.add, grid=source.grid,
            in_specs=(spec, BlockSpec(shape, lambda i, j: (0, 0))), out_specs=spec,
            out_shape=ShapeDtypeStruct(source.shape, source.dtype), peer=args.peer)(received, one)
        for index in range(count):
            send(program, source[index, 0].on(args.root), received[index, 0].on(args.peer), queue=2 if index == 0 else 0)
            send(program, transformed[index, 0].on(args.peer), replies[index, 0].on(args.root), queue=3 if index == 0 else 1)
        outputs = [program.export(replies[i, 0]) for i in range(count)] if program.rank == args.root else []
        program.realize()
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP)
        print(f'participant={program.rank} pid={os.getpid()} mode={args.mode}', flush=True)
        try:
            run(args, program, source, outputs, count)
        except Stopped:
            pass


# design/algorithm-sources.md#collectivesync_on_remote_fill
def run(args, program, source, outputs, count):
    if program.rank == args.peer:
        while True:
            signal.pause()
    if args.mode == 'independent':
        for index in range(1, count):
            with program.write(source[index, 0]) as target:
                target[...] = index
        pending = {output: index for index, output in enumerate(outputs) if index}
        while pending:
            for output in sync_on_some_remote_fill(*pending):
                print(f'published={pending.pop(output)} retained={output.array[0, :4].tolist()}', flush=True)
        print(f'unpublished_source=0 reply_ready={outputs[0].ready} retained_results={count - 1}', flush=True)
        return
    with program.write(source[0, 0]) as target:
        target[...] = 0
    sync_on_remote_fill(outputs[0])
    outputs[0].consume()
    mean = moment = 0.0
    for trial in range(args.rounds):
        first = 1 + trial * args.blocks
        pending = outputs[first:first + args.blocks]
        start = perf_counter()
        for offset, output in enumerate(pending):
            with program.write(source[first + offset, 0]) as target:
                target[...] = first + offset
            if args.mode == 'sync':
                sync_on_remote_fill(output)
        sync_on_remote_fill(*pending)
        elapsed = perf_counter() - start
        for output in pending:
            output.consume()
        delta = elapsed - mean
        mean += delta / (trial + 1)
        moment += delta * (elapsed - mean)
        variance = moment / trial if trial else 0.0
        print(f'mode={args.mode} trial={trial + 1} seconds={elapsed:.9f} '
              f'n={trial + 1} mean_seconds={mean:.9f} sample_variance={variance:.12g}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
