import argparse
import os
import signal
import sys
from time import perf_counter

import numpy as np

from mesh import BlockSpec, Program, ShapeDtypeStruct, kernels
from mesh.collective import send, sync_on_remote_fill


# design/algorithm-sources.md#collectivesync_on_remote_fill
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('stream', 'sync', 'deadlock', 'independent'), default='stream')
    parser.add_argument('--root', type=int, required=True)
    parser.add_argument('--peer', type=int, required=True)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--elements', type=int, default=16384)
    args = parser.parse_args()
    if min(args.blocks, args.rounds, args.elements) < 1 or args.root == args.peer:
        parser.error('Use positive dimensions and two distinct participants')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with Program(backend='cpu') as program:
        shape = (1, args.elements)
        count = 1 + args.blocks * args.rounds
        source = program.tensor((count, args.elements), shape)
        received = program.tensor(source.shape, shape)
        replies = program.tensor(source.shape, shape)
        one = program.tensor(shape)
        program.constant(one[0, 0], 1)
        spec = BlockSpec(shape, lambda i, j: (i, j))
        transformed = program.kernel_call(kernels.add, grid=source.grid,
            in_specs=(spec, BlockSpec(shape, lambda i, j: (0, 0))), out_specs=spec,
            out_shape=ShapeDtypeStruct(source.shape, source.dtype), peer=args.peer)(received, one)
        send(program, source.on(args.root), received.on(args.peer))
        send(program, transformed.on(args.peer), replies.on(args.root))
        outputs = [program.export(replies[i, 0]) for i in range(count)] if program.node == args.root else []
        program.realize()
        print(f'participant={program.node} pid={os.getpid()} mode={args.mode}', flush=True)
        if program.node == args.peer:
            signal.pause()
            return
        if args.mode == 'independent':
            for index in range(1, count):
                with program.write(source[index, 0]) as target:
                    target[...] = index
            pending = set(range(1, count))
            while pending:
                for index in tuple(pending):
                    if outputs[index].ready:
                        print(f'published={index} retained={outputs[index].array[0, :4].tolist()}', flush=True)
                        pending.remove(index)
                if program.report.code:
                    raise RuntimeError(f'Mesh execution error: {program.report.code}')
            print(f'unpublished_source=0 reply_ready={outputs[0].ready} retained_results={count - 1}', flush=True)
            return
        if args.mode == 'deadlock':
            print('WAIT reply[0] -> peer add -> source[0] -> write after this wait', flush=True)
            sync_on_remote_fill(outputs[0])
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
            for offset, output in enumerate(pending):
                if not np.all(output.array == first + offset + 1):
                    raise ArithmeticError('Remote addition result differs')
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
