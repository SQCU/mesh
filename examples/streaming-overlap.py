import argparse
import ctypes as C
import json
import os
import signal
import time

import numpy as np
from mesh import BlockSpec, Program, Submission


# design/algorithm-sources.md#streaming-overlap-measurement
def moments(values):
    mean = m2 = 0.0
    for count, value in enumerate(values, 1):
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    return dict(count=len(values), mean=mean if values else None,
                variance=m2 / (len(values) - 1) if len(values) > 1 else None)


# design/algorithm-sources.md#streaming-overlap-measurement
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int)
    parser.add_argument('mode', choices=('whole', 'streamed'))
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--tile', type=int, default=64)
    parser.add_argument('--trials', type=int, default=30)
    args = parser.parse_args()
    running = True

    # design/algorithm-sources.md#streaming-overlap-measurement
    def stop(*unused):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    print(json.dumps(dict(pid=os.getpid(), rank=args.rank, mode=args.mode)), flush=True)
    with Program(backend='cpu') as program:
        rows, tile, k, n, m = args.rows, args.tile, 256, 256, 256
        x = program.tensor((rows, k))
        w = program.tensor((k, n))
        p = program.tensor((rows, n))
        received = program.tensor((rows, n))
        v = program.tensor((n, m))
        y = program.tensor((rows, m))
        returned = program.tensor((rows, m))
        rng = np.random.default_rng(1701)
        weights = rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)
        projection = rng.standard_normal((n, m), dtype=np.float32) / np.sqrt(n)
        data = rng.standard_normal((rows, k), dtype=np.float32)
        program.constant(w[0, 0], weights)
        program.constant(v[0, 0], projection)
        program.copy(p.on(0), received.on(1), queue=0)
        program.copy(y.on(1), returned.on(0), queue=1)
        result = program.export(returned[0, 0]) if args.rank == 0 else None
        whole_input = program.export(received[0, 0]) if args.rank == 1 else None
        early = calls = 0
        count = (rows + tile - 1) // tile
        grid = (count,) if args.mode == 'streamed' else (1,)

        # design/algorithm-sources.md#streaming-overlap-measurement
        def region(ref, i):
            return ref.slice(i * tile, 0, min(tile, rows - i * tile), ref.shape[1]) if args.mode == 'streamed' else ref

        # design/algorithm-sources.md#streaming-overlap-measurement
        def index(i):
            return (0, 0)

        # design/algorithm-sources.md#streaming-overlap-measurement
        def prepare(coordinate, inputs, outputs):
            left, right, destination = inputs[0].array, inputs[1].array, outputs[0].array
            operands = tuple((left[i:i+tile], destination[i:i+tile]) for i in range(0, len(left), tile))

            # design/algorithm-sources.md#streaming-overlap-measurement
            def submit(binding, complete, context):
                nonlocal early, calls
                if whole_input is not None:
                    early += int(not whole_input.ready)
                for a, out in operands:
                    np.matmul(a, right, out=out)
                calls += 1
                complete(context, 0)
            return Submission(submit), None

        source, weight, output = (x, w, p) if args.rank == 0 else (received, v, y)
        program.call_native(prepare, grid=grid,
            inputs=[BlockSpec(source, index, region), BlockSpec(weight, index)],
            outputs=[BlockSpec(output, index, region)])
        program.realize()
        if args.rank == 1:
            while running:
                program.scan()
                if whole_input.ready and program.report.completed % grid[0] == 0:
                    whole_input.consume()
            print(json.dumps(dict(rank=1, mode=args.mode, calls=calls,
                consumers_before_full_receive=early)), flush=True)
            return
        samples = []
        maximum_error = 0.0
        reference = (data @ weights) @ projection
        for trial in range(args.trials + 5):
            with program.write(x[0, 0]) as buffer:
                buffer[...] = data
                began = time.perf_counter()
            while not result.ready:
                program.scan()
            elapsed = time.perf_counter() - began
            maximum_error = max(maximum_error, float(np.max(np.abs(result.array - reference))))
            if not np.allclose(result.array, reference, rtol=2e-4, atol=2e-4):
                raise ArithmeticError('Distributed contraction differs from the reference')
            result.consume()
            if trial >= 5:
                samples.append(elapsed)
        print(json.dumps(dict(rank=0, mode=args.mode, rows=rows, tile=tile,
            seconds=samples, statistics=moments(samples), max_absolute_error=maximum_error,
            numerical_calls_per_trial=2*count)), flush=True)


if __name__ == '__main__':
    main()
