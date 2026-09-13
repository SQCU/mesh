import argparse
import json
import os
import signal
import time

import numpy as np
from mesh import BlockSpec, Program, ShapeDtypeStruct


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
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--tile', type=int, default=64)
    parser.add_argument('--trials', type=int, default=3000)
    parser.add_argument('--depth', type=int, default=3)
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
        w = program.tensor((k, n))
        v = program.tensor((n, m))
        rng = np.random.default_rng(1701)
        weights = rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)
        projection = rng.standard_normal((n, m), dtype=np.float32) / np.sqrt(n)
        data = rng.standard_normal((rows, k), dtype=np.float32)
        program.constant(w[0, 0], weights)
        program.constant(v[0, 0], projection)
        early = calls = 0
        count = (rows + tile - 1) // tile
        grid = (count,) if args.mode == 'streamed' else (1,)

        # design/algorithm-sources.md#pallas-call-ergonomics
        def configure():
            x = program.tensor((rows, k))
            whole_input = None

            # design/algorithm-sources.md#pallas-call-ergonomics
            def kernel(left, right, destination):
                nonlocal early, calls
                if whole_input is not None:
                    early += int(not whole_input.present)
                for i in range(0, len(left), tile):
                    np.matmul(left[i:i+tile], right, out=destination[i:i+tile])
                calls += 1

            # design/algorithm-sources.md#pallas-call-ergonomics
            def row_index(i):
                return (i, 0)

            # design/algorithm-sources.md#pallas-call-ergonomics
            def weight_index(i):
                return (0, 0)

            block_rows = tile if args.mode == 'streamed' else rows
            matmul = program.kernel_call(kernel, grid=grid,
                in_specs=(BlockSpec((block_rows, k), row_index),
                          BlockSpec((k, n), weight_index)),
                out_specs=BlockSpec((block_rows, n), row_index),
                out_shape=ShapeDtypeStruct((rows, n), np.float32))
            p = matmul(x, w) if args.rank == 0 else program.tensor((rows, n))
            received = program.tensor((rows, n))
            if args.rank == 1:
                whole_input = received[0, 0]
            y = matmul(received, v) if args.rank == 1 else program.tensor((rows, m))
            returned = program.tensor((rows, m))
            program.copy(p.on(0), received.on(1), queue=0)
            program.copy(y.on(1), returned.on(0), queue=1)
            result = program.export(returned[0, 0]) if args.rank == 0 else None
            return x[0, 0], result

        slots = [configure() for _ in range(args.depth)]
        program.realize()
        if args.rank == 1:
            while running:
                program.scan()
            print(json.dumps(dict(rank=1, mode=args.mode, calls=calls,
                consumers_before_full_receive=early)), flush=True)
            return
        samples = []
        maximum_error = 0.0
        reference = (data @ weights) @ projection
        latencies = []
        warmups = 5 * args.depth
        total = warmups + args.trials
        submitted = completed = 0
        outstanding = {}
        terminal = []
        measurement_start = None
        measurement_end = None
        while completed < total:
            for slot, (input_ref, result) in enumerate(slots):
                if slot not in outstanding and submitted < total:
                    with program.write(input_ref) as buffer:
                        np.multiply(data, 1 + (submitted % 17) / 32, out=buffer)
                        began = time.perf_counter()
                    outstanding[slot] = (submitted, began)
                    submitted += 1
            program.scan()
            for slot, (input_ref, result) in enumerate(slots):
                if slot not in outstanding or not result.ready:
                    continue
                trial, began = outstanding.pop(slot)
                ended = time.perf_counter()
                if submitted < total:
                    result.consume()
                else:
                    terminal.append((result, trial))
                completed += 1
                if completed == warmups:
                    measurement_start = ended
                if completed > warmups:
                    latencies.append(ended - began)
                    measurement_end = ended
        for result, trial in terminal:
            expected = reference * (1 + (trial % 17) / 32)
            maximum_error = max(maximum_error, float(np.max(np.abs(result.array - expected))))
            if not np.allclose(result.array, expected, rtol=2e-4, atol=2e-4):
                raise ArithmeticError('Distributed contraction differs from the reference')
            result.consume()
        samples = latencies
        print(json.dumps(dict(rank=0, mode=args.mode, rows=rows, tile=tile,
            depth=args.depth, seconds=samples, statistics=moments(samples),
            steady_seconds=measurement_end-measurement_start,
            results_per_second=args.trials/(measurement_end-measurement_start), max_absolute_error=maximum_error,
            numerical_calls_per_trial=2*count)), flush=True)


if __name__ == '__main__':
    main()
