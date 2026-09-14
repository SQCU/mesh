import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from mesh import BlockSpec, Program, ShapeDtypeStruct, kernels


# design/algorithm-sources.md#streaming-overlap-measurement
def moments(values):
    mean = m2 = 0.0
    for count, value in enumerate(values, 1):
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    return dict(count=len(values), mean=mean if values else None,
                sample_variance=m2 / (len(values) - 1) if len(values) > 1 else None)


# design/algorithm-sources.md#streaming-overlap-measurement
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int, choices=(0, 1))
    parser.add_argument('mode', choices=('whole', 'streamed'))
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--tile', type=int, default=64)
    parser.add_argument('--trials', type=int, default=3000)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--trace')
    args = parser.parse_args()
    if min(args.rows, args.tile, args.trials, args.depth) < 1:
        parser.error('rows, tile, trials and depth must be positive')
    running = True

    # design/algorithm-sources.md#streaming-overlap-measurement
    def stop(*unused):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps(dict(pid=os.getpid(), rank=args.rank, mode=args.mode)), flush=True)
    with Program(backend=args.backend) as program:
        rows, tile, width = args.rows, args.tile, 256
        block_rows = min(tile, rows) if args.mode == 'streamed' else rows
        count = (rows + block_rows - 1) // block_rows
        w = program.tensor((width, width))
        v = program.tensor((width, width))
        rng = np.random.default_rng(1701)
        weights = rng.standard_normal((width, width), dtype=np.float32) / np.sqrt(width)
        projection = rng.standard_normal((width, width), dtype=np.float32) / np.sqrt(width)
        data = rng.standard_normal((rows, width), dtype=np.float32)
        program.constant(w[0, 0], weights)
        program.constant(v[0, 0], projection)

        # design/algorithm-sources.md#streaming-overlap-measurement
        def configure():
            x = program.tensor((rows, width), (block_rows, width))
            row = BlockSpec((block_rows, width), lambda i: (i, 0))
            weight = BlockSpec((width, width), lambda i: (0, 0))
            shape = ShapeDtypeStruct((rows, width), np.float32)
            p = program.kernel_call(kernels.matmul, grid=(count,),
                in_specs=(row, weight), out_specs=row, out_shape=shape, peer=0)(x, w)
            received = program.tensor(p.shape, p.block_shape, dtype=p.dtype)
            y = program.kernel_call(kernels.matmul, grid=(count,),
                in_specs=(row, weight), out_specs=row, out_shape=shape, peer=1)(received, v)
            returned = program.tensor(y.shape, y.block_shape, dtype=y.dtype)
            program.copy(p.on(0), received.on(1), queue=0)
            program.copy(y.on(1), returned.on(0), queue=0)
            results = tuple(program.export(ref) for ref in returned.blocks.values()) if args.rank == 0 else ()
            return tuple(x.blocks.values()), results

        slots = tuple(configure() for _ in range(args.depth))
        program.realize()
        print(json.dumps(dict(event='realized', rank=args.rank, mode=args.mode,
            backend=args.backend, regions=count, depth=args.depth)), flush=True)
        if args.rank == 1:
            while running:
                signal.pause()
            if args.trace:
                Path(args.trace).write_text(json.dumps(dict(compute=program.trace,
                    transfers=program.transfer_trace), indent=2) + '\n')
            print(json.dumps(dict(rank=1, mode=args.mode, runtime=program.report)), flush=True)
            return
        reference = (data @ weights) @ projection

        # design/algorithm-sources.md#streaming-overlap-measurement
        def run_batch(total):
            submitted = completed = 0
            outstanding = {}
            terminal = []
            latencies, first_sections = [], []
            began_batch = time.perf_counter()
            deadline = began_batch + 60
            while completed < total and running:
                for slot, (inputs, results) in enumerate(slots):
                    if slot in outstanding or submitted == total or not all(ref.writable for ref in inputs):
                        continue
                    began = time.perf_counter()
                    scale = 1 + (submitted % 17) / 32
                    for section, ref in enumerate(inputs):
                        start = section * block_rows
                        with program.write(ref) as buffer:
                            np.multiply(data[start:start + ref.shape[0]], scale, out=buffer)
                    outstanding[slot] = (submitted, began, False)
                    submitted += 1
                for slot, (_, results) in enumerate(slots):
                    if slot not in outstanding:
                        continue
                    trial, began, first_seen = outstanding[slot]
                    ready = tuple(result.ready for result in results)
                    if not first_seen and any(ready):
                        first_sections.append(time.perf_counter() - began)
                        outstanding[slot] = (trial, began, True)
                    if not all(ready):
                        continue
                    latencies.append(time.perf_counter() - began)
                    del outstanding[slot]
                    completed += 1
                    deadline = time.perf_counter() + 60
                    if submitted < total:
                        for result in results:
                            result.consume()
                    else:
                        terminal.append((results, trial))
                if time.perf_counter() > deadline:
                    raise TimeoutError(f'Contraction pipeline stalled: {program.report}')
            ended_batch = time.perf_counter()
            if not running:
                raise InterruptedError('Contraction observation interrupted')
            maximum_error = 0.0
            for results, trial in terminal:
                scale = 1 + (trial % 17) / 32
                for section, result in enumerate(results):
                    start = section * block_rows
                    expected = reference[start:start + result.ref.shape[0]] * scale
                    maximum_error = max(maximum_error, float(np.max(np.abs(result.array - expected))))
                    if not np.allclose(result.array, expected, rtol=2e-4, atol=2e-4):
                        raise ArithmeticError('Distributed contraction differs from the reference')
                    result.consume()
            return dict(seconds=latencies, statistics=moments(latencies),
                first_section_seconds=first_sections, first_section_statistics=moments(first_sections),
                steady_seconds=ended_batch - began_batch,
                results_per_second=total / (ended_batch - began_batch),
                max_absolute_error=maximum_error, checked_terminal_invocations=len(terminal))

        run_batch(5 * args.depth)
        measured = run_batch(args.trials)
        if args.trace:
            Path(args.trace).write_text(json.dumps(dict(compute=program.trace,
                transfers=program.transfer_trace), indent=2) + '\n')
        print(json.dumps(dict(rank=0, mode=args.mode, rows=rows, tile=tile,
            depth=args.depth, backend=args.backend, **measured,
            numerical_calls_per_trial=2 * count, runtime=program.report)), flush=True)


if __name__ == '__main__':
    main()
