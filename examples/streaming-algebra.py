import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from mesh import Program
from mesh.nn import ffn, rmsnorm, summed_embedding


# design/algorithm-sources.md#streamed-normalization-and-embedding
def reference_ffn(inputs, up, down):
    result = np.zeros((inputs[0].shape[0], down[0].shape[1]), np.float64)
    for weights, projection in zip(up, down):
        hidden = sum(value.astype(np.float64) @ weight for value, weight in zip(inputs, weights))
        result += (hidden / (1 + np.exp(-hidden))) @ projection
    return result


# design/algorithm-sources.md#streamed-normalization-and-embedding
def reference_norm(value, gamma):
    return value / np.sqrt(np.mean(value * value, axis=1, keepdims=True) + 1e-6) * gamma


# design/algorithm-sources.md#streamed-normalization-and-embedding
def summary(values):
    return dict(count=len(values), mean=float(np.mean(values)), sample_variance=float(np.var(values, ddof=1)) if len(values) > 1 else None)


# design/algorithm-sources.md#streamed-normalization-and-embedding
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--tile-k', type=int, default=64)
    parser.add_argument('--tile-columns', type=int, default=64)
    parser.add_argument('--trace')
    args = parser.parse_args()
    rows, width, tile = 256, 128, 64
    running = True

    # design/algorithm-sources.md#streamed-normalization-and-embedding
    def stop(*unused):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with Program(backend=args.backend) as program:
        rng = np.random.default_rng(701)

        # design/algorithm-sources.md#streamed-normalization-and-embedding
        def weight(value):
            tensor = program.tensor(value.shape, dtype=value.dtype)
            program.constant(tensor[0, 0], value)
            return tensor

        # design/algorithm-sources.md#streamed-normalization-and-embedding
        def weights(partitions):
            up = tuple(tuple(rng.standard_normal((width, width), dtype=np.float32) / 16
                       for _ in range(partitions)) for _ in range(2))
            down = tuple(rng.standard_normal((width, width), dtype=np.float32) / 16 for _ in range(2))
            return up, down, tuple(tuple(weight(w) for w in group) for group in up), tuple(weight(w) for w in down)

        first_up, first_down, first_u, first_d = weights(2)
        second_up, second_down, second_u, second_d = weights(1)
        gamma = np.ones((1, width), np.float32)
        scale = weight(gamma)
        tables = tuple(rng.standard_normal((32, width), dtype=np.float32) / 32 for _ in range(2))
        table_tensors = tuple(weight(value) for value in tables)
        ids = tuple(rng.integers(0, 32, (rows, 1), dtype=np.int64) for _ in range(2))
        index_tensors = tuple(weight(value) for value in ids)
        invocations = []

        # design/algorithm-sources.md#async-index-push-contract
        def exchange(value, sender, receiver):
            if args.local:
                return value
            received = program.tensor(value.shape, value.block_shape, dtype=value.dtype)
            program.copy(value.on(sender), received.on(receiver), queue=0)
            return received

        for run in range(args.runs):
            data = tuple(rng.standard_normal((rows, width), dtype=np.float32) / 8 for _ in range(2))
            inputs = tuple(program.tensor(value.shape, (tile, args.tile_k)) for value in data)
            a = ffn(program, inputs, first_u, first_d, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns,
                    exchange=lambda value: exchange(value, 0, 1))
            b = rmsnorm(program, a, scale, tile_rows=tile)
            c = summed_embedding(program, b, table_tensors, index_tensors, tile_rows=tile)
            d = ffn(program, (c,), second_u, second_d, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns,
                    exchange=lambda value: exchange(value, 1, 0))
            e = rmsnorm(program, d, scale, tile_rows=tile)
            stages = {'ffn1': a, 'rmsnorm1': b, 'summed_embedding': c} if args.rank == 1 else {'ffn2': d, 'rmsnorm2': e}
            probes = {name: {coordinate: program.export(ref) for coordinate, ref in value.blocks.items()}
                      for name, value in stages.items()}
            invocations.append((data, inputs, probes))
        program.realize()
        print(json.dumps(dict(event='realized', pid=os.getpid(), rank=args.rank, runs=args.runs)), flush=True)
        if args.rank == 1:
            observed = set()
            while running:
                for run, (_, _, probes) in enumerate(invocations):
                    for name, results in probes.items():
                        if (run, name) not in observed and results[1, 0].ready:
                            observed.add((run, name))
                            print(json.dumps(dict(event='section_ready', run=run, stage=name,
                                local_monotonic_ns=time.monotonic_ns(), section_zero_absent=not results[0, 0].ready)), flush=True)
                time.sleep(0.0001)
            if args.trace:
                Path(args.trace).write_text(json.dumps(dict(compute=program.trace, transfers=program.transfer_trace), indent=2) + '\n')
            return
        first_times, complete_times, errors = [], [], []
        for run, (data, inputs, probes) in enumerate(invocations):
            started = time.monotonic_ns()
            for tensor, value in zip(inputs, data):
                for (section, panel), ref in tensor.blocks.items():
                    if section == 0:
                        continue
                    with program.write(ref) as target:
                        r, c = section * tensor.block_shape[0], panel * tensor.block_shape[1]
                        target[...] = value[r:r + ref.shape[0], c:c + ref.shape[1]]
            results = probes['rmsnorm2']
            seen = {}
            while running and not results[1, 0].ready:
                for name, stage in probes.items():
                    if name not in seen and stage[1, 0].ready:
                        seen[name] = (time.monotonic_ns() - started) / 1e6
                if time.monotonic_ns() - started > 60_000_000_000:
                    raise TimeoutError(f'Independent section stalled: {program.report}')
                time.sleep(0.0001)
            if not running:
                return
            first_ms = (time.monotonic_ns() - started) / 1e6
            if results[0, 0].ready:
                raise ArithmeticError('An unpublished input section produced an output')
            for tensor, value in zip(inputs, data):
                for (section, panel), ref in tensor.blocks.items():
                    if section != 0:
                        continue
                    with program.write(ref) as target:
                        c = panel * tensor.block_shape[1]
                        target[...] = value[:ref.shape[0], c:c + ref.shape[1]]
            while running and not all(result.ready for result in results.values()):
                if time.monotonic_ns() - started > 60_000_000_000:
                    raise TimeoutError(f'Final sections stalled: {program.report}')
                time.sleep(0.0001)
            if not running:
                return
            complete_ms = (time.monotonic_ns() - started) / 1e6
            expected = reference_norm(reference_ffn(data, first_up, first_down), gamma)
            expected += sum(table[index[:, 0]] for table, index in zip(tables, ids))
            expected = reference_norm(reference_ffn((expected,), second_up, second_down), gamma)
            error = 0.0
            for (row, column), result in results.items():
                r, c = row * tile, column * args.tile_columns
                part = expected[r:r + result.ref.shape[0], c:c + result.ref.shape[1]]
                error = max(error, float(np.max(np.abs(result.array - part))))
                if not np.allclose(result.array, part, atol=3e-4, rtol=3e-4):
                    raise ArithmeticError(f'Gold chain numerical mismatch: {error}')
            first_times.append(first_ms); complete_times.append(complete_ms); errors.append(error)
            print(json.dumps(dict(event='gold', run=run, first_section_ms=first_ms, complete_ms=complete_ms,
                observed_stage_ms=seen, withheld_section=0, progressed_section=1,
                max_absolute_error=error, transport_queue=0, backend=args.backend, local=args.local)), flush=True)
            for stage in probes.values():
                for result in stage.values():
                    result.consume()
        if args.trace:
            Path(args.trace).write_text(json.dumps(dict(compute=program.trace, transfers=program.transfer_trace), indent=2) + '\n')
        report = program.report
        print(json.dumps(dict(event='summary', first_section_ms=summary(first_times), complete_ms=summary(complete_times),
            max_absolute_error=max(errors), runtime={name: getattr(report, name) for name, _ in report._fields_})), flush=True)


if __name__ == '__main__':
    main()
