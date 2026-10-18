import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import numpy as np  # noqa: E402
import kernels  # noqa: E402
from mesh import BlockSpec, Program, ShapeDtypeStruct  # noqa: E402
from mesh.collective import reduce, sync_on_remote_fill, sync_on_some_remote_fill  # noqa: E402
from nn import ffn, linear, rmsnorm, summed_embedding  # noqa: E402

STOP = (signal.SIGTERM, signal.SIGINT)


class Stopped(Exception):
    pass


# design/algorithm-sources.md#streaming-overlap-measurement
def stop(*_):
    raise Stopped


# design/algorithm-sources.md#nnffn
def reference_ffn(inputs, up, down):
    result = np.zeros((inputs[0].shape[0], down[0].shape[1]), np.float64)
    for weights, projection in zip(up, down):
        hidden = sum(value.astype(np.float64) @ weight for value, weight in zip(inputs, weights))
        result += (hidden / (1 + np.exp(-hidden))) @ projection
    return result


# design/algorithm-sources.md#nnrmsnorm
def reference_norm(value, gamma):
    return value / np.sqrt(np.mean(value * value, axis=1, keepdims=True) + 1e-6) * gamma


# design/algorithm-sources.md#streaming-overlap-measurement
def summary(values):
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                sample_variance=float(np.var(values, ddof=1)) if len(values) > 1 else None)


# design/algorithm-sources.md#streaming-overlap-measurement
def overlap(trace):
    compute = [(begin, end) for kind, _, _, begin, end, _ in trace if kind == 'kernel']
    crossings = [(begin, end) for kind, _, _, begin, end, _ in trace if kind in ('copy', 'reduce', 'broadcast', 'all_reduce')]
    return dict(crossings=len(crossings), kernels=len(compute),
                crossings_overlapping_compute=sum(any(b < e2 and b2 < e for b2, e2 in compute) for b, e in crossings))


# design/algorithm-sources.md#nnffn
def literature(args, program):
    rows, width, tile = 256, 128, 64
    rng = np.random.default_rng(701)

    # design/algorithm-sources.md#nnffn
    def weight(value):
        tensor = program.tensor(value.shape, dtype=value.dtype)
        program.constant(tensor[0, 0], value)
        return tensor

    # design/algorithm-sources.md#nnffn
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

    # design/algorithm-sources.md#index-hand-off
    def exchange(value, sender, receiver):
        if args.local:
            return value
        received = program.tensor(value.shape, value.block_shape, dtype=value.dtype)
        for (i, j), ref in value.blocks.items():
            program.copy(ref.on(sender), received[i, j].on(receiver), queue=i + (value.grid[0] if sender else 0))
        return received

    for run in range(args.runs + 1):
        data = tuple(rng.standard_normal((rows, width), dtype=np.float32) / 8 for _ in range(2))
        inputs = tuple(program.tensor(value.shape, (tile, args.tile_k)) for value in data)
        a = ffn(program, inputs, first_u, first_d, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns,
                exchange=lambda value: exchange(value, 0, 1))
        b = rmsnorm(program, a, scale, tile_rows=tile)
        c = summed_embedding(program, b, table_tensors, index_tensors, tile_rows=tile)
        d = ffn(program, (c,), second_u, second_d, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns,
                exchange=lambda value: exchange(value, 1, 0))
        e = rmsnorm(program, d, scale, tile_rows=tile)
        stages = {'ffn1': a, 'rmsnorm1': b, 'summed_embedding': c} if program.rank == 1 else {'ffn2': d, 'rmsnorm2': e}
        probes = {name: {coordinate: program.export(ref) for coordinate, ref in value.blocks.items()}
                  for name, value in stages.items()}
        invocations.append((data, inputs, probes))
    program.realize()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP)
    print(json.dumps(dict(event='realized', pid=os.getpid(), rank=program.rank, runs=args.runs, local=args.local)), flush=True)
    if program.rank == 1:
        watched = {results[1, 0]: (run, name, results[0, 0]) for run, (_, _, probes) in enumerate(invocations)
                   for name, results in probes.items()}
        while watched:
            for result in sync_on_some_remote_fill(*watched):
                run, name, first = watched.pop(result)
                print(json.dumps(dict(event='section_ready', run=run, stage=name, local_monotonic_ns=time.monotonic_ns(),
                                      section_zero_absent=not first.ready)), flush=True)
        while True:
            signal.pause()

    # design/algorithm-sources.md#streaming-overlap-measurement
    def publish(invocation, sections):
        data, inputs, _ = invocation
        for tensor, value in zip(inputs, data):
            for (section, panel), ref in tensor.blocks.items():
                if section not in sections:
                    continue
                with program.write(ref) as target:
                    r, c = section * tensor.block_shape[0], panel * tensor.block_shape[1]
                    target[...] = value[r:r + ref.shape[0], c:c + ref.shape[1]]

    warm_started = time.monotonic_ns()
    warm = invocations[0]
    publish(warm, range(1, rows // tile))
    warm_results = warm[2]['rmsnorm2']
    sync_on_remote_fill(warm_results[1, 0])
    if any(result.ready for coordinate, result in warm_results.items() if coordinate[0] == 0):
        raise ArithmeticError('An unpublished input section produced an output')
    publish(warm, (0,))
    sync_on_remote_fill(*warm_results.values())
    print(json.dumps(dict(event='warmup', elapsed_ms=(time.monotonic_ns() - warm_started) / 1e6)), flush=True)

    batch_started = time.monotonic_ns()
    for run, invocation in enumerate(invocations[1:], 1):
        publish(invocation, range(1 if run == 1 else 0, rows // tile))
    delayed = invocations[1][2]['rmsnorm2']
    sync_on_remote_fill(delayed[1, 0])
    first_ms = (time.monotonic_ns() - batch_started) / 1e6
    if any(result.ready for coordinate, result in delayed.items() if coordinate[0] == 0):
        raise ArithmeticError('An unpublished input section produced an output')
    publish(invocations[1], (0,))
    completed = {}
    waiting = {run: set(probes['rmsnorm2'].values()) for run, (_, _, probes) in enumerate(invocations[1:], 1)}
    while waiting:
        filled = set(sync_on_some_remote_fill(*(result for results in waiting.values() for result in results)))
        for run in tuple(waiting):
            waiting[run] -= filled
            if not waiting[run]:
                completed[run] = (time.monotonic_ns() - batch_started) / 1e6
                del waiting[run]
    batch_ms = (time.monotonic_ns() - batch_started) / 1e6
    errors = []
    for run, (data, _, probes) in enumerate(invocations):
        expected = reference_norm(reference_ffn(data, first_up, first_down), gamma)
        expected += sum(table[index[:, 0]] for table, index in zip(tables, ids))
        expected = reference_norm(reference_ffn((expected,), second_up, second_down), gamma)
        error = 0.0
        for (row, column), result in probes['rmsnorm2'].items():
            r, c = row * tile, column * result.ref.shape[1]
            part = expected[r:r + result.ref.shape[0], c:c + result.ref.shape[1]]
            error = max(error, float(np.max(np.abs(result.array - part))))
            if not np.allclose(result.array, part, atol=3e-4, rtol=3e-4):
                raise ArithmeticError(f'Gold chain numerical mismatch: {error}')
        errors.append(error)
        print(json.dumps(dict(event='gold', run=run, complete_ms=completed.get(run), max_absolute_error=error,
                              backend=args.backend, local=args.local)), flush=True)
    return dict(invocations=args.runs, batch_ms=batch_ms, invocations_per_second=args.runs * 1000 / batch_ms,
                first_section_ms=first_ms, completion_ms=summary(tuple(completed.values())), withheld_invocation=1,
                withheld_section=0, max_absolute_error=max(errors))


# design/algorithm-sources.md#nnffn
def tensor_parallel(args, program):
    values = np.load(args.input)
    up, down, consumer = (np.load(name, mmap_mode='r') for name in (args.up_weight, args.down_weight, args.consumer_weight))
    if up.shape[1] != down.shape[0] or not 0 < args.split <= up.shape[1]:
        raise ValueError('The split is a hidden column of the up projection')
    members = (args.root,) if args.local else (args.root, args.peer)
    spans = {args.root: (0, up.shape[1] if args.local else args.split), args.peer: (args.split, up.shape[1])}
    start, end = spans[program.rank]
    up_weight = program.tensor((up.shape[0], end - start), dtype=up.dtype)
    down_weight = program.tensor((end - start, down.shape[1]), dtype=down.dtype)
    program.constant(up_weight[0, 0], up[:, start:end])
    program.constant(down_weight[0, 0], down[start:end, :])
    consumer_weight = program.tensor(consumer.shape, dtype=consumer.dtype)
    program.constant(consumer_weight[0, 0], consumer)
    gamma = program.tensor((1, down.shape[1]), dtype=values.dtype)
    program.constant(gamma[0, 0], np.ones(gamma.shape, dtype=values.dtype))
    inputs, outputs = [], {}
    for instance in range(args.instances):
        x = program.tensor(values.shape, (args.tile_rows, values.shape[1] if args.residual else args.tile_k),
                           dtype=values.dtype)
        inputs.append(x)
        projected = ffn(program, (x,), ((up_weight,),), (down_weight,), tile_rows=args.tile_rows, tile_k=args.tile_k,
                        tile_columns=args.tile_columns)
        reduced = reduce(program, projected, root=args.root, peers=members)
        if args.normalize:
            if args.residual:
                if x.shape != reduced.shape:
                    raise ValueError('Residual input and reduced projection must have the same shape')
                reduced = program.kernel_call(kernels.add, grid=reduced.grid,
                    in_specs=(BlockSpec(reduced.block_shape, lambda i, j: (i, j)),
                              BlockSpec(reduced.block_shape, lambda i, j: (i, j))),
                    out_specs=BlockSpec(reduced.block_shape, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(reduced.shape, reduced.dtype), peer=args.root)(reduced, x)
            reduced = rmsnorm(program, reduced, gamma, tile_rows=args.tile_rows, peer=args.root)
        spec = BlockSpec(reduced.block_shape, lambda i, j: (i, j))
        activated = program.kernel_call(kernels.swish, grid=reduced.grid, in_specs=(spec,), out_specs=spec,
                                        out_shape=ShapeDtypeStruct(reduced.shape, reduced.dtype), peer=args.root)(reduced)
        consumed = linear(program, activated, consumer_weight, tile_rows=args.tile_rows, tile_k=args.tile_k,
                          tile_columns=args.tile_columns, peer=args.root)
        if program.rank == args.root:
            outputs.update({(instance, *index): program.export(ref) for index, ref in consumed.blocks.items()})
    program.realize()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP)
    began = time.monotonic_ns()
    for i, j in inputs[0].blocks:
        for x in inputs:
            ref = x[i, j]
            row, column = i * x.block_shape[0], j * x.block_shape[1]
            with program.write(ref) as destination:
                destination[...] = values[row:row + ref.shape[0], column:column + ref.shape[1]]
    if program.rank != args.root:
        while True:
            signal.pause()
    count = len(outputs)
    indices = {output: index for index, output in outputs.items()}
    while indices:
        for output in sync_on_some_remote_fill(*indices):
            print(indices.pop(output), output.array.tolist(), flush=True)
            output.consume()
    return dict(instances=args.instances, regions=count, elapsed_ms=(time.monotonic_ns() - began) / 1e6)


# design/algorithm-sources.md#nnffn
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', choices=('literature', 'tensor-parallel'), default='literature')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--runs', type=int, default=8)
    parser.add_argument('--tile-k', type=int, default=64)
    parser.add_argument('--tile-columns', type=int, default=64)
    parser.add_argument('--trace')
    parser.add_argument('--input')
    parser.add_argument('--up-weight')
    parser.add_argument('--down-weight')
    parser.add_argument('--consumer-weight')
    parser.add_argument('--root', type=int, default=0)
    parser.add_argument('--peer', type=int, default=1)
    parser.add_argument('--split', type=int)
    parser.add_argument('--instances', type=int, default=2)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--tile-rows', type=int, default=128)
    args = parser.parse_args()
    if args.runs < 1 or args.instances < 1:
        parser.error('--runs and --instances must be positive')
    if args.chain == 'tensor-parallel' and not all((args.input, args.up_weight, args.down_weight, args.consumer_weight,
                                                    args.split)):
        parser.error('the tensor-parallel chain takes --input, --up-weight, --down-weight, --consumer-weight and --split')
    if args.residual and not args.normalize:
        parser.error('--residual requires --normalize')
    if args.chain == 'tensor-parallel' and args.root == args.peer:
        parser.error('--root and --peer are two members')
    for number in STOP:
        signal.signal(number, stop)
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP)
    with Program(backend=args.backend, trace=bool(args.trace)) as program:
        if args.local and program.size != 1:
            parser.error('--local is world size 1: run one rank (WORLD_SIZE unset or 1)')
        if not args.local and program.size != 2:
            parser.error('the paired chain runs on two ranks (RANK, WORLD_SIZE=2, MASTER_ADDR, MASTER_PORT)')
        chain = literature if args.chain == 'literature' else tensor_parallel
        try:
            result = chain(args, program)
        except Stopped:
            result = {}
        if args.trace and program.trace is not None:
            Path(args.trace).write_text(json.dumps(dict(rank=program.rank, trace=program.trace)) + '\n')
        if result:
            print(json.dumps(dict(event='summary', chain=args.chain, rank=program.rank, backend=args.backend,
                                  local=args.local, **result, **(overlap(program.trace) if program.trace is not None else {}),
                                  report=vars(program.report))), flush=True)


if __name__ == '__main__':
    main()
