import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from mesh import Program, BlockSpec, ShapeDtypeStruct, kernels
from mesh.nn import ffn, linear, rmsnorm, summed_embedding


# design/algorithm-sources.md#streamed-normalization-and-embedding
def reference_ffn(inputs, up, down):
    dtype = np.float16 if inputs[0].dtype == np.float16 else np.float64
    result = np.zeros((inputs[0].shape[0], down[0].shape[1]), np.float64)
    for weights, projection in zip(up, down):
        hidden = sum(value.astype(np.float64) @ weight for value, weight in zip(inputs, weights))
        result += (hidden / (1 + np.exp(-hidden))).astype(dtype).astype(np.float64) @ projection
    return result.astype(dtype)


# design/algorithm-sources.md#streamed-normalization-and-embedding
def reference_norm(value, gamma):
    dtype = value.dtype
    value = value.astype(np.float64)
    return (value / np.sqrt(np.mean(value * value, axis=1, keepdims=True) + 1e-6) * gamma).astype(dtype)


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
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--tile-k', type=int, default=64)
    parser.add_argument('--tile-columns', type=int, default=64)
    parser.add_argument('--scatter-rows', type=int, default=6)
    parser.add_argument('--scatter-tile', type=int, default=2)
    parser.add_argument('--scatter-destinations', type=int, default=4)
    parser.add_argument('--trace')
    parser.add_argument('--xonotic', action='store_true')
    parser.add_argument('--coreml', nargs=3, metavar=('PYTHON', 'GENERATOR', 'CACHE'))
    args = parser.parse_args()
    updates_count, update_tile = args.scatter_rows, args.scatter_tile
    destinations_count = args.scatter_destinations
    if destinations_count < 4:
        parser.error('Scatter demonstration requires at least four destinations')
    early_destinations = tuple(i for i in range(destinations_count) if i != 2)
    if updates_count < 6 or not 1 <= update_tile < updates_count:
        parser.error('Scatter demonstration requires at least six rows and a smaller positive tile')
    last_chunk = (updates_count - 1) // update_tile
    last_start = last_chunk * update_tile
    if last_start < 4:
        parser.error('Scatter demonstration requires four rows before the withheld tile')
    rows, width, tile = 256, 128, 64
    dtype = np.dtype(args.dtype)
    running = True

    # design/algorithm-sources.md#streamed-normalization-and-embedding
    def stop(*unused):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with Program(backend=args.backend, coreml=args.coreml) as program:
        rng = np.random.default_rng(701)

        # design/algorithm-sources.md#streamed-normalization-and-embedding
        def weight(value):
            tensor = program.tensor(value.shape, dtype=value.dtype)
            program.constant(tensor[0, 0], value)
            return tensor

        # design/algorithm-sources.md#streamed-normalization-and-embedding
        def weights(partitions):
            up = tuple(tuple((rng.standard_normal((width, width), dtype=np.float32) / 16).astype(dtype)
                       for _ in range(partitions)) for _ in range(2))
            down = tuple((rng.standard_normal((width, width), dtype=np.float32) / 16).astype(dtype) for _ in range(2))
            return up, down, tuple(tuple(weight(w) for w in group) for group in up), tuple(weight(w) for w in down)

        first_up, first_down, first_u, first_d = weights(2)
        second_up, second_down, second_u, second_d = weights(1)
        gamma = np.ones((1, width), dtype)
        scale = weight(gamma)
        tables = tuple((rng.standard_normal((32, width), dtype=np.float32) / 32).astype(dtype) for _ in range(2))
        table_tensors = tuple(weight(value) for value in tables)
        ids = tuple(rng.integers(0, 32, (rows, 1), dtype=np.int64) for _ in range(2))
        index_tensors = tuple(weight(value) for value in ids)
        precision = program.export(linear(program,
            weight(np.array([[4096, 1, -4096, 1], [60000, 60000, -60000, -60000]], dtype=dtype)),
            weight(np.ones((4, 1), dtype=dtype)), tile_rows=2, tile_k=2, tile_columns=1, peer=0)[0, 0])
        strided_left = (np.arange(15, dtype=np.float32).reshape(5, 3) / 8 - 1).astype(dtype)
        strided_right = (np.arange(35, dtype=np.float32).reshape(7, 5) / 16 - 1).astype(dtype)
        strided = program.export(linear(program, weight(strided_left).T, weight(strided_right).T,
            tile_rows=3, tile_k=5, tile_columns=7, peer=0, output_dtype="float32")[0, 0])
        mapped_left, mapped_right = kernels.arguments(2)
        mapped = program.kernel_call(kernels.expression(kernels.dot(mapped_left, mapped_right, tile_k=3)),
            grid=(3,), in_specs=(BlockSpec((1, 5), lambda i: (2-i, 0)), BlockSpec((5, 7), lambda i: (0, 0))),
            out_specs=BlockSpec((1, 7), lambda i: (i, 0)),
            out_shape=ShapeDtypeStruct((3, 7), np.float32), peer=0)(weight(strided_left).T, weight(strided_right).T)
        mapped_results = tuple(program.export(mapped[i, 0]) for i in range(3))
        table_arg, index_arg, deferred_arg = kernels.arguments(3)
        _, column_arg = kernels.indices()
        index_data = np.array([[2], [2**53 + 1]], dtype=np.int64)
        table_data = np.arange(12, dtype=dtype).reshape(3, 4)
        indexed_body = 2 * table_arg.at(index_arg, column_arg, mask=(index_arg >= 0) & (index_arg < 3)) + kernels.select(index_arg.equal(2**53 + 1), 1, 0)
        deferred_input = program.tensor((2, 4), dtype=dtype)
        indexed_outputs = program.kernel_call(kernels.expression(indexed_body, deferred_arg * 3), grid=(1,),
            in_specs=(BlockSpec((3, 4), lambda i: (0, 0)), BlockSpec((2, 1), lambda i: (0, 0)),
                      BlockSpec((2, 4), lambda i: (0, 0))),
            out_specs=(BlockSpec((2, 4), lambda i: (0, 0)), BlockSpec((2, 4), lambda i: (0, 0))),
            out_shape=(ShapeDtypeStruct((2, 4), dtype), ShapeDtypeStruct((2, 4), dtype)), peer=0)(
                weight(table_data), weight(index_data), deferred_input)
        indexed, deferred_output = (program.export(value[0, 0]) for value in indexed_outputs)
        streamed_table = program.tensor((4, 4), (2, 4), dtype=dtype)
        streamed_indices = weight(np.array([[2], [3]], dtype=np.int64))
        source_arg, selected_arg = kernels.arguments(2)
        streamed_result = program.export(program.kernel_call(
            kernels.expression(2 * source_arg.at(selected_arg, column_arg)), grid=(1,),
            in_specs=(BlockSpec(None), BlockSpec((2, 1), lambda i: (0, 0))),
            out_specs=BlockSpec((2, 4), lambda i: (0, 0)),
            out_shape=ShapeDtypeStruct((2, 4), dtype), peer=0)(streamed_table, streamed_indices)[0, 0])
        scatter_indices = program.tensor((updates_count, 1), (update_tile, 1), dtype=np.int64)
        scatter_updates = program.tensor((updates_count, 4), (update_tile, 4), dtype=dtype)
        scatter_factors = program.tensor((updates_count, 1), (update_tile, 1), dtype=np.float32)
        scatter_valid = np.ones((updates_count, 1), dtype=bool)
        scatter_valid[-1 if updates_count-last_start > 1 else -2, 0] = False
        base_arg, destination_arg, update_arg, mask_arg, factor_arg = kernels.arguments(5)
        scatter = program.kernel_call(kernels.expression(kernels.indexed_add(
            base_arg, destination_arg, update_arg * factor_arg + 1, mask=mask_arg)), grid=(destinations_count,),
            in_specs=(BlockSpec(None),) * 5, out_specs=BlockSpec((1, 4), lambda i: (i, 0)),
            out_shape=ShapeDtypeStruct((destinations_count, 4), dtype), peer=0)(
                weight(np.zeros((destinations_count, 4), dtype=dtype)), scatter_indices, scatter_updates,
                weight(scatter_valid), scatter_factors)
        consumer_arg, = kernels.arguments(1)
        scatter_consumed = program.kernel_call(kernels.expression(consumer_arg * 2), grid=(destinations_count,),
            in_specs=(BlockSpec((1, 4), lambda i: (i, 0)),),
            out_specs=BlockSpec((1, 4), lambda i: (i, 0)),
            out_shape=ShapeDtypeStruct((destinations_count, 4), dtype), peer=0)(scatter)
        scatter_results = tuple(program.export(scatter_consumed[i, 0]) for i in range(destinations_count))
        xonotic_case = None
        if args.xonotic and args.rank == 0:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'xonotic'))
            from solver.strat import tensor as mx
            from solver.strat.tensor_metal import kernel_calls
            x_source = program.tensor((4, 4), (2, 4), dtype=np.float32)
            x_indices = program.tensor((1, 4), (1, 2), dtype=np.int64)
            x_tail = program.tensor((2, 4), dtype=np.float32)
            graph = mx.Graph()
            with graph:
                source = graph.input('source', (4, 4))
                indices = graph.input('indices', (4,), 'int64')
                tail = graph.input('tail', (2, 4))
                selected = source[indices, ::-1]
                joined = mx.concatenate((selected, tail), axis=0)
            lowered = kernel_calls(program, graph, (),
                {source.index: x_source, indices.index: x_indices, tail.index: x_tail},
                root_peer=0, tile_rows=2, tile_columns=4)
            observations = tuple(program.export(lowered[joined.index][i, 0]) for i in range(3))
            bindings = {}
            for name, references in (
                    ('source', tuple(sorted(x_source.blocks.items()))),
                    ('indices', tuple(sorted(x_indices.blocks.items()))),
                    ('tail', tuple(sorted(x_tail.blocks.items()))),
                    ('gathered', tuple(sorted(lowered[selected.index].blocks.items()))),
                    ('output', tuple(((i, 0), result.ref) for i, result in enumerate(observations)))):
                entries = []
                for coordinate, ref in references:
                    mapping = program.native.tensor_rows(ref.view.tensor, ref.view.extent)
                    entries.append(dict(coordinate=coordinate, first=mapping.first, count=mapping.count,
                        extent=ref.view.extent, offset=ref.view.offset, shape=ref.shape,
                        strides=(ref.view.row_stride, ref.view.column_stride), dtype=str(ref.dtype)))
                bindings[name] = entries
            print(json.dumps(dict(event='xonotic_indexed_bindings', rank=program.node, bindings=bindings)), flush=True)
            generations = []
            for generation in range(2):
                source_values = np.arange(16, dtype=np.float32).reshape(4, 4) + 32 * generation
                index_values = np.array([2, -1, 0, 1] if generation == 0 else [1, 0, -1, 2], dtype=np.int64)
                tail_values = np.arange(8, dtype=np.float32).reshape(2, 4) - 16 * (generation + 1)
                expected = np.concatenate((source_values.astype(np.float64)[index_values, ::-1], tail_values.astype(np.float64)))
                generations.append((source_values, index_values, tail_values, expected))
            xonotic_case = (x_source, x_indices, x_tail, observations, generations)
        invocations = []

        # design/algorithm-sources.md#async-index-push-contract
        def exchange(value, sender, receiver):
            if args.local:
                return value
            received = program.tensor(value.shape, value.block_shape, dtype=value.dtype)
            program.copy(value.on(sender), received.on(receiver), queue=0)
            return received

        fanout_source = program.tensor((2, 4), dtype=dtype)
        fanout_received = exchange(fanout_source, 0, 1)
        fanout_arg, fanout_factor = kernels.arguments(2)
        fanout_factors = weight(np.arange(1, 66, dtype=dtype).reshape(-1, 1))
        fanout_results = []
        for factor in range(1, 66):
            branch = program.kernel_call(kernels.expression(fanout_arg * fanout_factor), grid=(1,),
                in_specs=(BlockSpec((2, 4), lambda i: (0, 0)),
                          BlockSpec((1, 1), lambda i, factor=factor: (factor-1, 0))),
                out_specs=BlockSpec((2, 4), lambda i: (0, 0)),
                out_shape=ShapeDtypeStruct((2, 4), dtype), peer=0 if args.local else 1)(fanout_received, fanout_factors)
            fanout_results.append(program.export(exchange(branch, 1, 0)[0, 0]))
        for run in range(args.runs + 1):
            data = tuple((rng.standard_normal((rows, width), dtype=np.float32) / 8).astype(dtype) for _ in range(2))
            inputs = tuple(program.tensor(value.shape, (tile, args.tile_k), dtype=dtype) for value in data)
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

        # design/algorithm-sources.md#streaming-overlap-measurement
        def wait_for(results, attribute='ready'):
            deadline = time.monotonic_ns() + 60_000_000_000
            while running and not all(getattr(result, attribute) for result in results):
                if time.monotonic_ns() > deadline:
                    raise TimeoutError(f'Observation stalled on {attribute}: {program.report}')
                time.sleep(0.0001)
            if not running:
                raise InterruptedError('Gold observation interrupted')

        warm_started = time.monotonic_ns()
        warm = invocations[0]
        publish(warm, range(1, rows // tile))
        warm_results = warm[2]['rmsnorm2']
        wait_for((warm_results[1, 0],))
        if any(result.ready for coordinate, result in warm_results.items() if coordinate[0] == 0):
            raise ArithmeticError('An unpublished input section produced an output')
        publish(warm, (0,))
        wait_for(tuple(warm_results.values()))
        print(json.dumps(dict(event='warmup', elapsed_ms=(time.monotonic_ns() - warm_started) / 1e6)), flush=True)

        batch_started = time.monotonic_ns()
        for run, invocation in enumerate(invocations[1:], 1):
            publish(invocation, range(1 if run == 1 else 0, rows // tile))
        delayed = invocations[1][2]['rmsnorm2']
        wait_for((delayed[1, 0],))
        first_ms = (time.monotonic_ns() - batch_started) / 1e6
        if any(result.ready for coordinate, result in delayed.items() if coordinate[0] == 0):
            raise ArithmeticError('An unpublished input section produced an output')
        publish(invocations[1], (0,))
        completed = {}
        deadline = time.monotonic_ns() + 60_000_000_000
        while running and len(completed) < args.runs:
            for run, (_, _, probes) in enumerate(invocations[1:], 1):
                if run not in completed and all(result.ready for result in probes['rmsnorm2'].values()):
                    completed[run] = (time.monotonic_ns() - batch_started) / 1e6
            if time.monotonic_ns() > deadline:
                raise TimeoutError(f'Batch stalled: {program.report}')
            if len(completed) < args.runs:
                time.sleep(0.0001)
        if not running:
            return
        batch_ms = (time.monotonic_ns() - batch_started) / 1e6
        errors = []
        for run, (data, _, probes) in enumerate(invocations):
            expected = reference_norm(reference_ffn(data, first_up, first_down), gamma)
            for table, index in zip(tables, ids):
                expected = (expected.astype(np.float64) + table[index[:, 0]]).astype(expected.dtype)
            expected = reference_norm(reference_ffn((expected,), second_up, second_down), gamma)
            error = 0.0
            for (row, column), result in probes['rmsnorm2'].items():
                r, c = row * tile, column * args.tile_columns
                part = expected[r:r + result.ref.shape[0], c:c + result.ref.shape[1]]
                error = max(error, float(np.max(np.abs(result.array - part))))
                if not np.allclose(result.array, part, atol=3e-3 if dtype == np.float16 else 3e-4, rtol=3e-3 if dtype == np.float16 else 3e-4):
                    raise ArithmeticError(f'Gold chain numerical mismatch: {error}')
            errors.append(error)
            print(json.dumps(dict(event='gold', run=run, complete_ms=completed.get(run),
                max_absolute_error=error, transport_queue=0, backend=args.backend, local=args.local)), flush=True)
            for stage in probes.values():
                for result in stage.values():
                    result.consume()
        wait_for((precision, strided, indexed, *mapped_results))
        if not precision.ready or not np.array_equal(precision.array, np.array([[2], [0]], dtype=dtype)):
            raise ArithmeticError('Contraction lost cancellation across K panels')
        print(json.dumps(dict(event='precision', dtype=args.dtype, result=precision.array.tolist())), flush=True)
        precision.consume()
        strided_expected = strided_left.astype(np.float32).T @ strided_right.astype(np.float32).T
        if not strided.ready or not np.array_equal(strided.array, strided_expected):
            raise ArithmeticError(f'Strided contraction mismatch: ready={strided.ready}, actual={strided.array.tolist() if strided.ready else None}, expected={strided_expected.tolist()}')
        print(json.dumps(dict(event='strided_contraction', result=strided.array.tolist())), flush=True)
        strided.consume()
        for i, result in enumerate(mapped_results):
            if not np.array_equal(result.array, strided_expected[2-i:3-i]):
                raise ArithmeticError('Contraction ignored its input index map')
        print(json.dumps(dict(event='mapped_contraction', result=[result.array.tolist() for result in mapped_results])), flush=True)
        for result in mapped_results:
            result.consume()
        indexed_expected = np.stack((2 * table_data[2], np.ones(4, dtype=dtype)))
        if not indexed.ready or not np.array_equal(indexed.array, indexed_expected):
            raise ArithmeticError('Indexed expression lost integer identity or masked access semantics')
        print(json.dumps(dict(event='indexed', result=indexed.array.tolist())), flush=True)
        indexed.consume()
        if deferred_output.ready:
            raise ArithmeticError('Missing input produced an output')
        with program.write(deferred_input[0, 0]) as destination:
            destination[...] = 2
        wait_for((deferred_output,))
        if not np.array_equal(deferred_output.array, np.full((2, 4), 6, dtype=dtype)):
            raise ArithmeticError('Independent expression output differs')
        deferred_output.consume()
        for generation in range(2):
            wait_for((streamed_table[1, 0],), 'writable')
            with program.write(streamed_table[1, 0]) as destination:
                destination[...] = generation + 2
            if generation:
                with program.write(streamed_table[0, 0]) as destination:
                    destination[...] = 0
            wait_for((streamed_result,))
            if not np.array_equal(streamed_result.array, np.full((2, 4), 2 * (generation + 2), dtype=dtype)):
                raise ArithmeticError('Selected source occurrence was lost during indexed reuse')
            if not generation and streamed_table[0, 0].present:
                raise ArithmeticError('Unrelated table source was unexpectedly published')
            print(json.dumps(dict(event='dynamic_indexed', generation=generation,
                unrelated_source_present=streamed_table[0, 0].present,
                result=streamed_result.array.tolist())), flush=True)
            if not generation:
                streamed_result.consume()
        if xonotic_case is not None:
            x_source, x_indices, x_tail, observations, generations = xonotic_case
            for generation, (source_values, index_values, tail_values, expected) in enumerate(generations):
                wait_for((*x_source.blocks.values(), *x_indices.blocks.values(), x_tail[0, 0]), 'writable')
                early_source = 1 - generation
                with program.write(x_source[early_source, 0]) as destination:
                    destination[...] = source_values[2*early_source:2*early_source+2]
                with program.write(x_indices[0, 0]) as destination:
                    destination[...] = index_values[:2]
                with program.write(x_tail[0, 0]) as destination:
                    destination[...] = tail_values
                wait_for((observations[0], observations[2]))
                if observations[1].ready:
                    raise ArithmeticError(f'Xonotic output consumed an unpublished occurrence: generation={generation}')
                if not x_source[1-early_source, 0].writable or not x_indices[0, 1].writable:
                    raise ArithmeticError(f'Xonotic withheld input occurrence was claimed: generation={generation}')
                for index in (0, 2):
                    if not np.array_equal(observations[index].array, expected[2*index:2*index+2]):
                        raise ArithmeticError('Xonotic early gather/concatenate output differs')
                print(json.dumps(dict(event='xonotic_indexed_early', generation=generation,
                    withheld_source_block=1-early_source, withheld_index_block=1,
                    withheld_source_writable=x_source[1-early_source, 0].writable,
                    withheld_index_writable=x_indices[0, 1].writable,
                    output=[observations[index].array.tolist() for index in (0, 2)])), flush=True)
                with program.write(x_source[1-early_source, 0]) as destination:
                    destination[...] = source_values[2*(1-early_source):2*(1-early_source)+2]
                with program.write(x_indices[0, 1]) as destination:
                    destination[...] = index_values[2:]
                wait_for(observations)
                for index, result in enumerate(observations):
                    if not np.array_equal(result.array, expected[2*index:2*index+2]):
                        raise ArithmeticError('Xonotic gather/concatenate output differs after reuse')
                print(json.dumps(dict(event='xonotic_indexed_complete', generation=generation,
                    output=[result.array.tolist() for result in observations])), flush=True)
                for result in observations:
                    result.consume()
        for generation in range(2):
            routing = np.resize(np.array([0, 2, 0, 3], dtype=np.int64), updates_count).reshape(-1, 1)
            if destinations_count > 4:
                routing[4:last_start, 0] = 4 + np.arange(max(0, last_start-4)) % (destinations_count-4)
            routing[last_start:] = 2
            if generation:
                routing = np.where(routing == 0, 3, np.where(routing == 3, 0, routing))
                routing[1, 0] = 2**32 + 2
            update_values = np.arange(1+generation, updates_count+1+generation, dtype=dtype)[:, None]
            expected = np.zeros((destinations_count, 1), dtype=np.float32)
            selected = scatter_valid[:, 0] & (routing[:, 0] < destinations_count)
            np.add.at(expected, routing[selected, 0], update_values[selected].astype(np.float32)*(2+generation)+1)
            expected = (expected.astype(dtype)*2).astype(dtype)
            scatter_start = time.monotonic_ns()
            wait_for((*scatter_indices.blocks.values(), *scatter_updates.blocks.values(),
                      *scatter_factors.blocks.values()), 'writable')
            for i in range(last_chunk+1):
                with program.write(scatter_indices[i, 0]) as destination:
                    destination[...] = routing[update_tile*i:update_tile*(i+1)]
            for i in range(last_chunk):
                with program.write(scatter_factors[i, 0]) as destination:
                    destination[...] = 2 + generation
                with program.write(scatter_updates[i, 0]) as destination:
                    destination[...] = update_values[update_tile*i:update_tile*(i+1)]
            wait_for(tuple(scatter_results[i] for i in early_destinations))
            if scatter_results[2].ready or not scatter_updates[last_chunk, 0].writable:
                raise ArithmeticError('Delayed scatter contribution was not independent')
            first_scatter_ns = time.monotonic_ns() - scatter_start
            for i in early_destinations:
                if not np.array_equal(scatter_results[i].array, np.broadcast_to(expected[i], (1, 4))):
                    raise ArithmeticError('Early scattered sum or consumer differs')
            with program.write(scatter_updates[last_chunk, 0]) as destination:
                destination[...] = update_values[last_start:]
            if scatter_results[2].ready or not scatter_factors[last_chunk, 0].writable:
                raise ArithmeticError('Fused update ignored its missing coefficient operand')
            with program.write(scatter_factors[last_chunk, 0]) as destination:
                destination[...] = 2 + generation
            wait_for((scatter_results[2],))
            if not np.array_equal(scatter_results[2].array, np.broadcast_to(expected[2], (1, 4))):
                raise ArithmeticError('Duplicate or masked scatter contribution differs')
            print(json.dumps(dict(event='indexed_add', generation=generation, rows=updates_count, tile=update_tile, destinations=destinations_count, first_consumer_ns=first_scatter_ns,
                complete_ns=time.monotonic_ns()-scatter_start,
                delayed_destination=2, result=[result.array.tolist() for result in scatter_results])), flush=True)
            if not generation:
                for result in scatter_results:
                    result.consume()
        for generation in range(2):
            wait_for((fanout_source[0, 0],), 'writable')
            with program.write(fanout_source[0, 0]) as destination:
                destination[...] = generation + 2
            wait_for(tuple(fanout_results))
            for factor, result in enumerate(fanout_results, 1):
                if not np.array_equal(result.array, np.full((2, 4), (generation+2)*factor, dtype=dtype)):
                    raise ArithmeticError('Fanout consumed a stale source occurrence')
            print(json.dumps(dict(event='fanout', generation=generation, branches=len(fanout_results),
                remote=not args.local, first=fanout_results[0].array[0, 0].item(),
                last=fanout_results[-1].array[0, 0].item())), flush=True)
            if not generation:
                for result in fanout_results:
                    result.consume()
        if args.trace:
            Path(args.trace).write_text(json.dumps(dict(compute=program.trace, transfers=program.transfer_trace), indent=2) + '\n')
        report = program.report
        print(json.dumps(dict(event='summary', dtype=args.dtype, coreml=bool(args.coreml), invocations=args.runs, batch_ms=batch_ms,
            invocations_per_second=args.runs * 1000 / batch_ms, first_section_ms=first_ms,
            completion_ms=summary(tuple(completed.values())), withheld_invocation=1, withheld_section=0,
            max_absolute_error=max(errors), runtime={name: getattr(report, name) for name, _ in report._fields_})), flush=True)



if __name__ == '__main__':
    main()
