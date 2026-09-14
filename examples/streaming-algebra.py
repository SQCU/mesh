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
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--tile-rows', type=int, default=64)
    parser.add_argument('--tile-k', type=int, default=64)
    parser.add_argument('--tile-columns', type=int, default=64)
    parser.add_argument('--scatter-rows', type=int, default=6)
    parser.add_argument('--scatter-tile', type=int, default=2)
    parser.add_argument('--scatter-destinations', type=int, default=4)
    parser.add_argument('--trace')
    parser.add_argument('--xonotic', action='store_true')
    parser.add_argument('--coreml', nargs=3, metavar=('PYTHON', 'GENERATOR', 'CACHE'))
    args = parser.parse_args()
    if min(args.depth, args.runs, args.width, args.tile_rows, args.tile_k, args.tile_columns) < 1:
        parser.error('Chain depth, runs, width and tile sizes must be positive')
    if args.rows <= args.tile_rows:
        parser.error('Streaming chain requires at least two row sections')
    if args.runs < 1 or args.samples < 1:
        parser.error('Runs and samples must be positive')
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
    rows, width, tile = args.rows, args.width, args.tile_rows
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
            up = tuple(tuple((rng.standard_normal((width, width), dtype=np.float32) / np.float32(np.sqrt(2 * width))).astype(dtype)
                       for _ in range(partitions)) for _ in range(2))
            down = tuple((rng.standard_normal((width, width), dtype=np.float32) / np.float32(np.sqrt(2 * width))).astype(dtype) for _ in range(2))
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
        mapped_feature, mapped_inner = kernels.program_id(1) * 3 + kernels.arange(3).T, kernels.arange(5, tile=3)
        mapped_product = (mapped_left.at(0, mapped_inner) * mapped_right.at(mapped_inner, mapped_feature)).sum().T
        mapped = program.kernel_call(kernels.expression(mapped_product),
            grid=(3, 3), in_specs=(BlockSpec((1, 5), lambda i, j: (2-i, 0)), BlockSpec((5, 7), lambda i, j: (0, 0))),
            out_specs=BlockSpec((1, 3), lambda i, j: (i, j)),
            out_shape=ShapeDtypeStruct((3, 7), np.float32), peer=0)(weight(strided_left).T, weight(strided_right).T)
        mapped_results = tuple(program.export(ref) for _, ref in sorted(mapped.blocks.items()))
        composed_input = program.tensor((3, 5), block_shape=(1, 5), dtype=dtype)
        composed_bias = program.tensor((3, 7), block_shape=(1, 7), dtype=np.float32)
        composed_left, composed_right, bias = kernels.arguments(3)
        contraction = kernels.dot(composed_left, composed_right, tile_k=3)
        composed = program.kernel_call(kernels.expression(contraction * 2 + bias, contraction),
            grid=(3,), in_specs=(BlockSpec(None), BlockSpec(None), BlockSpec((1, 7), lambda i: (i, 0))),
            out_specs=(BlockSpec((1, 7), lambda i: (i, 0)), BlockSpec((1, 7), lambda i: (i, 0))),
            out_shape=(ShapeDtypeStruct((3, 7), np.float32), ShapeDtypeStruct((3, 7), np.float32)),
            peer=0)(composed_input, weight(strided_right).T, composed_bias)
        composed_results = tuple(tuple(program.export(tensor[i, 0]) for i in range(3)) for tensor in composed)
        nested_inputs = tuple(program.tensor(shape, block, dtype=dtype) for shape, block in
            (((3, 4), (1, 4)), ((4, 6), (4, 2)), ((6, 3), (2, 3))))
        nested_x, nested_up, nested_down = kernels.arguments(3)
        nested_hidden = kernels.dot(nested_x, nested_up, tile_k=2)
        activation = nested_hidden / (1 + (0 - nested_hidden).exp())
        nested_bodies = tuple(kernels.dot(value, nested_down, tile_k=2)
            for value in (activation, activation.astype(np.float16)))
        nested = program.kernel_call(kernels.expression(*nested_bodies), grid=(3,),
            in_specs=(BlockSpec(None),) * 3, out_specs=(BlockSpec((1, 3), lambda i: (i, 0)),) * 2,
            out_shape=(ShapeDtypeStruct((3, 3), np.float32),) * 2, peer=0)(*nested_inputs)
        nested_results = tuple(tuple(program.export(tensor[i, 0]) for i in range(3)) for tensor in nested)
        cast_arg, = kernels.arguments(1)
        cast_boundary = program.kernel_call(kernels.expression((cast_arg + 1).astype(np.float16) - cast_arg),
            grid=(1,), in_specs=(BlockSpec((1, 3), lambda i: (0, 0)),),
            out_specs=BlockSpec((1, 3), lambda i: (0, 0)), out_shape=ShapeDtypeStruct((1, 3), np.float32),
            peer=0)(weight(np.array([[4096, -4096, 1]], dtype=np.float32)))
        cast_result = program.export(cast_boundary[0, 0])
        cast_sums = program.kernel_call(kernels.expression(cast_arg.sum().astype(np.int64), cast_arg.astype(np.int64).sum()),
            grid=(1,), in_specs=(BlockSpec(None),), out_specs=(BlockSpec((1, 1), lambda i: (0, 0)),) * 2,
            out_shape=(ShapeDtypeStruct((1, 1), np.int64),) * 2, peer=0)(weight(np.array([[0.75, 0.75]], dtype=np.float32)))
        cast_sum_results = tuple(program.export(tensor[0, 0]) for tensor in cast_sums)
        mixed_values = tuple((np.arange(size, dtype=np.float32).reshape(shape) / 128 - 0.125)
            for size, shape in ((15, (3, 5)), (35, (5, 7))))
        mixed_left, mixed_right = kernels.arguments(2)
        mixed_output = program.kernel_call(kernels.expression(
            kernels.dot(mixed_left.astype(np.float16), mixed_right, tile_k=3),
            kernels.dot(mixed_left, mixed_right.astype(np.float16), tile_k=3)),
            grid=(1,), in_specs=(BlockSpec(None),) * 2,
            out_specs=(BlockSpec((3, 7), lambda i: (0, 0)),) * 2,
            out_shape=(ShapeDtypeStruct((3, 7), np.float32),) * 2, peer=0)(*(weight(value) for value in mixed_values))
        mixed_results = tuple(program.export(tensor[0, 0]) for tensor in mixed_output)
        norm_input = program.tensor((3, 6), (1, 2), dtype=dtype)
        norm_gamma = np.ones((1, 6), dtype=dtype)
        norm_output = rmsnorm(program, norm_input, weight(norm_gamma), tile_rows=1) if args.rank == 0 else program.tensor((3, 6), (1, 2), dtype=dtype)
        norm_results = {coordinate: program.export(ref) for coordinate, ref in norm_output.blocks.items()}
        integer_values = np.array([[2**53+65535, 1, -(2**53), 0, 0, 0],
            [2**53+2**32-1, 3, -(2**53), 0, 0, 0],
            [2**63-1, 1, 0, 0, 0, 0], [-(2**63), 0, 0, -1, 0, 0]], dtype=np.int64)
        integer_input = program.tensor(integer_values.shape, (1, 3), dtype=np.int64)
        integer_arg, = kernels.arguments(1)
        integer_sum = program.kernel_call(kernels.expression(integer_arg.sum()), grid=(4,),
            in_specs=(BlockSpec(None),), out_specs=BlockSpec((1, 1), lambda i: (i, 0)),
            out_shape=ShapeDtypeStruct((4, 1), np.int64), peer=0)(integer_input)
        integer_results = tuple(program.export(integer_sum[i, 0]) for i in range(4))
        integer_total_result = None
        quotient_cases = []
        for scalar, numerators, divisors in (
                (np.int64, (2**53+1, -7, 7, -7, -(2**63)), (3, 3, -3, -3, 3)),
                (np.uint64, (2**64-1, 2**63+1, 2**53+1), (3, 7, 2))):
            numerator, divisor = kernels.arguments(2)
            shape = (1, len(numerators))
            quotient = program.kernel_call(kernels.expression(numerator // divisor, numerator % divisor),
                grid=(1,), in_specs=(BlockSpec(shape, lambda i: (0, 0)),) * 2,
                out_specs=(BlockSpec(shape, lambda i: (0, 0)),) * 2,
                out_shape=(ShapeDtypeStruct(shape, scalar),) * 2, peer=0)(
                    weight(np.array([numerators], dtype=scalar)), weight(np.array([divisors], dtype=scalar)))
            quotient_cases.append((tuple(program.export(tensor[0, 0]) for tensor in quotient),
                [[a // b for a, b in zip(numerators, divisors)]], [[a % b for a, b in zip(numerators, divisors)]]))
        norm_generations = tuple((values, reference_norm(values, norm_gamma)) for values in
            ((np.arange(1, 19, dtype=np.float32).reshape(3, 6) / 16 + generation / 8).astype(dtype)
             for generation in range(2)))
        nested_generations = []
        for generation in range(2):
            values = tuple(((np.arange(np.prod(tensor.shape), dtype=np.float32).reshape(tensor.shape)
                + generation) / 32 - 0.25).astype(dtype) for tensor in nested_inputs)
            hidden = values[0].astype(np.float64) @ values[1].astype(np.float64)
            activated = hidden / (1 + np.exp(-hidden))
            expected = tuple(value @ values[2].astype(np.float64) for value in
                (activated, activated.astype(np.float16).astype(np.float64)))
            nested_generations.append((values, expected))
        table_arg, index_arg, deferred_arg = kernels.arguments(3)
        column_arg = kernels.arange(4)
        index_data = np.array([[2], [2**53 + 1]], dtype=np.int64)
        table_data = np.arange(12, dtype=dtype).reshape(3, 4)
        indexed_body = 2 * table_arg.at(index_arg, column_arg, mask=(index_arg >= 0) & (index_arg < 3)) + kernels.select(index_arg.equal(2**53 + 1), 1, 0)
        deferred_input = program.tensor((2, 4), dtype=dtype)
        logical_bounds = table_arg.reshape((3, 2, 2)).at(0, 0, column_arg, other=-1)
        indexed_outputs = program.kernel_call(kernels.expression(indexed_body, deferred_arg * 3, logical_bounds), grid=(1,),
            in_specs=(BlockSpec((3, 4), lambda i: (0, 0)), BlockSpec((2, 1), lambda i: (0, 0)),
                      BlockSpec((2, 4), lambda i: (0, 0))),
            out_specs=(BlockSpec((2, 4), lambda i: (0, 0)),) * 3,
            out_shape=(ShapeDtypeStruct((2, 4), dtype),) * 3, peer=0)(
                weight(table_data), weight(index_data), deferred_input)
        indexed, deferred_output, logical_bounds_result = (program.export(value[0, 0]) for value in indexed_outputs)
        streamed_table = program.tensor((4, 4), (2, 4), dtype=dtype)
        streamed_indices = weight(np.array([[2], [3]], dtype=np.int64))
        source_arg, selected_arg = kernels.arguments(2)
        streamed_rows = selected_arg.at(kernels.arange(2).T, 0)
        streamed_outputs = program.kernel_call(
            kernels.expression(2 * source_arg.at(streamed_rows, column_arg),
                source_arg.at(2, column_arg) + source_arg.at(selected_arg, column_arg)), grid=(1,),
            in_specs=(BlockSpec(None), BlockSpec((2, 1), lambda i: (0, 0))),
            out_specs=(BlockSpec((2, 4), lambda i: (0, 0)),) * 2,
            out_shape=(ShapeDtypeStruct((2, 4), dtype),) * 2, peer=0)(streamed_table, streamed_indices)
        streamed_results = tuple(program.export(tensor[0, 0]) for tensor in streamed_outputs)
        scatter_indices = program.tensor((updates_count, 1), (update_tile, 1), dtype=np.int64)
        scatter_updates = program.tensor((updates_count, 4), (update_tile, 4), dtype=dtype)
        scatter_factors = program.tensor((updates_count, 1), (update_tile, 1), dtype=np.float32)
        scatter_lookup = program.tensor((updates_count, 1), (update_tile, 1), dtype=np.int64)
        scatter_masks = program.tensor((updates_count, 1), (update_tile, 1), dtype=bool)
        scatter_valid = np.ones((updates_count, 1), dtype=bool)
        scatter_valid[-1 if updates_count-last_start > 1 else -2, 0] = False
        base_arg, destination_arg, update_arg, mask_arg, factor_arg, lookup_arg = kernels.arguments(6)
        update_row = kernels.arange(updates_count).T
        update_column = kernels.arange(4)
        scatter = program.kernel_call(kernels.expression(kernels.indexed_add(
            base_arg, destination_arg, update_arg.at(lookup_arg.at(update_row, 0), update_column) * factor_arg + 1,
            mask=mask_arg.at(update_row, 0))), grid=(destinations_count,),
            in_specs=(BlockSpec(None),) * 6, out_specs=BlockSpec((1, 4), lambda i: (i, 0)),
            out_shape=ShapeDtypeStruct((destinations_count, 4), dtype), peer=0)(
                weight(np.zeros((destinations_count, 4), dtype=dtype)), scatter_indices, scatter_updates,
                scatter_masks, scatter_factors, scatter_lookup)
        consumer_arg, = kernels.arguments(1)
        scatter_consumed = program.kernel_call(kernels.expression(consumer_arg * 2), grid=(destinations_count,),
            in_specs=(BlockSpec((1, 4), lambda i: (i, 0)),),
            out_specs=BlockSpec((1, 4), lambda i: (i, 0)),
            out_shape=ShapeDtypeStruct((destinations_count, 4), dtype), peer=0)(scatter)
        scatter_results = tuple(program.export(scatter_consumed[i, 0]) for i in range(destinations_count))
        xonotic_case = None
        xonotic_gradients = []
        xonotic_neighborhoods = []
        xonotic_expert = None
        xonotic_batched = None
        xonotic_boolean = None
        xonotic_reductions = None
        xonotic_elementary = None
        xonotic_ranges = None
        xonotic_random = None
        xonotic_integer_dots = []
        xonotic_ordering = []
        xonotic_composed_indexed = []
        xonotic_indexed_context = None
        xonotic_grouped_outer = None
        scatter_bases = {}
        xonotic_take_gradient = None
        xonotic_alias = None
        if args.xonotic:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'xonotic'))
            from solver.strat import tensor as mx
            from solver.strat.tensor_metal import kernel_calls
        if args.xonotic and args.rank == 0:
            x_source = program.tensor((4, 4), (2, 4), dtype=np.float32)
            x_indices = program.tensor((1, 4), (1, 2), dtype=np.int64)
            x_tail = program.tensor((2, 4), dtype=np.float32)
            integer_graph = mx.Graph()
            with integer_graph:
                integer_source = integer_graph.input('integer_rows', (4, 6), 'int64')
                integer_rows = mx.sum(integer_source, axis=1)
                integer_total = mx.sum(integer_source)
                integer_transposed = integer_graph.input('integer_columns', (6, 4), 'int64')
                integer_columns = mx.sum(integer_transposed, axis=0, keepdims=True)
            integer_lowered = kernel_calls(program, integer_graph, (),
                {integer_source.index: integer_input, integer_transposed.index: integer_input.T},
                outputs=(integer_rows, integer_columns, integer_total),
                root_peer=0, tile_rows=1, tile_columns=3)
            integer_results += tuple(program.export(ref)
                for value in (integer_rows, integer_columns)
                for _, ref in sorted(integer_lowered[value.index].blocks.items()))
            integer_total_result = program.export(integer_lowered[integer_total.index][0, 0])
            graph = mx.Graph()
            with graph:
                source = graph.input('source', (4, 4))
                indices = graph.input('indices', (4,), 'int64')
                tail = graph.input('tail', (2, 4))
                selected = source[indices, ::-1]
                joined = mx.concatenate((selected, tail), axis=0)
                means = mx.mean(source, axis=1)
                total_mean = mx.mean(source) * 4
                column_means = mx.mean(source, axis=0, keepdims=True)
                logical_indices = indices.reshape(2, 2, 1)
                taken = mx.take_along_axis(selected.reshape(2, 2, 4), logical_indices, axis=2)
                broadcast_taken = mx.take_along_axis(selected[:2].reshape(1, 2, 4), logical_indices, axis=2)
                transposed = selected[:, :, None].transpose(0, 2, 1)
                rank_joined = mx.concatenate((transposed, transposed), axis=1)
                rank_joined = rank_joined * graph.constant(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 4)) + 1
                reshaped = selected.reshape(2, 4, 2)[:, :, ::-1]
            logical_values = dict(take=taken, broadcast_take=broadcast_taken,
                transpose=transposed, concatenate=rank_joined, reshape_gather=reshaped)
            lowered = kernel_calls(program, graph, (),
                {source.index: x_source, indices.index: x_indices, tail.index: x_tail},
                outputs=(joined, means, total_mean, column_means, *logical_values.values()), root_peer=0, tile_rows=2, tile_columns=4)
            observations = tuple(program.export(lowered[joined.index][i, 0]) for i in range(3))
            mean_results = tuple(program.export(lowered[means.index][i, 0]) for i in range(2))
            total_result = program.export(lowered[total_mean.index][0, 0])
            column_mean_results = tuple(program.export(ref)
                for _, ref in sorted(lowered[column_means.index].blocks.items()))
            logical_results = {name: tuple(program.export(ref) for _, ref in sorted(lowered[value.index].blocks.items()))
                for name, value in logical_values.items()}
            bindings = {}
            for name, references in (
                    ('source', tuple(sorted(x_source.blocks.items()))),
                    ('indices', tuple(sorted(x_indices.blocks.items()))),
                    ('tail', tuple(sorted(x_tail.blocks.items()))),
                    ('gathered', tuple(sorted(lowered[selected.index].blocks.items()))),
                    ('output', tuple(((i, 0), result.ref) for i, result in enumerate(observations))),
                    *((name, tuple(sorted(lowered[value.index].blocks.items()))) for name, value in logical_values.items())):
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
            gradient_indices = program.tensor((1, 6), (1, 2), dtype=np.int64)
            cotangents = program.tensor((6, 4), (2, 4), dtype=np.float32)
            graph = mx.Graph()
            with graph:
                primal = graph.input('primal', (4, 4))
                indices = graph.input('indices', (6,), 'int64')
                cotangent = graph.input('cotangent', (6, 4))
                selected = primal[indices, :]
                gradient, = graph.vjp((selected,), (cotangent,), (primal,))
            lowered = kernel_calls(program, graph, (),
                {indices.index: gradient_indices, cotangent.index: cotangents},
                outputs=(gradient,), root_peer=0, tile_rows=1, tile_columns=4)
            gradient_results = tuple(program.export(lowered[gradient.index][i, 0]) for i in range(4))
            generations = []
            for generation in range(2):
                index_values = np.array([0, 0, 3, 3, 2, -2] if not generation else [3, 3, 0, 0, 2, -2], dtype=np.int64)
                cotangent_values = np.arange(1 + generation, 25 + generation, dtype=np.float32).reshape(6, 4)
                expected = np.zeros((4, 4), dtype=np.float64)
                np.add.at(expected, index_values % 4, cotangent_values.astype(np.float64))
                generations.append(((index_values,), cotangent_values, expected.reshape(4, 1, 4)))
            xonotic_gradients.append(('row', (gradient_indices,), cotangents, gradient_results, generations))
            row_indices = program.tensor((1, 4), (1, 2), dtype=np.int64)
            column_indices = program.tensor((1, 4), (1, 2), dtype=np.int64)
            gather_cotangents = program.tensor((4, 2), (1, 2), dtype=np.float32)
            graph = mx.Graph()
            with graph:
                primal = graph.input('primal', (4, 2, 3))
                rows_arg = graph.input('rows', (4,), 'int64')
                columns_arg = graph.input('columns', (4,), 'int64')
                cotangent = graph.input('cotangent', (4, 2))
                selected = primal[rows_arg, ::-1, columns_arg]
                gradient, = graph.vjp((selected,), (cotangent,), (primal,))
                transformed = gradient * 3 + 1
            lowered = kernel_calls(program, graph, (),
                {rows_arg.index: row_indices, columns_arg.index: column_indices, cotangent.index: gather_cotangents},
                outputs=(transformed,), root_peer=0, tile_rows=2, tile_columns=3)
            gather_results = tuple(program.export(ref) for _, ref in sorted(lowered[transformed.index].blocks.items()))
            generations = []
            for generation in range(2):
                rows_values = np.array([0, 0, 2, 2] if not generation else [3, 3, 2, 2], dtype=np.int64)
                column_values = np.array([1, 1, -1, -1] if not generation else [-1, -1, 3, 0], dtype=np.int64)
                cotangent_values = np.arange(1 + generation, 9 + generation, dtype=np.float32).reshape(4, 2)
                expected = np.zeros((4, 2, 3), dtype=np.float64)
                normalized = np.where(column_values < 0, column_values + 3, column_values)
                valid = (normalized >= 0) & (normalized < 3)
                np.add.at(expected, (rows_values[valid, None], np.array([1, 0])[None, :], normalized[valid, None]), cotangent_values[valid])
                generations.append(((rows_values, column_values), cotangent_values, expected * 3 + 1))
            xonotic_gradients.append(('advanced', (row_indices, column_indices), gather_cotangents, gather_results, generations))
            for name, base_shape, index_shape, update_shape, row_tile in (
                    ('matrix_scatter', (4, 4), (6,), (6, 1), 1),
                    ('rank_scatter', (4, 2, 3), (2, 3), (2, 3, 1, 1), 2)):
                index_storage = (1, 6) if len(index_shape) == 1 else index_shape
                index_block = (1, 2) if len(index_shape) == 1 else (1, 3)
                matrix_indices = program.tensor(index_storage, index_block, dtype=np.int64)
                matrix_updates = program.tensor((6, 1), (2, 1), dtype=np.float32)
                matrix_base = np.arange(np.prod(base_shape), dtype=np.float32).reshape(base_shape)
                supplied = {}
                if name == 'rank_scatter':
                    backing = program.tensor((4, 6), (1, 2), dtype=np.float32)
                    scatter_bases[name] = backing, matrix_base.reshape(4, 6)
                graph = mx.Graph()
                with graph:
                    if name == 'rank_scatter':
                        source_base = graph.input('base', (4, 6))
                        supplied[source_base.index] = backing
                        base_value = source_base.reshape(base_shape)
                    else:
                        base_value = graph.constant(matrix_base)
                    indices = graph.input('indices', index_shape, 'int64')
                    updates = graph.input('updates', update_shape)
                    scattered = base_value.at[indices].add(updates * 2 + 1)
                    transformed = scattered * 3
                lowered = kernel_calls(program, graph, (),
                    {**supplied, indices.index: matrix_indices, updates.index: matrix_updates},
                    outputs=(transformed,), root_peer=0, tile_rows=row_tile, tile_columns=base_shape[-1])
                matrix_results = tuple(program.export(ref) for _, ref in sorted(lowered[transformed.index].blocks.items()))
                generations = []
                for generation in range(2):
                    index_values = np.array([0, 0, 3, 3, 2, -2] if not generation else [3, 3, 0, 0, 2, -2], dtype=np.int64)
                    update_values = np.arange(1 + generation, 7 + generation, dtype=np.float32).reshape(6, 1)
                    expected = matrix_base.astype(np.float64)
                    expanded = np.broadcast_to((update_values * 2 + 1).reshape((6,) + (1,) * (len(base_shape)-1)), (6, *base_shape[1:]))
                    np.add.at(expected, index_values % 4, expanded)
                    generations.append(((index_values,), update_values, (expected * 3).reshape(4, -1, base_shape[-1])))
                xonotic_gradients.append((name, (matrix_indices,), matrix_updates, matrix_results, generations))
            # design/algorithm-sources.md#xonotic-neighborhood-algebra
            for gram in (False, True):
                graph = mx.Graph()
                with graph:
                    operands = tuple(graph.input(name, shape, dtype) for name, shape, dtype in (
                        ('query', (4, 3), 'float32'), ('keys', (4, 3), 'float32'),
                        ('vectors', (4, 3), 'float32'), ('indices', (4, 2), 'int64'),
                        ('weights', (4, 2), 'float32'), ('cotangent', (4, 3), 'float32')))
                    neighborhood = mx.neighborhood(*operands[:5], gram=gram)
                    derivatives = graph.vjp((neighborhood,), (operands[5],), tuple(operands[i] for i in (0, 1, 2, 4)))
                    consumers = tuple(result * 2 + 1 for result in (neighborhood, *derivatives))
                storage = tuple(program.tensor(value.shape, (1, value.shape[1]), dtype=value.dtype) for value in operands)
                lowered = kernel_calls(program, graph, (), dict(zip((value.index for value in operands), storage)),
                    outputs=consumers, root_peer=0, tile_rows=1, tile_columns=2)
                observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                    for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in consumers)
                generations = []
                for generation in range(2):
                    query = (np.arange(12, dtype=np.float32).reshape(4, 3) - 4 + generation) / 8
                    keys = (np.arange(12, dtype=np.float32).reshape(4, 3)[::-1] + 1 - generation) / 16
                    vectors = (np.arange(12, dtype=np.float32).reshape(4, 3) - 6 - generation) / 4
                    indices = np.array([[0, 0], [3, 1], [2, 2], [1, 3]] if not generation else
                                       [[1, 1], [0, 3], [2, 2], [3, 0]], dtype=np.int64)
                    weights = (np.arange(8, dtype=np.float32).reshape(4, 2) - 2 + generation) / 8
                    cotangent = (np.arange(12, dtype=np.float32).reshape(4, 3) + 2 + generation) / 16
                    q, k, v, w, g = (data.astype(np.float64) for data in (query, keys, vectors, weights, cotangent))
                    affinity = np.sum(q[:, None, :] * k[indices], axis=-1) / np.sqrt(3) if gram else np.ones((4, 2))
                    dot_v = np.sum(g[:, None, :] * v[indices], axis=-1)
                    result = np.sum((w * affinity)[..., None] * v[indices], axis=1)
                    dq, dk, dv = (np.zeros((4, 3), dtype=np.float64) for _ in range(3))
                    np.add.at(dv, indices, (w * affinity)[..., None] * g[:, None, :])
                    if gram:
                        dq = np.sum((w * dot_v / np.sqrt(3))[..., None] * k[indices], axis=1)
                        np.add.at(dk, indices, (w * dot_v / np.sqrt(3))[..., None] * q[:, None, :])
                    expected = tuple(data * 2 + 1 for data in (result, dq, dk, dv, dot_v * affinity))
                    generations.append(((query, keys, vectors, indices, weights, cotangent), expected))
                xonotic_neighborhoods.append((gram, storage, observations, generations))
            # design/algorithm-sources.md#xonotic-expert-indexed-contractions
            graph = mx.Graph()
            with graph:
                expert_inputs = tuple(graph.input(name, shape, dtype) for name, shape, dtype in (
                    ('rows', (4, 3), 'float32'), ('weights', (3, 3, 5), 'float32'),
                    ('selected', (4,), 'int64'), ('cotangent', (4, 5), 'float32')))
                projected = mx.expert_matmul(*expert_inputs[:3])
                derivatives = graph.vjp((projected,), (expert_inputs[3],), expert_inputs[:2])
                consumers = tuple(value * 2 + 1 for value in (projected, *derivatives))
            expert_storage = tuple(program.tensor(shape, block, dtype=dtype) for shape, block, dtype in (
                ((4, 3), (1, 2), np.float32), ((9, 5), (2, 2), np.float32),
                ((4, 1), (1, 1), np.int64), ((4, 5), (1, 2), np.float32)))
            lowered = kernel_calls(program, graph, (), dict(zip((value.index for value in expert_inputs), expert_storage)),
                outputs=consumers, root_peer=0, tile_rows=1, tile_k=2, tile_columns=2)
            observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in consumers)
            generations = []
            for generation in range(2):
                rows_values = (np.arange(12, dtype=np.float32).reshape(4, 3) - 5 + generation) / 8
                weight_values = (np.arange(45, dtype=np.float32).reshape(3, 3, 5) - 20 - generation) / 16
                selected_values = np.array([3, generation - 3, 2, generation], dtype=np.int64)
                rows_values[0, 0] = np.nan
                cotangent_values = (np.arange(20, dtype=np.float32).reshape(4, 5) + 1 + generation) / 8
                x, w, g = (value.astype(np.float64) for value in (rows_values, weight_values, cotangent_values))
                normalized = np.where(selected_values < 0, selected_values + w.shape[0], selected_values)
                valid = (normalized >= 0) & (normalized < w.shape[0])
                selected_weights = np.zeros((len(x), *w.shape[1:]), dtype=np.float64)
                selected_weights[valid] = w[normalized[valid]]
                projected = np.einsum('nd,ndh->nh', x, selected_weights)
                dx = np.einsum('nh,ndh->nd', g, selected_weights)
                dw = np.zeros_like(w)
                np.add.at(dw, normalized[valid], x[valid, :, None] * g[valid, None, :])
                generations.append(((rows_values, weight_values.reshape(9, 5), selected_values.reshape(4, 1), cotangent_values),
                                    tuple(value * 2 + 1 for value in (projected, dx, dw.reshape(9, 5)))))
            xonotic_expert = expert_storage, observations, generations
            # design/algorithm-sources.md#xonotic-batched-contractions
            graph = mx.Graph()
            with graph:
                batch_inputs = tuple(graph.input(name, shape, 'float32') for name, shape in (
                    ('left', (2, 5, 3)), ('right', (1, 7, 5)), ('cotangent', (2, 3, 7))))
                product = mx.matmul(*batch_inputs[:2], transpose_left=True, transpose_right=True)
                derivatives = graph.vjp((product,), (batch_inputs[2],), batch_inputs[:2])
                consumers = tuple(value * 2 + 1 for value in (product, *derivatives))
            batch_storage = tuple(program.tensor(shape, (1, 2), dtype=np.float32)
                                  for shape in ((10, 3), (7, 5), (6, 7)))
            lowered = kernel_calls(program, graph, (), dict(zip((value.index for value in batch_inputs), batch_storage)),
                outputs=consumers, root_peer=0, tile_rows=1, tile_k=2, tile_columns=2)
            observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in consumers)
            generations = []
            for generation in range(2):
                left = (np.arange(30, dtype=np.float32).reshape(2, 5, 3) - 13 + generation) / 8
                right = (np.arange(35, dtype=np.float32).reshape(1, 7, 5) - 17 - generation) / 16
                cotangent = (np.arange(42, dtype=np.float32).reshape(2, 3, 7) - 19 + generation) / 8
                l, r, g = (value.astype(np.float64) for value in (left, right, cotangent))
                product = np.matmul(l.swapaxes(-1, -2), r.swapaxes(-1, -2))
                dl = np.matmul(g, r).swapaxes(-1, -2)
                dr = np.matmul(g.swapaxes(-1, -2), l.swapaxes(-1, -2)).sum(axis=0, keepdims=True)
                expected = tuple((value * 2 + 1).reshape(shape) for value, shape in
                                 zip((product, dl, dr), ((6, 7), (10, 3), (7, 5))))
                generations.append(((left.reshape(10, 3), right.reshape(7, 5), cotangent.reshape(6, 7)), expected))
            xonotic_batched = batch_storage, observations, generations
            # design/algorithm-sources.md#xonotic-logical-pointwise
            graph = mx.Graph()
            with graph:
                truth_input = graph.input('truth_values', (2, 1, 3), 'float32')
                expanded = mx.broadcast_to(truth_input, (2, 2, 3))
                matrix_values = np.array([[0, 1, -2], [3, 0, np.nan]], dtype=np.float32)
                matrix = mx.broadcast_to(graph.constant(matrix_values), (2, 2, 3))
                scalar = mx.broadcast_to(graph.constant(np.float32(2)), (2, 2, 3))
                conjunction = mx.elementwise('logical_and', expanded, matrix)
                complement = mx.elementwise('logical_and', mx.elementwise('logical_not', expanded), scalar)
                mask = mx.elementwise('logical_or', conjunction, complement)
                direct_values = truth_input.reshape(2, 3)
                direct_scalar = mx.broadcast_to(graph.constant(np.float32(0)), (2, 3))
                direct = mx.elementwise('logical_or', mx.elementwise('logical_not', direct_values),
                    mx.elementwise('logical_and', direct_values, direct_scalar))
                consumer = mask.astype('float32') * 3 + 1 + direct.astype('float32').reshape(2, 1, 3)
            truth_storage = program.tensor((2, 3), (1, 3), dtype=np.float32)
            lowered = kernel_calls(program, graph, (), {truth_input.index: truth_storage},
                outputs=(consumer,), root_peer=0, tile_rows=1, tile_columns=2)
            result = lowered[consumer.index]
            observations = tuple((i * result.block_shape[0], j * result.block_shape[1], program.export(ref))
                                 for (i, j), ref in sorted(result.blocks.items()))
            generations = []
            for generation in range(2):
                data = np.array([[np.nan, 0, -2], [-0.0, 5, 0]] if not generation else
                                [[0, -3, np.nan], [4, 0, -0.0]], dtype=np.float32)
                expanded = np.broadcast_to(data[:, None, :], (2, 2, 3))
                expected = (np.logical_or(np.logical_and(expanded, matrix_values),
                                         np.logical_and(np.logical_not(expanded), 2)).astype(np.float32) * 3 + 1
                            + np.logical_or(np.logical_not(data), np.logical_and(data, 0))[:, None, :]).reshape(4, 3)
                generations.append((data, expected))
            xonotic_boolean = truth_storage, observations, generations
            # design/algorithm-sources.md#shared-associative-reductions
            graph = mx.Graph()
            with graph:
                reduction_inputs = tuple(graph.input(name, (2, 5), dtype) for name, dtype in
                                         (('signed', 'int64'), ('unsigned', 'uint64'), ('truth', 'float32'), ('extrema', 'float32'), ('fractional', 'float32')))
                cases, consumers = [], []
                for rank, shape, axis in ((1, (10,), 0), (2, (2, 5), 1), (3, (2, 1, 5), 2)):
                    for operand, operation in ((0, 'max'), (0, 'min'), (1, 'max'), (1, 'min'), (2, 'any'), (2, 'all'), (3, 'max'), (3, 'min')):
                        reduced = mx.reduce(operation, reduction_inputs[operand].reshape(shape), axis=axis, keepdims=True)
                        consumers.append(reduced.astype('int32') * 2 + 1 if operand == 2 else reduced + 0)
                        cases.append((rank, shape, axis, operand, operation))
            reduction_storage = tuple(program.tensor((2, 5), (1, 2), dtype=value.dtype) for value in reduction_inputs)
            lowered = kernel_calls(program, graph, (), dict(zip((value.index for value in reduction_inputs), reduction_storage)),
                outputs=consumers, root_peer=0, tile_rows=1, tile_columns=2)
            observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in consumers)
            argument, = kernels.arguments(1)
            for operation in ('any', 'all'):
                expression = getattr(argument.sum() > 0, operation)()
                result = program.kernel_call(kernels.expression(expression), grid=(2, 1),
                    in_specs=(BlockSpec(None),), out_specs=BlockSpec((1, 1), lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct((2, 1), np.bool_), peer=0)(reduction_storage[4])
                observations += (tuple((i, j, program.export(ref)) for (i, j), ref in sorted(result.blocks.items())),)
                cases.append((2, (2, 5), 1, 4, 'sum_' + operation))
            generations = []
            for generation in range(2):
                signed = np.array([[-2**63, -2**53-1, 0, 2**53+1, 2**63-1],
                                   [2**63-2, -2**63+1, 2**53+3, -2**53-3, 1]], dtype=np.int64)
                unsigned = np.array([[0, 2**53+1, 2**63, 2**64-2, 2**64-1],
                                     [1, 2**53+3, 2**63+1, 2**64-3, 2]], dtype=np.uint64)
                truth = np.array([[0, -0.0, 0, 0, 0], [.25, -.5, np.nan, 2, -3]] if not generation else
                                 [[np.nan, 0, -.25, 1, .125], [-0.0, np.nan, 0, 0, 0]], dtype=np.float32)
                extrema = np.full((2, 5), np.nan, dtype=np.float32)
                if not generation:
                    extrema[1] = [np.nan, -.25, 3, -0.0, np.nan]
                fractional = np.array([[.125, .25, 0, 0, 0], [-.125, -.25, 0, 0, 0]], dtype=np.float32)
                values = (signed[::-1] if generation else signed, unsigned[::-1] if generation else unsigned, truth, extrema,
                          fractional[::-1] if generation else fractional)
                expected = []
                for rank, shape, axis, operand, operation in cases:
                    data = values[operand].reshape(shape)
                    if operand == 4:
                        expected.append(getattr(np, operation[4:])(data.sum(axis=1, keepdims=True) > 0, axis=1, keepdims=True))
                        continue
                    result = ((np.fmax if operation == 'max' else np.fmin).reduce(data, axis=axis, keepdims=True,
                        initial=-np.inf if operation == 'max' else np.inf) if operand == 3 else
                        getattr(np, operation)(data, axis=axis, keepdims=True))
                    expected.append((result.astype(np.int32) * 2 + 1 if operand == 2 else result).reshape(-1, 1))
                generations.append((values, expected))
            xonotic_reductions = reduction_storage, cases, observations, generations
            # design/algorithm-sources.md#shared-elementary-functions
            graph = mx.Graph()
            with graph:
                elementary_inputs = tuple(graph.input(name, (2, 8), dtype) for name, dtype in
                    (('x', 'float32'), ('y', 'float32'), ('signed', 'int64'), ('unsigned', 'uint64')))
                operations = ('arcsinh', 'expm1', 'log1p', 'logaddexp', 'isfinite', 'abs', 'log', 'sqrt', 'power', 'power_square', 'floor_divide')
                consumers = []
                for operation in operations:
                    result = (mx.logaddexp(*elementary_inputs[:2]) if operation == 'logaddexp' else
                              mx.power(elementary_inputs[0], np.float32(.5 if operation == 'power' else 2)) if operation in ('power', 'power_square') else
                              elementary_inputs[0] // np.float32(3) if operation == 'floor_divide' else
                              getattr(mx, operation)(elementary_inputs[0]))
                    consumers.append(result.astype('int32') * 2 + 1 if operation == 'isfinite' else result * 2)
                consumers.extend(mx.where(mx.isfinite(value), value, 0) + 0 for value in elementary_inputs[2:])
                consumers.extend((~value) + 0 for value in elementary_inputs[2:])
                consumers.extend(value // np.array(3, dtype=value.dtype) for value in elementary_inputs[2:])
                consumers.extend(mx.abs(value) + 0 for value in elementary_inputs[2:])
            elementary_storage = tuple(program.tensor((2, 8), (1, 3), dtype=value.dtype) for value in elementary_inputs)
            lowered = kernel_calls(program, graph, (), dict(zip((value.index for value in elementary_inputs), elementary_storage)),
                outputs=consumers, root_peer=0, tile_rows=1, tile_columns=3)
            observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in consumers)
            generations = []
            for generation in range(2):
                x = np.array([[-0.0, 1e-7, -1e-7, -.75, 1e-4, 10, 1e30, -1e30],
                              [np.nan, np.inf, -np.inf, -1, -1.5, 0, 20, -20]], dtype=np.float32)
                y = np.array([[0, -1e-7, 1e-7, .75, -1e-4, -10, -np.inf, np.inf],
                              [1, np.inf, -np.inf, np.nan, np.inf, -np.inf, -20, 20]], dtype=np.float32)
                signed = np.array([[-2**63, 2**63-1, -2**53-1, 2**53+1, 0, -1, 1, 7],
                                   [2**63-2, -2**63+1, 2**53+3, -2**53-3, 2, -2, 8, -8]], dtype=np.int64)
                unsigned = np.array([[2**64-1, 2**63, 2**53+1, 0, 1, 2, 3, 4],
                                     [2**64-2, 2**63+1, 2**53+3, 5, 6, 7, 8, 9]], dtype=np.uint64)
                values = tuple(value[::-1] if generation else value for value in (x, y, signed, unsigned))
                x, y = (value.astype(np.float64) for value in values[:2])
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    expected = tuple((np.logaddexp(x, y) if operation == 'logaddexp' else
                                      np.power(x, np.full_like(x, .5 if operation == 'power' else 2)) if operation in ('power', 'power_square') else np.floor_divide(x, 3) if operation == 'floor_divide' else
                                      getattr(np, operation)(x))
                                     for operation in operations)
                    expected = tuple(value.astype(np.int32) * 2 + 1 if operation == 'isfinite' else value.astype(np.float32) * np.float32(2)
                                     for operation, value in zip(operations, expected)) + values[2:]
                    expected += tuple(np.bitwise_not(value) for value in values[2:])
                    expected += tuple(value // np.array(3, dtype=value.dtype) for value in values[2:])
                    expected += tuple(np.abs(value) for value in values[2:])
                generations.append((values, expected))
            xonotic_elementary = elementary_storage, (*operations, 'signed_isfinite', 'unsigned_isfinite',
                'signed_invert', 'unsigned_invert', 'signed_floor', 'unsigned_floor', 'signed_abs', 'unsigned_abs'), observations, generations
            # design/algorithm-sources.md#indexed-range-generation
            graph = mx.Graph()
            dimension = mx.Dimension('axis', (0,))
            range_cases = ((-7, 23, 3, 'int64'), (23, -7, -3, 'int64'),
                           (-2**53-21, -2**53+9, 3, 'int64'),
                           (2**63+5, 2**63+35, 3, 'uint64'),
                           (dimension-10, dimension, 1, 'int64'),
                           (1024.25, 1025.5, .125, 'float16'))
            with graph:
                range_inputs = tuple(graph.input(f'range_input_{i}', (2, 5), dtype)
                                     for i, (_, _, _, dtype) in enumerate(range_cases))
                generated = tuple(mx.arange(start, stop, step, dtype=dtype)
                                  for start, stop, step, dtype in range_cases)
                consumers = tuple((value.reshape(2, 5) + source) + graph.constant((dimension*3-1)//2, dtype=value.dtype)
                                  for value, source in zip(generated, range_inputs))
                empty_outputs = tuple(mx.arange(start, stop, step, dtype='float32') + np.float32(2)
                                      for start, stop, step in ((0, 0, 1), (5, 0, 1), (0, 5, -1)))
                empty_identities = tuple(mx.reduce(operation, value) for value in empty_outputs
                                         for operation in ('sum', 'any', 'all', 'max', 'min', 'mean'))
                integer_mean = mx.mean(range_inputs[0]) + np.float32(.25)
            range_storage = tuple(program.tensor((2, 5), (1, 3), dtype=value.dtype) for value in range_inputs)
            lowered = kernel_calls(program, graph, (10,), dict(zip((value.index for value in range_inputs), range_storage)),
                outputs=(*generated, *consumers, *empty_outputs, *empty_identities, integer_mean), root_peer=0, tile_rows=1, tile_columns=3)
            observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in (*generated, *consumers))
            references = tuple(((np.float32(start) + np.float32(step) * np.arange(10, dtype=np.float32)).astype(dtype)
                                if np.dtype(dtype).kind == 'f' else
                                np.array([start.resolve((10,)) + i*step if isinstance(start, mx.Dimension) else start+i*step
                                          for i in range(10)], dtype=dtype)).reshape(2, 5)
                               for start, stop, step, dtype in range_cases)
            if any(np.prod(lowered[value.index].shape) or lowered[value.index].blocks for value in empty_outputs):
                raise ArithmeticError('Empty range composition allocated a numerical output')
            empty_results = tuple(program.export(lowered[value.index][0, 0]) for value in empty_identities)
            mean_result = program.export(lowered[integer_mean.index][0, 0])
            xonotic_ranges = range_storage, observations, references, empty_results, mean_result
            # design/algorithm-sources.md#grouped-segment-reductions
            outer_x = program.tensor((67, 5), (33, 5), dtype=np.float32)
            outer_g = program.tensor((67, 7), (33, 7), dtype=np.float32)
            outer_indices = program.tensor((67, 1), (33, 1), dtype=np.int64)
            outer_base = program.tensor((5, 35), (1, 7), dtype=np.float32)
            for ref in outer_base.blocks.values():
                program.constant(ref, np.zeros(ref.shape, dtype=np.float32))
            base_arg, selected_arg, x_arg, g_arg = kernels.arguments(4)
            row, column = kernels.indices()
            update = x_arg.reshape((67, 5)).at(row, column//7) * g_arg.reshape((67, 7)).at(row, column%7)
            first_function = program.native.algebra_trace_count(program.handle)
            grouped = program.kernel_call(kernels.expression(kernels.indexed_add(base_arg, selected_arg, update)),
                grid=outer_base.grid, in_specs=(BlockSpec(None),)*4,
                out_specs=BlockSpec((1, 7), lambda i,j: (i,j)), out_shape=ShapeDtypeStruct((5, 35), np.float32), peer=0)(
                    outer_base, outer_indices, outer_x, outer_g)
            last_function = program.native.algebra_trace_count(program.handle)
            observations = tuple((i,j*grouped.block_shape[1],program.export(ref)) for (i,j),ref in sorted(grouped.blocks.items()))
            generations = []
            for generation in range(2):
                x = ((np.arange(335, dtype=np.float32).reshape(67, 5)%23)-11+generation)/32
                g = ((np.arange(469, dtype=np.float32).reshape(67, 7)%19)-9-generation)/64
                selected = (np.arange(67, dtype=np.int64)+2*generation)%5
                selected[::4] -= 5
                selected[1::11] = 5 if generation == 0 else -6
                normalized = np.where(selected < 0, selected+5, selected)
                valid = (normalized >= 0) & (normalized < 5)
                x[~valid, 0] = np.nan
                expected = np.zeros((5, 5, 7), dtype=np.float64)
                np.add.at(expected, normalized[valid], x[valid, :, None].astype(np.float64)*g[valid, None, :].astype(np.float64))
                generations.append((x, g, selected[:, None], expected.reshape(5, 35)))
            xonotic_grouped_outer = outer_x, outer_g, outer_indices, observations, first_function, last_function, generations
            # design/algorithm-sources.md#composable-indexed-contractions
            context_left, context_right = kernels.arguments(2)
            context_inner = kernels.arange(2)
            context_sum = (context_left.reshape((1, 2)).at(0, context_inner) *
                           context_right.reshape((2, 1)).at(context_inner, 0)).sum()
            context_outputs = program.kernel_call(kernels.expression(context_sum, context_sum+0, context_sum.astype(np.int64)),
                grid=(1,), in_specs=(BlockSpec(None),)*2, out_specs=(BlockSpec((1, 1), lambda i: (0, 0)),)*3,
                out_shape=(ShapeDtypeStruct((1, 1), np.int64),)*3, peer=0)(
                    weight(np.array([[.75, .75]], dtype=np.float32)), weight(np.array([[1], [1]], dtype=np.float32)))
            xonotic_indexed_context = tuple(program.export(tensor[0, 0]) for tensor in context_outputs)
            for dynamic, weight_dtype in ((True, np.float32), (False, np.float32), (True, np.float16)):
                source = program.tensor((2, 5), (1, 3), dtype=np.float32)
                weights = program.tensor((10 if dynamic else 5, 7), (2, 3), dtype=weight_dtype)
                selected = program.tensor((2, 1), (1, 1), dtype=np.int64)
                bias = program.tensor((2, 7), (1, 3), dtype=np.float32)
                weight_values = ((np.arange(np.prod(weights.shape), dtype=np.float32).reshape(weights.shape) % 17)-8)/16
                weight_values = weight_values.astype(weight_dtype)
                for (i,j),ref in weights.blocks.items():
                    row, column = i*weights.block_shape[0], j*weights.block_shape[1]
                    program.constant(ref, weight_values[row:row+ref.shape[0], column:column+ref.shape[1]])
                source_arg, weights_arg, selected_arg, bias_arg = kernels.arguments(4)
                row = kernels.program_id(0)
                column = kernels.program_id(1)*3 + kernels.indices()[1]
                feature = kernels.program_id(1)*3 + kernels.arange(3).T
                inner = kernels.arange(5, tile=3)
                chosen = selected_arg.reshape((2,)).at(row)
                weight_load = (weights_arg.reshape((2, 5, 7)).at(chosen, inner, feature) if dynamic else
                               weights_arg.reshape((5, 7)).at(inner, feature))
                product = source_arg.reshape((2, 5)).at(row, inner) * weight_load
                contraction = product.sum().T
                first_function = program.native.algebra_trace_count(program.handle)
                result = program.kernel_call(kernels.expression(contraction,
                    contraction*2 + bias_arg.reshape((2, 7)).at(row, column)), grid=(2, 3),
                    in_specs=(BlockSpec(None),)*4, out_specs=(BlockSpec((1, 3), lambda i,j: (i,j)),)*2,
                    out_shape=(ShapeDtypeStruct((2, 7), np.float32),)*2, peer=0)(source, weights, selected, bias)
                last_function = program.native.algebra_trace_count(program.handle)
                observations = tuple(tuple((i,j*tensor.block_shape[1],program.export(ref))
                    for (i,j),ref in sorted(tensor.blocks.items())) for tensor in result)
                generations = []
                for generation in range(2):
                    values = (np.array([[1, -2, 3, -4, 5], [-5, 4, -3, 2, -1]], dtype=np.float32)+generation)/8
                    selections = np.array([[generation], [1-generation]], dtype=np.int64)
                    bias_values = (np.arange(14, dtype=np.float32).reshape(2, 7)-7+generation)/32
                    table = weight_values.astype(np.float64).reshape((-1, 5, 7))
                    projected = np.stack([values[i].astype(np.float64) @ table[int(selections[i,0]) if dynamic else 0] for i in range(2)])
                    generations.append((values, selections, bias_values, (projected, projected*2+bias_values)))
                xonotic_composed_indexed.append((dynamic, np.dtype(weight_dtype).name, source, selected, bias, observations, first_function, last_function, generations))
            # design/algorithm-sources.md#stable-indexed-ordering
            for scalar, length in ((np.float32, 7), (np.float16, 7), (np.bool_, 7), (np.int64, 7), (np.uint64, 7), (np.float32, 131)):
                graph = mx.Graph()
                with graph:
                    ordering_input = graph.input('ordering_input', (2, 1, length), np.dtype(scalar).name)
                    indices = mx.argsort(ordering_input, axis=2)
                    ordered = mx.take_along_axis(ordering_input, indices, axis=2)
                    top_indices = mx.argpartition(ordering_input, -length+2, axis=-1)[:, :, :3]
                    top_values = mx.take_along_axis(ordering_input, top_indices, axis=2)
                    axis_indices = mx.argsort(ordering_input.reshape(2, length).transpose(1, 0), axis=0)
                    vector_indices = mx.argsort(ordering_input[0, 0, :], axis=None)
                    nonlast_indices = mx.argsort(ordering_input.transpose(0, 2, 1), axis=1)
                    ordering_outputs = (indices, ordered, top_indices, top_values, axis_indices, vector_indices, nonlast_indices)
                ordering_storage = program.tensor((2, length), (1, 129 if length > 128 else 3), dtype=scalar)
                first_function = program.native.algebra_trace_count(program.handle)
                lowered = kernel_calls(program, graph, (), {ordering_input.index: ordering_storage},
                    outputs=ordering_outputs, root_peer=0, tile_rows=1, tile_k=3, tile_columns=ordering_storage.block_shape[1])
                last_function = program.native.algebra_trace_count(program.handle)
                observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                    for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in ordering_outputs)
                generations = []
                for generation in range(2):
                    if np.dtype(scalar).kind == 'f':
                        values = np.array([[np.nan, 3, -0., 0., 3, np.nan, -np.inf],
                                           [np.inf, -0., np.nan, 3, 3, 0., -np.inf]], dtype=scalar)
                    elif scalar == np.bool_:
                        values = np.array([[True, False, True, True, False, False, True],
                                           [False, True, False, True, False, True, False]], dtype=scalar)
                    elif scalar == np.int64:
                        values = np.array([[2**63-1, 2**53+1, -1, -2**63, 2**53+1, 0, -1],
                                           [-1, 2**53+3, -2**63, 0, 2**53+3, 2**63-1, 0]], dtype=scalar)
                    else:
                        values = np.array([[2**64-1, 2**63+1, 2**53+1, 0, 2**53+1, 1, 2**64-1],
                                           [2**53+3, 0, 2**64-1, 1, 2**53+3, 2**63+1, 0]], dtype=scalar)
                    if length != 7:
                        values = np.stack([np.resize(row, length) for row in values])
                    if generation:
                        values = values[::-1, ::-1].copy()
                    order = np.argsort(values, axis=1, kind='stable').astype(np.uint32)
                    ordered_values = np.take_along_axis(values, order, axis=1)
                    generations.append((values, (order, ordered_values, order[:, :3], ordered_values[:, :3], order.T, order[:1], order.reshape(2*length, 1))))
                xonotic_ordering.append((np.dtype(scalar).name, ordering_storage, observations, first_function, last_function, generations))
            # design/algorithm-sources.md#typed-integer-contractions
            for scalar in (np.int32, np.uint32, np.int64, np.uint64, np.bool_):
                dtype_name = np.dtype(scalar).name
                boolean = scalar == np.bool_
                bits = np.dtype(scalar).itemsize*8
                graph = mx.Graph()
                with graph:
                    dot_input = graph.input('integer_dot', (2, 3, 2), dtype_name)
                    weight_values = np.array([[[1, 1, 0], [1, 0, 1]]] if boolean else [[[2, 4, 6], [3, 5, 7]]], dtype=scalar)
                    second_values = np.eye(2, dtype=scalar) if boolean else np.array([[2, 1], [1, 3]], dtype=scalar)
                    product = mx.matmul(dot_input, graph.constant(weight_values, dtype=dtype_name), transpose_left=True, transpose_right=True)
                    composed = mx.matmul(product, graph.constant(second_values, dtype=dtype_name))
                    consumer = ~composed if boolean else composed + graph.constant(1, dtype=dtype_name)
                    empty_product = mx.matmul(graph.constant(np.empty((2, 0), dtype=scalar), dtype=dtype_name),
                                              graph.constant(np.empty((0, 2), dtype=scalar), dtype=dtype_name))
                    mixed_products = ()
                    if scalar == np.int32:
                        mixed_products = (mx.matmul(graph.constant([[2, 1]], dtype='int32'),
                                                    graph.constant([[.5], [.25]], dtype='float32')),
                                          mx.matmul(graph.constant([[.5, .25]], dtype='float32'),
                                                    graph.constant([[2], [1]], dtype='int32')))
                dot_storage = program.tensor((6, 2), (1, 2), dtype=scalar)
                lowered = kernel_calls(program, graph, (), {dot_input.index: dot_storage},
                    outputs=(product, consumer, empty_product, *mixed_products), root_peer=0, tile_rows=1, tile_k=2, tile_columns=1)
                observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                    for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in (product, consumer))
                empty_results = tuple(program.export(ref) for _, ref in sorted(lowered[empty_product.index].blocks.items()))
                mixed_results = tuple(program.export(lowered[value.index][0, 0]) for value in mixed_products)
                rows_per_batch = [2, 2]
                if scalar == np.int64:
                    left_arg, right_arg = kernels.arguments(2)
                    fused_dot = program.kernel_call(kernels.expression(kernels.dot(left_arg, right_arg, tile_k=1)+np.int64(0)),
                        grid=(6, 2), in_specs=(BlockSpec(None),)*2,
                        out_specs=BlockSpec((1, 1), lambda i,j: (i,j)), out_shape=ShapeDtypeStruct((6, 2), np.int64), peer=0)(
                            dot_storage, weight(second_values))
                    observations += (tuple((i,j,program.export(ref)) for (i,j),ref in sorted(fused_dot.blocks.items())),)
                    rows_per_batch.append(3)
                generations = []
                for generation in range(2):
                    if boolean:
                        values = np.array([[[True, False], [False, True], [True, True]],
                                           [[False, True], [True, False], [False, False]]], dtype=scalar)
                    else:
                        limits = np.iinfo(scalar)
                        large = 2**(24 if bits == 32 else 53)+1
                        values = np.array([[[limits.max, large], [limits.min, 3], [5, 7]],
                                           [[large+2, limits.max-1], [11, limits.min], [13, 17]]], dtype=scalar)
                    if generation:
                        values = values[::-1, ::-1].copy()
                    projected = [[[sum(int(values[b, k, i])*int(weight_values[0, j, k]) for k in range(3))
                                   for j in range(2)] for i in range(2)] for b in range(2)]
                    chained = [[[sum(projected[b][i][k]*int(second_values[k, j]) for k in range(2)) + (0 if boolean else 1)
                                 for j in range(2)] for i in range(2)] for b in range(2)]
                    expected = []
                    for output_index, output in enumerate((projected, chained)):
                        flat = [value for batch in output for row in batch for value in row]
                        if boolean:
                            flat = [not bool(value) if output_index else bool(value) for value in flat]
                        else:
                            flat = [value % (1 << bits) for value in flat]
                            if np.dtype(scalar).kind == 'i':
                                flat = [value-(1 << bits) if value >= 1 << (bits-1) else value for value in flat]
                        expected.append(np.array(flat, dtype=scalar).reshape(4, 2))
                    if scalar == np.int64:
                        flat = [sum(int(row[k])*int(second_values[k,j]) for k in range(2)) % (1 << 64)
                                for row in values.reshape(6, 2) for j in range(2)]
                        expected.append(np.array([value-(1 << 64) if value >= 1 << 63 else value for value in flat], dtype=scalar).reshape(6, 2))
                    generations.append((values.reshape(6, 2), expected))
                xonotic_integer_dots.append((dtype_name, dot_storage, observations, rows_per_batch, empty_results, mixed_results, generations))
            # design/algorithm-sources.md#counter-based-random-generation
            known_answers = []
            for counter, key, expected in (
                    ((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
                    ((0xffffffff,)*4, (0xffffffff,)*2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
                    ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
                     (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))):
                outputs = program.kernel_call(kernels.expression(*kernels.philox4x32(counter, *key)), grid=(1,),
                    in_specs=(), out_specs=(BlockSpec((1, 1), lambda i: (0, 0)),)*4,
                    out_shape=(ShapeDtypeStruct((1, 1), np.uint32),)*4, peer=0)()
                known_answers.append((tuple(program.export(value[0, 0]) for value in outputs), expected))
            graph = mx.Graph()
            with graph:
                random_key = graph.input('random_key', (2,), 'uint32')
                random_input = graph.input('random_consumer', (2, 5))
                normal = mx.random.normal((2, 5), key=random_key)
                random_consumer = (normal + random_input) * np.float32(2)
            key_storage = program.tensor((1, 2), dtype=np.uint32)
            random_storage = program.tensor((2, 5), (1, 3), dtype=np.float32)
            lowered = kernel_calls(program, graph, (), {random_key.index: key_storage, random_input.index: random_storage},
                outputs=(normal, random_consumer), root_peer=0, tile_rows=1, tile_columns=3)
            random_observations = tuple(tuple((i * lowered[value.index].block_shape[0], j * lowered[value.index].block_shape[1], program.export(ref))
                for (i, j), ref in sorted(lowered[value.index].blocks.items())) for value in (normal, random_consumer))
            high_keys = program.tensor((2, 2), (1, 2), dtype=np.uint32)
            key_arg, = kernels.arguments(1)
            high_normal = program.kernel_call(kernels.expression(kernels.random_normal(key_arg.at(0, 0), key_arg.at(0, 1),
                2**33 + kernels.program_id(0)*4 + kernels.arange(4))), grid=(2,),
                in_specs=(BlockSpec((1, 2), lambda i: (i, 0)),),
                out_specs=BlockSpec((1, 4), lambda i: (i, 0)), out_shape=ShapeDtypeStruct((2, 4), np.float32), peer=0)(high_keys)
            high_arg, = kernels.arguments(1)
            high_consumer = program.kernel_call(kernels.expression(high_arg*2+1), grid=(2,),
                in_specs=(BlockSpec((1, 4), lambda i: (i, 0)),),
                out_specs=BlockSpec((1, 4), lambda i: (i, 0)), out_shape=ShapeDtypeStruct((2, 4), np.float32), peer=0)(high_normal)
            high_results = tuple(program.export(high_consumer[i, 0]) for i in range(2))
            generations = []
            for generation, key in enumerate(((0, 0), (0xa4093822, 0x299f31d0))):
                reference = []
                for ordinal in (*range(10), *range(2**33, 2**33+8)):
                    counter = (ordinal//2 & 0xffffffff, ordinal//2 >> 32, 0, 0)
                    key0, key1 = key
                    for round_index in range(10):
                        first, third = counter[0] * 0xd2511f53, counter[2] * 0xcd9e8d57
                        counter = ((third >> 32) ^ counter[1] ^ key0, third & 0xffffffff,
                                   (first >> 32) ^ counter[3] ^ key1, first & 0xffffffff)
                        key0, key1 = (key0+0x9e3779b9) & 0xffffffff, (key1+0xbb67ae85) & 0xffffffff
                    radius = np.sqrt(np.float32(-2) * np.log(np.float32((counter[0] >> 9)+.5) * np.float32(2**-23)))
                    angle = np.float32(2*np.pi) * (np.float32(counter[1] >> 9) * np.float32(2**-23))
                    reference.append(radius * (np.sin(angle) if ordinal & 1 else np.cos(angle)))
                values = np.arange(10, dtype=np.float32).reshape(2, 5)/8 + np.float32(generation/4)
                generations.append((np.array([key], dtype=np.uint32), values,
                    np.array(reference[:10], dtype=np.float32).reshape(2, 5), np.array(reference[10:], dtype=np.float32).reshape(2, 4)*np.float32(2)+np.float32(1)))
            xonotic_random = known_answers, key_storage, random_storage, random_observations, high_keys, high_results, generations
            take_indices = program.tensor((4, 1), (1, 1), dtype=np.int64)
            take_cotangents = program.tensor((4, 1), (1, 1), dtype=np.float32)
            graph = mx.Graph()
            with graph:
                primal = graph.input('primal', (1, 2, 4))
                indices = graph.input('indices', (2, 2, 1), 'int64')
                cotangent = graph.input('cotangent', (2, 2, 1))
                selected = mx.take_along_axis(primal, indices, axis=2)
                gradient, = graph.vjp((selected,), (cotangent,), (primal,))
                transformed = gradient * 2 + 1
            lowered = kernel_calls(program, graph, (),
                {indices.index: take_indices, cotangent.index: take_cotangents},
                outputs=(transformed,), root_peer=0, tile_rows=1, tile_columns=4)
            take_results = tuple(program.export(ref) for _, ref in sorted(lowered[transformed.index].blocks.items()))
            generations = []
            for generation in range(2):
                index_values = np.array([1, -1, 1, -1] if not generation else [-2, -5, 4, 0], dtype=np.int64).reshape(2, 2, 1)
                cotangent_values = np.arange(1 + generation, 5 + generation, dtype=np.float32).reshape(2, 2, 1)
                expected = np.zeros((2, 4), dtype=np.float64)
                normalized = np.where(index_values.reshape(-1) < 0, index_values.reshape(-1) + 4, index_values.reshape(-1))
                valid = (normalized >= 0) & (normalized < 4)
                np.add.at(expected, (np.array([0, 1, 0, 1])[valid], normalized[valid]), cotangent_values.reshape(-1)[valid])
                generations.append((index_values, cotangent_values, expected * 2 + 1))
            xonotic_take_gradient = (take_indices, take_cotangents, take_results, generations)
        invocations = []

        # design/algorithm-sources.md#async-index-push-contract
        def exchange(value, sender, receiver):
            if args.local:
                return value
            received = program.tensor(value.shape, value.block_shape, dtype=value.dtype)
            program.copy(value.on(sender), received.on(receiver), queue=0)
            return received

        # design/algorithm-sources.md#canonical-view-replication
        if args.xonotic:
            alias_source = program.tensor((2, 4), (1, 4), dtype=np.float32)
            graph = mx.Graph()
            with graph:
                alias_input = graph.input('alias_source', (2, 4))
                alias_output = alias_input.reshape(4, 2).transpose(1, 0)
            lowered = kernel_calls(program, graph, (), {alias_input.index: alias_source},
                outputs=(alias_output,), root_peer=0, tile_rows=2, tile_columns=1)
            receiver = 0 if args.local else 1
            alias = program.replicate(lowered[alias_output.index].on(0), receiver)
            alias_arg, = kernels.arguments(1)
            alias_consumer = program.kernel_call(kernels.expression(alias_arg*2+1), grid=(1, 4),
                in_specs=(BlockSpec((2, 1), lambda i,j: (i,j)),), out_specs=BlockSpec((2, 1), lambda i,j: (i,j)),
                out_shape=ShapeDtypeStruct((2, 4), np.float32), peer=receiver)(alias)
            returned = program.replicate(alias_consumer.on(receiver), 0)
            alias_results = tuple((i*returned.block_shape[0], j*returned.block_shape[1], program.export(ref))
                                  for (i,j),ref in sorted(returned.blocks.items()))
            xonotic_alias = alias_source, alias_results
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
        for run in range(args.runs):
            data = tuple((rng.standard_normal((rows, width), dtype=np.float32) / 8).astype(dtype) for _ in range(2))
            inputs = tuple(program.tensor(value.shape, (tile, args.tile_k), dtype=dtype) for value in data)
            # design/algorithm-sources.md#deep-composed-performance
            operands = inputs
            for depth in range(args.depth):
                up = first_u if depth == 0 else tuple((group[0],) for group in first_u)
                a = ffn(program, operands, up, first_d, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns,
                        exchange=None if args.local else lambda value: exchange(value, 0, 1))
                b = rmsnorm(program, a, scale, tile_rows=tile)
                c = summed_embedding(program, b, table_tensors, index_tensors, tile_rows=tile)
                d = ffn(program, (c,), second_u, second_d, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns,
                        exchange=None if args.local else lambda value: exchange(value, 1, 0))
                e = rmsnorm(program, d, scale, tile_rows=tile)
                operands = (e,)
            stages = {} if args.rank == 1 else {'ffn2': d, 'rmsnorm2': e}
            probes = {name: {coordinate: program.export(ref) for coordinate, ref in value.blocks.items()}
                      for name, value in stages.items()}
            invocations.append((data, inputs, probes))
        program.realize()
        print(json.dumps(dict(event='realized', pid=os.getpid(), rank=args.rank, runs=args.runs, depth=args.depth,
            rows=rows, width=width, tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns)), flush=True)
        if args.rank == 1:
            while running:
                time.sleep(0.0001)
            if args.trace:
                Path(args.trace).write_text(json.dumps(dict(compute=program.trace, routes=program.route_trace, transfers=program.transfer_trace), indent=2) + '\n')
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

        # design/algorithm-sources.md#streaming-overlap-measurement
        def wait_completed(indices, after):
            deadline = time.monotonic_ns() + 60_000_000_000
            while running:
                completed_functions = tuple(index for index in indices
                    if program.native.algebra_trace(program.handle, index).complete_ns >= after)
                if completed_functions:
                    return completed_functions
                if time.monotonic_ns() > deadline:
                    raise TimeoutError('No numerical consumer completed from the published partial operands')
                time.sleep(0.0001)
            raise InterruptedError('Partial consumer observation interrupted')

        expected_by_run = {}
        for run, (data, _, _) in enumerate(invocations, 1):
            # design/algorithm-sources.md#deep-composed-performance
            operands = data
            for depth in range(args.depth):
                up = first_up if depth == 0 else tuple((group[0],) for group in first_up)
                expected = reference_norm(reference_ffn(operands, up, first_down), gamma)
                for table, index in zip(tables, ids):
                    expected = (expected.astype(np.float64) + table[index[:, 0]]).astype(expected.dtype)
                expected = reference_norm(reference_ffn((expected,), second_up, second_down), gamma)
                operands = (expected,)
            expected_by_run[run] = expected

        warm_started = time.monotonic_ns()
        warm = invocations[0]
        publish(warm, range(1, (rows + tile - 1) // tile))
        warm_results = warm[2]['rmsnorm2']
        wait_for((warm_results[1, 0],))
        if any(result.ready for coordinate, result in warm_results.items() if coordinate[0] == 0):
            raise ArithmeticError('An unpublished input section produced an output')
        publish(warm, (0,))
        wait_for(tuple(warm_results.values()))
        print(json.dumps(dict(event='warmup', depth=args.depth, elapsed_ms=(time.monotonic_ns() - warm_started) / 1e6)), flush=True)

        for stage in warm[2].values():
            for result in stage.values():
                result.consume()
        first_results = tuple(result for _, _, probes in invocations for result in probes['rmsnorm2'].values())
        errors, sample_batches, sample_first, completion_observations = [], [], [], []
        for sample in range(args.samples):
            wait_for(tuple(ref for _, inputs, _ in invocations for tensor in inputs for ref in tensor.blocks.values()), 'writable')
            batch_started = time.monotonic_ns()
            for invocation in invocations:
                publish(invocation, range((rows + tile - 1) // tile))
            deadline = time.monotonic_ns() + 60_000_000_000
            while running and not any(result.ready for result in first_results):
                if time.monotonic_ns() > deadline:
                    raise TimeoutError(f'First output stalled: {program.report}')
                time.sleep(0.0001)
            first_ms = (time.monotonic_ns() - batch_started) / 1e6
            completed = {}
            deadline = time.monotonic_ns() + 60_000_000_000
            while running and len(completed) < args.runs:
                for run, (_, _, probes) in enumerate(invocations, 1):
                    if run not in completed and all(result.ready for result in probes['rmsnorm2'].values()):
                        completed[run] = (time.monotonic_ns() - batch_started) / 1e6
                if time.monotonic_ns() > deadline:
                    raise TimeoutError(f'Batch stalled: {program.report}')
                if len(completed) < args.runs:
                    time.sleep(0.0001)
            if not running:
                return
            batch_ms = (time.monotonic_ns() - batch_started) / 1e6
            for run, (data, _, probes) in enumerate(invocations, 1):
                expected = expected_by_run[run]
                error = 0.0
                for (row, column), result in probes['rmsnorm2'].items():
                    r, c = row * tile, column * args.tile_columns
                    part = expected[r:r + result.ref.shape[0], c:c + result.ref.shape[1]]
                    error = max(error, float(np.max(np.abs(result.array - part))))
                    if not np.allclose(result.array, part, atol=3e-3 if dtype == np.float16 else 3e-4, rtol=3e-3 if dtype == np.float16 else 3e-4):
                        raise ArithmeticError(f'Gold chain numerical mismatch: {error}')
                errors.append(error)
                print(json.dumps(dict(event='gold', sample=sample, run=run, depth=args.depth, complete_ms=completed.get(run),
                    max_absolute_error=error, transport_queue=0, backend=args.backend, local=args.local)), flush=True)
                for stage in probes.values():
                    for result in stage.values():
                        result.consume()
            sample_batches.append(batch_ms)
            sample_first.append(first_ms)
            completion_observations.extend(completed.values())
            print(json.dumps(dict(event='performance_sample', sample=sample, depth=args.depth,
                batch_ms=batch_ms, first_section_ms=first_ms, completion_ms=summary(tuple(completed.values())))), flush=True)
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
        for ordinal, result in enumerate(mapped_results):
            i, j = divmod(ordinal, mapped.grid[1])
            if not np.array_equal(result.array, strided_expected[2-i:3-i, 3*j:3*j+result.array.shape[1]]):
                raise ArithmeticError('Contraction ignored its input index map')
        print(json.dumps(dict(event='mapped_contraction', result=[result.array.tolist() for result in mapped_results])), flush=True)
        for result in mapped_results:
            result.consume()
        for generation in range(2):
            wait_for((*composed_input.blocks.values(), *composed_bias.blocks.values()), 'writable')
            composed_values = (strided_left.T + generation).astype(dtype)
            composed_expected = composed_values.astype(np.float32) @ strided_right.astype(np.float32).T
            bias_values = np.full((3, 7), generation + 1, dtype=np.float32)
            with program.write(composed_input[1, 0]) as target:
                target[...] = composed_values[1:2]
            wait_for((composed_results[1][1],))
            if composed_results[0][1].ready or any(composed_results[1][i].ready for i in (0, 2)):
                raise ArithmeticError('Composed contraction crossed an unpublished operand boundary')
            if not np.array_equal(composed_results[1][1].array, composed_expected[1:2]):
                raise ArithmeticError('Composed contraction numerical mismatch')
            with program.write(composed_bias[1, 0]) as target:
                target[...] = bias_values[1:2]
            wait_for((composed_results[0][1],))
            if not np.array_equal(composed_results[0][1].array, 2 * composed_expected[1:2] + bias_values[1:2]):
                raise ArithmeticError('Composed contraction epilogue mismatch')
            print(json.dumps(dict(event='composed_dot_early', generation=generation,
                withheld_rows=[0, 2], contraction=composed_results[1][1].array.tolist(),
                epilogue=composed_results[0][1].array.tolist())), flush=True)
            for i in (0, 2):
                with program.write(composed_input[i, 0]) as target:
                    target[...] = composed_values[i:i+1]
                with program.write(composed_bias[i, 0]) as target:
                    target[...] = bias_values[i:i+1]
            wait_for(tuple(result for output in composed_results for result in output))
            for output, expected in zip(composed_results, (2 * composed_expected + bias_values, composed_expected)):
                for i, result in enumerate(output):
                    if not np.array_equal(result.array, expected[i:i+1]):
                        raise ArithmeticError('Composed contraction repeated output mismatch')
                    result.consume()
        nested_projection = nested_inputs[2][0, 0]
        projection_rows = program.native.tensor_rows(nested_projection.view.tensor, nested_projection.view.extent)
        nested_trace = program.trace
        nested_consumers = tuple(index for index, entry in enumerate(nested_trace)
            if any(region['first'] < projection_rows.first + projection_rows.count and
                projection_rows.first < region['first'] + region['count'] for region in entry['inputs']))
        if not nested_consumers:
            raise ArithmeticError('Nested contraction omitted its projection operand')
        row_producers = {row: index for index, entry in enumerate(nested_trace)
            for region in entry['outputs'] for row in range(region['first'], region['first'] + region['count'])}
        nested_branches = []
        for tensor in nested:
            ref = tensor[1, 0]
            region = program.native.tensor_rows(ref.view.tensor, ref.view.extent)
            pending = set(range(region.first, region.first + region.count))
            ancestors = set()
            while pending:
                index = row_producers.get(pending.pop())
                if index is None or index in ancestors:
                    continue
                ancestors.add(index)
                pending.update(row for region in nested_trace[index]['inputs']
                    for row in range(region['first'], region['first'] + region['count']))
            nested_branches.append(tuple(index for index in nested_consumers if index in ancestors))
        if not all(nested_branches):
            raise ArithmeticError('Nested outputs must retain projection consumers')
        for generation, (values, expected) in enumerate(nested_generations):
            wait_for(tuple(ref for tensor in nested_inputs for ref in tensor.blocks.values()), 'writable')
            started = time.monotonic_ns()
            for tensor, coordinate, value in (
                    (nested_inputs[0], (1, 0), values[0][1:2]),
                    (nested_inputs[1], (0, 0), values[1][:, :2]),
                    (nested_inputs[2], (0, 0), values[2][:2])):
                with program.write(tensor[coordinate]) as target:
                    target[...] = value
            consumer_completed = tuple(wait_completed(branch, started) for branch in nested_branches)
            if any(result.ready for output in nested_results for result in output) or any(
                    not nested_inputs[1][0, i].writable for i in (1, 2)):
                raise ArithmeticError('Nested contraction crossed a withheld hidden-panel boundary')
            print(json.dumps(dict(event='nested_dot_partial', generation=generation,
                consumer_functions=consumer_completed, withheld_hidden_panels=[1, 2],
                projection_rows=dict(first=projection_rows.first, count=projection_rows.count))), flush=True)
            for i in (1, 2):
                with program.write(nested_inputs[1][0, i]) as target:
                    target[...] = values[1][:, 2*i:2*i+2]
                with program.write(nested_inputs[2][i, 0]) as target:
                    target[...] = values[2][2*i:2*i+2]
            wait_for(tuple(output[1] for output in nested_results))
            if any(output[i].ready for output in nested_results for i in (0, 2)):
                raise ArithmeticError('Nested contraction consumed an unpublished input row')
            print(json.dumps(dict(event='nested_dot_row', generation=generation,
                withheld_input_rows=[0, 2], output=[output[1].array.tolist() for output in nested_results])), flush=True)
            for i in (0, 2):
                with program.write(nested_inputs[0][i, 0]) as target:
                    target[...] = values[0][i:i+1]
            wait_for(tuple(result for output in nested_results for result in output))
            for output, reference in zip(nested_results, expected):
                for i, result in enumerate(output):
                    if not np.allclose(result.array, reference[i:i+1], atol=3e-4, rtol=3e-4):
                        raise ArithmeticError('Nested contraction numerical mismatch')
                    result.consume()
        wait_for((cast_result, *cast_sum_results))
        if not np.array_equal(cast_result.array, [[0, 0, 1]]) or tuple(result.array.item() for result in cast_sum_results) != (1, 0):
            raise ArithmeticError('Explicit cast moved across an arithmetic or reduction boundary')
        print(json.dumps(dict(event='cast_boundaries', rounded=cast_result.array.tolist(),
            sum_then_cast=cast_sum_results[0].array.item(), cast_then_sum=cast_sum_results[1].array.item())), flush=True)
        for result in (cast_result, *cast_sum_results):
            result.consume()
        wait_for(mixed_results)
        mixed_expected = mixed_values[0].astype(np.float64) @ mixed_values[1].astype(np.float64)
        if any(not np.array_equal(result.array, mixed_expected) for result in mixed_results):
            raise ArithmeticError('Mixed contraction matrix orientation mismatch')
        print(json.dumps(dict(event='mixed_contraction', shape=list(mixed_expected.shape),
            output=[result.array.tolist() for result in mixed_results])), flush=True)
        for result in mixed_results:
            result.consume()
        norm_trace = program.trace
        norm_source_rows = set()
        for j in (0, 1):
            ref = norm_input[1, j]
            region = program.native.tensor_rows(ref.view.tensor, ref.view.extent)
            norm_source_rows.update(range(region.first, region.first + region.count))
        norm_final_rows = set()
        for result in norm_results.values():
            ref = result.ref
            region = program.native.tensor_rows(ref.view.tensor, ref.view.extent)
            norm_final_rows.update(range(region.first, region.first + region.count))
        norm_partial_rows = {row for entry in norm_trace
            if any(row in norm_source_rows for region in entry['inputs'] for row in range(region['first'], region['first'] + region['count']))
            for region in entry['outputs'] for row in range(region['first'], region['first'] + region['count'])
            if row not in norm_final_rows}
        norm_consumers = tuple(index for index, entry in enumerate(norm_trace)
            if any(row in norm_partial_rows for region in entry['inputs'] for row in range(region['first'], region['first'] + region['count'])))
        if not norm_consumers:
            raise ArithmeticError('Normalization has no independently consumable feature partials')
        for generation, (values, expected) in enumerate(norm_generations):
            wait_for(tuple(norm_input.blocks.values()), 'writable')
            started = time.monotonic_ns()
            for j in (0, 1):
                with program.write(norm_input[1, j]) as target:
                    target[...] = values[1:2, 2*j:2*j+2]
            consumed = wait_completed(norm_consumers, started)
            if any(result.ready for result in norm_results.values()) or not norm_input[1, 2].writable:
                raise ArithmeticError('Normalization crossed an unpublished feature boundary')
            print(json.dumps(dict(event='reduction_partial', generation=generation,
                consumer_functions=consumed, partial_rows=sorted(norm_partial_rows), withheld_feature_panel=2)), flush=True)
            with program.write(norm_input[1, 2]) as target:
                target[...] = values[1:2, 4:6]
            wait_for(tuple(norm_results[1, j] for j in range(3)))
            if any(result.ready for (i, j), result in norm_results.items() if i != 1):
                raise ArithmeticError('Normalization consumed an unpublished row')
            print(json.dumps(dict(event='reduction_row', generation=generation,
                withheld_input_rows=[0, 2], output=[norm_results[1, j].array.tolist() for j in range(3)])), flush=True)
            for i in (0, 2):
                for j in range(3):
                    with program.write(norm_input[i, j]) as target:
                        target[...] = values[i:i+1, 2*j:2*j+2]
            wait_for(tuple(norm_results.values()))
            for (i, j), result in norm_results.items():
                if not np.allclose(result.array, expected[i:i+1, 2*j:2*j+2],
                        atol=3e-3 if dtype == np.float16 else 3e-4, rtol=3e-3 if dtype == np.float16 else 3e-4):
                    raise ArithmeticError('Streamed normalization numerical mismatch')
                result.consume()
        for i in range(4):
            for j in range(2):
                with program.write(integer_input[i, j]) as target:
                    target[...] = integer_values[i:i+1, 3*j:3*j+3]
            wait_for(integer_results[i::4])
            if any(result.ready for result in integer_results[i+1:4]):
                raise ArithmeticError('Integer reduction published an absent row')
        wait_for(integer_results)
        for result, expected in zip(integer_results, (65536, 2**32+2, -(2**63), 2**63-1) * (len(integer_results)//4)):
            if result.array.item() != expected:
                raise ArithmeticError('Integer reduction lost exact cancellation beyond floating-point precision')
        if integer_total_result is not None:
            wait_for((integer_total_result,))
            if integer_total_result.array.item() != 4295032833:
                raise ArithmeticError('Full matrix reduction lost modular integer accumulation')
            print(json.dumps(dict(event='integer_matrix_total', output=integer_total_result.array.item())), flush=True)
            integer_total_result.consume()
        print(json.dumps(dict(event='integer_reduction', output=[result.array.tolist() for result in integer_results])), flush=True)
        for result in integer_results:
            result.consume()
        for results, quotient, remainder in quotient_cases:
            wait_for(results)
            if results[0].array.tolist() != quotient or results[1].array.tolist() != remainder:
                raise ArithmeticError('Integer index quotient/remainder lost exact value or floor semantics')
            print(json.dumps(dict(event='integer_index_arithmetic', dtype=str(results[0].array.dtype),
                quotient=results[0].array.tolist(), remainder=results[1].array.tolist())), flush=True)
            for result in results:
                result.consume()
        indexed_expected = np.stack((2 * table_data[2], np.ones(4, dtype=dtype)))
        if not indexed.ready or not np.array_equal(indexed.array, indexed_expected):
            raise ArithmeticError('Indexed expression lost integer identity or masked access semantics')
        print(json.dumps(dict(event='indexed', result=indexed.array.tolist())), flush=True)
        indexed.consume()
        wait_for((logical_bounds_result,))
        if not np.array_equal(logical_bounds_result.array, [[0, 1, -1, -1], [0, 1, -1, -1]]):
            raise ArithmeticError('Logical axis bounds aliased another valid physical address')
        print(json.dumps(dict(event='logical_bounds', output=logical_bounds_result.array.tolist())), flush=True)
        logical_bounds_result.consume()
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
            streamed_values = np.arange(8, dtype=dtype).reshape(2, 4) + generation + 2
            with program.write(streamed_table[1, 0]) as destination:
                destination[...] = streamed_values
            if generation:
                with program.write(streamed_table[0, 0]) as destination:
                    destination[...] = 0
            wait_for(streamed_results)
            if any(not np.array_equal(result.array, expected) for result, expected in
                    zip(streamed_results, (2 * streamed_values, streamed_values[:1] + streamed_values))):
                raise ArithmeticError('Selected source occurrence was lost during indexed reuse')
            if not generation and streamed_table[0, 0].present:
                raise ArithmeticError('Unrelated table source was unexpectedly published')
            print(json.dumps(dict(event='dynamic_indexed', generation=generation,
                unrelated_source_present=streamed_table[0, 0].present,
                result=[result.array.tolist() for result in streamed_results])), flush=True)
            if not generation:
                for result in streamed_results:
                    result.consume()
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
                wait_for((observations[0], observations[2], mean_results[early_source]))
                mean_expected = source_values.astype(np.float64).mean(axis=1, keepdims=True)
                selected_expected = expected[:4]
                logical_expected = dict(
                    take=np.take_along_axis(selected_expected.reshape(2, 2, 4), index_values.reshape(2, 2, 1), axis=2).reshape(4, 1),
                    broadcast_take=np.take_along_axis(selected_expected[:2].reshape(1, 2, 4), index_values.reshape(2, 2, 1), axis=2).reshape(4, 1),
                    transpose=selected_expected, concatenate=np.repeat(selected_expected, 2, axis=0) * np.arange(1, 5) + 1,
                    reshape_gather=selected_expected.reshape(2, 4, 2)[:, :, ::-1].reshape(8, 2))
                logical_early = {name: results[:len(results)//2] for name, results in logical_results.items()}
                wait_for(tuple(result for results in logical_early.values() for result in results))
                for name, results in logical_early.items():
                    if not np.array_equal(np.concatenate([result.array for result in results]), logical_expected[name][:2*len(results)]):
                        raise ArithmeticError(f'Logical indexed early output differs: {name}')
                    if any(result.ready for result in logical_results[name][len(results):]):
                        raise ArithmeticError(f'Logical indexing consumed an unpublished region: {name}')
                print(json.dumps(dict(event='xonotic_logical_early', generation=generation,
                    withheld_source_block=1-early_source, withheld_index_block=1,
                    output={name: [result.array.tolist() for result in results] for name, results in logical_early.items()})), flush=True)
                if total_result.ready or any(result.ready for result in column_mean_results) or not np.array_equal(mean_results[early_source].array, mean_expected[2*early_source:2*early_source+2]):
                    raise ArithmeticError('Xonotic row reduction lost independent source progress')
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
                wait_for((*observations, *mean_results, total_result, *column_mean_results, *(result for results in logical_results.values() for result in results)))
                if not np.array_equal(np.concatenate([result.array for result in mean_results]), mean_expected) or total_result.array.item() != mean_expected.sum():
                    raise ArithmeticError('Xonotic matrix/vector reduction mismatch')
                if not np.array_equal(np.concatenate([result.array for result in column_mean_results], axis=1), source_values.mean(axis=0, keepdims=True)):
                    raise ArithmeticError('Xonotic column reduction mismatch')
                print(json.dumps(dict(event='xonotic_reductions', generation=generation, early_source_block=early_source,
                    means=[result.array.tolist() for result in mean_results],
                    column_means=[result.array.tolist() for result in column_mean_results], total=total_result.array.item())), flush=True)
                for index, result in enumerate(observations):
                    if not np.array_equal(result.array, expected[2*index:2*index+2]):
                        raise ArithmeticError('Xonotic gather/concatenate output differs after reuse')
                print(json.dumps(dict(event='xonotic_indexed_complete', generation=generation,
                    output=[result.array.tolist() for result in observations])), flush=True)
                for name, results in logical_results.items():
                    if not np.array_equal(np.concatenate([result.array for result in results]), logical_expected[name]):
                        raise ArithmeticError(f'Logical indexed output differs after reuse: {name}')
                print(json.dumps(dict(event='xonotic_logical_complete', generation=generation,
                    output={name: [result.array.tolist() for result in results] for name, results in logical_results.items()})), flush=True)
                for result in (*observations, *mean_results, total_result, *column_mean_results, *(result for results in logical_results.values() for result in results)):
                    result.consume()
        for name, index_tensors, cotangents, gradient_results, generations in xonotic_gradients:
            for generation, (index_values, cotangent_values, expected) in enumerate(generations):
                if name in scatter_bases:
                    backing, base_values = scatter_bases[name]
                    wait_for(tuple(backing.blocks.values()), 'writable')
                    for (i, j), ref in backing.blocks.items():
                        if i != 2:
                            with program.write(ref) as destination:
                                destination[...] = base_values[i:i+1, 2*j:2*j+ref.shape[1]]
                wait_for((*(ref for tensor in index_tensors for ref in tensor.blocks.values()), *cotangents.blocks.values()), 'writable')
                for tensor, values in zip(index_tensors, index_values):
                    for (i, j), ref in tensor.blocks.items():
                        row, column = i * tensor.block_shape[0], j * tensor.block_shape[1]
                        with program.write(ref) as destination:
                            destination[...] = values.reshape(tensor.shape)[row:row+ref.shape[0], column:column+ref.shape[1]]
                for i in range(2):
                    row = i * cotangents.block_shape[0]
                    with program.write(cotangents[i, 0]) as destination:
                        destination[...] = cotangent_values[row:row+destination.shape[0]]
                wait_for(tuple(gradient_results[i] for i in (0, 1, 3)))
                delayed = tuple(range(2, cotangents.grid[0]))
                if gradient_results[2].ready or any(not cotangents[i, 0].writable for i in delayed):
                    raise ArithmeticError('Indexed sum consumer lost independent contribution regions')
                for i in (0, 1, 3):
                    if not np.array_equal(gradient_results[i].array, expected[i]):
                        raise ArithmeticError('Early indexed sum consumer differs')
                print(json.dumps(dict(event='xonotic_indexed_sum_early', case=name, generation=generation,
                    shape_only_primal=not name.endswith('scatter'), withheld_contribution_blocks=delayed,
                    withheld_base_row=2 if name in scatter_bases else None,
                    output=[gradient_results[i].array.tolist() for i in (0, 1, 3)])), flush=True)
                if name in scatter_bases:
                    for (i, j), ref in backing.blocks.items():
                        if i == 2:
                            if not ref.writable:
                                raise ArithmeticError('Scatter consumed an unpublished base fragment')
                            with program.write(ref) as destination:
                                destination[...] = base_values[i:i+1, 2*j:2*j+ref.shape[1]]
                    if gradient_results[2].ready:
                        raise ArithmeticError('Scatter ignored its delayed updates after base publication')
                for i in delayed:
                    row = i * cotangents.block_shape[0]
                    with program.write(cotangents[i, 0]) as destination:
                        destination[...] = cotangent_values[row:row+destination.shape[0]]
                wait_for(gradient_results)
                if not np.array_equal(gradient_results[2].array, expected[2]):
                    raise ArithmeticError('Duplicate-index sum consumer differs')
                print(json.dumps(dict(event='xonotic_indexed_sum_complete', case=name, generation=generation,
                    output=[result.array.tolist() for result in gradient_results])), flush=True)
                if not generation:
                    for result in gradient_results:
                        result.consume()
        for gram, storage, observations, generations in xonotic_neighborhoods:
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in storage for ref in tensor.blocks.values()), 'writable')
                started = time.monotonic_ns()
                for operand, (tensor, data) in enumerate(zip(storage, values)):
                    for (i, j), ref in tensor.blocks.items():
                        if i != 2 or operand in (3, 4):
                            with program.write(ref) as destination:
                                destination[...] = data[i:i+1]
                early = tuple(result for target, results in enumerate(observations) for i, j, result in results
                              if i != 2 or (not gram and target in (1, 2)))
                wait_for(early)
                early_ms = (time.monotonic_ns() - started) / 1e6
                for target, (results, reference) in enumerate(zip(observations, expected)):
                    for i, j, result in results:
                        if (i != 2 or (not gram and target in (1, 2))) and not np.allclose(result.array, reference[i:i+1, j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5):
                            raise ArithmeticError('Neighborhood early consumer differs')
                if any(result.ready for target in (0, 3) for i, j, result in observations[target] if i == 2) or any(not storage[i][2, 0].writable for i in (0, 1, 2, 5)):
                    raise ArithmeticError('Neighborhood consumed a withheld source or cotangent row')
                print(json.dumps(dict(event='xonotic_neighborhood_early', gram=gram, generation=generation,
                    withheld_row=2, consumers=len(observations), elapsed_ms=early_ms)), flush=True)
                for operand in (0, 1, 2, 5):
                    with program.write(storage[operand][2, 0]) as destination:
                        destination[...] = values[operand][2:3]
                wait_for(tuple(result for results in observations for i, j, result in results))
                for results, reference in zip(observations, expected):
                    for i, j, result in results:
                        if not np.allclose(result.array, reference[i:i+1, j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5):
                            raise ArithmeticError('Neighborhood forward or derivative consumer differs after reuse')
                print(json.dumps(dict(event='xonotic_neighborhood_complete', gram=gram, generation=generation,
                    elapsed_ms=(time.monotonic_ns() - started) / 1e6,
                    output=[[(i, j, result.array.tolist()) for i, j, result in results] for results in observations])), flush=True)
                for results in observations:
                    for i, j, result in results:
                        result.consume()
        if xonotic_expert is not None:
            storage, observations, generations = xonotic_expert
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in storage for ref in tensor.blocks.values()), 'writable')
                started = time.monotonic_ns()
                for operand, (tensor, data) in enumerate(zip(storage, values)):
                    for (i, j), ref in tensor.blocks.items():
                        row, column = i * tensor.block_shape[0], j * tensor.block_shape[1]
                        if operand == 2 or (row < 6 if operand == 1 else row != 2):
                            with program.write(ref) as destination:
                                destination[...] = data[row:row+ref.shape[0], column:column+ref.shape[1]]
                early = tuple(result for target, results in enumerate(observations) for i, j, result in results
                              if (i < 6 if target == 2 else i != 2))
                wait_for(early)
                for target, results in enumerate(observations):
                    for i, j, result in results:
                        if (i < 6 if target == 2 else i != 2) and not np.allclose(
                                result.array, expected[target][i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5, equal_nan=True):
                            raise ArithmeticError('Expert early consumer differs')
                if any(result.ready for target, results in enumerate(observations) for i, j, result in results
                       if (i >= 6 if target == 2 else i == 2)):
                    raise ArithmeticError('Expert consumer ignored withheld numerical inputs')
                print(json.dumps(dict(event='xonotic_expert_early', generation=generation, withheld_row=2,
                    withheld_expert=2, empty_expert=1-generation, elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                pending_operand = 1 if not generation else 0
                for operand in (0, 1, 3):
                    if operand == pending_operand:
                        continue
                    tensor, data = storage[operand], values[operand]
                    for (i, j), ref in tensor.blocks.items():
                        row, column = i * tensor.block_shape[0], j * tensor.block_shape[1]
                        if (row >= 6 if operand == 1 else row == 2):
                            with program.write(ref) as destination:
                                destination[...] = data[row:row+ref.shape[0], column:column+ref.shape[1]]
                independent_target = 2 if not generation else 1
                independent = tuple(result for i, j, result in observations[independent_target]
                                    if (i >= 6 if independent_target == 2 else i == 2))
                wait_for(independent)
                if any(result.ready for i, j, result in observations[0] if i == 2):
                    raise ArithmeticError('Expert forward result ignored its remaining operand')
                for i, j, result in observations[independent_target]:
                    if not np.allclose(result.array, expected[independent_target][i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5, equal_nan=True):
                        raise ArithmeticError('Expert derivative retained an unused primal dependency')
                print(json.dumps(dict(event='xonotic_expert_independent_derivative', generation=generation,
                    target='weights' if independent_target == 2 else 'rows', withheld_operand=pending_operand,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                tensor, data = storage[pending_operand], values[pending_operand]
                for (i, j), ref in tensor.blocks.items():
                    row, column = i * tensor.block_shape[0], j * tensor.block_shape[1]
                    if (row >= 6 if pending_operand == 1 else row == 2):
                        with program.write(ref) as destination:
                            destination[...] = data[row:row+ref.shape[0], column:column+ref.shape[1]]
                wait_for(tuple(result for results in observations for i, j, result in results))
                for target, results in enumerate(observations):
                    for i, j, result in results:
                        if not np.allclose(result.array, expected[target][i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5, equal_nan=True):
                            raise ArithmeticError('Expert forward or derivative differs after reuse')
                print(json.dumps(dict(event='xonotic_expert_complete', generation=generation,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6,
                    output=[[(i,j,result.array.tolist()) for i,j,result in results] for results in observations])), flush=True)
                for results in observations:
                    for i, j, result in results:
                        result.consume()
        if xonotic_grouped_outer is not None:
            x_source, g_source, index_source, observations, first_function, last_function, generations = xonotic_grouped_outer
            outer_trace = program.trace
            published_rows = set()
            for tensor in (x_source, g_source):
                for chunk in (0, 1):
                    ref = tensor[chunk, 0]
                    region = program.native.tensor_rows(ref.view.tensor, ref.view.extent)
                    published_rows.update(range(region.first, region.first+region.count))
            readers = tuple(index for index in range(first_function, last_function)
                if any(row in published_rows for region in outer_trace[index]['inputs']
                       for row in range(region['first'], region['first']+region['count'])) or
                   any(entry['role'] == 1 and any(row in published_rows for row in range(entry['first'], entry['first']+entry['count']))
                       for entry in outer_trace[index]['indexed']))
            print(json.dumps(dict(event='grouped_outer_setup', first_function=first_function, last_function=last_function,
                rows=67, chunk_rows=33, experts=5, input_width=5, output_width=7)), flush=True)
            for generation, (x, g, selected, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in (x_source, g_source, index_source) for ref in tensor.blocks.values()), 'writable')
                started = time.monotonic_ns()
                for (i,j),ref in index_source.blocks.items():
                    row = i*index_source.block_shape[0]
                    with program.write(ref) as destination:
                        destination[...] = selected[row:row+ref.shape[0]]
                for tensor, data in ((x_source, x), (g_source, g)):
                    for chunk in (0, 1):
                        ref = tensor[chunk, 0]
                        row = chunk*tensor.block_shape[0]
                        with program.write(ref) as destination:
                            destination[...] = data[row:row+ref.shape[0]]
                completed_functions = wait_completed(readers, started)
                if not x_source[2, 0].writable or not g_source[2, 0].writable:
                    raise ArithmeticError('Grouped outer product consumed an unpublished final source chunk')
                late_group = int(selected[66, 0])
                late_group = late_group+5 if late_group < 0 else late_group
                early = tuple((i,j,result) for i,j,result in observations if i != late_group)
                wait_for(tuple(result for i,j,result in early))
                for i,j,result in early:
                    if not np.allclose(result.array, expected[i:i+result.array.shape[0],j:j+result.array.shape[1]], rtol=2e-5, atol=2e-6):
                        raise ArithmeticError('Grouped outer product stalled or corrupted an unrelated destination')
                print(json.dumps(dict(event='grouped_outer_partial', generation=generation,
                    source_rows=sorted(published_rows), completed_functions=completed_functions, withheld_row=66,
                    withheld_group=late_group, output=[(i,j,result.array.tolist()) for i,j,result in early],
                    elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                for tensor, data in ((x_source, x), (g_source, g)):
                    with program.write(tensor[2, 0]) as destination:
                        destination[...] = data[66:]
                wait_for(tuple(result for i,j,result in observations))
                for i,j,result in observations:
                    if not np.allclose(result.array, expected[i:i+result.array.shape[0],j:j+result.array.shape[1]], rtol=2e-5, atol=2e-6):
                        raise ArithmeticError('Grouped outer-product reduction differs from its valid contribution domain')
                print(json.dumps(dict(event='grouped_outer_complete', generation=generation,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6,
                    output=[(i,j,result.array.tolist()) for i,j,result in observations])), flush=True)
                for i,j,result in observations:
                    result.consume()
        if xonotic_indexed_context is not None:
            wait_for(xonotic_indexed_context)
            actual = tuple(result.array.item() for result in xonotic_indexed_context)
            if actual != (0, 1, 1) or any(result.array.dtype != np.dtype('int64') for result in xonotic_indexed_context):
                raise ArithmeticError(f'Indexed reduction accumulator/cast mismatch: {actual}')
            print(json.dumps(dict(event='composed_indexed_dtype_context', output=actual)), flush=True)
            for result in xonotic_indexed_context:
                result.consume()
        for dynamic, weight_dtype, source, selected, bias, observations, first_function, last_function, generations in xonotic_composed_indexed:
            contractions, epilogues = observations
            print(json.dumps(dict(event='composed_indexed_setup', dynamic=dynamic, weight_dtype=weight_dtype,
                first_function=first_function, last_function=last_function)), flush=True)
            for generation, (values, selections, bias_values, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in (source, selected, bias) for ref in tensor.blocks.values()), 'writable')
                started = time.monotonic_ns()
                if dynamic:
                    for (i,j),ref in selected.blocks.items():
                        with program.write(ref) as destination:
                            destination[...] = selections[i:i+1]
                for row in (generation, 1-generation):
                    for (i,j),ref in source.blocks.items():
                        if i == row:
                            column = j*source.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = values[i:i+1, column:column+ref.shape[1]]
                    wait_for(tuple(result for i,j,result in contractions if i == row))
                    for i,j,result in contractions:
                        if i == row and not np.allclose(result.array, expected[0][i:i+1,j:j+result.array.shape[1]], rtol=2e-5, atol=2e-6):
                            raise ArithmeticError('Composable indexed contraction differs')
                        if i != row and row == generation and result.ready:
                            raise ArithmeticError('Composable indexed contraction consumed an unpublished row')
                    if any(result.ready for i,j,result in epilogues if i == row):
                        raise ArithmeticError('Composable indexed epilogue consumed an unpublished bias')
                    print(json.dumps(dict(event='composed_indexed_contraction', dynamic=dynamic, weight_dtype=weight_dtype,
                        generation=generation, row=row, bias_absent=True, elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                    for (i,j),ref in bias.blocks.items():
                        if i == row:
                            column = j*bias.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = bias_values[i:i+1,column:column+ref.shape[1]]
                    wait_for(tuple(result for i,j,result in epilogues if i == row))
                    for i,j,result in epilogues:
                        if i == row and not np.allclose(result.array, expected[1][i:i+1,j:j+result.array.shape[1]], rtol=2e-5, atol=2e-6):
                            raise ArithmeticError('Composable indexed contraction epilogue differs')
                        if i != row and row == generation and result.ready:
                            raise ArithmeticError('Composable indexed epilogue consumed an unpublished input row')
                    print(json.dumps(dict(event='composed_indexed_epilogue', dynamic=dynamic, weight_dtype=weight_dtype,
                        generation=generation, row=row, elapsed_ms=(time.monotonic_ns()-started)/1e6,
                        output=[(i,j,result.array.tolist()) for i,j,result in epilogues if i == row])), flush=True)
                for results in observations:
                    for i,j,result in results:
                        result.consume()
        if xonotic_alias is not None:
            storage, observations = xonotic_alias
            for generation in range(2):
                wait_for(tuple(storage.blocks.values()), 'writable')
                values = np.arange(8, dtype=np.float32).reshape(2, 4) + np.float32(generation*10)
                expected = values.reshape(4, 2).T*np.float32(2)+np.float32(1)
                started = time.monotonic_ns()
                for row in (generation, 1-generation):
                    with program.write(storage[row, 0]) as destination:
                        destination[...] = values[row:row+1]
                    wait_for(tuple(result for i,j,result in observations if j//2 == row))
                    for i,j,result in observations:
                        if j//2 == row:
                            if not np.array_equal(result.array, expected[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                                raise ArithmeticError('Replicated fragmented alias lost its offsets or strides')
                        elif row == generation and result.ready:
                            raise ArithmeticError('Replicated alias consumed an unpublished source extent')
                    print(json.dumps(dict(event='xonotic_alias_early' if row == generation else 'xonotic_alias_complete',
                        generation=generation, published_row=row, remote=not args.local,
                        elapsed_ms=(time.monotonic_ns()-started)/1e6,
                        output=[(i,j,result.array.tolist()) for i,j,result in observations if j//2 == row])), flush=True)
                for i,j,result in observations:
                    result.consume()
        for dtype_name, storage, observations, first_function, last_function, generations in xonotic_ordering:
            ordering_trace = program.trace
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(storage.blocks.values()), 'writable')
                first_ref = storage[generation, 0]
                source = program.native.tensor_rows(first_ref.view.tensor, first_ref.view.extent)
                source_rows = set(range(source.first, source.first+source.count))
                readers = tuple(index for index in range(first_function, last_function)
                    if any(row in source_rows for region in ordering_trace[index]['inputs']
                           for row in range(region['first'], region['first']+region['count'])))
                started = time.monotonic_ns()
                with program.write(first_ref) as destination:
                    destination[...] = values[generation:generation+1, :first_ref.shape[1]]
                partial = wait_completed(readers, started)
                if any(result.ready for results in observations for i,j,result in results):
                    raise ArithmeticError('Ordering finalized before the rest of its axis arrived')
                print(json.dumps(dict(event='xonotic_ordering_source_partial', dtype=dtype_name, length=storage.shape[1], generation=generation,
                    source_rows=sorted(source_rows), completed_functions=partial, elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                for row in (generation, 1-generation):
                    for (i, j), ref in storage.blocks.items():
                        if i == row and (row != generation or j != 0):
                            column = j*storage.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = values[i:i+1, column:column+ref.shape[1]]
                    wait_for(tuple(result for output_index, results in enumerate(observations) for i,j,result in results
                                   if (i//storage.shape[1] if output_index == 6 else 0 if output_index == 5 else j if output_index == 4 else i) == row))
                    for output_index, (results, reference) in enumerate(zip(observations, expected)):
                        for i,j,result in results:
                            source_row = i//storage.shape[1] if output_index == 6 else 0 if output_index == 5 else j if output_index == 4 else i
                            if source_row == row:
                                target = reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]
                                if result.array.dtype != target.dtype or not np.array_equal(result.array, target, equal_nan=True):
                                    raise ArithmeticError(f'Stable ordering or gathered values differ: {dtype_name}, output={output_index}')
                                if result.array.dtype.kind == 'f' and not np.array_equal(np.signbit(result.array[target == 0]), np.signbit(target[target == 0])):
                                    raise ArithmeticError('Stable ordering changed the identity of a signed zero')
                            elif row == generation and result.ready:
                                raise ArithmeticError('Ordering consumed an unpublished independent row')
                    print(json.dumps(dict(event='xonotic_ordering_early' if row == generation else 'xonotic_ordering_complete',
                        dtype=dtype_name, length=storage.shape[1], generation=generation, published_row=row, elapsed_ms=(time.monotonic_ns()-started)/1e6,
                        output=[[(i,j,result.array.tolist()) for i,j,result in results
                                 if (i//storage.shape[1] if output_index == 6 else 0 if output_index == 5 else j if output_index == 4 else i) == row]
                                for output_index,results in enumerate(observations)])), flush=True)
                for results in observations:
                    for i,j,result in results:
                        result.consume()
        for dtype_name, storage, observations, rows_per_batch, empty_results, mixed_results, generations in xonotic_integer_dots:
            wait_for(empty_results)
            if any(np.any(result.array) or result.array.dtype != np.dtype(dtype_name) for result in empty_results):
                raise ArithmeticError('Empty integer contraction did not produce its typed zero identity')
            for result in empty_results:
                result.consume()
            wait_for(mixed_results)
            for result, expected, scalar in zip(mixed_results, (1, 1.25), ('int32', 'float32')):
                if result.array.dtype != np.dtype(scalar) or result.array.item() != expected:
                    raise ArithmeticError('Mixed contraction lost numerical conversion or output dtype')
                print(json.dumps(dict(event='xonotic_mixed_integer_dot', dtype=scalar, output=result.array.tolist())), flush=True)
                result.consume()
            for generation, (values, expected) in enumerate(generations):
                started = time.monotonic_ns()
                wait_for(tuple(storage.blocks.values()), 'writable')
                for batch in (generation, 1-generation):
                    for (i, j), ref in storage.blocks.items():
                        if i//3 == batch:
                            column = j*storage.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = values[i:i+ref.shape[0], column:column+ref.shape[1]]
                    wait_for(tuple(result for results, batch_rows in zip(observations, rows_per_batch) for i,j,result in results if i//batch_rows == batch))
                    for results, reference, batch_rows in zip(observations, expected, rows_per_batch):
                        for i, j, result in results:
                            if i//batch_rows == batch:
                                if result.array.dtype != np.dtype(dtype_name) or not np.array_equal(result.array,
                                        reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                                    raise ArithmeticError(f'Integer contraction lost typed modular or Boolean semantics: {dtype_name}')
                            elif batch == generation and result.ready:
                                raise ArithmeticError('Integer contraction consumed an unpublished batch')
                    print(json.dumps(dict(event='xonotic_integer_dot_early' if batch == generation else 'xonotic_integer_dot_complete',
                        dtype=dtype_name, generation=generation, published_batch=batch, elapsed_ms=(time.monotonic_ns()-started)/1e6,
                        output=[[(i,j,result.array.tolist()) for i,j,result in results if i//batch_rows == batch] for results,batch_rows in zip(observations, rows_per_batch)])), flush=True)
                for results in observations:
                    for i,j,result in results:
                        result.consume()
        if xonotic_random is not None:
            known_answers, key_storage, storage, observations, high_keys, high_results, generations = xonotic_random
            for results, expected in known_answers:
                wait_for(results)
                actual = tuple(result.array.item() for result in results)
                if actual != expected:
                    raise ArithmeticError('Philox differs from its published known-answer vector')
                print(json.dumps(dict(event='philox_known_answer', words=actual)), flush=True)
                for result in results:
                    result.consume()
            generated, consumers = observations
            for generation, (key, values, reference, high_reference) in enumerate(generations):
                started = time.monotonic_ns()
                wait_for((*key_storage.blocks.values(), *storage.blocks.values(), *high_keys.blocks.values()), 'writable')
                with program.write(key_storage[0, 0]) as destination:
                    destination[...] = key
                with program.write(high_keys[generation, 0]) as destination:
                    destination[...] = key
                wait_for((high_results[generation], *(result for i, j, result in generated)))
                if high_results[1-generation].ready:
                    raise ArithmeticError('Random producer consumed an unpublished key row')
                for i, j, result in generated:
                    if not np.allclose(result.array, reference[i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=5e-6, atol=2e-6):
                        raise ArithmeticError('Normal generator differs across tile or pair boundaries')
                if any(result.ready for i, j, result in consumers):
                    raise ArithmeticError('Random consumer read an unpublished input')
                expected = (reference + values) * np.float32(2)
                for row in (generation, 1-generation):
                    if row != generation:
                        with program.write(high_keys[row, 0]) as destination:
                            destination[...] = key
                        wait_for((high_results[row],))
                    if not np.allclose(high_results[row].array, high_reference[row:row+1], rtol=5e-6, atol=4e-6):
                        raise ArithmeticError('Normal generator discarded its high counter word')
                    for (i, j), ref in storage.blocks.items():
                        if i == row:
                            column = j * storage.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = values[i:i+1, column:column+ref.shape[1]]
                    wait_for(tuple(result for i, j, result in consumers if i == row))
                    for i, j, result in consumers:
                        if i == row:
                            if not np.allclose(result.array, expected[i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=5e-6, atol=4e-6):
                                raise ArithmeticError('Random downstream consumer differs after key reuse')
                        elif row == generation and result.ready:
                            raise ArithmeticError('Random consumer lost independent row progress')
                    print(json.dumps(dict(event='xonotic_random_early' if row == generation else 'xonotic_random_complete',
                        generation=generation, key=key.tolist(), published_row=row, elapsed_ms=(time.monotonic_ns()-started)/1e6,
                        high_ordinal_output=high_results[row].array.tolist(),
                        output=[(i,j,result.array.tolist()) for i,j,result in consumers if i == row])), flush=True)
                for result in (*high_results, *(result for results in observations for i,j,result in results)):
                    result.consume()
        if xonotic_ranges is not None:
            storage, observations, references, empty_results, mean_result = xonotic_ranges
            wait_for(empty_results)
            empty_expected = (0, False, True, -np.inf, np.inf, np.nan) * 3
            if any(not np.array_equal(result.array, np.full(result.array.shape, expected), equal_nan=True)
                   for result, expected in zip(empty_results, empty_expected)):
                raise ArithmeticError('Empty range reduction identity differs')
            print(json.dumps(dict(event='xonotic_empty_ranges', shapes=[[0]]*3, identities=[result.array.item() for result in empty_results])), flush=True)
            for result in empty_results:
                result.consume()
            generated_results, consumer_results = observations[:len(storage)], observations[len(storage):]
            for generation in range(2):
                started = time.monotonic_ns()
                wait_for(tuple(ref for tensor in storage for ref in tensor.blocks.values()), 'writable')
                wait_for(tuple(result for results in generated_results for i, j, result in results))
                for results, expected in zip(generated_results, references):
                    expected = expected.reshape(1, 10)
                    for i, j, result in results:
                        if not np.array_equal(result.array, expected[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                            raise ArithmeticError('Generated range lost exact index values')
                if any(result.ready for results in consumer_results for i, j, result in results):
                    raise ArithmeticError('Range consumer read an unpublished input')
                values = tuple((np.arange(10).reshape(2, 5) + generation).astype(tensor.dtype) for tensor in storage)
                expected = tuple(reference.astype(np.float32) + value.astype(np.float32) + np.float32(14) if value.dtype.kind == 'f' else
                                 reference + value + np.array(14, dtype=value.dtype) for reference, value in zip(references, values))
                for row in (generation, 1-generation):
                    for tensor, value in zip(storage, values):
                        for (i, j), ref in tensor.blocks.items():
                            if i == row:
                                column = j * tensor.block_shape[1]
                                with program.write(ref) as destination:
                                    destination[...] = value[i:i+1, column:column+ref.shape[1]]
                    wait_for(tuple(result for results in consumer_results for i, j, result in results if i == row))
                    for results, reference in zip(consumer_results, expected):
                        for i, j, result in results:
                            if i == row:
                                if not np.array_equal(result.array, reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                                    raise ArithmeticError('Range consumer differs after symbolic dimension composition')
                            elif row == generation and result.ready:
                                raise ArithmeticError('Range consumer lost independent row progress')
                    if row == generation:
                        if mean_result.ready:
                            raise ArithmeticError('Integer mean consumed a withheld range input row')
                    else:
                        wait_for((mean_result,))
                        if mean_result.array.dtype != np.dtype('float32') or mean_result.array.item() != 4.75 + generation:
                            raise ArithmeticError('Integer mean lost its fractional float32 result')
                    print(json.dumps(dict(event='xonotic_range_early' if row == generation else 'xonotic_range_complete',
                        generation=generation, published_row=row, elapsed_ms=(time.monotonic_ns()-started)/1e6,
                        integer_mean=None if row == generation else mean_result.array.item(),
                        output=[[(i, j, result.array.tolist()) for i, j, result in results if i == row] for results in consumer_results])), flush=True)
                mean_result.consume()
                for results in observations:
                    for i, j, result in results:
                        result.consume()
        if xonotic_elementary is not None:
            storage, operations, observations, generations = xonotic_elementary
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in storage for ref in tensor.blocks.values()), 'writable')
                early_row = generation
                started = time.monotonic_ns()
                for tensor, data in zip(storage, values):
                    for (i, j), ref in tensor.blocks.items():
                        if i == early_row:
                            column = j * tensor.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = data[i:i+1, column:column+ref.shape[1]]
                wait_for(tuple(result for results in observations for i, j, result in results if i == early_row))
                for operation, results, reference in zip(operations, observations, expected):
                    for i, j, result in results:
                        if i == early_row:
                            target = reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]
                            correct = np.array_equal(result.array, target) if result.array.dtype.kind in 'iu' else np.allclose(result.array, target, rtol=2e-5, atol=1e-12, equal_nan=True)
                            if operation in ('arcsinh', 'expm1', 'log1p', 'sqrt', 'floor_divide'):
                                correct &= np.array_equal(np.signbit(result.array[target == 0]), np.signbit(target[target == 0]))
                            if not correct:
                                raise ArithmeticError(f'Elementary early consumer differs: {operation}, region={(i,j)}, actual={result.array.tolist()}, expected={target.tolist()}')
                        elif result.ready:
                            raise ArithmeticError('Elementary consumer read a withheld row')
                print(json.dumps(dict(event='xonotic_elementary_early', generation=generation, withheld_row=1-early_row,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                for tensor, data in zip(storage, values):
                    for (i, j), ref in tensor.blocks.items():
                        if i != early_row:
                            column = j * tensor.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = data[i:i+1, column:column+ref.shape[1]]
                wait_for(tuple(result for results in observations for i, j, result in results))
                for operation, results, reference in zip(operations, observations, expected):
                    for i, j, result in results:
                        target = reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]
                        correct = np.array_equal(result.array, target) if result.array.dtype.kind in 'iu' else np.allclose(result.array, target, rtol=2e-5, atol=1e-12, equal_nan=True)
                        if operation in ('arcsinh', 'expm1', 'log1p', 'sqrt', 'floor_divide'):
                            correct &= np.array_equal(np.signbit(result.array[target == 0]), np.signbit(target[target == 0]))
                        if not correct:
                            raise ArithmeticError(f'Elementary consumer differs after reuse: {operation}, region={(i,j)}, actual={result.array.tolist()}, expected={target.tolist()}')
                print(json.dumps(dict(event='xonotic_elementary_complete', generation=generation,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6,
                    output={operation: [(i,j,result.array.tolist()) for i,j,result in results] for operation,results in zip(operations,observations)})), flush=True)
                for results in observations:
                    for i, j, result in results:
                        result.consume()
        if xonotic_reductions is not None:
            storage, cases, observations, generations = xonotic_reductions
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in storage for ref in tensor.blocks.values()), 'writable')
                started = time.monotonic_ns()
                early_row = generation
                for tensor, data in zip(storage, values):
                    for (i, j), ref in tensor.blocks.items():
                        if i == early_row:
                            column = j * tensor.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = data[i:i+1, column:column+ref.shape[1]]
                wait_for(tuple(result for case, results in zip(cases, observations) for i, j, result in results
                               if case[0] > 1 and i == early_row))
                for case, results, reference in zip(cases, observations, expected):
                    for i, j, result in results:
                        if case[0] > 1 and i == early_row:
                            if not np.array_equal(result.array, reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                                raise ArithmeticError('Typed reduction early consumer differs')
                        elif case[0] > 1 and result.ready:
                            raise ArithmeticError('Typed reduction consumed a withheld row')
                print(json.dumps(dict(event='xonotic_typed_reductions_early', generation=generation,
                    withheld_row=1-early_row, ranks=[1,2,3], elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                for tensor, data in zip(storage, values):
                    for (i, j), ref in tensor.blocks.items():
                        if i != early_row:
                            column = j * tensor.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = data[i:i+1, column:column+ref.shape[1]]
                wait_for(tuple(result for results in observations for i, j, result in results))
                for results, reference in zip(observations, expected):
                    for i, j, result in results:
                        if not np.array_equal(result.array, reference[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                            raise ArithmeticError('Typed reduction consumer differs after reuse')
                print(json.dumps(dict(event='xonotic_typed_reductions_complete', generation=generation,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6,
                    output=[dict(rank=case[0], operand=case[3], operation=case[4], values=[result.array.tolist() for i,j,result in results])
                            for case, results in zip(cases, observations)])), flush=True)
                for results in observations:
                    for i, j, result in results:
                        result.consume()
        if xonotic_boolean is not None:
            storage, observations, generations = xonotic_boolean
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(storage.blocks.values()), 'writable')
                early_row = generation
                with program.write(storage[early_row, 0]) as destination:
                    destination[...] = values[early_row:early_row+1]
                wait_for(tuple(result for i, j, result in observations if i // 2 == early_row))
                for i, j, result in observations:
                    if i // 2 == early_row:
                        if not np.array_equal(result.array, expected[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                            raise ArithmeticError('Boolean broadcast early consumer differs')
                    elif result.ready:
                        raise ArithmeticError('Boolean broadcast consumed a withheld source row')
                print(json.dumps(dict(event='xonotic_boolean_early', generation=generation,
                    withheld_row=1-early_row, nan_truth=True)), flush=True)
                with program.write(storage[1-early_row, 0]) as destination:
                    destination[...] = values[1-early_row:2-early_row]
                wait_for(tuple(result for i, j, result in observations))
                for i, j, result in observations:
                    if not np.array_equal(result.array, expected[i:i+result.array.shape[0], j:j+result.array.shape[1]]):
                        raise ArithmeticError('Boolean broadcast consumer differs after reuse')
                print(json.dumps(dict(event='xonotic_boolean_complete', generation=generation,
                    output=[(i,j,result.array.tolist()) for i,j,result in observations])), flush=True)
                for i, j, result in observations:
                    result.consume()
        if xonotic_batched is not None:
            storage, observations, generations = xonotic_batched
            for generation, (values, expected) in enumerate(generations):
                wait_for(tuple(ref for tensor in storage for ref in tensor.blocks.values()), 'writable')
                early_batch = generation
                started = time.monotonic_ns()
                for operand, (tensor, data) in enumerate(zip(storage, values)):
                    for (i, j), ref in tensor.blocks.items():
                        if operand == 1 or i // (5 if operand == 0 else 3) == early_batch:
                            column = j * tensor.block_shape[1]
                            with program.write(ref) as destination:
                                destination[...] = data[i:i+1, column:column+ref.shape[1]]
                early = tuple(result for target, results in enumerate(observations[:2]) for i, j, result in results
                              if i // (3 if target == 0 else 5) == early_batch)
                wait_for(early)
                for target, results in enumerate(observations):
                    for i, j, result in results:
                        if target < 2 and i // (3 if target == 0 else 5) == early_batch:
                            if not np.allclose(result.array, expected[target][i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5):
                                raise ArithmeticError('Batched contraction early consumer differs')
                        elif result.ready:
                            raise ArithmeticError('Batched contraction consumed a withheld batch')
                print(json.dumps(dict(event='xonotic_batched_early', generation=generation,
                    withheld_batch=1-early_batch, singleton_right_batch=True,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                tensor, data = storage[2], values[2]
                for (i, j), ref in tensor.blocks.items():
                    if i // 3 != early_batch:
                        column = j * tensor.block_shape[1]
                        with program.write(ref) as destination:
                            destination[...] = data[i:i+1, column:column+ref.shape[1]]
                wait_for(tuple(result for i, j, result in observations[1]))
                for i, j, result in observations[1]:
                    if not np.allclose(result.array, expected[1][i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5):
                        raise ArithmeticError('Batched input derivative retained its unused primal')
                if any(result.ready for i, j, result in observations[0] if i // 3 != early_batch) or any(result.ready for i, j, result in observations[2]):
                    raise ArithmeticError('Batched forward or weight derivative ignored its missing input')
                print(json.dumps(dict(event='xonotic_batched_independent_derivative', generation=generation,
                    withheld_left_batch=1-early_batch, elapsed_ms=(time.monotonic_ns()-started)/1e6)), flush=True)
                tensor, data = storage[0], values[0]
                for (i, j), ref in tensor.blocks.items():
                    if i // 5 != early_batch:
                        column = j * tensor.block_shape[1]
                        with program.write(ref) as destination:
                            destination[...] = data[i:i+1, column:column+ref.shape[1]]
                wait_for(tuple(result for results in observations for i, j, result in results))
                for target, results in enumerate(observations):
                    for i, j, result in results:
                        if not np.allclose(result.array, expected[target][i:i+result.array.shape[0], j:j+result.array.shape[1]], rtol=2e-5, atol=2e-5):
                            raise ArithmeticError('Batched contraction or broadcast derivative differs after reuse')
                print(json.dumps(dict(event='xonotic_batched_complete', generation=generation,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6,
                    output=[[(i,j,result.array.tolist()) for i,j,result in results] for results in observations])), flush=True)
                for results in observations:
                    for i, j, result in results:
                        result.consume()
        if xonotic_take_gradient is not None:
            take_indices, take_cotangents, take_results, generations = xonotic_take_gradient
            for generation, (index_values, cotangent_values, expected) in enumerate(generations):
                wait_for((*take_indices.blocks.values(), *take_cotangents.blocks.values()), 'writable')
                for i in range(4):
                    with program.write(take_indices[i, 0]) as destination:
                        destination[...] = index_values.reshape(4, 1)[i:i+1]
                for i in (0, 2):
                    with program.write(take_cotangents[i, 0]) as destination:
                        destination[...] = cotangent_values.reshape(4, 1)[i:i+1]
                wait_for((take_results[0],))
                if take_results[1].ready or any(not take_cotangents[i, 0].writable for i in (1, 3)):
                    raise ArithmeticError('Take derivative lost independent cotangent regions')
                if not np.array_equal(take_results[0].array, expected[:1]):
                    raise ArithmeticError('Early take derivative consumer differs')
                print(json.dumps(dict(event='xonotic_take_gradient_early', generation=generation,
                    primal_operand_allocated=False, withheld_cotangent_blocks=[1, 3],
                    output=take_results[0].array.tolist())), flush=True)
                for i in (1, 3):
                    with program.write(take_cotangents[i, 0]) as destination:
                        destination[...] = cotangent_values.reshape(4, 1)[i:i+1]
                wait_for(take_results)
                if not np.array_equal(take_results[1].array, expected[1:]):
                    raise ArithmeticError('Broadcast take derivative consumer differs')
                print(json.dumps(dict(event='xonotic_take_gradient_complete', generation=generation,
                    output=[result.array.tolist() for result in take_results])), flush=True)
                if not generation:
                    for result in take_results:
                        result.consume()
        for generation in range(4):
            empty = generation == 2
            routing = np.resize(np.array([0, 2, 0, 3], dtype=np.int64), updates_count).reshape(-1, 1)
            if destinations_count > 4:
                routing[4:last_start, 0] = 4 + np.arange(max(0, last_start-4)) % (destinations_count-4)
            routing[last_start:] = 2
            if generation == 1:
                routing = np.where(routing == 0, 3, np.where(routing == 3, 0, routing))
                routing[1, 0] = 2**32 + 2
            if empty:
                routing.fill(2**32 + 2)
            update_values = (1 + generation + np.arange(updates_count) % 31).astype(dtype)[:, None]
            lookup_values = np.arange(updates_count, dtype=np.int64)
            for start in range(0, updates_count, update_tile):
                length = min(update_tile, updates_count-start)
                lookup_values[start:start+length] = start + (length-1-np.arange(length)+generation) % length
            delayed = routing[:, 0] == 2
            lookup_values[delayed] = last_start + np.arange(np.count_nonzero(delayed)) % (updates_count-last_start)
            lookup_values[~scatter_valid[:, 0]] = last_start
            lookup_values = lookup_values[:, None]
            expected = np.zeros((destinations_count, 1), dtype=np.float32)
            selected = scatter_valid[:, 0] & (routing[:, 0] < destinations_count)
            np.add.at(expected, routing[selected, 0], update_values[lookup_values[selected, 0]].astype(np.float32)*(2+generation)+1)
            expected = (expected.astype(dtype)*2).astype(dtype)
            scatter_start = time.monotonic_ns()
            wait_for((*scatter_indices.blocks.values(), *scatter_updates.blocks.values(),
                      *scatter_factors.blocks.values(), *scatter_lookup.blocks.values(),
                      *scatter_masks.blocks.values()), 'writable')
            for i in range(last_chunk+1):
                with program.write(scatter_indices[i, 0]) as destination:
                    destination[...] = routing[update_tile*i:update_tile*(i+1)]
                with program.write(scatter_lookup[i, 0]) as destination:
                    destination[...] = lookup_values[update_tile*i:update_tile*(i+1)]
            for i in range(0 if empty else last_chunk):
                with program.write(scatter_masks[i, 0]) as destination:
                    destination[...] = scatter_valid[update_tile*i:update_tile*(i+1)]
                with program.write(scatter_factors[i, 0]) as destination:
                    destination[...] = 2 + generation
                with program.write(scatter_updates[i, 0]) as destination:
                    destination[...] = update_values[update_tile*i:update_tile*(i+1)]
            observed = tuple(range(destinations_count)) if empty else early_destinations
            wait_for(tuple(scatter_results[i] for i in observed))
            if (not empty and scatter_results[2].ready) or not scatter_updates[last_chunk, 0].writable or not scatter_masks[last_chunk, 0].writable:
                raise ArithmeticError('Delayed scatter contribution was not independent')
            first_scatter_ns = time.monotonic_ns() - scatter_start
            for i in observed:
                if not np.array_equal(scatter_results[i].array, np.broadcast_to(expected[i], (1, 4))):
                    raise ArithmeticError('Early scattered sum or consumer differs')
            for i in range(last_chunk+1) if empty else (last_chunk,):
                with program.write(scatter_masks[i, 0]) as destination:
                    destination[...] = scatter_valid[update_tile*i:update_tile*(i+1)]
                with program.write(scatter_updates[i, 0]) as destination:
                    destination[...] = update_values[update_tile*i:update_tile*(i+1)]
                if (not empty and scatter_results[2].ready) or not scatter_factors[i, 0].writable:
                    raise ArithmeticError('Fused update ignored its missing coefficient operand')
                with program.write(scatter_factors[i, 0]) as destination:
                    destination[...] = 2 + generation
            wait_for((scatter_results[2],))
            if not np.array_equal(scatter_results[2].array, np.broadcast_to(expected[2], (1, 4))):
                raise ArithmeticError('Duplicate or masked scatter contribution differs')
            print(json.dumps(dict(event='indexed_add', generation=generation, lookup_indices=lookup_values[:, 0].tolist(), rows=updates_count, tile=update_tile, destinations=destinations_count, first_consumer_ns=first_scatter_ns,
                complete_ns=time.monotonic_ns()-scatter_start,
                empty=empty, delayed_destination=None if empty else 2, withheld_mask_block=last_chunk,
                result=[result.array.tolist() for result in scatter_results])), flush=True)
            if generation < 3:
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
        report = program.report
        if args.trace:
            Path(args.trace).write_text(json.dumps(dict(compute=program.trace, routes=program.route_trace, transfers=program.transfer_trace), indent=2) + '\n')
        print(json.dumps(dict(event='summary', dtype=args.dtype, depth=args.depth, rows=rows, width=width,
            tile_rows=tile, tile_k=args.tile_k, tile_columns=args.tile_columns, coreml=bool(args.coreml), invocations=args.runs, batch_ms=batch_ms,
            invocations_per_second=args.runs * 1000 / batch_ms, first_section_ms=first_ms,
            completion_ms=summary(completion_observations), samples=args.samples,
            sample_batch_ms=summary(sample_batches), sample_first_section_ms=summary(sample_first), warmup_withheld_section=0,
            max_absolute_error=max(errors), runtime={name: getattr(report, name) for name, _ in report._fields_})), flush=True)



if __name__ == '__main__':
    main()
