from __future__ import annotations

from functools import lru_cache

from . import tensor as mx

PAGE_ROWS = 64
TILE = 32
PHYSICAL, START, VALID, EXPERT = range(4)


@lru_cache(maxsize=None)
def page_slots(capacity):
    return mx.arange(capacity, dtype=mx.int32)


def page_table(counts, row_capacity):
    capacity = 1 << (max(1, (row_capacity + len(counts) * (PAGE_ROWS - 1)) // PAGE_ROWS) - 1).bit_length()
    slots = page_slots(capacity)
    pages = (counts + PAGE_ROWS - 1) // PAGE_ROWS
    ends = mx.cumsum(pages)
    owner = mx.minimum(mx.searchsorted(ends, slots, side="right"), len(counts) - 1)
    page_start = mx.concatenate((mx.zeros((1,), dtype=mx.int32), ends[:-1]))
    row_start = mx.concatenate((mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)[:-1]))
    offset = (slots - page_start[owner]) * PAGE_ROWS
    valid = mx.maximum(0, mx.minimum(PAGE_ROWS, counts[owner] - offset))
    return mx.stop_gradient(mx.stack((slots, row_start[owner] + offset, valid, owner), axis=1).astype(mx.int32))


@lru_cache(maxsize=128)
def dense_table(rows):
    capacity = 1 << (max(1, (rows + PAGE_ROWS - 1) // PAGE_ROWS) - 1).bit_length()
    slots = page_slots(capacity)
    return mx.stack((slots, slots * PAGE_ROWS, mx.clip(rows - slots * PAGE_ROWS, 0, PAGE_ROWS),
                     mx.zeros_like(slots)), axis=1).astype(mx.int32)


def routed_table(experts, expert_count):
    counts = mx.zeros((expert_count,), dtype=mx.int32).at[experts].add(mx.ones(experts.shape, dtype=mx.int32))
    return page_table(counts, experts.size)


def padded(width):
    return (width + TILE - 1) // TILE * TILE


_NEIGHBORHOOD = mx.fast.metal_kernel(
    name="mesh_neighborhood_integral", input_names=["query", "keys", "values", "indices", "weights"],
    output_names=["integrated"],
    source="""
        uint observer = threadgroup_position_in_grid.y;
        uint page = threadgroup_position_in_grid.z;
        uint lane = thread_position_in_grid.x;
        float total[(WIDTH + 31) / 32] = {};
        for (uint item = 0; item < PAGE_ROWS; ++item) {
            uint neighbor = page * PAGE_ROWS + item;
            if (neighbor >= uint(weights_shape[1])) continue;
            uint at = observer * weights_shape[1] + neighbor;
            uint source = uint(indices[at]);
            float product = 0;
            if (GRAM) {
                for (uint d = lane; d < WIDTH; d += 32)
                    product += query[observer * WIDTH + d] * keys[source * WIDTH + d];
                product = simd_sum(product) * rsqrt(float(WIDTH));
            } else {
                product = 1;
            }
            float weight = weights[at] * product;
            for (uint d = lane; d < WIDTH; d += 32)
                total[d / 32] += weight * values[source * WIDTH + d];
        }
        uint pages = max(1, (weights_shape[1] + PAGE_ROWS - 1) / PAGE_ROWS);
        for (uint d = lane; d < WIDTH; d += 32)
            integrated[(observer * pages + page) * WIDTH + d] = total[d / 32];
    """,
)

_NEIGHBORHOOD_VJP = mx.fast.metal_kernel(
    name="mesh_neighborhood_integral_vjp", input_names=["query", "keys", "values", "indices", "weights", "cotangent"],
    output_names=["dquery", "dkeys", "dvalues", "dweights"], atomic_outputs=True,
    source="""
        uint observer = threadgroup_position_in_grid.y;
        uint page = threadgroup_position_in_grid.z;
        uint lane = thread_position_in_grid.x;
        float total[(WIDTH + 31) / 32] = {};
        for (uint item = 0; item < PAGE_ROWS; ++item) {
            uint neighbor = page * PAGE_ROWS + item;
            if (neighbor >= uint(weights_shape[1])) continue;
            uint at = observer * weights_shape[1] + neighbor;
            uint source = uint(indices[at]);
            float product = 0, value_product = 0;
            for (uint d = lane; d < WIDTH; d += 32) {
                if (GRAM)
                    product += query[observer * WIDTH + d] * keys[source * WIDTH + d];
                value_product += cotangent[observer * WIDTH + d] * values[source * WIDTH + d];
            }
            product = GRAM ? simd_sum(product) * rsqrt(float(WIDTH)) : 1;
            value_product = simd_sum(value_product);
            if (lane == 0)
                atomic_store_explicit(&dweights[at], product * value_product, memory_order_relaxed);
            float weight = weights[at];
            for (uint d = lane; d < WIDTH; d += 32) {
                atomic_fetch_add_explicit(&dvalues[source * WIDTH + d],
                    weight * product * cotangent[observer * WIDTH + d], memory_order_relaxed);
                if (GRAM) {
                    float metric = weight * value_product * rsqrt(float(WIDTH));
                    total[d / 32] += metric * keys[source * WIDTH + d];
                    atomic_fetch_add_explicit(&dkeys[source * WIDTH + d],
                        metric * query[observer * WIDTH + d], memory_order_relaxed);
                }
            }
        }
        for (uint d = lane; d < WIDTH; d += 32)
            atomic_fetch_add_explicit(&dquery[observer * WIDTH + d], total[d / 32], memory_order_relaxed);
    """,
)


@lru_cache(maxsize=None)
def neighborhood_operator(width, gram):
    @mx.custom_function
    def operation(query, keys, values, indices, weights):
        pages = max(1, (weights.shape[1] + PAGE_ROWS - 1) // PAGE_ROWS)
        partial = _NEIGHBORHOOD(inputs=[query, keys, values, indices, weights],
            template=[("WIDTH", width), ("GRAM", gram), ("PAGE_ROWS", PAGE_ROWS)],
            grid=(32, len(query), pages), threadgroup=(32, 1, 1),
            output_shapes=[(len(query), pages, width)], output_dtypes=[mx.float32])[0]
        return mx.sum(partial, axis=1)

    @operation.vjp
    def backward(primals, cotangent, output):
        query, keys, values, indices, weights = primals
        pages = max(1, (weights.shape[1] + PAGE_ROWS - 1) // PAGE_ROWS)
        dq, dk, dv, dw = _NEIGHBORHOOD_VJP(inputs=[*primals, cotangent],
            template=[("WIDTH", width), ("GRAM", gram), ("PAGE_ROWS", PAGE_ROWS)], grid=(32, len(query), pages),
            threadgroup=(32, 1, 1), output_shapes=[query.shape, keys.shape, values.shape, weights.shape],
            output_dtypes=[mx.float32] * 4, init_value=0)
        return dq, dk, dv, mx.zeros_like(indices), dw

    return operation


def neighborhood_integral(query, keys, values, indices, weights, gram=True):
    if isinstance(query, mx.Tensor):
        return mx.neighborhood(query, keys, values, indices, weights, gram)
    return neighborhood_operator(values.shape[1], gram)(query, keys, values, indices, weights)


_GATHER = mx.fast.metal_kernel(
    name="mesh_page_gather", input_names=["rows", "indices", "table"], output_names=["pages"],
    source="""
        const int WIDTH = (rows_shape[1] + TILE - 1) / TILE * TILE;
        uint slot = thread_position_in_grid.y;
        uint row = slot % PAGE_ROWS;
        uint entry = slot / PAGE_ROWS;
        uint column = thread_position_in_grid.x;
        bool active = row < uint(table[entry * 4 + 2]);
        uint position = select(uint(indices_shape[0] - 1), uint(table[entry * 4 + 1]) + row, active);
        uint source = uint(indices[position]);
        uint physical = uint(table[entry * 4]);
        float value = rows[source * rows_shape[1] + min(column, uint(rows_shape[1] - 1))];
        pages[(physical * PAGE_ROWS + row) * WIDTH + column] = select(0.0f, value, column < uint(rows_shape[1]));
    """,
)

_SCATTER = mx.fast.metal_kernel(
    name="mesh_page_scatter", input_names=["pages", "indices", "table", "shape"], output_names=["rows"],
    atomic_outputs=True,
    source="""
        uint slot = thread_position_in_grid.y;
        uint row = slot % PAGE_ROWS;
        uint entry = slot / PAGE_ROWS;
        uint column = thread_position_in_grid.x;
        bool active = row < uint(table[entry * 4 + 2]);
        uint position = select(uint(indices_shape[0] - 1), uint(table[entry * 4 + 1]) + row, active);
        uint target = uint(indices[position]);
        uint physical = uint(table[entry * 4]);
        float value = pages[(physical * PAGE_ROWS + row) * pages_shape[2] + column];
        atomic_fetch_add_explicit(&rows[target * shape[1] + column], select(0.0f, value, active), memory_order_relaxed);
    """,
)

_HEADER = """
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;
"""

_PROJECT = mx.fast.metal_kernel(
    name="mesh_page_project", input_names=["pages", "weights", "table"], output_names=["product"],
    header=_HEADER,
    source="""
        uint entry = threadgroup_position_in_grid.y;
        uint physical = uint(table[entry * 4]);
        uint expert = uint(table[entry * 4 + 3]);
        int inner = pages_shape[2];
        int columns = weights_shape[TRANSPOSE ? 1 : 2];
        int first_column = threadgroup_position_in_grid.x * TILE;
        auto a = tensor(const_cast<device float *>(pages + physical * PAGE_ROWS * inner),
            dextents<int, 2>{inner, PAGE_ROWS}, array<int, 2>{1, inner});
        auto b = tensor(const_cast<device float *>(weights + expert * inner * columns),
            dextents<int, 2>{TRANSPOSE ? inner : columns, TRANSPOSE ? columns : inner},
            array<int, 2>{1, TRANSPOSE ? inner : columns});
        auto c = tensor(product + physical * PAGE_ROWS * columns,
            dextents<int, 2>{columns, PAGE_ROWS}, array<int, 2>{1, columns});
        constexpr auto descriptor = matmul2d_descriptor(PAGE_ROWS, TILE, TILE,
            false, TRANSPOSE, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<descriptor, execution_simdgroup> operation;
        auto left = a.slice<TILE, PAGE_ROWS>(0, 0);
        auto right = b.slice<TILE, TILE>(TRANSPOSE ? 0 : first_column, TRANSPOSE ? first_column : 0);
        auto total = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
        #pragma unroll
        for (uint i = 0; i < total.get_capacity(); ++i)
            total[i] = 0;
        for (int k = 0; k < inner; k += TILE) {
            left = a.slice<TILE, PAGE_ROWS>(k, 0);
            right = b.slice<TILE, TILE>(TRANSPOSE ? k : first_column, TRANSPOSE ? first_column : k);
            operation.run(left, right, total);
        }
        total.store(c.slice<TILE, PAGE_ROWS>(first_column, 0));
    """,
)

_CONTRACT = mx.fast.metal_kernel(
    name="mesh_page_contract", input_names=["lhs", "rhs", "table"], output_names=["product"],
    header=_HEADER,
    source="""
        int rows = lhs_shape[2];
        int columns = rhs_shape[2];
        int first_row = threadgroup_position_in_grid.y * TILE;
        int first_column = threadgroup_position_in_grid.x * TILE;
        int expert = threadgroup_position_in_grid.z;
        auto a = tensor(const_cast<device float *>(lhs),
            dextents<int, 2>{rows, PAGE_ROWS}, array<int, 2>{1, rows});
        auto b = tensor(const_cast<device float *>(rhs),
            dextents<int, 2>{columns, PAGE_ROWS}, array<int, 2>{1, columns});
        constexpr auto descriptor = matmul2d_descriptor(TILE, TILE, PAGE_ROWS,
            true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<descriptor, execution_simdgroup> operation;
        auto left = a.slice<TILE, PAGE_ROWS>(first_row, 0);
        auto right = b.slice<TILE, PAGE_ROWS>(first_column, 0);
        auto total = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
        #pragma unroll
        for (uint i = 0; i < total.get_capacity(); ++i)
            total[i] = 0;
        for (int entry = 0; entry < table_shape[0]; ++entry) {
            int physical = table[entry * 4];
            auto pa = tensor(const_cast<device float *>(lhs + physical * PAGE_ROWS * rows),
                dextents<int, 2>{rows, PAGE_ROWS}, array<int, 2>{1, rows});
            auto pb = tensor(const_cast<device float *>(rhs + physical * PAGE_ROWS * columns),
                dextents<int, 2>{columns, PAGE_ROWS}, array<int, 2>{1, columns});
            auto block_left = pa.slice<TILE, PAGE_ROWS>(first_row, 0);
            auto block_right = pb.slice<TILE, PAGE_ROWS>(first_column, 0);
            auto block = operation.template get_destination_cooperative_tensor<decltype(block_left), decltype(block_right), float>();
            #pragma unroll
            for (uint i = 0; i < block.get_capacity(); ++i)
                block[i] = 0;
            operation.run(block_left, block_right, block);
            bool active = (table[entry * 4 + 2] > 0) & (table[entry * 4 + 3] == expert);
            #pragma unroll
            for (uint i = 0; i < total.get_capacity(); ++i)
                total[i] += select(0.0f, block[i], active);
        }
        auto output = tensor(product + expert * rows * columns,
            dextents<int, 2>{columns, rows}, array<int, 2>{1, columns});
        total.store(output.slice<TILE, TILE>(first_column, first_row));
    """,
)


@mx.custom_function
def gather_pages(rows, indices, table):
    width = padded(rows.shape[1])
    source = mx.concatenate((rows, mx.zeros((1, rows.shape[1]))))
    addresses = mx.concatenate((indices.astype(mx.int32), mx.array([len(rows)], dtype=mx.int32)))
    return _GATHER(inputs=[source, addresses, table], template=[("PAGE_ROWS", PAGE_ROWS), ("TILE", TILE)],
                   grid=(width, len(table) * PAGE_ROWS, 1), threadgroup=(TILE, 1, 1),
                   output_shapes=[(len(table), PAGE_ROWS, width)], output_dtypes=[mx.float32], init_value=0)[0]


def scatter_pages(pages, indices, table, shape):
    @mx.custom_function
    def operation(values):
        extent = (shape[0] + 1, padded(shape[1]))
        addresses = mx.concatenate((indices.astype(mx.int32), mx.array([shape[0]], dtype=mx.int32)))
        result = _SCATTER(inputs=[values, addresses, table, mx.array(extent, dtype=mx.int32)],
                        template=[("PAGE_ROWS", PAGE_ROWS)],
                        grid=(padded(shape[1]), len(table) * PAGE_ROWS, 1), threadgroup=(TILE, 1, 1),
                        output_shapes=[extent], output_dtypes=[mx.float32], init_value=0)[0]
        return result[:shape[0], :shape[1]]

    @operation.vjp
    def backward(primals, cotangent, output):
        return (gather_pages(cotangent, indices, table),)

    return operation(pages)


@gather_pages.vjp
def _gather_backward(primals, cotangent, output):
    rows, indices, table = primals
    return scatter_pages(cotangent, indices, table, rows.shape), mx.zeros_like(indices), mx.zeros_like(table)


def _project(pages, weights, table, transpose=False):
    columns = weights.shape[1 if transpose else 2]
    return _PROJECT(inputs=[pages, weights, table],
                    template=[("PAGE_ROWS", PAGE_ROWS), ("TILE", TILE), ("TRANSPOSE", transpose)],
                    grid=(columns, len(table), 1), threadgroup=(32, 1, 1),
                    output_shapes=[(len(table), PAGE_ROWS, columns)], output_dtypes=[mx.float32], init_value=0)[0]


@mx.custom_function
def project_pages(pages, weights, table):
    return _project(pages, weights, table)


@project_pages.vjp
def _project_backward(primals, cotangent, output):
    pages, weights, table = primals
    return (_project(cotangent, weights, table, True), contract_pages(pages, cotangent, table, len(weights)),
            mx.zeros_like(table))


def contract_pages(lhs, rhs, table, groups=1):
    @mx.custom_function
    def operation(left, right):
        return _CONTRACT(inputs=[left, right, table], template=[("PAGE_ROWS", PAGE_ROWS), ("TILE", TILE)],
                         grid=(right.shape[2], left.shape[2] // TILE, groups), threadgroup=(32, 1, 1),
                         output_shapes=[(groups, left.shape[2], right.shape[2])], output_dtypes=[mx.float32])[0]

    @operation.vjp
    def backward(primals, cotangent, output):
        left, right = primals
        return _project(right, cotangent, table, True), _project(left, cotangent, table)

    return operation(lhs, rhs)


def pad_weights(weights):
    return mx.pad(weights, [(0, 0)] * (weights.ndim - 2) +
                  [(0, padded(weights.shape[-2]) - weights.shape[-2]),
                   (0, padded(weights.shape[-1]) - weights.shape[-1])])


def multiply(rows, weights):
    table = dense_table(len(rows))
    indices = mx.arange(len(rows), dtype=mx.int32)
    pages = gather_pages(rows, indices, table)
    output = project_pages(pages, pad_weights(weights)[None, :, :], table)
    return scatter_pages(output, indices, table, (len(rows), weights.shape[1]))


def cross(lhs, rhs):
    table = dense_table(len(lhs))
    indices = mx.arange(len(lhs), dtype=mx.int32)
    output = contract_pages(gather_pages(lhs, indices, table), gather_pages(rhs, indices, table), table)
    return output[0, :lhs.shape[1], :rhs.shape[1]]


def experts(rows, weights, selected):
    table = routed_table(selected, len(weights))
    indices = mx.argsort(selected)
    output = project_pages(gather_pages(rows, indices, table), pad_weights(weights), table)
    return scatter_pages(output, indices, table, (len(rows), weights.shape[2]))


@lru_cache(maxsize=128)
def batch_table(batches, rows):
    counts = mx.full((batches,), rows, dtype=mx.int32)
    return page_table(counts, batches * rows), mx.arange(batches * rows, dtype=mx.int32)


def batched_multiply(rows, weights):
    if isinstance(rows, mx.Tensor):
        return mx.matmul(rows, weights)
    batches, count, width = rows.shape
    table, indices = batch_table(batches, count)
    pages = gather_pages(rows.reshape(-1, width), indices, table)
    output = project_pages(pages, pad_weights(weights), table)
    return scatter_pages(output, indices, table, (batches * count, weights.shape[-1])).reshape(batches, count, -1)


def batched_cross(lhs, rhs):
    if isinstance(lhs, mx.Tensor):
        return mx.matmul(lhs, rhs, transpose_left=True)
    batches, count, width = lhs.shape
    table, indices = batch_table(batches, count)
    left = gather_pages(lhs.reshape(-1, width), indices, table)
    right = gather_pages(rhs.reshape(-1, rhs.shape[-1]), indices, table)
    return contract_pages(left, right, table, batches)[:, :width, :rhs.shape[-1]]
