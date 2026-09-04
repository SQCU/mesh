from __future__ import annotations

import mlx.core as mx

TILE_EDGE = 16
SIMD_WIDTH = 32
INNER_TILE_EDGE = 16
FRAGMENT_EDGE = 8
MPP_OPERATION_ROWS = 16
MPP_TILE_ROWS = 32
MPP_TILE_COLUMNS = 32
MPP_WIDE_ROW_BLOCKS = 4
MPP_WIDE_TILE_ROWS = MPP_OPERATION_ROWS * MPP_WIDE_ROW_BLOCKS

_KERNEL = mx.fast.metal_kernel(
    name="mesh_matrix_product",
    input_names=["lhs", "rhs"],
    output_names=["product"],
    source="""
        uint column = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint local_column = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        threadgroup float lhs_tile[TILE][TILE];
        threadgroup float rhs_tile[TILE][TILE];
        float total = 0.0f;
        for (uint base = 0; base < INNER; base += TILE) {
            uint lhs_inner = base + local_column;
            uint rhs_inner = base + local_row;
            uint lhs_index = TRANSPOSE_LHS
                ? lhs_inner * ROWS + row
                : row * INNER + lhs_inner;
            uint rhs_index = TRANSPOSE_RHS
                ? column * INNER + rhs_inner
                : rhs_inner * COLUMNS + column;
            lhs_tile[local_row][local_column] = row < ROWS && lhs_inner < INNER
                ? float(lhs[lhs_index])
                : 0.0f;
            rhs_tile[local_row][local_column] = column < COLUMNS && rhs_inner < INNER
                ? float(rhs[rhs_index])
                : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint inner = 0; inner < TILE; ++inner)
                total += lhs_tile[local_row][inner] * rhs_tile[inner][local_column];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < ROWS && column < COLUMNS)
            product[row * COLUMNS + column] = total;
    """,
)

_MPP_KERNEL = mx.fast.metal_kernel(
    name="mesh_mpp_matrix_product",
    input_names=["lhs", "rhs"],
    output_names=["product"],
    header="""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    """,
    source="""
        uint lane = thread_index_in_simdgroup;
        uint tile_column = threadgroup_position_in_grid.x * TILE_COLUMNS;
        uint tile_row = threadgroup_position_in_grid.y * TILE_ROWS;
        uint quad = lane >> 2;
        uint fragment_row = ((quad & 4) | ((lane >> 1) & 3));
        uint fragment_column = ((quad & 2) | (lane & 1)) * 4;
        metal::vec<float, 8> accumulated_row0_column0(0.0f);
        metal::vec<float, 8> accumulated_row0_column1(0.0f);
        metal::vec<float, 8> accumulated_row1_column0(0.0f);
        metal::vec<float, 8> accumulated_row1_column1(0.0f);
        constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
            OPERATION_ROWS,
            TILE_COLUMNS,
            INNER_TILE,
            false,
            false,
            true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
        );
        mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> operation;
        auto cooperative_left = operation.template get_left_input_cooperative_tensor<float, float, float>();
        auto cooperative_right = operation.template get_right_input_cooperative_tensor<float, float, float>();
        auto cooperative_output = operation.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(cooperative_left)>,
            metal::remove_addrspace_t<decltype(cooperative_right)>,
            float
        >();
        #pragma clang loop unroll(disable)
        for (uint base = 0; base < INNER; base += INNER_TILE) {
            metal::vec<float, 8> left_values;
            metal::vec<float, 8> left_values_second;
            metal::vec<float, 8> right_first;
            metal::vec<float, 8> right_second;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
                    uint slot = i * 4 + j;
                    uint left_row = tile_row + fragment_row + i * 8;
                    uint left_row_second = left_row + OPERATION_ROWS;
                    uint left_inner = base + fragment_column + j;
                    uint left_index = TRANSPOSE_LHS
                        ? left_inner * ROWS + left_row
                        : left_row * INNER + left_inner;
                    if constexpr (ALIGNED_ROWS && ALIGNED_INNER)
                        left_values[slot] = float(lhs[left_index]);
                    else
                        left_values[slot] = left_row < ROWS && left_inner < INNER
                            ? float(lhs[left_index])
                            : 0.0f;
                    uint left_index_second = TRANSPOSE_LHS
                        ? left_inner * ROWS + left_row_second
                        : left_row_second * INNER + left_inner;
                    if constexpr (ALIGNED_ROWS && ALIGNED_INNER)
                        left_values_second[slot] = float(lhs[left_index_second]);
                    else
                        left_values_second[slot] = left_row_second < ROWS && left_inner < INNER
                            ? float(lhs[left_index_second])
                            : 0.0f;
                    uint right_inner = base + fragment_row + i * 8;
                    uint right_column_first = tile_column + fragment_column + j;
                    uint right_column_second = right_column_first + INNER_TILE;
                    uint right_index_first = TRANSPOSE_RHS
                        ? right_column_first * INNER + right_inner
                        : right_inner * COLUMNS + right_column_first;
                    uint right_index_second = TRANSPOSE_RHS
                        ? right_column_second * INNER + right_inner
                        : right_inner * COLUMNS + right_column_second;
                    if constexpr (ALIGNED_INNER && ALIGNED_COLUMNS) {
                        right_first[slot] = float(rhs[right_index_first]);
                        right_second[slot] = float(rhs[right_index_second]);
                    } else {
                        right_first[slot] = right_inner < INNER && right_column_first < COLUMNS
                            ? float(rhs[right_index_first])
                            : 0.0f;
                        right_second[slot] = right_inner < INNER && right_column_second < COLUMNS
                            ? float(rhs[right_index_second])
                            : 0.0f;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint slot = 0; slot < 8; ++slot) {
                cooperative_left[slot] = left_values[slot];
                cooperative_right[slot] = right_first[slot];
                cooperative_right[8 + slot] = right_second[slot];
                cooperative_output[slot] = accumulated_row0_column0[slot];
                cooperative_output[8 + slot] = accumulated_row0_column1[slot];
            }
            operation.run(cooperative_left, cooperative_right, cooperative_output);
            #pragma clang loop unroll(full)
            for (uint slot = 0; slot < 8; ++slot) {
                accumulated_row0_column0[slot] = cooperative_output[slot];
                accumulated_row0_column1[slot] = cooperative_output[8 + slot];
                cooperative_left[slot] = left_values_second[slot];
                cooperative_output[slot] = accumulated_row1_column0[slot];
                cooperative_output[8 + slot] = accumulated_row1_column1[slot];
            }
            operation.run(cooperative_left, cooperative_right, cooperative_output);
            #pragma clang loop unroll(full)
            for (uint slot = 0; slot < 8; ++slot) {
                accumulated_row1_column0[slot] = cooperative_output[slot];
                accumulated_row1_column1[slot] = cooperative_output[8 + slot];
            }
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; ++j) {
                uint slot = i * 4 + j;
                uint row = tile_row + fragment_row + i * 8;
                uint row_second = row + OPERATION_ROWS;
                uint column_first = tile_column + fragment_column + j;
                uint column_second = column_first + INNER_TILE;
                if constexpr (ALIGNED_ROWS && ALIGNED_COLUMNS) {
                    product[row * COLUMNS + column_first] = accumulated_row0_column0[slot];
                    product[row * COLUMNS + column_second] = accumulated_row0_column1[slot];
                    product[row_second * COLUMNS + column_first] = accumulated_row1_column0[slot];
                    product[row_second * COLUMNS + column_second] = accumulated_row1_column1[slot];
                } else {
                    if (row < ROWS && column_first < COLUMNS)
                        product[row * COLUMNS + column_first] = accumulated_row0_column0[slot];
                    if (row < ROWS && column_second < COLUMNS)
                        product[row * COLUMNS + column_second] = accumulated_row0_column1[slot];
                    if (row_second < ROWS && column_first < COLUMNS)
                        product[row_second * COLUMNS + column_first] = accumulated_row1_column0[slot];
                    if (row_second < ROWS && column_second < COLUMNS)
                        product[row_second * COLUMNS + column_second] = accumulated_row1_column1[slot];
                }
            }
        }
    """,
)

_WIDE_MPP_KERNEL = mx.fast.metal_kernel(
    name="mesh_wide_mpp_matrix_product",
    input_names=["lhs", "rhs"],
    output_names=["product"],
    header="""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    """,
    source="""
        uint lane = thread_index_in_simdgroup;
        uint tile_column = threadgroup_position_in_grid.x * TILE_COLUMNS;
        uint tile_row = threadgroup_position_in_grid.y * TILE_ROWS;
        uint quad = lane >> 2;
        uint fragment_row = ((quad & 4) | ((lane >> 1) & 3));
        uint fragment_column = ((quad & 2) | (lane & 1)) * 4;
        metal::vec<float, 8> accumulated_first[ROW_BLOCKS];
        metal::vec<float, 8> accumulated_second[ROW_BLOCKS];
        #pragma clang loop unroll(full)
        for (uint row_block = 0; row_block < ROW_BLOCKS; ++row_block) {
            accumulated_first[row_block] = metal::vec<float, 8>(0.0f);
            accumulated_second[row_block] = metal::vec<float, 8>(0.0f);
        }
        constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
            OPERATION_ROWS,
            TILE_COLUMNS,
            INNER_TILE,
            false,
            false,
            true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
        );
        mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> operation;
        auto cooperative_left = operation.template get_left_input_cooperative_tensor<float, float, float>();
        auto cooperative_right = operation.template get_right_input_cooperative_tensor<float, float, float>();
        auto cooperative_output = operation.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(cooperative_left)>,
            metal::remove_addrspace_t<decltype(cooperative_right)>,
            float
        >();
        #pragma clang loop unroll(disable)
        for (uint base = 0; base < INNER; base += INNER_TILE) {
            metal::vec<float, 8> right_first;
            metal::vec<float, 8> right_second;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
                    uint slot = i * 4 + j;
                    uint right_inner = base + fragment_row + i * 8;
                    uint right_column_first = tile_column + fragment_column + j;
                    uint right_column_second = right_column_first + INNER_TILE;
                    uint right_index_first = TRANSPOSE_RHS
                        ? right_column_first * INNER + right_inner
                        : right_inner * COLUMNS + right_column_first;
                    uint right_index_second = TRANSPOSE_RHS
                        ? right_column_second * INNER + right_inner
                        : right_inner * COLUMNS + right_column_second;
                    if constexpr (ALIGNED_INNER && ALIGNED_COLUMNS) {
                        right_first[slot] = float(rhs[right_index_first]);
                        right_second[slot] = float(rhs[right_index_second]);
                    } else {
                        right_first[slot] = right_inner < INNER && right_column_first < COLUMNS
                            ? float(rhs[right_index_first])
                            : 0.0f;
                        right_second[slot] = right_inner < INNER && right_column_second < COLUMNS
                            ? float(rhs[right_index_second])
                            : 0.0f;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint row_block = 0; row_block < ROW_BLOCKS; ++row_block) {
                metal::vec<float, 8> left_values;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2; ++i) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        uint slot = i * 4 + j;
                        uint left_row = tile_row + row_block * OPERATION_ROWS + fragment_row + i * 8;
                        uint left_inner = base + fragment_column + j;
                        uint left_index = TRANSPOSE_LHS
                            ? left_inner * ROWS + left_row
                            : left_row * INNER + left_inner;
                        if constexpr (ALIGNED_ROWS && ALIGNED_INNER)
                            left_values[slot] = float(lhs[left_index]);
                        else
                            left_values[slot] = left_row < ROWS && left_inner < INNER
                                ? float(lhs[left_index])
                                : 0.0f;
                    }
                }
                #pragma clang loop unroll(full)
                for (uint slot = 0; slot < 8; ++slot) {
                    cooperative_left[slot] = left_values[slot];
                    cooperative_right[slot] = right_first[slot];
                    cooperative_right[8 + slot] = right_second[slot];
                    cooperative_output[slot] = accumulated_first[row_block][slot];
                    cooperative_output[8 + slot] = accumulated_second[row_block][slot];
                }
                operation.run(cooperative_left, cooperative_right, cooperative_output);
                #pragma clang loop unroll(full)
                for (uint slot = 0; slot < 8; ++slot) {
                    accumulated_first[row_block][slot] = cooperative_output[slot];
                    accumulated_second[row_block][slot] = cooperative_output[8 + slot];
                }
            }
        }
        #pragma clang loop unroll(full)
        for (uint row_block = 0; row_block < ROW_BLOCKS; ++row_block) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
                    uint slot = i * 4 + j;
                    uint row = tile_row + row_block * OPERATION_ROWS + fragment_row + i * 8;
                    uint column_first = tile_column + fragment_column + j;
                    uint column_second = column_first + INNER_TILE;
                    if constexpr (ALIGNED_ROWS && ALIGNED_COLUMNS) {
                        product[row * COLUMNS + column_first] = accumulated_first[row_block][slot];
                        product[row * COLUMNS + column_second] = accumulated_second[row_block][slot];
                    } else {
                        if (row < ROWS && column_first < COLUMNS)
                            product[row * COLUMNS + column_first] = accumulated_first[row_block][slot];
                        if (row < ROWS && column_second < COLUMNS)
                            product[row * COLUMNS + column_second] = accumulated_second[row_block][slot];
                    }
                }
            }
        }
    """,
)

_EXPERT_KERNEL = mx.fast.metal_kernel(
    name="mesh_expert_matrix_product",
    input_names=["rows", "weights", "experts"],
    output_names=["product"],
    header="""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    """,
    source="""
        uint lane = thread_index_in_simdgroup;
        uint tile_column = threadgroup_position_in_grid.x * TILE_COLUMNS;
        uint tile_row = threadgroup_position_in_grid.y * TILE_ROWS;
        uint row_stop = min(uint(ROWS), tile_row + TILE_ROWS);
        uint first_expert = uint(experts[tile_row]);
        uint following_expert = uint(experts[row_stop - 1]);
        if (first_expert == following_expert) {
            uint quad = lane >> 2;
            uint fragment_row = ((quad & 4) | ((lane >> 1) & 3));
            uint fragment_column = ((quad & 2) | (lane & 1)) * 4;
            metal::vec<float, 8> accumulated_first(0.0f);
            metal::vec<float, 8> accumulated_second(0.0f);
            constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
                TILE_ROWS,
                TILE_COLUMNS,
                INNER_TILE,
                false,
                false,
                true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
            );
            mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> operation;
            auto cooperative_left = operation.template get_left_input_cooperative_tensor<float, float, float>();
            auto cooperative_right = operation.template get_right_input_cooperative_tensor<float, float, float>();
            auto cooperative_output = operation.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(cooperative_left)>,
                metal::remove_addrspace_t<decltype(cooperative_right)>,
                float
            >();
            for (uint base = 0; base < INNER; base += INNER_TILE) {
                metal::vec<float, 8> left_values;
                metal::vec<float, 8> right_first;
                metal::vec<float, 8> right_second;
                for (uint i = 0; i < 2; ++i) {
                    for (uint j = 0; j < 4; ++j) {
                        uint slot = i * 4 + j;
                        uint row = tile_row + fragment_row + i * 8;
                        uint left_inner = base + fragment_column + j;
                        left_values[slot] = row < ROWS && left_inner < INNER
                            ? float(rows[row * INNER + left_inner])
                            : 0.0f;
                        uint right_inner = base + fragment_row + i * 8;
                        uint column_first = tile_column + fragment_column + j;
                        uint column_second = column_first + INNER_TILE;
                        uint index_first = TRANSPOSE_WEIGHT
                            ? (first_expert * COLUMNS + column_first) * INNER + right_inner
                            : (first_expert * INNER + right_inner) * COLUMNS + column_first;
                        uint index_second = TRANSPOSE_WEIGHT
                            ? (first_expert * COLUMNS + column_second) * INNER + right_inner
                            : (first_expert * INNER + right_inner) * COLUMNS + column_second;
                        right_first[slot] = right_inner < INNER && column_first < COLUMNS
                            ? float(weights[index_first])
                            : 0.0f;
                        right_second[slot] = right_inner < INNER && column_second < COLUMNS
                            ? float(weights[index_second])
                            : 0.0f;
                    }
                }
                for (uint slot = 0; slot < 8; ++slot) {
                    cooperative_left[slot] = left_values[slot];
                    cooperative_right[slot] = right_first[slot];
                    cooperative_right[8 + slot] = right_second[slot];
                    cooperative_output[slot] = accumulated_first[slot];
                    cooperative_output[8 + slot] = accumulated_second[slot];
                }
                operation.run(cooperative_left, cooperative_right, cooperative_output);
                for (uint slot = 0; slot < 8; ++slot) {
                    accumulated_first[slot] = cooperative_output[slot];
                    accumulated_second[slot] = cooperative_output[8 + slot];
                }
            }
            for (uint i = 0; i < 2; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    uint slot = i * 4 + j;
                    uint row = tile_row + fragment_row + i * 8;
                    uint column_first = tile_column + fragment_column + j;
                    uint column_second = column_first + INNER_TILE;
                    if (row < ROWS && column_first < COLUMNS)
                        product[row * COLUMNS + column_first] = accumulated_first[slot];
                    if (row < ROWS && column_second < COLUMNS)
                        product[row * COLUMNS + column_second] = accumulated_second[slot];
                }
            }
        } else {
            uint column = tile_column + lane;
            if (column < COLUMNS) {
                for (uint row = tile_row; row < row_stop; ++row) {
                    uint expert = uint(experts[row]);
                    float total = 0.0f;
                    for (uint inner = 0; inner < INNER; ++inner) {
                        uint weight_index = TRANSPOSE_WEIGHT
                            ? (expert * COLUMNS + column) * INNER + inner
                            : (expert * INNER + inner) * COLUMNS + column;
                        total += float(rows[row * INNER + inner]) * float(weights[weight_index]);
                    }
                    product[row * COLUMNS + column] = total;
                }
            }
        }
    """,
)

_EXPERT_WEIGHT_GRADIENT_KERNEL = mx.fast.metal_kernel(
    name="mesh_expert_weight_gradient",
    input_names=["rows", "cotangent", "experts"],
    output_names=["gradient"],
    source="""
        uint column = thread_position_in_grid.x;
        uint inner = thread_position_in_grid.y;
        uint expert = thread_position_in_grid.z;
        uint local = thread_index_in_threadgroup;
        threadgroup uint first;
        threadgroup uint following;
        if (local == 0) {
            uint lower = 0;
            uint upper = ROWS;
            while (lower < upper) {
                uint middle = lower + (upper - lower) / 2;
                if (uint(experts[middle]) < expert)
                    lower = middle + 1;
                else
                    upper = middle;
            }
            first = lower;
            upper = ROWS;
            while (lower < upper) {
                uint middle = lower + (upper - lower) / 2;
                if (uint(experts[middle]) <= expert)
                    lower = middle + 1;
                else
                    upper = middle;
            }
            following = lower;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column < COLUMNS) {
            float total = 0.0f;
            for (uint row = first; row < following; ++row)
                total += float(rows[row * INNER + inner])
                    * float(cotangent[row * COLUMNS + column]);
            gradient[(expert * INNER + inner) * COLUMNS + column] = total;
        }
    """,
)

def _shape(lhs, rhs, transpose_lhs, transpose_rhs):
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"matrix operands must be rank 2; got {lhs.shape} and {rhs.shape}")
    rows = int(lhs.shape[1] if transpose_lhs else lhs.shape[0])
    lhs_inner = int(lhs.shape[0] if transpose_lhs else lhs.shape[1])
    rhs_inner = int(rhs.shape[1] if transpose_rhs else rhs.shape[0])
    columns = int(rhs.shape[0] if transpose_rhs else rhs.shape[1])
    if lhs_inner != rhs_inner:
        raise ValueError(f"matrix inner dimensions disagree: {lhs_inner} and {rhs_inner}")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"matrix dtypes disagree: {lhs.dtype} and {rhs.dtype}")
    if lhs.dtype != mx.float32:
        raise ValueError(f"matrix dtype must be float32; got {lhs.dtype}")
    return rows, lhs_inner, columns

def matrix_execution_schedule(rows, columns):
    if rows >= MPP_WIDE_TILE_ROWS and columns >= FRAGMENT_EDGE:
        return "mpp_wide"
    if rows >= FRAGMENT_EDGE and columns >= FRAGMENT_EDGE:
        return "mpp"
    return "threadgroup"

def _dispatch(lhs, rhs, transpose_lhs, transpose_rhs):
    rows, inner, columns = _shape(lhs, rhs, transpose_lhs, transpose_rhs)
    if rows == 0 or columns == 0 or inner == 0:
        return mx.zeros((rows, columns), dtype=lhs.dtype)
    schedule = matrix_execution_schedule(rows, columns)
    use_mpp = schedule != "threadgroup"
    use_wide_mpp = schedule == "mpp_wide"
    row_edge = MPP_WIDE_TILE_ROWS if use_wide_mpp else MPP_TILE_ROWS if use_mpp else TILE_EDGE
    column_edge = MPP_TILE_COLUMNS if use_mpp else TILE_EDGE
    threads = SIMD_WIDTH if use_mpp else TILE_EDGE
    grid = (
        ((columns + column_edge - 1) // column_edge) * threads,
        (rows + row_edge - 1) // row_edge,
        1,
    ) if use_mpp else (
        ((columns + column_edge - 1) // column_edge) * column_edge,
        ((rows + row_edge - 1) // row_edge) * row_edge,
        1,
    )
    kernel = _WIDE_MPP_KERNEL if use_wide_mpp else _MPP_KERNEL if use_mpp else _KERNEL
    template = [
        ("ROWS", rows),
        ("INNER", inner),
        ("COLUMNS", columns),
        ("TRANSPOSE_LHS", bool(transpose_lhs)),
        ("TRANSPOSE_RHS", bool(transpose_rhs)),
    ]
    if use_mpp:
        template.extend([
            ("TILE_ROWS", row_edge),
            ("TILE_COLUMNS", MPP_TILE_COLUMNS),
            ("OPERATION_ROWS", MPP_OPERATION_ROWS),
            ("INNER_TILE", INNER_TILE_EDGE),
            ("ALIGNED_ROWS", rows % row_edge == 0),
            ("ALIGNED_COLUMNS", columns % MPP_TILE_COLUMNS == 0),
            ("ALIGNED_INNER", inner % INNER_TILE_EDGE == 0),
        ])
        if use_wide_mpp:
            template.append(("ROW_BLOCKS", MPP_WIDE_ROW_BLOCKS))
    else:
        template.append(("TILE", TILE_EDGE))
    return kernel(
        inputs=[lhs, rhs],
        template=template,
        grid=grid,
        threadgroup=(threads, 1, 1) if use_mpp else (TILE_EDGE, TILE_EDGE, 1),
        output_shapes=[(rows, columns)],
        output_dtypes=[lhs.dtype],
    )[0]

@mx.custom_function
def matrix_multiply(lhs, rhs):
    return _dispatch(lhs, rhs, False, False)

@mx.custom_function
def matrix_multiply_transpose_left(lhs, rhs):
    return _dispatch(lhs, rhs, True, False)

@mx.custom_function
def matrix_multiply_transpose_right(lhs, rhs):
    return _dispatch(lhs, rhs, False, True)

@matrix_multiply.vjp
def _matrix_multiply_vjp(primals, cotangent, output):
    lhs, rhs = primals
    return (
        matrix_multiply_transpose_right(cotangent, rhs),
        matrix_multiply_transpose_left(lhs, cotangent),
    )

@matrix_multiply_transpose_left.vjp
def _matrix_multiply_transpose_left_vjp(primals, cotangent, output):
    lhs, rhs = primals
    return (
        matrix_multiply_transpose_right(rhs, cotangent),
        matrix_multiply(lhs, cotangent),
    )

@matrix_multiply_transpose_right.vjp
def _matrix_multiply_transpose_right_vjp(primals, cotangent, output):
    lhs, rhs = primals
    return (
        matrix_multiply(cotangent, rhs),
        matrix_multiply_transpose_left(cotangent, lhs),
    )

def linear(layer, rows):
    shape = rows.shape
    flat = rows.reshape(-1, shape[-1])
    product = matrix_multiply_transpose_right(flat, layer.weight)
    if "bias" in layer and layer.bias is not None:
        product = product + layer.bias
    return product.reshape(*shape[:-1], product.shape[-1])

def batched_matrix_vector(matrices, vectors):
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            f"matrix/vector inner dimensions disagree: {matrices.shape[-1]} and {vectors.shape[-1]}"
        )
    return mx.sum(matrices * vectors[..., None, :], axis=-1)

def _expert_dispatch(rows, weights, experts, transpose_weight):
    if rows.ndim != 2 or weights.ndim != 3 or experts.ndim != 1:
        raise ValueError(
            f"expert product ranks disagree: {rows.shape}, {weights.shape}, {experts.shape}"
        )
    if rows.shape[0] != experts.shape[0]:
        raise ValueError(f"expert row counts disagree: {rows.shape[0]} and {experts.shape[0]}")
    inner = int(rows.shape[1])
    weight_inner = int(weights.shape[2] if transpose_weight else weights.shape[1])
    columns = int(weights.shape[1] if transpose_weight else weights.shape[2])
    if inner != weight_inner:
        raise ValueError(f"expert inner dimensions disagree: {inner} and {weight_inner}")
    if rows.dtype != mx.float32 or weights.dtype != mx.float32:
        raise ValueError(f"expert operands must be float32; got {rows.dtype} and {weights.dtype}")
    row_mass = int(rows.shape[0])
    if row_mass == 0 or columns == 0 or inner == 0:
        return mx.zeros((row_mass, columns), dtype=rows.dtype)
    grid = (
        ((columns + MPP_TILE_COLUMNS - 1) // MPP_TILE_COLUMNS) * SIMD_WIDTH,
        (row_mass + MPP_OPERATION_ROWS - 1) // MPP_OPERATION_ROWS,
        1,
    )
    return _EXPERT_KERNEL(
        inputs=[rows, weights, experts],
        template=[
            ("ROWS", row_mass),
            ("INNER", inner),
            ("COLUMNS", columns),
            ("TILE_ROWS", MPP_OPERATION_ROWS),
            ("TILE_COLUMNS", MPP_TILE_COLUMNS),
            ("INNER_TILE", INNER_TILE_EDGE),
            ("TRANSPOSE_WEIGHT", bool(transpose_weight)),
        ],
        grid=grid,
        threadgroup=(SIMD_WIDTH, 1, 1),
        output_shapes=[(row_mass, columns)],
        output_dtypes=[rows.dtype],
    )[0]

def _expert_weight_gradient(rows, cotangent, experts, expert_mass):
    row_mass, inner = map(int, rows.shape)
    columns = int(cotangent.shape[1])
    if row_mass == 0 or inner == 0 or columns == 0:
        return mx.zeros((expert_mass, inner, columns), dtype=rows.dtype)
    thread_mass = 256
    return _EXPERT_WEIGHT_GRADIENT_KERNEL(
        inputs=[rows, cotangent, experts],
        template=[
            ("ROWS", row_mass),
            ("INNER", inner),
            ("COLUMNS", columns),
        ],
        grid=(
            ((columns + thread_mass - 1) // thread_mass) * thread_mass,
            inner,
            expert_mass,
        ),
        threadgroup=(thread_mass, 1, 1),
        output_shapes=[(expert_mass, inner, columns)],
        output_dtypes=[rows.dtype],
    )[0]

@mx.custom_function
def expert_matrix_multiply(rows, weights, experts):
    return _expert_dispatch(rows, weights, experts, False)

@expert_matrix_multiply.vjp
def _expert_matrix_multiply_vjp(primals, cotangent, output):
    rows, weights, experts = primals
    return (
        _expert_dispatch(cotangent, weights, experts, True),
        _expert_weight_gradient(rows, cotangent, experts, int(weights.shape[0])),
        mx.zeros_like(experts),
    )

__all__ = [
    "TILE_EDGE",
    "batched_matrix_vector",
    "expert_matrix_multiply",
    "linear",
    "matrix_execution_schedule",
    "matrix_multiply",
    "matrix_multiply_transpose_left",
    "matrix_multiply_transpose_right",
]
