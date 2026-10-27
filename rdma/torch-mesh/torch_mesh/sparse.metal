// Sparse products use the same TensorOps geometry as dense products. Preparation
// supplies each output tile and its inner intervals; no predicate or bitmap is
// searched while executing. Short intervals describe actual values, not padding.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct SparseShape { uint rows, inner, columns; };

template<size_t X, size_t Y>
inline auto sparse_extents(int x, int y) {
    if constexpr (X == dynamic_extent && Y == dynamic_extent) return extents<int, X, Y>{x, y};
    else if constexpr (X == dynamic_extent) return extents<int, X, Y>{x};
    else if constexpr (Y == dynamic_extent) return extents<int, X, Y>{y};
    else return extents<int, X, Y>{};
}

template<typename In, typename Out, int R, int C, int K, int Groups, bool FullRows, bool FullColumns, bool FullInner,
         int AM, int AK, int BK, int BN>
kernel void sparse_product(device In* X [[buffer(0)]], device In* W [[buffer(1)]],
                           device Out* Y [[buffer(2)]], device const uint4* tiles [[buffer(3)]],
                           device const uint2* pieces [[buffer(4)]], constant SparseShape& shape [[buffer(5)]],
                           uint group [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]]) {
    // Two records per SIMD tile: origin/extent, followed by its reduction range.
    const uint slot = (group * Groups + sg) * 2;
    const uint4 tile = tiles[slot], reduction = tiles[slot + 1];
    auto A = tensor(X, dextents<int, 2>{int(shape.inner), int(shape.rows)}, array<int, 2>{AK, AM});
    auto B = tensor(W, dextents<int, 2>{int(shape.columns), int(shape.inner)}, array<int, 2>{BN, BK});
    constexpr auto desc = matmul2d_descriptor(R, C, K, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint p = reduction.x; p < reduction.y; ++p) {
        const uint2 piece = pieces[p];
        // Extents stop at both the sparse boundary and the tensor boundary.
        // TensorOps zero-fills its missing lanes without a padded allocation.
        auto a = tensor(X + tile.x * AM + piece.x * AK,
                        sparse_extents<FullInner ? K : dynamic_extent, FullRows ? R : dynamic_extent>(int(piece.y), int(tile.z)),
                        array<int, 2>{AK, AM});
        auto b = tensor(W + piece.x * BK + tile.y * BN,
                        sparse_extents<FullColumns ? C : dynamic_extent, FullInner ? K : dynamic_extent>(int(tile.w), int(piece.y)),
                        array<int, 2>{BN, BK});
        op.run(a, b, acc);
    }
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
        const auto e = acc.get_multidimensional_index(i);
        if (e[1] < tile.z && e[0] < tile.w)
            Y[mesh_index<2>((tile.x + e[1]) * shape.columns + tile.y + e[0])] = Out(acc[i]);
    }
}
