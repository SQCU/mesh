// Sparse products use the same TensorOps geometry as dense products. Preparation
// supplies each output tile and its inner intervals; no predicate or bitmap is
// searched while executing. Short intervals describe actual values, not padding.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct SparseShape { uint rows, inner, columns; };

template<typename In, typename Weight, uint Bits, uint K, uint C, bool FullInner, bool FullColumns, typename Operation>
inline auto sparse_weight(thread const Operation &operation,
                          device const typename conditional<Bits == 0, Weight, uchar>::type *values,
                          device const float *scales, device const long *zeros,
                          ulong base, int2 extent, uint columns) {
    if constexpr (Bits) {
        return mesh_matrix_fragment_operand<false,In,Weight,FullInner,FullColumns>(operation,
            mesh_matrix_quantized_read<1,6,7,Bits,Weight>{values,scales,zeros,base,columns},extent,int2(0));
    } else {
        return mesh_matrix_operand<false,1,mesh_affine<1>(),K,C,mesh_matrix_stride<1,0>(),mesh_matrix_stride<1,1>(),
                                   In,Weight,FullInner,FullColumns>(operation,values,base,extent,int2(0),columns);
    }
}

template<typename In, typename Weight, typename Out, int R, int C, int K, int Groups, bool FullRows, bool FullColumns, bool FullInner,
         uint WeightBits = 0>
kernel void sparse_product(device const In* X [[buffer(0)]],
                           device const typename conditional<WeightBits == 0, Weight, uchar>::type* W [[buffer(1)]],
                           device Out* Y [[buffer(2)]], device const uint4* tiles [[buffer(3)]],
                           device const uint2* pieces [[buffer(4)]], constant SparseShape& shape [[buffer(5)]],
                           device const float* scales [[buffer(6)]], device const long* zeros [[buffer(7)]],
                           uint group [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]]) {
    // Two records per SIMD tile: origin/extent, followed by its reduction range.
    const uint slot = (group * Groups + sg) * 2;
    const uint4 tile = tiles[slot], reduction = tiles[slot + 1];
    constexpr long AM=mesh_matrix_stride<0,0>(), AK=mesh_matrix_stride<0,1>();
    constexpr auto desc = matmul2d_descriptor(R, C, K, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    using Left=decltype(mesh_matrix_operand<true,0,mesh_affine<0>(),R,K,AM,AK,In,Weight,FullRows,FullInner>(
        op,X,0,int2(K,R),int2(0),shape.inner));
    using Right=decltype(sparse_weight<In,Weight,WeightBits,K,C,FullInner,FullColumns>(
        op,W,scales,zeros,0,int2(C,K),shape.columns));
    auto acc = op.template get_destination_cooperative_tensor<Left,Right,float>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint p = reduction.x; p < reduction.y; ++p) {
        const uint2 piece = pieces[p];
        // Extents stop at both the sparse boundary and the tensor boundary.
        // TensorOps zero-fills its missing lanes without a padded allocation.
        auto a = mesh_matrix_operand<true,0,mesh_affine<0>(),R,K,AM,AK,In,Weight,FullRows,FullInner>(
            op,X,ulong(tile.x)*shape.inner+piece.x,int2(piece.y,tile.z),int2(0),shape.inner);
        auto b = sparse_weight<In,Weight,WeightBits,K,C,FullInner,FullColumns>(
            op,W,scales,zeros,ulong(piece.x)*shape.columns+tile.y,int2(tile.w,piece.y),shape.columns);
        op.run(a, b, acc);
    }
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
        const auto e = acc.get_multidimensional_index(i);
        if (e[1] < tile.z && e[0] < tile.w)
            Y[mesh_index<2>((ulong(tile.x) + e[1]) * shape.columns + tile.y + e[0])] = Out(acc[i]);
    }
}
