// Matrix operands use the page view's prepared index expression. A cooperative
// register fragment is a numerical operand, not another storage allocation.
#ifndef MESH_MATRIX_OPERAND
#define MESH_MATRIX_OPERAND
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

template<uint Binding> constexpr bool mesh_affine();
template<uint Binding, uint Axis> constexpr long mesh_stride();
template<uint Binding, uint Axis> constexpr long mesh_matrix_stride() {
    if constexpr (mesh_affine<Binding>()) return mesh_stride<Binding,Axis>();
    else return 0;
}

template <bool Left, typename L, typename R, typename Operation>
inline auto mesh_matrix_fragment(thread const Operation &operation) {
    if constexpr (Left) return operation.template get_left_input_cooperative_tensor<L,R,float>();
    else return operation.template get_right_input_cooperative_tensor<L,R,float>();
}

template <typename Fragment, typename Read>
inline void mesh_matrix_fill(thread Fragment &tile, thread const Read &read, int2 extent, int2 origin) {
    #pragma unroll
    for (uint i = 0; i < tile.get_capacity(); ++i) {
        if (tile.is_valid_element(i)) {
            auto at = tile.get_multidimensional_index(i);
            int2 position = origin + int2(at[0], at[1]);
            using Value = typename Fragment::element_type;
            tile[i] = position.x < extent.x && position.y < extent.y ? Value(read(position)) : Value(0);
        }
    }
}

template <uint Binding, typename Scalar>
struct mesh_matrix_read {
    device const Scalar *values;
    ulong base;
    uint columns;
    Scalar operator()(int2 at) const {
        return values[mesh_index<Binding>(base + ulong(at.y) * columns + at.x)];
    }
};

template <bool Left, uint Binding, bool Affine, uint Rows, uint Columns, long RowStride, long ColumnStride,
          typename L, typename R, typename Scalar, typename Operation>
inline auto mesh_matrix_operand(thread const Operation &operation, device const Scalar *values,
                                ulong base, int2 extent, int2 origin) {
    if constexpr (Affine) {
        auto view = tensor(const_cast<device Scalar *>(values) + mesh_index<Binding>(base),
                           dextents<int,2>{extent.x,extent.y}, array<int,2>{int(ColumnStride),int(RowStride)});
        return view.template slice<Columns,Rows>(origin.x,origin.y);
    } else {
        auto tile = mesh_matrix_fragment<Left,L,R>(operation);
        mesh_matrix_fill(tile, mesh_matrix_read<Binding,Scalar>{values,base,uint(extent.x)}, extent, origin);
        return tile;
    }
}
#endif
