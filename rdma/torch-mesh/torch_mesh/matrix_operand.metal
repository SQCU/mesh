// Matrix operands use the page view's prepared index expression. A cooperative
// register fragment is a numerical operand, not another storage allocation.
#ifndef MESH_MATRIX_OPERAND
#define MESH_MATRIX_OPERAND

// These scalar expressions also evaluate declared immutable initializers during
// preparation. Their types and conversion boundaries are identical in both
// compilations; only the surrounding address/command machinery is Metal-only.
template <unsigned Bits, bool Signed = true>
inline int mesh_matrix_code(unsigned raw) {
    constexpr unsigned sign = Signed ? 1u << (Bits - 1) : 0u;
    return int(raw ^ sign) - int(sign);
}
template <unsigned Bits, bool Signed = true>
inline int mesh_matrix_bits(unsigned byte, unsigned position) {
    return mesh_matrix_code<Bits,Signed>((byte >> position) & ((1u << Bits) - 1u));
}
template<typename Scalar, typename Code>
inline Scalar mesh_matrix_scaled(Code code, float scale, long zero) {
    return Scalar((float(code) - float(zero)) * scale);
}

#ifndef MESH_SCALAR_PREPARATION
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

template<uint Binding> constexpr bool mesh_affine();
template<uint Binding, uint Axis> constexpr long mesh_stride();
template<uint Binding> inline long mesh_position(ulong i);
template<uint Binding, uint Axis> constexpr long mesh_matrix_stride() {
    if constexpr (mesh_affine<Binding>()) return mesh_stride<Binding,Axis>();
    else return 0;
}

template <bool Left, typename L, typename R, typename Operation>
inline auto mesh_matrix_fragment(thread const Operation &operation) {
    if constexpr (Left) return operation.template get_left_input_cooperative_tensor<L,R,float>();
    else return operation.template get_right_input_cooperative_tensor<L,R,float>();
}

template <bool FullRows = false, bool FullColumns = false, typename Fragment, typename Read>
inline void mesh_matrix_fill(thread Fragment &tile, thread const Read &read, int2 extent, int2 origin) {
    #pragma unroll
    for (uint i = 0; i < tile.get_capacity(); ++i) {
        if (tile.is_valid_element(i)) {
            auto at = tile.get_multidimensional_index(i);
            int2 position = origin + int2(at[0], at[1]);
            using Value = typename Fragment::element_type;
            tile[i] = (FullColumns || position.x < extent.x) && (FullRows || position.y < extent.y)
                ? Value(read(position)) : Value(0);
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

template <uint Binding, uint Bits>
inline int mesh_matrix_packed_read(device const uchar *values, ulong logical) {
    return mesh_matrix_bits<Bits>(uint(values[mesh_index<Binding>(logical)]), uint(mesh_position<Binding>(logical)));
}

// Byte positions, bit lanes and quantization positions are independent compiled
// relations over the same logical matrix. Stored sparse coordinates have
// already been composed into them; no encoding descriptor is read here.
template <uint Binding, uint ScaleBinding, uint ZeroBinding, uint Bits, typename Scalar>
struct mesh_matrix_quantized_read {
    device const uchar *values;
    device const float *scales;
    device const long *zeros;
    ulong base;
    uint columns;
    Scalar operator()(int2 at) const {
        const ulong logical = base + ulong(at.y) * columns + at.x;
        return mesh_matrix_scaled<Scalar>(mesh_matrix_packed_read<Binding,Bits>(values,logical),
            scales[mesh_index<ScaleBinding>(logical)], zeros[mesh_index<ZeroBinding>(logical)]);
    }
};

template <bool Left, typename L, typename R, bool FullRows = false, bool FullColumns = false,
          typename Operation, typename Read>
inline auto mesh_matrix_fragment_operand(thread const Operation &operation, thread const Read &read,
                                         int2 extent, int2 origin) {
    auto tile = mesh_matrix_fragment<Left,L,R>(operation);
    mesh_matrix_fill<FullRows,FullColumns>(tile, read, extent, origin);
    return tile;
}

template<size_t X, size_t Y>
inline auto mesh_matrix_extents(int x, int y) {
    if constexpr (X == dynamic_extent && Y == dynamic_extent) return extents<int, X, Y>{x, y};
    else if constexpr (X == dynamic_extent) return extents<int, X, Y>{x};
    else if constexpr (Y == dynamic_extent) return extents<int, X, Y>{y};
    else return extents<int, X, Y>{};
}

template <bool Left, uint Binding, bool Affine, uint Rows, uint Columns, long RowStride, long ColumnStride,
          typename L, typename R, bool FullRows = false, bool FullColumns = false, typename Scalar, typename Operation>
inline auto mesh_matrix_operand(thread const Operation &operation, device const Scalar *values,
                                ulong base, int2 extent, int2 origin, uint columns) {
    if constexpr (Affine) {
        auto view = tensor(const_cast<device Scalar *>(values) + mesh_index<Binding>(base),
                           mesh_matrix_extents<FullColumns ? Columns : dynamic_extent,
                                               FullRows ? Rows : dynamic_extent>(extent.x, extent.y),
                           array<int,2>{int(ColumnStride),int(RowStride)});
        return view.template slice<Columns,Rows>(origin.x,origin.y);
    } else {
        return mesh_matrix_fragment_operand<Left,L,R,FullRows,FullColumns>(
            operation,mesh_matrix_read<Binding,Scalar>{values,base,columns},extent,origin);
    }
}

template <bool Left, uint Binding, bool Affine, uint Rows, uint Columns, long RowStride, long ColumnStride,
          typename L, typename R, typename Scalar, typename Operation>
inline auto mesh_matrix_operand(thread const Operation &operation, device const Scalar *values,
                                ulong base, int2 extent, int2 origin) {
    return mesh_matrix_operand<Left,Binding,Affine,Rows,Columns,RowStride,ColumnStride,L,R>(
        operation,values,base,extent,origin,uint(extent.x));
}
#endif
#endif
