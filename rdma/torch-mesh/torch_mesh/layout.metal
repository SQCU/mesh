// The compiler supplies element type, count and both address expressions.
// Every command covers a declared tile; the recorder sees its exact views.
#ifndef MESH_SCALAR_PREPARATION
#include <metal_stdlib>
using namespace metal;
#endif
#ifndef MESH_LAYOUT_SCALAR
#define MESH_LAYOUT_SCALAR
template<typename Destination, typename Source>
inline Destination mesh_layout_cast(Source value) {
    return Destination(value);
}
template<typename Destination, typename Source>
inline Destination mesh_layout_bitcast(Source value) {
#ifdef MESH_SCALAR_PREPARATION
    static_assert(sizeof(Destination) == sizeof(Source));
    Destination result;
    __builtin_memcpy(&result, &value, sizeof(result));
    return result;
#else
    return as_type<Destination>(value);
#endif
}
#endif

#ifndef MESH_SCALAR_PREPARATION
#if defined(MESH_COPY_INDEX_TYPE) && !defined(MESH_COPY_INDEX_VALUE)
#define MESH_COPY_INDEX_VALUE indices[mesh_index<MESH_COPY_INDEX_BINDING>(i)]
#endif
kernel void mesh_layout_copy(device const MESH_COPY_INPUT_TYPE *source [[buffer(0)]],
                             device MESH_COPY_OUTPUT_TYPE *destination [[buffer(1)]],
#ifdef MESH_COPY_QUANTIZED
                             device const float *scales [[buffer(2)]],
                             device const long *zeros [[buffer(3)]],
#endif
#ifdef MESH_COPY_INDEX_TYPE
                             device const MESH_COPY_INDEX_TYPE *indices [[buffer(MESH_COPY_INDEX_BINDING)]],
#endif
#ifdef MESH_COPY_PREDICATE_TYPE
                             device const MESH_COPY_PREDICATE_TYPE *predicate [[buffer(MESH_COPY_PREDICATE_BINDING)]],
#endif
                             uint i [[thread_position_in_grid]]) {
    if (i < MESH_COPY_COUNT) {
        const ulong output_i = i;
        const auto destination_index = MESH_COPY_DESTINATION;
        auto read = [&](ulong i) -> MESH_COPY_OUTPUT_TYPE {
#ifdef MESH_COPY_PREDICATE_TYPE
            if (!predicate[mesh_index<MESH_COPY_PREDICATE_BINDING>(i)]) return MESH_COPY_OUTPUT_TYPE(0);
#endif
#ifdef MESH_COPY_INDEX_VALUE
            const ulong source_index = ulong(MESH_COPY_INDEX_VALUE);
#else
            const ulong source_index = i;
#endif
            // Source and output positions are distinct declared domains.
            {
                const ulong i = source_index;
#ifdef MESH_COPY_PRESENT
                if (!(MESH_COPY_PRESENT)) return MESH_COPY_OUTPUT_TYPE(0);
#endif
#ifdef MESH_COPY_FILL
                const auto code = MESH_COPY_FILL;
#elif defined(MESH_COPY_BITS)
                const auto code = mesh_matrix_packed_read<0,MESH_COPY_BITS>(source,i);
#else
                const auto code = source[MESH_COPY_SOURCE];
#endif
#ifdef MESH_COPY_VALUE
                return MESH_COPY_VALUE;
#elif defined(MESH_COPY_QUANTIZED)
                return mesh_matrix_scaled<MESH_COPY_OUTPUT_TYPE>(code,
                    scales[mesh_index<2>(i)], zeros[mesh_index<3>(i)]);
#elif defined(MESH_COPY_BITCAST)
                return mesh_layout_bitcast<MESH_COPY_OUTPUT_TYPE>(code);
#else
                return mesh_layout_cast<MESH_COPY_OUTPUT_TYPE>(code);
#endif
            }
        };
#ifdef MESH_COPY_OUTPUT_BITS
        // A destination byte has one writer even when its codes come from
        // different selected source bytes. The final byte masks its tail.
        constexpr uint lanes = MESH_COPY_OUTPUT_WIDTH / MESH_COPY_OUTPUT_BITS;
        constexpr uint mask = (1u << MESH_COPY_OUTPUT_BITS) - 1u;
        uint value = 0;
        #pragma unroll
        for (uint lane = 0; lane < lanes; ++lane) {
            const ulong logical = output_i * lanes + lane;
            if (logical < MESH_COPY_LOGICAL_COUNT)
                value |= (uint(read(logical)) & mask) << (lane * MESH_COPY_OUTPUT_BITS);
        }
        destination[destination_index] = MESH_COPY_OUTPUT_TYPE(value);
#else
        destination[destination_index] = read(i);
#endif
    }
}
#endif
