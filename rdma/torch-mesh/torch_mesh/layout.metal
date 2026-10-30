// The compiler supplies element type, count and both address expressions.
// Every command covers a declared tile; the recorder sees its exact views.
#include <metal_stdlib>
using namespace metal;
kernel void mesh_layout_copy(device const MESH_COPY_INPUT_TYPE *source [[buffer(0)]],
                             device MESH_COPY_OUTPUT_TYPE *destination [[buffer(1)]],
#ifdef MESH_COPY_QUANTIZED
                             device const float *scales [[buffer(2)]],
                             device const long *zeros [[buffer(3)]],
#endif
                             uint i [[thread_position_in_grid]]) {
    if (i < MESH_COPY_COUNT) {
#ifdef MESH_COPY_FILL
        const auto code = MESH_COPY_FILL;
#elif defined(MESH_COPY_BITS)
        const auto code = mesh_matrix_packed_read<0,MESH_COPY_BITS>(source,i);
#else
        const auto code = source[MESH_COPY_SOURCE];
#endif
#ifdef MESH_COPY_QUANTIZED
        const auto value = mesh_matrix_scaled<MESH_COPY_OUTPUT_TYPE>(code,
            scales[mesh_index<2>(i)], zeros[mesh_index<3>(i)]);
#else
        const auto value = MESH_COPY_OUTPUT_TYPE(code);
#endif
        destination[MESH_COPY_DESTINATION] = value;
    }
}
