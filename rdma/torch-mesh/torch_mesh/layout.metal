// The compiler supplies element type, count and both address expressions.
// Every command covers a declared tile; the recorder sees its exact views.
#include <metal_stdlib>
using namespace metal;
kernel void mesh_layout_copy(device const MESH_COPY_INPUT_TYPE *source [[buffer(0)]],
                             device MESH_COPY_OUTPUT_TYPE *destination [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < MESH_COPY_COUNT) destination[MESH_COPY_DESTINATION] = MESH_COPY_OUTPUT_TYPE(source[MESH_COPY_SOURCE]);
}
