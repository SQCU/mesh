// Prepared sum axes and fixed binding maps. The compiler binds an optional
// existing result as a numerical operand; no scheduling or layout choice runs.
#include <metal_stdlib>
using namespace metal;
kernel void mesh_sum(device const MESH_SUM_INPUT *values [[buffer(0)]],
                     device MESH_SUM_OUTPUT *output [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= MESH_SUM_COUNT) return;
    MESH_SUM_ACCUMULATOR value = 0;
    for (uint j = 0; j < MESH_SUM_SHARES; ++j)
        value += MESH_SUM_ACCUMULATOR(values[mesh_index<0>(ulong(j) * MESH_SUM_COUNT + i)]);
    if constexpr (MESH_SUM_ADD) value += MESH_SUM_ACCUMULATOR(output[mesh_index<1>(i)]);
    output[mesh_index<1>(i)] = MESH_SUM_OUTPUT(value);
}
