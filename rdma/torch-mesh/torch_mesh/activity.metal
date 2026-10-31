// A scalar activity value masks fixed launch dimensions. Both inputs are
// ordinary page operands; submission neither reads them nor chooses a path.
#include <metal_stdlib>
using namespace metal;
kernel void mesh_activity(device const uint *activity [[buffer(0)]],
                          device const uint *enabled [[buffer(1)]], device uint *joint [[buffer(2)]]) {
    joint[0] = uint(activity[0] != 0u && mesh_payload(enabled) != 0u);
}
kernel void mesh_dispatch_mask(device const uint *joint [[buffer(0)]],
                               device const uint *dimensions [[buffer(1)]],
                               device uint *dispatches [[buffer(2)]],
                               constant uint &words [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i < words) dispatches[i] = dimensions[i] * joint[0];
}
