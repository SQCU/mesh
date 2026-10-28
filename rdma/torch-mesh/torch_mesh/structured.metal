// One local factor of a Walsh-Hadamard transform. Prepared views carry the
// stage's strided coordinates, page placement and exact dependency ranges.
// A later factor may read pages written locally or delivered by a peer.
#include <metal_stdlib>
using namespace metal;
template<uint B> inline long mesh_position(ulong i);
template <uint Width, ulong Normalize = 1, bool Signed = true, typename Input = float, typename Sign = float, bool Masked = false>
kernel void structured_butterfly(device const Input *x [[buffer(0)]], device float *y [[buffer(1)]],
                                 device const Sign *sign [[buffer(2)]], constant uint &signGroups [[buffer(3)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint t [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    threadgroup float v[Width];
    for (uint i = t; i < Width; i += threads) {
        float value;
        if constexpr (Masked) value = mesh_position<4>(ulong(group % signGroups) * Width + i)
            ? float(x[mesh_index<0>(ulong(group) * Width + i)]) : 0.0f;
        else value = float(x[mesh_index<0>(ulong(group) * Width + i)]);
        if constexpr (Signed) value *= float(sign[mesh_index<2>(ulong(group % signGroups) * Width + i)]);
        v[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < Width; h <<= 1) {
        for (uint i = t; i < Width / 2; i += threads) {
            const uint j = (i / h) * 2 * h + i % h;
            const float a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = t; i < Width; i += threads)
        y[mesh_index<1>(ulong(group) * Width + i)] = v[i] * rsqrt(float(Normalize));
}
