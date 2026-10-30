// One local factor of a Walsh-Hadamard transform. Prepared views carry the
// stage's strided coordinates, page placement and exact dependency ranges.
// A later factor may read pages written locally or delivered by a peer.
//
// Element e = (group % signGroups) * Width + i of a row is the sum of its sources: the row's read values
// [mesh_index<4>(e), mesh_index<4>(e + 1)), each element its own extent. Binding 4 is that relation, compiled from
// the factor's domain (structured.py prepare_stage): the identity where each element reads one value in place, a
// ragged relation where a selected input feeds the transform; a row holds mesh_index<4>(signGroups * Width) values.
#include <metal_stdlib>
using namespace metal;
template <uint Width, ulong Normalize = 1, bool Signed = true, typename Input = float, typename Sign = float>
kernel void structured_butterfly(device const Input *x [[buffer(0)]], device float *y [[buffer(1)]],
                                 device const Sign *sign [[buffer(2)]], constant uint &signGroups [[buffer(3)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint t [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    threadgroup float v[Width];
    const ulong element = ulong(group % signGroups) * Width;
    const long row = long(group / signGroups) * mesh_index<4>(ulong(signGroups) * Width);
    for (uint i = t; i < Width; i += threads) {
        float value = 0.0f;
        for (long k = mesh_index<4>(element + i), end = mesh_index<4>(element + i + 1); k < end; ++k)
            value += float(x[mesh_index<0>(row + k)]);
        if constexpr (Signed) value *= float(sign[mesh_index<2>(element + i)]);
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
