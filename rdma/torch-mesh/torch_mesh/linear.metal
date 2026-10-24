// torch_mesh::linear (_stream.mm linear; backend.py): y[R, N] = x[R, K] w[N, K]^T (+ bias[N]), fp16 in and out, fp32
// accumulation, w row-major (nn.Linear's layout), for the few rows of a step: one dispatch, a threadgroup per COLS
// output columns, its NSG simdgroups each a K slice of them, a simdgroup's lanes across its slice in V-element loads (V
// the widest that divides K: 4, 2 or 1), eight rows a pass; a simdgroup's sums reduced across its lanes (simd_sum), then
// the simdgroups' through threadgroup memory.  Each weight element is read once a pass.
#include <metal_stdlib>
using namespace metal;

template <uint V> inline float4 linear_load(device const half* p);
template <> inline float4 linear_load<4>(device const half* p) { return float4(*(device const half4*)p); }
template <> inline float4 linear_load<2>(device const half* p) { const half2 v = *(device const half2*)p; return float4(v.x, v.y, 0, 0); }
template <> inline float4 linear_load<1>(device const half* p) { return float4(*p, 0, 0, 0); }

template <uint V, uint COLS, uint NSG>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linear_rows(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* y [[buffer(2)]],
                        device const half* bias [[buffer(3)]], constant uint4& d [[buffer(4)]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][8][COLS];
    const uint R = d.x, K = d.y, N = d.z, n0 = tg * COLS;
    const uint span = (K / V + NSG - 1) / NSG * V, k0 = sg * span, k1 = min(K, k0 + span);
    device const half* wr[COLS];
    for (uint c = 0; c < COLS; c++) wr[c] = w + ulong(min(n0 + c, N - 1)) * K;
    for (uint r0 = 0; r0 < R; r0 += 8) {
        float acc[8][COLS];
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) acc[r][c] = 0.0f;
        device const half* xr[8];
        for (uint r = 0; r < 8; r++) xr[r] = x + ulong(min(r0 + r, R - 1)) * K;
        for (uint k = k0 + lane * V; k < k1; k += 32 * V) {
            float4 wv[COLS];
            for (uint c = 0; c < COLS; c++) wv[c] = linear_load<V>(wr[c] + k);
            for (uint r = 0; r < 8; r++) {
                const float4 xv = linear_load<V>(xr[r] + k);
                for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[c]);
            }
        }
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) {
                const float s = simd_sum(acc[r][c]);
                if (lane == 0) part[sg][r][c] = s;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8 * COLS) {
            const uint r = tid / COLS, c = tid % COLS;
            float s = 0.0f;
            for (uint g = 0; g < NSG; g++) s += part[g][r][c];
            if (r0 + r < R && n0 + c < N) y[ulong(r0 + r) * N + n0 + c] = half(s + (d.w ? float(bias[n0 + c]) : 0.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
