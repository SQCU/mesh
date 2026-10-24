// torch_mesh::linear (_stream.mm linear; backend.py): y[R, N] = x[R, K] w[N, K]^T (+ bias[N]), fp16 in and out, fp32
// accumulation, w's rows `ldw` apart (nn.Linear's layout, or rows padded to whole 4-element loads), for the few rows of a
// step: one dispatch, a threadgroup per COLS output columns, its NSG simdgroups each a K slice of them, a simdgroup's lanes
// across its slice four elements at a time, U loads of each row in flight, eight rows a pass; a simdgroup's sums reduced
// across its lanes (simd_sum), then the simdgroups' through threadgroup memory.  Each weight element is read once a pass.
// VW and VX are 4 where w's and x's rows lie on whole 4-element loads, else 1 (four loads of one).
#include <metal_stdlib>
using namespace metal;

struct linear_dims { uint R, K, N, B, LDW; };

template <uint V> inline float4 linear_load(device const half* p);
template <> inline float4 linear_load<4>(device const half* p) { return float4(*(device const half4*)p); }
template <> inline float4 linear_load<1>(device const half* p) { return float4(p[0], p[1], p[2], p[3]); }

template <uint VW, uint VX, uint COLS, uint NSG, uint U>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linear_rows(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* y [[buffer(2)]],
                        device const half* bias [[buffer(3)]], constant linear_dims& d [[buffer(4)]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG][8][COLS];
    const uint R = d.R, K = d.K, N = d.N, n0 = tg * COLS, K4 = K & ~3u;
    const uint span = (K4 / 4 + NSG - 1) / NSG * 4, k0 = sg * span, k1 = min(K4, k0 + span);
    device const half* wr[COLS];
    for (uint c = 0; c < COLS; c++) wr[c] = w + ulong(min(n0 + c, N - 1)) * d.LDW;
    for (uint r0 = 0; r0 < R; r0 += 8) {
        float acc[8][COLS];
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) acc[r][c] = 0.0f;
        device const half* xr[8];
        for (uint r = 0; r < 8; r++) xr[r] = x + ulong(min(r0 + r, R - 1)) * K;
        uint k = k0 + lane * 4;
        for (; k + 128 * (U - 1) < k1; k += 128 * U) {
            float4 wv[U][COLS];
            for (uint u = 0; u < U; u++)
                for (uint c = 0; c < COLS; c++) wv[u][c] = linear_load<VW>(wr[c] + k + 128 * u);
            for (uint u = 0; u < U; u++)
                for (uint r = 0; r < 8; r++) {
                    const float4 xv = linear_load<VX>(xr[r] + k + 128 * u);
                    for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[u][c]);
                }
        }
        for (; k < k1; k += 128) {
            float4 wv[COLS];
            for (uint c = 0; c < COLS; c++) wv[c] = linear_load<VW>(wr[c] + k);
            for (uint r = 0; r < 8; r++) {
                const float4 xv = linear_load<VX>(xr[r] + k);
                for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[c]);
            }
        }
        if (sg == 0 && K4 + lane < K)
            for (uint r = 0; r < 8; r++)
                for (uint c = 0; c < COLS; c++) acc[r][c] += float(xr[r][K4 + lane]) * float(wr[c][K4 + lane]);
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
            if (r0 + r < R && n0 + c < N) y[ulong(r0 + r) * N + n0 + c] = half(s + (d.B ? float(bias[n0 + c]) : 0.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
