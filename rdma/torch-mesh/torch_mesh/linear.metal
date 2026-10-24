// torch_mesh::linear (_stream.mm linear; backend.py): y[R, N] = act(x[R, K] w[N, K]^T (+ bias[N])), fp16 in and out, fp32
// accumulation, w's rows `ldw` apart (nn.Linear's layout, or rows padded to whole 4-element loads), for the few rows of a
// step: one dispatch, a threadgroup per COLS output columns, its NSG simdgroups each a K slice of them, a simdgroup's
// lanes across its slice four elements at a time, U loads of each row in flight, eight rows a pass; a simdgroup's sums
// reduced across its lanes (simd_sum), then the simdgroups' through threadgroup memory (linear_body), each output then
// handed to the kernel's store.  Each weight element is read once a pass.  x may be two segments, x0 [R, K0] and x1
// [R, K - K0], rows LD0 and LD1 apart (a cat of the two along K, never made: x0 = x1 and K0 = K where there is one),
// each segment's span of a simdgroup's K slice its own loop, no branch a load.
// VW and VX are 4 where w's and x's rows lie on whole 4-element loads, else 1 (four loads of one).  linear_rows stores into y, ACT its activation (0 none, 1
// tanh); linear_publish (compiled after the library's steps, fused.steps(), and argmax.metal) stores each output into a
// fused crossing's send segment through the collective's mesh_step_write and publishes it from its last threadgroup
// (mesh_step_enter, mesh_step_release at its repeat j: partition._fused, which runs the crossing).
#include <metal_stdlib>
using namespace metal;

struct linear_dims { uint R, K, N, B, LDW, K0, LD0, LD1; };

template <uint V> inline float4 linear_load(device const half* p);
template <> inline float4 linear_load<4>(device const half* p) { return float4(*(device const half4*)p); }
template <> inline float4 linear_load<1>(device const half* p) { return float4(p[0], p[1], p[2], p[3]); }

// elements k to k + 3 of a row, those at or past `hi` zero (a 1-element load's group may cross a segment's end; a
// 4-element load's never does: K0 and the slices are whole groups there)
template <uint V> inline float4 linear_part(device const half* p, uint k, uint hi) {
    if (V == 4) return linear_load<4>(p + k);
    float4 v;
    for (uint j = 0; j < 4; j++) v[j] = k + j < hi ? float(p[k + j]) : 0.0f;
    return v;
}

template <uint VW, uint VX, uint COLS, uint NSG, uint U, typename Store>
inline void linear_body(device const half* x, device const half* x1, device const half* w, const linear_dims d, thread const Store& store,
                        threadgroup float* part, uint sg, uint lane, uint tg, uint tid) {
    const uint R = d.R, K = d.K, N = d.N, n0 = tg * COLS, K4 = K & ~3u;
    const uint span = (K4 / 4 + NSG - 1) / NSG * 4, k0 = sg * span, k1 = min(K4, k0 + span);
    device const half* wr[COLS];
    for (uint c = 0; c < COLS; c++) wr[c] = w + ulong(min(n0 + c, N - 1)) * d.LDW;
    for (uint r0 = 0; r0 < R; r0 += 8) {
        float acc[8][COLS];
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) acc[r][c] = 0.0f;
        device const half* xr[8];
        device const half* xs[8];
        for (uint r = 0; r < 8; r++) {
            xr[r] = x + ulong(min(r0 + r, R - 1)) * d.LD0;
            xs[r] = x1 + ulong(min(r0 + r, R - 1)) * d.LD1;
        }
        for (uint seg = 0; seg < 2; seg++) {
            const uint lo = seg ? max(k0, d.K0) : k0, hi = seg ? k1 : min(k1, d.K0), off = seg ? d.K0 : 0;
            device const half* const* xx = seg ? xs : xr;
            uint k = lo + lane * 4;
            for (; k + 128 * (U - 1) < hi; k += 128 * U) {
                float4 wv[U][COLS];
                for (uint u = 0; u < U; u++)
                    for (uint c = 0; c < COLS; c++) wv[u][c] = linear_part<VW>(wr[c], k + 128 * u, hi);
                for (uint u = 0; u < U; u++)
                    for (uint r = 0; r < 8; r++) {
                        const float4 xv = linear_part<VX>(xx[r] - off, k + 128 * u, hi);
                        for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[u][c]);
                    }
            }
            for (; k < hi; k += 128) {
                float4 wv[COLS];
                for (uint c = 0; c < COLS; c++) wv[c] = linear_part<VW>(wr[c], k, hi);
                for (uint r = 0; r < 8; r++) {
                    const float4 xv = linear_part<VX>(xx[r] - off, k, hi);
                    for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[c]);
                }
            }
        }
        if (sg == 0 && K4 + lane < K)
            for (uint r = 0; r < 8; r++)
                for (uint c = 0; c < COLS; c++)
                    acc[r][c] += float(K4 + lane < d.K0 ? xr[r][K4 + lane] : xs[r][K4 + lane - d.K0]) * float(wr[c][K4 + lane]);
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) {
                const float s = simd_sum(acc[r][c]);
                if (lane == 0) part[(sg * 8 + r) * COLS + c] = s;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8 * COLS) {
            const uint r = tid / COLS, c = tid % COLS;
            float s = 0.0f;
            for (uint g = 0; g < NSG; g++) s += part[(g * 8 + r) * COLS + c];
            if (r0 + r < R && n0 + c < N) store(r0 + r, n0 + c, s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint ACT>
struct linear_store {
    device half* y;
    device const half* bias;
    uint N, B;
    void operator()(uint r, uint n, float s) const {
        const float v = s + (B ? float(bias[n]) : 0.0f);
        y[ulong(r) * N + n] = half(ACT == 1 ? tanh(v) : v);
    }
};

template <uint VW, uint VX, uint COLS, uint NSG, uint U, uint ACT>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linear_rows(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* y [[buffer(2)]],
                        device const half* bias [[buffer(3)]], constant linear_dims& d [[buffer(4)]], device const half* x1 [[buffer(5)]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG * 8 * COLS];
    const linear_store<ACT> store{y, bias, d.N, d.B};
    linear_body<VW, VX, COLS, NSG, U>(x, x1, w, d, store, part, sg, lane, tg, tid);
}

// Up to four products in one dispatch (torch_mesh::linears): problem p's threadgroups follow the ones before it
// (first[p]), its buffers 5p to 5p + 4 (x, w, y, bias, x1), its dims d[p] and activation act[p].
struct linear_group_words { linear_dims d[4]; uint first[4], act[4]; };

struct linear_act_store {
    device half* y;
    device const half* bias;
    uint N, B, ACT;
    void operator()(uint r, uint n, float s) const {
        const float v = s + (B ? float(bias[n]) : 0.0f);
        y[ulong(r) * N + n] = half(ACT == 1 ? tanh(v) : v);
    }
};

template <uint VW, uint VX, uint COLS, uint NSG, uint U>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linear_group(device const half* x0 [[buffer(0)]], device const half* w0 [[buffer(1)]], device half* y0 [[buffer(2)]],
                         device const half* b0 [[buffer(3)]], device const half* s0 [[buffer(4)]],
                         device const half* x1 [[buffer(5)]], device const half* w1 [[buffer(6)]], device half* y1 [[buffer(7)]],
                         device const half* b1 [[buffer(8)]], device const half* s1 [[buffer(9)]],
                         device const half* x2 [[buffer(10)]], device const half* w2 [[buffer(11)]], device half* y2 [[buffer(12)]],
                         device const half* b2 [[buffer(13)]], device const half* s2 [[buffer(14)]],
                         device const half* x3 [[buffer(15)]], device const half* w3 [[buffer(16)]], device half* y3 [[buffer(17)]],
                         device const half* b3 [[buffer(18)]], device const half* s3 [[buffer(19)]],
                         constant linear_group_words& g [[buffer(20)]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG * 8 * COLS];
    const uint p = tg >= g.first[3] ? 3 : tg >= g.first[2] ? 2 : tg >= g.first[1] ? 1 : 0;
    device const half* x = p == 0 ? x0 : p == 1 ? x1 : p == 2 ? x2 : x3;
    device const half* w = p == 0 ? w0 : p == 1 ? w1 : p == 2 ? w2 : w3;
    device half* y = p == 0 ? y0 : p == 1 ? y1 : p == 2 ? y2 : y3;
    device const half* b = p == 0 ? b0 : p == 1 ? b1 : p == 2 ? b2 : b3;
    device const half* s = p == 0 ? s0 : p == 1 ? s1 : p == 2 ? s2 : s3;
    const linear_act_store store{y, b, g.d[p].N, g.d[p].B, g.act[p]};
    linear_body<VW, VX, COLS, NSG, U>(x, s, w, g.d[p], store, part, sg, lane, tg - g.first[p], tid);
}

#ifdef MESH_METAL
template <uint PEERS>
struct linear_publish_words { uint R, K, N, B, LDW, K0, LD0, LD1, j; mesh_step_binding land[PEERS], publish[PEERS]; };

struct linear_step_store {
    device uchar* send;
    uint N;
    void operator()(uint r, uint n, float s) const { mesh_step_write<half>(send, ulong(r) * N + n, half(s)); }
};

template <uint VW, uint VX, uint COLS, uint NSG, uint U, uint PEERS>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linear_publish(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device uchar* send [[buffer(2)]],
                           device atomic_uint* arrivals [[buffer(3)]], device const half* x1 [[buffer(4)]],
                           constant linear_publish_words<PEERS>& p [[buffer(15)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint groups [[threadgroups_per_grid]]) {
    threadgroup float part[NSG * 8 * COLS];
    threadgroup uint last;
    for (uint q = 0; q < PEERS; q++) mesh_step_enter(*p.publish[q].p, p.publish[q].v, p.publish[q].progress, tg == 0 && tid == 0);
    const linear_dims d = {p.R, p.K, p.N, 0, p.LDW, p.K0, p.LD0, p.LD1};
    const linear_step_store store{send, p.N};
    linear_body<VW, VX, COLS, NSG, U>(x, x1, w, d, store, part, sg, lane, tg, tid);
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, groups, tid, &last)) {
        for (uint q = 0; q < PEERS; q++) mesh_step_release(*p.publish[q].p, p.publish[q].a, tid, p.j);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}
#endif
