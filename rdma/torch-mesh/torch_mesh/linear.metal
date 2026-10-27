// torch_mesh::linear (_stream.mm linear; backend.py): y[R, N] = act(x[R, K] w[N, K]^T (+ bias[N])), fp16 in and out, fp32
// accumulation, w's rows `ldw` apart (nn.Linear's layout, or rows padded to whole 4-element loads), for the few rows of a
// step: one dispatch, a threadgroup per row tile and COLS output columns, its NSG simdgroups each a K slice of them, a simdgroup's
// lanes across its slice four elements at a time, U loads of each row in flight, one configured row tile per threadgroup; a simdgroup's sums
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

struct linear_dims { uint R, K, N, B, LDW, K0, LD0, LD1, LDY, SDY; };

template <uint V> inline vec<float, MESH_LINEAR_VECTOR> linear_load(device const half* p);
template <> inline vec<float, MESH_LINEAR_VECTOR> linear_load<MESH_LINEAR_VECTOR>(device const half* p) { return vec<float, MESH_LINEAR_VECTOR>(*(device const vec<half, MESH_LINEAR_VECTOR>*)p); }
template <> inline vec<float, MESH_LINEAR_VECTOR> linear_load<1>(device const half* p) { vec<float, MESH_LINEAR_VECTOR> v;
    for (uint j = 0; j < MESH_LINEAR_VECTOR; j++) v[j] = float(p[j]);
    return v; }

template <uint VW, uint VX, uint COLS, uint NSG, uint U, typename Store>
inline void linear_body(device const half* x, device const half* x1, device const half* w, const linear_dims d, thread const Store& store,
                        threadgroup float* part, uint sg, uint lane, uint tg, uint tid) {
    const uint R = d.R, K = d.K, N = d.N;
    const uint columns = (N + COLS - 1) / COLS;
    const uint n0 = (tg % columns) * COLS, r0 = (tg / columns) * MESH_LINEAR_ROWS;
    const uint span = (K + NSG * MESH_LINEAR_VECTOR - 1) / (NSG * MESH_LINEAR_VECTOR) * MESH_LINEAR_VECTOR;
    const uint k0 = sg * span, k1 = min(K, k0 + span);
    device const half* wr[COLS];
    for (uint c = 0; c < COLS; c++) wr[c] = w + ulong(min(n0 + c, N - 1)) * d.LDW;
    {
        float acc[MESH_LINEAR_ROWS][COLS];
        for (uint r = 0; r < MESH_LINEAR_ROWS; r++)
            for (uint c = 0; c < COLS; c++) acc[r][c] = 0.0f;
        device const half* xr[MESH_LINEAR_ROWS];
        device const half* xs[MESH_LINEAR_ROWS];
        for (uint r = 0; r < MESH_LINEAR_ROWS; r++) {
            xr[r] = x + ulong(min(r0 + r, R - 1)) * d.LD0;
            xs[r] = x1 + ulong(min(r0 + r, R - 1)) * d.LD1;
        }
        for (uint seg = 0; seg < 2; seg++) {
            const uint lo = seg ? max(k0, d.K0) : k0, hi = seg ? k1 : min(k1, d.K0), off = seg ? d.K0 : 0;
            device const half* const* xx = seg ? xs : xr;
            const uint full = lo + (hi > lo ? (hi - lo) / MESH_LINEAR_VECTOR * MESH_LINEAR_VECTOR : 0);
            uint k = lo + lane * MESH_LINEAR_VECTOR;
            for (; k + (MESH_SIMD_WIDTH * MESH_LINEAR_VECTOR) * (U - 1) < full; k += (MESH_SIMD_WIDTH * MESH_LINEAR_VECTOR) * U) {
                vec<float, MESH_LINEAR_VECTOR> wv[U][COLS];
                for (uint u = 0; u < U; u++)
                    for (uint c = 0; c < COLS; c++) wv[u][c] = linear_load<VW>(wr[c] + k + (MESH_SIMD_WIDTH * MESH_LINEAR_VECTOR) * u);
                for (uint u = 0; u < U; u++)
                    for (uint r = 0; r < MESH_LINEAR_ROWS; r++) {
                        const vec<float, MESH_LINEAR_VECTOR> xv = linear_load<VX>(xx[r] + k - off + (MESH_SIMD_WIDTH * MESH_LINEAR_VECTOR) * u);
                        for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[u][c]);
                    }
            }
            for (; k < full; k += (MESH_SIMD_WIDTH * MESH_LINEAR_VECTOR)) {
                vec<float, MESH_LINEAR_VECTOR> wv[COLS];
                for (uint c = 0; c < COLS; c++) wv[c] = linear_load<VW>(wr[c] + k);
                for (uint r = 0; r < MESH_LINEAR_ROWS; r++) {
                    const vec<float, MESH_LINEAR_VECTOR> xv = linear_load<VX>(xx[r] + k - off);
                    for (uint c = 0; c < COLS; c++) acc[r][c] += dot(xv, wv[c]);
                }
            }
            if (full + lane < hi)
                for (uint r = 0; r < MESH_LINEAR_ROWS; r++)
                    for (uint c = 0; c < COLS; c++)
                        acc[r][c] += float(xx[r][full + lane - off]) * float(wr[c][full + lane]);
        }
        for (uint r = 0; r < MESH_LINEAR_ROWS; r++)
            for (uint c = 0; c < COLS; c++) {
                const float s = simd_sum(acc[r][c]);
                if (lane == 0) part[(sg * MESH_LINEAR_ROWS + r) * COLS + c] = s;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint output = tid; output < MESH_LINEAR_ROWS * COLS; output += NSG * MESH_SIMD_WIDTH) {
            const uint r = output / COLS, c = output % COLS;
            float s = 0.0f;
            for (uint g = 0; g < NSG; g++) s += part[(g * MESH_LINEAR_ROWS + r) * COLS + c];
            if (r0 + r < R && n0 + c < N) store(r0 + r, n0 + c, s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint ACT>
struct linear_store {
    device half* y;
    device const half* bias;
    uint LDY, SDY, B;
    void operator()(uint r, uint n, float s) const {
        const float v = s + (B ? float(bias[n]) : 0.0f);
        y[ulong(r) * LDY + ulong(n) * SDY] = half(ACT == 1 ? tanh(v) : v);
    }
};

template <uint VW, uint VX, uint COLS, uint NSG, uint U, uint ACT>
[[max_total_threads_per_threadgroup(NSG * MESH_SIMD_WIDTH)]]
kernel void linear_rows(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* y [[buffer(2)]],
                        device const half* bias [[buffer(3)]], constant linear_dims& d [[buffer(4)]], device const half* x1 [[buffer(5)]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NSG * MESH_LINEAR_ROWS * COLS];
    const linear_store<ACT> store{y, bias, d.LDY, d.SDY, d.B};
    linear_body<VW, VX, COLS, NSG, U>(x, x1, w, d, store, part, sg, lane, tg, tid);
}

struct linear_act_store {
    device half* y;
    device const half* bias;
    uint LDY, SDY, B, ACT;
    void operator()(uint r, uint n, float s) const {
        const float v = s + (B ? float(bias[n]) : 0.0f);
        y[ulong(r) * LDY + ulong(n) * SDY] = half(ACT == 1 ? tanh(v) : v);
    }
};

// linear_group is specialized by the compiler from the manifest binding capacity.

#ifdef MESH_METAL
template <uint CHANNELS>
struct linear_publish_words { uint R, K, N, B, LDW, K0, LD0, LD1, j; mesh_step_binding land[CHANNELS], publish[CHANNELS]; };

struct linear_step_store {
    device uchar* send;
    uint N;
    void operator()(uint r, uint n, float s) const { mesh_step_write<half>(send, ulong(r) * N + n, half(s)); }
};

template <uint VW, uint VX, uint COLS, uint NSG, uint U, uint CHANNELS>
[[max_total_threads_per_threadgroup(NSG * MESH_SIMD_WIDTH)]]
kernel void linear_publish(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device uchar* send [[buffer(2)]],
                           device atomic_uint* arrivals [[buffer(3)]], device const half* x1 [[buffer(4)]],
                           constant linear_publish_words<CHANNELS>& p [[buffer(15)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint groups [[threadgroups_per_grid]]) {
    threadgroup float part[NSG * MESH_LINEAR_ROWS * COLS];
    threadgroup uint last;
    for (uint q = 0; q < CHANNELS; q++) if (p.publish[q].p)
        mesh_step_enter(*p.publish[q].p, p.publish[q].v, p.publish[q].progress, tg == 0 && tid == 0);
    const linear_dims d = {p.R, p.K, p.N, 0, p.LDW, p.K0, p.LD0, p.LD1, p.N, 1};
    const linear_step_store store{send, p.N};
    linear_body<VW, VX, COLS, NSG, U>(x, x1, w, d, store, part, sg, lane, tg, tid);
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, groups, tid, &last)) {
        for (uint q = 0; q < CHANNELS; q++) if (p.publish[q].p)
            mesh_step_release(*p.publish[q].p, p.publish[q].a, tid, p.j);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}
#endif
