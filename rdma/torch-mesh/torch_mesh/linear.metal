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

// A threadgroup's output columns (linear_body's column map): column c of threadgroup tg, COLS adjacent ones by default.
struct linear_adjacent {
    template <uint COLS> uint n(uint tg, uint c) const { return tg * COLS + c; }
};

// A row block's sums handed to the store (linear_body's emission): by default each (row, column) sum in a thread of its
// own, store(row, column, sum); a store that takes a row's COLS sums at once overloads it (linear_cell_store, below).
template <uint COLS, uint NSG, typename Store, typename Cols>
inline void linear_emit(thread const Store& store, thread const Cols& cols, threadgroup float* part, uint r0, uint R, uint N, uint tg,
                        uint tid) {
    if (tid < 8 * COLS) {
        const uint r = tid / COLS, c = tid % COLS, n = cols.template n<COLS>(tg, c);
        float s = 0.0f;
        for (uint g = 0; g < NSG; g++) s += part[(g * 8 + r) * COLS + c];
        if (r0 + r < R && n < N) store(r0 + r, n, s);
    }
}

template <uint VW, uint VX, uint COLS, uint NSG, uint U, typename Store, typename Cols = linear_adjacent>
inline void linear_body(device const half* x, device const half* x1, device const half* w, const linear_dims d, thread const Store& store,
                        threadgroup float* part, uint sg, uint lane, uint tg, uint tid, thread const Cols& cols = Cols()) {
    const uint R = d.R, K = d.K, N = d.N, K4 = K & ~3u;
    const uint span = (K4 / 4 + NSG - 1) / NSG * 4, k0 = sg * span, k1 = min(K4, k0 + span);
    device const half* wr[COLS];
    for (uint c = 0; c < COLS; c++) wr[c] = w + ulong(min(cols.template n<COLS>(tg, c), N - 1)) * d.LDW;
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
        linear_emit<COLS, NSG>(store, cols, part, r0, R, N, tg, tid);
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

// problem P through its own inlined body, its buffers the kernel's own arguments: pointers selected by p cost the Mini
// 1.3 to 1.7 % of the two products' time at the LSTM's h16384 n3 shares and 16 to 18 % at h1024, the Studios 1 us or
// less (gbench3, 2026-10-24)
#define LINEAR_GROUP_CASE(P, X_, W_, Y_, B_, S_) case P: { const linear_act_store store{Y_, B_, g.d[P].N, g.d[P].B, g.act[P]}; \
    linear_body<VW, VX, COLS, NSG, U>(X_, S_, W_, g.d[P], store, part, sg, lane, tg - g.first[P], tid); break; }

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
    switch (p) {
        LINEAR_GROUP_CASE(0, x0, w0, y0, b0, s0)
        LINEAR_GROUP_CASE(1, x1, w1, y1, b1, s1)
        LINEAR_GROUP_CASE(2, x2, w2, y2, b2, s2)
        default: LINEAR_GROUP_CASE(3, x3, w3, y3, b3, s3)
    }
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

// The LSTM's tick as one dispatch (torch_mesh.compiler's fold of linears, torch's LSTMCell and allgatherv: _stream.mm
// cell_step): up to four layers' gates, each layer's threadgroup tg one unit of the layer's HR, its four columns that
// unit's gates (i, f, g and o: columns tg, HR + tg, 2 HR + tg and 3 HR + tg of the layer's [4 HR, K] weight,
// linear_gates), and each row's four sums handed together to linear_cell_store: the cell (torch's LSTMCell's arithmetic on
// the gates as linear stores them, fp16 with the bias added; c read and its new value written) and h written into this rank's send segment of the gather's crossing
// ([L, R, HR] at the layer) through the collective's mesh_step_write; the last threadgroup to arrive releases it at its
// repeat j (mesh_step_release), as linear_publish does.
struct linear_gates {
    uint HR;
    template <uint COLS> uint n(uint tg, uint c) const { return c * HR + tg; }
};

struct linear_cell_store {
    device const half* bias;
    device const float* c;
    device float* c1;
    device uchar* send;
    uint HR, R, layer, B;
};

inline float linear_sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }

template <uint COLS, uint NSG>
inline void linear_emit(thread const linear_cell_store& store, thread const linear_gates& cols, threadgroup float* part, uint r0, uint R,
                        uint N, uint tg, uint tid) {
    if (tid < 8 && r0 + tid < R && tg < store.HR) {
        const uint r = r0 + tid, j = tg, HR = store.HR;
        float s[4];
        for (uint c = 0; c < 4; c++) {
            s[c] = 0.0f;
            for (uint g = 0; g < NSG; g++) s[c] += part[(g * 8 + tid) * COLS + c];
            s[c] = float(half(s[c] + (store.B ? float(store.bias[c * HR + j]) : 0.0f)));  // the gate as linear stores it
        }
        const ulong at = ulong(r) * HR + j;
        const float cn = linear_sigmoid(s[1]) * store.c[at] + linear_sigmoid(s[0]) * tanh(s[2]);
        store.c1[at] = cn;
        mesh_step_write<half>(store.send, (ulong(store.layer) * store.R + r) * HR + j, half(linear_sigmoid(s[3]) * tanh(cn)));
    }
}

struct linears_cell_problem { linear_dims d; uint HR, layer, first, pad; };
template <uint PEERS>
struct linears_cell_words { linears_cell_problem p[4]; uint j, pad; mesh_step_binding land[PEERS], publish[PEERS]; };

// problem P through its own inlined body, its buffers the kernel's own arguments (as linear_group's)
#define LINEARS_CELL_CASE(P, X_, S_, W_, B_, C_, D_) case P: { const linears_cell_problem e = g.p[P]; \
    const linear_cell_store store{B_, C_, D_, send, e.HR, e.d.R, e.layer, e.d.B}; \
    linear_body<VW, VX, 4, NSG, U>(X_, S_, W_, e.d, store, part, sg, lane, tg - e.first, tid, linear_gates{e.HR}); break; }

template <uint VW, uint VX, uint NSG, uint U, uint PEERS>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linears_cell(device const half* x0 [[buffer(0)]], device const half* s0 [[buffer(1)]], device const half* w0 [[buffer(2)]],
                         device const half* b0 [[buffer(3)]], device const float* c0 [[buffer(4)]], device float* d0 [[buffer(5)]],
                         device const half* x1 [[buffer(6)]], device const half* s1 [[buffer(7)]], device const half* w1 [[buffer(8)]],
                         device const half* b1 [[buffer(9)]], device const float* c1 [[buffer(10)]], device float* d1 [[buffer(11)]],
                         device const half* x2 [[buffer(12)]], device const half* s2 [[buffer(13)]], device const half* w2 [[buffer(14)]],
                         device const half* b2 [[buffer(15)]], device const float* c2 [[buffer(16)]], device float* d2 [[buffer(17)]],
                         device const half* x3 [[buffer(18)]], device const half* s3 [[buffer(19)]], device const half* w3 [[buffer(20)]],
                         device const half* b3 [[buffer(21)]], device const float* c3 [[buffer(22)]], device float* d3 [[buffer(23)]],
                         device uchar* send [[buffer(24)]], device atomic_uint* arrivals [[buffer(25)]],
                         constant linears_cell_words<PEERS>& g [[buffer(26)]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint groups [[threadgroups_per_grid]]) {
    threadgroup float part[NSG * 8 * 4];
    threadgroup uint last;
    for (uint q = 0; q < PEERS; q++) mesh_step_enter(*g.publish[q].p, g.publish[q].v, g.publish[q].progress, tg == 0 && tid == 0);
    const uint p = tg >= g.p[3].first ? 3 : tg >= g.p[2].first ? 2 : tg >= g.p[1].first ? 1 : 0;
    switch (p) {
        LINEARS_CELL_CASE(0, x0, s0, w0, b0, c0, d0)
        LINEARS_CELL_CASE(1, x1, s1, w1, b1, c1, d1)
        LINEARS_CELL_CASE(2, x2, s2, w2, b2, c2, d2)
        default: LINEARS_CELL_CASE(3, x3, s3, w3, b3, c3, d3)
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, groups, tid, &last)) {
        for (uint q = 0; q < PEERS; q++) mesh_step_release(*g.publish[q].p, g.publish[q].a, tid, g.j);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}
#endif
