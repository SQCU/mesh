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

// x's source, the rows a pass of linear_body reads (block: rows r0 to r0 + 7, the last repeated past R): load<SEG, VX>
// gives elements k to k + 3 of row r's segment SEG (0: k < K0, 1: K0 up), those at or past hi zero, and one element k.
// linear_dense: x [R, K0] and x1 [R, K - K0], rows LD0 and LD1 apart (x = x1 and K0 = K where there is one).
struct linear_dense_rows {
    device const half* x[8];
    device const half* x1[8];
    uint K0;
    template <uint SEG, uint VX> float4 load(uint r, uint k, uint hi) const { return linear_part<VX>(SEG ? x1[r] - K0 : x[r], k, hi); }
    float one(uint r, uint k) const { return float(k < K0 ? x[r][k] : x1[r][k - K0]); }
};
struct linear_dense {
    device const half* x;
    device const half* x1;
    uint LD0, LD1, K0;
    linear_dense_rows block(uint r0, uint R) const {
        linear_dense_rows b;
        b.K0 = K0;
        for (uint r = 0; r < 8; r++) {
            b.x[r] = x + ulong(min(r0 + r, R - 1)) * LD0;
            b.x1[r] = x1 + ulong(min(r0 + r, R - 1)) * LD1;
        }
        return b;
    }
};

template <uint COLS> struct linear_cols { device const half* p[COLS]; };
template <uint COLS> struct linear_acc { float v[8][COLS]; };

// a simdgroup's span [lo, hi) of segment SEG: U loads of each row in flight, then one at a time
template <uint SEG, uint VW, uint VX, uint COLS, uint U, typename Rows>
inline void linear_span(thread const Rows& rows, thread const linear_cols<COLS>& wr, thread linear_acc<COLS>& acc, uint lo, uint hi,
                        uint lane) {
    uint k = lo + lane * 4;
    for (; k + 128 * (U - 1) < hi; k += 128 * U) {
        float4 wv[U][COLS];
        for (uint u = 0; u < U; u++)
            for (uint c = 0; c < COLS; c++) wv[u][c] = linear_part<VW>(wr.p[c], k + 128 * u, hi);
        for (uint u = 0; u < U; u++)
            for (uint r = 0; r < 8; r++) {
                const float4 xv = rows.template load<SEG, VX>(r, k + 128 * u, hi);
                for (uint c = 0; c < COLS; c++) acc.v[r][c] += dot(xv, wv[u][c]);
            }
    }
    for (; k < hi; k += 128) {
        float4 wv[COLS];
        for (uint c = 0; c < COLS; c++) wv[c] = linear_part<VW>(wr.p[c], k, hi);
        for (uint r = 0; r < 8; r++) {
            const float4 xv = rows.template load<SEG, VX>(r, k, hi);
            for (uint c = 0; c < COLS; c++) acc.v[r][c] += dot(xv, wv[c]);
        }
    }
}

template <uint VW, uint VX, uint COLS, uint NSG, uint U, typename Src, typename Store>
inline void linear_body(thread const Src& src, device const half* w, const linear_dims d, thread const Store& store,
                        threadgroup float* part, uint sg, uint lane, uint tg, uint tid) {
    const uint R = d.R, K = d.K, N = d.N, n0 = tg * COLS, K4 = K & ~3u;
    const uint span = (K4 / 4 + NSG - 1) / NSG * 4, k0 = sg * span, k1 = min(K4, k0 + span);
    linear_cols<COLS> wr;
    for (uint c = 0; c < COLS; c++) wr.p[c] = w + ulong(min(n0 + c, N - 1)) * d.LDW;
    for (uint r0 = 0; r0 < R; r0 += 8) {
        linear_acc<COLS> acc;
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) acc.v[r][c] = 0.0f;
        const auto rows = src.block(r0, R);
        linear_span<0, VW, VX, COLS, U>(rows, wr, acc, k0, min(k1, d.K0), lane);
        linear_span<1, VW, VX, COLS, U>(rows, wr, acc, max(k0, d.K0), k1, lane);
        if (sg == 0 && K4 + lane < K)
            for (uint r = 0; r < 8; r++)
                for (uint c = 0; c < COLS; c++) acc.v[r][c] += rows.one(r, K4 + lane) * float(wr.p[c][K4 + lane]);
        for (uint r = 0; r < 8; r++)
            for (uint c = 0; c < COLS; c++) {
                const float s = simd_sum(acc.v[r][c]);
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
    const linear_dense src{x, x1, d.LD0, d.LD1, d.K0};
    linear_body<VW, VX, COLS, NSG, U>(src, w, d, store, part, sg, lane, tg, tid);
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
    const linear_dense src{x, s, g.d[p].LD0, g.d[p].LD1, g.d[p].K0};
    linear_body<VW, VX, COLS, NSG, U>(src, w, g.d[p], store, part, sg, lane, tg - g.first[p], tid);
}

#ifdef MESH_METAL
// A fused crossing's landing read as x, so the dispatch that would land it goes (torch_mesh::linear_landed): every
// threadgroup awaits each peer's landing at its repeat j (mesh_step_enter, mesh_step_await), then reads the rows through
// linear_summed: x (rows LD0 apart, none where X0 is 0) plus ACT of the sum of every rank's part, each part a lease
// segment of [R, K] rows at its rank's start, this rank's read plainly and every peer's through mesh_step_read (the NIC
// wrote it).  `landed` takes ACT of the sum alone (keep: the users of the landing that are no product's x), each
// threadgroup a stride of its R x K elements, before its products.
template <uint VX>
inline float4 linear_landing(device const uchar* row, uint k, uint hi) {
    if (VX == 4) return float4(mesh_step_read<half4>(row, k / 4));
    float4 v;
    for (uint j = 0; j < 4; j++) v[j] = k + j < hi ? float(mesh_step_read<half>(row, k + j)) : 0.0f;
    return v;
}

template <uint ACT, uint RANKS>
struct linear_summed_rows {
    device const half* x[8];
    device const uchar* part[RANKS][8];
    uint rank, X0;
    template <uint SEG, uint VX> float4 load(uint r, uint k, uint hi) const {
        float4 s = 0.0f;
        for (uint q = 0; q < RANKS; q++)
            s += q == rank ? linear_part<VX>((device const half*)part[q][r], k, hi) : linear_landing<VX>(part[q][r], k, hi);
        if (ACT == 1) s = tanh(s);
        return X0 ? s + linear_part<VX>(x[r], k, hi) : s;
    }
    float one(uint r, uint k) const {
        float s = 0.0f;
        for (uint q = 0; q < RANKS; q++)
            s += q == rank ? float(((device const half*)part[q][r])[k]) : float(mesh_step_read<half>(part[q][r], k));
        if (ACT == 1) s = tanh(s);
        return X0 ? s + float(x[r][k]) : s;
    }
};

template <uint ACT, uint RANKS>
struct linear_summed {
    device const half* x;
    device const uchar* slot;
    uint LD0, K, rank, X0;
    uint starts[RANKS];
    linear_summed_rows<ACT, RANKS> block(uint r0, uint R) const {
        linear_summed_rows<ACT, RANKS> b;
        b.rank = rank, b.X0 = X0;
        for (uint r = 0; r < 8; r++) {
            const uint row = min(r0 + r, R - 1);
            b.x[r] = x + ulong(row) * LD0;
            for (uint q = 0; q < RANKS; q++) b.part[q][r] = slot + 2 * (ulong(starts[q]) + ulong(row) * K);
        }
        return b;
    }
    float landed(uint r, uint k) const {
        float s = 0.0f;
        for (uint q = 0; q < RANKS; q++) {
            device const uchar* row = slot + 2 * (ulong(starts[q]) + ulong(r) * K);
            s += q == rank ? float(((device const half*)row)[k]) : float(mesh_step_read<half>(row, k));
        }
        return ACT == 1 ? tanh(s) : s;
    }
};

template <uint PEERS>
struct linear_landed_words { uint R, K, N, B, LDW, LD0, X0, rank, keep, j; uint starts[PEERS + 1]; mesh_step_binding land[PEERS], publish[PEERS]; };

template <uint VW, uint VX, uint COLS, uint NSG, uint U, uint ACT, uint SACT, uint PEERS>
[[max_total_threads_per_threadgroup(NSG * 32)]]
kernel void linear_landed(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* y [[buffer(2)]],
                          device const half* bias [[buffer(3)]], device const uchar* slot [[buffer(4)]], device half* landed [[buffer(5)]],
                          constant linear_landed_words<PEERS>& p [[buffer(15)]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint groups [[threadgroups_per_grid]]) {
    threadgroup float part[NSG * 8 * COLS];
    for (uint q = 0; q < PEERS; q++) {
        mesh_step_enter(*p.land[q].p, p.land[q].v, p.land[q].progress, tg == 0 && tid == 0);
        mesh_step_await(*p.land[q].p, p.land[q].a, p.land[q].v, tid, p.j);
    }
    linear_summed<SACT, PEERS + 1> src;
    src.x = x, src.slot = slot, src.LD0 = p.LD0, src.K = p.K, src.rank = p.rank, src.X0 = p.X0;
    for (uint q = 0; q <= PEERS; q++) src.starts[q] = p.starts[q];
    if (p.keep)
        for (uint e = tg * NSG * 32 + tid; e < p.R * p.K; e += groups * NSG * 32) landed[e] = half(src.landed(e / p.K, e % p.K));
    const linear_dims d = {p.R, p.K, p.N, p.B, p.LDW, p.K, p.LD0, p.LD0};
    const linear_store<ACT> store{y, bias, p.N, p.B};
    linear_body<VW, VX, COLS, NSG, U>(src, w, d, store, part, sg, lane, tg, tid);
}

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
    const linear_dense src{x, x1, d.LD0, d.LD1, d.K0};
    linear_body<VW, VX, COLS, NSG, U>(src, w, d, store, part, sg, lane, tg, tid);
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, groups, tid, &last)) {
        for (uint q = 0; q < PEERS; q++) mesh_step_release(*p.publish[q].p, p.publish[q].a, tid, p.j);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}
#endif
