// Decode attention as one algorithm over a KV type (metal-microbench docs/kernels.md#decode-attention; native_attention.metal
// K7): a task of W lanes holds R query rows over a part of a KV range; chunk by chunk each lane scores its positions, the
// chunk's online softmax rescales the rows' maxima, sums and accumulators [Milakov & Gimelshein 2018], and each thread
// accumulates the value product of its channel slice d over its position group g; the groups are summed into group 0,
// and the parts of a range merge as the KV type says (Flash-Decoding [Dao et al. 2023]). The KV type A is the operand:
//
//   A::rows, A::simdgroups, A::slices (DS), A::groups (G = W / DS), A::grouprows (rows a round of the groups' sum),
//   A::fresh (the groups' memory aliases nothing the chunks read), A::score (float or float4: a lane's positions of a
//   row) and A::mask (bool or bool4); members words (2 R simdgroups floats), red (round (G - 1) DS float4s), s, lane, d, g;
//   chunk(i): whether the task holds chunk i; scores(i, sc): the lane's scores of chunk i, scaled, a dead position
//   -3e38, and their mask; put(r, p): a row's probabilities of the lane's positions into the chunk's memory (between the
//   chunk's two barriers); values(i, acc): the chunk's value product of the thread's channel slice; finish(m, l, o): the
//   parts' merge and the store.
//
// Each instance is a literal kernel of its operand: the KV type's constants are its own, and the recorder makes the
// dispatch's sizes and fixed constants literals (metal-microbench docs/kernels.md#one-kernel-interface).
#ifndef TORCH_MESH_ATTENTION
#define TORCH_MESH_ATTENTION
#include <metal_stdlib>
using namespace metal;

__attribute__((always_inline)) static inline float mesh_lanes_max(float v) { return v; }
__attribute__((always_inline)) static inline float mesh_lanes_max(float4 v) { return max(max(v.x, v.y), max(v.z, v.w)); }
__attribute__((always_inline)) static inline float mesh_lanes_sum(float v) { return v; }
__attribute__((always_inline)) static inline float mesh_lanes_sum(float4 v) { return v.x + v.y + v.z + v.w; }

// One chunk's online softmax of R rows: every lane's masked scores sc[r], the chunk's maxima and sums through `words`
// (2 R NSG floats), each row's probabilities to put(r, p) after the first barrier, and each row's running maximum m, sum
// l and accumulator acc rescaled after the second.
template <uint R, uint NSG, typename S, typename B, typename Put>
__attribute__((always_inline)) static inline void mesh_softmax_chunk(thread S *sc, B in, thread float *m, thread float *l, thread float4 *acc,
                                      threadgroup float *words, uint s, uint lane, Put put) {
    for (uint r = 0; r < R; r++) {
        const float top = simd_max(mesh_lanes_max(sc[r]));
        if (lane == 0) words[r * NSG + s] = top;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mn[R];
    for (uint r = 0; r < R; r++) {
        float top = m[r];
        for (uint q = 0; q < NSG; q++) top = max(top, words[r * NSG + q]);
        mn[r] = top;
        const S p = select(S(0.0f), exp(sc[r] - top), in);
        put(r, p);
        const float sum = simd_sum(mesh_lanes_sum(p));
        if (lane == 0) words[(R + r) * NSG + s] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < R; r++) {
        float total = 0.0f;
        for (uint q = 0; q < NSG; q++) total += words[(R + r) * NSG + q];
        const float alpha = exp(m[r] - mn[r]);
        l[r] = l[r] * alpha + total;
        acc[r] *= alpha;
        m[r] = mn[r];
    }
}

// The G position groups' accumulators of R rows summed into group 0 (thread: channel slice d of group g), RC rows a round
// through `red` ((G - 1) RC DS float4s); a round begins with a barrier unless Fresh holds for the first.
template <uint R, uint RC, uint DS, uint G, bool Fresh>
__attribute__((always_inline)) static inline void mesh_position_groups(thread const float4 *acc, thread float4 *o, threadgroup float4 *red, uint d, uint g) {
    for (uint r = 0; r < R; r++) o[r] = acc[r];
    if (G == 1) return;
    for (uint r0 = 0; r0 < R; r0 += RC) {
        if (!Fresh || r0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g > 0)
            for (uint r = 0; r < RC && r0 + r < R; r++) red[(r * (G - 1) + g - 1) * DS + d] = acc[r0 + r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0)
            for (uint r = 0; r < RC && r0 + r < R; r++)
                for (uint e = 1; e < G; e++) o[r0 + r] += red[(r * (G - 1) + e - 1) * DS + d];
    }
}

// The task over its chunks: their scores, softmax and value products, the groups' sum, the KV type's finish.
template <typename A>
__attribute__((always_inline)) static inline void mesh_attention(thread A &a) {
    constexpr uint R = A::rows;
    float m[R], l[R];
    float4 acc[R];
    for (uint r = 0; r < R; r++) { m[r] = -3.0e38f; l[r] = 0.0f; acc[r] = float4(0.0f); }
    for (uint i = 0; a.chunk(i); i++) {
        typename A::score sc[R];
        const typename A::mask in = a.scores(i, sc);
        mesh_softmax_chunk<R, A::simdgroups>(sc, in, m, l, acc, a.words, a.s, a.lane,
                                             [&](uint r, typename A::score p) { a.put(r, p); });
        a.values(i, acc);
    }
    float4 o[R];
    mesh_position_groups<R, A::grouprows, A::slices, A::groups, (A::fresh != 0)>(acc, o, a.red, a.d, a.g);
    a.finish(m, l, o);
}

// The Tile plan (metal-microbench docs/kernels.md#decode-attention, the verify plan; FlashInfer's tensor-core decode for
// grouped queries, llama.cpp's kernel_flash_attn_ext): a task is 8 query rows of one KV head (a stream's rows times the
// query heads that read it) over part j of the KV range, its scores and value products 8 x 8 simdgroup matrices. Each
// chunk is one page a simdgroup (A::simdgroups pages, C = 16 A::simdgroups positions): the page's scores S = Q K^T (two
// 8 x 8 blocks over D / 8 products), the rows' online softmax (8 / A::simdgroups rows a simdgroup, C / 32 positions a
// lane), the accumulators rescaled by the diagonal of the rows' factors, and O += P V over the chunk's pages, each
// simdgroup holding the 8 x 8 blocks of every (A::simdgroups)th column block. The operand A:
//
//   A::dims (D), A::simdgroups; members qs (8 D halves, the rows' queries: load_q writes them and ends with a barrier),
//   ss (8 C floats), ps (8 C halves), diag (64 floats), places (A::simdgroups ulongs), scale (the scores' factor), stride
//   (elements from a position's K or V line to the next's), s, lane, t; chunk(i): whether the task holds chunk i; page(i,
//   w, k, v): page w of chunk i, the addresses of its K and V lines (false where the task reads none of it); first(i, w):
//   the page's first position;
//   live(r, position): whether row r attends the position; finish(o, m, l): the parts' merge and the store, o the
//   simdgroup's column blocks, m and l its rows'.
template <typename A>
__attribute__((always_inline)) static inline void mesh_attention_tile(thread A &a) {
    constexpr uint D = A::dims, NSG = A::simdgroups, PAGE = 16, C = NSG * PAGE, DB = D / 8, OB = DB / NSG, RS = 8 / NSG > 0 ? 8 / NSG : 1;
    const uint s = a.s, lane = a.lane, stride = a.stride;
    for (uint e = a.t; e < 64; e += NSG * 32) a.diag[e] = 0.0f;
    a.load_q();
    float m[RS], l[RS];
    for (uint k = 0; k < RS; k++) { m[k] = -3.0e38f; l[k] = 0.0f; }
    simdgroup_float8x8 o[OB];
    for (uint b = 0; b < OB; b++) o[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint i = 0; a.chunk(i); i++) {
        ulong kq = 0ul, vq = 0ul;
        const bool cached = a.page(i, s, kq, vq);
        device const half *kp = (device const half *)kq;
        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), s1 = s0, s2 = s0, s3 = s0;
        if (cached)
            for (uint d0 = 0; d0 < DB; d0 += 8) {
                simdgroup_half8x8 k[16];
                for (uint u = 0; u < 8; u++) {
                    simdgroup_load(k[2 * u], kp + 8 * (d0 + u), stride);
                    simdgroup_load(k[2 * u + 1], kp + 8 * stride + 8 * (d0 + u), stride);
                }
                for (uint u = 0; u < 8; u += 2) {
                    simdgroup_half8x8 q0, q1;
                    simdgroup_load(q0, a.qs + 8 * (d0 + u), D, ulong2(0, 0), true);
                    simdgroup_load(q1, a.qs + 8 * (d0 + u) + 8, D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s0, k[2 * u], q0, s0);
                    simdgroup_multiply_accumulate(s1, k[2 * u + 1], q0, s1);
                    simdgroup_multiply_accumulate(s2, k[2 * u + 2], q1, s2);
                    simdgroup_multiply_accumulate(s3, k[2 * u + 3], q1, s3);
                }
            }
        s0.thread_elements() += s2.thread_elements();
        s1.thread_elements() += s3.thread_elements();
        simdgroup_store(s0, a.ss + s * PAGE, C, ulong2(0, 0), true);
        simdgroup_store(s1, a.ss + s * PAGE + 8, C, ulong2(0, 0), true);
        if (lane == 0) a.places[s] = cached ? vq : 0ul;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < RS && s * RS + k < 8; k++) {
            const uint r = s * RS + k;
            float x[C / 32];
            bool in[C / 32];
            float top = -3.0e38f;
            for (uint u = 0; u < C / 32; u++) {
                const uint c = lane + 32 * u, w = c / PAGE;
                in[u] = a.places[w] != 0ul && a.live(r, a.first(i, w) + c % PAGE);
                x[u] = in[u] ? a.ss[r * C + c] * a.scale : -3.0e38f;
                top = max(top, x[u]);
            }
            const float mn = max(m[k], simd_max(top));
            float sum = 0.0f;
            for (uint u = 0; u < C / 32; u++) {
                const float p = in[u] ? exp(x[u] - mn) : 0.0f;
                a.ps[r * C + lane + 32 * u] = half(p);
                sum += p;
            }
            const float alpha = exp(m[k] - mn);
            l[k] = l[k] * alpha + simd_sum(sum);
            m[k] = mn;
            if (lane == 0) a.diag[9 * r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scale;
        simdgroup_load(scale, a.diag, 8);
        for (uint b = 0; b < OB; b++) simdgroup_multiply(o[b], scale, o[b]);
        for (uint w = 0; w < NSG; w++) {
            const ulong place = a.places[w];
            if (place == 0ul) continue;
            device const half *v = (device const half *)place;
            for (uint h = 0; h < 2; h++) {
                simdgroup_half8x8 p, vm[OB];
                simdgroup_load(p, a.ps + w * PAGE + 8 * h, C);
                for (uint b = 0; b < OB; b++) simdgroup_load(vm[b], v + 8 * h * stride + 8 * (b * NSG + s), stride);
                for (uint b = 0; b < OB; b++) simdgroup_multiply_accumulate(o[b], p, vm[b], o[b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    a.finish(o, m, l);
}
#endif
