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

static inline float mesh_lanes_max(float v) { return v; }
static inline float mesh_lanes_max(float4 v) { return max(max(v.x, v.y), max(v.z, v.w)); }
static inline float mesh_lanes_sum(float v) { return v; }
static inline float mesh_lanes_sum(float4 v) { return v.x + v.y + v.z + v.w; }

// One chunk's online softmax of R rows: every lane's masked scores sc[r], the chunk's maxima and sums through `words`
// (2 R NSG floats), each row's probabilities to put(r, p) after the first barrier, and each row's running maximum m, sum
// l and accumulator acc rescaled after the second.
template <uint R, uint NSG, typename S, typename B, typename Put>
static inline void mesh_softmax_chunk(thread S *sc, B in, thread float *m, thread float *l, thread float4 *acc,
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
static inline void mesh_position_groups(thread const float4 *acc, thread float4 *o, threadgroup float4 *red, uint d, uint g) {
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
static inline void mesh_attention(thread A &a) {
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
#endif
