// Speculative decoding's drafter on the device (metal-microbench docs/kernels.md#the-drafter): DFlash2 [2ndgenMTP's
// parallel-block drafter; Chen et al. 2026, DFlash], one pass of K block rows [anchor, MASK x (K - 1)] at positions p0 ..
// p0 + K - 1 over the target's hidden states, its products the streamed family's instances, its attention a Tile plan
// operand (attention.metal) merged by the host's merge (metal-microbench's decode_attention_merge, over mesh_merge_row), and
// its proposals the selector's walk.
//
// The context: every position p before the anchor holds, in each draft layer's flat cache (one K = V head of HD
// channels a position), k_norm(rope(k(c_p))) and v_norm(k(c_p)), c_p = hidden_norm(fc(the target's taps of position p)),
// written from the prefill's rows and from each verify step's rows (their positions; a rejected row's line is written
// again before an anchor past it reads it). The block writes its own rows' lines at p0 .. p0 + K - 1, so DFlash's relation
// (the context before the anchor and the whole block) is the length p0 + K for every block row.
//
// The walk (2ndgenMTP's DFLASH2_HEAD, greedy): slot k's score of candidate v is softcap(base_k[v]) + (A e(prev) * H h_k)
// . P[v], P = B E (the draft vocabulary's rows, precomputed), over the C greatest base scores of the slot; prev the anchor,
// then the slot before's choice (A E precomputed over the draft vocabulary, A e(anchor) by its own pass).
#ifndef TORCH_MESH_DRAFTER
#define TORCH_MESH_DRAFTER
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MESH_DRAFT_GROUP = 16;

// A tap: the target's hidden rows after one of its tap layers into slot `slot` of each row's taps (TAPS a row). The rows
// lie row after row (`interleaved` 0), or four channels of every row a slice, slice after slice over `interleaved` rows
// (a LiteRT delegate's tensor: metal-microbench tools/prepare_e2b.py prefill_mask).
template <uint D, uint TAPS>
kernel void mesh_draft_tap(device const half *hidden [[buffer(0)]], device half *taps [[buffer(1)]], constant uint &slot [[buffer(2)]],
                           constant uint &interleaved [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
    const uint c = at.x, r = at.y;
    if (c >= D / 4) return;
    ((device half4 *)taps)[(r * TAPS + slot) * (D / 4) + c] =
        ((device const half4 *)hidden)[interleaved == 0u ? r * (D / 4) + c : c * interleaved + r];
}

// The block's input rows: the anchor's embedding row times scale (the target's table), then the mask row.
template <uint D>
kernel void mesh_draft_embed(device const half *table [[buffer(0)]], device const uint *state [[buffer(1)]],
                             device const half *mask [[buffer(2)]], device half *x [[buffer(3)]], constant float &scale [[buffer(4)]],
                             uint2 at [[thread_position_in_grid]]) {
    const uint c = at.x, k = at.y;
    if (c >= D / 4) return;
    const uint anchor = state[8];
    ((device half4 *)x)[k * (D / 4) + c] = k == 0 ? half4(float4(((device const half4 *)table)[ulong(anchor) * (D / 4) + c]) * scale)
                                                  : ((device const half4 *)mask)[c];
}

// y_k = (a_k + a0) x_k + (b_k + b0) x_{k-1} (x_{-1} = x_0), a_k | b_k the tap generator's product of row k (2 D / 16 channels),
// a0 | b0 its bias, one coefficient a group of 16 channels: channel c of row k, in fp32.
template <uint D>
static inline float mesh_draft_conv_value(device const half *x, device const half *taps, device const half *bias, uint k, uint c) {
    constexpr uint G = D / MESH_DRAFT_GROUP;
    const uint g = c / MESH_DRAFT_GROUP, p = k > 0 ? k - 1 : 0;
    const float a = float(taps[k * 2 * G + g]) + float(bias[g]), b = float(taps[k * 2 * G + G + g]) + float(bias[G + g]);
    return a * float(x[k * D + c]) + b * float(x[p * D + c]);
}

template <uint D>
kernel void mesh_draft_conv(device const half *x [[buffer(0)]], device const half *taps [[buffer(1)]], device const half *bias [[buffer(2)]],
                            device half *y [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
    if (at.x < D) y[at.y * D + at.x] = half(mesh_draft_conv_value<D>(x, taps, bias, at.y, at.x));
}

// Rows' heads finished (streamed.metal mesh_head over plain rows, precise trigonometry): threadgroup (h, row) of HD
// threads; h < H a query head of q (row stride qs) into qo, normed by qw and rotated over the first ROT frequency pairs;
// h = H the row's K = V line (k + ko, row stride ks) into both caches at the row's position: V normed without a weight, K
// normed by kw and rotated. A block row's (BLOCK, grid H + 1 by rows) position is state[1] + row, a context row's (grid 1
// by rows, its line alone) positions[row].
template <uint HD, uint H, uint ROT, bool BLOCK>
kernel void mesh_draft_heads(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *qw [[buffer(2)]],
                             device const half *kw [[buffer(3)]], device half *qo [[buffer(4)]], device half *keys [[buffer(5)]],
                             device half *values [[buffer(6)]], device const uint *state [[buffer(7)]], device const uint *positions [[buffer(8)]],
                             constant float &theta [[buffer(9)]], constant uint &qs [[buffer(10)]], constant uint &ks [[buffer(11)]],
                             constant uint &ko [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup float vals[HD];
    threadgroup float gs[HD];
    threadgroup float partial[32];
    const uint h = BLOCK ? tg.x : H, r = tg.y;
    const uint position = BLOCK ? state[1] + r : positions[r];
    device const float *none = nullptr;
    if (h < H) {
        mesh_head<true>(none, q, nullptr, nullptr, r, h * HD, HD, qs, 0, 0, qw, ROT, position, theta, qo + (ulong(r) * H + h) * HD, nullptr,
                        vals, gs, partial, i, HD);
        return;
    }
    mesh_head<true>(none, k, nullptr, nullptr, r, ko, HD, ks, 0, 0, nullptr, 0, position, theta, nullptr, values + ulong(position) * HD,
                    vals, gs, partial, i, HD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mesh_head<true>(none, k, nullptr, nullptr, r, ko, HD, ks, 0, 0, kw, ROT, position, theta, nullptr, keys + ulong(position) * HD,
                    vals, gs, partial, i, HD);
}

// The Tile plan's KV over a flat cache (attention.metal mesh_tile): page p the 16 lines from position 16 p of the one K = V
// head; 8 query rows (block row i / H, head i % H) a task, part j of J over the pages of [lo, limit) (mesh_keys of the block
// [state[1], state[1] + K) as rows attending each other, its window W + 1 from the block's first row: lo the anchor less the
// window W, limit state[1] + K), every row reading every position there.
template <uint HD>
struct mesh_draft_kv {
    device const half *keys;
    device const half *values;
    __attribute__((always_inline)) bool at(uint p, thread ulong &k, thread ulong &v) {
        k = ulong(keys + ulong(p) * 16 * HD);
        v = ulong(values + ulong(p) * 16 * HD);
        return true;
    }
};

template <uint HD, uint K, uint H, uint J, uint W>
kernel void mesh_draft_attention(device const half *Q [[buffer(0)]], device const half *keys [[buffer(1)]], device const half *values [[buffer(2)]],
                                 device float *mp [[buffer(3)]], device float *lp [[buffer(4)]], device float4 *op [[buffer(5)]],
                                 device const uint *state [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    typedef mesh_tile<HD, mesh_draft_kv<HD>> tile;
    threadgroup float region[tile::region];
    threadgroup float diag[64];
    threadgroup ulong places[tile::simdgroups];
    tile task{{keys, values}, Q, mp, lp, op};
    task.layout(region, diag, places);
    task.scale = 1.0f; task.stride = HD; task.s = sg; task.lane = lane; task.t = t; task.j = tg.y; task.parts = J;
    const uint2 bounds = mesh_keys(state[1], state[1] + K, W + 1, state[1], state[1] + K);
    for (uint r = 0; r < 8; ++r) {
        task.rows[r] = 8 * tg.x + r < K * H ? 8 * tg.x + r : 0xFFFFFFFFu;
        task.lo[r] = bounds.x;
        task.hi[r] = bounds.y;
    }
    task.span(J, true);
    mesh_attention_tile(task);
}

// The convolution after attention or the FFN, its norm and the residual add as one row (a threadgroup a row, in fp32):
// x = (x + gamma rms(c)) * scale, c = (a_k + a0) y_k + (b_k + b0) y_{k-1} the two-tap convolution of y (scale the layer
// scalar where SCALED). A trained post-FFN convolution's output passes fp16's range (its norm undoes any scale; 2ndgenMTP
// trains in bf16), so it is never stored.
template <uint D, bool SCALED>
kernel void mesh_draft_conv_norm_add(device const half *y [[buffer(0)]], device const half *taps [[buffer(1)]], device const half *bias [[buffer(2)]],
                                     device const half *gamma [[buffer(3)]], device half *x [[buffer(4)]], device const half *scalar [[buffer(5)]],
                                     uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                     uint threads [[threads_per_threadgroup]]) {
    threadgroup float vals[D];
    threadgroup float part[32];
    float s = 0.0f;
    for (uint c = t; c < D; c += threads) {
        vals[c] = mesh_draft_conv_value<D>(y, taps, bias, r, c);
        s += vals[c] * vals[c];
    }
    const float inv = streamed_rms(s, D, 1e-6f, part, t, threads), ls = SCALED ? float(scalar[0]) : 1.0f;
    for (uint c = t; c < D; c += threads) x[r * D + c] = half((float(x[r * D + c]) + vals[c] * inv * float(gamma[c])) * ls);
}

// Each slot's C candidates: the C greatest softcapped base scores of its row (V of a row of `stride`; the threshold by
// bisection over the row's range, then the positions at or above it in index order), into cand (index into the draft
// vocabulary) and score.
template <uint V, uint C, uint T>
kernel void mesh_draft_candidates(device const half *base [[buffer(0)]], device uint *cand [[buffer(1)]], device float *score [[buffer(2)]],
                                  constant float &cap [[buffer(3)]], constant uint &stride [[buffer(4)]], uint k [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N = V / T;
    threadgroup float part[T / 32];
    threadgroup uint counts[T / 32];
    threadgroup uint offsets[T / 32 + 1];
    float v[N];
    float top = -3.0e38f, low = 3.0e38f;
    for (uint u = 0; u < N; u++) {
        const float z = float(base[ulong(k) * stride + u * T + t]);
        v[u] = streamed_capped(z, cap);
        top = max(top, v[u]);
        low = min(low, v[u]);
    }
    top = simd_max(top); low = simd_min(low);
    if (lane == 0) { part[sg] = top; counts[sg] = as_type<uint>(low); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    top = part[0]; low = as_type<float>(counts[0]);
    for (uint u = 1; u < T / 32; u++) { top = max(top, part[u]); low = min(low, as_type<float>(counts[u])); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lo = low, hi = top;
    for (uint it = 0; it < 16; it++) {
        const float mid = 0.5f * (lo + hi);
        uint n = 0;
        for (uint u = 0; u < N; u++) n += v[u] >= mid;
        n = simd_sum(n);
        if (lane == 0) counts[sg] = n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint all = 0;
        for (uint u = 0; u < T / 32; u++) all += counts[u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (all >= C) lo = mid; else hi = mid;
    }
    uint n = 0;
    for (uint u = 0; u < N; u++) n += v[u] >= lo;
    const uint before = simd_prefix_exclusive_sum(n), within = simd_sum(n);
    if (lane == 0) counts[sg] = within;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        uint run = 0;
        for (uint u = 0; u < T / 32; u++) { offsets[u] = run; run += counts[u]; }
        offsets[T / 32] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = offsets[sg] + before;
    for (uint u = 0; u < N; u++)
        if (v[u] >= lo) {
            if (at < C) { cand[k * C + at] = u * T + t; score[k * C + at] = v[u]; }
            at++;
        }
    for (uint e = offsets[T / 32] + t; e < C; e += T) { cand[k * C + e] = 0; score[k * C + e] = -3.0e38f; }
}

// The predecessor projection of the anchor: pa = A e(anchor) (rank RK, the target's table unscaled), RK / 8 threadgroups of
// 8 simdgroups, a simdgroup an output.
template <uint D, uint RK>
kernel void mesh_draft_anchor(device const half *table [[buffer(0)]], device const half *A [[buffer(1)]], device const uint *state [[buffer(2)]],
                              device float *pa [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = tg * 8 + sg;
    const ulong anchor = state[8];
    float s = 0.0f;
    for (uint c = lane; c < D; c += 32) s += float(A[ulong(j) * D + c]) * float(table[anchor * D + c]);
    s = simd_sum(s);
    if (lane == 0) pa[j] = s;
}

// The selector's walk over S slots in one threadgroup of T threads: slot k's predecessor projection (pa for the anchor, then
// the row of AE = A E for the slot before's choice, E the draft vocabulary's rows of the target's table) times gate_k (h_gate
// of the slot's hidden row, row stride gs), each candidate's score plus that . P[candidate] (P = B E), the greatest (the
// first candidate on a tie: argmax.metal mesh_argmax over the threadgroup) its proposal, drafts[k + 1] = ids[that candidate]; drafts[0] the anchor. Sampling (inverse >
// 0): the scores over the temperature plus the Gumbel noise the target's row at p0 + k draws its token with (argmax.metal
// mesh_gumbel), so a draft is the token the target samples wherever the two distributions' arg maxima meet.
template <uint RK, uint C, uint S, uint T>
kernel void mesh_draft_select(device const float *pa [[buffer(0)]], device const half *ae [[buffer(1)]], device const half *gate [[buffer(2)]],
                              device const half *projected [[buffer(3)]], device const uint *cand [[buffer(4)]], device const float *score [[buffer(5)]],
                              device const int *ids [[buffer(6)]], device const uint *state [[buffer(7)]], device uint *drafts [[buffer(8)]],
                              constant float &inverse [[buffer(9)]], constant uint &seed [[buffer(10)]], constant uint &gs [[buffer(11)]],
                              uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NSG = T / 32;
    threadgroup float p[RK];
    threadgroup float best[C];
    threadgroup float tv[T / 32];
    threadgroup uint tk[T / 32];
    threadgroup uint previous;
    if (t == 0) drafts[0] = state[8];
    for (uint k = 0; k < S; k++) {
        for (uint j = t; j < RK; j += T)
            p[j] = (k == 0 ? pa[j] : float(ae[ulong(previous) * RK + j])) * float(gate[k * gs + j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = sg; c < C; c += NSG) {
            float s = 0.0f;
            const uint v = cand[k * C + c];
            for (uint j = lane; j < RK; j += 32) s += p[j] * float(projected[ulong(v) * RK + j]);
            s = simd_sum(s);
            if (lane == 0) best[c] = inverse > 0.0f ? (score[k * C + c] + s) * inverse + mesh_gumbel(seed, state[1] + k, uint(ids[v]))
                                                    : score[k * C + c] + s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = -INFINITY;
        uint pick = 0xFFFFFFFFu;
        for (uint c = t; c < C; c += T) mesh_take(top, pick, best[c], c);
        if (mesh_argmax<T>(top, pick, t, tv, tk)) {
            previous = cand[k * C + pick];
            drafts[k + 1] = uint(ids[previous]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Every member's step rows from the drafts (after their crossing): rows 1 .. R - 1 the proposals, unless the stream
// stopped (spec_step's flags bit 1: its rows repeat their positions), given (bit 2: its rows replay the history) or, with
// `hybrid` > 0, the prompt lookup's match this step (spec_step's state word 4) reached `hybrid` tokens: its rows stand.
template <uint R>
kernel void mesh_spec_drafts(device uint *state [[buffer(0)]], device const uint *drafts [[buffer(1)]], device uint *words [[buffer(2)]],
                             device uint *record [[buffer(3)]], constant uint &stride [[buffer(4)]], constant uint &hybrid [[buffer(5)]],
                             uint r [[thread_position_in_grid]]) {
    if (r == 0 || r >= R || (state[3] & 6u) != 0u || (hybrid > 0u && state[4] >= hybrid)) return;
    words[r * stride] = drafts[r];
    state[8 + r] = drafts[r];
    record[3 + r] = drafts[r];
}
#endif
