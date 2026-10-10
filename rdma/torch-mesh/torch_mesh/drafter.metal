// Speculative decoding's drafter on the device (metal-microbench docs/kernels.md#the-drafter): DFlash2 [2ndgenMTP's
// parallel-block drafter; Chen et al. 2026, DFlash], one pass of K block rows [anchor, MASK x (K - 1)] at positions p0 ..
// p0 + K - 1 over the target's hidden states, its products the streamed family's instances, its attention a Tile plan
// operand (attention.metal) and its proposals the selector's walk.
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

// A tap: the target's hidden rows after one of its tap layers into slot `slot` of each row's taps (TAPS a row).
template <uint D, uint TAPS>
kernel void mesh_draft_tap(device const half *hidden [[buffer(0)]], device half *taps [[buffer(1)]], constant uint &slot [[buffer(2)]],
                           uint2 at [[thread_position_in_grid]]) {
    const uint c = at.x, r = at.y;
    if (c >= D / 4) return;
    ((device half4 *)taps)[(r * TAPS + slot) * (D / 4) + c] = ((device const half4 *)hidden)[r * (D / 4) + c];
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
// a0 | b0 its bias, one coefficient a group of 16 channels.
template <uint D>
kernel void mesh_draft_conv(device const half *x [[buffer(0)]], device const half *taps [[buffer(1)]], device const half *bias [[buffer(2)]],
                            device half *y [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
    constexpr uint G = D / MESH_DRAFT_GROUP;
    const uint c = at.x, k = at.y;
    if (c >= D) return;
    const uint g = c / MESH_DRAFT_GROUP, p = k > 0 ? k - 1 : 0;
    const float a = float(taps[k * 2 * G + g]) + float(bias[g]), b = float(taps[k * 2 * G + G + g]) + float(bias[G + g]);
    y[k * D + c] = half(a * float(x[k * D + c]) + b * float(x[p * D + c]));
}

// Rows' per-head RMS norms and the proportional rotary (rope_theta, the first ROT frequencies of HD / 2, channel i paired
// with i + HD / 2): threadgroup (h, row) of HD threads; h < H a query head of q (row stride qs) into qo, h = H the row's
// K = V line (k + ko, row stride ks): K normed by kw and rotated, V normed without a weight, both at the row's position in
// the caches: a block row's (BLOCK, grid H + 1 by rows) at state[1] + row, a context row's (grid 1 by rows, its line alone)
// at positions[row].
template <uint HD, uint H, uint ROT, bool BLOCK>
kernel void mesh_draft_heads(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *qw [[buffer(2)]],
                             device const half *kw [[buffer(3)]], device half *qo [[buffer(4)]], device half *keys [[buffer(5)]],
                             device half *values [[buffer(6)]], device const uint *state [[buffer(7)]], device const uint *positions [[buffer(8)]],
                             constant float &theta [[buffer(9)]], constant uint &qs [[buffer(10)]], constant uint &ks [[buffer(11)]],
                             constant uint &ko [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[HD / 32];
    threadgroup float row[HD];
    const uint h = BLOCK ? tg.x : H, r = tg.y;
    const uint position = BLOCK ? state[1] + r : positions[r];
    const float x = h < H ? float(q[ulong(r) * qs + h * HD + i]) : float(k[ulong(r) * ks + ko + i]);
    const float s = simd_sum(x * x);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint u = 0; u < HD / 32; u++) total += part[u];
    const float inv = rsqrt(total / float(HD) + 1e-6f);
    const float normed = x * inv;
    if (h == H) values[ulong(position) * HD + i] = half(normed);
    row[i] = normed * float(h < H ? qw[i] : kw[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint f = i % (HD / 2);
    float out = row[i];
    if (f < ROT) {
        const float angle = float(position) * exp2(-float(2 * f) / float(HD) * log2(theta));
        const float c = precise::cos(angle), sn = precise::sin(angle);
        out = i < HD / 2 ? row[i] * c - row[i + HD / 2] * sn : row[i] * c + row[i - HD / 2] * sn;
    }
    if (h < H) qo[(ulong(r) * H + h) * HD + i] = half(out);
    else keys[ulong(position) * HD + i] = half(out);
}

// The Tile plan's operand over a flat cache: 8 query rows (block row i / H, head i % H) of the one K = V head, part j of J
// over the pages of [lo, limit) (lo the anchor less the window W, limit state[1] + K), every row reading every position
// there.
template <uint HD>
struct mesh_draft_tile {
    enum : uint { dims = HD, simdgroups = 4 };
    device const half *keys;
    device const half *values;
    device const half *Q;
    device float *mp;
    device float *lp;
    device float4 *op;
    threadgroup half *qs;
    threadgroup float *ss;
    threadgroup half *ps;
    threadgroup float *diag;
    threadgroup ulong *places;
    threadgroup float *stage;
    float scale;
    uint stride, s, lane, t, row0, total, J, j, base, pages, lo, limit;

    __attribute__((always_inline)) void load_q() {
        for (uint e = t; e < 2 * HD; e += 32 * simdgroups) {
            const uint i = row0 + e / (HD / 4);
            ((threadgroup half4 *)qs)[e] = i < total ? ((device const half4 *)Q)[ulong(i) * (HD / 4) + e % (HD / 4)] : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    __attribute__((always_inline)) bool chunk(uint i) { return j + J * simdgroups * i < pages; }
    __attribute__((always_inline)) bool page(uint i, uint w, thread ulong &k, thread ulong &v) {
        const uint p = j + J * (simdgroups * i + w);
        if (p >= pages) return false;
        k = ulong(keys + ulong(base + p) * 16 * HD);
        v = ulong(values + ulong(base + p) * 16 * HD);
        return true;
    }
    __attribute__((always_inline)) uint first(uint i, uint w) { return (base + j + J * (simdgroups * i + w)) * 16; }
    __attribute__((always_inline)) bool live(uint, uint position) { return position >= lo && position < limit; }
    __attribute__((always_inline)) void finish(thread simdgroup_float8x8 *o, thread const float *m, thread const float *l) {
        constexpr uint NSG = simdgroups, OB = HD / 8 / NSG, RS = 8 / NSG, D4 = HD / 4;
        for (uint b = 0; b < OB; b++) simdgroup_store(o[b], stage + 8 * (b * NSG + s), HD);
        if (lane == 0)
            for (uint k = 0; k < RS; k++) {
                const uint i = row0 + s * RS + k;
                if (i >= total) continue;
                mp[ulong(i) * J + j] = l[k] > 0.0f ? m[k] : -INFINITY;
                lp[ulong(i) * J + j] = l[k];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = t; e < 2 * HD; e += 32 * NSG) {
            const uint i = row0 + e / D4;
            if (i < total) op[(ulong(i) * J + j) * D4 + e % D4] = ((threadgroup const float4 *)stage)[e];
        }
    }
};

template <uint HD, uint K, uint H, uint J, uint W>
kernel void mesh_draft_attention(device const half *Q [[buffer(0)]], device const half *keys [[buffer(1)]], device const half *values [[buffer(2)]],
                                 device float *mp [[buffer(3)]], device float *lp [[buffer(4)]], device float4 *op [[buffer(5)]],
                                 device const uint *state [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NSG = mesh_draft_tile<HD>::simdgroups, C = 16 * NSG;
    threadgroup float region[8 * HD > 4 * HD + 12 * C ? 8 * HD : 4 * HD + 12 * C];
    threadgroup float diag[64];
    threadgroup ulong places[NSG];
    mesh_draft_tile<HD> task{keys, values, Q, mp, lp, op, (threadgroup half *)region, region + 4 * HD,
                             (threadgroup half *)(region + 4 * HD + 8 * C), diag, places, region};
    task.scale = 1.0f; task.stride = HD; task.s = sg; task.lane = lane; task.t = t; task.j = tg.y; task.J = J;
    task.row0 = 8 * tg.x; task.total = K * H;
    task.limit = state[1] + K;
    task.lo = state[1] > W ? state[1] - W : 0u;
    task.base = task.lo / 16;
    task.pages = (task.limit + 15) / 16 - task.base;
    mesh_attention_tile(task);
}

// The parts' merge of each query row (Flash-Decoding's), into O [K H HD] half.
template <uint HD, uint J>
kernel void mesh_draft_merge(device const float *mp [[buffer(0)]], device const float *lp [[buffer(1)]], device const float4 *op [[buffer(2)]],
                             device half *O [[buffer(3)]], uint i [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    float top = -3.0e38f;
    for (uint p = 0; p < J; p++) if (lp[i * J + p] > 0.0f) top = max(top, mp[i * J + p]);
    float total = 0.0f;
    float4 sum = float4(0.0f);
    for (uint p = 0; p < J; p++) {
        const float le = lp[i * J + p], w = le > 0.0f ? exp(mp[i * J + p] - top) : 0.0f;
        total += w * le;
        sum += w * op[(i * J + p) * (HD / 4) + d];
    }
    ((device half4 *)O)[i * (HD / 4) + d] = half4(total > 0.0f ? sum / total : float4(0.0f));
}

// x = (x + gamma rms(y)) * scale, a threadgroup a row (scale: the layer scalar where SCALED).
template <uint D, bool SCALED>
kernel void mesh_draft_norm_add(device const half *y [[buffer(0)]], device const half *gamma [[buffer(1)]], device half *x [[buffer(2)]],
                                device const half *scalar [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    float s = 0.0f;
    for (uint c = t; c < D; c += threads) { const float v = float(y[r * D + c]); s += v * v; }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint u = 0; u < (threads + 31) / 32; u++) total += part[u];
    const float inv = rsqrt(total / float(D) + 1e-6f), ls = SCALED ? float(scalar[0]) : 1.0f;
    for (uint c = t; c < D; c += threads)
        x[r * D + c] = half((float(x[r * D + c]) + float(y[r * D + c]) * inv * float(gamma[c])) * ls);
}

// The FFN's hidden: GELU (tanh form) of gate times up, each row's gate and up the halves of its gate|up row (width F each).
kernel void mesh_draft_gelu(device const half *gu [[buffer(0)]], device half *out [[buffer(1)]], constant uint &count [[buffer(2)]],
                            constant uint &width [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint r = i / width, c = i % width;
    const float g = float(gu[ulong(r) * 2 * width + c]), inner = clamp(0.7978845608f * (g + 0.044715f * g * g * g), -20.0f, 20.0f);
    out[i] = half(0.5f * g * (1.0f + tanh(inner)) * float(gu[ulong(r) * 2 * width + width + c]));
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
        v[u] = cap > 0.0f ? tanh(z / cap) * cap : z;
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
    for (uint it = 0; it < 24; it++) {
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
// first candidate on a tie) its proposal, drafts[k + 1] = ids[that candidate]; drafts[0] the anchor. Sampling (inverse >
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
        if (t == 0) {
            uint pick = 0;
            for (uint c = 1; c < C; c++) if (best[c] > best[pick]) pick = c;
            previous = cand[k * C + pick];
            drafts[k + 1] = uint(ids[previous]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Every member's step rows from the drafts (after their crossing): rows 1 .. R - 1 the proposals, unless the stream
// stopped (spec_step's flags bit 1: its rows repeat their positions).
template <uint R>
kernel void mesh_spec_drafts(device uint *state [[buffer(0)]], device const uint *drafts [[buffer(1)]], device uint *words [[buffer(2)]],
                             device uint *record [[buffer(3)]], constant uint &stride [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    if (r == 0 || r >= R || (state[3] & 2u) != 0u) return;
    words[r * stride] = drafts[r];
    state[8 + r] = drafts[r];
    record[3 + r] = drafts[r];
}
#endif
