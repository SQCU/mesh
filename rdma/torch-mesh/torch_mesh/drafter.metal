// Speculative decoding's drafter on the device (metal-microbench docs/kernels.md#the-drafter): DFlash2 [2ndgenMTP's
// parallel-block drafter; Chen et al. 2026, DFlash], one pass of K block rows [anchor, MASK x (K - 1)] at positions p0 ..
// p0 + K - 1 over the target's hidden states, its products the streamed family's instances, its attention a Tile plan
// operand (attention.metal) merged by the host's merge (metal-microbench's decode_attention_merge, over mesh_merge_row), and
// its proposals the selector's walk; or E2B's own MTP drafter (mesh_mtp_*, below), a chain of passes over the target's
// own caches.
//
// The context: every position p before the anchor holds, in each draft layer's flat cache (one K = V head of HD
// channels a position), k_norm(rope(k(c_p))) and v_norm(k(c_p)), c_p = hidden_norm(fc(the target's taps of position p)),
// written from the prefill's rows and from each verify step's rows (their positions; a rejected row's line is written
// again before an anchor past it reads it). The block writes its own rows' lines at p0 .. p0 + K - 1, so DFlash's relation
// (the context before the anchor and the whole block) is the length p0 + K for every block row.
//
// The walk (2ndgenMTP's DFLASH2_HEAD): slot k's score of candidate v is softcap(base_k[v]) + (A e(prev) * H h_k)
// . P[v], P = B E (the draft vocabulary's rows, precomputed), over the C greatest base scores of the slot; prev the anchor,
// then the slot before's choice (A E precomputed over the draft vocabulary, A e(anchor) by its own pass); its choice the
// draw of the scores at temperature 1 with the target's own Gumbel noise (mesh_draft_select).
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

// Each slot's C candidates: the C greatest softcapped base scores of its row (V of a row of `stride`) by argmax.metal
// mesh_top, in ascending index order, into cand (index into the draft vocabulary) and score; an absent one (0, -inf).
template <uint V, uint C, uint T>
kernel void mesh_draft_candidates(device const half *base [[buffer(0)]], device uint *cand [[buffer(1)]], device float *score [[buffer(2)]],
                                  constant float &cap [[buffer(3)]], constant uint &stride [[buffer(4)]], uint k [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint w[260 + T / 32];
    const float limit = cap;
    device const half *row = base + ulong(k) * stride;
    device uint *ci = cand + k * C;
    device float *si = score + k * C;
    auto z = [&](uint c) { return streamed_capped(float(row[c]), limit); };
    mesh_top<T>(z, 0, V, C, t, w, [&](uint i, uint c) { ci[i] = c == 0xffffffffu ? 0u : c; si[i] = c == 0xffffffffu ? -INFINITY : z(c); });
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
// of the slot's hidden row, row stride gs), each candidate's score plus that . P[candidate] (P = B E), and its proposal
// the draw of those scores at temperature 1 (argmax.metal mesh_draw over the C candidates, candidate c's token
// ids[cand[c]], keyed as the target's row at p0 + k; `seed` the stream's selection word): drafts[k + 1] = ids[the drawn
// candidate]; drafts[0] the anchor. The drafter's draw and the target's share their noise, so a draft is the token the
// target samples wherever the two draws meet. Each candidate's walked score (before the noise) is kept, walked[k C + c],
// for the speculation's keys (mesh_saguaro_keys).
template <uint RK, uint C, uint S, uint T>
kernel void mesh_draft_select(device const float *pa [[buffer(0)]], device const half *ae [[buffer(1)]], device const half *gate [[buffer(2)]],
                              device const half *projected [[buffer(3)]], device const uint *cand [[buffer(4)]], device const float *score [[buffer(5)]],
                              device const int *ids [[buffer(6)]], device const uint *state [[buffer(7)]], device uint *drafts [[buffer(8)]],
                              device const uint &seed [[buffer(9)]], constant uint &gs [[buffer(10)]], device float *walked [[buffer(11)]],
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
            if (lane == 0) {
                best[c] = score[k * C + c] + s;
                walked[k * C + c] = best[c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top;
        uint pick;
        if (mesh_draw<T>(top, pick, [&](uint c) { return best[c]; }, seed, state[1] + k, [&](uint c) { return uint(ids[cand[k * C + c]]); },
                         -INFINITY, 0, C, t, tv, tk)) {
            previous = cand[k * C + pick];
            drafts[k + 1] = uint(ids[previous]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Speculative speculative decoding's keys (metal-microbench docs/kernels.md#speculative-speculative-decoding; Kumar, Dao &
// May 2026, SSD and its tuned form Saguaro, arXiv:2603.03251, Alg. 1, Def 4 and Thm 12): after the walking member's block
// pass for the step's anchor (its slots' C candidates, cand and score, and the walk's scores, walked), each of the N
// cache slots an outcome of the step's verification, k of its drafts accepted (slots[i]) and the target's own token
// after them, keyed by slot k's candidate of rank ranks[i] (0 the likeliest) by its walked score (the base score at a slot
// the walk does not reach) plus the Gumbel noise the target's row k draws with (argmax.metal mesh_gumbel at the row's
// position p0 + k and the stream's seed), the token the step's row k + 1 holds left out (the target draws another where it
// rejects that row; Saguaro's sent-token exclusion): keys[i] its token (none where the slot has no such candidate),
// keys[N] the step count the cache was filled at (state[0], after the step's accept), and the slot's block state
// (`states`, `stride` words a slot): its anchor the key (row 0's token where none) at p0 + k + 1, the rest zero, for the
// fan-out's block passes after it (the next step's chains drafted while the verifying members verify this one).
template <uint R, uint C, uint N>
kernel void mesh_saguaro_keys(device const uint *state [[buffer(0)]], device const uint *cand [[buffer(1)]],
                              device const float *score [[buffer(2)]], device const float *walked [[buffer(3)]],
                              device const int *ids [[buffer(4)]], device const uint &seed [[buffer(5)]],
                              device const uint *slots [[buffer(6)]], device const uint *ranks [[buffer(7)]],
                              device uint *keys [[buffer(8)]], device uint *states [[buffer(9)]], constant uint &stride [[buffer(10)]],
                              uint i [[thread_position_in_grid]]) {
    if (i > N) return;
    if (i == N) {
        keys[N] = state[0];
        return;
    }
    const uint k = slots[i], f = ranks[i], p0 = state[1], walks = k + 1u < R;
    const uint sent = walks ? state[8u + k + 1u] : 0xffffffffu;
    float last = INFINITY;
    uint after = 0u, pick = 0xffffffffu;
    for (uint q = 0; q <= f; q++) {
        float top = -INFINITY;
        uint at = 0xffffffffu;
        for (uint c = 0; c < C; c++) {
            const uint e = k * C + c;
            if (score[e] == -INFINITY) continue;
            const uint token = uint(ids[cand[e]]);
            if (token == sent) continue;
            const float v = (walks ? walked[e] : score[e]) + mesh_gumbel(seed, p0 + k, token);
            if (q > 0u && (v > last || (v == last && c <= after))) continue;
            if (at == 0xffffffffu || v > top || (v == top && c < at)) { top = v; at = c; }
        }
        pick = at;
        if (at == 0xffffffffu) break;
        last = top; after = at;
    }
    const uint key = pick == 0xffffffffu ? 0xffffffffu : uint(ids[cand[k * C + pick]]);
    keys[i] = key;
    device uint *own = states + ulong(i) * stride;
    for (uint w = 0; w < stride; w++) own[w] = 0u;
    own[1] = p0 + k + 1u;
    own[8] = key == 0xffffffffu ? state[8] : key;
}

// A coded product's rows gathered from its drafting members (metal-microbench docs/kernels.md#the-drafter-as-a-share) placed
// at their natural rows: the export orders a product's rows by their energy, so each member writes its run of coded rows
// in code order, [rows][its run], the runs end to end in member order (`starts`: each run's first coded row, then the
// product's rows, `members` + 1 words), all-gathered into `runs`; coded row j of row r goes to y's row r at its natural
// row `order[j]`.  Thread (j, r).
kernel void mesh_draft_place(device half *y [[buffer(0)]], device const half *runs [[buffer(1)]], device const int *order [[buffer(2)]],
                             device const uint *starts [[buffer(3)]], constant uint &members [[buffer(4)]], uint2 at [[thread_position_in_grid]],
                             uint2 grid [[threads_per_grid]]) {
    const uint n = starts[members], j = at.x, r = at.y;
    if (j >= n) return;
    uint k = 0;
    while (k + 1 < members && j >= starts[k + 1]) k++;
    const uint a = starts[k], b = starts[k + 1];
    y[ulong(r) * n + uint(order[j])] = runs[ulong(grid.y) * a + ulong(r) * (b - a) + (j - a)];
}

// E2B's own trained MTP drafter (metal-microbench docs/kernels.md#the-drafter, E2B's MTP drafter: the LiteRT-LM export's
// tf_lite_mtp_drafter, signature mtp_drafter, its weights as stored by metal-microbench tools/draft/mtp_export.py;
// restored from metal-microbench e57247d and 65b05c6^ tools/prepare_e2b.py compose onto the engine's step): a chain of
// R - 1 passes at one position, each [the draft token's embedding, the hidden state it continues from] (2 D) projected
// to the drafter's width, four layers whose queries attend the target's caches of layers 13 and 14 (the engine's own
// decode attention over its pages, writing none), its own head over the target's tokens softcapped and drawn at
// temperature 1 with the noise of the verify row the draft is checked against, and its post-projection the next pass's
// hidden state.
//
// The chain's words (`mtp`, four): 0 the position its attention reads to (the step's p0 - 1: the cache holds the
// accepted positions below p0), 1 the position its queries rotate at (its anchor's p0 - 1: LiteRT-LM's drafter runs at
// the position before its first row, llm_litert_mtp_drafter.cc PrepareDrafterInputBuffers(position - 1)), 2 the stream's
// table row, 3 its anchor's position p0, the key of its first draft's draw (the verify row at p0 draws the token at
// p0 + 1 with it).  A step's chain (slot 0) continues from the verify's hidden state of the row that drew its anchor:
// row n of the previous step's verify (state word 5; its rows sat at p0 - n - 1 ..), the prefill's last row where n is 0
// on the first step (`activations`: the target's final-normed rows, metal-microbench dense_execution.swift output).  A
// speculation slot's chain (slot 1: Saguaro's next step from the outcome (k, key), drafted before this step's verify,
// its state at p0 + k + 1) continues from the step chain's projection after its pass k (`chain` row k + 1: the drafter's
// own stand-in for the target's hidden state at p0 + k) and attends the step's cache (positions below the step's p0;
// the rows the verify writes are not read): it costs acceptance, never correctness.  drafts[0] the chain's anchor.
template <uint D, uint R>
kernel void mesh_mtp_begin(device const uint *state [[buffer(0)]], device const uint *own [[buffer(1)]],
                           device const half *activations [[buffer(2)]], device const half *chain [[buffer(3)]],
                           device half *hidden [[buffer(4)]], device uint *mtp [[buffer(5)]], device uint *drafts [[buffer(6)]],
                           constant uint &slot [[buffer(7)]], uint c [[thread_position_in_grid]]) {
    const uint p0 = state[1], at = own[1];
    const uint row = slot == 0u ? min(state[5], R - 1u) : (at > p0 ? min(at - p0, R - 1u) : 1u);
    device const half *from = slot == 0u ? activations : chain;
    if (c < D) hidden[c] = from[ulong(row) * D + c];
    if (c == 0) {
        mtp[0] = max(p0, 1u) - 1u;
        mtp[1] = max(at, 1u) - 1u;
        mtp[2] = state[7];
        mtp[3] = at;
        drafts[0] = own[8];
    }
}

// A pass's input: the token's embedding row (the target's table times `scale`, the export's embedder's sqrt(D)), then
// the hidden state, 2 D halves.  The token tokens[pass]: the chain's drafts (row 0 its anchor), or a step's rows where the
// chain follows them (Saguaro's step chain: state words 8 ..).
template <uint D>
kernel void mesh_mtp_join(device const half *table [[buffer(0)]], device const uint *tokens [[buffer(1)]], device const half *hidden [[buffer(2)]],
                          device half *joined [[buffer(3)]], constant uint &pass [[buffer(4)]], constant float &scale [[buffer(5)]],
                          uint c [[thread_position_in_grid]]) {
    if (c >= D / 2) return;
    device half4 *out = (device half4 *)joined;
    out[c] = c < D / 4 ? half4(float4(((device const half4 *)table)[ulong(tokens[pass]) * (D / 4) + c]) * scale)
                       : ((device const half4 *)hidden)[c - D / 4];
}

// A pass's draw at temperature 1 (argmax.metal mesh_draw) over the head's rows [lo, hi) this member holds, row r the
// token order[r] where `ordered` (a coded head's row order: its outputs lie at their tokens) else r, each logit
// softcapped (cap tanh(z / cap)) and its noise keyed by the verify row the draft is checked against (word 3 plus the
// pass: the target's row at p0 + pass draws the token at p0 + pass + 1 with it), in the grid's blocks merged by the last
// to arrive (parts: the blocks' values, then their keys; arrival one word); the best (its value's bits, its token) into
// best[0], this member's candidate (a member's place among the vocabulary's holders: its offset).
template <uint T>
kernel void mesh_mtp_draw(device const half *logits [[buffer(0)]], device const int *order [[buffer(1)]], device const uint *mtp [[buffer(2)]],
                          device const uint &seed [[buffer(3)]], device uint2 *best [[buffer(4)]], coherent(device) device float *parts [[buffer(5)]],
                          device atomic_uint *arrival [[buffer(6)]], constant uint &lo [[buffer(7)]], constant uint &hi [[buffer(8)]],
                          constant float &cap [[buffer(9)]], constant uint &pass [[buffer(10)]], constant uint &ordered [[buffer(11)]],
                          uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tv[T / 32];
    threadgroup uint tk[T / 32];
    const uint span = (hi - lo + groups - 1) / groups, a = min(hi, lo + group * span), b = min(hi, a + span);
    auto token = [&](uint c) { return ordered != 0u ? uint(order[c]) : c; };
    float v;
    uint k;
    if (mesh_draw<T>(v, k, [&](uint c) { return streamed_capped(float(logits[token(c)]), cap); }, seed, mtp[3] + pass, token, -INFINITY,
                     a, b, lid, tv, tk, parts, (coherent(device) device uint *)(parts + groups), arrival, group, groups))
        best[0] = uint2(as_type<uint>(v), k == 0xffffffffu ? k : token(k));
}

// The pass's draft: the best of the `count` members' candidates (one; or the head's holders' after the vocabulary's
// select), in mesh_take's order, into drafts[pass + 1] below R (none drawn: the pass's own token again, a token the table
// holds).
template <uint R>
kernel void mesh_mtp_keep(device const uint2 *best [[buffer(0)]], device uint *drafts [[buffer(1)]], constant uint &count [[buffer(2)]],
                          constant uint &pass [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i != 0 || pass + 1u >= R) return;
    float v = -INFINITY;
    uint k = 0xffffffffu;
    for (uint m = 0; m < count; m++) mesh_take(v, k, best[m]);
    drafts[pass + 1u] = k == 0xffffffffu ? drafts[pass] : k;
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
#ifdef MESH_SPEC_ROWS
template [[host_name("mesh_spec_drafts_rows")]] [[kernel]] decltype(mesh_spec_drafts<MESH_SPEC_ROWS>) mesh_spec_drafts<MESH_SPEC_ROWS>;
#endif
#endif
