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
//   parts' merge (mesh_merge) and the store.
//
// Each instance is a literal kernel of its operand: the KV type's constants are its own, and the recorder makes the
// dispatch's sizes and fixed constants literals (metal-microbench docs/kernels.md#one-kernel-interface). Below it the
// parts' merge (mesh_merge, mesh_merge_row), the Tile plan (mesh_attention_tile over mesh_tile) and the shared prefix
// (mesh_shared: a range of positions several streams' rows share, attended once a task for rows of any of them, then
// each row's tail, merged as splits are).
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

// The parts' merge (Flash-Decoding's [Dao et al. 2023]; FlashInfer's state merge): the J parts of one row, each its maximum
// m_j, sum l_j and unnormalized accumulator o_j (P::m(j), P::l(j), P::o(j, d)), weighed e^(m_j - M) over the parts with
// l_j > 0. Each part's weight is held by lane j % 32 of every simdgroup (J <= 64) and shuffled, the sums and the total
// sum_j w_j l_j taken in part order (the sequential merge's arithmetic, to the bit: a lane tree's total moves LiteRT's
// int8 program, metal-microbench docs/kernels.md#decode-attention), one part-group; or with Shared computed by the first
// simdgroup into `weights` (J floats; the total, a lane tree's, and M into `scalars`, read by the store) and read from
// there after a barrier, the G part-groups (thread (d, g): channel slice d of DS) reading the accumulators 8 at a time and
// summing through `red` ((G - 1) DS float4s); group 0 calls store(M, sum_j w_j l_j, sum_j w_j o_j). Scheduled Single (a
// threadgroup a row: mesh_merge_row and the hosts' kernels) or LastArrival (the last of a row's parts to arrive:
// argmax.metal's mesh_last_arrival, the merge, then mesh_arrival_reset).
template <typename F, typename F4, uint DS>
struct mesh_split_parts {
    F ms;
    F ls;
    F4 os;
    ulong base;
    __attribute__((always_inline)) float m(uint j) { return ms[base + j]; }
    __attribute__((always_inline)) float l(uint j) { return ls[base + j]; }
    __attribute__((always_inline)) float4 o(uint j, uint d) { return os[(base + j) * DS + d]; }
};

__attribute__((always_inline)) static inline float4 mesh_merged(float total, float4 sum) { return total > 0.0f ? sum / total : float4(0.0f); }

template <uint DS, uint G, bool Shared = false, typename P, typename Store>
__attribute__((always_inline)) static inline void mesh_merge(P parts, uint J, uint d, uint g, uint lane, threadgroup float4 *red, Store store,
                                                            threadgroup float *weights = nullptr, threadgroup float *scalars = nullptr) {
    static_assert(DS % 32 == 0, "a simdgroup's lanes hold one part-group");
    static_assert(Shared || G == 1, "the lanes' weights are one part-group's");
    float top = -3.0e38f, w[2] = {0.0f, 0.0f}, total = 0.0f;
    if (!Shared || g * DS + d < 32) {
        for (uint p = lane; p < J; p += 32) if (parts.l(p) > 0.0f) top = max(top, parts.m(p));
        top = simd_max(top);
        if (Shared) {
            for (uint p = lane; p < J; p += 32) {
                const float le = parts.l(p), we = le > 0.0f ? exp(parts.m(p) - top) : 0.0f;
                weights[p] = we;
                total += we * le;
            }
            total = simd_sum(total);
            if (lane == 0) { scalars[0] = total; scalars[1] = top; }
        } else
            for (uint k = 0; k < 2; ++k) {
                const uint p = lane + 32 * k;
                w[k] = p < J && parts.l(p) > 0.0f ? exp(parts.m(p) - top) : 0.0f;
            }
    }
    if (Shared) threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 sum = float4(0.0f);
    for (uint p = g; p < J; p += 8 * G) {
        float4 y[8];
        for (uint u = 0; u < 8; ++u) y[u] = p + u * G < J ? parts.o(p + u * G, d) : float4(0.0f);
        for (uint u = 0; u < 8; ++u) {
            const uint q = p + u * G;
            const float wq = q < J ? (Shared ? weights[q] : simd_shuffle(w[q / 32], ushort(q % 32))) : 0.0f;
            sum += wq * y[u];
            if (!Shared && q < J) total += wq * parts.l(q);
        }
    }
    if (G > 1) {
        if (g > 0) red[(g - 1) * DS + d] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0) for (uint e = 1; e < G; ++e) sum += red[(e - 1) * DS + d];
    }
    if (g == 0) store(Shared ? scalars[1] : top, Shared ? scalars[0] : total, sum);
}

// The gathered store (metal-microbench docs/kernels.md#decode-attention, the epilogue): a thread's four outputs from `out`
// at their places in o's gathered input (O[n columns + inverse[c]] = scale[inverse[c]] value) where `columns` is nonzero,
// else in place. The places are 32-bit element indices (an operand of fewer than 2^32 elements; out and columns multiples
// of 4, so one column c for the four), held from the task's start to its store.
__attribute__((always_inline)) static inline void mesh_gathered_places(uint out, device const int *inverse, device const float *scale, uint columns,
                                                                      thread uint4 &dst, thread float4 &cs, uint stride) {
    dst = uint4(out) + uint4(0, 1, 2, 3);
    cs = float4(1.0f);
    if (columns != 0) {
        const uint c = out % columns;
        for (uint i = 0; i < 4; ++i) {
            const uint to = uint(inverse[c + i]);
            dst[i] = (out / columns) * stride + to;
            cs[i] = scale[to];
        }
    }
}

__attribute__((always_inline)) static inline void mesh_gathered_store(device half *O, uint4 dst, float4 scale, uint columns, float4 value) {
    if (columns == 0) { *(device half4 *)(O + dst.x) = half4(value); return; }
    for (uint i = 0; i < 4; ++i) O[dst[i]] = half(scale[i] * value[i]);
}

// Single, a threadgroup of D / 4 threads a row (thread d its float4 d): a row's J parts (P, mesh_split_parts) merged into
// row i of O, gathered (mesh_gathered_store: o's input, `columns` its width) or plain (columns 0). A host's kernel calls it
// (no kernel is defined here: a source including this file stays one kernel's).
template <uint D, typename P>
__attribute__((always_inline)) static inline void mesh_merge_row(P parts, uint i, uint J, uint d, uint lane, device half *O, device const int *inverse,
                                                                device const float *scale, uint columns, uint stride) {
    uint4 dst;
    float4 cs;
    mesh_gathered_places(i * D + 4 * d, inverse, scale, columns, dst, cs, stride);
    mesh_merge<D / 4, 1>(parts, J, d, 0u, lane, nullptr, [&](float, float total, float4 sum) { mesh_gathered_store(O, dst, cs, columns, mesh_merged(total, sum)); });
}

// The Tile plan (metal-microbench docs/kernels.md#decode-attention, the verify plan; FlashInfer's tensor-core decode for
// grouped queries, llama.cpp's kernel_flash_attn_ext): a task is 8 query rows of one KV head (a stream's rows times the
// query heads that read it) over part j of the KV range, its scores and value products 8 x 8 simdgroup matrices; its one
// operand is mesh_tile (below) over a KV type. Each
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
//   load_k/load_v(matrix, page_address, chunk, page, row, column): the task's compiled storage layout.
template <typename A>
__attribute__((always_inline)) static inline void mesh_attention_tile(thread A &a) {
    constexpr uint D = A::dims, NSG = A::simdgroups, PAGE = A::page_extent, C = NSG * PAGE, DB = D / 8, OB = DB / NSG, RS = 8 / NSG > 0 ? 8 / NSG : 1;
    static_assert(PAGE == MESH_ATTENTION_TILE_PAGE, "attention kernel and declared KV page layout disagree");
    const uint s = a.s, lane = a.lane;
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
                    a.load_k(k[2 * u], kp, i, s, 0u, 8 * (d0 + u));
                    a.load_k(k[2 * u + 1], kp, i, s, 8u, 8 * (d0 + u));
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
                for (uint b = 0; b < OB; b++) a.load_v(vm[b], v, i, w, 8 * h, 8 * (b * NSG + s));
                for (uint b = 0; b < OB; b++) simdgroup_multiply_accumulate(o[b], p, vm[b], o[b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    a.finish(o, m, l);
}

// The keys a query row attends, an interval [lo, hi) of logical positions (FlexAttention's mask over one index; metal-microbench
// docs/kernels.md#decode-attention, the tile operand): k <= q (Causal), or both q and k in a block [b0, b1) of rows that
// attend each other both ways (Block, a soft item's rows), k < len (End), and with a window W > 0, k + W > q (Window). A
// sliding layer's rows are (Causal or Block) and Window and End; a global layer's Causal and End (b0 = b1).
__attribute__((always_inline)) static inline uint2 mesh_keys(uint q, uint len, uint window, uint b0, uint b1) {
    const uint upper = min(len, q >= b0 && q < b1 ? max(q + 1, b1) : q + 1), lower = window > 0 && q + 1 > window ? q + 1 - window : 0u;
    return uint2(min(lower, upper), upper);
}

// A mask mod (metal-microbench docs/kernels.md#mask-mods) whose rows' [lo, hi) are exactly the keys they attend: keep() is
// never asked (exact), so the bound alone masks.
struct mesh_unmasked {
    static constexpr constant bool exact = true;
    __attribute__((always_inline)) bool keep(uint, uint, uint) const { return true; }
};

// The Tile plan's operand over a KV type KV (KV::at(p, k, v): logical page p's K and V line addresses of the task's KV head,
// false where it holds none): 8 query rows, row r Q's row rows[r] (~0u for none) attending [lo[r], hi[r]) and of those the
// keys its mask mod M keeps (M::exact: all of them; else mod.keep(query[r].x, query[r].y, key) of its batch row and query
// head); part j of J over the pages [base, base + pages) of the rows' union, each part's (m, l, O) at Q's row times `parts`
// plus j, through F and F4 (coherent(device) pointers where the last of a row's parts to arrive merges them in the same
// dispatch). Access supplies query and KV matrix loads; its default is the cached row layout.
// layout() places the threadgroup memory a wrapper declares (region floats, 64 diag floats, simdgroups places)
// and span() the pages of the rows' union and J, the configuration's part count (each part written: an empty one's l = 0,
// which the merge passes over).
struct mesh_attention_access {
    __attribute__((always_inline)) half4 query(device const half *q, ulong at) const { return ((device const half4 *)q)[at]; }
    __attribute__((always_inline)) void load(thread simdgroup_half8x8 &out, device const half *p, uint, uint row, uint column,
                                            uint stride, uint, bool) const {
        simdgroup_load(out, p + ulong(row) * stride + column, stride);
    }
};

inline uint mesh_matrix_lane(uint lane) {
    const uint q = lane / 4;
    return ((q & 4) + (lane / 2) % 4) * 8 + (q & 2) * 2 + (lane % 2) * 2;
}

template <uint D, typename KV, typename F = device float *, typename F4 = device float4 *, typename M = mesh_unmasked,
          typename Access = mesh_attention_access>
struct mesh_tile {
    enum : uint { dims = D, page_extent = KV::page_extent, simdgroups = 4, region = 8 * D > 4 * D + 12 * 64 ? 8 * D : 4 * D + 12 * 64 };
    KV kv;
    device const half *Q;
    F mp;
    F lp;
    F4 op;
    threadgroup half *qs;
    threadgroup float *ss;
    threadgroup half *ps;
    threadgroup float *diag;
    threadgroup ulong *places;
    threadgroup float *stage;
    float scale;
    uint stride, s, lane, t, J, j, base, pages, parts;
    uint rows[8], lo[8], hi[8];
    M mod;
    uint2 query[8];
    Access access;

    __attribute__((always_inline)) void layout(threadgroup float *region, threadgroup float *d, threadgroup ulong *p) {
        qs = (threadgroup half *)region; ss = region + 4 * D; ps = (threadgroup half *)(region + 4 * D + 8 * 64);
        diag = d; places = p; stage = region;
    }

    __attribute__((always_inline)) void span(uint splits) {
        uint low = 0xFFFFFFFFu, high = 0u;
        for (uint r = 0; r < 8; ++r)
            if (rows[r] != 0xFFFFFFFFu && hi[r] > lo[r]) { low = min(low, lo[r]); high = max(high, hi[r]); }
        if (high == 0u) low = 0u;
        base = low / page_extent;
        pages = (high + page_extent - 1) / page_extent - base;
        J = splits;
    }

    __attribute__((always_inline)) void load_q() {
        for (uint e = t; e < 2 * D; e += 32 * simdgroups) {
            const uint i = rows[e / (D / 4)];
            ((threadgroup half4 *)qs)[e] = i != 0xFFFFFFFFu ? access.query(Q, ulong(i) * (D / 4) + e % (D / 4)) : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    __attribute__((always_inline)) bool chunk(uint i) { return j + J * simdgroups * i < pages; }

    __attribute__((always_inline)) bool page(uint i, uint w, thread ulong &k, thread ulong &v) {
        const uint p = j + J * (simdgroups * i + w);
        return p < pages && kv.at(base + p, k, v);
    }

    __attribute__((always_inline)) uint first(uint i, uint w) { return (base + j + J * (simdgroups * i + w)) * page_extent; }

    __attribute__((always_inline)) void load_k(thread simdgroup_half8x8 &out, device const half *p, uint i, uint w, uint row, uint column) {
        access.load(out, p, first(i, w), row, column, stride, lane, true);
    }
    __attribute__((always_inline)) void load_v(thread simdgroup_half8x8 &out, device const half *p, uint i, uint w, uint row, uint column) {
        access.load(out, p, first(i, w), row, column, stride, lane, false);
    }

    __attribute__((always_inline)) bool live(uint r, uint position) {
        return position >= lo[r] && position < hi[r] && (M::exact || mod.keep(query[r].x, query[r].y, position));
    }

    __attribute__((always_inline)) void finish(thread simdgroup_float8x8 *o, thread const float *m, thread const float *l) {
        constexpr uint NSG = simdgroups, OB = D / 8 / NSG, RS = 8 / NSG, D4 = D / 4;
        for (uint b = 0; b < OB; b++) simdgroup_store(o[b], stage + 8 * (b * NSG + s), D);
        if (lane == 0)
            for (uint k = 0; k < RS; k++) {
                const uint i = rows[s * RS + k];
                if (i == 0xFFFFFFFFu) continue;
                mp[ulong(i) * parts + j] = l[k] > 0.0f ? m[k] : -INFINITY;
                lp[ulong(i) * parts + j] = l[k];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = t; e < 2 * D; e += 32 * NSG) {
            const uint i = rows[e / D4];
            if (i != 0xFFFFFFFFu) op[(ulong(i) * parts + j) * D4 + e % D4] = ((threadgroup const float4 *)stage)[e];
        }
    }
};

// The shared prefix (restored from metal-microbench 845838e^ kernels.swift paged_attn_slide_ar_shared and
// paged_attn_full_ar_shared, bootstrap.swift encPagedAttn{Slide,Full}ArSharedAndTail; metal-microbench
// docs/kernels.md#decode-attention): rows of several streams whose first positions [0, end) lie in the same physical pages
// (a prefix the streams adopted: the same pages in each one's block table) attend that range as one part of their
// partials and their own positions past it as another, each part the Tile plan (mesh_attention_tile over mesh_tile), and
// a row's parts merge as any split's (mesh_merge_row over its parts, the counted merge). The operand is the range: its
// end (a position, 16 times its pages) and the shared part's part count `splits`.
//
//   The shared part (845838e^'s shared kernel, split 0): a task is 8 of the rows sharing the range, of any streams and
//   query heads of one KV head (the old kernel's 4 slots times 2 heads), over the range's pages through a KV type that
//   maps them (any sharing row's block table), each row's keys [lo, hi) cut to [lo, min(hi, end)) (mesh_shared_part);
//   every one of its `splits` parts written (an empty one l = 0, which the merge passes over), at slots [0, splits) of
//   each row's parts. Each shared page's K and V lines are read once a task for all its rows: the range
//   ceil(R / 8) times a KV head for the R rows sharing it, where the per-row path reads it once in each stream's task
//   (ceil(S HG / 8) for a stream's S rows of HG query heads), so it reads less where a stream holds fewer than 8 rows of
//   a KV head (a member holding few of a KV head's query heads; a model of fewer query heads a KV head).
//   The tail (845838e^'s tail kernel, split 1): a sharing stream's task as the per-row path makes it, its rows' keys cut
//   to [max(lo, end), max(hi, end)), its parts at slots [splits, splits + J) (the partial pointers offset by `splits`
//   parts), J the tail's part count (mesh_shared_tail returns the row's, splits + J, the counted merge's). A stream
//   outside the sharing rows keeps the per-row path whole (its parts from slot 0).
//   The merge (845838e^'s paged_attn_split_reduce over 2 splits): mesh_merge_row over a row's splits + J parts, after both
//   parts' dispatches (a shared task holds several streams' rows, so no part of a row knows itself the last to arrive).
struct mesh_shared {
    uint end, splits;
    __attribute__((always_inline)) uint2 shared(uint2 keys) const { return uint2(min(keys.x, end), min(keys.y, end)); }
    __attribute__((always_inline)) uint2 tail(uint2 keys) const { return uint2(max(keys.x, end), max(keys.y, end)); }
};

// The shared part's task `item`: rows [8 item, 8 item + 8) of the `count` rows sharing the range, row e's Q row and keys
// (over all its positions, its mask mod's) given by row(e, query) as uint3(Q row, lo, hi) with its (batch row, query head)
// in query, each cut to the range; the pages of
// their union, every one of the range's `splits` parts. The task's other members are the caller's
// (layout, scale, stride, s, lane, t, j, parts), its KV type mapping the range's pages.
template <uint D, typename KV, typename F, typename F4, typename M, typename Access, typename Row>
__attribute__((always_inline)) static inline void mesh_shared_part(thread mesh_tile<D, KV, F, F4, M, Access> &task, mesh_shared range, uint item, uint count,
                                                                   Row row) {
    for (uint r = 0; r < 8; ++r) {
        const uint e = 8 * item + r;
        uint2 query = uint2(0u);
        const uint3 made = e < count ? row(e, query) : uint3(0xFFFFFFFFu, 0u, 0u);
        const uint2 keys = range.shared(made.yz);
        task.rows[r] = made.x;
        task.lo[r] = keys.x;
        task.hi[r] = keys.y;
        task.query[r] = query;
    }
    task.span(range.splits);
}

// A sharing stream's tail on its task (rows and keys as the per-row path fills them): each row's keys cut to past the
// range, its parts at slots [range.splits, range.splits + J) (the partial pointers offset), J = `splits`; the row's part
// count range.splits + J, the merge's.
template <uint D, typename KV, typename F, typename F4, typename M, typename Access>
__attribute__((always_inline)) static inline uint mesh_shared_tail(thread mesh_tile<D, KV, F, F4, M, Access> &task, mesh_shared range, uint splits) {
    for (uint r = 0; r < 8; ++r) {
        const uint2 keys = range.tail(uint2(task.lo[r], task.hi[r]));
        task.lo[r] = keys.x;
        task.hi[r] = keys.y;
    }
    task.mp += range.splits;
    task.lp += range.splits;
    task.op += ulong(range.splits) * (D / 4);
    task.span(splits);
    return range.splits + splits;
}
#endif
