// The selection (metal-microbench docs/kernels.md#speculative-decoding, sampling), each computation once: a row's candidates
// c, each with its value z(c) and its token id(c), and
// - the draw at temperature 1 (mesh_draw): the greatest z(c) + g(seed, key, id(c)), g the Gumbel noise (mesh_gumbel);
// - the normaliser (mesh_normaliser): (m, l) = (the greatest z, the sum of exp(z - m)), no token; log Z = m + log l, and
//   min-p's cutoff is m + log min_p;
// - the retained set (mesh_retain): a head, the greatest z, and a tail, the greatest z + g outside the head, written
//   (id, z) beside the row's (m, l).
// A row's split reduction finishes each (mesh_argmax: the simdgroups by shuffles, the threadgroup through threadgroup
// memory, the blocks by the last threadgroup to arrive, mesh_last_arrival, then mesh_arrival_reset), its merge mesh_take.
// Metal callers prepend this header (metal-microbench's engine prelude, drafter.metal, LiteRT's draw stage, the take of
// gathered (z + g, token) pairs); host tools reach it through argmax.py, whose kernels are mesh_draw_rows and mesh_retain_rows.
#ifndef TORCH_MESH_ARGMAX
#define TORCH_MESH_ARGMAX
#include <metal_stdlib>
using namespace metal;
template<uint B> inline long mesh_index(ulong i);
constant constexpr uint MESH_SELECTION_BINS = 1u << MESH_SELECTION_RADIX_BITS;
template <uint T> constant constexpr uint mesh_selection_words = MESH_SELECTION_BINS + MESH_SELECTION_STATE_WORDS + T / MESH_SIMD_WIDTH;

// The draw's noise: the greatest z_v + g(seed, position, v) is a sample of softmax(z) [Gumbel 1954; the Gumbel-max trick,
// Maddison, Tarlow & Minka 2014], and the greatest k a sample of k without replacement [Kool, van Hoof & Welling 2019];
// g the Gumbel noise of a hash of the three words (murmur3's finalizer, a bijection of 32 bits, folded over them) to a
// uniform u in (0, 1) of 23 bits (its greatest, 1 - 2^-24, an fp32 value below 1; 24 bits had rounded it to 1 and g to
// +inf), g = -log(-log u). Keyed by the position a row predicts from, a row computed again draws the same token.
__attribute__((always_inline)) static inline uint mesh_mix(uint h) {
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h;
}

__attribute__((always_inline)) static inline float mesh_gumbel(uint seed, uint position, uint token) {
    const uint h = mesh_mix(token + mesh_mix(position + mesh_mix(seed)));
    const float u = (float(h >> 9) + 0.5f) * (1.0f / 8388608.0f);
    return -precise::log(-precise::log(u));
}

// The candidate order: the greater value, then the lower key.
__attribute__((always_inline)) static inline void mesh_take(thread float &v, thread uint &k, float ov, uint ok) {
    if (ov > v || (ov == v && ok < k)) { v = ov; k = ok; }
}

// A candidate packed as a crossing carries it: (its value's bits, its key).
__attribute__((always_inline)) static inline void mesh_take(thread float &v, thread uint &k, uint2 packed) {
    mesh_take(v, k, as_type<float>(packed.x), packed.y);
}

// The normaliser's merge [Milakov & Gimelshein 2018, online softmax]: (m, l) and (om, ol) into (n, l e^(m - n) + ol e^(om - n)),
// n the greater m (-FLT_MAX at least, so no difference of infinities); a value z is (z, 1).
__attribute__((always_inline)) static inline void mesh_take(thread float &m, thread float &l, float om, float ol) {
    const float n = max(max(m, om), -FLT_MAX);
    l = l * exp(m - n) + ol * exp(om - n);
    m = n;
}

// LastArrival, a split reduction's finish by the last of its J threadgroups to arrive (mesh_argmax's blocks,
// attention.metal's parts' merge, metal-microbench's row epilogues, metal/rows.metal, and Neural Engine shares): each
// fences its stores (at the device's scope, or with System at the system's: a reader off the GPU) and counts its arrival
// in word `slot`; true, after a fence, in every thread of the last to arrive where `last` (a threadgroup word) is given,
// else (nullptr) in its thread 0 alone. The last makes the word 0 again for the dispatch's next run once its merge is
// done (mesh_arrival_reset: metal-microbench docs/kernels.md, counts left by a cut dispatch), off the merge's path.
template <bool System = false>
__attribute__((always_inline)) static inline void mesh_arrival_fence() {
    if constexpr (System) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
    else atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

template <bool System = false>
__attribute__((always_inline)) static inline bool mesh_last_arrival(device atomic_uint *arrivals, uint slot, uint J, uint t,
                                                                   threadgroup uint *last, uint contribution = 1u) {
    mesh_arrival_fence<System>();
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool mine = t == 0 && atomic_fetch_add_explicit(&arrivals[slot], contribution, memory_order_relaxed) == J - contribution;
    if (last) {
        if (t == 0) *last = mine ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mine = *last != 0u;
    }
    if (!mine) return false;
    mesh_arrival_fence<System>();
    return true;
}

__attribute__((always_inline)) static inline void mesh_arrival_reset(device atomic_uint *arrivals, uint slot, uint t) {
    if (t == 0) atomic_store_explicit(&arrivals[slot], 0u, memory_order_relaxed);
}

// A row's split reduction under mesh_take (a candidate (v, k) to the greatest, a normaliser (m, l) to the row's): the
// threadgroup's into thread 0's (with All, into every thread's; tv, tk T / MESH_SIMD_WIDTH each), and with `blocks` > 1 every block's
// into the last arriver's thread 0 (slot: row * blocks + block of the records pv, pk; arrival: the row's word). True for
// the threads holding the row's.
template <uint T, bool All = false, typename V, typename K>
__attribute__((always_inline)) static inline bool mesh_argmax(thread V &v, thread K &k, uint lid, threadgroup V *tv, threadgroup K *tk,
                                                              coherent(device) device V *pv = nullptr, coherent(device) device K *pk = nullptr,
                                                              device atomic_uint *arrival = nullptr, uint slot = 0, uint blocks = 1) {
    for (ushort d = MESH_SIMD_WIDTH / 2; d > 0; d /= 2) mesh_take(v, k, simd_shuffle_down(v, d), simd_shuffle_down(k, d));
    if (T > MESH_SIMD_WIDTH) {
        if (lid % MESH_SIMD_WIDTH == 0) { tv[lid / MESH_SIMD_WIDTH] = v; tk[lid / MESH_SIMD_WIDTH] = k; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) for (uint q = 1; q < T / MESH_SIMD_WIDTH; q++) mesh_take(v, k, tv[q], tk[q]);
    }
    if (All) {
        if (T > MESH_SIMD_WIDTH) {
            if (lid == 0) { tv[0] = v; tk[0] = k; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            v = tv[0]; k = tk[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else { v = simd_broadcast(v, 0); k = simd_broadcast(k, 0); }
        return true;
    }
    if (blocks <= 1) return lid == 0;
    if (lid == 0) { pv[slot] = v; pk[slot] = k; }
    if (!mesh_last_arrival(arrival, 0u, blocks, lid, nullptr)) return false;
    const uint row = slot - slot % blocks;
    v = pv[row]; k = pk[row];
    for (uint b = 1; b < blocks; b++) mesh_take(v, k, pv[row + b], pk[row + b]);
    mesh_arrival_reset(arrival, 0u, lid);
    return true;
}

// The draw at temperature 1 over a row's candidates c in [lo, hi): the greatest z(c) + g(seed, key, id(c)), id(c) the
// candidate's token (an engine row's first + c, the drafter's ids[cand[c]]), a candidate whose z is below `cutoff`
// (min-p's, else -inf) never taken; k the candidate (c), the blocks as mesh_argmax's.
template <uint T, typename Z, typename Id>
__attribute__((always_inline)) static inline bool mesh_draw(thread float &v, thread uint &k, Z z, uint seed, uint key, Id id, float cutoff,
                                                            uint lo, uint hi, uint lid, threadgroup float *tv, threadgroup uint *tk,
                                                            coherent(device) device float *pv = nullptr, coherent(device) device uint *pk = nullptr,
                                                            device atomic_uint *arrival = nullptr, uint slot = 0, uint blocks = 1) {
    v = -INFINITY; k = 0xffffffffu;
    for (uint c = lo + lid; c < hi; c += T) {
        const float x = z(c);
        mesh_take(v, k, x < cutoff ? -INFINITY : x + mesh_gumbel(seed, key, id(c)), c);
    }
    return mesh_argmax<T>(v, k, lid, tv, tk, pv, pk, arrival, slot, blocks);
}

// The normaliser of a row's candidates [lo, hi), (m, l), the blocks as mesh_argmax's.
template <uint T, typename Z>
__attribute__((always_inline)) static inline bool mesh_normaliser(thread float &m, thread float &l, Z z, uint lo, uint hi, uint lid,
                                                                  threadgroup float *tm, threadgroup float *tl,
                                                                  coherent(device) device float *pm = nullptr, coherent(device) device float *pl = nullptr,
                                                                  device atomic_uint *arrival = nullptr, uint slot = 0, uint blocks = 1) {
    m = -FLT_MAX; l = 0.0f;
    for (uint c = lo + lid; c < hi; c += T) mesh_take(m, l, z(c), 1.0f);
    return mesh_argmax<T>(m, l, lid, tm, tl, pm, pl, arrival, slot, blocks);
}

// A finite value's place in the order of values as an unsigned word, 0 for -inf, +inf and NaN (integer comparisons,
// which fast math keeps).
__attribute__((always_inline)) static inline uint mesh_order(float x) {
    const uint u = as_type<uint>(x), o = (u & 0x80000000u) ? ~u : u | 0x80000000u;
    return o - 0x00800000u < 0xff000000u ? o : 0u;
}

// A threadgroup prefix and total, shared by selection and tensor scans.
// Metal shuffles words; a wide integer carries both words through the same lanes.
template <typename V> inline V mesh_scan_prefix(V x, uint lid) { return simd_prefix_inclusive_sum(x); }
inline ulong mesh_scan_prefix(ulong x, uint lid) {
    for (ushort d = 1; d < MESH_SIMD_WIDTH; d *= 2) {
        const uint2 words = as_type<uint2>(x);
        const ulong other = as_type<ulong>(uint2(simd_shuffle_up(words.x, d), simd_shuffle_up(words.y, d)));
        if (lid % MESH_SIMD_WIDTH >= d) x += other;
    }
    return x;
}

template <uint T, typename V>
inline V mesh_prefix_sum(V x, uint lid, threadgroup V *sums, thread V &total) {
    x = mesh_scan_prefix(x, lid);
    if (lid % MESH_SIMD_WIDTH == MESH_SIMD_WIDTH - 1) sums[lid / MESH_SIMD_WIDTH] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    V before = 0;
    total = 0;
    for (uint q = 0; q < T / MESH_SIMD_WIDTH; q++) {
        before += q < lid / MESH_SIMD_WIDTH ? sums[q] : V(0);
        total += sums[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return before + x;
}

// The n greatest of candidates [lo, hi) by integer key(c), with zero excluded, in one threadgroup:
// radix selection of the n-th key's order word, a byte a pass over a 256-bin histogram, then the candidates in ascending
// order, each thread a contiguous span, put(i, c) at their places (put(i, 0xffffffff) past the last when fewer than n are
// eligible) [PyTorch topk: radix select, then gather]. Returns (o, c): the set is the candidates whose (order word, c) is
// (o, c) or precedes it in that order. w: mesh_selection_words<T> words.
template <uint T, typename K, typename P>
__attribute__((always_inline)) static inline uint2 mesh_top_ordered(K key, uint lo, uint hi, uint n, uint lid, threadgroup uint *w, P put) {
    if (n == 0) return uint2(0xffffffffu, 0u);
    threadgroup atomic_uint *bins = (threadgroup atomic_uint *)w;
    threadgroup uint *said = w + MESH_SELECTION_BINS, *sums = w + MESH_SELECTION_BINS + MESH_SELECTION_STATE_WORDS;
    uint prefix = 0, need = n, mask = 0;
    for (int shift = sizeof(uint) * 8 - MESH_SELECTION_RADIX_BITS; shift >= 0 && need != 0xffffffffu; shift -= MESH_SELECTION_RADIX_BITS) {
        for (uint d = lid; d < MESH_SELECTION_BINS; d += T) atomic_store_explicit(&bins[d], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = lo + lid; c < hi; c += T) {
            const uint o = key(c);
            if (o != 0u && (o & mask) == prefix) atomic_fetch_add_explicit(&bins[(o >> shift) & (MESH_SELECTION_BINS - 1)], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            uint d = MESH_SELECTION_BINS, above = 0;
            for (; d > 0; d--) {
                const uint h = atomic_load_explicit(&bins[d - 1], memory_order_relaxed);
                if (above + h >= need) break;
                above += h;
            }
            said[0] = d == 0 ? 0u : prefix | (d - 1) << shift;
            said[1] = d == 0 ? 0xffffffffu : need - above;
            said[2] = 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = said[0]; need = said[1]; mask |= (MESH_SELECTION_BINS - 1) << shift;
    }
    auto scan = [&](uint x, thread uint &total) {
        return mesh_prefix_sum<T>(x, lid, sums, total) - x;
    };
    const uint span = (hi - lo + T - 1) / T, a = min(lo + lid * span, hi), b = min(a + span, hi);
    uint above = 0, ties = 0, greater, tied;
    for (uint c = a; c < b; c++) {
        const uint o = key(c);
        above += o > prefix;
        ties += o == prefix && o != 0u;
    }
    uint at = scan(above, greater), t = scan(ties, tied);
    at += min(t, need);
    for (uint c = a; c < b; c++) {
        const uint o = key(c);
        if (o == 0u || o < prefix || (o == prefix && t >= need)) continue;
        if (o == prefix && ++t == need) said[2] = c;
        put(at++, c);
    }
    for (uint i = greater + min(tied, need) + lid; i < n; i += T) put(i, 0xffffffffu);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return uint2(prefix, said[2]);
}

// Float selection retains its finite-only sampling contract. Other tensor
// operations supply their own integer order to the same radix selection.
template <uint T, typename K, typename P>
__attribute__((always_inline)) static inline uint2 mesh_top(K key, uint lo, uint hi, uint n, uint lid, threadgroup uint *w, P put) {
    return mesh_top_ordered<T>([&](uint c) { return mesh_order(key(c)); }, lo, hi, n, lid, w, put);
}

// The retained set of a row's candidates [lo, hi) in one threadgroup: ids[i], zs[i] the head, the `head` greatest z, then
// the tail, the `tail` greatest z + g(seed, key, id) outside the head (the Gumbel-top-k with the draw's own noise, so two
// models' tails at a position coincide where their distributions do), each in ascending candidate order, so no place
// names the greatest; an absent entry (fewer finite z) is (-1, -inf). True in thread 0, holding the row's (m, l).
// w: mesh_selection_words<T> words.
template <uint T, typename Z, typename Id, typename Put>
__attribute__((always_inline)) static inline bool mesh_retain(thread float &m, thread float &l, Z z, uint seed, uint key, Id id, uint head,
                                                              uint tail, uint lo, uint hi, uint lid, threadgroup uint *w, Put write) {
    auto put = [&](uint i, uint c) { write(i, c == 0xffffffffu ? -1l : long(id(c)), c == 0xffffffffu ? -INFINITY : z(c)); };
    const uint2 top = mesh_top<T>(z, lo, hi, head, lid, w, put);
    mesh_top<T>([&](uint c) {
        const float x = z(c);
        const uint o = mesh_order(x);
        return o > top.x || (o == top.x && c <= top.y) ? -INFINITY : x + mesh_gumbel(seed, key, id(c));
    }, lo, hi, tail, lid, w, [&](uint i, uint c) { put(head + i, c); });
    return mesh_normaliser<T>(m, l, z, lo, hi, lid, (threadgroup float *)w, (threadgroup float *)w + T / MESH_SIMD_WIDTH);
}

// argmax.py's draw: row group.y of z [rows, v] in blocks (group.x), each row's token first + c (words: v, seed, first,
// blocks; parts: the blocks' records, values then keys).
template <uint T>
kernel void mesh_draw_rows(device const float *z [[buffer(0)]], device const uint *keys [[buffer(1)]], device const float *cutoff [[buffer(2)]],
                           device long *ids [[buffer(3)]], coherent(device) device float *parts [[buffer(4)]],
                           device atomic_uint *arrivals [[buffer(5)]], constant uint *words [[buffer(6)]],
                           uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tv[T / MESH_SIMD_WIDTH];
    threadgroup uint tk[T / MESH_SIMD_WIDTH];
    const uint v = words[0], first = words[2], blocks = words[3], r = group.y, span = (v + blocks - 1) / blocks,
               lo = min(v, group.x * span), hi = min(v, lo + span);
    float best;
    uint c;
    if (mesh_draw<T>(best, c, [&](uint i) { return z[mesh_index<0>(ulong(r) * v + i)]; }, words[1], keys[mesh_index<1>(r)],
                     [&](uint i) { return first + i; }, cutoff[mesh_index<2>(r)],
                     lo, hi, lid, tv, tk, parts, (coherent(device) device uint *)(parts + groups.y * blocks), &arrivals[r],
                     r * blocks + group.x, blocks))
        ids[mesh_index<3>(r)] = long(first + c);
}

// argmax.py's retain: row group.y of z [rows, v] in one threadgroup, its H + M entries and log Z (words: v, seed).
template <uint T, uint H, uint M>
kernel void mesh_retain_rows(device const float *z [[buffer(0)]], device const uint *positions [[buffer(1)]], device long *ids [[buffer(2)]],
                             device float *kept [[buffer(3)]], device float *log_z [[buffer(4)]], constant uint *words [[buffer(5)]],
                             uint2 group [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint w[mesh_selection_words<T>];
    const uint v = words[0], r = group.y;
    float m, l;
    if (mesh_retain<T>(m, l, [&](uint c) { return z[mesh_index<0>(ulong(r) * v + c)]; }, words[1], positions[mesh_index<1>(r)],
                       [&](uint c) { return c; }, H, M, 0, v, lid,
                       w, [&](uint i, long token, float value) {
                           ids[mesh_index<2>(ulong(r) * (H + M) + i)] = token;
                           kept[mesh_index<3>(ulong(r) * (H + M) + i)] = value;
                       }))
        log_z[mesh_index<4>(r)] = m + log(l);
}
#endif
