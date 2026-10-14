// The greatest of each row's candidates in one total order, greater value first and then the lower key (metal-microbench
// docs/kernels.md#one-kernel-interface, argmax): a row's range split over B threadgroups of T threads, each thread's
// candidates the caller's scan, the simdgroups' met by shuffles, the threadgroup's through threadgroup memory, and the
// blocks' by the last threadgroup to arrive (mesh_last_arrival, then mesh_arrival_reset). V and K are one candidate (float, uint) or
// four independent channels (float4, uint4). metal-microbench's engine instantiates it over a row of logits
// (argmax_partial) and its recorder over LiteRT's ARG_MAX channels (native_argmax.metal).
#ifndef TORCH_MESH_ARGMAX
#define TORCH_MESH_ARGMAX
#include <metal_stdlib>
using namespace metal;

// Sampling as an argmax (metal-microbench docs/kernels.md#speculative-decoding, sampling): argmax_v z_v / T + g(seed,
// position, v) is a sample of softmax(z / T) [Gumbel 1954; the Gumbel-max trick, Maddison, Tarlow & Minka 2014], g the
// Gumbel noise of a hash of the three words (murmur3's finalizer, a bijection of 32 bits, folded over them) to a uniform
// u in (0, 1) of 23 bits (its greatest, 1 - 2^-24, an fp32 value below 1; 24 bits had rounded it to 1 and g to +inf),
// g = -log(-log u). Keyed by the position a row predicts from, a row computed again draws the same token.
__attribute__((always_inline)) static inline uint mesh_mix(uint h) {
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h;
}

__attribute__((always_inline)) static inline float mesh_gumbel(uint seed, uint position, uint token) {
    const uint h = mesh_mix(token + mesh_mix(position + mesh_mix(seed)));
    const float u = (float(h >> 9) + 0.5f) * (1.0f / 8388608.0f);
    return -precise::log(-precise::log(u));
}

__attribute__((always_inline)) static inline void mesh_take(thread float &v, thread uint &k, float ov, uint ok) {
    if (ov > v || (ov == v && ok < k)) { v = ov; k = ok; }
}

// A candidate packed as a crossing carries it: (its value's bits, its key).
__attribute__((always_inline)) static inline void mesh_take(thread float &v, thread uint &k, uint2 packed) {
    mesh_take(v, k, as_type<float>(packed.x), packed.y);
}

__attribute__((always_inline)) static inline void mesh_take(thread float4 &v, thread uint4 &k, float4 ov, uint4 ok) {
    for (int c = 0; c < 4; c++)
        if (ov[c] > v[c] || (ov[c] == v[c] && ok[c] < k[c])) { v[c] = ov[c]; k[c] = ok[c]; }
}

// LastArrival, a split reduction's finish by the last of its J threadgroups to arrive (attention.metal's parts' merge,
// mesh_argmax's blocks, metal-microbench native_projection.metal's K parts, its recorder's publishing parts and Neural
// Engine shares): each fences its stores (at the device's scope, or with System at the system's: a reader off the GPU)
// and counts its arrival in word `slot`; true, after a fence, in every thread of the last to arrive where `last` (a
// threadgroup word) is given, else in its thread 0 alone. The last makes the word 0 again for the dispatch's next run once
// its merge is done (mesh_arrival_reset: metal-microbench docs/kernels.md, counts left by a cut dispatch), off the merge's
// path.
template <bool System = false>
__attribute__((always_inline)) static inline void mesh_arrival_fence() {
    if constexpr (System) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
    else atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

template <bool System = false>
__attribute__((always_inline)) static inline bool mesh_last_arrival(device atomic_uint *arrivals, uint slot, uint J, uint t,
                                                                   threadgroup uint *last = nullptr) {
    mesh_arrival_fence<System>();
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool mine = t == 0 && atomic_fetch_add_explicit(&arrivals[slot], 1u, memory_order_relaxed) == J - 1;
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

// The threadgroup's candidates into thread 0's (with All, into every thread's), and with `blocks` > 1 every block's into
// the last arriver's thread 0 (slot: row * blocks + block of the records pv, pk; arrival: the row's word). True for the
// threads holding the row's greatest.
template <uint T, bool All = false, typename V, typename K>
__attribute__((always_inline)) static inline bool mesh_argmax(thread V &v, thread K &k, uint lid, threadgroup V *tv, threadgroup K *tk,
                                                              coherent(device) device V *pv = nullptr, coherent(device) device K *pk = nullptr,
                                                              device atomic_uint *arrival = nullptr, uint slot = 0, uint blocks = 1) {
    for (ushort d = 16; d > 0; d /= 2) mesh_take(v, k, simd_shuffle_down(v, d), simd_shuffle_down(k, d));
    if (T > 32) {
        if (lid % 32 == 0) { tv[lid / 32] = v; tk[lid / 32] = k; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) for (uint q = 1; q < T / 32; q++) mesh_take(v, k, tv[q], tk[q]);
    }
    if (All) {
        if (T > 32) {
            if (lid == 0) { tv[0] = v; tk[0] = k; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            v = tv[0]; k = tk[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else { v = simd_broadcast(v, 0); k = simd_broadcast(k, 0); }
        return true;
    }
    if (blocks <= 1) return lid == 0;
    if (lid == 0) { pv[slot] = v; pk[slot] = k; }
    if (!mesh_last_arrival(arrival, 0u, blocks, lid)) return false;
    const uint first = slot - slot % blocks;
    for (uint b = 0; b < blocks; b++) mesh_take(v, k, pv[first + b], pk[first + b]);
    mesh_arrival_reset(arrival, 0u, lid);
    return true;
}
#endif
