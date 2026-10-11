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
// mesh_argmax's blocks, metal-microbench native_projection.metal's K parts): each fences its stores and counts its arrival
// in word `slot`; true in the last to arrive, after a fence. The last makes the word 0 again for the dispatch's next run
// once its merge is done (mesh_arrival_reset: metal-microbench docs/kernels.md, counts left by a cut dispatch), off the
// merge's path.
__attribute__((always_inline)) static inline bool mesh_last_arrival(device atomic_uint *arrivals, uint slot, uint J, uint t, threadgroup uint &last) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (t == 0) last = atomic_fetch_add_explicit(&arrivals[slot], 1u, memory_order_relaxed) == J - 1 ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last == 0u) return false;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    return true;
}

__attribute__((always_inline)) static inline void mesh_arrival_reset(device atomic_uint *arrivals, uint slot, uint t) {
    if (t == 0) atomic_store_explicit(&arrivals[slot], 0u, memory_order_relaxed);
}

// The threadgroup's candidates into thread 0's, and with `blocks` > 1 every block's into the last arriver's (slot: row *
// blocks + block of the records pv, pk; arrival: the row's word; `last` the threadgroup's word mesh_last_arrival sets).
// True for the one thread holding the row's greatest.
template <uint T, typename V, typename K>
__attribute__((always_inline)) static inline bool mesh_argmax(thread V &v, thread K &k, uint lid, threadgroup V *tv, threadgroup K *tk,
                                                              coherent(device) device V *pv, coherent(device) device K *pk,
                                                              device atomic_uint *arrival, uint slot, uint blocks, threadgroup uint &last) {
    for (ushort d = 16; d > 0; d /= 2) mesh_take(v, k, simd_shuffle_down(v, d), simd_shuffle_down(k, d));
    if (lid % 32 == 0) { tv[lid / 32] = v; tk[lid / 32] = k; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        for (uint q = 1; q < T / 32; q++) mesh_take(v, k, tv[q], tk[q]);
        if (blocks > 1) { pv[slot] = v; pk[slot] = k; }
    }
    if (blocks <= 1) return lid == 0;
    if (!mesh_last_arrival(arrival, 0u, blocks, lid, last) || lid != 0) return false;
    const uint first = slot - slot % blocks;
    for (uint b = 0; b < blocks; b++) mesh_take(v, k, pv[first + b], pk[first + b]);
    mesh_arrival_reset(arrival, 0u, lid);
    return true;
}
#endif
