// The greatest of each row's candidates in one total order, greater value first and then the lower key (metal-microbench
// docs/kernels.md#one-kernel-interface, argmax): a row's range split over B threadgroups of T threads, each thread's
// candidates the caller's scan, the simdgroups' met by shuffles, the threadgroup's through threadgroup memory, and the
// blocks' by the last threadgroup to arrive (its arrival word stored 0 again). V and K are one candidate (float, uint) or
// four independent channels (float4, uint4). metal-microbench's engine instantiates it over a row of logits
// (argmax_partial) and its recorder over LiteRT's ARG_MAX channels (native_argmax.metal).
#ifndef TORCH_MESH_ARGMAX
#define TORCH_MESH_ARGMAX
#include <metal_stdlib>
using namespace metal;

__attribute__((always_inline)) static inline void mesh_take(thread float &v, thread uint &k, float ov, uint ok) {
    if (ov > v || (ov == v && ok < k)) { v = ov; k = ok; }
}

__attribute__((always_inline)) static inline void mesh_take(thread float4 &v, thread uint4 &k, float4 ov, uint4 ok) {
    for (int c = 0; c < 4; c++)
        if (ov[c] > v[c] || (ov[c] == v[c] && ok[c] < k[c])) { v[c] = ov[c]; k[c] = ok[c]; }
}

// The threadgroup's candidates into thread 0's, and with `blocks` > 1 every block's into the last arriver's (slot: row *
// blocks + block of the records pv, pk; arrival: the row's word). True for the one thread holding the row's greatest.
template <uint T, typename V, typename K>
__attribute__((always_inline)) static inline bool mesh_argmax(thread V &v, thread K &k, uint lid, threadgroup V *tv, threadgroup K *tk,
                                                              coherent(device) device V *pv, coherent(device) device K *pk,
                                                              device atomic_uint *arrival, uint slot, uint blocks) {
    for (ushort d = 16; d > 0; d /= 2) mesh_take(v, k, simd_shuffle_down(v, d), simd_shuffle_down(k, d));
    if (lid % 32 == 0) { tv[lid / 32] = v; tk[lid / 32] = k; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0) return false;
    for (uint q = 1; q < T / 32; q++) mesh_take(v, k, tv[q], tk[q]);
    if (blocks > 1) {
        pv[slot] = v;
        pk[slot] = k;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(arrival, 1u, memory_order_relaxed) != blocks - 1) return false;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint first = slot - slot % blocks;
        for (uint b = 0; b < blocks; b++) mesh_take(v, k, pv[first + b], pk[first + b]);
        atomic_store_explicit(arrival, 0u, memory_order_relaxed);
    }
    return true;
}
#endif
