// The GPU side of the words mesh-metal.m prepares (design/prepared-machine.md M04, M07, M10, M12, M30), and of the wire's
// one small float format (FP8, nccl.h ncclFloat8e4m3 and ncclFloat8e5m2), defined once: a kernel that waits on a word,
// reads a landed slot, releases a cell or decodes FP8 includes this file and calls these from its own source, and no
// source splices them into another kernel's text (metal-microbench docs/acceptance-20261014.md A40, A54).
// libnccl-mesh's kernels include it, and so does every kernel that hands words to another agent (the host, a Neural
// Engine partner's thread) or reads FP8 (metal-microbench dsv41.swift). It is read from the library
// (the user's checkout, ~/mesh/rdma/mesh.metal).
#ifndef MESH_METAL
#define MESH_METAL
#pragma METAL internals : enable
#include <metal_stdlib>

// The word at `word` once it holds at least `target`, with a system-scope fence before each load (without it the M4
// did not observe the NIC's or the host's store: metal-microbench d8566be). No bound of its own: no libnccl-mesh wait
// takes it (theirs are mesh_wait_bounded's, design/recovery.md#five-seconds); its word's owner ends it (a host watcher's
// store). A reader orders its later loads after the observation with a system-scope fence of its own, after the barrier
// that gives the observation to its threads.
template <typename W>
__attribute__((always_inline)) static inline W mesh_wait(volatile coherent(system) device const W *word, ulong target) {
    W seen;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *word;
    } while (seen < target);
    return seen;
}

// The word at `word` once it holds at least `target`, or as it is after `polls` loads (each after mesh_wait's fence;
// 0 or 1: one load): every libnccl-mesh wait, at most nccl.h NCCL_MESH_POLLS loads, which hold the GPU at most
// MESH_PLATFORM_NS (design/recovery.md#five-seconds; a load's period the node GPU's, rdma/NCCL.md "Bounded waits").
// The word is still a canceller's (design/recovery.md Q16), and the caller tells a word short of `target` (the bound
// reached) from a landing.
template <typename W>
__attribute__((always_inline)) static inline W mesh_wait_bounded(volatile coherent(system) device const W *word, ulong target, uint polls) {
    W seen;
    uint k = 0u;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *word;
    } while (seen < target && ++k < polls);
    return seen;
}

// A landing's completion word (M07) once it holds 1 plus the invocation's cycle (M30; 0 in a one-pass program), or
// UINT64_MAX once its link was cancelled (M12), or as it is after `polls` loads; the word returned tells the three apart.
__attribute__((always_inline)) static inline ulong mesh_remote_wait_bounded(volatile coherent(system) device const ulong *completion, ulong cycle,
                                                                             uint polls) {
    return mesh_wait_bounded(completion, 1 + cycle, polls);
}

// A landed slot's value, loaded through volatile coherent(system) (the NIC wrote it) in the widest words that divide it.
template <typename T>
__attribute__((always_inline)) static inline T mesh_payload(volatile coherent(system) device const T *at) {
    T value;
    if (sizeof(T) % 4 == 0) {
        thread uint *to = (thread uint *)&value;
        for (uint w = 0u; w < sizeof(T) / 4; w++) to[w] = ((volatile coherent(system) device const uint *)at)[w];
    } else if (sizeof(T) % 2 == 0) {
        thread ushort *to = (thread ushort *)&value;
        for (uint w = 0u; w < sizeof(T) / 2; w++) to[w] = ((volatile coherent(system) device const ushort *)at)[w];
    } else {
        thread uchar *to = (thread uchar *)&value;
        for (uint w = 0u; w < sizeof(T); w++) to[w] = ((volatile coherent(system) device const uchar *)at)[w];
    }
    return value;
}

// Releases a cell: a system-scope fence orders every store before it (the payload, a SEND's slot) before `value` is
// stored into `cell`. A SEND cell's ready word takes its prepared argument plus the invocation's cycle (M04, M10, M30;
// mesh-collective.c mesh_host_publish is the host's release).
template <typename W>
__attribute__((always_inline)) static inline void mesh_publish(volatile coherent(system) device W *cell, ulong value) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
    *cell = W(value);
}

// FP8 E4M3 (fn: no infinity, S.1111.111 NaN) and, with e5, E5M2 (IEEE-like) [OCP 8-bit floating point, 2023]: a byte's
// value, and a float rounded to nearest even, saturating E4M3 to NaN and E5M2 to infinity past their largest; the host's
// codec is nccl-mesh.c's f8_to and f8_from, line for line.
static inline float f8_to(uchar v, bool e5) {
    const int mbits = e5 ? 2 : 3, bias = e5 ? 15 : 7, expf = (v >> mbits) & (e5 ? 31 : 15), mant = v & ((1 << mbits) - 1);
    float r;
    if (e5 && expf == 31) r = as_type<float>(mant ? 0x7fc00000u : 0x7f800000u);
    else if (!e5 && expf == 15 && mant == 7) r = as_type<float>(0x7fc00000u);
    else if (!expf) r = metal::ldexp(float(mant), 1 - bias - mbits);
    else r = metal::ldexp(1.0f + float(mant) / float(1 << mbits), expf - bias);
    return (v & 0x80) ? -r : r;
}

static inline uchar f8_from(float x, bool e5) {
    const int mbits = e5 ? 2 : 3, bias = e5 ? 15 : 7;
    const uint u = as_type<uint>(x), a = u & 0x7fffffffu;
    const uchar sign = (u >> 31) ? 0x80 : 0;
    if (a > 0x7f800000u) return sign | 0x7f;
    if (a == 0x7f800000u) return e5 ? (sign | 0x7c) : (sign | 0x7f);
    if (a == 0) return sign;
    int e = metal::max(int(a >> 23) - 127, 1 - bias);
    float n = metal::rint(metal::ldexp(as_type<float>(a), mbits - e));
    if (n >= float(1 << (mbits + 1))) { n *= 0.5f; e += 1; }
    int expf, mant;
    if (n < float(1 << mbits)) { expf = 0; mant = int(n); } else { expf = e + bias; mant = int(n) - (1 << mbits); }
    if (e5 ? expf >= 31 : (expf > 15 || (expf == 15 && mant == 7))) return e5 ? (sign | 0x7c) : (sign | 0x7f);
    return sign | uchar(expf << mbits) | uchar(mant);
}
#endif
