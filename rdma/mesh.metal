// The GPU side of the words mesh-metal.m prepares (design/transport.md, M07, M10, M12, M30), and of the wire's
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

// A landed word's dud bit (mesh.h MESH_DUD, design/transport.md section 10): set by the bridge where the unit came as a
// dud (1 + its cycle with it) or the link was lost (~0, which has it too). A landing's wait compares the word's other
// bits with 1 + the cycle it awaits, so a dud of an earlier cycle is no landing of a later one, and a landing with the
// bit is its group's failure, never data.
#define MESH_DUD 0x8000000000000000UL

// The word at `word` once it holds at least `target`, with a fence before each load and no bound: no wait takes it, and it and mesh_remote_wait are kept only for a libnccl-mesh.dylib built before 4efbec2f, whose STEPS
// call mesh_remote_wait (metal-microbench metal_runtime.swift joins a dylib's STEPS to this file); both go once no
// node's dylib is older (design/transport.md).
template <typename W>
__attribute__((always_inline)) static inline W mesh_wait(volatile coherent(system) device const W *word, ulong target) {
    W seen;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *word;
    } while (seen < target);
    return seen;
}

// A landing's completion word (M07) once its bits but MESH_DUD hold 1 plus the invocation's cycle (M30; 0 in a one-pass
// program), its link cancelled (M12: UINT64_MAX) among them; the word returned tells a landing from a dud's (MESH_DUD).
__attribute__((always_inline)) static inline ulong mesh_remote_wait(volatile coherent(system) device const ulong *completion, ulong cycle) {
    ulong seen;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *completion;
    } while ((seen & ~MESH_DUD) < 1 + cycle);
    return seen;
}

// The word at `word` once it holds at least `target`, or as it is after `polls` loads (0 or 1: one load), each after a
// system-scope fence: without it a running kernel sees a store of the NIC's or the host's late, on the M4 Pro about
// 3.2 ms in nearly every wait and on the M5 Max 2.3 ms at the 99th percentile, while the fence costs at most 0.13 us a
// wait (metal-microbench tools/poll_probe.m, both nodes, 2026-10-21: one kernel spinning across rounds, unfenced 3198 us
// median on the Mini and 0.50 us with a 2308 us p99 on the M5, fenced 0.58 us on both; a poll 0.23-0.26 us either way).
// Each wait makes at most nccl.h NCCL_MESH_POLLS loads, about 1.1 s at that period on both nodes: the bound exists
// because a GPU spin with no end halts the node (the Mini's watchdog reset, 2026-10-21), and the caller tells a word
// short of `target` (the bound reached) from a landing. A reader orders its later loads after the observation with a
// system-scope fence of its own, after the barrier that gives the observation to its threads (nccl-mesh-metal.m
// mesh_step_await).
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

// A landing's completion word (M07) once its bits but MESH_DUD hold 1 plus the invocation's cycle (M30; 0 in a one-pass
// program), its link cancelled (M12: UINT64_MAX) among them, or as it is after `polls` loads, each after a fence as
// mesh_wait_bounded's; the word returned tells the four apart: a landing, a dud's (MESH_DUD), the link's loss, the bound
// (its other bits short of 1 + cycle).
__attribute__((always_inline)) static inline ulong mesh_remote_wait_bounded(volatile coherent(system) device const ulong *completion, ulong cycle,
                                                                             uint polls) {
    ulong seen;
    uint k = 0u;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *completion;
    } while ((seen & ~MESH_DUD) < 1 + cycle && ++k < polls);
    return seen;
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
