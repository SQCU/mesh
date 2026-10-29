// The GPU side of the words mesh-metal.m prepares (design/transport.md, M07, M10, M12, M30), and of the wire's
// one small float format (FP8, nccl.h ncclFloat8e4m3 and ncclFloat8e5m2), defined once: a kernel that waits on a count,
// reads landed bytes, publishes a count or decodes FP8 includes this file and calls these from its own source, and no
// source splices them into another kernel's text (metal-microbench docs/acceptance-20261014.md A40, A54).
// libnccl-mesh's kernels include it, and so does every kernel that hands words to another party (the host, a Neural
// Engine partner's thread) or reads FP8 (metal-microbench dsv41.swift). It is read from the library
// (the user's checkout, ~/mesh/rdma/mesh.metal).
#ifndef MESH_METAL
#define MESH_METAL
#pragma METAL internals : enable
#include <metal_stdlib>

// The word at `word` once it holds at least `target`, or as it is after `polls` loads (0 or 1: one load), each after a
// system-scope fence: without it a running kernel sees a store of the NIC's or the host's late, on the M4 Pro about
// 3.2 ms in nearly every wait and on the M5 Max 2.3 ms at the 99th percentile, while the fence costs at most 0.13 us a
// wait (metal-microbench tools/poll_probe.m, both nodes, 2026-10-21: one kernel spinning across rounds, unfenced 3198 us
// median on the Mini and 0.50 us with a 2308 us p99 on the M5, fenced 0.58 us on both; a poll 0.23-0.26 us either way).
// Each wait makes at most nccl.h NCCL_MESH_POLLS loads, about 1.1 s at that period on both nodes: the bound exists
// because a GPU spin with no end halts the node (the Mini's watchdog reset, 2026-10-21), and the caller tells a word
// short of `target` (the bound reached) from a landing. A reader orders its later loads after the observation with a
// system-scope fence of its own, after the barrier that gives the observation to its threads (nccl-mesh-metal.m
// mesh_step_await). With `stop` (not null), the wait also ends once the word at `stop` is not 0, read every 2^10 loads:
// a dispatch's failure claim (nccl-mesh-kernels.h mesh_step_claim), so that once one wait of a dispatch has failed, every
// other wait of it ends within 2^10 loads instead of at its own bound (the Mini's watchdog reset, 2026-10-21: a dispatch's
// whole spin is what must stay bounded, and a wide dispatch runs its threadgroups' waits in waves).
template <typename W>
__attribute__((always_inline)) static inline W mesh_wait_bounded(volatile coherent(system) device const W *word, ulong target, uint polls,
                                                                  device metal::atomic_uint *stop) {
    W seen;
    uint k = 0u;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
        seen = *word;
    } while (seen < target && ++k < polls && ((k % MESH_STOP_POLL_PERIOD) != 0u || !stop || metal::atomic_load_explicit(stop, metal::memory_order_relaxed) == 0u));
    return seen;
}

template <typename W>
__attribute__((always_inline)) static inline W mesh_wait_bounded(volatile coherent(system) device const W *word, ulong target, uint polls) {
    return mesh_wait_bounded(word, target, polls, nullptr);
}

// A landed value (bytes the NIC wrote into a landing buffer), loaded through volatile coherent(system) in the widest words
// that divide it.
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

// Publishes a count: a system-scope fence orders every system-coherent store before it (the bytes a SEND reads from the
// client's own pages, stored system-coherent: a plain store reaches the NIC only once its command buffer has ended,
// metal-microbench docs/measurement.md, 2026-09-29) before `value` is stored into `cell` (a holding's `ready`,
// design/transport.md section 14; a failure word).
template <typename W>
__attribute__((always_inline)) static inline void mesh_publish(volatile coherent(system) device W *cell, ulong value) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
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
