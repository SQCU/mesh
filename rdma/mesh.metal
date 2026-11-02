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
// Preparation fixes only the load budget. This local counter starts at the
// first executed observation; no wall-clock timestamp ages while compiling.
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

// The same fixed-width representation handles scalar, vector and record
// values. Metal's implicit record assignment does not accept coherent device
// records; these statically sized loads/stores do, without a staging buffer.
template <typename T>
__attribute__((always_inline)) static inline void mesh_payload_store(volatile coherent(system) device T *at,T value) {
    if constexpr (metal::is_scalar<T>::value || metal::is_vector<T>::value) {
        *at = value;
    } else if (sizeof(T) % 4 == 0) {
        thread const uint *from = (thread const uint *)&value;
        for (uint w = 0u; w < sizeof(T) / 4; w++) ((volatile coherent(system) device uint *)at)[w] = from[w];
    } else if (sizeof(T) % 2 == 0) {
        thread const ushort *from = (thread const ushort *)&value;
        for (uint w = 0u; w < sizeof(T) / 2; w++) ((volatile coherent(system) device ushort *)at)[w] = from[w];
    } else {
        thread const uchar *from = (thread const uchar *)&value;
        for (uint w = 0u; w < sizeof(T); w++) ((volatile coherent(system) device uchar *)at)[w] = from[w];
    }
}
template <typename T>
__attribute__((always_inline)) static inline T mesh_value_read(volatile coherent(system) device const T *values,ulong index) {
    return mesh_payload(values+index);
}
template <typename T>
__attribute__((always_inline)) static inline T mesh_value_read(constant T *values,ulong index) {
    return values[index];
}
template <typename T>
__attribute__((always_inline)) static inline void mesh_value_write(volatile coherent(system) device T *values,ulong index,T value) {
    mesh_payload_store(values+index,value);
}

// Counts form a maximum: concurrent publishers cannot move a count backwards.
// The release fence orders payload stores before the system-visible atomic.
// The SDK's public atomic overload fixes device coherence and device scope;
// its underlying primitive also takes the system coherence/scope needed here.
// The target declares ulong atomic min/max support (Apple8 macOS, Apple9+).
// The primitive's arity is the OS compiler's: macOS 27's takes a fifth operand, its memory flags; 26's takes four (both
// at language 4.0, where neither predefines a macro that tells them apart: tools probed on m5ult1-3 at 27.0 and the
// Mini at 26.5.1, 2026-11-01). Overload resolution picks the form the compiler declares, at compile time: the first
// exists only where five operands are accepted. Restored to compile on every node by the bandwidth/overhead session:
// ebd8bd24 wrote the four-operand call alone, and every kernel including this file failed to build on the Studios.
// What must stay: both forms, the five-operand one preferred (its `int` against the other's `long`).
template <typename A, typename V>
__attribute__((always_inline)) static inline auto mesh_atomic_max(A word, V value, int)
    -> decltype(__metal_atomic_max_explicit(word, value, 0, 0, __METAL_MEMORY_FLAGS_NONE__), void()) {
    __metal_atomic_max_explicit(word, value, int(metal::memory_order_relaxed), MESH_SYSTEM_THREAD_SCOPE, __METAL_MEMORY_FLAGS_NONE__);
}
template <typename A, typename V>
__attribute__((always_inline)) static inline void mesh_atomic_max(A word, V value, long) {
    __metal_atomic_max_explicit(word, value, int(metal::memory_order_relaxed), MESH_SYSTEM_THREAD_SCOPE);
}
__attribute__((always_inline)) static inline void mesh_publish(volatile coherent(system) device ulong *cell, ulong value) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
    volatile coherent(system) device metal::atomic_ulong *word = (volatile coherent(system) device metal::atomic_ulong *)cell;
    mesh_atomic_max(&word->__s, value, 0);
}

// The count and its activity mask can be produced independently of the
// numerical contributors. One system-scope RMW orders their announcements;
// it does not order unrelated cuts or hold a lock across numerical work.
#define MESH_COUNT_RMW(name, primitive) \
template<class A> __attribute__((always_inline)) static inline auto name(A word,uint value,int) \
 -> decltype(primitive(word,value,0,0,__METAL_MEMORY_FLAGS_NONE__)) { \
 return primitive(word,value,int(metal::memory_order_relaxed),MESH_SYSTEM_THREAD_SCOPE,__METAL_MEMORY_FLAGS_NONE__); } \
template<class A> __attribute__((always_inline)) static inline uint name(A word,uint value,long) { \
 return primitive(word,value,int(metal::memory_order_relaxed),MESH_SYSTEM_THREAD_SCOPE); }
MESH_COUNT_RMW(mesh_count_add,__metal_atomic_fetch_add_explicit)
MESH_COUNT_RMW(mesh_count_or,__metal_atomic_fetch_or_explicit)
#undef MESH_COUNT_RMW
template<class A> static inline auto mesh_count_load(A word,int)
 -> decltype(__metal_atomic_load_explicit(word,0,0,__METAL_MEMORY_FLAGS_NONE__)) {
    return __metal_atomic_load_explicit(word,int(metal::memory_order_relaxed),MESH_SYSTEM_THREAD_SCOPE,__METAL_MEMORY_FLAGS_NONE__);
}
template<class A> static inline uint mesh_count_load(A word,long) {
    return __metal_atomic_load_explicit(word,int(metal::memory_order_relaxed),MESH_SYSTEM_THREAD_SCOPE);
}
template<class A> static inline auto mesh_count_exchange(A word,thread uint *expected,uint value,int)
 -> decltype(__metal_atomic_compare_exchange_weak_explicit(word,expected,value,0,0,0,__METAL_MEMORY_FLAGS_NONE__)) {
    return __metal_atomic_compare_exchange_weak_explicit(word,expected,value,int(metal::memory_order_relaxed),
        int(metal::memory_order_relaxed),MESH_SYSTEM_THREAD_SCOPE,__METAL_MEMORY_FLAGS_NONE__);
}
template<class A> static inline bool mesh_count_exchange(A word,thread uint *expected,uint value,long) {
    return __metal_atomic_compare_exchange_weak_explicit(word,expected,value,int(metal::memory_order_relaxed),
        int(metal::memory_order_relaxed),MESH_SYSTEM_THREAD_SCOPE);
}
static inline uint mesh_graph_count_load(device metal::atomic_uint *value) {
    return mesh_count_load(&((volatile coherent(system) device metal::atomic_uint *)value)->__s,0);
}
static inline bool mesh_graph_count_exchange(device metal::atomic_uint *value,thread uint *expected,uint desired) {
    return mesh_count_exchange(&((volatile coherent(system) device metal::atomic_uint *)value)->__s,expected,desired,0);
}
static inline void mesh_graph_coverage(volatile coherent(system) device metal::atomic_uint *state,
    volatile coherent(system) device metal::atomic_uint *arrivals,volatile coherent(system) device const uint *expected,
    uint announce,uint contribution,bool known) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device,metal::memory_order_seq_cst,
                             static_cast<metal::thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
    const uint count=contribution?mesh_count_add(&arrivals->__s,contribution,0)+contribution:0u;
    if(known) {
        if(count==*expected)mesh_count_or(&state->__s,2u,0);
        return;
    }
    const uint ready=mesh_count_or(&state->__s,announce,0)|announce;
    metal::atomic_thread_fence(metal::mem_flags::mem_device,metal::memory_order_seq_cst,
                             static_cast<metal::thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
    if((ready&5u)==5u&&mesh_count_or(&arrivals->__s,0u,0)==*expected)mesh_count_or(&state->__s,2u,0);
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
