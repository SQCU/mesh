#ifndef MESH_METAL
#define MESH_METAL
#pragma METAL internals : enable
#include <metal_stdlib>

template <typename W>
__attribute__((always_inline)) static inline W mesh_wait_bounded(volatile coherent(system) device const W *word, ulong target, uint polls,
                                                                  device metal::atomic_uint *stop) {
    W seen;
    uint k = 0u;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *word;
    } while (seen < target && ++k < polls && ((k & 1023u) != 0u || !stop || metal::atomic_load_explicit(stop, metal::memory_order_relaxed) == 0u));
    return seen;
}

template <typename W>
__attribute__((always_inline)) static inline W mesh_wait_bounded(volatile coherent(system) device const W *word, ulong target, uint polls) {
    return mesh_wait_bounded(word, target, polls, nullptr);
}

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

template <typename W>
__attribute__((always_inline)) static inline void mesh_publish(volatile coherent(system) device W *cell, ulong value) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
    *cell = W(value);
}

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
