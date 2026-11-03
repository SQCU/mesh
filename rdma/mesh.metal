// The kernel prelude (torch_mesh/kernels.py prepends it to every kernel it writes): a SEND source's stores and a
// landing's loads system-coherent, a release of `ready` after a system fence (design/transport.md section 1; the
// partial-tensor contract's I2, a producer publishes when its writes are visible), and the wire's FP8 codec. No
// kernel waits: a consumer of landed bytes runs once they are present (metal.h MetalRunLanded).
#ifndef MESH_METAL
#define MESH_METAL
#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef MESH_SYSTEM_THREAD_SCOPE
#define MESH_SYSTEM_THREAD_SCOPE 3 /* etc/compute/metal.def */
#endif

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
