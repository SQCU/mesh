#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// docs/writing-a-service.md#the-step-kernel
inline void publish(device atomic_uint* stamps, uint w, uint tid, thread uint& phase) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_store_explicit(&stamps[w], phase + 1, memory_order_relaxed);
    }
    phase++;
}

// docs/writing-a-service.md#the-step-kernel
template <uint G, uint PATIENCE>
inline bool await_stamps(device atomic_uint* stamps, uint target, uint sgid, uint lane,
                         threadgroup atomic_uint* stop, device atomic_uint* err, uint cause) {
    if (sgid == 0) {
        for (uint polls = 0;; polls++) {
            uint v = lane < G ? atomic_load_explicit(&stamps[lane], memory_order_relaxed) : 0xffffffffu;
            if (simd_all(v >= target)) break;
            if (polls > PATIENCE) {
                if (lane == 0) { atomic_store_explicit(stop, 1u, memory_order_relaxed); atomic_fetch_or_explicit(err, cause, memory_order_relaxed); }
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    return atomic_load_explicit(stop, memory_order_relaxed) == 0;
}

// docs/writing-a-service.md#the-step-kernel
template <uint RG>
inline void partial(device const half* src, uint ld, device const half* wp, uint k0, uint k1, thread simdgroup_float8x8* acc) {
    for (uint kt = k0; kt < k1; kt++) {
        simdgroup_half8x8 b;
        simdgroup_load(b, wp + kt * 64, 8);
        for (uint r = 0; r < RG; r++) {
            simdgroup_half8x8 a;
            simdgroup_load(a, src + r * 8 * ld + kt * 8, ld);
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
    }
}

// docs/writing-a-service.md#the-step-kernel
template <uint RG, uint KIN, uint TILES, uint SG, typename Epi>
inline void stage(device const half* src, uint ld, device const half* Wt, uint tile0, uint groups, uint ks, uint sgid,
                  threadgroup float* red, thread Epi& epi) {
    constexpr uint KT = KIN / 8;
    const uint items = TILES * groups, per = SG / ks;
    for (uint pass = 0; pass < items; pass += per) {
        uint item = pass + sgid / ks, seg = sgid % ks;
        bool active = item < items && sgid < per * ks;
        uint ct = item / groups, g = item % groups;
        simdgroup_float8x8 acc[RG];
        for (uint r = 0; r < RG; r++) acc[r] = simdgroup_float8x8(0.0f);
        if (active) partial<RG>(src + g * RG * 8 * ld, ld, Wt + (tile0 + ct) * KT * 64, seg * KT / ks, (seg + 1) * KT / ks, acc);
        for (uint r = 0; r < RG; r++) {
            if (ks > 1) {
                simdgroup_store(acc[r], red + sgid * 64, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (active && seg == 0)
                    for (uint j = 1; j < ks; j++) {
                        simdgroup_float8x8 p; simdgroup_load(p, red + (sgid + j) * 64, 8);
                        for (uint e = 0; e < 2; e++) acc[r].thread_elements()[e] += p.thread_elements()[e];
                    }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (active && seg == 0) epi(acc[r], ct, g * RG + r);
        }
    }
}

// docs/writing-a-service.md#the-step-kernel
struct Activated {
    device half* Y; uint col0, ld;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        simdgroup_half8x8 o;
        for (uint e = 0; e < 2; e++) o.thread_elements()[e] = half(tanh(acc.thread_elements()[e]));
        simdgroup_store(o, Y + rt * 8 * ld + col0 + ct * 8, ld);
    }
};

// docs/writing-a-service.md#the-step-kernel
struct Summand {
    device half* P; uint col0, ld;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        simdgroup_half8x8 o;
        for (uint e = 0; e < 2; e++) o.thread_elements()[e] = half(acc.thread_elements()[e]);
        simdgroup_store(o, P + rt * 8 * ld + col0 + ct * 8, ld);
    }
};

// docs/writing-a-service.md#the-step-kernel
template <uint TILE, uint K, uint S, uint G, uint RG, uint THREADS, uint PATIENCE>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void chain_step(device const half* At [[buffer(0)]], device const half* Bt [[buffer(1)]], device const half* E [[buffer(2)]],
                       device const half* Z [[buffer(3)]], device half* X [[buffer(4)]], device half* Y [[buffer(5)]],
                       device half* P [[buffer(6)]], device atomic_uint* stamps [[buffer(7)]], device atomic_uint* err [[buffer(8)]],
                       constant uint* words [[buffer(15)]],
                       uint tid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint w [[threadgroup_position_in_grid]]) {
    static_assert(TILE == 8, "simdgroup matrices are 8 x 8");
    constexpr uint SG = THREADS / 32, Sx = K / G, Sy = S / G;
    threadgroup float red[SG * 64];
    threadgroup atomic_uint stop;
    if (tid == 0) atomic_store_explicit(&stop, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool first = words[0] != 0;
    uint phase = words[1];
    const uint R = words[2], groups = R / (TILE * RG), ks0 = words[3], ks1 = words[4];
    for (uint q = tid; q < R * Sx; q += THREADS) {
        uint i = (q / Sx) * K + w * Sx + q % Sx;
        X[i] = half(float(E[i]) + (first ? 0.0f : tanh(float(Z[i]))));
    }
    publish(stamps, w, tid, phase);
    if (!await_stamps<G, PATIENCE>(stamps, phase, sgid, lane, &stop, err, 1u)) return;
    Activated activated{Y, w * Sy, S};
    stage<RG, K, Sy / 8, SG>(X, K, At, w * (Sy / 8), groups, ks0, sgid, red, activated);
    publish(stamps, w, tid, phase);
    if (!await_stamps<G, PATIENCE>(stamps, phase, sgid, lane, &stop, err, 2u)) return;
    Summand summand{P, w * Sx, K};
    stage<RG, S, Sx / 8, SG>(Y, S, Bt, w * (Sx / 8), groups, ks1, sgid, red, summand);
}
