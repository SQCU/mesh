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
template <uint G, uint SIMD, uint PATIENCE>
inline bool await_stamps(device atomic_uint* stamps, uint target, uint sgid, uint lane,
                         threadgroup atomic_uint* stop, device atomic_uint* err, uint cause) {
    if (sgid == 0) {
        for (uint polls = 0;; polls++) {
            bool seen = true;
            for (uint i = lane; i < G; i += SIMD) seen = seen && atomic_load_explicit(&stamps[i], memory_order_relaxed) >= target;
            if (simd_all(seen)) break;
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
template <uint N, uint G>
inline uint2 share(uint w) {
    constexpr uint q = N / G, r = N % G;
    return uint2(w * q + min(w, r), q + (w < r ? 1u : 0u));
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
template <uint RG, uint RB, uint KIN, uint SG, typename Epi>
inline void stage(device const half* src, uint ld, device const half* Wt, uint2 tiles, uint groups, uint ks, uint sgid,
                  threadgroup float* red, thread Epi& epi) {
    constexpr uint KT = KIN / 8;
    const uint items = tiles.y * groups, per = SG / ks;
    for (uint pass = 0; pass < items; pass += per) {
        uint item = pass + sgid / ks, seg = sgid % ks;
        bool active = item < items && sgid < per * ks;
        uint ct = item / groups, g = item % groups;
        simdgroup_float8x8 acc[RG];
        for (uint r = 0; r < RG; r++) acc[r] = simdgroup_float8x8(0.0f);
        if (active) partial<RG>(src + g * RG * 8 * ld, ld, Wt + (tiles.x + ct) * KT * 64, seg * KT / ks, (seg + 1) * KT / ks, acc);
        for (uint r0 = 0; r0 < RG && ks > 1; r0 += RB) {
            for (uint r = 0; r < RB; r++) simdgroup_store(acc[r0 + r], red + (sgid * RB + r) * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = 1; s < ks; s <<= 1) {
                if (active && seg % (2 * s) == 0)
                    for (uint r = 0; r < RB; r++) {
                        simdgroup_float8x8 p; simdgroup_load(p, red + ((sgid + s) * RB + r) * 64, 8);
                        for (uint e = 0; e < 2; e++) acc[r0 + r].thread_elements()[e] += p.thread_elements()[e];
                        if (2 * s < ks) simdgroup_store(acc[r0 + r], red + (sgid * RB + r) * 64, 8);
                    }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (active && seg == 0)
            for (uint r = 0; r < RG; r++) epi(acc[r], ct, g * RG + r);
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
struct chain_words { uint first, phase, rows, ks0, ks1, last, drains, pad; mesh_step_binding land, publish; };

// docs/writing-a-service.md#the-step-kernel
template <uint TILE, uint K, uint S, uint G, uint RG, uint RB, uint THREADS, uint SIMD, uint PATIENCE>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void chain_step(device const half* At [[buffer(0)]], device const half* Bt [[buffer(1)]], device const half* E [[buffer(2)]],
                       device half* out [[buffer(3)]], device half* X [[buffer(4)]], device half* Y [[buffer(5)]], device half* P [[buffer(6)]],
                       device atomic_uint* stamps [[buffer(7)]], device atomic_uint* err [[buffer(8)]],
                       device atomic_uint* arrivals [[buffer(9)]], device const half* Pprev [[buffer(10)]],
                       constant chain_words& words [[buffer(15)]],
                       uint tid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint w [[threadgroup_position_in_grid]]) {
    static_assert(TILE == 8, "simdgroup matrices are 8 x 8");
    static_assert(THREADS % SIMD == 0 && RG % RB == 0, "whole simdgroups; whole batches of row groups");
    constexpr uint SG = THREADS / SIMD;
    threadgroup float red[SG * RB * 64];
    threadgroup atomic_uint stop;
    threadgroup uint last;
    if (tid == 0) atomic_store_explicit(&stop, atomic_load_explicit(err, memory_order_relaxed) ? 1u : 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&stop, memory_order_relaxed)) return;
    const bool first = words.first != 0;
    uint phase = words.phase;
    const uint R = words.rows, groups = R / (TILE * RG), ks0 = words.ks0, ks1 = words.ks1;
    const uint2 xs = share<K / TILE, G>(w), ys = share<S / TILE, G>(w);
    const uint x0 = xs.x * TILE, xw = xs.y * TILE;
    constant mesh_step_binding& l = words.land;
    if (!first) {
        mesh_step_enter(*l.p, l.v, l.progress, w == 0 && tid == 0);
        mesh_step_await(*l.p, l.a, l.v, tid);
    } else if (w == 0) {
        mesh_arrival_reset(arrivals, 0, tid);
    }
    for (uint q = tid; q < R * xw; q += THREADS) {
        uint i = (q / xw) * K + x0 + q % xw;
        float z = 0.0f;
        if (!first) {
            z = tanh(float(half(float(Pprev[i]) + float(mesh_step_read<half>(l.d + l.p->dst, i)))));
            if (words.drains) out[i] = half(z);
        }
        X[i] = half(float(E[i]) + z);
    }
    if (words.last) return;
    publish(stamps, w, tid, phase);
    if (!await_stamps<G, SIMD, PATIENCE>(stamps, phase, sgid, lane, &stop, err, 1u)) return;
    Activated activated{Y, ys.x * TILE, S};
    stage<RG, RB, K, SG>(X, K, At, ys, groups, ks0, sgid, red, activated);
    publish(stamps, w, tid, phase);
    if (!await_stamps<G, SIMD, PATIENCE>(stamps, phase, sgid, lane, &stop, err, 2u)) return;
    Summand summand{P, x0, K};
    stage<RG, RB, S, SG>(Y, S, Bt, xs, groups, ks1, sgid, red, summand);
    constant mesh_step_binding& b = words.publish;
    mesh_step_enter(*b.p, b.v, b.progress, w == 0 && tid == 0);
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, G, tid, &last)) {
        mesh_step_release(*b.p, b.a, tid);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}
