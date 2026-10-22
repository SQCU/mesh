#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// docs/writing-a-service.md#the-step-kernel
template <uint N, uint G>
inline uint2 share(uint w) {
    constexpr uint q = N / G, r = N % G;
    return uint2(w * q + min(w, r), q + (w < r ? 1u : 0u));
}

// docs/writing-a-service.md#the-step-kernel
inline uint lane_element(uint lane) {
    uint q = lane / 4;
    return ((q & 4) + (lane / 2) % 4) * 8 + (q & 2) * 2 + (lane % 2) * 2;
}

// docs/writing-a-service.md#the-step-kernel
struct Rows {
    static constant constexpr const uint span = ~0u;
    device const half* p; uint ld;
    void fill(uint, uint) thread {}
    void load(thread simdgroup_half8x8& a, uint kt, uint row) const { simdgroup_load(a, p + row * ld + kt * 8, ld); }
};

// docs/writing-a-service.md#the-step-kernel
template <uint RG, typename A>
inline void partial(thread const A& src, uint row0, device const half* wp, uint k0, uint k1, thread simdgroup_float8x8* acc) {
    for (uint kt = k0; kt < k1; kt++) {
        simdgroup_half8x8 b;
        simdgroup_load(b, wp + kt * 64, 8);
        for (uint r = 0; r < RG; r++) {
            simdgroup_half8x8 a;
            src.load(a, kt, row0 + r * 8);
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
    }
}

// docs/writing-a-service.md#the-step-kernel
template <uint RG, uint RB, uint KIN, uint SG, typename A, typename Epi>
inline void stage(thread A& src, device const half* Wt, uint2 tiles, uint groups, uint ks, uint sgid,
                  threadgroup float* red, thread Epi& epi) {
    constexpr uint KT = KIN / 8;
    const uint items = tiles.y * groups, per = SG / ks;
    for (uint pass = 0; pass < items; pass += per) {
        uint item = pass + sgid / ks, seg = sgid % ks;
        bool active = item < items && sgid < per * ks;
        uint ct = item / groups, g = item % groups;
        simdgroup_float8x8 acc[RG];
        for (uint r = 0; r < RG; r++) acc[r] = simdgroup_float8x8(0.0f);
        for (uint c0 = 0, n; c0 < KT; c0 += n) {
            n = min(A::span, KT - c0);
            src.fill(c0, n);
            if (active) partial<RG>(src, g * RG * 8, Wt + (tiles.x + ct) * KT * 64, c0 + seg * n / ks, c0 + (seg + 1) * n / ks, acc);
        }
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
template <uint PEERS>
inline void rewrites(constant mesh_step_binding* sent, uint tid) {
    for (uint q = 0; q < PEERS; q++)
        if (sent[q].p) mesh_step_sent(*sent[q].p, sent[q].a, sent[q].v, tid);
}

// docs/writing-a-service.md#the-step-kernel
template <uint PEERS>
inline void release(device atomic_uint* arrivals, uint G, uint tid, threadgroup uint* last,
                    constant mesh_step_binding* publish) {
    mesh_arrival_fence<true>();
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) *last = atomic_fetch_add_explicit(arrivals, 1u, memory_order_relaxed) == G - 1 ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*last == 0u) return;
    for (uint q = 0; q < PEERS; q++) mesh_step_release(*publish[q].p, publish[q].a, tid);
    mesh_arrival_reset(arrivals, 0, tid);
}

// docs/writing-a-service.md#the-step-kernel
struct Activated {
    device half* Y; uint col0, ld, at;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        *(device half2*)(Y + (rt * 8 + at / 8) * ld + col0 + ct * 8 + at % 8) =
            half2(tanh(acc.thread_elements()[0]), tanh(acc.thread_elements()[1]));
    }
};

// docs/writing-a-service.md#the-step-kernel
struct Summand {
    device half* P; uint col0, ld, at;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        mesh_step_write<half2>((device uchar*)P, ((rt * 8 + at / 8) * ld + col0 + ct * 8 + at % 8) / 2,
                               half2(acc.thread_elements()[0], acc.thread_elements()[1]));
    }
};

// docs/writing-a-service.md#the-step-kernel
template <uint PEERS>
struct chain_words { uint first, drains, rows, ks0, ks1, pad; mesh_step_binding sent[PEERS], land[PEERS], publish[PEERS]; };

// docs/writing-a-service.md#the-step-kernel
template <uint STAGE, uint TILE, uint K, uint S, uint G, uint RG, uint RB, uint THREADS, uint SIMD, uint PEERS>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void chain_step(device const half* At [[buffer(0)]], device const half* Bt [[buffer(1)]], device const half* E [[buffer(2)]],
                       device half* out [[buffer(3)]], device half* X [[buffer(4)]], device half* Y [[buffer(5)]],
                       device half* P [[buffer(6)]], device atomic_uint* arrivals [[buffer(7)]],
                       device const half* Pprev [[buffer(8)]], constant chain_words<PEERS>& words [[buffer(15)]],
                       uint tid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint w [[threadgroup_position_in_grid]]) {
    static_assert(TILE == 8, "simdgroup matrices are 8 x 8");
    static_assert(THREADS % SIMD == 0 && RG % RB == 0, "whole simdgroups; whole batches of row groups");
    static_assert(STAGE < 3, "the land, f0 and f1");
    constexpr uint SG = THREADS / SIMD;
    const uint R = words.rows, groups = R / (TILE * RG);
    const uint2 xs = share<K / TILE, G>(w), ys = share<S / TILE, G>(w);
    if constexpr (STAGE == 0) {
        const bool first = words.first != 0;
        const uint x0 = xs.x * TILE, xw = xs.y * TILE;
        if (!first) {
            for (uint q = 0; q < PEERS; q++) {
                constant mesh_step_binding& l = words.land[q];
                mesh_step_enter(*l.p, l.v, l.progress, w == 0 && tid == 0);
                mesh_step_await(*l.p, l.a, l.v, tid);
            }
        } else if (w == 0) {
            mesh_arrival_reset(arrivals, 0, tid);
        }
        for (uint q = tid; q < R * xw; q += THREADS) {
            uint i = (q / xw) * K + x0 + q % xw;
            float z = 0.0f;
            if (!first) {
                float sum = float(Pprev[i]);
                for (uint q = 0; q < PEERS; q++) sum += float(mesh_step_read<half>(words.land[q].d + words.land[q].p->dst, i));
                z = tanh(float(half(sum)));
                if (words.drains) out[i] = half(z);
            }
            X[i] = half(float(mesh_step_read<half>((device const uchar *)E, i)) + z);
        }
    } else if constexpr (STAGE == 1) {
        threadgroup float red[SG * RB * 64];
        Activated activated{Y, ys.x * TILE, S, lane_element(lane)};
        Rows xs_rows{X, K};
        stage<RG, RB, K, SG>(xs_rows, At, ys, groups, words.ks0, sgid, red, activated);
    } else {
        threadgroup float red[SG * RB * 64];
        threadgroup uint last;
        Summand summand{P, xs.x * TILE, K, lane_element(lane)};
        Rows ys_rows{Y, S};
        rewrites<PEERS>(words.sent, tid);
        stage<RG, RB, S, SG>(ys_rows, Bt, xs, groups, words.ks1, sgid, red, summand);
        for (uint q = 0; q < PEERS; q++) mesh_step_enter(*words.publish[q].p, words.publish[q].v, words.publish[q].progress, w == 0 && tid == 0);
        release<PEERS>(arrivals, G, tid, &last, words.publish);
    }
}
