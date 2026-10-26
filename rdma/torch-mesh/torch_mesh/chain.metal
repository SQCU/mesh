#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

template <uint N, uint G>
inline uint2 share(uint w) {
    constexpr uint q = N / G, r = N % G;
    return uint2(w * q + min(w, r), q + (w < r ? 1u : 0u));
}

inline uint lane_element(uint lane) {
    uint q = lane / 4;
    return ((q & 4) + (lane / 2) % 4) * 8 + (q & 2) * 2 + (lane % 2) * 2;
}

struct Rows {
    coherent(device) device const half* p; uint ld, at;
    struct Tile {
        coherent(device) device const half* p; uint ld, at;
        half2 load(uint row) const { return *(coherent(device) device const half2*)(p + (row + at / 8) * ld); }
    };
    Tile tile(uint kt) const { return Tile{p + kt * 8 + at % 8, ld, at}; }
};

template <uint RG, typename A>
inline void partial(thread const A& src, uint row0, device const half* wp, uint k0, uint k1, thread simdgroup_float8x8* acc) {
    for (uint kt = k0; kt < k1; kt++) {
        simdgroup_half8x8 b;
        simdgroup_load(b, wp + kt * 64, 8);
        const auto t = src.tile(kt);
        for (uint r = 0; r < RG; r++) {
            simdgroup_half8x8 a;
            half2 v = t.load(row0 + r * 8);
            a.thread_elements()[0] = v[0];
            a.thread_elements()[1] = v[1];
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
    }
}

template <uint RG, uint RB, uint KIN, uint SG, typename A, typename Epi>
inline void stage(thread const A& src, device const half* Wt, uint2 tasks, uint groups, uint ks, uint sgid,
                  threadgroup float* red, thread Epi& epi) {
    constexpr uint KT = KIN / 8;
    const uint items = tasks.y, per = SG / ks;
    for (uint pass = 0; pass < items; pass += per) {
        uint item = pass + sgid / ks, seg = sgid % ks;
        bool active = item < items && sgid < per * ks;
        uint ct = (tasks.x + item) / groups, g = (tasks.x + item) % groups;
        simdgroup_float8x8 acc[RG];
        for (uint r = 0; r < RG; r++) acc[r] = simdgroup_float8x8(0.0f);
        if (active) partial<RG>(src, g * RG * 8, Wt + ct * KT * 64, seg * KT / ks, (seg + 1) * KT / ks, acc);
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

struct Activated {
    coherent(device) device half* Y; uint col0, ld, at;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        *(coherent(device) device half2*)(Y + (rt * 8 + at / 8) * ld + col0 + ct * 8 + at % 8) =
            half2(tanh(acc.thread_elements()[0]), tanh(acc.thread_elements()[1]));
    }
};

struct Summand {
    device half* P; uint col0, ld, at;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        mesh_step_write<half2>((device uchar*)P, ((rt * 8 + at / 8) * ld + col0 + ct * 8 + at % 8) / 2, 
                               half2(acc.thread_elements()[0], acc.thread_elements()[1]));
    }
};

// Processing stages above are restored from 1c0afa53^ (the coherent n-rank
// form of the original chain kernel). Physical bindings below use the current
// prepared write interface; rank count and cable count are separate operands.
template <uint CHANNELS>
struct chain_words {
    uint first, pad0, rows, ks0, ks1, last, width, actual_rows, previous_j, j, input_j, pad;
    mesh_step_binding land[CHANNELS], publish[CHANNELS], input[CHANNELS];
};

template <uint CHANNELS>
struct input_words { uint j, pad; mesh_step_binding channels[CHANNELS]; };

template <uint CHANNELS>
kernel void chain_publish_input(constant input_words<CHANNELS>& words [[buffer(15)]], uint tid [[thread_index_in_threadgroup]]) {
    for (uint q = 0; q < CHANNELS; q++) if (words.channels[q].p) {
        const auto p = words.channels[q];
        mesh_step_enter(*p.p, p.v, p.progress, tid == 0);
        mesh_step_release(*p.p, p.a, tid, words.j);
    }
}

template <uint TILE, uint K, uint S, uint R, uint G, uint RG, uint RB, uint THREADS, uint SIMD, uint RANKS, uint CHANNELS, uint PHASE>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void chain_step(device const half* At [[buffer(0)]], device const half* Bt [[buffer(1)]], device const half* E [[buffer(2)]],
                       coherent(device) device half* out [[buffer(3)]], coherent(device) device half* X [[buffer(4)]],
                       coherent(device) device half* Y [[buffer(5)]], device half* P [[buffer(6)]],
                       device atomic_uint* arrivals [[buffer(7)]],
                       device const uchar* previous [[buffer(8)]], constant chain_words<CHANNELS>& words [[buffer(15)]],
                       uint tid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint w [[threadgroup_position_in_grid]]) {
    static_assert(TILE == 8, "simdgroup matrices are 8 x 8");
    static_assert(THREADS % SIMD == 0 && RG % RB == 0, "whole simdgroups; whole batches of row groups");
    constexpr uint SG = THREADS / SIMD;
    threadgroup float red[SG * RB * 64];
    threadgroup uint last;
    constexpr uint groups = R / (TILE * RG);
    const uint ks0 = words.ks0, ks1 = words.ks1;
    const uint2 xs = share<K / TILE * groups, G>(w), ys = share<S / TILE * groups, G>(w);
    const uint2 elements = share<R * K, G>(w);
    // Each phase uses the existing numerical body. Dispatch dependencies are
    // compiled from X/Y accesses; no group waits for another to be scheduled.
    if constexpr (PHASE == 0) {
        if (!words.last && tid == 0) for (uint q = 0; q < CHANNELS; q++) if (words.input[q].p) {
            constant mesh_step_binding& input = words.input[q];
            mesh_step_enter(*input.p, input.v, input.progress, w == 0);
            mesh_step_counts(*input.p, input.a, input.v, words.input_j);
        }
        if (!words.first) {
            if (tid == 0) for (uint q = 0; q < CHANNELS; q++) if (words.land[q].p) {
                constant mesh_step_binding& l = words.land[q];
                mesh_step_enter(*l.p, l.v, l.progress, w == 0 && tid == 0);
                mesh_step_counts(*l.p, l.a, l.v, words.previous_j);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
        for (uint q = tid; q < elements.y; q += THREADS) {
            const uint i = elements.x + q;
            const uint row = i / K, column = i % K;
            const bool present = row < words.actual_rows && column < words.width;
            const uint logical = row * words.width + column;
            float z = 0.0f;
            if (!words.first) {
                float sum = 0.0f;
                for (uint rank = 0; rank < RANKS; rank++) sum += float(mesh_step_read<half>(previous, ulong(rank) * R * K + i));
                z = tanh(float(half(sum)));
                if (present) out[logical] = half(z);
            }
            if (!words.last) X[i] = half((present ? float(mesh_step_read<half>((device const uchar*)E, logical)) : 0.0f) + z);
        }
        return;
    }
    const uint at = lane_element(lane);
    if constexpr (PHASE == 1) {
        Activated activated{Y, 0, S, at};
        const Rows xs_rows{X, K, at};
        stage<RG, RB, K, SG>(xs_rows, At, ys, groups, ks0, sgid, red, activated);
        return;
    }
    Summand summand{P, 0, K, at};
    const Rows ys_rows{Y, S, at};
    stage<RG, RB, S, SG>(ys_rows, Bt, xs, groups, ks1, sgid, red, summand);
    for (uint q = 0; q < CHANNELS; q++) if (words.publish[q].p)
        mesh_step_enter(*words.publish[q].p, words.publish[q].v, words.publish[q].progress, w == 0 && tid == 0);
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, G, tid, &last)) {
        for (uint q = 0; q < CHANNELS; q++) if (words.publish[q].p)
            mesh_step_release(*words.publish[q].p, words.publish[q].a, tid, words.j);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}
