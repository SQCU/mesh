#include <metal_stdlib>
using namespace metal;

constant uint RANS_M_BITS = 10u;
constant uint RANS_M = 1u << RANS_M_BITS;
constant uint RANS_L = 1u << 16;

struct RansEncEntry {
    uint16_t freq;
    uint16_t cdf;
    uint32_t rcp;
    uint32_t shift;
};

struct RansRowMeta {
    uint32_t word_base;
    uint32_t total_words;
    half row_scale;
    uint16_t pad;
};

inline void fwht64_simd(thread float &v0, thread float &v1, uint lane) {
    float t0 = v0 + v1;
    float t1 = v0 - v1;
    v0 = t0;
    v1 = t1;
    for (ushort d = 1; d < 32; d <<= 1) {
        float p0 = simd_shuffle_xor(v0, d);
        float p1 = simd_shuffle_xor(v1, d);
        bool hi = (lane & d) != 0;
        v0 = hi ? (p0 - v0) : (v0 + p0);
        v1 = hi ? (p1 - v1) : (v1 + p1);
    }
    v0 *= 0.125f;
    v1 *= 0.125f;
}

kernel void rans32_encode_sym(
    device const uint8_t* syms [[buffer(0)]],
    device const RansEncEntry* enc_tbl [[buffer(1)]],
    device uint16_t* out_words [[buffer(2)]],
    device uint32_t* out_states [[buffer(3)]],
    device uint16_t* out_counts [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& max_words_per_lane [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= rows) return;

    threadgroup RansEncEntry tg_enc[256];
    for (uint i = lane; i < 256u; i += 32u) {
        tg_enc[i] = enc_tbl[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const uint8_t* xrow = syms + ulong(row) * cols;
    uint steps = cols >> 5;
    uint state = RANS_L;
    uint count = 0u;
    device uint16_t* dst = out_words + ulong(row) * 32u * max_words_per_lane + lane;

    for (int s = int(steps) - 1; s >= 0; --s) {
        uint sym = uint(xrow[(uint(s) << 5) + lane]);
        RansEncEntry e = tg_enc[sym];
        uint freq = uint(e.freq);
        uint x_max = freq << (32u - RANS_M_BITS);
        if (state >= x_max) {
            dst[(count++) << 5] = uint16_t(state & 0xFFFFu);
            state >>= 16u;
        }
        uint q;
        if (freq == 1u) {
            q = state;
        } else if (e.rcp == 0u) {
            q = state >> e.shift;
        } else {
            uint hi = mulhi(state, e.rcp);
            q = (hi + ((state - hi) >> 1u)) >> e.shift;
        }
        uint r = state - q * freq;
        state = (q << RANS_M_BITS) + uint(e.cdf) + r;
    }

    out_states[ulong(row) * 32u + lane] = state;
    out_counts[ulong(row) * 32u + lane] = uint16_t(count);
}

kernel void rans32_decode_to_int8(
    device const uint16_t* in_words [[buffer(0)]],
    device const uint32_t* in_states [[buffer(1)]],
    device const uint16_t* in_counts [[buffer(2)]],
    device const uint32_t* dec_slot_lut [[buffer(3)]],
    device int8_t* out_q [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& max_words_per_lane [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= rows) return;

    threadgroup uint32_t tg_lut[RANS_M];
    for (uint i = lane; i < RANS_M; i += 32u) {
        tg_lut[i] = dec_slot_lut[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const uint16_t* src = in_words + ulong(row) * 32u * max_words_per_lane + lane;
    uint state = in_states[ulong(row) * 32u + lane];
    int ptr = int(in_counts[ulong(row) * 32u + lane]);
    device int8_t* dst = out_q + ulong(row) * cols;
    uint steps = cols >> 5;

    for (uint s = 0u; s < steps; ++s) {
        uint slot = state & (RANS_M - 1u);
        uint packed = tg_lut[slot];
        int8_t val = int8_t(packed & 0xFFu);
        uint freq = (packed >> 8u) & 0xFFFu;
        uint bias = packed >> 20u;
        state = freq * (state >> RANS_M_BITS) + bias;
        if (state < RANS_L) {
            state = (state << 16u) | uint(src[(--ptr) << 5]);
        }
        dst[(s << 5) + lane] = val;
    }
}

kernel void rans32_compact_encode_had_int6(
    device const half* X [[buffer(0)]],
    device const RansEncEntry* enc_tbl [[buffer(1)]],
    device uint16_t* out_stream [[buffer(2)]],
    device uint32_t* out_states [[buffer(3)]],
    device uint16_t* out_counts [[buffer(4)]],
    device half* out_scales [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& max_words_per_lane [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= rows) return;

    threadgroup RansEncEntry tg_enc[64];
    for (uint i = lane; i < 64u; i += 32u) {
        tg_enc[i] = enc_tbl[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const half* xrow = X + ulong(row) * cols;
    uint nblk = cols >> 6;
    float rmax = 0.0f;
    for (uint b = 0u; b < nblk; ++b) {
        uint base = b << 6;
        float v0 = float(xrow[base + lane]);
        float v1 = float(xrow[base + lane + 32u]);
        fwht64_simd(v0, v1, lane);
        rmax = max(rmax, max(fabs(v0), fabs(v1)));
    }
    rmax = simd_max(rmax);
    float sc = max(rmax / 31.0f, 1e-8f);
    float inv_sc = 1.0f / sc;
    if (lane == 0u) out_scales[row] = half(sc);

    uint state = RANS_L;
    uint count = 0u;
    device uint16_t* dst = out_stream + ulong(row) * 32u * max_words_per_lane + lane;

    for (int b = int(nblk) - 1; b >= 0; --b) {
        uint base = uint(b) << 6;
        float v0 = float(xrow[base + lane]);
        float v1 = float(xrow[base + lane + 32u]);
        fwht64_simd(v0, v1, lane);

        int q1 = clamp(int(rint(v1 * inv_sc)), -31, 31) + 31;
        int q0 = clamp(int(rint(v0 * inv_sc)), -31, 31) + 31;

        uint syms2[2] = {uint(q1), uint(q0)};
        for (uint k = 0u; k < 2u; ++k) {
            RansEncEntry e = tg_enc[syms2[k]];
            uint freq = uint(e.freq);
            uint x_max = freq << (32u - RANS_M_BITS);
            if (state >= x_max) {
                dst[(count++) << 5] = uint16_t(state & 0xFFFFu);
                state >>= 16u;
            }
            uint q;
            if (freq == 1u) {
                q = state;
            } else if (e.rcp == 0u) {
                q = state >> e.shift;
            } else {
                uint hi = mulhi(state, e.rcp);
                q = (hi + ((state - hi) >> 1u)) >> e.shift;
            }
            uint r = state - q * freq;
            state = (q << RANS_M_BITS) + uint(e.cdf) + r;
        }
    }

    out_states[ulong(row) * 32u + lane] = state;
    out_counts[ulong(row) * 32u + lane] = uint16_t(count);
}
