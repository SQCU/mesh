#include <metal_stdlib>
using namespace metal;

constant uint RANS_M_BITS = 10u;
constant uint RANS_M = 1u << RANS_M_BITS;
constant uint RANS_L = 1u << 16;

constant float ALPHA_LUT[8] = {
    0.0f, 0.25f, 0.5f, 0.625f, 0.75f, 0.875f, 0.9375f, 1.0f
};

struct RansEncEntry {
    uint16_t freq;
    uint16_t cdf;
    uint32_t rcp;
    uint32_t shift;
};

struct MealyOp {
    uint16_t perm_row;
    uint8_t parent_li;
    uint8_t alpha_idx;
    half delta;
    half inv_delta;
};

kernel void mealy_rans32_encode_block(
    device const half* X [[buffer(0)]],
    device const MealyOp* ops [[buffer(1)]],
    device const RansEncEntry* enc_tbl [[buffer(2)]],
    device uint16_t* out_words [[buffer(3)]],
    device uint32_t* out_states [[buffer(4)]],
    device uint16_t* out_counts [[buffer(5)]],
    device half* blk_recon [[buffer(6)]],
    constant uint& num_blocks [[buffer(7)]],
    constant uint& block_rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& tile_steps [[buffer(10)]],
    constant uint& num_tiles [[buffer(11)]],
    constant uint& max_words_per_lane [[buffer(12)]],
    constant float& ctx_thresh [[buffer(13)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint blk = tgid.x;
    uint tile = tgid.y;
    if (blk >= num_blocks || tile >= num_tiles) return;

    threadgroup RansEncEntry tg_enc[2][64];
    for (uint i = lane; i < 128u; i += 32u) {
        tg_enc[i >> 6][i & 63u] = enc_tbl[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint total_steps = cols >> 5;
    uint s_beg = tile * tile_steps;
    uint s_end = min(s_beg + tile_steps, total_steps);
    uint n_s = (s_end > s_beg) ? (s_end - s_beg) : 0u;

    ulong blk_row_base = ulong(blk) * block_rows;
    uint8_t local_qc[16 * 20];

    for (uint li = 0u; li < block_rows; ++li) {
        MealyOp op = ops[blk_row_base + li];
        float alpha = ALPHA_LUT[op.alpha_idx & 7u];
        float delta = float(op.delta);
        float inv_delta = float(op.inv_delta);
        float thresh = 0.5f * delta;

        device const half* xrow = X + ulong(op.perm_row) * cols;
        device const half* hpar = blk_recon + (blk_row_base + op.parent_li) * cols;
        device half* hcur = blk_recon + (blk_row_base + li) * cols;

        bool is_key = (li == 0u) || (op.alpha_idx == 0u);
        for (uint j = 0u; j < n_s; ++j) {
            uint col = ((s_beg + j) << 5) + lane;
            float xv = float(xrow[col]);
            float pv = is_key ? 0.0f : float(hpar[col]);
            float pred = alpha * pv;
            int q = clamp(int(rint((xv - pred) * inv_delta)), -31, 31);
            float recon = pred + float(q) * delta;
            hcur[col] = half(recon);
            uint8_t c = is_key ? 1u : uint8_t(fabs(pv) > max(thresh, ctx_thresh));
            local_qc[li * 20u + j] = uint8_t((q + 31) | (c << 7u));
        }
    }

    uint state = RANS_L;
    uint count = 0u;
    ulong stream_idx = ulong(blk) * num_tiles + tile;
    device uint16_t* dst = out_words + stream_idx * 32u * max_words_per_lane + lane;

    for (int li = int(block_rows) - 1; li >= 0; --li) {
        for (int j = int(n_s) - 1; j >= 0; --j) {
            uint8_t qc = local_qc[uint(li) * 20u + uint(j)];
            uint sym = uint(qc & 0x7Fu);
            uint ctx = uint(qc >> 7u);
            RansEncEntry e = tg_enc[ctx][sym];
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

    out_states[stream_idx * 32u + lane] = state;
    out_counts[stream_idx * 32u + lane] = uint16_t(count);
}

kernel void mealy_rans32_decode_block(
    device const uint16_t* in_words [[buffer(0)]],
    device const uint32_t* in_states [[buffer(1)]],
    device const uint16_t* in_counts [[buffer(2)]],
    device const MealyOp* ops [[buffer(3)]],
    device const uint32_t* dec_lut [[buffer(4)]],
    device half* out_X [[buffer(5)]],
    device half* blk_recon [[buffer(6)]],
    constant uint& num_blocks [[buffer(7)]],
    constant uint& block_rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& tile_steps [[buffer(10)]],
    constant uint& num_tiles [[buffer(11)]],
    constant uint& max_words_per_lane [[buffer(12)]],
    constant float& ctx_thresh [[buffer(13)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint blk = tgid.x;
    uint tile = tgid.y;
    if (blk >= num_blocks || tile >= num_tiles) return;

    threadgroup uint32_t tg_lut[2][RANS_M];
    for (uint i = lane; i < 2u * RANS_M; i += 32u) {
        tg_lut[i >> RANS_M_BITS][i & (RANS_M - 1u)] = dec_lut[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint total_steps = cols >> 5;
    uint s_beg = tile * tile_steps;
    uint s_end = min(s_beg + tile_steps, total_steps);
    uint n_s = (s_end > s_beg) ? (s_end - s_beg) : 0u;

    ulong blk_row_base = ulong(blk) * block_rows;
    ulong stream_idx = ulong(blk) * num_tiles + tile;
    device const uint16_t* src = in_words + stream_idx * 32u * max_words_per_lane + lane;
    uint state = in_states[stream_idx * 32u + lane];
    int ptr = int(in_counts[stream_idx * 32u + lane]);

    for (uint li = 0u; li < block_rows; ++li) {
        MealyOp op = ops[blk_row_base + li];
        float alpha = ALPHA_LUT[op.alpha_idx & 7u];
        float delta = float(op.delta);
        float thresh = 0.5f * delta;

        device const half* hpar = blk_recon + (blk_row_base + op.parent_li) * cols;
        device half* hcur = blk_recon + (blk_row_base + li) * cols;
        device half* dst = out_X + ulong(op.perm_row) * cols;

        bool is_key = (li == 0u) || (op.alpha_idx == 0u);
        for (uint j = 0u; j < n_s; ++j) {
            uint col = ((s_beg + j) << 5) + lane;
            float pv = is_key ? 0.0f : float(hpar[col]);
            uint ctx = is_key ? 1u : uint(fabs(pv) > max(thresh, ctx_thresh));
            uint slot = state & (RANS_M - 1u);
            uint packed = tg_lut[ctx][slot];
            int q = int(int8_t(packed & 0xFFu));
            uint freq = (packed >> 8u) & 0xFFFu;
            uint bias = packed >> 20u;
            state = freq * (state >> RANS_M_BITS) + bias;
            if (state < RANS_L) {
                state = (state << 16u) | uint(src[(--ptr) << 5]);
            }
            half recon = half(alpha * pv + float(q) * delta);
            hcur[col] = recon;
            dst[col] = recon;
        }
    }
}
