#include <metal_stdlib>
using namespace metal;

kernel void dual_rvq_a4_decode_body(
    device const uchar4* in_idx [[buffer(0)]],
    device const uint16_t* row_src [[buffer(1)]],
    device const half4* B0 [[buffer(2)]],
    device const half4* B1 [[buffer(3)]],
    device const half4* B2 [[buffer(4)]],
    device const half4* B3 [[buffer(5)]],
    device const half4* mu [[buffer(6)]],
    device half4* out_X [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& vecs8_per_row [[buffer(9)]],
    constant float& body_std [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    threadgroup half4 tg0[512];
    threadgroup half4 tg1[512];
    threadgroup half4 tg2[512];
    threadgroup half4 tg3[512];

    uint stride = tgs.x * tgs.y;
    for (uint i = tid; i < 512u; i += stride) {
        tg0[i] = B0[i];
        tg1[i] = B1[i];
        tg2[i] = B2[i];
        tg3[i] = B3[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint v = gid.x;
    uint r = gid.y;
    if (v >= vecs8_per_row || r >= rows) return;

    uint src_r = uint(row_src[r]);
    uchar4 idx = in_idx[ulong(src_r) * vecs8_per_row + v];

    uint i0 = uint(idx.x) << 1u;
    uint i1 = uint(idx.y) << 1u;
    uint i2 = uint(idx.z) << 1u;
    uint i3 = uint(idx.w) << 1u;

    float4 lo = (float4(tg0[i0]) + float4(tg1[i1]) + float4(tg2[i2]) + float4(tg3[i3])) * body_std + float4(mu[v * 2u]);
    float4 hi = (float4(tg0[i0 + 1u]) + float4(tg1[i1 + 1u]) + float4(tg2[i2 + 1u]) + float4(tg3[i3 + 1u])) * body_std + float4(mu[v * 2u + 1u]);

    ulong dst = (ulong(r) * vecs8_per_row + v) * 2u;
    out_X[dst] = half4(lo);
    out_X[dst + 1u] = half4(hi);
}

kernel void dual_rvq_a4_decode_out(
    device const uchar* in_idx3 [[buffer(0)]],
    device const uint16_t* row_src [[buffer(1)]],
    device const half4* O0 [[buffer(2)]],
    device const half4* O1 [[buffer(3)]],
    device const half4* O2 [[buffer(4)]],
    device const half4* mu [[buffer(5)]],
    device const half4* o_std [[buffer(6)]],
    device half4* out_X [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& vecs4_per_row [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    threadgroup half4 tg0[256];
    threadgroup half4 tg1[256];
    threadgroup half4 tg2[256];

    uint stride = tgs.x * tgs.y;
    for (uint i = tid; i < 256u; i += stride) {
        tg0[i] = O0[i];
        tg1[i] = O1[i];
        tg2[i] = O2[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint v = gid.x;
    uint r = gid.y;
    if (v >= vecs4_per_row || r >= rows) return;

    uint src_r = uint(row_src[r]);
    ulong p = (ulong(src_r) * vecs4_per_row + v) * 3u;
    uint i0 = uint(in_idx3[p]);
    uint i1 = uint(in_idx3[p + 1u]);
    uint i2 = uint(in_idx3[p + 2u]);

    float4 acc = (float4(tg0[i0]) + float4(tg1[i1]) + float4(tg2[i2])) * float4(o_std[v]) + float4(mu[v]);
    out_X[ulong(r) * vecs4_per_row + v] = half4(acc);
}

kernel void kalomaze_27b_decode_body_to_dram(
    device const uint* packed_u32 [[buffer(0)]],
    device const uint16_t* row_src [[buffer(1)]],
    device const uchar* remap [[buffer(2)]],
    device const half4* B0 [[buffer(3)]],
    device const half4* B1 [[buffer(4)]],
    device const half4* B2 [[buffer(5)]],
    device const half4* B3 [[buffer(6)]],
    device const half4* mu [[buffer(7)]],
    device half4* out_X [[buffer(8)]],
    constant uint& rows [[buffer(9)]],
    constant uint& vecs8_per_row [[buffer(10)]],
    constant uint& words_per_row [[buffer(11)]],
    constant float& body_std [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    threadgroup half4 tg0[512];
    threadgroup half4 tg1[512];
    threadgroup half4 tg2[512];
    threadgroup half4 tg3[512];

    uint stride = tgs.x * tgs.y;
    for (uint i = tid; i < 512u; i += stride) {
        tg0[i] = B0[i];
        tg1[i] = B1[i];
        tg2[i] = B2[i];
        tg3[i] = B3[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint v = gid.x;
    uint r = gid.y;
    if (v >= vecs8_per_row || r >= rows) return;

    uint src_r = uint(row_src[r]);
    device const uint* row_w = packed_u32 + ulong(src_r) * words_per_row;
    uint bit_pos = v * 27u;
    uint w_idx = bit_pos >> 5u;
    uint sh = bit_pos & 31u;
    uint w_lo = row_w[w_idx];
    uint w_hi = (sh > 5u) ? row_w[w_idx + 1u] : 0u;
    uint u27 = ((w_lo >> sh) | (sh > 5u ? (w_hi << (32u - sh)) : 0u)) & 0x7FFFFFFu;

    uint l0 = u27 & 63u;
    uint l1 = (u27 >> 6u) & 127u;
    uint l2 = (u27 >> 13u) & 127u;
    uint l3 = (u27 >> 20u) & 127u;

    ulong rbase = ulong(v) * 448u;
    uint i0 = uint(remap[rbase + l0]) << 1u;
    uint i1 = uint(remap[rbase + 64u + l1]) << 1u;
    uint i2 = uint(remap[rbase + 192u + l2]) << 1u;
    uint i3 = uint(remap[rbase + 320u + l3]) << 1u;

    float4 lo = (float4(tg0[i0]) + float4(tg1[i1]) + float4(tg2[i2]) + float4(tg3[i3])) * body_std + float4(mu[v * 2u]);
    float4 hi = (float4(tg0[i0 + 1u]) + float4(tg1[i1 + 1u]) + float4(tg2[i2 + 1u]) + float4(tg3[i3 + 1u])) * body_std + float4(mu[v * 2u + 1u]);

    ulong dst = (ulong(r) * vecs8_per_row + v) * 2u;
    out_X[dst] = half4(lo);
    out_X[dst + 1u] = half4(hi);
}

kernel void kalomaze_27b_kv_stream_dot_sram(
    device const uint* packed_u32 [[buffer(0)]],
    device const uint16_t* row_src [[buffer(1)]],
    device const uchar* remap [[buffer(2)]],
    device const half4* B0 [[buffer(3)]],
    device const half4* B1 [[buffer(4)]],
    device const half4* B2 [[buffer(5)]],
    device const half4* B3 [[buffer(6)]],
    device const half4* mu [[buffer(7)]],
    device const half4* Q [[buffer(8)]],
    device float* out_scores [[buffer(9)]],
    constant uint& rows [[buffer(10)]],
    constant uint& vecs8_per_row [[buffer(11)]],
    constant uint& words_per_row [[buffer(12)]],
    constant uint& rows_per_tg [[buffer(13)]],
    constant float& body_std [[buffer(14)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    threadgroup half4 tg0[512];
    threadgroup half4 tg1[512];
    threadgroup half4 tg2[512];
    threadgroup half4 tg3[512];

    uint stride = tgs.x * tgs.y;
    for (uint i = tid; i < 512u; i += stride) {
        tg0[i] = B0[i];
        tg1[i] = B1[i];
        tg2[i] = B2[i];
        tg3[i] = B3[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint v = ltid.x;
    if (v >= vecs8_per_row) return;

    float4 qlo = float4(Q[v * 2u]);
    float4 qhi = float4(Q[v * 2u + 1u]);
    float4 mlo = float4(mu[v * 2u]);
    float4 mhi = float4(mu[v * 2u + 1u]);
    float q_dot_mu = dot(mlo, qlo) + dot(mhi, qhi);
    float4 qs_lo = qlo * body_std;
    float4 qs_hi = qhi * body_std;

    uint bit_pos = v * 27u;
    uint w_idx = bit_pos >> 5u;
    uint sh = bit_pos & 31u;
    ulong rbase = ulong(v) * 448u;

    uint r_start = tgid.y * rows_per_tg + ltid.y;
    uint r_end = min((tgid.y + 1u) * rows_per_tg, rows);
    for (uint r = r_start; r < r_end; r += tgs.y) {
        uint src_r = uint(row_src[r]);
        device const uint* row_w = packed_u32 + ulong(src_r) * words_per_row;
        uint w_lo = row_w[w_idx];
        uint w_hi = (sh > 5u) ? row_w[w_idx + 1u] : 0u;
        uint u27 = ((w_lo >> sh) | (sh > 5u ? (w_hi << (32u - sh)) : 0u)) & 0x7FFFFFFu;

        uint i0 = uint(remap[rbase + (u27 & 63u)]) << 1u;
        uint i1 = uint(remap[rbase + 64u + ((u27 >> 6u) & 127u)]) << 1u;
        uint i2 = uint(remap[rbase + 192u + ((u27 >> 13u) & 127u)]) << 1u;
        uint i3 = uint(remap[rbase + 320u + ((u27 >> 20u) & 127u)]) << 1u;

        float4 lo = float4(tg0[i0]) + float4(tg1[i1]) + float4(tg2[i2]) + float4(tg3[i3]);
        float4 hi = float4(tg0[i0 + 1u]) + float4(tg1[i1 + 1u]) + float4(tg2[i2 + 1u]) + float4(tg3[i3 + 1u]);
        float partial = dot(lo, qs_lo) + dot(hi, qs_hi) + q_dot_mu;
        float s_sum = simd_sum(partial);
        if ((v & 31u) == 0u) {
            out_scores[r * 2u + (v >> 5u)] = s_sum;
        }
    }
}

kernel void bf16_kv_stream_dot_baseline(
    device const half4* K [[buffer(0)]],
    device const half4* Q [[buffer(1)]],
    device float* out_scores [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& vecs8_per_row [[buffer(4)]],
    constant uint& rows_per_tg [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    uint v = ltid.x;
    if (v >= vecs8_per_row) return;
    float4 qlo = float4(Q[v * 2u]);
    float4 qhi = float4(Q[v * 2u + 1u]);
    uint r_start = tgid.y * rows_per_tg + ltid.y;
    uint r_end = min((tgid.y + 1u) * rows_per_tg, rows);
    for (uint r = r_start; r < r_end; r += tgs.y) {
        ulong p = (ulong(r) * vecs8_per_row + v) * 2u;
        float partial = dot(float4(K[p]), qlo) + dot(float4(K[p + 1u]), qhi);
        float s_sum = simd_sum(partial);
        if ((v & 31u) == 0u) {
            out_scores[r * 2u + (v >> 5u)] = s_sum;
        }
    }
}

kernel void kalomaze_27b_adc_qk_sram(
    device const uint* packed_u32 [[buffer(0)]],
    device const uint16_t* row_src [[buffer(1)]],
    device const half* adc_lut [[buffer(2)]],
    device float* out_scores [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& vecs8_per_row [[buffer(5)]],
    constant uint& words_per_row [[buffer(6)]],
    constant uint& rows_per_tg [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    uint v_base = tgid.x * 16u;
    uint v_cnt = min(16u, vecs8_per_row - v_base);

    threadgroup half tg_adc[16 * 448];
    uint total_lut = v_cnt * 448u;
    uint stride = tgs.x * tgs.y;
    device const half* src_lut = adc_lut + ulong(v_base) * 448u;
    for (uint i = tid; i < total_lut; i += stride) {
        tg_adc[i] = src_lut[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lv = ltid.x;
    if (lv >= v_cnt) return;
    uint v = v_base + lv;
    uint bit_pos = v * 27u;
    uint w_idx = bit_pos >> 5u;
    uint sh = bit_pos & 31u;
    uint lbase = lv * 448u;

    uint r_start = tgid.y * rows_per_tg + ltid.y;
    uint r_end = min((tgid.y + 1u) * rows_per_tg, rows);
    for (uint r = r_start; r < r_end; r += tgs.y) {
        uint src_r = uint(row_src[r]);
        device const uint* row_w = packed_u32 + ulong(src_r) * words_per_row;
        uint w_lo = row_w[w_idx];
        uint w_hi = (sh > 5u) ? row_w[w_idx + 1u] : 0u;
        uint u27 = ((w_lo >> sh) | (sh > 5u ? (w_hi << (32u - sh)) : 0u)) & 0x7FFFFFFu;

        float s = float(tg_adc[lbase + (u27 & 63u)])
                + float(tg_adc[lbase + 64u + ((u27 >> 6u) & 127u)])
                + float(tg_adc[lbase + 192u + ((u27 >> 13u) & 127u)])
                + float(tg_adc[lbase + 320u + ((u27 >> 20u) & 127u)]);
        float s_sum = simd_sum(s);
        if (lv == 0u) {
            out_scores[r * 4u + tgid.x] = s_sum;
        }
    }
}
