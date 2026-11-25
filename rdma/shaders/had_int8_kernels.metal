#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

inline void fwht64_simd_pair(thread float& a, thread float& b, uint lane) {
    float s = a + b;
    float d = a - b;
    a = s;
    b = d;
    #pragma unroll
    for (uint mask = 1u; mask <= 16u; mask <<= 1u) {
        float oa = simd_shuffle_xor(a, mask);
        float ob = simd_shuffle_xor(b, mask);
        bool hi = (lane & mask) != 0u;
        a = hi ? (oa - a) : (a + oa);
        b = hi ? (ob - b) : (b + ob);
    }
    a *= 0.125f;
    b *= 0.125f;
}

kernel void fwht64_had_int8_pack(
    device const half* x [[buffer(0)]],
    device char* q_out [[buffer(1)]],
    device half* scale_out [[buffer(2)]],
    constant uint& n_h [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint blk_global = tg.x * 4u + sg;
    uint row = blk_global / n_h;
    uint blk = blk_global % n_h;
    ulong base = ulong(row) * ulong(n_h * 64u) + ulong(blk * 64u);

    float a = float(x[base + lane * 2u]);
    float b = float(x[base + lane * 2u + 1u]);
    fwht64_simd_pair(a, b, lane);

    bool upper_half = (lane >= 16u);
    float abs_local = max(fabs(a), fabs(b));
    float amax_lo = simd_max(upper_half ? 0.0f : abs_local);
    float amax_hi = simd_max(upper_half ? abs_local : 0.0f);
    float amax = max(upper_half ? amax_hi : amax_lo, 1e-8f);
    half scale_h = half(amax / 127.0f);
    float inv_s = 1.0f / float(scale_h);

    int qa = clamp(int(rint(a * inv_s)), -127, 127);
    int qb = clamp(int(rint(b * inv_s)), -127, 127);
    q_out[base + lane * 2u] = char(qa);
    q_out[base + lane * 2u + 1u] = char(qb);

    if (lane == 0u) {
        ulong sbase = ulong(row) * ulong(n_h * 2u) + ulong(blk * 2u);
        scale_out[sbase] = half(max(amax_lo, 1e-8f) / 127.0f);
        scale_out[sbase + 1u] = half(max(amax_hi, 1e-8f) / 127.0f);
    }
}

kernel void fwht64_had_int8_unpack(
    device const char* q_in [[buffer(0)]],
    device const half* scale_in [[buffer(1)]],
    device const half* add_rhs [[buffer(2)]],
    device half* y_out [[buffer(3)]],
    constant uint& n_h [[buffer(4)]],
    constant uint& has_add [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint blk_global = tg.x * 4u + sg;
    uint row = blk_global / n_h;
    uint blk = blk_global % n_h;
    ulong base = ulong(row) * ulong(n_h * 64u) + ulong(blk * 64u);
    ulong sbase = ulong(row) * ulong(n_h * 2u) + ulong(blk * 2u);

    float s = float(scale_in[sbase + (lane >= 16u ? 1u : 0u)]);
    float a = float(q_in[base + lane * 2u]) * s;
    float b = float(q_in[base + lane * 2u + 1u]) * s;
    fwht64_simd_pair(a, b, lane);

    half ha = half(a);
    half hb = half(b);
    if (has_add) {
        ha += add_rhs[base + lane * 2u];
        hb += add_rhs[base + lane * 2u + 1u];
    }
    y_out[base + lane * 2u] = ha;
    y_out[base + lane * 2u + 1u] = hb;
}

kernel void had_int8_gemm_mpp(
    device int8_t* X [[buffer(0)]],
    device half* SX [[buffer(1)]],
    device int8_t* W [[buffer(2)]],
    device half* SW [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint RM = 32u, CN = 32u, KT = 32u;
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = group.x * 64u + (sg / 2u) * RM;
    uint col = group.y * 64u + (sg % 2u) * CN;
    if (row >= M || col >= N) return;

    auto a = tensor(X, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
    auto b = tensor(W, dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, int(K)});
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        RM, CN, KT, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

    auto acc_f = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    auto tile_i32 = op.get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();
    #pragma unroll
    for (uint i = 0; i < acc_f.get_capacity(); ++i) acc_f[i] = 0.0f;

    uint n_blocks = K / KT;
    for (uint blk = 0; blk < n_blocks; ++blk) {
        uint k = blk * KT;
        auto ta = a.slice<KT, RM>(int(k), int(row));
        auto tb = b.slice<KT, CN>(int(k), int(col));
        op.run(ta, tb, tile_i32);
        #pragma unroll
        for (uint i = 0; i < acc_f.get_capacity(); ++i) {
            auto idx = tile_i32.get_multidimensional_index(i);
            uint r = row + uint(idx[1]);
            uint c = col + uint(idx[0]);
            float sx = float(SX[ulong(r) * n_blocks + blk]);
            float sw = float(SW[ulong(c) * n_blocks + blk]);
            acc_f[i] += float(tile_i32[i]) * (sx * sw);
        }
    }

    auto out_t = op.get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    #pragma unroll
    for (uint i = 0; i < out_t.get_capacity(); ++i) out_t[i] = half(acc_f[i]);
    auto c = tensor(Y, dextents<int, 2>{int(N), int(M)}, array<int, 2>{1, int(N)});
    out_t.store(c.slice(int(col), int(row)));
}

kernel void raw_int8_gemm_mpp(
    device int8_t* X [[buffer(0)]],
    device int8_t* W [[buffer(1)]],
    device int32_t* Y [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint RM = 32u, CN = 32u, KT = 64u;
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = group.x * 64u + (sg / 2u) * RM;
    uint col = group.y * 64u + (sg % 2u) * CN;
    if (row >= M || col >= N) return;

    auto a = tensor(X, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
    auto b = tensor(W, dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, int(K)});
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        RM, CN, KT, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;

    for (uint k = 0; k < K; k += KT) {
        auto ta = a.slice<KT, RM>(int(k), int(row));
        auto tb = b.slice<KT, CN>(int(k), int(col));
        op.run(ta, tb, acc);
    }
    auto c = tensor(Y, dextents<int, 2>{int(N), int(M)}, array<int, 2>{1, int(N)});
    acc.store(c.slice(int(col), int(row)));
}

kernel void fwht64_had_int8_pack_per_row(
    device const half* x [[buffer(0)]],
    device int8_t* q_out [[buffer(1)]],
    device half* scale_out [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sg_max[4];
    ulong row_base = ulong(row) * ulong(K);
    uint n_h = K / 64u;

    float local_max = 0.0f;
    for (uint blk = sg; blk < n_h; blk += 4u) {
        ulong base = row_base + ulong(blk * 64u);
        float a = float(x[base + lane * 2u]);
        float b = float(x[base + lane * 2u + 1u]);
        fwht64_simd_pair(a, b, lane);
        local_max = max(local_max, max(fabs(a), fabs(b)));
    }
    local_max = simd_max(local_max);
    if (lane == 0u) sg_max[sg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_amax = max(max(sg_max[0], sg_max[1]), max(sg_max[2], sg_max[3]));
    half scale_h = half(max(row_amax, 1e-8f) / 127.0f);
    float inv_s = 1.0f / float(scale_h);
    if (tid == 0u) scale_out[row] = scale_h;

    for (uint blk = sg; blk < n_h; blk += 4u) {
        ulong base = row_base + ulong(blk * 64u);
        float a = float(x[base + lane * 2u]);
        float b = float(x[base + lane * 2u + 1u]);
        fwht64_simd_pair(a, b, lane);
        q_out[base + lane * 2u] = int8_t(clamp(int(rint(a * inv_s)), -127, 127));
        q_out[base + lane * 2u + 1u] = int8_t(clamp(int(rint(b * inv_s)), -127, 127));
    }
}

kernel void had_int8_per_row_gemm_mpp(
    device int8_t* X [[buffer(0)]],
    device const half* SX [[buffer(1)]],
    device int8_t* W [[buffer(2)]],
    device const half* SW [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint RM = 32u, CN = 32u, KT = 64u;
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = group.x * 64u + (sg / 2u) * RM;
    uint col = group.y * 64u + (sg % 2u) * CN;
    if (row >= M || col >= N) return;

    auto a = tensor(X, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
    auto b = tensor(W, dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, int(K)});
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        RM, CN, KT, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;

    for (uint k = 0; k < K; k += KT) {
        auto ta = a.slice<KT, RM>(int(k), int(row));
        auto tb = b.slice<KT, CN>(int(k), int(col));
        op.run(ta, tb, acc);
    }

    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        auto idx = acc.get_multidimensional_index(i);
        uint r = row + uint(idx[1]);
        uint c = col + uint(idx[0]);
        Y[ulong(r) * N + c] = half(float(acc[i]) * (float(SX[r]) * float(SW[c])));
    }
}

kernel void had_int8_per_row_gemm_64x64_mpp(
    device int8_t* X [[buffer(0)]],
    device const half* SX [[buffer(1)]],
    device int8_t* W [[buffer(2)]],
    device const half* SW [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr uint RM = 64u, CN = 64u, KT = 64u;
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = group.x * RM;
    uint col = group.y * CN;
    if (row >= M || col >= N) return;

    auto a = tensor(X, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
    auto b = tensor(W, dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, int(K)});
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        RM, CN, KT, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;

    for (uint k = 0; k < K; k += KT) {
        auto ta = a.slice<KT, RM>(int(k), int(row));
        auto tb = b.slice<KT, CN>(int(k), int(col));
        op.run(ta, tb, acc);
    }

    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        auto idx = acc.get_multidimensional_index(i);
        uint r = row + uint(idx[1]);
        uint c = col + uint(idx[0]);
        Y[ulong(r) * N + c] = half(float(acc[i]) * (float(SX[r]) * float(SW[c])));
    }
}

kernel void raw_fp16_gemm_mpp(
    device half* X [[buffer(0)]],
    device half* W [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint RM = 32u, CN = 32u, KT = 64u;
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = group.x * 64u + (sg / 2u) * RM;
    uint col = group.y * 64u + (sg % 2u) * CN;
    if (row >= M || col >= N) return;

    auto a = tensor(X, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
    auto b = tensor(W, dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, int(K)});
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        RM, CN, KT, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    for (uint k = 0; k < K; k += KT) {
        auto ta = a.slice<KT, RM>(int(k), int(row));
        auto tb = b.slice<KT, CN>(int(k), int(col));
        op.run(ta, tb, acc);
    }
    auto out_t = op.get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    #pragma unroll
    for (uint i = 0; i < out_t.get_capacity(); ++i) out_t[i] = half(acc[i]);
    auto c = tensor(Y, dextents<int, 2>{int(N), int(M)}, array<int, 2>{1, int(N)});
    out_t.store(c.slice(int(col), int(row)));
}

kernel void fwht64_had_int6_pack(
    device const half* x [[buffer(0)]],
    device uint* q_u32_out [[buffer(1)]],
    device half* scale_out [[buffer(2)]],
    constant uint& n_h [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint blk_global = tg.x * 4u + sg;
    uint row = blk_global / n_h;
    uint blk = blk_global % n_h;
    ulong base = ulong(row) * ulong(n_h * 64u) + ulong(blk * 64u);

    float a = float(x[base + lane * 2u]);
    float b = float(x[base + lane * 2u + 1u]);
    fwht64_simd_pair(a, b, lane);

    float amax = max(simd_max(max(fabs(a), fabs(b))), 1e-8f);
    half scale_h = half(amax / 31.0f);
    float inv_s = 1.0f / float(scale_h);

    uint ua = uint(clamp(int(rint(a * inv_s)), -31, 31) + 31);
    uint ub = uint(clamp(int(rint(b * inv_s)), -31, 31) + 31);
    uint u12 = (ua & 63u) | ((ub & 63u) << 6u);

    uint g = lane >> 3u;
    uint sub = lane & 7u;
    uint b0 = g << 3u;

    uint v0 = simd_shuffle(u12, ushort(b0 + 0u));
    uint v1 = simd_shuffle(u12, ushort(b0 + 1u));
    uint v2 = simd_shuffle(u12, ushort(b0 + 2u));
    uint v3 = simd_shuffle(u12, ushort(b0 + 3u));
    uint v4 = simd_shuffle(u12, ushort(b0 + 4u));
    uint v5 = simd_shuffle(u12, ushort(b0 + 5u));
    uint v6 = simd_shuffle(u12, ushort(b0 + 6u));
    uint v7 = simd_shuffle(u12, ushort(b0 + 7u));

    ulong wbase = ulong(blk_global) * 12u + ulong(g * 3u);
    if (sub == 0u) {
        q_u32_out[wbase + 0u] = v0 | (v1 << 12u) | ((v2 & 0xFFu) << 24u);
    } else if (sub == 1u) {
        q_u32_out[wbase + 1u] = (v2 >> 8u) | (v3 << 4u) | (v4 << 16u) | ((v5 & 0xFu) << 28u);
    } else if (sub == 2u) {
        q_u32_out[wbase + 2u] = (v5 >> 4u) | (v6 << 8u) | (v7 << 20u);
    }
    if (lane == 0u) {
        scale_out[blk_global] = scale_h;
    }
}

kernel void fwht64_had_int6_unpack(
    device const uint* q_u32_in [[buffer(0)]],
    device const half* scale_in [[buffer(1)]],
    device half* y_out [[buffer(2)]],
    constant uint& n_h [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint blk_global = tg.x * 4u + sg;
    uint row = blk_global / n_h;
    uint blk = blk_global % n_h;
    ulong base = ulong(row) * ulong(n_h * 64u) + ulong(blk * 64u);

    uint g = lane >> 3u;
    uint sub = lane & 7u;
    ulong wbase = ulong(blk_global) * 12u + ulong(g * 3u);
    uint w0 = q_u32_in[wbase + 0u];
    uint w1 = q_u32_in[wbase + 1u];
    uint w2 = q_u32_in[wbase + 2u];

    uint u12 = 0u;
    switch (sub) {
        case 0u: u12 = w0 & 0xFFFu; break;
        case 1u: u12 = (w0 >> 12u) & 0xFFFu; break;
        case 2u: u12 = (w0 >> 24u) | ((w1 & 0xFu) << 8u); break;
        case 3u: u12 = (w1 >> 4u) & 0xFFFu; break;
        case 4u: u12 = (w1 >> 16u) & 0xFFFu; break;
        case 5u: u12 = (w1 >> 28u) | ((w2 & 0xFFu) << 4u); break;
        case 6u: u12 = (w2 >> 8u) & 0xFFFu; break;
        default: u12 = (w2 >> 20u) & 0xFFFu; break;
    }

    float s = float(scale_in[blk_global]);
    float a = float(int(u12 & 63u) - 31) * s;
    float b = float(int((u12 >> 6u) & 63u) - 31) * s;
    fwht64_simd_pair(a, b, lane);
    y_out[base + lane * 2u] = half(a);
    y_out[base + lane * 2u + 1u] = half(b);
}
