// Streamed coded weights: the decompression as a producer whose consumer is an operand (metal-microbench
// docs/kernels.md "Streamed codes of the whole weight" and "Streaming the code into the matmul"; design/algorithm-
// sources.md#nnffn: C_ij = F_ij(X_i), Y_j = sum_i C_ij handed to its consumer).
//
// A coded matrix is 32 x 32 tiles, each its own width b (bits a value, 0 to 12), step and word offset; a lane of a
// simdgroup holds a tile row, two values a 32-bit word (paired layout), decoded as odd integers 2v + 1 - 2^b scaled by
// half the step. Every kernel over such a matrix is one of these producers applied to a consumer:
//
//   streamed_pairs(c, at, lane, f)     lane's row of tile `at` in registers: f(j, float2) for its 16 pairs, unscaled
//   streamed_panels(c, ..., f)         a threadgroup's simdgroups decode a 32 x 256 panel together (fp16, scaled),
//                                      then f(t0, panel)
//
// and the products' sums reach their own consumers through streamed_value (a K split's or a member's partials summed,
// the row's scale applied). Include after <metal_stdlib>; the tensor-op kernels need Metal 4.
#ifndef TORCH_MESH_STREAMED
#define TORCH_MESH_STREAMED
#include <metal_stdlib>
using namespace metal;

struct Coded {
    device const uint *bits;
    device const uchar *widths;
    device const float *steps;
    device const uint *offsets;
    uint tiles;
};

static inline Coded coded(device const uint *bits, device const uchar *widths, device const float *steps,
                          device const uint *offsets, uint tiles) {
    return Coded{bits, widths, steps, offsets, tiles};
}

#define STREAMED_WIDTHS(CALL) \
    switch (b) { \
        case 1: CALL(1); break; case 2: CALL(2); break; case 3: CALL(3); break; case 4: CALL(4); break; \
        case 5: CALL(5); break; case 6: CALL(6); break; case 7: CALL(7); break; case 8: CALL(8); break; \
        case 9: CALL(9); break; case 10: CALL(10); break; case 11: CALL(11); break; default: CALL(12); break; \
    }

template <uint B, uint J, typename Words>
static inline uint streamed_bits(Words w) {
    constexpr uint bit = J * B, word = bit >> 4, shift = bit & 15u, mask = ((1u << B) - 1u) * 0x00010001u;
    if (shift + B <= 16) return (shift ? w[word] >> (shift - 1) : w[word] << 1) & (mask << 1);
    constexpr uint low = ((1u << (16 - shift)) - 1u) * 0x00010001u;
    return ((w[word] >> (shift - 1)) & (low << 1)) | ((w[word + 1] << (17 - shift)) & ((mask ^ low) << 1));
}

template <uint B, uint J, typename Words>
static inline half2 streamed_half(Words w) {
    const uint value = streamed_bits<B, J>(w);
    if (B <= 9) return as_type<half2>(value | 0x64016401u) - half2(half(1024.0f + float(1u << B)));
    return half2(float2(float(value & 0xFFFFu), float(value >> 16)) + float2(1.0f - float(1u << B)));
}

template <uint B, uint J, typename Words>
static inline float2 streamed_float(Words w) {
    const uint value = streamed_bits<B, J>(w);
    if (B <= 9) return float2(as_type<half2>(value | 0x64016401u) - half2(half(1024.0f + float(1u << B))));
    return float2(float(value & 0xFFFFu), float(value >> 16)) + float2(1.0f - float(1u << B));
}

template <uint B, typename F>
static inline void streamed_pairs_of(thread const uint *w, F f) {
    f(0, streamed_float<B, 0>(w)); f(1, streamed_float<B, 1>(w)); f(2, streamed_float<B, 2>(w)); f(3, streamed_float<B, 3>(w));
    f(4, streamed_float<B, 4>(w)); f(5, streamed_float<B, 5>(w)); f(6, streamed_float<B, 6>(w)); f(7, streamed_float<B, 7>(w));
    f(8, streamed_float<B, 8>(w)); f(9, streamed_float<B, 9>(w)); f(10, streamed_float<B, 10>(w)); f(11, streamed_float<B, 11>(w));
    f(12, streamed_float<B, 12>(w)); f(13, streamed_float<B, 13>(w)); f(14, streamed_float<B, 14>(w)); f(15, streamed_float<B, 15>(w));
}

// lane's row of tile `at`, its pairs as odd integers to f(j, pair); returns the tile's width (0: nothing decoded)
template <typename F>
static inline uint streamed_pairs(Coded c, uint at, uint lane, F f) {
    const uint b = c.widths[at];
    if (b == 0) return 0;
    uint w[12];
    device const uint *source = c.bits + c.offsets[at] + lane * b;
    for (uint i = 0; i < 12; i++) w[i] = i < b ? source[i] : 0u;
#define STREAMED_PAIRS(B) streamed_pairs_of<B>(w, f)
    STREAMED_WIDTHS(STREAMED_PAIRS)
#undef STREAMED_PAIRS
    return b;
}

template <uint B>
static inline void streamed_row_of(device const uint *w, threadgroup half2 *row, half2 scale) {
    row[0] = streamed_half<B, 0>(w) * scale; row[1] = streamed_half<B, 1>(w) * scale; row[2] = streamed_half<B, 2>(w) * scale;
    row[3] = streamed_half<B, 3>(w) * scale; row[4] = streamed_half<B, 4>(w) * scale; row[5] = streamed_half<B, 5>(w) * scale;
    row[6] = streamed_half<B, 6>(w) * scale; row[7] = streamed_half<B, 7>(w) * scale; row[8] = streamed_half<B, 8>(w) * scale;
    row[9] = streamed_half<B, 9>(w) * scale; row[10] = streamed_half<B, 10>(w) * scale; row[11] = streamed_half<B, 11>(w) * scale;
    row[12] = streamed_half<B, 12>(w) * scale; row[13] = streamed_half<B, 13>(w) * scale; row[14] = streamed_half<B, 14>(w) * scale;
    row[15] = streamed_half<B, 15>(w) * scale;
}

// lane's row of tile `at` into `row` (16 half2), scaled by half the tile's step times `factor`; zeros where the tile
// is empty. Returns the tile's width.
static inline uint streamed_row(Coded c, uint at, uint lane, float factor, threadgroup half2 *row) {
    const uint b = c.widths[at];
    if (b == 0) {
        for (uint j = 0; j < 16; j++) row[j] = half2(0.0h);
        return 0;
    }
    const half2 scale = half2(half(0.5f * c.steps[at] * factor));
    device const uint *w = c.bits + c.offsets[at] + lane * b;
#define STREAMED_ROW(B) streamed_row_of<B>(w, row, scale)
    STREAMED_WIDTHS(STREAMED_ROW)
#undef STREAMED_ROW
    return b;
}

// A threadgroup's eight simdgroups decode panels of 8 tiles (32 x 256, row stride 264) of block `block`, panels
// [first, last), then f(t0, panel) runs on the whole threadgroup; a panel past the last tile is zero there.
template <typename F>
static inline void streamed_panels(Coded c, uint block, uint first, uint last, uint simd, uint lane,
                                   threadgroup half *panel, F f) {
    for (uint p = first; p < last; p++) {
        const uint t0 = p * 8;
        if (t0 >= c.tiles) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half2 *row = (threadgroup half2 *)(panel + lane * 264 + simd * 32);
        if (t0 + simd < c.tiles) streamed_row(c, block * c.tiles + t0 + simd, lane, 1.0f, row);
        else for (uint j = 0; j < 16; j++) row[j] = half2(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        f(t0, panel);
    }
}

// A product's sum for input row n at coded output row r over its shares (K splits of a dispatch; members' shares after
// a crossing), the row's scale applied: what every consumer of a product reads.
static inline float streamed_value(device const float *partials, ulong at, uint shares, uint stride, float scale) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint s = 0;
    for (; s + 4 <= shares; s += 4) {
        a0 += partials[at + ulong(s) * stride];
        a1 += partials[at + ulong(s + 1) * stride];
        a2 += partials[at + ulong(s + 2) * stride];
        a3 += partials[at + ulong(s + 3) * stride];
    }
    for (; s < shares; s++) a0 += partials[at + ulong(s) * stride];
    return scale * ((a0 + a1) + (a2 + a3));
}

// One input row: lane a tile row, simdgroup a tile, a threadgroup's simdgroups a K split (grid y) of a 32-row block
// (grid x); the split's sums its share of `partials`.
kernel void streamed_product(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                             device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                             device const half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                             constant uint &tiles [[buffer(6)]], constant uint &rows [[buffer(7)]],
                             constant uint &stride [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]],
                             uint sgs [[simdgroups_per_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
    threadgroup float sums[16][32];
    const Coded c = coded(bits, widths, steps, offsets, tiles);
    const uint t = group.y * sgs + simd, at = group.x * tiles + min(t, tiles - 1);
    half2 x[16];
    device const half2 *xp = (device const half2 *)(xs + min(t, tiles - 1) * 32);
    for (uint j = 0; j < 16; j++) x[j] = xp[j];
    float a = 0.0f;
    const uint b = t < tiles ? streamed_pairs(c, at, lane, [&](uint j, float2 d) {
        a = fma(d.x, float(x[j].x), fma(d.y, float(x[j].y), a));
    }) : 0u;
    sums[simd][lane] = b ? a * 0.5f * steps[at] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < 32; i += tpg.x) {
        float total = 0.0f;
        for (uint s = 0; s < sgs; s++) total += sums[s][i];
        partials[ulong(group.y) * stride + group.x * 32 + i] = total;
    }
}

#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Up to M input rows on the matrix units: streamed_panels feeding one threadgroup matmul2d (M x 32 x 256) a panel,
// `per` panels a threadgroup along K. Unfinished (finish == 0): the threadgroup's share to `partials`; finished (its
// panels cover K): each value with its row's scale, capped (cap > 0), stored at row order[r] of `ys` (row stride
// `stride`).
template <uint M>
kernel void streamed_panel(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                           device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                           device half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                           constant uint &tiles [[buffer(6)]], constant uint &outputs [[buffer(7)]],
                           constant uint &rows [[buffer(8)]], constant uint &per [[buffer(9)]],
                           constant uint &stride [[buffer(10)]], device half *ys [[buffer(11)]],
                           device const int *order [[buffer(12)]], device const float *rowScale [[buffer(13)]],
                           constant float &cap [[buffer(14)]], constant uint &finish [[buffer(15)]],
                           uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    threadgroup half panel[32 * 264];
    const Coded c = coded(bits, widths, steps, offsets, tiles);
    const uint blocks = (rows + M - 1) / M, row0 = (group.y % blocks) * M, share = group.y / blocks;
    auto X = tensor(xs, dextents<int, 2>{int(tiles * 32), int(rows)}, array<int, 2>{1, int(tiles * 32)});
    auto W = tensor(panel, dextents<int, 2>{256, 32}, array<int, 2>{1, 264});
    constexpr auto descriptor = matmul2d_descriptor(M, 32, 256, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    streamed_panels(c, group.x, share * per, share * per + per, simd, lane, panel, [&](uint t0, threadgroup half *) {
        auto weights = W.slice<256, 32>(0, 0);
        if (row0 + M <= rows && t0 + 8 <= tiles) {
            auto a = X.slice<256, M>(int(t0 * 32), int(row0));
            op.run(a, weights, acc);
        } else {
            auto a = X.slice(int(t0 * 32), int(row0));
            op.run(a, weights, acc);
        }
    });
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto e = acc.get_multidimensional_index(i);
        const uint n = row0 + uint(e[1]), r = group.x * 32 + uint(e[0]);
        if (n >= rows) continue;
        if (finish == 0) { partials[ulong(share) * stride + ulong(n) * outputs + r] = acc[i]; continue; }
        const float value = rowScale[r] * acc[i];
        ys[ulong(n) * stride + uint(order[r])] = half(cap > 0.0f ? tanh(value / cap) * cap : value);
    }
}

template [[host_name("streamed_panel_16")]] [[kernel]] decltype(streamed_panel<16>) streamed_panel<16>;
template [[host_name("streamed_panel_128")]] [[kernel]] decltype(streamed_panel<128>) streamed_panel<128>;
#endif
#endif
