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
// the row's scale applied), whole rows through streamed_rows (partials, or plain values a crossing summed), and a
// product's input is written through streamed_put (a function of the logical position, the column order and scale
// applied). Include after <metal_stdlib>; the tensor-op kernels need Metal 4.
#ifndef TORCH_MESH_STREAMED
#define TORCH_MESH_STREAMED
#include <metal_stdlib>
using namespace metal;

// Every kernel's constants, one block at buffer 15 (a caller sets the fields its kernel reads): tiles a 32-row block
// (columns / 32), a matrix's outputs (rows), input rows, panels a threadgroup, a row or share stride, shares to sum and
// their stride (split), whether a panel finishes, an element count, a width, inputs written, a softcap, an RMS epsilon.
struct streamed_dims { uint tiles, outputs, rows, per, stride, shares, split, finish, count, columns, inputs; float cap, eps; };

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

// The GELU of a gate value times an up value (tanh form, its inner term clamped to [-20, 20]).
static inline float streamed_gelu_of(float g, float u) {
    const float inner = clamp(0.7978845608f * (g + 0.044715f * g * g * g), -20.0f, 20.0f);
    return 0.5f * g * (1.0f + tanh(inner)) * u;
}

static inline float streamed_sum(float s, threadgroup float *partial, uint t, uint threads) {
    s = simd_sum(s);
    if (threads <= 32) return s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((t & 31u) == 0) partial[t >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < threads / 32; i++) total += partial[i];
    return total;
}

// A product's output row n at logical positions [first, first + count) into vals: from its partials (shares > 0:
// coded row r of [first, first + count) lands at order[r] - first, its shares summed and scaled) or, where a crossing
// has summed it into plain values, from `plain` (shares == 0, row stride `rows`). The row order permutes within the
// range. Ends with a threadgroup barrier.
static inline void streamed_rows(device const float *partials, device const half *plain, device const int *order,
                                 device const float *scale, uint n, uint first, uint count, uint rows, uint shares,
                                 uint stride, threadgroup float *vals, uint t, uint threads) {
    for (uint i = t; i < count; i += threads) {
        if (shares == 0) { vals[i] = float(plain[ulong(n) * rows + first + i]); continue; }
        const uint r = first + i;
        vals[uint(order[r]) - first] = streamed_value(partials, ulong(n) * rows + r, shares, stride, scale[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A product's input row n: coded column j is scale[j] * f(order[j]), f a value at a logical position.
template <typename F>
static inline void streamed_put(device half *input, device const int *order, device const float *scale, uint n,
                                uint columns, uint t, uint threads, F f) {
    for (uint j = t; j < columns; j += threads) input[ulong(n) * columns + j] = half(scale[j] * f(uint(order[j])));
}

// A product's outputs finished into plain values: y[n * stride + order[r]], capped (cap > 0).
kernel void streamed_finish(device const float *partials [[buffer(0)]], device half *y [[buffer(1)]],
                            device const int *order [[buffer(2)]], device const float *scale [[buffer(3)]],
                            constant streamed_dims &d [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    const uint n = i / d.outputs, r = i % d.outputs;
    const float value = streamed_value(partials, i, d.shares, d.split, scale[r]);
    y[n * d.stride + uint(order[r])] = half(d.cap > 0.0f ? tanh(value / d.cap) * d.cap : value);
}

// The coded inputs of up to three products sharing a column order, from plain rows x (grid: columns x rows).
kernel void streamed_gather(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                            device const int *order [[buffer(2)]], device const float *scale [[buffer(3)]],
                            device half *second [[buffer(5)]], device half *third [[buffer(6)]],
                            device const float *secondScale [[buffer(7)]], device const float *thirdScale [[buffer(8)]],
                            constant streamed_dims &d [[buffer(15)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x >= d.columns) return;
    const ulong at = ulong(i.y) * d.columns + i.x;
    const float v = float(x[ulong(i.y) * d.columns + uint(order[i.x])]);
    first[at] = half(scale[i.x] * v);
    if (d.inputs > 1) second[at] = half(secondScale[i.x] * v);
    if (d.inputs > 2) third[at] = half(thirdScale[i.x] * v);
}

// The coded inputs of up to two products of a rotated basis (int8 rotation stored transposed: row c rotated
// coordinate c, scaled by rotationScale[c]); with eps > 0 the RMS norm of x times gamma folds in (the rotation is
// linear). Four coded columns a simdgroup, 16 bytes a load.
kernel void streamed_rotate(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                            device half *second [[buffer(2)]], device const int *order [[buffer(3)]],
                            device const float *scale [[buffer(4)]], device const float *secondScale [[buffer(5)]],
                            device const char *rotation [[buffer(6)]], device const float *rotationScale [[buffer(7)]],
                            device const half *gamma [[buffer(11)]], constant streamed_dims &d [[buffer(15)]],
                            uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint C = 4;
    const uint columns = d.columns, rows = d.rows, outputs = d.inputs;
    const float eps = d.eps;
    const uint j0 = (group * 4 + simd) * C;
    if (j0 >= columns) return;
    device const half4 *gamma4 = (device const half4 *)gamma;
    const uint words = columns / 16;
    for (uint n = 0; n < rows; n++) {
        device const half4 *xr = (device const half4 *)(x + ulong(n) * columns);
        float acc[C];
        for (uint c = 0; c < C; c++) acc[c] = 0.0f;
        float squares = 0.0f;
        for (uint w = lane; w < words; w += 32) {
            float4 xv[4];
            for (uint k = 0; k < 4; k++) {
                xv[k] = float4(xr[4 * w + k]);
                if (eps > 0.0f) { squares += dot(xv[k], xv[k]); xv[k] *= float4(gamma4[4 * w + k]); }
            }
            for (uint c = 0; c < C; c++) {
                const uint4 bits = ((device const uint4 *)(rotation + ulong(order[min(j0 + c, columns - 1)]) * columns))[w];
                acc[c] += dot(float4(as_type<char4>(bits.x)), xv[0]) + dot(float4(as_type<char4>(bits.y)), xv[1])
                        + dot(float4(as_type<char4>(bits.z)), xv[2]) + dot(float4(as_type<char4>(bits.w)), xv[3]);
            }
        }
        const float rms = eps > 0.0f ? rsqrt(simd_sum(squares) / float(columns) + eps) : 1.0f;
        for (uint c = 0; c < C; c++) {
            const float total = simd_sum(acc[c]) * rms;
            const uint j = j0 + c;
            if (lane == 0 && j < columns) {
                const float value = total * rotationScale[order[j]];
                first[ulong(n) * columns + j] = half(scale[j] * value);
                if (outputs > 1) second[ulong(n) * columns + j] = half(secondScale[j] * value);
            }
        }
    }
}

// The GELU FFN's middle: hidden = GELU(gate) * up from the gate and up products' partials, in their (shared) coded
// row order, which the down product's columns follow.
kernel void streamed_gelu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                          device const float *gateScale [[buffer(2)]], device const float *upScale [[buffer(3)]],
                          device half *hidden [[buffer(4)]], constant streamed_dims &d [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    const uint r = i % d.outputs;
    hidden[i] = half(streamed_gelu_of(streamed_value(gate, i, d.shares, d.split, gateScale[r]),
                                      streamed_value(up, i, d.shares, d.split, upScale[r])));
}

// The matrix decoded to dense values in its row and column orders: dense[order[r]][columns[j]] = rowScale[r] *
// colScale[j] * value of coded row r, coded column j (a threadgroup a tile, a lane a row).
kernel void streamed_dense(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                           device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                           device half *dense [[buffer(4)]], device const int *order [[buffer(5)]],
                           device const float *rowScale [[buffer(6)]], device const int *columns [[buffer(8)]],
                           device const float *colScale [[buffer(9)]], constant streamed_dims &d [[buffer(15)]],
                           uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint tiles = d.tiles;
    const Coded c = coded(bits, widths, steps, offsets, tiles);
    const uint r = group.x * 32 + lane, at = group.x * tiles + group.y, j0 = group.y * 32;
    const float scale = 0.5f * steps[at] * rowScale[r];
    device half *out = dense + ulong(order[r]) * tiles * 32;
    if (streamed_pairs(c, at, lane, [&](uint j, float2 d) {
            out[columns[j0 + 2 * j]] = half(d.x * scale * colScale[j0 + 2 * j]);
            out[columns[j0 + 2 * j + 1]] = half(d.y * scale * colScale[j0 + 2 * j + 1]);
        }) == 0)
        for (uint j = 0; j < 32; j++) out[columns[j0 + j]] = half(0.0h);
}

// One input row: lane a tile row, simdgroup a tile, a threadgroup's simdgroups a K split (grid y) of a 32-row block
// (grid x); the split's sums its share of `partials`.
kernel void streamed_product(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                             device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                             device const half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                             constant streamed_dims &d [[buffer(15)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]],
                             uint sgs [[simdgroups_per_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
    threadgroup float sums[16][32];
    const uint tiles = d.tiles, stride = d.split;
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
                           device half *ys [[buffer(11)]], device const int *order [[buffer(12)]],
                           device const float *rowScale [[buffer(13)]], constant streamed_dims &d [[buffer(15)]],
                           uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    const uint tiles = d.tiles, outputs = d.outputs, rows = d.rows, per = d.per, finish = d.finish;
    const uint stride = d.finish ? d.stride : d.split;
    const float cap = d.cap;
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
