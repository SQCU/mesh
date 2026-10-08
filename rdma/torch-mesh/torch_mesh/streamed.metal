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

// A kernel's type for the recorder that composes kernels (metal-microbench docs/kernels.md#one-kernel-interface):
// MESH_KERNEL(row) before a kernel declares a threadgroup a row, touching only its own row of every binding.
#ifndef MESH_KERNEL
#define MESH_KERNEL(kind)
#endif

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

// lane's row of tile `at` loaded into w (its words, zero past them); returns the tile's width
static inline uint streamed_words(Coded c, uint at, uint lane, thread uint *w) {
    const uint b = c.widths[at];
    device const uint *source = c.bits + c.offsets[at] + lane * b;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 12; i++) w[i] = i < b ? source[i] : 0u;
    return b;
}

// loaded words of width b, their pairs as odd integers to f(j, pair)
template <typename F>
static inline void streamed_decode(thread const uint *w, uint b, F f) {
#define STREAMED_PAIRS(B) streamed_pairs_of<B>(w, f)
    STREAMED_WIDTHS(STREAMED_PAIRS)
#undef STREAMED_PAIRS
}

// lane's row of tile `at`, its pairs as odd integers to f(j, pair); returns the tile's width (0: nothing decoded)
template <typename F>
static inline uint streamed_pairs(Coded c, uint at, uint lane, F f) {
    uint w[12];
    const uint b = streamed_words(c, at, lane, w);
    if (b) streamed_decode(w, b, f);
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
// [first, last), then f(t0, panel) runs on the whole threadgroup; a panel past the last tile is zero there (its
// consumer reads its operand only within the matrix's width: a tensor operation's slice past it is not clipped).
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

// A product's outputs finished into plain values of T: y[n * stride + order[r]], over the row's factor (inputs > 0: its
// input row scaled by streamed_scale_rows), capped (cap > 0).
template <typename T>
kernel void streamed_finish(device const float *partials [[buffer(0)]], device T *y [[buffer(1)]],
                            device const int *order [[buffer(2)]], device const float *scale [[buffer(3)]],
                            device const float *factor [[buffer(4)]], constant streamed_dims &d [[buffer(15)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    const uint n = i / d.outputs, r = i % d.outputs;
    float value = streamed_value(partials, i, d.shares, d.split, scale[r]);
    if (d.inputs) value /= factor[n];
    y[n * d.stride + uint(order[r])] = T(d.cap > 0.0f ? tanh(value / d.cap) * d.cap : value);
}

template [[host_name("streamed_finish")]] [[kernel]] decltype(streamed_finish<half>) streamed_finish<half>;
template [[host_name("streamed_finish_float")]] [[kernel]] decltype(streamed_finish<float>) streamed_finish<float>;

// Rows x [n][d.columns] of T to fp16, each scaled by a power of two to largest magnitude in [0.5, 1) (exact: a
// product of a row is the scaled row's product over its factor; a backward's gradients lie far below fp16's normal
// range): out the scaled rows, factor[n] the scale. A threadgroup a row.
template <typename T>
kernel void streamed_scale_rows(device const T *x [[buffer(0)]], device half *out [[buffer(1)]],
                                device float *factor [[buffer(2)]], constant streamed_dims &d [[buffer(15)]],
                                uint2 group [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float most[8];
    const ulong base = ulong(group.x) * d.columns;
    float m = 0.0f;
    for (uint j = local.x; j < d.columns; j += 256) m = max(m, fabs(float(x[base + j])));
    m = simd_max(m);
    if (lane == 0) most[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = most[0];
    for (uint s = 1; s < 8; s++) m = max(m, most[s]);
    int e = 0;
    frexp(m, e);
    const float scale = m > 0.0f ? ldexp(1.0f, -e) : 1.0f;
    if (local.x == 0) factor[group.x] = scale;
    for (uint j = local.x; j < d.columns; j += 256) out[base + j] = half(float(x[base + j]) * scale);
}

template [[host_name("streamed_scale_rows_float")]] [[kernel]] decltype(streamed_scale_rows<float>) streamed_scale_rows<float>;
#if __METAL_VERSION__ >= 310
template [[host_name("streamed_finish_bfloat")]] [[kernel]] decltype(streamed_finish<bfloat>) streamed_finish<bfloat>;
template [[host_name("streamed_scale_rows_bfloat")]] [[kernel]] decltype(streamed_scale_rows<bfloat>) streamed_scale_rows<bfloat>;
#endif

// The coded inputs of up to three products sharing a column order, from plain rows x (grid: columns x rows); with
// stride s > 0 written in blocks of s columns, block-major ([columns / s][rows][s], rows = d.rows: a tensor operation's
// operand rows s apart whatever the width).
kernel void streamed_gather(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                            device const int *order [[buffer(2)]], device const float *scale [[buffer(3)]],
                            device half *second [[buffer(5)]], device half *third [[buffer(6)]],
                            device const float *secondScale [[buffer(7)]], device const float *thirdScale [[buffer(8)]],
                            constant streamed_dims &d [[buffer(15)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x >= d.columns) return;
    const ulong at = d.stride ? (ulong(i.x / d.stride) * d.rows + i.y) * d.stride + i.x % d.stride
                              : ulong(i.y) * d.columns + i.x;
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

// Up to R input rows on the ALUs (metal-microbench docs/kernels.md "Against LiteRT's kernels"; the study's direct_mv):
// a simdgroup T tiles of the threadgroup's d.per, every tile's words loaded before anything else, then each row's x
// pairs straight into registers, the producer's pairs consumed by one FMA chain a row; the simdgroups' sums reduced in
// threadgroup memory into partials[share][n][row] (coded row order, the row scale not applied). Rows past d.rows repeat
// the last row's loads (no branch on a load) and are not written.
template <uint R, uint T>
kernel void streamed_direct(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                            device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                            device const half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                            constant streamed_dims &d [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint simds [[simdgroups_per_threadgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    threadgroup float sums[8][R][32];
    const uint tiles = d.tiles, rows = d.rows, first = group.y * d.per, span = min(tiles, first + d.per) - first;
    const Coded c = coded(bits, widths, steps, offsets, tiles);
    uint w[T][12], b[T];
    float step[T];
#pragma clang loop unroll(full)
    for (uint i = 0; i < T; i++) {
        const uint t = simd + i * simds, at = group.x * tiles + first + t;
        b[i] = 0;
        step[i] = 0.0f;
        if (t < span) {
            b[i] = streamed_words(c, at, lane, w[i]);
            step[i] = 0.5f * steps[at];
        }
    }
    half2 x[T][R][16];
#pragma clang loop unroll(full)
    for (uint i = 0; i < T; i++) {
        device const half2 *xp = (device const half2 *)(xs + (first + min(simd + i * simds, span - 1)) * 32);
#pragma clang loop unroll(full)
        for (uint n = 0; n < R; n++)
#pragma clang loop unroll(full)
            for (uint j = 0; j < 16; j++) x[i][n][j] = xp[min(n, rows - 1) * (tiles * 16) + j];
    }
    float acc[R];
#pragma clang loop unroll(full)
    for (uint n = 0; n < R; n++) acc[n] = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < T; i++) {
        if (b[i] == 0) continue;
        float tile[R];
#pragma clang loop unroll(full)
        for (uint n = 0; n < R; n++) tile[n] = 0.0f;
        streamed_decode(w[i], b[i], [&](uint j, float2 v) {
#pragma clang loop unroll(full)
            for (uint n = 0; n < R; n++) tile[n] = fma(v.x, float(x[i][n][j].x), fma(v.y, float(x[i][n][j].y), tile[n]));
        });
#pragma clang loop unroll(full)
        for (uint n = 0; n < R; n++) acc[n] = fma(tile[n], step[i], acc[n]);
    }
    for (uint n = 0; n < R; n++) sums[simd][n][lane] = acc[n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < R * 32; i += simds * 32) {
        if (i / 32 >= rows) continue;
        float total = 0.0f;
        for (uint s = 0; s < simds; s++) total += sums[s][i / 32][i % 32];
        partials[ulong(group.y) * d.split + (i / 32) * d.outputs + group.x * 32 + i % 32] = total;
    }
}

template [[host_name("streamed_direct_1_2")]] [[kernel]] decltype(streamed_direct<1, 2>) streamed_direct<1, 2>;
template [[host_name("streamed_direct_2_1")]] [[kernel]] decltype(streamed_direct<2, 1>) streamed_direct<2, 1>;
template [[host_name("streamed_direct_4_1")]] [[kernel]] decltype(streamed_direct<4, 1>) streamed_direct<4, 1>;
template [[host_name("streamed_direct_8_1")]] [[kernel]] decltype(streamed_direct<8, 1>) streamed_direct<8, 1>;

#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// More than 16 input rows on the matrix units: streamed_panels feeding one threadgroup matmul2d (M x 32 x 256) a panel,
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
    streamed_panels(c, group.x, share * per, share * per + per, simd, lane, panel, [&](uint t0, threadgroup half *p) {
        auto weights = W.slice<256, 32>(0, 0);
        if (t0 + 8 > tiles) {
            const uint width = (tiles - t0) * 32;
            for (uint i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                auto e = acc.get_multidimensional_index(i);
                const uint n = row0 + uint(e[1]);
                if (n >= rows) continue;
                device const half *x = xs + ulong(n) * tiles * 32 + t0 * 32;
                threadgroup const half *w = p + uint(e[0]) * 264;
                float sum = 0.0f;
                for (uint k = 0; k < width; k++) sum = fma(float(x[k]), float(w[k]), sum);
                acc[i] += sum;
            }
        } else if (row0 + M <= rows) {
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

// Up to R input rows on the matrix units a simdgroup (the study's paired_mm): each simdgroup decodes its T tiles of the
// threadgroup's d.per one at a time into its own 32 x 32 scratch (streamed_row) and runs its own R x 32 x 32 tensor
// operation on it, no threadgroup barrier between tiles; the simdgroups' accumulators reduced in threadgroup memory into
// partials[share][n][row] (coded row order, the row scale not applied).
template <uint R, uint T>
kernel void streamed_tiles(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                           device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                           device half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                           constant streamed_dims &d [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint thread_id [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;
    threadgroup half scratch[8][32 * 40];
    threadgroup float reduce[4][R * 32];
    const uint tiles = d.tiles, rows = d.rows, first = group.y * d.per, span = min(tiles, first + d.per) - first;
    const Coded c = coded(bits, widths, steps, offsets, tiles);
    auto X = tensor(xs, dextents<int, 2>{int(tiles * 32), int(rows)}, array<int, 2>{1, int(tiles * 32)});
    auto W = tensor(&scratch[simd][0], dextents<int, 2>{32, 32}, array<int, 2>{1, 40});
    constexpr auto descriptor = matmul2d_descriptor(R, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint i = 0; i < T; i++) {
        const uint t = simd + i * 8;
        if (t >= span) break;
        if (streamed_row(c, group.x * tiles + first + t, lane, 1.0f, (threadgroup half2 *)&scratch[simd][lane * 40]) == 0) continue;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto weights = W.slice<32, 32>(0, 0);
        if (rows >= R) {
            auto a = X.slice<32, R>(int((first + t) * 32), 0);
            op.run(a, weights, acc);
        } else {
            auto a = X.slice(int((first + t) * 32), 0);
            op.run(a, weights, acc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd < 4)
        for (uint i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                auto e = acc.get_multidimensional_index(i);
                reduce[simd][uint(e[1]) * 32 + uint(e[0])] = acc[i];
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd >= 4)
        for (uint i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                auto e = acc.get_multidimensional_index(i);
                reduce[simd - 4][uint(e[1]) * 32 + uint(e[0])] += acc[i];
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < R * 32; i += 256) {
        if (i / 32 >= rows) continue;
        const float total = (reduce[0][i] + reduce[1][i]) + (reduce[2][i] + reduce[3][i]);
        partials[ulong(group.y) * d.split + (i / 32) * d.outputs + group.x * 32 + i % 32] = total;
    }
}

template [[host_name("streamed_tiles_16_4")]] [[kernel]] decltype(streamed_tiles<16, 4>) streamed_tiles<16, 4>;

template [[host_name("streamed_panel_128")]] [[kernel]] decltype(streamed_panel<128>) streamed_panel<128>;

// The transposed product, the same panels consumed the other way: partials[share][n][j] = sum over the coded rows r of
// the share's bands of x[n][r] * value(r, j) (x fp16 in coded row order, its scale applied, band-major: xs
// [outputs / 32][rows][32], so an operand's rows are 32 apart whatever the matrix's height (a tensor operation's
// strides past 2^16 elements corrupt its rows past the first); j a coded column). A threadgroup is a 256-column panel
// (group.x) and M input rows; its bands of 32 coded rows [share * per, share * per + per) are decoded by
// streamed_panels and consumed by a tensor operation, the band's block of xs times the panel. Columns past the matrix's
// are not written.
template <uint M>
kernel void streamed_panel_t(device const uint *bits [[buffer(0)]], device const uchar *widths [[buffer(1)]],
                             device const float *steps [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                             device half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                             constant streamed_dims &d [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                             uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    const uint tiles = d.tiles, outputs = d.outputs, rows = d.rows, per = d.per, columns = tiles * 32;
    threadgroup half panel[32 * 264];
    const Coded c = coded(bits, widths, steps, offsets, tiles);
    const uint blocks = (rows + M - 1) / M, row0 = (group.y % blocks) * M, share = group.y / blocks;
    auto W = tensor(panel, dextents<int, 2>{256, 32}, array<int, 2>{1, 264});
    using Operand = decltype(tensor(xs, dextents<int, 2>{32, int(rows)}, array<int, 2>{1, 32}));
    constexpr auto descriptor = matmul2d_descriptor(M, 256, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    auto acc = op.template get_destination_cooperative_tensor<Operand, decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    const uint last = min(outputs / 32, share * per + per);
    for (uint band = share * per; band < last; band++) {
        auto X = tensor(xs + ulong(band) * rows * 32, dextents<int, 2>{32, int(rows)}, array<int, 2>{1, 32});
        streamed_panels(c, band, group.x, group.x + 1, simd, lane, panel, [&](uint, threadgroup half *) {
            auto weights = W.slice<256, 32>(0, 0);
            if (row0 + M <= rows) {
                auto a = X.slice<32, M>(0, int(row0));
                op.run(a, weights, acc);
            } else {
                auto a = X.slice(0, int(row0));
                op.run(a, weights, acc);
            }
        });
    }
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto e = acc.get_multidimensional_index(i);
        const uint n = row0 + uint(e[1]), j = group.x * 256 + uint(e[0]);
        if (n < rows && j < columns) partials[ulong(share) * d.split + ulong(n) * columns + j] = acc[i];
    }
}

template [[host_name("streamed_panel_t_16")]] [[kernel]] decltype(streamed_panel_t<16>) streamed_panel_t<16>;
template [[host_name("streamed_panel_t_64")]] [[kernel]] decltype(streamed_panel_t<64>) streamed_panel_t<64>;
#endif
#endif
