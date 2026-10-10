// Streamed coded weights: the decompression as a producer whose consumer is an operand (metal-microbench
// docs/kernels.md#fragment-tiles; design/algorithm-sources.md#nnffn: C_ij = F_ij(X_i), Y_j = sum_i C_ij handed to its
// consumer).
//
// fragment-1. A coded matrix is 32 x 32 tiles, each its own width b (bits a value, 0 to 12) and step, in bands of 32
// output rows over all K tiles (band-major: a band's tiles in K order and contiguous). Lane l of a simdgroup holds, of
// every tile, the values W[8c(l) + q][kb(l) + 8j] (q < 8, j < 4; kb(l) = l1 | l2 << 1 | l4 << 2, c(l) = l0 | l3 << 1):
// the lane's elements of the right-input cooperative tensor of matmul2d<M, 32, 32> on the nodes read so far, up to a
// bit permutation of the element index and the output column (streamed_coopmap reads a node's). Its 32 codes are 16
// pairs P = 4p + j, row 8c + 2p in a word's low half and row 8c + 2p + 1 in its high half, both at k = kb + 8j; the low
// halves of the lane's b words concatenate the 16 low codes at bits b P, the high halves the high codes, so one shift
// and mask gives a pair, decoded as odd integers 2v + 1 - 2^b times half the step (OR 0x64016401 and a half2 subtract
// for b <= 9, a float conversion above). A tile of width b is b planes of 128 bytes: lane l's word w at byte 128 w + 4 l
// (one aligned word a lane, one 128-byte line a simdgroup, every width). Its tile word (uint32) holds its offset from
// its band's base in 128-byte units (bits 0-11), its width (12-15) and its step as fp16 (16-31); a band's base (uint32)
// is in 128-byte units from the codes' start. A row slice at 32 rows is a range of bands; a K slice at 32 columns a
// range of each band's tile words (streamed_dims.pitch: tile words a band, the operand's tile words bound at the
// slice's first).
//
// Every product is an instance of one decode with literal sizes, its widths the matrix's (WIDTHS, a mask: bit b for
// width b; a simdgroup-uniform switch over them a tile), its consumer an operand:
//
//   streamed_alu<R, SK, BANDS, J, TILES, WIDTHS, OUT, LUT, F>   1 to 4 rows on the ALUs: a threadgroup BANDS bands, SK
//       simdgroups a band, a K share of J (TILES / J tiles), F tiles a simdgroup in flight; each lane 8 R independent
//       sums (x times half the step: 4 R loads of x a tile); 1-bit tiles by tables of x sums where LUT (T[t][kb][m] =
//       sum over j in m of x[32 t + kb + 8 j], fp32: streamed_rotate writes them); OUT 0 partials [share][row][output],
//       1 finished
//   streamed_couple<R, SK, TILES, WIDTHS, F>                  gate's and up's products together, GELU(gate) * up
//   streamed_coop<M, SK, BANDS, J, TILES, WIDTHS, COOP>       8 to 16 rows: a tile decoded exactly into the right-input
//       cooperative tensor of matmul2d<M, 32, 32> at the node's slots (COOP), one operation a tile
//   streamed_panel<M, TILES, WIDTHS>, streamed_panel_t<M, TILES, WIDTHS>   prefill and the transposed product: eight
//       tiles decoded into threadgroup memory in natural order, one operation a 256-wide panel
//
// and the products' sums reach their own consumers through streamed_value (a K split's or a member's partials summed,
// the row's scale applied), whole rows through streamed_rows (partials, or plain values a crossing summed), and a
// product's input is written through streamed_put (a function of the logical position, the column order and scale
// applied). Include after <metal_stdlib>; the tensor-op kernels need Metal 4. The sizes are template arguments, so a
// caller appends the explicit instantiations of its shapes (streamed_instance) to this source.
#ifndef TORCH_MESH_STREAMED
#define TORCH_MESH_STREAMED
#include <metal_stdlib>
using namespace metal;

// A kernel's type for the recorder that composes kernels (metal-microbench docs/kernels.md#one-kernel-interface):
// MESH_KERNEL(row) before a kernel declares a threadgroup a row, touching only its own row of every binding;
// MESH_KERNEL(product) a K split: threadgroup (x, y) writes share y of output block x (the 32 outputs [32x, 32x + 32) of
// every input row) and nothing else of its output; MESH_KERNEL(finish) a thread an element of a product's output,
// reading only the shares of that element (streamed_block_element enumerates block x's elements).
#ifndef MESH_KERNEL
#define MESH_KERNEL(kind)
#endif

// Every kernel's constants, one block at buffer 15 (a caller sets the fields its kernel reads): tiles a 32-row block
// (columns / 32), a matrix's outputs (rows), input rows, panels a threadgroup, a row or share stride, shares to sum and
// their stride (split), whether a panel finishes, an element count, a width, inputs written, a softcap, an RMS epsilon,
// a consumer's flags and its layer, and tile words a band (pitch; 0: the matrix's own tiles).
struct streamed_dims { uint tiles, outputs, rows, per, stride, shares, split, finish, count, columns, inputs; float cap, eps; uint flags, layer, pitch; };

// A finish's elements of output block x, the shares product threadgroups (x, y) wrote: element k of
// streamed_block_elements(d) is streamed_block_element(x, k, d), its thread position in the finish's grid.
static inline uint streamed_block_elements(constant streamed_dims &d) { return 32u * (d.count / d.outputs); }
static inline uint streamed_block_element(uint x, uint k, constant streamed_dims &d) { return (k / 32u) * d.outputs + 32u * x + k % 32u; }

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

// A product's sum for input row n at coded output row r over its shares (K splits of a dispatch; members' shares after
// a crossing), the row's scale applied: what every consumer of a product reads. P is the partials' pointer type: a
// consumer run inside the product's last arriving threadgroup reads them coherent(device).
template <typename P>
static inline float streamed_value(P partials, ulong at, uint shares, uint stride, float scale) {
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
template <typename P>
static inline void streamed_rows(P partials, device const half *plain, device const int *order,
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

// A lane's k (kb(l) + 8 j) and rows (8 c(l) + q) in every tile.
static inline uint fragment_kb(uint lane) { return ((lane >> 1) & 3u) | (((lane >> 4) & 1u) << 2); }
static inline uint fragment_c(uint lane) { return (lane & 1u) | (((lane >> 3) & 1u) << 1); }

// The step of a tile word.
static inline float fragment_step(uint word) { return float(as_type<half>(ushort(word >> 16))); }

// A lane's pairs of a tile of width B at `at` (its first plane's word), as odd integers to f(P, pair).
template <uint B, typename F>
static inline __attribute__((always_inline)) void fragment_values(device const uint *at, F f) {
    uint w[B];
    for (uint i = 0; i < B; i++) w[i] = at[32 * i];
    f(0, streamed_float<B, 0>(w)); f(1, streamed_float<B, 1>(w)); f(2, streamed_float<B, 2>(w)); f(3, streamed_float<B, 3>(w));
    f(4, streamed_float<B, 4>(w)); f(5, streamed_float<B, 5>(w)); f(6, streamed_float<B, 6>(w)); f(7, streamed_float<B, 7>(w));
    f(8, streamed_float<B, 8>(w)); f(9, streamed_float<B, 9>(w)); f(10, streamed_float<B, 10>(w)); f(11, streamed_float<B, 11>(w));
    f(12, streamed_float<B, 12>(w)); f(13, streamed_float<B, 13>(w)); f(14, streamed_float<B, 14>(w)); f(15, streamed_float<B, 15>(w));
}

// The widths of a WIDTHS mask, a simdgroup-uniform switch: CALL(B) for the tile's width b where the mask holds it.
#define FRAGMENT_WIDTHS(WIDTHS, CALL) \
    switch (b) { \
        case 1: if ((WIDTHS) & 2u) CALL(1); break; case 2: if ((WIDTHS) & 4u) CALL(2); break; \
        case 3: if ((WIDTHS) & 8u) CALL(3); break; case 4: if ((WIDTHS) & 16u) CALL(4); break; \
        case 5: if ((WIDTHS) & 32u) CALL(5); break; case 6: if ((WIDTHS) & 64u) CALL(6); break; \
        case 7: if ((WIDTHS) & 128u) CALL(7); break; case 8: if ((WIDTHS) & 256u) CALL(8); break; \
        case 9: if ((WIDTHS) & 512u) CALL(9); break; case 10: if ((WIDTHS) & 1024u) CALL(10); break; \
        case 11: if ((WIDTHS) & 2048u) CALL(11); break; case 12: if ((WIDTHS) & 4096u) CALL(12); break; \
        default: break; \
    }

// A tile's planes at `at`, width b, into w: a straight run of b loads for each width of WIDTHS.
template <uint WIDTHS>
static inline __attribute__((always_inline)) void fragment_load(device const uint *at, uint b, thread uint (&w)[12]) {
#define FRAGMENT_LOAD(B) for (uint i = 0; i < B; i++) w[i] = at[32 * i]
    FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_LOAD)
#undef FRAGMENT_LOAD
}

// acc[n][2p], acc[n][2p + 1] += pair P = 4p + j of loaded words w times x[n][j] (x already times half the step).
template <uint B, uint R>
static inline __attribute__((always_inline)) void fragment_fma(thread const uint (&w)[12], thread const float (&x)[R][4], thread float (&acc)[R][8]) {
#define FRAGMENT_PAIR(P) { const float2 v = streamed_float<B, P>(w); \
        for (uint n = 0; n < R; n++) { \
            acc[n][2 * (P >> 2)] = fma(v.x, x[n][P & 3u], acc[n][2 * (P >> 2)]); \
            acc[n][2 * (P >> 2) + 1] = fma(v.y, x[n][P & 3u], acc[n][2 * (P >> 2) + 1]); } }
    FRAGMENT_PAIR(0) FRAGMENT_PAIR(1) FRAGMENT_PAIR(2) FRAGMENT_PAIR(3) FRAGMENT_PAIR(4) FRAGMENT_PAIR(5) FRAGMENT_PAIR(6)
    FRAGMENT_PAIR(7) FRAGMENT_PAIR(8) FRAGMENT_PAIR(9) FRAGMENT_PAIR(10) FRAGMENT_PAIR(11) FRAGMENT_PAIR(12) FRAGMENT_PAIR(13)
    FRAGMENT_PAIR(14) FRAGMENT_PAIR(15)
#undef FRAGMENT_PAIR
}

// A 1-bit tile's word w by its tables (T the lane's 16 sums of its tile column, row n at T + n stride): row 2p's
// nibble is its four codes at bits 4p of the low half (row 2p + 1 the high half's), the tile's sum step (T[m] - T[15] / 2).
template <uint R>
static inline __attribute__((always_inline)) void fragment_lut(uint w, device const float *T, uint stride, float step,
                                                               thread float (&acc)[R][8], thread float (&corr)[R]) {
    for (uint n = 0; n < R; n++) {
        device const float *t = T + n * stride;
        corr[n] = fma(0.5f * step, t[15], corr[n]);
        for (uint p = 0; p < 4; p++) {
            acc[n][2 * p] = fma(t[(w >> (4 * p)) & 15u], step, acc[n][2 * p]);
            acc[n][2 * p + 1] = fma(t[(w >> (16 + 4 * p)) & 15u], step, acc[n][2 * p + 1]);
        }
    }
}

// The lane's 8 sums of R rows reduced over the 8 lanes holding the same rows (kb's bits, lanes 2, 4, 16 apart): the
// lane's own row of the band, 8 c(l) + 4 l1 + 2 l2 + l4, and its sum of each row n.
template <uint R>
static inline __attribute__((always_inline)) uint fragment_reduce(thread const float (&acc)[R][8], uint lane, thread float (&out)[R]) {
    const bool b1 = (lane >> 1) & 1u, b2 = (lane >> 2) & 1u, b4 = (lane >> 4) & 1u;
    for (uint n = 0; n < R; n++) {
        float a4[4], a2[2];
        for (uint i = 0; i < 4; i++) a4[i] = (b1 ? acc[n][i + 4] : acc[n][i]) + simd_shuffle_xor(b1 ? acc[n][i] : acc[n][i + 4], 2);
        for (uint i = 0; i < 2; i++) a2[i] = (b2 ? a4[i + 2] : a4[i]) + simd_shuffle_xor(b2 ? a4[i] : a4[i + 2], 4);
        out[n] = (b4 ? a2[1] : a2[0]) + simd_shuffle_xor(b4 ? a2[0] : a2[1], 16);
    }
    return 8 * fragment_c(lane) + 4 * uint(b1) + 2 * uint(b2) + uint(b4);
}

// A tile in flight: its width, step, loaded words and x (times half the step).
template <uint R>
struct fragment_tile {
    uint b;
    float step;
    uint w[12];
    float x[R][4];
};

// The tile words of a simdgroup's walk (steps i at tiles part + i SK below span), lane i holding step i's: one load
// (STEPS <= 32; past that each step loads its own).
template <uint SK, uint STEPS>
static inline uint fragment_words(device const uint *words, uint span, uint part, uint lane) {
    const uint t = part + (STEPS <= 32 ? lane : 0u) * SK;
    return (STEPS <= 32 ? lane < STEPS : lane == 0) && t < span ? words[t] : 0u;
}

template <uint SK, uint STEPS>
static inline uint fragment_word(uint held, device const uint *words, uint span, uint part, uint i) {
    if (STEPS <= 32) return i < STEPS ? simd_shuffle(held, i) : 0u;
    const uint t = part + i * SK;
    return t < span ? words[t] : 0u;
}

// A tile's planes and x loaded (nothing for an empty or absent tile; a 1-bit tile read by tables only its word).
template <uint R, uint WIDTHS, uint LUT>
static inline __attribute__((always_inline)) void fragment_fetch(thread fragment_tile<R> &f, uint word, device const uint *planes,
        device const half *xs, uint columns, uint rows, uint k) {
    const uint b = (word >> 12) & 15u;
    f.b = b;
    f.step = fragment_step(word);
    if (b == 0) return;
    device const uint *at = planes + ((word & 0xFFFu) << 5);
    if (LUT && b == 1) { f.w[0] = at[0]; return; }
    fragment_load<WIDTHS>(at, b, f.w);
    const float h = 0.5f * f.step;
    for (uint n = 0; n < R; n++) {
        device const half *xr = xs + ulong(min(n, rows - 1)) * columns + k;
        for (uint j = 0; j < 4; j++) f.x[n][j] = float(xr[8 * j]) * h;
    }
}

// A fetched tile into the lane's sums (tables at T, row stride `stride`, where LUT reads its 1-bit tiles).
template <uint R, uint WIDTHS, uint LUT>
static inline __attribute__((always_inline)) void fragment_use(thread const fragment_tile<R> &f, device const float *T, uint stride,
        thread float (&acc)[R][8], thread float (&corr)[R]) {
    const uint b = f.b;
    if (LUT && b == 1) { fragment_lut<R>(f.w[0], T, stride, f.step, acc, corr); return; }
#define FRAGMENT_FMA(B) fragment_fma<B, R>(f.w, f.x, acc)
    FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_FMA)
#undef FRAGMENT_FMA
}

// A simdgroup's walk over its tiles of one band (share [first, first + span), tiles part, part + SK, ...), F (1 or 2)
// tiles in flight: their planes and x loaded before either is decoded. Each lane's 8 R sums (the rows past `rows` repeat the last
// row's loads).
template <uint R, uint SK, uint STEPS, uint WIDTHS, uint LUT, uint F>
static inline __attribute__((always_inline)) void fragment_walk(device const uint *codes, device const uint *words, uint base,
        device const float *tables, device const half *xs, uint columns, uint rows, uint first, uint span, uint part,
        uint lane, thread float (&acc)[R][8]) {
    const uint kb = fragment_kb(lane), held = fragment_words<SK, STEPS>(words, span, part, lane);
    float corr[R];
    for (uint n = 0; n < R; n++) { corr[n] = 0.0f; for (uint q = 0; q < 8; q++) acc[n][q] = 0.0f; }
    device const uint *planes = codes + (base << 5) + lane;
    for (uint i = 0; i < STEPS; i += F) {
        const uint ta = first + part + i * SK, tc = ta + SK;
        fragment_tile<R> a, c;
        fragment_fetch<R, WIDTHS, LUT>(a, fragment_word<SK, STEPS>(held, words, span, part, i), planes, xs, columns, rows, ta * 32 + kb);
        if (F > 1)
            fragment_fetch<R, WIDTHS, LUT>(c, fragment_word<SK, STEPS>(held, words, span, part, i + 1), planes, xs, columns, rows, tc * 32 + kb);
        fragment_use<R, WIDTHS, LUT>(a, tables + (ta * 8 + kb) * 16, columns * 4, acc, corr);
        if (F > 1) fragment_use<R, WIDTHS, LUT>(c, tables + (tc * 8 + kb) * 16, columns * 4, acc, corr);
    }
    if (LUT)
        for (uint n = 0; n < R; n++)
            for (uint q = 0; q < 8; q++) acc[n][q] -= corr[n];
}

// Up to R input rows on the ALUs (docs/kernels.md#fragment-tiles): a threadgroup BANDS bands of SK simdgroups each,
// share group.y of J along K (TILES / J tiles, x and the tables of the matrix's own K), F tiles a simdgroup in flight;
// each band's sums reduced over its
// lanes and simdgroups into partials[share][n][row] (OUT 0: coded row order, the row scale not applied) or finished
// (OUT 1, one share covering K: each with its row's scale, capped where cap > 0, at row order[r] of ys, row stride
// d.stride). Rows past d.rows repeat the last row's loads and are not written.
template <uint R, uint SK, uint BANDS, uint J, uint TILES, uint WIDTHS, uint OUT, uint LUT, uint F>
MESH_KERNEL(product)
kernel void streamed_alu(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                         device const uint *bands [[buffer(2)]], device const float *tables [[buffer(3)]],
                         device const half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                         device half *ys [[buffer(11)]], device const int *order [[buffer(12)]],
                         device const float *rowScale [[buffer(13)]], constant streamed_dims &d [[buffer(15)]],
                         uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr uint PER = (TILES + J - 1) / J, STEPS = (PER + SK - 1) / SK, COLUMNS = TILES * 32;
    threadgroup float sums[BANDS][SK][R][32];
    const uint local = simd / SK, part = simd % SK, band = group.x * BANDS + local, rows = d.rows;
    const uint first = group.y * PER, span = min(TILES, first + PER) - first, pitch = d.pitch ? d.pitch : TILES;
    float acc[R][8], out[R];
    fragment_walk<R, SK, STEPS, WIDTHS, LUT, F>(codes, tiles + band * pitch + first, bands[band], tables, xs, COLUMNS, rows,
                                             first, span, part, lane, acc);
    const uint row = fragment_reduce<R>(acc, lane, out);
    for (uint n = 0; n < R; n++) sums[local][part][n][row] = out[n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < BANDS * R * 32; i += BANDS * SK * 32) {
        const uint at = i / (R * 32), n = (i / 32) % R, r = i % 32;
        if (n >= rows) continue;
        float total = 0.0f;
        for (uint s = 0; s < SK; s++) total += sums[at][s][n][r];
        const uint output = (group.x * BANDS + at) * 32 + r;
        if (OUT == 0) { partials[ulong(group.y) * d.split + n * d.outputs + output] = total; continue; }
        const float value = rowScale[output] * total;
        ys[ulong(n) * d.stride + uint(order[output])] = half(d.cap > 0.0f ? tanh(value / d.cap) * d.cap : value);
    }
}

// The gate and up products of up to R input rows together, GELU(gate) * up between them (the study's gate_up_mm): a
// threadgroup a band of 32 hidden rows of both matrices over their whole K (the GELU needs both whole sums), SK
// simdgroups along K, F steps (1 or 2: 2F tiles) a simdgroup in flight: hidden[n][r] = GELU(gateScale[r] g) * upScale[r] u, in the hidden's code order (down's coded
// input). Rows past d.rows repeat the last row's loads and are not written.
template <uint R, uint SK, uint TILES, uint WIDTHS, uint F>
kernel void streamed_couple(device const uint *gateCodes [[buffer(0)]], device const uint *gateTiles [[buffer(1)]],
                            device const uint *gateBands [[buffer(2)]], device const half *gx [[buffer(4)]],
                            device const uint *upCodes [[buffer(6)]], device const uint *upTiles [[buffer(7)]],
                            device const uint *upBands [[buffer(8)]], device const half *ux [[buffer(9)]],
                            device const float *upScale [[buffer(10)]], device half *hidden [[buffer(11)]],
                            device const float *gateScale [[buffer(13)]], constant streamed_dims &d [[buffer(15)]],
                            uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr uint STEPS = (TILES + SK - 1) / SK, COLUMNS = TILES * 32;
    threadgroup float sums[2][SK][R][32];
    const uint band = group.x, rows = d.rows, pitch = d.pitch ? d.pitch : TILES, kb = fragment_kb(lane);
    device const uint *gw = gateTiles + band * pitch, *uw = upTiles + band * pitch;
    const uint gheld = fragment_words<SK, STEPS>(gw, TILES, simd, lane), uheld = fragment_words<SK, STEPS>(uw, TILES, simd, lane);
    device const uint *gp = gateCodes + (gateBands[band] << 5) + lane, *up = upCodes + (upBands[band] << 5) + lane;
    float gacc[R][8], uacc[R][8], gcorr[R], ucorr[R], gout[R], uout[R];
    for (uint n = 0; n < R; n++) for (uint q = 0; q < 8; q++) { gacc[n][q] = 0.0f; uacc[n][q] = 0.0f; }
    for (uint i = 0; i < STEPS; i += F) {
        const uint k = (simd + i * SK) * 32 + kb;
        fragment_tile<R> g, u, g1, u1;
        fragment_fetch<R, WIDTHS, 0>(g, fragment_word<SK, STEPS>(gheld, gw, TILES, simd, i), gp, gx, COLUMNS, rows, k);
        fragment_fetch<R, WIDTHS, 0>(u, fragment_word<SK, STEPS>(uheld, uw, TILES, simd, i), up, ux, COLUMNS, rows, k);
        if (F > 1) {
            fragment_fetch<R, WIDTHS, 0>(g1, fragment_word<SK, STEPS>(gheld, gw, TILES, simd, i + 1), gp, gx, COLUMNS, rows, k + 32 * SK);
            fragment_fetch<R, WIDTHS, 0>(u1, fragment_word<SK, STEPS>(uheld, uw, TILES, simd, i + 1), up, ux, COLUMNS, rows, k + 32 * SK);
        }
        fragment_use<R, WIDTHS, 0>(g, nullptr, 0, gacc, gcorr);
        fragment_use<R, WIDTHS, 0>(u, nullptr, 0, uacc, ucorr);
        if (F > 1) {
            fragment_use<R, WIDTHS, 0>(g1, nullptr, 0, gacc, gcorr);
            fragment_use<R, WIDTHS, 0>(u1, nullptr, 0, uacc, ucorr);
        }
    }
    const uint row = fragment_reduce<R>(gacc, lane, gout);
    fragment_reduce<R>(uacc, lane, uout);
    for (uint n = 0; n < R; n++) { sums[0][simd][n][row] = gout[n]; sums[1][simd][n][row] = uout[n]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < R * 32; i += SK * 32) {
        const uint n = i / 32, r = i % 32;
        if (n >= rows) continue;
        float gs = 0.0f, us = 0.0f;
        for (uint s = 0; s < SK; s++) { gs += sums[0][s][n][r]; us += sums[1][s][n][r]; }
        const uint output = band * 32 + r;
        hidden[ulong(n) * d.outputs + output] = half(streamed_gelu_of(gateScale[output] * gs, upScale[output] * us));
    }
}

// The matrix decoded to dense values in its row and column orders: dense[order[r]][columns[j]] = rowScale[r] *
// colScale[j] * value of coded row r, coded column j (a threadgroup a tile).
kernel void streamed_dense(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                           device const uint *bands [[buffer(2)]], device half *dense [[buffer(4)]],
                           device const int *order [[buffer(5)]], device const float *rowScale [[buffer(6)]],
                           device const int *columns [[buffer(8)]], device const float *colScale [[buffer(9)]],
                           constant streamed_dims &d [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint T = d.tiles, pitch = d.pitch ? d.pitch : T, word = tiles[group.x * pitch + group.y], b = (word >> 12) & 15u;
    const uint row0 = group.x * 32 + 8 * fragment_c(lane), k0 = group.y * 32 + fragment_kb(lane), width = T * 32;
    const float h = 0.5f * fragment_step(word);
    device const uint *at = codes + ((bands[group.x] + (word & 0xFFFu)) << 5) + lane;
    auto put = [&](uint P, float2 v) {
        const uint r = row0 + 2 * (P >> 2), k = k0 + 8 * (P & 3u);
        dense[ulong(order[r]) * width + uint(columns[k])] = half(v.x * h * rowScale[r] * colScale[k]);
        dense[ulong(order[r + 1]) * width + uint(columns[k])] = half(v.y * h * rowScale[r + 1] * colScale[k]);
    };
    if (b == 0) { for (uint P = 0; P < 16; P++) put(P, float2(0.0f)); return; }
#define FRAGMENT_DENSE(B) fragment_values<B>(at, put)
    FRAGMENT_WIDTHS(0x1FFEu, FRAGMENT_DENSE)
#undef FRAGMENT_DENSE
}

// A read of byte ranges (16-byte units: ranges[2i] the first, ranges[2i + 1] the count, d.count ranges of the buffer
// at 0) by the grid's threads, folded into one word stored only where it equals d.flags: a prefetch into the system
// cache, no spin.
kernel void streamed_touch(device const uint4 *data [[buffer(0)]], device const uint *ranges [[buffer(1)]],
                           device uint *sink [[buffer(2)]], constant streamed_dims &d [[buffer(15)]],
                           uint i [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    uint4 folded = uint4(0u);
    for (uint r = 0; r < d.count; r++) {
        const uint first = ranges[2 * r], count = ranges[2 * r + 1];
        for (uint u = i; u < count; u += threads) folded ^= data[first + u];
    }
    const uint word = folded.x ^ folded.y ^ folded.z ^ folded.w;
    if (word == d.flags) sink[0] = word;
}

// A product's outputs finished into plain values of T: y[n * stride + order[r]], over the row's factor (inputs > 0: its
// input row scaled by streamed_scale_rows), capped (cap > 0).
template <typename T>
MESH_KERNEL(finish)
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
// linear). Four coded columns a simdgroup, each summed within it, 16 bytes a load: a threadgroup of s simdgroups
// covers 4s columns. With flags bit 0 (eight simdgroups: a threadgroup a tile column t) it also writes the first
// product's tables for 1-bit tiles, tables[n][t][kb][m] = the sum over j in m of its coded input kb + 8 j of the tile
// (fp32, from the fp16 values it stored).
kernel void streamed_rotate(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                            device half *second [[buffer(2)]], device const int *order [[buffer(3)]],
                            device const float *scale [[buffer(4)]], device const float *secondScale [[buffer(5)]],
                            device const char *rotation [[buffer(6)]], device const float *rotationScale [[buffer(7)]],
                            device float *tables [[buffer(8)]],
                            device const half *gamma [[buffer(11)]], constant streamed_dims &d [[buffer(15)]],
                            uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint simds [[simdgroups_per_threadgroup]],
                            uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr uint C = 4;
    threadgroup float values[32];
    const uint columns = d.columns, rows = d.rows, outputs = d.inputs;
    const float eps = d.eps;
    const uint j0 = (group * simds + simd) * C;
    const bool tabled = (d.flags & 1u) != 0;
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
                const half put = half(scale[j] * value);
                first[ulong(n) * columns + j] = put;
                if (tabled) values[simd * C + c] = float(put);
                if (outputs > 1) second[ulong(n) * columns + j] = half(secondScale[j] * value);
            }
        }
        if (tabled) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_id < 128) {
                const uint kb = thread_id >> 4, m = thread_id & 15u;
                float sum = 0.0f;
                for (uint j = 0; j < 4; j++) if ((m >> j) & 1u) sum += values[kb + 8 * j];
                tables[(ulong(n) * (columns / 32) + group) * 128 + thread_id] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// The coded inputs of up to two products of a randomized Hadamard basis (QuIP#, QuaRot): rotated coordinate c = a N + j
// of T x = (H_K (x) H_N) diag(signs) x / sqrt(K N), H_N Sylvester's and H_K (int8, +-1, K x K, row-major: Paley's for
// E2B's K = 12) mixing the blocks; with eps > 0 the RMS norm of x times gamma folds in, as in streamed_rotate. A
// threadgroup a row, latency-bound, so every dependent load is issued first: the puts' order and scales, H_K's rows as
// sign masks. Simdgroup s takes blocks s, s + S, ...: a lane holds W = N / 32 consecutive values of a block, H_N is
// structured.metal's butterfly with its stages below W in registers and the five above as lane shuffles; then column j
// of the K blocks is read into registers by threads (g, j), g < threads / N, each mixing its K / (threads / N) rows
// into a second region, which the puts read in their column order. Two barriers. N in {32, 64, 128}, K <= 20 and a
// multiple of threads / N, K <= 3 S, columns <= 16 threads and <= 2560.
MESH_KERNEL(row)
kernel void streamed_hadamard(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                              device half *second [[buffer(2)]], device const int *order [[buffer(3)]],
                              device const float *scale [[buffer(4)]], device const float *secondScale [[buffer(5)]],
                              device const char *signs [[buffer(6)]], device const char *mix [[buffer(7)]],
                              device const half *gamma [[buffer(11)]], constant streamed_dims &d [[buffer(15)]],
                              uint n [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint threads [[threads_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint simds [[simdgroups_per_threadgroup]]) {
    constexpr uint P = 16, V = 4, R = 3, KM = 20, QM = 10;
    threadgroup float v[5120];
    threadgroup float partial[32];
    threadgroup uint masks[20];
    const uint columns = d.columns, N = d.per, shift = ctz(N), K = columns >> shift, W = N / 32;
    const float eps = d.eps;
    device const half *row = x + ulong(n) * columns;
    uint place[P];
    float firstScale[P], otherScale[P];
    for (uint p = 0; p < P; p++) {
        const uint j = t + p * threads;
        if (j < columns) { place[p] = uint(order[j]); firstScale[p] = scale[j]; otherScale[p] = secondScale[j]; }
    }
    if (t < K) {
        uint m = 0;
        for (uint b = 0; b < KM; b++) if (b < K && mix[t * K + b] < 0) m |= 1u << b;
        masks[t] = m;
    }
    float squares = 0.0f;
    for (uint q = 0; q < R; q++) {
        const uint b = simd + q * simds;
        if (b >= K) break;
        float r[V];
        for (uint u = 0; u < V; u++) if (u < W) {
            const uint i = (b << shift) + lane * W + u;
            const float value = float(row[i]);
            squares += value * value;
            r[u] = (eps > 0.0f ? value * float(gamma[i]) : value) * float(signs[i]);
        }
        for (uint h = 1; h < V; h <<= 1) if (h < W)
            for (uint u = 0; u < V; u++) if (u < W && !(u & h)) { const float a0 = r[u], a1 = r[u + h]; r[u] = a0 + a1; r[u + h] = a0 - a1; }
        for (uint h = 1; h < 32; h <<= 1)
            for (uint u = 0; u < V; u++) if (u < W) { const float other = simd_shuffle_xor(r[u], h); r[u] = (lane & h) ? other - r[u] : r[u] + other; }
        for (uint u = 0; u < V; u++) if (u < W) v[(b << shift) + lane * W + u] = r[u];
    }
    squares = simd_sum(squares);
    if (lane == 0) partial[simd] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float factor = rsqrt(float(columns));
    if (eps > 0.0f) {
        float total = 0.0f;
        for (uint s = 0; s < simds; s++) total += partial[s];
        factor *= rsqrt(total / float(columns) + eps);
    }
    const uint j = t & (N - 1), g = t >> shift, Q = K / (threads >> shift);
    threadgroup float *mixed = v + columns;
    float column[KM];
    for (uint b = 0; b < KM; b++) if (b < K) column[b] = v[(b << shift) | j];
    for (uint q = 0; q < QM; q++) if (q < Q) {
        const uint m = masks[g * Q + q];
        float sum = 0.0f;
        for (uint b = 0; b < KM; b++) if (b < K) sum += ((m >> b) & 1u) ? -column[b] : column[b];
        mixed[((g * Q + q) << shift) | j] = sum * factor;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < P; p++) {
        const uint at = t + p * threads;
        if (at >= columns) break;
        const float value = mixed[place[p]];
        first[ulong(n) * columns + at] = half(firstScale[p] * value);
        if (d.inputs > 1) second[ulong(n) * columns + at] = half(otherScale[p] * value);
    }
}

// The GELU FFN's middle: hidden = GELU(gate) * up from the gate and up products' partials, in their (shared) coded
// row order, which the down product's columns follow.
MESH_KERNEL(finish)
kernel void streamed_gelu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                          device const float *gateScale [[buffer(2)]], device const float *upScale [[buffer(3)]],
                          device half *hidden [[buffer(4)]], constant streamed_dims &d [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    const uint r = i % d.outputs;
    hidden[i] = half(streamed_gelu_of(streamed_value(gate, i, d.shares, d.split, gateScale[r]),
                                      streamed_value(up, i, d.shares, d.split, upScale[r])));
}

#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// A node's cooperative-tensor slots (COOP, read at load by streamed_coopmap): the right-input element of pair P's low
// code is p0 << COOP[0..2] | p1 << COOP[3..5] | j0 << COOP[6..8] | j1 << COOP[9..11] (its high code the next element),
// and an output column n is canonical row sum over i of bit COOP[12 + 3i..] of n, shifted to bit i.
template <uint COOP>
static inline uint fragment_element(uint P) {
    const uint p = P >> 2, j = P & 3u;
    return ((p & 1u) << (COOP & 7u)) | ((p >> 1) << ((COOP >> 3) & 7u)) | ((j & 1u) << ((COOP >> 6) & 7u)) | ((j >> 1) << ((COOP >> 9) & 7u));
}

template <uint COOP>
static inline uint fragment_row(uint n) {
    return ((n >> ((COOP >> 12) & 7u)) & 1u) | (((n >> ((COOP >> 15) & 7u)) & 1u) << 1) | (((n >> ((COOP >> 18) & 7u)) & 1u) << 2)
         | (((n >> ((COOP >> 21) & 7u)) & 1u) << 3) | (((n >> ((COOP >> 24) & 7u)) & 1u) << 4);
}

// A tile of width B into the right-input cooperative tensor rb, times half its step (exact for a power-of-two step).
template <uint B, uint COOP, typename T>
static inline __attribute__((always_inline)) void fragment_coop(device const uint *at, half h, thread T &rb) {
    fragment_values<B>(at, [&](uint P, float2 v) {
        const half2 value = B <= 9 ? half2(v) * half2(h) : half2(v * float(h));
        rb[fragment_element<COOP>(P)] = value.x;
        rb[fragment_element<COOP>(P) + 1] = value.y;
    });
}

// M input rows (8 or 16) on the matrix units a simdgroup: a threadgroup BANDS bands of SK simdgroups each, share group.y
// of J along K; each tile decoded into the right-input cooperative tensor of matmul2d<M, 32, 32> and multiplied by its
// 32 columns of x, one operation a tile; a band's simdgroups' sums reduced in threadgroup memory into
// partials[share][n][row] (coded row order, the row scale not applied).
template <uint M, uint SK, uint BANDS, uint J, uint TILES, uint WIDTHS, uint COOP>
MESH_KERNEL(product)
kernel void streamed_coop(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                          device const uint *bands [[buffer(2)]], device half *xs [[buffer(4)]],
                          device float *partials [[buffer(5)]], constant streamed_dims &d [[buffer(15)]],
                          uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr uint PER = (TILES + J - 1) / J, STEPS = (PER + SK - 1) / SK;
    threadgroup float sums[BANDS][SK][M * 32];
    const uint local = simd / SK, part = simd % SK, band = group.x * BANDS + local, rows = d.rows;
    const uint first = group.y * PER, span = min(TILES, first + PER) - first, pitch = d.pitch ? d.pitch : TILES;
    auto X = tensor(xs, dextents<int, 2>{int(TILES * 32), int(rows)}, array<int, 2>{1, int(TILES * 32)});
    auto shape = tensor(xs, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    constexpr auto descriptor = matmul2d_descriptor(M, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(shape), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    device const uint *words = tiles + band * pitch + first, *planes = codes + (bands[band] << 5) + lane;
    const uint held = fragment_words<SK, STEPS>(words, span, part, lane);
    for (uint i = 0; i < STEPS; i++) {
        const uint t = part + i * SK, word = fragment_word<SK, STEPS>(held, words, span, part, i), b = (word >> 12) & 15u;
        if (b == 0) continue;
        device const uint *at = planes + ((word & 0xFFFu) << 5);
        const half h = half(0.5f * fragment_step(word));
        auto rb = op.template get_right_input_cooperative_tensor<half, half, float>();
#define FRAGMENT_COOP(B) fragment_coop<B, COOP>(at, h, rb)
        FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_COOP)
#undef FRAGMENT_COOP
        if (rows >= M) {
            auto a = X.slice<32, M>(int((first + t) * 32), 0);
            op.run(a, rb, acc);
        } else {
            auto a = X.slice(int((first + t) * 32), 0);
            op.run(a, rb, acc);
        }
    }
    for (uint i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto e = acc.get_multidimensional_index(i);
            sums[local][part][uint(e[1]) * 32 + fragment_row<COOP>(uint(e[0]))] = acc[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < BANDS * M * 32; i += BANDS * SK * 32) {
        const uint at = i / (M * 32), n = (i / 32) % M, r = i % 32;
        if (n >= rows) continue;
        float total = 0.0f;
        for (uint s = 0; s < SK; s++) total += sums[at][s][n * 32 + r];
        partials[ulong(group.y) * d.split + n * d.outputs + (group.x * BANDS + at) * 32 + r] = total;
    }
}

// The right-input and destination element maps of matmul2d<M, 32, 32> at one simdgroup (fp16 operands, fp32
// destination): out[lane][0] the right input's capacity, then its (n, k) pairs, out[lane][65] the destination's, then
// its (n, row) pairs (n -1 where not valid). One dispatch of 32 threads at load (docs/kernels.md#fragment-tiles).
template <uint M>
kernel void streamed_coopmap(device int *out [[buffer(1)]], device half *xs [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr auto descriptor = matmul2d_descriptor(M, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;
    auto X = tensor(xs, dextents<int, 2>{32, int(M)}, array<int, 2>{1, 32});
    auto W = tensor(xs, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
    auto rb = op.template get_right_input_cooperative_tensor<half, half, float>();
    device int *o = out + lane * 130;
    o[0] = int(rb.get_capacity());
    for (uint i = 0; i < rb.get_capacity() && i < 32; ++i) {
        auto e = rb.get_multidimensional_index(i);
        o[1 + 2 * i] = rb.is_valid_element(i) ? int(e[0]) : -1;
        o[2 + 2 * i] = int(e[1]);
    }
    o[65] = int(acc.get_capacity());
    for (uint i = 0; i < acc.get_capacity() && i < 32; ++i) {
        auto e = acc.get_multidimensional_index(i);
        o[66 + 2 * i] = acc.is_valid_element(i) ? int(e[0]) : -1;
        o[67 + 2 * i] = int(e[1]);
    }
}

template [[host_name("streamed_coopmap_8")]] [[kernel]] decltype(streamed_coopmap<8>) streamed_coopmap<8>;
template [[host_name("streamed_coopmap_16")]] [[kernel]] decltype(streamed_coopmap<16>) streamed_coopmap<16>;

// Eight tiles of band `band`, tiles [t0, t0 + 8) (zero past TILES), decoded by the threadgroup's eight simdgroups into
// panel: natural order with value (k, n) at panel[k * KN + n] (Rows) or at panel[n * NK + k] (transposed), times half
// each tile's step, fp16.
template <uint TILES, uint WIDTHS, bool ROWS>
static inline __attribute__((always_inline)) void fragment_panel(device const uint *codes, device const uint *tiles, uint base,
        uint t0, uint simd, uint lane, threadgroup half *panel, uint stride) {
    const uint t = t0 + simd, c = fragment_c(lane), kb = fragment_kb(lane);
    const uint word = t < TILES ? tiles[t] : 0u, b = (word >> 12) & 15u;
    const float h = 0.5f * fragment_step(word);
    auto put = [&](uint P, float2 v) {
        const uint n = 8 * c + 2 * (P >> 2), k = 32 * simd + kb + 8 * (P & 3u);
        const half2 value = half2(v * h);
        if (ROWS) *(threadgroup half2 *)(panel + k * stride + n) = value;
        else { panel[n * stride + k] = value.x; panel[(n + 1) * stride + k] = value.y; }
    };
    if (b == 0) { for (uint P = 0; P < 16; P++) put(P, float2(0.0f)); return; }
    device const uint *at = codes + ((base + (word & 0xFFFu)) << 5) + lane;
#define FRAGMENT_PANEL(B) fragment_values<B>(at, put)
    FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_PANEL)
#undef FRAGMENT_PANEL
}

// More than 16 input rows on the matrix units: a threadgroup a band and M input rows (group.y: its blocks of M within
// share group.y / blocks), the band's tiles of the share (d.per panels) eight at a time decoded into threadgroup
// memory and multiplied by one matmul2d (M x 32 x 256). Unfinished (finish 0): the share to partials; finished (its
// panels cover K): each value with its row's scale, capped (cap > 0), at row order[r] of ys (row stride d.stride). A
// last panel past the matrix's width adds its tail element by element (a tensor operation's slice past its operand is
// not clipped).
template <uint M, uint TILES, uint WIDTHS>
kernel void streamed_panel(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                           device const uint *bands [[buffer(2)]], device half *xs [[buffer(4)]],
                           device float *partials [[buffer(5)]], device half *ys [[buffer(11)]],
                           device const int *order [[buffer(12)]], device const float *rowScale [[buffer(13)]],
                           constant streamed_dims &d [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr uint KN = 40;
    const uint outputs = d.outputs, rows = d.rows, per = d.per, finish = d.finish, pitch = d.pitch ? d.pitch : TILES;
    const uint stride = finish ? d.stride : d.split;
    const float cap = d.cap;
    threadgroup half panel[256 * KN];
    const uint blocks = (rows + M - 1) / M, row0 = (group.y % blocks) * M, share = group.y / blocks;
    auto X = tensor(xs, dextents<int, 2>{int(TILES * 32), int(rows)}, array<int, 2>{1, int(TILES * 32)});
    auto W = tensor(panel, dextents<int, 2>{32, 256}, array<int, 2>{1, int(KN)});
    constexpr auto descriptor = matmul2d_descriptor(M, 32, 256, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    device const uint *words = tiles + group.x * pitch;
    const uint base = bands[group.x];
    for (uint p = share * per; p < share * per + per; p++) {
        const uint t0 = p * 8;
        if (t0 >= TILES) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fragment_panel<TILES, WIDTHS, true>(codes, words, base, t0, simd, lane, panel, KN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto weights = W.slice<32, 256>(0, 0);
        if (t0 + 8 > TILES) {
            const uint width = (TILES - t0) * 32;
            for (uint i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                auto e = acc.get_multidimensional_index(i);
                const uint n = row0 + uint(e[1]);
                if (n >= rows) continue;
                device const half *x = xs + ulong(n) * TILES * 32 + t0 * 32;
                float sum = 0.0f;
                for (uint k = 0; k < width; k++) sum = fma(float(x[k]), float(panel[k * KN + uint(e[0])]), sum);
                acc[i] += sum;
            }
        } else if (row0 + M <= rows) {
            auto a = X.slice<256, M>(int(t0 * 32), int(row0));
            op.run(a, weights, acc);
        } else {
            auto a = X.slice(int(t0 * 32), int(row0));
            op.run(a, weights, acc);
        }
    }
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

// The transposed product, the same panels consumed the other way: partials[share][n][j] = sum over the coded rows r of
// the share's bands of x[n][r] * value(r, j) (x fp16 in coded row order, its scale applied, band-major: xs
// [outputs / 32][rows][32], so an operand's rows are 32 apart whatever the matrix's height (a tensor operation's
// strides past 2^16 elements corrupt its rows past the first); j a coded column). A threadgroup is a 256-column panel
// (group.x) and M input rows; its bands [share * per, share * per + per) each decoded into the panel (rows the band's 32)
// and multiplied by the band's block of xs. Columns past the matrix's are not written.
template <uint M, uint TILES, uint WIDTHS>
kernel void streamed_panel_t(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                             device const uint *bands [[buffer(2)]], device half *xs [[buffer(4)]],
                             device float *partials [[buffer(5)]], constant streamed_dims &d [[buffer(15)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr uint NK = 264, COLUMNS = TILES * 32;
    const uint outputs = d.outputs, rows = d.rows, per = d.per, pitch = d.pitch ? d.pitch : TILES;
    threadgroup half panel[32 * NK];
    const uint blocks = (rows + M - 1) / M, row0 = (group.y % blocks) * M, share = group.y / blocks;
    auto W = tensor(panel, dextents<int, 2>{256, 32}, array<int, 2>{1, int(NK)});
    using Operand = decltype(tensor(xs, dextents<int, 2>{32, int(rows)}, array<int, 2>{1, 32}));
    constexpr auto descriptor = matmul2d_descriptor(M, 256, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    auto acc = op.template get_destination_cooperative_tensor<Operand, decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    const uint last = min(outputs / 32, share * per + per);
    for (uint band = share * per; band < last; band++) {
        auto X = tensor(xs + ulong(band) * rows * 32, dextents<int, 2>{32, int(rows)}, array<int, 2>{1, 32});
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fragment_panel<TILES, WIDTHS, false>(codes, tiles + band * pitch, bands[band], group.x * 8, simd, lane, panel, NK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto weights = W.slice<256, 32>(0, 0);
        if (row0 + M <= rows) {
            auto a = X.slice<32, M>(0, int(row0));
            op.run(a, weights, acc);
        } else {
            auto a = X.slice(0, int(row0));
            op.run(a, weights, acc);
        }
    }
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto e = acc.get_multidimensional_index(i);
        const uint n = row0 + uint(e[1]), j = group.x * 256 + uint(e[0]);
        if (n < rows && j < COLUMNS) partials[ulong(share) * d.split + ulong(n) * COLUMNS + j] = acc[i];
    }
}
#endif
#endif
