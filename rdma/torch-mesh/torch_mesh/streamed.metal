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
//   streamed_alu<R, SK, BANDS, J, TILES, WIDTHS, OUT, LUT, F, LOADS, H>   1 to 4 rows on the ALUs: a threadgroup BANDS bands, SK
//       simdgroups a band, a K share of J (TILES / J tiles), F tiles a simdgroup in flight; each lane 8 R independent
//       sums (x times half the step: 4 R loads of x a tile); 1-bit tiles by tables of x sums where LUT (T[t][kb][m] =
//       sum over j in m of x[32 t + kb + 8 j], fp32: streamed_rotate writes them); OUT 0 partials [share][row][output],
//       1 finished; LOADS a tile's loads (fragment_load), every product's; H the sums (fragment_walk: fp32, or fp16 pair
//       sums added into them every H tiles)
//   streamed_couple<R, SK, TILES, WIDTHS, F, APART, LOADS, H>      gate's and up's products together, GELU(gate) * up
//   streamed_coop<M, SK, BANDS, J, TILES, WIDTHS, COOP, LOADS, RING>   8 to 16 rows: a tile decoded exactly into the right-input
//       cooperative tensor of matmul2d<M, 32, 32> at the node's slots (COOP), one operation a tile; RING 1 the study's
//       producer-consumer ring (producer simdgroups decoding into a double-buffered threadgroup ring, consumers multiplying);
//       RING r >= 8 the study's role split by rows, 32 to 256 rows (a prefill chunk's): every consumer multiplying every
//       producer's tile by r rows of its own
//   streamed_panel<M, TILES, WIDTHS, LOADS>, streamed_panel_t<M, TILES, WIDTHS, LOADS>   prefill and the transposed product: eight
//       tiles decoded into threadgroup memory in natural order, one operation a 256-wide panel
//
// and the products' sums reach their own consumers through streamed_value (a K split's or a member's partials summed,
// the row's scale applied), whole rows through streamed_rows (partials, or plain values a crossing summed), and a
// product's input is written through streamed_gather (a row's values at the column order, each column's scale
// applied); streamed_alu and streamed_coop take a consumer of their partials as their last template argument E, an
// epilogue run in the dispatch (streamed_plain: none). Include after <metal_stdlib>; the tensor-op kernels need Metal 4.
// The sizes are template arguments, so a caller appends the explicit instantiations of its shapes (streamed.py instance)
// to this source.
#ifndef TORCH_MESH_STREAMED
#define TORCH_MESH_STREAMED
#include <metal_stdlib>
using namespace metal;

// Every kernel's constants, one block at buffer 15 (a caller sets the fields its kernel reads): tiles a 32-row block
// (columns / 32), a matrix's outputs (rows), input rows, a count per unit (a panel's panels a threadgroup, a Hadamard
// block's columns, a product's epilogue's rows an arrival group: 0 the whole product, metal-microbench
// docs/kernels.md#stacked-products, the epilogues), a row or share stride, shares to sum and their stride (split),
// whether a panel finishes, an element count, a width, inputs written (a product's: the rows of an input slab, below), a
// softcap, an RMS epsilon, a consumer's flags and its layer, and tile words a band (pitch; 0: the matrix's own tiles).
// Sixteen words: the literal block (MESH_LITERAL, below) is four uint4.
struct streamed_dims { uint tiles, outputs, rows, per, stride, shares, split, finish, count, columns, inputs; float cap, eps; uint flags, layer, pitch; };

// A band word: its base in 128-byte units from the codes' start (bits 0-27) and its input slab (bits 28-31). Products
// sharing a column order stack into one matrix (metal-microbench docs/kernels.md#stacked-products: q|k|v, gate|up), one
// dispatch over all their bands; each band reads its own product's coded input, slab s at s * d.inputs rows of the
// input (an export's band words, one matrix's, are slab 0).
static inline uint streamed_base(uint word) { return word & 0x0FFFFFFFu; }
static inline ulong streamed_slab(uint word, uint rows, uint columns) { return ulong(word >> 28) * rows * columns; }

// Literal parts in a library compiled with MESH_LITERAL (metal-microbench docs/kernels.md#one-kernel-interface: the served
// engine's classes): a pipeline specialized at its command's block (function constants 200-203, sixteen words) and lane
// count (204) reads them as literals, so the compiler folds every size; unspecialized, and in a library compiled without
// MESH_LITERAL (torch-mesh's own, LiteRT's stages), the bound block and the builtin.
#if MESH_LITERAL
constant uint4 mesh_literal_0 [[function_constant(200)]];
constant uint4 mesh_literal_1 [[function_constant(201)]];
constant uint4 mesh_literal_2 [[function_constant(202)]];
constant uint4 mesh_literal_3 [[function_constant(203)]];
constant uint mesh_literal_lanes [[function_constant(204)]];
template <typename T> static inline T dims_of(constant T &bound) {
    if (!is_function_constant_defined(mesh_literal_0)) return bound;
    const uint4 words[4] = {mesh_literal_0, mesh_literal_1, mesh_literal_2, mesh_literal_3};
    T made;
    thread uint *w = reinterpret_cast<thread uint *>(&made);
    for (uint i = 0; i < sizeof(T) / 4; i++) w[i] = words[i / 4][i % 4];
    return made;
}
static inline uint lanes_of(uint builtin) { return is_function_constant_defined(mesh_literal_lanes) ? mesh_literal_lanes : builtin; }
// A scalar a kernel binds at buffer N (a size, a count, an origin), specialized as function constant 210 + N where the
// pipeline is its command's literal one, so the compiler folds it as it folds the block's sizes
constant uint mesh_scalar_0 [[function_constant(210)]];
constant uint mesh_scalar_1 [[function_constant(211)]];
constant uint mesh_scalar_2 [[function_constant(212)]];
constant uint mesh_scalar_3 [[function_constant(213)]];
constant uint mesh_scalar_4 [[function_constant(214)]];
constant uint mesh_scalar_5 [[function_constant(215)]];
constant uint mesh_scalar_6 [[function_constant(216)]];
constant uint mesh_scalar_7 [[function_constant(217)]];
constant uint mesh_scalar_8 [[function_constant(218)]];
constant uint mesh_scalar_9 [[function_constant(219)]];
constant uint mesh_scalar_10 [[function_constant(220)]];
constant uint mesh_scalar_11 [[function_constant(221)]];
constant uint mesh_scalar_12 [[function_constant(222)]];
constant uint mesh_scalar_13 [[function_constant(223)]];
constant uint mesh_scalar_14 [[function_constant(224)]];
constant uint mesh_scalar_15 [[function_constant(225)]];
constant uint mesh_scalar_16 [[function_constant(226)]];
constant uint mesh_scalar_17 [[function_constant(227)]];
constant uint mesh_scalar_18 [[function_constant(228)]];
constant uint mesh_scalar_19 [[function_constant(229)]];
constant uint mesh_scalar_20 [[function_constant(230)]];
constant uint mesh_scalar_21 [[function_constant(231)]];
constant uint mesh_scalar_22 [[function_constant(232)]];
constant uint mesh_scalar_23 [[function_constant(233)]];
constant uint mesh_scalar_24 [[function_constant(234)]];
constant uint mesh_scalar_25 [[function_constant(235)]];
constant uint mesh_scalar_26 [[function_constant(236)]];
constant uint mesh_scalar_27 [[function_constant(237)]];
constant uint mesh_scalar_28 [[function_constant(238)]];
constant uint mesh_scalar_29 [[function_constant(239)]];
constant uint mesh_scalar_30 [[function_constant(240)]];
constant uint mesh_scalar_31 [[function_constant(241)]];
#define MESH_SCALAR(N, T, bound) (is_function_constant_defined(mesh_scalar_##N) ? as_type<T>(mesh_scalar_##N) : (T)(bound))
#else
template <typename T> static inline T dims_of(constant T &bound) { return bound; }
static inline uint lanes_of(uint builtin) { return builtin; }
#define MESH_SCALAR(N, T, bound) ((T)(bound))
#endif

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

// Pair J of width B of a lane's loaded planes w, as odd integers: V float2 at every width, half2 exact to width 11.
template <uint B, uint J, typename V = float2, typename Words>
static inline V streamed_pair(Words w) {
    const uint value = streamed_bits<B, J>(w);
    if (B <= 9) return V(as_type<half2>(value | 0x64016401u) - half2(half(1024.0f + float(1u << B))));
    return V(float2(float(value & 0xFFFFu), float(value >> 16)) + float2(1.0f - float(1u << B)));
}

// A product's sum for input row n at coded output row r over its shares (K splits of a dispatch; members' shares after
// a crossing), the row's scale applied: what every consumer of a product reads. P is the partials' pointer type: a
// consumer run inside the product's last arriving threadgroup reads them coherent(device).
template <typename T = float, typename P>
static inline T streamed_value(P partials, ulong at, uint shares, uint stride, T scale) {
    T a0 = T(0.0f), a1 = T(0.0f), a2 = T(0.0f), a3 = T(0.0f);
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

// A product's epilogue (metal-microbench docs/kernels.md#stacked-products, the epilogues): a consumer of the product's
// partials run in its own dispatch, its operand E bound at buffer 16 and its arrival words at buffer 14. After a
// threadgroup has stored its share of output rows [first, first + width), every thread calls E::after with the partials
// (stored through E::partials: coherent(device) where another threadgroup of the dispatch reads them), the rows' order and
// scales, the K shares and E::stage floats of threadgroup memory (the product's own, past its reduction); E decides which
// threadgroup consumes what (the last to arrive of the threadgroups holding the rows it reads). streamed_plain consumes
// nothing: the partials are the product's output.
struct streamed_plain {
    enum : uint { stage = 1 };
    typedef device float *partials;
    template <typename P>
    void after(P, device const int *, device const float *, device atomic_uint *, streamed_dims, uint, uint, uint, threadgroup float *, uint,
               uint) constant {}
};

// The GELU of a gate value times an up value (tanh form, its inner term clamped to [-20, 20]).
static inline float streamed_gelu_of(float g, float u) {
    const float inner = clamp(0.7978845608f * (g + 0.044715f * g * g * g), -20.0f, 20.0f);
    return 0.5f * g * (1.0f + tanh(inner)) * u;
}

// The logit softcap: cap tanh(x / cap) where cap > 0, else x.
static inline float streamed_capped(float x, float cap) { return cap > 0.0f ? tanh(x / cap) * cap : x; }

// The threadgroup's sum of each thread's s (its simdgroups' through `partial`); Fresh where nothing reads `partial` before
// the call (the first barrier, which keeps a former sum's readers ahead of the writes, left out).
template <bool Fresh = false>
static inline float streamed_sum(float s, threadgroup float *partial, uint t, uint threads) {
    s = simd_sum(s);
    if (threads <= 32) return s;
    if (!Fresh) threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((t & 31u) == 0) partial[t >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < threads / 32; i++) total += partial[i];
    return total;
}

// The one Metal RMS norm (the host's: torch.nn.functional.rms_norm): the factor of a row of `count` elements whose
// squares the threadgroup's threads hold `s` of, rsqrt(the mean square + eps), the squares summed by streamed_sum.
template <bool Fresh = false>
static inline float streamed_rms(float s, uint count, float eps, threadgroup float *partial, uint t, uint threads) {
    return rsqrt(streamed_sum<Fresh>(s, partial, t, threads) / float(count) + eps);
}

// A partials pointer read four floats at a time, its coherence kept.
template <typename P> struct streamed_fours { typedef device const float4 *type; };
template <> struct streamed_fours<coherent(device) device float *> { typedef coherent(device) device const float4 *type; };

// A product's output row n at logical positions [first, first + count) into vals: from its partials (shares > 0:
// coded row r of [first, first + count) lands at order[r] - first, its shares summed and scaled) or, where a crossing
// has summed it into plain values, from `plain` (shares == 0, row stride `rows`). The row order permutes within the
// range. V elements a thread (1, or 4: float4 partials, int4 orders and float4 scales; first, count, rows and stride
// multiples of 4). Ends with a threadgroup barrier.
template <uint V = 1, typename P>
static inline void streamed_rows(P partials, device const half *plain, device const int *order,
                                 device const float *scale, uint n, uint first, uint count, uint rows, uint shares,
                                 uint stride, threadgroup float *vals, uint t, uint threads) {
    if (V == 4) {
        for (uint q = t; q < count / 4; q += threads) {
            const ulong at = (ulong(n) * rows + first) / 4 + q;
            if (shares == 0) { ((threadgroup float4 *)vals)[q] = float4(((device const half4 *)plain)[at]); continue; }
            const int4 o = ((device const int4 *)order)[first / 4 + q] - int(first);
            const float4 v = streamed_value((typename streamed_fours<P>::type)partials, at, shares, stride / 4, ((device const float4 *)scale)[first / 4 + q]);
            vals[o.x] = v.x; vals[o.y] = v.y; vals[o.z] = v.z; vals[o.w] = v.w;
        }
    } else {
        for (uint i = t; i < count; i += threads) {
            if (shares == 0) { vals[i] = float(plain[ulong(n) * rows + first + i]); continue; }
            const uint r = first + i;
            vals[uint(order[r]) - first] = streamed_value(partials, ulong(n) * rows + r, shares, stride, scale[r]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The rotary embedding of channel value x paired with y (channel i with i + dims / 2, `low` for the first half) at
// frequency pair `pair` of a head of `dims` channels: theta^(-2 pair / dims) radians a position. Precise takes the
// precise trigonometry (the drafter's training), else the fast.
template <bool Precise>
static inline float mesh_rope(float x, float y, bool low, uint pair, uint dims, uint position, float theta) {
    const float angle = float(position) * exp2(-float(2 * pair) / float(dims) * log2(theta));
    const float c = Precise ? precise::cos(angle) : cos(angle), sn = Precise ? precise::sin(angle) : sin(angle);
    return low ? x * c - y * sn : x * c + y * sn;
}

// One head of D channels of row n finished (metal-microbench docs/kernels.md#decode-attention, the heads): its values
// from streamed_rows (a product's partials, or plain rows: shares 0, row stride `rows`, from channel `first`), their RMS
// norm (streamed_rms, eps 1e-6), times gamma where given (staged in `gs` before the reductions), the rotary embedding over
// its first `pairs` frequency pairs at `position`, stored to `out` and to `line` (a KV line), each where given.
template <bool Precise, typename P>
static inline void mesh_head(P partials, device const half *plain, device const int *order, device const float *scale, uint n, uint first,
                             uint D, uint rows, uint shares, uint stride, device const half *gamma, uint pairs, uint position, float theta,
                             device half *out, device half *line, threadgroup float *vals, threadgroup float *gs, threadgroup float *partial,
                             uint t, uint threads) {
    for (uint i = t; i < D; i += threads) gs[i] = gamma ? float(gamma[i]) : 1.0f;
    streamed_rows(partials, plain, order, scale, n, first, D, rows, shares, stride, vals, t, threads);
    float s = 0.0f;
    for (uint i = t; i < D; i += threads) s += vals[i] * vals[i];
    const float rms = streamed_rms(s, D, 1e-6f, partial, t, threads);
    for (uint i = t; i < D; i += threads) {
        const uint p = i < D / 2 ? i : i - D / 2, other = i < D / 2 ? i + D / 2 : i - D / 2;
        const float x = vals[i] * rms * gs[i];
        const half value = half(p < pairs ? mesh_rope<Precise>(x, vals[other] * rms * gs[other], i < D / 2, p, D, position, theta) : x);
        if (out) out[i] = value;
        if (line) line[i] = value;
    }
}

// A lane's k (kb(l) + 8 j) and rows (8 c(l) + q) in every tile.
static inline uint fragment_kb(uint lane) { return ((lane >> 1) & 3u) | (((lane >> 4) & 1u) << 2); }
static inline uint fragment_c(uint lane) { return (lane & 1u) | (((lane >> 3) & 1u) << 1); }

// The step of a tile word.
static inline float fragment_step(uint word) { return float(as_type<half>(ushort(word >> 16))); }

// A lane's pairs of a tile of width B from its loaded planes w (fragment_load), as odd integers to f(P, pair).
template <uint B, typename W, typename F>
static inline __attribute__((always_inline)) void fragment_values(W w, F f) {
    f(0, streamed_pair<B, 0>(w)); f(1, streamed_pair<B, 1>(w)); f(2, streamed_pair<B, 2>(w)); f(3, streamed_pair<B, 3>(w));
    f(4, streamed_pair<B, 4>(w)); f(5, streamed_pair<B, 5>(w)); f(6, streamed_pair<B, 6>(w)); f(7, streamed_pair<B, 7>(w));
    f(8, streamed_pair<B, 8>(w)); f(9, streamed_pair<B, 9>(w)); f(10, streamed_pair<B, 10>(w)); f(11, streamed_pair<B, 11>(w));
    f(12, streamed_pair<B, 12>(w)); f(13, streamed_pair<B, 13>(w)); f(14, streamed_pair<B, 14>(w)); f(15, streamed_pair<B, 15>(w));
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

// The widest width of a WIDTHS mask.
#define FRAGMENT_MAX(W) ((W) >= 4096u ? 12u : (W) >= 2048u ? 11u : (W) >= 1024u ? 10u : (W) >= 512u ? 9u : (W) >= 256u ? 8u : \
    (W) >= 128u ? 7u : (W) >= 64u ? 6u : (W) >= 32u ? 5u : (W) >= 16u ? 4u : (W) >= 8u ? 3u : (W) >= 4u ? 2u : 1u)

// A tile's B planes at `at` into w, a straight run of loads.
template <uint B, uint N>
static inline __attribute__((always_inline)) void fragment_run(device const uint *at, thread uint (&w)[N]) {
    for (uint i = 0; i < B; i++) w[i] = at[32 * i];
}

// A tile's planes at `at`, width b, into w (metal-microbench docs/kernels.md#stacked-products, the loads): LOADS 1 the
// widest width's loads, each predicated on the tile's width, issued before the width's decode is chosen; LOADS 0 a
// straight run of b loads in a switch on the width (a decode's own case where the decode follows at once). Which a
// product takes is its instance's (the fit's loads factor).
template <uint WIDTHS, uint LOADS, uint N>
static inline __attribute__((always_inline)) void fragment_load(device const uint *at, uint b, thread uint (&w)[N]) {
    if (LOADS) {
        for (uint i = 0; i < N; i++) if (i < b) w[i] = at[32 * i];
        return;
    }
#define FRAGMENT_RUN(B) fragment_run<B>(at, w)
    FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_RUN)
#undef FRAGMENT_RUN
}

// acc[n][p] += pair P = 4p + j of loaded words w times x[n][j] (x already times half the step): rows 2p and 2p + 1 of
// the lane's 8 one multiply-add of V, componentwise, float2 the fp32 sums (each component the scalar fma's bits), half2
// the fp16 pair sums (metal-microbench docs/kernels.md#the-instance-fit, forms).
template <uint B, uint R, typename V, typename S, uint N>
static inline __attribute__((always_inline)) void fragment_fma(thread const uint (&w)[N], thread const S (&x)[R][4], thread V (&acc)[R][4]) {
#define FRAGMENT_PAIR(P) { const V v = streamed_pair<B, P, V>(w); \
        for (uint n = 0; n < R; n++) acc[n][P >> 2] = fma(v, V(x[n][P & 3u]), acc[n][P >> 2]); }
    FRAGMENT_PAIR(0) FRAGMENT_PAIR(1) FRAGMENT_PAIR(2) FRAGMENT_PAIR(3) FRAGMENT_PAIR(4) FRAGMENT_PAIR(5) FRAGMENT_PAIR(6)
    FRAGMENT_PAIR(7) FRAGMENT_PAIR(8) FRAGMENT_PAIR(9) FRAGMENT_PAIR(10) FRAGMENT_PAIR(11) FRAGMENT_PAIR(12) FRAGMENT_PAIR(13)
    FRAGMENT_PAIR(14) FRAGMENT_PAIR(15)
#undef FRAGMENT_PAIR
}

// The fp16 pair sums t added into the fp32 sums acc and cleared (every H tiles of a walk).
template <uint R>
static inline __attribute__((always_inline)) void fragment_flush(thread float2 (&acc)[R][4], thread half2 (&t)[R][4]) {
    for (uint n = 0; n < R; n++)
        for (uint p = 0; p < 4; p++) {
            acc[n][p] += float2(t[n][p]);
            t[n][p] = half2(0.0h);
        }
}

// A 1-bit tile's word w by its tables (T the lane's 16 sums of its tile column, row n at T + n stride): row 2p's
// nibble is its four codes at bits 4p of the low half (row 2p + 1 the high half's), the tile's sum step (T[m] - T[15] / 2).
template <uint R>
static inline __attribute__((always_inline)) void fragment_lut(uint w, device const float *T, uint stride, float step,
                                                               thread float2 (&acc)[R][4], thread float (&corr)[R]) {
    for (uint n = 0; n < R; n++) {
        device const float *t = T + n * stride;
        corr[n] = fma(0.5f * step, t[15], corr[n]);
        for (uint p = 0; p < 4; p++)
            acc[n][p] = fma(float2(t[(w >> (4 * p)) & 15u], t[(w >> (16 + 4 * p)) & 15u]), float2(step), acc[n][p]);
    }
}

// The lane's 8 sums of R rows (acc[n][p]: rows 2p and 2p + 1 of its 8) reduced over the 8 lanes holding the same rows
// (kb's bits, lanes 2, 4, 16 apart): the lane's own row of the band, 8 c(l) + 4 l1 + 2 l2 + l4, and its sum of each row n.
template <uint R>
static inline __attribute__((always_inline)) uint fragment_reduce(thread const float2 (&acc)[R][4], uint lane, thread float (&out)[R]) {
    const bool b1 = (lane >> 1) & 1u, b2 = (lane >> 2) & 1u, b4 = (lane >> 4) & 1u;
    for (uint n = 0; n < R; n++) {
        float2 a4[2];
        for (uint i = 0; i < 2; i++) a4[i] = (b1 ? acc[n][i + 2] : acc[n][i]) + simd_shuffle_xor(b1 ? acc[n][i] : acc[n][i + 2], 2);
        const float2 a2 = (b2 ? a4[1] : a4[0]) + simd_shuffle_xor(b2 ? a4[0] : a4[1], 4);
        out[n] = (b4 ? a2.y : a2.x) + simd_shuffle_xor(b4 ? a2.x : a2.y, 16);
    }
    return 8 * fragment_c(lane) + 4 * uint(b1) + 2 * uint(b2) + uint(b4);
}

// A tile's x element type: x times half the step in fp16 where its sums are the fp16 pair sums (H > 0), else fp32.
template <uint H> using fragment_x = metal::conditional_t<(H > 0), half, float>;

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

// A tile of word `word` (width b > 0) loaded: its planes into w and x times half its step into x (a 1-bit tile read by
// tables only its word).
template <uint R, uint WIDTHS, uint LUT, uint LOADS, uint N, typename S>
static inline __attribute__((always_inline)) void fragment_fetch(uint word, device const uint *planes, device const half *xs, uint columns,
        uint rows, uint k, thread uint (&w)[N], thread S (&x)[R][4]) {
    const uint b = (word >> 12) & 15u;
    device const uint *at = planes + ((word & 0xFFFu) << 5);
    if (LUT && b == 1) { w[0] = at[0]; return; }
    fragment_load<WIDTHS, LOADS>(at, b, w);
    const S h = S(0.5f * fragment_step(word));
    for (uint n = 0; n < R; n++) {
        device const half *xr = xs + ulong(min(n, rows - 1)) * columns + k;
        for (uint j = 0; j < 4; j++) x[n][j] = S(xr[8 * j]) * h;
    }
}

// A fetched tile (word `word`, width b > 0) into the lane's sums (tables at T, row stride `stride`, where LUT reads its
// 1-bit tiles): fp32, or with H > 0 the fp16 pair sums t (a 1-bit tile by tables into the fp32 sums).
template <uint R, uint WIDTHS, uint LUT, uint H, uint N, typename S>
static inline __attribute__((always_inline)) void fragment_use(uint word, thread const uint (&w)[N], thread const S (&x)[R][4], device const float *T,
        uint stride, thread float2 (&acc)[R][4], thread half2 (&t)[R][4], thread float (&corr)[R]) {
    const uint b = (word >> 12) & 15u;
    if (LUT && b == 1) { fragment_lut<R>(w[0], T, stride, fragment_step(word), acc, corr); return; }
    if constexpr (H > 0) {
#define FRAGMENT_FMA(B) fragment_fma<B, R>(w, x, t)
        FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_FMA)
#undef FRAGMENT_FMA
    } else {
#define FRAGMENT_FMA(B) fragment_fma<B, R>(w, x, acc)
        FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_FMA)
#undef FRAGMENT_FMA
    }
}

// A simdgroup's walk over its tiles of one band (share [first, first + span), tiles part, part + SK, ...), F (1 or 2)
// tiles in flight: their planes and x loaded before either is decoded (F 1: a tile's loads, x and decode in one region of
// its nonzero width, the hfma2 probe's walk: docs/kernels.md#the-decode-steps-couples, the four-row couple). Each lane's
// 8 R sums (acc[n][p]: rows 2p and 2p + 1; the rows past `rows` repeat the last row's loads), fp32, or with H > 0 the fp16
// pair sums added into them every H tiles and at the walk's end.
template <uint R, uint SK, uint STEPS, uint WIDTHS, uint LUT, uint F, uint LOADS, uint H>
static inline __attribute__((always_inline)) void fragment_walk(device const uint *codes, device const uint *words, uint base,
        device const float *tables, device const half *xs, uint columns, uint rows, uint first, uint span, uint part,
        uint lane, thread float2 (&acc)[R][4]) {
    static_assert(H % F == 0 && (H == 0 || FRAGMENT_MAX(WIDTHS) <= 11), "fp16 pair sums: whole steps of F, widths exact in half2");
    constexpr uint N = FRAGMENT_MAX(WIDTHS);
    const uint kb = fragment_kb(lane), held = fragment_words<SK, STEPS>(words, span, part, lane);
    float corr[R];
    half2 t[R][4];
    for (uint n = 0; n < R; n++) {
        corr[n] = 0.0f;
        for (uint p = 0; p < 4; p++) { acc[n][p] = float2(0.0f); t[n][p] = half2(0.0h); }
    }
    device const uint *planes = codes + (base << 5) + lane;
    for (uint i = 0; i < STEPS; i += F) {
        const uint ta = first + part + i * SK, wa = fragment_word<SK, STEPS>(held, words, span, part, i);
        uint w[N];
        fragment_x<H> x[R][4];
        if constexpr (F > 1) {
            const uint tc = ta + SK, wc = fragment_word<SK, STEPS>(held, words, span, part, i + 1);
            uint w1[N];
            fragment_x<H> x1[R][4];
            if ((wa >> 12) & 15u) fragment_fetch<R, WIDTHS, LUT, LOADS>(wa, planes, xs, columns, rows, ta * 32 + kb, w, x);
            if ((wc >> 12) & 15u) fragment_fetch<R, WIDTHS, LUT, LOADS>(wc, planes, xs, columns, rows, tc * 32 + kb, w1, x1);
            if ((wa >> 12) & 15u) fragment_use<R, WIDTHS, LUT, H>(wa, w, x, tables + (ta * 8 + kb) * 16, columns * 4, acc, t, corr);
            if ((wc >> 12) & 15u) fragment_use<R, WIDTHS, LUT, H>(wc, w1, x1, tables + (tc * 8 + kb) * 16, columns * 4, acc, t, corr);
        } else if ((wa >> 12) & 15u) {
            fragment_fetch<R, WIDTHS, LUT, LOADS>(wa, planes, xs, columns, rows, ta * 32 + kb, w, x);
            fragment_use<R, WIDTHS, LUT, H>(wa, w, x, tables + (ta * 8 + kb) * 16, columns * 4, acc, t, corr);
        }
        if constexpr (H > 0)
            if ((i + F) % H == 0 || i + F >= STEPS) fragment_flush<R>(acc, t);
    }
    if (LUT)
        for (uint n = 0; n < R; n++)
            for (uint p = 0; p < 4; p++) acc[n][p] -= float2(corr[n]);
}

// Up to R input rows on the ALUs (docs/kernels.md#fragment-tiles): a threadgroup BANDS bands of SK simdgroups each,
// share group.y of J along K (TILES / J tiles, x and the tables of the matrix's own K), F tiles a simdgroup in flight;
// each band's sums reduced over its
// lanes and simdgroups into partials[share][n][row] (OUT 0: coded row order, the row scale not applied, then consumed by
// the epilogue E) or finished (OUT 1, one share covering K: each with its row's scale, capped where cap > 0, at row
// order[r] of ys, row stride d.stride). Rows past d.rows repeat the last row's loads and are not written. LOADS:
// fragment_load's; H: fragment_walk's (0 the fp32 sums, else the fp16 pair sums added into them every H tiles).
template <uint R, uint SK, uint BANDS, uint J, uint TILES, uint WIDTHS, uint OUT, uint LUT, uint F, uint LOADS, uint H, typename E = streamed_plain>
kernel void streamed_alu(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                         device const uint *bands [[buffer(2)]], device const float *tables [[buffer(3)]],
                         device const half *xs [[buffer(4)]], device float *partials [[buffer(5)]],
                         device half *ys [[buffer(11)]], device const int *order [[buffer(12)]],
                         device const float *rowScale [[buffer(13)]], device atomic_uint *arrivals [[buffer(14)]],
                         constant streamed_dims &dims [[buffer(15)]], constant E &epilogue [[buffer(16)]],
                         uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
    constexpr uint PER = (TILES + J - 1) / J, STEPS = (PER + SK - 1) / SK, COLUMNS = TILES * 32, SUMS = BANDS * SK * R * 32;
    threadgroup float region[SUMS > E::stage ? SUMS : E::stage];
    threadgroup float (*sums)[SK][R][32] = (threadgroup float (*)[SK][R][32])region;
    const typename E::partials stores = (typename E::partials)partials;
    const uint local = simd / SK, part = simd % SK, band = group.x * BANDS + local, rows = d.rows;
    const uint first = group.y * PER, span = min(TILES, first + PER) - first, pitch = d.pitch ? d.pitch : TILES, word = bands[band];
    float2 acc[R][4];
    float out[R];
    fragment_walk<R, SK, STEPS, WIDTHS, LUT, F, LOADS, H>(codes, tiles + band * pitch + first, streamed_base(word), tables,
                                             xs + streamed_slab(word, d.inputs, COLUMNS), COLUMNS, rows, first, span, part, lane, acc);
    const uint row = fragment_reduce<R>(acc, lane, out);
    for (uint n = 0; n < R; n++) sums[local][part][n][row] = out[n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < BANDS * R * 32; i += BANDS * SK * 32) {
        const uint at = i / (R * 32), n = (i / 32) % R, r = i % 32;
        if (n >= rows) continue;
        float total = 0.0f;
        for (uint s = 0; s < SK; s++) total += sums[at][s][n][r];
        const uint output = (group.x * BANDS + at) * 32 + r;
        if (OUT == 0) { stores[ulong(group.y) * d.split + n * d.outputs + output] = total; continue; }
        const float value = rowScale[output] * total;
        ys[ulong(n) * d.stride + uint(order[output])] = half(streamed_capped(value, d.cap));
    }
    if (OUT == 0) epilogue.after(stores, order, rowScale, arrivals, d, group.x * BANDS * 32, BANDS * 32, J, region, thread_id, BANDS * SK * 32);
}

// The gate and up products of up to R input rows together, GELU(gate) * up between them (the study's gate_up_mm): a
// threadgroup a band of 32 hidden rows of both matrices over their whole K (the GELU needs both whole sums): APART 0,
// SK simdgroups along K each walking both matrices in lock step (2F tiles in flight); APART 1, SK simdgroups of each
// matrix (the first SK gate's, the rest up's: one walk's code for both, F tiles in flight). hidden[n][r] =
// GELU(gateScale[r] g) * upScale[r] u, in the hidden's code order (down's coded input). Rows past d.rows repeat the
// last row's loads and are not written. H: fragment_walk's.
template <uint R, uint SK, uint TILES, uint WIDTHS, uint F, uint APART, uint LOADS, uint H>
kernel void streamed_couple(device const uint *gateCodes [[buffer(0)]], device const uint *gateTiles [[buffer(1)]],
                            device const uint *gateBands [[buffer(2)]], device const half *gx [[buffer(4)]],
                            device const uint *upCodes [[buffer(6)]], device const uint *upTiles [[buffer(7)]],
                            device const uint *upBands [[buffer(8)]], device const half *ux [[buffer(9)]],
                            device const float *upScale [[buffer(10)]], device half *hidden [[buffer(11)]],
                            device const float *gateScale [[buffer(13)]], constant streamed_dims &dims [[buffer(15)]],
                            uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
    constexpr uint STEPS = (TILES + SK - 1) / SK, COLUMNS = TILES * 32, THREADS = (APART ? 64 : 32) * SK;
    threadgroup float sums[2][SK][R][32];
    const uint band = group.x, rows = d.rows, pitch = d.pitch ? d.pitch : TILES;
    if (APART) {
        const uint matrix = simd / SK, part = simd % SK;
        const bool up = matrix != 0;
        float2 acc[R][4];
        float out[R];
        fragment_walk<R, SK, STEPS, WIDTHS, 0, F, LOADS, H>(up ? upCodes : gateCodes, (up ? upTiles : gateTiles) + band * pitch,
                                                 streamed_base((up ? upBands : gateBands)[band]), nullptr, up ? ux : gx, COLUMNS,
                                                 rows, 0, TILES, part, lane, acc);
        const uint row = fragment_reduce<R>(acc, lane, out);
        for (uint n = 0; n < R; n++) sums[matrix][part][n][row] = out[n];
    } else {
        const uint kb = fragment_kb(lane);
        device const uint *gw = gateTiles + band * pitch, *uw = upTiles + band * pitch;
        const uint gheld = fragment_words<SK, STEPS>(gw, TILES, simd, lane), uheld = fragment_words<SK, STEPS>(uw, TILES, simd, lane);
        device const uint *gp = gateCodes + (streamed_base(gateBands[band]) << 5) + lane, *up = upCodes + (streamed_base(upBands[band]) << 5) + lane;
        float2 gacc[R][4], uacc[R][4];
        float gcorr[R], ucorr[R], gout[R], uout[R];
        half2 gt[R][4], ut[R][4];
        for (uint n = 0; n < R; n++)
            for (uint p = 0; p < 4; p++) { gacc[n][p] = float2(0.0f); uacc[n][p] = float2(0.0f); gt[n][p] = half2(0.0h); ut[n][p] = half2(0.0h); }
        for (uint i = 0; i < STEPS; i += F) {
            const uint k = (simd + i * SK) * 32 + kb;
            const uint gw0 = fragment_word<SK, STEPS>(gheld, gw, TILES, simd, i), uw0 = fragment_word<SK, STEPS>(uheld, uw, TILES, simd, i);
            uint g[FRAGMENT_MAX(WIDTHS)], u[FRAGMENT_MAX(WIDTHS)];
            fragment_x<H> gxs[R][4], uxs[R][4];
            if ((gw0 >> 12) & 15u) fragment_fetch<R, WIDTHS, 0, LOADS>(gw0, gp, gx, COLUMNS, rows, k, g, gxs);
            if ((uw0 >> 12) & 15u) fragment_fetch<R, WIDTHS, 0, LOADS>(uw0, up, ux, COLUMNS, rows, k, u, uxs);
            if constexpr (F > 1) {
                const uint gw1 = fragment_word<SK, STEPS>(gheld, gw, TILES, simd, i + 1), uw1 = fragment_word<SK, STEPS>(uheld, uw, TILES, simd, i + 1);
                uint g1[FRAGMENT_MAX(WIDTHS)], u1[FRAGMENT_MAX(WIDTHS)];
                fragment_x<H> gxs1[R][4], uxs1[R][4];
                if ((gw1 >> 12) & 15u) fragment_fetch<R, WIDTHS, 0, LOADS>(gw1, gp, gx, COLUMNS, rows, k + 32 * SK, g1, gxs1);
                if ((uw1 >> 12) & 15u) fragment_fetch<R, WIDTHS, 0, LOADS>(uw1, up, ux, COLUMNS, rows, k + 32 * SK, u1, uxs1);
                if ((gw0 >> 12) & 15u) fragment_use<R, WIDTHS, 0, H>(gw0, g, gxs, nullptr, 0, gacc, gt, gcorr);
                if ((uw0 >> 12) & 15u) fragment_use<R, WIDTHS, 0, H>(uw0, u, uxs, nullptr, 0, uacc, ut, ucorr);
                if ((gw1 >> 12) & 15u) fragment_use<R, WIDTHS, 0, H>(gw1, g1, gxs1, nullptr, 0, gacc, gt, gcorr);
                if ((uw1 >> 12) & 15u) fragment_use<R, WIDTHS, 0, H>(uw1, u1, uxs1, nullptr, 0, uacc, ut, ucorr);
            } else {
                if ((gw0 >> 12) & 15u) fragment_use<R, WIDTHS, 0, H>(gw0, g, gxs, nullptr, 0, gacc, gt, gcorr);
                if ((uw0 >> 12) & 15u) fragment_use<R, WIDTHS, 0, H>(uw0, u, uxs, nullptr, 0, uacc, ut, ucorr);
            }
            if constexpr (H > 0)
                if ((i + F) % H == 0 || i + F >= STEPS) { fragment_flush<R>(gacc, gt); fragment_flush<R>(uacc, ut); }
        }
        const uint row = fragment_reduce<R>(gacc, lane, gout);
        fragment_reduce<R>(uacc, lane, uout);
        for (uint n = 0; n < R; n++) { sums[0][simd][n][row] = gout[n]; sums[1][simd][n][row] = uout[n]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < R * 32; i += THREADS) {
        const uint n = i / 32, r = i % 32;
        if (n >= rows) continue;
        float gs = 0.0f, us = 0.0f;
        for (uint s = 0; s < SK; s++) { gs += sums[0][s][n][r]; us += sums[1][s][n][r]; }
        const uint output = band * 32 + r;
        hidden[ulong(n) * d.outputs + output] = half(streamed_gelu_of(gateScale[output] * gs, upScale[output] * us));
    }
}

// The bandwidth read (metal-microbench docs/kernels.md#the-instance-fit, Evidence; streamed.py bandwidth): every thread
// reads its stride of d.count 16-byte words and folds them, storing only where the fold is a value the read's filling never
// folds to, so every read is kept and nothing is written.
kernel void streamed_read(device const uint4 *in [[buffer(0)]], device uint *out [[buffer(1)]], constant streamed_dims &dims [[buffer(15)]],
                          uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    const streamed_dims d = dims_of(dims);
    uint4 acc = 0;
    for (uint i = gid; i < d.count; i += grid) acc ^= in[i];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[gid & 1023u] = acc.x;
}

// The matrix decoded to dense values in its row and column orders: dense[order[r]][columns[j]] = rowScale[r] *
// colScale[j] * value of coded row r, coded column j (a threadgroup a tile).
kernel void streamed_dense(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                           device const uint *bands [[buffer(2)]], device half *dense [[buffer(4)]],
                           device const int *order [[buffer(5)]], device const float *rowScale [[buffer(6)]],
                           device const int *columns [[buffer(8)]], device const float *colScale [[buffer(9)]],
                           constant streamed_dims &dims [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const streamed_dims d = dims_of(dims);
    const uint T = d.tiles, pitch = d.pitch ? d.pitch : T, word = tiles[group.x * pitch + group.y], b = (word >> 12) & 15u;
    const uint row0 = group.x * 32 + 8 * fragment_c(lane), k0 = group.y * 32 + fragment_kb(lane), width = T * 32;
    const float h = 0.5f * fragment_step(word);
    device const uint *at = codes + ((streamed_base(bands[group.x]) + (word & 0xFFFu)) << 5) + lane;
    auto put = [&](uint P, float2 v) {
        const uint r = row0 + 2 * (P >> 2), k = k0 + 8 * (P & 3u);
        dense[ulong(order[r]) * width + uint(columns[k])] = half(v.x * h * rowScale[r] * colScale[k]);
        dense[ulong(order[r + 1]) * width + uint(columns[k])] = half(v.y * h * rowScale[r + 1] * colScale[k]);
    };
    if (b == 0) { for (uint P = 0; P < 16; P++) put(P, float2(0.0f)); return; }
    uint w[12];
    fragment_load<0x1FFEu, 1>(at, b, w);
#define FRAGMENT_DENSE(B) fragment_values<B>(w, put)
    FRAGMENT_WIDTHS(0x1FFEu, FRAGMENT_DENSE)
#undef FRAGMENT_DENSE
}

// A product's outputs finished into plain values of T: y[n * stride + order[r]], over the row's factor (inputs > 0: its
// input row scaled by streamed_scale_rows), capped (cap > 0).
template <typename T>
kernel void streamed_finish(device const float *partials [[buffer(0)]], device T *y [[buffer(1)]],
                            device const int *order [[buffer(2)]], device const float *scale [[buffer(3)]],
                            device const float *factor [[buffer(4)]], constant streamed_dims &dims [[buffer(15)]],
                            uint i [[thread_position_in_grid]]) {
    const streamed_dims d = dims_of(dims);
    if (i >= d.count) return;
    const uint n = i / d.outputs, r = i % d.outputs;
    float value = streamed_value(partials, i, d.shares, d.split, scale[r]);
    if (d.inputs) value /= factor[n];
    y[n * d.stride + uint(order[r])] = T(streamed_capped(value, d.cap));
}

template [[host_name("streamed_finish")]] [[kernel]] decltype(streamed_finish<half>) streamed_finish<half>;
template [[host_name("streamed_finish_float")]] [[kernel]] decltype(streamed_finish<float>) streamed_finish<float>;

// Rows x [n][d.columns] of T to fp16, each scaled by a power of two to largest magnitude in [0.5, 1) (exact: a
// product of a row is the scaled row's product over its factor; a backward's gradients lie far below fp16's normal
// range): out the scaled rows, factor[n] the scale. A threadgroup a row.
template <typename T>
kernel void streamed_scale_rows(device const T *x [[buffer(0)]], device half *out [[buffer(1)]],
                                device float *factor [[buffer(2)]], constant streamed_dims &dims [[buffer(15)]],
                                uint2 group [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const streamed_dims d = dims_of(dims);
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

// Row n's coded inputs of up to three products sharing a column order (`inputs` of them) from its values (staged in
// threadgroup memory, or a device row read in place) times `factor`: four coded columns a thread, coded column j
// scale[j] * factor * values[order[j]]; with STRIDE s > 0 (a literal: a run-time stride's division cost every row its
// time) written in blocks of s columns, block-major ([columns / s][rows][s]: a tensor operation's operand rows s apart
// whatever the width). A row kernel ending in its consumers' inputs (metal-microbench docs/kernels.md#stacked-products,
// the rows) calls it with its own staged row.
template <uint STRIDE, typename V>
static inline void streamed_gather(V values, float factor, device const int *order, device half *first, device const float *scale,
                                   device half *second, device const float *secondScale, device half *third,
                                   device const float *thirdScale, uint inputs, uint columns, uint rows, uint n, uint t, uint threads) {
    for (uint q = t; q < columns / 4; q += threads) {
        const int4 o = ((device const int4 *)order)[q];
        const float4 v = factor * float4(float(values[o.x]), float(values[o.y]), float(values[o.z]), float(values[o.w]));
        const ulong at = STRIDE ? (ulong(4 * q / max(STRIDE, 1u)) * rows + n) * STRIDE + 4 * q % max(STRIDE, 1u) : ulong(n) * columns + 4 * q;
        *(device half4 *)(first + at) = half4(((device const float4 *)scale)[q] * v);
        if (inputs > 1) *(device half4 *)(second + at) = half4(((device const float4 *)secondScale)[q] * v);
        if (inputs > 2) *(device half4 *)(third + at) = half4(((device const float4 *)thirdScale)[q] * v);
    }
}

// The coded inputs of up to three products sharing a column order (d.inputs of them) from plain rows x, a threadgroup a
// row (metal-microbench docs/kernels.md#one-kernel-interface, typed rows): with Normed times the row's RMS norm's factor
// (eps d.eps; a norm's gamma folded into the products' column scales: docs/kernels.md#stacked-products); the row staged
// in threadgroup memory where WIDTH > 0 (d.columns at most WIDTH), else read in place (streamed_gather).
template <bool Normed, uint WIDTH, uint STRIDE>
kernel void streamed_row_gather(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                                device const int *order [[buffer(2)]], device const float *scale [[buffer(3)]],
                                device half *second [[buffer(5)]], device half *third [[buffer(6)]],
                                device const float *secondScale [[buffer(7)]], device const float *thirdScale [[buffer(8)]],
                                constant streamed_dims &dims [[buffer(15)]], uint n [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_threadgroup]], uint lanes [[threads_per_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
    const uint threads = lanes_of(lanes);
    threadgroup float values[WIDTH > 0 ? WIDTH : 1];
    threadgroup float partial[32];
    const uint columns = d.columns;
    if (WIDTH == 0) {
        streamed_gather<STRIDE>(x + ulong(n) * columns, 1.0f, order, first, scale, second, secondScale, third, thirdScale, d.inputs, columns, d.rows, n, t, threads);
        return;
    }
    device const half4 *row = (device const half4 *)(x + ulong(n) * columns);
    float s = 0.0f;
    for (uint q = t; q < columns / 4; q += threads) {
        const float4 v = float4(row[q]);
        ((threadgroup float4 *)values)[q] = v;
        s += dot(v, v);
    }
    const float factor = Normed ? streamed_rms(s, columns, d.eps, partial, t, threads) : 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    streamed_gather<STRIDE>((threadgroup const float *)values, factor, order, first, scale, second, secondScale, third, thirdScale, d.inputs, columns, d.rows, n, t, threads);
}

template [[host_name("streamed_gather")]] [[kernel]] decltype(streamed_row_gather<false, 0, 0>) streamed_row_gather<false, 0, 0>;
template [[host_name("streamed_gather_32")]] [[kernel]] decltype(streamed_row_gather<false, 0, 32>) streamed_row_gather<false, 0, 32>;

// The coded inputs of up to two products of a rotated basis (int8, stored in coded column order: row j the basis
// vector of coded column j, scaled by rotationScale[j], as the export writes it); with eps > 0 the RMS norm of x times
// gamma folds in (the rotation is
// linear). Four coded columns a simdgroup, each summed within it, 16 bytes a load: a threadgroup of s simdgroups
// covers 4s columns. With flags bit 0 (eight simdgroups: a threadgroup a tile column t) it also writes the first
// product's tables for 1-bit tiles, tables[n][t][kb][m] = the sum over j in m of its coded input kb + 8 j of the tile
// (fp32, from the fp16 values it stored).
kernel void streamed_rotate(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                            device half *second [[buffer(2)]],
                            device const float *scale [[buffer(4)]], device const float *secondScale [[buffer(5)]],
                            device const char *rotation [[buffer(6)]], device const float *rotationScale [[buffer(7)]],
                            device float *tables [[buffer(8)]],
                            device const half *gamma [[buffer(11)]], constant streamed_dims &dims [[buffer(15)]],
                            uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint simds [[simdgroups_per_threadgroup]],
                            uint thread_id [[thread_index_in_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
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
                const uint4 bits = ((device const uint4 *)(rotation + ulong(min(j0 + c, columns - 1)) * columns))[w];
                acc[c] += dot(float4(as_type<char4>(bits.x)), xv[0]) + dot(float4(as_type<char4>(bits.y)), xv[1])
                        + dot(float4(as_type<char4>(bits.z)), xv[2]) + dot(float4(as_type<char4>(bits.w)), xv[3]);
            }
        }
        const float rms = eps > 0.0f ? streamed_rms(squares, columns, eps, nullptr, lane, 32) : 1.0f;
        for (uint c = 0; c < C; c++) {
            const float total = simd_sum(acc[c]) * rms;
            const uint j = j0 + c;
            if (lane == 0 && j < columns) {
                const float value = total * rotationScale[j];
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
kernel void streamed_hadamard(device const half *x [[buffer(0)]], device half *first [[buffer(1)]],
                              device half *second [[buffer(2)]], device const int *order [[buffer(3)]],
                              device const float *scale [[buffer(4)]], device const float *secondScale [[buffer(5)]],
                              device const char *signs [[buffer(6)]], device const char *mix [[buffer(7)]],
                              device const half *gamma [[buffer(11)]], constant streamed_dims &dims [[buffer(15)]],
                              uint n [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint lanes [[threads_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint simds [[simdgroups_per_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
    const uint threads = lanes_of(lanes);
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
    const float rms = streamed_rms<true>(squares, columns, eps, partial, t, threads);
    const float factor = rsqrt(float(columns)) * (eps > 0.0f ? rms : 1.0f);
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

// The GELU FFN's middle: hidden = GELU(gate) * up in the gate and up products' (shared) coded row order, which the down
// product's columns follow: from their partials (gate's d.shares shares, up's d.inputs, each product its own
// instance's, its row scale applied) or, where they are 0, from their finished rows (T half); a row's values d.stride
// apart (0: d.outputs; gate|up stacked: twice it, up's from gate's d.outputs on), up's finished rows d.columns apart where
// nonzero (a per-layer table's); four outputs a thread.
template <typename T>
kernel void streamed_gelu(device const T *gate [[buffer(0)]], device const T *up [[buffer(1)]],
                          device const float *gateScale [[buffer(2)]], device const float *upScale [[buffer(3)]],
                          device half *hidden [[buffer(4)]], constant streamed_dims &dims [[buffer(15)]],
                          uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
    const uint n = g.y, first = (g.x * 256 + t) * 4;
    const ulong row = ulong(n) * (d.stride ? d.stride : d.outputs);
    for (uint r = first; r < min(first + 4, d.outputs); ++r)
        hidden[ulong(n) * d.outputs + r] = half(streamed_gelu_of(
            d.shares ? streamed_value(gate, row + r, d.shares, d.split, gateScale[r]) : float(gate[row + r]),
            d.inputs ? streamed_value(up, row + r, d.inputs, d.split, upScale[r]) : float(up[(d.columns ? ulong(n) * d.columns : row) + r])));
}

template [[host_name("streamed_gelu")]] [[kernel]] decltype(streamed_gelu<float>) streamed_gelu<float>;
template [[host_name("streamed_gelu_rows")]] [[kernel]] decltype(streamed_gelu<half>) streamed_gelu<half>;

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
template <uint B, uint COOP, typename W, typename T>
static inline __attribute__((always_inline)) void fragment_coop(W w, half h, thread T &rb) {
    fragment_values<B>(w, [&](uint P, float2 v) {
        const half2 value = B <= 9 ? half2(v) * half2(h) : half2(v * float(h));
        rb[fragment_element<COOP>(P)] = value.x;
        rb[fragment_element<COOP>(P) + 1] = value.y;
    });
}

// A tile of width B into a ring slot in natural order, value (k, n) at slot[k * FRAGMENT_RING + n], times half its step
// (fragment_coop's values: the same right input, its slots the canonical lanes').
#define FRAGMENT_RING 40u
template <uint B, typename W>
static inline __attribute__((always_inline)) void fragment_ring(W w, half h, threadgroup half *slot, uint lane) {
    const uint c = fragment_c(lane), kb = fragment_kb(lane);
    fragment_values<B>(w, [&](uint P, float2 v) {
        const half2 value = B <= 9 ? half2(v) * half2(h) : half2(v * float(h));
        *(threadgroup half2 *)(slot + (kb + 8 * (P & 3u)) * FRAGMENT_RING + 8 * c + 2 * (P >> 2)) = value;
    });
}

// A ring producer's step: the tile of word `word` (width b > 0) of the planes at `planes` decoded into its slot.
template <uint WIDTHS, uint LOADS>
static inline __attribute__((always_inline)) void fragment_ring_tile(uint word, device const uint *planes, threadgroup half *slot, uint lane) {
    const uint b = (word >> 12) & 15u;
    device const uint *from = planes + ((word & 0xFFFu) << 5);
    const half h = half(0.5f * fragment_step(word));
    uint w[FRAGMENT_MAX(WIDTHS)];
    if (LOADS) fragment_load<WIDTHS, 1>(from, b, w);
#define FRAGMENT_RING_TILE(B) { if (!LOADS) fragment_run<B>(from, w); fragment_ring<B>(w, h, slot, lane); }
    FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_RING_TILE)
#undef FRAGMENT_RING_TILE
}

// A simdgroup's destination acc into the share's partials at `at` (SK 1: rows below `rows`) or its band's sums in
// threadgroup memory (the rows its slots name: the node's COOP map, or Natural, a ring's natural order).
template <uint SK, uint COOP, bool Natural, typename A, typename P>
static inline void fragment_coop_out(thread A &acc, P stores, ulong at, uint outputs, uint rows, threadgroup float *sums) {
    for (uint i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto e = acc.get_multidimensional_index(i);
            const uint r = Natural ? uint(e[0]) : fragment_row<COOP>(uint(e[0])), n = uint(e[1]);
            if (SK == 1) { if (n < rows) stores[at + n * outputs + r] = acc[i]; }
            else sums[n * 32 + r] = acc[i];
        }
}

// M input rows (8 or 16) on the matrix units a simdgroup: a threadgroup BANDS bands of SK simdgroups each, share group.y
// of J along K; each tile decoded into the right-input cooperative tensor of matmul2d<M, 32, 32> and multiplied by its
// 32 columns of x, one operation a tile; a band's simdgroups' sums reduced in threadgroup memory (SK 1, a simdgroup a band:
// stored from the destination as it lies, the ceiling round's sk form) into partials[share][n][row] (coded row order, the
// row scale not applied), then consumed by the epilogue E. RING 1 (the study's producer-consumer ring, a9d17c7^
// tools/streamed_gemv.metal streamed_gemm; metal-microbench docs/kernels.md#the-decode-steps-couples): 2 SK simdgroups a
// band, the first SK producers decoding step s of their K parts into slot s & 1 of a threadgroup ring in natural order
// (fragment_ring), the other SK consumers multiplying step s - 1's from slot (s - 1) & 1, STEPS + 1 stages of one barrier
// each: the decode of one tile beside the operation of the one before it.
// RING r >= 8 (a power of two to 64 dividing M; the study's role split by rows, a9d17c7^ streamed_gemm with its tokens
// shared among consumers, metal-microbench docs/kernels.md#streaming-the-code-into-the-matmul: at 64 tokens 2 producers
// and 2 consumers of 32 rows, at 128 4 and 4, a band a threadgroup): M rows (32 to 256: a decode's verify rows, a
// prefill chunk's; the product's rows at most M), SK producers and M / r consumers a band, the producers decoding as
// RING 1's, each producer's lane 0 writing its tile's width into `live`, and consumer c multiplying every producer's
// slot of step s - 1 whose tile was nonzero by its rows [c r, c r + r) (matmul2d<r, 32, 32>), the band's K parts summed in
// its destination, stored to the partials as it lies (no reduction in threadgroup memory); each tile decoded once for
// all M rows.
template <uint M, uint SK, uint BANDS, uint J, uint TILES, uint WIDTHS, uint COOP, uint LOADS, uint RING, typename E = streamed_plain>
kernel void streamed_coop(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                          device const uint *bands [[buffer(2)]], device const half *xs [[buffer(4)]],
                          device float *partials [[buffer(5)]], device const int *order [[buffer(12)]],
                          device const float *rowScale [[buffer(13)]], device atomic_uint *arrivals [[buffer(14)]],
                          constant streamed_dims &dims [[buffer(15)]], constant E &epilogue [[buffer(16)]],
                          uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint thread_id [[thread_index_in_threadgroup]]) {
    const streamed_dims d = dims_of(dims);
    using namespace mpp::tensor_ops;
    constexpr bool SPLIT = RING >= 8;
    constexpr uint PER = (TILES + J - 1) / J, STEPS = (PER + SK - 1) / SK, SUMS = SPLIT ? 0 : BANDS * SK * M * 32;
    constexpr uint RR = RING ? RING : 1, GROUP = SPLIT ? SK + M / RR : (RING ? 2 : 1) * SK, OM = SPLIT ? RING : M;
    static_assert(RING <= 1 || (RING >= 8 && RING <= 64 && (RING & (RING - 1)) == 0 && M % RR == 0),
                  "RING 0, 1 (the paired ring) or the role split's consumer rows, a power of two from 8 to 64 dividing M");
    static_assert(BANDS * GROUP <= 32, "a threadgroup of at most 1024 threads");
    static_assert(!SPLIT || 2 * BANDS * SK * (32 * FRAGMENT_RING * 2 + 4) + 4 * E::stage <= 32768, "the role split's ring in 32 KB");
    threadgroup float region[SUMS > E::stage ? SUMS : E::stage];
    threadgroup float (*sums)[SK][M * 32] = (threadgroup float (*)[SK][M * 32])region;
    threadgroup half ring[RING ? 2 : 1][BANDS][SK][RING ? 32 * FRAGMENT_RING : 1];
    const typename E::partials stores = (typename E::partials)partials;
    const uint local = simd / GROUP, role = simd % GROUP, part = role % SK, band = group.x * BANDS + local, rows = d.rows;
    const uint first = group.y * PER, span = min(TILES, first + PER) - first, pitch = d.pitch ? d.pitch : TILES;
    const uint bw = bands[band];
    const ulong at = ulong(group.y) * d.split + band * 32;
    auto X = tensor(const_cast<device half *>(xs) + streamed_slab(bw, d.inputs, TILES * 32), dextents<int, 2>{int(TILES * 32), int(rows)}, array<int, 2>{1, int(TILES * 32)});
    constexpr auto descriptor = matmul2d_descriptor(OM, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;
    device const uint *words = tiles + band * pitch + first, *planes = codes + (streamed_base(bw) << 5) + lane;
    const uint held = !SPLIT || role < SK ? fragment_words<SK, STEPS>(words, span, part, lane) : 0u;
    if constexpr (SPLIT) {
        threadgroup uint live[2][BANDS][SK];
        auto W = tensor(&ring[0][0][0][0], dextents<int, 2>{32, 32}, array<int, 2>{1, int(FRAGMENT_RING)});
        auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
        for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
        const uint row0 = role >= SK ? (role - SK) * RING : 0u;
        for (uint s = 0; s <= STEPS; s++) {
            if (role < SK && s < STEPS) {
                const uint word = fragment_word<SK, STEPS>(held, words, span, part, s), b = (word >> 12) & 15u;
                if (lane == 0) live[s & 1u][local][part] = b;
                if (b != 0) fragment_ring_tile<WIDTHS, LOADS>(word, planes, ring[s & 1u][local][part], lane);
            }
            if (role >= SK && s > 0 && row0 < rows)
                for (uint p = 0; p < SK; p++) {
                    if (live[(s - 1) & 1u][local][p] == 0) continue;
                    auto right = tensor(&ring[(s - 1) & 1u][local][p][0], dextents<int, 2>{32, 32}, array<int, 2>{1, int(FRAGMENT_RING)});
                    const int k = int((first + p + (s - 1) * SK) * 32);
                    if (row0 + RING <= rows) {
                        auto a = X.slice<32, OM>(k, int(row0));
                        op.run(a, right, acc);
                    } else {
                        auto a = X.slice(k, int(row0));
                        op.run(a, right, acc);
                    }
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (role >= SK) fragment_coop_out<1, COOP, true>(acc, stores, at + ulong(row0) * d.outputs, d.outputs, rows - min(row0, rows), nullptr);
    } else if constexpr (RING) {
        auto W = tensor(&ring[0][0][0][0], dextents<int, 2>{32, 32}, array<int, 2>{1, int(FRAGMENT_RING)});
        auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
        for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
        for (uint s = 0; s <= STEPS; s++) {
            if (role < SK && s < STEPS) {
                const uint word = fragment_word<SK, STEPS>(held, words, span, part, s);
                if ((word >> 12) & 15u) fragment_ring_tile<WIDTHS, LOADS>(word, planes, ring[s & 1u][local][part], lane);
            }
            if (role >= SK && s > 0 && (fragment_word<SK, STEPS>(held, words, span, part, s - 1) >> 12) & 15u) {
                auto right = tensor(&ring[(s - 1) & 1u][local][part][0], dextents<int, 2>{32, 32}, array<int, 2>{1, int(FRAGMENT_RING)});
                const int k = int((first + part + (s - 1) * SK) * 32);
                if (rows >= M) {
                    auto a = X.slice<32, M>(k, 0);
                    op.run(a, right, acc);
                } else {
                    auto a = X.slice(k, 0);
                    op.run(a, right, acc);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (role >= SK) fragment_coop_out<SK, COOP, true>(acc, stores, at, d.outputs, rows, sums[local][part]);
    } else {
        auto shape = tensor(const_cast<device half *>(xs), dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
        auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(shape), float>();
        for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
        for (uint i = 0; i < STEPS; i++) {
            const uint t = part + i * SK, word = fragment_word<SK, STEPS>(held, words, span, part, i), b = (word >> 12) & 15u;
            if (b == 0) continue;
            device const uint *from = planes + ((word & 0xFFFu) << 5);
            const half h = half(0.5f * fragment_step(word));
            auto rb = op.template get_right_input_cooperative_tensor<half, half, float>();
            uint w[FRAGMENT_MAX(WIDTHS)];
            if (LOADS) fragment_load<WIDTHS, 1>(from, b, w);
#define FRAGMENT_COOP(B) { if (!LOADS) fragment_run<B>(from, w); fragment_coop<B, COOP>(w, h, rb); }
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
        fragment_coop_out<SK, COOP, false>(acc, stores, at, d.outputs, rows, sums[local][part]);
    }
    if (SK > 1 && !SPLIT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = thread_id; i < BANDS * M * 32; i += BANDS * GROUP * 32) {
            const uint which = i / (M * 32), n = (i / 32) % M, r = i % 32;
            if (n >= rows) continue;
            float total = 0.0f;
            for (uint s = 0; s < SK; s++) total += sums[which][s][n * 32 + r];
            stores[ulong(group.y) * d.split + n * d.outputs + (group.x * BANDS + which) * 32 + r] = total;
        }
    }
    epilogue.after(stores, order, rowScale, arrivals, d, group.x * BANDS * 32, BANDS * 32, J, region, thread_id, BANDS * GROUP * 32);
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
template <uint TILES, uint WIDTHS, bool ROWS, uint LOADS>
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
    uint w[FRAGMENT_MAX(WIDTHS)];
    if (LOADS) fragment_load<WIDTHS, 1>(at, b, w);
#define FRAGMENT_PANEL(B) { if (!LOADS) fragment_run<B>(at, w); fragment_values<B>(w, put); }
    FRAGMENT_WIDTHS(WIDTHS, FRAGMENT_PANEL)
#undef FRAGMENT_PANEL
}

// More than 16 input rows on the matrix units: a threadgroup a band and M input rows (group.y: its blocks of M within
// share group.y / blocks), the band's tiles of the share (d.per panels) eight at a time decoded into threadgroup
// memory and multiplied by one matmul2d (M x 32 x 256). Unfinished (finish 0): the share to partials; finished (its
// panels cover K): each value with its row's scale, capped (cap > 0), at row order[r] of ys (row stride d.stride). A
// last panel past the matrix's width adds its tail element by element (a tensor operation's slice past its operand is
// not clipped).
template <uint M, uint TILES, uint WIDTHS, uint LOADS>
kernel void streamed_panel(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                           device const uint *bands [[buffer(2)]], device const half *xs [[buffer(4)]],
                           device float *partials [[buffer(5)]], device half *ys [[buffer(11)]],
                           device const int *order [[buffer(12)]], device const float *rowScale [[buffer(13)]],
                           constant streamed_dims &dims [[buffer(15)]], uint2 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const streamed_dims d = dims_of(dims);
    using namespace mpp::tensor_ops;
    const uint KN = 40, outputs = d.outputs, rows = d.rows, per = d.per, finish = d.finish, pitch = d.pitch ? d.pitch : TILES;
    const uint stride = finish ? d.stride : d.split;
    const float cap = d.cap;
    threadgroup half panel[256 * 40];
    const uint blocks = (rows + M - 1) / M, row0 = (group.y % blocks) * M, share = group.y / blocks, bw = bands[group.x];
    xs += streamed_slab(bw, d.inputs, TILES * 32);
    // matmul2d takes no const element type: the view is the read-only binding's (xs is never written)
    auto X = tensor(const_cast<device half *>(xs), dextents<int, 2>{int(TILES * 32), int(rows)}, array<int, 2>{1, int(TILES * 32)});
    auto W = tensor(panel, dextents<int, 2>{32, 256}, array<int, 2>{1, int(KN)});
    constexpr auto descriptor = matmul2d_descriptor(M, 32, 256, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    device const uint *words = tiles + group.x * pitch;
    const uint base = streamed_base(bw);
    for (uint p = share * per; p < share * per + per; p++) {
        const uint t0 = p * 8;
        if (t0 >= TILES) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fragment_panel<TILES, WIDTHS, true, LOADS>(codes, words, base, t0, simd, lane, panel, KN);
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
        ys[ulong(n) * stride + uint(order[r])] = half(streamed_capped(value, cap));
    }
}

// The transposed product, the same panels consumed the other way: partials[share][n][j] = sum over the coded rows r of
// the share's bands of x[n][r] * value(r, j) (x fp16 in coded row order, its scale applied, band-major: xs
// [outputs / 32][rows][32], so an operand's rows are 32 apart whatever the matrix's height (a tensor operation's
// strides past 2^16 elements corrupt its rows past the first); j a coded column). A threadgroup is a 256-column panel
// (group.x) and M input rows; its bands [share * per, share * per + per) each decoded into the panel (rows the band's 32)
// and multiplied by the band's block of xs. Columns past the matrix's are not written.
template <uint M, uint TILES, uint WIDTHS, uint LOADS>
kernel void streamed_panel_t(device const uint *codes [[buffer(0)]], device const uint *tiles [[buffer(1)]],
                             device const uint *bands [[buffer(2)]], device const half *xs [[buffer(4)]],
                             device float *partials [[buffer(5)]], constant streamed_dims &dims [[buffer(15)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const streamed_dims d = dims_of(dims);
    using namespace mpp::tensor_ops;
    const uint NK = 264, COLUMNS = TILES * 32, outputs = d.outputs, rows = d.rows, per = d.per, pitch = d.pitch ? d.pitch : TILES;
    threadgroup half panel[32 * 264];
    const uint blocks = (rows + M - 1) / M, row0 = (group.y % blocks) * M, share = group.y / blocks;
    auto W = tensor(panel, dextents<int, 2>{256, 32}, array<int, 2>{1, int(NK)});
    using Operand = decltype(tensor(const_cast<device half *>(xs), dextents<int, 2>{32, int(rows)}, array<int, 2>{1, 32}));
    constexpr auto descriptor = matmul2d_descriptor(M, 256, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    auto acc = op.template get_destination_cooperative_tensor<Operand, decltype(W), float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    const uint last = min(outputs / 32, share * per + per);
    for (uint band = share * per; band < last; band++) {
        auto X = tensor(const_cast<device half *>(xs) + ulong(band) * rows * 32, dextents<int, 2>{32, int(rows)}, array<int, 2>{1, 32});
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fragment_panel<TILES, WIDTHS, false, LOADS>(codes, tiles + band * pitch, streamed_base(bands[band]), group.x * 8, simd, lane, panel, NK);
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
