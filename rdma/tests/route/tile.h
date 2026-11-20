// tile.h: the tiles' shared parts (fwd.metal, bwd.metal). The policy's network on the GPU, built on mm.metal's int8 x int8 -> int32 products (matmul2d), every product's
// epilogue and the next product's quantization fused in, a tile of rows carried on chip through its layers.
//
// The network (route.h): x = inp(token); two minGRU layers, each z, h~ = sigmoid, tanh of zh(x), the recurrence
// h = (1 - m z) h + m z h~ along a run's lanes (scan.h), a context one step from its refresh's state, xm = x + h,
// x' = xm + down(relu(up(xm))) (layer 1's lanes: their states only); w = out(x2 / rms(x2)) a context.
//
// Tiles (the shapes behind every choice, measured by stages.swift):
// - a tile is 64 rows (TM): the matmul tile the roofline sweep found best at these shapes (64 x 64, its products
//   N in 64-column chunks: the Walsh-Hadamard block and the sweep's best N), and the wgrad block (int8mm.h's
//   I8_WBLOCK), so a tile makes its own wgrad codes;
// - eight SIMD groups a tile: the chunked chain of products ran at 45 TOP/s with 8 as with 4 (stages), and 8 halve
//   the registers each thread holds for a row pass (a 64 x 192 output's rotated values: 48 a thread);
// - each product: its chunks' int32 sums stored to threadgroup memory S (64 x 68 words: rows 16-byte aligned, so the
//   cooperative store and the row pass move four words at a time; 65 (an odd stride, no bank conflicts in the column
//   pass) measured 15% slower over the chain of products), then a row pass: eight lanes a row,
//   eight consecutive columns a lane (int8mm.h's i8_fwht64_lanes), two rows a thread; the next product's codes into A.
// - a column pass (the learning calls): the same lanes over a chunk's columns, eight rows a lane: the wgrad codes of
//   the value just made, rotated along the rows, one scale a column of the tile (I8_WBLOCK), stored transposed (each
//   column's rows contiguous, as the wgrad product reads them); a gradient's column sums (its bias's gradient) beside.
//
// Rows: the lanes' ([run][pad64(K L)], a run's positions in order, its pad rows masked) then the contexts'. The
// learning calls' backward: three tasks, each its own threadgroups.
#ifndef ROUTE_TILE_H
#define ROUTE_TILE_H
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#include "int8mm.h"
#include "scan.h"
#include "split.h"
#include "upgrad.h"
#include "act.h"

enum : uint { D = 64, E = 192, IN = 48, FS = 40, L = 32, TM = 64, NSG = 8, NT = NSG * 32, LD = 68 };

// the matrices, in NetLayout's order
enum : uint { M_INP = 0, M_ZH0 = 1, M_UP0 = 2, M_DOWN0 = 3, M_ZH1 = 4, M_UP1 = 5, M_DOWN1 = 6, M_OUT = 7 };
// a matrix's codes: forward wf [pad64(rows)][kf] (rows rotated along the columns, a scale a row: wfs; bias b, padded),
// dgrad wb [kb][np] (columns rotated along the rows, a scale a column: wbs); offsets in bytes (codes) or floats
struct MatW { uint wf, wfs, b, wb, wbs, rows, cols, kf, kb, np, v, g; };
struct Net { MatW m[8]; };
// a matrix's wgrad operands: dY's codes transposed yt [pad64(3 rows)][RT] (task t's row c at t rows + c), their
// scales sy [RT / 64][3 rows], column sums dbp [RT / 64][3 rows]; X's xt [kf][RT], sx [RT / 64][kf]
struct MatG { uint yt, xt, sy, sx, dbp, b0, b1, p0; };   // b0, b1: the matrix's blocks (lane rows + contexts, or contexts)
struct Grad { MatG m[8]; uint RT, pad[3]; };

// ---- a tile's lanes ----
struct Lanes { uint t, sg, lane, q, li; };
static inline Lanes lanes_of(uint t) { Lanes c; c.t = t; c.sg = t / 32; c.lane = t % 32; c.q = c.lane / 8; c.li = c.lane % 8; return c; }
static inline uint row_of(Lanes c, uint s) { return 8 * c.sg + 4 * s + c.q; }   // a row pass's row, slot s (0, 1)
static inline void tg_barrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }

// one 64-column chunk of a product into S (int32 sums): A [TM][K] (threadgroup) against the chunk of weights.
// forward: wf rows [64 j, 64 j + 64), each K long (transposed); dgrad: wb's columns [64 j, ..), np apart
template <uint K> static inline void chunk_fwd(threadgroup int8_t *A, device const int8_t *w, threadgroup float *S) {
  auto tA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, TM));
  auto tB = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t *)w, dextents<int32_t, 2>(K, 64));
  auto tS = tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline>((threadgroup int32_t *)S, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, LD});
  constexpr auto d = matmul2d_descriptor(TM, 64, K, false, true);
  matmul2d<d, execution_simdgroups<NSG>> op;
  auto c = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  op.run(tA, tB, c);   // multiply: the destination written whole
  c.store(tS);
}
template <uint K> static inline void chunk_bwd(threadgroup int8_t *A, device const int8_t *w, uint np, threadgroup float *S) {
  auto tA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, TM));
  auto tB = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t *)w, dextents<int32_t, 2>(64, K), array<int32_t, 2>{1, int(np)});
  auto tS = tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline>((threadgroup int32_t *)S, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, LD});
  constexpr auto d = matmul2d_descriptor(TM, 64, K, false, false);
  matmul2d<d, execution_simdgroups<NSG>> op;
  auto c = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  op.run(tA, tB, c);
  c.store(tS);
}
// a row slot's eight values of chunk j through the epilogue: i8_out(sum, sa, sb[n], bias[n]) (n = 64 j + 8 li + k)
static inline void epilogue(threadgroup const float *S, Lanes c, uint row, float sa, device const float *sb, device const float *bias, uint j, thread float *v) {
  const uint n0 = 64 * j + 8 * c.li;
  const float4 s0 = *(device const float4 *)(sb + n0), s1 = *(device const float4 *)(sb + n0 + 4);
  float4 b0 = 0, b1 = 0;
  if (bias) { b0 = *(device const float4 *)(bias + n0); b1 = *(device const float4 *)(bias + n0 + 4); }
  const threadgroup int4 *x = (const threadgroup int4 *)((const threadgroup int *)S + row * LD + 8 * c.li);
  const int4 x0 = x[0], x1 = x[1];
  _Pragma("unroll") for (uint k = 0; k < 4; k++) { v[k] = i8_out(x0[k], sa, s0[k], b0[k]); v[4 + k] = i8_out(x1[k], sa, s1[k], b1[k]); }
}
// rows' values (two slots, NC chunks of eight) -> the next product's codes in A (NC 64 columns a row) and its scales sA:
// rotated (a copy), one scale a row; seed 0 to nearest, else stochastic at index (row0 + row) NC 64 + column
template <uint NC> static inline void put_codes(thread const float (&v)[2][NC][8], threadgroup int8_t *A, threadgroup float *sA, Lanes c, uint seed, uint row0) {
  _Pragma("unroll") for (uint s = 0; s < 2; s++) {
    const uint row = row_of(c, s);
    float r[NC][8], m = 0;
    _Pragma("unroll") for (uint j = 0; j < NC; j++) { _Pragma("unroll") for (uint k = 0; k < 8; k++) r[j][k] = v[s][j][k]; i8_fwht64_lanes(r[j], c.li); m = max(m, i8_absmax_lanes(r[j])); }
    const float inv = i8_inverse(m);
    if (c.li == 0) sA[row] = i8_scale(m);
    _Pragma("unroll") for (uint j = 0; j < NC; j++) *(threadgroup uint2 *)(A + row * NC * 64 + 64 * j + 8 * c.li) = i8_pack8(r[j], inv, seed, (row0 + row) * NC * 64 + 64 * j + 8 * c.li);
  }
}
// a slot's values of chunk j written into S as floats (for the column pass)
static inline void put_s(threadgroup float *S, Lanes c, uint row, thread const float *v) {
  threadgroup float4 *o = (threadgroup float4 *)(S + row * LD + 8 * c.li); o[0] = float4(v[0], v[1], v[2], v[3]); o[1] = float4(v[4], v[5], v[6], v[7]);
}
// the column pass over S's 64 columns (the tile's 64 rows): codes rotated along the rows, one scale a column, into
// xt + column (stride RT) at the tile's rows (row0 global); the first `valid` columns; sums (if given) the columns'
// unrotated sums. Stochastic index: column RT + row
static inline void col_codes(threadgroup const float *S, Lanes c, device int8_t *xt, uint RT, uint row0, device float *sc, device float *sums, uint valid, uint seed) {
  _Pragma("unroll") for (uint s = 0; s < 2; s++) {
    const uint col = 8 * c.sg + 4 * s + c.q;
    float v[8];
    _Pragma("unroll") for (uint k = 0; k < 8; k++) v[k] = S[(8 * c.li + k) * LD + col];
    if (sums) {
      float t = 0; for (uint k = 0; k < 8; k++) t += v[k];
      _Pragma("unroll") for (ushort st = 1; st < 8; st <<= 1) t += simd_shuffle_xor(t, st);
      if (c.li == 0 && col < valid) sums[col] = t;
    }
    i8_fwht64_lanes(v, c.li);
    const float m = i8_absmax_lanes(v), inv = i8_inverse(m);
    if (col < valid) {
      if (c.li == 0) sc[col] = i8_scale(m);
      *(device uint2 *)(xt + size_t(col) * RT + row0 + 8 * c.li) = i8_pack8(v, inv, seed, col * RT + row0 + 8 * c.li);
    }
  }
}
static inline uint seed_of(uint base, uint product, uint task) { return base ? (i8_mix(base ^ (product * 0x9e3779b1u) ^ (task * 0x85ebca6bu)) | 1u) : 0u; }

// the kernels' parameters
// runs, refreshes, a run's lane rows (KLp: K L padded to 64, or 32 for one refresh: a tile then two runs), contexts,
// lane rows (RL: B KLp padded to 64), a seed, the contexts' room (pad64), lane tiles a threadgroup (LT)
struct FP { uint B, K, KLp, R, RL, seed, Rp, LT; };
// a lane row's run and position in it
struct LanePos { uint b, pos; bool valid; };
static inline LanePos lane_pos(uint row, constant FP &p) { LanePos q; q.b = row / p.KLp; q.pos = row % p.KLp; q.valid = q.b < p.B && q.pos < p.K * L; return q; }
// what the learning forward stores for the backward: lanes' gates [rows][2][2D] (half), states [rows][2][D], up0's
// relu mask [rows][6]; contexts' gates, masks [R][12] (u0, u1), n [R][D], inv [R]
struct Keep { device half *lzc; device float *lh; device uint *lu; device half *czc; device uint *cu; device float *cn; device float *cinv; };

#define TILE_MEMORY threadgroup float S[TM * LD]; threadgroup int8_t A[TM * 192]; threadgroup float sA[TM];

// a row's relu mask over a 64-column chunk: two words (lanes li 0..3, 4..7), held by lanes li 0 and 4
static inline uint mask_word(thread const float *u, Lanes c) {
  uint bits = 0; for (uint k = 0; k < 8; k++) bits |= (u[k] > 0 ? 1u : 0u) << k;
  uint w = bits << (8 * (c.li % 4));
  w |= simd_shuffle_xor(w, ushort(1)); w |= simd_shuffle_xor(w, ushort(2));
  return w;
}
static inline bool mask_bit(device const uint *words, uint n) { return (words[n / 32] >> (n % 32)) & 1u; }
#endif
