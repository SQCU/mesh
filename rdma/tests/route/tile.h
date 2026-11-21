// tile.h: the tiles' shared parts (fwd.metal, bwd.metal; opt.metal its tables). The policy's network on the GPU, built
// on int8 x int8 -> int32 products on the matrix units (matmul2d), every product's epilogue and the next product's
// quantization fused in, a tile of rows carried on chip through its layers.
//
// The network (route.h): x = inp(token); two minGRU layers, each z, h~ = sigmoid, tanh of zh(x), the recurrence
// h = (1 - m z) h + m z h~ along a run's lanes (scan.h), a context one step from its refresh's state, xm = x + h,
// x' = xm + down(relu(up(xm))) (layer 1's lanes: their states only); w = out(x2 / rms(x2)) a context.
//
// Tiles:
// - a tile is 64 rows (TM), eight SIMD groups (NT threads), each product a threadgroup's: matmul2d over
//   execution_simdgroups<8>, a 64 x 64 output chunk (N in 64-column chunks: the Walsh-Hadamard block);
// - a product's int32 sums (or the rotation's fp32 values) stored to threadgroup memory S (64 x 68 words: rows 16-byte
//   aligned, the row pass four words a load), then a row pass: eight lanes a row, eight consecutive columns a lane, two
//   rows a thread (16 values a thread): the epilogue, and the next product's codes;
// - quantizing a row's 64-block (int8mm.h): its absmax over the eight lanes, the 12-bit grid into X (fp16), the
//   rotation Y = X H on the matrix units (fp16 operands, fp32 accumulation: exact on the grid's integers), Y's absmax,
//   the codes into A (or B), a scale a (row, block);
// - a column pass (the learning calls' wgrad codes of a value in S): a SIMD group's eight columns, both 32-row blocks
//   (I8_WBLOCK): a block's absmax, the grid, the rotation along the rows Y^T = X^T H32 by the SIMD group's product,
//   the codes stored transposed (each column's rows contiguous, as the wgrad product reads them), one scale a (column,
//   block); a gradient's column sums (its bias's gradient) beside.
// A threadgroup barrier stands between two phases (a product, a pass) wherever the later one writes what another
// SIMD group's rows were read from; a row pass that rewrites its own rows' room needs none (a row is a SIMD group's).
//
// Rows: the lanes' ([run][KLp], a run's positions in order, its pad rows masked; one refresh: a tile two runs) then the
// contexts'. The learning calls' backward: three tasks, each its own threadgroups.
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
// the kernels' tables (Net, Grad, FP) as bound: constant buffers (the Metal library's kernels), or device memory (the
// same kernels as MLX ops, qnn.py: ROUTE_TBL defined device const)
#ifndef ROUTE_TBL
#define ROUTE_TBL constant
#endif

enum : uint { D = 64, E = 192, IN = 48, FS = 40, L = 32, TM = 64, NSG = 8, NT = NSG * 32, LD = 68, LX = 68 };

// the matrices, in NetLayout's order
enum : uint { M_INP = 0, M_ZH0 = 1, M_UP0 = 2, M_DOWN0 = 3, M_ZH1 = 4, M_UP1 = 5, M_DOWN1 = 6, M_OUT = 7 };
// a matrix's codes: forward wf [pad64(rows)][kf] (one scale a row: wfs; bias b), dgrad wb [np][kb] (W's columns, one
// scale a column: wbs; a column a row, so the dgrad product reads its weights as the forward does); offsets in bytes
// (codes) or floats
struct MatW { uint wf, wfs, b, wb, wbs, rows, cols, kf, kb, np, v, g; };
struct Net { MatW m[8]; };
// a matrix's wgrad operands: dY's codes transposed yt [pad64(3 rows)][RT] (task t's row c at t rows + c), their
// scales sy [RT / 32][3 rows], column sums dbp [RT / 32][3 rows]; X's xt [kf][RT], sx [RT / 32][kf]
struct MatG { uint yt, xt, sy, sx, dbp, b0, b1, p0; };
struct Grad { MatG m[8]; uint RT, T, pad[2]; };   // T: the tasks (a gradient's dY rows: T rows; the learning step's three)

typedef tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> TgI8;
typedef tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> DevI8;
typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> TgH;
typedef tensor<device half, dextents<int32_t, 2>, tensor_inline> DevH;
typedef tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline> TgI32;
typedef tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> TgF;

// ---- a tile's lanes ----
struct Lanes { uint t, sg, lane, q, li; };
static inline Lanes lanes_of(uint t) { Lanes c; c.t = t; c.sg = t / 32; c.lane = t % 32; c.q = c.lane / 8; c.li = c.lane % 8; return c; }
static inline uint row_of(Lanes c, uint s) { return 8 * c.sg + 4 * s + c.q; }   // a row pass's row, slot s (0, 1)
static inline void tg_barrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }

// ---- the products (each a threadgroup's) ----
// one 64-column chunk into S (int32 sums): codes A [TM][64] (threadgroup, stride lda) . 64 rows of 64 codes at w
// (stride ldw): forward wf's rows [64 j, 64 j + 64) at K block b (w + 64 b), dgrad wb's
static inline void chunk(threadgroup const int8_t *A, uint lda, device const int8_t *w, uint ldw, threadgroup float *S) {
  TgI8 tA((threadgroup int8_t *)A, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, int(lda)});
  DevI8 tB((device int8_t *)w, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, int(ldw)});
  TgI32 tS((threadgroup int32_t *)S, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, LD});
  matmul2d<matmul2d_descriptor(TM, 64, 64, false, true), execution_simdgroups<NSG>> op;
  auto c = op.template get_destination_cooperative_tensor<TgI8, DevI8, int32_t>();
  op.run(tA, tB, c);
  c.store(tS);
}
// the rows' rotation: S = X [TM][64] (the grid, fp16) . H (64 x 64), fp32
static inline void hrow(threadgroup const half *X, device const half *H, threadgroup float *S) {
  TgH tX((threadgroup half *)X, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, LX});
  DevH tH((device half *)H, dextents<int32_t, 2>(64, 64));
  TgF tS(S, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, LD});
  matmul2d<matmul2d_descriptor(TM, 64, 64, false, false), execution_simdgroups<NSG>> op;
  auto c = op.template get_destination_cooperative_tensor<TgH, DevH, float>();
  op.run(tX, tH, c);
  c.store(tS);
}
// ---- the row pass ----
static inline void s_get(threadgroup const float *S, uint row, Lanes c, thread float *v) {
  const threadgroup float4 *x = (const threadgroup float4 *)(S + row * LD + 8 * c.li); const float4 a = x[0], b = x[1];
  for (uint k = 0; k < 4; k++) { v[k] = a[k]; v[4 + k] = b[k]; }
}
static inline void put_s(threadgroup float *S, Lanes c, uint row, thread const float *v) {
  threadgroup float4 *o = (threadgroup float4 *)(S + row * LD + 8 * c.li); o[0] = float4(v[0], v[1], v[2], v[3]); o[1] = float4(v[4], v[5], v[6], v[7]);
}
// a chunk's column vectors, the thread's eight columns (n = 64 j + 8 li + k): the scales sb, the bias (if given), loaded
// before the product is issued (its time covers their latency), once for both row slots
struct ColV { float4 s0, s1, b0, b1; };
static inline ColV colv(device const float *sb, device const float *bias, uint j, Lanes c) {
  const uint n0 = 64 * j + 8 * c.li; ColV v;
  v.s0 = *(device const float4 *)(sb + n0); v.s1 = *(device const float4 *)(sb + n0 + 4);
  if (bias) { v.b0 = *(device const float4 *)(bias + n0); v.b1 = *(device const float4 *)(bias + n0 + 4); } else { v.b0 = 0; v.b1 = 0; }
  return v;
}
// a row slot's eight values through the epilogue: i8_out(sum, sa, sb[n], bias[n]), or with acc i8_acc into v
static inline void epilogue(threadgroup const float *S, Lanes c, uint row, float sa, thread const ColV &cv, thread float *v, bool acc) {
  const threadgroup int4 *x = (const threadgroup int4 *)((const threadgroup int *)S + row * LD + 8 * c.li);
  const int4 x0 = x[0], x1 = x[1];
  if (acc) { for (uint k = 0; k < 4; k++) { v[k] = i8_acc(x0[k], sa, cv.s0[k], v[k]); v[4 + k] = i8_acc(x1[k], sa, cv.s1[k], v[4 + k]); } }
  else { for (uint k = 0; k < 4; k++) { v[k] = i8_out(x0[k], sa, cv.s0[k], cv.b0[k]); v[4 + k] = i8_out(x1[k], sa, cv.s1[k], cv.b1[k]); } }
}
static inline float max8(float m) { m = max(m, simd_shuffle_xor(m, ushort(1))); m = max(m, simd_shuffle_xor(m, ushort(2))); return max(m, simd_shuffle_xor(m, ushort(4))); }
// quantizing a row's block, its first half: the absmax m0 (returned), the grid into X; isum (if given): the grid
// integers' squares summed over the row (act.h's rms)
static inline float grid_row(thread const float *v, threadgroup half *X, Lanes c, uint row, thread int *isum) {
  float m0 = 0;
  for (uint k = 0; k < 8; k++) m0 = max(m0, abs(v[k]));
  m0 = max8(m0);
  const float gi = i8_grid_inverse(m0);
  float g[8];
  for (uint k = 0; k < 8; k++) g[k] = i8_grid(v[k], gi);
  if (isum) {
    int q = 0; for (uint k = 0; k < 8; k++) { const int z = int(g[k]); q += z * z; }
    q += simd_shuffle_xor(q, ushort(1)); q += simd_shuffle_xor(q, ushort(2)); q += simd_shuffle_xor(q, ushort(4));
    *isum = q;
  }
  threadgroup half4 *o = (threadgroup half4 *)(X + row * LX + 8 * c.li);
  o[0] = half4(g[0], g[1], g[2], g[3]); o[1] = half4(g[4], g[5], g[6], g[7]);
  return m0;
}
// eight codes of y (inv), packed (element k at byte k); stochastic (seed): element at0 + k's uniform, a hash a pair
static inline uint2 pack8(thread const float *y, float inv, uint seed, uint at0) {
  uint w[2] = {0, 0};
  if (seed) {
    for (uint k = 0; k < 8; k += 2) {
      const uint h = i8_mix(seed ^ (((at0 + k) >> 1) * 0x9e3779b1u));
      const float c0 = clamp(floor(y[k] * inv + float(h & 0xffffu) * (1.0f / 65536.0f)), -127.0f, 127.0f);
      const float c1 = clamp(floor(y[k + 1] * inv + float(h >> 16) * (1.0f / 65536.0f)), -127.0f, 127.0f);
      w[k / 4] |= (uint(uchar(int8_t(c0))) | (uint(uchar(int8_t(c1))) << 8)) << (8 * (k % 4));
    }
  } else {
    for (uint k = 0; k < 8; k++) w[k / 4] |= uint(uchar(int8_t(rint(y[k] * inv)))) << (8 * (k % 4));   // |y inv| <= 127
  }
  return uint2(w[0], w[1]);
}
// its second half, after the rotation into S: Y's absmax, the codes into dst [TM][ldd] at column 0, the scale into
// sdst[row]; stochastic at index at0 + 8 li + k (at0: the row's first column's)
static inline void codes_row(threadgroup const float *S, Lanes c, uint row, float m0, threadgroup int8_t *dst, uint ldd, threadgroup float *sdst, uint seed, uint at0) {
  float y[8]; s_get(S, row, c, y);
  float m = 0;
  for (uint k = 0; k < 8; k++) m = max(m, abs(y[k]));
  m = max8(m);
  const float inv = i8_inverse(m);
  if (c.li == 0) sdst[row] = i8_scale(m, m0, 8.0f);
  *(threadgroup uint2 *)(dst + row * ldd + 8 * c.li) = pack8(y, inv, seed, at0 + 8 * c.li);
}
// a whole row quantization of v[2][8] (both slots): grid, rotation, codes into dst, scales sdst; seed, index (row0 +
// row) cols + c0 + column. Ends at a barrier (dst ready for the product)
static inline void quant(thread const float (&v)[2][8], threadgroup float *S, threadgroup half *X, device const half *H, Lanes c,
                         threadgroup int8_t *dst, threadgroup float *sdst, uint seed, uint row0, uint cols, uint c0) {
  float m0[2];
  for (uint s = 0; s < 2; s++) m0[s] = grid_row(v[s], X, c, row_of(c, s), nullptr);
  tg_barrier(); hrow(X, H, S); tg_barrier();
  for (uint s = 0; s < 2; s++) { const uint row = row_of(c, s); codes_row(S, c, row, m0[s], dst, 64, sdst, seed, (row0 + row) * cols + c0); }
  tg_barrier();
}

// ---- the column pass: the wgrad codes of S's 64 columns (the tile's 64 rows, two blocks), scale k (4: X's, 8: dY's)
// into xt + column RT + row0 (row0 the tile's global row; the first `valid` columns), scales sc[b scs + column] (block
// b of the tile), sums (if given) the blocks' column sums at the same place; stochastic index column RT + row0 + row.
// A SIMD group its eight columns, both blocks, no threadgroup barrier but the last (S and X spent): a column's rows
// li + 8 k over its eight lanes (68 words a row: a load's 32 lanes on 32 banks), its block's absmax, the grid into the
// SIMD group's X [32 rows][16 = (column, block)], the rotation along the rows Y^T = X^T H32 on the matrix units (the
// SIMD group's product: (column, block) rows, a block's 32 rows its columns), Y's absmax a row by the reduction, the
// codes four consecutive rows a word
typedef matmul2d<matmul2d_descriptor(16, 32, 32, true, false), execution_simdgroup> OpC;
static inline void colcodes(threadgroup float *S, threadgroup half *X, threadgroup float *cm, device const half *H, Lanes c, device int8_t *xt, uint RT, uint row0,
                            device float *sc, uint scs, device float *sums, uint valid, float kscale, uint seed) {
  threadgroup half *Xs = X + 512 * c.sg; threadgroup float *cs = cm + 16 * c.sg;
  for (uint s = 0; s < 2; s++) {
    const uint cl = 4 * s + c.q, col = 8 * c.sg + cl;
    float v[8];
    for (uint k = 0; k < 8; k++) v[k] = S[(c.li + 8 * k) * LD + col];
    float m0[2] = {0, 0};
    for (uint k = 0; k < 8; k++) m0[k / 4] = max(m0[k / 4], abs(v[k]));
    m0[0] = max8(m0[0]); m0[1] = max8(m0[1]);
    if (sums) {
      float t[2] = {0, 0};
      for (uint k = 0; k < 8; k++) t[k / 4] += v[k];
      for (ushort st = 1; st < 8; st <<= 1) { t[0] += simd_shuffle_xor(t[0], st); t[1] += simd_shuffle_xor(t[1], st); }
      if (c.li == 0 && col < valid) { sums[col] = t[0]; sums[scs + col] = t[1]; }
    }
    const float g0 = i8_grid_inverse(m0[0]), g1 = i8_grid_inverse(m0[1]);
    for (uint k = 0; k < 8; k++) Xs[(c.li + 8 * (k % 4)) * 16 + cl + 8 * (k / 4)] = half(i8_grid(v[k], k < 4 ? g0 : g1));
    if (c.li == 0) { cs[cl] = m0[0]; cs[8 + cl] = m0[1]; }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  TgH tX(Xs, dextents<int32_t, 2>(16, 32));
  DevH tH((device half *)H, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
  OpC op;
  auto y = op.template get_destination_cooperative_tensor<TgH, DevH, float>();
  op.run(tX, tH, y);
  auto g = op.template get_destination_cooperative_tensor<TgH, DevH, float>();
  for (ushort i = 0; i < y.get_capacity(); i++) g[i] = abs(y[i]);
  auto rm = op.template get_row_reduction_destination_cooperative_tensor<TgH, DevH, float>();
  reduce_rows(g, rm, reduction_operation::max, 0.0f);
  // lane l: rows (column, block) m = mr, mr + 8, block rows n = nb + 0..3, nb + 16..19
  const uint mr = ((c.lane >> 1) & 3) | ((c.lane >> 4) << 2), nb = 4 * (c.lane & 1) + 8 * ((c.lane >> 3) & 1), col = 8 * c.sg + mr;
  if (col < valid) {
    for (uint b = 0; b < 2; b++) {
      const float m = rm[b], inv = i8_inverse(m);
      if ((c.lane & 9) == 0) sc[b * scs + col] = i8_scale(m, cs[mr + 8 * b], kscale);
      for (uint h = 0; h < 2; h++) {
        const ushort i = ushort(4 * b + 8 * h); const uint n = 32 * b + nb + 16 * h, at = col * RT + row0 + n;
        uint w = 0;
        if (seed) {
          for (uint k = 0; k < 4; k += 2) {
            const uint hh = i8_mix(seed ^ (((at + k) >> 1) * 0x9e3779b1u));
            const float c0 = clamp(floor(y[i + k] * inv + float(hh & 0xffffu) * (1.0f / 65536.0f)), -127.0f, 127.0f);
            const float c1 = clamp(floor(y[i + k + 1] * inv + float(hh >> 16) * (1.0f / 65536.0f)), -127.0f, 127.0f);
            w |= (uint(uchar(int8_t(c0))) | (uint(uchar(int8_t(c1))) << 8)) << (8 * k);
          }
        } else {
          for (uint k = 0; k < 4; k++) w |= uint(uchar(int8_t(rint(y[i + k] * inv)))) << (8 * k);
        }
        *(device uint *)(xt + size_t(col) * RT + row0 + n) = w;
      }
    }
  }
  tg_barrier();
}
static inline uint seed_of(uint base, uint product, uint task) { return base ? (i8_mix(base ^ (product * 0x9e3779b1u) ^ (task * 0x85ebca6bu)) | 1u) : 0u; }

// the kernels' parameters: runs, refreshes, a run's lane rows (KLp: K L padded to 64, or 32 for one refresh: a tile
// then two runs), contexts, lane rows (RL: B KLp padded to 64), a seed, the contexts' room (pad64), lane tiles a
// threadgroup (LT)
struct FP { uint B, K, KLp, R, RL, seed, Rp, LT; };
struct LanePos { uint b, pos; bool valid; };
static inline LanePos lane_pos(uint row, ROUTE_TBL FP &p) { LanePos q; q.b = row / p.KLp; q.pos = row % p.KLp; q.valid = q.b < p.B && q.pos < p.K * L; return q; }

// a row's relu mask over a 64-column chunk: two words (lanes li 0..3, 4..7), held by lanes li 0 and 4
static inline uint mask_word(thread const float *u, Lanes c) {
  uint bits = 0; for (uint k = 0; k < 8; k++) bits |= (u[k] > 0 ? 1u : 0u) << k;
  uint w = bits << (8 * (c.li % 4));
  w |= simd_shuffle_xor(w, ushort(1)); w |= simd_shuffle_xor(w, ushort(2));
  return w;
}
static inline bool mask_bit(device const uint *words, uint n) { return (words[n / 32] >> (n % 32)) & 1u; }
// the threadgroup memory: S the products' outputs, A the codes a chain's products share (x's), X the grid (fp16) or
// the second operand's codes (B, int8, aliased: their lifetimes are disjoint), sA, sB their scales, cm the column
// pass's absmaxes
#define TILE_MEMORY threadgroup float S[TM * LD]; threadgroup int8_t A[TM * 64]; threadgroup half X[TM * LX]; threadgroup float sA[TM], sB[TM], cm[128]; \
  threadgroup int8_t *Bq = (threadgroup int8_t *)X;
#endif
