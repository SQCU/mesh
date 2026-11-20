// sg.h: the SIMD group's tile (fwd.metal, bwd.metal). The policy's network on the GPU, every step a SIMD group's own:
// a tile is 32 rows (TR) a SIMD group owns through all its products, no threadgroup barrier between them (a
// threadgroup is NSG independent tiles; the lanes' learning forward alone meets once a layer, to compose a run's
// refreshes). Measured (stages, x/chainx): a SIMD group's 32 x 64 x 64 int8 product, operands on chip, 516 TOP/s;
// chained from device weights 477; with a vector epilogue (scale, bias, relu, codes scattered) 124: so every step that
// can be is a product on the matrix units, the vector ALU left a few operations an output.
//
// The network (route.h): x = inp(token); two minGRU layers, each z, h~ = sigmoid, tanh of zh(x), the recurrence
// h = (1 - m z) h + m z h~ along a run's lanes (scan.h), a context one step from its refresh's state, xm = x + h,
// x' = xm + down(relu(up(xm))) (layer 1's lanes: their states only); w = out(x2 / rms(x2)) a context.
//
// A tile's values live in registers in the products' destination layout (a 32 x 64 chunk: 64 values a thread; the
// layout probed and checked by stages: lane l holds rows r + 8 s (r = ((l >> 1) & 3) | ((l >> 4) << 2), s = 0..3) and
// columns cb + (i & 3) + 16 ((i >> 3) & 3) (cb = 4 (l & 1) + 8 ((l >> 3) & 1)); element i's row slot s = ((i >> 2) & 1) | ((i >> 5) << 1)).
// Quantizing a row's 64-block (int8mm.h): its absmax by reduce_rows, the 12-bit grid into X (fp16, threadgroup), the
// rotation y = X H on the matrix units (fp16, fp32 accumulation: exact), y's absmax by reduce_rows, the codes packed
// four a word into the next product's A. A column's 32-block (wgrad codes): its absmax by reduce_columns, the grid,
// Y^T = X^T H32 on the matrix units, the codes stored transposed four rows a word.
#ifndef ROUTE_SG_H
#define ROUTE_SG_H
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

enum : uint { D = 64, E = 192, IN = 48, FS = 40, L = 32, TR = 32, NSG = 4, NT = NSG * 32 };
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
struct Grad { MatG m[8]; uint RT, pad[3]; };

// ---- the tile's types ----
typedef tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> TgI8;
typedef tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> DevI8;
typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> TgH;
typedef tensor<device half, dextents<int32_t, 2>, tensor_inline> DevH;
typedef matmul2d<matmul2d_descriptor(TR, 64, 64, false, true), execution_simdgroup> Op;    // A [32][64] . W [64][64]^T
typedef matmul2d<matmul2d_descriptor(64, TR, TR, true, false), execution_simdgroup> OpT;   // X^T [64][32] . H32 [32][32]
// a chain's operands in registers: the codes an int8 left input, the grid an fp16 left input (their layouts the
// destination's, checked by stages), so a product's output feeds the next product with no memory between them
typedef decltype(Op().get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>()) LI;
typedef decltype(Op().get_left_input_cooperative_tensor<half, half, float>()) LH;
typedef decltype(Op().get_destination_cooperative_tensor<LI, DevI8, int32_t>()) CI;
typedef decltype(Op().get_destination_cooperative_tensor<LI, DevI8, float>()) CF;
typedef decltype(Op().get_destination_cooperative_tensor<TgI8, DevI8, half>()) CHh;
typedef decltype(Op().get_destination_cooperative_tensor<LH, DevH, float>()) CY;
typedef decltype(Op().get_row_reduction_destination_cooperative_tensor<LI, DevI8, float>()) RF;
typedef decltype(Op().get_row_reduction_destination_cooperative_tensor<LI, DevI8, int32_t>()) RI;
typedef decltype(Op().get_row_reduction_destination_cooperative_tensor<LH, DevH, float>()) RY;
typedef decltype(Op().get_column_reduction_destination_cooperative_tensor<LI, DevI8, float>()) CR;
typedef decltype(OpT().get_destination_cooperative_tensor<TgH, TgH, float>()) CT;
typedef decltype(OpT().get_row_reduction_destination_cooperative_tensor<TgH, TgH, float>()) RT_;
enum : ushort { NE = 64 };   // a thread's elements of a chunk

// ---- the layout ----
struct Lay { uint lane, r, cb; };
static inline Lay lay_of(uint lane) { Lay a; a.lane = lane; a.r = ((lane >> 1) & 3) | ((lane >> 4) << 2); a.cb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1); return a; }
// the lane holding row r of the same columns
static inline ushort lane_at(Lay a, uint r) { return ushort((a.lane & 0x9) | ((r & 3) << 1) | ((r >> 2) << 4)); }
static inline uint slot(ushort i) { return ((i >> 2) & 1) | ((i >> 5) << 1); }
static inline uint col_of(Lay a, ushort i) { return a.cb + (i & 3) + 16 * ((i >> 3) & 3); }
static inline uint row_of(Lay a, ushort i) { return a.r + 8 * slot(i); }
static inline uint row_s(Lay a, uint s) { return a.r + 8 * s; }
static inline uint cslot(ushort i) { return (i & 3) | (((i >> 3) & 3) << 2); }   // element i's column among the thread's 16
static inline uint col_j(Lay a, uint j) { return a.cb + (j & 3) + 16 * (j >> 2); }
// the transposed product's (Y^T [64 columns][32 rows]) layout: lane l element i at block row n = nb + (i & 3) +
// 16 ((i >> 3) & 1) (nb = cb), column m = r + 8 tslot(i)
static inline uint tslot(ushort i) { return ((i >> 2) & 1) | ((i >> 4) << 1); }

// one 64 x 64 block product into c: codes A (32 x 64, registers) . w's 64 rows of 64 codes (at ldw)
static inline void product(thread CI &c, thread LI &A, device const int8_t *w, uint ldw) {
  DevI8 tW((device int8_t *)w, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, int(ldw)});
  Op op; op.run(A, tW, c);
}
typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> DevF;
typedef tensor<device half, dextents<int32_t, 2>, tensor_inline> DevHf;
// a column vector (64 floats) broadcast over the tile's rows, by the cooperative tensor's own load (measured 4x the
// per-element loads' rate: x/sgb b30 against b28)
static inline void colvec(thread CF &v, device const float *p) { DevF t((device float *)p, dextents<int32_t, 2>(64, TR), array<int32_t, 2>{1, 0}); v.load(t); }
// a tile's rows of a row-major matrix (64 columns at ld, `rows` live rows, `cols` live columns; the rest 0) in and out
static inline void tile_load(thread CF &v, device const float *p, uint ld, uint rows, uint cols) {
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) v[i] = 0;
  if (rows) { DevF t((device float *)p, dextents<int32_t, 2>(cols, rows), array<int32_t, 2>{1, int(ld)}); v.load(t); }
}
static inline void tile_store(thread const CF &v, device float *p, uint ld, uint rows, uint cols) {
  if (rows) { DevF t(p, dextents<int32_t, 2>(cols, rows), array<int32_t, 2>{1, int(ld)}); v.store(t); }
}
static inline void tile_store_half(thread const CF &v, device half *p, uint ld, uint rows) {
  CHh h; _Pragma("unroll") for (ushort i = 0; i < NE; i++) h[i] = half(v[i]);
  if (rows) { DevHf t(p, dextents<int32_t, 2>(64, rows), array<int32_t, 2>{1, int(ld)}); h.store(t); }
}
// f = fma(float(sum) sa[row], sb[col], add): add a column vector (bias, or 0) or, with acc, f itself (i8_acc)
static inline void epi(thread CF &f, thread const CI &c, thread const float *sa, device const float *sb, device const float *bias, Lay a, bool acc) {
  CF s; colvec(s, sb);
  if (acc) { _Pragma("unroll") for (ushort i = 0; i < NE; i++) f[i] = fma(float(c[i]) * sa[slot(i)], s[i], f[i]); }
  else if (bias) { CF b; colvec(b, bias); _Pragma("unroll") for (ushort i = 0; i < NE; i++) f[i] = fma(float(c[i]) * sa[slot(i)], s[i], b[i]); }
  else { _Pragma("unroll") for (ushort i = 0; i < NE; i++) f[i] = fma(float(c[i]) * sa[slot(i)], s[i], 0.0f); }
}
// rows' largest |v| (four: the thread's row slots)
static inline void row_absmax(thread const CF &v, thread float *m) {
  CF g; _Pragma("unroll") for (ushort i = 0; i < NE; i++) g[i] = abs(v[i]);
  RF rm; reduce_rows(g, rm, reduction_operation::max, 0.0f);
  _Pragma("unroll") for (ushort s = 0; s < 4; s++) m[s] = rm[s];
}
// a chunk's rows quantized (int8mm.h's 64-block): codes into A, scales sa[4]; seed 0 to nearest, else stochastic at
// index (row0 + row) cols + c0 + column. isum (if given): the grid integers' squares' sums a row (act.h's rms); m0
// (if given): the rows' absmax
static inline void quant(thread const CF &v, thread LI &A, device const half *H, thread float *sa, Lay a,
                         uint seed, uint row0, uint cols, uint c0, thread int *isum, thread float *m0out) {
  float m0[4], gi[4];
  row_absmax(v, m0);
  _Pragma("unroll") for (ushort s = 0; s < 4; s++) { gi[s] = i8_grid_inverse(m0[s]); if (m0out) m0out[s] = m0[s]; }
  Op op; LH hx = op.get_left_input_cooperative_tensor<half, half, float>();
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) hx[i] = half(i8_grid(v[i], gi[slot(i)]));
  if (isum) {
    CI q; _Pragma("unroll") for (ushort i = 0; i < NE; i++) { const int z = int(hx[i]); q[i] = z * z; }
    RI rs; reduce_rows(q, rs, reduction_operation::sum, 0);
    _Pragma("unroll") for (ushort s = 0; s < 4; s++) isum[s] = rs[s];
  }
  DevH tH((device half *)H, dextents<int32_t, 2>(64, 64));
  CY y; op.run(hx, tH, y);
  CY g; _Pragma("unroll") for (ushort i = 0; i < NE; i++) g[i] = abs(y[i]);
  RY rm; reduce_rows(g, rm, reduction_operation::max, 0.0f);
  float inv[4];
  _Pragma("unroll") for (ushort s = 0; s < 4; s++) { inv[s] = i8_inverse(rm[s]); sa[s] = i8_scale(rm[s], m0[s], 8.0f); }
  if (seed) {
    _Pragma("unroll") for (ushort i = 0; i < NE; i += 2) {   // a pair's hash shared (int8mm.h)
      const uint at = (row0 + row_of(a, i)) * cols + c0 + col_of(a, i);
      const uint h = i8_mix(seed ^ ((at >> 1) * 0x9e3779b1u));
      const float x0 = y[i] * inv[slot(i)], x1 = y[i + 1] * inv[slot(i)];
      A[i] = int8_t(clamp(floor(x0 + float(h & 0xffffu) * (1.0f / 65536.0f)), -127.0f, 127.0f));
      A[i + 1] = int8_t(clamp(floor(x1 + float(h >> 16) * (1.0f / 65536.0f)), -127.0f, 127.0f));
    }
  } else {
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) A[i] = int8_t(rint(y[i] * inv[slot(i)]));   // |y inv| <= 127: no clamp needed
  }
}
// the wgrad codes of a chunk's columns (int8mm.h's 32-block, scale k: 4 X's, 8 dY's): into xt + column RT + row0
// (the tile's global row; columns 0..valid-1), scales sc[column], column sums (if given) sums[column]; stochastic
// index column RT + row0 + row. X, cm: the SIMD group's fp16 and float scratch
static inline void colcodes(thread const CF &v, device int8_t *xt, uint RT, uint row0, device float *sc, device float *sums, uint valid, float k, uint seed,
                            threadgroup half *X, threadgroup float *cm, device const half *H, Lay a) {
  CF g; _Pragma("unroll") for (ushort i = 0; i < NE; i++) g[i] = abs(v[i]);
  CR mc; reduce_columns(g, mc, reduction_operation::max, 0.0f);
  float gi[16];
  _Pragma("unroll") for (ushort j = 0; j < 16; j++) { gi[j] = i8_grid_inverse(mc[j]); if (a.r == 0) cm[col_j(a, j)] = mc[j]; }
  if (sums) {
    CF vv; _Pragma("unroll") for (ushort i = 0; i < NE; i++) vv[i] = v[i]; CR cs; reduce_columns(vv, cs, reduction_operation::sum, 0.0f);
    if (a.r == 0) _Pragma("unroll") for (ushort j = 0; j < 16; j++) if (col_j(a, j) < valid) sums[col_j(a, j)] = cs[j];
  }
  CHh hx;
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) hx[i] = half(i8_grid(v[i], gi[cslot(i)]));
  TgH tX(X, dextents<int32_t, 2>(64, TR));
  hx.store(tX);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  DevH tH((device half *)H, dextents<int32_t, 2>(TR, TR), array<int32_t, 2>{1, 64});
  OpT op; CT y; op.run(tX, tH, y);
  CT g2; _Pragma("unroll") for (ushort i = 0; i < NE; i++) g2[i] = abs(y[i]);
  RT_ rm; reduce_rows(g2, rm, reduction_operation::max, 0.0f);
  float inv[8];
  _Pragma("unroll") for (ushort s = 0; s < 8; s++) {
    const uint m = a.r + 8 * s; inv[s] = i8_inverse(rm[s]);
    if ((a.lane & 0x9) == 0 && m < valid) sc[m] = i8_scale(rm[s], cm[m], k);
  }
  // the codes staged in X (spent: the product has read it) as [column][32 rows], then each column's 32 bytes stored
  // contiguously (two 16-byte stores a column), the stochastic hash a pair of rows
  simdgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup uint *Q = (threadgroup uint *)X;
  _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) {
    const uint s = tslot(i), m = a.r + 8 * s, n = a.cb + 16 * ((i >> 3) & 1), idx = m * RT + row0 + n;
    uint w = 0;
    if (seed) {
      _Pragma("unroll") for (ushort q = 0; q < 4; q += 2) {
        const uint h = i8_mix(seed ^ (((idx + q) >> 1) * 0x9e3779b1u));
        const float x0 = y[i + q] * inv[s], x1 = y[i + q + 1] * inv[s];
        const float c0 = clamp(floor(x0 + float(h & 0xffffu) * (1.0f / 65536.0f)), -127.0f, 127.0f);
        const float c1 = clamp(floor(x1 + float(h >> 16) * (1.0f / 65536.0f)), -127.0f, 127.0f);
        w |= (uint(uchar(int8_t(c0))) | (uint(uchar(int8_t(c1))) << 8)) << (8 * q);
      }
    } else {
      _Pragma("unroll") for (ushort q = 0; q < 4; q++) w |= uint(uchar(int8_t(rint(y[i + q] * inv[s])))) << (8 * q);
    }
    Q[m * 8 + n / 4] = w;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("unroll") for (ushort k = 0; k < 4; k++) {
    const uint e = a.lane + 32 * k, m = e / 2, h = e % 2;
    if (m < valid) *(device uint4 *)(xt + size_t(m) * RT + row0 + 16 * h) = ((threadgroup const uint4 *)Q)[e];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}
static inline uint seed_of(uint base, uint product, uint task) { return base ? (i8_mix(base ^ (product * 0x9e3779b1u) ^ (task * 0x85ebca6bu)) | 1u) : 0u; }

// the kernels' parameters: runs, refreshes, a run's lane rows (KLp = L Kp, Kp the refreshes padded to 1, 2 or 4: a
// threadgroup's four tiles a whole number of runs), contexts, lane rows (RL = B KLp), a seed, the contexts' room
// (pad64), Kp
struct FP { uint B, K, KLp, R, RL, seed, Rp, Kp; };
struct LanePos { uint b, pos; bool valid; };
static inline LanePos lane_pos(uint row, constant FP &p) { LanePos q; q.b = row / p.KLp; q.pos = row % p.KLp; q.valid = q.b < p.B && q.pos < p.K * L; return q; }

// a chunk's relu mask bits: a row's 64 bits (two words) from the four lanes holding it; written by lane (l & 9) == 0
static inline void mask_words(thread const CF &u, Lay a, thread uint (&w)[4][2]) {
  _Pragma("unroll") for (ushort s = 0; s < 4; s++) { w[s][0] = 0; w[s][1] = 0; }
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) if (u[i] > 0) { const uint c = col_of(a, i); w[slot(i)][c / 32] |= 1u << (c % 32); }
  _Pragma("unroll") for (ushort s = 0; s < 4; s++) for (ushort h = 0; h < 2; h++) { uint x = w[s][h]; x |= simd_shuffle_xor(x, ushort(1)); x |= simd_shuffle_xor(x, ushort(8)); w[s][h] = x; }
}
static inline bool mask_bit(device const uint *words, uint n) { return (words[n / 32] >> (n % 32)) & 1u; }
// what the learning forward stores for the backward: lanes' gates [rows][2][2D] (half), states [rows][2][D], up0's
// relu mask [rows][6]; contexts' gates, masks [R][12] (u0, u1), n [R][D], inv [R]
#define SG_MEMORY threadgroup int8_t Aall[NSG][TR * 64]; threadgroup half Xall[NSG][TR * 64]; threadgroup float Call[NSG][64];
#define SG_TILE threadgroup int8_t *A = Aall[sg]; threadgroup half *X = Xall[sg]; threadgroup float *cm = Call[sg]; const Lay a = lay_of(lane);
#endif
