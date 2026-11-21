/* The int8 product's numerics, defined once: every user includes this header (the GPU's kernels, forward.c's deployed
   CPU forward, stages.swift's and model.swift's checks) rather than restating them.
   A product C = A B^T over a reduction of K terms, int8 x int8 -> int32:
   - quantizing a 64-block of values x (an activation row's 64 columns, or a weight row's): the block's 12-bit grid,
     m0 = max|x|, xi = rint(x gi), gi the power of two putting m0 gi in [1024, 2048) (integers |xi| <= 2048, exact in
     fp16); rotated by the Walsh-Hadamard matrix H (64 x 64, entries +-1, unnormalized): y = H xi, integers
     |y| <= 64 2048 < 2^24, so any order of the sum gives the
     same fp32 bits (the GPU multiplies by H on its matrix units, fp16 operands, fp32 accumulation; the CPU by the
     butterfly i8_fwht64); m = max|y|, codes q = y 127 / m to nearest (even) or stochastically (floor(x + u), u 16 bits
     of i8_mix of the element pair's index and a seed: unbiased below 2^-17 of a step), clamped to +-127 (i8_code);
     the scale s = m step / (127 k) (i8_scale, step = 1 / gi), so q s ~ H x / k: k = 8 for a 64-block (H^T H = 64 I = 8 8);
   - activations: a scale a (row, 64-block): every block of a row its own grid and scale (i8_quantize_blocks), so a
     product over K > 64 is a sum of K / 64 exact int32 block sums, each scaled: out = addend, then for each block
     out = fma(float(sum_b) sa_b, sb, out) (i8_acc; one block: i8_out, addend the bias, or the residual plus the bias);
   - weights: one grid and one scale a row (i8_quantize_row: m0 and m over the whole row), sb;
   - wgrad: blocks of I8_WBLOCK = 32 rows (the GPU's tile: a SIMD group's rows), each column of a block its own grid,
     rotated along the rows by H32 (32 x 32), scale k = 4 for X's codes and 8 for dY's (H32^T H32 = 32 I = 4 8),
     each block's exact sum accumulated in fp32 (i8_acc).
   A network's weights are stored as these codes and scales (model.swift writes them, forward.c reads them); the fp32
   master weights are the optimizer's alone. */
#ifndef INT8MM_H
#define INT8MM_H

#ifdef __METAL_VERSION__
#define I8_RINT(x) metal::rint(x)
#define I8_FLOOR(x) metal::floor(x)
#define I8_FMA(a, b, c) metal::fma(a, b, c)
#define I8_DIV(a, b) metal::precise::divide(a, b)
#define I8_CLAMP(x) metal::clamp(x, -127.0f, 127.0f)
#else
#include <math.h>
#include <stdint.h>
#include <string.h>
#define I8_RINT(x) rintf(x)
#define I8_FLOOR(x) floorf(x)
#define I8_FMA(a, b, c) fmaf(a, b, c)
#define I8_DIV(a, b) ((a) / (b))
#define I8_CLAMP(x) ((x) < -127.0f ? -127.0f : (x) > 127.0f ? 127.0f : (x))
#endif

enum { I8_BLOCK = 64, I8_WBLOCK = 32, I8_GRID = 2047 };

static inline uint32_t i8_mix(uint32_t h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
/* the grid: a power of two (no divide), gi = 2^(10 - e) for m0 in [2^e, 2^(e+1)), so m0 gi in [1024, 2048) and the
   grid integers |xi| <= 2048 (exact in fp16; 64 of them sum below 2^24); its step 1 / gi */
static inline uint32_t i8_bits(float x) {
#ifdef __METAL_VERSION__
  return as_type<uint32_t>(x);
#else
  uint32_t b; memcpy(&b, &x, 4); return b;
#endif
}
static inline float i8_from_bits(uint32_t b) {
#ifdef __METAL_VERSION__
  return as_type<float>(b);
#else
  float x; memcpy(&x, &b, 4); return x;
#endif
}
static inline uint32_t i8_grid_exponent(float m0) { const uint32_t e = (i8_bits(m0) >> 23) & 0xffu; return e < 11u ? 11u : e > 243u ? 243u : e; }
static inline float i8_grid_inverse(float m0) { return m0 > 0 ? i8_from_bits((264u - i8_grid_exponent(m0)) << 23) : 0.0f; }   /* xi = rint(x gi) */
static inline float i8_grid_step(float m0) { return i8_from_bits((i8_grid_exponent(m0) - 10u) << 23); }
static inline float i8_grid(float x, float gi) { return I8_RINT(x * gi); }
/* 1 / m for m >= 1 (the rotated grid's largest magnitude, an integer): the bit estimate and three Newton steps by fma
   (act.h's reciprocal), the same operations on every platform */
static inline float i8_recip(float y) {
  float r = i8_from_bits(0x7ef311c7u - i8_bits(y));
  for (int i = 0; i < 3; i++) r = I8_FMA(r, I8_FMA(-y, r, 1.0f), r);
  return r;
}
static inline float i8_inverse(float m) { return m > 0 ? 127.0f * i8_recip(m) : 0.0f; }   /* codes = y i8_inverse(max|y|), clamped */
/* the codes' scale: q s ~ H x / k, s = m step / (127 k) (two roundings, the constant 1 / (127 k) a float) */
static inline float i8_scale(float m, float m0, float k) { return m * i8_grid_step(m0) * (1.0f / (127.0f * k)); }
/* the stochastic uniform of element `at`: one hash a pair of elements (at, at ^ 1), its low 16 bits element 2j's, its
   high 16 bits 2j + 1's */
static inline float i8_uniform(uint32_t seed, uint32_t at) {
  const uint32_t h = i8_mix(seed ^ ((at >> 1) * 0x9e3779b1u));
  return (float)((h >> (16 * (at & 1))) & 0xffffu) * (1.0f / 65536.0f);
}
static inline int8_t i8_code(float v, float inv, uint32_t seed, uint32_t at) {   /* seed 0: to nearest */
  const float x = v * inv;
  const float r = seed ? I8_FLOOR(x + i8_uniform(seed, at)) : I8_RINT(x);
  return (int8_t)I8_CLAMP(r);
}
static inline float i8_out(int32_t sum, float sa, float sb, float bias) { return I8_FMA((float)sum * sa, sb, bias); }
static inline float i8_acc(int32_t sum, float sa, float sb, float acc) { return I8_FMA((float)sum * sa, sb, acc); }
/* a weight row's value from its master copy: W = g v / |v|, |v|^2 by fma in order, the scale correctly rounded */
static inline float i8_weight_gain(float g, float sumsq) {
#ifdef __METAL_VERSION__
  return metal::precise::divide(g, metal::precise::sqrt(metal::max(sumsq, 1e-30f)));
#else
  return g / sqrtf(sumsq > 1e-30f ? sumsq : 1e-30f);
#endif
}

#ifdef __METAL_VERSION__
/* the GPU's butterfly form of H (weights' kernels, a row in a SIMD group): a 64-block held by eight lanes (li = lane %
   8, its values v[k] the block's elements 8 li + k), unnormalized; on grid integers exact, so the same bits as H on
   the matrix units */
static inline void i8_fwht64_lanes(thread float *v, ushort li) {
  _Pragma("unroll") for (ushort st = 1; st < 8; st <<= 1) for (ushort k = 0; k < 8; k++) if (!(k & st)) { const float a = v[k], c = v[k | st]; v[k] = a + c; v[k | st] = a - c; }
  _Pragma("unroll") for (ushort st = 1; st < 8; st <<= 1) for (ushort k = 0; k < 8; k++) { const float o = metal::simd_shuffle_xor(v[k], st); v[k] = (li & st) ? o - v[k] : v[k] + o; }
}
/* eight codes packed little-endian (element k at byte k) */
static inline uint2 i8_pack8(thread const float *v, float inv, uint seed, uint at0) {
  uint lo = 0, hi = 0;
  _Pragma("unroll") for (uint k = 0; k < 8; k++) { const uint c = uint(uchar(i8_code(v[k], inv, seed, at0 + k))); if (k < 4) lo |= c << (8 * k); else hi |= c << (8 * (k - 4)); }
  return uint2(lo, hi);
}
#endif

#ifndef __METAL_VERSION__
/* the reference forms, on a CPU (the GPU's kernels compute the same) */
/* y = H u, unnormalized, n = 32 or 64, in place (on grid integers: exact) */
static inline void i8_fwht(float *u, int n) {
  for (int st = 1; st < n; st <<= 1) for (int r = 0; r < n; r++) if (!(r & st)) { const float a = u[r], c = u[r | st]; u[r] = a + c; u[r | st] = a - c; }
}
static inline void i8_fwht64(float *u) { i8_fwht(u, 64); }
/* n values (a block): grid, rotated, codes into q (element j's index at0 + j * stride); returns its scale (k) */
static inline float i8_quantize_block(const float *x, int n, int8_t *q, float k, uint32_t seed, uint32_t at0, uint32_t stride) {
  float v[64], m0 = 0, m = 0;
  for (int j = 0; j < n; j++) m0 = fmaxf(m0, fabsf(x[j]));
  const float gi = i8_grid_inverse(m0);
  for (int j = 0; j < n; j++) v[j] = i8_grid(x[j], gi);
  i8_fwht(v, n);
  for (int j = 0; j < n; j++) m = fmaxf(m, fabsf(v[j]));
  const float inv = i8_inverse(m);
  for (int j = 0; j < n; j++) q[j] = i8_code(v[j], inv, seed, at0 + (uint32_t)j * stride);
  return i8_scale(m, m0, k);
}
/* an activation row of `valid` values, padded with 0 to cols (a multiple of 64, <= 256): a scale a 64-block into s;
   element j's index for stochastic rounding at0 + j */
static inline void i8_quantize_blocks(const float *x, int valid, int cols, int8_t *q, float *s, uint32_t seed, uint32_t at0) {
  float v[256];
  for (int j = 0; j < cols; j++) v[j] = j < valid ? x[j] : 0;
  for (int b = 0; b < cols; b += 64) s[b / 64] = i8_quantize_block(v + b, 64, q + b, 8.0f, seed, at0 + (uint32_t)b, 1);
}
/* a weight row: one grid and one scale over the whole row (cols a multiple of 64); returns its scale */
static inline float i8_quantize_row(const float *x, int valid, int cols, int8_t *q) {
  float v[256], m0 = 0, m = 0;
  for (int j = 0; j < cols; j++) { v[j] = j < valid ? x[j] : 0; m0 = fmaxf(m0, fabsf(v[j])); }
  const float gi = i8_grid_inverse(m0);
  for (int j = 0; j < cols; j++) v[j] = i8_grid(v[j], gi);
  for (int b = 0; b < cols; b += 64) i8_fwht(v + b, 64);
  for (int j = 0; j < cols; j++) m = fmaxf(m, fabsf(v[j]));
  const float inv = i8_inverse(m);
  for (int j = 0; j < cols; j++) q[j] = i8_code(v[j], inv, 0, 0);
  return i8_scale(m, m0, 8.0f);
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
/* i8_quantize_blocks to nearest, four-wide (the butterfly's strides 1 and 2 inside a vector as x + s y, s = +-1: on
   grid integers exact); and an exact int8 dot product of k (a multiple of 16) terms */
static inline void i8_quantize_blocks_neon(const float *x, int valid, int cols, int8_t *q, float *s) {
  float32x4_t u[64];
  const float32x4_t s1 = {1, -1, 1, -1}, s2 = {1, 1, -1, -1}, lo = vdupq_n_f32(-127), hi = vdupq_n_f32(127);
  for (int i = 0; i < cols / 4; i++) {
    if (4 * i + 4 <= valid) u[i] = vld1q_f32(x + 4 * i);
    else { float t[4]; for (int k = 0; k < 4; k++) t[k] = 4 * i + k < valid ? x[4 * i + k] : 0; u[i] = vld1q_f32(t); }
  }
  for (int b = 0; b < cols / 4; b += 16) {
    float32x4_t *v = u + b, m4 = vdupq_n_f32(0);
    for (int i = 0; i < 16; i++) m4 = vmaxq_f32(m4, vabsq_f32(v[i]));
    const float m0 = vmaxvq_f32(m4);
    const float32x4_t gi = vdupq_n_f32(i8_grid_inverse(m0));
    for (int i = 0; i < 16; i++) v[i] = vrndnq_f32(vmulq_f32(v[i], gi));
    for (int i = 0; i < 16; i++) v[i] = vfmaq_f32(vtrn1q_f32(v[i], v[i]), vtrn2q_f32(v[i], v[i]), s1);
    for (int i = 0; i < 16; i++) v[i] = vfmaq_f32(vcombine_f32(vget_low_f32(v[i]), vget_low_f32(v[i])), vcombine_f32(vget_high_f32(v[i]), vget_high_f32(v[i])), s2);
    for (int st = 1; st < 16; st <<= 1) for (int i = 0; i < 16; i++) if (!(i & st)) { const float32x4_t a = v[i], c = v[i | st]; v[i] = vaddq_f32(a, c); v[i | st] = vsubq_f32(a, c); }
    m4 = vdupq_n_f32(0);
    for (int i = 0; i < 16; i++) m4 = vmaxq_f32(m4, vabsq_f32(v[i]));
    const float m = vmaxvq_f32(m4);
    const float32x4_t vi = vdupq_n_f32(i8_inverse(m));
    for (int i = 0; i < 16; i += 4) {
      int32x4_t c[4];
      for (int k = 0; k < 4; k++) c[k] = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(v[i + k], vi)), lo), hi));
      vst1q_s8(q + 4 * (b + i), vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(c[0]), vmovn_s32(c[1]))), vmovn_s16(vcombine_s16(vmovn_s32(c[2]), vmovn_s32(c[3])))));
    }
    s[b / 16] = i8_scale(m, m0, 8.0f);
  }
}
/* a 64-block's codes of a (16 at a time) against four rows of b (each `k` apart, at the same block): exact sums */
static inline int32x4_t i8_dot4(const int8_t *a, const int8_t *b, int k) {
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
  for (int j = 0; j < 64; j += 16) {
    const int8x16_t x = vld1q_s8(a + j);
    c0 = vdotq_s32(c0, x, vld1q_s8(b + j)); c1 = vdotq_s32(c1, x, vld1q_s8(b + k + j));
    c2 = vdotq_s32(c2, x, vld1q_s8(b + 2 * k + j)); c3 = vdotq_s32(c3, x, vld1q_s8(b + 3 * k + j));
  }
  return vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3));
}
#endif

/* a network's weights as stored (model.swift writes, forward.c reads): "RI8P", version 2, the matrix count; each matrix
   rows, cols, k (cols padded to 64), has-bias (int32 each), its codes [rows][k] int8, scales [rows] float, bias [rows]
   float (if it has one); all little-endian */
#define I8_PARAMS_MAGIC 0x50384952u
#define I8_PARAMS_VERSION 2
#endif
#endif
