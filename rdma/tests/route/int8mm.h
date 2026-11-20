/* The int8 product's numerics, defined once: every user includes this header (route.metal's GPU kernels, forward.c's
   deployed CPU forward, stages.swift's checks) rather than restating them.
   A product C = A B^T over a reduction of K terms, int8 x int8 -> int32:
   - each operand row (or wgrad's column block) rotated along the reduction within 64-blocks by the normalized
     Walsh-Hadamard transform y = H x / 8, H's entries +-1, its butterfly strides in the order 1, 2, 4, 8, 16, then 32
     with the 1/8 in the same step (i8_fwht64; on the GPU i8_fwht64_lanes, eight lanes a block, eight values a lane:
     the same steps); both operands rotated alike, so the product is unchanged;
   - one scale a row: s = max|y| / 127, codes q = y 127 / max|y| rounded to nearest (even), or stochastically
     (floor(x + u), u from i8_mix of the element's index and a seed: unbiased), clamped to +-127 (i8_code);
   - the int32 sum exact while K x 127^2 < 2^31 (K < 133,000); forward and dgrad one sum over the whole K (<= 256),
     wgrad blocks of I8_WBLOCK rows (one scale a column a block), each block's sum accumulated in fp32 (i8_acc).
     I8_WBLOCK is the producers' tile (64 rows), so a tile makes its own wgrad codes; measured (stages, the wgrad
     product at the network's shapes): 32-row blocks 32-35 TOP/s, 64 37-40, 128 43-46;
   - out = fma(float(sum) sa, sb, bias) (i8_out): one rounding a step, so every user's results agree bit for bit.
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
#define I8_RINT(x) rintf(x)
#define I8_FLOOR(x) floorf(x)
#define I8_FMA(a, b, c) fmaf(a, b, c)
#define I8_DIV(a, b) ((a) / (b))
#define I8_CLAMP(x) ((x) < -127.0f ? -127.0f : (x) > 127.0f ? 127.0f : (x))
#endif

enum { I8_BLOCK = 64, I8_WBLOCK = 64 };

static inline uint32_t i8_mix(uint32_t h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
static inline float i8_inverse(float m) { return m > 0 ? I8_DIV(127.0f, m) : 0.0f; }   /* codes = y i8_inverse(max|y|) */
static inline float i8_scale(float m) { return I8_DIV(m, 127.0f); }                     /* y ~ codes i8_scale(max|y|) */
static inline int8_t i8_code(float v, float inv, uint32_t seed, uint32_t at) {          /* seed 0: to nearest */
  const float x = v * inv;
  const float r = seed ? I8_FLOOR(x + (float)(i8_mix(seed ^ (at * 0x9e3779b1u)) >> 8) * (1.0f / 16777216.0f)) : I8_RINT(x);
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
/* the GPU's form of i8_fwht64: a 64-block held by eight lanes (li = lane % 8, its values v[k] the block's elements
   8 li + k): strides 1, 2, 4 inside the lane, 8, 16, 32 across the lanes (the 1/8 with the last) */
static inline void i8_fwht64_lanes(thread float *v, ushort li) {
  _Pragma("unroll") for (ushort st = 1; st < 8; st <<= 1) for (ushort k = 0; k < 8; k++) if (!(k & st)) { const float a = v[k], c = v[k | st]; v[k] = a + c; v[k | st] = a - c; }
  _Pragma("unroll") for (ushort st = 1; st < 4; st <<= 1) for (ushort k = 0; k < 8; k++) { const float o = metal::simd_shuffle_xor(v[k], st); v[k] = (li & st) ? o - v[k] : v[k] + o; }
  _Pragma("unroll") for (ushort k = 0; k < 8; k++) { const float o = metal::simd_shuffle_xor(v[k], ushort(4)); v[k] = (li & 4) ? (o - v[k]) * 0.125f : (v[k] + o) * 0.125f; }
}
/* the same transform with the block held strided: lane li holds elements li + 8 k (v[k]): strides 1, 2, 4 across the
   lanes, then 8, 16, 32 inside the lane (the 1/8 with the last): the same steps in the same order, so the same bits.
   (A column pass reads a 64-row column this way: rows li + 8 k of a tile, every lane of a load on its own bank.) */
static inline void i8_fwht64_strided(thread float *v, ushort li) {
  _Pragma("unroll") for (ushort st = 1; st < 8; st <<= 1) _Pragma("unroll") for (ushort k = 0; k < 8; k++) { const float o = metal::simd_shuffle_xor(v[k], st); v[k] = (li & st) ? o - v[k] : v[k] + o; }
  _Pragma("unroll") for (ushort st = 1; st < 4; st <<= 1) _Pragma("unroll") for (ushort k = 0; k < 8; k++) if (!(k & st)) { const float a = v[k], c = v[k | st]; v[k] = a + c; v[k | st] = a - c; }
  _Pragma("unroll") for (ushort k = 0; k < 4; k++) { const float a = v[k], c = v[k + 4]; v[k] = (a + c) * 0.125f; v[k + 4] = (a - c) * 0.125f; }
}
/* the eight lanes' largest |v| */
static inline float i8_absmax_lanes(thread const float *v) {
  float m = 0;
  _Pragma("unroll") for (ushort k = 0; k < 8; k++) m = metal::max(m, metal::abs(v[k]));
  _Pragma("unroll") for (ushort st = 1; st < 8; st <<= 1) m = metal::max(m, metal::simd_shuffle_xor(m, st));
  return m;
}
/* eight codes packed little-endian (element k at byte k) */
static inline uint2 i8_pack8(thread const float *v, float inv, uint seed, uint at0) {
  uint lo = 0, hi = 0;
  _Pragma("unroll") for (uint k = 0; k < 8; k++) { const uint c = uint(uchar(i8_code(v[k], inv, seed, at0 + k))); if (k < 4) lo |= c << (8 * k); else hi |= c << (8 * (k - 4)); }
  return uint2(lo, hi);
}
#endif

#ifndef __METAL_VERSION__
/* the reference forms, on a CPU (the GPU's kernels compute the same in SIMD form) */
static inline void i8_fwht64(float *u) {
  for (int st = 1; st < 32; st <<= 1) for (int r = 0; r < 64; r++) if (!(r & st)) { const float a = u[r], c = u[r | st]; u[r] = a + c; u[r | st] = a - c; }
  for (int r = 0; r < 32; r++) { const float a = u[r], c = u[r + 32]; u[r] = (a + c) * 0.125f; u[r + 32] = (a - c) * 0.125f; }
}
/* a row of `valid` values, padded with 0 to cols (a multiple of 64, <= 256): rotated, codes into q; returns its scale.
   Element j's index for stochastic rounding: at0 + j */
static inline float i8_quantize_row(const float *x, int valid, int cols, int8_t *q, uint32_t seed, uint32_t at0) {
  float v[256], m = 0;
  for (int j = 0; j < cols; j++) v[j] = j < valid ? x[j] : 0;
  for (int b = 0; b < cols; b += 64) i8_fwht64(v + b);
  for (int j = 0; j < cols; j++) m = fmaxf(m, fabsf(v[j]));
  const float inv = i8_inverse(m);
  for (int j = 0; j < cols; j++) q[j] = i8_code(v[j], inv, seed, at0 + (uint32_t)j);
  return i8_scale(m);
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
/* the same as i8_quantize_row to nearest, four-wide (strides 1 and 2 inside a vector as x + s y, s = +-1: the same sum
   or difference exactly); and an exact int8 dot product of k (a multiple of 16) terms */
static inline float i8_quantize_row_neon(const float *x, int valid, int cols, int8_t *q) {
  float32x4_t u[64], m4 = vdupq_n_f32(0);
  const float32x4_t s1 = {1, -1, 1, -1}, s2 = {1, 1, -1, -1}, eighth = vdupq_n_f32(0.125f);
  for (int i = 0; i < cols / 4; i++) {
    if (4 * i + 4 <= valid) u[i] = vld1q_f32(x + 4 * i);
    else { float t[4]; for (int k = 0; k < 4; k++) t[k] = 4 * i + k < valid ? x[4 * i + k] : 0; u[i] = vld1q_f32(t); }
  }
  for (int b = 0; b < cols / 4; b += 16) {
    float32x4_t *v = u + b;
    for (int i = 0; i < 16; i++) v[i] = vfmaq_f32(vtrn1q_f32(v[i], v[i]), vtrn2q_f32(v[i], v[i]), s1);
    for (int i = 0; i < 16; i++) v[i] = vfmaq_f32(vcombine_f32(vget_low_f32(v[i]), vget_low_f32(v[i])), vcombine_f32(vget_high_f32(v[i]), vget_high_f32(v[i])), s2);
    for (int st = 1; st < 8; st <<= 1) for (int i = 0; i < 16; i++) if (!(i & st)) { const float32x4_t a = v[i], c = v[i | st]; v[i] = vaddq_f32(a, c); v[i | st] = vsubq_f32(a, c); }
    for (int i = 0; i < 8; i++) { const float32x4_t a = v[i], c = v[i + 8]; v[i] = vmulq_f32(vaddq_f32(a, c), eighth); v[i + 8] = vmulq_f32(vsubq_f32(a, c), eighth); }
  }
  for (int i = 0; i < cols / 4; i++) m4 = vmaxq_f32(m4, vabsq_f32(u[i]));
  const float m = vmaxvq_f32(m4);
  const float32x4_t vi = vdupq_n_f32(i8_inverse(m)), lo = vdupq_n_f32(-127), hi = vdupq_n_f32(127);
  for (int i = 0; i < cols / 4; i += 4) {
    int32x4_t c[4];
    for (int k = 0; k < 4; k++) c[k] = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(u[i + k], vi)), lo), hi));
    vst1q_s8(q + 4 * i, vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(c[0]), vmovn_s32(c[1]))), vmovn_s16(vcombine_s16(vmovn_s32(c[2]), vmovn_s32(c[3])))));
  }
  return i8_scale(m);
}
/* four rows of b (k apart) against a: their exact sums */
static inline int32x4_t i8_dot4(const int8_t *a, const int8_t *b, int k) {
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
  for (int j = 0; j < k; j += 16) {
    const int8x16_t x = vld1q_s8(a + j);
    c0 = vdotq_s32(c0, x, vld1q_s8(b + j)); c1 = vdotq_s32(c1, x, vld1q_s8(b + k + j));
    c2 = vdotq_s32(c2, x, vld1q_s8(b + 2 * k + j)); c3 = vdotq_s32(c3, x, vld1q_s8(b + 3 * k + j));
  }
  return vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3));
}
#endif

/* a network's weights as stored (model.swift writes, forward.c reads): "RI8P", version 1, the matrix count; each matrix
   rows, cols, k (cols padded to 64), has-bias (int32 each), its codes [rows][k] int8, scales [rows] float, bias [rows]
   float (if it has one); all little-endian */
#define I8_PARAMS_MAGIC 0x50384952u
#endif
#endif
