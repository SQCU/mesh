/* The forward's elementwise numerics, defined once (fwd.metal's tiles and forward.c, the deployed forward): the gates'
   sigmoid and tanh, the readout's rms. Every step is an IEEE single operation (fma, add, multiply, an exact scaling
   by a power of two made from its bits; the rms's correctly rounded sqrt and divide), so the GPU (compiled without
   fast math) and the CPU compute the same bits. The platforms' own exp and tanh differ by an ulp, which at the gates'
   fp16 rounding moved a context's codes (a 0.02 difference in its w, measured by deploy); a correctly rounded divide
   a gate made the gates 45% of a layer's time (stages), so the reciprocals are Newton's iteration instead.
   exp: x = n ln2 + r (n = rint(x / ln2), r by a two-part ln2: |r| <= ln2 / 2), e^r by its degree-5 Taylor polynomial
   (truncation below 2.6e-6 relative there: the gates are rounded to fp16, 4.9e-4), times 2^n (its exponent bits).
   1 / y (y >= 1 here): from the bit-pattern estimate 0x7ef311c7 - bits(y) (within 12%), three Newton steps by fma
   (the error squares each step: 12% -> 1.4% -> 2e-4 -> 4e-8).
   The readout's rms is of the 12-bit grid the readout's codes are made from (int8mm.h): the grid integers' squares
   summed exactly in int32 (any order), so the GPU's reduction and the CPU's agree. */
#ifndef ROUTE_ACT_H
#define ROUTE_ACT_H
#ifdef __METAL_VERSION__
#define ACT_FMA(a, b, c) metal::fma(a, b, c)
#define ACT_RINT(x) metal::rint(x)
#define ACT_DIV(a, b) metal::precise::divide(a, b)
#define ACT_SQRT(x) metal::precise::sqrt(x)
static inline float act_from_bits(uint b) { return as_type<float>(b); }
static inline uint act_bits(float x) { return as_type<uint>(x); }
#else
#include <math.h>
#include <stdint.h>
#include <string.h>
#define ACT_FMA(a, b, c) fmaf(a, b, c)
#define ACT_RINT(x) rintf(x)
#define ACT_DIV(a, b) ((a) / (b))
#define ACT_SQRT(x) sqrtf(x)
static inline float act_from_bits(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }
static inline uint32_t act_bits(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }
#endif

static inline float act_exp(float x) {   /* |x| <= 40 here (the gates' clamped inputs, twice for tanh) */
  const float n = ACT_RINT(x * 1.44269504f);
  const float r = ACT_FMA(n, -1.42860677e-06f, ACT_FMA(n, -0.693145752f, x));   /* ln2 = 0.693145752 + 1.42860677e-06 */
  float p = 8.33333333e-03f;                                                     /* 1/120 */
  p = ACT_FMA(p, r, 4.16666667e-02f);
  p = ACT_FMA(p, r, 1.66666667e-01f); p = ACT_FMA(p, r, 0.5f); p = ACT_FMA(p, r, 1.0f); p = ACT_FMA(p, r, 1.0f);
  return p * act_from_bits((uint32_t)((int)n + 127) << 23);
}
static inline float act_recip(float y) {   /* y >= 1 */
  float r = act_from_bits(0x7ef311c7u - act_bits(y));
  for (int i = 0; i < 3; i++) r = ACT_FMA(r, ACT_FMA(-y, r, 1.0f), r);
  return r;
}
static inline float act_clamp(float x) { return x < -20.0f ? -20.0f : x > 20.0f ? 20.0f : x; }
static inline float act_sigmoid(float x) { return act_recip(1.0f + act_exp(-act_clamp(x))); }
/* tanh(clamp(x, -20, 20)): near 0 its odd series (|x| < 1/16: x - x^3/3 + 2 x^5/15, error below x^7 / 18), else
   1 - 2 / (e^2x + 1) (its cancellation there under 2^-20 relative) */
static inline float act_tanh(float x) {
  x = act_clamp(x);
  if (x > -0.0625f && x < 0.0625f) { const float q = x * x; return ACT_FMA(x * q, ACT_FMA(q, 0.133333333f, -0.333333333f), x); }
  return 1.0f - 2.0f * act_recip(act_exp(2.0f * x) + 1.0f);
}
/* the readout's 1 / rms over D = 64 values from their grid (m0 their largest magnitude, isum the grid integers' squares'
   exact sum): the grid's step u = m0 / 2047, mean square isum u^2 / 64 */
static inline float act_rms_inverse(int isum, float m0) {
  const float u = ACT_DIV(m0, 2047.0f);
  return ACT_DIV(1.0f, ACT_SQRT(ACT_FMA((float)isum * (u * u), 0.015625f, 1e-6f)));
}
#endif
