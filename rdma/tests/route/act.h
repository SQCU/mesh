/* The forward's elementwise numerics, defined once (fwd.metal's tiles and forward.c, the deployed forward): the gates'
   sigmoid and tanh, the readout's rms. Every step is an IEEE single operation (fma, add, multiply, a correctly
   rounded divide and sqrt, an exact scaling by a power of two), so the GPU (compiled without fast math) and the CPU
   compute the same bits; the platforms' own exp and tanh differ by an ulp, which at the gates' fp16 rounding moved a
   context's codes (a 0.02 difference in its w, measured by deploy).
   exp: x = n ln2 + r (n = rint(x / ln2), r by a two-part ln2: |r| <= ln2 / 2), e^r by its degree-7 Taylor polynomial
   (truncation below 2^-29 relative there), scaled by 2^n. */
#ifndef ROUTE_ACT_H
#define ROUTE_ACT_H
#ifdef __METAL_VERSION__
#define ACT_FMA(a, b, c) metal::fma(a, b, c)
#define ACT_RINT(x) metal::rint(x)
#define ACT_LDEXP(x, n) metal::ldexp(x, n)
#define ACT_DIV(a, b) metal::precise::divide(a, b)
#define ACT_SQRT(x) metal::precise::sqrt(x)
#else
#include <math.h>
#define ACT_FMA(a, b, c) fmaf(a, b, c)
#define ACT_RINT(x) rintf(x)
#define ACT_LDEXP(x, n) ldexpf(x, n)
#define ACT_DIV(a, b) ((a) / (b))
#define ACT_SQRT(x) sqrtf(x)
#endif

static inline float act_exp(float x) {   /* |x| <= 40 here (the gates' clamped inputs, twice for tanh) */
  const float n = ACT_RINT(x * 1.44269504f);
  const float r = ACT_FMA(n, -1.42860677e-06f, ACT_FMA(n, -0.693145752f, x));   /* ln2 = 0.693145752 + 1.42860677e-06 */
  float p = 1.98412698e-04f;                                                     /* 1/5040 */
  p = ACT_FMA(p, r, 1.38888889e-03f); p = ACT_FMA(p, r, 8.33333333e-03f); p = ACT_FMA(p, r, 4.16666667e-02f);
  p = ACT_FMA(p, r, 1.66666667e-01f); p = ACT_FMA(p, r, 0.5f); p = ACT_FMA(p, r, 1.0f); p = ACT_FMA(p, r, 1.0f);
  return ACT_LDEXP(p, (int)n);
}
static inline float act_clamp(float x) { return x < -20.0f ? -20.0f : x > 20.0f ? 20.0f : x; }
static inline float act_sigmoid(float x) { return ACT_DIV(1.0f, 1.0f + act_exp(-act_clamp(x))); }
/* tanh(clamp(x, -20, 20)): near 0 its odd series (|x| < 1/16: x - x^3/3 + 2 x^5/15, error below x^7 / 18), else
   1 - 2 / (e^2x + 1) (its cancellation there under 2^-20 relative) */
static inline float act_tanh(float x) {
  x = act_clamp(x);
  if (x > -0.0625f && x < 0.0625f) { const float q = x * x; return ACT_FMA(x * q, ACT_FMA(q, 0.133333333f, -0.333333333f), x); }
  return 1.0f - ACT_DIV(2.0f, act_exp(2.0f * x) + 1.0f);
}
/* the readout's 1 / rms over D = 64 values whose squares were summed in this order: eight lanes of eight (a lane's in
   order, by fma), then the lanes' sums pairwise at strides 1, 2, 4 (act_sumsq64 on a CPU) */
static inline float act_rms_inverse(float sumsq) { return ACT_DIV(1.0f, ACT_SQRT(ACT_FMA(sumsq, 0.015625f, 1e-6f))); }
#ifndef __METAL_VERSION__
static inline float act_sumsq64(const float *x) {
  float s[8];
  for (int l = 0; l < 8; l++) { float q = 0; for (int k = 0; k < 8; k++) q = fmaf(x[8 * l + k], x[8 * l + k], q); s[l] = q; }
  for (int st = 1; st < 8; st <<= 1) for (int l = 0; l < 8; l++) if (!(l & st)) { const float t = s[l] + s[l | st]; s[l] = t; s[l | st] = t; }
  return s[0];
}
#endif
#endif
