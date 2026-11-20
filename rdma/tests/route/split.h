/* A split's numerics, defined once: the simulator's draws (sim.c, CPU), the credit (net.metal, GPU) and tests/switch.c's
   router include this header rather than restating them (as int8mm.h for the products).

   A draw: a piece at node v still owing destinations S chooses, for each d in S, one of v's deg out-lanes; token
   tok = sum_d k_d deg^i (i the d's rank in S). Its record (struct route_draw) holds what every split of it reads: each
   (destination, out-lane)'s arrival estimate (v's lane at the draw's instant, then the refresh's cost-to-go), whether
   the lane reaches the destination without coming back (valid) and is allowed from the piece's path (mask).
   A split's features phi [FS]: a one-hot per (d, k) chosen, a one-hot per lane used, the latest and mean arrival over
   the stream's unit, the lanes used per destination. Its logit CLIP tanh(w . phi / sqrt FS); the table the softmax
   over the splits that are valid and allowed. */
#ifndef ROUTE_SPLIT_H
#define ROUTE_SPLIT_H

#ifdef __METAL_VERSION__
#define SPLIT_T thread                     /* a record or accumulator the caller holds */
#define SPLIT_W device                     /* a context's w */
#define SPLIT_TANH(x) metal::tanh(x)
#define SPLIT_EXP(x) metal::exp(x)
#define SPLIT_POPCOUNT(x) metal::popcount(x)
#else
#include <math.h>
#include <stdint.h>
#define SPLIT_T
#define SPLIT_W
#define SPLIT_TANH(x) tanhf(x)
#define SPLIT_EXP(x) expf(x)
#define SPLIT_POPCOUNT(x) __builtin_popcount(x)
#endif

enum { SPLIT_N = 6, SPLIT_DEG = 5, SPLIT_FS = 40 };   /* route.h's ROUTE_N, ROUTE_DEG, ROUTE_FS */
#define SPLIT_CLIP 10.0f
#define SPLIT_ROOT 6.32455532f                         /* sqrt(SPLIT_FS) */

/* a draw as made (sim.c) and as credited (net.metal): 160 bytes, no padding. row: the learning row (the caller's);
   a1, a2: the piece's latency and queueing advantages (the loader's) */
struct route_draw { float arrive[SPLIT_N * SPLIT_DEG]; uint32_t valid, mask, S, token, deg, ntok, row; float unit, a1, a2; };

/* split `tok` of a draw: the (d, k) slots it chooses, the lanes it uses, its three summary features, whether it is in
   the table (valid and allowed) */
struct split { uint32_t ok, used, n, slot[SPLIT_N]; float most, mean, share; };
static inline struct split split_of(SPLIT_T const struct route_draw *d, uint32_t tok) {
  struct split sp; sp.ok = 1; sp.used = 0; sp.n = 0;
  float most = 0, sum = 0;
  for (uint32_t dd = 0; dd < SPLIT_N; dd++) {
    if (!((d->S >> dd) & 1)) continue;
    const uint32_t k = tok % d->deg, b = dd * SPLIT_DEG + k; tok /= d->deg;
    sp.ok = sp.ok && ((d->valid >> b) & 1) && ((d->mask >> b) & 1);
    const float ar = d->arrive[b];
    most = ar > most ? ar : most; sum += ar; sp.used |= 1u << k; sp.slot[sp.n++] = b;
  }
  const float c = sp.n ? (float)sp.n : 1.0f;
  sp.most = most / d->unit; sp.mean = sum / c / d->unit; sp.share = (float)SPLIT_POPCOUNT(sp.used) / c;
  return sp;
}
static inline float split_dot(SPLIT_T const struct split *sp, SPLIT_W const float *w) {   /* w . phi */
  float a = w[35] * sp->most + w[36] * sp->mean + w[37] * sp->share;
  for (uint32_t i = 0; i < sp->n; i++) a += w[sp->slot[i]];
  for (uint32_t k = 0; k < SPLIT_DEG; k++) if ((sp->used >> k) & 1) a += w[30 + k];
  return a;
}
static inline float split_tanh(float dot) { const float x = dot / SPLIT_ROOT; return SPLIT_TANH(x < -20.0f ? -20.0f : x > 20.0f ? 20.0f : x); }
static inline float split_logit(float th) { return SPLIT_CLIP * th; }                              /* th: split_tanh */
static inline float split_slope(float th) { return SPLIT_CLIP * (1.0f - th * th) / SPLIT_ROOT; }   /* d logit / d dot */
static inline void split_add(SPLIT_T const struct split *sp, float c, SPLIT_T float *v) {          /* v += c phi */
  for (uint32_t i = 0; i < sp->n; i++) v[sp->slot[i]] += c;
  for (uint32_t k = 0; k < SPLIT_DEG; k++) if ((sp->used >> k) & 1) v[30 + k] += c;
  v[35] += c * sp->most; v[36] += c * sp->mean; v[37] += c * sp->share;
}

/* the table's log-sum-exp and E[psi] (psi = slope phi: d logit / d w), one split at a time (online: the sums rescaled
   as the maximum grows) */
struct split_lse { float m, s, e[SPLIT_FS]; };
static inline void split_lse_init(SPLIT_T struct split_lse *a) { a->m = -INFINITY; a->s = 0; for (uint32_t f = 0; f < SPLIT_FS; f++) a->e[f] = 0; }
static inline void split_lse_add(SPLIT_T struct split_lse *a, float logit, float slope, SPLIT_T const struct split *sp) {
  if (logit > a->m) { const float r = a->s > 0 ? SPLIT_EXP(a->m - logit) : 0.0f; a->s *= r; for (uint32_t f = 0; f < SPLIT_FS; f++) a->e[f] *= r; a->m = logit; }
  const float x = SPLIT_EXP(logit - a->m);
  a->s += x; split_add(sp, x * slope, a->e);
}
#endif
