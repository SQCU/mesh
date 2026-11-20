/* The policy's forward on a CPU, for deployment beside the bridge (route.h): the same two recurrent layers as
   policy.swift, over one decision point's lane tokens from a carried state, then one step for each context's token,
   then w = out . rms(output). Accelerate's BLAS for the products, so they run on the CPU's matrix units.
   theta's layout is policy.swift's Layout: inp [D][IN], binp; per layer z [D][GW], bz, (r [D][GW], br,) h [D][GW], bh,
   up [E][D], bu, down [D][E], bd; out [FS][D]. GW = 2D for the GRU (columns [h; x]), D for minGRU. */
#define ACCELERATE_NEW_LAPACK
#include "route.h"
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <string.h>

enum { D = ROUTE_D, E = ROUTE_E, IN = ROUTE_IN, FS = ROUTE_FS, L = ROUTE_L, C = ROUTE_CONTEXTS };

int route_params_count(int full) {
  const int gw = full ? 2 * D : D, layer = (full ? 3 : 2) * (D * gw + D) + E * D + E + D * E + D;
  return D * IN + D + 2 * layer + FS * D;
}
void route_params_bind(struct route_params *p, const float *theta, int full) {
  const int gw = full ? 2 * D : D;
  p->full = full; p->inp = theta; theta += D * IN; p->binp = theta; theta += D;
  for (int l = 0; l < 2; l++) {
    p->z[l] = theta; theta += D * gw; p->bz[l] = theta; theta += D;
    if (full) { p->r[l] = theta; theta += D * gw; p->br[l] = theta; theta += D; } else p->r[l] = p->br[l] = 0;
    p->h[l] = theta; theta += D * gw; p->bh[l] = theta; theta += D;
    p->up[l] = theta; theta += E * D; p->bu[l] = theta; theta += E;
    p->down[l] = theta; theta += D * E; p->bd[l] = theta; theta += D;
  }
  p->out = theta;
}

static inline float sig(float v) { return 1 / (1 + expf(-v)); }
/* y [n][m] = x [n][k] W^T (W [m][k], row stride ws) + b (or 0) (+ y if acc) */
static void dense(const float *x, int n, int k, const float *W, int ws, int m, const float *b, float *y, int acc) {
  if (b && !acc) for (int i = 0; i < n; i++) memcpy(y + (size_t)i * m, b, (size_t)m * sizeof(float));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, k, 1, x, k, W, ws, (b || acc) ? 1 : 0, y, m);
}
/* one layer's feed-forward on n rows in place: x = x + down relu(up x + bu) + bd */
static void ffn(const struct route_params *p, int l, float *x, int n, float *u) {
  dense(x, n, D, p->up[l], D, E, p->bu[l], u, 0);
  for (int i = 0; i < n * E; i++) u[i] = u[i] > 0 ? u[i] : 0;
  dense(u, n, E, p->down[l], E, D, p->bd[l], x, 1);
  for (int i = 0; i < n; i++) cblas_saxpy(D, 1, p->bd[l], 1, x + (size_t)i * D, 1);   /* dense(acc) skipped the bias: add it */
}
/* the GRU's gates for n rows sharing (or not) a state hp: z, c into zo, co; hp rows `hs` apart (0: one shared row) */
static void gru(const struct route_params *p, int l, const float *x, const float *hp, int hs, int n, float *zo, float *co, float *t1, float *t2) {
  const int gw = 2 * D;
  /* z = sig(Wz_h hp + Wz_x x + bz), r likewise; c = tanh(Wh_h (r hp) + Wh_x x + bh) */
  dense(x, n, D, p->z[l] + D, gw, D, p->bz[l], zo, 0);
  dense(x, n, D, p->r[l] + D, gw, D, p->br[l], t1, 0);
  if (hs) { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, D, D, 1, hp, hs, p->z[l], gw, 1, zo, D); cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, D, D, 1, hp, hs, p->r[l], gw, 1, t1, D); }
  else {
    float a[D], b[D];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, D, D, 1, p->z[l], gw, hp, 1, 0, a, 1); cblas_sgemv(CblasRowMajor, CblasNoTrans, D, D, 1, p->r[l], gw, hp, 1, 0, b, 1);
    for (int i = 0; i < n; i++) { cblas_saxpy(D, 1, a, 1, zo + (size_t)i * D, 1); cblas_saxpy(D, 1, b, 1, t1 + (size_t)i * D, 1); }
  }
  for (int i = 0; i < n * D; i++) { zo[i] = sig(zo[i]); t2[i] = sig(t1[i]) * hp[(size_t)(hs ? i / D * hs : 0) + i % D]; }
  dense(x, n, D, p->h[l] + D, gw, D, p->bh[l], co, 0);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, D, D, 1, t2, D, p->h[l], gw, 1, co, D);
  for (int i = 0; i < n * D; i++) co[i] = tanhf(co[i]);
}

/* scratch: at least route_forward_scratch(nc) floats */
size_t route_forward_scratch(int nc) { const int n = nc > L ? nc : L; return (size_t)n * (4 * D + E) + 16; }
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *s) {
  const int n = nc > L ? nc : L;
  float *x = s, *z = x + (size_t)n * D, *c = z + (size_t)n * D, *t1 = c + (size_t)n * D, *u = t1 + (size_t)n * D;
  /* the lanes, in turn, from the carried state */
  dense(lanes, nl, IN, p->inp, IN, D, p->binp, x, 0);
  for (int l = 0; l < 2; l++) {
    float *h = state + l * D;
    if (!p->full) {   /* minGRU: every position's gates at once, then the scan */
      dense(x, nl, D, p->z[l], D, D, p->bz[l], z, 0); dense(x, nl, D, p->h[l], D, D, p->bh[l], c, 0);
      for (int t = 0; t < nl; t++) for (int d = 0; d < D; d++) {
        const float zz = sig(z[t * D + d]), hn = (1 - zz) * h[d] + zz * tanhf(c[t * D + d]);
        h[d] = hn; x[t * D + d] += hn;
      }
    } else for (int t = 0; t < nl; t++) {   /* the GRU: one position at a time */
      float t2[D];
      gru(p, l, x + t * D, h, 0, 1, z, c, t1, t2);
      for (int d = 0; d < D; d++) { const float hn = (1 - z[d]) * h[d] + z[d] * c[d]; h[d] = hn; x[t * D + d] += hn; }
    }
    ffn(p, l, x, nl, u);
  }
  /* the contexts: one step each from that state */
  dense(tokens, nc, IN, p->inp, IN, D, p->binp, x, 0);
  for (int l = 0; l < 2; l++) {
    const float *h = state + l * D;
    if (!p->full) {
      dense(x, nc, D, p->z[l], D, D, p->bz[l], z, 0); dense(x, nc, D, p->h[l], D, D, p->bh[l], c, 0);
      for (int i = 0; i < nc * D; i++) { const float zz = sig(z[i]); x[i] += (1 - zz) * h[i % D] + zz * tanhf(c[i]); }
    } else {
      gru(p, l, x, h, 0, nc, z, c, t1, u);
      for (int i = 0; i < nc * D; i++) x[i] += (1 - z[i]) * h[i % D] + z[i] * c[i];
    }
    ffn(p, l, x, nc, u);
  }
  for (int i = 0; i < nc; i++) {   /* rms, then the readout */
    float m = 0; for (int d = 0; d < D; d++) m += x[i * D + d] * x[i * D + d];
    const float inv = 1 / sqrtf(m / D + 1e-6f);
    for (int d = 0; d < D; d++) x[i * D + d] *= inv;
  }
  dense(x, nc, D, p->out, D, FS, 0, w, 0);
}
