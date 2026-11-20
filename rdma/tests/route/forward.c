/* The policy's forward on a CPU, for deployment beside the bridge (route.h): net.swift's network, its weights as the
   trainer stored them (int8 codes and scales, int8mm.h's format: the codes the GPU trained with, read, never cast
   again), its products int8 x int8 -> int32 by int8mm.h's numerics (each input row rotated and quantized to nearest,
   exact sums by NEON's sdot, out = fma(float(sum) sa, sb, b)), each later step the GPU's own expression. Only the
   transcendentals (sigmoid, tanh, the rms's 1/sqrt) are the CPU's (vForce) rather than the GPU's, so an activation's
   code can differ where a value sits on a rounding boundary. Matrices in NetLayout's order: inp [D][IN], zh0 [2D][D],
   up0 [E][D], down0 [D][E], zh1, up1, down1, out [FS][D]. */
#include "route.h"
#include "int8mm.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = ROUTE_D, E = ROUTE_E, IN = ROUTE_IN, FS = ROUTE_FS, L = ROUTE_L };
static const int shapes[8][3] = {{D, IN, 1}, {2 * D, D, 1}, {E, D, 1}, {D, E, 1}, {2 * D, D, 1}, {E, D, 1}, {D, E, 1}, {FS, D, 0}};

int route_params_read(struct route_params *p, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  struct route_mat *m[8] = {&p->inp, &p->zh[0], &p->up[0], &p->down[0], &p->zh[1], &p->up[1], &p->down[1], &p->out};
  uint32_t h[3];
  int ok = fread(h, 4, 3, f) == 3 && h[0] == I8_PARAMS_MAGIC && h[1] == 1 && h[2] == 8;
  for (int i = 0; ok && i < 8; i++) {
    int32_t d[4];
    ok = fread(d, 4, 4, f) == 4 && d[0] == shapes[i][0] && d[1] == shapes[i][1] && d[2] == (shapes[i][1] + 63) / 64 * 64 && d[3] == shapes[i][2];
    if (!ok) break;
    struct route_mat *a = m[i];
    a->rows = d[0]; a->cols = d[1]; a->k = d[2];
    a->q = malloc((size_t)a->rows * a->k); a->s = malloc((size_t)a->rows * sizeof(float)); a->b = d[3] ? malloc((size_t)a->rows * sizeof(float)) : 0;
    ok = a->q && a->s && fread(a->q, 1, (size_t)a->rows * a->k, f) == (size_t)a->rows * a->k && fread(a->s, 4, (size_t)a->rows, f) == (size_t)a->rows
         && (!a->b || fread(a->b, 4, (size_t)a->rows, f) == (size_t)a->rows);
  }
  fclose(f);
  return ok ? 0 : -1;
}

/* y [n][rows of a] = act(x [n][valid] a^T + b) (+ res): the GPU's epilogue; act 0 none, 1 relu, 2 gates. Four outputs a
   pass, sharing the row's codes; their exact sums by pairwise adds; y = fma(float(sum) sx, s[o], b[o]) four-wide */
/* one row: y [rows of a] = act(x [valid] a^T + b) (+ res), the GPU's epilogue; act 0 none, 1 relu, 2 gates. Four
   outputs a pass, sharing the row's codes; their exact sums by pairwise adds; y = fma(float(sum) sx, s[o], b[o]) */
static void product_row(const struct route_mat *a, const float *x, float *out, const float *res, int act) {
  int8_t qx[256];
  const int k = a->k;
  const float sx = i8_quantize_row_neon(x, a->cols, k, qx);
  for (int o = 0; o < a->rows; o += 4) {
    const int32x4_t sum = i8_dot4(qx, a->q + (size_t)o * k, k);
    const float32x4_t bias = a->b ? vld1q_f32(a->b + o) : vdupq_n_f32(0);
    float32x4_t v = vfmaq_f32(bias, vmulq_n_f32(vcvtq_f32_s32(sum), sx), vld1q_f32(a->s + o));   /* i8_out, four-wide */
    if (act == 1) v = vmaxq_f32(v, vdupq_n_f32(0));
    if (res) v = vaddq_f32(v, vld1q_f32(res + o));
    vst1q_f32(out + o, v);
  }
  if (act == 2) {   /* the gates (vForce): z = 1 / (1 + exp(-y)), h~ = tanh(clamp(y, -20, 20)) */
    const int half = a->rows / 2;
    float t[128];
    for (int j = 0; j < half; j++) t[j] = -out[j];
    vvexpf(t, t, &half);
    for (int j = 0; j < half; j++) out[j] = 1 / (1 + t[j]);
    for (int j = half; j < a->rows; j++) out[j] = out[j] < -20 ? -20 : out[j] > 20 ? 20 : out[j];
    vvtanhf(out + half, out + half, &half);
    for (int j = 0; j < a->rows; j++) out[j] = (float)(_Float16)out[j];   /* stored fp16, as the GPU's gates are (net.swift ZC) */
  }
}
/* n rows, independent */
static void product(const struct route_mat *a, const float *x, int n, int ld, float *y, const float *res, int act) {
  for (int i = 0; i < n; i++) product_row(a, x + i * (size_t)ld, y + i * (size_t)a->rows, res ? res + i * (size_t)a->rows : 0, act);
}

size_t route_forward_scratch(int nc) { const int n = nc > L ? nc : L; return (size_t)n * (4 * D + E) + 64; }
/* the lanes in turn from the carried state (read and advanced): the refresh's state the contexts step from */
void route_forward_lanes(const struct route_params *p, const float *lanes, int nl, float *state, float *s) {
  float *x = s, *zc = x + (size_t)L * D, *xm = zc + (size_t)L * 2 * D, *u = xm + (size_t)L * D;
  product(&p->inp, lanes, nl, IN, x, 0, 0);
  for (int l = 0; l < 2; l++) {
    float *h = state + l * D;
    product(&p->zh[l], x, nl, D, zc, 0, 2);
    for (int t = 0; t < nl; t++) for (int d = 0; d < D; d++) {   /* net.metal's scan_fwd, m = 1 */
      const float mz = zc[t * 2 * D + d];
      h[d] = fmaf(1.0f - mz, h[d], mz * zc[t * 2 * D + D + d]);
      xm[t * D + d] = x[t * D + d] + h[d];
    }
    if (l == 0) { product(&p->up[0], xm, nl, D, u, 0, 1); product(&p->down[0], u, nl, E, x, xm, 0); }
  }
}
/* one context's w [FS]: one step from the refresh's state (its token) */
void route_forward_context(const struct route_params *p, const float *token, const float *state, float *w) {
  float cx[D], czc[2 * D], cxm[D], cu[E];
  product_row(&p->inp, token, cx, 0, 0);
  for (int l = 0; l < 2; l++) {
    const float *h = state + l * D;
    product_row(&p->zh[l], cx, czc, 0, 2);
    for (int d = 0; d < D; d++) cxm[d] = cx[d] + fmaf(czc[d], czc[D + d] - h[d], h[d]);   /* ctx_fwd */
    product_row(&p->up[l], cxm, cu, 0, 1); product_row(&p->down[l], cu, cx, cxm, 0);
  }
  float m = 0; for (int d = 0; d < D; d++) m += cx[d] * cx[d];   /* rms (rms_fwd), then the readout */
  const float inv = 1 / sqrtf(m / D + 1e-6f);
  for (int d = 0; d < D; d++) cx[d] *= inv;
  product_row(&p->out, cx, w, 0, 0);
}
/* one refresh, every context (the trainer's check, DEPLOY): the lanes, then each context */
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *s) {
  route_forward_lanes(p, lanes, nl, state, s);
  for (int i = 0; i < nc; i++) route_forward_context(p, tokens + (size_t)i * IN, state, w + (size_t)i * FS);
}
