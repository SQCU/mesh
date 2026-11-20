/* The policy's forward on a CPU, for deployment beside the bridge (route.h): model.swift's network (fwd.metal's kernels), its weights as the
   trainer stored them (int8 codes and scales, int8mm.h's format: the codes the GPU trained with, read, never cast
   again), its products int8 x int8 -> int32 by int8mm.h's numerics (each input row rotated and quantized to nearest,
   exact sums by NEON's sdot a 64-block, out = fma(float(sum_b) sa_b, sb, out) from the bias), each later step the GPU's own expression. The
   elementwise steps (sigmoid, tanh, the rms) are act.h's, as the GPU's: the same bits. Matrices in NetLayout's order: inp [D][IN], zh0 [2D][D],
   up0 [E][D], down0 [D][E], zh1, up1, down1, out [FS][D]. */
#include "route.h"
#include "scan.h"
#include "int8mm.h"
#include "act.h"
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
  int ok = fread(h, 4, 3, f) == 3 && h[0] == I8_PARAMS_MAGIC && h[1] == I8_PARAMS_VERSION && h[2] == 8;
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

/* one row: y [rows of a] = act(x [valid] a^T + addend), the GPU's epilogue (int8mm.h): the row's codes a scale a
   64-block (sx[b], times `mul` if not 1: the readout's 1 / rms), out = addend (the bias, plus res if given), then for
   each block fma(float(sum_b) sx[b], s[o], out), four outputs a pass; act 0 none, 1 relu, 2 gates */
static void product_row_scaled(const struct route_mat *a, const float *x, float *out, const float *res, int act, float mul) {
  int8_t qx[256]; float sx[4];
  const int k = a->k;
  i8_quantize_blocks_neon(x, a->cols, k, qx, sx);
  if (mul != 1.0f) for (int b = 0; b < k / 64; b++) sx[b] *= mul;
  for (int o = 0; o < a->rows; o += 4) {
    float32x4_t v = a->b ? vld1q_f32(a->b + o) : vdupq_n_f32(0);
    if (res) v = vaddq_f32(vld1q_f32(res + o), v);
    const float32x4_t sb = vld1q_f32(a->s + o);
    for (int b = 0; b < k / 64; b++) v = vfmaq_f32(v, vmulq_n_f32(vcvtq_f32_s32(i8_dot4(qx + 64 * b, a->q + (size_t)o * k + 64 * b, k)), sx[b]), sb);   /* i8_acc */
    if (act == 1) v = vmaxq_f32(v, vdupq_n_f32(0));
    vst1q_f32(out + o, v);
  }
  if (act == 2) {   /* the gates (act.h, as the GPU's), fp16 as the GPU's are (fwd.metal) */
    const int half = a->rows / 2;
    for (int j = 0; j < half; j++) out[j] = (float)(_Float16)act_sigmoid(out[j]);
    for (int j = half; j < a->rows; j++) out[j] = (float)(_Float16)act_tanh(out[j]);
  }
}
static void product_row(const struct route_mat *a, const float *x, float *out, const float *res, int act) { product_row_scaled(a, x, out, res, act, 1.0f); }
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
    for (int d = 0; d < D; d++) {   /* fwd.metal's lanes_fwd: the positions' affine maps scanned in scan.h's order */
      float A[L], B[L], hs[L];   /* all L positions, the unused lanes masked (identity), as the GPU's: the state is position L - 1's */
      for (int t = 0; t < L; t++) {
        const float mz = t < nl ? zc[t * 2 * D + d] : 0.0f;
        A[t] = 1.0f - mz; B[t] = mz * (t < nl ? zc[t * 2 * D + D + d] : 0.0f);
      }
      h[d] = scan_run(A, B, L, h[d], hs);
      for (int t = 0; t < nl; t++) xm[t * D + d] = x[t * D + d] + hs[t];
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
    for (int d = 0; d < D; d++) cxm[d] = cx[d] + fmaf(czc[d], czc[D + d] - h[d], h[d]);   /* ctx_fwd's step */
    product_row(&p->up[l], cxm, cu, 0, 1); product_row(&p->down[l], cu, cx, cxm, 0);
  }
  /* the readout: x2's codes, scaled by 1 / rms of the same grid (act.h, as ctx_fwd) */
  float m0 = 0; int isum = 0;
  for (int d = 0; d < D; d++) m0 = fmaxf(m0, fabsf(cx[d]));
  const float gi = i8_grid_inverse(m0);
  for (int d = 0; d < D; d++) { const int z = (int)i8_grid(cx[d], gi); isum += z * z; }
  product_row_scaled(&p->out, cx, w, 0, 0, act_rms_inverse(isum, m0));
}
/* one refresh, every context (the trainer's check, DEPLOY): the lanes, then each context */
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *s) {
  route_forward_lanes(p, lanes, nl, state, s);
  for (int i = 0; i < nc; i++) route_forward_context(p, tokens + (size_t)i * IN, state, w + (size_t)i * FS);
}
