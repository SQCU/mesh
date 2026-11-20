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
static void product(const struct route_mat *a, const float *x, int n, int ld, float *y, const float *res, int act, int8_t *qx) {
  const int k = a->k;
  for (int i = 0; i < n; i++) {
    const float sx = i8_quantize_row_neon(x + (size_t)i * ld, a->cols, k, qx);
    float *out = y + (size_t)i * a->rows;
    for (int o = 0; o < a->rows; o += 4) {
      const int8_t *w = a->q + (size_t)o * k;
      const int32x4_t sum = i8_dot4(qx, w, k);
      const float32x4_t bias = a->b ? vld1q_f32(a->b + o) : vdupq_n_f32(0);
      float v[4];
      vst1q_f32(v, vfmaq_f32(bias, vmulq_n_f32(vcvtq_f32_s32(sum), sx), vld1q_f32(a->s + o)));   /* i8_out, four-wide */
      for (int q = 0; q < 4; q++) {
        float t = v[q];
        if (act == 1) t = t > 0 ? t : 0;
        if (res) t += res[(size_t)i * a->rows + o + q];
        out[o + q] = t;
      }
    }
    if (act == 2) {   /* the gates, a row at once (vForce): z = 1 / (1 + exp(-y)), h~ = tanh(clamp(y, -20, 20)) */
      const int half = a->rows / 2;
      float t[128];
      for (int j = 0; j < half; j++) t[j] = -out[j];
      vvexpf(t, t, &half);
      for (int j = 0; j < half; j++) out[j] = 1 / (1 + t[j]);
      for (int j = half; j < a->rows; j++) out[j] = out[j] < -20 ? -20 : out[j] > 20 ? 20 : out[j];
      vvtanhf(out + half, out + half, &half);
    }
  }
}

size_t route_forward_scratch(int nc) { const int n = nc > L ? nc : L; return (size_t)n * (4 * D + E) + 64; }
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *s) {
  const int n = nc > L ? nc : L;
  float *x = s, *zc = x + (size_t)n * D, *xm = zc + (size_t)n * 2 * D, *u = xm + (size_t)n * D;
  int8_t qx[256];
  /* the lanes, in turn, from the carried state: layer 0 whole, layer 1 its states alone */
  product(&p->inp, lanes, nl, IN, x, 0, 0, qx);
  for (int l = 0; l < 2; l++) {
    float *h = state + l * D;
    product(&p->zh[l], x, nl, D, zc, 0, 2, qx);
    for (int t = 0; t < nl; t++) for (int d = 0; d < D; d++) {   /* net.metal's scan_fwd, m = 1 */
      const float mz = zc[t * 2 * D + d];
      h[d] = fmaf(1.0f - mz, h[d], mz * zc[t * 2 * D + D + d]);
      xm[t * D + d] = x[t * D + d] + h[d];
    }
    if (l == 0) { product(&p->up[0], xm, nl, D, u, 0, 1, qx); product(&p->down[0], u, nl, E, x, xm, 0, qx); }
  }
  /* the contexts: one step each from that state */
  product(&p->inp, tokens, nc, IN, x, 0, 0, qx);
  for (int l = 0; l < 2; l++) {
    const float *h = state + l * D;
    product(&p->zh[l], x, nc, D, zc, 0, 2, qx);
    for (int i = 0; i < nc; i++) for (int d = 0; d < D; d++) {   /* ctx_fwd */
      const float z = zc[i * 2 * D + d];
      xm[i * D + d] = x[i * D + d] + fmaf(z, zc[i * 2 * D + D + d] - h[d], h[d]);
    }
    product(&p->up[l], xm, nc, D, u, 0, 1, qx); product(&p->down[l], u, nc, E, x, xm, 0, qx);
  }
  for (int i = 0; i < nc; i++) {   /* rms (rms_fwd), then the readout */
    float m = 0; for (int d = 0; d < D; d++) m += x[i * D + d] * x[i * D + d];
    const float inv = 1 / sqrtf(m / D + 1e-6f);
    for (int d = 0; d < D; d++) x[i * D + d] *= inv;
  }
  product(&p->out, x, nc, D, w, 0, 0, qx);
}
