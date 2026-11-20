// The policy's elementwise parts (net.swift): its products are int8.metal's. D = 64 channels, rows of a layer's buffers:
// lane rows first ([run][refresh][lane]: a run's window one sequence of K L positions), then context rows.
#include <metal_stdlib>
#include "int8mm.h"
using namespace metal;
constant uint D = 64;

// minGRU's recurrence over each run's K L positions, a thread a (run, channel): h = (1 - m z) h + m z h~ (a padded lane,
// m = 0, leaves it), from h0 (the run's state at layer l); H every position's state, XM = X + H if xm, next the last.
// zc [rows][2D]: z, then h~ (the gates' product's epilogue applied sigmoid and tanh)
struct SP { uint runs, steps, layer, xm; };
kernel void scan_fwd(device const float *zc [[buffer(0)]], device const float *mask [[buffer(1)]], device const float *h0 [[buffer(2)]],
                     device float *H [[buffer(3)]], device const float *X [[buffer(4)]], device float *XM [[buffer(5)]], device float *next [[buffer(6)]],
                     constant SP &p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.runs * D) return;
  const uint b = i / D, ch = i % D;
  float h = h0[(b * 2 + p.layer) * D + ch];
  for (uint t = 0; t < p.steps; t++) {
    const uint r = b * p.steps + t;
    const float mz = mask[r] * zc[r * 2 * D + ch];
    h = fma(1.0f - mz, h, mz * zc[r * 2 * D + D + ch]);
    H[r * D + ch] = h;
    if (p.xm) XM[r * D + ch] = X[r * D + ch] + h;
  }
  next[(b * 2 + p.layer) * D + ch] = h;
}
// a context row: one step from its refresh's state H[pos[r]]: hn = (1 - z) hp + z h~, XM = X + hn
kernel void ctx_fwd(device const float *zc [[buffer(0)]], device const float *H [[buffer(1)]], device const int *pos [[buffer(2)]],
                    device const float *X [[buffer(3)]], device float *XM [[buffer(4)]], constant uint &rows [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= rows * D) return;
  const uint r = i / D, ch = i % D;
  const float z = zc[r * 2 * D + ch], hp = H[size_t(pos[r]) * D + ch];
  XM[i] = X[i] + fma(z, zc[r * 2 * D + D + ch] - hp, hp);
}
// n = y / rms(y), a SIMD group a row (two channels a lane); inv [rows] kept for the backward
kernel void rms_fwd(device const float *y [[buffer(0)]], device float *n [[buffer(1)]], device float *inv [[buffer(2)]], constant uint &rows [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  const uint r = gid / 32, l = gid % 32;
  if (r >= rows) return;
  const float a = y[r * D + l], b = y[r * D + 32 + l];
  const float s = rsqrt(simd_sum(a * a + b * b) / float(D) + 1e-6f);
  n[r * D + l] = a * s; n[r * D + 32 + l] = b * s;
  if (l == 0) inv[r] = s;
}

// ---- the backward: each task's gradient in the same buffers, tasks stacked ([task][rows][..]) ----
// dy = inv (dn - n mean(dn . n)), for tasks x rows rows
kernel void rms_bwd(device const float *dn [[buffer(0)]], device const float *n [[buffer(1)]], device const float *inv [[buffer(2)]], device float *dy [[buffer(3)]],
                    constant uint2 &p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {   // p: (rows, tasks x rows)
  const uint r = gid / 32, l = gid % 32;
  if (r >= p.y) return;
  const uint q = r % p.x;
  const float a = dn[r * D + l], b = dn[r * D + 32 + l], na = n[q * D + l], nb = n[q * D + 32 + l];
  const float m = simd_sum(a * na + b * nb) / float(D);
  dy[r * D + l] = inv[q] * (a - na * m); dy[r * D + 32 + l] = inv[q] * (b - nb * m);
}
// a context row's step, backward: from dXM (its hn's gradient), the gates' pre-activation gradients dzc and dhp (its
// refresh state's). dXM and dzc at [task][stride][..] from row off; dhp [task][rows][D]
struct CP { uint rows, tasks, stride, off; };
kernel void ctx_bwd(device const float *dXM [[buffer(0)]], device const float *zc [[buffer(1)]], device const float *H [[buffer(2)]], device const int *pos [[buffer(3)]],
                    device float *dzc [[buffer(4)]], device float *dhp [[buffer(5)]], constant CP &p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.tasks * p.rows * D) return;
  const uint t = i / (p.rows * D), r = i / D % p.rows, ch = i % D, row = t * p.stride + p.off + r;
  const float g = dXM[size_t(row) * D + ch], z = zc[size_t(p.off + r) * 2 * D + ch], c = zc[size_t(p.off + r) * 2 * D + D + ch], hp = H[size_t(pos[r]) * D + ch];
  dzc[size_t(row) * 2 * D + ch] = g * (c - hp) * z * (1.0f - z);
  dzc[size_t(row) * 2 * D + D + ch] = g * z * (1.0f - c * c);
  dhp[i] = g * (1.0f - z);
}
// the recurrence, backward, a thread a (task, run, channel), the positions reversed: the state's gradient g gathers, at a
// refresh's last lane, its contexts' dhp (rows seg[run K + k]), and at every position its own (dXM, if local); each
// position's gates' pre-activation gradients into dzc; g carried back through (1 - m z)
struct BP { uint runs, steps, layer, local, refreshes, L, stride, tasks, ctxRows, p0, p1, p2; };
kernel void scan_bwd(device const float *dXM [[buffer(0)]], device const float *dhp [[buffer(1)]], device const uint2 *seg [[buffer(2)]],
                     device const float *zc [[buffer(3)]], device const float *mask [[buffer(4)]], device const float *H [[buffer(5)]], device const float *h0 [[buffer(6)]],
                     device float *dzc [[buffer(7)]], constant BP &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.tasks * p.runs * D) return;
  const uint t = i / (p.runs * D), b = i / D % p.runs, ch = i % D;
  float g = 0;
  for (uint s = p.steps; s-- > 0;) {
    const uint r = b * p.steps + s, row = t * p.stride + r;
    if (s % p.L == p.L - 1) { const uint2 q = seg[b * p.refreshes + s / p.L]; for (uint c = q.x; c < q.y; c++) g += dhp[(size_t(t) * p.ctxRows + c) * D + ch]; }
    if (p.local) g += dXM[size_t(row) * D + ch];
    const float m = mask[r], z = zc[size_t(r) * 2 * D + ch], c = zc[size_t(r) * 2 * D + D + ch];
    const float hprev = s > 0 ? H[size_t(r - 1) * D + ch] : h0[(b * 2 + p.layer) * D + ch];
    dzc[size_t(row) * 2 * D + ch] = m * g * (c - hprev) * z * (1.0f - z);
    dzc[size_t(row) * 2 * D + D + ch] = m * z * g * (1.0f - c * c);
    g *= 1.0f - m * z;
  }
}

// ---- the parameters: a matrix W = diag(g) V / |V| (rows), V pinned to unit rows; its bias b ----
// mats [i]: (V's offset, rows, columns, g's offset), b at g's offset + rows (if it has one)
struct Mat { uint v, rows, cols, g; };
// W [rows][cols] of matrix m (a thread a row)
kernel void weff(device const float *theta [[buffer(0)]], device float *W [[buffer(1)]], constant Mat &m [[buffer(2)]], uint r [[thread_position_in_grid]]) {
  if (r >= m.rows) return;
  float s = 0;
  for (uint j = 0; j < m.cols; j++) { const float v = theta[m.v + r * m.cols + j]; s = fma(v, v, s); }
  const float k = i8_weight_gain(theta[m.g + r], s);   // int8mm.h's: correctly rounded
  for (uint j = 0; j < m.cols; j++) W[r * m.cols + j] = k * theta[m.v + r * m.cols + j];
}
// V's rows back to unit norm (after a step), every matrix's: rows of all of them in a list (matrix index, row)
kernel void pin(device float *theta [[buffer(0)]], device const Mat *mats [[buffer(1)]], device const uint2 *rows [[buffer(2)]], constant uint &n [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const Mat m = mats[rows[i].x]; const uint r = rows[i].y;
  float s = 0;
  for (uint j = 0; j < m.cols; j++) { const float v = theta[m.v + r * m.cols + j]; s = fma(v, v, s); }
  const float k = rsqrt(max(s, 1e-30f));
  for (uint j = 0; j < m.cols; j++) theta[m.v + r * m.cols + j] *= k;
}
// dW [tasks][rows][cols (+1: the bias's, if bias)] -> grads [task][P]: with u = V / |V| a row, dg = dW . u,
// dV = g / |V| (dW - (dW . u) u), db = the extra column
kernel void chain(device const float *dW [[buffer(0)]], device const float *theta [[buffer(1)]], device float *grads [[buffer(2)]], constant Mat &m [[buffer(3)]],
                  constant uint3 &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {   // p: (P, bias, tasks)
  if (i >= p.z * m.rows) return;
  const uint t = i / m.rows, r = i % m.rows, w = m.cols + p.y;
  const device float *d = dW + (size_t(t) * m.rows + r) * w;
  float s = 0, dot = 0;
  for (uint j = 0; j < m.cols; j++) { const float v = theta[m.v + r * m.cols + j]; s = fma(v, v, s); dot = fma(d[j], v, dot); }
  const float k = rsqrt(max(s, 1e-30f)); dot *= k;
  device float *g = grads + size_t(t) * p.x;
  const float gk = theta[m.g + r] * k;
  for (uint j = 0; j < m.cols; j++) g[m.v + r * m.cols + j] = gk * (d[j] - dot * k * theta[m.v + r * m.cols + j]);
  g[m.g + r] = dot;
  if (p.y) g[m.g + m.rows + r] = d[m.cols];
}
// the three gradients' Gram matrix: one threadgroup of 1024, fixed order
kernel void gram(device const float *g [[buffer(0)]], device float *G [[buffer(1)]], constant uint &P [[buffer(2)]], uint l [[thread_position_in_threadgroup]]) {
  threadgroup float part[6][32];
  float s[6] = {0, 0, 0, 0, 0, 0};
  for (uint i = l; i < P; i += 1024) {
    const float a = g[i], b = g[P + i], c = g[2 * P + i];
    s[0] = fma(a, a, s[0]); s[1] = fma(a, b, s[1]); s[2] = fma(a, c, s[2]); s[3] = fma(b, b, s[3]); s[4] = fma(b, c, s[4]); s[5] = fma(c, c, s[5]);
  }
  for (uint k = 0; k < 6; k++) { const float v = simd_sum(s[k]); if (l % 32 == 0) part[k][l / 32] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (l < 6) {
    float v = 0; for (uint q = 0; q < 32; q++) v += part[l][q];
    const uint at[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    G[at[l][0] * 3 + at[l][1]] = v; G[at[l][1] * 3 + at[l][0]] = v;
  }
}
// Adam on the aggregate (a: the three tasks' weights, the step size; c: the bias corrections), in place
kernel void adam(device float *theta [[buffer(0)]], device float *m [[buffer(1)]], device float *v [[buffer(2)]], device const float *g [[buffer(3)]],
                 constant float4 &a [[buffer(4)]], constant float4 &c [[buffer(5)]], constant uint &P [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= P) return;
  const float d = a.x * g[i] + a.y * g[P + i] + a.z * g[2 * P + i];
  m[i] = 0.9f * m[i] + 0.1f * d; v[i] = 0.999f * v[i] + 0.001f * d * d;
  theta[i] -= a.w * (m[i] / c.x) / (sqrt(v[i] / c.y) + 1e-8f);
}
