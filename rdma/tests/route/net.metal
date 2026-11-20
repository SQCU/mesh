// The policy's elementwise parts (net.swift): its products are int8.metal's. D = 64 channels, rows of a layer's buffers:
// lane rows first ([run][refresh][lane]: a run's window one sequence of K L positions), then context rows.
#include <metal_stdlib>
#include "int8mm.h"
#include "split.h"
#include "upgrad.h"
#include "scan.h"
using namespace metal;
constant uint D = 64;

// a 64-wide row held two values a lane (channels l, l + 32) -> its codes and scale, the transform in i8_fwht64's order
// (strides 1..16 across lanes, then 32 inside the lane with the 1/8); several 64-blocks of a row share one scale

// zc (the gates) fp16, dzc (their gradients) bf16: narrower, read only elementwise and by wgrad's column codes.
// minGRU's recurrence over each run's K L positions, a SIMD group a run (two channels a lane): h = (1 - m z) h + m z h~
// (a padded lane, m = 0, leaves it), from h0 (the run's state at layer l); H every position's state; if xm, XM = X + H
// and XM's codes (the feed-forward's input: row r, from code row 0); next the last state. zc [rows][2D]: z, then h~
struct SP { uint runs, steps, layer, xm; };
// the recurrence, forward: a threadgroup a (run, channel), a thread a position; the positions' affine maps scanned
// (scan.h: within each SIMD group, then the groups' totals); H, XM = X + H, and the last state. XM's codes: quant_rows
// after (one row-parallel pass over every row's stored XM)
kernel void scan_fwd(device const half *zc [[buffer(0)]], device const float *mask [[buffer(1)]], device const float *h0 [[buffer(2)]],
                     device float *H [[buffer(3)]], device const float *X [[buffer(4)]], device float *XM [[buffer(5)]], device float *next [[buffer(6)]],
                     constant SP &p [[buffer(7)]], uint w [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float TA[32], TB[32];
  const uint b = w / D, ch = w % D, S = p.steps, r = b * S + t;
  float A = 1, B = 0;
  if (t < S) { const float mz = mask[r] * float(zc[size_t(r) * 2 * D + ch]); A = 1.0f - mz; B = mz * float(zc[size_t(r) * 2 * D + D + ch]); }
  scan_affine_simd(A, B, lane);
  if (lane == 31) { TA[sg] = A; TB[sg] = B; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) { float a = lane < nsg ? TA[lane] : 1.0f, c = lane < nsg ? TB[lane] : 0.0f; scan_affine_simd(a, c, lane); if (lane < nsg) { TA[lane] = a; TB[lane] = c; } }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg > 0) { B = fma(A, TB[sg - 1], B); A = A * TA[sg - 1]; }
  if (t >= S) return;
  const float h = fma(A, h0[(b * 2 + p.layer) * D + ch], B);
  H[size_t(r) * D + ch] = h;
  if (p.xm) XM[size_t(r) * D + ch] = X[size_t(r) * D + ch] + h;
  if (t == S - 1) next[(b * 2 + p.layer) * D + ch] = h;
}
// a context row (a SIMD group): one step from its refresh's state H[pos[r]]: hn = (1 - z) hp + z h~, XM = X + hn, and
// XM's codes at code row (qrow + r)
kernel void ctx_fwd(device const half *zc [[buffer(0)]], device const float *H [[buffer(1)]], device const int *pos [[buffer(2)]],
                    device const float *X [[buffer(3)]], device float *XM [[buffer(4)]], constant uint2 &p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {   // p: (rows, unused)
  const uint r = gid / 32, lane = gid % 32;
  if (r >= p.x) return;
  float v[2];
  for (uint k = 0; k < 2; k++) {
    const uint ch = lane + 32 * k;
    const float z = float(zc[r * 2 * D + ch]), hp = H[size_t(pos[r]) * D + ch];
    v[k] = X[r * D + ch] + fma(z, float(zc[r * 2 * D + D + ch]) - hp, hp);
    XM[r * D + ch] = v[k];
  }
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
// a context row's step, backward (a SIMD group a task's row): from dXM (its hn's gradient), the gates' pre-activation
// gradients dzc (stored, and their codes, stochastic, code row = the row) and dhp (its refresh state's). dXM and dzc at
// [task][stride][..] from row off; dhp [task][rows][D]
struct CP { uint rows, tasks, stride, off, seed, p0, p1, p2; };
kernel void ctx_bwd(device const float *dXM [[buffer(0)]], device const half *zc [[buffer(1)]], device const float *H [[buffer(2)]], device const int *pos [[buffer(3)]],
                    device bfloat *dzc [[buffer(4)]], device float *dhp [[buffer(5)]], constant CP &p [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]) {
  const uint w = gid / 32, lane = gid % 32;
  if (w >= p.tasks * p.rows) return;
  const uint t = w / p.rows, r = w % p.rows, row = t * p.stride + p.off + r;
  float v[4];   // [z channel l, z channel l + 32, h~ channel l, h~ channel l + 32]: blocks 0 (z) and 1 (h~)
  for (uint k = 0; k < 2; k++) {
    const uint ch = lane + 32 * k;
    const float g = dXM[size_t(row) * D + ch], z = float(zc[size_t(p.off + r) * 2 * D + ch]), c = float(zc[size_t(p.off + r) * 2 * D + D + ch]), hp = H[size_t(pos[r]) * D + ch];
    v[k] = g * (c - hp) * z * (1.0f - z); v[2 + k] = g * z * (1.0f - c * c);
    dzc[size_t(row) * 2 * D + ch] = bfloat(v[k]); dzc[size_t(row) * 2 * D + D + ch] = bfloat(v[2 + k]);
    dhp[(size_t(t) * p.rows + r) * D + ch] = g * (1.0f - z);
  }
}
// each refresh's contexts' dhp summed (rows seg[s], contiguous), a thread a (task, refresh, channel): inj [task][refreshes][D]
kernel void seg_sum(device const float *dhp [[buffer(0)]], device const uint2 *seg [[buffer(1)]], device float *inj [[buffer(2)]],
                    constant uint3 &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {   // p: (refreshes, tasks, context rows)
  if (i >= p.y * p.x * D) return;
  const uint t = i / (p.x * D), sgi = i / D % p.x, ch = i % D;
  const uint2 q = seg[sgi];
  float g = 0;
  for (uint c = q.x; c < q.y; c++) g += dhp[(size_t(t) * p.z + c) * D + ch];
  inj[i] = g;
}
// the recurrence, backward, a SIMD group a (task, run), the positions reversed: the state's gradient g gathers, at a
// refresh's last lane, its contexts' dhp (summed by seg_sum: [task][run K + k][D]), and at every position its own (dXM, if local); each
// position's gates' pre-activation gradients into dzc and their codes (stochastic); g carried back through (1 - m z)
struct BP { uint runs, steps, layer, local, refreshes, L, stride, tasks, ctxRows, seed, p1, p2; };
// the recurrence, backward: a threadgroup a (task, run, channel), a thread a position, the positions reversed (thread
// i: position S - 1 - i). At a position g += u (u: its refresh's contexts' dhp at a refresh's last lane, and dXM if local),
// its gates' gradients from g, then g *= a (a = 1 - m z): the affine map g -> a g + a u, scanned (scan.h) from g = 0; a
// position's g is the scan before it plus its u. dzc's codes: quant_rows_b after (a row-parallel pass over the stored bf16)
kernel void scan_bwd(device const float *dXM [[buffer(0)]], device const float *dhp [[buffer(1)]], device const half *zc [[buffer(2)]],
                     device const float *mask [[buffer(3)]], device const float *H [[buffer(4)]], device const float *h0 [[buffer(5)]], device bfloat *dzc [[buffer(6)]],
                     constant BP &p [[buffer(7)]], uint w [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float TA[32], TB[32], G[1024];
  const uint ch = w % D, tb = w / D, t = tb / p.runs, b = tb % p.runs, S = p.steps;
  const uint s = S - 1 - i, r = b * S + s, row = t * p.stride + r;
  float u = 0, a = 1;
  if (i < S) {
    if (p.local) u = dXM[size_t(row) * D + ch];
    if (s % p.L == p.L - 1) u += dhp[(size_t(t) * p.runs * p.refreshes + b * p.refreshes + s / p.L) * D + ch];   // seg_sum's: the refresh's contexts'
    a = 1.0f - mask[r] * float(zc[size_t(r) * 2 * D + ch]);
  }
  float A = a, B = a * u;
  scan_affine_simd(A, B, lane);
  if (lane == 31) { TA[sg] = A; TB[sg] = B; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) { float x = lane < nsg ? TA[lane] : 1.0f, c = lane < nsg ? TB[lane] : 0.0f; scan_affine_simd(x, c, lane); if (lane < nsg) { TA[lane] = x; TB[lane] = c; } }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg > 0) { B = fma(A, TB[sg - 1], B); A = A * TA[sg - 1]; }
  G[i] = B;   // g after this position, from 0
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i >= S) return;
  const float g = (i > 0 ? G[i - 1] : 0.0f) + u;
  const float m = mask[r], z = float(zc[size_t(r) * 2 * D + ch]), cc = float(zc[size_t(r) * 2 * D + D + ch]);
  const float hprev = s > 0 ? H[size_t(r - 1) * D + ch] : h0[(b * 2 + p.layer) * D + ch];
  dzc[size_t(row) * 2 * D + ch] = bfloat(m * g * (cc - hprev) * z * (1.0f - z)); dzc[size_t(row) * 2 * D + D + ch] = bfloat(m * z * g * (1.0f - cc * cc));
}

// ---- the parameters: a matrix W = diag(g) V / |V| (rows), V pinned to unit rows; its bias b ----
// mats [i]: (V's offset, rows, columns, g's offset), b at g's offset + rows (if it has one)
struct Mat { uint v, rows, cols, g; };
// W [rows][cols] of matrix m (a thread a row)
// every matrix's W = g V / |V| a row, one dispatch: rows (matrix, row) as pin's, woff[matrix] its W's offset in the arena
kernel void weff(device const float *theta [[buffer(0)]], device float *Wall [[buffer(1)]], device const Mat *mats [[buffer(2)]], device const uint2 *rows [[buffer(3)]],
                 device const uint *woff [[buffer(4)]], constant uint &n [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const Mat m = mats[rows[i].x]; const uint r = rows[i].y;
  device float *W = Wall + woff[rows[i].x];
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
// dV = g / |V| (dW - (dW . u) u), db = the extra column. Every matrix's rows in one dispatch: rows (matrix, row) as pin's,
// info[matrix] = (its dW's offset in the arena, bias); p: (P, the rows, tasks)
kernel void chain(device const float *dWall [[buffer(0)]], device const float *theta [[buffer(1)]], device float *grads [[buffer(2)]], device const Mat *mats [[buffer(3)]],
                  device const uint2 *rows [[buffer(4)]], device const uint2 *info [[buffer(5)]], constant uint3 &p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.z * p.y) return;
  const uint t = i / p.y, mi = rows[i % p.y].x, r = rows[i % p.y].y;
  const Mat m = mats[mi]; const uint bias = info[mi].y, w = m.cols + bias;
  const device float *d = dWall + info[mi].x + (size_t(t) * m.rows + r) * w;
  float s = 0, dot = 0;
  for (uint j = 0; j < m.cols; j++) { const float v = theta[m.v + r * m.cols + j]; s = fma(v, v, s); dot = fma(d[j], v, dot); }
  const float k = rsqrt(max(s, 1e-30f)); dot *= k;
  device float *g = grads + size_t(t) * p.x;
  const float gk = theta[m.g + r] * k;
  for (uint j = 0; j < m.cols; j++) g[m.v + r * m.cols + j] = gk * (d[j] - dot * k * theta[m.v + r * m.cols + j]);
  g[m.g + r] = dot;
  if (bias) g[m.g + m.rows + r] = d[m.cols];
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
// UPGrad's weights from the Gram matrix of the step's gradients (upgrad.h), for adam: (alpha, rate)
kernel void upgrad_weights(device const float *gram [[buffer(0)]], device float4 &a [[buffer(1)]], constant float &rate [[buffer(2)]], constant float &inv [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {   // inv: the gradients a sum over that many ranks' (their mean's weights)
  if (i > 0) return;
  float G[9], alpha[3];
  for (int k = 0; k < 9; k++) G[k] = gram[k];
  upgrad3(G, alpha);
  a = float4(alpha[0] * inv, alpha[1] * inv, alpha[2] * inv, rate);
}
kernel void adam(device float *theta [[buffer(0)]], device float *m [[buffer(1)]], device float *v [[buffer(2)]], device const float *g [[buffer(3)]],
                 device const float4 &a [[buffer(4)]], constant float4 &c [[buffer(5)]], constant uint &P [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= P) return;
  const float d = a.x * g[i] + a.y * g[P + i] + a.z * g[2 * P + i];
  m[i] = 0.9f * m[i] + 0.1f * d; v[i] = 0.999f * v[i] + 0.001f * d * d;
  theta[i] -= a.w * (m[i] / c.x) / (sqrt(v[i] / c.y) + 1e-8f);
}

// ---- credit: each draw's score function under the learning forward's own w, psi(token) - E_p[psi] over its table
// (split.h, as the simulator's sampler and route_draw_score: one definition), times its importance ratio
// pi_current(token) / pi_behaviour(token) (wb: the w it was drawn under); then each learning row's three tasks' dw ----
constant uint FS = 40;
// a draw a SIMD group, its table's splits across the lanes; each lane's log-sum-exps online (split.h), then the lanes'
// merged. ll: the log ratio
kernel void credit_draws(device const route_draw *draws [[buffer(0)]], device const float *wbRows [[buffer(1)]], device const float *wcRows [[buffer(2)]],
                         device float *diff [[buffer(3)]], device float *ll [[buffer(4)]], device const uint *first [[buffer(5)]], constant uint2 &p [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {   // p: (draws, rows)
  const uint i = gid / 32, lane = gid % 32;
  if (i >= p.x) return;
  const route_draw d = draws[i];
  uint lo = 0, hi = p.y;   // the draw's row: the last row whose first draw is at or before it (first[] ascending)
  while (hi - lo > 1) { const uint mid = (lo + hi) / 2; if (first[mid] <= i) lo = mid; else hi = mid; }
  device const float *wb = wbRows + size_t(lo) * FS, *wc = wcRows + size_t(lo) * FS;
  split_lse cur; split_lse_init(&cur);
  float mb = -INFINITY, sb = 0, psi[SPLIT_FS];
  for (uint f = 0; f < FS; f++) psi[f] = 0;
  const split_set a = split_set_of(&d);   // the table's splits only (split.h)
  const uint chunk = (a.n + 31) / 32;   // a lane's contiguous chunk of the table, walked in registers
  for (split_walk k = split_walk_at(&a, lane * chunk, chunk); k.left; split_walk_next(&k)) {
    const split sp = split_of(&d, k.token);
    const float th = split_tanh(split_dot(&sp, wc)), lg = split_logit(th), lgb = split_logit(split_tanh(split_dot(&sp, wb)));
    if (lgb > mb) { sb *= sb > 0 ? exp(mb - lgb) : 0.0f; mb = lgb; }
    sb += exp(lgb - mb);
    split_lse_add(&cur, lg, split_slope(th), &sp);
  }
  // the drawn split (lane 0's): its logits and psi
  const split tsp = split_of(&d, d.token);
  const float tth = split_tanh(split_dot(&tsp, wc));
  const float lgTok = tsp.ok ? split_logit(tth) : -INFINITY, lgbTok = tsp.ok ? split_logit(split_tanh(split_dot(&tsp, wb))) : -INFINITY;
  if (lane == 0) split_add(&tsp, split_slope(tth), psi);
  const bool owner = lane == 0;
  const float M = simd_max(cur.m), scale = cur.s > 0 ? exp(cur.m - M) : 0.0f, S = simd_sum(cur.s * scale);
  const float Mb = simd_max(mb), Sb = simd_sum(sb > 0 ? sb * exp(mb - Mb) : 0.0f);
  const float lt = simd_sum(owner ? lgTok : 0.0f), lbt = simd_sum(owner ? lgbTok : 0.0f);
  const float logRatio = (lt - (M + log(S))) - (lbt - (Mb + log(Sb))), ratio = exp(logRatio);
  for (uint f = 0; f < FS; f++) {
    const float ef = simd_sum(cur.e[f] * scale) / S, pf = simd_sum(psi[f]);
    if (f % 32 == lane) diff[size_t(i) * FS + f] = ratio * (pf - ef);
  }
  if (lane == 0) ll[i] = logRatio;
}
// dw [task][row][FS] = (task 0: scale0[row], else scale) sum over the row's draws (first[row] .. first[row + 1]) of its
// advantage (task 0: 1, 1: a1, 2: a2) times diff, in order
kernel void credit_rows(device const float *diff [[buffer(0)]], device const route_draw *draws [[buffer(1)]], device const uint *first [[buffer(2)]],
                        device const float *scale0 [[buffer(3)]], device float *dw [[buffer(4)]], constant uint &R [[buffer(5)]], constant float &scale [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= 3 * R * FS) return;
  const uint t = gid / (R * FS), row = gid / FS % R, f = gid % FS;
  float acc = 0;
  for (uint i = first[row]; i < first[row + 1]; i++) acc = fma(t == 0 ? 1.0f : t == 1 ? draws[i].a1 : draws[i].a2, diff[size_t(i) * FS + f], acc);
  dw[gid] = acc * (t == 0 ? scale0[row] : scale);
}
