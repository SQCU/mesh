// The learning step's credit (split.h's numerics, as the simulator's sampler and the CPU's route_draw_score), compiled
// with fast math like the earlier backend's: no product, so none of act.h's bit-for-bit contract; its transcendentals
// (exp, log, tanh) precise were 2.4x its time (roofline). The check: the GPU's credit against split.h's on the CPU.
#include <metal_stdlib>
using namespace metal;
#include "split.h"
enum : uint { FS = 40 };

// ---- the credit, a draw eight lanes (four a SIMD group; a row's draws vary from one to hundreds: rows a group
// serialized them, a learning step's credit 20 ms against 1 a draw at the training's sizes, bench; and a table is
// typically a few splits: 32 lanes a draw left most idle and reduced its 40 features by 32-lane sums, 3.3 ms in the
// roofline's step): each draw's score function under the learning forward's w (its row's: d.row),
// psi(token) - E_p[psi] over its table (split.h, as the simulator's sampler and route_draw_score), its table's splits
// across the eight lanes in contiguous chunks, the reductions over the eight (xor 1, 2, 4), times its importance ratio
// pi_current(token) / pi_behaviour(token) (wb: the w it was drawn under) -> diff [draw][FS]; ll [draw]: the log ratio; lp
// [draw]: log pi_current(token) ----
static inline float sum8(float v) { v += simd_shuffle_xor(v, ushort(1)); v += simd_shuffle_xor(v, ushort(2)); return v + simd_shuffle_xor(v, ushort(4)); }
static inline float max8(float v) { v = max(v, simd_shuffle_xor(v, ushort(1))); v = max(v, simd_shuffle_xor(v, ushort(2))); return max(v, simd_shuffle_xor(v, ushort(4))); }
kernel void credit_draws(device const route_draw *draws [[buffer(0)]], device const float *wb [[buffer(1)]], device const float *wc [[buffer(2)]],
                         device float *diff [[buffer(3)]], device float *ll [[buffer(4)]], constant uint &n [[buffer(5)]], device float *lp [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint i = gid / 8, gl = gid % 8;
  const bool valid = i < n;   // a group past the end still joins its SIMD group's shuffles
  route_draw d;
  if (valid) d = draws[i]; else { d = draws[0]; d.row = 0; }
  device const float *wbr = wb + size_t(d.row) * FS, *wcr = wc + size_t(d.row) * FS;
  split_lse cur; split_lse_init(&cur);
  float mb = -INFINITY, sb = 0;
  const split_set a = split_set_of(&d);
  const uint chunk = (a.n + 7) / 8;
  for (split_walk k = split_walk_at(&a, gl * chunk, valid ? chunk : 0); k.left; split_walk_next(&k)) {
    const split sp = split_of(&d, k.token);
    const float th = split_tanh(split_dot(&sp, wcr)), lg = split_logit(th), lgb = split_logit(split_tanh(split_dot(&sp, wbr)));
    if (lgb > mb) { sb *= sb > 0 ? exp(mb - lgb) : 0.0f; mb = lgb; }
    sb += exp(lgb - mb);
    split_lse_add(&cur, lg, split_slope(th), &sp);
  }
  const split tsp = split_of(&d, d.token);   // the drawn split: its logits and psi
  const float tth = split_tanh(split_dot(&tsp, wcr));
  const float lgTok = tsp.ok ? split_logit(tth) : -INFINITY, lgbTok = tsp.ok ? split_logit(split_tanh(split_dot(&tsp, wbr))) : -INFINITY;
  float psi[SPLIT_FS];
  for (uint f = 0; f < FS; f++) psi[f] = 0;
  split_add(&tsp, split_slope(tth), psi);
  const float M = max8(cur.m), sc = cur.s > 0 ? exp(cur.m - M) : 0.0f, Ssum = sum8(cur.s * sc);
  const float Mb = max8(mb), Sb = sum8(sb > 0 ? sb * exp(mb - Mb) : 0.0f);
  const float logRatio = (lgTok - (M + log(Ssum))) - (lgbTok - (Mb + log(Sb))), ratio = exp(logRatio);
  for (uint f = 0; f < FS; f++) {
    const float e = sum8(cur.e[f] * sc) / Ssum;
    if (valid && f % 8 == gl) diff[size_t(i) * FS + f] = ratio * (psi[f] - e);
  }
  if (valid && gl == 0) { ll[i] = logRatio; lp[i] = lgTok - (M + log(Ssum)); }
}
// a thread a (task, row, feature): dw [task][row][f] = the row's draws' (first[row] .. first[row + 1]) diff, in order,
// each times its advantage (task 0: 1, 1: a1, 2: a2), times the task's scale (task 0: scale0[row], else `scale`)
struct CP { uint R, pad; float scale, pad2; };
kernel void credit_rows(device const float *diff [[buffer(0)]], device const route_draw *draws [[buffer(1)]], device const uint *first [[buffer(2)]],
                        device const float *scale0 [[buffer(3)]], device float *dw [[buffer(4)]], constant CP &p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= 3 * p.R * FS) return;
  const uint t = gid / (p.R * FS), row = gid / FS % p.R, f = gid % FS;
  float acc = 0;
  for (uint i = first[row]; i < first[row + 1]; i++) acc = fma(t == 0 ? 1.0f : t == 1 ? draws[i].a1 : draws[i].a2, diff[size_t(i) * FS + f], acc);
  dw[gid] = acc * (t == 0 ? scale0[row] : p.scale);
}
