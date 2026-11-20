// The learning step's credit and backward (tile.h's tiles): a task a threadgroup, each product's dgrad codes (row,
// stochastic) and the wgrad codes of every gradient it makes (column, stochastic; its column sums: the bias's
// gradient), the contexts' state gradients summed a refresh for the lanes' recurrence.
#include "tile.h"

// ---- the credit: a SIMD group a learning row, its draws in order; each draw's score function under the learning
// forward's w, psi(token) - E_p[psi] over its table (split.h, as the simulator's sampler and route_draw_score), times
// its importance ratio pi_current(token) / pi_behaviour(token) (wb: the w it was drawn under); the row's dw [3][R][FS]
// (task 0: scale0[row], tasks 1, 2: the draws' advantages a1, a2 and `scale`); ll [draw]: the log ratio ----
struct CP { uint R, pad; float scale, pad2; };
kernel void credit(device const route_draw *draws [[buffer(0)]], device const float *wb [[buffer(1)]], device const float *wc [[buffer(2)]],
                   device const uint *first [[buffer(3)]], device const float *scale0 [[buffer(4)]], device float *dw [[buffer(5)]], device float *ll [[buffer(6)]],
                   constant CP &p [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  const uint row = gid / 32, lane = gid % 32;
  if (row >= p.R) return;
  device const float *wbr = wb + size_t(row) * FS, *wcr = wc + size_t(row) * FS;
  float acc[3][2] = {{0, 0}, {0, 0}, {0, 0}};   // [task][features lane, lane + 32]
  for (uint i = first[row]; i < first[row + 1]; i++) {
    const route_draw d = draws[i];
    split_lse cur; split_lse_init(&cur);
    float mb = -INFINITY, sb = 0;
    const split_set a = split_set_of(&d);
    const uint chunk = (a.n + 31) / 32;   // a lane's contiguous chunk of the table, walked in registers
    for (split_walk k = split_walk_at(&a, lane * chunk, chunk); k.left; split_walk_next(&k)) {
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
    const float M = simd_max(cur.m), sc = cur.s > 0 ? exp(cur.m - M) : 0.0f, Ssum = simd_sum(cur.s * sc);
    const float Mb = simd_max(mb), Sb = simd_sum(sb > 0 ? sb * exp(mb - Mb) : 0.0f);
    const float logRatio = (lgTok - (M + log(Ssum))) - (lgbTok - (Mb + log(Sb))), ratio = exp(logRatio);
    const float adv[3] = {1.0f, d.a1, d.a2};
    for (uint f = 0; f < FS; f++) {
      const float diff = ratio * (psi[f] - simd_sum(cur.e[f] * sc) / Ssum);
      if (f % 32 == lane) for (uint tk = 0; tk < 3; tk++) acc[tk][f / 32] = fma(adv[tk], diff, acc[tk][f / 32]);
    }
    if (lane == 0) ll[i] = logRatio;
  }
  for (uint h = 0; h < 2; h++) {
    const uint f = lane + 32 * h;
    if (f < FS) for (uint tk = 0; tk < 3; tk++) dw[(size_t(tk) * p.R + row) * FS + f] = acc[tk][h] * (tk == 0 ? scale0[row] : p.scale);
  }
}

// a gradient's wgrad codes (the column pass over S's chunk j of a matrix's output, `rows` its outputs): task t's
// columns at t rows + 64 j of the matrix's dY codes, its scales and column sums at the tile's block
static inline void ycodes(threadgroup const float *S, Lanes c, device int8_t *gq, device float *gs, MatG g, uint rows, uint task, uint j, uint RT, uint row0, uint seed) {
  const uint M = 3 * rows, col0 = task * rows + 64 * j, blk = row0 / TM;
  col_codes(S, c, gq + g.yt + size_t(col0) * RT, RT, row0, gs + g.sy + blk * M + col0, gs + g.dbp + blk * M + col0, min(64u, rows - 64 * j), seed);
}
// a product's ids for the seeds (row codes: id, column codes: id + 64)
enum : uint { S_OUT = 1, S_DOWN1, S_UP1, S_ZH1, S_DOWN0, S_UP0, S_ZH0, S_INP, S_LZH1, S_LDOWN0, S_LUP0, S_LZH0, S_LINP };

// the step back through one context's state: h = hp + z (h~ - hp) with g = dL/dh: the gates' pre-activation gradients
// (z: g (h~ - hp) z (1 - z); h~: g z (1 - h~^2)) and hp's, g (1 - z)
static inline void step_back(thread const float *g, device const half *zc, device const float *hp, thread float *dz, thread float *dc, thread float *dh) {
  for (uint k = 0; k < 8; k++) {
    const float z = float(zc[k]), h = float(zc[D + k]);
    dz[k] = g[k] * (h - hp[k]) * z * (1.0f - z); dc[k] = g[k] * z * (1.0f - h * h); dh[k] = g[k] * (1.0f - z);
  }
}

// ---- the backward, contexts: a threadgroup a (tile of 64 contexts, task), from dw through every layer; each
// refresh's contexts' state gradients summed in the tile (rows of a refresh are contiguous): dhp [task][layer][Rp][D] at
// the row where the refresh's rows begin in the tile ----
kernel void ctx_bwd(device const float *dw [[buffer(0)]], device const int *pos [[buffer(1)]], device const float *st [[buffer(2)]], device const half *czc [[buffer(3)]],
                    device const uint *cu [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]],
                    constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                    device const float *cn [[buffer(12)]], device const float *cinv [[buffer(13)]], device float *dhp [[buffer(14)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  threadgroup int refid[TM];
  const Lanes c = lanes_of(t);
  const uint task = tg.y, r0 = tg.x * TM, g0 = p.RL + r0, RT = gr.RT;
  if (t < TM) refid[t] = r0 + t < p.R ? pos[r0 + t] / int(L) : -1;
  for (uint i = t; i < TM * 64; i += NT) {
    const uint r = i / 64, f = i % 64;
    S[r * LD + f] = f < FS && r0 + r < p.R ? dw[(size_t(task) * p.R + r0 + r) * FS + f] : 0.0f;
  }
  tg_barrier();
  ycodes(S, c, gq, gs, gr.m[M_OUT], FS, task, 0, RT, g0, seed_of(p.seed, S_OUT + 64, task));
  float x[2][1][8];
  for (uint s = 0; s < 2; s++) for (uint k = 0; k < 8; k++) x[s][0][k] = S[row_of(c, s) * LD + 8 * c.li + k];
  tg_barrier();
  put_codes<1>(x, A, sA, c, seed_of(p.seed, S_OUT, task), g0);
  tg_barrier();
  bool live[2]; int ref[2];
  for (uint s = 0; s < 2; s++) { const uint r = r0 + row_of(c, s); live[s] = r < p.R; ref[s] = live[s] ? refid[row_of(c, s)] : 0; }
  // the readout back: dn = dw out, then rms's backward dx2 = inv (dn - n mean(dn . n))
  { const MatW m = nw.m[M_OUT];
    chunk_bwd<64>(A, wq + m.wb, m.np, S); tg_barrier();
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s), r = r0 + row;
      float dn[8]; epilogue(S, c, row, sA[row], wsc + m.wbs, nullptr, 0, dn);
      float nn[8], q = 0;
      for (uint k = 0; k < 8; k++) { nn[k] = live[s] ? cn[size_t(r) * D + 8 * c.li + k] : 0.0f; q = fma(dn[k], nn[k], q); }
      for (ushort d = 1; d < 8; d <<= 1) q += simd_shuffle_xor(q, d);
      const float inv = live[s] ? cinv[r] : 0.0f, mn = q / float(D);
      for (uint k = 0; k < 8; k++) x[s][0][k] = inv * (dn[k] - nn[k] * mn);
    } }
  for (uint l = 2; l-- > 0;) {
    const uint zh = l ? M_ZH1 : M_ZH0, up = l ? M_UP1 : M_UP0, down = l ? M_DOWN1 : M_DOWN0;
    const uint sd = l ? S_DOWN1 : S_DOWN0, su = l ? S_UP1 : S_UP0, sz = l ? S_ZH1 : S_ZH0;
    // x: dL/dx_{l+1}: down's dY; its codes
    tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
    ycodes(S, c, gq, gs, gr.m[down], D, task, 0, RT, g0, seed_of(p.seed, sd + 64, task));
    tg_barrier(); put_codes<1>(x, A, sA, c, seed_of(p.seed, sd, task), g0); tg_barrier();
    // down's dgrad, relu's mask: du; up's dY
    float du[2][3][8];
    { const MatW md = nw.m[down];
      for (uint jj = 0; jj < 3; jj++) {
        chunk_bwd<64>(A, wq + md.wb + 64 * jj, md.np, S); tg_barrier();
        for (uint s = 0; s < 2; s++) {
          const uint row = row_of(c, s), r = r0 + row;
          epilogue(S, c, row, sA[row], wsc + md.wbs, nullptr, jj, du[s][jj]);
          device const uint *mw = cu + size_t(live[s] ? r : 0) * 12 + 6 * l;
          for (uint k = 0; k < 8; k++) if (!live[s] || !mask_bit(mw, 64 * jj + 8 * c.li + k)) du[s][jj][k] = 0;
        }
        tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), du[s][jj]); tg_barrier();
        ycodes(S, c, gq, gs, gr.m[up], E, task, jj, RT, g0, seed_of(p.seed, su + 64, task));
        tg_barrier();
      } }
    put_codes<3>(du, A, sA, c, seed_of(p.seed, su, task), g0); tg_barrier();
    // up's dgrad: dxm = dx + du up
    { const MatW mu = nw.m[up];
      chunk_bwd<192>(A, wq + mu.wb, mu.np, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { float y[8]; epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mu.wbs, nullptr, 0, y); for (uint k = 0; k < 8; k++) x[s][0][k] += y[k]; } }
    // the step back: the gates' gradients (zh's dY: z, then h~) and the state's, summed a refresh
    float dzc[2][2][8], dh[2][8];
    for (uint s = 0; s < 2; s++) {
      const uint r = r0 + row_of(c, s);
      device const half *zc = czc + (size_t(live[s] ? r : 0) * 2 + l) * 2 * D + 8 * c.li;
      device const float *hp = st + (size_t(ref[s]) * 2 + l) * D + 8 * c.li;
      step_back(x[s][0], zc, hp, dzc[s][0], dzc[s][1], dh[s]);
      if (!live[s]) for (uint k = 0; k < 8; k++) { dzc[s][0][k] = 0; dzc[s][1][k] = 0; dh[s][k] = 0; }
    }
    tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), dh[s]); tg_barrier();
    if (t < D) {   // a thread a channel down the tile's rows, a sum a refresh
      device float *out = dhp + ((size_t(task) * 2 + l) * p.Rp + r0) * D + t;
      float acc = 0; uint start = 0;
      for (uint r = 0; r < TM && refid[r] >= 0; r++) {
        acc += S[r * LD + t];
        if (r == TM - 1 || refid[r + 1] != refid[r]) { out[size_t(start) * D] = acc; acc = 0; start = r + 1; }
      }
    }
    for (uint j = 0; j < 2; j++) {
      tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), dzc[s][j]); tg_barrier();
      ycodes(S, c, gq, gs, gr.m[zh], 2 * D, task, j, RT, g0, seed_of(p.seed, sz + 64, task));
    }
    tg_barrier(); put_codes<2>(dzc, A, sA, c, seed_of(p.seed, sz, task), g0); tg_barrier();
    // zh's dgrad: dx_l = dxm + dzc zh
    { const MatW mz = nw.m[zh];
      chunk_bwd<128>(A, wq + mz.wb, mz.np, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { float y[8]; epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mz.wbs, nullptr, 0, y); for (uint k = 0; k < 8; k++) x[s][0][k] += y[k]; } }
  }
  // dx0: inp's dY
  tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
  ycodes(S, c, gq, gs, gr.m[M_INP], D, task, 0, RT, g0, seed_of(p.seed, S_INP + 64, task));
}

// the reverse recurrence over a tile's two groups of 32 positions (scan.h), a SIMD group 8 channels, a lane a position
// (latest first): for each channel the gradient arriving at each state, u (S's floats if `local`, plus the refresh's
// contexts' inj at a refresh's last lane), the carry e from the later tile (carry[ch], updated); each position's
// gates' gradients (z: m D (h~ - h_{t-1}) z (1 - z); h~: m D z (1 - h~^2), D = u + e_{t+1} the state's own gradient)
// written over S as bf16 pairs
static inline void scan_back(threadgroup float *S, threadgroup float *carry, threadgroup const float (&inj)[2][D], bool local, Lanes c,
                             device const float *mask, device const half *lzc, device const float *lh, device const float *h0, uint l, uint row0, constant FP &p) {
  for (uint cc = 0; cc < 8; cc++) {
    const uint ch = 8 * c.sg + cc;
    float ein = carry[ch];
    for (uint g = 2; g-- > 0;) {
      const uint r = 32 * g + 31 - c.lane, row = row0 + r; const LanePos lp = lane_pos(row, p), lg = lane_pos(row0 + 32 * g, p);
      if (lg.pos + 32 == p.KLp) ein = 0;   // a run's last group: nothing later
      const uint b = lp.b, pos = lp.pos; const bool valid = lp.valid;
      float u = local ? S[r * LD + ch] : 0.0f;
      if (r % L == L - 1 && valid) u += inj[g][ch];
      device const half *zc = lzc + (size_t(row) * 2 + l) * 2 * D;
      const float m = valid ? mask[size_t(b) * p.K * L + pos] : 0.0f, z = float(zc[ch]), h = float(zc[D + ch]);
      const float a = 1.0f - m * z;
      float Aa = a, Bb = a * u;
      scan_affine_simd(Aa, Bb, c.lane);
      const float e = fma(Aa, ein, Bb), later = simd_shuffle_up(e, ushort(1));
      const float Dh = u + (c.lane == 0 ? ein : later);
      const float hprev = pos > 0 ? lh[(size_t(row - 1) * 2 + l) * D + ch] : b < p.B ? h0[(b * 2 + l) * D + ch] : 0.0f;
      const float dz = m * Dh * (h - hprev) * z * (1.0f - z), dc = m * Dh * z * (1.0f - h * h);
      ((threadgroup uint *)S)[r * LD + ch] = uint(as_type<ushort>(bfloat(dz))) | (uint(as_type<ushort>(bfloat(dc))) << 16);
      ein = simd_broadcast(e, 31);
    }
    if (c.lane == 0) carry[ch] = ein;
  }
}
// a slot's bf16 pairs back as the gates' gradients
static inline void unpack_dzc(threadgroup const float *S, Lanes c, uint row, thread float *dz, thread float *dc) {
  for (uint k = 0; k < 8; k++) {
    const uint w = ((threadgroup const uint *)S)[row * LD + 8 * c.li + k];
    dz[k] = float(as_type<bfloat>(ushort(w & 0xffffu))); dc[k] = float(as_type<bfloat>(ushort(w >> 16)));
  }
}

// ---- the backward, lanes: a threadgroup a (run, task) (or for one refresh a (tile of two runs, task)), its tiles latest
// first (the reverse recurrence's carry):
// layer 1's states' gradients from the contexts (dhp, summed a refresh), its gates', then x1's (zh1's dgrad), layer 0's
// feed-forward, its recurrence (xm0's own gradient at every position, the contexts' at a refresh's end), x0's ----
kernel void lanes_bwd(device const float *mask [[buffer(0)]], device const float *h0 [[buffer(1)]], device const half *lzc [[buffer(2)]], device const float *lh [[buffer(3)]],
                      device const uint *lu [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]],
                      constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                      device const uint *seg [[buffer(12)]], device const float *dhp [[buffer(13)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  threadgroup float carry[2][D], inj[2][2][D];   // [layer][channel]; [layer][the tile's refresh][channel]
  const Lanes c = lanes_of(t);
  const uint task = tg.y, RT = gr.RT;
  if (t < 2 * D) carry[t / D][t % D] = 0;
  for (uint j = p.LT; j-- > 0;) {
    const uint row0 = (tg.x * p.LT + j) * TM;
    { // the tile's two refreshes' contexts' state gradients, a thread a (layer, refresh, channel)
      const uint l = t / 128, kk = t / 64 % 2, ch = t % 64; const LanePos lg = lane_pos(row0 + 32 * kk, p);
      float acc = 0;
      if (lg.valid) {
        const uint i = lg.b * p.K + lg.pos / L, x0 = seg[2 * i], x1 = seg[2 * i + 1];
        device const float *src = dhp + (size_t(task) * 2 + l) * p.Rp * D + ch;
        for (uint tt = x0 / TM; x1 > x0 && tt <= (x1 - 1) / TM; tt++) acc += src[size_t(tt == x0 / TM ? x0 : tt * TM) * D];
      }
      inj[l][kk][ch] = acc;
    }
    tg_barrier();
    // layer 1: the recurrence back from the contexts' gradients alone
    scan_back(S, carry[1], inj[1], false, c, mask, lzc, lh, h0, 1, row0, p);
    tg_barrier();
    float dzc[2][2][8];
    for (uint s = 0; s < 2; s++) unpack_dzc(S, c, row_of(c, s), dzc[s][0], dzc[s][1]);
    for (uint jj = 0; jj < 2; jj++) {
      tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), dzc[s][jj]); tg_barrier();
      ycodes(S, c, gq, gs, gr.m[M_ZH1], 2 * D, task, jj, RT, row0, seed_of(p.seed, S_LZH1 + 64, task));
    }
    tg_barrier(); put_codes<2>(dzc, A, sA, c, seed_of(p.seed, S_LZH1, task), row0); tg_barrier();
    float x[2][1][8];
    { const MatW mz = nw.m[M_ZH1];
      chunk_bwd<128>(A, wq + mz.wb, mz.np, S); tg_barrier();
      for (uint s = 0; s < 2; s++) epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mz.wbs, nullptr, 0, x[s][0]); }
    // x1's gradient: down0's dY; layer 0's feed-forward back
    tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
    ycodes(S, c, gq, gs, gr.m[M_DOWN0], D, task, 0, RT, row0, seed_of(p.seed, S_LDOWN0 + 64, task));
    tg_barrier(); put_codes<1>(x, A, sA, c, seed_of(p.seed, S_LDOWN0, task), row0); tg_barrier();
    float du[2][3][8];
    { const MatW md = nw.m[M_DOWN0];
      for (uint jj = 0; jj < 3; jj++) {
        chunk_bwd<64>(A, wq + md.wb + 64 * jj, md.np, S); tg_barrier();
        for (uint s = 0; s < 2; s++) {
          const uint row = row_of(c, s);
          epilogue(S, c, row, sA[row], wsc + md.wbs, nullptr, jj, du[s][jj]);
          device const uint *mw = lu + size_t(row0 + row) * 6;
          for (uint k = 0; k < 8; k++) if (!mask_bit(mw, 64 * jj + 8 * c.li + k)) du[s][jj][k] = 0;
        }
        tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), du[s][jj]); tg_barrier();
        ycodes(S, c, gq, gs, gr.m[M_UP0], E, task, jj, RT, row0, seed_of(p.seed, S_LUP0 + 64, task));
        tg_barrier();
      } }
    put_codes<3>(du, A, sA, c, seed_of(p.seed, S_LUP0, task), row0); tg_barrier();
    { const MatW mu = nw.m[M_UP0];
      chunk_bwd<192>(A, wq + mu.wb, mu.np, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { float y[8]; epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mu.wbs, nullptr, 0, y); for (uint k = 0; k < 8; k++) x[s][0][k] += y[k]; } }
    // layer 0: the recurrence back from xm0's gradient (every position) and the contexts'
    tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
    scan_back(S, carry[0], inj[0], true, c, mask, lzc, lh, h0, 0, row0, p);
    tg_barrier();
    for (uint s = 0; s < 2; s++) unpack_dzc(S, c, row_of(c, s), dzc[s][0], dzc[s][1]);
    for (uint jj = 0; jj < 2; jj++) {
      tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), dzc[s][jj]); tg_barrier();
      ycodes(S, c, gq, gs, gr.m[M_ZH0], 2 * D, task, jj, RT, row0, seed_of(p.seed, S_LZH0 + 64, task));
    }
    tg_barrier(); put_codes<2>(dzc, A, sA, c, seed_of(p.seed, S_LZH0, task), row0); tg_barrier();
    { const MatW mz = nw.m[M_ZH0];
      chunk_bwd<128>(A, wq + mz.wb, mz.np, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { float y[8]; epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mz.wbs, nullptr, 0, y); for (uint k = 0; k < 8; k++) x[s][0][k] += y[k]; } }
    tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
    ycodes(S, c, gq, gs, gr.m[M_INP], D, task, 0, RT, row0, seed_of(p.seed, S_LINP + 64, task));
    tg_barrier();
  }
}
