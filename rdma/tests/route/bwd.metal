// The learning step's backward (tile.h's tiles): a task a threadgroup, each product's dgrad codes (row, stochastic) and
// the wgrad codes of every gradient it makes (column, stochastic; its column sums: the bias's gradient), the contexts'
// state gradients summed a refresh (a product with the tile's segment matrix) for the lanes' recurrence.
#include "tile.h"

// a gradient's wgrad codes (the column pass over S's chunk j of a matrix's output, `rows` its outputs): task t's
// columns at t rows + 64 j of the matrix's dY codes, its scales and column sums at the tile's blocks
static inline void ycodes(threadgroup float *S, threadgroup half *X, threadgroup float *cm, device const half *H, Lanes c, device int8_t *gq, device float *gs, MatG g,
                          uint rows, uint task, uint j, uint RT, uint row0, uint seed) {
  const uint M = 3 * rows, col0 = task * rows + 64 * j, blk = row0 / I8_WBLOCK;
  colcodes(S, X, cm, H, c, gq + g.yt + size_t(col0) * RT, RT, row0, gs + g.sy + blk * M + col0, M, gs + g.dbp + blk * M + col0, min(64u, rows - 64 * j), 8.0f, seed);
}
// a product's ids for the seeds (row codes: id, column codes: id + 64)
enum : uint { S_OUT = 1, S_DOWN1, S_UP1, S_ZH1, S_DOWN0, S_UP0, S_ZH0, S_INP, S_LZH1, S_LDOWN0, S_LUP0, S_LZH0, S_LINP };

// a value's codes (into B) through a dgrad block: x = (or +=) codes . W's K block jj (wb's columns 64 jj); index
// (row0 + row) cols + 64 jj + column
static inline void dgrad_block(thread float (&x)[2][8], thread const float (&v)[2][8], MatW m, uint jj, bool first, threadgroup float *S, threadgroup half *X,
                               threadgroup int8_t *Bq, threadgroup float *sB, device const half *H, device const int8_t *wq, device const float *wsc, Lanes c,
                               uint seed, uint row0, uint cols) {
  quant(v, S, X, H, c, Bq, sB, seed, row0, cols, 64 * jj);
  const ColV cv1 = colv(wsc + m.wbs, nullptr, 0, c); chunk(Bq, 64, wq + m.wb + 64 * jj, m.kb, S); tg_barrier();
  for (uint s = 0; s < 2; s++) { const uint row = row_of(c, s); epilogue(S, c, row, sB[row], cv1, x[s], !first); }
}
// the wgrad codes of v (put into S first)
static inline void vcodes(thread const float (&v)[2][8], threadgroup float *S, threadgroup half *X, threadgroup float *cm, device const half *H, Lanes c,
                          device int8_t *gq, device float *gs, MatG g, uint rows, uint task, uint j, uint RT, uint row0, uint seed) {
  for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), v[s]);
  tg_barrier();
  ycodes(S, X, cm, H, c, gq, gs, g, rows, task, j, RT, row0, seed);
}
// the feed-forward back: x = dL/dx' (its codes in A with sA) -> x + (relu' (x down)) up; up's dY codes, mask bits from
// mw (row stride mstride; rows < nv)
static inline void ffn_back(thread float (&x)[2][8], threadgroup float *S, threadgroup int8_t *A, threadgroup half *X, threadgroup int8_t *Bq, threadgroup float *sA,
                            threadgroup float *sB, threadgroup float *cm, MatW mu, MatW md, device const int8_t *wq, device const float *wsc, device const half *H, Lanes c,
                            device const uint *mw, uint mstride, uint nv, device int8_t *gq, device float *gs, MatG gu, uint task, uint RT, uint row0, uint seedc, uint seedr) {
  for (uint jj = 0; jj < 3; jj++) {
    if (jj) tg_barrier();   // the last dgrad block's epilogue read S
    const ColV cv2 = colv(wsc + md.wbs, nullptr, jj, c); chunk(A, 64, wq + md.wb + 64 * jj * md.kb, md.kb, S); tg_barrier();
    float du[2][8];
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s);
      epilogue(S, c, row, sA[row], cv2, du[s], false);
      device const uint *m = mw + (row < nv ? row : 0) * mstride;
      for (uint k = 0; k < 8; k++) if (row >= nv || !mask_bit(m, 64 * jj + 8 * c.li + k)) du[s][k] = 0;
      put_s(S, c, row, du[s]);
    }
    tg_barrier();
    ycodes(S, X, cm, H, c, gq, gs, gu, E, task, jj, RT, row0, seedc);
    dgrad_block(x, du, mu, jj, false, S, X, Bq, sB, H, wq, wsc, c, seedr, row0, E);
  }
}

// ---- the backward, contexts: a threadgroup a (tile of 64 contexts, task), from dw through every layer; each
// refresh's contexts' state gradients summed in the tile (rows of a refresh are contiguous): dhp [task][layer][Rp][D] at
// the row where the refresh's rows begin in the tile ----
kernel void ctx_bwd(device const float *dw [[buffer(0)]], device const int *pos [[buffer(1)]], device const float *st [[buffer(2)]], device const half *czc [[buffer(3)]],
                    device const uint *cu [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]],
                    constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                    device const float *cn [[buffer(12)]], device const float *cinv [[buffer(13)]], device float *dhp [[buffer(14)]], device const half *H [[buffer(16)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  threadgroup int refid[TM];
  const Lanes c = lanes_of(t);
  const uint task = tg.y, r0 = tg.x * TM, g0 = p.RL + r0, RT = gr.RT;
  const uint nv = min(TM, p.R - r0);
  if (t < TM) refid[t] = r0 + t < p.R ? pos[r0 + t] / int(L) : -1;
  for (uint i = t; i < TM * 16; i += NT) {
    const uint r = i / 16, f = 4 * (i % 16);
    *(threadgroup float4 *)(S + r * LD + f) = f < FS && r < nv ? *(device const float4 *)(dw + (size_t(task) * p.R + r0 + r) * FS + f) : float4(0);
  }
  tg_barrier();
  float x[2][8];
  for (uint s = 0; s < 2; s++) s_get(S, row_of(c, s), c, x[s]);
  ycodes(S, X, cm, H, c, gq, gs, gr.m[M_OUT], FS, task, 0, RT, g0, seed_of(p.seed, S_OUT + 64, task));
  quant(x, S, X, H, c, A, sA, seed_of(p.seed, S_OUT, task), g0, 64, 0);
  bool live[2]; int ref[2];
  for (uint s = 0; s < 2; s++) { const uint row = row_of(c, s); live[s] = row < nv; ref[s] = live[s] ? refid[row] : 0; }
  // the readout back: dn = dw out, then rms's backward dx2 = inv (dn - n mean(dn . n))
  { const MatW m = nw.m[M_OUT];
    const ColV cv3 = colv(wsc + m.wbs, nullptr, 0, c); chunk(A, 64, wq + m.wb, m.kb, S); tg_barrier();
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s), r = r0 + row;
      float dn[8]; epilogue(S, c, row, sA[row], cv3, dn, false);
      float nn[8], q = 0;
      if (live[s]) { device const float4 *n4 = (device const float4 *)(cn + size_t(r) * D + 8 * c.li); const float4 a = n4[0], b = n4[1]; for (uint k = 0; k < 4; k++) { nn[k] = a[k]; nn[4 + k] = b[k]; } }
      else for (uint k = 0; k < 8; k++) nn[k] = 0;
      for (uint k = 0; k < 8; k++) q = fma(dn[k], nn[k], q);
      for (ushort d = 1; d < 8; d <<= 1) q += simd_shuffle_xor(q, d);
      const float inv = live[s] ? cinv[r] : 0.0f, mn = q / float(D);
      for (uint k = 0; k < 8; k++) x[s][k] = inv * (dn[k] - nn[k] * mn);
    } }
  for (uint l = 2; l-- > 0;) {
    const uint zh = l ? M_ZH1 : M_ZH0, up = l ? M_UP1 : M_UP0, down = l ? M_DOWN1 : M_DOWN0;
    const uint sd = l ? S_DOWN1 : S_DOWN0, su = l ? S_UP1 : S_UP0, sz = l ? S_ZH1 : S_ZH0;
    // x: dL/dx_{l+1}: down's dY; its codes; the feed-forward back: dxm
    vcodes(x, S, X, cm, H, c, gq, gs, gr.m[down], D, task, 0, RT, g0, seed_of(p.seed, sd + 64, task));
    quant(x, S, X, H, c, A, sA, seed_of(p.seed, sd, task), g0, 64, 0);
    ffn_back(x, S, A, X, Bq, sA, sB, cm, nw.m[up], nw.m[down], wq, wsc, H, c, cu + size_t(r0) * 12 + 6 * l, 12, nv, gq, gs, gr.m[up], task, RT, g0,
             seed_of(p.seed, su + 64, task), seed_of(p.seed, su, task));
    // the step back through the context's state (h = hp + z (h~ - hp)): the gates' gradients and hp's
    float dz[2][8], dc[2][8], dh[2][8];
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s), r = r0 + row;
      float z[8], h[8], hp[8];
      if (live[s]) {
        device const half4 *zc = (device const half4 *)(czc + (size_t(r) * 2 + l) * 2 * D + 8 * c.li);
        const half4 z0 = zc[0], z1 = zc[1], h0 = zc[D / 4], h1 = zc[D / 4 + 1];
        device const float4 *p4 = (device const float4 *)(st + (size_t(ref[s]) * 2 + l) * D + 8 * c.li); const float4 a = p4[0], b = p4[1];
        for (uint k = 0; k < 4; k++) { z[k] = z0[k]; z[4 + k] = z1[k]; h[k] = h0[k]; h[4 + k] = h1[k]; hp[k] = a[k]; hp[4 + k] = b[k]; }
      } else for (uint k = 0; k < 8; k++) { z[k] = 0; h[k] = 0; hp[k] = 0; }
      for (uint k = 0; k < 8; k++) {
        const float g = live[s] ? x[s][k] : 0.0f;
        dz[s][k] = g * (h[k] - hp[k]) * z[k] * (1.0f - z[k]); dc[s][k] = g * z[k] * (1.0f - h[k] * h[k]); dh[s][k] = g * (1.0f - z[k]);
      }
    }
    // dh summed a refresh: Sg [64 starts][64 rows] (1 where row j is in the segment starting at row i) . dh, fp16
    { threadgroup half *Sg = (threadgroup half *)S;
      tg_barrier();   // the dgrad block's epilogue read S
      for (uint s = 0; s < 2; s++) { threadgroup half4 *o = (threadgroup half4 *)(X + row_of(c, s) * LX + 8 * c.li);
        o[0] = half4(dh[s][0], dh[s][1], dh[s][2], dh[s][3]); o[1] = half4(dh[s][4], dh[s][5], dh[s][6], dh[s][7]); }
      { const uint i = t / 4, j0 = 16 * (t % 4); const int ri = refid[i]; const bool start = ri >= 0 && (i == 0 || refid[i - 1] != ri);
        for (uint j = j0; j < j0 + 16; j++) Sg[i * LX + j] = half(start && j >= i && refid[j] == ri ? 1.0f : 0.0f); }
      tg_barrier();
      TgH tS(Sg, dextents<int32_t, 2>(TM, TM), array<int32_t, 2>{1, LX}), tD(X, dextents<int32_t, 2>(64, TM), array<int32_t, 2>{1, LX});
      matmul2d<matmul2d_descriptor(TM, 64, TM, false, false), execution_simdgroups<NSG>> ops;
      auto sum = ops.template get_destination_cooperative_tensor<TgH, TgH, float>();
      ops.run(tS, tD, sum);
      device float *out = dhp + ((size_t(task) * 2 + l) * p.Rp + r0) * D;
      for (ushort i = 0; i < sum.get_capacity(); i += 4) {
        const auto at = sum.get_multidimensional_index(i); const uint row = at[1], col = at[0];
        if (refid[row] >= 0 && (row == 0 || refid[row - 1] != refid[row])) *(device float4 *)(out + size_t(row) * D + col) = float4(sum[i], sum[i + 1], sum[i + 2], sum[i + 3]);
      }
      tg_barrier(); }
    // the gates' gradients: zh's dY (z, then h~), its dgrad: dx_l = dxm + dzc zh
    vcodes(dz, S, X, cm, H, c, gq, gs, gr.m[zh], 2 * D, task, 0, RT, g0, seed_of(p.seed, sz + 64, task));
    vcodes(dc, S, X, cm, H, c, gq, gs, gr.m[zh], 2 * D, task, 1, RT, g0, seed_of(p.seed, sz + 64, task));
    dgrad_block(x, dz, nw.m[zh], 0, false, S, X, Bq, sB, H, wq, wsc, c, seed_of(p.seed, sz, task), g0, 128);
    dgrad_block(x, dc, nw.m[zh], 1, false, S, X, Bq, sB, H, wq, wsc, c, seed_of(p.seed, sz, task), g0, 128);
  }
  // dx0: inp's dY
  vcodes(x, S, X, cm, H, c, gq, gs, gr.m[M_INP], D, task, 0, RT, g0, seed_of(p.seed, S_INP + 64, task));
}

// the reverse recurrence over a tile's two groups of 32 positions (scan.h), a SIMD group 8 channels, a lane a position
// (latest first): for each channel the gradient arriving at each state, u (S's floats if `local`, plus the refresh's
// contexts' inj at a refresh's last lane), the carry e from the later tile (carry[ch], updated); each state's own
// gradient D = u + e_{t+1} written over S
static inline void scan_back(threadgroup float *S, threadgroup float *carry, threadgroup const float (&inj)[2][D], bool local, Lanes c,
                             device const float *mask, device const half *lzc, uint l, uint row0, constant FP &p) {
  for (uint cc = 0; cc < 8; cc++) {
    const uint ch = 8 * c.sg + cc;
    float ein = carry[ch];
    for (uint g = 2; g-- > 0;) {
      const uint r = 32 * g + 31 - c.lane, row = row0 + r; const LanePos lp = lane_pos(row, p), lg = lane_pos(row0 + 32 * g, p);
      if (lg.pos + 32 == p.KLp) ein = 0;   // a run's last group: nothing later
      float u = local ? S[r * LD + ch] : 0.0f;
      if (r % L == L - 1 && lp.valid) u += inj[g][ch];
      const float m = lp.valid ? mask[size_t(lp.b) * p.K * L + lp.pos] : 0.0f, z = float(lzc[(size_t(row) * 2 + l) * 2 * D + ch]);
      const float a = 1.0f - m * z;
      float Aa = a, Bb = a * u;
      scan_affine_lanes8(Aa, Bb, c.lane);
      const float e0 = ein;
      float e = 0;
      for (uint q = 0; q < 4; q++) { const float o = fma(Aa, ein, Bb); if (c.lane / 8 == q) e = o; ein = simd_shuffle(o, ushort(8 * q + 7)); }
      const float later = simd_shuffle_up(e, ushort(1));
      S[r * LD + ch] = u + (c.lane == 0 ? e0 : later);
    }
    if (c.lane == 0) carry[ch] = ein;
  }
}
// the gates' gradients from the states' (D in S): z: m D (h~ - h_{t-1}) z (1 - z); h~: m D z (1 - h~^2)
static inline void gates_back(threadgroup const float *S, Lanes c, device const float *mask, device const half *lzc, device const float *lh, device const float *h0,
                              uint l, uint row0, constant FP &p, thread float (&dz)[2][8], thread float (&dc)[2][8]) {
  for (uint s = 0; s < 2; s++) {
    const uint row = row_of(c, s); const LanePos lp = lane_pos(row0 + row, p);
    float Dh[8]; s_get(S, row, c, Dh);
    if (!lp.valid) { for (uint k = 0; k < 8; k++) { dz[s][k] = 0; dc[s][k] = 0; } continue; }
    const float m = mask[size_t(lp.b) * p.K * L + lp.pos];
    device const half4 *zc = (device const half4 *)(lzc + (size_t(row0 + row) * 2 + l) * 2 * D + 8 * c.li);
    const half4 z0 = zc[0], z1 = zc[1], h0v = zc[D / 4], h1v = zc[D / 4 + 1];
    device const float4 *hp4 = (device const float4 *)(lp.pos > 0 ? lh + (size_t(row0 + row - 1) * 2 + l) * D + 8 * c.li : h0 + (lp.b * 2 + l) * D + 8 * c.li);
    const float4 pa = hp4[0], pb = hp4[1];
    for (uint k = 0; k < 8; k++) {
      const float z = k < 4 ? float(z0[k]) : float(z1[k - 4]), h = k < 4 ? float(h0v[k]) : float(h1v[k - 4]), hprev = k < 4 ? pa[k] : pb[k - 4];
      dz[s][k] = m * Dh[k] * (h - hprev) * z * (1.0f - z); dc[s][k] = m * Dh[k] * z * (1.0f - h * h);
    }
  }
}

// ---- the backward, lanes: a threadgroup a (run, task) (or for one refresh a (tile of two runs, task)), its tiles latest
// first (the reverse recurrence's carry): layer 1's states' gradients from the contexts (dhp, summed a refresh), its
// gates', x1's (zh1's dgrad), layer 0's feed-forward, its recurrence (xm0's own gradient at every position, the
// contexts' at a refresh's end), x0's ----
kernel void lanes_bwd(device const float *mask [[buffer(0)]], device const float *h0 [[buffer(1)]], device const half *lzc [[buffer(2)]], device const float *lh [[buffer(3)]],
                      device const uint *lu [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]],
                      constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                      device const uint *seg [[buffer(12)]], device const float *dhp [[buffer(13)]], device const half *H [[buffer(16)]],
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
    scan_back(S, carry[1], inj[1], false, c, mask, lzc, 1, row0, p);
    tg_barrier();
    float dz[2][8], dc[2][8], x[2][8];
    gates_back(S, c, mask, lzc, lh, h0, 1, row0, p, dz, dc);
    vcodes(dz, S, X, cm, H, c, gq, gs, gr.m[M_ZH1], 2 * D, task, 0, RT, row0, seed_of(p.seed, S_LZH1 + 64, task));
    vcodes(dc, S, X, cm, H, c, gq, gs, gr.m[M_ZH1], 2 * D, task, 1, RT, row0, seed_of(p.seed, S_LZH1 + 64, task));
    dgrad_block(x, dz, nw.m[M_ZH1], 0, true, S, X, Bq, sB, H, wq, wsc, c, seed_of(p.seed, S_LZH1, task), row0, 128);
    dgrad_block(x, dc, nw.m[M_ZH1], 1, false, S, X, Bq, sB, H, wq, wsc, c, seed_of(p.seed, S_LZH1, task), row0, 128);
    // x1's gradient: down0's dY; layer 0's feed-forward back
    vcodes(x, S, X, cm, H, c, gq, gs, gr.m[M_DOWN0], D, task, 0, RT, row0, seed_of(p.seed, S_LDOWN0 + 64, task));
    quant(x, S, X, H, c, A, sA, seed_of(p.seed, S_LDOWN0, task), row0, 64, 0);
    ffn_back(x, S, A, X, Bq, sA, sB, cm, nw.m[M_UP0], nw.m[M_DOWN0], wq, wsc, H, c, lu + size_t(row0) * 6, 6, TM, gq, gs, gr.m[M_UP0], task, RT, row0,
             seed_of(p.seed, S_LUP0 + 64, task), seed_of(p.seed, S_LUP0, task));
    // layer 0: the recurrence back from xm0's gradient (every position) and the contexts'
    for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s]);
    tg_barrier();
    scan_back(S, carry[0], inj[0], true, c, mask, lzc, 0, row0, p);
    tg_barrier();
    gates_back(S, c, mask, lzc, lh, h0, 0, row0, p, dz, dc);
    vcodes(dz, S, X, cm, H, c, gq, gs, gr.m[M_ZH0], 2 * D, task, 0, RT, row0, seed_of(p.seed, S_LZH0 + 64, task));
    vcodes(dc, S, X, cm, H, c, gq, gs, gr.m[M_ZH0], 2 * D, task, 1, RT, row0, seed_of(p.seed, S_LZH0 + 64, task));
    dgrad_block(x, dz, nw.m[M_ZH0], 0, false, S, X, Bq, sB, H, wq, wsc, c, seed_of(p.seed, S_LZH0, task), row0, 128);
    dgrad_block(x, dc, nw.m[M_ZH0], 1, false, S, X, Bq, sB, H, wq, wsc, c, seed_of(p.seed, S_LZH0, task), row0, 128);
    vcodes(x, S, X, cm, H, c, gq, gs, gr.m[M_INP], D, task, 0, RT, row0, seed_of(p.seed, S_LINP + 64, task));
  }
}
