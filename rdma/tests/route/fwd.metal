// The forward's kernels (tile.h's tiles): a run's lanes, a tile of contexts.
#include "tile.h"
constant bool STORE [[function_constant(0)]];   // the learning forward: what the backward reads, and the wgrad codes
constant uint STAGE [[function_constant(1)]];   // stages.swift's build-up (0: everything)

// the wgrad X codes of the value in S (64 columns of the tile, at the matrix's column offset jj 64; kf its columns)
static inline void xcodes(threadgroup float *S, threadgroup half *X, threadgroup float *cm, device const half *Hb, Lanes c, device int8_t *gq, device float *gs, MatG g,
                          uint kf, uint jj, uint RT, uint row0) {
  colcodes(S, X, cm, Hb, c, gq + g.xt + size_t(jj) * 64 * RT, RT, row0, gs + g.sx + (row0 / I8_WBLOCK) * kf + 64 * jj, kf, nullptr, 64, 4.0f, 0);
}
// the feed-forward: x (xm, its codes in A with sA) -> xm + down(relu(up(xm))), down's blocks as each up chunk is made
// (its codes in B); ST: the masks (mw [row] of 6 words at row stride, rows < nv), down's X codes
template <bool ST>
static inline void ffn(thread float (&x)[2][8], threadgroup float *S, threadgroup int8_t *A, threadgroup half *X, threadgroup int8_t *Bq, threadgroup float *sA, threadgroup float *sB,
                       threadgroup float *cm, MatW mu, MatW md, device const int8_t *wq, device const float *wsc, device const half *H, Lanes c,
                       device uint *mw, uint mstride, uint nv, device int8_t *gq, device float *gs, MatG g, uint RT, uint row0) {
  for (uint s = 0; s < 2; s++) { const float4 b0 = *(device const float4 *)(wsc + md.b + 8 * c.li), b1 = *(device const float4 *)(wsc + md.b + 8 * c.li + 4);
    for (uint k = 0; k < 4; k++) { x[s][k] += b0[k]; x[s][4 + k] += b1[k]; } }   // the residual plus the bias, then the blocks
  for (uint jj = 0; jj < 3; jj++) {
    chunk(A, 64, wq + mu.wf + jj * 64 * mu.kf, mu.kf, S); tg_barrier();
    float u[2][8];
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s);
      epilogue(S, c, row, sA[row], wsc + mu.wfs, wsc + mu.b, jj, u[s], false);
      for (uint k = 0; k < 8; k++) u[s][k] = max(u[s][k], 0.0f);
      if (ST) { const uint wd = mask_word(u[s], c); if (c.li % 4 == 0 && row < nv) mw[row * mstride + 2 * jj + c.li / 4] = wd; put_s(S, c, row, u[s]); }
    }
    if (ST) { tg_barrier(); xcodes(S, X, cm, H, c, gq, gs, g, 192, jj, RT, row0); }
    quant(u, S, X, H, c, Bq, sB, 0, 0, 0, 0);
    chunk(Bq, 64, wq + md.wf + 64 * jj, md.kf, S); tg_barrier();
    for (uint s = 0; s < 2; s++) { const uint row = row_of(c, s); epilogue(S, c, row, sB[row], wsc + md.wfs, nullptr, 0, x[s], true); }
    tg_barrier();
  }
}

// ---- the forward, lanes: a threadgroup a run, its tiles in order (the recurrence's state carried), or for one refresh
// a tile of two runs ----
// lanes [B K][L][IN], mask [B K][L], h0 [B][2][D] -> states st [B K][2][D] (each refresh's last), next [B][2][D]
kernel void lanes_fwd(device const float *lanes [[buffer(0)]], device const float *mask [[buffer(1)]], device const float *h0 [[buffer(2)]],
                      device float *st [[buffer(3)]], device float *next [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]],
                      device const float *wsc [[buffer(7)]], constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]],
                      constant Grad &gr [[buffer(11)]], device half *lzc [[buffer(12)]], device float *lh [[buffer(13)]], device uint *lu [[buffer(14)]],
                      device const half *H [[buffer(16)]],
                      uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  threadgroup float carry[2][D];
  const Lanes c = lanes_of(t);
  const uint RT = gr.RT;
  for (uint j = 0; j < p.LT; j++) {
    const uint row0 = (tg * p.LT + j) * TM;   // the tile's first lane row
    // the tokens (IN of 64 columns, four a load, the rest 0)
    for (uint i = t; i < TM * IN / 4; i += NT) {
      const uint r = i / (IN / 4), q = i % (IN / 4); const LanePos lp = lane_pos(row0 + r, p);
      *(threadgroup float4 *)(S + r * LD + 4 * q) = lp.valid ? ((device const float4 *)(lanes + (size_t(lp.b) * p.K * L + lp.pos) * IN))[q] : float4(0);
    }
    *(threadgroup float4 *)(S + (t / 4) * LD + IN + 4 * (t % 4)) = float4(0);
    tg_barrier();
    float x[2][8];
    for (uint s = 0; s < 2; s++) s_get(S, row_of(c, s), c, x[s]);
    if (STORE) xcodes(S, X, cm, H, c, gq, gs, gr.m[M_INP], 64, 0, RT, row0);
    else tg_barrier();
    quant(x, S, X, H, c, A, sA, 0, 0, 0, 0);
    { const MatW m = nw.m[M_INP];
      chunk(A, 64, wq + m.wf, m.kf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, x[s], false); }
    for (uint l = 0; l < 2; l++) {
      tg_barrier();
      if (STORE) { for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s]); tg_barrier(); xcodes(S, X, cm, H, c, gq, gs, gr.m[l ? M_ZH1 : M_ZH0], 64, 0, RT, row0); }
      quant(x, S, X, H, c, A, sA, 0, 0, 0, 0);
      // zh_l: z (chunk 0), h~ (chunk 1), fp16; packed into S for the scan
      const MatW m = nw.m[l ? M_ZH1 : M_ZH0];
      float z[2][8], h[2][8];
      chunk(A, 64, wq + m.wf, m.kf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, z[s], false); for (uint k = 0; k < 8; k++) z[s][k] = float(half(act_sigmoid(z[s][k]))); }
      tg_barrier();
      chunk(A, 64, wq + m.wf + 64 * m.kf, m.kf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 1, h[s], false); for (uint k = 0; k < 8; k++) h[s][k] = float(half(act_tanh(h[s][k]))); }
      tg_barrier();
      for (uint s = 0; s < 2; s++) {
        const uint row = row_of(c, s);
        for (uint k = 0; k < 8; k++) ((threadgroup uint *)S)[row * LD + 8 * c.li + k] = as_type<uint>(half2(half(z[s][k]), half(h[s][k])));
        if (STORE) { device half *o = lzc + (size_t(row0 + row) * 2 + l) * 2 * D + 8 * c.li;
          *(device half4 *)o = half4(z[s][0], z[s][1], z[s][2], z[s][3]); *(device half4 *)(o + 4) = half4(z[s][4], z[s][5], z[s][6], z[s][7]);
          *(device half4 *)(o + D) = half4(h[s][0], h[s][1], h[s][2], h[s][3]); *(device half4 *)(o + D + 4) = half4(h[s][4], h[s][5], h[s][6], h[s][7]); }
      }
      tg_barrier();
      // the recurrence (scan.h): a SIMD group 8 channels, a lane a position, a group of eight positions scanned, the
      // groups in order from the state before them
      for (uint cc = 0; cc < 8; cc++) {
        const uint ch = 8 * c.sg + cc;
        float hin = carry[l][ch];   // set at a run's first group
        for (uint g = 0; g < 2; g++) {
          const uint r = 32 * g + c.lane; const LanePos lp = lane_pos(row0 + r, p), lg = lane_pos(row0 + 32 * g, p);
          if (lg.pos == 0) hin = lg.b < p.B ? h0[(lg.b * 2 + l) * D + ch] : 0.0f;   // a run's first group: its state h0
          const half2 zc = as_type<half2>(((threadgroup uint *)S)[r * LD + ch]);
          const float mz = (lp.valid ? mask[size_t(lp.b) * p.K * L + lp.pos] : 0.0f) * float(zc.x);
          float Aa = 1.0f - mz, Bb = mz * float(zc.y);
          scan_affine_lanes8(Aa, Bb, c.lane);
          float hv = 0;
          for (uint q = 0; q < 4; q++) { const float o = fma(Aa, hin, Bb); if (c.lane / 8 == q) hv = o; hin = simd_shuffle(o, ushort(8 * q + 7)); }
          S[r * LD + ch] = hv;
        }
        if (c.lane == 0) carry[l][ch] = hin;
      }
      tg_barrier();
      // xm = x + h; the states each refresh leaves
      for (uint s = 0; s < 2; s++) {
        const uint row = row_of(c, s); const LanePos lp = lane_pos(row0 + row, p);
        float hv[8]; s_get(S, row, c, hv);
        for (uint k = 0; k < 8; k++) x[s][k] += hv[k];
        if (STORE) { device float4 *o = (device float4 *)(lh + (size_t(row0 + row) * 2 + l) * D + 8 * c.li); o[0] = float4(hv[0], hv[1], hv[2], hv[3]); o[1] = float4(hv[4], hv[5], hv[6], hv[7]); }
        if (lp.pos % L == L - 1 && lp.valid) {
          device float4 *o = (device float4 *)(st + ((size_t(lp.b) * p.K + lp.pos / L) * 2 + l) * D + 8 * c.li); o[0] = float4(hv[0], hv[1], hv[2], hv[3]); o[1] = float4(hv[4], hv[5], hv[6], hv[7]);
          if (lp.pos == p.K * L - 1) { device float4 *n = (device float4 *)(next + (lp.b * 2 + l) * D + 8 * c.li); n[0] = o[0]; n[1] = o[1]; }
        }
      }
      if (l == 1) break;
      tg_barrier();
      if (STORE) { for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s]); tg_barrier(); xcodes(S, X, cm, H, c, gq, gs, gr.m[M_UP0], 64, 0, RT, row0); }
      quant(x, S, X, H, c, A, sA, 0, 0, 0, 0);
      if (STORE) ffn<true>(x, S, A, X, Bq, sA, sB, cm, nw.m[M_UP0], nw.m[M_DOWN0], wq, wsc, H, c, lu + size_t(row0) * 6, 6, TM, gq, gs, gr.m[M_DOWN0], RT, row0);
      else ffn<false>(x, S, A, X, Bq, sA, sB, cm, nw.m[M_UP0], nw.m[M_DOWN0], wq, wsc, H, c, lu + size_t(row0) * 6, 6, TM, gq, gs, gr.m[M_DOWN0], RT, row0);
    }
    tg_barrier();
  }
}

// ---- the forward, contexts: a threadgroup a tile of 64 contexts, through every layer, w out ----
// tokens [R][IN], pos [R] (a context's refresh: pos / L), states st [B K][2][D] -> w [R][FS]
// STAGE (stages.swift's build-up): 1 the input product alone, 2 and the first gates and step, 3 one whole layer
static inline void stage_out(thread const float (&x)[2][8], Lanes c, uint r0, uint R, device float *w) {
  for (uint s = 0; s < 2; s++) { const uint r = r0 + row_of(c, s); if (r < R && c.li < 5) for (uint k = 0; k < 8; k++) w[size_t(r) * FS + 8 * c.li + k] = x[s][k]; }
}
kernel void ctx_fwd(device const float *tokens [[buffer(0)]], device const int *pos [[buffer(1)]], device const float *st [[buffer(2)]], device float *w [[buffer(3)]],
                    constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]], constant FP &p [[buffer(8)]],
                    device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                    device half *czc [[buffer(12)]], device uint *cu [[buffer(13)]], device float *cn [[buffer(14)]], device float *cinv [[buffer(15)]],
                    device const half *H [[buffer(16)]],
                    uint tile [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  const Lanes c = lanes_of(t);
  const uint r0 = tile * TM, g0 = p.RL + r0, RT = gr.RT;   // the tile's first context, its global row (the wgrad operands')
  const uint nv = min(TM, p.R - r0);
  int ref[2];
  for (uint s = 0; s < 2; s++) { const uint r = r0 + row_of(c, s); ref[s] = r < p.R ? pos[r] / int(L) : 0; }
  for (uint i = t; i < TM * IN / 4; i += NT) {
    const uint r = i / (IN / 4), q = i % (IN / 4);
    *(threadgroup float4 *)(S + r * LD + 4 * q) = r0 + r < p.R ? ((device const float4 *)(tokens + size_t(r0 + r) * IN))[q] : float4(0);
  }
  *(threadgroup float4 *)(S + (t / 4) * LD + IN + 4 * (t % 4)) = float4(0);
  tg_barrier();
  float x[2][8];
  for (uint s = 0; s < 2; s++) s_get(S, row_of(c, s), c, x[s]);
  if (STORE) xcodes(S, X, cm, H, c, gq, gs, gr.m[M_INP], 64, 0, RT, g0);
  else tg_barrier();
  quant(x, S, X, H, c, A, sA, 0, 0, 0, 0);
  { const MatW m = nw.m[M_INP];
    chunk(A, 64, wq + m.wf, m.kf, S); tg_barrier();
    for (uint s = 0; s < 2; s++) epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, x[s], false); }
  if (STAGE == 1) { stage_out(x, c, r0, p.R, w); return; }
  for (uint l = 0; l < 2; l++) {
    const uint zh = l ? M_ZH1 : M_ZH0, up = l ? M_UP1 : M_UP0, down = l ? M_DOWN1 : M_DOWN0;
    tg_barrier();
    if (STORE) { for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s]); tg_barrier(); xcodes(S, X, cm, H, c, gq, gs, gr.m[zh], 64, 0, RT, g0); }
    quant(x, S, X, H, c, A, sA, 0, 0, 0, 0);
    // gates (fp16), then one step from the refresh's state: h = hp + z (h~ - hp), xm = x + h; the states read before
    // the gates' product (its time covers their latency)
    float hp[2][8];
    for (uint s = 0; s < 2; s++) { device const float4 *h4 = (device const float4 *)(st + (size_t(ref[s]) * 2 + l) * D + 8 * c.li); const float4 a = h4[0], b = h4[1];
      for (uint k = 0; k < 4; k++) { hp[s][k] = a[k]; hp[s][4 + k] = b[k]; } }
    { const MatW m = nw.m[zh];
      float z[2][8];
      chunk(A, 64, wq + m.wf, m.kf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, z[s], false); for (uint k = 0; k < 8; k++) z[s][k] = float(half(act_sigmoid(z[s][k]))); }
      tg_barrier();
      chunk(A, 64, wq + m.wf + 64 * m.kf, m.kf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) {
        const uint row = row_of(c, s);
        float hh[8]; epilogue(S, c, row, sA[row], wsc + m.wfs, wsc + m.b, 1, hh, false);
        for (uint k = 0; k < 8; k++) { hh[k] = float(half(act_tanh(hh[k]))); x[s][k] += fma(z[s][k], hh[k] - hp[s][k], hp[s][k]); }
        if (STORE && row < nv) { device half *o = czc + (size_t(r0 + row) * 2 + l) * 2 * D + 8 * c.li;
          *(device half4 *)o = half4(z[s][0], z[s][1], z[s][2], z[s][3]); *(device half4 *)(o + 4) = half4(z[s][4], z[s][5], z[s][6], z[s][7]);
          *(device half4 *)(o + D) = half4(hh[0], hh[1], hh[2], hh[3]); *(device half4 *)(o + D + 4) = half4(hh[4], hh[5], hh[6], hh[7]); }
      } }
    if (STAGE == 2) { stage_out(x, c, r0, p.R, w); return; }
    tg_barrier();
    if (STORE) { for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s]); tg_barrier(); xcodes(S, X, cm, H, c, gq, gs, gr.m[up], 64, 0, RT, g0); }
    quant(x, S, X, H, c, A, sA, 0, 0, 0, 0);
    if (STORE) ffn<true>(x, S, A, X, Bq, sA, sB, cm, nw.m[up], nw.m[down], wq, wsc, H, c, cu + size_t(r0) * 12 + 6 * l, 12, nv, gq, gs, gr.m[down], RT, g0);
      else ffn<false>(x, S, A, X, Bq, sA, sB, cm, nw.m[up], nw.m[down], wq, wsc, H, c, cu + size_t(r0) * 12 + 6 * l, 12, nv, gq, gs, gr.m[down], RT, g0);
    if (STAGE == 3) { stage_out(x, c, r0, p.R, w); return; }
  }
  // the readout: x2's codes and its rms from the same grid (act.h), w = out(x2) (sa / rms); n = x2 / rms for the backward
  float m0[2], inv[2];
  for (uint s = 0; s < 2; s++) {
    int isum; m0[s] = grid_row(x[s], X, c, row_of(c, s), &isum);
    inv[s] = act_rms_inverse(isum, i8_grid_step(m0[s]));
  }
  if (STORE) {
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s);
      float n[8]; for (uint k = 0; k < 8; k++) n[k] = x[s][k] * inv[s];
      if (row < nv) { device float4 *o = (device float4 *)(cn + size_t(r0 + row) * D + 8 * c.li); o[0] = float4(n[0], n[1], n[2], n[3]); o[1] = float4(n[4], n[5], n[6], n[7]); if (c.li == 0) cinv[r0 + row] = inv[s]; }
    }
  }
  tg_barrier(); hrow(X, H, S); tg_barrier();
  for (uint s = 0; s < 2; s++) { const uint row = row_of(c, s); codes_row(S, c, row, m0[s], A, 64, sA, 0, 0); }
  tg_barrier();
  if (STORE) {   // n's wgrad X codes (out's): n into S again
    for (uint s = 0; s < 2; s++) { float n[8]; for (uint k = 0; k < 8; k++) n[k] = x[s][k] * inv[s]; put_s(S, c, row_of(c, s), n); }
    tg_barrier(); xcodes(S, X, cm, H, c, gq, gs, gr.m[M_OUT], 64, 0, RT, g0);
  }
  { const MatW m = nw.m[M_OUT];
    chunk(A, 64, wq + m.wf, m.kf, S); tg_barrier();
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s), r = r0 + row;
      float y[8]; epilogue(S, c, row, sA[row] * inv[s], wsc + m.wfs, nullptr, 0, y, false);
      if (r < p.R && c.li < 5) { device float4 *o = (device float4 *)(w + size_t(r) * FS + 8 * c.li); o[0] = float4(y[0], y[1], y[2], y[3]); o[1] = float4(y[4], y[5], y[6], y[7]); }
    } }
}
