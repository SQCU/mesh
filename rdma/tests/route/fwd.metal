// The forward's kernels (tile.h's tiles): a run's lanes, a tile of contexts.
#include "tile.h"
constant bool STORE [[function_constant(0)]];   // the learning forward: what the backward reads, and the wgrad codes
constant uint STAGE [[function_constant(1)]];   // stages.swift's build-up (0: everything)

// ---- the forward, lanes: a threadgroup a run, its tiles in order (the recurrence's state carried), or for one refresh
// a tile of two runs ----
// lanes [B K][L][IN], mask [B K][L], h0 [B][2][D] -> states st [B K][2][D] (each refresh's last), next [B][2][D]
kernel void lanes_fwd(device const float *lanes [[buffer(0)]], device const float *mask [[buffer(1)]], device const float *h0 [[buffer(2)]],
                      device float *st [[buffer(3)]], device float *next [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]],
                      device const float *wsc [[buffer(7)]], constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]],
                      constant Grad &gr [[buffer(11)]], device half *lzc [[buffer(12)]], device float *lh [[buffer(13)]], device uint *lu [[buffer(14)]],
                      uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  threadgroup float carry[2][D];
  const Lanes c = lanes_of(t);
  for (uint j = 0; j < p.LT; j++) {
    const uint row0 = (tg * p.LT + j) * TM;   // the tile's first lane row
    // the tokens (IN of 64 columns, four a load, the rest 0): their wgrad codes (inp's X), their codes
    for (uint i = t; i < TM * IN / 4; i += NT) {
      const uint r = i / (IN / 4), q = i % (IN / 4); const LanePos lp = lane_pos(row0 + r, p);
      *(threadgroup float4 *)(S + r * LD + 4 * q) = lp.valid ? ((device const float4 *)(lanes + (size_t(lp.b) * p.K * L + lp.pos) * IN))[q] : float4(0);
    }
    *(threadgroup float4 *)(S + (t / 4) * LD + IN + 4 * (t % 4)) = float4(0);
    tg_barrier();
    if (STORE) { const MatG g = gr.m[M_INP]; col_codes(S, c, gq + g.xt, gr.RT, row0, gs + g.sx + (row0 / TM) * 64, nullptr, 64, 0); }
    float v1[2][1][8];
    for (uint s = 0; s < 2; s++) for (uint k = 0; k < 8; k++) v1[s][0][k] = S[row_of(c, s) * LD + 8 * c.li + k];
    tg_barrier();
    put_codes<1>(v1, A, sA, c, 0, 0);
    tg_barrier();
    // inp: x0
    float x[2][1][8];
    { const MatW m = nw.m[M_INP];
      chunk_fwd<64>(A, wq + m.wf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, x[s][0]); }
    for (uint l = 0; l < 2; l++) {
      if (STORE) {   // x_l: zh_l's wgrad X
        tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
        const MatG g = gr.m[l ? M_ZH1 : M_ZH0]; col_codes(S, c, gq + g.xt, gr.RT, row0, gs + g.sx + (row0 / TM) * 64, nullptr, 64, 0);
      }
      tg_barrier(); put_codes<1>(x, A, sA, c, 0, 0); tg_barrier();
      // zh_l: z (chunk 0), h~ (chunk 1), fp16; packed into S for the scan
      const MatW m = nw.m[l ? M_ZH1 : M_ZH0];
      float z[2][8], h[2][8];
      chunk_fwd<64>(A, wq + m.wf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, z[s]); for (uint k = 0; k < 8; k++) z[s][k] = float(half(act_sigmoid(z[s][k]))); }
      tg_barrier();
      chunk_fwd<64>(A, wq + m.wf + 64 * 64, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 1, h[s]); for (uint k = 0; k < 8; k++) h[s][k] = float(half(act_tanh(h[s][k]))); }
      tg_barrier();
      for (uint s = 0; s < 2; s++) {
        const uint row = row_of(c, s);
        for (uint k = 0; k < 8; k++) ((threadgroup uint *)S)[row * LD + 8 * c.li + k] = as_type<uint>(half2(half(z[s][k]), half(h[s][k])));
        if (STORE) for (uint k = 0; k < 8; k++) { const size_t at = (size_t(row0 + row) * 2 + l) * 2 * D + 8 * c.li + k; lzc[at] = half(z[s][k]); lzc[at + D] = half(h[s][k]); }
      }
      tg_barrier();
      // the recurrence (scan.h): a SIMD group 8 channels, a lane a position, the tile's two groups in order
      for (uint cc = 0; cc < 8; cc++) {
        const uint ch = 8 * c.sg + cc;
        float hin = carry[l][ch];
        for (uint g = 0; g < 2; g++) {
          const uint r = 32 * g + c.lane; const LanePos lp = lane_pos(row0 + r, p), lg = lane_pos(row0 + 32 * g, p);
          if (lg.pos == 0) hin = lg.b < p.B ? h0[(lg.b * 2 + l) * D + ch] : 0.0f;   // a run's first group: its state h0
          const half2 zc = as_type<half2>(((threadgroup uint *)S)[r * LD + ch]);
          const float mz = (lp.valid ? mask[size_t(lp.b) * p.K * L + lp.pos] : 0.0f) * float(zc.x);
          float Aa = 1.0f - mz, Bb = mz * float(zc.y);
          scan_affine_simd(Aa, Bb, c.lane);
          const float hv = fma(Aa, hin, Bb);
          S[r * LD + ch] = hv;
          hin = simd_broadcast(hv, 31);
        }
        if (c.lane == 0) carry[l][ch] = hin;
      }
      tg_barrier();
      // xm = x + h; the states each refresh leaves; layer 0's feed-forward
      for (uint s = 0; s < 2; s++) {
        const uint row = row_of(c, s); const LanePos lp = lane_pos(row0 + row, p);
        for (uint k = 0; k < 8; k++) {
          const float hv = S[row * LD + 8 * c.li + k];
          x[s][0][k] += hv;
          if (STORE) lh[(size_t(row0 + row) * 2 + l) * D + 8 * c.li + k] = hv;
          if (lp.pos % L == L - 1 && lp.valid) {
            st[((size_t(lp.b) * p.K + lp.pos / L) * 2 + l) * D + 8 * c.li + k] = hv;
            if (lp.pos == p.K * L - 1) next[(lp.b * 2 + l) * D + 8 * c.li + k] = hv;
          }
        }
      }
      if (l == 1) break;
      if (STORE) { tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
                   const MatG g = gr.m[M_UP0]; col_codes(S, c, gq + g.xt, gr.RT, row0, gs + g.sx + (row0 / TM) * 64, nullptr, 64, 0); }
      tg_barrier(); put_codes<1>(x, A, sA, c, 0, 0); tg_barrier();
      // up0: u = relu, its mask; its wgrad X (down0's) a chunk at a time
      float u[2][3][8];
      { const MatW mu = nw.m[M_UP0];
        for (uint jj = 0; jj < 3; jj++) {
          chunk_fwd<64>(A, wq + mu.wf + jj * 64 * 64, S); tg_barrier();
          for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mu.wfs, wsc + mu.b, jj, u[s][jj]); for (uint k = 0; k < 8; k++) u[s][jj][k] = max(u[s][jj][k], 0.0f); }
          if (STORE) {
            for (uint s = 0; s < 2; s++) { const uint wd = mask_word(u[s][jj], c); if (c.li % 4 == 0) lu[size_t(row0 + row_of(c, s)) * 6 + 2 * jj + c.li / 4] = wd; }
            tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), u[s][jj]); tg_barrier();
            const MatG g = gr.m[M_DOWN0]; col_codes(S, c, gq + g.xt + jj * 64 * gr.RT, gr.RT, row0, gs + g.sx + (row0 / TM) * 192 + 64 * jj, nullptr, 64, 0);
          }
          tg_barrier();
        } }
      put_codes<3>(u, A, sA, c, 0, 0); tg_barrier();
      // down0: x1 = xm + down(u)
      { const MatW md = nw.m[M_DOWN0];
        chunk_fwd<192>(A, wq + md.wf, S); tg_barrier();
        for (uint s = 0; s < 2; s++) { float y[8]; epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + md.wfs, wsc + md.b, 0, y); for (uint k = 0; k < 8; k++) x[s][0][k] += y[k]; } }
    }
    tg_barrier();
  }
}

// ---- the forward, contexts: a threadgroup a tile of 64 contexts, through every layer, w out ----
// tokens [R][IN], pos [R] (a context's refresh: pos / L), states st [B K][2][D] -> w [R][FS]
// STAGE (stages.swift's build-up): 1 the input product alone, 2 and the first gates and step, 3 one whole layer
static inline void stage_out(thread const float (&x)[2][1][8], Lanes c, uint r0, uint R, device float *w) {
  for (uint s = 0; s < 2; s++) { const uint r = r0 + row_of(c, s); if (r < R && c.li < 5) for (uint k = 0; k < 8; k++) w[size_t(r) * FS + 8 * c.li + k] = x[s][0][k]; }
}
kernel void ctx_fwd(device const float *tokens [[buffer(0)]], device const int *pos [[buffer(1)]], device const float *st [[buffer(2)]], device float *w [[buffer(3)]],
                    constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]], constant FP &p [[buffer(8)]],
                    device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                    device half *czc [[buffer(12)]], device uint *cu [[buffer(13)]], device float *cn [[buffer(14)]], device float *cinv [[buffer(15)]],
                    uint tile [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  TILE_MEMORY
  const Lanes c = lanes_of(t);
  const uint r0 = tile * TM, g0 = p.RL + r0;   // the tile's first context, its global row (the wgrad operands')
  int ref[2];   // the rows' refreshes (their states')
  for (uint s = 0; s < 2; s++) { const uint r = r0 + row_of(c, s); ref[s] = r < p.R ? pos[r] / int(L) : 0; }
  // the tokens, four columns a load (IN = 48: twelve a row), the pad columns 0
  for (uint i = t; i < TM * IN / 4; i += NT) {
    const uint r = i / (IN / 4), q = i % (IN / 4);
    *(threadgroup float4 *)(S + r * LD + 4 * q) = r0 + r < p.R ? ((device const float4 *)(tokens + size_t(r0 + r) * IN))[q] : float4(0);
  }
  *(threadgroup float4 *)(S + (t / 4) * LD + IN + 4 * (t % 4)) = float4(0);
  tg_barrier();
  if (STORE) { const MatG g = gr.m[M_INP]; col_codes(S, c, gq + g.xt, gr.RT, g0, gs + g.sx + (g0 / TM) * 64, nullptr, 64, 0); }
  float x[2][1][8];
  for (uint s = 0; s < 2; s++) for (uint k = 0; k < 8; k++) x[s][0][k] = S[row_of(c, s) * LD + 8 * c.li + k];
  tg_barrier();
  put_codes<1>(x, A, sA, c, 0, 0);
  tg_barrier();
  { const MatW m = nw.m[M_INP];
    chunk_fwd<64>(A, wq + m.wf, S); tg_barrier();
    for (uint s = 0; s < 2; s++) epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, x[s][0]); }
  if (STAGE == 1) { stage_out(x, c, r0, p.R, w); return; }
  for (uint l = 0; l < 2; l++) {
    const uint zh = l ? M_ZH1 : M_ZH0, up = l ? M_UP1 : M_UP0, down = l ? M_DOWN1 : M_DOWN0;
    if (STORE) { tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
                 const MatG g = gr.m[zh]; col_codes(S, c, gq + g.xt, gr.RT, g0, gs + g.sx + (g0 / TM) * 64, nullptr, 64, 0); }
    tg_barrier(); put_codes<1>(x, A, sA, c, 0, 0); tg_barrier();
    // gates (fp16), then one step from the refresh's state: h = hp + z (h~ - hp), xm = x + h; the states read before
    // the gates' product (its time covers their latency)
    float hp[2][8];
    for (uint s = 0; s < 2; s++) { device const float4 *h4 = (device const float4 *)(st + (size_t(ref[s]) * 2 + l) * D + 8 * c.li); const float4 a = h4[0], b = h4[1];
      for (uint k = 0; k < 4; k++) { hp[s][k] = a[k]; hp[s][4 + k] = b[k]; } }
    { const MatW m = nw.m[zh];
      float z[2][8];
      chunk_fwd<64>(A, wq + m.wf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + m.wfs, wsc + m.b, 0, z[s]); for (uint k = 0; k < 8; k++) z[s][k] = float(half(act_sigmoid(z[s][k]))); }
      tg_barrier();
      chunk_fwd<64>(A, wq + m.wf + 64 * 64, S); tg_barrier();
      for (uint s = 0; s < 2; s++) {
        const uint row = row_of(c, s);
        float hh[8]; epilogue(S, c, row, sA[row], wsc + m.wfs, wsc + m.b, 1, hh);
        for (uint k = 0; k < 8; k++) { hh[k] = float(half(act_tanh(hh[k]))); x[s][0][k] += fma(z[s][k], hh[k] - hp[s][k], hp[s][k]); }
        if (STORE && r0 + row < p.R) for (uint k = 0; k < 8; k++) { const size_t at = (size_t(r0 + row) * 2 + l) * 2 * D + 8 * c.li + k; czc[at] = half(z[s][k]); czc[at + D] = half(hh[k]); }
      } }
    if (STAGE == 2) { stage_out(x, c, r0, p.R, w); return; }
    if (STORE) { tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
                 const MatG g = gr.m[up]; col_codes(S, c, gq + g.xt, gr.RT, g0, gs + g.sx + (g0 / TM) * 64, nullptr, 64, 0); }
    tg_barrier(); put_codes<1>(x, A, sA, c, 0, 0); tg_barrier();
    float u[2][3][8];
    { const MatW mu = nw.m[up];
      for (uint jj = 0; jj < 3; jj++) {
        chunk_fwd<64>(A, wq + mu.wf + jj * 64 * 64, S); tg_barrier();
        for (uint s = 0; s < 2; s++) {
          epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + mu.wfs, wsc + mu.b, jj, u[s][jj]);
          for (uint k = 0; k < 8; k++) u[s][jj][k] = max(u[s][jj][k], 0.0f);
          if (STORE) { const uint wd = mask_word(u[s][jj], c), r = r0 + row_of(c, s); if (c.li % 4 == 0 && r < p.R) cu[size_t(r) * 12 + 6 * l + 2 * jj + c.li / 4] = wd; }
        }
        if (STORE) { tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), u[s][jj]); tg_barrier();
                     const MatG g = gr.m[down]; col_codes(S, c, gq + g.xt + jj * 64 * gr.RT, gr.RT, g0, gs + g.sx + (g0 / TM) * 192 + 64 * jj, nullptr, 64, 0); }
        tg_barrier();
      } }
    put_codes<3>(u, A, sA, c, 0, 0); tg_barrier();
    { const MatW md = nw.m[down];
      chunk_fwd<192>(A, wq + md.wf, S); tg_barrier();
      for (uint s = 0; s < 2; s++) { float y[8]; epilogue(S, c, row_of(c, s), sA[row_of(c, s)], wsc + md.wfs, wsc + md.b, 0, y); for (uint k = 0; k < 8; k++) x[s][0][k] += y[k]; } }
    if (STAGE == 3) { stage_out(x, c, r0, p.R, w); return; }
  }
  // the readout: n = x2 / rms(x2), w = out(n)
  for (uint s = 0; s < 2; s++) {
    float q = 0; for (uint k = 0; k < 8; k++) q = fma(x[s][0][k], x[s][0][k], q);   // act.h's order
    for (ushort d = 1; d < 8; d <<= 1) q += simd_shuffle_xor(q, d);
    const float inv = act_rms_inverse(q);
    for (uint k = 0; k < 8; k++) x[s][0][k] *= inv;
    const uint r = r0 + row_of(c, s);
    if (STORE && r < p.R) { for (uint k = 0; k < 8; k++) cn[size_t(r) * D + 8 * c.li + k] = x[s][0][k]; if (c.li == 0) cinv[r] = inv; }
  }
  if (STORE) { tg_barrier(); for (uint s = 0; s < 2; s++) put_s(S, c, row_of(c, s), x[s][0]); tg_barrier();
               const MatG g = gr.m[M_OUT]; col_codes(S, c, gq + g.xt, gr.RT, g0, gs + g.sx + (g0 / TM) * 64, nullptr, 64, 0); }
  tg_barrier(); put_codes<1>(x, A, sA, c, 0, 0); tg_barrier();
  { const MatW m = nw.m[M_OUT];
    chunk_fwd<64>(A, wq + m.wf, S); tg_barrier();
    for (uint s = 0; s < 2; s++) {
      const uint row = row_of(c, s), r = r0 + row;
      float y[8]; epilogue(S, c, row, sA[row], wsc + m.wfs, nullptr, 0, y);
      if (r < p.R && c.li < 5) for (uint k = 0; k < 8; k++) w[size_t(r) * FS + 8 * c.li + k] = y[k];
    } }
}
