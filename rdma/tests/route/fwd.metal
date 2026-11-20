// The forward's kernels (sg.h's tiles, a SIMD group each, no threadgroup barrier but the lanes' one a layer when a
// run's refreshes are composed): a run's lanes, a tile of contexts.
#include "sg.h"
constant bool STORE [[function_constant(0)]];   // the learning forward: what the backward reads, and the wgrad codes
constant uint STAGE [[function_constant(1)]];   // stages.swift's build-up (0: everything)

// element (s, j): the thread's row slot s, its column j of 16
static inline ushort el(uint s, uint j) { return ushort((j & 3) | (((j >> 2) & 3) << 3) | ((s & 1) << 2) | ((s >> 1) << 5)); }
static inline uint last_lane(Lay a) { return lane_at(a, 7); }   // the lane holding row r = 7 of the same columns

// the gates of zh's two chunks on A (x's codes, sa): z (fp16-rounded sigmoid), hh (fp16-rounded tanh)
static inline void gates(thread CF &z, thread CF &hh, threadgroup int8_t *A, thread const float *sa, MatW m, device const int8_t *wq, device const float *wsc, Lay a) {
  CI c;
  product(c, A, 64, wq + m.wf, m.kf); epi(z, c, sa, wsc + m.wfs, wsc + m.b, a, false);
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) z[i] = float(half(act_sigmoid(z[i])));
  product(c, A, 64, wq + m.wf + 64 * m.kf, m.kf); epi(hh, c, sa, wsc + m.wfs + 64, wsc + m.b + 64, a, false);
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) hh[i] = float(half(act_tanh(hh[i])));
}
// the feed-forward: x (xm, its codes in A with sa) -> xm + down(relu(up(xm))), down's blocks as each up chunk is made
// (its codes in X's room); STORE: the masks (mw: [row] of 6 words at row stride), down's X codes
template <bool ST>
static inline void ffn(thread CF &x, threadgroup int8_t *A, threadgroup half *X, threadgroup float *cm, thread const float *sa, MatW mu, MatW md, device const int8_t *wq,
                       device const float *wsc, device const half *H, Lay a, device uint *mw, uint mstride, uint rows_valid, device int8_t *gq, device float *gs, MatG g, uint RT, uint row0) {
  _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) { const float4 b = *(device const float4 *)(wsc + md.b + col_of(a, i)); for (ushort k = 0; k < 4; k++) x[i + k] += b[k]; }
  for (uint jj = 0; jj < 3; jj++) {
    CI c; CF u;
    product(c, A, 64, wq + mu.wf + jj * 64 * mu.kf, mu.kf); epi(u, c, sa, wsc + mu.wfs + 64 * jj, wsc + mu.b + 64 * jj, a, false);
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) u[i] = max(u[i], 0.0f);
    if (ST) {
      uint w[4][2]; mask_words(u, a, w);
      if ((a.lane & 0x9) == 0) for (uint s = 0; s < 4; s++) if (row_s(a, s) < rows_valid) { mw[row_s(a, s) * mstride + 2 * jj] = w[s][0]; mw[row_s(a, s) * mstride + 2 * jj + 1] = w[s][1]; }
      colcodes(u, gq + g.xt + jj * 64 * RT, RT, row0, gs + g.sx + (row0 / TR) * 192 + 64 * jj, nullptr, 64, 4.0f, 0, X, cm, H, a);
    }
    float su[4];
    quant(u, (threadgroup int8_t *)X, 64, X, H, su, a, 0, 0, 0, 0, nullptr, nullptr);
    product(c, (threadgroup int8_t *)X, 64, wq + md.wf + 64 * jj, md.kf); epi(x, c, su, wsc + md.wfs, nullptr, a, true);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- the forward, lanes: a SIMD group a (run, refresh) tile; a threadgroup's four tiles whole runs (Kp of them a run)
// lanes [B K][L][IN], mask [B K][L], h0 [B][2][D] -> states st [B K][2][D] (each refresh's last), next [B][2][D]
kernel void lanes_fwd(device const float *lanes [[buffer(0)]], device const float *mask [[buffer(1)]], device const float *h0 [[buffer(2)]],
                      device float *st [[buffer(3)]], device float *next [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]],
                      device const float *wsc [[buffer(7)]], constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]],
                      constant Grad &gr [[buffer(11)]], device half *lzc [[buffer(12)]], device float *lh [[buffer(13)]], device uint *lu [[buffer(14)]],
                      device const half *H [[buffer(16)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  SG_MEMORY
  threadgroup float tot[2][NSG][2][64];   // [layer][tile][A, B][channel]: each tile's total map
  SG_TILE
  const uint t = tg * NSG + sg, row0 = t * TR, b = t / p.Kp, k = t % p.Kp, RT = gr.RT;
  const bool live = b < p.B && k < p.K;
  const size_t base = (size_t(b) * p.K + k) * L;   // the tile's first position in lanes, mask
  float mk[4];
  for (uint s = 0; s < 4; s++) mk[s] = live ? mask[base + row_s(a, s)] : 0.0f;
  CF x;
  _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) {
    const uint col = col_of(a, i), row = row_of(a, i);
    const float4 v = live && col < IN ? *(device const float4 *)(lanes + (base + row) * IN + col) : float4(0);
    for (ushort q = 0; q < 4; q++) x[i + q] = v[q];
  }
  if (STORE) { const MatG g = gr.m[M_INP]; colcodes(x, gq + g.xt, RT, row0, gs + g.sx + (row0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a); }
  float sa[4];
  quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, nullptr, nullptr);
  { const MatW m = nw.m[M_INP]; CI c; product(c, A, 64, wq + m.wf, m.kf); epi(x, c, sa, wsc + m.wfs, wsc + m.b, a, false); }
  for (uint l = 0; l < 2; l++) {
    const MatW mz = nw.m[l ? M_ZH1 : M_ZH0];
    if (STORE) { const MatG g = gr.m[l ? M_ZH1 : M_ZH0]; colcodes(x, gq + g.xt, RT, row0, gs + g.sx + (row0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a); }
    quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, nullptr, nullptr);
    CF z, hh;
    gates(z, hh, A, sa, mz, wq, wsc, a);
    if (STORE && live) _Pragma("unroll") for (ushort i = 0; i < NE; i++) {
      const size_t at = (size_t(row0 + row_of(a, i)) * 2 + l) * 2 * D + col_of(a, i); lzc[at] = half(z[i]); lzc[at + D] = half(hh[i]);
    }
    // the recurrence (scan.h): each position's map (A, B) = (1 - m z, m z h~), scanned in its group of eight rows
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) { const float mz = mk[slot(i)] * z[i]; z[i] = 1.0f - mz; hh[i] = mz * hh[i]; scan_affine_group(z[i], hh[i], a.r, a.lane); }
    // the state each column enters the tile with: h0, or the run's earlier tiles' totals composed (K > 1)
    float hin[16];
    for (uint j = 0; j < 16; j++) hin[j] = live ? h0[(b * 2 + l) * D + col_j(a, j)] : 0.0f;
    if (p.K > 1) {
      for (uint j = 0; j < 16; j++) {
        float tA = simd_shuffle(z[el(0, j)], ushort(last_lane(a))), tB = simd_shuffle(hh[el(0, j)], ushort(last_lane(a)));
        for (uint s = 1; s < 4; s++) { const float As = simd_shuffle(z[el(s, j)], ushort(last_lane(a))), Bs = simd_shuffle(hh[el(s, j)], ushort(last_lane(a))); tB = fma(As, tB, Bs); tA = As * tA; }
        if (a.r == 0) { tot[l][sg][0][col_j(a, j)] = tA; tot[l][sg][1][col_j(a, j)] = tB; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < k; kk++) { const uint o = sg - k + kk; for (uint j = 0; j < 16; j++) hin[j] = fma(tot[l][o][0][col_j(a, j)], hin[j], tot[l][o][1][col_j(a, j)]); }
    }
    for (uint j = 0; j < 16; j++) {
      float h = hin[j];
      for (uint s = 0; s < 4; s++) { const ushort e = el(s, j); const float hv = fma(z[e], h, hh[e]); z[e] = hv; h = simd_shuffle(hv, ushort(last_lane(a))); }
    }
    // xm = x + h; the states each refresh leaves
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) {
      const uint row = row_of(a, i), col = col_of(a, i); const float hv = z[i];
      x[i] += hv;
      if (STORE && live) lh[(size_t(row0 + row) * 2 + l) * D + col] = hv;
      if (live && row == L - 1) { st[((size_t(b) * p.K + k) * 2 + l) * D + col] = hv; if (k == p.K - 1) next[(b * 2 + l) * D + col] = hv; }
    }
    if (l == 1) break;
    if (STORE) { const MatG g = gr.m[M_UP0]; colcodes(x, gq + g.xt, RT, row0, gs + g.sx + (row0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a); }
    quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, nullptr, nullptr);
    if (STORE) ffn<true>(x, A, X, cm, sa, nw.m[M_UP0], nw.m[M_DOWN0], wq, wsc, H, a, lu + size_t(row0) * 6, 6, live ? TR : 0, gq, gs, gr.m[M_DOWN0], RT, row0);
    else ffn<false>(x, A, X, cm, sa, nw.m[M_UP0], nw.m[M_DOWN0], wq, wsc, H, a, lu, 6, 0, gq, gs, gr.m[M_DOWN0], RT, row0);
  }
}

// ---- the forward, contexts: a SIMD group a tile of 32 contexts, through every layer, w out ----
// tokens [R][IN], pos [R] (a context's refresh: pos / L), states st [B K][2][D] -> w [R][FS]
// STAGE (stages.swift's build-up): 1 the input product alone, 2 and the first gates and step, 3 one whole layer
static inline void stage_out(thread const CF &x, Lay a, uint r0, uint R, device float *w) {
  for (ushort i = 0; i < NE; i++) { const uint r = r0 + row_of(a, i), col = col_of(a, i); if (r < R && col < FS) w[size_t(r) * FS + col] = x[i]; }
}
kernel void ctx_fwd(device const float *tokens [[buffer(0)]], device const int *pos [[buffer(1)]], device const float *st [[buffer(2)]], device float *w [[buffer(3)]],
                    constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]], constant FP &p [[buffer(8)]],
                    device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                    device half *czc [[buffer(12)]], device uint *cu [[buffer(13)]], device float *cn [[buffer(14)]], device float *cinv [[buffer(15)]],
                    device const half *H [[buffer(16)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  SG_MEMORY
  SG_TILE
  const uint r0 = (tg * NSG + sg) * TR, g0 = p.RL + r0, RT = gr.RT;
  if (r0 >= p.R) return;
  const uint nv = min(TR, p.R - r0);   // the tile's live rows
  int ref[4];
  for (uint s = 0; s < 4; s++) { const uint r = r0 + row_s(a, s); ref[s] = r < p.R ? pos[r] / int(L) : 0; }
  CF x;
  _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) {
    const uint col = col_of(a, i), r = r0 + row_of(a, i);
    const float4 v = r < p.R && col < IN ? *(device const float4 *)(tokens + size_t(r) * IN + col) : float4(0);
    for (ushort q = 0; q < 4; q++) x[i + q] = v[q];
  }
  if (STORE) { const MatG g = gr.m[M_INP]; colcodes(x, gq + g.xt, RT, g0, gs + g.sx + (g0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a); }
  float sa[4];
  quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, nullptr, nullptr);
  { const MatW m = nw.m[M_INP]; CI c; product(c, A, 64, wq + m.wf, m.kf); epi(x, c, sa, wsc + m.wfs, wsc + m.b, a, false); }
  if (STAGE == 1) { stage_out(x, a, r0, p.R, w); return; }
  for (uint l = 0; l < 2; l++) {
    const uint zh = l ? M_ZH1 : M_ZH0, up = l ? M_UP1 : M_UP0, down = l ? M_DOWN1 : M_DOWN0;
    if (STORE) { const MatG g = gr.m[zh]; colcodes(x, gq + g.xt, RT, g0, gs + g.sx + (g0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a); }
    quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, nullptr, nullptr);
    // gates (fp16), then one step from the refresh's state: h = hp + z (h~ - hp), xm = x + h
    CF z, hh;
    gates(z, hh, A, sa, nw.m[zh], wq, wsc, a);
    _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) {
      const uint col = col_of(a, i), row = row_of(a, i), r = r0 + row;
      const float4 hp = *(device const float4 *)(st + (size_t(ref[slot(i)]) * 2 + l) * D + col);
      for (ushort q = 0; q < 4; q++) x[i + q] += fma(z[i + q], hh[i + q] - hp[q], hp[q]);
      if (STORE && r < p.R) for (ushort q = 0; q < 4; q++) { const size_t at = (size_t(r) * 2 + l) * 2 * D + col + q; czc[at] = half(z[i + q]); czc[at + D] = half(hh[i + q]); }
    }
    if (STAGE == 2) { stage_out(x, a, r0, p.R, w); return; }
    if (STORE) { const MatG g = gr.m[up]; colcodes(x, gq + g.xt, RT, g0, gs + g.sx + (g0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a); }
    quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, nullptr, nullptr);
    if (STORE) ffn<true>(x, A, X, cm, sa, nw.m[up], nw.m[down], wq, wsc, H, a, cu + size_t(r0) * 12 + 6 * l, 12, nv, gq, gs, gr.m[down], RT, g0);
    else ffn<false>(x, A, X, cm, sa, nw.m[up], nw.m[down], wq, wsc, H, a, cu, 12, 0, gq, gs, gr.m[down], RT, g0);
    if (STAGE == 3) { stage_out(x, a, r0, p.R, w); return; }
  }
  // the readout: x2's codes and its rms from the same grid (act.h), w = out(x2) (sa / rms)
  int isum[4]; float m0[4], inv[4];
  quant(x, A, 64, X, H, sa, a, 0, 0, 0, 0, isum, m0);
  for (uint s = 0; s < 4; s++) { inv[s] = act_rms_inverse(isum[s], m0[s]); sa[s] *= inv[s]; }
  if (STORE) {
    CF n;
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) { n[i] = x[i] * inv[slot(i)]; const uint r = r0 + row_of(a, i); if (r < p.R) cn[size_t(r) * D + col_of(a, i)] = n[i]; }
    if ((a.lane & 0x9) == 0) for (uint s = 0; s < 4; s++) if (r0 + row_s(a, s) < p.R) cinv[r0 + row_s(a, s)] = inv[s];
    const MatG g = gr.m[M_OUT]; colcodes(n, gq + g.xt, RT, g0, gs + g.sx + (g0 / TR) * 64, nullptr, 64, 4.0f, 0, X, cm, H, a);
  }
  { const MatW m = nw.m[M_OUT]; CI c; CF y; product(c, A, 64, wq + m.wf, m.kf); epi(y, c, sa, wsc + m.wfs, nullptr, a, false);
    _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) { const uint r = r0 + row_of(a, i), col = col_of(a, i);
      if (r < p.R && col < FS) *(device float4 *)(w + size_t(r) * FS + col) = float4(y[i], y[i + 1], y[i + 2], y[i + 3]); } }
}
