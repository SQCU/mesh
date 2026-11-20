// The learning step's backward (sg.h's tiles, a SIMD group each): a task a threadgroup row, each product's dgrad codes
// (row, stochastic) and the wgrad codes of every gradient it makes (column, stochastic; its column sums: the bias's
// gradient), the contexts' state gradients summed a refresh (a product with the tile's segment matrix) for the lanes'
// recurrence.
#include "sg.h"

// a gradient's wgrad codes: the column codes of a chunk (j) of a matrix's output gradient (`rows` its outputs): task
// t's columns at t rows + 64 j of the matrix's dY codes, its scales and column sums at the tile's block
static inline void ycodes(thread const CF &v, device int8_t *gq, device float *gs, MatG g, uint rows, uint task, uint j, uint RT, uint row0, uint seed,
                          threadgroup half *X, threadgroup float *cm, device const half *H, Lay a) {
  const uint M = 3 * rows, col0 = task * rows + 64 * j, blk = row0 / TR;
  colcodes(v, gq + g.yt + size_t(col0) * RT, RT, row0, gs + g.sy + blk * M + col0, gs + g.dbp + blk * M + col0, min(64u, rows - 64 * j), 8.0f, seed, X, cm, H, a);
}
// a product's ids for the seeds (row codes: id, column codes: id + 64)
enum : uint { S_OUT = 1, S_DOWN1, S_UP1, S_ZH1, S_DOWN0, S_UP0, S_ZH0, S_INP, S_LZH1, S_LDOWN0, S_LUP0, S_LZH0, S_LINP };

// a block's codes (in X's room) through a dgrad block: acc += (or =) codes . W's block (wb: its rows' 64-block jj)
static inline void dgrad_block(thread CF &acc, thread const CF &v, MatW m, uint jj, bool first, threadgroup half *X, device const half *H, device const int8_t *wq,
                               device const float *wsc, Lay a, uint seed, uint row0, uint cols) {
  float s[4]; CI c; LI A;
  quant(v, A, H, s, a, seed, row0, cols, 64 * jj, nullptr, nullptr);
  product(c, A, wq + m.wb + 64 * jj, m.kb); epi(acc, c, s, wsc + m.wbs, nullptr, a, !first);
}
// the feed-forward back: x = dL/dx' (its codes in A with sa) -> x + (relu' (x down)) up; up's dY codes, mask bits from
// mw (row stride mstride; live rows only)
static inline void ffn_back(thread CF &x, thread LI &Ax, threadgroup half *X, threadgroup float *cm, thread const float *sa, MatW mu, MatW md,
                            device const int8_t *wq, device const float *wsc, device const half *H, Lay a, device const uint *mw, uint mstride, uint nv,
                            device int8_t *gq, device float *gs, MatG gu, uint task, uint RT, uint row0, uint seedc, uint seedr) {
  for (uint jj = 0; jj < 3; jj++) {
    CI c; CF du;
    product(c, Ax, wq + md.wb + 64 * jj * md.kb, md.kb); epi(du, c, sa, wsc + md.wbs + 64 * jj, nullptr, a, false);
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) { const uint row = row_of(a, i); if (row >= nv || !mask_bit(mw + row * mstride, 64 * jj + col_of(a, i))) du[i] = 0; }
    ycodes(du, gq, gs, gu, E, task, jj, RT, row0, seedc, X, cm, H, a);
    dgrad_block(x, du, mu, jj, false, X, H, wq, wsc, a, seedr, row0, E);
  }
}

// ---- the backward, contexts: a SIMD group a (tile of 32 contexts, task), from dw through every layer; each refresh's
// contexts' state gradients summed in the tile (rows of a refresh are contiguous): dhp [task][layer][Rp][D] at the row
// where the refresh's rows begin in the tile ----
kernel void ctx_bwd(device const float *dw [[buffer(0)]], device const int *pos [[buffer(1)]], device const float *st [[buffer(2)]], device const half *czc [[buffer(3)]],
                    device const uint *cu [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]],
                    constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                    device const float *cn [[buffer(12)]], device const float *cinv [[buffer(13)]], device float *dhp [[buffer(14)]], device const half *H [[buffer(16)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  SG_MEMORY
  SG_TILE
  const uint task = tg.y, r0 = (tg.x * NSG + sg) * TR, g0 = p.RL + r0, RT = gr.RT;
  if (r0 >= p.R) return;
  const uint nv = min(TR, p.R - r0);
  int ref[4];
  for (uint s = 0; s < 4; s++) { const uint r = r0 + row_s(a, s); ref[s] = r < p.R ? pos[r] / int(L) : -1; }
  CF x; tile_load(x, dw + (size_t(task) * p.R + r0) * FS, FS, nv, FS);
  ycodes(x, gq, gs, gr.m[M_OUT], FS, task, 0, RT, g0, seed_of(p.seed, S_OUT + 64, task), X, cm, H, a);
  float sa[4]; LI Ax;
  quant(x, Ax, H, sa, a, seed_of(p.seed, S_OUT, task), g0, 64, 0, nullptr, nullptr);
  // the readout back: dn = dw out, then rms's backward dx2 = inv (dn - n mean(dn . n))
  { const MatW m = nw.m[M_OUT]; CI c;
    product(c, Ax, wq + m.wb, m.kb); epi(x, c, sa, wsc + m.wbs, nullptr, a, false);
    CF nn, q;
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) { const uint r = r0 + row_of(a, i); nn[i] = r < p.R ? cn[size_t(r) * D + col_of(a, i)] : 0.0f; q[i] = x[i] * nn[i]; }
    RF rs; reduce_rows(q, rs, reduction_operation::sum, 0.0f);
    float inv[4], mn[4];
    for (uint s = 0; s < 4; s++) { const uint r = r0 + row_s(a, s); inv[s] = r < p.R ? cinv[r] : 0.0f; mn[s] = rs[s] / float(D); }
    _Pragma("unroll") for (ushort i = 0; i < NE; i++) x[i] = inv[slot(i)] * (x[i] - nn[i] * mn[slot(i)]); }
  for (uint l = 2; l-- > 0;) {
    const uint zh = l ? M_ZH1 : M_ZH0, up = l ? M_UP1 : M_UP0, down = l ? M_DOWN1 : M_DOWN0;
    const uint sd = l ? S_DOWN1 : S_DOWN0, su = l ? S_UP1 : S_UP0, sz = l ? S_ZH1 : S_ZH0;
    // x: dL/dx_{l+1}: down's dY; its codes; the feed-forward back: dxm
    ycodes(x, gq, gs, gr.m[down], D, task, 0, RT, g0, seed_of(p.seed, sd + 64, task), X, cm, H, a);
    quant(x, Ax, H, sa, a, seed_of(p.seed, sd, task), g0, 64, 0, nullptr, nullptr);
    ffn_back(x, Ax, X, cm, sa, nw.m[up], nw.m[down], wq, wsc, H, a, cu + size_t(r0) * 12 + 6 * l, 12, nv, gq, gs, gr.m[up], task, RT, g0,
             seed_of(p.seed, su + 64, task), seed_of(p.seed, su, task));
    // the step back through the context's state (h = hp + z (h~ - hp)): the gates' gradients and hp's
    CF dz, dc, dh;
    _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) {
      const uint col = col_of(a, i), row = row_of(a, i), r = r0 + row;
      const bool live = row < nv;
      const float4 hp = live ? *(device const float4 *)(st + (size_t(ref[slot(i)]) * 2 + l) * D + col) : float4(0);
      device const half *zc = czc + (size_t(live ? r : 0) * 2 + l) * 2 * D + col;
      for (ushort q = 0; q < 4; q++) {
        const float g = live ? x[i + q] : 0.0f, z = float(zc[q]), h = float(zc[D + q]);
        dz[i + q] = g * (h - hp[q]) * z * (1.0f - z); dc[i + q] = g * z * (1.0f - h * h); dh[i + q] = g * (1.0f - z);
      }
    }
    // dh summed a refresh: S [32 starts][32 rows] (1 where row j is in the segment starting at row i) . dh, fp16
    { threadgroup half *Sg = (threadgroup half *)A;   // A (x's codes) is spent
      CHh hd; _Pragma("unroll") for (ushort i = 0; i < NE; i++) hd[i] = half(dh[i]);
      TgH tD(X, dextents<int32_t, 2>(64, TR)); hd.store(tD);
      threadgroup int *rf = (threadgroup int *)cm;   // the tile's rows' refreshes (-1 past the end)
      rf[lane] = lane < nv ? pos[r0 + lane] / int(L) : -1;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < TR * TR; e += 32) {
        const uint i = e / TR, j = e % TR;
        const bool start = rf[i] >= 0 && (i == 0 || rf[i - 1] != rf[i]);
        Sg[e] = half(start && j >= i && rf[j] == rf[i] ? 1.0f : 0.0f);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      TgH tS(Sg, dextents<int32_t, 2>(TR, TR));
      constexpr auto ds = matmul2d_descriptor(TR, 64, TR, false, false);
      matmul2d<ds, execution_simdgroup> ops;
      auto sum = ops.get_destination_cooperative_tensor<TgH, TgH, float>();
      ops.run(tS, tD, sum);
      device float *out = dhp + ((size_t(task) * 2 + l) * p.Rp + r0) * D;
      _Pragma("unroll") for (ushort i = 0; i < NE; i += 4) {
        const auto at = sum.get_multidimensional_index(i); const uint row = at[1], col = at[0];
        if (rf[row] >= 0 && (row == 0 || rf[row - 1] != rf[row])) *(device float4 *)(out + size_t(row) * D + col) = float4(sum[i], sum[i + 1], sum[i + 2], sum[i + 3]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup); }
    // the gates' gradients: zh's dY (z, then h~), its dgrad: dx_l = dxm + dzc zh
    ycodes(dz, gq, gs, gr.m[zh], 2 * D, task, 0, RT, g0, seed_of(p.seed, sz + 64, task), X, cm, H, a);
    ycodes(dc, gq, gs, gr.m[zh], 2 * D, task, 1, RT, g0, seed_of(p.seed, sz + 64, task), X, cm, H, a);
    dgrad_block(x, dz, nw.m[zh], 0, false, X, H, wq, wsc, a, seed_of(p.seed, sz, task), g0, 128);
    dgrad_block(x, dc, nw.m[zh], 1, false, X, H, wq, wsc, a, seed_of(p.seed, sz, task), g0, 128);
  }
  // dx0: inp's dY
  ycodes(x, gq, gs, gr.m[M_INP], D, task, 0, RT, g0, seed_of(p.seed, S_INP + 64, task), X, cm, H, a);
}

// the reverse recurrence over a tile (scan.h's groups of eight rows, the tile's total composed with the run's later
// tiles' when K > 1): for each element the gradient arriving at its state u (consumed), the carry from the later tile
// (ein: a column's); each position's gates' gradients (z: m D (h~ - h_{t-1}) z (1 - z); h~: m D z (1 - h~^2), D = u +
// e_{t+1} the state's own gradient) into dz, dc
static inline ushort el2(uint s, uint j) { return ushort((j & 3) | (((j >> 2) & 3) << 3) | ((s & 1) << 2) | ((s >> 1) << 5)); }
static inline void scan_back(thread CF &u, thread CF &dz, thread CF &dc, thread const float *mk, Lay a, device const half *lzc, device const float *lh,
                             device const float *h0, uint l, uint row0, uint b, uint k, bool live, constant FP &p,
                             threadgroup float (*tot)[2][64], uint sg) {
  // each position's map (a, a u), a = 1 - m z
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) {
    const float z = live ? float(lzc[(size_t(row0 + row_of(a, i)) * 2 + l) * 2 * D + col_of(a, i)]) : 0.0f;
    const float am = 1.0f - mk[slot(i)] * z;
    dz[i] = am; dc[i] = am * u[i];
    scan_affine_group_reverse(dz[i], dc[i], a.r, a.lane);
  }
  const ushort first = lane_at(a, 0), next_lane = lane_at(a, min(a.r + 1, 7u));
  float ein[16];
  for (uint j = 0; j < 16; j++) ein[j] = 0;
  if (p.K > 1) {
    for (uint j = 0; j < 16; j++) {   // the tile's total: groups 3, 2, 1, 0 applied in turn
      float tA = simd_shuffle(dz[el2(3, j)], first), tB = simd_shuffle(dc[el2(3, j)], first);
      for (uint s = 3; s-- > 0;) { const float As = simd_shuffle(dz[el2(s, j)], first), Bs = simd_shuffle(dc[el2(s, j)], first); tB = fma(As, tB, Bs); tA = As * tA; }
      if (a.r == 0) { tot[sg][0][col_j(a, j)] = tA; tot[sg][1][col_j(a, j)] = tB; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = p.K; kk-- > k + 1;) { const uint o = sg + kk - k; for (uint j = 0; j < 16; j++) ein[j] = fma(tot[o][0][col_j(a, j)], ein[j], tot[o][1][col_j(a, j)]); }
  }
  // e_t from the group's carry, latest group first; D_t = u_t + e_{t+1}
  for (uint j = 0; j < 16; j++) {
    float e_in = ein[j];
    for (uint s = 4; s-- > 0;) {
      const ushort e = el2(s, j);
      const float ev = fma(dz[e], e_in, dc[e]), later = simd_shuffle(ev, next_lane);
      dz[e] = u[e] + (a.r == 7 ? e_in : later);   // D
      e_in = simd_shuffle(ev, first);
    }
  }
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) {
    const uint row = row_of(a, i), col = col_of(a, i), pos = k * L + row;
    device const half *zc = lzc + (size_t(row0 + row) * 2 + l) * 2 * D + col;
    const float z = live ? float(zc[0]) : 0.0f, h = live ? float(zc[D]) : 0.0f, m = mk[slot(i)], Dh = dz[i];
    const float hprev = !live ? 0.0f : pos > 0 ? lh[(size_t(row0 + row - 1) * 2 + l) * D + col] : h0[(b * 2 + l) * D + col];
    dz[i] = m * Dh * (h - hprev) * z * (1.0f - z); dc[i] = m * Dh * z * (1.0f - h * h);
  }
}

// ---- the backward, lanes: a SIMD group a (run, refresh) tile, a threadgroup row a task: layer 1's states' gradients
// from the contexts (dhp, summed a refresh), its gates', x1's (zh1's dgrad), layer 0's feed-forward, its recurrence
// (xm0's own gradient at every position, the contexts' at a refresh's end), x0's ----
kernel void lanes_bwd(device const float *mask [[buffer(0)]], device const float *h0 [[buffer(1)]], device const half *lzc [[buffer(2)]], device const float *lh [[buffer(3)]],
                      device const uint *lu [[buffer(4)]], constant Net &nw [[buffer(5)]], device const int8_t *wq [[buffer(6)]], device const float *wsc [[buffer(7)]],
                      constant FP &p [[buffer(8)]], device int8_t *gq [[buffer(9)]], device float *gs [[buffer(10)]], constant Grad &gr [[buffer(11)]],
                      device const uint *seg [[buffer(12)]], device const float *dhp [[buffer(13)]], device const half *H [[buffer(16)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  SG_MEMORY
  threadgroup float tot[2][NSG][2][64];
  SG_TILE
  const uint task = tg.y, t = tg.x * NSG + sg, row0 = t * TR, b = t / p.Kp, k = t % p.Kp, RT = gr.RT;
  const bool live = b < p.B && k < p.K;
  float mk[4];
  for (uint s = 0; s < 4; s++) mk[s] = live ? mask[(size_t(b) * p.K + k) * L + row_s(a, s)] : 0.0f;
  // the refresh's contexts' state gradients (both layers, the thread's 16 columns)
  float inj[2][16];
  for (uint l = 0; l < 2; l++) for (uint j = 0; j < 16; j++) inj[l][j] = 0;
  if (live) {
    const uint i = b * p.K + k, x0 = seg[2 * i], x1 = seg[2 * i + 1];
    for (uint l = 0; l < 2; l++) {
      device const float *src = dhp + (size_t(task) * 2 + l) * p.Rp * D;
      for (uint tt = x0 / TR; x1 > x0 && tt <= (x1 - 1) / TR; tt++) { const size_t at = size_t(tt == x0 / TR ? x0 : tt * TR) * D; for (uint j = 0; j < 16; j++) inj[l][j] += src[at + col_j(a, j)]; }
    }
  }
  CF u, dz, dc, x;
  // layer 1: the recurrence back from the contexts' gradients alone (at the refresh's last position)
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) u[i] = row_of(a, i) == L - 1 ? inj[1][cslot(i)] : 0.0f;
  scan_back(u, dz, dc, mk, a, lzc, lh, h0, 1, row0, b, k, live, p, tot[1], sg);
  ycodes(dz, gq, gs, gr.m[M_ZH1], 2 * D, task, 0, RT, row0, seed_of(p.seed, S_LZH1 + 64, task), X, cm, H, a);
  ycodes(dc, gq, gs, gr.m[M_ZH1], 2 * D, task, 1, RT, row0, seed_of(p.seed, S_LZH1 + 64, task), X, cm, H, a);
  dgrad_block(x, dz, nw.m[M_ZH1], 0, true, X, H, wq, wsc, a, seed_of(p.seed, S_LZH1, task), row0, 128);
  dgrad_block(x, dc, nw.m[M_ZH1], 1, false, X, H, wq, wsc, a, seed_of(p.seed, S_LZH1, task), row0, 128);
  // x1's gradient: down0's dY; layer 0's feed-forward back
  ycodes(x, gq, gs, gr.m[M_DOWN0], D, task, 0, RT, row0, seed_of(p.seed, S_LDOWN0 + 64, task), X, cm, H, a);
  float sa[4]; LI Ax;
  quant(x, Ax, H, sa, a, seed_of(p.seed, S_LDOWN0, task), row0, 64, 0, nullptr, nullptr);
  ffn_back(x, Ax, X, cm, sa, nw.m[M_UP0], nw.m[M_DOWN0], wq, wsc, H, a, lu + size_t(row0) * 6, 6, live ? TR : 0, gq, gs, gr.m[M_UP0], task, RT, row0,
           seed_of(p.seed, S_LUP0 + 64, task), seed_of(p.seed, S_LUP0, task));
  // layer 0: the recurrence back from xm0's gradient (every position) and the contexts'
  _Pragma("unroll") for (ushort i = 0; i < NE; i++) u[i] = (live ? x[i] : 0.0f) + (row_of(a, i) == L - 1 ? inj[0][cslot(i)] : 0.0f);
  scan_back(u, dz, dc, mk, a, lzc, lh, h0, 0, row0, b, k, live, p, tot[0], sg);
  ycodes(dz, gq, gs, gr.m[M_ZH0], 2 * D, task, 0, RT, row0, seed_of(p.seed, S_LZH0 + 64, task), X, cm, H, a);
  ycodes(dc, gq, gs, gr.m[M_ZH0], 2 * D, task, 1, RT, row0, seed_of(p.seed, S_LZH0 + 64, task), X, cm, H, a);
  dgrad_block(x, dz, nw.m[M_ZH0], 0, false, X, H, wq, wsc, a, seed_of(p.seed, S_LZH0, task), row0, 128);
  dgrad_block(x, dc, nw.m[M_ZH0], 1, false, X, H, wq, wsc, a, seed_of(p.seed, S_LZH0, task), row0, 128);
  if (!live) for (ushort i = 0; i < NE; i++) x[i] = 0;
  ycodes(x, gq, gs, gr.m[M_INP], D, task, 0, RT, row0, seed_of(p.seed, S_LINP + 64, task), X, cm, H, a);
}
