// The parameters' kernels: the weights' codes from theta, the gradients from the wgrad products' partial sums (the
// chain rule of W = diag(g) V / |V|), and the optimizer's step (the three gradients' Gram matrix, UPGrad's weights,
// Adam, V pinned to unit rows) in one threadgroup: theta is 72,872 floats, a few passes of 1,024 threads.
#include "tile.h"

// a work item: (matrix, row or column)
struct Item { uint m, i; };

// a SIMD group a matrix row (rows padded to 64): W = g V / |V| (fp32, Wf [kb][np] a matrix: the column pass's input),
// its forward codes (rotated along the columns, one scale a row) and bias; pad rows and columns 0
kernel void weights_rows(device const float *theta [[buffer(0)]], device int8_t *wq [[buffer(1)]], device float *wsc [[buffer(2)]], device float *Wf [[buffer(3)]],
                         constant Net &nw [[buffer(4)]], device const Item *items [[buffer(5)]], constant uint &n [[buffer(6)]], constant uint *wfoff [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid / 32 >= n) return;
  const Item it = items[gid / 32]; const MatW m = nw.m[it.m]; const uint r = it.i, lane = gid % 32;
  const bool live = r < m.rows;
  float v[8], s = 0;
  for (uint k = 0; k < 8; k++) { const uint j = 8 * lane + k; v[k] = live && j < m.cols ? theta[m.v + r * m.cols + j] : 0.0f; s = fma(v[k], v[k], s); }
  s = simd_sum(s);
  const float gain = live ? i8_weight_gain(theta[m.g + r], s) : 0.0f;
  float m0 = 0;
  for (uint k = 0; k < 8; k++) { v[k] *= gain; const uint j = 8 * lane + k; if (j < m.np) Wf[wfoff[it.m] + r * m.np + j] = v[k]; m0 = max(m0, abs(v[k])); }
  m0 = simd_max(m0);   // int8mm.h's weight row: one grid, one scale
  const float gi = i8_grid_inverse(m0);
  float mx = 0;
  for (uint k = 0; k < 8; k++) v[k] = i8_grid(v[k], gi);
  i8_fwht64_lanes(v, ushort(lane % 8));
  for (uint k = 0; k < 8; k++) mx = max(mx, abs(v[k]));
  mx = simd_max(mx); const float inv = i8_inverse(mx);
  if (8 * lane < m.kf) *(device uint2 *)(wq + m.wf + r * m.kf + 8 * lane) = i8_pack8(v, inv, 0, 0);
  if (lane == 0) { wsc[m.wfs + r] = i8_scale(mx, m0, 8.0f); wsc[m.b + r] = live && it.m != M_OUT ? theta[m.g + m.rows + r] : 0.0f; }
}
// a SIMD group a matrix column (columns padded to 64): its dgrad codes along the rows (padded to 64), one scale
kernel void weights_cols(device int8_t *wq [[buffer(1)]], device float *wsc [[buffer(2)]], device const float *Wf [[buffer(3)]], constant Net &nw [[buffer(4)]],
                         device const Item *items [[buffer(5)]], constant uint &n [[buffer(6)]], constant uint *wfoff [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (gid / 32 >= n) return;
  const Item it = items[gid / 32]; const MatW m = nw.m[it.m]; const uint col = it.i, lane = gid % 32;
  float v[8];
  float m0 = 0;
  for (uint k = 0; k < 8; k++) { const uint r = 8 * lane + k; v[k] = r < m.kb ? Wf[wfoff[it.m] + r * m.np + col] : 0.0f; m0 = max(m0, abs(v[k])); }
  m0 = simd_max(m0);
  const float gi = i8_grid_inverse(m0);
  float mx = 0;
  for (uint k = 0; k < 8; k++) v[k] = i8_grid(v[k], gi);
  i8_fwht64_lanes(v, ushort(lane % 8));
  for (uint k = 0; k < 8; k++) mx = max(mx, abs(v[k]));
  mx = simd_max(mx); const float inv = i8_inverse(mx);
  for (uint k = 0; k < 8; k++) { const uint r = 8 * lane + k; if (r < m.kb) wq[m.wb + col * m.kb + r] = i8_code(v[k], inv, 0, 0); }
  if (lane == 0) wsc[m.wbs + col] = i8_scale(mx, m0, 8.0f);
}

// the codes from the weights themselves (qnn.py's ops: W computed by the array library): w holds each matrix's W
// [rows][cols] at m.v and its bias [rows] at m.g (no bias: M_OUT); a SIMD group an item, the matrices' rows (items
// [0, nrows): W's forward codes rotated along the columns, one scale a row, the bias; pad rows 0) then their columns
// (items [nrows, n): dgrad codes along the rows, one scale a column), as weights_rows and weights_cols
kernel void codes(device const float *w [[buffer(0)]], device int8_t *wq [[buffer(1)]], device float *wsc [[buffer(2)]], constant Net &nw [[buffer(4)]],
                  device const Item *items [[buffer(5)]], constant uint2 &n [[buffer(6)]], uint gid [[thread_position_in_grid]]) {   // n: (items, rows' items)
  if (gid / 32 >= n.x) return;
  const bool row = gid / 32 < n.y;
  const Item it = items[gid / 32]; const MatW m = nw.m[it.m]; const uint i = it.i, lane = gid % 32;
  float v[8], m0 = 0;
  for (uint k = 0; k < 8; k++) {
    const uint j = 8 * lane + k, r = row ? i : j, col = row ? j : i;
    v[k] = r < m.rows && col < m.cols ? w[m.v + r * m.cols + col] : 0.0f; m0 = max(m0, abs(v[k]));
  }
  m0 = simd_max(m0);
  const float gi = i8_grid_inverse(m0);
  float mx = 0;
  for (uint k = 0; k < 8; k++) v[k] = i8_grid(v[k], gi);
  i8_fwht64_lanes(v, ushort(lane % 8));
  for (uint k = 0; k < 8; k++) mx = max(mx, abs(v[k]));
  mx = simd_max(mx); const float inv = i8_inverse(mx);
  if (row) {
    if (8 * lane < m.kf) *(device uint2 *)(wq + m.wf + i * m.kf + 8 * lane) = i8_pack8(v, inv, 0, 0);
    if (lane == 0) { wsc[m.wfs + i] = i8_scale(mx, m0, 8.0f); wsc[m.b + i] = i < m.rows && it.m != M_OUT ? w[m.g + i] : 0.0f; }
  } else {
    for (uint k = 0; k < 8; k++) { const uint r = 8 * lane + k; if (r < m.kb) wq[m.wb + i * m.kb + r] = i8_code(v[k], inv, 0, 0); }
    if (lane == 0) wsc[m.wbs + i] = i8_scale(mx, m0, 8.0f);
  }
}

// a SIMD group a (task, matrix, row): dW's row summed over the wgrad product's partial sums [z][3 rows][kf], db over
// the column sums [block][3 rows] of the matrix's blocks; then with u = V / |V| a row: dg = dW . u,
// dV = g / |V| (dW - (dW . u) u), db -> grads [task][P]
struct GI { uint m, task, row, partial, nz, dbp, b0, b1; };
kernel void grads(device const float *theta [[buffer(0)]], device const float *partial [[buffer(1)]], device const float *dbp [[buffer(2)]], device float *grads [[buffer(3)]],
                  constant Net &nw [[buffer(4)]], device const GI *items [[buffer(5)]], constant uint2 &p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {   // p: (items, P)
  if (gid / 32 >= p.x) return;
  const GI it = items[gid / 32]; const MatW m = nw.m[it.m]; const uint lane = gid % 32, mr = it.task * m.rows + it.row, M = 3 * m.rows;
  float dW[6], s = 0, dot = 0;   // columns lane + 32 i
  for (uint i = 0; i < 6; i++) {
    const uint j = lane + 32 * i; dW[i] = 0;
    if (j >= m.cols) continue;
    for (uint z = 0; z < it.nz; z++) dW[i] += partial[it.partial + (size_t(z) * M + mr) * m.kf + j];
    const float v = theta[m.v + it.row * m.cols + j];
    s = fma(v, v, s); dot = fma(dW[i], v, dot);
  }
  s = simd_sum(s); dot = simd_sum(dot);
  float db = 0;
  for (uint b = it.b0 + lane; b < it.b1; b += 32) db += dbp[it.dbp + size_t(b) * M + mr];
  db = simd_sum(db);
  const float k = rsqrt(max(s, 1e-30f)), u = dot * k, gk = theta[m.g + it.row] * k;
  device float *g = grads + size_t(it.task) * p.y;
  for (uint i = 0; i < 6; i++) { const uint j = lane + 32 * i; if (j < m.cols) g[m.v + it.row * m.cols + j] = gk * (dW[i] - u * k * theta[m.v + it.row * m.cols + j]); }
  if (lane == 0) { g[m.g + it.row] = u; if (it.m != M_OUT) g[m.g + m.rows + it.row] = db; }
}

// the optimizer's step, one threadgroup of 1,024: the Gram matrix of the three gradients (fixed order), UPGrad's
// weights (upgrad.h, scaled by inv: the gradients a sum over that many ranks'), Adam on the aggregate, V's rows pinned
struct OP { uint P, rows; float rate, inv, c1, c2, pad0, pad1; };
kernel void optimizer(device float *theta [[buffer(0)]], device float *mo [[buffer(1)]], device float *vo [[buffer(2)]], device const float *g [[buffer(3)]],
                      constant Net &nw [[buffer(4)]], device const Item *rows [[buffer(5)]], constant OP &p [[buffer(6)]], device float *gram [[buffer(7)]],
                      uint l [[thread_position_in_threadgroup]]) {
  threadgroup float part[6][32], alpha[3];
  float s[6] = {0, 0, 0, 0, 0, 0};
  for (uint i = l; i < p.P; i += 1024) {
    const float a = g[i], b = g[p.P + i], c = g[2 * p.P + i];
    s[0] = fma(a, a, s[0]); s[1] = fma(a, b, s[1]); s[2] = fma(a, c, s[2]); s[3] = fma(b, b, s[3]); s[4] = fma(b, c, s[4]); s[5] = fma(c, c, s[5]);
  }
  for (uint k = 0; k < 6; k++) { const float v = simd_sum(s[k]); if (l % 32 == 0) part[k][l / 32] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (l == 0) {
    float G[9], a[3];
    const uint at[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    for (uint k = 0; k < 6; k++) { float v = 0; for (uint q = 0; q < 32; q++) v += part[k][q]; G[at[k][0] * 3 + at[k][1]] = v; G[at[k][1] * 3 + at[k][0]] = v; }
    for (uint k = 0; k < 9; k++) gram[k] = G[k];
    upgrad3(G, a);
    for (uint k = 0; k < 3; k++) alpha[k] = a[k] * p.inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = l; i < p.P; i += 1024) {
    const float d = alpha[0] * g[i] + alpha[1] * g[p.P + i] + alpha[2] * g[2 * p.P + i];
    mo[i] = 0.9f * mo[i] + 0.1f * d; vo[i] = 0.999f * vo[i] + 0.001f * d * d;
    theta[i] -= p.rate * (mo[i] / p.c1) / (sqrt(vo[i] / p.c2) + 1e-8f);
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = l; i < p.rows; i += 1024) {
    const MatW m = nw.m[rows[i].m]; const uint r = rows[i].i;
    float q = 0;
    for (uint j = 0; j < m.cols; j++) { const float v = theta[m.v + r * m.cols + j]; q = fma(v, v, q); }
    const float k = rsqrt(max(q, 1e-30f));
    for (uint j = 0; j < m.cols; j++) theta[m.v + r * m.cols + j] *= k;
  }
}

// ---- the step's gradient accumulated over microbatches, and its statistics (route_net_gradient, route_net_apply) ----
// a gradient's squared norm on the matrix units: g (NPAD floats, zero past P; NPAD a multiple of 64 32) read as
// X [NPAD / 64][64], |g|^2 = trace(X^T X), X^T X accumulated over 32-row blocks (fp32 x fp32 -> fp32); a SIMD group
typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> DevF2;
static inline float sqnorm(device const float *g, uint npad) {
  constexpr auto d = matmul2d_descriptor(64, 64, 32, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroup> op;
  DevF2 x0((device float *)g, dextents<int32_t, 2>(64, 32));
  auto y = op.get_destination_cooperative_tensor<DevF2, DevF2, float>();
  for (ushort i = 0; i < y.get_capacity(); i++) y[i] = 0;
  for (uint r = 0; r < npad / 64; r += 32) { DevF2 x((device float *)g + size_t(r) * 64, dextents<int32_t, 2>(64, 32)); op.run(x, x, y); }
  float t = 0;
  for (ushort i = 0; i < y.get_capacity(); i++) { auto at = y.get_multidimensional_index(i); if (at[0] == at[1]) t += y[i]; }
  return simd_sum(t);
}
// gm, gsum: [3][NPAD] (a task's gradient, zero past P); gsum's exchanged tail (after its 3 NPAD floats): [rows, the
// microbatches' squared norms' sum (3 tasks), their count, 0, 0, 0]. The CPU's statistics st (route_net_stats): [0..7]
// the last microbatch (its rows, its squared norms (3), the accumulator's rows), [8..16] after apply, identical at every
// rank (total rows, the mean microbatch squared norm (3), the microbatches counted, their mean rows, the rows-weighted
// mean's squared norms (3))
struct AP { uint npad, accumulate, P, pad; float rows, pad1, pad2, pad3; };
// one microbatch: gm (its gradient, as the caller's scale makes it: its rows' mean) -> gsum = (accumulate ? gsum : 0)
// + rows gm, its squared norms counted. One threadgroup of 1,024 threads
kernel void accumulate(device const float *gm [[buffer(0)]], device float *gsum [[buffer(1)]], device float *st [[buffer(2)]], constant AP &p [[buffer(3)]],
                       uint l [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float nrm[3];
  if (sg < 3) { const float n = sqnorm(gm + size_t(sg) * p.npad, p.npad); if (lane == 0) nrm[sg] = n; }
  for (uint i = l; i < 3 * p.npad; i += 1024) gsum[i] = p.accumulate ? fma(p.rows, gm[i], gsum[i]) : p.rows * gm[i];
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  device float *tail = gsum + 3 * size_t(p.npad);
  if (l == 0) {
    const bool any = p.rows > 0;
    tail[0] = (p.accumulate ? tail[0] : 0.0f) + p.rows;
    for (uint t = 0; t < 3; t++) { const float n = any ? nrm[t] : 0.0f; tail[1 + t] = (p.accumulate ? tail[1 + t] : 0.0f) + n; st[1 + t] = n; }
    tail[4] = (p.accumulate ? tail[4] : 0.0f) + (any ? 1.0f : 0.0f); tail[5] = 0; tail[6] = 0; tail[7] = 0;
    st[0] = p.rows; st[4] = tail[0];
  }
}
// after the exchange (gsum summed over the ranks): grads [3][P] = gsum / total rows, the mean's squared norms (the
// sum's times 1 / rows^2), the statistics
kernel void finish(device const float *gsum [[buffer(0)]], device float *grads [[buffer(1)]], device float *st [[buffer(2)]], constant AP &p [[buffer(3)]],
                   uint l [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  device const float *tail = gsum + 3 * size_t(p.npad);
  const float rows = tail[0], k = rows > 0 ? 1.0f / rows : 0.0f;
  for (uint i = l; i < 3 * p.P; i += 1024) grads[i] = gsum[size_t(i / p.P) * p.npad + i % p.P] * k;
  if (sg < 3) {
    const float n = sqnorm(gsum + size_t(sg) * p.npad, p.npad) * k * k;
    if (lane == 0) { st[9 + sg] = tail[4] > 0 ? tail[1 + sg] / tail[4] : 0.0f; st[14 + sg] = n; }
  }
  if (l == 0) { st[8] = rows; st[12] = tail[4]; st[13] = tail[4] > 0 ? rows / tail[4] : 0.0f; }
}
