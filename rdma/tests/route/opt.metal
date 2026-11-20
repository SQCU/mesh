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
  for (uint k = 0; k < 8; k++) { v[k] *= gain; const uint j = 8 * lane + k; if (j < m.np) Wf[wfoff[it.m] + r * m.np + j] = v[k]; }
  i8_fwht64_lanes(v, ushort(lane % 8));
  const float mx = simd_max(i8_absmax_lanes(v)), inv = i8_inverse(mx);
  if (8 * lane < m.kf) *(device uint2 *)(wq + m.wf + r * m.kf + 8 * lane) = i8_pack8(v, inv, 0, 0);
  if (lane == 0) { wsc[m.wfs + r] = i8_scale(mx); wsc[m.b + r] = live && it.m != M_OUT ? theta[m.g + m.rows + r] : 0.0f; }
}
// a SIMD group a matrix column (columns padded to 64): its dgrad codes along the rows (padded to 64), one scale
kernel void weights_cols(device int8_t *wq [[buffer(1)]], device float *wsc [[buffer(2)]], device const float *Wf [[buffer(3)]], constant Net &nw [[buffer(4)]],
                         device const Item *items [[buffer(5)]], constant uint &n [[buffer(6)]], constant uint *wfoff [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (gid / 32 >= n) return;
  const Item it = items[gid / 32]; const MatW m = nw.m[it.m]; const uint col = it.i, lane = gid % 32;
  float v[8];
  for (uint k = 0; k < 8; k++) { const uint r = 8 * lane + k; v[k] = r < m.kb ? Wf[wfoff[it.m] + r * m.np + col] : 0.0f; }
  i8_fwht64_lanes(v, ushort(lane % 8));
  const float mx = simd_max(i8_absmax_lanes(v)), inv = i8_inverse(mx);
  for (uint k = 0; k < 8; k++) { const uint r = 8 * lane + k; if (r < m.kb) wq[m.wb + r * m.np + col] = i8_code(v[k], inv, 0, 0); }
  if (lane == 0) wsc[m.wbs + col] = i8_scale(mx);
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
