#ifndef MESH_OPT_H
#define MESH_OPT_H

#include <string>

namespace mesh_simdt {

inline std::string build_opt_msl() {
  return std::string(R"MSL(
#include <metal_stdlib>
using namespace metal;

inline void opt_atomic_add_f32(device atomic_uint* addr, float val) {
  if (val == 0.0f) return;
  uint expected = atomic_load_explicit(addr, memory_order_relaxed);
  for (;;) {
    float cur = as_type<float>(expected);
    uint desired = as_type<uint>(cur + val);
    if (atomic_compare_exchange_weak_explicit(addr, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
      break;
    }
  }
}

inline float opt_tg_sum_f32(float v, uint lane, uint sg, uint n_sg, threadgroup float* sh) {
  v = simd_sum(v);
  if (lane == 0u) sh[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  if (sg == 0u) {
    float x = (lane < n_sg) ? sh[lane] : 0.0f;
    tot = simd_sum(x);
    if (lane == 0u) sh[8] = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return sh[8];
}

// 1. MG_OP_OPT_TANGENT_NORM: tangent projection + atomic global L2 norm^2
// mode=0: EDM2 unit-norm row tangent proj (1 sg per row, 4 rows/TG)
// mode=1: type_embed mp_norm row tangent proj (rows 0..2 active, row 3 zero)
// mode=2: raw scalar copy + norm^2 (1 TG of 128 threads)
[[kernel]] void mg_opt_tangent_norm(
    const device float* w_master [[buffer(0)]],
    const device float* gw [[buffer(1)]],
    device float* g_proj [[buffer(2)]],
    device atomic_uint* norm_sq [[buffer(3)]],
    constant uint* u [[buffer(4)]],   // [m, k, k_stride, mode, row_off]
    constant float* f [[buffer(5)]],  // [eps_w, eps_n]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint M = u[0], K = u[1], K_stride = u[2], mode = u[3], row_off = u[4];
  const float eps_w = max(f[0], 1e-4f);
  const float eps_n = max(f[1], 1e-4f);

  if (mode == 2u) {
    float ss = 0.0f;
    for (uint i = tid; i < K; i += 128u) {
      float g = gw[row_off + i];
      g_proj[i] = g;
      ss += g * g;
    }
    threadgroup float sh[9];
    float tot = opt_tg_sum_f32(ss, lane, sg, 4u, sh);
    if (tid == 0u) opt_atomic_add_f32(norm_sq, tot);
    return;
  }

  const uint r = tgp * 4u + sg;
  if (r >= M) return;

  if (mode == 1u) {
    if (r >= 3u) {
      for (uint c = lane; c < K; c += 32u) g_proj[ulong(r) * K + c] = 0.0f;
      return;
    }
    const device float* wr = w_master + ulong(r) * K;
    const device float* gr = gw + ulong(row_off + r) * K_stride;
    float ssw = 0.0f;
    for (uint c = lane; c < K; c += 32u) { float v = wr[c]; ssw += v * v; }
    const float inv = rsqrt(simd_sum(ssw) / float(K) + eps_n);
    float dp = 0.0f;
    for (uint c = lane; c < K; c += 32u) dp += gr[c] * (wr[c] * inv);
    const float mdot = simd_sum(dp) / float(K);
    float ssg = 0.0f;
    for (uint c = lane; c < K; c += 32u) {
      float y = wr[c] * inv;
      float gp = inv * (gr[c] - mdot * y);
      g_proj[ulong(r) * K + c] = gp;
      ssg += gp * gp;
    }
    const float g2 = simd_sum(ssg);
    if (lane == 0u) opt_atomic_add_f32(norm_sq, g2);
    return;
  }

  // mode == 0: EDM2 unit-norm row tangent projection
  const device float* wr = w_master + ulong(r) * K;
  const device float* gr = gw + ulong(row_off + r) * K_stride;
  float ssw = 0.0f;
  for (uint c = lane; c < K; c += 32u) { float v = wr[c]; ssw += v * v; }
  const float nrm = sqrt(simd_sum(ssw));
  const float inv = 1.0f / max(nrm, eps_w);
  float dp = 0.0f;
  for (uint c = lane; c < K; c += 32u) dp += gr[c] * (wr[c] * inv);
  const float proj = (nrm >= eps_w) ? simd_sum(dp) : 0.0f;
  float ssg = 0.0f;
  for (uint c = lane; c < K; c += 32u) {
    float wu = wr[c] * inv;
    float gp = (gr[c] - proj * wu) * inv;
    g_proj[ulong(r) * K + c] = gp;
    ssg += gp * gp;
  }
  const float g2 = simd_sum(ssg);
  if (lane == 0u) opt_atomic_add_f32(norm_sq, g2);
}

// 2. MG_OP_OPT_MOM_PREP: 1 TG of 256 threads per 2D matrix!
// Clips g_proj by global norm_sq, updates Nesterov momentum, optionally initializes Aurora row precond D[M],
// computes Frobenius norm, and packs zero-padded BF16 X0[r_pad, c_pad] for NAX Newton-Schulz.
[[kernel]] void mg_opt_mom_prep(
    const device float* g_proj [[buffer(0)]],
    const device float* norm_sq [[buffer(1)]],
    const device float* opt_cfg [[buffer(2)]],
    device float* d_vec [[buffer(3)]],        // optional [M] when use_aurora_d
    device float* mom [[buffer(4)]],          // [M, K] in-place
    device float* g_eff [[buffer(5)]],        // [M, K]
    device bfloat* x_bf16 [[buffer(6)]],      // [r_pad, c_pad]
    constant uint* u [[buffer(7)]],           // [M, K, r_pad, c_pad, flags(bit0=transposed, bit1=use_aurora_d)]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = u[0], K = u[1], r_pad = u[2], c_pad = u[3], flags = u[4];
  const bool transposed = (flags & 1u) != 0u;
  const bool use_d = (flags & 2u) != 0u;
  const float mu = opt_cfg[2];
  const bool nesterov = opt_cfg[3] > 0.5f;
  const float grad_clip = opt_cfg[4];
  const float pp_beta = opt_cfg[5];
  const float eps_w = max(opt_cfg[6], 1e-4f);
  const float d_lo = opt_cfg[7], d_hi = opt_cfg[8];

  const float g_norm = sqrt(max(norm_sq[0], 0.0f));
  const float clip_sc = (grad_clip > 0.0f && g_norm > grad_clip) ? (grad_clip / max(g_norm, 1e-6f)) : 1.0f;

  const uint tot = M * K;
  for (uint idx = tid; idx < tot; idx += 256u) {
    float g = g_proj[idx] * clip_sc;
    float mv = mu * mom[idx] + g;
    mom[idx] = mv;
    g_eff[idx] = nesterov ? (g + mu * mv) : mv;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  threadgroup float tg_red[9];
  threadgroup float d_sh[1024];

  if (use_d) {
    float rn_sum = 0.0f;
    for (uint i = tid; i < M; i += 256u) {
      float s = 0.0f;
      const device float* gr = g_eff + ulong(i) * K;
      for (uint j = 0; j < K; j++) { float v = gr[j]; s += v * v; }
      float rn = max(sqrt(s), eps_w);
      d_sh[i] = rn;
      rn_sum += rn;
    }
    const float mean_rn = opt_tg_sum_f32(rn_sum, lane, sg, 8u, tg_red) / float(max(1u, M));
    for (uint i = tid; i < M; i += 256u) {
      float dv = clamp(pow(mean_rn / d_sh[i], pp_beta), d_lo, d_hi);
      d_sh[i] = dv;
      d_vec[i] = dv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float frob2 = 0.0f;
    for (uint idx = tid; idx < tot; idx += 256u) {
      uint i = idx / K;
      float v = d_sh[i] * g_eff[idx];
      frob2 += v * v;
    }
    const float inv_f = 1.0f / (sqrt(opt_tg_sum_f32(frob2, lane, sg, 8u, tg_red)) * 1.02f + eps_w);
    const uint xtot = r_pad * c_pad;
    for (uint idx = tid; idx < xtot; idx += 256u) {
      uint rr = idx / c_pad, cc = idx % c_pad;
      float v = (rr < K && cc < M) ? (d_sh[cc] * g_eff[ulong(cc) * K + rr] * inv_f) : 0.0f;
      x_bf16[idx] = bfloat(v);
    }
  } else {
    float frob2 = 0.0f;
    for (uint idx = tid; idx < tot; idx += 256u) {
      float v = g_eff[idx];
      frob2 += v * v;
    }
    const float inv_f = 1.0f / (sqrt(opt_tg_sum_f32(frob2, lane, sg, 8u, tg_red)) * 1.02f + eps_w);
    const uint xtot = r_pad * c_pad;
    for (uint idx = tid; idx < xtot; idx += 256u) {
      uint rr = idx / c_pad, cc = idx % c_pad;
      float v = 0.0f;
      if (transposed) {
        if (rr < K && cc < M) v = g_eff[ulong(cc) * K + rr] * inv_f;
      } else {
        if (rr < M && cc < K) v = g_eff[ulong(rr) * K + cc] * inv_f;
      }
      x_bf16[idx] = bfloat(v);
    }
  }
}

// 3. MG_OP_OPT_NS_AXPBY: out[i] = bfloat(f0 * float(a[i]) + f1 * float(b[i]))
[[kernel]] void mg_opt_ns_axpby(
    const device bfloat* a [[buffer(0)]],
    const device bfloat* b [[buffer(1)]],
    device bfloat* y [[buffer(2)]],
    constant uint* u [[buffer(3)]],   // [n]
    constant float* f [[buffer(4)]],  // [alpha, beta]
    uint gid [[thread_position_in_grid]]) {
  const uint n = u[0];
  const float alpha = f[0], beta = f[1];
  const uint base = gid * 4u;
  for (uint k = 0; k < 4u; k++) {
    uint i = base + k;
    if (i < n) y[i] = bfloat(alpha * float(a[i]) + beta * float(b[i]));
  }
}

// 4. MG_OP_OPT_AURORA_DREF: 1 TG of 256 threads!
// Refines tall-matrix row preconditioner D[M] from current X_bf16 (= P^T [r_pad, c_pad]) and re-packs X0.
[[kernel]] void mg_opt_aurora_dref(
    device bfloat* x_bf16 [[buffer(0)]],      // [r_pad, c_pad] in-place (transposed: r=K, c=M)
    const device float* g_eff [[buffer(1)]],  // [M, K]
    const device float* opt_cfg [[buffer(2)]],
    device float* d_vec [[buffer(3)]],        // [M] in-place
    constant uint* u [[buffer(4)]],           // [M, K, r_pad, c_pad]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = u[0], K = u[1], r_pad = u[2], c_pad = u[3];
  const float pp_beta = opt_cfg[5];
  const float eps_w = max(opt_cfg[6], 1e-4f);
  const float d_lo = opt_cfg[9], d_hi = opt_cfg[10];
  const float target_lev = float(K) / float(max(1u, M));

  threadgroup float tg_red[9];
  threadgroup float d_sh[1024];

  float d_sum = 0.0f;
  for (uint i = tid; i < M; i += 256u) {
    float rsq = 0.0f;
    for (uint j = 0; j < K; j++) {
      float p = float(x_bf16[ulong(j) * c_pad + i]);
      rsq += p * p;
    }
    rsq = max(rsq, eps_w);
    float dn = clamp(d_vec[i] * pow(target_lev / rsq, pp_beta), d_lo, d_hi);
    d_sh[i] = dn;
    d_sum += dn;
  }
  const float mean_d = max(opt_tg_sum_f32(d_sum, lane, sg, 8u, tg_red) / float(max(1u, M)), eps_w);
  for (uint i = tid; i < M; i += 256u) {
    float dn = d_sh[i] / mean_d;
    d_sh[i] = dn;
    d_vec[i] = dn;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint tot = M * K;
  float frob2 = 0.0f;
  for (uint idx = tid; idx < tot; idx += 256u) {
    uint i = idx / K;
    float v = d_sh[i] * g_eff[idx];
    frob2 += v * v;
  }
  const float inv_f = 1.0f / (sqrt(opt_tg_sum_f32(frob2, lane, sg, 8u, tg_red)) * 1.02f + eps_w);
  const uint xtot = r_pad * c_pad;
  for (uint idx = tid; idx < xtot; idx += 256u) {
    uint rr = idx / c_pad, cc = idx % c_pad;
    float v = (rr < K && cc < M) ? (d_sh[cc] * g_eff[ulong(cc) * K + rr] * inv_f) : 0.0f;
    x_bf16[idx] = bfloat(v);
  }
}

// 5. MG_OP_OPT_APPLY_2D: apply polar update + unit-norm + EMA + live BF16 pack + zero gw!
// 1 simdgroup = 32 threads per row, 4 rows/TG.
[[kernel]] void mg_opt_apply_2d(
    const device bfloat* x_bf16 [[buffer(0)]],
    const device float* opt_cfg [[buffer(1)]],
    device float* gw [[buffer(2)]],           // [m_live, k_pad] zeroed for [row_off..row_off+M-1]
    device float* w_master [[buffer(3)]],     // [M, K]
    device float* w_ema [[buffer(4)]],        // [M, K]
    device bfloat* w_live [[buffer(5)]],      // [m_live, k_pad]
    constant uint* u [[buffer(6)]],           // [M, K, r_pad, c_pad, k_pad, row_off, flags(bit0=trans, bit1=unit, bit2=type_emb)]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint M = u[0], K = u[1], c_pad = u[3], k_pad = u[4], row_off = u[5], flags = u[6];
  const uint i = tgp * 4u + sg;
  if (i >= M) return;

  const bool transposed = (flags & 1u) != 0u;
  const bool do_unit = (flags & 2u) != 0u;
  const bool is_te = (flags & 4u) != 0u;
  const float lr = opt_cfg[0];
  const float ema_decay = opt_cfg[11];
  const float eps_w = max(opt_cfg[12], 1e-6f);
  const float eps_n = max(opt_cfg[13], 1e-4f);
  const float step_sc = lr * sqrt(float(M) / float(max(1u, min(M, K))));

  device float* wm = w_master + ulong(i) * K;
  device float* we = w_ema + ulong(i) * K;
  float ss = 0.0f;
  for (uint j = lane; j < K; j += 32u) {
    float pv = transposed ? float(x_bf16[ulong(j) * c_pad + i]) : float(x_bf16[ulong(i) * c_pad + j]);
    float wn = wm[j] - step_sc * pv;
    wm[j] = wn;
    ss += wn * wn;
  }
  const float ss_tot = simd_sum(ss);
  const float inv_u = do_unit ? (1.0f / max(sqrt(ss_tot), eps_w)) : 1.0f;
  const float inv_te = is_te ? rsqrt(ss_tot / float(K) + eps_n) : 1.0f;

  device bfloat* wl = w_live + ulong(row_off + i) * k_pad;
  device float* gwr = gw + ulong(row_off + i) * k_pad;
  for (uint j = lane; j < K; j += 32u) {
    float wn = wm[j] * inv_u;
    if (do_unit) wm[j] = wn;
    we[j] = ema_decay * we[j] + (1.0f - ema_decay) * wn;
    if (!is_te) {
      wl[j] = bfloat(wn);
    } else if (i < 3u) {
      wl[j] = bfloat(wn * inv_te);
    }
    if (!is_te || i < 3u) gwr[j] = 0.0f;
  }
  for (uint j = K + lane; j < k_pad; j += 32u) {
    if (!is_te || i < 3u) {
      wl[j] = bfloat(0.0f);
      gwr[j] = 0.0f;
    }
  }
}

// 6. MG_OP_OPT_APPLY_1D: Adam for 1D UnitLinear (mode=0) or PlanWally scalar bank (mode=1)
[[kernel]] void mg_opt_apply_1d_unit(
    const device float* g_proj [[buffer(0)]],
    const device float* norm_sq [[buffer(1)]],
    const device float* opt_cfg [[buffer(2)]],
    const device uint* step_buf [[buffer(3)]],
    device float* mom [[buffer(4)]],
    device float* v_buf [[buffer(5)]],
    device float* w_master [[buffer(6)]],
    device float* w_ema [[buffer(7)]],
    device bfloat* w_live [[buffer(8)]],
    device float* gw [[buffer(9)]],
    constant uint* u [[buffer(10)]], // [K, k_pad, row_off]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint K = u[0], k_pad = u[1], row_off = u[2];
  const float slr = opt_cfg[1];
  const float mu = opt_cfg[2];
  const float grad_clip = opt_cfg[4];
  const float ema_decay = opt_cfg[11];
  const float eps_w = max(opt_cfg[12], 1e-6f);
  const float beta2 = opt_cfg[14];
  const float seps = opt_cfg[15];
  const float step = float(step_buf[0] + 1u);
  const float bc1 = 1.0f / max(1.0f - pow(mu, step), 1e-7f);
  const float bc2 = 1.0f / max(1.0f - pow(beta2, step), 1e-7f);

  const float g_norm = sqrt(max(norm_sq[0], 0.0f));
  const float clip_sc = (grad_clip > 0.0f && g_norm > grad_clip) ? (grad_clip / max(g_norm, 1e-6f)) : 1.0f;

  threadgroup float tg_red[9];
  float ss = 0.0f;
  for (uint j = tid; j < K; j += 128u) {
    float g = g_proj[j] * clip_sc;
    float m = mu * mom[j] + (1.0f - mu) * g;
    float v = beta2 * v_buf[j] + (1.0f - beta2) * g * g;
    mom[j] = m;
    v_buf[j] = v;
    float wn = w_master[j] - slr * (m * bc1) / (sqrt(v * bc2) + seps);
    w_master[j] = wn;
    ss += wn * wn;
  }
  const float inv_u = 1.0f / max(sqrt(opt_tg_sum_f32(ss, lane, sg, 4u, tg_red)), eps_w);
  device bfloat* wl = w_live + ulong(row_off) * k_pad;
  device float* gwr = gw + ulong(row_off) * k_pad;
  for (uint j = tid; j < K; j += 128u) {
    float wn = w_master[j] * inv_u;
    w_master[j] = wn;
    w_ema[j] = ema_decay * w_ema[j] + (1.0f - ema_decay) * wn;
    wl[j] = bfloat(wn);
    gwr[j] = 0.0f;
  }
  for (uint j = K + tid; j < k_pad; j += 128u) {
    wl[j] = bfloat(0.0f);
    gwr[j] = 0.0f;
  }
}

[[kernel]] void mg_opt_apply_scalars(
    device float* sc_grads [[buffer(0)]],     // [S_tot] zeroed at end!
    device float* norm_sq [[buffer(1)]],      // [1] zeroed at end!
    const device float* opt_cfg [[buffer(2)]],
    device uint* step_buf [[buffer(3)]],      // [1] incremented at end!
    device float* mom [[buffer(4)]],          // [S_tot]
    device float* v_buf [[buffer(5)]],        // [S_tot]
    device float* sc_master [[buffer(6)]],    // [S_tot]
    device float* sc_ema [[buffer(7)]],       // [S_tot]
    device float* scalars_live [[buffer(8)]],
    device float* hp_live [[buffer(9)]],
    const device float* bounds [[buffer(10)]], // [temp_lo, temp_hi, cg_b, ls_lo, ls_hi, vg_lo, vg_hi, dg_lo, dg_hi]
    device float* diag_opt [[buffer(11)]],     // [2]: grad_norm, mean_cont_std
    constant uint* u [[buffer(12)]],           // [S_tot, n_uniq_blk, depth, recurrent]
    uint tid [[thread_index_in_threadgroup]]) {
  const uint S_tot = u[0], n_uniq_blk = u[1], depth = u[2], recurrent = u[3];
  const float slr = opt_cfg[1];
  const float mu = opt_cfg[2];
  const float grad_clip = opt_cfg[4];
  const float ema_decay = opt_cfg[11];
  const float beta2 = opt_cfg[14];
  const float seps = opt_cfg[15];
  const uint step_next = step_buf[0] + 1u;
  const float step_f = float(step_next);
  const float bc1 = 1.0f / max(1.0f - pow(mu, step_f), 1e-7f);
  const float bc2 = 1.0f / max(1.0f - pow(beta2, step_f), 1e-7f);

  const float g_norm = sqrt(max(norm_sq[0], 0.0f));
  const float clip_sc = (grad_clip > 0.0f && g_norm > grad_clip) ? (grad_clip / max(g_norm, 1e-6f)) : 1.0f;

  if (tid < S_tot) {
    float g = sc_grads[tid] * clip_sc;
    float m = mu * mom[tid] + (1.0f - mu) * g;
    float v = beta2 * v_buf[tid] + (1.0f - beta2) * g * g;
    mom[tid] = m;
    v_buf[tid] = v;
    float wn = sc_master[tid] - slr * (m * bc1) / (sqrt(v * bc2) + seps);
    sc_master[tid] = wn;
    sc_ema[tid] = ema_decay * sc_ema[tid] + (1.0f - ema_decay) * wn;
    sc_grads[tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  if (tid == 0u) {
    const float t_lo = bounds[0], t_hi = bounds[1], cg_b = bounds[2];
    const float ls_lo = bounds[3], ls_hi = bounds[4];
    const float vg_lo = bounds[5], vg_hi = bounds[6];
    const float dg_lo = bounds[7], dg_hi = bounds[8];

    // Pack hp_live[0..37]
    float std_sum = 0.0f;
    for (uint i = 0; i < 11u; i++) {
      hp_live[i] = clamp(sc_master[i], -cg_b, cg_b);
      hp_live[11u + i] = sc_master[11u + i];
      float ls = clamp(sc_master[22u + i], ls_lo, ls_hi);
      hp_live[22u + i] = ls;
      std_sum += exp(ls);
    }
    hp_live[33] = clamp(sc_master[33], vg_lo, vg_hi);
    hp_live[34] = sc_master[34];
    hp_live[35] = sc_master[35];
    hp_live[36] = clamp(sc_master[36], dg_lo, dg_hi);
    hp_live[37] = clamp(sc_master[37], dg_lo, dg_hi);

    // Pack scalars_live: [0]=sigmoid(type_blend), [1]=clamp(exp(log_temp_cart)), [2]=clamp(exp(log_temp_rival)), ...
    scalars_live[0] = 1.0f / (1.0f + exp(-sc_master[40]));
    scalars_live[1] = clamp(exp(sc_master[38]), t_lo, t_hi);
    scalars_live[2] = clamp(exp(sc_master[39]), t_lo, t_hi);
    for (uint i = 0; i < 4u * n_uniq_blk; i++) {
      scalars_live[3u + i] = 1.0f / (1.0f + exp(-sc_master[41u + i]));
    }
    if (recurrent != 0u) {
      for (uint s = 0; s < depth; s++) {
        scalars_live[3u + 4u * n_uniq_blk + s] = 1.0f / (1.0f + exp(-sc_master[41u + 4u * n_uniq_blk + s]));
      }
    }
    diag_opt[0] = g_norm;
    diag_opt[1] = std_sum / 11.0f;
    norm_sq[0] = 0.0f;
    step_buf[0] = step_next;
  }
}
)MSL");
}

} // namespace mesh_simdt

#endif
