#ifndef MESH_SIMDT_BWD_H
#define MESH_SIMDT_BWD_H

#include <string>

namespace mesh_simdt {

inline std::string build_simdt_bwd_msl() {
  return std::string(R"MSL(
#include <metal_stdlib>
using namespace metal;

inline void atomic_add_f32(device atomic_float* addr, float val) {
  atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

inline float tg_sum_bwd(float v, uint lane, uint sg, uint n_sg, threadgroup float* scratch) {
  v = simd_sum(v);
  if (lane == 0u) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float x = (lane < n_sg) ? scratch[lane] : 0.0f;
    float tot = simd_sum(x);
    if (lane == 0u) scratch[8] = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float res = scratch[8];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return res;
}

inline float huber_val(float diff, float beta) {
  const float ad = fabs(diff);
  return (ad < beta) ? (0.5f * diff * diff / beta) : (ad - 0.5f * beta);
}

inline float huber_grad(float diff, float beta) {
  const float ad = fabs(diff);
  if (ad < beta) return diff / beta;
  return (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
}

// ---- 1. MG_OP_PLAN_LOSS_BWD ----
// Computes exact PPO clipped surrogate + discrete entropy + Winnie/Lou Huber + Dynamics Huber
// and writes d_heads_raw[E*N, 64] (BF16), d_cart_lg[E*N, J] (BF16), d_rival_lg[E*N, K] (BF16),
// plus atomic FP32 scalar grads & loss diagnostics!
//
// hp layout (48 floats):
//   [0..10]: cont_gain (clamped), [11..21]: cont_bias, [22..32]: cont_logstd (clamped),
//   [33]: val_gain (clamped), [34..35]: val_bias, [36]: dyn_drift_gain, [37]: dyn_act_gain,
//   [38]: cont_mean_scale, [39..41]: pack cfg,
//   [42]: cont_gain_bound, [43..44]: logstd_lo/hi, [45..46]: vg_lo/hi, [47]: dyn_lo (packed via lcfg)
// lcfg layout (16 floats):
//   [0]: actor_w, [1]: critic_w, [2]: dyn_w, [3]: ent_coef, [4]: ratio_clip, [5]: log_ratio_bound,
//   [6]: critic_beta, [7]: dyn_beta, [8]: inv_am_sum, [9]: inv_wm_sum, [10]: inv_lm_sum, [11]: inv_en,
//   [12]: grad_scale (1/accum), [13]: vg_hi, [14]: dyn_lo, [15]: dyn_hi
// scalar_grads layout (64 atomic_float):
//   [0..10]: d_cont_gain, [11..21]: d_cont_bias, [22..32]: d_cont_logstd,
//   [33]: d_val_gain, [34..35]: d_val_bias, [36]: d_dyn_drift_gain, [37]: d_dyn_act_gain,
//   [38]: d_log_temp_cart, [39]: d_log_temp_rival, [40]: d_type_blend,
//   [41..44]: d_blend_pool/mut/gram/ffn (blk 0..), [45..52]: d_step_blends / extra blks
// diag_out layout (8 atomic_float):
//   [0]: loss_actor, [1]: loss_w, [2]: loss_l, [3]: loss_dyn, [4]: ent_disc, [5]: ent_all, [6]: clip_frac, [7]: count
[[kernel]] void mg_plan_loss_bwd(
    const device bfloat* heads_raw [[buffer(0)]],   // [E*N, 64]
    const device bfloat* cart_lg [[buffer(1)]],     // [E*N, J]
    const device bfloat* rival_lg [[buffer(2)]],    // [E*N, K]
    const device float* hp [[buffer(3)]],           // [48]
    const device float* targ_f [[buffer(4)]],       // [E*N, 17]: cont[11], beh_lp, adv, ret, obs_drho, am, wm
    const device int* targ_i [[buffer(5)]],         // [E*N, 2]: cart_pick, rival_pick
    const device float* lcfg [[buffer(6)]],         // [16]
    device bfloat* d_heads_raw [[buffer(7)]],       // [E*N, 64]
    device bfloat* d_cart_lg [[buffer(8)]],         // [E*N, J]
    device bfloat* d_rival_lg [[buffer(9)]],        // [E*N, K]
    device atomic_float* sc_grads [[buffer(10)]],   // [64]
    device atomic_float* diag_out [[buffer(11)]],   // [8]
    constant uint* u [[buffer(12)]],                // [rows(E*N), J, K, rows_pad]
    uint row [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = u[0], J = u[1], K = u[2], rows_pad = u[3];
  const bool active_row = (row < rows);

  // Zero trailing padded rows of d_heads_raw so NAX dX/dW sees exact zeros
  if (!active_row) {
    if (row < rows_pad) {
      device bfloat* dh = d_heads_raw + ulong(row) * 64ul;
      for (uint c = 0; c < 64u; c++) dh[c] = bfloat(0.0f);
    }
    return;
  }

  const float actor_w = lcfg[0], critic_w = lcfg[1], dyn_w = lcfg[2], ent_coef = lcfg[3];
  const float r_clip = lcfg[4], lr_bound = lcfg[5], c_beta = lcfg[6], d_beta = lcfg[7];
  const float inv_am = lcfg[8], inv_wm = lcfg[9], inv_lm = lcfg[10], inv_en = lcfg[11];
  const float g_sc = lcfg[12];
  const float cm_sc = hp[38];
  const float cg_bnd = hp[42];
  const float ls_lo = hp[43], ls_hi = hp[44];
  const float vg_lo = hp[45], vg_hi = lcfg[13];
  const float dg_lo = lcfg[14], dg_hi = lcfg[15];

  const device bfloat* hr = heads_raw + ulong(row) * 64ul;
  const device float* tf = targ_f + ulong(row) * 17ul;
  const int cp = targ_i[ulong(row) * 2ul + 0ul];
  const int rp = targ_i[ulong(row) * 2ul + 1ul];

  const float beh_lp = tf[11], adv = tf[12], ret = tf[13], obs_drho = tf[14];
  const float am = tf[15], wm = tf[16];
  if (wm < -0.5f) {
    device bfloat* dcl0 = d_cart_lg + ulong(row) * J;
    for (uint j = 0; j < J; j++) dcl0[j] = bfloat(0.0f);
    device bfloat* drl0 = d_rival_lg + ulong(row) * K;
    for (uint k = 0; k < K; k++) drl0[k] = bfloat(0.0f);
    device bfloat* dh0 = d_heads_raw + ulong(row) * 64ul;
    for (uint c = 0; c < 64u; c++) dh0[c] = bfloat(0.0f);
    return;
  }
  const float lm = 1.0f - wm;

  // 1. Continuous Gaussian logp
  float raw_c[11], th_c[11], err_c[11], inv_std_c[11];
  float lp_cont = 0.0f;
  float h_cont = 0.0f;
  for (uint i = 0; i < 11u; i++) {
    raw_c[i] = float(hr[i]);
    const float u_v = hp[i] * raw_c[i] + hp[11u + i];
    const float th = tanh(clamp(u_v, -20.0f, 20.0f));
    th_c[i] = th;
    const float mu = float(bfloat(cm_sc * th));
    const float ls = hp[22u + i];
    const float istd = exp(-ls);
    inv_std_c[i] = istd;
    const float er = (tf[i] - mu) * istd;
    err_c[i] = er;
    lp_cont += -0.5f * (er * er + 2.0f * ls + 1.83787706641f);
    h_cont += ls + 1.41893853320f;
  }

  // 2. Cart categorical log_softmax + entropy
  const device bfloat* cl = cart_lg + ulong(row) * J;
  float max_c = -1e30f;
  for (uint j = 0; j < J; j++) max_c = max(max_c, float(cl[j]));
  float sum_ec = 0.0f;
  for (uint j = 0; j < J; j++) sum_ec += exp(float(cl[j]) - max_c);
  const float lse_c = max_c + log(max(sum_ec, 1e-30f));
  float h_cart = 0.0f;
  for (uint j = 0; j < J; j++) {
    const float lsm = float(cl[j]) - lse_c;
    const float pj = exp(lsm);
    h_cart -= pj * lsm;
  }
  const float lp_cart = float(cl[uint(clamp(cp, 0, int(J) - 1))]) - lse_c;

  // 3. Rival categorical log_softmax + entropy
  const device bfloat* rl = rival_lg + ulong(row) * K;
  float max_r = -1e30f;
  for (uint k = 0; k < K; k++) max_r = max(max_r, float(rl[k]));
  float sum_er = 0.0f;
  for (uint k = 0; k < K; k++) sum_er += exp(float(rl[k]) - max_r);
  const float lse_r = max_r + log(max(sum_er, 1e-30f));
  float h_riv = 0.0f;
  for (uint k = 0; k < K; k++) {
    const float lsm = float(rl[k]) - lse_r;
    const float pk = exp(lsm);
    h_riv -= pk * lsm;
  }
  const float lp_riv = float(rl[uint(clamp(rp, 0, int(K) - 1))]) - lse_r;

  // 4. PPO clipped surrogate + entropy
  const float logp = lp_cont + lp_cart + lp_riv;
  const float lr_raw = logp - beh_lp;
  const float lr = clamp(lr_raw, -lr_bound, lr_bound);
  const float clip_hi = log(1.0f + r_clip);
  const float clip_lo = log(1.0f - r_clip);
  const float sel = (adv >= 0.0f) ? min(lr, clip_hi) : max(lr, clip_lo);
  const float ratio_sel = exp(sel);
  const float surr = ratio_sel * adv;
  const float h_disc = h_cart + h_riv;

  const float act_loss_r = actor_w * (-(surr * am) * inv_am - ent_coef * h_disc * am * inv_am);
  const float clip_r = ((fabs(lr) > clip_hi) ? 1.0f : 0.0f) * am * inv_am;
  const float kl_r = ((exp(lr) - 1.0f) - lr) * am * inv_am;

  const bool in_lr = (lr_raw > -lr_bound) && (lr_raw < lr_bound);
  const bool not_clipped = (adv >= 0.0f) ? (lr < clip_hi) : (lr > clip_lo);
  const float d_logp = (in_lr && not_clipped) ? (-actor_w * am * inv_am * ratio_sel * adv * g_sc) : 0.0f;
  const float d_ent_coeff = actor_w * ent_coef * am * inv_am * g_sc;

  // Write d_cart_lg & d_rival_lg
  device bfloat* dcl = d_cart_lg + ulong(row) * J;
  for (uint j = 0; j < J; j++) {
    const float lsm = float(cl[j]) - lse_c;
    const float pj = exp(lsm);
    const float d_lp = (int(j) == cp ? 1.0f : 0.0f) - pj;
    const float d_h = pj * (lsm + h_cart);
    dcl[j] = bfloat(d_logp * d_lp + d_ent_coeff * d_h);
  }
  device bfloat* drl = d_rival_lg + ulong(row) * K;
  for (uint k = 0; k < K; k++) {
    const float lsm = float(rl[k]) - lse_r;
    const float pk = exp(lsm);
    const float d_lp = (int(k) == rp ? 1.0f : 0.0f) - pk;
    const float d_h = pk * (lsm + h_riv);
    drl[k] = bfloat(d_logp * d_lp + d_ent_coeff * d_h);
  }

  // Continuous head gradients -> d_heads_raw[0..10]
  device bfloat* dh = d_heads_raw + ulong(row) * 64ul;
  for (uint i = 0; i < 11u; i++) {
    const float d_mu = d_logp * err_c[i] * inv_std_c[i];
    const float d_u = d_mu * cm_sc * (1.0f - th_c[i] * th_c[i]);
    dh[i] = bfloat(d_u * hp[i]);
    const float d_cg = (fabs(hp[i]) < cg_bnd - 1e-6f) ? (d_u * raw_c[i]) : 0.0f;
    const float d_ls = (hp[22u + i] > ls_lo + 1e-6f && hp[22u + i] < ls_hi - 1e-6f)
                         ? (d_logp * (err_c[i] * err_c[i] - 1.0f))
                         : 0.0f;
    const float s_cg = simd_sum(d_cg);
    const float s_cb = simd_sum(d_u);
    const float s_ls = simd_sum(d_ls);
    if (lane == 0u) {
      atomic_add_f32(&sc_grads[i], s_cg);
      atomic_add_f32(&sc_grads[11u + i], s_cb);
      atomic_add_f32(&sc_grads[22u + i], s_ls);
    }
  }

  // 5. Winnie/Lou Critic Huber loss
  const float vg = hp[33];
  const float r11 = float(hr[11]), r12 = float(hr[12]);
  const float vw = vg * r11 + hp[34];
  const float vl = vg * r12 + hp[35];
  const float w_loss_r = critic_w * huber_val(vw - ret, c_beta) * wm * inv_wm;
  const float l_loss_r = critic_w * huber_val(vl - ret, c_beta) * lm * inv_lm;
  const float dvw = critic_w * huber_grad(vw - ret, c_beta) * wm * inv_wm * g_sc;
  const float dvl = critic_w * huber_grad(vl - ret, c_beta) * lm * inv_lm * g_sc;
  dh[11] = bfloat(dvw * vg);
  dh[12] = bfloat(dvl * vg);
  const float d_vg = (vg > vg_lo + 1e-6f && vg < vg_hi - 1e-6f) ? (dvw * r11 + dvl * r12) : 0.0f;

  // 6. Dynamics Huber loss
  const float ddg = hp[36], dag = hp[37];
  const float r13 = float(hr[13]);
  float dot_da = 0.0f;
  for (uint i = 0; i < 11u; i++) dot_da += float(hr[14u + i]) * tf[i];
  const float pred_drho = ddg * r13 + dag * dot_da;
  const float dyn_loss_r = dyn_w * huber_val(pred_drho - obs_drho, d_beta) * inv_en;
  const float dd = dyn_w * huber_grad(pred_drho - obs_drho, d_beta) * inv_en * g_sc;
  dh[13] = bfloat(dd * ddg);
  for (uint i = 0; i < 11u; i++) dh[14u + i] = bfloat(dd * dag * tf[i]);
  for (uint c = 25u; c < 64u; c++) dh[c] = bfloat(0.0f);

  const float d_ddg = (ddg > dg_lo + 1e-6f && ddg < dg_hi - 1e-6f) ? (dd * r13) : 0.0f;
  const float d_dag = (dag > dg_lo + 1e-6f && dag < dg_hi - 1e-6f) ? (dd * dot_da) : 0.0f;

  // Simd-reduce scalar grads & loss diagnostics before atomic add (32x less contention!)
  const float s_vg  = simd_sum(d_vg),  s_vb0 = simd_sum(dvw), s_vb1 = simd_sum(dvl);
  const float s_ddg = simd_sum(d_ddg), s_dag = simd_sum(d_dag);
  const float s_la  = simd_sum(act_loss_r), s_lw = simd_sum(w_loss_r), s_ll = simd_sum(l_loss_r), s_ld = simd_sum(dyn_loss_r);
  const float s_ed  = simd_sum(h_disc * am * inv_am), s_ea = simd_sum((h_cont + h_disc) * am * inv_am), s_cf = simd_sum(clip_r), s_kl = simd_sum(kl_r);
  if (lane == 0u) {
    atomic_add_f32(&sc_grads[33], s_vg);
    atomic_add_f32(&sc_grads[34], s_vb0);
    atomic_add_f32(&sc_grads[35], s_vb1);
    atomic_add_f32(&sc_grads[36], s_ddg);
    atomic_add_f32(&sc_grads[37], s_dag);
    atomic_add_f32(&diag_out[0], s_la);
    atomic_add_f32(&diag_out[1], s_lw);
    atomic_add_f32(&diag_out[2], s_ll);
    atomic_add_f32(&diag_out[3], s_ld);
    atomic_add_f32(&diag_out[4], s_ed);
    atomic_add_f32(&diag_out[5], s_ea);
    atomic_add_f32(&diag_out[6], s_cf);
    atomic_add_f32(&diag_out[7], s_kl);
  }
}

// ---- 2. MG_OP_BILINEAR_LOGITS_BWD (Q & K + log_temp) ----
// 2a: dQ [E*N, D] + d_log_temp (1 simdgroup = 32 threads per (e, n), 4 rows/TG)
[[kernel]] void mg_bilinear_logits_bwd_q(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* Kmat [[buffer(1)]],
    const device bfloat* d_lg [[buffer(2)]],
    const device int* ent_meta [[buffer(3)]],
    const device float* scalars [[buffer(4)]],
    device bfloat* dQ [[buffer(5)]],
    device atomic_float* sc_grads [[buffer(6)]],
    constant uint* u [[buffer(7)]],  // [E, N, C, D, sc_idx_temp, mask_own, M_ent, sc_grad_idx]
    constant float* f [[buffer(8)]], // [unit_eps, temp_lo, temp_hi, rows_pad]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], C = u[2], D = u[3], sc_idx = u[4], mask_own = u[5], M_ent = u[6], sg_idx = u[7];
  const uint row = tgp * 4u + sg;
  const uint rows = E * N;
  const uint rows_pad = uint(f[3]);
  if (row >= rows) {
    if (row < rows_pad) {
      device bfloat* dq = dQ + ulong(row) * D;
      for (uint d = lane; d < D; d += 32u) dq[d] = bfloat(0.0f);
    }
    return;
  }
  const uint e = row / N, n = row % N;
  const float eps = max(f[0], 1e-4f);
  const float temp = scalars[sc_idx];
  const bool t_unclamped = (temp > f[1] + 1e-5f) && (temp < f[2] - 1e-5f);

  const device bfloat* qr = Q + ulong(row) * D;
  float rq[8], duq[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ssq = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint d = lane + i * 32u;
    float v = (d < D) ? float(qr[d]) : 0.0f;
    rq[i] = v;
    duq[i] = 0.0f;
    ssq += v * v;
  }
  const float nq_raw = sqrt(simd_sum(ssq));
  const float nq = max(nq_raw, eps);
  const float inv_q = 1.0f / nq;
  for (uint i = 0; i < n_el; i++) rq[i] *= inv_q; // now unit(Q)

  const int my_team = mask_own ? ent_meta[ulong(e) * M_ent + n] : -999999;
  const device int* tids = ent_meta + ulong(E) * M_ent + ulong(e) * C;

  float d_lt = 0.0f;
  for (uint c = 0; c < C; c++) {
    if (mask_own && C >= 2u && my_team == tids[c]) continue;
    const float g_l = float(d_lg[ulong(row) * C + c]);
    const device bfloat* kr = Kmat + (ulong(e) * C + c) * D;
    float ssk = 0.0f, dp = 0.0f;
    float rk[8];
    for (uint i = 0; i < n_el; i++) {
      const uint d = lane + i * 32u;
      float kv = (d < D) ? float(kr[d]) : 0.0f;
      rk[i] = kv;
      ssk += kv * kv;
      dp += rq[i] * kv;
    }
    const float inv_k = 1.0f / max(sqrt(simd_sum(ssk)), eps);
    const float cos_v = simd_sum(dp) * inv_k;
    d_lt += g_l * temp * cos_v;
    const float w = temp * g_l * inv_k;
    for (uint i = 0; i < n_el; i++) duq[i] += w * rk[i];
  }

  float dot_qu = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_qu += duq[i] * rq[i];
  const float proj = (nq_raw >= eps) ? simd_sum(dot_qu) : 0.0f;
  device bfloat* dqr = dQ + ulong(row) * D;
  for (uint i = 0; i < n_el; i++) {
    const uint d = lane + i * 32u;
    if (d < D) dqr[d] = bfloat((duq[i] - proj * rq[i]) * inv_q);
  }
  if (lane == 0u && t_unclamped && d_lt != 0.0f) {
    atomic_add_f32(&sc_grads[sg_idx], d_lt);
  }
}

// 2b: dK [E*C, D] (1 simdgroup = 32 threads per (e, c), 4 rows/TG)
[[kernel]] void mg_bilinear_logits_bwd_k(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* Kmat [[buffer(1)]],
    const device bfloat* d_lg [[buffer(2)]],
    const device int* ent_meta [[buffer(3)]],
    const device float* scalars [[buffer(4)]],
    device bfloat* dK [[buffer(5)]],
    constant uint* u [[buffer(6)]],  // [E, N, C, D, sc_idx_temp, mask_own, M_ent, rows_pad]
    constant float* f [[buffer(7)]], // [unit_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], C = u[2], D = u[3], sc_idx = u[4], mask_own = u[5], M_ent = u[6], rows_pad = u[7];
  const uint row = tgp * 4u + sg;
  const uint rows = E * C;
  if (row >= rows) {
    if (row < rows_pad) {
      device bfloat* dk = dK + ulong(row) * D;
      for (uint d = lane; d < D; d += 32u) dk[d] = bfloat(0.0f);
    }
    return;
  }
  const uint e = row / C, c = row % C;
  const float eps = max(f[0], 1e-4f);
  const float temp = scalars[sc_idx];

  const device bfloat* kr = Kmat + ulong(row) * D;
  float rk[8], duk[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ssk = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint d = lane + i * 32u;
    float v = (d < D) ? float(kr[d]) : 0.0f;
    rk[i] = v;
    duk[i] = 0.0f;
    ssk += v * v;
  }
  const float nk_raw = sqrt(simd_sum(ssk));
  const float nk = max(nk_raw, eps);
  const float inv_k = 1.0f / nk;
  for (uint i = 0; i < n_el; i++) rk[i] *= inv_k;

  const int c_tid = mask_own ? ent_meta[ulong(E) * M_ent + ulong(e) * C + c] : -999999;

  for (uint n = 0; n < N; n++) {
    if (mask_own && C >= 2u && ent_meta[ulong(e) * M_ent + n] == c_tid) continue;
    const float g_l = float(d_lg[(ulong(e) * N + n) * C + c]);
    const device bfloat* qr = Q + (ulong(e) * N + n) * D;
    float ssq = 0.0f;
    float rq[8];
    for (uint i = 0; i < n_el; i++) {
      const uint d = lane + i * 32u;
      float qv = (d < D) ? float(qr[d]) : 0.0f;
      rq[i] = qv;
      ssq += qv * qv;
    }
    const float inv_q = 1.0f / max(sqrt(simd_sum(ssq)), eps);
    const float w = temp * g_l * inv_q;
    for (uint i = 0; i < n_el; i++) duk[i] += w * rq[i];
  }

  float dot_ku = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_ku += duk[i] * rk[i];
  const float proj = (nk_raw >= eps) ? simd_sum(dot_ku) : 0.0f;
  device bfloat* dkr = dK + ulong(row) * D;
  for (uint i = 0; i < n_el; i++) {
    const uint d = lane + i * 32u;
    if (d < D) dkr[d] = bfloat((duk[i] - proj * rk[i]) * inv_k);
  }
}

// ---- 3. MG_OP_PLAN_SLICE_NORM_BWD ----
// Merges d_p_h[E*N, D], d_c_h[E*J, D], d_t_h[E*K, D] and runs mp_norm_bwd(x_final) -> d_x[E*M_ent, D]
[[kernel]] void mg_plan_slice_norm_bwd(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* d_ph [[buffer(1)]],
    const device bfloat* d_ch [[buffer(2)]],
    const device bfloat* d_th [[buffer(3)]],
    device bfloat* d_x [[buffer(4)]],
    constant uint* u [[buffer(5)]],  // [E, N, J, K, D, rows_pad]
    constant float* f [[buffer(6)]], // [mp_norm_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], D = u[4], rows_pad = u[5];
  const uint M_ent = N + J + K;
  const uint row = tgp * 4u + sg;
  if (row >= E * M_ent) {
    if (row < rows_pad) {
      device bfloat* dx = d_x + ulong(row) * D;
      for (uint d = lane; d < D; d += 32u) dx[d] = bfloat(0.0f);
    }
    return;
  }
  const uint e = row / M_ent, m = row % M_ent;
  const float eps = max(f[0], 1e-4f);
  const device bfloat* src = (m < N) ? (d_ph + (ulong(e) * N + m) * D)
                           : ((m < N + J) ? (d_ch + (ulong(e) * J + (m - N)) * D)
                                          : (d_th + (ulong(e) * K + (m - N - J)) * D));
  const device bfloat* xr = x + ulong(row) * D;
  device bfloat* dxr = d_x + ulong(row) * D;

  float rx[8], rdy[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    float vx = (c < D) ? float(xr[c]) : 0.0f;
    float vg = (c < D) ? float(src[c]) : 0.0f;
    rx[i] = vx; rdy[i] = vg;
    ss += vx * vx;
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  float dot_yg = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_yg += rdy[i] * (rx[i] * inv);
  const float mean_yg = simd_sum(dot_yg) / float(D);
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) dxr[c] = bfloat(inv * (rdy[i] - mean_yg * (rx[i] * inv)));
  }
}

// ---- 4. MG_OP_MP_NORM_BWD (1 simdgroup = 32 threads per row, 4 rows/TG) ----
[[kernel]] void mg_mp_norm_bwd(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* dy [[buffer(1)]],
    device bfloat* dx [[buffer(2)]],
    constant uint* u [[buffer(3)]],  // [rows, D, accum_dx, rows_pad]
    constant float* f [[buffer(4)]], // [eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint rows = u[0], D = u[1], accum = u[2], rows_pad = u[3];
  const uint row = tgp * 4u + sg;
  if (row >= rows) {
    if (!accum && row < rows_pad) {
      device bfloat* dxr = dx + ulong(row) * D;
      for (uint c = lane; c < D; c += 32u) dxr[c] = bfloat(0.0f);
    }
    return;
  }
  const float eps = max(f[0], 1e-4f);
  const device bfloat* xr = x + ulong(row) * D;
  const device bfloat* dyr = dy + ulong(row) * D;
  device bfloat* dxr = dx + ulong(row) * D;
  float rx[8], rdy[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    float vx = (c < D) ? float(xr[c]) : 0.0f;
    float vg = (c < D) ? float(dyr[c]) : 0.0f;
    rx[i] = vx; rdy[i] = vg;
    ss += vx * vx;
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  float dot_yg = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_yg += rdy[i] * (rx[i] * inv);
  const float mean_yg = simd_sum(dot_yg) / float(D);
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) {
      float g = inv * (rdy[i] - mean_yg * (rx[i] * inv));
      dxr[c] = accum ? bfloat(float(dxr[c]) + g) : bfloat(g);
    }
  }
}

// ---- 5. MG_OP_MP_LERP_BWD (1 simdgroup = 32 threads per row, 4 rows/TG) ----
// Exact backward of y = mp_norm(mp_lerp(a, norm_b ? mp_norm(b) : b, sigmoid(logit)))
// Writes da (or += da if accum_da), db, and atomic_add(sc_grads[sg_idx], d_logit).
[[kernel]] void mg_mp_lerp_bwd(
    const device bfloat* a_in [[buffer(0)]],
    const device bfloat* b_in [[buffer(1)]],
    const device bfloat* dy_in [[buffer(2)]],
    const device float* scalars [[buffer(3)]],
    device bfloat* da_out [[buffer(4)]],
    device bfloat* db_out [[buffer(5)]],
    device atomic_float* sc_grads [[buffer(6)]],
    constant uint* u [[buffer(7)]],  // [rows, D, sc_idx, norm_b, sg_idx, accum_da, rows_pad]
    constant float* f [[buffer(8)]], // [mp_norm_eps, mp_lerp_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint rows = u[0], D = u[1], sc_idx = u[2], norm_b = u[3], sg_idx = u[4], accum_da = u[5], rows_pad = u[6];
  const uint row = tgp * 4u + sg;
  if (row >= rows) {
    if (row < rows_pad) {
      device bfloat* dbr = db_out + ulong(row) * D;
      for (uint c = lane; c < D; c += 32u) dbr[c] = bfloat(0.0f);
      if (!accum_da) {
        device bfloat* dar = da_out + ulong(row) * D;
        for (uint c = lane; c < D; c += 32u) dar[c] = bfloat(0.0f);
      }
    }
    return;
  }
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float t = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float inv_s = rsqrt((1.0f - t) * (1.0f - t) + t * t + eps_l);

  const device bfloat* ar = a_in + ulong(row) * D;
  const device bfloat* br = b_in + ulong(row) * D;
  const device bfloat* dyr = dy_in + ulong(row) * D;

  float ra[8], rb_raw[8], rbn[8], rh[8], rdy[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ssb = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    ra[i] = (c < D) ? float(ar[c]) : 0.0f;
    rb_raw[i] = (c < D) ? float(br[c]) : 0.0f;
    rdy[i] = (c < D) ? float(dyr[c]) : 0.0f;
    ssb += rb_raw[i] * rb_raw[i];
  }
  const float inv_b = norm_b ? rsqrt(simd_sum(ssb) / float(D) + eps_n) : 1.0f;

  float ssh = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    rbn[i] = norm_b ? float(bfloat(rb_raw[i] * inv_b)) : rb_raw[i];
    rh[i] = float(bfloat(((1.0f - t) * ra[i] + t * rbn[i]) * inv_s));
    ssh += rh[i] * rh[i];
  }
  const float inv_h = rsqrt(simd_sum(ssh) / float(D) + eps_n);

  // Backprop through outer mp_norm(h)
  float dot_yg = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_yg += rdy[i] * (rh[i] * inv_h);
  const float mean_yg = simd_sum(dot_yg) / float(D);

  float dt_sum = 0.0f;
  float rdbn[8];
  device bfloat* dar = da_out + ulong(row) * D;
  const float s_term = (2.0f * t - 1.0f) * inv_s * inv_s;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    const float dh = inv_h * (rdy[i] - mean_yg * (rh[i] * inv_h));
    if (c < D) {
      const float ga = (1.0f - t) * inv_s * dh;
      dar[c] = accum_da ? bfloat(float(dar[c]) + ga) : bfloat(ga);
      rdbn[i] = t * inv_s * dh;
      dt_sum += dh * ((rbn[i] - ra[i]) * inv_s - rh[i] * s_term);
    } else {
      rdbn[i] = 0.0f;
    }
  }
  const float d_logit = simd_sum(dt_sum) * t * (1.0f - t);
  if (lane == 0u && d_logit != 0.0f) atomic_add_f32(&sc_grads[sg_idx], d_logit);

  // Backprop through optional inner mp_norm(b)
  device bfloat* dbr = db_out + ulong(row) * D;
  if (norm_b) {
    float dot_bg = 0.0f;
    for (uint i = 0; i < n_el; i++) dot_bg += rdbn[i] * (rb_raw[i] * inv_b);
    const float mean_bg = simd_sum(dot_bg) / float(D);
    for (uint i = 0; i < n_el; i++) {
      const uint c = lane + i * 32u;
      if (c < D) dbr[c] = bfloat(inv_b * (rdbn[i] - mean_bg * (rb_raw[i] * inv_b)));
    }
  } else {
    for (uint i = 0; i < n_el; i++) {
      const uint c = lane + i * 32u;
      if (c < D) dbr[c] = bfloat(rdbn[i]);
    }
  }
}

// ---- 6. MG_OP_MP_SWIGLU_BWD ----
[[kernel]] void mg_mp_swiglu_bwd(
    const device bfloat* gu [[buffer(0)]],
    const device bfloat* d_hid [[buffer(1)]],
    device bfloat* d_gu [[buffer(2)]],
    constant uint* u [[buffer(3)]],  // [rows, H, rows_pad]
    constant float* f [[buffer(4)]], // [inv_silu_scale]
    uint2 gid [[thread_position_in_grid]]) {
  const uint rows = u[0], H = u[1], rows_pad = u[2];
  const uint col = gid.x, row = gid.y;
  if (col >= H || row >= rows_pad) return;
  const ulong base = ulong(row) * (2ul * H);
  if (row >= rows) {
    d_gu[base + col] = bfloat(0.0f);
    d_gu[base + H + col] = bfloat(0.0f);
    return;
  }
  const float inv_sc = f[0];
  const float g = float(gu[base + col]);
  const float up = float(gu[base + H + col]);
  const float dh = float(d_hid[ulong(row) * H + col]);
  const float sig = 1.0f / (1.0f + exp(-g));
  const float ms = (g * sig) * inv_sc;
  const float d_g = dh * up * inv_sc * sig * (1.0f + g * (1.0f - sig));
  const float d_u = dh * ms;
  d_gu[base + col] = bfloat(d_g);
  d_gu[base + H + col] = bfloat(d_u);
}

// ---- 7. MG_OP_CGT_BCAST_MERGE_BWD ----
// 1 TG per env e, 256 threads.
[[kernel]] void mg_cgt_bcast_merge_bwd(
    const device bfloat* global_raw [[buffer(0)]], // [E, M_ent, D]
    const device bfloat* tb_raw [[buffer(1)]],     // [E, K, D]
    const device bfloat* d_gram_in [[buffer(2)]],  // [E, M_ent, D]
    const device int* ent_meta [[buffer(3)]],
    device bfloat* d_global_raw [[buffer(4)]],     // [E, M_ent, D]
    device bfloat* d_tb_raw [[buffer(5)]],         // [E, K, D]
    constant uint* u [[buffer(6)]],                // [E, M_ent, K, D, EM_pad, EK_pad]
    constant float* f [[buffer(7)]],               // [mp_norm_eps]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], K = u[2], D = u[3];
  if (e >= E) return;
  const float eps_n = max(f[0], 1e-4f);

  threadgroup float tg_red[9];
  threadgroup bfloat tb_sh[16 * 256];
  threadgroup float tb_inv[16];
  threadgroup float d_tb_sh[16 * 256];
  threadgroup bfloat tg_sh[256];
  threadgroup float mean_tb_sh[256];
  threadgroup float tg_inv_sh;
  threadgroup int tids[16];

  if (tid < K) tids[tid] = ent_meta[ulong(E) * M_ent + ulong(e) * K + tid];
  for (uint k = 0; k < K; k++) {
    const float v = (tid < D) ? float(tb_raw[(ulong(e) * K + k) * D + tid]) : 0.0f;
    const float inv = rsqrt(tg_sum_bwd(v * v, lane, sg, 8u, tg_red) / float(D) + eps_n);
    if (tid < D) {
      tb_sh[k * D + tid] = bfloat(v * inv);
      d_tb_sh[k * D + tid] = 0.0f;
    }
    if (tid == 0u) tb_inv[k] = inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Recompute t_global = mp_norm(mean_k tb_sh[k])
  {
    const float inv_k = 1.0f / float(max(1u, K));
    float m_v = 0.0f;
    if (tid < D) {
      for (uint k = 0; k < K; k++) m_v += float(tb_sh[k * D + tid]);
      m_v *= inv_k;
      mean_tb_sh[tid] = m_v;
    }
    const float inv_g = rsqrt(tg_sum_bwd(m_v * m_v, lane, sg, 8u, tg_red) / float(D) + eps_n);
    if (tid < D) tg_sh[tid] = bfloat(m_v * inv_g);
    if (tid == 0u) tg_inv_sh = inv_g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float d_tg_acc = 0.0f;
  for (uint m = 0; m < M_ent; m++) {
    const int et = ent_meta[ulong(e) * M_ent + m];
    int match_k = -1;
    for (uint k = 0; k < K; k++) {
      if (et == tids[k]) { match_k = int(k); break; }
    }
    const float gr_v = (tid < D) ? float(global_raw[(ulong(e) * M_ent + m) * D + tid]) : 0.0f;
    const float inv_gv = rsqrt(tg_sum_bwd(gr_v * gr_v, lane, sg, 8u, tg_red) / float(D) + eps_n);
    const float gv = (tid < D) ? float(bfloat(gr_v * inv_gv)) : 0.0f;
    const float tv_raw = (tid < D) ? ((match_k >= 0) ? float(tb_sh[uint(match_k) * D + tid]) : float(tg_sh[tid])) : 0.0f;
    const float inv_tv = rsqrt(tg_sum_bwd(tv_raw * tv_raw, lane, sg, 8u, tg_red) / float(D) + eps_n);
    const float tv = (tid < D) ? float(bfloat(tv_raw * inv_tv)) : 0.0f;

    const float sm = (tid < D) ? float(bfloat(gv + tv)) : 0.0f;
    const float inv_sm = rsqrt(tg_sum_bwd(sm * sm, lane, sg, 8u, tg_red) / float(D) + eps_n);
    const float y_sm = sm * inv_sm;
    const float dy = (tid < D) ? float(d_gram_in[(ulong(e) * M_ent + m) * D + tid]) : 0.0f;
    const float mean_yg = tg_sum_bwd(dy * y_sm, lane, sg, 8u, tg_red) / float(D);
    const float d_sm = inv_sm * (dy - mean_yg * y_sm);

    // Backprop to global_raw[m]
    const float mean_gg = tg_sum_bwd(d_sm * gv, lane, sg, 8u, tg_red) / float(D);
    if (tid < D) {
      d_global_raw[(ulong(e) * M_ent + m) * D + tid] = bfloat(inv_gv * (d_sm - mean_gg * gv));
    }

    // Backprop to team_vals -> tb[match_k] or tg
    const float mean_tg = tg_sum_bwd(d_sm * tv, lane, sg, 8u, tg_red) / float(D);
    const float d_tv_raw = inv_tv * (d_sm - mean_tg * tv);
    if (tid < D) {
      if (match_k >= 0) d_tb_sh[uint(match_k) * D + tid] += d_tv_raw;
      else d_tg_acc += d_tv_raw;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Backprop d_tg_acc through t_global = mp_norm(mean_k tb_sh[k])
  {
    const float inv_g = tg_inv_sh;
    const float y_g = (tid < D) ? (mean_tb_sh[tid] * inv_g) : 0.0f;
    const float mean_yg = tg_sum_bwd(d_tg_acc * y_g, lane, sg, 8u, tg_red) / float(D);
    const float d_mean = inv_g * (d_tg_acc - mean_yg * y_g) / float(max(1u, K));
    if (tid < D) {
      for (uint k = 0; k < K; k++) d_tb_sh[k * D + tid] += d_mean;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Backprop d_tb_sh[k] through mp_norm(tb_raw[k])
  for (uint k = 0; k < K; k++) {
    const float inv_b = tb_inv[k];
    const float y_b = (tid < D) ? (float(tb_raw[(ulong(e) * K + k) * D + tid]) * inv_b) : 0.0f;
    const float dtb = (tid < D) ? d_tb_sh[k * D + tid] : 0.0f;
    const float mean_yg = tg_sum_bwd(dtb * y_b, lane, sg, 8u, tg_red) / float(D);
    if (tid < D) {
      d_tb_raw[(ulong(e) * K + k) * D + tid] = bfloat(inv_b * (dtb - mean_yg * y_b));
    }
  }
}

// ---- 8. MG_OP_CGT_TEAM_RIVAL_BWD ----
// 1 TG per env e, 256 threads.
[[kernel]] void mg_cgt_team_rival_bwd(
    const device bfloat* qt_raw [[buffer(0)]],
    const device bfloat* kt_raw [[buffer(1)]],
    const device bfloat* vt_raw [[buffer(2)]],
    const device bfloat* d_tmixed [[buffer(3)]],
    device bfloat* d_qt_raw [[buffer(4)]],
    device bfloat* d_kt_raw [[buffer(5)]],
    device bfloat* d_vt_raw [[buffer(6)]],
    constant uint* u [[buffer(7)]],  // [E, K, D, R_team]
    constant float* f [[buffer(8)]], // [mp_norm_eps]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], K = u[1], D = u[2], R = u[3];
  if (e >= E) return;
  const float eps_n = max(f[0], 1e-4f);
  const float inv_k = rsqrt(float(max(1u, K)));

  threadgroup float tg_red[9];
  threadgroup bfloat q_sh[16 * 128];
  threadgroup bfloat k_sh[16 * 128];
  threadgroup float q_inv[16], k_inv[16], v_inv[16];
  threadgroup float gram[16 * 16];
  threadgroup float d_gram[16 * 16];
  threadgroup bfloat d_mv_sh[16 * 256]; // 8 KB

  for (uint k = 0; k < K; k++) {
    const float qv = (tid < R) ? float(qt_raw[(ulong(e) * K + k) * R + tid]) : 0.0f;
    const float kv = (tid < R) ? float(kt_raw[(ulong(e) * K + k) * R + tid]) : 0.0f;
    const float vv = (tid < D) ? float(vt_raw[(ulong(e) * K + k) * D + tid]) : 0.0f;
    const float iq = rsqrt(tg_sum_bwd(qv * qv, lane, sg, 8u, tg_red) / float(R) + eps_n);
    const float ik = rsqrt(tg_sum_bwd(kv * kv, lane, sg, 8u, tg_red) / float(R) + eps_n);
    const float iv = rsqrt(tg_sum_bwd(vv * vv, lane, sg, 8u, tg_red) / float(D) + eps_n);
    if (tid < R) {
      q_sh[k * R + tid] = bfloat(qv * iq);
      k_sh[k * R + tid] = bfloat(kv * ik);
    }
    if (tid == 0u) { q_inv[k] = iq; k_inv[k] = ik; v_inv[k] = iv; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k1 = 0; k1 < K; k1++) {
    for (uint k2 = 0; k2 < K; k2++) {
      float dp = (tid < R) ? (float(q_sh[k1 * R + tid]) * float(k_sh[k2 * R + tid])) : 0.0f;
      const float tot = tg_sum_bwd(dp, lane, sg, 8u, tg_red);
      if (tid == 0u) gram[k1 * K + k2] = float(bfloat(tot * inv_k));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Backprop through mp_norm(gram @ v) -> d_mv_sh[K, D]
  for (uint k1 = 0; k1 < K; k1++) {
    float acc = 0.0f;
    if (tid < D) {
      for (uint k2 = 0; k2 < K; k2++) {
        float vv = float(bfloat(float(vt_raw[(ulong(e) * K + k2) * D + tid]) * v_inv[k2]));
        acc += gram[k1 * K + k2] * vv;
      }
    }
    const float inv_o = rsqrt(tg_sum_bwd(acc * acc, lane, sg, 8u, tg_red) / float(D) + eps_n);
    const float yo = acc * inv_o;
    const float dy = (tid < D) ? float(d_tmixed[(ulong(e) * K + k1) * D + tid]) : 0.0f;
    const float myg = tg_sum_bwd(dy * yo, lane, sg, 8u, tg_red) / float(D);
    if (tid < D) d_mv_sh[k1 * D + tid] = bfloat(inv_o * (dy - myg * yo));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // d_gram[k1, k2] = dot(d_mv[k1], v[k2]) * inv_k
  for (uint k1 = 0; k1 < K; k1++) {
    for (uint k2 = 0; k2 < K; k2++) {
      float dp = 0.0f;
      if (tid < D) {
        float vv = float(bfloat(float(vt_raw[(ulong(e) * K + k2) * D + tid]) * v_inv[k2]));
        dp = float(d_mv_sh[k1 * D + tid]) * vv;
      }
      const float tot = tg_sum_bwd(dp, lane, sg, 8u, tg_red);
      if (tid == 0u) d_gram[k1 * K + k2] = tot * inv_k;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // d_vt_raw[k2] = mp_norm_bwd(vt_raw[k2], sum_{k1} gram[k1, k2] * d_mv[k1])
  for (uint k2 = 0; k2 < K; k2++) {
    float dv = 0.0f;
    if (tid < D) {
      for (uint k1 = 0; k1 < K; k1++) dv += gram[k1 * K + k2] * float(d_mv_sh[k1 * D + tid]);
    }
    const float iv = v_inv[k2];
    const float yv = (tid < D) ? (float(vt_raw[(ulong(e) * K + k2) * D + tid]) * iv) : 0.0f;
    const float myg = tg_sum_bwd(dv * yv, lane, sg, 8u, tg_red) / float(D);
    if (tid < D) d_vt_raw[(ulong(e) * K + k2) * D + tid] = bfloat(iv * (dv - myg * yv));
  }

  // d_qt_raw[k1] & d_kt_raw[k2]
  for (uint k1 = 0; k1 < K; k1++) {
    float dq = 0.0f;
    if (tid < R) {
      for (uint k2 = 0; k2 < K; k2++) dq += d_gram[k1 * K + k2] * float(k_sh[k2 * R + tid]);
    }
    const float iq = q_inv[k1];
    const float yq = (tid < R) ? float(q_sh[k1 * R + tid]) : 0.0f;
    const float myg = tg_sum_bwd(dq * yq, lane, sg, 8u, tg_red) / float(R);
    if (tid < R) d_qt_raw[(ulong(e) * K + k1) * R + tid] = bfloat(iq * (dq - myg * yq));
  }
  for (uint k2 = 0; k2 < K; k2++) {
    float dk = 0.0f;
    if (tid < R) {
      for (uint k1 = 0; k1 < K; k1++) dk += d_gram[k1 * K + k2] * float(q_sh[k1 * R + tid]);
    }
    const float ik = k_inv[k2];
    const float yk = (tid < R) ? float(k_sh[k2 * R + tid]) : 0.0f;
    const float myg = tg_sum_bwd(dk * yk, lane, sg, 8u, tg_red) / float(R);
    if (tid < R) d_kt_raw[(ulong(e) * K + k2) * R + tid] = bfloat(ik * (dk - myg * yk));
  }
}

// ---- 9. MG_OP_CGT_GRAM_STAGE1_BWD ----
// Sub-kernel 9a: Team pool backward -> d_v_norm[E*M_ent, D] (init), d_xn[E*M_ent, D] (+=),
//   d_w_pool[D] (atomic FP32), d_blend_pool (atomic FP32).
[[kernel]] void mg_cgt_team_pool_bwd(
    const device bfloat* xn [[buffer(0)]],
    const device bfloat* v_norm [[buffer(1)]],
    const device bfloat* w_pool [[buffer(2)]],
    const device bfloat* d_t_in [[buffer(3)]],
    const device int* ent_meta [[buffer(4)]],
    const device float* scalars [[buffer(5)]],
    device bfloat* d_v_norm [[buffer(6)]],
    device bfloat* d_xn [[buffer(7)]],
    device atomic_float* d_w_pool [[buffer(8)]],
    device atomic_float* sc_grads [[buffer(9)]],
    constant uint* u [[buffer(10)]],  // [E, N, J, K, D, sc_idx_bp, sg_idx_bp]
    constant float* f [[buffer(11)]], // [mp_norm_eps, mp_lerp_eps]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], D = u[4], sc_idx = u[5], sg_idx = u[6];
  if (e >= E) return;
  const uint M_ent = N + J + K;
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float bp = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float inv_s = rsqrt((1.0f - bp) * (1.0f - bp) + bp * bp + eps_l);
  const float s_term = (2.0f * bp - 1.0f) * inv_s * inv_s;

  threadgroup float tg_red[9];
  threadgroup float sig_pool[256];
  threadgroup int ent_t[256];
  threadgroup float d_tp_raw_sh[16 * 256]; // 16 KB
  threadgroup bfloat tp_raw_sh[16 * 256];  // 8 KB (total ~25 KB <= 32 KB)
  threadgroup float den_sh[16];
  threadgroup int tids[16];

  if (tid < K) tids[tid] = ent_meta[ulong(E) * M_ent + ulong(e) * K + tid];

  for (uint m = 0; m < M_ent; m++) {
    const float xv = (tid < D) ? float(xn[(ulong(e) * M_ent + m) * D + tid]) : 0.0f;
    const float wv = (tid < D) ? float(w_pool[tid]) : 0.0f;
    const float tot = tg_sum_bwd(xv * wv, lane, sg, 8u, tg_red);
    if (tid == 0u) {
      sig_pool[m] = 1.0f / (1.0f + exp(-tot));
      ent_t[m] = ent_meta[ulong(e) * M_ent + m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float d_bp_sum = 0.0f;
  for (uint k = 0; k < K; k++) {
    const int tk = tids[k];
    float den_raw = 0.0f;
    for (uint m = 0; m < M_ent; m++) {
      if (ent_t[m] == tk) den_raw += sig_pool[m];
    }
    const float den = max(den_raw, 1e-4f);
    if (tid == 0u) den_sh[k] = den_raw;

    float q_reg = 0.0f;
    if (tid < D) {
      float sum_v = 0.0f;
      for (uint m = 0; m < M_ent; m++) {
        if (ent_t[m] == tk) sum_v += sig_pool[m] * float(v_norm[(ulong(e) * M_ent + m) * D + tid]);
      }
      q_reg = sum_v / den;
      tp_raw_sh[k * D + tid] = bfloat(q_reg);
    }
    const float inv_tp = rsqrt(tg_sum_bwd(q_reg * q_reg, lane, sg, 8u, tg_red) / float(D) + eps_n);
    const float tp = (tid < D) ? float(bfloat(q_reg * inv_tp)) : 0.0f;
    const float tc = (tid < D) ? float(xn[(ulong(e) * M_ent + (N + J + k)) * D + tid]) : 0.0f;
    const float h = (tid < D) ? float(bfloat(((1.0f - bp) * tc + bp * tp) * inv_s)) : 0.0f;
    const float inv_h = rsqrt(tg_sum_bwd(h * h, lane, sg, 8u, tg_red) / float(D) + eps_n);
    const float yh = h * inv_h;
    const float dy = (tid < D) ? float(d_t_in[(ulong(e) * K + k) * D + tid]) : 0.0f;
    const float myg = tg_sum_bwd(dy * yh, lane, sg, 8u, tg_red) / float(D);
    const float dh = inv_h * (dy - myg * yh);

    // Accumulate d_xn for team token N+J+k
    if (tid < D) {
      const ulong idx_t = (ulong(e) * M_ent + (N + J + k)) * D + tid;
      d_xn[idx_t] = bfloat(float(d_xn[idx_t]) + (1.0f - bp) * inv_s * dh);
    }
    const float dt_k = tg_sum_bwd((tid < D) ? (dh * ((tp - tc) * inv_s - h * s_term)) : 0.0f, lane, sg, 8u, tg_red);
    if (tid == 0u) d_bp_sum += dt_k * bp * (1.0f - bp);

    // Backprop through mp_norm(tp_raw) -> d_tp_raw_sh[k]
    const float d_tp = bp * inv_s * dh;
    const float mtp = tg_sum_bwd((tid < D) ? (d_tp * (q_reg * inv_tp)) : 0.0f, lane, sg, 8u, tg_red) / float(D);
    if (tid < D) d_tp_raw_sh[k * D + tid] = inv_tp * (d_tp - mtp * (q_reg * inv_tp));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u && d_bp_sum != 0.0f) atomic_add_f32(&sc_grads[sg_idx], d_bp_sum);

  // Backprop per entity m to d_v_norm[m], d_xn[m], and d_w_pool[tid]
  float d_wp_acc = 0.0f;
  const float wv = (tid < D) ? float(w_pool[tid]) : 0.0f;
  for (uint m = 0; m < M_ent; m++) {
    const int et = ent_t[m];
    int mk = -1;
    for (uint k = 0; k < K; k++) {
      if (et == tids[k]) { mk = int(k); break; }
    }
    const ulong idx_m = (ulong(e) * M_ent + m) * D + tid;
    if (mk < 0) {
      if (tid < D) d_v_norm[idx_m] = bfloat(0.0f);
      continue;
    }
    const uint uk = uint(mk);
    const float den_raw = den_sh[uk];
    const float den = max(den_raw, 1e-4f);
    const float sm = sig_pool[m];
    const float dtr = (tid < D) ? d_tp_raw_sh[uk * D + tid] : 0.0f;
    const float vv = (tid < D) ? float(v_norm[idx_m]) : 0.0f;
    const float tpr = (tid < D) ? float(tp_raw_sh[uk * D + tid]) : 0.0f;

    if (tid < D) d_v_norm[idx_m] = bfloat((sm / den) * dtr);

    const float num_d = tg_sum_bwd(dtr * (vv - ((den_raw >= 1e-4f) ? tpr : 0.0f)), lane, sg, 8u, tg_red);
    const float d_ps = (num_d / den) * sm * (1.0f - sm);
    if (tid < D) {
      d_xn[idx_m] = bfloat(float(d_xn[idx_m]) + d_ps * wv);
      d_wp_acc += d_ps * float(xn[idx_m]);
    }
  }
  if (tid < D && d_wp_acc != 0.0f) atomic_add_f32(&d_w_pool[tid], d_wp_acc);
}

// Sub-kernel 9b: Global Gram backward (general for ANY M_ent, tiled r_blk=32 -> 8 KB SRAM!)
// Computes: d_v_norm += r_norm @ d_kv, d_r_norm = (d_global_raw @ kv^T + v_norm @ d_kv^T)
// Launch: grid = (E, 1, 1) for M_ent <= 48 (fast associative path!), or ((R+31)/32, E, 1) general!
[[kernel]] void mg_cgt_global_gram_bwd_small(
    const device bfloat* r_norm [[buffer(0)]],      // [E, M_ent, R]
    const device bfloat* v_norm [[buffer(1)]],      // [E, M_ent, D]
    const device bfloat* d_glob [[buffer(2)]],      // [E, M_ent, D]
    device bfloat* d_v_norm [[buffer(3)]],          // [E, M_ent, D] (+=)
    device bfloat* d_r_norm [[buffer(4)]],          // [E, M_ent, R]
    constant uint* u [[buffer(5)]],                 // [E, M_ent, D, R]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], D = u[2], R = u[3];
  if (e >= E) return;
  const float inv_m = rsqrt(float(max(1u, M_ent)));
  const device bfloat* re = r_norm + ulong(e) * M_ent * R;
  const device bfloat* ve = v_norm + ulong(e) * M_ent * D;
  const device bfloat* dge = d_glob + ulong(e) * M_ent * D;
  device bfloat* dve = d_v_norm + ulong(e) * M_ent * D;
  device bfloat* dre = d_r_norm + ulong(e) * M_ent * R;

  threadgroup float G[48 * 48];  // 9 KB
  threadgroup float dG[48 * 48]; // 9 KB (total 18 KB SRAM!)

  // 1. Symmetric G[m1, m2] = dot(r[m1], r[m2]) * inv_m
  uint p_idx = 0u;
  for (uint m1 = 0; m1 < M_ent; m1++) {
    for (uint m2 = m1; m2 < M_ent; m2++) {
      if ((p_idx & 7u) == sg) {
        const device bfloat* r1 = re + ulong(m1) * R;
        const device bfloat* r2 = re + ulong(m2) * R;
        float dp = 0.0f;
        for (uint r = lane; r < R; r += 32u) dp += float(r1[r]) * float(r2[r]);
        const float val = simd_sum(dp) * inv_m;
        if (lane == 0u) { G[m1 * M_ent + m2] = val; G[m2 * M_ent + m1] = val; }
      }
      p_idx++;
    }
  }

  // 2. Non-symmetric dG[m1, m2] = dot(d_glob[m1], v[m2]) * inv_m
  uint q_idx = 0u;
  for (uint m1 = 0; m1 < M_ent; m1++) {
    for (uint m2 = 0; m2 < M_ent; m2++) {
      if ((q_idx & 7u) == sg) {
        const device bfloat* g1 = dge + ulong(m1) * D;
        const device bfloat* v2 = ve + ulong(m2) * D;
        float dp = 0.0f;
        for (uint c = lane; c < D; c += 32u) dp += float(g1[c]) * float(v2[c]);
        const float val = simd_sum(dp) * inv_m;
        if (lane == 0u) dG[m1 * M_ent + m2] = val;
      }
      q_idx++;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. d_v_norm += G @ d_glob (G is symmetric!)
  for (uint c = tid; c < D; c += 256u) {
    float dg_reg[48];
    for (uint m = 0; m < M_ent; m++) dg_reg[m] = float(dge[ulong(m) * D + c]);
    for (uint m1 = 0; m1 < M_ent; m1++) {
      float acc = 0.0f;
      const uint go = m1 * M_ent;
      for (uint m2 = 0; m2 < M_ent; m2++) acc += G[go + m2] * dg_reg[m2];
      const ulong idx = ulong(m1) * D + c;
      dve[idx] = bfloat(float(dve[idx]) + acc);
    }
  }

  // 4. d_r_norm[m1, r] = sum_{m2} (dG[m1, m2] + dG[m2, m1]) * r[m2, r]
  for (uint r = tid; r < R; r += 256u) {
    float r_reg[48];
    for (uint m = 0; m < M_ent; m++) r_reg[m] = float(re[ulong(m) * R + r]);
    for (uint m1 = 0; m1 < M_ent; m1++) {
      float acc = 0.0f;
      for (uint m2 = 0; m2 < M_ent; m2++) {
        acc += (dG[m1 * M_ent + m2] + dG[m2 * M_ent + m1]) * r_reg[m2];
      }
      dre[ulong(m1) * R + r] = bfloat(acc);
    }
  }
}

// Sub-kernel 9b-general (M_ent > 48): 2-pass exact kv/d_kv without [E, R, D] DRAM!
// Pass 1 (d_v): grid = ((D+31)/32, E, 1), 128 threads: computes d_kv[32, 32] in 4KB SRAM, accumulates d_v_norm += r @ d_kv
[[kernel]] void mg_cgt_global_gram_bwd_dv(
    const device bfloat* r_norm [[buffer(0)]],
    const device bfloat* d_glob [[buffer(1)]],
    device bfloat* d_v_norm [[buffer(2)]],
    constant uint* u [[buffer(3)]], // [E, M_ent, D, R]
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], D = u[2], R = u[3];
  const uint e = tgp.y, d0 = tgp.x * 32u;
  if (e >= E || d0 >= D) return;
  const float inv_m = rsqrt(float(max(1u, M_ent)));
  const uint c_local = tid & 31u, r_lane = tid >> 5u, c_glob = d0 + c_local;
  threadgroup float dkv_sh[32 * 32];
  for (uint r0 = 0; r0 < R; r0 += 32u) {
    const uint r_lim = min(32u, R - r0);
    for (uint rr = r_lane; rr < r_lim; rr += 4u) {
      float acc = 0.0f;
      if (c_glob < D) {
        for (uint m = 0; m < M_ent; m++) {
          acc += float(r_norm[(ulong(e) * M_ent + m) * R + (r0 + rr)]) * float(d_glob[(ulong(e) * M_ent + m) * D + c_glob]);
        }
      }
      dkv_sh[rr * 32u + c_local] = acc * inv_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint m = r_lane; m < M_ent; m += 4u) {
      if (c_glob < D) {
        const ulong idx = (ulong(e) * M_ent + m) * D + c_glob;
        float acc = float(d_v_norm[idx]);
        const device bfloat* rm = r_norm + (ulong(e) * M_ent + m) * R + r0;
        for (uint rr = 0; rr < r_lim; rr++) acc += float(rm[rr]) * dkv_sh[rr * 32u + c_local];
        d_v_norm[idx] = bfloat(acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Pass 2 (d_r): grid = ((R+31)/32, E, 1), 128 threads: loops d0 in chunks of 32, computes kv[32,32] & d_kv[32,32] in 8KB SRAM, accumulates d_r_norm
[[kernel]] void mg_cgt_global_gram_bwd_dr(
    const device bfloat* r_norm [[buffer(0)]],
    const device bfloat* v_norm [[buffer(1)]],
    const device bfloat* d_glob [[buffer(2)]],
    device bfloat* d_r_norm [[buffer(3)]],
    constant uint* u [[buffer(4)]], // [E, M_ent, D, R]
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], D = u[2], R = u[3];
  const uint e = tgp.y, r0 = tgp.x * 32u;
  if (e >= E || r0 >= R) return;
  const float inv_m = rsqrt(float(max(1u, M_ent)));
  const uint r_local = tid & 31u, lane4 = tid >> 5u, r_glob = r0 + r_local;
  threadgroup float kv_sh[32 * 32];
  threadgroup float dkv_sh[32 * 32];

  for (uint d0 = 0; d0 < D; d0 += 32u) {
    const uint d_lim = min(32u, D - d0);
    for (uint dd = lane4; dd < d_lim; dd += 4u) {
      float a_kv = 0.0f, a_dkv = 0.0f;
      if (r_glob < R) {
        for (uint m = 0; m < M_ent; m++) {
          const float rv = float(r_norm[(ulong(e) * M_ent + m) * R + r_glob]);
          a_kv  += rv * float(v_norm[(ulong(e) * M_ent + m) * D + (d0 + dd)]);
          a_dkv += rv * float(d_glob[(ulong(e) * M_ent + m) * D + (d0 + dd)]);
        }
      }
      kv_sh[r_local * 32u + dd] = a_kv * inv_m;
      dkv_sh[r_local * 32u + dd] = a_dkv * inv_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint m = lane4; m < M_ent; m += 4u) {
      if (r_glob < R) {
        const ulong idx = (ulong(e) * M_ent + m) * R + r_glob;
        float acc = (d0 == 0u) ? 0.0f : float(d_r_norm[idx]);
        const device bfloat* dgm = d_glob + (ulong(e) * M_ent + m) * D + d0;
        const device bfloat* vm  = v_norm + (ulong(e) * M_ent + m) * D + d0;
        for (uint dd = 0; dd < d_lim; dd++) {
          acc += float(dgm[dd]) * kv_sh[r_local * 32u + dd] + float(vm[dd]) * dkv_sh[r_local * 32u + dd];
        }
        d_r_norm[idx] = bfloat(acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- 10. MG_OP_PLAN_EMBED_CAT_BWD ----
// 1 simdgroup = 32 threads per (e, m) row, 4 rows/TG.
// Writes d_obs_enc, d_cart_enc, d_team_enc (zero-padding trailing rows), atomic_add d_type_embed[3, D], d_type_blend!
[[kernel]] void mg_plan_embed_cat_bwd(
    const device bfloat* obs_enc [[buffer(0)]],
    const device bfloat* cart_enc [[buffer(1)]],
    const device bfloat* team_enc [[buffer(2)]],
    const device bfloat* type_emb [[buffer(3)]], // [3, D] pre-normalized
    const device bfloat* d_x [[buffer(4)]],      // [E*M_ent, D]
    const device float* scalars [[buffer(5)]],
    device bfloat* d_obs_enc [[buffer(6)]],
    device bfloat* d_cart_enc [[buffer(7)]],
    device bfloat* d_team_enc [[buffer(8)]],
    device atomic_float* d_te_norm [[buffer(9)]], // [3, D] FP32
    device atomic_float* sc_grads [[buffer(10)]],
    constant uint* u [[buffer(11)]],              // [E, N, J, K, D, sc_idx_tb, sg_idx_tb]
    constant float* f [[buffer(12)]],             // [mp_norm_eps, mp_lerp_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], D = u[4], sc_idx = u[5], sg_idx = u[6];
  const uint M_ent = N + J + K;
  const uint row = tgp * 4u + sg;
  const uint e = row / M_ent, m = row % M_ent;
  if (e >= E) return;

  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float tb = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float inv_s = rsqrt((1.0f - tb) * (1.0f - tb) + tb * tb + eps_l);
  const float s_term = (2.0f * tb - 1.0f) * inv_s * inv_s;

  const device bfloat* src = nullptr;
  device bfloat* dst = nullptr;
  uint type_id = 0u;
  if (m < N) {
    src = obs_enc + (ulong(e) * N + m) * D;
    dst = d_obs_enc + (ulong(e) * N + m) * D;
    type_id = 0u;
  } else if (m < N + J) {
    src = cart_enc + (ulong(e) * J + (m - N)) * D;
    dst = d_cart_enc + (ulong(e) * J + (m - N)) * D;
    type_id = 1u;
  } else {
    src = team_enc + (ulong(e) * K + (m - N - J)) * D;
    dst = d_team_enc + (ulong(e) * K + (m - N - J)) * D;
    type_id = 2u;
  }
  const device bfloat* te = type_emb + ulong(type_id) * D;
  const device bfloat* dxr = d_x + ulong(row) * D;

  float rsrc[8], ra[8], rb[8], rh[8], rdy[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss1 = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    rsrc[i] = (c < D) ? float(src[c]) : 0.0f;
    rb[i]   = (c < D) ? float(te[c]) : 0.0f;
    rdy[i]  = (c < D) ? float(dxr[c]) : 0.0f;
    ss1 += rsrc[i] * rsrc[i];
  }
  const float inv1 = rsqrt(simd_sum(ss1) / float(D) + eps_n);

  float ss2 = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    ra[i] = float(bfloat(rsrc[i] * inv1));
    rh[i] = float(bfloat(((1.0f - tb) * ra[i] + tb * rb[i]) * inv_s));
    ss2 += rh[i] * rh[i];
  }
  const float inv2 = rsqrt(simd_sum(ss2) / float(D) + eps_n);

  float dot_yg = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_yg += rdy[i] * (rh[i] * inv2);
  const float mean_yg = simd_sum(dot_yg) / float(D);

  float rda[8];
  float dt_sum = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    const float dh = inv2 * (rdy[i] - mean_yg * (rh[i] * inv2));
    if (c < D) {
      rda[i] = (1.0f - tb) * inv_s * dh;
      const float g_te = tb * inv_s * dh;
      atomic_add_f32(&d_te_norm[ulong(type_id) * D + c], g_te);
      dt_sum += dh * ((rb[i] - ra[i]) * inv_s - rh[i] * s_term);
    } else {
      rda[i] = 0.0f;
    }
  }
  const float d_logit = simd_sum(dt_sum) * tb * (1.0f - tb);
  if (lane == 0u && d_logit != 0.0f) atomic_add_f32(&sc_grads[sg_idx], d_logit);

  float dot_ag = 0.0f;
  for (uint i = 0; i < n_el; i++) dot_ag += rda[i] * ra[i];
  const float mean_ag = simd_sum(dot_ag) / float(D);
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) dst[c] = bfloat(inv1 * (rda[i] - mean_ag * ra[i]));
  }
}

// ---- 11. MG_OP_UNIT_W_BWD ----
// Projects FP32 grad_Wu[N, K] through F.normalize(W, dim=-1) -> FP32 grad_W[N, K].
// 1 simdgroup = 32 threads per row n, 4 rows/TG.
[[kernel]] void mg_unit_w_bwd(
    const device float* w_raw [[buffer(0)]],   // [N, K] FP32 (actual param)
    const device float* g_wu [[buffer(1)]],    // [N, K] FP32 (accumulated grad wrt Wu)
    device float* g_w [[buffer(2)]],           // [N, K] FP32 (output grad wrt W; may alias g_wu)
    constant uint* u [[buffer(3)]],            // [N, K, K_stride_wu]
    constant float* f [[buffer(4)]],           // [eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint N = u[0], K = u[1], K_str = u[2];
  const uint row = tgp * 4u + sg;
  if (row >= N) return;
  const float eps = max(f[0], 1e-4f);
  const device float* wr = w_raw + ulong(row) * K;
  const device float* gr = g_wu + ulong(row) * K_str;
  device float* dst = g_w + ulong(row) * K;

  float ss = 0.0f, dp = 0.0f;
  for (uint c = lane; c < K; c += 32u) {
    float wv = wr[c], gv = gr[c];
    ss += wv * wv;
    dp += wv * gv;
  }
  const float nrm_raw = sqrt(simd_sum(ss));
  const float inv = 1.0f / max(nrm_raw, eps);
  const float dot_wu_g = (nrm_raw >= eps) ? (simd_sum(dp) * inv) : 0.0f;
  for (uint c = lane; c < K; c += 32u) {
    float wu = wr[c] * inv;
    dst[c] = (gr[c] - dot_wu_g * wu) * inv;
  }
}

// ---- 12. MG_OP_ZERO_BUF (fast GPU zero of FP32 grad/diag buffers or BF16 pad rows) ----
[[kernel]] void mg_zero_f32(
    device float* buf [[buffer(0)]],
    constant uint* u [[buffer(1)]], // [n_floats]
    uint i [[thread_position_in_grid]]) {
  if (i < u[0]) buf[i] = 0.0f;
}
)MSL");
}

} // namespace mesh_simdt
#endif
