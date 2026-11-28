#ifndef MESH_GRAPHFLOW_H
#define MESH_GRAPHFLOW_H

#include <string>

namespace mesh_simdt {

inline std::string build_graphflow_msl() {
  return std::string(R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float gf_safe_tanh(float x) {
  return tanh(clamp(x, -20.0f, 20.0f));
}

inline void gf_atomic_add_f32(device float* ptr, float val) {
  device atomic_uint* a = reinterpret_cast<device atomic_uint*>(ptr);
  uint old_u = atomic_load_explicit(a, memory_order_relaxed);
  for (uint iter = 0u; iter < 64u; iter++) {
    float new_f = as_type<float>(old_u) + val;
    uint new_u = as_type<uint>(new_f);
    if (atomic_compare_exchange_weak_explicit(a, &old_u, new_u, memory_order_relaxed, memory_order_relaxed)) break;
  }
}

inline float gf_tg_sum(float v, uint lane, uint sg, uint n_sg, threadgroup float* scratch) {
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

inline float gf_tg_max(float v, uint lane, uint sg, uint n_sg, threadgroup float* scratch) {
  v = simd_max(v);
  if (lane == 0u) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float x = (lane < n_sg) ? scratch[lane] : -1e30f;
    float tot = simd_max(x);
    if (lane == 0u) scratch[8] = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float res = scratch[8];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return res;
}

inline float gf_tg_min(float v, uint lane, uint sg, uint n_sg, threadgroup float* scratch) {
  v = simd_min(v);
  if (lane == 0u) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float x = (lane < n_sg) ? scratch[lane] : 1e30f;
    float tot = simd_min(x);
    if (lane == 0u) scratch[8] = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float res = scratch[8];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return res;
}

inline void gf_tau_embed(float tau, thread float out[8]) {
  const float pi_t = 3.14159265358979323846f * tau;
  out[0] = tau;
  out[1] = 1.0f - tau;
  out[2] = sin(pi_t);
  out[3] = cos(pi_t);
  out[4] = sin(2.0f * pi_t);
  out[5] = cos(2.0f * pi_t);
  out[6] = tau * tau;
  out[7] = sqrt(max(tau, 1e-4f));
}

// ============================================================================
// 1. MG_OP_GRAPH_PRENORM (1 TG per env e, 128 threads)
// Exact match to prenorm_graph_inputs in graph_flow_policy.py
// ============================================================================
[[kernel]] void mg_graph_prenorm(
    const device bfloat* obs_in [[buffer(0)]],
    const device bfloat* nav_in [[buffer(1)]],
    const device bfloat* carts_in [[buffer(2)]],
    const device bfloat* teams_in [[buffer(3)]],
    const device float* acc_in [[buffer(4)]],
    const device float* geo_ncfg [[buffer(5)]], // [0..3]=geo, [4..15]=ncfg[0..11]
    device bfloat* obs_out [[buffer(6)]],       // [EN_pad, 128]
    device bfloat* nav_out [[buffer(7)]],       // [EW_pad, 64]
    device bfloat* cart_out [[buffer(8)]],      // [EJ_pad, 64]
    device bfloat* team_out [[buffer(9)]],      // [EK_pad, 64]
    device int* ent_meta [[buffer(10)]],        // [E*(M_ent + K)]
    device int* bot_nav_id [[buffer(11)]],      // [E*N]
    constant uint* u [[buffer(12)]],            // [E, N, W_nav, J, K, A_dim]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4], A_dim = u[5];
  if (e >= E) return;
  const uint M_ent = N + W_nav + J + K;

  const float cx = geo_ncfg[0], cy = geo_ncfg[1], cz = geo_ncfg[2];
  const bool has_geo = (geo_ncfg[3] > 0.0f);
  const float mp_eps = max(geo_ncfg[4], 1e-4f);
  const float rad = max(geo_ncfg[3], geo_ncfg[5]);
  const float pos_fb = geo_ncfg[6];
  const float vel_sc = geo_ncfg[7];
  const float hp_off = geo_ncfg[8], hp_sc = geo_ncfg[9];
  const float ar_off = geo_ncfg[10], ar_sc = geo_ncfg[11];
  const float ammo_sc = geo_ncfg[12];
  const float str_max = geo_ncfg[13];
  const float clen_sc = geo_ncfg[14];
  const float cspd_sc = geo_ncfg[15];

  threadgroup float t_rho[32], t_bo[32], t_held[32], t_rate[32], t_rem[32], t_id[32], t_ex[64];
  threadgroup int c_nim[64], c_rh[64];
  threadgroup float tg_red[9];
  threadgroup float max_rho_sh, max_rate_sh, min_rem_sh;
  threadgroup int nim_tot_sh;

  // Step 1: Team CGT stats & ent_meta
  if (tid < K) {
    const ulong tb = (ulong(e) * K + tid) * 7ul;
    const float id_v = float(teams_in[tb + 0]);
    const float sc = float(teams_in[tb + 1]);
    const float held = float(teams_in[tb + 2]);
    const float rate = float(teams_in[tb + 3]);
    const float lim = max(float(teams_in[tb + 4]), 1e-6f);
    const float rem = max(lim - sc, 1e-6f);
    const float rho = (held > 0.0f) ? (rate * held / rem) : 0.0f;
    t_id[tid] = id_v; t_held[tid] = held; t_rate[tid] = rate; t_rem[tid] = rem; t_rho[tid] = rho;
    ent_meta[ulong(E) * M_ent + ulong(e) * K + tid] = int(rint(id_v));
    ent_meta[ulong(e) * M_ent + N + W_nav + J + tid] = int(rint(id_v));
  }
  if (tid < W_nav) {
    ent_meta[ulong(e) * M_ent + N + tid] = 0;
  }
  if (tid < J) {
    const ulong cb = (ulong(e) * J + tid) * 18ul;
    const float cp_cnt = max(float(carts_in[cb + 17]), 1.0f);
    const float cp_dep = max(float(carts_in[cb + 16]), 0.0f);
    const int rh = max(0, int(rint(cp_cnt - cp_dep)));
    c_rh[tid] = rh;
    c_nim[tid] = rh % 3;
    ent_meta[ulong(e) * M_ent + N + W_nav + tid] = int(rint(float(carts_in[cb + 3])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < K) {
    float bo = 0.0f;
    if (K > 1u) {
      float m = -1.0f;
      for (uint q = 0; q < K; q++) if (q != tid) m = max(m, t_rho[q]);
      bo = max(m, 0.0f);
    }
    t_bo[tid] = bo;
    const float r = t_rho[tid];
    t_ex[tid * 2u + 0u] = gf_safe_tanh(r);
    t_ex[tid * 2u + 1u] = (r - bo) / (r + bo + 1e-6f);
  }
  if (tid == 0u) {
    float mr = 1e-6f, mrt = 0.0f, mrm = 1e30f;
    for (uint k = 0; k < K; k++) {
      mr = max(mr, t_rho[k]); mrt = max(mrt, t_rate[k]); mrm = min(mrm, t_rem[k]);
    }
    max_rho_sh = mr; max_rate_sh = mrt; min_rem_sh = max(mrm, 1e-6f);
    int nt = 0;
    for (uint j = 0; j < J; j++) nt ^= c_nim[j];
    nim_tot_sh = nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 2: team_out [E*K, 64]
  for (uint k = 0; k < K; k++) {
    const ulong tb = (ulong(e) * K + k) * 7ul;
    float v = 0.0f;
    if (tid < 9u) {
      if (tid < 7u) {
        v = float(teams_in[tb + tid]);
        if (tid == 1u) v = v / max(float(teams_in[tb + 4]), 1.0f);
        else if (tid == 4u) v = 1.0f;
        else if (tid == 5u) v = 0.0f;
      } else v = t_ex[k * 2u + (tid - 7u)];
      v = asinh(v);
    }
    const float ss = gf_tg_sum((tid < 9u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 9.0f + mp_eps);
    if (tid < 64u) team_out[(ulong(e) * K + k) * 64ul + tid] = (tid < 9u) ? bfloat(v * inv) : bfloat(0.0f);
  }

  // Step 3: cart_out [E*J, 64]
  for (uint j = 0; j < J; j++) {
    const ulong cb = (ulong(e) * J + j) * 18ul;
    float v = 0.0f;
    if (tid < 22u) {
      if (tid < 18u) {
        v = float(carts_in[cb + tid]);
        const float plen = max(float(carts_in[cb + 2]), 1.0f);
        if (tid == 1u) v = v / plen;
        else if (tid == 2u) v = log(1.0f + plen / clen_sc);
        else if (tid == 4u) v = v / cspd_sc;
        else if (tid == 9u) v = has_geo ? ((v - cx) / rad) : (v / pos_fb);
        else if (tid == 10u) v = has_geo ? ((v - cy) / rad) : (v / pos_fb);
        else if (tid == 11u) v = has_geo ? ((v - cz) / rad) : (v / pos_fb);
      } else {
        const float plen = max(float(carts_in[cb + 2]), 1.0f);
        const float pfrac = clamp(float(carts_in[cb + 1]) / plen, 0.0f, 1.0f);
        const float cp_cnt = max(float(carts_in[cb + 17]), 1.0f);
        const float cp_dep = max(float(carts_in[cb + 16]), 0.0f);
        const float seg = pfrac * cp_cnt;
        if (tid == 18u) {
          const float dd = max(seg - floor(seg - 1e-6f), 0.05f);
          const float oid = float(carts_in[cb + 3]);
          int oidx = -1;
          for (uint k = 0; k < K; k++) if (oid == t_id[k]) { oidx = int(k); break; }
          float drd = 0.0f, denom = 1e-6f;
          if (oidx >= 0) {
            denom = t_rho[oidx] + t_bo[oidx] + 1e-6f;
            if (cp_dep > 0.0f && t_held[oidx] > 0.0f) drd = t_rho[oidx] / max(t_held[oidx], 1.0f);
          }
          v = gf_safe_tanh(drd / denom / dd);
        } else if (tid == 19u) {
          const float dp = max(ceil(seg + 1e-6f) - seg, 0.05f);
          const float drp = (cp_dep < cp_cnt) ? (max_rate_sh / min_rem_sh) : 0.0f;
          v = gf_safe_tanh((drp / (max_rho_sh + drp + 1e-6f)) / dp);
        } else if (tid == 20u) {
          const int nb = c_nim[j], nt = nim_tot_sh, tn = nb ^ nt;
          v = (nt != 0 && tn < nb && (nb - tn) <= c_rh[j]) ? 1.0f : 0.0f;
        } else {
          v = float(c_nim[j]) / 3.0f;
        }
      }
      v = asinh(v);
    }
    const float ss = gf_tg_sum((tid < 22u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 22.0f + mp_eps);
    if (tid < 64u) cart_out[(ulong(e) * J + j) * 64ul + tid] = (tid < 22u) ? bfloat(v * inv) : bfloat(0.0f);
  }

  // Step 4: nav_out [E*W_nav, 64] (20 active cols including committed steering mass from acc_in!)
  const float inv_n = 1.0f / float(max(1u, N));
  for (uint vn = 0; vn < W_nav; vn++) {
    const ulong nb = (ulong(e) * W_nav + vn) * 16ul;
    // Aggregate acc_in[:, vn] across bots n < N
    float rv_n = (tid < N) ? acc_in[(ulong(e) * N + tid) * A_dim + vn] : 0.0f;
    const float pos_s = gf_tg_sum((tid < N) ? max(rv_n, 0.0f) : 0.0f, lane, sg, 4u, tg_red);
    const float net_s = gf_tg_sum((tid < N) ? rv_n : 0.0f, lane, sg, 4u, tg_red);
    const float mx_v  = gf_tg_max((tid < N) ? rv_n : -1e30f, lane, sg, 4u, tg_red);
    const float mn_v  = gf_tg_min((tid < N) ? rv_n :  1e30f, lane, sg, 4u, tg_red);
    float v = 0.0f;
    if (tid < 20u) {
      if (tid < 16u) {
        v = float(nav_in[nb + tid]);
        if (tid == 0u) v = has_geo ? ((v - cx) / rad) : (v / pos_fb);
        else if (tid == 1u) v = has_geo ? ((v - cy) / rad) : (v / pos_fb);
        else if (tid == 2u) v = has_geo ? ((v - cz) / rad) : (v / pos_fb);
      } else if (tid == 16u) v = pos_s * inv_n;
      else if (tid == 17u) v = net_s * inv_n;
      else if (tid == 18u) v = mx_v;
      else v = mn_v;
      v = asinh(v);
    }
    const float ss = gf_tg_sum((tid < 20u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 20.0f + mp_eps);
    if (tid < 64u) nav_out[(ulong(e) * W_nav + vn) * 64ul + tid] = (tid < 20u) ? bfloat(v * inv) : bfloat(0.0f);
  }

  // Step 5: obs_out [E*N, 128] (96 active cols: obs(84) + r_self(4) + r_nav/r_team stats(4) + goal_kind one-hot(4))
  for (uint n = 0; n < N; n++) {
    const ulong ob = (ulong(e) * N + n) * 84ul;
    const device float* an = acc_in + (ulong(e) * N + n) * A_dim;
    const int nid = clamp(int(rint(float(obs_in[ob + 22]))), 0, int(max(1u, W_nav)) - 1);
    const int gk  = clamp(int(rint(float(obs_in[ob + 23]))), 0, 3);
    if (tid == 0u) {
      ent_meta[ulong(e) * M_ent + n] = int(rint(float(obs_in[ob + 1])));
      bot_nav_id[ulong(e) * N + n] = nid;
    }
    const float rn_v = (tid < W_nav) ? an[tid] : 0.0f;
    const float rn_ss = gf_tg_sum((tid < W_nav) ? (rn_v * rn_v) : 0.0f, lane, sg, 4u, tg_red);
    const float rn_mx = gf_tg_max((tid < W_nav) ? rn_v : -1e30f, lane, sg, 4u, tg_red);
    const float rn_mn = gf_tg_min((tid < W_nav) ? rn_v :  1e30f, lane, sg, 4u, tg_red);
    const float rt_v = (tid < K) ? an[W_nav + tid] : 0.0f;
    const float rt_ss = gf_tg_sum((tid < K) ? (rt_v * rt_v) : 0.0f, lane, sg, 4u, tg_red);

    float v = 0.0f;
    if (tid < 96u) {
      if (tid < 84u) {
        v = float(obs_in[ob + tid]);
        if (tid == 2u) v = (v - hp_off) / hp_sc;
        else if (tid == 3u) v = (v - ar_off) / ar_sc;
        else if (tid >= 4u && tid <= 9u) v = v / ammo_sc;
        else if (tid == 10u) v = has_geo ? ((v - cx) / rad) : (v / pos_fb);
        else if (tid == 11u) v = has_geo ? ((v - cy) / rad) : (v / pos_fb);
        else if (tid == 12u) v = has_geo ? ((v - cz) / rad) : (v / pos_fb);
        else if (tid >= 13u && tid <= 15u) v = v / vel_sc;
        else if (tid == 19u) v = clamp(v - float(obs_in[ob + 21]), 0.0f, str_max) / str_max;
        else if (tid == 20u || tid == 21u || tid == 22u ||
                 tid == 39u || tid == 40u || tid == 41u ||
                 tid == 54u || tid == 62u || tid == 79u || tid == 80u) v = 0.0f;
        else if (tid == 23u) v = float(gk) / 3.0f;
      } else if (tid < 88u) {
        v = an[W_nav + K + (tid - 84u)];
      } else if (tid == 88u) {
        v = sqrt(max(rn_ss / float(max(1u, W_nav)), 1e-8f));
      } else if (tid == 89u) {
        v = sqrt(max(rt_ss / float(max(1u, K)), 1e-8f));
      } else if (tid == 90u) {
        v = rn_mx;
      } else if (tid == 91u) {
        v = rn_mn;
      } else {
        v = (int(tid - 92u) == gk) ? 1.0f : 0.0f;
      }
      v = asinh(v);
    }
    const float ss = gf_tg_sum((tid < 96u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 96.0f + mp_eps);
    obs_out[(ulong(e) * N + n) * 128ul + tid] = (tid < 96u) ? bfloat(v * inv) : bfloat(0.0f);
  }
}

// ============================================================================
// 2. MG_OP_GRAPH_EMBED_CAT (1 TG per env e, 128 threads, D <= 512)
// Fuses 4-type mp_norm + type_embed lerp + equivariant afferent fold -> x_out[E*M_ent, D]
// ============================================================================
[[kernel]] void mg_graph_embed_cat(
    const device bfloat* obs_enc [[buffer(0)]],
    const device bfloat* nav_enc [[buffer(1)]],
    const device bfloat* cart_enc [[buffer(2)]],
    const device bfloat* team_enc [[buffer(3)]],
    const device bfloat* type_emb [[buffer(4)]], // [4, D] pre-normalized
    const device float* acc_in [[buffer(5)]],    // [E, N, A_dim]
    const device int* bot_nav_id [[buffer(6)]],  // [E, N]
    const device float* scalars [[buffer(7)]],   // [sc_tb, sc_ab]
    device bfloat* x_out [[buffer(8)]],          // [EM_pad, D]
    device bfloat* h_obs0_out [[buffer(9)]],     // [EN_pad, D] optional save for bwd
    device bfloat* aff_raw_out [[buffer(10)]],   // [EN_pad, D] optional save for bwd
    constant uint* u [[buffer(11)]],             // [E, N, W_nav, J, K, D, sc_tb|(sc_ab<<16), save_bwd]
    constant float* f [[buffer(12)]],            // [eps_n, eps_l]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4], D = u[5];
  const uint sc_tb = u[6] & 0xFFFFu, sc_ab = u[6] >> 16;
  const bool save_bwd = (u[7] != 0u);
  if (e >= E) return;
  const uint M_ent = N + W_nav + J + K;
  const uint A_dim = W_nav + K + 4u;
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float tb = clamp(scalars[sc_tb], 0.0f, 1.0f);
  const float ab = clamp(scalars[sc_ab], 0.0f, 1.0f);
  const float tb_inv = rsqrt((1.0f - tb) * (1.0f - tb) + tb * tb + eps_l);
  const float ab_inv = rsqrt((1.0f - ab) * (1.0f - ab) + ab * ab + eps_l);
  const uint n_el = min(4u, (D + 127u) / 128u);

  threadgroup float tg_red[9];

  // Helper lambda via inline loop for nav (type 1), cart (type 2), team (type 3)
  for (uint kind = 1u; kind <= 3u; kind++) {
    const uint cnt = (kind == 1u) ? W_nav : ((kind == 2u) ? J : K);
    const uint dst_base = (kind == 1u) ? N : ((kind == 2u) ? (N + W_nav) : (N + W_nav + J));
    const device bfloat* te = type_emb + ulong(kind) * D;
    for (uint i = 0; i < cnt; i++) {
      const device bfloat* src = (kind == 1u) ? (nav_enc + (ulong(e) * W_nav + i) * D)
                               : ((kind == 2u) ? (cart_enc + (ulong(e) * J + i) * D)
                                               : (team_enc + (ulong(e) * K + i) * D));
      float reg[4];
      float ss1 = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        float v = (c < D) ? float(src[c]) : 0.0f;
        reg[q] = v; ss1 += v * v;
      }
      const float inv1 = rsqrt(gf_tg_sum(ss1, lane, sg, 4u, tg_red) / float(D) + eps_n);
      float ss2 = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) {
          float a = reg[q] * inv1;
          float h = ((1.0f - tb) * a + tb * float(te[c])) * tb_inv;
          reg[q] = h; ss2 += h * h;
        }
      }
      const float inv2 = rsqrt(gf_tg_sum(ss2, lane, sg, 4u, tg_red) / float(D) + eps_n);
      device bfloat* dst = x_out + (ulong(e) * M_ent + dst_base + i) * D;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) dst[c] = bfloat(reg[q] * inv2);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Now embed bots (type 0) + fold equivariant nav_ctx + r_nav @ h_nav + r_team @ h_team!
  const device bfloat* te0 = type_emb;
  const device bfloat* h_nav_base = x_out + (ulong(e) * M_ent + N) * D;
  const device bfloat* h_tm_base  = x_out + (ulong(e) * M_ent + N + W_nav + J) * D;
  const float inv_sw = rsqrt(float(max(1u, W_nav)));
  const float inv_sk = rsqrt(float(max(1u, K)));

  for (uint n = 0; n < N; n++) {
    const device bfloat* src = obs_enc + (ulong(e) * N + n) * D;
    const device float* an = acc_in + (ulong(e) * N + n) * A_dim;
    const uint nid = uint(clamp(bot_nav_id[ulong(e) * N + n], 0, int(max(1u, W_nav)) - 1));

    float reg[4], aff[4];
    float ss1 = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      float v = (c < D) ? float(src[c]) : 0.0f;
      reg[q] = v; ss1 += v * v;
    }
    const float inv1 = rsqrt(gf_tg_sum(ss1, lane, sg, 4u, tg_red) / float(D) + eps_n);
    float ss2 = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) {
        float h = ((1.0f - tb) * (reg[q] * inv1) + tb * float(te0[c])) * tb_inv;
        reg[q] = h; ss2 += h * h;
      }
    }
    const float inv2 = rsqrt(gf_tg_sum(ss2, lane, sg, 4u, tg_red) / float(D) + eps_n);
    float ss_a = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) {
        reg[q] *= inv2;
        float av = float(h_nav_base[ulong(nid) * D + c]);
        float s_nav = 0.0f;
        for (uint v = 0; v < W_nav; v++) s_nav += an[v] * float(h_nav_base[ulong(v) * D + c]);
        float s_tm = 0.0f;
        for (uint k = 0; k < K; k++) s_tm += an[W_nav + k] * float(h_tm_base[ulong(k) * D + c]);
        av += s_nav * inv_sw + s_tm * inv_sk;
        aff[q] = av;
        ss_a += av * av;
      }
    }
    const float inv_a = rsqrt(gf_tg_sum(ss_a, lane, sg, 4u, tg_red) / float(D) + eps_n);
    device bfloat* dst = x_out + (ulong(e) * M_ent + n) * D;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) {
        float h0 = reg[q];
        float am = aff[q] * inv_a;
        if (save_bwd) {
          h_obs0_out[(ulong(e) * N + n) * D + c] = bfloat(h0);
          aff_raw_out[(ulong(e) * N + n) * D + c] = bfloat(aff[q]);
        }
        dst[c] = bfloat(((1.0f - ab) * h0 + ab * am) * ab_inv);
      }
    }
  }
}

// ============================================================================
// 3. MG_OP_GRAPH_EMBED_BWD (1 TG per env e, 128 threads)
// Exact backward of mg_graph_embed_cat -> d_{obs,nav,cart,team}_enc, d_type_embed, d_sc[tb, ab]
// ============================================================================
[[kernel]] void mg_graph_embed_bwd(
    const device bfloat* obs_enc [[buffer(0)]],
    const device bfloat* nav_enc [[buffer(1)]],
    const device bfloat* cart_enc [[buffer(2)]],
    const device bfloat* team_enc [[buffer(3)]],
    const device bfloat* type_emb [[buffer(4)]],
    const device bfloat* h_obs0 [[buffer(5)]],
    const device bfloat* aff_raw [[buffer(6)]],
    const device bfloat* dx_in [[buffer(7)]],
    const device float* acc_in [[buffer(8)]],
    const device int* bot_nav_id [[buffer(9)]],
    const device float* scalars [[buffer(10)]],
    device bfloat* d_obs_enc [[buffer(11)]],
    device bfloat* d_nav_enc [[buffer(12)]],
    device bfloat* d_cart_enc [[buffer(13)]],
    device bfloat* d_team_enc [[buffer(14)]],
    device float* d_type_emb [[buffer(15)]],
    device float* d_sc_grads [[buffer(16)]],
    device float* d_h_scratch [[buffer(17)]], // [E*(W_nav + K), D] f32 scratch for d_h_nav & d_h_team
    constant uint* u [[buffer(18)]],          // [E, N, W_nav, J, K, D, sc_tb|(sc_ab<<16), sg_tb|(sg_ab<<16)]
    constant float* f [[buffer(19)]],         // [eps_n, eps_l]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4], D = u[5];
  const uint sc_tb = u[6] & 0xFFFFu, sc_ab = u[6] >> 16;
  const uint sg_tb = u[7] & 0xFFFFu, sg_ab = u[7] >> 16;
  if (e >= E) return;
  const uint M_ent = N + W_nav + J + K;
  const uint A_dim = W_nav + K + 4u;
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float tb = clamp(scalars[sc_tb], 0.0f, 1.0f);
  const float ab = clamp(scalars[sc_ab], 0.0f, 1.0f);
  const float tb_d = (1.0f - tb) * (1.0f - tb) + tb * tb + eps_l;
  const float ab_d = (1.0f - ab) * (1.0f - ab) + ab * ab + eps_l;
  const float tb_inv = rsqrt(tb_d);
  const float ab_inv = rsqrt(ab_d);
  const float inv_sw = rsqrt(float(max(1u, W_nav)));
  const float inv_sk = rsqrt(float(max(1u, K)));
  const uint n_el = min(4u, (D + 127u) / 128u);

  threadgroup float tg_red[9];
  device float* dh_nav = d_h_scratch + ulong(e) * ulong(W_nav + K) * D;
  device float* dh_tm  = dh_nav + ulong(W_nav) * D;

  // Init dh_nav and dh_tm from dx_in
  for (uint v = 0; v < W_nav; v++) {
    const device bfloat* dxv = dx_in + (ulong(e) * M_ent + N + v) * D;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) dh_nav[ulong(v) * D + c] = float(dxv[c]);
    }
  }
  for (uint k = 0; k < K; k++) {
    const device bfloat* dxk = dx_in + (ulong(e) * M_ent + N + W_nav + J + k) * D;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) dh_tm[ulong(k) * D + c] = float(dxk[c]);
    }
  }

  float g_tb_acc = 0.0f, g_ab_acc = 0.0f;
  float g_te0[4] = {0,0,0,0}, g_te1[4] = {0,0,0,0}, g_te2[4] = {0,0,0,0}, g_te3[4] = {0,0,0,0};

  // Step 1: Backprop bots through ab-lerp -> d_h0, d_aff_raw -> scatter into dh_nav, dh_tm; then backprop d_h0 -> d_obs_enc
  const device bfloat* te0 = type_emb;
  for (uint n = 0; n < N; n++) {
    const device bfloat* dxn = dx_in + (ulong(e) * M_ent + n) * D;
    const device bfloat* h0p = h_obs0 + (ulong(e) * N + n) * D;
    const device bfloat* afp = aff_raw + (ulong(e) * N + n) * D;
    const device float* an = acc_in + (ulong(e) * N + n) * A_dim;
    const uint nid = uint(clamp(bot_nav_id[ulong(e) * N + n], 0, int(max(1u, W_nav)) - 1));

    float af[4], h0[4], dx[4];
    float ss_a = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      af[q] = (c < D) ? float(afp[c]) : 0.0f;
      h0[q] = (c < D) ? float(h0p[c]) : 0.0f;
      dx[q] = (c < D) ? float(dxn[c]) : 0.0f;
      ss_a += af[q] * af[q];
    }
    const float inv_a = rsqrt(gf_tg_sum(ss_a, lane, sg, 4u, tg_red) / float(D) + eps_n);
    float d_am[4], d_h0[4];
    float dot_ab = 0.0f, dot_a = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) {
        float am = af[q] * inv_a;
        float y = ((1.0f - ab) * h0[q] + ab * am) * ab_inv;
        d_h0[q] = dx[q] * (1.0f - ab) * ab_inv;
        d_am[q] = dx[q] * ab * ab_inv;
        dot_ab += dx[q] * ((am - h0[q]) * ab_inv - y * (2.0f * ab - 1.0f) / ab_d);
        dot_a += d_am[q] * am;
      }
    }
    g_ab_acc += gf_tg_sum(dot_ab, lane, sg, 4u, tg_red);
    const float ma = gf_tg_sum(dot_a, lane, sg, 4u, tg_red) / float(D);
    // Scatter d_aff_raw into dh_nav & dh_tm
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) {
        float d_af = (d_am[q] - (af[q] * inv_a) * ma) * inv_a;
        dh_nav[ulong(nid) * D + c] += d_af;
        for (uint v = 0; v < W_nav; v++) dh_nav[ulong(v) * D + c] += d_af * (an[v] * inv_sw);
        for (uint k = 0; k < K; k++) dh_tm[ulong(k) * D + c] += d_af * (an[W_nav + k] * inv_sk);
      }
    }

    // Backprop d_h0 through mp_norm(mp_lerp(mp_norm(obs_enc), te0, tb))
    const device bfloat* src = obs_enc + (ulong(e) * N + n) * D;
    float xr[4];
    float ss1 = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      xr[q] = (c < D) ? float(src[c]) : 0.0f;
      ss1 += xr[q] * xr[q];
    }
    const float inv1 = rsqrt(gf_tg_sum(ss1, lane, sg, 4u, tg_red) / float(D) + eps_n);
    float hv[4];
    float ss2 = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      hv[q] = (c < D) ? (((1.0f - tb) * (xr[q] * inv1) + tb * float(te0[c])) * tb_inv) : 0.0f;
      ss2 += hv[q] * hv[q];
    }
    const float inv2 = rsqrt(gf_tg_sum(ss2, lane, sg, 4u, tg_red) / float(D) + eps_n);
    float d2 = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) d2 += d_h0[q] * (hv[q] * inv2);
    }
    const float m2 = gf_tg_sum(d2, lane, sg, 4u, tg_red) / float(D);
    float da[4];
    float dot_tb = 0.0f, d1 = 0.0f;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) {
        float y2 = hv[q] * inv2;
        float dhv = (d_h0[q] - y2 * m2) * inv2;
        float a = xr[q] * inv1;
        float b = float(te0[c]);
        da[q] = dhv * (1.0f - tb) * tb_inv;
        g_te0[q] += dhv * tb * tb_inv;
        dot_tb += dhv * ((b - a) * tb_inv - hv[q] * (2.0f * tb - 1.0f) / tb_d);
        d1 += da[q] * a;
      }
    }
    g_tb_acc += gf_tg_sum(dot_tb, lane, sg, 4u, tg_red);
    const float m1 = gf_tg_sum(d1, lane, sg, 4u, tg_red) / float(D);
    device bfloat* dst = d_obs_enc + (ulong(e) * N + n) * D;
    for (uint q = 0; q < n_el; q++) {
      uint c = tid + q * 128u;
      if (c < D) dst[c] = bfloat((da[q] - (xr[q] * inv1) * m1) * inv1);
    }
  }

  // Step 2: Backprop nav (kind=1), cart (kind=2), team (kind=3)
  for (uint kind = 1u; kind <= 3u; kind++) {
    const uint cnt = (kind == 1u) ? W_nav : ((kind == 2u) ? J : K);
    const device bfloat* te = type_emb + ulong(kind) * D;
    for (uint i = 0; i < cnt; i++) {
      const device bfloat* src = (kind == 1u) ? (nav_enc + (ulong(e) * W_nav + i) * D)
                               : ((kind == 2u) ? (cart_enc + (ulong(e) * J + i) * D)
                                               : (team_enc + (ulong(e) * K + i) * D));
      device bfloat* dst = (kind == 1u) ? (d_nav_enc + (ulong(e) * W_nav + i) * D)
                         : ((kind == 2u) ? (d_cart_enc + (ulong(e) * J + i) * D)
                                         : (d_team_enc + (ulong(e) * K + i) * D));
      float xr[4], dy[4];
      float ss1 = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        xr[q] = (c < D) ? float(src[c]) : 0.0f;
        ss1 += xr[q] * xr[q];
        if (c < D) {
          if (kind == 1u) dy[q] = dh_nav[ulong(i) * D + c];
          else if (kind == 2u) dy[q] = float(dx_in[(ulong(e) * M_ent + N + W_nav + i) * D + c]);
          else dy[q] = dh_tm[ulong(i) * D + c];
        } else dy[q] = 0.0f;
      }
      const float inv1 = rsqrt(gf_tg_sum(ss1, lane, sg, 4u, tg_red) / float(D) + eps_n);
      float hv[4];
      float ss2 = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        hv[q] = (c < D) ? (((1.0f - tb) * (xr[q] * inv1) + tb * float(te[c])) * tb_inv) : 0.0f;
        ss2 += hv[q] * hv[q];
      }
      const float inv2 = rsqrt(gf_tg_sum(ss2, lane, sg, 4u, tg_red) / float(D) + eps_n);
      float d2 = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) d2 += dy[q] * (hv[q] * inv2);
      }
      const float m2 = gf_tg_sum(d2, lane, sg, 4u, tg_red) / float(D);
      float da[4];
      float dot_tb = 0.0f, d1 = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) {
          float dhv = (dy[q] - (hv[q] * inv2) * m2) * inv2;
          float a = xr[q] * inv1;
          float b = float(te[c]);
          da[q] = dhv * (1.0f - tb) * tb_inv;
          if (kind == 1u) g_te1[q] += dhv * tb * tb_inv;
          else if (kind == 2u) g_te2[q] += dhv * tb * tb_inv;
          else g_te3[q] += dhv * tb * tb_inv;
          dot_tb += dhv * ((b - a) * tb_inv - hv[q] * (2.0f * tb - 1.0f) / tb_d);
          d1 += da[q] * a;
        }
      }
      g_tb_acc += gf_tg_sum(dot_tb, lane, sg, 4u, tg_red);
      const float m1 = gf_tg_sum(d1, lane, sg, 4u, tg_red) / float(D);
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) dst[c] = bfloat((da[q] - (xr[q] * inv1) * m1) * inv1);
      }
    }
  }

  for (uint q = 0; q < n_el; q++) {
    uint c = tid + q * 128u;
    if (c < D) {
      gf_atomic_add_f32(&d_type_emb[0ul * D + c], g_te0[q]);
      gf_atomic_add_f32(&d_type_emb[1ul * D + c], g_te1[q]);
      gf_atomic_add_f32(&d_type_emb[2ul * D + c], g_te2[q]);
      gf_atomic_add_f32(&d_type_emb[3ul * D + c], g_te3[q]);
    }
  }
  if (tid == 0u) {
    gf_atomic_add_f32(&d_sc_grads[sg_tb], g_tb_acc * tb * (1.0f - tb));
    gf_atomic_add_f32(&d_sc_grads[sg_ab], g_ab_acc * ab * (1.0f - ab));
  }
}

// ============================================================================
// 4. MG_OP_NAV_GRAPH_MIX & BWD
//   mode 0: slice x_nav = xn[:, N:N+W_nav, :] -> [EW_pad, D]
//   mode 1: nav_diff[e,v,d] = sum_w nav_kern[v,w] * v_nav[e,w,d]
//   mode 2: xg = [mp_lerp(xn_bot, norm(nav_msg[bot_nav_id]), bn),
//                 mp_lerp(xn_nav, norm(nav_msg), bn), xn_rest]
// ============================================================================
[[kernel]] void mg_nav_graph_mix(
    const device bfloat* in0 [[buffer(0)]],      // m0: xn; m1: v_nav; m2: xn
    const device bfloat* in1 [[buffer(1)]],      // m2: nav_msg
    const device float* nav_kern [[buffer(2)]],  // m1: [W_nav, W_nav]
    const device int* bot_nav_id [[buffer(3)]],  // m2: [E, N]
    const device float* scalars [[buffer(4)]],   // m2: scalars
    device bfloat* out0 [[buffer(5)]],           // m0: x_nav; m1: nav_diff; m2: xg
    constant uint* u [[buffer(6)]],              // [E, N, W_nav, M_ent, D, mode, sc_bn, out_pad_rows]
    constant float* f [[buffer(7)]],             // [eps_n, eps_l]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], M_ent = u[3], D = u[4], mode = u[5], sc_bn = u[6], pad_rows = u[7];
  const uint row = tgp * 4u + sg;
  const uint n_el = min(8u, (D + 31u) / 32u);

  if (mode == 0u) {
    if (row >= pad_rows) return;
    device bfloat* dst = out0 + ulong(row) * D;
    if (row >= E * W_nav) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
      return;
    }
    const uint e = row / W_nav, v = row % W_nav;
    const device bfloat* src = in0 + (ulong(e) * M_ent + N + v) * D;
    for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = src[c]; }
  } else if (mode == 1u) {
    if (row >= pad_rows) return;
    device bfloat* dst = out0 + ulong(row) * D;
    if (row >= E * W_nav) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
      return;
    }
    const uint e = row / W_nav, v = row % W_nav;
    const device float* kw = nav_kern + ulong(v) * W_nav;
    const device bfloat* vn_e = in0 + ulong(e) * W_nav * D;
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) {
        float acc = 0.0f;
        for (uint w = 0; w < W_nav; w++) acc += kw[w] * float(vn_e[ulong(w) * D + c]);
        dst[c] = bfloat(acc);
      }
    }
  } else {
    if (row >= pad_rows) return;
    device bfloat* dst = out0 + ulong(row) * D;
    if (row >= E * M_ent) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
      return;
    }
    const uint e = row / M_ent, m = row % M_ent;
    const device bfloat* xr = in0 + ulong(row) * D;
    if (m >= N + W_nav) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = xr[c]; }
      return;
    }
    const uint vn = (m < N) ? uint(clamp(bot_nav_id[ulong(e) * N + m], 0, int(max(1u, W_nav)) - 1)) : (m - N);
    const device bfloat* mr = in1 + (ulong(e) * W_nav + vn) * D;
    const float eps_n = max(f[0], 1e-4f), eps_l = max(f[1], 1e-4f);
    const float bn = clamp(scalars[sc_bn], 0.0f, 1.0f);
    const float inv_l = rsqrt((1.0f - bn) * (1.0f - bn) + bn * bn + eps_l);
    float mr_r[8];
    float ss = 0.0f;
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      float v = (c < D) ? float(mr[c]) : 0.0f;
      mr_r[i] = v; ss += v * v;
    }
    const float inv_m = rsqrt(simd_sum(ss) / float(D) + eps_n);
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) dst[c] = bfloat(((1.0f - bn) * float(xr[c]) + bn * (mr_r[i] * inv_m)) * inv_l);
    }
  }
}

[[kernel]] void mg_nav_graph_mix_bwd(
    const device bfloat* in0 [[buffer(0)]],      // m0: d_x_nav; m1: d_nav_diff; m2: xn
    const device bfloat* in1 [[buffer(1)]],      // m2: nav_msg
    const device bfloat* in2 [[buffer(2)]],      // m2: d_xg
    const device float* nav_kern [[buffer(3)]],  // m1: [W_nav, W_nav]
    const device int* bot_nav_id [[buffer(4)]],  // m2: [E, N]
    const device float* scalars [[buffer(5)]],   // m2: scalars
    device bfloat* out0 [[buffer(6)]],           // m0: d_xn (+=); m1: d_v_nav; m2: d_xn
    device float* out1_f32 [[buffer(7)]],        // m2: d_nav_msg_f32 (or bf16 via separate pass? Wait: 1 TG per env avoids atomics!)
    device float* d_sc_grads [[buffer(8)]],      // m2: d_sc_grads
    constant uint* u [[buffer(9)]],              // [E, N, W_nav, M_ent, D, mode, sc_bn|(sg_bn<<16), pad_rows]
    constant float* f [[buffer(10)]],            // [eps_n, eps_l]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], M_ent = u[3], D = u[4], mode = u[5];
  const uint sc_bn = u[6] & 0xFFFFu, sg_bn = u[6] >> 16, pad_rows = u[7];

  if (mode == 0u) {
    // Accumulate d_x_nav [E*W_nav, D] into d_xn [E*M_ent, D] (4 rows per TG)
    const uint row = tgp * 4u + sg;
    if (row >= E * W_nav) return;
    const uint e = row / W_nav, v = row % W_nav;
    const device bfloat* src = in0 + ulong(row) * D;
    device bfloat* dst = out0 + (ulong(e) * M_ent + N + v) * D;
    const uint n_el = min(8u, (D + 31u) / 32u);
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) dst[c] = bfloat(float(dst[c]) + float(src[c]));
    }
  } else if (mode == 1u) {
    // Transpose geodesic diffusion: d_v_nav[e, w, d] = sum_v nav_kern[v, w] * d_nav_diff[e, v, d]
    const uint row = tgp * 4u + sg;
    if (row >= pad_rows) return;
    device bfloat* dst = out0 + ulong(row) * D;
    const uint n_el = min(8u, (D + 31u) / 32u);
    if (row >= E * W_nav) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
      return;
    }
    const uint e = row / W_nav, w = row % W_nav;
    const device bfloat* dd_e = in0 + ulong(e) * W_nav * D;
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) {
        float acc = 0.0f;
        for (uint v = 0; v < W_nav; v++) acc += nav_kern[ulong(v) * W_nav + w] * float(dd_e[ulong(v) * D + c]);
        dst[c] = bfloat(acc);
      }
    }
  } else {
    // mode == 2: 1 TG per env e (128 threads), ZERO atomics for d_nav_msg!
    // Writes out0 = d_xn [E*M_ent, D] (bf16), and casts out1_f32 as device bfloat* d_nav_msg [EW_pad, D]!
    const uint e = tgp;
    if (e >= E) return;
    device bfloat* d_nav_msg = reinterpret_cast<device bfloat*>(out1_f32);
    const float eps_n = max(f[0], 1e-4f), eps_l = max(f[1], 1e-4f);
    const float bn = clamp(scalars[sc_bn], 0.0f, 1.0f);
    const float den = (1.0f - bn) * (1.0f - bn) + bn * bn + eps_l;
    const float inv_l = rsqrt(den);
    const uint n_el = min(4u, (D + 127u) / 128u);
    threadgroup float tg_red[9];

    // Copy rest (carts + teams) directly
    for (uint m = N + W_nav; m < M_ent; m++) {
      const device bfloat* src = in2 + (ulong(e) * M_ent + m) * D;
      device bfloat* dst = out0 + (ulong(e) * M_ent + m) * D;
      for (uint q = 0; q < n_el; q++) { uint c = tid + q * 128u; if (c < D) dst[c] = src[c]; }
    }

    float g_bn_acc = 0.0f;
    // Loop over each nav node v: accumulate d_msg_norm from nav node v AND all bots with bot_nav_id == v!
    for (uint v = 0; v < W_nav; v++) {
      const device bfloat* mr = in1 + (ulong(e) * W_nav + v) * D;
      float mv[4];
      float ss = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        mv[q] = (c < D) ? float(mr[c]) : 0.0f;
        ss += mv[q] * mv[q];
      }
      const float inv_m = rsqrt(gf_tg_sum(ss, lane, sg, 4u, tg_red) / float(D) + eps_n);
      float dm_norm[4] = {0, 0, 0, 0};

      // Nav-node token m = N + v
      {
        const ulong row = ulong(e) * M_ent + N + v;
        const device bfloat* xr = in0 + row * D;
        const device bfloat* dy = in2 + row * D;
        device bfloat* dx = out0 + row * D;
        float dot_b = 0.0f;
        for (uint q = 0; q < n_el; q++) {
          uint c = tid + q * 128u;
          if (c < D) {
            float a = float(xr[c]), b = mv[q] * inv_m, g = float(dy[c]);
            float y = ((1.0f - bn) * a + bn * b) * inv_l;
            dx[c] = bfloat(g * (1.0f - bn) * inv_l);
            dm_norm[q] += g * bn * inv_l;
            dot_b += g * ((b - a) * inv_l - y * (2.0f * bn - 1.0f) / den);
          }
        }
        g_bn_acc += gf_tg_sum(dot_b, lane, sg, 4u, tg_red);
      }
      // Bots whose bot_nav_id == v
      for (uint n = 0; n < N; n++) {
        if (uint(clamp(bot_nav_id[ulong(e) * N + n], 0, int(max(1u, W_nav)) - 1)) != v) continue;
        const ulong row = ulong(e) * M_ent + n;
        const device bfloat* xr = in0 + row * D;
        const device bfloat* dy = in2 + row * D;
        device bfloat* dx = out0 + row * D;
        float dot_b = 0.0f;
        for (uint q = 0; q < n_el; q++) {
          uint c = tid + q * 128u;
          if (c < D) {
            float a = float(xr[c]), b = mv[q] * inv_m, g = float(dy[c]);
            float y = ((1.0f - bn) * a + bn * b) * inv_l;
            dx[c] = bfloat(g * (1.0f - bn) * inv_l);
            dm_norm[q] += g * bn * inv_l;
            dot_b += g * ((b - a) * inv_l - y * (2.0f * bn - 1.0f) / den);
          }
        }
        g_bn_acc += gf_tg_sum(dot_b, lane, sg, 4u, tg_red);
      }

      // Backprop dm_norm through mp_norm(nav_msg[e, v]) -> d_nav_msg[e, v]
      float dot_m = 0.0f;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) dot_m += dm_norm[q] * (mv[q] * inv_m);
      }
      const float mm = gf_tg_sum(dot_m, lane, sg, 4u, tg_red) / float(D);
      device bfloat* dmo = d_nav_msg + (ulong(e) * W_nav + v) * D;
      for (uint q = 0; q < n_el; q++) {
        uint c = tid + q * 128u;
        if (c < D) dmo[c] = bfloat((dm_norm[q] - (mv[q] * inv_m) * mm) * inv_m);
      }
    }
    if (tid == 0u) gf_atomic_add_f32(&d_sc_grads[sg_bn], g_bn_acc * bn * (1.0f - bn));
  }
}

// ============================================================================
// 5. MG_OP_DPP_GRAM & BWD (Exact FP32 Per-Team DPP Kernel B(I+B^T B)^{-1}B^T V!)
//   mode 0: slice x_bot = x_mid[:, :N, :] -> [EN_pad, D]
//   mode 1: 1 TG per (e, k), 128 threads, r_dpp <= 32, exact FP32 Cholesky in TG SRAM!
//   mode 2: merge x_mid2 = [mp_lerp(x_mid[:N], norm(dpp_raw), blend_dpp), x_mid[N:]]
// ============================================================================
[[kernel]] void mg_dpp_gram(
    const device bfloat* in0 [[buffer(0)]],      // m0: x_mid; m1: B_norm [EN_pad, R]; m2: x_mid
    const device bfloat* in1 [[buffer(1)]],      // m1: V_norm [EN_pad, D]; m2: dpp_raw [EN_pad, D]
    const device int* ent_meta [[buffer(2)]],    // m1: entity_teams + team_ids
    const device float* scalars [[buffer(3)]],   // m2: scalars
    device bfloat* out0 [[buffer(4)]],           // m0: x_bot; m1: Y_dpp [EN_pad, D]; m2: x_mid2
    constant uint* u [[buffer(5)]],              // [E, N, M_ent, K, D, R, mode, sc_bd|pad_rows]
    constant float* f [[buffer(6)]],             // [eps_n, eps_l]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], M_ent = u[2], K = u[3], D = u[4], R = u[5], mode = u[6], aux = u[7];

  if (mode == 0u) {
    const uint row = tgp * 4u + sg;
    if (row >= aux) return;
    device bfloat* dst = out0 + ulong(row) * D;
    const uint n_el = min(8u, (D + 31u) / 32u);
    if (row >= E * N) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
      return;
    }
    const uint e = row / N, n = row % N;
    const device bfloat* src = in0 + (ulong(e) * M_ent + n) * D;
    for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = src[c]; }
    return;
  }

  if (mode == 2u) {
    const uint row = tgp * 4u + sg;
    if (row >= E * M_ent) return;
    const uint e = row / M_ent, m = row % M_ent;
    const device bfloat* xr = in0 + ulong(row) * D;
    device bfloat* dst = out0 + ulong(row) * D;
    const uint n_el = min(8u, (D + 31u) / 32u);
    if (m >= N) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = xr[c]; }
      return;
    }
    const device bfloat* dr = in1 + (ulong(e) * N + m) * D;
    const float eps_n = max(f[0], 1e-4f), eps_l = max(f[1], 1e-4f);
    const float bd = clamp(scalars[aux], 0.0f, 1.0f);
    const float inv_l = rsqrt((1.0f - bd) * (1.0f - bd) + bd * bd + eps_l);
    float dv[8];
    float ss = 0.0f;
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      float v = (c < D) ? float(dr[c]) : 0.0f;
      dv[i] = v; ss += v * v;
    }
    const float inv_d = rsqrt(simd_sum(ss) / float(D) + eps_n);
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) dst[c] = bfloat(((1.0f - bd) * float(xr[c]) + bd * (dv[i] * inv_d)) * inv_l);
    }
    return;
  }

  // mode == 1: 1 TG per (e, k), 128 threads, exact FP32 Cholesky for I + Bs^T Bs (R <= 16)
  const uint ek = tgp;
  if (ek >= E * K) return;
  const uint e = ek / K, k = ek % K;
  const int my_tid = ent_meta[ulong(E) * M_ent + ulong(e) * K + k];
  const float inv_sr = rsqrt(float(max(1u, R)));

  threadgroup uint sh_idx[128];
  threadgroup uint sh_cnt;
  threadgroup float sh_B[128 * 16]; // supports N <= 128, R <= 16
  threadgroup float sh_L[16 * 16];

  if (tid == 0u) {
    uint cnt = 0u;
    for (uint n = 0; n < N; n++) {
      if (ent_meta[ulong(e) * M_ent + n] == my_tid) sh_idx[cnt++] = n;
    }
    sh_cnt = cnt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nk = sh_cnt;
  if (nk == 0u) return;

  // Load B_raw [nk, 64] (first R cols), apply mp_norm over R * inv_sr -> sh_B [nk, R]
  const float eps_n = max(f[0], 1e-4f);
  for (uint i = sg; i < nk; i += 4u) {
    const device bfloat* br = in0 + (ulong(e) * N + sh_idx[i]) * 64ul;
    const float v = (lane < R) ? float(br[lane]) : 0.0f;
    const float ss = simd_sum((lane < R) ? (v * v) : 0.0f);
    const float inv = rsqrt(ss / float(max(1u, R)) + eps_n);
    if (lane < R) sh_B[i * R + lane] = (v * inv) * inv_sr;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Build G = I + Bs^T Bs [R, R] in parallel across all 128 threads (R=16 => R*R=256!)
  for (uint pq = tid; pq < R * R; pq += 128u) {
    const uint p = pq / R, q = pq % R;
    float s = (p == q) ? 1.0f : 0.0f;
    for (uint i = 0; i < nk; i++) s += sh_B[i * R + p] * sh_B[i * R + q];
    sh_L[pq] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Exact FP32 Cholesky in-place (R <= 16: ~680 FLOPs, <0.2 us!)
  if (tid == 0u) {
    for (uint i = 0; i < R; i++) {
      for (uint j = 0; j <= i; j++) {
        float s = sh_L[i * R + j];
        for (uint p = 0; p < j; p++) s -= sh_L[i * R + p] * sh_L[j * R + p];
        if (i == j) sh_L[i * R + i] = sqrt(max(s, 1e-6f));
        else sh_L[i * R + j] = s / sh_L[j * R + j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Solve G z = Bs^T V[:, d] and write Y[idx[i], d] = Bs[i] . z for each channel d!
  for (uint d = tid; d < D; d += 128u) {
    float z[16];
    for (uint r = 0; r < R; r++) z[r] = 0.0f;
    for (uint i = 0; i < nk; i++) {
      const float vi = float(in1[(ulong(e) * N + sh_idx[i]) * D + d]);
      for (uint r = 0; r < R; r++) z[r] += sh_B[i * R + r] * vi;
    }
    // Forward solve L y = z
    for (uint i = 0; i < R; i++) {
      float s = z[i];
      for (uint p = 0; p < i; p++) s -= sh_L[i * R + p] * z[p];
      z[i] = s / sh_L[i * R + i];
    }
    // Back solve L^T x = y
    for (int i = int(R) - 1; i >= 0; i--) {
      float s = z[i];
      for (uint p = uint(i + 1); p < R; p++) s -= sh_L[p * R + uint(i)] * z[p];
      z[i] = s / sh_L[uint(i) * R + uint(i)];
    }
    for (uint i = 0; i < nk; i++) {
      float yv = 0.0f;
      for (uint r = 0; r < R; r++) yv += sh_B[i * R + r] * z[r];
      out0[(ulong(e) * N + sh_idx[i]) * D + d] = bfloat(yv);
    }
  }
}

[[kernel]] void mg_dpp_gram_bwd(
    const device bfloat* in0 [[buffer(0)]],      // m0: d_x_bot; m1: B_norm; m2: x_mid
    const device bfloat* in1 [[buffer(1)]],      // m1: V_norm; m2: dpp_raw
    const device bfloat* in2 [[buffer(2)]],      // m1: dY_dpp; m2: d_x_mid2
    const device int* ent_meta [[buffer(3)]],    // m1: ent_meta
    const device float* scalars [[buffer(4)]],   // m2: scalars
    device bfloat* out0 [[buffer(5)]],           // m0: d_x_mid (+=); m1: dB_norm; m2: d_x_mid
    device bfloat* out1 [[buffer(6)]],           // m1: dV_norm; m2: d_dpp_raw
    device float* d_sc_grads [[buffer(7)]],      // m2: d_sc_grads
    constant uint* u [[buffer(8)]],              // [E, N, M_ent, K, D, R, mode, sc_bd|(sg_bd<<16)]
    constant float* f [[buffer(9)]],             // [eps_n, eps_l]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], M_ent = u[2], K = u[3], D = u[4], R = u[5], mode = u[6];
  const uint sc_bd = u[7] & 0xFFFFu, sg_bd = u[7] >> 16;

  if (mode == 0u) {
    const uint row = tgp * 4u + sg;
    if (row >= E * N) return;
    const uint e = row / N, n = row % N;
    const device bfloat* src = in0 + ulong(row) * D;
    device bfloat* dst = out0 + (ulong(e) * M_ent + n) * D;
    const uint n_el = min(8u, (D + 31u) / 32u);
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) dst[c] = bfloat(float(dst[c]) + float(src[c]));
    }
    return;
  }

  if (mode == 2u) {
    const uint row = tgp * 4u + sg;
    if (row >= E * M_ent) return;
    const uint e = row / M_ent, m = row % M_ent;
    const device bfloat* xr = in0 + ulong(row) * D;
    const device bfloat* dy = in2 + ulong(row) * D;
    device bfloat* dx = out0 + ulong(row) * D;
    const uint n_el = min(8u, (D + 31u) / 32u);
    if (m >= N) {
      for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dx[c] = dy[c]; }
      return;
    }
    const device bfloat* dr = in1 + (ulong(e) * N + m) * D;
    device bfloat* ddr = out1 + (ulong(e) * N + m) * D;
    const float eps_n = max(f[0], 1e-4f), eps_l = max(f[1], 1e-4f);
    const float bd = clamp(scalars[sc_bd], 0.0f, 1.0f);
    const float den = (1.0f - bd) * (1.0f - bd) + bd * bd + eps_l;
    const float inv_l = rsqrt(den);
    float dv[8];
    float ss = 0.0f;
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      float v = (c < D) ? float(dr[c]) : 0.0f;
      dv[i] = v; ss += v * v;
    }
    const float inv_d = rsqrt(simd_sum(ss) / float(D) + eps_n);
    float db_norm[8];
    float dot_bd = 0.0f, dot_d = 0.0f;
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) {
        float a = float(xr[c]), b = dv[i] * inv_d, g = float(dy[c]);
        float y = ((1.0f - bd) * a + bd * b) * inv_l;
        dx[c] = bfloat(g * (1.0f - bd) * inv_l);
        db_norm[i] = g * bd * inv_l;
        dot_bd += g * ((b - a) * inv_l - y * (2.0f * bd - 1.0f) / den);
        dot_d += db_norm[i] * b;
      }
    }
    const float g_bd = simd_sum(dot_bd);
    const float md = simd_sum(dot_d) / float(D);
    if (lane == 0u) gf_atomic_add_f32(&d_sc_grads[sg_bd], g_bd * bd * (1.0f - bd));
    for (uint i = 0; i < n_el; i++) {
      uint c = lane + i * 32u;
      if (c < D) ddr[c] = bfloat((db_norm[i] - (dv[i] * inv_d) * md) * inv_d);
    }
    return;
  }

  // mode == 1: Exact analytical DPP backward!
  // dV = Bs W,  dBs = dY Z^T + V W^T - Bs (Z W^T + W Z^T)
  const uint ek = tgp;
  if (ek >= E * K) return;
  const uint e = ek / K, k = ek % K;
  const int my_tid = ent_meta[ulong(E) * M_ent + ulong(e) * K + k];
  const float inv_sr = rsqrt(float(max(1u, R)));

  threadgroup uint sh_idx[128];
  threadgroup uint sh_cnt;
  threadgroup float sh_B[128 * 16];
  threadgroup float sh_L[16 * 16];
  threadgroup float sh_M[16 * 16]; // Z W^T + W Z^T
  threadgroup float sh_dB[128 * 16];

  if (tid == 0u) {
    uint cnt = 0u;
    for (uint n = 0; n < N; n++) if (ent_meta[ulong(e) * M_ent + n] == my_tid) sh_idx[cnt++] = n;
    sh_cnt = cnt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nk = sh_cnt;
  if (nk == 0u) return;

  const float eps_n = max(f[0], 1e-4f);
  threadgroup float sh_invB[128];
  for (uint i = sg; i < nk; i += 4u) {
    const device bfloat* br = in0 + (ulong(e) * N + sh_idx[i]) * 64ul;
    const float v = (lane < R) ? float(br[lane]) : 0.0f;
    const float ss = simd_sum((lane < R) ? (v * v) : 0.0f);
    const float inv = rsqrt(ss / float(max(1u, R)) + eps_n);
    if (lane == 0u) sh_invB[i] = inv;
    if (lane < R) {
      sh_B[i * R + lane] = (v * inv) * inv_sr;
      sh_dB[i * R + lane] = 0.0f;
    }
  }
  for (uint pq = tid; pq < R * R; pq += 128u) {
    sh_M[pq] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint pq = tid; pq < R * R; pq += 128u) {
    const uint p = pq / R, q = pq % R;
    float s = (p == q) ? 1.0f : 0.0f;
    for (uint i = 0; i < nk; i++) s += sh_B[i * R + p] * sh_B[i * R + q];
    sh_L[pq] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    for (uint i = 0; i < R; i++) {
      for (uint j = 0; j <= i; j++) {
        float s = sh_L[i * R + j];
        for (uint p = 0; p < j; p++) s -= sh_L[i * R + p] * sh_L[j * R + p];
        if (i == j) sh_L[i * R + i] = sqrt(max(s, 1e-6f));
        else sh_L[i * R + j] = s / sh_L[j * R + j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Store Z[R, D] and W[R, D] in TG SRAM? Wait: R=16, D=256 => 16*256*4 = 16 KB!
  // Actually we only need 8 KB if we do Z/W in two tiny passes or store sh_Z[16*256] (16KB) — wait:
  // sh_B(8KB) + sh_dB(8KB) + sh_Z(16KB) = 32KB (right at TG limit!).
  // Even better: store sh_Z[16*64] per chunk of 64 channels (only 4KB!) -> ZERO memory pressure & exact!
  threadgroup float sh_Zc[16 * 64];
  threadgroup float sh_Wc[16 * 64];

  for (uint d0 = 0; d0 < D; d0 += 64u) {
    const uint d_len = min(64u, D - d0);
    if (tid < d_len) {
      const uint d = d0 + tid;
      float z[16], w[16];
      for (uint r = 0; r < R; r++) { z[r] = 0.0f; w[r] = 0.0f; }
      for (uint i = 0; i < nk; i++) {
        const ulong off = (ulong(e) * N + sh_idx[i]) * D + d;
        const float vi = float(in1[off]), dyi = float(in2[off]);
        for (uint r = 0; r < R; r++) {
          const float br = sh_B[i * R + r];
          z[r] += br * vi;
          w[r] += br * dyi;
        }
      }
      for (uint i = 0; i < R; i++) {
        float sz = z[i], sw = w[i];
        for (uint p = 0; p < i; p++) { sz -= sh_L[i * R + p] * z[p]; sw -= sh_L[i * R + p] * w[p]; }
        const float lii = sh_L[i * R + i];
        z[i] = sz / lii; w[i] = sw / lii;
      }
      for (int i = int(R) - 1; i >= 0; i--) {
        float sz = z[i], sw = w[i];
        for (uint p = uint(i + 1); p < R; p++) { sz -= sh_L[p * R + uint(i)] * z[p]; sw -= sh_L[p * R + uint(i)] * w[p]; }
        const float lii = sh_L[uint(i) * R + uint(i)];
        z[i] = sz / lii; w[i] = sw / lii;
      }
      for (uint r = 0; r < R; r++) {
        sh_Zc[r * 64u + tid] = z[r];
        sh_Wc[r * 64u + tid] = w[r];
      }
      // Write dV[idx[i], d] = Bs[i] . w
      for (uint i = 0; i < nk; i++) {
        float dv = 0.0f;
        for (uint r = 0; r < R; r++) dv += sh_B[i * R + r] * w[r];
        out1[(ulong(e) * N + sh_idx[i]) * D + d] = bfloat(dv);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate sh_M[p, q] += sum_{dc} (Z[p,dc]*W[q,dc] + W[p,dc]*Z[q,dc])
    for (uint pq = tid; pq < R * R; pq += 128u) {
      const uint p = pq / R, q = pq % R;
      float acc = 0.0f;
      for (uint dc = 0; dc < d_len; dc++) {
        acc += sh_Zc[p * 64u + dc] * sh_Wc[q * 64u + dc] + sh_Wc[p * 64u + dc] * sh_Zc[q * 64u + dc];
      }
      sh_M[pq] += acc;
    }
    // Accumulate sh_dB[i, r] += sum_{dc} (dY[i, d]*Z[r, dc] + V[i, d]*W[r, dc])
    for (uint idx = tid; idx < nk * R; idx += 128u) {
      const uint i = idx / R, r = idx % R;
      const ulong base = (ulong(e) * N + sh_idx[i]) * D + d0;
      float acc = 0.0f;
      for (uint dc = 0; dc < d_len; dc++) {
        acc += float(in2[base + dc]) * sh_Zc[r * 64u + dc] + float(in1[base + dc]) * sh_Wc[r * 64u + dc];
      }
      sh_dB[idx] += acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Finalize dB_norm -> mp_norm_bwd(B_raw) -> out0 [EN_pad, 64] (cols R..63 = 0)
  for (uint i = sg; i < nk; i += 4u) {
    const ulong row = ulong(e) * N + sh_idx[i];
    device bfloat* dst = out0 + row * 64ul;
    float d_bn = 0.0f, bn = 0.0f;
    if (lane < R) {
      float sub = 0.0f;
      for (uint p = 0; p < R; p++) sub += sh_B[i * R + p] * sh_M[p * R + lane];
      d_bn = (sh_dB[i * R + lane] - sub) * inv_sr;
      bn = sh_B[i * R + lane] * sqrt(float(max(1u, R)));
    }
    const float dot = simd_sum((lane < R) ? (d_bn * bn) : 0.0f) / float(max(1u, R));
    const float inv = sh_invB[i];
    dst[lane] = (lane < R) ? bfloat((d_bn - bn * dot) * inv) : bfloat(0.0f);
    dst[32u + lane] = bfloat(0.0f);
  }
}

// ============================================================================
// 6. MG_OP_GRAPH_SLICE_NORM & BWD
// Slices mp_norm(x_final) -> H_bot[EN_pad, D], H_nav[EW_pad, D], H_team[EK_pad, D]
// ============================================================================
[[kernel]] void mg_graph_slice_norm(
    const device bfloat* x [[buffer(0)]],
    device bfloat* h_bot [[buffer(1)]],
    device bfloat* h_nav [[buffer(2)]],
    device bfloat* h_team [[buffer(3)]],
    constant uint* u [[buffer(4)]], // [E, N, W_nav, J, K, D]
    constant float* f [[buffer(5)]], // [eps_n]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4], D = u[5];
  const uint M_ent = N + W_nav + J + K;
  const uint row = tgp * 4u + sg;
  if (row >= E * M_ent) return;
  const uint e = row / M_ent, m = row % M_ent;
  if (m >= N + W_nav && m < N + W_nav + J) return; // carts not needed by heads
  const float eps = max(f[0], 1e-4f);
  const device bfloat* xr = x + ulong(row) * D;
  device bfloat* dst = (m < N) ? (h_bot + (ulong(e) * N + m) * D)
                     : ((m < N + W_nav) ? (h_nav + (ulong(e) * W_nav + (m - N)) * D)
                                        : (h_team + (ulong(e) * K + (m - N - W_nav - J)) * D));
  float reg[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    uint c = lane + i * 32u;
    float v = (c < D) ? float(xr[c]) : 0.0f;
    reg[i] = v; ss += v * v;
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  for (uint i = 0; i < n_el; i++) {
    uint c = lane + i * 32u;
    if (c < D) dst[c] = bfloat(reg[i] * inv);
  }
}

[[kernel]] void mg_graph_slice_bwd(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* dh_bot [[buffer(1)]],
    const device bfloat* dh_nav [[buffer(2)]],
    const device bfloat* dh_team [[buffer(3)]],
    device bfloat* dx [[buffer(4)]],
    constant uint* u [[buffer(5)]], // [E, N, W_nav, J, K, D, EM_pad]
    constant float* f [[buffer(6)]], // [eps_n]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4], D = u[5], EM_pad = u[6];
  const uint M_ent = N + W_nav + J + K;
  const uint row = tgp * 4u + sg;
  if (row >= EM_pad) return;
  device bfloat* dst = dx + ulong(row) * D;
  const uint n_el = min(8u, (D + 31u) / 32u);
  if (row >= E * M_ent) {
    for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
    return;
  }
  const uint e = row / M_ent, m = row % M_ent;
  if (m >= N + W_nav && m < N + W_nav + J) {
    for (uint i = 0; i < n_el; i++) { uint c = lane + i * 32u; if (c < D) dst[c] = bfloat(0.0f); }
    return;
  }
  const float eps = max(f[0], 1e-4f);
  const device bfloat* xr = x + ulong(row) * D;
  const device bfloat* dy = (m < N) ? (dh_bot + (ulong(e) * N + m) * D)
                          : ((m < N + W_nav) ? (dh_nav + (ulong(e) * W_nav + (m - N)) * D)
                                             : (dh_team + (ulong(e) * K + (m - N - W_nav - J)) * D));
  float xv[8], gv[8];
  float ss = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    uint c = lane + i * 32u;
    xv[i] = (c < D) ? float(xr[c]) : 0.0f;
    gv[i] = (c < D) ? float(dy[c]) : 0.0f;
    ss += xv[i] * xv[i];
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  float dot = 0.0f;
  for (uint i = 0; i < n_el; i++) dot += gv[i] * (xv[i] * inv);
  const float md = simd_sum(dot) / float(D);
  for (uint i = 0; i < n_el; i++) {
    uint c = lane + i * 32u;
    if (c < D) dst[c] = bfloat((gv[i] - (xv[i] * inv) * md) * inv);
  }
}

// ============================================================================
// 7. MG_OP_FLOW_JET_SAMPLE (1 TG per env e, 128 threads, 1 Backbone NFE!)
// Integrates S_heun exact 2nd-order Heun/RK2 steps purely in TG SRAM (exact O(h^2)
// with full material/convective derivative!) & lifts u_vel = out_scale * B_H @ z.
// ============================================================================
[[kernel]] void mg_flow_jet_sample(
    const device bfloat* c_bot [[buffer(0)]],      // [EN_pad, 64]
    const device bfloat* c_team [[buffer(1)]],     // [EK_pad, 64]
    const device bfloat* B_nav_raw [[buffer(2)]],  // [EW_pad, 64]
    const device bfloat* B_tm_raw [[buffer(3)]],   // [EK_pad, 64]
    const device float* W_b_self [[buffer(4)]],    // [4, 32] f32
    const device float* fj_params [[buffer(5)]],   // packed f32: W1[128*104], b1[128], W2[32*64], b2[32]
    const device float* eps_in [[buffer(6)]],      // [E*N, 32] f32
    const device bfloat* val_raw [[buffer(7)]],    // [EN_pad, 64] (cols 0,1 = V_W, V_L raw)
    const device int* ent_meta [[buffer(8)]],      // [E*(M_ent + K)]
    const device float* hp [[buffer(9)]],          // [val_gain, val_b0, val_b1, u_clip, inv_silu, eps_n, eps_u, out_scale]
    device float* u_vel_out [[buffer(10)]],        // [E*N, A_dim]
    device float* u_lat_out [[buffer(11)]],        // [E*N, 32]
    device float* val_out [[buffer(12)]],          // [E*N, 2]
    constant uint* u [[buffer(13)]],               // [E, N, W_nav, J, K, S_heun]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4], S_heun = max(1u, u[5]);
  if (e >= E) return;
  const uint M_ent = N + W_nav + J + K;
  const uint A_dim = W_nav + K + 4u;
  const float val_g = hp[0], val_b0 = hp[1], val_b1 = hp[2];
  const float u_clip = hp[3], inv_silu = hp[4];
  const float eps_n = max(hp[5], 1e-4f), eps_u = max(hp[6], 1e-6f);
  const float out_scale = hp[7];

  const device float* W1 = fj_params;                       // [128, 104]
  const device float* b1 = W1 + 128u * 104u;                // [128]
  const device float* W2 = b1 + 128u;                       // [32, 64]
  const device float* b2 = W2 + 32u * 64u;                  // [32]

  threadgroup float sh_B[72 * 32];
  threadgroup float sh_inp[40];       // [z_fed(32), tau_emb(8)] only!
  threadgroup float sh_cf[64];
  threadgroup float sh_base_gu[128];  // Precomputed b1 + W1[:, 40:104] @ c_flow ONCE per bot (2.6x faster Heun!)
  threadgroup float sh_gu[128];
  threadgroup float sh_hid[64];
  threadgroup float sh_z[32];
  threadgroup float sh_v1[32];
  threadgroup float tg_red[9];

  // Each simdgroup normalizes rows of B_H (1 simdgroup = 32 lanes = exact d_e=32; stride 64 on B_nav/B_tm!)
  for (uint a = sg; a < A_dim; a += 4u) {
    float v = 0.0f;
    if (a < W_nav) v = float(B_nav_raw[(ulong(e) * W_nav + a) * 64ul + lane]);
    else if (a < W_nav + K) v = float(B_tm_raw[(ulong(e) * K + (a - W_nav)) * 64ul + lane]);
    else v = W_b_self[(a - W_nav - K) * 32u + lane];
    const float inv = rsqrt(simd_sum(v * v) + eps_u);
    sh_B[a * 32u + lane] = v * inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float h = 1.0f / float(S_heun);
  const float v_max = 3.0f;
  const float inv_vmax = 1.0f / v_max;
  for (uint n = 0; n < N; n++) {
    const uint t0 = uint(clamp(ent_meta[ulong(e) * M_ent + n] - 1, 0, int(max(1u, K)) - 1));
    if (tid == 0u) {
      const ulong vb = (ulong(e) * N + n) * 64ul;
      val_out[(ulong(e) * N + n) * 2ul + 0u] = val_g * float(val_raw[vb + 0u]) + val_b0;
      val_out[(ulong(e) * N + n) * 2ul + 1u] = val_g * float(val_raw[vb + 1u]) + val_b1;
    }
    if (out_scale == 0.0f) {
      if (tid < 32u) u_lat_out[(ulong(e) * N + n) * 32ul + tid] = 0.0f;
      if (tid < A_dim) u_vel_out[(ulong(e) * N + n) * A_dim + tid] = 0.0f;
      continue;
    }
    float cv = 0.0f;
    if (tid < 64u) {
      cv = float(c_bot[(ulong(e) * N + n) * 64ul + tid]) + float(c_team[(ulong(e) * K + t0) * 64ul + tid]);
    }
    const float c_inv = rsqrt(gf_tg_sum((tid < 64u) ? (cv * cv) : 0.0f, lane, sg, 4u, tg_red) / 64.0f + eps_n);
    if (tid < 64u) sh_cf[tid] = cv * c_inv;
    if (tid < 32u) sh_z[tid] = eps_in[(ulong(e) * N + n) * 32ul + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Precompute b1 + W1[:, 40:104] @ c_flow ONCE per bot outside the Heun loop!
    {
      const device float* w1r = W1 + ulong(tid) * 104ul;
      float acc_c = b1[tid];
      for (uint c = 0; c < 64u; c++) acc_c += w1r[40u + c] * sh_cf[c];
      sh_base_gu[tid] = acc_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S_heun exact 2nd-order Heun (RK2) steps with radially bounded input z_fed & bounded velocity v = v_max*tanh(v_raw/v_max)!
    for (uint s = 0; s < S_heun; s++) {
      const float tau1 = 1.0f - float(s) * h;
      const float tau2 = max(0.0f, tau1 - h);

      // Stage 1: z_fed = z * rsqrt(1 + mean(z^2)/4.0) (prevents SwiGLU quadratic blowup!)
      {
        const float zv = (tid < 32u) ? sh_z[tid] : 0.0f;
        const float z_ss = gf_tg_sum((tid < 32u) ? (zv * zv) : 0.0f, lane, sg, 4u, tg_red);
        const float z_sc = rsqrt(1.0f + (z_ss / 32.0f) * 0.25f);
        if (tid < 32u) sh_inp[tid] = zv * z_sc;
      }
      if (tid == 0u) {
        float te[8]; gf_tau_embed(tau1, te);
        for (uint q = 0; q < 8u; q++) sh_inp[32u + q] = te[q];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      {
        const device float* w1r = W1 + ulong(tid) * 104ul;
        float acc1 = sh_base_gu[tid];
        for (uint c = 0; c < 40u; c++) acc1 += w1r[c] * sh_inp[c];
        sh_gu[tid] = acc1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 64u) {
        const float g = sh_gu[tid];
        const float sig = 1.0f / (1.0f + exp(-clamp(g, -20.0f, 20.0f)));
        sh_hid[tid] = (g * sig * inv_silu) * sh_gu[64u + tid];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 32u) {
        const device float* w2r = W2 + ulong(tid) * 64ul;
        float acc2 = b2[tid];
        for (uint c = 0; c < 64u; c++) acc2 += w2r[c] * sh_hid[c];
        const float v1 = v_max * gf_safe_tanh(acc2 * inv_vmax);
        sh_v1[tid] = v1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Stage 2: z_pred = z + h * v1 -> z_fed2
      {
        const float zp = (tid < 32u) ? (sh_z[tid] + h * sh_v1[tid]) : 0.0f;
        const float z_ss = gf_tg_sum((tid < 32u) ? (zp * zp) : 0.0f, lane, sg, 4u, tg_red);
        const float z_sc = rsqrt(1.0f + (z_ss / 32.0f) * 0.25f);
        if (tid < 32u) sh_inp[tid] = zp * z_sc;
      }
      if (tid == 0u) {
        float te[8]; gf_tau_embed(tau2, te);
        for (uint q = 0; q < 8u; q++) sh_inp[32u + q] = te[q];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      {
        const device float* w1r = W1 + ulong(tid) * 104ul;
        float acc1 = sh_base_gu[tid];
        for (uint c = 0; c < 40u; c++) acc1 += w1r[c] * sh_inp[c];
        sh_gu[tid] = acc1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 64u) {
        const float g = sh_gu[tid];
        const float sig = 1.0f / (1.0f + exp(-clamp(g, -20.0f, 20.0f)));
        sh_hid[tid] = (g * sig * inv_silu) * sh_gu[64u + tid];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 32u) {
        const device float* w2r = W2 + ulong(tid) * 64ul;
        float acc2 = b2[tid];
        for (uint c = 0; c < 64u; c++) acc2 += w2r[c] * sh_hid[c];
        const float v2 = v_max * gf_safe_tanh(acc2 * inv_vmax);
        sh_z[tid] += (0.5f * h) * (sh_v1[tid] + v2);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 32u) u_lat_out[(ulong(e) * N + n) * 32ul + tid] = sh_z[tid];
    if (tid < A_dim) {
      float uv = 0.0f;
      if (tid != W_nav + t0) {
        for (uint d = 0; d < 32u; d++) uv += sh_B[tid * 32u + d] * sh_z[d];
        uv = clamp(out_scale * uv, -u_clip, u_clip);
      }
      u_vel_out[(ulong(e) * N + n) * A_dim + tid] = uv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ============================================================================
// 8. MG_OP_FLOW_JET_PREP (1 TG per env e, 128 threads, training)
// Computes B_H [E, A_dim, 32], 32x32 Cholesky ridge u_lat_targ [E*N, 32],
// packs inp_fj [(5*E*N)_pad, 128] (col 104 = 1.0 for bias!) and z_t [4*E*N, 32]
// ============================================================================
[[kernel]] void mg_flow_jet_prep(
    const device bfloat* c_bot [[buffer(0)]],      // [EN_pad, 64]
    const device bfloat* c_team [[buffer(1)]],     // [EK_pad, 64]
    const device bfloat* B_nav_raw [[buffer(2)]],  // [EW_pad, 32]
    const device bfloat* B_tm_raw [[buffer(3)]],   // [EK_pad, 32]
    const device float* W_b_self [[buffer(4)]],    // [4, 32]
    const device float* u_targ [[buffer(5)]],      // [E*N, A_dim]
    const device float* eps_train [[buffer(6)]],   // [4*E*N, 32]
    const device int* ent_meta [[buffer(7)]],      // [E*(M_ent + K)]
    const device float* taus [[buffer(8)]],        // [4]
    device float* B_H_out [[buffer(9)]],           // [E, A_dim, 32]
    device float* u_lat_out [[buffer(10)]],        // [E*N, 32]
    device bfloat* inp_fj_out [[buffer(11)]],      // [M_fj_pad, 128] (bf16)
    device float* z_t_out [[buffer(12)]],          // [4*E*N, 32]
    constant uint* u [[buffer(13)]],               // [E, N, W_nav, J, K, M_fj_pad]
    constant float* f [[buffer(14)]],              // [eps_n, eps_u]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], J = u[3], K = u[4];
  if (e >= E) return;
  const uint M_ent = N + W_nav + J + K;
  const uint A_dim = W_nav + K + 4u;
  const float eps_n = max(f[0], 1e-4f), eps_u = max(f[1], 1e-6f);

  threadgroup float sh_B[72 * 32];
  threadgroup float sh_L[32 * 32];
  threadgroup float sh_cf[64];
  threadgroup float sh_ul[32];
  threadgroup float tg_red[9];

  // 1. Unit-normalize B_H [A_dim, 32] -> sh_B & B_H_out (stride 64 on B_nav_raw & B_tm_raw!)
  for (uint a = sg; a < A_dim; a += 4u) {
    float v = 0.0f;
    if (a < W_nav) v = float(B_nav_raw[(ulong(e) * W_nav + a) * 64ul + lane]);
    else if (a < W_nav + K) v = float(B_tm_raw[(ulong(e) * K + (a - W_nav)) * 64ul + lane]);
    else v = W_b_self[(a - W_nav - K) * 32u + lane];
    const float bv = v * rsqrt(simd_sum(v * v) + eps_u);
    sh_B[a * 32u + lane] = bv;
    B_H_out[(ulong(e) * A_dim + a) * 32ul + lane] = bv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Build G_B = B_H^T B_H + 0.02 I_32 [32, 32] in parallel (tight ridge + exact Sherman-Morrison own-team mask!)
  for (uint pq = tid; pq < 1024u; pq += 128u) {
    const uint p = pq / 32u, q = pq % 32u;
    float s = (p == q) ? 0.02f : 0.0f;
    for (uint a = 0; a < A_dim; a++) s += sh_B[a * 32u + p] * sh_B[a * 32u + q];
    sh_L[pq] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Exact 32x32 FP32 Cholesky (10k FLOPs -> ~2 us!)
  if (tid == 0u) {
    for (uint i = 0; i < 32u; i++) {
      for (uint j = 0; j <= i; j++) {
        float s = sh_L[i * 32u + j];
        for (uint p = 0; p < j; p++) s -= sh_L[i * 32u + p] * sh_L[j * 32u + p];
        if (i == j) sh_L[i * 32u + i] = sqrt(max(s, 1e-6f));
        else sh_L[i * 32u + j] = s / sh_L[j * 32u + j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. For each bot n < N: solve exact own-team-masked (B^T M B + 0.02 I)^{-1} B^T M u_t via Sherman-Morrison!
  //    Let b = B[W_nav + t0], G_m = G_B - b b^T. Given x0 = G_B^{-1} rhs and y = G_B^{-1} b:
  //    G_m^{-1} rhs = x0 + y * (b^T x0) / max(1 - b^T y, 1e-4)  (EXACT!)
  threadgroup float sh_sm_y[32];
  const ulong EN = ulong(E) * ulong(N);
  for (uint n = 0; n < N; n++) {
    const ulong en_idx = ulong(e) * N + n;
    const uint t0 = uint(clamp(ent_meta[ulong(e) * M_ent + n] - 1, 0, int(max(1u, K)) - 1));
    const device float* ut = u_targ + en_idx * A_dim;

    if (tid < 32u) {
      float s = 0.0f;
      for (uint a = 0; a < A_dim; a++) {
        if (a != W_nav + t0) s += sh_B[a * 32u + tid] * ut[a];
      }
      sh_ul[tid] = s;
      sh_sm_y[tid] = sh_B[(W_nav + t0) * 32u + tid];
    }
    float cv = 0.0f;
    if (tid < 64u) {
      cv = float(c_bot[en_idx * 64ul + tid]) + float(c_team[(ulong(e) * K + t0) * 64ul + tid]);
    }
    const float c_inv = rsqrt(gf_tg_sum((tid < 64u) ? (cv * cv) : 0.0f, lane, sg, 4u, tg_red) / 64.0f + eps_n);
    if (tid < 64u) sh_cf[tid] = cv * c_inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
      // Solve L L^T x0 = rhs and L L^T y = b_own simultaneously
      for (uint i = 0; i < 32u; i++) {
        float sx = sh_ul[i], sy = sh_sm_y[i];
        for (uint p = 0; p < i; p++) {
          const float lip = sh_L[i * 32u + p];
          sx -= lip * sh_ul[p]; sy -= lip * sh_sm_y[p];
        }
        const float lii = sh_L[i * 32u + i];
        sh_ul[i] = sx / lii; sh_sm_y[i] = sy / lii;
      }
      for (int i = 31; i >= 0; i--) {
        float sx = sh_ul[i], sy = sh_sm_y[i];
        for (uint p = uint(i + 1); p < 32u; p++) {
          const float lpi = sh_L[p * 32u + uint(i)];
          sx -= lpi * sh_ul[p]; sy -= lpi * sh_sm_y[p];
        }
        const float lii = sh_L[uint(i) * 32u + uint(i)];
        sh_ul[i] = sx / lii; sh_sm_y[i] = sy / lii;
      }
      float bx = 0.0f, by = 0.0f;
      for (uint i = 0; i < 32u; i++) {
        const float bi = sh_B[(W_nav + t0) * 32u + i];
        bx += bi * sh_ul[i];
        by += bi * sh_sm_y[i];
      }
      const float coef = bx / max(1.0f - by, 1e-4f);
      for (uint i = 0; i < 32u; i++) sh_ul[i] += coef * sh_sm_y[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) u_lat_out[en_idx * 32ul + tid] = sh_ul[tid];

    // Pack 4 unbiased CFM rows (t = 0..3) with radially bounded z_fed = z_t * rsqrt(1 + mean(z_t^2)/4.0)!
    for (uint t = 0; t < 4u; t++) {
      const ulong row = ulong(t) * EN + en_idx;
      const float tau_v = taus[t];
      float te[8];
      gf_tau_embed(tau_v, te);
      const float zt_v = (tid < 32u) ? ((1.0f - tau_v) * sh_ul[tid] + tau_v * eps_train[row * 32ul + tid]) : 0.0f;
      const float z_ss = gf_tg_sum((tid < 32u) ? (zt_v * zt_v) : 0.0f, lane, sg, 4u, tg_red);
      const float z_sc = rsqrt(1.0f + (z_ss / 32.0f) * 0.25f);
      float val = 0.0f;
      if (tid < 32u) {
        z_t_out[row * 32ul + tid] = zt_v;
        val = zt_v * z_sc;
      } else if (tid < 40u) {
        val = te[tid - 32u];
      } else if (tid < 104u) {
        val = sh_cf[tid - 40u];
      } else if (tid == 104u) {
        val = 1.0f;
      }
      inp_fj_out[row * 128ul + tid] = bfloat(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ============================================================================
// 9. MG_OP_FLOW_JET_LOSS_BWD (1 TG per env e, 128 threads, training)
// Pure Unbiased Advantage-Weighted Boltzmann Flow Matching + Winnie/Lou + Dynamics!
// (Stock-bot anchor folded into advantage BEFORE ESS-targeted Boltzmann weighting!)
// ============================================================================
[[kernel]] void mg_flow_jet_loss_bwd(
    const device bfloat* vj_raw [[buffer(0)]],      // [M_fj_pad, 64] from NAX fj_w2 (cols 0..31 active)
    const device float* fj_b2 [[buffer(1)]],        // [32] f32
    const device float* B_H [[buffer(2)]],          // [E, A_dim, 32]
    const device bfloat* B_nav_raw [[buffer(3)]],   // [EW_pad, 64]
    const device bfloat* B_tm_raw [[buffer(4)]],    // [EK_pad, 64]
    const device float* W_b_self [[buffer(5)]],     // [4, 32]
    const device float* z_t [[buffer(6)]],          // [4*E*N, 32]
    const device float* eps_train [[buffer(7)]],    // [4*E*N, 32]
    const device float* u_targ [[buffer(8)]],       // [E*N, A_dim]
    const device bfloat* val_drift [[buffer(9)]],   // [EN_pad, 64] (col 0=VW, 1=VL, 2=drift)
    const device float* u_lat_targ [[buffer(10)]],  // [E*N, 32]
    const device float* dyn_act_w [[buffer(11)]],   // [32] unit-normalized f32
    const device float* targ_f [[buffer(12)]],      // [E*N, 6]: w_boltz, gae_ret, obs_drho, am, wm, t0
    const device float* lcfg [[buffer(13)]],        // 20 floats
    device bfloat* d_vj_raw [[buffer(14)]],         // [M_fj_pad, 64] (cols 0..31 active, 32..63 = 0)
    device float* d_fj_b2 [[buffer(15)]],           // [32] atomic f32
    device bfloat* d_B_nav [[buffer(16)]],          // [EW_pad, 64]
    device bfloat* d_B_tm [[buffer(17)]],           // [EK_pad, 64]
    device float* d_W_b_self [[buffer(18)]],        // [4, 32] atomic f32
    device bfloat* d_val_drift [[buffer(19)]],      // [EN_pad, 64]
    device float* d_dyn_act_w [[buffer(20)]],       // [32] atomic f32
    device float* d_sc_grads [[buffer(21)]],        // [64] atomic f32
    device float* diag_out [[buffer(22)]],          // [16] atomic f32
    constant uint* u [[buffer(23)]],                // [E, N, W_nav, K]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], W_nav = u[2], K = u[3];
  if (e >= E) return;
  const uint A_dim = W_nav + K + 4u;
  const ulong EN = ulong(E) * ulong(N);

  const float val_coef = lcfg[6], dyn_coef = lcfg[7];
  const float inv_am = lcfg[8], inv_wm = lcfg[9], inv_lm = lcfg[10], inv_en = lcfg[11];
  const float val_g = lcfg[12], val_b0 = lcfg[13], val_b1 = lcfg[14];
  const float dyn_dg = lcfg[15], dyn_ag = lcfg[16], gsc = lcfg[17], eps_u = max(lcfg[18], 1e-6f);
  const uint B_cur = uint(max(1.0f, rint(lcfg[19])));
  const float valid_env = (e < B_cur) ? 1.0f : 0.0f;
  const float v_max = 3.0f, inv_vmax = 1.0f / v_max;
  const float inv_A = 1.0f / float(A_dim);

  threadgroup float sh_B[72 * 32];
  threadgroup float sh_dB[72 * 32];
  threadgroup float sh_db2[32];
  threadgroup float sh_ddaw[32];
  threadgroup float sh_v[32], sh_vt[32], sh_ul[32], sh_ep[32];
  threadgroup float sh_dva[72];
  threadgroup float tg_red[9];

  for (uint idx = tid; idx < A_dim * 32u; idx += 128u) {
    sh_B[idx] = B_H[ulong(e) * A_dim * 32ul + idx];
    sh_dB[idx] = 0.0f;
  }
  if (tid < 32u) { sh_db2[tid] = 0.0f; sh_ddaw[tid] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float loc_l_vel = 0.0f, loc_l_val = 0.0f, loc_l_dyn = 0.0f;
  float loc_unorm = 0.0f, loc_vnorm = 0.0f;
  float g_vg = 0.0f, g_vb0 = 0.0f, g_vb1 = 0.0f, g_ddg = 0.0f, g_dag = 0.0f;

  for (uint n = 0; n < N; n++) {
    const ulong en_idx = ulong(e) * N + n;
    const device float* tf = targ_f + en_idx * 6ul;
    const float w_norm = tf[0] * valid_env;
    const float gae_ret = tf[1];
    const float obs_drho = tf[2];
    const float wm = tf[4] * valid_env;
    const float lm = (1.0f - tf[4]) * valid_env;
    const uint t0 = uint(clamp(int(rint(tf[5])), 0, int(max(1u, K)) - 1));
    const device float* ut = u_targ + en_idx * A_dim;

    // Critic & dynamics backward (strictly masked by valid_env so padded tail rows contribute EXACT 0!)
    float da_dot = (tid < 32u) ? (dyn_act_w[tid] * u_lat_targ[en_idx * 32ul + tid]) : 0.0f;
    const float da_raw = gf_tg_sum((tid < 32u) ? da_dot : 0.0f, lane, sg, 4u, tg_red);

    if (tid < 64u) {
      float dv_out = 0.0f;
      if (tid < 3u && valid_env > 0.5f) {
        const ulong vb = en_idx * 64ul;
        const float raw0 = float(val_drift[vb + 0u]);
        const float raw1 = float(val_drift[vb + 1u]);
        const float raw2 = float(val_drift[vb + 2u]);
        if (tid == 0u) {
          const float diff = (val_g * raw0 + val_b0) - gae_ret;
          const float ad = fabs(diff);
          const float h_loss = (ad <= 1.0f) ? (0.5f * diff * diff) : (ad - 0.5f);
          const float dh = (ad <= 1.0f) ? diff : ((diff > 0.0f) ? 1.0f : -1.0f);
          loc_l_val += h_loss * wm * inv_wm;
          const float d_vw = dh * (val_coef * wm * inv_wm) * gsc;
          dv_out = d_vw * val_g; g_vg += d_vw * raw0; g_vb0 += d_vw;
        } else if (tid == 1u) {
          const float diff = (val_g * raw1 + val_b1) - gae_ret;
          const float ad = fabs(diff);
          const float h_loss = (ad <= 1.0f) ? (0.5f * diff * diff) : (ad - 0.5f);
          const float dh = (ad <= 1.0f) ? diff : ((diff > 0.0f) ? 1.0f : -1.0f);
          loc_l_val += h_loss * lm * inv_lm;
          const float d_vl = dh * (val_coef * lm * inv_lm) * gsc;
          dv_out = d_vl * val_g; g_vg += d_vl * raw1; g_vb1 += d_vl;
        } else {
          const float diff = (dyn_dg * raw2 + dyn_ag * da_raw) - obs_drho;
          const float ad = fabs(diff);
          const float h_loss = (ad <= 1.0f) ? (0.5f * diff * diff) : (ad - 0.5f);
          const float dh = (ad <= 1.0f) ? diff : ((diff > 0.0f) ? 1.0f : -1.0f);
          loc_l_dyn += h_loss * inv_en;
          const float d_dp = dh * (dyn_coef * inv_en) * gsc;
          dv_out = d_dp * dyn_dg; g_ddg += d_dp * raw2; g_dag += d_dp * da_raw;
          tg_red[0] = d_dp * dyn_ag;
        }
      } else if (tid == 2u) {
        tg_red[0] = 0.0f;
      }
      d_val_drift[en_idx * 64ul + tid] = bfloat(dv_out);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
      sh_ddaw[tid] += tg_red[0] * u_lat_targ[en_idx * 32ul + tid];
      sh_ul[tid] = u_lat_targ[en_idx * 32ul + tid];
    }

    // Separate lightweight basis reconstruction term 0.20 * ||M(B_H u_lat - u_t)||^2 (trains B_H cleanly!)
    float er_sq = 0.0f;
    if (w_norm > 0.0f && tid < A_dim && tid != W_nav + t0) {
      float u_rec = 0.0f;
      for (uint d = 0; d < 32u; d++) u_rec += sh_B[tid * 32u + d] * sh_ul[d];
      const float err_r = u_rec - ut[tid];
      er_sq = err_r * err_r;
      const float dr_a = (2.0f * 0.20f * inv_A * w_norm * inv_am * gsc) * err_r;
      for (uint d = 0; d < 32u; d++) sh_dB[tid * 32u + d] += dr_a * sh_ul[d];
    }
    const float lr_n = gf_tg_sum(er_sq, lane, sg, 4u, tg_red) * inv_A;
    if (tid == 0u) loc_l_vel += 0.20f * lr_n * w_norm * inv_am;

    // 4 stratified jittered flow times (t = 0..3): consistent target v* = u_lat_targ - eps in masked action metric!
    const float w_flow = 0.25f * w_norm * inv_am;
    for (uint t = 0; t < 4u; t++) {
      const ulong row = ulong(t) * EN + en_idx;
      if (tid < 32u) {
        const float v_raw = float(vj_raw[row * 64ul + tid]) + fj_b2[tid];
        const float th = gf_safe_tanh(v_raw * inv_vmax);
        sh_v[tid]  = v_max * th;
        sh_vt[tid] = 1.0f - th * th; // exact d(v_max*tanh(x/v_max))/dx!
        sh_ep[tid] = eps_train[row * 32ul + tid];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float ev_sq = 0.0f, dv_a = 0.0f;
      if (w_flow > 0.0f && tid < A_dim && tid != W_nav + t0) {
        float err_v = 0.0f;
        for (uint d = 0; d < 32u; d++) {
          err_v += sh_B[tid * 32u + d] * (sh_v[d] - (sh_ul[d] - sh_ep[d]));
        }
        ev_sq = err_v * err_v;
        dv_a = (2.0f * inv_A * w_flow * gsc) * err_v;
        for (uint d = 0; d < 32u; d++) {
          sh_dB[tid * 32u + d] += dv_a * (sh_v[d] - (sh_ul[d] - sh_ep[d]));
        }
      }
      const float lv_t = gf_tg_sum(ev_sq, lane, sg, 4u, tg_red) * inv_A;
      if (tid == 0u) loc_l_vel += lv_t * w_flow;
      if (tid < A_dim) sh_dva[tid] = dv_a;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid < 32u) {
        float gv = 0.0f;
        for (uint a = 0; a < A_dim; a++) gv += sh_B[a * 32u + tid] * sh_dva[a];
        const float gv_raw = gv * sh_vt[tid];
        d_vj_raw[row * 64ul + tid] = bfloat(gv_raw);
        d_vj_raw[row * 64ul + 32u + tid] = bfloat(0.0f);
        sh_db2[tid] += gv_raw;
        if (t == 0u && valid_env > 0.5f) loc_vnorm += sh_v[tid] * sh_v[tid];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float un_a = (valid_env > 0.5f && tid < A_dim && tid != W_nav + t0) ? (ut[tid] * ut[tid]) : 0.0f;
    const float un_s = gf_tg_sum(un_a, lane, sg, 4u, tg_red) * inv_A;
    if (tid == 0u && valid_env > 0.5f) loc_unorm += sqrt(un_s) * inv_en;
  }

  // Backprop sh_dB [A_dim, 32] through unit_vec(B_raw) -> d_B_nav [EW_pad, 64], d_B_tm [EK_pad, 64], atomic d_W_b_self!
  for (uint a = sg; a < A_dim; a += 4u) {
    float raw = 0.0f;
    if (a < W_nav) raw = float(B_nav_raw[(ulong(e) * W_nav + a) * 64ul + lane]);
    else if (a < W_nav + K) raw = float(B_tm_raw[(ulong(e) * K + (a - W_nav)) * 64ul + lane]);
    else raw = W_b_self[(a - W_nav - K) * 32u + lane];
    const float inv = rsqrt(simd_sum(raw * raw) + eps_u);
    const float bn = raw * inv;
    const float g = sh_dB[a * 32u + lane];
    const float dot = simd_sum(g * bn);
    const float draw = (g - bn * dot) * inv;
    if (a < W_nav) {
      d_B_nav[(ulong(e) * W_nav + a) * 64ul + lane] = bfloat(draw);
      d_B_nav[(ulong(e) * W_nav + a) * 64ul + 32u + lane] = bfloat(0.0f);
    } else if (a < W_nav + K) {
      d_B_tm[(ulong(e) * K + (a - W_nav)) * 64ul + lane] = bfloat(draw);
      d_B_tm[(ulong(e) * K + (a - W_nav)) * 64ul + 32u + lane] = bfloat(0.0f);
    } else {
      gf_atomic_add_f32(&d_W_b_self[(a - W_nav - K) * 32u + lane], draw);
    }
  }

  if (tid < 32u) {
    gf_atomic_add_f32(&d_fj_b2[tid], sh_db2[tid]);
    gf_atomic_add_f32(&d_dyn_act_w[tid], sh_ddaw[tid]);
  }
  const float vn_tot = gf_tg_sum((tid < 32u) ? loc_vnorm : 0.0f, lane, sg, 4u, tg_red);
  if (tid == 0u || tid == 1u) {
    gf_atomic_add_f32(&d_sc_grads[0], g_vg);
    gf_atomic_add_f32(&diag_out[1], loc_l_val);
  }
  if (tid == 0u) {
    gf_atomic_add_f32(&d_sc_grads[1], g_vb0);
    gf_atomic_add_f32(&diag_out[0], loc_l_vel);
    gf_atomic_add_f32(&diag_out[3], loc_unorm);
    gf_atomic_add_f32(&diag_out[4], sqrt(vn_tot / max(1.0f, float(N) * 32.0f)) / float(B_cur));
  } else if (tid == 1u) {
    gf_atomic_add_f32(&d_sc_grads[2], g_vb1);
  } else if (tid == 2u) {
    gf_atomic_add_f32(&d_sc_grads[3], g_ddg);
    gf_atomic_add_f32(&d_sc_grads[4], g_dag);
    gf_atomic_add_f32(&diag_out[2], loc_l_dyn);
  }
}

// ============================================================================
// 10. MG_OP_FLOW_COND_BWD (1 TG per env e, 128 threads)
// Reduces d_inp_fj[0..3, e, n, 40:104] -> mp_norm_bwd(c_bot + c_team) -> d_c_bot, d_c_team
// ============================================================================
[[kernel]] void mg_flow_cond_bwd(
    const device bfloat* c_bot [[buffer(0)]],      // [EN_pad, 64]
    const device bfloat* c_team [[buffer(1)]],     // [EK_pad, 64]
    const device bfloat* d_inp_fj [[buffer(2)]],   // [M_fj_pad, 128]
    const device int* ent_meta [[buffer(3)]],      // [E*(M_ent + K)]
    device bfloat* d_c_bot [[buffer(4)]],          // [EN_pad, 64]
    device bfloat* d_c_team [[buffer(5)]],         // [EK_pad, 64]
    constant uint* u [[buffer(6)]],                // [E, N, M_ent, K]
    constant float* f [[buffer(7)]],               // [eps_n]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], M_ent = u[2], K = u[3];
  if (e >= E) return;
  const float eps_n = max(f[0], 1e-4f);
  const ulong EN = ulong(E) * ulong(N);

  threadgroup float sh_dct[16 * 64];
  threadgroup float tg_red[9];
  for (uint idx = tid; idx < K * 64u; idx += 128u) sh_dct[idx] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint n = 0; n < N; n++) {
    const ulong en_idx = ulong(e) * N + n;
    const uint t0 = uint(clamp(ent_meta[ulong(e) * M_ent + n] - 1, 0, int(max(1u, K)) - 1));
    float s = 0.0f, g = 0.0f;
    if (tid < 64u) {
      s = float(c_bot[en_idx * 64ul + tid]) + float(c_team[(ulong(e) * K + t0) * 64ul + tid]);
      for (uint t = 0; t < 4u; t++) g += float(d_inp_fj[(ulong(t) * EN + en_idx) * 128ul + 40u + tid]);
    }
    const float inv = rsqrt(gf_tg_sum((tid < 64u) ? (s * s) : 0.0f, lane, sg, 4u, tg_red) / 64.0f + eps_n);
    const float dot = gf_tg_sum((tid < 64u) ? (g * (s * inv)) : 0.0f, lane, sg, 4u, tg_red) / 64.0f;
    if (tid < 64u) {
      const float ds = (g - (s * inv) * dot) * inv;
      d_c_bot[en_idx * 64ul + tid] = bfloat(ds);
      sh_dct[t0 * 64u + tid] += ds;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint idx = tid; idx < K * 64u; idx += 128u) {
    const uint k = idx / 64u, c = idx % 64u;
    d_c_team[(ulong(e) * K + k) * 64ul + c] = bfloat(sh_dct[idx]);
  }
}
)MSL");
}

} // namespace mesh_simdt
#endif
