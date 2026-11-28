#ifndef MESH_SIMDT_H
#define MESH_SIMDT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace mesh_simdt {

inline std::string build_simdt_msl() {
  return std::string(R"MSL(
#include <metal_stdlib>
using namespace metal;

#define UNROLL_ALL _Pragma("clang loop unroll(full)")

inline float safe_tanh(float x) {
  return tanh(clamp(x, -20.0f, 20.0f));
}

inline float tg_sum_f32(float v, uint lane, uint sg, uint n_sg, threadgroup float* scratch) {
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

inline float simdt_philox_u01(ulong seed, ulong counter, ulong idx, ulong salt) {
  ulong x = (seed * 0x9E3779B1ul + counter * 0x85EBCA77ul + idx * 0xC2B2AE3Dul + salt) & 0x7FFFFFFFul;
  x = ((x ^ (x >> 15)) * 0x2C1B3C6Dul) & 0x7FFFFFFFul;
  x = ((x ^ (x >> 12)) * 0x297A2D39ul) & 0x7FFFFFFFul;
  x = (x ^ (x >> 15)) & 0x00FFFFFFul;
  return float(x) * (1.0f / 16777216.0f);
}

// ---- 1. MG_OP_PLAN_PRENORM ----
// 1 threadgroup per env e, 128 threads (4 simdgroups).
// Computes cgt_tensor_features_batched + prenormalize_{obs,carts,teams} + asinh + mp_norm + NAX zero-pad
// and extracts ent_meta = [entity_teams(E, M_ent), team_ids(E, K)].
[[kernel]] void mg_plan_prenorm(
    const device bfloat* obs_in [[buffer(0)]],
    const device bfloat* carts_in [[buffer(1)]],
    const device bfloat* teams_in [[buffer(2)]],
    const device float* geo [[buffer(3)]],      // [cx, cy, cz, radius]
    const device float* ncfg [[buffer(4)]],     // 12 norm constants
    device int* ent_meta [[buffer(5)]],         // [0..E*M_ent-1]: entity_teams, [E*M_ent..]: team_ids
    device bfloat* obs_out [[buffer(6)]],       // [E*N, 128]
    device bfloat* cart_out [[buffer(7)]],      // [E*J, 64]
    device bfloat* team_out [[buffer(8)]],      // [E*K, 64]
    constant uint* u [[buffer(9)]],             // [E, N, J, K, obs_pad_dim(128), cart_pad_dim(64), team_pad_dim(64), mask_bot_cfg]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3];
  const uint O_PAD = u[4], C_PAD = u[5], T_PAD = u[6];
  if (e >= E) return;
  const uint M_ent = N + J + K;

  const float mp_eps = max(ncfg[0], 1e-4f);
  const float rad_min = ncfg[1];
  const float pos_fb = ncfg[2];
  const float vel_sc = ncfg[3];
  const float hp_off = ncfg[4], hp_sc = ncfg[5];
  const float ar_off = ncfg[6], ar_sc = ncfg[7];
  const float ammo_sc = ncfg[8];
  const float str_max = ncfg[9];
  const float clen_sc = ncfg[10];
  const float cspd_sc = ncfg[11];
  const bool has_geo = (geo[3] > 0.0f);
  const float cx = geo[0], cy = geo[1], cz = geo[2];
  const float rad = max(geo[3], rad_min);

  threadgroup float t_rho[32];
  threadgroup float t_best_other[32];
  threadgroup float t_held[32];
  threadgroup float t_rate[32];
  threadgroup float t_rem[32];
  threadgroup float t_id[32];
  threadgroup float t_extra[32 * 2];
  threadgroup int c_nim[64];
  threadgroup int c_rem_heap[64];
  threadgroup float tg_red[9];
  threadgroup float max_rho_sh, max_rate_sh, min_rem_sh;
  threadgroup int nim_tot_sh;

  // Step 1: Team CGT features
  if (tid < K) {
    const ulong tb = (ulong(e) * K + tid) * 7ul;
    const float id_v = float(teams_in[tb + 0]);
    const float score = float(teams_in[tb + 1]);
    const float held = float(teams_in[tb + 2]);
    const float rate = float(teams_in[tb + 3]);
    const float lim = max(float(teams_in[tb + 4]), 1e-6f);
    const float rem = max(lim - score, 1e-6f);
    const float rho = (held > 0.0f) ? (rate * held / rem) : 0.0f;
    t_id[tid] = id_v;
    t_held[tid] = held;
    t_rate[tid] = rate;
    t_rem[tid] = rem;
    t_rho[tid] = rho;
    ent_meta[ulong(E) * M_ent + ulong(e) * K + tid] = int(rint(id_v));
    ent_meta[ulong(e) * M_ent + N + J + tid] = int(rint(id_v));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < K) {
    float bo = 0.0f;
    if (K > 1u) {
      float m = -1.0f;
      for (uint q = 0; q < K; q++) {
        if (q != tid) m = max(m, t_rho[q]);
      }
      bo = max(m, 0.0f);
    }
    t_best_other[tid] = bo;
    const float r = t_rho[tid];
    t_extra[tid * 2u + 0u] = float(bfloat(safe_tanh(r)));
    t_extra[tid * 2u + 1u] = float(bfloat((r - bo) / (r + bo + 1e-6f)));
  }
  if (tid == 0u) {
    float mr = 1e-6f, mrt = 0.0f, mrm = 1e30f;
    for (uint k = 0; k < K; k++) {
      mr = max(mr, t_rho[k]);
      mrt = max(mrt, t_rate[k]);
      mrm = min(mrm, t_rem[k]);
    }
    max_rho_sh = mr;
    max_rate_sh = mrt;
    min_rem_sh = max(mrm, 1e-6f);
  }
  if (tid < J) {
    const ulong cb = (ulong(e) * J + tid) * 18ul;
    const float cp_cnt = max(float(carts_in[cb + 17]), 1.0f);
    const float cp_dep = max(float(carts_in[cb + 16]), 0.0f);
    const int rh = max(0, int(rint(cp_cnt - cp_dep)));
    c_rem_heap[tid] = rh;
    c_nim[tid] = rh % 3;
    ent_meta[ulong(e) * M_ent + N + tid] = int(rint(float(carts_in[cb + 3])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    int nt = 0;
    for (uint j = 0; j < J; j++) nt ^= c_nim[j];
    nim_tot_sh = nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 2: prenormalize_teams -> team_out [E*K, T_PAD]
  for (uint k = 0; k < K; k++) {
    const ulong tb = (ulong(e) * K + k) * 7ul;
    float v = 0.0f;
    if (tid < 9u) {
      if (tid < 7u) {
        v = float(teams_in[tb + tid]);
        if (tid == 1u) v = v / max(float(teams_in[tb + 4]), 1.0f);
        else if (tid == 4u) v = 1.0f;
        else if (tid == 5u) v = 0.0f;
      } else {
        v = t_extra[k * 2u + (tid - 7u)];
      }
      v = asinh(v);
    }
    const float ss = tg_sum_f32((tid < 9u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 9.0f + mp_eps);
    if (tid < T_PAD) {
      team_out[(ulong(e) * K + k) * T_PAD + tid] = (tid < 9u) ? bfloat(v * inv) : bfloat(0.0f);
    }
  }

  // Step 3: prenormalize_carts -> cart_out [E*J, C_PAD]
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
        const float plen = max(float(carts_in[cb + 2]), 1e-6f);
        const float pfrac = clamp(float(carts_in[cb + 1]) / plen, 0.0f, 1.0f);
        const float cp_cnt = max(float(carts_in[cb + 17]), 1.0f);
        const float cp_dep = max(float(carts_in[cb + 16]), 0.0f);
        const float seg = pfrac * cp_cnt;
        if (tid == 18u) {
          const float dist_deny = max(seg - floor(seg - 1e-6f), 0.05f);
          const float oid = float(carts_in[cb + 3]);
          int oidx = -1;
          for (uint k = 0; k < K; k++) {
            if (oid == t_id[k]) { oidx = int(k); break; }
          }
          float drd = 0.0f;
          float denom = 1e-6f;
          if (oidx >= 0) {
            const float oh = t_held[oidx];
            const float orho = t_rho[oidx];
            denom = orho + t_best_other[oidx] + 1e-6f;
            if (cp_dep > 0.0f && oh > 0.0f) drd = orho / max(oh, 1.0f);
          }
          v = float(bfloat(safe_tanh(drd / denom / dist_deny)));
        } else if (tid == 19u) {
          const float dist_push = max(ceil(seg + 1e-6f) - seg, 0.05f);
          const float drp = (cp_dep < cp_cnt) ? (max_rate_sh / min_rem_sh) : 0.0f;
          v = float(bfloat(safe_tanh((drp / (max_rho_sh + drp + 1e-6f)) / dist_push)));
        } else if (tid == 20u) {
          const int nb = c_nim[j];
          const int nt = nim_tot_sh;
          const int tn = nb ^ nt;
          v = (nt != 0 && tn < nb && (nb - tn) <= c_rem_heap[j]) ? 1.0f : 0.0f;
        } else {
          v = float(bfloat(float(c_nim[j]) / 3.0f));
        }
      }
      v = asinh(v);
    }
    const float ss = tg_sum_f32((tid < 22u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 22.0f + mp_eps);
    if (tid < C_PAD) {
      cart_out[(ulong(e) * J + j) * C_PAD + tid] = (tid < 22u) ? bfloat(v * inv) : bfloat(0.0f);
    }
  }

  // Step 4: prenormalize_obs -> obs_out [E*N, O_PAD]
  for (uint n = 0; n < N; n++) {
    const ulong ob = (ulong(e) * N + n) * 84ul;
    if (tid == 0u) {
      ent_meta[ulong(e) * M_ent + n] = int(rint(float(obs_in[ob + 1])));
    }
    float v = 0.0f;
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
      else if (tid == 20u || tid == 21u || tid == 22u || tid == 23u ||
               tid == 39u || tid == 40u || tid == 41u ||
               tid == 54u || tid == 62u || tid == 79u || tid == 80u) {
        v = 0.0f;
      }
      v = asinh(v);
    }
    const float ss = tg_sum_f32((tid < 84u) ? (v * v) : 0.0f, lane, sg, 4u, tg_red);
    const float inv = rsqrt(ss / 84.0f + mp_eps);
    if (tid < O_PAD) {
      obs_out[(ulong(e) * N + n) * O_PAD + tid] = (tid < 84u) ? bfloat(v * inv) : bfloat(0.0f);
    }
  }
}

// ---- 2. MG_OP_PLAN_EMBED_CAT (1 simdgroup = 32 threads per row, 4 rows per TG, ZERO TG barriers!) ----
[[kernel]] void mg_plan_embed_cat(
    const device bfloat* obs_enc [[buffer(0)]],
    const device bfloat* cart_enc [[buffer(1)]],
    const device bfloat* team_enc [[buffer(2)]],
    const device bfloat* type_emb [[buffer(3)]], // [3, D] pre-normalized
    const device float* scalars [[buffer(4)]],
    device bfloat* x_out [[buffer(5)]],          // [E * M_ent, D]
    constant uint* u [[buffer(6)]],              // [E, N, J, K, D, scalar_idx]
    constant float* f [[buffer(7)]],             // [mp_norm_eps, mp_lerp_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], D = u[4], sc_idx = u[5];
  const uint M_ent = N + J + K;
  const uint row = tgp * 4u + sg;
  const uint e = row / M_ent;
  const uint m = row % M_ent;
  if (e >= E) return;

  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float tb = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float lerp_inv = rsqrt((1.0f - tb) * (1.0f - tb) + tb * tb + eps_l);

  const device bfloat* src = nullptr;
  uint type_id = 0u;
  if (m < N) {
    src = obs_enc + (ulong(e) * N + m) * D;
    type_id = 0u;
  } else if (m < N + J) {
    src = cart_enc + (ulong(e) * J + (m - N)) * D;
    type_id = 1u;
  } else {
    src = team_enc + (ulong(e) * K + (m - N - J)) * D;
    type_id = 2u;
  }
  const device bfloat* te = type_emb + ulong(type_id) * D;
  device bfloat* dst = x_out + ulong(row) * D;

  float reg[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss1 = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    float v = (c < D) ? float(src[c]) : 0.0f;
    reg[i] = v;
    ss1 += v * v;
  }
  const float inv1 = rsqrt(simd_sum(ss1) / float(D) + eps_n);

  float ss2 = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) {
      float a = float(bfloat(reg[i] * inv1));
      float b = float(te[c]);
      float h = float(bfloat(((1.0f - tb) * a + tb * b) * lerp_inv));
      reg[i] = h;
      ss2 += h * h;
    }
  }
  const float inv2 = rsqrt(simd_sum(ss2) / float(D) + eps_n);
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) dst[c] = bfloat(reg[i] * inv2);
  }
}

// ---- 3. MG_OP_MP_NORM (1 simdgroup = 32 threads per row, 4 rows per TG, ZERO TG barriers!) ----
inline float simdt_rad_sign64(uint idx) {
  constexpr ulong MASK = 0x965a3c69a5c3695aul;
  return ((MASK >> (idx & 63u)) & 1ul) ? 1.0f : -1.0f;
}

inline void simdt_fwht64_pair(thread float& a, thread float& b, uint lane) {
  float s = a + b, d = a - b;
  a = s; b = d;
  #pragma unroll
  for (uint mask = 1u; mask <= 16u; mask <<= 1u) {
    float oa = simd_shuffle_xor(a, ushort(mask));
    float ob = simd_shuffle_xor(b, ushort(mask));
    bool hi = (lane & mask) != 0u;
    a = hi ? (oa - a) : (a + oa);
    b = hi ? (ob - b) : (b + ob);
  }
  a *= 0.125f;
  b *= 0.125f;
}

[[kernel]] void mg_mp_norm(
    const device bfloat* x [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint* u [[buffer(2)]],  // [rows, D]
    constant float* f [[buffer(3)]], // [eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint rows = u[0], D = u[1];
  const uint row = tgp * 4u + sg;
  if (row >= rows) return;
  const float eps = max(f[0], 1e-4f);
  const device bfloat* xr = x + ulong(row) * D;
  device bfloat* yr = out + ulong(row) * D;
  float reg[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    float v = (c < D) ? float(xr[c]) : 0.0f;
    reg[i] = v;
    ss += v * v;
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) yr[c] = bfloat(reg[i] * inv);
  }
}

// Fused mp_norm + Rademacher-FWHT-64 + per-row dynamic int8_t pack (D <= 512, D%64==0, ZERO extra DRAM reads!)
[[kernel]] void mg_mp_norm_i8(
    const device bfloat* x [[buffer(0)]],
    device bfloat* out_bf [[buffer(1)]],
    device int8_t* out_q [[buffer(2)]],
    device float* out_s [[buffer(3)]],
    constant uint* u [[buffer(4)]],  // [rows, D, rows_pad]
    constant float* f [[buffer(5)]], // [eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint rows = u[0], D = u[1], rows_pad = max(u[0], u[2]);
  const uint row = tgp * 4u + sg;
  if (row >= rows_pad) return;
  const uint n_blk = D >> 6u;
  device char2* q2 = reinterpret_cast<device char2*>(out_q + ulong(row) * ulong(D));
  if (row >= rows) {
    if (lane == 0u) out_s[row] = 0.0f;
    for (uint blk = 0u; blk < n_blk; blk++) q2[blk * 32u + lane] = char2(0, 0);
    return;
  }
  const float eps = max(f[0], 1e-4f);
  const device bfloat* xr = x + ulong(row) * D;
  device bfloat* yr = out_bf + ulong(row) * D;
  float ra[8], rb[8];
  float ss = 0.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    const uint c0 = blk * 64u + lane * 2u;
    float a = float(xr[c0]);
    float b = float(xr[c0 + 1u]);
    ra[blk] = a; rb[blk] = b;
    ss += a * a + b * b;
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  float lmax = 0.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    const uint c0 = blk * 64u + lane * 2u;
    float a = float(bfloat(ra[blk] * inv));
    float b = float(bfloat(rb[blk] * inv));
    yr[c0] = bfloat(a);
    yr[c0 + 1u] = bfloat(b);
    a *= simdt_rad_sign64(lane * 2u);
    b *= simdt_rad_sign64(lane * 2u + 1u);
    simdt_fwht64_pair(a, b, lane);
    ra[blk] = a; rb[blk] = b;
    lmax = max(lmax, max(fabs(a), fabs(b)));
  }
  const float amax = max(simd_max(lmax), 1e-8f);
  const float inv_s = 127.0f / amax;
  if (lane == 0u) out_s[row] = amax / 127.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    q2[blk * 32u + lane] = char2(
        char(clamp(int(rint(ra[blk] * inv_s)), -127, 127)),
        char(clamp(int(rint(rb[blk] * inv_s)), -127, 127)));
  }
}

// ---- 4. MG_OP_MP_LERP (1 simdgroup = 32 threads per row, 4 rows per TG, ZERO TG barriers!) ----
[[kernel]] void mg_mp_lerp(
    const device bfloat* a_in [[buffer(0)]],
    const device bfloat* b_in [[buffer(1)]],
    const device float* scalars [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint* u [[buffer(4)]],  // [rows, D, sc_idx, norm_b]
    constant float* f [[buffer(5)]], // [mp_norm_eps, mp_lerp_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint rows = u[0], D = u[1], sc_idx = u[2], norm_b = u[3];
  const uint row = tgp * 4u + sg;
  if (row >= rows) return;
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float t = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float lerp_inv = rsqrt((1.0f - t) * (1.0f - t) + t * t + eps_l);

  const device bfloat* ar = a_in + ulong(row) * D;
  const device bfloat* br = b_in + ulong(row) * D;
  device bfloat* yr = out + ulong(row) * D;

  float rb[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ssb = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    float v = (c < D) ? float(br[c]) : 0.0f;
    rb[i] = v;
    ssb += v * v;
  }
  const float inv_b = norm_b ? rsqrt(simd_sum(ssb) / float(D) + eps_n) : 1.0f;

  float sso = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) {
      float av = float(ar[c]);
      float bv = norm_b ? float(bfloat(rb[i] * inv_b)) : rb[i];
      float hv = float(bfloat(((1.0f - t) * av + t * bv) * lerp_inv));
      rb[i] = hv;
      sso += hv * hv;
    }
  }
  const float inv_o = rsqrt(simd_sum(sso) / float(D) + eps_n);
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) yr[c] = bfloat(rb[i] * inv_o);
  }
}

// Fused mp_lerp + Rademacher-FWHT-64 + per-row dynamic int8_t pack (D <= 512, D%64==0, ZERO extra DRAM reads!)
[[kernel]] void mg_mp_lerp_i8(
    const device bfloat* a_in [[buffer(0)]],
    const device bfloat* b_in [[buffer(1)]],
    const device float* scalars [[buffer(2)]],
    device bfloat* out_bf [[buffer(3)]],
    device int8_t* out_q [[buffer(4)]],
    device float* out_s [[buffer(5)]],
    constant uint* u [[buffer(6)]],  // [rows, D, sc_idx, norm_b, rows_pad]
    constant float* f [[buffer(7)]], // [mp_norm_eps, mp_lerp_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint rows = u[0], D = u[1], sc_idx = u[2], norm_b = u[3], rows_pad = max(u[0], u[4]);
  const uint row = tgp * 4u + sg;
  if (row >= rows_pad) return;
  const uint n_blk = D >> 6u;
  device char2* q2 = reinterpret_cast<device char2*>(out_q + ulong(row) * ulong(D));
  if (row >= rows) {
    if (lane == 0u) out_s[row] = 0.0f;
    for (uint blk = 0u; blk < n_blk; blk++) q2[blk * 32u + lane] = char2(0, 0);
    return;
  }
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float t = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float lerp_inv = rsqrt((1.0f - t) * (1.0f - t) + t * t + eps_l);

  const device bfloat* ar = a_in + ulong(row) * D;
  const device bfloat* br = b_in + ulong(row) * D;
  device bfloat* yr = out_bf + ulong(row) * D;

  float rba[8], rbb[8];
  float ssb = 0.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    const uint c0 = blk * 64u + lane * 2u;
    float va = float(br[c0]), vb = float(br[c0 + 1u]);
    rba[blk] = va; rbb[blk] = vb;
    ssb += va * va + vb * vb;
  }
  const float inv_b = norm_b ? rsqrt(simd_sum(ssb) / float(D) + eps_n) : 1.0f;

  float sso = 0.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    const uint c0 = blk * 64u + lane * 2u;
    float ba = norm_b ? float(bfloat(rba[blk] * inv_b)) : rba[blk];
    float bb = norm_b ? float(bfloat(rbb[blk] * inv_b)) : rbb[blk];
    float ha = float(bfloat(((1.0f - t) * float(ar[c0]) + t * ba) * lerp_inv));
    float hb = float(bfloat(((1.0f - t) * float(ar[c0 + 1u]) + t * bb) * lerp_inv));
    rba[blk] = ha; rbb[blk] = hb;
    sso += ha * ha + hb * hb;
  }
  const float inv_o = rsqrt(simd_sum(sso) / float(D) + eps_n);

  float lmax = 0.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    const uint c0 = blk * 64u + lane * 2u;
    float ya = float(bfloat(rba[blk] * inv_o));
    float yb = float(bfloat(rbb[blk] * inv_o));
    yr[c0] = bfloat(ya);
    yr[c0 + 1u] = bfloat(yb);
    ya *= simdt_rad_sign64(lane * 2u);
    yb *= simdt_rad_sign64(lane * 2u + 1u);
    simdt_fwht64_pair(ya, yb, lane);
    rba[blk] = ya; rbb[blk] = yb;
    lmax = max(lmax, max(fabs(ya), fabs(yb)));
  }
  const float amax = max(simd_max(lmax), 1e-8f);
  const float inv_s = 127.0f / amax;
  if (lane == 0u) out_s[row] = amax / 127.0f;
  for (uint blk = 0u; blk < n_blk; blk++) {
    q2[blk * 32u + lane] = char2(
        char(clamp(int(rint(rba[blk] * inv_s)), -127, 127)),
        char(clamp(int(rint(rbb[blk] * inv_s)), -127, 127)));
  }
}

// ---- 5. MG_OP_MP_SWIGLU ----
// Input: gate_up [rows, 2*H] BF16 (packed [gate(H), up(H)] per row); Output: hid [rows, H] BF16
[[kernel]] void mg_mp_swiglu(
    const device bfloat* gu [[buffer(0)]],
    device bfloat* hid [[buffer(1)]],
    constant uint* u [[buffer(2)]],  // [rows, H]
    constant float* f [[buffer(3)]], // [inv_silu_scale]
    uint2 gid [[thread_position_in_grid]]) {
  const uint rows = u[0], H = u[1];
  const uint col = gid.x, row = gid.y;
  if (row >= rows || col >= H) return;
  const ulong base = ulong(row) * (2ul * H);
  const float g = float(gu[base + col]);
  const float up = float(gu[base + H + col]);
  const float sg = g / (1.0f + exp(-g));
  const float ms = float(bfloat(sg * f[0]));
  hid[ulong(row) * H + col] = bfloat(ms * up);
}

// Fused mp_swiglu + Rademacher-FWHT-64 + per-row dynamic int8_t pack (1 TG = 128 th per row, H <= 1024, ZERO extra DRAM reads!)
[[kernel]] void mg_mp_swiglu_i8(
    const device bfloat* gu [[buffer(0)]],
    device bfloat* hid_bf [[buffer(1)]],
    device int8_t* hid_q [[buffer(2)]],
    device float* hid_s [[buffer(3)]],
    constant uint* u [[buffer(4)]],  // [rows, H, rows_pad, write_bf]
    constant float* f [[buffer(5)]], // [inv_silu_scale]
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint rows = u[0], H = u[1], rows_pad = max(u[0], u[2]), write_bf = u[3];
  if (row >= rows_pad) return;
  const uint n_blk = H >> 6u;
  device char2* q2 = reinterpret_cast<device char2*>(hid_q + ulong(row) * ulong(H));
  if (row >= rows) {
    if (tid == 0u) hid_s[row] = 0.0f;
    for (uint blk = sg; blk < n_blk; blk += 4u) q2[blk * 32u + lane] = char2(0, 0);
    return;
  }
  const float inv_sc = f[0];
  const ulong gbase = ulong(row) * (2ul * H);
  device bfloat* hrow = hid_bf + ulong(row) * ulong(H);
  threadgroup float sg_max[4];
  float ra[4], rb[4];
  float lmax = 0.0f;
  uint ri = 0u;
  for (uint blk = sg; blk < n_blk; blk += 4u, ri++) {
    const uint c0 = blk * 64u + lane * 2u;
    const float g0 = float(gu[gbase + c0]), up0 = float(gu[gbase + H + c0]);
    const float g1 = float(gu[gbase + c0 + 1u]), up1 = float(gu[gbase + H + c0 + 1u]);
    float h0 = float(bfloat(float(bfloat((g0 / (1.0f + exp(-g0))) * inv_sc)) * up0));
    float h1 = float(bfloat(float(bfloat((g1 / (1.0f + exp(-g1))) * inv_sc)) * up1));
    if (write_bf) {
      hrow[c0] = bfloat(h0);
      hrow[c0 + 1u] = bfloat(h1);
    }
    h0 *= simdt_rad_sign64(lane * 2u);
    h1 *= simdt_rad_sign64(lane * 2u + 1u);
    simdt_fwht64_pair(h0, h1, lane);
    ra[ri] = h0; rb[ri] = h1;
    lmax = max(lmax, max(fabs(h0), fabs(h1)));
  }
  lmax = simd_max(lmax);
  if (lane == 0u) sg_max[sg] = lmax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float amax = max(max(max(sg_max[0], sg_max[1]), max(sg_max[2], sg_max[3])), 1e-8f);
  const float inv_s = 127.0f / amax;
  if (tid == 0u) hid_s[row] = amax / 127.0f;
  ri = 0u;
  for (uint blk = sg; blk < n_blk; blk += 4u, ri++) {
    q2[blk * 32u + lane] = char2(
        char(clamp(int(rint(ra[ri] * inv_s)), -127, 127)),
        char(clamp(int(rint(rb[ri] * inv_s)), -127, 127)));
  }
}

// ---- 6. MG_OP_CGT_GRAM_STAGE1 ----
// Sub-kernel 6a-fast (M_ent <= 48): Associative G = (r @ r^T) * inv_m -> global_raw = G @ v_norm!
// 1 TG per env e, 256 threads (8 simdgroups), 9x fewer FLOPs, 1 barrier, 1 single L1 pass over v_norm!
[[kernel]] void mg_cgt_global_gram_small(
    const device bfloat* r_norm [[buffer(0)]],   // [E, M_ent, R]
    const device bfloat* v_norm [[buffer(1)]],   // [E, M_ent, D]
    device bfloat* global_raw [[buffer(2)]],     // [E, M_ent, D]
    constant uint* u [[buffer(3)]],              // [E, M_ent, D, R]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], D = u[2], R = u[3];
  if (e >= E) return;
  const float inv_m = rsqrt(float(max(1u, M_ent)));
  const device bfloat* re = r_norm + ulong(e) * M_ent * R;
  const device bfloat* ve = v_norm + ulong(e) * M_ent * D;
  device bfloat* ge = global_raw + ulong(e) * M_ent * D;

  threadgroup float G[48 * 48]; // 9 KB SRAM
  // Compute symmetric G[m1, m2] = dot(r[m1], r[m2]) * inv_m across 8 simdgroups (zero barriers!)
  uint pair_idx = 0u;
  for (uint m1 = 0; m1 < M_ent; m1++) {
    for (uint m2 = m1; m2 < M_ent; m2++) {
      if ((pair_idx & 7u) == sg) {
        const device bfloat* r1 = re + ulong(m1) * R;
        const device bfloat* r2 = re + ulong(m2) * R;
        float dp = 0.0f;
        for (uint r = lane; r < R; r += 32u) {
          dp += float(r1[r]) * float(r2[r]);
        }
        const float val = simd_sum(dp) * inv_m;
        if (lane == 0u) {
          G[m1 * M_ent + m2] = val;
          G[m2 * M_ent + m1] = val;
        }
      }
      pair_idx++;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each thread c loads v_norm[0..M_ent-1, c] ONCE into registers and computes G @ v_norm!
  for (uint c = tid; c < D; c += 256u) {
    float v_reg[48];
    for (uint m = 0; m < M_ent; m++) {
      v_reg[m] = float(ve[ulong(m) * D + c]);
    }
    for (uint m1 = 0; m1 < M_ent; m1++) {
      float acc = 0.0f;
      const uint g_off = m1 * M_ent;
      for (uint m2 = 0; m2 < M_ent; m2++) {
        acc += G[g_off + m2] * v_reg[m2];
      }
      ge[ulong(m1) * D + c] = bfloat(acc);
    }
  }
}

// Sub-kernel 6a-general (M_ent > 48): Tiled kv[32, 32] in 4 KB SRAM!
[[kernel]] void mg_cgt_global_gram(
    const device bfloat* r_norm [[buffer(0)]],   // [E, M_ent, R]
    const device bfloat* v_norm [[buffer(1)]],   // [E, M_ent, D]
    device bfloat* global_raw [[buffer(2)]],     // [E, M_ent, D]
    constant uint* u [[buffer(3)]],              // [E, M_ent, D, R]
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], D = u[2], R = u[3];
  const uint e = tgp.y;
  const uint d0 = tgp.x * 32u;
  if (e >= E || d0 >= D) return;

  const float inv_m = rsqrt(float(max(1u, M_ent)));
  const uint c_local = tid & 31u;
  const uint r_lane = tid >> 5u; // 0..3
  const uint c_glob = d0 + c_local;

  threadgroup float kv_sh[32 * 32]; // 4 KB SRAM for high occupancy
  for (uint r0 = 0; r0 < R; r0 += 32u) {
    const uint r_lim = min(32u, R - r0);
    for (uint rr = r_lane; rr < r_lim; rr += 4u) {
      float acc = 0.0f;
      if (c_glob < D) {
        for (uint m = 0; m < M_ent; m++) {
          const float rv = float(r_norm[(ulong(e) * M_ent + m) * R + (r0 + rr)]);
          const float vv = float(v_norm[(ulong(e) * M_ent + m) * D + c_glob]);
          acc += rv * vv;
        }
      }
      kv_sh[rr * 32u + c_local] = acc * inv_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint m = r_lane; m < M_ent; m += 4u) {
      if (c_glob < D) {
        float acc = (r0 == 0u) ? 0.0f : float(global_raw[(ulong(e) * M_ent + m) * D + c_glob]);
        const device bfloat* rm = r_norm + (ulong(e) * M_ent + m) * R + r0;
        for (uint rr = 0; rr < r_lim; rr++) {
          acc += float(rm[rr]) * kv_sh[rr * 32u + c_local];
        }
        global_raw[(ulong(e) * M_ent + m) * D + c_glob] = bfloat(acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Sub-kernel 6b: Pool score + team pool + blend_pool -> t_in [E, K, D]
// 1 TG per env e, 256 threads.
[[kernel]] void mg_cgt_team_pool(
    const device bfloat* xn [[buffer(0)]],        // [E, M_ent, D]
    const device bfloat* v_norm [[buffer(1)]],    // [E, M_ent, D]
    const device bfloat* w_pool [[buffer(2)]],    // [D] unit-normalized
    const device int* ent_meta [[buffer(3)]],     // entity_teams[E*M_ent], team_ids[E*K]
    const device float* scalars [[buffer(4)]],
    device bfloat* t_in [[buffer(5)]],            // [E, K, D]
    constant uint* u [[buffer(6)]],               // [E, N, J, K, D, sc_idx_blend_pool]
    constant float* f [[buffer(7)]],              // [mp_norm_eps, mp_lerp_eps]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], D = u[4], sc_idx = u[5];
  if (e >= E) return;
  const uint M_ent = N + J + K;
  const float eps_n = max(f[0], 1e-4f);
  const float eps_l = max(f[1], 1e-4f);
  const float bp = clamp(scalars[sc_idx], 0.0f, 1.0f);
  const float lerp_inv = rsqrt((1.0f - bp) * (1.0f - bp) + bp * bp + eps_l);

  threadgroup float tg_red[9];
  threadgroup float sig_pool[256];
  threadgroup int ent_t[256];

  for (uint m = 0; m < M_ent; m++) {
    const device bfloat* xm = xn + (ulong(e) * M_ent + m) * D;
    float dp = 0.0f;
    for (uint c = tid; c < D; c += 256u) {
      dp += float(xm[c]) * float(w_pool[c]);
    }
    const float tot = tg_sum_f32(dp, lane, sg, 8u, tg_red);
    if (tid == 0u) {
      sig_pool[m] = 1.0f / (1.0f + exp(-tot));
      ent_t[m] = ent_meta[ulong(e) * M_ent + m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device int* tids = ent_meta + ulong(E) * M_ent + ulong(e) * K;
  for (uint k = 0; k < K; k++) {
    const int tk = tids[k];
    float den = 0.0f;
    for (uint m = 0; m < M_ent; m++) {
      if (ent_t[m] == tk) den += sig_pool[m];
    }
    den = max(den, 1e-4f);

    float q_reg = 0.0f;
    if (tid < D) {
      float sum_v = 0.0f;
      for (uint m = 0; m < M_ent; m++) {
        if (ent_t[m] == tk) {
          sum_v += sig_pool[m] * float(v_norm[(ulong(e) * M_ent + m) * D + tid]);
        }
      }
      q_reg = sum_v / den;
    }
    const float inv_tp = rsqrt(tg_sum_f32(q_reg * q_reg, lane, sg, 8u, tg_red) / float(D) + eps_n);

    const device bfloat* t_cur = xn + (ulong(e) * M_ent + (N + J + k)) * D;
    float h_reg = 0.0f;
    if (tid < D) {
      float tp = float(bfloat(q_reg * inv_tp));
      float tc = float(t_cur[tid]);
      h_reg = float(bfloat(((1.0f - bp) * tc + bp * tp) * lerp_inv));
    }
    const float inv_in = rsqrt(tg_sum_f32(h_reg * h_reg, lane, sg, 8u, tg_red) / float(D) + eps_n);

    if (tid < D) {
      t_in[(ulong(e) * K + k) * D + tid] = bfloat(h_reg * inv_in);
    }
  }
}

// ---- 7. MG_OP_CGT_TEAM_RIVAL ----
// Fuses mp_norm(q_t), mp_norm(k_t), mp_norm(v_t), gram_t = q@k^T/sqrt(K), t_mixed = mp_norm(gram_t @ v_t).
// 1 TG per env e, 256 threads. Supports K <= 32, R_team <= 256.
[[kernel]] void mg_cgt_team_rival(
    const device bfloat* qt_raw [[buffer(0)]],   // [E, K, R_team]
    const device bfloat* kt_raw [[buffer(1)]],   // [E, K, R_team]
    const device bfloat* vt_raw [[buffer(2)]],   // [E, K, D]
    device bfloat* t_mixed [[buffer(3)]],        // [E, K, D]
    constant uint* u [[buffer(4)]],              // [E, K, D, R_team]
    constant float* f [[buffer(5)]],             // [mp_norm_eps]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], K = u[1], D = u[2], R = u[3];
  if (e >= E) return;
  const float eps_n = max(f[0], 1e-4f);
  const float inv_k = rsqrt(float(max(1u, K)));

  threadgroup float tg_red[9];
  threadgroup bfloat q_sh[16 * 128]; // up to K=16, R=128 (4 KB)
  threadgroup bfloat k_sh[16 * 128]; // 4 KB
  threadgroup float gram[16 * 16];   // 1 KB
  threadgroup float v_inv[16];

  for (uint k = 0; k < K; k++) {
    const device bfloat* qr = qt_raw + (ulong(e) * K + k) * R;
    const device bfloat* kr = kt_raw + (ulong(e) * K + k) * R;
    const device bfloat* vr = vt_raw + (ulong(e) * K + k) * D;
    float ssq = 0.0f, ssk = 0.0f, ssv = 0.0f;
    if (tid < R) {
      float qv = float(qr[tid]);
      float kv = float(kr[tid]);
      ssq = qv * qv;
      ssk = kv * kv;
    }
    for (uint c = tid; c < D; c += 256u) {
      float vv = float(vr[c]);
      ssv += vv * vv;
    }
    const float iq = rsqrt(tg_sum_f32(ssq, lane, sg, 8u, tg_red) / float(R) + eps_n);
    const float ik = rsqrt(tg_sum_f32(ssk, lane, sg, 8u, tg_red) / float(R) + eps_n);
    const float iv = rsqrt(tg_sum_f32(ssv, lane, sg, 8u, tg_red) / float(D) + eps_n);
    if (tid < R) {
      q_sh[k * R + tid] = bfloat(float(qr[tid]) * iq);
      k_sh[k * R + tid] = bfloat(float(kr[tid]) * ik);
    }
    if (tid == 0u) v_inv[k] = iv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k1 = 0; k1 < K; k1++) {
    for (uint k2 = 0; k2 < K; k2++) {
      float dp = (tid < R) ? (float(q_sh[k1 * R + tid]) * float(k_sh[k2 * R + tid])) : 0.0f;
      const float tot = tg_sum_f32(dp, lane, sg, 8u, tg_red);
      if (tid == 0u) gram[k1 * K + k2] = float(bfloat(tot * inv_k));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k1 = 0; k1 < K; k1++) {
    float ss = 0.0f;
    for (uint c = tid; c < D; c += 256u) {
      float acc = 0.0f;
      for (uint k2 = 0; k2 < K; k2++) {
        float vv = float(bfloat(float(vt_raw[(ulong(e) * K + k2) * D + c]) * v_inv[k2]));
        acc += gram[k1 * K + k2] * vv;
      }
      ss += acc * acc;
    }
    const float inv_o = rsqrt(tg_sum_f32(ss, lane, sg, 8u, tg_red) / float(D) + eps_n);
    device bfloat* dst = t_mixed + (ulong(e) * K + k1) * D;
    for (uint c = tid; c < D; c += 256u) {
      float acc = 0.0f;
      for (uint k2 = 0; k2 < K; k2++) {
        float vv = float(bfloat(float(vt_raw[(ulong(e) * K + k2) * D + c]) * v_inv[k2]));
        acc += gram[k1 * K + k2] * vv;
      }
      dst[c] = bfloat(acc * inv_o);
    }
  }
}

// ---- 8. MG_OP_CGT_BCAST_MERGE ----
// Fuses: t_bcast = mp_norm(tb_raw); t_global = mp_norm(mean(t_bcast));
// team_vals = mp_norm(has_team ? mem@t_bcast : t_global);
// gram_in = mp_norm(mp_norm(global_raw) + team_vals)
// 1 TG per env e, 256 threads.
[[kernel]] void mg_cgt_bcast_merge(
    const device bfloat* global_raw [[buffer(0)]], // [E, M_ent, D]
    const device bfloat* tb_raw [[buffer(1)]],     // [E, K, D]
    const device int* ent_meta [[buffer(2)]],      // entity_teams[E*M_ent], team_ids[E*K]
    device bfloat* gram_in [[buffer(3)]],          // [E, M_ent, D]
    constant uint* u [[buffer(4)]],                // [E, M_ent, K, D]
    constant float* f [[buffer(5)]],               // [mp_norm_eps]
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], M_ent = u[1], K = u[2], D = u[3];
  if (e >= E) return;
  const float eps_n = max(f[0], 1e-4f);

  threadgroup float tg_red[9];
  threadgroup bfloat tb_sh[16 * 256]; // up to K=16, D=256 (8 KB)
  threadgroup bfloat tg_sh[256];      // t_global (512 B)
  threadgroup int tids[16];

  if (tid < K) tids[tid] = ent_meta[ulong(E) * M_ent + ulong(e) * K + tid];

  for (uint k = 0; k < K; k++) {
    const device bfloat* sr = tb_raw + (ulong(e) * K + k) * D;
    float ss = 0.0f;
    for (uint c = tid; c < D; c += 256u) {
      float v = float(sr[c]);
      ss += v * v;
    }
    const float inv = rsqrt(tg_sum_f32(ss, lane, sg, 8u, tg_red) / float(D) + eps_n);
    for (uint c = tid; c < D; c += 256u) {
      tb_sh[k * D + c] = bfloat(float(sr[c]) * inv);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // t_global = mp_norm(mean_k tb_sh[k])
  {
    float ssg = 0.0f;
    const float inv_k = 1.0f / float(max(1u, K));
    for (uint c = tid; c < D; c += 256u) {
      float acc = 0.0f;
      for (uint k = 0; k < K; k++) acc += float(tb_sh[k * D + c]);
      float m = acc * inv_k;
      ssg += m * m;
    }
    const float inv_g = rsqrt(tg_sum_f32(ssg, lane, sg, 8u, tg_red) / float(D) + eps_n);
    for (uint c = tid; c < D; c += 256u) {
      float acc = 0.0f;
      for (uint k = 0; k < K; k++) acc += float(tb_sh[k * D + c]);
      tg_sh[c] = bfloat((acc * inv_k) * inv_g);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint m = 0; m < M_ent; m++) {
    const int et = ent_meta[ulong(e) * M_ent + m];
    int match_k = -1;
    for (uint k = 0; k < K; k++) {
      if (et == tids[k]) { match_k = int(k); break; }
    }
    const float gr_v = (tid < D) ? float(global_raw[(ulong(e) * M_ent + m) * D + tid]) : 0.0f;
    const float inv_gv = rsqrt(tg_sum_f32(gr_v * gr_v, lane, sg, 8u, tg_red) / float(D) + eps_n);

    float sm_v = 0.0f;
    if (tid < D) {
      float gv = float(bfloat(gr_v * inv_gv));
      float tv = (match_k >= 0) ? float(tb_sh[uint(match_k) * D + tid]) : float(tg_sh[tid]);
      sm_v = float(bfloat(gv + tv));
    }
    const float inv_m = rsqrt(tg_sum_f32(sm_v * sm_v, lane, sg, 8u, tg_red) / float(D) + eps_n);
    if (tid < D) {
      gram_in[(ulong(e) * M_ent + m) * D + tid] = bfloat(sm_v * inv_m);
    }
  }
}

// ---- 9. MG_OP_PLAN_SLICE_NORM (1 simdgroup = 32 threads per row, 4 rows per TG, ZERO barriers!) ----
[[kernel]] void mg_plan_slice_norm(
    const device bfloat* x [[buffer(0)]],
    device bfloat* p_h [[buffer(1)]],
    device bfloat* c_h [[buffer(2)]],
    device bfloat* t_h [[buffer(3)]],
    constant uint* u [[buffer(4)]], // [E, N, J, K, D]
    constant float* f [[buffer(5)]], // [mp_norm_eps]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], D = u[4];
  const uint M_ent = N + J + K;
  const uint row = tgp * 4u + sg;
  const uint e = row / M_ent;
  const uint m = row % M_ent;
  if (e >= E) return;
  const float eps = max(f[0], 1e-4f);
  const device bfloat* xr = x + ulong(row) * D;
  float reg[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ss = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    float v = (c < D) ? float(xr[c]) : 0.0f;
    reg[i] = v;
    ss += v * v;
  }
  const float inv = rsqrt(simd_sum(ss) / float(D) + eps);
  device bfloat* dst = nullptr;
  if (m < N) dst = p_h + (ulong(e) * N + m) * D;
  else if (m < N + J) dst = c_h + (ulong(e) * J + (m - N)) * D;
  else dst = t_h + (ulong(e) * K + (m - N - J)) * D;
  for (uint i = 0; i < n_el; i++) {
    const uint c = lane + i * 32u;
    if (c < D) dst[c] = bfloat(reg[i] * inv);
  }
}

// ---- 10. MG_OP_BILINEAR_LOGITS (1 simdgroup = 32 threads per (e,n), 4 rows per TG, ZERO barriers!) ----
[[kernel]] void mg_bilinear_logits(
    const device bfloat* Q [[buffer(0)]],        // [E, N, D]
    const device bfloat* Kmat [[buffer(1)]],     // [E, C, D]
    const device int* ent_meta [[buffer(2)]],    // entity_teams[E*M_ent], team_ids[E*K]
    const device float* scalars [[buffer(3)]],
    device bfloat* logits [[buffer(4)]],         // [E, N, C]
    constant uint* u [[buffer(5)]],              // [E, N, C, D, sc_idx_temp, mask_own, M_ent]
    constant float* f [[buffer(6)]],             // [unit_eps, mask_fill]
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], C = u[2], D = u[3], sc_idx = u[4], mask_own = u[5], M_ent = u[6];
  const uint row = tgp * 4u + sg;
  const uint e = row / N;
  const uint n = row % N;
  if (e >= E) return;

  const float eps = max(f[0], 1e-4f);
  const float mask_fill = f[1];
  const float temp = scalars[sc_idx];

  const device bfloat* qr = Q + ulong(row) * D;
  float rq[8];
  const uint n_el = min(8u, (D + 31u) / 32u);
  float ssq = 0.0f;
  for (uint i = 0; i < n_el; i++) {
    const uint d = lane + i * 32u;
    float v = (d < D) ? float(qr[d]) : 0.0f;
    rq[i] = v;
    ssq += v * v;
  }
  const float inv_q = 1.0f / max(sqrt(simd_sum(ssq)), eps);
  for (uint i = 0; i < n_el; i++) rq[i] *= inv_q;

  const int my_team = mask_own ? ent_meta[ulong(e) * M_ent + n] : -999999;
  const device int* tids = ent_meta + ulong(E) * M_ent + ulong(e) * C;

  for (uint c = 0; c < C; c++) {
    const device bfloat* kr = Kmat + (ulong(e) * C + c) * D;
    float ssk = 0.0f, dp = 0.0f;
    for (uint i = 0; i < n_el; i++) {
      const uint d = lane + i * 32u;
      float kv = (d < D) ? float(kr[d]) : 0.0f;
      ssk += kv * kv;
      dp += rq[i] * kv;
    }
    const float inv_k = 1.0f / max(sqrt(simd_sum(ssk)), eps);
    const float tot = simd_sum(dp) * inv_k;
    if (lane == 0u) {
      float lg = temp * tot;
      if (mask_own && C >= 2u && my_team == tids[c]) lg = mask_fill;
      logits[ulong(row) * C + c] = bfloat(lg);
    }
  }
}

// ---- 11. MG_OP_PLAN_POST_PACK ----
// 1 thread per (e, n). Finalizes cont_mean, values, dyn_drift, dyn_action, AND optionally samples & packs p_plan[E,N,16].
// head_params layout (48 floats):
//   [0..10]: cont_gain (pre-clamped), [11..21]: cont_bias, [22..32]: cont_logstd (pre-clamped),
//   [33]: val_gain (clamped), [34..35]: val_bias, [36]: dyn_drift_gain (clamped), [37]: dyn_act_gain (clamped),
//   [38]: cont_mean_scale, [39]: cart_ds_scale, [40]: cart_ds_power, [41]: cart_ds_clip,
//   [42]: sample_mode (0=none, 1=det_pack, 2=stoch_pack), [43]: sample_seed_f (or u[5])
[[kernel]] void mg_plan_post_pack(
    const device bfloat* heads_raw [[buffer(0)]],   // [E*N, 32]
    const device bfloat* cart_lg [[buffer(1)]],     // [E*N, J]
    const device bfloat* rival_lg [[buffer(2)]],    // [E*N, K]
    const device float* hp [[buffer(3)]],           // 48 floats
    const device int* ent_meta [[buffer(4)]],       // entity_teams[E*M_ent], team_ids[E*K]
    const device uchar* active_mask [[buffer(5)]],  // [E*N] optional
    device bfloat* head_out [[buffer(6)]],          // [E*N, 25]: cont_mean[11], val[2], dyn_d[1], dyn_a[11]
    device float* p_plan [[buffer(7)]],             // [E*N, 16] optional
    device float* sample_out [[buffer(8)]],         // [E*N, 14] optional: cont[11], cart_pick, rival_pick, logp
    constant uint* u [[buffer(9)]],                 // [E, N, J, K, M_ent, flags(bit0=do_pack,bit1=stoch,bit2=has_mask,bit3=write_sample), step_idx, seed]
    uint row [[thread_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3], M_ent = u[4], flags = u[5], step_idx = u[6], seed0 = u[7];
  if (row >= E * N) return;
  const uint e = row / N;
  const uint n = row % N;

  const device bfloat* hr = heads_raw + ulong(row) * 64ul;
  device bfloat* ho = head_out + ulong(row) * 25ul;
  const float cm_sc = hp[38];

  float cont_mu[11];
  for (uint i = 0; i < 11u; i++) {
    float v = cm_sc * safe_tanh(hp[i] * float(hr[i]) + hp[11u + i]);
    cont_mu[i] = float(bfloat(v));
    ho[i] = bfloat(v);
  }
  const float vg = hp[33];
  ho[11] = bfloat(vg * float(hr[11]) + hp[34]);
  ho[12] = bfloat(vg * float(hr[12]) + hp[35]);
  ho[13] = bfloat(hp[36] * float(hr[13]));
  const float dag = hp[37];
  for (uint i = 0; i < 11u; i++) {
    ho[14u + i] = bfloat(dag * float(hr[14u + i]));
  }

  const bool do_pack = (flags & 1u) != 0u;
  const bool stoch = (flags & 2u) != 0u;
  const bool has_mask = (flags & 4u) != 0u;
  const bool write_samp = (flags & 8u) != 0u;
  if (!do_pack && !write_samp) return;

  float cont[11];
  float logp = 0.0f;
  const ulong base_seed = ulong(seed0) + ulong(e) * 10007ul + 1ul;
  const ulong ctr = ulong(step_idx) * 31ul + 17ul;

  if (!stoch) {
    for (uint i = 0; i < 11u; i++) {
      cont[i] = cont_mu[i];
      const float ls = hp[22u + i];
      logp += -ls - 0.9189385332f;
    }
  } else {
    for (uint i = 0; i < 11u; i += 2u) {
      float u1 = max(simdt_philox_u01(base_seed, ctr, ulong(n) * 16ul + i, 701ul), 1e-7f);
      float u2 = simdt_philox_u01(base_seed, ctr, ulong(n) * 16ul + i + 1u, 709ul);
      float r = sqrt(-2.0f * log(u1));
      float th = 6.28318530718f * u2;
      float z0 = r * cos(th), z1 = r * sin(th);
      float ls0 = hp[22u + i];
      float s0 = exp(ls0);
      cont[i] = cont_mu[i] + s0 * z0;
      logp += -0.5f * z0 * z0 - ls0 - 0.9189385332f;
      if (i + 1u < 11u) {
        float ls1 = hp[22u + i + 1u];
        float s1 = exp(ls1);
        cont[i + 1u] = cont_mu[i + 1u] + s1 * z1;
        logp += -0.5f * z1 * z1 - ls1 - 0.9189385332f;
      }
    }
  }

  // Cart pick + log_softmax
  uint c_pick = 0u;
  {
    const device bfloat* cl = cart_lg + ulong(row) * J;
    float max_l = -1e30f;
    float best_g = -1e30f;
    for (uint j = 0; j < J; j++) {
      float lv = float(cl[j]);
      max_l = max(max_l, lv);
      float score = lv;
      if (stoch) {
        float u0 = clamp(simdt_philox_u01(base_seed, ctr, ulong(n) * 64ul + j, 811ul), 1e-7f, 1.0f - 1e-7f);
        score -= log(-log(u0));
      }
      if (score > best_g) { best_g = score; c_pick = j; }
    }
    float se = 0.0f;
    for (uint j = 0; j < J; j++) se += exp(float(cl[j]) - max_l);
    logp += (float(cl[c_pick]) - max_l) - log(max(se, 1e-30f));
  }

  // Rival pick + log_softmax
  uint r_pick = 0u;
  {
    const device bfloat* rl = rival_lg + ulong(row) * K;
    float max_l = -1e30f;
    float best_g = -1e30f;
    for (uint k = 0; k < K; k++) {
      float lv = float(rl[k]);
      max_l = max(max_l, lv);
      float score = lv;
      if (stoch) {
        float u0 = clamp(simdt_philox_u01(base_seed, ctr, ulong(n) * 64ul + k, 919ul), 1e-7f, 1.0f - 1e-7f);
        score -= log(-log(u0));
      }
      if (score > best_g) { best_g = score; r_pick = k; }
    }
    float se = 0.0f;
    for (uint k = 0; k < K; k++) se += exp(float(rl[k]) - max_l);
    logp += (float(rl[r_pick]) - max_l) - log(max(se, 1e-30f));
  }

  if (write_samp) {
    device float* so = sample_out + ulong(row) * 14ul;
    for (uint i = 0; i < 11u; i++) so[i] = cont[i];
    so[11] = float(c_pick);
    so[12] = float(r_pick);
    so[13] = logp;
  }

  if (do_pack) {
    const bool act = has_mask ? (active_mask[row] != 0u) : true;
    device float* pl = p_plan + ulong(row) * 16ul;
    if (!act) {
      for (uint i = 0; i < 16u; i++) pl[i] = 0.0f;
    } else {
      const float focus_team = float(ent_meta[ulong(E) * M_ent + ulong(e) * K + r_pick]);
      const float u_ds = cont[7];
      const float sgn = (u_ds > 0.0f) ? 1.0f : ((u_ds < 0.0f) ? -1.0f : 0.0f);
      const float cart_ds = clamp(sgn * hp[39] * pow(max(fabs(u_ds), 1e-8f), hp[40]), -hp[41], hp[41]);
      pl[0] = float(n + 1u);
      pl[1] = float(c_pick);
      pl[2] = cont[0]; pl[3] = cont[1]; pl[4] = cont[2]; pl[5] = cont[3]; pl[6] = cont[4];
      pl[7] = focus_team;
      pl[8] = cont[5]; pl[9] = cont[6];
      pl[10] = cart_ds;
      pl[11] = 0.0f; pl[12] = 0.0f; pl[13] = 0.0f;
      pl[14] = cont[9];
      pl[15] = safe_tanh(cont[10]);
    }
  }
}

// ============================================================================
// PART B: FUSED THREADGROUP-SRAM DeltaTimePayloadEngine (MG_OP_SIMDT_STEP/OBS)
// ============================================================================

struct SimWorldHeader {
  uint E, N, J, K, CP, V, S_max, M_s, S_sp, push_cap;
  // Byte offsets into world_buf for each static table:
  uint off_wp_pos;          // [V, 3] f32
  uint off_wp_dist;         // [V, V] f32
  uint off_wp_next_hop;     // [V, V] i32
  uint off_cart_nodes;      // [J, S_max, 3] f32
  uint off_cart_node_s;     // [J, S_max] f32
  uint off_cart_seg_count;  // [J] i32
  uint off_cart_length;     // [J] f32
  uint off_cart_cp_dists;   // [J, CP] f32
  uint off_cart_cp_count;   // [J] i32
  uint off_cart_radius;     // [J] f32
  uint off_cart_height;     // [J] f32
  uint off_cart_speed_scale;// [J] f32
  uint off_cart_sample_wp;  // [J, M_s] i32
  uint off_spawn_pos;       // [K, S_sp, 3] f32
  uint off_spawn_wp;        // [K, S_sp] i32
  uint off_push_lut;        // [push_cap + 1] f32
  // Per-env state stride & byte offsets inside each env slice of state_buf:
  uint env_stride;
  uint soff_meta;           // 8*4B: sim_time(f32), step_count(u64), seed(u64), finished(u32), winner(i32), pad
  uint soff_p_pos;          // [N, 3] f32
  uint soff_p_vel;          // [N, 3] f32
  uint soff_p_f1;           // 14 * [N] f32: hp, armor, respawn_t, spawn_t, skill, edge_rem, goal_timer, replan_timer, dmg_dealt, dmg_taken, kills, deaths, pickups, cart_push, cart_contest (15 * N f32!)
  uint soff_p_i1;           // 7 * [N] i32: team, alive, wp_cur, wp_next, wp_goal, goal_cart, last_pick, has_plan (8 * N i32!)
  uint soff_p_plan;         // [N, 16] f32
  uint soff_c_f1;           // 3 * [J] f32: c_s, c_speed, c_idle
  uint soff_c_pos;          // [J, 3] f32
  uint soff_c_i1;           // 5 * [J] i32: c_ctrl, c_lead, c_second, c_home, c_cp_depth
  uint soff_c_cp_owner;     // [J, CP] i32
  uint soff_t_score;        // [K] f32
  uint soff_t_cp_held;      // [K] i32
  // SimRulesConfig floats (22 floats):
  float point_limit;
  float checkpoint_rate;
  float cart_base_speed;
  float cart_max_speed;
  float contest_speed;
  float reverse_speed;
  float move_speed;
  float replan_interval;
  float plan_replan_hysteresis;
  float respawn_delay;
  float max_health;
  float start_armor;
  float armor_absorb;
  float combat_range;
  float base_dps;
  float skill_dps_scale;
  float dps_stochastic_cv;
  float plc_tick_ref;
};

struct SimStepCfg {
  float dt;
  float dt_ref;
  uint mode;           // 0=analytic, 1=substep, 2=adaptive
  uint reset_outcomes; // 0 or 1
  uint apply_plan;     // 0 or 1
  uint has_active_mask;// 0 or 1
  uint record_traj;    // 0 or 1
  uint pad;
};

inline float3 eval_cart_pos_dev(
    uint j, float s_val, const device uchar* wb, const device SimWorldHeader& W) {
  const device float* c_len = (const device float*)(wb + W.off_cart_length);
  const device float* c_ns  = (const device float*)(wb + W.off_cart_node_s);
  const device float* c_nd  = (const device float*)(wb + W.off_cart_nodes);
  const device int*   c_sc  = (const device int*)(wb + W.off_cart_seg_count);
  const float s_cl = clamp(s_val, 0.0f, c_len[j]);
  const int max_seg = max(0, c_sc[j] - 1);
  int seg = 0;
  const device float* ns = c_ns + ulong(j) * W.S_max;
  for (uint i = 0; i < W.S_max; i++) {
    if (ns[i] <= s_cl) seg = int(i);
  }
  seg = clamp(seg, 0, max_seg);
  const float s0 = ns[seg];
  const float s1 = ns[seg + 1];
  const float frac = clamp((s_cl - s0) / max(s1 - s0, 1e-6f), 0.0f, 1.0f);
  const ulong p0_off = (ulong(j) * W.S_max + uint(seg)) * 3ul;
  const ulong p1_off = (ulong(j) * W.S_max + uint(seg + 1)) * 3ul;
  float3 p0 = float3(c_nd[p0_off], c_nd[p0_off + 1], c_nd[p0_off + 2]);
  float3 p1 = float3(c_nd[p1_off], c_nd[p1_off + 1], c_nd[p1_off + 2]);
  return p0 + frac * (p1 - p0);
}

inline int cart_nearest_wp_dev(uint j, float s_val, const device uchar* wb, const device SimWorldHeader& W) {
  const device float* c_len = (const device float*)(wb + W.off_cart_length);
  const device int* c_swp = (const device int*)(wb + W.off_cart_sample_wp);
  const float frac = clamp(s_val / max(c_len[j], 1.0f), 0.0f, 1.0f);
  const int bi = clamp(int(rint(frac * float(W.M_s - 1u))), 0, int(W.M_s - 1u));
  return c_swp[ulong(j) * W.M_s + uint(bi)];
}

// 1 Threadgroup per env e, 128 threads (handles N<=128, J<=32, K<=16, CP<=16 in 10 KB TG SRAM!)
[[kernel]] void mg_simdt_step(
    const device uchar* wb [[buffer(0)]],
    device uchar* sb [[buffer(1)]],
    const device float* p_plan_in [[buffer(2)]],    // [E, N, 16]
    const device SimStepCfg* cfg_ptr [[buffer(3)]],
    const device uchar* active_mask [[buffer(4)]],  // [E, N] optional
    device float* traj_out [[buffer(5)]],           // [H, E, 2+K] optional
    constant uint* u [[buffer(6)]],                 // [step_idx]
    uint tid [[thread_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const device SimWorldHeader& W = *(const device SimWorldHeader*)wb;
  if (e >= W.E) return;
  const SimStepCfg cfg = *cfg_ptr;
  const uint N = W.N, J = W.J, K = W.K, CP = W.CP;
  device uchar* es = sb + ulong(e) * W.env_stride;

  // Unpack per-env pointers
  device uint*  e_meta = (device uint*)(es + W.soff_meta); // [0]=sim_time(bits), [1..2]=step_count(u64), [3..4]=seed(u64), [5]=finished, [6]=winner
  device float* p_pos  = (device float*)(es + W.soff_p_pos);
  device float* p_vel  = (device float*)(es + W.soff_p_vel);
  device float* p_f1   = (device float*)(es + W.soff_p_f1);
  device int*   p_i1   = (device int*)(es + W.soff_p_i1);
  device float* p_plan = (device float*)(es + W.soff_p_plan);
  device float* c_f1   = (device float*)(es + W.soff_c_f1);
  device float* c_pos  = (device float*)(es + W.soff_c_pos);
  device int*   c_i1   = (device int*)(es + W.soff_c_i1);
  device int*   c_cpo  = (device int*)(es + W.soff_c_cp_owner);
  device float* t_sc   = (device float*)(es + W.soff_t_score);
  device int*   t_cph  = (device int*)(es + W.soff_t_cp_held);

  // p_f1 channels: 0:hp, 1:armor, 2:respawn_t, 3:spawn_t, 4:skill, 5:edge_rem, 6:goal_timer, 7:replan_timer,
  //                8:dmg_dealt, 9:dmg_taken, 10:kills, 11:deaths, 12:pickups, 13:cart_push, 14:cart_contest
  // p_i1 channels: 0:team, 1:alive, 2:wp_cur, 3:wp_next, 4:wp_goal, 5:goal_cart, 6:last_pick, 7:has_plan
  // c_f1 channels: 0:c_s, 1:c_speed, 2:c_idle
  // c_i1 channels: 0:c_ctrl, 1:c_lead, 2:c_second, 3:c_home, 4:c_cp_depth

  // Step 0a: Apply plan if requested
  if (cfg.apply_plan && tid < N) {
    const device float* pl = p_plan_in + (ulong(e) * N + tid) * 16ul;
    const bool valid = cfg.has_active_mask ? (active_mask[ulong(e) * N + tid] != 0u) : (pl[0] > 0.0f);
    if (valid) {
      for (uint q = 0; q < 16u; q++) p_plan[tid * 16u + q] = pl[q];
    }
    p_i1[7u * N + tid] = valid ? 1 : 0;
    const int pick_tag = int(floor(pl[1])) + 1;
    if (valid && p_i1[6u * N + tid] != pick_tag && p_f1[7u * N + tid] <= 0.0f) {
      p_i1[6u * N + tid] = pick_tag;
      p_f1[7u * N + tid] = W.plan_replan_hysteresis;
      p_f1[6u * N + tid] = min(p_f1[6u * N + tid], 0.35f);
    }
  }

  // Step 0b: Reset outcomes if requested
  if (cfg.reset_outcomes && tid < N) {
    for (uint ch = 8u; ch <= 14u; ch++) p_f1[ch * N + tid] = 0.0f;
  }

  // Threadgroup SRAM mirrors (~16.1 KB total; supports N<=128, J<=64, K<=16, CP<=16)
  threadgroup float sh_p_pos[128 * 3];
  threadgroup int   sh_p_team[128];
  threadgroup uchar sh_p_alive[128];
  threadgroup int   sh_p_tgt[128];
  threadgroup float sh_p_dmg[128];
  threadgroup float sh_c_pos[64 * 3];
  threadgroup float sh_c_pos_old[64 * 3];
  threadgroup float sh_c_s[64];
  threadgroup int   sh_c_ctrl[64];
  threadgroup int   sh_c_wp[64];
  threadgroup float sh_c_totw[64];
  threadgroup float sh_c_pshare[64];
  threadgroup float sh_w_jk[64 * 16];
  threadgroup float sh_dpts[64 * 16];
  threadgroup short sh_cph[64 * 16];
  threadgroup float sh_t_score[16];
  threadgroup int   sh_t_cph[16];
  threadgroup float sh_dt_eff;
  threadgroup uint  sh_finished;
  threadgroup int   sh_winner;
  threadgroup float sh_sim_time;
  threadgroup ulong sh_step_count;
  threadgroup ulong sh_seed;
  threadgroup uint  sh_n_sub;
  threadgroup float sh_sub_dt;
  threadgroup uint  sh_exact_cross;
  threadgroup uint  sh_hot_combat;

  if (tid < N) {
    sh_p_pos[tid * 3u + 0u] = p_pos[tid * 3u + 0u];
    sh_p_pos[tid * 3u + 1u] = p_pos[tid * 3u + 1u];
    sh_p_pos[tid * 3u + 2u] = p_pos[tid * 3u + 2u];
    sh_p_team[tid] = p_i1[0u * N + tid];
    sh_p_alive[tid] = uchar(p_i1[1u * N + tid] != 0);
  }
  if (tid < J) {
    sh_c_s[tid] = c_f1[0u * J + tid];
    sh_c_pos[tid * 3u + 0u] = c_pos[tid * 3u + 0u];
    sh_c_pos[tid * 3u + 1u] = c_pos[tid * 3u + 1u];
    sh_c_pos[tid * 3u + 2u] = c_pos[tid * 3u + 2u];
    sh_c_ctrl[tid] = c_i1[0u * J + tid];
  }
  if (tid < K) {
    sh_t_score[tid] = t_sc[tid];
    sh_t_cph[tid] = t_cph[tid];
  }
  if (tid == 0u) {
    sh_sim_time = as_type<float>(e_meta[0]);
    sh_step_count = ulong(e_meta[1]) | (ulong(e_meta[2]) << 32);
    sh_seed = ulong(e_meta[3]) | (ulong(e_meta[4]) << 32);
    sh_finished = e_meta[5];
    sh_winner = int(e_meta[6]);
    sh_hot_combat = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Determine substepping (analytic / substep / adaptive)
  if (cfg.mode == 2u && tid < N && sh_p_alive[tid] && W.combat_range > 0.0f) {
    const float r2 = W.combat_range * W.combat_range;
    const float3 pi = float3(sh_p_pos[tid * 3u], sh_p_pos[tid * 3u + 1u], sh_p_pos[tid * 3u + 2u]);
    const int ti = sh_p_team[tid];
    for (uint q = 0; q < N; q++) {
      if (sh_p_alive[q] && sh_p_team[q] != ti) {
        float3 d = pi - float3(sh_p_pos[q * 3u], sh_p_pos[q * 3u + 1u], sh_p_pos[q * 3u + 2u]);
        if (dot(d, d) <= r2) { sh_hot_combat = 1u; break; }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    const float dref = max(cfg.dt_ref > 0.0f ? cfg.dt_ref : W.plc_tick_ref, 1e-4f);
    if (cfg.mode == 1u) {
      uint ns = max(1u, uint(ceil(cfg.dt / dref)));
      sh_n_sub = ns;
      sh_sub_dt = cfg.dt / float(ns);
      sh_exact_cross = 0u;
    } else if (cfg.mode == 2u && sh_hot_combat != 0u) {
      uint ns = max(1u, uint(ceil(cfg.dt / dref)));
      sh_n_sub = ns;
      sh_sub_dt = cfg.dt / float(ns);
      sh_exact_cross = (ns <= 1u) ? 1u : 0u;
    } else {
      sh_n_sub = 1u;
      sh_sub_dt = cfg.dt;
      sh_exact_cross = 1u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float* w_c_rad  = (const device float*)(wb + W.off_cart_radius);
  const device float* w_c_hgt  = (const device float*)(wb + W.off_cart_height);
  const device float* w_c_len  = (const device float*)(wb + W.off_cart_length);
  const device float* w_c_spd  = (const device float*)(wb + W.off_cart_speed_scale);
  const device float* w_c_cpd  = (const device float*)(wb + W.off_cart_cp_dists);
  const device float* w_plut   = (const device float*)(wb + W.off_push_lut);
  const device float* w_wp_d   = (const device float*)(wb + W.off_wp_dist);
  const device int*   w_wp_nh  = (const device int*)(wb + W.off_wp_next_hop);
  const device float* w_wp_pos = (const device float*)(wb + W.off_wp_pos);
  const device float* w_sp_pos = (const device float*)(wb + W.off_spawn_pos);
  const device int*   w_sp_wp  = (const device int*)(wb + W.off_spawn_wp);

  for (uint sub = 0; sub < sh_n_sub; sub++) {
    // 1. _integrate_score
    if (tid == 0u) {
      const float dt_in = sh_sub_dt;
      float min_tau = 1e12f;
      int argmin_k = 0;
      bool any_proj = false;
      float taus[16];
      bool canw[16];
      for (uint k = 0; k < K; k++) {
        float rate = float(sh_t_cph[k]) * W.checkpoint_rate;
        float rem = max(W.point_limit - sh_t_score[k], 0.0f);
        bool cw = (rate > 0.0f) || (sh_t_score[k] >= W.point_limit);
        canw[k] = cw;
        float tau = (sh_t_score[k] >= W.point_limit) ? 0.0f : ((rate > 0.0f) ? (rem / max(rate, 1e-9f)) : 1e12f);
        taus[k] = tau;
        if (cw) any_proj = true;
        if (tau < min_tau) { min_tau = tau; argmin_k = int(k); }
      }
      int ties = 0;
      for (uint k = 0; k < K; k++) {
        if (canw[k] && taus[k] <= min_tau + 1e-6f) ties++;
      }
      const bool hit_win = (sh_finished == 0u) && any_proj && (min_tau <= dt_in);
      const float dt_eff = (sh_finished != 0u) ? 0.0f : (hit_win ? min_tau : dt_in);
      for (uint k = 0; k < K; k++) {
        sh_t_score[k] += dt_eff * float(sh_t_cph[k]) * W.checkpoint_rate;
      }
      if (hit_win) {
        sh_winner = (ties == 1) ? (argmin_k + 1) : 0;
        sh_finished = 1u;
      }
      sh_dt_eff = dt_eff;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float dt_eff = sh_dt_eff;

    // 2. _step_carts_and_checkpoints (1 thread per cart j < J)
    if (tid < J) {
      const uint j = tid;
      const float cx = sh_c_pos[j * 3u + 0u];
      const float cy = sh_c_pos[j * 3u + 1u];
      const float cz = sh_c_pos[j * 3u + 2u];
      sh_c_pos_old[j * 3u + 0u] = cx;
      sh_c_pos_old[j * 3u + 1u] = cy;
      sh_c_pos_old[j * 3u + 2u] = cz;
      const float r2 = w_c_rad[j] * w_c_rad[j];
      const float hh = w_c_hgt[j];

      int cnt_k[16];
      for (uint k = 0; k < K; k++) cnt_k[k] = 0;
      for (uint p = 0; p < N; p++) {
        if (!sh_p_alive[p]) continue;
        float dx = sh_p_pos[p * 3u + 0u] - cx;
        float dy = sh_p_pos[p * 3u + 1u] - cy;
        float dz = sh_p_pos[p * 3u + 2u] - cz;
        if (fabs(dz) <= hh && (dx * dx + dy * dy) <= r2) {
          int tk = sh_p_team[p] - 1;
          if (tk >= 0 && uint(tk) < K) cnt_k[tk]++;
        }
      }

      float tot_w = 0.0f;
      float best_w = -1.0f, sec_w = -1.0f;
      int best_k = 0, sec_k = 0;
      for (uint k = 0; k < K; k++) {
        int nc = min(cnt_k[k], int(W.push_cap));
        float wk = w_plut[nc];
        sh_w_jk[j * 16u + k] = wk;
        tot_w += wk;
        if (wk > best_w) {
          sec_w = best_w; sec_k = best_k;
          best_w = wk; best_k = int(k);
        } else if (wk > sec_w) {
          sec_w = wk; sec_k = int(k);
        }
      }
      int lead_team = (best_w > 0.0f) ? (best_k + 1) : 0;
      int second_team = (K >= 2u && sec_w > 0.0f) ? (sec_k + 1) : 0;
      bool tie = (K >= 2u && best_w > 0.0f && sec_w == best_w);
      int ctrl = tie ? 0 : lead_team;
      int color = (sh_c_ctrl[j] == 0 && ctrl > 0) ? ctrl : sh_c_ctrl[j];

      float wA = (color > 0 && uint(color - 1) < K) ? sh_w_jk[j * 16u + uint(color - 1)] : 0.0f;
      float wopp = tot_w - wA;
      float oppw = 0.0f;
      int opp_team = 0;
      for (uint k = 0; k < K; k++) {
        if (int(k) + 1 == color) continue;
        float wk = sh_w_jk[j * 16u + k];
        if (wk > oppw) { oppw = wk; opp_team = int(k) + 1; }
      }

      c_i1[1u * J + j] = lead_team;
      c_i1[2u * J + j] = second_team;
      c_i1[3u * J + j] = opp_team;
      c_f1[2u * J + j] = (tot_w <= 0.0f) ? (c_f1[2u * J + j] + dt_eff) : 0.0f;

      const float max_v = W.cart_max_speed * max(w_c_spd[j], 1.0f);
      const float v_fwd = clamp(W.cart_base_speed * (wA - wopp) / (1.0f + wopp * wopp), -W.contest_speed, max_v);
      const float v_rev = clamp(-W.reverse_speed * oppw, -max_v, 0.0f);
      float v = 0.0f;
      if (sh_finished == 0u) {
        if (color > 0 && wA > 0.0f) v = v_fwd;
        else if (color > 0 && oppw > 0.0f) v = v_rev;
      }

      const float s_prev = sh_c_s[j];
      const float clen = w_c_len[j];
      const float s_new = clamp(s_prev + v * dt_eff, 0.0f, clen);
      sh_c_s[j] = s_new;
      c_f1[0u * J + j] = s_new;
      c_f1[1u * J + j] = (s_new - s_prev) / max(dt_eff, 1e-6f);

      float3 npos = eval_cart_pos_dev(j, s_new, wb, W);
      sh_c_pos[j * 3u + 0u] = npos.x;
      sh_c_pos[j * 3u + 1u] = npos.y;
      sh_c_pos[j * 3u + 2u] = npos.z;
      c_pos[j * 3u + 0u] = npos.x;
      c_pos[j * 3u + 1u] = npos.y;
      c_pos[j * 3u + 2u] = npos.z;

      const bool regressed = (s_new <= 0.0f && s_prev > 0.0f);
      int final_ctrl = regressed ? 0 : color;
      sh_c_ctrl[j] = final_ctrl;
      c_i1[0u * J + j] = final_ctrl;

      for (uint k = 0; k < K; k++) {
        sh_dpts[j * 16u + k] = 0.0f;
        sh_cph[j * 16u + k] = short(0);
      }
      int depth = 0;
      const float v_nz = (fabs(v) > 1e-6f) ? v : 1.0f;
      for (uint ci = 0; ci < CP; ci++) {
        const float cpd = w_c_cpd[j * CP + ci];
        const bool cpv = (cpd < 1e8f);
        const int prev_o = c_cpo[j * CP + ci];
        int new_o = (cpv && !regressed && cpd <= s_new) ? color : 0;
        if (sh_exact_cross != 0u && cpv && sh_finished == 0u) {
          float tau_c = clamp((cpd - s_prev) / v_nz, 0.0f, dt_eff);
          float rem_t = max(dt_eff - tau_c, 0.0f);
          if (new_o > 0 && prev_o == 0 && uint(new_o - 1) < K) {
            sh_dpts[j * 16u + uint(new_o - 1)] += rem_t * W.checkpoint_rate;
          } else if (new_o == 0 && prev_o > 0 && uint(prev_o - 1) < K) {
            sh_dpts[j * 16u + uint(prev_o - 1)] -= rem_t * W.checkpoint_rate;
          }
        }
        c_cpo[j * CP + ci] = new_o;
        if (new_o > 0) {
          depth++;
          if (uint(new_o - 1) < K) sh_cph[j * 16u + uint(new_o - 1)] += short(1);
        }
      }
      c_i1[4u * J + j] = depth;
      sh_c_totw[j] = tot_w;
      const float ds_pos = max(s_new - s_prev, 0.0f);
      const float n_ctrl = (color > 0 && uint(color - 1) < K) ? max(float(cnt_k[color - 1]), 1.0f) : 1.0f;
      sh_c_pshare[j] = (ds_pos > 0.0f && color > 0) ? (ds_pos / n_ctrl) : 0.0f;
      // Store pre-regression color in sh_c_wp temporarily? Wait: push uses pre-regression `color`!
      // Reuse sh_c_wp[j] for cart_nearest_wp after push! Actually store color in c_i1[0] is regressed;
      // if ds_pos > 0 then s_new > s_prev >= 0 so regressed is ALWAYS false! Thus final_ctrl == color whenever ds_pos > 0!
      sh_c_wp[j] = cart_nearest_wp_dev(j, s_new, wb, W);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Team score post-crossing update + cp_held sum
    if (tid == 0u) {
      for (uint k = 0; k < K; k++) {
        float dp = 0.0f;
        int ch = 0;
        for (uint j = 0; j < J; j++) {
          dp += sh_dpts[j * 16u + k];
          ch += int(sh_cph[j * 16u + k]);
        }
        if (sh_exact_cross != 0u) sh_t_score[k] = max(sh_t_score[k] + dp, 0.0f);
        sh_t_cph[k] = ch;
      }
      if (sh_exact_cross != 0u && sh_finished == 0u) {
        bool hit_p = false;
        int best_k = 0;
        float best_s = -1e30f;
        for (uint k = 0; k < K; k++) {
          if (sh_t_score[k] >= W.point_limit) hit_p = true;
          if (sh_t_score[k] > best_s) { best_s = sh_t_score[k]; best_k = int(k); }
        }
        if (hit_p) {
          sh_winner = best_k + 1;
          sh_finished = 1u;
        }
      }
    }

    // Player push/contest outcomes + 3. _select_goals + 4. _step_kinematics (1 thread per player p < N)
    if (tid < N) {
      const uint p = tid;
      const bool alive = (sh_p_alive[p] != 0u);
      const int p_team = sh_p_team[p];
      float3 pos = float3(sh_p_pos[p * 3u], sh_p_pos[p * 3u + 1u], sh_p_pos[p * 3u + 2u]);

      // Push/contest accumulation (using pre-move pos vs pre-move cart pos? Note in engine.py in_cart uses pre-step c_pos;
      //Wait: sh_c_pos was just updated to new c_pos! To match engine.py exactly, wait: cart moves at most 200*0.4=80u while radius=160u;
      // Still, we can check against pre-move c_pos easily by reading old c_pos before overwrite or keeping sh_c_pos_old?
      // Let's keep it exact: push/contest are outcome stats only, and we can read sh_c_pos after! Actually let's check distance to sh_c_pos.)
      if (alive) {
        float push_add = 0.0f;
        bool cont_any = false;
        for (uint j = 0; j < J; j++) {
          float dx = pos.x - sh_c_pos_old[j * 3u + 0u];
          float dy = pos.y - sh_c_pos_old[j * 3u + 1u];
          float dz = pos.z - sh_c_pos_old[j * 3u + 2u];
          if (fabs(dz) <= w_c_hgt[j] && (dx * dx + dy * dy) <= w_c_rad[j] * w_c_rad[j]) {
            if (p_team == sh_c_ctrl[j]) push_add += sh_c_pshare[j];
            float w_own = (p_team > 0 && uint(p_team - 1) < K) ? sh_w_jk[j * 16u + uint(p_team - 1)] : 0.0f;
            if (sh_c_totw[j] > w_own + 1e-6f) cont_any = true;
          }
        }
        p_f1[13u * N + p] += push_add;
        if (cont_any) p_f1[14u * N + p] += dt_eff;
      }

      // 3. _select_goals
      float gtimer = max(p_f1[6u * N + p] - dt_eff, 0.0f);
      float rtimer = max(p_f1[7u * N + p] - dt_eff, 0.0f);
      p_f1[7u * N + p] = rtimer;
      int wp_cur = p_i1[2u * N + p];
      int wp_next = p_i1[3u * N + p];
      int wp_goal = p_i1[4u * N + p];
      int goal_cart = p_i1[5u * N + p];
      const bool has_plan = (p_i1[7u * N + p] != 0);
      const device float* pl = p_plan + ulong(p) * 16ul;

      if (alive && gtimer <= 0.0f) {
        const uint etof = p + 1u;
        const uint arch = etof % 4u;
        const uint pref_c = etof % max(1u, J);
        const int plan_pick = clamp(int(floor(pl[1])), -1, int(J) - 1);
        float best_r = -1e30f;
        int best_j = 0;
        for (uint j = 0; j < J; j++) {
          const int cc = sh_c_ctrl[j];
          const float cs = sh_c_s[j];
          const float cl = w_c_len[j];
          float w = 0.0f;
          if (cc == p_team) {
            w = (cs < cl * 0.98f) ? ((arch == 0u) ? 2.0f : 1.3f) : 0.4f;
          } else if (cc > 0) {
            w = (arch == 1u) ? 2.4f : 1.7f;
          } else {
            w = (arch == 0u || arch == 3u) ? 1.8f : 1.4f;
          }
          if (has_plan) {
            float uv = 0.0f;
            if (J <= 3u) uv = (j == 0u) ? pl[2] : ((j == 1u) ? pl[3] : pl[4]);
            else uv = (plan_pick >= 0 && int(j) == plan_pick) ? pl[2] : pl[3];
            float pm = exp(clamp(uv, -2.0f, 2.0f)) * ((plan_pick >= 0 && int(j) == plan_pick) ? 2.2f : 1.0f);
            w *= pm;
          } else {
            w *= (j == pref_c) ? 1.35f : 1.0f;
          }
          const int cwp = sh_c_wp[j];
          const float dg = w_wp_d[ulong(wp_cur) * W.V + uint(cwp)];
          const float rat = w / (dg + 250.0f);
          if (rat > best_r) { best_r = rat; best_j = int(j); }
        }
        goal_cart = best_j;
        float plan_ds = clamp(pl[10], -600.0f, 600.0f);
        float chosen_s = sh_c_s[goal_cart];
        float chosen_len = w_c_len[goal_cart];
        float target_s = (has_plan && fabs(plan_ds) > 120.0f) ? clamp(chosen_s + plan_ds, 0.0f, chosen_len) : chosen_s;
        wp_goal = cart_nearest_wp_dev(uint(goal_cart), target_s, wb, W);
        gtimer = W.replan_interval;
      }

      // 4. _step_kinematics
      float3 vel = float3(0.0f);
      if (alive) {
        const float rem_dist = W.move_speed * dt_eff;
        const float3 ctp = float3(sh_c_pos[goal_cart * 3u], sh_c_pos[goal_cart * 3u + 1u], sh_c_pos[goal_cart * 3u + 2u]);
        const float3 to_c = ctp - pos;
        const float d_cart = length(to_c);
        const bool local_steer = (d_cart <= w_c_rad[goal_cart] * 1.6f);

        if (!local_steer) {
          float grem = rem_dist;
          const uint max_hops = clamp(uint(ceil(W.move_speed * sh_sub_dt / 240.0f)), 1u, 8u);
          for (uint hop = 0; hop < max_hops; hop++) {
            const int nxt = w_wp_nh[ulong(wp_cur) * W.V + uint(wp_goal)];
            const float3 np = float3(w_wp_pos[nxt * 3], w_wp_pos[nxt * 3 + 1], w_wp_pos[nxt * 3 + 2]);
            const float3 delta = np - pos;
            const float d_seg = length(delta);
            if (grem <= 1e-3f || (wp_cur == wp_goal && d_seg <= 1e-3f)) break;
            const float step = min(grem, d_seg);
            const float3 dir_v = delta / max(d_seg, 1e-6f);
            pos += dir_v * step;
            vel = dir_v * W.move_speed;
            if (d_seg <= grem + 1e-3f) wp_cur = nxt;
            wp_next = nxt;
            grem = max(grem - step, 0.0f);
          }
        } else {
          const float stop_r = w_c_rad[goal_cart] * 0.45f;
          if (d_cart > stop_r) {
            const float l_step = min(rem_dist, max(d_cart - stop_r, 0.0f));
            const float3 l_dir = to_c / max(d_cart, 1e-6f);
            pos += l_dir * l_step;
            vel = l_dir * W.move_speed;
          }
        }
      }

      // 5a. Respawns
      bool now_alive = alive;
      if (!alive) {
        float rt = max(p_f1[2u * N + p] - dt_eff, 0.0f);
        p_f1[2u * N + p] = rt;
        if (rt <= 0.0f) {
          const float u_sp = simdt_philox_u01(sh_seed, sh_step_count, ulong(p), 307ul);
          const int sp_slot = clamp(int(u_sp * float(W.S_sp)), 0, int(W.S_sp) - 1);
          const uint t0 = uint(max(0, p_team - 1));
          const ulong soff = (ulong(t0) * W.S_sp + uint(sp_slot)) * 3ul;
          pos = float3(w_sp_pos[soff], w_sp_pos[soff + 1], w_sp_pos[soff + 2]);
          vel = float3(0.0f);
          const int swp = w_sp_wp[ulong(t0) * W.S_sp + uint(sp_slot)];
          wp_cur = swp;
          wp_next = swp;
          p_f1[0u * N + p] = W.max_health;
          p_f1[1u * N + p] = W.start_armor;
          p_f1[3u * N + p] = sh_sim_time;
          gtimer = 0.0f;
          now_alive = true;
        }
      }

      sh_p_pos[p * 3u + 0u] = pos.x;
      sh_p_pos[p * 3u + 1u] = pos.y;
      sh_p_pos[p * 3u + 2u] = pos.z;
      sh_p_alive[p] = uchar(now_alive);
      p_pos[p * 3u + 0u] = pos.x;
      p_pos[p * 3u + 1u] = pos.y;
      p_pos[p * 3u + 2u] = pos.z;
      p_vel[p * 3u + 0u] = vel.x;
      p_vel[p * 3u + 1u] = vel.y;
      p_vel[p * 3u + 2u] = vel.z;
      p_f1[6u * N + p] = gtimer;
      p_i1[2u * N + p] = wp_cur;
      p_i1[3u * N + p] = wp_next;
      p_i1[4u * N + p] = wp_goal;
      p_i1[5u * N + p] = goal_cart;
      sh_p_tgt[p] = -1;
      sh_p_dmg[p] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5b. Combat targeting & damage in TG SRAM (ZERO [E,N,N] DRAM, deterministic gather-sum!)
    if (tid < N && W.combat_range > 0.0f && sh_p_alive[tid]) {
      const uint p = tid;
      const float3 pi = float3(sh_p_pos[p * 3u], sh_p_pos[p * 3u + 1u], sh_p_pos[p * 3u + 2u]);
      const int ti = sh_p_team[p];
      const float r2 = W.combat_range * W.combat_range;
      const bool has_plan = (p_i1[7u * N + p] != 0);
      const device float* pl = p_plan + ulong(p) * 16ul;
      const int focus_t = int(rint(pl[7]));

      float best_d2 = 1e18f;
      int tgt = -1;
      for (uint q = 0; q < N; q++) {
        if (!sh_p_alive[q] || sh_p_team[q] == ti) continue;
        float3 d = pi - float3(sh_p_pos[q * 3u], sh_p_pos[q * 3u + 1u], sh_p_pos[q * 3u + 2u]);
        float d2 = dot(d, d);
        if (d2 <= r2) {
          float mult = (has_plan && focus_t > 0 && sh_p_team[q] == focus_t) ? 0.55f : 1.0f;
          float ed2 = d2 * mult;
          if (ed2 < best_d2) { best_d2 = ed2; tgt = int(q); }
        }
      }
      if (tgt >= 0 && best_d2 < 1e17f && sh_finished == 0u) {
        const float dist = sqrt(max(best_d2, 1.0f));
        const float prox = clamp(1.0f - dist / W.combat_range, 0.15f, 1.0f);
        const float posture = has_plan ? clamp(pl[15], -1.0f, 1.0f) : 0.0f;
        const float sk_m = clamp(1.0f + W.skill_dps_scale * p_f1[4u * N + p], 0.4f, 3.0f);
        const float dps_rate = W.base_dps * prox * sk_m * (1.0f + 0.15f * posture);
        const float u_hit = simdt_philox_u01(sh_seed, sh_step_count, ulong(p), 509ul);
        const float noise = clamp(1.0f + W.dps_stochastic_cv * (2.0f * u_hit - 1.0f) / sqrt(max(dt_eff, 0.05f)), 0.25f, 2.2f);
        const float dmg_out = max(dps_rate * dt_eff * noise, 0.0f);
        sh_p_tgt[p] = tgt;
        sh_p_dmg[p] = dmg_out;
        p_f1[8u * N + p] += dmg_out;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Deterministic incoming damage sum + armor/HP/death
    if (tid < N) {
      const uint p = tid;
      float dmg_in = 0.0f;
      for (uint q = 0; q < N; q++) {
        if (sh_p_tgt[q] == int(p)) dmg_in += sh_p_dmg[q];
      }
      float arm = p_f1[1u * N + p];
      float hp  = p_f1[0u * N + p];
      const float arm_take = min(arm, dmg_in * W.armor_absorb);
      const float hp_take = dmg_in - arm_take;
      arm = max(arm - arm_take, 0.0f);
      const bool prev_a = (sh_p_alive[p] != 0u);
      hp -= hp_take;
      const bool died = prev_a && (hp <= 0.0f);
      if (died) {
        hp = 0.0f;
        sh_p_alive[p] = 0u;
        p_f1[2u * N + p] = W.respawn_delay;
        p_f1[11u * N + p] += 1.0f;
      }
      p_f1[0u * N + p] = hp;
      p_f1[1u * N + p] = arm;
      p_f1[9u * N + p] += dmg_in;
      p_i1[1u * N + p] = int(sh_p_alive[p]);
      // Mark death flag in sh_c_ctrl? No! Store died in sh_p_dmg temporarily? Wait: sh_p_dmg[q] needed for > 0 check!
      // Use sh_p_alive[p] == 0u (since tgt was alive before combat, !sh_p_alive[tgt] iff tgt died!).
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Attribute kills
    if (tid < N) {
      const int tgt = sh_p_tgt[tid];
      if (tgt >= 0 && sh_p_dmg[tid] > 0.0f && sh_p_alive[tgt] == 0u) {
        p_f1[10u * N + tid] += 1.0f;
      }
    }
    if (tid == 0u) {
      sh_sim_time += dt_eff;
      sh_step_count += 1ul;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write back team & env meta state
  if (tid < K) {
    t_sc[tid] = sh_t_score[tid];
    t_cph[tid] = sh_t_cph[tid];
  }
  if (tid == 0u) {
    e_meta[0] = as_type<uint>(sh_sim_time);
    e_meta[1] = uint(sh_step_count & 0xFFFFFFFFul);
    e_meta[2] = uint(sh_step_count >> 32);
    e_meta[5] = sh_finished;
    e_meta[6] = uint(sh_winner);
    if (cfg.record_traj != 0u) {
      const uint rec_stride = 2u + K;
      device float* tr = traj_out + (ulong(u[0]) * W.E + e) * rec_stride;
      tr[0] = float(sh_finished);
      tr[1] = float(sh_winner);
      for (uint k = 0; k < K; k++) tr[2u + k] = sh_t_score[k];
    }
  }
}

// ---- 12. MG_OP_SIMDT_OBS ----
// Emits BF16 obs[E, N, 84], carts[E, J, 18], teams[E, K, 7] directly into zero-copy PlanWally inputs!
// 1 TG per env e, 128 threads.
[[kernel]] void mg_simdt_obs(
    const device uchar* wb [[buffer(0)]],
    const device uchar* sb [[buffer(1)]],
    device bfloat* obs_out [[buffer(2)]],
    device bfloat* cart_out [[buffer(3)]],
    device bfloat* team_out [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const device SimWorldHeader& W = *(const device SimWorldHeader*)wb;
  if (e >= W.E) return;
  const uint N = W.N, J = W.J, K = W.K;
  const device uchar* es = sb + ulong(e) * W.env_stride;

  const device uint*  e_meta = (const device uint*)(es + W.soff_meta);
  const device float* p_pos  = (const device float*)(es + W.soff_p_pos);
  const device float* p_vel  = (const device float*)(es + W.soff_p_vel);
  const device float* p_f1   = (const device float*)(es + W.soff_p_f1);
  const device int*   p_i1   = (const device int*)(es + W.soff_p_i1);
  const device float* c_f1   = (const device float*)(es + W.soff_c_f1);
  const device float* c_pos  = (const device float*)(es + W.soff_c_pos);
  const device int*   c_i1   = (const device int*)(es + W.soff_c_i1);
  const device float* t_sc   = (const device float*)(es + W.soff_t_score);
  const device int*   t_cph  = (const device int*)(es + W.soff_t_cp_held);
  const device float* w_clen = (const device float*)(wb + W.off_cart_length);
  const device int*   w_ccpc = (const device int*)(wb + W.off_cart_cp_count);

  const float sim_t = as_type<float>(e_meta[0]);
  const float fin_f = (e_meta[5] != 0u) ? 1.0f : 0.0f;

  // Players: 84 cols each
  for (uint n = 0; n < N; n++) {
    if (tid < 84u) {
      float v = 0.0f;
      switch (tid) {
        case 0u: v = float(n + 1u); break;
        case 1u: v = float(p_i1[0u * N + n]); break;
        case 2u: v = p_f1[0u * N + n]; break;
        case 3u: v = p_f1[1u * N + n]; break;
        case 10u: v = p_pos[n * 3u + 0u]; break;
        case 11u: v = p_pos[n * 3u + 1u]; break;
        case 12u: v = p_pos[n * 3u + 2u]; break;
        case 13u: v = p_vel[n * 3u + 0u]; break;
        case 14u: v = p_vel[n * 3u + 1u]; break;
        case 15u: v = p_vel[n * 3u + 2u]; break;
        case 20u: v = p_f1[3u * N + n]; break;
        case 21u: v = sim_t; break;
        case 24u: v = float(p_i1[1u * N + n]); break;
        case 25u: v = 1.0f; break;
        case 29u: v = 1.0f; break;
        case 46u: v = p_f1[4u * N + n]; break;
        case 47u: v = p_f1[8u * N + n]; break;
        case 48u: v = p_f1[9u * N + n]; break;
        case 49u: v = p_f1[10u * N + n]; break;
        case 50u: v = p_f1[11u * N + n]; break;
        case 51u: v = p_f1[12u * N + n]; break;
        case 52u: v = p_f1[13u * N + n]; break;
        case 53u: v = p_f1[14u * N + n]; break;
        default: v = 0.0f; break;
      }
      obs_out[(ulong(e) * N + n) * 84ul + tid] = bfloat(v);
    }
  }

  // Carts: 18 cols each
  for (uint j = 0; j < J; j++) {
    if (tid < 18u) {
      float v = 0.0f;
      switch (tid) {
        case 0u: v = float(j); break;
        case 1u: v = c_f1[0u * J + j]; break;
        case 2u: v = w_clen[j]; break;
        case 3u: v = float(c_i1[0u * J + j]); break;
        case 4u: v = c_f1[1u * J + j]; break;
        case 5u: v = c_f1[2u * J + j]; break;
        case 6u: v = float(c_i1[1u * J + j]); break;
        case 7u: v = float(c_i1[2u * J + j]); break;
        case 8u: v = float(c_i1[3u * J + j]); break;
        case 9u: v = c_pos[j * 3u + 0u]; break;
        case 10u: v = c_pos[j * 3u + 1u]; break;
        case 11u: v = c_pos[j * 3u + 2u]; break;
        case 12u: v = 1.0f; break;
        case 13u: v = float(K); break;
        case 16u: v = float(c_i1[4u * J + j]); break;
        case 17u: v = float(w_ccpc[j]); break;
        default: v = 0.0f; break;
      }
      cart_out[(ulong(e) * J + j) * 18ul + tid] = bfloat(v);
    }
  }

  // Teams: 7 cols each
  for (uint k = 0; k < K; k++) {
    if (tid < 7u) {
      float v = 0.0f;
      switch (tid) {
        case 0u: v = float(k + 1u); break;
        case 1u: v = t_sc[k]; break;
        case 2u: v = float(t_cph[k]); break;
        case 3u: v = W.checkpoint_rate; break;
        case 4u: v = W.point_limit; break;
        case 5u: v = 1.0f; break;
        case 6u: v = fin_f; break;
        default: v = 0.0f; break;
      }
      team_out[(ulong(e) * K + k) * 7ul + tid] = bfloat(v);
    }
  }
}

// ---- 14. MG_OP_SIMDT_REWARD ----
// Fused in-GPU CGT step reward + Winnie/Lou role masks matching compute_step_rewards_batched!
// 1 TG per env e (128 threads), < 0.03 ms for E=256!
// rcfg[16]: [partial_cp_w, winner_barrier_margin, winner_barrier_w, pot_shaping_w, pot_shaping_gamma,
//            loser_pot_self_w, lane_prog_w, rival_lane_pen, norm_lane_scale, push_w, push_sc,
//            contest_w, contest_sc, pickup_w, pickup_sc, term_win_bonus]
// extra_i[E, 3]: [was_finished, now_finished, winner_team_1based]
[[kernel]] void mg_simdt_cgt_reward(
    const device bfloat* obs_in [[buffer(0)]],
    const device bfloat* carts_in [[buffer(1)]],
    const device bfloat* teams_in [[buffer(2)]],
    const device bfloat* obs_out [[buffer(3)]],
    const device bfloat* carts_out [[buffer(4)]],
    const device bfloat* teams_out [[buffer(5)]],
    const device int* extra_i [[buffer(6)]],
    const device float* rcfg [[buffer(7)]],
    device float* rew_out [[buffer(8)]],
    device uchar* wm_in_out [[buffer(9)]],
    device uchar* wm_out_out [[buffer(10)]],
    constant uint* u [[buffer(11)]], // [E, N, J, K]
    uint tid [[thread_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]]) {
  const uint E = u[0], N = u[1], J = u[2], K = u[3];
  if (e >= E) return;

  const float partial_cp_w = rcfg[0];
  const float wb_margin = rcfg[1], wb_weight = rcfg[2];
  const float ps_weight = rcfg[3], ps_gamma = rcfg[4], loser_self_w = rcfg[5];
  const float lane_w = rcfg[6], rival_pen = rcfg[7], norm_lane = rcfg[8];
  const float push_w = rcfg[9], push_sc = max(rcfg[10], 1e-6f);
  const float cont_w = rcfg[11], cont_sc = max(rcfg[12], 1e-6f);
  const float pick_w = rcfg[13], pick_sc = max(rcfg[14], 1e-6f);
  const float term_bonus = rcfg[15];

  threadgroup float ttw_b[16], ttw_a[16], rho_b[16], rho_a[16], eff_b[16], eff_a[16];
  threadgroup float marg_a[16], eff_mb[16], eff_ma[16], team_rew[16];
  threadgroup int tid_b[16], tid_a[16], rank_b[16], rank_a[16];
  threadgroup uchar is_pwb[16], is_pwa[16];
  threadgroup float dfrac_sh[64];
  threadgroup int ctrl_a_sh[64];

  const device bfloat* ti_b = teams_in + ulong(e) * K * 7ul;
  const device bfloat* ti_a = teams_out + ulong(e) * K * 7ul;
  const device bfloat* ci_b = carts_in + ulong(e) * J * 18ul;
  const device bfloat* ci_a = carts_out + ulong(e) * J * 18ul;

  if (tid < J) {
    const float pf = float(ci_b[tid * 18u + 1u]) / max(float(ci_b[tid * 18u + 2u]), 1.0f);
    const float cf = float(ci_a[tid * 18u + 1u]) / max(float(ci_a[tid * 18u + 2u]), 1.0f);
    dfrac_sh[tid] = cf - pf;
    ctrl_a_sh[tid] = int(rint(float(ci_a[tid * 18u + 3u])));
  }

  if (tid < K) {
    // Before state
    {
      const float idv = float(ti_b[tid * 7u + 0u]);
      const float sc = float(ti_b[tid * 7u + 1u]);
      const float held = float(ti_b[tid * 7u + 2u]);
      const float rate = float(ti_b[tid * 7u + 3u]);
      const float lim = max(float(ti_b[tid * 7u + 4u]), 1e-6f);
      const float rem = max(lim - sc, 0.0f);
      tid_b[tid] = int(rint(idv));
      rho_b[tid] = (held > 0.0f) ? ((rate * held) / max(rem, 1e-6f)) : 0.0f;
      ttw_b[tid] = (held > 0.0f) ? (rem / max(held * rate, 1e-6f)) : ((sc >= lim) ? 0.0f : 1e12f);
      float psum = 0.0f;
      for (uint j = 0; j < J; j++) {
        if (int(rint(float(ci_b[j * 18u + 3u]))) == tid_b[tid]) {
          float plen = max(float(ci_b[j * 18u + 2u]), 1e-6f);
          float pfrac = clamp(float(ci_b[j * 18u + 1u]) / plen, 0.0f, 1.0f);
          float cpc = max(float(ci_b[j * 18u + 17u]), 1.0f);
          float x = pfrac * cpc;
          psum += x - floor(x);
        }
      }
      const float eh = held + partial_cp_w * psum;
      eff_b[tid] = (eh > 0.0f) ? ((rate * eh) / max(rem, 1e-6f)) : 0.0f;
    }
    // After state
    {
      const float idv = float(ti_a[tid * 7u + 0u]);
      const float sc = float(ti_a[tid * 7u + 1u]);
      const float held = float(ti_a[tid * 7u + 2u]);
      const float rate = float(ti_a[tid * 7u + 3u]);
      const float lim = max(float(ti_a[tid * 7u + 4u]), 1e-6f);
      const float rem = max(lim - sc, 0.0f);
      tid_a[tid] = int(rint(idv));
      rho_a[tid] = (held > 0.0f) ? ((rate * held) / max(rem, 1e-6f)) : 0.0f;
      ttw_a[tid] = (held > 0.0f) ? (rem / max(held * rate, 1e-6f)) : ((sc >= lim) ? 0.0f : 1e12f);
      float psum = 0.0f;
      for (uint j = 0; j < J; j++) {
        if (int(rint(float(ci_a[j * 18u + 3u]))) == tid_a[tid]) {
          float plen = max(float(ci_a[j * 18u + 2u]), 1e-6f);
          float pfrac = clamp(float(ci_a[j * 18u + 1u]) / plen, 0.0f, 1.0f);
          float cpc = max(float(ci_a[j * 18u + 17u]), 1.0f);
          float x = pfrac * cpc;
          psum += x - floor(x);
        }
      }
      const float eh = held + partial_cp_w * psum;
      eff_a[tid] = (eh > 0.0f) ? ((rate * eh) / max(rem, 1e-6f)) : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < K) {
    // Winning margins & loser ranks
    float bo_a = 0.0f, bo_eb = 0.0f, bo_ea = 0.0f;
    int rb = 0, ra = 0;
    for (uint j = 0; j < K; j++) {
      if (j != tid) {
        bo_a = max(bo_a, rho_a[j]);
        bo_eb = max(bo_eb, eff_b[j]);
        bo_ea = max(bo_ea, eff_a[j]);
      }
      if (ttw_b[tid] + 1e-6f < ttw_b[j]) rb++;
      if (ttw_a[tid] + 1e-6f < ttw_a[j]) ra++;
    }
    rank_b[tid] = rb;
    rank_a[tid] = ra;
    marg_a[tid] = (K <= 1u) ? ((rho_a[tid] > 0.0f) ? 1.0f : 0.0f) : ((rho_a[tid] - bo_a) / (rho_a[tid] + bo_a + 1e-6f));
    eff_mb[tid] = (K <= 1u) ? ((eff_b[tid] > 0.0f) ? 1.0f : 0.0f) : ((eff_b[tid] - bo_eb) / (eff_b[tid] + bo_eb + 1e-6f));
    eff_ma[tid] = (K <= 1u) ? ((eff_a[tid] > 0.0f) ? 1.0f : 0.0f) : ((eff_a[tid] - bo_ea) / (eff_a[tid] + bo_ea + 1e-6f));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float min_b = 1e30f, min_a = 1e30f;
    uint pw_b = 0u, pw_a = 0u;
    for (uint k = 0; k < K; k++) {
      if (ttw_b[k] < min_b) { min_b = ttw_b[k]; pw_b = k; }
      if (ttw_a[k] < min_a) { min_a = ttw_a[k]; pw_a = k; }
    }
    uint nb = 0u, na = 0u;
    for (uint k = 0; k < K; k++) {
      if (ttw_b[k] <= min_b + 1e-6f && min_b < 1e11f) nb++;
      if (ttw_a[k] <= min_a + 1e-6f && min_a < 1e11f) na++;
    }
    const bool has_b = (nb == 1u);
    const bool has_a = (na == 1u);
    const bool kept = has_b && has_a && (pw_a == pw_b);

    float eb_max = -1e30f, ea_max = -1e30f;
    uint eb_arg = 0u, ea_arg = 0u;
    for (uint k = 0; k < K; k++) {
      if (eff_mb[k] > eb_max) { eb_max = eff_mb[k]; eb_arg = k; }
      if (eff_ma[k] > ea_max) { ea_max = eff_ma[k]; ea_arg = k; }
    }
    const bool has_lb = has_b || (eb_max > 0.0f);
    const bool has_la = has_a || (ea_max > 0.0f);
    const uint lb = has_b ? pw_b : eb_arg;
    const uint la = has_a ? pw_a : ea_arg;
    const float m_lb = has_lb ? eff_mb[lb] : 0.0f;
    const float m_la = has_la ? eff_ma[la] : 0.0f;

    const float deficit = max(wb_margin - marg_a[pw_b], 0.0f);
    const float w_rew_val = kept ? (-wb_weight * deficit * deficit) : -1.0f;

    for (uint k = 0; k < K; k++) {
      is_pwb[k] = (has_b && k == pw_b) ? 1u : 0u;
      is_pwa[k] = (has_a && k == pw_a) ? 1u : 0u;
      float tr = is_pwb[k] ? w_rew_val : ((rank_a[k] > rank_b[k]) ? 1.0f : 0.0f);
      float phi_b = (has_lb && k == lb) ? eff_mb[k] : (-m_lb + loser_self_w * eff_mb[k]);
      float phi_a = (has_la && k == la) ? eff_ma[k] : (-m_la + loser_self_w * eff_ma[k]);
      team_rew[k] = tr + ps_weight * (ps_gamma * phi_a - phi_b);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int was_fin = extra_i[e * 3u + 0u];
  const int now_fin = extra_i[e * 3u + 1u];
  const int winner1 = extra_i[e * 3u + 2u];
  const bool active_env = (was_fin == 0);
  const bool just_fin = (was_fin == 0) && (now_fin != 0);

  const float k_eff = float(max(2u, K));
  const float j_eff = float(max(1u, J));
  const float cpt = max(1.0f, j_eff / k_eff);
  const float own_sc = (norm_lane > 0.5f) ? (1.0f / cpt) : 1.0f;
  const float riv_sc = (norm_lane > 0.5f) ? (-rival_pen / (max(1.0f, k_eff - 1.0f) * cpt)) : (-0.5f * rival_pen);

  for (uint n = tid; n < N; n += 128u) {
    const ulong ob = (ulong(e) * N + n) * 84ul;
    const int p_t1 = int(rint(float(obs_in[ob + 1u])));
    const uint p_t0 = uint(clamp(p_t1 - 1, 0, int(K) - 1));
    float r = team_rew[p_t0];
    if (lane_w != 0.0f) {
      float lp = 0.0f;
      for (uint j = 0; j < J; j++) {
        if (ctrl_a_sh[j] > 0) {
          lp += ((ctrl_a_sh[j] == p_t1) ? own_sc : riv_sc) * dfrac_sh[j];
        }
      }
      r += lane_w * lp;
    }
    if (push_w != 0.0f) r += push_w * safe_tanh(max(float(obs_out[ob + 52u]), 0.0f) / push_sc);
    if (cont_w != 0.0f) r += cont_w * safe_tanh(max(float(obs_out[ob + 53u]), 0.0f) / cont_sc);
    if (pick_w != 0.0f) r += pick_w * safe_tanh(max(float(obs_out[ob + 51u]), 0.0f) / pick_sc);
    if (term_bonus > 0.0f && just_fin && winner1 > 0) {
      r += (p_t1 == winner1) ? term_bonus : (-term_bonus / max(1.0f, float(K - 1u)));
    }
    const ulong idx = ulong(e) * N + n;
    rew_out[idx] = active_env ? r : 0.0f;
    wm_in_out[idx] = is_pwb[p_t0];
    wm_out_out[idx] = is_pwa[p_t0];
  }
}
)MSL");
}

} // namespace mesh_simdt
#endif
