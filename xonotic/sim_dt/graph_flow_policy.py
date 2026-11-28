"""Graph-Native + Exact DPP + 1-NFE Mealy-Jet Flow Policy & Advantage-Weighted Boltzmann Flow Matching.

Architectural contract:
  - Directed navigation waypoint graph (nav_nodes + geodesic nav_kern) in BOTH afferent input and efferent output.
  - Continuous steering velocity efferent u in R^{A_dim} (A_dim = W_nav + K + 4), integrated by the sim_dt
    exact exponential-relaxation state-space integrator r(t+dt) = e^{-dt/tau}*r(t) + tau*(1-e^{-dt/tau})*u(t).
    At u = 0, bots continuously relax to unmodified full-action-space stock HavocBot.
  - Current integrated accumulator r_acc fed back into afferent input (1-to-1 input<->output correspondence,
    zero network recurrence).
  - Shared wide entity mixing backbone: Directed Nav-Graph Diffusion + Global/Team CGT Gram +
    Restored Exact FP32 Per-Team DPP Diversity Kernel B(I + B^T B)^{-1} B^T V + wide MP-SwiGLU FFN.
  - ZERO cart_logits, ZERO rival_logits, ZERO discrete cart entropy, ZERO magic mode count M,
    ZERO binary coordinate coupling masks, 1 backbone NFE via 2nd-order Mealy-Jet Flow Efferent.
  - Trained with Advantage-Weighted Boltzmann Flow Matching + Jet-Straightening Loss:
    unique global optimum pi*(u|s) proportional to pi_old(u|s) * exp(A(s, u) / alpha), preserving
    all plural high-EV strategies across G > 1 counterfactual rollout groups.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from payload.tools.strategy_io_schema import OBS_WIDTH
from solver.strat.train_config import NormConfig


NAV_FEAT_IN = 16
NAV_PRE_DIM = 20
OBS_PRE_DIM = 96
CART_PRE_DIM = 22
TEAM_PRE_DIM = 9


@dataclass(frozen=True)
class GraphFlowConfig:
    d: int = 256
    depth: int = 4
    h: int = 1024
    r_gram: int = 128
    r_team: int = 64
    r_dpp: int = 16
    d_e: int = 32
    c_flow_dim: int = 64
    flow_hid: int = 64
    heun_steps: int = 6
    train_taus: tuple[float, ...] = (0.12, 0.36, 0.64, 0.88)
    sigma_base: float = 0.55
    recurrent: bool = True
    type_blend_init_logit: float = -1.8
    aff_blend_init_logit: float = -1.2
    nav_blend_init_logit: float = -1.1
    dpp_blend_init_logit: float = -1.1
    step_blend_init: float = 0.35
    value_gain_init: float = 0.5
    value_gain_bounds: tuple[float, float] = (0.05, 5.0)
    dyn_drift_gain_init: float = 0.25
    dyn_action_gain_init: float = 0.25
    dyn_gain_bounds: tuple[float, float] = (0.05, 2.0)
    ess_target: float = 0.55
    flow_w_min: float = 0.05
    flow_w_max: float = 8.0
    beta_bounds: tuple[float, float] = (0.25, 3.5)
    zero_anchor_lambda: float = 0.25
    v_max: float = 3.0
    ridge_lambda: float = 0.02
    basis_rec_weight: float = 0.20
    u_clip: float = 6.0


def compute_ess_boltzmann_weights(
    adv: torch.Tensor,
    actor_mask: torch.Tensor,
    *,
    u_sq_mean: torch.Tensor | None = None,
    zero_anchor_lambda: float = 0.0,
    ess_target: float = 0.55,
    w_min: float = 0.05,
    w_max: float = 8.0,
    beta_min: float = 0.25,
    beta_max: float = 3.5,
    group_size: int = 1,
    n_bisect: int = 16,
) -> tuple[torch.Tensor, float, float]:
    """Solve for Boltzmann inverse-temperature beta = 1/alpha via exact bisection to hold target ESS.

    Standardizes `adv` FIRST on active actors, THEN folds `-zero_anchor_lambda * u_sq_mean` so the
    confining stock-HavocBot quadratic energy is never cancelled out by batch re-centering/scaling.
    Excludes empty groups from bisection and alpha reporting; uses max-shifted unclipped logits so
    ESS is strictly monotonic in beta.
    Returns (w_norm [E, N] mean=1 on active actors, achieved ess_frac after bounds, mean alpha).
    """
    am = actor_mask.float()
    am_sum = am.sum().clamp_min(1.0)
    adv_f = adv.float()
    adv_mean = (adv_f * am).sum() / am_sum
    adv_var = ((adv_f - adv_mean).square() * am).sum() / am_sum
    a_norm = ((adv_f - adv_mean) / adv_var.clamp_min(1e-6).sqrt()).clamp(-6.0, 6.0)
    if u_sq_mean is not None and float(zero_anchor_lambda) > 0.0:
        score = a_norm - float(zero_anchor_lambda) * u_sq_mean.float()
    else:
        score = a_norm

    E, N = score.shape
    target = max(0.15, min(0.95, float(ess_target)))
    b_lo = max(1e-3, float(beta_min))
    b_hi = max(b_lo, float(beta_max))

    if group_size > 1 and E % group_size == 0:
        B = E // group_size
        sg = score.view(B, group_size * N)
        mg = am.view(B, group_size * N)
        cnt_raw = mg.sum(dim=-1, keepdim=True)
        has_act = cnt_raw > 0.5
        cnt = cnt_raw.clamp_min(1.0)
        lo = torch.full((B, 1), b_lo, device=adv.device, dtype=torch.float32)
        hi = torch.full((B, 1), b_hi, device=adv.device, dtype=torch.float32)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            logits = mid * sg
            lmax = logits.masked_fill(mg < 0.5, -1e9).max(dim=-1, keepdim=True).values
            w_u = torch.where(mg > 0.5, torch.exp((logits - lmax).clamp(-40.0, 0.0)), torch.zeros_like(logits))
            w = (w_u / w_u.sum(dim=-1, keepdim=True).clamp_min(1e-8) * cnt).clamp(w_min, w_max) * mg
            ess = w.sum(dim=-1, keepdim=True).square() / (w.square().sum(dim=-1, keepdim=True).clamp_min(1e-8) * cnt)
            too_flat = ess > target
            lo = torch.where(too_flat, mid, lo)
            hi = torch.where(too_flat, hi, mid)
        beta = 0.5 * (lo + hi)
        logits = beta * sg
        lmax = logits.masked_fill(mg < 0.5, -1e9).max(dim=-1, keepdim=True).values
        w_u = torch.where(mg > 0.5, torch.exp((logits - lmax).clamp(-40.0, 0.0)), torch.zeros_like(logits))
        w = (w_u / w_u.sum(dim=-1, keepdim=True).clamp_min(1e-8) * cnt).clamp(w_min, w_max) * mg
        w_flat = w.view(E, N)
        if bool(has_act.any().item()):
            mean_alpha = float((1.0 / beta[has_act]).mean().item())
        else:
            mean_alpha = 1.0 / b_lo
    else:
        lo_s, hi_s = b_lo, b_hi
        has_any = bool((am > 0.5).any().item())
        if has_any:
            for _ in range(n_bisect):
                mid_s = 0.5 * (lo_s + hi_s)
                logits = mid_s * score
                lmax_s = logits.masked_fill(am < 0.5, -1e9).max()
                w_u = torch.where(am > 0.5, torch.exp((logits - lmax_s).clamp(-40.0, 0.0)), torch.zeros_like(logits))
                w = (w_u / w_u.sum().clamp_min(1e-8) * am_sum).clamp(w_min, w_max) * am
                ess_s = float((w.sum().square() / (w.square().sum().clamp_min(1e-8) * am_sum)).item())
                if ess_s > target:
                    lo_s = mid_s
                else:
                    hi_s = mid_s
            beta_s = 0.5 * (lo_s + hi_s)
            logits = beta_s * score
            lmax_s = logits.masked_fill(am < 0.5, -1e9).max()
            w_u = torch.where(am > 0.5, torch.exp((logits - lmax_s).clamp(-40.0, 0.0)), torch.zeros_like(logits))
            w_flat = (w_u / w_u.sum().clamp_min(1e-8) * am_sum).clamp(w_min, w_max) * am
            mean_alpha = 1.0 / beta_s
        else:
            w_flat = torch.zeros_like(am)
            mean_alpha = 1.0 / b_lo

    w_norm = w_flat / (w_flat.sum() / am_sum).clamp_min(1e-6)
    ess_frac = float((w_norm.sum().square() / (w_norm.square().sum().clamp_min(1e-8) * am_sum)).item())
    return w_norm, ess_frac, mean_alpha


def _mp_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + max(float(eps), 1e-4))
    return (xf * inv).to(x.dtype)


def _mp_unit_vec(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.square().sum(dim=-1, keepdim=True) + max(float(eps), 1e-6))
    return (xf * inv).to(x.dtype)


def _mp_lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    af, bf = a.float(), b.float()
    tf = t.float().clamp(0.0, 1.0)
    inv = torch.rsqrt((1.0 - tf).square() + tf.square() + max(float(eps), 1e-4))
    return (((1.0 - tf) * af + tf * bf) * inv).to(a.dtype)


def _mp_silu(x: torch.Tensor, scale: float = 0.596) -> torch.Tensor:
    xf = x.float()
    return (F.silu(xf) / float(scale)).to(x.dtype)


def _unit_linear(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    wu = F.normalize(weight.float(), p=2, dim=-1, eps=max(float(eps), 1e-4)).to(x.dtype)
    return F.linear(x, wu)


def _tau_embed(tau: torch.Tensor) -> torch.Tensor:
    """8-dim smooth time embedding for decreasing flow time tau in [0, 1]."""
    t = tau.float().unsqueeze(-1)
    pi_t = math.pi * t
    return torch.cat(
        (
            t,
            1.0 - t,
            torch.sin(pi_t),
            torch.cos(pi_t),
            torch.sin(2.0 * pi_t),
            torch.cos(2.0 * pi_t),
            t * t,
            torch.sqrt(t.clamp_min(1e-4)),
        ),
        dim=-1,
    )


def sample_orthogonal_team_bot_noise(
    E_base: int,
    G: int,
    N: int,
    d_e: int,
    p_team: torch.Tensor,
    K: int,
    *,
    team_corr: float = 0.45,
    sigma: float = 0.55,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw orthogonalized/antithetic team+bot base noise [E_base * G, N, d_e] across G rollout branches.

    Team-shared component probes Strategy (coherent team-wide allocation across branches);
    per-bot component probes Tactic (individual bot deviations within the team).
    Gram-Schmidt orthogonalization across g=0..G-1 guarantees distinct plural branches from the same savestate.
    """
    dev = torch.device(device)
    B = int(E_base)
    G = max(1, int(G))

    def _ortho_group(raw: torch.Tensor) -> torch.Tensor:
        # raw: [B, G, S, d_e] -> orthogonalize across G along last dim (keeping variance = 1)
        if G <= 1:
            return raw
        if G == 2:
            # Exact antithetic pair!
            return torch.stack((raw[:, 0], -raw[:, 0]), dim=1)
        # Modified Gram-Schmidt across G vectors in R^{d_e} (for G <= d_e), preserving unit per-coord variance
        x = raw.clone()
        out_list = []
        for g in range(G):
            v = x[:, g]  # [B, S, d_e]
            for prev in out_list:
                dot = (v * prev).sum(dim=-1, keepdim=True)
                denom = prev.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
                v = v - (dot / denom) * prev
            norm = v.square().mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
            out_list.append(v / norm)
        return torch.stack(out_list, dim=1)

    z_team = torch.randn((B, G, max(1, K), d_e), generator=generator, device=dev, dtype=torch.float32)
    z_bot = torch.randn((B, G, N, d_e), generator=generator, device=dev, dtype=torch.float32)
    z_team = _ortho_group(z_team)
    z_bot = _ortho_group(z_bot)

    # Gather each bot's team noise using p_team [B*G, N] (1-based team IDs)
    t0 = (p_team.reshape(B, G, N).to(device=dev, dtype=torch.int64) - 1).clamp(0, max(1, K) - 1)
    zt_bot = torch.gather(z_team, 2, t0.unsqueeze(-1).expand(B, G, N, d_e))

    a_t = math.sqrt(max(0.0, min(1.0, float(team_corr))))
    a_b = math.sqrt(max(0.0, 1.0 - float(team_corr)))
    eps = float(sigma) * (a_t * zt_bot + a_b * z_bot)
    return eps.reshape(B * G, N, d_e)


def prenorm_graph_inputs(
    obs: torch.Tensor,
    nav: torch.Tensor,
    carts: torch.Tensor,
    teams: torch.Tensor,
    acc: torch.Tensor,
    geo: dict[str, torch.Tensor] | None,
    nc: NormConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graph-native afferent prenormalization preserving nav_id, goal_kind, and steering accumulator feedback.

    Returns:
      obs_pre [E, N, 96], nav_pre [E, W_nav, 20], cart_pre [E, J, 22], team_pre [E, K, 9],
      entity_teams [E, M_ent] int64, team_ids [E, K] int64, bot_nav_id [E, N] int64.
    """
    dev = obs.device
    E, N, _ = obs.shape
    W_nav = nav.shape[1]
    J = carts.shape[1]
    K = teams.shape[1]
    mp_eps = max(float(nc.mp_norm_eps), 1e-4)

    has_geo = geo is not None and "map_center" in geo and "map_radius" in geo and float(geo["map_radius"].item()) > 0.0
    if has_geo:
        center = geo["map_center"].to(device=dev, dtype=torch.float32).view(1, 1, 3)
        rad = max(float(geo["map_radius"].item()), float(nc.map_radius_min))
    else:
        center = torch.zeros((1, 1, 3), device=dev, dtype=torch.float32)
        rad = float(nc.pos_fallback_scale)

    # 1. Teams + CGT features [E, K, 9]
    tf = teams.float()
    t_id = tf[:, :, 0]
    t_score = tf[:, :, 1]
    t_held = tf[:, :, 2]
    t_rate = tf[:, :, 3]
    t_lim = tf[:, :, 4].clamp_min(1e-6)
    t_rem = (t_lim - t_score).clamp_min(1e-6)
    t_rho = torch.where(t_held > 0.0, t_rate * t_held / t_rem, torch.zeros_like(t_held))
    if K > 1:
        eye_k = torch.eye(K, device=dev, dtype=torch.bool).unsqueeze(0)
        best_other = t_rho.unsqueeze(1).expand(E, K, K).masked_fill(eye_k, -1.0).max(dim=-1).values.clamp_min(0.0)
    else:
        best_other = torch.zeros_like(t_rho)
    t_ex0 = torch.tanh(t_rho.clamp(-20.0, 20.0))
    t_ex1 = (t_rho - best_other) / (t_rho + best_other + 1e-6)

    team_raw = torch.zeros((E, K, TEAM_PRE_DIM), device=dev, dtype=torch.float32)
    team_raw[:, :, 0] = t_id
    team_raw[:, :, 1] = t_score / t_lim.clamp_min(1.0)
    team_raw[:, :, 2] = t_held
    team_raw[:, :, 3] = t_rate
    team_raw[:, :, 4] = 1.0
    team_raw[:, :, 5] = 0.0
    team_raw[:, :, 6] = tf[:, :, 6]
    team_raw[:, :, 7] = t_ex0
    team_raw[:, :, 8] = t_ex1
    team_pre = _mp_norm(torch.asinh(team_raw), eps=mp_eps)

    # 2. Carts + CGT features [E, J, 22]
    cf = carts.float()
    plen = cf[:, :, 2].clamp_min(1.0)
    pfrac = (cf[:, :, 1] / plen).clamp(0.0, 1.0)
    cp_cnt = cf[:, :, 17].clamp_min(1.0)
    cp_dep = cf[:, :, 16].clamp_min(0.0)
    rem_heap = torch.round(cp_cnt - cp_dep).clamp_min(0.0).to(torch.int64)
    c_nim = rem_heap % 3
    nim_tot = torch.zeros((E,), device=dev, dtype=torch.int64)
    for j_i in range(J):
        nim_tot = nim_tot ^ c_nim[:, j_i]
    seg = pfrac * cp_cnt
    dist_deny = (seg - torch.floor(seg - 1e-6)).clamp_min(0.05)
    dist_push = (torch.ceil(seg + 1e-6) - seg).clamp_min(0.05)

    oid = cf[:, :, 3]  # [E, J]
    match_t = (oid.unsqueeze(-1) == t_id.unsqueeze(1))  # [E, J, K]
    any_owner = match_t.any(dim=-1)
    oidx = match_t.float().argmax(dim=-1)
    oh = torch.gather(t_held, 1, oidx)
    orho = torch.gather(t_rho, 1, oidx)
    obo = torch.gather(best_other, 1, oidx)
    drd = torch.where(any_owner & (cp_dep > 0.0) & (oh > 0.0), orho / oh.clamp_min(1.0), torch.zeros_like(orho))
    denom_d = torch.where(any_owner, orho + obo + 1e-6, torch.full_like(orho, 1e-6))
    c_ex0 = torch.tanh((drd / denom_d / dist_deny).clamp(-20.0, 20.0))

    max_rho = t_rho.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
    max_rate = t_rate.max(dim=-1, keepdim=True).values
    min_rem = t_rem.min(dim=-1, keepdim=True).values.clamp_min(1e-6)
    drp = torch.where(cp_dep < cp_cnt, max_rate / min_rem, torch.zeros_like(cp_dep))
    c_ex1 = torch.tanh(((drp / (max_rho + drp + 1e-6)) / dist_push).clamp(-20.0, 20.0))
    tn = c_nim ^ nim_tot.unsqueeze(1)
    c_ex2 = ((nim_tot.unsqueeze(1) != 0) & (tn < c_nim) & ((c_nim - tn) <= rem_heap)).float()
    c_ex3 = c_nim.float() / 3.0

    cart_raw = torch.zeros((E, J, CART_PRE_DIM), device=dev, dtype=torch.float32)
    cart_raw[:, :, :18] = cf
    cart_raw[:, :, 1] = cf[:, :, 1] / plen
    cart_raw[:, :, 2] = torch.log1p(plen / float(nc.cart_len_scale))
    cart_raw[:, :, 4] = cf[:, :, 4] / float(nc.cart_speed_scale)
    cart_raw[:, :, 9:12] = (cf[:, :, 9:12] - center) / rad if has_geo else (cf[:, :, 9:12] / float(nc.pos_fallback_scale))
    cart_raw[:, :, 18] = c_ex0
    cart_raw[:, :, 19] = c_ex1
    cart_raw[:, :, 20] = c_ex2
    cart_raw[:, :, 21] = c_ex3
    cart_pre = _mp_norm(torch.asinh(cart_raw), eps=mp_eps)

    # 3. Directed nav-graph nodes [E, W_nav, 20] including committed steering mass from acc!
    nf = nav.float()
    af = acc.float()
    r_nav = af[:, :, :W_nav]  # [E, N, W_nav]
    r_team = af[:, :, W_nav : W_nav + K]  # [E, N, K]
    r_self = af[:, :, W_nav + K : W_nav + K + 4]  # [E, N, 4]

    nav_raw = torch.zeros((E, W_nav, NAV_PRE_DIM), device=dev, dtype=torch.float32)
    nav_raw[:, :, :16] = nf
    nav_raw[:, :, 0:3] = (nf[:, :, 0:3] - center) / rad if has_geo else (nf[:, :, 0:3] / float(nc.pos_fallback_scale))
    inv_n = 1.0 / float(max(1, N))
    nav_raw[:, :, 16] = r_nav.clamp_min(0.0).sum(dim=1) * inv_n
    nav_raw[:, :, 17] = r_nav.sum(dim=1) * inv_n
    nav_raw[:, :, 18] = r_nav.max(dim=1).values
    nav_raw[:, :, 19] = r_nav.min(dim=1).values
    nav_pre = _mp_norm(torch.asinh(nav_raw), eps=mp_eps)

    # 4. Players [E, N, 96] preserving nav_id (col 22), goal_kind (col 23) + acc summary & one-hot goal_kind!
    of = obs.float()
    bot_nav_id = torch.round(of[:, :, 22]).to(torch.int64).clamp(0, max(1, W_nav) - 1)
    goal_kind = torch.round(of[:, :, 23]).to(torch.int64).clamp(0, 3)

    obs_raw = torch.zeros((E, N, OBS_PRE_DIM), device=dev, dtype=torch.float32)
    obs_raw[:, :, :OBS_WIDTH] = of
    obs_raw[:, :, 2] = (of[:, :, 2] - float(nc.health_offset)) / float(nc.health_scale)
    obs_raw[:, :, 3] = (of[:, :, 3] - float(nc.armor_offset)) / float(nc.armor_scale)
    obs_raw[:, :, 4:10] = of[:, :, 4:10] / float(nc.ammo_scale)
    obs_raw[:, :, 10:13] = (of[:, :, 10:13] - center) / rad if has_geo else (of[:, :, 10:13] / float(nc.pos_fallback_scale))
    obs_raw[:, :, 13:16] = of[:, :, 13:16] / float(nc.vel_scale)
    obs_raw[:, :, 19] = (of[:, :, 19] - of[:, :, 21]).clamp(0.0, float(nc.strength_max_s)) / float(nc.strength_max_s)
    obs_raw[:, :, 20] = 0.0
    obs_raw[:, :, 21] = 0.0
    obs_raw[:, :, 22] = 0.0  # bot_nav_id is used strictly for equivariant graph indexing, never as an ordinal scalar
    obs_raw[:, :, 23] = goal_kind.float() / 3.0
    for zc in (39, 40, 41, 54, 62, 79, 80):
        obs_raw[:, :, zc] = 0.0

    obs_raw[:, :, 84:88] = r_self
    obs_raw[:, :, 88] = r_nav.square().mean(dim=-1).clamp_min(1e-8).sqrt()
    obs_raw[:, :, 89] = r_team.square().mean(dim=-1).clamp_min(1e-8).sqrt()
    obs_raw[:, :, 90] = r_nav.max(dim=-1).values
    obs_raw[:, :, 91] = r_nav.min(dim=-1).values
    obs_raw[:, :, 92:96] = F.one_hot(goal_kind, num_classes=4).float()
    obs_pre = _mp_norm(torch.asinh(obs_raw), eps=mp_eps)

    # Entity team IDs for M_ent = N + W_nav + J + K
    bot_t = torch.round(of[:, :, 1]).to(torch.int64)
    nav_t = torch.zeros((E, W_nav), device=dev, dtype=torch.int64)
    cart_t = torch.round(cf[:, :, 3]).to(torch.int64)
    team_ids = torch.round(t_id).to(torch.int64)
    entity_teams = torch.cat((bot_t, nav_t, cart_t, team_ids), dim=1)

    return obs_pre, nav_pre, cart_pre, team_pre, entity_teams, team_ids, bot_nav_id


class GraphDPPBlock(nn.Module):
    """Single entity-mixing block:
      1. Directed Nav-Graph geodesic diffusion (nav_kern @ V_nav) + bot local nav-node readout
      2. Global Gram + Team Rival Gram (O(M * r_gram) linear-time global + team interaction)
      3. Restored Exact FP32 Per-Team DPP Diversity Kernel B(I + B^T B)^{-1} B^T V on bot tokens
      4. Wide MP-SwiGLU FFN
    """

    def __init__(self, cfg: GraphFlowConfig, nc: NormConfig):
        super().__init__()
        self.d = cfg.d
        self.h = cfg.h
        self.r_gram = cfg.r_gram
        self.r_team = cfg.r_team
        self.r_dpp = cfg.r_dpp
        self.nc = nc

        # 1. Directed nav-graph mixing
        self.nav_val_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.nav_out_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.blend_nav = nn.Parameter(torch.tensor(float(cfg.nav_blend_init_logit)))

        # 2. Global + Team Rival Gram
        self.val_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.global_metric = nn.Linear(cfg.d, cfg.r_gram, bias=False)
        self.pool_score = nn.Linear(cfg.d, 1, bias=False)
        self.team_metric = nn.Linear(cfg.d, cfg.r_team, bias=False)
        self.rival_metric = nn.Linear(cfg.d, cfg.r_team, bias=False)
        self.team_val_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.team_mutate_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.team_bcast_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.gram_out = nn.Linear(cfg.d, cfg.d, bias=False)
        self.blend_pool = nn.Parameter(torch.tensor(-0.5))
        self.blend_team_mut = nn.Parameter(torch.tensor(-0.8))
        self.blend_gram = nn.Parameter(torch.tensor(-0.5))

        # 3. Exact FP32 Per-Team DPP Diversity Kernel
        self.dpp_b_proj = nn.Linear(cfg.d, cfg.r_dpp, bias=False)
        self.dpp_v_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.dpp_out_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.blend_dpp = nn.Parameter(torch.tensor(float(cfg.dpp_blend_init_logit)))

        # 4. Wide MP-SwiGLU FFN
        self.ffn_gate = nn.Linear(cfg.d, cfg.h, bias=False)
        self.ffn_up = nn.Linear(cfg.d, cfg.h, bias=False)
        self.ffn_down = nn.Linear(cfg.h, cfg.d, bias=False)
        self.blend_ffn = nn.Parameter(torch.tensor(-0.5))

    def _dpp_team_mix(
        self,
        h_bot: torch.Tensor,
        bot_teams: torch.Tensor,
        team_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Exact FP32 per-team DPP kernel Y_k = B_k (I + B_k^T B_k)^{-1} B_k^T V_k."""
        E, N, D = h_bot.shape
        K = team_ids.shape[1]
        r = self.r_dpp
        eps_n = self.nc.mp_norm_eps
        eps_u = self.nc.unit_norm_eps

        B = _mp_norm(_unit_linear(h_bot, self.dpp_b_proj.weight, eps_u), eps_n).float()  # [E, N, r]
        V = _mp_norm(_unit_linear(h_bot, self.dpp_v_proj.weight, eps_u), eps_n).float()  # [E, N, D]
        # Scale B by 1/sqrt(r) so B^T B has O(n_k) trace and I + B^T B is super well-conditioned
        Bs = B * (1.0 / math.sqrt(float(r)))

        # Vectorized across all (E, K) via team indicator mask [E, K, N]
        m = (bot_teams.unsqueeze(1) == team_ids.unsqueeze(2)).float()  # [E, K, N]
        Bm = Bs.unsqueeze(1) * m.unsqueeze(-1)  # [E, K, N, r]
        Vm = V.unsqueeze(1) * m.unsqueeze(-1)   # [E, K, N, D]

        G = torch.eye(r, device=h_bot.device, dtype=torch.float32).view(1, 1, r, r) + torch.einsum(
            "eknr,ekns->ekrs", Bm, Bm
        )  # [E, K, r, r] strictly SPD!
        rhs = torch.einsum("eknr,eknd->ekrd", Bm, Vm)  # [E, K, r, D]
        L = torch.linalg.cholesky(G)
        sol = torch.cholesky_solve(rhs, L)  # [E, K, r, D]
        Y_ek = torch.einsum("eknr,ekrd->eknd", Bm, sol)  # [E, K, N, D]
        Y = Y_ek.sum(dim=1).to(h_bot.dtype)  # [E, N, D] (each bot belongs to exactly 1 team)
        return _unit_linear(Y, self.dpp_out_proj.weight, eps_u)

    def forward(
        self,
        x: torch.Tensor,
        *,
        N: int,
        W_nav: int,
        J: int,
        K: int,
        entity_teams: torch.Tensor,
        team_ids: torch.Tensor,
        bot_nav_id: torch.Tensor,
        nav_kern: torch.Tensor,
    ) -> torch.Tensor:
        E, M_ent, D = x.shape
        eps_n = self.nc.mp_norm_eps
        eps_l = self.nc.mp_lerp_eps
        eps_u = self.nc.unit_norm_eps

        xn = _mp_norm(x, eps_n)

        # 1. Directed Nav-Graph Geodesic Diffusion + Bot Local Graph Readout
        x_bot = xn[:, :N, :]
        x_nav = xn[:, N : N + W_nav, :]
        x_rest = xn[:, N + W_nav :, :]

        v_nav = _mp_norm(_unit_linear(x_nav, self.nav_val_proj.weight, eps_u), eps_n)  # [E, W_nav, D]
        kern = nav_kern.to(device=x.device, dtype=torch.float32)
        if kern.ndim == 2:
            nav_diff = torch.einsum("vw,ewd->evd", kern, v_nav.float()).to(x.dtype)
        else:
            nav_diff = torch.einsum("evw,ewd->evd", kern, v_nav.float()).to(x.dtype)
        nav_msg = _unit_linear(nav_diff, self.nav_out_proj.weight, eps_u)
        bot_loc_msg = torch.gather(nav_msg, 1, bot_nav_id.unsqueeze(-1).expand(E, N, D))

        bn = torch.sigmoid(self.blend_nav)
        x_bot_g = _mp_lerp(x_bot, _mp_norm(bot_loc_msg, eps_n), bn, eps_l)
        x_nav_g = _mp_lerp(x_nav, _mp_norm(nav_msg, eps_n), bn, eps_l)
        xg = torch.cat((x_bot_g, x_nav_g, x_rest), dim=1)

        # 2. Global Gram + Team Rival Gram (1-to-1 match with compiled MG_OP_CGT_GRAM_*!)
        v = _mp_norm(_unit_linear(xg, self.val_proj.weight, eps_u), eps_n)
        r = _mp_norm(_unit_linear(xg, self.global_metric.weight, eps_u), eps_n)
        gram_s = torch.einsum("emr,emd->erd", r.float(), v.float()) / math.sqrt(float(max(1, M_ent)))
        global_raw = torch.einsum("emr,erd->emd", r.float(), gram_s).to(x.dtype)

        # Sigmoid team attention pooling over all member entities (matching mg_cgt_team_pool)
        sig = torch.sigmoid(_unit_linear(xg, self.pool_score.weight, eps_u).squeeze(-1).float())  # [E, M_ent]
        m_tk = (entity_teams.unsqueeze(1) == team_ids.unsqueeze(2)).float()  # [E, K, M_ent]
        w_m = m_tk * sig.unsqueeze(1)
        den = w_m.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        pooled = (torch.einsum("ekm,emd->ekd", w_m, v.float()) / den).to(x.dtype)
        t_tok = xg[:, N + W_nav + J :, :]
        t_in = _mp_norm(_mp_lerp(t_tok, _mp_norm(pooled, eps_n), torch.sigmoid(self.blend_pool), eps_l), eps_n)

        qt = _mp_norm(_unit_linear(t_in, self.team_metric.weight, eps_u), eps_n).float()
        kt = _mp_norm(_unit_linear(t_in, self.rival_metric.weight, eps_u), eps_n).float()
        vt = _mp_norm(_unit_linear(t_in, self.team_val_proj.weight, eps_u), eps_n).float()
        gram_t = torch.einsum("ekr,ejr->ekj", qt, kt) / math.sqrt(float(max(1, K)))
        t_mixed = _mp_norm(torch.einsum("ekj,ejd->ekd", gram_t, vt).to(x.dtype), eps_n)
        t_mut_raw = _unit_linear(t_mixed, self.team_mutate_proj.weight, eps_u)
        t_mut = _mp_norm(_mp_lerp(t_in, _mp_norm(t_mut_raw, eps_n), torch.sigmoid(self.blend_team_mut), eps_l), eps_n)
        t_bcast = _mp_norm(_unit_linear(t_mut, self.team_bcast_proj.weight, eps_u), eps_n)

        # Broadcast team representation to member entities & global mean to neutral entities (nav-nodes + neutral carts!)
        t_glob = _mp_norm(t_bcast.float().mean(dim=1, keepdim=True).to(x.dtype), eps_n)
        has_team = (m_tk.sum(dim=1, keepdim=False) > 0.5).unsqueeze(-1)  # [E, M_ent, 1]
        tb_mem = torch.einsum("ekm,ekd->emd", m_tk, t_bcast.float()).to(x.dtype)
        team_vals = _mp_norm(torch.where(has_team, tb_mem, t_glob.expand(E, M_ent, D)), eps_n)
        gram_in = _mp_norm(_mp_norm(global_raw, eps_n) + team_vals, eps_n)
        gram_out = _unit_linear(gram_in, self.gram_out.weight, eps_u)
        x_mid = _mp_norm(_mp_lerp(xg, _mp_norm(gram_out, eps_n), torch.sigmoid(self.blend_gram), eps_l), eps_n)

        # 3. Exact FP32 Per-Team DPP Diversity Kernel on the N bot tokens!
        dpp_raw = self._dpp_team_mix(x_mid[:, :N, :], entity_teams[:, :N], team_ids)
        x_bot_dpp = _mp_lerp(x_mid[:, :N, :], _mp_norm(dpp_raw, eps_n), torch.sigmoid(self.blend_dpp), eps_l)
        x_mid2 = torch.cat((x_bot_dpp, x_mid[:, N:, :]), dim=1)

        # 4. Wide MP-SwiGLU FFN
        gate = _unit_linear(x_mid2, self.ffn_gate.weight, eps_u)
        up = _unit_linear(x_mid2, self.ffn_up.weight, eps_u)
        hid = _mp_silu(gate, self.nc.mp_silu_scale) * up
        ffn_raw = _unit_linear(hid, self.ffn_down.weight, eps_u)
        return _mp_norm(_mp_lerp(x_mid2, _mp_norm(ffn_raw, eps_n), torch.sigmoid(self.blend_ffn), eps_l), eps_n)


class GraphFlowPolicy(nn.Module):
    """Graph-Native + Exact DPP + 1-NFE Mealy-Jet Flow Policy over continuous steering velocities."""

    def __init__(
        self,
        cfg: GraphFlowConfig = GraphFlowConfig(),
        norm_cfg: NormConfig = NormConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.norm_cfg = norm_cfg
        self.d = cfg.d
        self.h = cfg.h
        self.depth = cfg.depth
        self.recurrent = cfg.recurrent
        self.d_e = cfg.d_e
        self.c_dim = cfg.c_flow_dim

        # Afferent encoders
        self.enc_obs = nn.Linear(OBS_PRE_DIM, cfg.d, bias=False)
        self.enc_nav = nn.Linear(NAV_PRE_DIM, cfg.d, bias=False)
        self.enc_cart = nn.Linear(CART_PRE_DIM, cfg.d, bias=False)
        self.enc_team = nn.Linear(TEAM_PRE_DIM, cfg.d, bias=False)
        self.type_embed = nn.Parameter(torch.randn(4, cfg.d) * 0.1)
        self.type_blend = nn.Parameter(torch.tensor(float(cfg.type_blend_init_logit)))
        self.aff_blend = nn.Parameter(torch.tensor(float(cfg.aff_blend_init_logit)))

        # Backbone mixing blocks
        n_blk = 1 if cfg.recurrent else cfg.depth
        self.blocks = nn.ModuleList([GraphDPPBlock(cfg, norm_cfg) for _ in range(n_blk)])
        if cfg.recurrent:
            init_logit = math.log(cfg.step_blend_init / max(1.0 - cfg.step_blend_init, 1e-4))
            self.step_blends = nn.Parameter(torch.full((cfg.depth,), init_logit))
        else:
            self.register_parameter("step_blends", None)

        # Shared-IR linear Winnie/Lou critic + dynamics probes
        self.value_proj = nn.Linear(cfg.d, 2, bias=False)
        self.value_gain = nn.Parameter(torch.tensor(float(cfg.value_gain_init)))
        self.value_bias = nn.Parameter(torch.zeros(2))
        self.dyn_drift_proj = nn.Linear(cfg.d, 1, bias=False)
        self.dyn_action_proj = nn.Linear(cfg.d_e, 1, bias=False)
        self.dyn_drift_gain = nn.Parameter(torch.tensor(float(cfg.dyn_drift_gain_init)))
        self.dyn_action_gain = nn.Parameter(torch.tensor(float(cfg.dyn_action_gain_init)))

        # Equivariant target basis B_H in R^{(W_nav + K + 4) x d_e}
        self.basis_nav_proj = nn.Linear(cfg.d, cfg.d_e, bias=False)
        self.basis_team_proj = nn.Linear(cfg.d, cfg.d_e, bias=False)
        self.basis_self = nn.Parameter(torch.randn(4, cfg.d_e) * 0.2)

        # Single-NFE Closed-Loop SRAM Flow Efferent (2-layer SwiGLU vector field integrated by Heun/RK2!)
        self.flow_cond_bot = nn.Linear(cfg.d, cfg.c_flow_dim, bias=False)
        self.flow_cond_team = nn.Linear(cfg.d, cfg.c_flow_dim, bias=False)
        fj_in = cfg.d_e + 8 + cfg.c_flow_dim
        self.fj_w1 = nn.Linear(fj_in, 2 * cfg.flow_hid, bias=True)
        self.fj_w2 = nn.Linear(cfg.flow_hid, cfg.d_e, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1.0 / math.sqrt(max(1, m.weight.shape[1])))
                    if m.bias is not None:
                        m.bias.zero_()
            # Near-zero init on flow head output => smooth zero-mean perturbation around stock HavocBot
            self.fj_w2.weight.mul_(0.02)
            self.fj_w2.bias.zero_()
            project_graph_flow_unit_norm_(self, eps=self.norm_cfg.unit_norm_eps)

    def encode_backbone(
        self,
        obs: torch.Tensor,
        nav: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        acc: torch.Tensor,
        nav_kern: torch.Tensor,
        geo: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single backbone NFE -> shared wide IR H + equivariant basis B_H + flow conditioning c_flow + critics."""
        E, N, _ = obs.shape
        W_nav = nav.shape[1]
        J = carts.shape[1]
        K = teams.shape[1]
        nc = self.norm_cfg
        eps_n, eps_l, eps_u = nc.mp_norm_eps, nc.mp_lerp_eps, nc.unit_norm_eps

        obs_pre, nav_pre, cart_pre, team_pre, entity_teams, team_ids, bot_nav_id = prenorm_graph_inputs(
            obs, nav, carts, teams, acc, geo, nc
        )
        dt = self.enc_obs.weight.dtype
        h_obs = _mp_norm(_unit_linear(obs_pre.to(dt), self.enc_obs.weight, eps_u), eps_n)
        h_nav = _mp_norm(_unit_linear(nav_pre.to(dt), self.enc_nav.weight, eps_u), eps_n)
        h_cart = _mp_norm(_unit_linear(cart_pre.to(dt), self.enc_cart.weight, eps_u), eps_n)
        h_team = _mp_norm(_unit_linear(team_pre.to(dt), self.enc_team.weight, eps_u), eps_n)

        te = _mp_norm(self.type_embed.to(dt), eps_n)
        tb = torch.sigmoid(self.type_blend)
        h_obs = _mp_norm(_mp_lerp(h_obs, te[0].view(1, 1, -1), tb, eps_l), eps_n)
        h_nav = _mp_norm(_mp_lerp(h_nav, te[1].view(1, 1, -1), tb, eps_l), eps_n)
        h_cart = _mp_norm(_mp_lerp(h_cart, te[2].view(1, 1, -1), tb, eps_l), eps_n)
        h_team = _mp_norm(_mp_lerp(h_team, te[3].view(1, 1, -1), tb, eps_l), eps_n)

        # Equivariant afferent fold: bot's current nav cell + current integrated steering (r_nav, r_team)
        r_nav = acc[:, :, :W_nav].float()
        r_team = acc[:, :, W_nav : W_nav + K].float()
        nav_ctx = torch.gather(h_nav, 1, bot_nav_id.unsqueeze(-1).expand(E, N, self.d)).float()
        r_nav_ctx = torch.einsum("env,evd->end", r_nav, h_nav.float()) / math.sqrt(float(max(1, W_nav)))
        r_team_ctx = torch.einsum("enk,ekd->end", r_team, h_team.float()) / math.sqrt(float(max(1, K)))
        aff_msg = _mp_norm((nav_ctx + r_nav_ctx + r_team_ctx).to(dt), eps_n)
        h_obs = _mp_lerp(h_obs, aff_msg, torch.sigmoid(self.aff_blend), eps_l)

        x = torch.cat((h_obs, h_nav, h_cart, h_team), dim=1)
        for s in range(self.depth):
            blk = self.blocks[0] if self.recurrent else self.blocks[s]
            x_new = blk(
                x,
                N=N,
                W_nav=W_nav,
                J=J,
                K=K,
                entity_teams=entity_teams,
                team_ids=team_ids,
                bot_nav_id=bot_nav_id,
                nav_kern=nav_kern,
            )
            if self.recurrent:
                x = _mp_norm(_mp_lerp(x, x_new, torch.sigmoid(self.step_blends[s]), eps_l), eps_n)
            else:
                x = x_new

        H = _mp_norm(x, eps_n)
        H_bot = H[:, :N, :]
        H_nav = H[:, N : N + W_nav, :]
        H_team = H[:, N + W_nav + J :, :]

        # Linear Winnie/Lou value probes + drift dynamics on shared IR H_bot
        v_gain = self.value_gain.float().clamp(*self.cfg.value_gain_bounds)
        val_raw = _unit_linear(H_bot, self.value_proj.weight, eps_u).float()
        values = v_gain * val_raw + self.value_bias.float().view(1, 1, 2)
        d_gain = self.dyn_drift_gain.float().clamp(*self.cfg.dyn_gain_bounds)
        dyn_drift = d_gain * _unit_linear(H_bot, self.dyn_drift_proj.weight, eps_u).squeeze(-1).float()

        # Equivariant target basis B_H [E, A_dim, d_e], each row unit L2 norm!
        B_nav = _mp_unit_vec(_unit_linear(H_nav, self.basis_nav_proj.weight, eps_u), nc.mp_unit_vec_eps).float()
        B_team = _mp_unit_vec(_unit_linear(H_team, self.basis_team_proj.weight, eps_u), nc.mp_unit_vec_eps).float()
        B_self = _mp_unit_vec(self.basis_self, nc.mp_unit_vec_eps).float().unsqueeze(0).expand(E, 4, self.d_e)
        B_H = torch.cat((B_nav, B_team, B_self), dim=1)  # [E, W_nav + K + 4, d_e]

        # Per-bot flow conditioning c_flow [E, N, c_dim] (bot + its own team IR)
        bot_t0 = (entity_teams[:, :N] - 1).clamp(0, max(1, K) - 1)
        H_my_team = torch.gather(H_team, 1, bot_t0.unsqueeze(-1).expand(E, N, self.d))
        c_bot = _unit_linear(H_bot, self.flow_cond_bot.weight, eps_u)
        c_tm = _unit_linear(H_my_team, self.flow_cond_team.weight, eps_u)
        c_flow = _mp_norm(c_bot + c_tm, eps_n).float()

        return {
            "H_bot": H_bot,
            "H_nav": H_nav,
            "H_team": H_team,
            "B_H": B_H,
            "c_flow": c_flow,
            "value_winnie": values[:, :, 0],
            "value_lou": values[:, :, 1],
            "dyn_drift": dyn_drift,
            "bot_t0": bot_t0,
        }

    def flow_vel_eval(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        c_flow: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate closed-loop SRAM velocity field v_psi(z, tau; c_flow) = -dz/dtau (pointing tau: 1 -> 0).

        Feeds radially bounded z_fed = z * rsqrt(1 + mean(z^2)/4.0) into the SwiGLU MLP and bounds
        output velocity v = v_max * tanh(v_raw / v_max) so Heun ODE integration is globally Lipschitz.
        """
        zf = z.float()
        z_fed = zf * torch.rsqrt(1.0 + zf.square().mean(dim=-1, keepdim=True) * 0.25)
        te = _tau_embed(tau)
        inp = torch.cat((z_fed, te, c_flow.float()), dim=-1)
        gu = F.linear(inp, self.fj_w1.weight.float(), self.fj_w1.bias.float())
        g, u = gu.chunk(2, dim=-1)
        hid = (F.silu(g) / float(self.norm_cfg.mp_silu_scale)) * u
        v_raw = F.linear(hid, self.fj_w2.weight.float(), self.fj_w2.bias.float())
        v_max = float(self.cfg.v_max)
        return v_max * torch.tanh(v_raw / v_max)

    def lift_to_action(
        self,
        z_lat: torch.Tensor,
        B_H: torch.Tensor,
        bot_t0: torch.Tensor,
        W_nav: int,
        K: int,
    ) -> torch.Tensor:
        """Project latent velocity z_lat [..., E, N, d_e] through B_H [E, A_dim, d_e] and mask own-team slot."""
        u = torch.einsum("...end,ead->...ena", z_lat.float(), B_H.float())
        own_idx = (W_nav + bot_t0).unsqueeze(-1)
        if u.ndim == 4:
            own_idx = own_idx.unsqueeze(0).expand(u.shape[0], -1, -1, 1)
        return u.scatter(-1, own_idx, 0.0)

    @torch.no_grad()
    def sample_velocity(
        self,
        obs: torch.Tensor,
        nav: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        acc: torch.Tensor,
        nav_kern: torch.Tensor,
        geo: dict[str, torch.Tensor] | None = None,
        *,
        eps: torch.Tensor | None = None,
        heun_steps: int | None = None,
        output_scale: float = 1.0,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Single backbone NFE + S Heun/RK2 SRAM integration steps (exact O(h^2) w/ convective derivative!)."""
        E, N, _ = obs.shape
        W_nav = nav.shape[1]
        K = teams.shape[1]
        enc = self.encode_backbone(obs, nav, carts, teams, acc, nav_kern, geo=geo)
        B_H = enc["B_H"]
        c_flow = enc["c_flow"]
        bot_t0 = enc["bot_t0"]

        if float(output_scale) == 0.0:
            z = torch.zeros((E, N, self.d_e), device=obs.device, dtype=torch.float32)
            u_vel = torch.zeros((E, N, W_nav + K + 4), device=obs.device, dtype=torch.float32)
            return {
                "u_vel": u_vel, "u_lat": z,
                "value_winnie": enc["value_winnie"], "value_lou": enc["value_lou"],
                "B_H": B_H, "c_flow": c_flow,
            }

        if eps is None:
            if deterministic:
                z = torch.zeros((E, N, self.d_e), device=obs.device, dtype=torch.float32)
            else:
                z = torch.randn((E, N, self.d_e), device=obs.device, dtype=torch.float32) * float(self.cfg.sigma_base)
        else:
            z = eps.to(device=obs.device, dtype=torch.float32)

        S = int(self.cfg.heun_steps if heun_steps is None else heun_steps)
        h = 1.0 / float(max(1, S))
        for step in range(S):
            tau_cur = 1.0 - step * h
            tau_nxt = max(0.0, tau_cur - h)
            t1 = torch.full((E, N), tau_cur, device=obs.device, dtype=torch.float32)
            t2 = torch.full((E, N), tau_nxt, device=obs.device, dtype=torch.float32)
            v1 = self.flow_vel_eval(z, t1, c_flow)
            z_pred = z + h * v1
            v2 = self.flow_vel_eval(z_pred, t2, c_flow)
            z = z + (0.5 * h) * (v1 + v2)

        u_vel = (float(output_scale) * self.lift_to_action(z, B_H, bot_t0, W_nav, K)).clamp(-self.cfg.u_clip, self.cfg.u_clip)
        return {
            "u_vel": u_vel,
            "u_lat": z,
            "value_winnie": enc["value_winnie"],
            "value_lou": enc["value_lou"],
            "B_H": B_H,
            "c_flow": c_flow,
        }

    def compute_boltzmann_flow_loss(
        self,
        obs: torch.Tensor,
        nav: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        acc: torch.Tensor,
        nav_kern: torch.Tensor,
        geo: dict[str, torch.Tensor] | None,
        *,
        u_target: torch.Tensor,      # [E, N, A_dim] float32 realized rollout velocity
        adv: torch.Tensor,           # [E, N] float32 centered group/GAE advantage
        gae_ret: torch.Tensor,       # [E, N] float32 critic return target
        obs_drho: torch.Tensor,      # [E, N] float32 dynamics target
        actor_mask: torch.Tensor,    # [E, N] bool/float32
        wm_in: torch.Tensor,         # [E, N] bool/float32 (1=Winnie, 0=Lou)
        value_coef: float = 0.5,
        dyn_coef: float = 0.10,
        group_size: int = 1,
        noise: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """1-NFE Pure Unbiased Advantage-Weighted Boltzmann Flow Matching (ESS-targeted alpha) + Winnie/Lou Critic."""
        E, N, _ = obs.shape
        W_nav = nav.shape[1]
        K = teams.shape[1]
        dev = obs.device
        cfg = self.cfg

        enc = self.encode_backbone(obs, nav, carts, teams, acc, nav_kern, geo=geo)
        B_H = enc["B_H"]        # [E, A_dim, d_e]
        c_flow = enc["c_flow"]  # [E, N, c_dim]
        bot_t0 = enc["bot_t0"]  # [E, N]

        am = actor_mask.to(device=dev, dtype=torch.float32)
        wm = wm_in.to(device=dev, dtype=torch.float32)
        lm = 1.0 - wm
        am_sum = am.sum().clamp_min(1.0)
        wm_sum = wm.sum().clamp_min(1.0)
        lm_sum = lm.sum().clamp_min(1.0)

        # Mask own-team slot of u_target
        u_t = u_target.to(device=dev, dtype=torch.float32)
        own_idx = (W_nav + bot_t0).unsqueeze(-1)
        u_t = u_t.scatter(-1, own_idx, 0.0)

        # Standardize advantage FIRST, THEN fold confining stock-HavocBot energy -lambda_0 * ||u||^2/A_dim!
        u_sq_mean = u_t.square().mean(dim=-1)  # [E, N]
        w_norm, ess_frac, mean_alpha = compute_ess_boltzmann_weights(
            adv.to(device=dev, dtype=torch.float32),
            am,
            u_sq_mean=u_sq_mean,
            zero_anchor_lambda=cfg.zero_anchor_lambda,
            ess_target=cfg.ess_target,
            w_min=cfg.flow_w_min,
            w_max=cfg.flow_w_max,
            beta_min=cfg.beta_bounds[0],
            beta_max=cfg.beta_bounds[1],
            group_size=group_size,
        )

        # Exact own-team-masked ridge projection (B^T M B + ridge_lambda I)^{-1} B^T M u_t via Sherman-Morrison!
        with torch.no_grad():
            B_det = B_H.detach()
            G_B = torch.einsum("ead,eaf->edf", B_det, B_det) + float(cfg.ridge_lambda) * torch.eye(self.d_e, device=dev).unsqueeze(0)
            rhs_B = torch.einsum("ead,ena->edn", B_det, u_t)
            L_B = torch.linalg.cholesky(G_B)
            x0 = torch.cholesky_solve(rhs_B, L_B).transpose(1, 2)  # [E, N, d_e]
            b_own = torch.gather(B_det, 1, own_idx.expand(E, N, self.d_e))  # [E, N, d_e]
            y_sm = torch.cholesky_solve(b_own.transpose(1, 2), L_B).transpose(1, 2)  # [E, N, d_e]
            bx = (b_own * x0).sum(dim=-1, keepdim=True)
            by = (b_own * y_sm).sum(dim=-1, keepdim=True)
            u_lat_targ = x0 + y_sm * (bx / (1.0 - by).clamp_min(1e-4))

        # Pure unbiased CFM across T=4 stratified jittered decreasing flow times in 1 backbone NFE!
        if taus is None:
            taus_t = ((torch.arange(4, device=dev, dtype=torch.float32) + torch.rand(4, device=dev, dtype=torch.float32)) * 0.25).clamp(0.02, 0.98)
        else:
            taus_t = taus.to(device=dev, dtype=torch.float32)
        T = taus_t.shape[0]
        if noise is None:
            eps = torch.randn((T, E, N, self.d_e), device=dev, dtype=torch.float32) * float(cfg.sigma_base)
        else:
            eps = noise.to(device=dev, dtype=torch.float32)

        t_b = taus_t.view(T, 1, 1, 1)
        z_t = (1.0 - t_b) * u_lat_targ.unsqueeze(0) + t_b * eps  # [T, E, N, d_e]
        tau_grid = taus_t.view(T, 1, 1).expand(T, E, N)
        c_exp = c_flow.unsqueeze(0).expand(T, E, N, self.c_dim)

        v_pred = self.flow_vel_eval(z_t, tau_grid, c_exp)  # [T, E, N, d_e]

        # Consistent action-metric CFM loss on v* = u_lat_targ - eps + separate basis reconstruction loss!
        v_diff_act = self.lift_to_action(v_pred - (u_lat_targ.unsqueeze(0) - eps), B_H, bot_t0, W_nav, K)
        vel_err = v_diff_act.square().mean(dim=-1)  # [T, E, N]
        u_rec_act = self.lift_to_action(u_lat_targ, B_H, bot_t0, W_nav, K)
        rec_err = (u_rec_act - u_t).square().mean(dim=-1)  # [E, N]
        loss_vel = ((vel_err.mean(dim=0) + float(cfg.basis_rec_weight) * rec_err) * w_norm).sum() / am_sum

        # Winnie / Lou Huber critic loss
        ret_f = gae_ret.to(device=dev, dtype=torch.float32)
        vw = enc["value_winnie"]
        vl = enc["value_lou"]
        lw = (F.huber_loss(vw, ret_f, reduction="none", delta=1.0) * wm).sum() / wm_sum
        ll = (F.huber_loss(vl, ret_f, reduction="none", delta=1.0) * lm).sum() / lm_sum
        loss_val = lw + ll

        # Dynamics auxiliary loss
        da_gain = self.dyn_action_gain.float().clamp(*cfg.dyn_gain_bounds)
        dyn_act = da_gain * _unit_linear(
            u_lat_targ.to(self.dyn_action_proj.weight.dtype), self.dyn_action_proj.weight, self.norm_cfg.unit_norm_eps
        ).squeeze(-1).float()
        dyn_pred = enc["dyn_drift"] + dyn_act
        loss_dyn = F.huber_loss(dyn_pred, obs_drho.to(device=dev, dtype=torch.float32), reduction="mean", delta=1.0)

        loss_total = loss_vel + float(value_coef) * loss_val + float(dyn_coef) * loss_dyn

        with torch.no_grad():
            u_norm = float(u_sq_mean.sqrt().mean().item())
            v_norm = float(v_pred.square().mean(dim=-1).sqrt().mean().item())
        stats = {
            "loss_total": float(loss_total.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_val": float(loss_val.item()),
            "loss_dyn": float(loss_dyn.item()),
            "ess_frac": ess_frac,
            "alpha": mean_alpha,
            "u_norm": u_norm,
            "v_norm": v_norm,
        }
        return loss_total, stats


@torch.no_grad()
def project_graph_flow_unit_norm_(model: GraphFlowPolicy, eps: float = 1e-6) -> None:
    """Project all row-normalized backbone weights onto the unit sphere (preserving Mealy-Jet MLP scale)."""
    skip_prefixes = ("fj_w1", "fj_w2")
    for name, p in model.named_parameters():
        if p.ndim == 2 and not name.startswith(skip_prefixes):
            w = F.normalize(p.data.float(), p=2, dim=-1, eps=max(float(eps), 1e-4))
            p.data.copy_(w.to(p.dtype))


def clone_graph_flow_model(model: GraphFlowPolicy) -> GraphFlowPolicy:
    return copy.deepcopy(model)
