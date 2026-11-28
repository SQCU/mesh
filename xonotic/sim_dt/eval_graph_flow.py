#!/usr/bin/env python3
"""Eval-Time Behavior & Multimodal Strategy Diversity Analyzer for GraphFlowPolicy across 7 DR Regimes.

Evaluates `default` (stock HavocBot u=0), `main_init`, `main_mid`, and `main_final` across:
  1. balanced_baseline       : equal mechanics (skill=4), baseline physics
  2. stacked_main_weak       : Team 1 stacked weak (-8, 0.52x DPS) vs ace enemies (+20, 2.20x DPS)
  3. mixed_enemies_main_weak : Team 1 weak (-8) vs mixed enemies (1 ace +20, 1 strong +14, 1 weak -8)
  4. stacked_main_strong     : Team 1 ace (+20) vs weak enemies (-8)
  5. within_team_carry_weak  : 25% ace carries (+22) + 75% weak (-8) on every team
  6. fast_carts_high_tempo   : fast cart physics perturbation
  7. volatile_contest_slow   : volatile contest/reverse + slow falloff physics perturbation

Reports:
  - Verified bit-for-bit output_scale=0 -> stock HavocBot collapse
  - Outcome competence (PW-WR, 1st%, Margin, K/D, Push, Pickups, Rockets/Quad, Alive)
  - Realized goal-kind distribution (%Cart, %Item, %Enemy, %NavNode) proving full action-space usage
  - Continuous steering field telemetry (||u||, r_cart/item/enemy/posture, weak-bot rival focus,
    Projected Winner vs Loser shifts, within-team Weak vs Ace divergence)
  - Multimodal Rollout-Group Diversity (G=8 branches from IDENTICAL savestates: pairwise velocity L2 &
    cosine separation, realized position/goal divergence, and efferent velocity PCA spectrum!)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import TS
from sim_dt.dataset_sampler import make_rules_for_profile, sample_roster_skills
from sim_dt.graph_flow_policy import (
    GraphFlowConfig,
    GraphFlowPolicy,
    sample_orthogonal_team_bot_noise,
)
from sim_dt.metal_graphflow import CompiledGraphFlowPolicy
from sim_dt.metal_runtime import MetalDeltaTimePayloadEngine
from sim_dt.world_graph import build_gas_world_graph
from solver.strat.train_config import CONFIGS_DIR, load_training_config


EVAL_REGIMES: list[dict[str, Any]] = [
    {"name": "balanced_baseline",       "roster": "balanced_fixed",          "physics": "baseline"},
    {"name": "stacked_main_weak",       "roster": "stacked_main_weak",       "physics": "baseline"},
    {"name": "mixed_enemies_main_weak", "roster": "mixed_enemies_main_weak", "physics": "baseline"},
    {"name": "stacked_main_strong",     "roster": "stacked_main_strong",     "physics": "baseline"},
    {"name": "within_team_carry_weak",  "roster": "within_team_carry_weak",  "physics": "baseline"},
    {"name": "fast_carts_high_tempo",   "roster": "balanced_fixed",          "physics": "fast"},
    {"name": "volatile_contest_slow",   "roster": "balanced_fixed",          "physics": "volatile"},
]


def _projected_leader_idx(teams_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = teams_np[:, :, TS["SCORE"]]
    held = teams_np[:, :, TS["CHECKPOINTS_HELD"]]
    rate = teams_np[:, :, TS["CHECKPOINT_RATE"]]
    lim = np.maximum(teams_np[:, :, TS["SCORE_LIMIT"]], 1e-6)
    rem = np.maximum(lim - sc, 0.0)
    ttw = np.where(held > 0.0, rem / np.maximum(held * rate, 1e-6), np.where(sc >= lim, 0.0, 1e12))
    min_ttw = np.min(ttw, axis=1, keepdims=True)
    is_min = (ttw <= min_ttw + 1e-6) & (min_ttw < 1e11)
    has_pw = (is_min.sum(axis=1) == 1)
    pw_idx = np.argmin(ttw, axis=1)
    return np.where(has_pw, pw_idx, np.argmax(sc, axis=1)), has_pw


@torch.no_grad()
def evaluate_arm_on_regime(
    arm_name: str,
    model: GraphFlowPolicy | None,
    world: Any,
    regime: dict[str, Any],
    *,
    cfg: Any,
    E: int,
    H: int,
    seed: int,
    tau_rel: float = 2.5,
    n_phys_substeps: int = 2,
) -> dict[str, Any]:
    K, J, N, Wn = int(world.teams), int(world.carts), int(world.players_default), int(world.num_nav)
    A_dim = Wn + K + 4
    rules = make_rules_for_profile(float(cfg.sim_dt.point_limit_base), regime["physics"])
    eng = MetalDeltaTimePayloadEngine(world, num_envs=E, num_players=N, rules=rules, base_seed=seed)
    rng = np.random.default_rng(seed + 999)
    skills = sample_roster_skills(rng, E, N, K, regime["roster"])
    eng.set_skills(skills)
    geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}
    c_main = (
        CompiledGraphFlowPolicy(model, E=E, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern)
        if model is not None
        else None
    )

    p_team_1d = (np.arange(N, dtype=np.int32) % K) + 1
    main_idx = np.nonzero(p_team_1d == 1)[0]
    n_main = len(main_idx)
    main_mask_t = torch.from_numpy(p_team_1d == 1).view(1, N).expand(E, N)

    team_skill_mean = np.zeros((E, K), dtype=np.float32)
    for k in range(K):
        team_skill_mean[:, k] = skills[:, p_team_1d == (k + 1)].mean(axis=1)
    enemy_skills = team_skill_mean[:, 1:]
    strongest_enemy_0 = np.argmax(enemy_skills, axis=1) + 1  # 0-based index in 1..K-1
    weakest_enemy_0 = np.argmin(enemy_skills, axis=1) + 1

    gk_counts = np.zeros((4,), dtype=np.int64)
    tot_kills = np.zeros((E,), dtype=np.float64)
    tot_deaths = np.zeros((E,), dtype=np.float64)
    tot_push = np.zeros((E,), dtype=np.float64)
    tot_contest = np.zeros((E,), dtype=np.float64)
    tot_pickups = np.zeros((E,), dtype=np.float64)
    tot_alive = np.zeros((E,), dtype=np.float64)
    tot_quad = np.zeros((E,), dtype=np.float64)

    u_samples: list[np.ndarray] = []
    r_cart_l, r_item_l, r_en_l, r_post_l = [], [], [], []
    r_foc_lead, r_foc_str, r_foc_wk, u_norm_l, u_sat_l = [], [], [], [], []
    role_pw_l, bot_wk_l, bot_ace_l = [], [], []
    gk_step_l = []

    try:
        for step in range(H):
            obs, nav, carts, teams, acc, _ = eng.observe_graph_tensors()
            if c_main is not None:
                res = c_main.sample_velocity(obs, nav, carts, teams, acc, geo=geo)
                u_vel = res["u_vel"]
                eng.apply_vel(u_vel, main_mask_t, tau_rel=tau_rel)
            eng.step(float(cfg.sim_dt.dt), mode=str(cfg.sim_dt.mode), reset_outcomes=True, n_steps=1, tau_rel=tau_rel)
            for _ in range(1, n_phys_substeps):
                eng.step(float(cfg.sim_dt.dt), mode=str(cfg.sim_dt.mode), reset_outcomes=False, n_steps=1, tau_rel=tau_rel)

            sv = eng.state_views()
            tot_kills += sv["p_kills"][:, main_idx].numpy().sum(axis=1)
            tot_deaths += sv["p_deaths"][:, main_idx].numpy().sum(axis=1)
            tot_push += sv["p_cart_push"][:, main_idx].numpy().sum(axis=1)
            tot_contest += sv["p_cart_contest"][:, main_idx].numpy().sum(axis=1)
            tot_pickups += sv["p_pickups"][:, main_idx].numpy().sum(axis=1)
            tot_alive += sv["p_alive"][:, main_idx].numpy().mean(axis=1)
            tot_quad += (sv["p_strength_t"][:, main_idx].numpy() > 0.0).mean(axis=1)
            gk_m = sv["p_goal_kind"][:, main_idx].numpy()
            for k_i in range(4):
                gk_counts[k_i] += int((gk_m == k_i).sum())

            if c_main is not None:
                teams_np = teams.float().numpy()
                eff_lead_0, has_pw = _projected_leader_idx(teams_np)
                is_pw_t1 = has_pw & (eff_lead_0 == 0)
                r_m = sv["p_r_acc"][:, main_idx, :].numpy()  # [E, n_main, A_dim]
                u_m = res["u_vel"][:, main_idx, :].numpy()   # [E, n_main, A_dim]
                if step % 3 == 0:
                    u_samples.append(u_m.reshape(-1, A_dim))
                u_norm_l.append(np.sqrt(np.mean(u_m ** 2, axis=-1)))
                u_sat_l.append((np.abs(u_m) >= (float(model.cfg.u_clip) - 0.1)).mean(axis=-1))
                r_cart_l.append(r_m[:, :, Wn + K + 0])
                r_item_l.append(r_m[:, :, Wn + K + 1])
                r_en_l.append(r_m[:, :, Wn + K + 2])
                r_post_l.append(np.tanh(r_m[:, :, Wn + K + 3]))

                r_tm = r_m[:, :, Wn : Wn + K]  # [E, n_main, K]
                # Focus relative to mean over enemy teams 1..K-1
                r_tm_en_mean = r_tm[:, :, 1:].mean(axis=-1)
                e_ar = np.arange(E)[:, None]
                lead_en = np.where(eff_lead_0 > 0, eff_lead_0, 1)
                r_foc_lead.append(r_tm[e_ar, np.arange(n_main)[None, :], lead_en[:, None]] - r_tm_en_mean)
                r_foc_str.append(r_tm[e_ar, np.arange(n_main)[None, :], strongest_enemy_0[:, None]] - r_tm_en_mean)
                r_foc_wk.append(r_tm[e_ar, np.arange(n_main)[None, :], weakest_enemy_0[:, None]] - r_tm_en_mean)
                role_pw_l.append(np.broadcast_to(is_pw_t1[:, None], (E, n_main)).copy())
                sk_m = skills[:, main_idx]
                bot_wk_l.append(sk_m < 0.0)
                bot_ace_l.append(sk_m > 10.0)
                gk_step_l.append(gk_m.copy())

        end_scores = eng.t_score.numpy()
        end_fin = eng.t_finished.numpy()
    finally:
        if c_main is not None:
            c_main.close()
        eng.close()

    t1_sc = end_scores[:, 0]
    opp_sc_mean = end_scores[:, 1:].mean(axis=1)
    opp_sc_max = end_scores[:, 1:].max(axis=1)
    rank1_win = (t1_sc > opp_sc_max + 1e-6).astype(np.float64) + 0.5 * (np.abs(t1_sc - opp_sc_max) <= 1e-6)
    pw_beats = np.mean(
        np.where(t1_sc[:, None] > end_scores[:, 1:] + 1e-6, 1.0,
                 np.where(end_scores[:, 1:] > t1_sc[:, None] + 1e-6, 0.0, 0.5)),
        axis=1,
    )
    gk_tot = max(1, int(gk_counts.sum()))
    out: dict[str, Any] = {
        "arm": arm_name,
        "regime": regime["name"],
        "outcomes": {
            "first_place_rate": round(float(rank1_win.mean()), 4),
            "pairwise_win_rate": round(float(pw_beats.mean()), 4),
            "score_margin": round(float((t1_sc - opp_sc_mean).mean()), 2),
            "kd_ratio": round(float(tot_kills.sum() / max(tot_deaths.sum(), 1.0)), 3),
            "cart_push": round(float(tot_push.mean()), 1),
            "pickups": round(float(tot_pickups.mean()), 2),
            "quad_frac": round(float((tot_quad / H).mean()), 4),
            "alive_frac": round(float((tot_alive / H).mean()), 4),
            "score_out_frac": round(float(end_fin.mean()), 4),
            "gk_pct": [round(100.0 * c / gk_tot, 1) for c in gk_counts.tolist()],
        },
    }
    if not u_norm_l:
        return out

    def cat(lst): return np.concatenate([a.reshape(-1) for a in lst], axis=0)
    un_a = cat(u_norm_l); us_a = cat(u_sat_l); rc_a = cat(r_cart_l); ri_a = cat(r_item_l); re_a = cat(r_en_l); rp_a = cat(r_post_l)
    fl_a = cat(r_foc_lead); fs_a = cat(r_foc_str); fw_a = cat(r_foc_wk)
    pw_a = cat(role_pw_l); wk_a = cat(bot_wk_l); ace_a = cat(bot_ace_l); gk_a = cat(gk_step_l)

    def _sl(m: np.ndarray) -> dict[str, Any]:
        if int(m.sum()) == 0:
            return {"n": 0}
        return {
            "n": int(m.sum()),
            "u_norm": round(float(un_a[m].mean()), 3),
            "sat_frac": round(float(us_a[m].mean()), 4),
            "r_cart": round(float(rc_a[m].mean()), 3),
            "r_item": round(float(ri_a[m].mean()), 3),
            "r_enemy": round(float(re_a[m].mean()), 3),
            "posture": round(float(rp_a[m].mean()), 3),
            "foc_lead": round(float(fl_a[m].mean()), 3),
            "foc_strong": round(float(fs_a[m].mean()), 3),
            "foc_weak": round(float(fw_a[m].mean()), 3),
            "gk_pct": [round(100.0 * float((gk_a[m] == k_i).mean()), 1) for k_i in range(4)],
        }

    out["steering_all"] = _sl(np.ones_like(pw_a, dtype=bool))
    out["by_role"] = {"winnie": _sl(pw_a), "lou": _sl(~pw_a)}
    if int(wk_a.sum()) > 0 and int(ace_a.sum()) > 0:
        out["within_team"] = {"weak": _sl(wk_a), "ace": _sl(ace_a)}

    # PCA spectrum of efferent velocity field u (direct interpretability metric!)
    U_mat = np.concatenate(u_samples, axis=0)
    U_c = U_mat - U_mat.mean(axis=0, keepdims=True)
    cov = (U_c.T @ U_c) / max(1, U_c.shape[0] - 1)
    eig = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
    eig_sum = max(float(eig.sum()), 1e-12)
    p_eig = eig / eig_sum
    eff_rank = float(np.exp(-np.sum(np.where(p_eig > 1e-12, p_eig * np.log(np.maximum(p_eig, 1e-12)), 0.0))))
    out["pca"] = {
        "top1_var": round(float(p_eig[0]), 4),
        "top3_var": round(float(p_eig[:3].sum()), 4),
        "top8_var": round(float(p_eig[:8].sum()), 4),
        "eff_rank": round(eff_rank, 2),
    }
    return out


@torch.no_grad()
def evaluate_multimodal_branch_diversity(
    model: GraphFlowPolicy,
    world: Any,
    cfg: Any,
    *,
    B: int = 16,
    G: int = 8,
    H: int = 20,
    seed: int = 7777,
) -> dict[str, float]:
    """Fork G=8 branches from IDENTICAL mid-match savestates and measure genuine plural strategy divergence!"""
    K, J, N, Wn = int(world.teams), int(world.carts), int(world.players_default), int(world.num_nav)
    A_dim = Wn + K + 4
    rules = make_rules_for_profile(float(cfg.sim_dt.point_limit_base), "baseline")
    base = MetalDeltaTimePayloadEngine(world, num_envs=B, num_players=N, rules=rules, base_seed=seed)
    base.step(0.4, n_steps=18)  # advance into mid-match!
    snap = base.snapshot()
    fork = MetalDeltaTimePayloadEngine.fork_group(world, snap, G, rules=rules, common_random_numbers=True)
    base.close()

    E = B * G
    c_pol = CompiledGraphFlowPolicy(model, E=E, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern)
    geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}
    gen = torch.Generator(device="cpu").manual_seed(seed + 13)
    p_team_1d = (np.arange(N, dtype=np.int32) % K) + 1
    main_idx = np.nonzero(p_team_1d == 1)[0]
    main_mask = torch.from_numpy(p_team_1d == 1).view(1, N).expand(E, N)

    # First check exact output_scale=0 collapse!
    obs0, nav0, carts0, teams0, acc0, _ = fork.observe_graph_tensors()
    z0 = c_pol.sample_velocity(obs0, nav0, carts0, teams0, acc0, geo=geo, output_scale=0.0)
    zero_err = float(z0["u_vel"].abs().max().item())

    # Step H strategy ticks from IDENTICAL savestates with CRN physics!
    u_pair_l2 = []
    u_pair_cos = []
    for step in range(H):
        obs, nav, carts, teams, acc, _ = fork.observe_graph_tensors()
        p_team = torch.round(obs[:, :, 1].float()).to(torch.int64)
        eps = sample_orthogonal_team_bot_noise(B, G, N, model.d_e, p_team, K, team_corr=0.45, sigma=model.cfg.sigma_base, generator=gen)
        res = c_pol.sample_velocity(obs, nav, carts, teams, acc, geo=geo, eps=eps)
        u_vel = res["u_vel"]
        if step == 0:
            # At IDENTICAL initial state s_0: measure exact inter-branch velocity diversity!
            ug = u_vel[:, main_idx, :].view(B, G, -1)  # team 1 joint strategy vector per branch!
            diff = (ug[:, :, None, :] - ug[:, None, :, :]).square().mean(dim=-1).sqrt()
            un = ug / ug.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cos = torch.einsum("bgd,bhd->bgh", un, un)
            off = ~torch.eye(G, dtype=torch.bool).unsqueeze(0).expand(B, G, G)
            u_pair_l2.append(float(diff[off].mean().item()))
            u_pair_cos.append(float(cos[off].mean().item()))
        fork.apply_vel(u_vel, main_mask, tau_rel=2.5)
        fork.step(0.4, n_steps=2, tau_rel=2.5)

    # Measure realized trajectory divergence after H=20 steps (16s) under IDENTICAL CRN physics!
    pos_end = fork.p_pos[:, main_idx, :].view(B, G, len(main_idx), 3)
    p_dist = (pos_end[:, :, None] - pos_end[:, None, :]).norm(dim=-1).mean(dim=-1)  # [B, G, G]
    gk_end = fork.state_views()["p_goal_kind"][:, main_idx].view(B, G, len(main_idx))
    gk_dis = (gk_end[:, :, None] != gk_end[:, None, :]).float().mean(dim=-1)
    off = ~torch.eye(G, dtype=torch.bool).unsqueeze(0).expand(B, G, G)
    sc_end = fork.t_score[:, 0].view(B, G)

    # Conditional endpoint distribution probe: 128 i.i.d. draws at ONE fixed mid-match state s_0!
    c_128 = CompiledGraphFlowPolicy(model, E=128, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern)
    res_128 = c_128.sample_velocity(
        obs0[:1].expand(128, -1, -1).contiguous(),
        nav0[:1].expand(128, -1, -1).contiguous(),
        carts0[:1].expand(128, -1, -1).contiguous(),
        teams0[:1].expand(128, -1, -1).contiguous(),
        acc0[:1].expand(128, -1, -1).contiguous(),
        geo=geo,
    )
    U128 = res_128["u_vel"][:, main_idx, :].reshape(128, -1).numpy()  # [128, n_main * A_dim]
    u128_mu = U128.mean(axis=0)
    u128_c = U128 - u128_mu
    u128_spread = float(np.sqrt(np.mean(np.sum(u128_c ** 2, axis=1))))
    u128_mu_norm = float(np.linalg.norm(u128_mu))
    u128_sat = float((np.abs(U128) >= (float(model.cfg.u_clip) - 0.1)).mean())
    u128_rms = float(np.sqrt(np.mean(U128 ** 2)))
    u128_n = U128 / np.maximum(np.linalg.norm(U128, axis=1, keepdims=True), 1e-6)
    cos128 = u128_n @ u128_n.T
    u128_pair_cos = float(cos128[~np.eye(128, dtype=bool)].mean())
    cov128 = (u128_c.T @ u128_c) / 127.0
    ev128, evec128 = np.linalg.eigh(cov128)
    ev128 = np.maximum(ev128[::-1], 0.0)
    p128 = ev128 / max(float(ev128.sum()), 1e-12)
    eff_r128 = float(np.exp(-np.sum(np.where(p128 > 1e-12, p128 * np.log(np.maximum(p128, 1e-12)), 0.0))))
    pc1 = u128_c @ evec128[:, -1]
    pc1_z = (pc1 - pc1.mean()) / max(float(pc1.std()), 1e-6)
    skew1 = float(np.mean(pc1_z ** 3))
    kurt1 = float(np.mean(pc1_z ** 4))
    bimod1 = (skew1 ** 2 + 1.0) / max(kurt1, 1e-6)
    # Also measure single-bot p(u_0 | s_0) and B_H basis effective rank (guards against rank-1 basis collapse!)
    U128_b0 = res_128["u_vel"][:, main_idx[0], :].numpy()  # [128, A_dim]
    b0_mu = U128_b0.mean(axis=0)
    b0_c = U128_b0 - b0_mu
    b0_spr = float(np.sqrt(np.mean(np.sum(b0_c ** 2, axis=1))))
    b0_cov = (b0_c.T @ b0_c) / 127.0
    b0_ev, b0_evec = np.linalg.eigh(b0_cov)
    b0_ev = np.maximum(b0_ev[::-1], 0.0)
    b0_p = b0_ev / max(float(b0_ev.sum()), 1e-12)
    b0_eff_r = float(np.exp(-np.sum(np.where(b0_p > 1e-12, b0_p * np.log(np.maximum(b0_p, 1e-12)), 0.0))))
    b0_pc1 = b0_c @ b0_evec[:, -1]
    b0_z = (b0_pc1 - b0_pc1.mean()) / max(float(b0_pc1.std()), 1e-6)
    b0_bimod = (float(np.mean(b0_z ** 3)) ** 2 + 1.0) / max(float(np.mean(b0_z ** 4)), 1e-6)
    enc0 = model.encode_backbone(obs0[:1], nav0[:1], carts0[:1], teams0[:1], acc0[:1], world.nav_kern, geo=geo)
    sv_bh = np.linalg.svd(enc0["B_H"][0].float().numpy(), compute_uv=False)
    p_bh = (sv_bh ** 2) / max(float((sv_bh ** 2).sum()), 1e-12)
    bh_eff_r = float(np.exp(-np.sum(np.where(p_bh > 1e-12, p_bh * np.log(np.maximum(p_bh, 1e-12)), 0.0))))
    c_128.close()

    c_pol.close()
    fork.close()
    return {
        "zero_collapse_max_u": zero_err,
        "s0_branch_u_l2": round(float(np.mean(u_pair_l2)), 4),
        "s0_branch_u_cos": round(float(np.mean(u_pair_cos)), 4),
        "end_pos_divergence_u": round(float(p_dist[off].mean().item()), 1),
        "end_goal_kind_disagree": round(float(gk_dis[off].mean().item()), 4),
        "end_score_std_across_G": round(float(sc_end.std(dim=1).mean().item()), 2),
        "s0_128_u_rms": round(u128_rms, 4),
        "s0_128_sat_frac": round(u128_sat, 4),
        "s0_128_spread_over_mean": round(u128_spread / max(u128_mu_norm, 1e-6), 4),
        "s0_128_pair_cos": round(u128_pair_cos, 4),
        "s0_128_eff_rank": round(eff_r128, 2),
        "s0_128_pc1_var": round(float(p128[0]), 4),
        "s0_128_pc1_bimod": round(bimod1, 4),
        "s0_bot0_eff_rank": round(b0_eff_r, 2),
        "s0_bot0_spr_over_mu": round(b0_spr / max(float(np.linalg.norm(b0_mu)), 1e-6), 4),
        "s0_bot0_bimod": round(b0_bimod, 4),
        "bh_eff_rank": round(bh_eff_r, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-time behavior & multimodal diversity analyzer for GraphFlowPolicy")
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--teams", type=int, default=6)
    ap.add_argument("--carts", type=int, default=16)
    ap.add_argument("--players", type=int, default=36)
    ap.add_argument("--tiles", type=int, default=4)
    args = ap.parse_args()

    cfg = load_training_config(args.config)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    world = build_gas_world_graph(
        map_name=f"eval_gf_k{args.teams}_j{args.carts}_n{args.players}",
        teams=args.teams, carts=args.carts, players=args.players, tiles=args.tiles,
        seed=8101, device=torch.device("cpu"),
    )

    gf_cfg = GraphFlowConfig()
    models: dict[str, GraphFlowPolicy | None] = {"default": None}
    for name in ("main_init", "main_mid", "main_final"):
        p = ckpt_dir / f"{name}.pt"
        if p.exists():
            m = GraphFlowPolicy(gf_cfg, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            m.eval()
            models[name] = m

    t0 = time.perf_counter()
    results: dict[str, Any] = {"regimes": {}, "branch_diversity": {}}
    for ri, reg in enumerate(EVAL_REGIMES):
        results["regimes"][reg["name"]] = {}
        for ai, (arm_name, mdl) in enumerate(models.items()):
            results["regimes"][reg["name"]][arm_name] = evaluate_arm_on_regime(
                arm_name, mdl, world, reg,
                cfg=cfg, E=args.envs, H=args.horizon, seed=90000 + ri * 1000 + ai * 37,
            )

    for arm_name in ("main_init", "main_final"):
        if arm_name in models and models[arm_name] is not None:
            results["branch_diversity"][arm_name] = evaluate_multimodal_branch_diversity(models[arm_name], world, cfg)

    out_path = ckpt_dir / "graph_flow_behavior_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval-graph-flow] Done in {time.perf_counter()-t0:.2f}s (E={args.envs}, H={args.horizon}, W_nav={world.num_nav}, I_item={world.num_items}) -> {out_path}\n", flush=True)

    print("=== 1. OUTCOME COMPETENCE & REALIZED GOAL-KIND MIX ACROSS 7 DR REGIMES ===")
    print(f"{'Regime':24s} | {'Arm':11s} | {'PW-WR':6s} {'1st%':6s} {'Margin':7s} {'K/D':5s} {'Push':6s} {'Pick':5s} {'Quad%':5s} | {'GoalKind [Cart/Item/Enem/Nav]%':30s}")
    print("-" * 118)
    for reg in EVAL_REGIMES:
        rn = reg["name"]
        for arm_name in models:
            o = results["regimes"][rn][arm_name]["outcomes"]
            gk = o["gk_pct"]
            print(
                f"{rn:24s} | {arm_name:11s} | {o['pairwise_win_rate']:6.3f} {o['first_place_rate']:6.3f} "
                f"{o['score_margin']:+7.1f} {o['kd_ratio']:5.2f} {o['cart_push']:6.1f} {o['pickups']:5.1f} {o['quad_frac']*100:5.1f} | "
                f"{gk[0]:5.1f} / {gk[1]:5.1f} / {gk[2]:5.1f} / {gk[3]:5.1f}"
            )

    if "main_final" in models:
        print("\n=== 2. CONTINUOUS STEERING FIELD & PCA SPECTRUM (main_init vs main_final) ===")
        print(f"{'Regime':24s} | {'Arm':10s} | {'||u||':5s} {'Sat%':5s} {'rCart':6s} {'rItem':6s} {'rEnem':6s} {'Post':6s} | {'FocLd':6s} {'FocSt':6s} {'FocWk':6s} | {'PCA(1/3/8) EffR':18s}")
        print("-" * 122)
        for reg in EVAL_REGIMES:
            rn = reg["name"]
            for an in ("main_init", "main_final"):
                if an not in models:
                    continue
                r = results["regimes"][rn][an]
                s = r["steering_all"]
                p = r["pca"]
                print(
                    f"{rn:24s} | {an:10s} | {s['u_norm']:5.2f} {s['sat_frac']*100:5.1f} {s['r_cart']:+6.2f} {s['r_item']:+6.2f} {s['r_enemy']:+6.2f} {s['posture']:+6.2f} | "
                    f"{s['foc_lead']:+6.2f} {s['foc_strong']:+6.2f} {s['foc_weak']:+6.2f} | "
                    f"{p['top1_var']:.2f}/{p['top3_var']:.2f}/{p['top8_var']:.2f} ({p['eff_rank']:4.1f})"
                )

        print("\n=== 3. ROLE SHIFTS (WINNIE vs LOU) & WITHIN-TEAM MECHANICS (WEAK vs ACE) ===")
        for rn in ("balanced_baseline", "stacked_main_weak", "mixed_enemies_main_weak", "within_team_carry_weak"):
            r = results["regimes"][rn]["main_final"]
            for label, sl in [("Winnie(PW)", r["by_role"]["winnie"]), ("Lou(Loser)", r["by_role"]["lou"])]:
                if sl.get("n", 0) == 0: continue
                print(f"  [{rn:23s}] {label:10s}: rCart={sl['r_cart']:+.2f} rItem={sl['r_item']:+.2f} rEn={sl['r_enemy']:+.2f} Post={sl['posture']:+.2f} Foc(Ld={sl['foc_lead']:+.2f},Wk={sl['foc_weak']:+.2f}) GK={sl['gk_pct']}")
            if "within_team" in r:
                for label, sl in [("Weak(sk<0)", r["within_team"]["weak"]), ("Ace(sk>10)", r["within_team"]["ace"])]:
                    print(f"  [{rn:23s}] {label:10s}: rCart={sl['r_cart']:+.2f} rItem={sl['r_item']:+.2f} rEn={sl['r_enemy']:+.2f} Post={sl['posture']:+.2f} Foc(Ld={sl['foc_lead']:+.2f},Wk={sl['foc_weak']:+.2f}) GK={sl['gk_pct']}")

        print("\n=== 4. MULTIMODAL ROLLOUT-GROUP DIVERSITY & CONDITIONAL p(u|s0) ENDPOINTS ===")
        for an, dv in results["branch_diversity"].items():
            print(
                f"  {an:10s}: zero_err={dv['zero_collapse_max_u']:.1e} | "
                f"G8(u_L2={dv['s0_branch_u_l2']:.3f} u_cos={dv['s0_branch_u_cos']:+.3f} pos_div={dv['end_pos_divergence_u']:.1f}u gk_dis={dv['end_goal_kind_disagree']:.3f} sc_std={dv['end_score_std_across_G']:.2f}) | "
                f"p(u|s0,128)(uRms={dv['s0_128_u_rms']:.2f} sat={dv['s0_128_sat_frac']*100:.2f}% spr/mu={dv['s0_128_spread_over_mean']:.2f} cos={dv['s0_128_pair_cos']:+.3f} tmEffR={dv['s0_128_eff_rank']:.1f} bot0EffR={dv['s0_bot0_eff_rank']:.1f} BH_effR={dv['bh_eff_rank']:.1f} bimod={dv['s0_bot0_bimod']:.3f})"
            )


if __name__ == "__main__":
    main()
