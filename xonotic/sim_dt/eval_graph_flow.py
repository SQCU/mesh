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
    make_graph_flow_config,
    sample_orthogonal_team_bot_noise,
)
from sim_dt.metal_graphflow import CompiledGraphFlowPolicy
from sim_dt.metal_runtime import MetalDeltaTimePayloadEngine
from sim_dt.world_graph import build_gas_world_graph
from solver.strat.league_curriculum import alpha_rank_3team
from solver.strat.train_config import CONFIGS_DIR, load_training_config


UNSEEN_LEAGUE_WORLDS: list[dict[str, Any]] = [
    {"name": "unseen_k2_duel",   "scale": "small", "seed": 99101, "teams": 2,  "carts": 6,  "players": 16, "tiles": 2, "tile_size": 1536.0, "cp": 3, "physics": "baseline"},
    {"name": "unseen_k3_tri",    "scale": "small", "seed": 99102, "teams": 3,  "carts": 11, "players": 24, "tiles": 3, "tile_size": 1792.0, "cp": 4, "physics": "fast"},
    {"name": "unseen_k4_quad",   "scale": "small", "seed": 99103, "teams": 4,  "carts": 14, "players": 32, "tiles": 4, "tile_size": 1664.0, "cp": 4, "physics": "volatile"},
    {"name": "unseen_k6_mid",    "scale": "mid",   "seed": 99104, "teams": 6,  "carts": 20, "players": 42, "tiles": 4, "tile_size": 1792.0, "cp": 4, "physics": "baseline"},
    {"name": "unseen_k8_crowd",  "scale": "mid",   "seed": 99105, "teams": 8,  "carts": 28, "players": 56, "tiles": 6, "tile_size": 2048.0, "cp": 5, "physics": "slow"},
    {"name": "unseen_k12_mega",  "scale": "large", "seed": 99106, "teams": 12, "carts": 40, "players": 84, "tiles": 9, "tile_size": 1792.0, "cp": 4, "physics": "baseline"},
]


def fit_bradley_terry_elo(
    arms: list[str],
    wins: np.ndarray,
    games: np.ndarray,
    *,
    anchor: str = "default",
    anchor_elo: float = 1000.0,
    prior: float = 0.25,
    n_iter: int = 400,
) -> dict[str, float]:
    """Exact MM algorithm for Maximum-Likelihood Bradley-Terry Elo anchored at `anchor` = `anchor_elo`."""
    n = len(arms)
    w = wins.astype(np.float64) + prior
    g = games.astype(np.float64) + 2.0 * prior
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(g, 0.0)
    p = np.ones((n,), dtype=np.float64)
    w_tot = w.sum(axis=1)
    for _ in range(n_iter):
        denom = np.sum(g / (p[:, None] + p[None, :] + 1e-12), axis=1)
        p_new = np.maximum(w_tot / np.maximum(denom, 1e-12), 1e-9)
        p = p_new / np.exp(np.mean(np.log(p_new)))
    elo = 400.0 * np.log10(p)
    a_idx = arms.index(anchor) if anchor in arms else 0
    elo = elo - elo[a_idx] + float(anchor_elo)
    return {arms[i]: round(float(elo[i]), 1) for i in range(n)}


@torch.no_grad()
def evaluate_league_tournament(
    models: dict[str, GraphFlowPolicy | None],
    cfg: Any,
    *,
    E_match: int = 48,
    H: int = 72,
    seed: int = 88000,
) -> dict[str, Any]:
    """Held-out unseen parametric-world multi-team tournament across K=2..12 + balanced/mixed_dr rosters.

    Computes:
      - Exact Bradley-Terry Elo (overall, by scale Small/Mid/Large, by roster balanced vs mixed_dr) + 95% CIs
      - Full pairwise win-rate matrix across all league arms (including frozen exploiters & EMAs!)
      - Exact 3-team alpha-Rank stationary distribution
    """
    arms = list(models.keys())
    A = len(arms)
    arm_idx = {a: i for i, a in enumerate(arms)}
    wins_all = np.zeros((A, A), dtype=np.float64)
    games_all = np.zeros((A, A), dtype=np.float64)
    wins_by_scale = {sc: np.zeros((A, A), dtype=np.float64) for sc in ("small", "mid", "large")}
    games_by_scale = {sc: np.zeros((A, A), dtype=np.float64) for sc in ("small", "mid", "large")}
    wins_by_roster = {rm: np.zeros((A, A), dtype=np.float64) for rm in ("balanced_fixed", "mixed_dr")}
    games_by_roster = {rm: np.zeros((A, A), dtype=np.float64) for rm in ("balanced_fixed", "mixed_dr")}
    match_records: list[dict[str, float]] = []
    boot_outcomes: list[tuple[int, int, float]] = []

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    for wi, wspec in enumerate(UNSEEN_LEAGUE_WORLDS):
        K = int(wspec["teams"])
        J = int(wspec["carts"])
        N = int(wspec["players"])
        world = build_gas_world_graph(
            map_name=wspec["name"],
            teams=K, carts=J, players=N, tiles=int(wspec["tiles"]),
            tile_size=float(wspec["tile_size"]),
            checkpoints_per_cart=int(wspec["cp"]),
            seed=int(wspec["seed"]),
            device=torch.device("cpu"),
        )
        Wn = int(world.num_nav)
        A_dim = Wn + K + 4
        geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}
        rules = make_rules_for_profile(float(cfg.sim_dt.point_limit_base), wspec["physics"])
        compiled: dict[str, CompiledGraphFlowPolicy | None] = {
            a: (CompiledGraphFlowPolicy(m, E=E_match, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern) if m is not None else None)
            for a, m in models.items()
        }
        p_team_1d = (np.arange(N, dtype=np.int32) % K) + 1
        team_masks = [torch.from_numpy(p_team_1d == (k + 1)).view(1, N).expand(E_match, N) for k in range(K)]

        # Build balanced match seatings covering every arm pair evenly!
        seatings: list[list[str]] = []
        if K == 2:
            for i in range(A):
                for j in range(i + 1, A):
                    seatings.append([arms[i], arms[j]])
        else:
            # Cover every distinct pair (i, j) by placing them in seats 0 & 1 + rotating/randomizing remaining K-2 seats!
            n_lobbies = max(A * 2, 18)
            for li in range(n_lobbies):
                if K <= A:
                    perm = rng.permutation(A)[:K]
                    seatings.append([arms[int(x)] for x in perm])
                else:
                    # K > A (e.g. K=12, A=9): include every arm at least once + sample remaining uniformly
                    base = list(rng.permutation(A))
                    extra = list(rng.choice(A, size=K - A, replace=True))
                    full = base + extra
                    rng.shuffle(full)
                    seatings.append([arms[int(x)] for x in full])
            # Also guarantee direct head-to-head inclusion of key pairs (exploiters vs targets, main_final vs EMAs/default)
            key_pairs = [
                ("main_final", "default"), ("main_final", "main_init"), ("main_final", "ema_slow_final"),
                ("main_final", "ema_fast_final"), ("exploiter_1", "main_p1"), ("exploiter_1", "main_p2"),
                ("exploiter_1", "main_final"), ("exploiter_2", "main_p2"), ("exploiter_2", "main_final"),
            ]
            for a1, a2 in key_pairs:
                if a1 in arm_idx and a2 in arm_idx:
                    others = [arms[int(x)] for x in rng.choice(A, size=max(0, K - 2), replace=True)]
                    s = [a1, a2, *others]
                    rng.shuffle(s)
                    seatings.append(s)

        for rm in ("balanced_fixed", "mixed_dr"):
            eng = MetalDeltaTimePayloadEngine(world, num_envs=E_match, num_players=N, rules=rules, base_seed=seed + wi * 1000 + (0 if rm == "balanced_fixed" else 500))
            for si, seats in enumerate(seatings):
                eng.reset(torch.ones((E_match,), dtype=torch.bool))
                sk = sample_roster_skills(rng, E_match, N, K, rm)
                eng.set_skills(sk)
                arm_to_tmask: dict[str, torch.Tensor] = {}
                for ti, a in enumerate(seats):
                    if a == "default":
                        continue
                    arm_to_tmask[a] = team_masks[ti] if a not in arm_to_tmask else (arm_to_tmask[a] | team_masks[ti])

                q_ou_arm: dict[str, torch.Tensor] = {}
                obs, nav, carts, teams, acc, _ = eng.observe_graph_tensors()
                for _ in range(H):
                    u_vel = torch.zeros((E_match, N, A_dim), dtype=torch.float32)
                    has_vel = torch.zeros((E_match, N), dtype=torch.bool)
                    for a, tmask in arm_to_tmask.items():
                        cp = compiled[a]
                        if cp is None:
                            continue
                        rho_a = max(0.0, min(0.95, float(getattr(cp.cfg, "ou_rho", 0.0))))
                        eps_in = None
                        if rho_a > 0.0:
                            raw_e = torch.randn((E_match, N, int(cp.cfg.d_e)), dtype=torch.float32) * float(cp.cfg.sigma_base)
                            q_ou_arm[a] = raw_e if a not in q_ou_arm else (rho_a * q_ou_arm[a] + math.sqrt(max(0.0, 1.0 - rho_a * rho_a)) * raw_e)
                            eps_in = q_ou_arm[a]
                        res = cp.sample_velocity(obs, nav, carts, teams, acc, geo=geo, eps=eps_in)
                        u_vel = torch.where(tmask.unsqueeze(-1), res["u_vel"], u_vel)
                        has_vel = has_vel | tmask
                    eng.apply_vel(u_vel, has_vel, tau_rel=2.5)
                    eng.step(float(cfg.sim_dt.dt), mode=str(cfg.sim_dt.mode), reset_outcomes=True, n_steps=1, tau_rel=2.5)
                    eng.step(float(cfg.sim_dt.dt), mode=str(cfg.sim_dt.mode), reset_outcomes=False, n_steps=1, tau_rel=2.5)
                    obs, nav, carts, teams, acc, _ = eng.observe_graph_tensors()

                sc = eng.t_score.numpy()  # [E_match, K]
                # Record pairwise outcomes across every pair of distinct arms in this match!
                for ti in range(K):
                    ai = arm_idx[seats[ti]]
                    for tj in range(ti + 1, K):
                        aj = arm_idx[seats[tj]]
                        if ai == aj:
                            continue
                        w_ij = np.where(sc[:, ti] > sc[:, tj] + 1e-6, 1.0, np.where(sc[:, tj] > sc[:, ti] + 1e-6, 0.0, 0.5))
                        s_ij = float(w_ij.sum())
                        n_ij = float(E_match)
                        wins_all[ai, aj] += s_ij
                        wins_all[aj, ai] += n_ij - s_ij
                        games_all[ai, aj] += n_ij
                        games_all[aj, ai] += n_ij
                        wins_by_scale[wspec["scale"]][ai, aj] += s_ij
                        wins_by_scale[wspec["scale"]][aj, ai] += n_ij - s_ij
                        games_by_scale[wspec["scale"]][ai, aj] += n_ij
                        games_by_scale[wspec["scale"]][aj, ai] += n_ij
                        wins_by_roster[rm][ai, aj] += s_ij
                        wins_by_roster[rm][aj, ai] += n_ij - s_ij
                        games_by_roster[rm][ai, aj] += n_ij
                        games_by_roster[rm][aj, ai] += n_ij
                        boot_outcomes.append((ai, aj, s_ij / n_ij))
                if K == 3 and len(set(seats)) == 3:
                    for e_i in range(E_match):
                        match_records.append({seats[0]: float(sc[e_i, 0]), seats[1]: float(sc[e_i, 1]), seats[2]: float(sc[e_i, 2])})
            eng.close()

        for cp in compiled.values():
            if cp is not None:
                cp.close()

    elo_all = fit_bradley_terry_elo(arms, wins_all, games_all)
    elo_scale = {sc: fit_bradley_terry_elo(arms, wins_by_scale[sc], games_by_scale[sc]) for sc in ("small", "mid", "large")}
    elo_roster = {rm: fit_bradley_terry_elo(arms, wins_by_roster[rm], games_by_roster[rm]) for rm in ("balanced_fixed", "mixed_dr")}

    # Bootstrap 95% CIs over lobby blocks
    boot_elos: dict[str, list[float]] = {a: [] for a in arms}
    n_bo = len(boot_outcomes)
    if n_bo > 0:
        for _ in range(120):
            bw = np.zeros((A, A), dtype=np.float64)
            bg = np.zeros((A, A), dtype=np.float64)
            idxs = rng.integers(0, n_bo, size=n_bo)
            for idx_b in idxs:
                ai, aj, wr = boot_outcomes[int(idx_b)]
                bw[ai, aj] += wr
                bw[aj, ai] += 1.0 - wr
                bg[ai, aj] += 1.0
                bg[aj, ai] += 1.0
            be = fit_bradley_terry_elo(arms, bw, bg)
            for a in arms:
                boot_elos[a].append(be[a])
    elo_ci = {
        a: [round(float(np.percentile(boot_elos[a], 2.5)), 1), round(float(np.percentile(boot_elos[a], 97.5)), 1)]
        if boot_elos[a] else [elo_all[a], elo_all[a]]
        for a in arms
    }

    # Exact 3-team alpha-Rank (uses direct 3-team match records + pairwise fallback!)
    for i in range(A):
        for j in range(i + 1, A):
            if games_all[i, j] > 0:
                wr = wins_all[i, j] / games_all[i, j]
                # Feed aggregate pairwise matches into alpha_rank_3team alongside 3-team games
                match_records.append({arms[i]: float(wr), arms[j]: float(1.0 - wr)})
    arank = alpha_rank_3team(arms, match_records, alpha=4.0, m_pop=20)

    pw_mat = {
        arms[i]: {
            arms[j]: (round(float(wins_all[i, j] / games_all[i, j]), 3) if games_all[i, j] > 0 else None)
            for j in range(A)
        }
        for i in range(A)
    }
    return {
        "wall_s": round(time.perf_counter() - t0, 2),
        "total_pair_games": int(games_all.sum() // 2),
        "arms": arms,
        "elo_overall": elo_all,
        "elo_ci95": elo_ci,
        "elo_by_scale": elo_scale,
        "elo_by_roster": elo_roster,
        "alpha_rank": arank,
        "pairwise_wr": pw_mat,
    }


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
        q_ou_main: torch.Tensor | None = None
        for step in range(H):
            obs, nav, carts, teams, acc, _ = eng.observe_graph_tensors()
            if c_main is not None:
                rho_m = max(0.0, min(0.95, float(getattr(c_main.cfg, "ou_rho", 0.0))))
                eps_in = None
                if rho_m > 0.0:
                    raw_e = torch.randn((E, N, int(c_main.cfg.d_e)), dtype=torch.float32) * float(c_main.cfg.sigma_base)
                    q_ou_main = raw_e if q_ou_main is None else (rho_m * q_ou_main + math.sqrt(max(0.0, 1.0 - rho_m * rho_m)) * raw_e)
                    eps_in = q_ou_main
                res = c_main.sample_velocity(obs, nav, carts, teams, acc, geo=geo, eps=eps_in)
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


def _vendi_score_rbf(X: np.ndarray) -> float:
    """Exact Vendi Score VS_1 = exp(-sum_i lam_i log lam_i) on RBF Gram matrix (median-bandwidth heuristic)."""
    n = int(X.shape[0])
    if n < 2:
        return 1.0
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    off = d2[~np.eye(n, dtype=bool)]
    med = float(np.median(off)) if off.size > 0 else 1.0
    if med <= 1e-10:
        return 1.0
    K = np.exp(-d2 / (2.0 * med)) / float(n)
    ev = np.maximum(np.linalg.eigvalsh(K), 0.0)
    ev = ev / max(float(ev.sum()), 1e-12)
    return float(np.exp(-np.sum(np.where(ev > 1e-12, ev * np.log(np.maximum(ev, 1e-12)), 0.0))))


def _non_gaussianity_index(X: np.ndarray, rng: np.random.Generator) -> float:
    """Normalized Energy Distance ED(P, N(mu, Sigma)) / sqrt(tr(Sigma)) measuring non-Gaussianity of p(u | s_0)."""
    n, d = X.shape
    if n < 8:
        return 0.0
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / float(n - 1)
    tr_s = float(np.trace(cov))
    if tr_s <= 1e-10:
        return 0.0
    ev, evec = np.linalg.eigh(cov)
    ev = np.maximum(ev, 0.0)
    Z = rng.standard_normal((n, d))
    Y = mu[None, :] + (Z * np.sqrt(ev)[None, :]) @ evec.T
    d_xy = np.sqrt(np.maximum(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1), 0.0)).mean()
    d_xx = np.sqrt(np.maximum(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1), 0.0)).mean()
    d_yy = np.sqrt(np.maximum(np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1), 0.0)).mean()
    ed2 = max(2.0 * float(d_xy) - float(d_xx) - float(d_yy), 0.0)
    return float(np.sqrt(ed2) / np.sqrt(tr_s))


@torch.no_grad()
def evaluate_effector_diversity_suite(
    models: dict[str, GraphFlowPolicy | None],
    world: Any,
    cfg: Any,
    *,
    n_states: int = 16,
    n_draws: int = 64,
    qd_G: int = 16,
    qd_H: int = 16,
    seed: int = 79000,
) -> dict[str, Any]:
    """Production Effector Coinductive Vector Machine Diversity across 16 mid-match states x 5 PvP-Elo strata.

    Reports per arm:
      - Vendi Score (VS_1) over single-bot and team-joint p(u | s_0)
      - Non-Gaussianity Index (NGI = normalized Energy Distance vs moment-matched Gaussian)
      - Quality-Diversity High-EV Vendi Score (QD-VS over top-25% CRN rollout branches!)
      - BH_effR, bot0EffR, tmEffR, sat_frac, zero_collapse_max_u
    """
    K, J, N, Wn = int(world.teams), int(world.carts), int(world.players_default), int(world.num_nav)
    A_dim = Wn + K + 4
    geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}
    rng = np.random.default_rng(seed)
    p_team_1d = (np.arange(N, dtype=np.int32) % K) + 1
    main_idx = np.nonzero(p_team_1d == 1)[0]
    strata_regimes = ["balanced_fixed", "stacked_main_weak", "mixed_enemies_main_weak", "stacked_main_strong", "within_team_carry_weak"]

    # Harvest 16 canonical mid-match savestates across the 5 PvP-Elo roster strata!
    rules = make_rules_for_profile(float(cfg.sim_dt.point_limit_base), "baseline")
    eng_s = MetalDeltaTimePayloadEngine(world, num_envs=n_states, num_players=N, rules=rules, base_seed=seed)
    skills = np.zeros((n_states, N), dtype=np.float32)
    for si in range(n_states):
        rm = strata_regimes[si % len(strata_regimes)]
        skills[si] = sample_roster_skills(rng, 1, N, K, rm)[0]
    eng_s.set_skills(skills)
    eng_s.step(0.4, n_steps=20)
    snap_all = eng_s.snapshot()
    obs_s, nav_s, carts_s, teams_s, acc_s, _ = eng_s.observe_graph_tensors()
    obs_s, nav_s, carts_s, teams_s, acc_s = obs_s.clone(), nav_s.clone(), carts_s.clone(), teams_s.clone(), acc_s.clone()
    eng_s.close()

    out: dict[str, Any] = {}
    for arm_name, model in models.items():
        if model is None:
            continue
        c_draw = CompiledGraphFlowPolicy(model, E=n_draws, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern)
        z0 = c_draw.sample_velocity(
            obs_s[:1].expand(n_draws, -1, -1).contiguous(),
            nav_s[:1].expand(n_draws, -1, -1).contiguous(),
            carts_s[:1].expand(n_draws, -1, -1).contiguous(),
            teams_s[:1].expand(n_draws, -1, -1).contiguous(),
            acc_s[:1].expand(n_draws, -1, -1).contiguous(),
            geo=geo, output_scale=0.0,
        )
        zero_err = float(z0["u_vel"].abs().max().item())

        vs_bot_l, vs_tm_l, ngi_bot_l, ngi_tm_l = [], [], [], []
        b0_eff_l, tm_eff_l, bh_eff_l, sat_l = [], [], [], []
        for si in range(n_states):
            res = c_draw.sample_velocity(
                obs_s[si : si + 1].expand(n_draws, -1, -1).contiguous(),
                nav_s[si : si + 1].expand(n_draws, -1, -1).contiguous(),
                carts_s[si : si + 1].expand(n_draws, -1, -1).contiguous(),
                teams_s[si : si + 1].expand(n_draws, -1, -1).contiguous(),
                acc_s[si : si + 1].expand(n_draws, -1, -1).contiguous(),
                geo=geo,
            )
            U_tm = res["u_vel"][:, main_idx, :].reshape(n_draws, -1).numpy()
            U_b0 = res["u_vel"][:, main_idx[0], :].numpy()
            sat_l.append(float((np.abs(U_tm) >= (float(model.cfg.u_clip) - 0.1)).mean()))
            vs_bot_l.append(_vendi_score_rbf(U_b0))
            vs_tm_l.append(_vendi_score_rbf(U_tm))
            ngi_bot_l.append(_non_gaussianity_index(U_b0, rng))
            ngi_tm_l.append(_non_gaussianity_index(U_tm, rng))
            for arr, lst in ((U_b0, b0_eff_l), (U_tm, tm_eff_l)):
                ac = arr - arr.mean(axis=0, keepdims=True)
                ev = np.maximum(np.linalg.eigvalsh((ac.T @ ac) / float(n_draws - 1)), 0.0)
                p = ev / max(float(ev.sum()), 1e-12)
                lst.append(float(np.exp(-np.sum(np.where(p > 1e-12, p * np.log(np.maximum(p, 1e-12)), 0.0)))))
            if si < 4:
                enc = model.encode_backbone(obs_s[si : si + 1], nav_s[si : si + 1], carts_s[si : si + 1], teams_s[si : si + 1], acc_s[si : si + 1], world.nav_kern, geo=geo)
                sv = np.linalg.svd(enc["B_H"][0].float().numpy(), compute_uv=False)
                pb = (sv ** 2) / max(float((sv ** 2).sum()), 1e-12)
                bh_eff_l.append(float(np.exp(-np.sum(np.where(pb > 1e-12, pb * np.log(np.maximum(pb, 1e-12)), 0.0)))))
        c_draw.close()

        # Quality-Diversity High-EV Vendi Score (QD-VS): top-25% CRN branches from first 8 canonical states!
        B_qd = min(8, n_states)
        E_qd = B_qd * qd_G
        fork_q = MetalDeltaTimePayloadEngine(world, num_envs=E_qd, num_players=N, rules=rules, base_seed=seed + 313)
        src_u8 = np.frombuffer(snap_all.data, dtype=np.uint8).reshape(n_states, snap_all.env_stride)[:B_qd]
        fork_q._state_u8[:] = np.repeat(src_u8, qd_G, axis=0)
        c_qd = CompiledGraphFlowPolicy(model, E=E_qd, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern)
        main_mask_q = torch.from_numpy(p_team_1d == 1).view(1, N).expand(E_qd, N)
        gen_q = torch.Generator(device="cpu").manual_seed(seed + 701)
        start_sc = fork_q.t_score.clone()
        u0_team = None
        for step in range(qd_H):
            ob, nv, ct, tm, ac, _ = fork_q.observe_graph_tensors()
            pt = torch.round(ob[:, :, 1].float()).to(torch.int64)
            eps = sample_orthogonal_team_bot_noise(B_qd, qd_G, N, model.d_e, pt, K, team_corr=0.45, sigma=model.cfg.sigma_base, generator=gen_q)
            res = c_qd.sample_velocity(ob, nv, ct, tm, ac, geo=geo, eps=eps)
            if step == 0:
                u0_team = res["u_vel"][:, main_idx, :].reshape(B_qd, qd_G, -1).numpy()
            fork_q.apply_vel(res["u_vel"], main_mask_q, tau_rel=2.5)
            fork_q.step(0.4, n_steps=2, tau_rel=2.5)
        d_sc = (fork_q.t_score - start_sc).numpy()
        margin_q = (d_sc[:, 0] - d_sc[:, 1:].mean(axis=1)).reshape(B_qd, qd_G)
        qd_vs_l = []
        top_k = max(3, qd_G // 4)
        assert u0_team is not None
        for b_i in range(B_qd):
            top_idx = np.argsort(margin_q[b_i])[-top_k:]
            qd_vs_l.append(_vendi_score_rbf(u0_team[b_i, top_idx]))
        c_qd.close()
        fork_q.close()

        out[arm_name] = {
            "zero_err": zero_err,
            "vendi_bot0": round(float(np.mean(vs_bot_l)), 2),
            "vendi_team": round(float(np.mean(vs_tm_l)), 2),
            "ngi_bot0": round(float(np.mean(ngi_bot_l)), 4),
            "ngi_team": round(float(np.mean(ngi_tm_l)), 4),
            "qd_vendi_top25": round(float(np.mean(qd_vs_l)), 2),
            "bot0_eff_rank": round(float(np.mean(b0_eff_l)), 2),
            "team_eff_rank": round(float(np.mean(tm_eff_l)), 2),
            "bh_eff_rank": round(float(np.mean(bh_eff_l)), 2),
            "sat_frac": round(float(np.mean(sat_l)), 4),
        }
    return out


@torch.no_grad()
def evaluate_playerwise_pvp_elo(
    models: dict[str, GraphFlowPolicy | None],
    cfg: Any,
    *,
    focal_arms: list[str],
    E_block: int = 24,
    H: int = 64,
    seed: int = 86000,
) -> dict[str, Any]:
    """Playerwise Expected PvP Elo indexed by focal-player skill stratum [-8, 0, +4, +12, +20] across 4 pools.

    For each pool (P_stock, P_league_bal, P_league_asym, P_adversarial) and each focal-player skill stratum s_p:
      - Runs paired CRN matches where Seat 0's focal player (bot 0) has skill s_p and is controlled by `arm`
        vs identical CRN baseline where the same player is controlled by `default` (stock HavocBot).
      - Computes team-outcome PvP Elo-equivalent (anchored at default = 1000) + individual unique-attribution
        combat K/D, dmg_ratio, cart_push, and whole-block bootstrap 95% CIs + p10/median/p90/worst!
    """
    t0 = time.perf_counter()
    world = build_gas_world_graph(
        map_name="pvp_elo_k6_n36", teams=6, carts=18, players=36, tiles=4, tile_size=1792.0,
        checkpoints_per_cart=4, seed=99201, device=torch.device("cpu"),
    )
    K, J, N, Wn = int(world.teams), int(world.carts), int(world.players_default), int(world.num_nav)
    A_dim = Wn + K + 4
    geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}
    rules = make_rules_for_profile(float(cfg.sim_dt.point_limit_base), "baseline")
    strata = [-8.0, 0.0, 4.0, 12.0, 20.0]

    avail_league = [a for a in models.keys() if a != "default"]
    adv_cands = [a for a in avail_league if "exp" in a] or avail_league
    pools_spec = [
        ("P_stock",       "balanced_fixed", ["default"] * (K - 1)),
        ("P_league_bal",  "balanced_fixed", [avail_league[i % len(avail_league)] for i in range(K - 1)] if avail_league else ["default"] * (K - 1)),
        ("P_league_asym", "mixed_dr",       [avail_league[i % len(avail_league)] for i in range(K - 1)] if avail_league else ["default"] * (K - 1)),
        ("P_adversarial", "mixed_dr",       [adv_cands[i % len(adv_cands)] for i in range(K - 1)] if adv_cands else ["default"] * (K - 1)),
    ]

    compiled: dict[str, CompiledGraphFlowPolicy | None] = {
        a: (CompiledGraphFlowPolicy(m, E=E_block, N=N, W_nav=Wn, J=J, K=K, nav_kern=world.nav_kern) if m is not None else None)
        for a, m in models.items()
    }
    p_team_1d = (np.arange(N, dtype=np.int32) % K) + 1
    team_masks = [torch.from_numpy(p_team_1d == (k + 1)).view(1, N).expand(E_block, N) for k in range(K)]
    t1_idx = np.nonzero(p_team_1d == 1)[0]
    focal_bot = int(t1_idx[0])

    def _run_seat(seat0_arm: str, opp_seats: list[str], skills_mat: np.ndarray, base_seed: int) -> tuple[np.ndarray, dict[str, float]]:
        eng = MetalDeltaTimePayloadEngine(world, num_envs=E_block, num_players=N, rules=rules, base_seed=base_seed)
        eng.reset(torch.ones((E_block,), dtype=torch.bool))
        eng.set_skills(skills_mat)
        seats = [seat0_arm, *opp_seats]
        arm_to_tmask: dict[str, torch.Tensor] = {}
        for ti, a in enumerate(seats):
            if a == "default":
                continue
            arm_to_tmask[a] = team_masks[ti] if a not in arm_to_tmask else (arm_to_tmask[a] | team_masks[ti])
        kills, deaths, dmg_out, dmg_in, push = 0.0, 0.0, 0.0, 0.0, 0.0
        q_ou_arm: dict[str, torch.Tensor] = {}
        obs, nav, carts, teams, acc, _ = eng.observe_graph_tensors()
        for _ in range(H):
            u_vel = torch.zeros((E_block, N, A_dim), dtype=torch.float32)
            has_vel = torch.zeros((E_block, N), dtype=torch.bool)
            for a, tmask in arm_to_tmask.items():
                cp = compiled[a]
                if cp is None:
                    continue
                rho_a = max(0.0, min(0.95, float(getattr(cp.cfg, "ou_rho", 0.0))))
                eps_in = None
                if rho_a > 0.0:
                    raw_e = torch.randn((E_block, N, int(cp.cfg.d_e)), dtype=torch.float32) * float(cp.cfg.sigma_base)
                    q_ou_arm[a] = raw_e if a not in q_ou_arm else (rho_a * q_ou_arm[a] + math.sqrt(max(0.0, 1.0 - rho_a * rho_a)) * raw_e)
                    eps_in = q_ou_arm[a]
                res = cp.sample_velocity(obs, nav, carts, teams, acc, geo=geo, eps=eps_in)
                u_vel = torch.where(tmask.unsqueeze(-1), res["u_vel"], u_vel)
                has_vel = has_vel | tmask
            eng.apply_vel(u_vel, has_vel, tau_rel=2.5)
            eng.step(float(cfg.sim_dt.dt), mode=str(cfg.sim_dt.mode), reset_outcomes=True, n_steps=2, tau_rel=2.5)
            sv = eng.state_views()
            kills += float(sv["p_kills"][:, focal_bot].sum().item())
            deaths += float(sv["p_deaths"][:, focal_bot].sum().item())
            dmg_out += float(sv["p_dmg_dealt"][:, focal_bot].sum().item())
            dmg_in += float(sv["p_dmg_taken"][:, focal_bot].sum().item())
            push += float(sv["p_cart_push"][:, focal_bot].sum().item())
            obs, nav, carts, teams, acc, _ = eng.observe_graph_tensors()
        sc = eng.t_score.numpy()
        eng.close()
        pw = np.mean(
            np.where(sc[:, 0:1] > sc[:, 1:] + 1e-6, 1.0, np.where(sc[:, 1:] > sc[:, 0:1] + 1e-6, 0.0, 0.5)),
            axis=1,
        )
        return pw, {
            "kd": round(kills / max(deaths, 1.0), 3),
            "dmg_ratio": round(dmg_out / max(dmg_in, 1.0), 3),
            "push": round(push / float(E_block), 1),
        }

    def _wr_to_elo(wr: float, n_eff: float) -> float:
        p = (wr * n_eff + 0.5) / (n_eff + 1.0)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return float(1000.0 + 400.0 * np.log10(p / (1.0 - p)))

    rng = np.random.default_rng(seed)
    # Precompute paired CRN baseline (`default` in seat 0) per (pool, stratum)
    base_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, int]] = {}
    for pi, (pname, rmode, opp_seats) in enumerate(pools_spec):
        for si, sk_val in enumerate(strata):
            bseed = seed + pi * 1000 + si * 53
            sk_mat = sample_roster_skills(np.random.default_rng(bseed + 7), E_block, N, K, rmode)
            sk_mat[:, focal_bot] = float(sk_val)
            pw_def, _ = _run_seat("default", opp_seats, sk_mat, bseed)
            base_cache[(pname, sk_val)] = (pw_def, sk_mat, bseed)

    results: dict[str, Any] = {"strata": strata, "pools": [p[0] for p in pools_spec], "arms": {}}
    try:
        for arm in focal_arms:
            if arm not in models or models[arm] is None:
                continue
            by_stratum: dict[str, Any] = {}
            by_pool: dict[str, float] = {}
            all_strata_elos: list[float] = []
            boot_blocks: list[np.ndarray] = []

            pool_acc: dict[str, list[np.ndarray]] = {p[0]: [] for p in pools_spec}
            for sk_val in strata:
                s_wins_l: list[np.ndarray] = []
                kd_l, dr_l, pu_l = [], [], []
                for pname, _, opp_seats in pools_spec:
                    pw_def, sk_mat, bseed = base_cache[(pname, sk_val)]
                    pw_arm, cstat = _run_seat(arm, opp_seats, sk_mat, bseed)
                    # Paired CRN head-to-head outcome vs default at identical seed & focal skill!
                    paired_w = np.where(pw_arm > pw_def + 1e-6, 1.0, np.where(pw_def > pw_arm + 1e-6, 0.0, 0.5))
                    s_wins_l.append(paired_w)
                    pool_acc[pname].append(paired_w)
                    boot_blocks.append(paired_w)
                    kd_l.append(cstat["kd"])
                    dr_l.append(cstat["dmg_ratio"])
                    pu_l.append(cstat["push"])
                sw = np.concatenate(s_wins_l)
                elo_s = _wr_to_elo(float(sw.mean()), float(sw.size))
                all_strata_elos.append(elo_s)
                by_stratum[f"{sk_val:+.0f}"] = {
                    "elo": round(elo_s, 1),
                    "paired_wr": round(float(sw.mean()), 4),
                    "kd": round(float(np.mean(kd_l)), 3),
                    "dmg_ratio": round(float(np.mean(dr_l)), 3),
                    "push": round(float(np.mean(pu_l)), 1),
                }
            for pname, arrs in pool_acc.items():
                pw_p = np.concatenate(arrs)
                by_pool[pname] = round(_wr_to_elo(float(pw_p.mean()), float(pw_p.size)), 1)

            # Whole-block bootstrap 95% CI on overall playerwise PvP Elo
            all_w = np.concatenate(boot_blocks)
            overall_elo = _wr_to_elo(float(all_w.mean()), float(all_w.size))
            n_blk = len(boot_blocks)
            b_elos = []
            for _ in range(150):
                b_idx = rng.integers(0, n_blk, size=n_blk)
                bw = np.concatenate([boot_blocks[int(k)] for k in b_idx])
                b_elos.append(_wr_to_elo(float(bw.mean()), float(bw.size)))
            results["arms"][arm] = {
                "overall_pvp_elo": round(overall_elo, 1),
                "ci95": [round(float(np.percentile(b_elos, 2.5)), 1), round(float(np.percentile(b_elos, 97.5)), 1)],
                "p10_elo": round(float(np.percentile(all_strata_elos, 10)), 1),
                "median_elo": round(float(np.median(all_strata_elos)), 1),
                "p90_elo": round(float(np.percentile(all_strata_elos, 90)), 1),
                "worst_stratum_elo": round(float(min(all_strata_elos)), 1),
                "by_pool": by_pool,
                "by_stratum": by_stratum,
            }
    finally:
        for cp in compiled.values():
            if cp is not None:
                cp.close()
    results["wall_s"] = round(time.perf_counter() - t0, 2)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-time behavior & multimodal diversity analyzer for GraphFlowPolicy")
    ap.add_argument("--ckpt-dir", type=str, default="")
    ap.add_argument("--champion-specs", nargs="*", default=None, help="Cross-architecture tournament entries: label=/path/to/dir:ckpt_stem ...")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--teams", type=int, default=6)
    ap.add_argument("--carts", type=int, default=16)
    ap.add_argument("--players", type=int, default=36)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--league-eval", action="store_true")
    ap.add_argument("--league-envs", type=int, default=48)
    ap.add_argument("--league-horizon", type=int, default=72)
    ap.add_argument("--pvp-elo-eval", action="store_true")
    ap.add_argument("--diversity-eval", action="store_true")
    ap.add_argument("--full-eval", action="store_true")
    ap.add_argument("--arch", type=str, default="")
    ap.add_argument("--mhc-streams", type=int, default=0)
    ap.add_argument("--ou-rho", type=float, default=-1.0)
    args = ap.parse_args()

    cfg = load_training_config(args.config)
    world = build_gas_world_graph(
        map_name=f"eval_gf_k{args.teams}_j{args.carts}_n{args.players}",
        teams=args.teams, carts=args.carts, players=args.players, tiles=args.tiles,
        seed=8101, device=torch.device("cpu"),
    )

    def _load_cfg_for_dir(d: Path) -> GraphFlowConfig:
        am_p = d / "arch_meta.json"
        if am_p.exists():
            am = json.loads(am_p.read_text())
            return make_graph_flow_config(
                args.arch or str(am.get("arch", "tied4")),
                mhc_streams=args.mhc_streams if args.mhc_streams > 0 else int(am.get("mhc_streams", 1)),
                ou_rho=args.ou_rho if args.ou_rho >= 0.0 else float(am.get("ou_rho", 0.0)),
            )
        if args.arch or args.mhc_streams > 1 or args.ou_rho >= 0.0:
            return make_graph_flow_config(
                args.arch or "tied4",
                mhc_streams=max(1, int(args.mhc_streams)),
                ou_rho=max(0.0, float(args.ou_rho)),
            )
        return GraphFlowConfig()

    models: dict[str, GraphFlowPolicy | None] = {"default": None}
    champion_mode = bool(args.champion_specs)
    if champion_mode:
        ckpt_dir = Path(args.out_json).expanduser().resolve().parent if args.out_json else Path("/tmp")
        for spec in args.champion_specs:
            # Format: label=/path/to/dir:stem
            label, rest = spec.split("=", 1)
            dir_s, stem = rest.rsplit(":", 1)
            d = Path(dir_s).expanduser().resolve()
            gfc = _load_cfg_for_dir(d)
            p = d / f"{stem}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Champion checkpoint not found: {p}")
            m = GraphFlowPolicy(gfc, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            m.eval()
            models[label] = m
    else:
        if not args.ckpt_dir:
            raise ValueError("Provide --ckpt-dir or --champion-specs")
        ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
        gf_cfg = _load_cfg_for_dir(ckpt_dir)
        cand_names = [
            "main_init", "main_p1", "exploiter_1", "main_mid", "main_p2",
            "exploiter_2", "main_final", "ema_fast_final", "ema_slow_final", "ema_final",
            "best_online", "exp_0_final", "exp_1_final", "main_snap_final",
        ]
        # Also auto-discover metric-promoted frozen exploiters (`exp_f*.pt`) & snapshots (`main_s*.pt`)
        for extra_p in sorted(ckpt_dir.glob("exp_f*.pt")) + sorted(ckpt_dir.glob("main_s*.pt")):
            if extra_p.stem not in cand_names:
                cand_names.append(extra_p.stem)
        for name in cand_names:
            p = ckpt_dir / f"{name}.pt"
            if p.exists():
                m = GraphFlowPolicy(gf_cfg, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
                m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
                m.eval()
                models[name] = m

    t0 = time.perf_counter()
    results: dict[str, Any] = {"regimes": {}, "branch_diversity": {}}
    if not champion_mode:
        beh_arms = [a for a in ("default", "main_init", "main_p1", "exploiter_1", "main_mid", "main_p2", "exploiter_2", "exp_f0", "main_final", "ema_slow_final", "best_online") if a in models]
        for ri, reg in enumerate(EVAL_REGIMES):
            results["regimes"][reg["name"]] = {}
            for ai, arm_name in enumerate(beh_arms):
                results["regimes"][reg["name"]][arm_name] = evaluate_arm_on_regime(
                    arm_name, models[arm_name], world, reg,
                    cfg=cfg, E=args.envs, H=args.horizon, seed=90000 + ri * 1000 + ai * 37,
                )

        for arm_name in ("main_init", "exploiter_1", "exploiter_2", "exp_f0", "main_final", "ema_slow_final", "best_online"):
            if arm_name in models and models[arm_name] is not None:
                results["branch_diversity"][arm_name] = evaluate_multimodal_branch_diversity(models[arm_name], world, cfg)

    if (args.league_eval or args.full_eval or champion_mode) and len(models) >= 2:
        results["league_tournament"] = evaluate_league_tournament(
            models, cfg, E_match=int(args.league_envs), H=int(args.league_horizon),
        )

    top_league: list[str] = []
    if "league_tournament" in results:
        lt_e = results["league_tournament"]["elo_overall"]
        top_league = [a for a in sorted(lt_e.keys(), key=lambda x: lt_e[x], reverse=True) if a != "default"][:3]

    if (args.diversity_eval or args.full_eval or champion_mode) and len(models) >= 2:
        if champion_mode:
            div_keys = [a for a in models if a != "default" and models[a] is not None]
        else:
            cand_div = ["main_init", "exploiter_1", "exploiter_2", "exp_f0", "exp_1_final", "main_final", "main_snap_final", "ema_fast_final", "ema_slow_final", "best_online", *top_league]
            div_keys = list(dict.fromkeys(a for a in cand_div if a in models and models[a] is not None))
        div_models = {a: models[a] for a in div_keys}
        results["effector_diversity"] = evaluate_effector_diversity_suite(div_models, world, cfg)

    if (args.pvp_elo_eval or args.full_eval or champion_mode) and len(models) >= 2:
        if champion_mode:
            pvp_focal = [a for a in models if a != "default" and models[a] is not None]
        else:
            cand_pvp = ["main_init", "exploiter_1", "exploiter_2", "exp_f0", "main_final", "main_snap_final", "ema_fast_final", "ema_slow_final", "best_online", *top_league]
            pvp_focal = list(dict.fromkeys(a for a in cand_pvp if a in models and models[a] is not None))
        results["playerwise_pvp_elo"] = evaluate_playerwise_pvp_elo(models, cfg, focal_arms=pvp_focal)

    out_path = Path(args.out_json).expanduser().resolve() if args.out_json else (ckpt_dir / "graph_flow_behavior_eval.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval-graph-flow] Done in {time.perf_counter()-t0:.2f}s (E={args.envs}, H={args.horizon}, W_nav={world.num_nav}, I_item={world.num_items}) -> {out_path}\n", flush=True)

    if results["regimes"]:
        print("=== 1. OUTCOME COMPETENCE & REALIZED GOAL-KIND MIX ACROSS 7 DR REGIMES ===")
        print(f"{'Regime':24s} | {'Arm':14s} | {'PW-WR':6s} {'1st%':6s} {'Margin':7s} {'K/D':5s} {'Push':6s} {'Pick':5s} {'Quad%':5s} | {'GoalKind [Cart/Item/Enem/Nav]%':30s}")
        print("-" * 121)
        for reg in EVAL_REGIMES:
            rn = reg["name"]
            for arm_name in beh_arms:
                o = results["regimes"][rn][arm_name]["outcomes"]
                gk = o["gk_pct"]
                print(
                    f"{rn:24s} | {arm_name:14s} | {o['pairwise_win_rate']:6.3f} {o['first_place_rate']:6.3f} "
                    f"{o['score_margin']:+7.1f} {o['kd_ratio']:5.2f} {o['cart_push']:6.1f} {o['pickups']:5.1f} {o['quad_frac']*100:5.1f} | "
                    f"{gk[0]:5.1f} / {gk[1]:5.1f} / {gk[2]:5.1f} / {gk[3]:5.1f}"
                )

    if results["regimes"] and "main_final" in models:
        print("\n=== 2. CONTINUOUS STEERING FIELD & PCA SPECTRUM ===")
        print(f"{'Regime':24s} | {'Arm':14s} | {'||u||':5s} {'Sat%':5s} {'rCart':6s} {'rItem':6s} {'rEnem':6s} {'Post':6s} | {'FocLd':6s} {'FocSt':6s} {'FocWk':6s} | {'PCA(1/3/8) EffR':18s}")
        print("-" * 126)
        show_arms = [a for a in ("main_init", "exploiter_1", "exploiter_2", "main_final", "ema_slow_final") if a in beh_arms]
        for reg in EVAL_REGIMES:
            rn = reg["name"]
            for an in show_arms:
                r = results["regimes"][rn][an]
                s = r["steering_all"]
                p = r["pca"]
                print(
                    f"{rn:24s} | {an:14s} | {s['u_norm']:5.2f} {s['sat_frac']*100:5.1f} {s['r_cart']:+6.2f} {s['r_item']:+6.2f} {s['r_enemy']:+6.2f} {s['posture']:+6.2f} | "
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
                f"  {an:14s}: zero_err={dv['zero_collapse_max_u']:.1e} | "
                f"G8(u_L2={dv['s0_branch_u_l2']:.3f} u_cos={dv['s0_branch_u_cos']:+.3f} pos_div={dv['end_pos_divergence_u']:.1f}u gk_dis={dv['end_goal_kind_disagree']:.3f} sc_std={dv['end_score_std_across_G']:.2f}) | "
                f"p(u|s0,128)(uRms={dv['s0_128_u_rms']:.2f} sat={dv['s0_128_sat_frac']*100:.2f}% spr/mu={dv['s0_128_spread_over_mean']:.2f} cos={dv['s0_128_pair_cos']:+.3f} tmEffR={dv['s0_128_eff_rank']:.1f} bot0EffR={dv['s0_bot0_eff_rank']:.1f} BH_effR={dv['bh_eff_rank']:.1f} bimod={dv['s0_bot0_bimod']:.3f})"
            )

    if "league_tournament" in results:
        lt = results["league_tournament"]
        print(f"\n=== 5. UNSEEN PARAMETRIC-WORLD LEAGUE ELO & ALPHA-RANK ({lt['total_pair_games']:,} pair-games in {lt['wall_s']:.1f}s) ===")
        print(f"{'Arm':15s} | {'Elo (95% CI)':20s} | {'Small':6s} {'Mid':6s} {'Large':6s} | {'Bal':6s} {'MixDR':6s} | {'aRank':6s}")
        print("-" * 82)
        ranked = sorted(lt["arms"], key=lambda a: lt["elo_overall"][a], reverse=True)
        for a in ranked:
            ci = lt["elo_ci95"][a]
            print(
                f"{a:15s} | {lt['elo_overall'][a]:6.1f} [{ci[0]:6.1f},{ci[1]:6.1f}] | "
                f"{lt['elo_by_scale']['small'][a]:6.1f} {lt['elo_by_scale']['mid'][a]:6.1f} {lt['elo_by_scale']['large'][a]:6.1f} | "
                f"{lt['elo_by_roster']['balanced_fixed'][a]:6.1f} {lt['elo_by_roster']['mixed_dr'][a]:6.1f} | "
                f"{lt['alpha_rank'].get(a, 0.0):6.3f}"
            )
        print("\n=== 6. PAIRWISE WIN-RATE MATRIX (Row beats Col) ===")
        hdr = f"{'Arm':15s} | " + " ".join(f"{a[:7]:>7s}" for a in ranked)
        print(hdr)
        print("-" * len(hdr))
        for a in ranked:
            cells = []
            for b in ranked:
                v = lt["pairwise_wr"][a][b]
                cells.append("    -  " if a == b or v is None else f"{v:7.3f}")
            print(f"{a:15s} | " + " ".join(cells))

    if "effector_diversity" in results:
        print("\n=== 7. EFFECTOR COINDUCTIVE VECTOR MACHINE DIVERSITY p(u | s0) (16 states x 5 PvP strata) ===")
        print(f"{'Arm':15s} | {'VS_bot0':7s} {'VS_tm':7s} {'QD-VS':7s} | {'NGI_b0':7s} {'NGI_tm':7s} | {'BH_r':5s} {'b0_r':5s} {'tm_r':5s} {'Sat%':5s}")
        print("-" * 88)
        for an, d in results["effector_diversity"].items():
            print(
                f"{an:15s} | {d['vendi_bot0']:7.2f} {d['vendi_team']:7.2f} {d['qd_vendi_top25']:7.2f} | "
                f"{d['ngi_bot0']:7.4f} {d['ngi_team']:7.4f} | "
                f"{d['bh_eff_rank']:5.1f} {d['bot0_eff_rank']:5.1f} {d['team_eff_rank']:5.1f} {d['sat_frac']*100:5.2f}"
            )

    if "playerwise_pvp_elo" in results:
        pv = results["playerwise_pvp_elo"]
        print(f"\n=== 8. PLAYERWISE EXPECTED PVP ELO INDEXED BY SKILL STRATUM ({pv['wall_s']:.1f}s, default=1000) ===")
        print(f"{'Arm':15s} | {'PvP Elo (95% CI)':20s} | {'p10/med/p90 (min)':22s} | {'-8':6s} {'0':6s} {'+4':6s} {'+12':6s} {'+20':6s} | {'Pools(Stk/Bal/Asy/Adv)'}")
        print("-" * 124)
        for an, d in pv["arms"].items():
            ci = d["ci95"]
            bs = d["by_stratum"]
            bp = d["by_pool"]
            print(
                f"{an:15s} | {d['overall_pvp_elo']:6.1f} [{ci[0]:6.1f},{ci[1]:6.1f}] | "
                f"{d['p10_elo']:5.0f}/{d['median_elo']:5.0f}/{d['p90_elo']:5.0f} ({d['worst_stratum_elo']:5.0f}) | "
                f"{bs['-8']['elo']:6.1f} {bs['+0']['elo']:6.1f} {bs['+4']['elo']:6.1f} {bs['+12']['elo']:6.1f} {bs['+20']['elo']:6.1f} | "
                f"{bp['P_stock']:.0f}/{bp['P_league_bal']:.0f}/{bp['P_league_asym']:.0f}/{bp['P_adversarial']:.0f}"
            )


if __name__ == "__main__":
    main()
