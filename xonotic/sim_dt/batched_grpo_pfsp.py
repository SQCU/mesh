#!/usr/bin/env python3
"""GPU-Resident Batched GRPO + PFSP Trainer on DeltaTimePayloadEngine
with Live Zero-Copy DarkPlaces Transfer Evaluation.

Architecture:
  1. Parametric World Sampler: draws (K, J, N, M) worlds from WorldDistributionConfig
     and compiles them directly into MPS-resident WorldGraphIR tensors in <2 ms.
  2. Mid-Match Savestate Counterfactual GRPO: warms up B base matches to contested
     mid-game states, snapshots on GPU in O(1), and forks each state into G parallel
     counterfactual branches (E = B * G simultaneous matches, optional Common Random
     Numbers) for variance-reduced within-group advantage estimation blended with
     exact Winnie/Lou GAE.
  3. Batched PFSP League: permanently retains `default`, scripted anchors
     (`script_cart_rush`, `script_leader_gank`, `script_vanguard_ambush`,
     `script_powerup_roam`), `main_init`, `ema`, and frozen periodic snapshots,
     sampling opponents proportional to `(1 - p_win)^pfsp_power`.
  4. Fast Batched sim_dt Arena: evaluates 5+ league arms across held-out parametric
     scales via Bradley-Terry Elo (bootstrap 95% CIs) and 3-team alpha-Rank.
  5. Live Zero-Copy DarkPlaces Transfer Eval: loads the exact sim_dt-trained
     PlanWally checkpoints into live unthrottled darkplaces-dedicated servers
     (`sv_headless_unthrottled=1`, `g_payload_use_view=0`) backed by POSIX shm
     MeshVFSStore on held-out gas_world_bsp maps.
"""
from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from mapgen.bsp_stream import sample_world_spec, synthesize_world_bundle
from payload.tools.strategy_io_schema import CS, OBS, TS
from sim_dt.engine import SimRulesConfig
from sim_dt.metal_runtime import (
    CompiledPlanTrainer,
    CompiledPlanWally,
    MetalDeltaTimePayloadEngine,
)
from sim_dt.world_graph import WorldGraphIR, build_gas_world_graph
from solver.strat.aurora import Aurora
from solver.strat.league_curriculum import (
    SCRIPTED_ANCHORS,
    alpha_rank_3team,
    scripted_plan_rows,
)
from solver.strat.losses import (
    LossConfig,
    RewardConfig,
)
from solver.strat.mesh_vfs import MeshVFSStore
from solver.strat.outer_validation import compute_multiteam_elo
from solver.strat.plan_policy import (
    PLAN_WIDTH,
    PlanWally,
    clone_plan_model,
    ema_update,
    pack_plan_rows,
    pack_plan_tensor,
    plan_module_group,
    project_unit_norm_,
    sample_plan,
    weight_norm_report,
)
from solver.strat.plan_runner import MultiServerPlanHarness
from solver.strat.train_config import (
    CONFIGS_DIR,
    ActionPackConfig,
    OptimizerConfig,
    TrainingConfig,
    load_training_config,
)
from solver.xonwire import CART_KIND, OBSERVATION_KIND, PLAN_KIND, TEAM_KIND, Engine, RuntimeFrames, frames


MODULE_GROUPS = ("encoder", "gram_cgt", "resnet_ffn", "action_head", "value_head", "skill_id_head", "dynamics_head")


def _pick_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    return torch.bfloat16


def _pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(requested)


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _make_aurora(model: PlanWally, opt_cfg: OptimizerConfig) -> Aurora:
    grouped: dict[str, list[torch.nn.Parameter]] = {g: [] for g in MODULE_GROUPS}
    for n, p in model.named_parameters():
        if p.requires_grad:
            grouped.setdefault(plan_module_group(n), []).append(p)
    active = [(g, ps) for g, ps in grouped.items() if ps]
    return Aurora(
        [{"params": ps, "lr": opt_cfg.lr, "scalar_lr": opt_cfg.scalar_lr} for _, ps in active],
        lr=opt_cfg.lr,
        scalar_lr=opt_cfg.scalar_lr,
        mu=opt_cfg.mu,
        nesterov=opt_cfg.nesterov,
        pp_iterations=opt_cfg.pp_iterations,
        pp_beta=opt_cfg.pp_beta,
        ns_steps=opt_cfg.ns_steps,
        eps=opt_cfg.eps,
        ns_coeffs=opt_cfg.ns_coeffs,
        d_init_clamp=opt_cfg.d_init_clamp,
        d_iter_clamp=opt_cfg.d_iter_clamp,
        scalar_beta2=opt_cfg.scalar_beta2,
        scalar_eps=opt_cfg.scalar_eps,
    )


@torch.no_grad()
def scripted_plan_tensor(
    arm_name: str,
    obs: torch.Tensor,
    carts: torch.Tensor,
    teams: torch.Tensor,
) -> torch.Tensor:
    """GPU-vectorized [E, N, 16] PLAN generator matching league_curriculum.scripted_plan_rows."""
    e, n, _ = obs.shape
    j = carts.shape[1]
    device = obs.device
    out = torch.zeros((e, n, PLAN_WIDTH), device=device, dtype=torch.float32)
    if arm_name not in SCRIPTED_ANCHORS or n == 0 or j == 0:
        return out

    p_pos = obs[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1]  # [E, N, 3]
    c_pos = carts[:, :, CS["POS_X"] : CS["POS_Z"] + 1]  # [E, J, 3]
    dists = (p_pos[:, :, None, :] - c_pos[:, None, :, :]).square().sum(dim=-1).clamp_min(1e-6).sqrt()  # [E, N, J]
    nearest_cart = dists.argmin(dim=-1)  # [E, N]

    my_team = torch.round(obs[:, :, OBS["TEAM"]]).to(torch.int64)          # [E, N]
    cart_ctrl = torch.round(carts[:, :, CS["CONTROL_TEAM"]]).to(torch.int64)  # [E, J]
    team_ids = torch.round(teams[:, :, TS["ID"]]).to(torch.int64)          # [E, K]
    lead_idx = teams[:, :, TS["SCORE"]].argmax(dim=-1)                     # [E]
    leader_team = team_ids.gather(1, lead_idx[:, None])                    # [E, 1]

    own_mask = cart_ctrl[:, None, :] == my_team[:, :, None]                # [E, N, J]
    own_dists = torch.where(own_mask, dists, torch.full_like(dists, 1e18))
    own_min_d, own_pick = own_dists.min(dim=-1)
    own_cart = torch.where(own_min_d < 1e17, own_pick, nearest_cart)

    lead_mask = (cart_ctrl[:, None, :] == leader_team[:, :, None]) & (my_team[:, :, None] != leader_team[:, :, None])
    lead_dists = torch.where(lead_mask, dists, torch.full_like(dists, 1e18))
    lead_min_d, lead_pick = lead_dists.min(dim=-1)
    lead_cart = torch.where(lead_min_d < 1e17, lead_pick, nearest_cart)

    out[:, :, 0] = obs[:, :, OBS["ID"]]

    if arm_name == "script_cart_rush":
        cpick = own_cart
        out[:, :, 1] = cpick.to(torch.float32)
        slot = cpick.clamp(0, 2) + 2
        out.scatter_(2, slot.unsqueeze(-1), torch.full((e, n, 1), 1.6, device=device, dtype=torch.float32))
        out[:, :, 5] = 1.4
        out[:, :, 6] = -0.6
        out[:, :, 14] = 0.4
        out[:, :, 15] = -0.3
    elif arm_name == "script_leader_gank":
        cpick = lead_cart
        out[:, :, 1] = cpick.to(torch.float32)
        slot = cpick.clamp(0, 2) + 2
        out.scatter_(2, slot.unsqueeze(-1), torch.full((e, n, 1), 1.2, device=device, dtype=torch.float32))
        out[:, :, 5] = 0.6
        out[:, :, 6] = 1.4
        out[:, :, 7] = torch.where(my_team != leader_team, leader_team.expand(e, n).to(torch.float32), torch.zeros((e, n), device=device))
        out[:, :, 14] = 1.2
        out[:, :, 15] = 0.7
    elif arm_name == "script_powerup_roam":
        out[:, :, 1] = nearest_cart.to(torch.float32)
        out[:, :, 5] = 0.3
        out[:, :, 6] = 0.5
        out[:, :, 8] = 1.4
        out[:, :, 9] = 1.8
        out[:, :, 15] = 0.4
    elif arm_name == "script_vanguard_ambush":
        cpick = own_cart
        out[:, :, 1] = cpick.to(torch.float32)
        slot = cpick.clamp(0, 2) + 2
        out.scatter_(2, slot.unsqueeze(-1), torch.full((e, n, 1), 1.2, device=device, dtype=torch.float32))
        out[:, :, 5] = 1.1
        out[:, :, 6] = 0.8
        even = (torch.arange(n, device=device) % 2 == 0).view(1, n)
        out[:, :, 10] = torch.where(even, torch.full((e, n), 240.0, device=device), torch.zeros((e, n), device=device))
        out[:, :, 14] = 0.8
        out[:, :, 15] = 0.5
    return out


def _team_eff_rho_batched(
    carts: torch.Tensor,
    teams: torch.Tensor,
    obs: torch.Tensor,
    partial_cp_weight: float = 0.5,
) -> torch.Tensor:
    """Batched [E, J, 18], [E, K, 7], [E, N, 84] -> [E, N] effective finish rate."""
    scores = teams[:, :, TS["SCORE"]].to(torch.float32)
    held = teams[:, :, TS["CHECKPOINTS_HELD"]].to(torch.float32)
    rate = teams[:, :, TS["CHECKPOINT_RATE"]].to(torch.float32)
    limit = torch.clamp(teams[:, :, TS["SCORE_LIMIT"]].to(torch.float32), min=1e-6)
    rem = torch.clamp(limit - scores, min=1e-6)

    path_len = torch.clamp(carts[:, :, CS["PATH_LENGTH"]].to(torch.float32), min=1e-6)
    pos_frac = torch.clamp(carts[:, :, CS["PATH_POSITION"]].to(torch.float32) / path_len, 0.0, 1.0)
    cp_count = torch.clamp(carts[:, :, CS["CHECKPOINT_COUNT"]].to(torch.float32), min=1.0)
    seg_frac = (pos_frac * cp_count) - torch.floor(pos_frac * cp_count)

    owner = (carts[:, :, CS["CONTROL_TEAM"]].to(torch.float32)[:, :, None] == teams[:, :, TS["ID"]].to(torch.float32)[:, None, :]).to(torch.float32)  # [E, J, K]
    eff_held = held + float(partial_cp_weight) * torch.einsum("ejk,ej->ek", owner, seg_frac)
    rho = torch.where(eff_held > 0.0, (rate * eff_held) / rem, torch.zeros_like(scores))  # [E, K]
    mem = (obs[:, :, OBS["TEAM"]].to(torch.float32)[:, :, None] == teams[:, :, TS["ID"]].to(torch.float32)[:, None, :]).to(torch.float32)  # [E, N, K]
    return torch.einsum("enk,ek->en", mem, rho)


def _winning_margin_batched(rho: torch.Tensor) -> torch.Tensor:
    e, k = rho.shape
    if k <= 1:
        return torch.where(rho > 0.0, torch.ones_like(rho), torch.zeros_like(rho))
    eye = torch.eye(k, device=rho.device, dtype=torch.bool).unsqueeze(0)
    other = rho[:, None, :].expand(e, k, k).masked_fill(eye, -1.0)
    best_other = other.amax(dim=2).clamp_min(0.0)
    return (rho - best_other) / (rho + best_other + 1e-6)


@torch.no_grad()
def _state_role_stats_batched(
    carts: torch.Tensor,
    teams: torch.Tensor,
    partial_cp_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (proj_winner_idx[E], has_proj[E], loser_ranks[E, K], formal_margins[E, K], eff_margins[E, K])."""
    e, k, _ = teams.shape
    device = teams.device
    scores = teams[:, :, TS["SCORE"]].to(torch.float32)
    held = teams[:, :, TS["CHECKPOINTS_HELD"]].to(torch.float32)
    rate = teams[:, :, TS["CHECKPOINT_RATE"]].to(torch.float32)
    limit = torch.clamp(teams[:, :, TS["SCORE_LIMIT"]].to(torch.float32), min=1e-6)
    rem = torch.clamp(limit - scores, min=0.0)

    # Exact finish rate & time-to-win
    rho = torch.where(held > 0.0, (rate * held) / rem.clamp_min(1e-6), torch.zeros_like(scores))  # [E, K]
    ttw = torch.where(held > 0.0, rem / (held * rate).clamp_min(1e-6), torch.where(scores >= limit, torch.zeros_like(scores), torch.full_like(scores, 1e12)))

    # Unique projected winner (strict minimum finite time-to-win)
    min_ttw = ttw.amin(dim=-1, keepdim=True)
    is_min = (ttw <= min_ttw + 1e-6) & (min_ttw < 1e11)
    n_min = is_min.sum(dim=-1)
    has_proj = n_min == 1
    proj_idx = ttw.argmin(dim=-1)  # valid where has_proj

    # Loser rank = number of teams strictly slower than team i
    loser_ranks = (ttw[:, :, None] + 1e-6 < ttw[:, None, :]).sum(dim=-1).to(torch.int64)  # [E, K]

    formal_margins = _winning_margin_batched(rho)

    # Effective finish rate with partial checkpoint credit
    path_len = torch.clamp(carts[:, :, CS["PATH_LENGTH"]].to(torch.float32), min=1e-6)
    pos_frac = torch.clamp(carts[:, :, CS["PATH_POSITION"]].to(torch.float32) / path_len, 0.0, 1.0)
    cp_count = torch.clamp(carts[:, :, CS["CHECKPOINT_COUNT"]].to(torch.float32), min=1.0)
    seg_frac = (pos_frac * cp_count) - torch.floor(pos_frac * cp_count)
    owner = (carts[:, :, CS["CONTROL_TEAM"]].to(torch.float32)[:, :, None] == teams[:, :, TS["ID"]].to(torch.float32)[:, None, :]).to(torch.float32)
    eff_held = held + float(partial_cp_weight) * torch.einsum("ejk,ej->ek", owner, seg_frac)
    eff_rho = torch.where(eff_held > 0.0, (rate * eff_held) / rem.clamp_min(1e-6), torch.zeros_like(scores))
    eff_margins = _winning_margin_batched(eff_rho)
    return proj_idx, has_proj, loser_ranks, formal_margins, eff_margins


@torch.no_grad()
def compute_step_rewards_batched(
    obs_in: torch.Tensor,
    carts_in: torch.Tensor,
    teams_in: torch.Tensor,
    obs_out: torch.Tensor,
    carts_out: torch.Tensor,
    teams_out: torch.Tensor,
    just_finished: torch.Tensor,
    winner_team_1based: torch.Tensor,
    rcfg: RewardConfig,
    terminal_win_bonus: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-vectorized exact step reward matching game_value.role_rewards + losses.compute_step_rewards.

    Returns (reward[E, N], winner_mask_in[E, N], winner_mask_out[E, N]).
    """
    e, n, _ = obs_in.shape
    k = teams_in.shape[1]
    j = carts_in.shape[1]
    device = obs_in.device

    pw_b, has_b, ranks_b, _, eff_b = _state_role_stats_batched(carts_in, teams_in, partial_cp_weight=rcfg.cart_partial_cp_weight)
    pw_a, has_a, ranks_a, marg_a, eff_a = _state_role_stats_batched(carts_out, teams_out, partial_cp_weight=rcfg.cart_partial_cp_weight)

    k_idx = torch.arange(k, device=device, dtype=torch.int64)[None, :]  # [1, K]
    is_pw_b = has_b[:, None] & (k_idx == pw_b[:, None])
    is_pw_a = has_a[:, None] & (k_idx == pw_a[:, None])
    kept_lead = has_b & has_a & (pw_a == pw_b)

    # Winner reward: boundary barrier if kept lead, else -1.0 if lost lead
    pw_marg_a = marg_a.gather(1, pw_b[:, None]).squeeze(-1)
    deficit = (float(rcfg.winner_barrier_margin) - pw_marg_a).clamp_min(0.0)
    w_rew_val = torch.where(kept_lead, -float(rcfg.winner_barrier_weight) * deficit * deficit, torch.full((e,), -1.0, device=device))
    rank_up = (ranks_a > ranks_b).to(torch.float32)
    team_rew = torch.where(is_pw_b, w_rew_val[:, None].expand(e, k), rank_up)

    # Effective-leader potential shaping
    eff_b_max, eff_b_arg = eff_b.max(dim=-1)
    eff_a_max, eff_a_arg = eff_a.max(dim=-1)
    has_lb = has_b | (eff_b_max > 0.0)
    has_la = has_a | (eff_a_max > 0.0)
    lb = torch.where(has_b, pw_b, eff_b_arg)
    la = torch.where(has_a, pw_a, eff_a_arg)
    m_lb = torch.where(has_lb, eff_b.gather(1, lb[:, None]).squeeze(-1), torch.zeros((e,), device=device))
    m_la = torch.where(has_la, eff_a.gather(1, la[:, None]).squeeze(-1), torch.zeros((e,), device=device))
    phi_b = torch.where(has_lb[:, None] & (k_idx == lb[:, None]), eff_b, -m_lb[:, None] + float(rcfg.loser_potential_self_weight) * eff_b)
    phi_a = torch.where(has_la[:, None] & (k_idx == la[:, None]), eff_a, -m_la[:, None] + float(rcfg.loser_potential_self_weight) * eff_a)
    shaping = float(rcfg.potential_shaping_weight) * (float(rcfg.potential_shaping_gamma) * phi_a - phi_b)
    team_rew = team_rew + shaping

    # Map team rewards & winner masks to players [E, N]
    p_t0 = (torch.round(obs_in[:, :, OBS["TEAM"]]).to(torch.int64) - 1).clamp(0, k - 1)
    rew = team_rew.gather(1, p_t0)
    wm_in = is_pw_b.gather(1, p_t0)
    wm_out = is_pw_a.gather(1, p_t0)

    # Scale-invariant lane progress reward
    prev_frac = carts_in[:, :, CS["PATH_POSITION"]] / carts_in[:, :, CS["PATH_LENGTH"]].clamp_min(1.0)
    cur_frac = carts_out[:, :, CS["PATH_POSITION"]] / carts_out[:, :, CS["PATH_LENGTH"]].clamp_min(1.0)
    dfrac = cur_frac - prev_frac  # [E, J]
    ctrl_t = torch.round(carts_out[:, :, CS["CONTROL_TEAM"]]).to(torch.int64)  # [E, J]
    has_ctrl = (ctrl_t > 0).to(torch.float32)
    p_t1 = torch.round(obs_in[:, :, OBS["TEAM"]]).to(torch.int64)  # [E, N]

    k_eff = max(2, k)
    j_eff = max(1, j)
    if rcfg.normalize_lane_by_scale:
        carts_per_team = max(1.0, float(j_eff) / float(k_eff))
        own_scale = 1.0 / carts_per_team
        rival_scale = -float(rcfg.rival_lane_penalty) / (max(1.0, float(k_eff - 1)) * carts_per_team)
    else:
        own_scale = 1.0
        rival_scale = -0.5 * float(rcfg.rival_lane_penalty)

    is_own = (p_t1[:, :, None] == ctrl_t[:, None, :])  # [E, N, J]
    w_lane = torch.where(is_own, torch.full((e, n, j), own_scale, device=device), torch.full((e, n, j), rival_scale, device=device))
    lane_prog = (w_lane * (dfrac * has_ctrl)[:, None, :]).sum(dim=-1)

    # Personal push / contest / pickup deltas (DeltaTimePayloadEngine resets per-step outcomes on step())
    push_delta = obs_out[:, :, OBS["CART_PUSH"]].clamp_min(0.0)
    contest_delta = obs_out[:, :, OBS["CART_CONTEST"]].clamp_min(0.0)
    pickup_delta = obs_out[:, :, OBS["PICKUPS"]].clamp_min(0.0)

    rew = (
        rew
        + float(rcfg.lane_progress_weight) * lane_prog
        + float(rcfg.push_weight) * torch.tanh(push_delta / max(float(rcfg.push_scale), 1e-6))
        + float(rcfg.contest_weight) * torch.tanh(contest_delta / max(float(rcfg.contest_scale), 1e-6))
        + float(rcfg.pickup_weight) * torch.tanh(pickup_delta / max(float(rcfg.pickup_scale), 1e-6))
    )

    # Crisp terminal match-win / loss signal on score-out transitions
    if terminal_win_bonus > 0.0:
        won = just_finished[:, None] & (p_t1 == winner_team_1based[:, None]) & (winner_team_1based[:, None] > 0)
        lost = just_finished[:, None] & (p_t1 != winner_team_1based[:, None]) & (winner_team_1based[:, None] > 0)
        rew = rew + float(terminal_win_bonus) * won.to(torch.float32) - (float(terminal_win_bonus) / max(1.0, float(k - 1))) * lost.to(torch.float32)

    return rew, wm_in, wm_out


@torch.no_grad()
def act_batched_league(
    obs: torch.Tensor,
    carts: torch.Tensor,
    teams: torch.Tensor,
    geo: dict[str, torch.Tensor],
    seat_arms: list[str],
    compiled_arms: dict[str, CompiledPlanWally],
    *,
    train_arm: str = "main",
    deterministic_eval: bool = False,
    pack_cfg: ActionPackConfig = ActionPackConfig(),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """Dispatch compiled Metal 4 policies across all K teams in E parallel matches.

    Returns (plan_tensor[E, N, 16], has_plan[E, N], train_step_data_or_None).
    """
    e, n, _ = obs.shape
    device = obs.device
    plan = torch.zeros((e, n, PLAN_WIDTH), device=device, dtype=torch.float32)
    has_plan = torch.zeros((e, n), device=device, dtype=torch.bool)
    p_team = torch.round(obs[:, :, OBS["TEAM"]]).to(torch.int64)  # [E, N], 1..K

    arm_to_teams: dict[str, list[int]] = defaultdict(list)
    for ti, arm in enumerate(seat_arms):
        arm_to_teams[arm].append(ti + 1)

    train_data: dict[str, Any] | None = None

    for arm, t_list in arm_to_teams.items():
        mask = torch.zeros((e, n), device=device, dtype=torch.bool)
        for tid in t_list:
            mask = mask | (p_team == tid)
        if arm == "default":
            continue
        if arm in SCRIPTED_ANCHORS:
            sp = scripted_plan_tensor(arm, obs, carts, teams)
            plan = torch.where(mask.unsqueeze(-1), sp, plan)
            has_plan = has_plan | mask
            continue
        cpol = compiled_arms.get(arm)
        if cpol is None:
            continue
        out = cpol.forward(obs, carts, teams, geo=geo)
        det = deterministic_eval and (arm != train_arm)
        smp = sample_plan(out, deterministic=det)
        packed = pack_plan_tensor(obs, teams, smp, active_mask=mask, pack_cfg=pack_cfg)
        plan = torch.where(mask.unsqueeze(-1), packed, plan)
        has_plan = has_plan | mask

        if arm == train_arm:
            train_data = {
                "cont": smp.cont.detach(),
                "cart_pick": smp.cart_pick.detach(),
                "rival_pick": smp.rival_pick.detach(),
                "logp": smp.logp.detach(),
                "value_winnie": out.value_winnie.detach().to(torch.float32),
                "value_lou": out.value_lou.detach().to(torch.float32),
                "actor_mask": mask,
            }

    return plan, has_plan, train_data


def _sample_pfsp_seats(
    rng: np.random.Generator,
    k_teams: int,
    pool_arms: list[str],
    win_rates: dict[str, float],
    *,
    train_arm: str = "main",
    pfsp_power: float = 2.0,
    pfsp_min_weight: float = 0.05,
    pfsp_default_wr: float = 0.45,
) -> list[str]:
    """Sample K team seats: team 1 = train_arm, teams 2..K = PFSP opponents (always 1 anchor)."""
    seats = [train_arm]
    if k_teams <= 1:
        return seats
    cands = sorted(set(pool_arms))
    weights = np.asarray(
        [max(pfsp_min_weight, 1.0 - float(win_rates.get(a, pfsp_default_wr))) ** pfsp_power for a in cands],
        dtype=np.float64,
    )
    weights /= weights.sum()
    # Always guarantee at least 1 permanent anchor seat for non-collapse grounding
    seats.append("default" if rng.random() < 0.5 else str(rng.choice(SCRIPTED_ANCHORS)))
    for _ in range(k_teams - 2):
        seats.append(str(rng.choice(cands, p=weights)))
    return seats


@torch.no_grad()
def collect_grpo_rollout_group(
    world: WorldGraphIR,
    cfg: TrainingConfig,
    arm_models: dict[str, PlanWally | None],
    seat_arms: list[str],
    *,
    rng: np.random.Generator,
    seed_base: int,
) -> dict[str, Any]:
    """Warm up B base matches to mid-game, snapshot, fork into B*G counterfactual branches,
    roll H steps in lockstep on compiled Metal 4 (libmesh-graph.dylib), and compute hybrid
    GRPO + Winnie/Lou GAE advantages in unified memory."""
    sc = cfg.sim_dt
    B = int(sc.base_envs)
    G = int(sc.group_size)
    E = B * G
    H = int(sc.rollout_horizon)
    N = int(world.players_default)
    J = int(world.carts)
    K = int(world.teams)
    rules = SimRulesConfig(point_limit_base=float(sc.point_limit_base))

    main_mdl = arm_models["main"]
    assert main_mdl is not None
    roll_dev = torch.device("cpu")

    t0 = time.perf_counter()

    # Compile active neural arms once for Metal 4 NAX GIL-free execution
    active_neural = sorted({a for a in seat_arms if arm_models.get(a) is not None})
    warm_compiled = {a: CompiledPlanWally(arm_models[a], E=B, N=N, J=J, K=K, pack_cfg=cfg.action_pack) for a in active_neural}
    grp_compiled = {a: CompiledPlanWally(arm_models[a], E=E, N=N, J=J, K=K, pack_cfg=cfg.action_pack) for a in active_neural}
    base_eng = MetalDeltaTimePayloadEngine(world, num_envs=B, num_players=N, rules=rules, base_seed=seed_base)
    grp_eng: MetalDeltaTimePayloadEngine | None = None

    try:
        # 1. Warm up B base environments to contested mid-match savestates
        warmup_steps = int(rng.integers(sc.warmup_steps_min, max(sc.warmup_steps_min + 1, sc.warmup_steps_max + 1)))
        for _ in range(warmup_steps):
            obs_w, carts_w, teams_w, geo_w = base_eng.observe_tensors()
            plan_w, hp_w, _ = act_batched_league(
                obs_w, carts_w, teams_w, geo_w, seat_arms, warm_compiled,
                train_arm="main", pack_cfg=cfg.action_pack,
            )
            base_eng.apply_plan(plan_w, hp_w)
            base_eng.step(sc.dt, mode=sc.mode)
            if bool(base_eng.t_finished.any().item()):
                base_eng.reset(base_eng.t_finished)

        # 2. O(1) GPU Savestate + Counterfactual Group Fork (B -> B * G)
        snap = base_eng.snapshot()
        grp_eng = MetalDeltaTimePayloadEngine.fork_group(
            world, snap, group_size=G, rules=rules,
            common_random_numbers=bool(sc.common_random_numbers),
            seed_stride=10007,
        )

        # Preallocate BF16 rollout buffers in unified memory [H, E, ...] (zero uncompressed FP32 activation storage)
        act_dtype = torch.bfloat16
        buf_obs = torch.empty((H, E, N, obs_w.shape[-1]), device=roll_dev, dtype=act_dtype)
        buf_carts = torch.empty((H, E, J, carts_w.shape[-1]), device=roll_dev, dtype=act_dtype)
        buf_teams = torch.empty((H, E, K, teams_w.shape[-1]), device=roll_dev, dtype=act_dtype)
        buf_obs_next = torch.empty((H, E, N, obs_w.shape[-1]), device=roll_dev, dtype=act_dtype)
        buf_carts_next = torch.empty((H, E, J, carts_w.shape[-1]), device=roll_dev, dtype=act_dtype)
        buf_teams_next = torch.empty((H, E, K, teams_w.shape[-1]), device=roll_dev, dtype=act_dtype)
        buf_cont = torch.empty((H, E, N, 11), device=roll_dev, dtype=act_dtype)
        buf_cart_pick = torch.empty((H, E, N), device=roll_dev, dtype=torch.int16)
        buf_rival_pick = torch.empty((H, E, N), device=roll_dev, dtype=torch.int16)
        buf_beh_logp = torch.empty((H, E, N), device=roll_dev, dtype=torch.float32)
        buf_reward = torch.empty((H, E, N), device=roll_dev, dtype=torch.float32)
        buf_wm_in = torch.empty((H, E, N), device=roll_dev, dtype=torch.bool)
        buf_wm_out = torch.empty((H, E, N), device=roll_dev, dtype=torch.bool)
        buf_val_w = torch.empty((H, E, N), device=roll_dev, dtype=torch.float32)
        buf_val_l = torch.empty((H, E, N), device=roll_dev, dtype=torch.float32)
        buf_alive_env = torch.empty((H, E), device=roll_dev, dtype=torch.bool)
        buf_actor_mask = torch.empty((H, E, N), device=roll_dev, dtype=torch.bool)

        geo = {"map_center": world.map_center.to(roll_dev), "map_radius": world.map_radius.to(roll_dev)}
        start_scores = grp_eng.t_score.clone()

        for t in range(H):
            was_finished = grp_eng.t_finished.clone()
            obs_in, carts_in, teams_in, _ = grp_eng.observe_tensors()
            obs_in_c = obs_in.clone()
            carts_in_c = carts_in.clone()
            teams_in_c = teams_in.clone()

            plan, hp, tdata = act_batched_league(
                obs_in_c, carts_in_c, teams_in_c, geo, seat_arms, grp_compiled,
                train_arm="main", pack_cfg=cfg.action_pack,
            )
            assert tdata is not None
            grp_eng.apply_plan(plan, hp)
            grp_eng.step(sc.dt, mode=sc.mode)
            obs_out, carts_out, teams_out, _ = grp_eng.observe_tensors()
            just_finished = (~was_finished) & grp_eng.t_finished

            rew, wm_in, wm_out = compute_step_rewards_batched(
                obs_in_c, carts_in_c, teams_in_c,
                obs_out, carts_out, teams_out,
                just_finished, grp_eng.t_winner,
                cfg.losses.reward,
                terminal_win_bonus=float(sc.terminal_win_bonus),
            )
            active_env = ~was_finished
            rew = torch.where(active_env[:, None], rew, torch.zeros_like(rew))

            buf_obs[t] = obs_in_c.to(act_dtype)
            buf_carts[t] = carts_in_c.to(act_dtype)
            buf_teams[t] = teams_in_c.to(act_dtype)
            buf_obs_next[t] = obs_out.to(act_dtype)
            buf_carts_next[t] = carts_out.to(act_dtype)
            buf_teams_next[t] = teams_out.to(act_dtype)
            buf_cont[t] = tdata["cont"].to(act_dtype)
            buf_cart_pick[t] = tdata["cart_pick"].to(torch.int16)
            buf_rival_pick[t] = tdata["rival_pick"].to(torch.int16)
            buf_beh_logp[t] = tdata["logp"]
            buf_reward[t] = rew
            buf_wm_in[t] = wm_in
            buf_wm_out[t] = wm_out
            buf_val_w[t] = tdata["value_winnie"]
            buf_val_l[t] = tdata["value_lou"]
            buf_alive_env[t] = active_env
            buf_actor_mask[t] = tdata["actor_mask"] & active_env[:, None]

        # Bootstrap value at t = H
        obs_f, carts_f, teams_f, _ = grp_eng.observe_tensors()
        out_f = grp_compiled["main"].forward(obs_f, carts_f, teams_f, geo=geo)
        end_finished = grp_eng.t_finished.clone()
        end_scores = grp_eng.t_score.clone()
    finally:
        for c in warm_compiled.values():
            c.close()
        for c in grp_compiled.values():
            c.close()
        base_eng.close()
        if grp_eng is not None:
            grp_eng.close()

    v_boot_last = torch.where(buf_wm_out[H - 1], out_f.value_winnie.to(torch.float32), out_f.value_lou.to(torch.float32))
    v_boot_last = torch.where((~end_finished)[:, None], v_boot_last, torch.zeros_like(v_boot_last))

    # 3. Exact vectorized Winnie/Lou GAE backward pass [H, E, N]
    gamma = float(cfg.losses.gamma)
    lam = float(cfg.losses.gae_lambda)
    val_in = torch.where(buf_wm_in, buf_val_w, buf_val_l)  # [H, E, N]
    gae_adv = torch.zeros((H, E, N), device=roll_dev, dtype=torch.float32)
    next_gae = torch.zeros((E, N), device=roll_dev, dtype=torch.float32)
    for t in range(H - 1, -1, -1):
        still_going = buf_alive_env[t][:, None] & (buf_alive_env[t + 1][:, None] if t + 1 < H else (~end_finished)[:, None])
        same_role = buf_wm_in[t] == buf_wm_out[t]
        v_nxt = val_in[t + 1] if t + 1 < H else v_boot_last
        v_nxt = torch.where(still_going & same_role, v_nxt, torch.zeros_like(v_nxt))
        g_nxt = torch.where(still_going & same_role, next_gae, torch.zeros_like(next_gae))
        delta = buf_reward[t] + gamma * v_nxt - val_in[t]
        cur_gae = torch.where(buf_alive_env[t][:, None], delta + gamma * lam * g_nxt, torch.zeros_like(delta))
        gae_adv[t] = cur_gae
        next_gae = cur_gae
    gae_ret = gae_adv + val_in  # [H, E, N]

    # 4. Exact Counterfactual Within-Group GRPO Advantage [B, G] over pure CGT sparse W/L return
    disc_pows = (gamma ** torch.arange(H, device=roll_dev, dtype=torch.float32)).view(H, 1, 1)
    main_mask_0 = buf_actor_mask[0].to(torch.float32)
    main_cnt = main_mask_0.sum(dim=-1).clamp_min(1.0)
    ep_ret_agent = (buf_reward * disc_pows).sum(dim=0) + (gamma ** H) * v_boot_last
    ep_ret_team = (ep_ret_agent * main_mask_0).sum(dim=-1) / main_cnt
    d_scores = end_scores - start_scores

    grp_ret = ep_ret_team.view(B, G)
    grp_mean = grp_ret.mean(dim=1, keepdim=True)
    grp_std = grp_ret.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
    grpo_adv_bg = ((grp_ret - grp_mean) / grp_std).view(E)
    grpo_adv = grpo_adv_bg.view(1, E, 1).expand(H, E, N)

    # Normalize GAE across active main-team rows and blend with GRPO
    am_f = buf_actor_mask.to(torch.float32)
    am_sum = am_f.sum().clamp_min(1.0)
    gae_mean = (gae_adv * am_f).sum() / am_sum
    gae_var = (((gae_adv - gae_mean) ** 2) * am_f).sum() / am_sum
    gae_norm = (gae_adv - gae_mean) / gae_var.clamp_min(1e-6).sqrt()
    blended_adv = float(sc.grpo_weight) * grpo_adv + float(sc.gae_weight) * gae_norm

    score_outs_n = int(end_finished.sum().item())

    # Pairwise win-rates vs each opponent team over the rollout window for PFSP
    opp_wr_obs: dict[str, list[float]] = defaultdict(list)
    d_sc_cpu = d_scores.detach().cpu().numpy()
    for ti in range(1, K):
        opp_arm = seat_arms[ti]
        wins = np.where(d_sc_cpu[:, 0] > d_sc_cpu[:, ti] + 1e-6, 1.0, np.where(d_sc_cpu[:, ti] > d_sc_cpu[:, 0] + 1e-6, 0.0, 0.5))
        opp_wr_obs[opp_arm].append(float(wins.mean()))

    wall_s = time.perf_counter() - t0
    total_sim_s = (B * warmup_steps + E * H) * float(sc.dt)
    return {
        "obs": buf_obs,
        "carts": buf_carts,
        "teams": buf_teams,
        "obs_next": buf_obs_next,
        "carts_next": buf_carts_next,
        "teams_next": buf_teams_next,
        "cont": buf_cont,
        "cart_pick": buf_cart_pick,
        "rival_pick": buf_rival_pick,
        "beh_logp": buf_beh_logp,
        "wm_in": buf_wm_in,
        "actor_mask": buf_actor_mask,
        "adv": blended_adv,
        "gae_ret": gae_ret,
        "geo": geo,
        "opp_wr_obs": {a: float(np.mean(vs)) for a, vs in opp_wr_obs.items()},
        "stats": {
            "wall_s": round(wall_s, 3),
            "sim_s": round(total_sim_s, 1),
            "speedup_x": round(total_sim_s / max(wall_s, 1e-6), 1),
            "warmup_steps": warmup_steps,
            "score_outs": score_outs_n,
            "grpo_ret_mean": round(float(grp_ret.mean().item()), 4),
            "grpo_within_std": round(float(grp_ret.std(dim=1, unbiased=False).mean().item()), 4),
            "main_dscore_mean": round(float(d_scores[:, 0].mean().item()), 2),
            "opp_dscore_mean": round(float(d_scores[:, 1:].mean().item()), 2) if K > 1 else 0.0,
        },
    }


def update_batched_grpo_ppo(
    model: PlanWally,
    ema_model: PlanWally,
    optimizer: Aurora,
    rollout: dict[str, Any],
    *,
    cfg: TrainingConfig,
    entropy_coef: float,
    device: torch.device,
) -> dict[str, Any]:
    """Compiled Metal 4 NAX BF16 forward + loss + analytical backward + FP32 grad accum + Aurora."""
    t0 = time.perf_counter()
    lcfg: LossConfig = cfg.losses
    sc = cfg.sim_dt

    H, E, N, _ = rollout["obs"].shape
    J = rollout["carts"].shape[2]
    K = rollout["teams"].shape[2]
    M_tot = H * E
    entities_per_env = N + J + K

    obs_f = rollout["obs"].reshape(M_tot, N, -1)
    carts_f = rollout["carts"].reshape(M_tot, J, -1)
    teams_f = rollout["teams"].reshape(M_tot, K, -1)
    obs_nf = rollout["obs_next"].reshape(M_tot, N, -1)
    carts_nf = rollout["carts_next"].reshape(M_tot, J, -1)
    teams_nf = rollout["teams_next"].reshape(M_tot, K, -1)
    cont_f = rollout["cont"].reshape(M_tot, N, -1)
    cp_f = rollout["cart_pick"].reshape(M_tot, N)
    rp_f = rollout["rival_pick"].reshape(M_tot, N)
    blp_f = rollout["beh_logp"].reshape(M_tot, N)
    wm_f = rollout["wm_in"].reshape(M_tot, N)
    am_f = rollout["actor_mask"].reshape(M_tot, N).to(torch.float32)
    adv_f = rollout["adv"].reshape(M_tot, N).clamp(-lcfg.adv_clip, lcfg.adv_clip)
    ret_f = rollout["gae_ret"].reshape(M_tot, N)
    geo = rollout["geo"]

    with torch.no_grad():
        rho_c = _team_eff_rho_batched(carts_f, teams_f, obs_f, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
        rho_n = _team_eff_rho_batched(carts_nf, teams_nf, obs_nf, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
        obs_drho_f = torch.asinh(lcfg.dyn_drho_scale * (rho_n - rho_c))

    # Roofline-derived microbatch: keep entity-rows per forward inside cache-resident target window,
    # decoupled from statistical optimizer step count via FP32 GPU gradient accumulation.
    target_rows = max(256, int(sc.target_microbatch_rows))
    mb_envs = max(2, min(M_tot, int(round(target_rows / max(1, entities_per_env)))))
    n_micros = max(1, (M_tot + mb_envs - 1) // mb_envs)
    target_steps = max(1, int(sc.target_opt_steps_per_epoch))
    accum = max(1, int(round(n_micros / float(target_steps))))
    diags: list[dict[str, float]] = []
    opt_steps = 0

    trainer = CompiledPlanTrainer(model, E_mb=mb_envs, N=N, J=J, K=K)
    trainer.set_geo(geo)
    ls_lo, ls_hi = model.model_cfg.cont_logstd_bounds
    try:
        for _ in range(max(1, int(sc.ppo_epochs))):
            perm = torch.randperm(M_tot)
            optimizer.zero_grad(set_to_none=True)
            trainer.zero_grads_()
            in_accum = 0
            for start in range(0, M_tot, mb_envs):
                idx = perm[start : start + mb_envs]
                if idx.numel() < 2:
                    continue
                step_diag = trainer.microbatch_fwd_bwd(
                    obs_f[idx], carts_f[idx], teams_f[idx],
                    cont_f[idx], cp_f[idx], rp_f[idx],
                    blp_f[idx], adv_f[idx], ret_f[idx], obs_drho_f[idx],
                    am_f[idx], wm_f[idx],
                    lcfg=lcfg,
                    entropy_coef=float(entropy_coef),
                    grad_scale=1.0 / float(accum),
                )
                in_accum += 1

                is_last = (start + mb_envs) >= M_tot
                if in_accum >= accum or is_last:
                    trainer.export_grads_to_model_(model)
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip).item())
                    optimizer.step()
                    project_unit_norm_(model, eps=cfg.norm.unit_norm_eps)
                    ema_update(ema_model, model, decay=cfg.optimizer.ema_decay)
                    optimizer.zero_grad(set_to_none=True)
                    trainer.sync_weights_(model)
                    trainer.zero_grads_()
                    in_accum = 0
                    opt_steps += 1
                    with torch.no_grad():
                        diags.append({
                            "loss_total": float(step_diag["loss_total"]),
                            "loss_actor": float(step_diag["loss_actor"]),
                            "loss_critic": float(step_diag["loss_w"] + step_diag["loss_l"]),
                            "loss_dyn": float(step_diag["loss_dyn"]),
                            "entropy": float(step_diag["entropy"]),
                            "entropy_disc": float(step_diag["entropy_disc"]),
                            "clip_frac": float(step_diag["clip_frac"]),
                            "grad_norm": gn,
                            "cont_std": float(torch.exp(model.cont_logstd.detach().to(torch.float32).clamp(ls_lo, ls_hi)).mean().item()),
                        })
    finally:
        trainer.close()

    _sync_device(device)
    def _mean(k: str) -> float:
        return round(float(np.mean([d[k] for d in diags])), 4) if diags else 0.0

    return {
        "update_wall_s": round(time.perf_counter() - t0, 3),
        "opt_steps": opt_steps,
        "mb_envs": mb_envs,
        "mb_rows": mb_envs * entities_per_env,
        "accum_steps": accum,
        "frames_per_epoch": M_tot,
        "loss_total": _mean("loss_total"),
        "loss_actor": _mean("loss_actor"),
        "loss_critic": _mean("loss_critic"),
        "loss_dyn": _mean("loss_dyn"),
        "entropy": _mean("entropy"),
        "entropy_disc": _mean("entropy_disc"),
        "clip_frac": _mean("clip_frac"),
        "grad_norm": _mean("grad_norm"),
        "cont_std": _mean("cont_std"),
    }


@torch.no_grad()
def eval_sim_dt_league(
    arm_models: dict[str, PlanWally | None],
    eval_arms: list[str],
    *,
    cfg: TrainingConfig,
    device: torch.device,
    audit_model: PlanWally | None = None,
) -> dict[str, Any]:
    """Compiled Metal 4 sim_dt multi-scale Elo + 3-team alpha-Rank arena."""
    t0 = time.perf_counter()
    sc = cfg.sim_dt
    eval_worlds_spec = [
        {"name": "eval_k4_j12_n32", "seed": 7101, "teams": 4, "carts": 12, "players": 32, "tiles": 3},
        {"name": "eval_k6_j20_n48", "seed": 7102, "teams": 6, "carts": 20, "players": 48, "tiles": 4},
        {"name": "eval_k8_j28_n64", "seed": 7103, "teams": 8, "carts": 28, "players": 64, "tiles": 6},
    ]
    base_triplets = [
        ("default", "script_cart_rush", "main_final"),
        ("default", "main_init", "main_final"),
        ("default", "main_mid", "main_final"),
        ("script_cart_rush", "main_init", "main_mid"),
        ("script_cart_rush", "main_mid", "main_final"),
        ("default", "script_leader_gank", "main_init"),
        ("default", "script_leader_gank", "main_final"),
        ("script_leader_gank", "main_mid", "main_final"),
    ]
    triplets = [tr for tr in base_triplets if all(a in eval_arms for a in tr)]
    if not triplets and len(eval_arms) >= 2:
        triplets = [tuple(eval_arms[: min(3, len(eval_arms))])]

    matches: list[dict[str, float]] = []
    per_world_summary: list[dict[str, Any]] = []
    n_envs = max(8, int(sc.eval_matches_per_pair))
    horizon = int(sc.eval_horizon)
    rules = SimRulesConfig(point_limit_base=float(sc.point_limit_base))
    total_sim_s = 0.0
    neural_arms = sorted({a for tr in triplets for a in tr if arm_models.get(a) is not None})

    for ws in eval_worlds_spec:
        world = build_gas_world_graph(
            map_name=ws["name"], teams=ws["teams"], carts=ws["carts"],
            players=ws["players"], tiles=ws["tiles"], seed=ws["seed"], device=torch.device("cpu"),
        )
        geo = {"map_center": world.map_center, "map_radius": world.map_radius}
        K = int(world.teams)
        J = int(world.carts)
        N = int(ws["players"])
        score_outs_w = 0
        compiled_w = {a: CompiledPlanWally(arm_models[a], E=n_envs, N=N, J=J, K=K, pack_cfg=cfg.action_pack) for a in neural_arms}
        eng = MetalDeltaTimePayloadEngine(world, num_envs=n_envs, num_players=N, rules=rules, base_seed=ws["seed"] * 100)
        try:
            for ti, trip in enumerate(triplets):
                seat_arms = [trip[(k_i + ti) % len(trip)] for k_i in range(K)]
                eng.reset(base_seed=ws["seed"] * 100 + ti * 17)
                for _ in range(horizon):
                    obs, carts, teams, _ = eng.observe_tensors()
                    plan, hp, _ = act_batched_league(
                        obs, carts, teams, geo, seat_arms, compiled_w,
                        train_arm="", deterministic_eval=True, pack_cfg=cfg.action_pack,
                    )
                    eng.apply_plan(plan, hp)
                    eng.step(sc.dt, mode=sc.mode)
                total_sim_s += n_envs * horizon * float(sc.dt)
                score_outs_w += int(eng.t_finished.sum().item())
                sc_np = eng.t_score.detach().cpu().numpy()  # [E, K]
                for ei in range(n_envs):
                    by_arm: dict[str, list[float]] = defaultdict(list)
                    for k_i, arm in enumerate(seat_arms):
                        by_arm[arm].append(float(sc_np[ei, k_i]))
                    matches.append({a: float(np.mean(vs)) for a, vs in by_arm.items()})
        finally:
            for c in compiled_w.values():
                c.close()
            eng.close()
        per_world_summary.append({
            "world": ws["name"], "K": ws["teams"], "J": ws["carts"], "N": ws["players"],
            "batches": len(triplets), "matches": len(triplets) * n_envs, "score_outs": score_outs_w,
        })

    wall_s = time.perf_counter() - t0
    elo = compute_multiteam_elo(matches, anchor_arm="default", bootstrap_n=250)
    ar = alpha_rank_3team(eval_arms, matches, alpha=cfg.league.alpha_rank_alpha, m_pop=cfg.league.alpha_rank_m_pop)

    return {
        "wall_s": round(wall_s, 3),
        "sim_s": round(total_sim_s, 1),
        "speedup_x": round(total_sim_s / max(wall_s, 1e-6), 1),
        "n_matches": len(matches),
        "worlds": per_world_summary,
        "elo": elo,
        "alpha_rank": ar,
        "dead_neuron_fraction": None,
    }


def eval_darkplaces_transfer(
    arm_models: dict[str, PlanWally | None],
    eval_arms: list[str],
    *,
    cfg: TrainingConfig,
    device: torch.device,
    work_dir: Path,
) -> dict[str, Any]:
    """Live zero-copy DarkPlaces transfer evaluation using sv_headless_unthrottled=1, g_payload_use_view=0."""
    t0 = time.perf_counter()
    sc = cfg.sim_dt
    work_dir.mkdir(parents=True, exist_ok=True)
    vfs = MeshVFSStore(
        name=f"mesh_simdt_xfer_{int(time.time()) % 10000}",
        size_mb=128,
        capacity=1024,
    )
    harness = MultiServerPlanHarness.from_config(cfg.harness, vfs_store=vfs)
    if not Path(harness.dp_bin).exists():
        vfs.close()
        return {"error": f"darkplaces binary not found at {harness.dp_bin}"}

    # Held-out parametric gas_world_bsp worlds (tiles >= 3) streamed zero-copy over POSIX shm
    dp_specs = [
        {"seed": 9102, "family": "gas_world_bsp", "teams": 4, "carts": 12, "players": 32, "tiles": 3},
        {"seed": 9104, "family": "gas_world_bsp", "teams": 6, "carts": 18, "players": 36, "tiles": 4},
    ]
    n_ep = max(1, int(sc.darkplaces_eval_episodes))
    dp_specs = dp_specs[:n_ep]

    triplets = [
        ("default", "script_cart_rush", "main_final"),
        ("default", "main_init", "main_final"),
        ("default", "main_mid", "main_final"),
    ]
    triplets = [tr for tr in triplets if all(a in eval_arms for a in tr)]

    matches: list[dict[str, float]] = []
    per_episode: list[dict[str, Any]] = []
    total_sim_s = 0.0
    total_wall_s = 0.0
    base_port = 26600

    compiled_dp_cache: dict[tuple[str, int, int, int], CompiledPlanWally] = {}
    try:
        for si, sp in enumerate(dp_specs):
            spec_out, vfs_files = synthesize_world_bundle(sp, cfg.world, stitcher=None, name_prefix="simdt_xfer")
            vfs.clear()
            vfs.put_many(vfs_files)
            mapname = str(spec_out["name"])
            K = int(spec_out["teams"])
            N = int(spec_out["players"])
            maxc = min(255, N + 16)

            for ti, trip in enumerate(triplets):
                for rot in range(len(trip)):
                    seat_arms = [trip[(k_i + rot) % len(trip)] for k_i in range(K)]
                    ep_dir = work_dir / f"dp_s{si}_t{ti}_r{rot}"
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    engine = Engine("udp://127.0.0.1:0", harness.framebytes)
                    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
                    procs, logs = harness._spawn_servers(
                        ep_dir, engine.path, [sp["seed"] + ti * 31 + rot * 7], base_port + si * 40 + ti * 6 + rot,
                        map_names=[mapname],
                        skill=cfg.harness.default_skill,
                        teams_per_server=[K],
                        bots_per_server=[N],
                        maxclients_per_server=[maxc],
                        meta_per_server=[spec_out],
                        point_limit=float(sc.darkplaces_point_limit),
                    )
                    routes: dict[int, tuple[int, object]] = {}
                    last_scores: dict[int, float] = {}
                    last_sim_t = 0.0
                    finished = False
                    ep_t0 = time.monotonic()
                    max_wall = 90.0

                    try:
                        while not finished and (time.monotonic() - ep_t0) < max_wall:
                            if all(p.poll() is not None for p in procs):
                                break
                            readable, _, _ = select.select([engine], [], [], 0.05)
                            if not readable:
                                continue
                            for node, address, buf in engine.receive():
                                s = inbox.feed(buf)
                                if s is not None:
                                    routes[s] = (node, address)
                            while (ready := inbox.take()) is not None:
                                (session, tick_id), records, _ = ready
                                obs_raw = records[OBSERVATION_KIND][1]
                                cart_rows = records[CART_KIND][1]
                                team_rows = records[TEAM_KIND][1]
                                obs_rows = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
                                if len(obs_rows) == 0 or len(cart_rows) == 0 or len(team_rows) == 0:
                                    continue
                                last_sim_t = max(last_sim_t, float(obs_rows[:, OBS["ENGINE_TIME"]].max()))
                                for r in team_rows:
                                    last_scores[int(round(float(r[TS["ID"]])))] = float(r[TS["SCORE"]])
                                if bool(np.any(team_rows[:, TS["FINISHED"]] > 0.5)):
                                    finished = True

                                sv_maxc_obs = max(maxc, len(obs_raw))
                                bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                                bot_arms = np.asarray([seat_arms[(int(t) - 1) % len(seat_arms)] for t in bot_teams], dtype=object)
                                plan_mat = np.zeros((sv_maxc_obs, PLAN_WIDTH), dtype=np.float32)

                                obs_t = torch.as_tensor(obs_rows, dtype=torch.bfloat16).unsqueeze(0)
                                carts_t = torch.as_tensor(cart_rows, dtype=torch.bfloat16).unsqueeze(0)
                                teams_t = torch.as_tensor(team_rows, dtype=torch.bfloat16).unsqueeze(0)
                                all_xy = torch.cat((obs_t[0, :, OBS["POS_X"] : OBS["POS_Z"] + 1].float(), carts_t[0, :, CS["POS_X"] : CS["POS_Z"] + 1].float()), dim=0)
                                lo_p = all_xy.amin(dim=0)
                                hi_p = all_xy.amax(dim=0)
                                geo = {
                                    "map_center": 0.5 * (lo_p + hi_p),
                                    "map_radius": ((hi_p - lo_p).norm() * 0.5).clamp_min(128.0),
                                }

                                n_cur, j_cur, k_cur = len(obs_rows), len(cart_rows), len(team_rows)
                                for arm in set(seat_arms):
                                    if arm == "default":
                                        continue
                                    if arm in SCRIPTED_ANCHORS:
                                        sp_mat = scripted_plan_rows(arm, obs_rows, cart_rows, team_rows, bot_arms, maxclients=sv_maxc_obs)
                                        m_nz = sp_mat[:, 0] > 0.5
                                        plan_mat[m_nz] = sp_mat[m_nz]
                                        continue
                                    mdl = arm_models.get(arm)
                                    if mdl is None:
                                        continue
                                    ckey = (arm, n_cur, j_cur, k_cur)
                                    cpol = compiled_dp_cache.get(ckey)
                                    if cpol is None:
                                        cpol = CompiledPlanWally(mdl, E=1, N=n_cur, J=j_cur, K=k_cur, pack_cfg=cfg.action_pack)
                                        compiled_dp_cache[ckey] = cpol
                                    with torch.no_grad():
                                        out = cpol.forward(obs_t, carts_t, teams_t, geo=geo)
                                        out_1 = type(out)(
                                            cont_mean=out.cont_mean[0],
                                            cont_logstd=out.cont_logstd[0],
                                            cart_logits=out.cart_logits[0],
                                            rival_logits=out.rival_logits[0],
                                            cell_logits=None,
                                            value_winnie=out.value_winnie[0],
                                            value_lou=out.value_lou[0],
                                            dyn_drift=out.dyn_drift[0],
                                            dyn_action=out.dyn_action[0],
                                        )
                                        smp = sample_plan(out_1, deterministic=True)
                                    packed = pack_plan_rows(obs_rows, team_rows, smp, bot_arms, {arm}, maxclients=sv_maxc_obs, pack_cfg=cfg.action_pack)
                                    m_nz = packed[:, 0] > 0.5
                                    plan_mat[m_nz] = packed[m_nz]

                                if session in routes:
                                    node, addr = routes[session]
                                    req_id = records[OBSERVATION_KIND][0]["req_id"]
                                    engine.send(addr, node, frames(PLAN_KIND, req_id, tick_id, plan_mat, harness.framebytes, session))
                    finally:
                        harness._stop_servers(procs, logs)
                        engine.close()

                    ep_wall = time.monotonic() - ep_t0
                    total_wall_s += ep_wall
                    total_sim_s += last_sim_t
                    by_arm: dict[str, list[float]] = defaultdict(list)
                    for k_i, arm in enumerate(seat_arms):
                        by_arm[arm].append(float(last_scores.get(k_i + 1, 0.0)))
                    m_rec = {a: float(np.mean(vs)) for a, vs in by_arm.items()}
                    if last_sim_t > 1.0:
                        matches.append(m_rec)
                    per_episode.append({
                        "map": mapname, "K": K, "N": N, "triplet": list(trip), "rot": rot,
                        "finished": finished, "sim_s": round(last_sim_t, 2),
                        "wall_s": round(ep_wall, 2),
                        "speedup_x": round(last_sim_t / max(ep_wall, 1e-3), 2),
                        "scores": {a: round(v, 2) for a, v in m_rec.items()},
                    })
    finally:
        for cpol in compiled_dp_cache.values():
            cpol.close()
        vfs.close()

    elo = compute_multiteam_elo(matches, anchor_arm="default", bootstrap_n=250) if matches else {}
    ar = alpha_rank_3team(eval_arms, matches, alpha=cfg.league.alpha_rank_alpha, m_pop=cfg.league.alpha_rank_m_pop) if matches else {}
    return {
        "wall_s": round(time.perf_counter() - t0, 2),
        "total_sim_s": round(total_sim_s, 2),
        "mean_dp_speedup_x": round(total_sim_s / max(total_wall_s, 1e-3), 2),
        "n_matches": len(matches),
        "score_outs": sum(1 for e in per_episode if e["finished"]),
        "episodes": per_episode,
        "elo": elo,
        "alpha_rank": ar,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU-resident batched GRPO + PFSP trainer on DeltaTimePayloadEngine")
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=str, default="/tmp/sim_dt_grpo_pfsp_run")
    ap.add_argument("--iterations", type=int, default=0, help="Override config iterations if > 0")
    ap.add_argument("--base-envs", type=int, default=0, help="Override config base_envs if > 0")
    ap.add_argument("--group-size", type=int, default=0, help="Override config group_size if > 0")
    ap.add_argument("--skip-dp-eval", action="store_true", help="Skip live DarkPlaces transfer evaluation")
    ap.add_argument("--eval-only-dir", type=str, default="", help="Load main_init/mid/final.pt from dir and run DarkPlaces + sim_dt eval only")
    args = ap.parse_args()

    overrides: dict[str, Any] = {}
    if args.iterations > 0 or args.base_envs > 0 or args.group_size > 0 or args.skip_dp_eval:
        sd: dict[str, Any] = {}
        if args.iterations > 0:
            sd["iterations"] = args.iterations
            sd["mid_snapshot_iter"] = max(1, args.iterations // 2)
        if args.base_envs > 0:
            sd["base_envs"] = args.base_envs
        if args.group_size > 0:
            sd["group_size"] = args.group_size
        if args.skip_dp_eval:
            sd["darkplaces_transfer_eval"] = False
        overrides["sim_dt"] = sd

    cfg = load_training_config(args.config, overrides=overrides or None)
    sc = cfg.sim_dt
    device = _pick_device(args.device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seeds.model_init)
    np.random.seed(cfg.seeds.model_init)
    rng = np.random.default_rng(cfg.seeds.schedule_base)

    model_dtype = _pick_dtype(sc.dtype)
    if args.eval_only_dir:
        ckpt_dir = Path(args.eval_only_dir).expanduser().resolve()
        eval_models: dict[str, PlanWally | None] = {}
        for arm in ("main_init", "main_mid", "main_final"):
            p = ckpt_dir / f"{arm}.pt"
            if p.exists():
                m_i = PlanWally(model_cfg=cfg.model, norm_cfg=cfg.norm).to(device=device, dtype=model_dtype)
                m_i.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                m_i.eval()
                eval_models[arm] = m_i
        dp_arms = ["default", "script_cart_rush", *[a for a in ("main_init", "main_mid", "main_final") if a in eval_models]]
        print(f"[sim_dt-grpo] eval-only DarkPlaces transfer eval from {ckpt_dir} arms={dp_arms}", flush=True)
        dp_eval = eval_darkplaces_transfer(eval_models, dp_arms, cfg=cfg, device=device, work_dir=ckpt_dir / "dp_transfer_eval")
        if "elo" in dp_eval and "ratings" in dp_eval["elo"]:
            print(
                f"[sim_dt-grpo] DarkPlaces Transfer Eval ({dp_eval['score_outs']}/{dp_eval['n_matches']} score-outs, "
                f"mean {dp_eval['mean_dp_speedup_x']}x realtime, {dp_eval['wall_s']}s wall):",
                flush=True,
            )
            for arm in dp_arms:
                r = dp_eval["elo"]["ratings"].get(arm, {})
                print(
                    f"  {arm:20s}: Elo={r.get('elo', 0):6.1f} (CI95={r.get('ci95')}) "
                    f"alpha-Rank={dp_eval['alpha_rank'].get(arm, 0):.4f} mean_score={r.get('mean_score', 0):.1f}",
                    flush=True,
                )
        rep_p = ckpt_dir / "report.json"
        if rep_p.exists():
            rep = json.loads(rep_p.read_text())
            rep["darkplaces_transfer_eval"] = dp_eval
            rep_p.write_text(json.dumps(rep, indent=2))
            print(f"[sim_dt-grpo] updated -> {rep_p}", flush=True)
        else:
            (ckpt_dir / "dp_transfer_report.json").write_text(json.dumps(dp_eval, indent=2))
        return

    model = PlanWally(model_cfg=cfg.model, norm_cfg=cfg.norm).to(device=device, dtype=model_dtype)
    ema_model = clone_plan_model(model).to(device=device, dtype=model_dtype)
    main_init = clone_plan_model(model).to(device=device, dtype=model_dtype)
    main_mid: PlanWally | None = None
    optimizer = _make_aurora(model, cfg.optimizer)

    torch.save(main_init.state_dict(), out_dir / "main_init.pt")

    arm_models: dict[str, PlanWally | None] = {
        "main": model,
        "ema": ema_model,
        "main_init": main_init,
    }
    pfsp_pool = ["default", *SCRIPTED_ANCHORS, "main_init", "ema"]
    pfsp_wr: dict[str, float] = {a: float(cfg.league.pfsp_default_winrate) for a in pfsp_pool}

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[sim_dt-grpo] runtime=metal4-compiled (libmesh-graph.dylib) opt_device={device} dtype={model_dtype} params={n_params:,} ({n_params * 2 / 1e6:.2f} MB) "
        f"iters={sc.iterations} B={sc.base_envs} G={sc.group_size} (E={sc.base_envs * sc.group_size}) "
        f"H={sc.rollout_horizon} dt={sc.dt}s target_mb_rows={sc.target_microbatch_rows} crn={sc.common_random_numbers}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    t_train_start = time.perf_counter()
    total_sim_s = 0.0
    total_rollout_s = 0.0
    total_update_s = 0.0

    for it in range(1, sc.iterations + 1):
        spec = sample_world_spec(cfg.seeds.match_base + it * 101, cfg.world)
        map_name = f"simdt_gas_it{it}_k{spec['teams']}_j{spec['carts']}_n{spec['players']}"
        world = build_gas_world_graph(
            map_name=map_name,
            teams=int(spec["teams"]),
            carts=int(spec["carts"]),
            players=int(spec["players"]),
            tiles=int(spec["tiles"]),
            seed=int(spec["seed"]),
            device=torch.device("cpu"),
        )
        seats = _sample_pfsp_seats(
            rng, int(spec["teams"]), pfsp_pool, pfsp_wr,
            train_arm="main",
            pfsp_power=cfg.league.pfsp_power,
            pfsp_min_weight=cfg.league.pfsp_min_weight,
            pfsp_default_wr=cfg.league.pfsp_default_winrate,
        )
        ent_coef = cfg.losses.entropy_for_iteration("main", it, sc.iterations)

        rollout = collect_grpo_rollout_group(
            world, cfg, arm_models, seats, rng=rng, seed_base=cfg.seeds.match_base + it * 1009,
        )
        upd = update_batched_grpo_ppo(
            model, ema_model, optimizer, rollout, cfg=cfg, entropy_coef=ent_coef, device=device,
        )

        # Update PFSP win-rate estimates (EMA)
        alpha_ema = float(sc.pfsp_update_ema)
        for opp_arm, wr_val in rollout["opp_wr_obs"].items():
            prev = pfsp_wr.get(opp_arm, float(cfg.league.pfsp_default_winrate))
            pfsp_wr[opp_arm] = alpha_ema * prev + (1.0 - alpha_ema) * float(wr_val)

        if it == sc.mid_snapshot_iter:
            main_mid = clone_plan_model(model).to(device)
            arm_models["main_mid"] = main_mid
            pfsp_pool.append("main_mid")
            pfsp_wr["main_mid"] = 0.5
            torch.save(main_mid.state_dict(), out_dir / "main_mid.pt")

        total_sim_s += float(rollout["stats"]["sim_s"])
        total_rollout_s += float(rollout["stats"]["wall_s"])
        total_update_s += float(upd["update_wall_s"])

        rec = {
            "iter": it,
            "world": {"K": int(spec["teams"]), "J": int(spec["carts"]), "N": int(spec["players"]), "M": int(spec["tiles"])},
            "seats": seats,
            "rollout": rollout["stats"],
            "update": upd,
            "pfsp_wr": {k: round(v, 3) for k, v in sorted(pfsp_wr.items())},
        }
        history.append(rec)

        if it == 1 or it % max(1, sc.eval_interval) == 0 or it == sc.iterations:
            print(
                f"  iter {it:02d}/{sc.iterations} K={spec['teams']:2d} J={spec['carts']:2d} N={spec['players']:2d} | "
                f"roll={rollout['stats']['wall_s']:.2f}s ({rollout['stats']['speedup_x']:.0f}x) "
                f"upd={upd['update_wall_s']:.2f}s | loss={upd['loss_total']:.3f} ent={upd['entropy_disc']:.2f} "
                f"dScore(main={rollout['stats']['main_dscore_mean']:.1f}, opp={rollout['stats']['opp_dscore_mean']:.1f}) "
                f"wr(def={pfsp_wr.get('default', 0.5):.2f}, rush={pfsp_wr.get('script_cart_rush', 0.5):.2f})",
                flush=True,
            )

    train_wall_s = time.perf_counter() - t_train_start
    if main_mid is None:
        main_mid = clone_plan_model(model).to(device)
        arm_models["main_mid"] = main_mid
        torch.save(main_mid.state_dict(), out_dir / "main_mid.pt")

    main_final = clone_plan_model(model).to(device)
    arm_models["main_final"] = main_final
    torch.save(main_final.state_dict(), out_dir / "main_final.pt")

    eval_arms = ["default", "script_cart_rush", "script_leader_gank", "main_init", "main_mid", "main_final"]
    print(f"[sim_dt-grpo] training done in {train_wall_s:.2f}s ({total_sim_s / max(train_wall_s, 1e-6):.0f}x end-to-end). Running sim_dt league arena...", flush=True)
    sim_eval = eval_sim_dt_league(arm_models, eval_arms, cfg=cfg, device=device, audit_model=main_final)
    wnorm = weight_norm_report(main_final)

    print("[sim_dt-grpo] sim_dt Arena Elo & alpha-Rank:", flush=True)
    for arm in eval_arms:
        r = sim_eval["elo"]["ratings"].get(arm, {})
        print(
            f"  {arm:20s}: Elo={r.get('elo', 0):6.1f} (CI95={r.get('ci95')}) "
            f"alpha-Rank={sim_eval['alpha_rank'].get(arm, 0):.4f} mean_score={r.get('mean_score', 0):.1f}",
            flush=True,
        )

    rep_path = out_dir / "report.json"
    report = {
        "device": str(device),
        "config": cfg.to_dict(),
        "training_summary": {
            "iterations": sc.iterations,
            "train_wall_s": round(train_wall_s, 2),
            "rollout_wall_s": round(total_rollout_s, 2),
            "update_wall_s": round(total_update_s, 2),
            "total_sim_s": round(total_sim_s, 1),
            "rollout_speedup_x": round(total_sim_s / max(total_rollout_s, 1e-6), 1),
            "end_to_end_speedup_x": round(total_sim_s / max(train_wall_s, 1e-6), 1),
            "weight_norms": wnorm,
            "dead_neuron_fraction": sim_eval.get("dead_neuron_fraction"),
        },
        "history": history,
        "sim_dt_eval": sim_eval,
        "darkplaces_transfer_eval": None,
    }
    rep_path.write_text(json.dumps(report, indent=2))

    dp_eval: dict[str, Any] | None = None
    if sc.darkplaces_transfer_eval:
        print("[sim_dt-grpo] Running live zero-copy DarkPlaces transfer eval (sv_headless_unthrottled=1, g_payload_use_view=0)...", flush=True)
        dp_arms = ["default", "script_cart_rush", "main_init", "main_mid", "main_final"]
        dp_eval = eval_darkplaces_transfer(arm_models, dp_arms, cfg=cfg, device=device, work_dir=out_dir / "dp_transfer_eval")
        report["darkplaces_transfer_eval"] = dp_eval
        rep_path.write_text(json.dumps(report, indent=2))
        if "elo" in dp_eval and "ratings" in dp_eval["elo"]:
            print(
                f"[sim_dt-grpo] DarkPlaces Transfer Eval ({dp_eval['score_outs']}/{dp_eval['n_matches']} score-outs, "
                f"mean {dp_eval['mean_dp_speedup_x']}x realtime, {dp_eval['wall_s']}s wall):",
                flush=True,
            )
            for arm in dp_arms:
                r = dp_eval["elo"]["ratings"].get(arm, {})
                print(
                    f"  {arm:20s}: Elo={r.get('elo', 0):6.1f} (CI95={r.get('ci95')}) "
                    f"alpha-Rank={dp_eval['alpha_rank'].get(arm, 0):.4f} mean_score={r.get('mean_score', 0):.1f}",
                    flush=True,
                )
        else:
            print(f"[sim_dt-grpo] DarkPlaces Transfer Eval result: {dp_eval}", flush=True)

    print(f"[sim_dt-grpo] report saved -> {rep_path}", flush=True)


if __name__ == "__main__":
    main()
