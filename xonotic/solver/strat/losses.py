from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from payload.tools.strategy_io_schema import CS, OBS, TS

from .game_value import role_rewards


class PlanOutput(NamedTuple):
    cont_mean: torch.Tensor
    cont_logstd: torch.Tensor
    cart_logits: torch.Tensor
    rival_logits: torch.Tensor
    cell_logits: torch.Tensor
    value_winnie: torch.Tensor
    value_lou: torch.Tensor
    dyn_drift: torch.Tensor
    dyn_action: torch.Tensor
    h_next: torch.Tensor
    fsq_codes: torch.Tensor | None
    pred_skill: torch.Tensor | None = None


class PlanSample(NamedTuple):
    cont: torch.Tensor
    cart_pick: torch.Tensor
    rival_pick: torch.Tensor
    cell_pick: torch.Tensor
    logp: torch.Tensor


@dataclass
class PlanTransition:
    session: int
    obs_in: np.ndarray
    carts_in: np.ndarray
    teams_in: np.ndarray
    events_in: np.ndarray
    obs_out: np.ndarray
    carts_out: np.ndarray
    teams_out: np.ndarray
    events_out: np.ndarray
    cont: np.ndarray
    cart_pick: np.ndarray
    rival_pick: np.ndarray
    cell_pick: np.ndarray
    behavior_logp: np.ndarray
    reward: np.ndarray
    winner_mask: np.ndarray
    next_winner_mask: np.ndarray
    discount: np.ndarray
    actor_mask: np.ndarray
    successor_rows: np.ndarray
    geo_in: dict[str, np.ndarray] | None = None
    geo_out: dict[str, np.ndarray] | None = None
    h_in: np.ndarray | None = None
    h_out: np.ndarray | None = None
    map_name: str = "boil"
    priv_skill_in: np.ndarray | None = None
    priv_skill_out: np.ndarray | None = None
    gae_adv: np.ndarray | None = None
    gae_ret: np.ndarray | None = None


@dataclass(frozen=True)
class RewardConfig:
    winner_barrier_margin: float = 0.25
    winner_barrier_weight: float = 0.5
    potential_shaping_weight: float = 0.25
    potential_shaping_gamma: float = 0.95
    loser_potential_self_weight: float = 0.5
    cart_partial_cp_weight: float = 0.5
    lane_progress_weight: float = 0.25
    rival_lane_penalty: float = 0.8
    normalize_lane_by_scale: bool = True
    push_weight: float = 0.08
    push_scale: float = 25.0
    contest_weight: float = 0.06
    contest_scale: float = 1.0
    pickup_weight: float = 0.02
    pickup_scale: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardConfig":
        if not data:
            return cls()
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LossConfig:
    actor_weight: float = 1.0
    critic_weight: float = 0.5
    dyn_weight: float = 0.25
    skill_id_weight: float = 0.25
    moe_balance_weight: float = 0.01
    entropy_coef: float = 0.008
    entropy_main_start: float = 0.008
    entropy_main_pre_decay: float = 0.004
    entropy_main_post_decay: float = 0.006
    entropy_exploiter: float = 0.006
    ratio_clip: float = 0.2
    log_ratio_bound: float = 20.0
    rollout_is_threshold: float = 2.0
    adv_clip: float = 3.0
    adv_std_floor: float = 1e-4
    batch_adv_std_floor: float = 1e-3
    critic_huber_beta: float = 0.5
    dyn_huber_beta: float = 0.25
    skill_id_huber_beta: float = 0.5
    dyn_drho_scale: float = 10.0
    gamma: float = 0.95
    gae_lambda: float = 0.95
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LossConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            if f == "reward":
                kw[f] = RewardConfig.from_dict(data[f]) if isinstance(data[f], dict) else data[f]
            else:
                kw[f] = data[f]
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def entropy_for_iteration(self, role: str, iter_index_1based: int, total_main_iters: int, post_exploiter: bool = False) -> float:
        if role == "exploiter":
            return float(self.entropy_exploiter)
        decay = self.entropy_main_post_decay if post_exploiter else self.entropy_main_pre_decay
        frac = (iter_index_1based - 1) / max(1, total_main_iters - 1)
        return float(self.entropy_main_start - decay * frac)


DEFAULT_LOSS_CONFIG = LossConfig()


def clipped_policy_surrogate(log_ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    selected = torch.where(
        advantage >= 0,
        torch.clamp(log_ratio, max=math.log1p(clip)),
        torch.clamp(log_ratio, min=math.log1p(-clip)),
    )
    return torch.exp(selected) * advantage


def rollout_importance_weights(log_ratio: torch.Tensor, cfg: LossConfig = DEFAULT_LOSS_CONFIG) -> torch.Tensor:
    ratio = torch.exp(torch.clamp(log_ratio, -cfg.log_ratio_bound, cfg.log_ratio_bound))
    return torch.clamp(ratio, max=cfg.rollout_is_threshold).detach()


def plan_logp(
    out: PlanOutput,
    cont: torch.Tensor,
    cart_pick: torch.Tensor,
    rival_pick: torch.Tensor,
    cell_pick: torch.Tensor,
) -> torch.Tensor:
    inv_std = torch.exp(-out.cont_logstd)
    err = (cont - out.cont_mean) * inv_std
    lp_cont = -0.5 * (err.square() + 2.0 * out.cont_logstd + math.log(2.0 * math.pi)).sum(dim=-1)
    lp_cart = F.log_softmax(out.cart_logits, dim=-1).gather(-1, cart_pick.long().unsqueeze(-1)).squeeze(-1)
    lp_rival = F.log_softmax(out.rival_logits, dim=-1).gather(-1, rival_pick.long().unsqueeze(-1)).squeeze(-1)
    lp_cell = F.log_softmax(out.cell_logits, dim=-1).gather(-1, cell_pick.long().unsqueeze(-1)).squeeze(-1)
    return lp_cont + lp_cart + lp_rival + lp_cell


def plan_discrete_entropy(out: PlanOutput) -> torch.Tensor:
    p_cart = F.softmax(out.cart_logits, dim=-1)
    h_cart = -(p_cart * F.log_softmax(out.cart_logits, dim=-1)).sum(dim=-1)
    p_riv = F.softmax(out.rival_logits, dim=-1)
    h_riv = -(p_riv * F.log_softmax(out.rival_logits, dim=-1)).sum(dim=-1)
    p_cell = F.softmax(out.cell_logits, dim=-1)
    h_cell = -(p_cell * F.log_softmax(out.cell_logits, dim=-1)).sum(dim=-1)
    return h_cart + h_riv + h_cell


def plan_entropy(out: PlanOutput) -> torch.Tensor:
    h_cont = (out.cont_logstd + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)
    return h_cont + plan_discrete_entropy(out)


def compute_step_rewards(
    prev_context,
    prev_cartstate,
    cur_cartstate,
    prev_obs: np.ndarray,
    cur_obs: np.ndarray,
    prev_carts: np.ndarray,
    cur_carts: np.ndarray,
    succ: np.ndarray,
    n_teams: int,
    cfg: RewardConfig | None = None,
) -> np.ndarray:
    rcfg = cfg or DEFAULT_LOSS_CONFIG.reward
    rew = role_rewards(
        prev_context,
        prev_cartstate,
        cur_cartstate,
        barrier_margin=rcfg.winner_barrier_margin,
        barrier_weight=rcfg.winner_barrier_weight,
        shaping_weight=rcfg.potential_shaping_weight,
        shaping_gamma=rcfg.potential_shaping_gamma,
        loser_self_weight=rcfg.loser_potential_self_weight,
        partial_cp_weight=rcfg.cart_partial_cp_weight,
    )
    push_delta = np.asarray(
        [
            max(0.0, float(cur_obs[idx, OBS["CART_PUSH"]]) - float(prev_obs[i, OBS["CART_PUSH"]])) if idx >= 0 else 0.0
            for i, idx in enumerate(succ)
        ],
        dtype=np.float32,
    )
    contest_delta = np.asarray(
        [
            max(0.0, float(cur_obs[idx, OBS["CART_CONTEST"]]) - float(prev_obs[i, OBS["CART_CONTEST"]])) if idx >= 0 else 0.0
            for i, idx in enumerate(succ)
        ],
        dtype=np.float32,
    )
    pickup_delta = np.asarray(
        [
            max(0.0, float(cur_obs[idx, OBS["PICKUPS"]]) - float(prev_obs[i, OBS["PICKUPS"]])) if idx >= 0 else 0.0
            for i, idx in enumerate(succ)
        ],
        dtype=np.float32,
    )
    prev_frac = prev_carts[:, CS["PATH_POSITION"]] / np.maximum(prev_carts[:, CS["PATH_LENGTH"]], 1.0)
    cur_frac = cur_carts[:, CS["PATH_POSITION"]] / np.maximum(cur_carts[:, CS["PATH_LENGTH"]], 1.0)
    dfrac = cur_frac - prev_frac
    ctrl_t = cur_carts[:, CS["CONTROL_TEAM"]].astype(np.int64)
    p_teams_1 = prev_obs[:, OBS["TEAM"]].astype(np.int64)
    k_eff = max(2, int(n_teams))
    j_eff = max(1, min(len(dfrac), len(ctrl_t)))
    if rcfg.normalize_lane_by_scale:
        carts_per_team = max(1.0, float(j_eff) / float(k_eff))
        own_scale = 1.0 / carts_per_team
        rival_scale = -float(rcfg.rival_lane_penalty) / (max(1.0, float(k_eff - 1)) * carts_per_team)
    else:
        own_scale = 1.0
        rival_scale = -0.5 * float(rcfg.rival_lane_penalty)
    lane_prog = np.zeros((len(prev_obs),), dtype=np.float32)
    for ci in range(j_eff):
        if ctrl_t[ci] > 0:
            df = float(dfrac[ci])
            lane_prog += np.where(p_teams_1 == ctrl_t[ci], own_scale * df, rival_scale * df)
    return (
        rew
        + float(rcfg.lane_progress_weight) * lane_prog
        + float(rcfg.push_weight) * np.tanh(push_delta / max(float(rcfg.push_scale), 1e-6))
        + float(rcfg.contest_weight) * np.tanh(contest_delta / max(float(rcfg.contest_scale), 1e-6))
        + float(rcfg.pickup_weight) * np.tanh(pickup_delta / max(float(rcfg.pickup_scale), 1e-6))
    ).astype(np.float32)


def _geo_to_torch(geo: dict[str, np.ndarray] | None, device: str) -> dict[str, torch.Tensor] | None:
    if geo is None:
        return None
    out = {}
    for k in (
        "bot_geo", "cart_geo", "track_tokens", "track_teams", "cell_tokens",
        "item_tokens", "fovea_weights", "map_center", "map_radius",
    ):
        if k in geo:
            out[k] = torch.as_tensor(geo[k], dtype=torch.float32, device=device)
    return out


def _team_eff_rho(carts: torch.Tensor, teams: torch.Tensor, obs: torch.Tensor, partial_cp_weight: float = 0.5) -> torch.Tensor:
    scores = teams[:, TS["SCORE"]].to(torch.float32)
    held = teams[:, TS["CHECKPOINTS_HELD"]].to(torch.float32)
    rate = teams[:, TS["CHECKPOINT_RATE"]].to(torch.float32)
    limit = torch.clamp(teams[:, TS["SCORE_LIMIT"]].to(torch.float32), min=1e-6)
    rem = torch.clamp(limit - scores, min=1e-6)
    path_len = torch.clamp(carts[:, CS["PATH_LENGTH"]].to(torch.float32), min=1e-6)
    pos_frac = torch.clamp(carts[:, CS["PATH_POSITION"]].to(torch.float32) / path_len, 0.0, 1.0)
    cp_count = torch.clamp(carts[:, CS["CHECKPOINT_COUNT"]].to(torch.float32), min=1.0)
    seg_frac = (pos_frac * cp_count) - torch.floor(pos_frac * cp_count)
    owner = carts[:, CS["CONTROL_TEAM"]].to(torch.float32)[:, None] == teams[:, TS["ID"]].to(torch.float32)[None, :]
    eff_held = held + float(partial_cp_weight) * (owner.to(torch.float32).T @ seg_frac)
    rho = torch.where(eff_held > 0, (rate * eff_held) / rem, torch.zeros_like(scores))
    mem = obs[:, OBS["TEAM"]].to(torch.float32)[:, None] == teams[:, TS["ID"]].to(torch.float32)[None, :]
    return mem.to(torch.float32) @ rho


def attach_episode_gae_(
    model,
    by_session: dict[int, list[PlanTransition]],
    device: str,
    gae_lambda: float = DEFAULT_LOSS_CONFIG.gae_lambda,
) -> None:
    with torch.no_grad():
        for trs in by_session.values():
            if not trs:
                continue
            T = len(trs)
            v_in_list: list[np.ndarray] = []
            for tr in trs:
                obs_in = torch.as_tensor(tr.obs_in, dtype=torch.float32, device=device)
                carts_in = torch.as_tensor(tr.carts_in, dtype=torch.float32, device=device)
                teams_in = torch.as_tensor(tr.teams_in, dtype=torch.float32, device=device)
                evt_in = torch.as_tensor(np.array(tr.events_in, copy=True), dtype=torch.float32, device=device) if len(tr.events_in) else None
                geo_in = _geo_to_torch(tr.geo_in, device)
                h_in = torch.as_tensor(tr.h_in, dtype=torch.float32, device=device) if tr.h_in is not None else None
                priv_in = torch.as_tensor(tr.priv_skill_in, dtype=torch.float32, device=device) if tr.priv_skill_in is not None else None
                cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in, priv_skill=priv_in)
                wm = torch.as_tensor(tr.winner_mask, dtype=torch.bool, device=device)
                v_in_list.append(torch.where(wm, cur.value_winnie, cur.value_lou).cpu().numpy().astype(np.float32))

            last = trs[-1]
            obs_last = torch.as_tensor(last.obs_out, dtype=torch.float32, device=device)
            carts_last = torch.as_tensor(last.carts_out, dtype=torch.float32, device=device)
            teams_last = torch.as_tensor(last.teams_out, dtype=torch.float32, device=device)
            evt_last = torch.as_tensor(np.array(last.events_out, copy=True), dtype=torch.float32, device=device) if len(last.events_out) else None
            geo_last = _geo_to_torch(last.geo_out, device)
            h_last = torch.as_tensor(last.h_out, dtype=torch.float32, device=device) if last.h_out is not None else None
            priv_last = torch.as_tensor(last.priv_skill_out, dtype=torch.float32, device=device) if last.priv_skill_out is not None else None
            nxt_last = model(obs_last, carts_last, teams_last, evt_last, geo=geo_last, h_prev=h_last, priv_skill=priv_last)
            last_vw = nxt_last.value_winnie.cpu().numpy().astype(np.float32)
            last_vl = nxt_last.value_lou.cpu().numpy().astype(np.float32)

            next_adv: np.ndarray | None = None
            next_id_map: dict[int, int] = {}
            for t in range(T - 1, -1, -1):
                tr = trs[t]
                v_in = v_in_list[t]
                n = len(v_in)
                adv = np.zeros((n,), dtype=np.float32)
                eids = tr.obs_in[:, OBS["ID"]].astype(np.int64)
                for i in range(n):
                    disc = float(tr.discount[i])
                    succ = int(tr.successor_rows[i])
                    same_role = bool(tr.winner_mask[i]) == bool(tr.next_winner_mask[i])
                    v_next = 0.0
                    gae_next = 0.0
                    if same_role and disc > 0.0 and succ >= 0:
                        if t == T - 1:
                            if succ < len(last_vw):
                                v_next = float(last_vw[succ] if tr.next_winner_mask[i] else last_vl[succ])
                        else:
                            j = next_id_map.get(int(eids[i]), -1)
                            if j >= 0 and bool(trs[t + 1].winner_mask[j]) == bool(tr.next_winner_mask[i]):
                                v_next = float(v_in_list[t + 1][j])
                                gae_next = float(next_adv[j])
                    delta = float(tr.reward[i]) + disc * v_next - float(v_in[i])
                    adv[i] = delta + disc * gae_lambda * gae_next
                tr.gae_adv = adv
                tr.gae_ret = (adv + v_in).astype(np.float32)
                next_adv = adv
                next_id_map = {int(eids[i]): i for i in range(n)}


def plan_losses_for_transition(
    model,
    tr: PlanTransition,
    device: str = "cpu",
    *,
    loss_cfg: LossConfig | None = None,
    entropy_coef: float | None = None,
    actor_weight: float | None = None,
    critic_weight: float | None = None,
    dyn_weight: float | None = None,
    skill_id_weight: float | None = None,
    adv_mean: float = 0.0,
    adv_std: float = 1.0,
    world=None,
) -> tuple[dict[str, torch.Tensor], dict]:
    cfg = loss_cfg or DEFAULT_LOSS_CONFIG
    ent_c = cfg.entropy_coef if entropy_coef is None else float(entropy_coef)
    w_act = cfg.actor_weight if actor_weight is None else float(actor_weight)
    w_cri = cfg.critic_weight if critic_weight is None else float(critic_weight)
    w_dyn = cfg.dyn_weight if dyn_weight is None else float(dyn_weight)
    w_ski = cfg.skill_id_weight if skill_id_weight is None else float(skill_id_weight)

    obs_in = torch.as_tensor(tr.obs_in, dtype=torch.float32, device=device)
    carts_in = torch.as_tensor(tr.carts_in, dtype=torch.float32, device=device)
    teams_in = torch.as_tensor(tr.teams_in, dtype=torch.float32, device=device)
    evt_in = torch.as_tensor(np.array(tr.events_in, copy=True), dtype=torch.float32, device=device) if len(tr.events_in) else None
    geo_in = _geo_to_torch(tr.geo_in, device)
    h_in = torch.as_tensor(tr.h_in, dtype=torch.float32, device=device) if tr.h_in is not None else None
    priv_in = torch.as_tensor(tr.priv_skill_in, dtype=torch.float32, device=device) if tr.priv_skill_in is not None else None

    obs_out = torch.as_tensor(tr.obs_out, dtype=torch.float32, device=device)
    carts_out = torch.as_tensor(tr.carts_out, dtype=torch.float32, device=device)
    teams_out = torch.as_tensor(tr.teams_out, dtype=torch.float32, device=device)
    evt_out = torch.as_tensor(np.array(tr.events_out, copy=True), dtype=torch.float32, device=device) if len(tr.events_out) else None
    geo_out = _geo_to_torch(tr.geo_out, device)
    h_out = torch.as_tensor(tr.h_out, dtype=torch.float32, device=device) if tr.h_out is not None else None
    priv_out = torch.as_tensor(tr.priv_skill_out, dtype=torch.float32, device=device) if tr.priv_skill_out is not None else None

    cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in, world=world, priv_skill=priv_in)
    with torch.no_grad():
        nxt = model(obs_out, carts_out, teams_out, evt_out, geo=geo_out, h_prev=h_out, world=world, priv_skill=priv_out)

    cont = torch.as_tensor(tr.cont, dtype=torch.float32, device=device)
    cart_pick = torch.as_tensor(tr.cart_pick, dtype=torch.long, device=device)
    rival_pick = torch.as_tensor(tr.rival_pick, dtype=torch.long, device=device)
    cell_pick = torch.as_tensor(tr.cell_pick, dtype=torch.long, device=device)
    beh_lp = torch.as_tensor(tr.behavior_logp, dtype=torch.float32, device=device)
    reward = torch.as_tensor(tr.reward, dtype=torch.float32, device=device)
    wm = torch.as_tensor(tr.winner_mask, dtype=torch.bool, device=device)
    nwm = torch.as_tensor(tr.next_winner_mask, dtype=torch.bool, device=device)
    disc = torch.as_tensor(tr.discount, dtype=torch.float32, device=device)
    am = torch.as_tensor(tr.actor_mask, dtype=torch.float32, device=device)
    succ = torch.clamp(torch.as_tensor(tr.successor_rows, dtype=torch.long, device=device), min=0, max=max(0, obs_out.shape[0] - 1))
    valid_succ = torch.as_tensor(tr.successor_rows >= 0, dtype=torch.float32, device=device)

    val = torch.where(wm, cur.value_winnie, cur.value_lou)
    if tr.gae_adv is not None and tr.gae_ret is not None:
        raw_adv = torch.as_tensor(tr.gae_adv, dtype=torch.float32, device=device)
        target = torch.as_tensor(tr.gae_ret, dtype=torch.float32, device=device)
    else:
        boot = torch.where(nwm, nxt.value_winnie[succ], nxt.value_lou[succ])
        boot = torch.where((wm == nwm) & (disc > 0) & (valid_succ > 0), boot, 0.0)
        target = reward + disc * boot.detach()
        raw_adv = (target - val).detach()

    logpi = plan_logp(cur, cont, cart_pick, rival_pick, cell_pick)
    log_ratio = torch.clamp(logpi - beh_lp, -cfg.log_ratio_bound, cfg.log_ratio_bound)
    adv_norm = (raw_adv - float(adv_mean)) / max(float(adv_std), cfg.adv_std_floor)
    adv = torch.clamp(adv_norm, -cfg.adv_clip, cfg.adv_clip)
    am_sum = torch.clamp(am.sum(), min=1.0)
    surrogate = clipped_policy_surrogate(log_ratio, adv, cfg.ratio_clip)
    ent = (plan_entropy(cur) * am).sum() / am_sum
    ent_disc = (plan_discrete_entropy(cur) * am).sum() / am_sum
    actor_loss = w_act * (-(surrogate * am).sum() / am_sum - ent_c * ent_disc)

    w_mask = wm.to(torch.float32)
    l_mask = (~wm).to(torch.float32)
    w_loss = w_cri * (F.smooth_l1_loss(cur.value_winnie, target, reduction="none", beta=cfg.critic_huber_beta) * w_mask).sum() / torch.clamp(w_mask.sum(), min=1.0)
    l_loss = w_cri * (F.smooth_l1_loss(cur.value_lou, target, reduction="none", beta=cfg.critic_huber_beta) * l_mask).sum() / torch.clamp(l_mask.sum(), min=1.0)

    with torch.no_grad():
        rho_c = _team_eff_rho(carts_in, teams_in, obs_in, partial_cp_weight=cfg.reward.cart_partial_cp_weight)
        rho_n = _team_eff_rho(carts_out, teams_out, obs_out, partial_cp_weight=cfg.reward.cart_partial_cp_weight)[succ]
        obs_drho = torch.asinh(cfg.dyn_drho_scale * (rho_n - rho_c))
    pred_drho = cur.dyn_drift + (cur.dyn_action * cont).sum(dim=-1)
    dyn_loss = w_dyn * (F.smooth_l1_loss(pred_drho, obs_drho, reduction="none", beta=cfg.dyn_huber_beta) * valid_succ).sum() / torch.clamp(valid_succ.sum(), min=1.0)

    if priv_in is not None and cur.pred_skill is not None:
        skill_loss = w_ski * F.smooth_l1_loss(cur.pred_skill, priv_in, beta=cfg.skill_id_huber_beta)
        skill_mse = float((cur.pred_skill.detach() - priv_in).square().mean().item())
    else:
        skill_loss = torch.zeros((), dtype=torch.float32, device=device)
        skill_mse = None

    total = actor_loss + w_loss + l_loss + dyn_loss + skill_loss
    t_lo, t_hi = getattr(model, "temp_bounds", (0.5, 32.0))
    with torch.no_grad():
        lr_np = log_ratio.detach().cpu().numpy()[tr.actor_mask.astype(bool)] if np.any(tr.actor_mask) else np.empty((0,))
        ratio_np = np.exp(np.clip(lr_np, -cfg.log_ratio_bound, cfg.log_ratio_bound)) if lr_np.size else np.empty((0,))
        diag = {
            "loss_total": float(total.item()),
            "loss_actor": float(actor_loss.item()),
            "loss_w": float(w_loss.item()),
            "loss_l": float(l_loss.item()),
            "loss_dyn": float(dyn_loss.item()),
            "loss_skill": float(skill_loss.item()),
            "skill_id_mse": skill_mse,
            "entropy": float(ent.item()),
            "entropy_disc": float(ent_disc.item()),
            "cont_std_mean": float(torch.exp(cur.cont_logstd).mean().item()),
            "is_ratio_mean": float(ratio_np.mean()) if ratio_np.size else None,
            "is_clip_frac": float(np.mean(np.abs(lr_np) > math.log(1.0 + cfg.ratio_clip))) if lr_np.size else None,
            "winnie_mse": float(((cur.value_winnie - target).square() * w_mask).sum().item() / max(1.0, float(w_mask.sum().item()))) if float(w_mask.sum().item()) > 0 else None,
            "lou_mse": float(((cur.value_lou - target).square() * l_mask).sum().item() / max(1.0, float(l_mask.sum().item()))) if float(l_mask.sum().item()) > 0 else None,
            "reward_abs": float(reward.abs().mean().item()),
            "cart_temp": float(torch.clamp(torch.exp(model.log_temp_cart), t_lo, t_hi).item()),
            "cell_temp": float(torch.clamp(torch.exp(model.log_temp_cell), t_lo, t_hi).item()),
        }
    return {
        "total": total,
        "actor_pg": actor_loss,
        "critic_winnie": w_loss,
        "critic_lou": l_loss,
        "critic_both": w_loss + l_loss,
        "dynamics": dyn_loss,
        "skill_id": skill_loss,
    }, diag


def compute_batch_adv_stats(
    model,
    batch: list[PlanTransition],
    device: str = "cpu",
    loss_cfg: LossConfig | None = None,
) -> tuple[float, float]:
    cfg = loss_cfg or DEFAULT_LOSS_CONFIG
    errs = []
    with torch.no_grad():
        for tr in batch:
            am = tr.actor_mask.astype(bool)
            if not np.any(am):
                continue
            if tr.gae_adv is not None:
                errs.append(tr.gae_adv[am])
                continue
            obs_in = torch.as_tensor(tr.obs_in, dtype=torch.float32, device=device)
            carts_in = torch.as_tensor(tr.carts_in, dtype=torch.float32, device=device)
            teams_in = torch.as_tensor(tr.teams_in, dtype=torch.float32, device=device)
            evt_in = torch.as_tensor(np.array(tr.events_in, copy=True), dtype=torch.float32, device=device) if len(tr.events_in) else None
            geo_in = _geo_to_torch(tr.geo_in, device)
            h_in = torch.as_tensor(tr.h_in, dtype=torch.float32, device=device) if tr.h_in is not None else None
            priv_in = torch.as_tensor(tr.priv_skill_in, dtype=torch.float32, device=device) if tr.priv_skill_in is not None else None
            obs_out = torch.as_tensor(tr.obs_out, dtype=torch.float32, device=device)
            carts_out = torch.as_tensor(tr.carts_out, dtype=torch.float32, device=device)
            teams_out = torch.as_tensor(tr.teams_out, dtype=torch.float32, device=device)
            evt_out = torch.as_tensor(np.array(tr.events_out, copy=True), dtype=torch.float32, device=device) if len(tr.events_out) else None
            geo_out = _geo_to_torch(tr.geo_out, device)
            h_out = torch.as_tensor(tr.h_out, dtype=torch.float32, device=device) if tr.h_out is not None else None
            priv_out = torch.as_tensor(tr.priv_skill_out, dtype=torch.float32, device=device) if tr.priv_skill_out is not None else None
            cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in, priv_skill=priv_in)
            nxt = model(obs_out, carts_out, teams_out, evt_out, geo=geo_out, h_prev=h_out, priv_skill=priv_out)
            wm = torch.as_tensor(tr.winner_mask, dtype=torch.bool, device=device)
            nwm = torch.as_tensor(tr.next_winner_mask, dtype=torch.bool, device=device)
            disc = torch.as_tensor(tr.discount, dtype=torch.float32, device=device)
            succ = torch.clamp(torch.as_tensor(tr.successor_rows, dtype=torch.long, device=device), min=0, max=max(0, obs_out.shape[0] - 1))
            vs = torch.as_tensor(tr.successor_rows >= 0, dtype=torch.float32, device=device)
            val = torch.where(wm, cur.value_winnie, cur.value_lou)
            boot = torch.where(nwm, nxt.value_winnie[succ], nxt.value_lou[succ])
            boot = torch.where((wm == nwm) & (disc > 0) & (vs > 0), boot, 0.0)
            reward = torch.as_tensor(tr.reward, dtype=torch.float32, device=device)
            err = (reward + disc * boot - val).cpu().numpy()
            errs.append(err[am])
    if not errs:
        return 0.0, 1.0
    cat = np.concatenate(errs)
    return float(cat.mean()), float(max(cat.std(), cfg.batch_adv_std_floor))
