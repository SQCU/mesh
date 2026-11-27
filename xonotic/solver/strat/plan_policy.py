from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from payload.tools.strategy_io_schema import BOT_CONFIGURATION_COLUMNS, CS, EVT, EVT_WIDTH, OBS, OBS_WIDTH, TS

from .aurora import Aurora, aurora_init_weight_
from .game_value import cgt_tensor_features, cgt_tensor_features_batched
from .grad_snr import (
    grad_cosine,
    mccandlish_pair_from_grads,
    summarize_mccandlish_pairs,
)
from .losses import (
    DEFAULT_LOSS_CONFIG,
    LossConfig,
    PlanOutput,
    PlanSample,
    PlanTransition,
    _geo_to_torch,
    clipped_policy_surrogate,
    compute_batch_adv_stats,
    plan_discrete_entropy,
    plan_entropy,
    plan_logp,
    plan_losses_for_transition,
)
from .train_config import (
    DEFAULT_ACTION_PACK_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_NORM_CONFIG,
    DEFAULT_OPTIMIZER_CONFIG,
    ActionPackConfig,
    ModelConfig,
    NormConfig,
    OptimizerConfig,
)

PLAN_WIDTH = 16
PLAN_CONT_DIM = 11
BOT_GEO_DIM = 12
CART_GEO_DIM = 10
TRACK_TOK_DIM = 10
CELL_TOK_DIM = 10
ITEM_TOK_DIM = 9


def mp_norm(x: torch.Tensor, eps: float = DEFAULT_NORM_CONFIG.mp_norm_eps) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def mp_unit_vec(x: torch.Tensor, eps: float = DEFAULT_NORM_CONFIG.mp_unit_vec_eps) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def mp_silu(x: torch.Tensor, scale: float = DEFAULT_NORM_CONFIG.mp_silu_scale) -> torch.Tensor:
    return F.silu(x) / scale


def mp_lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = DEFAULT_NORM_CONFIG.mp_lerp_eps) -> torch.Tensor:
    t_c = torch.clamp(t, 0.0, 1.0)
    return ((1.0 - t_c) * a + t_c * b) * torch.rsqrt((1.0 - t_c).square() + t_c.square() + eps)


class MPLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        init_pp_iterations: int = DEFAULT_OPTIMIZER_CONFIG.init_pp_iterations,
        init_pp_beta: float = DEFAULT_OPTIMIZER_CONFIG.pp_beta,
        init_ns_steps: int = DEFAULT_OPTIMIZER_CONFIG.init_ns_steps,
        unit_norm_eps: float = DEFAULT_NORM_CONFIG.unit_norm_eps,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.unit_norm_eps = float(unit_norm_eps)
        w = torch.empty(self.out_features, self.in_features)
        aurora_init_weight_(
            w,
            pp_iterations=init_pp_iterations,
            pp_beta=init_pp_beta,
            ns_steps=init_ns_steps,
            unit_norm_eps=self.unit_norm_eps,
        )
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_unit = self.weight / self.weight.norm(dim=-1, keepdim=True).clamp_min(self.unit_norm_eps)
        return F.linear(x, w_unit)


@torch.no_grad()
def project_unit_norm_(model: nn.Module, eps: float = DEFAULT_NORM_CONFIG.unit_norm_eps) -> None:
    for m in model.modules():
        if isinstance(m, MPLinear):
            w = m.weight.data
            w.div_(w.norm(dim=-1, keepdim=True).clamp_min(eps))


def weight_norm_report(model: nn.Module) -> dict:
    norms = []
    for m in model.modules():
        if isinstance(m, MPLinear):
            norms.append(m.weight.detach().norm(dim=-1).cpu())
    if not norms:
        return {"layers": 0}
    cat = torch.cat(norms)
    return {
        "layers": len(norms),
        "rows": int(cat.numel()),
        "min_norm": round(float(cat.min().item()), 6),
        "max_norm": round(float(cat.max().item()), 6),
        "max_dev_from_1": round(float((cat - 1.0).abs().max().item()), 6),
    }


def make_aurora_optimizer(
    model: nn.Module,
    lr: float | None = None,
    scalar_lr: float | None = None,
    mu: float | None = None,
    pp_iterations: int | None = None,
    pp_beta: float | None = None,
    opt_cfg: OptimizerConfig | None = None,
) -> Aurora:
    cfg = opt_cfg or DEFAULT_OPTIMIZER_CONFIG
    return Aurora(
        model.parameters(),
        lr=cfg.lr if lr is None else float(lr),
        scalar_lr=cfg.scalar_lr if scalar_lr is None else float(scalar_lr),
        mu=cfg.mu if mu is None else float(mu),
        nesterov=cfg.nesterov,
        pp_iterations=cfg.pp_iterations if pp_iterations is None else int(pp_iterations),
        pp_beta=cfg.pp_beta if pp_beta is None else float(pp_beta),
        ns_steps=cfg.ns_steps,
        eps=cfg.eps,
        ns_coeffs=cfg.ns_coeffs,
        d_init_clamp=cfg.d_init_clamp,
        d_iter_clamp=cfg.d_iter_clamp,
        scalar_beta2=cfg.scalar_beta2,
        scalar_eps=cfg.scalar_eps,
    )


def make_edm2_optimizer(model: nn.Module, lr: float | None = None, opt_cfg: OptimizerConfig | None = None) -> Aurora:
    return make_aurora_optimizer(model, lr=lr, opt_cfg=opt_cfg)


def edm2_step(optimizer: torch.optim.Optimizer, model: nn.Module, grad_clip: float = DEFAULT_OPTIMIZER_CONFIG.grad_clip) -> float:
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
    optimizer.step()
    project_unit_norm_(model)
    return gnorm


def prenormalize_obs(
    obs: torch.Tensor,
    geo: dict[str, torch.Tensor] | None,
    norm_cfg: NormConfig = DEFAULT_NORM_CONFIG,
) -> torch.Tensor:
    x = obs.clone()
    if geo is not None and "map_center" in geo and "map_radius" in geo:
        center = geo["map_center"].to(obs.device, obs.dtype)
        radius = torch.clamp(geo["map_radius"].to(obs.device, obs.dtype), min=norm_cfg.map_radius_min)
        x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] = (x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] - center[None, :]) / radius
    else:
        x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] = x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] / norm_cfg.pos_fallback_scale
    x[:, OBS["VEL_X"] : OBS["VEL_Z"] + 1] = x[:, OBS["VEL_X"] : OBS["VEL_Z"] + 1] / norm_cfg.vel_scale
    x[:, OBS["HEALTH"]] = (x[:, OBS["HEALTH"]] - norm_cfg.health_offset) / norm_cfg.health_scale
    x[:, OBS["ARMOR"]] = (x[:, OBS["ARMOR"]] - norm_cfg.armor_offset) / norm_cfg.armor_scale
    x[:, OBS["AMMO_SHELLS"] : OBS["AMMO_FUEL"] + 1] = x[:, OBS["AMMO_SHELLS"] : OBS["AMMO_FUEL"] + 1] / norm_cfg.ammo_scale
    for col in (
        "ENGINE_TIME", "SPAWN_TIME", "STRENGTH_FINISHED", "RESPONSE_SEQ", "RESPONSE_TIME",
        "APPLIED_SEQUENCE", "OUTCOME_A_SEQ", "OUTCOME_B_SEQ", "CELL_X", "CELL_Y",
        "SWIZZLE_SCHEDULED_TIME", "SWIZZLE_ACTUAL_TIME", *BOT_CONFIGURATION_COLUMNS,
    ):
        if col == "STRENGTH_FINISHED":
            rem = torch.clamp(obs[:, OBS["STRENGTH_FINISHED"]] - obs[:, OBS["ENGINE_TIME"]], min=0.0, max=norm_cfg.strength_max_s) / norm_cfg.strength_max_s
            x[:, OBS[col]] = rem
        else:
            x[:, OBS[col]] = 0.0
    x = torch.asinh(x)
    return mp_norm(x, eps=norm_cfg.mp_norm_eps)


def prenormalize_carts(
    carts: torch.Tensor,
    cart_extra: torch.Tensor,
    geo: dict[str, torch.Tensor] | None,
    norm_cfg: NormConfig = DEFAULT_NORM_CONFIG,
) -> torch.Tensor:
    c = carts.clone()
    plen = torch.clamp(c[:, CS["PATH_LENGTH"]], min=1.0)
    c[:, CS["PATH_POSITION"]] = c[:, CS["PATH_POSITION"]] / plen
    c[:, CS["PATH_LENGTH"]] = torch.log1p(plen / norm_cfg.cart_len_scale)
    c[:, CS["SPEED"]] = c[:, CS["SPEED"]] / norm_cfg.cart_speed_scale
    if geo is not None and "map_center" in geo and "map_radius" in geo:
        center = geo["map_center"].to(carts.device, carts.dtype)
        radius = torch.clamp(geo["map_radius"].to(carts.device, carts.dtype), min=norm_cfg.map_radius_min)
        c[:, CS["POS_X"] : CS["POS_Z"] + 1] = (c[:, CS["POS_X"] : CS["POS_Z"] + 1] - center[None, :]) / radius
    else:
        c[:, CS["POS_X"] : CS["POS_Z"] + 1] = c[:, CS["POS_X"] : CS["POS_Z"] + 1] / norm_cfg.pos_fallback_scale
    c_geo = geo["cart_geo"] if (geo is not None and "cart_geo" in geo) else torch.zeros((carts.shape[0], CART_GEO_DIM), device=carts.device, dtype=carts.dtype)
    return mp_norm(torch.asinh(torch.cat((c, cart_extra, c_geo), dim=-1)), eps=norm_cfg.mp_norm_eps)


def prenormalize_teams(
    teams: torch.Tensor,
    team_extra: torch.Tensor,
    norm_cfg: NormConfig = DEFAULT_NORM_CONFIG,
) -> torch.Tensor:
    t = teams.clone()
    lim = torch.clamp(t[:, TS["SCORE_LIMIT"]], min=1.0)
    t[:, TS["SCORE"]] = t[:, TS["SCORE"]] / lim
    t[:, TS["SCORE_LIMIT"]] = 1.0
    t[:, TS["EPISODE"]] = 0.0
    return mp_norm(torch.asinh(torch.cat((t, team_extra), dim=-1)), eps=norm_cfg.mp_norm_eps)


class FSQBottleneck(nn.Module):

    def __init__(
        self,
        d: int,
        levels: tuple[int, ...] = DEFAULT_MODEL_CONFIG.fsq_levels,
        *,
        in_gain_init: float = DEFAULT_MODEL_CONFIG.fsq_in_gain_init,
        in_gain_bounds: tuple[float, float] = DEFAULT_MODEL_CONFIG.fsq_in_gain_bounds,
        blend_init_logit: float = DEFAULT_MODEL_CONFIG.fsq_blend_init_logit,
        std_min: float = DEFAULT_MODEL_CONFIG.fsq_std_min,
        norm_cfg: NormConfig = DEFAULT_NORM_CONFIG,
    ):
        super().__init__()
        self.levels = tuple(int(v) for v in levels)
        self.in_gain_bounds = (float(in_gain_bounds[0]), float(in_gain_bounds[1]))
        self.std_min = float(std_min)
        self.norm_cfg = norm_cfg
        self.proj_in = MPLinear(d, len(self.levels), unit_norm_eps=norm_cfg.unit_norm_eps)
        self.proj_out = MPLinear(len(self.levels), d, unit_norm_eps=norm_cfg.unit_norm_eps)
        self.in_gain = nn.Parameter(torch.tensor(float(in_gain_init)))
        self.blend_logit = nn.Parameter(torch.tensor(float(blend_init_logit)))
        half = torch.tensor([(v - 1) / 2.0 for v in self.levels], dtype=torch.float32)
        self.register_buffer("half_levels", half)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.clamp(self.in_gain, self.in_gain_bounds[0], self.in_gain_bounds[1])
        z_raw = self.proj_in(x)
        if z_raw.ndim == 2 and z_raw.shape[0] > 1:
            z_mean = z_raw.mean(dim=0, keepdim=True)
            z_std = z_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.std_min)
            z_raw = (z_raw - z_mean) / z_std
        elif z_raw.ndim == 3 and z_raw.shape[1] > 1:
            z_mean = z_raw.mean(dim=1, keepdim=True)
            z_std = z_raw.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.std_min)
            z_raw = (z_raw - z_mean) / z_std
        z = torch.tanh(g * z_raw) * self.half_levels
        q = torch.round(z)
        z_ste = z + (q - z).detach()
        scaled_q = z_ste / torch.clamp(self.half_levels, min=1.0)
        out = mp_lerp(x, self.proj_out(scaled_q), torch.sigmoid(self.blend_logit), eps=self.norm_cfg.mp_lerp_eps)
        return mp_norm(out, eps=self.norm_cfg.mp_norm_eps), q.detach()


class MPResCGTBlock(nn.Module):

    def __init__(
        self,
        d: int = DEFAULT_MODEL_CONFIG.d,
        h: int = DEFAULT_MODEL_CONFIG.h,
        r_gram: int = DEFAULT_MODEL_CONFIG.r_gram,
        r_dpp: int = DEFAULT_MODEL_CONFIG.r_dpp,
        init_blend: float = 0.25,
        norm_cfg: NormConfig = DEFAULT_NORM_CONFIG,
    ):
        super().__init__()
        self.d = d
        self.h = h
        self.r_gram = r_gram
        self.r_team = r_dpp
        self.r_dpp = r_dpp
        self.norm_cfg = norm_cfg
        eps = norm_cfg.unit_norm_eps

        self.val_proj = MPLinear(d, d, unit_norm_eps=eps)
        self.pool_score = MPLinear(d, 1, unit_norm_eps=eps)
        self.team_metric = MPLinear(d, self.r_team, unit_norm_eps=eps)
        self.rival_metric = MPLinear(d, self.r_team, unit_norm_eps=eps)
        self.team_val_proj = MPLinear(d, d, unit_norm_eps=eps)
        self.team_mutate_proj = MPLinear(d, d, unit_norm_eps=eps)
        self.team_bcast_proj = MPLinear(d, d, unit_norm_eps=eps)
        self.global_metric = MPLinear(d, r_gram, unit_norm_eps=eps)
        self.gram_out = MPLinear(d, d, unit_norm_eps=eps)

        self.ffn_gate = MPLinear(d, h, unit_norm_eps=eps)
        self.ffn_up = MPLinear(d, h, unit_norm_eps=eps)
        self.ffn_down = MPLinear(h, d, unit_norm_eps=eps)

        logit_b = math.log(init_blend / max(1e-4, 1.0 - init_blend))
        self.blend_pool = nn.Parameter(torch.tensor(0.0))
        self.blend_team_mut = nn.Parameter(torch.tensor(0.0))
        self.blend_gram = nn.Parameter(torch.tensor(logit_b))
        self.blend_ffn = nn.Parameter(torch.tensor(logit_b))
        self._last_ffn_act: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        entity_team_ids: torch.Tensor,
        team_ids: torch.Tensor,
        team_valid: torch.Tensor,
        world=None,
        team_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        nc = self.norm_cfg
        xn = mp_norm(x, eps=nc.mp_norm_eps)
        v = mp_norm(self.val_proj(xn), eps=nc.mp_norm_eps)
        m_rows = x.shape[0]
        k_teams = team_ids.shape[0]
        inv_m = 1.0 / math.sqrt(max(1, m_rows))
        inv_k = 1.0 / math.sqrt(max(1, k_teams))

        r = mp_norm(self.global_metric(xn), eps=nc.mp_norm_eps)
        if world is not None and world.parts("cross") is not None:
            lo, hi = world.span("cross", m_rows)
            r_c, v_c = world.copy(r), world.copy(v)
            kv_global = world.reduce(r_c[lo:hi].T @ v_c[lo:hi]) * inv_m
        else:
            kv_global = (r.T @ v) * inv_m
        global_vals = mp_norm(r @ kv_global, eps=nc.mp_norm_eps)

        mem = (entity_team_ids[:, None] == team_ids[None, :]) & team_valid[None, :]
        mem_f = mem.to(x.dtype)
        pool_w = mem_f * torch.sigmoid(self.pool_score(xn))
        if world is not None and world.parts("cross") is not None:
            lo, hi = world.span("cross", m_rows)
            pw_c, v_c = world.copy(pool_w), world.copy(v)
            team_sum = world.reduce(pw_c[lo:hi].T @ v_c[lo:hi])
            team_den = torch.clamp(world.reduce(pw_c[lo:hi].sum(dim=0, keepdim=True).T), min=1e-4)
        else:
            team_sum = pool_w.T @ v
            team_den = torch.clamp(pool_w.sum(dim=0, keepdim=True).T, min=1e-4)
        team_pool = mp_norm(team_sum / team_den, eps=nc.mp_norm_eps)

        if team_slice is not None and (team_slice[1] - team_slice[0]) == k_teams:
            t_cur = xn[team_slice[0] : team_slice[1]]
            t_in = mp_norm(mp_lerp(t_cur, team_pool, torch.sigmoid(self.blend_pool), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        else:
            t_in = team_pool

        q_t = mp_norm(self.team_metric(t_in), eps=nc.mp_norm_eps)
        k_t = mp_norm(self.rival_metric(t_in), eps=nc.mp_norm_eps)
        v_t = mp_norm(self.team_val_proj(t_in), eps=nc.mp_norm_eps)
        gram_t = (q_t @ k_t.T) * inv_k
        t_mixed = mp_norm(gram_t @ v_t, eps=nc.mp_norm_eps)
        t_mut = mp_norm(
            mp_lerp(t_in, mp_norm(self.team_mutate_proj(t_mixed), eps=nc.mp_norm_eps), torch.sigmoid(self.blend_team_mut), eps=nc.mp_lerp_eps),
            eps=nc.mp_norm_eps,
        )

        t_bcast = mp_norm(self.team_bcast_proj(t_mut), eps=nc.mp_norm_eps)
        has_team = mem_f.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        t_global = mp_norm(t_bcast.mean(dim=0, keepdim=True), eps=nc.mp_norm_eps)
        team_vals = mp_norm(has_team * (mem_f @ t_bcast) + (1.0 - has_team) * t_global, eps=nc.mp_norm_eps)

        gram_merged = mp_norm(self.gram_out(mp_norm(global_vals + team_vals, eps=nc.mp_norm_eps)), eps=nc.mp_norm_eps)
        x = mp_norm(mp_lerp(xn, gram_merged, torch.sigmoid(self.blend_gram), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)

        if world is not None and world.parts("ffn") is not None:
            lo, hi = world.span("ffn", self.h)
            x_c = world.copy(x)
            wg = self.ffn_gate.weight[lo:hi]
            wu = self.ffn_up.weight[lo:hi]
            wd = self.ffn_down.weight[:, lo:hi]
            wg = wg / wg.norm(dim=-1, keepdim=True).clamp_min(nc.unit_norm_eps)
            wu = wu / wu.norm(dim=-1, keepdim=True).clamp_min(nc.unit_norm_eps)
            wd_full_norm = self.ffn_down.weight.norm(dim=-1, keepdim=True).clamp_min(nc.unit_norm_eps)
            wd = wd / wd_full_norm
            hid = mp_silu(F.linear(x_c, wg), scale=nc.mp_silu_scale) * F.linear(x_c, wu)
            ffn_out = mp_norm(world.reduce(F.linear(hid, wd)), eps=nc.mp_norm_eps)
        else:
            hid = mp_silu(self.ffn_gate(x), scale=nc.mp_silu_scale) * self.ffn_up(x)
            self._last_ffn_act = hid.detach()
            ffn_out = mp_norm(self.ffn_down(hid), eps=nc.mp_norm_eps)
        x = mp_norm(mp_lerp(x, ffn_out, torch.sigmoid(self.blend_ffn), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        return x

    def forward_batched(
        self,
        x: torch.Tensor,
        entity_team_ids: torch.Tensor,
        team_ids: torch.Tensor,
        team_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Batched [E, M, d] forward sharing exact weights & math with forward()."""
        nc = self.norm_cfg
        xn = mp_norm(x, eps=nc.mp_norm_eps)
        v = mp_norm(self.val_proj(xn), eps=nc.mp_norm_eps)
        m_rows = x.shape[1]
        k_teams = team_ids.shape[1]
        inv_m = 1.0 / math.sqrt(max(1, m_rows))
        inv_k = 1.0 / math.sqrt(max(1, k_teams))

        r = mp_norm(self.global_metric(xn), eps=nc.mp_norm_eps)  # [E, M, r_gram]
        kv_global = torch.einsum("emr,emd->erd", r, v) * inv_m
        global_vals = mp_norm(torch.einsum("emr,erd->emd", r, kv_global), eps=nc.mp_norm_eps)

        mem_f = (entity_team_ids[:, :, None] == team_ids[:, None, :]).to(x.dtype)  # [E, M, K]
        pool_w = mem_f * torch.sigmoid(self.pool_score(xn))  # [E, M, K]
        team_sum = torch.einsum("emk,emd->ekd", pool_w, v)
        team_den = torch.clamp(pool_w.sum(dim=1).unsqueeze(-1), min=1e-4)
        team_pool = mp_norm(team_sum / team_den, eps=nc.mp_norm_eps)  # [E, K, d]

        if team_slice is not None and (team_slice[1] - team_slice[0]) == k_teams:
            t_cur = xn[:, team_slice[0] : team_slice[1]]
            t_in = mp_norm(mp_lerp(t_cur, team_pool, torch.sigmoid(self.blend_pool), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        else:
            t_in = team_pool

        q_t = mp_norm(self.team_metric(t_in), eps=nc.mp_norm_eps)
        k_t = mp_norm(self.rival_metric(t_in), eps=nc.mp_norm_eps)
        v_t = mp_norm(self.team_val_proj(t_in), eps=nc.mp_norm_eps)
        gram_t = torch.einsum("ekr,ejr->ekj", q_t, k_t) * inv_k
        t_mixed = mp_norm(torch.einsum("ekj,ejd->ekd", gram_t, v_t), eps=nc.mp_norm_eps)
        t_mut = mp_norm(
            mp_lerp(t_in, mp_norm(self.team_mutate_proj(t_mixed), eps=nc.mp_norm_eps), torch.sigmoid(self.blend_team_mut), eps=nc.mp_lerp_eps),
            eps=nc.mp_norm_eps,
        )

        t_bcast = mp_norm(self.team_bcast_proj(t_mut), eps=nc.mp_norm_eps)  # [E, K, d]
        has_team = mem_f.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        t_global = mp_norm(t_bcast.mean(dim=1, keepdim=True), eps=nc.mp_norm_eps)
        team_vals = mp_norm(has_team * torch.einsum("emk,ekd->emd", mem_f, t_bcast) + (1.0 - has_team) * t_global, eps=nc.mp_norm_eps)

        gram_merged = mp_norm(self.gram_out(mp_norm(global_vals + team_vals, eps=nc.mp_norm_eps)), eps=nc.mp_norm_eps)
        x = mp_norm(mp_lerp(xn, gram_merged, torch.sigmoid(self.blend_gram), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)

        hid = mp_silu(self.ffn_gate(x), scale=nc.mp_silu_scale) * self.ffn_up(x)
        self._last_ffn_act = hid.detach()
        ffn_out = mp_norm(self.ffn_down(hid), eps=nc.mp_norm_eps)
        x = mp_norm(mp_lerp(x, ffn_out, torch.sigmoid(self.blend_ffn), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        return x


class PlanWally(nn.Module):

    def __init__(
        self,
        d: int | None = None,
        depth: int | None = None,
        h: int | None = None,
        r_gram: int | None = None,
        r_dpp: int | None = None,
        recurrent: bool | None = None,
        use_fsq: bool | None = None,
        fsq_levels: tuple[int, ...] | None = None,
        use_mealy: bool | None = None,
        *,
        model_cfg: ModelConfig | None = None,
        norm_cfg: NormConfig | None = None,
    ):
        super().__init__()
        mc = model_cfg or DEFAULT_MODEL_CONFIG
        nc = norm_cfg or DEFAULT_NORM_CONFIG
        self.model_cfg = mc
        self.norm_cfg = nc

        self.d = int(mc.d if d is None else d)
        self.depth = max(1, int(mc.depth if depth is None else depth))
        self.h = int(mc.h if h is None else h)
        self.r_gram = int(mc.r_gram if r_gram is None else r_gram)
        self.r_dpp = int(mc.r_dpp if r_dpp is None else r_dpp)
        self.recurrent = bool(mc.recurrent if recurrent is None else recurrent)
        self.use_fsq = bool(mc.use_fsq if use_fsq is None else use_fsq)
        self.fsq_levels = tuple(int(v) for v in (mc.fsq_levels if fsq_levels is None else fsq_levels))
        self.use_mealy = bool(mc.use_mealy if use_mealy is None else use_mealy)
        self.temp_bounds = mc.temp_bounds

        eps = nc.unit_norm_eps
        self.enc_obs = MPLinear(OBS_WIDTH + BOT_GEO_DIM, self.d, unit_norm_eps=eps)
        self.enc_cart = MPLinear(18 + 4 + CART_GEO_DIM, self.d, unit_norm_eps=eps)
        self.enc_team = MPLinear(7 + 2, self.d, unit_norm_eps=eps)
        self.enc_evt = MPLinear(EVT_WIDTH, self.d, unit_norm_eps=eps)
        self.enc_track = MPLinear(TRACK_TOK_DIM, self.d, unit_norm_eps=eps)
        self.enc_cell = MPLinear(CELL_TOK_DIM, self.d, unit_norm_eps=eps)
        self.enc_item = MPLinear(ITEM_TOK_DIM, self.d, unit_norm_eps=eps)
        self.type_embed = nn.Parameter(mp_norm(torch.randn(6, self.d), eps=nc.mp_norm_eps))
        self.type_blend = nn.Parameter(torch.tensor(float(mc.type_blend_init_logit)))

        if self.use_mealy:
            self.mealy_gate = MPLinear(2 * self.d, self.d, unit_norm_eps=eps)
            self.mealy_cand = MPLinear(2 * self.d, self.d, unit_norm_eps=eps)
        else:
            self.mealy_gate = None
            self.mealy_cand = None

        self.fsq = (
            FSQBottleneck(
                self.d,
                levels=self.fsq_levels,
                in_gain_init=mc.fsq_in_gain_init,
                in_gain_bounds=mc.fsq_in_gain_bounds,
                blend_init_logit=mc.fsq_blend_init_logit,
                std_min=mc.fsq_std_min,
                norm_cfg=nc,
            )
            if self.use_fsq
            else None
        )

        init_blend = 1.0 / (self.depth + 1.0)
        if self.recurrent:
            self.blocks = nn.ModuleList([MPResCGTBlock(self.d, self.h, self.r_gram, self.r_dpp, init_blend=init_blend, norm_cfg=nc)])
            sb = float(mc.step_blend_init)
            self.step_blends = nn.Parameter(torch.full((self.depth,), math.log(sb / max(1e-4, 1.0 - sb))))
        else:
            self.blocks = nn.ModuleList([MPResCGTBlock(self.d, self.h, self.r_gram, self.r_dpp, init_blend=init_blend, norm_cfg=nc) for _ in range(self.depth)])
            self.step_blends = None

        self.cont_proj = MPLinear(self.d, PLAN_CONT_DIM, unit_norm_eps=eps)
        self.cont_gain = nn.Parameter(torch.full((PLAN_CONT_DIM,), float(mc.cont_gain_init)))
        self.cont_bias = nn.Parameter(torch.zeros(PLAN_CONT_DIM))
        with torch.no_grad():
            for idx_b, val_b in mc.cont_bias_init.items():
                if 0 <= int(idx_b) < PLAN_CONT_DIM:
                    self.cont_bias[int(idx_b)] = float(val_b)
        self.cont_logstd = nn.Parameter(torch.full((PLAN_CONT_DIM,), math.log(max(float(mc.cont_std_init), 1e-4))))

        self.cart_q = MPLinear(self.d, self.d, unit_norm_eps=eps)
        self.cart_k = MPLinear(self.d, self.d, unit_norm_eps=eps)
        self.rival_q = MPLinear(self.d, self.d, unit_norm_eps=eps)
        self.rival_k = MPLinear(self.d, self.d, unit_norm_eps=eps)
        self.cell_q = MPLinear(self.d, self.d, unit_norm_eps=eps)
        self.cell_k = MPLinear(self.d, self.d, unit_norm_eps=eps)
        log_t0 = math.log(max(float(mc.temp_init), 1e-4))
        self.log_temp_cart = nn.Parameter(torch.tensor(log_t0))
        self.log_temp_rival = nn.Parameter(torch.tensor(log_t0))
        self.log_temp_cell = nn.Parameter(torch.tensor(log_t0))
        self.log_cart_dist_bias = nn.Parameter(torch.tensor(math.log(max(float(mc.cart_dist_bias_init), 1e-4))))

        self.value_proj = MPLinear(self.d, 2, unit_norm_eps=eps)
        self.value_gain = nn.Parameter(torch.tensor(float(mc.value_gain_init)))
        self.value_bias = nn.Parameter(torch.zeros(2))
        self.value_priv_proj = MPLinear(3, 2, unit_norm_eps=eps)
        self.value_priv_gain = nn.Parameter(torch.tensor(float(mc.value_priv_gain_init)))
        self.skill_id_proj = MPLinear(self.d, 1, unit_norm_eps=eps)
        self.skill_id_gain = nn.Parameter(torch.tensor(float(mc.skill_id_gain_init)))
        self.skill_id_bias = nn.Parameter(torch.zeros(1))
        self.dyn_drift_proj = MPLinear(self.d, 1, unit_norm_eps=eps)
        self.dyn_drift_gain = nn.Parameter(torch.tensor(float(mc.dyn_drift_gain_init)))
        self.dyn_action_proj = MPLinear(self.d, PLAN_CONT_DIM, unit_norm_eps=eps)
        self.dyn_action_gain = nn.Parameter(torch.tensor(float(mc.dyn_action_gain_init)))

        project_unit_norm_(self, eps=eps)

    @classmethod
    def from_config(cls, model_cfg: ModelConfig, norm_cfg: NormConfig = DEFAULT_NORM_CONFIG) -> "PlanWally":
        return cls(model_cfg=model_cfg, norm_cfg=norm_cfg)

    def forward(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        events: torch.Tensor | None = None,
        geo: dict[str, torch.Tensor] | None = None,
        h_prev: torch.Tensor | None = None,
        world=None,
        priv_skill: torch.Tensor | None = None,
    ) -> PlanOutput:
        mc = self.model_cfg
        nc = self.norm_cfg
        n, j, k = obs.shape[0], carts.shape[0], teams.shape[0]
        device, dtype = obs.device, obs.dtype
        cart_valid = torch.ones(j, device=device, dtype=torch.bool)
        team_valid = torch.ones(k, device=device, dtype=torch.bool)
        cart_extra, team_extra = cgt_tensor_features(carts, teams, cart_valid, team_valid)

        tb = torch.sigmoid(self.type_blend)
        te = mp_norm(self.type_embed, eps=nc.mp_norm_eps)

        obs_pre = prenormalize_obs(obs, geo, norm_cfg=nc)
        bot_geo = mp_norm(torch.asinh(geo["bot_geo"]), eps=nc.mp_norm_eps) if (geo is not None and "bot_geo" in geo) else torch.zeros((n, BOT_GEO_DIM), device=device, dtype=dtype)
        obs_h = mp_lerp(mp_norm(self.enc_obs(torch.cat((obs_pre, bot_geo), dim=-1)), eps=nc.mp_norm_eps), te[0].unsqueeze(0), tb, eps=nc.mp_lerp_eps)

        if events is not None and events.shape[0] > 0:
            evt_h = mp_norm(self.enc_evt(mp_norm(torch.asinh(events), eps=nc.mp_norm_eps)), eps=nc.mp_norm_eps)
            evt_team = events[:, EVT["TEAM"]].to(torch.float32)
            obs_team = obs[:, OBS["TEAM"]].to(torch.float32)
            em = (obs_team[:, None] == evt_team[None, :]).to(dtype)
            ecnt = torch.clamp(em.sum(dim=1, keepdim=True), min=1.0)
            obs_h = mp_norm(
                mp_lerp(obs_h, mp_norm((em @ evt_h) / ecnt, eps=nc.mp_norm_eps), torch.tensor(mc.evt_blend, device=device, dtype=dtype), eps=nc.mp_lerp_eps),
                eps=nc.mp_norm_eps,
            )

        cart_pre = prenormalize_carts(carts, cart_extra, geo, norm_cfg=nc)
        cart_h = mp_lerp(mp_norm(self.enc_cart(cart_pre), eps=nc.mp_norm_eps), te[1].unsqueeze(0), tb, eps=nc.mp_lerp_eps)

        team_pre = prenormalize_teams(teams, team_extra, norm_cfg=nc)
        team_h = mp_lerp(mp_norm(self.enc_team(team_pre), eps=nc.mp_norm_eps), te[2].unsqueeze(0), tb, eps=nc.mp_lerp_eps)

        extra_tokens = []
        extra_teams = []

        if geo is not None and "track_tokens" in geo and geo["track_tokens"].shape[0] > 0:
            tr_h = mp_lerp(mp_norm(self.enc_track(mp_norm(torch.asinh(geo["track_tokens"]), eps=nc.mp_norm_eps)), eps=nc.mp_norm_eps), te[3].unsqueeze(0), tb, eps=nc.mp_lerp_eps)
            extra_tokens.append(tr_h)
            extra_teams.append(geo["track_teams"].to(torch.float32))

        cell_h = None
        if geo is not None and "cell_tokens" in geo and geo["cell_tokens"].shape[0] > 0:
            cell_h = mp_lerp(mp_norm(self.enc_cell(mp_norm(torch.asinh(geo["cell_tokens"]), eps=nc.mp_norm_eps)), eps=nc.mp_norm_eps), te[4].unsqueeze(0), tb, eps=nc.mp_lerp_eps)
            if "fovea_weights" in geo:
                obs_h = mp_norm(
                    mp_lerp(obs_h, mp_norm(geo["fovea_weights"] @ cell_h, eps=nc.mp_norm_eps), torch.tensor(mc.fovea_blend, device=device, dtype=dtype), eps=nc.mp_lerp_eps),
                    eps=nc.mp_norm_eps,
                )
            extra_tokens.append(cell_h)
            extra_teams.append(torch.zeros((cell_h.shape[0],), device=device, dtype=torch.float32))

        if geo is not None and "item_tokens" in geo and geo["item_tokens"].shape[0] > 0:
            it_h = mp_lerp(mp_norm(self.enc_item(mp_norm(torch.asinh(geo["item_tokens"]), eps=nc.mp_norm_eps)), eps=nc.mp_norm_eps), te[5].unsqueeze(0), tb, eps=nc.mp_lerp_eps)
            extra_tokens.append(it_h)
            extra_teams.append(torch.zeros((it_h.shape[0],), device=device, dtype=torch.float32))

        if self.mealy_gate is not None:
            h_in = obs_h.detach() if (h_prev is None or h_prev.shape != (n, self.d)) else mp_norm(h_prev, eps=nc.mp_norm_eps)
            cat_m = torch.cat((obs_h, h_in), dim=-1)
            z_gate = torch.sigmoid(self.mealy_gate(cat_m))
            cand = mp_norm(mp_silu(self.mealy_cand(cat_m), scale=nc.mp_silu_scale), eps=nc.mp_norm_eps)
            h_next = mp_norm(mp_lerp(h_in, cand, z_gate, eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
            obs_h = mp_norm(mp_lerp(obs_h, h_next, torch.tensor(mc.mealy_blend, device=device, dtype=dtype), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        else:
            h_next = obs_h.detach()

        seq_parts = [obs_h, cart_h, team_h] + extra_tokens
        team_parts = [
            obs[:, OBS["TEAM"]].to(torch.float32),
            carts[:, CS["CONTROL_TEAM"]].to(torch.float32),
            teams[:, TS["ID"]].to(torch.float32),
        ] + extra_teams
        x = mp_norm(torch.cat(seq_parts, dim=0), eps=nc.mp_norm_eps)
        entity_teams = torch.cat(team_parts, dim=0)
        team_ids = teams[:, TS["ID"]].to(torch.float32)

        team_slice = (n + j, n + j + k)
        if self.recurrent:
            blk = self.blocks[0]
            for step in range(self.depth):
                x_new = blk(x, entity_teams, team_ids, team_valid, world=world, team_slice=team_slice)
                x = mp_norm(mp_lerp(x, x_new, torch.sigmoid(self.step_blends[step]), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        else:
            for blk in self.blocks:
                x = blk(x, entity_teams, team_ids, team_valid, world=world, team_slice=team_slice)

        xn = mp_norm(x, eps=nc.mp_norm_eps)
        p_h = xn[:n]
        c_h = xn[n : n + j]
        t_h = xn[n + j : n + j + k]

        offset = n + j + k
        if geo is not None and "track_tokens" in geo:
            offset += geo["track_tokens"].shape[0]
        vcell_h = xn[offset : offset + cell_h.shape[0]] if cell_h is not None else c_h

        fsq_codes = None
        if self.fsq is not None:
            p_h, fsq_codes = self.fsq(p_h)

        g_cont = torch.clamp(self.cont_gain, -mc.cont_gain_bound, mc.cont_gain_bound)
        cont_mean = mc.cont_mean_scale * torch.tanh(g_cont * self.cont_proj(p_h) + self.cont_bias)
        cont_logstd = torch.clamp(self.cont_logstd, min=mc.cont_logstd_bounds[0], max=mc.cont_logstd_bounds[1]).unsqueeze(0).expand(n, -1)

        t_lo, t_hi = mc.temp_bounds
        temp_c = torch.clamp(torch.exp(self.log_temp_cart), t_lo, t_hi)
        temp_r = torch.clamp(torch.exp(self.log_temp_rival), t_lo, t_hi)
        temp_v = torch.clamp(torch.exp(self.log_temp_cell), t_lo, t_hi)

        p_pos = obs[:, OBS["POS_X"] : OBS["POS_Z"] + 1]
        c_pos = carts[:, CS["POS_X"] : CS["POS_Z"] + 1]
        d_pc = (p_pos[:, None, :] - c_pos[None, :, :]).square().sum(dim=-1).clamp_min(nc.mp_unit_vec_eps).sqrt()
        db_lo, db_hi = mc.cart_dist_bias_bounds
        dist_bias = torch.clamp(torch.exp(self.log_cart_dist_bias), db_lo, db_hi) * torch.log1p(d_pc / mc.cart_dist_scale)
        cart_logits = temp_c * (mp_unit_vec(self.cart_q(p_h), eps=nc.mp_unit_vec_eps) @ mp_unit_vec(self.cart_k(c_h), eps=nc.mp_unit_vec_eps).T) - dist_bias
        rival_logits = temp_r * (mp_unit_vec(self.rival_q(p_h), eps=nc.mp_unit_vec_eps) @ mp_unit_vec(self.rival_k(t_h), eps=nc.mp_unit_vec_eps).T)
        if k >= 2:
            own_mask = obs[:, OBS["TEAM"]].to(torch.float32)[:, None] == team_ids[None, :]
            rival_logits = rival_logits.masked_fill(own_mask, mc.rival_own_mask_fill)
        cell_logits = temp_v * (mp_unit_vec(self.cell_q(p_h), eps=nc.mp_unit_vec_eps) @ mp_unit_vec(self.cell_k(vcell_h), eps=nc.mp_unit_vec_eps).T)

        vg_lo, vg_hi = mc.value_gain_bounds
        vals = torch.clamp(self.value_gain, vg_lo, vg_hi) * self.value_proj(p_h) + self.value_bias
        if priv_skill is not None and priv_skill.shape[0] == n:
            ps = priv_skill.to(device=device, dtype=dtype)
            obs_t = obs[:, OBS["TEAM"]].to(torch.float32)
            t_mem = (obs_t[:, None] == team_ids[None, :]).to(dtype)
            t_cnt = torch.clamp(t_mem.sum(dim=0), min=1.0)
            t_mean = (t_mem.T @ ps) / t_cnt
            own_mean = t_mem @ t_mean
            if k > 1:
                rival_mask = obs_t[:, None] != team_ids[None, :]
                rival_vals = t_mean[None, :].expand(n, k).masked_fill(~rival_mask, mc.rival_own_mask_fill)
                rival_max = rival_vals.max(dim=-1).values
            else:
                rival_max = torch.zeros_like(ps)
            priv_feat = torch.stack((ps, own_mean, rival_max), dim=-1)
            vp_lo, vp_hi = mc.value_priv_gain_bounds
            vals = vals + torch.clamp(self.value_priv_gain, vp_lo, vp_hi) * self.value_priv_proj(priv_feat)

        sg_lo, sg_hi = mc.skill_id_gain_bounds
        dg_lo, dg_hi = mc.dyn_gain_bounds
        pred_skill = (torch.clamp(self.skill_id_gain, sg_lo, sg_hi) * self.skill_id_proj(p_h) + self.skill_id_bias).squeeze(-1)
        dyn_drift = (torch.clamp(self.dyn_drift_gain, dg_lo, dg_hi) * self.dyn_drift_proj(p_h)).squeeze(-1)
        dyn_action = torch.clamp(self.dyn_action_gain, dg_lo, dg_hi) * self.dyn_action_proj(p_h)
        return PlanOutput(
            cont_mean, cont_logstd, cart_logits, rival_logits, cell_logits,
            vals[:, 0], vals[:, 1], dyn_drift, dyn_action, h_next, fsq_codes, pred_skill,
        )

    def forward_batched(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        geo: dict[str, torch.Tensor] | None = None,
        h_prev: torch.Tensor | None = None,
    ) -> PlanOutput:
        """Zero-copy GPU batched forward over E simultaneous matches: obs[E,N,84], carts[E,J,18], teams[E,K,7]."""
        mc = self.model_cfg
        nc = self.norm_cfg
        e, n, _ = obs.shape
        j = carts.shape[1]
        k = teams.shape[1]
        device, dtype = obs.device, obs.dtype
        cart_extra, team_extra = cgt_tensor_features_batched(carts, teams)

        tb = torch.sigmoid(self.type_blend)
        te = mp_norm(self.type_embed, eps=nc.mp_norm_eps)

        # Batched prenormalize_obs
        xo = obs.clone()
        if geo is not None and "map_center" in geo and "map_radius" in geo:
            center = geo["map_center"].to(device, dtype)
            radius = torch.clamp(geo["map_radius"].to(device, dtype), min=nc.map_radius_min)
            xo[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1] = (xo[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1] - center[None, None, :]) / radius
        else:
            xo[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1] = xo[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1] / nc.pos_fallback_scale
        xo[:, :, OBS["VEL_X"] : OBS["VEL_Z"] + 1] = xo[:, :, OBS["VEL_X"] : OBS["VEL_Z"] + 1] / nc.vel_scale
        xo[:, :, OBS["HEALTH"]] = (xo[:, :, OBS["HEALTH"]] - nc.health_offset) / nc.health_scale
        xo[:, :, OBS["ARMOR"]] = (xo[:, :, OBS["ARMOR"]] - nc.armor_offset) / nc.armor_scale
        xo[:, :, OBS["AMMO_SHELLS"] : OBS["AMMO_FUEL"] + 1] = xo[:, :, OBS["AMMO_SHELLS"] : OBS["AMMO_FUEL"] + 1] / nc.ammo_scale
        for col in (
            "ENGINE_TIME", "SPAWN_TIME", "STRENGTH_FINISHED", "RESPONSE_SEQ", "RESPONSE_TIME",
            "APPLIED_SEQUENCE", "OUTCOME_A_SEQ", "OUTCOME_B_SEQ", "CELL_X", "CELL_Y",
            "SWIZZLE_SCHEDULED_TIME", "SWIZZLE_ACTUAL_TIME", *BOT_CONFIGURATION_COLUMNS,
        ):
            if col == "STRENGTH_FINISHED":
                rem = torch.clamp(obs[:, :, OBS["STRENGTH_FINISHED"]] - obs[:, :, OBS["ENGINE_TIME"]], min=0.0, max=nc.strength_max_s) / nc.strength_max_s
                xo[:, :, OBS[col]] = rem
            else:
                xo[:, :, OBS[col]] = 0.0
        obs_pre = mp_norm(torch.asinh(xo), eps=nc.mp_norm_eps)
        bot_geo = torch.zeros((e, n, BOT_GEO_DIM), device=device, dtype=dtype)
        obs_h = mp_lerp(mp_norm(self.enc_obs(torch.cat((obs_pre, bot_geo), dim=-1)), eps=nc.mp_norm_eps), te[0].view(1, 1, -1), tb, eps=nc.mp_lerp_eps)

        # Batched prenormalize_carts
        xc = carts.clone()
        plen = torch.clamp(xc[:, :, CS["PATH_LENGTH"]], min=1.0)
        xc[:, :, CS["PATH_POSITION"]] = xc[:, :, CS["PATH_POSITION"]] / plen
        xc[:, :, CS["PATH_LENGTH"]] = torch.log1p(plen / nc.cart_len_scale)
        xc[:, :, CS["SPEED"]] = xc[:, :, CS["SPEED"]] / nc.cart_speed_scale
        if geo is not None and "map_center" in geo and "map_radius" in geo:
            center = geo["map_center"].to(device, dtype)
            radius = torch.clamp(geo["map_radius"].to(device, dtype), min=nc.map_radius_min)
            xc[:, :, CS["POS_X"] : CS["POS_Z"] + 1] = (xc[:, :, CS["POS_X"] : CS["POS_Z"] + 1] - center[None, None, :]) / radius
        else:
            xc[:, :, CS["POS_X"] : CS["POS_Z"] + 1] = xc[:, :, CS["POS_X"] : CS["POS_Z"] + 1] / nc.pos_fallback_scale
        c_geo = torch.zeros((e, j, CART_GEO_DIM), device=device, dtype=dtype)
        cart_pre = mp_norm(torch.asinh(torch.cat((xc, cart_extra, c_geo), dim=-1)), eps=nc.mp_norm_eps)
        cart_h = mp_lerp(mp_norm(self.enc_cart(cart_pre), eps=nc.mp_norm_eps), te[1].view(1, 1, -1), tb, eps=nc.mp_lerp_eps)

        # Batched prenormalize_teams
        xt = teams.clone()
        lim = torch.clamp(xt[:, :, TS["SCORE_LIMIT"]], min=1.0)
        xt[:, :, TS["SCORE"]] = xt[:, :, TS["SCORE"]] / lim
        xt[:, :, TS["SCORE_LIMIT"]] = 1.0
        xt[:, :, TS["EPISODE"]] = 0.0
        team_pre = mp_norm(torch.asinh(torch.cat((xt, team_extra), dim=-1)), eps=nc.mp_norm_eps)
        team_h = mp_lerp(mp_norm(self.enc_team(team_pre), eps=nc.mp_norm_eps), te[2].view(1, 1, -1), tb, eps=nc.mp_lerp_eps)

        if self.mealy_gate is not None:
            h_in = obs_h.detach() if (h_prev is None or h_prev.shape != (e, n, self.d)) else mp_norm(h_prev, eps=nc.mp_norm_eps)
            cat_m = torch.cat((obs_h, h_in), dim=-1)
            z_gate = torch.sigmoid(self.mealy_gate(cat_m))
            cand = mp_norm(mp_silu(self.mealy_cand(cat_m), scale=nc.mp_silu_scale), eps=nc.mp_norm_eps)
            h_next = mp_norm(mp_lerp(h_in, cand, z_gate, eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
            obs_h = mp_norm(mp_lerp(obs_h, h_next, torch.tensor(mc.mealy_blend, device=device, dtype=dtype), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        else:
            h_next = obs_h.detach()

        x = mp_norm(torch.cat((obs_h, cart_h, team_h), dim=1), eps=nc.mp_norm_eps)  # [E, N+J+K, d]
        entity_teams = torch.cat(
            (
                obs[:, :, OBS["TEAM"]].to(torch.float32),
                carts[:, :, CS["CONTROL_TEAM"]].to(torch.float32),
                teams[:, :, TS["ID"]].to(torch.float32),
            ),
            dim=1,
        )
        team_ids = teams[:, :, TS["ID"]].to(torch.float32)
        team_slice = (n + j, n + j + k)

        if self.recurrent:
            blk = self.blocks[0]
            for step in range(self.depth):
                x_new = blk.forward_batched(x, entity_teams, team_ids, team_slice=team_slice)
                x = mp_norm(mp_lerp(x, x_new, torch.sigmoid(self.step_blends[step]), eps=nc.mp_lerp_eps), eps=nc.mp_norm_eps)
        else:
            for blk in self.blocks:
                x = blk.forward_batched(x, entity_teams, team_ids, team_slice=team_slice)

        xn = mp_norm(x, eps=nc.mp_norm_eps)
        p_h = xn[:, :n]
        c_h = xn[:, n : n + j]
        t_h = xn[:, n + j : n + j + k]

        fsq_codes = None
        if self.fsq is not None:
            p_h, fsq_codes = self.fsq(p_h)

        g_cont = torch.clamp(self.cont_gain, -mc.cont_gain_bound, mc.cont_gain_bound)
        cont_mean = mc.cont_mean_scale * torch.tanh(g_cont * self.cont_proj(p_h) + self.cont_bias)
        cont_logstd = torch.clamp(self.cont_logstd, min=mc.cont_logstd_bounds[0], max=mc.cont_logstd_bounds[1]).view(1, 1, -1).expand(e, n, -1)

        t_lo, t_hi = mc.temp_bounds
        temp_c = torch.clamp(torch.exp(self.log_temp_cart), t_lo, t_hi)
        temp_r = torch.clamp(torch.exp(self.log_temp_rival), t_lo, t_hi)
        temp_v = torch.clamp(torch.exp(self.log_temp_cell), t_lo, t_hi)

        p_pos = obs[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1]
        c_pos = carts[:, :, CS["POS_X"] : CS["POS_Z"] + 1]
        d_pc = (p_pos[:, :, None, :] - c_pos[:, None, :, :]).square().sum(dim=-1).clamp_min(nc.mp_unit_vec_eps).sqrt()
        db_lo, db_hi = mc.cart_dist_bias_bounds
        dist_bias = torch.clamp(torch.exp(self.log_cart_dist_bias), db_lo, db_hi) * torch.log1p(d_pc / mc.cart_dist_scale)
        q_c = mp_unit_vec(self.cart_q(p_h), eps=nc.mp_unit_vec_eps)
        k_c = mp_unit_vec(self.cart_k(c_h), eps=nc.mp_unit_vec_eps)
        cart_logits = temp_c * torch.einsum("end,ejd->enj", q_c, k_c) - dist_bias

        q_r = mp_unit_vec(self.rival_q(p_h), eps=nc.mp_unit_vec_eps)
        k_r = mp_unit_vec(self.rival_k(t_h), eps=nc.mp_unit_vec_eps)
        rival_logits = temp_r * torch.einsum("end,ekd->enk", q_r, k_r)
        if k >= 2:
            own_mask = obs[:, :, OBS["TEAM"]].to(torch.float32)[:, :, None] == team_ids[:, None, :]
            rival_logits = rival_logits.masked_fill(own_mask, mc.rival_own_mask_fill)

        q_v = mp_unit_vec(self.cell_q(p_h), eps=nc.mp_unit_vec_eps)
        k_v = mp_unit_vec(self.cell_k(c_h), eps=nc.mp_unit_vec_eps)
        cell_logits = temp_v * torch.einsum("end,ejd->enj", q_v, k_v)

        vg_lo, vg_hi = mc.value_gain_bounds
        vals = torch.clamp(self.value_gain, vg_lo, vg_hi) * self.value_proj(p_h) + self.value_bias
        sg_lo, sg_hi = mc.skill_id_gain_bounds
        dg_lo, dg_hi = mc.dyn_gain_bounds
        pred_skill = (torch.clamp(self.skill_id_gain, sg_lo, sg_hi) * self.skill_id_proj(p_h) + self.skill_id_bias).squeeze(-1)
        dyn_drift = (torch.clamp(self.dyn_drift_gain, dg_lo, dg_hi) * self.dyn_drift_proj(p_h)).squeeze(-1)
        dyn_action = torch.clamp(self.dyn_action_gain, dg_lo, dg_hi) * self.dyn_action_proj(p_h)
        return PlanOutput(
            cont_mean, cont_logstd, cart_logits, rival_logits, cell_logits,
            vals[:, :, 0], vals[:, :, 1], dyn_drift, dyn_action, h_next, fsq_codes, pred_skill,
        )

    def spec(self) -> dict:
        return {
            "d": self.d,
            "depth": self.depth,
            "h": self.h,
            "r_gram": self.r_gram,
            "r_dpp": self.r_dpp,
            "recurrent": self.recurrent,
            "use_fsq": self.use_fsq,
            "fsq_levels": list(self.fsq_levels) if self.use_fsq else [],
            "use_mealy": self.use_mealy,
            "edm2_mp": True,
            "hierarchical_team_gram": True,
        }


def sample_plan(out: PlanOutput, deterministic: bool = False) -> PlanSample:
    if deterministic:
        cont = out.cont_mean
        cart_pick = out.cart_logits.argmax(dim=-1)
        rival_pick = out.rival_logits.argmax(dim=-1)
        cell_pick = out.cell_logits.argmax(dim=-1)
    else:
        std = torch.exp(out.cont_logstd)
        cont = out.cont_mean + std * torch.randn_like(out.cont_mean)
        cart_pick = torch.distributions.Categorical(logits=out.cart_logits).sample()
        rival_pick = torch.distributions.Categorical(logits=out.rival_logits).sample()
        cell_pick = torch.distributions.Categorical(logits=out.cell_logits).sample()
    logp = plan_logp(out, cont, cart_pick, rival_pick, cell_pick)
    return PlanSample(cont, cart_pick, rival_pick, cell_pick, logp)


def pack_plan_rows(
    obs_rows: np.ndarray,
    team_rows: np.ndarray,
    sample: PlanSample,
    arms: np.ndarray,
    active_arms: set[str],
    maxclients: int = 16,
    cell_centroids: np.ndarray | None = None,
    cart_lengths: np.ndarray | None = None,
    pack_cfg: ActionPackConfig = DEFAULT_ACTION_PACK_CONFIG,
) -> np.ndarray:
    out = np.zeros((maxclients, PLAN_WIDTH), dtype=np.float32)
    cont_np = sample.cont.detach().cpu().numpy()
    cart_np = sample.cart_pick.detach().cpu().numpy()
    rival_np = sample.rival_pick.detach().cpu().numpy()
    cell_np = sample.cell_pick.detach().cpu().numpy()
    team_ids = team_rows[:, TS["ID"]].astype(np.float32)
    nw_lo, nw_hi = pack_cfg.nav_w_clip
    for i in range(len(obs_rows)):
        if str(arms[i]) not in active_arms:
            continue
        eid = int(round(float(obs_rows[i, OBS["ID"]])))
        if 1 <= eid <= maxclients:
            r_idx = int(rival_np[i])
            focus_team = float(team_ids[r_idx]) if 0 <= r_idx < len(team_ids) else 0.0
            c_idx = int(cell_np[i])
            if cell_centroids is not None and 0 <= c_idx < len(cell_centroids):
                nav_x, nav_y = float(cell_centroids[c_idx, 0]), float(cell_centroids[c_idx, 1])
                nav_w = float(np.log1p(np.exp(np.clip(cont_np[i, 8] - pack_cfg.nav_w_shift, nw_lo, nw_hi))))
            else:
                nav_x = nav_y = nav_w = 0.0
            cpick = int(cart_np[i])
            u_ds = float(cont_np[i, 7])
            cart_ds = float(
                np.clip(
                    math.copysign(pack_cfg.cart_ds_scale * (abs(u_ds) ** pack_cfg.cart_ds_power), u_ds),
                    -pack_cfg.cart_ds_clip,
                    pack_cfg.cart_ds_clip,
                )
            )
            posture = float(np.tanh(cont_np[i, 10]))
            out[eid - 1, 0] = float(eid)
            out[eid - 1, 1] = float(cpick)
            out[eid - 1, 2:7] = cont_np[i, 0:5]
            out[eid - 1, 7] = focus_team
            out[eid - 1, 8] = float(cont_np[i, 5])
            out[eid - 1, 9] = float(cont_np[i, 6])
            out[eid - 1, 10] = cart_ds
            out[eid - 1, 11] = nav_x
            out[eid - 1, 12] = nav_y
            out[eid - 1, 13] = nav_w
            out[eid - 1, 14] = float(cont_np[i, 9])
            out[eid - 1, 15] = posture
    return out


@torch.no_grad()
def pack_plan_tensor(
    obs: torch.Tensor,
    teams: torch.Tensor,
    sample: PlanSample,
    active_mask: torch.Tensor | None = None,
    pack_cfg: ActionPackConfig = DEFAULT_ACTION_PACK_CONFIG,
) -> torch.Tensor:
    """Zero-copy on-device [E, N, 16] PLAN packer matching pack_plan_rows bit-for-bit."""
    e, n, _ = obs.shape
    device = obs.device
    out = torch.zeros((e, n, PLAN_WIDTH), device=device, dtype=torch.float32)
    team_ids = teams[:, :, TS["ID"]].to(torch.float32)  # [E, K]
    r_idx = sample.rival_pick.long().clamp(0, max(0, team_ids.shape[1] - 1))
    focus_team = team_ids.gather(1, r_idx)
    u_ds = sample.cont[:, :, 7]
    cart_ds = (torch.sign(u_ds) * float(pack_cfg.cart_ds_scale) * (u_ds.abs().clamp_min(1e-8) ** float(pack_cfg.cart_ds_power))).clamp(
        -float(pack_cfg.cart_ds_clip), float(pack_cfg.cart_ds_clip)
    )
    posture = torch.tanh(sample.cont[:, :, 10])
    out[:, :, 0] = obs[:, :, OBS["ID"]]
    out[:, :, 1] = sample.cart_pick.to(torch.float32)
    out[:, :, 2:7] = sample.cont[:, :, 0:5]
    out[:, :, 7] = focus_team
    out[:, :, 8] = sample.cont[:, :, 5]
    out[:, :, 9] = sample.cont[:, :, 6]
    out[:, :, 10] = cart_ds
    out[:, :, 14] = sample.cont[:, :, 9]
    out[:, :, 15] = posture
    if active_mask is not None:
        m = active_mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
        out = torch.where(m, out, torch.zeros_like(out))
    return out


def plan_module_group(param_name: str) -> str:
    root = param_name.split(".", 1)[0]
    if root in (
        "enc_obs", "enc_cart", "enc_team", "enc_evt", "enc_track", "enc_cell", "enc_item",
        "type_embed", "type_blend", "mealy_gate", "mealy_cand",
    ):
        return "encoder"
    if root == "blocks":
        if any(k in param_name for k in ("ffn_", "blend_ffn")):
            return "resnet_ffn"
        return "gram_cgt"
    if root in ("step_blends", "fsq"):
        return "resnet_ffn"
    if root in (
        "cont_proj", "cont_gain", "cont_bias", "cont_logstd",
        "cart_q", "cart_k", "rival_q", "rival_k", "cell_q", "cell_k",
        "log_temp_cart", "log_temp_rival", "log_temp_cell", "log_cart_dist_bias",
    ):
        return "action_head"
    if root in ("value_proj", "value_gain", "value_bias", "value_priv_proj", "value_priv_gain"):
        return "value_head"
    if root in ("skill_id_proj", "skill_id_gain", "skill_id_bias"):
        return "skill_id_head"
    if root in ("dyn_drift_proj", "dyn_drift_gain", "dyn_action_proj", "dyn_action_gain"):
        return "dynamics_head"
    return "other"


def probe_plan_cross_server_mccandlish(
    model: PlanWally,
    transitions: list[PlanTransition],
    *,
    device: str = "cpu",
    max_per_server: int = 16,
    loss_cfg: LossConfig = DEFAULT_LOSS_CONFIG,
    components: tuple[str, ...] = ("total", "actor_pg", "critic_winnie", "critic_lou", "critic_both", "dynamics"),
) -> dict:
    by_srv: dict[int, list[PlanTransition]] = {}
    for tr in transitions:
        by_srv.setdefault(int(tr.session), []).append(tr)
    servers = [s for s, lst in sorted(by_srv.items()) if lst]
    if len(servers) < 2:
        return {"error": f"need >=2 servers, got {len(servers)}"}

    per_srv = min(max_per_server, min(len(by_srv[s]) for s in servers))
    sample_all = [tr for s in servers for tr in by_srv[s][-per_srv:]]
    adv_mean, adv_std = compute_batch_adv_stats(model, sample_all, device=device, loss_cfg=loss_cfg)

    params = list(model.named_parameters())
    names = [n for n, _ in params]
    tensors = [p for _, p in params]

    rep_grads = {c: [] for c in components}
    all_diags = []
    for s in servers:
        sub = by_srv[s][-per_srv:]
        sums = {c: {n: None for n in names} for c in components}
        for tr in sub:
            comps, diag = plan_losses_for_transition(
                model, tr, device=device, loss_cfg=loss_cfg, adv_mean=adv_mean, adv_std=adv_std,
            )
            all_diags.append(diag)
            active = [c for c in components if comps[c].requires_grad]
            for idx, c in enumerate(active):
                gs = torch.autograd.grad(comps[c] / len(sub), tensors, retain_graph=(idx + 1 < len(active)), allow_unused=True)
                acc = sums[c]
                for n, g in zip(names, gs):
                    if g is None:
                        continue
                    gc = g.detach().cpu().to(torch.float32)
                    acc[n] = gc if acc[n] is None else (acc[n] + gc)
        for c in components:
            rep_grads[c].append(sums[c])

    groups = ("encoder", "gram_cgt", "resnet_ffn", "action_head", "value_head", "dynamics_head")
    by_comp = {}
    by_comp_grp = {}
    for c in components:
        pair = mccandlish_pair_from_grads(rep_grads[c], b0_transitions=float(per_srv), b0_rows=float(per_srv * 9), generation=0)
        by_comp[c] = summarize_mccandlish_pairs([pair]) if pair is not None else None
        g_out = {}
        for grp in groups:
            sub_reps = [
                {n: g for n, g in rg.items() if g is not None and plan_module_group(n) == grp}
                for rg in rep_grads[c]
            ]
            if not any(sub_reps):
                continue
            gp = mccandlish_pair_from_grads(sub_reps, b0_transitions=float(per_srv), b0_rows=float(per_srv * 9), generation=0)
            if gp is not None:
                g_out[grp] = summarize_mccandlish_pairs([gp])
        by_comp_grp[c] = g_out

    def _m(k):
        vals = [d[k] for d in all_diags if d.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "spec": model.spec(),
        "param_count": sum(p.numel() for p in model.parameters()),
        "weight_norms": weight_norm_report(model),
        "servers": len(servers),
        "transitions_per_server": per_srv,
        "by_component": by_comp,
        "by_component_and_group": by_comp_grp,
        "cross_cos_actor_critic": float(np.mean([
            c for g_a, g_c in zip(rep_grads["actor_pg"], rep_grads["critic_both"])
            if (c := grad_cosine(g_a, g_c)) is not None
        ])) if rep_grads["actor_pg"] else None,
        "diagnostics": {
            k: _m(k)
            for k in (
                "loss_total", "loss_actor", "loss_w", "loss_l", "loss_dyn", "entropy",
                "is_ratio_mean", "is_clip_frac", "winnie_mse", "lou_mse", "reward_abs",
                "cart_temp", "cell_temp",
            )
        },
    }


@torch.no_grad()
def ema_update(ema_model: PlanWally, live_model: PlanWally, decay: float = DEFAULT_OPTIMIZER_CONFIG.ema_decay) -> None:
    for p_ema, p_live in zip(ema_model.parameters(), live_model.parameters()):
        p_ema.mul_(decay).add_(p_live, alpha=1.0 - decay)
    project_unit_norm_(ema_model)


def clone_plan_model(model: PlanWally) -> PlanWally:
    return copy.deepcopy(model)
