from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from payload.tools.strategy_io_schema import CS, EVT, EVT_WIDTH, OBS, OBS_WIDTH, TS

from .dpp import dpp_gram_operator
from .game_value import cgt_tensor_features
from .grad_snr import (
    grad_cosine,
    mccandlish_pair_from_grads,
    summarize_mccandlish_pairs,
)
from .policy_contract import POLICY_LOG_RATIO_BOUND, POLICY_RATIO_CLIP
from .policy_math import clipped_policy_surrogate, norm

PLAN_WIDTH = 16
PLAN_CONT_DIM = 11
BOT_GEO_DIM = 12
CART_GEO_DIM = 10
TRACK_TOK_DIM = 10
CELL_TOK_DIM = 10
ITEM_TOK_DIM = 9

# Continuous parametric flow slots:
# 0: w0, 1: w1, 2: w2, 3: w3 (cart_scale), 4: w4 (enemy_scale),
# 5: item_w, 6: pwr_w (powerups/mega/big-armor + respawn timing),
# 7: cart_ds_norm (*500u lead/lag along cart spline),
# 8: nav_w_raw (softplus(x - 0.8) Voronoi spatial flow weight),
# 9: atk_cyl (cart-cylinder combat priority in havocbot_chooseenemy),
# 10: posture (tanh -> local combat steering in havocbot_movetogoal).


class FSQBottleneck(nn.Module):
    """Finite Scalar Quantization bottleneck with straight-through estimator for sampleable
    structured parametric flow codes (zero codebook collapse)."""

    def __init__(self, d: int, levels: tuple[int, ...] = (5, 5, 5, 5)):
        super().__init__()
        self.levels = tuple(int(v) for v in levels)
        self.proj_in = nn.Linear(d, len(self.levels), bias=True)
        self.proj_out = nn.Linear(len(self.levels), d, bias=False)
        half = torch.tensor([(v - 1) / 2.0 for v in self.levels], dtype=torch.float32)
        self.register_buffer("half_levels", half)
        nn.init.normal_(self.proj_out.weight, std=0.05)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(self.proj_in(x)) * self.half_levels
        q = torch.round(z)
        z_ste = z + (q - z).detach()
        norm_q = z_ste / torch.clamp(self.half_levels, min=1.0)
        return x + self.proj_out(norm_q), q.detach()


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


class PlanSample(NamedTuple):
    cont: torch.Tensor
    cart_pick: torch.Tensor
    rival_pick: torch.Tensor
    cell_pick: torch.Tensor
    logp: torch.Tensor


class ResCGTBlock(nn.Module):
    """Pre-norm depthwise residual CGT block combining Team + Rival + DPP Gram interaction
    with a SwiGLU FFN and learnable residual scales (variance-preserving 1/sqrt(2*depth) init)."""

    def __init__(self, d: int = 128, h: int = 256, r_gram: int = 64, r_dpp: int = 32, init_scale: float = 0.5):
        super().__init__()
        self.d = d
        self.r_gram = r_gram
        self.team_metric = nn.Linear(d, r_gram, bias=False)
        self.rival_metric = nn.Linear(d, r_gram, bias=False)
        self.val_proj = nn.Linear(d, d, bias=False)
        self.dpp_metric = nn.Linear(d, r_dpp, bias=False)
        self.dpp_quality = nn.Linear(d, 1, bias=False)
        self.dpp_out = nn.Linear(d, d, bias=False)
        self.gram_out = nn.Linear(d, d, bias=False)

        self.ffn_gate = nn.Linear(d, h, bias=False)
        self.ffn_up = nn.Linear(d, h, bias=False)
        self.ffn_down = nn.Linear(h, d, bias=False)

        self.alpha_gram = nn.Parameter(torch.tensor(float(init_scale)))
        self.alpha_ffn = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(
        self,
        x: torch.Tensor,
        entity_team_ids: torch.Tensor,
        team_ids: torch.Tensor,
        team_valid: torch.Tensor,
    ) -> torch.Tensor:
        xn = norm(x)
        v = norm(self.val_proj(xn))
        rivals = self.rival_metric(xn)
        global_ctx = (rivals.T @ v) / math.sqrt(max(1, x.shape[0]))
        global_vals = (rivals @ global_ctx) / math.sqrt(self.r_gram)

        teams = self.team_metric(xn)
        mem = (entity_team_ids[:, None] == team_ids[None, :]) & team_valid[None, :]
        mem_f = mem.to(x.dtype)
        tv = (teams[:, :, None] * v[:, None, :]).reshape(x.shape[0], -1)
        team_sum = mem_f.T @ tv
        team_cnt = torch.clamp(mem_f.sum(dim=0, keepdim=True).T, min=1.0)
        team_mean = team_sum / team_cnt
        ctx = (mem_f @ team_mean).reshape(x.shape[0], self.r_gram, self.d)
        team_vals = (teams[:, :, None] * ctx).sum(dim=1) / math.sqrt(self.r_gram)

        dpp_basis = F.softplus(self.dpp_quality(xn)) * norm(self.dpp_metric(xn))
        dpp_ctx = self.dpp_out(dpp_gram_operator(dpp_basis, v))

        x = x + self.alpha_gram * self.gram_out(norm(global_vals + team_vals + dpp_ctx))

        xn2 = norm(x)
        ffn = self.ffn_down(F.silu(self.ffn_gate(xn2)) * self.ffn_up(xn2))
        x = x + self.alpha_ffn * ffn
        return x


class PlanWally(nn.Module):
    """Voronoi-foveated, cart-spline-aware recurrent CGT controller emitting 16-float
    parametric flow PLAN rows (cart lead/lag, Voronoi spatial flow, powerup/item flow,
    cart-cylinder combat focus, and local steering posture) with optional FSQ bottleneck."""

    def __init__(
        self,
        d: int = 128,
        depth: int = 4,
        h: int = 256,
        r_gram: int = 64,
        r_dpp: int = 32,
        recurrent: bool = True,
        use_fsq: bool = False,
        fsq_levels: tuple[int, ...] = (5, 5, 5, 5),
        use_mealy: bool = True,
    ):
        super().__init__()
        self.d = int(d)
        self.depth = max(1, int(depth))
        self.h = int(h)
        self.r_gram = int(r_gram)
        self.r_dpp = int(r_dpp)
        self.recurrent = bool(recurrent)
        self.use_fsq = bool(use_fsq)
        self.fsq_levels = tuple(int(v) for v in fsq_levels)
        self.use_mealy = bool(use_mealy)

        self.enc_obs = nn.Linear(OBS_WIDTH + BOT_GEO_DIM, d, bias=False)
        self.enc_cart = nn.Linear(18 + 4 + CART_GEO_DIM, d, bias=False)
        self.enc_team = nn.Linear(7 + 2, d, bias=False)
        self.enc_evt = nn.Linear(EVT_WIDTH, d, bias=False)
        self.enc_track = nn.Linear(TRACK_TOK_DIM, d, bias=False)
        self.enc_cell = nn.Linear(CELL_TOK_DIM, d, bias=False)
        self.enc_item = nn.Linear(ITEM_TOK_DIM, d, bias=False)
        # Token types: 0=obs, 1=cart, 2=team, 3=track_spline, 4=voronoi_cell, 5=major_item
        self.type_embed = nn.Parameter(torch.randn(6, d) * 0.05)

        self.mealy_gru = nn.GRUCell(d, d) if self.use_mealy else None
        self.fsq = FSQBottleneck(d, levels=self.fsq_levels) if self.use_fsq else None

        init_scale = 1.0 / math.sqrt(2.0 * self.depth)
        if self.recurrent:
            self.blocks = nn.ModuleList([ResCGTBlock(d, h, r_gram, r_dpp, init_scale=init_scale)])
            self.step_scales = nn.Parameter(torch.ones(self.depth))
        else:
            self.blocks = nn.ModuleList([ResCGTBlock(d, h, r_gram, r_dpp, init_scale=init_scale) for _ in range(self.depth)])
            self.step_scales = None

        self.cont_head = nn.Linear(d, PLAN_CONT_DIM, bias=True)
        with torch.no_grad():
            self.cont_head.weight.mul_(0.1)
            # Decision-boundary behavioral prior: mild cart-defender & powerup readiness, soft Voronoi weight
            self.cont_head.bias.zero_()
            self.cont_head.bias[6] = 0.15   # pwr_w mild positive readiness
            self.cont_head.bias[8] = -0.4   # nav_w_raw -> softplus(-1.2) ~= 0.26
            self.cont_head.bias[9] = 0.25   # atk_cyl mild cart-pusher combat focus
        self.cont_logstd = nn.Parameter(torch.full((PLAN_CONT_DIM,), math.log(0.42)))

        self.cart_q = nn.Linear(d, d, bias=False)
        self.cart_k = nn.Linear(d, d, bias=False)
        self.rival_q = nn.Linear(d, d, bias=False)
        self.rival_k = nn.Linear(d, d, bias=False)
        self.cell_q = nn.Linear(d, d, bias=False)
        self.cell_k = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.cart_q.weight.mul_(0.25)
            self.rival_q.weight.mul_(0.25)
            self.cell_q.weight.mul_(0.25)

        self.value_head = nn.Linear(d, 2, bias=True)
        nn.init.zeros_(self.value_head.bias)
        self.dyn_drift = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.dyn_drift.bias)
        self.dyn_action = nn.Linear(d, PLAN_CONT_DIM, bias=False)

    def forward(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        events: torch.Tensor | None = None,
        geo: dict[str, torch.Tensor] | None = None,
        h_prev: torch.Tensor | None = None,
    ) -> PlanOutput:
        n, j, k = obs.shape[0], carts.shape[0], teams.shape[0]
        device, dtype = obs.device, obs.dtype
        cart_valid = torch.ones(j, device=device, dtype=torch.bool)
        team_valid = torch.ones(k, device=device, dtype=torch.bool)
        cart_extra, team_extra = cgt_tensor_features(carts, teams, cart_valid, team_valid)

        bot_geo = geo["bot_geo"] if (geo is not None and "bot_geo" in geo) else torch.zeros((n, BOT_GEO_DIM), device=device, dtype=dtype)
        cart_geo = geo["cart_geo"] if (geo is not None and "cart_geo" in geo) else torch.zeros((j, CART_GEO_DIM), device=device, dtype=dtype)

        obs_h = norm(self.enc_obs(torch.asinh(torch.cat((obs, bot_geo), dim=-1)))) + self.type_embed[0]
        if events is not None and events.shape[0] > 0:
            evt_h = self.enc_evt(torch.asinh(events))
            evt_team = events[:, EVT["TEAM"]].to(torch.float32)
            obs_team = obs[:, OBS["TEAM"]].to(torch.float32)
            em = (obs_team[:, None] == evt_team[None, :]).to(dtype)
            ecnt = torch.clamp(em.sum(dim=1, keepdim=True), min=1.0)
            obs_h = obs_h + 0.25 * norm((em @ evt_h) / ecnt)

        cart_h = norm(self.enc_cart(torch.asinh(torch.cat((carts, cart_extra, cart_geo), dim=-1)))) + self.type_embed[1]
        team_h = norm(self.enc_team(torch.asinh(torch.cat((teams, team_extra), dim=-1)))) + self.type_embed[2]

        # Literal cart-spline, Voronoi cell, and typed major-item tokens
        extra_tokens = []
        extra_teams = []

        if geo is not None and "track_tokens" in geo and geo["track_tokens"].shape[0] > 0:
            tr_h = norm(self.enc_track(geo["track_tokens"])) + self.type_embed[3]
            extra_tokens.append(tr_h)
            extra_teams.append(geo["track_teams"].to(torch.float32))

        cell_h = None
        if geo is not None and "cell_tokens" in geo and geo["cell_tokens"].shape[0] > 0:
            cell_h = norm(self.enc_cell(geo["cell_tokens"])) + self.type_embed[4]
            if "fovea_weights" in geo:
                obs_h = obs_h + 0.35 * norm(geo["fovea_weights"] @ cell_h)
            extra_tokens.append(cell_h)
            extra_teams.append(torch.zeros((cell_h.shape[0],), device=device, dtype=torch.float32))

        if geo is not None and "item_tokens" in geo and geo["item_tokens"].shape[0] > 0:
            it_h = norm(self.enc_item(geo["item_tokens"])) + self.type_embed[5]
            extra_tokens.append(it_h)
            extra_teams.append(torch.zeros((it_h.shape[0],), device=device, dtype=torch.float32))

        if self.mealy_gru is not None:
            if h_prev is None or h_prev.shape != (n, self.d):
                h_in = torch.zeros((n, self.d), device=device, dtype=dtype)
            else:
                h_in = h_prev
            h_next = self.mealy_gru(obs_h, h_in)
            obs_h = obs_h + 0.35 * norm(h_next)
        else:
            h_next = obs_h.detach()

        seq_parts = [obs_h, cart_h, team_h] + extra_tokens
        team_parts = [
            obs[:, OBS["TEAM"]].to(torch.float32),
            carts[:, CS["CONTROL_TEAM"]].to(torch.float32),
            teams[:, TS["ID"]].to(torch.float32),
        ] + extra_teams
        x = torch.cat(seq_parts, dim=0)
        entity_teams = torch.cat(team_parts, dim=0)
        team_ids = teams[:, TS["ID"]].to(torch.float32)

        if self.recurrent:
            blk = self.blocks[0]
            for step in range(self.depth):
                x_new = blk(x, entity_teams, team_ids, team_valid)
                x = x + self.step_scales[step] * (x_new - x)
        else:
            for blk in self.blocks:
                x = blk(x, entity_teams, team_ids, team_valid)

        xn = norm(x)
        p_h = xn[:n]
        c_h = xn[n : n + j]
        t_h = xn[n + j : n + j + k]

        # Extract updated Voronoi cell representations for the spatial flow pointer
        offset = n + j + k
        if geo is not None and "track_tokens" in geo:
            offset += geo["track_tokens"].shape[0]
        if cell_h is not None:
            c_n = cell_h.shape[0]
            vcell_h = xn[offset : offset + c_n]
        else:
            vcell_h = c_h  # fallback to carts if no Voronoi map supplied

        fsq_codes = None
        if self.fsq is not None:
            p_h, fsq_codes = self.fsq(p_h)

        cont_mean = 1.6 * torch.tanh(self.cont_head(p_h))
        cont_logstd = torch.clamp(self.cont_logstd, min=-2.0, max=0.25).unsqueeze(0).expand(n, -1)

        cart_logits = (self.cart_q(p_h) @ self.cart_k(c_h).T) / math.sqrt(self.d)
        rival_logits = (self.rival_q(p_h) @ self.rival_k(t_h).T) / math.sqrt(self.d)
        if k >= 2:
            own_mask = obs[:, OBS["TEAM"]].to(torch.float32)[:, None] == team_ids[None, :]
            rival_logits = rival_logits.masked_fill(own_mask, -1e4)
        cell_logits = (self.cell_q(p_h) @ self.cell_k(vcell_h).T) / math.sqrt(self.d)

        vals = self.value_head(p_h)
        dyn_drift = self.dyn_drift(p_h).squeeze(-1)
        dyn_action = 0.25 * self.dyn_action(p_h)
        return PlanOutput(
            cont_mean, cont_logstd, cart_logits, rival_logits, cell_logits,
            vals[:, 0], vals[:, 1], dyn_drift, dyn_action, h_next, fsq_codes,
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


def plan_entropy(out: PlanOutput) -> torch.Tensor:
    h_cont = (out.cont_logstd + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)
    p_cart = F.softmax(out.cart_logits, dim=-1)
    h_cart = -(p_cart * F.log_softmax(out.cart_logits, dim=-1)).sum(dim=-1)
    p_riv = F.softmax(out.rival_logits, dim=-1)
    h_riv = -(p_riv * F.log_softmax(out.rival_logits, dim=-1)).sum(dim=-1)
    p_cell = F.softmax(out.cell_logits, dim=-1)
    h_cell = -(p_cell * F.log_softmax(out.cell_logits, dim=-1)).sum(dim=-1)
    return h_cont + h_cart + h_riv + h_cell


def pack_plan_rows(
    obs_rows: np.ndarray,
    team_rows: np.ndarray,
    sample: PlanSample,
    arms: np.ndarray,
    active_arms: set[str],
    maxclients: int = 16,
    cell_centroids: np.ndarray | None = None,
) -> np.ndarray:
    """Pack per-bot 16-float parametric flow PLAN rows indexed by 1-based edict slot
    matching mesh_scatter_rows(plc_str_h_plan, plc_pl_f_id, 1, nclients, 16)."""
    out = np.zeros((maxclients, PLAN_WIDTH), dtype=np.float32)
    cont_np = sample.cont.detach().cpu().numpy()
    cart_np = sample.cart_pick.detach().cpu().numpy()
    rival_np = sample.rival_pick.detach().cpu().numpy()
    cell_np = sample.cell_pick.detach().cpu().numpy()
    team_ids = team_rows[:, TS["ID"]].astype(np.float32)
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
                nav_w = float(np.log1p(np.exp(np.clip(cont_np[i, 8] - 0.8, -10.0, 10.0))))
            else:
                nav_x = nav_y = nav_w = 0.0
            cart_ds = float(np.clip(cont_np[i, 7] * 350.0, -600.0, 600.0))
            posture = float(np.tanh(cont_np[i, 10]))
            out[eid - 1, 0] = float(eid)
            out[eid - 1, 1] = float(cart_np[i])
            out[eid - 1, 2:7] = cont_np[i, 0:5]        # w0, w1, w2, w3, w4
            out[eid - 1, 7] = focus_team               # leader
            out[eid - 1, 8] = float(cont_np[i, 5])     # item_w
            out[eid - 1, 9] = float(cont_np[i, 6])     # pwr_w
            out[eid - 1, 10] = cart_ds                 # cart_ds
            out[eid - 1, 11] = nav_x                   # nav_x
            out[eid - 1, 12] = nav_y                   # nav_y
            out[eid - 1, 13] = nav_w                   # nav_w
            out[eid - 1, 14] = float(cont_np[i, 9])    # atk_cyl
            out[eid - 1, 15] = posture                 # posture
    return out


def plan_module_group(param_name: str) -> str:
    root = param_name.split(".", 1)[0]
    if root in ("enc_obs", "enc_cart", "enc_team", "enc_evt", "enc_track", "enc_cell", "enc_item", "type_embed", "mealy_gru"):
        return "encoder"
    if root == "blocks":
        if any(k in param_name for k in ("team_metric", "rival_metric", "val_proj", "dpp_", "gram_out", "alpha_gram")):
            return "gram_cgt"
        return "resnet_ffn"
    if root in ("step_scales", "fsq"):
        return "resnet_ffn"
    if root in ("cont_head", "cont_logstd", "cart_q", "cart_k", "rival_q", "rival_k", "cell_q", "cell_k"):
        return "action_head"
    if root == "value_head":
        return "value_head"
    if root in ("dyn_drift", "dyn_action"):
        return "dynamics_head"
    return "other"


def _team_eff_rho(carts: torch.Tensor, teams: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
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
    eff_held = held + 0.5 * (owner.to(torch.float32).T @ seg_frac)
    rho = torch.where(eff_held > 0, (rate * eff_held) / rem, torch.zeros_like(scores))
    mem = obs[:, OBS["TEAM"]].to(torch.float32)[:, None] == teams[:, TS["ID"]].to(torch.float32)[None, :]
    return mem.to(torch.float32) @ rho


def _geo_to_torch(geo: dict[str, np.ndarray] | None, device: str) -> dict[str, torch.Tensor] | None:
    if geo is None:
        return None
    out = {}
    for k in ("bot_geo", "cart_geo", "track_tokens", "track_teams", "cell_tokens", "item_tokens", "fovea_weights"):
        if k in geo:
            out[k] = torch.as_tensor(geo[k], dtype=torch.float32, device=device)
    return out


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


def plan_losses_for_transition(
    model: PlanWally,
    tr: PlanTransition,
    device: str = "cpu",
    *,
    entropy_coef: float = 0.01,
    actor_weight: float = 1.0,
    critic_weight: float = 0.5,
    dyn_weight: float = 0.25,
) -> tuple[dict[str, torch.Tensor], dict]:
    obs_in = torch.as_tensor(tr.obs_in, dtype=torch.float32, device=device)
    carts_in = torch.as_tensor(tr.carts_in, dtype=torch.float32, device=device)
    teams_in = torch.as_tensor(tr.teams_in, dtype=torch.float32, device=device)
    evt_in = torch.as_tensor(tr.events_in, dtype=torch.float32, device=device) if len(tr.events_in) else None
    geo_in = _geo_to_torch(tr.geo_in, device)
    h_in = torch.as_tensor(tr.h_in, dtype=torch.float32, device=device) if tr.h_in is not None else None

    obs_out = torch.as_tensor(tr.obs_out, dtype=torch.float32, device=device)
    carts_out = torch.as_tensor(tr.carts_out, dtype=torch.float32, device=device)
    teams_out = torch.as_tensor(tr.teams_out, dtype=torch.float32, device=device)
    evt_out = torch.as_tensor(tr.events_out, dtype=torch.float32, device=device) if len(tr.events_out) else None
    geo_out = _geo_to_torch(tr.geo_out, device)
    h_out = torch.as_tensor(tr.h_out, dtype=torch.float32, device=device) if tr.h_out is not None else None

    cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in)
    with torch.no_grad():
        nxt = model(obs_out, carts_out, teams_out, evt_out, geo=geo_out, h_prev=h_out)

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
    boot = torch.where(nwm, nxt.value_winnie[succ], nxt.value_lou[succ])
    boot = torch.where((wm == nwm) & (disc > 0) & (valid_succ > 0), boot, 0.0)
    target = reward + disc * boot.detach()
    error = target - val

    logpi = plan_logp(cur, cont, cart_pick, rival_pick, cell_pick)
    log_ratio = torch.clamp(logpi - beh_lp, -POLICY_LOG_RATIO_BOUND, POLICY_LOG_RATIO_BOUND)
    adv = torch.clamp(error.detach(), -2.0, 2.0)
    am_sum = torch.clamp(am.sum(), min=1.0)
    surrogate = clipped_policy_surrogate(log_ratio, adv, POLICY_RATIO_CLIP)
    ent = (plan_entropy(cur) * am).sum() / am_sum
    actor_loss = actor_weight * (-(surrogate * am).sum() / am_sum - entropy_coef * ent)

    w_mask = wm.to(torch.float32)
    l_mask = (~wm).to(torch.float32)
    w_loss = critic_weight * (F.smooth_l1_loss(cur.value_winnie, target, reduction="none", beta=0.5) * w_mask).sum() / torch.clamp(w_mask.sum(), min=1.0)
    l_loss = critic_weight * (F.smooth_l1_loss(cur.value_lou, target, reduction="none", beta=0.5) * l_mask).sum() / torch.clamp(l_mask.sum(), min=1.0)

    with torch.no_grad():
        rho_c = _team_eff_rho(carts_in, teams_in, obs_in)
        rho_n = _team_eff_rho(carts_out, teams_out, obs_out)[succ]
        obs_drho = torch.asinh(10.0 * (rho_n - rho_c))
    pred_drho = cur.dyn_drift + (cur.dyn_action * cont).sum(dim=-1)
    dyn_loss = dyn_weight * (F.smooth_l1_loss(pred_drho, obs_drho, reduction="none", beta=0.25) * valid_succ).sum() / torch.clamp(valid_succ.sum(), min=1.0)

    total = actor_loss + w_loss + l_loss + dyn_loss
    with torch.no_grad():
        lr_np = log_ratio.detach().cpu().numpy()[tr.actor_mask.astype(bool)] if np.any(tr.actor_mask) else np.empty((0,))
        ratio_np = np.exp(np.clip(lr_np, -20.0, 20.0)) if lr_np.size else np.empty((0,))
        diag = {
            "loss_total": float(total.item()),
            "loss_actor": float(actor_loss.item()),
            "loss_w": float(w_loss.item()),
            "loss_l": float(l_loss.item()),
            "loss_dyn": float(dyn_loss.item()),
            "entropy": float(ent.item()),
            "is_ratio_mean": float(ratio_np.mean()) if ratio_np.size else None,
            "is_clip_frac": float(np.mean(np.abs(lr_np) > math.log(1.0 + POLICY_RATIO_CLIP))) if lr_np.size else None,
            "winnie_mse": float(((cur.value_winnie - target).square() * w_mask).sum().item() / max(1.0, float(w_mask.sum().item()))) if float(w_mask.sum().item()) > 0 else None,
            "lou_mse": float(((cur.value_lou - target).square() * l_mask).sum().item() / max(1.0, float(l_mask.sum().item()))) if float(l_mask.sum().item()) > 0 else None,
            "reward_abs": float(reward.abs().mean().item()),
        }
    return {
        "total": total,
        "actor_pg": actor_loss,
        "critic_winnie": w_loss,
        "critic_lou": l_loss,
        "critic_both": w_loss + l_loss,
        "dynamics": dyn_loss,
    }, diag


def probe_plan_cross_server_mccandlish(
    model: PlanWally,
    transitions: list[PlanTransition],
    *,
    device: str = "cpu",
    max_per_server: int = 16,
    components: tuple[str, ...] = ("total", "actor_pg", "critic_winnie", "critic_lou", "critic_both", "dynamics"),
) -> dict:
    """Exact cross-server i.i.d. r=G McCandlish (2018) SNR estimator for PlanWally across
    G independent concurrent server trajectories, broken down by loss component and module group."""
    by_srv: dict[int, list[PlanTransition]] = {}
    for tr in transitions:
        by_srv.setdefault(int(tr.session), []).append(tr)
    servers = [s for s, lst in sorted(by_srv.items()) if lst]
    if len(servers) < 2:
        return {"error": f"need >=2 servers, got {len(servers)}"}

    per_srv = min(max_per_server, min(len(by_srv[s]) for s in servers))
    params = list(model.named_parameters())
    names = [n for n, _ in params]
    tensors = [p for _, p in params]

    rep_grads = {c: [] for c in components}
    all_diags = []
    for s in servers:
        sub = by_srv[s][-per_srv:]
        sums = {c: {n: None for n in names} for c in components}
        for tr in sub:
            comps, diag = plan_losses_for_transition(model, tr, device=device)
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
        "servers": len(servers),
        "transitions_per_server": per_srv,
        "by_component": by_comp,
        "by_component_and_group": by_comp_grp,
        "cross_cos_actor_critic": float(np.mean([
            c for g_a, g_c in zip(rep_grads["actor_pg"], rep_grads["critic_both"])
            if (c := grad_cosine(g_a, g_c)) is not None
        ])) if rep_grads["actor_pg"] else None,
        "diagnostics": {k: _m(k) for k in ("loss_total", "loss_actor", "loss_w", "loss_l", "loss_dyn", "entropy", "is_ratio_mean", "is_clip_frac", "winnie_mse", "lou_mse", "reward_abs")},
    }


@torch.no_grad()
def ema_update(ema_model: PlanWally, live_model: PlanWally, decay: float = 0.95) -> None:
    for p_ema, p_live in zip(ema_model.parameters(), live_model.parameters()):
        p_ema.mul_(decay).add_(p_live, alpha=1.0 - decay)


def clone_plan_model(model: PlanWally) -> PlanWally:
    return copy.deepcopy(model)
