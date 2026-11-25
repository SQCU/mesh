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

from .aurora import Aurora, aurora_init_weight_
from .game_value import cgt_tensor_features
from .grad_snr import (
    grad_cosine,
    mccandlish_pair_from_grads,
    summarize_mccandlish_pairs,
)
from .policy_contract import POLICY_LOG_RATIO_BOUND, POLICY_RATIO_CLIP
from .policy_math import clipped_policy_surrogate

PLAN_WIDTH = 16
PLAN_CONT_DIM = 11
BOT_GEO_DIM = 12
CART_GEO_DIM = 10
TRACK_TOK_DIM = 10
CELL_TOK_DIM = 10
ITEM_TOK_DIM = 9


# ── EDM2 (Karras et al. 2024) Magnitude-Preserving Primitives ────────────────────────────

def mp_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Unit-RMS token normalization: forces E[x_j^2] = 1 so internal activations never grow outliers."""
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def mp_unit_vec(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit L2 vector normalization (||x||_2 = 1) for cosine pointer heads & bounded kernel maps."""
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def mp_silu(x: torch.Tensor) -> torch.Tensor:
    """EDM2 variance-preserving SiLU (std(silu(N(0,1))) ≈ 0.596)."""
    return F.silu(x) / 0.596


def mp_lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """EDM2 magnitude-preserving residual blend of two unit-RMS tensors."""
    t_c = torch.clamp(t, 0.0, 1.0)
    return ((1.0 - t_c) * a + t_c * b) * torch.rsqrt((1.0 - t_c).square() + t_c.square() + 1e-8)


class MPLinear(nn.Module):
    """EDM2 unit-norm weight linear layer with Aurora-tuned Stiefel ∩ Row-Oblique initialization.
    Every output row w_i lies on the unit hypersphere ||w_i||_2 = 1 and starts as a tight frame
    with uniform row leverage (zero initial neuron favoritism, zero AdamW weight decay)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        w = torch.empty(self.out_features, self.in_features)
        aurora_init_weight_(w)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_unit = self.weight / self.weight.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.linear(x, w_unit)


@torch.no_grad()
def project_unit_norm_(model: nn.Module, eps: float = 1e-6) -> None:
    """EDM2 forced unit-norm weight projection after every optimizer step (replaces AdamW)."""
    for m in model.modules():
        if isinstance(m, MPLinear):
            w = m.weight.data
            w.div_(w.norm(dim=-1, keepdim=True).clamp_min(eps))


def weight_norm_report(model: nn.Module) -> dict:
    """Audit unit-norm weight invariant across all MPLinear layers."""
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
    lr: float = 0.012,
    scalar_lr: float = 2.5e-3,
    mu: float = 0.90,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
) -> Aurora:
    """Aurora leverage-aware spectral optimizer (replaces Adam/AdamW & fixes Muon tall-matrix neuron death)."""
    return Aurora(
        model.parameters(),
        lr=lr,
        scalar_lr=scalar_lr,
        mu=mu,
        nesterov=True,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
    )


def make_edm2_optimizer(model: nn.Module, lr: float = 0.012) -> Aurora:
    """Default EDM2 unit-norm optimizer: Aurora spectral optimizer + post-step unit-sphere projection."""
    return make_aurora_optimizer(model, lr=lr)


def edm2_step(optimizer: torch.optim.Optimizer, model: nn.Module, grad_clip: float = 1.0) -> float:
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
    optimizer.step()
    project_unit_norm_(model)
    return gnorm


def prenormalize_obs(obs: torch.Tensor, geo: dict[str, torch.Tensor] | None) -> torch.Tensor:
    """EDM2-style domain prenormalization: strips map-specific absolute coordinate drift
    and brings every physical observation column to O(1) unit scale before mp_norm."""
    x = obs.clone()
    if geo is not None and "map_center" in geo and "map_radius" in geo:
        center = geo["map_center"].to(obs.device, obs.dtype)
        radius = torch.clamp(geo["map_radius"].to(obs.device, obs.dtype), min=128.0)
        x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] = (x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] - center[None, :]) / radius
    else:
        x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] = x[:, OBS["POS_X"] : OBS["POS_Z"] + 1] / 1000.0
    x[:, OBS["VEL_X"] : OBS["VEL_Z"] + 1] = x[:, OBS["VEL_X"] : OBS["VEL_Z"] + 1] / 400.0
    x[:, OBS["HEALTH"]] = (x[:, OBS["HEALTH"]] - 100.0) / 100.0
    x[:, OBS["ARMOR"]] = (x[:, OBS["ARMOR"]] - 50.0) / 100.0
    x[:, OBS["AMMO_SHELLS"] : OBS["AMMO_FUEL"] + 1] = x[:, OBS["AMMO_SHELLS"] : OBS["AMMO_FUEL"] + 1] / 50.0
    # Zero out non-stationary absolute clock / sequence counters that leak wall time across waves
    for col in ("ENGINE_TIME", "SPAWN_TIME", "STRENGTH_FINISHED", "RESPONSE_SEQ", "RESPONSE_TIME",
                "APPLIED_SEQUENCE", "OUTCOME_A_SEQ", "OUTCOME_B_SEQ", "CELL_X", "CELL_Y",
                "SWIZZLE_SCHEDULED_TIME", "SWIZZLE_ACTUAL_TIME"):
        if col == "STRENGTH_FINISHED":
            rem = torch.clamp(obs[:, OBS["STRENGTH_FINISHED"]] - obs[:, OBS["ENGINE_TIME"]], min=0.0, max=30.0) / 30.0
            x[:, OBS[col]] = rem
        else:
            x[:, OBS[col]] = 0.0
    x = torch.asinh(x)
    return mp_norm(x)


def prenormalize_carts(carts: torch.Tensor, cart_extra: torch.Tensor, geo: dict[str, torch.Tensor] | None) -> torch.Tensor:
    c = carts.clone()
    plen = torch.clamp(c[:, CS["PATH_LENGTH"]], min=1.0)
    c[:, CS["PATH_POSITION"]] = c[:, CS["PATH_POSITION"]] / plen
    c[:, CS["PATH_LENGTH"]] = torch.log1p(plen / 500.0)
    c[:, CS["SPEED"]] = c[:, CS["SPEED"]] / 100.0
    if geo is not None and "map_center" in geo and "map_radius" in geo:
        center = geo["map_center"].to(carts.device, carts.dtype)
        radius = torch.clamp(geo["map_radius"].to(carts.device, carts.dtype), min=128.0)
        c[:, CS["POS_X"] : CS["POS_Z"] + 1] = (c[:, CS["POS_X"] : CS["POS_Z"] + 1] - center[None, :]) / radius
    else:
        c[:, CS["POS_X"] : CS["POS_Z"] + 1] = c[:, CS["POS_X"] : CS["POS_Z"] + 1] / 1000.0
    c_geo = geo["cart_geo"] if (geo is not None and "cart_geo" in geo) else torch.zeros((carts.shape[0], CART_GEO_DIM), device=carts.device, dtype=carts.dtype)
    return mp_norm(torch.asinh(torch.cat((c, cart_extra, c_geo), dim=-1)))


def prenormalize_teams(teams: torch.Tensor, team_extra: torch.Tensor) -> torch.Tensor:
    t = teams.clone()
    lim = torch.clamp(t[:, TS["SCORE_LIMIT"]], min=1.0)
    t[:, TS["SCORE"]] = t[:, TS["SCORE"]] / lim
    t[:, TS["SCORE_LIMIT"]] = 1.0
    t[:, TS["EPISODE"]] = 0.0
    return mp_norm(torch.asinh(torch.cat((t, team_extra), dim=-1)))


class FSQBottleneck(nn.Module):
    """Finite Scalar Quantization bottleneck with straight-through estimator on unit-RMS latents."""

    def __init__(self, d: int, levels: tuple[int, ...] = (5, 5, 5, 5)):
        super().__init__()
        self.levels = tuple(int(v) for v in levels)
        self.proj_in = MPLinear(d, len(self.levels))
        self.proj_out = MPLinear(len(self.levels), d)
        self.in_gain = nn.Parameter(torch.tensor(1.2))
        self.blend_logit = nn.Parameter(torch.tensor(-1.5))
        half = torch.tensor([(v - 1) / 2.0 for v in self.levels], dtype=torch.float32)
        self.register_buffer("half_levels", half)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.clamp(self.in_gain, 0.25, 4.0)
        z_raw = self.proj_in(x)
        if z_raw.shape[0] > 1:
            z_mean = z_raw.mean(dim=0, keepdim=True)
            z_std = z_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(0.35)
            z_raw = (z_raw - z_mean) / z_std
        z = torch.tanh(g * z_raw) * self.half_levels
        q = torch.round(z)
        z_ste = z + (q - z).detach()
        scaled_q = z_ste / torch.clamp(self.half_levels, min=1.0)
        out = mp_lerp(x, self.proj_out(scaled_q), torch.sigmoid(self.blend_logit))
        return mp_norm(out), q.detach()


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


class MPResCGTBlock(nn.Module):
    """Bitter-Lesson Wide Subspace ResNet + Plain Linear Gram CGT Block (EDM2 unit-norm + Aurora-ready):
    - Pure associative linear Gram R(R^T V) & per-team T(T^T V): zero hand-crafted nonlinear kernel hacks.
    - High-dimensional hidden stream (d=256, h=1024) where multi-layer rotate (MPLinear) + sum (mp_lerp)
      bundles sharpness, routing, and information separation into model-learned orthogonal subspaces.
    - Bounded-condition DPP repulsive Gram (I + g_dpp * B^T B)^{-1} (B^T V) for anti-clumping team split.
    - Native products.World 4-Studio Tensor Parallelism on 'cross' (row-Gram) and 'ffn' (column-shard)."""

    def __init__(self, d: int = 256, h: int = 1024, r_gram: int = 128, r_dpp: int = 64, init_blend: float = 0.25):
        super().__init__()
        self.d = d
        self.h = h
        self.r_gram = r_gram
        self.r_dpp = r_dpp

        self.team_metric = MPLinear(d, r_gram)
        self.rival_metric = MPLinear(d, r_gram)
        self.val_proj = MPLinear(d, d)
        self.dpp_metric = MPLinear(d, r_dpp)
        self.dpp_quality = MPLinear(d, 1)
        self.dpp_out = MPLinear(d, d)
        self.gram_out = MPLinear(d, d)

        self.ffn_gate = MPLinear(d, h)
        self.ffn_up = MPLinear(d, h)
        self.ffn_down = MPLinear(h, d)

        self.log_dpp_gain = nn.Parameter(torch.tensor(math.log(1.0)))
        logit_b = math.log(init_blend / max(1e-4, 1.0 - init_blend))
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
    ) -> torch.Tensor:
        xn = mp_norm(x)
        v = mp_norm(self.val_proj(xn))
        m_rows = x.shape[0]
        inv_m = 1.0 / math.sqrt(max(1, m_rows))

        # 1. Pure associative linear Global Gram: R (R^T V) — exact O(M*r*d), native 4-Studio row-TP
        r = mp_norm(self.rival_metric(xn))
        if world is not None and world.parts("cross") is not None:
            lo, hi = world.span("cross", m_rows)
            r_c, v_c = world.copy(r), world.copy(v)
            kv_global = world.reduce(r_c[lo:hi].T @ v_c[lo:hi]) * inv_m
        else:
            kv_global = (r.T @ v) * inv_m
        global_vals = mp_norm(r @ kv_global)

        # 2. Pure associative linear Per-Team Gram: T (T^T V) within each team
        t = mp_norm(self.team_metric(xn))
        mem = (entity_team_ids[:, None] == team_ids[None, :]) & team_valid[None, :]
        mem_f = mem.to(x.dtype)
        tv = (t[:, :, None] * v[:, None, :]).reshape(m_rows, -1)
        if world is not None and world.parts("cross") is not None:
            lo, hi = world.span("cross", m_rows)
            mem_c, tv_c = world.copy(mem_f), world.copy(tv)
            team_sum = world.reduce(mem_c[lo:hi].T @ tv_c[lo:hi])
            team_cnt = torch.clamp(world.reduce(mem_c[lo:hi].sum(dim=0, keepdim=True).T), min=1.0)
        else:
            team_sum = mem_f.T @ tv
            team_cnt = torch.clamp(mem_f.sum(dim=0, keepdim=True).T, min=1.0)
        team_mean = team_sum / team_cnt
        ctx_t = (mem_f @ team_mean).reshape(m_rows, self.r_gram, self.d)
        team_vals = mp_norm((t[:, :, None] * ctx_t).sum(dim=1))

        # 3. Bounded-condition DPP repulsive Gram: (I + g_dpp * B^T B)^{-1} (B^T V)
        q_dpp = torch.sigmoid(self.dpp_quality(xn))
        b_dpp = q_dpp * mp_unit_vec(self.dpp_metric(xn))
        g_dpp = torch.clamp(torch.exp(self.log_dpp_gain), 0.1, 8.0)
        if world is not None and world.parts("cross") is not None:
            lo, hi = world.span("cross", m_rows)
            b_c, v_c = world.copy(b_dpp), world.copy(v)
            btb = world.reduce(b_c[lo:hi].T @ b_c[lo:hi])
            btv = world.reduce(b_c[lo:hi].T @ v_c[lo:hi])
        else:
            btb = b_dpp.T @ b_dpp
            btv = b_dpp.T @ v
        cov = torch.eye(self.r_dpp, dtype=x.dtype, device=x.device) + g_dpp * btb
        if x.device.type == "mps":
            solved = torch.linalg.solve(cov.cpu(), btv.cpu()).to(x.device)
        else:
            solved = torch.linalg.solve(cov, btv)
        dpp_ctx = mp_norm(self.dpp_out(b_dpp @ solved))

        gram_merged = mp_norm(self.gram_out(mp_norm(global_vals + team_vals + dpp_ctx)))
        x = mp_norm(mp_lerp(xn, gram_merged, torch.sigmoid(self.blend_gram)))

        # 4. Wide Magnitude-Preserving SwiGLU FFN (tall h >> d protected from neuron death by Aurora!)
        if world is not None and world.parts("ffn") is not None:
            lo, hi = world.span("ffn", self.h)
            x_c = world.copy(x)
            wg = self.ffn_gate.weight[lo:hi]
            wu = self.ffn_up.weight[lo:hi]
            wd = self.ffn_down.weight[:, lo:hi]
            wg = wg / wg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            wu = wu / wu.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            wd_full_norm = self.ffn_down.weight.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            wd = wd / wd_full_norm
            hid = mp_silu(F.linear(x_c, wg)) * F.linear(x_c, wu)
            ffn_out = mp_norm(world.reduce(F.linear(hid, wd)))
        else:
            hid = mp_silu(self.ffn_gate(x)) * self.ffn_up(x)
            self._last_ffn_act = hid.detach()
            ffn_out = mp_norm(self.ffn_down(hid))
        x = mp_norm(mp_lerp(x, ffn_out, torch.sigmoid(self.blend_ffn)))
        return x


class PlanWally(nn.Module):
    """Aurora + EDM2 Wide Subspace Linear-Gram ResNet controller.
    - Pure associative linear Gram R(R^T V) + high-dimensional subspace stream (default d=256, h=1024).
    - Every linear weight row is Aurora-initialized (Stiefel ∩ Row-Oblique) and unit-norm clamped.
    - Optimized with Aurora (leverage-uniform polar orthogonalization) — zero AdamW, zero dead neurons."""

    def __init__(
        self,
        d: int = 256,
        depth: int = 4,
        h: int = 1024,
        r_gram: int = 128,
        r_dpp: int = 64,
        recurrent: bool = True,
        use_fsq: bool = True,
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

        self.enc_obs = MPLinear(OBS_WIDTH + BOT_GEO_DIM, d)
        self.enc_cart = MPLinear(18 + 4 + CART_GEO_DIM, d)
        self.enc_team = MPLinear(7 + 2, d)
        self.enc_evt = MPLinear(EVT_WIDTH, d)
        self.enc_track = MPLinear(TRACK_TOK_DIM, d)
        self.enc_cell = MPLinear(CELL_TOK_DIM, d)
        self.enc_item = MPLinear(ITEM_TOK_DIM, d)
        self.type_embed = nn.Parameter(mp_norm(torch.randn(6, d)))
        self.type_blend = nn.Parameter(torch.tensor(-1.8))  # sigmoid ~= 0.14

        # Magnitude-preserving Mealy state update (replaces unconstrained GRUCell)
        if self.use_mealy:
            self.mealy_gate = MPLinear(2 * d, d)
            self.mealy_cand = MPLinear(2 * d, d)
        else:
            self.mealy_gate = None
            self.mealy_cand = None

        self.fsq = FSQBottleneck(d, levels=self.fsq_levels) if self.use_fsq else None

        init_blend = 1.0 / (self.depth + 1.0)
        if self.recurrent:
            self.blocks = nn.ModuleList([MPResCGTBlock(d, h, r_gram, r_dpp, init_blend=init_blend)])
            self.step_blends = nn.Parameter(torch.full((self.depth,), math.log(0.35 / 0.65)))
        else:
            self.blocks = nn.ModuleList([MPResCGTBlock(d, h, r_gram, r_dpp, init_blend=init_blend) for _ in range(self.depth)])
            self.step_blends = None

        # Output heads: unit-norm projections + EXPLICIT bounded scalar gains/temperatures for sharp PDFs!
        self.cont_proj = MPLinear(d, PLAN_CONT_DIM)
        self.cont_gain = nn.Parameter(torch.full((PLAN_CONT_DIM,), 0.04))
        self.cont_bias = nn.Parameter(torch.zeros(PLAN_CONT_DIM))
        with torch.no_grad():
            self.cont_bias[6] = 0.10   # pwr_w mild readiness
            self.cont_bias[8] = -1.1   # nav_w soft boundary (~0.14)
            self.cont_bias[9] = 0.25   # atk_cyl mild cart focus
        self.cont_logstd = nn.Parameter(torch.full((PLAN_CONT_DIM,), math.log(0.38)))

        self.cart_q = MPLinear(d, d)
        self.cart_k = MPLinear(d, d)
        self.rival_q = MPLinear(d, d)
        self.rival_k = MPLinear(d, d)
        self.cell_q = MPLinear(d, d)
        self.cell_k = MPLinear(d, d)
        # Explicit scalar inverse-temperatures: allow arbitrarily steep categorical PDFs (up to temp=16)
        self.log_temp_cart = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_temp_rival = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_temp_cell = nn.Parameter(torch.tensor(math.log(2.0)))

        self.value_proj = MPLinear(d, 2)
        self.value_gain = nn.Parameter(torch.tensor(0.5))
        self.value_bias = nn.Parameter(torch.zeros(2))
        self.dyn_drift_proj = MPLinear(d, 1)
        self.dyn_drift_gain = nn.Parameter(torch.tensor(0.25))
        self.dyn_action_proj = MPLinear(d, PLAN_CONT_DIM)
        self.dyn_action_gain = nn.Parameter(torch.tensor(0.25))

        project_unit_norm_(self)

    def forward(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        events: torch.Tensor | None = None,
        geo: dict[str, torch.Tensor] | None = None,
        h_prev: torch.Tensor | None = None,
        world=None,
    ) -> PlanOutput:
        n, j, k = obs.shape[0], carts.shape[0], teams.shape[0]
        device, dtype = obs.device, obs.dtype
        cart_valid = torch.ones(j, device=device, dtype=torch.bool)
        team_valid = torch.ones(k, device=device, dtype=torch.bool)
        cart_extra, team_extra = cgt_tensor_features(carts, teams, cart_valid, team_valid)

        tb = torch.sigmoid(self.type_blend)
        te = mp_norm(self.type_embed)

        obs_pre = prenormalize_obs(obs, geo)
        bot_geo = mp_norm(torch.asinh(geo["bot_geo"])) if (geo is not None and "bot_geo" in geo) else torch.zeros((n, BOT_GEO_DIM), device=device, dtype=dtype)
        obs_h = mp_lerp(mp_norm(self.enc_obs(torch.cat((obs_pre, bot_geo), dim=-1))), te[0].unsqueeze(0), tb)

        if events is not None and events.shape[0] > 0:
            evt_h = mp_norm(self.enc_evt(mp_norm(torch.asinh(events))))
            evt_team = events[:, EVT["TEAM"]].to(torch.float32)
            obs_team = obs[:, OBS["TEAM"]].to(torch.float32)
            em = (obs_team[:, None] == evt_team[None, :]).to(dtype)
            ecnt = torch.clamp(em.sum(dim=1, keepdim=True), min=1.0)
            obs_h = mp_norm(mp_lerp(obs_h, mp_norm((em @ evt_h) / ecnt), torch.tensor(0.2, device=device, dtype=dtype)))

        cart_pre = prenormalize_carts(carts, cart_extra, geo)
        cart_h = mp_lerp(mp_norm(self.enc_cart(cart_pre)), te[1].unsqueeze(0), tb)

        team_pre = prenormalize_teams(teams, team_extra)
        team_h = mp_lerp(mp_norm(self.enc_team(team_pre)), te[2].unsqueeze(0), tb)

        extra_tokens = []
        extra_teams = []

        if geo is not None and "track_tokens" in geo and geo["track_tokens"].shape[0] > 0:
            tr_h = mp_lerp(mp_norm(self.enc_track(mp_norm(torch.asinh(geo["track_tokens"])))), te[3].unsqueeze(0), tb)
            extra_tokens.append(tr_h)
            extra_teams.append(geo["track_teams"].to(torch.float32))

        cell_h = None
        if geo is not None and "cell_tokens" in geo and geo["cell_tokens"].shape[0] > 0:
            cell_h = mp_lerp(mp_norm(self.enc_cell(mp_norm(torch.asinh(geo["cell_tokens"])))), te[4].unsqueeze(0), tb)
            if "fovea_weights" in geo:
                obs_h = mp_norm(mp_lerp(obs_h, mp_norm(geo["fovea_weights"] @ cell_h), torch.tensor(0.25, device=device, dtype=dtype)))
            extra_tokens.append(cell_h)
            extra_teams.append(torch.zeros((cell_h.shape[0],), device=device, dtype=torch.float32))

        if geo is not None and "item_tokens" in geo and geo["item_tokens"].shape[0] > 0:
            it_h = mp_lerp(mp_norm(self.enc_item(mp_norm(torch.asinh(geo["item_tokens"])))), te[5].unsqueeze(0), tb)
            extra_tokens.append(it_h)
            extra_teams.append(torch.zeros((it_h.shape[0],), device=device, dtype=torch.float32))

        # Magnitude-preserving Mealy state recursion (strictly unit-RMS!)
        if self.mealy_gate is not None:
            if h_prev is None or h_prev.shape != (n, self.d):
                h_in = obs_h.detach()
            else:
                h_in = mp_norm(h_prev)
            cat_m = torch.cat((obs_h, h_in), dim=-1)
            z_gate = torch.sigmoid(self.mealy_gate(cat_m))
            cand = mp_norm(mp_silu(self.mealy_cand(cat_m)))
            h_next = mp_norm(mp_lerp(h_in, cand, z_gate))
            obs_h = mp_norm(mp_lerp(obs_h, h_next, torch.tensor(0.25, device=device, dtype=dtype)))
        else:
            h_next = obs_h.detach()

        seq_parts = [obs_h, cart_h, team_h] + extra_tokens
        team_parts = [
            obs[:, OBS["TEAM"]].to(torch.float32),
            carts[:, CS["CONTROL_TEAM"]].to(torch.float32),
            teams[:, TS["ID"]].to(torch.float32),
        ] + extra_teams
        x = mp_norm(torch.cat(seq_parts, dim=0))
        entity_teams = torch.cat(team_parts, dim=0)
        team_ids = teams[:, TS["ID"]].to(torch.float32)

        if self.recurrent:
            blk = self.blocks[0]
            for step in range(self.depth):
                x_new = blk(x, entity_teams, team_ids, team_valid, world=world)
                x = mp_norm(mp_lerp(x, x_new, torch.sigmoid(self.step_blends[step])))
        else:
            for blk in self.blocks:
                x = blk(x, entity_teams, team_ids, team_valid, world=world)

        xn = mp_norm(x)
        p_h = xn[:n]
        c_h = xn[n : n + j]
        t_h = xn[n + j : n + j + k]

        offset = n + j + k
        if geo is not None and "track_tokens" in geo:
            offset += geo["track_tokens"].shape[0]
        if cell_h is not None:
            vcell_h = xn[offset : offset + cell_h.shape[0]]
        else:
            vcell_h = c_h

        fsq_codes = None
        if self.fsq is not None:
            p_h, fsq_codes = self.fsq(p_h)

        # Explicit gain-scaled continuous head (bounded, sharp, zero weight/activation blowup)
        g_cont = torch.clamp(self.cont_gain, -3.0, 3.0)
        cont_mean = 2.0 * torch.tanh(g_cont * self.cont_proj(p_h) + self.cont_bias)
        cont_logstd = torch.clamp(self.cont_logstd, min=-2.5, max=0.25).unsqueeze(0).expand(n, -1)

        # Cosine pointer heads x explicit bounded inverse-temperature (steep PDF without outlier growth!)
        temp_c = torch.clamp(torch.exp(self.log_temp_cart), 0.5, 32.0)
        temp_r = torch.clamp(torch.exp(self.log_temp_rival), 0.5, 32.0)
        temp_v = torch.clamp(torch.exp(self.log_temp_cell), 0.5, 32.0)

        cart_logits = temp_c * (mp_unit_vec(self.cart_q(p_h)) @ mp_unit_vec(self.cart_k(c_h)).T)
        rival_logits = temp_r * (mp_unit_vec(self.rival_q(p_h)) @ mp_unit_vec(self.rival_k(t_h)).T)
        if k >= 2:
            own_mask = obs[:, OBS["TEAM"]].to(torch.float32)[:, None] == team_ids[None, :]
            rival_logits = rival_logits.masked_fill(own_mask, -1e4)
        cell_logits = temp_v * (mp_unit_vec(self.cell_q(p_h)) @ mp_unit_vec(self.cell_k(vcell_h)).T)

        vals = torch.clamp(self.value_gain, 0.05, 5.0) * self.value_proj(p_h) + self.value_bias
        dyn_drift = (torch.clamp(self.dyn_drift_gain, 0.05, 2.0) * self.dyn_drift_proj(p_h)).squeeze(-1)
        dyn_action = torch.clamp(self.dyn_action_gain, 0.05, 2.0) * self.dyn_action_proj(p_h)
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
            "edm2_mp": True,
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
    cart_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Pack per-bot 16-float parametric flow PLAN rows indexed by 1-based edict slot.
    cart_ds scales proportionally to the selected cart's lane length for cross-map invariance."""
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
            cpick = int(cart_np[i])
            # Superlinear push-radius mapping: |u|<=0.8 stays <120u (inside cart push radius on all maps);
            # intentional |u|=1.3..2.0 reaches 180..365u (>120u) for vanguard/ambush lead.
            u_ds = float(cont_np[i, 7])
            cart_ds = float(np.clip(math.copysign(120.0 * (abs(u_ds) ** 1.6), u_ds), -480.0, 480.0))
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


def plan_module_group(param_name: str) -> str:
    root = param_name.split(".", 1)[0]
    if root in ("enc_obs", "enc_cart", "enc_team", "enc_evt", "enc_track", "enc_cell", "enc_item",
                "type_embed", "type_blend", "mealy_gate", "mealy_cand"):
        return "encoder"
    if root == "blocks":
        if any(k in param_name for k in ("team_metric", "rival_metric", "val_proj", "dpp_", "gram_out",
                                         "log_tau", "log_dpp", "blend_gram")):
            return "gram_cgt"
        return "resnet_ffn"
    if root in ("step_blends", "fsq"):
        return "resnet_ffn"
    if root in ("cont_proj", "cont_gain", "cont_bias", "cont_logstd",
                "cart_q", "cart_k", "rival_q", "rival_k", "cell_q", "cell_k",
                "log_temp_cart", "log_temp_rival", "log_temp_cell"):
        return "action_head"
    if root in ("value_proj", "value_gain", "value_bias"):
        return "value_head"
    if root in ("dyn_drift_proj", "dyn_drift_gain", "dyn_action_proj", "dyn_action_gain"):
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
    for k in ("bot_geo", "cart_geo", "track_tokens", "track_teams", "cell_tokens",
              "item_tokens", "fovea_weights", "map_center", "map_radius"):
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
    map_name: str = "boil"


def plan_losses_for_transition(
    model: PlanWally,
    tr: PlanTransition,
    device: str = "cpu",
    *,
    entropy_coef: float = 0.01,
    actor_weight: float = 1.0,
    critic_weight: float = 0.5,
    dyn_weight: float = 0.25,
    adv_mean: float = 0.0,
    adv_std: float = 1.0,
    world=None,
) -> tuple[dict[str, torch.Tensor], dict]:
    obs_in = torch.as_tensor(tr.obs_in, dtype=torch.float32, device=device)
    carts_in = torch.as_tensor(tr.carts_in, dtype=torch.float32, device=device)
    teams_in = torch.as_tensor(tr.teams_in, dtype=torch.float32, device=device)
    evt_in = torch.as_tensor(np.array(tr.events_in, copy=True), dtype=torch.float32, device=device) if len(tr.events_in) else None
    geo_in = _geo_to_torch(tr.geo_in, device)
    h_in = torch.as_tensor(tr.h_in, dtype=torch.float32, device=device) if tr.h_in is not None else None

    obs_out = torch.as_tensor(tr.obs_out, dtype=torch.float32, device=device)
    carts_out = torch.as_tensor(tr.carts_out, dtype=torch.float32, device=device)
    teams_out = torch.as_tensor(tr.teams_out, dtype=torch.float32, device=device)
    evt_out = torch.as_tensor(np.array(tr.events_out, copy=True), dtype=torch.float32, device=device) if len(tr.events_out) else None
    geo_out = _geo_to_torch(tr.geo_out, device)
    h_out = torch.as_tensor(tr.h_out, dtype=torch.float32, device=device) if tr.h_out is not None else None

    cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in, world=world)
    with torch.no_grad():
        nxt = model(obs_out, carts_out, teams_out, evt_out, geo=geo_out, h_prev=h_out, world=world)

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
    adv_norm = (error.detach() - float(adv_mean)) / max(float(adv_std), 1e-4)
    adv = torch.clamp(adv_norm, -3.0, 3.0)
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
            "cart_temp": float(torch.clamp(torch.exp(model.log_temp_cart), 0.5, 16.0).item()),
            "cell_temp": float(torch.clamp(torch.exp(model.log_temp_cell), 0.5, 16.0).item()),
        }
    return {
        "total": total,
        "actor_pg": actor_loss,
        "critic_winnie": w_loss,
        "critic_lou": l_loss,
        "critic_both": w_loss + l_loss,
        "dynamics": dyn_loss,
    }, diag


def compute_batch_adv_stats(model: PlanWally, batch: list[PlanTransition], device: str = "cpu") -> tuple[float, float]:
    """Compute batch-wide TD advantage mean & std so opponent-strength shifts cannot bias PPO updates."""
    errs = []
    with torch.no_grad():
        for tr in batch:
            obs_in = torch.as_tensor(tr.obs_in, dtype=torch.float32, device=device)
            carts_in = torch.as_tensor(tr.carts_in, dtype=torch.float32, device=device)
            teams_in = torch.as_tensor(tr.teams_in, dtype=torch.float32, device=device)
            evt_in = torch.as_tensor(np.array(tr.events_in, copy=True), dtype=torch.float32, device=device) if len(tr.events_in) else None
            geo_in = _geo_to_torch(tr.geo_in, device)
            h_in = torch.as_tensor(tr.h_in, dtype=torch.float32, device=device) if tr.h_in is not None else None
            obs_out = torch.as_tensor(tr.obs_out, dtype=torch.float32, device=device)
            carts_out = torch.as_tensor(tr.carts_out, dtype=torch.float32, device=device)
            teams_out = torch.as_tensor(tr.teams_out, dtype=torch.float32, device=device)
            evt_out = torch.as_tensor(np.array(tr.events_out, copy=True), dtype=torch.float32, device=device) if len(tr.events_out) else None
            geo_out = _geo_to_torch(tr.geo_out, device)
            h_out = torch.as_tensor(tr.h_out, dtype=torch.float32, device=device) if tr.h_out is not None else None
            cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in)
            nxt = model(obs_out, carts_out, teams_out, evt_out, geo=geo_out, h_prev=h_out)
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
            am = tr.actor_mask.astype(bool)
            if np.any(am):
                errs.append(err[am])
    if not errs:
        return 0.0, 1.0
    cat = np.concatenate(errs)
    return float(cat.mean()), float(max(cat.std(), 1e-3))


def probe_plan_cross_server_mccandlish(
    model: PlanWally,
    transitions: list[PlanTransition],
    *,
    device: str = "cpu",
    max_per_server: int = 16,
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
    adv_mean, adv_std = compute_batch_adv_stats(model, sample_all, device=device)

    params = list(model.named_parameters())
    names = [n for n, _ in params]
    tensors = [p for _, p in params]

    rep_grads = {c: [] for c in components}
    all_diags = []
    for s in servers:
        sub = by_srv[s][-per_srv:]
        sums = {c: {n: None for n in names} for c in components}
        for tr in sub:
            comps, diag = plan_losses_for_transition(model, tr, device=device, adv_mean=adv_mean, adv_std=adv_std)
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
        "diagnostics": {k: _m(k) for k in ("loss_total", "loss_actor", "loss_w", "loss_l", "loss_dyn", "entropy",
                                          "is_ratio_mean", "is_clip_frac", "winnie_mse", "lou_mse", "reward_abs",
                                          "cart_temp", "cell_temp")},
    }


@torch.no_grad()
def ema_update(ema_model: PlanWally, live_model: PlanWally, decay: float = 0.95) -> None:
    for p_ema, p_live in zip(ema_model.parameters(), live_model.parameters()):
        p_ema.mul_(decay).add_(p_live, alpha=1.0 - decay)
    project_unit_norm_(ema_model)


def clone_plan_model(model: PlanWally) -> PlanWally:
    return copy.deepcopy(model)
