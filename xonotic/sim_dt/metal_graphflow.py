"""Metal 4 NAX compiled inference (`CompiledGraphFlowPolicy`) and full forward +
Advantage-Weighted Boltzmann Flow Matching + analytical backward trainer (`CompiledGraphFlowTrainer`)
for `GraphFlowPolicy`, built directly over `libmesh-graph.dylib`.
"""
from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from payload.tools.strategy_io_schema import CART_WIDTH, OBS_WIDTH, TEAM_WIDTH
from solver.strat.train_config import OptimizerConfig

from .graph_flow_policy import (
    GraphFlowPolicy,
    compute_ess_boltzmann_weights,
    project_graph_flow_unit_norm_,
)
from .metal_runtime import (
    MG_BUF_STATE,
    MG_BUF_WEIGHT,
    MG_OP_CGT_BCAST_BWD,
    MG_OP_CGT_BCAST_MERGE,
    MG_OP_CGT_GRAM_STAGE1,
    MG_OP_CGT_RIVAL_BWD,
    MG_OP_CGT_STAGE1_BWD,
    MG_OP_CGT_TEAM_RIVAL,
    MG_OP_FWHT_PACK_I8,
    MG_OP_LINEAR,
    MG_OP_LINEAR_DW,
    MG_OP_LINEAR_DX,
    MG_OP_LINEAR_I8,
    MG_OP_MP_LERP,
    MG_OP_MP_LERP_BWD,
    MG_OP_MP_NORM,
    MG_OP_MP_NORM_BWD,
    MG_OP_MP_SWIGLU,
    MG_OP_MP_SWIGLU_BWD,
    MG_WIRE_AUTO,
    _align,
    _copy_bf16_to_ptr,
    _copy_f32_to_ptr,
    _load_mesh_graph_lib,
    _make_op,
    _ptr_to_torch_bf16,
    _ptr_to_torch_f32,
    _unit_weight_bf16,
)

MG_OP_GRAPH_PRENORM = 52
MG_OP_GRAPH_EMBED_CAT = 53
MG_OP_GRAPH_EMBED_BWD = 54
MG_OP_NAV_GRAPH_MIX = 55
MG_OP_NAV_GRAPH_MIX_BWD = 56
MG_OP_DPP_GRAM = 57
MG_OP_DPP_GRAM_BWD = 58
MG_OP_GRAPH_SLICE_NORM = 59
MG_OP_GRAPH_SLICE_BWD = 60
MG_OP_FLOW_JET_SAMPLE = 61
MG_OP_FLOW_JET_PREP = 62
MG_OP_FLOW_JET_LOSS_BWD = 63
MG_OP_FLOW_COND_BWD = 64


def _pad_rows_bf16(w: torch.Tensor, rows_pad: int) -> torch.Tensor:
    if w.shape[0] == rows_pad:
        return w.contiguous()
    out = torch.zeros((rows_pad, w.shape[1]), dtype=torch.bfloat16)
    out[: w.shape[0]] = w
    return out


def _unit_pad_bf16(w: torch.Tensor, eps: float, rows_pad: int | None = None, cols_pad: int | None = None) -> torch.Tensor:
    wu = _unit_weight_bf16(w, eps, pad_in=cols_pad)
    if rows_pad is not None and rows_pad > wu.shape[0]:
        return _pad_rows_bf16(wu, rows_pad)
    return wu


def _tangent_unit_grad(w: torch.Tensor, g_wu: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exact backward through F.normalize(w, p=2, dim=-1): (g - u * dot(g, u)) / norm(w)."""
    wf = w.detach().float()
    gf = g_wu.float()
    nrm = wf.square().sum(dim=-1, keepdim=True).clamp_min(max(eps, 1e-4) ** 2).sqrt()
    u = wf / nrm
    dot = (gf * u).sum(dim=-1, keepdim=True)
    return (gf - u * dot) / nrm


def _tangent_mp_norm_grad(w: torch.Tensor, g_wn: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Exact backward through mp_norm(w, eps) for type_embed [4, D]."""
    wf = w.detach().float()
    gf = g_wn.float()
    inv = torch.rsqrt(wf.square().mean(dim=-1, keepdim=True) + max(eps, 1e-4))
    y = wf * inv
    dot = (gf * y).mean(dim=-1, keepdim=True)
    return (gf - y * dot) * inv


class CompiledGraphFlowPolicy:
    """Zero-overhead, GIL-free Metal 4 NAX compiled 1-NFE Graph+DPP+Mealy-Jet Flow Policy."""

    def __init__(
        self,
        model: GraphFlowPolicy,
        E: int,
        N: int,
        W_nav: int,
        J: int,
        K: int,
        *,
        nav_kern: torch.Tensor,
    ):
        self.lib = _load_mesh_graph_lib()
        self.E = int(E)
        self.N = int(N)
        self.W_nav = int(W_nav)
        self.J = int(J)
        self.K = int(K)
        self.M_ent = self.N + self.W_nav + self.J + self.K
        self.A_dim = self.W_nav + self.K + 4
        self.cfg = model.cfg
        self.nc = model.norm_cfg
        self.d = int(model.d)
        self.h = int(model.h)
        self.r_gram = int(self.cfg.r_gram)
        self.r_team = int(self.cfg.r_team)
        self.r_dpp = int(self.cfg.r_dpp)
        self.d_e = int(self.cfg.d_e)
        self.depth = int(model.depth)
        self.recurrent = bool(model.recurrent)

        self.g = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for CompiledGraphFlowPolicy")

        self._w_bufs: dict[str, int] = {}
        self._wq_bufs: dict[str, tuple[int, int]] = {}
        self._g_repack: int | None = None
        self._build_graph(model, nav_kern)
        self.sync_weights_(model)
        rc = self.lib.mesh_graph_compile(self.g)
        if rc != 0:
            raise RuntimeError(f"mesh_graph_compile failed for CompiledGraphFlowPolicy: {rc}")

    def close(self) -> None:
        if getattr(self, "_g_repack", None):
            self.lib.mesh_graph_destroy(self._g_repack)
            self._g_repack = None
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    def __del__(self) -> None:
        self.close()

    def _add_buf(self, nbytes: int, kind: int = MG_BUF_STATE) -> int:
        return int(self.lib.mesh_graph_add_buffer(self.g, int(max(64, nbytes)), int(kind), None, -1, 0))

    def _add_op(self, op: Any) -> None:
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))

    def _build_graph(self, model: GraphFlowPolicy, nav_kern: torch.Tensor) -> None:
        E, N, Wn, J, K, M_ent, A_dim = self.E, self.N, self.W_nav, self.J, self.K, self.M_ent, self.A_dim
        D, H, Rg, Rt, Rd = self.d, self.h, self.r_gram, self.r_team, self.r_dpp
        EN_pad = _align(E * N, 64)
        EW_pad = _align(E * Wn, 64)
        EJ_pad = _align(E * J, 64)
        EK_pad = _align(E * K, 64)
        EM_pad = _align(E * M_ent, 64)
        M_tot, K_tot = E * M_ent, E * K

        # IO buffers
        self.buf_obs = self._add_buf(E * N * OBS_WIDTH * 2)
        self.buf_nav = self._add_buf(E * Wn * 16 * 2)
        self.buf_carts = self._add_buf(E * J * CART_WIDTH * 2)
        self.buf_teams = self._add_buf(E * K * TEAM_WIDTH * 2)
        self.buf_acc = self._add_buf(E * N * A_dim * 4)
        self.buf_eps = self._add_buf(E * N * 32 * 4)
        self.buf_geo_ncfg = self._add_buf(16 * 4)
        self.buf_nav_kern = self._add_buf(Wn * Wn * 4, MG_BUF_WEIGHT)
        self.buf_ent_meta = self._add_buf(E * (M_ent + K) * 4)
        self.buf_bot_nid = self._add_buf(E * N * 4)

        _copy_f32_to_ptr(
            nav_kern.detach().to(device="cpu", dtype=torch.float32).numpy(),
            self.lib.mesh_graph_buffer_ptr(self.g, self.buf_nav_kern),
        )

        n_uniq_blk = 1 if self.recurrent else self.depth
        self._sc_blk_base = 2
        self._sc_step_base = 2 + 6 * n_uniq_blk
        n_scalars = self._sc_step_base + (self.depth if self.recurrent else 0)
        self.buf_scalars = self._add_buf(max(16, n_scalars) * 4)
        self.buf_hp = self._add_buf(16 * 4)

        # Outputs
        self.buf_u_vel = self._add_buf(E * N * A_dim * 4)
        self.buf_u_lat = self._add_buf(E * N * 32 * 4)
        self.buf_val = self._add_buf(E * N * 2 * 4)

        # Weights
        w = self._w_bufs
        wq = self._wq_bufs
        repack_specs: list[tuple[str, int, int]] = []
        w["enc_obs"] = self._add_buf(D * 128 * 2, MG_BUF_WEIGHT)
        w["enc_nav"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["enc_cart"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["enc_team"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["type_embed"] = self._add_buf(4 * D * 2, MG_BUF_WEIGHT)

        def add_i8_w(key: str, rows: int, cols: int) -> None:
            w[key] = self._add_buf(rows * cols * 2, MG_BUF_WEIGHT)
            wq[key] = (self._add_buf(rows * cols), self._add_buf(rows * 4))
            repack_specs.append((key, rows, cols))

        for bi in range(n_uniq_blk):
            p = f"b{bi}."
            w[p + "nav_val_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "nav_out_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            add_i8_w(p + "val_proj", D, D)
            add_i8_w(p + "global_metric", Rg, D)
            w[p + "pool_score"] = self._add_buf(D * 2, MG_BUF_WEIGHT)
            w[p + "team_metric"] = self._add_buf(Rt * D * 2, MG_BUF_WEIGHT)
            w[p + "rival_metric"] = self._add_buf(Rt * D * 2, MG_BUF_WEIGHT)
            w[p + "team_val_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "team_mutate_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "team_bcast_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "gram_out"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "dpp_b_proj"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)
            w[p + "dpp_v_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "dpp_out_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            add_i8_w(p + "ffn_gate_up", 2 * H, D)
            add_i8_w(p + "ffn_down", D, H)

        w["flow_cond_bot"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)
        w["flow_cond_team"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)
        w["basis_nav_proj"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)
        w["basis_team_proj"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)
        w["val_drift_packed"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)
        w["basis_self"] = self._add_buf(4 * 32 * 4, MG_BUF_WEIGHT)
        w["fj_params"] = self._add_buf((128 * 104 + 128 + 32 * 64 + 32) * 4, MG_BUF_WEIGHT)

        # In-GPU INT8 weight repack graph
        self._g_repack = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        for key, rows, cols in repack_specs:
            iw = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, w[key])
            iq = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][0])
            is_ = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][1])
            op_r = _make_op(MG_OP_FWHT_PACK_I8, in_bufs=(iw,), out_bufs=(iq, is_), u=(rows, cols, 0))
            self.lib.mesh_graph_add_op(self._g_repack, ctypes.byref(op_r))
        self.lib.mesh_graph_compile(self._g_repack)

        # Scratch activation buffers
        b_obs_pre = self._add_buf(EN_pad * 128 * 2)
        b_nav_pre = self._add_buf(EW_pad * 64 * 2)
        b_cart_pre = self._add_buf(EJ_pad * 64 * 2)
        b_team_pre = self._add_buf(EK_pad * 64 * 2)
        b_obs_enc = self._add_buf(EN_pad * D * 2)
        b_nav_enc = self._add_buf(EW_pad * D * 2)
        b_cart_enc = self._add_buf(EJ_pad * D * 2)
        b_team_enc = self._add_buf(EK_pad * D * 2)

        b_x = self._add_buf(EM_pad * D * 2)
        b_xn = self._add_buf(EM_pad * D * 2)
        b_xg = self._add_buf(EM_pad * D * 2)
        b_xm = self._add_buf(EM_pad * D * 2)
        b_xm2 = self._add_buf(EM_pad * D * 2)
        b_tmp1 = self._add_buf(max(EM_pad * 2 * H, EW_pad * D, EN_pad * D) * 2)
        b_tmp2 = self._add_buf(max(EM_pad * H, EW_pad * D, EN_pad * D) * 2)
        b_tmp3 = self._add_buf(max(EM_pad * D, EW_pad * D, EN_pad * 64) * 2)
        b_tmp4 = self._add_buf(max(EM_pad * D, EN_pad * D, EK_pad * D) * 2)
        b_tmp5 = self._add_buf(max(EK_pad * D, EN_pad * 64) * 2)
        b_tmp6 = self._add_buf(max(EK_pad * Rt, EK_pad * D, EW_pad * 64) * 2)
        b_tmp7 = self._add_buf(max(EK_pad * Rt, EK_pad * D, EK_pad * 64) * 2)
        b_tmp8 = self._add_buf(max(EK_pad * D, EN_pad * 64) * 2)
        b_q = self._add_buf(EM_pad * max(D, H))
        b_s = self._add_buf(EM_pad * 4)

        eps_n = float(self.nc.mp_norm_eps)
        eps_l = float(self.nc.mp_lerp_eps)
        inv_silu = 1.0 / float(self.nc.mp_silu_scale)

        # 1. Prenorm
        self._add_op(_make_op(
            MG_OP_GRAPH_PRENORM,
            in_bufs=(self.buf_obs, self.buf_nav, self.buf_carts, self.buf_teams, self.buf_acc, self.buf_geo_ncfg),
            out_bufs=(b_obs_pre, b_nav_pre, b_cart_pre),
            u=(E | (N << 16), Wn | (J << 16), K | (A_dim << 16), b_team_pre, self.buf_ent_meta, self.buf_bot_nid),
        ))
        # 2. Encoders
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_obs_pre, w["enc_obs"]), out_bufs=(b_obs_enc,), u=(EN_pad, D, 128)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_nav_pre, w["enc_nav"]), out_bufs=(b_nav_enc,), u=(EW_pad, D, 64)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_cart_pre, w["enc_cart"]), out_bufs=(b_cart_enc,), u=(EJ_pad, D, 64)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_team_pre, w["enc_team"]), out_bufs=(b_team_enc,), u=(EK_pad, D, 64)))

        # 3. 4-type embed + equivariant afferent fold
        self._add_op(_make_op(
            MG_OP_GRAPH_EMBED_CAT,
            in_bufs=(b_obs_enc, b_nav_enc, b_cart_enc, b_team_enc, w["type_embed"], self.buf_acc),
            out_bufs=(b_x, -1, -1),
            u=(E | (N << 16), Wn | (J << 16), K | (D << 16), 0 | (1 << 16), self.buf_bot_nid, self.buf_scalars, 0),
            f=(eps_n, eps_l),
        ))

        # 4. Depth GraphDPPBlock passes
        J_eff = Wn + J
        for step in range(self.depth):
            bi = 0 if self.recurrent else step
            p = f"b{bi}."
            sc0 = self._sc_blk_base + 6 * bi

            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(b_x,), out_bufs=(b_xn,), u=(M_tot, D), f=(eps_n,)))

            # 4a. Directed Nav-Graph Geodesic Diffusion + Bot Local Readout
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX, in_bufs=(b_xn,), out_bufs=(b_tmp1,), u=(E, N, Wn, M_ent, D, 0, 0, EW_pad)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp1, w[p + "nav_val_proj"]), out_bufs=(b_tmp2,), u=(EW_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(b_tmp2,), out_bufs=(b_tmp1,), u=(E * Wn, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX, in_bufs=(b_tmp1, -1, self.buf_nav_kern), out_bufs=(b_tmp2,), u=(E, N, Wn, M_ent, D, 1, 0, EW_pad)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp2, w[p + "nav_out_proj"]), out_bufs=(b_tmp1,), u=(EW_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_NAV_GRAPH_MIX,
                in_bufs=(b_xn, b_tmp1, -1, self.buf_bot_nid, self.buf_scalars),
                out_bufs=(b_xg,),
                u=(E, N, Wn, M_ent, D, 2, sc0 + 0, EM_pad),
                f=(eps_n, eps_l),
            ))

            # 4b. Global Gram + Team Rival Gram (reusing exact MG_OP_CGT_GRAM_*!)
            self._add_op(_make_op(MG_OP_FWHT_PACK_I8, in_bufs=(b_xg,), out_bufs=(b_q, b_s), u=(EM_pad, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "val_proj"]), out_bufs=(b_tmp2,), u=(EM_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "global_metric"]), out_bufs=(b_tmp3,), u=(EM_pad, Rg, D, 0)))
            self._add_op(_make_op(
                MG_OP_CGT_GRAM_STAGE1,
                in_bufs=(b_xg, b_tmp2, b_tmp3, w[p + "pool_score"], self.buf_ent_meta, self.buf_scalars),
                out_bufs=(b_tmp4, b_tmp5),
                u=(E, N, J_eff, K, D, Rg, sc0 + 1),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp5, w[p + "team_metric"]), out_bufs=(b_tmp6,), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp5, w[p + "rival_metric"]), out_bufs=(b_tmp7,), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp5, w[p + "team_val_proj"]), out_bufs=(b_tmp8,), u=(EK_pad, D, D)))
            self._add_op(_make_op(MG_OP_CGT_TEAM_RIVAL, in_bufs=(b_tmp6, b_tmp7, b_tmp8), out_bufs=(b_tmp1,), u=(E, K, D, Rt), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp1, w[p + "team_mutate_proj"]), out_bufs=(b_tmp6,), u=(EK_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(b_tmp5, b_tmp6, self.buf_scalars), out_bufs=(b_tmp7,), u=(K_tot, D, sc0 + 2, 1), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp7, w[p + "team_bcast_proj"]), out_bufs=(b_tmp8,), u=(EK_pad, D, D)))
            self._add_op(_make_op(MG_OP_CGT_BCAST_MERGE, in_bufs=(b_tmp4, b_tmp8, self.buf_ent_meta), out_bufs=(b_tmp1,), u=(E, M_ent, K, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp1, w[p + "gram_out"]), out_bufs=(b_tmp2,), u=(EM_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(b_xg, b_tmp2, self.buf_scalars), out_bufs=(b_xm,), u=(M_tot, D, sc0 + 3, 1), f=(eps_n, eps_l)))

            # 4c. Exact FP32 Per-Team DPP Diversity Kernel on the N bot tokens!
            self._add_op(_make_op(MG_OP_DPP_GRAM, in_bufs=(b_xm,), out_bufs=(b_tmp1,), u=(E, N, M_ent, K, D, Rd, 0, EN_pad)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp1, w[p + "dpp_b_proj"]), out_bufs=(b_tmp3,), u=(EN_pad, 64, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp1, w[p + "dpp_v_proj"]), out_bufs=(b_tmp2,), u=(EN_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(b_tmp2,), out_bufs=(b_tmp4,), u=(E * N, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_DPP_GRAM, in_bufs=(b_tmp3, b_tmp4, self.buf_ent_meta), out_bufs=(b_tmp2,), u=(E, N, M_ent, K, D, Rd, 1, 0), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_tmp2, w[p + "dpp_out_proj"]), out_bufs=(b_tmp1,), u=(EN_pad, D, D)))
            self._add_op(_make_op(MG_OP_DPP_GRAM, in_bufs=(b_xm, b_tmp1, -1, self.buf_scalars), out_bufs=(b_xm2,), u=(E, N, M_ent, K, D, Rd, 2, sc0 + 4), f=(eps_n, eps_l)))

            # 4d. Wide compressed INT8@8 NAX MP-SwiGLU FFN!
            self._add_op(_make_op(MG_OP_FWHT_PACK_I8, in_bufs=(b_xm2,), out_bufs=(b_q, b_s), u=(EM_pad, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_gate_up"]), out_bufs=(b_tmp1,), u=(EM_pad, 2 * H, D, 0)))
            self._add_op(_make_op(MG_OP_MP_SWIGLU, in_bufs=(b_tmp1,), out_bufs=(b_tmp2, b_q, b_s), u=(M_tot, H, EM_pad, 0), f=(inv_silu,)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_down"]), out_bufs=(b_tmp3,), u=(EM_pad, D, H, 0)))
            if self.recurrent:
                self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(b_xm2, b_tmp3, self.buf_scalars), out_bufs=(b_tmp1,), u=(M_tot, D, sc0 + 5, 1), f=(eps_n, eps_l)))
                self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(b_x, b_tmp1, self.buf_scalars), out_bufs=(b_x,), u=(M_tot, D, self._sc_step_base + step, 0), f=(eps_n, eps_l)))
            else:
                self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(b_xm2, b_tmp3, self.buf_scalars), out_bufs=(b_x,), u=(M_tot, D, sc0 + 5, 1), f=(eps_n, eps_l)))

        # 5. Slice H_bot, H_nav, H_team + NAX head projections + 1-NFE Mealy-Jet SRAM sample!
        b_hb = b_obs_enc
        b_hn = b_nav_enc
        b_ht = b_team_enc
        self._add_op(_make_op(MG_OP_GRAPH_SLICE_NORM, in_bufs=(b_x,), out_bufs=(b_hb, b_hn, b_ht), u=(E, N, Wn, J, K, D), f=(eps_n,)))
        b_cb = b_tmp3
        b_ct = b_tmp5
        b_bn = b_tmp6
        b_bt = b_tmp7
        b_vd = b_tmp8
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_hb, w["flow_cond_bot"]), out_bufs=(b_cb,), u=(EN_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ht, w["flow_cond_team"]), out_bufs=(b_ct,), u=(EK_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_hn, w["basis_nav_proj"]), out_bufs=(b_bn,), u=(EW_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ht, w["basis_team_proj"]), out_bufs=(b_bt,), u=(EK_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_hb, w["val_drift_packed"]), out_bufs=(b_vd,), u=(EN_pad, 64, D)))

        self._add_op(_make_op(
            MG_OP_FLOW_JET_SAMPLE,
            in_bufs=(b_cb, b_ct, b_bn, b_bt, w["basis_self"], w["fj_params"]),
            out_bufs=(self.buf_u_vel, self.buf_u_lat, self.buf_val),
            u=(
                E | (N << 16),
                Wn | (J << 8) | (K << 16) | (int(self.cfg.heun_steps) << 24),
                self.buf_eps | (b_vd << 16),
                self.buf_ent_meta | (self.buf_hp << 16),
            ),
        ))

        # Host zero-copy views
        self.obs_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_obs), (E, N, OBS_WIDTH))
        self.nav_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_nav), (E, Wn, 16))
        self.carts_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_carts), (E, J, CART_WIDTH))
        self.teams_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_teams), (E, K, TEAM_WIDTH))
        self.acc_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_acc), (E, N, A_dim))
        self.eps_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_eps), (E, N, 32))
        self.geo_ncfg_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_geo_ncfg), (16,))
        self.u_vel_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_u_vel), (E, N, A_dim))
        self.u_lat_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_u_lat), (E, N, 32))
        self.val_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_val), (E, N, 2))

        self.geo_ncfg_view[4:16].copy_(torch.tensor([
            self.nc.mp_norm_eps, self.nc.map_radius_min, self.nc.pos_fallback_scale, self.nc.vel_scale,
            self.nc.health_offset, self.nc.health_scale, self.nc.armor_offset, self.nc.armor_scale,
            self.nc.ammo_scale, self.nc.strength_max_s, self.nc.cart_len_scale, self.nc.cart_speed_scale,
        ], dtype=torch.float32))

    @torch.no_grad()
    def sync_weights_(self, model: GraphFlowPolicy) -> None:
        eps = float(self.nc.unit_norm_eps)
        w = self._w_bufs
        def put_bf(key: str, t: torch.Tensor) -> None:
            _copy_bf16_to_ptr(t, self.lib.mesh_graph_buffer_ptr(self.g, w[key]))

        put_bf("enc_obs", _unit_pad_bf16(model.enc_obs.weight, eps, cols_pad=128))
        put_bf("enc_nav", _unit_pad_bf16(model.enc_nav.weight, eps, cols_pad=64))
        put_bf("enc_cart", _unit_pad_bf16(model.enc_cart.weight, eps, cols_pad=64))
        put_bf("enc_team", _unit_pad_bf16(model.enc_team.weight, eps, cols_pad=64))
        te_n = F.rms_norm(model.type_embed.detach().to(device="cpu", dtype=torch.float32), (self.d,), eps=max(self.nc.mp_norm_eps, 1e-4))
        put_bf("type_embed", te_n.to(torch.bfloat16))

        n_uniq_blk = 1 if self.recurrent else self.depth
        scalars = [
            float(torch.sigmoid(model.type_blend.detach().float()).item()),
            float(torch.sigmoid(model.aff_blend.detach().float()).item()),
        ]
        for bi in range(n_uniq_blk):
            blk = model.blocks[bi]
            p = f"b{bi}."
            put_bf(p + "nav_val_proj", _unit_pad_bf16(blk.nav_val_proj.weight, eps))
            put_bf(p + "nav_out_proj", _unit_pad_bf16(blk.nav_out_proj.weight, eps))
            put_bf(p + "val_proj", _unit_pad_bf16(blk.val_proj.weight, eps))
            put_bf(p + "global_metric", _unit_pad_bf16(blk.global_metric.weight, eps))
            put_bf(p + "pool_score", _unit_pad_bf16(blk.pool_score.weight, eps).view(-1))
            put_bf(p + "team_metric", _unit_pad_bf16(blk.team_metric.weight, eps))
            put_bf(p + "rival_metric", _unit_pad_bf16(blk.rival_metric.weight, eps))
            put_bf(p + "team_val_proj", _unit_pad_bf16(blk.team_val_proj.weight, eps))
            put_bf(p + "team_mutate_proj", _unit_pad_bf16(blk.team_mutate_proj.weight, eps))
            put_bf(p + "team_bcast_proj", _unit_pad_bf16(blk.team_bcast_proj.weight, eps))
            put_bf(p + "gram_out", _unit_pad_bf16(blk.gram_out.weight, eps))
            put_bf(p + "dpp_b_proj", _unit_pad_bf16(blk.dpp_b_proj.weight, eps, rows_pad=64))
            put_bf(p + "dpp_v_proj", _unit_pad_bf16(blk.dpp_v_proj.weight, eps))
            put_bf(p + "dpp_out_proj", _unit_pad_bf16(blk.dpp_out_proj.weight, eps))
            wg = _unit_pad_bf16(blk.ffn_gate.weight, eps)
            wu = _unit_pad_bf16(blk.ffn_up.weight, eps)
            put_bf(p + "ffn_gate_up", torch.cat((wg, wu), dim=0))
            put_bf(p + "ffn_down", _unit_pad_bf16(blk.ffn_down.weight, eps))
            scalars.extend([
                float(torch.sigmoid(blk.blend_nav.detach().float()).item()),
                float(torch.sigmoid(blk.blend_pool.detach().float()).item()),
                float(torch.sigmoid(blk.blend_team_mut.detach().float()).item()),
                float(torch.sigmoid(blk.blend_gram.detach().float()).item()),
                float(torch.sigmoid(blk.blend_dpp.detach().float()).item()),
                float(torch.sigmoid(blk.blend_ffn.detach().float()).item()),
            ])
        if self.recurrent:
            scalars.extend(float(v) for v in torch.sigmoid(model.step_blends.detach().float()).cpu().tolist()[: self.depth])
        _copy_f32_to_ptr(np.asarray(scalars, dtype=np.float32), self.lib.mesh_graph_buffer_ptr(self.g, self.buf_scalars))

        put_bf("flow_cond_bot", _unit_pad_bf16(model.flow_cond_bot.weight, eps))
        put_bf("flow_cond_team", _unit_pad_bf16(model.flow_cond_team.weight, eps))
        put_bf("basis_nav_proj", _unit_pad_bf16(model.basis_nav_proj.weight, eps, rows_pad=64))
        put_bf("basis_team_proj", _unit_pad_bf16(model.basis_team_proj.weight, eps, rows_pad=64))
        vd = torch.zeros((64, self.d), dtype=torch.bfloat16)
        vd[0:2] = _unit_pad_bf16(model.value_proj.weight, eps)
        vd[2:3] = _unit_pad_bf16(model.dyn_drift_proj.weight, eps)
        put_bf("val_drift_packed", vd)

        _copy_f32_to_ptr(model.basis_self.detach().float().cpu().numpy(), self.lib.mesh_graph_buffer_ptr(self.g, w["basis_self"]))
        fj_arr = np.concatenate([
            model.fj_w1.weight.detach().float().cpu().numpy().reshape(-1),
            model.fj_w1.bias.detach().float().cpu().numpy().reshape(-1),
            model.fj_w2.weight.detach().float().cpu().numpy().reshape(-1),
            model.fj_w2.bias.detach().float().cpu().numpy().reshape(-1),
        ])
        _copy_f32_to_ptr(fj_arr, self.lib.mesh_graph_buffer_ptr(self.g, w["fj_params"]))

        vg = float(model.value_gain.detach().float().clamp(*self.cfg.value_gain_bounds).item())
        vb = model.value_bias.detach().float().cpu().tolist()
        self._hp_base = np.array([
            vg, vb[0], vb[1], float(self.cfg.u_clip), 1.0 / float(self.nc.mp_silu_scale),
            float(self.nc.mp_norm_eps), float(self.nc.mp_unit_vec_eps), 1.0,
        ], dtype=np.float32)
        _copy_f32_to_ptr(self._hp_base, self.lib.mesh_graph_buffer_ptr(self.g, self.buf_hp))

        if self._g_repack:
            wt = ctypes.c_double(0.0)
            self.lib.mesh_graph_run_fused(self._g_repack, 1, 0, ctypes.byref(wt))

    def set_geo(self, geo: dict[str, torch.Tensor] | None) -> None:
        if geo is not None and "map_center" in geo and "map_radius" in geo:
            self.geo_ncfg_view[0:3].copy_(geo["map_center"].detach().to(device="cpu", dtype=torch.float32).view(3))
            self.geo_ncfg_view[3] = float(geo["map_radius"].detach().to(device="cpu", dtype=torch.float32).item())
        else:
            self.geo_ncfg_view[0:4].zero_()

    def set_nav_kern(self, nav_kern: torch.Tensor) -> None:
        _copy_f32_to_ptr(
            nav_kern.detach().to(device="cpu", dtype=torch.float32).numpy(),
            self.lib.mesh_graph_buffer_ptr(self.g, self.buf_nav_kern),
        )

    @torch.no_grad()
    def sample_velocity(
        self,
        obs: torch.Tensor,
        nav: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        acc: torch.Tensor,
        geo: dict[str, torch.Tensor] | None = None,
        *,
        eps: torch.Tensor | None = None,
        output_scale: float = 1.0,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        self.set_geo(geo)
        self._hp_base[7] = float(output_scale)
        _copy_f32_to_ptr(self._hp_base, self.lib.mesh_graph_buffer_ptr(self.g, self.buf_hp))
        self.obs_view.copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
        self.nav_view.copy_(nav.detach().to(device="cpu", dtype=torch.bfloat16))
        self.carts_view.copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
        self.teams_view.copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
        self.acc_view.copy_(acc.detach().to(device="cpu", dtype=torch.float32))
        if eps is not None:
            self.eps_view.copy_(eps.detach().to(device="cpu", dtype=torch.float32))
        elif deterministic:
            self.eps_view.zero_()
        else:
            self.eps_view.normal_(0.0, float(self.cfg.sigma_base))

        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))
        return {
            "u_vel": self.u_vel_view.clone(),
            "u_lat": self.u_lat_view.clone(),
            "value_winnie": self.val_view[:, :, 0].clone(),
            "value_lou": self.val_view[:, :, 1].clone(),
            "wall_ms": float(wall.value),
        }


class CompiledGraphFlowTrainer:
    """Full Metal 4 NAX forward + Advantage-Weighted Boltzmann Flow Matching + Jet-Straightening +
    exact analytical backward + FP32 gradient accumulation inside ONE command buffer!
    """

    def __init__(
        self,
        model: GraphFlowPolicy,
        E_mb: int,
        N: int,
        W_nav: int,
        J: int,
        K: int,
        *,
        nav_kern: torch.Tensor,
    ):
        self.lib = _load_mesh_graph_lib()
        self.E = int(E_mb)
        self.N = int(N)
        self.W_nav = int(W_nav)
        self.J = int(J)
        self.K = int(K)
        self.M_ent = self.N + self.W_nav + self.J + self.K
        self.A_dim = self.W_nav + self.K + 4
        self.cfg = model.cfg
        self.nc = model.norm_cfg
        self.d = int(model.d)
        self.h = int(model.h)
        self.r_gram = int(self.cfg.r_gram)
        self.r_team = int(self.cfg.r_team)
        self.r_dpp = int(self.cfg.r_dpp)
        self.depth = int(model.depth)
        self.recurrent = bool(model.recurrent)

        self.g = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for CompiledGraphFlowTrainer")

        self._w_bufs: dict[str, int] = {}
        self._wq_bufs: dict[str, tuple[int, int]] = {}
        self._repack_specs: list[tuple[str, int, int]] = []
        self._g_repack: int | None = None
        self._gw_bufs: dict[str, int] = {}
        self._gw_shapes: dict[str, tuple[int, ...]] = {}
        self._build_graph(model, nav_kern)
        self.sync_weights_(model)
        rc = self.lib.mesh_graph_compile(self.g)
        if rc != 0:
            raise RuntimeError(f"mesh_graph_compile failed for CompiledGraphFlowTrainer: {rc}")

    def close(self) -> None:
        if getattr(self, "_g_repack", None):
            self.lib.mesh_graph_destroy(self._g_repack)
            self._g_repack = None
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    def __del__(self) -> None:
        self.close()

    def _add_buf(self, nbytes: int, kind: int = MG_BUF_STATE) -> int:
        return int(self.lib.mesh_graph_add_buffer(self.g, int(max(64, nbytes)), int(kind), None, -1, 0))

    def _add_op(self, op: Any) -> None:
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))

    def _add_w_and_gw(self, key: str, rows: int, cols: int) -> None:
        self._w_bufs[key] = self._add_buf(rows * cols * 2, MG_BUF_WEIGHT)
        self._gw_bufs[key] = self._add_buf(rows * cols * 4, MG_BUF_STATE)
        self._gw_shapes[key] = (rows, cols)

    def _build_graph(self, model: GraphFlowPolicy, nav_kern: torch.Tensor) -> None:
        E, N, Wn, J, K, M_ent, A_dim = self.E, self.N, self.W_nav, self.J, self.K, self.M_ent, self.A_dim
        D, H, Rg, Rt, Rd = self.d, self.h, self.r_gram, self.r_team, self.r_dpp
        EN_pad = _align(E * N, 64)
        EW_pad = _align(E * Wn, 64)
        EJ_pad = _align(E * J, 64)
        EK_pad = _align(E * K, 64)
        EM_pad = _align(E * M_ent, 64)
        M_fj = 4 * E * N
        M_fj_pad = _align(M_fj, 64)
        M_tot, K_tot, EN_tot = E * M_ent, E * K, E * N

        # IO & target buffers
        self.buf_obs = self._add_buf(E * N * OBS_WIDTH * 2)
        self.buf_nav = self._add_buf(E * Wn * 16 * 2)
        self.buf_carts = self._add_buf(E * J * CART_WIDTH * 2)
        self.buf_teams = self._add_buf(E * K * TEAM_WIDTH * 2)
        self.buf_acc = self._add_buf(E * N * A_dim * 4)
        self.buf_u_targ = self._add_buf(E * N * A_dim * 4)
        self.buf_eps_tr = self._add_buf(4 * E * N * 32 * 4)
        self.buf_targ_f = self._add_buf(E * N * 6 * 4)
        self.buf_geo_ncfg = self._add_buf(16 * 4)
        self.buf_lcfg = self._add_buf(24 * 4)
        self.buf_taus = self._add_buf(16 * 4, MG_BUF_WEIGHT)
        self.buf_nav_kern = self._add_buf(Wn * Wn * 4, MG_BUF_WEIGHT)
        self.buf_ent_meta = self._add_buf(E * (M_ent + K) * 4)
        self.buf_bot_nid = self._add_buf(E * N * 4)

        _copy_f32_to_ptr(
            nav_kern.detach().to(device="cpu", dtype=torch.float32).numpy(),
            self.lib.mesh_graph_buffer_ptr(self.g, self.buf_nav_kern),
        )
        _copy_f32_to_ptr(
            np.asarray(list(self.cfg.train_taus), dtype=np.float32),
            self.lib.mesh_graph_buffer_ptr(self.g, self.buf_taus),
        )

        n_uniq_blk = 1 if self.recurrent else self.depth
        self._sc_blk_base = 2
        self._sc_step_base = 2 + 6 * n_uniq_blk
        n_scalars = self._sc_step_base + (self.depth if self.recurrent else 0)
        self.buf_scalars = self._add_buf(max(16, n_scalars) * 4)

        # Scalar grads: [0..4]=val/dyn gains&bias, [5]=type_blend, [6]=aff_blend, [7..]=blk blends, step_blends
        self._sg_blk_base = 7
        self._sg_step_base = 7 + 6 * n_uniq_blk
        self._n_sc_grads = max(64, self._sg_step_base + self.depth)
        self.buf_sc_grads = self._add_buf(self._n_sc_grads * 4)
        self.buf_diag = self._add_buf(16 * 4)

        # Weights + FP32 grads
        self._add_w_and_gw("enc_obs", D, 128)
        self._add_w_and_gw("enc_nav", D, 64)
        self._add_w_and_gw("enc_cart", D, 64)
        self._add_w_and_gw("enc_team", D, 64)
        self._w_bufs["type_embed"] = self._add_buf(4 * D * 2, MG_BUF_WEIGHT)
        self._gw_bufs["type_embed"] = self._add_buf(4 * D * 4)
        self._gw_shapes["type_embed"] = (4, D)

        def add_i8_w_and_gw(key: str, rows: int, cols: int) -> None:
            self._add_w_and_gw(key, rows, cols)
            self._wq_bufs[key] = (self._add_buf(rows * cols), self._add_buf(rows * 4))
            self._repack_specs.append((key, rows, cols))

        for bi in range(n_uniq_blk):
            p = f"b{bi}."
            self._add_w_and_gw(p + "nav_val_proj", D, D)
            self._add_w_and_gw(p + "nav_out_proj", D, D)
            add_i8_w_and_gw(p + "val_proj", D, D)
            add_i8_w_and_gw(p + "global_metric", Rg, D)
            self._w_bufs[p + "pool_score"] = self._add_buf(D * 2, MG_BUF_WEIGHT)
            self._gw_bufs[p + "pool_score"] = self._add_buf(D * 4)
            self._gw_shapes[p + "pool_score"] = (1, D)
            self._add_w_and_gw(p + "team_metric", Rt, D)
            self._add_w_and_gw(p + "rival_metric", Rt, D)
            self._add_w_and_gw(p + "team_val_proj", D, D)
            self._add_w_and_gw(p + "team_mutate_proj", D, D)
            self._add_w_and_gw(p + "team_bcast_proj", D, D)
            self._add_w_and_gw(p + "gram_out", D, D)
            self._add_w_and_gw(p + "dpp_b_proj", 64, D)
            self._add_w_and_gw(p + "dpp_v_proj", D, D)
            self._add_w_and_gw(p + "dpp_out_proj", D, D)
            add_i8_w_and_gw(p + "ffn_gate_up", 2 * H, D)
            add_i8_w_and_gw(p + "ffn_down", D, H)

        self._add_w_and_gw("flow_cond_bot", 64, D)
        self._add_w_and_gw("flow_cond_team", 64, D)
        self._add_w_and_gw("basis_nav_proj", 64, D)
        self._add_w_and_gw("basis_team_proj", 64, D)
        self._add_w_and_gw("val_drift_packed", 64, D)
        self._add_w_and_gw("fj_w1", 128, 128)
        self._add_w_and_gw("fj_w2", 64, 64)

        self._w_bufs["basis_self"] = self._add_buf(4 * 32 * 4, MG_BUF_WEIGHT)
        self._gw_bufs["basis_self"] = self._add_buf(4 * 32 * 4)
        self._gw_shapes["basis_self"] = (4, 32)
        self._w_bufs["fj_b2"] = self._add_buf(32 * 4, MG_BUF_WEIGHT)
        self._gw_bufs["fj_b2"] = self._add_buf(32 * 4)
        self._gw_shapes["fj_b2"] = (32,)
        self._w_bufs["dyn_action_proj"] = self._add_buf(32 * 4, MG_BUF_WEIGHT)
        self._gw_bufs["dyn_action_proj"] = self._add_buf(32 * 4)
        self._gw_shapes["dyn_action_proj"] = (1, 32)

        w, wq, gw = self._w_bufs, self._wq_bufs, self._gw_bufs

        self._g_repack = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        for key, rows, cols in self._repack_specs:
            iw = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, w[key])
            iq = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][0])
            is_ = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][1])
            op_r = _make_op(MG_OP_FWHT_PACK_I8, in_bufs=(iw,), out_bufs=(iq, is_), u=(rows, cols, 0))
            self.lib.mesh_graph_add_op(self._g_repack, ctypes.byref(op_r))
        self.lib.mesh_graph_compile(self._g_repack)

        b_q = self._add_buf(EM_pad * max(D, H))
        b_s = self._add_buf(EM_pad * 4)

        eps_n = float(self.nc.mp_norm_eps)
        eps_l = float(self.nc.mp_lerp_eps)
        eps_u = float(self.nc.mp_unit_vec_eps)
        inv_silu = 1.0 / float(self.nc.mp_silu_scale)

        # ======================== FORWARD PASS ========================
        b_obs_pre = self._add_buf(EN_pad * 128 * 2)
        b_nav_pre = self._add_buf(EW_pad * 64 * 2)
        b_cart_pre = self._add_buf(EJ_pad * 64 * 2)
        b_team_pre = self._add_buf(EK_pad * 64 * 2)
        b_obs_enc = self._add_buf(EN_pad * D * 2)
        b_nav_enc = self._add_buf(EW_pad * D * 2)
        b_cart_enc = self._add_buf(EJ_pad * D * 2)
        b_team_enc = self._add_buf(EK_pad * D * 2)
        b_h_obs0 = self._add_buf(EN_pad * D * 2)
        b_aff_raw = self._add_buf(EN_pad * D * 2)

        self._add_op(_make_op(
            MG_OP_GRAPH_PRENORM,
            in_bufs=(self.buf_obs, self.buf_nav, self.buf_carts, self.buf_teams, self.buf_acc, self.buf_geo_ncfg),
            out_bufs=(b_obs_pre, b_nav_pre, b_cart_pre),
            u=(E | (N << 16), Wn | (J << 16), K | (A_dim << 16), b_team_pre, self.buf_ent_meta, self.buf_bot_nid),
        ))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_obs_pre, w["enc_obs"]), out_bufs=(b_obs_enc,), u=(EN_pad, D, 128)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_nav_pre, w["enc_nav"]), out_bufs=(b_nav_enc,), u=(EW_pad, D, 64)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_cart_pre, w["enc_cart"]), out_bufs=(b_cart_enc,), u=(EJ_pad, D, 64)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_team_pre, w["enc_team"]), out_bufs=(b_team_enc,), u=(EK_pad, D, 64)))

        x_states = [self._add_buf(EM_pad * D * 2) for _ in range(self.depth + 1)]
        self._add_op(_make_op(
            MG_OP_GRAPH_EMBED_CAT,
            in_bufs=(b_obs_enc, b_nav_enc, b_cart_enc, b_team_enc, w["type_embed"], self.buf_acc),
            out_bufs=(x_states[0], b_h_obs0, b_aff_raw),
            u=(E | (N << 16), Wn | (J << 16), K | (D << 16), 0 | (1 << 16), self.buf_bot_nid, self.buf_scalars, 1),
            f=(eps_n, eps_l),
        ))

        J_eff = Wn + J
        st_acts: list[dict[str, int]] = []
        for s in range(self.depth):
            bi = 0 if self.recurrent else s
            p = f"b{bi}."
            sc0 = self._sc_blk_base + 6 * bi
            a = {
                "xn": self._add_buf(EM_pad * D * 2),
                "x_nav": self._add_buf(EW_pad * D * 2),
                "v_nav_raw": self._add_buf(EW_pad * D * 2),
                "v_nav": self._add_buf(EW_pad * D * 2),
                "nav_diff": self._add_buf(EW_pad * D * 2),
                "nav_msg": self._add_buf(EW_pad * D * 2),
                "xg": self._add_buf(EM_pad * D * 2),
                "v_raw": self._add_buf(EM_pad * D * 2),
                "v_norm": self._add_buf(EM_pad * D * 2),
                "r_raw": self._add_buf(EM_pad * Rg * 2),
                "r_norm": self._add_buf(EM_pad * Rg * 2),
                "global_raw": self._add_buf(EM_pad * D * 2),
                "t_in": self._add_buf(EK_pad * D * 2),
                "qt_raw": self._add_buf(EK_pad * Rt * 2),
                "kt_raw": self._add_buf(EK_pad * Rt * 2),
                "vt_raw": self._add_buf(EK_pad * D * 2),
                "t_mixed": self._add_buf(EK_pad * D * 2),
                "t_mut_raw": self._add_buf(EK_pad * D * 2),
                "t_mut": self._add_buf(EK_pad * D * 2),
                "tb_raw": self._add_buf(EK_pad * D * 2),
                "gram_in": self._add_buf(EM_pad * D * 2),
                "gram_out_raw": self._add_buf(EM_pad * D * 2),
                "x_mid": self._add_buf(EM_pad * D * 2),
                "x_bot": self._add_buf(EN_pad * D * 2),
                "dpp_b_raw": self._add_buf(EN_pad * 64 * 2),
                "dpp_v_raw": self._add_buf(EN_pad * D * 2),
                "dpp_v_norm": self._add_buf(EN_pad * D * 2),
                "dpp_y": self._add_buf(EN_pad * D * 2),
                "dpp_raw": self._add_buf(EN_pad * D * 2),
                "x_mid2": self._add_buf(EM_pad * D * 2),
                "gate_up": self._add_buf(EM_pad * 2 * H * 2),
                "hid": self._add_buf(EM_pad * H * 2),
                "ffn_raw": self._add_buf(EM_pad * D * 2),
                "x_new": self._add_buf(EM_pad * D * 2),
            }
            st_acts.append(a)

            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(x_states[s],), out_bufs=(a["xn"],), u=(M_tot, D), f=(eps_n,)))
            # Nav-graph mix
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX, in_bufs=(a["xn"],), out_bufs=(a["x_nav"],), u=(E, N, Wn, M_ent, D, 0, 0, EW_pad)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["x_nav"], w[p + "nav_val_proj"]), out_bufs=(a["v_nav_raw"],), u=(EW_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(a["v_nav_raw"],), out_bufs=(a["v_nav"],), u=(E * Wn, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX, in_bufs=(a["v_nav"], -1, self.buf_nav_kern), out_bufs=(a["nav_diff"],), u=(E, N, Wn, M_ent, D, 1, 0, EW_pad)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["nav_diff"], w[p + "nav_out_proj"]), out_bufs=(a["nav_msg"],), u=(EW_pad, D, D)))
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX, in_bufs=(a["xn"], a["nav_msg"], -1, self.buf_bot_nid, self.buf_scalars), out_bufs=(a["xg"],), u=(E, N, Wn, M_ent, D, 2, sc0 + 0, EM_pad), f=(eps_n, eps_l)))

            # Global + Team Rival Gram
            self._add_op(_make_op(MG_OP_FWHT_PACK_I8, in_bufs=(a["xg"],), out_bufs=(b_q, b_s), u=(EM_pad, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "val_proj"]), out_bufs=(a["v_raw"],), u=(EM_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "global_metric"]), out_bufs=(a["r_raw"],), u=(EM_pad, Rg, D, 0)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(a["v_raw"],), out_bufs=(a["v_norm"],), u=(M_tot, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(a["r_raw"],), out_bufs=(a["r_norm"],), u=(M_tot, Rg), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_CGT_GRAM_STAGE1, in_bufs=(a["xg"], a["v_norm"], a["r_norm"], w[p + "pool_score"], self.buf_ent_meta, self.buf_scalars), out_bufs=(a["global_raw"], a["t_in"]), u=(E, N, J_eff, K, D, Rg, sc0 + 1), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_in"], w[p + "team_metric"]), out_bufs=(a["qt_raw"],), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_in"], w[p + "rival_metric"]), out_bufs=(a["kt_raw"],), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_in"], w[p + "team_val_proj"]), out_bufs=(a["vt_raw"],), u=(EK_pad, D, D)))
            self._add_op(_make_op(MG_OP_CGT_TEAM_RIVAL, in_bufs=(a["qt_raw"], a["kt_raw"], a["vt_raw"]), out_bufs=(a["t_mixed"],), u=(E, K, D, Rt), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_mixed"], w[p + "team_mutate_proj"]), out_bufs=(a["t_mut_raw"],), u=(EK_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(a["t_in"], a["t_mut_raw"], self.buf_scalars), out_bufs=(a["t_mut"],), u=(K_tot, D, sc0 + 2, 1), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_mut"], w[p + "team_bcast_proj"]), out_bufs=(a["tb_raw"],), u=(EK_pad, D, D)))
            self._add_op(_make_op(MG_OP_CGT_BCAST_MERGE, in_bufs=(a["global_raw"], a["tb_raw"], self.buf_ent_meta), out_bufs=(a["gram_in"],), u=(E, M_ent, K, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["gram_in"], w[p + "gram_out"]), out_bufs=(a["gram_out_raw"],), u=(EM_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(a["xg"], a["gram_out_raw"], self.buf_scalars), out_bufs=(a["x_mid"],), u=(M_tot, D, sc0 + 3, 1), f=(eps_n, eps_l)))

            # Exact FP32 Per-Team DPP Diversity Kernel
            self._add_op(_make_op(MG_OP_DPP_GRAM, in_bufs=(a["x_mid"],), out_bufs=(a["x_bot"],), u=(E, N, M_ent, K, D, Rd, 0, EN_pad)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["x_bot"], w[p + "dpp_b_proj"]), out_bufs=(a["dpp_b_raw"],), u=(EN_pad, 64, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["x_bot"], w[p + "dpp_v_proj"]), out_bufs=(a["dpp_v_raw"],), u=(EN_pad, D, D)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(a["dpp_v_raw"],), out_bufs=(a["dpp_v_norm"],), u=(EN_tot, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_DPP_GRAM, in_bufs=(a["dpp_b_raw"], a["dpp_v_norm"], self.buf_ent_meta), out_bufs=(a["dpp_y"],), u=(E, N, M_ent, K, D, Rd, 1, 0), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["dpp_y"], w[p + "dpp_out_proj"]), out_bufs=(a["dpp_raw"],), u=(EN_pad, D, D)))
            self._add_op(_make_op(MG_OP_DPP_GRAM, in_bufs=(a["x_mid"], a["dpp_raw"], -1, self.buf_scalars), out_bufs=(a["x_mid2"],), u=(E, N, M_ent, K, D, Rd, 2, sc0 + 4), f=(eps_n, eps_l)))

            # Wide FFN (INT8@8 NAX)
            self._add_op(_make_op(MG_OP_FWHT_PACK_I8, in_bufs=(a["x_mid2"],), out_bufs=(b_q, b_s), u=(EM_pad, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_gate_up"]), out_bufs=(a["gate_up"],), u=(EM_pad, 2 * H, D, 0)))
            self._add_op(_make_op(MG_OP_MP_SWIGLU, in_bufs=(a["gate_up"],), out_bufs=(a["hid"], b_q, b_s), u=(M_tot, H, EM_pad, 1), f=(inv_silu,)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_down"]), out_bufs=(a["ffn_raw"],), u=(EM_pad, D, H, 0)))
            if self.recurrent:
                self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(a["x_mid2"], a["ffn_raw"], self.buf_scalars), out_bufs=(a["x_new"],), u=(M_tot, D, sc0 + 5, 1), f=(eps_n, eps_l)))
                self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(x_states[s], a["x_new"], self.buf_scalars), out_bufs=(x_states[s + 1],), u=(M_tot, D, self._sc_step_base + s, 0), f=(eps_n, eps_l)))
            else:
                self._add_op(_make_op(MG_OP_MP_LERP, in_bufs=(a["x_mid2"], a["ffn_raw"], self.buf_scalars), out_bufs=(x_states[s + 1],), u=(M_tot, D, sc0 + 5, 1), f=(eps_n, eps_l)))

        # Final slice + head projections + Mealy-Jet NAX MLP!
        b_hb = self._add_buf(EN_pad * D * 2)
        b_hn = self._add_buf(EW_pad * D * 2)
        b_ht = self._add_buf(EK_pad * D * 2)
        b_cb = self._add_buf(EN_pad * 64 * 2)
        b_ct = self._add_buf(EK_pad * 64 * 2)
        b_bn = self._add_buf(EW_pad * 64 * 2)
        b_bt = self._add_buf(EK_pad * 64 * 2)
        b_vd = self._add_buf(EN_pad * 64 * 2)
        b_BH = self._add_buf(E * A_dim * 32 * 4)
        b_ul = self._add_buf(E * N * 32 * 4)
        b_inp_fj = self._add_buf(M_fj_pad * 128 * 2)
        b_zt = self._add_buf(4 * E * N * 32 * 4)
        b_fj_gu = self._add_buf(M_fj_pad * 128 * 2)
        b_fj_hid = self._add_buf(M_fj_pad * 64 * 2)
        b_vj_raw = self._add_buf(M_fj_pad * 64 * 2)

        self._add_op(_make_op(MG_OP_GRAPH_SLICE_NORM, in_bufs=(x_states[self.depth],), out_bufs=(b_hb, b_hn, b_ht), u=(E, N, Wn, J, K, D), f=(eps_n,)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_hb, w["flow_cond_bot"]), out_bufs=(b_cb,), u=(EN_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ht, w["flow_cond_team"]), out_bufs=(b_ct,), u=(EK_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_hn, w["basis_nav_proj"]), out_bufs=(b_bn,), u=(EW_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ht, w["basis_team_proj"]), out_bufs=(b_bt,), u=(EK_pad, 64, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_hb, w["val_drift_packed"]), out_bufs=(b_vd,), u=(EN_pad, 64, D)))

        self._add_op(_make_op(
            MG_OP_FLOW_JET_PREP,
            in_bufs=(b_cb, b_ct, b_bn, b_bt, w["basis_self"], self.buf_u_targ),
            out_bufs=(b_BH, b_ul, b_inp_fj),
            u=(E | (N << 16), Wn | (J << 8) | (K << 16), M_fj_pad, self.buf_eps_tr | (self.buf_ent_meta << 16), self.buf_taus | (b_zt << 16)),
            f=(eps_n, eps_u),
        ))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_inp_fj, w["fj_w1"]), out_bufs=(b_fj_gu,), u=(M_fj_pad, 128, 128)))
        self._add_op(_make_op(MG_OP_MP_SWIGLU, in_bufs=(b_fj_gu,), out_bufs=(b_fj_hid,), u=(M_fj, 64, M_fj_pad, 1), f=(inv_silu,)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_fj_hid, w["fj_w2"]), out_bufs=(b_vj_raw,), u=(M_fj_pad, 64, 64)))

        # ======================== BACKWARD PASS ========================
        bd_vj = self._add_buf(M_fj_pad * 64 * 2)
        bd_fj_hid = self._add_buf(M_fj_pad * 64 * 2)
        bd_fj_gu = self._add_buf(M_fj_pad * 128 * 2)
        bd_inp_fj = self._add_buf(M_fj_pad * 128 * 2)
        bd_cb = self._add_buf(EN_pad * 64 * 2)
        bd_ct = self._add_buf(EK_pad * 64 * 2)
        bd_bn = self._add_buf(EW_pad * 64 * 2)
        bd_bt = self._add_buf(EK_pad * 64 * 2)
        bd_vd = self._add_buf(EN_pad * 64 * 2)
        bd_hb = self._add_buf(EN_pad * D * 2)
        bd_hn = self._add_buf(EW_pad * D * 2)
        bd_ht = self._add_buf(EK_pad * D * 2)

        bd_x = self._add_buf(EM_pad * D * 2)
        bd_x_prev = self._add_buf(EM_pad * D * 2)
        bd_x_new = self._add_buf(EM_pad * D * 2)
        bd_xm2 = self._add_buf(EM_pad * D * 2)
        bd_xm = self._add_buf(EM_pad * D * 2)
        bd_xg = self._add_buf(EM_pad * D * 2)
        bd_xn = self._add_buf(EM_pad * D * 2)
        bd_ffn_raw = self._add_buf(EM_pad * D * 2)
        bd_hid = self._add_buf(EM_pad * H * 2)
        bd_gu = self._add_buf(EM_pad * 2 * H * 2)
        bd_dpp_raw = self._add_buf(EN_pad * D * 2)
        bd_dpp_y = self._add_buf(EN_pad * D * 2)
        bd_dpp_b = self._add_buf(EN_pad * 64 * 2)
        bd_dpp_vn = self._add_buf(EN_pad * D * 2)
        bd_dpp_vr = self._add_buf(EN_pad * D * 2)
        bd_x_bot = self._add_buf(EN_pad * D * 2)
        bd_gram_out = self._add_buf(EM_pad * D * 2)
        bd_gram_in = self._add_buf(EM_pad * D * 2)
        bd_glob = self._add_buf(EM_pad * D * 2)
        bd_tb = self._add_buf(EK_pad * D * 2)
        bd_tmut = self._add_buf(EK_pad * D * 2)
        bd_tmut_raw = self._add_buf(EK_pad * D * 2)
        bd_tin = self._add_buf(EK_pad * D * 2)
        bd_tmixed = self._add_buf(EK_pad * D * 2)
        bd_qt = self._add_buf(EK_pad * Rt * 2)
        bd_kt = self._add_buf(EK_pad * Rt * 2)
        bd_vt = self._add_buf(EK_pad * D * 2)
        bd_vnorm = self._add_buf(EM_pad * D * 2)
        bd_rnorm = self._add_buf(EM_pad * Rg * 2)
        bd_vraw = self._add_buf(EM_pad * D * 2)
        bd_rraw = self._add_buf(EM_pad * Rg * 2)
        bd_nav_msg = self._add_buf(EW_pad * D * 2)
        bd_nav_diff = self._add_buf(EW_pad * D * 2)
        bd_v_nav = self._add_buf(EW_pad * D * 2)
        bd_v_nav_raw = self._add_buf(EW_pad * D * 2)
        bd_x_nav = self._add_buf(EW_pad * D * 2)

        # 1. Loss & Mealy-Jet head backward
        self._add_op(_make_op(
            MG_OP_FLOW_JET_LOSS_BWD,
            in_bufs=(b_vj_raw, w["fj_b2"], b_BH, b_bn, b_bt, w["basis_self"]),
            out_bufs=(bd_vj, bd_bn, bd_bt),
            u=(
                E | (N << 16) | (Wn << 24),
                K | (self.buf_sc_grads << 8) | (self.buf_diag << 20),
                b_zt | (self.buf_eps_tr << 16),
                self.buf_u_targ | (b_vd << 16),
                b_ul | (w["dyn_action_proj"] << 16),
                self.buf_targ_f | (self.buf_lcfg << 16),
                gw["fj_b2"] | (gw["basis_self"] << 16),
                bd_vd | (gw["dyn_action_proj"] << 16),
            ),
        ))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_vj, b_fj_hid), out_bufs=(gw["fj_w2"],), u=(64, 64, M_fj_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_vj, w["fj_w2"]), out_bufs=(bd_fj_hid,), u=(M_fj_pad, 64, 64, 0)))
        self._add_op(_make_op(MG_OP_MP_SWIGLU_BWD, in_bufs=(b_fj_gu, bd_fj_hid), out_bufs=(bd_fj_gu,), u=(M_fj, 64, M_fj_pad), f=(inv_silu,)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_fj_gu, b_inp_fj), out_bufs=(gw["fj_w1"],), u=(128, 128, M_fj_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_fj_gu, w["fj_w1"]), out_bufs=(bd_inp_fj,), u=(M_fj_pad, 128, 128, 0)))
        self._add_op(_make_op(MG_OP_FLOW_COND_BWD, in_bufs=(b_cb, b_ct, bd_inp_fj, self.buf_ent_meta), out_bufs=(bd_cb, bd_ct), u=(E, N, M_ent, K), f=(eps_n,)))

        # 2. Head linear dW & dX -> bd_hb, bd_hn, bd_ht
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_cb, b_hb), out_bufs=(gw["flow_cond_bot"],), u=(64, D, EN_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_ct, b_ht), out_bufs=(gw["flow_cond_team"],), u=(64, D, EK_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_bn, b_hn), out_bufs=(gw["basis_nav_proj"],), u=(64, D, EW_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_bt, b_ht), out_bufs=(gw["basis_team_proj"],), u=(64, D, EK_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_vd, b_hb), out_bufs=(gw["val_drift_packed"],), u=(64, D, EN_pad, 1)))

        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_cb, w["flow_cond_bot"]), out_bufs=(bd_hb,), u=(EN_pad, D, 64, 0)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_vd, w["val_drift_packed"]), out_bufs=(bd_hb,), u=(EN_pad, D, 64, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_bn, w["basis_nav_proj"]), out_bufs=(bd_hn,), u=(EW_pad, D, 64, 0)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_ct, w["flow_cond_team"]), out_bufs=(bd_ht,), u=(EK_pad, D, 64, 0)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_bt, w["basis_team_proj"]), out_bufs=(bd_ht,), u=(EK_pad, D, 64, 1)))

        self._add_op(_make_op(MG_OP_GRAPH_SLICE_BWD, in_bufs=(x_states[self.depth], bd_hb, bd_hn, bd_ht), out_bufs=(bd_x,), u=(E, N, Wn, J, K, D, EM_pad), f=(eps_n,)))

        # 3. Reverse loop through depth blocks
        cur_dx, nxt_dx = bd_x, bd_x_prev
        for s in range(self.depth - 1, -1, -1):
            bi = 0 if self.recurrent else s
            p = f"b{bi}."
            sc0 = self._sc_blk_base + 6 * bi
            sg0 = self._sg_blk_base + 6 * bi
            a = st_acts[s]

            if self.recurrent:
                self._add_op(_make_op(MG_OP_MP_LERP_BWD, in_bufs=(x_states[s], a["x_new"], cur_dx, self.buf_scalars), out_bufs=(nxt_dx, bd_x_new, self.buf_sc_grads), u=(M_tot, D, self._sc_step_base + s, 0, self._sg_step_base + s, 0, EM_pad), f=(eps_n, eps_l)))
                dx_blk_out = bd_x_new
            else:
                dx_blk_out = cur_dx

            # FFN bwd -> bd_xm2
            self._add_op(_make_op(MG_OP_MP_LERP_BWD, in_bufs=(a["x_mid2"], a["ffn_raw"], dx_blk_out, self.buf_scalars), out_bufs=(bd_xm2, bd_ffn_raw, self.buf_sc_grads), u=(M_tot, D, sc0 + 5, 1, sg0 + 5, 0, EM_pad), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_ffn_raw, a["hid"]), out_bufs=(gw[p + "ffn_down"],), u=(D, H, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_ffn_raw, w[p + "ffn_down"]), out_bufs=(bd_hid,), u=(EM_pad, H, D, 0)))
            self._add_op(_make_op(MG_OP_MP_SWIGLU_BWD, in_bufs=(a["gate_up"], bd_hid), out_bufs=(bd_gu,), u=(M_tot, H, EM_pad), f=(inv_silu,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_gu, a["x_mid2"]), out_bufs=(gw[p + "ffn_gate_up"],), u=(2 * H, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_gu, w[p + "ffn_gate_up"]), out_bufs=(bd_xm2,), u=(EM_pad, D, 2 * H, 1)))

            # DPP bwd -> bd_xm
            self._add_op(_make_op(MG_OP_DPP_GRAM_BWD, in_bufs=(a["x_mid"], a["dpp_raw"], bd_xm2, -1, self.buf_scalars), out_bufs=(bd_xm, bd_dpp_raw, self.buf_sc_grads), u=(E, N, M_ent, K, D, Rd, 2, (sc0 + 4) | ((sg0 + 4) << 16)), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_dpp_raw, a["dpp_y"]), out_bufs=(gw[p + "dpp_out_proj"],), u=(D, D, EN_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_dpp_raw, w[p + "dpp_out_proj"]), out_bufs=(bd_dpp_y,), u=(EN_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_DPP_GRAM_BWD, in_bufs=(a["dpp_b_raw"], a["dpp_v_norm"], bd_dpp_y, self.buf_ent_meta), out_bufs=(bd_dpp_b, bd_dpp_vn), u=(E, N, M_ent, K, D, Rd, 1, 0), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(a["dpp_v_raw"], bd_dpp_vn), out_bufs=(bd_dpp_vr,), u=(EN_tot, D, 0, EN_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_dpp_b, a["x_bot"]), out_bufs=(gw[p + "dpp_b_proj"],), u=(64, D, EN_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_dpp_vr, a["x_bot"]), out_bufs=(gw[p + "dpp_v_proj"],), u=(D, D, EN_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_dpp_b, w[p + "dpp_b_proj"]), out_bufs=(bd_x_bot,), u=(EN_pad, D, 64, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_dpp_vr, w[p + "dpp_v_proj"]), out_bufs=(bd_x_bot,), u=(EN_pad, D, D, 1)))
            self._add_op(_make_op(MG_OP_DPP_GRAM_BWD, in_bufs=(bd_x_bot,), out_bufs=(bd_xm,), u=(E, N, M_ent, K, D, Rd, 0, 0)))

            # Global + Team Rival Gram bwd -> bd_xg
            self._add_op(_make_op(MG_OP_MP_LERP_BWD, in_bufs=(a["xg"], a["gram_out_raw"], bd_xm, self.buf_scalars), out_bufs=(bd_xg, bd_gram_out, self.buf_sc_grads), u=(M_tot, D, sc0 + 3, 1, sg0 + 3, 0, EM_pad), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_gram_out, a["gram_in"]), out_bufs=(gw[p + "gram_out"],), u=(D, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_gram_out, w[p + "gram_out"]), out_bufs=(bd_gram_in,), u=(EM_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_CGT_BCAST_BWD, in_bufs=(a["global_raw"], a["tb_raw"], bd_gram_in, self.buf_ent_meta), out_bufs=(bd_glob, bd_tb), u=(E, M_ent, K, D, EM_pad, EK_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_tb, a["t_mut"]), out_bufs=(gw[p + "team_bcast_proj"],), u=(D, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_tb, w[p + "team_bcast_proj"]), out_bufs=(bd_tmut,), u=(EK_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_MP_LERP_BWD, in_bufs=(a["t_in"], a["t_mut_raw"], bd_tmut, self.buf_scalars), out_bufs=(bd_tin, bd_tmut_raw, self.buf_sc_grads), u=(K_tot, D, sc0 + 2, 1, sg0 + 2, 0, EK_pad), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_tmut_raw, a["t_mixed"]), out_bufs=(gw[p + "team_mutate_proj"],), u=(D, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_tmut_raw, w[p + "team_mutate_proj"]), out_bufs=(bd_tmixed,), u=(EK_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_CGT_RIVAL_BWD, in_bufs=(a["qt_raw"], a["kt_raw"], a["vt_raw"], bd_tmixed), out_bufs=(bd_qt, bd_kt, bd_vt), u=(E, K, D, Rt), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_qt, a["t_in"]), out_bufs=(gw[p + "team_metric"],), u=(Rt, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_kt, a["t_in"]), out_bufs=(gw[p + "rival_metric"],), u=(Rt, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_vt, a["t_in"]), out_bufs=(gw[p + "team_val_proj"],), u=(D, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_qt, w[p + "team_metric"]), out_bufs=(bd_tin,), u=(EK_pad, D, Rt, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_kt, w[p + "rival_metric"]), out_bufs=(bd_tin,), u=(EK_pad, D, Rt, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_vt, w[p + "team_val_proj"]), out_bufs=(bd_tin,), u=(EK_pad, D, D, 1)))
            self._add_op(_make_op(
                MG_OP_CGT_STAGE1_BWD,
                in_bufs=(a["xg"], a["v_norm"], a["r_norm"], bd_tin, bd_glob, self.buf_ent_meta),
                out_bufs=(bd_vnorm, bd_rnorm, bd_xg),
                u=(E | (N << 16), J_eff | (K << 16), D | (Rg << 16), (sc0 + 1) | ((sg0 + 1) << 16), w[p + "pool_score"], self.buf_scalars, gw[p + "pool_score"], self.buf_sc_grads),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(a["v_raw"], bd_vnorm), out_bufs=(bd_vraw,), u=(M_tot, D, 0, EM_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(a["r_raw"], bd_rnorm), out_bufs=(bd_rraw,), u=(M_tot, Rg, 0, EM_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_vraw, a["xg"]), out_bufs=(gw[p + "val_proj"],), u=(D, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_rraw, a["xg"]), out_bufs=(gw[p + "global_metric"],), u=(Rg, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_vraw, w[p + "val_proj"]), out_bufs=(bd_xg,), u=(EM_pad, D, D, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_rraw, w[p + "global_metric"]), out_bufs=(bd_xg,), u=(EM_pad, D, Rg, 1)))

            # Nav-graph mix bwd -> bd_xn
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX_BWD, in_bufs=(a["xn"], a["nav_msg"], bd_xg, -1, self.buf_bot_nid, self.buf_scalars), out_bufs=(bd_xn, bd_nav_msg, self.buf_sc_grads), u=(E, N, Wn, M_ent, D, 2, (sc0 + 0) | ((sg0 + 0) << 16), EM_pad), f=(eps_n, eps_l)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_nav_msg, a["nav_diff"]), out_bufs=(gw[p + "nav_out_proj"],), u=(D, D, EW_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_nav_msg, w[p + "nav_out_proj"]), out_bufs=(bd_nav_diff,), u=(EW_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX_BWD, in_bufs=(bd_nav_diff, -1, -1, self.buf_nav_kern), out_bufs=(bd_v_nav,), u=(E, N, Wn, M_ent, D, 1, 0, EW_pad)))
            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(a["v_nav_raw"], bd_v_nav), out_bufs=(bd_v_nav_raw,), u=(E * Wn, D, 0, EW_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_v_nav_raw, a["x_nav"]), out_bufs=(gw[p + "nav_val_proj"],), u=(D, D, EW_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_v_nav_raw, w[p + "nav_val_proj"]), out_bufs=(bd_x_nav,), u=(EW_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_NAV_GRAPH_MIX_BWD, in_bufs=(bd_x_nav,), out_bufs=(bd_xn,), u=(E, N, Wn, M_ent, D, 0, 0, E * Wn)))

            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(x_states[s], bd_xn), out_bufs=(nxt_dx,), u=(M_tot, D, 1 if self.recurrent else 0, EM_pad), f=(eps_n,)))
            cur_dx, nxt_dx = nxt_dx, cur_dx

        # 4. Encoder backward
        bd_obs_enc = self._add_buf(EN_pad * D * 2)
        bd_nav_enc = self._add_buf(EW_pad * D * 2)
        bd_cart_enc = self._add_buf(EJ_pad * D * 2)
        bd_team_enc = self._add_buf(EK_pad * D * 2)
        bd_h_scr = self._add_buf(E * (Wn + K) * D * 4)
        self._add_op(_make_op(
            MG_OP_GRAPH_EMBED_BWD,
            in_bufs=(b_obs_enc, b_nav_enc, b_cart_enc, b_team_enc, w["type_embed"], cur_dx),
            out_bufs=(bd_obs_enc, bd_nav_enc, bd_cart_enc),
            u=(
                E | (N << 16),
                Wn | (J << 8) | (K << 16) | ((D // 16) << 24),
                0 | (1 << 8) | (5 << 16) | (6 << 24),
                b_h_obs0 | (b_aff_raw << 16),
                self.buf_acc | (self.buf_bot_nid << 16),
                self.buf_scalars | (bd_team_enc << 16),
                gw["type_embed"] | (self.buf_sc_grads << 16),
                bd_h_scr,
            ),
            f=(eps_n, eps_l),
        ))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_obs_enc, b_obs_pre), out_bufs=(gw["enc_obs"],), u=(D, 128, EN_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_nav_enc, b_nav_pre), out_bufs=(gw["enc_nav"],), u=(D, 64, EW_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_cart_enc, b_cart_pre), out_bufs=(gw["enc_cart"],), u=(D, 64, EJ_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_team_enc, b_team_pre), out_bufs=(gw["enc_team"],), u=(D, 64, EK_pad, 1)))

        # Zero-copy host views
        self.obs_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_obs), (E, N, OBS_WIDTH))
        self.nav_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_nav), (E, Wn, 16))
        self.carts_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_carts), (E, J, CART_WIDTH))
        self.teams_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_teams), (E, K, TEAM_WIDTH))
        self.acc_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_acc), (E, N, A_dim))
        self.u_targ_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_u_targ), (E, N, A_dim))
        self.eps_tr_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_eps_tr), (4, E, N, 32))
        self.targ_f_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_targ_f), (E, N, 6))
        self.geo_ncfg_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_geo_ncfg), (16,))
        self.lcfg_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_lcfg), (24,))
        self.taus_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_taus), (4,))
        self.sc_grads_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_sc_grads), (self._n_sc_grads,))
        self.diag_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_diag), (16,))
        self._gw_views = {
            k: _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bid), self._gw_shapes[k])
            for k, bid in self._gw_bufs.items()
        }
        self.geo_ncfg_view[4:16].copy_(torch.tensor([
            self.nc.mp_norm_eps, self.nc.map_radius_min, self.nc.pos_fallback_scale, self.nc.vel_scale,
            self.nc.health_offset, self.nc.health_scale, self.nc.armor_offset, self.nc.armor_scale,
            self.nc.ammo_scale, self.nc.strength_max_s, self.nc.cart_len_scale, self.nc.cart_speed_scale,
        ], dtype=torch.float32))
        self._val_g_cur = float(self.cfg.value_gain_init)
        self._val_b_cur = [0.0, 0.0]
        self._dd_g_cur = float(self.cfg.dyn_drift_gain_init)
        self._da_g_cur = float(self.cfg.dyn_action_gain_init)

    def zero_grads_(self) -> None:
        self.sc_grads_view.zero_()
        self.diag_view.zero_()
        for v in self._gw_views.values():
            v.zero_()

    def set_geo(self, geo: dict[str, torch.Tensor] | None) -> None:
        if geo is not None and "map_center" in geo and "map_radius" in geo:
            self.geo_ncfg_view[0:3].copy_(geo["map_center"].detach().to(device="cpu", dtype=torch.float32).view(3))
            self.geo_ncfg_view[3] = float(geo["map_radius"].detach().to(device="cpu", dtype=torch.float32).item())
        else:
            self.geo_ncfg_view[0:4].zero_()

    def set_nav_kern(self, nav_kern: torch.Tensor) -> None:
        _copy_f32_to_ptr(
            nav_kern.detach().to(device="cpu", dtype=torch.float32).numpy(),
            self.lib.mesh_graph_buffer_ptr(self.g, self.buf_nav_kern),
        )

    @torch.no_grad()
    def sync_weights_(self, model: GraphFlowPolicy) -> None:
        eps = float(self.nc.unit_norm_eps)
        w = self._w_bufs
        def put_bf(key: str, t: torch.Tensor) -> None:
            _copy_bf16_to_ptr(t, self.lib.mesh_graph_buffer_ptr(self.g, w[key]))

        put_bf("enc_obs", _unit_pad_bf16(model.enc_obs.weight, eps, cols_pad=128))
        put_bf("enc_nav", _unit_pad_bf16(model.enc_nav.weight, eps, cols_pad=64))
        put_bf("enc_cart", _unit_pad_bf16(model.enc_cart.weight, eps, cols_pad=64))
        put_bf("enc_team", _unit_pad_bf16(model.enc_team.weight, eps, cols_pad=64))
        te_n = F.rms_norm(model.type_embed.detach().to(device="cpu", dtype=torch.float32), (self.d,), eps=max(self.nc.mp_norm_eps, 1e-4))
        put_bf("type_embed", te_n.to(torch.bfloat16))

        n_uniq_blk = 1 if self.recurrent else self.depth
        scalars = [
            float(torch.sigmoid(model.type_blend.detach().float()).item()),
            float(torch.sigmoid(model.aff_blend.detach().float()).item()),
        ]
        for bi in range(n_uniq_blk):
            blk = model.blocks[bi]
            p = f"b{bi}."
            put_bf(p + "nav_val_proj", _unit_pad_bf16(blk.nav_val_proj.weight, eps))
            put_bf(p + "nav_out_proj", _unit_pad_bf16(blk.nav_out_proj.weight, eps))
            put_bf(p + "val_proj", _unit_pad_bf16(blk.val_proj.weight, eps))
            put_bf(p + "global_metric", _unit_pad_bf16(blk.global_metric.weight, eps))
            put_bf(p + "pool_score", _unit_pad_bf16(blk.pool_score.weight, eps).view(-1))
            put_bf(p + "team_metric", _unit_pad_bf16(blk.team_metric.weight, eps))
            put_bf(p + "rival_metric", _unit_pad_bf16(blk.rival_metric.weight, eps))
            put_bf(p + "team_val_proj", _unit_pad_bf16(blk.team_val_proj.weight, eps))
            put_bf(p + "team_mutate_proj", _unit_pad_bf16(blk.team_mutate_proj.weight, eps))
            put_bf(p + "team_bcast_proj", _unit_pad_bf16(blk.team_bcast_proj.weight, eps))
            put_bf(p + "gram_out", _unit_pad_bf16(blk.gram_out.weight, eps))
            put_bf(p + "dpp_b_proj", _unit_pad_bf16(blk.dpp_b_proj.weight, eps, rows_pad=64))
            put_bf(p + "dpp_v_proj", _unit_pad_bf16(blk.dpp_v_proj.weight, eps))
            put_bf(p + "dpp_out_proj", _unit_pad_bf16(blk.dpp_out_proj.weight, eps))
            wg = _unit_pad_bf16(blk.ffn_gate.weight, eps)
            wu = _unit_pad_bf16(blk.ffn_up.weight, eps)
            put_bf(p + "ffn_gate_up", torch.cat((wg, wu), dim=0))
            put_bf(p + "ffn_down", _unit_pad_bf16(blk.ffn_down.weight, eps))
            scalars.extend([
                float(torch.sigmoid(blk.blend_nav.detach().float()).item()),
                float(torch.sigmoid(blk.blend_pool.detach().float()).item()),
                float(torch.sigmoid(blk.blend_team_mut.detach().float()).item()),
                float(torch.sigmoid(blk.blend_gram.detach().float()).item()),
                float(torch.sigmoid(blk.blend_dpp.detach().float()).item()),
                float(torch.sigmoid(blk.blend_ffn.detach().float()).item()),
            ])
        if self.recurrent:
            scalars.extend(float(v) for v in torch.sigmoid(model.step_blends.detach().float()).cpu().tolist()[: self.depth])
        _copy_f32_to_ptr(np.asarray(scalars, dtype=np.float32), self.lib.mesh_graph_buffer_ptr(self.g, self.buf_scalars))

        put_bf("flow_cond_bot", _unit_pad_bf16(model.flow_cond_bot.weight, eps))
        put_bf("flow_cond_team", _unit_pad_bf16(model.flow_cond_team.weight, eps))
        put_bf("basis_nav_proj", _unit_pad_bf16(model.basis_nav_proj.weight, eps, rows_pad=64))
        put_bf("basis_team_proj", _unit_pad_bf16(model.basis_team_proj.weight, eps, rows_pad=64))
        vd = torch.zeros((64, self.d), dtype=torch.bfloat16)
        vd[0:2] = _unit_pad_bf16(model.value_proj.weight, eps)
        vd[2:3] = _unit_pad_bf16(model.dyn_drift_proj.weight, eps)
        put_bf("val_drift_packed", vd)

        # Pack fj_w1 [128, 128] with col 104 = fj_w1.bias!
        w1p = torch.zeros((128, 128), dtype=torch.bfloat16)
        w1p[:, :104] = model.fj_w1.weight.detach().to(device="cpu", dtype=torch.bfloat16)
        w1p[:, 104] = model.fj_w1.bias.detach().to(device="cpu", dtype=torch.bfloat16)
        put_bf("fj_w1", w1p)
        w2p = torch.zeros((64, 64), dtype=torch.bfloat16)
        w2p[:32, :] = model.fj_w2.weight.detach().to(device="cpu", dtype=torch.bfloat16)
        put_bf("fj_w2", w2p)

        _copy_f32_to_ptr(model.basis_self.detach().float().cpu().numpy(), self.lib.mesh_graph_buffer_ptr(self.g, w["basis_self"]))
        _copy_f32_to_ptr(model.fj_w2.bias.detach().float().cpu().numpy(), self.lib.mesh_graph_buffer_ptr(self.g, w["fj_b2"]))
        daw = F.normalize(model.dyn_action_proj.weight.detach().float().cpu(), p=2, dim=-1, eps=max(eps, 1e-4)).view(-1).numpy()
        _copy_f32_to_ptr(daw, self.lib.mesh_graph_buffer_ptr(self.g, w["dyn_action_proj"]))

        self._val_g_cur = float(model.value_gain.detach().float().clamp(*self.cfg.value_gain_bounds).item())
        self._val_b_cur = model.value_bias.detach().float().cpu().tolist()
        self._dd_g_cur = float(model.dyn_drift_gain.detach().float().clamp(*self.cfg.dyn_gain_bounds).item())
        self._da_g_cur = float(model.dyn_action_gain.detach().float().clamp(*self.cfg.dyn_gain_bounds).item())

        if self._g_repack:
            wt = ctypes.c_double(0.0)
            self.lib.mesh_graph_run_fused(self._g_repack, 1, 0, ctypes.byref(wt))

    @torch.no_grad()
    def microbatch_fwd_bwd(
        self,
        obs: torch.Tensor,
        nav: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        acc: torch.Tensor,
        u_target: torch.Tensor,
        adv: torch.Tensor,
        gae_ret: torch.Tensor,
        obs_drho: torch.Tensor,
        actor_mask: torch.Tensor,
        wm_in: torch.Tensor,
        *,
        value_coef: float = 0.5,
        dyn_coef: float = 0.10,
        grad_scale: float = 1.0,
        group_size: int = 1,
        noise: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
    ) -> dict[str, float]:
        B_cur = int(obs.shape[0])
        assert B_cur <= self.E
        cfg = self.cfg
        self.diag_view.zero_()

        if B_cur == self.E:
            self.obs_view.copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
            self.nav_view.copy_(nav.detach().to(device="cpu", dtype=torch.bfloat16))
            self.carts_view.copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
            self.teams_view.copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
            self.acc_view.copy_(acc.detach().to(device="cpu", dtype=torch.float32))
            u_t_cpu = u_target.detach().to(device="cpu", dtype=torch.float32)
            self.u_targ_view.copy_(u_t_cpu)
            am_cpu = actor_mask.detach().to(device="cpu", dtype=torch.float32)
            wm_cpu = wm_in.detach().to(device="cpu", dtype=torch.float32)
            adv_cpu = adv.detach().to(device="cpu", dtype=torch.float32)
            ret_cpu = gae_ret.detach().to(device="cpu", dtype=torch.float32)
            drho_cpu = obs_drho.detach().to(device="cpu", dtype=torch.float32)
            t0_cpu = (torch.round(obs[:, :, 1].detach().to(device="cpu", dtype=torch.float32)) - 1.0).clamp(0.0, float(max(1, self.K) - 1))
        else:
            self.obs_view.zero_(); self.nav_view.zero_(); self.carts_view.zero_(); self.teams_view.zero_()
            self.acc_view.zero_(); self.u_targ_view.zero_()
            self.obs_view[:B_cur].copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
            self.nav_view[:B_cur].copy_(nav.detach().to(device="cpu", dtype=torch.bfloat16))
            self.carts_view[:B_cur].copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
            self.teams_view[:B_cur].copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
            self.acc_view[:B_cur].copy_(acc.detach().to(device="cpu", dtype=torch.float32))
            u_t_cpu = torch.zeros((self.E, self.N, self.A_dim), dtype=torch.float32)
            u_t_cpu[:B_cur].copy_(u_target.detach().to(device="cpu", dtype=torch.float32))
            self.u_targ_view.copy_(u_t_cpu)
            am_cpu = torch.zeros((self.E, self.N), dtype=torch.float32); am_cpu[:B_cur].copy_(actor_mask.detach().to(device="cpu", dtype=torch.float32))
            wm_cpu = torch.zeros((self.E, self.N), dtype=torch.float32); wm_cpu[:B_cur].copy_(wm_in.detach().to(device="cpu", dtype=torch.float32))
            adv_cpu = torch.zeros((self.E, self.N), dtype=torch.float32); adv_cpu[:B_cur].copy_(adv.detach().to(device="cpu", dtype=torch.float32))
            ret_cpu = torch.zeros((self.E, self.N), dtype=torch.float32); ret_cpu[:B_cur].copy_(gae_ret.detach().to(device="cpu", dtype=torch.float32))
            drho_cpu = torch.zeros((self.E, self.N), dtype=torch.float32); drho_cpu[:B_cur].copy_(obs_drho.detach().to(device="cpu", dtype=torch.float32))
            t0_cpu = torch.zeros((self.E, self.N), dtype=torch.float32)
            t0_cpu[:B_cur].copy_((torch.round(obs[:, :, 1].detach().to(device="cpu", dtype=torch.float32)) - 1.0).clamp(0.0, float(max(1, self.K) - 1)))

        # Solve ESS-targeted Boltzmann weights strictly on valid [:B_cur] rows (preserving grouped solve when B_cur % G == 0!)
        own_idx = (self.W_nav + t0_cpu[:B_cur].to(torch.int64)).unsqueeze(-1)
        u_masked_valid = u_t_cpu[:B_cur].scatter(-1, own_idx, 0.0)
        u_sq_valid = u_masked_valid.square().mean(dim=-1)
        g_eff = group_size if (group_size > 1 and B_cur % group_size == 0) else 1
        w_valid, ess_frac, mean_alpha = compute_ess_boltzmann_weights(
            adv_cpu[:B_cur],
            am_cpu[:B_cur],
            u_sq_mean=u_sq_valid,
            zero_anchor_lambda=cfg.zero_anchor_lambda,
            ess_target=cfg.ess_target,
            w_min=cfg.flow_w_min,
            w_max=cfg.flow_w_max,
            beta_min=cfg.beta_bounds[0],
            beta_max=cfg.beta_bounds[1],
            group_size=g_eff,
        )
        if B_cur == self.E:
            w_norm = w_valid
        else:
            w_norm = torch.zeros((self.E, self.N), dtype=torch.float32)
            w_norm[:B_cur].copy_(w_valid)

        am_sum = max(float(am_cpu[:B_cur].sum().item()), 1.0)
        wm_sum = max(float(wm_cpu[:B_cur].sum().item()), 1.0)
        lm_sum = max(float((1.0 - wm_cpu[:B_cur]).sum().item()), 1.0)

        self.targ_f_view[:, :, 0].copy_(w_norm)
        self.targ_f_view[:, :, 1].copy_(ret_cpu)
        self.targ_f_view[:, :, 2].copy_(drho_cpu)
        self.targ_f_view[:, :, 3].copy_(am_cpu)
        self.targ_f_view[:, :, 4].copy_(wm_cpu)
        self.targ_f_view[:, :, 5].copy_(t0_cpu)

        if noise is not None:
            self.eps_tr_view[:, :B_cur].copy_(noise.detach().to(device="cpu", dtype=torch.float32))
        else:
            self.eps_tr_view.normal_(0.0, float(cfg.sigma_base))

        if taus is None:
            taus_cpu = ((torch.arange(4, dtype=torch.float32) + torch.rand(4, dtype=torch.float32)) * 0.25).clamp(0.02, 0.98)
        else:
            taus_cpu = taus.detach().to(device="cpu", dtype=torch.float32).view(4)
        self.taus_view.copy_(taus_cpu)

        self.lcfg_view[:20].copy_(torch.tensor([
            float(taus_cpu[0]), float(taus_cpu[1]), float(taus_cpu[2]), float(taus_cpu[3]),
            0.0, 0.0, float(value_coef), float(dyn_coef),
            1.0 / am_sum, 1.0 / wm_sum, 1.0 / lm_sum, 1.0 / float(B_cur * self.N),
            self._val_g_cur, self._val_b_cur[0], self._val_b_cur[1],
            self._dd_g_cur, self._da_g_cur, float(grad_scale), float(self.nc.mp_unit_vec_eps),
            float(B_cur),
        ], dtype=torch.float32))

        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))

        d = self.diag_view.tolist()
        loss_vel, loss_val, loss_dyn = d[0], d[1], d[2]
        loss_tot = loss_vel + float(value_coef) * loss_val + float(dyn_coef) * loss_dyn
        return {
            "loss_total": loss_tot,
            "loss_vel": loss_vel,
            "loss_val": loss_val,
            "loss_dyn": loss_dyn,
            "ess_frac": ess_frac,
            "alpha": mean_alpha,
            "u_norm": d[3],
            "v_norm": d[4],
            "wall_ms": float(wall.value),
        }

    @torch.no_grad()
    def export_grads_to_model_(self, model: GraphFlowPolicy) -> None:
        """Apply exact unit-sphere / mp_norm tangent projection and write FP32 grads onto model.parameters()."""
        eps = float(self.nc.unit_norm_eps)
        eps_n = float(self.nc.mp_norm_eps)
        gv = self._gw_views
        sg = self.sc_grads_view

        def set_unit_g(param: torch.nn.Parameter, g_raw: torch.Tensor) -> None:
            param.grad = _tangent_unit_grad(param, g_raw, eps).to(param.dtype)

        set_unit_g(model.enc_obs.weight, gv["enc_obs"][:, :96])
        set_unit_g(model.enc_nav.weight, gv["enc_nav"][:, :20])
        set_unit_g(model.enc_cart.weight, gv["enc_cart"][:, :22])
        set_unit_g(model.enc_team.weight, gv["enc_team"][:, :9])
        model.type_embed.grad = _tangent_mp_norm_grad(model.type_embed, gv["type_embed"], eps_n).to(model.type_embed.dtype)
        model.type_blend.grad = sg[5].clone().to(model.type_blend.dtype)
        model.aff_blend.grad = sg[6].clone().to(model.aff_blend.dtype)

        n_uniq_blk = 1 if self.recurrent else self.depth
        for bi in range(n_uniq_blk):
            blk = model.blocks[bi]
            p = f"b{bi}."
            sg0 = self._sg_blk_base + 6 * bi
            set_unit_g(blk.nav_val_proj.weight, gv[p + "nav_val_proj"])
            set_unit_g(blk.nav_out_proj.weight, gv[p + "nav_out_proj"])
            set_unit_g(blk.val_proj.weight, gv[p + "val_proj"])
            set_unit_g(blk.global_metric.weight, gv[p + "global_metric"])
            set_unit_g(blk.pool_score.weight, gv[p + "pool_score"])
            set_unit_g(blk.team_metric.weight, gv[p + "team_metric"])
            set_unit_g(blk.rival_metric.weight, gv[p + "rival_metric"])
            set_unit_g(blk.team_val_proj.weight, gv[p + "team_val_proj"])
            set_unit_g(blk.team_mutate_proj.weight, gv[p + "team_mutate_proj"])
            set_unit_g(blk.team_bcast_proj.weight, gv[p + "team_bcast_proj"])
            set_unit_g(blk.gram_out.weight, gv[p + "gram_out"])
            set_unit_g(blk.dpp_b_proj.weight, gv[p + "dpp_b_proj"][: self.r_dpp])
            set_unit_g(blk.dpp_v_proj.weight, gv[p + "dpp_v_proj"])
            set_unit_g(blk.dpp_out_proj.weight, gv[p + "dpp_out_proj"])
            set_unit_g(blk.ffn_gate.weight, gv[p + "ffn_gate_up"][: self.h])
            set_unit_g(blk.ffn_up.weight, gv[p + "ffn_gate_up"][self.h :])
            set_unit_g(blk.ffn_down.weight, gv[p + "ffn_down"])
            blk.blend_nav.grad = sg[sg0 + 0].clone().to(blk.blend_nav.dtype)
            blk.blend_pool.grad = sg[sg0 + 1].clone().to(blk.blend_pool.dtype)
            blk.blend_team_mut.grad = sg[sg0 + 2].clone().to(blk.blend_team_mut.dtype)
            blk.blend_gram.grad = sg[sg0 + 3].clone().to(blk.blend_gram.dtype)
            blk.blend_dpp.grad = sg[sg0 + 4].clone().to(blk.blend_dpp.dtype)
            blk.blend_ffn.grad = sg[sg0 + 5].clone().to(blk.blend_ffn.dtype)

        if self.recurrent:
            model.step_blends.grad = sg[self._sg_step_base : self._sg_step_base + self.depth].clone().to(model.step_blends.dtype)

        set_unit_g(model.flow_cond_bot.weight, gv["flow_cond_bot"])
        set_unit_g(model.flow_cond_team.weight, gv["flow_cond_team"])
        set_unit_g(model.basis_nav_proj.weight, gv["basis_nav_proj"][:32])
        set_unit_g(model.basis_team_proj.weight, gv["basis_team_proj"][:32])
        set_unit_g(model.value_proj.weight, gv["val_drift_packed"][0:2])
        set_unit_g(model.dyn_drift_proj.weight, gv["val_drift_packed"][2:3])
        set_unit_g(model.dyn_action_proj.weight, gv["dyn_action_proj"])
        model.basis_self.grad = gv["basis_self"].clone().to(model.basis_self.dtype)

        # Unnormalized Flow MLP weights & biases
        model.fj_w1.weight.grad = gv["fj_w1"][:, :104].clone().to(model.fj_w1.weight.dtype)
        model.fj_w1.bias.grad = gv["fj_w1"][:, 104].clone().to(model.fj_w1.bias.dtype)
        model.fj_w2.weight.grad = gv["fj_w2"][:32, :].clone().to(model.fj_w2.weight.dtype)
        model.fj_w2.bias.grad = gv["fj_b2"].clone().to(model.fj_w2.bias.dtype)

        # Clamped gain gradients (0 outside bounds)
        vg = float(model.value_gain.detach().float().item())
        lo_v, hi_v = self.cfg.value_gain_bounds
        model.value_gain.grad = (sg[0].clone() if (lo_v < vg < hi_v) else torch.zeros_like(sg[0])).to(model.value_gain.dtype)
        model.value_bias.grad = sg[1:3].clone().to(model.value_bias.dtype)
        ddg = float(model.dyn_drift_gain.detach().float().item())
        dag = float(model.dyn_action_gain.detach().float().item())
        lo_d, hi_d = self.cfg.dyn_gain_bounds
        model.dyn_drift_gain.grad = (sg[3].clone() if (lo_d < ddg < hi_d) else torch.zeros_like(sg[3])).to(model.dyn_drift_gain.dtype)
        model.dyn_action_gain.grad = (sg[4].clone() if (lo_d < dag < hi_d) else torch.zeros_like(sg[4])).to(model.dyn_action_gain.dtype)


class GraphFlowOptimizer:
    """Fast FP32 Aurora-polar (for 2D unit-norm matrices) + AdamW (for Mealy-Jet MLP & scalars) step."""

    def __init__(self, model: GraphFlowPolicy, ema_model: GraphFlowPolicy | None, opt_cfg: OptimizerConfig):
        self.model = model
        self.ema_model = ema_model
        self.cfg = opt_cfg
        self.step_count = 0
        self.m_state: dict[str, torch.Tensor] = {}
        self.v_state: dict[str, torch.Tensor] = {}
        self.master: dict[str, torch.Tensor] = {
            n: p.detach().float().clone() for n, p in model.named_parameters()
        }

    @staticmethod
    def _ns_polar(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.float()
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        nrm = X.norm().clamp_min(1e-7)
        X = X / nrm
        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X
        return X.T if transposed else X

    @torch.no_grad()
    def step(self, *, ema_decay: float = 0.995) -> float:
        self.step_count += 1
        t = self.step_count
        lr_2d = float(self.cfg.lr)
        lr_1d = float(self.cfg.scalar_lr)
        mu = float(self.cfg.mu)
        b1, b2 = 0.9, 0.999

        tot_sq = 0.0
        for _, p in self.model.named_parameters():
            if p.grad is not None:
                gf = p.grad.float()
                if torch.isfinite(gf).all():
                    tot_sq += float(gf.square().sum().item())
        gnorm = math.sqrt(tot_sq)
        clip = 1.0 / max(1.0, gnorm / 5.0)

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.float() * clip
            if not torch.isfinite(g).all():
                continue
            wm = self.master[name]
            is_unit_2d = (p.ndim == 2 and min(p.shape) >= 8 and not name.startswith(("fj_w1", "fj_w2", "type_embed", "basis_self")))
            if is_unit_2d:
                m = self.m_state.get(name)
                if m is None:
                    m = g.clone()
                else:
                    m.mul_(mu).add_(g)
                self.m_state[name] = m
                upd = g + mu * m
                # Tangent projection + Newton-Schulz orthogonal polar step on unit sphere
                orth = self._ns_polar(upd, steps=5)
                scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
                wm.add_(orth, alpha=-lr_2d * scale)
                wm.copy_(F.normalize(wm, p=2, dim=-1, eps=1e-4))
            else:
                m = self.m_state.get(name)
                v = self.v_state.get(name)
                if m is None:
                    m = torch.zeros_like(wm)
                    v = torch.zeros_like(wm)
                    self.m_state[name] = m
                    self.v_state[name] = v
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                m_hat = m / (1.0 - b1 ** t)
                v_hat = v / (1.0 - b2 ** t)
                lr_cur = lr_1d * (0.5 if name.startswith("fj_") else 1.0)
                wm.addcdiv_(m_hat, v_hat.sqrt().add_(1e-8), value=-lr_cur)
                if name == "type_embed":
                    wm.copy_(F.rms_norm(wm, (wm.shape[-1],), eps=1e-4))
                elif name == "basis_self":
                    wm.copy_(F.normalize(wm, p=2, dim=-1, eps=1e-4))
                elif name == "value_gain":
                    wm.clamp_(*self.model.cfg.value_gain_bounds)
                elif name in ("dyn_drift_gain", "dyn_action_gain"):
                    wm.clamp_(*self.model.cfg.dyn_gain_bounds)
                elif p.ndim == 2 and not name.startswith(("fj_w1", "fj_w2")):
                    wm.copy_(F.normalize(wm, p=2, dim=-1, eps=1e-4))
            p.copy_(wm.to(p.dtype))

        if self.ema_model is not None:
            for (n, p), (_, pe) in zip(self.model.named_parameters(), self.ema_model.named_parameters()):
                pe.copy_((ema_decay * pe.float() + (1.0 - ema_decay) * self.master[n]).to(pe.dtype))
        return gnorm

    @torch.no_grad()
    def reload_master_from_model_(self) -> None:
        """Sync FP32 master weights after RDMA all-reduce across Studios."""
        for n, p in self.model.named_parameters():
            self.master[n].copy_(p.detach().float())

    @torch.no_grad()
    def export_master_flat_(self, dst: torch.Tensor) -> int:
        """Pack FP32 master parameters into a contiguous 1D buffer for lossless FP32 RDMA all-reduce."""
        off = 0
        for n, p in self.model.named_parameters():
            cnt = p.numel()
            dst[off : off + cnt].copy_(self.master[n].view(-1))
            off += cnt
        return off

    @torch.no_grad()
    def load_master_flat_(self, src: torch.Tensor, *, scale: float = 1.0) -> None:
        """Unpack averaged FP32 master parameters, re-project unit norms in FP32, and sync model."""
        eps = max(float(self.model.norm_cfg.unit_norm_eps), 1e-4)
        off = 0
        for n, p in self.model.named_parameters():
            cnt = p.numel()
            wm = self.master[n]
            wm.copy_(src[off : off + cnt].view(p.shape) * float(scale))
            if n == "type_embed":
                wm.copy_(F.rms_norm(wm, (wm.shape[-1],), eps=1e-4))
            elif n == "value_gain":
                wm.clamp_(*self.model.cfg.value_gain_bounds)
            elif n in ("dyn_drift_gain", "dyn_action_gain"):
                wm.clamp_(*self.model.cfg.dyn_gain_bounds)
            elif p.ndim == 2 and not n.startswith(("fj_w1", "fj_w2")):
                wm.copy_(F.normalize(wm, p=2, dim=-1, eps=eps))
            p.copy_(wm.to(p.dtype))
            off += cnt
