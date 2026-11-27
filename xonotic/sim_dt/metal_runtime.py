from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from payload.tools.strategy_io_schema import CART_WIDTH, OBS_WIDTH, TEAM_WIDTH
from solver.strat.losses import PlanOutput
from solver.strat.plan_policy import PLAN_CONT_DIM, PLAN_WIDTH, PlanWally
from solver.strat.train_config import DEFAULT_ACTION_PACK_CONFIG, ActionPackConfig

from .engine import SimRulesConfig
from .world_graph import WorldGraphIR


# --- libmesh-graph.dylib C ABI bindings (GIL-free on every CDLL call) ---

MG_BUF_WEIGHT = 0
MG_BUF_ARENA = 1
MG_BUF_STATE = 2
MG_BUF_VIEW = 3

MG_DTYPE_BF16 = 0
MG_DTYPE_F16 = 1
MG_DTYPE_F32 = 2
MG_DTYPE_I32 = 3
MG_DTYPE_U32 = 4

MG_WIRE_AUTO = 5

MG_OP_LINEAR = 2
MG_OP_PLAN_PRENORM = 16
MG_OP_PLAN_EMBED_CAT = 17
MG_OP_MP_NORM = 18
MG_OP_MP_LERP = 19
MG_OP_MP_SWIGLU = 20
MG_OP_CGT_GRAM_STAGE1 = 21
MG_OP_CGT_TEAM_RIVAL = 22
MG_OP_CGT_BCAST_MERGE = 23
MG_OP_PLAN_SLICE_NORM = 24
MG_OP_BILINEAR_LOGITS = 25
MG_OP_PLAN_POST_PACK = 26
MG_OP_SIMDT_STEP = 27
MG_OP_SIMDT_OBS = 28


class MeshGraphOp(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("in_buf", ctypes.c_int32 * 6),
        ("out_buf", ctypes.c_int32 * 3),
        ("u", ctypes.c_uint32 * 8),
        ("f", ctypes.c_float * 4),
    ]


_CACHED_LIB: ctypes.CDLL | None = None


def _load_mesh_graph_lib() -> ctypes.CDLL:
    global _CACHED_LIB
    if _CACHED_LIB is not None:
        return _CACHED_LIB
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "rdma" / ".build" / "libmesh-graph.dylib",
        repo_root / "rdma" / "libmesh-graph.dylib",
    ]
    lib_path = next((p for p in candidates if p.exists()), None)
    if lib_path is None:
        raise FileNotFoundError(
            f"libmesh-graph.dylib not found in {candidates}; run `make -C rdma .build/libmesh-graph.dylib`"
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.mesh_graph_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.mesh_graph_create.restype = ctypes.c_void_p
    lib.mesh_graph_destroy.argtypes = [ctypes.c_void_p]
    lib.mesh_graph_destroy.restype = None
    lib.mesh_graph_add_buffer.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    ]
    lib.mesh_graph_add_buffer.restype = ctypes.c_int
    lib.mesh_graph_import_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mesh_graph_import_buffer.restype = ctypes.c_int
    lib.mesh_graph_buffer_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mesh_graph_buffer_ptr.restype = ctypes.c_void_p
    lib.mesh_graph_add_op.argtypes = [ctypes.c_void_p, ctypes.POINTER(MeshGraphOp)]
    lib.mesh_graph_add_op.restype = ctypes.c_int
    lib.mesh_graph_compile.argtypes = [ctypes.c_void_p]
    lib.mesh_graph_compile.restype = ctypes.c_int
    lib.mesh_graph_run.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_double),
    ]
    lib.mesh_graph_run.restype = ctypes.c_double
    lib.mesh_graph_run_fused.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double),
    ]
    lib.mesh_graph_run_fused.restype = ctypes.c_double
    _CACHED_LIB = lib
    return lib


def _make_op(
    kind: int,
    *,
    dtype: int = MG_DTYPE_BF16,
    in_bufs: tuple[int, ...] = (),
    out_bufs: tuple[int, ...] = (),
    u: tuple[int, ...] = (),
    f: tuple[float, ...] = (),
) -> MeshGraphOp:
    op = MeshGraphOp()
    op.kind = int(kind)
    op.dtype = int(dtype)
    for i in range(6):
        op.in_buf[i] = int(in_bufs[i]) if i < len(in_bufs) else -1
    for i in range(3):
        op.out_buf[i] = int(out_bufs[i]) if i < len(out_bufs) else -1
    for i in range(8):
        op.u[i] = int(u[i]) if i < len(u) else 0
    for i in range(4):
        op.f[i] = float(f[i]) if i < len(f) else 0.0
    return op


def _align(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


def _nax_m_pad(m: int) -> int:
    if m < 16:
        return m
    if m == 16:
        return 16
    return _align(m, 32)


def _ptr_to_u8_array(ptr: int, nbytes: int) -> np.ndarray:
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    return np.ctypeslib.as_array(buf)


def _ptr_to_torch_bf16(ptr: int, shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    arr = np.ctypeslib.as_array((ctypes.c_uint16 * numel).from_address(ptr))
    return torch.from_numpy(arr).view(torch.bfloat16).view(*shape)


def _ptr_to_torch_f32(ptr: int, shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    arr = np.ctypeslib.as_array((ctypes.c_float * numel).from_address(ptr))
    return torch.from_numpy(arr).view(*shape)


def _ptr_to_torch_i32(ptr: int, shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    arr = np.ctypeslib.as_array((ctypes.c_int32 * numel).from_address(ptr))
    return torch.from_numpy(arr).view(*shape)


def _copy_bf16_to_ptr(t: torch.Tensor, ptr: int) -> None:
    src = t.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    u16 = src.view(torch.uint16).numpy()
    ctypes.memmove(ptr, u16.ctypes.data, u16.nbytes)


def _copy_f32_to_ptr(arr: np.ndarray, ptr: int) -> None:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    ctypes.memmove(ptr, a.ctypes.data, a.nbytes)


def _unit_weight_bf16(w: torch.Tensor, eps: float = 1e-6, pad_in: int | None = None) -> torch.Tensor:
    wu = F.normalize(w.detach().to(device="cpu", dtype=torch.float32), p=2, dim=-1, eps=max(float(eps), 1e-4))
    if pad_in is not None and pad_in > wu.shape[1]:
        out = torch.zeros((wu.shape[0], pad_in), dtype=torch.bfloat16)
        out[:, : wu.shape[1]] = wu.to(torch.bfloat16)
        return out
    return wu.to(torch.bfloat16).contiguous()


# ============================================================================
# 1. CompiledPlanWally — Stripped BF16 PlanWally lowered to Metal 4 NAX
# ============================================================================

class CompiledPlanWally:
    """
    Zero-overhead, GIL-free Metal 4 compiled execution of stripped BF16 PlanWally.
    All 23 linear projections hit Metal 4 MPP NAX (92.5 TFLOP/s BF16 on M5 Ultra)
    inside ONE MTLCommandBuffer + ONE MTLComputeCommandEncoder.
    """

    def __init__(
        self,
        model: PlanWally,
        E: int,
        N: int,
        J: int,
        K: int,
        *,
        pack_cfg: ActionPackConfig = DEFAULT_ACTION_PACK_CONFIG,
        do_pack: bool = False,
        stochastic: bool = False,
        write_sample: bool = False,
        sample_seed: int = 1,
        graph: int | None = None,
        ext_obs_buf: int | None = None,
        ext_carts_buf: int | None = None,
        ext_teams_buf: int | None = None,
        ext_geo_buf: int | None = None,
        ext_plan_buf: int | None = None,
        ext_active_mask_buf: int | None = None,
    ):
        assert not model.use_fsq and not model.use_mealy and not model.use_manifold_encoders, (
            "CompiledPlanWally targets stripped PlanWally (use_fsq=use_mealy=use_manifold_encoders=False)"
        )
        assert not model.use_cart_dist_bias and not model.use_cell_head and not model.use_aux_skill_head, (
            "CompiledPlanWally targets stripped PlanWally (no cart_dist_bias/cell_head/aux_skill_head)"
        )
        self.lib = _load_mesh_graph_lib()
        self.E = int(E)
        self.N = int(N)
        self.J = int(J)
        self.K = int(K)
        self.M_ent = self.N + self.J + self.K
        self.d = int(model.d)
        self.h = int(model.h)
        self.r_gram = int(model.r_gram)
        self.r_team = int(model.r_dpp)
        self.depth = int(model.depth)
        self.recurrent = bool(model.recurrent)
        self.mc = model.model_cfg
        self.nc = model.norm_cfg
        self.pack_cfg = pack_cfg
        self.do_pack = bool(do_pack)
        self.stochastic = bool(stochastic)
        self.write_sample = bool(write_sample)
        self.sample_seed = int(sample_seed)

        self._owns_graph = graph is None
        if graph is None:
            self.g = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
            if not self.g:
                raise RuntimeError("mesh_graph_create failed")
        else:
            self.g = graph

        self._w_bufs: dict[str, int] = {}
        self._build_graph(
            model,
            ext_obs_buf=ext_obs_buf,
            ext_carts_buf=ext_carts_buf,
            ext_teams_buf=ext_teams_buf,
            ext_geo_buf=ext_geo_buf,
            ext_plan_buf=ext_plan_buf,
            ext_active_mask_buf=ext_active_mask_buf,
        )
        self.sync_weights_(model)
        if self._owns_graph:
            rc = self.lib.mesh_graph_compile(self.g)
            if rc != 0:
                raise RuntimeError(f"mesh_graph_compile failed: {rc}")

    def close(self) -> None:
        if self._owns_graph and getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    def __del__(self) -> None:
        self.close()

    def _add_buf(self, nbytes: int, kind: int = MG_BUF_STATE) -> int:
        return int(self.lib.mesh_graph_add_buffer(self.g, int(max(64, nbytes)), int(kind), None, -1, 0))

    def _add_view(self, parent: int, off: int, nbytes: int) -> int:
        return int(self.lib.mesh_graph_add_buffer(self.g, int(nbytes), MG_BUF_VIEW, None, int(parent), int(off)))

    def _add_op(self, op: MeshGraphOp) -> None:
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))

    def _build_graph(
        self,
        model: PlanWally,
        *,
        ext_obs_buf: int | None,
        ext_carts_buf: int | None,
        ext_teams_buf: int | None,
        ext_geo_buf: int | None,
        ext_plan_buf: int | None,
        ext_active_mask_buf: int | None,
    ) -> None:
        E, N, J, K, M_ent = self.E, self.N, self.J, self.K, self.M_ent
        D, H, Rg, Rt = self.d, self.h, self.r_gram, self.r_team
        EN_pad = _nax_m_pad(E * N)
        EJ_pad = _nax_m_pad(E * J)
        EK_pad = _nax_m_pad(E * K)
        EM_pad = _nax_m_pad(E * M_ent)

        # Persistent IO buffers
        self.buf_obs = ext_obs_buf if ext_obs_buf is not None else self._add_buf(E * N * OBS_WIDTH * 2, MG_BUF_STATE)
        self.buf_carts = ext_carts_buf if ext_carts_buf is not None else self._add_buf(E * J * CART_WIDTH * 2, MG_BUF_STATE)
        self.buf_teams = ext_teams_buf if ext_teams_buf is not None else self._add_buf(E * K * TEAM_WIDTH * 2, MG_BUF_STATE)
        self.buf_geo = ext_geo_buf if ext_geo_buf is not None else self._add_buf(4 * 4, MG_BUF_STATE)
        self.buf_ncfg = self._add_buf(16 * 4, MG_BUF_WEIGHT)
        self.buf_ent_meta = self._add_buf(E * (M_ent + K) * 4, MG_BUF_STATE)

        # Scalar bank: [0]=type_blend, [1]=temp_cart, [2]=temp_rival,
        #   then per unique block (4 floats each: blend_pool, blend_team_mut, blend_gram, blend_ffn),
        #   then depth step_blends (if recurrent).
        n_uniq_blk = 1 if self.recurrent else self.depth
        self._sc_blk_base = 3
        self._sc_step_base = 3 + 4 * n_uniq_blk
        n_scalars = self._sc_step_base + (self.depth if self.recurrent else 0)
        self.buf_scalars = self._add_buf(max(16, n_scalars) * 4, MG_BUF_STATE)
        self.buf_head_params = self._add_buf(48 * 4, MG_BUF_STATE)

        # Output buffers
        self.buf_cart_logits = self._add_buf(E * N * J * 2, MG_BUF_STATE)
        self.buf_rival_logits = self._add_buf(E * N * K * 2, MG_BUF_STATE)
        self.buf_head_out = self._add_buf(E * N * 25 * 2, MG_BUF_STATE)
        self.buf_plan = ext_plan_buf if ext_plan_buf is not None else (
            self._add_buf(E * N * PLAN_WIDTH * 4, MG_BUF_STATE) if self.do_pack else -1
        )
        self.buf_sample = self._add_buf(E * N * 14 * 4, MG_BUF_STATE) if self.write_sample else -1
        self.buf_active_mask = ext_active_mask_buf if ext_active_mask_buf is not None else -1

        # Static BF16 unit-normalized weights (all NAX-aligned: K%64==0, N%32==0!)
        w = self._w_bufs
        w["enc_obs"] = self._add_buf(D * 128 * 2, MG_BUF_WEIGHT)
        w["enc_cart"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["enc_team"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["type_embed"] = self._add_buf(3 * D * 2, MG_BUF_WEIGHT)

        for bi in range(n_uniq_blk):
            p = f"b{bi}."
            w[p + "val_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "global_metric"] = self._add_buf(Rg * D * 2, MG_BUF_WEIGHT)
            w[p + "pool_score"] = self._add_buf(D * 2, MG_BUF_WEIGHT)
            w[p + "team_metric"] = self._add_buf(Rt * D * 2, MG_BUF_WEIGHT)
            w[p + "rival_metric"] = self._add_buf(Rt * D * 2, MG_BUF_WEIGHT)
            w[p + "team_val_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "team_mutate_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "team_bcast_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "gram_out"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "ffn_gate_up"] = self._add_buf(2 * H * D * 2, MG_BUF_WEIGHT)
            w[p + "ffn_down"] = self._add_buf(D * H * 2, MG_BUF_WEIGHT)

        w["cart_q"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["cart_k"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["rival_q"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["rival_k"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["heads_packed"] = self._add_buf(32 * D * 2, MG_BUF_WEIGHT)

        # Static shared activation arena (zero dynamic allocation per step!)
        # Layout sized with generous non-aliasing slots for peak live tensors.
        off = 0
        def alloc_slot(nbytes: int) -> tuple[int, int]:
            nonlocal off
            s_off = _align(off, 256)
            off = s_off + _align(max(64, nbytes), 256)
            return s_off, nbytes

        s_x = alloc_slot(EM_pad * D * 2)
        s_xn = alloc_slot(EM_pad * D * 2)
        s_a = alloc_slot(max(EN_pad * 128, EM_pad * D, EM_pad * 2 * H, EK_pad * D, EN_pad * D) * 2)
        s_b = alloc_slot(max(EJ_pad * 64, EM_pad * D, EM_pad * H, EJ_pad * D) * 2)
        s_c = alloc_slot(max(EK_pad * 64, EM_pad * Rg, EM_pad * D, EN_pad * D) * 2)
        s_d = alloc_slot(max(EM_pad * D, EN_pad * D, EJ_pad * D) * 2)
        s_e = alloc_slot(max(EK_pad * D, EJ_pad * D, EN_pad * D) * 2)
        s_f = alloc_slot(max(EK_pad * D, EK_pad * Rt, EN_pad * D) * 2)
        s_g = alloc_slot(max(EK_pad * D, EK_pad * Rt, EJ_pad * D) * 2)
        s_h = alloc_slot(max(EK_pad * D, EN_pad * 32) * 2)

        self.buf_arena = self._add_buf(off, MG_BUF_ARENA)
        b_x = self._add_view(self.buf_arena, *s_x)
        b_xn = self._add_view(self.buf_arena, *s_xn)
        b_a = self._add_view(self.buf_arena, *s_a)
        b_b = self._add_view(self.buf_arena, *s_b)
        b_c = self._add_view(self.buf_arena, *s_c)
        b_d = self._add_view(self.buf_arena, *s_d)
        b_e = self._add_view(self.buf_arena, *s_e)
        b_f = self._add_view(self.buf_arena, *s_f)
        b_g = self._add_view(self.buf_arena, *s_g)
        b_h = self._add_view(self.buf_arena, *s_h)

        eps_n = float(self.nc.mp_norm_eps)
        eps_l = float(self.nc.mp_lerp_eps)
        eps_u = float(self.nc.mp_unit_vec_eps)

        # Write static NormConfig
        ncfg_arr = np.array([
            self.nc.mp_norm_eps, self.nc.map_radius_min, self.nc.pos_fallback_scale, self.nc.vel_scale,
            self.nc.health_offset, self.nc.health_scale, self.nc.armor_offset, self.nc.armor_scale,
            self.nc.ammo_scale, self.nc.strength_max_s, self.nc.cart_len_scale, self.nc.cart_speed_scale,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        _copy_f32_to_ptr(ncfg_arr, self.lib.mesh_graph_buffer_ptr(self.g, self.buf_ncfg))

        # --- Lower stripped BF16 PlanWally IR ---
        # 1. Prenormalize + CGT features -> b_a (obs_pre), b_b (cart_pre), b_c (team_pre)
        self._add_op(_make_op(
            MG_OP_PLAN_PRENORM,
            in_bufs=(self.buf_obs, self.buf_carts, self.buf_teams, self.buf_geo, self.buf_ncfg, self.buf_ent_meta),
            out_bufs=(b_a, b_b, b_c),
            u=(E, N, J, K, 128, 64, 64, 0),
        ))

        # 2. 3 NAX Encoder GEMMs -> b_d (obs_enc), b_e (cart_enc), b_f (team_enc)
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_a, w["enc_obs"]), out_bufs=(b_d,), u=(EN_pad, D, 128)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_b, w["enc_cart"]), out_bufs=(b_e,), u=(EJ_pad, D, 64)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_c, w["enc_team"]), out_bufs=(b_f,), u=(EK_pad, D, 64)))

        # 3. Fused norm + type_embed lerp + concat + norm -> b_x [E*M_ent, D]
        self._add_op(_make_op(
            MG_OP_PLAN_EMBED_CAT,
            in_bufs=(b_d, b_e, b_f, w["type_embed"], self.buf_scalars),
            out_bufs=(b_x,),
            u=(E, N, J, K, D, 0),
            f=(eps_n, eps_l),
        ))

        # 4. Depth recurrent/distinct MPResCGTBlock passes
        M_tot = E * M_ent
        K_tot = E * K
        for step in range(self.depth):
            bi = 0 if self.recurrent else step
            p = f"b{bi}."
            sc0 = self._sc_blk_base + 4 * bi

            # xn = mp_norm(x)
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(b_x,), out_bufs=(b_xn,), u=(M_tot, D), f=(eps_n,)))

            # v_raw -> b_b, r_raw -> b_c
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_xn, w[p + "val_proj"]), out_bufs=(b_b,), u=(EM_pad, D, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_xn, w[p + "global_metric"]), out_bufs=(b_c,), u=(EM_pad, Rg, D)))

            # Stage 1: norm(v,r) + global_gram -> b_d, team_pool + blend_pool -> b_e (t_in)
            self._add_op(_make_op(
                MG_OP_CGT_GRAM_STAGE1,
                in_bufs=(b_xn, b_b, b_c, w[p + "pool_score"], self.buf_ent_meta, self.buf_scalars),
                out_bufs=(b_d, b_e),
                u=(E, N, J, K, D, Rg, sc0 + 0),
                f=(eps_n, eps_l),
            ))

            # Team rival projections: q_t -> b_f, k_t -> b_g, v_t -> b_h
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_e, w[p + "team_metric"]), out_bufs=(b_f,), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_e, w[p + "rival_metric"]), out_bufs=(b_g,), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_e, w[p + "team_val_proj"]), out_bufs=(b_h,), u=(EK_pad, D, D)))

            # Rival Gram mix -> b_a (t_mixed [E*K, D])
            self._add_op(_make_op(
                MG_OP_CGT_TEAM_RIVAL,
                in_bufs=(b_f, b_g, b_h),
                out_bufs=(b_a,),
                u=(E, K, D, Rt),
                f=(eps_n,),
            ))

            # team_mutate_proj -> b_f, lerp(t_in=b_e, norm(b_f)) -> b_g (t_mut), team_bcast_proj -> b_h
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_a, w[p + "team_mutate_proj"]), out_bufs=(b_f,), u=(EK_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_MP_LERP,
                in_bufs=(b_e, b_f, self.buf_scalars),
                out_bufs=(b_g,),
                u=(K_tot, D, sc0 + 1, 1),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_g, w[p + "team_bcast_proj"]), out_bufs=(b_h,), u=(EK_pad, D, D)))

            # Broadcast + merge(norm(global_raw=b_d) + team_vals) -> b_a (gram_in)
            self._add_op(_make_op(
                MG_OP_CGT_BCAST_MERGE,
                in_bufs=(b_d, b_h, self.buf_ent_meta),
                out_bufs=(b_a,),
                u=(E, M_ent, K, D),
                f=(eps_n,),
            ))

            # gram_out -> b_b, x_mid = mp_lerp(xn, norm(b_b), blend_gram) -> b_d
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_a, w[p + "gram_out"]), out_bufs=(b_b,), u=(EM_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_MP_LERP,
                in_bufs=(b_xn, b_b, self.buf_scalars),
                out_bufs=(b_d,),
                u=(M_tot, D, sc0 + 2, 1),
                f=(eps_n, eps_l),
            ))

            # FFN SwiGLU: packed gate_up -> b_a, mp_swiglu -> b_b, ffn_down -> b_c
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_d, w[p + "ffn_gate_up"]), out_bufs=(b_a,), u=(EM_pad, 2 * H, D)))
            self._add_op(_make_op(
                MG_OP_MP_SWIGLU,
                in_bufs=(b_a,),
                out_bufs=(b_b,),
                u=(M_tot, H),
                f=(1.0 / float(self.nc.mp_silu_scale),),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_b, w[p + "ffn_down"]), out_bufs=(b_c,), u=(EM_pad, D, H)))

            # x_new = mp_lerp(x_mid=b_d, norm(b_c), blend_ffn)
            if self.recurrent:
                self._add_op(_make_op(
                    MG_OP_MP_LERP,
                    in_bufs=(b_d, b_c, self.buf_scalars),
                    out_bufs=(b_a,),
                    u=(M_tot, D, sc0 + 3, 1),
                    f=(eps_n, eps_l),
                ))
                self._add_op(_make_op(
                    MG_OP_MP_LERP,
                    in_bufs=(b_x, b_a, self.buf_scalars),
                    out_bufs=(b_x,),
                    u=(M_tot, D, self._sc_step_base + step, 0),
                    f=(eps_n, eps_l),
                ))
            else:
                self._add_op(_make_op(
                    MG_OP_MP_LERP,
                    in_bufs=(b_d, b_c, self.buf_scalars),
                    out_bufs=(b_x,),
                    u=(M_tot, D, sc0 + 3, 1),
                    f=(eps_n, eps_l),
                ))

        # 5. Final norm & slice: x -> p_h (b_c), c_h (b_d), t_h (b_e)
        self._add_op(_make_op(
            MG_OP_PLAN_SLICE_NORM,
            in_bufs=(b_x,),
            out_bufs=(b_c, b_d, b_e),
            u=(E, N, J, K, D),
            f=(eps_n,),
        ))

        # 6. 5 Head NAX GEMMs:
        #    cart_q -> b_a, cart_k -> b_b, rival_q -> b_f, rival_k -> b_g, heads_raw[32] -> b_h
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_c, w["cart_q"]), out_bufs=(b_a,), u=(EN_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_d, w["cart_k"]), out_bufs=(b_b,), u=(EJ_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_c, w["rival_q"]), out_bufs=(b_f,), u=(EN_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_e, w["rival_k"]), out_bufs=(b_g,), u=(EK_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_c, w["heads_packed"]), out_bufs=(b_h,), u=(EN_pad, 32, D)))

        # 7. Bilinear cart & rival logits
        self._add_op(_make_op(
            MG_OP_BILINEAR_LOGITS,
            in_bufs=(b_a, b_b, self.buf_ent_meta, self.buf_scalars),
            out_bufs=(self.buf_cart_logits,),
            u=(E, N, J, D, 1, 0, M_ent),
            f=(eps_u, float(self.mc.rival_own_mask_fill)),
        ))
        self._add_op(_make_op(
            MG_OP_BILINEAR_LOGITS,
            in_bufs=(b_f, b_g, self.buf_ent_meta, self.buf_scalars),
            out_bufs=(self.buf_rival_logits,),
            u=(E, N, K, D, 2, 1 if K >= 2 else 0, M_ent),
            f=(eps_u, float(self.mc.rival_own_mask_fill)),
        ))

        # 8. Post-process heads + optional in-GPU sample & pack_plan
        flags = (1 if self.do_pack else 0) | (2 if self.stochastic else 0) | (4 if self.buf_active_mask >= 0 else 0) | (8 if self.write_sample else 0)
        self._add_op(_make_op(
            MG_OP_PLAN_POST_PACK,
            in_bufs=(b_h, self.buf_cart_logits, self.buf_rival_logits, self.buf_head_params, self.buf_ent_meta, self.buf_active_mask),
            out_bufs=(self.buf_head_out, self.buf_plan, self.buf_sample),
            u=(E, N, J, K, M_ent, flags, 0, self.sample_seed),
        ))

        # Zero-copy host views
        self.obs_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_obs), (E, N, OBS_WIDTH))
        self.carts_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_carts), (E, J, CART_WIDTH))
        self.teams_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_teams), (E, K, TEAM_WIDTH))
        self.geo_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_geo), (4,))
        self.cart_logits_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_cart_logits), (E, N, J))
        self.rival_logits_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_rival_logits), (E, N, K))
        self.head_out_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_head_out), (E, N, 25))
        self.plan_view = (
            _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_plan), (E, N, PLAN_WIDTH))
            if self.buf_plan >= 0
            else None
        )
        self.sample_view = (
            _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_sample), (E, N, 14))
            if self.buf_sample >= 0
            else None
        )
        self._cont_logstd_cached = torch.zeros((PLAN_CONT_DIM,), dtype=torch.bfloat16)

    @torch.no_grad()
    def sync_weights_(self, model: PlanWally) -> None:
        """Zero-recompilation weight upload (~0.08 ms for 2.96 MB BF16 on unified memory)."""
        eps = float(self.nc.unit_norm_eps)
        mc = model.model_cfg
        nc = model.norm_cfg
        w = self._w_bufs

        def put(key: str, tensor: torch.Tensor) -> None:
            _copy_bf16_to_ptr(tensor, self.lib.mesh_graph_buffer_ptr(self.g, w[key]))

        put("enc_obs", _unit_weight_bf16(model.enc_obs.weight, eps, pad_in=128))
        put("enc_cart", _unit_weight_bf16(model.enc_cart.weight, eps, pad_in=64))
        put("enc_team", _unit_weight_bf16(model.enc_team.weight, eps, pad_in=64))
        te_n = F.rms_norm(model.type_embed.detach().to(device="cpu", dtype=torch.bfloat16), (self.d,), eps=max(nc.mp_norm_eps, 1e-4))
        put("type_embed", te_n[:3])

        n_uniq_blk = 1 if self.recurrent else self.depth
        scalars = [
            float(torch.sigmoid(model.type_blend.detach().float()).item()),
            float(torch.clamp(torch.exp(model.log_temp_cart.detach().float()), mc.temp_bounds[0], mc.temp_bounds[1]).item()),
            float(torch.clamp(torch.exp(model.log_temp_rival.detach().float()), mc.temp_bounds[0], mc.temp_bounds[1]).item()),
        ]
        for bi in range(n_uniq_blk):
            blk = model.blocks[bi]
            p = f"b{bi}."
            put(p + "val_proj", _unit_weight_bf16(blk.val_proj.weight, eps))
            put(p + "global_metric", _unit_weight_bf16(blk.global_metric.weight, eps))
            put(p + "pool_score", _unit_weight_bf16(blk.pool_score.weight, eps).view(-1))
            put(p + "team_metric", _unit_weight_bf16(blk.team_metric.weight, eps))
            put(p + "rival_metric", _unit_weight_bf16(blk.rival_metric.weight, eps))
            put(p + "team_val_proj", _unit_weight_bf16(blk.team_val_proj.weight, eps))
            put(p + "team_mutate_proj", _unit_weight_bf16(blk.team_mutate_proj.weight, eps))
            put(p + "team_bcast_proj", _unit_weight_bf16(blk.team_bcast_proj.weight, eps))
            put(p + "gram_out", _unit_weight_bf16(blk.gram_out.weight, eps))
            wg = _unit_weight_bf16(blk.ffn_gate.weight, eps)
            wu = _unit_weight_bf16(blk.ffn_up.weight, eps)
            put(p + "ffn_gate_up", torch.cat((wg, wu), dim=0))
            put(p + "ffn_down", _unit_weight_bf16(blk.ffn_down.weight, eps))
            scalars.extend([
                float(torch.sigmoid(blk.blend_pool.detach().float()).item()),
                float(torch.sigmoid(blk.blend_team_mut.detach().float()).item()),
                float(torch.sigmoid(blk.blend_gram.detach().float()).item()),
                float(torch.sigmoid(blk.blend_ffn.detach().float()).item()),
            ])
        if self.recurrent:
            sb = torch.sigmoid(model.step_blends.detach().float()).cpu().tolist()
            scalars.extend(float(v) for v in sb[: self.depth])
        _copy_f32_to_ptr(np.asarray(scalars, dtype=np.float32), self.lib.mesh_graph_buffer_ptr(self.g, self.buf_scalars))

        put("cart_q", _unit_weight_bf16(model.cart_q.weight, eps))
        put("cart_k", _unit_weight_bf16(model.cart_k.weight, eps))
        put("rival_q", _unit_weight_bf16(model.rival_q.weight, eps))
        put("rival_k", _unit_weight_bf16(model.rival_k.weight, eps))

        hp_w = torch.zeros((32, self.d), dtype=torch.bfloat16)
        hp_w[0:11] = _unit_weight_bf16(model.cont_proj.weight, eps)
        hp_w[11:13] = _unit_weight_bf16(model.value_proj.weight, eps)
        hp_w[13:14] = _unit_weight_bf16(model.dyn_drift_proj.weight, eps)
        hp_w[14:25] = _unit_weight_bf16(model.dyn_action_proj.weight, eps)
        put("heads_packed", hp_w)

        hp = np.zeros((48,), dtype=np.float32)
        hp[0:11] = torch.clamp(model.cont_gain.detach().float(), -mc.cont_gain_bound, mc.cont_gain_bound).cpu().numpy()
        hp[11:22] = model.cont_bias.detach().float().cpu().numpy()
        ls = torch.clamp(model.cont_logstd.detach().float(), mc.cont_logstd_bounds[0], mc.cont_logstd_bounds[1]).cpu()
        self._cont_logstd_cached = ls.to(torch.bfloat16)
        hp[22:33] = ls.numpy()
        hp[33] = float(torch.clamp(model.value_gain.detach().float(), mc.value_gain_bounds[0], mc.value_gain_bounds[1]).item())
        hp[34:36] = model.value_bias.detach().float().cpu().numpy()
        hp[36] = float(torch.clamp(model.dyn_drift_gain.detach().float(), mc.dyn_gain_bounds[0], mc.dyn_gain_bounds[1]).item())
        hp[37] = float(torch.clamp(model.dyn_action_gain.detach().float(), mc.dyn_gain_bounds[0], mc.dyn_gain_bounds[1]).item())
        hp[38] = float(mc.cont_mean_scale)
        hp[39] = float(self.pack_cfg.cart_ds_scale)
        hp[40] = float(self.pack_cfg.cart_ds_power)
        hp[41] = float(self.pack_cfg.cart_ds_clip)
        _copy_f32_to_ptr(hp, self.lib.mesh_graph_buffer_ptr(self.g, self.buf_head_params))

    def set_geo(self, geo: dict[str, torch.Tensor] | None) -> None:
        if geo is not None and "map_center" in geo and "map_radius" in geo:
            c = geo["map_center"].detach().to(device="cpu", dtype=torch.float32).view(3)
            r = float(geo["map_radius"].detach().to(device="cpu", dtype=torch.float32).item())
            self.geo_view[0:3].copy_(c)
            self.geo_view[3] = r
        else:
            self.geo_view.zero_()

    def run_inplace(self, n_steps: int = 1, warm_steps: int = 0, fused_cb: bool = False) -> float:
        wall = ctypes.c_double(0.0)
        if fused_cb:
            return float(self.lib.mesh_graph_run_fused(self.g, int(n_steps), int(warm_steps), ctypes.byref(wall)))
        return float(self.lib.mesh_graph_run(self.g, int(n_steps), int(warm_steps), None, 0, ctypes.byref(wall)))

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        geo: dict[str, torch.Tensor] | None = None,
    ) -> PlanOutput:
        self.obs_view.copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
        self.carts_view.copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
        self.teams_view.copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
        self.set_geo(geo)
        self.run_inplace(1, 0)
        ho = self.head_out_view
        cont_logstd = self._cont_logstd_cached.view(1, 1, -1).expand(self.E, self.N, -1)
        return PlanOutput(
            cont_mean=ho[:, :, 0:11].clone(),
            cont_logstd=cont_logstd,
            cart_logits=self.cart_logits_view.clone(),
            rival_logits=self.rival_logits_view.clone(),
            cell_logits=None,
            value_winnie=ho[:, :, 11].clone(),
            value_lou=ho[:, :, 12].clone(),
            dyn_drift=ho[:, :, 13].clone(),
            dyn_action=ho[:, :, 14:25].clone(),
            h_next=None,
            fsq_codes=None,
            pred_skill=None,
        )


# ============================================================================
# 2. MetalDeltaTimePayloadEngine — Fused TG-SRAM GIL-free Metal 4 sim_dt
# ============================================================================

@dataclass
class MetalSimSnapshot:
    """Single contiguous env-major byte snapshot for O(1) single-memcpy restore/fork_group."""
    E: int
    N: int
    J: int
    K: int
    env_stride: int
    soff_meta: int
    data: bytes


def _build_world_blob(
    world: WorldGraphIR,
    E: int,
    N: int,
    rules: SimRulesConfig,
) -> tuple[bytes, dict[str, int]]:
    K = int(world.teams)
    J = int(world.carts)
    CP = int(world.max_checkpoints)
    V = int(world.num_waypoints)
    S_max = int(world.cart_nodes.shape[1])
    M_s = int(world.cart_sample_wp.shape[1])
    S_sp = int(world.spawn_pos.shape[1])
    push_cap = max(1, int(rules.push_cap))
    point_limit = float(rules.point_limit_base * max(1.0, J / float(max(1, K))))

    # Compute per-env contiguous state offsets (all 16B aligned, env_stride 64B aligned)
    so = 0
    def s_alloc(nbytes: int) -> int:
        nonlocal so
        cur = _align(so, 16)
        so = cur + _align(nbytes, 16)
        return cur

    soff_meta = s_alloc(8 * 4)
    soff_p_pos = s_alloc(N * 3 * 4)
    soff_p_vel = s_alloc(N * 3 * 4)
    soff_p_f1 = s_alloc(15 * N * 4)
    soff_p_i1 = s_alloc(8 * N * 4)
    soff_p_plan = s_alloc(N * 16 * 4)
    soff_c_f1 = s_alloc(3 * J * 4)
    soff_c_pos = s_alloc(J * 3 * 4)
    soff_c_i1 = s_alloc(5 * J * 4)
    soff_c_cp_owner = s_alloc(J * CP * 4)
    soff_t_score = s_alloc(K * 4)
    soff_t_cp_held = s_alloc(K * 4)
    env_stride = _align(so, 64)

    # Push weight LUT
    lut = [0.0]
    acc, term = 0.0, 1.0
    for _ in range(push_cap):
        acc += term
        term *= float(rules.push_falloff)
        lut.append(acc)

    tables: list[tuple[str, bytes]] = [
        ("wp_pos", np.ascontiguousarray(world.wp_pos.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("wp_dist", np.ascontiguousarray(world.wp_dist.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("wp_next_hop", np.ascontiguousarray(world.wp_next_hop.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("cart_nodes", np.ascontiguousarray(world.cart_nodes.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_node_s", np.ascontiguousarray(world.cart_node_s.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_seg_count", np.ascontiguousarray(world.cart_seg_count.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("cart_length", np.ascontiguousarray(world.cart_length.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_cp_dists", np.ascontiguousarray(world.cart_cp_dists.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_cp_count", np.ascontiguousarray(world.cart_cp_count.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("cart_radius", np.ascontiguousarray(world.cart_radius.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_height", np.ascontiguousarray(world.cart_height.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_speed_scale", np.ascontiguousarray(world.cart_speed_scale.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("cart_sample_wp", np.ascontiguousarray(world.cart_sample_wp.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("spawn_pos", np.ascontiguousarray(world.spawn_pos.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("spawn_wp", np.ascontiguousarray(world.spawn_wp.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("push_lut", np.ascontiguousarray(np.asarray(lut, dtype=np.float32)).tobytes()),
    ]

    hdr_fmt = "<39I18f"
    hdr_size = struct.calcsize(hdr_fmt)
    cur = _align(hdr_size, 64)
    offs: dict[str, int] = {}
    for name, raw in tables:
        cur = _align(cur, 16)
        offs[name] = cur
        cur += len(raw)
    total_bytes = _align(cur, 64)

    header = struct.pack(
        hdr_fmt,
        E, N, J, K, CP, V, S_max, M_s, S_sp, push_cap,
        offs["wp_pos"], offs["wp_dist"], offs["wp_next_hop"],
        offs["cart_nodes"], offs["cart_node_s"], offs["cart_seg_count"], offs["cart_length"],
        offs["cart_cp_dists"], offs["cart_cp_count"], offs["cart_radius"], offs["cart_height"],
        offs["cart_speed_scale"], offs["cart_sample_wp"], offs["spawn_pos"], offs["spawn_wp"],
        offs["push_lut"],
        env_stride,
        soff_meta, soff_p_pos, soff_p_vel, soff_p_f1, soff_p_i1, soff_p_plan,
        soff_c_f1, soff_c_pos, soff_c_i1, soff_c_cp_owner, soff_t_score, soff_t_cp_held,
        point_limit,
        float(rules.checkpoint_rate),
        float(rules.cart_base_speed),
        float(rules.cart_max_speed),
        float(rules.contest_speed),
        float(rules.reverse_speed),
        float(rules.move_speed),
        float(rules.replan_interval),
        float(rules.plan_replan_hysteresis),
        float(rules.respawn_delay),
        float(rules.max_health),
        float(rules.start_armor),
        float(rules.armor_absorb),
        float(rules.combat_range),
        float(rules.base_dps),
        float(rules.skill_dps_scale),
        float(rules.dps_stochastic_cv),
        float(rules.plc_tick_ref),
    )

    blob = bytearray(total_bytes)
    blob[0:hdr_size] = header
    for name, raw in tables:
        o = offs[name]
        blob[o : o + len(raw)] = raw

    layout = dict(
        env_stride=env_stride,
        soff_meta=soff_meta,
        soff_p_pos=soff_p_pos,
        soff_p_vel=soff_p_vel,
        soff_p_f1=soff_p_f1,
        soff_p_i1=soff_p_i1,
        soff_p_plan=soff_p_plan,
        soff_c_f1=soff_c_f1,
        soff_c_pos=soff_c_pos,
        soff_c_i1=soff_c_i1,
        soff_c_cp_owner=soff_c_cp_owner,
        soff_t_score=soff_t_score,
        soff_t_cp_held=soff_t_cp_held,
        point_limit=point_limit,
    )
    return bytes(blob), layout


class MetalDeltaTimePayloadEngine:
    """
    GIL-free Metal 4 TG-SRAM DeltaTimePayloadEngine inside libmesh-graph.dylib.
    Zero [E,N,N] or [E,N,J] DRAM allocation; O(1) single-memcpy snapshot/restore/fork_group;
    direct BF16 zero-copy observation sharing with CompiledPlanWally.
    """

    def __init__(
        self,
        world: WorldGraphIR,
        num_envs: int,
        num_players: int | None = None,
        *,
        rules: SimRulesConfig = SimRulesConfig(),
        base_seed: int = 1,
    ):
        self.lib = _load_mesh_graph_lib()
        self.world = world.to("cpu") if world.device.type != "cpu" else world
        self.rules = rules
        self.E = int(num_envs)
        self.K = int(self.world.teams)
        self.J = int(self.world.carts)
        self.N = int(num_players if num_players is not None else self.world.players_default)
        self.CP = int(self.world.max_checkpoints)
        assert self.N <= 128 and self.J <= 32 and self.K <= 16 and self.CP <= 16, (
            f"MetalDeltaTimePayloadEngine TG SRAM bounds: N<=128, J<=32, K<=16, CP<=16 (got {self.N},{self.J},{self.K},{self.CP})"
        )

        world_blob, self.layout = _build_world_blob(self.world, self.E, self.N, self.rules)
        self.point_limit = float(self.layout["point_limit"])
        self.env_stride = int(self.layout["env_stride"])

        # Step graph & obs graph share the exact same underlying MTLBuffers!
        self.g_step = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        self.g_obs = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        if not self.g_step or not self.g_obs:
            raise RuntimeError("mesh_graph_create failed for MetalDeltaTimePayloadEngine")

        wb_arr = np.frombuffer(world_blob, dtype=np.uint8)
        self.buf_world = int(self.lib.mesh_graph_add_buffer(
            self.g_step, len(world_blob), MG_BUF_WEIGHT, wb_arr.ctypes.data, -1, 0
        ))
        self.buf_state = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.env_stride, MG_BUF_STATE, None, -1, 0
        ))
        self.buf_plan = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.N * PLAN_WIDTH * 4, MG_BUF_STATE, None, -1, 0
        ))
        self.buf_cfg = int(self.lib.mesh_graph_add_buffer(
            self.g_step, 64, MG_BUF_STATE, None, -1, 0
        ))
        self.buf_active_mask = int(self.lib.mesh_graph_add_buffer(
            self.g_step, max(64, self.E * self.N), MG_BUF_STATE, None, -1, 0
        ))

        self.buf_obs = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.N * OBS_WIDTH * 2, MG_BUF_STATE, None, -1, 0
        ))
        self.buf_carts = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.J * CART_WIDTH * 2, MG_BUF_STATE, None, -1, 0
        ))
        self.buf_teams = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.K * TEAM_WIDTH * 2, MG_BUF_STATE, None, -1, 0
        ))

        op_step = _make_op(
            MG_OP_SIMDT_STEP,
            in_bufs=(self.buf_world, self.buf_state, self.buf_plan, self.buf_cfg, self.buf_active_mask),
            out_bufs=(-1,),
            u=(self.E, max(self.N, self.J, self.K)),
        )
        self.lib.mesh_graph_add_op(self.g_step, ctypes.byref(op_step))
        self.lib.mesh_graph_compile(self.g_step)

        # Standalone obs graph importing zero-copy buffers from g_step
        ow = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_world)
        os_ = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_state)
        oo = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_obs)
        oc = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_carts)
        ot = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_teams)
        op_obs = _make_op(MG_OP_SIMDT_OBS, in_bufs=(ow, os_), out_bufs=(oo, oc, ot), u=(self.E,))
        self.lib.mesh_graph_add_op(self.g_obs, ctypes.byref(op_obs))
        self.lib.mesh_graph_compile(self.g_obs)

        # Zero-copy unified-memory pointers & views
        self._state_ptr = int(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_state))
        self._state_u8 = _ptr_to_u8_array(self._state_ptr, self.E * self.env_stride).reshape(self.E, self.env_stride)
        self._cfg_ptr = int(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_cfg))
        self._plan_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_plan), (self.E, self.N, PLAN_WIDTH))
        self._mask_u8 = _ptr_to_u8_array(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_active_mask), self.E * self.N).reshape(self.E, self.N)
        self.obs_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_obs), (self.E, self.N, OBS_WIDTH))
        self.carts_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_carts), (self.E, self.J, CART_WIDTH))
        self.teams_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_teams), (self.E, self.K, TEAM_WIDTH))

        self._pending_apply_plan = False
        self._pending_has_mask = False
        self._geo = {
            "map_center": self.world.map_center.clone(),
            "map_radius": self.world.map_radius.clone(),
        }

        # Precompute initial cart positions at s = 0
        self._init_c_pos = self.world.cart_nodes[:, 0, :].detach().cpu().numpy().astype(np.float32)
        self._spawn_pos_np = self.world.spawn_pos.detach().cpu().numpy().astype(np.float32)
        self._spawn_wp_np = self.world.spawn_wp.detach().cpu().numpy().astype(np.int32)
        self.reset(base_seed=base_seed)

    def close(self) -> None:
        if getattr(self, "g_obs", None):
            self.lib.mesh_graph_destroy(self.g_obs)
            self.g_obs = None
        if getattr(self, "g_step", None):
            self.lib.mesh_graph_destroy(self.g_step)
            self.g_step = None

    def __del__(self) -> None:
        self.close()

    def _write_cfg(
        self, dt: float, mode: int, dt_ref: float, reset_outcomes: bool,
        apply_plan: bool, has_mask: bool, record_traj: bool = False,
    ) -> None:
        raw = struct.pack(
            "<2f6I",
            float(dt), float(dt_ref), int(mode),
            1 if reset_outcomes else 0,
            1 if apply_plan else 0,
            1 if has_mask else 0,
            1 if record_traj else 0,
            0,
        )
        ctypes.memmove(self._cfg_ptr, raw, len(raw))

    def reset(
        self,
        env_mask: torch.Tensor | np.ndarray | None = None,
        *,
        base_seed: int | None = None,
        skill_mean: float = 4.0,
        skill_std: float = 0.0,
    ) -> None:
        E, N, J, K = self.E, self.N, self.J, self.K
        L = self.layout
        full = env_mask is None
        if full:
            mask_np = np.ones((E,), dtype=bool)
        else:
            mask_np = np.asarray(env_mask.detach().cpu().numpy() if isinstance(env_mask, torch.Tensor) else env_mask, dtype=bool)
            if not np.any(mask_np):
                return

        n_sp = int(self._spawn_pos_np.shape[1])
        teams_1d = (np.arange(N, dtype=np.int32) % K) + 1
        sp_slot = (np.arange(N, dtype=np.int32) // K) % n_sp
        t0 = teams_1d - 1
        sp_pos = self._spawn_pos_np[t0, sp_slot]  # [N, 3]
        sp_wp = self._spawn_wp_np[t0, sp_slot]    # [N]
        pref_cart = ((np.arange(1, N + 1, dtype=np.int32)) % max(1, J))

        # Build template env bytes once, then broadcast to masked envs!
        tmpl = np.zeros((self.env_stride,), dtype=np.uint8)
        tmpl[L["soff_p_pos"] : L["soff_p_pos"] + N * 12].view(np.float32).reshape(N, 3)[:] = sp_pos
        p_f1 = tmpl[L["soff_p_f1"] : L["soff_p_f1"] + 15 * N * 4].view(np.float32).reshape(15, N)
        p_f1[0, :] = float(self.rules.max_health)
        p_f1[1, :] = float(self.rules.start_armor)
        p_f1[4, :] = float(skill_mean)
        p_i1 = tmpl[L["soff_p_i1"] : L["soff_p_i1"] + 8 * N * 4].view(np.int32).reshape(8, N)
        p_i1[0, :] = teams_1d
        p_i1[1, :] = 1
        p_i1[2, :] = sp_wp
        p_i1[3, :] = sp_wp
        p_i1[4, :] = sp_wp
        p_i1[5, :] = pref_cart
        p_i1[6, :] = -1
        p_i1[7, :] = 0
        tmpl[L["soff_c_pos"] : L["soff_c_pos"] + J * 12].view(np.float32).reshape(J, 3)[:] = self._init_c_pos

        env_ids = np.nonzero(mask_np)[0]
        if full:
            seeds = np.arange(1 if base_seed is None else int(base_seed), (1 if base_seed is None else int(base_seed)) + E, dtype=np.uint64)
            self._state_u8[:] = tmpl[None, :]
        else:
            prev_meta = self._state_u8[env_ids, L["soff_meta"] : L["soff_meta"] + 32].view(np.uint32)
            prev_seeds = prev_meta[:, 3].astype(np.uint64) | (prev_meta[:, 4].astype(np.uint64) << np.uint64(32))
            delta = np.uint64(int(base_seed) if base_seed is not None else 1000003)
            seeds = prev_seeds + delta
            # Preserve existing p_skill on partial reset (matches engine.py!)
            prev_skill = self._state_u8[env_ids, L["soff_p_f1"] + 4 * N * 4 : L["soff_p_f1"] + 5 * N * 4].copy()
            self._state_u8[env_ids] = tmpl[None, :]
            self._state_u8[env_ids, L["soff_p_f1"] + 4 * N * 4 : L["soff_p_f1"] + 5 * N * 4] = prev_skill

        meta_u32 = self._state_u8[env_ids, L["soff_meta"] : L["soff_meta"] + 32].view(np.uint32)
        meta_u32[:, 3] = (seeds & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        meta_u32[:, 4] = (seeds >> np.uint64(32)).astype(np.uint32)

        if full and skill_std > 0.0:
            # Match _philox_u01(seed, 0, 101, (E, N))
            idx = np.arange(N, dtype=np.int64)[None, :]
            sd = seeds.astype(np.int64)[:, None]
            x = (sd * 0x9E3779B1 + idx * 0xC2B2AE3D + 101) & 0x7FFFFFFF
            x = ((x ^ (x >> 15)) * 0x2C1B3C6D) & 0x7FFFFFFF
            x = ((x ^ (x >> 12)) * 0x297A2D39) & 0x7FFFFFFF
            u = ((x ^ (x >> 15)) & 0x00FFFFFF).astype(np.float32) * (1.0 / 16777216.0)
            sk = np.clip(float(skill_mean) + float(skill_std) * (2.0 * u - 1.0), 0.0, 20.0).astype(np.float32)
            self._state_u8[:, L["soff_p_f1"] + 4 * N * 4 : L["soff_p_f1"] + 5 * N * 4].view(np.float32)[:] = sk

        self._pending_apply_plan = False
        self._pending_has_mask = False

    def apply_plan(self, plan: torch.Tensor, active_mask: torch.Tensor | None = None) -> None:
        self._plan_view.copy_(plan.detach().to(device="cpu", dtype=torch.float32))
        self._pending_apply_plan = True
        if active_mask is not None:
            m = active_mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
            self._mask_u8[:] = m
            self._pending_has_mask = True
        else:
            self._pending_has_mask = False

    def step(
        self,
        dt: float = 0.4,
        *,
        mode: str = "analytic",
        dt_ref: float | None = None,
        reset_outcomes: bool = True,
        n_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        mode_i = 0 if mode == "analytic" else (1 if mode == "substep" else 2)
        dref = float(self.rules.plc_tick_ref if dt_ref is None else dt_ref)
        self._write_cfg(
            float(dt), mode_i, dref, reset_outcomes,
            self._pending_apply_plan, self._pending_has_mask, record_traj=False,
        )
        wall = ctypes.c_double(0.0)
        if n_steps == 1:
            self.lib.mesh_graph_run(self.g_step, 1, 0, None, 0, ctypes.byref(wall))
        else:
            self.lib.mesh_graph_run_fused(self.g_step, int(n_steps), 0, ctypes.byref(wall))
        self._pending_apply_plan = False
        self._pending_has_mask = False
        return {"finished": self.t_finished, "winner": self.t_winner, "wall_ms": float(wall.value)}

    def observe_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g_obs, 1, 0, None, 0, ctypes.byref(wall))
        return self.obs_view, self.carts_view, self.teams_view, self._geo

    def snapshot(self, env_ids: torch.Tensor | np.ndarray | None = None) -> MetalSimSnapshot:
        if env_ids is None:
            raw = self._state_u8.tobytes()
            cnt = self.E
        else:
            ids = np.asarray(env_ids.detach().cpu().numpy() if isinstance(env_ids, torch.Tensor) else env_ids, dtype=np.int64)
            raw = np.ascontiguousarray(self._state_u8[ids]).tobytes()
            cnt = len(ids)
        return MetalSimSnapshot(
            E=cnt, N=self.N, J=self.J, K=self.K,
            env_stride=self.env_stride, soff_meta=self.layout["soff_meta"], data=raw,
        )

    def restore(self, snap: MetalSimSnapshot) -> None:
        assert snap.E == self.E and snap.env_stride == self.env_stride
        ctypes.memmove(self._state_ptr, snap.data, len(snap.data))

    @classmethod
    def fork_group(
        cls,
        world: WorldGraphIR,
        snap: MetalSimSnapshot,
        group_size: int,
        *,
        rules: SimRulesConfig = SimRulesConfig(),
        common_random_numbers: bool = False,
        seed_stride: int = 10007,
    ) -> "MetalDeltaTimePayloadEngine":
        B, G = snap.E, int(group_size)
        eng = cls(world, num_envs=B * G, num_players=snap.N, rules=rules)
        src = np.frombuffer(snap.data, dtype=np.uint8).reshape(B, snap.env_stride)
        rep = np.repeat(src, G, axis=0)
        eng._state_u8[:] = rep
        if not common_random_numbers:
            sm = snap.soff_meta
            meta_u32 = eng._state_u8[:, sm : sm + 32].view(np.uint32)
            seeds = meta_u32[:, 3].astype(np.uint64) | (meta_u32[:, 4].astype(np.uint64) << np.uint64(32))
            g_off = np.tile(np.arange(G, dtype=np.uint64), B) * np.uint64(seed_stride)
            seeds = seeds + g_off
            meta_u32[:, 3] = (seeds & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            meta_u32[:, 4] = (seeds >> np.uint64(32)).astype(np.uint32)
        return eng

    # Zero-copy state inspection properties
    @property
    def c_s(self) -> torch.Tensor:
        o = self.layout["soff_c_f1"]
        return torch.from_numpy(self._state_u8[:, o : o + self.J * 4].view(np.float32).copy())

    @property
    def t_score(self) -> torch.Tensor:
        o = self.layout["soff_t_score"]
        return torch.from_numpy(self._state_u8[:, o : o + self.K * 4].view(np.float32).copy())

    @property
    def t_cp_held(self) -> torch.Tensor:
        o = self.layout["soff_t_cp_held"]
        return torch.from_numpy(self._state_u8[:, o : o + self.K * 4].view(np.int32).copy()).to(torch.int64)

    @property
    def t_finished(self) -> torch.Tensor:
        o = self.layout["soff_meta"]
        return torch.from_numpy((self._state_u8[:, o + 20 : o + 24].view(np.uint32)[:, 0] != 0).copy())

    @property
    def t_winner(self) -> torch.Tensor:
        o = self.layout["soff_meta"]
        return torch.from_numpy(self._state_u8[:, o + 24 : o + 28].view(np.int32)[:, 0].copy()).to(torch.int64)

    @property
    def p_pos(self) -> torch.Tensor:
        o = self.layout["soff_p_pos"]
        return torch.from_numpy(self._state_u8[:, o : o + self.N * 12].view(np.float32).reshape(self.E, self.N, 3).copy())

    @property
    def p_hp(self) -> torch.Tensor:
        o = self.layout["soff_p_f1"]
        return torch.from_numpy(self._state_u8[:, o : o + self.N * 4].view(np.float32).copy())

    @property
    def p_alive(self) -> torch.Tensor:
        o = self.layout["soff_p_i1"] + self.N * 4
        return torch.from_numpy((self._state_u8[:, o : o + self.N * 4].view(np.int32) != 0).copy())

    @property
    def sim_time(self) -> torch.Tensor:
        o = self.layout["soff_meta"]
        return torch.from_numpy(self._state_u8[:, o : o + 4].view(np.float32)[:, 0].copy())


# ============================================================================
# 3. CompiledRolloutGraph — Single-CB Closed-Loop sim_dt + PlanWally Rollout!
# ============================================================================

class CompiledRolloutGraph:
    """
    Fuses [MG_OP_SIMDT_OBS -> CompiledPlanWally (BF16 Metal 4 NAX + in-GPU sample/pack) -> MG_OP_SIMDT_STEP]
    into ONE single mesh_graph_t so an entire H-step multi-agent rollout across E parallel matches
    executes inside ONE single MTLCommandBuffer with ZERO Python calls, ZERO GIL, ZERO CPU sync!
    """

    def __init__(
        self,
        engine: MetalDeltaTimePayloadEngine,
        model: PlanWally,
        *,
        max_steps: int = 256,
        stochastic: bool = True,
        sample_seed: int = 1,
        pack_cfg: ActionPackConfig = DEFAULT_ACTION_PACK_CONFIG,
    ):
        self.lib = engine.lib
        self.engine = engine
        self.max_steps = int(max_steps)
        self.g = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for CompiledRolloutGraph")

        # Zero-copy import all engine buffers into the unified rollout graph!
        self.b_world = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_world)
        self.b_state = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_state)
        self.b_plan = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_plan)
        self.b_cfg = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_cfg)
        self.b_mask = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_active_mask)
        self.b_obs = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_obs)
        self.b_carts = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_carts)
        self.b_teams = self.lib.mesh_graph_import_buffer(self.g, engine.g_step, engine.buf_teams)

        # Trajectory summary buffer: [max_steps, E, 2 + K] f32 (finished, winner, t_score[0..K-1])
        self.b_traj = int(self.lib.mesh_graph_add_buffer(
            self.g, self.max_steps * engine.E * (2 + engine.K) * 4, MG_BUF_STATE, None, -1, 0
        ))

        # 1. Emit BF16 obs/carts/teams from simdt SoA state
        op_obs = _make_op(
            MG_OP_SIMDT_OBS,
            in_bufs=(self.b_world, self.b_state),
            out_bufs=(self.b_obs, self.b_carts, self.b_teams),
            u=(engine.E,),
        )
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op_obs))

        # 2. Lower full stripped BF16 PlanWally + in-GPU sample & pack into b_plan!
        self.policy = CompiledPlanWally(
            model,
            E=engine.E, N=engine.N, J=engine.J, K=engine.K,
            pack_cfg=pack_cfg,
            do_pack=True,
            stochastic=stochastic,
            write_sample=False,
            sample_seed=sample_seed,
            graph=self.g,
            ext_obs_buf=self.b_obs,
            ext_carts_buf=self.b_carts,
            ext_teams_buf=self.b_teams,
            ext_plan_buf=self.b_plan,
        )
        self.policy.set_geo(engine._geo)

        # 3. Step DeltaTimePayloadEngine consuming b_plan directly in GPU SRAM!
        op_step = _make_op(
            MG_OP_SIMDT_STEP,
            in_bufs=(self.b_world, self.b_state, self.b_plan, self.b_cfg, self.b_mask),
            out_bufs=(self.b_traj,),
            u=(engine.E, max(engine.N, engine.J, engine.K)),
        )
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op_step))

        rc = self.lib.mesh_graph_compile(self.g)
        if rc != 0:
            raise RuntimeError(f"mesh_graph_compile failed for CompiledRolloutGraph: {rc}")

        self.traj_view = _ptr_to_torch_f32(
            self.lib.mesh_graph_buffer_ptr(self.g, self.b_traj),
            (self.max_steps, engine.E, 2 + engine.K),
        )

    def close(self) -> None:
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    def __del__(self) -> None:
        self.close()

    def sync_weights_(self, model: PlanWally) -> None:
        self.policy.sync_weights_(model)

    def run_rollout(
        self,
        n_steps: int,
        *,
        dt: float = 0.4,
        mode: str = "analytic",
        dt_ref: float | None = None,
        record_traj: bool = True,
        fused_cb: bool = True,
        warm_steps: int = 0,
    ) -> dict[str, Any]:
        assert n_steps <= self.max_steps, f"n_steps={n_steps} > max_steps={self.max_steps}"
        mode_i = 0 if mode == "analytic" else (1 if mode == "substep" else 2)
        dref = float(self.engine.rules.plc_tick_ref if dt_ref is None else dt_ref)
        self.engine._write_cfg(
            float(dt), mode_i, dref,
            reset_outcomes=True, apply_plan=True, has_mask=False, record_traj=record_traj,
        )
        wall = ctypes.c_double(0.0)
        if fused_cb:
            ms_per_step = float(self.lib.mesh_graph_run_fused(self.g, int(n_steps), int(warm_steps), ctypes.byref(wall)))
        else:
            ms_per_step = float(self.lib.mesh_graph_run(self.g, int(n_steps), int(warm_steps), None, 0, ctypes.byref(wall)))
        return {
            "n_steps": int(n_steps),
            "wall_ms": float(wall.value),
            "ms_per_step": ms_per_step,
            "finished": self.engine.t_finished,
            "winner": self.engine.t_winner,
            "t_score": self.engine.t_score,
            "traj": self.traj_view[:n_steps] if record_traj else None,
        }
