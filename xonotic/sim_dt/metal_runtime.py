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

MG_WIRE_HAD8_GATHER = 4
MG_WIRE_AUTO = 5

MG_OP_LINEAR = 2
MG_OP_EW = 4
MG_OP_ALLREDUCE = 12
MG_EW_ADD = 0
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
MG_OP_LINEAR_DX = 29
MG_OP_LINEAR_DW = 30
MG_OP_PLAN_LOSS_BWD = 31
MG_OP_BILINEAR_BWD = 32
MG_OP_PLAN_SLICE_BWD = 33
MG_OP_MP_NORM_BWD = 34
MG_OP_MP_LERP_BWD = 35
MG_OP_MP_SWIGLU_BWD = 36
MG_OP_CGT_BCAST_BWD = 37
MG_OP_CGT_RIVAL_BWD = 38
MG_OP_CGT_STAGE1_BWD = 39
MG_OP_PLAN_EMBED_BWD = 40
MG_OP_UNIT_W_BWD = 41
MG_OP_ZERO_F32 = 42
MG_OP_FWHT_PACK_I8 = 43
MG_OP_LINEAR_I8 = 44
MG_OP_SIMDT_REWARD = 45
MG_OP_OPT_TANGENT_NORM = 46
MG_OP_OPT_MOM_PREP = 47
MG_OP_OPT_NS_AXPBY = 48
MG_OP_OPT_AURORA_DREF = 49
MG_OP_OPT_APPLY_2D = 50
MG_OP_OPT_APPLY_1D = 51


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
    lib.mesh_graph_add_crossing.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
    ]
    lib.mesh_graph_add_crossing.restype = ctypes.c_int
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
        tp_world_size: int = 1,
        tp_rank: int = 0,
        tp_comm_id: str = "",
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
        self.tp_world_size = max(1, int(tp_world_size))
        self.tp_rank = int(tp_rank)
        assert self.h % self.tp_world_size == 0 and (self.h // self.tp_world_size) % 64 == 0, (
            f"FFN hidden h={self.h} must divide evenly by tp_world_size={self.tp_world_size} into multiples of 64"
        )
        self.h_local = self.h // self.tp_world_size
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
            self.g = self.lib.mesh_graph_create(
                self.tp_world_size, self.tp_rank, tp_comm_id.encode("utf-8"), MG_WIRE_HAD8_GATHER if self.tp_world_size > 1 else MG_WIRE_AUTO,
            )
            if not self.g:
                raise RuntimeError("mesh_graph_create failed")
        else:
            self.g = graph

        self._w_bufs: dict[str, int] = {}
        self._wq_bufs: dict[str, tuple[int, int]] = {}
        self._g_repack: int | None = None
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
        if getattr(self, "_g_repack", None):
            self.lib.mesh_graph_destroy(self._g_repack)
            self._g_repack = None
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
        D, H, Rg, Rt = self.d, self.h_local, self.r_gram, self.r_team
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

        # Static BF16 + compressed had_int8 weights (repacked in-GPU via MG_OP_FWHT_PACK_I8)
        w = self._w_bufs
        wq = self._wq_bufs
        repack_specs: list[tuple[str, int, int]] = []
        w["enc_obs"] = self._add_buf(D * 128 * 2, MG_BUF_WEIGHT)
        w["enc_cart"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["enc_team"] = self._add_buf(D * 64 * 2, MG_BUF_WEIGHT)
        w["type_embed"] = self._add_buf(3 * D * 2, MG_BUF_WEIGHT)

        def add_i8_w(key: str, rows: int, cols: int) -> None:
            w[key] = self._add_buf(rows * cols * 2, MG_BUF_WEIGHT)
            wq[key] = (self._add_buf(rows * cols, MG_BUF_STATE), self._add_buf(rows * 4, MG_BUF_STATE))
            repack_specs.append((key, rows, cols))

        for bi in range(n_uniq_blk):
            p = f"b{bi}."
            add_i8_w(p + "val_proj", D, D)
            add_i8_w(p + "global_metric", Rg, D)
            w[p + "pool_score"] = self._add_buf(D * 2, MG_BUF_WEIGHT)
            w[p + "team_metric"] = self._add_buf(Rt * D * 2, MG_BUF_WEIGHT)
            w[p + "rival_metric"] = self._add_buf(Rt * D * 2, MG_BUF_WEIGHT)
            w[p + "team_val_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "team_mutate_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "team_bcast_proj"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            w[p + "gram_out"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
            add_i8_w(p + "ffn_gate_up", 2 * H, D)
            add_i8_w(p + "ffn_down", D, H)

        w["cart_q"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["cart_k"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["rival_q"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["rival_k"] = self._add_buf(D * D * 2, MG_BUF_WEIGHT)
        w["heads_packed"] = self._add_buf(64 * D * 2, MG_BUF_WEIGHT)

        # Build zero-copy in-GPU weight repack graph (W_bf16 -> W_had_i8 + W_s_f32 in <0.03 ms!)
        self._g_repack = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        for key, rows, cols in repack_specs:
            iw = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, w[key])
            iq = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][0])
            is_ = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][1])
            op_r = _make_op(MG_OP_FWHT_PACK_I8, in_bufs=(iw,), out_bufs=(iq, is_), u=(rows, cols, 0))
            self.lib.mesh_graph_add_op(self._g_repack, ctypes.byref(op_r))
        self.lib.mesh_graph_compile(self._g_repack)

        # Static shared activation arena (zero dynamic allocation per step!)
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
        s_h = alloc_slot(max(EK_pad * D, EN_pad * 64) * 2)
        s_q = alloc_slot(EM_pad * max(D, H))
        s_s = alloc_slot(EM_pad * 4)

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
        b_q = self._add_view(self.buf_arena, *s_q)
        b_s = self._add_view(self.buf_arena, *s_s)

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

            # Fused mp_norm(x) -> (b_xn, b_q, b_s) + compressed INT8@8 NAX val_proj & global_metric!
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(b_x,), out_bufs=(b_xn, b_q, b_s), u=(M_tot, D, EM_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "val_proj"]), out_bufs=(b_b,), u=(EM_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "global_metric"]), out_bufs=(b_c,), u=(EM_pad, Rg, D, 0)))

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

            # gram_out -> b_b, fused mp_lerp -> (b_d, b_q, b_s), compressed INT8@8 NAX FFN!
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_a, w[p + "gram_out"]), out_bufs=(b_b,), u=(EM_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_MP_LERP,
                in_bufs=(b_xn, b_b, self.buf_scalars),
                out_bufs=(b_d, b_q, b_s),
                u=(M_tot, D, sc0 + 2, 1, EM_pad),
                f=(eps_n, eps_l),
            ))

            # FFN SwiGLU: compressed INT8@8 NAX gate_up -> fused mp_swiglu_i8 (write_bf=0!) -> INT8@8 NAX ffn_down!
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_gate_up"]), out_bufs=(b_a,), u=(EM_pad, 2 * H, D, 0)))
            self._add_op(_make_op(
                MG_OP_MP_SWIGLU,
                in_bufs=(b_a,),
                out_bufs=(b_b, b_q, b_s),
                u=(M_tot, H, EM_pad, 0),
                f=(1.0 / float(self.nc.mp_silu_scale),),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_down"]), out_bufs=(b_c,), u=(EM_pad, D, H, 0)))
            if self.tp_world_size > 1:
                cid = int(self.lib.mesh_graph_add_crossing(self.g, EM_pad * D, 2, MG_WIRE_HAD8_GATHER, 0))
                self._add_op(_make_op(MG_OP_ALLREDUCE, in_bufs=(b_c,), out_bufs=(b_c,), u=(cid, EM_pad, D)))

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
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_c, w["heads_packed"]), out_bufs=(b_h,), u=(EN_pad, 64, D)))

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
            wd = _unit_weight_bf16(blk.ffn_down.weight, eps)
            if self.tp_world_size > 1:
                lo = self.tp_rank * self.h_local
                hi = lo + self.h_local
                wg = wg[lo:hi].contiguous()
                wu = wu[lo:hi].contiguous()
                wd = wd[:, lo:hi].contiguous()
            put(p + "ffn_gate_up", torch.cat((wg, wu), dim=0))
            put(p + "ffn_down", wd)
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

        hp_w = torch.zeros((64, self.d), dtype=torch.bfloat16)
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
        if self._g_repack:
            wt = ctypes.c_double(0.0)
            self.lib.mesh_graph_run_fused(self._g_repack, 1, 0, ctypes.byref(wt))

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

    @torch.no_grad()
    def forward_sample(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        geo: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-CB forward + in-GPU Philox sample + pack_plan (ZERO Python distribution overhead!)."""
        assert self.plan_view is not None and self.sample_view is not None, (
            "forward_sample requires CompiledPlanWally(do_pack=True, write_sample=True)"
        )
        self.obs_view.copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
        self.carts_view.copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
        self.teams_view.copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
        self.set_geo(geo)
        self.run_inplace(1, 0)
        sv = self.sample_view
        ho = self.head_out_view
        return {
            "plan": self.plan_view.clone(),
            "cont": sv[:, :, 0:11].to(torch.bfloat16),
            "cart_pick": sv[:, :, 11].to(torch.int16),
            "rival_pick": sv[:, :, 12].to(torch.int16),
            "logp": sv[:, :, 13].clone(),
            "value_winnie": ho[:, :, 11].to(torch.float32),
            "value_lou": ho[:, :, 12].to(torch.float32),
        }


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
    W_nav = int(world.num_nav)
    I_item = int(world.num_items)
    A_dim = W_nav + K + 4
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
    soff_p_f1 = s_alloc(17 * N * 4)
    soff_p_i1 = s_alloc(8 * N * 4)
    soff_p_plan = s_alloc(N * 16 * 4)
    soff_c_f1 = s_alloc(3 * J * 4)
    soff_c_pos = s_alloc(J * 3 * 4)
    soff_c_i1 = s_alloc(5 * J * 4)
    soff_c_cp_owner = s_alloc(J * CP * 4)
    soff_t_score = s_alloc(K * 4)
    soff_t_cp_held = s_alloc(K * 4)
    soff_item_cd = s_alloc(max(1, I_item) * 4)
    soff_p_r_acc = s_alloc(N * A_dim * 4)
    soff_p_u_vel = s_alloc(N * A_dim * 4)
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
        ("wp_to_nav", np.ascontiguousarray(world.wp_to_nav.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("nav_rep_wp", np.ascontiguousarray(world.nav_rep_wp.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("nav_pos", np.ascontiguousarray(world.nav_pos.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("nav_degree", np.ascontiguousarray(world.nav_degree.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("item_wp", np.ascontiguousarray(world.item_wp.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("item_nav", np.ascontiguousarray(world.item_nav.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("item_pos", np.ascontiguousarray(world.item_pos.detach().cpu().numpy(), dtype=np.float32).tobytes()),
        ("item_kind", np.ascontiguousarray(world.item_kind.detach().cpu().numpy(), dtype=np.int32).tobytes()),
        ("item_respawn", np.ascontiguousarray(world.item_respawn.detach().cpu().numpy(), dtype=np.float32).tobytes()),
    ]

    hdr_fmt = "<54I18f"
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
        W_nav, I_item, A_dim,
        offs["wp_pos"], offs["wp_dist"], offs["wp_next_hop"],
        offs["cart_nodes"], offs["cart_node_s"], offs["cart_seg_count"], offs["cart_length"],
        offs["cart_cp_dists"], offs["cart_cp_count"], offs["cart_radius"], offs["cart_height"],
        offs["cart_speed_scale"], offs["cart_sample_wp"], offs["spawn_pos"], offs["spawn_wp"],
        offs["push_lut"],
        offs["wp_to_nav"], offs["nav_rep_wp"], offs["nav_pos"], offs["nav_degree"],
        offs["item_wp"], offs["item_nav"], offs["item_pos"], offs["item_kind"], offs["item_respawn"],
        env_stride,
        soff_meta, soff_p_pos, soff_p_vel, soff_p_f1, soff_p_i1, soff_p_plan,
        soff_c_f1, soff_c_pos, soff_c_i1, soff_c_cp_owner, soff_t_score, soff_t_cp_held,
        soff_item_cd, soff_p_r_acc, soff_p_u_vel,
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
        W_nav=W_nav,
        I_item=I_item,
        A_dim=A_dim,
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
        soff_item_cd=soff_item_cd,
        soff_p_r_acc=soff_p_r_acc,
        soff_p_u_vel=soff_p_u_vel,
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
        self.W_nav = int(self.world.num_nav)
        self.I_item = int(self.world.num_items)
        self.A_dim = self.W_nav + self.K + 4
        assert self.N <= 128 and self.J <= 64 and self.K <= 16 and self.CP <= 16 and self.W_nav <= 64 and self.I_item <= 64, (
            f"MetalDeltaTimePayloadEngine TG SRAM bounds: N<=128, J<=64, K<=16, CP<=16, W_nav<=64, I_item<=64 "
            f"(got {self.N},{self.J},{self.K},{self.CP},{self.W_nav},{self.I_item})"
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
        self.buf_vel = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.N * self.A_dim * 4, MG_BUF_STATE, None, -1, 0
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
        self.buf_nav = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.W_nav * 16 * 2, MG_BUF_STATE, None, -1, 0
        ))
        self.buf_acc = int(self.lib.mesh_graph_add_buffer(
            self.g_step, self.E * self.N * self.A_dim * 4, MG_BUF_STATE, None, -1, 0
        ))

        op_step = _make_op(
            MG_OP_SIMDT_STEP,
            in_bufs=(self.buf_world, self.buf_state, self.buf_plan, self.buf_cfg, self.buf_active_mask, self.buf_vel),
            out_bufs=(-1,),
            u=(self.E, max(self.N, self.J, self.K, self.I_item)),
        )
        self.lib.mesh_graph_add_op(self.g_step, ctypes.byref(op_step))
        self.lib.mesh_graph_compile(self.g_step)

        # Standalone obs graph importing zero-copy buffers from g_step
        ow = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_world)
        os_ = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_state)
        oo = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_obs)
        oc = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_carts)
        ot = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_teams)
        onav = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_nav)
        oacc = self.lib.mesh_graph_import_buffer(self.g_obs, self.g_step, self.buf_acc)
        op_obs = _make_op(MG_OP_SIMDT_OBS, in_bufs=(ow, os_, onav, oacc), out_bufs=(oo, oc, ot), u=(self.E,))
        self.lib.mesh_graph_add_op(self.g_obs, ctypes.byref(op_obs))
        self.lib.mesh_graph_compile(self.g_obs)

        # Zero-copy unified-memory pointers & views
        self._state_ptr = int(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_state))
        self._state_u8 = _ptr_to_u8_array(self._state_ptr, self.E * self.env_stride).reshape(self.E, self.env_stride)
        self._cfg_ptr = int(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_cfg))
        self._plan_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_plan), (self.E, self.N, PLAN_WIDTH))
        self._vel_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_vel), (self.E, self.N, self.A_dim))
        self._mask_u8 = _ptr_to_u8_array(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_active_mask), self.E * self.N).reshape(self.E, self.N)
        self.obs_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_obs), (self.E, self.N, OBS_WIDTH))
        self.carts_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_carts), (self.E, self.J, CART_WIDTH))
        self.teams_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_teams), (self.E, self.K, TEAM_WIDTH))
        self.nav_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_nav), (self.E, self.W_nav, 16))
        self.acc_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g_step, self.buf_acc), (self.E, self.N, self.A_dim))

        self._pending_apply_plan = False
        self._pending_apply_vel = False
        self._pending_has_mask = False
        self._tau_rel = 2.5
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
        self, dt: float, mode: int, dt_ref: float, reset_outcomes: bool | int,
        apply_plan: bool, has_mask: bool, record_traj: bool = False,
        apply_vel: bool = False, tau_rel: float = 2.5,
    ) -> None:
        ro_i = int(reset_outcomes) if isinstance(reset_outcomes, int) and not isinstance(reset_outcomes, bool) else (1 if reset_outcomes else 0)
        raw = struct.pack(
            "<3f6I",
            float(dt), float(dt_ref), float(tau_rel), int(mode),
            ro_i,
            1 if apply_plan else 0,
            1 if has_mask else 0,
            1 if record_traj else 0,
            1 if apply_vel else 0,
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

        # Build template env bytes once, then broadcast to masked envs!
        tmpl = np.zeros((self.env_stride,), dtype=np.uint8)
        tmpl[L["soff_p_pos"] : L["soff_p_pos"] + N * 12].view(np.float32).reshape(N, 3)[:] = sp_pos
        p_f1 = tmpl[L["soff_p_f1"] : L["soff_p_f1"] + 17 * N * 4].view(np.float32).reshape(17, N)
        p_f1[0, :] = float(self.rules.max_health)
        p_f1[1, :] = float(self.rules.start_armor)
        p_f1[4, :] = float(skill_mean)
        p_i1 = tmpl[L["soff_p_i1"] : L["soff_p_i1"] + 8 * N * 4].view(np.int32).reshape(8, N)
        p_i1[0, :] = teams_1d
        p_i1[1, :] = 1
        p_i1[2, :] = sp_wp
        p_i1[3, :] = sp_wp
        p_i1[4, :] = sp_wp
        p_i1[5, :] = 0
        p_i1[6, :] = 0
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
        self._pending_apply_vel = False
        self._pending_has_mask = False

    def apply_plan(self, plan: torch.Tensor, active_mask: torch.Tensor | None = None) -> None:
        self._plan_view.copy_(plan.detach().to(device="cpu", dtype=torch.float32))
        self._pending_apply_plan = True
        self._pending_apply_vel = False
        if active_mask is not None:
            m = active_mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
            self._mask_u8[:] = m
            self._pending_has_mask = True
        else:
            self._pending_has_mask = False

    def apply_vel(
        self,
        vel: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        *,
        tau_rel: float = 2.5,
    ) -> None:
        self._vel_view.copy_(vel.detach().to(device="cpu", dtype=torch.float32))
        self._pending_apply_vel = True
        self._pending_apply_plan = False
        self._tau_rel = float(tau_rel)
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
        tau_rel: float | None = None,
    ) -> dict[str, torch.Tensor]:
        mode_i = 0 if mode == "analytic" else (1 if mode == "substep" else 2)
        dref = float(self.rules.plc_tick_ref if dt_ref is None else dt_ref)
        t_rel = float(self._tau_rel if tau_rel is None else tau_rel)
        ro_cfg: bool | int = 2 if (n_steps > 1 and reset_outcomes) else bool(reset_outcomes)
        self._write_cfg(
            float(dt), mode_i, dref, ro_cfg,
            self._pending_apply_plan, self._pending_has_mask, record_traj=False,
            apply_vel=self._pending_apply_vel, tau_rel=t_rel,
        )
        wall = ctypes.c_double(0.0)
        if n_steps == 1:
            self.lib.mesh_graph_run(self.g_step, 1, 0, None, 0, ctypes.byref(wall))
        else:
            self.lib.mesh_graph_run_fused(self.g_step, int(n_steps), 0, ctypes.byref(wall))
        self._pending_apply_plan = False
        self._pending_apply_vel = False
        self._pending_has_mask = False
        return {"finished": self.t_finished, "winner": self.t_winner, "wall_ms": float(wall.value)}

    def observe_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g_obs, 1, 0, None, 0, ctypes.byref(wall))
        return self.obs_view, self.carts_view, self.teams_view, self._geo

    def observe_graph_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g_obs, 1, 0, None, 0, ctypes.byref(wall))
        return self.obs_view, self.nav_view, self.carts_view, self.teams_view, self.acc_view, self._geo

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

    def state_views(self) -> dict[str, torch.Tensor]:
        """Zero-copy writable CPU unified-memory views into the live GPU state buffer."""
        L = self.layout
        E, N, J, K = self.E, self.N, self.J, self.K
        u8 = self._state_u8
        p_f1 = u8[:, L["soff_p_f1"] : L["soff_p_f1"] + 17 * N * 4].view(np.float32).reshape(E, 17, N)
        p_i1 = u8[:, L["soff_p_i1"] : L["soff_p_i1"] + 8 * N * 4].view(np.int32).reshape(E, 8, N)
        return {
            "p_pos": torch.from_numpy(u8[:, L["soff_p_pos"] : L["soff_p_pos"] + N * 12].view(np.float32).reshape(E, N, 3)),
            "p_hp": torch.from_numpy(p_f1[:, 0, :]),
            "p_armor": torch.from_numpy(p_f1[:, 1, :]),
            "p_respawn_t": torch.from_numpy(p_f1[:, 2, :]),
            "p_skill": torch.from_numpy(p_f1[:, 4, :]),
            "p_goal_timer": torch.from_numpy(p_f1[:, 6, :]),
            "p_dmg_dealt": torch.from_numpy(p_f1[:, 8, :]),
            "p_dmg_taken": torch.from_numpy(p_f1[:, 9, :]),
            "p_kills": torch.from_numpy(p_f1[:, 10, :]),
            "p_deaths": torch.from_numpy(p_f1[:, 11, :]),
            "p_pickups": torch.from_numpy(p_f1[:, 12, :]),
            "p_cart_push": torch.from_numpy(p_f1[:, 13, :]),
            "p_cart_contest": torch.from_numpy(p_f1[:, 14, :]),
            "p_rockets": torch.from_numpy(p_f1[:, 15, :]),
            "p_strength_t": torch.from_numpy(p_f1[:, 16, :]),
            "p_team": torch.from_numpy(p_i1[:, 0, :]),
            "p_alive": torch.from_numpy(p_i1[:, 1, :]),
            "p_wp_cur": torch.from_numpy(p_i1[:, 2, :]),
            "p_goal_tgt": torch.from_numpy(p_i1[:, 5, :]),
            "p_goal_cart": torch.from_numpy(p_i1[:, 5, :]),
            "p_goal_kind": torch.from_numpy(p_i1[:, 6, :]),
            "p_ctrl_mode": torch.from_numpy(p_i1[:, 7, :]),
            "p_r_acc": torch.from_numpy(u8[:, L["soff_p_r_acc"] : L["soff_p_r_acc"] + N * self.A_dim * 4].view(np.float32).reshape(E, N, self.A_dim)),
            "p_u_vel": torch.from_numpy(u8[:, L["soff_p_u_vel"] : L["soff_p_u_vel"] + N * self.A_dim * 4].view(np.float32).reshape(E, N, self.A_dim)),
            "item_cd": torch.from_numpy(u8[:, L["soff_item_cd"] : L["soff_item_cd"] + self.I_item * 4].view(np.float32).reshape(E, self.I_item)),
            "c_pos": torch.from_numpy(u8[:, L["soff_c_pos"] : L["soff_c_pos"] + J * 12].view(np.float32).reshape(E, J, 3)),
            "c_s": torch.from_numpy(u8[:, L["soff_c_f1"] : L["soff_c_f1"] + J * 4].view(np.float32).reshape(E, J)),
            "t_score": torch.from_numpy(u8[:, L["soff_t_score"] : L["soff_t_score"] + K * 4].view(np.float32).reshape(E, K)),
            "t_cp_held": torch.from_numpy(u8[:, L["soff_t_cp_held"] : L["soff_t_cp_held"] + K * 4].view(np.int32).reshape(E, K)),
        }

    def set_skills(self, skills: np.ndarray | torch.Tensor, env_ids: np.ndarray | torch.Tensor | None = None) -> None:
        """Write per-bot deathmatch mechanics skill [E, N] or [len(env_ids), N] directly into live GPU state."""
        L = self.layout
        N = self.N
        sk = skills.detach().cpu().numpy() if isinstance(skills, torch.Tensor) else np.asarray(skills)
        sk = np.ascontiguousarray(sk, dtype=np.float32)
        s0 = L["soff_p_f1"] + 4 * N * 4
        s1 = s0 + N * 4
        if env_ids is None:
            self._state_u8[:, s0:s1].view(np.float32)[:] = sk
        else:
            ids = env_ids.detach().cpu().numpy() if isinstance(env_ids, torch.Tensor) else np.asarray(env_ids, dtype=np.int64)
            self._state_u8[ids, s0:s1].view(np.float32)[:] = sk

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


# ============================================================================
# 4. CompiledPlanTrainer — Full Metal 4 NAX Forward + Loss + Backward Graph!
# ============================================================================

class CompiledPlanTrainer:
    """
    Full forward + PPO/GRPO/Winnie-Lou/Dynamics loss + analytical backward + FP32
    gradient accumulation for stripped BF16 PlanWally inside ONE Metal 4 command buffer.
    Zero PyTorch MPS eager forward/backward; zero BF16 AccumulateGrad overflow.
    """

    def __init__(
        self,
        model: PlanWally,
        E_mb: int,
        N: int,
        J: int,
        K: int,
        *,
        tp_world_size: int = 1,
        tp_rank: int = 0,
        tp_comm_id: str = "",
    ):
        assert not model.use_fsq and not model.use_mealy and not model.use_manifold_encoders
        assert not model.use_cart_dist_bias and not model.use_cell_head and not model.use_aux_skill_head
        self.lib = _load_mesh_graph_lib()
        self.E = int(E_mb)
        self.N = int(N)
        self.J = int(J)
        self.K = int(K)
        self.M_ent = self.N + self.J + self.K
        self.d = int(model.d)
        self.h = int(model.h)
        self.tp_world_size = max(1, int(tp_world_size))
        self.tp_rank = int(tp_rank)
        assert self.h % self.tp_world_size == 0 and (self.h // self.tp_world_size) % 64 == 0, (
            f"FFN hidden h={self.h} must divide evenly by tp_world_size={self.tp_world_size} into multiples of 64"
        )
        self.h_local = self.h // self.tp_world_size
        self.r_gram = int(model.r_gram)
        self.r_team = int(model.r_dpp)
        self.depth = int(model.depth)
        self.recurrent = bool(model.recurrent)
        self.mc = model.model_cfg
        self.nc = model.norm_cfg

        self.g = self.lib.mesh_graph_create(
            self.tp_world_size, self.tp_rank, tp_comm_id.encode("utf-8"),
            MG_WIRE_HAD8_GATHER if self.tp_world_size > 1 else MG_WIRE_AUTO,
        )
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for CompiledPlanTrainer")

        self._w_bufs: dict[str, int] = {}
        self._wq_bufs: dict[str, tuple[int, int]] = {}
        self._repack_specs: list[tuple[str, int, int]] = []
        self._g_repack: int | None = None
        self._gw_bufs: dict[str, int] = {}
        self._gw_shapes: dict[str, tuple[int, ...]] = {}
        self._build_graph(model)
        self.sync_weights_(model)
        rc = self.lib.mesh_graph_compile(self.g)
        if rc != 0:
            raise RuntimeError(f"mesh_graph_compile failed for CompiledPlanTrainer: {rc}")

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

    def _add_op(self, op: MeshGraphOp) -> None:
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))

    def _add_w_and_gw(self, key: str, rows: int, cols: int) -> None:
        self._w_bufs[key] = self._add_buf(rows * cols * 2, MG_BUF_WEIGHT)
        self._gw_bufs[key] = self._add_buf(rows * cols * 4, MG_BUF_STATE)
        self._gw_shapes[key] = (rows, cols)

    def _build_graph(self, model: PlanWally) -> None:
        E, N, J, K, M_ent = self.E, self.N, self.J, self.K, self.M_ent
        D, H, Rg, Rt = self.d, self.h_local, self.r_gram, self.r_team
        # Pad ALL row counts to multiples of 64 for Metal 4 NAX forward, dX, and dW!
        EN_pad = _align(E * N, 64)
        EJ_pad = _align(E * J, 64)
        EK_pad = _align(E * K, 64)
        EM_pad = _align(E * M_ent, 64)
        M_tot = E * M_ent
        K_tot = E * K
        EN_tot = E * N

        # IO & target buffers
        self.buf_obs = self._add_buf(E * N * OBS_WIDTH * 2, MG_BUF_STATE)
        self.buf_carts = self._add_buf(E * J * CART_WIDTH * 2, MG_BUF_STATE)
        self.buf_teams = self._add_buf(E * K * TEAM_WIDTH * 2, MG_BUF_STATE)
        self.buf_geo = self._add_buf(4 * 4, MG_BUF_STATE)
        self.buf_ncfg = self._add_buf(16 * 4, MG_BUF_WEIGHT)
        self.buf_ent_meta = self._add_buf(E * (M_ent + K) * 4, MG_BUF_STATE)

        self.buf_targ_f = self._add_buf(EN_tot * 17 * 4, MG_BUF_STATE)
        self.buf_targ_i = self._add_buf(EN_tot * 2 * 4, MG_BUF_STATE)
        self.buf_lcfg = self._add_buf(16 * 4, MG_BUF_STATE)
        self.buf_hp = self._add_buf(48 * 4, MG_BUF_STATE)

        n_uniq_blk = 1 if self.recurrent else self.depth
        self._sc_blk_base = 3
        self._sc_step_base = 3 + 4 * n_uniq_blk
        n_scalars = self._sc_step_base + (self.depth if self.recurrent else 0)
        self.buf_scalars = self._add_buf(max(16, n_scalars) * 4, MG_BUF_STATE)

        # Scalar grads [64] & diag_out [8] (FP32 atomic accumulators!)
        # Layout of sc_grads:
        #   [0..10]: cont_gain, [11..21]: cont_bias, [22..32]: cont_logstd,
        #   [33]: val_gain, [34..35]: val_bias, [36]: dyn_drift_gain, [37]: dyn_act_gain,
        #   [38]: log_temp_cart, [39]: log_temp_rival, [40]: type_blend,
        #   [41..41+4*n_uniq_blk-1]: blk blend_{pool,mut,gram,ffn},
        #   [41+4*n_uniq_blk..]: step_blends[0..depth-1]
        self._sg_blk_base = 41
        self._sg_step_base = 41 + 4 * n_uniq_blk
        self.buf_sc_grads = self._add_buf(max(64, self._sg_step_base + self.depth) * 4, MG_BUF_STATE)
        self.buf_diag = self._add_buf(8 * 4, MG_BUF_STATE)

        # Weights + FP32 accumulated weight grads
        self._add_w_and_gw("enc_obs", D, 128)
        self._add_w_and_gw("enc_cart", D, 64)
        self._add_w_and_gw("enc_team", D, 64)
        self._w_bufs["type_embed"] = self._add_buf(3 * D * 2, MG_BUF_WEIGHT)
        self._gw_bufs["type_embed"] = self._add_buf(3 * D * 4, MG_BUF_STATE)
        self._gw_shapes["type_embed"] = (3, D)

        def add_i8_w_and_gw(key: str, rows: int, cols: int) -> None:
            self._add_w_and_gw(key, rows, cols)
            self._wq_bufs[key] = (self._add_buf(rows * cols, MG_BUF_STATE), self._add_buf(rows * 4, MG_BUF_STATE))
            self._repack_specs.append((key, rows, cols))

        for bi in range(n_uniq_blk):
            p = f"b{bi}."
            add_i8_w_and_gw(p + "val_proj", D, D)
            add_i8_w_and_gw(p + "global_metric", Rg, D)
            self._w_bufs[p + "pool_score"] = self._add_buf(D * 2, MG_BUF_WEIGHT)
            self._gw_bufs[p + "pool_score"] = self._add_buf(D * 4, MG_BUF_STATE)
            self._gw_shapes[p + "pool_score"] = (1, D)
            self._add_w_and_gw(p + "team_metric", Rt, D)
            self._add_w_and_gw(p + "rival_metric", Rt, D)
            self._add_w_and_gw(p + "team_val_proj", D, D)
            self._add_w_and_gw(p + "team_mutate_proj", D, D)
            self._add_w_and_gw(p + "team_bcast_proj", D, D)
            self._add_w_and_gw(p + "gram_out", D, D)
            add_i8_w_and_gw(p + "ffn_gate_up", 2 * H, D)
            add_i8_w_and_gw(p + "ffn_down", D, H)

        self._add_w_and_gw("cart_q", D, D)
        self._add_w_and_gw("cart_k", D, D)
        self._add_w_and_gw("rival_q", D, D)
        self._add_w_and_gw("rival_k", D, D)
        self._add_w_and_gw("heads_packed", 64, D)
        w = self._w_bufs
        wq = self._wq_bufs
        gw = self._gw_bufs

        # Zero-copy in-GPU weight repack graph (W_bf16 -> W_had_i8 + W_s_f32 in <0.03 ms!)
        self._g_repack = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        for key, rows, cols in self._repack_specs:
            iw = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, w[key])
            iq = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][0])
            is_ = self.lib.mesh_graph_import_buffer(self._g_repack, self.g, wq[key][1])
            op_r = _make_op(MG_OP_FWHT_PACK_I8, in_bufs=(iw,), out_bufs=(iq, is_), u=(rows, cols, 0))
            self.lib.mesh_graph_add_op(self._g_repack, ctypes.byref(op_r))
        self.lib.mesh_graph_compile(self._g_repack)

        # Shared reusable INT8 activation quantization scratch for exact rollout<->train parity
        b_q = self._add_buf(EM_pad * max(D, H), MG_BUF_STATE)
        b_s = self._add_buf(EM_pad * 4, MG_BUF_STATE)

        # Write static NormConfig
        ncfg_arr = np.array([
            self.nc.mp_norm_eps, self.nc.map_radius_min, self.nc.pos_fallback_scale, self.nc.vel_scale,
            self.nc.health_offset, self.nc.health_scale, self.nc.armor_offset, self.nc.armor_scale,
            self.nc.ammo_scale, self.nc.strength_max_s, self.nc.cart_len_scale, self.nc.cart_speed_scale,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        _copy_f32_to_ptr(ncfg_arr, self.lib.mesh_graph_buffer_ptr(self.g, self.buf_ncfg))

        eps_n = float(self.nc.mp_norm_eps)
        eps_l = float(self.nc.mp_lerp_eps)
        eps_u = float(self.nc.mp_unit_vec_eps)
        inv_silu = 1.0 / float(self.nc.mp_silu_scale)

        # Saved forward activations (clean non-aliased BF16 buffers)
        b_obs_pre = self._add_buf(EN_pad * 128 * 2, MG_BUF_STATE)
        b_cart_pre = self._add_buf(EJ_pad * 64 * 2, MG_BUF_STATE)
        b_team_pre = self._add_buf(EK_pad * 64 * 2, MG_BUF_STATE)
        b_obs_enc = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        b_cart_enc = self._add_buf(EJ_pad * D * 2, MG_BUF_STATE)
        b_team_enc = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)

        # Forward encoder
        self._add_op(_make_op(
            MG_OP_PLAN_PRENORM,
            in_bufs=(self.buf_obs, self.buf_carts, self.buf_teams, self.buf_geo, self.buf_ncfg, self.buf_ent_meta),
            out_bufs=(b_obs_pre, b_cart_pre, b_team_pre),
            u=(E, N, J, K, 128, 64, 64, 0),
        ))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_obs_pre, w["enc_obs"]), out_bufs=(b_obs_enc,), u=(EN_pad, D, 128)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_cart_pre, w["enc_cart"]), out_bufs=(b_cart_enc,), u=(EJ_pad, D, 64)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_team_pre, w["enc_team"]), out_bufs=(b_team_enc,), u=(EK_pad, D, 64)))

        x_states = [self._add_buf(EM_pad * D * 2, MG_BUF_STATE) for _ in range(self.depth + 1)]
        self._add_op(_make_op(
            MG_OP_PLAN_EMBED_CAT,
            in_bufs=(b_obs_enc, b_cart_enc, b_team_enc, w["type_embed"], self.buf_scalars),
            out_bufs=(x_states[0],),
            u=(E, N, J, K, D, 0),
            f=(eps_n, eps_l),
        ))

        # Per-step saved forward activations for depth steps
        st_acts: list[dict[str, int]] = []
        for s in range(self.depth):
            bi = 0 if self.recurrent else s
            p = f"b{bi}."
            sc0 = self._sc_blk_base + 4 * bi
            a = {
                "xn": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "v_raw": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "v_norm": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "r_raw": self._add_buf(EM_pad * Rg * 2, MG_BUF_STATE),
                "r_norm": self._add_buf(EM_pad * Rg * 2, MG_BUF_STATE),
                "global_raw": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "t_in": self._add_buf(EK_pad * D * 2, MG_BUF_STATE),
                "qt_raw": self._add_buf(EK_pad * Rt * 2, MG_BUF_STATE),
                "kt_raw": self._add_buf(EK_pad * Rt * 2, MG_BUF_STATE),
                "vt_raw": self._add_buf(EK_pad * D * 2, MG_BUF_STATE),
                "t_mixed": self._add_buf(EK_pad * D * 2, MG_BUF_STATE),
                "t_mut_raw": self._add_buf(EK_pad * D * 2, MG_BUF_STATE),
                "t_mut": self._add_buf(EK_pad * D * 2, MG_BUF_STATE),
                "tb_raw": self._add_buf(EK_pad * D * 2, MG_BUF_STATE),
                "gram_in": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "gram_out_raw": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "x_mid": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "gate_up": self._add_buf(EM_pad * 2 * H * 2, MG_BUF_STATE),
                "hid": self._add_buf(EM_pad * H * 2, MG_BUF_STATE),
                "ffn_raw": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
                "x_new": self._add_buf(EM_pad * D * 2, MG_BUF_STATE),
            }
            st_acts.append(a)

            # Exact bit-identical compressed had_int8 forward matching CompiledPlanWally (DeclaredForward!)
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(x_states[s],), out_bufs=(a["xn"], b_q, b_s), u=(M_tot, D, EM_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "val_proj"]), out_bufs=(a["v_raw"],), u=(EM_pad, D, D, 0)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "global_metric"]), out_bufs=(a["r_raw"],), u=(EM_pad, Rg, D, 0)))
            # Normalize into separate v_norm/r_norm so v_raw/r_raw stay preserved for backward!
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(a["v_raw"],), out_bufs=(a["v_norm"],), u=(M_tot, D), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_MP_NORM, in_bufs=(a["r_raw"],), out_bufs=(a["r_norm"],), u=(M_tot, Rg), f=(eps_n,)))
            self._add_op(_make_op(
                MG_OP_CGT_GRAM_STAGE1,
                in_bufs=(a["xn"], a["v_norm"], a["r_norm"], w[p + "pool_score"], self.buf_ent_meta, self.buf_scalars),
                out_bufs=(a["global_raw"], a["t_in"]),
                u=(E, N, J, K, D, Rg, sc0 + 0),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_in"], w[p + "team_metric"]), out_bufs=(a["qt_raw"],), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_in"], w[p + "rival_metric"]), out_bufs=(a["kt_raw"],), u=(EK_pad, Rt, D)))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_in"], w[p + "team_val_proj"]), out_bufs=(a["vt_raw"],), u=(EK_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_CGT_TEAM_RIVAL,
                in_bufs=(a["qt_raw"], a["kt_raw"], a["vt_raw"]),
                out_bufs=(a["t_mixed"],),
                u=(E, K, D, Rt),
                f=(eps_n,),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_mixed"], w[p + "team_mutate_proj"]), out_bufs=(a["t_mut_raw"],), u=(EK_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_MP_LERP,
                in_bufs=(a["t_in"], a["t_mut_raw"], self.buf_scalars),
                out_bufs=(a["t_mut"],),
                u=(K_tot, D, sc0 + 1, 1),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["t_mut"], w[p + "team_bcast_proj"]), out_bufs=(a["tb_raw"],), u=(EK_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_CGT_BCAST_MERGE,
                in_bufs=(a["global_raw"], a["tb_raw"], self.buf_ent_meta),
                out_bufs=(a["gram_in"],),
                u=(E, M_ent, K, D),
                f=(eps_n,),
            ))
            self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(a["gram_in"], w[p + "gram_out"]), out_bufs=(a["gram_out_raw"],), u=(EM_pad, D, D)))
            self._add_op(_make_op(
                MG_OP_MP_LERP,
                in_bufs=(a["xn"], a["gram_out_raw"], self.buf_scalars),
                out_bufs=(a["x_mid"], b_q, b_s),
                u=(M_tot, D, sc0 + 2, 1, EM_pad),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_gate_up"]), out_bufs=(a["gate_up"],), u=(EM_pad, 2 * H, D, 0)))
            self._add_op(_make_op(MG_OP_MP_SWIGLU, in_bufs=(a["gate_up"],), out_bufs=(a["hid"], b_q, b_s), u=(M_tot, H, EM_pad, 1), f=(inv_silu,)))
            self._add_op(_make_op(MG_OP_LINEAR_I8, in_bufs=(b_q, b_s, *wq[p + "ffn_down"]), out_bufs=(a["ffn_raw"],), u=(EM_pad, D, H, 0)))
            if self.tp_world_size > 1:
                cid = int(self.lib.mesh_graph_add_crossing(self.g, EM_pad * D, 2, MG_WIRE_HAD8_GATHER, 0))
                self._add_op(_make_op(MG_OP_ALLREDUCE, in_bufs=(a["ffn_raw"],), out_bufs=(a["ffn_raw"],), u=(cid, EM_pad, D)))
            if self.recurrent:
                self._add_op(_make_op(
                    MG_OP_MP_LERP,
                    in_bufs=(a["x_mid"], a["ffn_raw"], self.buf_scalars),
                    out_bufs=(a["x_new"],),
                    u=(M_tot, D, sc0 + 3, 1),
                    f=(eps_n, eps_l),
                ))
                self._add_op(_make_op(
                    MG_OP_MP_LERP,
                    in_bufs=(x_states[s], a["x_new"], self.buf_scalars),
                    out_bufs=(x_states[s + 1],),
                    u=(M_tot, D, self._sc_step_base + s, 0),
                    f=(eps_n, eps_l),
                ))
            else:
                self._add_op(_make_op(
                    MG_OP_MP_LERP,
                    in_bufs=(a["x_mid"], a["ffn_raw"], self.buf_scalars),
                    out_bufs=(x_states[s + 1],),
                    u=(M_tot, D, sc0 + 3, 1),
                    f=(eps_n, eps_l),
                ))

        # Forward final slice & heads
        b_ph = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        b_ch = self._add_buf(EJ_pad * D * 2, MG_BUF_STATE)
        b_th = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        b_cq = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        b_ck = self._add_buf(EJ_pad * D * 2, MG_BUF_STATE)
        b_rq = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        b_rk = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        b_hr = self._add_buf(EN_pad * 64 * 2, MG_BUF_STATE)
        b_clg = self._add_buf(EN_tot * J * 2, MG_BUF_STATE)
        b_rlg = self._add_buf(EN_tot * K * 2, MG_BUF_STATE)

        self._add_op(_make_op(MG_OP_PLAN_SLICE_NORM, in_bufs=(x_states[self.depth],), out_bufs=(b_ph, b_ch, b_th), u=(E, N, J, K, D), f=(eps_n,)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ph, w["cart_q"]), out_bufs=(b_cq,), u=(EN_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ch, w["cart_k"]), out_bufs=(b_ck,), u=(EJ_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ph, w["rival_q"]), out_bufs=(b_rq,), u=(EN_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_th, w["rival_k"]), out_bufs=(b_rk,), u=(EK_pad, D, D)))
        self._add_op(_make_op(MG_OP_LINEAR, in_bufs=(b_ph, w["heads_packed"]), out_bufs=(b_hr,), u=(EN_pad, 64, D)))
        self._add_op(_make_op(MG_OP_BILINEAR_LOGITS, in_bufs=(b_cq, b_ck, self.buf_ent_meta, self.buf_scalars), out_bufs=(b_clg,), u=(E, N, J, D, 1, 0, M_ent), f=(eps_u, float(self.mc.rival_own_mask_fill))))
        self._add_op(_make_op(MG_OP_BILINEAR_LOGITS, in_bufs=(b_rq, b_rk, self.buf_ent_meta, self.buf_scalars), out_bufs=(b_rlg,), u=(E, N, K, D, 2, 1 if K >= 2 else 0, M_ent), f=(eps_u, float(self.mc.rival_own_mask_fill))))

        # ======================== BACKWARD PASS ========================
        # Reusable backward gradient buffers (all zero-initialized so padded rows stay 0!)
        bd_hr = self._add_buf(EN_pad * 64 * 2, MG_BUF_STATE)
        bd_clg = self._add_buf(EN_tot * J * 2, MG_BUF_STATE)
        bd_rlg = self._add_buf(EN_tot * K * 2, MG_BUF_STATE)
        bd_cq = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        bd_ck = self._add_buf(EJ_pad * D * 2, MG_BUF_STATE)
        bd_rq = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        bd_rk = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_ph = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        bd_ch = self._add_buf(EJ_pad * D * 2, MG_BUF_STATE)
        bd_th = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)

        bd_x = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_x_prev = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_x_new = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_x_mid = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_ffn_raw = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_hid = self._add_buf(EM_pad * H * 2, MG_BUF_STATE)
        bd_gu = self._add_buf(EM_pad * 2 * H * 2, MG_BUF_STATE)
        bd_xn = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_gram_out = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_gram_in = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_glob = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_tb = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_tmut = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_tmut_raw = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_tin = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_tmixed = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_qt = self._add_buf(EK_pad * Rt * 2, MG_BUF_STATE)
        bd_kt = self._add_buf(EK_pad * Rt * 2, MG_BUF_STATE)
        bd_vt = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        bd_vnorm = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_rnorm = self._add_buf(EM_pad * Rg * 2, MG_BUF_STATE)
        bd_vraw = self._add_buf(EM_pad * D * 2, MG_BUF_STATE)
        bd_rraw = self._add_buf(EM_pad * Rg * 2, MG_BUF_STATE)

        # 1. Loss + head backward
        self._add_op(_make_op(
            MG_OP_PLAN_LOSS_BWD,
            in_bufs=(b_hr, b_clg, b_rlg, self.buf_hp, self.buf_targ_f, self.buf_targ_i),
            out_bufs=(bd_hr, bd_clg, bd_rlg),
            u=(EN_tot, J, K, EN_pad, self.buf_lcfg, self.buf_sc_grads, self.buf_diag),
        ))

        # 2. Bilinear logits backward
        t_lo, t_hi = float(self.mc.temp_bounds[0]), float(self.mc.temp_bounds[1])
        self._add_op(_make_op(
            MG_OP_BILINEAR_BWD,
            in_bufs=(b_cq, b_ck, bd_clg, self.buf_ent_meta, self.buf_scalars, self.buf_sc_grads),
            out_bufs=(bd_cq, bd_ck),
            u=(E, N, J, D, 1, 0, M_ent, 38),
            f=(eps_u, t_lo, t_hi, float(EN_pad)),
        ))
        self._add_op(_make_op(
            MG_OP_BILINEAR_BWD,
            in_bufs=(b_rq, b_rk, bd_rlg, self.buf_ent_meta, self.buf_scalars, self.buf_sc_grads),
            out_bufs=(bd_rq, bd_rk),
            u=(E, N, K, D, 2, 1 if K >= 2 else 0, M_ent, 39),
            f=(eps_u, t_lo, t_hi, float(EN_pad)),
        ))

        # 3. Head linear dW & dX
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_hr, b_ph), out_bufs=(gw["heads_packed"],), u=(64, D, EN_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_cq, b_ph), out_bufs=(gw["cart_q"],), u=(D, D, EN_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_ck, b_ch), out_bufs=(gw["cart_k"],), u=(D, D, EJ_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_rq, b_ph), out_bufs=(gw["rival_q"],), u=(D, D, EN_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_rk, b_th), out_bufs=(gw["rival_k"],), u=(D, D, EK_pad, 1)))

        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_hr, w["heads_packed"]), out_bufs=(bd_ph,), u=(EN_pad, D, 64, 0)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_cq, w["cart_q"]), out_bufs=(bd_ph,), u=(EN_pad, D, D, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_rq, w["rival_q"]), out_bufs=(bd_ph,), u=(EN_pad, D, D, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_ck, w["cart_k"]), out_bufs=(bd_ch,), u=(EJ_pad, D, D, 0)))
        self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_rk, w["rival_k"]), out_bufs=(bd_th,), u=(EK_pad, D, D, 0)))

        # 4. Slice norm backward -> bd_x
        self._add_op(_make_op(
            MG_OP_PLAN_SLICE_BWD,
            in_bufs=(x_states[self.depth], bd_ph, bd_ch, bd_th),
            out_bufs=(bd_x,),
            u=(E, N, J, K, D, EM_pad),
            f=(eps_n,),
        ))

        # 5. Reverse loop through depth steps
        cur_dx = bd_x
        nxt_dx = bd_x_prev
        for s in range(self.depth - 1, -1, -1):
            bi = 0 if self.recurrent else s
            p = f"b{bi}."
            sc0 = self._sc_blk_base + 4 * bi
            sg0 = self._sg_blk_base + 4 * bi
            a = st_acts[s]

            if self.recurrent:
                self._add_op(_make_op(
                    MG_OP_MP_LERP_BWD,
                    in_bufs=(x_states[s], a["x_new"], cur_dx, self.buf_scalars),
                    out_bufs=(nxt_dx, bd_x_new, self.buf_sc_grads),
                    u=(M_tot, D, self._sc_step_base + s, 0, self._sg_step_base + s, 0, EM_pad),
                    f=(eps_n, eps_l),
                ))
                dx_blk_out = bd_x_new
            else:
                dx_blk_out = cur_dx

            # FFN residual lerp bwd -> bd_x_mid, bd_ffn_raw
            self._add_op(_make_op(
                MG_OP_MP_LERP_BWD,
                in_bufs=(a["x_mid"], a["ffn_raw"], dx_blk_out, self.buf_scalars),
                out_bufs=(bd_x_mid, bd_ffn_raw, self.buf_sc_grads),
                u=(M_tot, D, sc0 + 3, 1, sg0 + 3, 0, EM_pad),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_ffn_raw, a["hid"]), out_bufs=(gw[p + "ffn_down"],), u=(D, H, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_ffn_raw, w[p + "ffn_down"]), out_bufs=(bd_hid,), u=(EM_pad, H, D, 0)))
            self._add_op(_make_op(MG_OP_MP_SWIGLU_BWD, in_bufs=(a["gate_up"], bd_hid), out_bufs=(bd_gu,), u=(M_tot, H, EM_pad), f=(inv_silu,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_gu, a["x_mid"]), out_bufs=(gw[p + "ffn_gate_up"],), u=(2 * H, D, EM_pad, 1)))
            if self.tp_world_size > 1:
                self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_gu, w[p + "ffn_gate_up"]), out_bufs=(bd_ffn_raw,), u=(EM_pad, D, 2 * H, 0)))
                cid_b = int(self.lib.mesh_graph_add_crossing(self.g, EM_pad * D, 2, MG_WIRE_HAD8_GATHER, 0))
                self._add_op(_make_op(MG_OP_ALLREDUCE, in_bufs=(bd_ffn_raw,), out_bufs=(bd_ffn_raw,), u=(cid_b, EM_pad, D)))
                self._add_op(_make_op(MG_OP_EW, in_bufs=(bd_x_mid, bd_ffn_raw), out_bufs=(bd_x_mid,), u=(EM_pad * D, MG_EW_ADD)))
            else:
                self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_gu, w[p + "ffn_gate_up"]), out_bufs=(bd_x_mid,), u=(EM_pad, D, 2 * H, 1)))

            # Gram residual lerp bwd -> bd_xn, bd_gram_out
            self._add_op(_make_op(
                MG_OP_MP_LERP_BWD,
                in_bufs=(a["xn"], a["gram_out_raw"], bd_x_mid, self.buf_scalars),
                out_bufs=(bd_xn, bd_gram_out, self.buf_sc_grads),
                u=(M_tot, D, sc0 + 2, 1, sg0 + 2, 0, EM_pad),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_gram_out, a["gram_in"]), out_bufs=(gw[p + "gram_out"],), u=(D, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_gram_out, w[p + "gram_out"]), out_bufs=(bd_gram_in,), u=(EM_pad, D, D, 0)))

            # Broadcast merge bwd -> bd_glob, bd_tb
            self._add_op(_make_op(
                MG_OP_CGT_BCAST_BWD,
                in_bufs=(a["global_raw"], a["tb_raw"], bd_gram_in, self.buf_ent_meta),
                out_bufs=(bd_glob, bd_tb),
                u=(E, M_ent, K, D, EM_pad, EK_pad),
                f=(eps_n,),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_tb, a["t_mut"]), out_bufs=(gw[p + "team_bcast_proj"],), u=(D, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_tb, w[p + "team_bcast_proj"]), out_bufs=(bd_tmut,), u=(EK_pad, D, D, 0)))

            # Team mutate lerp bwd -> bd_tin, bd_tmut_raw
            self._add_op(_make_op(
                MG_OP_MP_LERP_BWD,
                in_bufs=(a["t_in"], a["t_mut_raw"], bd_tmut, self.buf_scalars),
                out_bufs=(bd_tin, bd_tmut_raw, self.buf_sc_grads),
                u=(K_tot, D, sc0 + 1, 1, sg0 + 1, 0, EK_pad),
                f=(eps_n, eps_l),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_tmut_raw, a["t_mixed"]), out_bufs=(gw[p + "team_mutate_proj"],), u=(D, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_tmut_raw, w[p + "team_mutate_proj"]), out_bufs=(bd_tmixed,), u=(EK_pad, D, D, 0)))

            # Team rival Gram bwd -> bd_qt, bd_kt, bd_vt
            self._add_op(_make_op(
                MG_OP_CGT_RIVAL_BWD,
                in_bufs=(a["qt_raw"], a["kt_raw"], a["vt_raw"], bd_tmixed),
                out_bufs=(bd_qt, bd_kt, bd_vt),
                u=(E, K, D, Rt),
                f=(eps_n,),
            ))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_qt, a["t_in"]), out_bufs=(gw[p + "team_metric"],), u=(Rt, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_kt, a["t_in"]), out_bufs=(gw[p + "rival_metric"],), u=(Rt, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_vt, a["t_in"]), out_bufs=(gw[p + "team_val_proj"],), u=(D, D, EK_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_qt, w[p + "team_metric"]), out_bufs=(bd_tin,), u=(EK_pad, D, Rt, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_kt, w[p + "rival_metric"]), out_bufs=(bd_tin,), u=(EK_pad, D, Rt, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_vt, w[p + "team_val_proj"]), out_bufs=(bd_tin,), u=(EK_pad, D, D, 1)))

            # Stage 1 bwd: team_pool + global_gram -> bd_vnorm, bd_rnorm, bd_xn (+=), d_w_pool (+=), d_blend_pool (+=)
            self._add_op(_make_op(
                MG_OP_CGT_STAGE1_BWD,
                in_bufs=(a["xn"], a["v_norm"], a["r_norm"], bd_tin, bd_glob, self.buf_ent_meta),
                out_bufs=(bd_vnorm, bd_rnorm, bd_xn),
                u=(
                    E | (N << 16), J | (K << 16), D | (Rg << 16), (sc0 + 0) | ((sg0 + 0) << 16),
                    w[p + "pool_score"], self.buf_scalars, gw[p + "pool_score"], self.buf_sc_grads,
                ),
                f=(eps_n, eps_l),
            ))
            # Backprop through mp_norm(v_raw) & mp_norm(r_raw)
            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(a["v_raw"], bd_vnorm), out_bufs=(bd_vraw,), u=(M_tot, D, 0, EM_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_MP_NORM_BWD, in_bufs=(a["r_raw"], bd_rnorm), out_bufs=(bd_rraw,), u=(M_tot, Rg, 0, EM_pad), f=(eps_n,)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_vraw, a["xn"]), out_bufs=(gw[p + "val_proj"],), u=(D, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_rraw, a["xn"]), out_bufs=(gw[p + "global_metric"],), u=(Rg, D, EM_pad, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_vraw, w[p + "val_proj"]), out_bufs=(bd_xn,), u=(EM_pad, D, D, 1)))
            self._add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(bd_rraw, w[p + "global_metric"]), out_bufs=(bd_xn,), u=(EM_pad, D, Rg, 1)))

            # Backprop through xn = mp_norm(x_in[s]) -> accum into nxt_dx!
            self._add_op(_make_op(
                MG_OP_MP_NORM_BWD,
                in_bufs=(x_states[s], bd_xn),
                out_bufs=(nxt_dx,),
                u=(M_tot, D, 1 if self.recurrent else 0, EM_pad),
                f=(eps_n,),
            ))
            cur_dx, nxt_dx = nxt_dx, cur_dx

        # 6. Encoder backward -> d_obs_enc, d_cart_enc, d_team_enc, d_type_embed, d_type_blend
        bd_obs_enc = self._add_buf(EN_pad * D * 2, MG_BUF_STATE)
        bd_cart_enc = self._add_buf(EJ_pad * D * 2, MG_BUF_STATE)
        bd_team_enc = self._add_buf(EK_pad * D * 2, MG_BUF_STATE)
        self._add_op(_make_op(
            MG_OP_PLAN_EMBED_BWD,
            in_bufs=(b_obs_enc, b_cart_enc, b_team_enc, w["type_embed"], cur_dx, self.buf_scalars),
            out_bufs=(bd_obs_enc, bd_cart_enc, bd_team_enc),
            u=(E | (N << 16), J | (K << 16), D, 0 | (40 << 16), gw["type_embed"], self.buf_sc_grads),
            f=(eps_n, eps_l),
        ))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_obs_enc, b_obs_pre), out_bufs=(gw["enc_obs"],), u=(D, 128, EN_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_cart_enc, b_cart_pre), out_bufs=(gw["enc_cart"],), u=(D, 64, EJ_pad, 1)))
        self._add_op(_make_op(MG_OP_LINEAR_DW, in_bufs=(bd_team_enc, b_team_pre), out_bufs=(gw["enc_team"],), u=(D, 64, EK_pad, 1)))

        # Host zero-copy views
        self.obs_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_obs), (E, N, OBS_WIDTH))
        self.carts_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_carts), (E, J, CART_WIDTH))
        self.teams_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_teams), (E, K, TEAM_WIDTH))
        self.geo_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_geo), (4,))
        self.targ_f_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_targ_f), (E, N, 17))
        self.targ_i_view = _ptr_to_torch_i32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_targ_i), (E, N, 2))
        self.lcfg_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_lcfg), (16,))
        self.sc_grads_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_sc_grads), (max(64, self._sg_step_base + self.depth),))
        self.diag_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.buf_diag), (8,))
        self._gw_views = {
            k: _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bid), self._gw_shapes[k])
            for k, bid in self._gw_bufs.items()
        }

    def zero_grads_(self) -> None:
        self.sc_grads_view.zero_()
        self.diag_view.zero_()
        for v in self._gw_views.values():
            v.zero_()

    @torch.no_grad()
    def sync_weights_(self, model: PlanWally) -> None:
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
            wd = _unit_weight_bf16(blk.ffn_down.weight, eps)
            if self.tp_world_size > 1:
                lo = self.tp_rank * self.h_local
                hi = lo + self.h_local
                wg = wg[lo:hi].contiguous()
                wu = wu[lo:hi].contiguous()
                wd = wd[:, lo:hi].contiguous()
            put(p + "ffn_gate_up", torch.cat((wg, wu), dim=0))
            put(p + "ffn_down", wd)
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

        hp_w = torch.zeros((64, self.d), dtype=torch.bfloat16)
        hp_w[0:11] = _unit_weight_bf16(model.cont_proj.weight, eps)
        hp_w[11:13] = _unit_weight_bf16(model.value_proj.weight, eps)
        hp_w[13:14] = _unit_weight_bf16(model.dyn_drift_proj.weight, eps)
        hp_w[14:25] = _unit_weight_bf16(model.dyn_action_proj.weight, eps)
        put("heads_packed", hp_w)

        hp = np.zeros((48,), dtype=np.float32)
        hp[0:11] = torch.clamp(model.cont_gain.detach().float(), -mc.cont_gain_bound, mc.cont_gain_bound).cpu().numpy()
        hp[11:22] = model.cont_bias.detach().float().cpu().numpy()
        hp[22:33] = torch.clamp(model.cont_logstd.detach().float(), mc.cont_logstd_bounds[0], mc.cont_logstd_bounds[1]).cpu().numpy()
        hp[33] = float(torch.clamp(model.value_gain.detach().float(), mc.value_gain_bounds[0], mc.value_gain_bounds[1]).item())
        hp[34:36] = model.value_bias.detach().float().cpu().numpy()
        hp[36] = float(torch.clamp(model.dyn_drift_gain.detach().float(), mc.dyn_gain_bounds[0], mc.dyn_gain_bounds[1]).item())
        hp[37] = float(torch.clamp(model.dyn_action_gain.detach().float(), mc.dyn_gain_bounds[0], mc.dyn_gain_bounds[1]).item())
        hp[38] = float(mc.cont_mean_scale)
        hp[42] = float(mc.cont_gain_bound)
        hp[43] = float(mc.cont_logstd_bounds[0])
        hp[44] = float(mc.cont_logstd_bounds[1])
        hp[45] = float(mc.value_gain_bounds[0])
        _copy_f32_to_ptr(hp, self.lib.mesh_graph_buffer_ptr(self.g, self.buf_hp))
        if self._g_repack:
            wt = ctypes.c_double(0.0)
            self.lib.mesh_graph_run_fused(self._g_repack, 1, 0, ctypes.byref(wt))

    def set_geo(self, geo: dict[str, torch.Tensor] | None) -> None:
        if geo is not None and "map_center" in geo and "map_radius" in geo:
            self.geo_view[0:3].copy_(geo["map_center"].detach().to(device="cpu", dtype=torch.float32).view(3))
            self.geo_view[3] = float(geo["map_radius"].detach().to(device="cpu", dtype=torch.float32).item())
        else:
            self.geo_view.zero_()

    @torch.no_grad()
    def microbatch_fwd_bwd(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        cont: torch.Tensor,
        cart_pick: torch.Tensor,
        rival_pick: torch.Tensor,
        beh_logp: torch.Tensor,
        adv: torch.Tensor,
        gae_ret: torch.Tensor,
        obs_drho: torch.Tensor,
        actor_mask: torch.Tensor,
        wm_in: torch.Tensor,
        *,
        lcfg: Any,
        entropy_coef: float,
        grad_scale: float = 1.0,
    ) -> dict[str, float]:
        """Run one fused Metal 4 NAX forward + loss + backward microbatch accumulating FP32 grads."""
        B_cur = int(obs.shape[0])
        assert B_cur <= self.E
        if B_cur == self.E:
            self.obs_view.copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
            self.carts_view.copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
            self.teams_view.copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
            tf = self.targ_f_view
            ti = self.targ_i_view
            tf[:, :, 0:11].copy_(cont.detach().to(device="cpu", dtype=torch.float32))
            tf[:, :, 11].copy_(beh_logp.detach().to(device="cpu", dtype=torch.float32))
            tf[:, :, 12].copy_(adv.detach().to(device="cpu", dtype=torch.float32))
            tf[:, :, 13].copy_(gae_ret.detach().to(device="cpu", dtype=torch.float32))
            tf[:, :, 14].copy_(obs_drho.detach().to(device="cpu", dtype=torch.float32))
            am_cpu = actor_mask.detach().to(device="cpu", dtype=torch.float32)
            wm_cpu = wm_in.detach().to(device="cpu", dtype=torch.float32)
            tf[:, :, 15].copy_(am_cpu)
            tf[:, :, 16].copy_(wm_cpu)
            ti[:, :, 0].copy_(cart_pick.detach().to(device="cpu", dtype=torch.int32))
            ti[:, :, 1].copy_(rival_pick.detach().to(device="cpu", dtype=torch.int32))
            am_sum = max(float(am_cpu.sum().item()), 1.0)
            wm_sum = max(float(wm_cpu.sum().item()), 1.0)
            lm_sum = max(float((1.0 - wm_cpu).sum().item()), 1.0)
            inv_en = 1.0 / float(self.E * self.N)
        else:
            # Partial tail batch: zero masks on trailing padded envs so they contribute 0 loss/grad
            self.obs_view.zero_(); self.carts_view.zero_(); self.teams_view.zero_()
            self.obs_view[:B_cur].copy_(obs.detach().to(device="cpu", dtype=torch.bfloat16))
            self.carts_view[:B_cur].copy_(carts.detach().to(device="cpu", dtype=torch.bfloat16))
            self.teams_view[:B_cur].copy_(teams.detach().to(device="cpu", dtype=torch.bfloat16))
            tf = self.targ_f_view; ti = self.targ_i_view
            tf.zero_(); ti.zero_()
            tf[B_cur:, :, 16] = -1.0
            tf[:B_cur, :, 0:11].copy_(cont.detach().to(device="cpu", dtype=torch.float32))
            tf[:B_cur, :, 11].copy_(beh_logp.detach().to(device="cpu", dtype=torch.float32))
            tf[:B_cur, :, 12].copy_(adv.detach().to(device="cpu", dtype=torch.float32))
            tf[:B_cur, :, 13].copy_(gae_ret.detach().to(device="cpu", dtype=torch.float32))
            tf[:B_cur, :, 14].copy_(obs_drho.detach().to(device="cpu", dtype=torch.float32))
            am_cpu = actor_mask.detach().to(device="cpu", dtype=torch.float32)
            wm_cpu = wm_in.detach().to(device="cpu", dtype=torch.float32)
            tf[:B_cur, :, 15].copy_(am_cpu)
            tf[:B_cur, :, 16].copy_(wm_cpu)
            ti[:B_cur, :, 0].copy_(cart_pick.detach().to(device="cpu", dtype=torch.int32))
            ti[:B_cur, :, 1].copy_(rival_pick.detach().to(device="cpu", dtype=torch.int32))
            am_sum = max(float(am_cpu.sum().item()), 1.0)
            wm_sum = max(float(wm_cpu.sum().item()), 1.0)
            lm_sum = max(float((1.0 - wm_cpu).sum().item()), 1.0)
            inv_en = 1.0 / float(B_cur * self.N)

        self.diag_view.zero_()
        mc = self.mc
        lc = np.array([
            float(lcfg.actor_weight), float(lcfg.critic_weight), float(lcfg.dyn_weight), float(entropy_coef),
            float(lcfg.ratio_clip), float(lcfg.log_ratio_bound), float(lcfg.critic_huber_beta), float(lcfg.dyn_huber_beta),
            1.0 / am_sum, 1.0 / wm_sum, 1.0 / lm_sum, inv_en,
            float(grad_scale), float(mc.value_gain_bounds[1]), float(mc.dyn_gain_bounds[0]), float(mc.dyn_gain_bounds[1]),
        ], dtype=np.float32)
        self.lcfg_view.copy_(torch.from_numpy(lc))

        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run_fused(self.g, 1, 0, ctypes.byref(wall))
        d = self.diag_view.tolist()
        return {
            "loss_actor": d[0],
            "loss_w": d[1],
            "loss_l": d[2],
            "loss_dyn": d[3],
            "loss_total": d[0] + d[1] + d[2] + d[3],
            "entropy_disc": d[4],
            "entropy": d[5],
            "clip_frac": d[6],
            "approx_kl": d[7],
        }

    @torch.no_grad()
    def export_grads_to_model_(self, model: PlanWally) -> None:
        """Apply exact unit-norm tangent projection to accumulated FP32 grads and write model.parameters().grad."""
        eps_w = max(float(self.nc.unit_norm_eps), 1e-4)
        eps_n = max(float(self.nc.mp_norm_eps), 1e-4)
        dev = model.enc_obs.weight.device
        dt = model.enc_obs.weight.dtype

        def set_unit_grad(param: torch.nn.Parameter, g_wu_full: torch.Tensor) -> None:
            k_in = param.shape[1]
            g_wu = g_wu_full[:, :k_in]
            w_f = param.detach().to(device="cpu", dtype=torch.float32)
            nrm = w_f.norm(dim=-1, keepdim=True)
            inv = 1.0 / nrm.clamp_min(eps_w)
            wu = w_f * inv
            proj = torch.where(nrm >= eps_w, (g_wu * wu).sum(dim=-1, keepdim=True), torch.zeros_like(nrm))
            g_w = (g_wu - proj * wu) * inv
            param.grad = g_w.to(device=dev, dtype=dt)

        gw = self._gw_views
        set_unit_grad(model.enc_obs.weight, gw["enc_obs"])
        set_unit_grad(model.enc_cart.weight, gw["enc_cart"])
        set_unit_grad(model.enc_team.weight, gw["enc_team"])

        # type_embed: backprop through mp_norm(type_embed[0..2])
        te_w = model.type_embed.detach().to(device="cpu", dtype=torch.float32)
        g_te_n = gw["type_embed"]
        inv_te = torch.rsqrt(te_w[:3].square().mean(dim=-1, keepdim=True) + eps_n)
        y_te = te_w[:3] * inv_te
        g_te = torch.zeros_like(te_w)
        g_te[:3] = inv_te * (g_te_n - (g_te_n * y_te).mean(dim=-1, keepdim=True) * y_te)
        model.type_embed.grad = g_te.to(device=dev, dtype=dt)

        sg = self.sc_grads_view
        model.type_blend.grad = sg[40].to(device=dev, dtype=dt)
        model.log_temp_cart.grad = sg[38].to(device=dev, dtype=dt)
        model.log_temp_rival.grad = sg[39].to(device=dev, dtype=dt)

        n_uniq_blk = 1 if self.recurrent else self.depth
        for bi in range(n_uniq_blk):
            blk = model.blocks[bi]
            p = f"b{bi}."
            sg0 = self._sg_blk_base + 4 * bi
            set_unit_grad(blk.val_proj.weight, gw[p + "val_proj"])
            set_unit_grad(blk.global_metric.weight, gw[p + "global_metric"])
            set_unit_grad(blk.pool_score.weight, gw[p + "pool_score"])
            set_unit_grad(blk.team_metric.weight, gw[p + "team_metric"])
            set_unit_grad(blk.rival_metric.weight, gw[p + "rival_metric"])
            set_unit_grad(blk.team_val_proj.weight, gw[p + "team_val_proj"])
            set_unit_grad(blk.team_mutate_proj.weight, gw[p + "team_mutate_proj"])
            set_unit_grad(blk.team_bcast_proj.weight, gw[p + "team_bcast_proj"])
            set_unit_grad(blk.gram_out.weight, gw[p + "gram_out"])
            g_gu = gw[p + "ffn_gate_up"]
            set_unit_grad(blk.ffn_gate.weight, g_gu[: self.h])
            set_unit_grad(blk.ffn_up.weight, g_gu[self.h : 2 * self.h])
            set_unit_grad(blk.ffn_down.weight, gw[p + "ffn_down"])
            blk.blend_pool.grad = sg[sg0 + 0].to(device=dev, dtype=dt)
            blk.blend_team_mut.grad = sg[sg0 + 1].to(device=dev, dtype=dt)
            blk.blend_gram.grad = sg[sg0 + 2].to(device=dev, dtype=dt)
            blk.blend_ffn.grad = sg[sg0 + 3].to(device=dev, dtype=dt)
        if self.recurrent:
            model.step_blends.grad = sg[self._sg_step_base : self._sg_step_base + self.depth].to(device=dev, dtype=dt)

        set_unit_grad(model.cart_q.weight, gw["cart_q"])
        set_unit_grad(model.cart_k.weight, gw["cart_k"])
        set_unit_grad(model.rival_q.weight, gw["rival_q"])
        set_unit_grad(model.rival_k.weight, gw["rival_k"])

        g_hp = gw["heads_packed"]
        set_unit_grad(model.cont_proj.weight, g_hp[0:11])
        set_unit_grad(model.value_proj.weight, g_hp[11:13])
        set_unit_grad(model.dyn_drift_proj.weight, g_hp[13:14])
        set_unit_grad(model.dyn_action_proj.weight, g_hp[14:25])

        model.cont_gain.grad = sg[0:11].to(device=dev, dtype=dt)
        model.cont_bias.grad = sg[11:22].to(device=dev, dtype=dt)
        model.cont_logstd.grad = sg[22:33].to(device=dev, dtype=dt)
        model.value_gain.grad = sg[33].to(device=dev, dtype=dt)
        model.value_bias.grad = sg[34:36].to(device=dev, dtype=dt)
        model.dyn_drift_gain.grad = sg[36].to(device=dev, dtype=dt)
        model.dyn_action_gain.grad = sg[37].to(device=dev, dtype=dt)


# ============================================================================
# 5. MetalCGTRewardComputer — Fused In-GPU CGT Step Reward (MG_OP_SIMDT_REWARD)
# ============================================================================

class MetalCGTRewardComputer:
    """Fused in-GPU CGT step reward + Winnie/Lou role masks (<0.05 ms for E=256)."""

    def __init__(self, E: int, N: int, J: int, K: int):
        self.lib = _load_mesh_graph_lib()
        self.E = int(E)
        self.N = int(N)
        self.J = int(J)
        self.K = int(K)
        self.g = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for MetalCGTRewardComputer")

        def add_b(nbytes: int) -> int:
            return int(self.lib.mesh_graph_add_buffer(self.g, max(64, int(nbytes)), MG_BUF_STATE, None, -1, 0))

        self.b_oi = add_b(self.E * self.N * OBS_WIDTH * 2)
        self.b_ci = add_b(self.E * self.J * CART_WIDTH * 2)
        self.b_ti = add_b(self.E * self.K * TEAM_WIDTH * 2)
        self.b_oo = add_b(self.E * self.N * OBS_WIDTH * 2)
        self.b_co = add_b(self.E * self.J * CART_WIDTH * 2)
        self.b_to = add_b(self.E * self.K * TEAM_WIDTH * 2)
        self.b_ex = add_b(self.E * 3 * 4)
        self.b_rc = add_b(16 * 4)
        self.b_rew = add_b(self.E * self.N * 4)
        self.b_wmi = add_b(self.E * self.N)
        self.b_wmo = add_b(self.E * self.N)

        op = _make_op(
            MG_OP_SIMDT_REWARD,
            in_bufs=(self.b_oi, self.b_ci, self.b_ti, self.b_oo, self.b_co, self.b_to),
            out_bufs=(self.b_rew, self.b_wmi, self.b_wmo),
            u=(self.E, self.N, self.J, self.K, self.b_ex, self.b_rc),
        )
        self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))
        self.lib.mesh_graph_compile(self.g)

        self.oi_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_oi), (self.E, self.N, OBS_WIDTH))
        self.ci_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_ci), (self.E, self.J, CART_WIDTH))
        self.ti_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_ti), (self.E, self.K, TEAM_WIDTH))
        self.oo_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_oo), (self.E, self.N, OBS_WIDTH))
        self.co_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_co), (self.E, self.J, CART_WIDTH))
        self.to_view = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_to), (self.E, self.K, TEAM_WIDTH))
        self.ex_view = _ptr_to_torch_i32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_ex), (self.E, 3))
        self.rc_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_rc), (16,))
        self.rew_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_rew), (self.E, self.N))
        self.wmi_u8 = _ptr_to_u8_array(self.lib.mesh_graph_buffer_ptr(self.g, self.b_wmi), self.E * self.N).reshape(self.E, self.N)
        self.wmo_u8 = _ptr_to_u8_array(self.lib.mesh_graph_buffer_ptr(self.g, self.b_wmo), self.E * self.N).reshape(self.E, self.N)

    def close(self) -> None:
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    def __del__(self) -> None:
        self.close()

    def set_config(self, rcfg: Any, terminal_win_bonus: float = 1.5) -> None:
        arr = np.array([
            float(rcfg.cart_partial_cp_weight),
            float(rcfg.winner_barrier_margin),
            float(rcfg.winner_barrier_weight),
            float(rcfg.potential_shaping_weight),
            float(rcfg.potential_shaping_gamma),
            float(rcfg.loser_potential_self_weight),
            float(rcfg.lane_progress_weight),
            float(rcfg.rival_lane_penalty),
            1.0 if bool(rcfg.normalize_lane_by_scale) else 0.0,
            float(rcfg.push_weight),
            float(rcfg.push_scale),
            float(rcfg.contest_weight),
            float(rcfg.contest_scale),
            float(rcfg.pickup_weight),
            float(rcfg.pickup_scale),
            float(terminal_win_bonus),
        ], dtype=np.float32)
        self.rc_view.copy_(torch.from_numpy(arr))

    @torch.no_grad()
    def compute(
        self,
        obs_in: torch.Tensor,
        carts_in: torch.Tensor,
        teams_in: torch.Tensor,
        obs_out: torch.Tensor,
        carts_out: torch.Tensor,
        teams_out: torch.Tensor,
        was_finished: torch.Tensor,
        now_finished: torch.Tensor,
        winner_team_1based: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.oi_view.copy_(obs_in.detach().to(device="cpu", dtype=torch.bfloat16))
        self.ci_view.copy_(carts_in.detach().to(device="cpu", dtype=torch.bfloat16))
        self.ti_view.copy_(teams_in.detach().to(device="cpu", dtype=torch.bfloat16))
        self.oo_view.copy_(obs_out.detach().to(device="cpu", dtype=torch.bfloat16))
        self.co_view.copy_(carts_out.detach().to(device="cpu", dtype=torch.bfloat16))
        self.to_view.copy_(teams_out.detach().to(device="cpu", dtype=torch.bfloat16))
        self.ex_view[:, 0].copy_(was_finished.detach().to(device="cpu", dtype=torch.int32))
        self.ex_view[:, 1].copy_(now_finished.detach().to(device="cpu", dtype=torch.int32))
        self.ex_view[:, 2].copy_(winner_team_1based.detach().to(device="cpu", dtype=torch.int32))
        wt = ctypes.c_double(0.0)
        self.lib.mesh_graph_run_fused(self.g, 1, 0, ctypes.byref(wt))
        return (
            self.rew_view.clone(),
            torch.from_numpy(self.wmi_u8.copy() != 0),
            torch.from_numpy(self.wmo_u8.copy() != 0),
        )


# ============================================================================
# 6. CompiledAuroraOptimizer — Single-CB Metal 4 NAX Aurora + Clip + EMA + Repack
# ============================================================================

@dataclass
class CompiledAuroraState:
    """Persistent FP32 optimizer state (momentum, Adam v, step) preserved across iterations."""
    mom: dict[str, np.ndarray]
    v: dict[str, np.ndarray]
    step: int = 0

    @classmethod
    def empty(cls) -> "CompiledAuroraState":
        return cls(mom={}, v={}, step=0)


class CompiledAuroraOptimizer:
    """
    Compiles the ENTIRE Aurora spectral optimizer step into ONE single Metal 4 command buffer:
      1. Exact unit-norm / mp_norm tangent projection + global L2 gradient norm^2
      2. Global L2 grad-clip + Nesterov momentum
      3. 5-step Metal 4 NAX Newton-Schulz + Aurora tall-matrix row-leverage D refinement
      4. EDM2 unit-norm projection + EMA update + live BF16 pack into CompiledPlanTrainer
      5. Scalar + 1D UnitLinear Adam update + live scalars/hp pack + zero_grads
      6. In-place had_int8 Rademacher-FWHT-64 weight repack into CompiledPlanTrainer!
    Zero PyTorch MPS calls and zero CPU<->MPS sync stalls inside the optimization loop!
    """

    def __init__(
        self,
        trainer: CompiledPlanTrainer,
        model: PlanWally,
        ema_model: PlanWally,
        opt_cfg: Any,
        state: CompiledAuroraState | None = None,
    ):
        self.lib = trainer.lib
        self.trainer = trainer
        self.opt_cfg = opt_cfg
        self.state = state if state is not None else CompiledAuroraState.empty()
        self.g = self.lib.mesh_graph_create(1, 0, b"", MG_WIRE_AUTO)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for CompiledAuroraOptimizer")

        D, H = trainer.d, trainer.h
        n_uniq_blk = 1 if trainer.recurrent else trainer.depth
        S_tot = 41 + 4 * n_uniq_blk + (trainer.depth if trainer.recurrent else 0)
        self._n_uniq_blk = n_uniq_blk
        self._S_tot = S_tot

        def add_b(nbytes: int) -> int:
            return int(self.lib.mesh_graph_add_buffer(self.g, max(64, int(nbytes)), MG_BUF_STATE, None, -1, 0))

        def imp(src_id: int) -> int:
            return int(self.lib.mesh_graph_import_buffer(self.g, trainer.g, src_id))

        def add_op(op: MeshGraphOp) -> None:
            self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))

        # Import trainer's live weight, grad, quantized weight, scalar & head_params buffers zero-copy
        iw = {k: imp(bid) for k, bid in trainer._w_bufs.items()}
        igw = {k: imp(bid) for k, bid in trainer._gw_bufs.items()}
        iwq = {k: (imp(qb), imp(sb)) for k, (qb, sb) in trainer._wq_bufs.items()}
        i_sc_live = imp(trainer.buf_scalars)
        i_hp_live = imp(trainer.buf_hp)
        i_sc_grads = imp(trainer.buf_sc_grads)

        # Shared config / diag / scratch buffers
        self.b_cfg = add_b(16 * 4)
        self.b_bounds = add_b(16 * 4)
        self.b_norm_sq = add_b(16)
        self.b_step = add_b(16)
        self.b_diag = add_b(16)

        max_r_pad = _align(D, 64)
        max_c_pad = _align(max(D, H), 64)
        s_x = add_b(max_r_pad * max_c_pad * 2)
        s_bx = add_b(max_r_pad * max_c_pad * 2)
        s_a = add_b(max_r_pad * max_r_pad * 2)
        s_a2 = add_b(max_r_pad * max_r_pad * 2)
        s_b = add_b(max_r_pad * max_r_pad * 2)
        s_geff = add_b(max(D, H) * D * 4)
        s_dvec = add_b(max(D, H) * 4)

        # Collect parameter descriptors matching PlanWally + CompiledPlanTrainer layout
        # 2D specs: (name, param, ema_param, live_key, k_pad, row_off, do_unit, is_te)
        self._p2d_specs: list[tuple[str, torch.nn.Parameter, torch.nn.Parameter, str, int, int, bool, bool]] = [
            ("enc_obs", model.enc_obs.weight, ema_model.enc_obs.weight, "enc_obs", 128, 0, True, False),
            ("enc_cart", model.enc_cart.weight, ema_model.enc_cart.weight, "enc_cart", 64, 0, True, False),
            ("enc_team", model.enc_team.weight, ema_model.enc_team.weight, "enc_team", 64, 0, True, False),
            ("type_embed", model.type_embed, ema_model.type_embed, "type_embed", D, 0, False, True),
        ]
        # 1D UnitLinear specs (min(shape) == 1 -> Adam branch in Aurora + unit-norm):
        self._p1d_specs: list[tuple[str, torch.nn.Parameter, torch.nn.Parameter, str, int, int]] = []
        for bi in range(n_uniq_blk):
            blk = model.blocks[bi]
            eblk = ema_model.blocks[bi]
            p = f"b{bi}."
            self._p2d_specs.extend([
                (p + "val_proj", blk.val_proj.weight, eblk.val_proj.weight, p + "val_proj", D, 0, True, False),
                (p + "global_metric", blk.global_metric.weight, eblk.global_metric.weight, p + "global_metric", D, 0, True, False),
                (p + "team_metric", blk.team_metric.weight, eblk.team_metric.weight, p + "team_metric", D, 0, True, False),
                (p + "rival_metric", blk.rival_metric.weight, eblk.rival_metric.weight, p + "rival_metric", D, 0, True, False),
                (p + "team_val_proj", blk.team_val_proj.weight, eblk.team_val_proj.weight, p + "team_val_proj", D, 0, True, False),
                (p + "team_mutate_proj", blk.team_mutate_proj.weight, eblk.team_mutate_proj.weight, p + "team_mutate_proj", D, 0, True, False),
                (p + "team_bcast_proj", blk.team_bcast_proj.weight, eblk.team_bcast_proj.weight, p + "team_bcast_proj", D, 0, True, False),
                (p + "gram_out", blk.gram_out.weight, eblk.gram_out.weight, p + "gram_out", D, 0, True, False),
                (p + "ffn_gate", blk.ffn_gate.weight, eblk.ffn_gate.weight, p + "ffn_gate_up", D, 0, True, False),
                (p + "ffn_up", blk.ffn_up.weight, eblk.ffn_up.weight, p + "ffn_gate_up", D, H, True, False),
                (p + "ffn_down", blk.ffn_down.weight, eblk.ffn_down.weight, p + "ffn_down", H, 0, True, False),
            ])
            self._p1d_specs.append((p + "pool_score", blk.pool_score.weight, eblk.pool_score.weight, p + "pool_score", D, 0))

        self._p2d_specs.extend([
            ("cart_q", model.cart_q.weight, ema_model.cart_q.weight, "cart_q", D, 0, True, False),
            ("cart_k", model.cart_k.weight, ema_model.cart_k.weight, "cart_k", D, 0, True, False),
            ("rival_q", model.rival_q.weight, ema_model.rival_q.weight, "rival_q", D, 0, True, False),
            ("rival_k", model.rival_k.weight, ema_model.rival_k.weight, "rival_k", D, 0, True, False),
            ("cont_proj", model.cont_proj.weight, ema_model.cont_proj.weight, "heads_packed", D, 0, True, False),
            ("value_proj", model.value_proj.weight, ema_model.value_proj.weight, "heads_packed", D, 11, True, False),
            ("dyn_action_proj", model.dyn_action_proj.weight, ema_model.dyn_action_proj.weight, "heads_packed", D, 14, True, False),
        ])
        self._p1d_specs.append(("dyn_drift_proj", model.dyn_drift_proj.weight, ema_model.dyn_drift_proj.weight, "heads_packed", D, 13))

        # Allocate per-parameter FP32 master, EMA, momentum, Adam v, and g_proj buffers
        self._bufs_2d: dict[str, tuple[int, int, int, int, tuple[int, int]]] = {}
        for name, param, _, _, _, _, _, _ in self._p2d_specs:
            m, k = int(param.shape[0]), int(param.shape[1])
            nb = m * k * 4
            self._bufs_2d[name] = (add_b(nb), add_b(nb), add_b(nb), add_b(nb), (m, k))

        self._bufs_1d: dict[str, tuple[int, int, int, int, int, int]] = {}
        for name, param, _, _, _, _ in self._p1d_specs:
            k = int(param.numel())
            nb = k * 4
            self._bufs_1d[name] = (add_b(nb), add_b(nb), add_b(nb), add_b(nb), add_b(nb), k)

        self.b_sc_m = add_b(S_tot * 4)
        self.b_sc_e = add_b(S_tot * 4)
        self.b_sc_mom = add_b(S_tot * 4)
        self.b_sc_v = add_b(S_tot * 4)
        self.b_sc_gp = add_b(S_tot * 4)

        eps_w = max(float(trainer.nc.unit_norm_eps), 1e-4)
        eps_n = max(float(trainer.nc.mp_norm_eps), 1e-4)
        ns_a, ns_b, ns_c = (float(v) for v in opt_cfg.ns_coeffs)
        ns_steps = int(opt_cfg.ns_steps)
        pp_iters = int(opt_cfg.pp_iterations)

        # --- Pass 1: Tangent projection + global L2 norm^2 across ALL 39 params ---
        for name, _, _, lkey, k_pad, row_off, _, is_te in self._p2d_specs:
            bm, _, _, bgp, (m, k) = self._bufs_2d[name]
            add_op(_make_op(
                MG_OP_OPT_TANGENT_NORM,
                in_bufs=(bm, igw[lkey]),
                out_bufs=(bgp, self.b_norm_sq),
                u=(m, k, k_pad, 1 if is_te else 0, row_off),
                f=(eps_w, eps_n),
            ))
        for name, _, _, lkey, k_pad, row_off in self._p1d_specs:
            bm, _, _, _, bgp, k = self._bufs_1d[name]
            add_op(_make_op(
                MG_OP_OPT_TANGENT_NORM,
                in_bufs=(bm, igw[lkey]),
                out_bufs=(bgp, self.b_norm_sq),
                u=(1, k, k_pad, 0, row_off),
                f=(eps_w, eps_n),
            ))
        add_op(_make_op(
            MG_OP_OPT_TANGENT_NORM,
            in_bufs=(-1, i_sc_grads),
            out_bufs=(self.b_sc_gp, self.b_norm_sq),
            u=(1, S_tot, S_tot, 2, 0),
            f=(eps_w, eps_n),
        ))

        def emit_ns(r_pad: int, c_pad: int) -> None:
            for _ in range(ns_steps):
                add_op(_make_op(MG_OP_LINEAR, in_bufs=(s_x, s_x), out_bufs=(s_a,), u=(r_pad, r_pad, c_pad)))
                add_op(_make_op(MG_OP_LINEAR, in_bufs=(s_a, s_a), out_bufs=(s_a2,), u=(r_pad, r_pad, r_pad)))
                add_op(_make_op(MG_OP_OPT_NS_AXPBY, in_bufs=(s_a, s_a2), out_bufs=(s_b,), u=(r_pad * r_pad,), f=(ns_b, ns_c)))
                add_op(_make_op(MG_OP_LINEAR_DX, in_bufs=(s_b, s_x), out_bufs=(s_bx,), u=(r_pad, c_pad, r_pad, 0)))
                add_op(_make_op(MG_OP_OPT_NS_AXPBY, in_bufs=(s_x, s_bx), out_bufs=(s_x,), u=(r_pad * c_pad,), f=(ns_a, 1.0)))

        # --- Pass 2: 2D Aurora spectral updates + unit-norm + EMA + live BF16 + zero gw ---
        for name, _, _, lkey, k_pad, row_off, do_unit, is_te in self._p2d_specs:
            bm, be, bmom, bgp, (m, k) = self._bufs_2d[name]
            trans = m > k
            r_pad = _align(min(m, k), 64)
            c_pad = _align(max(m, k), 64)
            use_d = trans and (pp_iters > 0)
            pflags = (1 if trans else 0) | (2 if use_d else 0)
            add_op(_make_op(
                MG_OP_OPT_MOM_PREP,
                in_bufs=(bgp, self.b_norm_sq, self.b_cfg, s_dvec if use_d else -1),
                out_bufs=(bmom, s_geff, s_x),
                u=(m, k, r_pad, c_pad, pflags),
            ))
            emit_ns(r_pad, c_pad)
            if use_d:
                for _ in range(pp_iters):
                    add_op(_make_op(
                        MG_OP_OPT_AURORA_DREF,
                        in_bufs=(s_x, s_geff, self.b_cfg),
                        out_bufs=(s_dvec,),
                        u=(m, k, r_pad, c_pad),
                    ))
                    emit_ns(r_pad, c_pad)
            aflags = (1 if trans else 0) | (2 if do_unit else 0) | (4 if is_te else 0)
            add_op(_make_op(
                MG_OP_OPT_APPLY_2D,
                in_bufs=(s_x, self.b_cfg, igw[lkey]),
                out_bufs=(bm, be, iw[lkey]),
                u=(m, k, r_pad, c_pad, k_pad, row_off, aflags),
            ))

        # --- Pass 3: 1D UnitLinear Adam updates (pool_score, dyn_drift_proj) ---
        for name, _, _, lkey, k_pad, row_off in self._p1d_specs:
            bm, be, bmom, bv, bgp, k = self._bufs_1d[name]
            add_op(_make_op(
                MG_OP_OPT_APPLY_1D,
                in_bufs=(bgp, self.b_norm_sq, self.b_cfg, self.b_step, bmom, bv),
                out_bufs=(bm, be, iw[lkey]),
                u=(k, k_pad, row_off, igw[lkey], 0, 0, 0, 0),
            ))

        # --- Pass 4: Scalar bank Adam + live scalars/hp pack + diag + step++ + zero sc_grads/norm_sq ---
        add_op(_make_op(
            MG_OP_OPT_APPLY_1D,
            in_bufs=(i_sc_grads, self.b_norm_sq, self.b_cfg, self.b_step, self.b_sc_mom, self.b_sc_v),
            out_bufs=(self.b_sc_m, self.b_sc_e, i_sc_live),
            u=(S_tot, n_uniq_blk, trainer.depth, 1 if trainer.recurrent else 0, i_hp_live, self.b_bounds, self.b_diag, 1),
        ))

        # --- Pass 5: In-place had_int8 weight repack directly inside the SAME command buffer! ---
        for key, rows, cols in trainer._repack_specs:
            add_op(_make_op(
                MG_OP_FWHT_PACK_I8,
                in_bufs=(iw[key],),
                out_bufs=(iwq[key][0], iwq[key][1]),
                u=(rows, cols, 0),
            ))

        rc = self.lib.mesh_graph_compile(self.g)
        if rc != 0:
            raise RuntimeError(f"mesh_graph_compile failed for CompiledAuroraOptimizer: {rc}")

        # Host unified-memory views
        self.cfg_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_cfg), (16,))
        self.bounds_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_bounds), (16,))
        self.step_view = _ptr_to_torch_i32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_step), (1,))
        self.diag_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_diag), (2,))
        self._load_config(trainer, opt_cfg)
        self.load_from_models_(model, ema_model)

    def close(self) -> None:
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    def __del__(self) -> None:
        self.close()

    def _load_config(self, trainer: CompiledPlanTrainer, oc: Any) -> None:
        mc = trainer.mc
        nc = trainer.nc
        cfg_arr = np.array([
            float(oc.lr), float(oc.scalar_lr), float(oc.mu), 1.0 if oc.nesterov else 0.0,
            float(oc.grad_clip), float(oc.pp_beta), float(oc.eps),
            float(oc.d_init_clamp[0]), float(oc.d_init_clamp[1]),
            float(oc.d_iter_clamp[0]), float(oc.d_iter_clamp[1]),
            float(oc.ema_decay), float(nc.unit_norm_eps), float(nc.mp_norm_eps),
            float(oc.scalar_beta2), float(oc.scalar_eps),
        ], dtype=np.float32)
        self.cfg_view.copy_(torch.from_numpy(cfg_arr))
        bnd_arr = np.array([
            float(mc.temp_bounds[0]), float(mc.temp_bounds[1]), float(mc.cont_gain_bound),
            float(mc.cont_logstd_bounds[0]), float(mc.cont_logstd_bounds[1]),
            float(mc.value_gain_bounds[0]), float(mc.value_gain_bounds[1]),
            float(mc.dyn_gain_bounds[0]), float(mc.dyn_gain_bounds[1]),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        self.bounds_view.copy_(torch.from_numpy(bnd_arr))

    def _pack_scalars(self, model: PlanWally) -> np.ndarray:
        vals: list[float] = []
        vals.extend(model.cont_gain.detach().float().cpu().view(-1).tolist())
        vals.extend(model.cont_bias.detach().float().cpu().view(-1).tolist())
        vals.extend(model.cont_logstd.detach().float().cpu().view(-1).tolist())
        vals.append(float(model.value_gain.detach().float().item()))
        vals.extend(model.value_bias.detach().float().cpu().view(-1).tolist())
        vals.append(float(model.dyn_drift_gain.detach().float().item()))
        vals.append(float(model.dyn_action_gain.detach().float().item()))
        vals.append(float(model.log_temp_cart.detach().float().item()))
        vals.append(float(model.log_temp_rival.detach().float().item()))
        vals.append(float(model.type_blend.detach().float().item()))
        for bi in range(self._n_uniq_blk):
            blk = model.blocks[bi]
            vals.extend([
                float(blk.blend_pool.detach().float().item()),
                float(blk.blend_team_mut.detach().float().item()),
                float(blk.blend_gram.detach().float().item()),
                float(blk.blend_ffn.detach().float().item()),
            ])
        if self.trainer.recurrent:
            vals.extend(model.step_blends.detach().float().cpu().view(-1).tolist()[: self.trainer.depth])
        return np.asarray(vals, dtype=np.float32)

    def _unpack_scalars_(self, model: PlanWally, arr: np.ndarray) -> None:
        dev = model.cont_gain.device
        dt = model.cont_gain.dtype
        t = torch.from_numpy(arr)
        model.cont_gain.copy_(t[0:11].to(device=dev, dtype=dt))
        model.cont_bias.copy_(t[11:22].to(device=dev, dtype=dt))
        model.cont_logstd.copy_(t[22:33].to(device=dev, dtype=dt))
        model.value_gain.copy_(t[33].to(device=dev, dtype=dt))
        model.value_bias.copy_(t[34:36].to(device=dev, dtype=dt))
        model.dyn_drift_gain.copy_(t[36].to(device=dev, dtype=dt))
        model.dyn_action_gain.copy_(t[37].to(device=dev, dtype=dt))
        model.log_temp_cart.copy_(t[38].to(device=dev, dtype=dt))
        model.log_temp_rival.copy_(t[39].to(device=dev, dtype=dt))
        model.type_blend.copy_(t[40].to(device=dev, dtype=dt))
        off = 41
        for bi in range(self._n_uniq_blk):
            blk = model.blocks[bi]
            blk.blend_pool.copy_(t[off + 0].to(device=dev, dtype=dt))
            blk.blend_team_mut.copy_(t[off + 1].to(device=dev, dtype=dt))
            blk.blend_gram.copy_(t[off + 2].to(device=dev, dtype=dt))
            blk.blend_ffn.copy_(t[off + 3].to(device=dev, dtype=dt))
            off += 4
        if self.trainer.recurrent:
            model.step_blends[: self.trainer.depth].copy_(t[off : off + self.trainer.depth].to(device=dev, dtype=dt))

    @torch.no_grad()
    def load_from_models_(self, model: PlanWally, ema_model: PlanWally) -> None:
        st = self.state
        for name, param, eparam, _, _, _, _, _ in self._p2d_specs:
            bm, be, bmom, _, shape = self._bufs_2d[name]
            _copy_f32_to_ptr(param.detach().to(device="cpu", dtype=torch.float32).numpy(), self.lib.mesh_graph_buffer_ptr(self.g, bm))
            _copy_f32_to_ptr(eparam.detach().to(device="cpu", dtype=torch.float32).numpy(), self.lib.mesh_graph_buffer_ptr(self.g, be))
            m_arr = st.mom.get(name)
            if m_arr is not None and m_arr.shape == shape:
                _copy_f32_to_ptr(m_arr, self.lib.mesh_graph_buffer_ptr(self.g, bmom))
        for name, param, eparam, _, _, _ in self._p1d_specs:
            bm, be, bmom, bv, _, k = self._bufs_1d[name]
            _copy_f32_to_ptr(param.detach().to(device="cpu", dtype=torch.float32).view(-1).numpy(), self.lib.mesh_graph_buffer_ptr(self.g, bm))
            _copy_f32_to_ptr(eparam.detach().to(device="cpu", dtype=torch.float32).view(-1).numpy(), self.lib.mesh_graph_buffer_ptr(self.g, be))
            if name in st.mom and st.mom[name].size == k:
                _copy_f32_to_ptr(st.mom[name], self.lib.mesh_graph_buffer_ptr(self.g, bmom))
                _copy_f32_to_ptr(st.v[name], self.lib.mesh_graph_buffer_ptr(self.g, bv))
        _copy_f32_to_ptr(self._pack_scalars(model), self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_m))
        _copy_f32_to_ptr(self._pack_scalars(ema_model), self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_e))
        if "__scalars__" in st.mom and st.mom["__scalars__"].size == self._S_tot:
            _copy_f32_to_ptr(st.mom["__scalars__"], self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_mom))
            _copy_f32_to_ptr(st.v["__scalars__"], self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_v))
        self.step_view[0] = int(st.step)

    @torch.no_grad()
    def step(self) -> tuple[float, float]:
        """Run 1 complete fused Metal 4 Aurora step (<1 ms!). Returns (grad_norm, cont_std)."""
        wt = ctypes.c_double(0.0)
        self.lib.mesh_graph_run_fused(self.g, 1, 0, ctypes.byref(wt))
        return float(self.diag_view[0].item()), float(self.diag_view[1].item())

    @torch.no_grad()
    def export_to_models_(self, model: PlanWally, ema_model: PlanWally) -> None:
        """Copy updated weights & persistent momentum/v state back ONCE at end of epoch."""
        st = self.state
        for name, param, eparam, _, _, _, _, _ in self._p2d_specs:
            bm, be, bmom, _, shape = self._bufs_2d[name]
            wm = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bm), shape)
            we = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, be), shape)
            param.copy_(wm.to(device=param.device, dtype=param.dtype))
            eparam.copy_(we.to(device=eparam.device, dtype=eparam.dtype))
            st.mom[name] = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bmom), shape).numpy().copy()
        for name, param, eparam, _, _, _ in self._p1d_specs:
            bm, be, bmom, bv, _, k = self._bufs_1d[name]
            wm = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bm), (1, k))
            we = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, be), (1, k))
            param.copy_(wm.to(device=param.device, dtype=param.dtype))
            eparam.copy_(we.to(device=eparam.device, dtype=eparam.dtype))
            st.mom[name] = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bmom), (k,)).numpy().copy()
            st.v[name] = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, bv), (k,)).numpy().copy()
        sc_m = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_m), (self._S_tot,)).numpy().copy()
        sc_e = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_e), (self._S_tot,)).numpy().copy()
        self._unpack_scalars_(model, sc_m)
        self._unpack_scalars_(ema_model, sc_e)
        st.mom["__scalars__"] = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_mom), (self._S_tot,)).numpy().copy()
        st.v["__scalars__"] = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_sc_v), (self._S_tot,)).numpy().copy()
        st.step = int(self.step_view[0].item())
