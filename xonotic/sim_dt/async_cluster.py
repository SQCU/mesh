#!/usr/bin/env python3
"""4-Studio Distributed Metal 4 Trainer:
  1. `--role rdma-dp`: 4-Studio All-Compute Data-Parallel over live /mesh0 TB5 RDMA
     (all 4 M5 Ultra Studios run PersistentMatchPool + AtemporalPPOTrainer on 256 GPU cores,
     averaging the 2.96 MB weights via /mesh0 flat-DMA MG_OP_ALLREDUCE in <3 ms!).
  2. `--role learner` / `--role actor`: Async Actor-Learner Split over LAN/TB5 TCP
     (actors precompute obs_drho to halve wire payload, stream AtemporalBatch non-stop while
     learner updates and pushes 2.96 MB BF16 weights asynchronously, logging approx_kl & staleness).
  3. `--role tp-bench`: 4-Studio FFN Tensor Parallelism benchmark over live /mesh0 HAD8_GATHER.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from sim_dt.dataset_sampler import AtemporalBatch, PersistentMatchPool
from sim_dt.metal_runtime import (
    MG_BUF_STATE,
    MG_DTYPE_BF16,
    MG_DTYPE_F32,
    MG_OP_ALLREDUCE,
    MG_WIRE_HAD8_GATHER,
    CompiledPlanWally,
    MetalDeltaTimePayloadEngine,
    _load_mesh_graph_lib,
    _make_op,
    _ptr_to_torch_bf16,
    _ptr_to_torch_f32,
)
from sim_dt.ppo_trainer import AtemporalPPOTrainer, _team_eff_rho_flat
from sim_dt.world_graph import build_gas_world_graph
from solver.strat.league_curriculum import SCRIPTED_ANCHORS
from solver.strat.plan_policy import PlanWally, clone_plan_model, project_unit_norm_
from solver.strat.train_config import CONFIGS_DIR, load_training_config

MG_WIRE_BF16 = 0
MG_WIRE_F32 = 1
MG_WIRE_GATHER = 2
MG_WIRE_HAD8 = 3
MSG_WEIGHTS = 1
MSG_BATCH = 2
MSG_STOP = 3


def _send_all(sock: socket.socket, data: bytes) -> None:
    mv = memoryview(data)
    while mv:
        n = sock.send(mv)
        if n <= 0:
            raise ConnectionError("socket send closed")
        mv = mv[n:]


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    mv = memoryview(buf)
    got = 0
    while got < nbytes:
        n = sock.recv_into(mv[got:])
        if n <= 0:
            raise ConnectionError("socket recv closed")
        got += n
    return bytes(buf)


def send_msg(sock: socket.socket, kind: int, payload: bytes) -> None:
    _send_all(sock, struct.pack("<II", int(kind), len(payload)))
    if payload:
        _send_all(sock, payload)


def recv_msg(sock: socket.socket) -> tuple[int, bytes]:
    hdr = _recv_exact(sock, 8)
    kind, length = struct.unpack("<II", hdr)
    payload = _recv_exact(sock, length) if length > 0 else b""
    return kind, payload


def serialize_batch(actor_id: int, batch: AtemporalBatch, lcfg: Any) -> bytes:
    """Compact wire serialization: precomputes obs_drho [M,N] bf16 and drops obs_next/carts_next/teams_next (>2x smaller!)."""
    if batch.obs_drho is not None:
        drho = batch.obs_drho.to(torch.bfloat16)
    else:
        assert batch.obs_next is not None and batch.carts_next is not None and batch.teams_next is not None
        rc = _team_eff_rho_flat(batch.carts, batch.teams, batch.obs, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
        rn = _team_eff_rho_flat(batch.carts_next, batch.teams_next, batch.obs_next, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
        drho = torch.asinh(lcfg.dyn_drho_scale * (rn - rc)).to(torch.bfloat16)

    meta = {
        "actor_id": int(actor_id),
        "policy_version": int(batch.policy_version),
        "world_meta": batch.world_meta,
        "opp_wr_obs": batch.opp_wr_obs,
        "stats": batch.stats,
        "geo_center": batch.geo["map_center"].float().cpu().tolist(),
        "geo_radius": float(batch.geo["map_radius"].float().item()),
    }
    meta_b = json.dumps(meta).encode("utf-8")
    tensors = [
        batch.obs, batch.carts, batch.teams,
        drho,
        batch.cont, batch.cart_pick, batch.rival_pick,
        batch.beh_logp, batch.adv, batch.gae_ret,
        batch.wm_in, batch.actor_mask,
    ]
    shapes_b = json.dumps([list(t.shape) for t in tensors]).encode("utf-8")
    raws = [t.contiguous().view(torch.uint8).numpy().tobytes() for t in tensors]
    hdr = struct.pack("<II", len(meta_b), len(shapes_b)) + struct.pack(f"<{len(raws)}I", *(len(r) for r in raws))
    return b"".join([hdr, meta_b, shapes_b, *raws])


def deserialize_batch(raw: bytes) -> tuple[int, AtemporalBatch]:
    dtypes = [
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16, torch.int16, torch.int16,
        torch.float32, torch.float32, torch.float32,
        torch.bool, torch.bool,
    ]
    n_t = len(dtypes)
    meta_len, shapes_len = struct.unpack_from("<II", raw, 0)
    lens = struct.unpack_from(f"<{n_t}I", raw, 8)
    off = 8 + 4 * n_t
    meta = json.loads(raw[off : off + meta_len].decode("utf-8"))
    off += meta_len
    shapes = json.loads(raw[off : off + shapes_len].decode("utf-8"))
    off += shapes_len
    ts: list[torch.Tensor] = []
    for dt, sh, ln in zip(dtypes, shapes, lens):
        arr = np.frombuffer(raw, dtype=np.uint8, count=ln, offset=off).copy()
        ts.append(torch.from_numpy(arr).view(dt).reshape(sh))
        off += ln
    geo = {
        "map_center": torch.tensor(meta["geo_center"], dtype=torch.float32),
        "map_radius": torch.tensor(meta["geo_radius"], dtype=torch.float32),
    }
    batch = AtemporalBatch(
        obs=ts[0], carts=ts[1], teams=ts[2],
        obs_next=None, carts_next=None, teams_next=None,
        obs_drho=ts[3].float(),
        cont=ts[4], cart_pick=ts[5], rival_pick=ts[6],
        beh_logp=ts[7], adv=ts[8], gae_ret=ts[9],
        wm_in=ts[10], actor_mask=ts[11],
        geo=geo,
        policy_version=int(meta["policy_version"]),
        world_meta=meta["world_meta"],
        opp_wr_obs=meta["opp_wr_obs"],
        stats=meta["stats"],
    )
    return int(meta["actor_id"]), batch


class RDMAWeightAllReducer:
    """/mesh0 TB5 flat-DMA parameter / delta all-reducer supporting:
      - `had8_gather` (default): 3.76x compressed HAD8_GATHER on BF16 step deltas ΔW = bf16(W - W_prev),
        accumulated into FP32 W_master with zero drift + exact FP32 scalar crossing for 4-Rank GradSNR!
      - `gather`: exact FP32 Steiner-multicast gather (`MG_WIRE_GATHER = 2`).
      - `f32`: exact FP32 in-switch NEON reduction (`MG_WIRE_F32 = 1`).
    """

    def __init__(
        self,
        model: Any,
        world_size: int,
        rank: int,
        comm_id: str,
        *,
        wire_mode: str = "had8_gather",
        n_params_override: int | None = None,
    ):
        self.lib = _load_mesh_graph_lib()
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.wire_mode_name = str(wire_mode)
        self.n_params = int(sum(p.numel() for p in model.parameters()) if n_params_override is None else n_params_override)
        self.dim = 256
        self.use_had8 = (self.wire_mode_name == "had8_gather")

        if self.use_had8:
            self.rows = (self.n_params + self.dim - 1) // self.dim
            self.n_pad = self.rows * self.dim
            self.g = self.lib.mesh_graph_create(self.world_size, self.rank, comm_id.encode("utf-8"), MG_WIRE_HAD8_GATHER)
            if not self.g:
                raise RuntimeError("mesh_graph_create failed for RDMAWeightAllReducer(had8_gather)")
            self.b_w = int(self.lib.mesh_graph_add_buffer(self.g, self.n_pad * 2, MG_BUF_STATE, None, -1, 0))
            cid_w = int(self.lib.mesh_graph_add_crossing(self.g, self.n_pad, 2, MG_WIRE_HAD8_GATHER, 0))
            op_w = _make_op(
                MG_OP_ALLREDUCE,
                dtype=MG_DTYPE_BF16,
                in_bufs=(self.b_w,),
                out_bufs=(self.b_w,),
                u=(cid_w, self.rows, self.dim),
            )
            self.lib.mesh_graph_add_op(self.g, ctypes.byref(op_w))
            # Tiny 1-page FP32 scalar crossing in the SAME CB for exact 4-Rank GradSNR (||ΔW_r||^2)!
            self.b_sq = int(self.lib.mesh_graph_add_buffer(self.g, 16 * 4, MG_BUF_STATE, None, -1, 0))
            cid_sq = int(self.lib.mesh_graph_add_crossing(self.g, 16, 4, MG_WIRE_GATHER, 0))
            op_sq = _make_op(
                MG_OP_ALLREDUCE,
                dtype=MG_DTYPE_F32,
                in_bufs=(self.b_sq,),
                out_bufs=(self.b_sq,),
                u=(cid_sq, 1, 16),
            )
            self.lib.mesh_graph_add_op(self.g, ctypes.byref(op_sq))
            rc = self.lib.mesh_graph_compile(self.g)
            if rc != 0:
                raise RuntimeError(f"mesh_graph_compile failed for RDMAWeightAllReducer(had8_gather): {rc}")
            self.w_view_bf16 = _ptr_to_torch_bf16(self.lib.mesh_graph_buffer_ptr(self.g, self.b_w), (self.n_pad,))
            self.sq_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_sq), (16,))
            self._tmp_f32 = torch.empty((self.n_params,), dtype=torch.float32)
            self.w_view = None
        else:
            wm_int = MG_WIRE_F32 if self.wire_mode_name == "f32" else MG_WIRE_GATHER
            # Reserve 1 extra FP32 slot at index self.n_params to piggyback ||delta_r||_2^2 for exact 4-Studio GradSNR!
            self.rows = (self.n_params + 1 + self.dim - 1) // self.dim
            self.n_pad = self.rows * self.dim
            self.g = self.lib.mesh_graph_create(self.world_size, self.rank, comm_id.encode("utf-8"), wm_int)
            if not self.g:
                raise RuntimeError("mesh_graph_create failed for RDMAWeightAllReducer")
            self.b_w = int(self.lib.mesh_graph_add_buffer(self.g, self.n_pad * 4, MG_BUF_STATE, None, -1, 0))
            cid = int(self.lib.mesh_graph_add_crossing(self.g, self.n_pad, 4, wm_int, 0))
            op = _make_op(
                MG_OP_ALLREDUCE,
                dtype=MG_DTYPE_F32,
                in_bufs=(self.b_w,),
                out_bufs=(self.b_w,),
                u=(cid, self.rows, self.dim),
            )
            self.lib.mesh_graph_add_op(self.g, ctypes.byref(op))
            rc = self.lib.mesh_graph_compile(self.g)
            if rc != 0:
                raise RuntimeError(f"mesh_graph_compile failed for RDMAWeightAllReducer: {rc}")
            self.w_view = _ptr_to_torch_f32(self.lib.mesh_graph_buffer_ptr(self.g, self.b_w), (self.n_pad,))
            self.w_view_bf16 = None
            self.sq_view = None
            self._tmp_f32 = None

    def close(self) -> None:
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    @torch.no_grad()
    def average_model_(self, model: Any, *, project_fn: Any = None) -> float:
        """All-reduce parameters across all ranks over /mesh0 TB5 RDMA and re-project unit norms."""
        inv_ws = 1.0 / float(self.world_size)
        if self.use_had8:
            assert self.w_view_bf16 is not None
            off = 0
            for p in model.parameters():
                n = p.numel()
                self.w_view_bf16[off : off + n].copy_(p.detach().to(torch.bfloat16).view(-1))
                off += n
            if off < self.n_pad:
                self.w_view_bf16[off:].zero_()
            wall = ctypes.c_double(0.0)
            self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))
            off = 0
            for p in model.parameters():
                n = p.numel()
                avg = self.w_view_bf16[off : off + n].float() * inv_ws
                p.copy_(avg.view(p.shape).to(device=p.device, dtype=p.dtype))
                off += n
        else:
            assert self.w_view is not None
            off = 0
            for p in model.parameters():
                n = p.numel()
                self.w_view[off : off + n].copy_(p.detach().float().view(-1))
                off += n
            if off < self.n_pad:
                self.w_view[off:].zero_()
            wall = ctypes.c_double(0.0)
            self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))
            off = 0
            for p in model.parameters():
                n = p.numel()
                avg = self.w_view[off : off + n] * inv_ws
                p.copy_(avg.view(p.shape).to(device=p.device, dtype=p.dtype))
                off += n
        if project_fn is not None:
            project_fn(model, eps=model.norm_cfg.unit_norm_eps)
        else:
            project_unit_norm_(model, eps=model.norm_cfg.unit_norm_eps)
        return float(wall.value)

    @torch.no_grad()
    def average_master_(self, opt: Any, *, prev_flat: torch.Tensor | None = None) -> tuple[float, float, float]:
        """All-reduce FP32 optimizer master weights (or HAD8_GATHER compressed BF16 step deltas!).

        Returns (rdma_wall_ms, cluster_grad_snr, cluster_delta_coh).
        """
        inv_ws = 1.0 / float(self.world_size)
        if self.use_had8 and prev_flat is not None:
            assert self.w_view_bf16 is not None and self.sq_view is not None and self._tmp_f32 is not None
            off = opt.export_master_flat_(self._tmp_f32)
            delta = self._tmp_f32[:off] - prev_flat[:off]
            local_sq = float(delta.square().sum().item())
            self.w_view_bf16[:off].copy_(delta.to(torch.bfloat16))
            if off < self.n_pad:
                self.w_view_bf16[off:].zero_()
            self.sq_view.zero_()
            self.sq_view[0] = local_sq

            wall = ctypes.c_double(0.0)
            self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))

            cluster_snr = 0.0
            cluster_coh = 0.0
            if self.world_size > 1:
                mean_sq = float(self.sq_view[0].item()) * inv_ws
                avg_delta = self.w_view_bf16[:off].float() * inv_ws
                sig_sq = float(avg_delta.square().sum().item())
                var_mean = max(mean_sq - sig_sq, 0.0) / float(self.world_size - 1)
                cluster_coh = sig_sq / max(mean_sq, 1e-12)
                cluster_snr = min(sig_sq / max(var_mean, 1e-12), 999.0)

            opt.apply_delta_flat_(prev_flat, self.w_view_bf16, scale=inv_ws)
            return float(wall.value), round(cluster_snr, 3), round(cluster_coh, 4)

        assert self.w_view is not None
        off = opt.export_master_flat_(self.w_view)
        if prev_flat is not None:
            local_sq = float((self.w_view[:off] - prev_flat).square().sum().item())
            self.w_view[off] = local_sq
            if off + 1 < self.n_pad:
                self.w_view[off + 1 :].zero_()
        elif off < self.n_pad:
            self.w_view[off:].zero_()

        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))

        cluster_snr = 0.0
        cluster_coh = 0.0
        if prev_flat is not None and self.world_size > 1:
            mean_sq = float(self.w_view[off].item()) * inv_ws
            avg_delta = self.w_view[:off] * inv_ws - prev_flat
            sig_sq = float(avg_delta.square().sum().item())
            var_mean = max(mean_sq - sig_sq, 0.0) / float(self.world_size - 1)
            cluster_coh = sig_sq / max(mean_sq, 1e-12)
            cluster_snr = min(sig_sq / max(var_mean, 1e-12), 999.0)

        opt.load_master_flat_(self.w_view, scale=inv_ws)
        return float(wall.value), round(cluster_snr, 3), round(cluster_coh, 4)


def run_rdma_dp(args: argparse.Namespace) -> None:
    """4-Studio All-Compute Data-Parallel: 4x rollout + 4x updates + <3 ms /mesh0 RDMA weight average."""
    overrides: dict[str, Any] = {}
    sd: dict[str, Any] = {}
    if args.iterations > 0:
        sd["iterations"] = args.iterations
    if args.base_envs > 0:
        sd["base_envs"] = args.base_envs
    if args.group_size > 0:
        sd["group_size"] = args.group_size
    if args.horizon > 0:
        sd["rollout_horizon"] = args.horizon
    if args.ppo_epochs > 0:
        sd["ppo_epochs"] = args.ppo_epochs
    if sd:
        overrides["sim_dt"] = sd
    cfg = load_training_config(args.config, overrides=overrides or None)

    torch.manual_seed(cfg.seeds.model_init)
    np.random.seed(cfg.seeds.model_init)
    model = PlanWally(model_cfg=cfg.model, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
    ema_model = clone_plan_model(model)
    main_init = clone_plan_model(model)

    reducer = RDMAWeightAllReducer(model, args.tp_world_size, args.tp_rank, args.tp_comm_id)
    variants: dict[str, PlanWally | None] = {"main": model, "ema": ema_model, "main_init": main_init}
    pool = PersistentMatchPool(
        cfg, variants,
        num_worlds=int(args.num_worlds),
        group_size=int(cfg.sim_dt.group_size),
        seed_base=cfg.seeds.match_base + int(args.tp_rank) * 100003,
        self_play_only=bool(args.self_play),
        roster_mode=str(args.roster_mode),
        randomize_physics=not bool(args.no_physics_dr),
    )
    learner = AtemporalPPOTrainer(model, ema_model, cfg)
    pfsp_pool = ["default", *SCRIPTED_ANCHORS, "main_init", "ema"]
    pfsp_wr = {a: float(cfg.league.pfsp_default_winrate) for a in pfsp_pool}

    total_iters = int(cfg.sim_dt.iterations)
    mid_iter = max(1, total_iters // 2)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is not None and args.tp_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(main_init.state_dict(), out_dir / "main_init.pt")

    ws = int(args.tp_world_size)
    t_start = time.perf_counter()
    tot_sim_s = 0.0
    tot_ag_steps = 0
    tot_rdma_ms = 0.0
    history: list[dict[str, Any]] = []

    try:
        for it in range(1, total_iters + 1):
            ent_coef = cfg.losses.entropy_for_iteration("main", it, total_iters)
            batch = pool.sample_batch(
                policy_version=learner.policy_version,
                pfsp_pool=pfsp_pool,
                pfsp_wr=pfsp_wr,
            )
            upd = learner.update(batch, entropy_coef=ent_coef)
            rdma_ms = reducer.average_model_(model)
            tot_rdma_ms += rdma_ms
            pool.sync_variant_("main", model)
            pool.sync_variant_("ema", ema_model)

            alpha_ema = float(cfg.sim_dt.pfsp_update_ema)
            for opp_arm, wr_val in batch.opp_wr_obs.items():
                prev = pfsp_wr.get(opp_arm, float(cfg.league.pfsp_default_winrate))
                pfsp_wr[opp_arm] = alpha_ema * prev + (1.0 - alpha_ema) * float(wr_val)

            if it == mid_iter and out_dir is not None and args.tp_rank == 0:
                torch.save(model.state_dict(), out_dir / "main_mid.pt")

            M, N, _ = batch.obs.shape
            tot_sim_s += float(batch.stats["sim_s"]) * ws
            tot_ag_steps += int(M * N) * ws
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            if args.tp_rank == 0:
                wm = batch.world_meta
                history.append({"iter": it, "world": wm, "rollout": batch.stats, "update": upd, "rdma_ms": round(rdma_ms, 2)})
                if it == 1 or it % max(1, args.log_every) == 0 or it == total_iters:
                    print(
                        f"  [RDMA-DP x{ws} dr={args.roster_mode}] iter {it:02d}/{total_iters} (R0 K={wm['K']:2d} J={wm['J']:2d} N={N:2d}) | "
                        f"roll={batch.stats['wall_s']:.2f}s upd={upd['update_wall_s']:.2f}s rdma={rdma_ms:.1f}ms | "
                        f"kl={upd['approx_kl']:.5f} loss={upd['loss_total']:.3f} ent={upd['entropy_disc']:.2f} "
                        f"dSc(m={batch.stats['main_dscore_mean']:.1f},o={batch.stats['opp_dscore_mean']:.1f}) | "
                        f"cluster={tot_ag_steps/elapsed/1e6:.2f}M ag-st/s ({tot_sim_s/elapsed:.0f}x RT)",
                        flush=True,
                    )
        if out_dir is not None and args.tp_rank == 0:
            torch.save(model.state_dict(), out_dir / "main_final.pt")
            torch.save(ema_model.state_dict(), out_dir / "ema_final.pt")
    finally:
        pool.close()
        learner.close()
        reducer.close()

    if args.tp_rank == 0:
        wall = max(time.perf_counter() - t_start, 1e-6)
        summary = {
            "world_size": ws,
            "roster_mode": args.roster_mode,
            "physics_dr": not bool(args.no_physics_dr),
            "total_iters": total_iters,
            "total_studio_batches": total_iters * ws,
            "wall_s": round(wall, 2),
            "avg_rdma_ms": round(tot_rdma_ms / max(1, total_iters), 2),
            "agent_steps_total": int(tot_ag_steps),
            "agent_steps_per_s": round(tot_ag_steps / wall, 1),
            "sim_speedup_x": round(tot_sim_s / wall, 1),
            "history": history,
        }
        if out_dir is not None:
            (out_dir / "train_report.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[RDMA-DP x{ws}] DONE {total_iters} global iters ({total_iters*ws} studio-batches, {tot_ag_steps/1e6:.1f}M ag-steps) in {wall:.2f}s | "
            f"avg_rdma={tot_rdma_ms/max(1,total_iters):.1f}ms | "
            f"cluster={tot_ag_steps/wall/1e6:.2f}M training-consumed ag-st/s | {tot_sim_s/wall:.0f}x RT!",
            flush=True,
        )


def run_learner(args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    sd: dict[str, Any] = {}
    if args.iterations > 0:
        sd["iterations"] = args.iterations
    if args.base_envs > 0:
        sd["base_envs"] = args.base_envs
    if args.group_size > 0:
        sd["group_size"] = args.group_size
    if args.horizon > 0:
        sd["rollout_horizon"] = args.horizon
    if args.ppo_epochs > 0:
        sd["ppo_epochs"] = args.ppo_epochs
    if sd:
        overrides["sim_dt"] = sd
    cfg = load_training_config(args.config, overrides=overrides or None)

    torch.manual_seed(cfg.seeds.model_init)
    np.random.seed(cfg.seeds.model_init)
    model = PlanWally(model_cfg=cfg.model, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
    ema_model = clone_plan_model(model)
    trainer = AtemporalPPOTrainer(model, ema_model, cfg)

    batch_q: queue.Queue[tuple[int, AtemporalBatch]] = queue.Queue(maxsize=max(4, args.num_actors * 2))
    actor_socks: list[tuple[socket.socket, threading.Lock]] = []
    socks_lock = threading.Lock()
    stop_evt = threading.Event()

    def broadcast_weights(ver: int, wbytes: bytes) -> None:
        payload = struct.pack("<I", int(ver)) + wbytes
        with socks_lock:
            targets = list(actor_socks)
        for s, lk in targets:
            try:
                with lk:
                    send_msg(s, MSG_WEIGHTS, payload)
            except Exception as ex:
                if not stop_evt.is_set():
                    print(f"[async-learner] broadcast err: {ex}", file=sys.stderr, flush=True)

    def handle_actor(conn: socket.socket, lk: threading.Lock) -> None:
        try:
            w0 = struct.pack("<I", int(trainer.policy_version)) + trainer.export_weights_bytes()
            with lk:
                send_msg(conn, MSG_WEIGHTS, w0)
            while not stop_evt.is_set():
                kind, payload = recv_msg(conn)
                if kind == MSG_BATCH:
                    item = deserialize_batch(payload)
                    while batch_q.full():
                        try:
                            batch_q.get_nowait()
                        except queue.Empty:
                            break
                    batch_q.put(item)
                elif kind == MSG_STOP:
                    break
        except Exception as ex:
            if not stop_evt.is_set():
                print(f"[async-learner] actor handler err: {ex}", file=sys.stderr, flush=True)

    host, port_s = args.bind.rsplit(":", 1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port_s)))
    srv.listen(16)
    print(f"[async-learner] listening on {args.bind} waiting for {args.num_actors} actors (target_iters={cfg.sim_dt.iterations})...", flush=True)

    for _ in range(args.num_actors):
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        lk = threading.Lock()
        with socks_lock:
            actor_socks.append((conn, lk))
        threading.Thread(target=handle_actor, args=(conn, lk), daemon=True).start()
        print(f"[async-learner] connected actor from {addr}", flush=True)

    total_iters = int(cfg.sim_dt.iterations)
    total_sim_s = 0.0
    total_agent_steps = 0
    history: list[dict[str, Any]] = []
    t_start: float | None = None

    try:
        for it in range(1, total_iters + 1):
            actor_id, batch = batch_q.get()
            if t_start is None:
                t_start = time.perf_counter()
            ent_coef = cfg.losses.entropy_for_iteration("main", it, total_iters)
            upd = trainer.update(batch, entropy_coef=ent_coef)
            wbytes = trainer.export_weights_bytes()
            threading.Thread(target=broadcast_weights, args=(trainer.policy_version, wbytes), daemon=True).start()

            M, N, _ = batch.obs.shape
            total_sim_s += float(batch.stats["sim_s"])
            total_agent_steps += int(M * N)
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            history.append({"iter": it, "actor_id": actor_id, "world": batch.world_meta, "rollout": batch.stats, "update": upd})

            print(
                f"  iter {it:02d}/{total_iters} [A{actor_id} K={batch.world_meta['K']:2d} J={batch.world_meta['J']:2d} N={N:2d}] | "
                f"upd={upd['update_wall_s']:.2f}s stale={upd['staleness_steps']:2d} kl={upd['approx_kl']:.5f} clip={upd['clip_frac']:.3f} | "
                f"loss={upd['loss_total']:.3f} ent={upd['entropy_disc']:.2f} "
                f"dSc(m={batch.stats['main_dscore_mean']:.1f},o={batch.stats['opp_dscore_mean']:.1f}) | "
                f"agg={total_agent_steps/elapsed/1e6:.2f}M ag-st/s ({total_sim_s/elapsed:.0f}x RT, {it/elapsed:.2f} it/s)",
                flush=True,
            )
    finally:
        stop_evt.set()
        with socks_lock:
            for s, lk in actor_socks:
                try:
                    with lk:
                        send_msg(s, MSG_STOP, b"")
                except Exception:
                    pass
                try:
                    s.close()
                except Exception:
                    pass
        srv.close()
        trainer.close()

    wall_s = time.perf_counter() - (t_start or time.perf_counter())
    print(
        f"[async-learner] DONE {total_iters} iters in {wall_s:.2f}s | "
        f"{total_iters/max(wall_s,1e-6):.2f} it/s | "
        f"{total_agent_steps/max(wall_s,1e-6)/1e6:.2f}M agent-steps/s | "
        f"{total_sim_s/max(wall_s,1e-6):.0f}x realtime",
        flush=True,
    )


def run_actor(args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    sd: dict[str, Any] = {}
    if args.base_envs > 0:
        sd["base_envs"] = args.base_envs
    if args.group_size > 0:
        sd["group_size"] = args.group_size
    if args.horizon > 0:
        sd["rollout_horizon"] = args.horizon
    if sd:
        overrides["sim_dt"] = sd
    cfg = load_training_config(args.config, overrides=overrides or None)

    torch.manual_seed(cfg.seeds.model_init)
    model = PlanWally(model_cfg=cfg.model, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
    main_init = clone_plan_model(model)
    variants: dict[str, PlanWally | None] = {"main": model, "main_init": main_init}
    pool = PersistentMatchPool(
        cfg, variants,
        num_worlds=int(args.num_worlds),
        group_size=int(cfg.sim_dt.group_size),
        seed_base=cfg.seeds.match_base + int(args.actor_id) * 100003,
        self_play_only=bool(args.self_play),
    )
    pfsp_pool = ["default", *SCRIPTED_ANCHORS, "main_init", "main"]
    pfsp_wr = {a: float(cfg.league.pfsp_default_winrate) for a in pfsp_pool}

    host, port_s = args.learner_addr.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for _ in range(50):
        try:
            sock.connect((host, int(port_s)))
            break
        except OSError:
            time.sleep(0.2)

    stop_evt = threading.Event()
    latest_w_lock = threading.Lock()
    latest_w: list[tuple[int, bytes] | None] = [None]
    got_first_w = threading.Event()
    send_q: queue.Queue[bytes] = queue.Queue(maxsize=2)

    def recv_loop() -> None:
        try:
            while not stop_evt.is_set():
                kind, payload = recv_msg(sock)
                if kind == MSG_WEIGHTS:
                    ver = struct.unpack_from("<I", payload, 0)[0]
                    with latest_w_lock:
                        latest_w[0] = (int(ver), payload[4:])
                    got_first_w.set()
                elif kind == MSG_STOP:
                    stop_evt.set()
                    break
        except Exception as ex:
            if not stop_evt.is_set():
                print(f"[async-actor {args.actor_id}] recv err: {ex}", file=sys.stderr, flush=True)
            stop_evt.set()

    def send_loop() -> None:
        try:
            while not stop_evt.is_set():
                try:
                    blob = send_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                send_msg(sock, MSG_BATCH, blob)
        except Exception as ex:
            if not stop_evt.is_set():
                print(f"[async-actor {args.actor_id}] send err: {ex}", file=sys.stderr, flush=True)
            stop_evt.set()

    threading.Thread(target=recv_loop, daemon=True).start()
    threading.Thread(target=send_loop, daemon=True).start()
    got_first_w.wait(timeout=30.0)

    local_ver = -1
    sent = 0
    try:
        while not stop_evt.is_set():
            with latest_w_lock:
                w_item = latest_w[0]
            if w_item is not None and w_item[0] > local_ver:
                ver, raw = w_item
                off = 0
                with torch.no_grad():
                    for _, p in model.named_parameters():
                        nb = p.numel() * 2
                        arr = np.frombuffer(raw, dtype=np.uint8, count=nb, offset=off).copy()
                        p.copy_(torch.from_numpy(arr).view(torch.bfloat16).reshape(p.shape))
                        off += nb
                pool.sync_variant_("main", model)
                local_ver = ver

            batch = pool.sample_batch(policy_version=max(0, local_ver), pfsp_pool=pfsp_pool, pfsp_wr=pfsp_wr)
            for opp, wr in batch.opp_wr_obs.items():
                pfsp_wr[opp] = 0.8 * pfsp_wr.get(opp, 0.45) + 0.2 * wr
            blob = serialize_batch(args.actor_id, batch, cfg.losses)
            while send_q.full() and not stop_evt.is_set():
                try:
                    send_q.get_nowait()
                except queue.Empty:
                    break
            if not stop_evt.is_set():
                send_q.put(blob)
                sent += 1
    finally:
        stop_evt.set()
        pool.close()
        try:
            sock.close()
        except Exception:
            pass
    print(f"[async-actor {args.actor_id}] done after sending {sent} batches (last ver={local_ver})", flush=True)


def run_tp_bench(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    w = build_gas_world_graph(map_name="tp", teams=11, carts=37, players=77, tiles=6, seed=7, device=torch.device("cpu"))
    eng = MetalDeltaTimePayloadEngine(w, num_envs=args.tp_envs, num_players=77, base_seed=1)
    obs, carts, teams, geo = eng.observe_tensors()
    obs, carts, teams = obs.clone(), carts.clone(), teams.clone()
    m = PlanWally().to(dtype=torch.bfloat16)

    cpol = CompiledPlanWally(
        m, E=args.tp_envs, N=77, J=37, K=11,
        do_pack=True, stochastic=False, write_sample=True, sample_seed=1,
        tp_world_size=args.tp_world_size, tp_rank=args.tp_rank, tp_comm_id=args.tp_comm_id,
    )
    out = cpol.forward(obs, carts, teams, geo=geo)
    t0 = time.perf_counter()
    for _ in range(20):
        out = cpol.forward(obs, carts, teams, geo=geo)
    ms = (time.perf_counter() - t0) * 50.0
    if args.tp_rank == 0:
        import torch.nn.functional as F
        c1 = CompiledPlanWally(m, E=args.tp_envs, N=77, J=37, K=11)
        out1 = c1.forward(obs, carts, teams, geo=geo)
        cos_c = F.cosine_similarity(out.cont_mean.float().flatten(), out1.cont_mean.float().flatten(), dim=0).item()
        cos_v = F.cosine_similarity(out.value_lou.float().flatten(), out1.value_lou.float().flatten(), dim=0).item()
        t1 = time.perf_counter()
        for _ in range(20):
            c1.forward(obs, carts, teams, geo=geo)
        ms1 = (time.perf_counter() - t1) * 50.0
        print(
            f"[TP={args.tp_world_size} /mesh0 HAD8] E={args.tp_envs} (M_rows={args.tp_envs*125}): "
            f"tp={ms:.2f} ms vs single={ms1:.2f} ms ({ms1/ms:.2f}x) | parity cos(cont={cos_c:.6f}, val={cos_v:.6f})",
            flush=True,
        )
        c1.close()
    cpol.close()
    eng.close()


def run_rdma_dp_flow(args: argparse.Namespace) -> None:
    """4-Studio All-Compute RDMA Data-Parallel for GraphFlowPolicy (Graph + DPP + 1-NFE Heun Flow + ESS Boltzmann CFM)."""
    from sim_dt.graph_flow_policy import (
        GraphFlowConfig,
        GraphFlowPolicy,
        clone_graph_flow_model,
        project_graph_flow_unit_norm_,
    )
    from sim_dt.graph_flow_trainer import GraphFlowRolloutPool, GraphFlowTrainer

    overrides: dict[str, Any] = {}
    sd: dict[str, Any] = {}
    if args.iterations > 0:
        sd["iterations"] = args.iterations
    if args.base_envs > 0:
        sd["base_envs"] = args.base_envs
    if args.group_size > 0:
        sd["group_size"] = args.group_size
    if args.horizon > 0:
        sd["rollout_horizon"] = args.horizon
    if args.ppo_epochs > 0:
        sd["ppo_epochs"] = args.ppo_epochs
    if sd:
        overrides["sim_dt"] = sd
    cfg = load_training_config(args.config, overrides=overrides or None)

    torch.manual_seed(cfg.seeds.model_init)
    np.random.seed(cfg.seeds.model_init)
    gf_cfg = GraphFlowConfig()
    model = GraphFlowPolicy(gf_cfg, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
    ema_model = clone_graph_flow_model(model)
    main_init = clone_graph_flow_model(model)

    ws = int(args.tp_world_size)
    reducer = RDMAWeightAllReducer(model, ws, args.tp_rank, args.tp_comm_id) if ws > 1 else None
    variants: dict[str, GraphFlowPolicy | None] = {"main": model, "ema": ema_model, "main_init": main_init}
    G = int(cfg.sim_dt.group_size) if int(cfg.sim_dt.group_size) > 0 else 4
    pool = GraphFlowRolloutPool(
        cfg, variants,
        num_worlds=int(args.num_worlds),
        group_size=G,
        seed_base=cfg.seeds.match_base + int(args.tp_rank) * 100003,
        world_seed_base=cfg.seeds.match_base,
        self_play_only=bool(args.self_play),
        roster_mode=str(args.roster_mode),
        randomize_physics=not bool(args.no_physics_dr),
    )
    learner = GraphFlowTrainer(model, ema_model, cfg)
    pfsp_pool = ["default", "main_init", "ema", "main"]
    pfsp_wr = {a: float(cfg.league.pfsp_default_winrate) for a in pfsp_pool}

    total_iters = int(cfg.sim_dt.iterations)
    mid_iter = max(1, total_iters // 2)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is not None and args.tp_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(main_init.state_dict(), out_dir / "main_init.pt")

    t_start = time.perf_counter()
    tot_sim_s = 0.0
    tot_ag_steps = 0
    tot_rdma_ms = 0.0
    history: list[dict[str, Any]] = []

    try:
        for it in range(1, total_iters + 1):
            batch = pool.sample_batch(
                policy_version=learner.policy_version,
                pfsp_pool=pfsp_pool,
                pfsp_wr=pfsp_wr,
            )
            upd = learner.update(batch, epochs=max(1, int(cfg.sim_dt.ppo_epochs)))
            rdma_ms, _, _ = (0.0, 0.0, 0.0)
            if reducer is not None:
                rdma_ms, _, _ = reducer.average_master_(learner.opt)
            tot_rdma_ms += rdma_ms
            pool.sync_variant_("main", model)
            pool.sync_variant_("ema", ema_model)

            alpha_ema = float(cfg.sim_dt.pfsp_update_ema)
            for opp_arm, wr_val in batch.opp_wr_obs.items():
                prev = pfsp_wr.get(opp_arm, float(cfg.league.pfsp_default_winrate))
                pfsp_wr[opp_arm] = alpha_ema * prev + (1.0 - alpha_ema) * float(wr_val)

            if it == mid_iter and out_dir is not None and args.tp_rank == 0:
                torch.save(model.state_dict(), out_dir / "main_mid.pt")

            M, N, _ = batch.obs.shape
            tot_sim_s += float(batch.stats["sim_s"]) * ws
            tot_ag_steps += int(M * 2 * N) * ws  # n_phys_substeps = 2
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            if args.tp_rank == 0:
                wm = batch.world_meta
                history.append({"iter": it, "world": wm, "rollout": batch.stats, "update": upd, "rdma_ms": round(rdma_ms, 2)})
                if it == 1 or it % max(1, args.log_every) == 0 or it == total_iters:
                    gk = batch.stats["gk_pct"]
                    print(
                        f"  [GF-RDMA x{ws} G={G} dr={args.roster_mode}] iter {it:02d}/{total_iters} "
                        f"(K={wm['K']:2d} J={wm['J']:2d} W={wm['W_nav']:2d} N={N:2d}) | "
                        f"roll={batch.stats['wall_s']:.2f}s upd={upd['update_wall_s']:.2f}s rdma={rdma_ms:.1f}ms | "
                        f"vel={upd['loss_vel']:.3f} val={upd['loss_val']:.3f} ess={upd['ess_frac']:.2f} a={upd['alpha']:.2f} | "
                        f"uRms={batch.stats['u_rms']:.2f} sat={batch.stats['sat_frac']:.3f} uSep={batch.stats['branch_u_sep']:.2f} "
                        f"GK={gk[0]:.0f}/{gk[1]:.0f}/{gk[2]:.0f}/{gk[3]:.0f}% "
                        f"dSc(m={batch.stats['main_dscore_mean']:.1f},o={batch.stats['opp_dscore_mean']:.1f}) | "
                        f"{tot_ag_steps/elapsed/1e6:.2f}M ag-st/s",
                        flush=True,
                    )
        if out_dir is not None and args.tp_rank == 0:
            torch.save(model.state_dict(), out_dir / "main_final.pt")
            torch.save(ema_model.state_dict(), out_dir / "ema_final.pt")
    finally:
        pool.close()
        learner.close()
        if reducer is not None:
            reducer.close()

    if args.tp_rank == 0:
        wall = max(time.perf_counter() - t_start, 1e-6)
        summary = {
            "policy": "graph-flow",
            "world_size": ws,
            "group_size": G,
            "iterations": total_iters,
            "roster_mode": str(args.roster_mode),
            "physics_dr": not bool(args.no_physics_dr),
            "wall_s": round(wall, 2),
            "total_sim_s": round(tot_sim_s, 1),
            "total_agent_steps": int(tot_ag_steps),
            "cluster_agent_steps_s": round(tot_ag_steps / wall, 0),
            "cluster_speedup_x": round(tot_sim_s / wall, 1),
            "mean_rdma_ms": round(tot_rdma_ms / max(1, total_iters), 2),
            "pfsp_wr": {k: round(v, 4) for k, v in pfsp_wr.items()},
            "history": history,
        }
        if out_dir is not None:
            (out_dir / "train_history.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[GF-RDMA x{ws} COMPLETE] {total_iters} iters in {wall:.1f}s | "
            f"{tot_ag_steps/1e6:.1f}M ag-steps @ {tot_ag_steps/wall/1e6:.2f}M ag-st/s "
            f"({tot_sim_s/wall:.0f}x RT) | rdma={tot_rdma_ms/max(1,total_iters):.2f}ms/iter",
            flush=True,
        )


def run_rdma_dp_league(args: argparse.Namespace) -> None:
    """5-Phase Full-DR Blender + Dual-EMA + Adversarial Frozen Exploiter League over 4-Studio /mesh0 RDMA.

    Phases:
      1. main_p1   : Main pretraining in full parametric DR blender (K=2..14, N=12..104, J=4..48,
                     tiles=2..9, tile_size/cp/point_limit DR, mixed_dr rosters, 4 physics profiles,
                     world refresh every refresh_every iters) vs PFSP [default, main_init, ema_fast, ema_slow, main]
      2. exploiter_1: Fresh-init adversarial exploiter trained strictly against frozen main_p1 -> frozen & added to league
      3. main_p2   : Main hardening vs PFSP including frozen exploiter_1
      4. exploiter_2: Warm-started adversarial counter trained strictly against frozen main_p2 -> frozen & added to league
      5. main_final: Final main hardening vs PFSP including both frozen exploiters + all snapshots + dual EMAs
    Logs exact within-Studio Step GradSNR (all/backbone/basis/flow/critic) + cross-Studio 4-Rank GradSNR!
    """
    from sim_dt.graph_flow_policy import GraphFlowConfig, GraphFlowPolicy, clone_graph_flow_model
    from sim_dt.graph_flow_trainer import GraphFlowRolloutPool, GraphFlowTrainer

    overrides: dict[str, Any] = {}
    sd: dict[str, Any] = {}
    if args.base_envs > 0:
        sd["base_envs"] = args.base_envs
    if args.group_size > 0:
        sd["group_size"] = args.group_size
    if args.horizon > 0:
        sd["rollout_horizon"] = args.horizon
    if args.ppo_epochs > 0:
        sd["ppo_epochs"] = args.ppo_epochs
    if sd:
        overrides["sim_dt"] = sd
    cfg = load_training_config(args.config, overrides=overrides or None)

    torch.manual_seed(cfg.seeds.model_init)
    np.random.seed(cfg.seeds.model_init)
    gf_cfg = GraphFlowConfig()
    main_model = GraphFlowPolicy(gf_cfg, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
    ema_fast = clone_graph_flow_model(main_model)
    ema_slow = clone_graph_flow_model(main_model)
    main_init = clone_graph_flow_model(main_model)

    ws = int(args.tp_world_size)
    reducer = (
        RDMAWeightAllReducer(main_model, ws, args.tp_rank, args.tp_comm_id, wire_mode=str(args.wire_mode))
        if ws > 1
        else None
    )
    variants: dict[str, GraphFlowPolicy | None] = {
        "main": main_model,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "main_init": main_init,
    }
    G = int(cfg.sim_dt.group_size) if int(cfg.sim_dt.group_size) > 0 else 4
    n_worlds = max(3, int(args.num_worlds))
    pool = GraphFlowRolloutPool(
        cfg, variants,
        num_worlds=n_worlds,
        group_size=G,
        seed_base=cfg.seeds.match_base + int(args.tp_rank) * 100003,
        world_seed_base=cfg.seeds.match_base,
        blender_dr=True,
        self_play_only=False,
        roster_mode=str(args.roster_mode),
        randomize_physics=not bool(args.no_physics_dr),
    )
    main_learner = GraphFlowTrainer(main_model, None, cfg)
    pool.attach_weight_bank_("main", main_learner.bank, main_model)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is not None and args.tp_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(main_init.state_dict(), out_dir / "main_init.pt")

    pfsp_pool = ["default", "main_init", "ema_fast", "ema_slow", "main"]
    pfsp_wr = {a: float(cfg.league.pfsp_default_winrate) for a in pfsp_pool}
    refresh_every = max(n_worlds, int(args.refresh_worlds_every))
    world_gen = 0
    global_it = 0
    tot_sim_s = 0.0
    tot_ag_steps = 0
    tot_rdma_ms = 0.0
    t_start = time.perf_counter()
    history: list[dict[str, Any]] = []
    phase_summaries: dict[str, Any] = {}
    prev_flat = torch.empty((sum(p.numel() for p in main_model.parameters()),), dtype=torch.float32)

    def _update_synced_emas_(src_opt: Any, fast_m: GraphFlowPolicy, slow_m: GraphFlowPolicy, d_fast: float = 0.90, d_slow: float = 0.97) -> None:
        with torch.no_grad():
            for (n, _), (_, pf), (_, ps) in zip(
                src_opt.model.named_parameters(), fast_m.named_parameters(), slow_m.named_parameters()
            ):
                wm = src_opt.master[n]
                pf.copy_((d_fast * pf.float() + (1.0 - d_fast) * wm).to(pf.dtype))
                ps.copy_((d_slow * ps.float() + (1.0 - d_slow) * wm).to(ps.dtype))

    def _run_phase(
        phase_name: str,
        n_iters: int,
        learner: GraphFlowTrainer,
        train_arm: str,
        *,
        exploiter_target: str | None = None,
        update_emas: bool = True,
    ) -> dict[str, Any]:
        nonlocal world_gen, global_it, tot_sim_s, tot_ag_steps, tot_rdma_ms
        p_t0 = time.perf_counter()
        p_recs: list[dict[str, Any]] = []
        p_wr_vs_target: list[float] = []
        if args.tp_rank == 0:
            w_desc = ", ".join(f"K={s.K}/J={s.J}/N={s.N}/W={s.W_nav}" for s in pool.slots[:6])
            print(
                f"\n=== [PHASE {phase_name}] arm={train_arm} iters={n_iters} target={exploiter_target or 'PFSP'} | "
                f"pool({len(pool.slots)}): {w_desc} ... ===",
                flush=True,
            )
        for pit in range(1, n_iters + 1):
            global_it += 1
            if global_it > 1 and ((global_it - 1) % refresh_every == 0):
                world_gen += 1
                pool.refresh_all_worlds_(world_gen)
                if args.tp_rank == 0:
                    w_desc = ", ".join(f"K={s.K}/J={s.J}/N={s.N}/W={s.W_nav}" for s in pool.slots[:6])
                    print(f"  [WORLD REFRESH gen={world_gen}] {w_desc} ...", flush=True)

            batch = pool.sample_batch(
                policy_version=learner.policy_version,
                pfsp_pool=pfsp_pool,
                pfsp_wr=pfsp_wr,
                train_arm=train_arm,
                exploiter_target=exploiter_target,
            )
            learner.opt.export_master_flat_(prev_flat)
            upd = learner.update(batch, epochs=max(1, int(cfg.sim_dt.ppo_epochs)))

            rdma_ms, c_snr, c_coh = (0.0, 0.0, 0.0)
            if reducer is not None:
                rdma_ms, c_snr, c_coh = reducer.average_master_(learner.opt, prev_flat=prev_flat)
            tot_rdma_ms += rdma_ms
            pool.sync_variant_(train_arm, learner.model)
            if update_emas:
                _update_synced_emas_(learner.opt, ema_fast, ema_slow)
                pool.sync_variant_("ema_fast", ema_fast)
                pool.sync_variant_("ema_slow", ema_slow)

            if exploiter_target is None:
                alpha_ema = float(cfg.sim_dt.pfsp_update_ema)
                for opp_arm, wr_val in batch.opp_wr_obs.items():
                    if opp_arm in pfsp_wr:
                        pfsp_wr[opp_arm] = alpha_ema * pfsp_wr[opp_arm] + (1.0 - alpha_ema) * float(wr_val)
            elif exploiter_target in batch.opp_wr_obs:
                p_wr_vs_target.append(float(batch.opp_wr_obs[exploiter_target]))

            M, N, _ = batch.obs.shape
            tot_sim_s += float(batch.stats["sim_s"]) * ws
            tot_ag_steps += int(M * 2 * N) * ws
            elapsed = max(time.perf_counter() - t_start, 1e-6)

            rec = {
                "global_iter": global_it,
                "phase": phase_name,
                "phase_iter": pit,
                "arm": train_arm,
                "world": batch.world_meta,
                "rollout": batch.stats,
                "update": upd,
                "rdma_ms": round(rdma_ms, 2),
                "cluster_snr": c_snr,
                "cluster_coh": c_coh,
            }
            p_recs.append(rec)
            if args.tp_rank == 0:
                history.append(rec)
                if pit == 1 or pit % max(1, args.log_every) == 0 or pit == n_iters:
                    wm = batch.world_meta
                    gk = batch.stats["gk_pct"]
                    g_snr = upd.get("grad_snr", 0.0)
                    g_coh = upd.get("grad_coh", 0.0)
                    live_f = upd.get("live_group_frac", 1.0)
                    tgt_str = f" wrTgt={batch.opp_wr_obs.get(exploiter_target, 0.5):.2f}" if exploiter_target else ""
                    print(
                        f"  [{phase_name:11s} {pit:02d}/{n_iters:02d}] (K={wm['K']:2d} J={wm['J']:2d} W={wm['W_nav']:2d} N={N:3d} lv={live_f:.2f}) | "
                        f"roll={batch.stats['wall_s']:.2f}s upd={upd['update_wall_s']:.2f}s rdma={rdma_ms:.0f}ms | "
                        f"vel={upd['loss_vel']:.3f} val={upd['loss_val']:.3f} ess={upd['ess_frac']:.2f} a={upd['alpha']:.2f} | "
                        f"SNR(st={g_snr:.2f}/c={g_coh:.2f}, 4r={c_snr:.2f}/c={c_coh:.2f}) | "
                        f"uRms={batch.stats['u_rms']:.2f} sat={batch.stats['sat_frac']:.3f} GK={gk[0]:.0f}/{gk[1]:.0f}/{gk[2]:.0f}/{gk[3]:.0f}% "
                        f"dSc(m={batch.stats['main_dscore_mean']:.1f},o={batch.stats['opp_dscore_mean']:.1f}){tgt_str} | "
                        f"{tot_ag_steps/elapsed/1e6:.2f}M/s",
                        flush=True,
                    )

        def _pmean(fn):
            vals = [fn(r) for r in p_recs]
            return round(float(np.mean(vals)), 4) if vals else 0.0

        return {
            "phase": phase_name,
            "arm": train_arm,
            "iters": n_iters,
            "wall_s": round(time.perf_counter() - p_t0, 2),
            "mean_loss_vel": _pmean(lambda r: r["update"]["loss_vel"]),
            "mean_loss_val": _pmean(lambda r: r["update"]["loss_val"]),
            "mean_live_group_frac": _pmean(lambda r: r["update"].get("live_group_frac", 1.0)),
            "mean_step_snr": _pmean(lambda r: r["update"].get("grad_snr", 0.0)),
            "mean_step_coh": _pmean(lambda r: r["update"].get("grad_coh", 0.0)),
            "mean_cluster_snr": _pmean(lambda r: r["cluster_snr"]),
            "mean_cluster_coh": _pmean(lambda r: r["cluster_coh"]),
            "mean_step_snr_by_grp": {
                g: _pmean(lambda r, gg=g: r["update"].get("grad_snr_by_grp", {}).get(gg, 0.0))
                for g in ("backbone", "basis", "flow", "critic")
            },
            "mean_sat_frac": _pmean(lambda r: r["rollout"]["sat_frac"]),
            "mean_dscore_margin": _pmean(lambda r: r["rollout"]["main_dscore_mean"] - r["rollout"]["opp_dscore_mean"]),
            "wr_vs_target_last8": round(float(np.mean(p_wr_vs_target[-8:])), 4) if p_wr_vs_target else None,
        }

    exp1_learner: GraphFlowTrainer | None = None
    exp2_learner: GraphFlowTrainer | None = None
    try:
        # Phase 1: Full-DR Pretraining of main
        phase_summaries["p1_main"] = _run_phase("p1_main", int(args.p1_iters), main_learner, "main")
        main_p1 = clone_graph_flow_model(main_model)
        pool.sync_variant_("main_p1", main_p1)
        if out_dir is not None and args.tp_rank == 0:
            torch.save(main_p1.state_dict(), out_dir / "main_p1.pt")

        # Phase 2: Fresh-init Adversarial Exploiter 1 targeting frozen main_p1!
        torch.manual_seed(cfg.seeds.model_init + 101)
        exp1_model = GraphFlowPolicy(gf_cfg, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
        exp1_learner = GraphFlowTrainer(exp1_model, None, cfg, lr_mult=float(cfg.optimizer.exploiter_lr_mult))
        pool.attach_weight_bank_("exploiter_1", exp1_learner.bank, exp1_model)
        phase_summaries["p2_exp1"] = _run_phase(
            "p2_exp1", int(args.exp_iters), exp1_learner, "exploiter_1",
            exploiter_target="main_p1", update_emas=False,
        )
        pool.detach_weight_bank_("exploiter_1", exp1_model)
        exp1_learner.close(); exp1_learner = None
        if out_dir is not None and args.tp_rank == 0:
            torch.save(exp1_model.state_dict(), out_dir / "exploiter_1.pt")
        pfsp_pool.extend(["main_p1", "exploiter_1"])
        pfsp_wr["main_p1"] = 0.45
        pfsp_wr["exploiter_1"] = 0.25  # prioritize neutralizing exploiter_1 immediately!

        # Phase 3: Main Hardening vs PFSP (including frozen exploiter_1!)
        phase_summaries["p3_main"] = _run_phase("p3_main", int(args.p2_iters), main_learner, "main")
        main_p2 = clone_graph_flow_model(main_model)
        pool.sync_variant_("main_p2", main_p2)
        if out_dir is not None and args.tp_rank == 0:
            torch.save(main_p2.state_dict(), out_dir / "main_p2.pt")

        # Phase 4: Warm-start Adversarial Exploiter 2 targeting frozen main_p2!
        torch.manual_seed(cfg.seeds.model_init + 202)
        exp2_model = clone_graph_flow_model(main_p2)
        with torch.no_grad():
            exp2_model.fj_w2.weight.add_(0.02 * torch.randn_like(exp2_model.fj_w2.weight))
            exp2_model.fj_w2.bias.add_(0.02 * torch.randn_like(exp2_model.fj_w2.bias))
        exp2_learner = GraphFlowTrainer(exp2_model, None, cfg, lr_mult=float(cfg.optimizer.exploiter_lr_mult))
        pool.attach_weight_bank_("exploiter_2", exp2_learner.bank, exp2_model)
        phase_summaries["p4_exp2"] = _run_phase(
            "p4_exp2", int(args.exp_iters), exp2_learner, "exploiter_2",
            exploiter_target="main_p2", update_emas=False,
        )
        pool.detach_weight_bank_("exploiter_2", exp2_model)
        exp2_learner.close(); exp2_learner = None
        if out_dir is not None and args.tp_rank == 0:
            torch.save(exp2_model.state_dict(), out_dir / "exploiter_2.pt")
        pfsp_pool.extend(["main_p2", "exploiter_2"])
        pfsp_wr["main_p2"] = 0.45
        pfsp_wr["exploiter_2"] = 0.25

        # Phase 5: Final Main Hardening vs Both Frozen Exploiters + Full League!
        phase_summaries["p5_main"] = _run_phase("p5_main", int(args.p3_iters), main_learner, "main")
        if out_dir is not None and args.tp_rank == 0:
            torch.save(main_model.state_dict(), out_dir / "main_final.pt")
            torch.save(ema_fast.state_dict(), out_dir / "ema_fast_final.pt")
            torch.save(ema_slow.state_dict(), out_dir / "ema_slow_final.pt")
    finally:
        if exp1_learner is not None:
            exp1_learner.close()
        if exp2_learner is not None:
            exp2_learner.close()
        pool.close()
        main_learner.close()
        if reducer is not None:
            reducer.close()

    if args.tp_rank == 0:
        wall = max(time.perf_counter() - t_start, 1e-6)
        summary = {
            "policy": "graph-flow-league",
            "world_size": ws,
            "group_size": G,
            "num_worlds": n_worlds,
            "world_refreshes": world_gen,
            "total_iters": global_it,
            "wall_s": round(wall, 2),
            "total_sim_s": round(tot_sim_s, 1),
            "total_agent_steps": int(tot_ag_steps),
            "cluster_agent_steps_s": round(tot_ag_steps / wall, 0),
            "cluster_speedup_x": round(tot_sim_s / wall, 1),
            "mean_rdma_ms": round(tot_rdma_ms / max(1, global_it), 2),
            "pfsp_wr": {k: round(v, 4) for k, v in pfsp_wr.items()},
            "phases": phase_summaries,
            "history": history,
        }
        if out_dir is not None:
            (out_dir / "league_history.json").write_text(json.dumps(summary, indent=2))
        print(
            f"\n[GF-LEAGUE x{ws} COMPLETE] {global_it} iters ({world_gen+1} world gens) in {wall:.1f}s | "
            f"{tot_ag_steps/1e6:.1f}M ag-steps @ {tot_ag_steps/wall/1e6:.2f}M ag-st/s ({tot_sim_s/wall:.0f}x RT)",
            flush=True,
        )
        for pn, ps in phase_summaries.items():
            g = ps["mean_step_snr_by_grp"]
            print(
                f"  phase={pn:8s} ({ps['iters']:2d}it, {ps['wall_s']:5.1f}s) | "
                f"vel={ps['mean_loss_vel']:.3f} val={ps['mean_loss_val']:.3f} lv={ps['mean_live_group_frac']:.2f} sat={ps['mean_sat_frac']:.4f} | "
                f"StepSNR={ps['mean_step_snr']:.2f} (bb={g['backbone']:.2f}, B={g['basis']:.2f}, fl={g['flow']:.2f}, cr={g['critic']:.2f}) | "
                f"4RankSNR={ps['mean_cluster_snr']:.2f} (coh={ps['mean_cluster_coh']:.3f}) | "
                f"dScMgn={ps['mean_dscore_margin']:+.1f}"
                + (f" wrVsTgt={ps['wr_vs_target_last8']:.3f}" if ps["wr_vs_target_last8"] is not None else ""),
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="4-Studio RDMA Data-Parallel + Async Actor-Learner + /mesh0 TP Trainer")
    ap.add_argument("--role", choices=("rdma-dp", "rdma-dp-flow", "rdma-dp-league", "learner", "actor", "tp-bench"), required=True)
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--bind", type=str, default="0.0.0.0:29810")
    ap.add_argument("--learner-addr", type=str, default="10.0.0.124:29810")
    ap.add_argument("--num-actors", type=int, default=3)
    ap.add_argument("--actor-id", type=int, default=1)
    ap.add_argument("--num-worlds", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--p1-iters", type=int, default=48)
    ap.add_argument("--p2-iters", type=int, default=48)
    ap.add_argument("--p3-iters", type=int, default=40)
    ap.add_argument("--exp-iters", type=int, default=24)
    ap.add_argument("--refresh-worlds-every", type=int, default=18)
    ap.add_argument("--base-envs", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--ppo-epochs", type=int, default=0)
    ap.add_argument("--self-play", action="store_true")
    ap.add_argument("--roster-mode", type=str, default="mixed_dr")
    ap.add_argument("--no-physics-dr", action="store_true")
    ap.add_argument("--log-every", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--tp-world-size", type=int, default=4)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--tp-comm-id", type=str, default="10.0.0.124:29820")
    ap.add_argument("--tp-envs", type=int, default=256)
    ap.add_argument("--wire-mode", choices=("had8_gather", "gather", "f32"), default="had8_gather")
    args = ap.parse_args()

    if args.role == "rdma-dp":
        run_rdma_dp(args)
    elif args.role == "rdma-dp-flow":
        run_rdma_dp_flow(args)
    elif args.role == "rdma-dp-league":
        run_rdma_dp_league(args)
    elif args.role == "learner":
        run_learner(args)
    elif args.role == "actor":
        run_actor(args)
    else:
        run_tp_bench(args)


if __name__ == "__main__":
    main()
