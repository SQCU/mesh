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
    MG_DTYPE_F32,
    MG_OP_ALLREDUCE,
    CompiledPlanWally,
    MetalDeltaTimePayloadEngine,
    _load_mesh_graph_lib,
    _make_op,
    _ptr_to_torch_f32,
)
from sim_dt.ppo_trainer import AtemporalPPOTrainer, _team_eff_rho_flat
from sim_dt.world_graph import build_gas_world_graph
from solver.strat.league_curriculum import SCRIPTED_ANCHORS
from solver.strat.plan_policy import PlanWally, clone_plan_model, project_unit_norm_
from solver.strat.train_config import CONFIGS_DIR, load_training_config

MG_WIRE_F32 = 2
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
    """Lossless FP32 parameter all-reduce over live /mesh0 TB5 flat-DMA (libmesh-graph.dylib)."""

    def __init__(self, model: PlanWally, world_size: int, rank: int, comm_id: str):
        self.lib = _load_mesh_graph_lib()
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.n_params = sum(p.numel() for p in model.parameters())
        self.dim = 256
        self.rows = (self.n_params + self.dim - 1) // self.dim
        self.n_pad = self.rows * self.dim
        self.g = self.lib.mesh_graph_create(self.world_size, self.rank, comm_id.encode("utf-8"), MG_WIRE_F32)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed for RDMAWeightAllReducer")
        self.b_w = int(self.lib.mesh_graph_add_buffer(self.g, self.n_pad * 4, MG_BUF_STATE, None, -1, 0))
        cid = int(self.lib.mesh_graph_add_crossing(self.g, self.n_pad, 4, MG_WIRE_F32, 0))
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

    def close(self) -> None:
        if getattr(self, "g", None):
            self.lib.mesh_graph_destroy(self.g)
            self.g = None

    @torch.no_grad()
    def average_model_(self, model: Any, *, project_fn: Any = None) -> float:
        """All-reduce FP32 parameters across all ranks over /mesh0 TB5 RDMA and re-project unit norms."""
        off = 0
        for p in model.parameters():
            n = p.numel()
            self.w_view[off : off + n].copy_(p.detach().float().view(-1))
            off += n
        if off < self.n_pad:
            self.w_view[off:].zero_()
        wall = ctypes.c_double(0.0)
        self.lib.mesh_graph_run(self.g, 1, 0, None, 0, ctypes.byref(wall))
        inv_ws = 1.0 / float(self.world_size)
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
            rdma_ms = 0.0
            if reducer is not None:
                rdma_ms = reducer.average_model_(model, project_fn=project_graph_flow_unit_norm_)
                learner.reload_after_rdma_()
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
                        f"uSep={batch.stats['branch_u_sep']:.2f} GK={gk[0]:.0f}/{gk[1]:.0f}/{gk[2]:.0f}/{gk[3]:.0f}% "
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


def main() -> None:
    ap = argparse.ArgumentParser(description="4-Studio RDMA Data-Parallel + Async Actor-Learner + /mesh0 TP Trainer")
    ap.add_argument("--role", choices=("rdma-dp", "rdma-dp-flow", "learner", "actor", "tp-bench"), required=True)
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--bind", type=str, default="0.0.0.0:29810")
    ap.add_argument("--learner-addr", type=str, default="10.0.0.124:29810")
    ap.add_argument("--num-actors", type=int, default=3)
    ap.add_argument("--actor-id", type=int, default=1)
    ap.add_argument("--num-worlds", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=8)
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
    args = ap.parse_args()

    if args.role == "rdma-dp":
        run_rdma_dp(args)
    elif args.role == "rdma-dp-flow":
        run_rdma_dp_flow(args)
    elif args.role == "learner":
        run_learner(args)
    elif args.role == "actor":
        run_actor(args)
    else:
        run_tp_bench(args)


if __name__ == "__main__":
    main()
