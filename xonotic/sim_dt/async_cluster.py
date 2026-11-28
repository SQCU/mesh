#!/usr/bin/env python3
"""4-Studio Async Actor-Learner + Native /mesh0 Tensor-Parallel Trainer.

Separates dataset sampling (PersistentMatchPool on Actor Studios, zero warmup) from
atemporal policy/value optimization (AtemporalPPOTrainer on Learner Studio), streaming
flat AtemporalBatch payloads and 2.96 MB BF16 weight updates asynchronously over high-speed
sockets while logging approx_kl and policy staleness as boring first-class metrics.

Also supports `--role tp-bench` to benchmark 4-Studio FFN Tensor Parallelism directly over
live /mesh0 TB5 RDMA (MG_WIRE_HAD8_GATHER) via libmesh-graph.dylib.
"""
from __future__ import annotations

import argparse
import io
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
from sim_dt.metal_runtime import CompiledPlanTrainer, CompiledPlanWally, MetalDeltaTimePayloadEngine
from sim_dt.ppo_trainer import AtemporalPPOTrainer
from sim_dt.world_graph import build_gas_world_graph
from solver.strat.league_curriculum import SCRIPTED_ANCHORS
from solver.strat.plan_policy import PlanWally, clone_plan_model
from solver.strat.train_config import CONFIGS_DIR, load_training_config

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


def serialize_batch(actor_id: int, batch: AtemporalBatch) -> bytes:
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
        batch.obs_next, batch.carts_next, batch.teams_next,
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
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
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
        obs_next=ts[3], carts_next=ts[4], teams_next=ts[5],
        cont=ts[6], cart_pick=ts[7], rival_pick=ts[8],
        beh_logp=ts[9], adv=ts[10], gae_ret=ts[11],
        wm_in=ts[12], actor_mask=ts[13],
        geo=geo,
        policy_version=int(meta["policy_version"]),
        world_meta=meta["world_meta"],
        opp_wr_obs=meta["opp_wr_obs"],
        stats=meta["stats"],
    )
    return int(meta["actor_id"]), batch


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
            except Exception:
                pass

    def handle_actor(conn: socket.socket, lk: threading.Lock) -> None:
        try:
            # Send initial weights immediately
            w0 = struct.pack("<I", int(trainer.policy_version)) + trainer.export_weights_bytes()
            with lk:
                send_msg(conn, MSG_WEIGHTS, w0)
            while not stop_evt.is_set():
                kind, payload = recv_msg(conn)
                if kind == MSG_BATCH:
                    item = deserialize_batch(payload)
                    # Always keep freshest batches if queue is full
                    while batch_q.full():
                        try:
                            batch_q.get_nowait()
                        except queue.Empty:
                            break
                    batch_q.put(item)
                elif kind == MSG_STOP:
                    break
        except Exception:
            pass

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
            # Async non-blocking weight push to all actors!
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
    if args.out_dir:
        od = Path(args.out_dir).expanduser().resolve()
        od.mkdir(parents=True, exist_ok=True)
        (od / "async_report.json").write_text(json.dumps({
            "wall_s": round(wall_s, 3),
            "iters_per_s": round(total_iters / max(wall_s, 1e-6), 3),
            "agent_steps_per_s": round(total_agent_steps / max(wall_s, 1e-6), 1),
            "sim_speedup_x": round(total_sim_s / max(wall_s, 1e-6), 1),
            "history": history,
        }, indent=2))


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
        except Exception:
            stop_evt.set()

    def send_loop() -> None:
        try:
            while not stop_evt.is_set():
                try:
                    blob = send_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                send_msg(sock, MSG_BATCH, blob)
        except Exception:
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
            blob = serialize_batch(args.actor_id, batch)
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
    """Measure 4-Studio FFN Tensor Parallelism over live /mesh0 TB5 RDMA (MG_WIRE_HAD8_GATHER)."""
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
    # Warmup
    out = cpol.forward(obs, carts, teams, geo=geo)
    t0 = time.perf_counter()
    for _ in range(20):
        out = cpol.forward(obs, carts, teams, geo=geo)
    ms = (time.perf_counter() - t0) * 50.0
    if args.tp_rank == 0:
        # Verify numerical parity vs single-node tp_world_size=1
        c1 = CompiledPlanWally(m, E=args.tp_envs, N=77, J=37, K=11)
        out1 = c1.forward(obs, carts, teams, geo=geo)
        import torch.nn.functional as F
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


def main() -> None:
    ap = argparse.ArgumentParser(description="4-Studio Async Actor-Learner + /mesh0 TP Trainer")
    ap.add_argument("--role", choices=("learner", "actor", "tp-bench"), required=True)
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--bind", type=str, default="0.0.0.0:29810")
    ap.add_argument("--learner-addr", type=str, default="100.103.69.53:29810")
    ap.add_argument("--num-actors", type=int, default=3)
    ap.add_argument("--actor-id", type=int, default=1)
    ap.add_argument("--num-worlds", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--base-envs", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--ppo-epochs", type=int, default=0)
    ap.add_argument("--self-play", action="store_true", help="Pure self-play 1x forward (no multi-arm 3x multiplier)")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--tp-world-size", type=int, default=4)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--tp-comm-id", type=str, default="100.103.69.53:29820")
    ap.add_argument("--tp-envs", type=int, default=256)
    args = ap.parse_args()

    if args.role == "learner":
        run_learner(args)
    elif args.role == "actor":
        run_actor(args)
    else:
        run_tp_bench(args)


if __name__ == "__main__":
    main()
