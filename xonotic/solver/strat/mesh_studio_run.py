#!/usr/bin/env python3
"""Four-Studio Apple M5 Ultra Distributed Aurora Training, RDMA TP Benchmark, & Parallel Elo Runner.

Runs identically on all 4 Mac Studios over Thunderbolt RDMA (`torch.distributed` `backend="mesh"`
on `device="mps:0"`), executing:
  1. `tp_bench`: 4-Studio `products.World` Tensor Parallelism (`cross` + `ffn`) on superlarge physical
     states (N = 128..8192) verifying exact numerical agreement with single-rank MPS & collective speed.
  2. `train`: 24-server concurrent Xonotic rollout (6 servers/Studio: 2xboil, 2xstormkeep, 2xcatharsis)
     with exact 4-machine RDMA McCandlish SNR, global advantage standardization, synchronous Aurora
     leverage-aware spectral steps, EDM2 unit-norm projection, and 5 progressive league waves.
  3. `elo`: 4-Studio parallel multi-map seat-rotated CRN Elo tournament (boil, stormkeep, catharsis,
     and unseen zero-shot afterslime simultaneously, 48 matches total) + dead-neuron & FSQ audits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import CS, EVT, OBS, TS
from solver.strat.buffers import ObservationMemory
from solver.strat.game_value import effective_leader, formal_value_record, role_rewards
from solver.strat.grad_snr import summarize_mccandlish_pairs
from solver.strat.outer_validation import compute_multiteam_elo, extract_behavioral_motifs
from solver.strat.plan_policy import (
    PLAN_CONT_DIM,
    PLAN_WIDTH,
    MPLinear,
    PlanTransition,
    PlanWally,
    _geo_to_torch,
    clone_plan_model,
    edm2_step,
    ema_update,
    make_aurora_optimizer,
    pack_plan_rows,
    plan_losses_for_transition,
    plan_module_group,
    project_unit_norm_,
    sample_plan,
    weight_norm_report,
)
from solver.strat.plan_runner import MultiServerPlanHarness
from solver.strat.products import World
from solver.strat.runtime import build_runtime_frame
from solver.xonwire import CART_KIND, OBSERVATION_KIND, PLAN_KIND, TEAM_KIND, Engine, RuntimeFrames, frames


MODEL_KWARGS = dict(
    d=256,
    h=1024,
    r_gram=128,
    r_dpp=64,
    depth=4,
    recurrent=True,
    use_fsq=True,
    use_mealy=True,
)


def _init_mesh_world(device: str):
    import torch_mesh
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="mesh",
            **({"device_id": torch.device("mps", 0)} if device == "mps" else {}),
        )
        return dist.get_rank(), dist.get_world_size(), torch_mesh
    return 0, 1, torch_mesh


def _broadcast_model_(model: PlanWally, torch_mesh, rank: int, world_size: int, device: str) -> None:
    if world_size <= 1:
        return
    params = [p for p in model.parameters() if p.numel() > 0]
    total = sum(p.numel() for p in params)
    buf = torch_mesh.empty((total,), dtype=torch.float32, device=device)
    with torch.no_grad():
        if rank == 0:
            off = 0
            for p in params:
                n = p.numel()
                buf[off : off + n].copy_(p.data.reshape(-1))
                off += n
        dist.broadcast(buf, src=0)
        if rank != 0:
            off = 0
            for p in params:
                n = p.numel()
                p.data.copy_(buf[off : off + n].view_as(p.data))
                off += n


def _sync_adv_stats(model: PlanWally, batch: list[PlanTransition], torch_mesh, world_size: int, device: str) -> tuple[float, float]:
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
    cat = np.concatenate(errs) if errs else np.zeros((0,), dtype=np.float32)
    stats = torch_mesh.empty((3,), dtype=torch.float32, device=device)
    stats[0] = float(cat.size)
    stats[1] = float(cat.sum()) if cat.size else 0.0
    stats[2] = float(np.square(cat).sum()) if cat.size else 0.0
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    s_cpu = stats.cpu().numpy()
    cnt = max(float(s_cpu[0]), 1.0)
    mean = float(s_cpu[1] / cnt)
    var = max(float(s_cpu[2] / cnt) - mean * mean, 1e-6)
    return mean, max(math.sqrt(var), 1e-3)


def _rdma_mccandlish_from_flat(
    flat_g: torch.Tensor,
    slices_by_group: dict[str, list[tuple[int, int]]],
    torch_mesh,
    world_size: int,
    b0_transitions: float,
    device: str,
    step_id: int,
) -> tuple[torch.Tensor, dict, dict[str, dict]]:
    """All-reduce flat gradient vector across all R physical Studios over Thunderbolt RDMA and
    compute exact R-machine McCandlish SNR globally and per module group."""
    r = float(world_size)
    groups = ("encoder", "gram_cgt", "resnet_ffn", "action_head", "value_head", "dynamics_head")
    local_sq = torch_mesh.empty((1 + len(groups),), dtype=torch.float32, device=device)
    local_sq[0] = flat_g.square().sum()
    for gi, grp in enumerate(groups):
        s_val = torch.tensor(0.0, dtype=torch.float32, device=device)
        for lo, hi in slices_by_group[grp]:
            s_val = s_val + flat_g[lo:hi].square().sum()
        local_sq[1 + gi] = s_val

    sum_g = torch_mesh.empty(flat_g.shape, dtype=torch.float32, device=device)
    sum_g.copy_(flat_g)
    if world_size > 1:
        dist.all_reduce(sum_g, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    mean_g = sum_g / r

    def _pair_from_sums(mean_sq_norm: float, sum_rank_sq_norms: float) -> dict:
        if world_size <= 1:
            return {
                "generation": step_id,
                "replicas": 1,
                "b0_transitions": b0_transitions,
                "b0_rows": b0_transitions * 9.0,
                "b_eff_transitions": b0_transitions,
                "b_eff_rows": b0_transitions * 9.0,
                "signal_sq": mean_sq_norm,
                "noise_var": 0.0,
                "b_noise_transitions": 0.0,
                "b_noise_rows": 0.0,
                "step_efficiency": 1.0,
                "cosine": 1.0,
            }
        var_per_rank = max(0.0, (sum_rank_sq_norms - r * mean_sq_norm) / (r - 1.0))
        signal_sq = mean_sq_norm - var_per_rank / r
        b_eff = r * b0_transitions
        if signal_sq > 1e-18:
            b_noise = b0_transitions * var_per_rank / signal_sq
            eff = b_eff / (b_eff + b_noise)
        else:
            b_noise = float("inf")
            eff = 0.0
        # Average pairwise cosine across R ranks: (||sum g||^2 - sum ||g_k||^2) / ((R-1) sum ||g_k||^2)
        cos = (r * r * mean_sq_norm - sum_rank_sq_norms) / max((r - 1.0) * sum_rank_sq_norms, 1e-18)
        return {
            "generation": step_id,
            "replicas": int(world_size),
            "b0_transitions": float(b0_transitions),
            "b0_rows": float(b0_transitions * 9.0),
            "b_eff_transitions": float(b_eff),
            "b_eff_rows": float(b_eff * 9.0),
            "signal_sq": float(signal_sq),
            "noise_var": float(var_per_rank),
            "b_noise_transitions": float(b_noise) if math.isfinite(b_noise) else None,
            "b_noise_rows": float(b_noise * 9.0) if math.isfinite(b_noise) else None,
            "step_efficiency": float(eff),
            "cosine": float(max(-1.0, min(1.0, cos))),
        }

    sq_cpu = local_sq.cpu().tolist()
    tot_mean_sq = float(mean_g.square().sum().item())
    global_pair = _pair_from_sums(tot_mean_sq, float(sq_cpu[0]))

    grp_pairs = {}
    for gi, grp in enumerate(groups):
        if not slices_by_group[grp]:
            continue
        gm_sq = 0.0
        for lo, hi in slices_by_group[grp]:
            gm_sq += float(mean_g[lo:hi].square().sum().item())
        grp_pairs[grp] = _pair_from_sums(gm_sq, float(sq_cpu[1 + gi]))
    return mean_g, global_pair, grp_pairs


def run_tp_bench(device: str, out_path: Path):
    world = World(device=device, products={"cross": [1, 1, 1, 1], "ffn": [1, 1, 1, 1]})
    rank, size = world.rank, world.size
    torch.manual_seed(1234)
    model = PlanWally(**MODEL_KWARGS).to(device)
    model.eval()

    # Exactness & scaling sweep across physical state sizes N = 128, 1024, 4096, 8192
    # Each block does cross (rank-128 linear Gram over N tokens) + wide FFN (256 -> 1024 -> 256)
    results = []
    team_ids = torch.tensor([1, 2, 3], dtype=torch.long, device=device)
    team_valid = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    for n_tok in (128, 1024, 4096, 8192):
        torch.manual_seed(5000 + n_tok)
        x = torch.randn(n_tok, MODEL_KWARGS["d"], dtype=torch.float32).to(device)
        t_ids = torch.randint(1, 4, (n_tok,), dtype=torch.long).to(device)
        blk = model.blocks[0]

        # Warmup
        with torch.no_grad():
            _ = blk(x, t_ids, team_ids, team_valid, world=world)
            if device == "mps":
                torch.mps.synchronize()

        # Single-node reference on rank 0
        t0 = time.perf_counter()
        iters = 8 if n_tok <= 1024 else 4
        with torch.no_grad():
            for _ in range(iters):
                y_ref = blk(x, t_ids, team_ids, team_valid, world=None)
            if device == "mps":
                torch.mps.synchronize()
        single_ms = (time.perf_counter() - t0) * 1e3 / iters

        # 4-Studio RDMA TP timed
        world.allreduce(torch.zeros((1,), dtype=torch.float32, device=device))
        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                y_tp = blk(x, t_ids, team_ids, team_valid, world=world)
            if device == "mps":
                torch.mps.synchronize()
        tp_ms = (time.perf_counter() - t1) * 1e3 / iters

        # Full TP forward + backward exactness at N=128 and N=1024
        max_fwd_err = float((y_tp - y_ref).abs().max().item())
        max_bwd_err = None
        if n_tok <= 1024:
            x_ref = x.clone().requires_grad_(True)
            x_tp = x.clone().requires_grad_(True)
            blk.zero_grad(set_to_none=True)
            loss_ref = blk(x_ref, t_ids, team_ids, team_valid, world=None).square().mean()
            loss_ref.backward()
            g_ref = x_ref.grad.detach().clone()
            blk.zero_grad(set_to_none=True)
            loss_tp = blk(x_tp, t_ids, team_ids, team_valid, world=world).square().mean()
            loss_tp.backward()
            g_tp = x_tp.grad.detach().clone()
            max_bwd_err = float((g_tp - g_ref).abs().max().item())

        rec = {
            "n_tokens": n_tok,
            "single_studio_mps_ms": round(single_ms, 3),
            "four_studio_rdma_tp_ms": round(tp_ms, 3),
            "speedup": round(single_ms / max(tp_ms, 1e-6), 3),
            "max_abs_fwd_err": round(max_fwd_err, 7),
            "max_abs_bwd_err": round(max_bwd_err, 7) if max_bwd_err is not None else None,
        }
        results.append(rec)

    if rank == 0:
        out = {
            "mode": "tp_bench",
            "world_size": size,
            "device": device,
            "host": os.uname().nodename,
            "model_spec": model.spec(),
            "param_count": sum(p.numel() for p in model.parameters()),
            "products": {"cross": [1, 1, 1, 1], "ffn": [1, 1, 1, 1]},
            "benchmarks": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_sync_wave(
    harness: MultiServerPlanHarness,
    work_dir: Path,
    *,
    rank: int,
    world_size: int,
    torch_mesh,
    device: str,
    seeds: list[int],
    map_names: list[str],
    base_port: int,
    arm_models: dict[str, PlanWally | None],
    team_arms: tuple[str, ...],
    train_arm: str,
    optimizer: torch.optim.Optimizer,
    ema_model: PlanWally,
    ema_decay: float,
    n_sync_steps: int,
    local_batch_size: int,
    ppo_epochs: int,
    gamma: float,
    grad_clip: float,
    max_wave_s: float = 300.0,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine("udp://127.0.0.1:0", harness.framebytes)
    sock = engine.path
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    procs, logs = harness._spawn_servers(work_dir, sock, seeds, base_port, map_names=map_names)

    model = arm_models[train_arm]
    assert model is not None
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    slices_by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    off = 0
    for n, p in zip(names, params):
        sz = p.numel()
        slices_by_group[plan_module_group(n)].append((off, off + sz))
        off += sz

    routes: dict[int, tuple[int, object]] = {}
    sess_order: dict[int, int] = {}
    sess_maps: dict[int, tuple[str, object]] = {}
    obs_mems: dict[int, ObservationMemory] = {}
    prev_state: dict[int, dict] = {}
    mealy_states: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)
    fsq_code_counts: dict[tuple[int, ...], int] = defaultdict(int)
    fresh_buf: list[PlanTransition] = []

    sync_steps = 0
    deferred = 0
    total_responses = 0
    total_local_transitions = 0
    snr_pairs: list[dict] = []
    group_snr_pairs: dict[str, list[dict]] = defaultdict(list)
    step_logs: list[dict] = []
    rdma_ms_list: list[float] = []
    step_ms_list: list[float] = []
    t_wave = time.monotonic()

    try:
        while sync_steps < n_sync_steps:
            # Collect local_batch_size transitions on this Studio (all 6 local servers active)
            while len(fresh_buf) < local_batch_size and (time.monotonic() - t_wave) < max_wave_s:
                readable, _, _ = select.select([engine], [], [], 0.05)
                if not readable:
                    continue
                for node, address, buf in engine.receive():
                    s = inbox.feed(buf)
                    if s is not None:
                        routes[s] = (node, address)
                while (ready := inbox.take()) is not None:
                    (session, tick_id), records, event_batches = ready
                    if session not in sess_order:
                        sess_order[session] = len(sess_order)
                        obs_mems[session] = ObservationMemory()
                    s_idx = sess_order[session]

                    obs_raw = records[OBSERVATION_KIND][1]
                    cart_rows = records[CART_KIND][1]
                    team_rows = records[TEAM_KIND][1]
                    present_mask = obs_raw[:, OBS["PRESENT"]] > 0.5
                    obs_rows = obs_raw[present_mask]
                    if len(obs_rows) == 0 or len(cart_rows) == 0 or len(team_rows) == 0:
                        continue

                    if session not in sess_maps:
                        sess_maps[session] = harness._identify_map(cart_rows)
                    sess_map_name, sess_manifold = sess_maps[session]
                    cell_centroids = sess_manifold.cell_centroids if sess_manifold is not None else None

                    ep_id = ObservationMemory.identity(session, team_rows[0, TS["EPISODE"]])
                    mem = obs_mems[session]
                    mem.observe(ep_id, event_batches)
                    evt_rows = mem.rows
                    geo_np = sess_manifold.build_features(obs_rows, cart_rows, team_rows, evt_rows) if sess_manifold is not None else None

                    tick = build_runtime_frame(obs_rows, cart_rows, team_rows)
                    n_teams = len(team_rows)
                    shift = (s_idx + rank) % len(team_arms)
                    sess_arms = tuple(team_arms[(i + shift) % len(team_arms)] for i in range(len(team_arms)))

                    n_bots = len(obs_rows)
                    bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                    bot_arms = np.asarray([sess_arms[(int(t) - 1) % len(sess_arms)] if int(t) >= 1 else "default" for t in bot_teams])

                    cont_all = np.zeros((n_bots, PLAN_CONT_DIM), dtype=np.float32)
                    cart_all = np.zeros((n_bots,), dtype=np.int64)
                    rival_all = np.zeros((n_bots,), dtype=np.int64)
                    cell_all = np.zeros((n_bots,), dtype=np.int64)
                    logp_all = np.zeros((n_bots,), dtype=np.float32)
                    plan_matrix = np.zeros((harness.maxclients, PLAN_WIDTH), dtype=np.float32)
                    train_h_in: np.ndarray | None = None

                    obs_t = torch.as_tensor(obs_rows, dtype=torch.float32, device=device)
                    carts_t = torch.as_tensor(cart_rows, dtype=torch.float32, device=device)
                    teams_t = torch.as_tensor(team_rows, dtype=torch.float32, device=device)
                    evt_t = torch.as_tensor(np.array(evt_rows, copy=True), dtype=torch.float32, device=device) if len(evt_rows) else None
                    geo_t = _geo_to_torch(geo_np, device)

                    with torch.no_grad():
                        for arm_name, mdl in arm_models.items():
                            if mdl is None:
                                continue
                            sel = bot_arms == arm_name
                            if not np.any(sel):
                                continue
                            st_map = mealy_states[(session, arm_name)]
                            h_prev_np = np.zeros((n_bots, mdl.d), dtype=np.float32)
                            for bi in range(n_bots):
                                eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                                if obs_rows[bi, OBS["ALIVE"]] > 0.5 and eid in st_map:
                                    h_prev_np[bi] = st_map[eid]
                            h_prev_t = torch.as_tensor(h_prev_np, dtype=torch.float32, device=device)
                            out = mdl(obs_t, carts_t, teams_t, evt_t, geo=geo_t, h_prev=h_prev_t)
                            if arm_name == train_arm and out.fsq_codes is not None:
                                codes_np = out.fsq_codes.detach().cpu().numpy().astype(np.int64)
                                for bi in np.flatnonzero(sel):
                                    fsq_code_counts[tuple(int(v) for v in codes_np[bi])] += 1
                            samp = sample_plan(out, deterministic=(arm_name != train_arm))
                            h_next_np = out.h_next.detach().cpu().numpy()
                            for bi in range(n_bots):
                                eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                                if obs_rows[bi, OBS["ALIVE"]] > 0.5:
                                    st_map[eid] = h_next_np[bi]
                                else:
                                    st_map.pop(eid, None)
                            if arm_name == train_arm:
                                train_h_in = h_prev_np
                            cont_all[sel] = samp.cont.cpu().numpy()[sel]
                            cart_all[sel] = samp.cart_pick.cpu().numpy()[sel]
                            rival_all[sel] = samp.rival_pick.cpu().numpy()[sel]
                            cell_all[sel] = samp.cell_pick.cpu().numpy()[sel]
                            logp_all[sel] = samp.logp.cpu().numpy()[sel]
                            part = pack_plan_rows(
                                obs_rows, team_rows, samp, bot_arms, {arm_name},
                                harness.maxclients, cell_centroids=cell_centroids,
                                cart_lengths=cart_rows[:, CS["PATH_LENGTH"]],
                            )
                            mask_rows = part[:, 0] > 0
                            plan_matrix[mask_rows] = part[mask_rows]

                    dest = routes.get(session)
                    if dest is not None:
                        node, addr = dest
                        req_id = records[OBSERVATION_KIND][0]["req_id"]
                        engine.send(addr, node, frames(PLAN_KIND, req_id, tick_id, plan_matrix, harness.framebytes, session))
                    total_responses += 1

                    prev = prev_state.get(session)
                    if prev is not None and prev["episode"] == ep_id:
                        p_ids = prev["obs"][:, OBS["ID"]].astype(np.int64)
                        c_ids = obs_rows[:, OBS["ID"]].astype(np.int64)
                        id_to_idx = {int(eid): idx for idx, eid in enumerate(c_ids)}
                        succ = np.asarray([id_to_idx.get(int(eid), -1) for eid in p_ids], dtype=np.int64)
                        rew = role_rewards(prev["tick"].context, prev["tick"].cartstate, tick.cartstate)
                        prev_obs = prev["obs"]
                        push_delta = np.asarray([
                            max(0.0, float(obs_rows[idx, OBS["CART_PUSH"]]) - float(prev_obs[i, OBS["CART_PUSH"]])) if idx >= 0 else 0.0
                            for i, idx in enumerate(succ)
                        ], dtype=np.float32)
                        contest_delta = np.asarray([
                            max(0.0, float(obs_rows[idx, OBS["CART_CONTEST"]]) - float(prev_obs[i, OBS["CART_CONTEST"]])) if idx >= 0 else 0.0
                            for i, idx in enumerate(succ)
                        ], dtype=np.float32)
                        pickup_delta = np.asarray([
                            max(0.0, float(obs_rows[idx, OBS["PICKUPS"]]) - float(prev_obs[i, OBS["PICKUPS"]])) if idx >= 0 else 0.0
                            for i, idx in enumerate(succ)
                        ], dtype=np.float32)
                        p_carts = prev["carts"]
                        prev_frac = p_carts[:, CS["PATH_POSITION"]] / np.maximum(p_carts[:, CS["PATH_LENGTH"]], 1.0)
                        cur_frac = cart_rows[:, CS["PATH_POSITION"]] / np.maximum(cart_rows[:, CS["PATH_LENGTH"]], 1.0)
                        dfrac = cur_frac - prev_frac
                        ctrl_t = cart_rows[:, CS["CONTROL_TEAM"]].astype(np.int64)
                        p_teams_1 = prev_obs[:, OBS["TEAM"]].astype(np.int64)
                        lane_prog = np.zeros((len(prev_obs),), dtype=np.float32)
                        for ci in range(min(len(dfrac), len(ctrl_t))):
                            if ctrl_t[ci] > 0:
                                lane_prog += np.where(p_teams_1 == ctrl_t[ci], float(dfrac[ci]), -0.4 * float(dfrac[ci]))
                        rew = (
                            rew
                            + 0.25 * lane_prog
                            + 0.015 * np.tanh(push_delta / 25.0)
                            + 0.012 * np.tanh(contest_delta / 1.0)
                            + 0.008 * np.tanh(pickup_delta)
                        )
                        w_before = effective_leader(prev["tick"].context, prev["tick"].cartstate)
                        w_after = effective_leader(tick.context, tick.cartstate)
                        p_teams_0 = np.asarray(prev["tick"].context.team_of, dtype=np.int64)
                        wm = p_teams_0 == w_before
                        nwm = np.asarray([
                            (int(tick.context.team_of[idx]) == w_after) if idx >= 0 else False for idx in succ
                        ], dtype=bool)
                        disc = np.where((succ >= 0) & (~tick.cartstate.finished), gamma, 0.0).astype(np.float32)
                        actor_mask = (prev["arms"] == train_arm) & (succ >= 0)
                        if np.any(actor_mask):
                            tr = PlanTransition(
                                session=int(session),
                                obs_in=prev["obs"].copy(),
                                carts_in=prev["carts"].copy(),
                                teams_in=prev["teams"].copy(),
                                events_in=prev["events"].copy(),
                                obs_out=obs_rows.copy(),
                                carts_out=cart_rows.copy(),
                                teams_out=team_rows.copy(),
                                events_out=evt_rows[-64:].copy() if len(evt_rows) > 64 else evt_rows.copy(),
                                cont=prev["cont"].copy(),
                                cart_pick=prev["cart_pick"].copy(),
                                rival_pick=prev["rival_pick"].copy(),
                                cell_pick=prev["cell_pick"].copy(),
                                behavior_logp=prev["logp"].copy(),
                                reward=rew.astype(np.float32),
                                winner_mask=wm,
                                next_winner_mask=nwm,
                                discount=disc,
                                actor_mask=actor_mask,
                                successor_rows=succ,
                                geo_in=prev["geo"],
                                geo_out=geo_np,
                                h_in=prev["h_in"],
                                h_out=train_h_in,
                                map_name=sess_map_name,
                            )
                            fresh_buf.append(tr)
                            total_local_transitions += 1

                    prev_state[session] = {
                        "episode": ep_id,
                        "tick": tick,
                        "obs": obs_rows.copy(),
                        "carts": cart_rows.copy(),
                        "teams": team_rows.copy(),
                        "events": evt_rows[-64:].copy() if len(evt_rows) > 64 else evt_rows.copy(),
                        "geo": geo_np,
                        "h_in": train_h_in.copy() if train_h_in is not None else None,
                        "arms": bot_arms.copy(),
                        "cont": cont_all,
                        "cart_pick": cart_all,
                        "rival_pick": rival_all,
                        "cell_pick": cell_all,
                        "logp": logp_all,
                    }

            # All 4 Studios have collected local_batch_size transitions -> Synchronous RDMA Aurora Step!
            batch = fresh_buf[:local_batch_size]
            fresh_buf = fresh_buf[local_batch_size:]
            if not batch:
                break

            t_step0 = time.perf_counter()
            adv_mean, adv_std = _sync_adv_stats(model, batch, torch_mesh, world_size, device)
            last_diags = []
            last_gnorm = 0.0
            step_skipped = False
            for ep in range(max(1, ppo_epochs)):
                optimizer.zero_grad(set_to_none=True)
                inv_n = 1.0 / len(batch)
                for tr in batch:
                    comps, diag = plan_losses_for_transition(
                        model, tr, device=device, adv_mean=adv_mean, adv_std=adv_std,
                    )
                    (comps["total"] * inv_n).backward()
                    if ep == 0:
                        last_diags.append(diag)

                flat_g = torch_mesh.empty((total_params,), dtype=torch.float32, device=device)
                off_g = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        flat_g[off_g : off_g + sz].copy_(p.grad.detach().reshape(-1))
                    else:
                        flat_g[off_g : off_g + sz].zero_()
                    off_g += sz

                t_rdma0 = time.perf_counter()
                mean_g, gpair, grp_pairs = _rdma_mccandlish_from_flat(
                    flat_g, slices_by_group, torch_mesh, world_size, float(len(batch)), device, sync_steps,
                )
                if device == "mps":
                    torch.mps.synchronize()
                rdma_ms_list.append((time.perf_counter() - t_rdma0) * 1e3)

                if ep == 0:
                    snr_pairs.append(gpair)
                    for grp, gp in grp_pairs.items():
                        group_snr_pairs[grp].append(gp)
                    # If 4-machine signal_sq <= 0 on epoch 0, skip this noisy step cleanly across all 4 Studios
                    if float(gpair["signal_sq"]) <= 0.0:
                        deferred += 1
                        step_skipped = True
                        optimizer.zero_grad(set_to_none=True)
                        break

                off_g = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is None:
                        p.grad = mean_g[off_g : off_g + sz].view_as(p.data).clone()
                    else:
                        p.grad.copy_(mean_g[off_g : off_g + sz].view_as(p.data))
                    off_g += sz
                last_gnorm = edm2_step(optimizer, model, grad_clip=grad_clip)

            if not step_skipped:
                ema_update(ema_model, model, decay=ema_decay)
                if (sync_steps + 1) % 5 == 0:
                    _broadcast_model_(model, torch_mesh, rank, world_size, device)
                    _broadcast_model_(ema_model, torch_mesh, rank, world_size, device)
                def _m(k):
                    vals = [d[k] for d in last_diags if d.get(k) is not None]
                    return float(np.mean(vals)) if vals else None
                step_logs.append({
                    "step": sync_steps,
                    "global_batch": len(batch) * world_size,
                    "grad_norm": round(last_gnorm, 4),
                    "loss_total": _m("loss_total"),
                    "loss_actor": _m("loss_actor"),
                    "entropy": _m("entropy"),
                    "cart_temp": _m("cart_temp"),
                    "cell_temp": _m("cell_temp"),
                    "step_eff": gpair["step_efficiency"],
                    "cosine": gpair["cosine"],
                })
            step_ms_list.append((time.perf_counter() - t_step0) * 1e3)
            sync_steps += 1
    finally:
        harness._stop_servers(procs, logs)
        engine.close()

    # Final sync at end of wave
    _broadcast_model_(model, torch_mesh, rank, world_size, device)
    _broadcast_model_(ema_model, torch_mesh, rank, world_size, device)

    tot_c = sum(fsq_code_counts.values())
    fsq_ent = 0.0
    if tot_c > 0:
        probs = np.asarray(list(fsq_code_counts.values()), dtype=np.float64) / tot_c
        fsq_ent = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

    return {
        "wall_s": round(time.monotonic() - t_wave, 2),
        "sync_steps_attempted": sync_steps,
        "updates_applied": len(step_logs),
        "deferred_noisy_steps": deferred,
        "local_responses": total_responses,
        "local_transitions": total_local_transitions,
        "global_transitions_est": total_local_transitions * world_size,
        "mean_rdma_allreduce_ms": round(float(np.mean(rdma_ms_list)), 3) if rdma_ms_list else None,
        "mean_sync_step_ms": round(float(np.mean(step_ms_list)), 2) if step_ms_list else None,
        "snr_4studio": summarize_mccandlish_pairs(snr_pairs),
        "snr_4studio_by_group": {g: summarize_mccandlish_pairs(ps) for g, ps in group_snr_pairs.items() if ps},
        "fsq_unique_codes": len(fsq_code_counts),
        "fsq_entropy_bits": round(fsq_ent, 3),
        "weight_norms": weight_norm_report(ema_model),
        "last_step": step_logs[-1] if step_logs else None,
    }


def _audit_dead_neurons(model: PlanWally, harness: MultiServerPlanHarness, work_dir: Path, device: str, base_port: int) -> dict:
    """Measure exact per-neuron activation & gradient health across all 1,024 wide FFN units."""
    res = harness.run_episode_batch(
        work_dir / "neuron_audit",
        seeds=[9901, 9902],
        arm_models={"eval": model, "default": None},
        team_arms=("eval", "default", "default"),
        duration_s=10.0,
        rotate_seats=False,
        deterministic=True,
        device=device,
        base_port=base_port,
        map_names=["boil", "stormkeep"],
    )
    trs = res["transitions"][:12]
    if not trs:
        return {"error": "no transitions"}
    model.zero_grad(set_to_none=True)
    for tr in trs:
        comps, _ = plan_losses_for_transition(model, tr, device=device)
        (comps["total"] / len(trs)).backward()

    ffn_reports = []
    tot_units = 0
    tot_dead = 0
    for bi, blk in enumerate(model.blocks):
        g_up = blk.ffn_up.weight.grad.detach().cpu() if blk.ffn_up.weight.grad is not None else torch.zeros_like(blk.ffn_up.weight.cpu())
        rn = g_up.norm(dim=-1)
        med = float(rn.median().item())
        dead = int((rn < 0.01 * max(med, 1e-12)).sum().item())
        tot_units += int(rn.numel())
        tot_dead += dead
        ffn_reports.append({
            "block": bi,
            "units": int(rn.numel()),
            "dead_units": dead,
            "grad_norm_cv": round(float(rn.std().item() / max(float(rn.mean().item()), 1e-12)), 4),
            "min_over_med": round(float(rn.min().item() / max(med, 1e-12)), 4),
        })
    model.zero_grad(set_to_none=True)
    return {
        "total_wide_ffn_units": tot_units,
        "dead_units": tot_dead,
        "dead_fraction": round(tot_dead / max(1, tot_units), 6),
        "blocks": ffn_reports,
    }


def run_train_and_elo(args):
    device = args.device
    rank, world_size, torch_mesh = _init_mesh_world(device)
    stage_dir = Path(args.stage_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ckpt_dir = out_dir / "ckpts"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    harness = MultiServerPlanHarness(
        dp_bin=str(stage_dir / "darkplaces-dedicated"),
        basedir=str(stage_dir / "Xonotic"),
        userdir=str(stage_dir / "userdir"),
        framebytes=8192,
        strategy_interval=0.4,
        payload_speed=80.0,
        maxclients=16,
    )

    torch.manual_seed(2026)
    live = PlanWally(**MODEL_KWARGS).to(device)
    ema = clone_plan_model(live).to(device)
    _broadcast_model_(live, torch_mesh, rank, world_size, device)
    _broadcast_model_(ema, torch_mesh, rank, world_size, device)
    opt = make_aurora_optimizer(live, lr=args.lr, scalar_lr=args.scalar_lr, mu=0.90, pp_iterations=2, pp_beta=0.5)

    snapshots: dict[str, PlanWally] = {"aur_w0": clone_plan_model(ema).to(device)}
    if rank == 0:
        torch.save({"model": ema.state_dict(), "spec": ema.spec()}, ckpt_dir / "studio4_aurora_w0_ema.pt")

    wave_configs = [
        ("w1", ("train", "default", "default")),
        ("w2", ("train", "default", "aur_w1")),
        ("w3", ("train", "aur_w1", "aur_w2")),
        ("w4", ("train", "default", "aur_w3")),
        ("w5", ("train", "aur_w2", "aur_w4")),
    ][: args.waves]

    train_maps = ["boil", "boil", "stormkeep", "stormkeep", "catharsis", "catharsis"]
    wave_reports = []
    t_train0 = time.monotonic()

    for wi, (wname, team_arms) in enumerate(wave_configs, start=1):
        arm_models: dict[str, PlanWally | None] = {"train": live, "default": None}
        for a in team_arms:
            if a in snapshots:
                arm_models[a] = snapshots[a]
        seeds = [10000 + wi * 1000 + rank * 100 + j for j in range(len(train_maps))]
        w_rep = _run_sync_wave(
            harness,
            out_dir / f"work_r{rank}_{wname}",
            rank=rank,
            world_size=world_size,
            torch_mesh=torch_mesh,
            device=device,
            seeds=seeds,
            map_names=train_maps,
            base_port=args.base_port + wi * 10,
            arm_models=arm_models,
            team_arms=team_arms,
            train_arm="train",
            optimizer=opt,
            ema_model=ema,
            ema_decay=0.94,
            n_sync_steps=args.steps_per_wave,
            local_batch_size=args.local_batch,
            ppo_epochs=2,
            gamma=0.95,
            grad_clip=1.0,
        )
        snap_name = f"aur_{wname}"
        snapshots[snap_name] = clone_plan_model(ema).to(device)
        if rank == 0:
            torch.save({"model": ema.state_dict(), "spec": ema.spec()}, ckpt_dir / f"studio4_aurora_{wname}_ema.pt")
            w_rep["wave"] = wname
            w_rep["opponent_arms"] = list(team_arms)
            wave_reports.append(w_rep)
            snr = w_rep.get("snr_4studio") or {}
            print(
                f"[wave {wname}] wall={w_rep['wall_s']}s upd={w_rep['updates_applied']}/{w_rep['sync_steps_attempted']} "
                f"def={w_rep['deferred_noisy_steps']} eff={snr.get('step_efficiency')} B_noise={snr.get('b_noise_transitions')} "
                f"rdma_ms={w_rep['mean_rdma_allreduce_ms']} fsq_ent={w_rep['fsq_entropy_bits']}",
                flush=True,
            )

    train_wall_s = round(time.monotonic() - t_train0, 2)

    # ── Parallel 4-Studio Multi-Map CRN Elo Tournament (48 matches across 4 maps simultaneously!) ──
    # Rank 0: boil | Rank 1: stormkeep | Rank 2: catharsis | Rank 3: afterslime (ZERO-SHOT HELD-OUT!)
    eval_maps = ["boil", "stormkeep", "catharsis", "afterslime"]
    my_map = eval_maps[rank % len(eval_maps)]
    eval_models: dict[str, PlanWally | None] = {"default": None, **snapshots}
    # Balanced round-robin triplets covering default + aur_w0..aur_w5 (each wave meets default + adjacent/final waves)
    avail = [f"aur_w{k}" for k in range(args.waves + 1)]
    triplets = [
        ("default", avail[0], avail[-1]),
        ("default", avail[1] if len(avail) > 1 else avail[0], avail[-1]),
        ("default", avail[2] if len(avail) > 2 else avail[0], avail[-2] if len(avail) > 2 else avail[-1]),
        (avail[1] if len(avail) > 1 else "default", avail[3] if len(avail) > 3 else avail[-1], avail[-1]),
    ]
    t_elo0 = time.monotonic()
    local_score_matches = []
    local_comp_matches = []
    local_fsq_eval = {}
    for ti, trip in enumerate(triplets):
        # 3 seat-rotated servers per triplet = 12 matches per map = 48 matches total across 4 Studios!
        e_seeds = [50000 + ti * 100 + j for j in range(3)]
        e_res = harness.run_episode_batch(
            out_dir / f"elo_r{rank}_{my_map}_t{ti}",
            seeds=e_seeds,
            arm_models=eval_models,
            team_arms=trip,
            duration_s=args.elo_duration_s,
            rotate_seats=True,
            deterministic=True,
            device=device,
            base_port=args.base_port + 200 + ti * 10,
            map_names=[my_map] * len(e_seeds),
        )
        local_score_matches.extend(e_res["score_matches"])
        local_comp_matches.extend(e_res["composite_matches"])
        for a, st in (e_res.get("fsq_stats") or {}).items():
            local_fsq_eval[a] = st

    local_payload = {
        "rank": rank,
        "host": os.uname().nodename,
        "map": my_map,
        "score_matches": local_score_matches,
        "composite_matches": local_comp_matches,
        "fsq_eval": local_fsq_eval,
        "elo_wall_s": round(time.monotonic() - t_elo0, 2),
    }

    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    # Dead-neuron & weight-norm audit on rank 0
    neuron_audit = None
    if rank == 0:
        neuron_audit = _audit_dead_neurons(ema, harness, out_dir / "audit_r0", device, args.base_port + 300)

        all_sc = []
        all_comp = []
        per_map_elo = {}
        for g in gathered:
            all_sc.extend(g["score_matches"])
            all_comp.extend(g["composite_matches"])
            per_map_elo[g["map"]] = {
                "host": g["host"],
                "matches": len(g["score_matches"]),
                "score_elo": compute_multiteam_elo(g["score_matches"], anchor_arm="default"),
                "composite_elo": compute_multiteam_elo(g["composite_matches"], anchor_arm="default"),
            }
        overall_sc_elo = compute_multiteam_elo(all_sc, anchor_arm="default")
        overall_comp_elo = compute_multiteam_elo(all_comp, anchor_arm="default")
        comp_ratings = overall_comp_elo.get("ratings") or {}

        # McCandlish S_eff vs EMA Checkpoint Elo correlation
        cum_seff = []
        elo_pts = []
        running = 0.0
        for wi, wrep in enumerate(wave_reports, start=1):
            snr = wrep.get("snr_4studio") or {}
            eff = float(snr.get("step_efficiency") or 0.0)
            running += eff * float(wrep.get("updates_applied", 0))
            ck = f"aur_w{wi}"
            if ck in comp_ratings:
                cum_seff.append(running)
                elo_pts.append(float(comp_ratings[ck]["elo"]))
        corr_r = None
        if len(cum_seff) >= 3 and np.std(cum_seff) > 1e-9 and np.std(elo_pts) > 1e-9:
            corr_r = round(float(np.corrcoef(cum_seff, elo_pts)[0, 1]), 4)

        summary = {
            "mode": "train_and_elo",
            "world_size": world_size,
            "device": device,
            "servers_concurrent": 6 * world_size,
            "model_spec": ema.spec(),
            "param_count": sum(p.numel() for p in ema.parameters()),
            "train_wall_s": train_wall_s,
            "total_updates_applied": sum(w["updates_applied"] for w in wave_reports),
            "total_deferred_noisy": sum(w["deferred_noisy_steps"] for w in wave_reports),
            "total_global_transitions_est": sum(w["global_transitions_est"] for w in wave_reports),
            "waves": wave_reports,
            "final_weight_norms": weight_norm_report(ema),
            "final_neuron_audit": neuron_audit,
            "elo_total_matches": len(all_sc),
            "elo_overall_score": overall_sc_elo,
            "elo_overall_composite": overall_comp_elo,
            "elo_by_map": per_map_elo,
            "snr_seff_vs_comp_elo_pearson_r": corr_r,
            "cum_seff_trajectory": [round(v, 3) for v in cum_seff],
            "comp_elo_trajectory": [round(v, 2) for v in elo_pts],
        }
        out_path = out_dir / "studio4_aurora_report.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print("REPORT_WRITTEN:", str(out_path), flush=True)
        print(json.dumps({
            "train_wall_s": summary["train_wall_s"],
            "total_updates_applied": summary["total_updates_applied"],
            "elo_total_matches": summary["elo_total_matches"],
            "elo_overall_score": summary["elo_overall_score"],
            "elo_overall_composite": summary["elo_overall_composite"],
            "afterslime_zeroshot": per_map_elo.get("afterslime", {}),
            "dead_fraction": (neuron_audit or {}).get("dead_fraction"),
            "max_dev_from_1": summary["final_weight_norms"].get("max_dev_from_1"),
            "snr_seff_vs_comp_elo_pearson_r": corr_r,
        }, indent=2), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tp_bench", "train_and_elo"], required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--out-dir", default="~/xonotic-runs/studio4-aurora")
    ap.add_argument("--waves", type=int, default=5)
    ap.add_argument("--steps-per-wave", type=int, default=18)
    ap.add_argument("--local-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--scalar-lr", type=float, default=2.5e-3)
    ap.add_argument("--base-port", type=int, default=26000)
    ap.add_argument("--elo-duration-s", type=float, default=24.0)
    args = ap.parse_args()

    if args.mode == "tp_bench":
        run_tp_bench(args.device, Path(args.out_dir).expanduser() / "tp_bench.json")
    else:
        run_train_and_elo(args)


if __name__ == "__main__":
    main()
