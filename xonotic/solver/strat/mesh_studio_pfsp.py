#!/usr/bin/env python3
"""Four-Studio PFSP + alpha-Rank + Skewed-Mechanical-Roster + Masked-Actor/Skill-ID Aurora Runner.

Runs identically on all 4 Apple M5 Ultra Mac Studios over live Thunderbolt RDMA
(`torch.distributed` `backend="mesh"`, `device="mps:0"`):
  - 24 concurrent Xonotic servers (6/Studio: 2xboil, 2xstormkeep, 2xcatharsis with lane-scaled slowmo)
  - Skewed mechanical rosters (`bots.txt` calibrated via 1v1 DM ladder, `skill=7`)
  - Masked actor (`BOT_CONFIGURATION_COLUMNS` zeroed) + Asymmetric Critic + Auxiliary Online Skill-ID head
  - PFSP opponent sampling over permanent anchors (`default` + 4 scripted archetypes) + league history
    + dedicated Main Exploiter wave + FCP mixed-teammate squads + physics perturbations
  - 4-Studio parallel 48-match Multi-Map CRN Elo + 3-team alpha-Rank + Underdog Roster Stress Test
    + Online Skill-ID inference correlation audit.
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

from payload.tools.strategy_io_schema import CS, OBS, TS
from solver.strat.aurora import Aurora
from solver.strat.buffers import ObservationMemory
from solver.strat.curriculum import PERTURBATIONS
from solver.strat.game_value import effective_leader, formal_value_record, role_rewards
from solver.strat.grad_snr import summarize_mccandlish_pairs
from solver.strat.league_curriculum import (
    SCRIPTED_ANCHORS,
    alpha_rank_3team,
    sample_pfsp_server_assignments,
    scripted_plan_rows,
)
from solver.strat.mech_calibration import (
    extract_privileged_skill_target,
    sample_skewed_roster,
    write_bots_txt,
)
from solver.strat.outer_validation import compute_multiteam_elo, extract_behavioral_motifs
from solver.strat.plan_policy import (
    PLAN_CONT_DIM,
    PLAN_WIDTH,
    PlanTransition,
    PlanWally,
    _geo_to_torch,
    clone_plan_model,
    ema_update,
    pack_plan_rows,
    plan_losses_for_transition,
    plan_module_group,
    project_unit_norm_,
    sample_plan,
    weight_norm_report,
)
from solver.strat.plan_runner import MultiServerPlanHarness
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
MODULE_GROUPS = ("encoder", "gram_cgt", "resnet_ffn", "action_head", "value_head", "skill_id_head", "dynamics_head")
MAP_SLOWMO = {"boil": 1.0, "stormkeep": 1.2, "catharsis": 1.8, "afterslime": 1.8}


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


def _make_grouped_aurora(model: PlanWally, lr: float, scalar_lr: float) -> Aurora:
    grouped: dict[str, list[torch.nn.Parameter]] = {g: [] for g in MODULE_GROUPS}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grouped.setdefault(plan_module_group(n), []).append(p)
    active = [(g, ps) for g, ps in grouped.items() if ps]
    opt = Aurora(
        [{"params": ps, "lr": lr, "scalar_lr": scalar_lr} for _, ps in active],
        lr=lr,
        scalar_lr=scalar_lr,
        mu=0.90,
        nesterov=True,
        pp_iterations=2,
        pp_beta=0.5,
    )
    opt._group_names = [g for g, _ in active]
    return opt


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
            priv_in = torch.as_tensor(tr.priv_skill_in, dtype=torch.float32, device=device) if tr.priv_skill_in is not None else None

            obs_out = torch.as_tensor(tr.obs_out, dtype=torch.float32, device=device)
            carts_out = torch.as_tensor(tr.carts_out, dtype=torch.float32, device=device)
            teams_out = torch.as_tensor(tr.teams_out, dtype=torch.float32, device=device)
            evt_out = torch.as_tensor(np.array(tr.events_out, copy=True), dtype=torch.float32, device=device) if len(tr.events_out) else None
            geo_out = _geo_to_torch(tr.geo_out, device)
            h_out = torch.as_tensor(tr.h_out, dtype=torch.float32, device=device) if tr.h_out is not None else None
            priv_out = torch.as_tensor(tr.priv_skill_out, dtype=torch.float32, device=device) if tr.priv_skill_out is not None else None

            cur = model(obs_in, carts_in, teams_in, evt_in, geo=geo_in, h_prev=h_in, priv_skill=priv_in)
            nxt = model(obs_out, carts_out, teams_out, evt_out, geo=geo_out, h_prev=h_out, priv_skill=priv_out)
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
    r = float(world_size)
    local_sq = torch_mesh.empty((1 + len(MODULE_GROUPS),), dtype=torch.float32, device=device)
    local_sq[0] = flat_g.square().sum()
    for gi, grp in enumerate(MODULE_GROUPS):
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
        mean_local_sq = sum_rank_sq_norms / r
        if world_size <= 1:
            return {
                "generation": step_id, "replicas": 1,
                "mean_local_norm": math.sqrt(max(0.0, mean_local_sq)),
                "global_norm": math.sqrt(max(0.0, mean_sq_norm)),
                "signal_sq": mean_sq_norm, "noise_sq": 0.0, "cosine": 1.0,
                "b0_transitions": float(b0_transitions), "b0_rows": float(b0_transitions * 9.0),
                "step_efficiency": 1.0,
            }
        noise_sq = max(0.0, (r / (r - 1.0)) * (mean_local_sq - mean_sq_norm))
        signal_sq = (r * mean_sq_norm - mean_local_sq) / (r - 1.0)
        b_eff = r * b0_transitions
        if signal_sq > 1e-18:
            b_noise = b0_transitions * noise_sq / signal_sq
            eff = b_eff / (b_eff + b_noise)
        else:
            eff = 0.0
        cos = signal_sq / mean_local_sq if mean_local_sq > 1e-18 else 0.0
        return {
            "generation": step_id, "replicas": int(world_size),
            "mean_local_norm": math.sqrt(max(0.0, mean_local_sq)),
            "global_norm": math.sqrt(max(0.0, mean_sq_norm)),
            "signal_sq": float(signal_sq), "noise_sq": float(noise_sq),
            "cosine": float(max(-1.0, min(1.0, cos))),
            "b0_transitions": float(b0_transitions), "b0_rows": float(b0_transitions * 9.0),
            "step_efficiency": float(eff),
        }

    sq_cpu = local_sq.cpu().tolist()
    tot_mean_sq = float(mean_g.square().sum().item())
    global_pair = _pair_from_sums(tot_mean_sq, float(sq_cpu[0]))

    grp_pairs = {}
    for gi, grp in enumerate(MODULE_GROUPS):
        if not slices_by_group[grp]:
            continue
        gm_sq = 0.0
        for lo, hi in slices_by_group[grp]:
            gm_sq += float(mean_g[lo:hi].square().sum().item())
        grp_pairs[grp] = _pair_from_sums(gm_sq, float(sq_cpu[1 + gi]))
    return mean_g, global_pair, grp_pairs


def _run_pfsp_sync_wave(
    harness: MultiServerPlanHarness,
    work_dir: Path,
    *,
    rank: int,
    world_size: int,
    torch_mesh,
    device: str,
    seeds: list[int],
    map_names: list[str],
    server_configs: list[dict],
    base_port: int,
    arm_models: dict[str, PlanWally | None],
    train_arm: str,
    optimizer: Aurora,
    ema_model: PlanWally,
    ema_decay: float,
    n_sync_steps: int,
    local_batch_size: int,
    entropy_coef: float,
    gamma: float = 0.95,
    grad_clip: float = 1.0,
    skill: int = 7,
    rng_seed: int = 1000,
    max_wave_s: float = 320.0,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    # Generate per-server skewed mechanical bots.txt + physics perturbations + lane-scaled slowmo!
    bot_cfg_files: list[Path] = []
    extra_cvars: list[dict[str, float]] = []
    slowmos: list[float] = []
    roster_kinds: list[str] = []
    for idx, (mname, scfg) in enumerate(zip(map_names, server_configs)):
        r_kind, entries = sample_skewed_roster(rng)
        roster_kinds.append(r_kind)
        bfile = work_dir / f"roster_{idx}_{r_kind}.txt"
        write_bots_txt(bfile, entries)
        bot_cfg_files.append(bfile)
        extra_cvars.append(dict(PERTURBATIONS.get(scfg["perturbation"], {})))
        slowmos.append(float(MAP_SLOWMO.get(mname, 1.0)))

    engine = Engine("udp://127.0.0.1:0", harness.framebytes)
    sock = engine.path
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    procs, logs = harness._spawn_servers(
        work_dir, sock, seeds, base_port,
        map_names=map_names,
        bot_config_files=bot_cfg_files,
        skill=skill,
        extra_cvars=extra_cvars,
        slowmo_per_server=slowmos,
    )

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
    telemetry_by_sess: dict[int, list[dict]] = defaultdict(list)
    fresh_buf: list[PlanTransition] = []

    sync_steps = 0
    total_responses = 0
    total_local_transitions = 0
    snr_pairs: list[dict] = []
    group_snr_pairs: dict[str, list[dict]] = defaultdict(list)
    step_logs: list[dict] = []
    rdma_ms_list: list[float] = []
    skill_pred_pairs: list[tuple[float, float]] = []
    t_wave = time.monotonic()

    try:
        while sync_steps < n_sync_steps:
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
                    scfg = server_configs[s_idx % len(server_configs)]
                    base_team_arms = scfg["team_arms"]
                    fcp_mixed = bool(scfg["fcp_mixed"])

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
                    priv_skill_np = extract_privileged_skill_target(obs_rows)

                    tick = build_runtime_frame(obs_rows, cart_rows, team_rows)
                    n_teams = len(team_rows)
                    shift = (s_idx + rank) % len(base_team_arms)
                    sess_arms = tuple(base_team_arms[(i + shift) % len(base_team_arms)] for i in range(len(base_team_arms)))

                    n_bots = len(obs_rows)
                    bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                    bot_arms = np.asarray([sess_arms[(int(t) - 1) % len(sess_arms)] if int(t) >= 1 else "default" for t in bot_teams], dtype=object)

                    # FCP mixed-teammate: on designated FCP servers, make 1 bot on train's team run stock `default`!
                    if fcp_mixed:
                        train_indices = [bi for bi in range(n_bots) if bot_arms[bi] == train_arm]
                        if len(train_indices) >= 2:
                            bot_arms[train_indices[0]] = "default"

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

                    # Scripted permanent anchor rows first
                    proj_leader = tick.game_value.projected_role
                    for s_arm in set(str(a) for a in bot_arms):
                        if s_arm in SCRIPTED_ANCHORS:
                            s_part = scripted_plan_rows(
                                s_arm, obs_rows, cart_rows, team_rows, bot_arms,
                                harness.maxclients, cell_centroids=cell_centroids,
                                projected_leader_idx=proj_leader,
                            )
                            m_rows = s_part[:, 0] > 0
                            plan_matrix[m_rows] = s_part[m_rows]

                    # Neural policy arms (actor NEVER receives priv_skill!)
                    with torch.no_grad():
                        for arm_name in sorted(set(str(a) for a in bot_arms)):
                            mdl = arm_models.get(arm_name)
                            if mdl is None:
                                continue
                            sel = np.asarray([str(a) == arm_name for a in bot_arms], dtype=bool)
                            if not np.any(sel):
                                continue
                            st_map = mealy_states[(session, arm_name)]
                            h_prev_np = np.zeros((n_bots, mdl.d), dtype=np.float32)
                            for bi in range(n_bots):
                                eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                                if obs_rows[bi, OBS["ALIVE"]] > 0.5 and eid in st_map:
                                    h_prev_np[bi] = st_map[eid]
                            h_prev_t = torch.as_tensor(h_prev_np, dtype=torch.float32, device=device)
                            out = mdl(obs_t, carts_t, teams_t, evt_t, geo=geo_t, h_prev=h_prev_t, priv_skill=None)
                            if arm_name == train_arm:
                                if out.fsq_codes is not None:
                                    codes_np = out.fsq_codes.detach().cpu().numpy().astype(np.int64)
                                    for bi in np.flatnonzero(sel):
                                        fsq_code_counts[tuple(int(v) for v in codes_np[bi])] += 1
                                if out.pred_skill is not None and len(skill_pred_pairs) < 4000:
                                    ps_hat = out.pred_skill.detach().cpu().numpy()
                                    for bi in range(n_bots):
                                        skill_pred_pairs.append((float(ps_hat[bi]), float(priv_skill_np[bi])))
                            samp = sample_plan(out, deterministic=False)
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

                    # Track match telemetry for live PFSP win-rate updates
                    if len(telemetry_by_sess[session]) < 260:
                        telemetry_by_sess[session].append({
                            "t": time.monotonic() - t_wave,
                            "teams": n_teams,
                            "team_policy_arms": list(sess_arms),
                            "carts": [{name.lower(): float(row[col]) for name, col in CS.items()} for row in cart_rows],
                            "assignments": [{"edict": int(obs_rows[i, OBS["ID"]]), "team": int(bot_teams[i]), "arm": str(bot_arms[i])} for i in range(n_bots)],
                            "realized_events": [],
                            "resources": [],
                            "game_value": formal_value_record(tick.game_value, tick.cartstate),
                            "learning": {},
                            "policy_updates": {},
                        })

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
                        disc = np.where((succ >= 0) & (not tick.cartstate.finished), gamma, 0.0).astype(np.float32)
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
                                priv_skill_in=prev["priv_skill"].copy(),
                                priv_skill_out=priv_skill_np.copy(),
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
                        "priv_skill": priv_skill_np.copy(),
                    }

            batch = fresh_buf[:local_batch_size]
            fresh_buf = []
            if not batch:
                break

            adv_mean, adv_std = _sync_adv_stats(model, batch, torch_mesh, world_size, device)
            optimizer.zero_grad(set_to_none=True)
            inv_n = 1.0 / len(batch)
            last_diags = []
            for tr in batch:
                comps, diag = plan_losses_for_transition(
                    model, tr, device=device,
                    entropy_coef=entropy_coef,
                    adv_mean=adv_mean, adv_std=adv_std,
                )
                (comps["total"] * inv_n).backward()
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

            snr_pairs.append(gpair)
            for grp, gp in grp_pairs.items():
                group_snr_pairs[grp].append(gp)

            grp_scales: dict[int, float] = {}
            for gi, grp_name in enumerate(optimizer._group_names):
                gp = grp_pairs.get(grp_name)
                if gp is None:
                    grp_scales[gi] = 0.0
                else:
                    eff_g = float(gp.get("step_efficiency") or 0.0)
                    grp_scales[gi] = max(0.15, math.sqrt(eff_g)) if float(gp.get("signal_sq") or 0.0) > 0.0 else 0.0

            if any(v > 0.0 for v in grp_scales.values()):
                off_g = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is None:
                        p.grad = mean_g[off_g : off_g + sz].view_as(p.data).clone()
                    else:
                        p.grad.copy_(mean_g[off_g : off_g + sz].view_as(p.data))
                    off_g += sz
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
                optimizer.step(group_lr_scales=grp_scales)
                project_unit_norm_(model)
                ema_update(ema_model, model, decay=ema_decay)
                if (sync_steps + 1) % 5 == 0:
                    _broadcast_model_(model, torch_mesh, rank, world_size, device)
                    _broadcast_model_(ema_model, torch_mesh, rank, world_size, device)
                def _m(k):
                    vals = [d[k] for d in last_diags if d.get(k) is not None]
                    return float(np.mean(vals)) if vals else None
                step_logs.append({
                    "step": sync_steps,
                    "grad_norm": round(gnorm, 4),
                    "loss_total": _m("loss_total"),
                    "skill_id_mse": _m("skill_id_mse"),
                    "entropy": _m("entropy"),
                    "step_eff": gpair["step_efficiency"],
                })
            sync_steps += 1
    finally:
        harness._stop_servers(procs, logs)
        engine.close()

    _broadcast_model_(model, torch_mesh, rank, world_size, device)
    _broadcast_model_(ema_model, torch_mesh, rank, world_size, device)

    # Extract local match outcomes vs each opponent for PFSP win-rate updates
    local_comp_matches = []
    for sess_id, recs in sorted(telemetry_by_sess.items()):
        mot = extract_behavioral_motifs(recs)
        comp_map = {}
        for _, ti in (mot.get("per_team") or {}).items():
            a = ti["arm"]
            comp_map[a] = float(ti["score_end"]) + 5.0 * int(ti["max_checkpoints_held"]) + 0.1 * int(ti["cart_push_ticks"])
        if len(comp_map) >= 2:
            local_comp_matches.append(comp_map)

    tot_c = sum(fsq_code_counts.values())
    fsq_ent = 0.0
    if tot_c > 0:
        probs = np.asarray(list(fsq_code_counts.values()), dtype=np.float64) / tot_c
        fsq_ent = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

    skill_corr = None
    if len(skill_pred_pairs) >= 20:
        arr = np.asarray(skill_pred_pairs, dtype=np.float64)
        if arr[:, 0].std() > 1e-6 and arr[:, 1].std() > 1e-6:
            skill_corr = round(float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]), 4)

    return {
        "wall_s": round(time.monotonic() - t_wave, 2),
        "updates_applied": len(step_logs),
        "sync_steps_attempted": sync_steps,
        "local_transitions": total_local_transitions,
        "global_transitions_est": total_local_transitions * world_size,
        "mean_rdma_allreduce_ms": round(float(np.mean(rdma_ms_list)), 3) if rdma_ms_list else None,
        "snr_4studio": summarize_mccandlish_pairs(snr_pairs),
        "snr_4studio_by_group": {g: summarize_mccandlish_pairs(ps) for g, ps in group_snr_pairs.items() if ps},
        "fsq_unique_codes": len(fsq_code_counts),
        "fsq_entropy_bits": round(fsq_ent, 3),
        "online_skill_id_corr": skill_corr,
        "roster_kinds": roster_kinds,
        "local_comp_matches": local_comp_matches,
        "last_step": step_logs[-1] if step_logs else None,
    }


def _eval_triplet_with_scripted(
    harness: MultiServerPlanHarness,
    work_dir: Path,
    *,
    seeds: list[int],
    arm_models: dict[str, PlanWally | None],
    team_arms: tuple[str, ...],
    mapname: str,
    duration_s: float,
    device: str,
    base_port: int,
    bot_config_files: list[Path | None] | None = None,
    skill: int = 7,
    rotate_seats: bool = True,
) -> dict:
    """Evaluate a 3-team triplet (supporting neural policies, `default`, and permanent scripted
    anchors) and also collect masked-actor online Skill-ID predictions."""
    work_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine("udp://127.0.0.1:0", harness.framebytes)
    sock = engine.path
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    slowmos = [float(MAP_SLOWMO.get(mapname, 1.0))] * len(seeds)
    procs, logs = harness._spawn_servers(
        work_dir, sock, seeds, base_port,
        map_names=[mapname] * len(seeds),
        bot_config_files=bot_config_files,
        skill=skill,
        slowmo_per_server=slowmos,
    )
    routes: dict[int, tuple[int, object]] = {}
    sess_order: dict[int, int] = {}
    sess_maps: dict[int, tuple[str, object]] = {}
    obs_mems: dict[int, ObservationMemory] = {}
    mealy_states: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)
    telemetry_by_sess: dict[int, list[dict]] = defaultdict(list)
    skill_preds_early: list[tuple[float, float]] = []
    skill_preds_late: list[tuple[float, float]] = []
    t_start = time.monotonic()

    try:
        while time.monotonic() - t_start < duration_s:
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
                obs_rows = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
                if len(obs_rows) == 0 or len(cart_rows) == 0 or len(team_rows) == 0:
                    continue
                if session not in sess_maps:
                    sess_maps[session] = harness._identify_map(cart_rows)
                _, sess_manifold = sess_maps[session]
                cell_centroids = sess_manifold.cell_centroids if sess_manifold is not None else None
                ep_id = ObservationMemory.identity(session, team_rows[0, TS["EPISODE"]])
                mem = obs_mems[session]
                mem.observe(ep_id, event_batches)
                evt_rows = mem.rows
                geo_np = sess_manifold.build_features(obs_rows, cart_rows, team_rows, evt_rows) if sess_manifold is not None else None
                priv_skill_np = extract_privileged_skill_target(obs_rows)

                tick = build_runtime_frame(obs_rows, cart_rows, team_rows)
                n_teams = len(team_rows)
                shift = s_idx if rotate_seats and len(team_arms) > 1 else 0
                sess_arms = tuple(team_arms[(i + shift) % len(team_arms)] for i in range(len(team_arms)))
                n_bots = len(obs_rows)
                bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                bot_arms = np.asarray([sess_arms[(int(t) - 1) % len(sess_arms)] if int(t) >= 1 else "default" for t in bot_teams], dtype=object)

                plan_matrix = np.zeros((harness.maxclients, PLAN_WIDTH), dtype=np.float32)
                for s_arm in set(str(a) for a in bot_arms):
                    if s_arm in SCRIPTED_ANCHORS:
                        s_part = scripted_plan_rows(
                            s_arm, obs_rows, cart_rows, team_rows, bot_arms,
                            harness.maxclients, cell_centroids=cell_centroids,
                            projected_leader_idx=tick.game_value.projected_role,
                        )
                        m_rows = s_part[:, 0] > 0
                        plan_matrix[m_rows] = s_part[m_rows]

                obs_t = torch.as_tensor(obs_rows, dtype=torch.float32, device=device)
                carts_t = torch.as_tensor(cart_rows, dtype=torch.float32, device=device)
                teams_t = torch.as_tensor(team_rows, dtype=torch.float32, device=device)
                evt_t = torch.as_tensor(np.array(evt_rows, copy=True), dtype=torch.float32, device=device) if len(evt_rows) else None
                geo_t = _geo_to_torch(geo_np, device)

                elapsed_s = time.monotonic() - t_start
                with torch.no_grad():
                    for arm_name in sorted(set(str(a) for a in bot_arms)):
                        mdl = arm_models.get(arm_name)
                        if mdl is None:
                            continue
                        sel = np.asarray([str(a) == arm_name for a in bot_arms], dtype=bool)
                        if not np.any(sel):
                            continue
                        st_map = mealy_states[(session, arm_name)]
                        h_prev_np = np.zeros((n_bots, mdl.d), dtype=np.float32)
                        for bi in range(n_bots):
                            eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                            if obs_rows[bi, OBS["ALIVE"]] > 0.5 and eid in st_map:
                                h_prev_np[bi] = st_map[eid]
                        out = mdl(obs_t, carts_t, teams_t, evt_t, geo=geo_t, h_prev=torch.as_tensor(h_prev_np, dtype=torch.float32, device=device), priv_skill=None)
                        if out.pred_skill is not None and arm_name == "pfsp_w5":
                            ph = out.pred_skill.detach().cpu().numpy()
                            dst_list = skill_preds_early if elapsed_s < 8.0 else skill_preds_late
                            if len(dst_list) < 2000:
                                for bi in range(n_bots):
                                    dst_list.append((float(ph[bi]), float(priv_skill_np[bi])))
                        samp = sample_plan(out, deterministic=False)
                        h_nxt = out.h_next.detach().cpu().numpy()
                        for bi in range(n_bots):
                            eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                            if obs_rows[bi, OBS["ALIVE"]] > 0.5:
                                st_map[eid] = h_nxt[bi]
                            else:
                                st_map.pop(eid, None)
                        part = pack_plan_rows(
                            obs_rows, team_rows, samp, bot_arms, {arm_name},
                            harness.maxclients, cell_centroids=cell_centroids,
                            cart_lengths=cart_rows[:, CS["PATH_LENGTH"]],
                        )
                        m_rows = part[:, 0] > 0
                        plan_matrix[m_rows] = part[m_rows]

                dest = routes.get(session)
                if dest is not None:
                    node, addr = dest
                    req_id = records[OBSERVATION_KIND][0]["req_id"]
                    engine.send(addr, node, frames(PLAN_KIND, req_id, tick_id, plan_matrix, harness.framebytes, session))
                telemetry_by_sess[session].append({
                    "t": elapsed_s,
                    "teams": n_teams,
                    "team_policy_arms": list(sess_arms),
                    "carts": [{name.lower(): float(row[col]) for name, col in CS.items()} for row in cart_rows],
                    "assignments": [{"edict": int(obs_rows[i, OBS["ID"]]), "team": int(bot_teams[i]), "arm": str(bot_arms[i])} for i in range(n_bots)],
                    "realized_events": [],
                    "resources": [],
                    "game_value": formal_value_record(tick.game_value, tick.cartstate),
                    "learning": {},
                    "policy_updates": {},
                })
    finally:
        harness._stop_servers(procs, logs)
        engine.close()

    score_matches, comp_matches = [], []
    for sess_id, recs in sorted(telemetry_by_sess.items()):
        mot = extract_behavioral_motifs(recs)
        sc_m, cp_m = {}, {}
        for _, ti in (mot.get("per_team") or {}).items():
            a = ti["arm"]
            sc = float(ti["score_end"])
            sc_m[a] = sc
            cp_m[a] = sc + 5.0 * int(ti["max_checkpoints_held"]) + 0.1 * int(ti["cart_push_ticks"])
        if len(sc_m) >= 2:
            score_matches.append(sc_m)
            comp_matches.append(cp_m)
    return {
        "score_matches": score_matches,
        "composite_matches": comp_matches,
        "skill_preds_early": skill_preds_early,
        "skill_preds_late": skill_preds_late,
    }


def _audit_dead_neurons(model: PlanWally, harness: MultiServerPlanHarness, work_dir: Path, device: str, base_port: int) -> dict:
    res = harness.run_episode_batch(
        work_dir / "neuron_audit",
        seeds=[9901, 9902],
        arm_models={"eval": model, "default": None},
        team_arms=("eval", "default", "default"),
        duration_s=10.0,
        rotate_seats=False,
        deterministic=False,
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
    ffn_reports, tot_units, tot_dead = [], 0, 0
    for bi, blk in enumerate(model.blocks):
        g_up = blk.ffn_up.weight.grad.detach().cpu() if blk.ffn_up.weight.grad is not None else torch.zeros_like(blk.ffn_up.weight.cpu())
        rn = g_up.norm(dim=-1)
        med = float(rn.median().item())
        dead = int((rn < 0.01 * max(med, 1e-12)).sum().item())
        tot_units += int(rn.numel())
        tot_dead += dead
        ffn_reports.append({
            "block": bi, "units": int(rn.numel()), "dead_units": dead,
            "grad_norm_cv": round(float(rn.std().item() / max(float(rn.mean().item()), 1e-12)), 4),
            "min_over_med": round(float(rn.min().item() / max(med, 1e-12)), 4),
        })
    model.zero_grad(set_to_none=True)
    return {"total_wide_ffn_units": tot_units, "dead_units": tot_dead, "dead_fraction": round(tot_dead / max(1, tot_units), 6), "blocks": ffn_reports}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--out-dir", default="~/xonotic-runs/studio4-pfsp-mech")
    ap.add_argument("--steps-per-wave", type=int, default=16)
    ap.add_argument("--local-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--scalar-lr", type=float, default=4e-3)
    ap.add_argument("--base-port", type=int, default=26000)
    ap.add_argument("--elo-duration-s", type=float, default=32.0)
    args = ap.parse_args()

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
    main_live = PlanWally(**MODEL_KWARGS).to(device)
    main_ema = clone_plan_model(main_live).to(device)
    _broadcast_model_(main_live, torch_mesh, rank, world_size, device)
    _broadcast_model_(main_ema, torch_mesh, rank, world_size, device)
    main_opt = _make_grouped_aurora(main_live, lr=args.lr, scalar_lr=args.scalar_lr)

    snapshots: dict[str, PlanWally] = {"pfsp_w0": clone_plan_model(main_ema).to(device)}
    if rank == 0:
        torch.save({"model": main_ema.state_dict(), "spec": main_ema.spec()}, ckpt_dir / "pfsp_w0_ema.pt")

    train_maps = ["boil", "boil", "stormkeep", "stormkeep", "catharsis", "catharsis"]
    win_rates_vs_opp: dict[str, float] = {}
    pfsp_wins: dict[str, float] = defaultdict(float)
    pfsp_games: dict[str, float] = defaultdict(float)
    wave_reports = []
    t_train0 = time.monotonic()

    # 5 Main PFSP Waves + 1 Dedicated Main Exploiter wave inserted after Wave 3!
    schedule = [
        ("w1", "main", 0.0100),
        ("w2", "main", 0.0075),
        ("w3", "main", 0.0050),
        ("exp1", "exploiter", 0.0080),
        ("w4", "main", 0.0035),
        ("w5", "main", 0.0025),
    ]

    for wi, (wname, role, ent_coef) in enumerate(schedule, start=1):
        rng_sched = np.random.default_rng(40000 + wi * 100 + rank)
        if role == "exploiter":
            # Dedicated Main Exploiter initialized from anchor w0, trained directly to crush pfsp_w3 + default!
            exp_live = clone_plan_model(snapshots["pfsp_w0"]).to(device)
            exp_ema = clone_plan_model(exp_live).to(device)
            _broadcast_model_(exp_live, torch_mesh, rank, world_size, device)
            _broadcast_model_(exp_ema, torch_mesh, rank, world_size, device)
            exp_opt = _make_grouped_aurora(exp_live, lr=args.lr * 1.25, scalar_lr=args.scalar_lr * 1.25)
            arm_models: dict[str, PlanWally | None] = {"exploiter": exp_live, "default": None, **snapshots}
            scfg = [
                {"team_arms": ("exploiter", "pfsp_w3", "default"), "fcp_mixed": False, "perturbation": "baseline"},
                {"team_arms": ("exploiter", "pfsp_w3", "pfsp_w3"), "fcp_mixed": False, "perturbation": "fast"},
                {"team_arms": ("exploiter", "pfsp_w3", "script_cart_rush"), "fcp_mixed": False, "perturbation": "volatile"},
                {"team_arms": ("exploiter", "pfsp_w3", "pfsp_w2"), "fcp_mixed": False, "perturbation": "baseline"},
                {"team_arms": ("exploiter", "pfsp_w3", "script_leader_gank"), "fcp_mixed": False, "perturbation": "slow"},
                {"team_arms": ("exploiter", "pfsp_w3", "default"), "fcp_mixed": False, "perturbation": "baseline"},
            ]
            t_arm, t_opt, t_ema = "exploiter", exp_opt, exp_ema
            steps_this = max(12, args.steps_per_wave - 4)
        else:
            league_arms = list(snapshots.keys())
            ema_ref = f"pfsp_w{int(wname[1:]) - 1}" if f"pfsp_w{int(wname[1:]) - 1}" in snapshots else None
            scfg = sample_pfsp_server_assignments(
                rng_sched, len(train_maps),
                train_arm="train", ema_arm=ema_ref,
                league_arms=league_arms, win_rates_vs_opp=win_rates_vs_opp,
            )
            # After Exploiter is frozen, force at least 1 server per Studio against exploiter_1!
            if "exploiter_1" in snapshots:
                scfg[2]["team_arms"] = ("train", "exploiter_1", "default")
            arm_models = {"train": main_live, "default": None, **snapshots}
            t_arm, t_opt, t_ema = "train", main_opt, main_ema
            steps_this = args.steps_per_wave

        seeds = [20000 + wi * 1000 + rank * 100 + j for j in range(len(train_maps))]
        w_rep = _run_pfsp_sync_wave(
            harness, out_dir / f"work_r{rank}_{wname}",
            rank=rank, world_size=world_size, torch_mesh=torch_mesh, device=device,
            seeds=seeds, map_names=train_maps, server_configs=scfg,
            base_port=args.base_port + wi * 10,
            arm_models=arm_models, train_arm=t_arm,
            optimizer=t_opt, ema_model=t_ema, ema_decay=0.90,
            n_sync_steps=steps_this, local_batch_size=args.local_batch,
            entropy_coef=ent_coef, skill=7, rng_seed=8000 + wi * 100 + rank,
        )

        # Gather match outcomes across all 4 Studios to update global PFSP win rates identically!
        if world_size > 1:
            all_m_list = [None for _ in range(world_size)]
            dist.all_gather_object(all_m_list, w_rep["local_comp_matches"])
            wave_matches = [m for sub in all_m_list for m in sub]
        else:
            wave_matches = w_rep["local_comp_matches"]

        if role == "main":
            for m in wave_matches:
                if "train" in m:
                    v_tr = float(m["train"])
                    for oa, vo in m.items():
                        if oa == "train":
                            continue
                        s = 1.0 if v_tr > float(vo) + 1e-6 else (0.0 if float(vo) > v_tr + 1e-6 else 0.5)
                        pfsp_wins[oa] += s
                        pfsp_games[oa] += 1.0
                        win_rates_vs_opp[oa] = pfsp_wins[oa] / pfsp_games[oa]
            snap_name = f"pfsp_{wname}"
            snapshots[snap_name] = clone_plan_model(main_ema).to(device)
            if rank == 0:
                torch.save({"model": main_ema.state_dict(), "spec": main_ema.spec()}, ckpt_dir / f"{snap_name}_ema.pt")
        else:
            snapshots["exploiter_1"] = clone_plan_model(exp_ema).to(device)
            if rank == 0:
                torch.save({"model": exp_ema.state_dict(), "spec": exp_ema.spec()}, ckpt_dir / "exploiter_1_ema.pt")

        if rank == 0:
            w_rep["wave"] = wname
            w_rep["role"] = role
            w_rep["entropy_coef"] = ent_coef
            w_rep["pfsp_win_rates_snapshot"] = {k: round(v, 3) for k, v in sorted(win_rates_vs_opp.items())}
            del w_rep["local_comp_matches"]
            wave_reports.append(w_rep)
            snr = w_rep.get("snr_4studio") or {}
            bg = w_rep.get("snr_4studio_by_group") or {}
            sk_eff = (bg.get("skill_id_head") or {}).get("step_efficiency")
            print(
                f"[{wname}:{role}] wall={w_rep['wall_s']}s upd={w_rep['updates_applied']} "
                f"eff={snr.get('step_efficiency')} skill_eff={sk_eff} skill_r={w_rep['online_skill_id_corr']} "
                f"fsq_ent={w_rep['fsq_entropy_bits']} rdma_ms={w_rep['mean_rdma_allreduce_ms']}",
                flush=True,
            )

    train_wall_s = round(time.monotonic() - t_train0, 2)

    # ── Eval A: 4-Studio Parallel 48-Match Multi-Map Seat-Rotated CRN Elo + 3-Team alpha-Rank ──
    eval_maps = ["boil", "stormkeep", "catharsis", "afterslime"]
    my_map = eval_maps[rank % len(eval_maps)]
    eval_models: dict[str, PlanWally | None] = {"default": None, **snapshots}
    eval_triplets = [
        ("default", "pfsp_w0", "pfsp_w5"),
        ("default", "pfsp_w2", "pfsp_w5"),
        ("default", "exploiter_1", "pfsp_w5"),
        ("script_cart_rush", "pfsp_w3", "pfsp_w5"),
    ]
    local_sc, local_cp = [], []
    local_sk_early, local_sk_late = [], []
    for ti, trip in enumerate(eval_triplets):
        # Use balanced + skewed rosters during Elo eval (1 balanced, 2 skewed per triplet)
        e_seeds = [60000 + ti * 100 + j for j in range(3)]
        bfiles = []
        rng_e = np.random.default_rng(70000 + rank * 100 + ti)
        for j in range(3):
            rk, ents = sample_skewed_roster(rng_e, archetype=("balanced_mid" if j == 0 else "carry_vs_balanced"))
            bf = out_dir / f"eval_r{rank}_{my_map}_t{ti}_s{j}_{rk}.txt"
            write_bots_txt(bf, ents)
            bfiles.append(bf)
        er = _eval_triplet_with_scripted(
            harness, out_dir / f"elo_r{rank}_{my_map}_t{ti}",
            seeds=e_seeds, arm_models=eval_models, team_arms=trip,
            mapname=my_map, duration_s=args.elo_duration_s,
            device=device, base_port=args.base_port + 200 + ti * 10,
            bot_config_files=bfiles, skill=7, rotate_seats=True,
        )
        local_sc.extend(er["score_matches"])
        local_cp.extend(er["composite_matches"])
        local_sk_early.extend(er["skill_preds_early"])
        local_sk_late.extend(er["skill_preds_late"])

    # ── Eval B: Asymmetric Underdog Stress Test (Team 1 = Elo~700 Underdog vs Team 2 = Elo~1250 Overdog vs Team 3 = Stock) ──
    # Paired CRN: exact same disadvantaged Team-1 mechanical roster piloted by `pfsp_w5` vs piloted by `default`!
    underdog_entries = [
        ("U1_novice", 1, "mech_novice"), ("U2_novice", 1, "mech_novice"), ("U3_potato", 1, "mech_potato"),
        ("O1_carry", 2, "mech_carry"), ("O2_carry", 2, "mech_carry"), ("O3_vet", 2, "mech_veteran"),
        ("M1_stock", 3, "mech_stock"), ("M2_stock", 3, "mech_stock"), ("M3_stock", 3, "mech_stock"),
    ]
    u_file = out_dir / f"underdog_r{rank}.txt"
    write_bots_txt(u_file, underdog_entries)
    u_seeds = [80000 + rank * 10 + j for j in range(3)]
    u_pfsp = _eval_triplet_with_scripted(
        harness, out_dir / f"underdog_r{rank}_{my_map}_pfsp",
        seeds=u_seeds, arm_models={"pfsp_w5": snapshots["pfsp_w5"], "default": None},
        team_arms=("pfsp_w5", "default", "default"),
        mapname=my_map, duration_s=args.elo_duration_s,
        device=device, base_port=args.base_port + 300,
        bot_config_files=[u_file] * 3, skill=7, rotate_seats=False,
    )
    u_def = _eval_triplet_with_scripted(
        harness, out_dir / f"underdog_r{rank}_{my_map}_default",
        seeds=u_seeds, arm_models={"default_underdog": None, "default": None},
        team_arms=("default_underdog", "default", "default"),
        mapname=my_map, duration_s=args.elo_duration_s,
        device=device, base_port=args.base_port + 310,
        bot_config_files=[u_file] * 3, skill=7, rotate_seats=False,
    )

    local_payload = {
        "rank": rank, "host": os.uname().nodename, "map": my_map,
        "score_matches": local_sc, "composite_matches": local_cp,
        "skill_preds_early": local_sk_early[:1000], "skill_preds_late": local_sk_late[:1000],
        "underdog_pfsp_comp": [m.get("pfsp_w5", 0.0) for m in u_pfsp["composite_matches"]],
        "underdog_pfsp_score": [m.get("pfsp_w5", 0.0) for m in u_pfsp["score_matches"]],
        "underdog_default_comp": [m.get("default_underdog", 0.0) for m in u_def["composite_matches"]],
        "underdog_default_score": [m.get("default_underdog", 0.0) for m in u_def["score_matches"]],
    }

    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    if rank == 0:
        neuron_audit = _audit_dead_neurons(main_ema, harness, out_dir / "audit_r0", device, args.base_port + 350)
        all_sc = [m for g in gathered for m in g["score_matches"]]
        all_cp = [m for g in gathered for m in g["composite_matches"]]
        per_map = {}
        for g in gathered:
            per_map[g["map"]] = {
                "host": g["host"], "matches": len(g["score_matches"]),
                "score_elo": compute_multiteam_elo(g["score_matches"], anchor_arm="default"),
                "composite_elo": compute_multiteam_elo(g["composite_matches"], anchor_arm="default"),
            }
        overall_sc = compute_multiteam_elo(all_sc, anchor_arm="default")
        overall_cp = compute_multiteam_elo(all_cp, anchor_arm="default")
        all_arms = sorted({a for m in all_cp for a in m})
        arank_comp = alpha_rank_3team(all_arms, all_cp, alpha=2.0)
        arank_score = alpha_rank_3team(all_arms, all_sc, alpha=2.0)

        # Online Skill-ID correlation (masked actor predicting latent mechanical skill!)
        def _corr_mse(pairs):
            if len(pairs) < 20:
                return {"n": len(pairs), "pearson_r": None, "mse": None}
            a = np.asarray(pairs, dtype=np.float64)
            r_val = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if (a[:, 0].std() > 1e-6 and a[:, 1].std() > 1e-6) else None
            return {"n": len(pairs), "pearson_r": round(r_val, 4) if r_val is not None else None, "mse": round(float(np.mean((a[:, 0] - a[:, 1]) ** 2)), 4)}

        early_all = [p for g in gathered for p in g["skill_preds_early"]]
        late_all = [p for g in gathered for p in g["skill_preds_late"]]

        u_pfsp_c = [v for g in gathered for v in g["underdog_pfsp_comp"]]
        u_def_c = [v for g in gathered for v in g["underdog_default_comp"]]
        u_pfsp_s = [v for g in gathered for v in g["underdog_pfsp_score"]]
        u_def_s = [v for g in gathered for v in g["underdog_default_score"]]

        summary = {
            "mode": "pfsp_mech_league",
            "world_size": world_size,
            "device": device,
            "servers_concurrent": 6 * world_size,
            "model_spec": main_ema.spec(),
            "param_count": sum(p.numel() for p in main_ema.parameters()),
            "train_wall_s": train_wall_s,
            "total_updates_applied": sum(w["updates_applied"] for w in wave_reports),
            "waves": wave_reports,
            "final_weight_norms": weight_norm_report(main_ema),
            "final_neuron_audit": neuron_audit,
            "elo_total_matches": len(all_sc),
            "elo_overall_score": overall_sc,
            "elo_overall_composite": overall_cp,
            "alpha_rank_composite": arank_comp,
            "alpha_rank_score": arank_score,
            "elo_by_map": per_map,
            "online_skill_id_audit": {
                "early_0_to_8s": _corr_mse(early_all),
                "late_8s_plus": _corr_mse(late_all),
            },
            "underdog_roster_stress_test": {
                "matches": len(u_pfsp_c),
                "pfsp_w5_underdog_mean_comp": round(float(np.mean(u_pfsp_c)), 3) if u_pfsp_c else 0.0,
                "default_underdog_mean_comp": round(float(np.mean(u_def_c)), 3) if u_def_c else 0.0,
                "pfsp_w5_underdog_mean_score": round(float(np.mean(u_pfsp_s)), 3) if u_pfsp_s else 0.0,
                "default_underdog_mean_score": round(float(np.mean(u_def_s)), 3) if u_def_s else 0.0,
            },
        }
        out_path = out_dir / "studio4_pfsp_mech_report.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print("PFSP_REPORT_WRITTEN:", str(out_path), flush=True)
        print(json.dumps({
            "train_wall_s": summary["train_wall_s"],
            "total_updates": summary["total_updates_applied"],
            "elo_overall_score": {k: v["elo"] for k, v in (overall_sc.get("ratings") or {}).items()},
            "elo_overall_composite": {k: v["elo"] for k, v in (overall_cp.get("ratings") or {}).items()},
            "alpha_rank_composite": arank_comp,
            "alpha_rank_score": arank_score,
            "online_skill_id_audit": summary["online_skill_id_audit"],
            "underdog_roster_stress_test": summary["underdog_roster_stress_test"],
            "dead_fraction": neuron_audit.get("dead_fraction"),
            "max_dev_from_1": summary["final_weight_norms"].get("max_dev_from_1"),
        }, indent=2), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
