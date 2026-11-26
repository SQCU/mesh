#!/usr/bin/env python3
"""Four-Studio Large Multi-Episode Batch PFSP + Native Calibrated HavocBot League Runner.

Design principles:
  1. Large multi-episode batches (B_global = 1,024 strided across 24 FULL distinct live
     episodes per update, spanning 24-43s of game time per episode across all 4 Studios) so
     each optimizer step marginalizes over map x opponent x native-roster variance natively.
  2. Zero synthetic handicaps: rosters are drawn directly from Xonotic's calibrated native
     HavocBot parameter -> 1v1 DM Elo map using batch-isolated per-bot `bot_base_skill`.
  3. Masked actor (`BOT_CONFIGURATION_COLUMNS` zeroed) + Asymmetric Critic + Online Skill-ID
     head supervised by each bot's empirical calibrated 1v1 DM Elo.
  4. True PFSP league + permanent anchors (`default` + 4 scripted archetypes) + FCP mixed
     teammates + a dedicated multi-iteration Main Exploiter (counterexample generator) frozen
     into the league pool.
  5. Held-out 4-Studio parallel 48-match Multi-Map CRN Elo + 3-Team alpha-Rank + Online
     Skill-ID audit as the sole performance attribution surface.
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
from solver.strat.game_value import effective_leader, formal_value_record
from solver.strat.grad_snr import summarize_mccandlish_pairs
from solver.strat.league_curriculum import (
    SCRIPTED_ANCHORS,
    alpha_rank_3team,
    sample_pfsp_server_assignments,
    scripted_plan_rows,
)
from solver.strat.losses import (
    LossConfig,
    attach_episode_gae_,
    compute_step_rewards,
    plan_losses_for_transition,
)
from solver.strat.mech_calibration import (
    CALIBRATED_NATIVE_REGRESSION,
    extract_privileged_skill_target,
    load_native_bots,
    sample_native_lobby_roster,
    write_native_bots_txt,
)
from solver.strat.outer_validation import compute_multiteam_elo, extract_behavioral_motifs
from solver.strat.phase_fingerprint import PhaseSpaceAccumulator, compare_fingerprints, strip_raw_maps
from solver.strat.plan_policy import (
    PLAN_CONT_DIM,
    PLAN_WIDTH,
    PlanTransition,
    PlanWally,
    _geo_to_torch,
    clone_plan_model,
    ema_update,
    pack_plan_rows,
    plan_module_group,
    project_unit_norm_,
    sample_plan,
    weight_norm_report,
)
from solver.strat.plan_runner import MultiServerPlanHarness
from solver.strat.runtime import build_runtime_frame
from solver.strat.train_config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    OptimizerConfig,
    TrainingConfig,
    load_training_config,
)
from solver.xonwire import CART_KIND, OBSERVATION_KIND, PLAN_KIND, TEAM_KIND, Engine, RuntimeFrames, frames


MODULE_GROUPS = ("encoder", "gram_cgt", "resnet_ffn", "action_head", "value_head", "skill_id_head", "dynamics_head")


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


def _make_grouped_aurora(model: PlanWally, opt_cfg: OptimizerConfig, lr_mult: float = 1.0) -> Aurora:
    grouped: dict[str, list[torch.nn.Parameter]] = {g: [] for g in MODULE_GROUPS}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grouped.setdefault(plan_module_group(n), []).append(p)
    active = [(g, ps) for g, ps in grouped.items() if ps]
    lr = opt_cfg.lr * lr_mult
    scalar_lr = opt_cfg.scalar_lr * lr_mult
    opt = Aurora(
        [{"params": ps, "lr": lr, "scalar_lr": scalar_lr} for _, ps in active],
        lr=lr,
        scalar_lr=scalar_lr,
        mu=opt_cfg.mu,
        nesterov=opt_cfg.nesterov,
        pp_iterations=opt_cfg.pp_iterations,
        pp_beta=opt_cfg.pp_beta,
        ns_steps=opt_cfg.ns_steps,
        eps=opt_cfg.eps,
        ns_coeffs=opt_cfg.ns_coeffs,
        d_init_clamp=opt_cfg.d_init_clamp,
        d_iter_clamp=opt_cfg.d_iter_clamp,
        scalar_beta2=opt_cfg.scalar_beta2,
        scalar_eps=opt_cfg.scalar_eps,
    )
    opt._group_names = [g for g, _ in active]
    return opt


def _sync_adv_stats(
    model: PlanWally,
    batch: list[PlanTransition],
    torch_mesh,
    world_size: int,
    device: str,
    loss_cfg: LossConfig,
) -> tuple[float, float]:
    errs = []
    with torch.no_grad():
        for tr in batch:
            am = tr.actor_mask.astype(bool)
            if not np.any(am):
                continue
            if tr.gae_adv is not None:
                errs.append(tr.gae_adv[am])
                continue
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
    return mean, max(math.sqrt(var), loss_cfg.batch_adv_std_floor)


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


def _collect_full_episodes(
    harness: MultiServerPlanHarness,
    work_dir: Path,
    *,
    cfg: TrainingConfig,
    rank: int,
    device: str,
    seeds: list[int],
    map_names: list[str],
    server_configs: list[dict],
    base_port: int,
    arm_models: dict[str, PlanWally | None],
    train_arm: str,
    regression: dict,
    native_bots: list[dict],
    rng_seed: int,
    map_specs: list[dict] | None = None,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    bot_cfg_files: list[Path] = []
    extra_cvars: list[dict[str, float]] = []
    slowmos: list[float] = []
    teams_per_sv: list[int] = []
    bots_per_sv: list[int] = []
    maxc_per_sv: list[int] = []
    meta_per_sv: list[dict] = []
    roster_kinds: list[str] = []
    for idx, (mname, scfg) in enumerate(zip(map_names, server_configs)):
        spec = map_specs[idx] if (map_specs is not None and idx < len(map_specs)) else None
        sv_k = int(spec["teams"]) if spec else 3
        sv_n = int(spec["players"]) if spec else 9
        sv_maxc = int(spec.get("maxclients", min(255, sv_n + 15))) if spec else harness.maxclients
        r_kind, entries = sample_native_lobby_roster(
            rng, regression=regression, native_bots=native_bots, n_teams=sv_k, n_bots=sv_n, league_cfg=cfg.league,
        )
        roster_kinds.append(r_kind)
        bfile = work_dir / f"native_roster_{idx}_{r_kind}.txt"
        write_native_bots_txt(bfile, entries)
        bot_cfg_files.append(bfile)
        extra_cvars.append(dict(PERTURBATIONS.get(scfg["perturbation"], {})))
        slowmos.append(float(cfg.harness.map_slowmo.get(mname, 1.0)))
        teams_per_sv.append(sv_k)
        bots_per_sv.append(sv_n)
        maxc_per_sv.append(sv_maxc)
        meta_per_sv.append(spec or {})

    engine = Engine("udp://127.0.0.1:0", harness.framebytes)
    sock = engine.path
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    procs, logs = harness._spawn_servers(
        work_dir, sock, seeds, base_port,
        map_names=map_names,
        bot_config_files=bot_cfg_files,
        skill=cfg.harness.default_skill,
        extra_cvars=extra_cvars,
        slowmo_per_server=slowmos,
        teams_per_server=teams_per_sv,
        bots_per_server=bots_per_sv,
        maxclients_per_server=maxc_per_sv,
        meta_per_server=meta_per_sv,
        point_limit=cfg.harness.point_limit,
    )

    routes: dict[int, tuple[int, object]] = {}
    sess_order: dict[int, int] = {}
    sess_maps: dict[int, tuple[str, object]] = {}
    obs_mems: dict[int, ObservationMemory] = {}
    prev_state: dict[int, dict] = {}
    mealy_states: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)
    fsq_code_counts: dict[tuple[int, ...], int] = defaultdict(int)
    telemetry_by_sess: dict[int, list[dict]] = defaultdict(list)
    by_session_transitions: dict[int, list[PlanTransition]] = defaultdict(list)
    live_dims: dict[int, dict] = {}
    finished_sessions: set[int] = set()
    sess_first_episode: dict[int, tuple] = {}
    expected_sessions = len(seeds)
    skill_pred_pairs: list[tuple[float, float]] = []
    fp_train_acc = PhaseSpaceAccumulator(label=f"train:{train_arm}", cell_size=cfg.eval.fingerprint_cell_size)
    fp_default_acc = PhaseSpaceAccumulator(label="ref:default", cell_size=cfg.eval.fingerprint_cell_size)

    t0 = time.monotonic()
    try:
        while len(finished_sessions) < expected_sessions and (time.monotonic() - t0) < cfg.harness.max_episode_wall_s:
            if all(p.poll() is not None for p in procs):
                break
            readable, _, _ = select.select([engine], [], [], 0.05)
            if not readable:
                continue
            for node, address, buf in engine.receive():
                s = inbox.feed(buf)
                if s is not None:
                    routes[s] = (node, address)
            while (ready := inbox.take()) is not None:
                (session, tick_id), records, event_batches = ready
                if int(session) in finished_sessions:
                    continue
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
                obs_rows = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
                if len(obs_rows) == 0 or len(cart_rows) == 0 or len(team_rows) == 0:
                    continue

                if session not in sess_maps:
                    sess_maps[session] = harness._identify_map(cart_rows)
                sess_map_name, sess_manifold = sess_maps[session]
                cell_centroids = sess_manifold.cell_centroids if sess_manifold is not None else None

                ep_id = ObservationMemory.identity(session, team_rows[0, TS["EPISODE"]])
                if session not in sess_first_episode:
                    sess_first_episode[session] = ep_id
                elif ep_id != sess_first_episode[session]:
                    finished_sessions.add(int(session))
                    continue
                mem = obs_mems[session]
                mem.observe(ep_id, event_batches)
                evt_rows = mem.rows
                geo_np = sess_manifold.build_features(obs_rows, cart_rows, team_rows, evt_rows) if sess_manifold is not None else None
                priv_skill_np = extract_privileged_skill_target(obs_rows, regression=regression)

                tick = build_runtime_frame(obs_rows, cart_rows, team_rows)
                n_teams = len(team_rows)
                shift = (s_idx + rank) % len(base_team_arms)
                sess_arms = tuple(base_team_arms[(i + shift) % len(base_team_arms)] for i in range(len(base_team_arms)))

                n_bots = len(obs_rows)
                bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                bot_arms = np.asarray([sess_arms[(int(t) - 1) % len(sess_arms)] if int(t) >= 1 else "default" for t in bot_teams], dtype=object)
                if fcp_mixed:
                    tr_idxs = [bi for bi in range(n_bots) if bot_arms[bi] == train_arm]
                    if len(tr_idxs) >= 2:
                        bot_arms[tr_idxs[0]] = "default"

                # Accumulate live 5-layer Phase-Space & Markov-Flow fingerprints per policy arm!
                if isinstance(event_batches, dict):
                    ev_vals = [v for v in event_batches.values() if v is not None and len(v) > 0]
                    step_evts = np.concatenate(ev_vals, axis=0) if ev_vals else None
                elif isinstance(event_batches, (list, tuple)) and len(event_batches) > 0:
                    ev_vals = [v for v in event_batches if v is not None and len(v) > 0]
                    step_evts = np.concatenate(ev_vals, axis=0) if ev_vals else None
                else:
                    step_evts = None
                tr_team_set = {int(bot_teams[bi]) for bi in range(n_bots) if bot_arms[bi] == train_arm}
                df_team_set = {int(bot_teams[bi]) for bi in range(n_bots) if bot_arms[bi] == "default"}
                if tr_team_set:
                    fp_train_acc.update(obs_raw, step_evts, team_filter=tr_team_set)
                if df_team_set:
                    fp_default_acc.update(obs_raw, step_evts, team_filter=df_team_set)

                cont_all = np.zeros((n_bots, PLAN_CONT_DIM), dtype=np.float32)
                cart_all = np.zeros((n_bots,), dtype=np.int64)
                rival_all = np.zeros((n_bots,), dtype=np.int64)
                cell_all = np.zeros((n_bots,), dtype=np.int64)
                logp_all = np.zeros((n_bots,), dtype=np.float32)
                sv_maxc = int(obs_raw.shape[0])
                plan_matrix = np.zeros((sv_maxc, PLAN_WIDTH), dtype=np.float32)
                live_dims[int(session)] = {
                    "teams": int(len(team_rows)),
                    "carts": int(len(cart_rows)),
                    "bots": int(n_bots),
                    "maxclients": sv_maxc,
                }
                train_h_in: np.ndarray | None = None

                obs_t = torch.as_tensor(obs_rows, dtype=torch.float32, device=device)
                carts_t = torch.as_tensor(cart_rows, dtype=torch.float32, device=device)
                teams_t = torch.as_tensor(team_rows, dtype=torch.float32, device=device)
                evt_t = torch.as_tensor(np.array(evt_rows, copy=True), dtype=torch.float32, device=device) if len(evt_rows) else None
                geo_t = _geo_to_torch(geo_np, device)

                for s_arm in set(str(a) for a in bot_arms):
                    if s_arm in SCRIPTED_ANCHORS:
                        s_part = scripted_plan_rows(
                            s_arm, obs_rows, cart_rows, team_rows, bot_arms,
                            sv_maxc, cell_centroids=cell_centroids,
                            projected_leader_idx=tick.game_value.projected_role,
                        )
                        m_rows = s_part[:, 0] > 0
                        plan_matrix[m_rows] = s_part[m_rows]

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
                        if arm_name == train_arm:
                            if out.fsq_codes is not None:
                                codes_np = out.fsq_codes.detach().cpu().numpy().astype(np.int64)
                                for bi in np.flatnonzero(sel):
                                    fsq_code_counts[tuple(int(v) for v in codes_np[bi])] += 1
                            if out.pred_skill is not None and (time.monotonic() - t0) >= cfg.eval.skill_id_early_cutoff_s:
                                ps_hat = out.pred_skill.detach().cpu().numpy()
                                for bi in range(n_bots):
                                    skill_pred_pairs.append((float(ps_hat[bi]), float(priv_skill_np[bi])))
                        samp = sample_plan(out, deterministic=False)
                        h_nxt = out.h_next.detach().cpu().numpy()
                        for bi in range(n_bots):
                            eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                            if obs_rows[bi, OBS["ALIVE"]] > 0.5:
                                st_map[eid] = h_nxt[bi]
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
                            sv_maxc, cell_centroids=cell_centroids,
                            cart_lengths=cart_rows[:, CS["PATH_LENGTH"]],
                            pack_cfg=cfg.action_pack,
                        )
                        m_rows = part[:, 0] > 0
                        plan_matrix[m_rows] = part[m_rows]

                dest = routes.get(session)
                if dest is not None:
                    node, addr = dest
                    req_id = records[OBSERVATION_KIND][0]["req_id"]
                    engine.send(addr, node, frames(PLAN_KIND, req_id, tick_id, plan_matrix, harness.framebytes, session))

                telemetry_by_sess[session].append({
                    "t": time.monotonic() - t0,
                    "teams": n_teams,
                    "team_policy_arms": [sess_arms[i % len(sess_arms)] for i in range(n_teams)],
                    "carts": [{name.lower(): float(row[col]) for name, col in CS.items()} for row in cart_rows],
                    "assignments": [{"edict": int(obs_rows[i, OBS["ID"]]), "team": int(bot_teams[i]), "arm": str(bot_arms[i])} for i in range(n_bots)],
                    "realized_events": [], "resources": [],
                    "game_value": formal_value_record(tick.game_value, tick.cartstate),
                    "learning": {}, "policy_updates": {},
                })

                prev = prev_state.get(session)
                if prev is not None and prev["episode"] == ep_id:
                    p_ids = prev["obs"][:, OBS["ID"]].astype(np.int64)
                    c_ids = obs_rows[:, OBS["ID"]].astype(np.int64)
                    id_to_idx = {int(eid): idx for idx, eid in enumerate(c_ids)}
                    succ = np.asarray([id_to_idx.get(int(eid), -1) for eid in p_ids], dtype=np.int64)
                    rew = compute_step_rewards(
                        prev["tick"].context,
                        prev["tick"].cartstate,
                        tick.cartstate,
                        prev["obs"],
                        obs_rows,
                        prev["carts"],
                        cart_rows,
                        succ,
                        n_teams=n_teams,
                        cfg=cfg.losses.reward,
                    )
                    w_before = effective_leader(prev["tick"].context, prev["tick"].cartstate)
                    w_after = effective_leader(tick.context, tick.cartstate)
                    p_teams_0 = np.asarray(prev["tick"].context.team_of, dtype=np.int64)
                    wm = p_teams_0 == w_before
                    nwm = np.asarray([
                        (int(tick.context.team_of[idx]) == w_after) if idx >= 0 else False for idx in succ
                    ], dtype=bool)
                    disc = np.where((succ >= 0) & (not tick.cartstate.finished), cfg.losses.gamma, 0.0).astype(np.float32)
                    actor_mask = (prev["arms"] == train_arm) & (succ >= 0)
                    if np.any(actor_mask):
                        by_session_transitions[int(session)].append(PlanTransition(
                            session=int(session),
                            obs_in=prev["obs"].copy(), carts_in=prev["carts"].copy(),
                            teams_in=prev["teams"].copy(), events_in=prev["events"].copy(),
                            obs_out=obs_rows.copy(), carts_out=cart_rows.copy(),
                            teams_out=team_rows.copy(), events_out=evt_rows.copy(),
                            cont=prev["cont"].copy(), cart_pick=prev["cart_pick"].copy(),
                            rival_pick=prev["rival_pick"].copy(), cell_pick=prev["cell_pick"].copy(),
                            behavior_logp=prev["logp"].copy(), reward=rew.astype(np.float32),
                            winner_mask=wm, next_winner_mask=nwm, discount=disc,
                            actor_mask=actor_mask, successor_rows=succ,
                            geo_in=prev["geo"], geo_out=geo_np,
                            h_in=prev["h_in"], h_out=train_h_in, map_name=sess_map_name,
                            priv_skill_in=prev["priv_skill"].copy(), priv_skill_out=priv_skill_np.copy(),
                        ))

                if tick.cartstate.finished:
                    finished_sessions.add(int(session))
                    continue

                prev_state[session] = {
                    "episode": ep_id, "tick": tick,
                    "obs": obs_rows.copy(), "carts": cart_rows.copy(),
                    "teams": team_rows.copy(), "events": evt_rows.copy(),
                    "geo": geo_np, "h_in": train_h_in.copy() if train_h_in is not None else None,
                    "arms": bot_arms.copy(), "cont": cont_all,
                    "cart_pick": cart_all, "rival_pick": rival_all,
                    "cell_pick": cell_all, "logp": logp_all, "priv_skill": priv_skill_np.copy(),
                }
    finally:
        harness._stop_servers(procs, logs)
        engine.close()

    local_score_matches = []
    for sess_id, recs in sorted(telemetry_by_sess.items()):
        mot = extract_behavioral_motifs(recs)
        sc_map = {}
        for _, ti in (mot.get("per_team") or {}).items():
            a = ti["arm"]
            sc_map[a] = max(sc_map.get(a, -1e9), float(ti["score_end"]))
        if len(sc_map) >= 2:
            local_score_matches.append(sc_map)

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
        "rollout_wall_s": round(time.monotonic() - t0, 2),
        "by_session_transitions": dict(by_session_transitions),
        "n_episodes": len(by_session_transitions),
        "n_score_outs": len(finished_sessions),
        "n_expected_sessions": expected_sessions,
        "n_raw_transitions": sum(len(v) for v in by_session_transitions.values()),
        "local_comp_matches": local_score_matches,
        "fsq_unique_codes": len(fsq_code_counts),
        "fsq_entropy_bits": round(fsq_ent, 3),
        "online_skill_id_corr": skill_corr,
        "roster_kinds": roster_kinds,
        "live_dims": list(live_dims.values()),
        "fp_train": fp_train_acc.finalize(),
        "fp_default": fp_default_acc.finalize(),
    }


def _collect_training_transitions(by_session: dict[int, list[PlanTransition]], target_local: int = 0) -> list[PlanTransition]:
    """Return transitions across all episodes unconditionally when target_local <= 0."""
    sessions = [s for s in sorted(by_session) if by_session[s]]
    if not sessions:
        return []
    if target_local <= 0:
        return [tr for s in sessions for tr in by_session[s]]
    per_ep = target_local // len(sessions)
    rem = target_local - per_ep * len(sessions)
    out: list[PlanTransition] = []
    for si, s in enumerate(sessions):
        want = max(1, per_ep + (1 if si < rem else 0))
        trs = by_session[s]
        if len(trs) <= want:
            out.extend(trs)
        else:
            idxs = np.linspace(0, len(trs) - 1, num=want, dtype=np.int64)
            out.extend(trs[int(i)] for i in idxs)
    return out


def _large_batch_rdma_aurora_update(
    model: PlanWally,
    ema_model: PlanWally,
    optimizer: Aurora,
    batch: list[PlanTransition],
    *,
    cfg: TrainingConfig,
    rank: int,
    world_size: int,
    torch_mesh,
    device: str,
    entropy_coef: float,
    iter_id: int,
) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    slices_by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    off = 0
    for n, p in zip(names, params):
        sz = p.numel()
        slices_by_group[plan_module_group(n)].append((off, off + sz))
        off += sz

    ppo_epochs = max(1, cfg.optimizer.ppo_epochs)
    t0 = time.perf_counter()
    adv_mean, adv_std = _sync_adv_stats(model, batch, torch_mesh, world_size, device, cfg.losses)
    ep0_snr = None
    ep0_group_snr = {}
    rdma_ms = []
    last_diags = []
    gnorms = []

    for ep in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        inv_n = 1.0 / max(1, len(batch))
        diags = []
        for tr in batch:
            comps, diag = plan_losses_for_transition(
                model, tr, device=device,
                loss_cfg=cfg.losses,
                entropy_coef=entropy_coef,
                adv_mean=adv_mean, adv_std=adv_std,
            )
            (comps["total"] * inv_n).backward()
            diags.append(diag)
        last_diags = diags

        flat_g = torch_mesh.empty((total_params,), dtype=torch.float32, device=device)
        off_g = 0
        for p in params:
            sz = p.numel()
            if p.grad is not None:
                flat_g[off_g : off_g + sz].copy_(p.grad.detach().reshape(-1))
            else:
                flat_g[off_g : off_g + sz].zero_()
            off_g += sz

        tr0 = time.perf_counter()
        mean_g, gpair, grp_pairs = _rdma_mccandlish_from_flat(
            flat_g, slices_by_group, torch_mesh, world_size, float(len(batch)), device, iter_id * ppo_epochs + ep,
        )
        if device == "mps":
            torch.mps.synchronize()
        rdma_ms.append((time.perf_counter() - tr0) * 1e3)
        if ep == 0:
            ep0_snr = gpair
            ep0_group_snr = grp_pairs

        off_g = 0
        for p in params:
            sz = p.numel()
            if p.grad is None:
                p.grad = mean_g[off_g : off_g + sz].view_as(p.data).clone()
            else:
                p.grad.copy_(mean_g[off_g : off_g + sz].view_as(p.data))
            off_g += sz
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip).item())
        gnorms.append(gn)
        optimizer.step()
        project_unit_norm_(model, eps=cfg.norm.unit_norm_eps)
        ema_update(ema_model, model, decay=cfg.optimizer.ema_decay)

    _broadcast_model_(model, torch_mesh, rank, world_size, device)
    _broadcast_model_(ema_model, torch_mesh, rank, world_size, device)

    def _m(k):
        vals = [d[k] for d in last_diags if d.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    if world_size > 1:
        bl_list = [None for _ in range(world_size)]
        dist.all_gather_object(bl_list, len(batch))
        b_global_exact = int(sum(int(x) for x in bl_list))
    else:
        b_global_exact = len(batch)

    return {
        "update_wall_s": round(time.perf_counter() - t0, 2),
        "b_local": len(batch),
        "b_global": b_global_exact,
        "ppo_epochs_applied": len(gnorms),
        "mean_rdma_ms": round(float(np.mean(rdma_ms)), 3) if rdma_ms else None,
        "grad_norm": round(float(np.mean(gnorms)), 4) if gnorms else 0.0,
        "loss_total": _m("loss_total"),
        "skill_id_mse": _m("skill_id_mse"),
        "entropy": _m("entropy"),
        "entropy_disc": _m("entropy_disc"),
        "cont_std_mean": _m("cont_std_mean"),
        "snr_4studio": summarize_mccandlish_pairs([ep0_snr]) if ep0_snr is not None else None,
        "snr_4studio_by_group": {g: summarize_mccandlish_pairs([gp]) for g, gp in ep0_group_snr.items()},
    }


def _eval_triplet_native(
    harness: MultiServerPlanHarness,
    work_dir: Path,
    *,
    cfg: TrainingConfig,
    seeds: list[int],
    arm_models: dict[str, PlanWally | None],
    team_arms: tuple[str, ...],
    mapname: str,
    device: str,
    base_port: int,
    regression: dict,
    native_bots: list[dict],
    rng_seed: int,
    map_spec: dict | None = None,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    sv_k = int(map_spec["teams"]) if map_spec else 3
    sv_n = int(map_spec["players"]) if map_spec else 9
    sv_maxc = int(map_spec.get("maxclients", min(255, sv_n + 15))) if map_spec else harness.maxclients
    bfiles = []
    for j in range(len(seeds)):
        dist_kind = "gaussian_lobby" if j == 0 else ("team_skewed" if j == 1 else "within_team_skewed")
        rk, ents = sample_native_lobby_roster(
            rng, regression=regression, native_bots=native_bots, distribution=dist_kind,
            n_teams=sv_k, n_bots=sv_n, league_cfg=cfg.league,
        )
        bf = work_dir / f"eval_roster_{j}_{rk}.txt"
        write_native_bots_txt(bf, ents)
        bfiles.append(bf)

    engine = Engine("udp://127.0.0.1:0", harness.framebytes)
    sock = engine.path
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    slowmos = [float(cfg.harness.map_slowmo.get(mapname, 1.0))] * len(seeds)
    procs, logs = harness._spawn_servers(
        work_dir, sock, seeds, base_port,
        map_names=[mapname] * len(seeds),
        bot_config_files=bfiles,
        skill=cfg.harness.default_skill,
        slowmo_per_server=slowmos,
        teams_per_server=[sv_k] * len(seeds),
        bots_per_server=[sv_n] * len(seeds),
        maxclients_per_server=[sv_maxc] * len(seeds),
        meta_per_server=[map_spec or {}] * len(seeds),
        point_limit=cfg.harness.point_limit,
    )
    routes: dict[int, tuple[int, object]] = {}
    sess_order: dict[int, int] = {}
    sess_maps: dict[int, tuple[str, object]] = {}
    obs_mems: dict[int, ObservationMemory] = {}
    mealy_states: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)
    telemetry_by_sess: dict[int, list[dict]] = defaultdict(list)
    finished_sessions: set[int] = set()
    sess_first_episode: dict[int, tuple] = {}
    expected_sessions = len(seeds)
    skill_preds_early: list[tuple[float, float]] = []
    skill_preds_late: list[tuple[float, float]] = []
    t_start = time.monotonic()

    try:
        while len(finished_sessions) < expected_sessions and (time.monotonic() - t_start) < cfg.harness.max_episode_wall_s:
            if all(p.poll() is not None for p in procs):
                break
            readable, _, _ = select.select([engine], [], [], 0.05)
            if not readable:
                continue
            for node, address, buf in engine.receive():
                s = inbox.feed(buf)
                if s is not None:
                    routes[s] = (node, address)
            while (ready := inbox.take()) is not None:
                (session, tick_id), records, event_batches = ready
                if int(session) in finished_sessions:
                    continue
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
                if session not in sess_first_episode:
                    sess_first_episode[session] = ep_id
                elif ep_id != sess_first_episode[session]:
                    finished_sessions.add(int(session))
                    continue
                mem = obs_mems[session]
                mem.observe(ep_id, event_batches)
                evt_rows = mem.rows
                geo_np = sess_manifold.build_features(obs_rows, cart_rows, team_rows, evt_rows) if sess_manifold is not None else None
                priv_skill_np = extract_privileged_skill_target(obs_rows, regression=regression)

                tick = build_runtime_frame(obs_rows, cart_rows, team_rows)
                n_teams = len(team_rows)
                shift = s_idx % len(team_arms)
                sess_arms = tuple(team_arms[(i + shift) % len(team_arms)] for i in range(len(team_arms)))
                n_bots = len(obs_rows)
                bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                bot_arms = np.asarray([sess_arms[(int(t) - 1) % len(sess_arms)] if int(t) >= 1 else "default" for t in bot_teams], dtype=object)

                sv_maxc_obs = int(obs_raw.shape[0])
                plan_matrix = np.zeros((sv_maxc_obs, PLAN_WIDTH), dtype=np.float32)
                for s_arm in set(str(a) for a in bot_arms):
                    if s_arm in SCRIPTED_ANCHORS:
                        s_part = scripted_plan_rows(
                            s_arm, obs_rows, cart_rows, team_rows, bot_arms,
                            sv_maxc_obs, cell_centroids=cell_centroids,
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
                        if out.pred_skill is not None and arm_name == "main_final":
                            ph = out.pred_skill.detach().cpu().numpy()
                            dst_list = skill_preds_early if elapsed_s < cfg.eval.skill_id_early_cutoff_s else skill_preds_late
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
                            sv_maxc_obs, cell_centroids=cell_centroids,
                            cart_lengths=cart_rows[:, CS["PATH_LENGTH"]],
                            pack_cfg=cfg.action_pack,
                        )
                        m_rows = part[:, 0] > 0
                        plan_matrix[m_rows] = part[m_rows]

                dest = routes.get(session)
                if dest is not None:
                    node, addr = dest
                    req_id = records[OBSERVATION_KIND][0]["req_id"]
                    engine.send(addr, node, frames(PLAN_KIND, req_id, tick_id, plan_matrix, harness.framebytes, session))
                telemetry_by_sess[session].append({
                    "t": elapsed_s, "teams": n_teams, "team_policy_arms": [sess_arms[i % len(sess_arms)] for i in range(n_teams)],
                    "carts": [{name.lower(): float(row[col]) for name, col in CS.items()} for row in cart_rows],
                    "assignments": [{"edict": int(obs_rows[i, OBS["ID"]]), "team": int(bot_teams[i]), "arm": str(bot_arms[i])} for i in range(n_bots)],
                    "realized_events": [], "resources": [],
                    "game_value": formal_value_record(tick.game_value, tick.cartstate),
                    "learning": {}, "policy_updates": {},
                })
                if tick.cartstate.finished:
                    finished_sessions.add(int(session))
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
            sc_m[a] = max(sc_m.get(a, -1e9), sc)
            cp_m[a] = max(cp_m.get(a, -1e9), sc)
        if len(sc_m) >= 2:
            score_matches.append(sc_m)
            comp_matches.append(cp_m)
    return {
        "score_matches": score_matches,
        "composite_matches": comp_matches,
        "n_score_outs": len(finished_sessions),
        "n_expected_sessions": expected_sessions,
        "skill_preds_early": skill_preds_early,
        "skill_preds_late": skill_preds_late,
    }


def _audit_dead_neurons(
    model: PlanWally,
    harness: MultiServerPlanHarness,
    work_dir: Path,
    device: str,
    base_port: int,
    cfg: TrainingConfig,
) -> dict:
    res = harness.run_episode_batch(
        work_dir / "neuron_audit",
        seeds=list(cfg.seeds.neuron_audit_seeds),
        arm_models={"eval": model, "default": None},
        team_arms=("eval", "default", "default"),
        duration_s=cfg.eval.neuron_audit_duration_s,
        rotate_seats=False,
        deterministic=False,
        device=device,
        base_port=base_port,
        map_names=list(cfg.eval.neuron_audit_maps),
        loss_cfg=cfg.losses,
        pack_cfg=cfg.action_pack,
    )
    trs = res["transitions"]
    if not trs:
        return {"error": "no transitions"}
    model.zero_grad(set_to_none=True)
    for tr in trs:
        comps, _ = plan_losses_for_transition(model, tr, device=device, loss_cfg=cfg.losses)
        (comps["total"] / len(trs)).backward()
    ffn_reports, tot_units, tot_dead = [], 0, 0
    for bi, blk in enumerate(model.blocks):
        g_up = blk.ffn_up.weight.grad.detach().cpu() if blk.ffn_up.weight.grad is not None else torch.zeros_like(blk.ffn_up.weight.cpu())
        rn = g_up.norm(dim=-1)
        med = float(rn.median().item())
        dead = int((rn < cfg.eval.dead_unit_rel_threshold * max(med, 1e-12)).sum().item())
        tot_units += int(rn.numel())
        tot_dead += dead
        ffn_reports.append({
            "block": bi, "units": int(rn.numel()), "dead_units": dead,
            "grad_norm_cv": round(float(rn.std().item() / max(float(rn.mean().item()), 1e-12)), 4),
            "min_over_med": round(float(rn.min().item() / max(med, 1e-12)), 4),
        })
    model.zero_grad(set_to_none=True)
    return {"total_wide_ffn_units": tot_units, "dead_units": tot_dead, "dead_fraction": round(tot_dead / max(1, tot_units), 6), "blocks": ffn_reports}


def _build_overrides_from_args(args) -> dict:
    ov: dict = {"schedule": {}, "harness": {}, "optimizer": {}}
    if args.device is not None:
        ov["schedule"]["device"] = args.device
    if args.out_dir is not None:
        ov["schedule"]["out_dir"] = args.out_dir
    if args.calibration_json is not None:
        ov["schedule"]["calibration_json"] = args.calibration_json
    if args.init_ckpt is not None:
        ov["schedule"]["init_ckpt"] = args.init_ckpt
    if args.supermap_manifest is not None:
        ov["schedule"]["supermap_manifest"] = args.supermap_manifest
    if args.train_maps is not None:
        ov["schedule"]["train_maps"] = [m.strip() for m in args.train_maps.split(",") if m.strip()]
    if args.servers_per_node is not None:
        ov["schedule"]["servers_per_node"] = args.servers_per_node
    if args.main_iters is not None:
        ov["schedule"]["main_iters"] = args.main_iters
    if args.exploiter_iters is not None:
        ov["schedule"]["exploiter_iters"] = args.exploiter_iters
    if args.local_batch is not None:
        ov["schedule"]["local_batch"] = args.local_batch
    if args.stage_dir is not None:
        ov["harness"]["stage_dir"] = args.stage_dir
    if args.max_episode_wall_s is not None:
        ov["harness"]["max_episode_wall_s"] = args.max_episode_wall_s
    if args.point_limit is not None:
        ov["harness"]["point_limit"] = args.point_limit
    if args.base_port is not None:
        ov["harness"]["base_port"] = args.base_port
    if args.ppo_epochs is not None:
        ov["optimizer"]["ppo_epochs"] = args.ppo_epochs
    if args.lr is not None:
        ov["optimizer"]["lr"] = args.lr
    if args.scalar_lr is not None:
        ov["optimizer"]["scalar_lr"] = args.scalar_lr
    return {k: v for k, v in ov.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_TRAINING_CONFIG_PATH))
    ap.add_argument("--losses", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--calibration-json", default=None)
    ap.add_argument("--main-iters", type=int, default=None)
    ap.add_argument("--exploiter-iters", type=int, default=None)
    ap.add_argument("--max-episode-wall-s", type=float, default=None)
    ap.add_argument("--point-limit", type=float, default=None)
    ap.add_argument("--local-batch", type=int, default=None)
    ap.add_argument("--ppo-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--scalar-lr", type=float, default=None)
    ap.add_argument("--base-port", type=int, default=None)
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--train-maps", default=None)
    ap.add_argument("--supermap-manifest", default=None)
    ap.add_argument("--servers-per-node", type=int, default=None)
    args = ap.parse_args()

    cfg = load_training_config(
        config_path=args.config,
        losses_path=args.losses or None,
        overrides=_build_overrides_from_args(args),
    )
    device = cfg.schedule.device
    rank, world_size, torch_mesh = _init_mesh_world(device)
    stage_dir = Path(cfg.harness.stage_dir).expanduser()
    out_dir = Path(cfg.schedule.out_dir).expanduser()
    ckpt_dir = out_dir / "ckpts"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "effective_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    regression = CALIBRATED_NATIVE_REGRESSION
    if cfg.schedule.calibration_json:
        cal_path = Path(cfg.schedule.calibration_json).expanduser()
        if rank == 0 and cal_path.exists():
            try:
                cal_data = json.loads(cal_path.read_text())
                regression = cal_data.get("parametric_regression") or CALIBRATED_NATIVE_REGRESSION
            except Exception:
                regression = CALIBRATED_NATIVE_REGRESSION
        if world_size > 1:
            reg_box = [regression]
            dist.broadcast_object_list(reg_box, src=0)
            regression = reg_box[0]
    native_bots = load_native_bots(stage_dir / cfg.harness.data_pk3_rel)

    supermap_specs: list[dict] = []
    if cfg.schedule.supermap_manifest:
        mpath = Path(cfg.schedule.supermap_manifest).expanduser()
        if mpath.exists():
            supermap_specs = json.loads(mpath.read_text())
    train_maps = list(cfg.schedule.train_maps)
    servers_per_node = (
        int(cfg.schedule.servers_per_node)
        if cfg.schedule.servers_per_node > 0
        else (3 if supermap_specs else len(train_maps))
    )
    harness = MultiServerPlanHarness.from_config(cfg.harness, is_supermap=bool(supermap_specs))

    torch.manual_seed(cfg.seeds.model_init)
    main_live = PlanWally.from_config(cfg.model, cfg.norm).to(device)
    if cfg.schedule.init_ckpt and rank == 0:
        ckpt_p = Path(cfg.schedule.init_ckpt).expanduser()
        if ckpt_p.exists():
            sd = torch.load(ckpt_p, map_location="cpu")
            main_live.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
            print(f"[RESUME] Loaded init checkpoint on rank 0: {ckpt_p}", flush=True)
    main_ema = clone_plan_model(main_live).to(device)
    _broadcast_model_(main_live, torch_mesh, rank, world_size, device)
    _broadcast_model_(main_ema, torch_mesh, rank, world_size, device)
    main_opt = _make_grouped_aurora(main_live, cfg.optimizer, lr_mult=1.0)

    snapshots: dict[str, PlanWally] = {"main_init": clone_plan_model(main_ema).to(device)}
    if rank == 0:
        torch.save({"model": main_ema.state_dict(), "spec": main_ema.spec()}, ckpt_dir / "main_init_ema.pt")

    fp_iter0_ref: dict | None = None
    win_rates_vs_opp: dict[str, float] = {}
    pfsp_wins: dict[str, float] = defaultdict(float)
    pfsp_games: dict[str, float] = defaultdict(float)
    iter_reports = []
    t_train0 = time.monotonic()

    main_iters = cfg.schedule.main_iters
    exp_iters = cfg.schedule.exploiter_iters
    pre_exp = max(1, (main_iters + 1) // 2) if exp_iters > 0 and main_iters >= 2 else main_iters
    schedule = []
    for k in range(1, pre_exp + 1):
        ent = cfg.losses.entropy_for_iteration("main", k, main_iters, post_exploiter=False)
        schedule.append((f"m{k:02d}", "main", ent))
    for k in range(1, exp_iters + 1):
        ent = cfg.losses.entropy_for_iteration("exploiter", k, main_iters)
        schedule.append((f"exp{k:02d}", "exploiter", ent))
    for k in range(pre_exp + 1, main_iters + 1):
        ent = cfg.losses.entropy_for_iteration("main", k, main_iters, post_exploiter=True)
        schedule.append((f"m{k:02d}", "main", ent))

    exp_live = exp_ema = exp_opt = None
    exp_target_name = None

    for idx_it, (it_name, role, ent_coef) in enumerate(schedule, start=1):
        rng_sched = np.random.default_rng(cfg.seeds.schedule_base + idx_it * 100 + rank)
        if supermap_specs:
            if cfg.schedule.match_supermap_scale_across_ranks:
                iter_spec = supermap_specs[(idx_it - 1) % len(supermap_specs)]
                iter_specs = [iter_spec] * servers_per_node
                iter_maps = [iter_spec["name"]] * servers_per_node
            else:
                iter_specs = [supermap_specs[(rank * servers_per_node + idx_it + j) % len(supermap_specs)] for j in range(servers_per_node)]
                iter_maps = [sp["name"] for sp in iter_specs]
        else:
            iter_specs = None
            iter_maps = train_maps[:servers_per_node] if servers_per_node < len(train_maps) else train_maps

        if role == "exploiter":
            if exp_live is None:
                exp_target_name = "main_mid"
                snapshots[exp_target_name] = clone_plan_model(main_ema).to(device)
                exp_live = clone_plan_model(snapshots["main_init"]).to(device)
                exp_ema = clone_plan_model(exp_live).to(device)
                _broadcast_model_(exp_live, torch_mesh, rank, world_size, device)
                _broadcast_model_(exp_ema, torch_mesh, rank, world_size, device)
                exp_opt = _make_grouped_aurora(exp_live, cfg.optimizer, lr_mult=cfg.optimizer.exploiter_lr_mult)
            base_exp_cfgs = [
                {
                    "team_arms": (
                        "exploiter",
                        exp_target_name if item["opp1"] == "{target}" else item["opp1"],
                        exp_target_name if item["opp2"] == "{target}" else item["opp2"],
                    ),
                    "fcp_mixed": bool(item.get("fcp_mixed", False)),
                    "perturbation": str(item.get("perturbation", "baseline")),
                }
                for item in cfg.league.exploiter_server_configs
            ]
            scfg = [base_exp_cfgs[j % len(base_exp_cfgs)] for j in range(len(iter_maps))]
            arm_models: dict[str, PlanWally | None] = {"exploiter": exp_live, "default": None, **snapshots}
            t_arm, t_model, t_ema, t_opt = "exploiter", exp_live, exp_ema, exp_opt
        else:
            league_arms = list(snapshots.keys())
            ema_ref = "main_mid" if "main_mid" in snapshots else "main_init"
            scfg = sample_pfsp_server_assignments(
                rng_sched, len(iter_maps),
                train_arm="train", ema_arm=ema_ref,
                league_arms=league_arms, win_rates_vs_opp=win_rates_vs_opp,
                pfsp_power=cfg.league.pfsp_power,
                pfsp_min_weight=cfg.league.pfsp_min_weight,
                pfsp_default_winrate=cfg.league.pfsp_default_winrate,
            )
            if "exploiter_1" in snapshots:
                if len(scfg) > 2:
                    scfg[2]["team_arms"] = ("train", "exploiter_1", "default")
                elif rank % 2 == 1:
                    scfg[0]["team_arms"] = ("train", "exploiter_1", "default")
            arm_models = {"train": main_live, "default": None, **snapshots}
            t_arm, t_model, t_ema, t_opt = "train", main_live, main_ema, main_opt

        seeds = [cfg.seeds.match_base + idx_it * 1000 + rank * 100 + j for j in range(len(iter_maps))]
        iter_port = cfg.harness.base_port + (idx_it % cfg.harness.port_iter_modulo) * cfg.harness.port_stride_per_iter
        col = _collect_full_episodes(
            harness, out_dir / f"roll_r{rank}_{it_name}",
            cfg=cfg, rank=rank, device=device, seeds=seeds, map_names=iter_maps,
            server_configs=scfg, base_port=iter_port,
            arm_models=arm_models, train_arm=t_arm,
            regression=regression, native_bots=native_bots,
            rng_seed=cfg.seeds.rollout_rng_base + idx_it * 100 + rank,
            map_specs=iter_specs,
        )
        attach_episode_gae_(t_model, col["by_session_transitions"], device=device, gae_lambda=cfg.losses.gae_lambda)
        batch = _collect_training_transitions(col["by_session_transitions"], cfg.schedule.local_batch)
        upd = _large_batch_rdma_aurora_update(
            t_model, t_ema, t_opt, batch,
            cfg=cfg, rank=rank, world_size=world_size, torch_mesh=torch_mesh, device=device,
            entropy_coef=ent_coef, iter_id=idx_it,
        )

        if world_size > 1:
            all_m_list = [None for _ in range(world_size)]
            all_d_list = [None for _ in range(world_size)]
            all_so_list = [None for _ in range(world_size)]
            dist.all_gather_object(all_m_list, col["local_comp_matches"])
            dist.all_gather_object(all_d_list, col.get("live_dims") or [])
            dist.all_gather_object(all_so_list, (col.get("n_score_outs", 0), col.get("n_expected_sessions", 0)))
            iter_matches = [m for sub in all_m_list for m in sub]
            iter_dims = [d for sub in all_d_list for d in sub]
            iter_score_outs = sum(x[0] for x in all_so_list)
            iter_expected = sum(x[1] for x in all_so_list)
        else:
            iter_matches = col["local_comp_matches"]
            iter_dims = col.get("live_dims") or []
            iter_score_outs = col.get("n_score_outs", 0)
            iter_expected = col.get("n_expected_sessions", 0)

        if role == "main":
            for m in iter_matches:
                if "train" in m:
                    v_tr = float(m["train"])
                    for oa, vo in m.items():
                        if oa == "train":
                            continue
                        s = 1.0 if v_tr > float(vo) + 1e-6 else (0.0 if float(vo) > v_tr + 1e-6 else 0.5)
                        pfsp_wins[oa] += s
                        pfsp_games[oa] += 1.0
                        win_rates_vs_opp[oa] = pfsp_wins[oa] / pfsp_games[oa]
            if it_name == f"m{pre_exp:02d}":
                snapshots["main_mid"] = clone_plan_model(main_ema).to(device)
                if rank == 0:
                    torch.save({"model": main_ema.state_dict(), "spec": main_ema.spec()}, ckpt_dir / "main_mid_ema.pt")
            if it_name == f"m{main_iters:02d}":
                snapshots["main_final"] = clone_plan_model(main_ema).to(device)
                if rank == 0:
                    torch.save({"model": main_ema.state_dict(), "spec": main_ema.spec()}, ckpt_dir / "main_final_ema.pt")
        elif it_name == f"exp{exp_iters:02d}":
            snapshots["exploiter_1"] = clone_plan_model(exp_ema).to(device)
            if rank == 0:
                torch.save({"model": exp_ema.state_dict(), "spec": exp_ema.spec()}, ckpt_dir / "exploiter_1_ema.pt")

        if rank == 0:
            fp_tr = col.get("fp_train")
            fp_df = col.get("fp_default")
            if fp_iter0_ref is None and fp_tr is not None and fp_tr.get("alive_samples", 0) > 0:
                fp_iter0_ref = fp_tr
            drift_vs_default = (
                compare_fingerprints(fp_tr, fp_df)
                if (fp_tr and fp_df and fp_tr.get("alive_samples", 0) > 20 and fp_df.get("alive_samples", 0) > 20)
                else None
            )
            drift_vs_iter0 = (
                compare_fingerprints(fp_tr, fp_iter0_ref)
                if (fp_tr and fp_iter0_ref and fp_tr.get("alive_samples", 0) > 20)
                else None
            )
            snr = upd.get("snr_4studio") or {}
            bg = upd.get("snr_4studio_by_group") or {}
            rec = {
                "iter": it_name, "role": role, "episodes_global": len(iter_dims) or col["n_episodes"] * world_size,
                "score_outs_global": iter_score_outs, "expected_episodes_global": iter_expected,
                "live_dims": iter_dims,
                "raw_transitions_global_est": col["n_raw_transitions"] * world_size,
                "b_global": upd["b_global"],
                "rollout_s": col["rollout_wall_s"], "update_s": upd["update_wall_s"],
                "rdma_ms": upd["mean_rdma_ms"], "grad_norm": upd["grad_norm"],
                "entropy": upd["entropy"], "entropy_disc": upd["entropy_disc"],
                "cont_std_mean": upd["cont_std_mean"], "skill_id_mse": upd["skill_id_mse"],
                "online_skill_id_corr": col["online_skill_id_corr"],
                "fsq_codes": col["fsq_unique_codes"], "fsq_ent": col["fsq_entropy_bits"],
                "snr_4studio": snr,
                "snr_4studio_by_group": bg,
                "phase_fingerprint_train": strip_raw_maps(fp_tr) if fp_tr else None,
                "drift_vs_default": drift_vs_default,
                "drift_vs_iter0": drift_vs_iter0,
            }
            iter_reports.append(rec)
            w_def = (drift_vs_default or {}).get("weirdness_index")
            jsd_f = (drift_vs_default or {}).get("flux_jsd")
            sig_tr = ((fp_tr or {}).get("L2_markov_flux") or {}).get("entropy_production_sigma")
            flgs = (drift_vs_default or {}).get("flags") or []
            dim_str = ""
            if iter_dims:
                ks = [d["teams"] for d in iter_dims]
                js = [d["carts"] for d in iter_dims]
                ns = [d["bots"] for d in iter_dims]
                dim_str = f" K=[{min(ks)}..{max(ks)}] J=[{min(js)}..{max(js)}] N=[{min(ns)}..{max(ns)}]"
            print(
                f"[{it_name}:{role:9s}] score_outs={iter_score_outs}/{iter_expected}{dim_str} B={rec['b_global']} "
                f"roll={rec['rollout_s']}s upd={rec['update_s']}s eff={snr.get('step_efficiency')} "
                f"act_eff={(bg.get('action_head') or {}).get('step_efficiency')} "
                f"sk_r={rec['online_skill_id_corr']} std={rec['cont_std_mean']:.3f} "
                f"| FP sig={sig_tr} flux_jsd={jsd_f} weird={w_def} flags={flgs}",
                flush=True,
            )

    train_wall_s = round(time.monotonic() - t_train0, 2)

    if supermap_specs:
        eval_idxs = [int(round(r * (len(supermap_specs) - 1) / max(1, world_size - 1))) for r in range(world_size)]
        my_eval_spec = supermap_specs[eval_idxs[rank % len(eval_idxs)]]
        my_map = my_eval_spec["name"]
        eval_n_seeds = cfg.eval.eval_seeds_supermap
    else:
        eval_maps = list(cfg.eval.eval_maps)
        my_eval_spec = None
        my_map = eval_maps[rank % len(eval_maps)]
        eval_n_seeds = cfg.eval.eval_seeds_standard
    eval_models: dict[str, PlanWally | None] = {"default": None, **snapshots}
    eval_triplets = [trip for trip in cfg.eval.eval_triplets if all((a in eval_models or a in SCRIPTED_ANCHORS) for a in trip)]
    local_sc, local_cp, local_sk_early, local_sk_late = [], [], [], []
    local_eval_so = 0
    local_eval_exp = 0
    for ti, trip in enumerate(eval_triplets):
        er = _eval_triplet_native(
            harness, out_dir / f"elo_r{rank}_{my_map}_t{ti}",
            cfg=cfg,
            seeds=[cfg.seeds.eval_match_base + ti * 100 + j for j in range(eval_n_seeds)],
            arm_models=eval_models, team_arms=trip,
            mapname=my_map,
            device=device,
            base_port=cfg.harness.base_port + cfg.eval.eval_port_offset + ti * cfg.eval.eval_triplet_port_stride,
            regression=regression, native_bots=native_bots,
            rng_seed=cfg.seeds.eval_rng_base + rank * 100 + ti,
            map_spec=my_eval_spec,
        )
        local_sc.extend(er["score_matches"])
        local_cp.extend(er["composite_matches"])
        local_eval_so += int(er.get("n_score_outs", 0))
        local_eval_exp += int(er.get("n_expected_sessions", 0))
        local_sk_early.extend(er["skill_preds_early"])
        local_sk_late.extend(er["skill_preds_late"])

    local_payload = {
        "rank": rank, "host": os.uname().nodename, "map": my_map,
        "score_matches": local_sc, "composite_matches": local_cp,
        "eval_score_outs": local_eval_so, "eval_expected": local_eval_exp,
        "skill_preds_early": local_sk_early, "skill_preds_late": local_sk_late,
    }
    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    if rank == 0:
        neuron_audit = _audit_dead_neurons(
            main_ema, harness, out_dir / "audit_r0", device,
            cfg.harness.base_port + cfg.eval.neuron_audit_port_offset, cfg,
        )
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
        arank_comp = alpha_rank_3team(all_arms, all_cp, alpha=cfg.league.alpha_rank_alpha, m_pop=cfg.league.alpha_rank_m_pop)
        arank_score = alpha_rank_3team(all_arms, all_sc, alpha=cfg.league.alpha_rank_alpha, m_pop=cfg.league.alpha_rank_m_pop)

        def _corr_mse(pairs):
            if len(pairs) < 20:
                return {"n": len(pairs), "pearson_r": None, "mse": None}
            a = np.asarray(pairs, dtype=np.float64)
            r_val = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if (a[:, 0].std() > 1e-6 and a[:, 1].std() > 1e-6) else None
            return {"n": len(pairs), "pearson_r": round(r_val, 4) if r_val is not None else None, "mse": round(float(np.mean((a[:, 0] - a[:, 1]) ** 2)), 4)}

        early_all = [p for g in gathered for p in g["skill_preds_early"]]
        late_all = [p for g in gathered for p in g["skill_preds_late"]]

        summary = {
            "mode": "large_batch_native_pfsp",
            "config": cfg.to_dict(),
            "world_size": world_size,
            "device": device,
            "episodes_per_batch": servers_per_node * world_size,
            "b_global_per_step": cfg.schedule.local_batch * world_size,
            "total_training_episodes": sum(r["episodes_global"] for r in iter_reports),
            "train_wall_s": train_wall_s,
            "iterations": iter_reports,
            "final_weight_norms": weight_norm_report(main_ema),
            "final_neuron_audit": neuron_audit,
            "elo_total_matches": len(all_sc),
            "elo_overall_score": overall_sc,
            "elo_overall_composite": overall_cp,
            "alpha_rank_score": arank_score,
            "alpha_rank_composite": arank_comp,
            "elo_by_map": per_map,
            "online_skill_id_audit": {
                "early_0_to_8s": _corr_mse(early_all),
                "late_8s_plus": _corr_mse(late_all),
            },
        }
        out_path = out_dir / "studio4_largebatch_native_report.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print("LARGEBATCH_REPORT_WRITTEN:", str(out_path), flush=True)
        print(json.dumps({
            "train_wall_s": summary["train_wall_s"],
            "total_training_episodes": summary["total_training_episodes"],
            "elo_overall_score": {k: (v["elo"], v["ci95"], round(v["mean_score"], 2)) for k, v in (overall_sc.get("ratings") or {}).items()},
            "elo_overall_composite": {k: (v["elo"], v["ci95"], round(v["mean_score"], 2)) for k, v in (overall_cp.get("ratings") or {}).items()},
            "alpha_rank_score": arank_score,
            "alpha_rank_composite": arank_comp,
            "online_skill_id_audit": summary["online_skill_id_audit"],
            "dead_fraction": neuron_audit.get("dead_fraction"),
            "max_dev_from_1": summary["final_weight_norms"].get("max_dev_from_1"),
        }, indent=2), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
