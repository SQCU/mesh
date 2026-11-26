from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from payload.tools.strategy_io_schema import CS, EVT, OBS, TS
from solver.xonwire import (
    CART_KIND,
    OBSERVATION_KIND,
    PLAN_KIND,
    TEAM_KIND,
    Engine,
    RuntimeFrames,
    frames,
)

from .buffers import ObservationMemory
from .game_value import effective_leader, formal_value_record, role_rewards
from .grad_snr import mccandlish_pair_from_grads, summarize_mccandlish_pairs
from .map_manifold import MapManifold, load_map_manifold
from .outer_validation import extract_behavioral_motifs
from .plan_policy import (
    PLAN_CONT_DIM,
    PLAN_WIDTH,
    PlanTransition,
    PlanWally,
    _geo_to_torch,
    compute_batch_adv_stats,
    edm2_step,
    ema_update,
    pack_plan_rows,
    plan_losses_for_transition,
    plan_module_group,
    sample_plan,
)
from .runtime import build_runtime_frame


class MultiServerPlanHarness:
    """High-throughput G-server concurrent DarkPlaces rollout & training harness over
    loopback AF_INET UDP (udp://127.0.0.1:0) and 16-float parametric flow PLAN rows."""

    def __init__(
        self,
        *,
        dp_bin: str = "/tmp/xonotic-live-cgt/darkplaces-dedicated",
        basedir: str = "/Users/mdot/dox/xonotic/Xonotic",
        userdir: str = "/tmp/xonotic-live-cgt/userdir",
        framebytes: int = 8192,
        strategy_interval: float = 0.4,
        payload_speed: float = 80.0,
        slowmo: float = 1.0,
        maxclients: int = 16,
        manifold: MapManifold | None = None,
    ):
        self.dp_bin = dp_bin
        self.basedir = basedir
        self.userdir = userdir
        self.framebytes = int(framebytes)
        self.strategy_interval = float(strategy_interval)
        self.payload_speed = float(payload_speed)
        self.slowmo = float(slowmo)
        self.maxclients = int(maxclients)
        self.manifold = manifold
        self.manifolds_by_map: dict[str, MapManifold] = {}
        maps_dir = Path(userdir) / "data" / "maps"
        if maps_dir.exists():
            for ent in sorted(maps_dir.glob("*.ent")):
                try:
                    self.manifolds_by_map[ent.stem] = load_map_manifold(ent)
                except Exception:
                    pass
        if self.manifold is None:
            self.manifold = self.manifolds_by_map.get("boil") or next(iter(self.manifolds_by_map.values()), None)

    def _identify_map(self, cart_rows: np.ndarray) -> tuple[str, MapManifold | None]:
        if not self.manifolds_by_map or len(cart_rows) == 0:
            return "unknown", self.manifold
        lens = np.sort(cart_rows[:, CS["PATH_LENGTH"]].astype(np.float64))
        best_name, best_m, best_err = "unknown", self.manifold, float("inf")
        for name, m in self.manifolds_by_map.items():
            if len(m.splines) != len(lens):
                continue
            ml = np.sort(np.asarray([s.length for s in m.splines], dtype=np.float64))
            err = float(np.abs(ml - lens).mean())
            if err < best_err:
                best_err = err
                best_name = name
                best_m = m
        return best_name, best_m

    def _spawn_servers(
        self,
        work_dir: Path,
        sock: str,
        seeds: list[int],
        base_port: int,
        map_names: list[str] | None = None,
        bot_config_files: list[Path | None] | None = None,
        skill: int = 5,
        extra_cvars: list[dict[str, float]] | None = None,
        slowmo_per_server: list[float] | None = None,
        teams_per_server: list[int] | None = None,
        bots_per_server: list[int] | None = None,
        maxclients_per_server: list[int] | None = None,
        meta_per_server: list[dict] | None = None,
        point_limit: float | None = None,
    ):
        procs, logs = [], []
        maps = map_names if (map_names is not None and len(map_names) == len(seeds)) else ["boil"] * len(seeds)
        for idx, (seed, mname) in enumerate(zip(seeds, maps)):
            sv_userdir = work_dir / f"userdir_{idx}"
            sv_data = sv_userdir / "data"
            sv_data.mkdir(parents=True, exist_ok=True)
            for item in Path(self.userdir, "data").iterdir():
                dst = sv_data / item.name
                if not dst.exists():
                    os.symlink(item, dst)
            bcfg_name = None
            if bot_config_files is not None and idx < len(bot_config_files) and bot_config_files[idx] is not None:
                src_bcfg = Path(bot_config_files[idx])
                dst_bcfg = sv_data / "custom_bots.txt"
                dst_bcfg.write_text(src_bcfg.read_text())
                bcfg_name = "custom_bots.txt"
            logf = open(work_dir / f"server_{idx}_{mname}.log", "w")
            logs.append(logf)
            env = dict(os.environ)
            env["MESH_EXPERT_SOCKET"] = sock
            env["MESH_EXPERT_FRAMEBYTES"] = str(self.framebytes)
            sv_teams = int(teams_per_server[idx]) if (teams_per_server is not None and idx < len(teams_per_server)) else 3
            sv_bots = int(bots_per_server[idx]) if (bots_per_server is not None and idx < len(bots_per_server)) else 9
            sv_maxc = (
                int(maxclients_per_server[idx])
                if (maxclients_per_server is not None and idx < len(maxclients_per_server))
                else max(self.maxclients, min(255, sv_bots + 15))
            )
            sv_maxc = max(sv_bots + 1, min(255, sv_maxc))
            sv_meta = meta_per_server[idx] if (meta_per_server is not None and idx < len(meta_per_server)) else {}
            cmd = [
                self.dp_bin, "-xonotic",
                "-basedir", self.basedir,
                "-userdir", str(sv_userdir),
                "+exec", "gamemodes-payload.cfg",
                "+sv_public", "0",
                "+port", str(base_port + idx),
                "+maxplayers", str(sv_maxc),
                "+sv_random_seed", str(seed),
                "+g_payload", "1",
                "+g_payload_teams_override", str(sv_teams),
                "+g_payload_plan_only", "1",
                "+g_warmup", "0",
                "+g_start_delay", "0",
                "+g_payload_warmup", "1",
                "+g_spawn_swizzle", "1",
                "+bot_join_empty", "1",
                "+bot_number", str(sv_bots),
                "+skill", str(int(skill)),
                "+g_payload_speed", str(self.payload_speed),
                "+g_payload_strategy_interval", str(self.strategy_interval),
            ]
            if point_limit is not None and float(point_limit) > 0:
                cmd.extend(["+g_payload_point_limit", str(float(point_limit))])
            if bcfg_name is not None:
                cmd.extend(["+bot_prefix", "", "+bot_suffix", "", "+bot_config_file", bcfg_name])
            if extra_cvars is not None and idx < len(extra_cvars) and extra_cvars[idx]:
                for ck, cv in extra_cvars[idx].items():
                    cmd.extend([f"+{ck}", str(cv)])
            sv_slowmo = (
                float(slowmo_per_server[idx])
                if (slowmo_per_server is not None and idx < len(slowmo_per_server))
                else float(self.slowmo)
            )
            if sv_slowmo != 1.0:
                cmd.extend(["+slowmo", str(sv_slowmo)])
            cmd.extend(["+map", str(mname)])
            procs.append(subprocess.Popen(cmd, cwd=self.basedir, env=env, stdout=logf, stderr=subprocess.STDOUT))
            pub_host = os.environ.get("MESH_PUBLIC_HOST", "127.0.0.1")
            reg_payload = {
                "node": os.uname().nodename,
                "host": pub_host,
                "port": base_port + idx,
                "map": str(mname),
                "teams": sv_teams,
                "carts": int(sv_meta.get("carts", 0)),
                "tiles": int(sv_meta.get("tiles", 0)),
                "target_bots": sv_bots,
            }
            self._browser_notify("register", f"{os.uname().nodename}-{os.getpid()}-{idx}", reg_payload)
            time.sleep(0.15)
        return procs, logs

    @staticmethod
    def _browser_notify(action: str, server_id: str, payload: dict | None = None):
        url = os.environ.get("MESH_SERVER_BROWSER", "").rstrip("/")
        if not url:
            return
        import json as _json
        import urllib.request as _req
        try:
            if action == "register" and payload is not None:
                data = _json.dumps({"id": server_id, **payload}).encode()
                r = _req.Request(f"{url}/api/servers/register", data=data, headers={"Content-Type": "application/json"}, method="POST")
                _req.urlopen(r, timeout=0.5).close()
            elif action == "delete":
                r = _req.Request(f"{url}/api/servers/{server_id}", method="DELETE")
                _req.urlopen(r, timeout=0.5).close()
        except Exception:
            pass

    @staticmethod
    def _stop_servers(procs, logs):
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except OSError:
                    pass
                try:
                    p.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                        p.wait(timeout=2.0)
                    except OSError:
                        pass
                time.sleep(0.05)
        for f in logs:
            f.close()

    def run_episode_batch(
        self,
        work_dir: Path,
        *,
        seeds: list[int],
        arm_models: dict[str, PlanWally | None],
        team_arms: tuple[str, ...],
        duration_s: float = 35.0,
        rotate_seats: bool = True,
        deterministic: bool = False,
        train_arm: str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        ema_model: PlanWally | None = None,
        ema_decay: float = 0.95,
        min_batch_transitions: int = 16,
        min_groups_per_step: int = 2,
        ppo_epochs: int = 2,
        gamma: float = 0.95,
        grad_clip: float = 1.0,
        device: str = "cpu",
        base_port: int = 27000,
        map_names: list[str] | None = None,
        bot_config_files: list[Path | None] | None = None,
        skill: int = 5,
        extra_cvars: list[dict[str, float]] | None = None,
        slowmo_per_server: list[float] | None = None,
        custom_plan_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MapManifold | None], np.ndarray] | None = None,
    ) -> dict:
        """Run G concurrent live Xonotic servers in PLAN-only mode over loopback UDP,
        maintaining per-map Voronoi/cart-spline geometry & recurrent Mealy state per bot."""
        work_dir.mkdir(parents=True, exist_ok=True)
        engine = Engine("udp://127.0.0.1:0", self.framebytes)
        sock = engine.path
        inbox = RuntimeFrames(self.framebytes, plan_only=True)
        procs, logs = self._spawn_servers(
            work_dir, sock, seeds, base_port,
            map_names=map_names,
            bot_config_files=bot_config_files,
            skill=skill,
            extra_cvars=extra_cvars,
            slowmo_per_server=slowmo_per_server,
        )

        routes: dict[int, tuple[int, object]] = {}
        sess_order: dict[int, int] = {}
        sess_maps: dict[int, tuple[str, MapManifold | None]] = {}
        obs_mems: dict[int, ObservationMemory] = {}
        prev_state: dict[int, dict] = {}
        mealy_states: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)
        telemetry_by_sess: dict[int, list[dict]] = defaultdict(list)
        trajectory_samples: dict[int, list[dict]] = defaultdict(list)
        team_dmg_matrix: dict[str, float] = defaultdict(float)
        fsq_code_counts: dict[str, dict[tuple[int, ...], int]] = defaultdict(lambda: defaultdict(int))
        all_transitions: list[PlanTransition] = []
        fresh_buf: list[PlanTransition] = []
        next_batch_target = int(min_batch_transitions)
        deferred_batches = 0
        update_logs: list[dict] = []
        online_snr_pairs: list[dict] = []
        online_group_pairs: dict[str, list[dict]] = defaultdict(list)
        plan_focus_pw_hits = defaultdict(list)

        t_start = time.monotonic()
        total_responses = 0
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
                    present_mask = obs_raw[:, OBS["PRESENT"]] > 0.5
                    obs_rows = obs_raw[present_mask]
                    if len(obs_rows) == 0 or len(cart_rows) == 0 or len(team_rows) == 0:
                        continue

                    if session not in sess_maps:
                        sess_maps[session] = self._identify_map(cart_rows)
                    sess_map_name, sess_manifold = sess_maps[session]
                    cell_centroids = sess_manifold.cell_centroids if sess_manifold is not None else None

                    ep_id = ObservationMemory.identity(session, team_rows[0, TS["EPISODE"]])
                    mem = obs_mems[session]
                    mem.observe(ep_id, event_batches)
                    evt_rows = mem.rows
                    geo_np = (
                        sess_manifold.build_features(obs_rows, cart_rows, team_rows, evt_rows)
                        if sess_manifold is not None
                        else None
                    )

                    tick = build_runtime_frame(obs_rows, cart_rows, team_rows)
                    n_teams = len(team_rows)
                    shift = s_idx if rotate_seats and len(team_arms) > 1 else 0
                    sess_arms = tuple(team_arms[(i + shift) % len(team_arms)] for i in range(len(team_arms)))
                    bot_teams = obs_rows[:, OBS["TEAM"]].astype(np.int64)
                    bot_arms = np.asarray([sess_arms[(int(t) - 1) % len(sess_arms)] for t in bot_teams])
                    n_bots = len(obs_rows)

                    # Emit 16-float parametric flow PLAN rows
                    cont_all = np.zeros((n_bots, PLAN_CONT_DIM), dtype=np.float32)
                    cart_all = np.zeros((n_bots,), dtype=np.int64)
                    rival_all = np.zeros((n_bots,), dtype=np.int64)
                    cell_all = np.zeros((n_bots,), dtype=np.int64)
                    logp_all = np.zeros((n_bots,), dtype=np.float32)
                    plan_matrix = np.zeros((int(obs_raw.shape[0]), PLAN_WIDTH), dtype=np.float32)
                    train_h_in: np.ndarray | None = None
                    train_h_next: np.ndarray | None = None

                    if custom_plan_fn is not None:
                        plan_matrix = custom_plan_fn(obs_rows, cart_rows, team_rows, evt_rows, bot_arms, sess_manifold)
                    else:
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
                                    alive = obs_rows[bi, OBS["ALIVE"]] > 0.5
                                    if alive and eid in st_map:
                                        h_prev_np[bi] = st_map[eid]
                                h_prev_t = torch.as_tensor(h_prev_np, dtype=torch.float32, device=device)
                                out = mdl(obs_t, carts_t, teams_t, evt_t, geo=geo_t, h_prev=h_prev_t)
                                if out.fsq_codes is not None:
                                    q_np = out.fsq_codes.detach().cpu().numpy().astype(np.int64)
                                    for bi in np.flatnonzero(sel):
                                        fsq_code_counts[arm_name][tuple(int(v) for v in q_np[bi])] += 1
                                samp = sample_plan(out, deterministic=deterministic)
                                h_nxt_np = out.h_next.detach().cpu().numpy()
                                for bi in range(n_bots):
                                    eid = int(round(float(obs_rows[bi, OBS["ID"]])))
                                    if obs_rows[bi, OBS["ALIVE"]] > 0.5:
                                        st_map[eid] = h_nxt_np[bi]
                                    else:
                                        st_map.pop(eid, None)
                                if arm_name == (train_arm or "matrix_fusion"):
                                    train_h_in = h_prev_np
                                    train_h_next = h_nxt_np
                                cont_all[sel] = samp.cont.cpu().numpy()[sel]
                                cart_all[sel] = samp.cart_pick.cpu().numpy()[sel]
                                rival_all[sel] = samp.rival_pick.cpu().numpy()[sel]
                                cell_all[sel] = samp.cell_pick.cpu().numpy()[sel]
                                logp_all[sel] = samp.logp.cpu().numpy()[sel]
                                part = pack_plan_rows(
                                    obs_rows, team_rows, samp, bot_arms, {arm_name},
                                    self.maxclients, cell_centroids=cell_centroids,
                                    cart_lengths=cart_rows[:, CS["PATH_LENGTH"]],
                                )
                                mask_rows = part[:, 0] > 0
                                plan_matrix[mask_rows] = part[mask_rows]

                    # Build transition from previous tick of this session
                    prev = prev_state.get(session)
                    if prev is not None and prev["episode"] == ep_id:
                        p_ids = prev["obs"][:, OBS["ID"]].astype(np.int64)
                        c_ids = obs_rows[:, OBS["ID"]].astype(np.int64)
                        id_to_idx = {int(eid): idx for idx, eid in enumerate(c_ids)}
                        succ = np.asarray([id_to_idx.get(int(eid), -1) for eid in p_ids], dtype=np.int64)
                        rew = role_rewards(prev["tick"].context, prev["tick"].cartstate, tick.cartstate)
                        # True per-step deltas (OBS["CART_PUSH/CONTEST/PICKUPS"] are cumulative match counters!)
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
                        # Scale-invariant lane-progress shaping (so long-lane maps carry dense checkpoint-aligned signal)
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
                        actor_mask = (prev["arms"] == (train_arm or "matrix_fusion")) & (succ >= 0)
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
                        all_transitions.append(tr)
                        if train_arm is not None and np.any(actor_mask):
                            fresh_buf.append(tr)

                    pw = tick.game_value.projected_role
                    if pw is not None:
                        for i in range(n_bots):
                            t0 = int(bot_teams[i]) - 1
                            if t0 != int(pw) and bot_arms[i] != "default":
                                plan_focus_pw_hits[str(bot_arms[i])].append(1.0 if int(rival_all[i]) == int(pw) else 0.0)

                    dest = routes.get(session)
                    if dest is not None:
                        node, addr = dest
                        req_id = records[OBSERVATION_KIND][0]["req_id"]
                        engine.send(addr, node, frames(PLAN_KIND, req_id, tick_id, plan_matrix, self.framebytes, session))
                    total_responses += 1

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

                    # Compact spatial & causal-flow audit sample per tick
                    trajectory_samples[session].append({
                        "t": float(obs_rows[0, OBS["ENGINE_TIME"]]),
                        "pos": obs_rows[:, OBS["POS_X"] : OBS["POS_Z"] + 1].copy(),
                        "vel": obs_rows[:, OBS["VEL_X"] : OBS["VEL_Z"] + 1].copy(),
                        "team": bot_teams.copy(),
                        "alive": (obs_rows[:, OBS["ALIVE"]] > 0.5).copy(),
                        "strength": (obs_rows[:, OBS["STRENGTH_FINISHED"]] > obs_rows[:, OBS["ENGINE_TIME"]]).copy(),
                        "pickups": obs_rows[:, OBS["PICKUPS"]].copy(),
                        "cart_push": obs_rows[:, OBS["CART_PUSH"]].copy(),
                        "cart_contest": obs_rows[:, OBS["CART_CONTEST"]].copy(),
                        "dmg_dealt": obs_rows[:, OBS["ENEMY_DAMAGE_DEALT"]].copy(),
                        "cells": geo_np["bot_cells"].copy() if geo_np is not None else np.zeros((n_bots,), dtype=np.int64),
                        "cart_pos": cart_rows[:, CS["POS_X"] : CS["POS_Z"] + 1].copy(),
                        "cart_s": cart_rows[:, CS["PATH_POSITION"]].copy(),
                    })

                    realized_events = [
                        {
                            "kind": {5: "damage", 6: "kill", 7: "pickup", 8: "score_win" if row[EVT["TEAM"]] > 0 else "tie"}.get(int(row[EVT["KIND"]]), "other"),
                            "actor": int(row[EVT["OBSERVER"]]),
                            "actor_team": int(row[EVT["TEAM"]]),
                            "subject": int(row[EVT["SUBJECT"]]),
                            "value": float(row[EVT["AMOUNT"]]),
                        }
                        for arrived in mem.arrivals.values()
                        for row in arrived
                        if row[EVT["KIND"]] >= 5
                    ]
                    eid_to_team = {int(round(float(obs_rows[i, OBS["ID"]]))): int(bot_teams[i]) for i in range(n_bots)}
                    for ev in realized_events:
                        if ev["kind"] == "damage":
                            st = eid_to_team.get(int(ev["subject"]), 0)
                            if ev["actor_team"] > 0 and st > 0 and st != ev["actor_team"]:
                                team_dmg_matrix[f"{ev['actor_team']}->{st}"] += float(ev["value"])
                    resources = []
                    for t_1 in range(1, n_teams + 1):
                        sel = obs_rows[bot_teams == t_1]
                        spds = np.linalg.norm(sel[:, OBS["VEL_X"] : OBS["VEL_Z"] + 1], axis=1) if len(sel) else np.zeros(0)
                        resources.append({"team": t_1, "players": len(sel), "mean_speed": float(spds.mean()) if len(spds) else 0.0})
                    telemetry_by_sess[session].append({
                        "t": time.monotonic() - t_start,
                        "teams": n_teams,
                        "team_policy_arms": list(sess_arms),
                        "carts": [{name.lower(): float(row[col]) for name, col in CS.items()} for row in cart_rows],
                        "assignments": [{"edict": int(obs_rows[i, OBS["ID"]]), "team": int(bot_teams[i]), "arm": str(bot_arms[i])} for i in range(n_bots)],
                        "realized_events": realized_events,
                        "resources": resources,
                        "game_value": formal_value_record(tick.game_value, tick.cartstate),
                        "learning": {},
                        "policy_updates": {},
                    })

                    if train_arm is not None and optimizer is not None and arm_models.get(train_arm) is not None:
                        distinct_groups = sorted({tr.session for tr in fresh_buf})
                        if len(fresh_buf) >= next_batch_target and len(distinct_groups) >= min_groups_per_step:
                            mdl = arm_models[train_arm]
                            upd_info = self._ppo_update_with_snr(
                                mdl, optimizer, fresh_buf, distinct_groups,
                                ppo_epochs=ppo_epochs, grad_clip=grad_clip, device=device,
                                step_id=len(update_logs),
                                online_snr_pairs=online_snr_pairs,
                                online_group_pairs=online_group_pairs,
                            )
                            fresh_buf = []
                            next_batch_target = int(min_batch_transitions)
                            if ema_model is not None:
                                ema_update(ema_model, mdl, decay=ema_decay)
                            update_logs.append(upd_info)
        finally:
            self._stop_servers(procs, logs)
            engine.close()

        match_motifs = {}
        score_matches = []
        composite_matches = []
        score_matches_by_map: dict[str, list[dict]] = defaultdict(list)
        composite_matches_by_map: dict[str, list[dict]] = defaultdict(list)
        for sess_id, recs in sorted(telemetry_by_sess.items()):
            mot = extract_behavioral_motifs(recs)
            mname = sess_maps.get(sess_id, ("unknown", None))[0]
            mot["map"] = mname
            match_motifs[str(sess_id)] = mot
            sc_map, comp_map = {}, {}
            for _, t_info in (mot.get("per_team") or {}).items():
                a = t_info["arm"]
                sc = float(t_info["score_end"])
                cps = int(t_info["max_checkpoints_held"])
                push = int(t_info["cart_push_ticks"])
                sc_map[a] = sc
                comp_map[a] = sc + 5.0 * cps + 0.1 * push
            if len(sc_map) >= 2:
                score_matches.append(sc_map)
                composite_matches.append(comp_map)
                score_matches_by_map[mname].append(sc_map)
                composite_matches_by_map[mname].append(comp_map)

        fsq_stats = {}
        for arm_name, cmap in fsq_code_counts.items():
            tot_c = sum(cmap.values())
            if tot_c > 0:
                probs = np.asarray(list(cmap.values()), dtype=np.float64) / tot_c
                ent = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())
                fsq_stats[arm_name] = {
                    "unique_codes": len(cmap),
                    "total_samples": int(tot_c),
                    "code_entropy_bits": round(ent, 3),
                }

        return {
            "total_responses": total_responses,
            "sessions": len(telemetry_by_sess),
            "session_maps": {str(k): v[0] for k, v in sess_maps.items()},
            "responses_per_session": {str(k): len(v) for k, v in telemetry_by_sess.items()},
            "transitions": all_transitions,
            "n_transitions": len(all_transitions),
            "updates": len(update_logs),
            "deferred_batches": deferred_batches,
            "update_logs": update_logs,
            "online_cross_server_snr": summarize_mccandlish_pairs(online_snr_pairs),
            "online_cross_server_snr_by_group": {
                g: summarize_mccandlish_pairs(ps) for g, ps in online_group_pairs.items() if ps
            },
            "plan_focus_pw_rate": {a: float(np.mean(v)) for a, v in plan_focus_pw_hits.items() if v},
            "score_matches": score_matches,
            "composite_matches": composite_matches,
            "score_matches_by_map": dict(score_matches_by_map),
            "composite_matches_by_map": dict(composite_matches_by_map),
            "match_motifs": match_motifs,
            "trajectory_samples": dict(trajectory_samples),
            "team_dmg_matrix": dict(team_dmg_matrix),
            "fsq_stats": fsq_stats,
            "transport_errors": dict(engine.errors),
        }

    def _ppo_update_with_snr(
        self,
        model: PlanWally,
        optimizer: torch.optim.Optimizer,
        batch: list[PlanTransition],
        groups: list[int],
        *,
        ppo_epochs: int,
        grad_clip: float,
        device: str,
        step_id: int,
        online_snr_pairs: list[dict],
        online_group_pairs: dict[str, list[dict]],
        allow_snr_defer: bool = True,
    ) -> dict:
        # Map-stratified split into two balanced replica groups so both halves see identical map mix
        sess_by_map: dict[str, list[int]] = defaultdict(list)
        for tr in batch:
            if tr.session not in sess_by_map[tr.map_name]:
                sess_by_map[tr.map_name].append(tr.session)
        left_groups: set[int] = set()
        for mn in sorted(sess_by_map):
            s_list = sorted(sess_by_map[mn])
            for idx, s in enumerate(s_list):
                if idx % 2 == 0:
                    left_groups.add(s)
        half0 = [tr for tr in batch if tr.session in left_groups]
        half1 = [tr for tr in batch if tr.session not in left_groups]
        if not half0 or not half1:
            left_groups = set(groups[0::2])
            half0 = [tr for tr in batch if tr.session in left_groups]
            half1 = [tr for tr in batch if tr.session not in left_groups]
        can_split = len(half0) > 0 and len(half1) > 0
        ordered = (half0 + half1) if can_split else batch
        split_at = len(half0)
        last_diags = []
        last_gnorm = 0.0
        adv_mean, adv_std = compute_batch_adv_stats(model, ordered, device=device)

        for epoch in range(max(1, ppo_epochs)):
            optimizer.zero_grad(set_to_none=True)
            half0_grads = None
            inv_n = 1.0 / len(ordered)
            for idx, tr in enumerate(ordered):
                comps, diag = plan_losses_for_transition(
                    model, tr, device=device, adv_mean=adv_mean, adv_std=adv_std,
                )
                (comps["total"] * inv_n).backward()
                if epoch == 0:
                    last_diags.append(diag)
                if epoch == 0 and can_split and idx == split_at - 1:
                    half0_grads = {
                        name: p.grad.detach().cpu().to(torch.float64).clone()
                        for name, p in model.named_parameters()
                        if p.grad is not None
                    }
            if epoch == 0 and can_split and half0_grads is not None:
                s0 = len(ordered) / float(len(half0))
                s1 = len(ordered) / float(len(half1))
                g0 = {k: v * s0 for k, v in half0_grads.items()}
                g1 = {}
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    tot_cpu = p.grad.detach().cpu().to(torch.float64)
                    diff = tot_cpu - half0_grads[name] if name in half0_grads else tot_cpu
                    g1[name] = diff * s1
                pair = mccandlish_pair_from_grads(
                    [g0, g1], b0_transitions=0.5 * len(ordered), b0_rows=0.5 * len(ordered) * 9.0, generation=step_id
                )
                if pair is not None:
                    online_snr_pairs.append(pair)
                for grp in ("encoder", "gram_cgt", "resnet_ffn", "action_head", "value_head", "dynamics_head"):
                    sg0 = {k: v for k, v in g0.items() if plan_module_group(k) == grp}
                    sg1 = {k: v for k, v in g1.items() if plan_module_group(k) == grp}
                    if sg0 or sg1:
                        gp = mccandlish_pair_from_grads(
                            [sg0, sg1], b0_transitions=0.5 * len(ordered), b0_rows=0.5 * len(ordered) * 9.0, generation=step_id
                        )
                        if gp is not None:
                            online_group_pairs[grp].append(gp)
            last_gnorm = edm2_step(optimizer, model, grad_clip=grad_clip)

        def _m(k):
            vals = [d[k] for d in last_diags if d.get(k) is not None]
            return float(np.mean(vals)) if vals else None

        return {
            "step": step_id,
            "batch": len(batch),
            "groups": len(groups),
            "grad_norm": last_gnorm,
            "loss_total": _m("loss_total"),
            "loss_actor": _m("loss_actor"),
            "loss_w": _m("loss_w"),
            "loss_l": _m("loss_l"),
            "loss_dyn": _m("loss_dyn"),
            "entropy": _m("entropy"),
            "is_ratio_mean": _m("is_ratio_mean"),
            "winnie_mse": _m("winnie_mse"),
            "lou_mse": _m("lou_mse"),
        }
