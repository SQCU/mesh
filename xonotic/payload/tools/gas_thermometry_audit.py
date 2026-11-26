#!/usr/bin/env python3
"""Live Kinetic-Gas & Membrane-Joint Thermometry Auditor for Multi-Tile Payload Worlds.

Measures directly from darkplaces-dedicated live 6D phase-space observations (POS_X..Z, VEL_X..Z):
  - Bulk gas state: per-tile density n_m (players/Mqu^2), spatial density CV, v_rms, T_bulk
  - Membrane thermometry per joint e=(A,B): crossing flux Q_e (players/s), transit temperature T_e,
    thermal transmission ratio theta_e = T_e / T_bulk, normal-flow efficiency eta_perp, and
    empirical Clausing transmission probability kappa_e = Q_e / ((n_A v_A + n_B v_B) * W_e / pi)
  - Multi-scale gameplay regularity: cross-team spawn overlaps, >=1s non-pushing stuck rate,
    active cart fraction, checkpoint captures, and distinct teams scoring (up to K=24, J=256, N=255).
"""
from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import CS, OBS, TS
from solver.strat.plan_runner import MultiServerPlanHarness
from solver.xonwire import CART_KIND, OBSERVATION_KIND, TEAM_KIND, Engine, RuntimeFrames


def _tile_of(x: float, y: float, z: float, tiles: list[dict]) -> int:
    for t in tiles:
        if t["x0"] <= x <= t["x1"] and t["y0"] <= y <= t["y1"]:
            if "z0" not in t or (t["z0"] <= z <= t["z1"]):
                return int(t["id"])
    # Inside a joint throat or slightly outside AABB: assign nearest tile center
    best_id, best_d2 = -1, float("inf")
    for t in tiles:
        cx, cy = t["center"][0], t["center"][1]
        cz = t["center"][2] if len(t["center"]) >= 3 else 0.0
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        if d2 < best_d2:
            best_d2, best_id = d2, int(t["id"])
    return best_id


def run_gas_audit(
    stage_dir: Path,
    mapname: str,
    joints_path: Path,
    *,
    teams: int,
    bots: int,
    duration_s: float,
    slowmo: float,
    base_port: int,
) -> dict:
    jdoc = json.loads(joints_path.read_text())
    tiles = jdoc["tiles"]
    joints = jdoc["joints"]
    maxclients = min(255, max(16, bots + 4))

    harness = MultiServerPlanHarness(
        dp_bin=str(stage_dir / "darkplaces-dedicated"),
        basedir=str(stage_dir / "Xonotic"),
        userdir=str(stage_dir / "userdir"),
        framebytes=8192,
        strategy_interval=0.5,
        payload_speed=80.0,
        maxclients=maxclients,
    )
    work_dir = Path(f"/tmp/gas_audit_{mapname}")
    work_dir.mkdir(parents=True, exist_ok=True)
    eng = Engine("udp://127.0.0.1:0", harness.framebytes)
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    procs, logs = harness._spawn_servers(
        work_dir, eng.path, [9600], base_port,
        map_names=[mapname], skill=7,
        extra_cvars=[{"g_payload_teams_override": int(teams), "bot_number": int(bots)}],
        slowmo_per_server=[slowmo],
    )

    ticks = 0
    tile_occ_sum = np.zeros((len(tiles),), dtype=np.float64)
    tile_v2_sum = np.zeros((len(tiles),), dtype=np.float64)
    tile_v_sum = np.zeros((len(tiles),), dtype=np.float64)
    tile_cnt = np.zeros((len(tiles),), dtype=np.int64)

    joint_crossings = np.zeros((len(joints),), dtype=np.int64)
    joint_v2_samples: list[list[float]] = [[] for _ in joints]
    joint_vn2_samples: list[list[float]] = [[] for _ in joints]

    prev_pos: dict[int, np.ndarray] = {}
    prev_tile: dict[int, int] = {}
    prev_alive: dict[int, bool] = {}
    pos_hist: dict[int, tuple[float, np.ndarray]] = {}
    spawn_coords: list[tuple[int, np.ndarray]] = []
    stuck_n = stuck_d = alive_n = in_push_n = dz_miss_n = 0

    max_frac = None
    max_cp = None
    max_team_scores = None
    n_carts_seen = 0
    n_teams_seen = 0
    n_bots_max = 0
    first_tick_t = None

    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            r, _, _ = select.select([eng], [], [], 0.05)
            if not r:
                continue
            for _, _, buf in eng.receive():
                inbox.feed(buf)
            while (ready := inbox.take()) is not None:
                _, recs, _ = ready
                obs_raw = recs[OBSERVATION_KIND][1]
                carts = recs[CART_KIND][1]
                team_rows = recs[TEAM_KIND][1]
                obs = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
                if len(obs) == 0 or len(carts) == 0 or len(team_rows) == 0:
                    continue
                now_t = time.monotonic()
                if first_tick_t is None:
                    first_tick_t = now_t
                ticks += 1
                n_carts_seen = max(n_carts_seen, len(carts))
                n_teams_seen = max(n_teams_seen, len(team_rows))
                n_bots_max = max(n_bots_max, len(obs))

                frac = carts[:, CS["PATH_POSITION"]] / np.maximum(carts[:, CS["PATH_LENGTH"]], 1.0)
                cp_d = carts[:, CS["CHECKPOINT_DEPTH"]].astype(np.int64)
                if max_frac is None or len(max_frac) != len(frac):
                    max_frac = frac.copy()
                    max_cp = cp_d.copy()
                else:
                    max_frac = np.maximum(max_frac, frac)
                    max_cp = np.maximum(max_cp, cp_d)

                tsc = team_rows[:, TS["SCORE"]].astype(np.float64)
                if max_team_scores is None or len(max_team_scores) != len(tsc):
                    max_team_scores = tsc.copy()
                else:
                    max_team_scores = np.maximum(max_team_scores, tsc)

                c_xy = carts[:, [CS["POS_X"], CS["POS_Y"]]]
                c_z = carts[:, CS["POS_Z"]]

                for row in obs:
                    eid = int(row[OBS["ID"]])
                    tm = int(row[OBS["TEAM"]])
                    alive = row[OBS["ALIVE"]] > 0.5
                    if not alive:
                        prev_alive[eid] = False
                        prev_pos.pop(eid, None)
                        prev_tile.pop(eid, None)
                        pos_hist.pop(eid, None)
                        continue
                    alive_n += 1
                    p = np.array([row[OBS["POS_X"]], row[OBS["POS_Y"]], row[OBS["POS_Z"]]], dtype=np.float64)
                    v = np.array([row[OBS["VEL_X"]], row[OBS["VEL_Y"]], row[OBS["VEL_Z"]]], dtype=np.float64)
                    v_xy2 = float(v[0] * v[0] + v[1] * v[1])
                    v_xy = math.sqrt(v_xy2)

                    if not prev_alive.get(eid, False):
                        spawn_coords.append((tm, p))
                        pos_hist[eid] = (now_t, p)

                    # Bulk tile density & temperature accumulation
                    tid = _tile_of(p[0], p[1], p[2], tiles)
                    if 0 <= tid < len(tiles):
                        tile_occ_sum[tid] += 1.0
                        tile_v2_sum[tid] += v_xy2
                        tile_v_sum[tid] += v_xy
                        tile_cnt[tid] += 1

                    # Membrane joint crossing & throat thermometry (physical doorways + B3313 portal membranes)
                    p_old = prev_pos.get(eid)
                    t_old_id = prev_tile.get(eid)
                    for ji, jt in enumerate(joints):
                        if jt.get("type") == "portal":
                            ta, tb = int(jt["tile_a"]), int(jt["tile_b"])
                            hw = 0.5 * float(jt["width"]) + 48.0
                            hl = 112.0
                            # Check vestibule proximity on either side of the portal membrane
                            for o_key, n_key in (("origin", "normal"), ("origin_b", "normal_b")):
                                if o_key not in jt:
                                    continue
                                ox, oy, oz = float(jt[o_key][0]), float(jt[o_key][1]), float(jt[o_key][2])
                                nx, ny = float(jt[n_key][0]), float(jt[n_key][1])
                                tx, ty = -ny, nx
                                dx, dy, dz = p[0] - ox, p[1] - oy, p[2] - oz
                                if abs(dz) <= 112.0 and abs(dx * nx + dy * ny) <= hl and abs(dx * tx + dy * ty) <= hw:
                                    vn = v[0] * nx + v[1] * ny
                                    joint_v2_samples[ji].append(v_xy2)
                                    joint_vn2_samples[ji].append(float(vn * vn))
                            # Portal transit crossing when a live player jumps tile_a <-> tile_b
                            if prev_alive.get(eid, False) and t_old_id is not None and t_old_id != tid:
                                if (t_old_id == ta and tid == tb) or (t_old_id == tb and tid == ta):
                                    joint_crossings[ji] += 1
                                    n_use = jt["normal_b"] if tid == tb else jt["normal"]
                                    vn = v[0] * float(n_use[0]) + v[1] * float(n_use[1])
                                    joint_v2_samples[ji].append(v_xy2)
                                    joint_vn2_samples[ji].append(float(vn * vn))
                            continue

                        ox, oy = float(jt["origin"][0]), float(jt["origin"][1])
                        nx, ny = float(jt["normal"][0]), float(jt["normal"][1])
                        tx, ty = -ny, nx
                        hw = 0.5 * float(jt["width"]) + 48.0
                        hl = 0.5 * float(jt["length"]) + 80.0
                        dx, dy = p[0] - ox, p[1] - oy
                        s_norm = dx * nx + dy * ny
                        s_tang = dx * tx + dy * ty
                        if abs(s_norm) <= hl and abs(s_tang) <= hw:
                            vn = v[0] * nx + v[1] * ny
                            joint_v2_samples[ji].append(v_xy2)
                            joint_vn2_samples[ji].append(float(vn * vn))
                        if p_old is not None and prev_alive.get(eid, False):
                            odx, ody = p_old[0] - ox, p_old[1] - oy
                            s_old = odx * nx + ody * ny
                            t_old = odx * tx + ody * ty
                            if s_old * s_norm < 0.0 and max(abs(s_tang), abs(t_old)) <= hw and max(abs(s_norm), abs(s_old)) <= 450.0:
                                joint_crossings[ji] += 1

                    # Nearest cart push-zone & real >=1s non-pushing stuck check
                    d_xy = np.linalg.norm(c_xy - p[:2], axis=1)
                    j = int(np.argmin(d_xy))
                    in_push_now = False
                    if d_xy[j] < 160.0:
                        if abs(float(c_z[j]) - p[2]) < 96.0:
                            in_push_n += 1
                            in_push_now = True
                        else:
                            dz_miss_n += 1
                    if eid in pos_hist and (now_t - pos_hist[eid][0]) >= 1.0:
                        if not in_push_now:
                            stuck_d += 1
                            if float(np.linalg.norm(p - pos_hist[eid][1])) < 24.0:
                                stuck_n += 1
                        pos_hist[eid] = (now_t, p)

                    prev_pos[eid] = p
                    prev_tile[eid] = tid
                    prev_alive[eid] = True
    finally:
        harness._stop_servers(procs, logs)
        eng.close()

    # Compute bulk gas & joint membrane thermodynamic metrics
    eff_game_s = max(1.0, (time.monotonic() - (first_tick_t or t0)) * slowmo)
    tile_n = np.zeros((len(tiles),), dtype=np.float64)
    tile_vmean = np.zeros((len(tiles),), dtype=np.float64)
    tile_vrms = np.zeros((len(tiles),), dtype=np.float64)
    for m, t in enumerate(tiles):
        occ = tile_occ_sum[m] / max(1, ticks)
        tile_n[m] = occ / max(float(t["area_mqu2"]), 1e-6)
        if tile_cnt[m] > 0:
            tile_vmean[m] = tile_v_sum[m] / tile_cnt[m]
            tile_vrms[m] = math.sqrt(tile_v2_sum[m] / tile_cnt[m])

    tot_cnt = int(tile_cnt.sum())
    bulk_v2 = float(tile_v2_sum.sum() / max(1, tot_cnt))
    bulk_vrms = math.sqrt(bulk_v2)
    bulk_vmean = float(tile_v_sum.sum() / max(1, tot_cnt))
    mean_n = float(tile_n.mean()) if len(tile_n) else 0.0
    n_cv = float(tile_n.std() / max(mean_n, 1e-6)) if len(tile_n) > 1 else 0.0

    joint_reports = []
    for ji, jt in enumerate(joints):
        ta, tb = int(jt["tile_a"]), int(jt["tile_b"])
        W_e = float(jt["width"])
        Q_obs = float(joint_crossings[ji]) / eff_game_s
        # Theoretical ideal 2-way 2D effusion flux: (n_A v_A + n_B v_B) * W_e / pi (n in players/qu^2)
        Q_ideal = ((tile_n[ta] * 1e-6 * tile_vmean[ta]) + (tile_n[tb] * 1e-6 * tile_vmean[tb])) * W_e / math.pi
        kappa = (Q_obs / Q_ideal) if Q_ideal > 1e-6 else None
        v2s = joint_v2_samples[ji]
        vn2s = joint_vn2_samples[ji]
        if v2s:
            t_e = float(np.mean(v2s))
            theta = t_e / max(bulk_v2, 1.0)
            eta_perp = float(np.mean(vn2s)) / max(t_e, 1.0)
            vrms_e = math.sqrt(t_e)
        else:
            theta = eta_perp = vrms_e = None
        joint_reports.append({
            "id": ji, "width": W_e, "crossings": int(joint_crossings[ji]),
            "Q_obs_per_s": round(Q_obs, 4), "Q_ideal_per_s": round(Q_ideal, 4),
            "kappa_clausing": round(kappa, 3) if kappa is not None else None,
            "vrms_transit": round(vrms_e, 1) if vrms_e is not None else None,
            "theta_T_ratio": round(theta, 3) if theta is not None else None,
            "eta_perp": round(eta_perp, 3) if eta_perp is not None else None,
        })

    # Aggregate membrane summary across all active joints
    act_j = [r for r in joint_reports if r["theta_T_ratio"] is not None]
    kappas = [r["kappa_clausing"] for r in joint_reports if r["kappa_clausing"] is not None]

    # Cross-team spawn overlaps (< 200u)
    xov = 0
    min_x = None
    # Subsample spawn_coords if huge (N=240)
    sp = spawn_coords[:400]
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            if sp[i][0] != sp[j][0]:
                d = float(np.linalg.norm(sp[i][1] - sp[j][1]))
                if min_x is None or d < min_x:
                    min_x = d
                if d < 200.0:
                    xov += 1

    # Log parse
    env_d = tele = 0
    lfiles = list(work_dir.glob(f"server_0_{mapname}.log"))
    if lfiles:
        for ln in lfiles[0].read_text(errors="ignore").splitlines():
            lo = ln.lower()
            if "telefrag" in lo:
                tele += 1
            if any(k in lo for k in ("fell to", "void", "lava", "slime", "drowned", "crushed", "suicide", "killed himself")):
                env_d += 1

    carts_moved = int((max_frac >= 0.05).sum()) if max_frac is not None else 0
    carts_cp1 = int((max_cp >= 1).sum()) if max_cp is not None else 0
    teams_scoring = int((max_team_scores > 0.0).sum()) if max_team_scores is not None else 0

    return {
        "map": mapname,
        "requested": {"teams": teams, "bots": bots, "tiles": len(tiles), "joints": len(joints), "joint_width_u": jdoc["joint_width_u"]},
        "observed": {"ticks": ticks, "eff_game_s": round(eff_game_s, 1), "bots_max": n_bots_max, "teams_seen": n_teams_seen, "carts_seen": n_carts_seen},
        "bulk_gas": {
            "mean_density_per_mqu2": round(mean_n, 2),
            "tile_density_cv": round(n_cv, 3),
            "v_mean_u_s": round(bulk_vmean, 1),
            "v_rms_u_s": round(bulk_vrms, 1),
        },
        "membrane_thermometry": {
            "total_crossings": int(joint_crossings.sum()),
            "mean_Q_obs_per_joint_s": round(float(np.mean([r["Q_obs_per_s"] for r in joint_reports])), 4) if joint_reports else 0.0,
            "mean_Q_ideal_per_joint_s": round(float(np.mean([r["Q_ideal_per_s"] for r in joint_reports])), 4) if joint_reports else 0.0,
            "mean_kappa_clausing": round(float(np.mean(kappas)), 3) if kappas else None,
            "mean_vrms_transit": round(float(np.mean([r["vrms_transit"] for r in act_j])), 1) if act_j else None,
            "mean_theta_T_ratio": round(float(np.mean([r["theta_T_ratio"] for r in act_j])), 3) if act_j else None,
            "mean_eta_perp": round(float(np.mean([r["eta_perp"] for r in act_j])), 3) if act_j else None,
        },
        "gameplay_regularity": {
            "cross_team_spawn_overlaps_200u": xov,
            "min_cross_team_spawn_obs_u": round(min_x, 0) if min_x is not None else None,
            "bot_stuck_pct": round(100.0 * stuck_n / max(1, stuck_d), 2),
            "in_push_pct": round(100.0 * in_push_n / max(1, alive_n), 2),
            "dz_miss_pct": round(100.0 * dz_miss_n / max(1, in_push_n + dz_miss_n), 2),
            "carts_moved_5pct": f"{carts_moved}/{n_carts_seen}",
            "carts_captured_cp1": f"{carts_cp1}/{n_carts_seen}",
            "teams_scoring": f"{teams_scoring}/{n_teams_seen}",
            "max_team_score": round(float(max_team_scores.max()), 1) if max_team_scores is not None and len(max_team_scores) else 0.0,
            "env_deaths": env_d,
            "telefrags": tele,
        },
        "joints": joint_reports,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--map", required=True)
    ap.add_argument("--joints-json", required=True)
    ap.add_argument("--teams", type=int, required=True)
    ap.add_argument("--bots", type=int, required=True)
    ap.add_argument("--duration-s", type=float, default=35.0)
    ap.add_argument("--slowmo", type=float, default=1.5)
    ap.add_argument("--base-port", type=int, default=26900)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    res = run_gas_audit(
        Path(args.stage_dir).expanduser(),
        args.map,
        Path(args.joints_json).expanduser(),
        teams=args.teams,
        bots=args.bots,
        duration_s=args.duration_s,
        slowmo=args.slowmo,
        base_port=args.base_port,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res, indent=2))
    g, m, r = res["bulk_gas"], res["membrane_thermometry"], res["gameplay_regularity"]
    print(
        f"[GAS] {args.map:16s} K={res['observed']['teams_seen']:2d} J={res['observed']['carts_seen']:3d} N={res['observed']['bots_max']:3d} "
        f"| n={g['mean_density_per_mqu2']:4.2f}/Mqu2 (cv={g['tile_density_cv']:.2f}) v_bulk={g['v_rms_u_s']:5.1f} "
        f"| Q={m['mean_Q_obs_per_joint_s']:.3f}/s v_tr={m['mean_vrms_transit']} theta={m['mean_theta_T_ratio']} eta={m['mean_eta_perp']} kappa={m['mean_kappa_clausing']} "
        f"| xov={r['cross_team_spawn_overlaps_200u']} stuck={r['bot_stuck_pct']}% moved={r['carts_moved_5pct']} cp1={r['carts_captured_cp1']} sc_teams={r['teams_scoring']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
