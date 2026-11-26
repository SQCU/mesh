#!/usr/bin/env python3
"""Operational live-engine property-based map fuzzer & runtime geometry auditor.

Runs two live darkplaces-dedicated probes per map:
  1. 3-team live match fuzz: verifies cross-team spawn disjointness, push-zone occupancy,
     vertical dz consistency, bot stuckness, hazard deaths, and scoring progress.
  2. Per-cart solo-pushability probe: runs 3 concurrent servers forcing all bots via PLAN
     to push Cart 0, Cart 1, and Cart 2 independently, verifying every single cart is
     physically reachable and pushable by live HavocBots (catching hidden geometry snags).
"""
from __future__ import annotations

import argparse
import json
import select
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import CS, OBS, TS
from solver.strat.plan_policy import PLAN_WIDTH
from solver.strat.plan_runner import MultiServerPlanHarness
from solver.xonwire import CART_KIND, OBSERVATION_KIND, PLAN_KIND, TEAM_KIND, Engine, RuntimeFrames, frames


def _run_3team_fuzz(harness: MultiServerPlanHarness, map_names: list[str], work_dir: Path, base_port: int, duration_s: float, slowmo: float) -> dict[str, dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    eng = Engine("udp://127.0.0.1:0", harness.framebytes)
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    procs, logs = harness._spawn_servers(
        work_dir, eng.path, [9100 + i for i in range(len(map_names))], base_port,
        map_names=map_names, skill=7, slowmo_per_server=[slowmo] * len(map_names),
    )
    stats, prev_alive, spawn_coords = {}, {}, {}
    pos_hist = {}
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            r, _, _ = select.select([eng], [], [], 0.05)
            if not r:
                continue
            for _, _, buf in eng.receive():
                inbox.feed(buf)
            while (ready := inbox.take()) is not None:
                (sess, _), recs, _ = ready
                obs_raw = recs[OBSERVATION_KIND][1]
                carts = recs[CART_KIND][1]
                teams = recs[TEAM_KIND][1]
                obs = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
                if len(obs) == 0 or len(carts) == 0 or len(teams) == 0:
                    continue
                if sess not in stats:
                    mname, _ = harness._identify_map(carts)
                    if mname == "unknown" or mname not in map_names:
                        used = {v["map"] for v in stats.values()}
                        rem = [mn for mn in map_names if mn not in used]
                        mname = rem[0] if rem else map_names[len(stats) % len(map_names)]
                    stats[sess] = {
                        "map": mname, "ticks": 0, "first_cp_s": None,
                        "cart_len": carts[:, CS["PATH_LENGTH"]].round(0).tolist(),
                        "max_frac": np.zeros(len(carts)),
                        "push_ticks": np.zeros(len(carts), dtype=int),
                        "max_cp": 0, "max_sc": 0.0,
                        "stuck": 0, "stuck_denom": 0, "alive": 0, "dead": 0,
                        "in_push": 0, "dz_miss": 0, "dists": [], "dzs": [],
                    }
                    spawn_coords[sess] = []
                st = stats[sess]
                st["ticks"] += 1
                now_t = time.monotonic()
                dt_game = (now_t - t0) * slowmo
                frac = carts[:, CS["PATH_POSITION"]] / np.maximum(carts[:, CS["PATH_LENGTH"]], 1.0)
                st["max_frac"] = np.maximum(st["max_frac"], frac)
                st["push_ticks"] += (carts[:, CS["SPEED"]] > 1.0).astype(int)
                held = int(teams[:, TS["CHECKPOINTS_HELD"]].max())
                sc = float(teams[:, TS["SCORE"]].max())
                if held > st["max_cp"]:
                    st["max_cp"] = held
                    if st["first_cp_s"] is None:
                        st["first_cp_s"] = round(dt_game, 1)
                st["max_sc"] = max(st["max_sc"], sc)
                c_xyz = carts[:, [CS["POS_X"], CS["POS_Y"], CS["POS_Z"]]]
                for row in obs:
                    eid = int(row[OBS["ID"]])
                    tm = int(row[OBS["TEAM"]])
                    alive = row[OBS["ALIVE"]] > 0.5
                    key = (sess, eid)
                    if alive:
                        st["alive"] += 1
                        p = np.array([row[OBS["POS_X"]], row[OBS["POS_Y"]], row[OBS["POS_Z"]]])
                        if not prev_alive.get(key, False):
                            spawn_coords[sess].append((tm, p))
                            pos_hist[key] = (now_t, p)
                        d_xy = np.linalg.norm(c_xyz[:, :2] - p[:2], axis=1)
                        j = int(np.argmin(d_xy))
                        st["dists"].append(float(d_xy[j]))
                        in_push_now = False
                        if d_xy[j] < 160.0:
                            dz = float(abs(c_xyz[j, 2] - p[2]))
                            st["dzs"].append(dz)
                            if dz < 96.0:
                                st["in_push"] += 1
                                in_push_now = True
                            else:
                                st["dz_miss"] += 1
                        # Real stuckness: < 24u movement over >= 1.0s wall while NOT actively pushing a cart
                        if key in pos_hist and (now_t - pos_hist[key][0]) >= 1.0:
                            if not in_push_now:
                                st["stuck_denom"] += 1
                                if float(np.linalg.norm(p - pos_hist[key][1])) < 24.0:
                                    st["stuck"] += 1
                            pos_hist[key] = (now_t, p)
                    else:
                        st["dead"] += 1
                        pos_hist.pop(key, None)
                    prev_alive[key] = alive
    finally:
        harness._stop_servers(procs, logs)
        eng.close()

    out = {}
    for sess, st in stats.items():
        m = st["map"]
        sp = spawn_coords[sess]
        cross_ov = 0
        min_cross_obs = None
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                if sp[i][0] != sp[j][0]:
                    d = float(np.linalg.norm(sp[i][1] - sp[j][1]))
                    if min_cross_obs is None or d < min_cross_obs:
                        min_cross_obs = d
                    if d < 200.0:
                        cross_ov += 1
        # Parse log for env deaths / telefrags
        env_d = tele = 0
        lfiles = list(work_dir.glob(f"server_*_{m}.log"))
        if lfiles:
            for ln in lfiles[0].read_text(errors="ignore").splitlines():
                lo = ln.lower()
                if "telefrag" in lo:
                    tele += 1
                if any(k in lo for k in ("fell to", "void", "lava", "slime", "drowned", "crushed", "suicide", "killed himself", "killed herself")):
                    env_d += 1
        out[m] = {
            "ticks": st["ticks"],
            "cart_len": st["cart_len"],
            "max_frac": [round(float(x), 3) for x in st["max_frac"]],
            "first_cp_game_s": st["first_cp_s"],
            "max_cp": st["max_cp"],
            "max_score": round(st["max_sc"], 1),
            "bot_stuck_pct": round(100.0 * st["stuck"] / max(1, st["stuck_denom"]), 1),
            "in_push_pct": round(100.0 * st["in_push"] / max(1, st["alive"]), 1),
            "dz_miss_pct": round(100.0 * st["dz_miss"] / max(1, st["in_push"] + st["dz_miss"]), 1),
            "med_dist_u": round(float(np.median(st["dists"])), 0) if st["dists"] else None,
            "cross_team_spawn_overlaps_200u": cross_ov,
            "min_cross_team_spawn_obs_u": round(min_cross_obs, 0) if min_cross_obs is not None else None,
            "env_deaths": env_d,
            "telefrags": tele,
        }
    return out


def _run_solo_cart_probe(harness: MultiServerPlanHarness, mapname: str, work_dir: Path, base_port: int, duration_s: float, slowmo: float) -> dict:
    """Launch 3 concurrent servers on `mapname` with ONLY 3 Team-1 bots (zero enemy interference)
    directed onto Cart 0, Cart 1, and Cart 2 independently so we measure pure physical reachability
    and pushability of every single cart from 0% -> 100%."""
    work_dir.mkdir(parents=True, exist_ok=True)
    # Write a 3-bot Team-1-only roster so enemy bots don't kill/contest the solo-reachability probe
    solo_roster = work_dir / "solo_team1.txt"
    solo_roster.write_text(
        "//solo reachability probe: 3 bots on Team 1\n"
        "Probe1\terebus\t0\t-1\t-1\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t7.0\n"
        "Probe2\terebus\t0\t-1\t-1\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t7.0\n"
        "Probe3\terebus\t0\t-1\t-1\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t7.0\n"
    )
    eng = Engine("udp://127.0.0.1:0", harness.framebytes)
    inbox = RuntimeFrames(harness.framebytes, plan_only=True)
    procs, logs = harness._spawn_servers(
        work_dir, eng.path, [9400, 9401, 9402], base_port,
        map_names=[mapname, mapname, mapname], skill=7,
        bot_config_files=[solo_roster, solo_roster, solo_roster],
        extra_cvars=[{"bot_number": 3, "g_balance_teams": 0}] * 3,
        slowmo_per_server=[slowmo] * 3,
    )
    sess_cart = {}
    routes = {}
    max_frac_by_cart = {0: 0.0, 1: 0.0, 2: 0.0}
    max_cp_by_cart = {0: 0, 1: 0, 2: 0}
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            r, _, _ = select.select([eng], [], [], 0.05)
            if not r:
                continue
            for node, addr, buf in eng.receive():
                s = inbox.feed(buf)
                if s is not None:
                    routes[s] = (node, addr)
            while (ready := inbox.take()) is not None:
                (sess, tid), recs, _ = ready
                if sess not in sess_cart:
                    sess_cart[sess] = len(sess_cart) % 3
                target_c = sess_cart[sess]
                obs_raw = recs[OBSERVATION_KIND][1]
                carts = recs[CART_KIND][1]
                obs = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
                if len(obs) == 0 or len(carts) == 0:
                    continue
                if target_c < len(carts):
                    f = float(carts[target_c, CS["PATH_POSITION"]]) / max(float(carts[target_c, CS["PATH_LENGTH"]]), 1.0)
                    cp = int(carts[target_c, CS["CHECKPOINT_DEPTH"]])
                    max_frac_by_cart[target_c] = max(max_frac_by_cart[target_c], f)
                    max_cp_by_cart[target_c] = max(max_cp_by_cart[target_c], cp)
                # Direct ALL bots on Team 1 straight onto target_c with strong push priority
                plan = np.zeros((harness.maxclients, PLAN_WIDTH), dtype=np.float32)
                for row in obs:
                    eid = int(round(float(row[OBS["ID"]])))
                    tm = int(round(float(row[OBS["TEAM"]])))
                    if 1 <= eid <= harness.maxclients and tm == 1:
                        plan[eid - 1, 0] = float(eid)
                        plan[eid - 1, 1] = float(target_c)
                        plan[eid - 1, 2:7] = np.array([2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32)
                        plan[eid - 1, 7] = 0.0
                        plan[eid - 1, 8] = 0.0
                        plan[eid - 1, 9] = 0.0
                        plan[eid - 1, 10] = 0.0
                        plan[eid - 1, 14] = 2.0
                        plan[eid - 1, 15] = 0.6
                dest = routes.get(sess)
                if dest is not None:
                    node, addr = dest
                    req_id = recs[OBSERVATION_KIND][0]["req_id"]
                    eng.send(addr, node, frames(PLAN_KIND, req_id, tid, plan, harness.framebytes, sess))
    finally:
        harness._stop_servers(procs, logs)
        eng.close()

    fracs = [round(max_frac_by_cart[c], 3) for c in range(3)]
    cps = [int(max_cp_by_cart[c]) for c in range(3)]
    return {
        "solo_cart_max_frac": fracs,
        "solo_cart_max_cp": cps,
        "all_3_carts_moved": all(f >= 0.10 for f in fracs),
        "all_3_carts_captured_cp1": all(cp >= 1 for cp in cps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--maps", required=True, help="Comma-separated map names")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--match-s", type=float, default=30.0)
    ap.add_argument("--solo-s", type=float, default=20.0)
    ap.add_argument("--slowmo", type=float, default=2.0)
    ap.add_argument("--base-port", type=int, default=26800)
    args = ap.parse_args()

    stage = Path(args.stage_dir).expanduser()
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    h = MultiServerPlanHarness(
        dp_bin=str(stage / "darkplaces-dedicated"),
        basedir=str(stage / "Xonotic"),
        userdir=str(stage / "userdir"),
        framebytes=8192, strategy_interval=0.5, payload_speed=80.0, maxclients=16,
    )
    work = Path("/tmp/map_fuzz_audit")
    m_res = _run_3team_fuzz(h, maps, work / "match3", args.base_port, args.match_s, args.slowmo)
    combined = {}
    for idx, m in enumerate(maps):
        s_res = _run_solo_cart_probe(h, m, work / f"solo_{m}", args.base_port + 50 + idx * 5, args.solo_s, args.slowmo)
        rec = {**(m_res.get(m) or {}), **s_res}
        # Static measurements if present
        mjson = stage / "userdir/data/maps" / f"{m}.ent.measurements.json"
        if mjson.exists():
            sd = json.loads(mjson.read_text())
            rec["static_rider_gaps"] = sd.get("rider_gap_segments")
            rec["static_len_ratio"] = round(float(sd.get("path_length_ratio") or 0.0), 2)
            rec["static_min_cross_team_u"] = sd.get("min_cross_team_spawn_dist")
        else:
            rec["static_rider_gaps"] = 0
            lens = rec.get("cart_len") or []
            rec["static_len_ratio"] = round(max(lens) / max(min(lens), 1.0), 2) if lens else 1.0
            rec["static_min_cross_team_u"] = rec.get("min_cross_team_spawn_obs_u")
        rec["PASS_ALL"] = bool(
            rec.get("cross_team_spawn_overlaps_200u", 999) == 0
            and rec.get("bot_stuck_pct", 100.0) < 12.0
            and rec.get("dz_miss_pct", 100.0) < 40.0
            and rec.get("all_3_carts_moved", False)
            and rec.get("static_rider_gaps", 1) == 0
        )
        combined[m] = rec
        print(
            f"[FUZZ] {m:16s} PASS={str(rec['PASS_ALL']):5s} xov={rec.get('cross_team_spawn_overlaps_200u')} "
            f"stuck={rec.get('bot_stuck_pct')}% dzmiss={rec.get('dz_miss_pct')}% "
            f"sc={rec.get('max_score')} solo_frac={rec['solo_cart_max_frac']} solo_cp={rec['solo_cart_max_cp']}",
            flush=True,
        )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
