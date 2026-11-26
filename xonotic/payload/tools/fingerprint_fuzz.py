#!/usr/bin/env python3
"""Cart-Free 1v1/2v2/3v3 DM/TDM & B3313 Supermap Phase-Space Fingerprint Fuzz Runner.

Runs live darkplaces-dedicated matches across:
  - Pure cart-free Deathmatch / Team Deathmatch (`g_dm 1` / `g_tdm 1`, `g_mesh_telemetry 1`)
  - Payload stock maps and B3313 real-map variant supermaps (`g_payload 1`)
and extracts the full 5-layer `PhaseSpaceFingerprint` + optional portal-membrane thermometry.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import OBS
from solver.strat.phase_fingerprint import PhaseSpaceAccumulator, compare_fingerprints, strip_raw_maps
from solver.xonwire import EVENT_KIND, OBSERVATION_KIND, Engine, RuntimeFrames


def run_live_fingerprint(
    stage_dir: Path,
    mapname: str,
    *,
    mode: str,         # "dm", "tdm", "plc"
    teams: int,
    bots: int,
    duration_s: float,
    slowmo: float,
    base_port: int,
    label: str,
) -> dict:
    work_dir = Path(f"/tmp/fp_audit_{label}")
    work_dir.mkdir(parents=True, exist_ok=True)
    sv_userdir = work_dir / "userdir_0"
    sv_data = sv_userdir / "data"
    sv_data.mkdir(parents=True, exist_ok=True)
    for item in (stage_dir / "userdir" / "data").iterdir():
        dst = sv_data / item.name
        if not dst.exists():
            os.symlink(item, dst)

    eng = Engine("udp://127.0.0.1:0", 8192)
    inbox = RuntimeFrames(8192, plan_only=True)
    logf = open(work_dir / f"server_{label}.log", "w")

    env = dict(os.environ)
    env["MESH_EXPERT_SOCKET"] = eng.path
    env["MESH_EXPERT_FRAMEBYTES"] = "8192"

    maxclients = min(255, max(16, bots + 4))
    cmd = [
        str(stage_dir / "darkplaces-dedicated"), "-xonotic",
        "-basedir", str(stage_dir / "Xonotic"),
        "-userdir", str(sv_userdir),
        "+sv_public", "0",
        "+port", str(base_port),
        "+maxplayers", str(maxclients),
        "+sv_random_seed", "9700",
        "+g_warmup", "0",
        "+g_start_delay", "0",
        "+bot_join_empty", "1",
        "+bot_number", str(bots),
        "+skill", "7",
        "+g_payload_strategy_interval", "0.2",
        "+g_payload_plan_only", "1",
    ]
    if mode == "dm":
        cmd.extend(["+exec", "gamemodes-payload.cfg", "+g_dm", "1", "+g_tdm", "0", "+g_payload", "0", "+set", "g_mesh_telemetry", "1", "+fraglimit", "0", "+timelimit", "0"])
    elif mode == "tdm":
        cmd.extend(["+exec", "gamemodes-payload.cfg", "+g_dm", "0", "+g_tdm", "1", "+g_tdm_teams_override", str(teams), "+g_payload", "0", "+set", "g_mesh_telemetry", "1", "+fraglimit", "0", "+timelimit", "0"])
    else:
        cmd.extend(["+exec", "gamemodes-payload.cfg", "+g_payload", "1", "+g_payload_teams_override", str(teams), "+g_payload_warmup", "1", "+g_spawn_swizzle", "1", "+g_payload_speed", "80"])
    if slowmo != 1.0:
        cmd.extend(["+slowmo", str(slowmo)])
    cmd.extend(["+map", mapname])

    proc = subprocess.Popen(cmd, cwd=str(stage_dir / "Xonotic"), env=env, stdout=logf, stderr=subprocess.STDOUT)
    acc = PhaseSpaceAccumulator(label=label, cell_size=256.0)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            r, _, _ = select.select([eng], [], [], 0.05)
            if not r:
                continue
            for _, _, buf in eng.receive():
                inbox.feed(buf)
            while (ready := inbox.take()) is not None:
                _, recs, evts = ready
                obs_raw = recs.get(OBSERVATION_KIND, (None, None))[1]
                evt_raw = np.concatenate(evts, axis=0) if evts else None
                acc.update(obs_raw, evt_raw)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        eng.close()

    fp = acc.finalize()
    fp["meta"] = {"map": mapname, "mode": mode, "teams": teams, "bots": bots}
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--map", required=True)
    ap.add_argument("--mode", choices=("dm", "tdm", "plc"), required=True)
    ap.add_argument("--teams", type=int, default=2)
    ap.add_argument("--bots", type=int, default=6)
    ap.add_argument("--duration-s", type=float, default=35.0)
    ap.add_argument("--slowmo", type=float, default=1.5)
    ap.add_argument("--base-port", type=int, default=26970)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    fp = run_live_fingerprint(
        Path(args.stage_dir).expanduser(),
        args.map,
        mode=args.mode, teams=args.teams, bots=args.bots,
        duration_s=args.duration_s, slowmo=args.slowmo,
        base_port=args.base_port, label=args.label,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(fp, indent=2))
    l1, l2, l3, l4, l5 = (
        fp["L1_kinetic_spatial"], fp["L2_markov_flux"], fp["L3_pair_correlation"],
        fp["L4_combat_thermometry"], fp["L5_resource_weapon"],
    )
    print(
        f"[FP] {args.label:22s} ticks={fp['ticks']:3d} cells={l1['active_cells']:3d} H={l1['occupancy_entropy_bits']:.2f} "
        f"v={l1['v_rms_u_s']:5.1f} str={l1['straightness_mean']} stk={l1['stuck_pct']}% "
        f"| tr/ps={l2['transitions_per_player_s']:.2f} coh={l2['flux_coherence_J_over_F']:.2f} sig={l2['entropy_production_sigma']:.3f} "
        f"| clump={l3['ally_clump_160u_pct']}% dmg={l4['damage_events']} kill={l4['kill_events']} r_eng={l4['engage_r_mean']} stk_adv={l4['stack_adv_mean']} "
        f"| pkp/m={l5['pickups_per_player_min']:.1f} stack={l5['mean_stack_hp_ap']:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
