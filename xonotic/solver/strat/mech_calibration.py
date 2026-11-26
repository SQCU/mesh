#!/usr/bin/env python3
"""1v1 Deathmatch Mechanical Skill Calibration & Skewed-Roster `bots.txt` Generator.

Calibrates Xonotic's 12 per-bot HavocBot kinematic/mechanical skill axes (aim, offset, mouse,
think, AI think, move, dodge, keyboard, ping, weapon, aggression, range) via live 1v1 DM
tournaments across the 4 Apple M5 Ultra Mac Studios, mapping each profile to an empirical
1v1 mechanical Elo (anchored at `mech_stock = 1000.0`).

Provides:
  - `write_bots_txt`: emits exact single-TAB Xonotic `bots.txt` with `bot_forced_team` (1..3)
  - `sample_skewed_roster`: packs 3-team payload matches with deliberately unfair/asymmetric
    rosters (carry+potatoes, underdog vs overdog, aim-turret vs movement-demon specialists)
  - `extract_privileged_skill_target`: computes true latent per-bot mechanical skill target
    from raw OBS `BOT_CONFIGURATION_COLUMNS` for training-only asymmetric critic & online
    skill-ID supervision (while the actor itself masks all 13 bot-config columns to zero).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import BOT_CONFIGURATION_COLUMNS, OBS
from solver.server_browser import probe_darkplaces_server
from solver.strat.outer_validation import compute_multiteam_elo


# 12 HavocBot mechanical skill columns in exact bots.txt TAB order (argv 6..17):
# keyboard, moving, dodging, ping, weapon, aggressivity, range, aiming, calmhand(offset), mouse, fightthink, aithink
SKILL_FIELD_NAMES = (
    "keyboard", "moving", "dodging", "ping", "weapon",
    "aggressivity", "range", "aiming", "offset", "mouse",
    "fightthink", "aithink",
)

# 9 mechanical archetypes spanning ~600 Elo mechanical potato -> ~1450+ Elo aimbot,
# plus orthogonal kinematic specialists (aim_turret vs movement_demon).
MECHANICAL_PROFILES: dict[str, tuple[float, ...]] = {
    "mech_potato":  (-3.0, -3.0, -3.0, -3.0, -3.0, -1.0,  0.0, -3.0, -3.0, -3.0, -3.0, -3.0),
    "mech_novice":  (-2.0, -2.0, -2.0, -2.0, -2.0, -0.5,  0.0, -2.0, -2.0, -2.0, -2.0, -2.0),
    "mech_casual":  (-1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, -1.0, -1.0, -1.0, -1.0),
    "mech_stock":   ( 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "mech_veteran": ( 1.0,  1.2,  1.2,  1.0,  1.2,  0.5,  0.0,  1.2,  1.2,  1.2,  1.2,  1.2),
    "mech_carry":   ( 2.0,  2.5,  2.5,  2.0,  2.5,  0.8,  0.2,  2.5,  2.5,  2.5,  2.5,  2.5),
    "mech_aimbot":  ( 3.0,  3.5,  3.5,  3.0,  3.5,  1.0,  0.3,  4.0,  3.5,  3.5,  3.5,  3.5),
    "mech_turret":  (-2.0, -2.5, -2.5,  1.0,  2.5,  0.5,  1.0,  3.5,  3.0,  3.0,  2.5,  1.5),
    "mech_mover":   ( 2.5,  3.5,  3.5,  1.0, -1.0,  0.8, -0.8, -2.5, -2.5, -2.0,  1.0,  1.0),
}

# Prior mechanical ratings before live 1v1 DM calibration (updated directly from DM Elo JSON)
DEFAULT_CALIBRATED_ELO: dict[str, float] = {
    "mech_potato":  620.0,
    "mech_novice":  760.0,
    "mech_casual":  890.0,
    "mech_stock":  1000.0,
    "mech_veteran": 1140.0,
    "mech_carry":   1290.0,
    "mech_aimbot":  1430.0,
    "mech_turret":  1120.0,
    "mech_mover":   920.0,
}

MODELS = ("ignis", "erebus", "umbra", "nyx", "pyria", "seraphina", "gak", "ignismasked", "gakmasked")


def write_bots_txt(path: Path, entries: list[tuple[str, int, str]]) -> Path:
    """Write a single-TAB Xonotic bots.txt file.
    `entries` is a list of `(bot_name, forced_team_1_to_3_or_0, profile_name)`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["//auto-generated calibrated mechanical roster (single-TAB separated)"]
    for idx, (bname, forced_team, prof_name) in enumerate(entries):
        vec = MECHANICAL_PROFILES[prof_name]
        model = MODELS[idx % len(MODELS)]
        shirt = (idx * 3 + 1) % 15
        pants = (idx * 5 + 2) % 15
        cols = [bname, model, "0", str(shirt), str(pants), str(int(forced_team))] + [f"{v:.2f}" for v in vec]
        lines.append("\t".join(cols))
    path.write_text("\n".join(lines) + "\n")
    return path


def sample_skewed_roster(rng: np.random.Generator, archetype: str | None = None) -> tuple[str, list[tuple[str, int, str]]]:
    """Generate a 9-bot (3 teams × 3 bots) mechanical roster spanning both balanced and
    deliberately unfair/asymmetric team strengths to force roster-adaptive strategy."""
    presets = (
        "carry_vs_balanced",
        "underdog_vs_overdog",
        "specialist_clash",
        "double_carry_vs_trio",
        "extreme_skew",
        "balanced_mid",
    )
    kind = archetype or str(rng.choice(presets, p=[0.24, 0.24, 0.18, 0.14, 0.12, 0.08]))
    if kind == "carry_vs_balanced":
        team_profs = [
            ["mech_aimbot", "mech_potato", "mech_potato"],
            ["mech_stock", "mech_stock", "mech_stock"],
            ["mech_veteran", "mech_casual", "mech_novice"],
        ]
    elif kind == "underdog_vs_overdog":
        team_profs = [
            ["mech_novice", "mech_novice", "mech_potato"],
            ["mech_carry", "mech_carry", "mech_veteran"],
            ["mech_stock", "mech_casual", "mech_veteran"],
        ]
    elif kind == "specialist_clash":
        team_profs = [
            ["mech_turret", "mech_mover", "mech_stock"],
            ["mech_carry", "mech_potato", "mech_mover"],
            ["mech_aimbot", "mech_novice", "mech_turret"],
        ]
    elif kind == "double_carry_vs_trio":
        team_profs = [
            ["mech_carry", "mech_carry", "mech_potato"],
            ["mech_veteran", "mech_veteran", "mech_veteran"],
            ["mech_aimbot", "mech_casual", "mech_potato"],
        ]
    elif kind == "extreme_skew":
        team_profs = [
            ["mech_potato", "mech_potato", "mech_potato"],
            ["mech_aimbot", "mech_aimbot", "mech_aimbot"],
            ["mech_carry", "mech_stock", "mech_novice"],
        ]
    else:
        team_profs = [
            ["mech_stock", "mech_stock", "mech_stock"],
            ["mech_stock", "mech_stock", "mech_stock"],
            ["mech_stock", "mech_stock", "mech_stock"],
        ]

    # Randomly permute which team index (1, 2, 3) gets which mechanical roster so seat & skill are orthogonal
    perm = rng.permutation(3)
    entries: list[tuple[str, int, str]] = []
    for team_idx in range(3):
        profs = team_profs[int(perm[team_idx])]
        for slot, prof in enumerate(profs):
            bname = f"T{team_idx+1}S{slot+1}_{prof.replace('mech_', '')}"
            entries.append((bname, team_idx + 1, prof))
    return kind, entries


def extract_privileged_skill_target(obs_rows: np.ndarray) -> np.ndarray:
    """Compute true latent standardized mechanical skill scalar z_i in [-2, +2] for each bot row
    from raw OBS `BOT_CONFIGURATION_COLUMNS` (training-only privileged label; hidden from actor!).

    Weighting matches HavocBot's `READSKILL(f, w, r)` multipliers and emphasizes aim/offset/mouse/
    think/move/dodge/weapon execution capacity. For human players (all bot_*skill == 0), returns 0.0."""
    if len(obs_rows) == 0:
        return np.zeros((0,), dtype=np.float32)
    # READSKILL weights in bot.qc: aim*2, offset*2, mouse*1, move*2, dodge*2, weapon*2, think*1, ai*2, kb*0.5
    aim = obs_rows[:, OBS["BOT_AIMSKILL"]] / 2.0
    off = obs_rows[:, OBS["BOT_OFFSETSKILL"]] / 2.0
    mouse = obs_rows[:, OBS["BOT_MOUSESKILL"]] / 1.0
    move = obs_rows[:, OBS["BOT_MOVESKILL"]] / 2.0
    dodge = obs_rows[:, OBS["BOT_DODGESKILL"]] / 2.0
    weap = obs_rows[:, OBS["BOT_WEAPONSKILL"]] / 2.0
    think = obs_rows[:, OBS["BOT_THINKSKILL"]] / 1.0
    ai = obs_rows[:, OBS["BOT_AISKILL"]] / 2.0
    # Weighted composite in roughly [-3.5, +3.5] -> scale by 1/2.0 into [-1.8, +1.8]
    raw = 0.26 * aim + 0.16 * off + 0.12 * mouse + 0.14 * move + 0.12 * dodge + 0.10 * weap + 0.05 * think + 0.05 * ai
    return np.clip(raw / 2.0, -2.5, 2.5).astype(np.float32)


def run_1v1_dm_batch(
    *,
    dp_bin: str,
    basedir: str,
    userdir: str,
    work_dir: Path,
    matchups: list[tuple[str, str]],
    base_port: int = 28500,
    duration_s: float = 22.0,
    slowmo: float = 3.0,
    skill: int = 7,
    mapname: str = "boil",
    max_parallel: int = 6,
) -> list[dict[str, float]]:
    """Run live 1v1 Deathmatch duels between calibrated mechanical bot profiles and read exact
    final frag counts via DarkPlaces UDP `getstatus`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float]] = []

    for chunk_start in range(0, len(matchups), max_parallel):
        chunk = matchups[chunk_start : chunk_start + max_parallel]
        procs = []
        logs = []
        meta = []
        for idx, (prof_a, prof_b) in enumerate(chunk):
            port = base_port + idx
            sv_dir = work_dir / f"dm_{chunk_start + idx}_{prof_a}_vs_{prof_b}"
            sv_data = sv_dir / "data"
            sv_data.mkdir(parents=True, exist_ok=True)
            for item in Path(userdir, "data").iterdir():
                dst = sv_data / item.name
                if not dst.exists():
                    os.symlink(item, dst)
            name_a = f"A_{prof_a}"
            name_b = f"B_{prof_b}"
            bots_file = sv_data / "dm_bots.txt"
            write_bots_txt(bots_file, [(name_a, 0, prof_a), (name_b, 0, prof_b)])
            logf = open(sv_dir / "server.log", "w")
            logs.append(logf)
            cmd = [
                dp_bin, "-xonotic",
                "-basedir", basedir,
                "-userdir", str(sv_dir),
                "+sv_public", "0",
                "+port", str(port),
                "+maxplayers", "4",
                "+gametype", "dm",
                "+g_payload", "0",
                "+g_warmup", "0",
                "+g_start_delay", "0",
                "+fraglimit", "0",
                "+timelimit", "0",
                "+bot_join_empty", "1",
                "+bot_number", "2",
                "+bot_prefix", "",
                "+bot_suffix", "",
                "+bot_config_file", "dm_bots.txt",
                "+skill", str(skill),
                "+slowmo", str(slowmo),
                "+sv_random_seed", str(9000 + chunk_start + idx),
                "+map", mapname,
            ]
            procs.append(subprocess.Popen(cmd, cwd=basedir, stdout=logf, stderr=subprocess.STDOUT))
            meta.append((port, prof_a, prof_b, name_a, name_b))
            time.sleep(0.1)

        time.sleep(duration_s)

        for port, prof_a, prof_b, name_a, name_b in meta:
            st = probe_darkplaces_server("127.0.0.1", port, timeout=0.6)
            sc_a = sc_b = None
            if st.get("online"):
                for pl in st.get("players", []):
                    nm = str(pl.get("name", ""))
                    if name_a in nm:
                        sc_a = float(pl.get("score", 0))
                    elif name_b in nm:
                        sc_b = float(pl.get("score", 0))
            if sc_a is not None and sc_b is not None:
                results.append({prof_a: sc_a, prof_b: sc_b})

        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=3.0)
                except Exception:
                    try:
                        p.kill()
                        p.wait(timeout=2.0)
                    except Exception:
                        pass
        for f in logs:
            f.close()
    return results


def main():
    ap = argparse.ArgumentParser(description="4-Studio Parallel 1v1 DM Mechanical Skill Calibration Ladder")
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--out-dir", default="~/xonotic-runs/mech-calibration")
    ap.add_argument("--duration-s", type=float, default=20.0)
    ap.add_argument("--slowmo", type=float, default=3.0)
    ap.add_argument("--skill", type=int, default=7)
    ap.add_argument("--map", default="boil")
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--base-port", type=int, default=28500)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    dist = None
    if world_size > 1:
        import torch
        import torch.distributed as _dist
        import torch_mesh  # noqa: F401
        _dist.init_process_group(backend="mesh", device_id=torch.device("mps", 0))
        dist = _dist
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    profs = list(MECHANICAL_PROFILES.keys())
    # Full round-robin (9*8/2 = 36 unique pairs) + anchor pairs vs mech_stock = 48 1v1 duels
    all_pairs = [(profs[i], profs[j]) for i in range(len(profs)) for j in range(i + 1, len(profs))]
    for p in profs:
        if p != "mech_stock":
            all_pairs.append((p, "mech_stock"))
    # Shard evenly across the 4 Studios!
    my_pairs = [pair for idx, pair in enumerate(all_pairs) if idx % world_size == rank]

    stage = Path(args.stage_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    t0 = time.monotonic()
    local_matches = run_1v1_dm_batch(
        dp_bin=str(stage / "darkplaces-dedicated"),
        basedir=str(stage / "Xonotic"),
        userdir=str(stage / "userdir"),
        work_dir=out_dir / f"work_r{rank}",
        matchups=my_pairs,
        base_port=args.base_port,
        duration_s=args.duration_s,
        slowmo=args.slowmo,
        skill=args.skill,
        mapname=args.map,
        max_parallel=args.max_parallel,
    )
    elapsed = round(time.monotonic() - t0, 2)

    if dist is not None:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, {"rank": rank, "host": os.uname().nodename, "matches": local_matches, "wall_s": elapsed})
    else:
        gathered = [{"rank": 0, "host": os.uname().nodename, "matches": local_matches, "wall_s": elapsed}]

    if rank == 0:
        all_matches = [m for g in gathered for m in g["matches"]]
        elo = compute_multiteam_elo(all_matches, anchor_arm="mech_stock", anchor_elo=1000.0)
        out = {
            "mode": "mech_dm_calibration",
            "world_size": world_size,
            "map": args.map,
            "base_skill": args.skill,
            "slowmo": args.slowmo,
            "duration_s": args.duration_s,
            "wall_s": max(g["wall_s"] for g in gathered),
            "matches_completed": len(all_matches),
            "matches_scheduled": len(all_pairs),
            "elo_anchor_stock_1000": elo,
            "calibrated_elo_map": {k: v["elo"] for k, v in (elo.get("ratings") or {}).items()},
            "profiles": {k: list(v) for k, v in MECHANICAL_PROFILES.items()},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "mech_calibration.json"
        out_path.write_text(json.dumps(out, indent=2))
        print("CALIBRATION_WRITTEN:", str(out_path), flush=True)
        print(json.dumps({
            "wall_s": out["wall_s"],
            "matches_completed": out["matches_completed"],
            "ratings": {k: (v["elo"], v["ci95"], round(v["mean_score"], 1)) for k, v in sorted((elo.get("ratings") or {}).items(), key=lambda kv: -kv[1]["elo"])},
        }, indent=2), flush=True)

    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
