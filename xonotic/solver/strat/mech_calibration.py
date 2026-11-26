#!/usr/bin/env python3
"""Native HavocBot Parameter -> 1v1 DM Elo Calibration & Inverse Roster Sampler.

Uses Xonotic's real native HavocBot personalities (the 18 built-in bots from `bots.txt` inside
`xonotic-20230620-data.pk3`) paired with our batch-isolated per-bot native `bot_base_skill`
field (19th TAB column in `bots.txt`, protected by `PRVM_ViewSkill` and packed per-row in
`bot_batch.c`) to:
  1. Run offline 4-Studio parallel 1v1 Deathmatch reference-game tournaments directly across
     Xonotic's native HavocBot parametric space `(base_skill in [0..15], 12D personality)`.
  2. Fit an empirical parametric map `E_hat(base_skill, theta_12d) -> 1v1 DM Elo` (with
     held-out R^2 and RMSE in Elo points).
  3. Invert that map (`sample_roster_from_target_elos`) to draw native HavocBot rosters
     matching any target Elo distribution (e.g. human-lobby Gaussian, bimodal, or skewed)
     without any hand-invented handicaps or out-of-distribution skill caricatures.
  4. Compute each bot's empirical calibrated 1v1 DM Elo directly from raw OBS
     `BOT_CONFIGURATION_COLUMNS` (training-only privileged target for asymmetric critic &
     online Skill-ID head, completely masked out of the actor).
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
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import OBS
from solver.server_browser import probe_darkplaces_server
from solver.strat.outer_validation import compute_multiteam_elo


SKILL_FIELD_NAMES = (
    "keyboard", "moving", "dodging", "ping", "weapon",
    "aggressivity", "range", "aiming", "offset", "mouse",
    "fightthink", "aithink",
)

# Exact 18 native HavocBot personalities from Xonotic's built-in bots.txt
# (name, model, skin, shirt, pants, 12D native skill vector)
NATIVE_BOTS: list[dict] = [
    {"name": "Hellfire",     "model": "ignis",           "skin": 0, "shirt":  4, "pants":  0, "vec": ( 0.0,  0.0, -0.5, -1.0,  1.0,  1.0, -0.5, -1.0, -1.0,  2.0,  0.5, -1.0)},
    {"name": "Toxic",        "model": "gakmasked",       "skin": 0, "shirt": 14, "pants":  7, "vec": (-1.0, -1.5, -0.5,  0.0,  1.0,  0.0,  0.0,  0.0,  2.0, -0.5, -0.5,  1.0)},
    {"name": "Discovery",    "model": "erebus",          "skin": 0, "shirt":  2, "pants":  6, "vec": ( 0.0, -1.0, -0.5, -0.5,  1.0, -0.5,  0.5,  1.5, -0.5, -1.0,  1.0,  0.5)},
    {"name": "Pegasus",      "model": "umbra",           "skin": 0, "shirt": 13, "pants": 11, "vec": ( 1.0,  1.0,  1.0,  1.0, -1.0,  0.0,  0.5,  0.0, -2.0,  0.0, -1.0,  0.0)},
    {"name": "Eureka",       "model": "umbra",           "skin": 0, "shirt": 12, "pants":  7, "vec": ( 0.0,  0.0, -1.5, -0.5, -0.5,  0.0,  0.0,  0.0,  0.0, -0.5,  1.5,  1.5)},
    {"name": "Airhead",      "model": "ignis",           "skin": 0, "shirt": 11, "pants":  1, "vec": (-1.0, -1.5, -1.0, -0.5,  1.0,  1.0, -1.0,  1.0, -0.5,  1.0,  0.5,  0.0)},
    {"name": "Gator",        "model": "gak",             "skin": 0, "shirt":  3, "pants": 10, "vec": ( 0.0,  1.0,  0.0,  0.5, -0.5,  0.5, -0.5, -1.0,  0.0,  0.0, -0.5,  0.0)},
    {"name": "Delirium",     "model": "gakmasked",       "skin": 0, "shirt":  8, "pants": 12, "vec": ( 0.0, -1.0, -1.0, -1.0,  0.0,  2.0,  0.0,  1.0,  0.0,  2.0, -1.0, -1.0)},
    {"name": "Death",        "model": "gakmasked",       "skin": 0, "shirt":  4, "pants": 11, "vec": (-0.5,  0.0,  0.0,  1.0, -0.5,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0)},
    {"name": "Scorcher",     "model": "ignismasked",     "skin": 0, "shirt": 13, "pants": 13, "vec": ( 0.0, -1.0,  0.0, -0.5,  0.5,  1.0,  0.0,  1.0, -2.0,  1.0,  0.0,  0.0)},
    {"name": "Necrotic",     "model": "nyx",             "skin": 0, "shirt": 12, "pants": 14, "vec": ( 0.0,  0.0,  0.0,  1.0,  0.0, -1.0, -0.5, -1.0,  1.0,  0.0,  0.0,  0.0)},
    {"name": "Dominator",    "model": "nyx",             "skin": 0, "shirt":  3, "pants":  9, "vec": ( 0.0,  0.0,  0.0,  2.0, -1.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0)},
    {"name": "Thunderstorm", "model": "erebus",          "skin": 0, "shirt": 13, "pants":  6, "vec": ( 0.0,  0.0,  0.0, -0.5, -1.0,  1.0,  0.5,  0.5, -0.5,  0.0,  0.5,  0.0)},
    {"name": "Mystery",      "model": "pyria",           "skin": 0, "shirt":  9, "pants": 14, "vec": ( 1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  0.0,  1.0,  1.0, -2.0, -1.0, -1.0)},
    {"name": "Lion",         "model": "ignismasked",     "skin": 0, "shirt":  0, "pants":  4, "vec": ( 1.0,  1.5,  2.0, -1.0, -1.0, -1.0,  1.0,  0.0,  1.0, -0.5, -1.0, -1.0)},
    {"name": "Sensible",     "model": "seraphina",       "skin": 0, "shirt":  9, "pants":  9, "vec": ( 0.0,  0.0,  0.5, -1.0,  0.0, -1.0,  0.0, -1.0,  2.5, -1.5,  1.0,  0.5)},
    {"name": "Shadow",       "model": "seraphinamasked", "skin": 0, "shirt":  4, "pants":  8, "vec": (-0.5,  2.0,  1.0,  0.0,  0.0, -1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -0.5)},
    {"name": "Resurrection", "model": "umbra",           "skin": 0, "shirt":  1, "pants":  1, "vec": ( 0.0, -0.5, -0.5,  0.0,  2.0, -1.0, -1.0, -1.0,  0.0, -1.0,  1.0,  1.0)},
]

# Empirically calibrated from 96 live 4-Studio 1v1 DM duels across native HavocBot space (ridge=40.0,
# R^2=0.664, LOO R^2=+0.440, LOO RMSE=51.7 Elo, anchor s05_Neutral=1000.0).
CALIBRATED_NATIVE_REGRESSION: dict = {
    "feature_names": ["intercept", "base_skill", "log1p_base_skill", *SKILL_FIELD_NAMES],
    "weights": [
        929.2899, 9.6428, 1.7357, 0.6691, 3.2524, 0.5211, 1.8221,
        2.2705, -5.7903, -3.0233, -6.2523, 0.1331, -2.4793, 5.1195, 0.7342,
    ],
    "r2": 0.664,
    "rmse_elo": 40.06,
    "loo_r2": 0.4401,
    "loo_rmse_elo": 51.71,
    "elo_min": 822.8,
    "elo_max": 1134.4,
    "elo_span": 311.6,
    "elo_mean": 989.2,
    "elo_std": 69.1,
}


def sample_native_lobby_roster(
    rng: np.random.Generator,
    regression: dict | None = None,
    native_bots: list[dict] | None = None,
    distribution: str | None = None,
    n_teams: int = 3,
    n_bots: int = 9,
    league_cfg=None,
) -> tuple[str, list[dict]]:
    reg = regression or CALIBRATED_NATIVE_REGRESSION
    lo = float(reg.get("elo_min", 830.0))
    hi = float(reg.get("elo_max", 1130.0))
    mu = float(reg.get("elo_mean", 990.0))
    K = max(2, int(n_teams))
    N = max(K, int(n_bots))
    dist_probs = getattr(league_cfg, "roster_dist_probs", None) or {
        "gaussian_lobby": 0.35, "team_skewed": 0.30, "within_team_skewed": 0.25, "uniform_native": 0.10,
    }
    dist_names = tuple(dist_probs.keys())
    p_arr = np.asarray([float(dist_probs[k]) for k in dist_names], dtype=np.float64)
    p_arr /= p_arr.sum()
    g_std = float(getattr(league_cfg, "roster_gaussian_std", 75.0))
    ts_std = float(getattr(league_cfg, "roster_team_skew_std", 28.0))
    wt_stds = getattr(league_cfg, "roster_within_stds", (25.0, 35.0, 25.0))
    kind = distribution or str(rng.choice(dist_names, p=p_arr))
    teams = [(i % K) + 1 for i in range(N)]
    if kind == "gaussian_lobby":
        targets = np.clip(rng.normal(mu, g_std, size=N), lo, hi).tolist()
    elif kind == "team_skewed":
        tier_means = np.linspace(lo + 0.15 * (hi - lo), hi - 0.10 * (hi - lo), K)
        tier_means = tier_means[rng.permutation(K)]
        targets = [float(np.clip(rng.normal(tier_means[tm - 1], ts_std), lo, hi)) for tm in teams]
    elif kind == "within_team_skewed":
        per_tm_count: dict[int, int] = {}
        targets = []
        for tm in teams:
            slot = per_tm_count.get(tm, 0)
            per_tm_count[tm] = slot + 1
            if slot % 3 == 0:
                t_mu, t_sd = hi - 0.08 * (hi - lo), float(wt_stds[0])
            elif slot % 3 == 1:
                t_mu, t_sd = mu, float(wt_stds[1])
            else:
                t_mu, t_sd = lo + 0.18 * (hi - lo), float(wt_stds[2])
            targets.append(float(np.clip(rng.normal(t_mu, t_sd), lo, hi)))
    else:
        targets = rng.uniform(lo, hi, size=N).tolist()

    entries = sample_roster_from_target_elos(targets, reg, rng, native_bots=native_bots, teams=teams)
    return kind, entries


def load_native_bots(pk3_path: Path | None = None) -> list[dict]:
    """Load the exact native bots.txt roster from Xonotic's data pk3 if available, else NATIVE_BOTS."""
    if pk3_path is not None and pk3_path.exists():
        try:
            with zipfile.ZipFile(pk3_path) as z:
                txt = z.read("bots.txt").decode("utf-8", errors="replace")
            out = []
            for raw in txt.splitlines():
                s = raw.strip()
                if not s or s.startswith(("//", "#")):
                    continue
                parts = raw.split("\t")
                if len(parts) >= 18:
                    out.append({
                        "name": parts[0],
                        "model": parts[1],
                        "skin": int(float(parts[2])),
                        "shirt": int(float(parts[3])),
                        "pants": int(float(parts[4])),
                        "vec": tuple(float(x) for x in parts[6:18]),
                    })
            if out:
                return out
        except Exception:
            pass
    return list(NATIVE_BOTS)


def write_native_bots_txt(path: Path, entries: list[dict]) -> Path:
    """Write a single-TAB Xonotic bots.txt using native 12D personality vectors + optional 19th
    column `base_skill` (isolated per-bot via our `BOT_SKILL(this)` QuakeC + `bot_batch.c` patch).

    Each entry dict has: `name`, `team` (0 for DM, 1..3 for payload), `base_skill` (float 0..17),
    and `bot` (dict from `NATIVE_BOTS`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["//native HavocBot roster with per-bot isolated bot_base_skill (col 18)"]
    for e in entries:
        b = e["bot"]
        cols = [
            str(e["name"]),
            str(b["model"]),
            str(int(b.get("skin", 0))),
            str(int(b.get("shirt", -1))),
            str(int(b.get("pants", -1))),
            str(int(e.get("team", 0))),
            *[f"{float(v):.2f}" for v in b["vec"]],
            f"{float(e['base_skill']):.2f}",
        ]
        lines.append("\t".join(cols))
    path.write_text("\n".join(lines) + "\n")
    return path


def build_calibration_competitors(native_bots: list[dict]) -> list[dict]:
    """Construct a rich grid of native HavocBot competitors spanning Xonotic's real native
    `base_skill` ladder (`0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15`) crossed with diverse native
    personalities from `bots.txt` (22 competitors total, anchored at `skill=5` neutral)."""
    skill_levels = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)
    competitors = []
    # 2 distinct native personalities per skill level so both base_skill and 12D personality
    # variance are directly identified in the regression!
    for si, sk in enumerate(skill_levels):
        for rep in range(2):
            bot = native_bots[(si * 2 + rep) % len(native_bots)]
            cid = f"s{int(sk):02d}_{bot['name']}"
            competitors.append({"id": cid, "base_skill": float(sk), "bot": bot})
    # Plus neutral reference anchor at skill=5.0 with zero 12D offset
    competitors.append({
        "id": "s05_Neutral",
        "base_skill": 5.0,
        "bot": {"name": "Neutral", "model": "erebus", "skin": 0, "shirt": 2, "pants": 6, "vec": (0.0,) * 12},
    })
    return competitors


def _design_features(base_skill: float, vec12: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Feature vector [1, base_skill, log1p(base_skill), 12D native HavocBot offsets] for
    parametric Elo regression."""
    s = float(max(0.0, base_skill))
    return np.concatenate(([1.0, s, math.log1p(s)], np.asarray(vec12, dtype=np.float64)))


def fit_native_elo_regression(competitors: list[dict], ratings: dict[str, dict], ridge: float = 40.0) -> dict:
    """Fit ridge regression from native HavocBot parameters `(base_skill, log1p(base_skill), theta_12d)`
    to empirical 1v1 DM Elo, reporting in-sample & leave-one-out R^2 and RMSE (Elo points)."""
    X_rows, y_vals, ids = [], [], []
    for c in competitors:
        if c["id"] not in ratings:
            continue
        X_rows.append(_design_features(c["base_skill"], c["bot"]["vec"]))
        y_vals.append(float(ratings[c["id"]]["elo"]))
        ids.append(c["id"])
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    p = X.shape[1]
    reg = np.eye(p, dtype=np.float64) * ridge
    reg[0, 0] = 0.0  # do not regularize intercept
    w = np.linalg.solve(X.T @ X + reg, X.T @ y)
    pred = X @ w
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    r2 = float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))

    # Leave-one-out cross-validation R^2 / RMSE
    loo_preds = []
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        wi = np.linalg.solve(X[mask].T @ X[mask] + reg, X[mask].T @ y[mask])
        loo_preds.append(float(X[i] @ wi))
    loo_preds = np.asarray(loo_preds)
    loo_rmse = float(np.sqrt(np.mean((y - loo_preds) ** 2)))
    loo_r2 = float(1.0 - np.sum((y - loo_preds) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))

    # Also fit simple 2-feature curve (base_skill, log1p(base_skill)) to isolate base_skill vs 12D personality!
    Xs = X[:, :3]
    regs = np.diag([0.0, 0.1, 0.1])
    ws = np.linalg.solve(Xs.T @ Xs + regs, Xs.T @ y)
    r2_base_only = float(1.0 - np.sum((y - Xs @ ws) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))

    feat_names = ["intercept", "base_skill", "log1p_base_skill", *SKILL_FIELD_NAMES]
    return {
        "feature_names": feat_names,
        "weights": [round(float(v), 4) for v in w],
        "weights_by_name": {k: round(float(v), 4) for k, v in zip(feat_names, w)},
        "r2": round(r2, 4),
        "rmse_elo": round(rmse, 2),
        "loo_r2": round(loo_r2, 4),
        "loo_rmse_elo": round(loo_rmse, 2),
        "r2_base_skill_only": round(r2_base_only, 4),
        "elo_min": round(float(y.min()), 1),
        "elo_max": round(float(y.max()), 1),
        "elo_span": round(float(y.max() - y.min()), 1),
        "elo_mean": round(float(y.mean()), 1),
        "elo_std": round(float(y.std()), 1),
    }


def predict_native_elo(base_skill: float, vec12: tuple[float, ...] | np.ndarray, regression: dict) -> float:
    w = np.asarray(regression["weights"], dtype=np.float64)
    return float(_design_features(base_skill, vec12) @ w)


def sample_roster_from_target_elos(
    target_elos: list[float],
    regression: dict,
    rng: np.random.Generator,
    native_bots: list[dict] | None = None,
    teams: list[int] | None = None,
) -> list[dict]:
    """Invert the fitted native HavocBot Elo model: for each desired target Elo e_i, pick a
    native `bots.txt` personality + native `base_skill in [0.0, 15.0]` whose calibrated 1v1 DM
    Elo matches e_i as closely as possible."""
    bots = native_bots or NATIVE_BOTS
    grid = np.linspace(0.0, 15.0, 61)
    entries = []
    for idx, te in enumerate(target_elos):
        # Sample 4 candidate native personalities and solve for best base_skill in [0, 15]
        best_err, best_sk, best_b = float("inf"), 5.0, bots[0]
        for cand_i in rng.choice(len(bots), size=min(6, len(bots)), replace=False):
            b = bots[int(cand_i)]
            preds = np.asarray([predict_native_elo(float(s), b["vec"], regression) for s in grid])
            j = int(np.argmin(np.abs(preds - float(te))))
            err = float(abs(preds[j] - float(te)))
            if err < best_err:
                best_err, best_sk, best_b = err, float(grid[j]), b
        tm = int(teams[idx]) if (teams is not None and idx < len(teams)) else (idx // 3) + 1
        entries.append({
            "name": f"T{tm}P{idx+1}_{best_b['name']}_s{best_sk:.1f}",
            "team": tm,
            "base_skill": best_sk,
            "target_elo": float(te),
            "predicted_elo": round(predict_native_elo(best_sk, best_b["vec"], regression), 1),
            "bot": best_b,
        })
    return entries


def extract_privileged_skill_target(obs_rows: np.ndarray, regression: dict | None = None) -> np.ndarray:
    """Compute true latent standardized mechanical Elo target z_i directly from raw OBS
    `BOT_CONFIGURATION_COLUMNS` (including our per-bot `OBS['BOT_GLOBAL_SKILL'] = BOT_SKILL(it)`).
    Training-only privileged label; completely masked out of the actor!"""
    if len(obs_rows) == 0:
        return np.zeros((0,), dtype=np.float32)
    base_sk = obs_rows[:, OBS["BOT_GLOBAL_SKILL"]].astype(np.float64)
    # Invert READSKILL multipliers from bot.qc to recover raw bots.txt 12D vector
    vec12 = np.stack([
        obs_rows[:, OBS["HAVOCBOT_KEYBOARDSKILL"]] / 0.5,
        obs_rows[:, OBS["BOT_MOVESKILL"]] / 2.0,
        obs_rows[:, OBS["BOT_DODGESKILL"]] / 2.0,
        obs_rows[:, OBS["BOT_PINGSKILL"]] / 0.5,
        obs_rows[:, OBS["BOT_WEAPONSKILL"]] / 2.0,
        obs_rows[:, OBS["BOT_AGGRESSKILL"]] / 1.0,
        obs_rows[:, OBS["BOT_RANGEPREFERENCE"]] / 1.0,
        obs_rows[:, OBS["BOT_AIMSKILL"]] / 2.0,
        obs_rows[:, OBS["BOT_OFFSETSKILL"]] / 2.0,
        obs_rows[:, OBS["BOT_MOUSESKILL"]] / 1.0,
        obs_rows[:, OBS["BOT_THINKSKILL"]] / 1.0,
        obs_rows[:, OBS["BOT_AISKILL"]] / 2.0,
    ], axis=-1)
    if regression is not None and "weights" in regression:
        w = np.asarray(regression["weights"], dtype=np.float64)
        mu = float(regression.get("elo_mean", 1000.0))
        sd = max(float(regression.get("elo_std", 200.0)), 25.0)
        elos = np.asarray([_design_features(base_sk[i], vec12[i]) @ w for i in range(len(obs_rows))], dtype=np.float64)
        return np.clip((elos - mu) / sd, -3.0, 3.0).astype(np.float32)
    # Fallback before calibration file is loaded: standardize per-bot native base_skill around 5.0
    return np.clip((base_sk - 5.0) / 4.0, -2.5, 2.5).astype(np.float32)


def run_1v1_dm_batch(
    *,
    dp_bin: str,
    basedir: str,
    userdir: str,
    work_dir: Path,
    matchups: list[tuple[dict, dict]],
    base_port: int = 28500,
    duration_s: float = 22.0,
    slowmo: float = 3.0,
    server_skill: int = 7,
    mapname: str = "boil",
    max_parallel: int = 6,
) -> list[dict[str, float]]:
    """Run live 1v1 DM duels with per-bot isolated `bot_base_skill` inside the same server
    and read exact final frags via UDP `getstatus`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float]] = []

    for chunk_start in range(0, len(matchups), max_parallel):
        chunk = matchups[chunk_start : chunk_start + max_parallel]
        procs, logs, meta = [], [], []
        for idx, (ca, cb) in enumerate(chunk):
            port = base_port + idx
            sv_dir = work_dir / f"dm_{chunk_start + idx}_{ca['id']}_vs_{cb['id']}"
            sv_data = sv_dir / "data"
            sv_data.mkdir(parents=True, exist_ok=True)
            for item in Path(userdir, "data").iterdir():
                dst = sv_data / item.name
                if not dst.exists():
                    os.symlink(item, dst)
            name_a = f"A_{ca['id']}"
            name_b = f"B_{cb['id']}"
            write_native_bots_txt(
                sv_data / "dm_bots.txt",
                [
                    {"name": name_a, "team": 0, "base_skill": ca["base_skill"], "bot": ca["bot"]},
                    {"name": name_b, "team": 0, "base_skill": cb["base_skill"], "bot": cb["bot"]},
                ],
            )
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
                "+skill", str(server_skill),
                "+slowmo", str(slowmo),
                "+sv_random_seed", str(12000 + chunk_start + idx),
                "+map", mapname,
            ]
            procs.append(subprocess.Popen(cmd, cwd=basedir, stdout=logf, stderr=subprocess.STDOUT))
            meta.append((port, ca["id"], cb["id"], name_a, name_b))
            time.sleep(0.08)

        time.sleep(duration_s)

        for port, id_a, id_b, name_a, name_b in meta:
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
                results.append({id_a: sc_a, id_b: sc_b})

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
    ap = argparse.ArgumentParser(description="4-Studio Offline Native HavocBot Parameter -> 1v1 DM Elo Calibration")
    ap.add_argument("--stage-dir", default="~/xonotic-stage")
    ap.add_argument("--out-dir", default="~/xonotic-runs/native-dm-calibration")
    ap.add_argument("--duration-s", type=float, default=22.0)
    ap.add_argument("--slowmo", type=float, default=3.0)
    ap.add_argument("--map", default="boil")
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--base-port", type=int, default=28500)
    ap.add_argument("--duels", type=int, default=96)
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

    stage = Path(args.stage_dir).expanduser()
    native_bots = load_native_bots(stage / "Xonotic/data/xonotic-20230620-data.pk3")
    competitors = build_calibration_competitors(native_bots)
    anchor = next(c for c in competitors if c["id"] == "s05_Neutral")

    # Connected Swiss/banded + anchor schedule covering every native competitor cleanly (96 duels across 4 Studios!)
    rng = np.random.default_rng(20261125)
    all_pairs: list[tuple[dict, dict]] = []
    # 1. Every competitor plays the skill=5 neutral anchor (22 duels)
    for c in competitors:
        if c["id"] != anchor["id"]:
            all_pairs.append((c, anchor))
    # 2. Adjacent + skip-1 skill ladder comparisons (42 duels)
    ordered = sorted(competitors, key=lambda c: (c["base_skill"], c["id"]))
    for i in range(len(ordered) - 1):
        all_pairs.append((ordered[i], ordered[i + 1]))
    for i in range(len(ordered) - 2):
        all_pairs.append((ordered[i], ordered[i + 2]))
    # 3. Fill up to `args.duels` with random cross-tier native pairings (spanning full 0..15 ladder)
    while len(all_pairs) < args.duels:
        i, j = rng.choice(len(competitors), size=2, replace=False)
        all_pairs.append((competitors[int(i)], competitors[int(j)]))
    all_pairs = all_pairs[: args.duels]

    my_pairs = [pair for idx, pair in enumerate(all_pairs) if idx % world_size == rank]
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
        server_skill=7,
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
        elo = compute_multiteam_elo(all_matches, anchor_arm="s05_Neutral", anchor_elo=1000.0)
        ratings = elo.get("ratings") or {}
        reg = fit_native_elo_regression(competitors, ratings, ridge=40.0)
        # Also verify inverse sampler round-trip on a realistic human-like lobby [700..1300]
        demo_targets = [700.0, 800.0, 900.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1300.0]
        demo_roster = sample_roster_from_target_elos(demo_targets, reg, np.random.default_rng(7), native_bots)

        out = {
            "mode": "native_havocbot_dm_calibration",
            "world_size": world_size,
            "map": args.map,
            "slowmo": args.slowmo,
            "duration_s": args.duration_s,
            "wall_s": max(g["wall_s"] for g in gathered),
            "matches_completed": len(all_matches),
            "matches_scheduled": len(all_pairs),
            "anchor": "s05_Neutral = 1000.0",
            "elo_ratings": elo,
            "parametric_regression": reg,
            "demo_inverse_roster": [
                {"name": r["name"], "team": r["team"], "base_skill": r["base_skill"], "target_elo": r["target_elo"], "predicted_elo": r["predicted_elo"]}
                for r in demo_roster
            ],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "native_dm_calibration.json"
        out_path.write_text(json.dumps(out, indent=2))
        print("NATIVE_CALIBRATION_WRITTEN:", str(out_path), flush=True)
        print(json.dumps({
            "wall_s": out["wall_s"],
            "matches_completed": out["matches_completed"],
            "elo_span": reg["elo_span"],
            "elo_min_max": [reg["elo_min"], reg["elo_max"]],
            "r2": reg["r2"],
            "r2_base_skill_only": reg["r2_base_skill_only"],
            "loo_r2": reg["loo_r2"],
            "rmse_elo": reg["rmse_elo"],
            "loo_rmse_elo": reg["loo_rmse_elo"],
            "top_bottom_ratings": [
                (k, v["elo"], round(v["mean_score"], 1))
                for k, v in sorted(ratings.items(), key=lambda kv: -kv[1]["elo"])
            ],
        }, indent=2), flush=True)

    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
