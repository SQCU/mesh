"""j-space linear-probe measurement on recorded server rows, no simulation (9855a829^:runs/jspace_probe.py, X5, onto the
09-14 policy program).

    PYTHONPATH=xonotic:rdma bin/mesh-python -B -m solver.strat.jspace_probe --journal STEPS \\
        --checkpoint CKPT.npz [--checkpoint LATER.npz ...] --device cpu --out PROBE.json

Asks one question: does the TRAINED IR carry linearly decodable game semantics that a random-initialized policy of the
same architecture and a random projection of the same inputs do not?  Controls, side by side: two random-init
policies (seeds 0 and 7), a random projection of the raw inputs to the IR's width, the raw inputs themselves, and
shuffled labels (which MUST fail, or the probes are lying).

Provenance.  The input is the responder's training journal (strat_responder.py, TrainingJournal): every step's
OBSERVATION, CART, TEAM and STATE rows and perceived events as the engine published them, rebuilt into the policy's
frame by the responder's own constructor (inputs.native_inputs).  The IR is each participant's own row of
Strategy.ir.  The trained policies are the checkpoints the responder saved (each one a point of the old training-budget
sweep, named by its update count); nothing is retrained here.

What is and is not tautological.  The policy reads the OBSERVATION and CART rows, so their columns' targets are tagged
`in_x` (decoding them measures preservation); the game value's projected-winner hierarchy (game_value.hierarchy_rows)
is not an input (decoding it measures induced structure), and the verdict is read off that group."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rdma"))
sys.path.insert(0, str(ROOT / "xonotic"))

from payload.tools.strategy_io_schema import CS, OBS, OBS_AMMO_COLUMNS, OBS_WEAPON_COLUMNS
from solver.strat.cast_header import Wally, parameter_seed
from solver.strat.checkpoint_state import load_policy
from solver.strat.game_value import hierarchy_rows
from solver.strat.inputs import native_inputs
from solver.strat.journal import TrainingJournal
from solver.strat.runtime import build_runtime_frame
from solver.strat.scale_config import SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK, strategy_widths
from solver.strat.state_steering import PAGE_WIDTH, STATE_HEADER
from solver.strat.strategy import strategy
from solver.xonwire import CART_KIND, OBSERVATION_KIND, STATE_KIND, TEAM_KIND

ap = argparse.ArgumentParser()
ap.add_argument("--journal", required=True, help="a responder training journal (<online checkpoint>.runstate.npz.steps.*)")
ap.add_argument("--checkpoint", action="append", required=True, help="a trained matrix_fusion checkpoint; repeat for a sweep")
ap.add_argument("--arm", default="matrix_fusion")
ap.add_argument("--device", required=True)
ap.add_argument("--limit", type=int, default=0, help="only the first N steps")
ap.add_argument("--integration-interval", type=float, required=True, help="the journal's run's strategy_interval")
ap.add_argument("--relaxation-time", type=float, required=True, help="the journal's run's relaxation_time")
ap.add_argument("--scale-rank", type=int, default=SCALE_RANK)
ap.add_argument("--scale-hidden", type=int, default=SCALE_HIDDEN)
ap.add_argument("--scale-experts", type=int, default=SCALE_EXPERTS)
ap.add_argument("--scale-topk", type=int, default=SCALE_TOPK)
ap.add_argument("--out", required=True)
args = ap.parse_args()

recs = []
for _, step in TrainingJournal(args.journal).read():
    records = step["records"]
    obs, cart, team = (records[kind][1] for kind in (OBSERVATION_KIND, CART_KIND, TEAM_KIND))
    events = [rows for rows in (step["event_frames"].values() if isinstance(step["event_frames"], dict) else ())]
    context = np.concatenate(events) if events else np.zeros((0, 18), np.float32)
    pages, chorus = native_inputs(records[STATE_KIND][1], obs, cart, team, context,
                                  args.integration_interval, args.relaxation_time, None)
    if not len(chorus.state):
        continue
    rows = obs[chorus.participant_rows]
    schema = tuple(np.unique(pages.headers[:, STATE_HEADER['SCHEMA']]).astype(np.int64).tolist())
    recs.append(dict(seq=len(recs), chorus=chorus, rows=rows, carts=cart, tick=build_runtime_frame(rows, cart, team),
                     schema=schema))
    if args.limit and len(recs) >= args.limit:
        break
if not recs:
    raise SystemExit(f"{args.journal}: no step with participants")
print(f"[data] steps={len(recs)} participant_rows={sum(len(r['rows']) for r in recs)}", flush=True)
WIDTHS = strategy_widths(PAGE_WIDTH, args.scale_rank, args.scale_hidden, args.scale_experts, args.scale_topk)


def targets(rec):
    rows, tick, carts = rec["rows"], rec["tick"], rec["carts"]
    n = len(rows)
    hier, mask = hierarchy_rows(tick.context, tick.cartstate, tick.game_value)
    words = rows[:, [OBS[name] for name in OBS_WEAPON_COLUMNS]].astype(np.int64)
    bits = np.concatenate([(words[:, [w]] >> np.arange(24)) & 1 for w in range(words.shape[1])], axis=1)
    wbit = int(np.argmax(bits.var(0))) if n > 1 else 0
    depth = np.asarray(tick.cartstate.pos, dtype=float)
    return dict(
        team=rows[:, OBS["TEAM"]].astype(float),
        hier_score=hier[:, 0].astype(float),
        hier_held=hier[:, 1].astype(float),
        hier_rank=hier[:, 4].astype(float),
        is_pw=mask.astype(float),
        pw_team=np.full(n, float(tick.game_value.projected_role) if tick.game_value.projected_role is not None else -1.0),
        succ_denial=hier[:, 6].astype(float),
        cart_depth_total=np.full(n, float(depth.sum())),
        cart0_depth=np.full(n, float(depth[0]) if len(depth) else 0.0),
        n_controlled=np.full(n, float((np.asarray(tick.cartstate.control) >= 0).sum())),
        health=rows[:, OBS["HEALTH"]].astype(float),
        armor=rows[:, OBS["ARMOR"]].astype(float),
        ammo=rows[:, [OBS[name] for name in OBS_AMMO_COLUMNS]].sum(1).astype(float),
        speed=np.linalg.norm(rows[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1], axis=1).astype(float),
        tss=(rows[:, OBS["ENGINE_TIME"]] - rows[:, OBS["SPAWN_TIME"]]).astype(float),
        alive=rows[:, OBS["ALIVE"]].astype(float),
        n_weapons=bits.sum(1).astype(float),
        weapon_bit=bits[:, wbit].astype(float),
        cell=rows[:, OBS["CELL_X"]].astype(float),
        tick=np.full(n, float(rec["seq"])),
    )


META_ROWS = [targets(r) for r in recs]
META = {key: np.concatenate([m[key] for m in META_ROWS]) for key in META_ROWS[0]}
RAW = np.concatenate([np.c_[r["chorus"].state, r["rows"]] for r in recs], 0).astype(np.float64)
print(f"[data] raw input dims={RAW.shape[1]} rows={RAW.shape[0]}", flush=True)

IN_X = {"team", "health", "armor", "ammo", "speed", "tss", "alive", "n_weapons", "weapon_bit", "cell",
        "cart_depth_total", "cart0_depth", "n_controlled"}


def policy(seed, checkpoint=None):
    torch.manual_seed(parameter_seed(seed, "matrix_fusion"))
    model = Wally(WIDTHS)
    model.state_schema = str(recs[0]["schema"])
    if checkpoint:
        load_policy(model, checkpoint, args.arm)
    return model.to(args.device)


def ir_of(model):
    out = []
    with torch.no_grad():
        for rec in recs:
            frame = [torch.from_numpy(np.ascontiguousarray(value)).to(args.device) for value in rec["chorus"]]
            ir = strategy(model, *frame).ir.cpu().numpy().astype(np.float64)
            out.append(ir[:, :WIDTHS.d_ir])
    return np.concatenate(out, 0)


rng = np.random.default_rng(0)
ticks = np.unique(META["tick"])
perm = rng.permutation(ticks)
train_ticks = set(perm[:int(0.6 * len(perm))].tolist())
tr = np.array([t in train_ticks for t in META["tick"]])
te = ~tr


def r2(F, y, lam=1e-3):
    ok = np.isfinite(y)
    a, b = tr & ok, te & ok
    if a.sum() < 10 or b.sum() < 10 or np.std(y[b]) < 1e-9:
        return None
    A = np.c_[F[a], np.ones(a.sum())]
    B = np.c_[F[b], np.ones(b.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A[:, :-1] = (A[:, :-1] - mu) / sd
    B[:, :-1] = (B[:, :-1] - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[a])
    return float(round(1.0 - np.sum((y[b] - B @ w) ** 2) / np.sum((y[b] - y[b].mean()) ** 2), 4))


def acc(F, y, lam=1e-3):
    ok = np.isfinite(y)
    a, b = tr & ok, te & ok
    if a.sum() < 10 or b.sum() < 10:
        return None
    yi = y.astype(int)
    cls = np.unique(yi[a])
    if len(cls) < 2:
        return None
    A = np.c_[F[a], np.ones(a.sum())]
    B = np.c_[F[b], np.ones(b.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A[:, :-1] = (A[:, :-1] - mu) / sd
    B[:, :-1] = (B[:, :-1] - mu) / sd
    Y = (yi[a][:, None] == cls[None, :]).astype(float)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    pred = cls[np.argmax(B @ W, 1)]
    counts = np.array([(yi[b] == v).sum() for v in np.unique(yi[b])], float)
    return {"acc": round(float(np.mean(pred == yi[b])), 4),
            "majority": round(float(counts.max() / counts.sum()), 4),
            "n_classes": int(len(cls))}


REG = ["hier_score", "hier_held", "hier_rank", "succ_denial", "cart_depth_total", "cart0_depth", "n_controlled",
       "health", "armor", "ammo", "speed", "tss", "n_weapons", "cell"]
CLS = ["is_pw", "pw_team", "team", "weapon_bit", "alive"]


def score(F, shuffled=False):
    out = {}
    local = np.random.default_rng(1234)
    for target in REG:
        y = META[target].astype(float).copy()
        if shuffled:
            local.shuffle(y)
        out[target] = {"r2": r2(F, y), "in_x": target in IN_X}
    for target in CLS:
        y = META[target].astype(float).copy()
        if shuffled:
            local.shuffle(y)
        result = acc(F, y)
        out[target] = dict(result or {}, in_x=target in IN_X)
    return out


def rank(M, tol=1e-6):
    return int(np.linalg.matrix_rank(M - M.mean(0), tol=tol))


def spectrum(M):
    centered = M - M.mean(0)
    sv = np.linalg.svd(centered, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum()) or 1.0
    fraction = np.cumsum(energy) / total
    participation = float((energy.sum() ** 2) / ((energy ** 2).sum() or 1.0))
    return {"rank_1e-6": rank(M),
            "effective_rank_participation": round(participation, 3),
            "dims_for_90pct_variance": int(np.searchsorted(fraction, 0.90) + 1),
            "dims_for_99pct_variance": int(np.searchsorted(fraction, 0.99) + 1),
            "top_singular_value": round(float(sv[0]), 4)}


sweep = {}
for path in args.checkpoint:
    model = policy(0, path)
    updates = getattr(model, "checkpoint_updates", None)
    sweep[f"IR_trained_{updates}updates"] = {"checkpoint": os.path.abspath(path), "updates": updates,
                                             "load_exception": getattr(model, "checkpoint_load_exception", None),
                                             "ir": ir_of(model)}
IR_T = list(sweep.values())[-1]["ir"]
IR_R = ir_of(policy(0))
IR_R7 = ir_of(policy(7))
P = RAW @ (rng.standard_normal((RAW.shape[1], WIDTHS.d_ir)) / np.sqrt(RAW.shape[1]))

out = {
    "provenance": {
        "journal": os.path.abspath(args.journal), "n_steps": len(recs),
        "n_participant_rows": int(RAW.shape[0]), "simulation_used": False,
        "frame": "inputs.native_inputs over the journal's engine rows",
        "trained": [{"checkpoint": value["checkpoint"], "updates": value["updates"],
                     "load_exception": value["load_exception"]} for value in sweep.values()],
        "split": "by step, 60/40, seed 0",
        "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
    },
    "ir_width": {"d": WIDTHS.d_ir, "spec_requirement": ">=128d"},
    "ranks": {"raw_inputs": rank(RAW), "raw_dims": int(RAW.shape[1]), "n_rows": int(RAW.shape[0])},
    "spectra": {"raw_inputs": spectrum(RAW), "IR_trained": spectrum(IR_T),
                "IR_random_init": spectrum(IR_R), "control_randproj": spectrum(P)},
    "ir_scale": {"trained_absmax": round(float(np.abs(IR_T).max()), 4),
                 "trained_std": round(float(IR_T.std()), 4),
                 "random_init_absmax": round(float(np.abs(IR_R).max()), 4)},
    "targets_in_model_input_x": sorted(IN_X),
    "training_budget_sweep": {name: {"updates": value["updates"],
                                     "effective_rank": spectrum(value["ir"])["effective_rank_participation"],
                                     "probes": score(value["ir"])} for name, value in sweep.items()},
    "probes": {
        "IR_trained": score(IR_T),
        "IR_random_init_s0": score(IR_R),
        "IR_random_init_s7": score(IR_R7),
        "control_randproj": score(P),
        "control_shuffled_labels_IR_trained": score(IR_T, shuffled=True),
        "reference_raw_inputs": score(RAW),
    },
}
with open(args.out, "w") as target:
    json.dump(out, target, indent=2)
print(json.dumps({key: out[key] for key in ("provenance", "ranks", "spectra")}, indent=2))
print("WROTE", args.out)
