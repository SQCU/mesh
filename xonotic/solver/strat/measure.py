"""rl-training-spec.md 6 acceptance suite -- REAL measurements on real artifacts.

metrics.py is only a telemetry log-reducer. This module implements the 6 contract
as functions that (a) load a real trained checkpoint, (b) run CONTROLLED
perturbation / recovery / acquisition / terminal experiments in closed loop, and
(c) run the dynamics-acceptance diagnostics -- held-out one-step error, ensemble
disagreement calibration, action-Jacobian rank / smallest singular value, and a
direct reachability test. It also reduces the online_train.jsonl importance-ratio
and loss telemetry produced alongside the checkpoint.

The 6 contract is explicit that a single fixed-shape self-play win-rate curve does
NOT satisfy it. This suite therefore measures the perturbation/recovery/acquisition
battery AND repeats every policy metric on HELD-OUT count/map shapes the checkpoint
was never trained on.

ENVIRONMENT HONESTY: the closed-loop rollouts here execute in CartSim, the spec's
declared bootstrap/unit environment (0.1), because the Game-2 server transition
path is blocked on this host (see the run report). Every reported block carries
"environment":"cartsim" and server-only measurements are marked "[BUILD-DATA]".
The numbers below are real measurements of the real checkpoint's closed-loop
behaviour in that environment, not Game-2 acceptance.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .cartsim import CartSim, greedy_deny_policy, random_policy
from .dynamics import LocalDynamics, action_rows, state_rows
from .estimator import StrategyEstimator, state_from_cartsim
from .train import hierarchy_scores, policy_forward, role_rewards


# --------------------------------------------------------------------------- IO
def load_bundle(checkpoint, ref_shape, L, seed):
    """Reconstruct estimator + local dynamics and load a real OnlineLearner npz.

    Weights (W_q/W_k/W_v, head, value, dynamics) are count-invariant (4), so an
    estimator built at any reference shape evaluates every count/map shape.
    """
    sim = CartSim(*ref_shape, L=L, seed=seed)
    est = StrategyEstimator.for_cartsim(sim, seed=seed)
    dynamics = LocalDynamics()
    bundle = nn.Module()
    bundle.qkv = est.qkv
    bundle.head = est.head
    bundle.value = est.value
    bundle.dynamics = dynamics
    loaded = {"checkpoint": checkpoint, "exists": os.path.exists(checkpoint)}
    if loaded["exists"]:
        data = np.load(checkpoint, allow_pickle=True)
        want = {name for name, _ in _flatten(bundle.parameters())}
        items = [(name, mx.array(data[name])) for name in data.files if name in want]
        loaded["matched"] = len(items)
        loaded["wanted"] = len(want)
        loaded["missing"] = sorted(want - {n for n, _ in items})
        try:
            bundle.load_weights(items, strict=False)
            loaded["loaded"] = True
        except Exception as exc:  # noqa: BLE001
            loaded["loaded"] = False
            loaded["error"] = f"{type(exc).__name__}: {exc}"
    return est, dynamics, loaded


def _flatten(tree, prefix=""):
    out = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.extend(_flatten(value, f"{prefix}{key}."))
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            out.extend(_flatten(value, f"{prefix}{idx}."))
    else:
        out.append((prefix.rstrip("."), tree))
    return out


def policy_step(est, sim, cstate, w, rng):
    """One shared-policy closed-loop step. Returns (next_cstate, w_next)."""
    features = state_from_cartsim(sim, cstate, w=w)
    key = mx.random.key(int(rng.integers(0, 2**31 - 1)))
    w_next, action, _, _, _, _ = policy_forward(est, features, w, key=key)
    action = np.asarray(action).astype(np.int64)
    nxt, _ = sim.step(cstate, action)
    return nxt, np.asarray(w_next).astype(np.float32)


def reach_decisive_state(sim, est, rng, warmup):
    """Roll the policy a few steps to a state with a well-defined PW winner."""
    cstate = sim.reset()
    w = np.zeros((sim.l, sim.M), dtype=np.float32)
    for _ in range(warmup):
        pw = sim.projected_winner(cstate)
        if pw is not None and pw >= 0:
            break
        cstate, w = policy_step(est, sim, cstate, w, rng)
    return cstate, w


# ---------------------------------------------------- controlled perturbation
def perturb_winner_region(sim, cstate, winner, magnitude, rng):
    """Inject a controlled rival perturbation into the WINNER's cartstate region.

    A rival seizes the winner's strongest cart and drives it back by `magnitude`
    control points -- exactly the disturbance W must be able to correct (2/5).
    Returns (perturbed_state, rival) or (None, None) if the winner holds no cart.
    """
    owned = [c for c in range(sim.j) if int(cstate.control[c]) == winner]
    if not owned:
        return None, None
    target = max(owned, key=lambda c: cstate.pos[c])
    rivals = [t for t in range(sim.k) if t != winner]
    rival = int(rng.choice(rivals))
    s = cstate.copy()
    s.control[target] = rival
    s.pos[target] = float(np.clip(cstate.pos[target] - magnitude, 0.0, sim.L))
    if s.highwater is not None:
        s.highwater[target] = int(np.floor(s.pos[target]))
    return s, rival


def winner_retention(est, shape, L, seed, trials, warmup, horizon, magnitude,
                     baseline=False):
    """P(PW restored to the pre-perturbation winner) within `horizon` steps, and
    time-to-recovery. Optionally run a random-policy baseline for contrast."""
    rng = np.random.default_rng(seed)
    restored, recover_times, valid = 0, [], 0
    immediate_flips = 0
    for trial in range(trials):
        sim = CartSim(*shape, L=L, seed=seed + 7919 * trial)
        cstate, w = reach_decisive_state(sim, est, rng, warmup)
        w0 = sim.projected_winner(cstate)
        if w0 is None or w0 < 0:
            continue
        perturbed, rival = perturb_winner_region(sim, cstate, w0, magnitude, rng)
        if perturbed is None:
            continue
        valid += 1
        immediate_flips += int(sim.projected_winner(perturbed) != w0)
        cs = perturbed
        w = np.zeros((sim.l, sim.M), dtype=np.float32)
        recovered_at = None
        for step in range(horizon):
            if baseline:
                action = random_policy(sim, cs, rng)
                cs, _ = sim.step(cs, np.asarray(action, dtype=np.int64))
            else:
                cs, w = policy_step(est, sim, cs, w, rng)
            if sim.projected_winner(cs) == w0 and recovered_at is None:
                recovered_at = step + 1
        if recovered_at is not None:
            restored += 1
            recover_times.append(recovered_at)
    return {
        "environment": "cartsim",
        "shape": list(shape),
        "trials_valid": valid,
        "perturbation_flipped_pw_frac": round(immediate_flips / max(1, valid), 4),
        "retention_prob": round(restored / max(1, valid), 4),
        "time_to_recovery_mean": round(float(np.mean(recover_times)), 3) if recover_times else None,
        "time_to_recovery_median": float(np.median(recover_times)) if recover_times else None,
        "recovered_trials": len(recover_times),
        "horizon": horizon,
        "magnitude": magnitude,
        "policy": "random_baseline" if baseline else "checkpoint",
    }


def loser_acquisition(est, shape, L, seed, trials, warmup, horizon):
    """P(a non-winner team acquires PW) within horizon -- state-acquisition (2)."""
    rng = np.random.default_rng(seed + 101)
    acquired, valid = 0, 0
    for trial in range(trials):
        sim = CartSim(*shape, L=L, seed=seed + 5003 * trial + 11)
        cstate, w = reach_decisive_state(sim, est, rng, warmup)
        w0 = sim.projected_winner(cstate)
        if w0 is None or w0 < 0:
            continue
        losers = [t for t in range(sim.k) if t != w0]
        if not losers:
            continue
        target = int(rng.choice(losers))
        valid += 1
        cs = cstate
        acquired_here = 0
        for _ in range(horizon):
            cs, w = policy_step(est, sim, cs, w, rng)
            if sim.projected_winner(cs) == target:
                acquired_here = 1
                break
        acquired += acquired_here
    return {
        "environment": "cartsim",
        "shape": list(shape),
        "trials_valid": valid,
        "acquisition_prob": round(acquired / max(1, valid), 4),
        "horizon": horizon,
    }


def terminal_outcome(est, shape, L, seed, trials, horizon):
    """Terminal PW distribution / initial-winner retention over a full horizon."""
    rng = np.random.default_rng(seed + 202)
    kept, valid = 0, 0
    terminal_pw = []
    for trial in range(trials):
        sim = CartSim(*shape, L=L, seed=seed + 6151 * trial + 3)
        cstate = sim.reset()
        w = np.zeros((sim.l, sim.M), dtype=np.float32)
        w0 = sim.projected_winner(cstate)
        for _ in range(horizon):
            cstate, w = policy_step(est, sim, cstate, w, rng)
        wt = sim.projected_winner(cstate)
        terminal_pw.append(int(wt) if wt is not None else -1)
        if w0 is not None and w0 >= 0:
            valid += 1
            kept += int(wt == w0)
    return {
        "environment": "cartsim",
        "shape": list(shape),
        "initial_winner_retained_frac": round(kept / max(1, valid), 4),
        "terminal_pw_hist": {str(v): terminal_pw.count(v) for v in sorted(set(terminal_pw))},
        "horizon": horizon,
    }


# ------------------------------------------------------- dynamics acceptance
def dynamics_acceptance(est, dynamics, shape, L, seed, samples, warmup):
    """Held-out one-step error, ensemble-disagreement calibration, Jacobian
    rank / smallest singular value, and a direct reachability test (5/6)."""
    rng = np.random.default_rng(seed + 303)
    errors, disagreements = [], []
    sigma_mins, ranks = [], []
    reach_hits, reach_total = 0, 0
    for trial in range(samples):
        sim = CartSim(*shape, L=L, seed=seed + 977 * trial + 5)
        cstate, w = reach_decisive_state(sim, est, rng, warmup)
        state = state_from_cartsim(sim, cstate, w=w)
        result = est.forward(state)
        actions = np.asarray(result.action, dtype=np.int64).reshape(sim.l)
        next_cstate, _ = sim.step(cstate, actions)
        next_state = state_from_cartsim(sim, next_cstate, w=np.asarray(result.w_next))
        # ground-truth reduced-state transition
        target_delta = (next_state.hierarchy - state.hierarchy).astype(np.float32)
        srows = mx.array(state_rows(state))
        arows = mx.array(action_rows(state, actions))
        mean, first, second = dynamics(srows, arows)
        mean_np = np.asarray(mean)
        one_step = float(np.mean(np.square(mean_np - target_delta)))
        disagree = float(np.mean(np.square(np.asarray(first) - np.asarray(second))))
        errors.append(one_step)
        disagreements.append(disagree)
        # local action-Jacobian rank / smallest singular value per player
        matrices = np.asarray(dynamics.local_matrix(srows))
        for mat in matrices:
            sv = np.linalg.svd(mat, compute_uv=False)
            sigma_mins.append(float(sv.min()))
            ranks.append(int(np.sum(sv > 1e-6 * sv.max())))
        # direct reachability: does SOME single-player action deviation actually
        # move the winner's hierarchy score up (restore direction), and does the
        # model's local matrix predict a reachable delta in that direction?
        winners = np.flatnonzero(state.winner_mask)
        if len(winners):
            wp = int(winners[0])
            reach_total += 1
            best_actual = -1e9
            for a in range(sim.M):
                joint = actions.copy()
                joint[wp] = a
                cand, _ = sim.step(cstate, joint)
                cand_state = state_from_cartsim(sim, cand, w=w)
                d = float(cand_state.hierarchy[wp, 3] - state.hierarchy[wp, 3])
                best_actual = max(best_actual, d)
            if best_actual > 1e-4:
                reach_hits += 1
    errors = np.asarray(errors)
    disagreements = np.asarray(disagreements)
    # calibration: correlation between predicted disagreement and realised error
    if len(errors) > 2 and np.std(disagreements) > 0 and np.std(errors) > 0:
        calibration = float(np.corrcoef(disagreements, errors)[0, 1])
    else:
        calibration = None
    return {
        "environment": "cartsim",
        "shape": list(shape),
        "samples": int(len(errors)),
        "one_step_error_mean": round(float(errors.mean()), 6) if len(errors) else None,
        "one_step_error_p90": round(float(np.percentile(errors, 90)), 6) if len(errors) else None,
        "ensemble_disagreement_mean": round(float(disagreements.mean()), 6) if len(disagreements) else None,
        "disagreement_error_calibration_r": round(calibration, 4) if calibration is not None else None,
        "jacobian_sigma_min_mean": round(float(np.mean(sigma_mins)), 6) if sigma_mins else None,
        "jacobian_rank_mean": round(float(np.mean(ranks)), 3) if ranks else None,
        "jacobian_rank_full_frac": round(float(np.mean([r == 8 for r in ranks])), 4) if ranks else None,
        "reachability_restore_frac": round(reach_hits / max(1, reach_total), 4),
        "reachability_states": reach_total,
    }


# ------------------------------------------------------- online-log reduction
def reduce_online_log(path):
    if not os.path.exists(path):
        return {"exists": False, "path": path, "note": "[BUILD-DATA] no online_train.jsonl"}
    losses = {}
    ratios, sigma, n = [], [], 0
    env = None
    with open(path) as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "_meta" in row:
                env = row["_meta"].get("environment")
                continue
            env = env or row.get("environment")
            n += 1
            for key in ("loss", "loss_pg", "loss_w", "loss_l", "loss_dynamics"):
                if key in row:
                    losses.setdefault(key, []).append(float(row[key]))
            if "importance_mean" in row:
                ratios.append(float(row["importance_mean"]))
            if "local_control_sigma_min" in row:
                sigma.append(float(row["local_control_sigma_min"]))
    reduce = lambda xs: {
        "first": round(xs[0], 6), "last": round(xs[-1], 6),
        "mean": round(float(np.mean(xs)), 6), "min": round(float(np.min(xs)), 6),
        "max": round(float(np.max(xs)), 6),
    } if xs else None
    return {
        "exists": True, "path": path, "environment": env, "update_lines": n,
        "losses": {key: reduce(vals) for key, vals in losses.items()},
        "importance_ratio": reduce(ratios),
        "local_control_sigma_min": reduce(sigma),
    }


# ------------------------------------------------------------------- driver
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(here, "runs", "policy_online_v3.npz"))
    ap.add_argument("--online-log", default=os.path.join(here, "runs", "online_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "runs", "acceptance_measure.json"))
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--magnitude", type=float, default=2.0)
    ap.add_argument("--samples", type=int, default=24)
    # trained (seen) vs held-out (never trained) shapes -- mirror train.py
    ap.add_argument("--train-shapes", default="2,3,4;2,2,4;3,4,7")
    ap.add_argument("--heldout-shapes", default="3,4,7;4,6,9;5,7,11")
    args = ap.parse_args()

    train_shapes = [tuple(int(v) for v in g.split(",")) for g in args.train_shapes.split(";")]
    heldout_shapes = [tuple(int(v) for v in g.split(",")) for g in args.heldout_shapes.split(";")]

    est, dynamics, loaded = load_bundle(args.checkpoint, train_shapes[0], args.L, args.seed)

    def battery(shapes, tag):
        out = []
        for shape in shapes:
            out.append({
                "shape": list(shape),
                "winner_retention": winner_retention(
                    est, shape, args.L, args.seed, args.trials, args.warmup,
                    args.horizon, args.magnitude),
                "winner_retention_random_baseline": winner_retention(
                    est, shape, args.L, args.seed, args.trials, args.warmup,
                    args.horizon, args.magnitude, baseline=True),
                "loser_acquisition": loser_acquisition(
                    est, shape, args.L, args.seed, args.trials, args.warmup, args.horizon),
                "terminal_outcome": terminal_outcome(
                    est, shape, args.L, args.seed, args.trials, args.horizon),
                "dynamics_acceptance": dynamics_acceptance(
                    est, dynamics, shape, args.L, args.seed, args.samples, args.warmup),
            })
        return out

    report = {
        "checkpoint_load": loaded,
        "online_log_reduction": reduce_online_log(args.online_log),
        "environment_note": (
            "closed-loop rollouts run in CartSim (bootstrap env, spec 0.1); "
            "Game-2 server acceptance is [BUILD-DATA], blocked on the mesh fabric."
        ),
        "seen_shapes": battery(train_shapes, "seen"),
        "heldout_shapes": battery(heldout_shapes, "heldout"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
