"""Drive the OnlineLearner over REAL transitions across >=2 curriculum shapes.

WHY THIS EXISTS / HONESTY TAG
-----------------------------
Per rl-training-spec.md 0.1 the PRIMARY training environment is the Xonotic
dedicated server (strat_responder --train over the RDMA mesh). That path is
CURRENTLY BLOCKED on this machine: the single mesh fabric region (/mesh0) is
occupied by the sacrosanct live 26012 server (mesh-stat client:0 => solverless),
and the online-learner code (online.py/dynamics.py) is uncommitted on the MBP
while mlx lives only on the mini. See the run report for the exact evidence.

This driver is the honest fallback that still answers Task 1's core question:
does the built-but-unrun OnlineLearner actually CONSUME real transitions and
UPDATE the single shared weight tensor across >=2 curriculum shapes WITHOUT
resizing? It feeds the learner transitions from CartSim (the spec's declared
bootstrap / unit environment, 0.1) -- NOT the Game-2 server. Every telemetry
line is tagged environment="cartsim_bootstrap" so it can never be mistaken for
Game-2 evidence. The learner, losses, importance-ratio correction, dynamics
ensemble, and checkpoint format exercised here are the exact same objects the
server loop drives.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx

from .cartsim import CartSim
from .estimator import StrategyEstimator, state_from_cartsim
from .head import strategy_log_prob
from .online import OnlineLearner
from .train import hierarchy_scores, role_rewards


def run(shapes, steps_per_shape, off_policy_players, seed, out_path, ckpt_path,
        learning_rate, L):
    rng = np.random.default_rng(seed)
    # ONE estimator, ONE learner -- shared weights reused across every shape.
    base = CartSim(*shapes[0], L=L, seed=seed)
    est = StrategyEstimator.for_cartsim(base, seed=seed)
    learner = OnlineLearner(est, learning_rate=learning_rate, checkpoint=ckpt_path)

    # Fingerprint the shared weight tensor W_q before any update, to prove the
    # SAME tensor (same shape) is what gets revised across all shapes.
    wq0 = np.asarray(est.qkv.W_q).copy()
    wq_shape = tuple(wq0.shape)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stream = open(out_path, "w")
    header = {
        "_meta": {
            "environment": "cartsim_bootstrap",
            "not_game2_evidence": True,
            "reason": "server/mesh path blocked; see run report",
            "shapes": [list(s) for s in shapes],
            "steps_per_shape": steps_per_shape,
            "off_policy_players": off_policy_players,
            "seed": seed,
            "W_q_shape": list(wq_shape),
            "learning_rate": learning_rate,
        }
    }
    stream.write(json.dumps(header) + "\n")
    stream.flush()

    n_lines = 0
    started = time.time()
    for shape in shapes:
        k, j, l = shape
        sim = CartSim(k, j, l, L=L, seed=int(rng.integers(1, 2**31 - 1)))
        cstate = sim.reset()
        w = np.zeros((l, sim.M), dtype=np.float32)
        previous = None
        for step in range(steps_per_shape):
            state = state_from_cartsim(sim, cstate, w=w)
            result = est.forward(state)
            actions = np.asarray(result.action, dtype=np.int64).reshape(l)
            w_next = np.asarray(result.w_next, dtype=np.float32).reshape(l, sim.M)
            # on-policy log-prob of the sampled joint action
            target_logp = np.asarray(
                strategy_log_prob(result.w_next, mx.array(actions), est.temperature),
                dtype=np.float32,
            )
            behavior_logp = target_logp.copy()
            # participant-local off-policy: a few rows draw uniform exploratory
            # actions and carry their own behavior log-prob (0.1 off-policy rule)
            off = np.zeros(l, dtype=bool)
            n_off = min(off_policy_players, l)
            if n_off:
                chosen = rng.choice(l, size=n_off, replace=False)
                for p in chosen:
                    elig = np.flatnonzero(state.eligible[p]) if state.eligible is not None else np.arange(sim.M)
                    actions[p] = int(rng.choice(elig))
                    behavior_logp[p] = -np.log(max(1, len(elig)))
                off[chosen] = True

            if previous is not None:
                metrics = learner.observe(previous, state, cstate)
                if metrics is not None:
                    _pwb = sim.projected_winner(previous["cartstate"])
                    _pwa = sim.projected_winner(cstate)
                    pw_b = int(_pwb) if _pwb is not None else -1
                    pw_a = int(_pwa) if _pwa is not None else -1
                    h_b = hierarchy_scores(sim, previous["cartstate"])
                    h_a = hierarchy_scores(sim, cstate)
                    line = {
                        "environment": "cartsim_bootstrap",
                        "shape": [k, j, l],
                        "t": round(time.time() - started, 3),
                        "W_q_shape": list(np.asarray(est.qkv.W_q).shape),
                        "pw_before": pw_b,
                        "pw_after": pw_a,
                        "pw_changed": int(pw_b != pw_a),
                        "hierarchy_delta_max": float(np.max(np.abs(h_a - h_b))),
                        "off_policy_rows": int(off.sum()),
                        **{key: metrics[key] for key in sorted(metrics)},
                    }
                    stream.write(json.dumps(line) + "\n")
                    stream.flush()
                    n_lines += 1

            next_cstate, _ = sim.step(cstate, actions)
            previous = {
                "sim": sim,
                "state": state,
                "cartstate": cstate,
                "w_in": w.copy(),
                "w_out": w_next.copy(),
                "actions": actions.copy(),
                "behavior_logp": behavior_logp.copy(),
            }
            cstate = next_cstate
            w = w_next
        # flush the tail of this shape as terminal before switching shapes
        if previous is not None:
            tail = learner.flush(previous["state"], previous["cartstate"], terminal=True)
            if tail is not None:
                stream.write(json.dumps({
                    "environment": "cartsim_bootstrap",
                    "shape": [k, j, l], "terminal_flush": True,
                    **{key: tail[key] for key in sorted(tail)},
                }) + "\n")
                stream.flush()
                n_lines += 1

    stream.close()
    learner.save(ckpt_path)

    # Prove the shared tensor moved but did NOT change shape.
    wq1 = np.asarray(est.qkv.W_q)
    delta = float(np.linalg.norm(wq1 - wq0))
    summary = {
        "telemetry_lines": n_lines,
        "updates": learner.updates,
        "W_q_shape_before": list(wq_shape),
        "W_q_shape_after": list(wq1.shape),
        "W_q_shape_unchanged": wq_shape == tuple(wq1.shape),
        "W_q_l2_change": delta,
        "shapes_trained": [list(s) for s in shapes],
        "checkpoint": ckpt_path,
        "telemetry": out_path,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--off-policy-players", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(here, "runs", "online_train.jsonl"))
    ap.add_argument("--checkpoint", default=os.path.join(here, "runs", "policy_online_v3.npz"))
    ap.add_argument("--shapes", default="2,3,4;3,4,7;2,2,4")
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in group.split(",")) for group in args.shapes.split(";")]
    run(shapes, args.steps, args.off_policy_players, args.seed, args.out,
        args.checkpoint, args.learning_rate, args.L)


if __name__ == "__main__":
    main()
