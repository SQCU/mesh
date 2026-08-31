"""Surrogate-2 measurement: anticipatory (oracle) replicator update vs the standard
integrator, SAME trained weights, both selectable by the `est.anticipatory` switch.

Abdelraouf & Shamma: standard RD is the integrator g_R(s)=1/s (score z = ∫p); oracle /
anticipatory RD is g_OR(s)=1+1/s (score z = ∫p + p, cumulative payoff augmented by its
instantaneous value) and UNIFORMLY DOMINATES standard RD — larger gain, smaller phase
lag, so the score tracks the payoff better and cumulative reward is higher.

Our score is the per-(player,instrument) weight w; the payoff is the head velocity dw/dt.
Standard: sample from w_next = w + Δ·dw/dt. Anticipatory: sample from w_next + lead·dw/dt.
This script plays identical self-play matches under both modes (identical seeds, identical
weights) and reports cumulative team-0 role reward, PW-control, and a score-vs-payoff
phase-lag proxy (cross-correlation lag between the mean score and the mean payoff signal).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .cartsim import CartSim, greedy_deny_policy
from .dominance_driver import load_estimator, role_rewards
from .estimator import state_from_cartsim, strategy_forward

RUNS = os.path.join(os.path.dirname(__file__), "runs")


def _xcorr_lag(score, payoff, max_lag=5):
    """Lag (in steps) maximizing correlation of `score` shifted vs `payoff`.

    Positive lag = score LAGS the payoff. Less lag (closer to 0 / more negative) is better
    tracking — the oracle-RD prediction. Returns (best_lag, best_corr).
    """
    s = np.asarray(score, dtype=np.float64)
    p = np.asarray(payoff, dtype=np.float64)
    s = s - s.mean()
    p = p - p.mean()
    if np.allclose(s, 0) or np.allclose(p, 0):
        return 0, 0.0
    best_lag, best = 0, -2.0
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, b = s, p
        else:
            a, b = s[lag:], p[:-lag]   # score delayed by `lag` aligned to earlier payoff
        if len(a) < 3:
            continue
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            continue
        c = float(np.dot(a, b) / denom)
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


def play(est, sim, seed, n_steps, anticipatory, lead):
    import mlx.core as mx

    state = sim.reset()
    w = np.zeros((sim.l, sim.M), dtype=np.float32)
    cum_reward = 0.0
    pw_control = 0
    score_series, payoff_series = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        feats = state_from_cartsim(sim, state, w=w)
        key = mx.random.key(int(rng.integers(0, 2**31 - 1)))
        out = strategy_forward(
            est, feats, w, key=key, anticipatory=anticipatory, lead=lead
        )
        a0 = np.asarray(out["action"]).astype(np.int64)
        w = np.asarray(out["w_next"]).astype(np.float32)
        # team-0 mean score (sampling logits) and mean payoff (dw/dt) this step
        team0 = np.asarray(sim.team_of) == 0
        score_series.append(float(np.asarray(out["score"])[team0].mean()))
        payoff_series.append(float(np.asarray(out["dw_dt"])[team0].mean()))
        # opponents: greedy_deny controls every non-focus team
        actions = greedy_deny_policy(sim, state)
        for p in range(sim.l):
            if int(sim.team_of[p]) == 0:
                actions[p] = a0[p]
        nxt, _ = sim.step(state, actions)
        cum_reward += float(role_rewards(sim, state, nxt)[0])
        if sim.projected_winner(nxt) == 0:
            pw_control += 1
        state = nxt
    lag, corr = _xcorr_lag(score_series, payoff_series)
    return {
        "cumulative_reward": cum_reward,
        "pw_control_fraction": pw_control / max(1, n_steps),
        "score_payoff_lag": lag,
        "score_payoff_corr": corr,
    }


def measure(checkpoint, shapes, n_games=40, n_steps=24, lead=1.0, seed0=9000):
    rows = []
    for (k, j, l) in shapes:
        std_r, ant_r = [], []
        est = None
        for g in range(n_games):
            sim = CartSim(k, j, l, L=8, seed=seed0 + g)
            if est is None:
                est, loaded, detail = load_estimator(sim, checkpoint)
                measure._detail = detail
            else:
                est, _, _ = load_estimator(sim, checkpoint)
            # identical sim seed + weights; only the update rule differs
            std_r.append(play(est, CartSim(k, j, l, L=8, seed=seed0 + g),
                              seed0 + g, n_steps, anticipatory=False, lead=lead))
            ant_r.append(play(est, CartSim(k, j, l, L=8, seed=seed0 + g),
                              seed0 + g, n_steps, anticipatory=True, lead=lead))
        def agg(rs, key):
            return float(np.mean([r[key] for r in rs]))
        row = {
            "shape": [k, j, l],
            "standard": {
                "cumulative_reward": agg(std_r, "cumulative_reward"),
                "pw_control_fraction": agg(std_r, "pw_control_fraction"),
                "score_payoff_lag": agg(std_r, "score_payoff_lag"),
            },
            "anticipatory": {
                "cumulative_reward": agg(ant_r, "cumulative_reward"),
                "pw_control_fraction": agg(ant_r, "pw_control_fraction"),
                "score_payoff_lag": agg(ant_r, "score_payoff_lag"),
            },
        }
        row["delta_cumulative_reward"] = (
            row["anticipatory"]["cumulative_reward"] - row["standard"]["cumulative_reward"]
        )
        row["delta_pw_control"] = (
            row["anticipatory"]["pw_control_fraction"] - row["standard"]["pw_control_fraction"]
        )
        row["delta_lag"] = (
            row["anticipatory"]["score_payoff_lag"] - row["standard"]["score_payoff_lag"]
        )
        rows.append(row)
    n = len(rows)
    return {
        "config": {
            "n_games": n_games, "n_steps": n_steps, "lead": lead,
            "checkpoint": getattr(measure, "_detail", "n/a"),
            "note": "same weights; anticipatory adds lead*dw/dt (oracle RD z=∫p+p) to the "
                    "sampling score. Positive delta_cumulative_reward => anticipatory "
                    "dominates; negative delta_lag => less score-vs-payoff phase lag.",
        },
        "per_shape": rows,
        "summary": {
            "shapes_anticipatory_reward_ge_standard":
                int(sum(1 for r in rows if r["delta_cumulative_reward"] >= -1e-9)),
            "shapes_total": n,
            "mean_delta_cumulative_reward":
                float(np.mean([r["delta_cumulative_reward"] for r in rows])),
            "mean_delta_lag": float(np.mean([r["delta_lag"] for r in rows])),
        },
    }


def main():
    os.makedirs(RUNS, exist_ok=True)
    checkpoint = os.path.join(RUNS, "policy_online_v3.npz")
    shapes = [(2, 3, 4), (3, 4, 6), (4, 4, 8), (3, 5, 7), (4, 6, 9)]
    out = measure(checkpoint, shapes)
    with open(os.path.join(RUNS, "anticipatory_vs_standard.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    for r in out["per_shape"]:
        print(f"shape {r['shape']}: Δreward={r['delta_cumulative_reward']:+.4f} "
              f"Δpw={r['delta_pw_control']:+.4f} Δlag={r['delta_lag']:+.3f}")


if __name__ == "__main__":
    main()
