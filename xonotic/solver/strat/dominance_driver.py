"""Cross-team (all-to-all) vs team-state-only Pareto-dominance demonstration.

Runs the two artifacts of `design/dominance-demo.md`:

  * Part 2 -- `aliasing_counterexample.json`: an explicit pair of global cartstates
    aliased under the team-only partition (identical own-team view, different `SUCC`).
    Shows the team-only policy emits an IDENTICAL instrument distribution on both
    (blind), the all-to-all `StrategyEstimator` and the all-to-all `greedy_deny`
    reference emit DIFFERENT (correct) actions, and the one-step RELATIVE objective
    (PW-denial / possession, rl-training-spec §5) differs -- quantifying the forced
    denial regret the coarse policy cannot avoid.

  * Part 3 -- `dominance_headtohead.json`: many CartSim matches, all-to-all policy vs
    the no-regret team-only baseline (both as the focus team against a common
    all-to-all `greedy_deny` opponent), across training and held-out shapes. Measures
    Pareto dominance across terminal win, PW-control fraction, winner-retention, and
    loser-acquisition, with per-seed win counts + a Wilson CI, and instruments the
    baseline's real-match aliasing frequency and cumulative objective cost.

Needs mlx (the `StrategyEstimator` forward). Run on the Apple mini:
    ~/.venv-mesh/bin/python -m solver.strat.dominance_driver
Everything else (CartSim, game, baseline) is pure numpy.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .cartsim import (
    CartSim,
    decode_instrument,
    greedy_deny_policy,
    instrument_index,
    to_carts,
)
from .game import projected_winner, succession, team_nimbers
from .baseline_teamonly import TeamOnlyBaseline, own_team_view

RUNS = os.path.join(os.path.dirname(__file__), "runs")


# reward / objective helpers: the ONE canonical team-level definitions live in
# train.py (was a numpy replica here). `role_rewards` here returns the per-TEAM vector
# (callers read team 0 as [0]); train.role_rewards expands the same thing per player.
from .train import hierarchy_scores, team_role_rewards as role_rewards


def leader_deepest_cart(sim, state, leader):
    best_c, best_d = -1, -1
    ctrl = np.asarray(state.control)
    for c in range(sim.j):
        if int(ctrl[c]) == leader:
            d = int(np.floor(state.pos[c]))
            if d > best_d:
                best_d, best_c = d, c
    return best_c


# --------------------------------------------------------------------------- #
# estimator (all-to-all) wrapper: load a trained checkpoint, act as one team
# --------------------------------------------------------------------------- #
def load_estimator(sim, checkpoint, seed=0):
    import mlx.core as mx
    import mlx.nn as nn
    from .estimator import StrategyEstimator
    from .dynamics import LocalDynamics

    est = StrategyEstimator.for_cartsim(sim, seed=seed)
    bundle = nn.Module()
    bundle.qkv = est.qkv
    bundle.encoder = est.encoder
    bundle.head = est.head
    bundle.value = est.value
    bundle.dynamics = LocalDynamics()
    loaded = False
    detail = "random-init"
    if checkpoint and os.path.exists(checkpoint):
        data = np.load(checkpoint, allow_pickle=False)
        items = [
            (k, mx.array(data[k]))
            for k in data.files
            if not k.startswith("__opt__") and k != "__updates__"
        ]
        matched = 0
        params = dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(bundle.parameters()))
        keep = []
        for k, v in items:
            if k in params and tuple(params[k].shape) == tuple(v.shape):
                keep.append((k, v))
                matched += 1
        bundle.load_weights(keep, strict=False)
        loaded = matched > 0
        detail = f"loaded {matched}/{len(items)} tensors from {os.path.basename(checkpoint)}"
    return est, loaded, detail


def estimator_action(est, sim, state, w):
    """Sampled per-player instrument indices from the all-to-all estimator."""
    from .estimator import state_from_cartsim

    feats = state_from_cartsim(sim, state, w=w)
    result = est.forward(feats)
    action = np.asarray(result.action).astype(np.int64)
    w_next = np.asarray(result.w_next).astype(np.float32)
    return action, w_next


# --------------------------------------------------------------------------- #
# Part 2: the aliasing counterexample
# --------------------------------------------------------------------------- #
def build_counterexample(checkpoint):
    # 3 teams, 3 carts, 6 players (2/team). team0 = acting focus team (a loser).
    sim = CartSim(3, 3, 6, team_of=[0, 1, 2, 0, 1, 2], L=8, seed=0)

    def mk(depths, control):
        s = sim.reset()
        s.pos = np.array(depths, dtype=np.float64)
        s.control = np.array(control, dtype=np.int64)
        s.banked = np.zeros(sim.k)
        s.highwater = np.floor(s.pos).astype(np.int64)
        return s

    # team0 controls ONLY cart0 at depth 2 in BOTH states (own-view fixed).
    # Rival configuration is SWAPPED between the two non-own carts, so PW/SUCC differ
    # but the own-team partition cannot tell the two states apart.
    A = mk([2, 1, 3], [0, 1, 2])  # leader = team2 on cart2 (deeper)
    B = mk([2, 3, 1], [0, 1, 2])  # leader = team1 on cart1

    def snap(S):
        carts = to_carts(S)
        return {
            "pos": [int(x) for x in S.pos],
            "control": [int(x) for x in S.control],
            "nimbers": {int(k): int(v) for k, v in team_nimbers(carts, range(sim.k)).items()},
            "PW": (None if projected_winner(carts, range(sim.k)) is None
                   else int(projected_winner(carts, range(sim.k)))),
            "SUCC": [[int(t), int(v)] for t, v in succession(carts, range(sim.k))],
            "own_view_team0": {
                "agg": own_team_view(sim, S, 0)["agg"].tolist(),
                "cart_desc": own_team_view(sim, S, 0)["cart_desc"].tolist(),
                "own_carts": own_team_view(sim, S, 0)["own_carts"],
            },
        }

    vA, vB = own_team_view(sim, A, 0), own_team_view(sim, B, 0)
    own_view_identical = bool(
        np.allclose(vA["agg"], vB["agg"]) and np.allclose(vA["cart_desc"], vB["cart_desc"])
    )

    # team-only marginal instrument distribution for a team0 player (must be identical)
    bl = TeamOnlyBaseline(seed=1)
    dA = bl.action_distribution(sim, A, 0)
    dB = bl.action_distribution(sim, B, 0)
    teamonly_identical = bool(np.allclose(dA, dB))
    labels = [list(decode_instrument(m, sim.j)) for m in range(sim.M)]

    # all-to-all greedy_deny reference (provably targets the leader's deepest cart)
    gA = greedy_deny_policy(sim, A)
    gB = greedy_deny_policy(sim, B)
    greedy_team0 = {
        "A": [list(decode_instrument(int(gA[p]), sim.j)) for p in [0, 3]],
        "B": [list(decode_instrument(int(gB[p]), sim.j)) for p in [0, 3]],
    }

    # all-to-all trained estimator argmax action for the team0 players
    est, loaded, detail = load_estimator(sim, checkpoint)
    w0 = np.zeros((sim.l, sim.M), dtype=np.float32)
    from .estimator import state_from_cartsim
    import mlx.core as mx

    def est_argmax_team0(S):
        feats = state_from_cartsim(sim, S, w=w0)
        r = est.forward(feats)
        wn = np.asarray(r.w_next)
        acts = wn.argmax(axis=1)
        return [list(decode_instrument(int(acts[p]), sim.j)) for p in [0, 3]]

    est_team0 = {"A": est_argmax_team0(A), "B": est_argmax_team0(B)}

    # one-step RELATIVE objective (PW-denial, §5) under a controlled protocol:
    # rivals abandon (idle), team0 suppresses a chosen non-own cart. Denying the
    # projected winner (PW leaves the incumbent) is objective success.
    def one_step_deny(S, cart):
        a = np.full(sim.l, instrument_index("idle", 0, sim.j), dtype=np.int64)
        for p in [0, 3]:
            a[p] = instrument_index("suppress_cart", cart, sim.j)
        nx, info = sim.step(S, a)
        pb, pa = info["pw_before"], info["pw_after"]
        denied = int(pb is not None and pa != pb)  # projected winner dislodged
        return {
            "pw_before": None if pb is None else int(pb),
            "pw_after": None if pa is None else int(pa),
            "denied_projected_winner": denied,
            "team0_shaped_reward": float(role_rewards(sim, S, nx)[0]),
        }

    # correct target: leader's deepest cart in each state
    correct = {
        "A": leader_deepest_cart(sim, A, projected_winner(to_carts(A), range(sim.k))),
        "B": leader_deepest_cart(sim, B, projected_winner(to_carts(B), range(sim.k))),
    }
    outcomes = {
        "A": {f"suppress_cart{c}": one_step_deny(A, c) for c in [1, 2]},
        "B": {f"suppress_cart{c}": one_step_deny(B, c) for c in [1, 2]},
    }

    # forced regret: a team-only policy MUST use a target t independent of the swap
    # (its distribution is identical on A,B). Denial objective success for a fixed
    # blind target t in {cart1, cart2}:
    def deny_success(state_key, cart):
        return outcomes[state_key][f"suppress_cart{cart}"]["denied_projected_winner"]

    reg = {}
    for t in [1, 2]:
        success = deny_success("A", t) + deny_success("B", t)  # out of 2
        reg[f"fixed_target_cart{t}"] = {
            "denials_out_of_2": int(success),
            "denial_failures_out_of_2": int(2 - success),
        }
    all_to_all_denials = deny_success("A", correct["A"]) + deny_success("B", correct["B"])
    forced_denial_failures = min(reg[f"fixed_target_cart{t}"]["denial_failures_out_of_2"] for t in [1, 2])

    return {
        "setup": {
            "k": sim.k, "j": sim.j, "l": sim.l, "L": sim.L,
            "team_of": [int(x) for x in sim.team_of],
            "focus_team": 0,
            "protocol": "rivals idle (abandon); team0's two players suppress the chosen non-own cart",
            "instrument_labels": labels,
        },
        "states": {"A": snap(A), "B": snap(B)},
        "own_view_identical": own_view_identical,
        "team_only_instrument_distribution": {
            "A": [round(float(x), 4) for x in dA],
            "B": [round(float(x), 4) for x in dB],
            "identical": teamonly_identical,
        },
        "all_to_all_actions": {
            "greedy_deny_reference_team0": greedy_team0,
            "trained_estimator_team0_argmax": est_team0,
            "estimator_checkpoint": detail,
            "estimator_loaded": loaded,
            "estimator_differs_A_vs_B": bool(est_team0["A"] != est_team0["B"]),
            "greedy_differs_A_vs_B": bool(greedy_team0["A"] != greedy_team0["B"]),
        },
        "correct_target_cart": correct,
        "one_step_outcomes": outcomes,
        "forced_regret": {
            "blind_fixed_target": reg,
            "all_to_all_denials_out_of_2": int(all_to_all_denials),
            "team_only_forced_denial_failures_out_of_2": int(forced_denial_failures),
            "team_only_min_failure_rate": forced_denial_failures / 2.0,
            "all_to_all_failure_rate": (2 - all_to_all_denials) / 2.0,
            "note": (
                "The team-only policy emits one distribution for both aliased states, so it "
                "denies the projected winner in at most one of the two; the all-to-all policy "
                "denies in both. The shaped scalar reward is partition-symmetric (it scores "
                "aggregate rival strength, not rival identity), so the aliasing cost is "
                "carried by the PW-denial objective (spec §5), not the shaping term."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Part 3: empirical head-to-head
# --------------------------------------------------------------------------- #
def play_match(sim, focus_policy, seed, n_steps, est=None, baseline=None, perturb=0.15):
    """One match: focus_policy controls team 0; greedy_deny controls all other teams.

    focus_policy in {"estimator","baseline"}. Returns per-match metrics for team 0:
    terminal win, PW-control fraction, retention (hold PW next step), acquisition
    (gain PW next step), and -- for the baseline -- aliasing diagnostics (loser-steps
    where its blind suppress target != the true leader's cart, and the objective cost).
    """
    state = sim.reset()
    rng = np.random.default_rng(seed)
    w = np.zeros((sim.l, sim.M), dtype=np.float32)
    pw_control = 0
    retain_num = retain_den = 0
    acq_num = acq_den = 0
    alias_loser_steps = 0
    alias_mis_target = 0
    alias_cost = 0.0
    for step in range(n_steps):
        pw_before = sim.projected_winner(state)
        # opponent (all-to-all) baseline actions for every team, then overwrite team0
        opp = greedy_deny_policy(sim, state)
        actions = opp.copy()
        if focus_policy == "estimator":
            a0, w = estimator_action(est, sim, state, w)
        else:
            a0 = baseline.act(sim, state, teams=[0], sample=True)
        for p in range(sim.l):
            if int(sim.team_of[p]) == 0:
                actions[p] = a0[p]

        # aliasing instrumentation for the blind baseline as a loser
        if focus_policy == "baseline" and pw_before is not None and pw_before != 0:
            alias_loser_steps += 1
            true_leader_cart = leader_deepest_cart(sim, state, pw_before)
            # the baseline's realised team0 target(s)
            p0 = [p for p in range(sim.l) if int(sim.team_of[p]) == 0]
            kinds = [decode_instrument(int(a0[p]), sim.j) for p in p0]
            supp_targets = [c for (kind, c) in kinds if kind == "suppress_cart"]
            mis = (not supp_targets) or all(c != true_leader_cart for c in supp_targets)
            if mis:
                alias_mis_target += 1

        next_state, info = sim.step(state, actions)
        # exogenous rival perturbation (winner-retention-under-perturbation regime):
        # with prob `perturb`, a random rival pushes one extra body against team0's
        # leading cart, testing whether team0 restores its margin.
        if rng.random() < perturb and pw_before == 0:
            lc = leader_deepest_cart(sim, next_state, 0)
            if lc >= 0:
                pa = np.full(sim.l, instrument_index("idle", 0, sim.j), dtype=np.int64)
                # one rival body suppresses team0's lead cart
                rivals = [p for p in range(sim.l) if int(sim.team_of[p]) != 0]
                if rivals:
                    pa[rivals[int(rng.integers(len(rivals)))]] = instrument_index(
                        "suppress_cart", lc, sim.j
                    )
                    next_state, _ = sim.step(next_state, pa)

        pw_after = sim.projected_winner(next_state)
        if pw_after == 0:
            pw_control += 1
        if pw_before == 0:
            retain_den += 1
            retain_num += int(pw_after == 0)
            if focus_policy == "baseline":
                pass
        else:
            acq_den += 1
            acq_num += int(pw_after == 0)

        if focus_policy == "baseline":
            # objective cost of aliasing: role reward gap vs the all-to-all correct
            # denial from the SAME state (counterfactual, does not affect the match).
            if pw_before is not None and pw_before != 0:
                true_leader_cart = leader_deepest_cart(sim, state, pw_before)
                if true_leader_cart >= 0:
                    corr = opp.copy()
                    for p in range(sim.l):
                        if int(sim.team_of[p]) == 0:
                            corr[p] = instrument_index("suppress_cart", true_leader_cart, sim.j)
                    cf_next, _ = sim.step(state, corr)
                    alias_cost += float(
                        role_rewards(sim, state, cf_next)[0] - role_rewards(sim, state, next_state)[0]
                    )
            baseline.update(role_rewards(sim, state, next_state))
        state = next_state
        w = w if focus_policy == "estimator" else w
    terminal_win = int(sim.projected_winner(state) == 0)
    out = {
        "terminal_win": terminal_win,
        "pw_control_fraction": pw_control / max(1, n_steps),
        "retention": retain_num / max(1, retain_den),
        "acquisition": acq_num / max(1, acq_den),
        "retention_trials": retain_den,
        "acquisition_trials": acq_den,
    }
    if focus_policy == "baseline":
        out["alias_loser_steps"] = alias_loser_steps
        out["alias_mis_target_steps"] = alias_mis_target
        out["alias_mis_target_rate"] = alias_mis_target / max(1, alias_loser_steps)
        out["alias_cumulative_objective_cost"] = alias_cost
    return out


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def headtohead(checkpoint, n_games=60, n_steps=20, seed0=7000):
    train_shapes = [(2, 3, 4), (3, 3, 6), (3, 4, 6), (4, 4, 8)]
    heldout_shapes = [(3, 5, 7), (4, 6, 9), (5, 4, 10)]
    dims = ["terminal_win", "pw_control_fraction", "retention", "acquisition"]
    per_shape = []
    baseline = TeamOnlyBaseline(seed=1234)  # ONE shared no-regret learner across shapes

    def eval_shape(shape, heldout):
        k, j, l = shape
        est_rows, bl_rows = [], []
        est = None
        for g in range(n_games):
            sim = CartSim(k, j, l, L=8, seed=seed0 + g)
            if est is None:
                est, loaded, detail = load_estimator(sim, checkpoint)
                eval_shape._detail = detail
            else:
                # rebuild estimator per shape (count-invariant weights, new sim shape)
                est, _, _ = load_estimator(sim, checkpoint)
            est_rows.append(play_match(sim, "estimator", seed0 + g, n_steps, est=est))
            sim_b = CartSim(k, j, l, L=8, seed=seed0 + g)
            bl_rows.append(play_match(sim_b, "baseline", seed0 + g, n_steps, baseline=baseline))
        agg = {"shape": list(shape), "heldout": heldout, "n_games": n_games}
        for d in dims:
            e = np.array([r[d] for r in est_rows], dtype=float)
            b = np.array([r[d] for r in bl_rows], dtype=float)
            wins = int(np.sum(e > b))
            ties = int(np.sum(e == b))
            agg[d] = {
                "estimator_mean": float(e.mean()),
                "baseline_mean": float(b.mean()),
                "delta": float(e.mean() - b.mean()),
                "per_game_estimator_wins": wins,
                "per_game_ties": ties,
                "per_game_baseline_wins": int(np.sum(e < b)),
                "wilson95_estimator_geq_baseline": wilson_ci(
                    int(np.sum(e >= b)), n_games
                ),
            }
        agg["alias_mis_target_rate_mean"] = float(
            np.mean([r["alias_mis_target_rate"] for r in bl_rows])
        )
        agg["alias_cumulative_objective_cost_mean"] = float(
            np.mean([r["alias_cumulative_objective_cost"] for r in bl_rows])
        )
        agg["alias_loser_steps_total"] = int(sum(r["alias_loser_steps"] for r in bl_rows))
        agg["alias_mis_target_steps_total"] = int(
            sum(r["alias_mis_target_steps"] for r in bl_rows)
        )
        # Pareto verdict for this shape
        agg["estimator_dominates_all_dims"] = all(agg[d]["delta"] >= -1e-9 for d in dims)
        agg["estimator_strict_on_some_dim"] = any(agg[d]["delta"] > 1e-6 for d in dims)
        return agg

    for shape in train_shapes:
        per_shape.append(eval_shape(shape, False))
    for shape in heldout_shapes:
        per_shape.append(eval_shape(shape, True))

    # overall verdict
    dominated_shapes = sum(
        1 for s in per_shape if s["estimator_dominates_all_dims"] and s["estimator_strict_on_some_dim"]
    )
    dim_win = {d: sum(1 for s in per_shape if s[d]["delta"] > 1e-6) for d in dims}
    dim_lose = {d: sum(1 for s in per_shape if s[d]["delta"] < -1e-6) for d in dims}
    return {
        "config": {
            "n_games": n_games,
            "n_steps": n_steps,
            "train_shapes": [list(s) for s in train_shapes],
            "heldout_shapes": [list(s) for s in heldout_shapes],
            "opponent": "greedy_deny (all-to-all reference) controls every non-focus team",
            "estimator_checkpoint": getattr(eval_shape, "_detail", "n/a"),
            "dims": dims,
        },
        "per_shape": per_shape,
        "verdict": {
            "shapes_with_strict_pareto_dominance": dominated_shapes,
            "shapes_total": len(per_shape),
            "dim_win_shape_counts": dim_win,
            "dim_lose_shape_counts": dim_lose,
        },
    }


def main():
    os.makedirs(RUNS, exist_ok=True)
    checkpoint = os.path.join(RUNS, "policy_online_v3.npz")
    started = time.time()
    print("[part2] building aliasing counterexample ...", flush=True)
    ce = build_counterexample(checkpoint)
    with open(os.path.join(RUNS, "aliasing_counterexample.json"), "w") as f:
        json.dump(ce, f, indent=2)
    print(json.dumps(ce["forced_regret"], indent=2), flush=True)
    print(f"[part3] head-to-head ... ({time.time()-started:.1f}s)", flush=True)
    hh = headtohead(checkpoint)
    with open(os.path.join(RUNS, "dominance_headtohead.json"), "w") as f:
        json.dump(hh, f, indent=2)
    print(json.dumps(hh["verdict"], indent=2), flush=True)
    print(f"[done] {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
