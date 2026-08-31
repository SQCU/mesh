"""CartSim BOOTSTRAP trainer — NOT the primary environment, NOT Game-2 evidence.

Per rl-training-spec.md §0.1 training IS the Xonotic server process
(strat_responder --train + curriculum). CartSim here is a bootstrap prior, a cheap
policy smoke, and a unit environment only; any win-rate / PW-control curve it produces
is NOT evidence about Game 2. Outputs are tagged bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .cartsim import CartSim, greedy_deny_policy, to_carts
from .dpp import dpp_marginals
from .dynamics import LocalDynamics, action_rows, dynamics_loss, state_rows
from .estimator import StrategyEstimator, state_from_cartsim
from .game import team_nimbers
from .head import integrate_weights, strategy_log_prob
from .qkv import team_pool
from .value import select_role_value


def policy_forward(
    est: StrategyEstimator,
    state,
    w_in: np.ndarray,
    action: Optional[mx.array] = None,
    key: Optional[mx.array] = None,
):
    """Thin wrapper over the single canonical estimator.strategy_forward (no re-inlined
    copy of the forward pass — it lived here and in estimator.forward; now one definition)."""
    from .estimator import strategy_forward

    out = strategy_forward(est, state, w_in, action=action, key=key)
    return (
        out["w_next"],
        out["action"],
        out["logpi"],
        out["value"],
        out["winner_value"],
        out["loser_value"],
    )


def hierarchy_scores(sim: CartSim, state) -> np.ndarray:
    nimbers = team_nimbers(to_carts(state), range(sim.k))
    values = np.array([nimbers.get(team, 0) for team in range(sim.k)], dtype=np.float32)
    scale = float(max(1.0, values.max(initial=0.0), sim.L))
    scores = np.zeros(sim.k, dtype=np.float32)
    for team in range(sim.k):
        rivals = np.delete(values, team)
        scores[team] = np.tanh(
            (values[team] - rivals.mean() if len(rivals) else values[team]) / scale
        )
    return scores


def team_role_rewards(
    sim: CartSim,
    before,
    after,
    margin_weight: float = 0.1,
) -> np.ndarray:
    """Per-TEAM asymmetric role reward (rl-training-spec §2), length k. The single
    canonical definition — train.role_rewards expands it per-player; dominance_driver /
    anticipatory_measure read team 0 directly (was inlined as a numpy replica there)."""
    pw_before = sim.projected_winner(before)
    pw_after = sim.projected_winner(after)
    h_before = hierarchy_scores(sim, before)
    h_after = hierarchy_scores(sim, after)
    team_reward = np.zeros(sim.k, dtype=np.float32)
    for team in range(sim.k):
        if team == pw_before:
            retained = 1.0 if pw_after == team else -1.0
            team_reward[team] = retained + margin_weight * (
                h_after[team] - h_before[team]
            )
        else:
            acquired = 1.0 if pw_after == team else 0.0
            team_reward[team] = h_after[team] - h_before[team] + acquired
    return team_reward


def role_rewards(
    sim: CartSim,
    before,
    after,
    margin_weight: float = 0.1,
) -> np.ndarray:
    """Per-PLAYER role reward: the canonical team reward projected through team_of."""
    return team_role_rewards(sim, before, after, margin_weight)[
        np.asarray(sim.team_of, dtype=np.int64)
    ]


def collect_rollout(
    est: StrategyEstimator,
    sim: CartSim,
    n_steps: int,
    rng: np.random.Generator,
):
    cstate = sim.reset()
    w = np.zeros((sim.l, sim.M), dtype=np.float32)
    records = []
    for _ in range(n_steps):
        state = state_from_cartsim(sim, cstate, w=w)
        result = est.forward(state)
        action = np.asarray(result.action).astype(np.int64)
        w_next = np.asarray(result.w_next).astype(np.float32)
        next_cstate, _ = sim.step(cstate, action)
        records.append(
            {
                "state": cstate.copy(),
                "next_state": next_cstate.copy(),
                "w_in": w.copy(),
                "w_out": w_next.copy(),
                "action": action,
            }
        )
        cstate = next_cstate
        w = w_next
    return records


def eval_vs_reference(
    est: StrategyEstimator,
    shape,
    L,
    n_steps,
    n_games,
    rng,
    seed,
):
    k, j, l = shape
    wins = 0
    retained = 0
    retention_trials = 0
    acquired = 0
    acquisition_trials = 0
    for game in range(n_games):
        sim = CartSim(k, j, l, L=L, seed=seed + game)
        state = sim.reset()
        w = np.zeros((l, sim.M), dtype=np.float32)
        for _ in range(n_steps):
            pw_before = sim.projected_winner(state)
            features = state_from_cartsim(sim, state, w=w)
            key = mx.random.key(int(rng.integers(0, 2**31 - 1)))
            w_next, action, _, _, _, _ = policy_forward(
                est, features, w, key=key
            )
            action = np.asarray(action).astype(np.int64)
            reference = greedy_deny_policy(sim, state)
            for player in range(l):
                if int(sim.team_of[player]) != 0:
                    action[player] = reference[player]
            next_state, _ = sim.step(state, action)
            pw_after = sim.projected_winner(next_state)
            if pw_before == 0:
                retention_trials += 1
                retained += int(pw_after == 0)
            else:
                acquisition_trials += 1
                acquired += int(pw_after == 0)
            state = next_state
            w = np.asarray(w_next).astype(np.float32)
        wins += int(sim.projected_winner(state) == 0)
    return {
        "win_rate": wins / max(1, n_games),
        "retention": retained / max(1, retention_trials),
        "acquisition": acquired / max(1, acquisition_trials),
    }


def train(
    *,
    iters=500,
    batch=6,
    n_steps=16,
    k=2,
    j=3,
    l=4,
    L=6,
    gamma=0.95,
    lr=3e-3,
    c_w=0.5,
    c_l=0.5,
    c_dyn=0.25,
    c_explore=0.01,
    c_reg=1e-3,
    weight_decay=1e-4,
    delta=0.5,
    temperature=1.0,
    seed=0,
    eval_games=8,
    log_every=1,
    outdir=None,
    population_shapes=None,
    heldout_shapes=None,
):
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "train_log_v3.jsonl")
    ckpt_path = os.path.join(outdir, "policy_ckpt_v3.npz")
    population_shapes = population_shapes or [
        (k, j, l),
        (2, 2, 4),
        (2, 4, 6),
        (3, 3, 6),
        (3, 5, 8),
        (4, 4, 8),
    ]
    heldout_shapes = heldout_shapes or [(3, 4, 7), (4, 6, 9), (5, 7, 11)]
    population_shapes = list(dict.fromkeys(tuple(x) for x in population_shapes))
    heldout_shapes = list(dict.fromkeys(tuple(x) for x in heldout_shapes))
    rng = np.random.default_rng(seed)
    base_sim = CartSim(*population_shapes[0], L=L, seed=seed)
    est = StrategyEstimator.for_cartsim(
        base_sim,
        delta=delta,
        temperature=temperature,
        seed=seed,
    )
    dynamics = LocalDynamics()
    bundle = nn.Module()
    bundle.qkv = est.qkv
    bundle.encoder = est.encoder
    bundle.head = est.head
    bundle.value = est.value
    bundle.dynamics = dynamics
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def make_loss(rollouts):
        def loss_fn():
            pg_sum = mx.zeros(())
            w_sum = mx.zeros(())
            l_sum = mx.zeros(())
            dyn_sum = mx.zeros(())
            reg_sum = mx.zeros(())
            uncertainty_sum = mx.zeros(())
            winner_count = 0
            loser_count = 0
            row_count = 0
            for sim, records in rollouts:
                for record in records:
                    state = state_from_cartsim(
                        sim, record["state"], w=record["w_in"]
                    )
                    next_state = state_from_cartsim(
                        sim, record["next_state"], w=record["w_out"]
                    )
                    action = mx.array(record["action"])
                    w_next, _, logpi, value, winner_value, loser_value = policy_forward(
                        est, state, record["w_in"], action=action
                    )
                    _, _, _, next_value, _, _ = policy_forward(
                        est, next_state, record["w_out"], action=action
                    )
                    dyn_state = mx.stop_gradient(mx.array(state_rows(state)))
                    dyn_action = mx.stop_gradient(
                        mx.array(action_rows(state, record["action"]))
                    )
                    target_delta = mx.stop_gradient(
                        mx.array(next_state.hierarchy - state.hierarchy)
                    )
                    uncertainty = dynamics.uncertainty(dyn_state, dyn_action)
                    reward = mx.array(
                        role_rewards(sim, record["state"], record["next_state"])
                    ) + c_explore * mx.stop_gradient(uncertainty)
                    td_target = reward + gamma * mx.stop_gradient(next_value)
                    adv = td_target - value
                    pg_sum = pg_sum - mx.sum(mx.stop_gradient(adv) * logpi)
                    winner_mask = mx.array(state.winner_mask)
                    loser_mask = mx.logical_not(winner_mask)
                    w_sum = w_sum + mx.sum(
                        mx.square(winner_value - td_target)
                        * winner_mask.astype(mx.float32)
                    )
                    l_sum = l_sum + mx.sum(
                        mx.square(loser_value - td_target)
                        * loser_mask.astype(mx.float32)
                    )
                    dyn_sum = dyn_sum + dynamics_loss(
                        dynamics, dyn_state, dyn_action, target_delta
                    )
                    reg_sum = reg_sum + mx.mean(mx.square(w_next))
                    uncertainty_sum = uncertainty_sum + mx.sum(uncertainty)
                    winner_count += int(np.sum(state.winner_mask))
                    loser_count += int(np.sum(~state.winner_mask))
                    row_count += sim.l
            L_pg = pg_sum / max(1, row_count)
            L_w = w_sum / max(1, winner_count)
            L_l = l_sum / max(1, loser_count)
            L_dyn = dyn_sum / max(1, sum(len(records) for _, records in rollouts))
            L_reg = reg_sum / max(1, sum(len(records) for _, records in rollouts))
            mean_uncertainty = uncertainty_sum / max(1, row_count)
            total = (
                L_pg
                + c_w * L_w
                + c_l * L_l
                + c_dyn * L_dyn
                + c_reg * L_reg
            )
            return total, (L_pg, L_w, L_l, L_dyn, L_reg, mean_uncertainty)

        return loss_fn

    with open(log_path, "w") as stream:
        stream.write(
            json.dumps(
                {
                    "_config": {
                        "iters": iters,
                        "batch": batch,
                        "n_steps": n_steps,
                        "population_shapes": population_shapes,
                        "heldout_shapes": heldout_shapes,
                        "L": L,
                        "gamma": gamma,
                        "lr": lr,
                        "c_w": c_w,
                        "c_l": c_l,
                        "c_dyn": c_dyn,
                        "c_explore": c_explore,
                        "c_reg": c_reg,
                        "seed": seed,
                    }
                }
            )
            + "\n"
        )
    started = time.time()
    for iteration in range(iters):
        rollouts = []
        for item in range(batch):
            shape = population_shapes[int(rng.integers(0, len(population_shapes)))]
            sim = CartSim(*shape, L=L, seed=seed + iteration * batch + item)
            rollouts.append((sim, collect_rollout(est, sim, n_steps, rng)))
        loss_fn = make_loss(rollouts)
        (total, parts), grads = nn.value_and_grad(bundle, loss_fn)()
        optimizer.update(bundle, grads)
        mx.eval(bundle.parameters(), optimizer.state, total)
        values = [float(np.asarray(value)) for value in parts]
        known = eval_vs_reference(
            est,
            population_shapes[iteration % len(population_shapes)],
            L,
            n_steps,
            eval_games,
            rng,
            10_000 + iteration * 97,
        )
        heldout = eval_vs_reference(
            est,
            heldout_shapes[iteration % len(heldout_shapes)],
            L,
            n_steps,
            eval_games,
            rng,
            20_000 + iteration * 97,
        )
        sample_sim, sample_records = rollouts[0]
        sample_state = state_from_cartsim(
            sample_sim,
            sample_records[0]["state"],
            w=sample_records[0]["w_in"],
        )
        matrix = np.asarray(
            dynamics.local_matrix(mx.array(state_rows(sample_state)))
        )
        sigma_min = float(
            np.mean([np.linalg.svd(row, compute_uv=False).min() for row in matrix])
        )
        row = {
            "iter": iteration,
            "loss_total": float(np.asarray(total)),
            "loss_pg": values[0],
            "loss_w": values[1],
            "loss_l": values[2],
            "loss_dynamics": values[3],
            "loss_reg": values[4],
            "model_uncertainty": values[5],
            "local_control_sigma_min": sigma_min,
            "known": known,
            "heldout": heldout,
            "sec": round(time.time() - started, 2),
        }
        if iteration % log_every == 0 or iteration == iters - 1:
            with open(log_path, "a") as stream:
                stream.write(json.dumps(row) + "\n")
        if iteration % 25 == 0 or iteration == iters - 1:
            print(json.dumps(row), flush=True)
            flat = dict(tree_flatten(bundle.parameters()))
            np.savez(ckpt_path, **{name: np.asarray(value) for name, value in flat.items()})
    flat = dict(tree_flatten(bundle.parameters()))
    np.savez(ckpt_path, **{name: np.asarray(value) for name, value in flat.items()})
    return log_path, ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--n_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_games", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    train(
        iters=args.iters,
        batch=args.batch,
        n_steps=args.n_steps,
        seed=args.seed,
        eval_games=args.eval_games,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
