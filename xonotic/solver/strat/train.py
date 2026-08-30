"""Real RLVR training loop for the shared-weight strategy policy (rl-training-spec §2-3).

This is the on-policy REINFORCE loop the spec's §6 marks ``[BUILD]`` ("the rollout loop,
replay buffer, reward/value/advantage/policy-gradient -- none exist as code"). It trains
``W_all`` (the qkv projections + mixing head) plus the value stack against the deterministic
Game-1 self-play simulator :mod:`cartsim`, exactly per the four definitions:

  * ROLLOUT      -- on-policy self-play on :class:`cartsim.CartSim`; the integrated
                    strategy weights ``w`` flow across steps (forward-Euler replicator).
  * REWARD       -- RELATIVE, verifiable, terminal + zero-mean across teams (§2.1/§5):
                    at the terminal cartstate the projected-winner ``PW`` team scores +1,
                    centered by the mean over teams so it is purely relative (deny+acquire
                    the path-to-victory slot), NOT monotone progress / entry / level.
  * VALUE        -- per-player VECTOR ``V_phi in R^l`` off the final intermediate with
                    ``SUCC`` folded in (anticipatory); regressed to the discounted return.
  * ADVANTAGE    -- potential-based one-step TD ``A = r + gamma*V(s') - V(s)`` (§2.1);
                    the player's own component weights its own log-prob (REINFORCE).

Loss (§3): ``L = L_pg + c_v*L_v + c_aux*L_aux + c_reg*L_reg`` with ``L_pg`` using
``A.detach()``, ``L_reg`` an L2-toward-0 on the strategy logits, and weight decay via AdamW.
Only ``W_all`` + value/aux heads learn; ``s,b,PW,SUCC`` and the FPS are stopgrad (§2.1/§4),
which the estimator already enforces with ``stop_gradient`` on every feature.

Movement metric (honest, un-fabricated): every iteration a small EVAL pits the learned
team 0 against the fixed ``greedy_deny`` reference on the other teams; ``win_rate`` =
P(PW_terminal == team 0), ``pw_control_frac`` = fraction of eval steps team 0 holds PW.
A learning policy should pull both above the reference-vs-reference chance baseline.

Run (on the mlx host -- the Mac mini, ~/.venv-mesh):
    python -m solver.strat.train --iters 500
Logs one JSON line per iteration to runs/train_log.jsonl and checkpoints W_all to
runs/policy_ckpt.npz.

Spec: rl-training-spec.md §2-§5 ; payload-spec §2.3-§2.5,§3.2,§5 ; siblings cartsim /
      estimator / head / value / qkv / game.
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

from .cartsim import CartSim, greedy_deny_policy
from .dpp import dpp_marginals
from .estimator import StrategyEstimator, state_from_cartsim
from .head import integrate_weights, strategy_log_prob
from .game import succ_feature


# --------------------------------------------------------------------------- #
# The differentiable policy+value forward, factored so the SAME code path both
# samples (rollout) and replays a stored action (loss). Reads the estimator's
# learned sub-modules directly; every feature enters detached (stop_gradient).
# --------------------------------------------------------------------------- #

def policy_forward(est: StrategyEstimator, state, w_in: np.ndarray,
                   action: Optional[mx.array] = None, key: Optional[mx.array] = None):
    """One strategy step through W_all + value. Returns (w_next, action, logpi, V, Vtilde).

    ``w_in`` (l,M) is the current integrated strategy-weight state (carried across the
    rollout). If ``action`` is None the instrument is categorically SAMPLED (rollout,
    payload-spec §5); otherwise ``action`` is replayed and its log-prob recomputed under
    gradient (REINFORCE replay, rl-training-spec §2.1). All features are stopgrad.
    """
    l, M = est.l, est.M
    x = mx.stop_gradient(mx.array(np.asarray(state.x, dtype=np.float32)))
    beta = mx.stop_gradient(mx.array(np.asarray(state.beta, dtype=np.float32)))
    z = mx.stop_gradient(mx.array(np.asarray(state.z, dtype=np.float32)))
    w = mx.stop_gradient(mx.array(np.asarray(w_in, dtype=np.float32)))

    Q = est.qkv.query(x, beta)                 # (l,d) = A_player
    K = est.qkv.key(z)                         # (M,d)
    scores = Q @ K.T                           # (l,M)
    appetite = mx.logaddexp(scores, mx.zeros_like(scores))   # softplus >= 0
    quality = mx.mean(appetite, axis=0)        # (M,)
    diag_k = mx.stop_gradient(dpp_marginals(quality, K))     # (M,) detached feature
    dw_dt = est.head(diag_k, appetite)         # (l,M)
    w_next = integrate_weights(w, dw_dt, est.delta)          # (l,M)

    if action is None:
        action = mx.random.categorical(w_next / est.temperature, axis=-1, key=key)
    logpi = strategy_log_prob(w_next, action, temperature=est.temperature)  # (l,)

    succ_np = succ_feature(state.carts, teams=state.teams).astype(np.float32)
    succ = mx.stop_gradient(mx.array(succ_np))
    succ_rows = mx.broadcast_to(succ[None, :], (l, est.succ_dim))
    final_intermediate = mx.concatenate([Q, appetite, dw_dt, succ_rows], axis=1)
    V, Vtilde = est.value(final_intermediate, Q)             # (l,l),(l,l)
    return w_next, action, logpi, V, Vtilde


# --------------------------------------------------------------------------- #
# Reward: RELATIVE, terminal, zero-mean across teams (rl-training-spec §2.1/§5).
# --------------------------------------------------------------------------- #

def terminal_reward_vector(sim: CartSim, term_state) -> np.ndarray:
    """Per-player relative terminal reward ``g in R^l`` (deny+acquire the PW slot).

    The projected-winner team at the terminal cartstate scores 1, every team's raw score
    is then centered by the mean over teams so the signal is purely RELATIVE (mean 0 over
    the population) -- it rewards HOLDING the path-to-victory relative to rivals, never
    absolute progress / entry / level (§5). Returns length-l (player-indexed via team_of).
    """
    pw = sim.projected_winner(term_state)
    base = np.zeros(sim.k, dtype=np.float32)
    if pw is not None:
        base[pw] = 1.0
    base = base - base.mean()                  # relative: zero-mean across teams
    return np.array([base[int(sim.team_of[p])] for p in range(sim.l)], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Rollout: on-policy self-play; carry w across steps; record replay records.
# --------------------------------------------------------------------------- #

def collect_rollout(est: StrategyEstimator, sim: CartSim, n_steps: int, rng: np.random.Generator):
    """One self-play rollout. Returns (records, term_state).

    Each record = dict(cstate, w_in, action(np,l)) -- enough to rebuild the StrategyState
    and replay the action under gradient. ``w`` (integrated logits) flows step to step.
    """
    cstate = sim.reset()
    w = np.zeros((sim.l, sim.M), dtype=np.float32)
    records = []
    for t in range(n_steps):
        state = state_from_cartsim(sim, cstate, w=w)
        key = mx.random.key(int(rng.integers(0, 2**31 - 1)))
        w_next, action, _lp, _V, _Vt = policy_forward(est, state, w, action=None, key=key)
        act_np = np.asarray(action).astype(np.int64)
        records.append({"cstate": cstate.copy(), "w_in": w.copy(), "action": act_np})
        cstate, _info = sim.step(cstate, act_np)
        w = np.asarray(w_next).astype(np.float32)   # carry integrated weights
    return records, cstate


# --------------------------------------------------------------------------- #
# Eval: learned team 0 vs fixed greedy_deny reference on other teams.
# --------------------------------------------------------------------------- #

def eval_vs_reference(est: StrategyEstimator, sim_seed: int, k, j, l, L, n_steps, n_games, rng):
    """win_rate = P(PW_terminal==team0); pw_control_frac = mean fraction of steps PW==team0."""
    wins = 0
    pw_ctrl = 0.0
    total_steps = 0
    for g in range(n_games):
        sim = CartSim(k, j, l, L=L, seed=sim_seed + g)
        cstate = sim.reset()
        w = np.zeros((sim.l, sim.M), dtype=np.float32)
        for t in range(n_steps):
            state = state_from_cartsim(sim, cstate, w=w)
            key = mx.random.key(int(rng.integers(0, 2**31 - 1)))
            w_next, action, _lp, _V, _Vt = policy_forward(est, state, w, action=None, key=key)
            act = np.asarray(action).astype(np.int64)
            ref = greedy_deny_policy(sim, cstate)          # full (l,) reference actions
            for p in range(sim.l):
                if int(sim.team_of[p]) != 0:               # non-team-0 plays reference
                    act[p] = ref[p]
            pw = sim.projected_winner(cstate)
            if pw == 0:
                pw_ctrl += 1.0
            total_steps += 1
            cstate, _info = sim.step(cstate, act)
            w = np.asarray(w_next).astype(np.float32)
        if sim.projected_winner(cstate) == 0:
            wins += 1
    return wins / max(1, n_games), pw_ctrl / max(1, total_steps)


# --------------------------------------------------------------------------- #
# The training driver.
# --------------------------------------------------------------------------- #

def train(
    *, iters=500, batch=6, n_steps=16, k=2, j=3, l=4, L=6,
    gamma=0.95, lr=3e-3, c_v=0.5, c_aux=0.25, c_reg=1e-3, weight_decay=1e-4,
    delta=0.5, temperature=1.0, seed=0, eval_games=8, log_every=1,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "train_log.jsonl")
    ckpt_path = os.path.join(outdir, "policy_ckpt.npz")

    rng = np.random.default_rng(seed)
    train_sim = CartSim(k, j, l, L=L, seed=seed)
    est = StrategyEstimator.for_cartsim(train_sim, delta=delta, temperature=temperature, seed=seed)

    # Wrap the three learned modules so an mlx optimizer + nn.value_and_grad can drive them.
    bundle = nn.Module()
    bundle.qkv = est.qkv
    bundle.head = est.head
    bundle.value = est.value
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def make_loss(batch_rollouts):
        """Closure: L over a batch of pre-collected rollouts (records + terminal reward)."""
        def loss_fn():
            l_pg = mx.zeros(())
            l_v = mx.zeros(())
            l_aux = mx.zeros(())
            l_reg = mx.zeros(())
            adv_accum = mx.zeros(())
            n_sp = 0  # step-player count
            for (records, g_vec) in batch_rollouts:
                T = len(records)
                g_rows = mx.array(np.tile(g_vec[None, :], (l, 1)))   # (l,l) terminal reward rows
                # forward every step (differentiable); collect V, logpi, w_next
                Vs, logpis, wnexts = [], [], []
                for rec in records:
                    state = state_from_cartsim(train_sim, rec["cstate"], w=rec["w_in"])
                    act = mx.array(rec["action"])
                    w_next, _a, logpi, V, Vtilde = policy_forward(est, state, rec["w_in"], action=act)
                    Vs.append(V); logpis.append(logpi); wnexts.append(w_next)
                    l_aux = l_aux + mx.mean(mx.sum(mx.square(Vtilde - mx.stop_gradient(V)), axis=-1))
                    l_reg = l_reg + mx.mean(mx.square(w_next))
                # returns (MC, discounted terminal) + one-step TD advantage
                for t in range(T):
                    ret_t = (gamma ** (T - 1 - t)) * g_rows            # (l,l) return target
                    l_v = l_v + mx.mean(mx.sum(mx.square(Vs[t] - ret_t), axis=-1))
                    v_next = Vs[t + 1] if t < T - 1 else mx.zeros((l, l))
                    r_t = g_rows if t == T - 1 else mx.zeros((l, l))
                    A_t = r_t + gamma * v_next - Vs[t]                 # (l,l) potential-based
                    a_own = mx.diagonal(A_t)                           # (l,) player's own component
                    l_pg = l_pg - mx.sum(mx.stop_gradient(a_own) * logpis[t])
                    adv_accum = adv_accum + mx.sum(a_own)
                    n_sp += l
            n = 1.0 / max(1, n_sp)
            L_pg = l_pg * n
            L_v = l_v * n
            L_aux = l_aux * n
            L_reg = l_reg * n
            total = L_pg + c_v * L_v + c_aux * L_aux + c_reg * L_reg
            return total, (L_pg, L_v, L_aux, L_reg, adv_accum * n)
        return loss_fn

    # write header line noting config (real run provenance)
    with open(log_path, "w") as f:
        f.write(json.dumps({
            "_config": dict(iters=iters, batch=batch, n_steps=n_steps, k=k, j=j, l=l, L=L,
                            gamma=gamma, lr=lr, c_v=c_v, c_aux=c_aux, c_reg=c_reg,
                            weight_decay=weight_decay, delta=delta, temperature=temperature,
                            seed=seed, eval_games=eval_games)
        }) + "\n")

    t0 = time.time()
    for it in range(iters):
        # --- collect a batch of on-policy rollouts (sampling, no grad) ---
        batch_rollouts = []
        for b in range(batch):
            records, term = collect_rollout(est, train_sim, n_steps, rng)
            g_vec = terminal_reward_vector(train_sim, term)
            batch_rollouts.append((records, g_vec))

        # --- one policy-gradient + value-regression step ---
        loss_fn = make_loss(batch_rollouts)
        lg = nn.value_and_grad(bundle, loss_fn)
        (total, parts), grads = lg()
        optimizer.update(bundle, grads)
        mx.eval(bundle.parameters(), optimizer.state, total)
        L_pg, L_v, L_aux, L_reg, mean_adv = [float(np.asarray(x)) for x in (parts[0], parts[1], parts[2], parts[3], parts[4])]

        # --- honest movement metric: learned team0 vs greedy reference ---
        win_rate, pw_ctrl = eval_vs_reference(est, sim_seed=10_000 + it * 97,
                                              k=k, j=j, l=l, L=L, n_steps=n_steps,
                                              n_games=eval_games, rng=rng)

        row = {
            "iter": it,
            "loss_total": float(np.asarray(total)),
            "loss_pg": L_pg,
            "loss_v": L_v,
            "loss_aux": L_aux,
            "loss_reg": L_reg,
            "mean_advantage": mean_adv,
            "win_rate": win_rate,
            "pw_control_frac": pw_ctrl,
            "sec": round(time.time() - t0, 2),
        }
        if it % log_every == 0 or it == iters - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        if it % 25 == 0 or it == iters - 1:
            print(f"[{it:4d}] Lpg={L_pg:+.4f} Lv={L_v:.4f} adv={mean_adv:+.4f} "
                  f"win={win_rate:.3f} pwctrl={pw_ctrl:.3f} ({row['sec']}s)", flush=True)
            # checkpoint W_all + value stack
            flat = dict(tree_flatten(bundle.parameters()))
            np.savez(ckpt_path, **{k2: np.asarray(v) for k2, v in flat.items()})

    # final checkpoint
    flat = dict(tree_flatten(bundle.parameters()))
    np.savez(ckpt_path, **{k2: np.asarray(v) for k2, v in flat.items()})
    print(f"done: {iters} iters in {time.time()-t0:.1f}s -> {log_path}", flush=True)
    return log_path, ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--n_steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_games", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    train(iters=args.iters, batch=args.batch, n_steps=args.n_steps,
          seed=args.seed, eval_games=args.eval_games, lr=args.lr)


if __name__ == "__main__":
    main()
