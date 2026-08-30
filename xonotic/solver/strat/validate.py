"""Runtime smoke test wiring the whole strategy stack together on small synthetic sizes.

This is an end-to-end *plumbing* check for the layer-3 strategy core: it instantiates the
committed modules on toy sizes (k=5 teams, j=4 objectives/instruments, l=16 players) and
asserts the load-bearing invariants of `design/rl-training-spec.md`,
`design/payload-spec.md`, and `design/dpp-mixing-and-overlay.md`. It is NOT a numerical
accuracy test -- it verifies shapes, ranges, determinism, and the computed-vs-learned-vs-
frozen gradient discipline hold when the real modules are wired in the real order:

    game (PW/SUCC)  ->  featurize (belief + assemble s,SUCC)  ->  dpp (diag K)
                                                                        |
                                              head (RMSNorm->SwiGLU) -> per-player velocity
                                                                        |
                                              value (V_phi in R^l)  ->  advantage A
                                                                        |
                                              buffers (observe / replay)  ->  one PG step

Two games / one policy (`rl-training-spec.md` §0): PW/SUCC are COMPUTED deterministically
over cartstate (Game-1, numpy, `game.py`); the frozen FPS C-program (Game-2) is not
touched here. The single shared-weight policy is the mixing head's ``W_all``
(`head.py`); per-team/per-player distinctions are ACTIVATIONS (the per-player ``b`` and
``diag(K)`` rows), never separate weights. The value head grounds the potential-based
advantage ``A = R + gamma*V(s') - V(s)`` (`value.py`), and the replay buffer holds the
``(state, activations, action, logpi)`` transitions the REINFORCE gradient replays
(`buffers.py`).

Assertions (each prints PASS/FAIL; the process exits nonzero if any fail):

1.  FORWARD/SHAPE   -- the head emits a per-player velocity of shape ``(l, j)``.
2.  DIAG_K_RANGE    -- ``diag(K)`` is in ``[0, 1]`` per instrument (marginal inclusion).
3.  PW_SUCC_DETERM  -- PW/SUCC are deterministic (same input -> identical output).
4.  PW_NIMSUM       -- PW matches the nim-sum on the hand-checked case: one cart at d:2
                       beats two carts at d:1 (``1 XOR 1 = 0``).
5.  VALUE_SHAPE     -- the value head returns a per-player vector in ``R^l``.
6.  ADVANTAGE_FINITE-- ``A = R + gamma*V(s') - V(s)`` is finite and in ``R^l``.
7.  PG_UPDATES_WALL -- one policy-gradient step updates ONLY ``W_all`` (the head); the
                       computed features ``s, b, diag(K), PW, SUCC`` carry NO gradient.

Learned/differentiable parts (dpp, head, value, sampling) use mlx (Apple, matches the
mini); deterministic parts (PW, SUCC, V-cell featurization) use numpy/plain python -- the
same computed/learned split the specs require. Run:

    uv run --with mlx --with numpy python3 -m xonotic.solver.strat.validate

Spec: `rl-training-spec.md` §0-§4 (two games, computed vs learned, reward/value/advantage/
      policy-gradient); `payload-spec.md` §2.2-§2.5 (belief, mix); `dpp-mixing-and-overlay.md`
      §2 (diag(K) + RMSNorm->SwiGLU head).

Public surface
--------------
- ``build_synthetic_world`` : construct the toy (carts, VCellMap, observations) inputs.
- ``run_forward``           : wire game->featurize->dpp->head->value into one forward pass.
- ``CHECKS``                : the ordered list of ``(name, fn)`` assertion callables.
- ``main``                  : run every check, print PASS/FAIL, return an exit code.
"""

from __future__ import annotations

import sys

import numpy as np
import mlx.core as mx

from . import game as G
from . import dpp as D
from . import head as H
from . import value as V
from . import featurize as F
from . import buffers as B


# --------------------------------------------------------------------------- #
# Toy sizes (the smoke-test problem).
# --------------------------------------------------------------------------- #
K_TEAMS = 5        # k -- number of teams (activations A_team, not separate weights).
J_OBJ = 4          # j -- number of objectives / instruments (the head's M / output width).
L_PLAYERS = 16     # l -- number of players (per-player VECTOR reward/value: R, V in R^l).
GAMMA = 0.95       # discount for the potential-based advantage (rl-training-spec §2.1).
DELTA = 0.1        # forward-Euler cadence for the replicator step (head.integrate_weights).
SEED = 20260829


def build_synthetic_world(seed: int = SEED):
    """Construct the toy world inputs for the wiring test (deterministic under ``seed``).

    Returns a dict with:
      * ``carts``        : the Game-1 cartstate -- the hand-checked position, one team ("A")
                           holding a single cart at depth 2 and one team ("B") holding two
                           carts at depth 1 (``1 XOR 1 = 0``; PW must be "A"), padded with
                           further teams so the roster has ``k`` teams.
      * ``teams``        : the explicit ``k``-team roster.
      * ``vcmap``        : a stage-2 :class:`featurize.VCellMap` over synthetic node
                           positions (so the belief pipeline is exercised for real).
      * ``rows``         : synthetic observation rows for the belief ingest.
      * ``bot_positions``: ``l`` player world positions (for per-player belief).
      * ``quality``      : per-instrument appetite ``q`` (length ``j``) for the DPP.
      * ``inst_features``: per-instrument diversity keys ``(j, F)`` for the DPP Gram.
      * ``b_appetite``   : per-player appetite ``(l, j)`` -- the head's ``b`` activations.
    """
    rng = np.random.RandomState(seed)

    # Hand-checked Nim position: A = {d:2}, B = {d:1, d:1}; extra teams get shallow carts.
    carts = [("A", 2), ("B", 1), ("B", 1), ("C", 1), ("D", 1), ("E", 1)]
    teams = ["A", "B", "C", "D", "E"][:K_TEAMS]

    # Stage-2 segmented map over synthetic item/waypoint node positions.
    node_positions = rng.rand(40, 2) * 100.0
    vcmap = F.segment_vcells(node_positions)

    # A few gated observation rows deposited at V-cells (payload-spec §2.2.1 ingest).
    rows = [
        {"cell": 0, "item_type": 1.0, "observed_enemy": 1.0, "time": 1.0},
        {"cell": 1, "standability": 1.0, "time": 1.0},
        {"cell": min(2, vcmap.n_cells - 1), "last_threat": 1.0, "time": 0.5},
    ]

    # l player positions (each resolves to a V-cell for its egocentric belief).
    bot_positions = rng.rand(L_PLAYERS, 2) * 100.0

    # DPP inputs: per-instrument appetite q (>=0) and per-instrument diversity keys.
    quality = np.abs(rng.rand(J_OBJ)).astype(np.float32)
    inst_features = rng.rand(J_OBJ, 6).astype(np.float32)

    # Per-player appetite b (the head's per-player activations A_player).
    b_appetite = rng.rand(L_PLAYERS, J_OBJ).astype(np.float32)

    return dict(
        carts=carts, teams=teams, vcmap=vcmap, rows=rows,
        bot_positions=bot_positions, quality=quality,
        inst_features=inst_features, b_appetite=b_appetite,
    )


def run_forward(world: dict):
    """Wire game -> featurize -> dpp -> head -> value into one forward pass.

    Returns a dict of the intermediate/terminal tensors the checks assert over:
      * ``pw``            : PW(s) (the projected-winner team id or None).
      * ``succ``          : SUCC(s) (ordered ``[(team, marginal_denial_value)]``).
      * ``global_feats``  : the assembled global feature vector (s + SUCC guaranteed in it),
                            used here as the value head's "final intermediate" proxy.
      * ``diag_k``        : the DPP marginal-inclusion vector ``diag(K)`` (length ``j``).
      * ``velocity``      : the per-player strategy velocity ``dw/dt`` ``(l, j)``.
      * ``value``         : ``V_phi`` per-player value vector in ``R^l``.
      * ``head`` / ``value_head`` : the instantiated mlx modules (the learned ``W_all``
                            surface + the value head).
    """
    carts, teams = world["carts"], world["teams"]

    # --- Game-1: computed PW / SUCC over cartstate (numpy, deterministic, stopgrad). ---
    pw = G.projected_winner(carts, teams=teams)
    succ = G.succession(carts, teams=teams)

    # --- Featurize: per-player belief + the assembled global feature vector. ---
    # beliefs_for_bots computes the ingest+contraction ONCE and each bot is a masked sum.
    betas = F.beliefs_for_bots(world["rows"], world["vcmap"], world["bot_positions"],
                               now=1.0, T=2.0)                     # (l, rank)
    # One assembled global feature vector (s + succ guaranteed members) -> value input.
    # Use player 0's belief as the exemplar egocentric context for the shared intermediate.
    gf = F.assemble_features(x_b=np.ones(3, dtype=np.float32), beta_b=betas[0],
                             cartstate=carts, teams=teams, succ=succ)
    global_feats = gf.vector

    # --- DPP: instrument appetite + diversity keys -> diag(K) marginal inclusion. ---
    diag_k = D.dpp_marginals(mx.array(world["quality"]), mx.array(world["inst_features"]))

    # --- Head: RMSNorm -> SwiGLU, [diag(K); b] -> per-player velocity dw/dt. ---
    head = H.MixingHead(n_instruments=J_OBJ)
    b = mx.array(world["b_appetite"])                              # (l, j)
    velocity = head(mx.stop_gradient(diag_k), mx.stop_gradient(b)) # (l, j)

    # --- Value: linear projection on the final intermediate -> per-player V in R^l. ---
    value_head = V.ValueHead(d_intermediate=int(global_feats.shape[0]), l=L_PLAYERS)
    value = value_head(mx.array(global_feats))                    # (l,)

    return dict(pw=pw, succ=succ, global_feats=global_feats, diag_k=diag_k,
                velocity=velocity, value=value, head=head, value_head=value_head,
                betas=betas, b=b)


# --------------------------------------------------------------------------- #
# The checks. Each returns (ok: bool, detail: str).
# --------------------------------------------------------------------------- #

def check_forward_shape(world, fwd):
    """The head emits a per-player strategy velocity of shape ``(l, j)`` (payload-spec §2.5)."""
    v = fwd["velocity"]
    ok = tuple(v.shape) == (L_PLAYERS, J_OBJ) and bool(mx.all(mx.isfinite(v)).item())
    return ok, f"velocity shape={tuple(v.shape)} expected=({L_PLAYERS}, {J_OBJ}), finite={ok}"


def check_diag_k_range(world, fwd):
    """``diag(K)`` is a marginal-inclusion vector in ``[0, 1]`` per instrument (dpp §2)."""
    dk = np.array(fwd["diag_k"])
    ok = dk.shape == (J_OBJ,) and bool(np.all(dk >= -1e-6)) and bool(np.all(dk <= 1.0 + 1e-6))
    return ok, f"diag(K)={np.round(dk, 4).tolist()} in [0,1]={ok}"


def check_pw_succ_deterministic(world, fwd):
    """PW/SUCC are DETERMINISTIC: same cartstate -> identical PW and SUCC (rl-spec §1)."""
    carts, teams = world["carts"], world["teams"]
    pw1 = G.projected_winner(carts, teams=teams)
    pw2 = G.projected_winner(carts, teams=teams)
    s1 = G.succession(carts, teams=teams)
    s2 = G.succession(carts, teams=teams)
    # also confirm the numpy succ feature is byte-for-byte stable
    f1 = G.succ_feature(carts, teams=teams)
    f2 = G.succ_feature(carts, teams=teams)
    ok = (pw1 == pw2) and (s1 == s2) and np.array_equal(f1, f2)
    return ok, f"PW {pw1}=={pw2}, SUCC {s1}=={s2}, succ_feature identical={np.array_equal(f1, f2)}"


def check_pw_nimsum(world, fwd):
    """PW matches the nim-sum hand check: one cart at d:2 beats two carts at d:1 (rl-spec §1).

    Team A holds {d:2} (nimber 2); team B holds {d:1, d:1} (``1 XOR 1 = 0``, no live
    threat). PW must be "A". Also spot-check the nim_sum primitive directly.
    """
    pw = fwd["pw"]
    nimbers = G.team_nimbers(world["carts"], teams=world["teams"])
    prim_ok = (G.nim_sum([1, 1]) == 0) and (G.nim_sum([2]) == 2)
    ok = (pw == "A") and (nimbers.get("A") == 2) and (nimbers.get("B") == 0) and prim_ok
    return ok, (f"PW={pw} (expect A); nimbers A={nimbers.get('A')} B={nimbers.get('B')}; "
                f"nim_sum([1,1])=0 & nim_sum([2])=2 -> {prim_ok}")


def check_value_shape(world, fwd):
    """The value head returns a per-player VECTOR value in ``R^l`` -- never scalar (rl-spec §2.1)."""
    val = fwd["value"]
    ok = tuple(val.shape) == (L_PLAYERS,) and bool(mx.all(mx.isfinite(val)).item())
    return ok, f"V shape={tuple(val.shape)} expected=({L_PLAYERS},), finite={ok}"


def check_advantage_finite(world, fwd):
    """``A = R + gamma*V(s') - V(s)`` is finite and in ``R^l`` (potential-based; rl-spec §2.1).

    Builds a synthetic per-player reward ``R in R^l`` and a next-state value ``V(s')`` from a
    perturbed intermediate, then forms the advantage with ``value.advantage``.
    """
    rng = np.random.RandomState(SEED + 1)
    v_s = fwd["value"]
    reward = mx.array(rng.randn(L_PLAYERS).astype(np.float32))
    # V(s') from a perturbed "next" intermediate through the same value head.
    next_feats = mx.array(fwd["global_feats"]) + mx.array(
        rng.randn(*fwd["global_feats"].shape).astype(np.float32) * 0.1)
    v_next = fwd["value_head"](next_feats)
    A = V.advantage(reward, v_s, v_next, gamma=GAMMA)
    ok = tuple(A.shape) == (L_PLAYERS,) and bool(mx.all(mx.isfinite(A)).item())
    return ok, f"A shape={tuple(A.shape)}, all finite={bool(mx.all(mx.isfinite(A)).item())}"


def check_pg_updates_only_wall(world, fwd):
    """One policy-gradient step updates ONLY ``W_all``; features/PW/SUCC carry NO gradient.

    Reproduces the REINFORCE step of `rl-training-spec.md` §2.1 / §3:
      * the actor logits are the head's per-player velocity over instruments (integrated one
        forward-Euler step from a zero weight state);
      * an action is SAMPLED (payload-spec §5, not argmax) as behavior;
      * the per-player advantage ``A`` weights ``log pi(a|.)`` in ``L_pg = -E[A.detach() * logpi]``;
      * the gradient is taken w.r.t. the head parameters ONLY (``W_all``).

    Asserts three things:
      (a) the gradient set is exactly the head's parameters (``W_all``) and at least one is
          nonzero -- so ``W_all`` genuinely updates;
      (b) applying the step changes the head weights;
      (c) the COMPUTED features ``diag(K)`` and ``b`` carry NO gradient -- differentiating
          the same loss w.r.t. the raw feature arrays yields exactly zero (they enter via
          ``stop_gradient``), and ``SUCC``'s feature is a numpy array with no autograd at all.
    """
    head = fwd["head"]

    diag_k = fwd["diag_k"]                 # computed feature (dpp, mlx but detached below)
    b = fwd["b"]                           # per-player appetite activation
    # per-player advantage weights (detached into the actor per §3 L_pg = -E[A.detach()*logpi])
    rng = np.random.RandomState(SEED + 2)
    adv = mx.array(rng.randn(L_PLAYERS).astype(np.float32))

    # Behavior: sample one instrument per player from the current (detached) logits.
    dk_sg, b_sg = mx.stop_gradient(diag_k), mx.stop_gradient(b)
    behavior_logits = H.integrate_weights(mx.zeros((L_PLAYERS, J_OBJ)), head(dk_sg, b_sg), DELTA)
    action, _ = H.sample_strategy(behavior_logits, key=mx.random.key(SEED + 3))
    action = mx.stop_gradient(action)

    def pg_loss(model):
        # features enter DETACHED (stopgrad on diag(K), b, PW, SUCC per rl-spec §2.1)
        logits = H.integrate_weights(mx.zeros((L_PLAYERS, J_OBJ)), model(dk_sg, b_sg), DELTA)
        logpi = H.strategy_log_prob(logits, action)          # (l,)
        return -mx.mean(mx.stop_gradient(adv) * logpi)        # L_pg

    import mlx.nn as nn
    loss, grads = nn.value_and_grad(head, pg_loss)(head)

    # (a) the grad tree is exactly W_all (the head params), with at least one nonzero grad.
    grad_leaves = _leaves(grads)
    param_leaves = _leaves(head.parameters())
    any_nonzero = any(bool(mx.sum(mx.abs(g)).item() > 0.0) for g in grad_leaves)
    same_structure = len(grad_leaves) == len(param_leaves) and len(grad_leaves) > 0

    # (b) applying the step actually moves W_all.
    before = [np.array(p) for p in param_leaves]
    import mlx.optimizers as optim
    opt = optim.SGD(learning_rate=0.1)
    opt.update(head, grads)
    mx.eval(head.parameters())
    after = _leaves(head.parameters())
    moved = any(not np.allclose(a0, np.array(a1)) for a0, a1 in zip(before, after))

    # (c) the computed features carry NO gradient. Differentiate the SAME loss w.r.t. the
    #     raw (pre-stopgrad) feature arrays: because they enter via stop_gradient the grad
    #     is exactly zero. SUCC's feature is numpy (no autograd surface at all).
    def loss_wrt_features(raw_dk, raw_b):
        logits = H.integrate_weights(
            mx.zeros((L_PLAYERS, J_OBJ)),
            head(mx.stop_gradient(raw_dk), mx.stop_gradient(raw_b)), DELTA)
        logpi = H.strategy_log_prob(logits, action)
        return -mx.mean(mx.stop_gradient(adv) * logpi)

    g_dk, g_b = mx.grad(loss_wrt_features, argnums=(0, 1))(diag_k, b)
    feats_no_grad = (float(mx.sum(mx.abs(g_dk)).item()) == 0.0
                     and float(mx.sum(mx.abs(g_b)).item()) == 0.0)
    succ_is_numpy = isinstance(G.succ_feature(world["carts"], teams=world["teams"]), np.ndarray)

    ok = same_structure and any_nonzero and moved and feats_no_grad and succ_is_numpy
    detail = (f"W_all grad-tree matches params={same_structure}, nonzero grad={any_nonzero}, "
              f"weights moved={moved}; feature grads zero (stopgrad)={feats_no_grad}, "
              f"SUCC feature numpy(no-autograd)={succ_is_numpy}")
    return ok, detail


def _leaves(tree):
    """Flatten an mlx parameter/gradient pytree (nested dict/list) to a list of arrays."""
    out = []
    if isinstance(tree, dict):
        for v in tree.values():
            out.extend(_leaves(v))
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            out.extend(_leaves(v))
    elif isinstance(tree, mx.array):
        out.append(tree)
    return out


CHECKS = [
    ("FORWARD_SHAPE", check_forward_shape),
    ("DIAG_K_RANGE", check_diag_k_range),
    ("PW_SUCC_DETERMINISTIC", check_pw_succ_deterministic),
    ("PW_NIMSUM_1@d2>2@d1", check_pw_nimsum),
    ("VALUE_SHAPE_R^l", check_value_shape),
    ("ADVANTAGE_FINITE", check_advantage_finite),
    ("PG_UPDATES_ONLY_W_all", check_pg_updates_only_wall),
]


def main() -> int:
    """Run every check, print PASS/FAIL per assertion, return an exit code (0=all pass)."""
    mx.random.seed(SEED)
    world = build_synthetic_world()
    fwd = run_forward(world)

    print("=" * 72)
    print(f"strat.validate  --  k={K_TEAMS} teams, j={J_OBJ} objectives, l={L_PLAYERS} players")
    print("=" * 72)

    failures = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(world, fwd)
        except Exception as exc:  # a raised exception is a failed check, not a crash.
            ok, detail = False, f"EXCEPTION {type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name:<26} {detail}")

    print("-" * 72)
    total = len(CHECKS)
    print(f"{total - failures}/{total} checks passed"
          + ("" if failures == 0 else f"  ({failures} FAILED)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
