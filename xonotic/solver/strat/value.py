"""The VALUE head V_phi and the auxiliary query-projection Vtilde.

The learned value stack of `design/rl-training-spec.md` §2 (definitions §2.1, normalized
register §2.2, loss §3). Per-player VECTOR value -- never scalar -- grounded in the
relative terminal outcome so the potential-based advantage can telescope cycles to zero
(§2.1 ADVANTAGE, §5): the value head exists *because* ``A = R + gamma*V(s') - V(s)`` is
policy-invariant only when ``V`` is a real learned potential.

Two projections, both linear (rl-training-spec §2.1 / §2.2 "value"):

- ``ValueHead`` = **V_phi(s, b, SUCC) in R^l** -- a linear projection on the strategy
  estimator's **final intermediate**, emitting the per-player value VECTOR in R^l. Its
  inputs include ``SUCC`` (folded into the intermediate upstream via
  ``game.succ_feature``), which is what makes ``V`` **anticipatory** (§2.2: "inputs
  superset ``SUCC`` => anticipatory").

- ``AuxValueHead`` = **Vtilde** -- a linear projection on the **query** ``q_b``,
  regressed toward ``V_phi`` by imitation (§2.1: "Auxiliary ``Vtilde`` = linear
  projection on the query, regressed to ``V``"; §3: ``L_aux = || Vtilde - V_phi.detach() ||^2``).
  The query already carries known self-state and observed world (`payload-spec.md` §2.3
  ``q_b = W_q [x_b ; beta_b]``), so ``Vtilde`` is a cheap value estimate available from
  the query alone, taught to track the full-intermediate value.

Both are the learned surface: only these projections (part of ``W_all`` + the value
head) carry the gradient. The features they read -- the final intermediate, the query,
and the ``SUCC``/PW folded into them -- are ``stopgrad`` from the policy-gradient's
point of view (§2.1 POLICY-GRADIENT), and the deterministic Game-1 features (PW, SUCC)
arrive already detached from ``game.py`` (numpy). ``V_phi`` itself is trained by the
value regression ``L_v`` (§3), and it is ``V_phi.detach()`` that supervises ``Vtilde``.

Learned/differentiable => **mlx** (Apple, matches the mini), following the sibling
modules ``dpp.py`` / ``head.py``.

Loss terms this file provides (`rl-training-spec.md` §3):
    L_v   = E[ || V_phi - return(R) ||^2 ]          (value regression to the relative return)
    L_aux = E[ || Vtilde - V_phi.detach() ||^2 ]    (query projection imitates the value)

Spec: `rl-training-spec.md` §2.1 (VALUE, ADVANTAGE), §2.2 (normalized register),
      §3 (loss: L_v, L_aux). Companion: `payload-spec.md` §2.3 (the query ``q_b``).

Public surface
--------------
- ``ValueHead``        : nn.Module, linproj(final intermediate) -> per-player V in R^l.
- ``AuxValueHead``     : nn.Module, linproj(query q_b) -> per-player Vtilde in R^l.
- ``StrategyValue``    : nn.Module bundling both; ``__call__`` -> ``(V, Vtilde)``.
- ``value_loss``       : L_v, per-player squared error V vs relative return.
- ``aux_value_loss``   : L_aux, Vtilde imitates V_phi.detach() (stop-grad on V).
- ``advantage``        : potential-based A = R + gamma*V(s') - V(s) (feature; §2.1).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ValueHead(nn.Module):
    """**V_phi**: linear projection on the estimator's final intermediate -> R^l.

    Per-player VALUE VECTOR (`rl-training-spec.md` §2.1). Input is the strategy
    estimator's *final intermediate* for each player -- and, because ``SUCC`` is folded
    into that intermediate upstream (``game.succ_feature`` concatenated before this
    projection), ``V`` is anticipatory (§2.2). Output is the per-player vector in R^l;
    the leading axis is the player axis, so a whole team/rollout batches naturally.

    ``d_intermediate`` is the width of the final intermediate *as fed here* -- i.e.
    including any concatenated ``succ_feature`` -- and ``l`` is the per-player reward /
    value dimension (the same ``l`` as ``R in R^l``).
    """

    def __init__(self, d_intermediate: int, l: int):
        super().__init__()
        self.proj = nn.Linear(d_intermediate, l)

    def __call__(self, final_intermediate: mx.array) -> mx.array:
        """``V_phi`` in R^l per player. ``final_intermediate``: ``(..., d_intermediate)``."""
        return self.proj(final_intermediate)


class AuxValueHead(nn.Module):
    """**Vtilde**: linear projection on the query ``q_b`` -> R^l, imitates ``V_phi``.

    Auxiliary per-player value read straight off the query (`rl-training-spec.md`
    §2.1 / §2.2). Trained by ``aux_value_loss`` to track ``V_phi.detach()`` (§3
    ``L_aux``) -- a cheap value estimate from the query alone. ``d_query`` is the query
    width (`payload-spec.md` §2.3 ``q_b``); ``l`` matches ``ValueHead``'s ``l``.
    """

    def __init__(self, d_query: int, l: int):
        super().__init__()
        self.proj = nn.Linear(d_query, l)

    def __call__(self, query: mx.array) -> mx.array:
        """``Vtilde`` in R^l per player. ``query``: ``(..., d_query)`` (the ``q_b``)."""
        return self.proj(query)


class StrategyValue(nn.Module):
    """Bundle of the two heads: ``V_phi`` (from the intermediate) and ``Vtilde`` (from q).

    Convenience container so the value stack is one trainable object. ``__call__``
    returns ``(V, Vtilde)`` and ``losses`` returns the ``(L_v, L_aux)`` pair of
    `rl-training-spec.md` §3. Note the two heads are separate ``nn.Linear`` projections
    of two *different* inputs (the final intermediate vs the query), exactly as §2.1
    specifies -- not a shared trunk.
    """

    def __init__(self, d_intermediate: int, d_query: int, l: int):
        super().__init__()
        self.value = ValueHead(d_intermediate, l)
        self.aux = AuxValueHead(d_query, l)

    def __call__(self, final_intermediate: mx.array, query: mx.array):
        """Return ``(V_phi, Vtilde)`` for the given intermediate and query."""
        return self.value(final_intermediate), self.aux(query)

    def losses(self, final_intermediate: mx.array, query: mx.array, returns: mx.array):
        """Return ``(L_v, L_aux)`` per `rl-training-spec.md` §3 for one batch.

        ``returns`` is the per-player relative return ``in R^l`` that supervises ``V_phi``.
        ``L_aux`` uses ``stop_gradient`` on ``V_phi`` so the query head imitates the
        value without dragging the value toward the (cheaper) query estimate.
        """
        v = self.value(final_intermediate)
        vtilde = self.aux(query)
        return value_loss(v, returns), aux_value_loss(vtilde, v)


def value_loss(v: mx.array, returns: mx.array) -> mx.array:
    """**L_v** = E[ || V_phi - return(R) ||^2 ] (`rl-training-spec.md` §3).

    Per-player squared error: the L2 over the ``l``-vector for each player, averaged
    over the player/batch axes. ``v`` and ``returns`` are both ``(..., l)``.
    """
    return mx.mean(mx.sum(mx.square(v - returns), axis=-1))


def aux_value_loss(vtilde: mx.array, v: mx.array) -> mx.array:
    """**L_aux** = E[ || Vtilde - V_phi.detach() ||^2 ] (`rl-training-spec.md` §3).

    The query projection imitates the value; ``stop_gradient`` on ``v`` makes it a
    one-way regression target (the value is not pulled toward ``Vtilde``). Both
    ``(..., l)``; per-player squared error averaged over the player/batch axes.
    """
    return mx.mean(mx.sum(mx.square(vtilde - mx.stop_gradient(v)), axis=-1))


def advantage(reward: mx.array, v_s: mx.array, v_next: mx.array, gamma: float = 1.0) -> mx.array:
    """**A_u** = R_u + gamma*V_phi(s') - V_phi(s) -- potential-based, per-player (§2.1).

    The operative dense signal (`rl-training-spec.md` §2.1: the advantage, not a
    hand-authored per-tick reward). Potential-based / TD, so around any cycle the shaping
    telescopes to zero -- the anti-bistability mechanism, and the reason the value head
    exists (§2.1, §5). Per-player VECTOR in R^l (never scalar): all of ``reward``,
    ``v_s``, ``v_next`` are ``(..., l)``.

    Detached into the actor by the loss (§3 ``L_pg`` uses ``A.detach()``); this returns
    the raw potential-based advantage and does not itself stop gradients -- the training
    loop applies ``stop_gradient`` where §3 requires it.
    """
    return reward + gamma * v_next - v_s
