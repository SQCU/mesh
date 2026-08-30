"""Game-1: the COMPUTED, deterministic strategy game over cartstate.

This module is the closed-form Game-1 of `design/rl-training-spec.md` §1 (and its
normalized restatement §2.2). It is **computed, deterministic, and NOT learned** --
the policy and value heads read `PW`/`SUCC` as *features*, and the policy gradient
treats them as ``stopgrad`` (§2.1 POLICY-GRADIENT: "``s,b,SUCC,PW`` and the FPS
C-program are ``stopgrad``"). Nothing here differentiates; it is pure numpy / plain
python by design (rl-training-spec §4: "computed (deterministic): ``s, b, PW, SUCC``";
the learned Game-2 *realization* lives behind the frozen FPS C-program, out of scope).

The two objects this file computes, both over `cartstate` (cart depths + control):

- ``projected_winner`` = **PW(s)** -- the nim-sum "who wins if nothing else changes"
  (rl-training-spec §1). The multi-cart position is Nim-structured: each cart at its
  depth-under-control on its golden path is a heap of size ``depth``, and a team's
  standing is the nim-sum (XOR) of the heaps it controls. The spec's worked example is
  the invariant this file is tested against: **one cart at d:2 beats two carts at d:1,
  i.e. ``1 XOR 1 = 0``** -- two shallow carts cancel to a zero nimber (no live threat),
  while depth concentrated into one cart (nimber ``2``) is a decisive threat. PW is the
  team holding the largest live (nonzero) nimber.

- ``succession`` = **SUCC(s)** -- backward induction over cartstate that recomputes PW
  under *successive decrements of the current leader's carts* and returns the ordered
  ``[(team, marginal_denial_value)]`` succession (rl-training-spec §1, §2.2). Denying
  the leader (pushing its leading cart back one control point = one depth decrement)
  eventually flips PW to the next team; the number of decrements to force each handover
  is that successor's ``marginal_denial_value``. Folding the whole succession into one
  immediate-frame allocation is exactly what makes the policy **anticipatory /
  time-smooth** (§1): gang the leader only to its marginal need, pre-empt the
  next-in-line, instead of reacting to each lead flip after it happens.

Cartstate model (matches `design/payload-strategy-spec.md` §1-§2 and `cart-force-field.md`):
each cart rides one golden path; its entire state is a scalar ``depth`` (integer
depth-under-control -- banked control points along arclength ``s in [0, L]``, origin=0)
plus which team, if any, **controls** it. Position is reversible: denial reverses a
cart's place on the path, which is precisely the decrement SUCC induces over.

Spec: `rl-training-spec.md` §1 (computed Game-1: PW nim-sum, SUCC backward induction),
      §2.2 (normalized register), §4 (computed vs learned vs frozen).

Public surface
--------------
- ``Cart``               : namedtuple ``(control, depth)`` -- one cart's cartstate.
- ``as_carts``           : normalize a heterogeneous cart sequence to ``list[Cart]``.
- ``nim_sum``            : XOR-reduce of heap sizes (the nimber primitive).
- ``team_nimbers``       : per-team nim-sum of controlled cart depths.
- ``projected_winner``   : PW(s) -- the nim-sum projected winner (team or None).
- ``succession``         : SUCC(s) -- ordered ``[(team, marginal_denial_value)]``.
- ``succ_feature``       : deterministic numpy featurization of SUCC for the estimator
                           intermediate (so PW/SUCC enter the learned heads as detached
                           anticipatory features; rl-training-spec §2.2 "inputs superset SUCC").
"""

from __future__ import annotations

from collections import namedtuple
from typing import Iterable, Optional, Sequence

import numpy as np

# One cart's cartstate: which team CONTROLS it (hashable id, or None = uncontrolled)
# and its integer depth-under-control (banked control points along the golden path).
Cart = namedtuple("Cart", ("control", "depth"))


def as_carts(carts: Iterable) -> list:
    """Normalize a heterogeneous cart sequence into ``list[Cart]``.

    Accepts, per element, a :class:`Cart`, a ``(control, depth)`` pair, or a mapping
    with ``control``/``depth`` keys (``team`` is accepted as an alias for ``control``).
    ``depth`` is coerced to a non-negative int; a negative depth clamps to 0 (a cart at
    origin contributes an empty heap). ``control`` is left as-is (any hashable, or None).
    """
    out = []
    for c in carts:
        if isinstance(c, Cart):
            control, depth = c.control, c.depth
        elif isinstance(c, dict):
            control = c.get("control", c.get("team"))
            depth = c.get("depth", 0)
        else:  # sequence/tuple (control, depth)
            control, depth = c[0], c[1]
        depth = int(depth)
        if depth < 0:
            depth = 0
        out.append(Cart(control, depth))
    return out


def _teams_of(carts: Sequence, teams: Optional[Iterable]) -> list:
    """Deterministic, sorted team roster: controls seen in `carts` union `teams`."""
    seen = {c.control for c in carts if c.control is not None}
    if teams is not None:
        seen |= set(teams)
    # sorted() gives a stable, deterministic tie-break order for argmax winners.
    return sorted(seen, key=lambda t: (0, t) if isinstance(t, (int, float)) else (1, str(t)))


def nim_sum(values: Iterable[int]) -> int:
    """XOR-reduce of heap sizes -- the nimber of a disjunctive sum of Nim heaps.

    ``nim_sum([]) == 0``. This is the primitive behind the spec's worked example
    (`rl-training-spec.md` §1): ``nim_sum([1, 1]) == 0`` (two carts at d:1 cancel),
    while ``nim_sum([2]) == 2`` (one cart at d:2 is a live threat).
    """
    acc = 0
    for v in values:
        acc ^= int(v)
    return acc


def team_nimbers(carts: Iterable, teams: Optional[Iterable] = None) -> dict:
    """Per-team nim-sum (XOR) of the depths of the carts that team controls.

    A cart with ``depth == 0`` or ``control is None`` contributes nothing (empty heap /
    no controller). Teams with no live carts appear with nimber ``0`` (either seen with
    only depth-0 carts, or supplied via `teams`). See `rl-training-spec.md` §1.
    """
    carts = as_carts(carts)
    roster = _teams_of(carts, teams)
    nimbers = {t: 0 for t in roster}
    for c in carts:
        if c.control is None or c.depth == 0:
            continue
        nimbers[c.control] = nimbers.get(c.control, 0) ^ c.depth
    return nimbers


def projected_winner(carts: Iterable, teams: Optional[Iterable] = None):
    """**PW(s)** -- the nim-sum projected winner: "who wins if nothing else changes".

    Closed-form over cartstate (`rl-training-spec.md` §1). Compute each team's nimber
    (``team_nimbers``) and return the team holding the largest *live* (nonzero) nimber;
    A strict maximum is required. Returns ``None`` when every team's nimber is 0 or
    several teams tie for the largest live nimber, because no unique team then holds
    the path-to-victory position.

    This is a FEATURE the policy and value read; it is not learned and carries no
    gradient (spec §2.1: ``PW`` is ``stopgrad``).
    """
    nimbers = team_nimbers(carts, teams)
    best = max(nimbers.values(), default=0)
    leaders = [team for team, value in nimbers.items() if value == best]
    return leaders[0] if best > 0 and len(leaders) == 1 else None


def _leader_deepest_cart(carts: list, leader) -> Optional[int]:
    """Index of `leader`'s deepest controlled cart (ties -> lowest index); None if none."""
    idx, best_depth = None, 0
    for i, c in enumerate(carts):
        if c.control == leader and c.depth > best_depth:
            best_depth, idx = c.depth, i
    return idx


def succession(carts: Iterable, teams: Optional[Iterable] = None) -> list:
    """**SUCC(s)** -- backward induction over cartstate; ordered succession list.

    Recompute ``PW`` under *successive decrements of the current leader's carts* and
    record who inherits and how much denial it took (`rl-training-spec.md` §1, §2.2).

    Model of a decrement: push the current leader's *deepest* controlled cart back one
    control point (``depth -= 1``) -- position is reversible (`payload-strategy-spec.md`
    §2), and a cart that reaches ``depth 0`` is at origin (an empty heap; here we neutral
    it by dropping its controller so it stops contributing to any nimber). After each
    decrement recompute ``PW``; when leadership passes to a new team, that team is the
    next in the succession and its ``marginal_denial_value`` is the number of decrements
    applied *since the previous handover* -- the marginal denial to force this handover.

    Returns ``[(team, marginal_denial_value), ...]`` where element 0 is the **current**
    projected winner with denial ``0`` (the incumbent), and each following entry is the
    next inheritor with the marginal decrements to reach it. Terminates when the position
    is neutralized (``PW`` becomes ``None`` -- no live leader left to deny) or every team
    has been enumerated. Deterministic; total depth strictly decreases each step, so the
    loop always terminates.

    This folded succession is what makes the downstream policy anticipatory (§1): the
    immediate allocation can gang the leader only to its *marginal* need and pre-empt the
    next-in-line, rather than reacting to each lead flip after it fires.
    """
    carts = as_carts(carts)
    roster = _teams_of(carts, teams)

    leader = projected_winner(carts, roster)
    if leader is None:
        return []  # no live path-to-victory holder; nothing to deny / no succession.

    order = [(leader, 0)]           # incumbent first, marginal denial 0.
    seen = {leader}
    cur = leader
    steps = 0                       # total decrements applied so far.
    prev_threshold = 0              # decrements at the previous handover.
    guard = sum(c.depth for c in carts) + 1  # strictly-decreasing depth bounds the loop.

    while steps < guard and len(seen) < len(roster):
        idx = _leader_deepest_cart(carts, cur)
        if idx is None:
            break  # current leader controls no live cart to push back.
        d = carts[idx].depth - 1
        # A cart pushed to origin (depth 0) recolors away from the leader: drop control
        # so it no longer contributes to `cur`'s nimber (position reset, not a score).
        carts[idx] = Cart(carts[idx].control if d > 0 else None, d)
        steps += 1

        nxt = projected_winner(carts, roster)
        if nxt != cur:
            if nxt is None:
                break  # position neutralized; no further live succession under denial.
            if nxt not in seen:
                order.append((nxt, steps - prev_threshold))
                seen.add(nxt)
                prev_threshold = steps
            cur = nxt
    return order


def succ_feature(
    carts: Iterable,
    teams: Optional[Iterable] = None,
    succ: Optional[Sequence] = None,
) -> np.ndarray:
    """Deterministic numpy featurization of SUCC, for folding into the estimator.

    Produces the fixed-width, **detached** anticipatory feature by which PW/SUCC enter
    the learned value and policy heads (`rl-training-spec.md` §2.2: value "inputs
    superset ``SUCC`` => anticipatory"; §2.1: these features are ``stopgrad``). The
    caller concatenates this onto the strategy estimator's final intermediate before the
    value/aux projections in ``value.py`` -- keeping the deterministic Game-1 features
    (numpy) and the learned Game-2 heads (mlx) cleanly split.

    Layout over the sorted team roster ``T`` (``|T|`` = number of teams):
      * ``[0:|T|]``   -- ``marginal_denial_value`` per team in roster order, ``0`` for a
                         team absent from the succession (unreachable under leader denial).
      * ``[|T|:2|T|]``-- one-hot of the current projected winner (all-zero if PW is None).
      * ``[2|T|]``    -- the incumbent's total denial budget = sum of marginal denials
                         (the aggregate decrements to exhaust the whole succession).
    Returns a ``float32`` vector of length ``2*|T| + 1``. Order matches ``team_nimbers``'
    roster so features align across PW/SUCC/value calls on the same cartstate.
    """
    carts = as_carts(carts)
    roster = _teams_of(carts, teams)
    index = {t: i for i, t in enumerate(roster)}
    n = len(roster)

    if succ is None:
        succ = succession(carts, roster)

    denial = np.zeros(n, dtype=np.float32)
    leader_onehot = np.zeros(n, dtype=np.float32)
    total = 0.0
    for k, (team, val) in enumerate(succ):
        if team in index:
            denial[index[team]] = float(val)
        total += float(val)
        if k == 0:  # element 0 is the incumbent / current projected winner.
            if team in index:
                leader_onehot[index[team]] = 1.0
    return np.concatenate([denial, leader_onehot, np.array([total], dtype=np.float32)])
