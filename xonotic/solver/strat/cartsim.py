"""Game-1 cartstate self-play simulator: the REAL environment the policy optimizes against.

This is the fast, deterministic self-play simulator of the Game-1 *strategy game* in
cartstate notation (``design/rl-training-spec.md`` §0-§1, ``design/payload-strategy-spec.md``
§1-§2). It is the environment the shared-weight policy (``estimator.StrategyEstimator``)
acts in and is trained against by REINFORCE: a ``step()`` consumes each player's
instrument allocation, applies the committed push/pull law to ``cartstate``, and exposes
the deterministic nim-sum projected-winner ``PW`` and its backward-induction succession
``SUCC`` (computed by the sibling numpy Game-1 module :mod:`game`, NOT learned) as the
projected-winner oracle and the anticipatory feature the policy/value read.

Two games (``rl-training-spec`` §0). This file is Game-1 ONLY: the strategy game in
cartstate notation, closed-form and deterministic. It is *not* the Xonotic FPS (Game-2,
the frozen C program) — the push/pull law here is the analytic strategy-surface model of
``payload-strategy-spec`` §2, a right-sized stand-in for the FPS realization so the policy
can be trained and measured without the engine in the loop. Everything here is numpy /
plain python (``rl-training-spec`` §4: "computed (deterministic)"); the learned surface
(``estimator``, mlx) sits on top and treats this environment's outputs as ``stopgrad``
features.

The committed law (``payload-strategy-spec`` §1-§2), implemented in :meth:`CartSim.step`:

  * ``k`` teams, ``j`` carts, ``l`` players. Each cart rides one path of ``L`` control
    points; its whole state is a scalar position ``s in [0, L]`` plus which team (if any)
    **controls / colors** it. ``team_of`` maps each player to its team.
  * **Control = plurality of presence.** A player's allocation to ``push_cart_i`` or
    ``suppress_cart_i`` places that body in cart ``i``'s cylinder; the team with the
    strict plurality of present bodies controls the cart this tick (§1).
  * **Two regimes, distinguished by the color team's PRESENCE, not its advantage** (§2):
      - Regime A (contested, ``w_A > 0``): ``v = clamp( speed*(w_A - w_opp)/(1+w_opp^2),
        -contest_speed, +max_speed )`` — bounded, damped tug-of-war that stays local.
      - Regime B (abandoned, ``w_A = 0``, some opponent present): ``v = -reverse_speed*w_B``
        (``B`` = strongest present opposing team), a linear walk home; at origin the cart
        **recolors** to ``B``.
  * **Banked score is monotone** (§1 correction note): an upward control-point crossing
    under control banks a point; a downward crossing does NOT un-bank it. Cart POSITION
    reverses; banked SCORE never does. :meth:`CartSim.monotone_objective` returns the
    non-decreasing banked total (the "monotone progress" line — a diagnostic, NOT the RL
    objective; the operative objective is the RELATIVE ``PW`` possession, §5).

The projected-winner oracle (``rl-training-spec`` §1). ``PW(s)`` and ``SUCC(s)`` are read
straight from :mod:`game` on the discretized cartstate (``depth = floor(s)`` per cart,
``control`` = color, uncontrolled -> ``None``). They are the closed-form nim-sum winner
and its succession — computed, deterministic, never learned — and are exactly the
features the policy conditions on and the value regresses through.

Public surface
--------------
- ``INSTRUMENT_KINDS``       : ("push_cart", "suppress_cart", "idle") — the Game-1 movers.
- ``n_instruments(j)``       : instrument count for ``j`` carts = ``2*j + 1``.
- ``instrument_index``/``decode_instrument`` : (kind, cart) <-> flat instrument index.
- ``CartState``              : dataclass — pos, control(color), banked, L, t (the cartstate).
- ``CartSim``                : the simulator — ``reset`` / ``step`` / ``rollout`` + oracle.
- ``to_carts``               : CartState -> ``list[game.Cart]`` (depth=floor(pos), color).
- ``random_policy`` / ``greedy_deny_policy`` : numpy reference policies for self-play
                               without the mlx estimator (real, seeded, deterministic).

Spec: ``rl-training-spec.md`` §0-§1, §4-§5 ; ``payload-strategy-spec.md`` §1-§3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import numpy as np

from .game import Cart, projected_winner, succession

__all__ = [
    "INSTRUMENT_KINDS",
    "n_instruments",
    "instrument_index",
    "decode_instrument",
    "CartState",
    "CartSim",
    "to_carts",
    "random_policy",
    "greedy_deny_policy",
]


# The Game-1 movers. ``payload-spec`` §2.1's full instrument set adds contest_post /
# hunt_rival / explore_cell; those do not move cartstate (they touch posts / observation),
# so the closed-form cartstate simulator carries only the two that push/pull a cart plus a
# no-op idle. ``estimator`` can still emit over a wider instrument set; ``CartSim`` maps any
# non-cart instrument onto "idle" (no presence contributed).
INSTRUMENT_KINDS: tuple[str, ...] = ("push_cart", "suppress_cart", "idle")


def n_instruments(j_carts: int) -> int:
    """Number of Game-1 instruments for ``j`` carts: ``push_i`` + ``suppress_i`` + idle."""
    return 2 * int(j_carts) + 1


def instrument_index(kind: str, cart: int, j_carts: int) -> int:
    """Flat instrument index for ``(kind, cart)`` over ``j`` carts.

    Layout: ``[push_0 .. push_{j-1}, suppress_0 .. suppress_{j-1}, idle]``. ``idle`` ignores
    ``cart``. Raises on an unknown kind or an out-of-range cart.
    """
    j = int(j_carts)
    if kind == "idle":
        return 2 * j
    if not (0 <= int(cart) < j):
        raise ValueError(f"cart {cart} out of range [0,{j})")
    if kind == "push_cart":
        return int(cart)
    if kind == "suppress_cart":
        return j + int(cart)
    raise ValueError(f"unknown instrument kind {kind!r}; expected one of {INSTRUMENT_KINDS}")


def decode_instrument(idx: int, j_carts: int) -> tuple[str, int]:
    """Inverse of :func:`instrument_index`: flat index -> ``(kind, cart)`` (idle -> cart -1)."""
    j = int(j_carts)
    i = int(idx)
    if i == 2 * j:
        return ("idle", -1)
    if 0 <= i < j:
        return ("push_cart", i)
    if j <= i < 2 * j:
        return ("suppress_cart", i - j)
    raise ValueError(f"instrument index {idx} out of range [0,{2 * j + 1})")


@dataclass
class CartState:
    """One cartstate ``s``: cart positions + color + monotone banked score.

    ``pos`` is the continuous arclength ``s in [0, L]`` per cart; ``control`` is the cart
    COLOR (team id in ``[0,k)`` or ``-1`` uncontrolled); ``banked`` is the per-team monotone
    banked-point total; ``L`` the number of control points; ``t`` the step counter. The
    nim heap depth Game-1 reads is ``floor(pos)`` (see :func:`to_carts`).
    """

    pos: np.ndarray               # (j,) float, arclength in [0, L]
    control: np.ndarray           # (j,) int   team color, -1 = uncontrolled
    banked: np.ndarray            # (k,) float monotone banked points per team
    L: int                        # control points per path
    t: int = 0                    # step index
    highwater: np.ndarray = field(default=None)  # (j,) int highest banked control point/cart

    def copy(self) -> "CartState":
        return CartState(
            pos=self.pos.copy(),
            control=self.control.copy(),
            banked=self.banked.copy(),
            L=self.L,
            t=self.t,
            highwater=None if self.highwater is None else self.highwater.copy(),
        )


def to_carts(state: CartState) -> list:
    """``CartState`` -> ``list[game.Cart]`` for the nim-sum ``PW``/``SUCC`` oracle.

    Each cart becomes ``Cart(control=color_or_None, depth=floor(pos))`` — the heap size is
    the integer control-point depth the cart currently holds. Uncontrolled carts
    (``control == -1``) pass ``None`` so they contribute no heap (``game`` §1).
    """
    carts = []
    for c in range(len(state.pos)):
        col = int(state.control[c])
        depth = int(np.floor(state.pos[c]))
        carts.append(Cart(None if col < 0 else col, depth))
    return carts


class CartSim:
    """Fast deterministic self-play simulator of the Game-1 cartstate game.

    ``k`` teams, ``j`` carts, ``l`` players (``team_of`` = player->team). Constructs the
    initial cartstate, applies per-player instrument allocations through the committed
    push/pull law (``payload-strategy-spec`` §2), and exposes the nim-sum projected-winner
    oracle (``game.projected_winner`` / ``game.succession``) as the RELATIVE objective the
    policy optimizes. Deterministic given the seed; pure numpy.

    Parameters
    ----------
    k_teams, j_carts, l_players : int
        The three population counts (``rl-training-spec`` §0 ``k`` teams, ``l`` players;
        ``payload-strategy-spec`` §1 ``j``... here ``j`` = carts, per this task's convention).
    team_of : (l,) int, optional
        Player -> team map. Defaults to a balanced round-robin ``p % k``.
    L : int
        Control points per path (path length). Default 8.
    dt : float
        Integration step for ``s(t+dt) = clamp(s + v*dt, 0, L)`` (§2). Default 1.0.
    speed, max_speed, contest_speed, reverse_speed : float
        The push/pull law constants (§2). ``speed`` scales the contested drive;
        ``max_speed`` caps forward motion; ``contest_speed`` caps backward drift while a
        defense is present; ``reverse_speed`` is the per-body capture walk when abandoned.
    seed : int, optional
        RNG seed for :meth:`reset`'s initial layout and the reference policies.
    """

    def __init__(
        self,
        k_teams: int,
        j_carts: int,
        l_players: int,
        team_of: Optional[Sequence[int]] = None,
        *,
        L: int = 8,
        dt: float = 1.0,
        speed: float = 1.0,
        max_speed: float = 1.0,
        contest_speed: float = 0.25,
        reverse_speed: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if k_teams < 2:
            raise ValueError("need at least 2 teams for a relative objective")
        if j_carts < 1 or l_players < 1:
            raise ValueError("need >=1 cart and >=1 player")
        self.k = int(k_teams)
        self.j = int(j_carts)
        self.l = int(l_players)
        self.L = int(L)
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_speed = float(max_speed)
        self.contest_speed = float(contest_speed)
        self.reverse_speed = float(reverse_speed)
        self.M = n_instruments(self.j)  # instrument count = 2j+1
        if team_of is None:
            team_of = [p % self.k for p in range(self.l)]
        team_of = np.asarray(team_of, dtype=np.int64)
        if team_of.shape != (self.l,):
            raise ValueError(f"team_of must be length l={self.l}; got {team_of.shape}")
        if team_of.min() < 0 or team_of.max() >= self.k:
            raise ValueError("team_of entries must be in [0,k)")
        self.team_of = team_of
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ reset
    def reset(self) -> CartState:
        """Fresh cartstate: carts near origin, colored round-robin, nothing banked.

        Each cart starts at a small positive position (so ``PW`` is defined at t=0) colored
        by a distinct team round-robin, giving a live nim position to contest immediately.
        Deterministic given the constructor seed.
        """
        pos = np.full(self.j, 1.0, dtype=np.float64)
        # stagger initial depths so the opening nim position is non-degenerate
        for c in range(self.j):
            pos[c] = 1.0 + (c % max(1, self.L - 1))
        pos = np.clip(pos, 0.0, self.L)
        control = np.array([c % self.k for c in range(self.j)], dtype=np.int64)
        banked = np.zeros(self.k, dtype=np.float64)
        highwater = np.floor(pos).astype(np.int64)
        return CartState(pos=pos, control=control, banked=banked, L=self.L, t=0,
                         highwater=highwater)

    # ---------------------------------------------------------------- presence
    def _presence(self, actions: np.ndarray) -> np.ndarray:
        """Per-cart per-team presence weights ``w[c, t]`` from player allocations.

        A player assigned ``push_c`` or ``suppress_c`` places one body in cart ``c``'s
        cylinder for its team; ``idle`` contributes nothing. Returns ``(j, k)``.
        """
        w = np.zeros((self.j, self.k), dtype=np.float64)
        for p in range(self.l):
            kind, cart = decode_instrument(int(actions[p]), self.j)
            if kind == "idle":
                continue
            w[cart, self.team_of[p]] += 1.0
        return w

    # -------------------------------------------------------------------- step
    def step(self, state: CartState, actions: Sequence[int]) -> tuple[CartState, dict]:
        """Apply per-player instrument allocations to cartstate (``payload-strategy`` §2).

        ``actions`` is a length-``l`` int array of instrument indices (see
        :func:`instrument_index`). Returns ``(next_state, info)`` where ``info`` carries the
        per-cart velocities, per-cart presence, the projected winner before/after, the
        succession, and the monotone objective — the real per-step record a rollout buffers.

        The law, per cart ``c`` with color ``A = control[c]`` and presence ``w[c, .]``:
          * ``w_A = w[c, A]`` (0 if uncontrolled), ``w_opp = sum_{t!=A} w[c, t]``.
          * Regime A (``w_A > 0``): damped contested drive, clamped to
            ``[-contest_speed, +max_speed]``.
          * Regime B (``w_A == 0``, ``w_opp > 0``): ``v = -reverse_speed * w_B`` (capture);
            at origin the cart recolors to the strongest present opponent ``B``.
          * No presence: ``v = 0``.
        Banking is monotone: an upward integer control-point crossing under control banks a
        point into that team; downward motion never un-banks (§1 correction).
        """
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape != (self.l,):
            raise ValueError(f"actions must be length l={self.l}; got {actions.shape}")
        s = state.copy()
        if s.highwater is None:
            s.highwater = np.floor(s.pos).astype(np.int64)

        w = self._presence(actions)                       # (j, k)
        pw_before = projected_winner(to_carts(state), teams=range(self.k))
        velocities = np.zeros(self.j, dtype=np.float64)

        for c in range(self.j):
            color = int(s.control[c])
            w_c = w[c]
            w_A = w_c[color] if color >= 0 else 0.0
            w_opp_total = float(w_c.sum() - (w_A if color >= 0 else 0.0))

            if w_A > 0.0:  # Regime A — contested, damped, local
                v = self.speed * (w_A - w_opp_total) / (1.0 + w_opp_total ** 2)
                v = float(np.clip(v, -self.contest_speed, self.max_speed))
            elif w_opp_total > 0.0:  # Regime B — abandoned, linear capture walk home
                # strongest present opposing team
                opp = w_c.copy()
                if color >= 0:
                    opp[color] = -1.0
                B = int(np.argmax(opp))
                w_B = float(w_c[B])
                v = -self.reverse_speed * w_B
            else:  # nobody present
                v = 0.0

            velocities[c] = v
            old = float(s.pos[c])
            new = float(np.clip(old + v * self.dt, 0.0, self.L))
            s.pos[c] = new

            # monotone banking: upward integer crossings under control bank points
            if color >= 0 and new > old:
                new_floor = int(np.floor(new))
                if new_floor > int(s.highwater[c]):
                    gained = new_floor - int(s.highwater[c])
                    s.banked[color] += float(gained)
                    s.highwater[c] = new_floor

            # origin collapse under Regime B -> recolor to the capturing opponent
            if new <= 0.0 and w_A == 0.0 and w_opp_total > 0.0:
                opp = w_c.copy()
                if color >= 0:
                    opp[color] = -1.0
                s.control[c] = int(np.argmax(opp))
                s.highwater[c] = 0  # position reset (score already banked, monotone)

        s.t = state.t + 1
        pw_after = projected_winner(to_carts(s), teams=range(self.k))
        succ = succession(to_carts(s), teams=range(self.k))
        info = {
            "presence": w,
            "velocities": velocities,
            "pw_before": pw_before,
            "pw_after": pw_after,
            "succession": succ,
            "monotone_objective": float(s.banked.sum()),
            "banked": s.banked.copy(),
        }
        return s, info

    # ----------------------------------------------------------------- oracle
    def projected_winner(self, state: CartState):
        """``PW(s)`` — the nim-sum projected winner (``game.projected_winner``)."""
        return projected_winner(to_carts(state), teams=range(self.k))

    def succession(self, state: CartState) -> list:
        """``SUCC(s)`` — the backward-induction succession (``game.succession``)."""
        return succession(to_carts(state), teams=range(self.k))

    def monotone_objective(self, state: CartState) -> float:
        """The monotone banked-score total (diagnostic; the RL objective is relative, §5)."""
        return float(state.banked.sum())

    # --------------------------------------------------------------- rollout
    def rollout(
        self,
        policy: Callable[["CartSim", CartState], np.ndarray],
        n_steps: int,
        state: Optional[CartState] = None,
    ) -> list:
        """Run ``n_steps`` of self-play under ``policy``; return the transition list.

        ``policy(sim, state) -> actions`` (length-``l`` instrument indices). Each element of
        the returned list is a dict ``{step, state, actions, next_state, info}`` — the real
        environment record a :class:`~solver.strat.buffers.ReplayBuffer` consumes. This is
        the loop REINFORCE trains against; ``policy`` can be a reference numpy policy (below)
        or the mlx ``estimator`` wrapped to emit instrument indices.
        """
        s = state if state is not None else self.reset()
        transitions = []
        for step in range(int(n_steps)):
            actions = np.asarray(policy(self, s), dtype=np.int64).reshape(-1)
            nxt, info = self.step(s, actions)
            transitions.append({
                "step": step,
                "state": s,
                "actions": actions,
                "next_state": nxt,
                "info": info,
            })
            s = nxt
        return transitions


# --------------------------------------------------------------------------- #
# Reference numpy policies: exercise CartSim as a real environment WITHOUT the
# mlx estimator (deterministic, seeded). These are honest baselines, not the
# learned policy — the estimator (mlx) is plugged in as `policy` for training.
# --------------------------------------------------------------------------- #

def random_policy(sim: "CartSim", state: CartState, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Uniform-random instrument allocation over all ``l`` players (a broad-sampling floor)."""
    g = rng if rng is not None else sim._rng
    return g.integers(0, sim.M, size=sim.l)


def greedy_deny_policy(sim: "CartSim", state: CartState) -> np.ndarray:
    """Relative-objective heuristic: deny the projected winner, else push your own cart.

    Each player looks at ``PW(s)``: if its team is NOT the projected winner it suppresses
    the leader's deepest cart (deny the path-to-victory); otherwise it pushes its own team's
    shallowest live cart (acquire / extend). This is the RELATIVE objective of
    ``rl-training-spec`` §5 in closed form — a deterministic reference the learned policy is
    measured against, never a stand-in for it.
    """
    carts = to_carts(state)
    pw = projected_winner(carts, teams=range(sim.k))
    # leader's deepest cart
    leader_cart = -1
    best_depth = -1
    for c in range(sim.j):
        if int(state.control[c]) == (pw if pw is not None else -999):
            d = int(np.floor(state.pos[c]))
            if d > best_depth:
                best_depth, leader_cart = d, c
    actions = np.empty(sim.l, dtype=np.int64)
    for p in range(sim.l):
        team = int(sim.team_of[p])
        if pw is not None and team != pw and leader_cart >= 0:
            actions[p] = instrument_index("suppress_cart", leader_cart, sim.j)
        else:
            # push this team's shallowest controlled cart (or cart p%j if none controlled)
            own = [c for c in range(sim.j) if int(state.control[c]) == team]
            if own:
                target = min(own, key=lambda c: state.pos[c])
            else:
                target = p % sim.j
            actions[p] = instrument_index("push_cart", target, sim.j)
    return actions
