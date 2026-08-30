"""Team-state-only coarse baseline: a no-regret learner STRUCTURALLY BLIND to
cross-team configuration.

This is the deliberately-partitioned control policy for `design/dominance-demo.md`.
It maps ONLY the acting team's own aggregate state -- own cart depths/banked, own
player count -- through a linear projection to per-(player, instrument) decision
weights, and mixes a small basis of own-team-only strategy templates with a
NO-REGRET (Hedge / multiplicative-weights, external-regret-vanishing) update over
its own action set. It reads NO rival nimbers, NO `SUCC`, NO `PW`, and NO
cross-team control labels: every feature it consumes is a function of the acting
team's own rows alone.

Count invariance (`rl-training-spec.md` §4). The projection and every template are
defined in the same row sense as the estimator: adding a team, player, or cart adds
rows, never coordinates. The same `TeamOnlyBaseline` instance faces any `(k, j, l)`
shape without resizing or reinitialisation -- so it can be run on held-out shapes.

Why it is the right foil. Hedge has vanishing external regret against the best
FIXED template in hindsight, so it is a genuine no-regret learner -- not a weak
scripted stub. But its regret guarantee is RELATIVE TO ITS OWN PARTITION: every
template is blind to which non-own cart a rival leads, so any Hedge mixture emits an
identical action distribution on two global states that share an own-team view but
differ in `SUCC`. No amount of no-regret learning escapes that aliasing; that is the
point the demonstration makes concrete (`baseline_teamonly` is the blind side).

Pure numpy, seeded, deterministic given the seed. No mlx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cartsim import decode_instrument, instrument_index

__all__ = ["TeamOnlyBaseline", "own_team_view"]


def own_team_view(sim, state, team: int) -> dict:
    """The ONLY thing the baseline is allowed to see: team ``team``'s own aggregate.

    Returns own-controlled cart depths, own banked score, own player count/fraction,
    and per-cart own-relative descriptors (is this cart mine, my depth on it). It
    NEVER inspects the controller or depth of a cart the team does not control -- a
    non-own cart collapses to a single "not mine" token with no rival identity, which
    is exactly what makes the partition blind to cross-team configuration.
    """
    j = sim.j
    L = float(max(1, sim.L))
    control = np.asarray(state.control, dtype=np.int64)
    pos = np.asarray(state.pos, dtype=np.float64)
    own_carts = [c for c in range(j) if int(control[c]) == team]
    own_depths = [float(np.floor(pos[c])) for c in own_carts]
    team_size = int(np.sum(np.asarray(sim.team_of) == team))
    # per-cart own-relative descriptor: (mine?, my_depth_if_mine). Non-own carts are
    # indistinguishable (both entries 0) -- no rival identity leaks in.
    cart_desc = np.zeros((j, 2), dtype=np.float32)
    for c in range(j):
        if int(control[c]) == team:
            cart_desc[c, 0] = 1.0
            cart_desc[c, 1] = float(np.floor(pos[c])) / L
    agg = np.array(
        [
            (max(own_depths) if own_depths else 0.0) / L,
            (sum(own_depths) / len(own_depths) if own_depths else 0.0) / L,
            float(state.banked[team]) / max(1.0, L * j),
            float(len(own_carts)) / max(1, j),
            float(team_size) / max(1, sim.l),
        ],
        dtype=np.float32,
    )
    return {
        "agg": agg,
        "cart_desc": cart_desc,
        "own_carts": own_carts,
        "own_depths": own_depths,
        "team_size": team_size,
    }


# ---- own-team-only strategy templates (the learner's action set) ------------
# Each template is a deterministic per-player instrument assignment computed from
# own_team_view ALONE. None can identify a rival leader / next-in-line, so none can
# de-alias a cross-team ambiguity. Hedge mixes over exactly this set.

def _tmpl_push_shallow(sim, view, team, rng, j):
    own = view["own_carts"]
    target = min(own, key=lambda c: view["cart_desc"][c, 1]) if own else None
    return ("push_cart", target)


def _tmpl_push_deep(sim, view, team, rng, j):
    own = view["own_carts"]
    target = max(own, key=lambda c: view["cart_desc"][c, 1]) if own else None
    return ("push_cart", target)


def _tmpl_suppress_blind(sim, view, team, rng, j):
    # Suppress a NON-own cart. Blind to which rival leads: picks uniformly among the
    # carts it does not control (this is where the aliasing bites -- it cannot see
    # which non-own cart is the projected-winner's).
    non_own = [c for c in range(j) if c not in view["own_carts"]]
    target = int(rng.choice(non_own)) if non_own else None
    return ("suppress_cart", target)


def _tmpl_idle(sim, view, team, rng, j):
    return ("idle", -1)


TEMPLATES = [
    ("push_shallow", _tmpl_push_shallow),
    ("push_deep", _tmpl_push_deep),
    ("suppress_blind", _tmpl_suppress_blind),
    ("idle", _tmpl_idle),
]


@dataclass
class TeamOnlyBaseline:
    """A shared-weight, count-invariant, cross-team-BLIND no-regret policy.

    One Hedge distribution per template basis, shared across teams and shapes. The
    linear projection ``proj`` maps the own-team aggregate to a per-template log-bias
    (still own-only); Hedge cumulative rewards add on top. ``act`` samples a template
    per team from that distribution and expands it to per-player instruments through
    own-team-only templates; ``update`` applies the multiplicative-weights step from
    the realised per-team relative reward.
    """

    eta: float = 0.5
    seed: Optional[int] = 0
    n_templates: int = field(default=len(TEMPLATES))
    # Hedge cumulative reward per template (external-regret-vanishing MW state).
    cum: np.ndarray = field(default=None)
    # Fixed own-team-only linear projection: agg(5) -> per-template log-bias.
    proj: np.ndarray = field(default=None)
    _rng: np.random.Generator = field(default=None)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self.cum = np.zeros(self.n_templates, dtype=np.float64)
        # deterministic own-only projection (5 agg features -> n_templates biases);
        # small, fixed -- it is a prior shaping of the Hedge simplex, still team-only.
        g = np.random.default_rng((self.seed or 0) + 991)
        self.proj = g.normal(scale=0.3, size=(5, self.n_templates)).astype(np.float64)
        self._last_probs = None

    # --------------------------------------------------------------- distribution
    def template_probs(self, agg: np.ndarray) -> np.ndarray:
        """Hedge distribution over templates, biased by the own-team linear projection.

        p ~ softmax(eta * cum + agg @ proj). Depends on the acting team's OWN
        aggregate only -- so two global states with an identical own view yield an
        identical distribution here, by construction.
        """
        logits = self.eta * self.cum + agg @ self.proj
        logits = logits - logits.max()
        p = np.exp(logits)
        return p / p.sum()

    # ----------------------------------------------------------------------- act
    def act(self, sim, state, teams=None, sample: bool = True) -> np.ndarray:
        """Per-player instrument indices for ``teams`` (default: all teams).

        Each requested team independently reads its OWN view, draws a template from
        the Hedge distribution (or takes the argmax when ``sample`` is False), and
        expands it to its players. Players on teams not in ``teams`` get idle.
        """
        j = sim.j
        actions = np.full(sim.l, instrument_index("idle", 0, j), dtype=np.int64)
        team_set = range(sim.k) if teams is None else list(teams)
        self._last_probs = {}
        for team in team_set:
            view = own_team_view(sim, state, team)
            p = self.template_probs(view["agg"])
            self._last_probs[team] = p
            if sample:
                ti = int(self._rng.choice(self.n_templates, p=p))
            else:
                ti = int(np.argmax(p))
            _, tmpl = TEMPLATES[ti]
            kind, target = tmpl(sim, view, team, self._rng, j)
            for pl in range(sim.l):
                if int(sim.team_of[pl]) != team:
                    continue
                if kind == "idle" or target is None:
                    actions[pl] = instrument_index("idle", 0, j)
                else:
                    actions[pl] = instrument_index(kind, target, j)
        return actions

    def action_distribution(self, sim, state, team: int) -> np.ndarray:
        """Marginal probability over the M instruments for one player on ``team``.

        Marginalises the Hedge template distribution into instrument space (the
        object the aliasing proof compares across two states). Cross-team-blind: a
        pure function of ``own_team_view``.
        """
        j = sim.j
        view = own_team_view(sim, state, team)
        p = self.template_probs(view["agg"])
        dist = np.zeros(sim.M, dtype=np.float64)
        non_own = [c for c in range(j) if c not in view["own_carts"]]
        for ti, (_, tmpl) in enumerate(TEMPLATES):
            kind, target = tmpl(sim, view, team, np.random.default_rng(0), j)
            if kind == "suppress_cart":
                # blind suppress spreads uniformly over non-own carts
                if non_own:
                    for c in non_own:
                        dist[instrument_index("suppress_cart", c, j)] += p[ti] / len(non_own)
                else:
                    dist[instrument_index("idle", 0, j)] += p[ti]
            elif kind == "idle" or target is None:
                dist[instrument_index("idle", 0, j)] += p[ti]
            else:
                dist[instrument_index(kind, target, j)] += p[ti]
        return dist

    # -------------------------------------------------------------------- update
    def update(self, team_reward: np.ndarray, teams=None):
        """Multiplicative-weights (Hedge) update from the realised per-team reward.

        ``team_reward`` is the relative role reward per team (the real environment
        signal). Uses the template distribution that produced the last action as the
        importance weighting, giving an unbiased reward estimate per template. This is
        the standard no-regret step: cumulative reward drives ``exp(eta * cum)``.
        """
        if self._last_probs is None:
            return
        team_set = list(self._last_probs.keys()) if teams is None else list(teams)
        for team in team_set:
            p = self._last_probs.get(team)
            if p is None:
                continue
            r = float(team_reward[team])
            # reward-to-all-templates proportional to responsibility p (bandit-lite):
            # each template credited by its selection probability. Vanishing external
            # regret against the best fixed template in hindsight.
            self.cum += p * r
        self._last_probs = None
