from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import xor
from typing import Hashable, Iterable, Mapping, Sequence


State = Hashable
Role = Hashable


def mex(values: Iterable[int]) -> int:
    values = {int(value) for value in values if int(value) >= 0}
    value = 0
    while value in values:
        value += 1
    return value


@dataclass(frozen=True)
class RoleValue:
    mobility: int
    options: tuple[State, ...]
    complete: bool


@dataclass(frozen=True)
class GameValue:
    kind: str
    nimber: int | None
    role_values: Mapping[Role, RoleValue]
    reason: str | None = None


@dataclass(frozen=True)
class ComponentBelief:
    probabilities: Mapping[State, float]
    unknown_probability: float = 0.0


@dataclass(frozen=True)
class NimberBelief:
    probabilities: Mapping[int, float]
    unresolved_probability: float

    def probability(self, nimber: int) -> float:
        return float(self.probabilities.get(int(nimber), 0.0))


def _ordered(values: Iterable[State]) -> tuple[State, ...]:
    return tuple(sorted(set(values), key=repr))


class FiniteGameGraph:
    def __init__(
        self,
        options_by_role: Mapping[State, Mapping[Role, Iterable[State]]],
        roles: Iterable[Role],
        complete_options: Iterable[tuple[State, Role]] | None = None,
    ) -> None:
        self.roles = _ordered(roles) or ("player",)
        self._options = {
            state: {role: _ordered(options) for role, options in by_role.items()}
            for state, by_role in options_by_role.items()
        }
        if complete_options is None:
            states = set(self._options)
            for by_role in self._options.values():
                for options in by_role.values():
                    states.update(options)
            complete_options = ((state, role) for state in states for role in self.roles)
        self._complete = set(complete_options)

    @classmethod
    def impartial(
        cls,
        options: Mapping[State, Iterable[State]],
        roles: Iterable[Role] = ("player",),
    ) -> "FiniteGameGraph":
        roles = _ordered(roles)
        normalized = {state: tuple(successors) for state, successors in options.items()}
        return cls(
            {state: {role: successors for role in roles} for state, successors in normalized.items()},
            roles,
        )

    @classmethod
    def partizan(
        cls,
        options_by_role: Mapping[State, Mapping[Role, Iterable[State]]],
        roles: Iterable[Role] | None = None,
    ) -> "FiniteGameGraph":
        if roles is None:
            roles = {role for options in options_by_role.values() for role in options}
        return cls(options_by_role, roles)

    def options(self, state: State, role: Role | None = None) -> tuple[State, ...]:
        if role is not None:
            return self._options.get(state, {}).get(role, ())
        return _ordered(
            option
            for current_role in self.roles
            for option in self.options(state, current_role)
        )

    def options_complete(self, state: State, role: Role) -> bool:
        return (state, role) in self._complete

    def reachable(self, state: State) -> tuple[State, ...]:
        seen = set()
        pending = [state]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self.options(current))
        return _ordered(seen)

    def classification(self, state: State) -> str:
        for current in self.reachable(state):
            sets = [set(self.options(current, role)) for role in self.roles]
            if sets and any(options != sets[0] for options in sets[1:]):
                return "partizan"
        return "impartial"

    def role_values(self, state: State) -> dict[Role, RoleValue]:
        return {
            role: RoleValue(
                len(self.options(state, role)),
                self.options(state, role),
                self.options_complete(state, role),
            )
            for role in self.roles
        }

    def evaluate(self, state: State) -> GameValue:
        reachable = self.reachable(state)
        if any(not self.options_complete(current, role) for current in reachable for role in self.roles):
            return GameValue("unresolved", None, self.role_values(state), "incomplete option graph")
        if self.classification(state) == "partizan":
            return GameValue("partizan", None, self.role_values(state))
        cache: dict[State, int] = {}
        visiting = set()

        def grundy(current: State) -> int:
            if current in cache:
                return cache[current]
            if current in visiting:
                raise RuntimeError
            visiting.add(current)
            value = mex(grundy(option) for option in self.options(current, self.roles[0]))
            visiting.remove(current)
            cache[current] = value
            return value

        try:
            nimber = grundy(state)
        except RuntimeError:
            return GameValue("unresolved", None, self.role_values(state), "cyclic option graph")
        return GameValue("impartial", nimber, self.role_values(state))


def disjunctive_sum_options(
    graphs: Sequence[FiniteGameGraph],
    states: Sequence[State],
    role: Role | None = None,
) -> tuple[tuple[State, ...], ...]:
    states = tuple(states)
    moves = []
    for index, (graph, state) in enumerate(zip(graphs, states)):
        for option in graph.options(state, role):
            successor = list(states)
            successor[index] = option
            moves.append(tuple(successor))
    return tuple(moves)


def disjunctive_sum_value(
    graphs: Sequence[FiniteGameGraph],
    states: Sequence[State],
) -> GameValue:
    values = [graph.evaluate(state) for graph, state in zip(graphs, states)]
    if all(value.kind == "impartial" for value in values):
        return GameValue("impartial", reduce(xor, (value.nimber for value in values), 0), {})
    roles = _ordered(role for graph in graphs for role in graph.roles)
    role_values = {
        role: RoleValue(
            len(disjunctive_sum_options(graphs, states, role)),
            disjunctive_sum_options(graphs, states, role),
            all(graph.options_complete(state, role) for graph, state in zip(graphs, states) if role in graph.roles),
        )
        for role in roles
    }
    kind = "partizan" if any(value.kind == "partizan" for value in values) else "unresolved"
    reasons = "; ".join(value.reason for value in values if value.reason)
    return GameValue(kind, None, role_values, reasons or None)


def _coerce_belief(belief: ComponentBelief | Mapping[State, float]) -> ComponentBelief:
    if isinstance(belief, ComponentBelief):
        return belief
    total = sum(max(0.0, float(weight)) for weight in belief.values())
    if total == 0.0:
        return ComponentBelief({}, 1.0)
    return ComponentBelief({state: max(0.0, float(weight)) / total for state, weight in belief.items()})


def belief_nimber_distribution(
    graphs: Sequence[FiniteGameGraph],
    beliefs: Sequence[ComponentBelief | Mapping[State, float]],
) -> NimberBelief:
    distribution = {0: 1.0}
    for graph, source in zip(graphs, beliefs):
        belief = _coerce_belief(source)
        component: dict[int, float] = defaultdict(float)
        allowed = max(0.0, 1.0 - max(0.0, float(belief.unknown_probability)))
        total = sum(max(0.0, float(probability)) for probability in belief.probabilities.values())
        scale = min(1.0, allowed / total) if total else 0.0
        for state, probability in belief.probabilities.items():
            value = graph.evaluate(state)
            if value.kind == "impartial":
                component[int(value.nimber)] += max(0.0, float(probability)) * scale
        combined: dict[int, float] = defaultdict(float)
        for left, left_probability in distribution.items():
            for right, right_probability in component.items():
                combined[left ^ right] += left_probability * right_probability
        distribution = dict(combined)
    known = sum(distribution.values())
    return NimberBelief(dict(sorted(distribution.items())), max(0.0, 1.0 - known))


def parse_cartstate(state: State):
    """Recover (depths, controls, teams) from a responder cartstate key.

    The responder keys the cart subgame by
        (map_key, episode, k, depth_levels, controls)
    (`strat_responder.py`, the `game_state` tuple). Anything that does not have
    that shape is not a cartstate and gets no closed form.
    """
    if not isinstance(state, tuple) or len(state) != 5:
        return None
    _, _, k, depths, controls = state
    if not isinstance(k, int) or not isinstance(depths, (tuple, list)) or not isinstance(controls, (tuple, list)):
        return None
    if len(depths) != len(controls):
        return None
    try:
        depths = [int(value) for value in depths]
        controls = [int(value) for value in controls]
    except (TypeError, ValueError):
        return None
    return depths, controls, list(range(max(1, k)))


class EmpiricalTransitionGraph:
    """Options learned by watching real transitions.

    A watched graph is incomplete until every reachable state has been marked
    complete, which for the cart subgame only happens when a cart is delivered.
    Where the position is a cartstate the option graph does not have to be
    watched at all -- `evaluate_cartstate` writes it down in closed form -- so
    that is what `evaluate` falls back to instead of reporting "incomplete
    option graph" for the whole match (AGENDA B11).
    """

    def __init__(self, roles: Iterable[Role], cart_levels: int = 8) -> None:
        self.roles = _ordered(roles)
        self.cart_levels = int(cart_levels)
        self._counts: dict[State, dict[Role, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self._complete: set[tuple[State, Role]] = set()

    def observe(self, state: State, successor: State, role: Role, weight: float = 1.0) -> None:
        self.roles = _ordered((*self.roles, role))
        self._counts[state][role][successor] += max(0.0, float(weight))

    def mark_complete(self, state: State, role: Role | None = None) -> None:
        if role is not None:
            self.roles = _ordered((*self.roles, role))
        for current_role in self.roles if role is None else (role,):
            self._complete.add((state, current_role))

    def observe_terminal(self, state: State) -> None:
        self.mark_complete(state)

    def transition_probabilities(self, state: State, role: Role) -> dict[State, float]:
        counts = self._counts.get(state, {}).get(role, Counter())
        total = sum(counts.values())
        return {successor: count / total for successor, count in counts.items()} if total else {}

    def snapshot(self) -> FiniteGameGraph:
        options = {
            state: {role: tuple(counts) for role, counts in by_role.items()}
            for state, by_role in self._counts.items()
        }
        return FiniteGameGraph(options, self.roles, self._complete)

    def evaluate(self, state: State) -> GameValue:
        value = self.snapshot().evaluate(state)
        if value.kind != "unresolved":
            return value
        parsed = parse_cartstate(state)
        if parsed is None:
            return value
        depths, controls, teams = parsed
        closed = evaluate_cartstate(depths, controls, teams, self.cart_levels)
        return GameValue(closed.kind, closed.nimber, closed.role_values,
                         closed.reason or "closed-form cart option graph")



# =============================================================================
#  The cart subgame's option graph, in closed form.
# -----------------------------------------------------------------------------
#  AGENDA B11. `EmpiricalTransitionGraph` learns options by WATCHING transitions,
#  so `FiniteGameGraph.evaluate` can only ever say "incomplete option graph"
#  until `observe_terminal` has been called on every reachable state. On the real
#  Game-2 run no cart was ever delivered, `mark_complete` was therefore never
#  reached, and the CGT value came back
#      {"kind": "unresolved", "nimber": null, "reason": "incomplete option graph"}
#  on 228 of 228 lines. Nothing was wrong with the evaluator: it was being asked
#  to price a graph that had no edges.
#
#  The cart subgame does not need to be observed. SPEC 1 calls it
#      "a nim-like counting-game-over-payload-carts"
#  and its options follow from the cart rules, so they are enumerated here and
#  every state is complete by construction.
#
#  ONE CART = ONE COMPONENT of a disjunctive sum. Its position is the number of
#  levels still to be covered, r = levels - depth, floored by the cart's banked
#  point (score is monotone: sv_payload.qc banks and never un-banks).
#
#    * A NEUTRAL cart (no controlling team) is IMPARTIAL: the cylinder occupancy
#      law (sv_payload.qc `payload_occupancy`) lets any team present move it, so
#      every role has the same options -- reduce r to any smaller value. That is
#      a Nim heap, and backward induction over mex reproduces Grundy(r) = r, so
#      the nim-sum in game.py is derived here rather than asserted.
#
#    * A CONTROLLED cart is PARTIZAN and gets NO nimber. Regime A/B of the cart
#      velocity law (AGENDA R2) gives the controlling team and its opponents
#      genuinely different moves: the color team advances the cart, an opponent
#      reverses it toward the origin, where it recolors neutral (sv_payload.qc
#      "regressed to origin, recolored neutral"). Left and Right options differ,
#      a Grundy value does not exist, and none is invented (AGENDA B10).
#
#  So the sum resolves to an exact nimber exactly when every cart is neutral,
#  and to an explicit "partizan" otherwise. Neither answer is "incomplete".
# =============================================================================

NEUTRAL = None


def _neutral_cart_graph(levels: int) -> FiniteGameGraph:
    # Impartial: one role, options of r are every strictly smaller position.
    return FiniteGameGraph.impartial({r: tuple(range(r)) for r in range(levels + 1)})


def _controlled_cart_graph(levels: int, holder: Role, teams: Sequence[Role]) -> FiniteGameGraph:
    # Partizan: the holder advances (r decreases toward delivery), everyone else
    # reverses (r increases back toward the origin). Positions are r = levels - depth.
    roles = tuple(teams) or (holder,)
    options: dict[State, dict[Role, tuple[State, ...]]] = {}
    for r in range(levels + 1):
        by_role: dict[Role, tuple[State, ...]] = {}
        for role in roles:
            if role == holder:
                by_role[role] = tuple(range(r))
            else:
                by_role[role] = tuple(range(r + 1, levels + 1))
        options[r] = by_role
    return FiniteGameGraph.partizan(options, roles)


def cart_components(
    depths: Sequence[int],
    controls: Sequence[Role],
    teams: Sequence[Role],
    levels: int,
    floors: Sequence[int] | None = None,
) -> tuple[list[FiniteGameGraph], list[State]]:
    """Closed-form option graphs and positions for one cartstate.

    `controls[c]` is the controlling team, or None / a negative index for a
    neutral cart (the engine writes 0 for uncontrolled, the responder maps it to
    -1). `floors[c]` is the cart's banked level, which its position can never
    fall back below.
    """
    graphs, states = [], []
    floors = list(floors) if floors is not None else [0] * len(depths)
    for index, depth in enumerate(depths):
        holder = controls[index] if index < len(controls) else NEUTRAL
        if holder is not None and not isinstance(holder, str) and holder < 0:
            holder = NEUTRAL
        floor = max(0, min(int(floors[index]), levels))
        span = max(0, levels - floor)
        position = max(0, min(int(depth) - floor, span))
        remaining = span - position
        if holder is NEUTRAL:
            graphs.append(_neutral_cart_graph(span))
        else:
            graphs.append(_controlled_cart_graph(span, holder, teams))
        states.append(remaining)
    return graphs, states


def evaluate_cartstate(
    depths: Sequence[int],
    controls: Sequence[Role],
    teams: Sequence[Role],
    levels: int,
    floors: Sequence[int] | None = None,
) -> GameValue:
    """Price a real server cartstate as a disjunctive sum of cart components."""
    if not list(depths):
        return GameValue("impartial", 0, {})
    graphs, states = cart_components(depths, controls, teams, levels, floors)
    return disjunctive_sum_value(graphs, states)


__all__ = [
    "ComponentBelief",
    "EmpiricalTransitionGraph",
    "FiniteGameGraph",
    "GameValue",
    "NimberBelief",
    "RoleValue",
    "belief_nimber_distribution",
    "cart_components",
    "parse_cartstate",
    "evaluate_cartstate",
    "disjunctive_sum_options",
    "disjunctive_sum_value",
    "mex",
]
