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
    for index, (graph, state) in enumerate(zip(graphs, states, strict=True)):
        for option in graph.options(state, role):
            successor = list(states)
            successor[index] = option
            moves.append(tuple(successor))
    return tuple(moves)


def disjunctive_sum_value(
    graphs: Sequence[FiniteGameGraph],
    states: Sequence[State],
) -> GameValue:
    values = [graph.evaluate(state) for graph, state in zip(graphs, states, strict=True)]
    if all(value.kind == "impartial" for value in values):
        return GameValue("impartial", reduce(xor, (value.nimber for value in values), 0), {})
    roles = _ordered(role for graph in graphs for role in graph.roles)
    role_values = {
        role: RoleValue(
            len(disjunctive_sum_options(graphs, states, role)),
            disjunctive_sum_options(graphs, states, role),
            all(graph.options_complete(state, role) for graph, state in zip(graphs, states, strict=True) if role in graph.roles),
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
    for graph, source in zip(graphs, beliefs, strict=True):
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


class EmpiricalTransitionGraph:
    def __init__(self, roles: Iterable[Role]) -> None:
        self.roles = _ordered(roles)
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
        return self.snapshot().evaluate(state)


__all__ = [
    "ComponentBelief",
    "EmpiricalTransitionGraph",
    "FiniteGameGraph",
    "GameValue",
    "NimberBelief",
    "RoleValue",
    "belief_nimber_distribution",
    "disjunctive_sum_options",
    "disjunctive_sum_value",
    "mex",
]
