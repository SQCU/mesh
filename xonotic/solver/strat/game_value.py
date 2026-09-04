from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
import math
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
    enumerated_mass: int

@dataclass(frozen=True)
class GameValue:
    nimber: int | None
    role_values: Mapping[Role, RoleValue]
    reachable_state_mass: int
    reachable_role_state_mass: int
    enumerated_role_state_mass: int
    role_option_symmetric_difference_mass: int
    cycle_state_mass: int
    projected_role: Role | None = None
    portfolio_nimbers: Mapping[Role, int] = field(default_factory=dict)
    role_ranks: Mapping[Role, int] = field(default_factory=dict)
    succession: tuple[tuple[Role, int], ...] = ()

@dataclass(frozen=True)
class ComponentBelief:
    probabilities: Mapping[State, float]
    unknown_probability: float = 0.0

@dataclass(frozen=True)
class NimberBelief:
    probabilities: Mapping[int, float]
    undefined_probability: float

    def probability(self, nimber: int) -> float:
        return float(self.probabilities.get(int(nimber), 0.0))

def _ordered(values: Iterable[State]) -> tuple[State, ...]:
    return tuple(sorted(set(values), key=repr))

class FiniteGameGraph:
    def __init__(
        self,
        options_by_role: Mapping[State, Mapping[Role, Iterable[State]]],
        roles: Iterable[Role],
        enumerated_options: Iterable[tuple[State, Role]] | None = None,
    ) -> None:
        self.roles = _ordered(roles) or ("player",)
        self._options = {
            state: {role: _ordered(options) for role, options in by_role.items()}
            for state, by_role in options_by_role.items()
        }
        if enumerated_options is None:
            states = set(self._options)
            for by_role in self._options.values():
                for options in by_role.values():
                    states.update(options)
            enumerated_options = ((state, role) for state in states for role in self.roles)
        self._enumerated = set(enumerated_options)

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

    def option_enumeration_mass(self, state: State, role: Role) -> int:
        return int((state, role) in self._enumerated)

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

    def role_option_symmetric_difference_mass(self, states: Iterable[State]) -> int:
        mass = 0
        for state in states:
            options = [set(self.options(state, role)) for role in self.roles]
            incidence = Counter(option for role_options in options for option in role_options)
            mass += sum(count * (len(options) - count) for count in incidence.values())
        return mass

    def cycle_state_mass(self, states: Iterable[State]) -> int:
        mass = 0
        for root in states:
            seen = set()
            pending = list(self.options(root))
            while pending:
                current = pending.pop()
                if current == root:
                    mass += 1
                    break
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(self.options(current))
        return mass

    def role_values(self, state: State) -> dict[Role, RoleValue]:
        return {
            role: RoleValue(
                len(self.options(state, role)),
                self.option_enumeration_mass(state, role),
            )
            for role in self.roles
        }

    def evaluate(self, state: State) -> GameValue:
        reachable = self.reachable(state)
        reachable_role_state_mass = len(reachable) * len(self.roles)
        enumerated_role_state_mass = sum(
            self.option_enumeration_mass(current, role)
            for current in reachable for role in self.roles
        )
        difference_mass = self.role_option_symmetric_difference_mass(reachable)
        cycle_state_mass = self.cycle_state_mass(reachable)
        measurements = dict(
            role_values=self.role_values(state),
            reachable_state_mass=len(reachable),
            reachable_role_state_mass=reachable_role_state_mass,
            enumerated_role_state_mass=enumerated_role_state_mass,
            role_option_symmetric_difference_mass=difference_mass,
            cycle_state_mass=cycle_state_mass,
        )
        if enumerated_role_state_mass != reachable_role_state_mass or difference_mass or cycle_state_mass:
            return GameValue(nimber=None, **measurements)
        cache: dict[State, int] = {}

        def grundy(current: State) -> int:
            if current in cache:
                return cache[current]
            value = mex(grundy(option) for option in self.options(current, self.roles[0]))
            cache[current] = value
            return value

        return GameValue(nimber=grundy(state), **measurements)

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
    roles = _ordered(role for graph in graphs for role in graph.roles)
    role_values = {
        role: RoleValue(
            len(disjunctive_sum_options(graphs, states, role)),
            int(all(
                graph.option_enumeration_mass(state, role)
                for graph, state in zip(graphs, states) if role in graph.roles
            )),
        )
        for role in roles
    }
    component_masses = [value.reachable_state_mass for value in values]
    reachable_state_mass = math.prod(component_masses)
    reachable_role_state_mass = reachable_state_mass * len(roles)
    enumerated_role_state_mass = sum(
        all(
            role not in graph.roles or value.enumerated_role_state_mass == value.reachable_role_state_mass
            for graph, value in zip(graphs, values)
        ) for role in roles
    ) * reachable_state_mass
    difference_mass = sum(
        value.role_option_symmetric_difference_mass
        * math.prod(component_masses[:index] + component_masses[index + 1:])
        for index, value in enumerate(values)
    )
    cycle_state_mass = reachable_state_mass - math.prod(
        value.reachable_state_mass - value.cycle_state_mass for value in values
    )
    nimbers = [value.nimber for value in values]
    nimber = reduce(xor, nimbers, 0) if all(value is not None for value in nimbers) else None
    return GameValue(
        nimber, role_values, reachable_state_mass, reachable_role_state_mass,
        enumerated_role_state_mass, difference_mass, cycle_state_mass,
    )

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
            if value.nimber is not None:
                component[int(value.nimber)] += max(0.0, float(probability)) * scale
        combined: dict[int, float] = defaultdict(float)
        for left, left_probability in distribution.items():
            for right, right_probability in component.items():
                combined[left ^ right] += left_probability * right_probability
        distribution = dict(combined)
    known = sum(distribution.values())
    return NimberBelief(dict(sorted(distribution.items())), max(0.0, 1.0 - known))

def parse_cartstate(state: State):
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
    def __init__(self, roles: Iterable[Role], cart_levels: int = 8) -> None:
        self.roles = _ordered(roles)
        self.cart_levels = int(cart_levels)
        self._counts: dict[State, dict[Role, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self._enumerated: set[tuple[State, Role]] = set()

    def observe(self, state: State, successor: State, role: Role, weight: float = 1.0) -> None:
        self.roles = _ordered((*self.roles, role))
        self._counts[state][role][successor] += max(0.0, float(weight))

    def mark_enumerated(self, state: State, role: Role | None = None) -> None:
        if role is not None:
            self.roles = _ordered((*self.roles, role))
        for current_role in self.roles if role is None else (role,):
            self._enumerated.add((state, current_role))

    def observe_terminal(self, state: State) -> None:
        self.mark_enumerated(state)

    def transition_probabilities(self, state: State, role: Role) -> dict[State, float]:
        counts = self._counts.get(state, {}).get(role, Counter())
        total = sum(counts.values())
        return {successor: count / total for successor, count in counts.items()} if total else {}

    def snapshot(self) -> FiniteGameGraph:
        options = {
            state: {role: tuple(counts) for role, counts in by_role.items()}
            for state, by_role in self._counts.items()
        }
        return FiniteGameGraph(options, self.roles, self._enumerated)

    def evaluate(self, state: State) -> GameValue:
        value = self.snapshot().evaluate(state)
        if value.enumerated_role_state_mass == value.reachable_role_state_mass:
            return value
        parsed = parse_cartstate(state)
        if parsed is None:
            return value
        depths, controls, teams = parsed
        return evaluate_cartstate(depths, controls, teams, self.cart_levels)

NEUTRAL = None

def cart_projection(
    depths: Sequence[int],
    controls: Sequence[Role],
    teams: Sequence[Role],
) -> tuple[Role | None, dict[Role, int], dict[Role, int], tuple[tuple[Role, int], ...]]:
    roles = _ordered(teams)
    rows = [
        [controls[index] if index < len(controls) else NEUTRAL, max(0, int(depth))]
        for index, depth in enumerate(depths)
    ]

    def coordinates():
        values = {role: 0 for role in roles}
        for holder, depth in rows:
            if holder in values and depth:
                values[holder] ^= depth
        return values

    def leader(values):
        maximum = max(values.values(), default=0)
        found = [role for role, value in values.items() if value == maximum]
        return found[0] if maximum > 0 and len(found) == 1 else None

    values = coordinates()
    projected = leader(values)
    ranks = {
        role: sum(
            value > other
            for other_role, other in values.items()
            if other_role != role and other_role != projected
        )
        for role, value in values.items()
    }
    if projected is not None:
        ranks[projected] = max(0, len(roles) - 1)
    order = []
    current = projected
    if current is not None:
        order.append((current, 0))
        seen = {current}
        steps = 0
        previous = 0
        while steps <= sum(row[1] for row in rows) and len(seen) < len(roles):
            candidates = [
                (row[1], index) for index, row in enumerate(rows)
                if row[0] == current and row[1] > 0
            ]
            if not candidates:
                break
            _, index = max(candidates)
            rows[index][1] -= 1
            if rows[index][1] == 0:
                rows[index][0] = NEUTRAL
            steps += 1
            following = leader(coordinates())
            if following != current:
                if following is None:
                    break
                if following not in seen:
                    order.append((following, steps - previous))
                    seen.add(following)
                    previous = steps
                current = following
    return projected, values, ranks, tuple(order)

def _neutral_cart_graph(levels: int, teams: Sequence[Role]) -> FiniteGameGraph:
    return FiniteGameGraph.impartial(
        {r: tuple(range(r)) for r in range(levels + 1)},
        tuple(teams) or ("player",),
    )

def _controlled_cart_graph(levels: int, holder: Role, teams: Sequence[Role]) -> FiniteGameGraph:
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
            graphs.append(_neutral_cart_graph(span, teams))
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
    roles = _ordered(teams) or ("player",)
    projected, portfolio_nimbers, ranks, order = cart_projection(depths, controls, roles)
    if not list(depths):
        return GameValue(0, {}, 0, 0, 0, 0, 0, projected,
                         portfolio_nimbers, ranks, order)
    floors = list(floors) if floors is not None else [0] * len(depths)
    mobility = {role: 0 for role in roles}
    state_masses = []
    difference_masses = []
    cycle_masses = []
    nimber = 0
    nimber_mass = 0
    for index, depth in enumerate(depths):
        holder = controls[index] if index < len(controls) else NEUTRAL
        if holder is not None and not isinstance(holder, str) and holder < 0:
            holder = NEUTRAL
        floor = max(0, min(int(floors[index]), levels))
        span = max(0, levels - floor)
        progress = max(0, min(int(depth) - floor, span))
        remaining = span - progress
        if holder is NEUTRAL:
            state_mass = remaining + 1
            component_nimber = remaining
            for role in roles:
                mobility[role] += remaining
        elif holder not in roles:
            state_mass = span - remaining + 1
            component_nimber = span - remaining
            for role in roles:
                mobility[role] += span - remaining
        elif len(roles) == 1 or span == 0:
            state_mass = remaining + 1
            component_nimber = remaining
            mobility[holder] += remaining
        else:
            state_mass = span + 1
            component_nimber = None
            component_difference = (len(roles) - 1) * span * state_mass
            component_cycle = state_mass
            for role in roles:
                mobility[role] += remaining if role == holder else span - remaining
        if not (holder in roles and len(roles) > 1 and span > 0):
            component_difference = 0
            component_cycle = 0
        state_masses.append(state_mass)
        difference_masses.append(component_difference)
        cycle_masses.append(component_cycle)
        if component_nimber is not None:
            nimber ^= component_nimber
            nimber_mass += 1
    role_values = {
        role: RoleValue(value, 1) for role, value in mobility.items()
    }
    reachable_state_mass = math.prod(state_masses)
    reachable_role_state_mass = reachable_state_mass * len(roles)
    difference_mass = sum(
        mass * math.prod(state_masses[:index] + state_masses[index + 1:])
        for index, mass in enumerate(difference_masses)
    )
    cycle_state_mass = reachable_state_mass - math.prod(
        state_mass - cycle_mass
        for state_mass, cycle_mass in zip(state_masses, cycle_masses)
    )
    return GameValue(
        nimber if nimber_mass == len(depths) else None,
        role_values,
        reachable_state_mass,
        reachable_role_state_mass,
        reachable_role_state_mass,
        difference_mass,
        cycle_state_mass,
        projected,
        portfolio_nimbers,
        ranks,
        order,
    )

__all__ = [
    "ComponentBelief",
    "EmpiricalTransitionGraph",
    "FiniteGameGraph",
    "GameValue",
    "NimberBelief",
    "RoleValue",
    "belief_nimber_distribution",
    "cart_components",
    "cart_projection",
    "parse_cartstate",
    "evaluate_cartstate",
    "disjunctive_sum_options",
    "disjunctive_sum_value",
    "mex",
]
