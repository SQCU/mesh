from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .game import succession, team_nimbers
from .qkv import QKVProjector, QKVShapes, team_pool

X_WIDTH = 16
BELIEF_WIDTH = 8
INSTRUMENT_WIDTH = 16
RELATION_WIDTH = 16
HIERARCHY_WIDTH = 8

__all__ = [
    "StrategyState",
    "ForwardResult",
    "StrategyEstimator",
    "state_from_cartsim",
    "state_with_observations",
    "state_with_instruments",
    "X_WIDTH",
    "BELIEF_WIDTH",
    "INSTRUMENT_WIDTH",
    "HIERARCHY_WIDTH",
    "RELATION_WIDTH",
]


def _mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("StrategyEstimator.forward requires mlx") from exc
    return mx


@dataclass
class StrategyState:
    x: np.ndarray
    beta: np.ndarray
    z: np.ndarray
    relation: np.ndarray
    hierarchy: np.ndarray
    winner_mask: np.ndarray
    w: np.ndarray
    carts: list
    teams: Sequence
    team_of: Sequence[int]
    eligible: Optional[np.ndarray] = None


class ForwardResult(NamedTuple):
    action: object
    logpi: object
    value: object
    dw_dt: object
    w_next: object
    diag_k: object
    appetite: object
    winner_value: object
    loser_value: object
    q_team: object


class StrategyEstimator:
    def __init__(
        self,
        shapes: QKVShapes,
        *,
        delta: float = 0.5,
        temperature: float = 1.0,
        hidden: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        mx = _mlx()
        from .head import MixingHead
        from .value import StrategyValue

        self.shapes = shapes
        self.delta = float(delta)
        self.temperature = float(temperature)
        self.k = shapes.k_teams
        self.M = shapes.j_instruments
        self.l = shapes.l_players
        self._seed = seed
        self._rng_counter = 0
        if seed is not None:
            mx.random.seed(seed)
        self.qkv = QKVProjector(shapes, seed=seed)
        self.head = MixingHead(hidden=hidden)
        self.d_intermediate = 2 * shapes.d + 4 + HIERARCHY_WIDTH
        self.value = StrategyValue(self.d_intermediate)
        self.last: Optional[ForwardResult] = None

    @classmethod
    def for_cartsim(
        cls,
        sim,
        *,
        d: int = 16,
        d_v: int = 16,
        delta: float = 0.5,
        temperature: float = 1.0,
        hidden: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "StrategyEstimator":
        shapes = QKVShapes(
            k_teams=sim.k,
            j_instruments=sim.M,
            l_players=sim.l,
            d_x=X_WIDTH,
            d_beta=BELIEF_WIDTH,
            d_z=INSTRUMENT_WIDTH,
            d=d,
            d_v=d_v,
        )
        return cls(
            shapes,
            delta=delta,
            temperature=temperature,
            hidden=hidden,
            seed=seed,
        )

    def forward(self, state: StrategyState) -> ForwardResult:
        mx = _mlx()
        from .dpp import dpp_marginals
        from .head import integrate_weights, sample_strategy
        from .value import select_role_value

        x = mx.stop_gradient(mx.array(np.asarray(state.x, dtype=np.float32)))
        beta = mx.stop_gradient(mx.array(np.asarray(state.beta, dtype=np.float32)))
        z = mx.stop_gradient(mx.array(np.asarray(state.z, dtype=np.float32)))
        relation = mx.stop_gradient(mx.array(np.asarray(state.relation, dtype=np.float32)))
        hierarchy = mx.stop_gradient(mx.array(np.asarray(state.hierarchy, dtype=np.float32)))
        winner_mask = mx.stop_gradient(mx.array(np.asarray(state.winner_mask, dtype=bool)))
        w = mx.stop_gradient(mx.array(np.asarray(state.w, dtype=np.float32)))
        q = self.qkv.query(x, beta)
        keys = self.qkv.key(z)
        q_team = team_pool(q, list(state.team_of), len(state.teams))
        team_index = mx.array(np.asarray(state.team_of, dtype=np.int32))
        q_context = q + q_team[team_index]
        scores = q_context @ keys.T
        appetite = mx.logaddexp(scores, mx.zeros_like(scores))
        eligible = None
        if state.eligible is not None:
            eligible = mx.stop_gradient(mx.array(np.asarray(state.eligible, dtype=bool)))
            appetite = mx.where(eligible, appetite, mx.zeros_like(appetite))
        team_diag = []
        for team in range(len(state.teams)):
            team_rows = team_index == team
            denominator = mx.maximum(mx.sum(team_rows), 1)
            quality = mx.sum(appetite * team_rows[:, None], axis=0) / denominator
            if eligible is not None:
                available = mx.any(eligible & team_rows[:, None], axis=0)
                quality = mx.where(available, quality, mx.zeros_like(quality))
            team_diag.append(dpp_marginals(quality, keys))
        diag_k = mx.stop_gradient(mx.stack(team_diag)[team_index])
        dw_dt = self.head(diag_k, appetite, relation)
        w_next = integrate_weights(w, dw_dt, self.delta)
        if eligible is not None:
            w_next = mx.where(eligible, w_next, mx.full(w_next.shape, -1e9))
        key = None
        if self._seed is not None:
            key = mx.random.key(self._seed + self._rng_counter)
            self._rng_counter += 1
        action, logpi = sample_strategy(w_next, self.temperature, key)
        stats = mx.concatenate(
            [
                mx.mean(appetite, axis=1, keepdims=True),
                mx.max(appetite, axis=1, keepdims=True),
                mx.mean(dw_dt, axis=1, keepdims=True),
                mx.max(dw_dt, axis=1, keepdims=True),
            ],
            axis=1,
        )
        rows = mx.concatenate([q, q_team[team_index], stats, hierarchy], axis=1)
        winner_value, loser_value = self.value(rows)
        value = select_role_value(winner_value, loser_value, winner_mask)
        result = ForwardResult(
            action,
            logpi,
            value,
            dw_dt,
            w_next,
            diag_k,
            appetite,
            winner_value,
            loser_value,
            q_team,
        )
        self.last = result
        return result

    def learned_params(self) -> dict:
        return {
            "qkv": self.qkv.learned_params(),
            "head": self.head.parameters(),
            "value": self.value.parameters(),
        }


def _hierarchy_rows(carts, teams, team_of, L):
    nimbers = team_nimbers(carts, teams)
    order = succession(carts, teams)
    denial = {team: value for team, value in order}
    winner = order[0][0] if order else None
    scale = float(max(1, max(nimbers.values(), default=0), L))
    total_depth = float(max(1, sum(cart.depth for cart in carts)))
    rows = np.zeros((len(team_of), HIERARCHY_WIDTH), dtype=np.float32)
    masks = np.zeros(len(team_of), dtype=bool)
    for p, team in enumerate(team_of):
        own = float(nimbers.get(team, 0))
        rivals = [float(nimbers.get(other, 0)) for other in teams if other != team]
        rival_max = max(rivals, default=0.0)
        rival_mean = sum(rivals) / max(1, len(rivals))
        below = sum(own > rival for rival in rivals)
        above = sum(own < rival for rival in rivals)
        rows[p] = (
            own / scale,
            rival_max / scale,
            rival_mean / scale,
            (own - rival_mean) / scale,
            (below - above) / max(1, len(rivals)),
            float(team == winner),
            float(denial.get(team, 0)) / total_depth,
            1.0 / max(1, len(teams)),
        )
        masks[p] = team == winner
    return rows, masks


def state_from_cartsim(sim, cstate, *, w: Optional[np.ndarray] = None) -> StrategyState:
    from .cartsim import decode_instrument, to_carts

    k, j, l, M = sim.k, sim.j, sim.l, sim.M
    teams = list(range(k))
    team_of = np.asarray(sim.team_of, dtype=np.int64)
    carts = to_carts(cstate)
    nimbers = team_nimbers(carts, teams)
    winner = sim.projected_winner(cstate)
    Lf = float(max(1, sim.L))
    x = np.zeros((l, X_WIDTH), dtype=np.float32)
    beta = np.zeros((l, BELIEF_WIDTH), dtype=np.float32)
    z = np.zeros((M, INSTRUMENT_WIDTH), dtype=np.float32)
    relation = np.zeros((l, M, RELATION_WIDTH), dtype=np.float32)
    team_sizes = np.bincount(team_of, minlength=k)
    for p, team in enumerate(team_of):
        owned = [c for c in range(j) if int(cstate.control[c]) == team]
        own_depths = [float(cstate.pos[c]) for c in owned]
        rivals = [float(v) for other, v in nimbers.items() if other != team]
        x[p] = (
            max(own_depths, default=0.0) / Lf,
            sum(own_depths) / max(1, len(own_depths)) / Lf,
            float(cstate.banked[team]) / max(1.0, Lf * j),
            float(team == winner),
            float(nimbers.get(int(team), 0)) / Lf,
            max(rivals, default=0.0) / Lf,
            float(team_sizes[team]) / max(1, l),
            1.0 / max(1, k),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        )
        cart_rows = np.zeros((j, 4), dtype=np.float32)
        for c in range(j):
            control = int(cstate.control[c])
            cart_rows[c] = (
                float(cstate.pos[c]) / Lf,
                float(control == team),
                float(control >= 0 and control != team),
                float(control < 0),
            )
        beta[p] = np.concatenate([cart_rows.mean(axis=0), cart_rows.max(axis=0)])
    for m in range(M):
        kind, cart = decode_instrument(m, j)
        kind_index = 0 if kind == "push_cart" else 1 if kind == "suppress_cart" else 7
        z[m, kind_index] = 1.0
        z[m, 8] = 1.0
        z[m, 13] = 1.0
        if cart >= 0:
            progress = float(cstate.pos[cart]) / Lf
            z[m, 9] = 1.0 - progress if kind == "push_cart" else progress
            z[m, 10] = progress
            z[m, 15] = float(cstate.control[cart] >= 0)
        for p, team in enumerate(team_of):
            control = int(cstate.control[cart]) if cart >= 0 else -1
            relation[p, m, 0] = float(control == team)
            relation[p, m, 1] = float(control >= 0 and control != team)
            relation[p, m, 2] = 1.0
            relation[p, m, 8] = 1.0
            relation[p, m, 9] = 1.0
            relation[p, m, 13] = float(cstate.pos[cart]) / Lf if cart >= 0 else 0.0
            relation[p, m, 14] = 1.0
    hierarchy, winner_mask = _hierarchy_rows(carts, teams, team_of, sim.L)
    if w is None:
        w = np.zeros((l, M), dtype=np.float32)
    else:
        w = np.asarray(w, dtype=np.float32).reshape(l, M)
    return StrategyState(
        x,
        beta,
        z,
        relation,
        hierarchy,
        winner_mask,
        w,
        carts,
        teams,
        team_of.tolist(),
    )


def state_with_observations(state: StrategyState, rows, columns) -> StrategyState:
    rows = np.asarray(rows, dtype=np.float32)
    x = state.x.copy()
    velocity = np.sqrt(
        np.square(rows[:, columns["VEL_X"]])
        + np.square(rows[:, columns["VEL_Y"]])
        + np.square(rows[:, columns["VEL_Z"]])
    )
    x[:, 8:] = np.stack(
        [
            rows[:, columns["HEALTH"]] / 100.0,
            rows[:, columns["ARMOR"]] / 100.0,
            rows[:, columns["AMMO"]],
            velocity,
            rows[:, columns["POWER"]] / 30.0,
            rows[:, columns["TSS"]] / 30.0,
            rows[:, columns["ALIVE"]],
            rows[:, columns["CONTROL"]],
        ],
        axis=1,
    )
    return StrategyState(
        x,
        state.beta,
        state.z,
        state.relation,
        state.hierarchy,
        state.winner_mask,
        state.w,
        state.carts,
        state.teams,
        state.team_of,
        state.eligible,
    )


def state_with_instruments(state: StrategyState, batch, w=None) -> StrategyState:
    players = state.x.shape[0]
    instruments = batch.descriptors.shape[0]
    weights = np.zeros((players, instruments), dtype=np.float32) if w is None else np.asarray(
        w, dtype=np.float32
    ).reshape(players, instruments)
    return StrategyState(
        state.x,
        state.beta,
        np.asarray(batch.descriptors, dtype=np.float32),
        np.asarray(batch.relations, dtype=np.float32),
        state.hierarchy,
        state.winner_mask,
        weights,
        state.carts,
        state.teams,
        state.team_of,
        np.asarray(batch.eligible, dtype=bool),
    )
