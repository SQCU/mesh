from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .featurize import BELIEF_RANK
from .game import team_nimbers
from .qkv import QKVProjector, QKVShapes
from .runtime import CartSnapshot, GameContext, carts, hierarchy_rows, winner

X_WIDTH = 48
BELIEF_WIDTH = BELIEF_RANK   # one definition: featurize.PHI's row count
INSTRUMENT_WIDTH = 16
RELATION_WIDTH = 16
HIERARCHY_WIDTH = 8
IR_WIDTH = 128


def _mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("StrategyEstimator requires mlx") from exc
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
    score: object
    diag_k: object
    appetite: object
    winner_value: object
    loser_value: object
    ir: object
    gram: object


class StrategyEstimator:
    def __init__(
        self,
        shapes: QKVShapes,
        *,
        delta: float = 0.5,
        temperature: float = 1.0,
        hidden: Optional[int] = None,
        seed: Optional[int] = None,
        anticipatory: bool = False,
        lead: float = 1.0,
    ) -> None:
        mx = _mlx()
        from .gram import GramSwiGLU
        from .head import MixingHead
        from .value import StrategyValue

        self.shapes = shapes
        self.delta = float(delta)
        self.temperature = float(temperature)
        self.anticipatory = bool(anticipatory)
        self.lead = float(lead)
        self._seed = seed
        self._rng_counter = 0
        if seed is not None:
            mx.random.seed(seed)
        self.qkv = QKVProjector(shapes, seed=seed)
        self.encoder = GramSwiGLU(shapes.d)
        self.head = MixingHead(hidden=hidden)
        self.value = StrategyValue(shapes.d)
        self.last: Optional[ForwardResult] = None

    @classmethod
    def for_runtime(
        cls,
        k: int,
        l: int,
        *,
        d: int = IR_WIDTH,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Build the operator for a match with ``k`` teams and ``l`` players.

        ``k`` and ``l`` are DESCRIPTIVE: they are recorded on ``QKVShapes`` and
        used nowhere else. No learned parameter's shape depends on them --
        ``W_q`` is ``(d, X_WIDTH + BELIEF_WIDTH)``, ``W_k`` is ``(d,
        INSTRUMENT_WIDTH)``, ``GramSwiGLU`` is sized by ``d``,
        ``MixingHead.in_dim`` is the fixed 21-feature row and the value probes
        are ``Linear(d, 1)``. Team / cart / player counts enter only as ROW
        COUNTS of the data. That is what lets ONE estimator (and one learner)
        span matches of different shape; `strat_responder.EstCache` re-proves it
        by comparing parameter trees at every newly seen ``(k, j, l)`` rather
        than taking this docstring's word for it.
        """
        return cls(
            QKVShapes(k, 1, l, X_WIDTH, BELIEF_WIDTH, INSTRUMENT_WIDTH, d),
            seed=seed,
            **kwargs,
        )

    def forward(self, state: StrategyState) -> ForwardResult:
        mx = _mlx()
        key = None
        if self._seed is not None:
            key = mx.random.key(self._seed + self._rng_counter)
            self._rng_counter += 1
        out = strategy_forward(self, state, state.w, key=key)
        result = ForwardResult(*(out[name] for name in ForwardResult._fields))
        self.last = result
        return result

    def learned_params(self) -> dict:
        return {
            "qkv": self.qkv.learned_params(),
            "encoder": self.encoder.parameters(),
            "head": self.head.parameters(),
            "value": self.value.parameters(),
        }


def strategy_forward(est, state, w_in, *, action=None, key=None, anticipatory=None, lead=None):
    mx = _mlx()
    from .dpp import dpp_marginals
    from .gram import edge_features
    from .head import integrate_weights, strategy_log_prob
    from .value import select_role_value

    anticipatory = est.anticipatory if anticipatory is None else anticipatory
    lead = est.lead if lead is None else lead
    x = mx.stop_gradient(mx.array(np.asarray(state.x, dtype=np.float32)))
    beta = mx.stop_gradient(mx.array(np.asarray(state.beta, dtype=np.float32)))
    z = mx.stop_gradient(mx.array(np.asarray(state.z, dtype=np.float32)))
    relation = mx.stop_gradient(mx.array(np.asarray(state.relation, dtype=np.float32)))
    winner_mask = mx.stop_gradient(mx.array(np.asarray(state.winner_mask, dtype=bool)))
    weights = mx.stop_gradient(mx.array(np.asarray(w_in, dtype=np.float32)))
    keys = est.qkv.key(z)
    projected = est.qkv.query(x, beta)
    edges = mx.stop_gradient(mx.array(edge_features(
        state.team_of, np.asarray(state.hierarchy), np.asarray(state.winner_mask))))
    ir, gram = est.encoder(projected, edges)
    appetite = mx.logaddexp(ir @ keys.T, mx.zeros((ir.shape[0], keys.shape[0])))
    eligible = None
    if state.eligible is not None:
        eligible = mx.stop_gradient(mx.array(np.asarray(state.eligible, dtype=bool)))
        appetite = mx.where(eligible, appetite, mx.zeros_like(appetite))
    team_index = mx.array(np.asarray(state.team_of, dtype=np.int32))
    team_diag = []
    for team in range(len(state.teams)):
        member = team_index == team
        quality = mx.sum(appetite * member[:, None], axis=0) / mx.maximum(mx.sum(member), 1)
        if eligible is not None:
            quality = mx.where(mx.any(eligible & member[:, None], axis=0), quality, 0.0)
        team_diag.append(dpp_marginals(quality, keys, method="inverse_diff"))
    diag_k = mx.stack(team_diag)[team_index]
    dw_dt = est.head(diag_k, appetite, relation)
    w_next = integrate_weights(weights, dw_dt, est.delta)
    if eligible is not None:
        w_next = mx.where(eligible, w_next, mx.zeros_like(w_next))
    score = w_next + lead * dw_dt if anticipatory else w_next
    if eligible is not None:
        score = mx.where(eligible, score, mx.full(score.shape, -1e9))
    if action is None:
        action = mx.random.categorical(score / est.temperature, axis=-1, key=key)
    logpi = strategy_log_prob(score, action, est.temperature)
    winner_value, loser_value = est.value(ir)
    value = select_role_value(winner_value, loser_value, winner_mask)
    return {
        "action": action,
        "logpi": logpi,
        "value": value,
        "dw_dt": dw_dt,
        "w_next": w_next,
        "score": score,
        "diag_k": diag_k,
        "appetite": appetite,
        "winner_value": winner_value,
        "loser_value": loser_value,
        "ir": ir,
        "gram": gram,
    }


def state_from_runtime(
    context: GameContext,
    snapshot: CartSnapshot,
    rows,
    columns,
    beta,
    batch,
    w=None,
) -> StrategyState:
    rows = np.asarray(rows, dtype=np.float32)
    players = len(context.team_of)
    instruments = len(batch.instruments)
    x = np.zeros((players, X_WIDTH), dtype=np.float32)
    cart_rows = carts(snapshot)
    nimbers = team_nimbers(cart_rows, context.teams)
    current = winner(context, snapshot)
    team_sizes = np.bincount(context.team_of, minlength=len(context.teams))
    scale = float(max(1, snapshot.levels))
    for player, team in enumerate(context.team_of):
        owned = [snapshot.pos[index] for index, control in enumerate(snapshot.control) if control == team]
        rivals = [nimbers[other] for other in context.teams if other != team]
        x[player, :8] = (
            max(owned, default=0.0) / scale,
            float(np.mean(owned)) / scale if owned else 0.0,
            float(nimbers.get(team, 0)) / scale,
            max(rivals, default=0.0) / scale,
            float(np.mean(rivals)) / scale if rivals else 0.0,
            team_sizes[team] / max(1, players),
            1.0 / max(1, len(context.teams)),
            float(team == current),
        )
    x[:, 8] = rows[:, columns["HEALTH"]] / 100.0
    x[:, 9] = rows[:, columns["ARMOR"]] / 100.0
    x[:, 10] = rows[:, columns["AMMO"]]
    x[:, 11:14] = rows[:, columns["POS_X"]:columns["POS_Z"] + 1]
    x[:, 14:17] = rows[:, columns["VEL_X"]:columns["VEL_Z"] + 1]
    x[:, 17] = rows[:, columns["POWER"]] / 30.0
    x[:, 18] = rows[:, columns["TSS"]] / 30.0
    x[:, 19] = rows[:, columns["ALIVE"]]
    x[:, 20] = rows[:, columns["CONTROL"]]
    x[:, 21] = rows[:, columns["NCART"]] / max(1, len(snapshot.pos))
    x[:, 22] = rows[:, columns["NCART_D"]]
    x[:, 23] = rows[:, columns["CELL"]] / 1024.0
    weapons = np.rint(rows[:, columns["WEAPONS"]]).astype(np.uint32)
    x[:, 24:48] = ((weapons[:, None] >> np.arange(24, dtype=np.uint32)) & 1).astype(np.float32)
    hierarchy, winner_mask = hierarchy_rows(context, snapshot)
    weights = np.zeros((players, instruments), dtype=np.float32) if w is None else np.asarray(w, dtype=np.float32).reshape(players, instruments)
    return StrategyState(
        x,
        np.asarray(beta, dtype=np.float32),
        np.asarray(batch.descriptors, dtype=np.float32),
        np.asarray(batch.relations, dtype=np.float32),
        hierarchy,
        winner_mask,
        weights,
        cart_rows,
        context.teams,
        context.team_of,
        np.asarray(batch.eligible, dtype=bool),
    )


__all__ = [
    "StrategyState",
    "ForwardResult",
    "StrategyEstimator",
    "strategy_forward",
    "state_from_runtime",
    "X_WIDTH",
    "BELIEF_WIDTH",
    "INSTRUMENT_WIDTH",
    "RELATION_WIDTH",
    "HIERARCHY_WIDTH",
    "IR_WIDTH",
]
