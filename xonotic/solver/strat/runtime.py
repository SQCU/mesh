from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .game_value import evaluate_cartstate
from .instruments import (
    CartTarget,
    Participant,
    build_instruments,
    weights_from_table,
)
from payload.tools.strategy_io_schema import CS, OBS

BEHAVIOR_MEASURE_NAMES = (
    "enemy_damage_dealt",
    "enemy_damage_taken",
    "enemy_kills",
    "deaths",
    "pickups",
    "cart_push",
    "cart_contest",
)
POLICY_ACTOR_WEIGHT = 4.0
POLICY_ENTROPY_WEIGHT = 0.01
POLICY_ENTROPY_FLOOR = 1.0
POLICY_RATIO_CLIP = 0.2
SPARSE_REWARD_CONTRACT = {
    "source": "cart_state_transition",
    "winner_rows": {"event": "projected_winner_loses_role", "value": -1.0},
    "loser_rows": {"event": "upward_loser_rank_flip", "value": 1.0},
    "otherwise": 0.0,
    "terminal_reward": None,
}
SPARSE_REWARD_FINGERPRINT = hashlib.sha256(
    json.dumps(
        SPARSE_REWARD_CONTRACT, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest()[:16]

@dataclass(frozen=True)
class GameContext:
    teams: tuple[int, ...]
    team_of: tuple[int, ...]
    levels: int = 256

@dataclass
class CartSnapshot:
    pos: np.ndarray
    control: np.ndarray
    levels: int = 256

@dataclass(frozen=True)
class RuntimeFrame:
    context: GameContext
    cartstate: CartSnapshot
    batch: object
    semantics: np.ndarray
    winner_mask: np.ndarray
    weights: np.ndarray
    projected: object
    succession: tuple
    game_value: object

def build_runtime_frame(rows, cart_rows, targets, k, weight_table=None, levels=256,
                        navigation=None):
    obs = np.asarray(rows, dtype=np.float32)
    carts_raw = np.asarray(cart_rows, dtype=np.float32)
    team_of = tuple((np.asarray(obs[:, OBS["TEAM"]], dtype=np.int64) - 1).tolist())
    context = GameContext(tuple(range(int(k))), team_of, int(levels))
    controls_raw = np.asarray(carts_raw[:, CS["CONTROL_TEAM"]], dtype=np.int64)
    controls = np.where(controls_raw >= 1, controls_raw - 1, -1)
    path_position = np.asarray(carts_raw[:, CS["PATH_POSITION"]], dtype=np.float32)
    path_length = np.asarray(carts_raw[:, CS["PATH_LENGTH"]], dtype=np.float32)
    depth = np.divide(path_position, path_length, out=np.zeros_like(path_position), where=path_length != 0)
    snapshot = CartSnapshot(
        depth, controls.astype(np.int64), int(levels),
    )
    participants = tuple(
        Participant(
            int(row[OBS["ID"]]), int(row[OBS["TEAM"]]),
            (int(row[OBS["CELL_X"]]), int(row[OBS["CELL_Y"]])),
            tuple(float(value) for value in row[OBS["POS_X"]:OBS["POS_Z"] + 1]),
            float(row[OBS["ALIVE"]]), float(row[OBS["HEALTH"]]),
            float(row[OBS["ARMOR"]]),
            tuple(float(row[OBS[name]]) for name in (
                "AMMO_SHELLS", "AMMO_BULLETS", "AMMO_ROCKETS",
                "AMMO_CELLS", "AMMO_PLASMA", "AMMO_FUEL",
            )),
            float(row[OBS["SPAWN_TIME"]]), float(row[OBS["ENGINE_TIME"]]),
        )
        for row in obs
    )
    cart_targets = tuple(
        CartTarget(
            int(row[CS["ID"]]), int(row[CS["CONTROL_TEAM"]]),
            float(row[CS["PATH_POSITION"]]), float(row[CS["PATH_LENGTH"]]),
            float(row[CS["SPEED"]]),
            tuple(float(value) for value in row[CS["POS_X"]:CS["POS_Z"] + 1]),
        )
        for row in carts_raw
    )
    items, rivals, cells = targets
    batch = build_instruments(
        participants, cart_targets, items, rivals, cells, navigation=navigation,
    )
    game_value = formal_game_value(context, snapshot)
    semantics, winner_mask = hierarchy_rows(context, snapshot, game_value)
    return RuntimeFrame(
        context, snapshot, batch, semantics, winner_mask,
        weights_from_table(batch, weight_table),
        game_value.projected_role,
        game_value.succession,
        game_value,
    )

def formal_game_value(context: GameContext, snapshot: CartSnapshot):
    depths = [
        int(np.floor(np.clip(position, 0, 1) * snapshot.levels))
        for position in snapshot.pos
    ]
    controls = [int(control) for control in snapshot.control]
    return evaluate_cartstate(depths, controls, context.teams, snapshot.levels)

def formal_value_record(value, state, levels):
    return {
        "nimber": None if value.nimber is None else int(value.nimber),
        "projected_role": None if value.projected_role is None else int(value.projected_role),
        "portfolio_nimbers": {
            str(role): int(number) for role, number in value.portfolio_nimbers.items()
        },
        "role_ranks": {
            str(role): int(rank) for role, rank in value.role_ranks.items()
        },
        "succession": [
            [int(role), int(amount)] for role, amount in value.succession
        ],
        "levels": int(levels),
        "state": state,
        "mobility": {
            str(role): int(row.mobility) for role, row in value.role_values.items()
        },
        "option_enumeration_mass": {
            str(role): int(row.enumerated_mass) for role, row in value.role_values.items()
        },
        "reachable_state_mass": int(value.reachable_state_mass),
        "reachable_role_state_mass": int(value.reachable_role_state_mass),
        "enumerated_role_state_mass": int(value.enumerated_role_state_mass),
        "role_option_symmetric_difference_mass": int(
            value.role_option_symmetric_difference_mass
        ),
        "cycle_state_mass": int(value.cycle_state_mass),
    }

def formal_projection_record(value, teams):
    return {
        "PW": 0 if value.projected_role is None else int(value.projected_role) + 1,
        "SUCC": [
            [int(role) + 1, float(amount)] for role, amount in value.succession
        ],
        "loser_ranks": [int(value.role_ranks.get(role, 0)) for role in teams],
    }

def winner(context: GameContext, snapshot: CartSnapshot):
    return formal_game_value(context, snapshot).projected_role

def loser_ranks(context: GameContext, snapshot: CartSnapshot, value=None) -> np.ndarray:
    value = value or formal_game_value(context, snapshot)
    teams = np.asarray(context.teams, dtype=np.int64)
    ranks = np.zeros(len(context.teams), dtype=np.int64)
    ranks[teams] = np.asarray([value.role_ranks.get(team, 0) for team in context.teams], dtype=np.int64)
    return ranks

def hierarchy_rows(context: GameContext, snapshot: CartSnapshot, value=None):
    value = value or formal_game_value(context, snapshot)
    nimbers = value.portfolio_nimbers
    order = value.succession
    denial = {team: amount for team, amount in order}
    current = value.projected_role
    scale = float(max(1, snapshot.levels, max(nimbers.values(), default=0)))
    total = float(max(1, sum(
        int(np.floor(np.clip(position, 0, 1) * snapshot.levels))
        for position in snapshot.pos
    )))
    rows = np.zeros((len(context.team_of), 8), dtype=np.float32)
    mask = np.zeros(len(context.team_of), dtype=bool)
    for player, team in enumerate(context.team_of):
        own = float(nimbers.get(team, 0))
        rivals = [float(nimbers.get(other, 0)) for other in context.teams if other != team]
        center = float(np.mean(rivals)) if rivals else 0.0
        rows[player] = (
            own / scale,
            max(rivals, default=0.0) / scale,
            center / scale,
            (own - center) / scale,
            (sum(own > rival for rival in rivals) - sum(own < rival for rival in rivals)) / max(1, len(rivals)),
            float(team == current),
            float(denial.get(team, 0)) / total,
            1.0 / max(1, len(context.teams)),
        )
        mask[player] = team == current
    return rows, mask

def role_rewards(context: GameContext, before: CartSnapshot, after: CartSnapshot) -> np.ndarray:
    before_value = formal_game_value(context, before)
    after_value = formal_game_value(context, after)
    before_winner = before_value.projected_role
    after_winner = after_value.projected_role
    before_ranks = loser_ranks(context, before, before_value)
    after_ranks = loser_ranks(context, after, after_value)
    teams = np.asarray(context.teams, dtype=np.int64)
    winner_mask = teams == before_winner
    rank_flip = after_ranks[teams] > before_ranks[teams]
    team_rewards = np.where(
        winner_mask,
        -float(after_winner != before_winner),
        rank_flip.astype(np.float32),
    ).astype(np.float32)
    return team_rewards[np.asarray(context.team_of, dtype=np.int64)]
