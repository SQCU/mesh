from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game_value import CartSnapshot, GameContext, formal_game_value, hierarchy_rows
from payload.tools.strategy_io_schema import CS, OBS, TS

BEHAVIOR_MEASURE_NAMES = (
    "enemy_damage_dealt",
    "enemy_damage_taken",
    "enemy_kills",
    "deaths",
    "pickups",
    "cart_push",
    "cart_contest",
)

@dataclass(frozen=True)
class RuntimeFrame:
    context: GameContext
    cartstate: CartSnapshot
    semantics: np.ndarray
    game_value: object

def build_runtime_frame(rows, cart_rows, team_rows):
    obs = np.asarray(rows, dtype=np.float32)
    carts_raw = np.asarray(cart_rows, dtype=np.float32)
    team_of = tuple((np.asarray(obs[:, OBS["TEAM"]], dtype=np.int64) - 1).tolist())
    context = GameContext(tuple(range(len(team_rows))), team_of)
    team_rows = np.asarray(team_rows, dtype=np.float32)
    controls_raw = np.asarray(carts_raw[:, CS["CONTROL_TEAM"]], dtype=np.int64)
    controls = np.where(controls_raw >= 1, controls_raw - 1, -1)
    path_position = np.asarray(carts_raw[:, CS["PATH_POSITION"]], dtype=np.float32)
    path_length = np.asarray(carts_raw[:, CS["PATH_LENGTH"]], dtype=np.float32)
    depth = np.divide(path_position, path_length, out=np.zeros_like(path_position), where=path_length != 0)
    snapshot = CartSnapshot(
        depth, controls.astype(np.int64),
        carts_raw[:, CS["CHECKPOINT_DEPTH"]].astype(np.int64),
        carts_raw[:, CS["CHECKPOINT_COUNT"]].astype(np.int64),
        team_rows[:, TS["SCORE"]], team_rows[:, TS["CHECKPOINTS_HELD"]].astype(np.int64),
        float(team_rows[0, TS["SCORE_LIMIT"]]), float(team_rows[0, TS["CHECKPOINT_RATE"]]),
        int(team_rows[0, TS["EPISODE"]]), bool(team_rows[0, TS["FINISHED"]]),
    )
    game_value = formal_game_value(context, snapshot)
    semantics, _ = hierarchy_rows(context, snapshot, game_value)
    return RuntimeFrame(context, snapshot, semantics, game_value)
