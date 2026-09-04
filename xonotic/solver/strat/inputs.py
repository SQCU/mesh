from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["ChorusArrays", "assemble", "player_features", "scatter_gather_participants"]

def player_features(obs_rows: np.ndarray) -> np.ndarray:
    from payload.tools.strategy_io_schema import (
        OBS, OBS_WEAPON_COLUMNS, WEAPON_WORD_BITS, XAN_SCALAR_COLUMNS,
    )

    rows = np.asarray(obs_rows, dtype=np.float32)
    scalars = rows[:, [OBS[name] for name in XAN_SCALAR_COLUMNS]]
    words = rows[:, [OBS[name] for name in OBS_WEAPON_COLUMNS]].astype(np.int64)
    shifts = np.arange(WEAPON_WORD_BITS, dtype=np.int64)
    weapons = ((words[:, :, None] >> shifts) & 1).reshape(len(rows), -1).astype(np.float32)
    return np.concatenate((scalars, weapons), axis=1)

class ChorusArrays(NamedTuple):
    xan: np.ndarray
    zed: np.ndarray
    cell_slots: np.ndarray
    gigi: np.ndarray
    semantics: np.ndarray
    team_ids: np.ndarray
    weights: np.ndarray
    action_mass: np.ndarray
    delta: np.ndarray
    control_weight: np.ndarray
    exploration_weight: np.ndarray

def scatter_gather_participants(chorus, participant_ids, requested_ids):
    participant_ids = np.asarray(participant_ids, dtype=np.int64).reshape(-1)
    requested_ids = np.asarray(requested_ids, dtype=np.int64).reshape(-1)
    positions_by_id = {int(value): index for index, value in enumerate(participant_ids)}
    positions = np.asarray([positions_by_id.get(int(value), -1) for value in requested_ids])
    present = positions >= 0

    def rows(value, fill=0):
        value = np.asarray(value)
        out = np.full((len(requested_ids), *value.shape[1:]), fill, dtype=value.dtype)
        out[present] = value[positions[present]]
        return out

    return ChorusArrays(
        xan=rows(chorus.xan),
        zed=np.asarray(chorus.zed),
        cell_slots=np.asarray(chorus.cell_slots),
        gigi=rows(chorus.gigi),
        semantics=rows(chorus.semantics),
        team_ids=rows(chorus.team_ids),
        weights=rows(chorus.weights),
        action_mass=rows(chorus.action_mass, 1),
        delta=np.asarray(chorus.delta),
        control_weight=np.asarray(chorus.control_weight),
        exploration_weight=np.asarray(chorus.exploration_weight),
    ), present

def assemble(
    obs_rows: np.ndarray,
    batch,
    cell_slots: np.ndarray,
    gigi: np.ndarray,
    semantics: np.ndarray,
    weights: np.ndarray,
    delta: float,
    control_weight: float,
    exploration_weight: float,
    team_ids: np.ndarray,
) -> ChorusArrays:
    return ChorusArrays(
        xan=np.asarray(obs_rows, dtype=np.float32),
        zed=np.asarray(batch.descriptors, dtype=np.float32),
        cell_slots=np.asarray(cell_slots, dtype=np.float32),
        gigi=np.asarray(gigi, dtype=np.float32),
        semantics=np.asarray(semantics, dtype=np.float32),
        team_ids=np.asarray(team_ids, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        action_mass=np.asarray(batch.action_mass, dtype=np.float32),
        delta=np.asarray(delta, dtype=np.float32),
        control_weight=np.asarray(control_weight, dtype=np.float32),
        exploration_weight=np.asarray(exploration_weight, dtype=np.float32),
    )
