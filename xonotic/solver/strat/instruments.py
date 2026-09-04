from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from payload.tools.strategy_io_schema import (
    CONTROL_WIDTH, INSTRUMENT_KIND, SC, TARGET_KIND,
)

class InstrumentKind(str, Enum):
    PUSH_CART = "push_cart"
    SUPPRESS_CART = "suppress_cart"
    CONTEST_POST = "contest_post"
    HUNT_RIVAL = "hunt_rival"
    EXPLORE_CELL = "explore_cell"
    SPAWN_TIMING = "spawn_timing"
    IDLE = "idle"

KINDS = tuple(InstrumentKind)
KIND_INDEX = {kind: i for i, kind in enumerate(KINDS)}
DESCRIPTOR_FIELDS = (
    "push_cart", "suppress_cart", "contest_post", "hunt_rival",
    "explore_cell", "spawn_timing", "idle",
    "available", "position_x", "position_y", "position_z",
    "path_position", "path_length", "speed", "respawn_time",
    "health", "observed_time",
)
DESCRIPTOR_WIDTH = len(DESCRIPTOR_FIELDS)
STOCK_WALK_SPEED = 400.0

@dataclass(frozen=True)
class Participant:
    participant_id: int
    team: int
    cell: tuple[int, int]
    position: tuple[float, float, float] | None = None
    alive: float = 1.0
    health: float = 0.0
    armor: float = 0.0
    ammo: tuple[float, ...] = ()
    spawn_time: float = 0.0
    engine_time: float = 0.0

@dataclass(frozen=True)
class CartTarget:
    cart_id: int
    control_team: int
    path_position: float
    path_length: float
    speed: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass(frozen=True)
class ItemTarget:
    item_id: int
    cell: tuple[int, int]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    available: float = 1.0
    respawn_time: float = 0.0
    observed_time: float = 0.0
    observed_by: tuple[int, ...] = ()

@dataclass(frozen=True)
class RivalTarget:
    rival_id: int
    team: int
    cell: tuple[int, int]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: float = 0.0
    observed_time: float = 0.0
    observed_by: tuple[int, ...] = ()

@dataclass(frozen=True)
class CellTarget:
    cell_id: tuple[int, int]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    observed_time: float = 0.0
    observed_by: tuple[int, ...] = ()
    target_id: int = -1

@dataclass(frozen=True)
class Instrument:
    kind: InstrumentKind
    subject: object = -1
    team: int = 0
    cell: tuple[int, int] = (0, 0)
    position: tuple[float, float, float] | None = None
    available: float = 1.0
    path_position: float = 0.0
    path_length: float = 0.0
    speed: float = 0.0
    respawn_time: float = 0.0
    health: float = 0.0
    observed_time: float = 0.0
    observed_by: tuple[int, ...] = ()

@dataclass(frozen=True)
class InstrumentBatch:
    participants: tuple[Participant, ...]
    instruments: tuple[Instrument, ...]
    descriptors: np.ndarray
    action_mass: np.ndarray
    navigation: object = None

    def index(self, kind: InstrumentKind | str, subject=None) -> int:
        wanted = InstrumentKind(kind)
        matches = [i for i, inst in enumerate(self.instruments)
                   if inst.kind == wanted and (subject is None or inst.subject == subject)]
        return matches[0]

@dataclass(frozen=True)
class Commitment:
    walking_distance: float
    walking_time: float
    extension: float
    horizon: float

def _descriptor(inst: Instrument) -> np.ndarray:
    row = np.zeros(DESCRIPTOR_WIDTH, dtype=np.float32)
    row[KIND_INDEX[inst.kind]] = 1.0
    row[len(KINDS):] = (
        inst.available, *(inst.position or (0.0, 0.0, 0.0)), inst.path_position, inst.path_length,
        inst.speed, inst.respawn_time, inst.health, inst.observed_time,
    )
    return row

def _action_mass(actor: Participant, inst: Instrument) -> float:
    return float(not inst.observed_by or actor.team in inst.observed_by)

def build_instruments(
    participants: Sequence[Participant],
    carts: Sequence[CartTarget] = (),
    items: Sequence[ItemTarget] = (),
    rivals: Sequence[RivalTarget] = (),
    cells: Sequence[CellTarget] = (),
    navigation=None,
) -> InstrumentBatch:
    actors = tuple(participants)
    instruments: list[Instrument] = []
    for cart in carts:
        base = dict(
            subject=cart.cart_id, team=cart.control_team, position=cart.position,
            path_position=cart.path_position, path_length=cart.path_length,
            speed=cart.speed,
        )
        instruments.append(Instrument(InstrumentKind.PUSH_CART, **base))
        instruments.append(Instrument(InstrumentKind.SUPPRESS_CART, **base))
    for item in items:
        instruments.append(Instrument(
            InstrumentKind.CONTEST_POST, item.item_id, cell=item.cell,
            position=item.position, available=item.available,
            respawn_time=item.respawn_time, observed_time=item.observed_time,
            observed_by=item.observed_by,
        ))
    for rival in rivals:
        instruments.append(Instrument(
            InstrumentKind.HUNT_RIVAL, rival.rival_id, team=rival.team,
            cell=rival.cell, position=rival.position, health=rival.health,
            observed_time=rival.observed_time, observed_by=rival.observed_by,
        ))
    for cell in cells:
        instruments.append(Instrument(
            InstrumentKind.EXPLORE_CELL, cell.target_id, cell=cell.cell_id,
            position=cell.position, observed_time=cell.observed_time,
            observed_by=cell.observed_by,
        ))
    instruments.extend((
        Instrument(InstrumentKind.SPAWN_TIMING),
        Instrument(InstrumentKind.IDLE),
    ))
    refs = tuple(instruments)
    descriptors = np.stack([_descriptor(inst) for inst in refs])
    action_mass = np.empty((len(actors), len(refs)), dtype=np.float32)
    for p, actor in enumerate(actors):
        for m, inst in enumerate(refs):
            action_mass[p, m] = _action_mass(actor, inst)
    return InstrumentBatch(actors, refs, descriptors, action_mass, navigation)

def response_rows(batch: InstrumentBatch, actions, controls) -> np.ndarray:
    count = len(batch.participants)
    chosen = np.asarray(actions, dtype=np.int64).reshape(count)
    values = np.asarray(controls, dtype=np.float32).reshape(count, CONTROL_WIDTH)
    target_kinds = np.full(len(batch.instruments), TARGET_KIND["NONE"], dtype=np.float32)
    target_ids = np.full(len(batch.instruments), -1, dtype=np.float32)
    target_cells = np.zeros((len(batch.instruments), 2), dtype=np.float32)
    instrument_kinds = np.asarray([
        INSTRUMENT_KIND[inst.kind.name] for inst in batch.instruments
    ], dtype=np.float32)
    for index, inst in enumerate(batch.instruments):
        if inst.kind in (InstrumentKind.PUSH_CART, InstrumentKind.SUPPRESS_CART):
            target_kinds[index] = TARGET_KIND["CART"]
            target_ids[index] = inst.subject
        elif inst.kind == InstrumentKind.CONTEST_POST:
            target_kinds[index] = TARGET_KIND["ITEM"]
            target_ids[index] = inst.subject
        elif inst.kind == InstrumentKind.HUNT_RIVAL:
            target_kinds[index] = TARGET_KIND["RIVAL"]
            target_ids[index] = inst.subject
        elif inst.kind == InstrumentKind.EXPLORE_CELL:
            target_kinds[index] = TARGET_KIND["CELL"]
            target_ids[index] = inst.subject
            target_cells[index] = inst.cell
    out = np.zeros((count, len(SC)), dtype=np.float32)
    out[:, SC["INSTRUMENT_KIND"]] = instrument_kinds[chosen]
    out[:, SC["TARGET_KIND"]] = target_kinds[chosen]
    out[:, SC["TARGET_ID"]] = target_ids[chosen]
    chosen_cells = target_cells[chosen]
    out[:, SC["TARGET_CELL_X"]] = chosen_cells[:, 0]
    out[:, SC["TARGET_CELL_Y"]] = chosen_cells[:, 1]
    out[:, SC["GAIN"]] = values[:, 0]
    out[:, SC["SPAWN"]] = values[:, 2]
    out[:, SC["COMMIT"]] = np.asarray([
        assignment_commitment(
            actor, batch.instruments[action], values[index, 1], batch.navigation,
        ).horizon
        for index, (actor, action) in enumerate(zip(batch.participants, chosen))
    ], dtype=np.float32)
    return out

def assignment_horizon(actor: Participant, instrument: Instrument, residual=0.0,
                       navigation=None) -> float:
    return assignment_commitment(actor, instrument, residual, navigation).horizon

def assignment_commitment(actor: Participant, instrument: Instrument, residual=0.0,
                          navigation=None) -> Commitment:
    if instrument.position is None or instrument.kind in (
        InstrumentKind.IDLE, InstrumentKind.SPAWN_TIMING,
    ):
        return Commitment(0.0, 0.0, 0.0, 0.0)
    if navigation is None:
        distance = float(np.linalg.norm(
            np.asarray(instrument.position, dtype=np.float64)
            - np.asarray(actor.position, dtype=np.float64)
        ))
    else:
        distance = navigation.walking_distance(actor.position, instrument.position)
    walking_time = distance / STOCK_WALK_SPEED
    extension = float(np.logaddexp(0.0, float(residual)))
    return Commitment(distance, walking_time, extension, walking_time + extension)

def weights_from_table(batch: InstrumentBatch, table=None) -> np.ndarray:
    table = {} if table is None else table
    out = np.zeros((len(batch.participants), len(batch.instruments)), dtype=np.float32)
    for p, actor in enumerate(batch.participants):
        for m, instrument in enumerate(batch.instruments):
            out[p, m] = table.get(
                (actor.participant_id, instrument.kind.value, instrument.subject), 0.0
            )
    return out

def update_weight_table(batch: InstrumentBatch, weights, table=None):
    out = {} if table is None else dict(table)
    values = np.asarray(weights, dtype=np.float32)
    for p, actor in enumerate(batch.participants):
        for m, instrument in enumerate(batch.instruments):
            if batch.action_mass[p, m] > 0:
                out[(actor.participant_id, instrument.kind.value, instrument.subject)] = float(values[p, m])
    return out
