from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from strategy_io_schema import SC, encode_target


class InstrumentKind(str, Enum):
    PUSH_CART = "push_cart"
    SUPPRESS_CART = "suppress_cart"
    CONTEST_POST = "contest_post"
    HUNT_RIVAL = "hunt_rival"
    EXPLORE_CELL = "explore_cell"
    SPAWN_TIMING = "spawn_timing"
    TRAVEL_COMMITMENT = "travel_commitment"
    IDLE = "idle"


KINDS = tuple(InstrumentKind)
KIND_INDEX = {kind: i for i, kind in enumerate(KINDS)}
DESCRIPTOR_FIELDS = (
    "push_cart", "suppress_cart", "contest_post", "hunt_rival",
    "explore_cell", "spawn_timing", "travel_commitment", "idle",
    "available", "urgency", "progress", "motion", "uncertainty",
    "visible", "temporal", "value",
)
RELATION_FIELDS = (
    "same_team", "opposing_team", "observed", "same_cell",
    "dx", "dy", "dz", "distance", "actor_alive", "actor_health",
    "actor_armor", "actor_ammo", "actor_tss", "target_value",
    "target_available", "target_uncertainty",
)
DESCRIPTOR_WIDTH = len(DESCRIPTOR_FIELDS)
RELATION_WIDTH = len(RELATION_FIELDS)


@dataclass(frozen=True)
class Participant:
    participant_id: int
    team: int
    cell: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alive: float = 1.0
    health: float = 1.0
    armor: float = 0.0
    ammo: float = 0.0
    time_since_spawn: float = 0.0


@dataclass(frozen=True)
class CartTarget:
    cart_id: int
    control_team: int
    depth: float
    speed: float = 0.0
    progress: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ItemTarget:
    item_id: int
    cell: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    available: float = 1.0
    value: float = 1.0
    respawn_phase: float = 0.0
    observed_by: tuple[int, ...] = ()


@dataclass(frozen=True)
class RivalTarget:
    rival_id: int
    team: int
    cell: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    threat: float = 1.0
    age: float = 0.0
    uncertainty: float = 0.0
    observed_by: tuple[int, ...] = ()


@dataclass(frozen=True)
class CellTarget:
    cell_id: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: float = 1.0
    uncertainty: float = 0.0
    value: float = 0.0
    observed_by: tuple[int, ...] = ()


@dataclass(frozen=True)
class Instrument:
    kind: InstrumentKind
    subject: int = -1
    team: int = 0
    cell: int = -1
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    available: float = 1.0
    urgency: float = 0.0
    progress: float = 0.0
    motion: float = 0.0
    uncertainty: float = 0.0
    visible: float = 1.0
    temporal: float = 0.0
    value: float = 0.0
    observed_by: tuple[int, ...] = ()


@dataclass(frozen=True)
class InstrumentBatch:
    participants: tuple[Participant, ...]
    instruments: tuple[Instrument, ...]
    descriptors: np.ndarray
    relations: np.ndarray
    eligible: np.ndarray

    def index(self, kind: InstrumentKind | str, subject: int | None = None) -> int:
        wanted = InstrumentKind(kind)
        matches = [i for i, inst in enumerate(self.instruments)
                   if inst.kind == wanted and (subject is None or inst.subject == subject)]
        return matches[0]


def _descriptor(inst: Instrument) -> np.ndarray:
    row = np.zeros(DESCRIPTOR_WIDTH, dtype=np.float32)
    row[KIND_INDEX[inst.kind]] = 1.0
    row[8:] = (
        inst.available, inst.urgency, inst.progress, inst.motion,
        inst.uncertainty, inst.visible, inst.temporal, inst.value,
    )
    return row


def _relation(actor: Participant, inst: Instrument) -> tuple[np.ndarray, bool]:
    seen = not inst.observed_by or actor.team in inst.observed_by
    known_team = inst.team > 0
    same_team = known_team and actor.team == inst.team
    opposing = known_team and actor.team != inst.team
    same_cell = inst.cell >= 0 and actor.cell == inst.cell
    delta = np.asarray(inst.position, dtype=np.float32) - np.asarray(actor.position, dtype=np.float32)
    row = np.asarray((
        same_team, opposing, seen, same_cell, delta[0], delta[1], delta[2],
        np.linalg.norm(delta), actor.alive, actor.health, actor.armor, actor.ammo,
        actor.time_since_spawn, inst.value, inst.available, inst.uncertainty,
    ), dtype=np.float32)
    eligible = seen and (inst.kind != InstrumentKind.HUNT_RIVAL or not same_team)
    if actor.alive < 0.5:
        eligible = eligible and inst.kind in (InstrumentKind.SPAWN_TIMING, InstrumentKind.IDLE)
    else:
        eligible = eligible and inst.kind != InstrumentKind.SPAWN_TIMING
    return row, eligible


def build_instruments(
    participants: Sequence[Participant],
    carts: Sequence[CartTarget] = (),
    items: Sequence[ItemTarget] = (),
    rivals: Sequence[RivalTarget] = (),
    cells: Sequence[CellTarget] = (),
) -> InstrumentBatch:
    actors = tuple(participants)
    instruments: list[Instrument] = []
    for cart in carts:
        base = dict(
            subject=cart.cart_id, team=cart.control_team, position=cart.position,
            progress=cart.depth, motion=cart.speed, value=cart.progress,
        )
        instruments.append(Instrument(InstrumentKind.PUSH_CART, urgency=1.0 - cart.depth, **base))
        instruments.append(Instrument(InstrumentKind.SUPPRESS_CART, urgency=cart.depth, **base))
    for item in items:
        instruments.append(Instrument(
            InstrumentKind.CONTEST_POST, item.item_id, cell=item.cell,
            position=item.position, available=item.available,
            urgency=item.value, visible=1.0, temporal=item.respawn_phase,
            value=item.value, observed_by=item.observed_by,
        ))
    for rival in rivals:
        instruments.append(Instrument(
            InstrumentKind.HUNT_RIVAL, rival.rival_id, team=rival.team,
            cell=rival.cell, position=rival.position, urgency=rival.threat,
            uncertainty=rival.uncertainty, visible=1.0, temporal=rival.age,
            value=rival.threat, observed_by=rival.observed_by,
        ))
    for cell in cells:
        instruments.append(Instrument(
            InstrumentKind.EXPLORE_CELL, cell.cell_id, cell=cell.cell_id,
            position=cell.position, urgency=cell.uncertainty,
            uncertainty=cell.uncertainty, visible=cell.visible,
            value=cell.value, observed_by=cell.observed_by,
        ))
    instruments.extend((
        Instrument(InstrumentKind.SPAWN_TIMING, urgency=1.0),
        Instrument(InstrumentKind.TRAVEL_COMMITMENT, urgency=1.0),
        Instrument(InstrumentKind.IDLE),
    ))
    refs = tuple(instruments)
    descriptors = np.stack([_descriptor(inst) for inst in refs])
    relations = np.empty((len(actors), len(refs), RELATION_WIDTH), dtype=np.float32)
    eligible = np.empty((len(actors), len(refs)), dtype=bool)
    for p, actor in enumerate(actors):
        for m, inst in enumerate(refs):
            relations[p, m], eligible[p, m] = _relation(actor, inst)
    return InstrumentBatch(actors, refs, descriptors, relations, eligible)


def _values(value, count: int, default: float, dtype=np.float32) -> np.ndarray:
    source = default if value is None else value
    return np.array(np.broadcast_to(np.asarray(source, dtype=dtype), (count,)), copy=True)


def decode_allocations(
    batch: InstrumentBatch,
    actions=None,
    intensity=None,
    lanes=None,
    commitments=None,
    spawn_delays=None,
    lead=None,
) -> np.ndarray:
    count = len(batch.participants)
    idle = batch.index(InstrumentKind.IDLE)
    chosen = _values(actions, count, idle, np.int64) % len(batch.instruments)
    gain = np.maximum(0.0, _values(intensity, count, 1.0))
    lane = _values(lanes, count, np.nan)
    commit = _values(commitments, count, np.nan)
    spawn = _values(spawn_delays, count, np.nan)
    leaders = _values(lead, count, 0.0)
    out = np.zeros((count, len(SC)), dtype=np.float32)
    out[:, SC["LEAD"]] = leaders
    for p, action in enumerate(chosen):
        inst = batch.instruments[int(action)]
        actor = batch.participants[p]
        if inst.kind in (InstrumentKind.PUSH_CART, InstrumentKind.SUPPRESS_CART):
            out[p, SC["TARGET"]] = encode_target("cart", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
            default_lane = inst.progress if inst.kind == InstrumentKind.PUSH_CART else min(1.0, inst.progress + 0.15)
            out[p, SC["LANE"]] = np.clip(default_lane if np.isnan(lane[p]) else lane[p], 0.0, 1.0)
        elif inst.kind == InstrumentKind.CONTEST_POST:
            out[p, SC["TARGET"]] = encode_target("item", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
        elif inst.kind == InstrumentKind.HUNT_RIVAL:
            out[p, SC["TARGET"]] = encode_target("rival", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
            out[p, SC["HUNT"]] = gain[p]
        elif inst.kind == InstrumentKind.EXPLORE_CELL:
            out[p, SC["TARGET"]] = encode_target("cell", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
            out[p, SC["EXPLORE"]] = gain[p]
        elif inst.kind == InstrumentKind.SPAWN_TIMING:
            out[p, SC["TARGET"]] = encode_target("cell", max(0, actor.cell))
            out[p, SC["SPAWN"]] = max(0.0, 1.0 if np.isnan(spawn[p]) else spawn[p])
        elif inst.kind == InstrumentKind.TRAVEL_COMMITMENT:
            out[p, SC["TARGET"]] = encode_target("cell", max(0, actor.cell))
            out[p, SC["COMMIT"]] = max(0.0, 1.0 if np.isnan(commit[p]) else commit[p])
    return out


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
            if batch.eligible[p, m]:
                out[(actor.participant_id, instrument.kind.value, instrument.subject)] = float(values[p, m])
    return out
