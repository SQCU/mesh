from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

# The schema is the single canonical engine<->solver contract; it lives beside the QC
# it mirrors (xonotic/payload/tools/strategy_io_schema.py). Resolve it on sys.path
# rather than duplicating it here.
import sys as _sys, pathlib as _pathlib
_TOOLS = _pathlib.Path(__file__).resolve().parents[2] / "payload" / "tools"
if str(_TOOLS) not in _sys.path:
    _sys.path.insert(0, str(_TOOLS))
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
    "target_available", "target_uncertainty", "target_winner",
    "target_rank", "target_nimber", "target_denial",
)
DESCRIPTOR_WIDTH = len(DESCRIPTOR_FIELDS)


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
    target_winner: float = 0.0
    target_rank: float = 0.0
    target_nimber: float = 0.0
    target_denial: float = 0.0
    observed_by: tuple[int, ...] = ()


@dataclass(frozen=True)
class InstrumentBatch:
    participants: tuple[Participant, ...]
    instruments: tuple[Instrument, ...]
    descriptors: np.ndarray
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


def _eligible(actor: Participant, inst: Instrument) -> bool:
    """MASK -- which actions EXIST for this actor. A game RULE, not a hint.

    The hand-authored 20-column relation row that used to live here is deleted:
    it violated SPEC §7 (it fed the solver `target_nimber`, `target_denial`,
    `target_winner`, `target_rank` — the very conclusions the learned operator
    exists to produce — plus a copy of the actor's own state per instrument).
    Whatever is worth knowing about an (actor, instrument) pair is learned:
    the score is ``quinn(...) @ kay(...).T``, computed, never stored.
    """
    seen = not inst.observed_by or actor.team in inst.observed_by
    known_team = inst.team > 0
    same_team = known_team and actor.team == inst.team
    opposing = known_team and actor.team != inst.team
    eligible = seen and (inst.kind != InstrumentKind.HUNT_RIVAL or not same_team)
    if inst.kind == InstrumentKind.PUSH_CART:
        eligible = eligible and not opposing
    if inst.kind == InstrumentKind.SUPPRESS_CART:
        eligible = eligible and opposing
    if actor.alive < 0.5:
        eligible = eligible and inst.kind in (InstrumentKind.SPAWN_TIMING, InstrumentKind.IDLE)
    else:
        eligible = eligible and inst.kind != InstrumentKind.SPAWN_TIMING
    return eligible


def build_instruments(
    participants: Sequence[Participant],
    carts: Sequence[CartTarget] = (),
    items: Sequence[ItemTarget] = (),
    rivals: Sequence[RivalTarget] = (),
    cells: Sequence[CellTarget] = (),
    team_roles=None,
) -> InstrumentBatch:
    actors = tuple(participants)
    roles = {} if team_roles is None else team_roles
    instruments: list[Instrument] = []
    for cart in carts:
        role = roles.get(cart.control_team, (0.0, 0.0, 0.0, 0.0))
        base = dict(
            subject=cart.cart_id, team=cart.control_team, position=cart.position,
            progress=cart.depth, motion=cart.speed, value=cart.progress,
            target_winner=role[0], target_rank=role[1],
            target_nimber=role[2], target_denial=role[3],
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
        role = roles.get(rival.team, (0.0, 0.0, 0.0, 0.0))
        instruments.append(Instrument(
            InstrumentKind.HUNT_RIVAL, rival.rival_id, team=rival.team,
            cell=rival.cell, position=rival.position, urgency=rival.threat,
            uncertainty=rival.uncertainty, visible=1.0, temporal=rival.age,
            value=rival.threat, observed_by=rival.observed_by,
            target_winner=role[0], target_rank=role[1],
            target_nimber=role[2], target_denial=role[3],
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
    eligible = np.empty((len(actors), len(refs)), dtype=bool)
    for p, actor in enumerate(actors):
        for m, inst in enumerate(refs):
            eligible[p, m] = _eligible(actor, inst)
    return InstrumentBatch(actors, refs, descriptors, eligible)


# Travel commitment (AGENDA F4). `COMMIT` reaches the engine as
#     this.bot_strategytime = max(this.bot_strategytime, time + this.plc_str_commit)
# (sv_payload_strategy_io.qc), i.e. SECONDS for which the bot holds its current
# objective instead of re-diceing it. It was previously written only by the
# TRAVEL_COMMITMENT instrument, so on the real Game-2 run it was nonzero on 1 of
# 3150 assignments and `bot_strategytime` was effectively never driven. It is a
# property of an ASSIGNMENT, not of one instrument: committing to a target means
# committing to the trip to it, so every objective carries the horizon its own
# travel implies.
#
# Positions are engine units / 1024 (the OBS/CS position columns), and a Xonotic
# player runs at roughly 400 u/s, so one normalized unit of separation is about
# 1024/400 seconds of travel.
COMMIT_SECONDS_PER_UNIT = 1024.0 / 400.0
COMMIT_BASE_SECONDS = 0.6
COMMIT_MIN_SECONDS = 0.25
COMMIT_MAX_SECONDS = 30.0
_DISTANCE = RELATION_FIELDS.index("distance")
# Instruments that must never pin bot_strategytime: IDLE holds no objective, and
# SPAWN_TIMING is only eligible for a dead actor, who is not travelling.
NO_COMMIT_KINDS = (InstrumentKind.IDLE, InstrumentKind.SPAWN_TIMING)


def travel_horizon(batch: "InstrumentBatch", participant: int, action: int, scale: float = 1.0) -> float:
    """Seconds an actor should hold this assignment, from the trip it implies.

    The separation is read out of the relation row the operator itself consumes,
    so the horizon and the model input cannot disagree. A target that carries no
    position (an instrument with no world location, e.g. the placeholder rows)
    contributes no travel and falls back to the base horizon.
    """
    instrument = batch.instruments[int(action)]
    if instrument.kind in NO_COMMIT_KINDS:
        return 0.0
    located = any(abs(value) > 0.0 for value in instrument.position)
    travel = float(batch.relations[participant, int(action), _DISTANCE]) if located else 0.0
    seconds = float(scale) * (COMMIT_BASE_SECONDS + travel * COMMIT_SECONDS_PER_UNIT)
    return float(np.clip(seconds, COMMIT_MIN_SECONDS, COMMIT_MAX_SECONDS))


def _values(value, count: int, default: float, dtype=np.float32) -> np.ndarray:
    source = default if value is None else value
    return np.array(np.broadcast_to(np.asarray(source, dtype=dtype), (count,)), copy=True)


def decode_allocations(
    batch: InstrumentBatch,
    actions=None,
    intensity=None,
    commitments=None,
    spawn_delays=None,
) -> np.ndarray:
    count = len(batch.participants)
    idle = batch.index(InstrumentKind.IDLE)
    chosen = _values(actions, count, idle, np.int64) % len(batch.instruments)
    gain = np.maximum(0.0, _values(intensity, count, 1.0))
    commit = _values(commitments, count, np.nan)
    spawn = _values(spawn_delays, count, np.nan)
    out = np.zeros((count, len(SC)), dtype=np.float32)
    out[:, SC["TARGET"]] = -1
    for p, action in enumerate(chosen):
        inst = batch.instruments[int(action)]
        actor = batch.participants[p]
        # Every assignment carries its travel-commitment horizon, whatever the
        # objective is; `commitments` is the policy's scalar on that horizon,
        # not the horizon itself.
        out[p, SC["COMMIT"]] = travel_horizon(
            batch, p, int(action), 1.0 if np.isnan(commit[p]) else float(commit[p])
        )
        if inst.kind in (InstrumentKind.PUSH_CART, InstrumentKind.SUPPRESS_CART):
            out[p, SC["TARGET"]] = encode_target("cart", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
        elif inst.kind == InstrumentKind.CONTEST_POST:
            out[p, SC["TARGET"]] = encode_target("item", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
        elif inst.kind == InstrumentKind.HUNT_RIVAL:
            out[p, SC["TARGET"]] = encode_target("rival", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
        elif inst.kind == InstrumentKind.EXPLORE_CELL:
            out[p, SC["TARGET"]] = encode_target("cell", inst.subject)
            out[p, SC["GAIN"]] = gain[p]
        elif inst.kind == InstrumentKind.SPAWN_TIMING:
            out[p, SC["SPAWN"]] = max(0.0, 1.0 if np.isnan(spawn[p]) else spawn[p])
        elif inst.kind == InstrumentKind.TRAVEL_COMMITMENT:
            # Still an instrument in its own right -- "hold position and commit,
            # pick up no other objective" -- but no longer the only writer of
            # COMMIT; its horizon comes from the same formula as every other
            # assignment.
            out[p, SC["TARGET"]] = encode_target("cell", max(0, actor.cell))
            out[p, SC["GAIN"]] = gain[p]
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
