from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class EventKind(Enum):
    ITEM_SPAWN = "item_spawn"
    ITEM_DESPAWN = "item_despawn"
    ENEMY_SEEN = "enemy_seen"

    @property
    def is_status(self):
        return self in (EventKind.ITEM_SPAWN, EventKind.ITEM_DESPAWN)


@dataclass(frozen=True)
class Observation:
    team: int
    observer: int
    t: float
    cell: int
    kind: EventKind
    subject: int
    in_frustum: bool = True
    los_clear: bool = True
    vcell_dist: float = 0.0
    payload: dict | None = None


@dataclass(frozen=True)
class ContextualEvent:
    team: int
    observer: int
    t: float
    cell: int
    kind: EventKind
    subject: int
    payload: dict | None


class ObservationBuffer:
    def __init__(self, capacity=None):
        self.capacity = capacity
        self._events = {}
        self._last_status = {}

    def observe(self, observation):
        if not observation.in_frustum or not observation.los_clear or observation.vcell_dist > 2:
            return None
        kind = observation.kind if isinstance(observation.kind, EventKind) else EventKind(observation.kind)
        if kind.is_status:
            key = observation.team, observation.subject
            if self._last_status.get(key) == kind:
                return None
            self._last_status[key] = kind
        event = ContextualEvent(
            observation.team,
            observation.observer,
            observation.t,
            observation.cell,
            kind,
            observation.subject,
            observation.payload,
        )
        self._events.setdefault(observation.team, deque(maxlen=self.capacity)).append(event)
        return event

    def teams(self):
        return list(self._events)

    def events(self, team):
        return list(self._events.get(team, ()))

    def clear(self):
        self._events.clear()
        self._last_status.clear()

    def __len__(self):
        return sum(map(len, self._events.values()))


__all__ = ["EventKind", "Observation", "ContextualEvent", "ObservationBuffer"]
