from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class EventKind(Enum):
    ITEM_SPAWN = "item_spawn"
    ITEM_DESPAWN = "item_despawn"
    ENEMY_SEEN = "enemy_seen"

    @property
    def is_item_presence(self):
        return self in (EventKind.ITEM_SPAWN, EventKind.ITEM_DESPAWN)

@dataclass(frozen=True)
class Observation:
    team: int
    observer: int
    t: float
    cell: tuple[int, int]
    kind: EventKind
    subject: int
    payload: dict | None = None

@dataclass(frozen=True)
class ContextualEvent:
    team: int
    observer: int
    t: float
    cell: tuple[int, int]
    kind: EventKind
    subject: int
    payload: dict | None

class ObservationBuffer:
    def __init__(self):
        self._events = {}

    def observe(self, observation):
        kind = observation.kind if isinstance(observation.kind, EventKind) else EventKind(observation.kind)
        event = ContextualEvent(
            observation.team,
            observation.observer,
            observation.t,
            observation.cell,
            kind,
            observation.subject,
            observation.payload,
        )
        key = (kind.is_item_presence, observation.subject)
        events = self._events.setdefault(observation.team, {})
        previous = events.get(key)
        if previous is None or event.t >= previous.t:
            events[key] = event
        return event

    def teams(self):
        return list(self._events)

    def events(self, team):
        return sorted(
            self._events.get(team, {}).values(),
            key=lambda event: (event.t, event.subject, event.kind.value),
        )

    def clear(self):
        self._events.clear()

    def __len__(self):
        return sum(map(len, self._events.values()))

__all__ = ["EventKind", "Observation", "ContextualEvent", "ObservationBuffer"]
