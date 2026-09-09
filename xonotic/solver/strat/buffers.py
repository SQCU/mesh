from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from payload.tools.strategy_io_schema import EVT, EVT_WIDTH


class ObservationMemory:
    def __init__(self):
        self.histories, self.counts, self.unattributed, self.arrivals = {}, {}, {}, {}
        self.episode = None
        self.empty = np.empty((0, EVT_WIDTH), dtype=np.float32)
        self.coverage = {'complete': True, 'source': 'received_episode_events'}

    @staticmethod
    def identity(session, episode):
        return f'{int(session) >> 24}:{int(session) & 0xffffff}:{int(episode)}'

    def observe(self, episode, batches):
        changed = episode != self.episode
        self.episode = episode
        self._append(episode, self.empty)
        arrivals = {}
        for key, rows in batches.items():
            rows = np.asarray(rows, dtype=np.float32)
            if rows.shape[1] != EVT_WIDTH:
                self.unattributed[key] = rows
                self.coverage = {'complete': False, 'source': 'received_episode_events',
                                 'reason': 'retained_events_without_deposit_episode'}
                continue
            for owner in np.unique(rows[:, EVT['EPISODE']]):
                identity = self.identity(key[0], owner)
                selected = rows[rows[:, EVT['EPISODE']] == owner]
                self._append(identity, selected)
                arrivals.setdefault(identity, []).append(selected)
        self.arrivals = {owner: np.concatenate(parts) for owner, parts in arrivals.items()}
        return changed

    def _append(self, identity, rows):
        previous = self.counts.get(identity, 0)
        count = previous + len(rows)
        storage = self.histories.get(identity, self.empty)
        if identity not in self.histories or count > len(storage):
            replacement = np.empty((1 << (max(1, count) - 1).bit_length(), EVT_WIDTH), dtype=np.float32)
            replacement[:previous] = storage[:previous]
            storage = self.histories[identity] = replacement
        storage[previous:count] = rows
        self.counts[identity] = count

    @property
    def storage(self):
        return self.histories.get(self.episode, self.empty)

    @property
    def count(self):
        return self.counts.get(self.episode, 0)

    @property
    def rows(self):
        result = self.storage[:self.count].view()
        result.flags.writeable = False
        return result

    def export_state(self):
        return {'episode': self.episode, 'histories': {owner: rows[:self.counts[owner]] for owner, rows in self.histories.items()},
                'unattributed': self.unattributed, 'coverage': self.coverage}

    @classmethod
    def restore(cls, saved):
        memory = cls()
        memory.episode = saved.get('episode_key')
        payload = saved.get('observation_memory')
        if isinstance(payload, dict):
            memory.episode = payload['episode']
            for owner, rows in payload['histories'].items():
                memory._append(owner, rows)
            memory.unattributed = payload['unattributed']
            memory.coverage = payload['coverage']
        else:
            if payload is not None:
                memory.unattributed['legacy_snapshot'] = payload
            history = saved.get('history', {})
            for sequence, entry in history.get('requests', {}).get('entries', {}).items():
                frame = getattr((entry.get('state') or {}).get('frame'), 'chorus', None)
                if frame is not None:
                    memory.unattributed[('legacy_source_frame', sequence)] = frame.context
            memory.coverage = {'complete': False, 'source': 'legacy_snapshot',
                               'reason': 'retained_event_ownership_not_recorded'}
        return memory, memory.report()

    def report(self):
        return {'episode_id': self.episode, 'rows': self.count, 'capacity': len(self.storage),
                'new_rows': len(self.arrivals.get(self.episode, self.empty)),
                'new_rows_by_episode': {owner: len(rows) for owner, rows in self.arrivals.items()},
                'retained_episodes': len(self.histories), 'retained_rows': sum(self.counts.values()),
                'unattributed_rows': sum(len(rows) for rows in self.unattributed.values()),
                'histories': {owner: {'rows': self.counts[owner], 'capacity': len(rows)} for owner, rows in self.histories.items()},
                **self.coverage}

class EventKind(Enum):
    ITEM_SPAWN = "item_spawn"
    ITEM_DESPAWN = "item_despawn"
    ENEMY_SEEN = "enemy_seen"

@dataclass(frozen=True)
class ContextualEvent:
    team: int
    observer: int
    t: float
    cell: tuple[int, int]
    kind: EventKind
    subject: int
    payload: dict | None

__all__ = ["EventKind", "ContextualEvent", "ObservationMemory"]
