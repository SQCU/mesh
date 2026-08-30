from __future__ import annotations

import time
from collections import deque

import numpy as np

from .buffers import EventKind, Observation, ObservationBuffer
from .featurize import SLOT_DIM, VCellMap


class LiveBelief:
    def __init__(self, decay=8.0, capacity=4096, signature_capacity=8192):
        self.decay = float(decay)
        self.buffer = ObservationBuffer(capacity=capacity)
        self.signature_capacity = int(signature_capacity)
        self.key = None
        self.tick = None
        self.now = 0.0
        self.wall = time.monotonic()
        self.cells = {}
        self.edges = set()
        self.player_cells = {}
        self.signatures = set()
        self.signature_order = deque()
        self.accepted = 0
        self.duplicates = 0
        self.invalid = 0

    def reset(self, key=None, tick=None):
        self.buffer.clear()
        self.key = key
        self.tick = tick
        self.now = 0.0
        self.wall = time.monotonic()
        self.cells.clear()
        self.edges.clear()
        self.player_cells.clear()
        self.signatures.clear()
        self.signature_order.clear()
        self.accepted = 0
        self.duplicates = 0
        self.invalid = 0

    def sync(self, key, tick):
        changed = self.key != key or (self.tick is not None and tick < self.tick)
        if changed:
            self.reset(key, tick)
        else:
            self.key = key
            self.tick = tick
        return changed

    def _remember(self, signature):
        if signature in self.signatures:
            return False
        if len(self.signature_order) >= self.signature_capacity:
            self.signatures.discard(self.signature_order.popleft())
        self.signatures.add(signature)
        self.signature_order.append(signature)
        return True

    def ingest(self, rows, columns):
        pending = []
        for row in np.atleast_2d(rows):
            try:
                values = tuple(float(row[columns[name]]) for name in
                               ("CELL", "KIND", "TEAM", "SUBJECT", "VALUE", "TIME"))
                if not np.all(np.isfinite(values)):
                    self.invalid += 1
                    continue
                if values[5] <= 0:
                    self.invalid += int(any(values))
                    continue
                signature = tuple(round(v, 6) for v in values)
                if not self._remember(signature):
                    self.duplicates += 1
                    continue
                pending.append(values)
            except (IndexError, KeyError, TypeError, ValueError):
                self.invalid += 1
        pending.sort(key=lambda values: values[5])
        deposited = 0
        for cell, raw_kind, team, subject, value, stamp in pending:
            code = int(round(raw_kind))
            kind = EventKind.ITEM_DESPAWN if code == 0 else (
                EventKind.ITEM_SPAWN if code == 1 else EventKind.ENEMY_SEEN
            )
            payload = {"value": float(value), "raw_kind": code}
            event = self.buffer.observe(Observation(
                int(round(team)), -1, float(stamp), int(round(cell)), kind,
                int(round(subject)), True, True, 0.0, payload,
            ))
            deposited += int(event is not None)
            self.now = max(self.now, float(stamp))
        self.accepted += deposited
        return deposited

    @staticmethod
    def cell_of_position(x, y):
        gx = int(np.floor(float(x) * 4.0))
        gy = int(np.floor(float(y) * 4.0))
        return (gx * 131 + gy) & 1023

    def _put_cell(self, cell, position):
        position = np.asarray(position, dtype=np.float64)[:2]
        old = self.cells.get(cell)
        self.cells[cell] = position if old is None else 0.8 * old + 0.2 * position

    def _link(self, left, right):
        if left != right:
            self.edges.add(tuple(sorted((int(left), int(right)))))

    def _update_cells(self, rows, columns):
        players = []
        by_id = {}
        by_team = {}
        player_positions = {}
        for row in np.atleast_2d(rows):
            participant = int(round(row[columns["ID"]]))
            team = int(round(row[columns["TEAM"]]))
            position = np.asarray((row[columns["POS_X"]], row[columns["POS_Y"]]),
                                  dtype=np.float64)
            cell = self.cell_of_position(*position)
            previous = self.player_cells.get(participant)
            if previous is not None:
                self._link(previous, cell)
            self.player_cells[participant] = cell
            player_positions.setdefault(cell, []).append(position)
            record = (participant, team, cell, position)
            players.append(record)
            by_id[participant] = record
            by_team.setdefault(team, []).append(record)
        for cell, positions in player_positions.items():
            self._put_cell(cell, np.mean(positions, axis=0))
        event_positions = {}
        for team in self.buffer.teams():
            observers = by_team.get(int(team), [])
            for event in self.buffer.events(team):
                position = None
                subject = by_id.get(int(event.subject))
                if event.kind == EventKind.ENEMY_SEEN and subject is not None:
                    position = subject[3]
                elif observers:
                    position = np.mean([record[3] for record in observers], axis=0)
                elif self.cells:
                    position = np.mean(list(self.cells.values()), axis=0)
                else:
                    position = np.zeros(2, dtype=np.float64)
                event_positions.setdefault(int(event.cell), []).append(position)
                if observers:
                    nearest = min(observers, key=lambda record: np.linalg.norm(record[3] - position))
                    self._link(int(event.cell), nearest[2])
        for cell, positions in event_positions.items():
            self._put_cell(cell, np.mean(positions, axis=0))
        return players

    def _vcmap(self):
        ids = sorted(self.cells)
        if not ids:
            self.cells[0] = np.zeros(2, dtype=np.float64)
            ids = [0]
        index = {cell: i for i, cell in enumerate(ids)}
        positions = np.asarray([self.cells[cell] for cell in ids], dtype=np.float64)
        n = len(ids)
        adjacency = [set() for _ in ids]
        for left, right in self.edges:
            if left in index and right in index:
                adjacency[index[left]].add(index[right])
                adjacency[index[right]].add(index[left])
        if n > 1:
            d2 = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
            for i in range(n):
                for j in np.argsort(d2[i], kind="stable")[1:min(n, 3)]:
                    adjacency[i].add(int(j))
                    adjacency[int(j)].add(i)
        distance = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(distance, 0.0)
        for i, neighbors in enumerate(adjacency):
            for j in neighbors:
                distance[i, j] = 1.0
        for mid in range(n):
            distance = np.minimum(distance, distance[:, mid, None] + distance[None, mid, :])
        vcmap = VCellMap(positions, np.ones(n), float(n), distance, 2.0,
                         positions, np.arange(n), band=(0.0, 1.0))
        return ids, index, vcmap

    def _team_features(self, team, ids, index, now):
        features = np.zeros((len(ids), SLOT_DIM), dtype=np.float64)
        times = np.full((len(ids), SLOT_DIM), -np.inf, dtype=np.float64)
        item_latest = {}
        enemy_latest = {}
        for event in self.buffer.events(team):
            if event.kind.is_status:
                item_latest[event.subject] = event
            else:
                enemy_latest[(event.subject, event.cell)] = event
        by_cell_items = {}
        by_cell_enemies = {}
        for event in item_latest.values():
            by_cell_items.setdefault(int(event.cell), []).append(event)
        for event in enemy_latest.values():
            by_cell_enemies.setdefault(int(event.cell), []).append(event)
        for cell, events in by_cell_items.items():
            if cell not in index:
                continue
            i = index[cell]
            stamp = max(float(event.t) for event in events)
            available = np.mean([event.kind == EventKind.ITEM_SPAWN for event in events])
            phases = [max(0.0, float((event.payload or {}).get("value", 0.0)))
                      for event in events if event.kind == EventKind.ITEM_DESPAWN]
            features[i, 0] = available
            features[i, 1] = max(phases, default=0.0) / (10.0 + max(phases, default=0.0))
            features[i, 2] = 1.0
            features[i, 6] = 1.0
            times[i, (0, 1, 2, 6)] = stamp
        for cell, events in by_cell_enemies.items():
            if cell not in index:
                continue
            i = index[cell]
            stamp = max(float(event.t) for event in events)
            threat = max(float((event.payload or {}).get("value", 0.0)) for event in events)
            features[i, 2] = 1.0
            features[i, 4] = np.tanh(max(0.0, threat) / 100.0)
            features[i, 5] = 1.0 - np.exp(-len(events))
            features[i, 6] = 1.0
            times[i, (2, 4, 5, 6)] = stamp
        prior = np.asarray((0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
        observed = np.isfinite(times)
        age = np.maximum(0.0, float(now) - np.where(observed, times, float(now)))
        rho = np.where(observed, np.exp(-age / self.decay), 0.0)
        return rho * features + (1.0 - rho) * prior[None]

    @staticmethod
    def _project(features):
        phi = np.asarray([
            (1, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 1),
            (0, 0, 1, 0, 0, 0, 0),
            (1, -1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 1, 0),
        ], dtype=np.float64)
        return features @ phi.T

    def beliefs(self, rows, columns, now=None):
        wall = time.monotonic()
        if now is None:
            self.now += max(0.0, wall - self.wall)
        else:
            self.now = max(self.now, float(now))
        self.wall = wall
        players = self._update_cells(rows, columns)
        ids, index, vcmap = self._vcmap()
        team_projection = {}
        out = np.zeros((len(players), 8), dtype=np.float32)
        for p, (_, team, cell, _) in enumerate(players):
            if team not in team_projection:
                team_projection[team] = self._project(
                    self._team_features(team, ids, index, self.now)
                )
            weights = vcmap.spatial_mask(index[cell])
            total = float(weights.sum())
            if total <= 0:
                weights[index[cell]] = 1.0
                total = 1.0
            out[p] = ((weights / total) @ team_projection[team]).astype(np.float32)
        diagnostics = {
            "cells": len(ids),
            "edges": len(self.edges),
            "events": len(self.buffer),
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "invalid": self.invalid,
            "teams": len(self.buffer.teams()),
            "mean_norm": round(float(np.linalg.norm(out, axis=1).mean()), 6) if len(out) else 0.0,
        }
        return out, diagnostics

    def instrument_targets(self, rows, columns):
        from .instruments import CellTarget, ItemTarget, RivalTarget

        player_team = {
            int(round(row[columns["ID"]])): int(round(row[columns["TEAM"]]))
            for row in np.atleast_2d(rows)
        }
        item_events = {}
        rival_events = {}
        for observer_team in self.buffer.teams():
            for event in self.buffer.events(observer_team):
                key = int(event.subject)
                target = item_events if event.kind.is_status else rival_events
                previous = target.get((key, int(observer_team)))
                if previous is None or event.t >= previous.t:
                    target[(key, int(observer_team))] = event
        items = []
        for subject in sorted({key[0] for key in item_events}):
            events = [event for (item, _), event in item_events.items() if item == subject]
            latest = max(events, key=lambda event: event.t)
            position = self.cells.get(int(latest.cell), np.zeros(2))
            respawn = max(0.0, float((latest.payload or {}).get("value", 0.0)))
            items.append(ItemTarget(
                subject, int(latest.cell), (float(position[0]), float(position[1]), 0.0),
                float(latest.kind == EventKind.ITEM_SPAWN), 1.0,
                respawn / (10.0 + respawn),
                tuple(sorted({team for (item, team) in item_events if item == subject})),
            ))
        rivals = []
        for subject in sorted({key[0] for key in rival_events}):
            events = [event for (rival, _), event in rival_events.items() if rival == subject]
            latest = max(events, key=lambda event: event.t)
            position = self.cells.get(int(latest.cell), np.zeros(2))
            threat = max(float((event.payload or {}).get("value", 0.0)) for event in events)
            age = max(0.0, self.now - float(latest.t))
            rivals.append(RivalTarget(
                subject, player_team.get(subject, 0), int(latest.cell),
                (float(position[0]), float(position[1]), 0.0), threat, age,
                1.0 - np.exp(-age / self.decay),
                tuple(sorted({team for (rival, team) in rival_events if rival == subject})),
            ))
        cells = [
            CellTarget(
                int(cell), (float(position[0]), float(position[1]), 0.0),
                1.0, 1.0 / (1.0 + sum(cell in edge for edge in self.edges)),
                float(sum(cell in edge for edge in self.edges)),
            )
            for cell, position in sorted(self.cells.items())
        ]
        return items, rivals, cells
