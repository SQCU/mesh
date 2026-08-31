"""Live adapter between the mesh event/observation stream and the belief stages.

This file owns NO belief algebra. It maintains the V-cell dictionary, the
observed navigable edges and the per-team observation buffer from the live
mesh rows, folds them into observation rows, and then calls the canonical
stages in :mod:`featurize` — ``segment_vcells`` (stage 2, which is what SETS
the horizon), ``build_cell_slots`` + ``temporal_contraction`` (stage 3) and
``beliefs_for_bots`` (stages 4-5). It previously re-implemented all four
stages inline with a constant-literal Phi, a hardcoded support radius of 2.0
and a normalization the formula does not have, while the canonical module sat
unused; that copy is deleted.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from .buffers import EventKind, Observation, ObservationBuffer
from .featurize import (
    BELIEF_RANK,
    PHI,
    SLOT_DIM,
    UNINFORMATIVE_PRIOR,
    beliefs_for_bots,
    receptive_report,
    segment_vcells,
)

# The two-sided receptive-field band stage 2 sizes the horizon to.
BELIEF_BAND = (0.05, 0.15)


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
        self.cell_teams = {}
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
        self.cell_teams.clear()
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

    def _put_cell(self, cell, position, team=None):
        position = np.asarray(position, dtype=np.float64)[:2]
        old = self.cells.get(cell)
        self.cells[cell] = position if old is None else 0.8 * old + 0.2 * position
        if team is not None:
            self.cell_teams.setdefault(int(cell), set()).add(int(team))

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
            player_positions.setdefault((cell, team), []).append(position)
            record = (participant, team, cell, position)
            players.append(record)
            by_id[participant] = record
            by_team.setdefault(team, []).append(record)
        for (cell, team), positions in player_positions.items():
            self._put_cell(cell, np.mean(positions, axis=0), team)
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
                event_positions.setdefault((int(event.cell), int(team)), []).append(position)
                if observers:
                    nearest = min(observers, key=lambda record: np.linalg.norm(record[3] - position))
                    self._link(int(event.cell), nearest[2])
        for (cell, team), positions in event_positions.items():
            self._put_cell(cell, np.mean(positions, axis=0), team)
        return players

    def _vcmap(self):
        """Stage 2, via the CANONICAL :func:`featurize.segment_vcells`.

        Nodes are the V-cells discovered so far; the navigable adjacency is the
        set of transitions actually observed (a player moved cell i -> cell j,
        or an event was linked to its nearest observer's cell), unioned with a
        2-nearest-neighbour stand-in so isolated cells are not stranded at
        infinite graph distance.

        BLOCKER (engine side, not ours): the mesh OBS/EVT schema carries a
        ``CELL`` id but no waypoint-link table, so the real navigable graph is
        not available here and the kNN union is a stand-in.  Supplying real
        waypoint links would make this exactly the spec's "fuse contiguous
        navigable paths".

        The support radius is NOT set here -- ``segment_vcells`` sets it from
        the 5-15% receptive-field band.
        """
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
        vcmap = segment_vcells(positions, adjacency=[sorted(a) for a in adjacency],
                               band=BELIEF_BAND)
        return ids, index, vcmap

    def _slot_rows(self, team, index, vcmap):
        """Fold this TEAM's observation buffer into stage-2/3 observation rows.

        One row per observed V-cell, carrying the merged ``SLOT_FIELDS`` vector
        and the latest timestamp seen in that cell.  This is ingest only -- the
        forgetting (stage 3), the mask (stage 4) and the integration (stage 5)
        are the canonical :mod:`featurize` functions, called by :meth:`beliefs`.
        """
        item_latest = {}
        enemy_latest = {}
        for event in self.buffer.events(team):
            if event.kind.is_status:
                item_latest[event.subject] = event
            else:
                enemy_latest[(event.subject, event.cell)] = event
        slots = {}
        stamps = {}

        def _slot(cell):
            if cell not in slots:
                slots[cell] = np.zeros(SLOT_DIM, dtype=np.float64)
                stamps[cell] = -np.inf
            return slots[cell]

        by_cell_items = {}
        by_cell_enemies = {}
        for event in item_latest.values():
            by_cell_items.setdefault(int(event.cell), []).append(event)
        for event in enemy_latest.values():
            by_cell_enemies.setdefault(int(event.cell), []).append(event)
        for cell, events in by_cell_items.items():
            if cell not in index:
                continue
            slot = _slot(cell)
            phases = [max(0.0, float((event.payload or {}).get("value", 0.0)))
                      for event in events if event.kind == EventKind.ITEM_DESPAWN]
            slot[0] = float(np.mean([event.kind == EventKind.ITEM_SPAWN for event in events]))
            slot[1] = max(phases, default=0.0) / (10.0 + max(phases, default=0.0))
            slot[2] = 1.0
            stamps[cell] = max(stamps[cell], max(float(event.t) for event in events))
        for cell, events in by_cell_enemies.items():
            if cell not in index:
                continue
            slot = _slot(cell)
            threat = max(float((event.payload or {}).get("value", 0.0)) for event in events)
            slot[2] = 1.0
            slot[4] = float(np.tanh(max(0.0, threat) / 100.0))
            slot[5] = 1.0 - float(np.exp(-len(events)))
            stamps[cell] = max(stamps[cell], max(float(event.t) for event in events))
        return [
            {"cell": int(vcmap.node_cell[index[cell]]), "time": stamps[cell], "slot": slot}
            for cell, slot in slots.items()
        ]

    def beliefs(self, rows, columns, now=None):
        wall = time.monotonic()
        if now is None:
            self.now += max(0.0, wall - self.wall)
        else:
            self.now = max(self.now, float(now))
        self.wall = wall
        players = self._update_cells(rows, columns)
        ids, index, vcmap = self._vcmap()
        out = np.zeros((len(players), BELIEF_RANK), dtype=np.float32)
        by_team = {}
        for p, (_, team, cell, _) in enumerate(players):
            by_team.setdefault(team, []).append((p, int(vcmap.node_cell[index[cell]])))
        for team, members in by_team.items():
            betas = beliefs_for_bots(
                self._slot_rows(team, index, vcmap), vcmap,
                [cell for _, cell in members],
                Phi=PHI, now=self.now, T=self.decay, f_prior=UNINFORMATIVE_PRIOR,
            )
            for (p, _), beta in zip(members, betas):
                out[p] = beta.astype(np.float32)
        # E3: the band is the target of the stage-2 construction, which sizes
        # the horizon against EVERY cell, so that is the population it is
        # checked on. The realized distribution over the cells bots actually
        # occupy is reported alongside it -- it is a consequence, not the knob.
        occupied = sorted({cell for members in by_team.values() for _, cell in members})
        band = receptive_report(vcmap)
        band["occupied"] = {
            key: receptive_report(vcmap, cells=occupied)[key]
            for key in ("n", "median", "min", "max")
        }
        diagnostics = {
            "cells": len(ids),
            "edges": len(self.edges),
            "events": len(self.buffer),
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "invalid": self.invalid,
            "teams": len(self.buffer.teams()),
            "mean_norm": round(float(np.linalg.norm(out, axis=1).mean()), 6) if len(out) else 0.0,
            "receptive": band,
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
                tuple(sorted(self.cell_teams.get(int(cell), ()))),
            )
            for cell, position in sorted(self.cells.items())
        ]
        return items, rivals, cells
