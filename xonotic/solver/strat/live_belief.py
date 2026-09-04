from __future__ import annotations

import json
import time

import numpy as np

from .buffers import EventKind, Observation, ObservationBuffer
from .featurize import (
    SLOT_DIM,
    UNINFORMATIVE_PRIOR,
    build_observation_slots,
    receptive_report,
    segment_vcells,
    temporal_contraction,
    vcell_from_navigation,
)

BELIEF_BAND = (0.05, 0.15)

from payload.tools.strategy_io_schema import CELL_EXTENT, EVT_KIND

class LiveBelief:
    def __init__(self, decay=8.0, navigation=None):
        self.decay = float(decay)
        self.buffer = ObservationBuffer()
        self.key = None
        self.tick = None
        self.now = 0.0
        self.wall = time.monotonic()
        self.cells = {}
        self.cell_times = {}
        self.cell_teams = {}
        self.edges = set()

        self.nav_links = {}
        self.nav_link_rows = 0
        self.player_cells = {}
        self.deposited_rows = 0
        self.nonfinite_rows = 0
        self.zero_time_rows = 0
        self.parse_error_rows = 0
        self.link_diagnostics = {}
        self._vcmap_cache = None
        self._topology_revision = 0
        self.navigation_vcmap = None
        self.navigation_metadata = {}
        if navigation is not None:
            self.load_navigation(navigation)

    def load_navigation(self, navigation):
        if isinstance(navigation, str):
            with open(navigation) as handle:
                navigation = json.load(handle)
        payload = navigation.get("navigation_realization", navigation) or {}
        if not payload.get("nodes"):
            self.navigation_metadata = {
                "relation": payload.get("relation"),
                "schema": payload.get("schema"),
                "state": "navigation_realization_unavailable",
            }
            return
        self.navigation_vcmap = vcell_from_navigation(payload, BELIEF_BAND)
        self.navigation_metadata = {
            "relation": payload.get("relation"),
            "schema": payload.get("schema"),
            "realization_id": payload.get("realization_id"),
            "node_mass": len(payload.get("nodes", ())),
            "edge_mass": len(payload.get("edges", ())),
            "voronoi_cell_mass": len((payload.get("voronoi") or {}).get("site_nodes", ())),
        }
        self._install_navigation_cells()

    def _install_navigation_cells(self):
        if self.navigation_vcmap is None:
            return
        for cell, position in enumerate(self.navigation_vcmap.centroids):
            self.cells[cell] = np.asarray(position, dtype=np.float64)

    def _cell_key(self, cell, position):
        if self.navigation_vcmap is not None:
            return self.navigation_vcmap.assign_cell(np.asarray(position, dtype=np.float64)[:2])
        return tuple(cell)

    @staticmethod
    def _actuator_cell(position):
        point = np.asarray(position, dtype=np.float64)
        return tuple(np.floor(point[:2] / CELL_EXTENT).astype(np.int64).tolist())

    def reset(self, key=None, tick=None):
        self.buffer.clear()
        self.key = key
        self.tick = tick
        self.now = 0.0
        self.wall = time.monotonic()
        self.cells.clear()
        self.cell_times.clear()
        self.cell_teams.clear()
        self.edges.clear()
        self.nav_links.clear()
        self.nav_link_rows = 0
        self.player_cells.clear()
        self.deposited_rows = 0
        self.nonfinite_rows = 0
        self.zero_time_rows = 0
        self.parse_error_rows = 0
        self.link_diagnostics = {}
        self._vcmap_cache = None
        self._topology_revision = 0
        self._install_navigation_cells()

    def sync(self, key, tick):
        changed = self.key != key or (self.tick is not None and tick < self.tick)
        if changed:
            self.reset(key, tick)
        else:
            self.key = key
            self.tick = tick
        return changed

    def ingest(self, rows, columns):
        pending = []
        for row in np.atleast_2d(rows):
            try:
                values = tuple(float(row[columns[name]]) for name in
                               ("KIND", "TIME", "OBSERVER", "TEAM", "SUBJECT",
                                "CELL_X", "CELL_Y", "TARGET_CELL_X", "TARGET_CELL_Y",
                                "POS_X", "POS_Y", "POS_Z", "RESPAWN_TIME", "HEALTH",
                                "LINK_LENGTH", "AMOUNT"))
                if not np.all(np.isfinite(values)):
                    self.nonfinite_rows += 1
                    continue
                if values[1] <= 0:
                    self.zero_time_rows += int(any(values))
                    continue
                pending.append(values)
            except (IndexError, KeyError, TypeError, ValueError):
                self.parse_error_rows += 1
        pending.sort(key=lambda values: values[1])
        deposited = 0
        for (raw_kind, stamp, observer, team, subject, cell_x, cell_y,
             target_cell_x, target_cell_y, px, py, pz,
             respawn_time, health, link_length, amount) in pending:
            code = int(raw_kind)
            cell = (int(cell_x), int(cell_y))
            target_cell = (int(target_cell_x), int(target_cell_y))
            if code == EVT_KIND["CELL_LINK"]:
                self._nav_link(cell, target_cell, link_length)
                self.now = max(self.now, float(stamp))
                continue
            if code > EVT_KIND["CELL_LINK"]:
                self.now = max(self.now, float(stamp))
                continue
            kind = EventKind.ITEM_DESPAWN if code == 0 else (
                EventKind.ITEM_SPAWN if code == 1 else EventKind.ENEMY_SEEN
            )
            position = (float(px), float(py), float(pz))
            cell = self._cell_key(cell, position)
            payload = {
                "raw_kind": code, "respawn_time": float(respawn_time),
                "health": float(health), "link_length": float(link_length),
                "amount": float(amount),
                "position": position,
            }
            event = self.buffer.observe(Observation(
                int(team), int(observer), float(stamp), cell, kind,
                int(subject), payload,
            ))
            deposited += int(event is not None)
            self.now = max(self.now, float(stamp))
        self.deposited_rows += deposited
        return deposited

    def _nav_link(self, left, right, length):
        if self.navigation_vcmap is not None:
            self.nav_link_rows += 1
            return
        a, b = tuple(left), tuple(right)
        if a == b:
            return
        key = (min(a, b), max(a, b))
        length = float(length)
        if not np.isfinite(length) or length < 0:
            return
        previous = self.nav_links.get(key)
        if previous is None or length < previous:
            self.nav_links[key] = length
            self._topology_revision += 1
        self.nav_link_rows += 1

    def _put_cell(self, cell, position, team=None, stamp=None):
        cell = int(cell) if np.isscalar(cell) else tuple(cell)
        position = np.asarray(position, dtype=np.float64)[:2]
        if cell not in self.cells:
            self._topology_revision += 1
        self.cells[cell] = position
        if stamp is not None:
            self.cell_times[cell] = float(stamp)
        if team is not None:
            self.cell_teams.setdefault(cell, set()).add(int(team))

    def _link(self, left, right):
        if self.navigation_vcmap is not None:
            return
        if left != right:
            edge = tuple(sorted((tuple(left), tuple(right))))
            if edge not in self.edges:
                self.edges.add(edge)
                self._topology_revision += 1

    def _update_cells(self, rows, columns):
        players = []
        by_id = {}
        for row in np.atleast_2d(rows):
            participant = int(row[columns["ID"]])
            team = int(row[columns["TEAM"]])
            position = np.asarray((row[columns["POS_X"]], row[columns["POS_Y"]]),
                                  dtype=np.float64)
            raw_cell = (int(row[columns["CELL_X"]]), int(row[columns["CELL_Y"]]))
            cell = self._cell_key(raw_cell, position)
            previous = self.player_cells.get(participant)
            if previous is not None:
                self._link(previous, cell)
            self.player_cells[participant] = cell
            self._put_cell(cell, position, team, row[columns["ENGINE_TIME"]])
            record = (participant, team, cell, position)
            players.append(record)
            by_id[participant] = record
        for team in self.buffer.teams():
            for event in self.buffer.events(team):
                position = (event.payload or {}).get("position", (0.0, 0.0, 0.0))
                self._put_cell(event.cell, position, int(team), event.t)
                observer = by_id.get(int(event.observer))
                if observer is not None:
                    self._link(event.cell, observer[2])
        return players

    def _vcmap(self):
        if self.navigation_vcmap is not None:
            ids = list(range(self.navigation_vcmap.n_cells))
            index = {cell: cell for cell in ids}
            self.link_diagnostics = {
                **self.navigation_metadata,
                "nav_link_rows": self.nav_link_rows,
                "link_source": "navigation_realization",
                "vcmap_rebuilt": False,
            }
            return ids, index, self.navigation_vcmap
        signature = self._topology_revision
        if self._vcmap_cache is not None and self._vcmap_cache[0] == signature:
            _, ids, index, vcmap, diagnostics = self._vcmap_cache
            self.link_diagnostics = dict(diagnostics, nav_link_rows=self.nav_link_rows,
                                         vcmap_rebuilt=False)
            return ids, index, vcmap
        ids = set(self.cells)
        for left, right in self.nav_links:
            ids.add(left)
            ids.add(right)
        if not ids:
            self.cells[(0, 0)] = np.zeros(2, dtype=np.float64)
            ids = {(0, 0)}
        ids = sorted(ids)
        index = {cell: i for i, cell in enumerate(ids)}
        n = len(ids)
        adjacency = [set() for _ in ids]
        edge_lengths = {}
        for (left, right), length in self.nav_links.items():
            i, j = index[left], index[right]
            adjacency[i].add(j)
            adjacency[j].add(i)
            edge_lengths[(min(i, j), max(i, j))] = float(length)
        linked = len(edge_lengths)
        if not linked:
            for left, right in self.edges:
                if left in index and right in index:
                    adjacency[index[left]].add(index[right])
                    adjacency[index[right]].add(index[left])
        positions = self._node_positions(ids, index)
        vcmap = segment_vcells(positions, adjacency=[sorted(a) for a in adjacency],
                               band=BELIEF_BAND, edge_lengths=edge_lengths)
        self.link_diagnostics = {
            "nav_links": len(self.nav_links),
            "nav_link_rows": self.nav_link_rows,
            "link_edges_used": linked,
            "observed_edges": len(self.edges),
            "link_source": "waypoint_links" if linked else "observed_player_transitions",
            "cells_from_links_only": int(sum(cell not in self.cells for cell in ids)),
            "vcmap_rebuilt": True,
        }
        self._vcmap_cache = (signature, ids, index, vcmap, dict(self.link_diagnostics))
        return ids, index, vcmap

    def _node_positions(self, ids, index):
        positions = np.zeros((len(ids), 2), dtype=np.float64)
        for cell, i in index.items():
            positions[i] = (np.asarray(cell, dtype=np.float64) + 0.5) * CELL_EXTENT
        return positions

    def _slot_rows(self, team, index, vcmap):
        rows = []
        for event in self.buffer.events(team):
            cell = int(event.cell) if np.isscalar(event.cell) else tuple(event.cell)
            if cell not in index:
                continue
            payload = event.payload or {}
            kind = int(payload.get("raw_kind", 0))
            position = payload.get("position", (0.0, 0.0, 0.0))
            slot = np.zeros(SLOT_DIM, dtype=np.float64)
            if 0 <= kind < 4:
                slot[kind] = 1.0
            slot[4:7] = np.asarray(position, dtype=np.float64)
            slot[7:11] = (
                float(payload.get("respawn_time", 0.0)),
                float(payload.get("health", 0.0)),
                float(payload.get("link_length", 0.0)),
                float(payload.get("amount", 0.0)),
            )
            rows.append({
                "cell": int(vcmap.node_cell[index[cell]]),
                "time": float(event.t), "slot": slot,
            })
        return rows

    def chorus(self, rows, columns, now=None):
        players = self._update_cells(rows, columns)
        ids, index, vcmap = self._vcmap()
        stamp = self.now if now is None else float(now)
        teams = sorted({int(record[1]) for record in players})
        blocks = []
        spans = {}
        block_cells = {}
        for team in teams:
            observed, obs_time, seen, slot_cells = build_observation_slots(
                self._slot_rows(team, index, vcmap), vcmap, stamp
            )
            begin = sum(len(block) for block in blocks)
            blocks.append(temporal_contraction(
                observed, obs_time, stamp, self.decay, UNINFORMATIVE_PRIOR, seen
            ))
            spans[team] = (begin, begin + len(observed))
            block_cells[team] = slot_cells
        cell_slots = np.concatenate(blocks, axis=0)
        gigi = np.zeros((len(players), len(cell_slots)), dtype=np.float64)
        for player, (_, team, cell, position) in enumerate(players):
            node = index.get(cell)
            fused = int(vcmap.node_cell[node]) if node is not None else vcmap.assign_cell(position)
            begin, end = spans[team]
            gigi[player, begin:end] = vcmap.spatial_mask(fused)[block_cells[team]]
        self.last_diagnostics = dict(
            receptive_report(vcmap), teams=len(teams), slots=len(cell_slots),
            deposited_rows=self.deposited_rows,
            nonfinite_rows=self.nonfinite_rows, zero_time_rows=self.zero_time_rows,
            parse_error_rows=self.parse_error_rows,
            **self.link_diagnostics,
        )
        return cell_slots, gigi

    def diagnostics(self):
        return dict(getattr(self, "last_diagnostics", {}))

    def instrument_targets(self, rows, columns):
        from .instruments import CellTarget, ItemTarget, RivalTarget

        player_team = {
            int(row[columns["ID"]]): int(row[columns["TEAM"]])
            for row in np.atleast_2d(rows)
        }
        item_events = {}
        rival_events = {}
        for observer_team in self.buffer.teams():
            for event in self.buffer.events(observer_team):
                key = int(event.subject)
                target = item_events if event.kind.is_item_presence else rival_events
                previous = target.get((key, int(observer_team)))
                if previous is None or event.t >= previous.t:
                    target[(key, int(observer_team))] = event
        items = []
        for subject in sorted({key[0] for key in item_events}):
            events = [event for (item, _), event in item_events.items() if item == subject]
            latest = max(events, key=lambda event: event.t)
            position = (latest.payload or {}).get("position", (0.0, 0.0, 0.0))
            respawn = float((latest.payload or {}).get("respawn_time", 0.0))
            items.append(ItemTarget(
                subject, self._actuator_cell(position), tuple(float(value) for value in position),
                float(latest.kind == EventKind.ITEM_SPAWN), respawn, float(latest.t),
                tuple(sorted({team for (item, team) in item_events if item == subject})),
            ))
        rivals = []
        for subject in sorted({key[0] for key in rival_events}):
            events = [event for (rival, _), event in rival_events.items() if rival == subject]
            latest = max(events, key=lambda event: event.t)
            position = (latest.payload or {}).get("position", (0.0, 0.0, 0.0))
            health = float((latest.payload or {}).get("health", 0.0))
            rivals.append(RivalTarget(
                subject, player_team.get(subject, 0), self._actuator_cell(position),
                tuple(float(value) for value in position), health, float(latest.t),
                tuple(sorted({team for (rival, team) in rival_events if rival == subject})),
            ))

        cells = []
        for cell in sorted(self.cells):
            position = np.asarray(self.cells[cell], dtype=np.float64)
            spatial = tuple(position.tolist()) + ((0.0,) if len(position) == 2 else ())
            cells.append(CellTarget(
                self._actuator_cell(spatial), spatial,
                float(self.cell_times.get(cell, 0.0)),
                tuple(sorted(self.cell_teams.get(cell, ()))),
                int(cell) if np.isscalar(cell) else -1,
            ))
        return items, rivals, cells
