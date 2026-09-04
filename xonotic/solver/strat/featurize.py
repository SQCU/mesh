from __future__ import annotations

import heapq

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

_BAND_LO = 0.05
_BAND_HI = 0.15

def _dijkstra_all_pairs(n: int, edges: Mapping) -> np.ndarray:
    dist = np.full((n, n), np.inf, dtype=np.float64)
    for src in range(n):
        dist[src, src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[src, u]:
                continue
            for v, w in edges[u]:
                nd = d + w
                if nd < dist[src, v]:
                    dist[src, v] = nd
                    heapq.heappush(pq, (nd, v))
    return dist

class VCellMap:
    def __init__(self, centroids, areas, map_area, graph_dist, support_radius,
                 node_positions, node_cell, node_edges, band=(_BAND_LO, _BAND_HI)):
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.areas = np.asarray(areas, dtype=np.float64)
        self.map_area = float(map_area)
        self.graph_dist = np.asarray(graph_dist, dtype=np.float64)
        self.support_radius = float(support_radius)
        self.node_positions = np.asarray(node_positions, dtype=np.float64)
        self.node_cell = np.asarray(node_cell, dtype=np.int64)
        self.node_edges = tuple(tuple(row) for row in node_edges)
        self._walking_cache = {}
        self.band = (float(band[0]), float(band[1]))

    @property
    def n_cells(self) -> int:
        return int(self.centroids.shape[0])

    def assign_cell(self, position) -> int:
        p = np.asarray(position, dtype=np.float64)
        width = min(self.node_positions.shape[1], p.size)
        d2 = np.sum((self.node_positions[:, :width] - p[None, :width]) ** 2, axis=-1)
        return int(self.node_cell[int(np.argmin(d2))])

    def assign_node(self, position) -> int:
        p = np.asarray(position, dtype=np.float64)
        width = min(self.node_positions.shape[1], p.size)
        d2 = np.sum((self.node_positions[:, :width] - p[None, :width]) ** 2, axis=-1)
        return int(np.argmin(d2))

    def walking_distance(self, left, right) -> float:
        source = self.assign_node(left)
        target = self.assign_node(right)
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        width = min(self.node_positions.shape[1], left.size, right.size)
        if source == target:
            return float(np.linalg.norm(left[:width] - right[:width]))
        distances = self._walking_cache.get(source)
        if distances is None:
            distances = np.full(len(self.node_positions), np.inf, dtype=np.float64)
            distances[source] = 0.0
            pending = [(0.0, source)]
            while pending:
                distance, node = heapq.heappop(pending)
                if distance > distances[node]:
                    continue
                for neighbor, length in self.node_edges[node]:
                    candidate = distance + length
                    if candidate < distances[neighbor]:
                        distances[neighbor] = candidate
                        heapq.heappush(pending, (candidate, neighbor))
            self._walking_cache[source] = distances
        source_offset = np.linalg.norm(left[:width] - self.node_positions[source, :width])
        target_offset = np.linalg.norm(right[:width] - self.node_positions[target, :width])
        return float(source_offset + distances[target] + target_offset)

    def receptive_fraction(self, cell_idx: int) -> float:
        within = self.graph_dist[cell_idx] <= self.support_radius + 1e-9
        return float(self.areas[within].sum() / self.map_area)

    def spatial_mask(self, cell_idx: int) -> np.ndarray:
        d = self.graph_dist[cell_idx]
        R = self.support_radius
        with np.errstate(over="ignore", invalid="ignore"):
            g = np.exp(-4.0 * (d / R) ** 2) if R > 0 else (d == 0).astype(np.float64)
        g = np.where(d <= R + 1e-9, g, 0.0)
        g = np.where(np.isfinite(d), g, 0.0)
        return g.astype(np.float64)

def segment_vcells(
    node_positions,
    adjacency: Optional[Sequence] = None,
    map_area: Optional[float] = None,
    cell_areas: Optional[Sequence] = None,
    band=(_BAND_LO, _BAND_HI),
    sliver_floor_frac: float = 0.25,
    edge_lengths: Optional[Mapping] = None,
    node_cell: Optional[Sequence] = None,
) -> VCellMap:
    pos = np.asarray(node_positions, dtype=np.float64)
    if pos.ndim != 2:
        raise ValueError("node_positions must be (N, dim)")
    n = pos.shape[0]
    if n == 0:
        raise ValueError("need at least one item/waypoint node to segment V-cells")
    lo, hi = float(band[0]), float(band[1])

    if cell_areas is None:
        areas0 = np.ones(n, dtype=np.float64)
    else:
        areas0 = np.asarray(cell_areas, dtype=np.float64).copy()
        if areas0.shape != (n,):
            raise ValueError("cell_areas must be (N,)")
    if map_area is None:
        map_area = float(areas0.sum())
    map_area = float(map_area)

    if areas0.sum() > 0:
        areas0 = areas0 * (map_area / areas0.sum())

    if adjacency is None:
        adj = [[] for _ in range(n)]
    else:
        adj = [sorted({int(j) for j in nb if int(j) != i}) for i, nb in enumerate(adjacency)]

        for i in range(n):
            for j in adj[i]:
                if i not in adj[j]:
                    adj[j].append(i)
        adj = [sorted(set(s)) for s in adj]

    if node_cell is None:
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        cell_area = areas0.copy()
        positive_areas = areas0[areas0 > 0]
        sliver = sliver_floor_frac * float(np.median(positive_areas)) if len(positive_areas) else 0.0
        ceiling = hi * map_area
        for i in sorted(range(n), key=lambda a: (areas0[a], a)):
            ri = find(i)
            if cell_area[ri] >= sliver:
                continue
            best, best_area = None, None
            for j in adj[i]:
                rj = find(j)
                if rj == ri:
                    continue
                merged = cell_area[ri] + cell_area[rj]
                if merged <= ceiling and (best is None or cell_area[rj] < best_area):
                    best, best_area = rj, cell_area[rj]
            if best is not None:
                parent[ri] = best
                cell_area[best] += cell_area[ri]
        roots = sorted({find(i) for i in range(n)})
        root_to_cell = {root: cell for cell, root in enumerate(roots)}
        node_cell = np.array([root_to_cell[find(i)] for i in range(n)], dtype=np.int64)
    else:
        supplied = np.asarray(node_cell, dtype=np.int64)
        if supplied.shape != (n,) or np.any(supplied < 0):
            raise ValueError("node_cell must assign every navigation node")
        identities = sorted(set(int(value) for value in supplied))
        remap = {identity: cell for cell, identity in enumerate(identities)}
        node_cell = np.asarray([remap[int(value)] for value in supplied], dtype=np.int64)
    C = int(node_cell.max()) + 1

    centroids = np.zeros((C, pos.shape[1]), dtype=np.float64)
    areas = np.zeros(C, dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for i in range(n):
        c = node_cell[i]
        centroids[c] += pos[i]
        areas[c] += areas0[i]
        counts[c] += 1.0
    centroids /= np.maximum(counts[:, None], 1.0)

    lengths = {}
    if edge_lengths:
        for (a, b), value in edge_lengths.items():
            a, b = int(a), int(b)
            if not (0 <= a < n and 0 <= b < n):
                continue
            lengths[(min(a, b), max(a, b))] = float(value)
    pair_weight = {}
    for i in range(n):
        ci = int(node_cell[i])
        for j in adj[i]:
            cj = int(node_cell[j])
            if ci == cj:
                continue
            key = (min(ci, cj), max(ci, cj))
            supplied = lengths.get((min(i, int(j)), max(i, int(j))))
            w = (float(supplied) if supplied is not None
                 else float(np.linalg.norm(centroids[ci] - centroids[cj])))
            if key not in pair_weight or w < pair_weight[key]:
                pair_weight[key] = w
    edges = [list() for _ in range(C)]
    for (ci, cj), w in pair_weight.items():
        edges[ci].append((cj, w))
        edges[cj].append((ci, w))
    graph_dist = _dijkstra_all_pairs(C, edges)

    candidates = np.unique(graph_dist[np.isfinite(graph_dist)])
    if candidates.size == 0:
        support_radius = 0.0
    else:
        def _median_frac(R):
            within = graph_dist <= R + 1e-9
            return float(np.median((within * areas[None, :]).sum(axis=1) / map_area))

        if _median_frac(float(candidates[-1])) < lo:
            support_radius = float(candidates[-1])
        else:
            low, high = 0, candidates.size - 1
            while low < high:
                mid = (low + high) // 2
                if _median_frac(float(candidates[mid])) >= lo:
                    high = mid
                else:
                    low = mid + 1
            support_radius = float(candidates[low])

    node_edges = [list() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            key = (min(i, int(j)), max(i, int(j)))
            length = lengths.get(key)
            if length is None:
                length = float(np.linalg.norm(pos[i] - pos[int(j)]))
            node_edges[i].append((int(j), float(length)))
    return VCellMap(centroids, areas, map_area, graph_dist, support_radius,
                    pos, node_cell, node_edges, band=(lo, hi))

def vcell_from_navigation(realization, band=(_BAND_LO, _BAND_HI)):
    payload = realization.get("navigation_realization", realization)
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    positions = nodes[:, :2]
    adjacency = [set() for _ in nodes]
    lengths = {}
    for left, right, length in payload.get("edges", ()):
        left, right = int(left), int(right)
        adjacency[left].add(right)
        adjacency[right].add(left)
        lengths[(min(left, right), max(left, right))] = float(length)
    measure = np.asarray(payload.get("node_measure", np.ones(len(nodes))), dtype=np.float64)
    total = float(payload.get("total_measure", measure.sum()))
    if total <= 0:
        measure = np.ones(len(nodes), dtype=np.float64)
        total = float(len(nodes))
    return segment_vcells(
        positions, [sorted(row) for row in adjacency], total, measure,
        band=band, edge_lengths=lengths,
        node_cell=(payload.get("voronoi") or {}).get("owner"),
    )

SLOT_FIELDS = (
    "item_gone", "item_here", "enemy_here", "rival_here",
    "position_x", "position_y", "position_z", "respawn_time",
    "health", "link_length", "amount",
)
SLOT_DIM = len(SLOT_FIELDS)
_SLOT_INDEX = {name: i for i, name in enumerate(SLOT_FIELDS)}

def build_observation_slots(rows: Iterable, vcmap: VCellMap, now: float):
    slots = []
    times = []
    cells = []

    def _get(row, key, default=None):
        if isinstance(row, Mapping):
            return row.get(key, default)
        return getattr(row, key, default)

    for row in rows:
        cell = _get(row, "cell")
        if cell is None:
            posn = _get(row, "position")
            if posn is None:
                raise ValueError("observation row needs 'cell' or 'position'")
            cell = vcmap.assign_cell(posn)
        cells.append(int(cell))
        times.append(float(_get(row, "time", now)))
        slot = _get(row, "slot")
        if slot is not None:
            v = np.asarray(slot, dtype=np.float64).reshape(-1)
            if v.shape[0] != SLOT_DIM:
                raise ValueError(f"slot must be length {SLOT_DIM}")
        else:
            v = np.zeros(SLOT_DIM, dtype=np.float64)
            for name, idx in _SLOT_INDEX.items():
                if name == "seen":
                    continue
                val = _get(row, name)
                if val is not None:
                    v[idx] = float(val)
        slots.append(v)
    return (
        np.asarray(slots, dtype=np.float64).reshape(-1, SLOT_DIM),
        np.asarray(times, dtype=np.float64),
        np.ones(len(slots), dtype=bool),
        np.asarray(cells, dtype=np.int64),
    )

def temporal_contraction(f_obs, obs_time, now: float, T: float,
                         f_prior=None, seen=None) -> np.ndarray:
    f_obs = np.asarray(f_obs, dtype=np.float64)
    E, F = f_obs.shape
    if T <= 0:
        raise ValueError("T (forgetting time-constant) must be > 0")
    dt = float(now) - np.asarray(obs_time, dtype=np.float64)
    dt = np.maximum(dt, 0.0)
    rho = np.exp(-dt / float(T))
    if seen is not None:
        rho = np.where(np.asarray(seen, dtype=bool), rho, 0.0)
    if f_prior is None:
        prior = np.zeros((E, F), dtype=np.float64)
    else:
        prior = np.asarray(f_prior, dtype=np.float64)
        if prior.ndim == 1:
            prior = np.broadcast_to(prior, (E, F))
    return rho[:, None] * f_obs + (1.0 - rho[:, None]) * prior

UNINFORMATIVE_PRIOR = np.zeros(SLOT_DIM, dtype=np.float64)

def receptive_report(vcmap: VCellMap, cells=None) -> dict:
    idx = range(vcmap.n_cells) if cells is None else [int(c) for c in cells]
    fr = np.asarray([vcmap.receptive_fraction(int(c)) for c in idx], dtype=np.float64)
    if fr.size == 0:
        return {"n": 0, "median": 0.0, "min": 0.0, "max": 0.0,
                "band": list(vcmap.band), "in_band": False,
                "support_radius": vcmap.support_radius, "n_cells": vcmap.n_cells}
    lo, hi = vcmap.band
    med = float(np.median(fr))
    return {
        "n": int(fr.size),
        "median": round(med, 6),
        "min": round(float(fr.min()), 6),
        "max": round(float(fr.max()), 6),
        "band": [float(lo), float(hi)],
        "in_band": bool(lo - 1e-9 <= med <= hi + 1e-9),
        "support_radius": round(float(vcmap.support_radius), 6),
        "n_cells": int(vcmap.n_cells),
    }

__all__ = [
    "SLOT_FIELDS", "SLOT_DIM", "UNINFORMATIVE_PRIOR",
    "VCellMap", "segment_vcells", "vcell_from_navigation", "build_observation_slots", "temporal_contraction",
    "receptive_report",
]
