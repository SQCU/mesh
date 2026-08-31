"""Belief pipeline — V-cell segmentation, RHO's contraction, GIGI's mask.

The belief INTEGRATION is NOT here: ``gigi @ phil(wally, cell_slots)`` in
strategy.py is the one definition. A constant PHI and a second integration used
to live in this file and are deleted — a fixed Phi is exactly the narrow
hand-authored bottleneck SPEC §7 forbids, and PHIL is a learned cast member.
This module produces ``cell_slots`` and ``gigi`` for the composer and stops.

This is the deterministic featurization the strategy operator's ``beta`` half is
computed by.  It is plain numpy: nothing here differentiates.  Its input is the
per-team **observation buffer** (frustum + LOS + 2-V-cell gated in the engine),
never omniscient world state, so enemy positions reach the operator only as
events spatialized here.

The four committed stages, in order:

  2. **V-cell segmentation** — Voronoi atoms over item/waypoint nodes, fuse
     contiguous *navigable* atoms, then SET the support radius so the median
     receptive field lands in the ``[5%, 15%]`` band of map area.  The horizon
     is an output of this construction, not a tuned constant.
     -> :func:`segment_vcells` -> :class:`VCellMap` (:meth:`receptive_fraction`
     is the two-sided quantity the band is checked against).
  3. **Temporal contraction** — ``f^eff = rho f^obs + (1-rho) f^prior``,
     ``rho = exp(-dt/T)``.  -> :func:`temporal_contraction`.
  4. **Spatial mask** — bounded-support graph-distance kernel
     ``g(dist_graph(c(b), .))``.  -> :meth:`VCellMap.spatial_mask`.
  5. **Egocentric integration** — ``beta_b = sum_c g_c * Phi f_c^eff``, the ONLY
     spatial mixing operator in the system.  -> :func:`egocentric_integration`,
     :func:`belief`, :func:`beliefs_for_bots`.

There is exactly one implementation of each stage and it lives here;
``live_belief.LiveBelief`` is an ADAPTER that folds the live event buffer into
observation rows and then calls these functions.  Nothing may re-inline them.

``PHI`` and ``UNINFORMATIVE_PRIOR`` are the canonical stage-5 projection and
stage-3 prior; ``PHI`` is a fixed (BELIEF_RANK, SLOT_DIM) read-out matrix — with
SLOT_DIM = 7 its rank is at most 7, it is not a rank reduction, it is the fixed
slot->belief read-out the operator's ``d_beta`` expects.
"""
from __future__ import annotations

import heapq

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np



# Default band on the receptive-field fraction of map area (payload-spec §2.2.2).
_BAND_LO = 0.05
_BAND_HI = 0.15


# --------------------------------------------------------------------------- #
# Stage 2: V-cell segmentation (Voronoi over nodes + navigable fusion).
# --------------------------------------------------------------------------- #

def _knn_adjacency(positions: np.ndarray, k: int) -> list:
    """Symmetric k-nearest-neighbour adjacency -- the LAST-RESORT graph, not the real one.

    The real navigable graph is map data and it is now streamed: the engine walks
    ``g_waypoints`` and emits every stock ``waypoint_get_link`` whose endpoint hashes to
    a different V-cell as a ``PLC_EVT_KIND_CELL_LINK`` (kind 4) perception row, with the
    link's own length. ``live_belief.LiveBelief`` collects those and passes them here as
    ``adjacency`` + ``edge_lengths``, which is the payload-spec §2.2.2 "fuse contiguous
    navigable paths". This kNN construction is reached only when a caller supplies no
    adjacency at all -- e.g. the first ticks of a match, before the engine's link sweep
    has produced any rows.
    """
    n = len(positions)
    k = max(1, min(k, n - 1))
    adj = [set() for _ in range(n)]
    d2 = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    for i in range(n):
        order = np.argsort(d2[i], kind="stable")
        added = 0
        for j in order:
            if j == i:
                continue
            adj[i].add(int(j))
            adj[int(j)].add(i)  # symmetrize -> undirected navigable graph
            added += 1
            if added >= k:
                break
    return [sorted(s) for s in adj]


def _dijkstra_all_pairs(n: int, edges: Mapping) -> np.ndarray:
    """All-pairs shortest navigable graph distance (Dijkstra per source, plain python).

    ``edges[i]`` = iterable of ``(j, weight)``. Unreachable pairs get ``inf``. Deterministic.
    """
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
    """The stage-2 segmented map: V-cells, navigable graph, and the bounded horizon.

    Holds, over ``C`` fused V-cells (payload-spec §2.2.2):
      * ``centroids`` ``(C, dim)`` -- cell centroids (mean of member node positions).
      * ``areas`` ``(C,)`` -- per-cell area (normalized so ``sum == map_area``).
      * ``map_area`` -- total map area.
      * ``graph_dist`` ``(C, C)`` -- all-pairs navigable graph distance (Dijkstra),
        the argument to the stage-4 mask ``g(dist_graph(c(b), c))``.
      * ``support_radius`` -- the bounded support ``R`` of the mask; this IS the horizon
        / context mask, chosen so the median receptive-field fraction lands in ``band``
        (§2.2: "The horizon is not a parameter choice ... set by the stage-2 construction").
      * ``node_cell`` ``(N,)`` -- for each original node, its fused cell id (Voronoi +
        fusion membership), so ``assign_cell`` can place a bot by nearest node.

    The mask bandwidth is ``sigma = support_radius / 2`` so ``g`` decays to
    ``exp(-4) ~ 0.018`` at the support edge -- effectively **bounded support** (§2.2.4).
    """

    def __init__(self, centroids, areas, map_area, graph_dist, support_radius,
                 node_positions, node_cell, band=(_BAND_LO, _BAND_HI)):
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.areas = np.asarray(areas, dtype=np.float64)
        self.map_area = float(map_area)
        self.graph_dist = np.asarray(graph_dist, dtype=np.float64)
        self.support_radius = float(support_radius)
        self.node_positions = np.asarray(node_positions, dtype=np.float64)
        self.node_cell = np.asarray(node_cell, dtype=np.int64)
        self.band = (float(band[0]), float(band[1]))

    @property
    def n_cells(self) -> int:
        return int(self.centroids.shape[0])

    def assign_cell(self, position) -> int:
        """The V-cell a world position falls in = the cell of its nearest item/waypoint node.

        Voronoi-over-nodes assignment (payload-spec §2.2.2): nearest node, then that
        node's fused cell. This is how a bot's own cell ``c(b)`` is resolved for the
        stage-4 mask. Deterministic nearest (stable argmin).
        """
        p = np.asarray(position, dtype=np.float64)
        d2 = np.sum((self.node_positions - p[None, :]) ** 2, axis=-1)
        return int(self.node_cell[int(np.argmin(d2))])

    def receptive_fraction(self, cell_idx: int) -> float:
        """Fraction of map area within the bounded mask support of ``cell_idx`` (§2.2.2 band).

        The two-sided quantity the segmentation targets to ``[~5%, ~15%]``: the summed
        area of cells with ``graph_dist <= support_radius`` from ``cell_idx``, over map area.
        """
        within = self.graph_dist[cell_idx] <= self.support_radius + 1e-9
        return float(self.areas[within].sum() / self.map_area)

    def spatial_mask(self, cell_idx: int) -> np.ndarray:
        """**Stage 4**: bounded-support graph-distance kernel ``g(dist_graph(c(b), .))``.

        Returns ``(C,)`` weights ``g_c = exp(-4 (d/R)^2)`` for ``d = graph_dist[cell_idx, c]``
        within the support radius ``R`` (the horizon), and exactly ``0`` beyond it --
        bounded support (payload-spec §2.2.4). A parallel weighting, not a recurrence; the
        support radius is the context mask set by stage 2. Unreachable cells (``inf``
        distance) get weight 0.
        """
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
    knn: int = 6,
    sliver_floor_frac: float = 0.25,
    edge_lengths: Optional[Mapping] = None,
) -> VCellMap:
    """**Stage 2** -- Voronoi over item/waypoint nodes, fuse navigable cells, set horizon.

    Implements payload-spec §2.2.2: partition the map into Voronoi cells over
    item/waypoint nodes, fuse contiguous **navigable** paths until the stage-4
    distance-decay mask bounds each bot's receptive field, two-sided, to
    ``[band_lo, band_hi]`` of map area (default ``[5%, 15%]``). The receptive field
    (the horizon / context mask) is NOT a free parameter -- it is *the output* of this
    construction (§2.2: "there is no fixed-vs-learned fork").

    Construction (deterministic):
      1. Each item/waypoint node seeds one Voronoi atom cell.
      2. Build the navigable graph: ``adjacency`` if given (the real waypoint links,
         streamed as kind-4 perception rows), else a symmetric kNN fallback
         (:func:`_knn_adjacency`).
      3. **Fuse** contiguous navigable atoms whose area is a *sliver* (below
         ``sliver_floor_frac * band_hi`` of map area) into their smallest navigable
         neighbour, never letting a fused cell exceed ``band_hi`` of map area -- so no
         single cell alone can blow the receptive-field ceiling. (With uniform equal
         areas this is a no-op; it matters when ``cell_areas`` is heterogeneous.)
      4. Recompute the fused navigable graph and its all-pairs graph distances.
      5. **Set the horizon**: pick the support radius ``R`` (smallest over the distinct
         inter-cell graph distances) whose *median* receptive-field fraction is ``>=
         band_lo``; if that median would exceed ``band_hi`` it is the tightest feasible
         radius (a single cell already sits under the ceiling by step 3). ``R`` is the
         bounded support of the stage-4 mask.

    Parameters
    ----------
    node_positions : ``(N, dim)`` item/waypoint node coordinates.
    adjacency      : optional per-node iterable of neighbour node indices (navigable
                     links). If ``None``, a symmetric kNN graph is derived.
    edge_lengths   : optional ``{(i, j): length}`` over NODE index pairs (order
                     irrelevant) giving the real traversal length of a navigable
                     link -- the engine's own ``waypoint_get_link`` length, streamed
                     as ``PLC_EVT_KIND_CELL_LINK``. Where a fused cell pair has a
                     supplied length the graph edge uses it (shortest constituent
                     link); where it does not, the centroid distance is used.
    map_area       : total map area; defaults to ``sum(cell_areas)`` or ``N`` (unit atoms).
    cell_areas     : optional per-node atom area; defaults to uniform ``map_area / N``.
    band           : ``(lo, hi)`` receptive-field fraction band; default ``(0.05, 0.15)``.
    knn            : neighbours for the fallback kNN navigable graph.
    sliver_floor_frac : sliver threshold as a fraction of the MEDIAN atom area (only
                     atoms below this fuse; a uniform grid is untouched).

    Returns a :class:`VCellMap`.
    """
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
    # Normalize atom areas so they sum to map_area (keeps fractions well-defined).
    if areas0.sum() > 0:
        areas0 = areas0 * (map_area / areas0.sum())

    if adjacency is None:
        adj = _knn_adjacency(pos, knn)
    else:
        adj = [sorted({int(j) for j in nb if int(j) != i}) for i, nb in enumerate(adjacency)]
        # symmetrize supplied edges (navigable links are undirected)
        for i in range(n):
            for j in adj[i]:
                if i not in adj[j]:
                    adj[j].append(i)
        adj = [sorted(set(s)) for s in adj]

    # --- step 3: fuse navigable slivers via union-find, respecting the ceiling. ---
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell_area = areas0.copy()
    # A sliver is an atom much smaller than the TYPICAL atom (relative to the median
    # atom area), not relative to the whole map -- so a uniform node grid (all atoms the
    # same size) is left untouched and only genuine outlier slivers get fused. Merges are
    # still capped by the receptive-field ceiling so no fused cell exceeds band_hi.
    sliver = sliver_floor_frac * float(np.median(areas0))
    ceiling = hi * map_area
    # Deterministic order: smallest atom first.
    for i in sorted(range(n), key=lambda a: (areas0[a], a)):
        ri = find(i)
        if cell_area[ri] >= sliver:
            continue
        # smallest navigable neighbour cell we can merge into without exceeding ceiling
        best, best_area = None, None
        for j in adj[i]:
            rj = find(j)
            if rj == ri:
                continue
            merged = cell_area[ri] + cell_area[rj]
            if merged <= ceiling and (best is None or cell_area[rj] < best_area):
                best, best_area = rj, cell_area[rj]
        if best is not None:
            rj = best
            parent[ri] = rj
            cell_area[rj] += cell_area[ri]

    # relabel roots -> contiguous cell ids
    roots = sorted({find(i) for i in range(n)})
    root_to_cell = {r: c for c, r in enumerate(roots)}
    node_cell = np.array([root_to_cell[find(i)] for i in range(n)], dtype=np.int64)
    C = len(roots)

    centroids = np.zeros((C, pos.shape[1]), dtype=np.float64)
    areas = np.zeros(C, dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for i in range(n):
        c = node_cell[i]
        centroids[c] += pos[i]
        areas[c] += areas0[i]
        counts[c] += 1.0
    centroids /= np.maximum(counts[:, None], 1.0)

    # --- step 4: fused navigable graph + all-pairs graph distance. ---
    # A fused cell pair may be joined by several node-level links; the traversal
    # cost between the two cells is the SHORTEST of them. Real link lengths win
    # over the centroid-distance fallback, which only stands in for edges whose
    # length the map never supplied (an observed cell transition, say).
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

    # --- step 5: set the horizon R so median receptive fraction lands in band. ---
    # median_frac(R) is nondecreasing in R (widening the support can only add
    # cells), so the smallest admissible R is found by bisection over the
    # distinct finite graph distances instead of a linear scan -- same answer,
    # O(log|candidates|) mask evaluations, which is what makes stage 2
    # affordable at live cadence.
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

    return VCellMap(centroids, areas, map_area, graph_dist, support_radius,
                    pos, node_cell, band=(lo, hi))


# --------------------------------------------------------------------------- #
# Observation buffer -> per-cell slots (incomplete-information, per-team).
# --------------------------------------------------------------------------- #

# Per-cell slot layout (payload-spec §2.2.2). Deterministic, fixed width F = 7.
SLOT_FIELDS = (
    "item_type",         # numeric item-type code (0 = none)
    "respawn_phase",     # respawn-phase estimate in [0, 1] (0 = ready / unknown)
    "standability",      # navigable/standable score in [0, 1]
    "lane_membership",   # golden-path lane id the cell belongs to (0 = none)
    "last_threat",       # last observed threat magnitude at the cell
    "observed_enemy",    # observed enemy presence (count / indicator)
    "seen",              # 1.0 if any observation deposited here, else 0.0 (prior gate)
)
SLOT_DIM = len(SLOT_FIELDS)
_SLOT_INDEX = {name: i for i, name in enumerate(SLOT_FIELDS)}


def build_cell_slots(rows: Iterable, vcmap: VCellMap, now: float):
    """Fold the per-team observation buffer into per-cell slots ``f_c^obs`` + observe times.

    This is the incomplete-information ingest of payload-spec §2.2.1: each row is a
    timestamped **contextual event some bot actually saw** (already frustum + LOS +
    2-V-cell gated upstream -- that gate is engine ``[BUILD]``; nothing enters here that
    no bot observed). We deposit each event at its V-cell, keeping the **most recent**
    event per cell (later ``time`` wins) and stamping the cell's last-observed time. A
    cell with no observation stays unseen (``seen = 0``) -> stage 3 relaxes it fully to
    the prior (genuinely hidden; the stealth mechanic of §2.2.1 falls straight out).

    Row schema (mapping or object with attributes). Position OR cell required:
      * ``cell``            : V-cell id, OR
      * ``position``        : world position -> ``vcmap.assign_cell`` (Voronoi).
      * ``time``            : event timestamp (defaults to ``now``).
      * ``slot``            : optional length-``SLOT_DIM`` vector to write wholesale, OR
      * any of ``SLOT_FIELDS`` names as scalar fields (item_type, respawn_phase,
        standability, lane_membership, last_threat, observed_enemy).

    Returns ``(f_obs, obs_time, seen)``:
      * ``f_obs``   ``(C, SLOT_DIM)`` per-cell observed slot vectors (0 where unseen).
      * ``obs_time````(C,)`` last-observed timestamp per cell (``now`` where unseen; the
                    resulting ``dt = 0`` is harmless -- ``seen = 0`` still forces prior).
      * ``seen``    ``(C,)`` bool: whether any event was deposited (the temporal gate).
    """
    C = vcmap.n_cells
    f_obs = np.zeros((C, SLOT_DIM), dtype=np.float64)
    obs_time = np.full(C, float(now), dtype=np.float64)
    latest = np.full(C, -np.inf, dtype=np.float64)
    seen = np.zeros(C, dtype=bool)

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
        cell = int(cell)
        t = float(_get(row, "time", now))
        if t < latest[cell]:
            continue  # keep the most-recent event per cell
        latest[cell] = t
        obs_time[cell] = t
        seen[cell] = True
        slot = _get(row, "slot")
        if slot is not None:
            v = np.asarray(slot, dtype=np.float64).reshape(-1)
            if v.shape[0] != SLOT_DIM:
                raise ValueError(f"slot must be length {SLOT_DIM}")
            f_obs[cell] = v
        else:
            for name, idx in _SLOT_INDEX.items():
                if name == "seen":
                    continue
                val = _get(row, name)
                if val is not None:
                    f_obs[cell, idx] = float(val)
        f_obs[cell, _SLOT_INDEX["seen"]] = 1.0
    return f_obs, obs_time, seen


# --------------------------------------------------------------------------- #
# Stage 3: temporal contraction (the buffer forgets).
# --------------------------------------------------------------------------- #

def temporal_contraction(f_obs, obs_time, now: float, T: float,
                         f_prior=None, seen=None) -> np.ndarray:
    """**Stage 3** -- ``f_c^eff = rho(dt) f_c^obs + (1 - rho(dt)) f_c^prior``.

    payload-spec §2.2.3: ``rho(dt) = exp(-dt / T)`` with ``dt = now - obs_time``. Stale
    observations relax to an uninformative prior -- the buffer forgets. A never-observed
    cell (``seen[c] == False``) is forced fully to the prior (``rho = 0``) regardless of
    its stamped time, so an unseen body/pickup is absent from the belief (§2.2.1 stealth).

    Parameters
    ----------
    f_obs   : ``(C, F)`` observed slots (from :func:`build_cell_slots`).
    obs_time: ``(C,)`` per-cell last-observed timestamps.
    now     : current strategy-step time.
    T       : forgetting time-constant (``> 0``).
    f_prior : ``(C, F)`` or ``(F,)`` uninformative prior; default zeros (fully forgets).
    seen    : optional ``(C,)`` bool mask; unseen cells clamp ``rho = 0``.

    Returns ``f_eff`` ``(C, F)``.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)
    C, F = f_obs.shape
    if T <= 0:
        raise ValueError("T (forgetting time-constant) must be > 0")
    dt = float(now) - np.asarray(obs_time, dtype=np.float64)
    dt = np.maximum(dt, 0.0)
    rho = np.exp(-dt / float(T))
    if seen is not None:
        rho = np.where(np.asarray(seen, dtype=bool), rho, 0.0)
    if f_prior is None:
        prior = np.zeros((C, F), dtype=np.float64)
    else:
        prior = np.asarray(f_prior, dtype=np.float64)
        if prior.ndim == 1:
            prior = np.broadcast_to(prior, (C, F))
    return rho[:, None] * f_obs + (1.0 - rho[:, None]) * prior


# --------------------------------------------------------------------------- #
# Stage 5: egocentric low-rank integration (the ONLY spatial mixing operator).
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Canonical stage-3 prior and stage-5 read-out (one definition, used by everyone)
# --------------------------------------------------------------------------- #

# Uninformative prior a never-observed / fully-forgotten cell relaxes to.
# Ordering matches SLOT_FIELDS: item_type, respawn_phase, standability,
# lane_membership, last_threat, observed_enemy, seen.  "Uninformative" = the
# item is as likely present as absent, the cell is assumed standable, and no
# threat / enemy / observation is asserted.
UNINFORMATIVE_PRIOR = np.asarray((0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)

# The fixed stage-5 read-out Phi (BELIEF_RANK, SLOT_DIM).  Its rank is at most
# SLOT_DIM = 7; it is the fixed slot -> belief read-out whose width the strategy
# operator's d_beta expects, NOT a rank reduction.
PHI = np.asarray([
    (1,  0, 0, 0, 0, 0, 0),   # item availability
    (0,  1, 0, 0, 0, 0, 0),   # respawn phase
    (0,  0, 0, 0, 1, 0, 0),   # last threat
    (0,  0, 0, 0, 0, 1, 0),   # observed enemy presence
    (0,  0, 0, 0, 0, 0, 1),   # seen / not seen (the incomplete-information gate)
    (0,  0, 1, 0, 0, 0, 0),   # standability
    (1, -1, 0, 0, 0, 0, 0),   # availability net of respawn wait
    (0,  0, 0, 0, 1, 1, 0),   # threat + presence (danger)
], dtype=np.float64)


def receptive_report(vcmap: VCellMap, cells=None) -> dict:
    """Measure the stage-2 receptive-field band on a concrete VCellMap.

    Returns the min / median / max of :meth:`VCellMap.receptive_fraction` over
    ``cells`` (default: every cell), the band, and whether the MEDIAN lands
    inside it.  This is the quantity stage 2 targets; without it the ``[5%,15%]``
    bound is an unchecked claim.
    """
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
    "SLOT_FIELDS", "SLOT_DIM", "PHI", "BELIEF_RANK", "UNINFORMATIVE_PRIOR",
    "VCellMap", "segment_vcells", "build_cell_slots", "temporal_contraction",
    "egocentric_integration", "belief", "beliefs_for_bots", "receptive_report",
]
