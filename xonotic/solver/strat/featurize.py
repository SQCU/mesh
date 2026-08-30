"""Belief pipeline + global feature assembly (the ONLY spatial mixing operator).

This module is the deterministic featurization of `design/payload-spec.md` §2.2
("Observation buffer -> belief") together with the global-feature assembly that
feeds the learned query ``q_b = W_q [x_b ; beta_b]`` of §2.3. It is **computed,
deterministic, numpy/plain-python** -- it is the ``b`` (belief) half of the
computed features ``(s, b, PW, SUCC)`` that `design/rl-training-spec.md` §4 lists
as "computed (deterministic)" and §2.1 marks ``stopgrad`` into the policy
gradient. Nothing here differentiates; the learned surface (``W_q``/``W_k``/``W_v``,
the DPP head, the value head -- mlx) reads ``beta_b`` and ``s`` already detached.

The pipeline runs under INCOMPLETE INFORMATION (payload-spec §2.2 opening: "reason
under incomplete information rather than off omniscient world state ... everything
spatial enters here"). Its input is the per-team **observation buffer** -- the
timestamped contextual events some bot actually saw (frustum + LOS + 2-V-cell gated;
that gate is engine ``[BUILD]`` per §2.2.1 / §4.5, upstream of this file) -- NOT
omniscient world state. Enemy positions reach the strategy operator ONLY as observed
events spatialized here (§2.2.1: "Enemy positions are featurized ONLY through here").

The four committed stages (payload-spec §2.2), in order:

  2. **V-cell segmentation** -- Voronoi cells over item/waypoint nodes, fuse
     contiguous *navigable* paths until the stage-4 distance-decay mask bounds each
     bot's receptive field two-sided to ``[~5%, ~15%]`` of map area. The horizon is
     NOT a free parameter -- it is set by this construction (§2.2: "The horizon is
     not a parameter choice ... there is no fixed-vs-learned fork").
     -> :func:`segment_vcells` returning a :class:`VCellMap`.
  3. **Temporal contraction** -- ``f_c^eff = rho(dt)*f_c^obs + (1-rho(dt))*f_c^prior``,
     ``rho(dt) = exp(-dt/T)``. Stale observations relax to an uninformative prior:
     the buffer forgets. -> :func:`temporal_contraction`.
  4. **Spatial mask** -- a bounded-support graph-distance kernel
     ``g(dist_graph(c(b), c))`` weights cells by navigable graph distance from the
     bot's cell; its support radius IS the horizon / context mask. A parallel
     operation, not a recurrence. -> :meth:`VCellMap.spatial_mask`.
  5. **Egocentric low-rank integration** --
     ``beta_b = sum_c g(dist_graph(c(b), c)) * Phi * f_c^eff`` (``Phi`` low-rank).
     Precompute ``Phi * f_c^eff`` once over the map = ``O(C*rank)``, then
     ``O(horizon)`` per bot -> scales with MAP SIZE, not player count (§2.2.5). Two
     bots in the same cell with the same observations get the same ``beta_b`` because
     their inputs are identical; there is no "team belief" object.
     -> :func:`egocentric_integration`, and the end-to-end :func:`belief`.

This stage-5 integration is the system's ONLY spatial mixing operator (§2.2: "We
never introduced attention or any other spatial mixer"); its kernel ``g`` plays the
role softmax attention would. ``x_b`` is the bot's OWN known state; ``beta_b`` is the
observed, occlusion-gated, spatialized world around it.

Global feature assembly (payload-spec §2.3, rl-training-spec §1). :func:`assemble_features`
concatenates ``[x_b ; beta_b ; s ; PW/SUCC feature]`` into the vector the learned query
projects, and **guarantees cartstate ``s`` is a member** -- rl-training-spec §1: "``s`` =
cartstate ... a **guaranteed member of the global feature vector** (if the emit path
doesn't guarantee it, that is a ``[BUILD]`` bug)". Here that guarantee is enforced: ``s``
is required and always included, with a named slice, and its absence raises rather than
silently dropping. ``s`` enters detached (stopgrad); the PW/SUCC feature comes from the
sibling deterministic Game-1 module :mod:`game` (also numpy).

Spec: `payload-spec.md` §2.2 (belief pipeline: V-cells, temporal, spatial, egocentric),
      §2.3 (query inputs ``[x_b ; beta_b]``); `rl-training-spec.md` §1 (``s`` guaranteed
      member; ``b`` computed), §2.1/§2.2 (``b`` is stopgrad), §4 (computed vs learned).

Public surface
--------------
- ``VCellMap``               : the segmented map -- centroids, areas, navigable graph,
                               graph-distance matrix, chosen bounded-support horizon.
- ``segment_vcells``         : stage 2 -- Voronoi + navigable fusion, horizon set to band.
- ``build_cell_slots``       : fold per-team observation rows -> per-cell ``f_c^obs`` +
                               last-observed times (incomplete-information buffer).
- ``temporal_contraction``   : stage 3 -- ``f^eff = rho*f^obs + (1-rho)*f^prior``.
- ``egocentric_integration`` : stage 5 -- ``beta_b = sum_c g * Phi * f_c^eff``.
- ``belief``                 : stages 3-5 end-to-end for one bot -> ``beta_b``.
- ``beliefs_for_bots``       : team-buffer precompute (``Phi*f^eff`` once) + per-bot betas.
- ``cartstate_vector``       : deterministic numeric encoding of cartstate ``s``.
- ``assemble_features``      : global feature vector; ``s`` a GUARANTEED member.
- ``GlobalFeatures``         : the assembled vector plus its named slices.
"""

from __future__ import annotations

import heapq
from collections import namedtuple
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .game import as_carts, succ_feature

# Default band on the receptive-field fraction of map area (payload-spec §2.2.2).
_BAND_LO = 0.05
_BAND_HI = 0.15


# --------------------------------------------------------------------------- #
# Stage 2: V-cell segmentation (Voronoi over nodes + navigable fusion).
# --------------------------------------------------------------------------- #

def _knn_adjacency(positions: np.ndarray, k: int) -> list:
    """Symmetric k-nearest-neighbour navigable adjacency (a stand-in for real nav edges).

    The real navigable graph is map data (waypoint links; engine ``[BUILD]``); when the
    caller does not supply ``adjacency`` this derives a deterministic symmetric kNN graph
    over node positions so the pipeline is exercisable on synthetic input. Preferring an
    explicit ``adjacency`` is the intended path (payload-spec §2.2.2 fuses *navigable*
    paths, which only the map knows).
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
      2. Build the navigable graph: ``adjacency`` if given (preferred -- real waypoint
         links), else a symmetric kNN stand-in (:func:`_knn_adjacency`).
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
    edges = [list() for _ in range(C)]
    seen_pair = set()
    for i in range(n):
        ci = node_cell[i]
        for j in adj[i]:
            cj = node_cell[j]
            if ci == cj:
                continue
            key = (min(ci, cj), max(ci, cj))
            if key in seen_pair:
                continue
            seen_pair.add(key)
            w = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            edges[ci].append((cj, w))
            edges[cj].append((ci, w))
    graph_dist = _dijkstra_all_pairs(C, edges)

    # --- step 5: set the horizon R so median receptive fraction lands in band. ---
    finite = graph_dist[np.isfinite(graph_dist)]
    candidates = np.unique(finite)
    support_radius = candidates[-1] if candidates.size else 0.0
    for R in candidates:
        within = graph_dist <= R + 1e-9
        # per-cell fraction of map area within support R
        fracs = (within * areas[None, :]).sum(axis=1) / map_area
        med = float(np.median(fracs))
        if med >= lo:
            support_radius = float(R)
            break
    else:
        support_radius = float(candidates[-1]) if candidates.size else 0.0

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

def _resolve_phi(Phi, F: int) -> np.ndarray:
    """Return the low-rank projection matrix ``Phi`` ``(rank, F)``; default identity passthrough."""
    if Phi is None:
        return np.eye(F, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    if Phi.ndim != 2 or Phi.shape[1] != F:
        raise ValueError(f"Phi must be (rank, {F})")
    return Phi


def egocentric_integration(vcmap: VCellMap, cell_idx: int, f_eff, Phi=None) -> np.ndarray:
    """**Stage 5** -- ``beta_b = sum_c g(dist_graph(c(b), c)) * Phi * f_c^eff``.

    payload-spec §2.2.5, the system's ONLY spatial mixing operator. The bounded-support
    graph-distance mask ``g`` (stage 4) is the egocentric weighting; ``Phi`` is the fixed
    low-rank projection. Computed as ``beta_b = (g @ (f_eff @ Phi.T))`` -- equivalently
    ``Phi @ (sum_c g_c f_c^eff)`` -- so the ``Phi * f_c^eff`` term precomputes once over
    the map (see :func:`beliefs_for_bots`) and only the ``g``-weighted sum is per-bot.

    Two bots in the same cell with the same ``f_eff`` get the same ``beta_b`` (identical
    inputs; there is no team-belief object). Returns ``beta_b`` ``(rank,)``.
    """
    f_eff = np.asarray(f_eff, dtype=np.float64)
    Phi = _resolve_phi(Phi, f_eff.shape[1])
    g = vcmap.spatial_mask(cell_idx)                 # (C,)
    projected = f_eff @ Phi.T                          # (C, rank)  = Phi * f_c^eff per cell
    return g @ projected                               # (rank,)


def belief(rows, vcmap: VCellMap, bot_position_or_cell, Phi=None, now: float = 0.0,
           T: float = 1.0, f_prior=None) -> np.ndarray:
    """End-to-end stages 3-5 for ONE bot: observation buffer -> ``beta_b``.

    Composes :func:`build_cell_slots` (ingest the per-team buffer),
    :func:`temporal_contraction` (stage 3), and :func:`egocentric_integration`
    (stages 4-5) for a single bot at ``bot_position_or_cell`` (a world position, resolved
    via ``vcmap.assign_cell``, or an int cell id). Returns the egocentric belief
    ``beta_b`` ``(rank,)`` (payload-spec §2.2). ``T`` is the stage-3 forgetting constant;
    ``f_prior`` the uninformative prior (default zeros).
    """
    f_obs, obs_time, seen = build_cell_slots(rows, vcmap, now)
    f_eff = temporal_contraction(f_obs, obs_time, now, T, f_prior=f_prior, seen=seen)
    cell = bot_position_or_cell if isinstance(bot_position_or_cell, (int, np.integer)) \
        else vcmap.assign_cell(bot_position_or_cell)
    return egocentric_integration(vcmap, int(cell), f_eff, Phi=Phi)


def beliefs_for_bots(rows, vcmap: VCellMap, bot_positions_or_cells, Phi=None,
                     now: float = 0.0, T: float = 1.0, f_prior=None) -> np.ndarray:
    """Team-buffer belief for many bots: precompute ``Phi * f_c^eff`` once, then per-bot.

    The scaling claim of payload-spec §2.2.5 made explicit: ingest + stage-3 contraction +
    the ``Phi * f_c^eff`` projection are computed ONCE over the ``C`` cells
    (``O(C*rank)``), then each bot is only a ``g``-weighted sum over its bounded-support
    horizon (``O(horizon)``) -- so cost scales with **map size, not player count**. All
    bots read the same per-team buffer, so two bots in the same cell get identical
    ``beta_b`` (no team-belief object; §2.2.5).

    ``bot_positions_or_cells`` : iterable of world positions and/or int cell ids.
    Returns ``betas`` ``(B, rank)`` in input order.
    """
    f_obs, obs_time, seen = build_cell_slots(rows, vcmap, now)
    f_eff = temporal_contraction(f_obs, obs_time, now, T, f_prior=f_prior, seen=seen)
    Phi = _resolve_phi(Phi, f_eff.shape[1])
    projected = f_eff @ Phi.T                          # (C, rank) -- precomputed once
    out = []
    for b in bot_positions_or_cells:
        cell = b if isinstance(b, (int, np.integer)) else vcmap.assign_cell(b)
        g = vcmap.spatial_mask(int(cell))
        out.append(g @ projected)
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, projected.shape[1]))


# --------------------------------------------------------------------------- #
# Cartstate s encoding + global feature assembly (s is a GUARANTEED member).
# --------------------------------------------------------------------------- #

def cartstate_vector(carts, teams: Optional[Iterable] = None) -> np.ndarray:
    """Deterministic numeric encoding of cartstate ``s`` (cart depths + control).

    ``s`` is the cartstate of rl-training-spec §1 -- "cart depths + control". This encodes
    each cart as ``[depth, control_id_or_-1]`` in a stable order, flattened, so ``s`` is a
    fixed, deterministic vector that :func:`assemble_features` can guarantee into the
    global feature vector. Control ids are mapped to their index in the sorted team roster
    (``-1`` for uncontrolled); depths are the integer depth-under-control. Uses
    :func:`game.as_carts` so the cart normalization matches the Game-1 module exactly.

    Returns a ``float32`` vector of length ``2 * len(carts)`` (``[d0, ctrl0, d1, ctrl1, ...]``).
    """
    cs = as_carts(carts)
    # deterministic control->index over the sorted roster (mirrors game._teams_of order)
    seen = {c.control for c in cs if c.control is not None}
    if teams is not None:
        seen |= set(teams)
    roster = sorted(seen, key=lambda t: (0, t) if isinstance(t, (int, float)) else (1, str(t)))
    index = {t: i for i, t in enumerate(roster)}
    out = np.empty(2 * len(cs), dtype=np.float32)
    for i, c in enumerate(cs):
        out[2 * i] = float(c.depth)
        out[2 * i + 1] = float(index[c.control]) if c.control is not None else -1.0
    return out


# The assembled global feature vector + its named slices (payload-spec §2.3 query input).
GlobalFeatures = namedtuple("GlobalFeatures", ("vector", "slices", "names"))


def assemble_features(x_b, beta_b, cartstate, teams: Optional[Iterable] = None,
                      instruments=None, succ=None, extra=None) -> GlobalFeatures:
    """Assemble the global feature vector; **cartstate ``s`` is a GUARANTEED member**.

    Builds the vector the learned query projects, ``q_b = W_q [x_b ; beta_b]``
    (payload-spec §2.3), extended with the computed Game-1 features so the estimator's
    intermediate carries them: the concatenation, in order, of

        [ x_b ; beta_b ; s ; succ_feature(s) ; instruments? ; extra? ]

    where ``x_b`` is the bot's OWN known engine state (§2.1), ``beta_b`` the egocentric
    belief (this module, stages 3-5), ``s`` the cartstate (:func:`cartstate_vector`), and
    ``succ_feature`` the deterministic PW/SUCC anticipatory feature from the sibling
    Game-1 module :mod:`game`. All are computed / ``stopgrad`` (rl-training-spec §2.1);
    only the downstream projections learn.

    **The guarantee.** rl-training-spec §1: ``s`` "is a **guaranteed member of the global
    feature vector** (if the emit path doesn't guarantee it, that is a ``[BUILD]`` bug)".
    Here it is enforced structurally: ``cartstate`` is REQUIRED, always occupies the named
    ``"s"`` slice, and a missing/empty cartstate raises ``ValueError`` rather than
    silently dropping ``s``. The returned ``slices`` maps each block name to its
    ``slice`` in ``vector`` so callers (and tests) can assert ``"s"`` membership.

    Parameters
    ----------
    x_b        : the bot's own engine state vector (``(d_x,)``); may be empty.
    beta_b     : the egocentric belief ``(rank,)`` from :func:`belief`.
    cartstate  : the cartstate ``s`` -- carts sequence, or a pre-encoded ``s`` vector.
                 REQUIRED and non-empty (the guarantee).
    teams      : optional explicit team roster (passed to the cartstate/succ encoders).
    instruments: optional ``(M, d_z)`` or flat per-instrument descriptor block to append.
    succ       : optional precomputed succession (else derived from ``cartstate`` carts).
    extra      : optional extra flat feature block to append.

    Returns a :class:`GlobalFeatures` ``(vector, slices, names)``.
    """
    # --- cartstate s: required, encoded, and guaranteed into the vector. ---
    if cartstate is None:
        raise ValueError("cartstate s is a GUARANTEED member of the feature vector "
                         "(rl-training-spec §1); it must be provided, not None")
    arr = np.asarray(cartstate)
    if arr.dtype != object and arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        s_vec = arr.astype(np.float32)          # already a pre-encoded s vector
        carts = None
    else:
        carts = list(cartstate)
        s_vec = cartstate_vector(carts, teams=teams)
    if s_vec.size == 0:
        raise ValueError("cartstate s encoded to an empty vector; s must be a nonempty "
                         "member of the global feature vector (rl-training-spec §1)")

    blocks = []
    names = []
    slices = {}
    cursor = 0

    def _add(name, block):
        nonlocal cursor
        v = np.asarray(block, dtype=np.float32).reshape(-1)
        blocks.append(v)
        slices[name] = slice(cursor, cursor + v.size)
        names.append(name)
        cursor += v.size

    _add("x_b", np.asarray(x_b, dtype=np.float32).reshape(-1) if x_b is not None else np.zeros(0, np.float32))
    _add("beta_b", np.asarray(beta_b, dtype=np.float32).reshape(-1) if beta_b is not None else np.zeros(0, np.float32))
    _add("s", s_vec)  # <-- the guaranteed member

    # deterministic PW/SUCC anticipatory feature (only when carts, not a raw s vector)
    if carts is not None:
        _add("succ", succ_feature(carts, teams=teams, succ=succ))

    if instruments is not None:
        _add("instruments", np.asarray(instruments, dtype=np.float32).reshape(-1))
    if extra is not None:
        _add("extra", np.asarray(extra, dtype=np.float32).reshape(-1))

    vector = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    # Enforce the guarantee: s must be present and nonempty in the assembled vector.
    if "s" not in slices or (slices["s"].stop - slices["s"].start) == 0:
        raise AssertionError("cartstate s missing from the global feature vector "
                             "(rl-training-spec §1 [BUILD] bug)")
    return GlobalFeatures(vector=vector, slices=slices, names=tuple(names))
