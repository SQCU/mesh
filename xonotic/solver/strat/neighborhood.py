from __future__ import annotations

from . import tensor as mx
import mlx.nn as nn
import numpy as np

from .featurize import receptive_report, vcell_from_navigation
from .matmul import linear
from .policy_math import norm
from .paged_matrix import neighborhood_integral


class NavigationRows:
    def __init__(self, realization):
        payload = realization.get("navigation_realization", realization)
        positions = np.asarray(payload.get("nodes", ()), dtype=np.float32).reshape(-1, 3)
        if len(positions) == 0:
            self.map = None
            self.nodes = np.zeros((0, 4), dtype=np.float32)
            self.edges = np.zeros((0, 3), dtype=np.float32)
            self.cells = np.zeros((0, 2), dtype=np.float32)
            self.source_cells = np.zeros((0,), dtype=np.int64)
            self.report = {"spatial_geometry": "unavailable", "nodes": 0, "edges": 0, "cells": 0,
                "in_band": False, "measure_dimension": payload.get("measure_dimension")}
            return
        self.map = vcell_from_navigation(payload)
        measure = np.asarray(payload.get("node_measure", np.ones(len(positions))), dtype=np.float32)
        self.nodes = np.concatenate((positions, measure[:, None]), axis=1)
        self.edges = np.asarray(payload.get("edges", ()), dtype=np.float32).reshape(-1, 3)
        voronoi = payload.get("voronoi", {})
        sites = np.asarray(voronoi.get("site_nodes", [np.flatnonzero(self.map.node_cell == cell)[0]
            for cell in range(self.map.n_cells)]), dtype=np.int64)
        measures = np.asarray(voronoi.get("cell_measure", self.map.areas), dtype=np.float32)
        self.cells = np.stack((sites, measures), axis=1).astype(np.float32)
        self.source_cells = np.concatenate((self.map.node_cell,
            self.map.node_cell[self.edges[:, 0].astype(np.int64)], self.map.node_cell[sites]))
        self.edge_cells = self.map.node_cell[self.edges[:, 1].astype(np.int64)]
        self.report = receptive_report(self.map)
        self.report.update({"measure_dimension": int(payload.get("measure_dimension", 1)),
            "nodes": len(self.nodes), "edges": len(self.edges), "cells": len(self.cells),
            "realization_id": payload.get("realization_id")})

    def prepare(self, observation_positions, event_positions, event_visibility=None):
        if self.map is None:
            shape = (len(observation_positions), len(event_positions))
            return {"navigation_nodes": self.nodes, "navigation_edges": self.edges, "navigation_cells": self.cells,
                "neighborhood_indices": np.broadcast_to(np.arange(shape[1], dtype=np.int32), shape),
                "neighborhood_distances": np.zeros(shape, dtype=np.float32),
                "neighborhood_present": np.ones(shape, dtype=bool) if event_visibility is None else np.asarray(event_visibility, dtype=bool),
                "neighborhood_radius": np.ones(shape[0], dtype=np.float32)}
        destinations = np.asarray([self.map.assign_cell(position) for position in observation_positions], dtype=np.int64)
        event_cells = np.asarray([self.map.assign_cell(position) for position in event_positions], dtype=np.int64)
        source_cells = np.concatenate((event_cells, self.source_cells))
        distances = self.map.graph_dist[destinations[:, None], source_cells[None, :]].copy()
        start = len(event_cells) + len(self.nodes)
        distances[:, start:start + len(self.edges)] = np.minimum(distances[:, start:start + len(self.edges)],
            self.map.graph_dist[destinations[:, None], self.edge_cells[None, :]])
        radius = self.map.support_radii[destinations]
        present = np.isfinite(distances) & (distances <= radius[:, None] + 1e-9)
        if event_visibility is not None:
            present[:, :len(event_cells)] &= np.asarray(event_visibility, dtype=bool)
        capacity = 1 << (max(1, int(present.sum(axis=1).max(initial=0))) - 1).bit_length()
        indices = np.zeros((len(destinations), capacity), dtype=np.int32)
        distance = np.zeros((len(destinations), capacity), dtype=np.float32)
        active = np.zeros((len(destinations), capacity), dtype=bool)
        for row, neighbors in enumerate(present):
            selected = np.flatnonzero(neighbors)
            indices[row, :len(selected)] = selected
            distance[row, :len(selected)] = distances[row, selected]
            active[row, :len(selected)] = True
        return {"navigation_nodes": self.nodes, "navigation_edges": self.edges, "navigation_cells": self.cells,
            "neighborhood_indices": indices, "neighborhood_distances": distance,
            "neighborhood_present": active, "neighborhood_radius": np.asarray(radius, dtype=np.float32)}


class LocalNeighborhood(nn.Module):
    def __init__(self, width, gram=True):
        super().__init__()
        self.node_projection = nn.Linear(4, width, bias=False)
        self.edge_projection = nn.Linear(3, width, bias=False)
        self.cell_projection = nn.Linear(2, width, bias=False)
        if gram:
            self.metric = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.log_decay = mx.array(0.)

    def encode(self, nodes, edges, cells):
        return mx.concatenate((linear(self.node_projection, nodes), linear(self.edge_projection, edges),
            linear(self.cell_projection, cells)), axis=0)

    def temporal_weights(self, ages):
        return mx.exp(-mx.logaddexp(mx.zeros_like(self.log_decay), self.log_decay) * mx.maximum(ages, 0))

    def __call__(self, destinations, sources, indices, distances, present, radius,
                 source_times=None, observer_times=None, source_temporal=None, *, gram=True):
        source_times = mx.zeros((sources.shape[0],)) if source_times is None else source_times
        observer_times = mx.zeros((destinations.shape[0],)) if observer_times is None else observer_times
        source_temporal = mx.zeros((sources.shape[0],), dtype=mx.bool_) if source_temporal is None else source_temporal
        sources = mx.concatenate((sources, mx.zeros((1, sources.shape[-1]), dtype=sources.dtype)))
        source_times = mx.concatenate((source_times, mx.zeros((1,), dtype=source_times.dtype)))
        source_temporal = mx.concatenate((source_temporal, mx.zeros((1,), dtype=mx.bool_)))
        values = linear(self.value, norm(sources) if gram else sources)
        radius = mx.broadcast_to(radius, (destinations.shape[0],))
        kernel = mx.exp(-4 * mx.square(distances / mx.maximum(radius[:, None], 1e-20)))
        ages = observer_times[:, None] - source_times[indices]
        temporal = mx.where(source_temporal[indices], self.temporal_weights(ages), 1)
        weights = mx.where(present, kernel * temporal, 0)
        keys = linear(self.metric, norm(sources)) if gram else sources
        query = linear(self.metric, norm(destinations)) if gram else destinations
        integrated = neighborhood_integral(query, keys, values, indices, weights, gram)
        return linear(self.output, integrated)
