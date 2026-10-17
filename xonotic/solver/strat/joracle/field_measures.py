from __future__ import annotations

import numpy as np


COORDINATES = (
    ("assignments", "per-player state applications", "action_history.assignments"),
    ("learning", "optimizer updates", "online.OnlineLearner.learn"),
    ("model.x", "full engine state", "state_steering.unpack_pages"),
    ("model.local_neighborhood", "per-player local neighborhood message", "neighborhood.LocalNeighborhood"),
    ("model.source_features", "literal policy inputs and structural masks", "action_history.source_features"),
    ("model.hierarchy", "game hierarchy", "runtime.build_runtime_frame"),
    ("model.j", "shared action and value representation", "strategy.strategy"),
    ("model.rate", "full state-update rate", "strategy.read_heads / state_steering.rate_distribution"),
    ("model.residual", "integrated residual", "state_steering.integrate"),
    ("policy_comparison", "full-vector policy divergence", "policy_reports.compare_policies"),
    ("moe", "expert routing and balancing", "cast_header.scale_operator"),
    ("game_value", "CGT game value", "game_value.formal_game_value"),
)

def _dig(frame, path):
    node = frame
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, 0
        node = node[part]
    return node, 1

def _measure(value, key_mass, advertised_shape=None):
    base = {
        "key_mass": key_mass,
        "value_mass": int(key_mass and value is not None),
        "coordinate_mass": 0,
        "finite_mass": 0,
        "nonzero_mass": 0,
        "integral": None,
        "mean": None,
        "variance": None,
        "shape": advertised_shape,
    }
    if not base["value_mass"]:
        return base
    if isinstance(value, dict):
        base["coordinate_mass"] = len(value)
        base["shape"] = [len(value)]
        return base
    if isinstance(value, str):
        base["coordinate_mass"] = len(value)
        base["shape"] = [len(value)]
        return base
    if isinstance(value, (list, tuple)) and any(isinstance(item, (dict, str)) for item in value):
        base["coordinate_mass"] = len(value)
        base["shape"] = [len(value)]
        return base
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        base["coordinate_mass"] = 1
        base["shape"] = list(np.shape(value))
        return base
    flat = array.reshape(-1)
    finite = flat[np.isfinite(flat)]
    base.update(
        coordinate_mass=int(flat.size),
        finite_mass=int(finite.size),
        nonzero_mass=int(np.count_nonzero(finite)),
        integral=float(finite.sum()) if finite.size else None,
        mean=float(finite.mean()) if finite.size else None,
        variance=float(finite.var()) if finite.size else None,
        shape=list(array.shape),
    )
    return base

def field_measures(frame, model_frame=None):
    if not frame:
        return {"frame_mass": 0, "fields": [], "x_blocks": [], "totals": {}}
    model_frame = model_frame or frame
    shapes = (model_frame.get("model") or {}).get("shapes") or {}
    fields = []
    for path, label, owner in COORDINATES:
        source = model_frame if path.startswith("model.") else frame
        value, key_mass = _dig(source, path)
        measure = _measure(value, key_mass, shapes.get(path.split(".")[-1]))
        fields.append({"path": path, "label": label, "owner": owner, **measure})
    blocks = []
    x = (model_frame.get("model") or {}).get("x")
    matrix = np.asarray(x, dtype=np.float64) if x is not None else np.empty((0, 0))
    width = matrix.shape[1] if matrix.ndim == 2 else 0
    for label, lo, hi in (("full state vector", 0, width),):
        observed_hi = min(hi, matrix.shape[1]) if matrix.ndim == 2 else lo
        chunk = matrix[:, lo:observed_hi] if matrix.ndim == 2 and observed_hi > lo else np.empty((0, 0))
        finite = np.isfinite(chunk)
        blocks.append({
            "label": label,
            "cols": [lo, hi],
            "declared_width": hi - lo,
            "observed_width": max(0, observed_hi - lo),
            "row_mass": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
            "coordinate_mass": int(chunk.size),
            "finite_mass": int(finite.sum()),
            "nonzero_mass": int(((chunk != 0) & finite).sum()),
            "nonzero_columns": int(np.any((chunk != 0) & finite, axis=0).sum()) if chunk.size else 0,
        })
    totals = {
        name: sum(int(field[name]) for field in fields)
        for name in ("key_mass", "value_mass", "coordinate_mass", "finite_mass", "nonzero_mass")
    }
    return {"frame_mass": 1, "fields": fields, "x_blocks": blocks, "totals": totals}

__all__ = ["COORDINATES", "field_measures"]
