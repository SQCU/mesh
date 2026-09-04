from __future__ import annotations

import numpy as np

from payload.tools.strategy_io_schema import XAN_BLOCKS

COORDINATES = (
    ("PW", "projected winner", "strat_responder.winner()"),
    ("SUCC", "succession / denial budget", "strat_responder.succession()"),
    ("carts", "cart depth / control / speed", "strat_responder cart rows"),
    ("resources", "per-team resource pools", "strat_responder.team_resources()"),
    ("strategy_focus", "cross-team focus matrix", "strat_responder.strategy_focus"),
    ("assignments", "per-player assignment rows", "strat_responder assignments"),
    ("belief", "belief coordinates", "live_belief.chorus()"),
    ("instrument_counts", "instrument counts", "instruments.build_instruments()"),
    ("update", "online update coordinates", "online.OnlineLearner.update()"),
    ("model.x", "engine input x", "inputs.player_features()"),
    ("model.beta", "per-player belief beta", "strategy.strategy()"),
    ("model.z", "instrument descriptors z", "instrument batch descriptors"),
    ("model.hierarchy", "per-team hierarchy rows", "runtime.hierarchy_rows()"),
    ("model.w", "integrated weight state", "instruments.weights_from_table()"),
    ("model.j", "selected J", "strategy.strategy()"),
    ("model.pooled", "pooled value row", "strategy.strategy()"),
    ("model.coupling", "coupling matrix", "strategy.strategy()"),
    ("model.score", "action logits", "strategy.strategy()"),
    ("model.winner_value", "value head W", "cast_header.winnie"),
    ("model.loser_value", "value head L", "cast_header.lou"),
    ("model.diag_k", "DPP diag(K)", "dpp.dpp_marginals()"),
    ("model.action_mass", "observed action-support measure", "instruments.build_instruments()"),
    ("model.dw_dt", "weight velocity", "cast_header.gia_uma_dov"),
    ("update.advantage", "advantage", "online.OnlineLearner.update()"),
    ("game_value", "CGT game value", "game_value.py"),
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
    for label, lo, hi in XAN_BLOCKS:
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
