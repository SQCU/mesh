"""What the viewer EXPECTS in the live stream, and what is actually there.

The point of this module is the negative space.  design/AGENDA.md R19 records a
run whose model input was rank 4 because the per-player resource block was never
wired in, and whose per-player state was zero while every downstream number kept
rendering as if it had been measured.  So the viewer never defaults a missing
field to zero: it reports `absent`, `shape_only` or `all_zero` and says which
producer would have to emit it.

`status` values:
    present     the field is there and carries at least one nonzero value
    all_zero    the field is there, correct shape, and identically zero
    shape_only  the shape is advertised (model.shapes) but the values are not emitted
    absent      the key does not exist in the frame at all
"""

from __future__ import annotations

import numpy as np


def _dig(frame, path):
    node = frame
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


def _classify(value):
    if value is None:
        return "all_zero" if value is None else "present", 0.0
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return "all_zero", 0.0
    finite = np.isfinite(array)
    nonzero = float(np.mean((array != 0) & finite)) if array.size else 0.0
    return ("present" if nonzero > 0 else "all_zero"), nonzero


# path, human label, which producer owns it, why the viewer wants it
FIELDS = [
    ("PW",                "projected winner",            "strat_responder.winner()",        "behavior: the PW timeline"),
    ("SUCC",              "succession / denial budget",  "strat_responder.succession()",    "behavior: denial budget per team"),
    ("carts",             "cart depth / ctrl / speed",   "strat_responder cart rows",       "behavior: the cart subgame itself"),
    ("resources",         "per-team resource pools",     "strat_responder.team_resources()", "behavior: health/armor/ammo/weapons per team"),
    ("strategy_focus",    "cross-team focus matrix",     "strat_responder strategy_focus",  "behavior: who is hunting whom"),
    ("assignments",       "per-player assignment rows",  "strat_responder assignments",     "behavior: instrument + target + gain/lane"),
    ("belief",            "belief diagnostics",          "live_belief.beliefs()",           "observation map-reduce health"),
    ("instrument_counts", "instrument mix",              "instruments.build_instruments()", "behavior: the vocabulary actually offered"),
    ("update",            "online training metrics",     "online.OnlineLearner.update()",   "training: losses, importance, updates"),
    ("model.x",           "engine input x",              "estimator.state_from_runtime()",  "INTERNALS: what actually entered the matmul (R19)"),
    ("model.beta",        "per-bot belief beta",         "live_belief.beliefs()",           "INTERNALS: the belief half of the query"),
    ("model.z",           "instrument descriptors z",    "instruments batch.descriptors",   "INTERNALS: the DPP feature matrix"),
    ("model.hierarchy",   "per-team hierarchy rows",     "runtime.hierarchy_rows()",        "behavior: hierarchy / winner mask"),
    ("model.w",           "integrated weight state",     "instruments.weights_from_table()", "INTERNALS: the state the velocity integrates"),
    ("model.ir",          "final IR",                    "estimator GramSwiGLU encoder",    "INTERNALS: the j-space the probes read"),
    ("model.gram",        "Gram matrix",                 "gram.GramSwiGLU",                 "INTERNALS: the all-to-all coupling"),
    ("model.score",       "action logits",               "estimator strategy_forward",      "INTERNALS: the sampled distribution"),
    ("model.winner_value", "value head W",               "value.StrategyValue.winner",      "INTERNALS: the W linear probe output"),
    ("model.loser_value", "value head L",                "value.StrategyValue.loser",       "INTERNALS: the L linear probe output"),
    ("model.diag_k",      "DPP diag(K)",                 "dpp.dpp_marginals()",             "INTERNALS: marginal inclusion, the diversity signal"),
    ("model.appetite",    "appetite (IR @ keys)",        "estimator strategy_forward",      "INTERNALS: pre-DPP per-instrument affinity"),
    ("model.dw_dt",       "weight velocity dw/dt",       "head.MixingHead",                 "INTERNALS: the emitted velocity, not the decision"),
    ("model.relation",    "player x instrument relations", "instruments batch.relations",   "INTERNALS: the O(n^2) relation rows"),
    ("update.advantage",  "advantage",                   "online.OnlineLearner.update()",   "INTERNALS: the quantity optimization increases (SPEC 5)"),
    ("game_value",        "CGT game value / nimber",     "game_value.py",                   "behavior: does the combinatorial value RESOLVE at all (B11)"),
]

# x column blocks, named, so a zeroed per-player resource block is visible at a
# glance.  Layout quoted from estimator.state_from_runtime.
X_BLOCKS = [
    ("cart-game scalars", 0, 8, False),
    ("health", 8, 9, True),
    ("armor", 9, 10, True),
    ("ammo", 10, 11, True),
    ("position", 11, 14, True),
    ("velocity", 14, 17, True),
    ("powerup", 17, 18, True),
    ("time since spawn", 18, 19, True),
    ("alive", 19, 20, True),
    ("bot/human control", 20, 21, True),
    ("nearest cart", 21, 22, True),
    ("nearest cart dist", 22, 23, True),
    ("V-cell", 23, 24, True),
    ("weapon bitset (24 bits)", 24, 48, True),
]


def audit(frame, model_frame=None):
    """Field-presence report.

    `frame` is the newest telemetry line; `model_frame` is the newest line that
    actually carries the sampled model arrays (the responder emits them every
    --model-sample-every ticks).  The `model.*` paths are read from the latter so
    a sampling gap is never reported as a missing field, and everything else is
    read from the former so the behavior fields describe THIS tick.
    """
    if not frame:
        return {"available": False, "fields": [], "x_blocks": [], "summary": {}}
    model_frame = model_frame or frame
    shapes = (model_frame.get("model") or {}).get("shapes") or {}
    rows = []
    for path, label, owner, why in FIELDS:
        source = model_frame if path.startswith("model.") else frame
        value, found = _dig(source, path)
        if not found:
            leaf = path.split(".")[-1]
            status = "shape_only" if path.startswith("model.") and leaf in shapes else "absent"
            nonzero = 0.0
        elif value is None:
            status, nonzero = "absent", 0.0
        elif isinstance(value, (dict, str, bool)):
            status, nonzero = "present", 1.0
        elif isinstance(value, (int, float)):
            # A scalar that is legitimately 0 (PW=0 means "no projected winner")
            # is PRESENT.  Its value is carried in the shape column instead.
            status, nonzero = "present", 1.0 if value else 0.0
        else:
            try:
                status, nonzero = _classify(value)
            except Exception:
                status, nonzero = "present", 1.0
        rows.append({
            "path": path, "label": label, "owner": owner, "why": why,
            "status": status, "nonzero_fraction": round(nonzero, 4),
            "shape": (
                [value] if isinstance(value, (int, float)) and not isinstance(value, bool)
                else list(np.shape(value)) if status in ("present", "all_zero") and not isinstance(value, (dict, str, bool))
                else shapes.get(path.split(".")[-1])
            ),
        })
    blocks = []
    x = (model_frame.get("model") or {}).get("x")
    if x is not None:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim == 2:
            for label, lo, hi, per_player in X_BLOCKS:
                if hi > matrix.shape[1]:
                    blocks.append({"label": label, "cols": [lo, hi], "status": "absent",
                                   "per_player": per_player, "nonzero_cols": 0, "width": hi - lo})
                    continue
                chunk = matrix[:, lo:hi]
                nonzero_cols = int((np.abs(chunk).sum(axis=0) > 0).sum())
                blocks.append({
                    "label": label, "cols": [lo, hi], "per_player": per_player,
                    "width": hi - lo, "nonzero_cols": nonzero_cols,
                    "status": "present" if nonzero_cols else "all_zero",
                })
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "available": True,
        "fields": rows,
        "x_blocks": blocks,
        "summary": counts,
    }


__all__ = ["audit", "FIELDS", "X_BLOCKS"]
