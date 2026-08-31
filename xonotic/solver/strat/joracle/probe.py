"""Rolling linear probes on the LIVE IR — the j-oracle proper.

Same methodology as design/jspace-probe.md, made continuous:

  * ridge least squares, lambda = 1e-3, features standardized on the train split
    with a bias column appended (identical to runs/jspace_probe.py:111-118);
  * one-vs-rest ridge for the categorical targets, accuracy reported against the
    test-split majority baseline (:120-129);
  * split BY TICK, 60/40, fixed seed, so no test row shares a server tick with a
    training row;
  * two controls, always shown next to the score, because a probe score without
    its controls is not evidence:
       (a) RANDOM PROJECTION of the same raw input x at the same width as the IR
           -- if the IR does not beat this, the IR learned nothing the input did
           not already hand it;
       (b) SHUFFLED LABELS on the IR itself -- if this is not ~0 / ~majority, the
           probe is fitting noise and every other number on the page is void.
    A third column, the raw input x itself, marks the targets that are
    tautological (the target IS an input column).

The one control design/jspace-probe.md ran that this cannot run is the
random-INIT encoder, because that needs a second forward pass through the model
and this process never touches the model.  It is reported as not-measurable
rather than silently omitted.

Rank panel: effective rank of the IR over the rolling window, of the raw input x,
and of beta.  R19's headline pathology was `rank(raw input) == 4`; a rank readout
next to the IR width is what makes that visible at a glance if it recurs.
"""

from __future__ import annotations

import threading
import time

import numpy as np

LAMBDA = 1e-3
SPLIT_SEED = 0
TRAIN_FRACTION = 0.6

# name, tautological?, note
REGRESSION_TARGETS = [
    ("health",            True,  "x[:,8] -- the SPEC-3 headline: did health enter the matmul at all"),
    ("armor",             True,  "x[:,9]"),
    ("ammo",              True,  "x[:,10]"),
    ("speed",             True,  "|x[:,14:17]|"),
    ("n_weapons",         True,  "popcount of the x[:,24:48] weapon bitset"),
    ("dist_nearest_cart", True,  "x[:,22]"),
    ("own_nimber",        True,  "x[:,2] -- tautological, kept because jspace-probe.md reports it"),
    ("max_rival_nimber",  True,  "x[:,3]"),
    ("total_cart_depth",  False, "sum of cart depths; tick-level"),
    ("n_controlled",      False, "carts controlled by this player's team"),
    ("succ_denial",       False, "this team's succession denial budget"),
    ("gain",              False, "the emitted instrument gain -- a policy OUTPUT"),
    ("logp",              False, "log pi of the chosen action"),
    ("winner_value",      False, "the W head's own output; a probe of the IR against its own readout"),
    ("loser_value",       False, "the L head's own output"),
    ("advantage",         False, "requires update.advantage in the stream"),
]

CLASSIFICATION_TARGETS = [
    ("is_pw",      False, "is this player's team the projected winner"),
    ("pw_team",    False, "which team is the projected winner"),
    ("instr_kind", False, "the instrument kind actually chosen (7-way)"),
    ("team",       False, "this player's team id"),
    ("controller", True,  "bot vs human; x[:,20]"),
]


def _finite(array):
    return np.asarray(array, dtype=np.float64)


def rows_from_frame(frame):
    """Per-player feature/target rows for one telemetry frame, or None.

    Every field is read defensively.  A target that cannot be built from this
    frame comes back as NaN for the whole frame and is dropped per-target at fit
    time, never imputed.
    """
    model = frame.get("model") or {}
    ir = model.get("ir")
    x = model.get("x")
    if ir is None or x is None:
        return None
    ir = _finite(ir)
    x = _finite(x)
    if ir.ndim != 2 or x.ndim != 2 or ir.shape[0] != x.shape[0] or ir.shape[0] == 0:
        return None
    n = ir.shape[0]
    nan = np.full(n, np.nan)

    assignments = frame.get("assignments") or []
    assignments = sorted(assignments, key=lambda a: a.get("row", 0)) if len(assignments) == n else []

    def column(index, scale=1.0):
        return x[:, index] * scale if x.shape[1] > index else nan.copy()

    def from_assignments(key, cast=float):
        if not assignments:
            return nan.copy()
        out = np.empty(n)
        for i, a in enumerate(assignments):
            try:
                out[i] = cast(a.get(key))
            except Exception:
                out[i] = np.nan
        return out

    teams = from_assignments("team")
    pw = frame.get("PW")
    carts = frame.get("carts") or []
    depths = [c.get("depth", 0.0) for c in carts]
    ctrl = [c.get("ctrl", 0) for c in carts]
    succ = {int(a): float(b) for a, b in (frame.get("SUCC") or []) if a is not None}

    n_controlled = np.array([
        sum(1 for c in ctrl if c == t) if np.isfinite(t) else np.nan for t in teams
    ], dtype=np.float64)
    succ_denial = np.array([
        succ.get(int(t), np.nan) if np.isfinite(t) else np.nan for t in teams
    ], dtype=np.float64)

    weapons = x[:, 24:48].sum(axis=1) if x.shape[1] >= 48 else nan.copy()
    velocity = np.linalg.norm(x[:, 14:17], axis=1) if x.shape[1] >= 17 else nan.copy()

    update = frame.get("update") or {}
    advantage = update.get("advantage")
    advantage_row = np.full(n, float(advantage)) if isinstance(advantage, (int, float)) else nan.copy()

    def head(name):
        value = model.get(name)
        if value is None:
            return nan.copy()
        value = _finite(value).reshape(-1)
        return value if value.shape[0] == n else nan.copy()

    kinds = [a.get("kind", "?") for a in assignments] if assignments else ["?"] * n
    controller = [a.get("controller", "?") for a in assignments] if assignments else ["?"] * n

    regression = {
        "health": column(8),
        "armor": column(9),
        "ammo": column(10),
        "speed": velocity,
        "n_weapons": weapons,
        "dist_nearest_cart": column(22),
        "own_nimber": column(2),
        "max_rival_nimber": column(3),
        "total_cart_depth": np.full(n, float(sum(depths))) if depths else nan.copy(),
        "n_controlled": n_controlled,
        "succ_denial": succ_denial,
        "gain": from_assignments("gain"),
        "logp": from_assignments("target_logp"),
        "winner_value": head("winner_value"),
        "loser_value": head("loser_value"),
        "advantage": advantage_row,
    }
    classification = {
        "is_pw": np.array([1 if (pw is not None and np.isfinite(t) and int(t) == int(pw)) else 0 for t in teams]),
        "pw_team": np.full(n, int(pw) if isinstance(pw, int) else -1),
        "instr_kind": np.array(kinds, dtype=object),
        "team": np.array([int(t) if np.isfinite(t) else -1 for t in teams]),
        "controller": np.array(controller, dtype=object),
    }
    return {
        "ir": ir,
        "x": x,
        "beta": _finite(model.get("beta")) if model.get("beta") is not None else None,
        "regression": regression,
        "classification": classification,
        "tick": int(frame.get("resp_id") or 0),
        "epoch": int(frame.get("_epoch") or 0),
    }


def _effective_rank(matrix, tol=1e-6):
    """(hard rank, participation-ratio effective rank) of a centered matrix."""
    if matrix is None or matrix.size == 0 or matrix.ndim != 2:
        return None, None
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, None
    hard = int(np.linalg.matrix_rank(centered, tol=tol))
    total = values.sum()
    if total <= 0:
        return hard, 0.0
    p = values / total
    p = p[p > 0]
    effective = float(np.exp(-(p * np.log(p)).sum()))
    return hard, round(effective, 3)


def _ridge_r2(features, y, train, test):
    ok = np.isfinite(y)
    a, b = train & ok, test & ok
    if a.sum() < 10 or b.sum() < 10 or np.std(y[b]) < 1e-9:
        return None
    A = np.c_[features[a], np.ones(a.sum())]
    B = np.c_[features[b], np.ones(b.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0) + 1e-8
    A[:, :-1] = (A[:, :-1] - mu) / sd
    B[:, :-1] = (B[:, :-1] - mu) / sd
    try:
        w = np.linalg.solve(A.T @ A + LAMBDA * np.eye(A.shape[1]), A.T @ y[a])
    except np.linalg.LinAlgError:
        return None
    residual = np.sum((y[b] - B @ w) ** 2)
    total = np.sum((y[b] - y[b].mean()) ** 2)
    return round(float(1.0 - residual / total), 4)


def _ridge_accuracy(features, y, train, test):
    classes = np.unique(y[train])
    if len(classes) < 2 or train.sum() < 10 or test.sum() < 10:
        return None
    A = np.c_[features[train], np.ones(train.sum())]
    B = np.c_[features[test], np.ones(test.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0) + 1e-8
    A[:, :-1] = (A[:, :-1] - mu) / sd
    B[:, :-1] = (B[:, :-1] - mu) / sd
    Y = (y[train][:, None] == classes[None, :]).astype(float)
    try:
        W = np.linalg.solve(A.T @ A + LAMBDA * np.eye(A.shape[1]), A.T @ Y)
    except np.linalg.LinAlgError:
        return None
    predicted = classes[np.argmax(B @ W, axis=1)]
    counts = np.array([(y[test] == v).sum() for v in np.unique(y[test])], dtype=float)
    return {
        "acc": round(float(np.mean(predicted == y[test])), 4),
        "majority": round(float(counts.max() / counts.sum()), 4),
        "n_classes": int(len(classes)),
    }


class RollingProbe:
    """Accumulates live player-rows and recomputes the probe table in a thread."""

    def __init__(self, *, max_rows=4000, min_rows=120, min_ticks=8, interval=4.0, seed=SPLIT_SEED,
                 rows_per_feature=16.0):
        self.max_rows = int(max_rows)
        self.min_rows = int(min_rows)
        self.min_ticks = int(min_ticks)
        # A ridge probe on d features fitted to fewer than a few d rows is
        # underdetermined: it interpolates the train split and explodes on the
        # test split, which shows up as a hugely NEGATIVE R^2 on the shuffled
        # control.  jspace_probe.py had 1278 test rows against 16-40 features.
        # The live probe therefore waits until the window is wide enough rather
        # than publishing a degenerate fit.
        self.rows_per_feature = float(rows_per_feature)
        self.interval = float(interval)
        self.seed = int(seed)
        self.lock = threading.Lock()
        self.buffer = []                # list of rows_from_frame results
        self.report = {"available": False, "reason": "no rows yet"}
        self.computed_at = None
        self.compute_ms = None
        self.errors = 0
        self.last_error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="joracle-probe", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def ingest(self, frame):
        row = rows_from_frame(frame)
        if row is None:
            return
        with self.lock:
            self.buffer.append(row)
            total = sum(item["ir"].shape[0] for item in self.buffer)
            while self.buffer and total > self.max_rows:
                total -= self.buffer[0]["ir"].shape[0]
                self.buffer.pop(0)

    def _loop(self):
        while not self._stop.is_set():
            start = time.time()
            try:
                report = self.compute()
                with self.lock:
                    self.report = report
                    self.computed_at = time.time()
                    self.compute_ms = round((time.time() - start) * 1000, 1)
            except Exception as exc:
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            deadline = time.time() + self.interval
            while time.time() < deadline and not self._stop.is_set():
                time.sleep(0.1)

    # -- the measurement -----------------------------------------------------
    def compute(self):
        with self.lock:
            buffer = list(self.buffer)
        if not buffer:
            return {"available": False, "reason": "no frames carrying model.ir and model.x yet"}

        widths_ir = {item["ir"].shape[1] for item in buffer}
        widths_x = {item["x"].shape[1] for item in buffer}
        if len(widths_ir) > 1 or len(widths_x) > 1:
            # the responder restarted at a different shape; keep only the newest
            newest = (buffer[-1]["ir"].shape[1], buffer[-1]["x"].shape[1])
            buffer = [i for i in buffer if (i["ir"].shape[1], i["x"].shape[1]) == newest]

        ir = np.concatenate([item["ir"] for item in buffer], axis=0)
        x = np.concatenate([item["x"] for item in buffer], axis=0)
        betas = [item["beta"] for item in buffer if item["beta"] is not None and item["beta"].ndim == 2]
        beta = np.concatenate(betas, axis=0) if betas and len({b.shape[1] for b in betas}) == 1 else None
        ticks = np.concatenate([
            np.full(item["ir"].shape[0], item["tick"] + 1_000_000 * item["epoch"]) for item in buffer
        ])
        n_rows = ir.shape[0]
        unique_ticks = np.unique(ticks)

        ir_rank, ir_effective = _effective_rank(ir)
        x_rank, x_effective = _effective_rank(x)
        beta_rank, beta_effective = _effective_rank(beta) if beta is not None else (None, None)
        x_nonzero_cols = int((np.abs(x).sum(axis=0) > 0).sum())

        required_rows = max(self.min_rows, int(self.rows_per_feature * ir.shape[1]))
        geometry = {
            "rows": int(n_rows),
            "required_rows": required_rows,
            "rows_per_feature": round(n_rows / max(1, ir.shape[1]), 2),
            "ticks": int(len(unique_ticks)),
            "ir_width": int(ir.shape[1]),
            "ir_rank": ir_rank,
            "ir_effective_rank": ir_effective,
            "x_width": int(x.shape[1]),
            "x_rank": x_rank,
            "x_effective_rank": x_effective,
            "x_nonzero_columns": x_nonzero_cols,
            "beta_width": None if beta is None else int(beta.shape[1]),
            "beta_rank": beta_rank,
            "spec_ir_width_floor": 128,
        }

        if n_rows < required_rows or len(unique_ticks) < self.min_ticks:
            return {
                "available": False,
                "reason": (f"accumulating: {n_rows}/{required_rows} player-rows "
                           f"({self.rows_per_feature:g} per IR dim), {len(unique_ticks)}/{self.min_ticks} ticks"),
                "geometry": geometry,
                "pathology": self._pathology(geometry, None),
            }

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(unique_ticks)
        train_ticks = set(permutation[: max(1, int(TRAIN_FRACTION * len(permutation)))].tolist())
        train = np.array([t in train_ticks for t in ticks])
        test = ~train
        if train.sum() < 10 or test.sum() < 10:
            return {"available": False, "reason": "split degenerate", "geometry": geometry,
                    "pathology": self._pathology(geometry, None)}

        # control (a): random Gaussian projection of the raw input, IR width
        projector = np.random.default_rng(self.seed + 1).standard_normal((x.shape[1], ir.shape[1])) / np.sqrt(x.shape[1])
        randproj = x @ projector
        shuffle_rng = np.random.default_rng(self.seed + 2)

        regression = []
        for name, tautological, note in REGRESSION_TARGETS:
            y = np.concatenate([
                np.asarray(item["regression"].get(name), dtype=np.float64) for item in buffer
            ])
            shuffled = y.copy()
            shuffle_rng.shuffle(shuffled)
            row = {
                "target": name, "kind": "r2", "tautological": tautological, "note": note,
                "ir": _ridge_r2(ir, y, train, test),
                "randproj": _ridge_r2(randproj, y, train, test),
                "shuffled": _ridge_r2(ir, shuffled, train, test),
                "raw_x": _ridge_r2(x, y, train, test),
                "n_finite": int(np.isfinite(y).sum()),
            }
            row["delta_vs_randproj"] = (
                round(row["ir"] - row["randproj"], 4)
                if row["ir"] is not None and row["randproj"] is not None else None
            )
            # Per-target honesty.  A shuffled-label R^2 must land NEAR ZERO; a
            # large value in either direction means the ridge fit is degenerate
            # for THIS target (tick-level targets are the usual cause: all rows
            # in a tick share the value, so the effective sample is the tick
            # count, not the row count) and the target's other columns say
            # nothing.
            row["control_ok"] = row["shuffled"] is not None and abs(row["shuffled"]) <= 0.10
            regression.append(row)

        classification = []
        for name, tautological, note in CLASSIFICATION_TARGETS:
            y = np.concatenate([
                np.asarray(item["classification"].get(name), dtype=object) for item in buffer
            ])
            shuffled = y.copy()
            shuffle_rng.shuffle(shuffled)
            row = {
                "target": name, "kind": "acc", "tautological": tautological, "note": note,
                "ir": _ridge_accuracy(ir, y, train, test),
                "randproj": _ridge_accuracy(randproj, y, train, test),
                "shuffled": _ridge_accuracy(ir, shuffled, train, test),
                "raw_x": _ridge_accuracy(x, y, train, test),
            }
            if row["ir"] and row["randproj"]:
                row["delta_vs_randproj"] = round(row["ir"]["acc"] - row["randproj"]["acc"], 4)
            else:
                row["delta_vs_randproj"] = None
            row["control_ok"] = bool(
                row["shuffled"] and row["shuffled"]["acc"] <= row["shuffled"]["majority"] + 0.10
            )
            classification.append(row)

        verdict = self._verdict(regression, classification)
        return {
            "available": True,
            "method": {
                "estimator": "ridge least squares, lambda=1e-3, standardized + bias",
                "split": f"by tick, {int(TRAIN_FRACTION*100)}/{100-int(TRAIN_FRACTION*100)}, seed {self.seed}",
                "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                "controls": [
                    "random Gaussian projection of raw x at IR width (control a)",
                    "shuffled labels on the IR (control b)",
                    "raw x itself (tautology marker)",
                ],
                "control_not_available": (
                    "random-INIT encoder: needs a second forward pass through the model; "
                    "this process never touches the model, so it is NOT measured here. "
                    "See design/jspace-probe.md for the offline run that does."
                ),
                "simulation_used": False,
            },
            "geometry": geometry,
            "regression": regression,
            "classification": classification,
            "verdict": verdict,
            "pathology": self._pathology(geometry, verdict),
        }

    # -- readings ------------------------------------------------------------
    @staticmethod
    def _verdict(regression, classification):
        """Does the IR beat the random-projection control anywhere non-tautological?

        Only targets whose OWN shuffled-label control landed at chance are
        counted; a target the ridge fit degenerately is reported as degenerate
        rather than being read either way.
        """
        beats, considered, degenerate = [], 0, []
        scored = 0
        for row in list(regression) + list(classification):
            if row["delta_vs_randproj"] is None:
                continue
            scored += 1
            if not row["control_ok"]:
                degenerate.append(row["target"])
                continue
            if row["tautological"]:
                continue
            considered += 1
            if row["delta_vs_randproj"] > 0.05:
                beats.append({"target": row["target"], "delta": row["delta_vs_randproj"]})
        worst = 0.0
        for row in regression:
            if row["shuffled"] is not None:
                worst = max(worst, abs(row["shuffled"]))
        honest = scored > 0 and len(degenerate) <= 0.3 * scored
        if considered == 0:
            reading = ("no non-tautological target has an admissible control on this window "
                       "-- accumulate more ticks before reading anything here")
        elif beats:
            reading = (f"IR beats the random-projection control on {len(beats)}/{considered} "
                       "non-tautological targets with admissible controls")
        else:
            reading = ("IR beats the random-projection control NOWHERE non-tautological "
                       "-- this is the R19 condition")
        return {
            "non_tautological_targets": considered,
            "beats_random_projection": beats,
            "n_beats": len(beats),
            "degenerate_targets": degenerate,
            "scored_targets": scored,
            "shuffled_label_control_passes": honest,
            "worst_shuffled_r2": round(worst, 4),
            "reading": reading,
        }

    @staticmethod
    def _pathology(geometry, verdict):
        """The R19 detector, stated as explicit alarms."""
        alarms = []
        x_rank = geometry.get("x_rank")
        x_nonzero = geometry.get("x_nonzero_columns")
        x_width = geometry.get("x_width")
        ir_width = geometry.get("ir_width")
        ir_rank = geometry.get("ir_rank")
        if x_rank is not None and x_rank <= 5:
            alarms.append({
                "id": "rank-collapsed-input", "severity": "critical",
                "text": f"input matrix rank {x_rank} over {geometry['rows']} rows -- "
                        "R19 measured rank 4 when the per-player resource block was unwired",
            })
        if x_nonzero is not None and x_width and x_nonzero <= 8:
            alarms.append({
                "id": "per-player-state-zeroed", "severity": "critical",
                "text": f"only {x_nonzero}/{x_width} input columns are nonzero -- "
                        "health/armor/ammo/weapons are not entering the matmul (SPEC 3, AGENDA E9)",
            })
        if ir_width is not None and ir_width < geometry.get("spec_ir_width_floor", 128):
            alarms.append({
                "id": "ir-too-narrow", "severity": "warning",
                "text": f"IR width {ir_width} < SPEC 8 floor 128",
            })
        if ir_rank is not None and ir_width and ir_rank < 0.1 * ir_width:
            alarms.append({
                "id": "ir-rank-collapse", "severity": "warning",
                "text": f"IR rank {ir_rank} over width {ir_width}: the wide IR is embedding a much smaller signal",
            })
        rows_per_feature = geometry.get("rows_per_feature")
        if rows_per_feature is not None and rows_per_feature < 4:
            alarms.append({
                "id": "underdetermined-probe", "severity": "warning",
                "text": f"only {rows_per_feature} player-rows per IR dimension in the window; "
                        "the ridge fit is underdetermined until more ticks accumulate",
            })
        if verdict is not None:
            if not verdict["shuffled_label_control_passes"]:
                alarms.append({
                    "id": "probe-dishonest", "severity": "critical",
                    "text": (f"{len(verdict['degenerate_targets'])} of {verdict['scored_targets']} targets "
                             f"failed their own shuffled-label control (worst |R2| "
                             f"{verdict['worst_shuffled_r2']}): the ridge fit is degenerate on this window "
                             f"and those columns are not admissible -- "
                             + ", ".join(verdict["degenerate_targets"][:8])),
                })
            elif verdict["n_beats"] == 0 and verdict["non_tautological_targets"] > 0:
                alarms.append({
                    "id": "no-jspace", "severity": "critical",
                    "text": "no non-tautological target beats the random-projection control "
                            "-- the R19 verdict (no semantically-rich j-space) holds on this run",
                })
        return {"alarms": alarms, "clear": not alarms}

    def status(self):
        with self.lock:
            report = self.report
            computed_at = self.computed_at
        return {
            "computed_at": computed_at,
            "age": None if computed_at is None else round(time.time() - computed_at, 1),
            "compute_ms": self.compute_ms,
            "errors": self.errors,
            "last_error": self.last_error,
            "report": report,
        }


__all__ = ["RollingProbe", "rows_from_frame", "REGRESSION_TARGETS", "CLASSIFICATION_TARGETS"]
