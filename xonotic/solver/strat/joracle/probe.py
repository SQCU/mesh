from __future__ import annotations

import json
import os
import threading
import tempfile
import time
from collections import deque

import numpy as np

from payload.tools.strategy_io_schema import (
    WEAPON_WORD_BITS, state_coordinate_kind,
)
from solver.strat.row_window import RowWindow
from solver.strat.joracle.display import compact_j_report
from solver.strat.joracle.artifact import scalar_report, write_report, matrix_factor

def _finite(array):
    return np.asarray(array, dtype=np.float64)

def literal_coordinate_delta(source, source_labels, target, target_labels):
    source = _finite(source).reshape(-1)
    target = _finite(target).reshape(-1)
    source_index = {name: index for index, name in enumerate(source_labels)}
    values = []
    labels = []
    categorical = []
    for target_index, name in enumerate(target_labels):
        if name not in source_index:
            continue
        source_value = source[source_index[name]]
        target_value = target[target_index]
        kind = state_coordinate_kind(name)
        if kind == "categorical":
            source_finite = bool(np.isfinite(source_value))
            target_finite = bool(np.isfinite(target_value))
            categorical.append({
                "coordinate": name,
                "source": int(source_value) if source_finite else None,
                "target": int(target_value) if target_finite else None,
                "source_finite_mass": float(source_finite),
                "target_finite_mass": float(target_finite),
            })
        elif kind == "bitset":
            for bit in range(WEAPON_WORD_BITS):
                labels.append(f"{name}.bit.{bit}")
                values.append(
                    ((int(target_value) >> bit) & 1) - ((int(source_value) >> bit) & 1)
                    if np.isfinite(source_value) and np.isfinite(target_value) else np.nan
                )
        else:
            labels.append(name)
            values.append(float(target_value - source_value))
    return values, labels, categorical

def literal_state_lens_coordinates(state, labels):
    state = _finite(state)
    parts = []
    names = []
    for index, name in enumerate(labels):
        values = state[:, index]
        kind = state_coordinate_kind(name)
        if kind == "bitset":
            finite = np.isfinite(values)
            words = np.zeros(len(values), dtype=np.int64)
            words[finite] = values[finite].astype(np.int64)
            bits = ((words[:, None] >> np.arange(WEAPON_WORD_BITS)) & 1).astype(np.float64)
            bits[~finite] = np.nan
            parts.append(bits)
            names.extend(f"{name}.bit.{bit}" for bit in range(WEAPON_WORD_BITS))
        elif kind == "categorical":
            categories = np.unique(values[np.isfinite(values)])
            parts.append((values[:, None] == categories[None, :]).astype(np.float64))
            names.extend(f"{name}={int(value)}" for value in categories)
            if not np.isfinite(values).all():
                parts.append((~np.isfinite(values)).reshape(-1, 1).astype(np.float64))
                names.append(f"{name}=nonfinite")
        else:
            parts.append(values.reshape(-1, 1))
            names.append(name)
    return (
        np.concatenate(parts, axis=1) if parts else np.empty((len(state), 0), dtype=np.float64),
        names,
    )

def literal_source_coordinates(model, server_state_labels=()):
    x = _finite(model["x"])
    n = x.shape[0]
    features = model.get("source_features")
    source = x if features is None else np.broadcast_to(_finite(features), (n, np.size(features)))
    names = list(model.get("x_labels") or (f"state.{index}" for index in range(x.shape[1]))) if features is None else list(model["source_labels"])
    state = _finite(model.get("server_state")) if model.get("server_state") is not None else None
    state_names = list(server_state_labels)
    if state is None or state.ndim != 2 or state.shape[0] != n:
        state = np.empty((n, 0), dtype=np.float64)
        state_names = []
    return source, names, state, state_names

def rows_from_frame(frame):
    model = frame.get("model") or {}
    x = model.get("x")
    if x is None:
        return None
    x = _finite(x)
    if x.ndim != 2 or x.shape[0] == 0:
        return None
    n = x.shape[0]
    j = _finite(model.get("j")) if model.get("j") is not None else None
    if j is None or j.ndim != 2 or j.shape[0] != n:
        j = np.empty((n, 0), dtype=np.float64)
    assignments = frame.get("assignments") or []
    assignments = sorted(assignments, key=lambda a: a.get("row", 0)) if len(assignments) == n else []
    participant_ids = np.asarray(
        [int(item.get("edict") or 0) for item in assignments], dtype=np.int64,
    ) if assignments else None
    lens_input, lens_names, server_state, server_state_labels = literal_source_coordinates(
        model, frame.get("server_state_labels") or (),
    )
    return {
        "row_mass": n,
        "j": j,
        "j_labels": tuple(model.get("j_labels") or (
            f"j.{index}" for index in range(j.shape[1])
        )),
        "x": x,
        "lens_input": lens_input,
        "lens_names": lens_names,
        "server_state": server_state,
        "server_state_labels": server_state_labels,
        "participant_ids": participant_ids,
        "composer": {
            name: _finite(model[name])
            for name in model.get("composer_inputs") or ()
            if model.get(name) is not None
        },
        "matrix_fusion_intervention": model.get("matrix_fusion_intervention"),
        "request_seq": int(
            frame.get("request_seq")
            or (assignments[0].get("request_seq") if assignments else 0)
            or 0
        ),
        "tick": int(frame.get("resp_id") or 0),
        "epoch": int(frame.get("_epoch") or 0),
    }

def literal_outcome_coordinates(values):
    out = {}
    for name, value in values.items():
        numeric = isinstance(value, (int, float))
        out[f"{name}.value_present_mass"] = 1.0
        out[f"{name}.value_numeric_mass"] = float(numeric)
        out[f"{name}.value_finite_mass"] = float(numeric and np.isfinite(value))
        if numeric:
            out[name] = float(value)
    return out

def transition_from_frame(frame):
    assignments = frame.get("assignments") or []
    sources = {
        (int(item.get("response_seq") or 0), int(item.get("edict") or 0)): item
        for item in frame.get("measure_sources") or ()
    }
    return {
        "sources": [
            {
                **item,
                "state_labels": item.get("state_labels", frame.get("server_state_labels") or ()),
                "feature_labels": item.get(
                    "feature_labels", (frame.get("model") or {}).get("lens_names") or (),
                ),
            }
            for item in sources.values()
        ],
        "state_references": [
            {
                "edict": int(item.get("edict") or 0),
                "response_seq": int(sequence or 0),
                "relation": relation,
                "applied_state_observation": int(bool(item.get("applied_state_current"))),
                "successor_state": item.get("successor_state"),
                "successor_state_labels": item.get("successor_state_labels") or (),
                "source": sources.get((
                    int(sequence or 0),
                    int(item.get("edict") or 0),
                )),
            }
            for item in assignments
            for relation, sequence in (
                ("state_application", item.get("applied_response_seq")),
            )
            if int(sequence or 0) > 0
        ],
        "applied": [
            {
                "edict": int(item.get("edict") or 0),
                "response_seq": int(routed.get("response_seq") or 0),
                "outcomes": literal_outcome_coordinates(routed.get("outcomes") or {}),
                "source": sources.get((
                    int(routed.get("response_seq") or 0),
                    int(item.get("edict") or 0),
                )),
            }
            for item in assignments
            for routed in item.get("routed_outcomes") or ()
            if int(routed.get("response_seq") or 0) > 0
        ],
        "events": [
            {
                "edict": int(event.get("actor") or 0),
                "response_seq": int(event.get("response_seq") or 0),
                "outcomes": {
                    f"{event.get('kind') or 'unknown'}.mass": 1.0,
                    f"{event.get('kind') or 'unknown'}.value_present_mass": float(
                        "value" in event
                    ),
                    f"{event.get('kind') or 'unknown'}.value_numeric_mass": float(
                        isinstance(event.get("value"), (int, float))
                    ),
                    f"{event.get('kind') or 'unknown'}.value_finite_mass": float(
                        isinstance(event.get("value"), (int, float))
                        and np.isfinite(event.get("value"))
                    ),
                    **({
                        f"{event.get('kind') or 'unknown'}.value": float(event["value"]),
                    } if isinstance(event.get("value"), (int, float))
                         and np.isfinite(event.get("value")) else {}),
                },
                "source": sources.get((
                    int(event.get("response_seq") or 0),
                    int(event.get("actor") or 0),
                )),
            }
            for event in frame.get("realized_events") or ()
            if int(event.get("response_seq") or 0) > 0 and event.get('episode_id') == frame.get('episode_id')
        ],
        "state_labels": list(frame.get("server_state_labels") or ()),
        "source_window": dict(frame.get("source_window") or {}),
        "tick": int(frame.get("resp_id") or 0),
        "epoch": int(frame.get("_epoch") or 0),
    }

def _transition_mass(item):
    return max(
        1, len(item["sources"]),
        len(item["state_references"]) + len(item["applied"]) + len(item["events"]),
    )

def _scalar_measure(values, j=None, controls=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    observed = values[finite]
    measure = {
        "mass": len(values),
        "finite_mass": int(finite.sum()),
        "nonfinite_mass": int((~finite).sum()),
        "integral": float(observed.sum()),
        "mean": None if len(observed) == 0 else float(observed.mean()),
        "variance": None,
    }
    if len(observed):
        centered = observed - observed.mean()
        measure["variance"] = float(np.mean(centered * centered))
    if j is not None:
        j = np.asarray(j, dtype=np.float64)
        aligned = j.ndim == 2 and j.shape[0] == len(values)
        joint = finite & np.isfinite(j).all(axis=1) if aligned else np.zeros(len(values), dtype=bool)
        measure["j_joint_finite_mass"] = int(joint.sum())
        measure["j_row_alignment_residual_mass"] = 0 if aligned else abs(
            len(values) - (j.shape[0] if j.ndim == 2 else 0)
        )
        if joint.any():
            joint_values = values[joint]
            joint_j = j[joint]
            measure["j_cross_moment"] = np.mean(
                joint_j * joint_values[:, None], axis=0,
            )
            measure["j_covariance"] = np.mean(
                (joint_j - joint_j.mean(axis=0, keepdims=True))
                * (joint_values - joint_values.mean())[:, None], axis=0,
            )
        else:
            measure["j_cross_moment"] = None
            measure["j_covariance"] = None
    if controls is not None:
        controls = np.asarray(controls, dtype=np.float64)
        aligned = controls.ndim == 2 and controls.shape[0] == len(values)
        joint = finite & np.isfinite(controls).all(axis=1) if aligned else np.zeros(len(values), dtype=bool)
        measure["control_joint_finite_mass"] = int(joint.sum())
        measure["control_row_alignment_residual_mass"] = 0 if aligned else abs(
            len(values) - (controls.shape[0] if controls.ndim == 2 else 0)
        )
        if joint.any():
            joint_values = values[joint]
            joint_controls = controls[joint]
            measure["control_cross_moment"] = np.mean(
                joint_controls * joint_values[:, None], axis=0,
            )
            measure["control_covariance"] = np.mean(
                (joint_controls - joint_controls.mean(axis=0, keepdims=True))
                * (joint_values - joint_values.mean())[:, None], axis=0,
            )
        else:
            measure["control_cross_moment"] = None
            measure["control_covariance"] = None
    return measure

def _joint_scalar_measure(values, j_rows, j_labels, controls=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    base = _scalar_measure(values, controls=controls)
    groups = {}
    for index, (row, labels) in enumerate(zip(j_rows, j_labels)):
        shape = np.asarray(row).shape
        groups.setdefault((shape, tuple(labels)), []).append(index)
    strata = []
    for (shape, labels), indices in groups.items():
        selected_values = values[indices]
        selected_j = np.asarray([j_rows[index] for index in indices], dtype=np.float64)
        strata.append({
            "j_coordinates": shape[0] if len(shape) == 1 else 0,
            "j_labels": list(labels),
            **_scalar_measure(selected_values, selected_j),
        })
    base["j_strata"] = strata
    base["j_strata_mass"] = len(strata)
    base["j_joint_finite_mass"] = sum(
        int(stratum["j_joint_finite_mass"]) for stratum in strata
    )
    base["j_row_alignment_residual_mass"] = sum(
        int(stratum["j_row_alignment_residual_mass"]) for stratum in strata
    )
    base["j_cross_moment"] = strata[0]["j_cross_moment"] if len(strata) == 1 else None
    base["j_covariance"] = strata[0]["j_covariance"] if len(strata) == 1 else None
    return base

def sum_measures(measures):
    mass = sum(int(measure["mass"]) for measure in measures)
    integral = sum(float(measure["integral"]) for measure in measures)
    square_integral = sum(float(measure["square_integral"]) for measure in measures)
    mean = integral / mass
    return {
        "mass": mass,
        "integral": integral,
        "square_integral": square_integral,
        "mean": mean,
        "variance": max(0.0, square_integral / mass - mean * mean),
        "minimum": min(float(measure["minimum"]) for measure in measures),
        "maximum": max(float(measure["maximum"]) for measure in measures),
    }

def authoritative_state_strata(rows, state_key, labels_key, projection_name):
    state_rows = [
        row for row in rows
        if np.asarray(row.get(state_key)).ndim == 1
        and np.asarray(row.get("j")).ndim == 1
    ]
    groups = {}
    for row in state_rows:
        key = (
            np.asarray(row[state_key]).shape,
            np.asarray(row["j"]).shape,
            tuple(row.get(labels_key) or ()),
            tuple(row.get("j_labels") or ()),
            str(row.get("policy_arm") or "unknown"),
            str(row.get("channel") or "source"),
        )
        groups.setdefault(key, []).append(row)
    strata = []
    for (state_shape, j_shape, labels, j_labels, policy_arm, channel), items in groups.items():
        wire_state = np.asarray([item[state_key] for item in items], dtype=np.float64)
        state, state_labels = literal_state_lens_coordinates(wire_state, labels)
        j = np.asarray([item["j"] for item in items], dtype=np.float64)
        finite = np.isfinite(state).all(axis=1) & np.isfinite(j).all(axis=1)
        finite_state = state[finite]
        finite_j = j[finite]
        centered_state = finite_state - finite_state.mean(axis=0, keepdims=True) if len(finite_state) else finite_state
        centered_j = finite_j - finite_j.mean(axis=0, keepdims=True) if len(finite_j) else finite_j
        strata.append({
            "mass": len(items),
            "finite_atom_mass": int(finite.sum()),
            "nonfinite_atom_mass": int((~finite).sum()),
            "wire_state_coordinates": state_shape[0],
            "wire_state_labels": list(labels),
            "state_coordinates": state.shape[1],
            "j_coordinates": j_shape[0],
            "j_labels": list(j_labels),
            "policy_arm": policy_arm,
            "channel": channel,
            "state_labels": state_labels,
            "state_integral": finite_state.sum(axis=0),
            "state_mean": None if len(finite_state) == 0 else finite_state.mean(axis=0),
            "state_variance": None if len(finite_state) == 0 else np.mean(centered_state * centered_state, axis=0),
            "state_j_cross_moment": None if len(finite_state) == 0 else matrix_factor(finite_state, finite_j, len(finite_state)),
            "state_j_covariance": None if len(finite_state) == 0 else matrix_factor(centered_state, centered_j, len(finite_state)),
            projection_name: empirical_affine_projection(
                j, state, j_labels, state_labels,
            ),
        })
    newest_key = next(reversed(groups), None)
    return state_rows, groups.get(newest_key, []), strata, newest_key

def empirical_affine_projection(domain, codomain, domain_labels=(), codomain_labels=()):
    domain = _finite(domain)
    codomain = _finite(codomain)
    domain_mass = domain.shape[0] if domain.ndim == 2 else 0
    codomain_mass = codomain.shape[0] if codomain.ndim == 2 else 0
    aligned = domain.ndim == 2 and codomain.ndim == 2 and domain_mass == codomain_mass
    atom_mass = domain_mass if aligned else 0
    base = {
        "definition": "minimum-norm affine L2 projection over the observed finite empirical measure",
        "atom_mass": atom_mass,
        "domain_atom_mass": domain_mass,
        "codomain_atom_mass": codomain_mass,
        "row_alignment_residual_mass": abs(domain_mass - codomain_mass),
        "domain_coordinates": domain.shape[1] if domain.ndim == 2 else 0,
        "codomain_coordinates": codomain.shape[1] if codomain.ndim == 2 else 0,
        "domain_labels": list(domain_labels),
        "codomain_labels": list(codomain_labels),
    }
    empty = {
        "decomposition_mass": 0,
        "decomposition_residual_mass": 0,
        "domain_numerical_rank": 0,
        "domain_singular_values": [],
        "operator": None,
        "offset": None,
        "target_centered_square_integral": None,
        "image_centered_square_integral": None,
        "residual_square_integral": None,
        "residual_mean_square": None,
    }
    if not aligned:
        return {**base, "finite_atom_mass": 0, "nonfinite_atom_mass": 0, **empty}
    finite = np.isfinite(domain).all(axis=1) & np.isfinite(codomain).all(axis=1)
    source = domain[finite]
    target = codomain[finite]
    finite_mass = source.shape[0]
    base.update({
        "finite_atom_mass": finite_mass,
        "nonfinite_atom_mass": atom_mass - finite_mass,
    })
    if finite_mass == 0:
        return {**base, **empty}
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    try:
        u, singular_values, vt = np.linalg.svd(centered_source, full_matrices=False)
        threshold = np.finfo(centered_source.dtype).eps * max(centered_source.shape) * (singular_values[0] if len(singular_values) else 0)
        keep = singular_values > threshold
        rank = np.count_nonzero(keep)
        inverse_transpose = (u[:, keep] / singular_values[keep]) @ vt[keep]
    except np.linalg.LinAlgError:
        return {**base, **empty, "decomposition_residual_mass": 1}
    image = (centered_source @ inverse_transpose.T) @ centered_target
    residual = centered_target - image
    target_square = np.sum(centered_target * centered_target, axis=0)
    image_square = np.sum(image * image, axis=0)
    residual_square = np.sum(residual * residual, axis=0)
    return {
        **base,
        "decomposition_mass": 1,
        "decomposition_residual_mass": 0,
        "domain_numerical_rank": int(rank),
        "domain_singular_values": singular_values,
        "operator": matrix_factor(centered_target, inverse_transpose),
        "offset": (target_mean - (source_mean @ inverse_transpose.T) @ centered_target),
        "target_centered_square_integral": target_square,
        "image_centered_square_integral": image_square,
        "residual_square_integral": residual_square,
        "residual_mean_square": (residual_square / finite_mass),
    }

def matrix_fusion_intervention_measures(frames):
    if not frames:
        return None
    measure_names = sorted({name for frame in frames for name in frame["measures"]})
    return {
        "definition": "same-state paired pushforwards under independent participant and residual fusion interventions with shared parameters",
        "left": "matrix_fusion",
        "participant_right": "participant_fusion_ablated",
        "residual_right": "residual_fusion_ablated",
        "frame_mass": len(frames),
        "participant_measure": _scalar_measure([frame["participant_mass"] for frame in frames]),
        "instrument_measure": _scalar_measure([frame["instrument_mass"] for frame in frames]),
        "j_width_measure": _scalar_measure([frame["j_width"] for frame in frames]),
        "measures": {
            name: sum_measures([
                frame["measures"][name] for frame in frames if name in frame["measures"]
            ])
            for name in measure_names
        },
    }

def literal_j_measures(buffer, transitions):
    source_window = next((
        dict(transition["source_window"])
        for transition in reversed(transitions)
        if transition.get("source_window")
    ), {})
    exact_sources = {}
    for transition in transitions:
        for source in transition.get("sources") or ():
            key = (
                int(transition["epoch"]), int(source.get("response_seq") or 0),
                int(source.get("edict") or 0),
            )
            exact_sources[key] = source
    lens_rows = list(exact_sources.values())
    for item in buffer:
        sequence = int(item["request_seq"])
        if sequence <= 0 or item["j"].shape[1] == 0:
            continue
        for index in range(item["j"].shape[0]):
            key = None if item["participant_ids"] is None else (
                int(item["epoch"]), sequence, int(item["participant_ids"][index]),
            )
            if key in exact_sources:
                continue
            lens_rows.append({
                "features": item["lens_input"][index],
                "feature_labels": item["lens_names"],
                "j": item["j"][index],
                "j_labels": item["j_labels"],
                "server_state": item["server_state"][index],
                "state_labels": item["server_state_labels"],
            })
    lens_rows = [
        source for source in lens_rows
        if np.asarray(source.get("features")).ndim == 1
        and np.asarray(source.get("j")).ndim == 1
    ]
    newest_lens_key = (
        np.asarray(lens_rows[-1]["features"]).shape,
        np.asarray(lens_rows[-1]["j"]).shape,
        tuple(lens_rows[-1].get("feature_labels") or ()),
        tuple(lens_rows[-1].get("j_labels") or ()),
    ) if lens_rows else None
    lens_strata = {}
    for source in lens_rows:
        key = (
            np.asarray(source["features"]).shape,
            np.asarray(source["j"]).shape,
            tuple(source.get("feature_labels") or ()),
            tuple(source.get("j_labels") or ()),
        )
        lens_strata.setdefault(key, []).append(source)
    coordinate_rows = lens_strata.get(newest_lens_key, [])
    lens = np.asarray([source["features"] for source in coordinate_rows], dtype=np.float64)
    j = np.asarray([source["j"] for source in coordinate_rows], dtype=np.float64)
    if lens.ndim != 2:
        lens = np.empty((0, 0), dtype=np.float64)
    if j.ndim != 2:
        j = np.empty((0, 0), dtype=np.float64)
    coordinate_finite = np.isfinite(lens).all(axis=1) & np.isfinite(j).all(axis=1)
    finite_lens = lens[coordinate_finite]
    finite_j = j[coordinate_finite]
    lens_centered = finite_lens - finite_lens.mean(axis=0, keepdims=True) if len(finite_lens) else finite_lens
    j_centered = finite_j - finite_j.mean(axis=0, keepdims=True) if len(finite_j) else finite_j
    mass = len(j)
    coordinate_strata = []
    for (feature_shape, j_shape, labels, j_labels), rows in lens_strata.items():
        stratum_features = np.asarray([source["features"] for source in rows], dtype=np.float64)
        stratum_j = np.asarray([source["j"] for source in rows], dtype=np.float64)
        finite = np.isfinite(stratum_features).all(axis=1) & np.isfinite(stratum_j).all(axis=1)
        finite_features = stratum_features[finite]
        finite_j_rows = stratum_j[finite]
        centered_features = finite_features - finite_features.mean(axis=0, keepdims=True) if len(finite_features) else finite_features
        centered_j = finite_j_rows - finite_j_rows.mean(axis=0, keepdims=True) if len(finite_j_rows) else finite_j_rows
        input_variance = np.mean(centered_features * centered_features, axis=0) if len(finite_features) else None
        j_variance = np.mean(centered_j * centered_j, axis=0) if len(finite_j_rows) else None
        cross_covariance = matrix_factor(centered_features, centered_j, len(finite_features)) if len(finite_features) else None
        coordinate_strata.append({
            "mass": len(rows),
            "finite_atom_mass": int(finite.sum()),
            "nonfinite_atom_mass": int((~finite).sum()),
            "input_coordinates": feature_shape[0],
            "j_coordinates": j_shape[0],
            "j_labels": list(j_labels),
            "input_labels": list(labels),
            "input_integral": finite_features.sum(axis=0),
            "j_integral": finite_j_rows.sum(axis=0),
            "input_variance": input_variance,
            "j_variance": j_variance,
            "cross_covariance": cross_covariance,
            "j_to_source_feature_affine_projection": empirical_affine_projection(
                stratum_j, stratum_features,
                j_labels, labels,
            ),
        })
    joined = []
    state_reference_mass = {}
    state_reference_exact_source_mass = {}
    applied_mass = 0
    event_mass = 0
    applied_exact_source_mass = 0
    event_exact_source_mass = 0

    def join(
        target, observation, channel, outcomes, state_delta=None, state_labels=(),
        state_categorical_transitions=(), successor_state=None, successor_state_labels=(),
    ):
        exact = observation.get("source")
        if exact is None:
            return
        source_j = np.asarray(exact["j"], dtype=np.float64)
        source_j_labels = tuple(exact.get("j_labels") or (
            f"j.{index}" for index in range(len(source_j))
        ))
        action = exact["action"]
        policy_arm = str(exact.get("policy_arm") or "unknown")
        behavior = str(exact.get("behavior") or "unknown")
        source_state = np.asarray(exact.get("server_state"), dtype=np.float64)
        source_features = np.asarray(exact.get("features"), dtype=np.float64)
        if successor_state is not None:
            state_delta, state_labels, state_categorical_transitions = literal_coordinate_delta(
                source_state, exact.get("state_labels") or (),
                successor_state, successor_state_labels,
            )
        joined.append({
            "j": source_j,
            "j_labels": source_j_labels,
            "action": action,
            "policy_arm": policy_arm,
            "behavior": behavior,
            "source_state": source_state,
            "source_features": source_features,
            "channel": channel,
            "outcomes": outcomes,
            "state_delta": state_delta,
            "state_labels": state_labels,
            "state_categorical_transitions": state_categorical_transitions,
            "successor_state": successor_state,
            "successor_state_labels": successor_state_labels,
        })

    for target in transitions:
        for reference in target["state_references"]:
            relation = reference["relation"]
            state_reference_mass[relation] = state_reference_mass.get(relation, 0) + 1
            state_reference_exact_source_mass[relation] = (
                state_reference_exact_source_mass.get(relation, 0)
                + (reference.get("source") is not None)
            )
            join(
                target, reference, relation,
                {
                    "mass": 1.0,
                    "applied_state_observation": float(reference["applied_state_observation"]),
                },
                successor_state=reference.get("successor_state"),
                successor_state_labels=reference.get("successor_state_labels") or (),
            )
        for applied in target["applied"]:
            applied_mass += 1
            applied_exact_source_mass += applied.get("source") is not None
            join(target, applied, "behavior", applied["outcomes"])
        for event in target["events"]:
            event_mass += 1
            event_exact_source_mass += event.get("source") is not None
            join(target, event, "event", event["outcomes"])
    def outcome_measure_map(rows):
        measures = {}
        for name in sorted({
            f"{item['channel']}.{field}"
            for item in rows for field in item["outcomes"]
        }):
            channel, field = name.split(".", 1)
            selected = [
                item for item in rows
                if item["channel"] == channel and field in item["outcomes"]
            ]
            values = np.asarray([item["outcomes"][field] for item in selected], dtype=np.float64)
            j_rows = [item["j"] for item in selected]
            controls = np.asarray([item["action"]["velocity"] for item in selected], dtype=np.float64)
            measures[name] = _joint_scalar_measure(
                values, j_rows, [item["j_labels"] for item in selected], controls,
            )
        return measures

    def state_measure_map(rows):
        groups = {}
        for item in rows:
            if item["channel"] in state_reference_mass and item["state_delta"] is not None:
                key = (item["channel"], item["policy_arm"], tuple(item["state_labels"]), tuple(item["j_labels"]), len(item["action"]["velocity"]))
                groups.setdefault(key, []).append(item)
        measures = []
        for (channel, arm, labels, j_labels, width), items in groups.items():
            values = np.asarray([item["state_delta"] for item in items], dtype=np.float64)
            controls = np.asarray([item["action"]["velocity"] for item in items], dtype=np.float64)
            j = np.asarray([item["j"] for item in items], dtype=np.float64)
            finite = np.isfinite(values)
            mass = finite.sum(axis=0)
            total = np.where(finite, values, 0).sum(axis=0)
            mean = total / np.maximum(mass, 1)
            centered = np.where(finite, values - mean, 0)
            joint = finite.all(axis=1) & np.isfinite(controls).all(axis=1) & np.isfinite(j).all(axis=1)
            source, target, latent = values[joint], controls[joint], j[joint]
            count = len(source)
            left = source - source.mean(axis=0) if count else source
            right = target - target.mean(axis=0) if count else target
            measures.append({"channel": channel, "policy_arm": arm, "state_labels": labels, "j_labels": j_labels,
                "mass": len(items), "finite_mass": mass, "integral": total, "mean": mean,
                "variance": np.sum(centered * centered, axis=0) / np.maximum(mass, 1),
                "joint_finite_mass": count, "j_covariance": matrix_factor(left, latent, count),
                "control_covariance": {"representation": "left_transpose_times_right_over_mass",
                    "left": left, "right": right, "mass": count, "shape": [len(labels), width]}})
        return measures

    def categorical_transition_measure_map(rows):
        accumulators = {}
        for item in rows:
            if item["channel"] not in state_reference_mass:
                continue
            j_row = np.asarray(item["j"], dtype=np.float64)
            controls = np.asarray(item["action"]["velocity"], dtype=np.float64)
            for transition in item["state_categorical_transitions"]:
                key = (
                    item["channel"],
                    transition["coordinate"],
                    int(transition["source_finite_mass"]),
                    int(transition["source"] or 0),
                    int(transition["target_finite_mass"]),
                    int(transition["target"] or 0),
                    tuple(j_row.shape),
                    tuple(item["j_labels"]),
                )
                measure = accumulators.setdefault(key, {
                    "mass": 0,
                    "j_finite_mass": 0,
                    "control_finite_mass": 0,
                    "j_integral": np.zeros_like(j_row),
                    "control_integral": np.zeros_like(controls),
                })
                measure["mass"] += 1
                if np.isfinite(j_row).all():
                    measure["j_finite_mass"] += 1
                    measure["j_integral"] += j_row
                if np.isfinite(controls).all():
                    measure["control_finite_mass"] += 1
                    measure["control_integral"] += controls
        return [
            {
                "channel": channel,
                "coordinate": coordinate,
                "source": source if source_finite else None,
                "target": target if target_finite else None,
                "source_finite_mass": source_finite * measure["mass"],
                "target_finite_mass": target_finite * measure["mass"],
                "j_coordinates": j_shape[0] if len(j_shape) == 1 else 0,
                "j_labels": list(j_labels),
                "mass": measure["mass"],
                "j_finite_mass": measure["j_finite_mass"],
                "j_nonfinite_mass": measure["mass"] - measure["j_finite_mass"],
                "j_integral": measure["j_integral"],
                "j_mean": (
                    (measure["j_integral"] / measure["j_finite_mass"])
                    if measure["j_finite_mass"] else None
                ),
                "control_finite_mass": measure["control_finite_mass"],
                "control_nonfinite_mass": measure["mass"] - measure["control_finite_mass"],
                "control_integral": measure["control_integral"],
                "control_mean": (
                    (measure["control_integral"] / measure["control_finite_mass"])
                    if measure["control_finite_mass"] else None
                ),
            }
            for (
                channel, coordinate, source_finite, source,
                target_finite, target, j_shape, j_labels,
            ), measure in sorted(accumulators.items())
        ]

    def outcome_projection_strata(rows):
        groups = {}
        for item in rows:
            labels = tuple(sorted(item["outcomes"]))
            if labels:
                groups.setdefault((
                    item["policy_arm"], item["channel"], labels,
                    tuple(item["j"].shape), item["j_labels"],
                ), []).append(item)
        return [
            {
                "policy_arm": policy_arm,
                "channel": channel,
                "outcome_labels": list(labels),
                "j_coordinates": j_shape[0] if len(j_shape) == 1 else 0,
                "j_labels": list(j_labels),
                "mass": len(items),
                "j_to_outcome_affine_projection": empirical_affine_projection(
                    np.asarray([item["j"] for item in items], dtype=np.float64),
                    np.asarray([
                        [item["outcomes"][name] for name in labels] for item in items
                    ], dtype=np.float64),
                    j_labels, labels,
                ),
            }
            for (policy_arm, channel, labels, j_shape, j_labels), items in groups.items()
        ]

    def state_delta_projection_strata(rows):
        groups = {}
        for item in rows:
            labels = tuple(item["state_labels"])
            if (
                item["channel"] in state_reference_mass and item["state_delta"] is not None
                and len(item["state_delta"]) == len(labels)
            ):
                groups.setdefault((
                    item["policy_arm"], item["channel"], labels,
                    tuple(item["j"].shape), item["j_labels"],
                ), []).append(item)
        return [
            {
                "policy_arm": policy_arm,
                "channel": channel,
                "state_labels": list(labels),
                "j_coordinates": j_shape[0] if len(j_shape) == 1 else 0,
                "j_labels": list(j_labels),
                "mass": len(items),
                "j_to_state_delta_affine_projection": empirical_affine_projection(
                    np.asarray([item["j"] for item in items], dtype=np.float64),
                    np.asarray([item["state_delta"] for item in items], dtype=np.float64),
                    j_labels, labels,
                ),
            }
            for (policy_arm, channel, labels, j_shape, j_labels), items in groups.items()
        ]

    outcome_measures = outcome_measure_map(joined)
    action_measures = []
    action_keys = sorted({(item["channel"], item["policy_arm"], item["behavior"],
                           len(item["action"]["velocity"])) for item in joined})
    for channel, policy_arm, behavior, width in action_keys:
        selected = [item for item in joined if
                    (item["channel"], item["policy_arm"], item["behavior"], len(item["action"]["velocity"]))
                    == (channel, policy_arm, behavior, width)]
        names = sorted({name for item in selected for name in item["outcomes"]})
        velocities = np.asarray([item["action"]["velocity"] for item in selected], dtype=np.float64)
        action_measures.append({
            "channel": channel, "policy_arm": policy_arm, "behavior": behavior,
            "state_width": width, "mass": len(selected),
            "rate_integral": velocities.sum(axis=0), "rate_mean": velocities.mean(axis=0),
            "outcomes": {name: _scalar_measure(values) for name in names
                         if (values := [item["outcomes"][name] for item in selected if name in item["outcomes"]])},
        })
    state_delta_measures = state_measure_map(joined)
    state_categorical_transition_measures = categorical_transition_measure_map(joined)
    joined_outcome_projection_strata = outcome_projection_strata(joined)
    joined_state_delta_projection_strata = state_delta_projection_strata(joined)
    policy_arm_measures = {}
    for arm in sorted({
        str(source.get("policy_arm") or "unknown") for source in exact_sources.values()
    } | {item["policy_arm"] for item in joined}):
        arm_sources = [
            source for source in exact_sources.values()
            if str(source.get("policy_arm") or "unknown") == arm
        ]
        arm_rows = [item for item in joined if item["policy_arm"] == arm]
        policy_arm_measures[arm] = {
            "source_mass": len(arm_sources),
            "state_reference_mass": {
                relation: sum(item["channel"] == relation for item in arm_rows)
                for relation in state_reference_mass
            },
            "behavior_mass": {
                behavior: sum(
                    str(source.get("behavior") or "unknown") == behavior
                    for source in arm_sources
                )
                for behavior in sorted({
                    str(source.get("behavior") or "unknown") for source in arm_sources
                })
            },
            "delivery_mass": sum(item["channel"] == "delivery" for item in arm_rows),
            "outcome_mass": sum(item["channel"] == "behavior" for item in arm_rows),
            "event_mass": sum(item["channel"] == "event" for item in arm_rows),
            "outcome_measures": outcome_measure_map(arm_rows),
            "state_delta_measures": state_measure_map(arm_rows),
            "state_categorical_transition_measures": categorical_transition_measure_map(arm_rows),
        }
    behavior_policy_measures = {}
    for behavior in sorted({item["behavior"] for item in joined}):
        behavior_rows = [item for item in joined if item["behavior"] == behavior]
        behavior_policy_measures[behavior] = {
            "source_mass": sum(
                str(source.get("behavior") or "unknown") == behavior
                for source in exact_sources.values()
            ),
            "state_reference_mass": {
                relation: sum(item["channel"] == relation for item in behavior_rows)
                for relation in state_reference_mass
            },
            "delivery_mass": sum(item["channel"] == "delivery" for item in behavior_rows),
            "outcome_mass": sum(item["channel"] == "behavior" for item in behavior_rows),
            "event_mass": sum(item["channel"] == "event" for item in behavior_rows),
            "outcome_measures": outcome_measure_map(behavior_rows),
            "state_delta_measures": state_measure_map(behavior_rows),
            "state_categorical_transition_measures": categorical_transition_measure_map(behavior_rows),
        }
    composer_measures = {}
    for name in sorted({name for item in buffer for name in item["composer"]}):
        values = np.concatenate([
            item["composer"][name].reshape(-1)
            for item in buffer if name in item["composer"]
        ]).astype(np.float64)
        composer_measures[name] = _scalar_measure(values)
        composer_measures[name]["shapes"] = [
            list(item["composer"][name].shape) for item in buffer if name in item["composer"]
        ]
    intervention_frames = [
        item["matrix_fusion_intervention"] for item in buffer if item.get("matrix_fusion_intervention") is not None
    ]
    matrix_fusion_intervention = matrix_fusion_intervention_measures(intervention_frames)
    finite_mass = len(finite_j)
    input_variance = np.mean(lens_centered * lens_centered, axis=0) if finite_mass else None
    j_variance = np.mean(j_centered * j_centered, axis=0) if finite_mass else None
    cross_covariance = (
        matrix_factor(lens_centered, j_centered, finite_mass)
        if finite_mass else None
    )
    state_rows, exact_state_sources, source_state_strata, newest_state_key = (
        authoritative_state_strata(
            list(exact_sources.values()), "server_state", "state_labels",
            "j_to_authoritative_state_affine_projection",
        )
    )
    successor_state_rows, exact_successor_states, successor_state_strata, newest_successor_key = (
        authoritative_state_strata(
            [item for item in joined if item["channel"] in state_reference_mass],
            "successor_state", "successor_state_labels",
            "j_to_authoritative_successor_state_affine_projection",
        )
    )
    if exact_state_sources:
        source_wire_states = np.asarray([
            source["server_state"] for source in exact_state_sources
        ], dtype=np.float64)
        source_states, source_state_labels = literal_state_lens_coordinates(
            source_wire_states, exact_state_sources[-1].get("state_labels") or [],
        )
        source_j = np.asarray([source["j"] for source in exact_state_sources], dtype=np.float64)
        source_state_finite = np.isfinite(source_states).all(axis=1) & np.isfinite(source_j).all(axis=1)
        finite_source_states = source_states[source_state_finite]
        finite_source_j = source_j[source_state_finite]
        source_state_mean = finite_source_states.mean(axis=0) if len(finite_source_states) else None
        source_state_centered = finite_source_states - source_state_mean[None, :] if len(finite_source_states) else finite_source_states
        source_j_centered = finite_source_j - finite_source_j.mean(axis=0, keepdims=True) if len(finite_source_j) else finite_source_j
        source_state_variance = np.mean(source_state_centered * source_state_centered, axis=0) if len(finite_source_states) else None
        source_state_cross_moment = matrix_factor(finite_source_states, finite_source_j, len(finite_source_states)) if len(finite_source_states) else None
        source_state_covariance = matrix_factor(source_state_centered, source_j_centered, len(finite_source_states)) if len(finite_source_states) else None
    else:
        source_states = np.empty((0, 0), dtype=np.float64)
        source_j = np.empty((0, j.shape[1]), dtype=np.float64)
        source_state_labels = []
        source_state_finite = np.empty(0, dtype=bool)
        source_state_mean = None
        source_state_variance = None
        source_state_cross_moment = None
        source_state_covariance = None
    outcome_covariance = [
        value for measure in outcome_measures.values() if measure.get("j_covariance") is not None for value in measure["j_covariance"]
    ]
    state_covariance = [
        measure['j_covariance']['frobenius_norm'] for measure in state_delta_measures if measure.get('j_covariance') is not None
    ]
    return {
        "j_lens": {
            "mass": mass,
            "finite_atom_mass": finite_mass,
            "nonfinite_atom_mass": mass - finite_mass,
            "definition": "empirical joint measure of full state inputs and policy intermediate rows",
            "source_atom_mass": len(exact_sources),
            "coordinate_atom_mass": len(coordinate_rows),
            "all_coordinate_atom_mass": len(lens_rows),
            "coordinate_strata_mass": len(coordinate_strata),
            "coordinate_strata": coordinate_strata,
            "input_coordinates": lens.shape[1],
            "j_coordinates": j.shape[1],
            "composer_families": len(composer_measures),
            "input_variance_integral": None if input_variance is None else float(input_variance.sum()),
            "j_variance_integral": None if j_variance is None else float(j_variance.sum()),
            "cross_covariance_frobenius": None if cross_covariance is None else cross_covariance['frobenius_norm'],
            "input_labels": coordinate_rows[-1].get("feature_labels") or [] if coordinate_rows else [],
            "j_labels": coordinate_rows[-1].get("j_labels") or [] if coordinate_rows else [],
            "input_integral": finite_lens.sum(axis=0),
            "j_integral": finite_j.sum(axis=0),
            "input_variance": input_variance,
            "j_variance": j_variance,
            "cross_covariance": cross_covariance,
            "composer_measures": composer_measures,
            "matrix_fusion_intervention": matrix_fusion_intervention,
        },
        "j_oracle": {
            "definition": "exact response-sequence pushforward measure from authoritative source state, J and full state-rate vector to subsequent server state and state-application outcomes",
            "source_window": source_window,
            "source_coordinate_mass": mass,
            "source_atom_mass": len(exact_sources),
            "source_state_coordinate_mass": len(exact_state_sources),
            "source_state_all_atom_mass": len(state_rows),
            "source_state_strata_mass": len(source_state_strata),
            "source_state_strata": source_state_strata,
            "source_state_mass": len(source_states),
            "source_state_finite_atom_mass": int(source_state_finite.sum()),
            "source_state_nonfinite_atom_mass": int((~source_state_finite).sum()),
            "source_wire_state_coordinates": 0 if newest_state_key is None else newest_state_key[0][0],
            "source_wire_state_labels": [] if newest_state_key is None else list(newest_state_key[2]),
            "source_state_coordinates": source_states.shape[1],
            "source_state_labels": source_state_labels,
            "source_state_integral": (
                source_states[source_state_finite].sum(axis=0)
                if len(source_states) else []
            ),
            "source_state_mean": source_state_mean,
            "source_state_variance": source_state_variance,
            "source_state_j_cross_moment": source_state_cross_moment,
            "source_state_j_covariance": source_state_covariance,
            "source_state_variance_integral": None if source_state_variance is None else float(source_state_variance.sum()),
            "source_state_j_covariance_frobenius": None if source_state_covariance is None else source_state_covariance['frobenius_norm'],
            "successor_state_atom_mass": len(successor_state_rows),
            "successor_state_coordinate_mass": len(exact_successor_states),
            "successor_state_strata_mass": len(successor_state_strata),
            "successor_state_strata": successor_state_strata,
            "successor_wire_state_coordinates": 0 if newest_successor_key is None else newest_successor_key[0][0],
            "successor_wire_state_labels": [] if newest_successor_key is None else list(newest_successor_key[2]),
            "state_reference_measures": {
                relation: {
                    "mass": relation_mass,
                    "exact_source_mass": state_reference_exact_source_mass.get(relation, 0),
                    "joined_mass": sum(item["channel"] == relation for item in joined),
                    "unjoined_mass": relation_mass - sum(item["channel"] == relation for item in joined),
                }
                for relation, relation_mass in state_reference_mass.items()
            },
            "state_application_mass": state_reference_mass.get("state_application", 0),
            "state_application_joined_mass": sum(item["channel"] == "state_application" for item in joined),
            "delivery_mass": state_reference_mass.get("delivery", 0),
            "delivery_exact_source_mass": state_reference_exact_source_mass.get("delivery", 0),
            "delivery_joined_mass": sum(item["channel"] == "delivery" for item in joined),
            "applied_mass": applied_mass,
            "applied_exact_source_mass": applied_exact_source_mass,
            "applied_joined_mass": sum(item["channel"] == "behavior" for item in joined),
            "event_mass": event_mass,
            "event_exact_source_mass": event_exact_source_mass,
            "event_joined_mass": sum(item["channel"] == "event" for item in joined),
            "unjoined_delivery_mass": state_reference_mass.get("delivery", 0) - sum(item["channel"] == "delivery" for item in joined),
            "unjoined_applied_mass": applied_mass - sum(item["channel"] == "behavior" for item in joined),
            "unjoined_event_mass": event_mass - sum(item["channel"] == "event" for item in joined),
            "outcome_coordinates": len(outcome_measures),
            "state_delta_coordinates": sum(len(group['state_labels']) for group in state_delta_measures),
            "state_categorical_transition_coordinates": len(state_categorical_transition_measures),
            "state_categorical_transition_atom_mass": sum(
                measure["mass"] for measure in state_categorical_transition_measures
            ),
            "outcome_variance_integral": float(sum(float(value.get("variance") or 0) for value in outcome_measures.values())),
            "state_delta_variance_integral": float(sum(np.sum(value["variance"]) for value in state_delta_measures)),
            "outcome_j_covariance_frobenius": float(np.linalg.norm(outcome_covariance)),
            "state_delta_j_covariance_frobenius": float(np.linalg.norm(state_covariance)),
            "rate_labels": [f"rate.{index}" for index in range(max(
                (len(source["action"]["velocity"]) for source in exact_sources.values()), default=0))],
            "outcome_measures": outcome_measures,
            "state_delta_measures": state_delta_measures,
            "state_categorical_transition_measures": state_categorical_transition_measures,
            "outcome_affine_projection_strata": joined_outcome_projection_strata,
            "state_delta_affine_projection_strata": joined_state_delta_projection_strata,
            "action_measures": action_measures,
            "policy_arm_measures": policy_arm_measures,
            "behavior_policy_measures": behavior_policy_measures,
        },
}

class LiteralJWindow:
    def __init__(self, max_rows=4000):
        self.max_rows = max(1, int(max_rows))
        self.coordinates = RowWindow(self.max_rows, lambda item: item["row_mass"])
        self.transition_window = RowWindow(self.max_rows, _transition_mass)
        self.sequence = 0

    def reset(self):
        self.coordinates.clear()
        self.transition_window.clear()
        self.sequence = 0

    def ingest(self, frame):
        transition = transition_from_frame(frame)
        incoming_transition_mass = _transition_mass(transition)
        row = rows_from_frame(frame)
        self.max_rows = max(self.max_rows, incoming_transition_mass)
        if row is not None:
            self.max_rows = max(self.max_rows, row["row_mass"])
        self.coordinates.capacity = self.max_rows
        self.transition_window.capacity = self.max_rows
        self.sequence += 1
        self.transition_window.put(self.sequence, transition)
        if row is not None:
            self.coordinates.put(self.sequence, row)

    def measure(self):
        buffer, transitions, observation_window = self.snapshot()
        report = literal_j_measures(buffer, transitions)
        report["observation_window"] = observation_window
        return report

    def snapshot(self):
        return list(self.coordinates.values()), list(self.transition_window.values()), {
            "row_capacity": self.max_rows,
            "ingested_coordinate_row_mass": self.coordinates.ingested_row_mass,
            "retained_coordinate_row_mass": self.coordinates.row_mass,
            "evicted_coordinate_row_mass": self.coordinates.evicted_row_mass,
            "ingested_transition_row_mass": self.transition_window.ingested_row_mass,
            "retained_transition_row_mass": self.transition_window.row_mass,
            "evicted_transition_row_mass": self.transition_window.evicted_row_mass,
        }

class LiteralJReporter:
    def __init__(self, max_rows=4000, interval=20.0, artifact_path=None):
        self.artifact_path = os.path.abspath(artifact_path or os.path.join(tempfile.mkdtemp(prefix="mesh-j-"), "j-measures"))
        self.max_rows = max(1, int(max_rows))
        self.interval = max(0.1, float(interval))
        self.lock = threading.Lock()
        self.pending = deque()
        self.generation = 0
        self.report = {}
        self.revision = 0
        self.errors = 0
        self.last_error = None
        self.pending_coordinate_row_mass = 0
        self.pending_transition_row_mass = 0
        self.ingested_coordinate_row_mass = 0
        self.ingested_transition_row_mass = 0
        self.evicted_pending_coordinate_row_mass = 0
        self.evicted_pending_transition_row_mass = 0
        self.stop_event = threading.Event()
        self.measure_event = threading.Event()
        self.needs_initial_measure = True
        self.thread = threading.Thread(target=self.run, name="literal-j-measures", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.measure_event.set()
        self.thread.join()

    def reset(self):
        with self.lock:
            self.pending.clear()
            self.generation += 1
            self.report = {}
            self.revision += 1
            self.needs_initial_measure = True
            self.pending_coordinate_row_mass = 0
            self.pending_transition_row_mass = 0
            self.ingested_coordinate_row_mass = 0
            self.ingested_transition_row_mass = 0
            self.evicted_pending_coordinate_row_mass = 0
            self.evicted_pending_transition_row_mass = 0

    def ingest(self, frame):
        sequence = int(frame.get("request_seq") or 0)
        coordinate_mass = sum(
            int(source.get("response_seq") or 0) == sequence
            for source in frame.get("measure_sources") or ()
        )
        transition_mass = _transition_mass(transition_from_frame(frame))
        with self.lock:
            self.max_rows = max(self.max_rows, coordinate_mass, transition_mass)
            self.pending.append((frame, coordinate_mass, transition_mass))
            self.pending_coordinate_row_mass += coordinate_mass
            self.pending_transition_row_mass += transition_mass
            self.ingested_coordinate_row_mass += coordinate_mass
            self.ingested_transition_row_mass += transition_mass
            while self.pending and (
                self.pending_coordinate_row_mass > self.max_rows
                or self.pending_transition_row_mass > self.max_rows
            ):
                _, removed_coordinates, removed_transitions = self.pending.popleft()
                self.pending_coordinate_row_mass -= removed_coordinates
                self.pending_transition_row_mass -= removed_transitions
                self.evicted_pending_coordinate_row_mass += removed_coordinates
                self.evicted_pending_transition_row_mass += removed_transitions
            if self.needs_initial_measure:
                self.needs_initial_measure = False
                self.measure_event.set()

    def snapshot(self):
        with self.lock:
            return self.revision, {
                **self.report,
                "measurement_computation": {
                    "revision": self.revision,
                    "error_mass": self.errors,
                    "last_error": self.last_error,
                    "pending_coordinate_row_mass": self.pending_coordinate_row_mass,
                    "pending_transition_row_mass": self.pending_transition_row_mass,
                },
            }

    def run(self):
        window = LiteralJWindow(self.max_rows)
        generation = 0
        while True:
            self.measure_event.wait(self.interval)
            self.measure_event.clear()
            with self.lock:
                current_generation = self.generation
                frames = self.pending
                self.pending = deque()
                self.pending_coordinate_row_mass = 0
                self.pending_transition_row_mass = 0
                reporter_window = {
                    "ingested_coordinate_row_mass": self.ingested_coordinate_row_mass,
                    "ingested_transition_row_mass": self.ingested_transition_row_mass,
                    "evicted_pending_coordinate_row_mass": self.evicted_pending_coordinate_row_mass,
                    "evicted_pending_transition_row_mass": self.evicted_pending_transition_row_mass,
                }
            if current_generation != generation:
                window = LiteralJWindow(self.max_rows)
                generation = current_generation
            if not frames:
                if self.stop_event.is_set():
                    break
                continue
            try:
                for frame, _, _ in frames:
                    window.ingest(frame)
                report = window.measure()
                report["observation_window"].update({
                    **reporter_window,
                    "evicted_coordinate_row_mass": (
                        reporter_window["evicted_pending_coordinate_row_mass"]
                        + window.coordinates.evicted_row_mass
                    ),
                    "evicted_transition_row_mass": (
                        reporter_window["evicted_pending_transition_row_mass"]
                        + window.transition_window.evicted_row_mass
                    ),
                })
                published = {"measure_projection": "section_scalars_and_structure_sizes", **scalar_report(report)}
                path = f"{self.artifact_path}.{generation}.npz"
                sampled_at = time.time()
                write_report(path, {"sampled_at": sampled_at, "generation": generation, **report})
                published["artifacts"] = {"j": {"path": path, "content_type": "application/octet-stream", "format": "numpy-npz-tree-v2", "sampled_at": sampled_at, "generation": generation}}
                newest = frames[-1][0]
                model = {"response": newest["resp_id"], "t": newest.get("t"), "row_identity": "edict",
                         "row_outputs": [{"row": source["edict"], "arm": source["policy_arm"], "j": source["j"]}
                                         for source in newest.get("measure_sources", []) if source["response_seq"] == newest["request_seq"]]}
                write_report(path + '.view.npz', {'sampled_at': sampled_at, 'generation': generation,
                    'full_artifact': path, 'artifact_format': 'numpy-npz-tree-v2', 'model': model, **compact_j_report(report)})
                with self.lock:
                    if generation == self.generation:
                        self.report = published
                        self.revision += 1
            except Exception as error:
                with self.lock:
                    self.errors += 1
                    self.last_error = f"{type(error).__name__}: {error}"
                    self.revision += 1


__all__ = [
    "LiteralJReporter", "LiteralJWindow", "literal_j_measures",
    "rows_from_frame", "transition_from_frame", "matrix_fusion_intervention_measures",
    "sum_measures", "empirical_affine_projection", "literal_coordinate_delta",
    "literal_source_coordinates", "literal_state_lens_coordinates",
]
