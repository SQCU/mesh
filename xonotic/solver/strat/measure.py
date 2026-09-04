from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "payload", "tools"))

from payload.tools.strategy_io_schema import CS, EVT, OBS

OBS_LOG_COLUMNS = tuple(OBS)
CART_LOG_COLUMNS = tuple(CS)
EVT_LOG_COLUMNS = tuple(EVT)
L_LEVELS = 8

def _floats(parts):
    return [float(value) for value in parts]

def parse_server_log(path):
    from payload.tools.strategy_io_schema import CART_WIDTH, EVT_WIDTH, OBS_WIDTH

    ticks = collections.OrderedDict()
    pool = None
    with open(path, errors="replace") as handle:
        for line in handle:
            if "[PLCPOOL]" in line:
                pool = (pool or []) + [line.strip()]
                continue
            index = line.find("[PLC")
            if index < 0:
                continue
            parts = line[index:].split()
            tag = parts[0]
            if tag not in ("[PLCOBS]", "[PLCCART]", "[PLCEVT]", "[PLCPUB]"):
                continue
            if tag == "[PLCPUB]":
                continue
            try:
                seq = int(float(parts[1]))
                key = int(float(parts[2]))
                values = _floats(parts[3:])
            except ValueError:
                continue
            tick = ticks.setdefault(seq, {"seq": seq, "obs": [], "cart": [], "evt": [], "edicts": []})
            if tag == "[PLCOBS]" and len(values) == len(OBS_LOG_COLUMNS):
                row = np.zeros(OBS_WIDTH, dtype=np.float32)
                for name, value in zip(OBS_LOG_COLUMNS, values):
                    row[OBS[name]] = value
                tick["obs"].append(row)
                tick["edicts"].append(key)
            elif tag == "[PLCCART]" and len(values) == len(CART_LOG_COLUMNS):
                row = np.zeros(CART_WIDTH, dtype=np.float32)
                for name, value in zip(CART_LOG_COLUMNS, values):
                    row[CS[name]] = value
                tick["cart"].append((key, row))
            elif tag == "[PLCEVT]" and len(values) == len(EVT_LOG_COLUMNS):
                row = np.zeros(EVT_WIDTH, dtype=np.float32)
                for name, value in zip(EVT_LOG_COLUMNS, values):
                    row[EVT[name]] = value
                tick["evt"].append(row)
    out = []
    for seq, tick in ticks.items():
        carts = [row for _, row in sorted(tick["cart"], key=lambda item: item[0])]
        out.append({
            "seq": seq,
            "obs": np.stack(tick["obs"]) if tick["obs"] else np.zeros((0, OBS_WIDTH), dtype=np.float32),
            "cart": np.stack(carts) if carts else np.zeros((0, CART_WIDTH), dtype=np.float32),
            "evt": np.stack(tick["evt"]) if tick["evt"] else np.zeros((0, EVT_WIDTH), dtype=np.float32),
            "edicts": tick["edicts"],
        })
    return out, (pool or [])

def _batch_for(tick, belief=None):
    from solver.strat.instruments import CartTarget, Participant, build_instruments
    from solver.strat.live_belief import LiveBelief

    rows = tick["obs"]
    active = np.flatnonzero(np.asarray(rows[:, OBS["TEAM"]], dtype=np.int64) >= 1)
    rows = rows[active]
    if not len(rows):
        return None, rows
    participants = [
        Participant(
            int(rows[p, OBS["ID"]]), int(rows[p, OBS["TEAM"]]),
            (int(rows[p, OBS["CELL_X"]]), int(rows[p, OBS["CELL_Y"]])),
            tuple(float(v) for v in rows[p, OBS["POS_X"]:OBS["POS_Z"] + 1]),
            float(rows[p, OBS["ALIVE"]]),
            float(rows[p, OBS["HEALTH"]]), float(rows[p, OBS["ARMOR"]]),
            tuple(float(rows[p, OBS[name]]) for name in (
                "AMMO_SHELLS", "AMMO_BULLETS", "AMMO_ROCKETS",
                "AMMO_CELLS", "AMMO_PLASMA", "AMMO_FUEL",
            )),
            float(rows[p, OBS["SPAWN_TIME"]]), float(rows[p, OBS["ENGINE_TIME"]]),
        )
        for p in range(len(rows))
    ]
    carts = [
        CartTarget(
            int(row[CS["ID"]]), int(row[CS["CONTROL_TEAM"]]),
            float(row[CS["PATH_POSITION"]]), float(row[CS["PATH_LENGTH"]]),
            float(row[CS["SPEED"]]),
            (float(row[CS["POS_X"]]), float(row[CS["POS_Y"]]), float(row[CS["POS_Z"]])),
        )
        for row in tick["cart"]
    ]
    if belief is None:
        belief = LiveBelief()
    if len(tick["evt"]):
        belief.ingest(tick["evt"], EVT)
    belief.chorus(rows, OBS)
    items, rivals, cells = belief.instrument_targets(rows, OBS)
    return build_instruments(
        participants, carts, items, rivals, cells,
        navigation=belief.navigation_vcmap,
    ), rows

def cmd_rows(args):
    from solver.strat.live_belief import LiveBelief

    ticks, pool = parse_server_log(args.log)
    belief = LiveBelief()
    written = 0
    nonzero = collections.Counter()
    with open(args.out, "w") as handle:
        for tick in ticks:
            if not len(tick["obs"]) or not len(tick["cart"]):
                continue
            batch, rows = _batch_for(tick, belief)
            if batch is None:
                continue
            for name in OBS_LOG_COLUMNS:
                if np.any(np.abs(rows[:, OBS[name]]) > 0):
                    nonzero[name] += 1
            handle.write(json.dumps({
                "seq": tick["seq"],
                "obs_columns": list(OBS_LOG_COLUMNS),
                "obs": rows[:, [OBS[name] for name in OBS_LOG_COLUMNS]].tolist(),
                "cart_columns": list(CART_LOG_COLUMNS),
                "cart": tick["cart"][:, [CS[name] for name in CART_LOG_COLUMNS]].tolist(),
                "evt": tick["evt"].tolist(),
                "instrument_kinds": [inst.kind.value for inst in batch.instruments],
                "instrument_subjects": [inst.subject for inst in batch.instruments],
                "z": batch.descriptors.tolist(),
                "action_mass": batch.action_mass.tolist(),
            }) + "\n")
            written += 1
    summary = {
        "log": os.path.abspath(args.log),
        "out": os.path.abspath(args.out),
        "pool": pool,
        "ticks_logged": len(ticks),
        "ticks_written": written,
        "player_rows": int(sum(len(tick["obs"]) for tick in ticks)),
        "ticks_with_a_nonzero_value_per_observation_column": dict(sorted(nonzero.items())),
    }
    print(json.dumps(summary, indent=2))

def cmd_cgt(args):
    from solver.strat.game_value import evaluate_cartstate
    from solver.strat.runtime import formal_projection_record, formal_value_record

    source_measures = collections.Counter()
    closed_measures = collections.Counter()
    nimbers = collections.Counter()
    residuals = collections.Counter()
    missing = collections.Counter()
    compared = collections.Counter()
    total = 0
    with open(args.telemetry) as handle:
        for raw in handle:
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue
            value = line.get("game_value")
            if not value:
                continue
            total += 1
            for name in (
                "reachable_state_mass", "reachable_role_state_mass",
                "enumerated_role_state_mass",
                "role_option_symmetric_difference_mass", "cycle_state_mass",
            ):
                if isinstance(value.get(name), (int, float)):
                    source_measures[name] += value[name]
            _, _, k, depths, controls = value["state"]
            levels = value.get("levels", args.levels if args.levels is not None else L_LEVELS)
            result = evaluate_cartstate(depths, controls, list(range(int(k))), int(levels))
            closed_measures["reachable_state_mass"] += result.reachable_state_mass
            closed_measures["reachable_role_state_mass"] += result.reachable_role_state_mass
            closed_measures["enumerated_role_state_mass"] += result.enumerated_role_state_mass
            closed_measures["role_option_symmetric_difference_mass"] += result.role_option_symmetric_difference_mass
            closed_measures["cycle_state_mass"] += result.cycle_state_mass
            if result.nimber is not None:
                nimbers[result.nimber] += 1
            expected = formal_value_record(result, value["state"], levels)
            expected.pop("state")
            expected.pop("levels")
            for name, target in expected.items():
                if name not in value:
                    missing[name] += 1
                else:
                    compared[name] += 1
                    residuals[name] += int(value[name] != target)
            wire = formal_projection_record(result, range(int(k)))
            for name, target in wire.items():
                if name not in line:
                    missing[name] += 1
                else:
                    compared[name] += 1
                    residuals[name] += int(line[name] != target)
            if "levels" not in value:
                missing["levels"] += 1
            else:
                compared["levels"] += 1
                residuals["levels"] += int(int(value["levels"]) != int(levels))
    print(json.dumps({
        "telemetry": os.path.abspath(args.telemetry),
        "lines_with_a_cart_game_value": total,
        "source_measure_integrals": dict(source_measures),
        "closed_form_measure_integrals": dict(closed_measures),
        "closed_form_nimber_measure": {str(key): value for key, value in nimbers.items()},
        "closed_form_nimber_atom_mass": sum(nimbers.values()),
        "semantic_coordinate_measure": dict(compared),
        "semantic_residual_measure": dict(residuals),
        "semantic_residual_mass": sum(residuals.values()),
        "missing_semantic_coordinate_measure": dict(missing),
        "missing_semantic_coordinate_mass": sum(missing.values()),
    }, indent=2))

def _tensor_measure(observed, reference):
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = observed - reference
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).tiny)
    return {
        "shape": list(observed.shape),
        "coordinate_mass": int(observed.size),
        "finite_coordinate_mass": int(np.isfinite(observed).sum()),
        "reference_finite_coordinate_mass": int(np.isfinite(reference).sum()),
        "maximum_absolute_residual": float(np.max(np.abs(difference))),
        "maximum_relative_residual": float(np.max(np.abs(difference)) / scale),
        "residual_square_integral": float(np.sum(difference * difference)),
    }

def _timed(operation, samples):
    import mlx.core as mx

    warm = operation()
    mx.eval(warm)
    durations = []
    value = warm
    for _ in range(samples):
        started = time.perf_counter()
        value = operation()
        mx.eval(value)
        durations.append(time.perf_counter() - started)
    return value, durations

def cmd_matrix(args):
    import mlx.core as mx

    from solver.strat.dpp import dpp_marginals
    from solver.strat.matmul import (
        matrix_execution_schedule,
        matrix_multiply,
        matrix_multiply_transpose_left,
        matrix_multiply_transpose_right,
    )

    rng = np.random.default_rng(args.seed)
    rows, inner, columns = args.rows, args.inner, args.columns
    arrays = {
        "ab": (
            rng.standard_normal((rows, inner), dtype=np.float32),
            rng.standard_normal((inner, columns), dtype=np.float32),
            matrix_multiply,
            lambda left, right: np.matmul(left.astype(np.float64), right.astype(np.float64)),
        ),
        "atb": (
            rng.standard_normal((inner, rows), dtype=np.float32),
            rng.standard_normal((inner, columns), dtype=np.float32),
            matrix_multiply_transpose_left,
            lambda left, right: np.matmul(left.astype(np.float64).T, right.astype(np.float64)),
        ),
        "abt": (
            rng.standard_normal((rows, inner), dtype=np.float32),
            rng.standard_normal((columns, inner), dtype=np.float32),
            matrix_multiply_transpose_right,
            lambda left, right: np.matmul(left.astype(np.float64), right.astype(np.float64).T),
        ),
    }
    products = {}
    for name, (left, right, operation, reference_operation) in arrays.items():
        observed, elapsed = _timed(
            lambda left=left, right=right, operation=operation: operation(
                mx.array(left), mx.array(right),
            ),
            args.samples,
        )
        reference = reference_operation(left, right)
        cotangent = rng.standard_normal(reference.shape, dtype=np.float32)
        _, reverse = mx.vjp(
            operation,
            (mx.array(left), mx.array(right)),
            (mx.array(cotangent),),
        )
        mx.eval(*reverse)
        if name == "ab":
            reverse_reference = (
                np.matmul(cotangent.astype(np.float64), right.astype(np.float64).T),
                np.matmul(left.astype(np.float64).T, cotangent.astype(np.float64)),
            )
        elif name == "atb":
            reverse_reference = (
                np.matmul(right.astype(np.float64), cotangent.astype(np.float64).T),
                np.matmul(left.astype(np.float64), cotangent.astype(np.float64)),
            )
        else:
            reverse_reference = (
                np.matmul(cotangent.astype(np.float64), right.astype(np.float64)),
                np.matmul(cotangent.astype(np.float64).T, left.astype(np.float64)),
            )
        products[name] = {
            "forward": _tensor_measure(observed, reference),
            "reverse_left": _tensor_measure(reverse[0], reverse_reference[0]),
            "reverse_right": _tensor_measure(reverse[1], reverse_reference[1]),
            "elapsed_s": {
                "mass": len(elapsed),
                "minimum": min(elapsed),
                "maximum": max(elapsed),
                "mean": float(np.mean(elapsed)),
                "variance": float(np.var(elapsed)),
            },
            "flops_per_sample": int(2 * rows * inner * columns),
            "execution": {
                "forward": matrix_execution_schedule(*reference.shape),
                "reverse_left": matrix_execution_schedule(*reverse_reference[0].shape),
                "reverse_right": matrix_execution_schedule(*reverse_reference[1].shape),
            },
        }
    quality = np.log1p(np.exp(rng.standard_normal(rows, dtype=np.float32))).astype(np.float32)
    features = rng.standard_normal((rows, inner), dtype=np.float32)
    dpp_observed, dpp_elapsed = _timed(
        lambda: dpp_marginals(mx.array(quality), mx.array(features)),
        args.samples,
    )
    normalized = features.astype(np.float64)
    normalized /= np.sqrt(np.mean(normalized * normalized, axis=1, keepdims=True) + 1e-12)
    weighted = quality.astype(np.float64)[:, None] * normalized
    covariance = np.eye(inner, dtype=np.float64) + np.matmul(weighted.T, weighted)
    dpp_reference = np.sum(
        weighted * np.linalg.solve(covariance, weighted.T).T,
        axis=1,
    ).clip(0, 1)
    print(json.dumps({
        "definition": "owned relaxed-FP32 matrix boundary and DPP compared with an FP64 dense reference",
        "rows": rows,
        "inner": inner,
        "columns": columns,
        "samples": args.samples,
        "products": products,
        "dpp": {
            **_tensor_measure(dpp_observed, dpp_reference),
            "elapsed_s": {
                "mass": len(dpp_elapsed),
                "minimum": min(dpp_elapsed),
                "maximum": max(dpp_elapsed),
                "mean": float(np.mean(dpp_elapsed)),
                "variance": float(np.var(dpp_elapsed)),
            },
            "feature_dimension_iterations": inner,
        },
    }, indent=2))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    rows = sub.add_parser("rows", help="server log -> observation/z/action-measure JSONL")
    rows.add_argument("log")
    rows.add_argument("--out", required=True)
    rows.set_defaults(func=cmd_rows)

    cgt = sub.add_parser("cgt", help="cart-subgame resolve rate over real telemetry")
    cgt.add_argument("telemetry")
    cgt.add_argument("--levels", type=int)
    cgt.set_defaults(func=cmd_cgt)

    matrix = sub.add_parser("matrix", help="owned matrix and DPP numerical measures")
    matrix.add_argument("--rows", type=int, required=True)
    matrix.add_argument("--inner", type=int, required=True)
    matrix.add_argument("--columns", type=int, required=True)
    matrix.add_argument("--samples", type=int, required=True)
    matrix.add_argument("--seed", type=int, default=20260903)
    matrix.set_defaults(func=cmd_matrix)

    args = ap.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    main()
