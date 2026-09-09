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

from payload.tools.strategy_io_schema import CS, EVT, OBS, TS

OBS_LOG_COLUMNS = tuple(OBS)
CART_LOG_COLUMNS = tuple(CS)
EVT_LOG_COLUMNS = tuple(EVT)
TEAM_LOG_COLUMNS = tuple(TS)

def _floats(parts):
    return [float(value) for value in parts]

def parse_server_log(path):
    from payload.tools.strategy_io_schema import CART_WIDTH, EVT_WIDTH, OBS_WIDTH, TEAM_WIDTH

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
            if tag not in ("[PLCOBS]", "[PLCCART]", "[PLCTEAM]", "[PLCEVT]", "[PLCPUB]"):
                continue
            if tag == "[PLCPUB]":
                continue
            try:
                seq = int(float(parts[1]))
                key = int(float(parts[2]))
                values = _floats(parts[2:] if tag == "[PLCTEAM]" else parts[3:])
            except ValueError:
                continue
            tick = ticks.setdefault(seq, {"seq": seq, "obs": [], "cart": [], "team": [], "evt": [], "edicts": []})
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
            elif tag == "[PLCTEAM]" and len(values) == len(TEAM_LOG_COLUMNS):
                tick["team"].append((key, np.asarray(values, dtype=np.float32)))
            elif tag == "[PLCEVT]" and len(values) == len(EVT_LOG_COLUMNS):
                row = np.zeros(EVT_WIDTH, dtype=np.float32)
                for name, value in zip(EVT_LOG_COLUMNS, values):
                    row[EVT[name]] = value
                tick["evt"].append(row)
    out = []
    for seq, tick in ticks.items():
        carts = [row for _, row in sorted(tick["cart"], key=lambda item: item[0])]
        teams = [row for _, row in sorted(tick["team"], key=lambda item: item[0])]
        out.append({
            "seq": seq,
            "obs": np.stack(tick["obs"]) if tick["obs"] else np.zeros((0, OBS_WIDTH), dtype=np.float32),
            "cart": np.stack(carts) if carts else np.zeros((0, CART_WIDTH), dtype=np.float32),
            "team": np.stack(teams) if teams else np.zeros((0, TEAM_WIDTH), dtype=np.float32),
            "evt": np.stack(tick["evt"]) if tick["evt"] else np.zeros((0, EVT_WIDTH), dtype=np.float32),
            "edicts": tick["edicts"],
        })
    return out, (pool or [])

def cmd_rows(args):
    ticks, pool = parse_server_log(args.log)
    written = 0
    nonzero = collections.Counter()
    with open(args.out, "w") as handle:
        for tick in ticks:
            if not len(tick["obs"]) or not len(tick["cart"]):
                continue
            rows = tick["obs"]
            for name in OBS_LOG_COLUMNS:
                if np.any(np.abs(rows[:, OBS[name]]) > 0):
                    nonzero[name] += 1
            handle.write(json.dumps({
                "seq": tick["seq"],
                "obs_columns": list(OBS_LOG_COLUMNS),
                "obs": rows[:, [OBS[name] for name in OBS_LOG_COLUMNS]].tolist(),
                "cart_columns": list(CART_LOG_COLUMNS),
                "cart": tick["cart"][:, [CS[name] for name in CART_LOG_COLUMNS]].tolist(),
                "team_columns": list(TEAM_LOG_COLUMNS),
                "team": tick["team"].tolist(),
                "evt": tick["evt"].tolist(),
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
    from solver.strat.game_value import GAME_CONTRACT
    from solver.strat.game_value import cart_snapshot_record, formal_game_value, formal_projection_record, formal_value_record, GameContext

    residuals = collections.Counter()
    compared = collections.Counter()
    obsolete = total = 0
    previous = None
    with open(args.telemetry) as handle:
        for raw in handle:
            line = json.loads(raw)
            value = line.get("game_value")
            if not value:
                continue
            if value.get("contract") != GAME_CONTRACT:
                obsolete += 1
                continue
            snapshot = cart_snapshot_record(value["state"])
            teams = tuple(range(len(snapshot.scores)))
            result = formal_game_value(GameContext(teams, ()), snapshot)
            expected = formal_value_record(result, snapshot)
            for name, target in expected.items():
                compared[name] += 1
                residuals[name] += int(value.get(name) != target)
            for name, target in formal_projection_record(result, teams).items():
                compared[name] += 1
                residuals[name] += int(line.get(name) != target)
            residuals["checkpoint_accounting"] += sum(abs(int(v)) for v in expected["held_checkpoint_residual"])
            if previous is not None and previous.episode == snapshot.episode:
                residuals["score_decrease"] += int(np.sum(snapshot.scores < previous.scores))
                compared["score_decrease"] += len(teams)
            previous = snapshot
            total += 1
    print(json.dumps({
        "telemetry": os.path.abspath(args.telemetry),
        "game_contract": GAME_CONTRACT,
        "lines_with_a_cart_game_value": total,
        "obsolete_game_records": obsolete,
        "semantic_coordinate_measure": dict(compared),
        "semantic_residual_measure": dict(residuals),
        "semantic_residual_mass": sum(residuals.values()),
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
    from solver.strat.paged_matrix import PAGE_ROWS, TILE
    from solver.strat.matmul import (
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
                "forward": f"paged_{PAGE_ROWS}x{TILE}",
                "reverse_left": f"paged_{PAGE_ROWS}x{TILE}",
                "reverse_right": f"paged_{PAGE_ROWS}x{TILE}",
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
