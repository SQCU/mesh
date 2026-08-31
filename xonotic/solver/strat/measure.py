"""Measurements over REAL run artifacts.

Every number this module produces comes from a real Xonotic server's own log or
from a real responder telemetry file. Nothing here simulates the game (SPEC 13),
and there are no tests -- these are measurements.

Three subcommands:

  rows      Parse a server log's `[PLCOBS]` / `[PLCCART]` / `[PLCEVT]` lines --
            written by `payload_strategy_log` in sv_payload_strategy_io.qc, read
            back off the very fields `mesh_gather` sweeps -- into one JSONL
            record per strategy tick, with the per-player observation rows, the
            instrument descriptors `z` and the relation rows reconstructed for
            it. AGENDA E9/E10: the j-space probe needs `z` and the relation rows
            and neither was ever logged.

  cgt       Resolve rate of the closed-form cart-subgame evaluator over a real
            telemetry file's `game_value.state` positions. AGENDA B11.

  commit    Distribution of the travel-commitment horizon actually written into
            the COMPLETE column, over a real run. AGENDA F4.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "payload", "tools"))

from strategy_io_schema import CS, EVT, OBS, SC  # noqa: E402

# Column order the QC logger emits, verbatim from payload_strategy_log().
OBS_LOG_COLUMNS = ("ID", "TEAM", "HEALTH", "ARMOR", "AMMO", "POS_X", "POS_Y", "POS_Z",
                   "VEL_X", "VEL_Y", "VEL_Z", "WEAPONS", "POWER", "TSS", "CELL",
                   "NCART", "NCART_D", "ALIVE", "CONTROL")
CART_LOG_COLUMNS = ("ID", "DEPTH", "LENGTH", "CTRL", "SPEED", "IDLE", "BANKMASK",
                    "PROGRESS", "POS_X", "POS_Y", "POS_Z")
EVT_LOG_COLUMNS = ("CELL", "KIND", "TEAM", "SUBJECT", "VALUE", "TIME")
L_LEVELS = 8


def _floats(parts):
    return [float(value) for value in parts]


def parse_server_log(path):
    """Group a server log's staged rows by publish sequence.

    Yields dicts {seq, obs: ndarray(l, OBS_WIDTH), cart: ndarray(j, CART_WIDTH),
    evt: ndarray(n, EVT_WIDTH), edicts: [...]} in sequence order.
    """
    from strategy_io_schema import CART_WIDTH, EVT_WIDTH, OBS_WIDTH

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
            if tag == "[PLCOBS]" and len(values) >= len(OBS_LOG_COLUMNS):
                row = np.zeros(OBS_WIDTH, dtype=np.float32)
                for name, value in zip(OBS_LOG_COLUMNS, values):
                    row[OBS[name]] = value
                tick["obs"].append(row)
                tick["edicts"].append(key)
            elif tag == "[PLCCART]" and len(values) >= 8:
                row = np.zeros(CART_WIDTH, dtype=np.float32)
                for name, value in zip(CART_LOG_COLUMNS, values):
                    row[CS[name]] = value
                tick["cart"].append((key, row))
            elif tag == "[PLCEVT]" and len(values) >= len(EVT_LOG_COLUMNS):
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
    """Rebuild the instrument batch the operator would see for one tick.

    Uses the same constructors the live responder uses, so `z` and the relation
    rows written here are the ones the model consumes, not a paraphrase.
    """
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
            int(round(rows[p, OBS["CELL"]])),
            tuple(float(v) for v in rows[p, OBS["POS_X"]:OBS["POS_Z"] + 1]),
            float(rows[p, OBS["ALIVE"]]),
            float(rows[p, OBS["HEALTH"]]) / 100.0,
            float(rows[p, OBS["ARMOR"]]) / 100.0,
            float(rows[p, OBS["AMMO"]]),
            float(rows[p, OBS["TSS"]]),
        )
        for p in range(len(rows))
    ]
    carts = [
        CartTarget(
            int(round(row[CS["ID"]])), int(round(row[CS["CTRL"]])),
            float(row[CS["DEPTH"]]), float(row[CS["SPEED"]]), float(row[CS["PROGRESS"]]),
            (float(row[CS["POS_X"]]), float(row[CS["POS_Y"]]), float(row[CS["POS_Z"]])),
        )
        for row in tick["cart"]
    ]
    if belief is None:
        belief = LiveBelief()
    if len(tick["evt"]):
        belief.ingest(tick["evt"], EVT)
    # `beliefs` is what populates the per-cell positions the item/rival/cell
    # targets are placed at; skipping it leaves every non-cart target at the
    # world origin.
    belief.beliefs(rows, OBS)
    items, rivals, cells = belief.instrument_targets(rows, OBS)
    return build_instruments(participants, carts, items, rivals, cells), rows


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
                "obs": np.round(rows[:, [OBS[name] for name in OBS_LOG_COLUMNS]], 6).tolist(),
                "cart_columns": list(CART_LOG_COLUMNS),
                "cart": np.round(tick["cart"][:, [CS[name] for name in CART_LOG_COLUMNS]], 6).tolist(),
                "evt": np.round(tick["evt"], 6).tolist(),
                "instrument_kinds": [inst.kind.value for inst in batch.instruments],
                "instrument_subjects": [int(inst.subject) for inst in batch.instruments],
                "z": np.round(batch.descriptors, 6).tolist(),
                "relation": np.round(batch.relations, 6).tolist(),
                "eligible": batch.eligible.astype(int).tolist(),
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
    from solver.strat.game_value import EmpiricalTransitionGraph, evaluate_cartstate

    logged = collections.Counter()
    closed = collections.Counter()
    nimbers = collections.Counter()
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
            logged[(value.get("kind"), value.get("reason"))] += 1
            _, _, k, depths, controls = value["state"]
            result = evaluate_cartstate(depths, controls, list(range(int(k))), args.levels)
            closed[result.kind] += 1
            if result.kind == "impartial":
                nimbers[result.nimber] += 1
    del EmpiricalTransitionGraph
    print(json.dumps({
        "telemetry": os.path.abspath(args.telemetry),
        "lines_with_a_cart_game_value": total,
        "as_logged": {f"{kind}:{reason}": count for (kind, reason), count in logged.items()},
        "closed_form": dict(closed),
        "closed_form_nimbers": {str(key): value for key, value in nimbers.items()},
        "resolved_before": total - logged.get(("unresolved", "incomplete option graph"), 0),
        "resolved_after": sum(count for kind, count in closed.items() if kind != "unresolved"),
    }, indent=2))


def cmd_commit(args):
    from solver.strat.instruments import decode_allocations

    from solver.strat.live_belief import LiveBelief

    ticks, _ = parse_server_log(args.log)
    belief = LiveBelief()
    values = []
    by_kind = collections.defaultdict(list)
    rng = np.random.default_rng(args.seed)
    for tick in ticks:
        if not len(tick["obs"]) or not len(tick["cart"]):
            continue
        batch, rows = _batch_for(tick, belief)
        if batch is None:
            continue
        count = len(batch.participants)
        actions = np.array([rng.choice(np.flatnonzero(batch.eligible[p])) for p in range(count)])
        out = decode_allocations(batch, actions, intensity=np.ones(count),
                                 commitments=np.ones(count), spawn_delays=np.ones(count))
        for p, action in enumerate(actions):
            value = float(out[p, SC["COMMIT"]])
            values.append(value)
            by_kind[batch.instruments[int(action)].kind.value].append(value)
    array = np.asarray(values, dtype=np.float64)
    print(json.dumps({
        "log": os.path.abspath(args.log),
        "assignments": int(array.size),
        "commit_nonzero": int(np.count_nonzero(array)),
        "commit_nonzero_fraction": round(float(np.count_nonzero(array) / max(1, array.size)), 6),
        "seconds": {
            "min": round(float(array.min()), 4) if array.size else None,
            "median": round(float(np.median(array)), 4) if array.size else None,
            "mean": round(float(array.mean()), 4) if array.size else None,
            "max": round(float(array.max()), 4) if array.size else None,
        },
        "by_instrument_kind": {
            kind: {"n": len(items), "nonzero": int(np.count_nonzero(items)),
                   "median_seconds": round(float(np.median(items)), 4)}
            for kind, items in sorted(by_kind.items())
        },
    }, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    rows = sub.add_parser("rows", help="server log -> observation/z/relation JSONL")
    rows.add_argument("log")
    rows.add_argument("--out", required=True)
    rows.set_defaults(func=cmd_rows)

    cgt = sub.add_parser("cgt", help="cart-subgame resolve rate over real telemetry")
    cgt.add_argument("telemetry")
    cgt.add_argument("--levels", type=int, default=L_LEVELS)
    cgt.set_defaults(func=cmd_cgt)

    commit = sub.add_parser("commit", help="travel-commitment horizon over a real run")
    commit.add_argument("log")
    commit.add_argument("--seed", type=int, default=20260831)
    commit.set_defaults(func=cmd_commit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
