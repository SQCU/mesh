from __future__ import annotations

import argparse, json
from collections import defaultdict

import numpy as np


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(rows):
    counts = defaultdict(int)
    updates = defaultdict(list)
    behavior = defaultdict(int)
    controllers = defaultdict(int)
    previous = None
    for row in rows:
        for assignment in row.get("assignments", []):
            behavior[assignment.get("behavior", "unknown")] += 1
            controllers[assignment.get("controller", "unknown")] += 1
        update = row.get("update")
        if update:
            for key, value in update.items():
                if isinstance(value, (int, float)):
                    updates[key].append(float(value))
        if previous is not None and (row.get("k"), row.get("j"), row.get("l")) == (
            previous.get("k"), previous.get("j"), previous.get("l")
        ):
            before, after = int(previous.get("PW", 0)), int(row.get("PW", 0))
            if before:
                counts["retention_trials"] += 1
                counts["retained"] += int(before == after)
            teams = {int(item["team"]) for item in previous.get("assignments", [])}
            for team in teams:
                if team != before:
                    counts["acquisition_trials"] += 1
                    counts["acquired"] += int(after == team)
            before_carts = [(cart.get("ctrl"), cart.get("depth")) for cart in previous.get("carts", [])]
            after_carts = [(cart.get("ctrl"), cart.get("depth")) for cart in row.get("carts", [])]
            counts["transitions"] += 1
            counts["cart_changes"] += int(before_carts != after_carts)
        previous = row
    means = {key: float(np.mean(values)) for key, values in updates.items() if values}
    return {
        "ticks": len(rows),
        "transitions": counts["transitions"],
        "cart_changes": counts["cart_changes"],
        "winner_retention": counts["retained"] / max(1, counts["retention_trials"]),
        "winner_retention_trials": counts["retention_trials"],
        "loser_acquisition": counts["acquired"] / max(1, counts["acquisition_trials"]),
        "loser_acquisition_trials": counts["acquisition_trials"],
        "behavior_rows": dict(behavior),
        "controller_rows": dict(controllers),
        "update_means": means,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {path: summarize(read_jsonl(path)) for path in args.telemetry}
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
