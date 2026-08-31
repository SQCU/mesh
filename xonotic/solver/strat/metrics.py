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
    resources = defaultdict(list)
    resource_change = defaultdict(list)
    focus = None
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
        for team in row.get("resources", []):
            for key in ("alive", "health", "armor", "ammo", "weapon_slots", "mean_speed", "power"):
                resources[key].append(float(team.get(key, 0)))
        current_focus = np.asarray(row.get("strategy_focus", []), dtype=np.int64)
        if current_focus.ndim == 2 and current_focus.size:
            if focus is None or focus.shape != current_focus.shape:
                focus = current_focus.copy()
            else:
                focus += current_focus
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
            counts["winner_flips"] += int(before != after)
            old_resources = {int(item["team"]): item for item in previous.get("resources", [])}
            for team in row.get("resources", []):
                old = old_resources.get(int(team["team"]))
                if old:
                    for key in ("alive", "health", "armor", "ammo", "weapon_slots", "power"):
                        resource_change[key].append(abs(float(team.get(key, 0)) - float(old.get(key, 0))))
        previous = row
    means = {key: float(np.mean(values)) for key, values in updates.items() if values}
    return {
        "ticks": len(rows),
        "transitions": counts["transitions"],
        "cart_changes": counts["cart_changes"],
        "winner_flips": counts["winner_flips"],
        "winner_retention": counts["retained"] / max(1, counts["retention_trials"]),
        "winner_retention_trials": counts["retention_trials"],
        "loser_acquisition": counts["acquired"] / max(1, counts["acquisition_trials"]),
        "loser_acquisition_trials": counts["acquisition_trials"],
        "behavior_rows": dict(behavior),
        "controller_rows": dict(controllers),
        "resource_means_per_team_tick": {key: float(np.mean(values)) for key, values in resources.items() if values},
        "resource_mean_absolute_change": {key: float(np.mean(values)) for key, values in resource_change.items() if values},
        "strategy_focus": focus.tolist() if focus is not None else [],
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
