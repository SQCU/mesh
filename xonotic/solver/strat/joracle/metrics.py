from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict

import numpy as np

from .follow import split_source

def quotient(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None

def resource_fields(team):
    values = {
        key: float(team[key])
        for key in ("alive", "health", "armor", "mean_speed")
        if isinstance(team.get(key), (int, float))
    }
    for group in ("ammo", "weapon_words"):
        for name, value in (team.get(group) or {}).items():
            if isinstance(value, (int, float)):
                values[f"{group}.{name}"] = float(value)
    return values

def read_jsonl(source, host_key_alias="mesh-mini"):
    host, path = split_source(source)
    if host is None:
        with open(path) as stream:
            lines = list(stream)
    else:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes",
                                 "-o", f"HostName={host}",
                                 "-o", f"HostKeyAlias={host_key_alias}",
                                 "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                                 host_key_alias, "cat", path],
                                capture_output=True, text=True, timeout=30)
        lines = result.stdout.splitlines()
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "resp_id" in value:
            rows.append(value)
    return rows

def summarize(rows):
    counts = defaultdict(int)
    updates = defaultdict(list)
    behavior = defaultdict(int)
    controllers = defaultdict(int)
    resources = defaultdict(list)
    resource_change = defaultdict(list)
    outcomes = defaultdict(float)
    damage_focus = None
    kill_focus = None
    previous = None
    for row in rows:
        for assignment in row.get("assignments") or []:
            behavior[assignment.get("behavior", "unknown")] += 1
            controllers[assignment.get("controller", "unknown")] += 1
            counts["assignments"] += 1
            counts["response_observed"] += int(assignment.get("applied_response_seq", 0) > 0)
            counts["source_matched"] += int(bool(assignment.get("applied_state_current")))
            counts["first_execution"] += int(bool(assignment.get("first_execution")))
            for routed in assignment.get("routed_outcomes") or ():
                counts["routed_outcome_intervals"] += 1
                for key, value in (routed.get("outcomes") or {}).items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        outcomes[key] += float(value)
        k = int(row.get("k") or 0)
        for event in row.get("realized_events") or []:
            kind = event.get("kind")
            counts[f"event_{kind}"] += 1
            actor = int(event.get("actor_team") or 0)
            subject = int(event.get("subject_team") or 0)
            counts["event_value_present"] += int("value" in event)
            source_value = event.get("value")
            numeric = isinstance(source_value, (int, float))
            counts["event_value_numeric"] += int(numeric)
            finite = bool(numeric and np.isfinite(source_value))
            counts["event_value_finite"] += int(finite)
            if kind == "damage" and finite:
                outcomes["realized_damage"] += float(source_value)
            if kind == "damage" and finite and 0 < actor <= k and 0 < subject <= k:
                if damage_focus is None or damage_focus.shape != (k, k):
                    damage_focus = np.zeros((k, k), dtype=np.float64)
                value = float(source_value)
                damage_focus[actor - 1, subject - 1] += value
            if kind == "kill" and 0 < actor <= k and 0 < subject <= k:
                if kill_focus is None or kill_focus.shape != (k, k):
                    kill_focus = np.zeros((k, k), dtype=np.float64)
                kill_focus[actor - 1, subject - 1] += 1
            if kind == "score_win":
                outcomes[f"score_win_team_{actor}"] += 1
            if kind == "tie":
                outcomes["ties"] += 1
        for key, value in (row.get("update") or {}).items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                updates[key].append(float(value))
        for team in row.get("resources") or []:
            for key, value in resource_fields(team).items():
                resources[key].append(value)
        shape = (row.get("k"), row.get("j"), row.get("l"))
        previous_shape = None if previous is None else (previous.get("k"), previous.get("j"), previous.get("l"))
        if previous is not None and shape == previous_shape:
            before, after = int(previous.get("PW", 0)), int(row.get("PW", 0))
            if before:
                counts["retention_trials"] += 1
                counts["retained"] += int(before == after)
            for team in {int(item["team"]) for item in previous.get("assignments") or []}:
                if team != before:
                    counts["acquisition_trials"] += 1
                    counts["acquired"] += int(after == team)
            before_carts = [(cart.get("control_team"), cart.get("depth")) for cart in previous.get("carts") or []]
            after_carts = [(cart.get("control_team"), cart.get("depth")) for cart in row.get("carts") or []]
            counts["transitions"] += 1
            counts["cart_changes"] += int(before_carts != after_carts)
            counts["winner_flips"] += int(before != after)
            counts["hierarchy_flips"] += int(previous.get("loser_ranks") != row.get("loser_ranks"))
            old_resources = {int(item["team"]): item for item in previous.get("resources") or []}
            for team in row.get("resources") or []:
                old = old_resources.get(int(team["team"]))
                if old:
                    before_values = resource_fields(old)
                    after_values = resource_fields(team)
                    for key in before_values.keys() & after_values.keys():
                        resource_change[key].append(abs(after_values[key] - before_values[key]))
        previous = row
    return {
        "ticks": len(rows), "transitions": counts["transitions"],
        "cart_changes": counts["cart_changes"], "winner_flips": counts["winner_flips"],
        "hierarchy_flips": counts["hierarchy_flips"],
        "winner_retention": quotient(counts["retained"], counts["retention_trials"]),
        "winner_retention_trials": counts["retention_trials"],
        "loser_acquisition": quotient(counts["acquired"], counts["acquisition_trials"]),
        "loser_acquisition_trials": counts["acquisition_trials"],
        "behavior_rows": dict(behavior), "controller_rows": dict(controllers),
        "application": {
            "assignments": counts["assignments"],
            "response_observed": counts["response_observed"],
            "source_matched": counts["source_matched"],
            "first_executions": counts["first_execution"],
        },
        "realized": {
            "outcome_deltas": dict(outcomes),
            "damage_events": counts["event_damage"],
            "kill_events": counts["event_kill"],
            "pickup_events": counts["event_pickup"],
            "event_value_present_mass": counts["event_value_present"],
            "event_value_numeric_mass": counts["event_value_numeric"],
            "event_value_finite_mass": counts["event_value_finite"],
            "damage_focus": [] if damage_focus is None else damage_focus.tolist(),
            "kill_focus": [] if kill_focus is None else kill_focus.tolist(),
        },
        "resource_means_per_team_tick": {key: float(np.mean(value)) for key, value in resources.items() if value},
        "resource_mean_absolute_change": {key: float(np.mean(value)) for key, value in resource_change.items() if value},
        "update_means": {key: float(np.mean(value)) for key, value in updates.items() if value},
    }

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--host-key-alias", default="mesh-mini")
    args = parser.parse_args(argv)
    result = {source: summarize(read_jsonl(source, args.host_key_alias)) for source in args.telemetry}
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(encoded + "\n")
    print(encoded)

if __name__ == "__main__":
    main()
