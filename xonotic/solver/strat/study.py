from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from .joracle.probe import matrix_fusion_intervention_measures
from .runtime import BEHAVIOR_MEASURE_NAMES
from .policy_contract import STUDY_ARMS

ARMS = STUDY_ARMS
PRESSURE_KINDS = ("hunt_rival", "suppress_cart", "contest_post")
ATTACK_KINDS = PRESSURE_KINDS + ("push_cart",)
CART_ACTION_KINDS = ("push_cart", "suppress_cart")
OBJECTIVE_KINDS = CART_ACTION_KINDS + ("idle",)
REALIZATION_FIELDS = (
    "mode", "source_weight_mass", "live_weight_mass", "loaded_weight_mass", "composable_weight_mass",
    "source_only_weight_mass", "live_only_weight_mass", "shape_difference_mass",
    "nonfinite_weight_mass", "load_exception", "source_arm", "live_arm",
    "source_version", "live_version", "source_architecture", "live_architecture",
    "source_reward_contract", "live_reward_contract",
    "checkpoint_sha256", "lineage_initial_sha256",
)

def quotient(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None

def observed_quotient(numerator, denominator, observation_mass):
    return quotient(numerator, denominator) if observation_mass else None

def integer_coordinate(*values):
    for value in values:
        if isinstance(value, (int, float)) and np.isfinite(value):
            return int(value)
    return None

def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out

def telemetry_path(record):
    value = ((record.get("artifacts") or {}).get("telemetry") or {}).get("path")
    if value:
        return value
    source = record.get("commands", {}).get("responder", [])
    try:
        return source[source.index("--telemetry") + 1]
    except (ValueError, IndexError):
        return ""

def strategy_measure_records(execution):
    profile = (execution or {}).get("operating_profile") or {}
    out = []
    for node, producers in (profile.get("producer_measure_records") or {}).items():
        for producer in producers:
            measures = producer.get("measures") or {}
            if "j_lens" in measures or "j_oracle" in measures:
                out.append({
                    "node": node,
                    "name": producer.get("name"),
                    "pid": producer.get("pid"),
                    "sampled_at": producer.get("sampled_at"),
                    "labels": producer.get("labels"),
                    "operations": producer.get("operations"),
                    "observation_window": measures.get("observation_window"),
                    "j_lens": measures.get("j_lens"),
                    "j_oracle": measures.get("j_oracle"),
                })
    return out

def round_results(rows):
    seen = set()
    out = []
    for row in rows:
        for event in row.get("realized_events") or []:
            if event.get("kind") not in ("capture", "tie"):
                continue
            actor = integer_coordinate(event.get("actor_team"))
            event_time = event.get("time")
            event_time = float(event_time) if isinstance(event_time, (int, float)) and np.isfinite(event_time) else None
            key = (event.get("kind"), actor, event_time)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out

def arm_metrics(record, rows):
    cfg = record.get("configuration") or {}
    team_arms = cfg.get("team_policy_arms") or []
    fallback = cfg.get("policy_arm", "unknown")
    players = defaultdict(set)
    values = defaultdict(lambda: defaultdict(float))
    value_mass = defaultdict(lambda: defaultdict(int))
    kinds = defaultdict(lambda: defaultdict(int))
    causal = defaultdict(lambda: defaultdict(float))
    exposure = defaultdict(float)
    selected_exposure = defaultdict(float)
    routed_mass = defaultdict(lambda: defaultdict(int))
    event_mass = defaultdict(lambda: defaultdict(int))
    unattributed_player_seconds = 0.0
    previous_engine_time = None
    engine_time_frame_mass = 0
    engine_time_coordinate_mass = 0
    engine_time_finite_coordinate_mass = 0
    engine_time_nonmonotone_frame_mass = 0
    routed_outcome_row_mass = 0
    routed_outcome_attributed_row_mass = 0
    routed_outcome_coordinate_mass = 0
    routed_outcome_finite_coordinate_mass = 0
    event_row_mass = 0
    event_attributed_row_mass = 0
    event_numeric_value_mass = 0
    event_finite_value_mass = 0
    spawn_events = {}
    for row in rows:
        carts = row.get("carts") or []
        row_arms = set(team_arms) or {fallback}
        for cart in carts:
            for arm in row_arms:
                causal[arm]["cart_ticks"] += 1
                if "supports_player" in cart:
                    causal[arm]["cart_support_observation_ticks"] += 1
                    causal[arm]["cart_support_ticks"] += int(bool(cart["supports_player"]))
            controller = integer_coordinate(cart.get("control_team"))
            if controller is not None and controller > 0:
                control_arm = str(
                    team_arms[controller - 1]
                    if controller <= len(team_arms) else fallback
                )
                causal[control_arm]["controlled_cart_ticks"] += 1
                depth = cart.get("depth")
                if isinstance(depth, (int, float)) and np.isfinite(depth):
                    causal[control_arm]["controlled_cart_depth"] += float(depth)
                    causal[control_arm]["controlled_cart_depth_mass"] += 1
                speed = cart.get("speed")
                if isinstance(speed, (int, float)) and np.isfinite(speed):
                    causal[control_arm]["controlled_cart_speed"] += abs(float(speed))
                    causal[control_arm]["controlled_cart_speed_mass"] += 1
        assignments = row.get("assignments") or []
        engine_time_frame_mass += 1
        engine_time_coordinate_mass += len(assignments)
        engine_times = [
            float(assignment["engine_time"])
            for assignment in assignments
            if isinstance(assignment.get("engine_time"), (int, float))
            and np.isfinite(assignment["engine_time"])
        ]
        engine_time_finite_coordinate_mass += len(engine_times)
        engine_time = float(np.median(engine_times)) if engine_times else None
        engine_time_nonmonotone_frame_mass += int(
            engine_time is not None and previous_engine_time is not None
            and engine_time < previous_engine_time
        )
        interval = (
            max(0.0, engine_time - previous_engine_time)
            if engine_time is not None and previous_engine_time is not None else 0.0
        )
        if engine_time is not None:
            previous_engine_time = engine_time
        for assignment in assignments:
            team = integer_coordinate(assignment.get("team"))
            arm = assignment.get("policy_arm") or (
                team_arms[team - 1]
                if team is not None and 0 < team <= len(team_arms) else fallback
            )
            player = integer_coordinate(assignment.get("edict"))
            kind = str(assignment.get("kind") or "unknown")
            applied_current = bool(assignment.get("applied_action_current"))
            applied_action = assignment.get("applied_action")
            applied_source = assignment.get("applied_policy_arm")
            applied_arm = None if applied_source is None else str(applied_source)
            arm = str(arm)
            if assignment.get("spawn_swizzle_active"):
                spawn_events[(
                    arm, player, integer_coordinate(assignment.get("spawn_swizzle_epoch")),
                )] = assignment
            selected_exposure[arm] += interval
            if applied_arm is None:
                unattributed_player_seconds += interval
            else:
                exposure[applied_arm] += interval
            if player is not None:
                players[arm].add(player)
            if applied_arm is not None:
                if player is not None:
                    players[applied_arm].add(player)
                causal[applied_arm]["applied_source_rows"] += 1
            kinds[arm][kind] += 1
            causal[arm]["rows"] += 1
            if kind in ATTACK_KINDS:
                causal[arm]["attack"] += 1
            if kind == "idle":
                causal[arm]["selected_stock"] += 1
            if kind in CART_ACTION_KINDS:
                causal[arm]["selected_cart"] += 1
            if kind in OBJECTIVE_KINDS:
                causal[arm]["selected_objective"] += 1
            if applied_current and applied_arm is not None:
                causal[applied_arm]["applied_rows"] += 1
                for key in ("routed_current", "goal_current", "goal_match", "target_touch", "touch_current"):
                    causal[applied_arm][key] += int(bool(assignment.get(key)))
                if applied_action == "idle":
                    causal[applied_arm]["applied_stock"] += 1
                    causal[applied_arm]["stock_cart_goal"] += int(assignment.get("goal_kind") == "cart")
                if applied_action in CART_ACTION_KINDS:
                    causal[applied_arm]["applied_cart"] += 1
                    causal[applied_arm]["cart_routed"] += int(bool(assignment.get("routed_current")))
                    causal[applied_arm]["cart_goal"] += int(bool(assignment.get("goal_current")))
                    causal[applied_arm]["cart_touch"] += int(bool(assignment.get("touch_current")))
                if applied_action in OBJECTIVE_KINDS:
                    causal[applied_arm]["applied_objective"] += 1
                    causal[applied_arm]["objective_cart_goal"] += int(assignment.get("goal_kind") == "cart")
            if applied_current and applied_arm is not None and applied_action in ATTACK_KINDS:
                causal[applied_arm]["applied_attack"] += 1
                causal[applied_arm]["attack_routed"] += int(bool(assignment.get("routed_current")))
                causal[applied_arm]["attack_goal"] += int(bool(assignment.get("goal_current")))
                causal[applied_arm]["attack_touch"] += int(bool(assignment.get("touch_current")))
            for routed in assignment.get("routed_outcomes") or ():
                routed_outcome_row_mass += 1
                outcome_coordinates = routed.get("outcomes") or {}
                routed_outcome_coordinate_mass += len(outcome_coordinates)
                routed_outcome_finite_coordinate_mass += sum(
                    isinstance(value, (int, float)) and np.isfinite(value)
                    for value in outcome_coordinates.values()
                )
                routed_source = routed.get("policy_arm")
                if routed_source is None:
                    continue
                routed_outcome_attributed_row_mass += 1
                routed_arm = str(routed_source)
                routed_mass[routed_arm]["rows"] += 1
                routed_mass[routed_arm]["coordinates"] += len(outcome_coordinates)
                routed_mass[routed_arm]["finite_coordinates"] += sum(
                    isinstance(value, (int, float)) and np.isfinite(value)
                    for value in outcome_coordinates.values()
                )
                if player is not None:
                    players[routed_arm].add(player)
                causal[routed_arm]["applied_policy_rows"] += 1
                for key, value in outcome_coordinates.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        values[routed_arm][key] += float(value)
                        value_mass[routed_arm][key] += 1
        for event in row.get("realized_events") or []:
            event_row_mass += 1
            event_source = event.get("policy_arm")
            kind = str(event.get("kind") or "unknown")
            source_value = event.get("value")
            numeric = isinstance(source_value, (int, float))
            event_numeric_value_mass += int(numeric)
            finite = bool(numeric and np.isfinite(source_value))
            event_finite_value_mass += int(finite)
            if event_source is None:
                continue
            arm = str(event_source)
            event_attributed_row_mass += 1
            event_mass[arm]["rows"] += 1
            event_mass[arm]["numeric_values"] += int(numeric)
            event_mass[arm]["finite_values"] += int(finite)
            if not numeric:
                continue
            value = float(source_value)
            if not finite:
                continue
            if kind == "damage":
                values[arm]["realized_damage"] += value
                value_mass[arm]["realized_damage"] += 1
                if isinstance(event.get("aligned_target"), (bool, int)):
                    values[arm]["aligned_damage"] += value * int(bool(event["aligned_target"]))
                    value_mass[arm]["aligned_damage"] += 1
            elif kind == "kill":
                values[arm]["realized_kills"] += 1
                value_mass[arm]["realized_kills"] += 1
                if isinstance(event.get("aligned_target"), (bool, int)):
                    values[arm]["aligned_kills"] += int(bool(event["aligned_target"]))
                    value_mass[arm]["aligned_kills"] += 1
    out = {}
    for arm in sorted(set(players) | set(values) | set(causal) | {
        key[0] for key in spawn_events
    }):
        n = len(players[arm])
        player_seconds = exposure[arm]
        rows_n = causal[arm]["rows"]
        applied_rows_n = causal[arm]["applied_source_rows"]
        arm_spawn_events = [
            event for key, event in spawn_events.items() if key[0] == arm
        ]
        realized_spawns = [
            event for event in arm_spawn_events
            if isinstance(event.get("spawn_swizzle_actual_time"), (int, float))
            and np.isfinite(event["spawn_swizzle_actual_time"])
            and float(event["spawn_swizzle_actual_time"]) >= 0
        ]
        spawn_latencies = [
            float(event["spawn_swizzle_actual_time"])
            - float(event["spawn_swizzle_scheduled_time"])
            for event in realized_spawns
            if isinstance(event.get("spawn_swizzle_scheduled_time"), (int, float))
            and np.isfinite(event["spawn_swizzle_scheduled_time"])
        ]
        total_kinds = sum(kinds[arm].values())
        probabilities = (
            np.asarray(list(kinds[arm].values()), dtype=np.float64) / total_kinds
            if total_kinds else np.empty(0, dtype=np.float64)
        )
        aggressive = quotient(sum(kinds[arm][key] for key in PRESSURE_KINDS), total_kinds)
        behavior_observation_mass = sum(
            bool(value_mass[arm][name]) for name in BEHAVIOR_MEASURE_NAMES
        )
        out[arm] = {
            "players": len(players[arm]),
            "observed_player_seconds": player_seconds,
            "selected_player_seconds": selected_exposure[arm],
            "global_unattributed_player_seconds": unattributed_player_seconds,
            "player_second_estimator": "right_endpoint_engine_time",
            "engine_time_frame_mass": engine_time_frame_mass,
            "engine_time_coordinate_mass": engine_time_coordinate_mass,
            "engine_time_finite_coordinate_mass": engine_time_finite_coordinate_mass,
            "engine_time_nonfinite_coordinate_mass": (
                engine_time_coordinate_mass - engine_time_finite_coordinate_mass
            ),
            "engine_time_nonmonotone_frame_mass": engine_time_nonmonotone_frame_mass,
            "routed_outcome_row_mass": routed_mass[arm]["rows"],
            "routed_outcome_coordinate_mass": routed_mass[arm]["coordinates"],
            "routed_outcome_finite_coordinate_mass": routed_mass[arm]["finite_coordinates"],
            "routed_outcome_nonfinite_coordinate_mass": (
                routed_mass[arm]["coordinates"] - routed_mass[arm]["finite_coordinates"]
            ),
            "global_routed_outcome_row_mass": routed_outcome_row_mass,
            "global_routed_outcome_attributed_row_mass": routed_outcome_attributed_row_mass,
            "global_routed_outcome_unattributed_row_mass": (
                routed_outcome_row_mass - routed_outcome_attributed_row_mass
            ),
            "global_routed_outcome_coordinate_mass": routed_outcome_coordinate_mass,
            "global_routed_outcome_finite_coordinate_mass": routed_outcome_finite_coordinate_mass,
            "global_routed_outcome_nonfinite_coordinate_mass": (
                routed_outcome_coordinate_mass - routed_outcome_finite_coordinate_mass
            ),
            "event_row_mass": event_mass[arm]["rows"],
            "event_numeric_value_mass": event_mass[arm]["numeric_values"],
            "event_non_numeric_value_mass": event_mass[arm]["rows"] - event_mass[arm]["numeric_values"],
            "event_finite_value_mass": event_mass[arm]["finite_values"],
            "event_nonfinite_value_mass": event_mass[arm]["numeric_values"] - event_mass[arm]["finite_values"],
            "global_event_row_mass": event_row_mass,
            "global_event_attributed_row_mass": event_attributed_row_mass,
            "global_event_unattributed_row_mass": event_row_mass - event_attributed_row_mass,
            "global_event_numeric_value_mass": event_numeric_value_mass,
            "global_event_non_numeric_value_mass": event_row_mass - event_numeric_value_mass,
            "global_event_finite_value_mass": event_finite_value_mass,
            "global_event_nonfinite_value_mass": event_numeric_value_mass - event_finite_value_mass,
            "outcome_coordinate_observation_measure": {
                name: value_mass[arm][name] for name in sorted(value_mass[arm])
            },
            "damage_per_player_second": observed_quotient(
                values[arm]["enemy_damage_dealt"], player_seconds,
                value_mass[arm]["enemy_damage_dealt"],
            ),
            "kills_per_player_minute": observed_quotient(
                60 * values[arm]["enemy_kills"], player_seconds,
                value_mass[arm]["enemy_kills"],
            ),
            "deaths_per_player_minute": observed_quotient(
                60 * values[arm]["deaths"], player_seconds,
                value_mass[arm]["deaths"],
            ),
            "cart_push_per_player_second": observed_quotient(
                values[arm]["cart_push"], player_seconds,
                value_mass[arm]["cart_push"],
            ),
            "cart_contest_per_player_second": observed_quotient(
                values[arm]["cart_contest"], player_seconds,
                value_mass[arm]["cart_contest"],
            ),
            "behavior_coordinate_observation_mass": behavior_observation_mass,
            "behavior_coordinate_missing_mass": (
                len(BEHAVIOR_MEASURE_NAMES) - behavior_observation_mass
            ),
            "aggressive_assignment_rate": aggressive,
            "selected_row_mass": rows_n,
            "applied_source_row_mass": applied_rows_n,
            "applied_current_row_mass": causal[arm]["applied_rows"],
            "objective_attack_assignment_rate": quotient(causal[arm]["attack"], rows_n),
            "selected_stock_preservation_rate": quotient(causal[arm]["selected_stock"], rows_n),
            "selected_cart_action_rate": quotient(causal[arm]["selected_cart"], rows_n),
            "selected_objective_duty_rate": quotient(causal[arm]["selected_objective"], rows_n),
            "source_matched_application_fraction": quotient(causal[arm]["applied_rows"], applied_rows_n),
            "routed_outcome_rows_per_selected_row": quotient(causal[arm]["applied_policy_rows"], rows_n),
            "applied_stock_preservation_rate": quotient(causal[arm]["applied_stock"], applied_rows_n),
            "applied_cart_action_rate": quotient(causal[arm]["applied_cart"], applied_rows_n),
            "applied_objective_duty_rate": quotient(causal[arm]["applied_objective"], applied_rows_n),
            "stock_cart_goal_fraction": quotient(causal[arm]["stock_cart_goal"], causal[arm]["applied_stock"]),
            "cart_action_route_fraction": quotient(causal[arm]["cart_routed"], causal[arm]["applied_cart"]),
            "cart_action_goal_fraction": quotient(causal[arm]["cart_goal"], causal[arm]["applied_cart"]),
            "cart_action_touch_fraction": quotient(causal[arm]["cart_touch"], causal[arm]["applied_cart"]),
            "objective_cart_goal_fraction": quotient(causal[arm]["objective_cart_goal"], causal[arm]["applied_objective"]),
            "cart_push_per_applied_objective_row": observed_quotient(
                values[arm]["cart_push"], causal[arm]["applied_objective"],
                value_mass[arm]["cart_push"],
            ),
            "cart_push_per_applied_cart_row": observed_quotient(
                values[arm]["cart_push"], causal[arm]["applied_cart"],
                value_mass[arm]["cart_push"],
            ),
            "controlled_cart_fraction": quotient(causal[arm]["controlled_cart_ticks"], causal[arm]["cart_ticks"]),
            "controlled_cart_depth_observation_mass": causal[arm]["controlled_cart_depth_mass"],
            "controlled_cart_speed_observation_mass": causal[arm]["controlled_cart_speed_mass"],
            "controlled_cart_depth_mean": quotient(
                causal[arm]["controlled_cart_depth"], causal[arm]["controlled_cart_depth_mass"],
            ),
            "controlled_cart_speed_mean": quotient(
                causal[arm]["controlled_cart_speed"], causal[arm]["controlled_cart_speed_mass"],
            ),
            "cart_support_observation_mass": causal[arm]["cart_support_observation_ticks"],
            "cart_support_mass": causal[arm]["cart_support_ticks"],
            "cart_support_fraction": (
                causal[arm]["cart_support_ticks"] / causal[arm]["cart_support_observation_ticks"]
                if causal[arm]["cart_support_observation_ticks"] else None
            ),
            "spawn_schedule_event_mass": len(arm_spawn_events),
            "spawn_schedule_realized_mass": len(realized_spawns),
            "spawn_schedule_pending_mass": len(arm_spawn_events) - len(realized_spawns),
            "spawn_schedule_realized_fraction": quotient(len(realized_spawns), len(arm_spawn_events)),
            "spawn_schedule_latency_mean": (
                float(np.mean(spawn_latencies)) if spawn_latencies else None
            ),
            "spawn_schedule_latency_measure": scalar_measure(spawn_latencies),
            "spawn_schedule_lane_measure": scalar_measure(
                event.get("spawn_swizzle_lane") for event in realized_spawns
            ),
            "spawn_schedule_spot_measure": scalar_measure(
                event.get("spawn_swizzle_spot") for event in realized_spawns
            ),
            "spawn_schedule_cohort_measure": scalar_measure(
                event.get("spawn_swizzle_cohort") for event in realized_spawns
            ),
            "spawn_schedule_generation_measure": scalar_measure(
                event.get("spawn_swizzle_generation") for event in realized_spawns
            ),
            "spawn_schedule_player_count_measure": scalar_measure(
                event.get("spawn_swizzle_player_count") for event in arm_spawn_events
            ),
            "spawn_schedule_spot_count_measure": scalar_measure(
                event.get("spawn_swizzle_spot_count") for event in arm_spawn_events
            ),
            "spawn_schedule_slot_count_measure": scalar_measure(
                event.get("spawn_swizzle_slot_count") for event in arm_spawn_events
            ),
            "applied_attack_rate": quotient(causal[arm]["applied_attack"], applied_rows_n),
            "executed_attack_rate": quotient(causal[arm]["attack_routed"], applied_rows_n),
            "executed_attack_fraction": quotient(causal[arm]["attack_routed"], causal[arm]["applied_attack"]),
            "attack_goal_rate": quotient(causal[arm]["attack_goal"], applied_rows_n),
            "attack_touch_rate": quotient(causal[arm]["attack_touch"], applied_rows_n),
            "realized_damage_per_player_second": observed_quotient(
                values[arm]["realized_damage"], player_seconds,
                value_mass[arm]["realized_damage"],
            ),
            "aligned_damage_per_player_second": observed_quotient(
                values[arm]["aligned_damage"], player_seconds,
                value_mass[arm]["aligned_damage"],
            ),
            "realized_kills_per_player_minute": observed_quotient(
                60 * values[arm]["realized_kills"], player_seconds,
                value_mass[arm]["realized_kills"],
            ),
            "aligned_kills_per_player_minute": observed_quotient(
                60 * values[arm]["aligned_kills"], player_seconds,
                value_mass[arm]["aligned_kills"],
            ),
            "action_entropy_nats": (
                float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
                if total_kinds else None
            ),
            "causal": {key: quotient(causal[arm][key], applied_rows_n) for key in ("routed_current", "goal_current", "goal_match", "target_touch", "touch_current")},
        }
    return out

def environment_metrics(rows):
    carts = [cart for row in rows for cart in (row.get("carts") or ())]
    support = [cart.get("supports_player") for cart in carts if "supports_player" in cart]
    rollback = [cart.get("rollback_active") for cart in carts if "rollback_active" in cart]
    spawn_events = {}
    for row in rows:
        for assignment in row.get("assignments") or ():
            if assignment.get("spawn_swizzle_active"):
                spawn_events[(
                    integer_coordinate(assignment.get("edict")),
                    integer_coordinate(assignment.get("spawn_swizzle_epoch")),
                )] = assignment
    spawn_events = list(spawn_events.values())
    realized_spawns = [
        event for event in spawn_events
        if float(event.get("spawn_swizzle_actual_time", -1)) >= 0
    ]
    wave_lane_mass = defaultdict(int)
    wave_spot_mass = defaultdict(int)
    for event in realized_spawns:
        scheduled = event.get("spawn_swizzle_scheduled_time")
        scheduled = float(scheduled) if isinstance(scheduled, (int, float)) and np.isfinite(scheduled) else None
        wave_lane_mass[(scheduled, integer_coordinate(event.get("spawn_swizzle_lane")))] += 1
        wave_spot_mass[(scheduled, integer_coordinate(event.get("spawn_swizzle_spot")))] += 1
    speeds = [
        float(cart["speed"]) for cart in carts
        if isinstance(cart.get("speed"), (int, float)) and np.isfinite(cart["speed"])
    ]
    return {
        "telemetry_frame_mass": len(rows),
        "cart_atom_mass": len(carts),
        "cart_team_pair_atom_mass": (
            sum(team_counts) if (team_counts := [
                int(cart["team_count"]) for cart in carts
                if isinstance(cart.get("team_count"), (int, float))
                and np.isfinite(cart["team_count"])
            ]) else None
        ),
        "cart_team_count_measure": scalar_measure(cart.get("team_count") for cart in carts),
        "path_length_measure": scalar_measure(cart.get("path_length") for cart in carts),
        "path_nonpositive_mass": sum(
            isinstance(cart.get("path_length"), (int, float))
            and cart.get("path_length") <= 0 for cart in carts
        ),
        "support_observation_mass": len(support),
        "support_mass": sum(bool(value) for value in support),
        "support_absent_mass": len(carts) - len(support),
        "support_false_mass": sum(not bool(value) for value in support),
        "rollback_observation_mass": len(rollback),
        "rollback_active_mass": sum(bool(value) for value in rollback),
        "rollback_target_measure": scalar_measure(
            cart.get("rollback_target") for cart in carts
        ),
        "cart_speed_measure": scalar_measure(speeds),
        "forward_motion_mass": sum(value > 0 for value in speeds),
        "reverse_motion_mass": sum(value < 0 for value in speeds),
        "stationary_motion_mass": sum(value == 0 for value in speeds),
        "spawn_schedule_event_mass": len(spawn_events),
        "spawn_schedule_realized_mass": len(realized_spawns),
        "spawn_schedule_pending_mass": len(spawn_events) - len(realized_spawns),
        "spawn_schedule_player_count_measure": scalar_measure(
            event.get("spawn_swizzle_player_count") for event in spawn_events
        ),
        "spawn_schedule_spot_count_measure": scalar_measure(
            event.get("spawn_swizzle_spot_count") for event in spawn_events
        ),
        "spawn_schedule_slot_count_measure": scalar_measure(
            event.get("spawn_swizzle_slot_count") for event in spawn_events
        ),
        "spawn_schedule_ticket_measure": scalar_measure(
            event.get("spawn_swizzle_ticket") for event in spawn_events
        ),
        "spawn_schedule_cohort_measure": scalar_measure(
            event.get("spawn_swizzle_cohort") for event in spawn_events
        ),
        "spawn_schedule_cohort_count_measure": scalar_measure(
            event.get("spawn_swizzle_cohort_count") for event in spawn_events
        ),
        "spawn_schedule_generation_measure": scalar_measure(
            event.get("spawn_swizzle_generation") for event in spawn_events
        ),
        "spawn_schedule_latency_measure": scalar_measure(
            float(event["spawn_swizzle_actual_time"]) - float(event["spawn_swizzle_scheduled_time"])
            for event in realized_spawns
        ),
        "spawn_schedule_lane_atom_mass": len(wave_lane_mass),
        "spawn_schedule_spot_atom_mass": len(wave_spot_mass),
        "spawn_schedule_wave_lane_multiplicity_measure": scalar_measure(wave_lane_mass.values()),
        "spawn_schedule_wave_spot_multiplicity_measure": scalar_measure(wave_spot_mass.values()),
    }

def interval(mean, count, variance, z=1.96):
    if count <= 1:
        return [None, None]
    radius = z * math.sqrt(max(0.0, variance) / count)
    return [mean - radius, mean + radius]

def mean_interval(values, z=1.96):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return [None, None]
    radius = z * float(values.std(ddof=1)) / math.sqrt(len(values))
    mean = float(values.mean())
    return [max(0.0, mean - radius), min(1.0, mean + radius)]

def value_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if len(values) else 0.0
    variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    return interval(mean, len(values), variance)

def block_key(record, pair):
    cfg = record.get("configuration") or {}
    perturbation = (record.get("perturbation") or {}).get("name", "unknown")
    players = cfg.get("players_per_team")
    players = tuple(players) if isinstance(players, list) else (players,)
    return (
        tuple(pair), perturbation, cfg.get("seed"), cfg.get("map"), cfg.get("teams"),
        players, cfg.get("carts"), cfg.get("skill"),
        cfg.get("pair", cfg.get("study_repetition")),
    )

def record_score(rounds, team_arms, first):
    scores = []
    for kind, winner, _ in rounds:
        if kind == "tie":
            scores.append(0.5)
        elif winner is None or winner <= 0 or winner > len(team_arms):
            continue
        else:
            scores.append(1.0 if team_arms[winner - 1] == first else 0.0)
    return float(np.mean(scores)) if scores else None

def ordered_pair(left, right):
    order = {arm: index for index, arm in enumerate(ARMS)}
    return (left, right) if order.get(left, len(order)) <= order.get(right, len(order)) else (right, left)

def policy_realization(arm, provenance):
    source = (provenance or {}).get(arm) or {}
    return {
        "assigned_arm": arm,
        **{name: source.get(name) for name in REALIZATION_FIELDS},
        "checkpoint_updates": source.get("updates"),
    }

def realization_coordinate(arm, provenances):
    atoms = defaultdict(int)
    updates = []
    for provenance in provenances:
        atom = policy_realization(arm, provenance)
        update = atom.pop("checkpoint_updates")
        atoms[json.dumps(atom, sort_keys=True, separators=(",", ":"))] += 1
        if isinstance(update, (int, float)):
            updates.append(update)
    return {
        "assigned_arm": arm,
        "source_atoms": [
            {**json.loads(atom), "mass": mass}
            for atom, mass in sorted(atoms.items())
        ],
        "checkpoint_updates_measure": scalar_measure(updates),
    }

def realization_elo(contests):
    coordinates = {}
    matches = []
    for contest in contests:
        keys = []
        for coordinate in contest["realizations"]:
            key = json.dumps(coordinate, sort_keys=True, separators=(",", ":"))
            coordinates[key] = coordinate
            keys.append(key)
        matches.append({**contest, "arms": tuple(keys)})
    measured = elo(matches)
    ratings = measured.pop("ratings")
    pair_measures = measured.pop("pair_measures")
    return {
        **measured,
        "coordinates": [
            {"coordinate": coordinates[key], "elo": ratings.get(key)}
            for key in sorted(coordinates)
        ],
        "pair_measures": [
            {
                **{name: value for name, value in row.items() if name not in ("left", "right")},
                "left": coordinates[row["left"]],
                "right": coordinates[row["right"]],
            }
            for row in pair_measures.values()
        ],
    }

def pairwise_summary(contests, paired_metrics):
    grouped = defaultdict(list)
    perturbations = defaultdict(lambda: defaultdict(list))
    round_counts = defaultdict(int)
    perturbation_rounds = defaultdict(lambda: defaultdict(int))
    for match in contests:
        left, right = match["arms"]
        first, second = ordered_pair(left, right)
        score = match["score"] if left == first else 1 - match["score"]
        grouped[(first, second)].append(float(score))
        perturbations[(first, second)][match["perturbation"]].append(float(score))
        round_counts[(first, second)] += int(match.get("rounds") or 0)
        perturbation_rounds[(first, second)][match["perturbation"]] += int(match.get("rounds") or 0)
    out = {}
    for pair in sorted(set(grouped) | set(paired_metrics)):
        first, second = pair
        scores = grouped[pair]
        by_perturbation = {
            name: {
                "blocks": len(values),
                "rounds": perturbation_rounds[pair][name],
                "expectation": float(np.mean(values)),
                "interval_95": mean_interval(values),
            }
            for name, values in sorted(perturbations[pair].items())
        }
        lifts = {}
        metric_rows = paired_metrics[pair]
        metric_names = sorted({name for row in metric_rows for name, value in row.items() if isinstance(value, (int, float))})
        for key in metric_names:
            values = np.asarray([
                row[key] for row in metric_rows
                if key in row and np.isfinite(row[key])
            ], dtype=np.float64)
            if len(values):
                mean = float(values.mean())
                lifts[key] = {
                    "matches": len(values),
                    "mean": mean,
                    "interval_95": interval(mean, len(values), float(values.var(ddof=1)) if len(values) > 1 else 0.0),
                }
        lifts_by_perturbation = {}
        for perturbation in sorted({row.get("perturbation") for row in metric_rows if row.get("perturbation")}):
            selected_rows = [row for row in metric_rows if row.get("perturbation") == perturbation]
            lifts_by_perturbation[perturbation] = {
                key: {
                    "blocks": len([
                        row for row in selected_rows
                        if key in row and np.isfinite(row[key])
                    ]),
                    "mean": float(np.mean([
                        row[key] for row in selected_rows
                        if key in row and np.isfinite(row[key])
                    ])),
                    "interval_95": value_interval([
                        row[key] for row in selected_rows
                        if key in row and np.isfinite(row[key])
                    ]),
                }
                for key in metric_names
                if any(key in row and np.isfinite(row[key]) for row in selected_rows)
            }
        total = float(np.sum(scores))
        out[f"{first}_vs_{second}"] = {
            "first": first,
            "second": second,
            "blocks": len(scores),
            "rounds": round_counts[pair],
            "expectation": total / len(scores) if scores else None,
            "interval_95": mean_interval(scores),
            "by_perturbation": by_perturbation,
            "perturbation_minimum": min((row["expectation"] for row in by_perturbation.values()), default=None),
            "perturbation_expectation_measure": scalar_measure(
                row["expectation"] for row in by_perturbation.values()
            ),
            "baseline_centered_perturbation_atoms": [
                {
                    "perturbation": name,
                    "expectation_delta": row["expectation"] - by_perturbation["baseline"]["expectation"],
                }
                for name, row in by_perturbation.items()
                if "baseline" in by_perturbation
            ],
            "paired_lifts": lifts,
            "paired_lifts_by_perturbation": lifts_by_perturbation,
        }
    return out

def elo(matches):
    arms = sorted({arm for match in matches for arm in match["arms"]})
    index = {arm: position for position, arm in enumerate(arms)}
    aggregates = defaultdict(lambda: [0.0, 0])
    for match in matches:
        key = tuple(match["arms"])
        aggregates[key][0] += float(match["score"])
        aggregates[key][1] += 1
    matrix = []
    targets = []
    pair_measures = {}
    for (left, right), (score, count) in aggregates.items():
        probability = score / count
        finite_log_odds = 0.0 < probability < 1.0
        difference = (
            400.0 * math.log10(probability / (1.0 - probability))
            if finite_log_odds else None
        )
        coordinate = {
            "finite": difference,
            "positive_infinity_mass": int(probability == 1.0),
            "negative_infinity_mass": int(probability == 0.0),
        }
        pair_measures[f"{left}_vs_{right}"] = {
            "left": left,
            "right": right,
            "block_mass": count,
            "left_score_integral": score,
            "left_score_expectation": probability,
            "finite_log_odds_mass": int(finite_log_odds),
            "boundary_log_odds_mass": int(not finite_log_odds),
            "elo_difference": difference,
            "elo_difference_coordinate": coordinate,
        }
        if difference is None:
            continue
        row = np.zeros(len(arms), dtype=np.float64)
        row[index[left]] = math.sqrt(count)
        row[index[right]] = -math.sqrt(count)
        matrix.append(row)
        targets.append(difference * math.sqrt(count))
    if arms:
        matrix.append(np.ones(len(arms), dtype=np.float64))
        targets.append(0.0)
    design = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(design)) if design.size else 0
    ratings = {}
    residual_square_integral = None
    if arms and rank == len(arms):
        coordinate = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = design @ coordinate - target
        residual_square_integral = float(residual @ residual)
        ratings = {
            arm: 1500.0 + float(coordinate[index[arm]]) for arm in arms
        }
    return {
        "ratings": ratings,
        "pair_measures": pair_measures,
        "arm_mass": len(arms),
        "finite_log_odds_pair_mass": sum(
            row["finite_log_odds_mass"] for row in pair_measures.values()
        ),
        "boundary_log_odds_pair_mass": sum(
            row["boundary_log_odds_mass"] for row in pair_measures.values()
        ),
        "design_row_mass": len(matrix),
        "design_rank": rank,
        "design_rank_deficiency": len(arms) - rank,
        "residual_square_integral": residual_square_integral,
    }

def directed_elo_coordinates(rating_measure, pairs):
    measures = (rating_measure or {}).get("pair_measures") or {}
    out = []
    for left, right in pairs:
        row = next((value for value in measures.values()
                    if {value.get("left"), value.get("right")} == {left, right}), None)
        if row is None:
            out.append({"left": left, "right": right, "block_mass": 0,
                        "left_score_expectation": None,
                        "elo_difference_coordinate": None})
            continue
        same_direction = row["left"] == left
        probability = row["left_score_expectation"] if same_direction else 1.0 - row["left_score_expectation"]
        source = row["elo_difference_coordinate"]
        coordinate = {
            "finite": source["finite"] if same_direction or source["finite"] is None else -source["finite"],
            "positive_infinity_mass": source["positive_infinity_mass"] if same_direction else source["negative_infinity_mass"],
            "negative_infinity_mass": source["negative_infinity_mass"] if same_direction else source["positive_infinity_mass"],
        }
        out.append({"left": left, "right": right,
                    "block_mass": row["block_mass"],
                    "left_score_integral": probability * row["block_mass"],
                    "left_score_expectation": probability,
                    "elo_difference_coordinate": coordinate})
    return out

def scalar_measure(values):
    source = list(values)
    numeric = [float(value) for value in source if isinstance(value, (int, float))]
    values = np.asarray([value for value in numeric if np.isfinite(value)], dtype=np.float64)
    prefix = {
        "source_mass": len(source),
        "numeric_mass": len(numeric),
        "non_numeric_mass": len(source) - len(numeric),
        "nonfinite_mass": len(numeric) - len(values),
    }
    if not len(values):
        return {**prefix, "mass": 0, "integral": 0.0, "mean": None, "variance": None,
                "minimum": None, "median": None, "maximum": None}
    return {
        **prefix,
        "mass": len(values),
        "integral": float(values.sum()),
        "mean": float(values.mean()),
        "variance": float(values.var()),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }

def atom_measure(values):
    values = list(values)
    atoms = defaultdict(int)
    for value in values:
        atoms[json.dumps(value, sort_keys=True, separators=(",", ":"))] += 1
    return {
        "mass": len(values),
        "atoms": [
            {"coordinate": json.loads(value), "mass": mass}
            for value, mass in sorted(atoms.items())
        ],
    }

def metric_measures(rows):
    numeric = sorted({
        key for row in rows for key, value in row.items()
        if key != "perturbation" and isinstance(value, (int, float))
    })
    names = sorted({str(row.get("perturbation", "unknown")) for row in rows})
    by_perturbation = {
        name: {
            key: scalar_measure(
                row.get(key) for row in rows
                if str(row.get("perturbation", "unknown")) == name
            )
            for key in numeric
        }
        for name in names
    }
    expectations = {
        key: [
            measure["mean"] for measure in (
                by_perturbation[name][key] for name in names
            ) if measure["mean"] is not None
        ]
        for key in numeric
    }
    baseline = by_perturbation.get("baseline") or {}
    return {
        "coordinates": {key: scalar_measure(row.get(key) for row in rows) for key in numeric},
        "by_perturbation": by_perturbation,
        "perturbation_expectations": {
            key: scalar_measure(values) for key, values in expectations.items()
        },
        "baseline_centered_perturbation_atoms": {
            key: [
                {
                    "perturbation": name,
                    "expectation_delta": by_perturbation[name][key]["mean"] - baseline[key]["mean"],
                }
                for name in names
                if key in baseline and baseline[key]["mean"] is not None
                and by_perturbation[name][key]["mean"] is not None
            ]
            for key in numeric
        },
    }

def optimization_measures(record_rows):
    def categorical_pairs(rows, source, live):
        pairs = defaultdict(int)
        for row in rows:
            pairs[(row.get(source), row.get(live))] += 1
        return [
            {"source": source_value, "live": live_value, "mass": mass}
            for (source_value, live_value), mass in sorted(
                pairs.items(), key=lambda item: repr(item[0])
            )
        ]

    training = []
    heldout = []
    update_values = defaultdict(lambda: defaultdict(list))
    training_by_arm = defaultdict(list)
    heldout_by_arm = defaultdict(list)
    heldout_by_record = defaultdict(dict)
    for record, rows in record_rows:
        cfg = record.get("configuration") or {}
        if cfg.get("split") != "heldout":
            arm = str(cfg.get("policy_arm") or "unknown")
            updates = [row["update"] for row in rows if isinstance(row.get("update"), dict)]
            for update in updates:
                for name, value in update.items():
                    if isinstance(value, (int, float)):
                        update_values[arm][name].append(value)
            checkpoint = (record.get("artifacts") or {}).get("checkpoint_out") or {}
            initial = (record.get("artifacts") or {}).get("checkpoint_initial") or {}
            observation = {
                "record": record.get("id"),
                "arm": arm,
                "update_row_mass": len(updates),
                "telemetry_update_counter_measure": scalar_measure(
                    row.get("updates") for row in rows if "updates" in row
                ),
                "checkpoint": checkpoint,
                "initial_checkpoint": initial,
            }
            training.append(observation)
            training_by_arm[arm].append(observation)
        checkpoints = (
            (record.get("realization_measures") or {}).get("checkpoint_observations") or {}
        )
        for arm, source in checkpoints.items():
            source_mass = int(source.get("source_weight_mass") or 0)
            observation = {
                "record": record.get("id"),
                "arm": str(arm),
                **source,
                "loaded_source_weight_fraction": quotient(
                    int(source.get("loaded_weight_mass") or 0), source_mass,
                ),
                "loaded_live_weight_fraction": quotient(
                    int(source.get("loaded_weight_mass") or 0),
                    int(source.get("live_weight_mass") or 0),
                ),
            }
            heldout.append(observation)
            heldout_by_arm[str(arm)].append(observation)
            heldout_by_record[record.get("id")][str(arm)] = observation
    arms = {}
    for arm in sorted(set(training_by_arm) | set(heldout_by_arm)):
        training_rows = training_by_arm[arm]
        heldout_rows = heldout_by_arm[arm]
        arms[arm] = {
            "training_record_mass": len(training_rows),
            "training_update_row_mass": sum(row["update_row_mass"] for row in training_rows),
            "update_coordinate_measures": {
                name: scalar_measure(values)
                for name, values in sorted(update_values[arm].items())
            },
            "positive_gradient_norm_mass": sum(
                isinstance(value, (int, float)) and np.isfinite(value) and value > 0
                for value in update_values[arm].get("gradient_norm", ())
            ),
            "checkpoint_output_exists_mass": sum(
                bool(row["checkpoint"].get("exists")) for row in training_rows
            ),
            "initial_checkpoint_exists_mass": sum(
                bool(row["initial_checkpoint"].get("exists")) for row in training_rows
            ),
            "checkpoint_output_bytes_measure": scalar_measure(
                row["checkpoint"].get("bytes") for row in training_rows
            ),
            "initial_checkpoint_bytes_measure": scalar_measure(
                row["initial_checkpoint"].get("bytes") for row in training_rows
            ),
            "heldout_checkpoint_observation_mass": len(heldout_rows),
            "heldout_loaded_source_weight_fraction_measure": scalar_measure(
                row.get("loaded_source_weight_fraction") for row in heldout_rows
            ),
            "heldout_loaded_live_weight_fraction_measure": scalar_measure(
                row.get("loaded_live_weight_fraction") for row in heldout_rows
            ),
            "heldout_composable_weight_mass_measure": scalar_measure(
                row.get("composable_weight_mass") for row in heldout_rows
            ),
            "heldout_checkpoint_updates_measure": scalar_measure(
                row.get("updates") for row in heldout_rows
            ),
            "heldout_shape_difference_mass_measure": scalar_measure(
                row.get("shape_difference_mass") for row in heldout_rows
            ),
            "heldout_source_only_weight_mass_measure": scalar_measure(
                row.get("source_only_weight_mass") for row in heldout_rows
            ),
            "heldout_live_only_weight_mass_measure": scalar_measure(
                row.get("live_only_weight_mass") for row in heldout_rows
            ),
            "heldout_nonfinite_weight_mass_measure": scalar_measure(
                row.get("nonfinite_weight_mass") for row in heldout_rows
            ),
            "heldout_load_exception_measure": categorical_pairs(
                heldout_rows, "load_exception", "live_arm",
            ),
            "source_live_arm_pair_measure": categorical_pairs(
                heldout_rows, "source_arm", "live_arm",
            ),
            "source_live_version_pair_measure": categorical_pairs(
                heldout_rows, "source_version", "live_version",
            ),
            "source_live_architecture_pair_measure": categorical_pairs(
                heldout_rows, "source_architecture", "live_architecture",
            ),
            "source_live_reward_contract_pair_measure": categorical_pairs(
                heldout_rows, "source_reward_contract", "live_reward_contract",
            ),
            "checkpoint_lineage_pair_measure": categorical_pairs(
                heldout_rows, "checkpoint_sha256", "lineage_initial_sha256",
            ),
        }
    initial_lineage = atom_measure(
        {
            "record": record,
            "trained_checkpoint_sha256": sources.get("matrix_fusion", {}).get("checkpoint_sha256"),
            "trained_lineage_initial_sha256": sources.get("matrix_fusion", {}).get("lineage_initial_sha256"),
            "initial_checkpoint_sha256": sources.get("initial_policy", {}).get("checkpoint_sha256"),
        }
        for record, sources in heldout_by_record.items()
        if "initial_policy" in sources
    )
    return {
        "training": training,
        "heldout": heldout,
        "by_arm": arms,
        "initial_policy_lineage_measure": initial_lineage,
    }

def map_space_measures(named_maps, rows):
    carts = [
        cart for row in rows
        for cart in (row["measurements"].get("cart_path_measures") or ())
    ]
    teams = [
        team for row in rows
        for team in (row["measurements"].get("team_objective_measures") or ())
    ]
    observed = {row["map"] for row in rows if row["map"]}
    return {
        "catalog_named_mass": len(named_maps),
        "map_observation_mass": len(rows),
        "observed_name_mass": len(observed),
        "catalog_observation_incidence_mass": sum(name in observed for name in named_maps),
        "entity_realization_id_mass": len({
            row.get("realization_id") for row in rows if row.get("realization_id")
        }),
        "entity_realization_reuse_measure": scalar_measure(
            row.get("realization_reuse_mass") for row in rows
        ),
        "cart_atom_mass": len(carts),
        "cart_path_segment_measure": scalar_measure(cart.get("path_segments") for cart in carts),
        "cart_path_length_measure": scalar_measure(cart.get("path_length") for cart in carts),
        "cart_rider_gap_segment_measure": scalar_measure(cart.get("rider_gap_segments") for cart in carts),
        "cart_spawn_route_measure": scalar_measure(cart.get("spawn_routes") for cart in carts),
        "cart_finite_spawn_route_measure": scalar_measure(cart.get("finite_spawn_routes") for cart in carts),
        "cart_path_non_degenerate_measure": scalar_measure(cart.get("path_non_degenerate_mass") for cart in carts),
        "cart_rider_continuous_measure": scalar_measure(cart.get("rider_continuous_mass") for cart in carts),
        "cart_spawn_reachable_measure": scalar_measure(cart.get("spawn_reachable_mass") for cart in carts),
        "cart_advanceable_measure": scalar_measure(cart.get("advanceable_mass") for cart in carts),
        "cart_zero_path_segment_mass": sum(
            isinstance(cart.get("path_segments"), (int, float))
            and int(cart["path_segments"]) == 0 for cart in carts
        ),
        "cart_zero_finite_spawn_route_mass": sum(
            isinstance(cart.get("finite_spawn_routes"), (int, float))
            and int(cart["finite_spawn_routes"]) == 0 for cart in carts
        ),
        "team_atom_mass": len(teams),
        "team_cart_objective_incidence_measure": scalar_measure(
            team.get("capture_cart_count") for team in teams
        ),
        "team_cart_capture_pair_measure": scalar_measure(
            row["measurements"].get("team_cart_capture_pair_mass") for row in rows
        ),
        "team_cart_advanceable_pair_measure": scalar_measure(
            row["measurements"].get("team_cart_advanceable_pair_mass") for row in rows
        ),
        "team_cart_nonadvanceable_pair_measure": scalar_measure(
            row["measurements"].get("team_cart_nonadvanceable_pair_mass") for row in rows
        ),
        "team_cart_spawn_unreachable_pair_measure": scalar_measure(
            row["measurements"].get("team_cart_spawn_unreachable_pair_mass") for row in rows
        ),
        "team_cart_rider_discontinuous_pair_measure": scalar_measure(
            row["measurements"].get("team_cart_rider_discontinuous_pair_mass") for row in rows
        ),
        "team_cart_path_degenerate_pair_measure": scalar_measure(
            row["measurements"].get("team_cart_path_degenerate_pair_mass") for row in rows
        ),
        "generic_spawn_measure": scalar_measure(
            row["measurements"].get("generic_spawns") for row in rows
        ),
        "original_team_labeled_spawn_measure": scalar_measure(
            row["measurements"].get("original_team_labeled_spawn_mass") for row in rows
        ),
        "residual_team_labeled_spawn_measure": scalar_measure(
            row["measurements"].get("residual_team_labeled_spawn_mass") for row in rows
        ),
        "spawn_path_clearance_measure": scalar_measure(
            row["measurements"].get("spawn_path_clearance_min") for row in rows
        ),
        "spawn_cart_origin_clearance_measure": scalar_measure(
            row["measurements"].get("spawn_cart_origin_clearance_min") for row in rows
        ),
        "spawn_cart_origin_clearance_residual_measure": scalar_measure(
            row["measurements"].get("spawn_cart_origin_clearance_residual_mass") for row in rows
        ),
        "spawn_cart_nonfinite_distance_measure": scalar_measure(
            row["measurements"].get("spawn_cart_nonfinite_distances") for row in rows
        ),
        "stock_navigation_spawn_access_map_mass": sum(
            row["measurements"].get("spawn_access_relation") == "stock_playerbot_navigation"
            for row in rows
        ),
        "spawn_cart_origin_occupancy_pair_measure": scalar_measure(
            row["measurements"].get("spawn_cart_origin_occupancy_pair_mass") for row in rows
        ),
        "cart_origin_separation_measure": scalar_measure(
            row["measurements"].get("origin_separation") for row in rows
        ),
        "nominal_end_to_end_time_ratio_measure": scalar_measure(
            row["measurements"].get("nominal_end_to_end_time_ratio") for row in rows
        ),
        "goals_minus_teams_measure": scalar_measure(
            int(row["measurements"]["goals"]) - int(row["measurements"]["teams"])
            for row in rows
            if isinstance(row["measurements"].get("goals"), (int, float))
            and isinstance(row["measurements"].get("teams"), (int, float))
        ),
        "measurement_schema_measure": scalar_measure(
            row["measurements"].get("schema") for row in rows
        ),
    }

def fabric_space_measures(rows):
    points = [
        (row, point) for row in rows
        for point in ((row.get("roofline") or {}).get("points") or ())
    ]
    nodes = [
        (row, point, name, node) for row, point in points
        for name, node in (point.get("nodes") or {}).items()
    ]
    producer_records = [
        (row, node, producer) for row in rows
        for node, producers in (
            ((row.get("roofline") or {}).get("producer_measure_records") or {}).items()
        )
        for producer in producers
    ]
    operations = [
        (row, node, producer, producer.get("operations") or {})
        for row, node, producer in producer_records
    ]
    operation_digests = defaultdict(lambda: defaultdict(int))
    for row, node, producer, operation in operations:
        labels = producer.get("labels") or {}
        atom = (
            node, labels.get("host"), labels.get("host_role"),
            operation.get("scale_model_digest"),
        )
        operation_digests[row.get("id")][atom] += 1
    return {
        "record_mass": len(rows),
        "roofline_observation_mass": sum(row.get("roofline") is not None for row in rows),
        "point_atom_mass": len(points),
        "node_atom_mass": len(nodes),
        "producer_measure_record_mass": len(producer_records),
        "configured_team_measure": scalar_measure(row.get("teams") for row in rows),
        "realized_player_measure": scalar_measure(row.get("players") for row in rows),
        "configured_cart_measure": scalar_measure(row.get("carts") for row in rows),
        "realized_scale_coordinate_measure": atom_measure(
            {
                "teams": row.get("teams"),
                "players": row.get("players"),
                "carts": row.get("carts"),
                "residual_rank": (row.get("strategy_widths") or {}).get("residual_rank"),
                "hidden_width": (row.get("strategy_widths") or {}).get("hidden_width"),
                "experts": (row.get("strategy_widths") or {}).get("experts"),
                "topk": (row.get("strategy_widths") or {}).get("topk"),
            }
            for row in rows
        ),
        "residual_rank_measure": scalar_measure(
            (row.get("strategy_widths") or {}).get("residual_rank") for row in rows
        ),
        "hidden_width_measure": scalar_measure(
            (row.get("strategy_widths") or {}).get("hidden_width") for row in rows
        ),
        "expert_count_measure": scalar_measure(
            (row.get("strategy_widths") or {}).get("experts") for row in rows
        ),
        "expert_topk_measure": scalar_measure(
            (row.get("strategy_widths") or {}).get("topk") for row in rows
        ),
        "producer_node_mass_measure": scalar_measure(
            point.get("producer_node_mass") for _, point in points
        ),
        "producer_host_mass_measure": scalar_measure(
            point.get("producer_host_mass") for _, point in points
        ),
        "producer_node_identity_measure": atom_measure(
            name for _, point in points for name in (point.get("producer_nodes") or ())
        ),
        "producer_host_identity_measure": atom_measure(
            name for _, point in points for name in (point.get("producer_hosts") or ())
        ),
        "role_node_measure": atom_measure(
            {"role": role, "node": node}
            for _, point in points
            for role, names in (point.get("role_nodes") or {}).items()
            for node in names
        ),
        "role_host_measure": atom_measure(
            {"role": role, "host": host}
            for _, point in points
            for role, names in (point.get("role_hosts") or {}).items()
            for host in names
        ),
        "required_role_distinct_node_pair_mass_measure": scalar_measure(
            point.get("required_role_distinct_node_pair_mass") for _, point in points
        ),
        "required_role_distinct_host_pair_mass_measure": scalar_measure(
            point.get("required_role_distinct_host_pair_mass") for _, point in points
        ),
        "distributed_deadline_load_measure": scalar_measure(
            point.get("maximum_deadline_load") for _, point in points
        ),
        "distributed_deadline_margin_measure": scalar_measure(
            1.0 - float(point["maximum_deadline_load"])
            for _, point in points
            if isinstance(point.get("maximum_deadline_load"), (int, float))
        ),
        "local_only_deadline_load_measure": scalar_measure(
            point.get("maximum_local_only_deadline_load") for _, point in points
        ),
        "local_only_deadline_lower_load_measure": scalar_measure(
            point.get("maximum_local_only_deadline_lower_load") for _, point in points
        ),
        "local_only_deadline_upper_load_measure": scalar_measure(
            point.get("maximum_local_only_deadline_upper_load") for _, point in points
        ),
        "local_only_deadline_lower_overrun_measure": scalar_measure(
            float(point["maximum_local_only_deadline_lower_load"]) - 1.0
            for _, point in points
            if isinstance(point.get("maximum_local_only_deadline_lower_load"), (int, float))
        ),
        "roofline_lower_fraction_measure": scalar_measure(
            node.get("roofline_lower_fraction") for _, _, _, node in nodes
        ),
        "roofline_upper_fraction_measure": scalar_measure(
            node.get("roofline_upper_fraction") for _, _, _, node in nodes
        ),
        "machine_memory_lower_fraction_measure": scalar_measure(
            node.get("machine_memory_lower_fraction") for _, _, _, node in nodes
        ),
        "machine_memory_upper_fraction_measure": scalar_measure(
            node.get("machine_memory_upper_fraction") for _, _, _, node in nodes
        ),
        "fp32_lower_gflops_s_measure": scalar_measure(
            node.get("fp32_lower_gflops_s") for _, _, _, node in nodes
        ),
        "fp32_upper_gflops_s_measure": scalar_measure(
            node.get("fp32_upper_gflops_s") for _, _, _, node in nodes
        ),
        "workload_memory_lower_gbs_measure": scalar_measure(
            node.get("workload_memory_lower_gbs") for _, _, _, node in nodes
        ),
        "workload_memory_upper_gbs_measure": scalar_measure(
            node.get("workload_memory_upper_gbs") for _, _, _, node in nodes
        ),
        "remote_request_row_measure": scalar_measure(
            node.get("remote_request_row_mass") for _, _, _, node in nodes
        ),
        "remote_output_row_measure": scalar_measure(
            node.get("remote_output_row_mass") for _, _, _, node in nodes
        ),
        "remote_output_row_fraction_measure": scalar_measure(
            node.get("remote_output_row_fraction") for _, _, _, node in nodes
        ),
        "producer_operation_scale_measure": atom_measure(
            {
                "record": row.get("id"),
                "node": node,
                "host": (producer.get("labels") or {}).get("host"),
                "role": (producer.get("labels") or {}).get("host_role"),
                "residual_rows": operation.get("residual_rows"),
                "residual_rank": operation.get("residual_rank"),
                "experts": operation.get("experts"),
                "topk": operation.get("topk"),
                "scale_model_digest": operation.get("scale_model_digest"),
                "remote_request_row_mass": operation.get("remote_request_row_mass"),
                "remote_output_row_mass": operation.get("remote_output_row_mass"),
            }
            for _, node, producer, operation in operations
        ),
        "scale_model_digest_relation_measure": atom_measure(
            {
                "record": record,
                "producer_digest_atoms": [
                    {
                        "node": node, "host": host, "role": role,
                        "scale_model_digest": digest, "mass": mass,
                    }
                    for (node, host, role, digest), mass in sorted(atoms.items(), key=repr)
                ],
            }
            for record, atoms in operation_digests.items()
        ),
    }

def fabric_measures(run_dir, records):
    supervisor = load_jsonl(os.path.join(run_dir, "supervisor.jsonl"))
    supervisor_start = next((
        row for row in reversed(supervisor)
        if row.get("event") == "supervisor_start"
    ), {})
    named_maps = sorted(supervisor_start.get("maps") or [])
    map_rows = []
    scale_rows = []
    for record in records:
        cfg = record.get("configuration") or {}
        entity = record.get("entity") or {}
        measures = entity.get("measurements") or {}
        map_rows.append({
            "source": "match",
            "id": record.get("id"),
            "map": cfg.get("map") or measures.get("map"),
            "teams": integer_coordinate(cfg.get("teams"), measures.get("teams")),
            "carts": integer_coordinate(cfg.get("carts"), measures.get("carts")),
            "measurements": measures,
            "artifact": (record.get("artifacts") or {}).get("measurements"),
            "returncode": entity.get("returncode"),
            "realization_id": entity.get("realization_id"),
            "realization_reuse_mass": entity.get("realization_reuse_mass"),
        })
        peak = (record.get("realized") or {}).get("peak_configuration") or {}
        roofline_artifact = (record.get("artifacts") or {}).get("roofline") or {}
        roofline = None
        path = roofline_artifact.get("path")
        if path and os.path.isfile(path):
            try:
                with open(path) as stream:
                    roofline = json.load(stream)
            except (OSError, json.JSONDecodeError):
                roofline = None
        execution = record.get("execution") or {}
        realized_players = peak.get("players")
        realized_carts = peak.get("carts")
        scale_rows.append({
            "id": record.get("id"),
            "teams": integer_coordinate(peak.get("teams"), cfg.get("teams")),
            "players": integer_coordinate(realized_players),
            "carts": integer_coordinate(realized_carts, cfg.get("carts")),
            "strategy_widths": cfg.get("strategy_widths") or {},
            "server_returncode": (execution.get("server") or {}).get("returncode"),
            "responder_returncode": (execution.get("responder") or {}).get("returncode"),
            "expert_returncode": (execution.get("expert") or {}).get("returncode"),
            "roofline": roofline,
            "roofline_artifact": roofline_artifact,
        })
    return {
        "maps": {
            "named": named_maps,
            "observed_names": sorted({row["map"] for row in map_rows if row["map"]}),
            "measure": map_space_measures(named_maps, map_rows),
            "observations": map_rows,
        },
        "fabric": {
            "measure": fabric_space_measures(scale_rows),
            "observations": scale_rows,
        },
    }

def summarize(run_dir):
    records = load_jsonl(os.path.join(run_dir, "matches.jsonl"))
    contests = []
    measurements = defaultdict(list)
    outcomes = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    paired_metrics = defaultdict(list)
    blocks = defaultdict(list)
    record_observations = []
    record_rows = []
    for record in records:
        cfg = record.get("configuration") or {}
        rows = load_jsonl(telemetry_path(record))
        record_rows.append((record, rows))
        metrics = arm_metrics(record, rows)
        team_arms = cfg.get("team_policy_arms") or []
        unique_arms = tuple(dict.fromkeys(team_arms))
        rounds = round_results(rows)
        execution = record.get("execution") or {}
        perturbation = (record.get("perturbation") or {}).get("name", "unknown")
        latest = rows[-1] if rows else {}
        matrix_fusion_intervention = matrix_fusion_intervention_measures([
            intervention for row in rows
            if (intervention := (row.get("model") or {}).get("matrix_fusion_intervention")) is not None
        ])
        record_observations.append({
            "id": record.get("id"),
            "telemetry_rows": len(rows),
            "round_outcomes": rounds,
            "team_policy_arms": team_arms,
            "measured_policy_arms": sorted(metrics),
            "strategy_widths": cfg.get("strategy_widths"),
            "responder_restarts": len(execution.get("responder_restarts") or []),
            "expert_restarts": len(execution.get("expert_restarts") or []),
            "server_returncode": (execution.get("server") or {}).get("returncode"),
            "responder_returncode": (execution.get("responder") or {}).get("returncode"),
            "expert_returncode": (execution.get("expert") or {}).get("returncode"),
            "policy_provenance": latest.get("policy_provenance"),
            "scale_operator": ((latest.get("model") or {}).get("scale_operator")),
            "matrix_fusion_intervention": matrix_fusion_intervention,
            "environment": environment_metrics(rows),
            "work_samples": sum(isinstance(row.get("work"), dict) for row in rows),
            "strategy_measure_records": strategy_measure_records(execution),
        })
        for arm, values in metrics.items():
            measurements[arm].append({"perturbation": perturbation, **values})
        if len(unique_arms) == 2:
            pair = ordered_pair(*unique_arms)
            blocks[block_key(record, pair)].append({
                "record": record,
                "metrics": metrics,
                "rounds": rounds,
                "team_arms": team_arms,
                "provenance": latest.get("policy_provenance") or {},
                "leg": int(cfg.get("leg") or 0),
            })
    block_observations = []
    for key, legs in sorted(blocks.items(), key=lambda item: repr(item[0])):
        by_leg = defaultdict(list)
        for leg in legs:
            by_leg[leg["leg"]].append(leg)
        ids = [leg["record"].get("id") for leg in legs]
        row = {
            "arms": list(key[0]),
            "perturbation": key[1],
            "records": ids,
            "leg_multiplicity": {str(name): len(values) for name, values in sorted(by_leg.items())},
            "rounds": sum(len(leg["rounds"]) for leg in legs),
            "round_winner_coordinate_mass": sum(
                winner is not None for leg in legs for _, winner, _ in leg["rounds"]
            ),
            "round_time_coordinate_mass": sum(
                event_time is not None for leg in legs for _, _, event_time in leg["rounds"]
            ),
            "paired_score": None,
            "leg_pair_mass": 0,
            "unpaired_leg_mass": 0,
            "mirrored_leg_pair_mass": 0,
            "measured_mirrored_leg_pair_mass": 0,
            "scored_mirrored_leg_pair_mass": 0,
            "team_slot_equality_measure": scalar_measure([]),
        }
        first_group = by_leg.get(1) or []
        second_group = by_leg.get(2) or []
        pair_atoms = []
        for first_leg, second_leg in zip(first_group, second_group):
            first_arms, second_arms = first_leg["team_arms"], second_leg["team_arms"]
            equalities = None
            if len(first_arms) == len(second_arms) and first_arms:
                equalities = sum(
                    left == right for left, right in zip(first_arms, second_arms)
                )
            scores = [
                record_score(leg["rounds"], leg["team_arms"], key[0][0])
                for leg in (first_leg, second_leg)
            ]
            pair_atoms.append({
                "first": first_leg,
                "second": second_leg,
                "team_slot_equalities": equalities,
                "scores": scores,
            })
        row["leg_pair_mass"] = len(pair_atoms)
        row["unpaired_leg_mass"] = abs(len(first_group) - len(second_group))
        row["team_slot_equality_measure"] = scalar_measure(
            atom["team_slot_equalities"] for atom in pair_atoms
        )
        mirrored_atoms = [
            atom for atom in pair_atoms if atom["team_slot_equalities"] == 0
        ]
        measured_atoms = [
            atom for atom in mirrored_atoms
            if all(
                key[0][0] in leg["metrics"] and key[0][1] in leg["metrics"]
                for leg in (atom["first"], atom["second"])
            )
        ]
        scored_atoms = [
            atom for atom in mirrored_atoms if all(score is not None for score in atom["scores"])
        ]
        row["mirrored_leg_pair_mass"] = len(mirrored_atoms)
        row["measured_mirrored_leg_pair_mass"] = len(measured_atoms)
        row["scored_mirrored_leg_pair_mass"] = len(scored_atoms)
        for atom in measured_atoms:
            block_lifts = []
            for leg in (atom["first"], atom["second"]):
                left, right = leg["metrics"][key[0][0]], leg["metrics"][key[0][1]]
                numeric = set(left) & set(right)
                block_lifts.append({
                    name: float(left[name] - right[name])
                    for name in numeric
                    if isinstance(left[name], (int, float))
                    and isinstance(right[name], (int, float))
                    and name != "players"
                })
            paired_metrics[key[0]].append({
                "perturbation": key[1],
                **{
                    name: float(np.mean([value[name] for value in block_lifts if name in value]))
                    for name in sorted({name for value in block_lifts for name in value})
                },
            })
        block_scores = []
        for atom in scored_atoms:
            first_leg, second_leg = atom["first"], atom["second"]
            scores = atom["scores"]
            first, second = key[0]
            block_score = float(np.mean(scores))
            block_scores.append(block_score)
            atom_ids = [first_leg["record"].get("id"), second_leg["record"].get("id")]
            contests.append({
                "arms": key[0],
                "realizations": tuple(
                    realization_coordinate(
                        arm, (first_leg["provenance"], second_leg["provenance"]),
                    )
                    for arm in key[0]
                ),
                "score": block_score,
                "rounds": len(first_leg["rounds"]) + len(second_leg["rounds"]),
                "records": atom_ids,
                "perturbation": key[1],
            })
            outcomes[first][key[1]][0] += block_score
            outcomes[first][key[1]][1] += 1
            outcomes[second][key[1]][0] += 1 - block_score
            outcomes[second][key[1]][1] += 1
        if block_scores:
            row["paired_score"] = float(np.mean(block_scores))
        block_observations.append(row)
    rating_measure = elo(contests)
    realization_rating_measure = realization_elo(contests)
    ratings = rating_measure["ratings"]
    pairs = pairwise_summary(contests, paired_metrics)
    arms = {}
    for arm in sorted(measurements):
        rows = measurements[arm]
        numeric = sorted({
            key for row in rows for key, value in row.items()
            if isinstance(value, (int, float)) and key != "players"
        })
        per_perturbation = {
            perturbation: score / count
            for perturbation, (score, count) in outcomes[arm].items()
            if count
        }
        values = list(per_perturbation.values())
        behavior = metric_measures(rows)
        arms[arm] = {
            "elo": ratings.get(arm),
            "matches": len(rows),
            "means": {
                key: behavior["coordinates"][key]["mean"]
                for key in numeric
            },
            "win_expectation_by_perturbation": per_perturbation,
            "perturbation_minimum": min(values) if values else None,
            "perturbation_variance": float(np.var(values)) if values else None,
            "behavior_measure": behavior,
        }
    return {
        "run_dir": os.path.abspath(run_dir),
        "record_count": len(records),
        "paired_block_count": len(contests),
        "paired_round_count": sum(match["rounds"] for match in contests),
        "record_observations": record_observations,
        "block_observations": block_observations,
        "ratings": ratings,
        "rating_measure": rating_measure,
        "directed_elo_coordinates": directed_elo_coordinates(rating_measure, (
            ("matrix_fusion", "initial_policy"),
            ("matrix_fusion", "participant_fusion_ablated"),
            ("matrix_fusion", "residual_fusion_ablated"),
            ("matrix_fusion", "ffn"),
            ("ffn", "linear"),
            ("linear", "default"),
        )),
        "realization_rating_measure": realization_rating_measure,
        "arms": arms,
        "pairwise": pairs,
        "optimization_measures": optimization_measures(record_rows),
        "fabric_measures": fabric_measures(run_dir, records),
    }

def write_report(path, result):
    path = os.path.abspath(os.path.expanduser(path))
    temporary = path + ".new"
    with open(temporary, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    return path

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = summarize(args.run_dir)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        write_report(args.output, result)
    print(encoded)

if __name__ == "__main__":
    main()
