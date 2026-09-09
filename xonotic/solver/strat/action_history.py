from collections import defaultdict
import json
import os
import tempfile
import time

import numpy as np

from payload.tools.strategy_io_schema import BOT_CONFIGURATION_COLUMNS, OBS, OBS_OUTCOME_COUNTER_COLUMNS, STRATEGY_DEADLINE_S
from .inputs import participant_successors
from .row_window import RowWindow
from .game_value import GAME_CONTRACT, role_rewards, winner


def observation_clock(row):
    times = [item["engine_time"] for item in row.get("assignments", ())
             if isinstance(item.get("engine_time"), (int, float)) and np.isfinite(item["engine_time"])]
    episode = row.get('episode_id', ((row.get("game_value") or {}).get("state") or {}).get("episode"))
    return episode, float(np.median(times)) if times else None


def observed_interval(previous, current):
    return (max(0.0, current[1] - previous[1])
            if previous is not None and current[0] == previous[0]
            and current[1] is not None and previous[1] is not None else 0.0)


def round_results(rows):
    seen, results = set(), []
    for row in rows:
        events = row.get("realized_events", ()) if (row.get("game_value") or {}).get("contract") == GAME_CONTRACT else ()
        for event in events:
            if event.get('episode_id') != row.get('episode_id'):
                continue
            if event.get("kind") in ("score_win", "tie"):
                actor, stamp = event.get("actor_team"), event.get("time")
                result = (event["kind"], int(actor) if isinstance(actor, (int, float)) and np.isfinite(actor) else None,
                          float(stamp) if isinstance(stamp, (int, float)) and np.isfinite(stamp) else None)
                key = (observation_clock(row)[0], result)
                if key not in seen:
                    seen.add(key)
                    results.append(result)
    return results


class ExecutionEvaluation:
    def __init__(self):
        self.previous = None
        self.versions = {}
        self.exposure = defaultdict(float)
        self.team_exposure = defaultdict(float)
        self.rounds = []
        self.seen_outcomes = set()
        self.unattributed_seconds = self.changing_source_seconds = 0.0
        self.unlabelled_seconds = 0.0
        self.prior_sources = {}
        self.finished = False

    def ingest(self, row):
        current = observation_clock(row)
        elapsed = observed_interval(self.previous, current)
        reset = self.previous is not None and (current[0] != self.previous[0] or
                (current[1] is not None and self.previous[1] is not None and current[1] < self.previous[1]))
        if reset:
            self.unlabelled_seconds += sum(self.team_exposure.values())
            self.exposure.clear()
            self.team_exposure.clear()
            self.prior_sources.clear()
            self.seen_outcomes.clear()
            self.finished = False
        elapsed = 0.0 if self.finished else elapsed
        self.previous = current
        sources = {}
        for item in row.get("assignments", ()):
            participant, team = item["edict"], item["team"]
            self.team_exposure[team] += elapsed
            source = (item.get("applied_policy_arm"), item.get("applied_behavior"),
                      item.get("applied_policy_lineage"), item.get("applied_policy_updates"))
            sources[participant] = source
            self.changing_source_seconds += elapsed * (self.prior_sources.get(participant) != source)
            if source[0] is None or not item.get("applied_state_current"):
                self.unattributed_seconds += elapsed
            else:
                key = json.dumps(source, separators=(",", ":"))
                atom = self.versions.setdefault(key, {"arm": source[0], "behavior": source[1], "lineage": source[2],
                    "updates": source[3], "observed_player_seconds": 0.0, "first_executions": 0})
                atom["observed_player_seconds"] += elapsed
                atom["first_executions"] += int(bool(item.get("first_execution")))
                self.exposure[(team, key)] += elapsed
        self.prior_sources = sources
        for outcome in round_results([row]):
            if outcome not in self.seen_outcomes:
                self.seen_outcomes.add(outcome)
                self.rounds.append({"episode": current[0], "kind": outcome[0], "winning_team": outcome[1], "time": outcome[2],
                    "contributors": [{"team": team, "policy": key, "observed_player_seconds": seconds,
                                      "team_exposure_share": seconds / self.team_exposure[team],
                                      "won": team == outcome[1]} for (team, key), seconds in self.exposure.items() if seconds > 0]})
                self.exposure.clear()
                self.team_exposure.clear()
                self.finished = True

    def report(self):
        return {"contract": "observed-execution-history-v1", "versions": self.versions, "rounds": self.rounds,
                "unattributed_player_seconds": self.unattributed_seconds,
                "source_change_interval_seconds": self.changing_source_seconds,
                "unlabelled_player_seconds": self.unlabelled_seconds + sum(self.team_exposure.values())}


def execution_evaluation(rows):
    result = ExecutionEvaluation()
    for row in rows:
        result.ingest(row)
    return result.report()


def source_features(frame):
    parts, labels = [], []
    for name, value in zip(frame._fields, frame):
        value = np.asarray(value)
        parts.append(value.reshape(-1))
        labels.extend(name + '[' + ','.join(map(str, index)) + ']' for index in np.ndindex(value.shape))
    return np.concatenate(parts), tuple(labels)


def view_observation(frame, row):
    from .state_steering import state_labels
    layout = frame.layout[row]
    present = np.flatnonzero(layout[:, 3:].reshape(-1) > 0)
    labels = state_labels(layout)
    return frame.state[row, present], [labels[index] for index in present]


def retained_source_rows(rows):
    return {participant: {key: value for key, value in row.items()
        if key not in ('features', 'server_state', 'feature_labels', 'state_labels')}
        for participant, row in rows.items()}


class ActionHistory:
    def __init__(self, capacity, max_bytes=256 << 20, directory=None):
        self.requests = RowWindow(capacity, lambda entry: len(entry["rows"]))
        self.max_bytes = int(max_bytes)
        self.directory = directory
        self.previous = None
        self.attribution = {}
        self.first_executions = set()

    def clear(self):
        self.requests.clear()
        self.previous = None
        self.attribution = {}
        self.first_executions.clear()

    def export_state(self):
        requests = {key: value for key, value in vars(self.requests).items() if key != "mass"}
        requests['entries'] = {sequence: {**entry, 'rows': retained_source_rows(entry['rows'])}
            for sequence, entry in self.requests.items()}
        return {"requests": requests,
                "previous": self.previous, "attribution": self.attribution,
                "first_executions": self.first_executions}

    def restore_state(self, state):
        self.requests.__dict__.update(state["requests"])
        self.previous = state["previous"]
        self.attribution = state["attribution"]
        self.first_executions = state["first_executions"]
        for entry in self.requests.values():
            self.restore_features(entry)

    @staticmethod
    def restore_features(entry):
        source = entry['state']
        if source is not None and hasattr(source.get('frame'), 'chorus'):
            features, feature_labels = source_features(source['frame'].chorus)
            for index, participant in enumerate(source['participant_ids']):
                observed, labels = view_observation(source['frame'].chorus, index)
                entry['rows'][int(participant)].update(features=features,
                    server_state=observed, feature_labels=feature_labels, state_labels=labels)
            entry['_bytes'] = ActionHistory.entry_bytes(entry['rows'], source)

    def get(self, sequence, default=None):
        entry = self.requests.get(sequence)
        return default if entry is None else entry["rows"]

    def items(self):
        return ((sequence, entry["rows"]) for sequence, entry in self.requests.items() if '_spill' not in entry)

    def materialize(self, entry):
        from .checkpoint_state import unpack_state
        path = entry.get('_spill')
        if path is not None:
            with np.load(path, allow_pickle=False) as payload:
                stored = unpack_state(payload)
            entry.update(state=stored['state'], rows=stored['rows'], _bytes=stored['_bytes'])
            entry.pop('_spill')
            self.restore_features(entry)
        return entry

    def spill(self, entry):
        from .checkpoint_state import atomic_save, pack_state
        if self.directory is None:
            self.directory = tempfile.mkdtemp(prefix='mesh-action-history-')
        path = os.path.join(self.directory, str(time.time_ns()) + '.npz')
        stored = {key: entry[key] for key in ('state', 'rows', '_bytes')}
        stored['rows'] = retained_source_rows(entry['rows'])
        atomic_save(path, pack_state(stored))
        entry['_spill'] = path
        entry['state'] = {key: value for key, value in entry['state'].items() if key not in ('frame', 'velocity')}
        entry['rows'] = {participant: {key: value for key, value in row.items() if key not in
            ('velocity', 'residual', 'features', 'server_state', 'feature_labels', 'state_labels')}
            for participant, row in entry['rows'].items()}
        entry['_bytes'] = sum(value.nbytes for value in entry['state'].values() if isinstance(value, np.ndarray))

    def collect_spills(self):
        if self.directory is not None and os.path.isdir(self.directory):
            retained = {entry['_spill'] for entry in self.requests.values() if '_spill' in entry}
            for name in os.listdir(self.directory):
                path = os.path.join(self.directory, name)
                if name.endswith('.npz') and path not in retained:
                    os.unlink(path)

    @staticmethod
    def entry_bytes(rows, state):
        nbytes = getattr(state['frame'], 'nbytes', 0) if state is not None else 0
        arrays, labels = {}, {}
        if state is not None:
            arrays.update((id(value), value) for value in state.values() if isinstance(value, np.ndarray))
        for row in rows.values():
            arrays.update((id(value), value) for value in row.values() if isinstance(value, np.ndarray))
            labels.update((id(value), value) for name, value in row.items() if name.endswith('_labels'))
        nbytes += sum(value.nbytes for value in arrays.values())
        nbytes += sum(len(name) + 64 for names in labels.values() for name in names)
        return nbytes

    def put(self, sequence, rows, retain=(), state=None):
        size = len(rows)
        self.requests.put(sequence, {"rows": rows, "state": state, "consumed": set(),
                                    "returns": np.zeros(size, dtype=np.float32),
                                    "discount": np.ones(size, dtype=np.float32), '_bytes': self.entry_bytes(rows, state)}, tuple(self.requests.entries))
        self.previous = state
        retained = {*retain, sequence}
        while self.nbytes > self.max_bytes:
            oldest = next((key for key, entry in self.requests.items() if key not in retained and '_spill' not in entry), None)
            if oldest is None:
                break
            self.spill(self.requests.get(oldest))

    @property
    def nbytes(self):
        return sum(entry.get('_bytes', 0) for entry in self.requests.values())

    def measure(self):
        return {**self.requests.measure(), **self.attribution, 'bytes': self.nbytes, 'byte_capacity': self.max_bytes,
                'spilled_requests': sum('_spill' in entry for entry in self.requests.values()),
                'retained_over_capacity_bytes': max(0, self.nbytes - self.max_bytes)}

    def advance(self, tick, frame, participant_ids, rows, application_sequences, learners, winning_team=None):
        records = {arm: [] for arm in learners}
        self.first_executions.clear()
        self.attribution = {}
        previous = self.previous
        if previous is None or previous["snapshot"].finished:
            return records
        now = float(np.median(rows[:, OBS["ENGINE_TIME"]]))
        elapsed = max(0.0, now - previous["engine_time"])
        before_role = winner(previous["context"], previous["snapshot"])
        after_role = tick.game_value.projected_role
        gamma = next(iter(learners.values())).gamma if learners else 0.95
        team_context = type(previous["context"])(previous["context"].teams, previous["context"].teams)
        rewards = role_rewards(team_context, previous["snapshot"], tick.cartstate)
        for entry in self.requests.values():
            state = entry["state"]
            if state is not None:
                teams = np.asarray(state["context"].team_of)
                entry["returns"] += entry["discount"] * rewards[teams]
                entry["discount"] *= gamma ** (elapsed / STRATEGY_DEADLINE_S) * ((teams == before_role) == (teams == after_role))
        groups = {}
        missing = repeated = 0
        for row, applied_sequence in zip(rows, application_sequences):
            participant = int(row[OBS["ID"]])
            sequence = int(applied_sequence)
            entry = self.requests.get(sequence)
            decision = None if entry is None else entry["rows"].get(participant)
            matched = decision is not None
            if matched and entry["state"] is not None and row[OBS["CONTROL"]] >= 0.5:
                if participant in entry["consumed"]:
                    repeated += 1
                else:
                    groups.setdefault(sequence, set()).add(participant)
            else:
                missing += int(sequence > 0 and row[OBS["CONTROL"]] >= 0.5)

        def transition(state, arm, selected, value_weight):
            state = self.materialize(self.requests.get(state['request_seq']))['state']
            learning = learners[arm]
            ids = state["participant_ids"]
            successors = participant_successors(frame.chorus, state['frame'].chorus)
            applied = np.isin(ids, list(selected)) & (successors >= 0)
            source = self.requests.get(state["request_seq"])
            age = max(0.0, now - state["engine_time"])
            discount = (learning.bootstrap_gamma ** (age / STRATEGY_DEADLINE_S)
                        if arm == "terminal_win" or value_weight else source["discount"])
            return {
                "context": state["context"], "frame": state["frame"], "next_frame": frame,
                "snapshot": state["snapshot"], "next_snapshot": tick.cartstate,
                "velocity": state["velocity"],
                "behavior_logp": state["behavior_logp"], "successor_rows": successors,
                "train_mask": np.ones(len(ids), dtype=bool),
                "actor_mask": applied & (state["behavior_arms"] == arm),
                "behavior_updates": state["policy_updates"][arm],
                "behavior_arms": state["behavior_arms"], "behavior_versions": state["policy_updates"],
                "sparse_return": (role_rewards(state["context"], state["snapshot"], tick.cartstate)
                                  if value_weight else source["returns"].copy()),
                "bootstrap_discount": discount, "winning_team": winning_team,
                "value_weight": value_weight, "request_seq": state["request_seq"],
                "interval_seconds": age,
            }

        for arm in learners:
            records[arm].append(transition(previous, arm, groups.get(previous["request_seq"], ()), 1.0))
            for sequence, selected in groups.items():
                if sequence != previous["request_seq"]:
                    records[arm].append(transition(self.requests.get(sequence)["state"], arm, selected, 0.0))
        for sequence, selected in groups.items():
            self.requests.get(sequence)["consumed"].update(selected)
            self.first_executions.update((sequence, participant) for participant in selected)
        self.attribution = {
            "executed_source_groups": len(groups), "first_execution_rows": sum(map(len, groups.values())),
            "repeated_execution_rows": repeated, "unmatched_execution_rows": missing,
            "delayed_execution_rows": sum(len(selected) for seq, selected in groups.items() if seq != previous["request_seq"]),
            "value_intervals": 1, "interval_seconds": elapsed,
        }
        observed = dict(zip(map(int, participant_ids), map(int, application_sequences)))
        for sequence, entry in tuple(self.requests.items()):
            if entry['state'] is not previous and all(observed.get(int(player), sequence + 1) > sequence for player in entry['rows']):
                self.requests.entries.pop(sequence)
                self.requests.row_mass -= len(entry['rows'])
                self.requests.evicted_row_mass += len(entry['rows'])
        return records

    def assignments(self, rows, application_sequences, request_seq, velocity,
                    residual, behavior_logp, arms, versions, provenance):
        assignments = []
        decisions = {}
        for index, row in enumerate(rows):
            participant = int(row[OBS["ID"]])
            sequence = int(application_sequences[index])
            applied = self.get(sequence, {}).get(participant)
            arm = str(arms[index])
            source = provenance.get(arm, {})
            decision = {
                "policy_arm": arm, "behavior": arm,
                "policy_updates": versions.get(arm),
                "policy_lineage": source.get("lineage_initial_sha256"),
                "velocity": velocity[index].copy(), "residual": residual[index].copy(),
                "behavior_logp": float(behavior_logp[index]),
            }
            decisions[participant] = decision
            assignments.append({
                "edict": participant, "team": int(row[OBS["TEAM"]]),
                "controller": "bot" if row[OBS["CONTROL"]] >= 0.5 else "human",
                "engine_time": float(row[OBS["ENGINE_TIME"]]),
                "spawn_time": float(row[OBS['SPAWN_TIME']]),
                "bot_configuration": {name.lower(): float(row[OBS[name]]) for name in BOT_CONFIGURATION_COLUMNS}
                    if row[OBS['BOT_CONFIG_SCHEMA']] == 1 else None,
                "bot_configuration_schema": int(row[OBS['BOT_CONFIG_SCHEMA']]),
                "policy_arm": arm, "behavior": arm, "request_seq": request_seq,
                "policy_updates": versions.get(arm),
                "state_width": int(velocity.shape[1]),
                "velocity_l2": float(np.linalg.norm(velocity[index])),
                "residual_l2": float(np.linalg.norm(residual[index])),
                "behavior_logp": float(behavior_logp[index]),
                "applied_response_seq": sequence,
                "applied_state_current": applied is not None,
                "applied_policy_arm": None if applied is None else applied["policy_arm"],
                "applied_behavior": None if applied is None else applied["behavior"],
                "applied_policy_updates": None if applied is None else applied["policy_updates"],
                "applied_policy_lineage": None if applied is None else applied["policy_lineage"],
                "first_execution": (sequence, participant) in self.first_executions,
                "outcome_totals": {name: float(row[OBS[column]])
                                   for name, column in OBS_OUTCOME_COUNTER_COLUMNS},
            })
        return assignments, decisions
