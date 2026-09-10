from __future__ import annotations

import json
import os
import time
from collections import deque
from contextlib import nullcontext

from . import tensor as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten, tree_map

from .checkpoint_audit import audit_checkpoint
from .checkpoint_state import (
    ARCH_KEY, ARCH_SPEC_KEY, POLICY_KEY, POLICY_VERSION_KEY, RNG_KEY,
    REWARD_CONTRACT_KEY, LINEAGE_INITIAL_KEY, POLICY_VERSIONS, architecture_fingerprint,
    architecture_spec, tensor_tree_measurement, whole_tensor_tree,
    Payload, atomic_save, checkpoint_metadata, checkpoint_parameters, open_checkpoint,
)
from .game_value import reward_fingerprint, winner
from .policy_contract import POLICY_ACTOR_WEIGHT, POLICY_MOE_BALANCE_WEIGHT, POLICY_RATIO_CLIP, POLICY_ROLLOUT_IS_THRESHOLD, POLICY_LOG_RATIO_BOUND
from .policy_math import clipped_policy_surrogate, clip_grad_norm, rollout_importance_weights

LOSS_WEIGHTS = {
    "actor":  POLICY_ACTOR_WEIGHT,
    "winnie": 0.5,
    "lou":    0.5,
    "moe_balance": POLICY_MOE_BALANCE_WEIGHT,
}

OPTIMIZER_WEIGHT_DECAY = 1e-4

METRIC_NAMES = ("loss_pg", "loss_w", "loss_l", "importance_mean",
    "advantage", "advantage_importance_weighted", "advantage_w", "advantage_l",
    "reward_w", "reward_l", "winner_rows", "loser_rows", "role_change_fraction",
    "state_entropy", "fresh_value_target_variance", "loss_moe_balance")
ACTOR_METRIC_INDICES = [METRIC_NAMES.index(name) for name in
    ("loss_pg", "importance_mean", "state_entropy")]

from .replay import Replay
from .strategy import strategy, logp_of
from .execution import PolicyProgram, InputArena
from .storage import TensorStorage

class OnlineLearner:
    def __init__(
        self,
        wally,
        *,
        learning_rate: float = 3e-4,
        gradient_clip: float = 1.0,
        gamma: float = 0.95,
        checkpoint=None,
        load_checkpoint=None,
        replay_capacity: int = 1024,
        replay_memory_mb: float = 256.0,
        replay_precision: str = "float32",
        replay_batch: int = 8,
        seed: int = 20260831,
        policy_forward=strategy,
        policy_arm: str = "matrix_fusion",
        match_metadata=None,
        replay_weight=0.5,
    ):
        self.wally = wally
        self.gamma = float(gamma)
        self.gradient_clip = abs(float(gradient_clip))
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=OPTIMIZER_WEIGHT_DECAY)
        self.optimizer.init(self.wally.trainable_parameters())
        self.checkpoint = checkpoint
        self.updates = 0
        self.replay = Replay(replay_capacity, int(replay_memory_mb * (1 << 20)))
        self.episode = Replay(replay_capacity, int(replay_memory_mb * (1 << 20)))
        self.episode_id = None
        self.awaiting_outcomes = {}
        self.outcomes = {}
        self.match_metadata = dict(match_metadata or {})
        self.replay_weight = float(replay_weight)
        self.completed_episodes = 0
        self.truncated_episodes = 0
        self.replay_precision = replay_precision
        self.replay_batch = max(1, int(replay_batch))
        self.rng = np.random.default_rng(seed)
        self.transitions = 0
        self.gradient_steps = 0
        self.rebuild_seconds = 0.0
        self.rebuild_calls = 0
        self.forward_seconds = 0.0
        self.ratios = deque(maxlen=256)
        self._ratio_sink = []
        self.policy_forward = policy_forward
        self.policy_arm = policy_arm
        self.bootstrap_gamma = 1.0 if policy_arm == "terminal_win" else self.gamma
        self.training_contract = {"update_schedule": "state-rate-application-v1", "gradient_steps_per_group": 1, "outcome_gradient_steps": 0, "gamma": self.gamma, "replay_weight": self.replay_weight, "replay_batch": self.replay_batch, "learning_rate": learning_rate, "gradient_clip": self.gradient_clip, "loss_weights": LOSS_WEIGHTS, "action_representation": "full_state_rate", "integrator": "exact_exponential_relaxation", "historical_target": "observed_behavior_return", "actor_source": "own_first_observed_state_application", "historical_actor_weight": 0.0,
                                  "actor_objective": "decoupled_ppo", "proximal_policy": "reuse_train_logp", "rollout_is": "per_action", "rollout_is_threshold": POLICY_ROLLOUT_IS_THRESHOLD, "rollout_is_batch_normalize": False, "rollout_rs": None, "log_ratio_bound": POLICY_LOG_RATIO_BOUND,
                                  "optimizer": "AdamW", "weight_decay": OPTIMIZER_WEIGHT_DECAY,
                                  "sampler": "philox4x32-10_box_muller", "sampling_key_schedule": "uint64_counter"}
        self.schedule_start_updates = 0
        self.reward_fingerprint = reward_fingerprint(policy_arm)
        self.architecture = architecture_fingerprint(self.wally)
        self.loaded_weight_mass = 0
        self.live_weight_mass = len(tree_flatten(self.wally.parameters()))
        self.loaded_optimizer_moment_mass = 0
        self.live_optimizer_moment_mass = 0
        self.optimizer_moment_measurement = {}
        self.initial_checkpoint_sha256 = None
        self.continuation = {"mode": "fresh", "source": None, "source_updates": None, "exception": None}
        source = load_checkpoint or checkpoint
        if source is not None and os.path.exists(source):
            self._load_full(source)
        self.program = PolicyProgram(self.wally, self.policy_forward, learner=self)
        self.input_arena = InputArena()
        self.item_storage = TensorStorage()
        self.loss_traces = self.update_traces = self.accumulation_traces = 0

    def _load_full(self, source, payload=None):
        initial_parameters = tree_flatten(self.wally.parameters())
        initial_optimizer = tree_flatten(self.optimizer.state)
        state = {}
        audit = {}
        weights = {}
        source_metadata = {}
        error = None
        try:
            with (open_checkpoint(source) if payload is None else nullcontext(Payload(payload))) as data:
                source_metadata = checkpoint_metadata(data)
                source_weights = [(key, data[key]) for key in data.files if not key.startswith("__")]
                weights = tensor_tree_measurement(initial_parameters, source_weights)
                audit = audit_checkpoint(data)
                if audit["nonfinite_arrays"]:
                    raise ValueError(f"nonfinite checkpoint state: {audit['groups']}")
                parameters, weights = checkpoint_parameters(self.wally, source_metadata, source_weights,
                    self.policy_arm, self.reward_fingerprint, training=True)
                moments, measurement = whole_tensor_tree(
                    initial_optimizer,
                    [(key[7:], data[key]) for key in data.files if key.startswith("__opt__")],
                )
                replay = Replay(self.replay.capacity, self.replay.max_bytes)
                if not replay.restore_payload(data):
                    raise ValueError("checkpoint has no replay state")
                episode = Replay(self.episode.capacity, self.episode.max_bytes)
                episode.restore_payload(data, "__episode__")
                episodes = json.loads(str(data["__episodes__"])) if "__episodes__" in data.files else {}
                awaiting = {}
                for index, identity in enumerate(episodes.get("awaiting", ())):
                    pool = Replay(self.episode.capacity, self.episode.max_bytes)
                    pool.restore_payload(data, f"__awaiting_{index}__")
                    awaiting[identity] = pool
                rng = np.random.default_rng()
                rng.bit_generator.state = json.loads(str(data[RNG_KEY]))
                counters = {name: int(data["__" + name + "__"]) for name in (
                    "updates", "transitions", "gradient_steps", "completed_episodes", "truncated_episodes",
                )}
                source_training = json.loads(str(data["__training_contract__"]))
                boundary = (int(data["__schedule_start_updates__"])
                            if source_training.get("update_schedule") == self.training_contract["update_schedule"]
                            else counters["updates"])
                lineage = str(data[LINEAGE_INITIAL_KEY]) if LINEAGE_INITIAL_KEY in data.files else None
            self.wally.load_weights(tree_flatten(parameters), strict=True)
            self.optimizer.state.clear()
            self.optimizer.state.update(moments)
            self.replay = replay
            self.episode = episode
            self.episode_id = episodes.get("current")
            self.awaiting_outcomes = awaiting
            self.outcomes = episodes.get("outcomes", {})
            self.rng = rng
            for name, value in counters.items():
                setattr(self, name, value)
            self.schedule_start_updates = boundary
            self.initial_checkpoint_sha256 = lineage
            self.loaded_weight_mass = weights["source_mass"]
            self.loaded_optimizer_moment_mass = measurement["source_mass"]
            self.optimizer_moment_measurement = measurement
            state = {"source_training_contract": source_training}
        except Exception as exception:
            self.wally.load_weights(initial_parameters, strict=True)
            self.optimizer.state.clear()
            self.optimizer.state.update(tree_unflatten(initial_optimizer))
            error = f"{type(exception).__name__}: {exception}"
        self.live_optimizer_moment_mass = len(initial_optimizer)
        self.continuation = {
            "mode": "resumed" if error is None else "fresh",
            "source": source, "source_updates": audit.get("source_updates"),
            "exception": error, "audit": audit,
        }
        for name, value in {
            **source_metadata,
            "source_weight_mass": weights.get("source_mass", 0),
            "composable_weight_mass": weights.get("composable_mass", 0),
            "source_only_weight_mass": weights.get("source_only_mass", 0),
            "live_only_weight_mass": weights.get("live_only_mass", self.live_weight_mass),
            "shape_difference_mass": weights.get("shape_difference_mass", 0),
            "live_architecture": self.architecture,
            "path": source, "loaded_weight_mass": self.loaded_weight_mass,
            "live_weight_mass": self.live_weight_mass,
            "nonfinite_weight_mass": audit.get("groups", {}).get("parameters", {}).get("nonfinite_arrays", 0),
            "load_exception": error, "updates": self.updates,
            "lineage_initial_sha256": self.initial_checkpoint_sha256,
            "continuation": self.continuation,
        }.items():
            setattr(self.wally, "checkpoint_" + name, value)
        print(json.dumps({
            "event": "online_checkpoint_measurement", "path": source,
            "loaded_weight_mass": self.loaded_weight_mass, "live_weight_mass": self.live_weight_mass,
            "loaded_optimizer_moment_mass": self.loaded_optimizer_moment_mass,
            "live_optimizer_moment_mass": self.live_optimizer_moment_mass,
            "updates": self.updates, "transitions": self.transitions,
            "replay_atom_mass": len(self.replay), "replay_bytes": self.replay.nbytes,
            "training_contract": self.training_contract,
            "schedule_start_updates": self.schedule_start_updates,
            "continuation": self.continuation, **state,
        }), flush=True)

    def transition(
        self,
        context,
        frame,
        next_frame,
        snapshot,
        next_snapshot,
        velocity,
        behavior_logp,
        train_mask,
        sparse_return,
        bootstrap_discount,
        actor_mask,
        behavior_updates,
        behavior_arms,
        behavior_versions,
        winning_team,
        successor_rows,
        value_weight,
        request_seq,
        interval_seconds,
    ):
        players = len(frame.chorus.state)
        reward = np.asarray(sparse_return)
        before_winner = winner(context, snapshot)
        after_winner = winner(context, next_snapshot)
        teams = np.asarray(context.team_of, dtype=np.int64)
        if self.policy_arm == "terminal_win":
            reward = np.zeros(players, dtype=np.float32) if winning_team is None else ((winning_team > 0) & (teams == winning_team - 1)).astype(np.float32)
        discount = np.array(np.broadcast_to(
            bootstrap_discount,
            (players,),
        ), dtype=np.float32, copy=True)
        discount *= winning_team is None
        successors = np.asarray(successor_rows, dtype=np.int64)
        discount *= successors >= 0
        return {
            "frame_in": frame,
            "value_weight": float(value_weight),
            "request_seq": request_seq,
            "interval_seconds": interval_seconds,
            "frame_out": next_frame,
            "successor_rows": successors,
            "velocity": np.asarray(velocity, dtype=np.float32),
            "behavior_logp": np.asarray(behavior_logp, dtype=np.float32),
            "train_mask": np.asarray(train_mask, dtype=bool),
            "reward": np.asarray(reward, dtype=np.float32),
            "winner_mask": teams == before_winner,
            "next_winner_mask": teams == after_winner,
            "discount": discount,
            "teams": teams,
            "actor_mask": np.asarray(actor_mask, dtype=bool),
            "behavior_updates": int(behavior_updates),
            "behavior_arms": np.asarray(behavior_arms),
            "behavior_versions": dict(behavior_versions),
            "configuration": json.dumps(self.match_metadata.get("configuration", {}), sort_keys=True),
            "match_id": self.match_metadata.get("match_id", "unspecified"),
        }

    def _tensor_loss(self, current_inputs, next_inputs, item):
        self.loss_traces += 1
        value_scale, actor_scale = item['value_scale'], item['actor_scale']
        velocity_mx = mx.array(item["velocity"])
        behavior_logp_mx = mx.array(item["behavior_logp"])
        train_mask = item['train_mask']
        train_weight = train_mask.astype(mx.float32)
        train_count = mx.maximum(mx.sum(train_weight), 1.0)
        actor_weight = train_weight * item['actor_mask'].astype(mx.float32)
        actor_count = mx.maximum(mx.sum(actor_weight), 1.0)
        reward = mx.array(item["reward"])
        winner_mask = mx.array(item["winner_mask"]).astype(mx.bool_)
        next_winner_mask = mx.array(item["next_winner_mask"]).astype(mx.bool_)

        current = self.policy_forward(self.wally, *current_inputs)
        following = self.policy_forward(self.wally, *next_inputs)

        actor_active = (actor_weight > 0) & (actor_scale > 0)
        logpi = logp_of(current, velocity_mx, actor_active)

        value = mx.where(winner_mask, current.value_winnie, current.value_lou)
        successor_rows = item['successor_rows']
        next_values_w = following.value_winnie[mx.maximum(successor_rows, 0)]
        next_values_l = following.value_lou[mx.maximum(successor_rows, 0)]
        bootstrap = mx.where(next_winner_mask, next_values_w, next_values_l)
        bootstrap = mx.where(((winner_mask == next_winner_mask) | (self.policy_arm == "terminal_win")) & (mx.array(item["discount"]) > 0), bootstrap, 0.0)
        target = reward + item["discount"] * mx.stop_gradient(bootstrap)
        target = mx.where(item['target_present'], item['value_target'], target)
        error = target - value

        proximal_logp = mx.stop_gradient(logpi)
        rollout_log_ratio = proximal_logp - mx.where(actor_active, behavior_logp_mx, 0.0)
        ratio = rollout_importance_weights(rollout_log_ratio)
        td_advantage = mx.stop_gradient(error)
        actor = -mx.sum(
            ratio * clipped_policy_surrogate(logpi - proximal_logp, td_advantage, POLICY_RATIO_CLIP) * actor_weight
        ) / actor_count

        present = mx.ones_like(current.rate.mean) if current.rate.present is None else current.rate.present
        state_count = mx.maximum(mx.sum(present, axis=-1), 1)
        state_entropy = mx.sum(
            (mx.sum((current.rate.log_scale + 0.5 * np.log(2 * np.pi * np.e)) * present, axis=-1) / state_count)
            * actor_weight
        ) / actor_count
        winner_weight = winner_mask.astype(mx.float32) * train_weight
        loser_weight = (~winner_mask).astype(mx.float32) * train_weight
        winner_loss = mx.sum(mx.square(current.value_winnie - target) * winner_weight) / mx.maximum(mx.sum(winner_weight), 1.0)
        loser_loss = mx.sum(mx.square(current.value_lou - target) * loser_weight) / mx.maximum(mx.sum(loser_weight), 1.0)

        total = (
            actor_scale * LOSS_WEIGHTS["actor"] * actor
            + value_scale * (LOSS_WEIGHTS["winnie"] * winner_loss
            + LOSS_WEIGHTS["lou"] * loser_loss
            + LOSS_WEIGHTS["moe_balance"] * current.scale_balance)
        )

        advantage = mx.sum(mx.stop_gradient(error) * train_weight) / train_count
        weighted_advantage = mx.sum(mx.stop_gradient(ratio * error) * train_weight) / train_count
        winner_count = mx.sum(winner_mask.astype(mx.float32) * train_weight)
        loser_count = mx.sum((~winner_mask).astype(mx.float32) * train_weight)
        winner_advantage = mx.sum(mx.stop_gradient(ratio * error) * winner_mask * train_weight) / mx.maximum(winner_count, 1.0)
        loser_advantage = mx.sum(mx.stop_gradient(ratio * error) * (~winner_mask) * train_weight) / mx.maximum(loser_count, 1.0)
        winner_reward = mx.sum(reward * winner_mask * train_weight) / mx.maximum(winner_count, 1.0)
        loser_reward = mx.sum(reward * (~winner_mask) * train_weight) / mx.maximum(loser_count, 1.0)
        role_change = mx.sum((winner_mask != next_winner_mask).astype(mx.float32) * train_weight) / train_count
        target_mean = mx.sum(target * train_weight) / train_count
        target_variance = mx.sum(mx.square(target - target_mean) * train_weight) / train_count
        return total, (mx.stack([actor, winner_loss, loser_loss,
                                mx.sum(ratio * train_weight) / train_count,
                                advantage, weighted_advantage,
                                winner_advantage, loser_advantage, winner_reward,
                                loser_reward, winner_count, loser_count, role_change,
                                state_entropy, target_variance, current.scale_balance]), rollout_log_ratio, ratio, actor_active)

    def realize_items(self, frame):
        players, pages = self.program.capacity[:2]
        width = frame.layout.shape[-1] - 3
        shapes = {name: ((players,), bool if name.endswith('mask') else
            np.int64 if name == 'successor_rows' else np.float32)
            for name in ('behavior_logp', 'reward', 'discount', 'value_target', 'winner_mask',
                'next_winner_mask', 'train_mask', 'actor_mask', 'successor_rows')}
        shapes.update(velocity=((players, pages * width), np.float32),
            value_scale=((), np.float32), actor_scale=((), np.float32), target_present=((), bool),
            weights=((len(METRIC_NAMES),), np.float32))
        self.item_storage.realize(shapes)

    def _prepare_item(self, item, value_scale, actor_scale, storage):
        n = len(item['behavior_logp'])
        for value in storage.host.values():
            value.fill(0)
        for name in ('behavior_logp', 'reward', 'discount', 'value_target', 'winner_mask',
                     'next_winner_mask', 'train_mask', 'actor_mask', 'successor_rows'):
            storage.host[name][:n] = item.get(name, 0) if name == 'value_target' else item[name]
        storage.host['velocity'][:n, :item['velocity'].shape[1]] = item['velocity']
        storage.host['value_scale'][...] = value_scale
        storage.host['actor_scale'][...] = actor_scale
        storage.host['target_present'][...] = 'value_target' in item
        payload = {name: value for name, value in storage.values.items() if name != 'weights'}
        before, after = (self.input_arena.load(item[name], slot)
            for slot, name in enumerate(('chorus_in', 'chorus_out')))
        return before, after, payload

    def prepare_update(self, items, replay_items):
        items = list(items)
        if not items:
            return None
        rebuild_t0 = time.perf_counter()
        items = [self.replay.materialize(item) for item in items]
        replay_items = [self.replay.materialize(item) for item in replay_items]
        self.rebuild_seconds += time.perf_counter() - rebuild_t0
        self.rebuild_calls += 1
        self.input_arena.reserve(*(frame for item in items + replay_items for frame in
            (item['chorus_in'], item['chorus_out'])), count=2, capacity=self.program.capacity)
        self.program.bind_capacity(self.input_arena.capacity)
        self.program.reserve(items[0]['chorus_in'])
        replay_weight = self.replay_weight if replay_items else 0.0
        value_mass = max(sum(item.get('value_weight', 1.0) for item in items), 1.0)
        actor_rows = [int(np.count_nonzero(item['actor_mask'] & item['train_mask'])) for item in items]
        actor_mass = max(sum(actor_rows), 1)
        prepared = []
        for item, rows in zip(items, actor_rows):
            value_weight = item.get('value_weight', 1.0) / value_mass
            actor_weight = rows / actor_mass
            weights = np.full(len(METRIC_NAMES), value_weight, dtype=np.float32)
            weights[ACTOR_METRIC_INDICES] = actor_weight
            prepared.append((item, (1.0 - replay_weight) * value_weight, actor_weight, weights))
        for item in replay_items:
            prepared.append((item, replay_weight / len(replay_items), 0.0,
                np.zeros(len(METRIC_NAMES), dtype=np.float32)))
        return items, replay_items, prepared

    def learn(self, prepared_update):
        items, replay_items, prepared = prepared_update
        proximal_updates = self.updates
        ratios = []
        storage = self.item_storage
        for index, (item, value_scale, actor_scale, weights) in enumerate(prepared):
            arguments = self._prepare_item(item, value_scale, actor_scale, storage)
            log_ratio, ratio, active = self.program.native.contribute(*arguments, weights, index == 0)
            self.loss_traces = 1
            self.accumulation_traces = 1
            mask = np.asarray(active)
            if np.any(mask):
                ratios.append(np.stack((np.asarray(log_ratio), np.asarray(ratio)), axis=-1)[mask])
        if ratios:
            self.ratios.append(np.concatenate(ratios))
        total, parts, gradient_norm = self.program.native.update()
        self.update_traces = 1
        self.updates += 1
        self.gradient_steps += 1
        parts = np.asarray(parts)
        metrics = {name: float(parts[i]) for i, name in enumerate(METRIC_NAMES)}
        metrics["gradient_norm"] = float(np.asarray(gradient_norm))
        metrics["gradient_clip"] = self.gradient_clip
        metrics.update(
            loss=float(np.asarray(total)),
            updates=self.updates,
            proximal_updates=proximal_updates,
            schedule_updates=self.updates - self.schedule_start_updates,
            schedule_start_updates=self.schedule_start_updates,
            training_contract=self.training_contract,
            execution={**self.program.report(), 'loss_traces': self.loss_traces,
                       'optimizer_traces': self.update_traces,
                       'accumulation_traces': self.accumulation_traces,
                       'source_group_size': len(prepared), 'resident_transition_slots': 1,
                       'gradient_reduction': 'ordered_local',
                       'training_input_storage': self.input_arena.report(),
                       'transition_storage': {'slots': 1, 'allocations': self.item_storage.allocations,
                            'bytes': self.item_storage.nbytes}},
            bootstrap_gamma=self.bootstrap_gamma,
            gradient_steps=1,
            batch=len(items),
            replay_batch=len(replay_items),
            replay_behavior_age_updates=float(np.mean([self.updates - item.get("behavior_updates", self.updates) for item in replay_items])) if replay_items else None,
            actor_rows=sum(int(np.asarray(item.get("actor_mask", item["train_mask"])).sum()) for item in items),
            historical_actor_rows=0,
            value_rows=sum(int(np.asarray(item["train_mask"]).sum()) for item in items + replay_items if item.get("value_weight", 1.0)),
            behavior_age_updates=float(np.average(
                [proximal_updates - item.get("behavior_updates", proximal_updates) for item in items],
                weights=[np.count_nonzero(item["actor_mask"] & item["train_mask"]) for item in items],
            )) if any(np.any(item["actor_mask"] & item["train_mask"]) for item in items) else None,
            value_target_variance=(1.0 - (self.replay_weight if replay_items else 0.0)) * metrics["fresh_value_target_variance"] + (self.replay_weight * float(np.mean([np.var(item["value_target"]) for item in replay_items])) if replay_items else 0.0),
            value_target_semantics="observed_behavior_return" if all("value_target" in item for item in items) else "fresh_td_and_historical_behavior_return",
        )
        return metrics

    def observe_attributed(self, records):
        fresh = [self.transition(**record) for record in records]
        observations = [item for item in fresh if item["value_weight"]]
        for item in observations:
            teams = {int(team): index for index, team in enumerate(item["teams"])}
            for prior in self.episode.items():
                indices = np.asarray([teams.get(int(team), -1) for team in prior["teams"]])
                present = indices >= 0
                prior["value_target"] += prior["return_factor"] * item["reward"][indices] * present
                prior["return_factor"] *= item["discount"][indices] * present * ((prior["winner_mask"] == item["next_winner_mask"][indices]) | (self.policy_arm == "terminal_win"))
            self.episode.push(dict(
                item, value_target=item["reward"].copy(),
                return_factor=item["discount"] * ((item["winner_mask"] == item["next_winner_mask"]) | (self.policy_arm == "terminal_win")),
            ))
        self.transitions += len(observations)
        out = self._train_fresh(fresh)
        if out is not None:
            out["attributed_groups"] = len(fresh)
            out["attributed_rows"] = int(sum(
                np.asarray(item["train_mask"], dtype=bool).sum() for item in fresh
            ))
        return out

    def switch_episode(self, identity):
        if identity != self.episode_id:
            if self.episode_id is not None and len(self.episode):
                self.awaiting_outcomes[self.episode_id] = self.episode
            self.episode = Replay(self.episode.capacity, self.episode.max_bytes)
            self.episode_id = identity

    def end_episode(self, winning_team=None, episode_id=None):
        identity = episode_id if episode_id is not None else self.episode_id
        if identity is not None and identity in self.outcomes:
            return {"outcome": self.outcomes[identity], "duplicate": True, "retained_states": 0}
        pool = self.episode if identity == self.episode_id else self.awaiting_outcomes.get(identity)
        if pool is None:
            return {"outcome": winning_team, "unobserved_episode": identity, "retained_states": 0}
        items = list(pool.items())
        metrics = {"outcome": winning_team, "retained_states": 0, "unlabelled_states": 0}
        if winning_team is None:
            self.truncated_episodes += 1
            metrics["unlabelled_states"] = len(items)
        else:
            self.completed_episodes += 1
            if identity is not None:
                self.outcomes[identity] = int(winning_team)
            if self.policy_arm == "terminal_win":
                for item in items:
                    item["value_target"] = ((winning_team > 0) & (item["teams"] == winning_team - 1)).astype(np.float32)
            metrics["retained_states"] = self.replay.retain(items)
        pool.clear()
        self.awaiting_outcomes.pop(identity, None)
        metrics.update(gradient_steps=0, completed_episodes=self.completed_episodes, truncated_episodes=self.truncated_episodes, replay_size=len(self.replay), replay=self.replay.report(), updates=self.updates)
        return metrics

    def _train_fresh(self, fresh):
        if not fresh:
            self.replay.release_unreferenced()
            return None
        self.replay.release_unreferenced()
        prepared = self.prepare_update(fresh, self.replay.sample(self.replay_batch, self.rng))
        out = self.learn(prepared)
        report = self.replay.report()
        out.update(
            credited_steps=sum(bool(item.get("value_weight", 1.0)) for item in fresh),
            credited_action_rows=sum(int(np.asarray(item["actor_mask"]).sum()) for item in fresh),
            gradient_steps=1,
            replay_size=len(self.replay),
            replay_capacity=self.replay.capacity,
            replay_mb=round(self.replay.nbytes / (1 << 20), 3),
            replay_bytes_per_state=report["bytes_per_transition"],
            replay_frames=report["frames"],
            replay_precision=self.replay_precision,
            replay_mean_age=round(self.replay.mean_age(self.replay.items()), 2) if len(self.replay) else 0.0,
            steps_per_transition=round(self.gradient_steps / max(1, self.transitions), 3),
            feature_rebuild_ms=round(1000.0 * self.rebuild_seconds / max(1, self.rebuild_calls), 3),
            importance_ratio=self.ratio_report(),
        )
        return out

    def ratio_report(self):
        if not self.ratios:
            return None
        log_ratios, values = np.concatenate(self.ratios).T
        quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        return {
            "semantics": "proximal_over_behavior_truncated",
            "n": int(values.size),
            "mean": float(values.mean()),
            "clipped_fraction": round(float(np.mean(log_ratios > np.log(POLICY_ROLLOUT_IS_THRESHOLD))), 5),
            "numerical_floor_fraction": round(float(np.mean(log_ratios < -POLICY_LOG_RATIO_BOUND)), 5),
            "effective_sample_fraction": float(np.square(values.astype(np.float64).sum()) / (values.size * np.square(values.astype(np.float64)).sum())),
            "log_ratio_quantiles": np.quantile(log_ratios, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).tolist(),
            "quantiles": quantiles.tolist(),
        }

    def export_payload(self):
        payload = {name: np.asarray(value) for name, value in tree_flatten(self.wally.parameters())}
        for name, value in tree_flatten(self.optimizer.state):
            payload["__opt__" + name] = np.asarray(value)
        payload["__updates__"] = np.asarray(self.updates)
        payload["__transitions__"] = np.asarray(self.transitions)
        payload["__gradient_steps__"] = np.asarray(self.gradient_steps)
        payload["__completed_episodes__"] = np.asarray(self.completed_episodes)
        payload["__truncated_episodes__"] = np.asarray(self.truncated_episodes)
        payload["__training_contract__"] = np.asarray(json.dumps(self.training_contract, sort_keys=True))
        payload["__schedule_start_updates__"] = np.asarray(self.schedule_start_updates)
        payload[ARCH_KEY] = np.asarray(self.architecture)
        payload[ARCH_SPEC_KEY] = np.asarray(
            json.dumps(architecture_spec(self.wally), separators=(",", ":"))
        )
        payload[RNG_KEY] = np.asarray(
            json.dumps(self.rng.bit_generator.state, separators=(",", ":"))
        )
        payload[POLICY_KEY] = np.asarray(self.policy_arm)
        payload[POLICY_VERSION_KEY] = np.asarray(POLICY_VERSIONS.get(self.policy_arm, POLICY_VERSIONS["linear"]))
        payload[REWARD_CONTRACT_KEY] = np.asarray(self.reward_fingerprint)
        payload["__continuation__"] = np.asarray(json.dumps(self.continuation, sort_keys=True))
        if self.initial_checkpoint_sha256:
            payload[LINEAGE_INITIAL_KEY] = np.asarray(self.initial_checkpoint_sha256)
        payload.update(self.replay.export_payload())
        payload.update(self.episode.export_payload("__episode__"))
        payload["__episodes__"] = np.asarray(json.dumps({"current": self.episode_id,
            "awaiting": list(self.awaiting_outcomes), "outcomes": self.outcomes}))
        for index, pool in enumerate(self.awaiting_outcomes.values()):
            payload.update(pool.export_payload(f"__awaiting_{index}__"))
        return payload

    def save(self, path=None):
        target = path or self.checkpoint
        if target is not None:
            atomic_save(target, self.export_payload())
