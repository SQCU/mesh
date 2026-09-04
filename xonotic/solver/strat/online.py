from __future__ import annotations

import json
import os
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .cast_header import dina_state, elle
from .checkpoint_state import (
    ARCH_KEY, ARCH_SPEC_KEY, POLICY_KEY, POLICY_VERSION_KEY, RNG_KEY,
    REWARD_CONTRACT_KEY, LINEAGE_INITIAL_KEY, POLICY_VERSIONS, architecture_fingerprint,
    architecture_spec, tensor_tree_measurement, whole_tensor_tree,
    load_module_checkpoint,
)
from .runtime import (
    reward_fingerprint,
    POLICY_ACTOR_WEIGHT,
    POLICY_ENTROPY_FLOOR,
    POLICY_ENTROPY_WEIGHT,
    POLICY_RATIO_CLIP,
    role_rewards,
    winner,
)
from .matmul import matrix_multiply
from .policy_contract import is_matrix_fusion_arm

LOSS_WEIGHTS = {
    "actor":  POLICY_ACTOR_WEIGHT,
    "entropy": POLICY_ENTROPY_WEIGHT,
    "winnie": 0.5,
    "lou":    0.5,
    "vera":   0.25,
    "dina":   0.25,
    "elle":   1e-3,
}

from .replay import Replay
from .instruments import KINDS
from .strategy import strategy, dynamics, log_probs, logp_of

def clipped_policy_surrogate(ratio, advantage):
    clipped = mx.clip(
        ratio, 1.0 - POLICY_RATIO_CLIP, 1.0 + POLICY_RATIO_CLIP,
    )
    return mx.minimum(ratio * advantage, clipped * advantage)

def entropy_floor_penalty(entropy):
    return mx.square(mx.maximum(POLICY_ENTROPY_FLOOR - entropy, 0.0))

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
        replay_steps: int = 4,
        seed: int = 20260831,
        policy_forward=strategy,
        policy_arm: str = "matrix_fusion",
        match_metadata=None,
        replay_weight=0.5,
    ):
        self.wally = wally
        self.gamma = float(gamma)
        self.gradient_clip = abs(float(gradient_clip))
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.checkpoint = checkpoint
        self.updates = 0
        self.replay = Replay(replay_capacity, int(replay_memory_mb * (1 << 20)))
        self.episode = Replay(replay_capacity, int(replay_memory_mb * (1 << 20)))
        self.match_metadata = dict(match_metadata or {})
        self.replay_weight = float(replay_weight)
        self.completed_episodes = 0
        self.truncated_episodes = 0
        self.replay_precision = replay_precision
        self.replay_batch = max(1, int(replay_batch))
        self.replay_steps = max(0, int(replay_steps))
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
        self.reward_fingerprint = reward_fingerprint(policy_arm)
        self.architecture = architecture_fingerprint(self.wally)
        self.loaded_weight_mass = 0
        self.live_weight_mass = len(tree_flatten(self.wally.parameters()))
        self.loaded_optimizer_moment_mass = 0
        self.live_optimizer_moment_mass = 0
        self.optimizer_moment_measurement = {}
        self.initial_checkpoint_sha256 = None
        source = load_checkpoint or checkpoint
        if source is not None and os.path.exists(source):
            self._load_full(source)

    def _load_full(self, source):
        state = load_module_checkpoint(
            self.wally, source, self.policy_arm, self.reward_fingerprint,
        )
        self.loaded_weight_mass = state["loaded_weight_mass"]
        self.initial_checkpoint_sha256 = state["lineage_initial_sha256"]
        self.live_weight_mass = state["live_weight_mass"]
        replay_atom_mass = 0
        replay_exception = None
        optimizer_exception = None
        with np.load(source, allow_pickle=False) as data:
            keys = list(data.files)
            if "__updates__" in keys:
                self.updates = int(np.asarray(data["__updates__"]))
            if "__transitions__" in keys:
                self.transitions = int(np.asarray(data["__transitions__"]))
            if "__gradient_steps__" in keys:
                self.gradient_steps = int(np.asarray(data["__gradient_steps__"]))
            self.completed_episodes = int(data["__completed_episodes__"]) if "__completed_episodes__" in keys else 0
            self.truncated_episodes = int(data["__truncated_episodes__"]) if "__truncated_episodes__" in keys else 0
            self.optimizer.init(self.wally.trainable_parameters())
            live_moments = dict(tree_flatten(self.optimizer.state))
            source_moments = {
                key[7:]: np.asarray(data[key]) for key in keys if key.startswith("__opt__")
            }
            moment_measurement = tensor_tree_measurement(
                live_moments.items(), source_moments.items(),
            )
            self.optimizer_moment_measurement = moment_measurement
            try:
                source_state, _ = whole_tensor_tree(
                    live_moments.items(), source_moments.items(),
                )
                self.optimizer.state = source_state
                self.loaded_optimizer_moment_mass = moment_measurement["source_mass"]
            except Exception as error:
                self.optimizer.state = tree_unflatten(list(live_moments.items()))
                optimizer_exception = f"{type(error).__name__}: {error}"
            self.live_optimizer_moment_mass = len(live_moments)
            try:
                self.replay.restore_payload(data)
                replay_atom_mass = len(self.replay)
            except Exception as error:
                replay_exception = f"{type(error).__name__}: {error}"
            if RNG_KEY in keys:
                self.rng.bit_generator.state = json.loads(str(data[RNG_KEY]))
        print(json.dumps({
            "event": "online_checkpoint_measurement", "path": source,
            "source_arm": state["source_arm"], "source_version": state["source_version"],
            "source_architecture": state["source_architecture"], "live_arm": self.policy_arm,
            "live_version": state["live_version"], "live_architecture": self.architecture,
            "source_reward_contract": state["source_reward_contract"],
            "live_reward_contract": self.reward_fingerprint,
            "source_weight_mass": state["source_weight_mass"],
            "live_weight_mass": self.live_weight_mass,
            "loaded_weight_mass": self.loaded_weight_mass,
            "composable_weight_mass": state["composable_weight_mass"],
            "source_only_weight_mass": state["source_only_weight_mass"],
            "live_only_weight_mass": state["live_only_weight_mass"],
            "shape_difference_mass": state["shape_difference_mass"],
            "nonfinite_weight_mass": state["nonfinite_weight_mass"],
            "load_exception": state["load_exception"],
            "source_optimizer_moment_mass": len(source_moments),
            "live_optimizer_moment_mass": self.live_optimizer_moment_mass,
            "loaded_optimizer_moment_mass": self.loaded_optimizer_moment_mass,
            "optimizer_moment_measurement": self.optimizer_moment_measurement,
            "replay_atom_mass": replay_atom_mass, "replay_bytes": self.replay.nbytes,
            "optimizer_exception": optimizer_exception,
            "replay_exception": replay_exception,
            "updates": self.updates, "transitions": self.transitions,
        }), flush=True)

    def transition(
        self,
        context,
        frame,
        next_frame,
        snapshot,
        next_snapshot,
        actions,
        controls,
        behavior_logp,
        train_mask=None,
        dynamics_mask=None,
        sparse_return=None,
        bootstrap_discount=None,
        actor_mask=None,
        behavior_updates=0,
        behavior_arms=None,
        behavior_versions=None,
    ):
        players = np.asarray(self.replay.frame(frame).xan).shape[0]
        reward = (role_rewards(context, snapshot, next_snapshot)
                  if sparse_return is None else np.asarray(sparse_return))
        before_winner = winner(context, snapshot)
        after_winner = winner(context, next_snapshot)
        teams = np.asarray(context.team_of, dtype=np.int64)
        discount = np.array(np.broadcast_to(
            self.gamma if bootstrap_discount is None else bootstrap_discount,
            (players,),
        ), dtype=np.float32, copy=True)
        return {
            "frame_in": frame,
            "frame_out": next_frame,
            "actions": np.asarray(actions, dtype=np.int32),
            "controls": np.asarray(controls, dtype=np.float32),
            "behavior_logp": np.asarray(behavior_logp, dtype=np.float32),
            "train_mask": np.ones(players, dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool),
            "dynamics_mask": np.ones(players, dtype=bool) if dynamics_mask is None else np.asarray(dynamics_mask, dtype=bool),
            "roster_row_residual_mass": 0,
            "reward": np.asarray(reward, dtype=np.float32),
            "winner_mask": teams == before_winner,
            "next_winner_mask": teams == after_winner,
            "discount": discount,
            "teams": teams,
            "actor_mask": np.ones(players, dtype=bool) if actor_mask is None else np.asarray(actor_mask, dtype=bool),
            "behavior_updates": int(behavior_updates),
            "behavior_arms": np.asarray(behavior_arms if behavior_arms is not None else [self.policy_arm] * players),
            "behavior_versions": dict(behavior_versions or {}),
            "configuration": json.dumps(self.match_metadata.get("configuration", {}), sort_keys=True),
            "match_id": self.match_metadata.get("match_id", "unspecified"),
        }

    def _item_loss(self, item, value_scale=1.0, actor_scale=1.0):
        actions_mx = mx.array(item["actions"])
        controls_mx = mx.array(item["controls"])
        behavior_logp_mx = mx.array(item["behavior_logp"])
        train_mask = mx.array(item.get("train_mask", np.ones_like(item["actions"], dtype=bool))).astype(mx.bool_)
        train_weight = train_mask.astype(mx.float32)
        dynamics_mask = mx.array(item.get("dynamics_mask", np.ones_like(item["actions"], dtype=bool))).astype(mx.bool_)
        train_count = mx.maximum(mx.sum(train_weight), 1.0)
        actor_weight = train_weight * mx.array(item.get("actor_mask", item["train_mask"])).astype(mx.float32)
        actor_count = mx.maximum(mx.sum(actor_weight), 1.0)
        reward = mx.array(item["reward"])
        winner_mask = mx.array(item["winner_mask"]).astype(mx.bool_)
        next_winner_mask = mx.array(item["next_winner_mask"]).astype(mx.bool_)
        current_action_mass = mx.array(item["chorus_in"].action_mass)

        current = self.policy_forward(self.wally, *(mx.array(a) for a in item["chorus_in"]))
        following = current if "value_target" in item and not actor_scale else self.policy_forward(self.wally, *(mx.array(a) for a in item["chorus_out"]))

        logpi = logp_of(current, actions_mx, controls_mx)

        value = mx.where(winner_mask, current.value_winnie, current.value_lou)
        bootstrap = mx.where(
            winner_mask,
            mx.where(next_winner_mask, following.value_winnie, 0.0),
            mx.where(next_winner_mask, 0.0, following.value_lou),
        )
        target = reward + item["discount"] * mx.stop_gradient(bootstrap)
        if "value_target" in item:
            target = mx.array(item["value_target"])
        error = target - value

        ratio = mx.exp(logpi - behavior_logp_mx) if actor_scale else mx.ones_like(logpi)
        self._ratio_sink.append((
            mx.stop_gradient(ratio),
            np.asarray(item.get(
                "actor_mask", np.ones_like(item["actions"], dtype=bool),
            )) & bool(actor_scale),
        ))
        td_advantage = mx.stop_gradient(error)
        actor = -mx.sum(
            clipped_policy_surrogate(ratio, td_advantage) * actor_weight
        ) / actor_count

        probabilities = mx.exp(log_probs(current))
        kind_probabilities = matrix_multiply(
            probabilities,
            mx.array(item["chorus_in"].zed[:, :len(KINDS)]),
        )
        semantic_entropy = -mx.sum(
            kind_probabilities
            * mx.log(mx.maximum(kind_probabilities, 1e-12))
            * train_weight[:, None]
        ) / train_count
        entropy_penalty = entropy_floor_penalty(semantic_entropy)

        winner_weight = winner_mask.astype(mx.float32) * train_weight
        loser_weight = (~winner_mask).astype(mx.float32) * train_weight
        winner_loss = mx.sum(mx.square(current.value_winnie - target) * winner_weight) / mx.maximum(mx.sum(winner_weight), 1.0)
        loser_loss = mx.sum(mx.square(current.value_lou - target) * loser_weight) / mx.maximum(mx.sum(loser_weight), 1.0)

        aux = 0.5 * (
            mx.sum(mx.square(current.aux_winnie - mx.stop_gradient(current.value_winnie)) * train_weight) / train_count
            + mx.sum(mx.square(current.aux_lou - mx.stop_gradient(current.value_lou)) * train_weight) / train_count
        )
        if is_matrix_fusion_arm(self.policy_arm) and actor_scale:
            y = mx.stop_gradient(current.query)
            u = mx.stop_gradient(mx.take_along_axis(
                current.ir, actions_mx[:, None, None], axis=1
            )[:, 0, :])
            target_delta = mx.stop_gradient(
                dina_state(self.wally, following.query)
                - dina_state(self.wally, current.query)
            )
            predicted = dynamics(self.wally, y, u)
            dynamics_weight = (train_weight * dynamics_mask.astype(mx.float32))[:, None]
            dynamics_count = mx.maximum(mx.sum(dynamics_weight) * predicted.first.shape[-1], 1.0)
            dynamics_value = 0.5 * (
                mx.sum(mx.square(predicted.first - target_delta) * dynamics_weight) / dynamics_count
                + mx.sum(mx.square(predicted.second - target_delta) * dynamics_weight) / dynamics_count
            )
            dynamics_error = mx.sum(mx.square(predicted.mean - target_delta) * dynamics_weight) / dynamics_count
            dynamics_disagreement = mx.sum(predicted.disagreement * train_weight) / train_count
        else:
            dynamics_value = mx.array(0.0)
            dynamics_error = mx.array(0.0)
            dynamics_disagreement = mx.array(0.0)

        regularization = elle(
            current.logits, current_action_mass * train_weight[:, None],
        )

        total = (
            actor_scale * (LOSS_WEIGHTS["actor"] * actor
            + LOSS_WEIGHTS["entropy"] * entropy_penalty
            + LOSS_WEIGHTS["dina"] * dynamics_value
            + LOSS_WEIGHTS["elle"] * regularization)
            + value_scale * (LOSS_WEIGHTS["winnie"] * winner_loss
            + LOSS_WEIGHTS["lou"] * loser_loss
            + LOSS_WEIGHTS["vera"] * aux)
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
        return total, mx.stack([actor, winner_loss, loser_loss, dynamics_value,
                                regularization, mx.sum(ratio * train_weight) / train_count, aux,
                                dynamics_error, advantage, weighted_advantage,
                                winner_advantage, loser_advantage, winner_reward,
                                loser_reward, winner_count, loser_count, role_change,
                                dynamics_disagreement, semantic_entropy,
                                entropy_penalty])

    def learn(self, items, replay_items=()):
        items = list(items)
        if not items:
            return None
        rebuild_t0 = time.perf_counter()
        items = [self.replay.materialize(item) for item in items]
        replay_items = [self.replay.materialize(item) for item in replay_items]
        self.rebuild_seconds += time.perf_counter() - rebuild_t0
        self.rebuild_calls += 1

        def loss_fn():
            replay_weight = self.replay_weight if replay_items else 0.0
            losses, parts = zip(*(self._item_loss(item, 1.0 - replay_weight) for item in items))
            total = mx.mean(mx.stack(losses))
            if replay_items:
                historical, _ = zip(*(self._item_loss(item, replay_weight, 0.0) for item in replay_items))
                total = total + mx.mean(mx.stack(historical))
            return total, mx.mean(mx.stack(parts), axis=0)

        self._ratio_sink = []
        (total, parts), gradients = nn.value_and_grad(self.wally, loss_fn)()
        self._ratio_sink = [(ratio, mask) for ratio, mask in self._ratio_sink if np.any(mask)]
        if self._ratio_sink:
            mx.eval(*(ratio for ratio, _ in self._ratio_sink))
            self.ratios.append(np.concatenate(
                [np.asarray(ratio).reshape(-1)[np.asarray(mask, dtype=bool).reshape(-1)]
                 for ratio, mask in self._ratio_sink]))
            self._ratio_sink = []
        gradients, gradient_norm = optim.clip_grad_norm(gradients, self.gradient_clip)
        self.optimizer.update(self.wally, gradients)
        mx.eval(self.wally.parameters(), self.optimizer.state, total, parts, gradient_norm)
        self.updates += 1
        self.gradient_steps += 1
        names = ("loss_pg", "loss_w", "loss_l", "loss_dynamics", "loss_reg",
                 "importance_mean", "loss_aux_values", "model_one_step_error",
                 "advantage", "advantage_importance_weighted", "advantage_w",
                 "advantage_l", "reward_w", "reward_l", "winner_rows",
                 "loser_rows", "role_change_fraction", "model_uncertainty",
                 "semantic_entropy", "entropy_floor_penalty")
        parts = np.asarray(parts)
        metrics = {name: float(parts[i]) for i, name in enumerate(names)}
        metrics["gradient_norm"] = float(np.asarray(gradient_norm))
        metrics["gradient_clip"] = self.gradient_clip
        metrics["local_control_sigma_min"] = self._control_sigma(items)
        metrics.update(
            loss=float(np.asarray(total)),
            updates=self.updates,
            gradient_steps=1,
            batch=len(items),
            replay_batch=len(replay_items),
            replay_behavior_age_updates=float(np.mean([self.updates - item.get("behavior_updates", self.updates) for item in replay_items])) if replay_items else None,
            actor_rows=sum(int(np.asarray(item.get("actor_mask", item["train_mask"])).sum()) for item in items),
            value_rows=sum(int(np.asarray(item["train_mask"]).sum()) for item in items + replay_items),
            behavior_age_updates=float(np.mean([self.updates - item.get("behavior_updates", self.updates) for item in items])),
            value_target_variance=float(np.mean([np.var(item.get("value_target", item["reward"])) for item in items + replay_items])),
            value_target_semantics="observed_behavior_return" if all("value_target" in item for item in items) else "fresh_td_and_historical_behavior_return",
        )
        return metrics

    def _control_sigma(self, items):
        if not is_matrix_fusion_arm(self.policy_arm):
            return float("nan")
        values = []
        for item in items[:1]:
            current = self.policy_forward(self.wally, *(mx.array(a) for a in item["chorus_in"]))
            actions = mx.array(item["actions"])
            chosen = mx.take_along_axis(current.ir, actions[:, None, None], axis=1)[:, 0, :]
            matrix = dynamics(self.wally, current.query, chosen).matrix
            mx.eval(matrix)
            mask = np.asarray(item.get("train_mask", np.ones_like(item["actions"], dtype=bool)), dtype=bool)
            for row in np.asarray(matrix)[mask]:
                if not np.isfinite(row).all():
                    continue
                try:
                    values.append(float(np.linalg.svd(row, compute_uv=False)[-1]))
                except np.linalg.LinAlgError:
                    continue
        return float(np.mean(values)) if values else float("nan")

    def observe_attributed(self, records):
        fresh = [self.transition(**record) for record in records]
        for item in fresh:
            teams = {int(team): index for index, team in enumerate(item["teams"])}
            for prior in self.episode.items():
                indices = np.asarray([teams.get(int(team), -1) for team in prior["teams"]])
                present = indices >= 0
                prior["value_target"] += prior["return_factor"] * item["reward"][indices] * present
                prior["return_factor"] *= self.gamma * present * (prior["winner_mask"] == item["next_winner_mask"][indices])
            self.episode.push(dict(
                item, value_target=item["reward"].copy(),
                return_factor=item["discount"] * (item["winner_mask"] == item["next_winner_mask"]),
            ))
        self.transitions += len(fresh)
        out = self._train_fresh(fresh) if self.policy_arm != "terminal_win" else None
        if out is not None:
            out["attributed_groups"] = len(fresh)
            out["attributed_rows"] = int(sum(
                np.asarray(item["train_mask"], dtype=bool).sum() for item in fresh
            ))
        return out

    def end_episode(self, winning_team=None):
        items = list(self.episode.items())
        metrics = {"outcome": winning_team, "retained_states": 0, "unlabelled_states": 0}
        steps = 0
        if winning_team is None:
            self.truncated_episodes += 1
            metrics["unlabelled_states"] = len(items)
        else:
            self.completed_episodes += 1
            if self.policy_arm == "terminal_win":
                for item in items:
                    item["value_target"] = ((winning_team > 0) & (item["teams"] == winning_team - 1)).astype(np.float32)
                    item["reward"] = item["value_target"].copy()
                    item["discount"] = np.zeros_like(item["discount"])
                for _ in range(max(1, self.replay_steps)):
                    batch = self.episode.sample(self.replay_batch, self.rng)
                    if batch:
                        metrics.update(self.learn(batch, self.replay.sample(self.replay_batch, self.rng)))
                        steps += 1
            metrics["retained_states"] = self.replay.retain(items)
        self.episode.clear()
        metrics.update(gradient_steps=steps, completed_episodes=self.completed_episodes, truncated_episodes=self.truncated_episodes, replay_size=len(self.replay), replay=self.replay.report(), updates=self.updates)
        return metrics

    def _train_fresh(self, fresh):
        if not fresh:
            self.replay.release_unreferenced()
            return None
        self.replay.release_unreferenced()
        out = self.learn(fresh, self.replay.sample(self.replay_batch, self.rng))
        report = self.replay.report()
        out.update(
            credited_steps=len(fresh),
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
        values = np.concatenate(self.ratios)
        quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        return {
            "n": int(values.size),
            "mean": round(float(values.mean()), 5),
            "clipped_fraction": round(float(np.mean(
                (values <= 1.0 - POLICY_RATIO_CLIP + 1e-6)
                | (values >= 1.0 + POLICY_RATIO_CLIP - 1e-6)
            )), 5),
            "quantiles": [round(float(q), 5) for q in quantiles],
        }

    def save(self, path=None):
        target = path or self.checkpoint
        if target is None:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        temporary = target + ".new.npz"
        payload = {name: np.asarray(value) for name, value in tree_flatten(self.wally.parameters())}
        for name, value in tree_flatten(self.optimizer.state):
            payload["__opt__" + name] = np.asarray(value)
        payload["__updates__"] = np.asarray(self.updates)
        payload["__transitions__"] = np.asarray(self.transitions)
        payload["__gradient_steps__"] = np.asarray(self.gradient_steps)
        payload["__completed_episodes__"] = np.asarray(self.completed_episodes)
        payload["__truncated_episodes__"] = np.asarray(self.truncated_episodes)
        payload["__training_contract__"] = np.asarray(json.dumps({"gamma": self.gamma, "replay_weight": self.replay_weight, "replay_batch": self.replay_batch, "terminal_gradient_steps": self.replay_steps, "historical_target": "observed_behavior_return", "actor_source": "own_fresh_behavior"}, sort_keys=True))
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
        if self.initial_checkpoint_sha256:
            payload[LINEAGE_INITIAL_KEY] = np.asarray(self.initial_checkpoint_sha256)
        payload.update(self.replay.export_payload())
        np.savez(temporary, **payload)
        os.replace(temporary, target)
