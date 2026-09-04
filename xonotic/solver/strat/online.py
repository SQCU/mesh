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

from .cast_header import Wally, Widths, dina_state, elle
from .checkpoint_state import (
    ARCH_KEY, ARCH_SPEC_KEY, POLICY_KEY, POLICY_VERSION_KEY, RNG_KEY,
    REWARD_CONTRACT_KEY, LINEAGE_INITIAL_KEY, POLICY_VERSIONS, architecture_fingerprint,
    architecture_spec, tensor_tree_measurement, whole_tensor_tree,
    load_module_checkpoint,
)
from .runtime import (
    SPARSE_REWARD_FINGERPRINT,
    POLICY_ACTOR_WEIGHT,
    POLICY_ENTROPY_FLOOR,
    POLICY_ENTROPY_WEIGHT,
    POLICY_RATIO_CLIP,
    role_rewards,
    winner,
)
from .matmul import matrix_multiply

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
        importance_clip: float = 2.0,
        dynamics=None,
        checkpoint=None,
        load_checkpoint=None,
        credit_horizon: int = 5,
        replay_capacity: int = 0,
        replay_memory_mb: float = 256.0,
        replay_precision: str = "float32",
        replay_batch: int = 8,
        replay_steps: int = 4,
        seed: int = 20260831,
        policy_forward=strategy,
        policy_arm: str = "matrix_fusion",
    ):
        self.wally = wally
        self.gamma = float(gamma)
        self.importance_clip = float(importance_clip)
        self.gradient_clip = abs(float(gradient_clip))
        self.bundle = wally
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.checkpoint = checkpoint
        self.updates = 0
        self.credit_horizon = max(1, int(credit_horizon))
        self.pending = []
        self.replay = Replay(replay_capacity, int(replay_memory_mb * (1 << 20)))
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
        self.architecture = architecture_fingerprint(self.bundle)
        self.loaded_weight_mass = 0
        self.live_weight_mass = len(tree_flatten(self.bundle.parameters()))
        self.loaded_optimizer_moment_mass = 0
        self.live_optimizer_moment_mass = 0
        self.optimizer_moment_measurement = {}
        self.initial_checkpoint_sha256 = None
        source = load_checkpoint or checkpoint
        if source is not None and os.path.exists(source):
            self._load_full(source)

    def _load_full(self, source):
        state = load_module_checkpoint(
            self.bundle, source, self.policy_arm, SPARSE_REWARD_FINGERPRINT,
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
            self.optimizer.init(self.bundle.trainable_parameters())
            live_moments = dict(tree_flatten(self.optimizer.state))
            source_moments = {
                key[7:]: np.asarray(data[key]).copy() for key in keys if key.startswith("__opt__")
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
            "live_reward_contract": SPARSE_REWARD_FINGERPRINT,
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
        dyn_frame,
        snapshot,
        next_snapshot,
        actions,
        controls,
        behavior_logp,
        train_mask=None,
        dynamics_mask=None,
        sparse_return=None,
        bootstrap_discount=None,
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
        }

    def _item_loss(self, item):
        actions_mx = mx.array(item["actions"])
        controls_mx = mx.array(item["controls"])
        behavior_logp_mx = mx.array(item["behavior_logp"])
        train_mask = mx.array(item.get("train_mask", np.ones_like(item["actions"], dtype=bool))).astype(mx.bool_)
        train_weight = train_mask.astype(mx.float32)
        dynamics_mask = mx.array(item.get("dynamics_mask", np.ones_like(item["actions"], dtype=bool))).astype(mx.bool_)
        train_count = mx.maximum(mx.sum(train_weight), 1.0)
        reward = mx.array(item["reward"])
        winner_mask = mx.array(item["winner_mask"]).astype(mx.bool_)
        next_winner_mask = mx.array(item["next_winner_mask"]).astype(mx.bool_)
        current_action_mass = mx.array(item["chorus_in"].action_mass)

        current = self.policy_forward(self.wally, *(mx.array(a) for a in item["chorus_in"]))
        following = self.policy_forward(self.wally, *(mx.array(a) for a in item["chorus_out"]))

        logpi = logp_of(current, actions_mx, controls_mx)

        value = mx.where(winner_mask, current.value_winnie, current.value_lou)
        bootstrap = mx.where(
            winner_mask,
            mx.where(next_winner_mask, following.value_winnie, 0.0),
            mx.where(next_winner_mask, 0.0, following.value_lou),
        )
        target = reward + item["discount"] * mx.stop_gradient(bootstrap)
        error = target - value

        ratio = mx.exp(logpi - behavior_logp_mx)
        self._ratio_sink.append((
            mx.stop_gradient(ratio),
            np.asarray(item.get(
                "train_mask", np.ones_like(item["actions"], dtype=bool),
            )),
        ))
        td_advantage = mx.stop_gradient(error)
        actor = -mx.sum(
            clipped_policy_surrogate(ratio, td_advantage) * train_weight
        ) / train_count

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
        if self.policy_arm == "matrix_fusion":
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
            LOSS_WEIGHTS["actor"] * actor
            + LOSS_WEIGHTS["entropy"] * entropy_penalty
            + LOSS_WEIGHTS["winnie"] * winner_loss
            + LOSS_WEIGHTS["lou"] * loser_loss
            + LOSS_WEIGHTS["vera"] * aux
            + LOSS_WEIGHTS["dina"] * dynamics_value
            + LOSS_WEIGHTS["elle"] * regularization
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

    def learn(self, items):
        items = list(items)
        if not items:
            return None
        rebuild_t0 = time.perf_counter()
        items = [self.replay.materialize(item) for item in items]
        self.rebuild_seconds += time.perf_counter() - rebuild_t0
        self.rebuild_calls += 1

        def loss_fn():
            losses, parts = zip(*(self._item_loss(item) for item in items))
            return mx.mean(mx.stack(losses)), mx.mean(mx.stack(parts), axis=0)

        self._ratio_sink = []
        scale_executor = getattr(self.wally, "scale_executor", None)
        if scale_executor is not None:
            scale_executor.begin_gradient_batch()
        (total, parts), gradients = nn.value_and_grad(self.bundle, loss_fn)()
        remote_batch = None if scale_executor is None else scale_executor.commit_gradient_batch()
        if self._ratio_sink:
            mx.eval(*(ratio for ratio, _ in self._ratio_sink))
            self.ratios.append(np.concatenate(
                [np.asarray(ratio).reshape(-1)[np.asarray(mask, dtype=bool).reshape(-1)]
                 for ratio, mask in self._ratio_sink]))
            self._ratio_sink = []
        gradients, gradient_norm = optim.clip_grad_norm(gradients, self.gradient_clip)
        self.optimizer.update(self.bundle, gradients)
        mx.eval(self.bundle.parameters(), self.optimizer.state, total, parts, gradient_norm)
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
        if remote_batch is not None:
            metrics["remote_scale_gradient_atoms"] = int(remote_batch[1])
            metrics["remote_scale_gradient_norm"] = float(remote_batch[2])
            metrics["remote_scale_updates"] = int(remote_batch[3])
        metrics["local_control_sigma_min"] = self._control_sigma(items)
        metrics.update(
            loss=float(np.asarray(total)),
            updates=self.updates,
            batch=len(items),
        )
        return metrics

    def _control_sigma(self, items):
        if self.policy_arm != "matrix_fusion":
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

    def update(self, *args, **kwargs):
        item = self.replay.push(self.transition(*args, **kwargs))
        self.transitions += 1
        return self.learn([item])

    def observe(
        self, previous, next_frame, next_snapshot, *, terminal=False,
    ):
        reward = role_rewards(previous["context"], previous["snapshot"], next_snapshot)
        self.pending.append({
            "previous": previous,
            "reward": reward,
            "immediate_next_frame": next_frame,
            "immediate_next_snapshot": next_snapshot,
        })
        changed = self._cart_signature(previous["snapshot"]) != self._cart_signature(next_snapshot)
        return self.flush(next_frame, next_snapshot, terminal=terminal) if terminal or changed or len(self.pending) >= self.credit_horizon else None

    def flush(self, next_frame, next_snapshot, *, terminal=False):
        if not self.pending:
            self.replay.release_unreferenced()
            return None
        rewards = [item["reward"] for item in self.pending]
        fresh = []
        for start, item in enumerate(self.pending):
            total = np.zeros_like(rewards[start])
            factor = 1.0
            for reward in rewards[start:]:
                total += factor * reward
                factor *= self.gamma
            previous = item["previous"]
            fresh.append(self.replay.push(self.transition(
                previous["context"], previous["frame"], next_frame,
                item["immediate_next_frame"],
                previous["snapshot"], next_snapshot,
                previous["actions"], previous["controls"], previous["behavior_logp"], sparse_return=total,
                train_mask=previous.get("train_mask"),
                bootstrap_discount=0.0 if terminal else factor,
            )))
        self.pending.clear()
        return self._train_fresh(fresh)

    def observe_attributed(self, records):
        fresh = [self.replay.push(self.transition(**record)) for record in records]
        out = self._train_fresh(fresh)
        if out is not None:
            out["attributed_groups"] = len(fresh)
            out["attributed_rows"] = int(sum(
                np.asarray(item["train_mask"], dtype=bool).sum() for item in fresh
            ))
        return out

    def _train_fresh(self, fresh):
        if not fresh:
            self.replay.release_unreferenced()
            return None
        self.transitions += len(fresh)
        self.replay.release_unreferenced()
        metrics = [self.learn(fresh)]
        ages = [0.0]
        for _ in range(self.replay_steps):
            batch = self.replay.sample(self.replay_batch, self.rng)
            if not batch:
                break
            ages.append(self.replay.mean_age(batch))
            metrics.append(self.learn(batch))
        metrics = [m for m in metrics if m]
        if not metrics:
            return None
        out = {}
        for key in sorted(set().union(*(row.keys() for row in metrics))):
            values = [row[key] for row in metrics if key in row]
            out[key] = values[-1] if key in ("updates", "batch") else float(np.mean(values))
            if len(values) != len(metrics):
                out[key + "_sample_mass"] = len(values)
        report = self.replay.report()
        out.update(
            credited_steps=len(fresh),
            gradient_steps=len(metrics),
            replay_size=len(self.replay),
            replay_capacity=self.replay.capacity,
            replay_mb=round(self.replay.nbytes / (1 << 20), 3),
            replay_bytes_per_state=report["bytes_per_transition"],
            replay_frames=report["frames"],
            replay_precision=self.replay_precision,
            replay_mean_age=round(float(np.mean(ages)), 2),
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

    @staticmethod
    def _cart_signature(snapshot):
        return tuple(np.floor(np.clip(snapshot.pos, 0, 1) * snapshot.levels).astype(np.int64)), tuple(snapshot.control.astype(np.int64))

    def save(self, path=None):
        target = path or self.checkpoint
        if target is None:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        temporary = target + ".new.npz"
        payload = {name: np.asarray(value) for name, value in tree_flatten(self.bundle.parameters())}
        for name, value in tree_flatten(self.optimizer.state):
            payload["__opt__" + name] = np.asarray(value)
        payload["__updates__"] = np.asarray(self.updates)
        payload["__transitions__"] = np.asarray(self.transitions)
        payload["__gradient_steps__"] = np.asarray(self.gradient_steps)
        payload[ARCH_KEY] = np.asarray(self.architecture)
        payload[ARCH_SPEC_KEY] = np.asarray(
            json.dumps(architecture_spec(self.bundle), separators=(",", ":"))
        )
        payload[RNG_KEY] = np.asarray(
            json.dumps(self.rng.bit_generator.state, separators=(",", ":"))
        )
        payload[POLICY_KEY] = np.asarray(self.policy_arm)
        payload[POLICY_VERSION_KEY] = np.asarray(POLICY_VERSIONS.get(self.policy_arm, POLICY_VERSIONS["linear"]))
        payload[REWARD_CONTRACT_KEY] = np.asarray(SPARSE_REWARD_FINGERPRINT)
        if self.initial_checkpoint_sha256:
            payload[LINEAGE_INITIAL_KEY] = np.asarray(self.initial_checkpoint_sha256)
        payload.update(self.replay.export_payload())
        np.savez(temporary, **payload)
        os.replace(temporary, target)
