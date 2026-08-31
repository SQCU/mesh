from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .cast_header import Wally, Widths, elle
from .strategy import strategy, dynamics
from .replay_store import RawReplayBuffer
from .runtime import role_rewards


ARCH_KEY = "__arch__"
ARCH_SPEC_KEY = "__arch_spec__"


def architecture_spec(module) -> list:
    """Sorted ``[name, shape]`` list for every parameter leaf of ``module``.

    This IS the architecture as far as a checkpoint is concerned: change the IR
    width, add the Gram's learned metric, swap an MLP value head for a linear
    probe, and this list changes.
    """
    return sorted(
        [name, [int(d) for d in value.shape]]
        for name, value in tree_flatten(module.parameters())
    )


def architecture_fingerprint(module) -> str:
    return hashlib.sha256(
        json.dumps(architecture_spec(module), separators=(",", ":")).encode()
    ).hexdigest()[:16]


REPLAY_NOTE = """The ring stores RAW ENGINE ROWS, not derived features.

`replay_store.RawReplayBuffer` keeps the per-player OBS rows, the cart rows,
the action, the behaviour log-prob and the credited return, and rebuilds `x`,
`hierarchy`, `winner_mask`, `z`, `relation` and `eligible` at SAMPLE time
through `replay_store.featurize_tick` -- the same function the live responder
calls, so a replayed state is identical to the live one by construction.

The R24 ring cached `StrategyState`s instead.  On the real Game-2 shape that is
374 KB per transition with the dense dense per-pair relation block (now deleted) at
56% of it, and at the design shape (l=256) the relation block is over 90% and a
transition costs megabytes -- so the ring was MEMORY bound, holding 574
transitions, and the "data-bound" limit of R24/R25 was a storage-format defect.
"""


class CheckpointArchitectureMismatch(RuntimeError):
    """A checkpoint was written by a DIFFERENT architecture than the live model.

    Raised instead of partially loading.  ``load_weights(..., strict=False)``
    used to swallow this: a 128d model resuming a 16d checkpoint restored
    almost nothing and reported success.
    """


class OnlineLearner:
    def __init__(
        self,
        estimator,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.95,
        importance_clip: float = 2.0,
        dynamics=None,
        checkpoint=None,
        load_checkpoint=None,
        credit_horizon: int = 5,
        on_architecture_mismatch: str = "refuse",
        replay_capacity: int = 200000,
        replay_memory_mb: float = 256.0,
        replay_precision: str = "float32",
        replay_batch: int = 8,
        replay_steps: int = 4,
        seed: int = 20260831,
    ):
        self.estimator = estimator
        self.gamma = float(gamma)
        self.importance_clip = float(importance_clip)
        self.dynamics = dynamics or LocalDynamics()
        self.bundle = nn.Module()
        self.bundle.dynamics = self.dynamics
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.checkpoint = checkpoint
        self.updates = 0
        self.credit_horizon = max(1, int(credit_horizon))
        self.pending = []
        self.replay = RawReplayBuffer(replay_capacity, replay_memory_mb,
                                      precision=replay_precision)
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
        if on_architecture_mismatch not in ("refuse", "reinit"):
            raise ValueError("on_architecture_mismatch must be 'refuse' or 'reinit'")
        self.on_architecture_mismatch = on_architecture_mismatch
        self.architecture = architecture_fingerprint(self.bundle)
        self.loaded = False
        self.loaded_optimizer = False
        source = load_checkpoint or checkpoint
        if source is not None and os.path.exists(source):
            self._load_full(source)

    def _load_full(self, source):
        """Resume, or refuse.  Never a silent partial load.

        The checkpoint carries the architecture fingerprint that wrote it.  If
        it disagrees with the live model -- or is absent, which is what every
        pre-fingerprint checkpoint looks like -- the resume is REFUSED unless
        the caller explicitly asked for ``on_architecture_mismatch="reinit"``,
        in which case the mismatch is announced and training starts from a
        fresh initialization instead of a hollowed-out one.
        """
        data = np.load(source, allow_pickle=False)
        keys = list(data.files)
        stored = str(data[ARCH_KEY]) if ARCH_KEY in keys else None
        if stored != self.architecture:
            local = json.dumps(architecture_spec(self.bundle), separators=(",", ":"))
            saved = str(data[ARCH_SPEC_KEY]) if ARCH_SPEC_KEY in keys else "(none recorded)"
            message = (
                f"checkpoint {source} was written by architecture {stored!r}; "
                f"this model is {self.architecture!r}. Refusing to partially load.\n"
                f"  checkpoint parameters: {saved}\n"
                f"  live parameters:       {local}"
            )
            if self.on_architecture_mismatch == "refuse":
                raise CheckpointArchitectureMismatch(message)
            print(f"[online] ARCHITECTURE MISMATCH -- re-initializing.\n{message}", flush=True)
            return
        weights = [
            (key, mx.array(data[key]))
            for key in keys
            if not key.startswith("__")
        ]
        self.bundle.load_weights(weights, strict=True)
        self.loaded = True
        if "__updates__" in keys:
            self.updates = int(np.asarray(data["__updates__"]))
        if "__transitions__" in keys:
            self.transitions = int(np.asarray(data["__transitions__"]))
        if "__gradient_steps__" in keys:
            self.gradient_steps = int(np.asarray(data["__gradient_steps__"]))
        restored, replay_note = self.replay.load_arrays(data, keys)
        if replay_note:
            print(f"[online] replay buffer NOT restored: {replay_note}", flush=True)
        moments = [(key[7:], mx.array(data[key])) for key in keys if key.startswith("__opt__")]
        if moments:
            self.optimizer.init(self.bundle.trainable_parameters())
            self.optimizer.state = tree_unflatten(moments)
            self.loaded_optimizer = True
        print(
            f"[online] resumed {source}: arch={self.architecture} weights={self.loaded} "
            f"optimizer={self.loaded_optimizer} updates={self.updates} "
            f"replay={restored}/{self.replay.capacity} "
            f"({self.replay.nbytes / (1 << 20):.1f} MB) transitions={self.transitions}",
            flush=True,
        )

    def transition(
        self,
        context,
        frame,
        next_frame,
        dyn_frame,
        snapshot,
        next_snapshot,
        actions,
        behavior_logp,
        reward_override=None,
        bootstrap_discount=None,
    ):
        """One completed, replayable transition — three frame references and
        the three per-player vectors that are NOT recomputable.

        `frame`/`next_frame`/`dyn_frame` are ids into the ring's interned frame
        table.  `w_in` is `frame.w`; `w_out` is `dyn_frame.w` (the responder's
        `previous["w_out"] = w_in.copy()` of the immediate successor is exactly
        that array); `target_delta` is `dyn.hierarchy - state.hierarchy`, both
        of which come back out of the cart rows.  None of them are stored.

        The reward is resolved HERE, at collection time, so a replayed item
        never needs the `GameContext`/`CartSnapshot` it came from.
        """
        state = self.replay.frames[frame].state()
        players = np.asarray(state.hierarchy).shape[0]
        # Every array below is indexed BY PLAYER, so a transition whose
        # endpoints have different player counts is not a transition of the
        # same object.  Caught here with the two shapes named.  Callers close
        # the credit segment at a roster change rather than crediting across it.
        for name, other in (("next_state", next_frame), ("dynamics target", dyn_frame)):
            rows = self.replay.frames[other].obs_int.shape[0]
            if rows != players:
                raise ValueError(
                    f"cross-roster transition: state has {players} players, {name} has "
                    f"{rows}. A roster change must close the credit segment, not be "
                    f"credited across."
                )
        reward = (role_rewards(context, snapshot, next_snapshot)
                  if reward_override is None else np.asarray(reward_override))
        return {
            "frame": frame,
            "next_frame": next_frame,
            "dyn_frame": dyn_frame,
            "actions": np.asarray(actions, dtype=np.int32),
            "behavior_logp": np.asarray(behavior_logp, dtype=np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "discount": self.gamma if bootstrap_discount is None else float(bootstrap_discount),
        }

    def _item_loss(self, item):
        state, next_state = item["state"], item["next_state"]
        actions_mx = mx.array(item["actions"])
        behavior_logp_mx = mx.array(item["behavior_logp"])
        dynamics_state_rows = state_rows(state)
        current = strategy(self.wally, *item["chorus_in"])
        following = strategy(self.wally, *item["chorus_out"])
        dynamics_state = mx.stop_gradient(mx.array(dynamics_state_rows))
        dynamics_action = mx.stop_gradient(mx.array(action_rows(state, item["actions"])))
        target_delta = mx.stop_gradient(mx.array(item["target_delta"]))
        reward = mx.array(item["reward"])
        winner_mask = mx.array(state.winner_mask).astype(mx.bool_)
        next_winner_mask = mx.array(next_state.winner_mask).astype(mx.bool_)
        bootstrap = mx.where(
            winner_mask,
            mx.where(next_winner_mask, following["winner_value"], 0.0),
            mx.where(next_winner_mask, 0.0, following["loser_value"]),
        )
        target = reward + item["discount"] * mx.stop_gradient(bootstrap)
        error = target - current["value"]
        ratio = mx.stop_gradient(mx.minimum(
            mx.exp(current["logpi"] - behavior_logp_mx), self.importance_clip))
        self._ratio_sink.append(ratio)
        actor = -mx.mean(mx.stop_gradient(ratio * error) * current["logpi"])
        winner_weight = winner_mask.astype(mx.float32) * ratio
        loser_weight = (~winner_mask).astype(mx.float32) * ratio
        winner_loss = mx.sum(mx.square(current["winner_value"] - target) * winner_weight) / mx.maximum(mx.sum(winner_weight), 1.0)
        loser_loss = mx.sum(mx.square(current["loser_value"] - target) * loser_weight) / mx.maximum(mx.sum(loser_weight), 1.0)
        dynamics_mean, dynamics_first, dynamics_second = self.dynamics(dynamics_state, dynamics_action)
        dynamics_value = 0.5 * (
            mx.mean(mx.square(dynamics_first - target_delta))
            + mx.mean(mx.square(dynamics_second - target_delta))
        )
        dynamics_uncertainty = mx.mean(mx.square(dynamics_first - dynamics_second))
        dynamics_error = mx.mean(mx.square(dynamics_mean - target_delta))
        regularization = mx.mean(mx.square(current["w_next"]))
        total = (actor + 0.5 * winner_loss + 0.5 * loser_loss
                 + 0.25 * dynamics_value + 1e-3 * regularization)
        advantage = mx.mean(mx.stop_gradient(error))
        weighted_advantage = mx.mean(mx.stop_gradient(ratio * error))
        winner_count = mx.sum(winner_mask.astype(mx.float32))
        loser_count = mx.sum((~winner_mask).astype(mx.float32))
        winner_advantage = mx.sum(mx.stop_gradient(ratio * error) * winner_mask) / mx.maximum(winner_count, 1.0)
        loser_advantage = mx.sum(mx.stop_gradient(ratio * error) * (~winner_mask)) / mx.maximum(loser_count, 1.0)
        winner_reward = mx.sum(reward * winner_mask) / mx.maximum(winner_count, 1.0)
        loser_reward = mx.sum(reward * (~winner_mask)) / mx.maximum(loser_count, 1.0)
        role_change = mx.mean((winner_mask != next_winner_mask).astype(mx.float32))
        return total, mx.stack([actor, winner_loss, loser_loss, dynamics_value,
                                regularization, mx.mean(ratio), dynamics_uncertainty,
                                dynamics_error, advantage, weighted_advantage,
                                winner_advantage, loser_advantage, winner_reward,
                                loser_reward, winner_count, loser_count, role_change])

    def learn(self, items):
        """ONE gradient step on a minibatch of transitions.

        The rows of different transitions have different player and instrument
        counts, so the minibatch is the MEAN of the per-transition scalar
        losses rather than a stacked tensor — the operator is count-invariant,
        so this is the only shape-agnostic way to batch it.
        """
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
        (total, parts), gradients = nn.value_and_grad(self.bundle, loss_fn)()
        if self._ratio_sink:
            mx.eval(*self._ratio_sink)
            self.ratios.append(np.concatenate(
                [np.asarray(r).reshape(-1) for r in self._ratio_sink]))
            self._ratio_sink = []
        self.optimizer.update(self.bundle, gradients)
        mx.eval(self.bundle.parameters(), self.optimizer.state, total, parts)
        self.updates += 1
        self.gradient_steps += 1
        rows = state_rows(items[-1]["state"])
        matrices = np.asarray(self.dynamics.local_matrix(mx.array(rows)))
        singular = [np.linalg.svd(matrix, compute_uv=False).min() for matrix in matrices]
        names = ("loss_pg", "loss_w", "loss_l", "loss_dynamics", "loss_reg",
                 "importance_mean", "model_uncertainty", "model_one_step_error",
                 "advantage", "advantage_importance_weighted", "advantage_w",
                 "advantage_l", "reward_w", "reward_l", "winner_rows",
                 "loser_rows", "role_change_fraction")
        parts = np.asarray(parts)
        metrics = {name: float(parts[i]) for i, name in enumerate(names)}
        metrics.update(
            loss=float(np.asarray(total)),
            local_control_sigma_min=float(np.mean(singular)),
            updates=self.updates,
            batch=len(items),
        )
        return metrics

    def update(self, *args, **kwargs):
        """Collect one transition, buffer it, and take one step on it."""
        item = self.replay.push(self.transition(*args, **kwargs))
        self.transitions += 1
        return self.learn([item])

    def observe(self, previous, next_frame, next_snapshot, *, terminal=False):
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
                previous["actions"], previous["behavior_logp"], reward_override=total,
                bootstrap_discount=0.0 if terminal else factor,
            )))
        self.transitions += len(fresh)
        self.pending.clear()
        self.replay.release_unreferenced()

        # The fresh segment enters immediately; then the ring is replayed, so a
        # collected state is trained on many times instead of exactly once.
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
        out = {
            key: metrics[-1][key] if key in ("updates", "batch")
            else float(np.mean([row[key] for row in metrics]))
            for key in metrics[0]
        }
        report = self.replay.report()
        out.update(
            credited_steps=len(fresh),
            gradient_steps=len(metrics),
            replay_size=len(self.replay),
            replay_capacity=self.replay.capacity,
            replay_mb=round(self.replay.nbytes / (1 << 20), 3),
            replay_bytes_per_state=report["bytes_per_transition"],
            replay_frames=report["frames"],
            replay_target_tables=report["target_tables"],
            replay_precision=report["precision"],
            replay_mean_age=round(float(np.mean(ages)), 2),
            steps_per_transition=round(self.gradient_steps / max(1, self.transitions), 3),
            feature_rebuild_ms=round(1000.0 * self.rebuild_seconds / max(1, self.rebuild_calls), 3),
            importance_ratio=self.ratio_report(),
        )
        return out

    def ratio_report(self):
        """The distribution of the clipped participant-local importance ratio.

        The ring now holds orders of magnitude more experience, so the mean
        sample age is much older by design.  This is what lets the clip be
        judged against the staleness that actually occurs instead of against
        an assumed one.
        """
        if not self.ratios:
            return None
        values = np.concatenate(self.ratios)
        quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        return {
            "n": int(values.size),
            "mean": round(float(values.mean()), 5),
            "clipped_fraction": round(float(np.mean(values >= self.importance_clip - 1e-6)), 5),
            "quantiles": [round(float(q), 5) for q in quantiles],
        }

    @staticmethod
    def _cart_signature(snapshot):
        return tuple(np.floor(snapshot.pos).astype(np.int64)), tuple(snapshot.control.astype(np.int64))

    def save(self, path=None):
        target = path or self.checkpoint
        if target is None:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        temporary = target + ".new.npz"
        payload = {name: np.asarray(value) for name, value in tree_flatten(self.bundle.parameters())}
        for name, value in tree_flatten(self.optimizer.state):
            try:
                payload["__opt__" + name] = np.asarray(value)
            except Exception:
                pass
        payload["__updates__"] = np.asarray(self.updates)
        payload["__transitions__"] = np.asarray(self.transitions)
        payload["__gradient_steps__"] = np.asarray(self.gradient_steps)
        payload.update(self.replay.to_arrays())   # the buffer is checkpoint state
        payload[ARCH_KEY] = np.asarray(self.architecture)
        payload[ARCH_SPEC_KEY] = np.asarray(
            json.dumps(architecture_spec(self.bundle), separators=(",", ":"))
        )
        np.savez(temporary, **payload)
        os.replace(temporary, target)
