from __future__ import annotations

import hashlib
import json
import os
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .dynamics import LocalDynamics, action_rows, state_rows
from .estimator import strategy_forward
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


_STATE_ARRAYS = ("x", "beta", "z", "relation", "hierarchy", "winner_mask", "w", "eligible")
_ITEM_ARRAYS = ("w_in", "w_out", "actions", "behavior_logp", "reward", "target_delta")


def _state_bytes(state) -> int:
    return sum(np.asarray(getattr(state, name)).nbytes for name in _STATE_ARRAYS
               if getattr(state, name) is not None)


class ReplayBuffer:
    """A ring of completed transitions, reused for many gradient steps.

    The learner used to consume each featurized state exactly once: `pending`
    is a CREDIT queue (it exists to compute the discounted return over a
    segment), and `flush` cleared it after one gradient step per entry.  Every
    state the server paid for was then thrown away.

    Here the completed transitions are pushed into a ring instead.  Staleness
    is bounded by the refill rate -- oldest out -- and the off-policy
    correction that licenses the reuse is the clipped importance ratio already
    in the loss (`min(exp(logpi - behavior_logp), importance_clip)`); nothing
    else is added.

    Bounded by BOTH a transition count and a memory budget, because one
    transition carries two full `StrategyState`s and the per-(player,
    instrument) relation block dominates them: at l=12 players and m=354
    instruments that block alone is 12*354*16*4 = 272 KB per state.
    """

    def __init__(self, capacity: int = 2048, memory_budget_mb: float = 256.0):
        self.capacity = int(capacity)
        self.budget = int(float(memory_budget_mb) * (1 << 20))
        self.items = deque()
        self.nbytes = 0
        self.seq = 0

    def __len__(self):
        return len(self.items)

    def push(self, item):
        item = dict(item)
        item["seq"] = self.seq
        self.seq += 1
        item["nbytes"] = (_state_bytes(item["state"]) + _state_bytes(item["next_state"])
                          + sum(np.asarray(item[k]).nbytes for k in _ITEM_ARRAYS))
        self.items.append(item)
        self.nbytes += item["nbytes"]
        while self.items and (len(self.items) > self.capacity or self.nbytes > self.budget):
            self.nbytes -= self.items.popleft()["nbytes"]
        return item

    def sample(self, n, rng):
        if not self.items:
            return []
        picks = rng.integers(0, len(self.items), size=min(int(n), len(self.items)))
        return [self.items[int(i)] for i in picks]

    def mean_age(self, sampled):
        if not sampled:
            return 0.0
        return float(np.mean([self.seq - item["seq"] for item in sampled]))

    # --- persistence: part of the atomic checkpoint (R9) ---
    def to_arrays(self) -> dict:
        out = {"__replay_n__": np.asarray(len(self.items)),
               "__replay_seq__": np.asarray(self.seq)}
        for i, item in enumerate(self.items):
            for side in ("state", "next_state"):
                st = item[side]
                for name in _STATE_ARRAYS:
                    out[f"__replay__{i}/{side}/{name}"] = np.asarray(getattr(st, name))
                out[f"__replay__{i}/{side}/teams"] = np.asarray(st.teams, dtype=np.int64)
                out[f"__replay__{i}/{side}/team_of"] = np.asarray(st.team_of, dtype=np.int64)
            for name in _ITEM_ARRAYS:
                out[f"__replay__{i}/{name}"] = np.asarray(item[name])
            out[f"__replay__{i}/discount"] = np.asarray(float(item["discount"]))
            out[f"__replay__{i}/seq"] = np.asarray(int(item["seq"]))
        return out

    def load_arrays(self, data, keys):
        from .estimator import StrategyState

        if "__replay_n__" not in keys:
            return 0
        n = int(np.asarray(data["__replay_n__"]))
        self.items.clear()
        self.nbytes = 0
        for i in range(n):
            sides = {}
            for side in ("state", "next_state"):
                fields = {name: np.asarray(data[f"__replay__{i}/{side}/{name}"])
                          for name in _STATE_ARRAYS}
                sides[side] = StrategyState(
                    fields["x"], fields["beta"], fields["z"], fields["relation"],
                    fields["hierarchy"], fields["winner_mask"], fields["w"], None,
                    tuple(int(t) for t in np.asarray(data[f"__replay__{i}/{side}/teams"])),
                    tuple(int(t) for t in np.asarray(data[f"__replay__{i}/{side}/team_of"])),
                    fields["eligible"],
                )
            item = {"state": sides["state"], "next_state": sides["next_state"],
                    "discount": float(np.asarray(data[f"__replay__{i}/discount"]))}
            for name in _ITEM_ARRAYS:
                item[name] = np.asarray(data[f"__replay__{i}/{name}"])
            item["seq"] = int(np.asarray(data[f"__replay__{i}/seq"]))
            item["nbytes"] = (_state_bytes(item["state"]) + _state_bytes(item["next_state"])
                              + sum(np.asarray(item[k]).nbytes for k in _ITEM_ARRAYS))
            self.items.append(item)
            self.nbytes += item["nbytes"]
        self.seq = int(np.asarray(data["__replay_seq__"]))
        return len(self.items)


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
        replay_capacity: int = 2048,
        replay_memory_mb: float = 256.0,
        replay_batch: int = 8,
        replay_steps: int = 4,
        seed: int = 20260831,
    ):
        self.estimator = estimator
        self.gamma = float(gamma)
        self.importance_clip = float(importance_clip)
        self.dynamics = dynamics or LocalDynamics()
        self.bundle = nn.Module()
        self.bundle.qkv = estimator.qkv
        self.bundle.encoder = estimator.encoder
        self.bundle.head = estimator.head
        self.bundle.value = estimator.value
        self.bundle.dynamics = self.dynamics
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.checkpoint = checkpoint
        self.updates = 0
        self.credit_horizon = max(1, int(credit_horizon))
        self.pending = []
        self.replay = ReplayBuffer(replay_capacity, replay_memory_mb)
        self.replay_batch = max(1, int(replay_batch))
        self.replay_steps = max(0, int(replay_steps))
        self.rng = np.random.default_rng(seed)
        self.transitions = 0
        self.gradient_steps = 0
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
        restored = self.replay.load_arrays(data, keys)
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
        state,
        next_state,
        snapshot,
        next_snapshot,
        w_in,
        w_out,
        actions,
        behavior_logp,
        reward_override=None,
        bootstrap_discount=None,
        dynamics_next_state=None,
    ):
        """One completed, replayable transition — everything the loss needs.

        The reward and the dynamics target are resolved HERE, at collection
        time, so a replayed item never needs the `GameContext`/`CartSnapshot`
        it came from and the buffer stores only arrays.
        """
        target_state = dynamics_next_state or next_state
        reward = (role_rewards(context, snapshot, next_snapshot)
                  if reward_override is None else np.asarray(reward_override))
        return {
            "state": state,
            "next_state": next_state,
            "w_in": np.asarray(w_in, dtype=np.float32),
            "w_out": np.asarray(w_out, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int32),
            "behavior_logp": np.asarray(behavior_logp, dtype=np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "target_delta": np.asarray(target_state.hierarchy - state.hierarchy, dtype=np.float32),
            "discount": self.gamma if bootstrap_discount is None else float(bootstrap_discount),
        }

    def _item_loss(self, item):
        state, next_state = item["state"], item["next_state"]
        actions_mx = mx.array(item["actions"])
        behavior_logp_mx = mx.array(item["behavior_logp"])
        dynamics_state_rows = state_rows(state)
        current = strategy_forward(self.estimator, state, item["w_in"], action=actions_mx)
        following = strategy_forward(self.estimator, next_state, item["w_out"], action=actions_mx)
        dynamics_state = mx.stop_gradient(mx.array(dynamics_state_rows))
        dynamics_action = mx.stop_gradient(mx.array(action_rows(state, item["actions"])))
        target_delta = mx.stop_gradient(mx.array(item["target_delta"]))
        reward = mx.array(item["reward"])
        target = reward + item["discount"] * mx.stop_gradient(following["value"])
        error = target - current["value"]
        ratio = mx.stop_gradient(mx.minimum(
            mx.exp(current["logpi"] - behavior_logp_mx), self.importance_clip))
        actor = -mx.mean(mx.stop_gradient(ratio * error) * current["logpi"])
        winner_mask = mx.array(state.winner_mask).astype(mx.float32)
        winner_weight = winner_mask * ratio
        loser_weight = (1.0 - winner_mask) * ratio
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
        return total, mx.stack([actor, winner_loss, loser_loss, dynamics_value,
                                regularization, mx.mean(ratio), dynamics_uncertainty,
                                dynamics_error])

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

        def loss_fn():
            losses, parts = zip(*(self._item_loss(item) for item in items))
            return mx.mean(mx.stack(losses)), mx.mean(mx.stack(parts), axis=0)

        (total, parts), gradients = nn.value_and_grad(self.bundle, loss_fn)()
        self.optimizer.update(self.bundle, gradients)
        mx.eval(self.bundle.parameters(), self.optimizer.state, total, parts)
        self.updates += 1
        self.gradient_steps += 1
        rows = state_rows(items[-1]["state"])
        matrices = np.asarray(self.dynamics.local_matrix(mx.array(rows)))
        singular = [np.linalg.svd(matrix, compute_uv=False).min() for matrix in matrices]
        names = ("loss_pg", "loss_w", "loss_l", "loss_dynamics", "loss_reg",
                 "importance_mean", "model_uncertainty", "model_one_step_error")
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

    def observe(self, previous, next_state, next_snapshot, *, terminal=False):
        reward = role_rewards(previous["context"], previous["snapshot"], next_snapshot)
        self.pending.append({
            "previous": previous,
            "reward": reward,
            "immediate_next_state": next_state,
            "immediate_next_snapshot": next_snapshot,
        })
        changed = self._cart_signature(previous["snapshot"]) != self._cart_signature(next_snapshot)
        return self.flush(next_state, next_snapshot, terminal=terminal) if terminal or changed or len(self.pending) >= self.credit_horizon else None

    def flush(self, next_state, next_snapshot, *, terminal=False):
        if not self.pending:
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
                previous["context"], previous["state"], next_state,
                previous["snapshot"], next_snapshot, previous["w_in"], previous["w_out"],
                previous["actions"], previous["behavior_logp"], reward_override=total,
                bootstrap_discount=0.0 if terminal else factor,
                dynamics_next_state=item["immediate_next_state"],
            )))
        self.transitions += len(fresh)
        self.pending.clear()

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
        out.update(
            credited_steps=len(fresh),
            gradient_steps=len(metrics),
            replay_size=len(self.replay),
            replay_capacity=self.replay.capacity,
            replay_mb=round(self.replay.nbytes / (1 << 20), 3),
            replay_mean_age=round(float(np.mean(ages)), 2),
            steps_per_transition=round(self.gradient_steps / max(1, self.transitions), 3),
        )
        return out

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
