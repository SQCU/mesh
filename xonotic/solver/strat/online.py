from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .dynamics import LocalDynamics, action_rows, state_rows
from .train import policy_forward, role_rewards


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
        credit_horizon: int = 5,
    ):
        self.estimator = estimator
        self.gamma = float(gamma)
        self.importance_clip = float(importance_clip)
        self.dynamics = dynamics or LocalDynamics()
        self.bundle = nn.Module()
        self.bundle.qkv = estimator.qkv
        self.bundle.head = estimator.head
        self.bundle.value = estimator.value
        self.bundle.dynamics = self.dynamics
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.checkpoint = checkpoint
        self.updates = 0
        self.credit_horizon = max(1, int(credit_horizon))
        self.pending = []
        self.loaded = False
        if checkpoint is not None and os.path.exists(checkpoint):
            try:
                self.bundle.load_weights(checkpoint)
                self.loaded = True
            except Exception as exc:
                print(f"[online] checkpoint load failed ({exc}); continuing from policy state", flush=True)

    def update(
        self,
        sim,
        state,
        next_state,
        cartstate,
        next_cartstate,
        w_in,
        w_out,
        actions,
        behavior_logp,
        reward_override=None,
        bootstrap_discount=None,
        dynamics_next_state=None,
    ):
        actions_mx = mx.array(np.asarray(actions, dtype=np.int32))
        behavior_logp_mx = mx.array(np.asarray(behavior_logp, dtype=np.float32))
        dynamics_state_rows = state_rows(state)

        def loss_fn():
            w_next, _, target_logp, value, winner_value, loser_value = policy_forward(
                self.estimator, state, w_in, action=actions_mx
            )
            _, _, _, next_value, _, _ = policy_forward(
                self.estimator, next_state, w_out, action=actions_mx
            )
            dyn_state = mx.stop_gradient(mx.array(dynamics_state_rows))
            dyn_action = mx.stop_gradient(mx.array(action_rows(state, actions)))
            dynamics_target_state = dynamics_next_state or next_state
            target_delta = mx.stop_gradient(
                mx.array(dynamics_target_state.hierarchy - state.hierarchy)
            )
            reward = mx.array(
                role_rewards(sim, cartstate, next_cartstate)
                if reward_override is None else reward_override
            )
            discount = self.gamma if bootstrap_discount is None else bootstrap_discount
            td_target = reward + discount * mx.stop_gradient(next_value)
            td_error = td_target - value
            ratio = mx.exp(target_logp - behavior_logp_mx)
            ratio = mx.minimum(ratio, self.importance_clip)
            ratio = mx.stop_gradient(ratio)
            actor = -mx.mean(
                mx.stop_gradient(ratio * td_error) * target_logp
            )
            winner_mask = mx.array(state.winner_mask).astype(mx.float32)
            loser_mask = 1.0 - winner_mask
            winner_weight = winner_mask * ratio
            loser_weight = loser_mask * ratio
            winner_loss = mx.sum(
                mx.square(winner_value - td_target) * winner_weight
            ) / mx.maximum(mx.sum(winner_weight), 1.0)
            loser_loss = mx.sum(
                mx.square(loser_value - td_target) * loser_weight
            ) / mx.maximum(mx.sum(loser_weight), 1.0)
            dynamics_mean, dynamics_first, dynamics_second = self.dynamics(
                dyn_state, dyn_action
            )
            dynamics_value = 0.5 * (
                mx.mean(mx.square(dynamics_first - target_delta))
                + mx.mean(mx.square(dynamics_second - target_delta))
            )
            dynamics_uncertainty = mx.mean(mx.square(dynamics_first - dynamics_second))
            dynamics_error = mx.mean(mx.square(dynamics_mean - target_delta))
            regularization = mx.mean(mx.square(w_next))
            total = (
                actor
                + 0.5 * winner_loss
                + 0.5 * loser_loss
                + 0.25 * dynamics_value
                + 1e-3 * regularization
            )
            return total, (
                actor,
                winner_loss,
                loser_loss,
                dynamics_value,
                mx.mean(ratio),
                dynamics_uncertainty,
                dynamics_error,
            )

        (total, parts), gradients = nn.value_and_grad(self.bundle, loss_fn)()
        self.optimizer.update(self.bundle, gradients)
        mx.eval(self.bundle.parameters(), self.optimizer.state, total)
        self.updates += 1
        matrices = np.asarray(self.dynamics.local_matrix(mx.array(dynamics_state_rows)))
        singular = [np.linalg.svd(matrix, compute_uv=False).min() for matrix in matrices]
        return {
            "loss": float(np.asarray(total)),
            "loss_pg": float(np.asarray(parts[0])),
            "loss_w": float(np.asarray(parts[1])),
            "loss_l": float(np.asarray(parts[2])),
            "loss_dynamics": float(np.asarray(parts[3])),
            "importance_mean": float(np.asarray(parts[4])),
            "model_uncertainty": float(np.asarray(parts[5])),
            "model_one_step_error": float(np.asarray(parts[6])),
            "local_control_sigma_min": float(np.mean(singular)),
            "updates": self.updates,
        }

    def observe(self, previous, next_state, next_cartstate, *, terminal=False):
        reward = role_rewards(previous["sim"], previous["cartstate"], next_cartstate)
        self.pending.append(
            dict(
                previous=previous,
                reward=reward,
                immediate_next_state=next_state,
                immediate_next_cartstate=next_cartstate,
            )
        )
        changed = self._cart_signature(previous["cartstate"]) != self._cart_signature(next_cartstate)
        if terminal or changed or len(self.pending) >= self.credit_horizon:
            return self.flush(next_state, next_cartstate, terminal=terminal)
        return None

    def flush(self, next_state, next_cartstate, *, terminal=False):
        if not self.pending:
            return None
        metrics = []
        rewards = [item["reward"] for item in self.pending]
        for start, item in enumerate(self.pending):
            total = np.zeros_like(rewards[start])
            factor = 1.0
            for reward in rewards[start:]:
                total += factor * reward
                factor *= self.gamma
            previous = item["previous"]
            metrics.append(
                self.update(
                    previous["sim"], previous["state"], next_state,
                    previous["cartstate"], next_cartstate,
                    previous["w_in"], previous["w_out"], previous["actions"],
                    previous["behavior_logp"], reward_override=total,
                    bootstrap_discount=0.0 if terminal else factor,
                    dynamics_next_state=item["immediate_next_state"],
                )
            )
        self.pending.clear()
        keys = metrics[0].keys()
        return {
            key: metrics[-1][key] if key == "updates" else float(np.mean([row[key] for row in metrics]))
            for key in keys
        } | {"credited_steps": len(metrics)}

    @staticmethod
    def _cart_signature(cartstate):
        return (
            tuple(np.floor(np.asarray(cartstate.pos)).astype(np.int64).tolist()),
            tuple(np.asarray(cartstate.control, dtype=np.int64).tolist()),
        )

    def save(self, path=None):
        target = path or self.checkpoint
        if target is None:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        temporary = target + ".new.npz"
        flat = dict(tree_flatten(self.bundle.parameters()))
        np.savez(temporary, **{name: np.asarray(value) for name, value in flat.items()})
        os.replace(temporary, target)
