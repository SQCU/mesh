from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .estimator import BELIEF_WIDTH, HIERARCHY_WIDTH, X_WIDTH

STATE_WIDTH = X_WIDTH + BELIEF_WIDTH + HIERARCHY_WIDTH
ACTION_WIDTH = 96

__all__ = [
    "STATE_WIDTH",
    "ACTION_WIDTH",
    "ActionLinearHead",
    "LocalDynamics",
    "state_rows",
    "action_rows",
    "dynamics_loss",
    "guided_actions",
]


class ActionLinearHead(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.state_up = nn.Linear(STATE_WIDTH, hidden)
        self.drift = nn.Linear(hidden, HIERARCHY_WIDTH)
        self.matrix = nn.Linear(hidden, HIERARCHY_WIDTH * ACTION_WIDTH)

    def local_matrix(self, state: mx.array) -> mx.array:
        hidden = nn.silu(self.state_up(state))
        return self.matrix(hidden).reshape(
            state.shape[0], HIERARCHY_WIDTH, ACTION_WIDTH
        )

    def __call__(self, state: mx.array, action: mx.array) -> mx.array:
        hidden = nn.silu(self.state_up(state))
        drift = self.drift(hidden)
        matrix = self.local_matrix(state)
        return drift + mx.sum(matrix * action[:, None, :], axis=-1)


class LocalDynamics(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.first = ActionLinearHead(hidden)
        self.second = ActionLinearHead(hidden)

    def __call__(self, state: mx.array, action: mx.array):
        first = self.first(state, action)
        second = self.second(state, action)
        return 0.5 * (first + second), first, second

    def uncertainty(self, state: mx.array, action: mx.array) -> mx.array:
        _, first, second = self(state, action)
        return mx.mean(mx.square(first - second), axis=-1)

    def local_matrix(self, state: mx.array) -> mx.array:
        return 0.5 * (
            self.first.local_matrix(state) + self.second.local_matrix(state)
        )


def state_rows(state) -> np.ndarray:
    return np.concatenate([state.x, state.beta, state.hierarchy], axis=1).astype(np.float32)


def action_rows(state, actions) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    players = len(actions)
    chosen_z = state.z[actions]
    out = np.zeros((players, ACTION_WIDTH), dtype=np.float32)
    for focal in range(players):
        tokens = np.concatenate(
            [chosen_z, state.relation[focal, actions]], axis=1
        )
        own = tokens[focal]
        out[focal] = np.concatenate(
            [own, tokens.mean(axis=0), tokens.max(axis=0)]
        )
    return out


def dynamics_loss(
    model: LocalDynamics,
    state: mx.array,
    action: mx.array,
    target_delta: mx.array,
) -> mx.array:
    _, first, second = model(state, action)
    return 0.5 * (
        mx.mean(mx.square(first - target_delta))
        + mx.mean(mx.square(second - target_delta))
    )


def guided_actions(
    model: LocalDynamics,
    state,
    policy_logits,
    base_actions,
    *,
    temperature: float = 1.0,
    control_weight: float = 0.5,
    exploration_weight: float = 0.05,
    max_probe: float = 1.0,
):
    logits = np.asarray(policy_logits, dtype=np.float32) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    actions = np.asarray(base_actions, dtype=np.int64).copy()
    state_mx = mx.array(state_rows(state))
    diagnostics = []
    for player in range(len(actions)):
        scores = np.zeros(logits.shape[1], dtype=np.float32)
        predicted = np.zeros(logits.shape[1], dtype=np.float32)
        uncertainty = np.zeros(logits.shape[1], dtype=np.float32)
        for action in range(logits.shape[1]):
            joint = actions.copy()
            joint[player] = action
            action_mx = mx.array(action_rows(state, joint))
            mean, first, second = model(state_mx, action_mx)
            delta = np.asarray(mean)[player]
            spread = float(np.mean(np.square(np.asarray(first)[player] - np.asarray(second)[player])))
            if state.winner_mask[player]:
                target = delta[3] + 0.5 * delta[5] - 0.25 * max(0.0, delta[1])
            else:
                target = delta[3] + 0.5 * delta[4] + 0.25 * delta[0] - 0.25 * delta[1]
            probe = min(float(np.sqrt(max(0.0, spread))), float(max_probe))
            predicted[action] = target
            uncertainty[action] = probe
            scores[action] = logp[player, action] + control_weight * target + exploration_weight * probe
        actions[player] = int(np.argmax(scores))
        diagnostics.append(
            dict(
                player=player,
                action=int(actions[player]),
                score=float(scores[actions[player]]),
                predicted_role_delta=float(predicted[actions[player]]),
                probe=float(uncertainty[actions[player]]),
            )
        )
    return actions, diagnostics
