from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "RoleValueHead",
    "StrategyValue",
    "select_role_value",
    "masked_value_loss",
    "advantage",
]


class RoleValueHead(nn.Module):
    def __init__(self, width: int, hidden: int = 32):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, 1)

    def __call__(self, rows: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(rows)))[..., 0]


class StrategyValue(nn.Module):
    def __init__(self, d_intermediate: int, d_query: int | None = None, l: int | None = None):
        super().__init__()
        self.winner = RoleValueHead(d_intermediate)
        self.loser = RoleValueHead(d_intermediate)

    def __call__(self, rows: mx.array, query: mx.array | None = None):
        return self.winner(rows), self.loser(rows)


def select_role_value(
    winner_value: mx.array,
    loser_value: mx.array,
    winner_mask: mx.array,
) -> mx.array:
    return mx.where(winner_mask, winner_value, loser_value)


def masked_value_loss(
    estimate: mx.array,
    target: mx.array,
    mask: mx.array,
) -> mx.array:
    weights = mask.astype(estimate.dtype)
    return mx.sum(mx.square(estimate - target) * weights) / mx.maximum(mx.sum(weights), 1.0)


def advantage(
    reward: mx.array,
    v_s: mx.array,
    v_next: mx.array,
    gamma: float = 1.0,
) -> mx.array:
    return reward + gamma * v_next - v_s
