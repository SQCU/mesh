from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RoleValueHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.probe = nn.Linear(width, 1)

    def __call__(self, rows: mx.array) -> mx.array:
        return self.probe(rows)[..., 0]


class StrategyValue(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.winner = RoleValueHead(width)
        self.loser = RoleValueHead(width)

    def __call__(self, rows: mx.array):
        return self.winner(rows), self.loser(rows)


def select_role_value(winner_value: mx.array, loser_value: mx.array, winner_mask: mx.array):
    return mx.where(winner_mask, winner_value, loser_value)


def masked_value_loss(estimate: mx.array, target: mx.array, mask: mx.array):
    weights = mask.astype(estimate.dtype)
    return mx.sum(mx.square(estimate - target) * weights) / mx.maximum(mx.sum(weights), 1.0)


def advantage(reward: mx.array, v_s: mx.array, v_next: mx.array, gamma: float = 1.0):
    return reward + gamma * v_next - v_s


__all__ = ["RoleValueHead", "StrategyValue", "select_role_value", "masked_value_loss", "advantage"]
