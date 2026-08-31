from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class QKVShapes:
    k_teams: int
    j_instruments: int
    l_players: int
    d_x: int
    d_beta: int
    d_z: int
    d: int

    def __post_init__(self):
        if min(self.k_teams, self.j_instruments, self.l_players, self.d_x, self.d_beta, self.d_z, self.d) < 1:
            raise ValueError("projection dimensions must be positive")


class QKVProjector(nn.Module):
    def __init__(self, shapes: QKVShapes, *, seed: Optional[int] = None):
        super().__init__()
        self.shapes = shapes
        if seed is not None:
            mx.random.seed(seed)
        self.W_q = mx.random.uniform(
            low=-(shapes.d_x + shapes.d_beta) ** -0.5,
            high=(shapes.d_x + shapes.d_beta) ** -0.5,
            shape=(shapes.d, shapes.d_x + shapes.d_beta),
        )
        self.W_k = mx.random.uniform(
            low=-shapes.d_z ** -0.5,
            high=shapes.d_z ** -0.5,
            shape=(shapes.d, shapes.d_z),
        )

    def query(self, x, beta):
        return mx.concatenate((mx.array(x), mx.array(beta)), axis=1) @ self.W_q.T

    def key(self, z):
        return mx.array(z) @ self.W_k.T

    def learned_params(self):
        return {"W_q": self.W_q, "W_k": self.W_k}


__all__ = ["QKVShapes", "QKVProjector"]
