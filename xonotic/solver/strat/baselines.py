from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .cast_header import norm
from .instruments import DESCRIPTOR_WIDTH, KINDS
from .matmul import linear, matrix_multiply, matrix_multiply_transpose_right
from .strategy import Strategy, measure_log_density, normalized_log_probs

BASELINE_OUTPUT_WIDTH = len(KINDS) + DESCRIPTOR_WIDTH + 10

class BaselinePolicy(nn.Module):
    def __init__(self, arm, input_width, hidden_width=256):
        super().__init__()
        self.arm = arm
        self.hidden = nn.Linear(input_width, hidden_width, bias=False) if arm == "ffn" else None
        self.output = nn.Linear(
            hidden_width if arm == "ffn" else input_width,
            BASELINE_OUTPUT_WIDTH,
            bias=False,
        )

    def __call__(self, rows):
        hidden = nn.silu(linear(self.hidden, rows)) if self.hidden is not None else rows
        return linear(self.output, hidden)

def baseline_strategy(
    policy,
    xan,
    zed,
    cell_slots,
    gigi,
    semantics,
    team_ids,
    weights,
    action_mass,
    delta,
    control_weight,
    exploration_weight,
):
    belief = matrix_multiply(gigi, cell_slots) / mx.maximum(mx.sum(gigi, axis=-1, keepdims=True), 1)
    features = norm(mx.concatenate([xan, semantics, belief], axis=-1))
    emitted = policy(features)
    kinds = emitted[:, :len(KINDS)]
    descriptor = emitted[:, len(KINDS):len(KINDS) + DESCRIPTOR_WIDTH]
    values = emitted[:, -10:]
    normalized_descriptors = norm(zed)
    scores = (
        matrix_multiply_transpose_right(kinds, zed[:, :len(KINDS)])
        + matrix_multiply_transpose_right(descriptor, normalized_descriptors) / (DESCRIPTOR_WIDTH ** 0.5)
    )
    logits = measure_log_density(scores, action_mass)
    zeros = mx.zeros_like(scores)
    coupling = mx.zeros((xan.shape[0], xan.shape[0]), dtype=xan.dtype)
    allocation = mx.exp(normalized_log_probs(logits))
    relation = scores[..., None]
    pooled = mx.sum(relation * allocation[..., None], axis=1)
    return Strategy(
        zeros,
        logits,
        relation,
        features,
        values[:, 0],
        values[:, 1],
        values[:, 2],
        values[:, 3],
        coupling,
        belief,
        pooled,
        weights,
        mx.zeros(scores.shape[1], dtype=xan.dtype),
        zeros,
        zeros,
        mx.zeros((0,), dtype=xan.dtype),
        mx.zeros((0,), dtype=xan.dtype),
        mx.broadcast_to(values[:, None, 4:7], (*scores.shape, 3)),
        mx.broadcast_to(values[:, None, 7:10], (*scores.shape, 3)),
        mx.ones((xan.shape[0],), dtype=xan.dtype),
    )

def default_strategy(
    policy,
    xan,
    zed,
    cell_slots,
    gigi,
    semantics,
    team_ids,
    weights,
    action_mass,
    delta,
    control_weight,
    exploration_weight,
):
    belief = matrix_multiply(gigi, cell_slots) / mx.maximum(mx.sum(gigi, axis=-1, keepdims=True), 1)
    features = norm(mx.concatenate([xan, semantics, belief], axis=-1))
    idle = mx.broadcast_to(zed[None, :, len(KINDS) - 1], action_mass.shape)
    scores = mx.log(idle)
    logits = measure_log_density(scores, action_mass)
    zeros = mx.zeros_like(scores)
    values = mx.zeros(xan.shape[0], dtype=xan.dtype)
    coupling = mx.zeros((xan.shape[0], xan.shape[0]), dtype=xan.dtype)
    relation = idle[..., None]
    allocation = mx.exp(normalized_log_probs(logits))
    return Strategy(
        zeros,
        logits,
        relation,
        features,
        values,
        values,
        values,
        values,
        coupling,
        belief,
        mx.sum(relation * allocation[..., None], axis=1),
        weights,
        mx.zeros(scores.shape[1], dtype=xan.dtype),
        zeros,
        zeros,
        mx.zeros((0,), dtype=xan.dtype),
        mx.zeros((0,), dtype=xan.dtype),
        mx.zeros((*scores.shape, 3), dtype=xan.dtype),
        mx.zeros((*scores.shape, 3), dtype=xan.dtype),
        mx.zeros((xan.shape[0],), dtype=xan.dtype),
    )
