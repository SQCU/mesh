from __future__ import annotations

import math

from . import tensor as mx
from mlx.utils import tree_flatten, tree_unflatten

from .policy_contract import POLICY_LOG_RATIO_BOUND, POLICY_ROLLOUT_IS_THRESHOLD


def norm(rows, eps=1e-6):
    scale = mx.stop_gradient(mx.maximum(mx.max(mx.abs(rows), axis=-1, keepdims=True), 1.0))
    scaled = rows / scale
    return scaled * mx.rsqrt(mx.mean(mx.square(scaled), axis=-1, keepdims=True) + eps / scale / scale)


def clip_grad_norm(gradients, limit):
    leaves = tree_flatten(gradients)
    scale = mx.maximum(mx.max(mx.stack([mx.max(mx.abs(value)) for _, value in leaves])), 1.0)
    magnitude = mx.sqrt(sum(mx.sum(mx.square(value / scale)) for _, value in leaves))
    factor = mx.minimum(1.0, (limit / scale) / mx.where(magnitude > 0, magnitude, 1.0))
    return tree_unflatten([(name, value * factor) for name, value in leaves]), scale * magnitude


def control_log_scale(raw):
    return mx.arcsinh(raw)


def clipped_policy_surrogate(log_ratio, advantage, clip):
    selected = mx.where(advantage >= 0, mx.minimum(log_ratio, math.log1p(clip)),
                        mx.maximum(log_ratio, math.log1p(-clip)))
    return mx.exp(selected) * advantage


def rollout_importance_weights(log_ratio):
    ratio = mx.exp(mx.clip(log_ratio, -POLICY_LOG_RATIO_BOUND, POLICY_LOG_RATIO_BOUND))
    return mx.stop_gradient(mx.minimum(ratio, POLICY_ROLLOUT_IS_THRESHOLD))
