from __future__ import annotations

import math

import torch

from .losses import clipped_policy_surrogate, rollout_importance_weights


def norm(rows, eps=1e-6):
    scale = torch.clamp(rows.abs().amax(dim=-1, keepdim=True), min=1.0).detach()
    scaled = rows / scale
    return scaled * torch.rsqrt(scaled.square().mean(dim=-1, keepdim=True) + eps / scale / scale)


def clip_grad_norm(gradients, limit, reduce=None):
    """57628982^:policy_math.clip_grad_norm over named gradients.  `reduce`, where some gradients are shards whose
    peers hold the rest (products.World.reduce_norm), gives the shards' global maximum and squared sum."""
    values = [value for value in gradients.values() if value is not None and value.numel()]
    local = float(torch.stack([value.abs().max() for value in values]).max()) if values else 0.0
    scale = max(reduce('max', local) if reduce is not None else local, 1.0)
    squared = float(torch.stack([(value / scale).square().sum() for value in values]).sum()) if values else 0.0
    magnitude = math.sqrt(reduce('sum', squared) if reduce is not None else squared)
    factor = min(1.0, (limit / scale) / (magnitude if magnitude > 0 else 1.0))
    return {name: None if value is None else value * factor for name, value in gradients.items()}, scale * magnitude


def control_log_scale(raw):
    return torch.asinh(raw)
