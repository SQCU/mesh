from __future__ import annotations

import mlx.core as mx

from .matmul import (
    matrix_multiply,
    matrix_multiply_transpose_left,
    matrix_multiply_transpose_right,
)

def _as_f32(value):
    result = mx.array(value)
    return result if result.dtype == mx.float32 else result.astype(mx.float32)

def _rms_norm(rows, eps):
    return rows * mx.rsqrt(mx.mean(rows * rows, axis=1, keepdims=True) + eps)

def feature_gram(features, normalize=True, eps=1e-12):
    rows = _as_f32(features)
    if rows.ndim != 2:
        raise ValueError(f"features must be 2-D; got {rows.shape}")
    if normalize:
        rows = _rms_norm(rows, eps)
    gram = matrix_multiply_transpose_right(rows, rows)
    return 0.5 * (gram + gram.T)

def build_L(quality, features, normalize=True, eps=1e-12):
    quality = _as_f32(quality)
    if quality.ndim != 1:
        raise ValueError(f"quality must be 1-D; got {quality.shape}")
    gram = feature_gram(features, normalize, eps)
    if gram.shape[0] != quality.shape[0]:
        raise ValueError(f"quality N={quality.shape[0]} disagrees with features N={gram.shape[0]}")
    kernel = quality[:, None] * gram * quality[None, :]
    return 0.5 * (kernel + kernel.T)

def _prepare(kernel):
    kernel = _as_f32(kernel)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"L must be square; got {kernel.shape}")
    kernel = 0.5 * (kernel + kernel.T)
    eye = mx.eye(kernel.shape[0], dtype=mx.float32)
    return kernel, eye

def positive_definite_conjugate_gradient(matrix, rhs):
    solution = mx.zeros_like(rhs)
    residual = rhs
    direction = residual
    residual_norm = mx.sum(residual * residual, axis=0, keepdims=True)
    initial_norm = residual_norm
    epsilon = mx.array(mx.finfo(matrix.dtype).eps, dtype=matrix.dtype)
    matrix_scale = mx.mean(mx.abs(mx.diagonal(matrix)))
    norm_floor = epsilon * mx.maximum(initial_norm, epsilon)
    action_floor = norm_floor * mx.maximum(matrix_scale, epsilon)
    for _ in range(int(matrix.shape[0])):
        action = matrix_multiply(matrix, direction)
        denominator = mx.sum(direction * action, axis=0, keepdims=True)
        scale = residual_norm / (denominator + action_floor)
        solution = solution + direction * scale
        following = residual - action * scale
        following_norm = mx.sum(following * following, axis=0, keepdims=True)
        direction = following + direction * (
            following_norm / (residual_norm + norm_floor)
        )
        residual = following
        residual_norm = following_norm
    return solution

def marginal_inclusion(kernel):
    kernel, eye = _prepare(kernel)
    return mx.clip(mx.diagonal(positive_definite_conjugate_gradient(eye + kernel, kernel)), 0, 1)

def marginal_kernel(kernel):
    kernel, eye = _prepare(kernel)
    result = positive_definite_conjugate_gradient(eye + kernel, kernel)
    return 0.5 * (result + result.T)

def dpp_marginals(quality, features, normalize=True, eps=1e-12):
    quality = _as_f32(quality)
    rows = _as_f32(features)
    if quality.ndim != 1 or rows.ndim != 2 or quality.shape[0] != rows.shape[0]:
        raise ValueError(f"quality/features shapes disagree: {quality.shape}, {rows.shape}")
    if normalize:
        rows = _rms_norm(rows, eps)
    weighted = quality[:, None] * rows
    covariance = mx.eye(rows.shape[1], dtype=mx.float32) + matrix_multiply_transpose_left(weighted, weighted)
    dual = positive_definite_conjugate_gradient(covariance, weighted.T).T
    return mx.clip(mx.sum(weighted * dual, axis=1), 0, 1)

__all__ = [
    "build_L",
    "dpp_marginals",
    "feature_gram",
    "marginal_inclusion",
    "marginal_kernel",
    "positive_definite_conjugate_gradient",
]
