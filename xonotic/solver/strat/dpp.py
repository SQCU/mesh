from __future__ import annotations

import mlx.core as mx

from .policy_math import norm

from .matmul import (
    matrix_multiply,
    matrix_multiply_transpose_left,
    matrix_multiply_transpose_right,
)

def _as_f32(value):
    result = mx.array(value)
    return result if result.dtype == mx.float32 else result.astype(mx.float32)

@mx.custom_function
def positive_definite_conjugate_gradient(matrix, rhs):
    solution = mx.zeros_like(rhs)
    residual = rhs
    direction = residual
    residual_norm = mx.sum(residual * residual, axis=0, keepdims=True)
    epsilon = mx.array(mx.finfo(matrix.dtype).eps, dtype=matrix.dtype)
    tolerance = epsilon * epsilon * mx.maximum(residual_norm, epsilon)
    for _ in range(int(matrix.shape[0])):
        action = matrix_multiply(matrix, direction)
        denominator = mx.sum(direction * action, axis=0, keepdims=True)
        active = residual_norm > tolerance
        scale = mx.where(active, residual_norm, 0.0) / mx.where(active, denominator, 1.0)
        solution = solution + direction * scale
        following = residual - action * scale
        following_norm = mx.sum(following * following, axis=0, keepdims=True)
        direction = following + direction * (
            mx.where(active, following_norm, 0.0) / mx.where(active, residual_norm, 1.0)
        )
        residual = following
        residual_norm = following_norm
    return solution

@positive_definite_conjugate_gradient.vjp
def _positive_definite_conjugate_gradient_vjp(primals, cotangent, output):
    matrix, rhs = primals
    adjoint = positive_definite_conjugate_gradient(matrix.T, cotangent)
    return -matrix_multiply_transpose_right(adjoint, output), adjoint

def dpp_marginals(quality, features, normalize=True, eps=1e-12):
    quality = _as_f32(quality)
    rows = _as_f32(features)
    if quality.ndim != 1 or rows.ndim != 2 or quality.shape[0] != rows.shape[0]:
        raise ValueError(f"quality/features shapes disagree: {quality.shape}, {rows.shape}")
    if normalize:
        rows = norm(rows, eps)
    weighted = quality[:, None] * rows
    covariance = mx.eye(rows.shape[1], dtype=mx.float32) + matrix_multiply_transpose_left(weighted, weighted)
    dual = positive_definite_conjugate_gradient(covariance, weighted.T).T
    return mx.clip(mx.sum(weighted * dual, axis=1), 0, 1)

__all__ = [
    "dpp_marginals",
    "positive_definite_conjugate_gradient",
]
