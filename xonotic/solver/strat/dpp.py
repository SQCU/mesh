from __future__ import annotations

import torch

from .policy_math import norm

from .matmul import (
    matrix_multiply,
    matrix_multiply_transpose_left,
)

def _as_f32(value):
    result = torch.as_tensor(value)
    return result if result.dtype == torch.float32 else result.to(torch.float32)

def positive_definite_conjugate_gradient(matrix, rhs):
    """57628982^'s conjugate gradient, one column a system, the matrix's order in iterations; autograd derives its
    derivative through the iterations."""
    solution = torch.zeros_like(rhs)
    residual = rhs
    direction = residual
    residual_norm = (residual * residual).sum(dim=0, keepdim=True)
    epsilon = torch.finfo(matrix.dtype).eps
    tolerance = epsilon * epsilon * torch.clamp(residual_norm, min=epsilon)
    for _ in range(int(matrix.shape[0])):
        action = matrix_multiply(matrix, direction)
        denominator = (direction * action).sum(dim=0, keepdim=True)
        active = residual_norm > tolerance
        scale = torch.where(active, residual_norm, 0.0) / torch.where(active, denominator, 1.0)
        solution = solution + direction * scale
        following = residual - action * scale
        following_norm = (following * following).sum(dim=0, keepdim=True)
        direction = following + direction * (
            torch.where(active, following_norm, 0.0) / torch.where(active, residual_norm, 1.0)
        )
        residual = following
        residual_norm = following_norm
    return solution

def dpp_marginals(quality, features, normalize=True, eps=1e-12):
    quality = _as_f32(quality)
    rows = _as_f32(features)
    if quality.ndim != 1 or rows.ndim != 2 or quality.shape[0] != rows.shape[0]:
        raise ValueError(f"quality/features shapes disagree: {tuple(quality.shape)}, {tuple(rows.shape)}")
    if normalize:
        rows = norm(rows, eps)
    weighted = quality[:, None] * rows
    covariance = torch.eye(rows.shape[1], dtype=torch.float32, device=rows.device) + matrix_multiply_transpose_left(weighted, weighted)
    dual = positive_definite_conjugate_gradient(covariance, weighted.T).T
    return torch.clamp((weighted * dual).sum(dim=1), 0, 1)

def dpp_gram_operator(basis, values):
    basis = _as_f32(basis)
    values = _as_f32(values)
    r = basis.shape[1]
    covariance = torch.eye(r, dtype=basis.dtype, device=basis.device) + matrix_multiply_transpose_left(basis, basis)
    rhs = matrix_multiply_transpose_left(basis, values)
    if basis.device.type == "mps":
        solved = torch.linalg.solve(covariance.cpu(), rhs.cpu()).to(basis.device)
    else:
        solved = torch.linalg.solve(covariance, rhs)
    return matrix_multiply(basis, solved)


__all__ = [
    "dpp_gram_operator",
    "dpp_marginals",
    "positive_definite_conjugate_gradient",
]
