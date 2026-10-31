from __future__ import annotations

import torch


def matrix_multiply(lhs, rhs):
    return lhs @ rhs


def matrix_multiply_transpose_left(lhs, rhs):
    return lhs.transpose(-2, -1) @ rhs


def matrix_multiply_transpose_right(lhs, rhs):
    return lhs @ rhs.transpose(-2, -1)


def linear(layer, rows):
    shape = rows.shape
    product = matrix_multiply_transpose_right(rows.reshape(-1, shape[-1]), layer.weight)
    if getattr(layer, "bias", None) is not None:
        product = product + layer.bias
    return product.reshape(*shape[:-1], product.shape[-1])


def batched_matrix_vector(matrices, vectors):
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            f"matrix/vector inner dimensions disagree: {matrices.shape[-1]} and {vectors.shape[-1]}"
        )
    return (matrices * vectors[..., None, :]).sum(dim=-1)


def expert_matrix_multiply(rows, weights, experts):
    """Indexed matrix operands; the shared product lowerer consumes the index pages."""
    return torch.bmm(rows[:, None, :], weights.index_select(0, experts))[:, 0, :]



__all__ = [
    "batched_matrix_vector",
    "expert_matrix_multiply",
    "linear",
    "matrix_multiply",
    "matrix_multiply_transpose_left",
    "matrix_multiply_transpose_right",
]
