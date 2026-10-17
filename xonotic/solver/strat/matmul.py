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
    """57628982^:paged_matrix.experts: each row through its expert's matrix, rows grouped per expert (the routed
    table), one product an expert present."""
    output = rows.new_zeros((rows.shape[0], weights.shape[2]))
    for expert in torch.unique(experts).tolist():
        selected = torch.nonzero(experts == expert).reshape(-1)
        output = output.index_add(0, selected, rows[selected] @ weights[expert])
    return output


__all__ = [
    "batched_matrix_vector",
    "expert_matrix_multiply",
    "linear",
    "matrix_multiply",
    "matrix_multiply_transpose_left",
    "matrix_multiply_transpose_right",
]
