from __future__ import annotations

from . import tensor as mx

from . import paged_matrix

def _shape(lhs, rhs, transpose_lhs, transpose_rhs):
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"matrix operands must be rank 2; got {lhs.shape} and {rhs.shape}")
    rows = int(lhs.shape[1] if transpose_lhs else lhs.shape[0])
    lhs_inner = int(lhs.shape[0] if transpose_lhs else lhs.shape[1])
    rhs_inner = int(rhs.shape[1] if transpose_rhs else rhs.shape[0])
    columns = int(rhs.shape[0] if transpose_rhs else rhs.shape[1])
    if lhs_inner != rhs_inner:
        raise ValueError(f"matrix inner dimensions disagree: {lhs_inner} and {rhs_inner}")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"matrix dtypes disagree: {lhs.dtype} and {rhs.dtype}")
    if lhs.dtype != mx.float32:
        raise ValueError(f"matrix dtype must be float32; got {lhs.dtype}")
    return rows, lhs_inner, columns

def _dispatch(lhs, rhs, transpose_lhs, transpose_rhs):
    if isinstance(lhs, mx.Tensor):
        return mx.matmul(lhs, rhs, transpose_lhs, transpose_rhs)
    rows, inner, columns = _shape(lhs, rhs, transpose_lhs, transpose_rhs)
    if rows == 0 or columns == 0 or inner == 0:
        return mx.zeros((rows, columns), dtype=lhs.dtype)
    right = rhs.T if transpose_rhs else rhs
    return paged_matrix.cross(lhs, right) if transpose_lhs else paged_matrix.multiply(lhs, right)

@mx.custom_function
def matrix_multiply(lhs, rhs):
    return _dispatch(lhs, rhs, False, False)

@mx.custom_function
def matrix_multiply_transpose_left(lhs, rhs):
    return _dispatch(lhs, rhs, True, False)

@mx.custom_function
def matrix_multiply_transpose_right(lhs, rhs):
    return _dispatch(lhs, rhs, False, True)

@matrix_multiply.vjp
def _matrix_multiply_vjp(primals, cotangent, output):
    lhs, rhs = primals
    return (
        matrix_multiply_transpose_right(cotangent, rhs),
        matrix_multiply_transpose_left(lhs, cotangent),
    )

@matrix_multiply_transpose_left.vjp
def _matrix_multiply_transpose_left_vjp(primals, cotangent, output):
    lhs, rhs = primals
    return (
        matrix_multiply_transpose_right(rhs, cotangent),
        matrix_multiply(lhs, cotangent),
    )

@matrix_multiply_transpose_right.vjp
def _matrix_multiply_transpose_right_vjp(primals, cotangent, output):
    lhs, rhs = primals
    return (
        matrix_multiply(cotangent, rhs),
        matrix_multiply_transpose_left(cotangent, lhs),
    )

def linear(layer, rows):
    shape = rows.shape
    flat = rows.reshape(-1, shape[-1])
    product = matrix_multiply_transpose_right(flat, layer.weight)
    if "bias" in layer and layer.bias is not None:
        product = product + layer.bias
    return product.reshape(*shape[:-1], product.shape[-1])

def batched_matrix_vector(matrices, vectors):
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            f"matrix/vector inner dimensions disagree: {matrices.shape[-1]} and {vectors.shape[-1]}"
        )
    return mx.sum(matrices * vectors[..., None, :], axis=-1)

def expert_matrix_multiply(rows, weights, experts):
    if isinstance(rows, mx.Tensor):
        return mx.expert_matmul(rows, weights, experts)
    return paged_matrix.experts(rows, weights, experts)

__all__ = [
    "batched_matrix_vector",
    "expert_matrix_multiply",
    "linear",
    "matrix_multiply",
    "matrix_multiply_transpose_left",
    "matrix_multiply_transpose_right",
]
