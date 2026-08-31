from __future__ import annotations

import mlx.core as mx


_DEFAULT_JITTER = 1e-6


def _as_f32(value):
    result = mx.array(value)
    return result if result.dtype == mx.float32 else result.astype(mx.float32)


def feature_gram(features, normalize=True, eps=1e-12):
    rows = _as_f32(features)
    if rows.ndim != 2:
        raise ValueError(f"features must be 2-D; got {rows.shape}")
    if normalize:
        rows = rows / mx.maximum(mx.sqrt(mx.sum(rows * rows, axis=1, keepdims=True)), eps)
    gram = rows @ rows.T
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


def _prepare(kernel, jitter, detached=False):
    kernel = _as_f32(kernel)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"L must be square; got {kernel.shape}")
    kernel = 0.5 * (kernel + kernel.T)
    eye = mx.eye(kernel.shape[0], dtype=mx.float32)
    if jitter > 0:
        scale = mx.maximum(mx.mean(mx.diagonal(kernel)), mx.array(1.0, dtype=mx.float32))
        kernel = kernel + jitter * (mx.stop_gradient(scale) if detached else scale) * eye
    return kernel, eye


def marginal_inclusion(kernel, method="eigh", jitter=_DEFAULT_JITTER, return_spectrum=False):
    kernel, eye = _prepare(kernel, jitter)
    if method == "eigh":
        values, vectors = mx.linalg.eigh(kernel, stream=mx.cpu)
        values = mx.maximum(values, 0)
        result = mx.clip(mx.sum(vectors * vectors * (values / (1 + values))[None, :], axis=1), 0, 1)
        return (result, values) if return_spectrum else result
    if method == "inverse":
        if return_spectrum:
            raise ValueError("return_spectrum requires method='eigh'")
        return mx.clip(1 - mx.diagonal(mx.linalg.inv(eye + kernel, stream=mx.cpu)), 0, 1)
    raise ValueError(f"unknown method {method!r}")


@mx.custom_function
def _inv_diag_marginal(kernel):
    inverse = mx.linalg.inv(mx.eye(kernel.shape[0], dtype=kernel.dtype) + kernel, stream=mx.cpu)
    return mx.clip(1 - mx.diagonal(inverse), 0, 1)


@_inv_diag_marginal.vjp
def _inv_diag_marginal_vjp(primals, cotangent, output):
    kernel = primals[0] if isinstance(primals, (tuple, list)) else primals
    inverse = mx.linalg.inv(mx.eye(kernel.shape[0], dtype=kernel.dtype) + kernel, stream=mx.cpu)
    return inverse @ (cotangent[:, None] * inverse)


def marginal_inclusion_diff(kernel, jitter=_DEFAULT_JITTER):
    return _inv_diag_marginal(_prepare(kernel, jitter, detached=True)[0])


def marginal_kernel(kernel, jitter=_DEFAULT_JITTER):
    kernel, eye = _prepare(kernel, jitter)
    result = mx.linalg.solve(eye + kernel, kernel, stream=mx.cpu)
    return 0.5 * (result + result.T)


def dpp_marginals(quality, features, normalize=True, method="eigh", jitter=_DEFAULT_JITTER, return_spectrum=False):
    kernel = build_L(quality, features, normalize)
    if method == "inverse_diff":
        if return_spectrum:
            raise ValueError("return_spectrum requires method='eigh'")
        return marginal_inclusion_diff(kernel, jitter)
    return marginal_inclusion(kernel, method, jitter, return_spectrum)
