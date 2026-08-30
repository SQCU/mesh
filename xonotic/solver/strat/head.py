from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "MixingHead",
    "rms_norm",
    "swiglu",
    "integrate_weights",
    "sample_strategy",
    "strategy_log_prob",
    "strategy_probs",
]


def rms_norm(x: mx.array, weight: mx.array | None = None, eps: float = 1e-6) -> mx.array:
    y = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if weight is None else y * weight


def swiglu(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
    b_gate: mx.array | None = None,
    b_up: mx.array | None = None,
    b_down: mx.array | None = None,
) -> mx.array:
    g = x @ w_gate
    u = x @ w_up
    if b_gate is not None:
        g = g + b_gate
    if b_up is not None:
        u = u + b_up
    out = (nn.silu(g) * u) @ w_down
    return out if b_down is None else out + b_down


class MixingHead(nn.Module):
    def __init__(
        self,
        n_instruments: int | None = None,
        hidden: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_dim = 21
        self.hidden = 32 if hidden is None else int(hidden)
        self.eps = float(eps)
        self.norm_weight = mx.ones((self.in_dim,))
        scale_in = self.in_dim ** -0.5
        scale_h = self.hidden ** -0.5
        self.w_gate = mx.random.normal((self.in_dim, self.hidden)) * scale_in
        self.w_up = mx.random.normal((self.in_dim, self.hidden)) * scale_in
        self.w_down = mx.random.normal((self.hidden, 1)) * scale_h * 0.1

    def features(
        self,
        diag_k: mx.array,
        appetite: mx.array,
        relation: mx.array | None = None,
    ) -> mx.array:
        diag = mx.broadcast_to(diag_k[None, :], appetite.shape)
        if relation is None:
            relation = mx.zeros((*appetite.shape, 16), dtype=appetite.dtype)
        mean_appetite = mx.broadcast_to(
            mx.mean(appetite, axis=-1, keepdims=True), appetite.shape
        )
        max_appetite = mx.broadcast_to(
            mx.max(appetite, axis=-1, keepdims=True), appetite.shape
        )
        mean_diag = mx.broadcast_to(mx.mean(diag_k), appetite.shape)
        return mx.concatenate(
            [
                diag[..., None],
                appetite[..., None],
                mean_appetite[..., None],
                max_appetite[..., None],
                mean_diag[..., None],
                relation,
            ],
            axis=-1,
        )

    def __call__(
        self,
        diag_k: mx.array,
        appetite: mx.array,
        relation: mx.array | None = None,
    ) -> mx.array:
        x = rms_norm(self.features(diag_k, appetite, relation), self.norm_weight, self.eps)
        return swiglu(x, self.w_gate, self.w_up, self.w_down)[..., 0]


def integrate_weights(w: mx.array, dw_dt: mx.array, delta: float) -> mx.array:
    return w + dw_dt * delta


def strategy_probs(logits: mx.array, temperature: float = 1.0) -> mx.array:
    return mx.softmax(logits / temperature, axis=-1)


def strategy_log_prob(
    logits: mx.array,
    action: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    scaled = logits / temperature
    logp = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    return mx.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def sample_strategy(
    logits: mx.array,
    temperature: float = 1.0,
    key: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    action = mx.random.categorical(logits / temperature, axis=-1, key=key)
    return action, strategy_log_prob(logits, action, temperature)
