from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .matmul import (
    expert_matrix_multiply,
    linear,
    matrix_multiply,
    matrix_multiply_transpose_left,
    matrix_multiply_transpose_right,
)

__all__ = [
    "Widths", "Wally", "parameter_seed",
    "phil", "quinn", "kay", "val",
    "ir_query", "ir_value", "dina_state", "dina_action", "dina_readout", "dee",
    "team_gram_matrix", "rival_gram_matrix", "participant_gram_matrix",
    "scale_project", "scale_route", "scale_moe", "scale_back", "scale_probe", "scale_fuse",
    "gia_uma_dov", "actuator",
    "winnie", "lou", "vera_winnie", "vera_lou",
    "dina_drift", "dina_matrix",
    "tau", "elle",
]

def parameter_seed(seed, namespace):
    return int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{namespace}".encode()).digest()[:4], "little",
    )

@dataclass(frozen=True)
class Widths:
    d_x: int
    d_z: int
    d_c: int
    d_sem: int = 8

    d_beta: int = 128
    d: int = 128
    d_v: int = 128
    d_ir: int = 128
    h: int = 341
    r: int = 128
    r_e: int = 128
    d_y: int = 128
    d_u: int = 128
    d_scale: int = 128
    scale_h: int = 341
    scale_experts: int = 8
    scale_topk: int = 2

class Wally(nn.Module):
    def __init__(self, w: Widths):
        super().__init__()
        self.w = w

        self.phil = nn.Linear(w.d_c, w.d_beta, bias=False)

        self.quinn = nn.Linear(w.d_x + w.d_beta + w.d_sem, w.d, bias=False)

        self.kay = nn.Linear(w.d_z, w.d, bias=False)

        self.val = nn.Linear(w.d_z, w.d_v, bias=False)
        self.ir_query = nn.Linear(w.d, w.d_ir, bias=False)
        self.ir_value = nn.Linear(w.d_v, w.d_ir, bias=False)
        self.dina_state = nn.Linear(w.d, w.d_y, bias=False)
        self.dina_action = nn.Linear(w.d_ir, w.d_u, bias=False)
        self.dina_readout = nn.Linear(w.d_y, w.d, bias=False)

        self.team_metric = nn.Linear(w.d, w.r, bias=False)

        self.rival_metric = nn.Linear(w.d, w.r_e, bias=False)

        self.gia = nn.Linear(w.d_ir, w.h, bias=False)
        self.uma = nn.Linear(w.d_ir, w.h, bias=False)
        self.dov = nn.Linear(w.h, 1, bias=False)
        self.actuator = nn.Linear(w.d_ir, 6, bias=False)

        self.winnie = nn.Linear(w.d_ir, 1, bias=False)

        self.lou = nn.Linear(w.d_ir, 1, bias=False)

        self.vera_winnie = nn.Linear(w.d, 1, bias=False)
        self.vera_lou = nn.Linear(w.d, 1, bias=False)

        self.dina_drift_first = nn.Linear(w.d_y, w.d_y, bias=False)
        self.dina_matrix_first = nn.Linear(w.d_y, w.d_y * w.d_u, bias=False)
        self.dina_drift_second = nn.Linear(w.d_y, w.d_y, bias=False)
        self.dina_matrix_second = nn.Linear(w.d_y, w.d_y * w.d_u, bias=False)
        self.scale_in = nn.Linear(w.d_ir, w.d_scale, bias=False)
        self.scale_router = nn.Linear(w.d_scale, w.scale_experts, bias=False)
        self.scale_w1 = mx.random.normal((w.scale_experts, w.d_scale, w.scale_h)) / (w.d_scale ** 0.5)
        self.scale_w2 = mx.random.normal((w.scale_experts, w.scale_h, w.d_scale)) / (w.scale_h ** 0.5)
        self.scale_out = nn.Linear(w.d_scale, w.d_ir, bias=False)
        self.scale_probe = mx.random.normal((w.d_scale,)) / (w.d_scale ** 0.5)

        self.tau_raw = mx.zeros(())

def phil(wally: Wally, cell_slots: mx.array) -> mx.array:
    return linear(wally.phil, cell_slots)

def norm(rows: mx.array) -> mx.array:
    return rows * mx.rsqrt(mx.mean(mx.square(rows), axis=-1, keepdims=True) + 1e-6)

def quinn(wally: Wally, xan: mx.array, bea: mx.array, semantics: mx.array) -> mx.array:
    return linear(wally.quinn, mx.concatenate([xan, bea, semantics], axis=-1))

def kay(wally: Wally, zed: mx.array) -> mx.array:
    return linear(wally.kay, zed)

def val(wally: Wally, zed: mx.array) -> mx.array:
    return linear(wally.val, zed)

def ir_query(wally: Wally, query: mx.array) -> mx.array:
    return linear(wally.ir_query, query)

def ir_value(wally: Wally, value: mx.array) -> mx.array:
    return linear(wally.ir_value, value)

def dina_state(wally: Wally, query: mx.array) -> mx.array:
    return linear(wally.dina_state, query)

def dina_action(wally: Wally, ir: mx.array) -> mx.array:
    return linear(wally.dina_action, ir)

def dina_readout(wally: Wally, state: mx.array) -> mx.array:
    return linear(wally.dina_readout, state)

def dee(quality: mx.array, keys: mx.array) -> mx.array:
    from .dpp import dpp_marginals

    return dpp_marginals(quality, keys)

def team_gram_matrix(wally: Wally, rows: mx.array) -> mx.array:
    projected = linear(wally.team_metric, rows)
    return matrix_multiply_transpose_right(projected, projected) / (wally.w.r ** 0.5)

def rival_gram_matrix(wally: Wally, rows: mx.array) -> mx.array:
    projected = linear(wally.rival_metric, rows)
    return matrix_multiply_transpose_right(projected, projected) / (wally.w.r_e ** 0.5)

def participant_gram_matrix(wally: Wally, rows: mx.array, team_ids: mx.array) -> mx.array:
    same_team = team_ids[:, None] == team_ids[None, :]
    return rival_gram_matrix(wally, rows) + same_team * team_gram_matrix(wally, rows)

def gia_uma_dov(wally: Wally, ir: mx.array) -> mx.array:
    normed = ir * mx.rsqrt(mx.mean(ir * ir, axis=-1, keepdims=True) + 1e-6)
    gated = nn.silu(linear(wally.gia, normed)) * linear(wally.uma, normed)
    return linear(wally.dov, gated)[..., 0]

def actuator(wally: Wally, ir: mx.array) -> mx.array:
    return linear(wally.actuator, ir)

def scale_project(wally: Wally, rows: mx.array) -> mx.array:
    return linear(wally.scale_in, rows)

def scale_route(wally: Wally, rows: mx.array) -> tuple[mx.array, mx.array]:
    scores = linear(wally.scale_router, rows)
    topk = wally.w.scale_topk
    experts = mx.stop_gradient(mx.argpartition(-scores, topk - 1, axis=-1)[:, :topk])
    gates = mx.take_along_axis(mx.softmax(scores, axis=-1), experts, axis=-1)
    return experts, gates / mx.sum(gates, axis=-1, keepdims=True)

def scale_moe(wally: Wally, rows: mx.array, experts: mx.array, gates: mx.array) -> mx.array:
    topk = experts.shape[-1]
    flat_experts = experts.reshape(-1)
    order = mx.argsort(flat_experts)
    selected = mx.take(flat_experts, order)
    tokens = order // topk
    routed = mx.take(rows, tokens, axis=0)
    hidden = nn.silu(expert_matrix_multiply(routed, wally.scale_w1, selected))
    values = expert_matrix_multiply(hidden, wally.scale_w2, selected)
    weights = mx.take(gates.reshape(-1), order)
    return mx.zeros_like(rows).at[tokens].add(values * weights[:, None])

def scale_back(wally: Wally, rows: mx.array) -> mx.array:
    return linear(wally.scale_out, rows)

def scale_probe(wally: Wally) -> mx.array:
    return wally.scale_probe

def scale_fuse(wally: Wally, ir: mx.array, execute_remote=True,
               residual_fusion_scale=None) -> tuple[mx.array, mx.array, mx.array]:
    residual_fusion_scale = float(
        getattr(wally, "residual_fusion_scale", 1.0)
        if residual_fusion_scale is None else residual_fusion_scale
    )
    executor = getattr(wally, "scale_executor", None) if execute_remote else None
    if executor is not None:
        remote = executor(ir, residual_fusion_scale)
        if remote is not None:
            return remote[0] * residual_fusion_scale, remote[1], remote[2]
    shape = ir.shape
    flat = ir.reshape(-1, wally.w.d_ir)
    physical = int(flat.shape[0])
    rows = norm(scale_project(wally, flat))
    experts, gates = scale_route(wally, rows)
    residual = norm(rows + scale_moe(wally, rows, experts, gates))
    gram = matrix_multiply_transpose_left(residual, residual) / physical
    context = matrix_multiply(mx.tanh(gram), scale_probe(wally)[:, None])[:, 0]
    delta = scale_back(wally, norm(residual * context[None, :])).reshape(shape)
    delta = delta * residual_fusion_scale
    stats = mx.stack([
        mx.min(gram),
        mx.max(gram),
        mx.sum(mx.isfinite(gram)).astype(gram.dtype),
    ])
    load = mx.zeros((wally.w.scale_experts,), dtype=ir.dtype).at[
        experts.reshape(-1)
    ].add(mx.ones(experts.shape, dtype=ir.dtype).reshape(-1))
    return delta, stats, load

def winnie(wally: Wally, ir: mx.array) -> mx.array:
    return linear(wally.winnie, ir)[..., 0]

def lou(wally: Wally, ir: mx.array) -> mx.array:
    return linear(wally.lou, ir)[..., 0]

def vera_winnie(wally: Wally, query: mx.array) -> mx.array:
    return linear(wally.vera_winnie, query)[..., 0]

def vera_lou(wally: Wally, query: mx.array) -> mx.array:
    return linear(wally.vera_lou, query)[..., 0]

def dina_drift(wally: Wally, y: mx.array) -> mx.array:
    return linear(wally.dina_drift_first, y), linear(wally.dina_drift_second, y)

def dina_matrix(wally: Wally, y: mx.array) -> mx.array:
    first = linear(wally.dina_matrix_first, y)
    second = linear(wally.dina_matrix_second, y)
    shape = (*first.shape[:-1], wally.w.d_y, wally.w.d_u)
    return first.reshape(*shape), second.reshape(*shape)

def tau(wally: Wally) -> mx.array:
    return mx.exp(mx.clip(wally.tau_raw, -3.0, 3.0))

def elle(logits: mx.array, measure: mx.array) -> mx.array:
    finite = mx.where(measure > 0, logits, 0)
    return mx.sum(finite * finite * measure) / mx.maximum(mx.sum(measure), 1)
