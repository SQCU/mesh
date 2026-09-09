from __future__ import annotations

from payload.tools.strategy_io_schema import OBS_WIDTH

import hashlib
from dataclasses import dataclass

from . import tensor as mx
import mlx.nn as nn

from .policy_math import norm

from . import paged_matrix

from .matmul import (
    linear,
    expert_matrix_multiply,
    matrix_multiply,
    matrix_multiply_transpose_right,
)

def parameter_seed(seed, namespace):
    return int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{namespace}".encode()).digest()[:4], "little",
    )

@dataclass(frozen=True)
class Widths:
    d_x: int
    d_c: int
    d_obs: int = OBS_WIDTH

    d: int = 128
    d_ir: int = 128
    h: int = 341
    r: int = 128
    r_e: int = 128
    d_scale: int = 128
    scale_h: int = 341
    scale_experts: int = 8
    scale_topk: int = 2

class Wally(nn.Module):
    def __init__(self, w: Widths):
        super().__init__()
        self.w = w

        self.phil = nn.Linear(w.d_c, w.d, bias=False)

        self.quinn = nn.Linear(6 * w.d_x + 20, w.d, bias=False)
        self.participant = nn.Linear(w.d_obs + 2, w.d, bias=False)
        self.cart = nn.Linear(18, w.d, bias=False)
        self.team = nn.Linear(7, w.d, bias=False)
        from .neighborhood import LocalNeighborhood
        self.neighborhood = LocalNeighborhood(w.d)
        self.input_gate = nn.Linear(w.d, w.h, bias=False)
        self.input_value = nn.Linear(w.d, w.h, bias=False)
        self.input_out = nn.Linear(w.h, w.d, bias=False)

        self.ir_query = nn.Linear(w.d, w.d_ir, bias=False)

        self.team_metric = nn.Linear(w.d, w.r, bias=False)

        self.rival_metric = nn.Linear(w.d, w.r_e, bias=False)

        self.gia = nn.Linear(w.d_ir, w.h, bias=False)
        self.uma = nn.Linear(w.d_ir, w.h, bias=False)
        self.dov = nn.Linear(w.h, w.d_ir, bias=False)
        self.heads = nn.Linear(w.d_ir, 2 * w.d_x + 2, bias=False)

        self.scale_in = nn.Linear(w.d_ir, w.d_scale, bias=False)
        self.scale_router = nn.Linear(w.d_scale, w.scale_experts, bias=False)
        self.scale_w1 = mx.random.normal((w.scale_experts, w.d_scale, w.scale_h)) / (w.d_scale ** 0.5)
        self.scale_w3 = mx.random.normal((w.scale_experts, w.d_scale, w.scale_h)) / (w.d_scale ** 0.5)
        self.scale_w2 = mx.random.normal((w.scale_experts, w.scale_h, w.d_scale)) / (w.scale_h ** 0.5)
        self.scale_out = nn.Linear(w.d_scale, w.d_ir, bias=False)


def encode_rows(wally: Wally, rows: mx.array) -> mx.array:
    normalized = norm(rows)
    return rows + linear(wally.input_out, silu(linear(wally.input_gate, normalized)) * linear(wally.input_value, normalized))

def ir_query(wally: Wally, query: mx.array) -> mx.array:
    return linear(wally.ir_query, query)

def gia_uma_dov(wally: Wally, ir: mx.array) -> mx.array:
    normed = norm(ir)
    gated = silu(linear(wally.gia, normed)) * linear(wally.uma, normed)
    return linear(wally.dov, gated)

def scale_route(rows, router, topk):
    scores = matrix_multiply_transpose_right(rows, router)
    experts = mx.stop_gradient(mx.argpartition(-scores, topk - 1, axis=-1)[:, :topk])
    log_affinity = -mx.logaddexp(mx.zeros_like(scores), -scores)
    affinities = mx.exp(log_affinity - mx.max(log_affinity, axis=-1, keepdims=True))
    probabilities = affinities / mx.sum(affinities, axis=-1, keepdims=True)
    selected = mx.take_along_axis(affinities, experts, axis=-1)
    return experts, selected / mx.sum(selected, axis=-1, keepdims=True), probabilities

def scale_balance(experts, probabilities, valid):
    present = valid.reshape(-1).astype(probabilities.dtype)
    count = mx.maximum(mx.sum(present), 1.0)
    load = mx.zeros((probabilities.shape[-1],), dtype=probabilities.dtype).at[
        experts.reshape(-1)
    ].add(mx.broadcast_to(present[:, None], experts.shape).reshape(-1))
    importance = mx.sum(probabilities * present[:, None], axis=0) / count
    balance = probabilities.shape[-1] * mx.sum(mx.stop_gradient(load) * importance) / (experts.shape[-1] * count)
    return mx.stop_gradient(load), balance

def silu(value):
    return value * mx.sigmoid(value)


def scale_moe(rows, weight1, weight2, weight3, experts, gates):
    topk = experts.shape[-1]
    selected = experts.reshape(-1)
    tokens = mx.arange(experts.size) // topk
    repeated = rows[tokens]
    hidden = silu(expert_matrix_multiply(repeated, weight1, selected)) * expert_matrix_multiply(repeated, weight3, selected)
    values = expert_matrix_multiply(hidden, weight2, selected).reshape(rows.shape[0], topk, rows.shape[1])
    return mx.sum(values * gates[..., None], axis=1)


def scale_operator(ir, weight_in, router, weight1, weight2, weight3, weight_out, valid, *, topk):
    shape = ir.shape
    valid = mx.stop_gradient(valid)
    flat = ir.reshape(-1, weight_in.shape[1])
    rows = mx.where(valid.reshape(-1, 1), norm(matrix_multiply_transpose_right(flat, weight_in)), 0)
    experts, gates, probabilities = scale_route(rows, router, topk)
    residual = norm(rows + scale_moe(rows, weight1, weight2, weight3, experts, gates))
    delta = matrix_multiply_transpose_right(residual, weight_out).reshape(shape)
    stats = mx.stop_gradient(mx.stack((mx.min(residual), mx.max(residual), mx.sum(mx.isfinite(residual)).astype(ir.dtype))))
    load, balance = scale_balance(experts, probabilities, valid)
    return delta, stats, load, balance

def scale_fuse(wally: Wally, ir: mx.array, execute_remote=True,
               residual_fusion_scale=None, valid=None) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    strength = float(getattr(wally, "residual_fusion_scale", 1.0)
                     if residual_fusion_scale is None else residual_fusion_scale)
    executor = getattr(wally, "scale_executor", None) if execute_remote else None
    valid = mx.ones(ir.shape[:-1], dtype=mx.bool_) if valid is None else valid
    tensors = (ir, wally.scale_in.weight, wally.scale_router.weight, wally.scale_w1,
               wally.scale_w2, wally.scale_w3, wally.scale_out.weight, valid.astype(ir.dtype))
    if executor is not None:
        delta, stats, load, balance = executor(*tensors, topk=wally.w.scale_topk)
    else:
        delta, stats, load, balance = scale_operator(*tensors, topk=wally.w.scale_topk)
    return delta * strength, stats, load, balance
