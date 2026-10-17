from __future__ import annotations

from payload.tools.strategy_io_schema import OBS_WIDTH

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .policy_math import norm

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
    """57628982^:cast_header.Wally on torch.  `scale_shard` (lo, hi) keeps this rank's columns of the routed experts'
    hidden width (their up and gate projections' columns, the down projection's rows): the scale product's share
    where the world divides it (scale_rpc.RemoteScale); the whole width otherwise.  Every parameter is drawn whole from
    torch's generator in one order, then sliced, so every rank's parameters are the same draws."""

    def __init__(self, w: Widths, scale_shard=None):
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
        lo, hi = scale_shard or (0, w.scale_h)
        self.scale_shard = (lo, hi)
        self.sharded = frozenset() if scale_shard is None else frozenset(("scale_w1", "scale_w2", "scale_w3"))
        self.scale_w1 = nn.Parameter((torch.randn((w.scale_experts, w.d_scale, w.scale_h)) / (w.d_scale ** 0.5))[:, :, lo:hi].contiguous())
        self.scale_w3 = nn.Parameter((torch.randn((w.scale_experts, w.d_scale, w.scale_h)) / (w.d_scale ** 0.5))[:, :, lo:hi].contiguous())
        self.scale_w2 = nn.Parameter((torch.randn((w.scale_experts, w.scale_h, w.d_scale)) / (w.scale_h ** 0.5))[:, lo:hi].contiguous())
        self.scale_out = nn.Linear(w.d_scale, w.d_ir, bias=False)


def encode_rows(wally: Wally, rows):
    normalized = norm(rows)
    return rows + linear(wally.input_out, silu(linear(wally.input_gate, normalized)) * linear(wally.input_value, normalized))

def ir_query(wally: Wally, query):
    return linear(wally.ir_query, query)

def gia_uma_dov(wally: Wally, ir):
    normed = norm(ir)
    gated = silu(linear(wally.gia, normed)) * linear(wally.uma, normed)
    return linear(wally.dov, gated)

def scale_route(rows, router, topk):
    scores = matrix_multiply_transpose_right(rows, router)
    experts = torch.topk(scores, topk, dim=-1).indices.detach()
    log_affinity = -torch.logaddexp(torch.zeros_like(scores), -scores)
    affinities = torch.exp(log_affinity - log_affinity.amax(dim=-1, keepdim=True))
    probabilities = affinities / affinities.sum(dim=-1, keepdim=True)
    selected = torch.gather(affinities, -1, experts)
    return experts, selected / selected.sum(dim=-1, keepdim=True), probabilities

def scale_balance(experts, probabilities, valid):
    present = valid.reshape(-1).to(probabilities.dtype)
    count = torch.clamp(present.sum(), min=1.0)
    load = torch.zeros((probabilities.shape[-1],), dtype=probabilities.dtype, device=probabilities.device).index_add(
        0, experts.reshape(-1), present[:, None].expand(experts.shape).reshape(-1))
    importance = (probabilities * present[:, None]).sum(dim=0) / count
    balance = probabilities.shape[-1] * (load.detach() * importance).sum() / (experts.shape[-1] * count)
    return load.detach(), balance

def silu(value):
    return value * torch.sigmoid(value)


def scale_moe(rows, weight1, weight2, weight3, experts, gates):
    topk = experts.shape[-1]
    selected = experts.reshape(-1)
    tokens = torch.arange(experts.numel(), device=rows.device) // topk
    repeated = rows[tokens]
    hidden = silu(expert_matrix_multiply(repeated, weight1, selected)) * expert_matrix_multiply(repeated, weight3, selected)
    values = expert_matrix_multiply(hidden, weight2, selected).reshape(rows.shape[0], topk, rows.shape[1])
    return (values * gates[..., None]).sum(dim=1)


def scale_operator(ir, weight_in, router, weight1, weight2, weight3, weight_out, valid, *, topk, world=None):
    """57628982^'s scale operator.  With a `world` that divides it, weight1..3 are this rank's hidden columns and the
    routed experts' output is the shares' sum (products: copy in, reduce out); the routing, norms and projections are
    every rank's alike."""
    shape = ir.shape
    valid = valid.detach()
    flat = ir.reshape(-1, weight_in.shape[1])
    rows = torch.where(valid.reshape(-1, 1) != 0, norm(matrix_multiply_transpose_right(flat, weight_in)), 0.0)
    experts, gates, probabilities = scale_route(rows, router, topk)
    if world is not None and world.size > 1:
        moe = world.reduce(scale_moe(world.copy(rows), weight1, weight2, weight3, experts, world.copy(gates)))
    else:
        moe = scale_moe(rows, weight1, weight2, weight3, experts, gates)
    residual = norm(rows + moe)
    delta = matrix_multiply_transpose_right(residual, weight_out).reshape(shape)
    stats = torch.stack((residual.min(), residual.max(), torch.isfinite(residual).sum().to(ir.dtype))).detach()
    load, balance = scale_balance(experts, probabilities, valid)
    return delta, stats, load, balance

def scale_fuse(wally: Wally, ir, execute_remote=True,
               residual_fusion_scale=None, valid=None):
    strength = float(getattr(wally, "residual_fusion_scale", 1.0)
                     if residual_fusion_scale is None else residual_fusion_scale)
    executor = getattr(wally, "scale_executor", None) if execute_remote or wally.sharded else None
    valid = torch.ones(ir.shape[:-1], dtype=torch.bool, device=ir.device) if valid is None else valid
    tensors = (ir, wally.scale_in.weight, wally.scale_router.weight, wally.scale_w1,
               wally.scale_w2, wally.scale_w3, wally.scale_out.weight, valid.to(ir.dtype))
    if executor is not None:
        delta, stats, load, balance = executor(*tensors, topk=wally.w.scale_topk)
    else:
        delta, stats, load, balance = scale_operator(*tensors, topk=wally.w.scale_topk)
    return delta * strength, stats, load, balance
