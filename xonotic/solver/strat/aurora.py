"""Aurora: Leverage-Aware Spectral Optimizer (Tilde Research, 2026) & Stiefel ∩ Row-Oblique Init.

For tall weight matrices (m > n, e.g. wide MLP up/gate projections), plain Muon polar(G) = U V^T
inherits non-uniform row leverage scores ||P_{i,:}||_2^2, starving a subset of neurons into a
self-reinforcing dead-neuron feedback loop. Aurora iteratively refines a diagonal row preconditioner
D so polar(D G) lies near the intersection of the Stiefel and Row-Oblique manifolds (uniform row
leverage across every neuron), falling back cleanly to Muon polar(G) for square/wide matrices (m <= n).

Paired with EDM2 unit-norm weight projection (project_unit_norm_) and Aurora-tuned tight-frame
initialization (aurora_init_weight_) so weights begin and remain on the unit hypersphere with
zero AdamW weight decay and zero initial neuron favoritism.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def polar_newton_schulz(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = coeffs
    X = G.to(torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def aurora_polar_update(
    G: torch.Tensor,
    *,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    ns_steps: int = 5,
    eps: float = 1e-7,
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    d_init_clamp: tuple[float, float] = (0.1, 10.0),
    d_iter_clamp: tuple[float, float] = (0.05, 20.0),
) -> tuple[torch.Tensor, dict[str, float]]:
    m, n = G.shape
    if m <= n or pp_iterations <= 0:
        P = polar_newton_schulz(G, steps=ns_steps, eps=eps, coeffs=ns_coeffs)
        row_sq = P.to(torch.float32).square().sum(dim=-1)
        cv = float(row_sq.std() / row_sq.mean().clamp_min(1e-8)) if m > 1 else 0.0
        return P, {"row_leverage_cv": cv, "tall": 0.0}

    target_lev = float(n) / float(m)
    row_norm = G.to(torch.float32).norm(dim=-1).clamp_min(eps)
    D = (row_norm.mean() / row_norm).pow(pp_beta).clamp(d_init_clamp[0], d_init_clamp[1])
    P = polar_newton_schulz(D[:, None] * G.to(torch.float32), steps=ns_steps, eps=eps, coeffs=ns_coeffs)
    for _ in range(pp_iterations):
        row_sq = P.square().sum(dim=-1).clamp_min(eps)
        ratio = (target_lev / row_sq).pow(pp_beta)
        D = (D * ratio).clamp(d_iter_clamp[0], d_iter_clamp[1])
        D = D / D.mean().clamp_min(eps)
        P = polar_newton_schulz(D[:, None] * G.to(torch.float32), steps=ns_steps, eps=eps, coeffs=ns_coeffs)

    row_sq = P.square().sum(dim=-1)
    cv = float(row_sq.std() / row_sq.mean().clamp_min(1e-8))
    return P.to(G.dtype), {"row_leverage_cv": cv, "tall": 1.0}


@torch.no_grad()
def aurora_init_weight_(
    weight: torch.Tensor,
    pp_iterations: int = 3,
    pp_beta: float = 0.5,
    ns_steps: int = 7,
    unit_norm_eps: float = 1e-6,
) -> torch.Tensor:
    if weight.ndim != 2:
        return weight
    g = torch.randn_like(weight, dtype=torch.float32)
    P, _ = aurora_polar_update(g, pp_iterations=pp_iterations, pp_beta=pp_beta, ns_steps=ns_steps)
    P = P / P.norm(dim=-1, keepdim=True).clamp_min(unit_norm_eps)
    weight.copy_(P.to(weight.dtype))
    return weight


class Aurora(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.015,
        scalar_lr: float = 3e-3,
        mu: float = 0.90,
        nesterov: bool = True,
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
        ns_steps: int = 5,
        eps: float = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        d_init_clamp: tuple[float, float] = (0.1, 10.0),
        d_iter_clamp: tuple[float, float] = (0.05, 20.0),
        scalar_beta2: float = 0.99,
        scalar_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=float(lr),
            scalar_lr=float(scalar_lr),
            mu=float(mu),
            nesterov=bool(nesterov),
            pp_iterations=int(pp_iterations),
            pp_beta=float(pp_beta),
            ns_steps=int(ns_steps),
            eps=float(eps),
            ns_coeffs=tuple(float(v) for v in ns_coeffs),
            d_init_clamp=tuple(float(v) for v in d_init_clamp),
            d_iter_clamp=tuple(float(v) for v in d_iter_clamp),
            scalar_beta2=float(scalar_beta2),
            scalar_eps=float(scalar_eps),
        )
        super().__init__(params, defaults)
        self.last_metrics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None, group_lr_scales: dict[int, float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        tall_cvs = []
        all_cvs = []
        for g_idx, group in enumerate(self.param_groups):
            scale = float(group_lr_scales.get(g_idx, 1.0)) if group_lr_scales is not None else 1.0
            if scale <= 0.0:
                continue
            lr = group["lr"] * scale
            scalar_lr = group["scalar_lr"] * scale
            mu = group["mu"]
            nesterov = group["nesterov"]
            pp_iters = group["pp_iterations"]
            pp_beta = group["pp_beta"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            ns_coeffs = group["ns_coeffs"]
            d_init_clamp = group["d_init_clamp"]
            d_iter_clamp = group["d_iter_clamp"]
            scalar_beta2 = group["scalar_beta2"]
            scalar_eps = group["scalar_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]

                if p.ndim == 2 and min(p.shape) >= 2:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g)
                    buf.mul_(mu).add_(g)
                    g_eff = g.add(buf, alpha=mu) if nesterov else buf

                    P, info = aurora_polar_update(
                        g_eff,
                        pp_iterations=pp_iters,
                        pp_beta=pp_beta,
                        ns_steps=ns_steps,
                        eps=eps,
                        ns_coeffs=ns_coeffs,
                        d_init_clamp=d_init_clamp,
                        d_iter_clamp=d_iter_clamp,
                    )
                    all_cvs.append(info["row_leverage_cv"])
                    if info["tall"] > 0.5:
                        tall_cvs.append(info["row_leverage_cv"])

                    m, n = p.shape
                    row_rms_scale = math.sqrt(float(m) / float(min(m, n)))
                    p.add_(P, alpha=-lr * row_rms_scale)
                else:
                    buf = state.get("momentum_buffer")
                    v = state.get("sq_avg")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g)
                        v = state["sq_avg"] = torch.zeros_like(g)
                        state["step"] = 0
                    state["step"] += 1
                    buf.mul_(mu).add_(g, alpha=1.0 - mu)
                    v.mul_(scalar_beta2).addcmul_(g, g, value=1.0 - scalar_beta2)
                    b_hat = buf / (1.0 - mu ** state["step"])
                    v_hat = v / (1.0 - scalar_beta2 ** state["step"])
                    p.addcdiv_(b_hat, v_hat.sqrt().add_(scalar_eps), value=-scalar_lr)

        self.last_metrics = {
            "tall_row_leverage_cv": round(sum(tall_cvs) / max(1, len(tall_cvs)), 4) if tall_cvs else 0.0,
            "all_row_leverage_cv": round(sum(all_cvs) / max(1, len(all_cvs)), 4) if all_cvs else 0.0,
        }
        return loss
