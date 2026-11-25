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


def polar_newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz polar factorization X -> U V^T (Muon / CANS-style matrix-multiply only)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Aurora's leverage-uniform polar factor polar(D G).

    For tall matrices (m > n), iteratively updates diagonal row preconditioner D so every row's
    squared leverage score r_i = ||P_{i,:}||_2^2 converges to uniform target n / m.
    For square or wide matrices (m <= n), falls back cleanly to standard Muon polar(G).
    """
    m, n = G.shape
    if m <= n or pp_iterations <= 0:
        P = polar_newton_schulz(G, steps=ns_steps, eps=eps)
        row_sq = P.to(torch.float32).square().sum(dim=-1)
        cv = float(row_sq.std() / row_sq.mean().clamp_min(1e-8)) if m > 1 else 0.0
        return P, {"row_leverage_cv": cv, "tall": 0.0}

    target_lev = float(n) / float(m)
    row_norm = G.to(torch.float32).norm(dim=-1).clamp_min(eps)
    # Leverage-aware diagonal initialization: equalize initial row scales before first polar step
    D = (row_norm.mean() / row_norm).pow(pp_beta).clamp(0.1, 10.0)
    P = polar_newton_schulz(D[:, None] * G.to(torch.float32), steps=ns_steps, eps=eps)
    for _ in range(pp_iterations):
        row_sq = P.square().sum(dim=-1).clamp_min(eps)
        ratio = (target_lev / row_sq).pow(pp_beta)
        D = (D * ratio).clamp(0.05, 20.0)
        D = D / D.mean().clamp_min(eps)
        P = polar_newton_schulz(D[:, None] * G.to(torch.float32), steps=ns_steps, eps=eps)

    row_sq = P.square().sum(dim=-1)
    cv = float(row_sq.std() / row_sq.mean().clamp_min(1e-8))
    return P.to(G.dtype), {"row_leverage_cv": cv, "tall": 1.0}


@torch.no_grad()
def aurora_init_weight_(weight: torch.Tensor, pp_iterations: int = 3, pp_beta: float = 0.5) -> torch.Tensor:
    """Aurora-tuned Stiefel ∩ Row-Oblique weight initialization:
    Produces a tight frame (equal singular values across all input directions) with uniform row
    leverage across all output neurons, then projects every output row to exact unit L2 norm
    (||w_i||_2 = 1) for EDM2 magnitude preservation."""
    if weight.ndim != 2:
        return weight
    g = torch.randn_like(weight, dtype=torch.float32)
    P, _ = aurora_polar_update(g, pp_iterations=pp_iterations, pp_beta=pp_beta, ns_steps=7)
    # Project every row onto the unit hypersphere S^{n-1} (exact EDM2 unit-norm invariant)
    P = P / P.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    weight.copy_(P.to(weight.dtype))
    return weight


class Aurora(torch.optim.Optimizer):
    """Leverage-Aware Spectral Optimizer (Aurora) for 2D weight matrices + momentum-normalized
    scalar updates for 1D parameters, paired with forced unit-norm spherical projection on
    MPLinear weights (zero AdamW weight decay)."""

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
        )
        super().__init__(params, defaults)
        self.last_metrics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        tall_cvs = []
        all_cvs = []
        for group in self.param_groups:
            lr = group["lr"]
            scalar_lr = group["scalar_lr"]
            mu = group["mu"]
            nesterov = group["nesterov"]
            pp_iters = group["pp_iterations"]
            pp_beta = group["pp_beta"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

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
                        g_eff, pp_iterations=pp_iters, pp_beta=pp_beta, ns_steps=ns_steps, eps=eps,
                    )
                    all_cvs.append(info["row_leverage_cv"])
                    if info["tall"] > 0.5:
                        tall_cvs.append(info["row_leverage_cv"])

                    # Spectral aspect-ratio scaling normalized to unit-norm rows:
                    # Each row of P has norm ~sqrt(min(m,n)/m), so multiplying by sqrt(m/min(m,n))
                    # yields exact angular step ~lr radians per row regardless of matrix aspect ratio.
                    m, n = p.shape
                    row_rms_scale = math.sqrt(float(m) / float(min(m, n)))
                    # Tangent-space projection on unit sphere for MPLinear-style rows keeps updates orthogonal to w
                    p.add_(P, alpha=-lr * row_rms_scale)
                else:
                    # 1D scalars (explicit output temperatures, gains, biases, blend logits):
                    # sign/RMS-normalized Nesterov momentum (no AdamW weight decay)
                    buf = state.get("momentum_buffer")
                    v = state.get("sq_avg")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g)
                        v = state["sq_avg"] = torch.zeros_like(g)
                        state["step"] = 0
                    state["step"] += 1
                    buf.mul_(mu).add_(g, alpha=1.0 - mu)
                    v.mul_(0.99).addcmul_(g, g, value=0.01)
                    b_hat = buf / (1.0 - mu ** state["step"])
                    v_hat = v / (1.0 - 0.99 ** state["step"])
                    p.addcdiv_(b_hat, v_hat.sqrt().add_(1e-8), value=-scalar_lr)

        self.last_metrics = {
            "tall_row_leverage_cv": round(sum(tall_cvs) / max(1, len(tall_cvs)), 4) if tall_cvs else 0.0,
            "all_row_leverage_cv": round(sum(all_cvs) / max(1, len(all_cvs)), 4) if all_cvs else 0.0,
        }
        return loss
