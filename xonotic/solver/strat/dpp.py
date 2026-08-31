"""Symmetric L-ensemble DPP: the per-instrument marginal-inclusion signal.

This module emits the *honest, repulsion-only diversity signal* that the mixing
head consumes. Per `design/dpp-mixing-and-overlay.md` section 2 ("The DPP
intermediate is a marginal inclusion vector, NOT the determinant." [FIRM]), the
object handed downstream is the per-instrument MARGINAL INCLUSION VECTOR

    diag(K),   K = L (I + L)^-1

where `L` is the symmetric PSD L-ensemble kernel (the quality/diversity Gram of
the review's Part A) and `K` is its marginal kernel. `K_ii` is the marginal
probability that instrument `i` appears in a sample from the DPP -- one
repulsion-shaped number per instrument answering "how much does this instrument
belong in a diverse selection." We deliberately do NOT reduce to `det(L_Y)`: the
determinant is a single scalar for a whole subset and "can't be two things,"
throwing away exactly the per-instrument structure the head needs (doc section 2).

The kernel here is strictly SYMMETRIC and PSD -- repulsion only. Per doc
section 1 ("The false fork and its dissolution" [FIRM]) the diversify-vs-pile-on
regime switch is NOT a kernel property (no NDPP, no attraction entries); it is
the job of the downstream learned RMSNorm -> SwiGLU mixing head's gate. This
module's sole responsibility is to emit a clean `diag(K)`; it does not decide
behavior.

The L-ensemble uses the standard quality/diversity factorization (review Part A):

    L = diag(q) . S . diag(q),   S_ij = <phi_i, phi_j>  (feature Gram, unit rows)

so `q_i >= 0` is instrument i's raw appetite/quality (the DPP quality terms the
doc calls `b` when concatenated for the head) and `phi_i` is its (normalized)
diversity feature / key. `S` is a similarity Gram in [-1, 1] on unit rows; `L` is
PSD by construction.

Where this sits in the gradient graph. `diag(K)` is a per-instrument feature into
the learned mixing head. The estimator calls `dpp_marginals(..., method="inverse_diff")`
(-> `marginal_inclusion_diff`), which returns K = I-(I+L)^-1 with an ANALYTIC custom
vjp (this mlx build ships no Inverse/Eigh vjp), and does NOT stop_gradient the result
— so the REINFORCE reward gradient flows back through `diag(K)` into the quality /
appetite that builds `L`, shaping the coupling. The `eigh` and plain `inverse` paths
are retained for diagnostics/spectrum (not differentiable here). The module is mlx so
the signal stays on-device.

Numerical stability
-------------------
`diag(K)` is computed two ways, both stable:

  * "eigh"    -- symmetrize L, add jitter, symmetric eigendecomposition, clamp
                 eigenvalues to >= 0, then  k_i = sum_j V_ij^2 * lam_j/(1+lam_j).
                 Also yields the spectrum for logging (effective rank / expected
                 sample size = sum_j lam_j/(1+lam_j) = sum_i diag(K)).
  * "inverse" -- the exact identity  K = I - (I+L)^-1 , so
                 diag(K)_i = 1 - [(I+L)^-1]_ii .  `I+L` has eigenvalues >= 1, so
                 it is always well-conditioned; this path needs no eigh.

Both return the same vector up to floating point. `eigh` is the default because
the doc/spec calls for eigh "as needed" and the spectrum is useful downstream;
`inverse` is offered for the well-conditioned fast path.

Public functions
----------------
  feature_gram(features, normalize=True)      -> S   (similarity Gram)
  build_L(quality, features, normalize=True)  -> L   (symmetric PSD L-ensemble)
  marginal_kernel(L, ...)                     -> K   (full marginal kernel)
  marginal_inclusion(L, ...)                  -> diag(K)  (the emitted signal)
  dpp_marginals(quality, features, ...)       -> diag(K) end-to-end, with spectrum

All accept mlx arrays or anything `mx.array` can ingest (numpy, lists) and return
mlx arrays so the result stays on-device and differentiable.
"""
from __future__ import annotations

import mlx.core as mx

__all__ = [
    "feature_gram",
    "build_L",
    "marginal_kernel",
    "marginal_inclusion",
    "dpp_marginals",
    "marginal_inclusion_diff",
]

# Default jitter added to L's diagonal before an eigendecomposition, in units
# scaled by the mean diagonal so it is invariant to the overall quality scale.
# Small enough not to move diag(K) meaningfully, large enough to keep eigh well
# posed on a rank-deficient (repeated-feature) Gram.
_DEFAULT_JITTER = 1e-6


def _as_f32(a) -> mx.array:
    x = mx.array(a)
    if x.dtype != mx.float32:
        x = x.astype(mx.float32)
    return x


def feature_gram(features, normalize: bool = True, eps: float = 1e-12) -> mx.array:
    """Similarity Gram `S_ij = <phi_i, phi_j>` over instrument keys.

    Implements the diversity half of the review's Part A quality/diversity
    factorization used by `build_L` (doc `dpp-mixing-and-overlay.md` section 2).

    Parameters
    ----------
    features : (N, F) array-like
        One diversity feature / key row `phi_i` per instrument.
    normalize : bool
        If True (default) rows are L2-normalized first, so `S` is a cosine
        similarity Gram with unit diagonal and off-diagonals in [-1, 1] -- the
        canonical DPP similarity matrix. If False, the raw inner-product Gram is
        returned (still symmetric PSD).
    eps : float
        Floor on row norm to avoid division by zero for a zero feature row.

    Returns
    -------
    S : (N, N) mlx.array, symmetric PSD.
    """
    phi = _as_f32(features)
    if phi.ndim != 2:
        raise ValueError(f"features must be 2-D (N, F); got shape {phi.shape}")
    if normalize:
        norm = mx.sqrt(mx.sum(phi * phi, axis=1, keepdims=True))
        phi = phi / mx.maximum(norm, eps)
    S = phi @ phi.T
    return 0.5 * (S + S.T)  # kill asymmetric floating-point dust


def build_L(quality, features, normalize: bool = True, eps: float = 1e-12) -> mx.array:
    """Symmetric PSD L-ensemble  `L = diag(q) . S . diag(q)`.

    The standard quality/diversity DPP factorization (review Part A, cited by
    `dpp-mixing-and-overlay.md` section 2): `q_i >= 0` is instrument i's raw
    appetite/quality (the head's `b` terms) and `S = feature_gram(features)` is
    the diversity similarity. Symmetric and repulsion-only by construction -- no
    nonsymmetric / attraction terms (doc section 1, fork dissolved [FIRM]).

    Parameters
    ----------
    quality : (N,) array-like
        Per-instrument appetite / quality `q_i`. Should be non-negative; not
        clamped here so gradients pass cleanly, but negative entries only flip a
        sign that `diag(q) S diag(q)` squares away on the diagonal.
    features : (N, F) array-like
        Per-instrument diversity keys `phi_i`; see `feature_gram`.
    normalize, eps :
        Forwarded to `feature_gram`.

    Returns
    -------
    L : (N, N) mlx.array, symmetric PSD L-ensemble kernel.
    """
    q = _as_f32(quality)
    if q.ndim != 1:
        raise ValueError(f"quality must be 1-D (N,); got shape {q.shape}")
    S = feature_gram(features, normalize=normalize, eps=eps)
    if S.shape[0] != q.shape[0]:
        raise ValueError(
            f"quality N={q.shape[0]} disagrees with features N={S.shape[0]}"
        )
    L = q[:, None] * S * q[None, :]
    return 0.5 * (L + L.T)


def _prep_L(L, jitter: float) -> tuple[mx.array, mx.array]:
    """Symmetrize L, add scale-aware diagonal jitter, return (L_prepped, N-eye)."""
    Lm = _as_f32(L)
    if Lm.ndim != 2 or Lm.shape[0] != Lm.shape[1]:
        raise ValueError(f"L must be a square 2-D matrix; got shape {Lm.shape}")
    n = Lm.shape[0]
    Lm = 0.5 * (Lm + Lm.T)
    eye = mx.eye(n, dtype=mx.float32)
    if jitter and jitter > 0.0:
        # scale jitter by the mean diagonal so it is invariant to quality scale
        scale = mx.maximum(mx.mean(mx.diagonal(Lm)), mx.array(1.0, dtype=mx.float32))
        Lm = Lm + (jitter * scale) * eye
    return Lm, eye


def marginal_inclusion(
    L,
    method: str = "eigh",
    jitter: float = _DEFAULT_JITTER,
    return_spectrum: bool = False,
):
    """Per-instrument marginal-inclusion vector  `diag(K)`,  `K = L (I+L)^-1`.

    THE quantity this module exists to emit (doc `dpp-mixing-and-overlay.md`
    section 2 [FIRM]): the repulsion-shaped signal fed to the mixing head, one
    number per instrument, NOT the scalar determinant.

    Parameters
    ----------
    L : (N, N) array-like
        Symmetric PSD L-ensemble kernel (e.g. from `build_L`).
    method : {"eigh", "inverse"}
        "eigh"    -- symmetric eigendecomposition; clamp eigenvalues to >= 0;
                     diag(K)_i = sum_j V_ij^2 * lam_j / (1 + lam_j). Also exposes
                     the spectrum. Default.
        "inverse" -- exact identity K = I - (I+L)^-1, so
                     diag(K)_i = 1 - [(I+L)^-1]_ii. `I+L` has eigenvalues >= 1, so
                     always well-conditioned; no eigh required.
    jitter : float
        Diagonal jitter (scaled by mean diagonal) added before the solve/eigh for
        numerical stability on rank-deficient Grams. Set 0 to disable.
    return_spectrum : bool
        If True, also return the eigenvalues of L (ascending). Only the "eigh"
        path produces them; requesting them with method="inverse" raises.

    Returns
    -------
    diag_K : (N,) mlx.array in [0, 1], the marginal inclusion probabilities.
             `sum(diag_K)` is the expected DPP sample size (effective rank).
    eigenvalues : (N,) mlx.array, only if return_spectrum=True (method="eigh").
    """
    Lm, eye = _prep_L(L, jitter)

    if method == "eigh":
        # L symmetric PSD -> real eigendecomposition. mlx returns ascending evals.
        evals, evecs = mx.linalg.eigh(Lm, stream=mx.cpu)
        evals = mx.maximum(evals, 0.0)              # clamp jitter-induced negatives
        gains = evals / (1.0 + evals)               # lam / (1 + lam) per mode
        # diag(K)_i = sum_j V_ij^2 * gains_j
        diag_K = mx.sum((evecs * evecs) * gains[None, :], axis=1)
        diag_K = mx.clip(diag_K, 0.0, 1.0)
        if return_spectrum:
            return diag_K, evals
        return diag_K

    if method == "inverse":
        if return_spectrum:
            raise ValueError("return_spectrum requires method='eigh'")
        # K = I - (I+L)^-1  =>  diag(K) = 1 - diag((I+L)^-1)
        inv = mx.linalg.inv(eye + Lm, stream=mx.cpu)
        diag_K = 1.0 - mx.diagonal(inv)
        return mx.clip(diag_K, 0.0, 1.0)

    raise ValueError(f"method must be 'eigh' or 'inverse'; got {method!r}")


def _prep_sym(L, jitter: float):
    """Symmetrize L and add detached scale-aware jitter (differentiable matmuls only)."""
    Lm = _as_f32(L)
    Lm = 0.5 * (Lm + Lm.T)
    n = Lm.shape[0]
    eye = mx.eye(n, dtype=mx.float32)
    if jitter and jitter > 0.0:
        scale = mx.maximum(mx.mean(mx.diagonal(Lm)), mx.array(1.0, dtype=mx.float32))
        Lm = Lm + (jitter * mx.stop_gradient(scale)) * eye
    return Lm, eye


@mx.custom_function
def _inv_diag_marginal(Lm: mx.array) -> mx.array:
    """diag(K) = 1 - diag((I+Lm)^-1) with an analytic vjp (mlx has no Inverse vjp).

    Lm must already be symmetric (+jitter). Forward uses the exact inverse; the custom
    vjp supplies the gradient so the reward can shape the DPP coupling through `Lm`.
    """
    n = Lm.shape[0]
    eye = mx.eye(n, dtype=Lm.dtype)
    B = mx.linalg.inv(eye + Lm, stream=mx.cpu)
    return mx.clip(1.0 - mx.diagonal(B), 0.0, 1.0)


@_inv_diag_marginal.vjp
def _inv_diag_marginal_vjp(primals, cotangent, output):
    # diag_K_i = 1 - B_ii, B = (I+Lm)^-1. For Y=inv(X): Xbar = -Y^T Ybar Y^T.
    # Bbar = -diag(c) => Lmbar = B diag(c) B (B symmetric here).
    Lm = primals[0] if isinstance(primals, (tuple, list)) else primals
    n = Lm.shape[0]
    eye = mx.eye(n, dtype=Lm.dtype)
    B = mx.linalg.inv(eye + Lm, stream=mx.cpu)
    c = cotangent
    return B @ (c[:, None] * B)


def marginal_inclusion_diff(L, jitter: float = _DEFAULT_JITTER) -> mx.array:
    """Differentiable per-instrument `diag(K)` (K = I-(I+L)^-1) via the custom-vjp core.

    Same value as `marginal_inclusion(L, method="inverse")` but autodiff-differentiable
    w.r.t. `L` (hence w.r.t. the appetite/quality that builds `L`) in an mlx that ships no
    Inverse/Eigh vjp. This is the path the estimator uses so the DPP is not stop_gradient'd.
    """
    Lm, _ = _prep_sym(L, jitter)
    return _inv_diag_marginal(Lm)


def marginal_kernel(L, jitter: float = _DEFAULT_JITTER) -> mx.array:
    """Full marginal kernel  `K = L (I + L)^-1`  (symmetric).

    Provided for diagnostics / the intercentrality overlay; the per-step message
    to the mixing head is `diag(K)` alone (`marginal_inclusion`), per doc
    section 2. Computed via the well-conditioned solve `(I+L) K = L`.
    """
    Lm, eye = _prep_L(L, jitter)
    K = mx.linalg.solve(eye + Lm, Lm, stream=mx.cpu)
    return 0.5 * (K + K.T)


def dpp_marginals(
    quality,
    features,
    normalize: bool = True,
    method: str = "eigh",
    jitter: float = _DEFAULT_JITTER,
    return_spectrum: bool = False,
):
    """End-to-end: instrument quality + keys -> `diag(K)` (+ optional spectrum).

    Convenience composing `build_L` then `marginal_inclusion`. This is the signal
    the mixing head reads, concatenated with the raw appetite `b = quality`
    (doc `dpp-mixing-and-overlay.md` section 2:  dw/dt = Head([ diag(K) ++ b ])).

    Parameters
    ----------
    quality  : (N,) per-instrument appetite/quality  q_i.
    features : (N, F) per-instrument diversity keys phi_i.
    normalize, method, jitter, return_spectrum :
        Forwarded to `feature_gram` / `marginal_inclusion`.

    Returns
    -------
    diag_K : (N,) mlx.array marginal inclusion vector.
    eigenvalues : (N,) mlx.array, only if return_spectrum=True (method="eigh").
    """
    L = build_L(quality, features, normalize=normalize)
    if method == "inverse_diff":
        if return_spectrum:
            raise ValueError("return_spectrum requires method='eigh'")
        return marginal_inclusion_diff(L, jitter=jitter)
    return marginal_inclusion(
        L, method=method, jitter=jitter, return_spectrum=return_spectrum
    )
