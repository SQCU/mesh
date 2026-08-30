"""Mixing head: DPP diversity signal + appetite -> per-player strategy VELOCITY.

This is layer-3's mixing head. It turns the honest, symmetric, repulsion-only DPP
signal into behavior under a learned gate, per the committed architecture in
`design/dpp-mixing-and-overlay.md` §2 and `design/payload-spec.md` §2.5:

    dw_b/dt = SwiGLU( RMSNorm( [ diag(K) ; b ] ) )

    diag(K) = marginal inclusion vector, K = L (I + L)^-1   (repulsion-shaped, §2.4)
    b       = per-instrument appetite / quality (the DPP quality terms q_i)

The DPP kernel's only job is to emit an honest diversity SIGNAL (`diag(K)`, NOT the
scalar determinant -- the determinant "can't be two things", `diag(K)` carries one
repulsion-shaped number per instrument). Mapping that signal to behavior is this
LEARNED OPERATOR's job. The **SwiGLU gate is the diversify/pile-on switch**
(`dpp-mixing-and-overlay.md` §2): when it reads high *shared* appetite concentrated on
one instrument it opens the concentration (pile-on / coalition) path and gates the
diversity signal down; otherwise it passes the diversity signal through as spread. The
pile-on regime is ONE of the head's two gated outputs, not a bolted-on parallel term --
no nonsymmetric DPP, nothing beyond RMSNorm -> SwiGLU is required.

The head output is a VELOCITY on the integrated weight state, not an allocation
snapshot. Integration is one forward-Euler step of a replicator flow at the strategy
cadence (`dpp-mixing-and-overlay.md` §4, `payload-spec.md` §3.2):

    w += (dw/dt) * Delta                                    (coprocessor-side)

`Delta` (the emit cadence) is the forward-Euler step size -- a STABILITY parameter, not
a scheduling knob (larger Delta pushes toward discrete-time replicator chaos).

Selection over the integrated weight state is **weighted sampling, NOT argmax/MAP**
(`payload-spec.md` §5): the strategy logits are sampled categorically, L2-regularized
toward 0 so untrained the policy is a broad weighted sampling and with training it peaks
without collapsing. The log-prob of the sampled instrument is exposed for the REINFORCE
policy gradient (`rl-training-spec.md` §2.1 / §3): only `W_all` (this head's weights)
learns; `diag(K)`, `b`, PW, SUCC and the frozen FPS C-program are all stopgrad.

Learned/differentiable parts (the head, sampling log-probs) use **mlx** (Apple). The
deterministic Game-1 features that feed `diag(K)`/`b` (PW, SUCC, coupling `kappa`,
V-cell featurization) are computed elsewhere in numpy/plain python and enter here already
detached -- this module never differentiates through them.

Spec: `dpp-mixing-and-overlay.md` §2 (mixing head, SwiGLU gate) + §4 (Euler cadence);
      `payload-spec.md` §2.5 (mix) + §3.2 (step) + §5 (sampling, not MAP).

Public surface
--------------
- ``MixingHead``            : nn.Module, RMSNorm -> SwiGLU, [diag(K); b] -> dw/dt.
- ``rms_norm``              : functional RMSNorm.
- ``swiglu``                : functional SwiGLU (gate * silu, projected).
- ``integrate_weights``     : forward-Euler ``w += (dw/dt) * Delta`` (the flow step).
- ``sample_strategy``       : weighted (categorical) sampling over logits + its log-prob.
- ``strategy_log_prob``     : log-prob of a *given* action (for replay / REINFORCE).
- ``strategy_probs``        : softmax of the strategy logits (diagnostic).
"""

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


# --------------------------------------------------------------------------- #
# Functional pieces of the committed head: RMSNorm -> SwiGLU.
# --------------------------------------------------------------------------- #

def rms_norm(x: mx.array, weight: mx.array | None = None, eps: float = 1e-6) -> mx.array:
    """Root-mean-square layer norm over the last (feature) axis.

    The first stage of the committed head (`dpp-mixing-and-overlay.md` §2:
    ``Head = RMSNorm -> SwiGLU``). Normalizes the concatenated ``[diag(K); b]`` feature
    row by its RMS so the diversity signal and the appetite enter the gate at a common
    scale. ``weight`` is an optional learned per-feature gain (broadcast over the batch).
    """
    scale = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * scale
    if weight is not None:
        y = y * weight
    return y


def swiglu(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
    b_gate: mx.array | None = None,
    b_up: mx.array | None = None,
    b_down: mx.array | None = None,
) -> mx.array:
    """SwiGLU: ``down( silu(x @ w_gate) * (x @ w_up) )``.

    The second stage of the committed head. The gated branch ``silu(x @ w_gate)`` is the
    **regime switch** (`dpp-mixing-and-overlay.md` §2): it is the learned gate that turns
    the single repulsion-shaped ``diag(K)`` into either diversify (spread) or concentrate
    (pile-on / coalition) behavior. All weight matrices are ``(in, out)``; ``x`` is
    ``(..., in)``.
    """
    g = x @ w_gate
    if b_gate is not None:
        g = g + b_gate
    u = x @ w_up
    if b_up is not None:
        u = u + b_up
    h = nn.silu(g) * u
    out = h @ w_down
    if b_down is not None:
        out = out + b_down
    return out


# --------------------------------------------------------------------------- #
# The mixing head module.
# --------------------------------------------------------------------------- #

class MixingHead(nn.Module):
    """RMSNorm -> SwiGLU head emitting a per-player strategy VELOCITY over instruments.

    Implements ``dw_b/dt = SwiGLU( RMSNorm( [ diag(K) ; b ] ) )``
    (`payload-spec.md` §2.5, `dpp-mixing-and-overlay.md` §2). Deliberately small: its only
    expressive burden is to route one repulsion-shaped signal into one of two behaviors
    under the SwiGLU gate. The projection widths are candidate choices (the doc marks them
    non-fixed); the RMSNorm -> SwiGLU shape and the ``diag(K)``-not-determinant / velocity
    -not-snapshot interface are firm.

    These weights are part of the single shared policy ``W_all`` (`rl-training-spec.md`
    §0/§4): there is ONE head for the whole population; per-team ``A_team`` / per-player
    ``A_player`` distinctions live in the *activations* (the per-player ``b`` and
    ``diag(K)`` rows fed in), never in separate head weights. The policy gradient updates
    these weights only.

    Parameters
    ----------
    n_instruments : int
        M -- the number of instruments (push cart k, suppress cart k, contest post p,
        hunt rival r, explore cell c, ...; `payload-spec.md` §2.1/§4.4). Output width.
    hidden : int, optional
        SwiGLU hidden width. Defaults to ``4 * n_instruments`` (candidate, not fixed).
    eps : float
        RMSNorm epsilon.

    Call
    ----
    ``head(diag_k, b) -> dw_dt``  where
        ``diag_k`` : ``(..., M)`` marginal inclusion vector (broadcast over players);
        ``b``      : ``(P, M)`` per-player appetite / quality;
        ``dw_dt``  : ``(P, M)`` per-player velocity over instruments.

    Both inputs are treated as detached features here -- this module differentiates only
    through its own weights (stopgrad on ``diag(K)``/``b``, `rl-training-spec.md` §2.1).
    """

    def __init__(self, n_instruments: int, hidden: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.n_instruments = int(n_instruments)
        self.in_dim = 2 * self.n_instruments  # [diag(K) ; b]
        self.hidden = int(hidden) if hidden is not None else 4 * self.n_instruments
        self.eps = float(eps)

        # RMSNorm gain over the concatenated [diag(K); b] feature row.
        self.norm_weight = mx.ones((self.in_dim,))

        # SwiGLU projections. gate/up: in_dim -> hidden ; down: hidden -> M.
        scale_in = 1.0 / (self.in_dim ** 0.5)
        scale_h = 1.0 / (self.hidden ** 0.5)
        self.w_gate = mx.random.normal((self.in_dim, self.hidden)) * scale_in
        self.w_up = mx.random.normal((self.in_dim, self.hidden)) * scale_in
        # Down projection initialized small so the untrained velocity is near zero
        # (broad-sampling / L2-toward-0 regime of payload-spec.md §5).
        self.w_down = mx.random.normal((self.hidden, self.n_instruments)) * scale_h * 0.1

    def features(self, diag_k: mx.array, b: mx.array) -> mx.array:
        """Concatenate ``[diag(K) ; b]`` with ``diag(K)`` broadcast over players.

        ``diag(K)`` is per-instrument (shared, `payload-spec.md` §2.4); ``b`` is
        per-player-per-instrument. Returns the ``(P, 2M)`` input row for the head.
        """
        if diag_k.ndim < b.ndim:
            diag_k = mx.broadcast_to(diag_k, b.shape)
        return mx.concatenate([diag_k, b], axis=-1)

    def __call__(self, diag_k: mx.array, b: mx.array) -> mx.array:
        """Per-player strategy velocity ``dw_b/dt`` over instruments. See class doc."""
        x = self.features(diag_k, b)
        x = rms_norm(x, self.norm_weight, self.eps)
        return swiglu(x, self.w_gate, self.w_up, self.w_down)


# --------------------------------------------------------------------------- #
# The forward-Euler flow step (coprocessor-side integration).
# --------------------------------------------------------------------------- #

def integrate_weights(w: mx.array, dw_dt: mx.array, delta: float) -> mx.array:
    """One forward-Euler replicator step: ``w_new = w + (dw/dt) * Delta``.

    The integration of the strategy weight state at the strategy cadence, run
    coprocessor-side / off-engine (`payload-spec.md` §3.2 ``step``,
    `dpp-mixing-and-overlay.md` §4). ``Delta`` is the emit-cadence step size and a
    STABILITY parameter (§4): larger Delta -> lower cadence -> toward discrete-time
    replicator chaos. The engine later reads the integrated **absolute** ``w`` as
    routerating bias (`payload-spec.md` §3.1 / §4.1); this function does not decide the
    instantaneous-vs-time-averaged trust question (§4, [OPEN]) -- it emits the
    instantaneous state and a running average, if wanted, is the caller's.

    Returns the new absolute weight state (same shape as ``w``).
    """
    return w + dw_dt * delta


# --------------------------------------------------------------------------- #
# Selection: weighted sampling over the strategy logits (NOT argmax/MAP), with
# the log-prob of the sampled action for the REINFORCE policy gradient.
# --------------------------------------------------------------------------- #

def strategy_probs(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """Softmax of the strategy logits over instruments (diagnostic / expected weights).

    The integrated absolute weight state ``w`` over instruments IS the strategy logits
    (`payload-spec.md` §5: weighted sampling over strategies, L2-regularized toward
    logit 0 so untrained -> broad, trained -> peaked). Not used for selection (that is
    sampled, below) -- exposed for logging and for the value/return machinery.
    """
    return mx.softmax(logits / temperature, axis=-1)


def strategy_log_prob(logits: mx.array, action: mx.array, temperature: float = 1.0) -> mx.array:
    """Log-prob ``log pi(action | logits)`` of a GIVEN instrument choice.

    The differentiable quantity the REINFORCE policy gradient weights by the advantage
    (`rl-training-spec.md` §2.1: ``grad J = E[ sum_u A_u (.) grad log pi(a_u|...) ]``;
    §3: ``L_pg = -E[ sum A.detach() (.) log pi ]``). Used when replaying a buffered
    ``(state, activation, action, logpi)`` transition -- pass the stored ``action`` and
    the freshly recomputed ``logits`` so the gradient flows into ``W_all`` only.

    ``logits`` : ``(..., M)``. ``action`` : integer index array ``(...)`` selecting an
    instrument along the last axis. Returns per-row log-prob ``(...)``.
    """
    logp = logits / temperature - mx.logsumexp(logits / temperature, axis=-1, keepdims=True)
    return mx.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def sample_strategy(
    logits: mx.array,
    temperature: float = 1.0,
    key: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Weighted (categorical) sample over the strategy logits, plus its log-prob.

    **Sampling, NOT argmax / MAP** (`payload-spec.md` §5): the policy is a weighted
    sampling over strategies; greedy selection is explicitly excluded so that untrained
    the head is a broad weighted sampling of effective strategies and with training it
    peaks without collapsing to "only some actions happening". This is the action the
    bot commits to; the returned log-prob is the ``log pi(a|s,b,SUCC)`` term of the
    REINFORCE gradient (`rl-training-spec.md` §2.1).

    Parameters
    ----------
    logits : ``(..., M)``
        The integrated strategy weight state over instruments (the logits).
    temperature : float
        Softmax temperature; ``1.0`` = the trained sharpness. Higher = broader sampling.
    key : mx.array, optional
        PRNG key for reproducible sampling (``mx.random.key(seed)``). If ``None`` the
        global mlx RNG is used.

    Returns
    -------
    (action, log_prob)
        ``action``   : integer index array ``(...)`` -- the sampled instrument per row;
        ``log_prob`` : ``(...)`` -- log-prob of that sampled action (for the policy grad).
    """
    scaled = logits / temperature
    action = mx.random.categorical(scaled, axis=-1, key=key)
    log_prob = strategy_log_prob(logits, action, temperature=temperature)
    return action, log_prob
