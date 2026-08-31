"""STRATEGY — the composer. It owns nothing.

This file is the strategy parametric function, in one place. It imports ONLY cast
functions from ``cast_header`` and wires their outputs to their inputs.

What this file may NOT do, by construction:
  * declare a parameter                      -- every parameter is in cast_header
  * fabricate content (zeros, constants,
    hand-authored features, magic widths)    -- SPEC §7
  * re-implement a cast function             -- one definition, imported
  * hold state between calls                 -- it is a pure composition

What it receives are the COMPUTED CHORUS values (``design/CAST.md``): the three
things the spec licenses as computed rather than learned, plus the engine's own
raw rows. They arrive already computed; this file never derives them.

    XAN   x        raw per-player engine rows          (l, d_x)
    ZED   z        per-instrument descriptors          (m, d_z)
    cells f_eff    temporally-contracted cell slots    (c, d_c)   -- RHO applied
    gigi  g        bounded-support spatial mask        (l, c)     -- GIGI
    DEE   diag(K)  DPP marginal inclusion              (m,)
    PIA/SUE/NIM    closed-form cartstate semantics     (l, d_sem) -- read, not learned
    MASK  eligible which actions EXIST                 (l, m) bool

The per-(player, instrument) score is ``quinn · kay`` — a dot product of two
>=128d learned vectors, computed here and never stored.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from .cast_header import (
    Wally,
    dina_drift,
    dina_matrix,
    gia_uma_dov,
    graham,
    kay,
    lou,
    phil,
    quinn,
    rex,
    tau,
    val,
    vera_lou,
    vera_winnie,
    winnie,
)

__all__ = ["Strategy", "strategy", "dynamics", "log_probs", "logp_of", "act", "integrate"]


class Strategy(NamedTuple):
    """Everything the composition produces. No member is stored anywhere."""

    dw_dt: mx.array        # (l, m)   DOV's velocity, one per instrument row
    logits: mx.array       # (l, m)   masked score / TAU, for sampling
    ir: mx.array           # (l, m, d_ir)
    query: mx.array        # (l, d)
    value_winnie: mx.array # (l,)     WINNIE, on the IR
    value_lou: mx.array    # (l,)     LOU, on the IR
    aux_winnie: mx.array   # (l,)     VERA_WINNIE, on the query
    aux_lou: mx.array      # (l,)     VERA_LOU, on the query
    coupling: mx.array     # (l, l)   GRAHAM + REX, the all-to-all term


def strategy(
    wally: Wally,
    xan: mx.array,        # (l, d_x)   raw engine rows
    zed: mx.array,        # (m, d_z)   instrument descriptors
    cell_slots: mx.array, # (c, d_c)   RHO-contracted cell slots
    gigi: mx.array,       # (l, c)     GIGI's bounded-support mask
    dee: mx.array,        # (m,)       DEE, the DPP marginal inclusion
    semantics: mx.array,  # (l, d_sem) PIA / SUE / NIM, read-only
    mask: mx.array,       # (l, m)     MASK: which actions exist
) -> Strategy:
    """Compose the cast into the strategy parametric function.

    Every line below is a call to an imported cast function or a wiring of their
    outputs. Nothing is invented here.
    """
    # BEA -- the belief. The only spatial mixing operator in the system: PHIL
    # projects the cell slots, GIGI weights them by bounded graph distance.
    bea = gigi @ phil(wally, cell_slots)                       # (l, d_beta)

    # QUINN -- the query; the only place XAN and BEA meet.
    q = quinn(wally, xan, bea)                                 # (l, d)

    # KAY / VAL -- per-instrument key and behavioural value.
    k = kay(wally, zed)                                        # (m, d)
    v = val(wally, zed)                                        # (m, d_v)

    # The per-(player, instrument) score: a dot product, computed, never stored.
    score = q @ k.T                                            # (l, m)

    # GRAHAM + REX -- the all-to-all coupling over the player rows, O(l²).
    coupling = graham(wally, q) + rex(wally, q)                # (l, l)

    # The IR every probe and the head read. The coupling mixes player rows; the
    # score and DEE select per instrument; VAL carries what pursuing it implies.
    mixed = coupling @ q                                       # (l, d)
    ir = (
        mixed[:, None, :]
        + (score * dee[None, :])[:, :, None] * v[None, :, :]
    )                                                          # (l, m, d_ir)

    # GIA / UMA / DOV -- the velocity. No normalisation: NORM is dropped.
    dw_dt = gia_uma_dov(wally, ir)                             # (l, m)

    # Sampling is weighted, never argmax; MASK removes actions that do not exist.
    logits = mx.where(mask, score / tau(wally), -mx.inf)       # (l, m)

    # WINNIE / LOU on the IR; VERA_WINNIE / VERA_LOU on the query. Two values
    # wherever a value is estimated.
    pooled = mx.sum(ir * mask[:, :, None], axis=1)             # (l, d_ir)
    return Strategy(
        dw_dt=dw_dt,
        logits=logits,
        ir=ir,
        query=q,
        value_winnie=winnie(wally, pooled),
        value_lou=lou(wally, pooled),
        aux_winnie=vera_winnie(wally, q),
        aux_lou=vera_lou(wally, q),
        coupling=coupling,
    )


def dynamics(wally: Wally, y: mx.array, u: mx.array) -> mx.array:
    """DINA -- the local action-linear model, ``Δy = b(y) + A(y)·u``.

    y (..., d_y) reduced state, u (..., d_u) reduced action -> (..., d_y).
    """
    return dina_drift(wally, y) + mx.squeeze(
        dina_matrix(wally, y) @ u[..., None], axis=-1
    )


# --------------------------------------------------------------------------
# Policy read-out. ONE definition, used by the responder to ACT and by the
# learner to EVALUATE. Two copies would make exp(logpi_target - logpi_behavior)
# compare numbers produced by different code, which is the failure the project
# law exists to prevent (see design/CAST.md).
# --------------------------------------------------------------------------


def log_probs(out: Strategy) -> mx.array:
    """Normalised log-probabilities over instruments.  ``(l, m)``"""
    return out.logits - mx.logsumexp(out.logits, axis=-1, keepdims=True)


def logp_of(out: Strategy, actions: mx.array) -> mx.array:
    """log pi of actions already taken — the LEARNER's read-out.  ``(l,)``"""
    return mx.take_along_axis(log_probs(out), actions[:, None], axis=-1)[:, 0]


def act(out: Strategy, key: mx.array) -> tuple[mx.array, mx.array]:
    """Sample actions and return their log pi — the RESPONDER's read-out.

    Weighted sampling, never argmax (SPEC §10). Returns ``(actions, logp)`` from
    the SAME log_probs the learner will later evaluate.
    """
    actions = mx.random.categorical(out.logits, key=key)
    return actions, logp_of(out, actions)


def integrate(w: mx.array, dw_dt: mx.array, delta: float) -> mx.array:
    """The replicator step, ``w <- w + (dw/dt)*delta``. ONE definition.

    `delta` is the strategy cadence, and therefore the forward-Euler step size
    and a stability parameter — not a free scheduling knob.
    """
    return w + dw_dt * delta
