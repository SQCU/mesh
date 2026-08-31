"""CAST_HEADER — every learned parameter of the strategy program, defined once.

This file is the ONLY place in the strategy program where a parameter exists.
Every function below owns cast members (see ``design/CAST.md``); every function
elsewhere composes these and owns nothing.

Each definition carries, en-comment:
  * the cast member(s) it holds,
  * the ``torch.nn`` notation for the function, and
  * the tensor shape.

The implementation is mlx (that is what runs on the mini); the ``nn`` notation in
the comments is the lingua franca for what the function IS.

WIDTH RULE (SPEC §8). Every learned side is >= 128 and free above:
    d_beta, d, d_v, d_ir, h, r, r_e, d_y, d_u   -- knobs with a 128 floor
The only small widths in the program are the raw input widths the engine hands us:
    d_x   raw per-player engine row
    d_z   per-instrument descriptor
    d_c   per-cell slot vector

COUNT INVARIANCE. No shape below mentions k (teams), j (carts) or l (players).
Adding a team, cart or player adds ROWS, never columns.

NORM IS NOT A CAST MEMBER. RMSNorm is applied where the spec calls for it, but it
holds NO LEARNED PARAMETERS -- it is the parameter-free ``x * rsqrt(mean(x*x) + eps)``,
with no learned gain vector. It therefore has no entry in the cast: the cast names
parameter groups, and a parameterless operation owns none.

VERA IS TWO. Wherever a value is estimated there are two values to estimate, so
the auxiliary probe is VERA_WINNIE and VERA_LOU, never one Vera.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "Widths", "Wally",
    "phil", "quinn", "kay", "val",
    "graham", "rex",
    "gia_uma_dov",
    "winnie", "lou", "vera_winnie", "vera_lou",
    "dina_drift", "dina_matrix",
    "tau", "elle",
]

MIN_LEARNED_WIDTH = 128


@dataclass(frozen=True)
class Widths:
    """The knobs. Learned widths carry a 128 floor; given widths are the engine's."""

    # given by the engine / the game -- small, not ours to inflate
    d_x: int          # raw per-player engine row
    d_z: int          # per-instrument descriptor
    d_c: int          # per-cell slot vector

    # learned -- every one >= 128, free above
    d_beta: int = 128  # belief
    d: int = 128       # query / key space
    d_v: int = 128     # behavioural value
    d_ir: int = 128    # the IR
    h: int = 341      # SwiGLU hidden, conventionally ~ 8/3 * d
    r: int = 128       # Graham's metric rank (<= d)
    r_e: int = 128     # Rex's pair-form rank
    d_y: int = 128     # Dina's reduced state
    d_u: int = 128     # Dina's reduced action

    def __post_init__(self) -> None:
        for name in ("d_beta", "d", "d_v", "d_ir", "h", "r", "r_e", "d_y", "d_u"):
            got = getattr(self, name)
            if got < MIN_LEARNED_WIDTH:
                raise ValueError(
                    f"{name}={got} is below the learned-width floor "
                    f"{MIN_LEARNED_WIDTH}: a narrow learned side strangles the "
                    f"gradient (SPEC §8)."
                )
        if self.r > self.d:
            raise ValueError(f"Graham's rank r={self.r} must be <= d={self.d}.")


class Wally(nn.Module):
    """WALLY -- ``W_all``, the one shared weight set.

    One Wally for the whole match, all teams, all players. Teams and players are
    not separate learners: ADA (``A_team``) and PIP (``A_player``) are activation
    ROWS that select into Wally, and carry no weights of their own.
    """

    def __init__(self, w: Widths):
        super().__init__()
        self.w = w

        # PHIL -- Φ, the low-rank cell projection inside the belief.
        # nn.Linear(d_c, d_beta, bias=False)          weight (d_beta, d_c)
        self.phil = nn.Linear(w.d_c, w.d_beta, bias=False)

        # QUINN -- W_q. The only place XAN (raw self-state) and BEA (belief) meet.
        # nn.Linear(d_x + d_beta, d, bias=False)      weight (d, d_x + d_beta)
        self.quinn = nn.Linear(w.d_x + w.d_beta, w.d, bias=False)

        # KAY -- W_k. Quinn·Kay IS the per-(player, instrument) score: a dot
        # product of two >=128d learned vectors, COMPUTED, never stored.
        # nn.Linear(d_z, d, bias=False)               weight (d, d_z)
        self.kay = nn.Linear(w.d_z, w.d, bias=False)

        # VAL -- W_v, the per-instrument behavioural value.
        # nn.Linear(d_z, d_v, bias=False)             weight (d_v, d_z)
        self.val = nn.Linear(w.d_z, w.d_v, bias=False)

        # GRAHAM -- A, the metric factor of the Gram: M = A Aᵀ, G = Z M Zᵀ.
        # The all-to-all coupling. No gradient to Graham => the mesh is decoration.
        # nn.Linear(d, r, bias=False)                 weight (r, d)
        self.graham = nn.Linear(w.d, w.r, bias=False)

        # REX -- the additive pair form, low-rank, reading LEARNED row content.
        # It must never be fed hand-authored edge rows (SPEC §7).
        # nn.Linear(d, r_e, bias=False)               weight (r_e, d)
        self.rex = nn.Linear(w.d, w.r_e, bias=False)

        # GIA / UMA / DOV -- the SwiGLU trio, on the IR. Gia is the regime switch
        # (diversify vs pile-on); Dov emits dw/dt, one scalar per instrument row.
        # RMSNorm is applied to the input inside gia_uma_dov, PARAMETER-FREE:
        # there is no learned gain, so no cast member.
        # nn.Linear(d_ir, h, bias=False)              weight (h, d_ir)
        # nn.Linear(d_ir, h, bias=False)              weight (h, d_ir)
        # nn.Linear(h, 1, bias=False)                 weight (1, h)
        self.gia = nn.Linear(w.d_ir, w.h, bias=False)
        self.uma = nn.Linear(w.d_ir, w.h, bias=False)
        self.dov = nn.Linear(w.h, 1, bias=False)

        # WINNIE -- the PRESERVATION value probe. Linear, on the final IR.
        # nn.Linear(d_ir, 1, bias=False)              weight (1, d_ir)
        self.winnie = nn.Linear(w.d_ir, 1, bias=False)

        # LOU -- the ACQUISITION value probe. Linear, on the final IR. Not
        # Winnie's sign-flip: a different target, deliberately.
        # nn.Linear(d_ir, 1, bias=False)              weight (1, d_ir)
        self.lou = nn.Linear(w.d_ir, 1, bias=False)

        # VERA_WINNIE / VERA_LOU -- the auxiliary probes on the QUERY, regressed
        # toward Winnie and Lou respectively. Two, because wherever a value is
        # estimated there are two values to estimate.
        # nn.Linear(d, 1, bias=False)                 weight (1, d)   [each]
        self.vera_winnie = nn.Linear(w.d, 1, bias=False)
        self.vera_lou = nn.Linear(w.d, 1, bias=False)

        # DINA -- the action-linear dynamics ensemble, Δy = b(y) + A(y)·u.
        # The only per-state operator in the cast.
        # nn.Linear(d_y, d_y, bias=False)             weight (d_y, d_y)
        # nn.Linear(d_y, d_y * d_u, bias=False)       weight (d_y*d_u, d_y)
        self.dina_drift = nn.Linear(w.d_y, w.d_y, bias=False)
        self.dina_matrix = nn.Linear(w.d_y, w.d_y * w.d_u, bias=False)

        # TAU -- sampling temperature. Selection is weighted sampling, never argmax.
        # a learned scalar                            shape ()
        self.tau_raw = mx.zeros(())


# --------------------------------------------------------------------------
# The cast functions. Every one owns parameters; nothing else in the program
# may. A caller imports these and composes them.
# --------------------------------------------------------------------------


def phil(wally: Wally, cell_slots: mx.array) -> mx.array:
    """PHIL -- Φ. Project per-cell slot vectors into the belief space.

    nn: ``nn.Linear(d_c, d_beta, bias=False)``      weight ``(d_beta, d_c)``
    in  ``(..., d_c)``   out ``(..., d_beta)``
    """
    return wally.phil(cell_slots)


def quinn(wally: Wally, xan: mx.array, bea: mx.array) -> mx.array:
    """QUINN -- W_q. Per-player query from raw self-state and belief.

    nn: ``nn.Linear(d_x + d_beta, d, bias=False)``  weight ``(d, d_x + d_beta)``
    in  ``(l, d_x)``, ``(l, d_beta)``   out ``(l, d)``
    """
    return wally.quinn(mx.concatenate([xan, bea], axis=-1))


def kay(wally: Wally, zed: mx.array) -> mx.array:
    """KAY -- W_k. Per-instrument key.

    nn: ``nn.Linear(d_z, d, bias=False)``           weight ``(d, d_z)``
    in  ``(m, d_z)``   out ``(m, d)``

    The per-(player, instrument) score is ``quinn(...) @ kay(...).T`` -- computed
    on demand, never materialised as a stored pair tensor.
    """
    return wally.kay(zed)


def val(wally: Wally, zed: mx.array) -> mx.array:
    """VAL -- W_v. Per-instrument behavioural value.

    nn: ``nn.Linear(d_z, d_v, bias=False)``         weight ``(d_v, d_z)``
    in  ``(m, d_z)``   out ``(m, d_v)``
    """
    return wally.val(zed)


def graham(wally: Wally, rows: mx.array) -> mx.array:
    """GRAHAM -- A, giving the Gram ``G = Z M Zᵀ`` with ``M = A Aᵀ``.

    nn: ``nn.Linear(d, r, bias=False)``             weight ``(r, d)``
    in  ``(n, d)``   out ``(n, n)``  -- the all-to-all coupling, O(n²) by nature.
    """
    projected = wally.graham(rows)                       # (n, r)
    return projected @ projected.T / (wally.w.r ** 0.5)  # (n, n)


def rex(wally: Wally, rows: mx.array) -> mx.array:
    """REX -- the additive pair form, from LEARNED row content only.

    nn: ``nn.Linear(d, r_e, bias=False)``           weight ``(r_e, d)``
    in  ``(n, d)``   out ``(n, n)``

    SPEC §7: Rex must never be fed hand-authored edge features. Any pairwise
    fact worth having is to be learned from the rows, not written by hand.
    """
    projected = wally.rex(rows)                          # (n, r_e)
    return projected @ projected.T / (wally.w.r_e ** 0.5)


def gia_uma_dov(wally: Wally, ir: mx.array) -> mx.array:
    """GIA / UMA / DOV -- the SwiGLU head on the IR, emitting ``dw/dt``.

    nn: ``nn.Linear(d_ir, h, bias=False)`` (Gia, gate)   weight ``(h, d_ir)``
        ``nn.Linear(d_ir, h, bias=False)`` (Uma, up)     weight ``(h, d_ir)``
        ``nn.Linear(h, 1,  bias=False)``   (Dov, down)   weight ``(1, h)``
    in  ``(l, m, d_ir)``   out ``(l, m)``  -- one velocity per instrument row.

    RMSNorm is applied to the input, PARAMETER-FREE (no learned gain vector), so
    it contributes no cast member. Gia is the regime switch -- she opens the concentrate path on high shared
    appetite and otherwise passes the diversify signal.
    """
    normed = ir * mx.rsqrt(mx.mean(ir * ir, axis=-1, keepdims=True) + 1e-6)
    return (nn.silu(wally.gia(normed)) * wally.uma(normed)) @ wally.dov.weight.T[..., 0]


def winnie(wally: Wally, ir: mx.array) -> mx.array:
    """WINNIE -- the PRESERVATION value, a linear probe on the final IR.

    nn: ``nn.Linear(d_ir, 1, bias=False)``          weight ``(1, d_ir)``
    in  ``(..., d_ir)``   out ``(...,)``  -- one scalar per activation row.
    """
    return wally.winnie(ir)[..., 0]


def lou(wally: Wally, ir: mx.array) -> mx.array:
    """LOU -- the ACQUISITION value, a linear probe on the final IR.

    nn: ``nn.Linear(d_ir, 1, bias=False)``          weight ``(1, d_ir)``
    in  ``(..., d_ir)``   out ``(...,)``  -- one scalar per activation row.
    """
    return wally.lou(ir)[..., 0]


def vera_winnie(wally: Wally, query: mx.array) -> mx.array:
    """VERA_WINNIE -- auxiliary probe on the QUERY, regressed toward Winnie.

    nn: ``nn.Linear(d, 1, bias=False)``             weight ``(1, d)``
    in  ``(l, d)``   out ``(l,)``
    """
    return wally.vera_winnie(query)[..., 0]


def vera_lou(wally: Wally, query: mx.array) -> mx.array:
    """VERA_LOU -- auxiliary probe on the QUERY, regressed toward Lou.

    nn: ``nn.Linear(d, 1, bias=False)``             weight ``(1, d)``
    in  ``(l, d)``   out ``(l,)``

    Wherever a value is estimated there are two values to estimate; the auxiliary
    probe is therefore a pair, never a single Vera.
    """
    return wally.vera_lou(query)[..., 0]


def dina_drift(wally: Wally, y: mx.array) -> mx.array:
    """DINA (drift) -- ``b(y)`` of ``Δy = b(y) + A(y)·u``.

    nn: ``nn.Linear(d_y, d_y, bias=False)``         weight ``(d_y, d_y)``
    in  ``(..., d_y)``   out ``(..., d_y)``
    """
    return wally.dina_drift(y)


def dina_matrix(wally: Wally, y: mx.array) -> mx.array:
    """DINA (matrix) -- ``A(y)`` of ``Δy = b(y) + A(y)·u``, the per-state operator.

    nn: ``nn.Linear(d_y, d_y * d_u, bias=False)``   weight ``(d_y*d_u, d_y)``
    in  ``(..., d_y)``   out ``(..., d_y, d_u)``
    """
    flat = wally.dina_matrix(y)
    return flat.reshape(*flat.shape[:-1], wally.w.d_y, wally.w.d_u)


def tau(wally: Wally) -> mx.array:
    """TAU -- sampling temperature, strictly positive. Sampling, never argmax.

    a learned scalar                                shape ``()``
    """
    return mx.exp(wally.tau_raw)


def elle(logits: mx.array) -> mx.array:
    """ELLE -- the L2-toward-zero pull on the logits.

    Not a parameter: a penalty over the policy's own output, so that untrained is
    broad weighted sampling and trained peaks without collapsing to one action.
    in  ``(l, m)``   out ``()``
    """
    return mx.mean(logits * logits)
