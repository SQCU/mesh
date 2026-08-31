# CAST — every parameter group in the spec, named

Alice-and-Bob notation for this project. Half the confusion in this repo came from
saying "the relation thing" for three different objects, or "the value" for two. Every
named object below has exactly one referent.

**Invariant over the whole cast:** no parameter's shape depends on `k` (teams), `j`
(carts) or `l` (players). Adding a team, a cart or a player adds ROWS, never columns.
`d ≥ 128` everywhere a width is learned (SPEC §8).

---

## WALLY — `W_all`, the one shared weight set

Every learned parameter below is a member of Wally. There is one Wally for the whole
match, all teams, all players. Teams and players are not separate learners; they are
**activations that select into Wally**:

- **ADA** — `A_team`, the per-team activation rows.
- **PIP** — `A_player`, the per-player activation rows.

Ada and Pip are *not parameters*. They carry no weights of their own; they are which
rows of Wally light up for whom.

## The learned cast (the RL gradient's job is to fill these)

| name | symbol | shape | what it does |
|---|---|---|---|
| **QUINN** | `W_q` | `(d, d_x + d_β)` | projects raw self-state **and** belief into the per-player query. The only place Xan and Bea meet. |
| **KAY** | `W_k` | `(d, d_z)` | projects an instrument descriptor into its key. `Quinn·Kay` **is** the per-(player, instrument) score — the scalar the spec calls for, computed, never stored. |
| **VAL** | `W_v` | `(d_v, d_z)` | the per-instrument *behavioural* value: what pursuing this instrument implies, mixed by the allocation. |
| **GRAHAM** | `A`, with `M = AAᵀ` | `(d, d)` | the learned PSD metric of the Gram, `G = Z M Zᵀ`. The all-to-all coupling. If Graham gets no gradient, there is no learned coupling and the mesh is decoration. |
| **REX** | `w_rel` | `(d_edge,)` | the additive bilinear pair term, `G = ZMZᵀ + E·w_rel`. **Currently fed hand-authored edge rows — that is the §7 violation.** Rex should be reading learned content, not my guesses. |
| **GIA / UMA / DOV** | `w_gate` / `w_up` / `w_down` | `(in,h)`, `(in,h)`, `(h,out)` | the SwiGLU trio. **Gia is the regime switch** — she decides diversify-vs-pile-on from the DPP signal and shared appetite. Dov emits `dw/dt`. |
| **WINNIE** | `W_φ` | `(d_ir, 1)` | the **preservation** value probe. Linear, on the final IR. Trained only on rows whose team holds the path. |
| **LOU** | `L_ψ` | `(d_ir, 1)` | the **acquisition** value probe. Linear, on the final IR. Trained only on rows whose team does not. Not Winnie's sign-flip — different target, deliberately. |
| **VERA_WINNIE** | `Ṽ_φ` | `(d, 1)` | auxiliary probe on the query, regressed toward Winnie. |
| **VERA_LOU** | `Ṽ_ψ` | `(d, 1)` | auxiliary probe on the query, regressed toward Lou. Wherever a value is estimated there are TWO values to estimate, so Vera is always a pair. |
| **DINA** | `b_η(y)`, `A_η(y)` | `(S,)`, `(S, A)` | the action-linear dynamics ensemble, `Δy = b(y) + A(y)u`. The only per-state operator in the cast. |
| **TAU** | `τ` | scalar | sampling temperature. Selection is weighted sampling, never argmax. |
| **ELLE** | L2-toward-0 | scalar | pulls logits to zero so untrained is broad sampling and trained peaks without collapsing. |

## The computed chorus (read-only; `stopgrad`; NOT learned, NOT hand-guessed)

The spec licenses exactly three computed things. Everything else must be learned.

| name | symbol | licensed by | what it is |
|---|---|---|---|
| **XAN** | `x_b` | given | raw per-player engine state: position, velocity, health, armor, weapons bitmask, ammo, powerups, team. The engine's own rows. |
| **ZED** | `z_m` | given | the per-instrument descriptor. |
| **BEA** | `β_b` | `payload-spec §2.2` | the belief — the *only* spatial mixing operator in the system. Built by three helpers below. |
| **RHO** | `ρ(Δt)` | §2.2 | temporal contraction toward an uninformative prior — the buffer forgetting. |
| **GIGI** | `g(dist)` | §2.2 | the bounded-support spatial mask; her support radius is an **output** of the fuse-to-5–15% construction, never a constant. |
| **PIA** | `PW(s)` | `rl-training-spec §1` | the projected winner, nim-sum over cartstate. Closed-form. |
| **SUE** | `SUCC(s)` | §1 | the succession under successive decrement of the leader. Closed-form. This is the anticipation that lets the policy be time-smooth. |
| **NIM** | `N_i(s)` | §1 | per-team standing in the hierarchy. Closed-form. |
| **DEE** | `diag(K)`, `K = L(I+L)⁻¹` | `dpp-mixing-and-overlay` | the DPP marginal-inclusion signal. One repulsion-shaped number per instrument. **Not** the determinant. |

## Frozen, deliberately

| name | what | why |
|---|---|---|
| **CEE** | the Xonotic C program / FPS layer | `stopgrad` by choice, not by nature. There *is* a gradient here; we decline to take it. Steering is skill-orthogonal: we bias where a bot commits, never how it aims. |

## The one legitimate non-feature

**MASK** — eligibility. `eligible = seen and (kind ≠ HUNT_RIVAL or not same_team) and …`
This is a game *rule* about which actions exist, not a hint about which are good. Masks
define the action set; features pre-answer the question. Masks stay.

---

## How to use these names

- "Graham is getting no gradient" — the coupling isn't learned, the Gram is decoration.
- "Rex is being fed hand-authored rows" — the §7 feature-engineering violation.
- "Winnie and Lou are MLPs" — the §5 linear-probe violation (was true; fixed in R24).
- "Pia resolved 0/228" — the CGT never priced a real state (was true; fixed in R25).
- "Gigi's radius is a constant" — the receptive-field bound was never computed (was true; fixed in R24).
- "Bea is inlined twice" — `live_belief` re-implementing `featurize` (was true; fixed in R24).
- "Xan was zeroed" — the per-player state never reached the matmul (R19; corrected by R25 to a logging failure).
- "Dee is stop-gradient'd" — the coupling can't be shaped by reward (was true; fixed in R10).

---

## Tensor shapes, and the ≥128 rule

> i want you to name the parameters in the spec and to describe their tensor shape,
> which will be a minimum of 128 for at least one 'side' of a matrix for many matrices,
> but still be variable beyond 128...

**Given widths** — set by the engine and the game, small, not ours to inflate:
`d_x` raw per-player engine row · `d_z` per-instrument descriptor · `d_c` per-cell slot.

**Learned widths** — every one **≥128, free above**; these are the knobs:
`d_β` belief · `d` query/key space · `d_v` behavioural value · `d_ir` the IR ·
`h` SwiGLU hidden (conventionally ≈ 8/3·d) · `r` Graham's metric rank (≤ d) ·
`d_y`, `d_u` Dina's reduced state / action widths.

| name | tensor | shape | the ≥128 side |
|---|---|---|---|
| **PHIL** `Φ` | belief projection (LEARNED — a constant Φ was a defect) | `(d_β, d_c)` | `d_β` |
| **QUINN** `W_q` | query | `(d, d_x + d_β)` | `d`, and `d_β` within the input |
| **KAY** `W_k` | key | `(d, d_z)` | `d` |
| **VAL** `W_v` | behavioural value | `(d_v, d_z)` | `d_v` |
| **GRAHAM** `A` | metric factor, `M = AAᵀ ∈ (d,d)` | `(d, r)` | `d`; `r` free ≤ `d` |
| **REX** | pair bilinear form (low-rank) | `(d, r_e)` | `d` |
| **GIA** `w_gate` | SwiGLU gate | `(d_ir, h)` | both |
| **UMA** `w_up` | SwiGLU up | `(d_ir, h)` | both |
| **DOV** `w_down` | SwiGLU down → `dw/dt` | `(h, 1)` per instrument row | `h` |
| **WINNIE** `W_φ` | preservation probe | `(d_ir, 1)` | `d_ir` |
| **LOU** `L_ψ` | acquisition probe | `(d_ir, 1)` | `d_ir` |
| **VERA_WINNIE** `Ṽ_φ` | aux probe on the query → Winnie | `(d, 1)` | `d` |
| **VERA_LOU** `Ṽ_ψ` | aux probe on the query → Lou | `(d, 1)` | `d` |
| **DINA** | `b(y)`, `A(y)` | `(d_y,)`, `(d_y, d_u)` | both |

The rule, once: **every matrix has at least one learned side ≥128, and the only small
numbers in the cast are the raw input widths the engine hands us.** A 16, a 21 or a 32
anywhere else is a bottleneck strangling the gradient.

Per-pair quantities are **computed, never stored**: `Quinn·Kay` is the per-(player,
instrument) score, a dot product of two ≥128d learned vectors. Storing an
`(l, m, small)` hand-authored row instead is both the §7 feature-engineering violation
and the reason a transition costs 374 KB.

## Where the code diverges (level-3, measured)

    d = 16          (IR was later widened to 128; d itself was not)
    d_β = 8         RELATION_WIDTH = 16     EDGE_WIDTH = 12    HIERARCHY_WIDTH = 8
    head.py:51  self.in_dim = 5 + RELATION_WIDTH   -> 21
    head.py:52  self.hidden = 32
    head.py:97  rms_norm(self.features(diag_k, appetite, relation), ...)

**The policy head is `21 → 32 → out`, fed by `diag_k`, `appetite` and the hand-authored
relation row — the Gram-IR never reaches it.** `IR_WIDTH = 128` feeds only Winnie and
Lou. So Graham's output never touches an action; the 128-wide learned representation is
a side-channel the critics read while the behaviour is decided by 21 hand-made numbers.

That is SPEC §7 failing precisely where it points:

> if we're spending flops on a gram matrix that gram matrix better fucking end up in
> the IR consumed by subsequent probes

It reaches the probes. It does not reach the policy. This is not a narrow
implementation of the algorithm — it is a different algorithm wearing its variable
names.

---

## Implementation: one header, one composer

> we should be able to describe every single function of learned parameters in the
> strategy program in terms of pytorch dot nn notation. we should be able to say which
> 'cast' member they are. we should be able to say their shape. all of this can be said
> en-comment over where a function over parameters is defined. we can then import only
> functions from the cast_header pyfile into whatever context these are computed. we do
> not need to and will not be 'reusing' existing code. we are replacing the existing
> code with a header of castfunctions and a caller which composes them and introduces
> no parameters, creates no tensors, does nothing at all but compose imported functions
> actually.

- **`xonotic/solver/strat/cast_header.py`** — the ONLY place a parameter exists. Every
  function carries, en-comment, its cast member, its `torch.nn` notation and its shape.
  `Widths` **raises** if any learned side is below 128.
- **`xonotic/solver/strat/strategy.py`** — the composer. Imports only cast functions;
  declares no parameter, fabricates no tensor, re-implements nothing, holds no state.
  Verified: no `nn.`, no `mx.zeros/ones/full/random`, no literal tensor construction.

Amendments made by this instruction:
- **NORM is not a cast member: RMSNorm holds no learned parameters.** The operation is
  still applied where the spec calls for it — the parameter-free
  `x * rsqrt(mean(x²) + eps)` — but it has no learned gain vector, so it names no
  parameter group. The cast lists parameter groups; a parameterless operation owns none.
- **VERA is two** — `VERA_WINNIE` and `VERA_LOU` — because wherever a value is estimated
  there are two values to estimate.
- **PHIL is learned**, not part of the computed chorus. A constant-literal Φ was one of
  the defects found in the belief pipeline (R24), and a fixed Φ is exactly the narrow
  hand-authored bottleneck §7 forbids.
