# CAST — every parameter group in the spec, named

Alice-and-Bob notation for this project. Half the confusion in this repo came from
saying "the relation thing" for three different objects, or "the value" for two. Every
named object below has exactly one referent.

**Invariant over the whole cast:** no parameter's shape depends on `k` (teams), `j`
(carts) or `l` (players). Adding a team, a cart or a player adds ROWS, never columns.
Every learned width is realized literally from the run configuration. Hardware saturation
measurements select useful configurations; the model layer does not impose a width class.

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
| **QUINN** | `W_q` | `(d, d_x + d_β + d_sem)` | projects raw self-state, belief, and the closed-form hierarchy into the per-player query. The only place Xan, Bea, and PIA/SUE/NIM meet. |
| **KAY** | `W_k` | `(d, d_z)` | projects an instrument descriptor into its key. `Quinn·Kay` **is** the per-(player, instrument) score — the scalar the spec calls for, computed, never stored. |
| **VAL** | `W_v` | `(d_v, d_z)` | the per-instrument *behavioural* value: what pursuing this instrument implies, mixed by the allocation. |
| **IR_QUERY / IR_VALUE** | `W_iq`, `W_iv` | `(d_ir,d)`, `(d_ir,d_v)` | place independently variable query and behavioural-value widths in the shared IR without an equality assumption. |
| **TEAM_METRIC** | `A`, with `M = AᵀA` | `(r, d)` | the learned PSD metric used to construct the same-team Gram matrix, `G = Z M Zᵀ`. If this factor gets no gradient, there is no learned same-team coupling. |
| **RIVAL_METRIC** | `R` | `(r_e,d)` | the learned metric factor used to construct the rival-row Gram matrix. It consumes no hand-authored edge rows. |
| **SCALE_IN / ROUTER** | `W_si`, `W_sr` | `(R,d_ir)`, `(E,R)` | lifts every participant/instrument relation into the residual basis and selects top-k experts from live content. |
| **SCALE_EXPERTS** | `W_1`, `W_2` | `(E,R,F)`, `(E,F,R)` | the sorted gather-matmul MoE. Routing is hard in identity and soft in weight, so the selected gate and both expert matrices receive the actor gradient. |
| **SCALE_PROBE / SCALE_OUT** | `p_R`, `W_so` | `(R)`, `(d_ir,R)` | probes the nonlinear residual-feature Gram matrix and returns its fused result to the final IR before every policy, value, and dynamics head. |
| **GIA / UMA / DOV** | `w_gate` / `w_up` / `w_down` | `(in,h)`, `(in,h)`, `(h,out)` | the SwiGLU trio. **Gia is the regime switch** — she decides diversify-vs-pile-on from the DPP signal and shared appetite. Dov emits `dw/dt`. |
| **ACTUATOR** | `W_a` | `(6, d_ir)` | emits Gaussian means and log-scales for gain, travel extension, and spawn on every participant/instrument IR row. Gain and spawn are copied literally; the extension is softplus-mapped and added to canonical stock walking time to produce commitment. |
| **WINNIE** | `W_φ` | `(d_ir, 1)` | the **preservation** value probe. Linear, on the final IR. Trained only on rows whose team holds the path. |
| **LOU** | `L_ψ` | `(d_ir, 1)` | the **acquisition** value probe. Linear, on the final IR. Trained only on rows whose team does not. Not Winnie's sign-flip — different target, deliberately. |
| **VERA_WINNIE** | `Ṽ_φ` | `(d, 1)` | auxiliary probe on the query, regressed toward Winnie. |
| **VERA_LOU** | `Ṽ_ψ` | `(d, 1)` | auxiliary probe on the query, regressed toward Lou. Wherever a value is estimated there are TWO values to estimate, so Vera is always a pair. |
| **DINA_STATE / ACTION / READOUT** | `W_y`, `W_u`, `W_o` | `(d_y,d)`, `(d_u,d_ir)`, `(d,d_y)` | maps independently variable query, IR, and reduced-state widths into and out of the local dynamics space. |
| **DINA ×2** | `b_η1(y)`, `A_η1(y)`, `b_η2(y)`, `A_η2(y)` | two copies of `(d_y,)`, `(d_y,d_u)` | the two-model action-linear dynamics ensemble. Its mean supplies correction; disagreement supplies bounded exploration. |
| **TAU** | `τ` | scalar | sampling temperature. Selection is weighted sampling, never argmax. |
| **ELLE** | L2-toward-0 | scalar | pulls logits to zero so untrained is broad sampling and trained peaks without collapsing. |

## The computed chorus (read-only; `stopgrad`; NOT learned, NOT hand-guessed)

The spec licenses exactly three computed things. Everything else must be learned.

| name | symbol | licensed by | what it is |
|---|---|---|---|
| **XAN** | `x_b` | given | raw per-player engine state: unscaled position, velocity, health, armor, six ammo resources, absolute powerup/spawn/engine timestamps, alive state, and a lossless expansion of all three stock weapon words. IDs and team labels route rows but do not enter this learned vector. |
| **ZED** | `z_m` | given | the instrument kind tag and literal availability, position, path position/length, speed, respawn timestamp, health, and observation timestamp. |
| **BEA** | `β_b` | `payload-spec §2.2` | the belief — the *only* spatial mixing operator in the system. Built by three helpers below. |
| **RHO** | `ρ(Δt)` | §2.2 | temporal contraction toward an uninformative prior — the buffer forgetting. |
| **GIGI** | `g(dist)` | §2.2 | the bounded-support spatial mask; her support radius is an **output** of the fuse-to-5–15% construction, never a constant. |
| **PIA** | `PW(s)` | `rl-training-spec §1` | the projected winner, nim-sum over cartstate. Closed-form. |
| **SUE** | `SUCC(s)` | §1 | the succession under successive decrement of the leader. Closed-form. This is the anticipation that lets the policy be time-smooth. |
| **NIM** | `N_i(s)` | §1 | per-team standing in the hierarchy. Closed-form. |
| **DEE** | `diag(K)`, `K = L(I+L)⁻¹` | `dpp-mixing-and-overlay` | the differentiable DPP marginal-inclusion signal computed inside the canonical forward from learned appetite and Kay keys. One repulsion-shaped number per instrument. **Not** the determinant. |

## Frozen, deliberately

| name | what | why |
|---|---|---|
| **CEE** | the Xonotic C program / FPS layer | `stopgrad` by choice, not by nature. There *is* a gradient here; we decline to take it. Steering is skill-orthogonal: we bias where a bot commits, never how it aims. |

## The action-support measure

`action_mass[p,m]` is one exactly when instrument `m` belongs to participant `p`'s
team observation sigma-algebra and zero otherwise. It contains no opinion about whether
an observed action is useful: push, suppress, hunt, spawn timing, travel, and idle all
remain in support whenever their source row is observable. Sampling uses this as its
base measure. The policy supplies the learned density over that measure.

---

## How to use these names

- "Team metric is getting no gradient" — the same-team Gram matrix is not learned.
- "Rival metric is being fed hand-authored rows" — the §7 feature-engineering violation.
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
`d_x` raw per-player engine row · `d_z` per-instrument descriptor · `d_c` cell-localized entity slot ·
`d_sem` the eight closed-form PIA/SUE/NIM values.

**Learned widths** — every one **≥128, free above**; these are the knobs:
`d_β` belief · `d` query/key space · `d_v` behavioural value · `d_ir` the IR ·
`h` SwiGLU hidden (conventionally ≈ 8/3·d) · `r` same-team metric rank (≤ d) ·
`d_y`, `d_u` Dina's reduced state / action widths · `R` residual-feature rank ·
`F` expert hidden width. The release values are `R=2048`, `F=4096`, `E=32`, top-k 2.

| name | tensor | shape | the ≥128 side |
|---|---|---|---|
| **PHIL** `Φ` | belief projection (LEARNED — a constant Φ was a defect) | `(d_β, d_c)` | `d_β` |
| **QUINN** `W_q` | query | `(d, d_x + d_β + d_sem)` | `d`, and `d_β` within the input |
| **KAY** `W_k` | key | `(d, d_z)` | `d` |
| **VAL** `W_v` | behavioural value | `(d_v, d_z)` | `d_v` |
| **IR_QUERY** | query into IR | `(d_ir,d)` | both |
| **IR_VALUE** | value into IR | `(d_ir,d_v)` | both |
| **TEAM_METRIC** `A` | same-team metric factor, `M = AᵀA ∈ (d,d)` | `(r, d)` | `d`; `r` free ≤ `d` |
| **RIVAL_METRIC** | rival-row metric factor | `(r_e, d)` | `d` |
| **SCALE_IN** | relation IR into residual basis | `(R,d_ir)` | both |
| **SCALE_ROUTER** | top-k expert scores | `(E,R)` | `R` |
| **SCALE_EXPERTS** | routed SwiGLU matrices | `(E,R,F)`, `(E,F,R)` | `R`, `F` |
| **SCALE_PROBE** | nonlinear Gram-matrix readout | `(R)` | `R` |
| **SCALE_OUT** | residual basis into final IR | `(d_ir,R)` | both |
| **GIA** `w_gate` | SwiGLU gate | `(d_ir, h)` | both |
| **UMA** `w_up` | SwiGLU up | `(d_ir, h)` | both |
| **DOV** `w_down` | SwiGLU down → `dw/dt` | `(h, 1)` per instrument row | `h` |
| **ACTUATOR** | response control distribution | `(6,d_ir)` | `d_ir` |
| **WINNIE** `W_φ` | preservation probe | `(d_ir, 1)` | `d_ir` |
| **LOU** `L_ψ` | acquisition probe | `(d_ir, 1)` | `d_ir` |
| **VERA_WINNIE** `Ṽ_φ` | aux probe on the query → Winnie | `(d, 1)` | `d` |
| **VERA_LOU** `Ṽ_ψ` | aux probe on the query → Lou | `(d, 1)` | `d` |
| **DINA_STATE** | query into reduced state | `(d_y,d)` | both |
| **DINA_ACTION** | IR into reduced action | `(d_u,d_ir)` | both |
| **DINA_READOUT** | reduced delta into query | `(d,d_y)` | both |
| **DINA ×2** | `b(y)`, `A(y)` | two copies of `(d_y,)`, `(d_y,d_u)` | both |

The rule, once: **every matrix has at least one learned side ≥128, and the only small
numbers in the cast are the raw input widths the engine hands us.** A 16, a 21 or a 32
anywhere else is a bottleneck strangling the gradient.

Per-pair quantities are **computed, never stored**: `Quinn·Kay` is the per-(player,
instrument) score, a dot product of two ≥128d learned vectors. Storing an
`(l, m, small)` hand-authored row instead is both the §7 feature-engineering violation
and the reason a transition costs 374 KB.

## The deleted level-3 divergence

    d = 16          (IR was later widened to 128; d itself was not)
    d_β = 8         RELATION_WIDTH = 16     EDGE_WIDTH = 12    HIERARCHY_WIDTH = 8
    head.py:51  self.in_dim = 5 + RELATION_WIDTH   -> 21
    head.py:52  self.hidden = 32
    head.py:97  rms_norm(self.features(diag_k, appetite, relation), ...)

**The policy head is `21 → 32 → out`, fed by `diag_k`, `appetite` and the hand-authored
relation row — the matrix-fused IR never reaches it.** `IR_WIDTH = 128` feeds only Winnie and
Lou. So the same-team Gram matrix never touches an action; the 128-wide learned representation is
a side-channel the critics read while the behaviour is decided by 21 hand-made numbers.

That is SPEC §7 failing precisely where it points:

> if we're spending flops on a gram matrix that gram matrix better fucking end up in
> the IR consumed by subsequent probes

That implementation was deleted. The canonical composer now computes DEE from learned
appetite and Kay keys, sends the same-team and rival Gram matrices through the IR projections and GIA/UMA/DOV,
integrates `W`, and samples from the one-cadence-ahead state. PIA/SUE/NIM enter Quinn;
DINA's ensemble mean and disagreement add corrective and bounded exploratory score.
Consequently the actor loss has a path through every named policy group rather than
using the IR only as a critic side channel.

The policy equation is now, in one forward:

```
q = RMS(QUINN(XAN, BEA, PIA/SUE/NIM))          a = q KAY(ZED)^T / sqrt(d)
C = RIVAL_GRAM_MATRIX(q) + same_team * TEAM_GRAM_MATRIX(q)
H_pm = (C / team-or-rival row measure) IR_QUERY(q) + a_pm DEE_m IR_VALUE(VAL(ZED_m))
S = RMS(SCALE_IN(flatten(H)))                   e,g = topk(SCALE_ROUTER(S))
U = RMS(S + sum_e g_e SCALE_EXPERT_e(S))        G_R = U^T U / (l m)
H' = RMS(H + reshape(SCALE_OUT(RMS(U * (tanh(G_R) SCALE_PROBE)))))
dw/dt = DOV(silu(GIA(H')) * UMA(H'))            W' = tanh(W + delta dw/dt)
controls = ACTUATOR(H')
logits = (W' + c correction_DINA + e uncertainty_DINA) / TAU + log(action_mass)
H_p = RMS(sum_m exp(logpi_pm) H'_pm)
```

Configured scale rank, hidden width, and expert count determine the parameter tensors;
live participant and instrument rows determine the residual population. Those model
coordinates are reported with every run and are not a prescribed player-count operating
point. The actor gradient measure reaches `SCALE_IN`, `ROUTER`, both expert matrices,
`SCALE_PROBE`, and `SCALE_OUT`; the independent residual-fusion intervention measures
the contribution returned from that complete path.

The numerical path is part of the availability contract. ELLE integrates logits over
the observed action-support measure, so coordinates outside that measure do not enter
its square integral. DPP marginal inclusion is computed through the feature-side
covariance identity and a dimension-derived conjugate-gradient composition built from
the repository's Metal matrix products. No host solve or eigenspectrum surrogate leaves
the tensor execution graph. The learner clips the global gradient norm before AdamW and
publishes both the unclipped norm and clip threshold. Checkpoint realization measures
source, live, shape-difference, non-finite, composable, loaded, and restored masses, then
assigns the complete named parameter tree only when exact realization succeeds. The
complete preceding tree remains active after any exception. Optimizer moments follow the
same whole-tree finite-coordinate operation, so no compatible-leaf intersection can
create a hybrid state and no non-finite source can replace a working tree.

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

---

## PROJECT LAW — no duplication, no inlining, forever

> duplicated and inlined code is forbidden in projects which use policy gradient
> methods. modular nd central code only, forever.

This is not a style rule, and the reason is specific to policy gradients: the update
is weighted by `exp(logπ_target − logπ_behavior)`. If the sampling-time forward and the
training-time forward are two pieces of code, the two log-probs come from two functions
that can silently drift apart, and the importance ratio — and therefore the gradient —
becomes meaningless **with no error raised**. Duplication in a PG project is a
correctness failure that reports success.

Every duplication found in this repo did exactly that:
- `live_belief` re-inlined `featurize`'s belief pipeline (constant Φ, hardcoded radius,
  an extra normalisation) while the canonical functions sat dead — R20.
- the forward pass was inlined in both `estimator` and `train` — R10.
- `role_rewards` / `hierarchy_scores` were duplicated in `train` and
  `dominance_driver` — R10.
- `buffers.py`'s perception gate was dead because `live_belief` passed `True, True, 0.0`
  as literals — R20.

**Enforcement is structural, not vigilance:**
- `cast_header.py` — the only place a parameter exists. One definition per cast member.
- `strategy.py` — the only composition. It declares no parameter, fabricates no tensor,
  re-implements nothing, holds no state.
- `inputs.py` — assembles the computed chorus and contains no arithmetic beyond
  `asarray`; the moment it computes a feature it has become the thing §7 forbids.
- Everything else imports these. A second definition of any of them is a defect on
  sight, not a candidate to be compared against the first.

There are no variant implementations of the strategy program. There is one.
