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
| **NORM** | RMSNorm weight | `(in_dim,)` | scales the head's input. |
| **GIA / UMA / DOV** | `w_gate` / `w_up` / `w_down` | `(in,h)`, `(in,h)`, `(h,out)` | the SwiGLU trio. **Gia is the regime switch** — she decides diversify-vs-pile-on from the DPP signal and shared appetite. Dov emits `dw/dt`. |
| **WINNIE** | `W_φ` | `(d_ir, 1)` | the **preservation** value probe. Linear, on the final IR. Trained only on rows whose team holds the path. |
| **LOU** | `L_ψ` | `(d_ir, 1)` | the **acquisition** value probe. Linear, on the final IR. Trained only on rows whose team does not. Not Winnie's sign-flip — different target, deliberately. |
| **VERA** | `Ṽ` | `(d_q, 1)` | the auxiliary probe on the query, regressed toward Winnie/Lou. Grounds the early projection. |
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
| **PHIL** | `Φ` | §2.2 | the low-rank cell projection inside Bea. |
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
