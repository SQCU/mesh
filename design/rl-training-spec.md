# RL training specification — loss, reward, value, advantage, policy gradient

Authoritative definitions for the strategy learning stack. Where this and other docs
disagree, this doc is intent. Companion: `payload-spec.md` (data/exec flow),
`dpp-mixing-and-overlay.md` (mixing head), `playerbot-interface.md` (levers). Modality:
`[FIRM]`, `[OPEN]`, `[BUILD]` = not-yet-coded.

## 0. Spine (corrected, authoritative) `[FIRM]`

- **Two games.** (1) The *strategy game* in cartstate notation — the projected winner
  and its backward-induction succession are **closed-form over cartstate** (Game 1,
  computed, not learned). (2) The *Xonotic FPS* in point-and-shoot notation — the C
  program, **frozen / `stopgrad` / out of scope** (Game 2). A parametric **adapter**
  bridges them; the adapter and `W_all` are the only trainable surfaces touching the
  FPS boundary.
- **Objective is RELATIVE.** Take the path-to-victory away from whoever holds it and
  acquire it. NOT absolute progress (→ race), NOT entry (→ oscillation), NOT level
  (→ stall). See §5.
- **Policy = one shared-weight multi-agent policy.** Weights `W_all`; per-team
  activations `A_team`, per-player activations `A_player`. Emits a per-player strategy
  **velocity** `dw/dt` over instruments, sampled, with log-probs. Only `W_all` (+ value
  head + adapter) learns; teams/players are activations, not separate learners.

## 1. Computed, deterministic, Game-1 (NOT learned) `[FIRM]`

- `s` = cartstate (cart depths + control) — a **guaranteed member of the global feature
  vector** (if the emit path doesn't guarantee it, that is a `[BUILD]` bug).
- `b` = belief (egocentric, occlusion-gated observation integration; see payload-spec §2.2).
- `PW(s)` = projected winner: closed-form, the nim-sum "who wins if nothing else
  changes". One cart at d:2 beats two carts at d:1 (1⊕1 = 0).
- `SUCC(s)` = backward induction over cartstate: recompute `PW` under successive
  decrements of the current leader's carts → ordered `[(team, marginal_denial_value)]`.
  Deterministic. It is what makes the policy **anticipatory / time-smooth** — the whole
  succession is folded into one immediate-frame allocation (gang the leader only to its
  marginal need, pre-empt the next-in-line, etc.) instead of reacting to each flip after
  it happens. `PW`/`SUCC` are FEATURES the policy and value read; they are not learned.
  What *is* learned is only the Game-2 **realization** (can an allocation actually
  decrement the leader through the frozen FPS).

## 2. The four definitions

### 2.1 Algorithms register

- **REWARD** `R_u ∈ R^l` — per-player (NOT scalar), verifiable from scoreboard+cartstate,
  RELATIVE (deny+acquire the path-to-victory slot). Excludes monotone-progress, entry,
  level (§5). Exact raw expression `[OPEN]`; the operative dense signal is the advantage
  (§2.1 ADVANTAGE), not a hand-authored per-tick reward. **CODE: none.**
- **VALUE** `V_φ(s,b,SUCC) ∈ R^l` — per-player, a linear projection on the strategy
  estimator's final intermediate. Auxiliary `Ṽ` = linear projection on the query,
  regressed to `V`. Inputs include `SUCC` ⇒ anticipatory. **CODE: none.**
- **ADVANTAGE** `Â_u = R_u + γ·V_φ(s_{u+1}) − V_φ(s_u)` — per-player, potential-based
  (TD). Around any cycle the shaping **telescopes to 0** (kills oscillation and stall
  farming) and is **policy-invariant**; this is the anti-bistability mechanism, and the
  reason the value head exists. **CODE: none.**
- **POLICY GRADIENT** `∇_θ J = E_rollout[ Σ_u ( Â_u ⊙ ∇_θ log π_θ(a_u | s_u,b_u,SUCC_u) ) ]`
  — updates `θ = W_all` only; `s,b,SUCC,PW` and the FPS C-program are `stopgrad`.
  Per-player `Â` weights per-player log-probs (`A_player`), summed into shared `W_all`.
  On-policy REINFORCE over self-play rollouts; per-step `(state, activations, action,
  logπ)` buffered to the crediting horizon and replayed. **CODE: none.**

### 2.2 Normalized register

- `PW(s), SUCC(s)` = closed-form over cartstate (Game-1, computed, ¬learned); realization = learned (Game-2, frozen-FPS).
- reward: `R∈R^l` per-player, verifiable(scoreboard+cartstate), RELATIVE(deny+acquire PW-slot); ¬progress(race) ¬entry(oscillate) ¬level(stall); operative=advantage, raw-`R` OPEN. code=∅.
- value: `V_φ∈R^l` per-player=linproj(final-intermediate); inputs⊇`SUCC`⇒anticipatory; aux `Ṽ`=linproj(query)≈`V`. code=∅.
- advantage: `Â=R+γV(s')−V(s)` per-player, potential/TD ⇒ cycles→0 (¬oscillate ¬stall), policy-invariant. code=∅.
- policy-grad: `∇θJ=E_roll[Σ Â⊙∇θ logπθ(a|s,b,SUCC)]`; `θ=W_all` only; `s,b,SUCC,PW,C-prog`=stopgrad; per-player `Â`→`A_player`→shared `W_all`; REINFORCE/self-play; buffer(state,activation,a,logπ)→horizon→replay. code=∅.

## 3. Loss `[FIRM]`

`L = L_pg + c_v·L_v + c_aux·L_aux + c_reg·L_reg`
- `L_pg = −E[ Σ_u Â_u.detach() ⊙ log π_θ(a_u|·) ]` (policy gradient; advantage stop-grad into the actor).
- `L_v = E[ ‖V_φ − return(R)‖² ]` per-player value regression to the (relative) return.
- `L_aux = E[ ‖Ṽ − V_φ.detach()‖² ]` query-projection imitates the value.
- `L_reg` = L2-toward-0 on the strategy logits (broad sampling untrained; peaks with training) + weight decay.

## 4. Computed vs learned vs frozen `[FIRM]`

- computed (deterministic): `s, b, PW, SUCC`.
- learned: `W_all` (with `A_team`/`A_player` activations), the value head, `Ṽ`, the adapter.
- frozen / stopgrad: the FPS C program (never trained — deliberate, not intrinsic), and the computed features above.

## 5. Why not the simpler rewards `[FIRM]`

- entry (reward the flip *to* winning) → competitive limit cycle: teams trade the lead to farm re-entries.
- level (reward being the current winner) → noncompetitive stall: the winner prolongs the win-state, never delivers.
- monotone progress (reward banked delivery) → race: rewards your own carts regardless of denial; fastest team wins, no strategy.
All three reward a reversible/maintainable STATE. The cure is the potential-based
advantage (§2.1) over a value grounded in the relative terminal outcome, plus the
relative objective — un-farmable by cycle, duration, or self-progress.

## 6. Open / build

- `[OPEN]` exact raw `R` expression (constraints in §2.1/§5); crediting horizon; rollout segmentation; `γ`; `c_*`.
- `[BUILD]` the strategy estimator must expose `logπ`, per-player activations, and the final intermediate for training; the rollout loop, replay buffer, reward/value/advantage/policy-gradient — none exist as code.
