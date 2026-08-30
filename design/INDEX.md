# design/ INDEX — concept → authoritative definition

One-screen map from each strategy-stack concept to the doc + section that *defines* it,
cross-checked against `rl-training-spec.md` (THE intent; where any doc disagrees, it
wins). Modality markers `[FIRM]`/`[OPEN]`/`[BUILD]` are the source docs'. Remaining
contradictions are listed at the bottom — **flagged, not fixed**.

Docs: `rl` = rl-training-spec.md · `pay` = payload-spec.md · `dpp` = dpp-mixing-and-overlay.md ·
`pbi` = playerbot-interface.md · `sl` = strategy-layers-and-modality.md · `qkv` = strategy-qkv.md ·
`cff` = cart-force-field.md · `pss` = payload-strategy-spec.md · `chk` = review-checklist.md.

## Concept → defining section

| Concept | Defining section (authoritative) | Restated / used in |
|---|---|---|
| **Two games** (Game-1 computed cartstate PW/SUCC · Game-2 frozen/stopgrad FPS C-prog + adapter) | `rl §0`, `rl §4` `[FIRM]` | `pay §0` (mechanics vs control) |
| **Winner reward/value W** preserve the path-to-victory region against rival perturbations; not progress/speed | `rl §2` `[FIRM]` | `rl §5` (state-targeting/disturbance control) |
| **Loser reward/value L** promote self and demote every rival until acquisition; not the symmetric dual of W | `rl §2` `[FIRM]` | `rl §3` (role-gated actor critic) |
| **Advantage** `A=r_role+γV_role(s')−V_role(s)`; TD baseline, not a shaping theorem | `rl §3` `[FIRM]` | `rl §6` (separate W/L regression) |
| **Policy gradient** `∇θJ=E[Σ A∇θ logπ]`, one shared `θ` across all counts | `rl §3` | `pay §5`, `pss §5` (REINFORCE, L2→0) |
| **Learned dynamics** local action-linear ensemble over reduced state/action rows; probing + reachability diagnostics | `rl §1`, `rl §5` `[FIRM]/[BUILD]` | Brown–Papadimitriou–Roughgarden control connection |
| **Count invariance** pointwise row encoders + invariant reductions + shared scalar action/value heads | `rl §4` `[FIRM]` | trained/held-out count distribution in `rl §6` |
| **Stopgrad boundary** observed features and empirical transitions are data; policy, W/L, adapter, and dynamics estimator learn | `rl §1`, `rl §3` `[FIRM]` | `pay` intro |
| **PW** projected winner = nim-sum over cartstate | `rl §1` `[FIRM]` | `pay §2.6` (nimber), `qkv §0` |
| **SUCC** backward induction over cart-decrements → ordered marginal-denial | `rl §1` `[FIRM]` | `pay §2.6` (explicit backward induction) |
| **Belief** `β_b` egocentric, occlusion-gated (frustum+LOS+2-Vcell), per-bot not per-team | `pay §2.2` `[FIRM][BUILD]` | `sl §2`, `rl §1` (b) |
| **DPP diag(K)** marginal-inclusion vector, `K=L(I+L)⁻¹` (NOT determinant) | `dpp §2` `[FIRM]` | `pay §2.4` |
| **Mixing head** `dw/dt=SwiGLU(RMSNorm([diag(K);b]))`, SwiGLU gate = regime switch | `dpp §2` `[FIRM]` | `pay §2.5` |
| **W_all / A_team / A_player** one shared-weight policy, per-team & per-player *activations* | `rl §0` `[FIRM]` | `dpp §3` (team-/bot-scale κ), `pay §3.2` (both scales integrate off-engine) |
| **Intercentrality overlay / "leader"=swing** `argmax intercentrality((I−aκ)⁻¹)`; "leader" ONLY as this readout | `dpp §3` (computable `[FIRM]`, consumption `[OPEN]`) | `pay §2.6`, `qkv §0` (κ=V Gm Vᵀ) |
| **Cadence / stability** `w += (dw/dt)·Δ` forward-Euler; Δ = chaos-threshold param | `dpp §4`, `pay §1`,`§3.2` `[FIRM]` | `pss §*` |
| **Control lever** `navigation_routerating(...,f,...)`, skill-orthogonal; scatter path | `pbi §0`,`§1`,`§4` (file:line) | `pay §4` |
| **Smooth cart force field** `ds/dt=f(pos,health,team)` C^∞ (soft membership/vitality/control) | `cff` (whole) | `pss §2` (hard law it smooths) |

## Remaining contradictions (flagged, NOT fixed)

1. **Bank reversibility — superseded but not retracted in `pss`.**
   `pss §1` ("Banks are reversible… retreat past a banked point loses the bank") and
   `pss §2`,`§3` (lines 65,76,187-188: "un-banking on downward crossings") directly
   contradict the monotone-score retraction in `sl §1` ("RETRACTS the un-banking idea…
   banked score does not un-bank"), `pay §0` ("banked score never does [reverse]"),
   `strategy-layers-notes.md`, and `chk A5` (owner "killed the un-banking"). `pss` is the
   older doc; per `rl`/`pay` intent, position reverses but **score is monotone**.

2. **Overlay consumption — `[FIRM]` in `pay` vs `[OPEN]` in `dpp`.**
   `pay §2.6` states explicit backward induction is "**required, not an optional
   overlay**" with "the nimber-leading team commits; trailing teams best-respond"
   (asymmetric / leader-first flow). `dpp §3`,`§5` hold exactly that asymmetric
   consumption **`[OPEN]`** and warn that naming the leader-commits/followers-best-respond
   dynamic "Stackelberg" now is the forbidden illocutionary move. Firm everywhere: SUCC
   as a *computed feature*. Contested: whether the velocity *flow* consumes it asymmetrically.

3. **Two groundings of "projected winner / leader / swing."**
   `rl §1` grounds PW/SUCC as a **deterministic nim-sum over raw cartstate** (Game-1,
   closed-form, stopgrad, NOT learned). `pay §2.6`/`dpp §3`/`qkv §0` ground the
   swing/leader as **argmax intercentrality over the coupling `κ = V Gm Vᵀ`** — an
   embedding Gram downstream of learned/frozen projections, not raw cartstate. Whether
   PW(nim-sum) and swing(intercentrality) are the same object computed two ways, or two
   distinct readouts, is unresolved; `rl`'s "computed over cartstate, stopgrad" and
   `κ`'s embedding provenance are in tension.

4. **"Value" term overload (terminology hazard, not a hard contradiction).**
   `sl §1` layer-3 is a "value-function-**LIKE**" operator (explicitly *not* a solved
   value function); `rl §2` defines the asymmetric `W` and `L` return estimators.
   Distinct objects sharing the word "value"; downstream code must not conflate.

5. **Observation source — doc intent vs current code (doc/code gap, noted for completeness).**
   `pay §2.2` and `pbi §0`,`§2.2` both intend perception-gated observations (frustum+LOS+
   2-Vcell) as the ONLY path enemy positions enter; the live gather feeds **omniscient**
   world state (`pbi §2.2`, the "riskiest gap"). The docs agree on intent — this is a
   `[BUILD]` gap against code, not a doc-vs-doc contradiction.
