# Policy-improvement report — strategy self-play REINFORCE

**Status:** `[MEASURED]`. Every number below is computed directly from the training log; none
are hand-authored. Companion intent doc: `rl-training-spec.md` (§2.1 ADVANTAGE, §5 why-not).

## 0. Provenance (the only source)

- **Source file:** `xonotic/solver/strat/runs/train_log.jsonl` (601 lines = 1 config header +
  600 per-iter records, `iter` 0→599). Read, not synthesized.
- **Run config (line 1, verbatim):**
  `{"iters":600,"batch":6,"n_steps":16,"k":2,"j":3,"l":4,"L":6,"gamma":0.95,"lr":0.003,`
  `"c_v":0.5,"c_aux":0.25,"c_reg":0.001,"weight_decay":0.0001,"delta":0.5,"temperature":1.0,`
  `"seed":0,"eval_games":16}`
- Wall clock: last record `sec=165.92` → the full 600-iter run took ~166 s (small self-play
  batch=6, right-sized for a shared host).

## 1. Did the policy improve? Yes — both outcome metrics rise with a strongly significant slope.

`win_rate` and `pw_control_frac` are the two outcome signals in the log (evaluated over
`eval_games=16` each iter). Both are policy-invariant *outcome* quantities — the true win
reward the shaping is supposed to steer toward, NOT the shaped advantage — so they are the
honest test of improvement.

| metric            | start mean (iter 0–19) | end mean (iter 580–599) | Δ        | OLS slope /iter | t-stat | Pearson r |
|-------------------|-----------------------:|------------------------:|---------:|----------------:|-------:|----------:|
| `win_rate`        | 0.28750                | 0.54688                 | +0.25938 | +5.269e-04      | +16.9  | +0.568    |
| `pw_control_frac` | 0.45059                | 0.65918                 | +0.20859 | +3.427e-04      | +22.2  | +0.672    |

100-iter binned trajectory (real, monotone-upward after a mid-run plateau):

```
iters   0- 99: win_rate=0.3494  pw_control_frac=0.4788
iters 100-199: win_rate=0.4088  pw_control_frac=0.5013
iters 200-299: win_rate=0.4056  pw_control_frac=0.4957
iters 300-399: win_rate=0.4581  pw_control_frac=0.5133
iters 400-499: win_rate=0.5775  pw_control_frac=0.5858
iters 500-599: win_rate=0.6162  pw_control_frac=0.6689
```

`win_rate` climbs from ~0.35 to ~0.62 (linear fit 0.31→0.63 over the run); `pw_control_frac`
from ~0.48 to ~0.67 (fit 0.44→0.64). t ≈ 17 and 22 on ~600 points: the upward trend is not
noise. The self-play win_rate crossing and holding above 0.5 against the shared-weight
opponent pool is the improvement signature.

## 2. Loss trends — value regression converges, policy-gradient stays balanced, reg peaks by design.

| loss term    | start mean (0–19) | end mean (580–599) | Δ        | slope /iter | note |
|--------------|------------------:|-------------------:|---------:|------------:|------|
| `loss_v`     | +0.13654          | +0.07299           | −0.06354 | −7.69e-05 (t=−8.0) | value head fitting the relative return (§3 `L_v`) |
| `loss_aux`   | +0.06939          | +0.00595           | −0.06344 | −3.79e-05 | query-projection `Ṽ`→`V` imitation collapses toward 0 (§3 `L_aux`) |
| `loss_pg`    | −0.00305          | −0.00406           | −0.00101 | −3.10e-08 (t≈0) | flat ~0 — REINFORCE term stays balanced, no runaway |
| `loss_reg`   | +0.18840          | +1.97672           | +1.79    | +3.14e-03 | **rises on purpose**: L2-toward-0 on strategy logits "peaks with training" (spec §3 `L_reg`) as sampling sharpens from broad to decisive |
| `loss_total` | +0.08276          | +0.03590           | −0.04685 | −4.48e-05 | net down |

The value loss halving (0.137→0.073, t=−8.0) is what the potential-based advantage needs: a
well-fit `V_φ` makes `Â = R + γV(s') − V(s)` a low-variance, near-unbiased signal. `loss_pg`
sitting at ≈0 with no drift is the expected REINFORCE behavior once advantage is centered
(the actor gets a mean-zero gradient signal, not a diverging one). The lone *rising* term,
`loss_reg`, is the spec-predicted logit-peaking (§3), not instability.

## 3. Mean advantage — centered near zero, exactly as the potential-based construction requires.

Raw from the log: overall mean `mean_advantage = -0.00007`, fraction of iters with positive
mean advantage = 0.445, first-half mean = -0.00019, second-half mean = +0.00006.

`Â` hugging 0 (|mean| < 1e-4, straddling zero) is the empirical fingerprint of the
**telescoping** property: around any cycle the shaping potential cancels, so the *average*
advantage carries no free progress/entry/level bonus to farm — only the local relative
denial+acquire signal survives. If the advantage were a hand-authored dense reward it would
show a persistent positive bias; it does not.

## 4. The policy-improvement property, as the theorem it is.

The advantage in `rl-training-spec.md` §2.1 is **potential-based reward shaping**:
`Â_u = R_u + γ·V_φ(s_{u+1}) − V_φ(s_u)`, i.e. the true reward `R` plus a shaping term
`F(s,s') = γΦ(s') − Φ(s)` with potential `Φ = V_φ`.

By **Ng–Harada–Russell (1999)**, potential-based shaping is *policy-invariant*: it leaves the
set of optimal policies of the underlying MDP unchanged (its total contribution telescopes to
0 around every trajectory/cycle). Therefore on-policy REINFORCE ascending
`∇_θ J = E[ Σ_u Â_u ⊙ ∇_θ log π_θ(a_u|·) ]` performs stochastic gradient ascent on the *same*
expected return as the true win reward `R` — the shaping only reduces gradient variance and
speeds credit assignment; it cannot move the optimum. So *if* the training ascends at all, it
ascends toward the win-reward optimum. Section 1 is the measured confirmation that it does
ascend: `win_rate` +0.26 and `pw_control_frac` +0.21 end-vs-start, both with t > 16 — real
movement of the invariant outcome metric, produced by a gradient on the shaped (variance-
reduced, mean-≈0) advantage of §3. Theorem says "same optimum"; the curve says "and we are
climbing toward it."

This is also the §5 anti-bistability payoff made concrete: because the shaping telescopes,
the run cannot farm entry (limit-cycle), level (stall), or monotone progress (race) — and the
measured `mean_advantage ≈ 0` with a still-rising `win_rate` is precisely "improving on the
relative objective without a farmable state bonus."

## 5. Caveats (honest).

- 16 eval games/iter → `win_rate` resolution is 1/16 = 0.0625; the per-iter series is noisy
  (Pearson r 0.57/0.67), which is why bin/window means and the t-stat carry the claim, not any
  single iter.
- A mid-run plateau (bins 100–299) precedes the strongest gains (bins 400–599); improvement is
  real but not strictly monotone iter-to-iter.
- `seed=0`, single run: this reports one trajectory, not a seed-averaged mean ± CI.

## 6. Bottom line.

Policy **improved**. Outcome metrics: `win_rate` 0.29→0.55 (fit 0.31→0.63, t=+16.9),
`pw_control_frac` 0.45→0.66 (fit 0.44→0.64, t=+22.2). Value regression converged
(`loss_v` 0.137→0.073), policy-gradient stayed balanced (`loss_pg`≈0), advantage stayed
centered (`mean_advantage`≈−7e-5). The potential-based advantage is policy-invariant
(Ng–Harada–Russell), so REINFORCE on it targets the true-win optimum — and the measured
curve shows it moving there. Source: `xonotic/solver/strat/runs/train_log.jsonl` (600 iters).
