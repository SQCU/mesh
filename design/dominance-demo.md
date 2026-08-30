# Cross-team information Pareto-dominates a team-state-only baseline

Demonstration that a strategy policy **with cross-team (all-to-all) information**
Pareto-dominates a **team-state-only** control policy on the relative deny/acquire
cart game (Game-1, `rl-training-spec.md` §1, §5) — *even when the baseline is a
genuine no-regret learner*. Two real artifacts back every number here:

- `xonotic/solver/strat/runs/aliasing_counterexample.json` — the analytic construction.
- `xonotic/solver/strat/runs/dominance_headtohead.json` — the empirical head-to-head.

Environment: `CartSim` (the fast, real Game-1 self-play env). Estimator forward passes
ran on the Apple mini (`mesh-mini`, `~/.venv-mesh`, mlx); the trained checkpoint
`runs/policy_online_v3.npz` loaded **27/27 tensors**. Reproduce with:

```
ssh mesh-mini 'cd ~/dox/mesh/xonotic && ~/.venv-mesh/bin/python -m solver.strat.dominance_driver'
```

The two policies:

- **All-to-all** — `estimator.StrategyEstimator`, which conditions on the deterministic
  cross-team hierarchy features (`PW`, `SUCC` succession/denial, per-team nimbers, and the
  rival-control `relation` tensor) built in `estimator.state_from_cartsim` +
  `game.succession`.
- **Team-only** — `baseline_teamonly.TeamOnlyBaseline`: a count-invariant policy that maps
  **only the acting team's own aggregate** (own cart depths/banked, own player count)
  through a linear projection to per-player instrument weights, mixed by a **Hedge /
  multiplicative-weights no-regret update** over its own action set (four own-team-only
  strategy templates). It is *structurally* blind to cross-team configuration: a non-own
  cart collapses to a single "not mine" token with no rival identity or depth.

---

## 1. Analytic counterexample — the partition provably mis-acts

Construction (`k=3` teams, `j=3` carts, `l=6` players, `L=8`, focus = team 0, a loser).
Team 0 controls **only cart 0 at depth 2** in both states; the two rival carts are
**swapped** between the states:

| state | positions `[c0,c1,c2]` | control | nimbers | PW | SUCC |
|-------|------------------------|---------|---------|----|------|
| **A** | `[2, 1, 3]` | `[t0, t1, t2]` | `{t0:2, t1:1, t2:3}` | **team 2** | `[[2,0]]` |
| **B** | `[2, 3, 1]` | `[t0, t1, t2]` | `{t0:2, t1:3, t2:1}` | **team 1** | `[[1,0]]` |

- **`own_view_identical: true`** — team 0's own aggregate + own-relative cart descriptors
  are byte-identical across A and B.
- **`team_only_instrument_distribution.identical: true`** — the baseline's marginal
  instrument distribution for a team-0 player is identical on A and B
  (`[0.541, 0, 0, 0, 0.099, 0.099, 0.261]` over
  `[push0, push1, push2, suppress0, suppress1, suppress2, idle]`). By construction it
  **cannot** distinguish suppress-cart1 from suppress-cart2 — both are "not mine."
- **All-to-all emits DIFFERENT, correct actions.** Correct denial target = the leader's
  deepest cart: **cart 2 in A, cart 1 in B**. Both all-to-all policies split them:
  - trained `StrategyEstimator` argmax: `A → suppress cart2`, `B → suppress cart1`
    (`estimator_differs_A_vs_B: true`);
  - `greedy_deny` reference: identical, and provably correct (`greedy_differs_A_vs_B: true`).

**One-step relative objective (PW-denial, §5).** Under a controlled protocol (rivals
abandon; team 0's two players suppress the chosen cart), suppressing the **leader's** cart
dislodges the projected winner (`PW → None`); suppressing the other cart leaves it:

| | suppress cart1 | suppress cart2 |
|---|---|---|
| **A** (PW=t2) | PW 2→2, denied=0 | PW 2→**None**, denied=1 |
| **B** (PW=t1) | PW 1→**None**, denied=1 | PW 1→1, denied=0 |

**Forced regret.** Because the team-only policy must use one target-distribution for both
aliased states, any fixed blind target denies the projected winner in **exactly one** of
the two states:

- `team_only_forced_denial_failures_out_of_2 = 1` → **min failure rate 0.5**;
- `all_to_all_failure_rate = 0.0` (denies in both).

This is irreducible: it holds for *every* Hedge weight vector, so **no-regret learning
cannot close it** — the regret bound is relative to the best action *within the team-only
partition*, and the best in-partition action still fails on one of the pair.

**Honest subtlety (in the artifact).** The *shaped scalar* reward `role_rewards` is
**identical (0.0624)** for all four (state, target) cells: it scores aggregate rival
strength, not rival identity, so it is itself partition-symmetric. The aliasing cost is
therefore carried by the **PW-denial / possession objective (§5)**, not by the shaping
term. This is why the head-to-head below reads the cost off terminal/PW-control outcomes.

---

## 2. Empirical head-to-head — 60 matches/shape, 20 steps

Both policies play the focus team (team 0) against a **common all-to-all opponent**
(`greedy_deny` controls every non-focus team). One shared `TeamOnlyBaseline` (no-regret
state carried across all shapes). Metrics: terminal win, PW-control fraction, winner
retention (hold PW next step, under a 15% exogenous rival perturbation regime), loser
acquisition. Δ = estimator mean − baseline mean; Wilson-95 is on P(estimator ≥ baseline)
per game.

| shape | held-out | term-win Δ | PW-ctrl Δ | retention Δ | acquisition Δ | Pareto? |
|-------|:---:|---:|---:|---:|---:|:---:|
| `[2,3,4]` | | +0.517 | +0.466 | +0.336 | +0.615 | **strict** |
| `[3,3,6]` | | +0.900 | +0.095 | +0.000 | +0.100 | strict\* |
| `[3,4,6]` | | +0.767 | +0.572 | +0.479 | +0.614 | **strict** |
| `[4,4,8]` | | +0.000 | +0.050 | +0.000 | +0.053 | strict\* |
| `[3,5,7]` | ✓ | +0.733 | +0.592 | +0.660 | +0.377 | **strict** |
| `[4,6,9]` | ✓ | **−0.017** | +0.008 | **−0.146** | +0.025 | **NO** |
| `[5,4,10]`| ✓ | +0.000 | +0.054 | +0.000 | +0.058 | strict\* |

\* dominance carried by PW-control + acquisition; terminal-win/retention are `0` for
**both** policies at these shapes (no team secures/holds PW within 20 steps → degenerate
tie, `retention_trials = 0`). Weak (non-strict) on those two dims.

Per-game win counts are lopsided where dominance is strict (e.g. `[3,4,6]`: PW-control
60/0, retention 60/0, acquisition 60/0; Wilson-95 for P(est≥base) `[0.94, 1.00]`). The
baseline's PW-control fraction is **≤ 0.10 on every shape** and terminal win **≤ 0.10**;
the blind policy essentially never secures the path to victory against an all-to-all
opponent.

**Aliasing frequency in real matches.** The baseline mis-targets the true projected
winner's cart on a rising fraction of its loser-steps as cross-team ambiguity grows:
`alias_mis_target_rate` = 0.14 (`[2,3,4]`) → 0.38 (`[4,4,8]`) → **0.67 (`[4,6,9]`)**. The
shaped-reward "cost" of these mis-targets is near-zero/negative — the same
partition-symmetry as §1 — confirming the cost lives in the objective, not the shaping.

---

## 3. Honest verdict

**Not** global strict Pareto-dominance. The result is:

- **Strict Pareto-dominance on 6 of 7 shapes** (including 2 of 3 held-out shapes), where
  strict means estimator ≥ baseline on all four dimensions and strictly greater on at
  least one. On 3 of those the dominance is *weak on terminal-win/retention* (both zero)
  and strict only on PW-control + acquisition.
- **Dominance-on-average overall**: every dimension's shape-win count favors the estimator
  (PW-control 7/7, acquisition 7/7, terminal-win 4/7, retention 3/7; the estimator loses a
  dimension on only 1 shape).
- **One clean failure**: the largest held-out extrapolation `[4,6,9]` (9 players, 6 carts —
  the furthest from the training range) **breaks Pareto**: the trained estimator loses on
  terminal-win (−0.017) and retention (−0.146). At this scale the estimator's *learned*
  generalization degrades even though the baseline's aliasing is *worst* here (0.67). This
  is a limit of the current checkpoint's extrapolation, not of the information argument.

So: **the analytic counterexample is unconditional** (the team-only partition provably
cannot match the all-to-all denial objective on aliased states, for any no-regret
weights); **the empirical dominance is strong but not universal at this training scale**,
holding on 6/7 shapes and on-average across all, with a real regression at the largest
held-out shape.

### Notes required for honesty

**(a) No unverifiable citation.** I could **not** verify arXiv `2603.03173` and do not cite
it. The argument rests on two things I *can* state cleanly: **Blackwell informativeness**
(the all-to-all observation is a refinement of the team-only observation — team 0's own
view is a deterministic garbling of the global cartstate — so no team-only decision rule
can outperform the best all-to-all rule in expectation on a payoff that depends on the
discarded coordinates), and **no-regret is partition-relative** (Hedge/MW guarantees
vanishing regret *against the best fixed action in its own action set*; when the action
set is measurable only w.r.t. the coarse partition, the comparator itself is
partition-limited, so the guarantee does not reach across an aliased pair). §1 realizes
both concretely.

**(b) The cross-team coupling here is deterministic, not a learned Gram.** The estimator's
cross-team information enters as **deterministic hierarchy/`SUCC` features**
(`game.succession`, per-team nimbers, the rival-control `relation` tensor) — *not* a
learned O(n²) all-pairs Gram/attention over rival rows. So this demonstrates the value of
**cross-team information**, cleanly: even fixed, hand-computed cross-team features
de-alias the counterexample and win the head-to-head, while the team-only policy cannot
see them at all.

*Hypothesis (marked as such, not demonstrated here):* a **learned** cross-team mixing (an
O(n²) rival-to-rival Gram feeding the appetite/denial head, replacing the fixed `SUCC`
reduction) would be expected to **widen** the margin — the deterministic `SUCC` collapses
the succession to a single scalar denial budget per team, discarding higher-order rival
interaction structure (e.g. multi-rival gang/pre-empt trade-offs) that a learned mixing
could exploit, and would plausibly also *repair* the `[4,6,9]` extrapolation failure by
learning a count-robust rival-interaction kernel rather than relying on the current
checkpoint's generalization. This is a conjecture; it is not shown by these two artifacts.
