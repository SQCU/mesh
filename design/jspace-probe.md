# j-space probe — is the final IR semantically decodable?

**Verdict: NO.** There is no semantically-rich j-space right now. Every semantic
feature that a linear probe can read out of the trained IR is read *equally well*
out of a random-init IR and (with one caveat, below) out of a random projection of
the same inputs. The trained checkpoint's IR carries no decodable semantics beyond
what the 4-dimensional cart-game input already contains, and it carries **zero**
per-player semantics (health / armor / ammo / weapon held / position) because those
features were never in the model's input on this run.

Artifacts: `xonotic/solver/strat/runs/jspace_probe.json` (numbers),
`xonotic/solver/strat/runs/jspace_probe.py` (the measurement, runs on `mesh-mini`).

## Method (no simulation)

Per SPEC §13, `cartsim` was not imported and nothing here is re-simulated. Inputs are
the **228 real lines** of `runs/game2_train.jsonl` (`environment: game2_server`,
`mode: online_train`), giving **3150 real player-rows**. The IR is computed on exactly
`estimator.strategy_forward`'s path:

    R0 = qkv.query([x ; beta]);  Rp = relattn.RelationalEncoder(R0, edge_features(...))
    value_rows = [Rp ; team_pool(Rp)[team] ; stats(4) ; hierarchy(8) ; behavior_mix(16)]

Probes: ridge least-squares (regression, R²) and one-vs-rest ridge classification
(accuracy vs majority baseline), **split by line, 60/40**, so no test row shares a
server tick with a training row. Controls: (a) random Gaussian projection of the same
raw `[x;beta]` inputs at the same width (16d and 40d); (b) shuffled labels.

## Real IR width

| quantity | value |
|---|---|
| trained row width `d` (`qkv.W_q` is `(16, 24)`) | **16** |
| value-head input width (`value.winner.up.weight` is `(32, 60)`) | **60** |
| dims of that 60 actually reconstructible from the real log | **40** (`Rp` 16 + `team_pool` 16 + `hierarchy` 8) |
| SPEC §8 requirement (*"under 128d? maybe you were slippin"*) | **≥128** |
| local rewrite (`estimator.IR_WIDTH`, GramSwiGLU tree) | 128, **but no checkpoint exists for it** |
| **rank of the real input matrix `[x;beta]` over all 3150 rows** | **4** |

The value heads are also **not linear probes**: `value.RoleValueHead` is
`Linear(60→32) → silu → Linear(32→1)`, a 2-layer MLP. SPEC §5 says the value estimators
are "trained as linear probes upon the final IR". That is not what the code does.

## Probe scores (test split, 1278 rows)

| target | Rp trained (16d) | Rp random-init | value_rows trained (40d) | ctrl: rand-proj 16d | ctrl: shuffled labels | raw inputs (24d) |
|---|---|---|---|---|---|---|
| own nimber | R²=1.0 | **1.0** | 1.0 | **1.0** | −0.010 | 1.0 |
| max rival nimber | 1.0 | **1.0** | 1.0 | **1.0** | −0.001 | 1.0 |
| hierarchy margin | 1.0 | **1.0** | 1.0 | **1.0** | −0.010 | 1.0 |
| total cart depth | 1.0 | **1.0** | 1.0 | 0.9955 | −0.005 | 0.9955 |
| cart-0 depth | 1.0 | **1.0** | 1.0 | 0.9955 | −0.004 | 0.9955 |
| # carts controlled | 0.9563 | **0.9566** | 0.9563 | 0.6788 | −0.003 | 0.6788 |
| SUCC denial budget | 1.0 | **1.0** | 1.0 | **1.0** | −0.006 | 1.0 |
| instrument `gain` | 0.2878 | **0.2864** | 0.2880 | 0.0815 | −0.001 | 0.0815 |
| instrument `lane` | 0.2126 | **0.2126** | 0.2126 | 0.1668 | −0.006 | 0.1668 |
| chosen-action logp | 0.1243 | **0.1238** | 0.1248 | 0.0048 | −0.001 | 0.0048 |
| **is my team PW** | acc=**1.00** (maj .948) | **1.00** | 1.00 | **1.00** | 0.952 | 1.00 |
| **which team is PW** | acc=**1.00** (maj .793) | **1.00** | 1.00 | **1.00** | 0.804 | 1.00 |
| **instrument kind (7-way)** | acc=0.6135 (maj .416) | **0.6150** | 0.6150 | 0.5751 | 0.414 | 0.5751 |
| my team id (5-way) | acc=0.4710 (maj .239) | **0.4710** | 0.4710 | 0.4710 | 0.235 | 0.4710 |

Reading it:

* **Control (b), shuffled labels, is passed everywhere** — R² ≈ 0, accuracy = majority.
  The probes are honest; the fit is not overfitting noise.
* **Control (a), random projection, is NOT beaten by the trained IR on any target that
  matters.** PW, nimber, hierarchy margin, denial budget all hit R²=1.0 / acc=1.00 on a
  *random* projection of the same inputs, because those quantities are literally input
  columns (`x[0,1,3,4,5]`, `hierarchy[0..7]`). Decoding them is tautological, not evidence
  of a j-space.
* The four places the IR *does* beat the raw/random-projection control —
  `n_controlled` 0.956 vs 0.679, `gain` 0.288 vs 0.082, `lane` 0.213 vs 0.167, `logp`
  0.124 vs 0.005 — are **identical for the trained and the random-init encoder**
  (0.9563 vs 0.9566; 0.2878 vs 0.2864; 0.2126 vs 0.2126; 0.1243 vs 0.1238). That gain is
  the encoder's *nonlinearity* (silu + softmax attention) manufacturing nonlinear
  functions of the same rank-4 input, not learned semantics.

## Trained vs random-init — does the value gradient ground the projections?

On this evidence, **no measurable grounding**. The trained checkpoint `policy_online_v3.npz`
loaded 23 of its 39 tensors into the checkpoint-era architecture (all of `qkv.W_q/W_k/W_v`,
both `relattn` layers, both value heads — see `checkpoint_compat.matched` in the JSON), so
the entire IR path is the genuinely trained one. Its probe scores match a fresh random-init
model to 3 decimal places on 14 of 14 targets. The only difference the training left is
**numerical spread**, not decodable content:

    rank(Rp trained) = 14      rank(Rp random-init) = 9      rank(raw inputs) = 4

i.e. training un-collapsed the encoder's row space from 9 to 14 effective dimensions, but
put nothing new and linearly readable into it.

SPEC §7 asks that "the value gradient BETTER put trivially semantically measurable features
into the learned projections". Measured: it did not, on this run.

## Exact blockers in the real data

`runs/game2_train.jsonl` logs only `carts` (depth/ctrl/speed/progress), `PW`, `SUCC`,
aggregate `belief` stats, `instrument_counts`, and per-row `assignments`
(team/kind/action/subject/target/gain/lane/commit/spawn/logp). It does **not** log the
per-player observation rows. So these requested targets are **not measurable at all**:

* health, armor, ammo, **weapon bitmask / "holds a rocket launcher"**, position, velocity,
  distance-to-nearest-cart, per-player `POWER`/`TSS`/`ALIVE`/`CONTROL`;
* the per-player belief `beta` (only aggregate `mean_norm` is logged);
* the instrument descriptors `z` and the `(player × instrument × 16)` relation rows.

Consequently `x[2]` (banked) and `x[8:16]` and all of `beta` were zero in this measurement,
and 20 of the 60 value-row dims (`stats(4)` + `behavior_mix(16)`, both downstream of `z`)
could not be reconstructed. `runs/cartserver_telemetry.jsonl` (225 lines) is strictly poorer
— it has no `assignments`, only a `scatter` of edicts.

**This is not only a logging gap: it is the finding.** SPEC §3 requires the policy to
integrate "THE HEALTH OF ALL PLAYERBOTS AND THEIR AMMO COUNTS AND GUNS". On the Game-2 run
that produced this checkpoint the model's own input `x` carried a **rank-4** signal — the
cart-game scalars and nothing else. A j-space cannot contain "who is holding a rocket
launcher" when the rocket launcher never entered the matmul.

Two further blockers found while measuring:

1. **All checkpoints on disk are architecture-stale w.r.t. the local tree.** The Mac's
   `xonotic/solver/strat/` was rewritten (GramSwiGLU, `IR_WIDTH=128`, `X_WIDTH=48`);
   `policy_online_v3.npz`, `policy_ckpt_v3.npz`, `game2_online.npz` and `policy_ckpt.npz`
   are all from the `relattn` / `d=16` / `X_WIDTH=16` architecture that still lives on
   `mesh-mini`. There is **no trained checkpoint for the 128d rewrite**, so the measurement
   was run against the checkpoint-era tree on the mini — the only place where trained
   weights and real data agree. Worse, `online.OnlineLearner._load_full` calls
   `load_weights(..., strict=False)`, which will *silently* resume the 128d rewrite from a
   16d checkpoint with almost nothing restored.
2. **The CGT value never resolved.** `game_value.kind == "unresolved"`,
   `nimber == null`, `reason == "incomplete option graph"` on **228 of 228** lines. The
   combinatorial-game value that is supposed to price states was never computed during this
   run, so it could not be used as a probe target either.

## Is width the limiting factor?

**No — not yet.** The input is rank 4 and the IR is rank 14. A 16-dimensional row is already
more than wide enough to hold everything the input contains. Widening `d` from 16 to 128
without also feeding `x[8:16]` (health/armor/ammo/weapons) and a real `beta` will produce a
128-dimensional embedding of a 4-dimensional signal. SPEC §8's ≥128d is necessary for the
*intended* input (48 engine dims + 8 belief dims + a 3150-row Gram) and should be honored,
but width is the **second** blocker; the first is that the semantic inputs are absent.

## Is a semantically-poor j-space possible if the linear algebra is right?

**Yes — and this is the honest analytic answer, independent of the measurement.**

What the value gradient guarantees: the value heads read the IR, so gradient descent on the
value loss pushes the IR to be linearly separable *along the directions that reduce value
error*. If the value target is scalar (here: `role_rewards`, one number per player per step),
that constrains **at most one direction per head** — two, for the winner/loser pair. Nothing
in the objective asks the remaining `d − 2` directions to mean anything. There is no term
anywhere in `online.OnlineLearner.update`'s loss (`actor + 0.5·winner_loss + 0.5·loser_loss
+ 0.25·dynamics_value + 1e-3·regularization`) that rewards decodability of health, ammo, or
weapon-held.

So the guarantee is: **value-relevant directions, not arbitrary semantics.** A feature only
lands in the j-space if (i) it is present in the input, and (ii) it changes the value target
in a way the reward can see. "Who is holding a rocket launcher" satisfies neither on this
run: (i) is zeroed out of `x`, and (ii) `role_rewards` is computed purely from cart nimbers
and PW transitions, so weapon state has literally zero gradient path to the loss. A perfectly
correct linear algebra implementation will therefore still yield a semantically-poor IR here.
That is not a bug in the algebra; it is the objective and the featurization.

The fix implied by SPEC §3 and §7 is not a wider `d`. It is: log and feed the per-player
engine rows into `x`, and make the reward sensitive to them — otherwise the value gradient
has nothing to ground.

## Reproduce

    scp xonotic/solver/strat/runs/jspace_probe.py mesh-mini:/tmp/
    ssh mesh-mini '~/.venv-mesh/bin/python /tmp/jspace_probe.py'
