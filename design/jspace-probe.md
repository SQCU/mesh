# j-space probe — is the final IR semantically decodable?

**Verdict: PARTIAL.** There *is* a j-space now, and it is measurably richer than
anything R19 or R24 could see: the operator turns a **rank-33** real input into a
**full-rank-128** IR, every non-tautological game quantity is linearly readable out
of it, and it beats a 128d random projection of the same inputs on **every target the
raw inputs do not already saturate, at every training budget measured**. Two things
are still not established. First, that the *value gradient* is what builds the space:
the untrained encoder of the same architecture scores within ~0.01 of the trained one
on most targets and **better** on the cart-control ones, so the separation is the
architecture (Gram + SwiGLU at 128d), not the learning. Second, R24's collapse is
**deferred and softened, not eliminated**: 190x more data buys a flat curve out to
16,200 updates — where R24 had already fallen to `speed` R^2 = -0.98 — but by 32,400
updates the cart-control probes give up about 0.14 R^2 while the value loss keeps
falling. So the guarantee SPEC 6 asks for — "a big embedded vector which allows
learned linear projections to map something which can in theory be used to compute
value to policy actions over seemingly irrelevant errata" — is met, and met by the
*architecture*, with the *training* adding a little and eventually taking a little
back.

Artifacts: `xonotic/solver/strat/runs/jspace_probe.json` (numbers),
`xonotic/solver/strat/runs/jspace_probe.py` (the measurement, runs on `mesh-mini`),
`xonotic/solver/strat/runs/vcell_link_band.json` (the V-cell A/B below).

## The data, and where it came from

R25 recorded a real run of **993 strategy ticks / 11,909 per-player rows** and named
`obs_rows.jsonl` as its product. That file was written into a throwaway location, but
**the server log it was made from is preserved** — `run_final.log`, 11,940,227 bytes,
2026-08-30 19:56, carrying `[PLCPUB]/[PLCOBS]/[PLCCART]/[PLCEVT]` and the pool proof
`[PLCPOOL] evt_base 5227 rows 256 contiguous 1 / cart_base 5483 rows 4 contiguous 1
obs_base 1 rows 16`. Re-running the shipped `measure.py rows` over it reproduces
R25's numbers exactly:

    ticks_logged 999   ticks_written 993   player_rows 11909
    ID/TEAM/HEALTH/AMMO/POS/WEAPONS/CELL/NCART/NCART_D/TSS/ALIVE/CONTROL  993/993
    ARMOR 913/993      VEL 918/993

so the measurement below is on the data R25 describes, regenerated rather than
recovered. Nothing is simulated (SPEC 13); `cartsim` is not imported and does not
exist. A second real log, `run_links.log` (519 publishes, the run that carries the new
kind-4 waypoint links), is used for the V-cell measurement in the last section.

**What the probe consumes.** One JSONL record per tick with the engine's own
observation rows, cart rows and perception events, plus the instrument descriptors
`z` and relation rows built by the same `build_instruments` the live operator calls.
`beta` is recomputed by replaying the same `LiveBelief` over the same event stream.
The state is then built by `estimator.state_from_runtime` — the responder's own
constructor — so no column is reconstructed by hand. This is the specific thing R19
got wrong: `runs/jspace_probe.py:47` used to hardcode `x[8:16] = 0`, which is where
its "rank 4" came from (R25's correction).

**Training** is a replay of the real transitions through the *shipped*
`OnlineLearner.observe`/`flush` loop with its replay ring — not a bespoke optimizer.
Probes are ridge least-squares (R^2) and one-vs-rest ridge classification (accuracy
vs the test-set majority), **split by tick, 60/40**, 7,140 train / 4,769 test rows.

**Tautology tagging.** The IR is a function of `(x, beta)` only — `z` and the relation
rows reach the mixing head, never the encoder. Each target is therefore tagged
`in x`: *yes* means it is literally a column of the operator's input and decoding it
measures preservation; *no* means it is not in the input at all and decoding it
measures induced structure. The verdict is read off the `no` rows.

## Real IR width and rank

| quantity | R19 (228 lines) | R24 (62 lines) | **now (993 ticks)** |
|---|---|---|---|
| IR width `d` | 16 | 128 | **128** |
| rank of the real input `[x ; beta]` | 4 | 33 | **33** (`x` alone 25, of 56 dims) |
| player rows | 3,150 | 744 | **11,909** |
| numerical rank of the trained IR | 14 | — | **128** (full) |
| numerical rank of a random-init IR | 9 | — | **128** (full) |
| numerical rank of the 128d random projection | — | — | **33** (it cannot exceed its input) |

## V-cell stage 2 now consumes the real navigable links

R24 shipped stage 2 with an honest hole: the mesh schema carried a `CELL` id but no
waypoint-link table, so `live_belief._vcmap` unioned *observed cell transitions* with a
**2-nearest-neighbour stand-in** rather than fusing contiguous **navigable** paths as
payload-spec 2.2.2 requires. The engine now streams the stock nav graph itself —
`payload_emit_cell_links()` walks `g_waypoints`, and for every `waypoint_get_link`
whose endpoint hashes into a different V-cell emits a kind-4 (`PLC_EVT_KIND_CELL_LINK`)
perception row with `CELL` = source, `SUBJECT` = destination, `VALUE` = link
length / 1024. The sweep cycles rather than latching because `g_waypoints` fills late:
`[PLCLINK] pass 1 over 467 waypoints, 3681 cell-link rows emitted`,
`[PLCLINK] pass 2 over 467 waypoints, 7395 cell-link rows emitted`.

`LiveBelief.ingest` now routes kind 4 into a navigable-link table
(`{(cell_a, cell_b): length}`, shortest wins) instead of the observation buffer — the
old `else: ENEMY_SEEN` branch would have filed every waypoint link as a sighted enemy —
and `_vcmap` passes that table to `segment_vcells` as `adjacency` **and** as the new
`edge_lengths`, so the graph distance the horizon is bisected against is the engine's
own traversal length rather than a centroid distance. A cell that only ever appears as
a link endpoint has no observed position (the cell id is a lossy
`(floor(x/256)*131 + floor(y/256)) & 1023` hash); those get a position propagated
breadth-first from their navigable neighbours, and that position is used only for
centroids, never for a graph distance.

A/B on the same real log (`run_links.log`, 513 usable ticks), same canonical
`segment_vcells`, the stand-in reproduced by dropping the kind-4 rows:

| | **kind-4 navigable links** | old 2-NN stand-in |
|---|---|---|
| ticks | 513 | 513 |
| median receptive fraction inside `[5%, 15%]` | **512 / 513** | 512 / 513 |
| median-of-medians | 0.05136 | 0.05333 |
| min / max of the per-tick median | 0.025 / 0.080 | 0.050 / 0.1818 |
| V-cells at the last tick | **334** | 254 |
| support radius (an OUTPUT of the bisection) | 0.9165 -> 0.9447 | 0.0 -> 0.5638 |
| distinct navigable links / link rows accepted | **1547 / 8048** | 0 / 0 |
| wall clock for 513 ticks | **10.28 s** (20 ms/tick) | 51.76 s |

Reading it honestly: the band was already being held, so this is not a rescue of a
broken bound — it is a change of *what the bound is computed over*. The link graph
carries **80 more V-cells** than were ever observed (they are navigable but nobody
walked them), the radius is now a real traversal length, and stage 2 got **5x faster**
because the map's topology stops changing once the sweep saturates, whereas the
observed-transition graph churned on every tick and forced a full all-pairs Dijkstra
each time. The one out-of-band tick is tick 2, where only 13 links over 40 nodes had
arrived and even the largest candidate radius reaches 1/40 = 2.5% of map area — an
opening-seconds transient of the cycling sweep, recovered by tick 3.

On the 993-tick run the engine predates the kind-4 emit, so that run reports
`link_source: observed+knn_fallback` and **992 / 993** ticks inside the band — the
fallback path is still measured, still correct, and now explicitly labelled in the
belief diagnostics of every tick.

## Three model-side changes this measurement required

**1. `EstCache.get` ignored the shape.** It returned the first estimator forever, so a
learner spanning matches of different team/cart/player counts would silently keep the
first shape — which is why the learner had to be per-match. The fix is not a per-shape
cache (that forgets the learner at every roster change); it is the observation that the
estimator is *genuinely* shape-agnostic — `W_q` is `(d, X_WIDTH + BELIEF_WIDTH)`, `W_k`
is `(d, INSTRUMENT_WIDTH)`, `GramSwiGLU` is sized by `d`, `MixingHead.in_dim` is the
fixed 21-feature row, the value probes are `Linear(d, 1)`; `k`, `j`, `l` enter only as
row counts — plus an *executed* proof of it. On every newly seen `(k, j, l)` a
throwaway estimator is built at that shape and its `architecture_spec` (the same
`[name, shape]` list R24's checkpoint fingerprint hashes) is compared leaf-for-leaf
against the live one; equal means reuse, different means a loud
`CheckpointArchitectureMismatch`. Verified on the two real shapes this run contains,
`(5, 3, 12)` x 992 and `(5, 3, 5)` x 1: same estimator object, fingerprint
`2329e58031db2fb7` unchanged, forward passes `ir=(12, 128)` and `ir=(5, 128)` both
finite, and an injected genuinely-different tree refused.

A latent bug fell out of it: a *roster* change (5 players -> 12) used to be credited
across as if it were a transition, reaching numpy as `operands could not be broadcast
together with shapes (12,8) (5,8)`. An instrument-set change stays representable (R24's
fix); a roster change now closes the credit segment, and `online.transition` names both
shapes instead of letting the broadcast error surface four frames away.

**2. The CGT was invisible.** `strat_responder` had stopped logging `game_value`
entirely, so R25's `0/228 -> 228/228` resolution never reached the live stream — the
pre-fix telemetry line carries `PW` and `SUCC` and nothing else about the cart game.
The per-tick call is back, logging `kind`, `nimber`, `reason`, the `(map_key, episode,
k, depths, controls)` state `measure.py cgt` reads back, per-role mobility, and a
running tally in the 5-second console line. Over the same 993 real ticks the closed
form resolves **993 / 993**: `{"partizan": 905, "impartial": 88}`, `nimbers {8: 88}`,
`reason: None` everywhere — the 905 partizan ticks are exactly the ones with a
controlled cart, and no nimber is faked for them.

**3. The DPP inverse died on real instrument counts.** Not a probe artifact: with one
V-cell instrument per discovered cell the batch reaches **549 instruments**, the DPP
kernel `L = diag(q) K^ K^T diag(q)` has entries `O(3e5)` at rank <= `d` = 128, and in
float32 the `+I` that makes `I + L` positive definite is five orders of magnitude below
the diagonal — LAPACK's LU hits an exactly-zero pivot and the shipped path dies with
`[Inverse::eval_cpu] LU factorization failed with error code 548`, the pivot index,
exactly where the rank runs out. The live responder would fail the same way. Fixed by
evaluating the same quantity in the `d`-dimensional dual (Woodbury: `((1+e)I + BB^T)^-1
= (I - B M^-1 B^T)/(1+e)` with `M = (1+e)I_d + B^T B`), with an exact hand-written VJP
that also stays in the dual. Only a `128 x 128` matrix with eigenvalues `>= 1 + e` is
ever inverted, so conditioning no longer depends on the instrument count, and the cost
drops from `O(m^3)` to `O(m d^2)`. Verified against the shipped
`dpp.dpp_marginals(..., method="inverse_diff")` on real rows: value agrees to
`<= 3.6e-6` and the gradient to `<= 2.2e-5` on a gradient of scale `0.55`.

## The measurement

993 ticks, 11,909 player rows, split by tick 60/40 (7,140 train / 4,769 test), ridge
probes, seed 0. Training is 5 budgets of the shipped loop over the same real
transitions — 2,025 to **32,400** gradient steps, 15,856 transitions, replay ring
261 items at its 255.55 MB ceiling (the ring is memory-bound, not count-bound, because
a single state now carries a `(12, 549, 16)` relation tensor). Architecture fingerprint
`c592fbe875fb7914`. Value loss 0.1126 -> 0.00406.

Read the `in x = no` rows: those are the quantities that are *not* columns of the
operator's input, so decoding them is induced structure rather than preservation.
`succ_denial` is constant on this run's test split and therefore not measurable — it is
reported as `—` rather than as a score.

| target | in `x` | IR trained | IR rand-init s0 | IR rand-init s7 | ctrl rand-proj 128d | raw inputs | ctrl shuffled |
|---|---|---|---|---|---|---|---|
| `cart_depth_total` | no | **0.9155** | 0.9404 | 0.9278 | 0.8971 | 0.8971 | -0.0172 |
| `n_controlled` | no | **0.7154** | 0.9388 | 0.9305 | 0.5175 | 0.5174 | -0.0129 |
| `cgt_nimber` | no | **0.7084** | 0.9339 | 0.9260 | 0.5180 | 0.5180 | -0.0220 |
| `n_eligible` | no | **0.8491** | 0.8944 | 0.8964 | 0.7011 | 0.7011 | -0.0301 |
| `nearest_dist` | no | **0.8836** | 0.9043 | 0.9071 | 0.8658 | 0.8658 | -0.0187 |
| `hier_margin` | no | **0.9968** | 0.9926 | 0.9920 | 1.0000 | 1.0000 | -0.0184 |
| `succ_denial` | no | **—** | — | — | — | — | — |
| `team (maj 0.2499)` | no | **0.9824** | 0.9906 | 0.9977 | 0.9698 | 0.9698 | 0.2485 |
| `pw_team (maj 0.78)` | no | **0.9998** | 0.9992 | 0.9990 | 1.0000 | 1.0000 | 0.7658 |
| `cgt_impartial (maj 0.9159)` | no | **0.9889** | 0.9975 | 0.9983 | 0.9579 | 0.9579 | 0.9168 |
| `nearest_kind (maj 0.7358)` | no | **0.8729** | 0.8771 | 0.8778 | 0.8551 | 0.8551 | 0.7232 |
| `own_nimber` | yes | **0.9987** | 0.9968 | 0.9953 | 1.0000 | 1.0000 | -0.0157 |
| `max_rival` | yes | **0.9765** | 0.9487 | 0.9367 | 1.0000 | 1.0000 | -0.0176 |
| `cart0_depth` | yes | **0.9146** | 0.9344 | 0.9211 | 0.9075 | 0.9075 | -0.0174 |
| `health` | yes | **0.9983** | 0.9955 | 0.9966 | 1.0000 | 1.0000 | -0.0187 |
| `armor` | yes | **0.9981** | 0.9953 | 0.9974 | 1.0000 | 1.0000 | -0.0199 |
| `ammo` | yes | **0.9999** | 0.9997 | 0.9996 | 1.0000 | 1.0000 | -0.0176 |
| `speed` | yes | **0.3677** | 0.4191 | 0.3830 | 0.2416 | 0.2416 | -0.0139 |
| `dist_to_cart` | yes | **0.9999** | 0.9999 | 0.9999 | 1.0000 | 1.0000 | -0.0163 |
| `tss` | yes | **0.9999** | 0.9998 | 0.9998 | 1.0000 | 1.0000 | -0.0139 |
| `n_weapons` | yes | **0.9995** | 0.9984 | 0.9989 | 1.0000 | 1.0000 | -0.0263 |
| `cell` | yes | **0.9984** | 0.9923 | 0.9936 | 1.0000 | 1.0000 | -0.0215 |
| `is_pw (maj 0.87)` | yes | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8673 |
| `weapon_bit (maj 0.6947)` | yes | **0.9168** | 0.9247 | 0.9331 | 0.8362 | 0.8362 | 0.7004 |
| `alive (maj 0.9782)` | yes | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9822 |

The **shuffled-label control passes everywhere**, on every target and every feature
set: regression R^2 in `[-0.030, -0.013]`, classification within 0.002 of the test
majority. The probes are not manufacturing signal.

## Does decodability collapse with training?

R24's finding was a monotonic collapse — `speed` R^2 falling to **-0.98 at 11,400
updates** on 62 telemetry lines — as the IR fell onto the value-relevant subspace.
With 190x more data that curve does not reproduce. Decodability is **flat from 2,025
to 16,200 updates**, and then degrades *mildly* by 32,400:

| target | in `x` | 1ep | 2ep | 4ep | 8ep | 16ep | rand-init | rand-proj |
|---|---|---|---|---|---|---|---|---|
| `cart_depth_total` | no | 0.9364 | 0.9355 | 0.9394 | 0.9392 | 0.9155 | 0.9404 | 0.8971 |
| `n_controlled` | no | 0.8645 | 0.8557 | 0.8668 | 0.8588 | 0.7154 | 0.9388 | 0.5175 |
| `cgt_nimber` | no | 0.8642 | 0.8550 | 0.8652 | 0.8589 | 0.7084 | 0.9339 | 0.5180 |
| `n_eligible` | no | 0.8996 | 0.8928 | 0.9002 | 0.8785 | 0.8491 | 0.8944 | 0.7011 |
| `nearest_dist` | no | 0.9129 | 0.9104 | 0.9130 | 0.9203 | 0.8836 | 0.9043 | 0.8658 |
| `hier_margin` | no | 0.9974 | 0.9979 | 0.9986 | 0.9990 | 0.9968 | 0.9926 | 1.0000 |
| `succ_denial` | no | — | — | — | — | — | — | — |
| `team` | no | 0.9962 | 0.9952 | 0.9952 | 0.9948 | 0.9824 | 0.9906 | 0.9698 |
| `pw_team` | no | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9992 | 1.0000 |
| `cgt_impartial` | no | 0.9935 | 0.9935 | 0.9960 | 0.9971 | 0.9889 | 0.9975 | 0.9579 |
| `nearest_kind` | no | 0.8744 | 0.8746 | 0.8725 | 0.8706 | 0.8729 | 0.8771 | 0.8551 |
| `own_nimber` | yes | 0.9991 | 0.9992 | 0.9995 | 0.9994 | 0.9987 | 0.9968 | 1.0000 |
| `max_rival` | yes | 0.9812 | 0.9865 | 0.9878 | 0.9928 | 0.9765 | 0.9487 | 1.0000 |
| `cart0_depth` | yes | 0.9326 | 0.9317 | 0.9352 | 0.9363 | 0.9146 | 0.9344 | 0.9075 |
| `health` | yes | 0.9994 | 0.9995 | 0.9995 | 0.9996 | 0.9983 | 0.9955 | 1.0000 |
| `armor` | yes | 0.9993 | 0.9994 | 0.9995 | 0.9994 | 0.9981 | 0.9953 | 1.0000 |
| `ammo` | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9997 | 1.0000 |
| `speed` | yes | 0.3929 | 0.3927 | 0.4044 | 0.3892 | 0.3677 | 0.4191 | 0.2416 |
| `dist_to_cart` | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1.0000 |
| `tss` | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9998 | 1.0000 |
| `n_weapons` | yes | 0.9998 | 0.9998 | 0.9998 | 0.9999 | 0.9995 | 0.9984 | 1.0000 |
| `cell` | yes | 0.9992 | 0.9993 | 0.9994 | 0.9995 | 0.9984 | 0.9923 | 1.0000 |
| `is_pw` | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `weapon_bit` | yes | 0.9274 | 0.9323 | 0.9262 | 0.9237 | 0.9168 | 0.9247 | 0.8362 |
| `alive` | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

| budget | updates| loss | effective rank (participation) |
|---|---|---|---|
| 1ep | 2025 | 0.27619 | 1.288 |
| 2ep | 4050 | 0.16493 | 1.305 |
| 4ep | 8100 | 0.04131 | 1.326 |
| 8ep | 16200 | 0.01 | 1.403 |
| 16ep | 32400 | 0.00406 | 1.728 |

So: **the collapse is deferred and softened, not eliminated.** At 11,400-equivalent
budget (8 epochs, 16,200 updates) nothing has degraded at all — `speed` is 0.3892 where
R24 had -0.98, and `nearest_dist` is at its best value of the whole sweep. Push to
32,400 and the cart-control targets give up about 0.14 R^2 (`n_controlled` 0.8588 ->
0.7154, `cgt_nimber` 0.8589 -> 0.7084) while the value loss keeps falling
(0.0100 -> 0.00406). That is the same phenomenon R24 named, arriving roughly 3x later
and an order of magnitude weaker.

And it is **not** a rank collapse. The IR's participation-ratio effective rank goes the
other way — **1.288 -> 1.728** across the sweep, dims-for-99%-variance 5 -> 10 — while
its top singular value falls 1442 -> 822. The IR is *spreading out* as it trains and
simultaneously getting worse at the two cart-control probes: a re-allocation of the
representation toward what the value loss pays for, not a contraction of it.

## Effective rank on real data

| | rank (1e-6) | effective rank (participation) | dims for 90% var | dims for 99% var | top singular value |
|---|---|---|---|---|---|
| raw input `[x ; beta]` (56 dims) | **33** | 1.260 | 2 | 5 | 1799.8 |
| IR, trained 32,400 updates | **128** | **1.728** | 3 | 10 | 822.0 |
| IR, random-init | **128** | 1.374 | 2 | 8 | 1508.2 |
| control: 128d random projection of the input | 33 | 1.187 | 1 | 5 | 3095.9 |

The operator is doing real work in the width SPEC 8 asks for: a rank-33 input becomes a
numerically full-rank 128d IR, which a linear map of the input provably cannot do (the
random projection is stuck at 33). Trained IR absmax 12.81, std 0.95 — bounded, finite
on all 11,909 rows, no saturation.

## Verdict

**PARTIAL — there is a semantically-rich j-space, but the value gradient is not what
makes it rich.**

*Yes*, on three counts. (i) Every non-tautological game quantity is linearly readable
out of the IR (best budget in the sweep): which team a player is on (0.996, majority
0.250), the projected winner (1.000), whether the cart subgame is impartial (0.997),
how many carts are controlled (0.867), the CGT nimber (0.865), the kind and distance of the nearest
eligible instrument (0.873 / 0.920) — the last two coming from `z`/`relation`, which
never touch the encoder at all. (ii) The trained IR beats the 128d random-projection
control on **every** target the raw inputs do not already saturate — at every budget
in the sweep, including the degraded 32,400-update endpoint — and by large margins
where it matters (values below at the best budget, 8,100 updates): `n_controlled` 0.867 vs 0.518, `cgt_nimber` 0.865 vs 0.518,
`n_eligible` 0.900 vs 0.701, `speed` 0.404 vs 0.242, `weapon_bit` 0.926 vs 0.836.
(iii) SPEC 6's "seemingly irrelevant errata" is specifically satisfied: *who is holding
a rocket launcher* decodes at 0.926 from the IR against 0.836 from a random projection
of the same inputs, on a run with 9 distinct weapon bits observed — R19 could not even
ask this question because the field was zeroed in its reconstruction.

*But not clean*, on two counts. (i) The random-init encoder of the same architecture is
within 0.01 of the trained one on most targets and **better** on 5 of the 11
non-tautological ones — decisively so on `n_controlled` (0.939 vs 0.867) and
`cgt_nimber` (0.934 vs 0.865). The separation from the random projection is the *Gram +
SwiGLU nonlinearity at 128d*; training adds a little (`hier_margin` 0.9926 -> 0.9990,
`max_rival` 0.9487 -> 0.9928, `cell` 0.9923 -> 0.9995, `pw_team` 0.9992 -> 1.0000) and
subtracts a little elsewhere. SPEC 7 asks that "the value gradient BETTER put trivially
sematnically measurable features into the learned projections"; what is measured is
that the *architecture* puts them there and the value gradient mostly leaves them
alone. (ii) Decodability still degrades at large budget.

R19's verdict was NO and it was measured on a probe that had zeroed the per-player
input. R24's verdict was "not a clean win, and the binding constraint is data." The
data constraint is now removed — 993 ticks, 11,909 rows, rank 33, all 19 OBS columns
live, `z` and `relation` real — and the answer moved from NO to PARTIAL. The remaining
gap is not data and not width. It is that a scalar value loss per player per head
constrains very few directions, exactly as R19's analytic note said; nothing in the
objective rewards decodability of anything else, so what decodes is what the operator's
*geometry* preserves. Making the j-space *earned* rather than *inherited* needs an
objective that pays for more than one scalar — not more rows.
