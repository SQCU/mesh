# Strategy vectors, the Gram, and a real QKV over them

This answers four questions the owner put to the solver's strategic core, with exact
references into `xonotic/solver/worker.py`, and proposes an architecture that gives the
solve a genuine query/key/value structure over strategy vectors, makes the query/value
geometry learnable toward the reward, and keeps the Gram as a separated, named strategic
object instead of a folded scalar. Line numbers are against the tree at the time of
writing.

## 0. What actually flows through the solve

Per tick, `CtxSolver.solve` (worker.py:248) does, in the mlx path `_mlx` (worker.py:255):

1. `H = C @ Ein` — the rolling context window `C` (T x REQ_WIDTH, T=4096) is lifted into a
   RES=2048 residual basis. `Ein` is frozen random (`ctx_weights`, worker.py:130, seeded by
   `_init(SEED)`).
2. `e = argmax(H @ Rt)` — data-dependent hard routing of each of the T rows to one of
   EXPERTS=8 experts. `Rt` frozen random.
3. per expert `Hs @ W1[i] -> relu -> @ W2[i]`, scattered back; `Z = H + expert(H)` — a routed
   MoE feedforward with a residual. `W1,W2` frozen random.
4. `Gm = (Z.T @ Z) / T` — a **RES x RES** second moment. This is a covariance **over the
   feature axis**, summed over the T rows. It is emphatically not a row x row (token x token)
   attention matrix, which would be `Z @ Z.T`.
5. `Zcur = Z[idx]` — the current tick's rows in the residual basis.
6. Outputs:
   - `G  = Zcur @ (Gm @ Og)`  -> per-team score (TEAMS), worker.py:274. `Og` frozen random.
   - `G2 = Zcur @ (Gm @ Og2)` -> per-(cart,team) score (K_CARTS*TEAMS), worker.py:277.
   - `Zc = (Zcur @ Gm) * (RES/tr)` -> the current rows, transformed by the Gram and
     trace-normalised, RES-wide (worker.py:276). This is the feature that reaches the
     strategy layer.
   - `V  = group_mask @ Z` (worker.py:278): `_group_mask` (worker.py:228) is
     (TEAMS*NINST x T); each row is a mean over the window rows of one (team, instrument)
     group minus the global mean. So **V is a per-(team,instrument) mean residual
     embedding**, TEAMS*NINST x RES.
   - `Kp = V @ Gm @ V.T` (worker.py:279): the Gram-metric inner product between those group
     embeddings, TEAMS*NINST x TEAMS*NINST, normalised to `kappa` by `_kappa_norm`
     (worker.py:240) into a unit-diagonal coupling field.

## 1. Do we use the routed/sparse MoE to compute useful QUERY features?

Precisely: the routed experts produce a **per-row RES-wide residual feature vector** `Z` — a
data-dependent, routed, nonlinear transform of the window. That is all. There is no
query/key dot product over `Z` at the row level; nothing attends. The expert output is a
**feature transform**, consumed two ways: (a) linearly projected through `Gm @ Og`/`Gm @ Og2`
into team/cart scores, and (b) pooled by `_group_mask` into the per-(team,instrument)
embeddings `V`.

The single place anything resembling a query/key comparison appears is `Kp = V @ Gm @ V.T`
(worker.py:279): `V` rows are group embeddings and `Kp` is their pairwise similarity **under
the Gm metric**. That is structurally a key-key Gram over strategy embeddings — the raw
material of attention — but it is immediately reduced by `_kappa_norm` to a normalised scalar
field `kappa`, of which the allocation reads only the same-instrument cross-team entries
`kappa[(j-1)*NINST+c, (ctl-1)*NINST+c]` as the scalar `gram` / `gram_weight` comp
(worker.py:602, 615). So: **the MoE gives a frozen random feature transform, not query
features. A latent embedding-similarity object (`Kp`) is computed every tick and thrown
away down to one scalar.**

Note also the second, older MoE (`weights`/`scores_mlx`, worker.py:91-110) is used **only**
by the `nearest`/`inverted` frozen A/B policies, never by `trained`.

## 2. Do we have useful KEY or VALUE features?

No QKV decomposition exists in the trained solve. The pipeline is: routed MoE feedforward
(a feature transform) -> feature-covariance Gram (a second moment) -> linear score heads +
pooled group embeddings compared under that Gram metric. There are **no value vectors
carrying utility that get mixed by attention weights**, and **no query-key softmax**. The
allocation utilities `u0` (worker.py:592-618) are hand-built scalars per (cart,role)/post;
the learned `W_logit` term `phi @ self.W[bucket]` (worker.py:622) is a single linear head
over a pooled residual `phi` (worker.py:688: per-team mean of `Zc`, L2-normalised) — an
output head, not a value. **Finding: KEY/VALUE structure is absent.** The nearest analogue
to keys is `V` (group embeddings); the nearest analogue to a value is the analytic `u0`; the
nearest analogue to a query is `phi`; but they are never composed as attention.

## 3. Are we optimizing the queries/values toward the reward?

Learned:
- The 14 named THETA scalars (worker.py:72) — temperatures, appetites, weights — by
  REINFORCE with a per-team EMA baseline (`Strategy.update`, worker.py:817).
- `W_logit`: `self.W` (NBUCKET x RES x NALLOC), worker.py:492, by policy gradient
  (`gW` accumulation worker.py:722, applied worker.py:823), L2-decayed toward zero.

Frozen random (never see the reward gradient):
- The entire MoE and score heads: `Ein, Rt, W1, W2, Og, Og2` — all from `_init(SEED)`
  (worker.py:130-133), never updated.
- `Gm` is parameter-free (recomputed each solve).
- `V`, `kappa` parameter-free.

So **we optimise only (i) 14 analytic scalars and (ii) one linear logit head `W` over a
frozen-random pooled feature.** The query/value-like representations — the residual lift,
the routing, the score projections, the group embeddings — are frozen random and never
optimised toward reward. This is exactly the reservoir / random-feature regime: **an output
head learned on top of frozen random features.** `W_logit` reads a query-like vector `phi`
but does not reshape it; it just reads it out.

## 4. Separating the Gram as a strategic object; strategies as query/value vectors

Current reality: the Gram `Gm` is **folded twice** — (a) baked into the score linear map
`Zcur @ Gm @ Og`, where it acts as a fixed preconditioner on a random head, and (b) folded
into `kappa` via `V @ Gm @ V.T` then collapsed to a unit-diagonal scalar coupling. In neither
place is it a named, manipulable object over a strategy-vector space; it is a scalar-valued
side effect.

### Proposed architecture

Treat three things as first-class:

**A. Strategy vectors.** We already build them and discard the structure. Split the group
embeddings `V` into:
- per-team **query** embeddings `q_j` (a team's activity signature in the residual basis),
  taken from the team's pooled current residual `phi_j` (worker.py:688) and/or its `V`
  rows;
- per-(opponent, instrument) **key** embeddings `k_{o,i}` from the `V` rows — literally
  "what is rival `o` doing on instrument `i`";
- **values** `val_i` carrying the analytic utility of committing to instrument `i` (the
  `u0` contributions: controller state, post rank, `timing`).

**B. The Gram as a separated whitening / contest operator.** Do not fold `Gm` into the
score. Name it. Two honest roles, both already justified by the demo doc's orthogonalisation
story (`design/mesh-coprocessor-demo.md`, block-CGS2 + Newton-Schulz):
- **Whitening**: `Gm^{-1/2}` decorrelates the residual/strategy space so that
  query.key similarity is a Mahalanobis inner product `q^T Gm^{-1} k` rather than a raw dot
  product dominated by high-variance feature directions. Compute `Gm^{-1/2}` by
  Newton-Schulz (the same primitive the demo already relies on) and keep it as
  `self.white`. The diagonal restriction `white = 1/sqrt(diag(Gm))` is the cheap rank-1
  version and is what the minimal step ships.
- **Contest operator**: `Kp = V Gm V.T` stays available in full (not collapsed) as the
  explicit team x team coupling over instruments — the coalition signal the demo doc calls
  out ("the correlation between teams' unmet objectives *is* the coalition signal").

**C. Attention producing the allocation.** For team `j`, cell `c` mapped to instrument `i`:

```
logit(j,c) = u0(j,c)                                  # analytic prior, unchanged
           + qkv_weight * < q_j , k_{agg,i} >_{M,white}   # Gram-whitened, metric-M attention
pi_j = softmax(logit / tau)                           # the allocation, as today
```

where `< a,b >_{M,white} = sum_d white[d]^2 * M[d] * a[d] * b[d]`, `k_{agg,i}` is the
opponent-aggregated key for instrument `i`, and `M` (RES) is a **learnable diagonal
Mahalanobis metric** over the whitened strategy space. `M` reshapes the effective query
`q_j' = M . white^2 . q_j` — i.e. it is a **learnable diagonal query projection over the
expert features**, trained by REINFORCE, which is categorically different from the frozen-
feature output head `W_logit`. The full design promotes `M` to a low-rank `Wq/Wk/Wv`
(RES -> d) and `Gm^{-1/2}` to the full Newton-Schulz whitening; the diagonal case is the
rank-restricted first step.

This satisfies the three asks: (i) a real query/key/value attention over strategy vectors
(`q_j`, `k_{o,i}`, `val_i`); (ii) the query/value geometry (`M`, and later `Wq/Wk/Wv`) is
learned toward the reward, not just an output head over frozen features; (iii) the Gram is a
separated named object — `self.white` (whitening) and full `Kp` (contest) — never a folded
scalar.

### Budget

All new work is on top of quantities the solve already computes (`V`, `Zc`, `diag(Gm)`).
Per tick, over <=TEAMS teams and <=NALLOC=20 cells: build keys `Kcells` (NALLOC x RES),
form `q' = M . white^2 . q_j` (RES), `att = Kcells @ q'` (NALLOC). This is a few hundred
kFLOP against the ~104 GFLOP solve and the 25.8 ms measured mlx tick — negligible, well
within the ~100 ms budget. No new wire fields: `q_j`, `k`, `val` are all derived from the
existing request stream and window, so `bridge/PORT.md` is untouched (an internal solve
change).

## Minimal implemented step (shipped behind `--qkv`)

`worker.py`, off by default so every frozen baseline stays bit-identical:

- `CtxSolver` exposes two named Gram objects: `self.white` (RES = `1/sqrt(diag(Gm)+eps)`,
  the separated diagonal whitening) and `self.Vg` (the full per-(team,instrument) group
  embeddings `V`, previously discarded).
- `Strategy` gains a learnable diagonal metric `self.M` (RES, init ones) and a scalar
  `qkv_weight` (init 0.0). When `--qkv` is set, `_allocate` adds
  `qkv_weight * (Kcells @ (M . white^2 . q_j))` to `u0`, where `q_j` is the team's pooled
  normalised residual and `Kcells` are opponent-aggregated whitened key embeddings per cell.
  REINFORCE trains `M` and `qkv_weight`; L2 decay pulls `M` toward ones and `qkv_weight`
  toward zero; `qkv_weight` is hard-clipped. Persistence (`save`/`__init__`) carries `M` and
  `qkv_weight`; old `.npz` files load unchanged (guarded on key presence).
- With `qkv_weight` init 0 and the branch gated on the flag, `--qkv` off reproduces the
  current policy exactly (verified bit-stable in `--check`).
- The whitened query and each whitened key are L2-normalised before the metric-`M` inner
  product, so `att` is a bounded cosine (`|att| <= ~1`) and `qkv_weight` learns an O(1)
  weight the L2 regulariser can balance. Without this the raw residual keys times the
  whitening drive the logits to ~1e6 and the softmax collapses (measured: policy entropy
  0.02 nats, one instrument at mass 0.98).

Cutover to live: the flag is opt-in; nothing changes for the running worker until it is
launched with `--qkv`. See the report for the exact command.

### Validation findings

- `worker.py --check` is green on the mini (mlx), including two new assertions: `qkv inert
  at init` (qkv-on reproduces qkv-off returns bit-for-bit, so the frozen baseline is
  bit-stable) and `qkv trains, deterministic and bounded` (qkvw and M move, stay finite,
  `|qkvw|<=4`, deterministic across two runs).
- End-to-end tick (solve + strategy, qkv on) at 50 bots / 5 teams / 4 carts: median
  27.4 ms, max 29.6 ms -- well inside the ~100 ms budget (the bounded attention adds
  ~1.6 ms over the 25.8 ms solve).
- The `--synth` `good` gate is a finicky 13-way conjunction of stochastic thresholds that
  neither the baseline nor qkv passes across all (seed, ctx): at the default seed both exit
  1 on the same qkv-independent condition (a fresh qkv-off lead_bias counterfactual clears
  by 0.018 vs a 0.03 threshold). Where it is informative, qkv is the more stable of the
  two: at seed 7 / ctx 4096 the baseline W_logit head collapses (policy entropy 0.07 nats,
  concentration 0.94, returns -7.5 -> -4.7) while qkv stays regularised (entropy 2.65,
  concentration 0.34, returns +1.0 -> +8.6, banking exact) -- the bounded attention resists
  the collapse the raw frozen-feature head falls into.

## Future work

- Promote the diagonal `M` to low-rank `Wq/Wk/Wv` (RES -> d~=32) so queries, keys and values
  are independently learnable projections, not a shared diagonal metric.
- Replace diagonal `white` with full `Gm^{-1/2}` by Newton-Schulz (reuse the demo's
  orthogonalisation kernel), so whitening is a true decorrelation, not just per-feature scale.
- Surface the full contest operator `Kp` (uncollapsed) to the allocation as an explicit
  coalition term, replacing the single `gram_weight` scalar.
- Consider making the score heads `Og/Og2` (currently frozen random) learnable, closing the
  reservoir gap end to end.
