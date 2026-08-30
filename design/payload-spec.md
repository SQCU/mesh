# Payload strategy — data-flow and execution-flow specification

This is the specification the mechanics, the strategy operator, and the playerbot
interface all answer to. It articulates what data exists, how it flows into the
learned linear algebra, and — the part that decides whether any of this matters — how
the computed output causally changes what bots do. Where this document and the code
disagree, this document states the intent and the code is the bug.

**There is no current strategy implementation.** Source that resembles strategy
computation is disregarded babble; the spec, the linear-algebra terms, and the
transcript are the only reference. The RDMA scatter/gather builtins are real substrate;
their *use* by any existing strategy code is not.

Modality is tracked with `[FIRM]` (decided), `[BUILD]` (required engine work that does
not yet exist). `[OPEN]` is used only where the transcript genuinely has not yet chosen.

Companion docs: `strategy-layers-and-modality.md`, `cart-force-field.md`,
`dpp-mixing-and-overlay.md`, `lit/gram-dpp-velocity-stackelberg.md`,
`playerbot-interface.md`.

## 0. The layers, one line each `[FIRM]`

1. **Mechanics** — ground-truth dynamics: k cart lanes, control cylinders, monotone
   score `d(score_j)/dt = sum_lanes control_j(lane)*depth(lane)`, `depth` measured
   along each cart's **golden path** (arclength `s` in [0,L], origin->end); cart
   position `s` reverses, banked score never does.
2. **Featurization** — world -> vectors (section 2).
3. **Strategy operator** — vectors -> per-bot weight velocities (section 2.3-2.6).
4. **Control interface** — weights -> bot behavior (section 4), skill-orthogonal.

RDMA/mesh is **substrate** running layer 3 at the strategy cadence; it is not a layer.

## 1. Two clocks `[FIRM]`

- **Game tick** (~72 Hz, in-engine): physics, control-cylinder occupancy, cart
  integration, each bot's goal selection.
- **Strategy step** (1-10 Hz, on the mesh): the entire featurize -> Q/K/V -> DPP ->
  mixing-head -> flow-step pipeline, rebuilt online every step.

The strategy step is a forward-Euler step of a replicator flow (2.5, 3.2), so its
period `Delta` is simultaneously the integration step size and a stability parameter
with a discrete-time chaos threshold — not a free scheduling knob. Rebuilding every
buffer online each strategy step, over many bots, is why a mesh of computers is
non-vacuous even with efficient kernels.

## 2. Data flow

### 2.1 Featurization inputs

- **Per-bot engine vector `x_b`** `[FIRM]` — the bot's OWN, known state: position,
  velocity, health, armor, `weapons` bitfield (embedded), ammo pools, powerup timers,
  team. Read from Xonotic.
- **Per-instrument descriptor `z_m`** `[FIRM]` — for each instrument m in {push cart k,
  suppress cart k, contest post p, hunt rival r, explore cell c}: the features
  describing that target. Instruments referencing an enemy (suppress, hunt) are
  well-defined only for enemies the team has OBSERVED (2.2).
- **The observation buffer** — feeds the belief (2.2).

### 2.2 Observation buffer -> belief

This pipeline makes the strategy reason under **incomplete information** rather than off
omniscient world state. There is no omniscient enemy or item state anywhere in the
system; everything spatial enters here.

1. **A bot observation `[FIRM]` `[BUILD]`.** An observation is a map feature that
   (a) is in a bot's **view frustum**, (b) is **unoccluded** by a line-of-sight
   raycast, and (c) is **within 2 V-cells** of range — a hard cap so a pathological
   long-sightline custom map cannot let one glance rewrite the whole buffer. All three
   gates necessary. A pass deposits a timestamped contextual event at that cell — "RL
   gone at cell c", "enemy of team t at cell c" — into the per-team buffer. Nothing
   enters that no bot actually saw. Two consequences fall straight out:
   - **Stealth is emergent, not a stat.** A body or taken pickup that no enemy bot
     looked at (out of frustum, occluded, or beyond 2 cells) is absent from the enemy
     team's buffer, so their next strategy step cannot condition on it. Moving unseen is
     genuinely hidden; a bot that sweeps its view gathers team intel.
   - **Enemy positions are featurized ONLY through here.** An enemy bot's position
     reaches the strategy operator only as an observed event in some teammate's buffer,
     spatialized by stage 5. There is no other path.
2. **V-cell segmentation `[FIRM]`.** Partition the map into Voronoi cells over
   item/waypoint nodes; fuse contiguous **navigable** paths until the distance-decay
   context mask (stage 4) bounds each bot's receptive field, two-sided, to no less than
   ~5% and no more than ~15% of map area. Each cell holds a slot vector `f_c` (item
   type, respawn-phase estimate, standability, lane membership, last-threat,
   last-observed time, observed-enemy presence).
3. **Temporal contraction `[FIRM]`.** `f_c^eff = rho(dt)*f_c^obs + (1-rho(dt))*f_c^prior`,
   `rho(dt) = exp(-dt/T)`. Stale observations relax to an uninformative prior — the
   buffer forgets.
4. **Spatial mask `[FIRM]`.** A bot in cell `c(b)` weights cells by graph distance
   through a bounded-support kernel `g(dist_graph(c(b), c))`. The support radius is the
   horizon = the context mask; a parallel operation, not a recurrence.
5. **Egocentric integration `[FIRM]`.**
   ```
   beta_b = sum_c  g(dist_graph(c(b), c)) * Phi * f_c^eff        (Phi low-rank)
   ```
   Precompute `Phi*f_c^eff` once over the map = O(C*rank), then O(horizon) per bot —
   **scales with map size, not player count**. Two bots in the same cell with the same
   observations get the same `beta_b` because their inputs are identical, not because a
   "team belief" object exists. There is no team belief; only bots have beliefs.

**The belief integration is the system's ONLY spatial mixing operator `[FIRM]`.** We
never introduced attention or any other spatial mixer, so stage 5 is the sole place
spatially-distributed information becomes a spatialized intermediate. Its
graph-distance kernel `g` plays the role softmax attention would: it is how pickup
state, exploration state, AND observed enemy positions get mixed into `q_b`, each
position-weighted and egocentric-weighted. `x_b` is the bot's own known state; `beta_b`
is the observed, occlusion-gated, spatialized world around it. Without stage 5 there is
no geometry in the strategy — only a bag of unplaced scalars.

**The horizon is not a parameter choice.** It is set by the stage-2 construction —
fuse navigable cells until the receptive field lands in the [~5%, ~15%] band. That
construction is the answer; there is no fixed-vs-learned fork.

### 2.3 Query / key / value `[FIRM shape; transforms candidate]`

```
q_b = W_q * [ x_b ; beta_b ]      (per-bot query: known self-state AND observed world)
k_m = W_k * z_m                    (per-instrument learned key)
v_m = W_v * z_m                    (per-instrument learned behavioral value)
```
Learned: `W_q, W_k, W_v`. Given: `x_b, z_m, beta_b`.

### 2.4 Coupling and the DPP signal `[FIRM]`

Form the L-ensemble kernel `L` over instrument keys/quality. The operator consumes the
per-instrument **marginal inclusion vector**, not the determinant:
```
K = L (I + L)^-1 ,   signal_m = diag(K)_m
```
one repulsion-shaped number per instrument. Separately the Gram/coupling `kappa` is
materialized for the overlay (2.6).

### 2.5 Mixing head -> weight velocity `[FIRM]`

```
dw_b/dt = SwiGLU( RMSNorm( [ diag(K) ; b ] ) )
```
`b` = per-instrument appetite/quality. The **SwiGLU gate is the regime switch**: high
*shared* appetite concentrated on one instrument opens the concentration (pile-on /
coalition) path; otherwise it passes the diversity (spread) signal. Symmetric DPP
supplies the honest one-sided signal; the head supplies the two-regime behavior. No
nonsymmetric DPP; nothing beyond RMSNorm+SwiGLU is required.

### 2.6 Nimber value and explicit backward induction `[FIRM]`

The multi-cart position is Nim-structured: each cart at its depth-under-control on its
golden path is a heap. The projected winner `PW(s)` is **computed deterministically as
the nim-sum over cartstate** — per-team XOR of controlled cart depths; the team holding
the largest live nimber wins all-else-equal (one cart at d:2 beats two carts at d:1,
since 1 XOR 1 = 0). It is closed-form over cartstate (Game 1) — NOT a learned quantity
and NOT a centrality heuristic. See `rl-training-spec.md` §1 and
`xonotic/solver/strat/game.py`. (Earlier drafts computed a `argmax intercentrality((I -
a*kappa)^-1)` "swing" over a learned coupling and called it the nimber; that was a
relabel of a centrality heuristic and is retracted — the nimber is the nim-sum above.)

`SUCC(s)` is **explicit backward induction** over that nimber-valued position: recompute
`PW` under successive decrements of the current leader's carts, yielding the ordered
succession `[(team, marginal_denial_value)]`. Required, not an optional overlay: it folds
the whole succession into one immediate-frame allocation so the policy is anticipatory
and time-smooth (gang the leader only to its marginal need, pre-empt the next-in-line,
loot for power) instead of reacting to each flip after it has happened. `PW`/`SUCC` are
stopgrad FEATURES the policy and value read; they are never learned. What is learned is
only the Game-2 realization — whether an allocation can actually decrement the leader
through the frozen FPS. Ontology discipline: "leader" is admissible only as the derived
`PW(s)`/succession readout — never a primitive entity or a per-agent character.

## 3. Execution flow

### 3.1 Per game tick (engine, ~72 Hz)

Control-cylinder occupancy, cart velocity + monotone score integration, and — layer 4 —
each bot reads its latest absolute weight vector `w_b` as routerating biases (section 4)
to choose a goal. Between strategy updates `w_b` is held (optionally lerped for
smoothness; the lerp is cosmetic, NOT the flow integration).

### 3.2 Per strategy step (mesh, 1-10 Hz)

```
gather   per-bot observation + state rows from the engine
rebuild  observation buffer -> V-cells -> belief beta_b            (2.2, online)
featurize x_b, z_m ; project q_b, k_m, v_m                          (2.3)
couple   L -> K=L(I+L)^-1 -> diag(K) ; materialize kappa            (2.4)
mix      dw_b/dt = SwiGLU(RMSNorm([diag(K); b]))                    (2.5)
step     w_b += (dw_b/dt) * Delta        (ONE forward-Euler flow step)
induct   nimber value + backward induction over the multi-cart position;
         swing = argmax intercentrality((I - a*kappa)^-1)          (2.6)
scatter  per-bot absolute w_b (and swing) back to the engine
```

**Both scales integrate off-engine `[FIRM]`.** Team-scale and bot-scale weight states
both integrate on the mesh, never in the engine — this is just the offloading thesis:
the engine reads a buffer, and scaling the mesh (17 more minis) only changes how often
that buffer refreshes, it never puts an allreduce inside the engine's core loop. The
flow step runs at the strategy cadence; the engine receives the integrated **absolute**
`w_b` and reads it as bias. `Delta` = the strategy step = the stability parameter.

## 4. The playerbot interface (grounded; execution-flow tail)

Every claim from real bot code; see `playerbot-interface.md` for file:line context.

### 4.1 The single lever `[FIRM, exists]`

`navigation_routerating(this, e, f, rangebias)` (`navigation.qc:1220`) is the only way a
goal enters a bot's route: it discounts a base rating `f` by travel cost
(`navigation.qc:1417`) and keeps the argmax into `navigation_bestgoal` (`:1419-1423`).
**Our per-instrument weight for the goal that instrument targets is the `f` argument.**
Skill-orthogonality is structural: routerating sets only `goalentity`/route; aim and
dodge live in `aim.qc:164-239`, driven by `skill`/`bot_*aim*` cvars, and never read
routerating. We bias where a bot commits; we cannot touch how it fights.

### 4.2 Per-bot hook `[FIRM, exists]`

`.havocbot_role` (`havocbot.qc:64`), installed per-gametype via `HavocBot_ChooseRole`
(`sv_payload.qc:664`); the payload rater is `havocbot_goalrating_payload`
(`sv_payload.qc:606`). Clean per-bot, no global patching.

### 4.3 Scatter path reachable `[FIRM, exists — substrate]`

The RDMA scatter/gather builtins can carry per-bot values into the payload rater,
proving the scatter -> routerating path is wired at the substrate level. The columns and
their semantics are defined by this spec.

### 4.4 Instruments -> raters

- push cart k, suppress cart k, contest post p: raters exist (payload role).
- **hunt rival r**: single-target primitive exists (`navigation.qc:1227`), not wired
  into the payload role, no per-target variant. `[BUILD]`
- **explore cell c**: only a *random* waypoint rater exists (`roles.qc:16`); a
  cell-targeted rater must be built. `[BUILD]`
- spawn timing: `respawn_time`/`respawn_flags` fields exist
  (`client.qc:1341,2154,2162`); needs a scatter column. `[BUILD]`
- travel commitment: `bot_strategytime` exists (`navigation.qc:49`); needs a scatter
  column. `[BUILD]`

### 4.5 Must-build list

1. `[BUILD]` per-target `hunt rival` rater.
2. `[BUILD]` cell-targeted `explore` rater.
3. `[BUILD]` spawn-timing scatter column.
4. `[BUILD]` travel-commitment scatter column.
5. `[BUILD, foundational]` the perception-gated observation buffer of section 2.2 (the
   frustum + LOS + 2-V-cell gate). This is where the incomplete-information pillar and
   the stealth mechanic live, and the only path by which enemy positions enter the
   system. Foundational, not a patch.

## 5. Training and selection `[FIRM]`

Both were decided from the outset and restated repeatedly; neither is a fork.

- **Selection is sampling, not MAP.** The policy is a weighted sampling over strategies,
  L1/L2-regularized toward logit 0 so that untrained it is a broad weighted sampling of
  effective strategies and with training it peaks without collapsing to "only some
  actions happening". Greedy/MAP selection is excluded by that requirement.
- **The outer objective is REINFORCE** over match outcomes (many winning and losing
  bots across many matches = the calibration signal), with the L2-toward-zero penalty.
  Not a DPP-likelihood objective: there are no target selections to fit, only reward.

This is the standing commitment — minimal feature engineering, maximal weight on the
learned parameters, and only ever the mechanism that works. Decisions are derived from
it, not reopened as forks.
