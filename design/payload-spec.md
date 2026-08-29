# Payload strategy — data-flow and execution-flow specification

This is the specification the mechanics, the strategy operator, and the playerbot
interface all answer to. It articulates what data exists, how it flows into the
learned linear algebra, and — the part that decides whether any of this matters — how
the computed output causally changes what bots do. Where this document and the code
disagree, this document states the intent and the code is the bug.

Modality is tracked with `[FIRM]` (decided), `[OPEN]` (a modeling choice still ours to
make), and `[BUILD]` (required engine work that does not yet exist). The existing
`sv_payload_mesh.qc` bridge is an **RDMA demonstrator**, not this spec: it proves the
scatter->behavior path is reachable, but its column schema and semantics are not
normative here.

Companion docs: `strategy-layers-and-modality.md` (the four layers), `cart-force-field.md`
(the continuous cart law), `dpp-mixing-and-overlay.md` (the mixing head + overlay),
`lit/gram-dpp-velocity-stackelberg.md` (the literature), `playerbot-interface.md` (the
grounded interface, file:line).

## 0. The layers, one line each `[FIRM]`

1. **Mechanics** — ground-truth dynamics: k cart lanes, control cylinders, monotone
   score `d(score_j)/dt = sum_lanes control_j(lane)*depth(lane)`; cart position `s`
   reverses, banked score never does.
2. **Featurization** — world -> vectors (this spec, section 2).
3. **Strategy operator** — vectors -> per-bot weight velocities (section 2.3-2.6).
4. **Control interface** — weights -> bot behavior (section 4), skill-orthogonal.

RDMA/mesh is **substrate** running layer 3 at the strategy cadence; it is not a layer.

## 1. Two clocks `[FIRM]`

The system runs on two decoupled cadences:

- **Game tick** (~72 Hz, in-engine): physics, control-cylinder occupancy, cart
  integration, and each bot's goal selection.
- **Strategy step** (1-10 Hz, on the mesh/coprocessor): the entire featurize -> Q/K/V
  -> DPP -> mixing-head -> flow-step pipeline, rebuilt online every step.

The strategy step is a **forward-Euler step of a replicator flow** (see 2.5, 3.2), so
its period `Delta` is simultaneously the integration step size and a stability
parameter with a discrete-time chaos threshold — not a free scheduling knob `[FIRM]`.
Rebuilding every buffer online each strategy step, over many bots, is precisely why a
mesh of computers is non-vacuous even with efficient kernels.

## 2. Data flow

### 2.1 Featurization inputs (given by the engine)

- **Per-bot engine vector `x_b`** — position, velocity, health, armor, `weapons`
  bitfield (embedded), ammo pools, powerup timers, team. All read from Xonotic.
- **Per-instrument descriptor `z_m`** — for each instrument m in {push cart k,
  suppress cart k, contest post p, hunt rival r, explore cell c}: the world features
  that describe that target (cart depth+control, post clock phase, rival tuple, cell
  belief).
- **The observation buffer** — feeds the belief pipeline (2.2).

### 2.2 Observation buffer -> belief (the tricky pipeline)

This is the pipeline that makes the strategy reason under **incomplete information**
rather than off omniscient world state. It has five stages.

1. **Observations `[FIRM]` `[BUILD: perception-gating]`.** Bots deposit timestamped,
   contextual events into a per-team buffer — "saw item_a spawn at cell c", "saw
   item_b gone" — from **bot perception**, not global truth. These are not ground
   truth and carry no authoritative respawn timer. NOTE: the demonstrator currently
   feeds this from omniscient `ITS_AVAILABLE` (`sv_payload_mesh.qc:186`) and the
   enemy-visibility traceline is commented out (`roles.qc:171-175`); replacing that
   with perception-gated writes is a required build and the single correctness risk to
   the incomplete-information pillar (section 4.5, gap 5).
2. **V-cell segmentation `[FIRM]`.** Partition the map into Voronoi cells over
   item/waypoint nodes; fuse contiguous **navigable** paths until the distance-decay
   context mask (stage 4) bounds each bot's receptive field, two-sided, to no less
   than ~5% and no more than ~15% of map area. Each cell holds a slot vector `f_c`
   (item type, respawn-phase estimate, standability, lane membership, last-threat,
   last-observed time).
3. **Temporal contraction `[FIRM]`.** Each cell relaxes toward an uninformative prior
   with age: `f_c^eff = rho(dt)*f_c^obs + (1 - rho(dt))*f_c^prior`, `rho(dt) =
   exp(-dt / T)`. Stale observations become uninformative — the buffer forgets.
4. **Spatial mask `[FIRM]`.** A bot in cell `c(b)` weights cells by graph distance
   through a bounded-support kernel `g(dist_graph(c(b), c))`. The support radius is the
   horizon = the context mask. It is a parallel operation, not a recurrence.
5. **Egocentric integration `[FIRM]`.** The belief is a pointwise, low-rank readout at
   the bot's cell:
   ```
   beta_b = sum_c  g(dist_graph(c(b), c)) * Phi * f_c^eff        (Phi low-rank)
   ```
   Cost: precompute the field `Phi*f_c^eff` once over the map = O(C * rank), then
   O(horizon) per bot readout — **scales with map size, not player count**. Two bots in
   the same cell with the same observations get the same `beta_b` because their inputs
   are identical, not because a shared "team belief" object exists. There is no team
   belief; only bots have beliefs.

Horizon parameterization (`g`'s support, `Phi`'s rank) is fixed-scalar vs learned:
`[OPEN]`.

### 2.3 Query / key / value `[FIRM shape; transforms candidate]`

```
q_b = W_q * [ x_b ; beta_b ]      (per-bot query: engine state AND egocentric belief)
k_m = W_k * z_m                    (per-instrument learned key)
v_m = W_v * z_m                    (per-instrument learned behavioral value)
```
Learned: `W_q, W_k, W_v`. Given: `x_b, z_m, beta_b`.

### 2.4 Coupling and the DPP signal `[FIRM]`

Form the L-ensemble kernel `L` over instrument keys/quality. The strategy operator does
**not** consume the determinant. It consumes the per-instrument **marginal inclusion
vector**:
```
K = L (I + L)^-1 ,   signal_m = diag(K)_m
```
one repulsion-shaped number per instrument — "how much does m belong in a diverse
commitment." Separately, the Gram/coupling `kappa` (the second-moment over
(team,instrument) or (bot,instrument) embeddings) is materialized for the overlay
(2.6).

### 2.5 Mixing head -> weight velocity `[FIRM]`

A learned gated head maps the DPP signal plus raw appetite to a per-instrument
**velocity**:
```
dw_b/dt = SwiGLU( RMSNorm( [ diag(K) ; b ] ) )
```
where `b` is per-instrument appetite/quality. The **SwiGLU gate is the regime switch**:
reading high *shared* appetite concentrated on one instrument, it opens the
concentration (pile-on / coalition) path; otherwise it passes the diversity (spread)
signal. The symmetric DPP supplies an honest one-sided signal; the head supplies the
two-regime behavior. No nonsymmetric DPP; nothing beyond RMSNorm+SwiGLU is required.

### 2.6 Multiscale overlay `[FIRM computable; OPEN consumed]`

A closed-form readout over the coupling `kappa`:
```
swing = argmax_i  intercentrality_i( (I - a*kappa)^-1 )
```
gives per-node action intensities (Bonacich) and the swing/key instrument in one linear
solve. Built at **team-scale `kappa`** it names the swing across teams/lanes; at
**bot-scale `kappa`**, the swing within a team — same operator, nested = multiscale.
Ontology discipline: "leader" is admissible ONLY as `argmax(intercentrality)`, a
derived readout, never a primitive entity or a per-agent character. `[OPEN]` whether
the flow (3.2) consumes the overlay asymmetrically (which alone would be "Stackelberg
dynamics") or it stays diagnostic.

## 3. Execution flow

### 3.1 Per game tick (engine, ~72 Hz)

Occupancy of each control cylinder, cart velocity + monotone score integration, and —
the layer-4 consumption — each bot reads its **latest absolute weight vector `w_b`** as
routerating biases (section 4) to choose a goal. Between strategy updates `w_b` is held
(optionally lerped for smoothness; the lerp is cosmetic, NOT the flow integration).

### 3.2 Per strategy step (coprocessor, 1-10 Hz)

```
gather   per-bot observation + state rows from the engine (mesh gather)
rebuild  observation buffer -> V-cells -> belief beta_b            (2.2, online)
featurize x_b, z_m ; project q_b, k_m, v_m                          (2.3)
couple   L -> K=L(I+L)^-1 -> diag(K) ; materialize kappa            (2.4)
mix      dw_b/dt = SwiGLU(RMSNorm([diag(K); b]))                    (2.5)
step     w_b += (dw_b/dt) * Delta        (ONE forward-Euler flow step)
overlay  swing = argmax intercentrality((I - a*kappa)^-1)          (2.6)
scatter  per-bot absolute w_b (and swing) back to the engine
```

**Integration locus `[DERIVED from FIRM velocity+cadence; confirm]`.** The flow
integration `w += (dw/dt)*Delta` lives **coprocessor-side at the strategy cadence**, so
`Delta` = the strategy step = the stability parameter. The engine receives the
integrated **absolute** `w_b`, not `dw/dt`. This is the only reading consistent with
"emit a velocity" + "cadence = step size": engine-side per-tick integration would make
the step size the game tick and re-couple the clocks. The demonstrator scatters
absolute weights with an inter-plan lerp, which is compatible with this reading. Flag
if you intended engine-side integration instead. `[OPEN]` instantaneous `w` vs a
running time-average (replicator orbits in zero-sum/cyclic regions).

## 4. The playerbot interface (grounded; execution-flow tail)

Every claim here is from real bot code; see `playerbot-interface.md` for full context.

### 4.1 The single lever `[FIRM, exists]`

`navigation_routerating(this, e, f, rangebias)` (`navigation.qc:1220`) is the only way a
goal enters a bot's route: it discounts a base rating `f` by travel cost
(`navigation.qc:1417`) and keeps the argmax into `navigation_bestgoal`
(`:1419-1423`). **Our per-instrument weight for the goal that instrument targets is the
`f` argument.** Skill-orthogonality is structural: routerating sets only
`goalentity`/route; aim and dodge live in `aim.qc:164-239`, driven purely by `skill` /
`bot_*aim*` cvars, and never read routerating. We bias where a bot commits; we cannot
and do not touch how it fights.

### 4.2 Per-bot hook `[FIRM, exists]`

Per-bot strategy is a swappable function pointer `.havocbot_role` (`havocbot.qc:64`),
installed per-gametype via `HavocBot_ChooseRole` (`sv_payload.qc:664`); the payload
rater is `havocbot_goalrating_payload` (`sv_payload.qc:606`). Clean per-bot, no global
patching.

### 4.3 Coprocessor bridge reachable `[FIRM, exists — demonstrator]`

`mesh_open/gather/scatter/publish/poll` builtins (`sv_payload_mesh.qc:18-23`) already
carry per-bot values into the payload rater, proving the scatter -> routerating path.
The columns are ours to define; the demonstrator's schema is not this spec.

### 4.4 Instruments -> raters

- push cart k, suppress cart k, contest post p: raters exist (payload role).
- **hunt rival r**: single-target rating primitive exists (`navigation.qc:1227`) but is
  not wired into the payload role and has no per-target variant. `[BUILD]`
- **explore cell c**: only a *random* waypoint rater exists (`roles.qc:16`); a
  cell-targeted rater must be built. `[BUILD]`
- spawn timing: `respawn_time`/`respawn_flags` fields exist (`client.qc:1341,2154,2162`);
  needs a scatter column. `[BUILD]`
- travel commitment: `bot_strategytime` exists (`navigation.qc:49`); needs a scatter
  column. `[BUILD]`

### 4.5 Must-build list (strict)

1. `[BUILD]` per-target `hunt rival` rater.
2. `[BUILD]` cell-targeted `explore` rater.
3. `[BUILD]` spawn-timing scatter column.
4. `[BUILD]` travel-commitment scatter column.
5. `[BUILD, riskiest]` perception-gated observation buffer. The current omniscient feed
   (2.2, stage 1) is a correctness bug against the incomplete-information pillar: if the
   coprocessor plans on state the bots could not perceive, the belief pipeline is
   theater. This is the one build that can invalidate the "real causal impact" claim,
   and it is the highest priority.

## 5. Open decisions carried forward

- Integration locus: coprocessor-side derived (3.2); confirm.
- Minimap horizon: fixed scalar vs learned (2.2).
- Overlay consumption: does the flow read the swing (Stackelberg dynamics) or is it
  diagnostic (2.6).
- Instantaneous `w` vs time-averaged (3.2).
- DPP selection: sample vs greedy-MAP; outer loss: DPP-likelihood vs reward
  policy-gradient (from the lit review).
- Whether to compute the key-player overlay as a *reference* to check the learned
  operator, or stay pure-REINFORCE.
