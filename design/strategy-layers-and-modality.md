# Four layers, one substrate, and the modality discipline

This records what the strategy project *is* once four abstractions that had been
collapsing into a single "the solver" are pulled apart, plus the RDMA/mesh substrate
that runs one of them fast. It also records the corrected picture of featurization and
belief, the committed shape of the per-bot mixing, why the game is Xonotic, and — kept
strictly separate — the hypotheses still under review. Where this document and the code
disagree, this document is the intent and the code is the bug. One exception, stated
plainly below: the RDMA source written so far follows no committed algorithm and is not
a spec; it must not be read back as one.

## 0. Modality discipline (read this first)

The single most load-bearing rule of this document is the distinction between *naming a
technique* and *specifying it*. Saying "we will probably need a Stackelberg here" is an
illocutionary act — a statement of intent about where the design is heading — not a
specification that a Stackelberg game is now part of the mechanics. Throughout, firm
decisions and open hypotheses are marked, and the two are never allowed to bleed:

  [FIRM]  a decision reached and committed; the code answers to it.
  [OPEN]  a hypothesis named but NOT decided; may change entirely.

A named-but-undecided technique is [OPEN] even when it is named confidently. The owner
is strict about this; preserve the markers.

## 1. The four layers and the substrate  [FIRM]

The project separates four abstractions, plus a substrate that is not a layer of the
game at all:

1. **Mechanics** — the game's ground-truth dynamics. `k` cart lanes, control
   cylinders, and a MONOTONE score model: team score is the integral of a rate,

   ```
   d(score_j)/dt = sum_over_lanes  control_j(lane) * depth(lane)
   ```

   The cart POSITION `s` is reversible (a cart can be pushed back down its lane), but
   SCORE never decreases — it is an accumulated integral, not a function of current
   position. This RETRACTS the earlier "un-banking" idea (that retreating past a banked
   point destroys the bank); that idea was wrong. Position reverses; banked score does
   not un-bank. Depth and control are the two factors the rate multiplies.

2. **Featurization** — world -> vectors. The read layer. (Section 2.)

3. **Strategy operator** — vectors -> allocation. This is the learned linear algebra:
   the value-function-LIKE computation. "Like" is exact wording — there is NO claim it
   reaches a global optimum of anything; it is a learned operator that behaves in the
   role a value function plays, not a solved value function.

4. **Control interface** — allocation -> bot behavior, via a goal-rating bias plus a
   spawn/travel commitment. It is deliberately SKILL-ORTHOGONAL: it steers WHERE a bot
   commits (which lane, which objective), never HOW WELL the bot duels once it is
   there. The learned policy cannot cheat by making bots aim better; it can only move
   their commitment.

**Substrate (not a layer).** RDMA/mesh runs layer 3 fast. It is infrastructure, not a
part of the game's abstraction stack. Consequently: the RDMA source written so far is a
*mere demonstrator* — it exercises the transport and the kernels, it follows no
committed strategy algorithm, and it is NOT a specification. Do not treat that source as
authoritative for what layer 3 computes; this document and its successors are.

## 2. Featurization and belief — corrected  [FIRM]

The corrections here retract a wrong mental model ("team belief") and replace it with a
per-team observation buffer plus strictly per-bot beliefs.

**There is no "team belief."** What a team has is a BUFFER of contextual OBSERVATIONS
deposited by its bots: match-state-dependent events such as "saw item_a spawn", "saw
item_b despawn", "saw an enemy cross lane 2". These observations are NOT ground truth
and must NOT be replaced by precomputed ground-truth timers. The entire point is that
the feature vectors support INFERENCE UNDER INCOMPLETE INFORMATION — a bot that has not
seen item_a spawn does not know it spawned. A precomputed timer would delete exactly the
uncertainty the whole apparatus exists to reason about.

**Only bots have beliefs.** A belief is:

- egocentric — it belongs to one bot, at that bot's current position;
- low-rank — a compressed pointwise integration, not a dense field;
- a pointwise integration of the observation buffer, evaluated at the bot's current
  V-cell, with two modifiers:
  (i) temporal contraction — a stale belief contracts toward an uninformative prior;
  (ii) spatial masking — observations are masked by distance from the bot.

**Why V-cells.** The bots' positions induce Voronoi cells. Contiguous NAVIGABLE paths
between cells are fused until a distance-decay CONTEXT MASK bounds the region. The
context mask is a PARALLEL op, NOT recurrent — it is a masking/decay applied in one
shot, not a sequential walk. Its job is to bound, two-sided, how much of the map each
bot's belief spans:

   ```
   5%  <=  map-area( bot's belief support )  <=  15%
   ```

no less than about 5%, no more than about 15% of the map, at bounded parallel compute.
A bot reasons about its neighborhood, not the whole map, and not a single cell.

**Why the mesh is non-vacuous.** Every observation buffer is rebuilt ONLINE every
strategy-estimation step. That online reconstruction, across many bots at a real game
cadence, is the reason a mesh of computers is not redundant even though the individual
kernels are efficient — the work is the many-bot, per-step rebuild, not one heavy solve.

## 3. Per-bot mixing — a QKV whose coupling is a DPP kernel

**[FIRM] shape; [OPEN] the specific transforms.** The mixing has a committed shape; the
exact projection matrices are candidate choices, not yet fixed.

Each bot forms a QUERY that is a learned projection of TWO inputs concatenated:

   ```
   query_bot = W_q * [ x_engine  ++  belief_bot ]
   ```

where `x_engine` is the bot's raw Xonotic engine vector and `belief_bot` is its
egocentric belief integration from section 2. The concatenation is the point: the query
is a function of both what the engine reports and what the bot has inferred.

Instruments carry learned KEY and behavioral VALUE projections:

   ```
   key_i   = W_k * instrument_i
   value_i = W_v * instrument_i     (the behavioral payload of committing to i)
   ```

**The coupling is a DPP kernel.**  [FIRM]  The coupling/mixing object — the Gram over
queries and keys — is decided to be a **determinantal point process (DPP) kernel**. Its
DETERMINANT induces DIVERSITY over the committed instruments: an anti-redundancy prior
that discourages many bots from piling onto the same instrument unless the payoff
justifies the collapse of diversity. The determinant, not a scalar similarity, is the
operative quantity.

**Output is a velocity, not an allocation snapshot.**  [FIRM]  Strategy and tactics are
emitted as a VELOCITY on an integrated weight state:

   ```
   dw/dt         (team-scale and bot-scale weight states)
   ```

The engine integrates `dw/dt` between strategy updates. This decouples the strategy
cadence from the game tick: the solver need not run every tick; it sets a velocity and
the engine integrates the weight state forward until the next strategy update. Both a
team-scale and a bot-scale weight state exist.

## 4. Why Xonotic  [FIRM]

The choice of Xonotic is mechanically motivated, not incidental. Basic weapons carry
splash and knockback. Splash soft-threatens (perturbs) opponents without killing them;
knockback PHYSICALLY DISPLACES enemy bodies. Because control in the mechanics is a
function of bodies inside a cart's control cylinder, knockback that pushes an enemy body
OUT of the cylinder DIRECTLY reduces the opposing presence term in the cart-velocity
law — the same `control_j` / presence quantity the score rate and the push/pull law
read.

The consequence is the design's key affordance: "suppress a cart" is achievable by
ZONING, without kills. That gives the learned value operator a mechanical,
SKILL-ORTHOGONAL path to succeed — it can win position by displacing bodies, which the
control interface can actually command, rather than by out-dueling, which it cannot. The
free parameters of the learned operator should end up ENCODING this path: the design
expects the learned weights to discover zoning-by-knockback as a lever precisely because
it is the lever the control interface exposes.

## 5. Open hypotheses — NOT decided  [OPEN]

Most items here are named but undecided; they may change entirely, and are recorded so
the intent is legible, not because they are committed. **Two items — (d) reference
values and the Nim entry — have since been RESOLVED** by the authoritative specs
(`rl-training-spec.md`, `dpp-mixing-and-overlay.md`, `payload-spec.md`) and are marked
`[FIRM]` inline below rather than moved, so the reasoning trail stays intact. The
remaining items ((a), (b), (c)) are still open.

**(a) Stackelberg as LEARNED, not hand-declared.**  [OPEN]  Leadership/backward-
induction may enter, but as a LEARNED READOUT rather than a hand-coded game tree:
"leadership" as a learned readout of a team's resource concentration — armor, health,
rocket-launchers on surviving players near a lane. Commitment would be represented as
MOMENTUM in the velocity/weight state (section 3), and followers respond to the leader's
trajectory. Whether any explicit Stackelberg structure is declared at all remains open;
the leaning is toward letting it be learned.

**(b) DPP diversity as the non-dominated response; coalition as its complement.**
[OPEN]  DPP diversity-against-the-leader may serve as the non-dominated-response
operator (spread away from what the leader commits). Its complement is COALITION —
piling onto the leader's key lane — which is the LOW-diversity regime of the same DPP
kernel. Which regime is correct in a given state would be PRICED by REINFORCE, not
hand-selected.

**(c) The governing principle: REPRESENT, don't EVALUATE.**  [OPEN as a principle under
test]  Hand-featurize only enough that leadership, dominance, and swing are
REPRESENTABLE; push ALL evaluation into learned parameters calibrated by many matches.
The hand-built part supplies a basis in which the phenomena can be expressed; it does
not score them.

**(d) Minimap horizon and reference values — RESOLVED.**  [FIRM]  Both items once
listed here are now settled:
- *Does the system compute a Grundy / key-player REFERENCE value?* YES, over the
  reductions where it is closed-form. The projected winner `PW` and its succession
  `SUCC` are the COMPUTED nim-sum and its backward induction over the CARTSTATE game
  (see the Nim entry below and `rl-training-spec.md` §1), and the key-player SWING is
  the closed-form intercentrality readout over the coupling `kappa`
  (`dpp-mixing-and-overlay.md` §3). What is left to REINFORCE is the learned VALUE of
  the full partizan position, not these computed features.
- *The minimap horizon* is set by the §2 V-cell construction (fuse navigable cells
  until the belief support lands in the [~5%, ~15%] band); per `payload-spec.md` §2.2
  there is no fixed-vs-learned fork to decide.

**Nim — RESOLVED into two games (was: "[OPEN], analogy only").**  [FIRM, per
`rl-training-spec.md` §0–§1, §4 and `payload-spec.md` §2.6]  The earlier "analogy only,
no Grundy in closed form" framing conflated two games and is now settled by SPLITTING
them. This is the honest reconciliation of the apparent contradiction between this
entry and `payload-spec.md` §2.6, which names a nimber value for the multi-cart
position:

- **The full game is partizan / scoring.** With presence, control cylinders, item
  economies, and the frozen-FPS realization all live, the full position has **no
  closed-form Grundy value** — its value is LEARNED (the per-player value head,
  `rl-training-spec.md` §2). This is the sense in which the old entry was right: "the
  position value cannot be read off cheaply, which is why the solve is heavy."
- **The CARTSTATE game is the tractable reduction.** Restricted to cart
  depths-under-control as heaps, the position admits a nimber-preserving reduction
  (the FUN2022 nimber-preserving-reduction result). Over THAT reduction the projected
  winner `PW` — the nim-sum "who wins if nothing else moves" (one cart at depth 2 beats
  two carts at depth 1, since 1⊕1 = 0) — and its succession `SUCC` — backward induction
  over successive decrements of the leader's carts — are **COMPUTED in closed form**,
  deterministic, NOT learned. This is exactly the nimber value `payload-spec.md` §2.6
  names, and the same `PW`/`SUCC` that `rl-training-spec.md` §1 defines as computed
  Game-1 features.

So Nim is neither "mere analogy" nor "the whole game is a Nim variant": the cartstate
reduction IS a computed nim-sum (Game 1, deterministic), while the FULL game's value is
learned (Game 2's frozen-FPS realization is what the policy must actually achieve).
`PW`/`SUCC` are FEATURES the learned value and policy read; they are stopgrad, never
themselves learned (`rl-training-spec.md` §1, §4).
