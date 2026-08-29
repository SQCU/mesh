---
name: Strategy layers + modality — four abstractions, DPP-QKV mixing, learned-not-declared
description: Normalized companion to strategy-layers-and-modality.md. Four-layer framing (mechanics/featurization/strategy-operator/control) + RDMA substrate; monotone-score retraction of un-banking; per-team observation buffer w/ per-bot egocentric beliefs over V-cells (5-15% mask); per-bot QKV whose coupling IS a DPP kernel emitting dw/dt velocity; Xonotic=knockback-zoning skill-orthogonal lever; Stackelberg/coalition/Grundy left OPEN.
type: project
metadata:
  node_type: memory
  register: memory-hook (frontmatter name/description/type + compressed normalized bullets, per ~/.claude-personal/.../memory/*.md); modality markers [FIRM]/[OPEN] added because the owner is strict about firm-vs-hypothesis. If a different normalized convention was wanted, this is the spot to correct it.
---

CONVENTION NOTE (read first): this file follows the owner's auto-memory hook style —
YAML frontmatter (name / description / type) over terse normalized bullets with
compressed hooks — NOT the discursive `design/*.md` prose style. Firm decisions carry
[FIRM]; undecided hypotheses carry [OPEN]. Companion prose w/ full reasoning:
`design/strategy-layers-and-modality.md` (that doc is the intent where code disagrees).

## Modality discipline
- Naming a technique ("we'll probably need a Stackelberg") = ILLOCUTIONARY intent, NOT a
  spec. [FIRM discipline] Named-but-undecided stays [OPEN] however confidently named.
- RDMA source written so far = a mere RDMA DEMONSTRATOR following NO committed algorithm;
  it is NOT a spec and must not be read back as one. [FIRM]

## Four layers + substrate  [FIRM]
1. Mechanics — ground-truth dynamics. k cart lanes + control cylinders. MONOTONE score:
   d(score_j)/dt = sum_lanes control_j(lane)*depth(lane). Cart POSITION s reversible;
   SCORE never decreases (integral of a rate). RETRACTS earlier "un-banking" idea (WRONG:
   position reverses, banked score does not).
2. Featurization — world->vectors (read layer).
3. Strategy operator — vectors->allocation; the learned linear algebra; value-function-
   LIKE (NO claim of any global optimum).
4. Control interface — allocation->bot behavior via goal-rating bias + spawn/travel
   commitment. SKILL-ORTHOGONAL: steers WHERE a bot commits, never HOW WELL it duels.
- RDMA/mesh = SUBSTRATE running layer 3 fast; NOT a layer of the game.

## Featurization / belief  [FIRM corrections]
- NO "team belief." Per-team = a BUFFER of contextual OBSERVATIONS deposited by bots
  (match-state events: "saw item_a spawn", "saw item_b despawn"). NOT ground truth; must
  NOT be swapped for precomputed ground-truth timers — point is INFERENCE UNDER
  INCOMPLETE INFORMATION.
- Only BOTS have beliefs. A belief = egocentric, low-rank, POINTWISE integration of the
  observation buffer at the bot's current V-cell, with (i) temporal contraction toward an
  uninformative prior when stale, (ii) spatial masking by distance.
- V-cells: Voronoi cells; fuse contiguous NAVIGABLE paths until a distance-decay CONTEXT
  MASK (PARALLEL op, NOT recurrent) bounds, two-sided, belief map-area per bot:
  5% <= area <= 15%, at bounded parallel compute.
- Every buffer rebuilt ONLINE every strategy-estimation step. That online rebuild over
  many bots at real cadence = WHY the mesh is non-vacuous even w/ efficient kernels.

## Per-bot mixing = QKV, coupling IS a DPP kernel
- [FIRM shape / OPEN specific transforms] Each bot QUERY = learned projection of TWO
  concatenated inputs: raw Xonotic engine vector ++ egocentric belief integration.
- Instruments carry learned KEY + behavioral VALUE projections.
- [FIRM] Coupling/mixing (the Gram over q,k) IS a DPP KERNEL: its DETERMINANT induces
  DIVERSITY over committed instruments (anti-redundancy).
- [FIRM] Output = VELOCITY on an integrated weight state: dw/dt at team-scale AND
  bot-scale; engine integrates between strategy updates => strategy cadence DECOUPLED
  from game tick.

## Why Xonotic  [FIRM]
- Basic weapons have splash + knockback -> perturb/soft-threaten opponents.
- Knockback PHYSICALLY DISPLACES enemy bodies OUT of a cart's control cylinder ->
  DIRECTLY reduces the opposing presence term in the cart-velocity law.
- => "suppress a cart" achievable by ZONING w/o kills => learned value gets a mechanical,
  SKILL-ORTHOGONAL path to win. Free params should end up ENCODING this.

## Open hypotheses — NOT decided  [OPEN]
- (a) Stackelberg/backward-induction may enter as LEARNED not hand-declared: "leadership"
  = learned readout of team resource concentration (armor/health/rocket-launchers on
  surviving players near a lane); commitment = MOMENTUM in the velocity/weight state;
  followers respond to leader trajectory.
- (b) DPP diversity-against-the-leader = candidate non-dominated-response operator;
  COALITION (pile onto leader's key lane) = complementary LOW-diversity regime of the same
  DPP; REINFORCE prices which.
- (c) Principle "REPRESENT don't EVALUATE": hand-featurize only enough that leadership/
  dominance/swing are representable; push ALL evaluation into learned params calibrated by
  many matches.
- (d) Minimap horizon (fixed scalar vs learned) + whether to ever compute a Grundy/key-
  player REFERENCE value or leave it to REINFORCE — both under review.
- Nim = guiding ANALOGY (throw your team to reverse a leader's lane; position value only
  approximable, and that ambiguity is WHY the solve is heavy). Analogy, NOT a claim of
  exact equivalence.
