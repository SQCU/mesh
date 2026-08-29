---
name: DPP mixing + intercentrality overlay — dissolved regime fork, diag(K) head, leader-as-readout
description: Normalized companion to dpp-mixing-and-overlay.md. Resolves the lit-review fork (symmetric PSD DPP = repulsion only): REJECT the nonsymmetric-DPP/second-head fork; kernel emits an HONEST diversity signal, a LEARNED operator maps signal->behavior. Mixing head = [ diag(K)=marginal inclusion vector, NOT det ; raw appetite b ] -> RMSNorm -> SwiGLU -> dw/dt velocity, REINFORCE-trained; SwiGLU GATE IS the diversify/concentrate switch (pile-on = one gated output, not a bolt-on). Multiscale intercentrality overlay = argmax over (I-a*kappa)^-1 on the Gram the solver already forms -> Bonacich intensities + swing, team-scale AND bot-scale; "leader" admissible ONLY as this argmax (derived readout, NOT a primitive/character). dw/dt integrate = forward-Euler => emit cadence = stability param w/ chaos threshold. OPEN: whether the flow CONSUMES the overlay asymmetrically (=Stackelberg) vs diagnostic; instantaneous-w vs time-average.
type: project
metadata:
  node_type: memory
  register: memory-hook (frontmatter name/description/type + compressed normalized bullets, matching strategy-layers-notes.md); modality markers [FIRM]/[OPEN] carried through because the owner is strict. Companion prose w/ full reasoning + citations: design/dpp-mixing-and-overlay.md (intent where code disagrees). Resolves design/lit/gram-dpp-velocity-stackelberg.md; extends design/strategy-layers-and-modality.md (layer 3).
---

CONVENTION NOTE (read first): owner's auto-memory hook style — YAML frontmatter
(name/description/type) over terse normalized bullets — matching strategy-layers-notes.md,
NOT the discursive design/*.md prose. Firm decisions carry [FIRM]; undecided ones [OPEN].
These are DESIGN commitments resolving the lit review's SETTLED/INFERENCE/OPEN into
FIRM/OPEN; a result can be SETTLED in the review yet feed an OPEN choice here. Companion
prose + citations: design/dpp-mixing-and-overlay.md (that doc is intent where code disagrees).

## The false fork, dissolved  [FIRM]
- Lit review fork: symmetric PSD DPP models REPULSION ONLY -> to hold the pile-on/coalition
  (positive-correlation) regime in the SAME kernel, seemingly need (1) nonsymmetric DPP (NDPP)
  or (2) a separate additive concentration head. REJECTED as framed.
- Rests on a false premise: that the KERNEL must MEAN both regimes. It must not.
- Resolution: DPP kernel's job = emit an HONEST diversity SIGNAL; mapping signal->behavior =
  a LEARNED OPERATOR's job (downstream). Contorting the kernel (NDPP) to "mean" both = the
  rejected fake/larpy move (representational job dressed as a kernel property). NDPP NOT
  adopted; kernel stays symmetric/repulsion-only. Option (2) was the right instinct stated
  clumsily -> concentration is one of the head's gated outputs (section below), not a bolt-on.

## Mixing architecture  [FIRM]
- DPP intermediate = per-instrument MARGINAL INCLUSION VECTOR diag(K), K = L (I+L)^-1 (marginal
  kernel of the L-ensemble). One repulsion-shaped number per instrument = "how much it belongs
  in a diverse selection." RETRACTS "emit the scalar determinant": det = one number for a whole
  subset ("can't be two things"), folding the Gram to it throws away per-instrument structure.
  diag(K) still can't-be-two-things per instrument yet carries signal for ALL at once. (det still
  trains L via its gradient; it is just not the per-step message to the head.)
- Head: dw/dt = Head( [ diag(K) ++ b ] ), b = raw appetite/quality (the DPP q_i). Head =
  RMSNorm -> SwiGLU. Nothing beyond RMSNorm + SwiGLU required.
- The SwiGLU GATE IS the regime switch (not a 2nd kernel, not a 2nd head): turns the single
  repulsion-shaped diag(K) into either/both of {diversify=spread, concentrate=pile-on/coalition}.
  Gate reads high SHARED appetite concentrated on one instrument -> opens concentration path;
  else passes diversity signal. Substitutes-vs-complements dichotomy lives HERE = a learned gate
  reading appetite alongside the honest signal, NOT the sign of a kernel entry. Pile-on = ONE of
  the head's two gated outputs, NOT a parallel additive term.
- Output = VELOCITY dw/dt on the integrated weight state; trained by REINFORCE (policy-gradient
  outer loss, NOT DPP likelihood). Two substantive firm requirements = expose diag(K) not the
  det, and emit a velocity not a snapshot. RMSNorm->SwiGLU firm; projection widths = candidate.

## Multiscale intercentrality overlay  [FIRM computable / OPEN consumed]
- Closed-form readout over kappa (the kappa = V Gm V^T Gram the solver already forms, same
  second-moment matrix the DPP reads): intensities = (I - a*kappa)^-1 * 1 (Bonacich, a=coupling
  strength); swing = argmax over nodes of the aggregate-intensity DROP on removal. Swing is NOT
  most-Bonacich-central -> it's the INTERCENTRALITY argmax (credits a node's contribution to
  OTHERS' centralities too), from the same inverse. One linear solve. [FIRM computable]
- MULTISCALE by construction: kappa over TEAM-scale embeddings -> swing across teams/lanes;
  kappa over BOT-scale embeddings -> swing within a team. One operator, nested scales (matches
  team-scale + bot-scale weight states).
- ONTOLOGY DISCIPLINE [FIRM]: "leader" admissible ONLY as the intercentrality argmax = a
  DERIVED, closed-form READOUT of state. NOT a primitive entity in featurization, NOT a per-agent
  "character"/role, NOT named by any feature vector. Corrects earlier error (leadership smuggled
  in as an imagined tracked entity). Every downstream "leader/swing/key player" MUST resolve to
  this readout and nothing else.
- [OPEN] Whether the velocity flow CONSUMES the overlay: (i) DIAGNOSTIC only (head emits dw/dt
  symmetrically, ignores who the swing is), vs (ii) ASYMMETRIC — swing's weight-state commits/
  moves FIRST, others' dw/dt condition on it. Only (ii) earns "Stackelberg dynamics." Overlay is
  settled-computable; its CONSUMPTION is the open modeling choice (naming it now = the
  illocutionary-vs-specified error the project forbids).

## Cadence / stability  [FIRM context from lit review]
- w += (dw/dt)*Delta = forward-Euler on a replicator flow => emit cadence = step size Delta = a
  STABILITY parameter w/ a discrete-time CHAOS threshold (discrete replicator chaotic where the
  continuous flow is benign; larger Delta / lower cadence -> instability). NOT a free scheduling
  knob. [FIRM]
- [OPEN] Zero-sum/cyclic games ORBIT (only the time-average converges, instantaneous w cycles) =>
  trust instantaneous w vs a running average is undecided; turns on whether our game is monotone
  enough for POINTWISE (not just time-averaged) convergence (lit review leaves open).

## Firm vs open (roll-up)
- FIRM (do not reopen w/o owner): fork dissolved (honest symmetric kernel + learned mixing, no
  NDPP); intermediate = diag(K) not det; head = RMSNorm->SwiGLU emitting REINFORCE-trained dw/dt,
  SwiGLU gate = the diversify/concentrate switch; intercentrality overlay computable in one solve
  + multiscale; "leader" = only ever the overlay argmax.
- OPEN: overlay consumed asymmetrically (=Stackelberg) vs diagnostic; instantaneous w vs time-
  average (pending monotonicity); coupling strength a + which scales build kappa; head projection
  widths.
