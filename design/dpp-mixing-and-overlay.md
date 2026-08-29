# DPP mixing, the multiscale overlay, and the dissolved regime fork

This records a set of now-firm linear-algebra decisions for the strategy operator (layer 3
of `strategy-layers-and-modality.md`), taken in response to the literature review in
`lit/gram-dpp-velocity-stackelberg.md`. That review surfaced a sharp-looking fork — a
symmetric PSD DPP models repulsion only, so representing the pile-on/coalition regime in the
*same* kernel appears to demand either a nonsymmetric DPP or a bolted-on concentration head.
This document resolves that fork and fixes the shape of the mixing head, the multiscale
readout that reads leadership off the coupling matrix, and the ontology discipline that keeps
"leader" from becoming an imagined entity. It does NOT restate the review; it cites it and
records what was decided. Where this document and the code disagree, this document is the
intent and the code is the bug.

Modality markers are as in `strategy-layers-and-modality.md`:

  [FIRM]  a decision reached and committed; the code answers to it.
  [OPEN]  a hypothesis named but NOT decided; may change entirely.

The review used its own [SETTLED]/[INFERENCE]/[OPEN] line for what the literature does and
does not establish; here [FIRM]/[OPEN] are about what THIS DESIGN has committed. A result can
be [SETTLED] in the review and still feed an [OPEN] design choice, and vice versa. The owner
is strict about the line; preserve the markers.

## 1. The false fork and its dissolution  [FIRM]

The review's Part D closed on a fork: to let REINFORCE switch between the diversify (spread)
and the pile-on (coalition) regimes inside one learned object, you appear to need either
(1) a nonsymmetric DPP (NDPP), whose dropped symmetry constraint can carry positive
correlation/attraction as well as repulsion, or (2) a separate additive concentration head
competing with a diversity-only DPP. A symmetric PSD DPP models repulsion ONLY — that part is
settled and is not in dispute here.

The fork is REJECTED as framed. It rests on an assumption that is not ours: that the DPP
kernel must itself *mean* both regimes — that the kernel is the thing which decides
diversify-vs-concentrate. It does not, and it should not be made to. The DPP kernel's job is
to emit an honest diversity SIGNAL. Mapping that signal to behavior is a LEARNED OPERATOR's
job, downstream of the kernel. Contorting the kernel into a nonsymmetric object so that a
single matrix can "represent both regimes" is the rejected fake/larpy move: it dresses a
representational job up as a kernel property to make the architecture *sound* like it captures
both regimes, at the cost of a harder-to-learn, harder-to-interpret object. We do the thing
that works instead: an honest, symmetric, repulsion-only kernel that emits a clean diversity
signal, and a learned mixing head that turns that signal into behavior. NDPP is not adopted.
The "separate additive concentration head" of option (2) was the right instinct stated
clumsily — see section 2; concentration is one of the head's gated outputs, not a parallel
bolt-on.

## 2. The mixing architecture  [FIRM]

**The DPP intermediate is a marginal inclusion vector, NOT the determinant.**  [FIRM]  The
object handed downstream from the DPP is the per-instrument MARGINAL INCLUSION VECTOR

   ```
   diag(K),   K = L (I + L)^-1
   ```

where `L` is the L-ensemble kernel (the quality/diversity Gram of the review's Part A) and `K`
is its marginal kernel. `K_ii` is the marginal probability that instrument `i` appears in a
sample from the DPP: one repulsion-shaped number per instrument answering "how much does this
instrument belong in a diverse selection." This RETRACTS the earlier framing that made the
determinant `det(L_Y)` the operative quantity emitted downstream. The determinant is a single
scalar for a whole subset — it "can't be two things," and folding the Gram down to it throws
away exactly the per-instrument structure the head needs. `diag(K)` is the indicator that
still can't be two things per instrument, yet carries the relevant signal for every instrument
at once. (The determinant remains the object whose gradient trains `L`; it is not the
per-step message to the head.)

**A learned gated head maps the signal to weight velocities.**  [FIRM]  The head takes the
diversity signal concatenated with the raw appetite/quality and emits per-instrument weight
velocities:

   ```
   dw/dt = Head( [ diag(K)  ++  b ] ),   b = raw appetite / quality (the DPP quality terms q_i)
   Head  = RMSNorm -> SwiGLU
   ```

Nothing beyond RMSNorm followed by SwiGLU is required. The head is deliberately small; its
expressive burden is only to route one repulsion-shaped signal into one of two behaviors under
a learned gate.

**The SwiGLU gate is the regime switch.**  [FIRM]  The SwiGLU GATE — not a second kernel, not
a second head — is the mechanism that turns the single repulsion-shaped `diag(K)` into either
or both of {diversify (spread), concentrate (pile-on/coalition)}. When the gate reads high
SHARED appetite concentrated on one instrument (many bots' `b` mass pointing at the same
instrument), it opens the concentration path and the diversity signal is gated down; otherwise
it passes the diversity signal through as spread. This is where the substitutes-vs-complements
dichotomy of the review's Part D actually lives in the architecture: not as the sign of a
kernel entry, but as the state of a learned gate reading appetite alongside the honest
repulsion signal. The pile-on regime is therefore ONE of the head's two gated outputs, not a
bolted-on parallel term — which is what the earlier "separate additive concentration head"
phrasing was reaching for.

**The output is a velocity, trained by REINFORCE.**  [FIRM]  The head output is a VELOCITY
`dw/dt` on the integrated weight state (consistent with layer 3's velocity emission), trained
by REINFORCE — the policy-gradient outer loss, not the DPP likelihood. The two substantive
requirements, and the whole of what is firm about the head's interface: expose `diag(K)` (the
marginal inclusion vector) rather than the scalar determinant, and emit a velocity rather than
an allocation snapshot. RMSNorm -> SwiGLU is the committed head; the exact projection widths
are candidate choices, not fixed.

## 3. The multiscale overlay: intercentrality as a derived readout

**[FIRM that it is computable; OPEN whether the flow consumes it.]**

**The closed-form readout.**  [FIRM computable]  The solver already forms a Gram/coupling
matrix `kappa` (the `kappa = V Gm V^T` object of `strategy-qkv.md`, the same second-moment
matrix the DPP reads for diversity). A single linear solve over it,

   ```
   intensities  =  (I - a*kappa)^-1 * 1          (Bonacich centrality; a = coupling strength)
   swing        =  argmax over nodes of the drop in aggregate intensity when that node is removed
   ```

gives per-node action intensities and the SWING / key instrument in closed form. This is the
key-player intercentrality result of the review's Part C (Ballester-Calvo-Armengol-Zenou):
the swing is NOT the most Bonacich-central node; it is the argmax of intercentrality, which
also credits a node for its contribution to *others'* centralities, and it comes from the same
inverse `(I - a*kappa)^-1`. It is one linear solve over a matrix the solver already
materializes. That it is computable is firm.

**Multiscale by construction.**  [FIRM]  The same operator nests across granularities by
choosing what `kappa` is built over:

   ```
   kappa over team-scale embeddings  ->  swing ACROSS teams / lanes
   kappa over bot-scale embeddings   ->  swing WITHIN a team
   ```

One operator, nested scales — matching the team-scale and bot-scale weight states of layer 3.

**Ontology discipline — record this explicitly.**  [FIRM]  "Leader" is admissible ONLY as the
argmax of the intercentrality overlay: a DERIVED, closed-form READOUT of state. It is NOT a
primitive entity in the featurization, NOT a per-agent "character" or role a bot carries, and
NOT anything the feature vectors name. This corrects an earlier error in which leadership was
smuggled in as an imagined entity that the model was to track. There is no leader object;
there is only the argmax of a solve over state. Whenever "leader," "swing," or "key player"
appears anywhere downstream, it MUST resolve to this readout and to nothing else. (This is the
same discipline layer 3 already applies to "leadership as a learned readout of state," now
pinned to a specific closed form.)

**[OPEN] Whether the flow consumes the overlay.**  The overlay is settled-computable; whether
the velocity flow CONSUMES it is the open modeling choice. Two live possibilities:

- The overlay stays a DIAGNOSTIC readout — computed, logged, available, but the `dw/dt` of
  every instrument is emitted symmetrically by the section-2 head without reference to who the
  swing is.
- The flow consumes the overlay ASYMMETRICALLY: the identified swing's weight-state commits /
  moves first, and every other instrument's `dw/dt` conditions on that committed motion. Only
  this asymmetric consumption — a leader committing and followers best-responding to the
  commitment — would earn the word "Stackelberg dynamics" for the flow. Naming it now would be
  the illocutionary-vs-specified error the project forbids; the structure is not committed.

The computability is firm; the consumption is [OPEN].

## 4. Cadence and stability  [FIRM context from the lit review]

The velocity is integrated as `w += (dw/dt) * Delta`. This is forward-Euler on a replicator
flow (review Part B), which fixes two things about cadence:

- **Emit cadence is a stability parameter, not a scheduling knob.**  [FIRM context]  `Delta`,
  set by the emit cadence, is the forward-Euler step size. Discrete-time replicator dynamics
  can be chaotic where the continuous flow is benign, and larger `Delta` (lower emit cadence)
  pushes toward that instability. So the cadence carries a discrete-time chaos threshold; it is
  chosen against stability, not freely for scheduling convenience.

- **Instantaneous vs time-averaged weight.**  [OPEN]  Zero-sum / cyclic games orbit the
  equilibrium under the replicator flow — only the TIME-AVERAGE of the trajectory converges,
  the instantaneous `w` cycles. Whether the design trusts the instantaneous `w` or a running
  average of it is therefore undecided, and turns on whether our game is monotone enough for
  pointwise (not just time-averaged) convergence — which the review leaves open. Recorded here
  as [OPEN] so the cadence and the trust-the-instant question are not conflated with the firm
  step-size fact above.

## 5. What this leaves open

For legibility, the [OPEN] items above, collected — none are committed:

- whether the velocity flow consumes the intercentrality overlay asymmetrically (would earn
  "Stackelberg dynamics") or the overlay stays diagnostic (section 3);
- whether to trust the instantaneous `w` or a running time-average, pending whether the game
  is monotone enough for pointwise convergence (section 4);
- the coupling strength `a` and the exact scales at which `kappa` is built (section 3), and the
  head's projection widths (section 2) — candidate choices, not fixed.

Firm, by contrast, and not to be reopened without the owner: the fork is dissolved (honest
symmetric kernel + learned mixing, no NDPP); the DPP intermediate is `diag(K)` not the
determinant; the head is RMSNorm -> SwiGLU emitting a REINFORCE-trained velocity with the
SwiGLU gate as the diversify/concentrate switch; the intercentrality overlay is computable in
one solve and multiscale by construction; and "leader" is only ever the overlay's argmax.
