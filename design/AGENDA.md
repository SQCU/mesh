# Agenda — the mutable checklist (payload strategy)

`SPECIFICATION.md` is the immutable quote index (what this must be); this file is the
living status of every frontier (what it presently is). Checkmarks mutate; the Record
below only appends.

## Charter (User, vine-polycompiler stratagem, spread here)

> instead of makign a timestamped or historically versioned document, it's worth
> documenting a current agenda in the form of a mutable checklist (changing a
> checkmark from empty to partial to full to partial should require a piece of
> appended record, block quoting something like code or artifacts or algorihtmic
> work in some other document. you know how wiki text works? yeah like that.)

## Provenance law (binding — see SPECIFICATION.md)

Level-1 = transcript block quotes (the only spec). Level-2 = "the user said…" without a
quote (not spec). Level-3 = "the repo says/does/is X" — permitted ONLY when block-quoting
code, an algebra, or a proof; otherwise, by the law, it is a lie.

## Evidence standard

- User block quotes are level-1 authority for what is *required*; they are not evidence
  of what is *done*. Requirements come from quotes; checkmarks come from artifacts.
- A user crash-out about a state is a **discovery obligation**, not a state transition.
  The frontier keeps its mark until discovered evidence (proof, type-systematic claim,
  logged result, artifact) is appended to the Record.
- Admissible evidence for THIS project: a checked proof / algebra; a real run artifact
  under `xonotic/solver/strat/runs/`; a clean `progs.dat` build output; a commit hash of
  landed code. **Results produced against `cartsim` are NOT admissible** (SPEC §13: no
  fake re-simulation).

## Protocol

- States: `[ ]` unattended · `[~]` partial · `[x]` full.
- Any transition, in either direction, requires one appended Record entry (`R<n>`)
  block-quoting code, an artifact, or algorithmic work. No record, no flip.
- Assertions alone (user's or agent's) are never the quoted evidence.
- Record entries are append-only: never edited, deleted, or reordered.

## The checklist

### Spec & method
- [x] `SPECIFICATION.md` — verbatim quote index of the strategy topic (R1)
- [~] Re-ground every design doc to level-1 quotes (papers → level-3) (R2)

### Reward / value / advantage (SPEC §4, §5)
- [~] Reward = sparse at the projected-winner transition; not score, not whole-game RLVR (R3)
- [~] W/L are reward *definitions*; value estimators are LINEAR PROBES on the final IR;
  policy params optimized to increase advantage (R4)
- [ ] Value probes actually read the Gram IR, and advantage optimization actually shapes
  the input projections (SPEC §6, §7) — unverified

### The Gram IR / j-space (SPEC §6, §7, §8)
- [ ] The learned operator is a **Gram + SwiGLU**, NOT softmax attention (R5 — REGRESSED:
  the shipped `relattn.py` is softmax attention)
- [ ] The IR is **wide (≥128d)**, semantically grounded by the value gradient (R5 —
  REGRESSED: current `d_row≈16`)
- [~] DPP kernel (diversity) + velocity-on-integrated-weight update (R6)
- [~] Sampling with L1/L2-toward-logit-0 regularization (R6)

### Observation buffer + spatial belief (SPEC §3)
- [~] Perception-gated observation buffer + egocentric belief; where does spatial mixing
  enter the flow (R7 — belief pipeline exists but the running path re-inlines/bypasses it)

### The QC boundary (SPEC §12)
- [x] matmul = WHAT, stock navmesh = HOW; no policy/navigation in playerbot code (R8)

### Substrate & honesty
- [ ] Delete `cartsim` and every cartsim-derived result (SPEC §13) — REGRESSED: cartsim
  still present, and de18d7a/a9a24fe/97c4bf5 report cartsim numbers (R9)
- [~] Real Game-2 server training at scale; resumability proven (R10)
- [~] Expressivity/dominance proof (aliasing counterexample; the deep-MHA-resnet argument)
  written to the spec (R11)
- [x] Statewise, no recurrence (SPEC §2) — no recurrent/sequential learned model exists (R12)

---

## Record (append-only)

### R1 — 2026-08-31 — SPECIFICATION.md seeded `[x]`
`design/SPECIFICATION.md` committed as a 13-section verbatim quote index (session
`d3ad4328`, timestamps 2026-08-29…31), each normative sentence a user block quote.

### R2 — 2026-08-31 — Doc re-grounding seeded `[~]`
Prior design docs pre-date the provenance law. Contradictions to fix: `MULTISCALE.md` §3
block-quotes *papers* as the spec (violates the law); `rl-training-spec.md` and earlier
docs carry paraphrased normative sentences. Only the quote index governs.

### R3/R4 — 2026-08-31 — Reward/value/advantage `[~]`
Defined by SPEC §4/§5. `rl-training-spec.md` (7e0bb02) and the W/L two-head value
(f238c4d) encode the asymmetric role-rewards + role-gated value probes. Gap: SPEC §5 says
the value estimators are LINEAR PROBES on the final IR — current `value.py` heads are 2-layer
MLPs on a narrow intermediate, not linear probes on a wide Gram IR.

### R5 — 2026-08-31 — Gram IR / j-space `[ ]` REGRESSED
Commit 97c4bf5 shipped `relattn.py` as **softmax attention** — contradicts SPEC §8:

> im pretty sure a gram matrix and a swiglu were described earlier

and the IR is `d_row≈16`, contradicting SPEC §8:

> how wide did you think the hidden states were supposed to be for this? under 128d?

Required: replace softmax attention with the Gram; widen the IR to ≥128d; ensure the Gram
output lands in the IR the probes read (SPEC §7).

### R6 — 2026-08-29 — DPP + velocity + sampling `[~]`
SPEC §9/§10. DPP marginal `diag(K)=1-diag((I+L)^-1)` is real and now differentiable
(97c4bf5); velocity-on-weights + anticipatory switch present; categorical sampling +
L2-to-0 reg present.

### R7 — 2026-08-31 — Observation buffer `[~]`
`buffers.py` (perception-gated) + `featurize.py` (egocentric belief) exist; the running
path re-inlines the belief in `live_belief.py`, leaving `featurize.egocentric_integration`
dead (reported in 97c4bf5). Spatial mixing's place in the flow is unsettled.

### R8 — 2026-08-31 — QC boundary `[x]`
Commit d096ca3: removed the golden-path-node and near-cart-node navigation from QC; adapter
reduced to `navigation_routerating` on entities; `progs.dat` rebuilt clean. Satisfies
SPEC §12.

### R9 — 2026-08-31 — cartsim `[ ]` REGRESSED
SPEC §13 forbids testing on a fake re-simulation. `cartsim.py` and cartsim-derived numbers
in de18d7a, a9a24fe, 97c4bf5 are inadmissible evidence and must be deleted.

### R10 — 2026-08-30 — Real Game-2 training `[~]`
Commit 316f382: resumability PROVEN (two `kill -9`+resume cycles, byte-identical weights,
`runs/resume_proof.jsonl`); 228 real `game2_server` transitions collected across shapes
without resize. At-scale training blocked on freeing `/mesh0`.

### R11 — 2026-08-31 — Expressivity proof `[~]`
The analytic aliasing counterexample (team-only forced-regret 1/2) is in
`runs/aliasing_counterexample.json`; the "softmax ⇒ deep multi-head-attention-resnet"
proof and the "multi-head is central not perf" argument are in transcript, not yet lifted
into the spec.

### R12 — 2026-08-31 — Statewise/no recurrence `[x]`
SPEC §2. No recurrent or sequential learned model exists in `xonotic/solver/strat/`; the
policy is statewise, the hard reasoning is closed-form (`game.py` PW/SUCC).
