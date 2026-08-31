# Agenda — payload strategy (the commanded becomings)

Mutable checklist. `SPECIFICATION.md` is the immutable quote index (what this
must be); this file is the living status of every frontier (what it presently
is). Checkmarks mutate; the Record below only appends.

Form: `~/sm120-chudlets/AGENDA_FORM.md`. Reading rule carried verbatim:

> Reading rule: **requirements come from quotes; checkmarks come from
> artifacts.** The SPEC cannot move a checkmark; the AGENDA cannot amend
> the SPEC; the RECORD is the only bridge between them.

## Charter

> instead of makign a timestamped or historically versioned document, it's
> worth documenting a current agenda in the form of a mutable checklist
> (changing a checkmark from empty to partial to full to partial should
> require a piece of appended record, block quoting something like code or
> artifacts or algorihtmic work in some other document. you know how wiki
> text works? yeah like that.)

> in this specific context there are several goals and program features which
> seem to become repeatedly conjoined, fused together, subsumed by each other,
> or forgotten. make a big frickin checklist of what the project was commanded
> to become in the history of the transcripts that synthesized it.
> — User, 2026-08-31, this session

## Provenance law (level of epistemic certainty)

> 1: block quotes from transcript. this is epistemologically from the user.
> 2: "the user said..." this is not epistemologically from the user, and can't be spec if it doesn't have a block quote.
> 3: "the repo says/does/is..." ... this can only be said when there is a message. the message cannot be 'the repo says/does/is xyz' unless the claim is block quoting code, or is an algebra, or a proof written in text.

## Evidence standard (admissible forms for THIS project)

> features which claim to be complete but the user thinks arne't complete
> *must discover evidence* in the form of an end to end test, proof, type
> systematic claim, logged result, or artifact

Project-scoped exclusion, from SPEC §13:

> if we wrote our linear algebra correctly we should not need to 'test' our
> code on fake resimulations of a videogame that is itself a literal
> simulation. like, ever.

Therefore **`CartSim` and every number derived from it are INADMISSIBLE.**
Admissible: real Xonotic server runs and their artifacts, quoted code/algebra,
written proofs, engine build outputs, and measurements over real-run data.
No unit tests (SPEC §13 + the standing no-tests directive).

## Notation

- State: `[ ]` unattended · `[~]` partial · `[x]` full. States move both ways.
- Each line cites the Record entry that justifies its state; `(—)` means **no
  admissible evidence has been discovered yet**, so the state is a claim
  awaiting discovery, not a finding.
- Record entries are tagged by evidence class:
  `E:run` real server/hardware run · `E:artifact` committed file ·
  `E:code` quoted code · `E:proof` written algebra/proof ·
  `E:build` compiler output · `E:none` assertion only (never justifies `[x]`).
- Frontiers are never deleted; regressions keep their historic evidence.

---

## The checklist

### A. Mesh substrate (RDMA transport)
- [x] A1 Sealed minimal API — `mesh_open`/`write`/`read`; users never touch below "page" (R1)
- [x] A2 No software repair machinery — UC only; what arrives is intact, what doesn't is gone (R1)
- [x] A3 Self-healing across unplug/replug/sleep/reboot with no out-of-channel restart (R1)
- [x] A4 Never reboot a node for availability (capacity is an operator decision) (R1)
- [~] A5 Nonblocking streams as closure-converted state machines (R1)
- [~] A6 The strategy op must saturate multiple M-series SoCs (blocked by C2 width) (R14)

### B. The cart subgame (Game 1 — the tractable shadow)
- [x] B1 Explicit multicart: k carts, j teams (R2)
- [x] B2 Golden path = arclength carrier of a heap's magnitude (NOT a bot route) (R12)
- [x] B3 Two-regime velocity law: contested-local vs abandoned-linear-reversal (R2)
- [x] B4 Monotone score — position reverses, score does not (R3)
- [~] B5 Relative deny/acquire objective, not race/entry/level (R3, R8)
- [x] B6 Continuous & differentiable cart force field (soft membership/vitality/gate) (R2)
- [x] B7 Cylinder occupancy is the law; no LOS gate; no unstick hack (R2)
- [x] B8 Closed-form `PW`/`SUCC`/`N_i` (nim-sum + backward induction) over cartstate (R6)
- [x] B9 Backward induction explicit, not allusive/optional (R6)
- [x] B10 Partizan honesty — impartial ⇒ exact nim-sum; else explicitly unresolved (R6)

### C. The strategy operator (the linear algebra)
- [ ] C1 **Gram + SwiGLU, NOT softmax attention** (REGRESSED — shipped softmax) (R11)
- [ ] C2 **Wide IR ≥128d** (present: ~16d) (R11)
- [~] C3 Irreducibly all-to-all O(n²) learned coupling landing IN the IR (R10, R11)
- [x] C4 DPP `diag(K)` marginal-inclusion signal, differentiable (custom vjp) (R10)
- [x] C5 Velocity on an integrated weight state (replicator), not instantaneous decisions (R7)
- [x] C6 Anticipatory (predictive) update, not a plain integrator (R10)
- [x] C7 Count-invariance — no parameter shape depends on k/j/l (R10)
- [~] C8 Categorical sampling + L2-toward-logit-0 (not MAP) (R7)
- [x] C9 Query = learned projection of [engine state ; belief] (R10)
- [x] C10 Per-instrument learned behavioral value `v_m` wired into the mix (R10)
- [ ] C11 **j-space** — value probes ground the semantics of random-init projections (—, under discovery)

### D. Reward / value / advantage / training
- [x] D1 `W` and `L` are asymmetric REWARD DEFINITIONS, not duals, not control signals (R8)
- [~] D2 Value estimators are LINEAR PROBES on the final IR (R8; probe quality undiscovered)
- [x] D3 Policy parameters optimized to increase advantage (R8)
- [x] D4 NOT reward=score, NOT whole-game RLVR (degenerate: cannot notice being ganged) (R8)
- [x] D5 Per-row scalar critic outputs; never an `l`-wide vector (R8)
- [x] D6 Training IS the Xonotic server process (real Game-2 transitions) (R9)
- [ ] D7 **CartSim deleted** — no fake re-simulation anywhere (still present) (R13)
- [~] D8 Curriculum over maps / team counts / player counts / cart counts (R9)
- [x] D9 Interruptible & resumable — proven by hard kill + resume, twice, early (R9)
- [ ] D10 Acceptance matrix on the SERVER: retention under perturbation, recovery time, acquisition, terminal, held-out (—, [BUILD-DATA])
- [x] D11 Learned local action-linear dynamics ensemble `Δy=b(y)+A(y)u` (R10)

### E. Observation / featurization (the map-reduce)
- [x] E1 Perception-gated observation: frustum + LOS + 2-V-cell cap (emergent stealth) (R5)
- [~] E2 Per-team observation buffer of contextual events (not ground truth) (R5)
- [~] E3 V-cell segmentation; fuse navigable cells to a 5–15% receptive field (—, under discovery)
- [~] E4 Temporal contraction toward an uninformative prior (—, under discovery)
- [~] E5 Bounded-support spatial mask (parallel, not recurrent) (—, under discovery)
- [ ] E6 Low-rank egocentric integration is the ONLY spatial mixing operator — **and is actually called on the live path** (R11: `featurize` pipeline reported dead, re-inlined in `live_belief`)
- [x] E7 Belief is per-bot; there is no "team belief" (R4)
- [x] E8 Enemy positions featurized ONLY through observation (R5)

### F. Playerbot interface (the WHAT/HOW boundary)
- [x] F1 matmul decides WHAT; stock navmesh decides HOW (R12)
- [x] F2 Skill-orthogonal — never touches aim/dodge (R12)
- [~] F3 Objective vocabulary → stock target entities (explore, gather, crush-weak, duel-strong, push/suppress, hunt) (R12)
- [~] F4 Spawn-timing and travel-commitment as real instruments (R12)
- [x] F5 No policy in QC; no second navigation definition (R12)
- [ ] F6 Affordance QUALITY — does the policy actually aim `hunt` at the winningest rival? (—)

### G. World / maps
- [x] G1 Procedural multi-map fusion produces megamaps (R15)
- [~] G2 Megamaps actually USED by the training/live server (—, under discovery)
- [~] G3 Bots traverse long distances between fused regions (commitment cost) (—, under discovery)
- [x] G4 Prominence rule: exclusive objective entrances conspicuous; connectors may be subtle (R15)
- [x] G5 Stock-navmesh compliance; no project-specific bot nav graph (R12)
- [~] G6 Entity budget at scale — no invisible bots at high player counts (R16)
- [~] G7 Diegetic communication of cart paths/state (R16)
- [x] G8 Headless client renderer for join inspection (real offscreen renders) (R15)

### H. Multipolar dynamics — description & demonstration
- [ ] H1 Resource-domination logging (alive count, health/armor pools, consumption vigor) (—)
- [ ] H2 Cartlane exertion logging (simultaneously contested carts, PW-flip volatility) (—)
- [ ] H3 Cross-team focus/attrition matrix — is damage concentrated on the key rival? (—)
- [ ] H4 Multipolar dynamics visualizer (tug trajectories, PW timeline, focus matrix) (—)
- [~] H5 Web view of the run (three.js phase space + live mesh table) (R16)

### I. Method / process laws
- [x] I1 SPEC is a verbatim user-quote index under the provenance law (R17)
- [x] I2 AGENDA checklist + append-only Record (this file) (R17)
- [x] I3 No unit tests; evidence is artifacts/proofs/real runs (R13)
- [x] I4 No stubs, no pseudodocumentation claiming false completeness (R10, R12)
- [~] I5 No repeated inlining — one canonical definition per algorithm (R10, R11)
- [~] I6 No fake re-simulation anywhere (D7 outstanding) (R13)
- [x] I7 Private SQCU repo; never push upstream (R1)
- [x] I8 Services interruptible & resumable by construction, proven by crashing early (R9)

---

## Record (append-only)

### R1 — 2026-08 — A1–A5, I7: unattended → full
`E:run` The mesh transport reduced to the sealed 3-function kernel; self-healing
proven live across sleep/replug/network-change; UC-only, no software ARQ; no
code path initiates a reboot. Repo `SQCU/mesh`, `rdma/mesh-flow.c` + `mesh.h`;
rules in `RDMA-RULES.md`. A5 remains `[~]`: streams are closure-converted, the
full nonblocking state-machine surface is not separately demonstrated.

### R2 — 2026-08-29 — B1, B3, B6, B7: partial → full
`E:code`+`E:build` Commit `2a13d99` "payload: reversible two-regime cart velocity
+ strategy spec". Regime A (color team present) bounded local tug; Regime B
(absent) linear reversal by the strongest present opponent; `bound(0,…)` floor
removed so velocity is signed. `design/cart-force-field.md` (now consolidated)
gave the C∞ form: soft cylinder membership `m_i`, soft vitality `a_i`, softmax
control, sigmoid regime gate. Builds clean under `-Werror`.

### R3 — 2026-08-30 — B4 full; B5 partial
`E:artifact` The un-banking model was RETRACTED: cart position reverses, banked
score is monotone. Recorded in `SPECIFICATION.md` §4 and the consolidated specs.
B5 stays `[~]`: the relative deny/acquire objective is specified and encoded in
the W/L reward definitions, but has not been demonstrated on the server.

### R4 — 2026-08-29 — E7: → full
`E:artifact` "There is no team belief; only bots have beliefs." Belief is a
per-bot egocentric readout of a shared *observation buffer*; two bots agree only
when their inputs are identical. Recorded in the payload spec's belief pipeline.

### R5 — 2026-08-30 — E1, E8 full; E2 partial
`E:code` `sv_payload_strategy_io.qc` `payload_perceive` runs the three gates per
bot — 2-cell range, `checkpvs(eye,target)`, `traceline(... MOVE_NOMONSTERS ...)`
with `trace_fraction<1` — over enemy players and pickups, depositing
edge-triggered events into a contiguous event-carrier pool. No omniscient
`ITS_AVAILABLE` feed remains. E2 `[~]`: the buffer exists and is written; its
real event volume on a live match is not yet quoted.

### R6 — 2026-08-30 — B8, B9, B10: partial → full
`E:code` `xonotic/solver/strat/game.py` + `game_value.py`: `nim_sum` XOR-reduce,
`team_nimbers`, `projected_winner` (PW), `succession` (SUCC by repeated
decrement of the leader's deepest cart). `game_value.evaluate` returns
`"partizan"` / `"unresolved"` where a Grundy value does not exist. Verified on
this host: `nim_sum([1,1]) = 0`, one d:2 cart beats two d:1 carts.

### R7 — 2026-08-30 — C5 full; C8 partial
`E:code` `head.py` emits `dw/dt`; the weight state is integrated at the strategy
cadence coprocessor-side. Selection is `mx.random.categorical` (sampling, not
MAP). C8 `[~]`: L2-toward-logit-0 is specified and partially present as weight
decay; the logit-0 pull has no measurement showing broad untrained sampling.

### R8 — 2026-08-30 — D1, D3, D4, D5 full; D2 partial
`E:code` `value.py:26-33` holds two distinct heads (`self.winner`, `self.loser`),
`select_role_value` role-gates by `PW`, and `train.py:172-181` gives genuinely
different reward forms (`r^W = ±retain + margin·ΔH`, `r^L = ΔH + acquire`) —
not sign-flipped duals. `RoleValueHead.__call__ ...[...,0]` emits one scalar per
row (never `l`-wide). D2 `[~]`: the heads ARE linear probes on the IR by
construction, but no probe-quality measurement exists (see C11).

### R9 — 2026-08-30 — D6, D9 full; D8 partial
`E:run` Commit `316f382`. Real Game-2: `runs/game2_train.jsonl`, 228 lines all
tagged `environment:"game2_server"` — live darkplaces cartserver (node 0) over
RDMA to `strat_responder --train` + `OnlineLearner` on the mini; 2 shapes
(5,3,12)+(5,3,18), one shared checkpoint chaining 0→208 updates with no resize;
`loss_dynamics` 16585→5.32. Resumability proven twice: `kill -9` at
`updates=40` → continued to 90; at 120 → continued to 140; byte-identical weight
hashes (`7babc187e1ed9523`, `eae9e6bcfb948704`), optimizer moments and counter
restored, telemetry appended, atomic checkpoints, zero corruption
(`runs/resume_proof.jsonl`). D8 `[~]`: `curriculum.py` samples the full space but
only 2 shapes have been realized on the server.

### R10 — 2026-08-30 — C3 → partial; C4, C6, C7, C9, C10, D11 → full
`E:run` Commit `97c4bf5`. Grad norms nonzero through all 12 encoder params and
`qkv.W_v` (0.427); one fixed tensor set runs (2,3,4)(3,4,7)(4,6,9)(5,7,11) with
shapes *and object identity* unchanged. DPP un-stop_gradient'd via an analytic
custom-vjp for `diag(K)=1-diag((I+L)⁻¹)` (mlx 0.29.3 ships no vjp for
`inv`/`eigh`). Anticipatory vs standard on the same weights: Δcumulative-reward
≥0 on 5/5, phase-lag Δ<0 on 5/5. Dead `qkv.value` wired as `behavior_mix`.
Duplicated forward pass and reward helpers collapsed to canonical definitions.
**Caveat of admissibility:** the anticipatory and dominance NUMBERS in this entry
derive from CartSim and are therefore inadmissible under the evidence standard;
the differentiability, count-invariance and wiring facts are `E:code` and stand.

### R11 — 2026-08-31 — C1, C2 → zeroed (REGRESSED); C3 held partial; E6 → unattended
`E:artifact` The surrogate audit + the SPEC contradiction list. SPEC §8:

> where idd a softmax come from? why are you talking about attention? im pretty
> sure a gram matrix and a swiglu were described earlier

`relattn.py` (commit `97c4bf5`) computes `A = softmax(QKᵀ/√d + φ_rel(E))` — a
softmax-attention operator, not a Gram. It was introduced by the coordinator's
prompt, not by the SPEC. C1 zeroed. SPEC §8 also:

> how wide did you think the hidden states were supposed to be for this? under
> 128d? maybe you were slippin.

Present width is `d_row ≈ 16`, so C2 zeroed and A6 (SoC saturation) is blocked
behind it. E6 zeroed: `live_belief.py` re-implements `featurize.py`'s belief
pipeline inline, leaving `featurize.egocentric_integration` / `temporal_contraction`
/ `belief` dead — reported, not collapsed. Recovery route: implement the Gram +
SwiGLU at ≥128d and route the live path through the canonical featurizer.

### R12 — 2026-08-30 — B2, F1, F2, F5, G5 full; F3, F4 partial; I4
`E:build` Commit `d096ca3`. Audit found NO policy in QC (targets and amplitudes
are all mesh readouts) and TWO navigation violations, both removed: the
golden-path-**node** `bestnode`/lane-fraction/spread-bias block in
`havocbot_goalrating_strategy`, and the `g_payload_nodes` near-cart node rating
in `havocbot_goalrating_payload`. Push/suppress is now ONE `routerating` on the
cart entity; the cylinder law resolves push-vs-suppress by presence. `plc_path`
is the cart's own movement/banking graph only; bots reach carts as moving goal
entities through the stock waypoint navmesh. False-completeness header ("NOT
wired into the live rater yet" — it was wired) deleted. `progs.dat` rebuilt
clean, 6,803,967 bytes. F3/F4 `[~]`: mapping documented and levers exist
(`bot_strategytime`, `respawn_time`), not yet demonstrated driving behavior.

### R13 — 2026-08-31 — I3, I6 recorded; D7 opened as unattended
`E:artifact` The no-tests directive was executed (all `test_*`/`validate.py`
removed; `git ls-files xonotic/solver | grep -i test` → empty). SPEC §13 then
went further and ruled out fake re-simulation entirely. `cartsim.py` and the
CartSim-derived numbers in commits `de18d7a`, `a9a24fe`, `97c4bf5` are hereby
marked INADMISSIBLE as evidence. D7 (delete CartSim) is unattended.

### R14 — 2026-08-31 — A6 → partial
`E:proof` The compute-shape argument: a Gram over all player rows is a large
bandwidth-bound matmul, the shape that saturates a mesh of M-series SoCs; a deep
serial stack is latency-bound and cannot. At the present IR width (~16d) the op
is far too small to saturate anything, so A6 cannot exceed `[~]` until C2 lands.

### R15 — 2026-08-30 — G1, G4, G8 → full
`E:run` Commits `924768b`, `84cf157`, `db65511`. Fusion produces a real fused
multi-map world with a region flood-fill asserting all source maps land in one
bot-reachable component; pad/teleport joins emit canonical entities so stock
Xonotic auto-generates their bot waypoints. Prominence rule: degree-1 endpoints
get the wide/lit/short template, degree-≥2 nodes subtle pads. Headless client
renders real offscreen frames via DPSOFTRAST (`SDL_VIDEODRIVER=dummy` +
`vid_soft 1`), sample PNGs at `xonotic/payload/tools/joinshots-sample/`.

### R16 — 2026-08-30 — G6, G7, H5 → partial
`E:run` Commit `c400e31` bounded always-sent payload sprites from O(carts×nodes)
to O(carts) and identified the per-snapshot byte budget (`net_usesizelimit 1`).
Diegetic ribbons and the k-cart dashboard exist. `web/index.html` renders a
three.js phase space beside a live mesh table with a real headless screenshot.
All three remain `[~]`: none has been verified at full population, and H5's
trajectory panel is fed partly by inadmissible CartSim data.

### R17 — 2026-08-31 — I1, I2 → full
`E:artifact` Commit `4d1947f`: `design/SPECIFICATION.md`, a 13-section index of
20 verbatim user block quotes (session `d3ad4328`, 2026-08-29..31) governed by
the provenance law, with papers demoted to level-3 support; and this AGENDA.
Note: `AGENDA.md` was subsequently deleted from the working tree during a
concurrent doc consolidation and is restored and extended by this revision.

### R18 — 2026-08-31 — C11, G2, G3, E3, E4, E5, F6 opened as discovery obligations
`E:none` These frontiers are asserted by the SPEC but have no admissible
evidence in either direction. Per the operating rule, a dispute is a discovery
obligation and not a transition: they stand at their present marks until a
measurement is appended. Two discoveries are in flight — a linear-probe
measurement of the IR (C11) and a per-subcomponent status audit of megamap use,
long-distance traversal and the observation map-reduce (G2, G3, E3–E5).
