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
- [~] A6 The strategy op must saturate multiple M-series SoCs — C2 width landed; current multi-host saturation/necessity remains to be measured (R14, R24, R41)
- [x] A7 Reusable sealed mesh API incorporated into real high-teamcount game projects, with sustained integration experience distinct from solver completeness (R41)

### B. The cart subgame (Game 1 — the tractable shadow)
- [x] B1 Explicit multicart: k carts, j teams (R2)
- [x] B2 Golden path = arclength carrier of a heap's magnitude (NOT a bot route) (R12)
- [x] B3 Two-regime velocity law: contested-local vs abandoned-linear-reversal (R2)
- [x] B4 Monotone score — position reverses, score does not (R3)
- [x] B5 Relative deny/acquire objective, not race/entry/level (R3, R8, R32)
- [x] B6 Continuous & differentiable cart force field (soft membership/vitality/gate) (R2)
- [x] B7 Cylinder occupancy is the law; no LOS gate; no unstick hack (R2)
- [x] B8 Closed-form `PW`/`SUCC`/`N_i` — nim-sum now DERIVED by backward induction, not asserted (R6, R25)
- [x] B9 Backward induction explicit, not allusive/optional (R6)
- [x] B10 Partizan honesty — impartial ⇒ exact nim-sum; else explicitly unresolved (R6)
- [x] B11 The CGT evaluator RESOLVES on real server states — 0/228 → 228/228 (R25)

### C. The strategy operator (the linear algebra)
- [x] C1 **Gram-matrix fusion + SwiGLU, NOT softmax attention** (R24)
- [x] C2 **Wide IR ≥128d** (R24)
- [x] C3 Irreducibly all-to-all O(n²) learned coupling landing IN the IR (R24)
- [x] C4 DPP `diag(K)` marginal-inclusion signal, differentiable (custom vjp) (R10)
- [x] C5 Velocity on an integrated weight state (replicator), not instantaneous decisions (R7)
- [x] C6 Anticipatory (predictive) update, not a plain integrator (R10)
- [x] C7 Count-invariance — no parameter shape depends on k/j/l (R10)
- [~] C8 Categorical sampling + L2-toward-logit-0 (not MAP) (R7)
- [x] C9 Query = learned projection of [engine state ; belief] (R10)
- [x] C10 Per-instrument learned behavioral value `v_m` wired into the mix (R10)
- [~] C11 **j-space** — trained IR now beats random-init/random-projection, but collapses with training; data-bound (R24)
- [x] C12 Value heads are LINEAR probes on the IR (R24)

### D. Reward / value / advantage / training
- [x] D1 `W` and `L` are asymmetric REWARD DEFINITIONS, not duals, not control signals (R8, R32)
- [x] D2 Value estimators are LINEAR PROBES on the final IR (R24)
- [x] D3 Policy parameters optimized to increase role-selected advantage (R8, R32)
- [x] D4 NOT reward=score, NOT whole-game RLVR (degenerate: cannot notice being ganged) (R8)
- [x] D5 Shared `128→1` W/L probes yield `l` row scalars; never one global scalar or `l` outputs per row (R8, R32)
- [x] D6 Training IS the Xonotic server process (real Game-2 transitions) (R9)
- [x] D7 **CartSim deleted** — no fake re-simulation anywhere (R24)
- [x] D8 Curriculum over maps / team counts / player counts / cart counts (R25)
- [x] D9 Interruptible & resumable — handled stop restores weights, optimizer, counters, and replay (R9, R32)
- [ ] D10 Server transition measure matrix: retention under perturbation, recovery time, acquisition, terminal, held-out (—, [BUILD-DATA])
- [x] D11 Learned local action-linear dynamics ensemble `Δy=b(y)+A(y)u` (R10)
- [x] D12 Checkpoint/architecture integrity — fingerprint + strict load, mismatch loudly reinitializes a complete model (R24)
- [x] D13 **Replay buffer of hundreds–thousands of featurized states, reused across the losses** (R24)

### E. Observation / featurization (the map-reduce)
- [x] E1 Perception-gated observation: frustum + LOS + 2-V-cell cap (emergent stealth) (R5)
- [~] E2 Per-team observation buffer of contextual events — live but sparse (R20)
- [x] E3 V-cell segmentation to a 5–15% receptive field — MEASURED, radius is an output (R24)
- [x] E4 Temporal contraction toward an uninformative prior (canonical) (R24)
- [x] E5 Bounded-support spatial mask (canonical, radius solved) (R24)
- [x] E6 Low-rank egocentric integration is the ONLY spatial mixer AND is called on the live path (R24)
- [x] E7 Belief is per-bot; there is no "team belief" (R4)
- [x] E8 Enemy positions featurized ONLY through observation (R5)
- [x] E9 **Full per-player resource state ENTERS the matmul** — input rank 4 → 33 (R24)
- [x] E10 Per-player rows, `z` descriptors, hierarchy, eligibility and integrated weights are LOGGED on real runs (R25)
- [x] E11 The canonical `featurize.py` belief pipeline exists and is the one that runs (R24)

### F. Playerbot interface (the WHAT/HOW boundary)
- [x] F1 matmul decides WHAT; stock navmesh decides HOW (R12)
- [x] F2 Skill-orthogonal — never touches aim/dodge (R12)
- [~] F3 Objective vocabulary → stock target entities (explore, gather, crush-weak, duel-strong, push/suppress, hunt) (R12)
- [x] F4 Spawn-timing and travel-commitment as real instruments — commit nonzero 0.03% → 96.79% (R25)
- [x] F5 No policy in QC; no second navigation definition (R12)
- [ ] F6 Affordance QUALITY — does the policy actually aim `hunt` at the winningest rival? (—)

### G. World / maps
- [x] G1 Procedural multi-map fusion produces megamaps — bots CROSS tiles; 12-bot boot, OBJECT ERROR 0 (R35)
- [x] G2 Megamaps actually USED by the training/live server (R20)
- [x] G3 Bots traverse long distances between fused regions — now cart↔cart, not just spawn→cart (R25)
- [x] G4 Prominence rule: exclusive objective entrances conspicuous; connectors may be subtle (R15)
- [x] G11 Procedural geometry — 56 doorways CUT into stock map brushwork (R28)
- [x] G12 Connectivity solvers + metrics over solid occupancy — proxies DELETED with mapfuse; measured on the oracle (R35)
- [x] G13 Offline rendered-fusion measures — joinshot 6/6 real frames + frame visual distributions; pre-compile geometry measurement in 1.4 s (R35)
- [x] G14 **Placement is real: suitability selection + bridge/stub taxonomy + 3D bin pack + geometry edit; refusal deleted** (R28)
- [x] G5 Stock-navmesh compliance; no project-specific bot nav graph (R12)
- [x] G15 **A world-space geometry oracle over the ASSEMBLED fused world (solid/trace/clearance/standable)** — `negspace`, two entry points, one law (R35)
- [ ] G16 **Budgets DERIVED from a measured relationship, not tuned until one test passed** (ZEROED, R30)
- [~] G6 Entity budget at scale — no invisible bots at high player counts; waypoint-sprite spam REGRESSED in live client (R22)
- [~] G7 Diegetic communication of cart paths/state — duplicate "CART 2" labels observed (R22)
- [x] G8 Headless client renderer for join inspection (real offscreen renders) (R15)
- [x] G9 The curriculum can SELECT the fused megamap; megamaps recognised, not hardcoded (R25)
- [x] G10 Cart origins distributed ACROSS fused regions — cart↔cart traversal 6679 → 24475 (R25)

### H. Multipolar dynamics — description & demonstration
- [ ] H1 Resource-domination logging (alive count, health/armor pools, consumption vigor) (—)
- [ ] H2 Cartlane exertion logging (simultaneously contested carts, PW-flip volatility) (—)
- [ ] H3 Cross-team focus/attrition matrix — is damage concentrated on the key rival? (—)
- [ ] H4 Multipolar dynamics visualizer (tug trajectories, PW timeline, focus matrix) (—)
- [~] H5 Web view of the run (three.js phase space + live mesh table) (R16)
- [~] H6 **A supervised demonstration server+client pair that outlives any agent** — integrated demo already demonstrated; current curriculum ownership and unattended release lifecycle remain separately accountable (R27, R40, R41)

### I. Method / process laws
- [x] I1 SPEC is a verbatim user-quote index under the provenance law (R17)
- [x] I2 AGENDA checklist + append-only Record (this file) (R17)
- [x] I3 No unit tests; evidence is artifacts/proofs/real runs (R13)
- [~] I10 **uv manages every Python environment** — both installed node runtimes now report UV/Python 3.12.14 and matching locks; every future workload's launcher still requires runtime evidence (R29, R41)
- [x] I4 No stubs, no pseudodocumentation claiming false completeness (R10, R12)
- [~] I5 No repeated inlining — one canonical definition per algorithm (R10, R11)
- [~] I6 No fake re-simulation anywhere (D7 outstanding) (R13)
- [x] I7 Private SQCU repo; never push upstream (R1)
- [x] I8 Services interruptible & resumable by construction, proven by crashing early (R9)
- [x] I9 **Learner and supervisor are SERVICES** — `run(schedule)` deleted, `serve()` never returns, restarts logged (R25)

---

## Current work order

R41 and [RELEASE-CLOSURE.md](RELEASE-CLOSURE.md) govern current work: availability
repairs, reconstructible branches/dependencies, and reconciled closure first; then
define the next learning-transaction and behavioral measurements. Mesh integration is
already demonstrated. C2 is no longer an unimplemented prerequisite, and the old
system-Python-3.9 observation is not the current installed-runtime state.

## Historical work order — opened 2026-08-31, superseded by R41

Ordering is forced by R19/R20: the input is rank-4 with per-player state zeroed and
the belief map-reduce is a constant-Φ substitute, so **widening the operator before
feeding it buys nothing.** Feed it, then widen it, then show it.

**Track 1 — model core** (`solver/strat`: estimator, qkv, head, value, relattn,
featurize, live_belief, dpp, online, train, strat_responder)
1. E9 — wire full per-player resource state (health/armor/ammo/weapon-bitmask/pos/
   vel/dist-to-cart) into `x` on the live path; report real input rank before/after.
2. E11/E3 — restore canonical `featurize.py` (705 lines at HEAD), delete the inlined
   substitute in `live_belief.py:162-273`, route the live path through the canonical
   functions, enforce the 5–15% receptive-field bound.
3. C1/C2 — delete the softmax-attention operator; implement the all-to-all **Gram-matrix fusion**
   + SwiGLU at **IR ≥128d**, differentiable end-to-end, count-invariant, O(n²), with
   the matrix-fused coupling output landing IN the IR the probes consume.
4. C12/D2 — value heads become actual **linear** probes on the final IR (still two
   asymmetric role-gated heads, still per-row scalar).
5. D7 — delete `CartSim` and every importer.
6. D12 — architecture fingerprint in checkpoints; loud complete reinitialization
   instead of silent `strict=False` partial loads.

**Track 2 — engine data path + world** (QC strategy IO, tools, game_value,
curriculum, instruments)
7. E9/E10 (engine half) — find where per-player values are lost between
   `payload_strategy_gather` and the responder; fix the schema/staging; LOG per-player
   rows + `z` descriptors + relation rows. Rebuild `progs.dat` clean.
8. B11 — make the CGT evaluator resolve on real cartstates (228/228 currently
   `unresolved`), keeping partizan honesty; report resolve rate before/after.
9. G9 — curriculum must be able to select the fused megamap (`locate_asset` globs
   `data/*maps*.pk3`, missing `zzzz-fused.pk3`).
10. G10 — distribute cart origins ACROSS fused regions so cart CHOICE imposes
    traversal (all 68 cart nodes currently in one region); re-verify by union-find.
11. F4 — make travel-commitment a real per-assignment quantity (written on 1/3150).

**Track 3 — the demonstration** (`solver/strat/joracle`, `web`)
12. Live telemetry side-channel: per tick publish cartstate/PW/SUCC, hierarchy,
    per-player assignments, and the internals — IR rows, W/L outputs, advantage,
    `diag(K)` — showing absences rather than faking them.
13. **The j-oracle viewer, continuous**: behavior (cart tug, PW timeline with flips,
    cross-team focus matrix) beside internals (rolling linear probes on the live IR
    **with the random-projection and shuffled-label controls displayed**, plus IR
    effective rank and width) — so an R19-class pathology is visible at a glance.
14. Demo wiring: one command brings up a real cartserver (free port), the mlx
    responder on the mini, and the viewer; the **on-device Xonotic client connects to
    that same server**; the viewer survives a server restart and reattaches.

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
`E:code` `xonotic/solver/strat/game_value.py`: impartial-component XOR reduction,
`team_nimbers`, `projected_winner` (PW), `succession` (SUCC by repeated
decrement of the leader's deepest cart). `GameValue.nimber` is null for partizan
or incompletely enumerated reachable options. Verified on
this host: the XOR of `[1, 1]` is zero, while one depth-two cart beats two depth-one carts.

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
softmax-attention operator, not a Gram matrix. It was introduced by the coordinator's
prompt, not by the SPEC. C1 zeroed. SPEC §8 also:

> how wide did you think the hidden states were supposed to be for this? under
> 128d? maybe you were slippin.

Present width is `d_row ≈ 16`, so C2 zeroed and A6 (SoC saturation) is blocked
behind it. E6 zeroed: `live_belief.py` re-implements `featurize.py`'s belief
pipeline inline, leaving `featurize.egocentric_integration` / `temporal_contraction`
/ `belief` dead — reported, not collapsed. Recovery route: implement the Gram-matrix fusion +
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
`E:proof` The compute-shape argument: a Gram matrix over all player rows is a large
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

### R19 — 2026-08-31 — C11, D2 → zeroed; C12, B11, E9, E10, D12 opened
`E:run` The retired pre-contract j-space snapshot, summarized in the revision record
and superseded by `design/jspace-probe.md`, used 228 real Game-2 lines / 3150
player-rows, ridge probes, a 60/40 split, and no CartSim.

The measured induced-coordinate mass was **zero**. The shuffled-label control
passes everywhere (the probes are honest), but **nothing beats the
random-projection control on any non-tautological target**, and where the IR does
beat raw inputs the trained and random-init encoders agree to three decimals:

    n_controlled   Rp trained 0.9563 | random-init 0.9566 | rand-proj 0.6788
    gain           Rp trained 0.2878 | random-init 0.2864 | rand-proj 0.0815
    logp           Rp trained 0.1243 | random-init 0.1238 | rand-proj 0.0048
    instr. kind    acc .6135          | .6150             | .5751  (maj .416)

i.e. the separation is the encoder's nonlinearity, not learned semantics.
Trained-vs-random differs only in rank (14 vs 9) over an input matrix of **rank 4**.

Root cause, and the session's most consequential finding: **the per-player resource
state never entered the matmul.** `game2_train.jsonl` logs no per-player observation
rows — health, armor, ammo, weapon bitmask ("holds rocket launcher"), position,
velocity, distance-to-cart and `beta` were all **zero in the model's own input on
this run**. So SPEC §3 —

> the POLICY is integrating FULL RELEVANT GAME STATE FEATURES like THE HEALTH OF
> ALL PLAYERBOTS AND THEIR AMMO COUNTS AND GUNS

— is not merely unmeasured, it is **unwired** (E9 zeroed). Instrument descriptors
`z` and relation rows are also unlogged, leaving 20 of 60 value-row dims
unreconstructible (E10 zeroed). Width is NOT the limiter: a rank-4 input does not
fill 16d, let alone 128d — so C2 must be fixed *with* E9, not instead of it.

D2/C12 zeroed against SPEC §5 ("value estimators trained as linear probes upon the
final IR"): `RoleValueHead` is `Linear(60→32)→silu→Linear(32→1)`, an MLP.

B11 opened: `game_value` returned `{"kind":"unresolved","nimber":null,"reason":
"incomplete option graph"}` on **228/228** lines — the CGT value never resolved
during the real run.

D12 opened: every checkpoint on disk is architecture-stale w.r.t. the local tree
(MatrixFusionSwiGLU/`X_WIDTH=48`/`d=128` locally vs `relattn`/`d=16` on the mini), and
`OnlineLearner._load_full` uses `load_weights(..., strict=False)` — a 128d rewrite
would **silently** "resume" from a 16d checkpoint restoring almost nothing.

Analytic answer to "can the linear algebra be right and the j-space still be poor?"
**Yes.** The value loss is one scalar per player per head, so it constrains at most
two directions; nothing in the objective rewards decodability of anything else. The
guarantee is *value-relevant directions, not arbitrary semantics*. "Who holds a
rocket launcher" fails twice over: it is zeroed out of `x`, and `role_rewards` is
computed purely from cart nimbers and PW transitions, so weapon state has **no
gradient path to the loss at all**.

### R20 — 2026-08-31 — G2 → full; G3, E2, E4, E5 substantiated; E3, E11, G9, G10 zeroed
`E:run`+`E:code` The megamap / observation map-reduce audit
(`design/megamap-observation-status.md`, 600 lines, every line reference verified).

**G2 full.** The megamap is what actually runs: `zzzz-fused.pk3` (16,417,377 B) is
mounted in `Xonotic/data/`, and the Game-2 cartserver log line 103 reads
`Loaded maps/fused.ent`, then `cart 0: 30 path nodes, length 10920.210938`. A
separate 14 MB server log shows 244 match starts alternating `fused` /
`runningmanctf`, 50 bots on 5 teams, carts banking to `s 3480.6`.

**G9 zeroed.** `curriculum.py:94`/`:529` default to `runningmanctf`, and
`locate_asset` (`:228-236`) globs only `data/*maps*.pk3` — which
`zzzz-fused.pk3` does not match. The curriculum literally cannot select the fused
map; the megamap runs by launch config, not by curriculum.

**G3 partial, G10 zeroed.** Union-find over the 960 entity origins in the deployed
`fused.ent` (700-unit XY link) recovers exactly 3 disjoint regions:
`size=611 plc_nodes=0 spawns=17`, `size=282 plc_nodes=68 spawns=12`,
`size=67 plc_nodes=0 spawns=12`. **All 68 cart-path nodes sit in ONE region while
29 of 41 spawns are in the other two.** Spawn→nearest-cart-node distance: min 156,
median 5195, max 9857; 26/41 spawns exceed 4000 units. So spawning imposes a join
traversal, but choosing cart A over cart B does not — the megamap only half-creates
the strategic commitment cost.

**F4 substantiated at `[~]`.** The plumbing is real —
`SC["COMMIT"]=5` → `mesh_scatter(..., PLC_SC_COMMIT, ...)`
(`sv_payload_strategy_io.qc:232`) →
`this.bot_strategytime = max(this.bot_strategytime, time + this.plc_str_commit)`
(`:276`) — but `instruments.py:258-260` writes `COMMIT` only on the
`TRAVEL_COMMITMENT` branch, and the real run shows 3150 assignments distributed
`explore_cell 1346, push_cart 1224, contest_post 400, hunt_rival 156,
spawn_timing 20, suppress_cart 3, travel_commitment 1` — i.e. **`commit>0` on 1 row
of 3150 (0.03%)**. Plumbed, effectively unused.

**E3 zeroed; E4/E5 partial; E6 corroborated; E11 zeroed.** `live_belief.py:162-273`
re-inlines all four belief stages and *that copy is on the live path*
(`strat_responder.py:14,165,264,265` → `estimator.py:141` → `qkv.py:43`,
`concatenate((x, beta)) @ W_q.T`). The inlined substitute uses
`support_radius=2.0`, `areas=np.ones(n)`, `T=self.decay`, a **constant-literal Φ**
(`:240-249`) and an extra `weights/total` normalization the spec formula does not
have; **the 5–15% receptive-field bound is computed nowhere** (E3). Meanwhile the
canonical `featurize.py` has been truncated in the worktree from **705 lines at
HEAD** (`segment_vcells:216`, `temporal_contraction:473`,
`egocentric_integration:525`, `beliefs_for_bots:562`, `receptive_fraction:189`) to
**27 lines** (stage-4 `spatial_mask` only); a repo-wide grep for `egocentric` now
hits only a docstring in `joinshot.py` (E11).

The audit's own risk measures, recorded verbatim as the reason E-group states move:

> **#5, the observation map-reduce.** It is the only one where the file that *is*
> the spec was deleted from the worktree while an unreviewed hand-inlined
> substitute — constant Φ, hardcoded T and radius, different normalization, no
> receptive-field bound — runs in its place, and the swap is invisible from
> outside because `beta` still flows into `qkv.query`. Nothing errors; the
> `[FIRM]` formula simply is not the one being computed.

**E2 partial.** The QC gate is genuinely all three conditions
(`sv_payload_strategy_io.qc:82-104`), and real belief rows grew cells 12→80,
edges 1→103, teams 1→5, `invalid 0`; but over 228 ticks / 18.4 s / 12 bots the
aggregate is `deposited 32, accepted 1096, duplicates 6430` — live but sparse.
Defect for I5: the Python mirror `buffers.py:49-51` is dead because
`live_belief.py:93-94` passes `True, True, 0.0` as literals.

### R21 — 2026-08-31 — I9 opened (self-terminating learner, finite supervisor)
`E:code` Diagnosis of why the live run stopped. Not a crash and not a restart cycle:
the learner has a **wall-clock suicide** and nothing is a service.

    strat_responder.py:124   ap.add_argument("--secs", type=float, default=90.0)
    strat_responder.py:202   while time.time() - t0 < args.secs:

Attribution by `git blame`: the `--secs` deadline entered in **`de18d7a`**
(2026-08-29 19:28, my dispatched agent's one-shot demo runner) and hardened into the
learner's main loop in **`f238c4d`**. It was a demo script's bounded run that became
the learner's lifetime. **I introduced this fragility.**

The would-be restarter is also finite — `curriculum.py:507` is
`for index, item in enumerate(schedule):` inside `run(schedule)`, i.e. a batch job
that returns when the schedule is exhausted. So: the learner exits on a timer, the
batch supervisor exits on a finite list, and the dedicated server is left up with no
learner attached — which is the observed state (server PID 73488 on 26031 alive,
`live.jsonl` last write 19:29, no `strat_responder` on the mini).

The distinction this frontier encodes, per the user this session: **restarting a
server/match repeatedly is correct and desirable** (it is how the resumability
contract of R9 is exercised); a learner or server that stops permanently for no
stated reason is a defect. Per-match duration is legitimate — it belongs to the
MATCH, not to the learner's lifetime.

Recovery route, corrected by the owner the same session — **delete the incorrect
branch entirely**, do not park it behind a flag:

> delete incorrect branches entirely

`--secs` and the deadline loop are DELETED from `strat_responder.py` (not defaulted
to unbounded, not kept as an opt-in bounded mode); the finite-schedule termination
and any `--once`/`--n-matches` equivalent are DELETED from `curriculum.py`. Exactly
one lifetime path remains on each side: run until signalled. The supervisor keeps
starting matches, relaunches the learner if it dies, and logs every restart with its
reason, leaning on the already-proven atomic checkpoint/resume (R9). A wrong
alternative is removed from prod, harness and docs in the same change — a flagged
dead branch is the same fragility with a switch on it.
Also removed `xonotic/solver/strat/test_runtime.py` (untracked, added this session,
violates the standing no-tests directive).

### R22 — 2026-08-31 — G1 zeroed; G11, G12, G13 opened; G6, G7 defects recorded
`E:run` A real in-client screenshot of the fused map on a live server (observer,
bots playing, clock 16:12) supplied by the owner, plus their statement:

> the map fusion code was also clearly unfinished and didn't satisfy any written
> constraint in a few obvious ways, incl. missing viewers, msising client renderers,
> missing features related to geometry fusion, total absence of procedural geometry,
> and no connectivity solvers or trivial visualizers or metrics over connectivity and
> navmesh solutions

What the frame shows: the world is **almost entirely black void**, with a single
small isolated island of structures floating in an orange-skied rectangle — not a
coherent fused megamap; **dozens of overlapping `WAYPOINT` sprite labels** stacked
into an unreadable cluster; and **two separate markers both labelled `CART 2`** plus
one `CART`.

This retracts R15's `[x]` on G1. R15 graded fusion complete from OFFLINE artifacts
(BSP/pk3 byte sizes, "3 maps, 3 joins", a flood-fill boolean) — none of which
observed whether the fused world *renders as a world*. That is the AGENDA form's own
failure mode stated in reverse: a checkmark was moved by an artifact that did not
measure the requirement. The evidence standard is unchanged, but the admissible
artifact for "fusion works" now must include a real render.

G11 opened (procedural geometry absent — `mapgen.py`'s parametric q3map2 pipeline is
not producing geometry in the shipped fusion). G12 opened (connectivity solvers and
quantitative metrics over connectivity and navmesh solutions, not a flood-fill
boolean). G13 opened (viewers/visualizers that would have caught this offline).
G6 regressed in evidence: the node-sprite bound of R16 is not in effect on this run.
G7 defect: duplicate cart labelling.

### R23 — 2026-08-31 — D13 opened (no replay buffer; every state used once)
`E:code` Owner's question about what is buffered for training, answered from the code.

There is **no replay buffer in the package** — `grep -rniE 'class .*replay|replay_buffer'`
returns nothing. What exists is a short CREDIT queue:

    online.py:61    credit_horizon: int = 5
    online.py:197   self.pending.append({ "previous": ..., "reward": ..., ... })
    online.py:204   return self.flush(...) if terminal or changed or len(self.pending) >= self.credit_horizon else None
    online.py:211   for start, item in enumerate(self.pending):   # one update() per item
    online.py:225   self.pending.clear()

`flush()` computes the discounted MC return from each start index, takes **one
gradient step per pending item**, then clears. So every featurized state is consumed
exactly once, with residency <= 5 transitions — frequently fewer, because a cart
signature change forces an early flush. The 82 updates of the 2026-08-30 live run
were 82 states, each seen once.

The pending entry already carries the correct replay tuple (`context, state,
snapshot, w_in, w_out, actions, behavior_logp`, assembled at
`strat_responder.py:502-504`), and the correction that LICENSES reuse is already
implemented and running:

    online.py:165   ratio = mx.stop_gradient(mx.minimum(mx.exp(current["logpi"] - behavior_logp_mx), self.importance_clip))
    online.py:166   actor = -mx.mean(mx.stop_gradient(ratio * error) * current["logpi"])
    online.py:169-170  winner_weight = winner_mask * ratio ; loser_weight = loser_mask * ratio

i.e. participant-local clipped importance weighting on the actor and both critic
heads. The staleness correction exists; nothing is ever allowed to become stale.

Requirement, quoted:

> presumably we have a buffer of hundreds to thousands of featurized game states we
> can repeatedly run through our training losses; whether the states are stale or not
> does not actually matter as much as it might sound, so long as the buffer is
> refilled with fresher states eventually

Cost of the present shape: real Game-2 transitions are the expensive resource (the
reason the mesh exists) and each buys exactly one gradient step; consecutive strategy
ticks are near-identical so updates are correlated and high-variance; and the value
heads/probes, which need many states to fit, are fitted 5-at-a-time on a streaming
window.

Recovery route: push completed transitions (with their already-computed returns) into
a ring buffer of hundreds–thousands; sample minibatches per update with multiple
passes; evict oldest so the buffer tracks the current policy; keep the existing
clipped ratio as the sole off-policy mechanism; include the buffer in the atomic
checkpoint/resume state (R9). Dispatched to the model-core owner.

### R24 — 2026-08-31 — C1,C2,C3,C12,D2,D7,D12,D13,E3,E4,E5,E6,E9,E11 → full; C11 → partial; I9 partial
`E:run`+`E:code` Commit `1d34864`. The R19/R20 wounds closed in the forced order —
feed the operator before widening it — verified against a real `cross_rdma_live`
Game-2 run (62 telemetry lines, 744 player-rows), all work done in a scratch copy on
the mini; nothing on port 26012 touched.

**E9.** All 19 OBS columns arrive; 17 populate `x` (`X_WIDTH=48`, incl. a 24-bit
weapon expansion). **Input rank 4 → 26 (`x` alone), 33 with `beta`.** 9 distinct
weapon bits observed, mean 3.15 weapons/player — "who is holding what" is in the
matmul. Only `POWER` is zero, and genuinely so (nobody held Strength that run), not a
wiring gap. The headline SPEC §3 requirement is implemented.

**E11/E3/E4/E5/E6.** `featurize.py` restored 27 → 614 lines; the inlined substitute
at `live_belief.py:162-273` (constant Φ, radius 2.0, extra normalization) deleted;
`LiveBelief` is now an adapter onto the canonical stages. **E3 measured for the first
time on real geometry: support radius 0.0 → 0.641 — an OUTPUT of a bisection, not a
constant — with the all-cell median receptive fraction inside [5%,15%] on 62/62
ticks.** Full pipeline over 62 ticks in 0.61 s, so it is affordable at live cadence.

**C1/C2/C3.** The superseding matrix-fusion operator constructs an exact participant
Gram matrix from the direct sum of rival-projection rows and team-one-hot tensor
team-projection rows. Thus `C = G_rival + same_team * G_team`; no hand-authored relation
term or softmax remains. The coupling matrix lands in the IR and the reward gradient
reaches both learned metric factors.
Count-invariance across 62 real states × 51 instrument counts and n∈{2,3,5,8,12}:
shapes and object identity constant. Renormalizing by `d` instead of `√d` was
required — at 128d the old scaling overflowed the DPP inverse (`LU factorization
failed 257`) on real rows.

**C12/D2.** `RoleValueHead` is `nn.Linear(width, 1)` — actual linear probes.

**D12.** Architecture fingerprint + `strict=True`; the 16d-era `policy_online_v3.npz`
is REFUSED with both parameter lists printed. The hardcoded 14-leaf loader is gone.

**D13.** Ring buffer (2048 transitions / 256 MB, oldest-out); `flush()` pushes the
credited segment then takes `replay_steps` sampled steps; off-policy correction is
the pre-existing clipped ratio. Measured: **gradient steps/transition 1.000 → 5.000,
mean age of sampled transitions 0 → 91, held-out critic MSE 0.01254 → 0.01066
(−15%)**. Atomic checkpoint 257 MB in 0.34 s, resume 2.75 s with buffer restored.

**I9 (learner half).** `--secs` and the deadline loop deleted outright; the loop runs
until SIGINT/SIGTERM/SIGHUP and drains into the final flush + checkpoint.

**Defect fixed.** `strat_responder` set `previous = None` whenever the instrument
batch changed, so on runs where the instrument set turns over every tick the learner
saw **zero** transitions.

**C11 → `[~]`, honestly.** R19's headline is gone: trained and random-init no longer
agree, and at 1 epoch the trained IR beats random-init on 9/11 targets and beats a
128d random projection on `cart_depth_total`, `kind`, `team` and `weapon_bit` —
weapon identity decodes better from the trained IR (0.910) than from a random
projection of the inputs (0.843), which was impossible when the field was zeroed. But
it is not a clean win: decodability **degrades monotonically with training** (speed
r² → −0.98 at 11400 updates) as the IR collapses onto the value-relevant subspace —
R19's analytic caveat, now measured as a curve. The binding constraint is **data**
(62 lines, one `(k,j,l)` shape), not the architecture.

**Open hand-off (I9 supervisor half).** `curriculum.py:343` still passes
`--train --secs <duration>` and will now `SystemExit(2)`; `xonotic/README.md:17` has
the same stale invocation. Also reported: the OBS/EVT schema carries `CELL` but no
waypoint-link table, so V-cell stage 2 unions observed cell transitions with a 2-NN
stand-in rather than literally fusing contiguous *navigable* paths.

### R25 — 2026-08-31 — R19 PARTLY CORRECTED; B11, E10, G3, G9, G10, F4, D8, I9 → full
`E:run`+`E:code` Commit `b7f6759`. Engine data path, CGT, and world commitment cost.

**Correction to R19 — the record must carry this.** R19 concluded that per-player
resource state "never entered the matmul". That inference was **wrong**. The run-era
responder did call `state_with_observations(...)`; what actually happened is that the
j-space probe *reconstructed* `x` from the telemetry log, and
the retired probe hardcoded zeros for the unlogged health, armor, ammunition, and
related coordinates. **The rank-4 input was an artifact of the reconstruction, not the model's
input.** E9 was a LOGGING failure (E10), not a wiring failure. R19's other findings —
trained≈random-init, the MLP-not-probe deviation, the 228/228 `unresolved` CGT, the
`strict=False` hazard — stand. R19 is left unedited per the append-only rule; this
entry supersedes its E9 claim.

**What was really lost: the CART rows.** In all 228 real lines carts 1–2 read
`id 0, depth 0, length 0.0` while the same run's engine log says
`cart 1: 22 path nodes, length 8269.58` / `cart 2: 16 path nodes, length 5786.72`.
The gather assumed `payload_carts[0]` (and later `plc_str_cart_pool[0]`) began a
CONSECUTIVE edict run:

    /* darkplaces prvm_edict.c:267 — first REUSABLE free slot, not an append */
    for (i = prog->reserved_edicts + 1; i < prog->num_edicts; i++)
        if (PRVM_ED_CanAlloc(prog, e)) { PRVM_ED_ClearEdict(prog, e); return e; }
    /* xonotic/darkplaces-work/mesh_ipc.c — reads n CONSECUTIVE edicts */
    m->req[row*m->width + col] = prog->edictsfields[(first + row)*stride + fld];

`payload_str_pool_run()` now constructs and re-verifies the contiguous run and logs
the proof: `[PLCPOOL] evt_base 5227 rows 256 contiguous 1 / cart_base 5483 rows 4
contiguous 1`. **Second real defect:** a cart is a brush model (`view_ofs = mins`,
`sv_payload.qc:743`), so its world position is `origin + view_ofs`; `NCART`/`NCART_D`
compared against the UNOFFSET origin — **every nearest-cart column of every run so far
was wrong** by the brush `mins`. Fixed; cart world position added to the cart columns.
Two silent truncations from `MAX_PARMS = 8` in `strcat`/`sprintf` also found and fixed.

**B11 → full: 0/228 → 228/228.** The run used `EmpiricalTransitionGraph`, which marks
options complete only via `observe_terminal` — i.e. only on delivery. No cart was
delivered in 18.4 s, so it was pricing a graph with no edges (worsened by the zeroed
cart rows). A closed-form cart option graph was added: a **neutral** cart is impartial
(cylinder occupancy lets any team move it) and backward induction **derives** its
Grundy value as `r = levels − depth`, so `game_value.py`'s XOR reduction is now *proved rather than
asserted*; a **controlled** cart is partizan (Regime A/B give holder and opponents
different moves) and gets **no nimber**. Real result on the 228 lines:
`{"impartial": 147, "partizan": 81}`, `nimbers {8: 147}` — the 81 partizan lines are
exactly those with a controlled cart. No nimber is faked.

**G9/G10/G3.** `locate_asset` globs all `data/*.pk3`; megamaps are *recognised* (fusion
marker or `joins.json`), never hardcoded; `--maps auto` → `['fused','runningmanctf']`
with the held-out split intact. `host_components()` discovers N regions at runtime by
union-find and round-robins carts across them. Per-region `plc_nodes` **0/68/0 →
13/30/17**; cart↔cart traversal **0↔1 6679 → 24475**, **0↔2 7330 → 27370**;
spawn→nearest-cart median **5195 → 357**, `>4000u` **26/41 → 0/41**. The commitment
cost moved from *joining the game* to *choosing a cart* — precisely R20's complaint.
Reported honestly: curvature on fused k=3 rises to 168% (hard constraints still
`solid_viol 0 float_viol 0 corridor_viol 0 PASS`).

**F4 → full.** `travel_horizon()` gives every assignment its own horizon:
`commit_nonzero 11527/11909 = 96.79%` (was `1/3150 = 0.03%`), the 382 zeros being
exactly IDLE + SPAWN_TIMING.

**I9 → full.** `run(schedule)` deleted; `Curriculum.serve()` never returns (an
exhausted schedule regenerates with `seed + cycle`), the learner is relaunched on exit,
and every transition is logged to `supervisor.jsonl`. No `--once`/`--n-matches` added.

**E10 → full.** `payload_strategy_log()` emits `[PLCPUB]/[PLCCART]/[PLCOBS]/[PLCEVT]`
read back off the staged fields, so the log *is* the gather source. Real coverage over
993 ticks / 11,909 player-rows: ID/TEAM/HEALTH/AMMO/POS/WEAPONS/CELL/NCART/NCART_D/
TSS/ALIVE/CONTROL 993/993; ARMOR 913/993 and VEL 918/993, genuinely zero when nobody
has armor or everyone is standing. Each record carries `z` (25×16) and `relation`
(5×25×16) built by the same `build_instruments` the live operator calls.

**Waypoint links.** New `PLC_EVT_KIND_CELL_LINK = 4` streams the stock nav graph's cell
adjacency onto the perception ring (`waypoint_get_link`), so V-cell stage 2 can fuse
truly *navigable* paths instead of a 2-NN stand-in. It cycles rather than latching
(a one-shot pass caught 33 of 469 waypoints because `g_waypoints` fills late):
`[PLCLINK] pass 1 over 467 waypoints, 3681 cell-link rows emitted`.

Open, routed to the model-core side: `EstCache.get` ignores `(k, j, l)` so a learner
spanning differently-shaped matches would silently keep the first shape (this is why
the learner is currently per-match); `featurize`'s V-cell stage 2 must consume kind 4
instead of the 2-NN stand-in; and `strat_responder` no longer logs `game_value` at all,
so the newly-resolving CGT is not yet visible in telemetry.

### R26 — 2026-08-31 — G14, H6 opened (one specification error at two layers)
`E:code` Two findings with the same root: I specified the ARTIFACT instead of the
OUTCOME, so agents delivered evidence rather than a working system.

**G14 — the fusion layout solves the wrong problem.** `mapfuse.py` assigns maps to
lattice cells in INDEX ORDER (`cp = [(m % cols, m // cols) for m in range(j)]`), sizes
non-uniform bands to the widest map per column, centers each map in its cell, nudges a
tile only WITHIN its own band, and REFUSES a join past a constant
(`MAXCORLEN = 6000.0`, line 25; `if math.dist(sa, sb) > MAXCORLEN` ~line 1414). A grep
for anneal/optimi/cost/objective/swap/permut over the file finds **no placement search
and no cost function at all**. So the fused world is maps floating in a void in index
order, joined by straight tubes.

The assigned problem is **place-and-route**, NP-hard on two axes: cell assignment is a
**Quadratic Assignment Problem** (which map where, minimizing socket-to-socket cost
under the adjacency graph), and placement is irregular-volume floorplanning (a map's
WALKABLE interior sits at an arbitrary offset from its bounding box — exactly why wide
stock maps produced kilometre corridors). The unstated-because-obvious procedural
insight: **the bridge maps absorb the residual mismatch** — you generate geometry to
span whatever gap placement leaves, you never refuse. The spec's own toolkit names
"portal and jump pad and verticality", and **a teleporter has no length constraint**,
so `MAXCORLEN`-as-refusal is self-inflicted: the catalogue already contains an
unbounded-length connector and the code declines to use it. R25's "honest gap" framing
(4 of 40 joins violating the one-non-navigable-join-per-map budget) was therefore
wrong — that is not a gap, it is the wrong solution shape reporting its own constant
as a property of the world.

**H6 — nobody owns the demonstration.** Every agent brief was written as verification:
"bring up a server, capture telemetry, TERM cleanly". A server plus logs proves a
claim; a connected client is only needed for a human to SEE it. Track 3's brief said
"Document the exact client connect command" — the connect was specified to be
*documented*, not *made*. Agents complied exactly. Compounding: agents are headless so
a GUI client is outside their evidence loop, and TERM-on-exit (correct hygiene for a
throwaway) is structurally incompatible with a demo that outlives the agent. Result is
port churn with no owner — 26012 → 26031 → 26032 → 26042 — and a client that died and
was never restarted. Recovery route: the demo pair must be a supervised service, and
the client attach must be part of bring-up rather than a printed instruction.

### R27 — 2026-08-31 — I10 opened; H6 → partial (uv, and the client is supervised)
`E:run` Two corrections, both mine.

**I10 — I fixed code to accommodate a broken environment.** I reported that
`game_value.py`'s `zip(..., strict=True)` (Python 3.10+) was a blocker because "the
mini has exactly one interpreter, Python 3.9.6", and removed the `strict=True`. That
was an excuse based on a dependency. The owner's standing rule is uv for all Python
environment management, and **uv was already installed on the mini** at
`~/.local/bin/uv` (v0.12.6) — it simply is not on the non-interactive ssh PATH, and
The legacy environment had been built on Apple CommandLineTools Python 3.9.6:

    home = /Library/Developer/CommandLineTools/usr/bin
    version = 3.9.6

Superseded by the locked `/usr/local/mesh/.venv` runtime realized by `install.sh`.
→ **Python 3.12.14, mlx 0.32.2, numpy 2.5.2**, GPU matmul verified, `zip(strict=)`
available. Every launch path in `joracle/demo.sh` switched off the system-python venv
onto the uv env. The rule going forward: an environment limitation is a thing to fix,
never a constraint to code around.

**H6 → `[~]` — the client is now supervised.** Observed live: the client connected,
joined YELLOW, played (frag messages in its log), then `Connection timed out` — because
the server pid changed underneath it (25358 → 78386). Per the owner's rule a server
restart is *correct*; a client left unattached across one is the defect.
`joracle/client-keep.sh` now relaunches the client whenever its process is gone or its
log shows a timeout/disconnect, truncating the log per launch so a stale match cannot
re-fire, and appending every event with its reason to `/tmp/xonclient-keep.events`:

    2026-08-31T03:48:53Z supervisor_start addr=127.0.0.1:26042
    2026-08-31T03:48:53Z client_absent reason=process_gone -> relaunch
    2026-08-31T03:48:53Z client_start pid=6220 addr=127.0.0.1:26042

Still `[~]`: the *server* remains owned by whichever agent brought it up, so the pair
outlives a client crash but not the agent. The server half needs the same treatment.

### R28 — 2026-08-31 — G1, G11, G12, G13, G14 → full (the geometry edit)
`E:run` Commit `f1cd522`, built to the owner's re-statement of the constraint:

> pick a list of maps that seem well suited to having geometry edited to make them
> diegetically connect to other neighboring maps by litearlly changing their geometry
> to have doors, galleries, passageways, etc., which either continue or newly appear
> in existing maps in plausible spots. then solve a 3d bin packing problem, evne
> poorly, where 'bridge maps' (more than 3 connection sites) are joined by procedural
> geometry to 'stub maps' (fewer than 3, more than 1 connection sites).

**Selection + taxonomy.** A ray marches from each cardinal-extreme node of a map's
largest bot-reachable stand-on-able waypoint component through the carver's own
exact-plane solid predicate. A site requires first solid 24–640u out, 8–384u thick
(a wall *panel*, not bedrock), nothing within 224u behind, door-sized standing room in
front, open space beyond — classed `continue` (narrow standing room / nav dead-end)
or `newcut` (broad exterior-reading wall, gets an architrave). It rejects honestly:
`dance`'s shell is patch-mesh curvature with no brush between x=1872 and x=3072, so it
yields no ray hits. Over all 29 navigable stock maps: **25 BRIDGE (>3 sites), 4 STUB
(2–3), 1 rejected** (`nexballarena`, 1 site).

**The geometry edit — the deliverable.** 56 doorways cut: **46 continuing an existing
passage, 10 new openings on exterior walls**; 444 source brushes split into 825 convex
remainders (the wall stays where it was, as thick as it was, minus a door); 417 wall
surfaces re-cut into 479 clipped surfaces carrying their own texture; reveal surfaced,
threshold laid, jamb/header architrave into the outer face; waypoints chained through.
Wall thickness cut through: min 8 / median 64 / max 256.

**Pack, kept simple as instructed.** Shelf pack with real Z levels (4×4×2 lattice, 29
cells for 29 tiles); cells ranked by lattice-neighbour count and tiles by site count so
bridges land in high-adjacency cells and stubs in corners; tiles anchored on their
WALKABLE centre and median floor, not their bbox.

**Refusal deleted.** `grep -c MAXCORLEN mapfuse.py` → 0. Against the 39-tile
cap-clipped build: cart-navigable joins 26/40 → **28/36**; non-cart joins per tile
max 3 (VIOLATED on 4 edges) → **max 1 (HELD)**; edges dropped/refused **8 → 0**;
corridor median 4791 → **3295** (31% shorter) with the long tail now visible rather
than capped. Placement door-gap objective 190995 → 117234 (38.6%).

**Evidence.** Per-doorway camera pairs (inside looking at the new opening in the host
map's own wall; outside looking back at the facade): **42/42 frames observed, zero missing** —
`p04_erbium_continue_in.png`, `p11_geoplanetary_newcut_in.png`,
`p01_silentsiege_continue_out.png` read as architecture. Real boot: stock
`darkplaces-dedicated` on port 26071, 29 tiles / 166 MB BSP, 3 carts pathed
(485/383/425 nodes), 8 bots over 5 teams, live gameplay, **zero runaway errors**;
flood-fill 29/29, 1 component, hop-diameter 9, walking diameter 86823u.

**Three defects found by watching RSS and by booting.** (1) `Src.__init__` called
`mkentfile.Bsp(data)` and never used the result; that helper grids brush AABBs with an
unguarded `range()`, so one brush bounded only by oblique planes (stock `catharsis` has
twelve) becomes a ~1e15-iteration loop — the loader ate **75 GB RSS** and never
returned. Call deleted; catharsis loads in 0.8 s / 0.7 GB. (2) The same defect from the
entity side (**33 GB**): `axialize()` re-emits the 18 offending stock brushes inside
axial clamp planes at hull+4096 — shape untouched, AABB finite; peak build RSS back to
~6.8 GB. (3) A **second runaway ceiling** visible only by booting: stock
`waypoint_loadall → waypoint_get → boxesoverlap` is O(n²) per frame and at 900
waypoints climbs into `navigation_markroutes_nearestwaypoints`. A waypoint budget
(`--wpcap`, default 600) shaped like the proven entity budget fixes it, with the link
graph **contracted onto the survivors** so reachability is preserved (flood-fill 100%).

Honest remainder: render evidence is at 8 tiles (the 29-tile world is proven by the
real boot); four `relocate_spawnpoint … could not get out of solid` object errors
remain, non-fatal.

### R29 — 2026-08-31 — G1 → partial; I10 → partial (deploying found what building did not)
`E:run` Deployed the 29-tile geometry-edited megamap to the live demo
(`/tmp/fzfull/data/maps/fused.pk3`, 48.4 MB → `Xonotic/data/zzzz-fused.pk3`) and
cycled the server. Three things came out of it, none visible from the build.

**The client supervisor worked unattended** — the mechanism H6 exists for, proving
itself on a real restart:

    2026-08-31T04:40:17Z client_dropped reason=Connection timed out -> relaunch
    2026-08-31T04:40:19Z client_start pid=35332 addr=127.0.0.1:26042

**G1 → `[~]`: the megamap crashes the server at RUNTIME above ~8 bots.** The map
loads correctly — `Loaded maps/fused.ent`, carts pathed at **485/383/425 nodes**
(the old 3-map fusion was 30/22/16) — bots join, gameplay runs, and then:

    Quake Error: Host_Error: server runaway loop counter hit limit of 10000000 jumps

This is the SAME ceiling for the third time in a third place. R28 lifted it at
worldspawn (entity budget) and added a waypoint budget for the O(n²)
`waypoint_get`/`boxesoverlap` scan, and the fusion agent's own boot passed — **with 8
bots**. The demo runs **12**, and the deployed map carries **613 waypoints against a
600 cap**. So the ceiling is bot-count sensitive and the budget was tuned to the bot
count the agent chose rather than the one the demo uses. Relaunched at 6 bots and it
survives with zero runaway errors. Recovery route: tighten `--wpcap` for the
full-pool map and boot-test at the demo's actual bot count, not a convenient one.

**I10 → `[~]`: the uv switch is committed but not deployed.** The responders actually
running on the mini are still executing under system Python:

    /Library/Developer/CommandLineTools/.../Python3.framework/Versions/3.9/... -m solver.strat.strat_responder

`demo.sh` was switched to `mesh-python`, but the already-running responders predate
the edit and nothing restarted them. The environment is created and verified; the
running system does not reflect it. Same error class as the rest of this stretch —
reporting a change instead of verifying the live system reflects it.

**Supervisor gap found and closed.** After the crash-and-relaunch the client came up
while the server was down, sat at a menu, and was never noticed: the supervisor
detected `process_gone` and `connection timed out` but not *launched-but-never-
attached* — nothing in the log says "timed out" when there was never a connection to
lose. Added a `client_never_attached` check (no connect confirmation within 45 s of
launch → relaunch).

### R30 — 2026-08-31 — G15, G16 opened; G12, G13 demoted (the proxy pattern, named)
`E:run`+`E:code` The megamap's runtime assertion cascade, and the accountability for it.

Observed on the live server:

    SVQC OBJECT ERROR in relocate_spawnpoint: could not get out of solid at all!
    NOTE: Spawnpoint at '8202.0 -11548.0 4342.0' needs to be moved out of solid
    --- CUT HERE ---
    assertion failed: `!IL_CONTAINS(this, it)`
    VM_remove: tried to remove the null entity or a reserved entity!

Spawnpoints ship buried in solid; stock Xonotic's relocate search fails, its error
path double-inserts into an IntrusiveList, and the assert cascades. The in-solid drop
(`mapfuse.py:763-779`) tests

    src.solid_brush_at([o[0], o[1], o[2] + dz])

— the SOURCE map's predicate at SOURCE coordinates, sampling three points straight up.
The spawn's real position is `o + off` in the FUSED world, after tile packing (now
with Z-level stacking), after the doorway cuts split brushwork, beside connector
geometry. Both failing spawns are at extreme Z (4342, −3287), consistent with the
stacked levels.

**G15 — the missing tool, and the pattern it names.** EVERY fusion validator works on
a proxy rather than on the assembled geometry: `solid_brush_at` is source-space; the
frame void-fraction distribution measures how black the screen is; the flood-fill reports waypoint-graph
connectivity; joinview measures path length; fusegraph measures abstract topology.
**Nothing can answer "is this point, in the assembled fused world, inside geometry?"**
So G12 and G13 are demoted to `[~]`: they measure real things, but not the thing that
fails. The oracle (`solid_at`/`trace`/`clearance`/`standable` over the assembled world,
one shared definition, correct across offsets and Z levels) should have been built
first, with spawn, cart-path and doorway validation rebuilt on top of it.

**G16 — every limit was set to whatever made the current test pass:** the waypoint cap
(600) tuned against an 8-bot boot while the demo runs 12 and the map shipped 613
waypoints; the entity budget tuned until worldspawn stopped tripping; `MAXCORLEN 6000`
invented so a hard case could be refused; cell assignment left in index order because
nothing forced a placement solve. Budgets must be derived from the measured
relationship (the stock waypoint scan is O(n²) per frame and scales with bot count)
with the bot count an explicit input, and boot-tested at the count the demo uses.

**Accountability.** These were my briefs. I asked for evidence that a thing RAN and
never for a tool that understands what was BUILT, then accepted those proxies and moved
checkmarks with them twice — R15 graded fusion `[x]` on BSP byte sizes and a flood-fill
boolean; R28 graded five frontiers full on renders plus one 8-bot boot. The AGENDA's own
reading rule (*checkmarks come from artifacts*) requires the artifact to measure the
requirement, and it did not.

**Self-inflicted incident, recorded.** The `client_never_attached` check I added in R29
fired on a 45 s timer while the client was still precaching the 166 MB megamap, turning
the supervisor into a relaunch thrash loop that leaked three client processes.
Rewritten: relaunch on never-attached only when there is BOTH no connection AND a log
that has stopped growing for 120 s, and `launch()` now kills any prior client so
orphans cannot accumulate.

### R31 — 2026-08-31 — proxies are DELETED, not demoted
`E:none` (instruction) Owner:

> delete proxies instead of apologizing for them

R30 named the proxy pattern and demoted G12/G13 to `[~]`. That was still the wrong
disposition, and it repeats a ruling already made this session ("delete incorrect
branches entirely"): a stand-in kept beside the correct mechanism is the same defect
with a label on it, and something trusts it again later. My oracle brief compounded it
by saying "keep the existing frame measures and flood-fill, but they are now
secondary".

Corrected instruction to the oracle work: **delete** the source-space in-solid spawn
check and `solid_brush_at` if it has no legitimate source-space caller left; delete any
bounding-box stand-in consulted about the assembled world, including
`mkentfile.Bsp`'s AABB-from-plane-distance grid wherever fusion touches it (it produced
both the 75 GB unguarded-`range()` loop and phantom "no floor" violations on clear
corridors); delete hand-rolled corridor tube sampling in favour of oracle
`trace`/`clearance`. **Exactly one definition of solidity may exist afterwards.**

The render distributions and the graph component measure survive only under their true names —
"frame void fraction and color-level count" and "waypoint-graph component mass" — and may not be
cited as evidence that geometry is correct, nor gate any spawn / cart-node / doorway /
connector decision. G12 and G13 therefore go to `[ ]`: they are not partially-done, they
are to be rebuilt on the oracle.

### R32 — 2026-08-30 — B5 full; D1, D3, D5, D9 made literal
`E:code`+`E:run` The user-specified loser objective is now the executable objective,
not a signed hierarchy proxy. For `k` teams, player/team incidence
`P∈{0,1}^{l×k}`, team nimbers `n`, winner one-hot `q`, loser mask `u=1-q`, and
`C_ij=1[n_i>n_j]`, `runtime.py` computes

    rho = (I-diag(q)) C u + (k-1)q
    rW  = -q ⊙ (1-q')
    rL  = (1-q) ⊙ 1[rho' > rho]

and lifts the disjoint team events to player rows with `P`. Thus W's sparse event is
only loss of the current winning path; the optimal W estimator predicts its discounted
negative incidence before role exit; winning rows optimize the shared policy toward
positive W advantage, preserving or restoring robustness without a speed objective.
L's sparse event is only an upward loser-rank flip; the optimal L estimator predicts
the discounted count of those events before role exit; losing rows optimize the shared
policy toward positive L advantage, approaching successive rank boundaries and finally
acquiring the winning path. Holding and falling have zero immediate L reward, but a
successor farther from the next upward event has lower learned value and therefore
negative TD advantage.

The final IR is `H∈R^{l×128}`. Two independent shared `128→1` probes produce
`vW,vL∈R^l`. The actor receives `AW+AL∈R^l`, with W and L bootstrapping only their own
successor head; a role change terminates the old return. This is neither one global
scalar broadcast across contradictory teams nor an `l`-output head attached to every
row.

Real-server evidence, recomputed by the current runtime over 1,800 rows from
`mesh-mini:/tmp/mesh-joracle/output/live.jsonl`: 1,783 actual consecutive transitions,
shapes `(k,j,l)=(4,3,12),(4,3,13),(4,4,12),(4,4,13)`, reward support exactly
`{-1,0,+1}`, 18 upward-L team events, 10 W-loss team events, and identical event values
for every teammate row. Examples include an unchanged winner with a loser moving
`rho:[0,3,0,0]→[2,3,0,0]` and receiving `+1`, and a winner loss with all of that
winner's rows receiving `-1`.

Live deployment exposed a separate resumability defect: the learner wrote
`--online-checkpoint` but defaulted its read source to `--checkpoint`, so an absent
inference checkpoint restarted learning at zero. The read and write source are now the
same online checkpoint unless `--resume-checkpoint` explicitly overrides it. After a
handled SIGTERM, the RDMA-attached responder restored architecture `c592fbe875fb7914`,
weights, optimizer, update 2,085, all 1,261 replay transitions (106.1 MB), then advanced
to update 2,105. PID 33201 remains the live `joracle_demo` responder.

### R32 — 2026-08-31 — target scale is l≈256 / 8 teams; every measurement so far was on a degenerate instance
`E:proof`+`E:code` Owner:

> dont undershoot om player or team count. lets assume we wat mechancis and gameplay
> system scales which tidily scale to ~256 playerbots and 5+ teams at a time (why was
> all of this stuff adding multiple objectives then introducing 'strategy layers' that
> only make sense in terms of bulk allocation of walls of healthpoints and spawns in
> effectively rts-attack-move commands?)

**The strategy layer is bulk RTS-style allocation, and it is degenerate below that
scale.** At l=12: "allocate 3 bodies to cart 2" is trivially enumerable; DPP diversity
over 5 carts + posts + rivals has nothing to spread; W/L asymmetry, succession, denial
and coalition-vs-spread are indistinguishable from noise; and the W role's "hold the
region, restore margin under perturbation" has no mass to hold a line with. So every
measurement this session — the j-space collapse, "trained ≈ random-init", the 574-state
horizon — was taken on **a different, easier game**, not on a small sample of this one.

Serialization at the real target (14 GB, l=256, 8 teams, j=5):

    m=64  (aggregated/RTS)   derived 1.13 MB ->     12,699 states  | raw 28.30 KB ->   518,787   (41x)
    m=512 (per-rival)        derived 8.59 MB ->      1,668 states  | raw 28.30 KB ->   518,787  (311x)
    relation block alone = 89–93% of every derived transition
    raw @ float16            14.15 KB       ->  1,037,574 states

Raw is **O(l)**, derived is **O(l·m)**, so the storage error of R23/R24 grows with
exactly the scale the design targets — at l=256 the derived format leaves the learner
1,668 states, i.e. no memory. Deep serialization survives easily in the raw format:
~519k states fp32 (144 h at 1 Hz, 29 h at 5 Hz) or ~1.04M at fp16.

Two design consequences, both free:
- **Per-rival instruments are the wrong shape at this scale**, not merely expensive: you
  do not issue 255 individual hunt orders, you attack-move a mass at a region or a cart.
  Aggregated/RTS instruments (m≈64: mass-to-cart-k, suppress-cart-k, hold/contest-region,
  explore-region, spawn-wave, travel-commit) are 8x cheaper to store AND the correct
  semantics.
- **The belief pipeline already scales right** and needs no change: precompute
  `Phi*f_c` once over the V-cells (O(C·rank)), then O(horizon) per bot readout — it is
  map-size-bound, not player-bound, so 256 bots is 256 cheap readouts.

### R33 — 2026-08-31 — correction to R32: the action space is fine; the DENSE PAIR TENSOR is not
`E:proof` Owner:

> "m=512 per-rival instruments is the wrong shape at 256 players," are you over-reaching
> or resorting to some kind of idiolect? what does this mean? each playerbot still
> nevertheless needs an action vector...?

Over-reach, corrected. R32 claimed per-rival instruments are "the wrong shape". That
conflated two different things:

1. **The per-player action space.** Every playerbot needs an action vector and it may
   legitimately address a specific rival — "hunt rival r" is a real affordance (the QC
   routerates a specific rival entity), and choosing *which* enemy to focus (the
   winningest rival in your way) is precisely the strategic content the design exists to
   produce. Nothing about 256 players removes that.
2. **The dense `(l, m, 16)` relation tensor.** THIS is the defect: at l=256, m=512 it is
   8.59 MB per transition, 93% of the stored state, and dense over pairs that are mostly
   meaningless — a bot on the far side of the map carries a relation row for hunting a
   rival it has never observed and cannot reach.

The repair is representational and preserves the action space entirely: either
**candidate sets** (each bot's live instrument list is the carts, the posts, and the
rivals in its observed neighborhood — a property the perception-gated observation buffer
already computes — giving `(l, m_eff, 16)` with `m_eff` in the dozens), or a **factored
head** (a distribution over ~8 instrument TYPES plus a pointer/attention over candidate
targets, storing `(l, types)` + candidate features instead of a flat 512-way categorical).

Also struck: R32's implication that "attack-move a mass" means aggregated instruments.
That phrase describes what the **team-scale allocation emits** (bulk allocation of health
and spawn mass across objectives); it is not a restriction on an individual bot's action
vector. A team-level allocation resolves *down* to individual bots targeting individual
rivals. Collapsing those two levels into one claim produced jargon standing in for an
argument.

R32's other findings stand unchanged: the l≈256 / 8-team target, the O(l) vs O(l·m)
storage asymmetry, that prior measurements were taken on a degenerate instance, and that
the belief pipeline already scales correctly.

### R34 — 2026-08-31 — decimation, not tile count, is what breaks bot navigation
`E:run` Re-profile of the rebuilt k6 after the duplicate-carve repair (`1313194`).
The fix did not merely shrink the tile-count penalty — **it inverted it.**

    world                        unstuck    stmt   stmt/call  %proceed
    k2  (2 tiles, 268wp, 0% dec)   1,266   0.80M      630       87%
    k6  OLD (816wp, 28,305 degen)  3,011   2.61M      867       84%
    k6  NEW (816wp, 0 degen)         612   0.55M      897       97%
    29-tile shipped (96% dec)     12,099  71.94M    5,946       14%
    29-tile n=1100  (67% dec)     11,815  70.24M    5,945       15%

k6's unstuck rate fell 80% and landed at **48% of k2's**, with the best
goal-planning proceed rate of any world measured. Essentially all of the apparent
tile-count penalty was the duplicate-carve defect, so the "second independent
term" previously written into FUSION-SPEC §8.6 is retired: at 0% decimation, six
tiles is not worse than two, and **waypoint decimation is the sole identified
driver**. The clean contrast is 87–97% proceed at 0% decimation against 14% at 96%.

Severity is a decimation effect too, and the larger half: 630 / 897 / 5,946
statements per unstuck call means the 29-tile world is ~7x costlier per call *on
top of* being ~10–20x more frequent — that product is the 90x statement gap.

**This indicts the waypoint budget itself.** `--wpcap` was added to survive the
O(n²) `waypoint_get`/`boxesoverlap` runaway ceiling at 29 tiles (R28), and at 96%
decimation it takes bot goal-planning from ~90% to 14% proceed. The budget was a
consequence of a scale nobody asked for, and it was actively destroying
navigation. At the 2–6 tile target (R32) no decimation is needed at all — which
is the second time the maximal-configuration choice turned out to be the cause of
the machinery built to survive it.

Two honest limits recorded by the measuring agent, unprompted:
- It **withdrew one of its own columns**: frag counts do not survive cross-scale
  comparison (k6-new 41 frags vs k2's 130 while healthier on every direct
  navigation metric — a 57.9 MB world versus 11.3 MB with the same 12 bots, i.e.
  bot density, not navigation). The unstuck ratio is per-decision and scale-free,
  which is why it survives where frags do not.
- 612 is a single 60 s window carrying roster variance. "No detectable positive
  term at k=6" is what the data supports; "six tiles beats two" would need a
  second window.


### R35 — 2026-08-31 — G1, G12, G13, G15 → full; the aperture is a parameter, not a carve

**The category error, and the deletion.** `mapfuse.py` authored architecture by
writing BSP lumps, so it had to synthesise the tree, VIS and lightmaps by hand and
none of them were real: 49,152 grey lightmaps, 0 visdata, 2 clusters, 2.0 GB RSS.
`design/MAPGEN-ROADMAP.md` had ruled this out in one line before any of it was
written — "Do not write a CSG/brush library. Emit `.map` text and let q3map2 do the
BSP tree, VIS, lightmaps and collision." Deleted entire (2,557 lines), not
refactored; `placement.py` keeps only what decides WHERE tiles go and WHETHER they
connect, and authors no geometry. Via q3map2 the same placement gives 3,236
clusters, 16.43% average visible, 83.6% cullable.

**Apertures (MAPGEN-ROADMAP stage 2) were never written.** `strip()` gains a skip
set; that one change opens the shell. Because the gap is chosen during the sweep,
its facing, free volume and vantages are known by construction — nothing recovered
by ray marching, nothing that can disagree with the geometry because it IS the
geometry. Each ships PLUGGED so the standalone map still seals; a join drops the
plug, so a joined map cannot differ from the one that was validated.

**G15 closed, one law.** `solid(p) == ns.cell_at(p) < 0` with two entry points:
compiled BSP and authored source. Pre-compile validation of assembled k=2 source
takes **1.4 s and catches exactly the 2 spawns** the engine reports as
`relocate_spawnpoint`, against 108 s to compile first; the 12-bot boot then logs
`OBJECT ERROR = 0` (was 2). A second, source-side oracle existed briefly and was
deleted rather than kept beside it.

**Merge invariant (§8.13a): no per-tile `common/lightgrid` brush may survive a
merge.** It clips the compiled world to its own volume and fails silently and
totally — brushes stay in the lump, the volume is culled, **q3map2 reports no leak,
and the map boots with a tile missing.** This, not decompile fidelity, was
"warfare fills as outside"; warfare round-trips alone at 1,366 clusters vs stock's
1,362. Dropped unconditionally in `mapsrc.place_tile`, the single entry point.

**Bots cross (G1).** 35/35 points along the channel with 0 gaps, one navmesh
component, a 6-waypoint 1,891 u crossing path with 6/6 fit.

**Frames (G13).** joinshot observed 6/6 frames with zero missing or unreadable. Four stacked causes: the dummy SDL
driver has no GL and `vid_soft`'s surface is SDL 1.2 API in an SDL 2.32.70 binary,
so it died before any map loaded; it launched the STOCK Xonotic.app, which lacks
builtin #656 and died on connect with an identical "0 frames" symptom; that binary
needs `-xonotic`; and the build has no PNG writer, so screenshots were TGA.

**Method notes, all of them corrections to how this was measured.**
- *File size cannot grade a frame.* 320x200x24 TGA is 192,018 bytes every time,
  content-independent. The coordinate "sized like the samples" could not
  have told a black frame from a good one. Void-fraction + level-count replaces it.
- *Which binaries EXIST is not which binary RAN.* Checking the tree found one
  client; joinshot was launching a fourth outside it. Any tool launching a client
  must use the project build.
- *RETRACTED: "a sample can miss, and did."* I reported a marched floor query
  differing from a closed-form one (203/207 vs 205/207; 338/341 vs 339/341) as
  evidence that the samples missed real points. It was the reverse. `negspace`,
  an independent exact implementation, returns 205/207 and 339/341 — agreeing
  with the MARCH. My closed-form ray was the outlier and had a seam bug: a ray
  cast exactly along the shared face of two abutting brushes passes between them.
  NAV-SPEC §10's sampling disqualifier may still be right, but this was not
  evidence for it; it was my own defect, cited against the tool that was correct.
  The general lesson is the one that keeps recurring here — two implementations
  agreeing against a third is evidence about the third.
- *Fix the defect BEFORE the fold, then fold against a live reference.* Differential
  testing against the oracle being retired caught four bugs in the replacement,
  including interval-propagation bounds that silently dropped every plug brush
  (sealed mouths read open) and an inverted expansion sign that let a box centred on
  a point the module called solid still "fit". Deleting first would have shipped all
  four.
- *A leak-free compile proves nothing about tiles.* The lightgrid failure reports no
  leak at all.
- *The player hull is 32x32x69*, not a symmetric 32x32x48.
- *An exact test can thread a seam.* Aperture vantages were aimed at ring
  VERTICES, which are the boundary between two strip segments, so a radial ray
  ran along the shared face of two abutting plug brushes and passed through a
  sealed wall — seed 23 reported its mouth correctly sealed while the sightline
  through it stayed clear. Aim at segment centres. Sampling hid this; exactness
  exposed it, which is the honest argument for exactness rather than the
  retracted one above.

### R36 — 2026-08-31 — the built artifacts are PROGRAMS; inventory before deletion

The server is a VM and `progs.dat` is the program: statements, globals, a string
table, a field/stat layout. Five of them existed, unversioned, with no recorded
order of supersession — so "which program is running" had no answer, and a
session was spent attributing behaviour to source the running build did not
contain.

**A built .dat has no diff to merge, but it IS a distinct program.** It is
dominated by weight by globals, constants and the field/stat layout, not by
statements, so two builds that differ disagree about DATA — including STAT
indices and field offsets. A server `progs.dat` from one build with a client
`csprogs.dat` from another reads a *different stat slot*, silently, with no error
on either side. Hence the build SET is the artifact (`set_id` over all three
outputs), not three files. And a .dat built from uncommitted source that has
since changed cannot be regenerated at all: it can be measured and never rebuilt.

**Inventory before deletion, via symbol tables** (`tools/qcdump.py` reads the
function and globaldef name tables straight out of the .dat — mtimes and sizes
prove nothing):

| build | functions | has that current lacks |
|---|---|---|
| build-qc 08-30 | 13,622 | `payload_carts##GET/SET`, `plc_str_cart_pool##GET/SET` |
| plc-home 02:06 | 13,489 | the same four |
| deployed 10:57 | 13,643 | **none** |
| current 11:53 | 13,644 | — |

Resolved, each one:
* `payload_carts##GET/SET` — an older *declaration form*. The list survives as the
  plain global `g_payload_carts`, used in eight places. Not a lost feature.
* `plc_str_cart_pool[4]` — a STATIC four-slot array, replaced by
  `plc_str_cart_base = payload_str_pool_run(payload_cart_count, 1)`, sized to the
  actual cart count. The old program capped carts at four; the new one is
  count-invariant, which the spec requires.
* `plc_str_last_cell/kind/subj` grew 5 → 6 slots in current — the team/lane
  count generalisation.

So the chain is a clean supersession, build-qc → plc-home → deployed → current,
each strictly adding, and nothing needs porting forward before the old ones go.

**Method note.** I deleted `render/plc-home/data/*.dat` BEFORE running this
inventory, on the grounds that nothing referenced it — which is an argument about
callers, not about contents. It was recoverable from git and turned out to carry
nothing unique, but that was luck. Read a program's symbols before deleting it;
"nothing imports it" does not mean "it does nothing".

### R37 — 2026-09-01 — the native bot tick has an active V-cell plan; all 30 cart maps pass

The strategy response decoder has an atomic typed commit, but the matrix-fusion/MoE strategy is
a global computation and is not a causal player scheduler. An earlier worktree-only
Python `VCellPlan` merely permuted completed response rows and inverted the permutation;
it never established the tick-loop requirement. Its absence from live use was evidence
of unfinished integration, not evidence that the requirement should be deleted.

The active implementation now lives at the engine boundary that owns bot controller
updates. `StartFrame` opens one batch and advances `bot_think` for every bot before
DarkPlaces invokes the per-client `SV_PlayerPhysics` callbacks. Native `VCellPlan`
derives three-dimensional cells and a causal radius from live hull dimensions, speed,
and frame time, and compacts the occupied color waves. The radius covers the two-sided
controller-output horizon; perception does not expand it to a map-scale enemy-search
radius. Every world-aware bot decision runs once under ordinary ordered QuakeC semantics
and deposits a complete row when its pure keyboard transform is due. Fourteen input
coordinates per row scatter into the native structure-of-arrays working sets and nine
output coordinates per row stable-gather at the barriers. The transform uses
compiler-selected SIMD with no fixed row ceiling or invented lane width. Shared player
physics and combat remain after the controller barriers in `StartFrame`.

This completes the player/controller scatter-gather boundary without pretending the QC VM
itself is multithreaded or that copying its float fields rolls back RNG, traces, entity
allocation, and world links. `[VCELLMERGE]` exposes input coordinates, output coordinates,
and row-buffer bytes; `[BOTWARP]`, `[VCELL]`, and `[BOTQC]`
expose working sets, barriers, rows, SIMD rows, and scalar fallback mass. Native release
and QC builds establish the composed interface; live measures establish its realized
population and cost.

The corrected live 16-bot measurement formed 12 working sets over 16 occupied cells,
reached a three-row working set, and derived a two-cell causal radius from a 69-unit cell
extent. Once controller staging was active, 13 rows entered the vector kernel and its
cumulative staged mass reached 392 while scalar and fallback mass remained zero. That
earlier actor/client-only transaction copied 7,533,008 bytes but did not capture the
engine-owned state needed for a valid transaction, so the row-buffer measurement
supersedes its coordinate and byte totals. The earlier 145-cell radius was the
enemy-detection distance mistakenly
used as a same-tick write horizon; it is no longer part of the scheduler interface.

The fused-map failure was not bad geometry. The negative-space cache retained convex
free cells but discarded the compiled BSP's blocking-brush index. Point traces remained
right, while swept cart hulls were incorrectly required to fit within one arbitrary BSP
leaf; fusion repartitions those leaves, so valid source paths became false “burrows.”
The cache now persists and restores the engine-equivalent blocker index. A swept query
over 150–180k free-cell rows fell to roughly 0.2–0.3 ms against nearby solid brushes,
and all nine transformed segments of the four accepted runningmanctf tracks agree
between source and fused coordinates.

The real fused measurement artifact now reports 8 teams, 4 carts, grounded path lengths
1,105 / 1,246 / 1,246 / 752, four distinct origins separated by 4,371 units, zero
head-on flow, and 240 unbiased generic spawns with 560 units minimum cart clearance.
Together with the prior artifacts, 30/30 selectable maps report `advanceable`,
`winnable`, `spawn_unbiased`, and `headon_free` true.

The existing live 26042 process supplies the gameplay-scale evidence that was previously
missing. Its current process has run for nearly three hours with 255 bots plus the
observer, 256 configured teams, 32 carts all reporting support, and no `Host_Error`,
SVQC object error, or network-buffer overflow in that process. All 256 team indices were
exercised in two map windows; four completed matches were won by teams 220, 99, 97, and
145. Eight map windows account for at least 9,017 simulated seconds over 10,656 wall
seconds (84.6%) while one host core stays approximately 97% occupied. This proves the
gameplay/protocol/spawn/cart scale, while also measuring the remaining serial-controller
cost rather than hiding it.

This closes static map validity and functional approximately-256-player gameplay. It
does not close the causal strategy study: that live process has no responder and no
`[PLCBARRIER]` commits. Realized matrix-fusion-vs-FFN-vs-linear-vs-default behavior/Elo evidence
and a measured two-host run proving the remote expert is both executed and required for
the planning deadline remain outstanding while the Mini is partitioned.

### R38 — 2026-09-01 — metrics replace the removed harnesses; population sampling becomes hardware-driven

All repository test programs, fixtures, embedded QC cases, compiler suites, build
targets, cached bytecode, and generated test maps are deleted. Runtime measurements,
held-out policy study records, map state measurements, health endpoints, and
telemetry remain because they report the behavior of the shipped system rather than a
parallel harness.

The strategy curriculum no longer allocates `maxplayers` from the initial roster. It
reads the engine's compiled `MAX_SCOREBOARD`, starts at one live participant quantum per
team, doubles only to find a hardware boundary, and bisects the observed interval. Each
point comes from the existing in-memory telemetry publishers on every meshed node. A
point has a complete distance coordinate when every node publishes deadline-bearing
work; the target center minimizes squared distance to minimum-node roofline saturation,
the MacBook whole-machine bandwidth envelope around 50 percent, deadline load, and role
mass. Runtime population changes happen through `sv_cmd setbots`
while the server is running, so later human or bot connections are not excluded
by the initial process command.

Team and cart lists now seed only the first cycle. Each later cycle takes the
minimum-distance whole-mesh observation and samples half/center/double neighborhoods
around its team and cart counts, bounded by the compiled `NUM_TEAMS` and
`PLC_MAX_CARTS`. The player seed becomes the measured admitted count per team. Thus all
three axes move around measured hardware pressure rather than returning forever to a
number written before the run.

The distributed evaluator previously labeled the Mini responder
`joracle_distributed_evaluation` while its MacBook expert silently published
`game2_server`, and the expert published no deadline. The whole-mesh reporter could
therefore never join the two halves into one complete-coordinate observation. Both halves now use
one launch identity and the shared strategy cadence supplies both deadline records.

At closure time both nodes are connected, reachable, and sampled by the 8787 reporter,
with no unavailable telemetry node. No strategy producer is running, so achieved
workload FLOP/byte rates correctly remain unavailable rather than being inferred from
unrelated host activity. The next release evidence is the adaptive distributed run and
the mirrored matrix-fusion/FFN/linear/default perturbation study; no fixed roster result can close
either obligation.

### R39 — 2026-09-01 — the strategy study exposes surrogate mass directly

Distributed scale is now a match-owned service rather than an undocumented prerequisite.
The curriculum starts the expert on the game host before the server, stages the current
solver and RDMA runtime when that host is remote, keeps its checkpoint identity aligned
with the responder, records its artifacts and health, and visibly restarts it after an
exit. Node routing reverses automatically when the game host is remote. A held-out matrix-fusion
record retains the local-fallback call mass, executed rank, top-k MoE row mass, padded
physical row mass, and the distinct host identities from the whole-mesh 8787 artifact.
No record is removed from the study for any value of those coordinates.

The hardware necessity claim is measured rather than inferred from model size. At a
sparse live cadence, after the game response has already been returned, the responder
runs the same rows and checkpoint through the local scale operator, compares its output
to the remote output with a tolerance derived from float32 epsilon and the learned
widths, and substitutes that measured duration for all remote calls in the measured
plan. The report retains distributed deadline slack and observed all-local-substitution
slack as separate signed measures.

The causal study now schedules every mirrored base-arm pair and reports the adjacent
matrix-fusion/FFN, FFN/linear, and linear/default score differences with realized attack, damage,
and competition-utility lifts under every perturbation family. It also schedules a
same-checkpoint participant-fusion and residual-fusion interventions whose only behavioral changes are independently zeroed
participant and residual-feature Gram-matrix contributions. Its realized outcomes and the
same aggression/robustness measurements remain parallel coordinates.

Map measurement schema 7 emits shared team incidence, finite-route mass, median and
first-claim ratios, per-spawn ratios, cart-origin clearance,
spawn-approach-plus-cart traversal distributions, rider-gap mass, per-cart path atoms,
and per-team objective incidence. Fast, slow, and
volatile perturbations now vary spawn-wave cadence, population-derived respawn regimes,
and pusher dropoff as well as cart dynamics, so robustness is not a synonym for surviving
one ruleset.

Both nodes are currently live in the reporter, but the MacBook is simultaneously carrying
multiple unrelated 90–136% CPU jobs, 44% GPU residency, and desktop/audio work. No release
workload was launched into that condition. Until live producers exist, R39 is an
implementation closure and the distributed empirical sample mass remains zero rather
than being manufactured from static checks.

### R40 — 2026-09-01 — one distributed release owner, self-realizing remote game host

`evaluate-distributed.sh` no longer stages checkpoints and independently launches an
expert and a responder without a game. It resolves the live Mini address and execs the
curriculum, which is now the only owner of build, map catalog, game, expert, responder,
population sampling, training, mirrored study, artifacts, and restart transitions.

The remote boundary accepts one SSH command and uses it uniformly for process control,
rsync, game launch, expert launch, and TERM. The Mini needs no pre-existing checkout:
each match moves its remote runtime toward current intent by incrementally staging the
dedicated engine, complete local Xonotic data path, match userdir, solver, payload schema,
RDMA client runtime, and any active matrix-fusion checkpoint. Nothing is indexed by commit.

The expert relay's Unix socket is fixed by the engine, so its ownership is fixed too.
One stable PID file now spans match directories. Before each launch and after each match,
the curriculum verifies that PID's command, sends TERM, and waits for the worker's normal
teardown. A supervisor crash can therefore leave recoverable state, not a worker that
permanently occupies the socket outside the next supervisor's accounting.

The capacity reader now treats the compiled `MAX_SCOREBOARD`, `NUM_TEAMS`, and
`PLC_MAX_CARTS` values as ceilings instead of taking the maximum of those values and a
requested roster. `auto` discovers the 30 engine-selectable maps; `_hudsetup` remains
explicitly runnable but its authored `forbidden` and `noautomaplist` declarations no
longer inject it into payload calibration or held-out study. The estimator's baseline
kind and descriptor widths now come from the canonical instrument schema rather than a
duplicated action-count literal.

At this closure the 8787 fabric record contains both nodes, no stale or unavailable
node, zero strategy producers, and zero claimed strategy rows. The MacBook remains busy
with unrelated transformer, WebKit GPU, bridge, and window-server work; the Mini reports
one fully occupied performance cluster. The distributed release has therefore not been
started into the user's audio/render deadline. Its empirical workload mass remains the
next state transition, not a static claim.

### R41 — 2026-09-04 — credit the integration result; separate repairs, builds, and rollout

**E:run (historical), E:code, E:build, operator observation.**

> the project runtime actually demonstrates the mesh computation being used, and in fact being used for dozens of variations of a game-playing team-up-and-take-down high teamcount strategy puzzle, even if the strategy solver was never workign to feature spec...

> basically confirming that the RDMA api actually became usable as something like an importable header which is safely used and incorporated into projects by subagents without patterns of  tampering and meddling that corrupted host or device kernel memory state or repeatedly crashed or softlocked either machine.

— Operator, 2026-09-04, review follow-up.

The existing `xonotic/README.md` execution record independently states:

> MLX on node 1 consumed 223 observation/cart/event snapshots, emitted
> 83 responses, and completed 82 online updates in 45 seconds.

A7 records achieved integration, not universal freedom from defects. A6 remains partial,
but C2 is no longer its implementation blocker. H6 remains partial: the demonstrated
game integration and R40's unified curriculum owner do not establish every unattended
release-lifecycle property. I10 remains partial for its universal scope, but the old
running-system-Python-3.9 description is superseded by both node runtime reports:

> "python": "3.12.14"
>
> "uv": "0.12.6"
>
> "packages": {"mlx": "0.32.2", "mlx-metal": "0.32.2", "numpy": "2.5.2"}

The pre-review September work is preserved as `codex/september-runtime`; the repair
branch is `codex/availability-closure`. The neighboring compiler's existing capacity
repair is named `codex/mesh-capacity` and also carried as the mesh-owned
`vendor/netradiant-capacity.patch`. A fresh named-upstream-branch reconstruction built
the compiler, rather than relying on the neighboring checkout's existing binary.

The bridge repair maintains the published header and ABI. Its submitted-page bitmap
returns arena ownership after queue retirement, failed posts return their descriptors,
and HUP reaches the normal teardown. Expert request replay uses session/operation
identity and cached replies without replaying completed gradient updates. Provisioning
retains loaded providers and stages source/Python generations instead of routinely
stopping services or mutating a working environment in place.

Both Macs built the bridge and client library. Shell syntax and Python imports passed;
the isolated Python realization reported `mesh runtime ready`, then `mesh runtime
retained` on the second run. No live bridge was restarted and no study was launched.
The current sampled zero traffic is not the historical workload's empirical sample mass.

[RELEASE-CLOSURE.md](RELEASE-CLOSURE.md) now maps all seventeen goal obligations to
their implementation/evidence surfaces. It leaves deployment and current-generation
measurements explicit, including the stale-local-model counterfactual defect. The next
learning episode would measure the complete current split-expert transaction and its
recovery; the comparative study would measure solver behavior and causal benefit.
Neither would be the first demonstration that the mesh API is usable.
