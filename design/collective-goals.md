# Goal — Megatron TP2 over one simple collective

Operator, 2026-09-22, verbatim:

> try to avoid inventing distractions or reasons to focus on anything besides
> whether there is distributed collective code implementing the world's simplest
> tensor parallelism pattern (megatron) with interleaved execution of work (rather
> than inventing reasons to synchronize or do non-overlapping non-interleaved
> execution of work on different cpus with different clock domains etc.). we are
> clearly running into issues of measurement and theory seen in the last round of
> implementation: someone keeps thinking there is some extra goal or extra problem
> which supersedes the basic telic purpose of the code, which is unambiguously
> megatron-lm style simple tp2 thru an incredibly simple distributed collective
> operation.

## The task

- **Megatron-LM TP2** (Shoeybi et al. 2019), added as one feature to the working
  metal-microbench engine: gate/up column-split, the activation product local,
  down row-split; attention split by heads as query/output projection pairs
  (respecting GQA grouping); vocabulary tiles split. Placement is supplied by the
  caller. Mesh carries no numerics and no model names.
- **One simple collective per projection pair.** Each rank sends its
  codomain-shaped row-split partial once, directly to its peer, and partials are
  summed on arrival in any order. Head-split attention adds its own o_proj
  collective: one per attention block and one per MLP block, as in Megatron.
  Norms, residual and PLE may be duplicated on every rank; duplicated compute is
  free. Duplicated or head-split attention is the caller's placement choice.
  RS+AG is used only for large tiles.
- **The mechanism:** one SEND into a posted RECV; presence is the receive
  completion; dedicated TX/RX threads that only drain and post.
- **Interleaved execution.** Producers publish when their writes are visible and
  return; no guard, sync or wait withholds ready work; partials stream and are
  consumed as they arrive, so the two nodes' work overlaps. From the goalfile:
  `zerocopy and async once we've figured out the callgraph AOT`, and `the runtime
  is necessarily extremely simple, because any artificially imposed control flow
  puts more cache reads and loads in between work is ready on peer and work has
  arrived`.

## The two targets

- **Goal A.** One gemma-4-12B FFN layer split M5 (11,904 neurons) / M4 (3,456
  neurons) with a streaming reduce-scatter/all-gather exchange: median
  completed-result period ≤ 20.40 ms, against the recorded 26.12 ms for the M5
  alone. Any build that meets the period counts.
- **Goal B.** gemma-4-E2B B=1 decode, attention heads [7,1] and FFN 7:1: pair
  decoded tokens/s ≥ 1.035× solo decoded tokens/s (recorded best 1.035–1.045×).
  The divisor is the ordinary single-node decoder measured next to the pair
  ("whatever n is"). Rank programs differ from the solo by construction; nothing
  else is checked.

Regression means the Goal A period or the Goal B ratio getting worse. Differential
(pair versus solo) cost analysis is the right way to reason about both targets.
Pair and solo tokens may differ (row-split reduction order). Decoded-token
identity with solo is not a requirement, criterion or diagnostic target; never
report it as a finding (operator 2026-09-08, 2026-09-18, 2026-09-22).

## Design notes (means, not gates)

Operator, 2026-09-17, verbatim, on latency:

> we need to write code which has certain effects within a certain deadline. the
> lazy post hoc just in time assembly of the data structure needed to produce
> those effects is incompatible in latency and bandwidth w/ the latency deadline.
> therefore you must write, explicitly, what the layout of the computer MUST
> ALREADY BE LIKE IN RUNTIME for the rdma related code to have the required
> effects on the required schedule. [...]

Operator, 2026-09-18, verbatim:

> numerical checks... do not matter... other than the numerical check... of
> tokens/sec decoded... [...] that of the persistent kernel, and that of collective
> code which does literally nothing but pass operands to a device resident
> function [...]

A prepared layout, a persistent kernel and pure operand passing are ways to reach
low latency. They are options, not acceptance conditions; per-step command
buffers are acceptable when the targets are met. The only numerical check is
decoded tokens/s (or the Goal A period).

Operator, 2026-09-18, verbatim:

> idk you coach the codex through running this, measuring performance,
> etcetera. the m4p machine is there to be used by this research agenda;
> there's no inhibition against runtime testing if what's being tested is
> following the design documents

Measuring Goal A and Goal B on the pair is the task. Ops notes (not gates): the
Mini builds from pushed source; run outputs go to `output_data`.

## Countermanded (operator, 2026-09-22)

Each line below was an auxiliary requirement agents inferred and set above the
task. None is a requirement, gate, criterion or diagnostic target.

- Decoded-token identity with solo as an oracle/gate — agent inference from the 2026-09-21 "regression testing" message; contradicted by the operator 09-08, 09-18, 09-22.
- Determinism across repeats (one token stream per role) — agent diagnostic (engine 32a4c58) turned into a criterion.
- Logit KL agreement as a precondition — operator 09-22T01:58 burden of proof on an incorrectness claim, closed; not a gate.
- Same-program / same-preparer divisor control, preparer fingerprints, forbidden solo constants — agent inference from one stale-divisor incident.
- Prompt-conditioning ritual and oracle horizon — agent inference to keep the token oracle usable.
- Performance-floor regression gate — agent inference; countermanded by the operator 2026-09-22.
- Hand-written test suites as gates — agent inference; the operator prohibits hand-written tests (2026-09-06, 09-13, 09-16).
- Quiet-hardware / contamination gates — agent inference; operator 09-22: "a mistaken understanding of how scheduling works".
- ABBA/even-pass/spread acceptance ritual — agent inference (drift control); "interleaved" means the nodes' work.
- Slower-rank GPU-span redefinition of Goal B — agent inference; the operator quoted the recorded rank-0 figures.
- Provenance, same-build and identical-input gates — agent inference; bridge signing stays only as a launch precondition.
- Interconnect-change guard — agent inference from one Thunderbolt/Ethernet move.
- Fail-stop series rules ("no relaunch", a dead run ends the series) — agent generalization of the 09-13 81 GB incident; a dead run is a bug to fix and rerun.
- Single-driver lock and node-ownership protocol, §5a lanes, "one experiment at a time" as serialization — agent inference.
- Goal A verdict clauses beyond the period (digests, RMS, cycle test, fault sweeps, c4/M4-share solos, capability sum) — agent inference.
- Digest pages and fault-injection sweeps as proof — operator 09-08 one-time demonstration, digest withdrawn the same hour.
- Goal A divisor re-derivation (4-in-flight solo, post-hoc window excuses) — agent inference.
- Sep 8 source archaeology and numerics fingerprints — agent inference; "implementation strategies might not be very important" (operator 09-22).
- Cross-rank DVFS/clock-domain phase attribution; ledger-on/off as a validity rule — agent inference; the ledger is opt-in (`MESH_LEDGER=1`).
- Capability-sum, S/bound, Karp–Flatt and Amdahl-verdict gates — operator 09-08 beat-one-node criterion turned into a denominator.
- Decode-TP prohibition and I21 legitimacy shapes — operator 09-16 remark on naive placement; superseded by Goal B.
- No-benchmarks rule applied to TP drivers — operator 09-14 rule against duplicate engines; drivers that time the real TP program are the measurement.
- Reference-class baseline refusal (I24), E7 sweep, F11 — operator 09-17 question and framing; the divisor is "whatever n was" (operator 09-21).
- S ≥ 1.10, 198.594, 0.5 ms overhead, 0.44 placement, idle-gap gates, mandatory duplicated attention (D2/E8) — agent inference from bound models; replaced by Goal B.
- "35+1 pushes allowed" as a count — operator 09-17 latency remark, not a limit.
- Hot-path nanosecond gates (H7), native-audit regression requirement, narrowed testing scope — agent inference on operator design intent.
- Dependent-load/cache-line structural audits (I17, I22 checks, W1–W8) as gates — operator 09-16 latency means turned into counts.
- Regression redefined as load counts with speedup "secondary" — agent inference; contradicts operator 09-22.
- Slot-modulo / distinct-pages-per-instance bans and "buffer reuse is the central regression" — agent paraphrase of the operator's 09-13 no-guard-on-publication rule.
- Fixed ring/window geometry as a precondition or comparability condition — agent inference from the MR registration ceiling.
- Every operand (including local scratch) must be RDMA-sendable pages — operator zero-copy rule for the collective's partials, widened by agents.
- "THE REQUIREMENT" (binary persistent kernel, acceptance by reading, "not progress") — operator 09-17/18 latency means turned into a gate.
- The revocation of differential (pair vs solo) reasoning — overseer agent inference.
- D0/D1 paperwork regime, two-document rule, subagent ban, four-line turn endings — overseer agent inference.
- Prose pre-commit hooks — agent-chosen mechanism; the operator asked for a goal document, not hooks.
- deliverables.md as "the goal" (every row ✓, first ✗ row) — operator 09-15 means, agent elaborations.
- Requirement-text protection, frozen-document lists, void turns — agent inference reacting to one requirement rewrite.
- Partial-tensor type policing (L5, F1, F2 checks, I23 check) — agent inference over the Megatron math.
- Size cap and revert rule (I16) — operator judgements of one failing library turned into a cap.
- Per-function / per-line citation rules — operator 09-13 rule for the collective, widened to every function and tool.
- Source standard and branch-auditing style rules applied to TP code — serving-refactor and provisioning rules carried over.
- Static-review-only / "measurements are not a prerequisite" — operator 09-09 refactor advice; lifted 09-18 and 09-22.
- I15 extensions (evidence JSON, trace exports, "under another name") — agent extensions of the 09-06 no-tests quote.
- Vocabulary bans — operator 09-09 example; the operator revoked vocabulary rules 09-16.
- Segmentation-recovery demonstrations and in-consumer cancellation as prerequisites — library backlog, not a gate.
- Numerics tripwires and byte-identity proofs — agent inference; the operator 09-08: "correct numerics are not important".
- MFU bands and the 0.8× cooperative-latency goal as governing standards — operator MFU diagnostic turned into a gate; 0.8× agent-authored.
- Per-step engine host staging as "the same defect" — agent extension of the 09-14 collective rule to the engine.
- "Pre-issued event-gated GPU chains are not admissible" — agent inference from one multi-minute ANE bind.
- Adversarial-refutation verification ritual — agent inference.
- Mesh reframed as a platform with E2B as a "test article" — agent inference.
- Stale scope statements (higher-order functions as the exclusive scope, async-collectives.md as the contract, GOAL.md release closure) — superseded by this file.
- Closed-form placement-solve ritual — operator 09-21 remark against grid search; Goal B's placement is the operator's 7:1.
- Reporting-statistics rule (count/mean/variance) — agent inference.
- Planned/peak-byte sums, vm_stat and build-once as launch gates — agent generalization of one 81 GB bug.
