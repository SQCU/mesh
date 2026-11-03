# Static policy flow review, September 6, 2026

The operator requested another review of the overall data and execution flow,
including deduplication, extraction of inlined operations and functional refactoring.
This follows the [whole-program review](../policy-whole-program-20260906/README.md).
The current implementation is described in [POLICY-PROGRAM.md](../../design/POLICY-PROGRAM.md).

## Findings and changes

| Source defect | Result |
|---|---|
| Native input assembly was divided between a long responder block and a generic assembler with fabricated optional fields | `inputs.native_inputs` owns native owner joins, page/header assembly, visibility and navigation support, producing the complete 30-field Chorus tuple. The old assembler was deleted. |
| The shared source frame was created through whichever learner appeared first | `Frame.capture` owns source identity and byte size independently of either learner. Replay stores references to that shared frame. Its unused interning API and serial were removed. |
| Every policy repacked the same source; realization and historical training repeated packing too | `PolicyProgram` shares each source/capacity packing within an emission or learner update. This cache contains raw input tensors, never learned representations or predictions. Different capacities remain distinct. |
| Mixed-policy evaluation, comparison capture, assigned-row emission and recovered-action integration were inlined in the responder | `execution.emit_policies` performs this sequence once through the existing policy programs, preserving each policy's complete counterfactual outputs and publishing assigned rows. |
| Inference checkpoint loading accepted compatible versions/shapes without enforcing native schema and reward semantics | Inference and continuation now use the same metadata and whole-parameter-tree interpretation. Both compare architecture/native schema, version and reward. Continuation requires the exact arm; inference permits declared architecture-equivalent arms with matching rewards. Failure reports retain the complete available model. |
| RPC requesters were embedded in the responder and rebuilt custom differentiation wrappers on calls | Both requester classes now live beside their worker operators in `scale_rpc.py`. Each instance reuses its wrapper; routed wrappers are keyed by configured top-k. The operation and VJP arithmetic are unchanged. |
| Runstate persisted event/watermark subsets but omitted pending snapshots, partial reassemblers and packets deferred during RPC waits | `RuntimeFrames` exports/restores its own complete received state, including fragment bitmaps. Runstate also saves the game-packet backlog. Legacy checkpoints restore only the fields they actually contain. |
| Journal recovery advanced watermarks separately from live snapshot consumption | Both now use `accept_snapshot`, which retires sessions and removes already-consumed pending snapshots. Recovery also removes the exact consumed event assemblies, preventing restored snapshots from being processed again after replay. |
| Live and recovered view reports separately selected source words; spill/export separately stripped regenerated reports | Shared functions own view-word selection and retained source rows. Restored reports retain the same source features and address labels. |
| History accounting omitted arrays retained in the issued-action record | It now counts issued velocity, likelihood and ownership arrays, deduplicated by array identity, alongside source/report arrays. This remains retained-data accounting, not a measurement of total Python process memory. |
| Learner transition defaults could fabricate actor ownership, identity successors or returns when a caller omitted actual provenance | The sole producer supplies these fields explicitly and the learner requires them. The second reward fallback was deleted. Existing reward formulas, value targets and actor eligibility are unchanged. |
| Model construction duplicated almost identical ablation forwarding functions; runtime records copied unused derived state | Static fusion strengths select the same model composition. Unused wrappers, fields and responder helpers were removed. |

## Responsibility and execution map

| Stage | Owner and consumer |
|---|---|
| Receive/reassemble and synchronize | `xonwire.RuntimeFrames` → responder records; `buffers.ObservationMemory` retains original event episode ownership. |
| Assemble complete model inputs | `inputs.native_inputs` → `ChorusArrays`; `Frame.capture` → shared source history. |
| Prepare and evaluate both models | `execution.PolicyProgram` / `emit_policies` → common model composition and heads, complete comparisons, assigned rates. |
| Publish and integrate | `StatePages.response` → native STRATEGY command; native private-view exponential integration returns actual installed actuator state and applied identity in STATE. |
| Attribute and optimize | `ActionHistory.advance` → required transition source fields → `OnlineLearner`, using the same `PolicyProgram`; replay retains actual source/successor frames. |
| Continue after restart | `checkpoint_state` owns parameter semantics; learner owns optimizer/history; `RuntimeFrames` owns transport; responder owns the runstate transaction and action-journal position. |
| Report | `source_features` and `view_observation` reconstruct source reports from the retained Chorus frame both live and after recovery. |

The two live policies remain separate parameter trees with their respective
objectives. Both evaluate the same source observations; each actor learns from its
own first observed applications, and value learning can use other teams and
retained experience. No historical-opponent league was added or made a requirement.
This refactor changes no head parameterization, native word layout, reward formula
or model version: matrix 20, baseline 18, architecture 11.

## Evidence and limits

[Source review record](source-review.json) records the actual production syntax and
CLI import checks, their results, and source-file fingerprints. AST parsing covered
65 production Python sources. The responder, matrix worker and curriculum `--help`
entry points completed successfully. `git diff --check` passed. Source comparison
used snapshots taken at the beginning of this turn where available; the worktree
also contains earlier and concurrent changes, so this record makes no claim about
the entire Git diff or a clean checkout.

No tests, substitute numerical harnesses, model evaluations or native builds were
run in this follow-up. No services, remote workloads or RDMA primitives were changed
or exercised. Import and source checks do not establish crash recovery under live
traffic, gradient correctness after wrapper reuse, training throughput or match
performance. Native build evidence belongs to the preceding review.

Received transport state survives at the checkpoint boundary. Packets never
received and fragments received only after the last checkpoint remain outside that
claim. Recovery restores the saved transport framing parameters; a migration to a
different packet framing is not established by this review. Saving incomplete
assemblies increases checkpoint volume by their retained staging storage.

Each distinct source/capacity packing still allocates, and capacity growth still
realizes another graph. Sharing duplicate packs is not a persistent-buffer arena.
RPC still stages tensors and weights rather than owning a resident distributed
optimizer. Complete source reporting and retained event identities still grow with
their actual input history. These remain concrete execution costs; no feature
reduction was introduced to conceal them.
