# Policy feature integration — September 6, 2026

This directory records the removal of unsupported feature preprocessing and the
extra feature-Gram/probe branch, restoration of local v-cell integration, and
sigmoid routed SwiGLU with ordinary balancing loss and no shared expert.
[The provenance audit](../../design/FEATURE-GRAM-PROVENANCE.md) separates original
user requirements from the assistant's September 1 benchmark-derived change.
[The current axis contract](../../design/POLICY-AXIS-AUDIT.md) lists every raw
row family, source projection and explicit contraction.

## Evidence

| Artifact | What it establishes |
|---|---|
| `provenance.json` | Selected original user and assistant messages with source files, line numbers, authorship and timestamps |
| `policy-axis-audit.json` | Literal input widths, 42 logical matrix contractions, explicit reductions, row permutation/isolation and effect of raw fields |
| `validation.json` | Final unit-test outcomes and scoped execution measurements |
| `native-runtime.json` | Actual native engine, bots and optimizer; literal raw features in replay; navigation/age gradients; event-history and outcome recovery |
| `native-runtime-notes.md` | Native test setup, deliberate late-event/ownerless input cases and preserved failures |
| `rdma-cross.json` | Before final input-FFN prenormalization: eight actual remote calls with no local fallback; rectangular forward and both VJPs; changed native inputs through prepared regions; all policy parameter gradients |
| `rdma-runtime.json` | Failed large native RDMA test: too few optimizer updates; bridge processes changed during independently confirmed backend R&D; final-backend cause remains inconclusive |
| `rdma-runtime-before-receive-fix.json` | First full RDMA match timed out below eight optimizer updates while large STATE assemblies repeatedly restarted; bridges remained healthy |
| `execution-local.json`, `execution-mini.json` | Current host and earlier Mini prepared graph reuse and actual Metal buffer/heap allocation observations |
| `input-ffn-prenorm-validation.json`, `input-ffn-prenorm-counterfactual.json` | Residual FFN magnitude correction, identical-weight native-frame comparison and focused tests |
| `source-sha256.json` | Source identities for this completed validation; hashes identify evidence and are not deployment selectors |

The final host suite passed 129 tests. The final isolated native test used
matrix-policy version 19, baseline version 17 and architecture version 10. It
delivered 1,336 raw events, made 15 optimizer
updates and attributed all 20 observed actor rows, with no view faults. It retained
a deliberately late first EVENT frame and 24 events received in an ownerless
snapshot. All seven neighborhood parameter groups changed, including the age
coefficient and local Gram metric. Its 20 checks include graceful continuation,
abrupt exit after a durable action, journal replay, repeated terminal recovery
and engine disk-outcome replay. See the report for the exact measured counts.

The separate RDMA cross test, run before the final input-FFN correction, used a
257-by-13 left matrix and a 257-by-19 right
matrix. Forward and both operand derivatives matched locally exactly in that
sample. It then used a captured complete native frame through the actual policy;
changed inputs affected outputs while source/finish/decode each retained one
trace. The largest full parameter-gradient discrepancy was `3.73e-9`. It validates
that recorded version, not a complete RDMA rerun of final version 19. Earlier
Mini tests passed 126 checks; the later focused receive and prenormalization
suites passed 25 and 70 checks respectively. The final complete 129-test suite
and final native run were host-only.

On an identical-weight eight-bot native frame, the unnormalized input SwiGLU
branch reached maximum magnitude 2,058. Normalizing only that branch's input
reduced its maximum to 0.490. The embedding residual retained its original
magnitude; native fields still reach their first learned projection intact.
Tests check the residual bypass, bounded branch, row isolation and derivatives.
This correction reduces a demonstrated amplification; it does not establish the
exact cause of earlier native resource-type faults or eliminate all invalid
states reachable through unconstrained exploration.

Native working directories and checkpoints remain on the local machine under
`runtime-*` and `rdma-*`; their bulky contents are ignored by version control.
The compact reports identify those paths. Earlier failed runs remain available
for reproduction instead of being overwritten by the passing run.

## Receive-loop failure

The first full game-over-RDMA test transported state snapshots of 12,459 frames.
Its policies did optimize, but completed input snapshots were too sparse to reach
the requested update count in 180 seconds. Reassembly reported incomplete STATE
requests being replaced. Both bridge processes and configured capacities survived
with no increase in their bad-frame counters.

Inspection found that `Mesh.read` could keep yielding batches until the receive
ring became empty. The responder called `RuntimeFrames.take` only after that drain,
so a continuous producer could prevent the policy from consuming already complete
snapshots. The correction bounds each responder/RPC receive pass by complete
borrowed batches, preserving every row in a batch before reevaluating ready work,
response completion or cancellation. This is a scheduling correction; it does
not establish the cause of every missing transport fragment.

The second large RDMA run also failed its update threshold. The user confirmed
another session was developing distributed primitives and reductions, and bridge
PIDs changed during that test. Counter deltas spanning those replacements cannot
identify packet loss. The separate source-level completion-ring discard path and
the test's native view faults are recorded in `receive-fairness-rca.md`; neither
is erased by the concurrency finding. Further transport validation is deferred
while that session works. Our final validation uses an isolated local transport,
and all owned test engines and workers have exited.

## Limits

The final local graph-reuse probe passed, but its allocation check failed: zero
new Metal allocations during eight forward/emission calls and 33 during eight
optimizer calls, after warmup. The earlier local result (three optimizer
allocations) is preserved in `execution-local-before-prenorm.json`. The earlier
Mini sample recorded zero allocations and stable graph traces. These samples
exclude host packing, reporting, transport and capacity growth. Persistent
allocation-free execution remains unmet; fixed prepared shapes alone do not
establish that property.

Navigation support is measured in serialized metric-graph length. It is not a
walkable-floor-area measurement. Per-cell support outside the requested 5–15%
band is reported when the graph's components or cell granularity make the band
unattainable. Learned representations remain finite-rank, and raw numeric IDs
remain available to learning; row permutation invariance does not imply arbitrary
ID-renaming invariance. These checks establish data flow, gradients and execution
recovery, not playing strength or sustained many-team performance.
