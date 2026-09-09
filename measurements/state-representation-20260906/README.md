# State representation and fixed-execution evidence — September 6

The implementation and causal review are in
[POLICY-STATE-STEERING.md](../../design/POLICY-STATE-STEERING.md) and
[STATE-REDUCTION-RCA.md](../../design/STATE-REDUCTION-RCA.md).
The operator quotes remain the specification. This directory records both successful
checks and failures; allocation-free execution is not marked complete.

## Final results

| Check | Evidence | Result |
| --- | --- | --- |
| Representation, gradients, grouped optimizer equivalence, continuation, paged kernels, telemetry and HTTP coordinate paging | [Laptop](laptop-tests.log), [Mini](mini-tests.log) | 62 tests pass on each machine |
| Warmed forward/emission and two-source combined update | [Laptop](laptop-execution.json), [Mini](mini-execution.json) | One trace each for forward, emission and update; eight measured solves of each mode |
| Device allocation probe | Same execution files | Laptop: zero forward/emission allocations, two allocations during grouped updates; Mini: zero in both modes. The laptop allocation check fails |
| Native view/integrator fixture | [Native report](laptop-native.json) | Independent views, typed words/references, movement, head-up/head-down control and finite-difference checks |
| Final native match with isolated socket/shared-memory transport | [Runtime report](laptop-runtime-final.json) | 15 updates, 17 actor rows, 509,952 maximum represented state words, zero unmatched rows, zero view faults; interruption/journal/terminal/engine recovery checks pass |
| Final native match over resident RDMA bridges | [RDMA report](rdma-reporting-repair.json) | Both Mini policies reach 14 updates, resume and receive durable outcomes; 104 matched applications, zero unmatched rows, zero new bridge errors |
| Bridge state after application exit | [Original report](rdma-reporting-repair.json), [passive follow-up](rdma-post-exit-recovery.json) | Original assertion fails on `retiring` snapshots. Follow-up sees both paired/responsive with unchanged PIDs and capacities; no bridge was signalled or restarted |

The RDMA harness has been corrected to observe retirement/re-pairing for up to
30 seconds and to preserve independent cleanup errors. The original failed snapshot
assertion is retained as recorded, not rewritten into a successful harness exit.
The application recovery checks in that run had already passed.

The final allocation harness exercises the actual combined loss/backward/optimizer
program with fresh actor and value-only source groups. It reports graph reuse and
allocation reuse separately and exits nonzero when either fails. Its process-local
probe counts Metal device `newBufferWithLength:options:`,
`newBufferWithBytes:length:options:` and `newHeapWithDescriptor:` calls, calibrated
with a forced allocation. It does not instrument every private driver or heap
suballocation API. Input staging, transport, reporting and capacity realization are
outside the timed numerical region. The two final laptop allocations were 1,972
bytes each. No allocation-free guarantee follows from the Mini's finite clean window.

Median warmed durations in this configuration were about 1.46 ms for laptop
forward/emission and 8.49 ms for the two-source update; Mini durations were 1.96 ms
and 10.55 ms. These are a small fixed-capacity numerical workload, not full-match
throughput, thermal balance or hardware FLOP-counter measurements.

## Failures and intermediate checks retained

- [Pre-repair representation witness](../state-representation-audit-20260906.json):
  fixed-address value swaps, policy-uniform page increments and missing residual
  feedback. It predates the repairs.
- [First RDMA failure](rdma-verification.json), [learner log](rdma-checkpoint-failure.log):
  expanded checkpoint labels exceeded NumPy's string representation. The string-table
  and array-index codec repaired it.
- [Initial successful RDMA rerun](rdma-final.json): 73 matched applications after
  the checkpoint fix, before masked kernels and the combined update.
- [Masked-kernel RDMA run](rdma-masked-kernels.json): 96 matched applications before
  the final combined update and report serializer changes.
- [Report expansion failure](rdma-report-expansion-failure.json),
  [learner log](rdma-report-expansion-failure.log): both policies trained, but a
  roughly 3 GB label manifest and 384 MB viewer JSON delayed shutdown. A second
  cleanup error prevented the old harness from writing its normal report.
- [Final learner log](rdma-final-learner.log),
  [resumed learner log](rdma-final-learner-resumed.log) and
  [artifact sizes](rdma-artifact-sizes.json): the shared lossless codec and binary
  columnar sidecars were exercised in the successful application-recovery run.
  Completed full artifacts were 208–390 MB, with 17–21 MB sidecars and roughly
  34 MB full-artifact manifests. Observation windows differ from the failure run.
- `emission-*-investigation.json`, `fused-*-allocation*.json` and
  `prepared-group-allocation.json` retain allocation investigations. Synchronization
  and extra warm-up did not reliably satisfy the allocation requirement and were
  removed. The earlier `laptop-runtime.json` is the first full-row socket match;
  `laptop-runtime-final.json` covers the final implementation.

These are historical measurements. The operator removed the test suites and
verification harnesses on September 6; their assertions do not define the policy
specification.

[Source hashes](source-sha256.json) identify the files examined; they are evidence,
not fetch selectors or deployment pins. The repository had extensive pre-existing
changes, which were retained. Competitive playing strength, universal policy
expressivity, persistent input/intermediate arenas, complete split-RPC backward
staging, worst-case workload throughput and RDMA driver-deadlock recovery remain
separate unresolved claims.
