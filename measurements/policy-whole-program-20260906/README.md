# Whole policy program review, September 6, 2026

Follow-up: [static flow refactor](../policy-static-flow-20260906/README.md) consolidates ingress, emission and checkpoint interpretation, shares packed inputs and persists partial transport state. This record describes the preceding review.

The operator requested a complete data/execution-flow review and deletion of all
repository-owned tests. The controlling message is preserved verbatim in
[SPECIFICATION §24](../../design/SPECIFICATION.md#24-whole-program-review-and-deletion-of-tests-september-6-2026).
The resulting implementation is described in one canonical
[policy program page](../../design/POLICY-PROGRAM.md).

## Main findings and changes

| Defect | Completed change |
|---|---|
| Actor-private FFN and raw state/residual projections bypassed the critic's representation | All learned output heads read the same final H. One bias-free `D → 2w+2` projection produces page-rate coordinates and W/L scalar contributions. The common readout is shared by the main model and learned comparison arms. |
| Partial repairs left duplicate and unused tensor computations | Deleted alternate unused Grams, duplicated global-factor projection, unused Gaussian helpers, unused output fields and the separately integrated mean trajectory. Source, local/RPC execution, learning and reporting call the same model owners. |
| Installed actuator state and read age were absent from inputs | STATE now carries six word arrays and 15 headers, including held forcing, anchor residual, actual command timing and each source word's actual last-read time. The first page projection receives 1,556 coordinates at width 256. |
| Vacant observation slots and public topology had incorrect structural treatment | Native PRESENT distinguishes vacant slots. Team-zero topology remains unowned public local context. The unused `relations` tensor and fabricated native packer defaults are deleted. |
| Late events acquired the arrival episode | EVENT18 carries its deposited episode; transport preserves source session/tick/request; memory, journals and reports preserve original episode ownership. Legacy rows lacking that information stay explicitly unattributed. |
| Reports silently substituted smaller source vectors after recovery | One source extraction enumerates the full 30-field Chorus frame both live and after restore. Shared source arrays avoid per-decision copies; reported H excludes prepared page padding. |
| Reports and losses retained obsolete behavior assumptions | Deleted categorical attack/stock/goal/entropy metrics, stale applied-action bindings, the rate penalty, entropy-floor loss and early auxiliary query-value loss. Actor, W/L values and ordinary MoE balancing remain. All actual MoE diagnostics reach telemetry. |
| Tests and duplicate prose were treated as architectural authority | Deleted repository-owned test/verification sources and cached first-party harness copies. Replaced competing current model descriptions with pointers to the canonical program; dated evidence remains historical. |

The earlier VERA query-imitation task had explicit user provenance. Its early
representation tap is retired by the later common-IR instruction, rather than
reclassified as historically invented.

## Evidence

- [Complete input/output inventory](inputs-outputs.md): source fields, native
  read/write boundaries, projection dimensions, actuator identity and remaining
  representation limits.
- [Execution inventory](execution.md): model composition, all output consumers,
  optimizer/RPC/history/reporting ownership, event attribution and remaining
  execution limits.
- [Deleted test-source inventory](deleted-tests.json): individual paths and line
  counts: 53 sources, 8,325 lines, plus 43 cached bytecode files, including
  first-party cached harness copies. Historical measurement
  JSON/logs and vendored dependency sources are separate from these repository-owned
  tests. Existing executable artifacts used by concurrent backend work were preserved.
- [Native and QC compilation record](build/input-contract-build.json): dedicated
  C engine and all three QC programs compiled successfully, with build logs and
  source/output fingerprints. No native services or remote workloads were started.
- Production Python syntax parsing completed for 65 source files under the solver
  and payload tools. The responder's actual `--help` import path completed. Viewer
  JavaScript syntax checking and `git diff --check` completed without errors.

No test suite or substitute numerical acceptance harness was run or created.
Compilation establishes source compatibility; it does not establish live policy
learning, native runtime overhead, or playing strength. This change was not deployed.
Policy versions are matrix 20 / baseline 18, architecture 11; old parameter/native
layouts do not silently become this program.

## Boundaries that remain

Input packing still allocates per prepared frame, growing capacities still trace
new graphs, and split RPC still stages tensors/weights. The custom kernels' fixed
tile shapes do not establish persistent application buffers or a resident remote
optimizer. This review did not modify the concurrently developed backend primitives.

At the default D=128, a page's 512 mean/scale outputs still have local rank at most
128 for fixed head weights. Every captured native word has an actuator coordinate;
that fact does not prove arbitrary instantaneous control. Native capture covers
witnessed reads, not every possible engine intermediate. Applied sequence includes
attempted/faulted view invocations and stock retries. Opponent rows remain globally
available, so local event ownership is not a full partial-observation model.

Complete source histories also remain costly for J reporting, and retained event
histories/identities grow. Missing historical controller attribution and missing
team-composition rating machinery remain visibly missing rather than synthesized
from retired action categories.
