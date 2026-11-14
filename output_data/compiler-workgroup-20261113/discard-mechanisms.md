# Compilation result retention — working review, 13 November 2026

This is a dated source-review and implementation record under
[the workgroup assignment](../../design/compiler-workgroup-2026-11-13.md#concurrent-units).
It is not a new specification or an indefinitely current description of source.
The requirements and cited roles are the existing
[compilation-composition request](../../design/compilation-composition-change-request.md), especially items 2,
3, 6 and 7. `~/schedule-emit` supplies the runnable example of explicit stage composition, not a requirement to
copy its modules, Python executor, or exact types.

## Review coverage

Codex supervisors are following actual producers, transformed representations and consumers throughout these
scopes; Claude workers implement the agreed connections. A search match is a lead, not a finding.

| Scope | Supervisor | Included boundaries |
|---|---|---|
| Python compiler and callers | frontends | Export, placement, devices, schedule, all target entrypoints, runtime, calibration and archive; active fanout correction |
| Native mesh compilation | compiler | Graph results, planner, collective/dataflow program generation, storage and synchronization, saved programs and loading, adjacent native target interfaces |
| Metal and LiteRT compilation | executor | Provider export, retained typed Program, transformations, lowering, recording, native programs, E2B preparation/execution and archive |
| Connected ownership and replacement | root | Reconcile the whole path, preserve established local contracts, assign paired changes, inspect actual deletion of displaced mechanisms |

The inventory remains open while these scopes are being inspected. No whole-codebase completion is claimed by
the first findings below. Final emission may discharge an IR property into executable behavior; that is different
from losing it and requiring another consumer to guess or solve it again.

## Findings and connected changes

| Result and producer | Loss and downstream consequence | Replacement and status |
|---|---|---|
| Native planner repeat operands and synchronization proof results | `plan_signature` expanded one finite proof window, then freed the operands. The Python consumer composed that saved window with arbitrary requested invocation counts. | `61531cca` retains the operands and proof results, shares one expansion, and realizes the requested span before composition. `ddda9ac8` records the complete native/Metal rebuild and installed ABI/archive checks. Old finite records still load; absent repeat operands are not fabricated. |
| Inductor fallback storage aliases | Alias names did not propagate mutations to root/sibling versions or preserve the physical typed writer through per-buffer DCE. | `b0fb5029` connects existing alias facts through upstream mutation outputs and scheduler consumers. `be70e342` records no-hook numerical and actual body/Target/archive evidence. Other upstream producer coverage remains explicitly identified in the patch README. |
| Native publication and completion identities | Per-transfer lowering treated shared native actions as independent, copying shared storage after an earlier publication exposed it to readers. | `01163a03` retains role-valid native action representatives, without serializing addresses or recreating native demand grouping. The paired static lowering consumer is in progress. |
| Exact writer/read causes from `schedule.dependence` | Taking only result `[0]` discarded causes from the existing `dependences_views` call. The fanout draft guessed source versions from generic predecessors and compensated with extra waits. | Retain actual RAW writer/read correspondence, remap it through assembly and consume it during copy/publication transformation. No second original conflict solve or current-execution wait workaround. In progress. |
| Logical source versions | Planned value keys `(sender, root)` and demand keys `(collective, root)` collapse genuinely different source writes onto one physical/root identity and one release word. | Carry logical source version into existing value/lifetime and transfer operands, independently of physical alias/root correspondence. The existing allocation owner places the resulting values; no additional allocator or compensating waits. Paired producer/consumer correction in progress. |
| Torch export input kinds | `export.read` merged PARAMETER, BUFFER and CONSTANT_TENSOR into the same state pair. `driver.lowered` inferred immutability from absence of internal BUFFER_MUTATION, embedding live state and removing its argument. | Carry actual input kind to constant selection, invocation and archive. Parameters and buffers remain live operands, including updates across device binding. Existing placement worker implementing the complete connection. |
| Owned graph-analysis results | Capacity probes in `dependences_views`, `contract`, `contract_unrolled` and `quotient` compute and discard results, then fill calls solve them again. Python wrappers, recorder and its outer Actors sizing/fill layer repeat this work. | Return the already-owned Conflict/Dependence result once; update every actual Python/native/Actors consumer and delete the solve-twice API path. Paired native and consumer workers implementing. Signature serialization size queries merely count retained parts and are not this defect. |
| Logical transfer extents at native plan emission | Hop emission has logical cut, value, instruction and phase-field operands but stores only physical hop coordinates. `mesh-metal` later sorts physical section addresses and reverse-maps hops to recover logical extents. | Retain the logical readiness/sending extents through the existing planner result and archive, consume them directly in Metal preparation, and remove `hop_extents`/sorted inverse lookup. Exact archive handoff under review. |
| Typed numerical forms in the old Metal target | `Lowering.reads` calls textual emitters with dummy bindings to discover accesses; `fit`/`settle` consume that reconstruction. `ordered` derives another relation instead of projecting the supplied solved relation. | Whole target-input/result replacement under review. New hazards introduced by lowering remain legitimate analysis inputs; rediscovering the original graph is not their substitute. |
| Existing non-CPU target entrypoints | `targeted` returns residual FX modules and reconstructs placement/receive facts. These callable legacy paths are disconnected from the current CPU-only driver. | Inventory their feature obligations and actual callers, then displace the partial-lowering interfaces with completed typed results. Dormant alternate paths are distinguished from active CPU behavior. No deletion of required target coverage is authorized by this finding. |

## Review evidence and limits

The first source observations above cover mesh through `ddda9ac8` plus `01163a03` and the active fanout edits.
The preceding local-contract ledger is [composition-progress.md](composition-progress.md); completed local
connections remain premises unless this work changes their producer, representation or consumer.

The fanout worker initially spent about eight minutes reading without source edits. Codex redirected the same
Opus 5.5 session to the agreed producer/consumer checkpoint. Codex then rejected its pre-RECV publication flush,
extra current-execution completion waits and guessed writer versions. These observations concern specific work
and corrections; they do not establish motive. The retained-cause correction is now the assigned implementation.

The Metal/LiteRT audit also traced retained IRModel → typed export → canonical `metal_ir_append`, saved kernel
LTO objects and final binaries, and ordinary binary loading. These inspected handoffs are connected; the mere
absence of a source IR in a final executable is not a defect. Remaining transform/recording analysis reuse is
still being investigated.
