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
| Native publication and completion identities | Per-transfer lowering treated shared native actions as independent, copying shared storage after an earlier publication exposed it to readers. | `01163a03` retains actual action representatives; `b5155090` consumes them in fanout copy/effect construction. `426340ae` shares that producer between Inductor and the typed emitter. Typed late-plan binding remains active. |
| Exact writer/read causes from `schedule.dependence` | Taking only result `[0]` discarded causes from the existing `dependences_views` call. The fanout draft guessed source versions from generic predecessors and compensated with extra waits. | `b5155090` retains RAW writer/read correspondence, remaps it through assembly and consumes it in copy/publication transformation. No second original conflict solve or current-execution wait workaround. |
| Logical source versions | Planned value keys `(sender, root)` and demand keys `(collective, root)` collapse genuinely different source writes onto one physical/root identity and one release word. | `b5155090` carries logical source versions into existing value/lifetime and transfer operands, independently of physical root identity. Existing allocation owns placement; no second allocator or compensating waits. |
| Torch export input kinds | `export.read` merged PARAMETER, BUFFER and CONSTANT_TENSOR into the same state pair. `driver.lowered` inferred immutability from absence of internal BUFFER_MUTATION, embedding live state and removing its argument. | `e107c5e6` retains InputKind through constant selection, live invocation and archive. Compiled/restored CPU observations cover changed parameters, persistent/nonpersistent buffers and visible buffer mutation. |
| Owned graph-analysis results | Capacity probes in `dependences_views`, `contract`, `contract_unrolled` and `quotient` compute and discard results, then fill calls solve them again. Python wrappers, recorder and its outer Actors sizing/fill layer repeat this work. | `5a43e4b6`, metal `ee02e6c4` and mesh `0dcdcd15` connect owned Conflict/Dependence/Actors results to every inspected caller and delete the solve-twice API. Canonical rebuild and fresh ABI consumption pass. Serialization size queries merely count retained data. |
| Logical transfer extents at native plan emission | Hop emission has logical cut, value, instruction and phase-field operands but stores only physical hop coordinates. `mesh-metal` later sorts physical section addresses and reverse-maps hops to recover logical extents. | `6cec6b10` and metal `442eba53` retain actual demand slices through Planned/archive and consume them directly. Physical inverse lookup is deleted. Canonical rebuild and fresh Planned ABI consumption pass. |
| Typed numerical forms in the old Metal target | `Lowering.reads` calls textual emitters with dummy bindings to discover accesses; `fit`/`settle` consume that reconstruction. `ordered` derives another relation instead of projecting the supplied solved relation. | `426340ae` displaces this emitter/probing body with existing typed numerical construction. Roles, aliases, metadata and task-to-operation correspondence survive. Whole-machine composition, late planning and native execution remain active; construction alone does not close the connection. |
| Existing non-CPU target entrypoints | `targeted` returns residual FX modules and reconstructs placement/receive facts. These callable legacy paths are disconnected from the current CPU-only driver. | Inventory their feature obligations and actual callers, then displace the partial-lowering interfaces with completed typed results. Dormant alternate paths are distinguished from active CPU behavior. No deletion of required target coverage is authorized by this finding. |

| Local storage eligibility proofs | Cost selection reduced solved candidates to roots; installation repeated geometric analysis. | Metal `d6084ff4` retains ordered eligibility candidates through profitability and final admission. Changed ownership/memory limits still affect admission. Included in canonical rebuild. |
| Sending delay and signature identity | Native binding substituted residue-table length for SENT's prior-copy delay; signatures dropped role/slot/generation. | Metal `732f58c9` retains actual delay/startup/identity and uses canonical planner span expansion. Included in canonical rebuild. |
| Immutable numerical construction | `program.make` retired numerical construction with per-plan resources after routing/cancellation changes. Finalization mutates lowered modules, preventing direct reuse. | Active ownership separation and specialization: retain immutable lowered Programs and completed invariant target outputs; bind only changed plan operands. Recovery reset of runtime state is separately identified. |
| Supplied source causal relation | Typed construction keeps task-to-operation IDs but initially conveys source dependencies only as traversal order; native Relation receives buffer conflicts and Later/Step givens. | Active paired input and remapping through transformation to dispatches. Actual source edges must survive; no manufactured serial chain. |
| Actual compiled storage and actors before planning | Early source planning lacks storage/actors created or erased by numerical lowering; a later correction solve duplicates planning. | Active phase correction: source declarations/assembly, typed numerical lowering with actual Actors/Storages, one canonical plan, specialization/finalization. Local preparation reuses Bff and placement without fake bridge events. |
| Selected native archive recipe | Load restores executable/waits but reruns MetalProgramProject to reconstruct selected sites/conditions/ordinals. E2B does not consume native Save/Load. | Active pointer-free recipe retention and relocation. Actual E2B persistence connection remains open. |
| External input/state bindings | The old native route assumes prepared buffers; rebuilding or blanket staging would lose actual source storage/alias binding and add work. | Active root/argument/fixed-occurrence correspondence. Invocation binds addresses/resources; sufficient in-flight binding storage must use existing lifetimes without compensating waits. |
| Exact unchanged-effect conflict results | Conflict frees the solved vector after a Boolean result; fusion/barrier consumers ask related restricted questions again. | Active ownership review/implementation over actual effect geometry and mutation. Changed operands legitimately require analysis; neither stale reuse nor eager unused pair solving is acceptable. |
| Compiled cost-unit provenance | examples/tp.py taken interprets grouped FX; transport reaches an orphan queue interface. Whole-group elapsed does not match its per-node cost table. | Open actual calibration/stage-cost consumer connection, using retained compiled unit/fusion provenance. No second evaluator or residual FX runtime. |
| Captured caller-visible input mutation | export.read rejects USER_INPUT_MUTATION despite upstream retained target/result correspondence. | Active narrow export/runtime/archive connection; no dynamic-control architecture expansion. |

## Review evidence and limits

This update covers committed connections through mesh `426340ae` and metal `732f58c9`, plus explicitly identified active edits. The canonical E2B/native-bench build and fresh imports pass for owned results, extents and sending-generation changes. That build also compiled an unfinished specialization scaffold; it does not establish its runtime correctness or the new target connection. No live bridge/fleet or performance result is claimed.
The preceding local-contract ledger is [composition-progress.md](composition-progress.md); completed local
connections remain premises unless this work changes their producer, representation or consumer.

The fanout worker initially spent about eight minutes reading without source edits. Codex redirected the same
Opus 5.5 session to the agreed producer/consumer checkpoint. Codex then rejected its pre-RECV publication flush,
extra current-execution completion waits and guessed writer versions. These observations concern specific work
and corrections; they do not establish motive. The retained-cause correction is committed in `b5155090`.

The Metal/LiteRT audit also traced retained IRModel → typed export → canonical `metal_ir_append`, saved kernel
LTO objects and final binaries, and ordinary binary loading. These inspected handoffs are connected; the mere
absence of a source IR in a final executable is not a defect. Remaining transform/recording analysis reuse is
still being investigated.

The source-domain audit was corrected against the actual operator turns in the source-lowering request item 2
and composition request's 9 November withdrawal. Data-dependent control flow is excluded there. Successful
Torch export alone does not admit cond/while into this assignment. Static wrappers, tensor attributes and
caller-visible mutations need examination within the assigned fixed graph domain; they are not authority for
an additional control-flow runtime.
