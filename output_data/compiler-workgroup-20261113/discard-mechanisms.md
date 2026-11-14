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

## Subsequent source review and immediate graph consolidation

This addendum is a checkpoint through mesh `93e2c196` and metal `9a548ba3`, with
the uncommitted work below explicitly distinguished. It supersedes the earlier
table's status for these connections, not the governing requirements.

- Mesh `e1c6b2da` completes captured input mutation through export, caller aliases
  and the reference Target/archive path. Final native mutation destinations remain
  part of the ongoing typed consumer connection.
- Metal `c0d31bb6` separates immutable numerical construction from specialization
  and retains completed invariant target emission. Mesh `16261afc` produces a local
  plan from actual compiled Actors and Storages; `6ebf9690` records the rebuilt
  Storage24/Local136 interface. Subsequent Local144 save/load edits are active.
- Metal `9a548ba3` retains unaffected function Work results and batches only
  undecided access pairs. The preceding conflict cache avoids repeated geometric
  solves, but still reduces original records to hazard kinds. It is therefore
  being connected to the common retained relation, rather than treated as a
  completed information-preserving interface.
- Metal `8cf23a29` retains external-root and argument-table operands in native
  artifacts. Actual invocation and output ownership are still being connected;
  that serialization commit does not establish working execution.

The immediate operator instruction is recorded verbatim in the workgroup
assignment: consolidate graph algorithms in the existing `mesh-graph.h/.c` and
keep solved data available through views. Review found three unfinished consumer
boundaries: Python graph algorithms and per-wrapper reconstruction; native
parameter-dependence traversal and reduced conflict caching; recorder quotient
and synchronization code that retains only edge pairs. The native graph owner is
now being extended to retain original witnesses, edge-to-witness rows, and derived
edge-to-parent correspondence. The directed search implementation is being shared
with the existing preorder entrypoint. These source changes are not yet claimed
as reviewed, rebuilt, connected completion.

The first Python migration draft reconstructed Rows for each wrapper and copied
native results into lists or sets. Codex rejected that draft because it moved
algorithm bodies without completing the retained-result handoff. The same Claude
session is replacing it and connecting the actual schedule consumer. A separate
review found that typed construction ignores existing `out`/RECV `to` destination
views for some pure operations. Its fresh temporary loses assigned storage
identity; the correction is to pass the existing destination into the existing
lowerer output operand, not add another allocation algorithm or dismiss the
original hazard witness.

The next canonical build is coordinated around the new native result interfaces.
The previous successful E2B/native-bench build covered Local136, Storage24 and
native v30; it supplies no execution or performance evidence for these subsequent
uncommitted graph and invocation changes.

The next canonical E2B/native-bench build passed. The reviewed working checkpoints
are mesh `26861434` (shared traversal and retained relation producer), metal
`09a604f1` (native parameter search consumer), and metal `4db95f8d` (original
geometry ownership, full witnesses, and shared function-graph traversal). Mesh
`7753bdcf` and `87906007` connect local plan persistence and the machine caller;
metal `8cd3eab2` connects native external arguments. Remaining work at this gate:
Python closure must consume the same retained graph; recorder Actors/Prepare
must consume relation views; original source witnesses must seed the native
geometry results; function-graph results must survive related queries; native
relation archive load must relocate the retained projection instead of rebuilding
it. These are active paired consumer assignments. Build success does not close
them or establish runtime/performance equivalence.

## Caller review checkpoint, 14 November UTC

The operator's immediate instruction is: "\"Those callers are centralized only
partway.\" then this must happen immediately". This checkpoint records evidence
and unfinished handoffs; it adds no requirements.

- Mesh `9388eb3e` and `355c8399` connect the original query, retained closure order
  and full graph archive. Subsequent origin/map additions remain active and need
  a matching rebuild; these commits do not validate that later ABI.
- Mesh `071378b5` connects source relation and operation/value correspondence to
  the native constructor, retains source producer and random-site annotations,
  and removes the unreachable old emitter. Native source now imports exact
  source witnesses. Pairwise alias correspondence and immutable shared maps are
  under review. Numerical source-cause discharge remains unfinished: grouping
  all operations of one source vertex, or inventing root-wide pair obligations,
  was rejected. Actual operation dependencies must feed the actual fusion caller.
- Metal `2b507039` and mesh `0574f658` retain recorder and actor result owners
  through their immediate consumers. Generic archive origin/map persistence and
  the legacy network actor archive are still being connected.
- Mesh `d0336fc0` moves event redundancy and class-local source/sink calculation
  into the shared graph implementation and connects scheduling to those results.
  Review is following the event result into the actual function/planning consumer;
  the local graph's existence alone does not close this handoff.
- Metal `945f0661` and mesh `fc55e40b` replace Target's buffer-copy-count/LCM
  reconstruction with a direct read of the prepared program's existing warm and
  period fields. Mesh `04511507` removes unused AOTI wrappers and their explicit
  build rule after checking their callers; no replacement execution mode was added.
- The existing CPU FFN client completed with error 0.0 after `448d47c7` corrected
  a read that accidentally created a STEP value. Its native artifact was reloaded
  by Target.prepare. This is neither an outer archive.save/load measurement nor
  evidence for GPU, transport or comparative performance. The next matched build
  and those specific caller measurements remain assigned to their existing owners.

Corrections are being made in the same Opus 5.5 sessions, with Codex reviewing
the producer, retained result and consuming call together. Moving an algorithm
body, retaining an unused field, or passing a build does not by itself establish
that a caller stopped discarding and reconstructing the result.

## Subsequent connected checkpoints

- Mesh `04e1eed3` / metal `79d7e7a6`: full retained query DAG and correspondence
  maps, actual operation-relation fusion caller, and original function-analysis
  results. The matching canonical build passed. Mesh `3fc7f3d7` records the
  existing CPU client and outer filesystem archive run: errors 0.0 and
  6.902263703523204e-05 for two distinct inputs; unchanged signature; fresh
  outputs; first output unchanged after the second invocation. These are
  observations, not an invented numerical tolerance or GPU/transport result.
- Metal `bc2be2c3` removes the copied dependence arrays and obsolete accessor:
  dispatch ordering and replay ordering read the retained relation through row
  views. Metal `32f4960a` completes all callers of the one retained-relation
  registration function and removes its unused copying wrapper.
- Mesh `bbb7ed1f` records the exact source value/channel carriers. Review found
  no missing execution link among those current producer variants. The proposed
  extra quotient for interleaved fusion groups was withdrawn after inspecting
  the actual consecutive-group producer; no speculative machinery was added.
- Mesh `c15f33b1` connects the event result's native dependence array through
  Function/Plan without reconstruction. It also includes unchanged concurrent
  Function/archive ABI edits; their validation remains owned by the compiler
  workgroup, separately from the event-array address/lifetime observation.
- Metal `df43fff9` connects preparation to retained incoming rows and one DFS
  forest result. The initial per-query retained-search draft was corrected:
  caching whole searches would use quadratic storage. The actual caller now
  consumes one linear-size result and canonical ancestor queries.
- Metal `0a506f6c`, `3d9c8334`, and mesh `7fc00282` connect fixed GPU argument
  sites, borrowed backing ownership, and actual native/archive consumers.
  Their new matched build and client observations are pending at this checkpoint.

The remaining discovered public callers are being assigned as complete local
handoffs: finite sharding propagation; call-analysis paths/width over retained
results; and first/last use lifetimes. The latter is a concrete binding defect:
two remaining callers use the old two-argument helper while a later change gave
it three arguments for a now-displaced consumer. Exact Fraction partitioning
also has two implementations; the callers will share its existing owner without
changing numerical precision. These assignments do not reopen numerical
lowering, create another evaluator, or authorize deleting supported public paths.
