# Memorandum: the order in which the outstanding compilation requirements are implemented

2026-11-09. Written by Claude at the operator's request, from the session
`.claude-personal/projects/-Users-mdot-metal-microbench/55974018-f56d-4c8b-8973-6fc61b49140c.jsonl`.
Code was read at mesh 7df87fe4 and metal-microbench 7483572e; the reference program at `~/schedule-emit` 881a347.

## The turns

The operator, 2026-11-09, in order:

> "The rest widened operator coverage and built observation machinery around an executor that was already slated for replacement." an unfortunate decadence that seems to hang over much of these proejcts... if we review... yet again... what of the requirements presented in the change request documents et al  entail subsuming and replacing all features and responsibilities of particular modules, what are the modules in mesh and microbench which need to reach total closure in what order to mitigate/prevent these procrastinatory deluges of edits to obsolete dependent code?

> do our external reuqirements for meshlib allow arbitrary control flow inside of programs or do they require exactly the opposite? Review this. we shall not ocnsider arbitrary python to be part of arbitrary torch; programmers who want to run arbitrary data dependent control flow between sections of linear algebra are to learn to write that sort of code on their own time and outside of our mandate and the library features we provide, which are specifically for torch like tensor apis w/ scheduling, transport, and function fusion as needed to support lowest attainable overhead scheduling and transport.

> contiguity is a scheduled operation that must be justified by a solver; in torch making a tensor contiguous is a *call* which necessarily introduces extra overhead in redundant writes, overconstraint on the placement of literal memory in literal addresses, etcetera. [...] arbitrary extra constraitns includes things like guards, syncs, and waits, by the way, if it somehow wasn't discernible.

> correctness is not a malleable property where code can be made 'more correct' by adding more guards, syncs, waits, calls, constraints, overrides, etcetera; all of those are divergent and unspecified behavior; at least three full weeks of rewrites in this repository happened *only* because agents decided to invent new specifications and requirements at every interface between functions. the slightest hint of modesty or restraint would have resulted in formally correct compile time and runtime callgraphs which preserved every program property the repo gave treatment; the interface-level indulgences specifically and uniquely added errors and runtime faults. if there is anything you should be 'adding' to change requests or other templated work schema for subagents, it should be commentary on only the virtue of retraint and modesty in their assigned work, not presumptions of how problems ought to be approached or interpreted beyond templated work orders.

> what are the requirements from the outstanding topics in the change request documents whose path to implementation is now clarified by the counterfactual program-over-IR and lowerer? this should give us a definite list of requirements and an order in which they must be implemented, alongside a rough estiamte of the numebr of lines of code each ultimately requires if tackled in the right order and by centralizing responsibilities into modules rather than unextensible arboreal inlined sprawl

> document this as a memorandum which cites the change request document literals so it's easier to 'remember' this ordering across whatever subagent mediated work will be involved in its implementation

The operator, 2026-11-09, later, of a report that item 4 stopped because "Nothing else supplies that order to
e2b_replay, and the plan runs before pages are placed, so it cannot take the job over yet":

> well our task here isn't to rehabilitate code but to categorically replace it, including categorically replacing inverted control flows e.g. situatinos wehre we pretend client prgorams are running 'before' 'pages' are 'placed'

The operator, 2026-11-09, later, of a report that item 4 stopped on "Programs that are given no plan":

> return an error and halt? we don't need a fallback path and hte good faith reading of our requirements is to cull code paths that interact with 'fallbacks'.

The operator, same turn, of "The per-value access record. No form is stated for five cases: which copy an access
uses, frame tables, transport words, fixed buffers outside the mapping, and the shares mark" and of "Host and device
parts of a call":

> what does this actually mean...?

> we're talking about apple silicon which doesn't have a host-device model, it has shm, command buffers, and cooperative tensors? what is the question here asking in non-idiomatic language?

The operator, 2026-11-10 UTC, of Claude's sentence that the solve "needs to know which values each one reads and
which it writes" and that "the replacement states it early":

> so this early topic is relating to one of the unimplemented change request documents, which ultimately describes blah blah blah loop unrolling and dependency ordering optimization solvers as producing exactly the data structures needed to produce indices of which pages (literally, an abstraction over pointers) are 'safe' or 'available' for reading or writing at any particular moment in execution, as the optimizing solver produces memory safe scheduling-allocation solutions by coincidence of producing optimal solutions.

The operator, 2026-11-10 UTC, of plan-to-bridge-change-request.md R4:

> "- Coflow ordering is cited but not required. R4 orders the blocking kind by R5's static schedule; Varys's smallest-bottleneck-first is named and explicitly not asked for." then review the change request, rewrite it if necessary s.t. it establishes that this optimizing solver output / scheduling / lowering property is not an evocative metaphor but is in fact a plain behavioral requirement for runtime execution. if there is no source applying this, we can simply accept that it is an outstanding requirement taht needs placement in the implementation ordering of our current data flow and architecture work

The operator, 2026-11-10, of the review hooks' findings on the commits of this order (comments that are not
citations, graph walks outside the graph module, and the change-request reviews):

> this, alongside all other inlining and review results, must be acted on as part of this review i suppose

The operator, the same day, of no item having closed:

> is there a vast citational gap we cannot close or do we need to revise our change request documents to be totally closed as specifications of implementations instead of leaving a bunch of 'stops' or apparent free choices up in the air?

## The reference program

The reference for this memorandum was `~/schedule-emit` at 881a347, a separate repository. The source inventory
reported 724 lines and 31 functions. Those counts describe that revision, not its execution or coverage.

## The requirements, in order

A later item consumes what an earlier item produces. The order below describes those data dependencies.

### 1. The scope is static capture; control flow that depends on data, and shapes known only at run time, are rejected once, at export

- **Clauses:**
  - compilation-composition-change-request.md, "The turns": "supporting all pytorch programs which would be
    cudagraph compileable".
  - aot-compilation-change-request.md, "A1. The input is each program's captured dataflow graph, with shapes and
    types": "the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'".
  - torch-source-lowering-change-request.md, "2. The capture is a static dataflow graph".
  - The operator's second turn above.
- **Replaces:** the handling of IF and WHILE whose condition is tensor data: the loop stack, predicate reads and
  drains (`MetalLoopTest`, `MetalDrain`), trip-count observation, and the scoped regions of
  compilation-composition-change-request.md "Nested communication regions: producer and consumer handoff", struck
  on 2026-11-09 ("Withdrawn, 2026-11-09" in that request).
- **With it:** torch-source-lowering-change-request.md item 5 (2026-11-09), for the operators whose lowerings repeated
  a body while a predicate on tensor data held: each is one call of a supplier's function, or is refused with a plain
  message.
- **Reference:** `schedule_emit/export.py`, 62 lines.
- **Estimate:** 60–120 lines. It removes scope, so every later item is smaller once it is done.

### 2. The values that pass between stages have their types in one shared header

- **Clauses:** compilation-composition-change-request.md, "3. Put each representation and decision in its
  owner"; "6. Carry compilation results through their consumers": "finish each producer → representation → consumer
  connection before extracting auxiliary helpers"; the operator's turn quoted there: "are w running into integratino
  errors where two interfaces 'don't line up well' because they are literally serializing and deserializing
  equivalent or related types over and over again".
- **Replaces:** records that restate another module's result. About 157 struct, enum and class types were declared
  across the files written on 2026-11-09, 45 of them private to `metal_recording.m`.
- **Reference:** `schedule_emit/ir.py`, 76 lines: two operators and two data types.
- **Estimate:** 100–200 lines.

### 3. Placement is an operand; a value that crosses processors gets a send and a receive as nodes of the same graph

- **Clauses:**
  - compilation-composition-change-request.md, "2. Compose the requests on the same program's representations":
    "a value's sharding, consumer kind, accesses, lifetimes, transfers and dependences describe the same program
    through lowering."
  - plan-to-bridge-change-request.md, "R3. An execution is a task graph"; "R8. A program is its function calls".
  - torch-capture-change-request.md, "9. Every node replays": "every node runs its captured program by replay, its
    collectives included."
  - The operator's third turn above, for any copy this stage adds: "contiguity is a scheduled operation that must be
    justified by a solver". A copy is added only where the solver justifies it, and it is then a scheduled operation.
- **Replaces:** compilation-composition-change-request.md, "1. Finish lowering the whole assigned domain": "the split
  in `rdma/mesh_compile.py program()` where `replayed` excludes nodes whose declarations name transferred values";
  "5. Consolidate repeated implementations at their shared owner": "`mesh_compile.py _outputs` and `_natives`: both
  traverse the same call graph".
- **Reference:** `schedule_emit/placement.py` and `devices.py`, 107 lines.
- **Estimate:** 300–500 lines.

### 4. One scheduling solve orders functions and transfers together, with cost as an operand

- **Clauses:**
  - compilation-composition-change-request.md, "3.": "The second scheduling solve, copied graph walks, runtime placement of
    program values and alternative execution paths are the existing requests' named replacements, not independent
    facilities to maintain beside their replacements."
  - "4. Execute the compiled decisions with the specified runtime semantics": "placement and costs arrive as
    operands; compilation realizes the graph's storage, transfers, ordering and synchronization."
  - plan-to-bridge-change-request.md, "R5. The order of transfers is a static schedule with costs".
  - plan-to-bridge-change-request.md, "R4. Functions that need an operand whole and functions that take pages in any
    order": "The code must schedule both kinds of function, and must not tell the bridge which is which." Its
    "Behaviour at run time", statement 3: "Each cable sends its pages in the order R5's program gives."
  - disjoint-pages-change-request.md, "Unrolling by modulo variable expansion" (M. Lam, PLDI 1988; B. R. Rau, MICRO
    1994): "For each value, L is the smallest lag at which every read of it in execution t happens before its write
    in execution t + L. The unroll factor u is the largest L." Which pages may be read or written at a position of
    the schedule is this solve's result; it is not a separate record (the operator's turn above: the solvers
    produce "exactly the data structures needed to produce indices of which pages [...] are 'safe' or 'available'
    for reading or writing at any particular moment in execution").
  - integration-finishing-change-request.md, "2. A node's replayed commands in the completion-time order"; "3. A
    weight on each task".
  - torch-capture-change-request.md, "Source for item 2's execution-time estimates (2026-11-09)": "Replaces: the
    absence of a producer for t(v), and the consequent constant `HOST_LAG`."
- **Replaces:** compilation-composition-change-request.md, "6.", items 1–3: "`MetalProgramPrepare` calls
  `mesh_graph_dependences` and then `mesh_graph_synchronization`, which reconstructs the same dependence edges
  internally"; "`MetalRecorderProgram` receives call handles and reconstructs its ordering inputs from buffer
  footprints". `MetalProgramPrepare` is 320 lines; `metal_recording.m` and `tools/e2b_replay.m` call none of
  `mesh_graph_schedule`, `mesh_graph_list_schedule`, `mesh_graph_completion_times` or `mesh_graph_order`.
- **Reference:** `schedule_emit/schedule.py`, 164 lines; the solve is about 35.
- **Estimate:** 100–250 lines.

### 5. Storage decisions are made by the compile side and reach the runtime as data

- **Clauses:** disjoint-pages-change-request.md, "D.3 Claude's proposal", which that request names as the unit of
  work and which holds over C3, leaving out D.3.1, which plan-to-bridge-change-request.md supersedes;
  plan-to-bridge-change-request.md, "R2. Every page is placed at compile time and named by the page table";
  compilation-composition-change-request.md, "4.": "Readiness determines firing under the compiled order" and
  "Replay and host/bridge execution use the emitted decisions; they do not independently choose replacements for
  them."
- **Replaces:** lifetime and footprint reconstruction when a program is prepared or loaded.
- **Reference:** `schedule_emit/schedule.py`, `argument_buffers`, about 30 lines, by Bhattacharyya, Sriram and Lee
  (1997) and Lam (1988), both already cited by these requests.
- **Estimate:** 50–150 lines.

### 6. Lowering is a separate function from the scheduled program to each platform's compiler input

- **Clauses:**
  - compilation-composition-change-request.md, "Correction: generated target source is the final emission":
    "generated MSL or native target source is the final source emission of the mesh lowering pipeline, passed to the
    target compiler. Subsequent mesh transformations do not infer semantics by parsing or matching that output."
  - "7. Preserve the properties while compiling away the abstract machinery": "Emit fixed target/API operations with
    bound operands, in the scheduled order" (as amended 2026-11-09, "Withdrawn, 2026-11-09").
- **Replaces:** in the uncommitted work at metal-microbench 7483572e, the recovery of each call site's address by
  running `dsymutil` and an `lldb` script over the compiled library (`metal_lower.mm`, `NativeBuild`).
- **Reference:** `schedule_emit/lower_inductor.py`, 103 lines. It imports nothing from scheduling; what one compiler
  needs is derived inside it.
- **Estimate:** 300–600 lines.

### 7. The runtime binds addresses and calls; it waits only for a value a function reads

- **Clauses:** compilation-composition-change-request.md, "7.", Replaces: "the residual position/task/guard/loop-stack
  interpreter in `MetalProgramExecute`/`MetalNativeStep`, whole-prefix predicate drains without a reaching-writer
  justification in `MetalLoopTest`, and repeated static command-kind, cut, pass and binding discovery inside
  `MetalRowEncode`/`MetalReplayRun`. Compiling an outer caller which still invokes those generic decoders does not
  complete their replacement."; collective-goals-2026-09-14.md: "the *runtime* is necessarily extremely ismple,
  ebcause any artificially imposed control flow puts more cache reads and loads in between 'work is ready on peer'
  and 'work has arrived to this meshnode from peer and is being used'."
- **Also:** plan-to-bridge-change-request.md R4, "Behaviour at run time", statements 1, 2 and 4:
  "A function that needs its operand whole starts when every page of that operand is present, and not before. It
  waits for nothing else."; "Each task starts when its own page is present. It does not wait for another page of the
  same operand."; "The bridge receives that order and nothing about either kind."
- **Replaces:** `metal_recording.m` from `MetalReplayRun` to the end of the file: about 2,480 lines and 91
  definitions. Its public interface, 30 functions in `metal_recording.h`, falls to load, bind and call.
- **Reference:** `schedule_emit/runtime.py`, 75 lines. It examines no graph.
- **Estimate:** 150–300 lines.
- **Undecided:** whether the capture half of `metal_recording.m` (about its first 2,090
  lines) remains once the provider publishes the program. No clause read for this memorandum names it.

### 8. One archive format, owned by one module in both directions; loading solves nothing

- **Clauses:** aot-compilation-change-request.md, "A8. The IR operand": "The compilation must carry one IR from
  partitioning to serialization, as an operand of its own."
- **Replaces:** about 19 save, load and framing functions across `metal_ir.mm`, `metal_lower.mm`, `metal_recording.m`,
  `metal_native.mm` and `metal_neural.mm`; the artifact format was revised from version 18 to 22 on 2026-11-09.
- **Reference:** `schedule_emit/archive.py`, 116 lines.
- **Estimate:** 200–400 lines.

### 9. Callers use the smaller interface

- **Clauses:** compilation-composition-change-request.md, "5.": "`tools/e2b_replay.m` separately orchestrates prepared
  work. Compile the required ordering once and consume it through the shared native program; delete displaced
  orchestration."; plan-to-bridge-change-request.md, "R7. Nothing outside the bridge knows the transport".
- **Replaces:** orchestration in `tools/e2b_replay.m` (1,718 lines), `tools/native_bench.m` (542) and
  `rdma/mesh-metal.m` (355).
- **Reference:** `schedule_emit/driver.py`, 21 lines: it only composes the stages.
- **Estimate:** 300–500 lines.

### 10. The synchronization signature is printed from the scheduled program

- **Clauses:** synchronization-signature-change-request.md, "1. The signature: the compiler's own statement of every
  synchronization"; "3. The monitor: outside the program, only observing".
- **Replaces:** provenance kept by the runtime for the signature's sake, and the monitor's cases for the control flow
  item 1 removes.
- **Estimate:** 100–200 lines.

### 11. Evidence

- **Clauses:** compilation-composition-change-request.md, "Complete replacement workgroup": "E2B must complete prefill
  and decode on all five nodes"; "Measure decode throughput after prefill and transport overhead/bandwidth as
  `docs/measurement.md` requires"; integration-finishing-change-request.md, "1. Every mesh machine's bridge runs".
- **Before timing:** a comparison of structure against compilation-composition-change-request.md requirement 7:
  "Graph transformations may fuse vertices, remove edges made internal to fused work, and replace storage
  representations. The contract preserves the required observations and ordering, not a literal source graph or one
  runtime action per vertex." The scheduled program is compared with the exported one for what that sentence
  requires: every required observation and ordering is kept, and each function, send, receive, copy or wait the
  scheduled program has beyond the exported one is one a clause or the solver justifies.
- **Also:** plan-to-bridge-change-request.md R4, "Behaviour at run time": "No quoted source says
  how a run is shown to have done 1 to 3". The record of a run that shows them is outstanding and belongs here.
- **Also:** `examples/pl.py`, design/parallel-layers-example.md: a language model whose blocks add
  attention and feed-forward outputs computed from one normalized input, as a program a run is made of.
- **Estimate:** about 100 lines for the structural comparison; none for measurement.

## Why this order

- Item 1 removes scope, so it is first.
- Items 2–5 supply the compile-time results that item 7's runtime consumes.
- Item 6 precedes item 7, because the runtime calls what lowering returned.
- Items 8–11 consume the scheduled program. Of 165 commits in the two repositories on 2026-11-09, about 28 went to
  signature observation and 12 to save and load, built around the runtime that item 7 replaces, and about 10 to
  scheduling and native execution.

Summed, the estimates are about 1,800 to 3,300 lines written or revised, and at least 4,500 removed by items 4, 7
and 9.

## For whoever is assigned an item

Do the item as its clause states it and nothing else. Add no guard, wait, copy, check, setting or interface the
clause and the program's dependencies do not require. Name things by what they are: a function, a value, a call.
