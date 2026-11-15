# Complete compiler composition replacement — work order, 14 November 2026

This dated order applies the operator's whole-flow correction to the existing
[composition request](../../design/compilation-composition-change-request.md),
especially its numerical lowerer boundary, items 2/3/5/6/7, S8, and complete
replacement workgroup. The assigned domain and numerical mechanisms remain those
of AOT A1–A10 and source-lowering 1/2/3/5/6. The literal reference is
`/Users/mdot/schedule-emit` at `f41c64a`: its driver composes transformations of
explicit operands, its lowerer calls the original compiler, and its consumers use
the returned results.

## Operator correction, 15 November 2026

> then we halt all work on anything related to any consumers until weh ave made the correct data structure at the producer side. then we only correct direct functional neighors of the producer in our callgraph, and so on.

> there was a very simple assignment and we even have a very simple reference codebase we can index against

This instruction sets the current work sequence. All eight implementation turns
were interrupted, their queues cleared, and their current source preserved in
git. Only worker104 resumes for the capture producer: `export.export`, its
immediate result construction, and the producer's data definition. The reference
is `schedule-emit/schedule_emit/export.py`'s `export`/`read` and
`schedule_emit/ir.py`'s `Exported`, at `f41c64a`. They return captured graph,
signature, state and result data before the numerical lowerer is called.

Codex reviews the concrete producer result against those source boundaries and
Mesh's assigned domain. Work then advances to that producer's direct functional
neighbors in the call graph, consuming the established result. The assignments
below retain their scope as history; their concurrent consumer work is stopped.

The operator asked why duplicated and inlined responsibilities were being
discovered and patched individually, then instructed: “what strategy have you not
been using that was cleraly required by the change rqeuest documents in
composition? use that strategy.” This workgroup replaces complete
producer/result/consumer connections. Its units include every alternative
implementation and lost-result consumer identified in the accounts below.

## Reference-based review, 15 November 2026

Operator:

> why are these secitons of the optimization and IR transformatino data incomplete or basically shit instead of satisfying the properties tehy were assigned? review the other program, schedule-emit or whatever it was called to figure out review standards to set this right in concrete and tangible terms

This source review reads `schedule-emit` at `f41c64a` and Mesh at `027ab655`.
The reference's tracked `schedule_emit` source matches that commit. Its existing
modified emitted archives were left untouched. The observations below concern
source structure; this review supplies no numerical or performance measurement.
All eight Claude workers are held during this review.

The assigned properties come from [composition](../../design/compilation-composition-change-request.md)
items 2/3/6/7 and its numerical-lowerer boundary, and
[AOT](../../design/aot-compilation-change-request.md) A1/A5/A6/A8/A9/A10.
The reference demonstrates how responsibilities compose. Mesh's full numerical
domain, retained analysis results, native execution and saved compilation remain
the assigned scope; the reference's smaller domain and threaded runtime serve
their stated demonstration roles.

| Property reviewed | Concrete reference mechanism | Evidence in an implementation review |
| --- | --- | --- |
| A producer returns the result required by its next stage. | `export.export/read` supplies the graph, signature roles, state and output TreeSpec; `driver.compile_module` passes returned values directly through placement, devices and schedule. | Identify the original producing symbol, its returned operand, and the immediate consumer's actual reads. Account for the complete result convention, including state, mutation, aliases and effects where the assigned source contains them. |
| A transformation completes its graph change. | `placement.insert_send_recv` creates the channel's send/receive and replaces every remote reader's operand. `devices.on_devices` inserts the copy, redirects the send and updates the receive prototype. | Show the input relation and the resulting relation, including all affected uses and annotations. The output graph implements the specified transformation before another stage consumes it. |
| One algorithm owns each solved decision. | `schedule.heft_placed` returns orders; `assemble` consumes those orders using original Node references. `argument_buffers` returns `inputs/feedforward`; `runtime.Program` reads those values. | Trace the solver's actual result into the next consumer. For Mesh, retain the original analysis/witness/allocation result as A8 and composition 6/7 require, alongside its compiled realization. Changed operands and their new analyses have explicit source correspondence. |
| An adapter satisfies the original consumer's complete contract. | `lower_inductor.adapt` supplies received shape/stride/dtype/device, returns sent values for Inductor's output-stride contract, and `lower_recv_strided` marks the original buffer in Inductor's lifetime set. | Name the consumer function and its actual operand convention, then identify the original operation that establishes each required property. A field's presence establishes retention; the consumer's use establishes the handoff. |
| Numerical compilation belongs to the original compiler. | `lower_inductor.lower` gives each scheduled GraphModule to `torch.compile(backend="inductor")`. | Follow the scheduled body into the cited compiler entry and its complete returned result. Mesh's added access, binding, source and reusable-emission operands come from their original producers and reach their named consumers. |
| Saved and executable forms consume the same decisions. | `archive.save/load` uses `torch.export` for the scheduled graphs and stores the associated inputs, channels, feedforward and result metadata; `Program` consumes those fields. | Identify the saved sections/references and the loaded operand read by execution. Mesh additionally preserves the source and derived results required by A8/A9 and composition 7. |

A concrete reference example is a result on processor `p` read by two operations
on processor `q`. `insert_send_recv` gives both readers the same receive Node and
channel. `on_devices` gives that receive its actual destination-device prototype
and puts the copy in the graph. `schedule` includes the send-to-receive dependence
and `assemble` maps the readers' operands through `copied`. The lowerer compiles
that returned graph. The review follows those actual objects and uses; a record
containing a channel number by itself would establish only one part of the work.

### Findings at this source checkpoint

- `export.Captured.captured` now retains the original capture result, but its
  slice of `fw_graph_outs` supplies only public prototypes to the next interpreted
  call. Input mutation, returned-input identity and the original alias/autograd
  conventions remain unestablished there. The original AOT forward epilogues own
  these semantics and expect their own output conventions. Commits `1296425f`
  and `027ab655` establish data wiring and explicit context ownership, **not a
  completed capture transformation**. This is the immediate unfinished boundary.
- The subsequent `export.capture` still invokes the displaced constructor,
  unpacks the old four-part `bound` record and constructs the displaced
  `Exported` interface. Its `held/roots/chain/written` tables reconstruct roles
  and bindings around those old fields. It remains halted under the producer-first
  sequence. Its ordinary outer operations also need explicit destinations when
  that direct consumer is assigned.
- `lower_inductor.TEMPLATE_DECLARED/TEMPLATE_RANGES` are `Unretained` values;
  `_template` puts the latter into `Kernel.ranges`. Original template declaration
  and access producers have yet to return their results through this interface.
  The markers identify unfinished coverage of the assigned domain.
- `torch-inductor-aot-lowered-result.patch` retains original planning owners,
  but `LoweredAOTI.emission`, `_emission_context`, `_planned_state` and
  `restore_planned` resume through a closure and selected copies of ambient or
  mutable state. They leave the emitter's complete explicit operand contract
  unfinished. A successful patch application or build establishes neither
  reusable emission nor saved ownership.

The existing change-request reusable form supplies the work order: **producer
and operands; mechanism and source; complete result and immediate consumer;
composition with the other requests in their cited roles**. Codex review records
the actual source symbols and follows one complete transformation through that
boundary. AST, import, patch and build observations retain their narrower meaning.
The user-ordered sequence advances outward when the local semantic handoff is
established. The subsequent instruction below sets the complete module units for
this correction.

## Complete affected-module revision, 15 November 2026

Operator:

> x omits y z wtill ws, and l depends on k. all of these are a manifest of specific defects or revisions that must be made all at once. write them out explicitly into the repo as documentation then address them all (alongside all other errors in the same module) all at once before further work

The following defects form one correction workgroup. The linked module accounts
enumerate the affected functions and remaining same-module defects before their
implementing workers resume. These are assignments for complete module revisions;
individual field or method edits remain ordinary intermediate commits.

| Defect | Required revision | Module unit and review account |
| --- | --- | --- |
| C1. `Captured.captured` returns a public-output slice while mutation, identity, alias, autograd, token and subclass conventions remain incomplete at the next call. | Establish the complete capture result and its original upstream public-call convention, preserving all of those operands and properties through the immediate use. | Capture: `export.py`, its result definitions and original capture-result patches; [frontend account](frontends/whole-composition-ownership-20261114.md). |
| C2. `capture` consumes the displaced constructor, bound-record and Exported interfaces. | Replace this module's entire old producer/consumer composition with the established original result and explicit source correspondence. | Same complete capture unit. |
| C3. `capture` reconstructs roles and bindings through holder tables and old projections; ordinary outer operations have incomplete destinations. `calls` retains a node-class rejection list. | Account for every source operation/result through the cited capture and graph transformation owners, with the complete assigned domain and explicit resulting uses. | Same complete capture unit; the function-by-function account includes obsolete helpers, comments and all remaining same-module assumptions. |
| P1. CPU template declarations and ranges/accesses are represented by `Unretained`, including `Kernel.ranges`. | Return the complete original template/epilogue declaration, access, workspace and source results from their producing compiler stages into the same complete result used for other bodies. | Compiler production: `lower_inductor.py` and its directly relevant original-result patches; [compiler account](compiler/whole-lowering-ownership-20261114.md). |
| P2. Reusable emission depends on a closure and selected copies/restoration of ambient and mutable state. | Expose the original emission stage's explicit operands and result; consume the retained planning result through that interface for each supplied configuration. | Same complete compiler-production unit, including `torch-inductor-aot-lowered-result.patch`. |
| P3. The other result constructors and compiler adaptations in that module require a complete coverage account. | Review every serving function and direct producer patch for omitted results, recovered types/roles, duplicated compiler algorithms and incomplete original calling conventions, then address the recorded defects in this same revision. | Same complete compiler-production unit. |

Codex owns the reference comparison, the complete interface specification and the
review of the resulting module revisions. Existing Claude workers implement those
documented units. The runtime, archive, placement and native-consumer work stays
held while these producer units are completed. Original compiler mechanisms and
the existing change-request requirements supply the implementation; the workgroup
record identifies defects and their disposition rather than adding another
parser, gate or compiler design.

### Current native source corrects a historical allegation

At Metal `b275b8c4`, `Synchronization.Before` retains operation IDs and calls the
operation `Conflict` path. `Hazards` reaches `Decided`, which adds results for
previously unanswered pairs; `Answer` reads `mesh_graph_relation_witness` and
retains known answers, including absence of conflict. The witness query reads the
existing relation. Source locations: `metal_lower.mm:1204`, `metal_ir.mm:831`,
`:866`, `:903`, `:933`, and Mesh `rdma/mesh-graph.c:1502` at `027ab655`.
This is positive evidence for that specific retained-result handoff. Its complete
access coverage, transformations, archive and execution remain separate work.

## Evidence and source scope

The compiler/frontend source checkpoint is Mesh `8cbc2ab9`, including the saved
graph owner in `4e1a2e85`. Compiler evidence is committed at `2f4a662a`, frontend
evidence at `06c50521`. Native source is Metal `4f592b61`, with matching build and
symbol evidence at `7700acc8`. Those checks establish checkpoints; the connected
compiler path remains unfinished.

- [Compiler ownership](compiler/whole-lowering-ownership-20261114.md) and its JSON
  snapshot account for the Python compiler modules, installed upstream stages,
  and dependency patches.
- [Frontend ownership](frontends/whole-composition-ownership-20261114.md) accounts
  for capture, placement, source/call analysis, copies, archive, binding,
  invocation, costs and their RDMA consumers.
- [Native ownership](executor/native-flow-supervisor-account.md) accounts for
  native construction, mappings, protocol, regions, replay, provider, final APIs,
  saved results and the graph/planner/bridge dependency set. Codex's account
  supersedes the Claude mechanical inventory's incorrect mapping-discard claim.

The earlier barrier-consumer re-solve finding is historical. The current source
trace above records the operation-based retained-witness handoff that replaced it.

## Connected assignments

| Unit | Producer and original mechanism | Result and consumers | Displaced responsibility | Implementing ownership |
|---|---|---|---|---|
| Original compiler result | Original Torch compile/AOT stages, Scheduler, MPS scoped backend result and CPU KernelGroup/CppKernelProxy; composition 2/6 and its numerical boundary | Returned body/control, AOT interface transforms, selected calls, source correspondence, layout/access/workspace and ABI operands feed ordinary analysis/allocation and final target emission | Factory capture list, premature compiled-callable output, positional reconstruction of AOT bindings, flattened entry authority, repeated numerical code generation | Compiler 68 and 66, with native and archive consumers |
| Numerical methods | Cited QDWH, LAPACK, Padé, Philox and fragment formulations enter that same original compiler; source-lowering 5/6 | Complete method bodies and selected native calls preserve precision, conditions, state, workspace, INFO and results through the shared consumers | Remaining Region/Generic scalar/tensor compiler, its record replay, duplicated LAPACK prototype/workspace responsibility | Compiler 67 with 66; frontend copy callers and native selected-call owner |
| Access, mapping and work results | Actual upstream typed index/domain/access result and canonical graph/ISL, mapping, obligation and metric producers; composition 3/6, AOT A5/A6/A10 | Original correlation, witnesses, configuration, footprints, page origins, supplied costs and known/unknown work feed dependencies, readiness, barriers, lifetimes, native bindings, observations and archive | Python index classifiers and whole-binding substitutions, unchanged-candidate mapping recomputation, member-barrier re-solving, repeated obligation scans, opaque-work-as-zero | Native 131 paired with compiler 68/66 |
| Copies, bindings and saved owners | Original declared transfers, source/destination views and canonical native copy operation; AOT A5/A8/A9/A10 | Every staging, landing, result and copy-only caller consumes the same explicit operands; source/declaration query and produced compiler results remain attached to the saved compilation | `Target.composite` geometry-to-meta-tensor-to-record-to-Builder route, omitted declaration/query owners and unretained lowered/control results | Frontend 104 paired with compiler 66/67 and native 131 |
| Region, order and replay results | Original region forest and graph postorder; actual order/cut/part/command and lifetime producers; composition 3/4/6/7 | Construction, lifetime widening, replay partition, repeat lookup, conditional completion and generated command entry consume retained structure and cut kinds | Recovered ancestry from span lengths, repeated enclosing-depth walks, repeated movable-set construction, switch/arm rediscovery and dispatch-to-part scans | Native supervisor assigns one complete region/order consumer unit, coordinating shared declarations with 131 |
| Typed LiteRT producer | Actual whole LiteRT/MLDrift compiler graph and selected method owners; composition's complete replacement and source-lowering 1 | Whole typed body/layout/effect/workspace/source results enter the same analysis, allocation, composition and native/archive consumers | Per-node model reparsing and the parallel broad handwritten numerical fallback/lowering pipeline | Native supervisor assigns a distinct provider unit; compiler/native result boundary is shared |

The independently authored source implementation of each behavior has one owner.
Call sites use that implementation by name or consume its returned data. Compiler
inlining of a shared definition realizes that same implementation. Calling a
canonical algorithm on a changed graph or different physical address domain is a
different computation; its result remains attached to that operand. Distinct
first/later programs specialize the same retained numerical result through the
original emitter.

The original source and native result interfaces determine the necessary data
projection. Existing draft classes and signatures are implementation artifacts,
not constraints on the replacement. The operator's 15 November sequence above
establishes the producer result first and advances through direct functional
neighbors. Shared-file edits have explicit ownership.

The deliberately malformed legacy references stay in their original locations.
The accounts distinguish these references from live competing implementations.
Existing capture/invocation contracts and saved native relocation consumers remain
connected to their original result owners.

## Execution and review

Codex supervisors own structural judgments and compare each completed connection
with the literal reference and cited request. Claude Opus 5.5 workers implement
the connected units. Current complete briefs replace the earlier local-edit
briefs; process handles and subsequent observations are recorded by each owner.

Commits preserve ordinary working checkpoints. Source/ABI/build checks establish
their actual checked properties. Once the paired compiler and native source are
coherent, existing sanctioned calling paths provide observations of the assigned
program-domain properties. Completion covers numerical/state/alias/control,
compiled dependency and lifetime, target, saved-program and distributed consumers
through the same data flow. The full thread goal remains active.
