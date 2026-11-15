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

The native account's barrier-consumer correction also applies here:
`Synchronization.Add` in `metal_lower.mm` receives `Accesses(p, operation)` and
partitions those unchanged views by internal memory space. `Before` concatenates
them and calls the vector `Conflict` overload, which solves their intersections
again and frees the witnesses. This consumer belongs in the same access/witness
handoff as the other native relation consumers.

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
