# Change request: compose the existing requests as one compilation

2026-11-08. The work is the nine requests named below, composed through their representations and existing modules.
Their clauses identify work; the external sources define the methods and the roles in the composition. The algorithms
already selected by those clauses retain their cited definitions. The table uses their corrections and withdrawals.
Code locations below were read at mesh 9c9f8f3c.

## The turns

The operator, 2026-11-08, excerpts in order:

> except the user repeatedly insists upon supporting all pytorch programs which would be cudagraph compileable, and keeps insisting that this is simpler and a smaller scope tahn trying to support only a subset of pytorch. why? answer this immediately, and how it relates to the topic of the centralization of 'decisions' in the runtime to a single compilation stage

> is this hard or easy? given a program a which already lowers all liner algebra within a's domain, is it hard or easy to also lower all linear algebra wtihin a's domain with a different program b?

> so. we need t close out this meta review. are all of the long list of requirements in the change request documents a very long list of many different requierments, or a short list of very simple requiements which cross-cut across multiple abstractinos, elaborated more and more detail for implementing agents who refused to do the plain meaning of the assigned work?

> figure out how to cite this high level review to computer science literature, algorithms, and data structures to make it templateable within the change reuqest document format; there is no reason why a vhagne request document can't say 'we are doing these k other change request documents s.t. these documents are used in these {citedroles}'. unless you can think of some really complicated reason why this abstract requirement isn't templateable in the change request scheme?

## Sources for the roles

### S1. Structured representations and reusable transformations

N. Vasilache et al., *Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor
Compiler Construction*, 2022, [§2.1](https://arxiv.org/html/2202.03293v1#S2.SS1):

> tiling and fusion transformations are both fully generic in the operations and data types they operate upon.

Sections 2.1 and 3 describe progressive lowering through structured tensor operations, tiled operations, vectors and
buffers, with transformations applied while their required structure remains available. Section 2.1 also identifies
the limits of that particular lowering path: its representation is not a claim to cover every computation by itself.

### S2. Complete conversion and composition of lowering rules

MLIR, [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/), “Modes of Conversion”:

> A full conversion legalizes all input operations, and is only successful if all operations are properly legalized to the given conversion target.

“Rewrite Pattern Specification” constructs a graph of conversions: a rule from `bar.add` to `baz.add` and another
from `baz.add` to `foo.add` supply the composed lowering. The data structures are operations, typed operands/results,
rewrite patterns and a conversion target. This supplies the distinction between an intermediate conversion and a
completed lowering.

### S3. One representation of computation, parallelization and communication

C. Unger et al., *Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and
Parallelization*, OSDI 2022, [§1, “Unified graph representation”](https://www.cs.cmu.edu/~zhihaoj2/papers/unity_osdi22.pdf).
Its parallel computation graph (PCG) represents computation, parallelism and data movement together; transformations
are graph substitutions. This is the role of a common representation, not a requirement to replace the selected mesh
algorithms with Unity's search algorithm.

### S4. Transformations across computation and communication

A. Jangda et al., *Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning
Workloads*, ASPLOS 2022, [CoCoNet, abstract](https://arxiv.org/abs/2105.05720v5):

> fusion or overlapping of communication and computation

The program represents both kinds of operation; semantics-preserving transformations operate on them and the compiler
generates their combined execution. These are the roles of the numerical and collective lowerings in one program.

### S5. Modules own decisions across processing stages

D. L. Parnas, *On the Criteria To Be Used in Decomposing Systems into Modules*, CACM 15(12), 1972,
[“The Criteria”](https://www.cs.lafayette.edu/~gexia/cs301/resources/parnas.html), DOI 10.1145/361598.361623:

> Every module in the second decomposition is characterized by its knowledge of a design decision which it hides from all others.

“Improvement in Circular Shift Module,” items 1–2, groups a data structure with its access/modification procedures,
and a routine with the generation of its calling sequence. Module boundaries therefore follow responsibility for the
representation or decision, including when several stages use it.

### S6. Static inputs become a residual program

N. D. Jones, C. K. Gomard and P. Sestoft, *Partial Evaluation and Automatic Program Generation*, 1993,
[§4.2.2, Definition 4.1, p. 71](https://web.archive.org/web/2021id_/https://www.itu.dk/people/sestoft/pebook/jonesgomardsestoft-a4.pdf):
`[[p]] [d1, d2] = [[r]] d2` for every remaining input `d2`. Specialization incorporates the static input `d1` into
residual program `r`. The subject program, static inputs and remaining inputs are distinct operands.

### S7. Compile assignment and order; execute on data availability

S. S. Bhattacharyya, S. Sriram and E. A. Lee, *Optimizing Synchronization in Multiprocessor DSP Systems*, IEEE TSP
45(6), 1997, [§1, preprint p. 3](https://ptolemy.berkeley.edu/papers/97/synchronization/synch.pdf), describing Lee and
Ha's self-timed strategy:

> retain the processor assignment and actor ordering specified by the fully static schedule

Actor firing depends on actual input availability. The data structures are the dataflow graph, processor assignments,
per-processor actor orders and the communication/synchronization graphs.

S1, S2, S4 and S5 were read on their linked pages on 2026-11-08; S3 and S7 from the linked PDFs. S6 was reread in the
previously downloaded author's book, §4.2.2; its extraction is also recorded in
[specialization-sources-2026-11-07.md](specialization-sources-2026-11-07.md). MLIR supplies the cited compiler methods;
the implementation sites are the existing mesh and metal-microbench modules below.

## The nine requests and their roles

The rows identify the work being composed. They are not nine independent compilers or an execution order for passes.

| Request and applicable clauses | Role in the compilation or its execution | Representation and method |
|---|---|---|
| [page-queue](page-queue-change-request.md), “What the collective decides” and “The functions,” with its supersession | Transport interface and responsibility boundary (S5) | Declared transfers become per-hop page lists; physical transport belongs to the bridge. |
| [disjoint-pages](disjoint-pages-change-request.md), C3 with later corrections | Storage analysis and realization (S1, S6) | Value instances, lifetimes and page assignments, including modulo variable expansion. |
| [plan-to-bridge](plan-to-bridge-change-request.md), R1–R8 as corrected | Task/consumer representation and transport lowering (S3, S4, S7) | Views, joins, per-page tasks, weighted dependence graphs and serialized transfer instructions. |
| [aot-compilation](aot-compilation-change-request.md), A1–A10 and the current appendices | Whole-program composition, configuration operands and serialization (S1–S3, S6) | Exported function graphs, the graph of calls between them, sharding, weights and A8's IR. |
| [problems-of-14b9a757](problems-of-14b9a757-change-request.md), current Changes and settled stops | Corrections to transformation semantics and resource lifetimes (S1, S6, S7) | Views, collective transfers, persistent requests and cyclic lifetimes; its specified algorithms repair the same representations. |
| [torch-capture](torch-capture-change-request.md), items 1–9 with replacements | Capture/specialization, storage realization and complete replay (S1, S2, S6) | Recorded commands and operands, masks/worklists, usage intervals and replayed programs. |
| [torch-source-lowering](torch-source-lowering-change-request.md), items 1–3 | Numerical and synchronization lowering (S1–S4, S7) | Typed functions lower to CPU/GPU work; graph dependences lower to Metal orderings. Item 4 remains withdrawn. |
| [integration-finishing](integration-finishing-change-request.md), items 1–7 | Composition of existing producers and consumers, and restoration of their owners (S5, S7) | Published scheduling keys, task weights, the shared masked graph walk and per-commit feedback; the configured bridges run the resulting program. |
| [synchronization-signature](synchronization-signature-change-request.md), items 1–6 | Derived description and observation of the emitted synchronization (S7) | The signature comes from lowered IR; its specified monitor observes execution of that artifact. |

## Requirements

### 1. Finish lowering the whole assigned domain

- **Source:** the operator's coverage requirement above; S1's reusable transformations and S2's full conversion.
- **Applies to:** aot-compilation A1/A10, source-lowering 1–3 and capture 1/8/9.
- **Replaces:** the split in `rdma/mesh_compile.py program()` where `replayed` excludes nodes whose declarations name
  transferred values; also its rejection of repeated calls to a compiled function in the replayed block.
- **Write:** reuse the source lowerings and their composable representations to complete the assigned program domain.
  Compile both each numerical function and the graph of calls between functions. Lowering covers every rank and its
  collectives, and each placed CPU or GPU function. The source-lowering and capture requests retain their different
  Torch and LiteRT mechanisms. An intermediate conversion is completed before execution; its unfinished operations
  do not become a second execution regime. A missing lowering within the assigned domain is unfinished work, not a
  new restriction on that domain.

### 2. Compose the requests on the same program's representations

- **Source:** S3's PCG and graph substitutions; S4's computation/communication transformations; S1's progressive
  representations. The numerical and graph algorithms are those selected in the requests' applicable clauses.
- **Applies to:** all nine rows of the table.
- **Write:** revise the existing `IR`/`mesh_function` path and its producers and consumers so that a value's sharding,
  consumer kind, accesses, lifetimes, transfers and dependences describe the same program through lowering. Numerical
  decomposition, fusion and communication transformations keep the information needed by subsequent scheduling,
  allocation and synchronization. The output of each analysis or lowering is the operand its consumers actually use.
  Changes to the graph are reflected in those dependent results before emission. The signature describes that
  emitted program. Each row's work is completed in this composition, including its specified deletion or replacement.

### 3. Put each representation and decision in its existing owner

- **Source:** S5's information hiding and ownership of data-structure procedures and calling sequences; the operator's
  centralization requirement above.
- **Applies to:** aot-compilation A10, plan-to-bridge R7, source-lowering 3 and integration-finishing 2–5.
- **Write:**
  - `rdma/mesh_graph.py` and `rdma/mesh-graph.{h,c}` supply the shared graph operations through caller operands.
  - `rdma/mesh_compile.py` and `rdma/mesh-collective.c` compose the program and its plan through those operations.
  - `metal_recording.m` consumes the compiled order and lowers it to its backend's execution primitives.
  - `rdma/mesh-flow.c` and the bridge retain physical transport objects and device binding.
  - Revise callers to use those owners and remove the displaced copies of their decisions. The second scheduling
    solve, copied graph walks, runtime placement of program values and alternative execution paths are the existing
    requests' named replacements, not independent facilities to maintain beside their replacements.

### 4. Execute the compiled decisions with the specified runtime semantics

- **Source:** S6's specialization equation and S7's self-timed execution. The sources quoted by capture and
  synchronization-signature specify the concrete backend and observation operations.
- **Applies to:** aot-compilation A2/A3/A5–A10, plan-to-bridge R2–R8, capture, integration-finishing and
  synchronization-signature.
- **Write:** placement and costs arrive as operands; compilation realizes the graph's storage, transfers, ordering
  and synchronization. Runtime consumes the resulting programs and the remaining inputs, including actual arrival
  and completion state and permitted indexes within compiled shapes. Readiness determines firing under the compiled
  order. The bridge binds physical transport as R7 specifies. Replay and host/bridge execution use the emitted
  decisions; they do not independently choose replacements for them. The required signature and its external
  observation cover this execution. Performance and numerical comparison retain the operator's requirements in
  metal-microbench `docs/measurement.md`.

## Implementation correction, 2026-11-08

The operator, continuing the implementation assignment, verbatim:

> " but CPU execution and several cross-owner fixes remain open." this is a false premise subagents rpeeatedly try to construct and propagate; the apple silicon backend *is* a cpu and we are more htan capable of compiling a 'kernel' which contains the lowest attainable overhead program which will launch all cpu-sided, all ane-sided, and all mps-sided activity for the entire compute graph. we are not required to treat 'host python gil' as the 'cpu' and the 'launcehd commands' as the 'device' like in pytorch for cuda!

> deduplicate code taht's bloating by becoming duplicated; centralize and unenclose behaviors that are being implemented twice in twod ifferent closures spurring from the same producer or root function body. nothing we need as a requirement in our code actually involves having hundreds of functions or tens of thousands of lines of code.

> this is trivially effected through the change request documentary format and claude agents

### 5. Consolidate repeated implementations at their shared owner

- **Source:** S5, “The Criteria” and “Improvement in Circular Shift Module,” items 1–2: a module owns a decision,
  its representation and the generation of its calling sequence. S6, Definition 4.1: static operands specialize
  one subject program into its residual program. S1 supplies reusable transformations over structured operands;
  S2 requires completing the conversion. The operator's corrections above state the scope and required deletion.
- **Applies to:** requirements 1–4 here; aot-compilation A1/A10; source-lowering 1–3; capture 8/9;
  integration-finishing 2–5; synchronization-signature 1–6.
- **Roles and representations:** numerical functions and their calls are compilation inputs; placements, value
  instances and readiness are operands. CPU, ANE and MPS are resources of the Apple Silicon target. The compiled
  native program launches and coordinates the graph. Python capture/preparation is not its CPU execution model.
  The signature is derived from that program, through the graph module's representation and printers.
- **Replaces:** duplicated closures or traversals derived from the same producer, duplicated representations and
  scheduling/lowering decisions, and retained execution bodies displaced by the native program. In the working
  implementation based on mesh f0f96962 and metal-microbench 623d6674, the concrete sites include:
  - `mesh_compile.py _outputs` and `_natives`: both traverse the same call graph for every value-copy residue and
    remap the same captured results. `program.make` also rebuilds the same bound-input list. Specialization owns
    native operands, result bindings and their retained storage together; their consumers use its products.
  - `metal_recording.m MetalReplayEncoding` and `MetalReplaySelfTimed`: both construct, encode and commit commands
    from the same replay producer. `tools/e2b_replay.m` separately orchestrates prepared work. Compile the required
    ordering once and consume it through the shared native program; delete displaced orchestration. The later
    synchronization-signature request explicitly retracts the earlier proposal's per-pass commit/queue wait.
  - `mesh-collective.c plan_extents`: consumers needing the same strided-view traversal use this traversal through
    its owning module. They do not implement their own enumeration. Signature producers/consumers likewise use
    `mesh-graph`'s schema and printers; a copied Python structure or second scheduling solve is displaced.
- **Write:** centralize each repeated behavior in its existing owner, make closed-over inputs explicit operands
  where consumers need to share it, reconnect all callers, and remove the duplicate bodies and their dead helpers.
  Sharing a wrapper while retaining two implementations does not perform this replacement. Preserve the assigned
  numerical domain, collectives, page placement and lifetime semantics. Complete producer-to-consumer connections
  in the same revision: page groups and sent counters, native result bindings, and signature identities must agree
  at both ends. File assignments coordinate edits; they do not create new execution boundaries or owners.
  Report deleted/replaced bodies and the direct build results separately from runtime or performance evidence.

## Correction: complete the IR handoffs first

The operator, continuing the same assignment, verbatim:

> i want all of this dramatically simplified in refactoring as i know a priori that there will be immense duplication of algorithms and methods by implementers that chose to cheerily ignore the existing code and suppose tehy wer ebeing given totally new problems never touched before in that repository

> "Because we centralized helper implementations without completing the IR handoffs. Consumers still reconstruct facts their producers computed" to make myself clear: tehj point of the citation exercises was to make it really ovious that we complete the ir handoffs first and we don't try to 'write helpers' for interfaces we haven't implemetned yet.

> the change request documents exist to make this mistake hard to repeatedly run into by accident, basically?

> well figure out how to get like 8 claude subagents to do this concurrently instead of implementing it yourself. use the documentary standards of the repo to avoid getting lost in the weeds for days rewriting code one function at a time yourself

### 6. Carry compilation results through their consumers

- **Source:** S1's transformations preserve the structured operands required by subsequent lowerings; S3 represents
  computation and communication in the same graph; S5 assigns the representation and its calling sequence to the
  same owner; S7 executes the compiled assignment and order. The specific graph algorithms remain those cited by
  aot-compilation A4/A6–A10 and source-lowering item 3. This clause applies those existing methods, rather than
  proposing another algorithm or framework.
- **Applies to:** requirements 1–5 here; aot-compilation A1/A4/A6–A10; plan-to-bridge R2–R8;
  source-lowering 1–3; capture 3/6/8/9; integration-finishing 2–5; synchronization-signature 1–6.
- **Replaces:** these concrete incomplete handoffs in the current implementation, in priority order:
  1. `MetalProgramPrepare` calls `mesh_graph_dependences` and then `mesh_graph_synchronization`, which reconstructs
     the same dependence edges internally. The first edge result must be the operand of synchronization reduction.
  2. `mesh_plan` computes the calls' dependency relation and synchronization, but `MetalRecorderProgram` receives
     call handles and reconstructs its ordering inputs from buffer footprints. The plan signature is loaded and
     composed only after native preparation. Preserve the plan's relevant task identities, dependence/order results
     and transformation provenance as compilation operands of native preparation; the transport's existing runtime
     synchronization remains its consumer too. The diagnostic signature is derived from the resulting compilation,
     not substituted for an executable interface by merely printing it later.
  3. `MetalSchedule` reduces its graph to command levels and permutation. Replay preparation then constructs another
     dependence graph around the same commands and later passes. Preserve and remap the existing graph through
     command transformation, add the newly lowered vertices/edges, and supply it to ordering and emission.
  4. Typed/strided/affine accesses are present in the frontend and masked-pass analysis. Scheduling consumers instead
     repeatedly manufacture whole-buffer conflicts. Carry the actual access/effect representation into the consumers
     that need it; a newly introduced instruction adds its actual effects to that representation.
- **Write:** finish each producer → representation → consumer connection before extracting auxiliary helpers.
  Reuse the existing graph types and algorithms. Distinguish a genuinely new lower-level constraint from a repeated
  derivation of an existing one, and retain the identity mapping through lowering. Change the producer, interface,
  consumer and displaced reconstruction together. An unused new field, a new helper without a consumer, a second
  graph beside an ignored first graph, or diagnostic output alone does not finish a handoff. Preserve independent
  calls, interleaved computation/transport, exact copy lifetimes, placed CPU/MPS work, the full assigned numerical
  domain and the LiteRT path. Do not replace them with blanket order, waits, whole-buffer effects or another runtime.

The implementation is divided into these eight connected units. File ownership coordinates concurrent writes;
it does not divide the compiler's semantics. Session assignments and progress belong in `output_data`.

| Unit | Complete handoff | Source boundary and displaced work |
|---|---|---|
| 6A. Retained graph results | Existing dependence construction → reduction/order → retained results consumed by 6B/6C | `mesh-graph.c/h` graph construction/reduction/schedule APIs; remove rebuilding of a caller's already supplied edge relation. Interface work is complete only with its real callers. |
| 6B. Capture to replay | Captured command graph → transformed/reordered command graph → later-pass ordering and encoding | `metal_recording.m` capture ownership, `MetalSchedule`, `MetalOrder`, replay's later-pass graph/cuts; retain and remap edges instead of rescanning the same commands. |
| 6C. Plan to native program | Plan/call graph plus genuinely new placed effects → compiled native synchronization and execution | `metal_recording.h` program API and `metal_recording.m` native program preparation/execution; consume 6D/6E operands and 6A results, remove repeated analysis and independent reconstruction of supplied constraints. |
| 6D. Planning result publication | `plan_graph`/`plan_schedule` results → owned plan data → native and transport consumers | `mesh-collective.c/h`, `mesh.h`, `mesh.py`; preserve existing solves and make their relevant results and identities available to 6C/6E. Do not introduce a second solve or change the bridge protocol. |
| 6E. Torch program composition | Exported functions and call graph → plan inputs/results → native program input | `mesh_compile.py` `_Export`, call analysis, `program`, `_specialize`, `_native`, `_replays`; connect 6D to 6C, preserving repeated calls and actual placed operands. |
| 6F. Access/effect continuity | Source/typed view effects → recorded dispatch effects → scheduling footprints | `mesh_compile.py` numerical lowering/capture/transfer preparation, `operator_shaders.metal`, recorder access analysis/footprints; preserve existing typed and affine results and connect them to 6B/6C instead of discarding precision or reparsing available facts. |
| 6G. Compilation to signature | The retained compilation graph and emitted primitives → signature record → existing consumers | `mesh-graph.c` signature section, `mesh_graph.py`, recorder signature/emitter section, `mesh-sync-monitor.py`; retain provenance and identity through 6A–6F, removing reconstruction that duplicates preparation. No separate evaluator. |
| 6H. LiteRT and transport consumers | Prepared E2B program and transport operands → the same native program API and completion | `tools/e2b_replay.m`, `mesh-metal.m/h`; adopt 6C/6D interfaces and remove caller-side replicas of decisions now owned by the compiler. Preserve the existing operational measurement path. |

Every unit identifies the actual producer and consumer before changing their interface, publishes the exact contract
to the adjacent unit, and finishes the connection in this revision. Existing source is revised in place; temporary
compatibility paths are removed once their callers are converted. Direct builds establish buildability; runtime and
performance claims require the existing operational evidence. The supervisor coordinates these units and checks the
combined changes; these units are implementation assignments to Claude workers.

The operator clarifies the same lowering obligation, verbatim:

> its the job of a lowerer to fuse those vertices together, which of course means we need to be co-designing the IR with the consumeres who need to be getting ready to  consume type annoations and graph contents in these manners.
> " still rejects received operands" the idea that a consuming function can 'reject' operands is a claudelish decadence and doesn't come from any cited literature, does it?

For requirements 1, 2 and 6, retaining identities means retaining the correspondence through transformation;
it does not require one runtime operation per source vertex. S3's graph substitutions and S4's joint computation
and communication transformations supply the cited roles for fusion. Producers and consumers must agree on the
typed operands, effects, readiness and execution identities that let the lowerer combine vertices and discharge
internal dependencies, while preserving externally observable dependencies and value lifetimes. Endpoint types
guide that transformation; they do not freeze the source graph into separate launches or blanket waits.
S2's full-conversion criterion diagnoses unfinished conversion; it does not authorize an implementer to remove
received operands from the operator's assigned domain. In this revision, the received-operand refusal is repaired
by completing the representation and its consuming lowering, including later-pass identity remapping and the
native execution-step binding. Deleting the refusal without those consumers is not completion.

The operator, during the integrated review, verbatim:

> how much of the code in these sections is still inlining or duplicating 'responsibilities' from or for other aglorithms used or implemented elsehwere in the smae code? are w running into integratino errors where two interfaces 'don't line up well' because they are literally serializing and deserializing equivalent or related types over and over again in sprawling inlined code that, if modularized, would be mostly one-liners and compact invocations of existing functions and types?

> okay so all of these inlined algorihtm isseus should be handled by modulearizing our code a lot ore than it is right now. for example the literal graph algorithms header which was introduced ages ago in the repository, and is the only sensible place to put graph algorithms (and the only place ot import them as a header).

> commit first before 'finishing' anyof this stuff. commits aren't special breakpoints.

## Correction: generated target source is the final emission

The operator, during the structural review, verbatim:

> "do not carry enough provenance" say it more plainly: is typed data being retyped by a second implicity parser after being 'sorta serialized' for consumers from a previous IR?

> then this is a broad structural requirement which needs to be fixed by psasing annotation into lowering instead of treating lowering as a blackbox operation which acts upon 'strings'. that's absurd; worst possible way to write a compiler? find literally any reference compiler which deosn't use string latents and then code review to find out what went stinky and stupid and foul and stupid

> we don't want or need incremental pity-ports which pretend the current behavior of the data flow is adherent with a spec or system of requirements merely beacuse it runs; the 'lets turn things into text hten blindly parse them' is *simply wrong* as a lowering. it is *simply not correct*. why would compiled proglang output source be anything but the *final* lowering stage in any representation we are using if we were following the requirements in the change request documents? where does this come from?

- **Applies to:** items 1–6, especially 2, 3, 5 and 6F; source-lowering 1–3; capture 3/8/9;
  synchronization-signature 1/6. S1 and S2 already assign structured operations and typed operands/results to lowering.
- **Reference implementation:** MLIR's
  [Toy-to-Affine lowering](https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp)
  consumes operation objects, their result types and operand values, constructs lower-level operations with a rewriter,
  and replaces the source operations. MLIR's [Linalg representation](https://mlir.llvm.org/docs/Dialects/Linalg/)
  carries indexing maps, iterator kinds, shaped operands and computation regions through transformations.
  [BufferizableOpInterface](https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize) supplies memory
  read/write and alias relations to analysis and rewriting. These are concrete methods for the roles S1/S2 assign.
- **Replaces:** recorder transformations whose semantic operands are shader strings: scalar/quantization
  specialization, operand promotion, epilogue/map/row fusion, reduction/packed-weight rewriting, coherent-store
  insertion and masked-pass construction. `MetalEntry`, `MetalRowFootprint` and `MetalMaskedPass` are instances
  of this boundary failure; replacing their parsers alone does not complete the requirement.
- **Write:** the producing Torch and LiteRT compiler interfaces supply the operation, binding, type, shape,
  indexing, effect, alias and transfer/generation facts required by their consumers. Lowerings transform those
  representations, preserving or updating the corresponding identities, accesses and dependencies when they fuse,
  split, specialize or rebind operations. Consumers query the resulting annotations and relations. A plan edge
  retains the value/effect/transfer relation that makes it necessary until the consuming lowering has represented
  that obligation in the target execution.
- **Emission boundary:** generated MSL or native target source is the final source emission of the mesh lowering
  pipeline, passed to the target compiler. Subsequent mesh transformations do not infer semantics by parsing or
  matching that output. Source spelling, generated identifier names and pretty-printing are not optimization
  preconditions. Retaining tokens for some consumers while the other passes continue rewriting emitted strings
  leaves this requirement incomplete.
- **Replacement and evidence:** retain the numerical, fusion, communication and synchronization behavior required
  by the cited requests while replacing the bodies that implement it through text. Finish the connected producer,
  transformation and consumer before claiming completion. A successful build or workload does not establish that
  these representation and ownership requirements were followed. Review the actual dataflow, then use the existing
  numerical and performance measurements to evaluate its execution.

### Performance correction: compile the transformations, not their textual symptoms

The operator, 2026-11-09, verbatim:

> " Recovering measured performance justified restoring useful behavior;" put even 10 minutes of analysis into why this stringly typed edit was being used as a performance strategy, and how the performance gains accomplished by stringly typed reads and writes can be had by *nay other means*, once you're willing to like actually compile shit instead of chickening out?

- **Applies to:** the preceding correction and items 1–6. Retaining the performance mechanisms and replacing
  their representation are the same implementation assignment.
- **Sources and roles:**
  - Dawn/Tint's [Metal shader translation](https://github.com/google/dawn/blob/main/src/dawn/native/metal/ShaderModuleMTL.mm)
    obtains a program, converts it to an IR module and invokes its MSL writer. The
    [writer](https://github.com/google/dawn/blob/main/src/tint/lang/msl/writer/writer.cc), `Generate`, transforms the
    module with `Raise` before `Print`. This supplies an actual producer/transform/emission boundary, including
    for the compiler family whose emitted identifiers the recorder currently matches. These are reference source
    interfaces, not a claim that the installed LiteRT binary exports its internal IR through its public C API.
  - MLIR's [elementwise fusion implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp)
    composes operand indexing maps and remaps operation operands/results. Its
    [bufferization](https://mlir.llvm.org/docs/Bufferization/) uses tensor use-def, alias and equivalence relations
    to determine storage. These supply fusion and intermediate-storage elimination, rather than rediscovery of
    those relations from buffer names or shader text.
  - MLIR's [SCCP pass](https://mlir.llvm.org/docs/Passes/#-sccp), based on Wegman and Zadeck (1991), propagates
    known constants through IR. S6 supplies the specialization equation. Static operation, dtype, shape, stride
    and configuration operands become the residual program's instructions/constants; they are not repeatedly
    interpreted by generic shaders merely because their representation fits in a metadata buffer.
  - MLIR's [workgroup reduction lowering](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Transforms/AllReduceLowering.cpp)
    constructs subgroup reductions and workgroup exchanges from reduction operations. Apple's
    [MLX reduction](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal),
    [packed arithmetic](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h) and
    [system-scope access](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal)
    provide concrete target mechanisms. S4 supplies composition of computation and communication. Their roles
    do not authorize copying unrelated execution policies or narrowing the assigned domain.
- **Replaces:** the preceding recorder bodies, `MetalNativeProjection`/`MetalNativeArgmax`'s source-based
  identification and extraction, and `_pointwise`/`_reduced`/`_contract`'s premature commitment to generic
  dispatches where the static program still requires specialization, fusion and parallel scheduling.
- **Write:** preserve the numerical operations and their permitted precision/rounding, express maps and epilogues
  as operation regions, retain values across their uses, and lower reductions and packed data to the selected
  target operations. Preserve each actual effect and observable result while removing unnecessary materialization.
  Compile producer publication and consumer acquisition from the same range/page dependencies used by transport
  scheduling. Retain the existing tuned numerical bodies where they implement these operations; supply their
  parameters and bindings from typed operands before emission. Operation equivalence and applicability follow
  the structured operands and effects, not an entry-point name, generated variable name or operator spelling.
  Carry static facts as compiler operands and keep genuinely varying values as runtime inputs.
- **Performance obligation:** complete the execution mapping as well as operation coverage. Fusion removes
  launches and sometimes memory traffic; promotion removes materialization; subgroup lowering removes shared
  exchanges; publication fusion removes separate visibility/publication work. They are distinct transformations.
  Fusion must account for repeated producer evaluation, live registers and synchronization scope. Typing alone,
  compiling a generic interpreter of metadata, or replaying unchanged per-operation kernels does not accomplish
  these transformations. The same final kernel, bindings and execution schedule can be emitted from structured
  operations; source matching adds no execution capability. The existing measurement standard remains applicable.

## Reusable form

The same change-request form can name several existing requests in one requirement:

- **Applies to:** `<request files and exact current clauses>`.
- **Source:** `<external author/source, section and exact method defining the roles>`.
- **Roles and representations:** `<which clauses define inputs, transformations, analyses, outputs or execution;
  the named graphs, types, tables and schedules they share>`.
- **Replaces:** `<existing split, duplicated decision or incomplete producer/consumer connection>`.
- **Write:** implement those clauses as `<the cited composition>`, with `<producer>` supplying `<representation>`
  to `<consumer>` through `<responsible existing module>`; revise the callers and remove the displaced implementation
  as the named clauses require. State the coverage and property that must hold through the complete composition.

The source establishes the method; the request references identify its application. An author of a compiler method
does not need to have named these repositories or their change-request filenames.
