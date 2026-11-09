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

The initial handoff revision was divided into the eight connected units below. The subsequent source/performance
reviews found that their implementation did not finish numerical lowering. The
[complete replacement workgroup](#complete-replacement-workgroup) below supersedes this work division for the
remaining implementation; these identifiers remain here for existing source citations. File ownership coordinates
concurrent writes; it does not divide the compiler's semantics. Session assignments and progress belong in `output_data`.

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

### Complete replacement workgroup

The operator, 2026-11-09, verbatim:

> okay then lay out how

> the upstream compiler interface is going to be finished and all of the 'downstream' defective code removed all in one commit workgroup all at once, removal *by supremacy* of osme other different code that has all of the same feature coverage, and the same performance or better, so it's impossible to feign to revert the downstream slopcode yet again

- **Applies to:** items 1–6 and both corrections above, composing the nine requests in their existing cited roles.
  This specifies the replacement assignment. It does not record an implemented interface or measured replacement.
- **Sources and roles:** S1/S2 supply typed progressive lowering and completed conversion; S3/S4 supply joint
  transformation of computation and communication; S5 supplies ownership; S6/S7 supply specialization and execution.
  The fusion, bufferization, reduction and target methods in the performance correction remain the methods to use.
  Three upstream source boundaries make their application concrete:
  - ML Drift's [GpuModelBuilder](https://github.com/google-ai-edge/ml-drift/blob/main/ml_drift/common/gpu_model_builder.h)
    accepts tensor handles, operation attributes, reduction axes and weight/quantization descriptions.
    [GpuModel](https://github.com/google-ai-edge/ml-drift/blob/main/ml_drift/common/gpu_model.h) retains node inputs,
    outputs and tensor descriptors. Preserve these facts where they are produced. Its node names are optional
    debugging fields, not an operation contract. Its
    [GPUOperation](https://github.com/google-ai-edge/ml-drift/blob/main/ml_drift/common/task/gpu_operation.h)
    still contains `code_`, and its elementwise/linkable descriptors contain code strings. Wrapping those objects
    does not supply the missing structured body. Revise the producing lowering to construct typed operation
    regions and instructions instead of making those strings the interchange representation.
  - Dawn's [Metal translation](https://github.com/google/dawn/blob/main/src/dawn/native/metal/ShaderModuleMTL.mm)
    has the concrete `ProgramToLoweredIR` → `msl::writer::Generate` seam. Tint's
    [writer](https://github.com/google/dawn/blob/main/src/tint/lang/msl/writer/writer.cc) raises an IR module before
    printing MSL. Mesh transformations belong before final printing. Extend the target representation/emitter
    where required operations are missing; system-scope publication is not permission to splice the printed text.
  - LiteRT's [GPU accelerator build](https://github.com/google-ai-edge/LiteRT/blob/main/litert/runtime/accelerators/gpu/BUILD)
    defines the ML Drift WebGPU accelerator and its delegate dependencies, including a macOS dynamic-library target.
    These are source components to connect and build, not an exported whole-program mesh compiler API. No cited
    source already supplies that complete API; its construction is the integration work assigned here.
- **Replaces:** metal-microbench's production E2B linkage to the wheel's `liblitert-lm`, its generated-source
  interception as an optimization interface, the source transformations listed below, and their alternative
  preparation paths. The current Makefile builds SDK headers from local LiteRT/Dawn checkouts but links the
  packaged provider. The local Dawn header revision has not been established as the wheel's embedded compiler
  revision. Compiling headers or an adapter beside that binary does not connect its internal IR.

**Write the connected compiler interface.** Build a compatible source provider with the LiteRT runtime, delegate,
ML Drift and Dawn/Tint dependencies that actually execute these lowerings. Track the adapter, build description and
upstream changes in canonical source, not hand edits under `.build`. Resolve API/ABI compatibility and record the
built dependency revisions with the existing measurement record. The production E2B link must use those artifacts;
the packaged executable can remain an external numerical/performance reference. Building a host executable while
still loading a prebuilt GPU provider leaves the assignment unfinished.

Both frontends supply the existing program composition with the following connected operands. These are roles of
the representation, not a prescription to create five duplicate object models:

| Operand | Producer → consumer |
|---|---|
| Numerical operations | Torch FX/decompositions and LiteRT's typed model/operation construction → regions with typed values, shapes, layouts, indexing maps, reduction axes, quantization and numerical precision semantics → specialization and fusion. Retain constants separately from varying inputs. |
| Kernel instructions | Numerical lowering → owned typed shader modules/regions, including explicit operand and result bindings → target lowering and final emission. Construct the mesh path directly through IR builders; emitting generated WGSL or MSL and reparsing it as the missing interface does not finish this handoff. |
| Values and effects | The same value identities → views, aliases, ranges/pages, reads/writes and generation/lifetime facts → allocation, collective/transport planning, masking and publication. Bind model values to kernel operands to placed storage explicitly. |
| Program relations | Existing call, dependence and collective-plan results → remapped relations through fusion, splitting and placement → the shared graph algorithms and native program. Retain the value/effect/transfer cause of an edge. Update affected results when transformation changes their operands. |
| Executable artifact | Completed lowering → compiled CPU work, GPU pipelines, bindings, dispatches, transfers and synchronization plus their derived signature → the existing native executor and bridge. CPU execution is native Apple Silicon work; Python preparation is not the CPU target. |

The compiler owns these operands until finalization, including every module and constant used by later passes.
A callback borrowing one shader during `newLibraryWithSource` cannot supply that ownership or fuse its dispatch
with another one. The provider publishes the program and its bodies before execution; finalization occurs after
the program's calls and transport relations are available. An identity correspondence can be many-to-many after
fusion or splitting. A tensor's reduction or quantization meaning is supplied by its producer, not guessed from
an arbitrary Tint instruction sequence. Authored kernel sources may be compiler inputs and terminal emitter
templates; emitted source is never subsequently inspected to recover operands, effects or optimization decisions.
Tuned projection/argmax arithmetic is retained through typed parameters and computation regions, including their
map/epilogue operands. A template hole filled by extracting somebody else's generated source is not such a region.
The C++ compiler owns Tint objects; C/Objective-C/Python consumers use owned handles and typed operand/binding
records through the existing program boundary. They do not serialize an operation into a string to cross it.
Provider cache hits must preserve this handoff too: Dawn's translation runs inside a compilation-cache lookup,
so publishing IR only from the cache-miss callback is insufficient. Cache the required representation or completed
artifact with its compilation inputs; a cached shader string cannot substitute for the program and its relations.

**Displace the downstream bodies with their completed consumers.** This is a feature-preservation map for the
replacement, not a list of optional fast paths. Existing common graph algorithms stay in `mesh-graph`/`mesh_graph`.
Numerical transformations belong to the compiler; recording and execution consume their results.

| Displaced implementation | Replacement consuming typed operands; performance mechanism retained |
|---|---|
| `MetalSpecializeScalars`, `MetalLiteralLanes`, `MetalFoldQuantization`; Torch's interpretation of static operator/layout metadata | Constant propagation and specialization before target emission; remove repeated static loads, branches and index arithmetic while preserving numerical semantics. |
| `MetalPromoteOperand`, `MetalFuseEpilogue`, `MetalInlineMaps`, `MetalFuseRows` and their `MetalMapShape`/`MetalEntry`/`MetalRowFootprint`/`MetalRowKernel` recognizers | Compose computation regions and indexing maps, remap uses and eliminate dead intermediates. Preserve observable stores and dependencies; remove eligible launches and materialization without introducing repeated expensive producer work. |
| `MetalNativeProjection`/`MetalNativeArgmax` source recognition and extraction, `MetalPackedWeights`, `MetalProjectionReduction`, `MetalNormalizationReduction` | Select/construct the retained tuned algorithms from typed operations, layouts and reduction domains. Preserve packed arithmetic, parallel reductions and split work; propagate `MetalNativePlan` facts directly instead of rereading `MN_*` defines. |
| `MetalCoherentStores`, source-spliced publication/acquisition and `MetalMaskedPass` | Lower typed memory effects and page readiness to coherent access, publication, worklists and later-pass kernels. Preserve computation/transport overlap and fused publication. Carry any required target extension through the IR and emitter. |
| `MetalParsedSpace` and `MetalCommandAccesses` as semantic reconstruction from emitted source | Use the compiler's instruction call/use-def and access relations. Remove the parser and its dead helpers once both scheduling and signature consumers use those relations. |
| Replay-footprint reconstruction and the default/non-self-timed preparation bypass | Prepare one native program from retained plan relations and actual lowered effects. Emit its required synchronization regardless of the caller; remove the bypass and displaced orchestration. Preserve the existing shared executor. |

**Eight connected implementation responsibilities.** This replaces the earlier 6A–6H division for this revision.
The supervisor assigns Claude Opus 5.5 workers and records sessions, handoffs, corrective steering and results in
`output_data`. Adjacent workers co-design and consume the interface immediately. They do not independently invent
parallel schemas and ask an integration worker to reconcile them at the end. A shared file has one editing owner;
that owner incorporates the adjacent worker's required changes.

| Worker | Complete responsibility and adjacent handoff |
|---|---|
| 1. Upstream producer and build | Own the source provider, tracked upstream adapter/build changes and production library linkage. Supply the typed model/operation and kernel-construction interfaces to 2–4; preserve buffer binding, initialization and zero-copy imports for 8. Finish the actual provider's coverage, including generated-body producers, rather than exporting strings. |
| 2. Representation and ownership | Own the shared compiler interface and module/value lifetime. Connect 1 and 5 to existing program composition, with the binding/identity/effect correspondence consumed by 3, 4, 6 and 7. No unused metadata fields or parallel plan schema. |
| 3. Numerical transformations | Own specialization, region/index-map fusion, promotion and intermediate elimination, using 2's operands and producing 4's regions plus remapped effects for 6. Replace the source transformations' behavior across the assigned domain. |
| 4. Target lowering | Own typed instruction construction and target emission, including tuned projection/argmax, packed arithmetic, parallel reductions and the target operations needed by 6. Revise `metal_native` and its templates; emit final source only after transformations. |
| 5. Torch frontend and native CPU | Own `mesh_compile.py`'s numerical/program handoff and target calls. Feed the same compiler from typed FX/decompositions; finish native CPU and GPU lowering instead of prematurely committing to generic per-operation dispatches. Preserve the complete assigned Torch domain, repeated calls, mutations and structured results. |
| 6. Collective, transport and effects | Own the connection through `mesh-collective`, `mesh-graph` and `mesh-metal`: supplied dependencies, range/page effects, lifetime, readiness, masking and publication. Consume transformed relations and provide 4/7 the exact obligations; add genuinely new target constraints without rebuilding discarded equivalents. |
| 7. Native preparation and deletion | Own `metal_recording`'s conversion to an executor of the compiled artifact, the emitted synchronization/signature connection and removal of every displaced parser/rewrite/preparation body above. Consume 2/4/6 and supply the same native program API to 5/8. |
| 8. E2B integration and operational evidence | Own `tools/e2b_replay.m` and the existing diagnostic/measurement callers, including `tools/native_bench.m`, through the new interfaces. Work with 1/7 on the executable throughout; coordinate combined builds, numerical/performance records and final coverage/deletion review with all workers. |

**One replacement landing.** The finished change includes producer, representations, transformations, all callers,
the production build and the deletions together. Metal-microbench and mesh are separate Git repositories, so the
landing is one coordinated changeset with one integration commit in each affected repository; no Git commit spans
both. Working commits can record and synchronize development under the existing workflow. They are not separate
deployments of a partial compiler. Source synchronization uses commits/checkouts, never copied uncommitted source.

Before calling that landing complete, record in the existing integration document: the live producer → IR →
consumer paths; every displaced body's replacement and deletion; direct builds of all affected callers; numerical
divergences using the existing workload tools; and the prescribed performance evidence. E2B must complete prefill
and decode on all five nodes. Use the existing Torch/collective operational workloads for their corresponding
behavior. Measure decode throughput after prefill and transport overhead/bandwidth as `docs/measurement.md`
requires, relating observations to launches, materialization, arithmetic and synchronization. Retain the fastest
applicable recorded behavior and the required bandwidth/overhead tradeoffs; no new evaluator or search grid.
Workload evidence does not narrow the assigned domain to those examples. Any missing lowering or performance
mechanism is repaired in this replacement before completion, not hidden by retaining the old execution path.

The landed source has no legacy source-rewrite fallback, dual compiler mode or exception path back to the removed
implementation. Source text may be retained as a final diagnostic artifact; source matching is no longer an
execution dependency. Git history can of course be reverted, but restoring the old bodies would then be an explicit
architectural reversal: they would have no unique supported operation or demonstrated performance advantage left
to justify their restoration. This is the operator's removal by a complete replacement, not deletion of capability.

### Source bindings for compact effects and scalar lowering

These bind implementation methods to the existing complete-conversion and representation requirements; they do not
replace the assigned domain with the operations exercised by a diagnostic.

- **Compact effects — source:** William Pugh, [The Omega Test](https://www.cs.umd.edu/~pugh/papers/omega.pdf), the
  author's expanded 1992 version, §§2.1–2.3, establishes exact dependence analysis as integer feasibility.
  **Implementation source:** Sven Verdoolaege, [Integer Set Library manual](https://libisl.sourceforge.io/manual.pdf),
  Introduction, Creating New Sets and Relations, and Properties: construct affine constraints through the typed
  `isl_constraint` API and query `isl_basic_set_is_empty`, using exact arbitrary-precision arithmetic. ISL's algorithms
  differ from Omega's; linking ISL is not an implementation of Pugh's elimination steps.
  **Role:** decide whether two bounded affine byte-access sets intersect.
  The compiler supplies the access relations; `mesh-graph` owns their exact integer feasibility calculation and
  returns dependence causes. Scheduling, fusion and lifetime consumers use that relation. Enumerating tensor
  elements or replacing a strided set by its bounding interval is not this method. Exact arithmetic must survive
  elimination. The paper states exponential worst-case complexity; its practical results do not establish a general
  polynomial bound for these queries.
- **Software binary64 — source:** John R. Hauser, [Berkeley SoftFloat 3e, Library Interface](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html),
  §§4, 6–8; [FDLIBM](https://netlib.org/fdlibm/), the relevant function's implementation and dependencies for
  transcendental functions. **Role:** implement the scalar operations of a typed binary64 kernel when the selected
  target lacks native binary64 instructions. The compiler transforms typed instructions and preserves storage,
  conversions, special values and the required rounding semantics. SoftFloat supplies basic arithmetic, comparison
  and conversion algorithms; it is not a source for every transcendental function. Retain the upstream notices for
  derived source. Authored library source is a compiler input; generated target source remains terminal output.

### Nested communication regions: producer and consumer handoff

This is the next open implementation connection of items 1, 5 and 6, units 6D–6F, and workgroup
responsibilities 2, 5–7. It applies AOT A7/A10 and appendix item 6, disjoint-pages C3.3, and the complete
replacement's typed transfer/generation requirement. It does not change the bridge protocol or introduce a
second executor. The build27 numerical/control observations recorded in the workgroup's `output_data` do not
establish distributed communication inside a dynamic body.

**Code fact and displaced boundary.** `mesh_compile.partition` currently partitions the outer FX graph. A generic
control call gathers operands before `_interpreted` recursively lowers its body numerically. The body does not
pass through the existing partition, collective construction and plan consumers. `_Export` and `mesh_plan` carry
a flat repeated call block. Its range generation `{every, producer, ordinal}` maps a consumer invocation to a
producer invocation; native step words and `MetalCallExecution` carry that invocation through all its trips.
`Joins` publishes the call's sent result after its outermost control. These facts correctly describe a fixed
received operand reused by the body. They do not describe a fresh collective created in that body. Each bridge
plan execution visits its compiled hops once: putting optional or repeated transfers into that unchanged parent
plan can leave an untaken hop waiting or reuse an earlier phase.
A source region is not necessarily one communication plan: `mesh-flow.c`'s `lane_take` selects the earliest
delivered execution, and `link_send_drain` retains that execution's cursor when its next supplied/phase word is
not ready. If a parent's later hop requires a nested child's result, retaining the parent execution on that lane
can prevent the child's later delivery from running. The compiler therefore projects a region into maximal
straight-line communication segments at the required nested call/control boundaries, retaining each segment's
original region and ports. Existing graph ordering/scheduling owners determine those cuts and dependencies;
there is no default plan per numerical operation. The segment port relations carry pre-call values into the
child and its actual yielded values into the continuation, with the corresponding producing activation intact.

**Source roles.** MLIR's [SCF while, condition and yield](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-scfwhileop)
provide explicit region arguments, carried values and result correspondence, including zero trips. Their role
here is preserving structured value/control links, not selecting a communication schedule. MPI 4.1
[Persistent Communication Requests](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node83.htm), §4.9,
separates binding persistent operands from starting and completing repeated operations. Its role here is the
already compiled plan's activation and buffer lifetime, not an MPI wire protocol. The existing Lam/BSL and exact
dependence owners remain responsible for storage reuse, scheduling and synchronization; a dynamic trip count
must not be silently supplied as a static repetition count to their existing flat-call inputs.

OpenXLA's [`SpmdPartitioner::PreprocessCallSites`, `HandleWhile` and `HandleConditional`](https://github.com/openxla/xla/blob/main/xla/service/spmd/spmd_partitioner.cc)
align caller/callee shardings and retain partitioned control operands and results; the conditional predicate is
replicated across participating partitions. Their role is recursive partitioning with explicit boundary layouts,
using the same sharding and collective owners inside and outside control regions. This does not adopt another
transport scheduler.

**Retained operands and responsibility.**

| Producer and representation | Consumer and required preservation |
|---|---|
| Frontend: nested region identity from the owning function, original control node and arm/test/body role; its parent and actual control activation. | Shared compiler keeps the same region/call identity through partitioning and lowering. An untaken arm has no activation; a zero-trip loop has its initial test and selected exit values, but no body activation. |
| Frontend: each flattened argument/result's source value, formal/result ordinal, storage/alias identity and sharding/layout; initial, carried, additional/captured and yielded correspondence. | Partitioning uses the existing propagation, resharding and collective owners in every region. Lowering retains borrowed parent values in their producer's domain and preserves the actual child producer of a yielded or aliased result. |
| Compiler/plan: the persistent communicating region's actual activation and each range's producing activation, alongside its transfer, view and effects. | Native call entry supplies the monotonically increasing plan execution only when that region executes. Acquisition, publication and copy selection consume that exact identity. A loop trip number or observer feedback distance is not a substitute. Fixed invocation correspondence remains the existing static case. |
| Shared effect/lifetime owner: the complete uses of borrowed inputs, child outputs and aliases, including enclosing consumers and transport completion. | Preparation binds the resulting finite copies and last-use/completion relations. A child return alone does not free an escaping output. Later reentry reuses storage only after the retained last use, without a guessed trip bound or an unconditional added wait. |
| Plan owner: scoped call/control relations and solved allocations/schedules. | Native caller and signature consume the same results. The old global cyclic call order, affine cross-call thresholds and `horizon × calls-per-block` coordinate apply only where their premises hold; optional children must not be appended to that flat cycle. Save/load retains the resolved results and relocates live bindings without solving them again. |

A dynamic child's activation count need not be affine in its parent's invocation count. The existing wire
program fixes a child's buffer addresses by its own execution copy. A borrowed parent's numerical view still
selects its parent's copy. Direct transport aliasing therefore requires the retained address correspondence;
where it does not hold, the existing typed copy/reshard lowering supplies the required child-domain transfer
value and may fuse its producer's store when legal. This preserves source alias semantics while meeting the
compiled transport's actual storage constraint. A generation word alone cannot repair an incorrectly bound
buffer, and no runtime bridge descriptor remapping is introduced.

The frontend's concrete source records are `Region {id, parent, control, role, path, gm, ports, inputs, outputs,
links, children, ir}`, `Port {scope, node, index, value, storage}`, and `Link {source, target, kind}`. A region ID
names a lexical control site; reusing a GraphModule does not reuse that activation identity. `control` is the
original owning FX node; ordered children retain then/else or test/body roles. `index` is a flattened result
ordinal; `value` retains the original typed metadata, and `storage` its scoped alias identity. Links distinguish
borrowed, initial, carried, yielded and zero-trip exit values. The region's existing partition IR owns sharding
and layout; no second layout schema is inferred from names. Paths serve source provenance and annotations only.
One compiler Program retains the regions of an outer call. The plan owner projects its communication segments
from that ordered IR and these ports; it does not reparse FX or create separate recorder captures for each body.

**Connected edit boundary.** Frontend owns recursive region partition/instantiation and the original port links;
compiler owns shared typed region/generation operands and their transformed effects; executor owns scoped
`mesh_plan` consumption, native activation/copy/completion and the saved signature connection. A communicating
region uses persistent child plans through the existing protocol. Their preparation, activation and lifetime
consumers must land together; no unused region metadata, parent-generation replay of a child collective, or
numerical-only branch fallback constitutes this handoff. The concrete C interface follows agreement on these
producer facts and their existing consumers. Existing operational callers will exercise taken/untaken and
zero/repeated nested activations, borrowed inputs, carried outputs and communication as the connected code becomes
runnable; no separate evaluator is introduced.

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
