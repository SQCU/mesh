# Change request: compose the existing requests as one compilation

## The turns

The operator, 2026-11-08, excerpts in order:

> except the user repeatedly insists upon supporting all pytorch programs which would be cudagraph compileable, and keeps insisting that this is simpler and a smaller scope tahn trying to support only a subset of pytorch. why? answer this immediately, and how it relates to the topic of the centralization of 'decisions' in the runtime to a single compilation stage

> is this hard or easy? given a program a which already lowers all liner algebra within a's domain, is it hard or easy to also lower all linear algebra wtihin a's domain with a different program b?

> so. we need t close out this meta review. are all of the long list of requirements in the change request documents a very long list of many different requierments, or a short list of very simple requiements which cross-cut across multiple abstractinos, elaborated more and more detail for implementing agents who refused to do the plain meaning of the assigned work?

> figure out how to cite this high level review to computer science literature, algorithms, and data structures to make it templateable within the change reuqest document format; there is no reason why a vhagne request document can't say 'we are doing these k other change request documents s.t. these documents are used in these {citedroles}'. unless you can think of some really complicated reason why this abstract requirement isn't templateable in the change request scheme?

## The data flow

The operator, 2026-11-10, verbatim:

> for example we have a literal second repository iwth a priced and measured linecount which demonstrates the exact data flow mesh must have. what is that repo and are we using the exact same data flow yet? if we aren't, what's the cope? what's the excuse? who made up a limit saying we have to underperform our own examples to ourselves?

That repository is `~/schedule-emit`. Every requirement below is written against its data flow. Each stage is a
function from a value to a value:

| Stage | Function in `schedule_emit/` | Takes | Produces |
|---|---|---|---|
| export | `export.py` `export(module, args)` | a program and its example arguments | `ir.Exported`: the program's graph, its state, its inputs and its outputs |
| placement | `placement.py` `place(graph_module, placement)` | the exported graph; `placement`, an operand giving each node's processor | the same graph with each node's processor written on it and, for each value read on another processor, one `send` node and one `recv` node |
| devices | `devices.py` `on_devices(graph_module, devices)` | the placed graph; each processor's device | the placed graph with each node on its processor's device; a copy to another device is a node |
| schedule | `schedule.py` `schedule(module, cost)` and `argument_buffers(modules, output_processor)` | the placed graph; `cost`, an operand giving each node's cost | per processor, its nodes in scheduled order; per processor, its inputs and their buffers |
| the scheduled value | `ir.py` `Scheduled` | the results above | one value: `exported`, `modules`, `devices`, `output_processor`, `channels`, `inputs`, `feedforward` |
| lowering | `lower_inductor.py` `lower(scheduled)` | the scheduled value | per processor, a compiled function |
| runtime | `runtime.py` `Program(scheduled, functions)` | the scheduled value; the compiled functions | the program's results, each call |
| archive | `archive.py` `save(directory, scheduled)`, `load(directory)` | the scheduled value | files; the scheduled value again |

There is one schedule. Lowering, the runtime and the archive consume its result, including references to retained
inputs and derived results. The [14 November clarification](#clarification-2026-11-14-one-compilation-over-retained-results)
states how this composition extends the algorithms already used for transport to the other compilation scopes.

## Sources

### S2. Complete conversion and composition of lowering rules

MLIR, [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/), “Modes of Conversion”:

> A full conversion legalizes all input operations, and is only successful if all operations are properly legalized to the given conversion target.

### S3. One representation of computation, parallelization and communication

C. Unger et al., *Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and
Parallelization*, OSDI 2022, [§1, “Unified graph representation”](https://www.cs.cmu.edu/~zhihaoj2/papers/unity_osdi22.pdf).
Its parallel computation graph (PCG) represents computation, parallelism and data movement together; transformations
are graph substitutions.

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

The data structure is the fully static schedule: each function's processor assignment, and each processor's order of
its functions. The algorithm is the self-timed strategy: each processor calls its functions in that order, and a
function is called when the data it reads is available.

### S8. Analysis results become lower IR; abstract operations disappear

MLIR, [One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/#what-is-one-shot-bufferize):
whole-function SSA use-def analysis builds alias/equivalence sets and decides in-place storage before rewriting
tensor operations to buffer operations. Rewriting consumes those decisions.

MLIR's [affine lowering](https://mlir.llvm.org/docs/Passes/#-lower-affine) replaces affine operations with
loops over index ranges and arithmetic and removes unused maps/sets.

### S9. Algorithm and schedule compile into loops, storage and target code

J. Ragan-Kelley et al., *Decoupling Algorithms from Schedules for Easy Optimization of Image Processing Pipelines*,
SIGGRAPH 2012, [§§5.1–5.2 and Figure 5](https://commit.csail.mit.edu/papers/2012/ragan-kelley-siggraph12-halide.pdf).
Halide lowers the algorithm and schedule into imperative loops and storage, infers bounds, and lowers through
LLVM to target code. Bounds that depend on input sizes become generated expressions. Scheduling decisions are
realized in the program; the original functional pipeline need not be reconstructed at each invocation.

## The nine requests

The third column is read from the stages of `~/schedule-emit` against each clause's title. No request states it.

| Request and applicable clauses | What it writes | Stage that produces it → stage that reads it |
|---|---|---|
| [page-queue](page-queue-change-request.md), “What the collective decides” and “The functions,” with its supersession | Declared transfers become per-hop page lists. | schedule → the runtime's bridge, which alone performs the transport |
| [disjoint-pages](disjoint-pages-change-request.md), C3 with later corrections | Value instances, lifetimes and page assignments, including modulo variable expansion. | schedule → runtime |
| [plan-to-bridge](plan-to-bridge-change-request.md), R1–R8 as corrected | Views, joins, per-page tasks, weighted dependence graphs and serialized transfer instructions. | placement (R3, R8) → schedule (R5) → archive and the runtime's bridge (R1, R7) |
| [aot-compilation](aot-compilation-change-request.md), A1–A10 and the current appendices | Exported function graphs, the graph of calls between them, sharding, weights and A8's IR. | export (A1, A10), with the placement operand (A2) and the cost operand (A3) → schedule (A6, A7) → archive (A8, A9) |
| [problems-of-14b9a757](problems-of-14b9a757-change-request.md), items 1–12 with its “Sources for the stops” sections | Views, collective transfers, persistent requests and cyclic lifetimes; its specified algorithms correct the same graph and schedule. | placement → schedule → runtime |
| [torch-capture](torch-capture-change-request.md), items 1–9 with replacements | Recorded commands and operands, masks/worklists, usage intervals and the recorded programs every node calls. | export (item 1) → schedule (items 2, 6, 7) → lowering (item 8) → runtime (item 9) |
| [torch-source-lowering](torch-source-lowering-change-request.md), items 1–3 | Typed functions lower to CPU/GPU work; graph dependences lower to Metal orderings. | the scheduled value → lowering |
| [integration-finishing](integration-finishing-change-request.md), items 1–7 | Published scheduling keys, task weights, the shared masked graph walk and per-commit feedback; the configured bridges run the resulting program. | the cost operand (item 3) → schedule (item 2) → runtime (item 1) |
| [synchronization-signature](synchronization-signature-change-request.md), items 1–6 | The signature comes from lowered IR; its specified monitor observes execution of that artifact. | lowering → the signature; the monitor reads it outside the program |

## Requirements

### 1. Finish lowering the whole assigned domain

- **Source:** the operator's coverage requirement above; S2.
- **Applies to:** aot-compilation A1/A10, source-lowering 1–3 and capture 1/8/9.
- **The data structure** is the exported graph of each numerical function, and the graph of calls between functions
  (aot-compilation A1, A10).
- **The algorithm** is full conversion (S2): every operation of the input is converted to the target, and the
  conversion succeeds only when all of them are.
- **Composes:** export produces both graphs. Lowering takes the scheduled value and converts every function in it: for
  every rank, its numerical functions, its collectives (the `send` and `recv` nodes placement added), and each placed
  CPU or GPU function.
- **Replaces:** the split in `rdma/mesh_compile.py program()` where `replayed` excludes nodes whose declarations name
  transferred values; also its rejection of repeated calls to a compiled function in the `replayed` block.
- **Write:** Compile both each numerical function and the graph of calls between functions. Lowering covers every rank
  and its collectives, and each placed CPU or GPU function. An intermediate conversion is completed before the program
  runs. A missing lowering within the assigned domain is unfinished work, not a new restriction on that domain.

### 2. Compose the requests on the same program's representations

- **Source:** S3. The numerical and graph algorithms are those selected in the requests' applicable clauses.
- **Applies to:** all nine rows of the table.
- **The data structure** is one graph for computation and data movement together (S3's parallel computation graph):
  its nodes are function calls and, for each value read on another processor, a `send` and a `recv`; its edges are
  values, each with its sharding, consumer kind, accesses and lifetimes.
- **The algorithm** is graph substitution (S3): a transformation replaces a subgraph of that graph with another.
- **Composes:** export produces the graph. Placement adds the `send` and `recv` nodes to it. One schedule orders
  functions and transfers together and produces the scheduled value. Lowering, the runtime and the archive take the
  scheduled value. The signature is printed from what lowering emits.
- **Write:** A value's sharding,
  consumer kind, accesses, lifetimes, transfers and dependences describe the same program through lowering. Numerical
  decomposition, fusion and communication transformations keep the information needed by subsequent scheduling,
  allocation and synchronization. The output of each analysis or lowering is the operand its consumers actually use.
  Changes to the graph are reflected in those dependent results before emission. The signature describes that
  emitted program. Each row's work is completed in this composition, including its specified deletion or replacement.

### 3. Each graph function and each decision is computed in one place

- **Source:** the operator's centralization requirement above, and the operator's turn under requirement 6: "the
  literal graph algorithms header which was introduced ages ago in the repository, and is the only sensible place to
  put graph algorithms (and the only place ot import them as a header)".
- **Applies to:** aot-compilation A10, plan-to-bridge R7, source-lowering 3 and integration-finishing 2–5.
- **The data structure** is requirement 2's graph, passed by each caller as an operand.
- **The algorithms** are the graph algorithms those clauses name.
- **Composes:** placement and the one schedule call the graph functions; the schedule's result is in the scheduled
  value, which lowering and the runtime read.
- **Write:**
  - `rdma/mesh_graph.py` and `rdma/mesh-graph.{h,c}` define the shared graph functions, which take the caller's
    operands.
  - Callers call those functions, and their own copies are deleted. Deleted, as the existing requests name them: the
    second scheduling solve, copied graph walks, placement of program values at run time and alternative execution
    paths.

### 4. Execute the compiled decisions with the specified runtime semantics

- **Source:** S6's specialization equation and S7's self-timed execution. The sources quoted by capture and
  synchronization-signature specify the concrete backend and observation operations.
- **Applies to:** aot-compilation A2/A3/A5–A10, plan-to-bridge R2–R8, capture, integration-finishing and
  synchronization-signature.
- **The data structure** is the static schedule (S7): each function's processor assignment, which the placement
  operand gives, and each processor's order of its functions and transfers, which the one schedule computes. In S6's
  equation the placement and the costs are the static input `d1`, the scheduled value is the residual program `r`, and
  the remaining inputs `d2` are the arrival and completion state and the permitted indexes within compiled shapes.
- **The algorithm** is specialization (S6) at compilation and self-timed execution (S7) at run time.
- **Composes:** placement takes the placement operand and the schedule takes the cost operand; the schedule produces
  the scheduled value; the runtime and the bridge read it.
- **Write:** placement and costs arrive as operands; compilation realizes the graph's storage, transfers, ordering
  and synchronization. Runtime consumes the resulting programs and the remaining inputs, including actual arrival
  and completion state and permitted indexes within compiled shapes. A function is called, in the compiled order, when
  the data it reads is available. The bridge binds physical transport as R7 specifies. Every node's execution and the
  bridge's use the order and placement in the scheduled value; neither computes its own. The required signature and
  its external observation cover this execution. Performance and numerical comparison follow the operator's
  requirements in metal-microbench `docs/measurement.md`.

## Implementation correction, 2026-11-08

The operator, continuing the implementation assignment, verbatim:

> " but CPU execution and several cross-owner fixes remain open." this is a false premise subagents rpeeatedly try to construct and propagate; the apple silicon backend *is* a cpu and we are more htan capable of compiling a 'kernel' which contains the lowest attainable overhead program which will launch all cpu-sided, all ane-sided, and all mps-sided activity for the entire compute graph. we are not required to treat 'host python gil' as the 'cpu' and the 'launcehd commands' as the 'device' like in pytorch for cuda!

> deduplicate code taht's bloating by becoming duplicated; centralize and unenclose behaviors that are being implemented twice in twod ifferent closures spurring from the same producer or root function body. nothing we need as a requirement in our code actually involves having hundreds of functions or tens of thousands of lines of code.

> this is trivially effected through the change request documentary format and claude agents

### 5. One implementation of each repeated behavior

- **Source:** the operator's corrections above, which state the scope and the required deletion. S6, Definition 4.1:
  static operands specialize one subject program into its residual program.
- **Applies to:** requirements 1–4 here; aot-compilation A1/A10; source-lowering 1–3; capture 8/9;
  integration-finishing 2–5; synchronization-signature 1–6.
- **The data structure** is one compiled native program per node, which launches all CPU, ANE and MPS work of the
  graph (the operator's first turn above). Numerical functions and their calls are compilation inputs; placements,
  value instances and readiness are operands. CPU, ANE and MPS are resources of the Apple Silicon target. Python
  capture and preparation are not its CPU execution.
- **The algorithm** is specialization (S6): one subject program and its static operands give the residual program.
- **Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it. The signature
  is printed from that program by the graph module's printers.
- **Replaces:** duplicated closures or traversals derived from the same producer, duplicated representations and
  scheduling/lowering decisions, and execution bodies the native program replaces. The sites include:
  - `mesh_compile.py _outputs` and `_natives`: both traverse the same call graph for every value-copy residue and
    remap the same captured results. `program.make` also rebuilds the same bound-input list. Specialization computes
    native operands, result bindings and their storage once; their consumers read its result.
  - `metal_recording.m MetalReplayEncoding` and `MetalReplaySelfTimed`: both construct, encode and commit commands
    from the same recorded commands. `tools/e2b_replay.m` separately orchestrates prepared work. Compile the required
    ordering once and consume it through the shared native program; delete displaced orchestration. The later
    synchronization-signature request explicitly retracts the earlier proposal's per-pass commit/queue wait.
  - `mesh-collective.c plan_extents`: the strided-view traversal has one definition, and every consumer that needs
    it calls that. Signature producers and consumers likewise call `mesh-graph`'s schema and printers; a copied
    Python structure or second scheduling solve is deleted.
- **Write:** each repeated behavior has one definition; where consumers share it, its closed-over inputs are explicit
  operands; every caller calls it; the duplicate bodies and their dead helpers are deleted.
  Sharing a wrapper while two implementations remain does not perform this replacement. The emitted program preserves
  the assigned numerical domain, collectives, page placement and lifetime semantics. Producer and consumer change in
  the same revision: page groups and sent counters, native result bindings, and signature identities agree at both
  ends.

## Correction: "complete the ir handoffs first"

The operator, continuing the same assignment, verbatim:

> i want all of this dramatically simplified in refactoring as i know a priori that there will be immense duplication of algorithms and methods by implementers that chose to cheerily ignore the existing code and suppose tehy wer ebeing given totally new problems never touched before in that repository

> "Because we centralized helper implementations without completing the IR handoffs. Consumers still reconstruct facts their producers computed" to make myself clear: tehj point of the citation exercises was to make it really ovious that we complete the ir handoffs first and we don't try to 'write helpers' for interfaces we haven't implemetned yet.

> the change request documents exist to make this mistake hard to repeatedly run into by accident, basically?

> well figure out how to get like 8 claude subagents to do this concurrently instead of implementing it yourself. use the documentary standards of the repo to avoid getting lost in the weeds for days rewriting code one function at a time yourself

### 6. Carry compilation results through their consumers

- **Source:** S3 represents computation and communication in the same graph; S7 executes the compiled assignment and
  order. The specific graph algorithms remain those cited by aot-compilation A4/A6–A10 and source-lowering item 3.
- **Applies to:** requirements 1–5 here; aot-compilation A1/A4/A6–A10; plan-to-bridge R2–R8;
  source-lowering 1–3; capture 3/6/8/9; integration-finishing 2–5; synchronization-signature 1–6.
- **The data structure** is requirement 2's graph with its dependence edges, each carrying the access that causes it,
  and requirement 4's static schedule computed from that graph.
- **The algorithms** are those cited by aot-compilation A4/A6–A10 and source-lowering item 3.
- **Composes:** each item below names a function that produces a result and the function that takes that result as
  its operand.
- **Replaces:** these, in the present code:
  1. `MetalProgramPrepare` calls `mesh_graph_dependences` and then `mesh_graph_synchronization`, which reconstructs
     the same dependence edges internally. The first edge result must be the operand of synchronization reduction.
  2. `mesh_plan` computes the calls' dependency relation and synchronization, but `MetalRecorderProgram` receives
     call handles and reconstructs its ordering inputs from buffer footprints. The schedule's signature is loaded and
     composed only after native preparation. The schedule's task identities, its dependence and order results, and
     the transformation that produced each are compilation operands of native preparation; the transport's runtime
     synchronization reads the same results. The signature is printed from the resulting compilation.
  3. `MetalSchedule` reduces its graph to command levels and permutation. Preparation of the recorded commands then
     constructs another dependence graph around the same commands and later passes. The one graph is remapped through
     command transformation, gains the newly lowered vertices/edges, and is the operand of ordering and emission.
  4. Typed/strided/affine accesses are present in the frontend and masked-pass analysis. Scheduling consumers instead
     repeatedly manufacture whole-buffer conflicts. Carry the actual access/effect representation into the consumers
     that need it; a newly introduced instruction adds its actual effects to that representation.
- **Write:** finish each producer → representation → consumer connection before extracting auxiliary helpers.
  Distinguish a genuinely new lower-level constraint from a repeated
  derivation of an existing one, and retain the identity mapping through lowering. Change the producer, interface,
  consumer and displaced reconstruction together. An unused new field, a new helper without a consumer, a second
  graph beside an ignored first graph, or diagnostic output alone does not finish a connection. The emitted program
  preserves independent calls, interleaved computation/transport, exact copy lifetimes, placed CPU/MPS work, the full
  assigned numerical domain and the LiteRT path. A total order, waits, whole-buffer effects or another runtime do not
  replace them.

The operator clarifies the same lowering obligation, verbatim:

> its the job of a lowerer to fuse those vertices together, which of course means we need to be co-designing the IR with the consumeres who need to be getting ready to  consume type annoations and graph contents in these manners.
> " still rejects received operands" the idea that a consuming function can 'reject' operands is a claudelish decadence and doesn't come from any cited literature, does it?

The operator, during the integrated review, verbatim:

> how much of the code in these sections is still inlining or duplicating 'responsibilities' from or for other aglorithms used or implemented elsehwere in the smae code? are w running into integratino errors where two interfaces 'don't line up well' because they are literally serializing and deserializing equivalent or related types over and over again in sprawling inlined code that, if modularized, would be mostly one-liners and compact invocations of existing functions and types?

> okay so all of these inlined algorihtm isseus should be handled by modulearizing our code a lot ore than it is right now. for example the literal graph algorithms header which was introduced ages ago in the repository, and is the only sensible place to put graph algorithms (and the only place ot import them as a header).

> commit first before 'finishing' anyof this stuff. commits aren't special breakpoints.

## Correction: generated target source is the final emission

The operator, during the structural review, verbatim:

> "do not carry enough provenance" say it more plainly: is typed data being retyped by a second implicity parser after being 'sorta serialized' for consumers from a previous IR?

> then this is a broad structural requirement which needs to be fixed by psasing annotation into lowering instead of treating lowering as a blackbox operation which acts upon 'strings'. that's absurd; worst possible way to write a compiler? find literally any reference compiler which deosn't use string latents and then code review to find out what went stinky and stupid and foul and stupid

> we don't want or need incremental pity-ports which pretend the current behavior of the data flow is adherent with a spec or system of requirements merely beacuse it runs; the 'lets turn things into text hten blindly parse them' is *simply wrong* as a lowering. it is *simply not correct*. why would compiled proglang output source be anything but the *final* lowering stage in any representation we are using if we were following the requirements in the change request documents? where does this come from?

- **Applies to:** items 1–6, especially 2, 3 and 5; source-lowering 1–3; capture 3/8/9;
  synchronization-signature 1/6.
- **The data structure** is the operation as an object with typed operands and results. MLIR's
  [Linalg representation](https://mlir.llvm.org/docs/Dialects/Linalg/)
  carries indexing maps, iterator kinds, shaped operands and computation regions through transformations.
  [BufferizableOpInterface](https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize) supplies memory
  read/write and alias relations to analysis and rewriting.
- **The algorithm** is rewriting. MLIR's
  [Toy-to-Affine lowering](https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp)
  consumes operation objects, their result types and operand values, constructs lower-level operations with a rewriter,
  and replaces the source operations.
- **Composes:** the Torch and LiteRT compiler interfaces produce the operations and their annotations at export.
  Placement, the schedule and lowering take and return them. Lowering emits MSL or native target source, which the
  target compiler takes. No function of mesh takes that source.
- **Replaces:** recorder transformations whose semantic operands are shader strings: scalar/quantization
  specialization, operand promotion, epilogue/map/row fusion, reduction/packed-weight rewriting, coherent-store
  insertion and masked-pass construction. `MetalEntry`, `MetalRowFootprint` and `MetalMaskedPass` are among them;
  replacing their parsers alone does not complete the requirement.
- **Write:** the producing Torch and LiteRT compiler interfaces supply the operation, binding, type, shape,
  indexing, effect, alias and transfer/generation facts required by their consumers. Lowerings transform those
  representations, preserving or updating the corresponding identities, accesses and dependencies when they fuse,
  split, specialize or rebind operations. Consumers query the resulting annotations and relations. A dependence edge
  of the schedule carries the value/effect/transfer relation that makes it necessary until the lowering that reads it
  has represented that obligation in the target execution.
- **Emission boundary:** generated MSL or native target source is the final source emission of the mesh lowering
  pipeline, passed to the target compiler. Subsequent mesh transformations do not infer semantics by parsing or
  matching that output. Source spelling, generated identifier names and pretty-printing are not optimization
  preconditions. Retaining tokens for some consumers while the other passes continue rewriting emitted strings
  leaves this requirement incomplete.
- **Property preserved:** the emitted program has the numerical, fusion, communication and synchronization behavior
  required by the cited requests; the bodies that implement it through text are replaced. That the program builds
  and runs does not establish this requirement (the operator's third turn above).

### Performance correction: compile the transformations, not their textual symptoms

The operator, 2026-11-09, verbatim:

> " Recovering measured performance justified restoring useful behavior;" put even 10 minutes of analysis into why this stringly typed edit was being used as a performance strategy, and how the performance gains accomplished by stringly typed reads and writes can be had by *nay other means*, once you're willing to like actually compile shit instead of chickening out?

- **Applies to:** the preceding correction and items 1–6. The transformations that give the performance and the
  replacement of their representation are the same implementation assignment.
- **Data structures and algorithms:**
  - **Emission.** The data structure is Tint's IR module. Dawn/Tint's
    [Metal shader translation](https://github.com/google/dawn/blob/main/src/dawn/native/metal/ShaderModuleMTL.mm)
    obtains a program, converts it to an IR module and invokes its MSL writer. The
    [writer](https://github.com/google/dawn/blob/main/src/tint/lang/msl/writer/writer.cc), `Generate`, transforms the
    module with `Raise` before `Print`.
  - **Fusion and storage.** MLIR's
    [elementwise fusion implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp)
    composes operand indexing maps and remaps operation operands/results. Its
    [bufferization](https://mlir.llvm.org/docs/Bufferization/) uses tensor use-def, alias and equivalence relations
    to determine storage (S8). These supply fusion and intermediate-storage elimination, rather than rediscovery of
    those relations from buffer names or shader text.
  - **Constants.** MLIR's [SCCP pass](https://mlir.llvm.org/docs/Passes/#-sccp), based on Wegman and Zadeck (1991),
    propagates known constants through IR. S6 supplies the specialization equation. Static operation, dtype, shape,
    stride and configuration operands become the residual program's instructions/constants; generic shaders do not
    read them again at run time merely because their representation fits in a metadata buffer.
  - **Reductions.** MLIR's
    [workgroup reduction lowering](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Transforms/AllReduceLowering.cpp)
    constructs subgroup reductions and workgroup exchanges from reduction operations. Apple's
    [MLX reduction](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal),
    [packed arithmetic](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h) and
    [system-scope access](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal)
    provide concrete target mechanisms.
- **Composes:** each of these is a function from operations to operations inside lowering, which takes the scheduled
  value; `Print` comes after all of them.
- **Replaces:** the preceding recorder bodies, `MetalNativeProjection`/`MetalNativeArgmax`'s source-based
  identification and extraction, and `_pointwise`/`_reduced`/`_contract`'s premature commitment to generic
  dispatches where the static program still requires specialization, fusion and parallel scheduling.
- **Write:** preserve the numerical operations and their permitted precision/rounding, express maps and epilogues
  as operation regions, retain values across their uses, and lower reductions and packed data to the selected
  target operations. Preserve each actual effect and observable result while removing unnecessary materialization.
  Compile producer publication and consumer acquisition from the same range/page dependencies used by transport
  scheduling. Operation equivalence and applicability follow
  the structured operands and effects, not an entry-point name, generated variable name or operator spelling.
  Carry static facts as compiler operands and keep genuinely varying values as runtime inputs.
- **Performance obligation:** complete the execution mapping as well as operation coverage. Fusion removes
  launches and sometimes memory traffic; promotion removes materialization; subgroup lowering removes shared
  exchanges; publication fusion removes separate visibility/publication work. They are distinct transformations.
  Fusion must account for repeated producer evaluation, live registers and synchronization scope. Typing alone,
  compiling a generic shader that reads metadata at run time, or calling unchanged per-operation kernels does not
  accomplish these transformations. The same final kernel, bindings and execution schedule can be emitted from
  structured operations; source matching adds no execution capability. The measurement standard is metal-microbench
  `docs/measurement.md`.

### Complete replacement, all in one commit workgroup

The operator, 2026-11-09, verbatim:

> okay then lay out how

> the upstream compiler interface is going to be finished and all of the 'downstream' defective code removed all in one commit workgroup all at once, removal *by supremacy* of osme other different code that has all of the same feature coverage, and the same performance or better, so it's impossible to feign to revert the downstream slopcode yet again

- **Applies to:** items 1–6 and both corrections above.
- **The algorithms** are those of the performance correction: fusion, bufferization, constant propagation and
  reduction lowering.
- **The data structures** are in three upstream sources:
  - ML Drift's [GpuModelBuilder](https://github.com/google-ai-edge/ml-drift/blob/main/ml_drift/common/gpu_model_builder.h)
    accepts tensor handles, operation attributes, reduction axes and weight/quantization descriptions.
    [GpuModel](https://github.com/google-ai-edge/ml-drift/blob/main/ml_drift/common/gpu_model.h) retains node inputs,
    outputs and tensor descriptors. These facts are carried from where they are produced. Its node names are optional
    debugging fields, not an operation contract. Its
    [GPUOperation](https://github.com/google-ai-edge/ml-drift/blob/main/ml_drift/common/task/gpu_operation.h)
    still contains `code_`, and its elementwise/linkable descriptors contain code strings. Wrapping those objects
    does not supply the missing structured body. Revise the producing lowering to construct typed operation
    regions and instructions instead of making those strings the interchange representation.
  - Dawn's [Metal translation](https://github.com/google/dawn/blob/main/src/dawn/native/metal/ShaderModuleMTL.mm)
    calls `ProgramToLoweredIR` and then `msl::writer::Generate`. Tint's
    [writer](https://github.com/google/dawn/blob/main/src/tint/lang/msl/writer/writer.cc) raises an IR module before
    printing MSL. Mesh transformations belong before final printing. Extend the target representation/emitter
    where required operations are missing; system-scope publication is not permission to splice the printed text.
  - LiteRT's [GPU accelerator build](https://github.com/google-ai-edge/LiteRT/blob/main/litert/runtime/accelerators/gpu/BUILD)
    defines the ML Drift WebGPU accelerator and its delegate dependencies, including a macOS dynamic-library target.
    These are source components to connect and build. No cited source supplies a whole-program mesh compiler
    interface; writing it is the work assigned here.
- **Composes:** the table of operands below gives, for each operand, the function that produces it and the functions
  that take it.
- **Replaces:** metal-microbench's production E2B linkage to the wheel's `liblitert-lm`, its generated-source
  interception as an optimization interface, the source transformations listed below, and their alternative
  preparation paths.

The replacement is finished all in one commit workgroup: the compiler interface is finished and the downstream bodies
are removed together, by other code with the same feature coverage and the same performance or better (the operator's
turn above).

**Write the connected compiler interface.** Build from source the LiteRT runtime, delegate, ML Drift and Dawn/Tint
dependencies that actually execute these lowerings. Track the adapter, build description and upstream changes in
canonical source, not hand edits under `.build`. The production E2B link uses those built libraries. An executable
that still loads a prebuilt GPU library leaves the assignment unfinished.

Both frontends supply the following operands:

| Operand | Producer → consumer |
|---|---|
| Numerical operations | Torch FX/decompositions and LiteRT's typed model/operation construction → regions with typed values, shapes, layouts, indexing maps, reduction axes, quantization and numerical precision semantics → specialization and fusion. Constants are kept apart from varying inputs. |
| Kernel instructions | Numerical lowering → typed shader modules/regions, including explicit operand and result bindings → target lowering and final emission. Construct the mesh path directly through IR builders; emitting generated WGSL or MSL and reparsing it as the missing interface does not finish this connection. |
| Values and effects | The same value identities → views, aliases, ranges/pages, reads/writes and generation/lifetime facts → allocation, collective/transport scheduling, masking and publication. Bind model values to kernel operands to placed storage explicitly. |
| Program relations | Call, dependence and collective-schedule results → remapped relations through fusion, splitting and placement → the shared graph algorithms and native program. Each edge carries the value/effect/transfer that causes it. Update affected results when transformation changes their operands. |
| Executable artifact | Completed lowering → compiled CPU work, GPU pipelines, bindings, dispatches, transfers and synchronization plus their derived signature → the native executor and bridge. CPU execution is native Apple Silicon work; Python preparation is not the CPU target. |

The compiler keeps these operands until finalization, including every module and constant used by later passes.
A callback borrowing one shader during `newLibraryWithSource` cannot supply them or fuse its dispatch
with another one. The compiler publishes the program and its bodies before execution; finalization occurs after
the program's calls and transport relations are available. An identity correspondence can be many-to-many after
fusion or splitting. A tensor's reduction or quantization meaning is supplied by its producer, not guessed from
an arbitrary Tint instruction sequence. Authored kernel sources may be compiler inputs and terminal emitter
templates; emitted source is never subsequently inspected to recover operands, effects or optimization decisions.
Tuned projection/argmax arithmetic is carried through typed parameters and computation regions, including their
map/epilogue operands. A template hole filled by extracting somebody else's generated source is not such a region.
Tint objects are created and destroyed in the C++ compiler; C/Objective-C/Python consumers use handles and typed
operand/binding records through the program boundary. They do not serialize an operation into a string to cross it.
Cache hits carry the same operands: Dawn's translation runs inside a compilation-cache lookup,
so publishing IR only from the cache-miss callback is insufficient. Cache the required representation or completed
artifact with its compilation inputs; a cached shader string cannot substitute for the program and its relations.

**The downstream bodies and what replaces each.** Graph algorithms are defined in `mesh-graph`/`mesh_graph`.
Numerical transformations are computed by the compiler; recording and execution consume their results.

| Replaced implementation | Replacement, taking typed operands; the performance it keeps |
|---|---|
| `MetalSpecializeScalars`, `MetalLiteralLanes`, `MetalFoldQuantization`; Torch's reading of static operator/layout metadata at run time | Constant propagation and specialization before target emission; remove repeated static loads, branches and index arithmetic while preserving numerical semantics. |
| `MetalPromoteOperand`, `MetalFuseEpilogue`, `MetalInlineMaps`, `MetalFuseRows` and their `MetalMapShape`/`MetalEntry`/`MetalRowFootprint`/`MetalRowKernel` recognizers | Compose computation regions and indexing maps, remap uses and eliminate dead intermediates. Preserve observable stores and dependencies; remove eligible launches and materialization without introducing repeated expensive producer work. |
| `MetalNativeProjection`/`MetalNativeArgmax` source recognition and extraction, `MetalPackedWeights`, `MetalProjectionReduction`, `MetalNormalizationReduction` | Select/construct the tuned algorithms from typed operations, layouts and reduction domains. Preserve packed arithmetic, parallel reductions and split work; propagate `MetalNativePlan` facts directly instead of rereading `MN_*` defines. |
| `MetalCoherentStores`, source-spliced publication/acquisition and `MetalMaskedPass` | Lower typed memory effects and page readiness to coherent access, publication, worklists and later-pass kernels. Preserve computation/transport overlap and fused publication. Carry any required target extension through the IR and emitter. |
| `MetalParsedSpace` and `MetalCommandAccesses` as semantic reconstruction from emitted source | Use the compiler's instruction call/use-def and access relations. Remove the parser and its dead helpers once both scheduling and signature consumers use those relations. |
| Footprint reconstruction for the recorded commands, and the default (non-self-timed) preparation path | Emit the shared native program's required synchronization regardless of the caller; delete that path and the orchestration it replaces. |

After the replacement the source has no source-rewrite fallback, second compiler mode or path back to the removed
implementation, and source matching is not an execution dependency.

### Source bindings for compact effects and scalar lowering

- **Compact effects.** The data structure is a bounded affine set of byte accesses, constructed as affine constraints
  through the typed `isl_constraint` API (Sven Verdoolaege,
  [Integer Set Library manual](https://libisl.sourceforge.io/manual.pdf), Introduction, Creating New Sets and
  Relations, and Properties). The problem is exact dependence analysis as integer feasibility (William Pugh,
  [The Omega Test](https://www.cs.umd.edu/~pugh/papers/omega.pdf), the author's expanded 1992 version, §§2.1–2.3).
  The algorithm is ISL's `isl_basic_set_is_empty`, using exact arbitrary-precision arithmetic. ISL's algorithms
  differ from Omega's; linking ISL is not an implementation of Pugh's elimination steps. It decides whether two
  bounded affine byte-access sets intersect.
  **Composes:** the compiler produces the access relations; a function of `mesh-graph` computes their exact integer
  feasibility and returns dependence causes; scheduling, fusion and lifetime consumers take that relation.
  Enumerating tensor elements or replacing a strided set by its bounding interval is not this method. Exact
  arithmetic must survive elimination. The paper states exponential worst-case complexity; its practical results do
  not establish a general polynomial bound for these queries.
- **Software binary64.** The algorithms for basic arithmetic, comparison and conversion are those of John R. Hauser,
  [Berkeley SoftFloat 3e, Library Interface](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html),
  §§4, 6–8; for transcendental functions, [FDLIBM](https://netlib.org/fdlibm/), the relevant function's
  implementation and dependencies. SoftFloat is not a source for every transcendental function. They implement the
  scalar operations of a typed binary64 kernel when the selected target lacks native binary64 instructions.
  **Composes:** lowering takes the typed binary64 instructions and emits target instructions; it preserves storage,
  conversions, special values and the required rounding semantics. Derived source carries the upstream notices.
  Authored library source is a compiler input; generated target source remains terminal output.

### 7. Preserve the properties while compiling away the abstract machinery

The operator, 2026-11-09:

> " Execution will reuse the loop stack." does this mean that a runtime has a loop stack? is a runtime using the most compact most optimized program which has the same data flow dependency ordering and so on as the IR implies or is there an extra virtualization, sync, wait, guard, or runtime check implied by this representation? the performance ceiling of our final lowering is paramount in this scope of work...

> lets make sure that our documentation of change requests and so on correctly cites reference systems where abstract problems are solved, producing lowerable-IRs, which are themselves then optimizing-compiled into abstraction-agnostic-forms which have the same properties (by compilation) with or without an explicit representation of the types, metadata, graphs, etc that we know will still be true about the runtimes.

- **Source:** S8, S9, S6, S7, and S3's graph substitutions.
- **Applies to:** requirements 1–6, all nine composed requests, and the complete replacement above.
- **The data structure** is the scheduled value: typed numerical IR with its dependencies, placement, storage and
  communication annotations, and requirement 4's static schedule. Analysis solves the abstract problem
  once and supplies its results to optimizing consumers. Typed numerical IR, dependencies, placement, storage and
  communication annotations are ordinary values, available through retained references. Derived structures may be
  appended alongside their inputs in paged storage. Lowering realizes their decisions in instructions, addresses,
  target commands and synchronization. Removing repeated interpretation from execution does not require deleting
  the solved results or their page-table references, or moving them to a separate diagnostic representation.
- **The algorithm** is lowering of the algorithm and its schedule into an imperative program and its storage (S9),
  with storage decided by analysis before the rewriting that consumes the decision (S8), static operands specialized
  into the program (S6), and self-timed execution of what remains (S7).
- **Composes:** the schedule produces the scheduled value. Lowering takes it and emits the native program: the table
  below gives, per operand, what the emitted program preserves and which repeated work it removes. The runtime calls that
  program; the archive stores the scheduled value.
- **Replaces:** the code of `MetalProgramExecute`/`MetalNativeStep` that reads positions, tasks, guards and a loop
  stack at run time; `MetalLoopTest` and the whole-queue waits it makes, which are deleted and not replaced; and
  repeated static command-kind, cut, pass and binding discovery inside `MetalRowEncode`/`MetalReplayRun`. Compiling
  an outer caller which still invokes those generic decoders does not complete their replacement.
- **Write:** pass the solved schedule and effects into native target
  lowering. Emit fixed target/API operations with bound operands, in the scheduled order;
  compose and optimize CPU bodies with them, and fuse target-local work where legal. CPU bodies stay lowerable until
  composition; they are not reduced before it to opaque function pointers. The executor bodies this replaces are
  deleted with it.

| Compiler operand or solved result | Property preserved in the emitted program | Repeated work removed from execution |
|---|---|---|
| Typed numerical operations, shapes, views and alias/effect relations | Specified numerical semantics, permitted floating-point transformations, mutations, addressing and required lifetimes | Repeating type, shape or alias analysis; decoding known operation kinds |
| Partitioned computation, transfer dependencies and compiled order | Required producer/consumer ordering, publication and readiness; legal overlap and fused work | Reconstructing vertices, edges or partitions; repeating the scheduling solve |
| Realized storage, copies and completion dependencies | Correct value instance, non-overlapping live storage where required, reuse only after completion | Repeated lifetime analysis or parsing allocation metadata on each task |
| Selected ICB runs, bindings, copies/fills, barriers and API boundaries | The selected target commands with their required ordering and operands | Rediscovering command kinds, passes, cuts, bindings or callable targets |
| Lowered synchronization and the cause recorded for each | The required externally observable synchronization property of this executable | Reconstructing the compiler graph to recover synchronization causes already retained with the result |

Graph transformations may fuse vertices, internalize edges and derive storage representations. The transformed
graph retains references to its source relations and solved results; it need not execute one action per source
vertex. A consumer whose decisions have been specialized no longer has to interpret those relations. That does
not make the relations themselves disposable. Dropping an annotation or reference and reconstructing it later,
reparsing emitted source to recover it, or silently forgetting its property is an incomplete lowering.

Residual runtime operands are actual changing data: inputs, supported dynamic indexes, addresses bound
at preparation/load, arrival/completion and resource state. Their arithmetic and necessary API work
remain compiled operations. A remaining wait must follow from actual producer/consumer effects and the selected
target mechanism. Unrelated queued work does not justify a wait for the whole queue. Readiness is not permission to
repeat placement or dependency analysis.

Performance is an independent requirement of this replacement (the operator above: "the performance ceiling of our
final lowering is paramount"). Semantic equivalence does not excuse an extra run-time decoder, launch, transfer,
guard or wait introduced only to interpret already-solved compiler decisions. Removing the outermost decoder while the same
dispatch runs underneath it does not satisfy the requirement.

## Withdrawn, 2026-11-09

The operator, 2026-11-09, in order:

> do our external reuqirements for meshlib allow arbitrary control flow inside of programs or do they require exactly the opposite? Review this. we shall not ocnsider arbitrary python to be part of arbitrary torch; programmers who want to run arbitrary data dependent control flow between sections of linear algebra are to learn to write that sort of code on their own time and outside of our mandate and the library features we provide, which are specifically for torch like tensor apis w/ scheduling, transport, and function fusion as needed to support lowest attainable overhead scheduling and transport.

> let me preempt a list of common concerns:
> "what about mixture of experts"
> that is a *scatter* or a *sparse tensor operation*. it is not data dependent control flow; it is data dependent data flow aka data dependent sparsity aka literally more linear algebra of a trivially optimizable form.
> "what if i really want a graph break"
> we do not need graph breaks in our example lowerer.

> strike out anything and everything which contradicts the remarks reaffirmed (yet again) that we still aren't creating fake problems for ourselves wrt 'what if there's an arbitrary python program pretended to be part of a torch program and we need to play geometry dash for 12 hours inside of a graph for a larped reason gosh what are we gonna do now'.

The outer call graph and tensor shapes are known for AOT compilation. Conditions and iterations inside the
called numerical algorithms remain part of those algorithms, including the methods in source-lowering items 5/6.
They do not authorize an execution mechanism that repeatedly chooses placement, ordering or lowering rules.

## Clarification, 2026-11-14: one compilation over retained results

The operator, in order, verbatim:

> "Once dependencies have determined instruction order, the executable need not carry a graph object to maintain that order." otoh the graph object can be presented literally as a serialized format which has a header supplied list of functions to interact with it, which is readable from within the 'page table' scheme of accessible buffers as any other sort of value. in this sense we can 'preserve' properties of an IR or lowering by literally concatenating more derived data structures to the pages where we already wrote sections of IR, lowering, typing, optimizatino, whatever. if we think about this model seriously, the only thing we need to 'preserve' the results of an optimization or IR or lowering stage are to 1: make them efficient and compact in how they are data-structured 2: refusing to delete them or the references to their page tabling for no reason
> try to think about this from the user's persepctive here. the user doesn't want something that has tenso f thousands of lines of code. the user thinks that this project has simple, compact requirements. the user leaves ceratin scopes of this problem loosely defined because the right solutions are abstract, airy, simple, and relatively free.

> somewhat clearly an aot compilation can have branches inside of a called algorithm. but there are probably no needs to have branches inside of the 'thing we do to lower the results of optimizing compiling' into a runtime, and a runtime should probably be branchless-in-control-flow (it does one boring thing voer and over again, even if that boring thing might involve branching 'data flow' that involves applying pure functions to explicit operands over and over again)

> okay lets clean up undermining wording and 'reactivate' (derrida) what it is that we are working on with this project as we do so. we know, for example, that we already have wroking dependency solvers nad loop unrolling optimizers and optimizing allocators which were written and tested for the subproblem of *transport*. most of what we need to do for the 'full problem' is to simply modularize those algorithms and use them the same way for all of the other scopes that the optimizing compiler must handle: placing and scheduling function parameters, placing and scheduling function operands, placing and scheduling function execution, and i suppose some epilogue and prologue fusion to overlap execution of some kinds of transport and operations with each other within runtimes. most of this is extremely generic abstract programming over callgraphs; there is reference literature to consult mostly to orient us towards our requirements being very simple and easy (as we are only interested in handling aot cases involving exported serialiations of a language which already exists (pytorch) with known parsers (pytorch) and a known emitter (pytorch... and fxgraphs i guess).).

**Applies to:** items 2–7 here; AOT A1/A5/A6/A8–A10; source-lowering items 2/3/5/6; and their implementation
assignments across mesh and metal-microbench. This clarifies the existing work; it does not add a compiler framework.

**Sources and roles:** AOT A1 supplies PyTorch's capture and exported representation. AOT A5 and its appendix item 1
supply Lam's unrolling and modulo variable expansion, happens-before and chain-based allocation. AOT A6 supplies
the scheduling algorithms; A10 supplies call independence and graph analysis. S3 supplies graph substitutions for
computation and communication together; S6 supplies specialization; S8/S9 supply consumers of solved analysis and
schedule results. These citations identify reusable problems and methods, not separate implementations for each use.

| Compilation scope | Operands of the shared algorithms | Result consumed by lowering |
|---|---|---|
| Function parameters and operand values | Known shapes, layouts, aliases, uses and placement | Bound storage, addresses and lifetimes |
| Repeated calls and their values | Call dependencies, delays and known iteration bounds | Expanded occurrences, copies and reuse assignments |
| Function execution and transport | Tasks, resources, costs and dependency relations | Scheduled operations and required synchronization |
| Prologue/epilogue fusion | Numerical bodies, accesses and the same producer/consumer relations | Composed operations with internal edges and fused publication/acquisition where applicable |

**Write:** expose and reuse the cited algorithms through their owning modules, with explicit operands and retained
results. Complete the consumer connections in item 6; do not copy transport-specific orchestration into each scope.
A compact serialized result and the header functions that access it can use the same paged storage as other values.
Append derived structures and retain their references instead of serializing and reparsing equivalent facts.
Retained sections use ordinary ownership and lifetime rules.
Final emission consumes the solved choices; execution repeatedly invokes the compiled operations on their operands.
Preserve the numerical algorithms' conditions and selected-work semantics without introducing an interpreter for
static scheduling or lowering decisions. Deliberately open representation choices call for the smallest composition
that satisfies these requirements, recorded as implementation choices under AOT's existing documentary rule.

Implementation starting points from the source reading at mesh `774f08e4`: `rdma/mesh-graph.h` exposes retained
relations, `mesh_graph_chains`, `mesh_graph_completion_times` and `mesh_graph_heap_order`; `rdma/mesh-collective.c`
contains `plan_unroll`, their transport consumers and local storage planning. `mesh_graph_relation_save/load` in
`rdma/mesh-graph.c` are concrete places to inspect retained serialized relations and borrowed views. These are
locations for modularization and reuse, not a claim that every required consumer or runtime path is complete.

## Reusable form

The same change-request form can name several existing requests in one requirement:

- **Applies to:** `<request files and exact current clauses>`.
- **Source:** `<external author, title, year and section>`.
- **The data structure** is `<the named graph, type, table or schedule, with its source>`.
- **The algorithm** is `<the named method, with its source and its steps>`.
- **Composes:** `<the function that produces the value>` → `<the type and its fields>` → `<the function that takes
  it, through the existing result or its reference/view>`.
- **Property preserved:** `<what the emitted program preserves, which decisions it specializes, and how the
  solved results remain accessible>`.
- **Replaces:** `<the code deleted>`.
- **Write:** `<the changes, by file>`.
