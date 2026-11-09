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
