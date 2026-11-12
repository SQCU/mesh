# Requirements by stage

Every item of the eleven change requests, under the stage of [data-flow.md](data-flow.md) that takes it, or under one
of the places outside those stages that "How it is written" names. The items under "no stage named" are copied here
and are not restated.

## The turns

The operator, 2026-11-10 21:02 (quoted in [data-flow.md](data-flow.md)):

> for any change requests outstanding before that separate repo was made, what structural chagnes require the total reappraisal and total rewriting of the change request documents with respect to the strictly available xample? What hapened tothe user's statements that a definite data flow be written exactly as documentation and used to recondition and restate all outstanding requirements?

The operator, 2026-11-12:

> "- Not yet changed: the earlier requests are still not restated against data-flow.md, so reviewers will keep discounting them.' then address this at priority, up to and including all 'hooks' or scripted review functionalities

## How it is written

- The stages are those of [data-flow.md](data-flow.md) and [the-compiler.md](the-compiler.md):
  `compile_module(module, args, processors, placement, cost) -> Scheduled` is `export -> place -> on_devices ->
  schedule -> argument_buffers`; `lower(scheduled)`; `Program(scheduled, functions)`; `save` and `load`.
- Five sections are not stages of that flow. A commit's changed files say which sections it is read against:
  - graph module: `rdma/mesh_graph.py`, `rdma/mesh-graph.h` and `rdma/mesh-graph.c`, which `place`, `schedule` and
    `lower` call with their operands. compilation-composition-change-request.md:153: "`rdma/mesh_graph.py` and
    `rdma/mesh-graph.{h,c}` define the shared graph functions, which take the caller's operands."
  - bridge: the bridge process and the transport, `rdma/*.c` and `rdma/*.h` other than the graph module and
    `rdma/mesh-collective.c`, and `compiler/mesh_compiler/bridge.py`; it takes the page lists `schedule` produces,
    and `Program` reaches it. No document assigns these files; the assignment is Claude's, made from where the
    section's clauses say they are effected.
  - caller: `examples/`, `compiler/samples.py`, launchers and measurements, which give `compile_module` its operands.
    The assignment is Claude's, as the bridge's is.
  - machines: the condition of machines and services, which no file of either repository holds.
  - documents: `design/`, `docs/` and the records under metal-microbench `output_data/`. The assignment is Claude's.
- `rdma/mesh-collective.c`, which holds `mesh_plan`, is read against the schedule stage.
  disjoint-pages-change-request.md:383-384: "**Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`) takes the
  program's values and its dependence graph and writes the static schedule".
- "no stage named" holds the items whose clauses place them in a flow that is none of the above, each with the words
  that say so. They are not restated against [data-flow.md](data-flow.md): the operator's 2026-11-10 21:02 turn above
  asks for that, and it is outstanding for them.
- Each entry is an item of a change request as [change-requests-index.md](change-requests-index.md) names it, the
  lines of its request, and its clause copied from those lines. `[...]` joins passages that are not adjacent. Where a
  reading's copy differed from the file, the clause is the whole paragraphs or list items of the cited lines. The
  request is the text that governs; this document adds no requirement to it.
- "Placed by" quotes the words of a document that put the clause at its stage. Where it says "the reader", no
  document says, and the stage is the assignment of the reading named below.
- Whether an item is covered or outstanding is in the index, with its date and commit, and not here.
- The assignment was read on 2026-11-12 from the requests at mesh 60dc2097, one reader a request; the readings are in
  metal-microbench `output_data/change-request-audit-20261112/stages.json`. [review-warnings.md](review-warnings.md),
  rules 4 and 6, say how a reviewer uses it.

## export

`export`: the program to its dataflow graph, the type `Exported`.

### export: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98)
  > Each stage is a function from a value to a value. The reference program ~/schedule-emit implements this data flow,
  > and its function and type names are given beside each stage. 1. **Export** (schedule_emit/export.py `export`): a
  > program in, its dataflow graph out (`ir.Exported`). 2. **Placement** (placement.py `place`): the exported graph
  > and the placement operand in; the same graph out, with each operation's processor and with send and recv nodes
  > added. 3. **Devices** (devices.py `on_devices`): the placed graph and the mapping of processors onto devices in;
  > the placed graph with each operation on its device out. 4. **Schedule** (schedule.py `schedule`): the placed graph
  > and the cost operand in; each processor's operations in scheduled order out. There is one schedule. 5. **The
  > scheduled value** (`ir.Scheduled`): what the four stages produce. 6. **Lowering** (lower_inductor.py `lower`),
  > **the runtime** (runtime.py `Program`) and **the archive** (archive.py `save`, `load`) each take the scheduled
  > value.
  - Placed by: The clause names every stage itself, the first being "1. **Export** (schedule_emit/export.py `export`):
    a program in, its dataflow graph out (`ir.Exported`)." (design/aot-compilation-change-request.md:89); the rest at
    design/aot-compilation-change-request.md:90-98
- **A1 data structure and algorithm** (design/aot-compilation-change-request.md:123-128)
  > **The data structure is the exported program's dataflow graph:** a graph of operations whose tensors have their
  > shapes and element types. Nothing at run time changes a shape. It is `torch.export`'s "full graph representation
  > of a PyTorch model" (PyTorch v2.11.0, docs/source/user_guide/torch_compiler/export.md). - **The algorithm is
  > `torch.export`'s ahead-of-time trace.** `export` "produces a traced graph representing only the Tensor computation
  > of the function in an Ahead-of-Time (AOT) fashion, which can subsequently be executed with different inputs or
  > serialized" (torch 2.14.0, torch/export/__init__.py).
  - Placed by: "**Stage:** export." (design/aot-compilation-change-request.md:122)
- **A1 not a torch.compile backend** (design/aot-compilation-change-request.md:129-129)
  > **It is not a `torch.compile` backend.**
  - Placed by: "**Stage:** export." (design/aot-compilation-change-request.md:122)
- **A1 produced by, consumed by (the type)** (design/aot-compilation-change-request.md:147-148)
  > **Produced by, consumed by.** Export takes the torch program and returns its graph. A2's placement takes that
  > graph. The reference program's type for it is `ir.Exported`, which holds a `torch.fx.GraphModule`.
  - Placed by: "Export takes the torch program and returns its graph." (design/aot-compilation-change-request.md:147);
    "**Stage:** export." (design/aot-compilation-change-request.md:122)
- **A1 every function of a program** (design/aot-compilation-change-request.md:150-151)
  > Every function of a torch program is captured and compiled before the program runs, the same at every rank, and
  > nothing runs eagerly between its compiled calls. The program's calls between functions are A10's graph.
  - Placed by: "**Stage:** export." (design/aot-compilation-change-request.md:122); "| A10 | export (the calls'
    graph), then an analysis whose output the schedule and the runtime take |"
    (design/aot-compilation-change-request.md:111)
- **A1 a program run through another framework's delegate** (design/aot-compilation-change-request.md:152-154)
  > **A program run through another framework's delegate.** Its graph is the recorder's capture of the delegate's
  > commands (metal_recording.m). For the program the code runs today on LiteRT, tools/prepare_e2b.py reads the
  > model's graph.
  - Placed by: "**Stage:** export." (design/aot-compilation-change-request.md:122)
- **A10 data structure: the calls' dataflow graph** (design/aot-compilation-change-request.md:596-604)
  > **The data structure is the calls' dataflow graph.** It is Pathways' program tracer: "a new program tracer (Figure
  > 2) that a user can wrap around a block of Python code that calls many compiled functions. The tracer generates a
  > single PATHWAYS program where each compiled function is represented by a computation node in a dataflow graph"
  > (Barham et al., 2022, §3). - For torch, the tracer is torch.fx's symbolic tracer (J. Reed, Z. DeVito, H. He, A.
  > Ussery and J. Ansel, torch.fx: practical program capture and transformation for deep learning in Python, MLSys
  > 2022). Each compiled function is a leaf, so a call of it is one node of the graph and is not traced into
  > (`Tracer.is_leaf_module`, `torch.fx.wrap`). Each function itself is captured as A1 states. - Its reads and writes
  > are its nodes' tensor arguments and results, which is what Bernstein's conditions take.
  - Placed by: "**Stage:** export produces the calls' graph." (design/aot-compilation-change-request.md:591)
- **A10 algorithms: independence, commuting operations, the bound**
  (design/aot-compilation-change-request.md:607-617): stated under no stage named, which its clause names with this
  stage.
- **The files and functions: rdma/mesh_compile.py** (design/aot-compilation-change-request.md:1064-1069): stated under
  place, which its clause names with this stage.
- **The files and functions: metal-microbench tools/prepare_e2b.py**
  (design/aot-compilation-change-request.md:1087-1087)
  > **tools/prepare_e2b.py:** reads the model's graph (A1); declares each consumer's kind (item 5).
  - Placed by: "**Stage:** export." (design/aot-compilation-change-request.md:122) with "For the program the code runs
    today on LiteRT, tools/prepare_e2b.py reads the model's graph."
    (design/aot-compilation-change-request.md:153-154); "For the program the code runs today, tools/prepare_e2b.py
    does the declaring:" (design/aot-compilation-change-request.md:993-994)
- **The files and functions: metal-microbench metal_recording.m** (design/aot-compilation-change-request.md:1088-1088)
  > **metal_recording.m:** the recorder's capture of a delegate's commands (A1).
  - Placed by: "**Stage:** export." (design/aot-compilation-change-request.md:122) with "Its graph is the recorder's
    capture of the delegate's commands (metal_recording.m)." (design/aot-compilation-change-request.md:152-153)

### export: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **1 / data structure and Composes (export)** (design/compilation-composition-change-request.md:111-122)
  > - **The data structure** is the exported graph of each numerical function, and the graph of calls between
  > functions (aot-compilation A1, A10). - **The algorithm** is full conversion (S2): every operation of the input is
  > converted to the target, and the conversion succeeds only when all of them are. - **Composes:** export produces
  > both graphs. Lowering takes the scheduled value and converts every function in it: for every rank, its numerical
  > functions, its collectives (the `send` and `recv` nodes placement added), and each placed CPU or GPU function. -
  > **Replaces:** the split in `rdma/mesh_compile.py program()` where `replayed` excludes nodes whose declarations
  > name transferred values; also its rejection of repeated calls to a compiled function in the `replayed` block. -
  > **Write:** Compile both each numerical function and the graph of calls between functions. Lowering covers every
  > rank and its collectives, and each placed CPU or GPU function. An intermediate conversion is completed before the
  > program runs. A missing lowering within the assigned domain is unfinished work, not a new restriction on that
  > domain.
  - Placed by: "**Composes:** export produces both graphs." design/compilation-composition-change-request.md:115
- **2 / data structure: send and recv nodes in the one graph**
  (design/compilation-composition-change-request.md:128-134): stated under place, which its clause names with this
  stage.
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **6 / Replaces 4** (design/compilation-composition-change-request.md:256-258): stated under schedule, which its
  clause names with this stage.
- **Correction: generated target source / Composes: both frontends at export**
  (design/compilation-composition-change-request.md:302-303)
  > **Composes:** the Torch and LiteRT compiler interfaces produce the operations and their annotations at export.
  > Placement, the schedule and lowering take and return them.
  - Placed by: "the Torch and LiteRT compiler interfaces produce the operations and their annotations at export."
    design/compilation-composition-change-request.md:302
- **Correction: generated target source / Write: frontends supply facts; lowerings preserve identities**
  (design/compilation-composition-change-request.md:309-312)
  > the producing Torch and LiteRT compiler interfaces supply the operation, binding, type, shape, indexing, effect,
  > alias and transfer/generation facts required by their consumers. Lowerings transform those representations,
  > preserving or updating the corresponding identities, accesses and dependencies when they fuse, split, specialize
  > or rebind operations. Consumers query the resulting annotations and relations.
  - Placed by: "the Torch and LiteRT compiler interfaces produce the operations and their annotations at export."
    design/compilation-composition-change-request.md:302
- **Complete replacement / operand table row 1: Numerical operations**
  (design/compilation-composition-change-request.md:425-425)
  > | Numerical operations | Torch FX/decompositions and LiteRT's typed model/operation construction → regions with
  > typed values, shapes, layouts, indexing maps, reduction axes, quantization and numerical precision semantics →
  > specialization and fusion. Constants are kept apart from varying inputs. |
  - Placed by: "Both frontends supply the following operands:" design/compilation-composition-change-request.md:421;
    "the Torch and LiteRT compiler interfaces produce the operations and their annotations at export."
    design/compilation-composition-change-request.md:302
- **Complete replacement / many-to-many identities; producer-supplied meaning**
  (design/compilation-composition-change-request.md:434-436)
  > An identity correspondence can be many-to-many after fusion or splitting. A tensor's reduction or quantization
  > meaning is supplied by its producer, not guessed from an arbitrary Tint instruction sequence.
  - Placed by: "Both frontends supply the following operands:" design/compilation-composition-change-request.md:421;
    "the Torch and LiteRT compiler interfaces produce the operations and their annotations at export."
    design/compilation-composition-change-request.md:302
- **Withdrawn, 2026-11-09: the stated scope** (design/compilation-composition-change-request.md:556-556)
  > A program's graph has no node that branches or repeats on tensor data, and its shapes are fixed when it is
  > compiled.
  - Placed by: the reader.

### export: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R2 where it composes** (design/plan-to-bridge-change-request.md:2112-2121): stated under schedule, which its
  clause names with this stage.
- **R3 the task graph: V and w** (design/plan-to-bridge-change-request.md:2137-2137, 2141-2141, 2143-2144): stated
  under schedule, which its clause names with this stage.
- **R3 where it composes** (design/plan-to-bridge-change-request.md:2147-2155): stated under schedule, which its
  clause names with this stage.
- **R8 data structure and algorithm: one offline allocation at compile time**
  (design/plan-to-bridge-change-request.md:2429-2432): stated under schedule, which its clause names with this stage.

### export: [torch-capture-change-request.md](torch-capture-change-request.md)

- **3, torch scope** (design/torch-capture-change-request.md:521-522)
  > **In the torch scope,** design/torch-source-lowering-change-request.md item 1 writes this item
  > (plan-to-bridge-change-request.md R4: "A function that takes pages in any order has one task for each page.").
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    (design/torch-source-lowering-change-request.md:129-130)
- **8. The torch scope's final lowering** (design/torch-capture-change-request.md:796-830)
  > - **Replaces:** the torch scope's kernels, which are torch's MPS kernels, one ATen operator at a time (mesh
  > 03297e52). - **Source:** read and quoted in design/final-lowering-sources-2026-11-06.md. - J. Ansel et al.,
  > PyTorch 2, ASPLOS 2024, §1: "The framework only has visibility of a single operator at a time, and thus cannot
  > automatically perform optimizations, like fusion or scheduling, that cross operator boundaries." - T. Chen et al.,
  > TVM, OSDI 2018, abstract: "TVM solves optimization challenges specific to deep learning, such as high-level
  > operator fusion, mapping to arbitrary hardware primitives, and memory latency hiding." - J. Ragan-Kelley et al.,
  > Halide, PLDI 2013, §1: "decoupling the algorithm definition from its execution strategy". - P. Tillet, H. T. Kung,
  > D. Cox, Triton, MAPL 2019, §5.2.1: "Nested tiling strategies … aim at decomposing tiles into micro-tiles and
  > eventually nano-tiles in order to fit a machine's compute capabilities and memory hierarchy as tightly as
  > possible." - B. Spector et al., ThunderKittens (arXiv:2410.20399), §1: "any high performance framework must
  > prioritize keeping tensor cores at high utilization whenever possible." - C. Cecka, CuTe layout algebra
  > (arXiv:2603.02298), abstract: "These instructions prescribe fixed, often complex data layouts that must be
  > correctly propagated through the entire execution pipeline to ensure both correctness and optimal performance." -
  > T. Dao et al., FlashAttention, NeurIPS 2022, abstract: "an IO-aware exact attention algorithm that uses tiling to
  > reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM." - Y. Xu et
  > al., GSPMD (arXiv:2105.04663), §1: "GSPMD will run compiler passes that complete the sharding specification on the
  > entire computation graph, and transform it into a mathematically equivalent, parallelized computation to run on
  > each device." - N. Shazeer et al., Mesh-TensorFlow, NeurIPS 2018, abstract: "A Mesh-TensorFlow graph compiles into
  > a SPMD program consisting of parallel operations coupled with collective communication primitives such as
  > Allreduce." - L.-W. Chang et al., FLUX (arXiv:2406.06858), abstract: "Flux overdecomposes communication and
  > computation operations into much finer-grained operations and further fuses them into a larger kernel to
  > effectively hide communication without compromising kernel efficiency." - S. Zheng et al., TileLink
  > (arXiv:2503.20313), abstract: "TileLink decouples the design space of communication and computation, linking these
  > two parts via tile-centric primitives." - A. Jangda et al., CoCoNet, ASPLOS 2022, §1: "we need to schedule the
  > MatMul kernel to compute chunks in the order the AllReduce kernel communicates them." - Triton-distributed
  > (arXiv:2504.19442), §2.1: "The core of our programming model includes three concepts: symmetric memory, signal
  > exchange, and async-task." - Metal Performance Primitives, MPPTensorOpsMatMul2d.h:216–218 (macOS 26.4 SDK):
  > "cooperative_tensor owns thread private data and divides the data for entire tensor among threads (participating
  > in the scope of operation) in implementation defined manner." - **Written** in
  > design/torch-source-lowering-change-request.md, items 1–3.
  - Placed by: "Item 2 is the export stage's scope. Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    (design/torch-source-lowering-change-request.md:129-130); "Item 3's ordering is the schedule stage's."
    (design/torch-source-lowering-change-request.md:132)

### export: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **2 / Write: the exported graph** (design/torch-source-lowering-change-request.md:189-190)
  > The data structure is the exported graph: the Core ATen IR graph that `torch.export.export` returns for each
  > function, with every shape compiled.
  - Placed by: "This is the export stage's scope. The export stage produces the exported graph."
    design/torch-source-lowering-change-request.md:191
- **2 / Write: the export stage produces it** (design/torch-source-lowering-change-request.md:191-191)
  > This is the export stage's scope. The export stage produces the exported graph.
  - Placed by: "This is the export stage's scope. The export stage produces the exported graph."
    design/torch-source-lowering-change-request.md:191
- **2 / Write: dynamic control flow is not compiled** (design/torch-source-lowering-change-request.md:193-195)
  > A program with dynamic control flow is not compiled, whether the control flow is based on CPU data or on GPU data,
  > and whether or not it is written with a higher order operator (`torch.cond`, `while_loop`, `map`, `scan`). The
  > exception in the PyTorch sentence quoted above is not taken.
  - Placed by: "Item 2 is the export stage's scope." design/torch-source-lowering-change-request.md:129
- **2 / Write: a dimension known only at run time** (design/torch-source-lowering-change-request.md:195-196)
  > A program with a dimension known only at run time is not compiled either (design/aot-compilation-change-request.md
  > A1: "shapes are compiled").
  - Placed by: "Item 2 is the export stage's scope." design/torch-source-lowering-change-request.md:129

## place

`place`: the placement operand applied, the sharding, the sends and receives of collectives.

### place: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **A1 produced by, consumed by (the type)** (design/aot-compilation-change-request.md:147-148): stated under export,
  which its clause names with this stage.
- **A2 data structure, part 1 (sharding specs)** (design/aot-compilation-change-request.md:167-169)
  > **The data structure is the placement operand,** of two parts: - sharding specs on the function's tensors over a
  > logical device mesh (Y. Xu et al., GSPMD: general and scalable parallelization for ML computation graphs, arXiv
  > 2105.04663, 2021);
  - Placed by: "**Stage:** placement, then devices." (design/aot-compilation-change-request.md:166); "Placement takes
    A1's graph and the sharding specs and returns the partitioned graph with its collectives"
    (design/aot-compilation-change-request.md:177-178); "**A placement also shards.** The placement operand gives a
    tensor's sharding over machines." (design/the-compiler.md:46)
- **A2 algorithm (GSPMD's partitioning)** (design/aot-compilation-change-request.md:174-176)
  > **The algorithm is GSPMD's partitioning.** It consumes sharding annotations on a few tensors, propagates them to
  > every operator by compiler passes over a dataflow IR, and inserts the collectives. It "produces a single program
  > for all partitions" (GSPMD §1).
  - Placed by: "Placement takes A1's graph and the sharding specs and returns the partitioned graph with its
    collectives" (design/aot-compilation-change-request.md:177-178); "**Stage:** placement, then devices."
    (design/aot-compilation-change-request.md:166)
- **A2 produced by, consumed by** (design/aot-compilation-change-request.md:177-181)
  > **Produced by, consumed by.** The caller passes the operand. Placement takes A1's graph and the sharding specs and
  > returns the partitioned graph with its collectives, which A4 turns into sends and receives of pages. Devices takes
  > that graph and the mapping onto the map's nodes. A6's schedule takes the result, in A8's IR. In the reference
  > program the operand is `placement`, a function from an fx node to a processor name, and `processors`, a dict from
  > processor name to torch device.
  - Placed by: "**Stage:** placement, then devices." (design/aot-compilation-change-request.md:166); the clause's own
    "Placement takes A1's graph and the sharding specs" (design/aot-compilation-change-request.md:177)
- **A2 the chooser is separate** (design/aot-compilation-change-request.md:182-182): stated under caller, which its
  clause names with this stage.
- **A4 stage** (design/aot-compilation-change-request.md:219-221): stated under schedule, which its clause names with
  this stage.
- **A4 algorithm: AllReduce** (design/aot-compilation-change-request.md:235-237): stated under schedule, which its
  clause names with this stage.
- **A4 algorithm: AllGather** (design/aot-compilation-change-request.md:238-240): stated under schedule, which its
  clause names with this stage.
- **A4 algorithm: ReduceScatter** (design/aot-compilation-change-request.md:241-244): stated under schedule, which its
  clause names with this stage.
- **A4 algorithm: CollectivePermute** (design/aot-compilation-change-request.md:245-246): stated under schedule, which
  its clause names with this stage.
- **A4 algorithm: AllToAll** (design/aot-compilation-change-request.md:247-248): stated under schedule, which its
  clause names with this stage.
- **A4 where the partitioning inserts ReduceScatter and CollectivePermute**
  (design/aot-compilation-change-request.md:255-263)
  > **Where the partitioning inserts ReduceScatter and CollectivePermute:** where GSPMD's does. - ReduceScatter goes
  > where a reduction's result is sharded: "gradients will be communicated with ReduceScatter instead of AllReduce"
  > (GSPMD §3.2). - CollectivePermute goes into halo exchange, "We use the CollectivePermute operator to exchange halo
  > data between partitions" (§4.3), and into a pipeline's buffer shifting, "GSPMD will turn the buffer shifting into
  > cross-device communication via CollectivePermute" (§3.3). - The partitioning reaches them by partitioning every
  > operator, as A2 states ("propagates them to every operator"). GSPMD gives the rule for the operators that need
  > CollectivePermute: "The most typical use case of halo exchange is for partitinoning window-based operators (e.g.,
  > Convolution, ReduceWindow)" (§4.3).
  - Placed by: "Placement inserts the collectives (A2)." (design/aot-compilation-change-request.md:219); "Placement
    takes A1's graph and the sharding specs and returns the partitioned graph with its collectives"
    (design/aot-compilation-change-request.md:177-178)
- **A4 a node to which the placement operand maps no device (mesh_plan's part)**
  (design/aot-compilation-change-request.md:267-269): stated under schedule, which its clause names with this stage.
- **A8 data structure (the type mesh_plan takes)** (design/aot-compilation-change-request.md:502-512): stated under
  scheduled value, which its clause names with this stage.
- **A8 carried as one IR from partitioning to serialization** (design/aot-compilation-change-request.md:502-503,
  510-512): stated under scheduled value, which its clause names with this stage.
- **The files and functions: rdma/mesh_compile.py** (design/aot-compilation-change-request.md:1064-1069)
  > **rdma/mesh_compile.py:** - capture by `torch.export` (A1); - `_operator`: the partitioning of every operator, the
  > window-based ones by CollectivePermute (A4); - `_all_gather` declares one broadcast from each source (A4); -
  > `_program`: the program's nodes, passed as MESH_PROGRAM (A7); - A10's analysis calls rdma/mesh_graph.py's
  > functions and holds no copy of them.
  - Placed by: The clause names a file and, per line, the requirement it belongs to; those requirements' words:
    "Placement inserts the collectives (A2)." (design/aot-compilation-change-request.md:219), "The partitioning
    reaches them by partitioning every operator, as A2 states" (design/aot-compilation-change-request.md:261),
    "**Stage:** export." (design/aot-compilation-change-request.md:122), "A torch program's launcher passes them as
    MESH_PROGRAM beside MESH_GROUP (rdma/mesh_compile.py `_program`)."
    (design/aot-compilation-change-request.md:376-377), "**Its graph operations are the graph module's.**"
    (design/aot-compilation-change-request.md:620)
- **The files and functions: rdma/mesh-collective.c (fed by A2's collectives)**
  (design/aot-compilation-change-request.md:1074-1076): stated under schedule, which its clause names with this stage.
- **The files and functions: rdma/mesh-collective.c (routing, u, queue sizes, sets)**
  (design/aot-compilation-change-request.md:1074-1078): stated under schedule, which its clause names with this stage.

### place: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **1 / Write: every rank, its collectives, each placed CPU or GPU function**
  (design/compilation-composition-change-request.md:115-122): stated under lower, which its clause names with this
  stage.
- **2 / data structure: send and recv nodes in the one graph**
  (design/compilation-composition-change-request.md:128-134)
  > - **The data structure** is one graph for computation and data movement together (S3's parallel computation
  > graph): its nodes are function calls and, for each value read on another processor, a `send` and a `recv`; its
  > edges are values, each with its sharding, consumer kind, accesses and lifetimes. - **The algorithm** is graph
  > substitution (S3): a transformation replaces a subgraph of that graph with another. - **Composes:** export
  > produces the graph. Placement adds the `send` and `recv` nodes to it. One schedule orders functions and transfers
  > together and produces the scheduled value. Lowering, the runtime and the archive take the scheduled value. The
  > signature is printed from what lowering emits.
  - Placed by: "Placement adds the `send` and `recv` nodes to it."
    design/compilation-composition-change-request.md:132
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **3 / Write, second bullet: callers call them, own copies deleted**
  (design/compilation-composition-change-request.md:150-157): stated under graph module, which its clause names with
  this stage.
- **3 / Write, second bullet: the four named deletions** (design/compilation-composition-change-request.md:155-157):
  stated under schedule, which its clause names with this stage.
- **4 / Write: placement and costs are operands** (design/compilation-composition-change-request.md:170-173): stated
  under caller, which its clause names with this stage.
- **Correction: generated target source / Composes: both frontends at export**
  (design/compilation-composition-change-request.md:302-303): stated under export, which its clause names with this
  stage.
- **Complete replacement / operand table row 4: Program relations**
  (design/compilation-composition-change-request.md:428-428): stated under scheduled value, which its clause names
  with this stage.
- **7 / residual runtime operands and waits** (design/compilation-composition-change-request.md:531-535): stated under
  Program, which its clause names with this stage.

### place: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R1 where it composes** (design/plan-to-bridge-change-request.md:1970-1977): stated under schedule, which its
  clause names with this stage.
- **R3 where it composes** (design/plan-to-bridge-change-request.md:2147-2155): stated under schedule, which its
  clause names with this stage.
- **R5 where it composes** (design/plan-to-bridge-change-request.md:2351-2362): stated under schedule, which its
  clause names with this stage.
- **R6 data structure and no run-time algorithm** (design/plan-to-bridge-change-request.md:2387-2392)
  > **The data structure is the transfers of the static schedule:** the send and receive nodes that placement adds to
  > the graph, which R5 writes into each cable's lists. **No algorithm runs for it at run time.** The problem is the
  > metadata problem of dynamic sparse data exchange (T. Hoefler, C. Siebert and A. Lumsdaine, Scalable communication
  > protocols for dynamic sparse data exchange, PPoPP 2010). Its run-time protocols learn who sends to whom. Here it
  > is answered at compile time: every rank computes every transfer of the static schedule (R5).
  - Placed by: "Produced by placement, as `struct mesh_declared` with its `source` and `destination`
    (rdma/mesh-collective.h), and by the schedule (R5), as `struct mesh_hop` with its `node` and `direction`
    (rdma/mesh.h)." design/plan-to-bridge-change-request.md:2393-2395; "A collective that the sharding requires is
    `send` and `recv` nodes that `place` adds." design/the-compiler.md:46-47
- **R6 where it composes** (design/plan-to-bridge-change-request.md:2393-2396)
  > **Where it composes.** Produced by placement, as `struct mesh_declared` with its `source` and `destination`
  > (rdma/mesh-collective.h), and by the schedule (R5), as `struct mesh_hop` with its `node` and `direction`
  > (rdma/mesh.h). Consumed by the bridge at each node, which reads its own lists. This requirement has no code of its
  > own.
  - Placed by: "Produced by placement, as `struct mesh_declared` with its `source` and `destination`
    (rdma/mesh-collective.h), and by the schedule (R5), as `struct mesh_hop` with its `node` and `direction`
    (rdma/mesh.h). Consumed by the bridge at each node, which reads its own lists."
    design/plan-to-bridge-change-request.md:2393-2395
- **R8 where it composes: placement and weights are operands** (design/plan-to-bridge-change-request.md:2438-2439):
  stated under caller, which its clause names with this stage.

### place: [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

- **1, step 1** (design/problems-of-14b9a757-change-request.md:125-126)
  > Block j of member i (j ≠ i) is a transfer from the tensor view of block j in i's operand to the tensor view of
  > block i in j's result: a send at i and a receive at j (MSCCLang §4, quoted above).
  - Placed by: "the placement stage writes step 1's send and receive nodes and step 2's `copy_` node into the exported
    graph. The schedule reads them as nodes of that graph. In `mesh_plan`'s operand each transfer of step 1 is one
    `struct mesh_declared`, its two tensor views the `from` and the `to`."
    design/problems-of-14b9a757-change-request.md:131-133
- **1, step 2** (design/problems-of-14b9a757-change-request.md:127-129)
  > Block i is one local copy from the operand's tensor view to the result's tensor view, as MPIR_Localcopy does it.
  > It is torch's `copy_` ("Copies the elements from src into self tensor and returns self", torch 2.14.0
  > _tensor_docs.py), a node of the program, with no synchronize and no clone.
  - Placed by: "the placement stage writes step 1's send and receive nodes and step 2's `copy_` node into the exported
    graph. The schedule reads them as nodes of that graph." design/problems-of-14b9a757-change-request.md:131-132
- **1, step 3** (design/problems-of-14b9a757-change-request.md:130-130)
  > The local copy is absent only where the operand is the result with the same type map (MPI's in-place condition).
  - Placed by: "the placement stage writes step 1's send and receive nodes and step 2's `copy_` node into the exported
    graph." design/problems-of-14b9a757-change-request.md:131-132
- **1, Change: the refusal is deleted** (design/problems-of-14b9a757-change-request.md:149-149)
  > The refusal is deleted.
  - Placed by: "rdma/mesh_compile.py:1480–1484 (`_inside`) raises NotImplementedError for every `all-to-all`"
    design/problems-of-14b9a757-change-request.md:135; "the placement stage writes step 1's send and receive nodes and
    step 2's `copy_` node into the exported graph." design/problems-of-14b9a757-change-request.md:131-132
- **2, step 1** (design/problems-of-14b9a757-change-request.md:157-158)
  > Where the producer of member i's input writes through an `out` tensor, its `out` is the tensor view of block i of
  > the result. That is NCCL's in-place condition.
  - Placed by: "the placement stage sets the producer node's `out`, or writes the `copy_` node, in the exported graph.
    The schedule reads them as nodes of that graph." design/problems-of-14b9a757-change-request.md:161-162
- **2, step 2** (design/problems-of-14b9a757-change-request.md:159-160)
  > Otherwise block i is one local copy into that tensor view, as MPICH's `sendbuf != MPI_IN_PLACE` branch does, with
  > the same form as item 1's.
  - Placed by: "the placement stage sets the producer node's `out`, or writes the `copy_` node, in the exported graph.
    The schedule reads them as nodes of that graph." design/problems-of-14b9a757-change-request.md:161-162
- **2, Change: both refusals are deleted** (design/problems-of-14b9a757-change-request.md:184-184)
  > Both refusals are deleted.
  - Placed by: "rdma/mesh_compile.py:1497–1504 (`_direct`) raises where the producer of the member's own block is not
    an out variant, a multi-output result or a ReduceScatter." design/problems-of-14b9a757-change-request.md:164-165;
    "rdma/mesh_compile.py:1430–1432 (`_strided`) raises for "a transfer's end in no placed value"."
    design/problems-of-14b9a757-change-request.md:166; "the placement stage sets the producer node's `out`, or writes
    the `copy_` node, in the exported graph." design/problems-of-14b9a757-change-request.md:161-162
- **3, Change: the rule** (design/problems-of-14b9a757-change-request.md:190-194)
  > A tensor view node's entry in the memory plan is its base's entry, with the view's offset, sizes and strides as
  > torch computes them. - **Composition:** the placement stage writes a transfer's end that is a tensor view node as
  > its base's value with that offset, those sizes and those strides (item 10's form), as the `from` or the `to`
  > (`struct mesh_strided`) of a `struct mesh_declared` in `mesh_plan`'s operand.
  - Placed by: "the placement stage writes a transfer's end that is a tensor view node as its base's value with that
    offset, those sizes and those strides (item 10's form), as the `from` or the `to` (`struct mesh_strided`) of a
    `struct mesh_declared` in `mesh_plan`'s operand. `mesh_plan` reads it."
    design/problems-of-14b9a757-change-request.md:192-194; "- **Memory plan:** every value's pages and offsets."
    (under "What `mesh_plan` writes") design/problems-of-14b9a757-change-request.md:104
- **3, Change: the result torch allocates** (design/problems-of-14b9a757-change-request.md:200-200)
  > Only a result torch allocates outside the memory plan's pages has no entry. That case is item 2's local copy.
  - Placed by: "That case is item 2's local copy." design/problems-of-14b9a757-change-request.md:200; item 2: "the
    placement stage sets the producer node's `out`, or writes the `copy_` node, in the exported graph."
    design/problems-of-14b9a757-change-request.md:161-162
- **4, step 1** (design/problems-of-14b9a757-change-request.md:208-209)
  > **A replicated operand's window** is a DynamicSlice of the local operand: a tensor view of its pages, with no
  > transfer.
  - Placed by: "the placement stage (in XLA, the SPMD partitioner) writes the CollectivePermutes' send and receive
    nodes, the DynamicSlice tensor view nodes and the masking nodes into the exported graph. The schedule reads them
    as nodes of that graph." design/problems-of-14b9a757-change-request.md:217-219
- **4, step 2** (design/problems-of-14b9a757-change-request.md:210-211)
  > **A sharded operand's window** is A.2's sequence for each spatial dimension: the halo exchange of the maximum
  > sizes by CollectivePermute (item 5), then DynamicSlice, then masking.
  - Placed by: "the placement stage (in XLA, the SPMD partitioner) writes the CollectivePermutes' send and receive
    nodes, the DynamicSlice tensor view nodes and the masking nodes into the exported graph. The schedule reads them
    as nodes of that graph." design/problems-of-14b9a757-change-request.md:217-219
- **4, step 3** (design/problems-of-14b9a757-change-request.md:212-214)
  > **Several sharded dimensions:** the exchanges run one sharded dimension at a time, in increasing dimension order.
  > Each exchange's operand is the previous exchange's result, so the second exchange's halos carry the first's, and
  > the corners arrive with them. A dimension with one shard is skipped.
  - Placed by: "the placement stage (in XLA, the SPMD partitioner) writes the CollectivePermutes' send and receive
    nodes, the DynamicSlice tensor view nodes and the masking nodes into the exported graph. The schedule reads them
    as nodes of that graph." design/problems-of-14b9a757-change-request.md:217-219
- **4, step 4** (design/problems-of-14b9a757-change-request.md:215-216)
  > **The partition's own shard**, between its halos, is in place where its producer's `out` is the tensor view of it
  > in the window (item 2's condition), and one local copy otherwise.
  - Placed by: "(item 2's condition)" design/problems-of-14b9a757-change-request.md:216; item 2: "the placement stage
    sets the producer node's `out`, or writes the `copy_` node, in the exported graph. The schedule reads them as
    nodes of that graph." design/problems-of-14b9a757-change-request.md:161-162
- **4, Change: the refusals are deleted** (design/problems-of-14b9a757-change-request.md:259-259)
  > The refusals are deleted, the one for more than one dimension among them.
  - Placed by: "rdma/mesh_compile.py:1490–1493 and 1497–1504 raise where a partition's own shard of a windowed operand
    is not written in place." design/problems-of-14b9a757-change-request.md:221-222; "Two partitioned spatial
    dimensions raise at the same place." design/problems-of-14b9a757-change-request.md:224; "the placement stage (in
    XLA, the SPMD partitioner) writes the CollectivePermutes' send and receive nodes, the DynamicSlice tensor view
    nodes and the masking nodes into the exported graph." design/problems-of-14b9a757-change-request.md:217-218
- **5, step 1** (design/problems-of-14b9a757-change-request.md:265-265)
  > For each pair (p, q) with p ≠ q, the operand is sent from p to q: a send at p and a receive at q.
  - Placed by: "the placement stage writes the send and receive nodes, the `copy_` node and the zeros into the
    exported graph. The schedule reads them as nodes of that graph."
    design/problems-of-14b9a757-change-request.md:268-269
- **5, step 2** (design/problems-of-14b9a757-change-request.md:266-266)
  > A pair (p, p) is item 1's local copy at p.
  - Placed by: "the placement stage writes the send and receive nodes, the `copy_` node and the zeros into the
    exported graph. The schedule reads them as nodes of that graph."
    design/problems-of-14b9a757-change-request.md:268-269
- **5, step 3** (design/problems-of-14b9a757-change-request.md:267-267)
  > A replica that is no target outputs zeros.
  - Placed by: "the placement stage writes the send and receive nodes, the `copy_` node and the zeros into the
    exported graph. The schedule reads them as nodes of that graph."
    design/problems-of-14b9a757-change-request.md:268-269
- **5, Change: the refusal is deleted** (design/problems-of-14b9a757-change-request.md:279-279)
  > The refusal is deleted.
  - Placed by: "rdma/mesh_compile.py:1788–1790 raises for any pair whose source is its target."
    design/problems-of-14b9a757-change-request.md:271; "the placement stage writes the send and receive nodes, the
    `copy_` node and the zeros into the exported graph." design/problems-of-14b9a757-change-request.md:268-269
- **6, step 1** (design/problems-of-14b9a757-change-request.md:285-285)
  > Test torch's condition for the new tensor view's dimensions: ∀ i = d, …, d+k−1, stride[i] = stride[i+1] ×
  > size[i+1].
  - Placed by: "the placement stage writes each tensor view node as step 2's tensor view or as step 3's copy node, in
    the exported graph. The schedule reads it as a node of that graph, and `mesh_plan` gives the copy's `out` value
    its pages." design/problems-of-14b9a757-change-request.md:288-290
- **6, step 2** (design/problems-of-14b9a757-change-request.md:286-286)
  > Where it holds, the node is a tensor view.
  - Placed by: "the placement stage writes each tensor view node as step 2's tensor view or as step 3's copy node, in
    the exported graph. The schedule reads it as a node of that graph, and `mesh_plan` gives the copy's `out` value
    its pages." design/problems-of-14b9a757-change-request.md:288-290
- **6, step 3** (design/problems-of-14b9a757-change-request.md:287-287)
  > Otherwise it is `contiguous` into a value in the memory plan, run as an out-variant copy.
  - Placed by: "the placement stage writes each tensor view node as step 2's tensor view or as step 3's copy node, in
    the exported graph. The schedule reads it as a node of that graph, and `mesh_plan` gives the copy's `out` value
    its pages." design/problems-of-14b9a757-change-request.md:288-290
- **7.2, step 1** (design/problems-of-14b9a757-change-request.md:354-355)
  > Each collective has one completion wait at each node: torch's `wait_tensor`, placed at the graph node before the
  > first use of the collective's result there.
  - Placed by: "the placement stage writes the `wait_tensor` node into the exported graph."
    design/problems-of-14b9a757-change-request.md:359
- **7.2, step 2** (design/problems-of-14b9a757-change-request.md:356-356): stated under Program, which its clause
  names with this stage.
- **7.2, step 3** (design/problems-of-14b9a757-change-request.md:357-358): stated under schedule, which its clause
  names with this stage.
- **8, step 1** (design/problems-of-14b9a757-change-request.md:418-419)
  > An invariant's sharding at the partitioner is replicated: every node holds it whole ("All devices have the same
  > full data"; `ExportedProgram`'s state_dict; Megatron's master weight "on all processes").
  - Placed by: "the placement stage (the partitioner) writes steps 1 to 3 as DynamicSlice tensor view nodes in the
    exported graph" design/problems-of-14b9a757-change-request.md:431-432
- **8, step 2** (design/problems-of-14b9a757-change-request.md:420-422)
  > A value made from invariants by tensor views alone is replicated too: a tensor view preserves its operand's
  > dimensions, the sharding of a preserved dimension is propagated from input to output, and propagation creates no
  > new sharding pattern (GSPMD §3.5), so a replicated operand's tensor view is replicated at every node.
  - Placed by: "the placement stage (the partitioner) writes steps 1 to 3 as DynamicSlice tensor view nodes in the
    exported graph" design/problems-of-14b9a757-change-request.md:431-432
- **8, step 3** (design/problems-of-14b9a757-change-request.md:423-424)
  > A collective whose operand is such a value is a resharding from replicated, so it is the DynamicSlice of its
  > result's sharding, at each node, with no transfer. No transfer has an invariant as its source.
  - Placed by: "the placement stage (the partitioner) writes steps 1 to 3 as DynamicSlice tensor view nodes in the
    exported graph" design/problems-of-14b9a757-change-request.md:431-432
- **8, step 4** (design/problems-of-14b9a757-change-request.md:425-426)
  > Each distinct DynamicSlice a node's program reads of an attribute, or of a value made from an attribute by tensor
  > views alone, is its own never-freed value at that node (7.1, step 5).
  - Placed by: "the placement stage (the partitioner) writes steps 1 to 3 as DynamicSlice tensor view nodes in the
    exported graph, and step 4's never-freed values into `mesh_plan`'s operand, each a `struct mesh_value` of kind
    MESH_INVARIANT. `mesh_plan` gives those values their pages." design/problems-of-14b9a757-change-request.md:431-433
- **10, Change: the representation** (design/problems-of-14b9a757-change-request.md:476-477): stated under scheduled
  value, which its clause names with this stage.
- **10, Composition: the placement stage writes it** (design/problems-of-14b9a757-change-request.md:479-480)
  > - **Composition:** the placement stage writes it for each transfer's end, as `struct mesh_strided` in
  > `mesh_plan`'s operand. `mesh_plan` reads it.
  - Placed by: "the placement stage writes it for each transfer's end, as `struct mesh_strided` in `mesh_plan`'s
    operand. `mesh_plan` reads it." design/problems-of-14b9a757-change-request.md:479-480
- **10, Change: the refusal is deleted** (design/problems-of-14b9a757-change-request.md:487-487)
  > The refusal is deleted.
  - Placed by: "rdma/mesh-collective.h:430–432 (`MESH_STRIDED_DIMS 8u`, `size[8]`, `stride[8]`);
    rdma/mesh_compile.py:1441–1443 refuses more." design/problems-of-14b9a757-change-request.md:481-482; "the
    placement stage writes it for each transfer's end, as `struct mesh_strided` in `mesh_plan`'s operand."
    design/problems-of-14b9a757-change-request.md:479-480

### place: [unequal-shards-change-request.md](unequal-shards-change-request.md)

- **1 - Data structure** (design/unequal-shards-change-request.md:91-92)
  > a tuple of integers for each sharded dimension, RaggedShard's `local_units`, its length the size of the mesh
  > dimension. Equal integers are the even sharding.
  - Placed by: "the placement operand carries the tuple, of `local_units`' type. `place` consumes it with the
    dimension's extent" design/unequal-shards-change-request.md:97-98; "A sharding is part of the placement operand.
    Items 1, 5, 8, 13 and 15 state what the operand carries for a sharded dimension."
    design/unequal-shards-change-request.md:54-55
- **1 - Algorithm** (design/unequal-shards-change-request.md:93-96)
  > HAP §5.1's rounding. It takes the tuple and the dimension's extent. Device j's size is its integer's fraction of
  > the tuple's sum, of the extent, set to the nearest integer; while the sizes do not sum to the extent, the size
  > whose change adds the least rounding error is reduced or increased by one. It writes one whole size per device,
  > the sizes summing to the extent.
  - Placed by: "`place` consumes it with the dimension's extent and writes the sizes, an integer array of one entry
    per device" design/unequal-shards-change-request.md:97-98
- **1 - Stage** (design/unequal-shards-change-request.md:97-99)
  > the placement operand carries the tuple, of `local_units`' type. `place` consumes it with the dimension's extent
  > and writes the sizes, an integer array of one entry per device, which are the `counts` of items 2 and 3.
  - Placed by: "`place` consumes it with the dimension's extent and writes the sizes"
    design/unequal-shards-change-request.md:97-98
- **2 - Data structure** (design/unequal-shards-change-request.md:124-125)
  > MPI's two arrays, `counts` (MPI's `recvcounts` and `sendcounts`) and `displs`, each with one entry per device of
  > the mesh dimension.
  - Placed by: "the sends and receives of each of these collectives are `send` and `recv` nodes `place` adds to the
    `GraphModule`, with `counts` and `displs`, each an integer array of one entry per device."
    design/unequal-shards-change-request.md:130-131
- **2 - Algorithm** (design/unequal-shards-change-request.md:126-129)
  > MPI 4.1's collectives of varying count, §7.5 to §7.10.2 as quoted. An all-gather, a reduce-scatter, an all-to-all,
  > a gather and a scatter over a dimension sharded by item 1 each carry the two arrays. The block of device j is sent
  > at `counts[j]` elements and placed at `displs[j]`. A reduce-scatter's result is n consecutive blocks, block i of
  > `counts[i]` elements, sent to device i.
  - Placed by: "the sends and receives of each of these collectives are `send` and `recv` nodes `place` adds to the
    `GraphModule`, with `counts` and `displs`" design/unequal-shards-change-request.md:130-131
- **2 - Stage** (design/unequal-shards-change-request.md:130-132)
  > the sends and receives of each of these collectives are `send` and `recv` nodes `place` adds to the `GraphModule`,
  > with `counts` and `displs`, each an integer array of one entry per device. `on_devices` and `schedule` take the
  > `GraphModule` with those nodes.
  - Placed by: "`send` and `recv` nodes `place` adds to the `GraphModule`"
    design/unequal-shards-change-request.md:130-131; "`on_devices` and `schedule` take the `GraphModule` with those
    nodes." design/unequal-shards-change-request.md:131-132
- **3 - Data structure / Algorithm** (design/unequal-shards-change-request.md:143-146)
  > - **Data structure:** the `counts` and `displs` of item 2. - **Algorithm:** item 1's rounding. The counts of item
  > 2 are computed from the tuple of item 1 and the dimension's extent, and every device of the collective is given
  > the same arrays. What device j sends is `counts[j]` elements at every device's reading of it.
  - Placed by: "`place` writes the arrays once for a collective; that collective's nodes on every device carry them."
    design/unequal-shards-change-request.md:147
- **3 - Stage** (design/unequal-shards-change-request.md:147-147)
  > `place` writes the arrays once for a collective; that collective's nodes on every device carry them.
  - Placed by: "`place` writes the arrays once for a collective" design/unequal-shards-change-request.md:147
- **4 - Data structure** (design/unequal-shards-change-request.md:165-167)
  > GSPMD §4.1's padded shape and mask. The buffer a device holds for a sharded dimension has one size for all
  > devices, rounded up; what lies past a device's own shard is arbitrary. The mask compares the index along the
  > dimension (GSPMD's Iota) with an offset computed from the device (GSPMD's PartitionId).
  - Placed by: "the select is a node `place` adds before the reducing operator, on each device, from that device's
    size of item 1." design/unequal-shards-change-request.md:171-172; "Items 2, 3, 4, 9, 10 and 12 state nodes that
    `place` adds to the graph." design/unequal-shards-change-request.md:55-56
- **4 - Algorithm / Stage** (design/unequal-shards-change-request.md:168-172)
  > - **Algorithm:** GSPMD §4.1's masking. Before an operator that reduces over that dimension, the elements past the
  > device's shard are replaced by the identity of the reduction, by a select on the mask. The padding differs by
  > device. - **Stage:** the select is a node `place` adds before the reducing operator, on each device, from that
  > device's size of item 1.
  - Placed by: "the select is a node `place` adds before the reducing operator, on each device, from that device's
    size of item 1." design/unequal-shards-change-request.md:171-172
- **5 - Data structure** (design/unequal-shards-change-request.md:180-180)
  > the tuple of item 1, one tuple for both weights.
  - Placed by: "the placement operand carries the one tuple on the first weight's column dimension and the second
    weight's row dimension. `place` consumes it and adds the nodes of the reduction."
    design/unequal-shards-change-request.md:184-185
- **5 - Algorithm** (design/unequal-shards-change-request.md:181-183)
  > Megatron-LM §3's split. The first weight of a two-layer block is cut along its columns and the second along its
  > rows by the same tuple, so the second takes the first's shard without a collective; the second's output is reduced
  > across devices.
  - Placed by: "`place` consumes it and adds the nodes of the reduction." design/unequal-shards-change-request.md:185
- **5 - Stage** (design/unequal-shards-change-request.md:184-185)
  > the placement operand carries the one tuple on the first weight's column dimension and the second weight's row
  > dimension. `place` consumes it and adds the nodes of the reduction.
  - Placed by: "`place` consumes it and adds the nodes of the reduction." design/unequal-shards-change-request.md:185
- **6 - Stage** (design/unequal-shards-change-request.md:246-248): stated under no stage named, which its clause names
  with this stage.
- **8 - Data structure** (design/unequal-shards-change-request.md:294-295)
  > Striped Attention §2.2.1's stripes: device r of N holds the tokens whose index in the original sequence is r
  > modulo N.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:301
- **8 - Algorithm (a): tokens permuted before the first embedding** (design/unequal-shards-change-request.md:296-297)
  > Where attention under a causal mask is sharded along the sequence, the input tokens are permuted before the first
  > embedding into the stripes.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:301; "Where an item's quotations state where its step sits in
    the program, its Stage line follows them: ... item 8's Striped Attention passages ("before the model's first
    embedding layer", "no extra per-layer communication required")." design/unequal-shards-change-request.md:60-62
- **8 - Algorithm (b): the layout kept forward** (design/unequal-shards-change-request.md:297-298)
  > The sequence keeps that layout through every function of the program, forward and backward.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes. `place` adds no communication at an attention to bring the sequence into or out of this layout."
    design/unequal-shards-change-request.md:301-302
- **8 - Algorithm (b): the layout kept backward** (design/unequal-shards-change-request.md:297-298)
  > The sequence keeps that layout through every function of the program, forward and backward.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes. `place` adds no communication at an attention to bring the sequence into or out of this layout."
    design/unequal-shards-change-request.md:301-302
- **8 - Algorithm (c): position ids and target ids** (design/unequal-shards-change-request.md:298-299)
  > Position ids, and the target ids of a loss, are permuted the same way.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:301
- **8 - Algorithm (d): the mask in original order** (design/unequal-shards-change-request.md:299-299)
  > Each attention's mask is causal in the tokens' original order.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:301
- **8 - Algorithm (e): a block the mask inhibits entirely** (design/unequal-shards-change-request.md:299-300)
  > A block of queries and keys that the mask inhibits entirely is not computed.
  - Placed by: "the stripes are the sharding of the sequence dimension in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:301
- **8 - Stage** (design/unequal-shards-change-request.md:301-302)
  > the stripes are the sharding of the sequence dimension in the placement operand, which `place` consumes. `place`
  > adds no communication at an attention to bring the sequence into or out of this layout.
  - Placed by: "the placement operand, which `place` consumes. `place` adds no communication at an attention"
    design/unequal-shards-change-request.md:301-302
- **9 - Data structure / Algorithm** (design/unequal-shards-change-request.md:314-316)
  > - **Data structure:** the `counts` and `displs` of item 2. - **Algorithm:** GSPMD §4.5's resharding by its four
  > operators: AllGather, AllToAll, CollectivePermute and DynamicSlice. Each of the four carries the counts of item 2
  > where its dimension is sharded by item 1.
  - Placed by: "the four are nodes `place` adds to the graph." design/unequal-shards-change-request.md:317
- **9 - Stage** (design/unequal-shards-change-request.md:317-317)
  > the four are nodes `place` adds to the graph.
  - Placed by: "the four are nodes `place` adds to the graph." design/unequal-shards-change-request.md:317
- **10 - Data structure** (design/unequal-shards-change-request.md:325-325)
  > a halo size for each device, and their maximum.
  - Placed by: "the collective permute's sends and receives and the slice are nodes `place` adds to the graph."
    design/unequal-shards-change-request.md:328; "Items 2, 3, 4, 9, 10 and 12 state nodes that `place` adds to the
    graph." design/unequal-shards-change-request.md:55-56
- **10 - Algorithm / Stage** (design/unequal-shards-change-request.md:326-328)
  > - **Algorithm:** GSPMD §4.3's halo exchange. Halos are exchanged by a collective permute at the largest halo over
  > the devices, and each device slices off what exceeds its own. - **Stage:** the collective permute's sends and
  > receives and the slice are nodes `place` adds to the graph.
  - Placed by: "the collective permute's sends and receives and the slice are nodes `place` adds to the graph."
    design/unequal-shards-change-request.md:328
- **12 - Data structure** (design/unequal-shards-change-request.md:373-376)
  > MPI's subarray, three arrays for each device: `array_of_sizes`, `array_of_subsizes` and `array_of_starts`. A
  > device's buffer of item 4 is the full array and its shard of item 1 is a subarray of it: the buffer's size along
  > the sharded dimension is at least the number of elements the device owns, as a local array's leading dimension is
  > at least LOCr.
  - Placed by: "`place` adds the collective's `send` and `recv` nodes (item 2); for the subarray these carry the three
    arrays, each an array of integers with one entry per dimension of the buffer."
    design/unequal-shards-change-request.md:386-387
- **12 - Algorithm (subarray sent)** (design/unequal-shards-change-request.md:377-379)
  > a collective of item 2 is given, for each device, the sizes of the full array, the sizes of the subarray and where
  > it starts, and sends the subarray: `counts[j]` is the subarray's number of elements. The padding is not sent.
  - Placed by: "`place` adds the collective's `send` and `recv` nodes (item 2); for the subarray these carry the three
    arrays" design/unequal-shards-change-request.md:386-387
- **12 - Stage** (design/unequal-shards-change-request.md:386-388)
  > `place` adds the collective's `send` and `recv` nodes (item 2); for the subarray these carry the three arrays,
  > each an array of integers with one entry per dimension of the buffer. No quoted source states which function
  > chooses the method, or the type of the value that records the choice.
  - Placed by: "`place` adds the collective's `send` and `recv` nodes (item 2)"
    design/unequal-shards-change-request.md:386
- **13 - Data structure** (design/unequal-shards-change-request.md:440-443)
  > one array of P+1 separator indices for P devices, s_0 = 0 ≤ s_1 ≤ ... ≤ s_P = N, the same at every device (PETSc's
  > ownership ranges; the partition Π of Pınar, Tabak and Aykanat). The rows are held in consecutive ranges, the first
  > range by device 0, the next by device 1, and so on. A device holds the values of its rows, concatenated, and the
  > length of each of its rows (`row_lengths`).
  - Placed by: "They are the sharding of the row dimension in the placement operand, which `place` consumes."
    design/unequal-shards-change-request.md:452; "Items 1, 5, 8, 13 and 15 state what the operand carries for a
    sharded dimension." design/unequal-shards-change-request.md:54-55
- **13 - Stage** (design/unequal-shards-change-request.md:451-452): stated under no stage named, which its clause
  names with this stage.
- **14 - Stage** (design/unequal-shards-change-request.md:473-473): stated under no stage named, which its clause
  names with this stage.
- **15 - Algorithm** (design/unequal-shards-change-request.md:487-489)
  > `torch.chunk`'s sizes. A sharded dimension with no tuple of item 1 has them: every device the same size except the
  > last, and trailing devices empty where the extent does not allow more. Item 1's rounding applies only where a
  > tuple is given.
  - Placed by: "`place` writes these sizes where the placement operand carries no tuple for a sharded dimension"
    design/unequal-shards-change-request.md:490
- **15 - Stage** (design/unequal-shards-change-request.md:490-491)
  > `place` writes these sizes where the placement operand carries no tuple for a sharded dimension; they are the
  > `counts` of items 2 and 3.
  - Placed by: "`place` writes these sizes where the placement operand carries no tuple for a sharded dimension"
    design/unequal-shards-change-request.md:490

### place: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **1 / Write: the operator lowering produces them on every backend**
  (design/torch-source-lowering-change-request.md:166-167): stated under no stage named, which its clause names with
  this stage.
- **2 / Write: the operator lowerings, then placement, consume it**
  (design/torch-source-lowering-change-request.md:191-192): stated under no stage named, which its clause names with
  this stage.
- **3 / Write: each function's processor** (design/torch-source-lowering-change-request.md:244-244)
  > Each function's processor is the placement operand together with the sharding.
  - Placed by: "**placement** (`placement.py`, `place`): the exported graph and the placement operand to the placed
    graph, which has each node's processor" design/torch-source-lowering-change-request.md:121-122; "**A placement
    also shards.** The placement operand gives a tensor's sharding over machines." design/the-compiler.md:46
- **5 / Write: it is an operator lowering** (design/torch-source-lowering-change-request.md:295-296): stated under no
  stage named, which its clause names with this stage.
- **6 / Write: nodes of the exported graph on every processor**
  (design/torch-source-lowering-change-request.md:388-392): stated under no stage named, which its clause names with
  this stage.

### place: [page-queue-change-request.md](page-queue-change-request.md)

- **Claude's proposal: The changes, by file: The callers: tools/mesh/grid.py**
  (design/page-queue-change-request.md:260-261): stated under caller, which its clause names with this stage.

## on_devices

`on_devices`: processors onto devices and machines.

### on_devices: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **A2 data structure, part 2 (mapping onto the map's nodes)** (design/aot-compilation-change-request.md:170-173)
  > the mapping of the mesh's devices onto the map's nodes (rdma/mesh-map.h). The logical mesh and its mapping onto
  > physical devices are Alpa's (L. Zheng et al., Alpa: automating inter- and intra-operator parallelism for
  > distributed deep learning, OSDI 2022, §4.1): "For a set of physical devices, there can be multiple logical views",
  > and the mapping is chosen by a pass of its own.
  - Placed by: "Devices takes that graph and the mapping onto the map's nodes."
    (design/aot-compilation-change-request.md:178-179); "**Stage:** placement, then devices."
    (design/aot-compilation-change-request.md:166)
- **A2 produced by, consumed by** (design/aot-compilation-change-request.md:177-181): stated under place, which its
  clause names with this stage.

### on_devices: [unequal-shards-change-request.md](unequal-shards-change-request.md)

- **2 - Stage** (design/unequal-shards-change-request.md:130-132): stated under place, which its clause names with
  this stage.

## schedule

`schedule`: the one solve that orders functions and transfers with cost as an operand, and what is decided with it at compile time: storage, routes, pages, lifetimes, bounds.

### schedule: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **A2 produced by, consumed by** (design/aot-compilation-change-request.md:177-181): stated under place, which its
  clause names with this stage.
- **A3 data structure** (design/aot-compilation-change-request.md:195-197)
  > **The data structure is the weights of the task graph:** a weight w on each task and a weight c on each edge (O.
  > Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007; Y.-K. Kwok and I. Ahmad, ACM Comput. Surv., 1999), and
  > the cables' capacities, from the map.
  - Placed by: "**Stage:** the cost operand of the schedule (A6), and the capacities of the routing (A4)."
    (design/aot-compilation-change-request.md:194)
- **A3 no algorithm computes them** (design/aot-compilation-change-request.md:198-201)
  > **No algorithm computes them here.** The compilation takes every weight its router and its scheduler use as an
  > operand. The task graph's weights w and c are given inputs of the scheduling problem (Sinnen; Kwok and Ahmad). The
  > router's capacities and demands are the given inputs of concurrent multicommodity flow (N. Garg and J. Könemann,
  > Faster and simpler algorithms for multicommodity flow and other fractional packing problems, FOCS 1998).
  - Placed by: "**Stage:** the cost operand of the schedule (A6), and the capacities of the routing (A4)."
    (design/aot-compilation-change-request.md:194)
- **A3 produced by, consumed by (router and scheduler)** (design/aot-compilation-change-request.md:202-203)
  > A4's routing takes the capacities, and A6's schedule takes w and c.
  - Placed by: The clause's own "A6's schedule takes w and c" (design/aot-compilation-change-request.md:202-203);
    "**Stage:** the cost operand of the schedule (A6), and the capacities of the routing (A4)."
    (design/aot-compilation-change-request.md:194)
- **A3 produced by, consumed by (the caller passes the operand; mesh's type)**
  (design/aot-compilation-change-request.md:202-204): stated under caller, which its clause names with this stage.
- **A4 stage** (design/aot-compilation-change-request.md:219-221)
  > **Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's function, routes
  > them and lowers them to sends and receives of pages (rdma/mesh-collective.c). The reference program's `place` adds
  > its send and recv nodes at placement and routes nothing.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:219)
- **A4 data structure: the instruction DAG** (design/aot-compilation-change-request.md:223-225)
  > **The instruction DAG.** MSCCLang (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS
  > 2023, §4 and §5) lowers operations on chunks into an instruction DAG of sends, receives and receive-reduce-copies.
  > A relay's forward is its fused receive-copy-send. Here a chunk is a page.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:219); "`mesh_plan` takes them, the map
    and A3's capacities, and produces the instruction DAG with each instruction's route."
    (design/aot-compilation-change-request.md:249-250)
- **A4 data structure and algorithm: packed spanning trees, packing**
  (design/aot-compilation-change-request.md:226-231)
  > **Packed spanning trees,** the routes. Blink (G. Wang et al., Blink: fast and generic collectives for distributed
  > ML, MLSys 2020) builds collectives on them. §3.1: a broadcast's optimal rate is reached "by finding the maximal
  > packing of a number of directed spanning trees or arborescences in the graph". - **The algorithms:** -
  > **Packing.** Blink §3.2: the packing is a multiplicative-weight update for fractional packing. `mesh_graph_pack`'s
  > Garg and Könemann rounds are a multiplicative-weight method of the same kind.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:219); the clause's own
    "`mesh_graph_pack`'s Garg and Könemann rounds" (design/aot-compilation-change-request.md:230-231)
- **A4 algorithm: joint routing** (design/aot-compilation-change-request.md:216-217, 232-234)
  > > routing within tb5 linked collectives is supported; attempting to involve nodes with no tb5 dma connection to
  > any node > within the collective is not allowed. [...] **Joint routing.** All of one function's collectives are
  > routed jointly over the Thunderbolt-linked nodes of the map, as concurrent multicommodity flow, the problem Garg
  > and Könemann's rounds solve (N. Garg and J. Könemann, FOCS 1998).
  - Placed by: "**Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's
    function, routes them and lowers them to sends and receives of pages (rdma/mesh-collective.c)."
    (design/aot-compilation-change-request.md:219-220)
- **A4 algorithm: AllReduce** (design/aot-compilation-change-request.md:235-237)
  > **AllReduce.** Blink §3.3: "to do an AllReduce operation on the directed graph, we first run a reduce operation to
  > a chosen root vertex using the undirected graph and then do a broadcast operation from the root vertex using the
  > same tree but with links going in the reverse direction."
  - Placed by: "**Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's
    function, routes them and lowers them to sends and receives of pages (rdma/mesh-collective.c)."
    (design/aot-compilation-change-request.md:219-220); design/the-compiler.md:46-47 has "A collective that the
    sharding requires is `send` and `recv` nodes that `place` adds."
- **A4 algorithm: AllGather** (design/aot-compilation-change-request.md:238-240)
  > **AllGather** is one broadcast from each source over its packed trees: §3.1's broadcast, once for each source, as
  > rdma/mesh_compile.py `_all_gather` declares it. That composition is this document's; Blink's own §4.1 states
  > "AllGather is AllReduce without using a reduction function".
  - Placed by: "**Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's
    function, routes them and lowers them to sends and receives of pages (rdma/mesh-collective.c)."
    (design/aot-compilation-change-request.md:219-220); design/the-compiler.md:46-47 has "A collective that the
    sharding requires is `send` and `recv` nodes that `place` adds."
- **A4 algorithm: ReduceScatter** (design/aot-compilation-change-request.md:241-244)
  > **ReduceScatter** "is semantically equivalent to an AllReduce followed by a DynamicSlice where each partition gets
  > one slice of the fully reduced data. An efficient implementation has only half the cost of AllReduce" (GSPMD
  > §4.2). Here each slice is reduced to the member that keeps it, over its packed trees: the first half of Blink
  > §3.3's AllReduce, with that member as the root. That reading of GSPMD's half is this document's.
  - Placed by: "**Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's
    function, routes them and lowers them to sends and receives of pages (rdma/mesh-collective.c)."
    (design/aot-compilation-change-request.md:219-220); design/the-compiler.md:46-47 has "A collective that the
    sharding requires is `send` and `recv` nodes that `place` adds."
- **A4 algorithm: CollectivePermute** (design/aot-compilation-change-request.md:245-246)
  > **CollectivePermute** "exchanges data among a list of source-destination pairs" (GSPMD §4.2): one point-to-point
  > demand a pair, routed as AllToAll's are.
  - Placed by: "**Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's
    function, routes them and lowers them to sends and receives of pages (rdma/mesh-collective.c)."
    (design/aot-compilation-change-request.md:219-220); design/the-compiler.md:46-47 has "A collective that the
    sharding requires is `send` and `recv` nodes that `place` adds."
- **A4 algorithm: AllToAll** (design/aot-compilation-change-request.md:247-248)
  > **AllToAll** is point-to-point demands, one per ordered pair of ranks, routed jointly as concurrent multicommodity
  > flow.
  - Placed by: "**Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's
    function, routes them and lowers them to sends and receives of pages (rdma/mesh-collective.c)."
    (design/aot-compilation-change-request.md:219-220); design/the-compiler.md:46-47 has "A collective that the
    sharding requires is `send` and `recv` nodes that `place` adds."
- **A4 produced by, consumed by (mesh_plan's types)** (design/aot-compilation-change-request.md:252-254)
  > `mesh_plan` takes `struct mesh_function`, whose `declared` are the collectives' `struct mesh_declared`
  > (rdma/mesh-collective.h). It writes the instructions as `struct plan_entry` (rdma/mesh-collective.c) and from them
  > the `struct mesh_hop` lists (rdma/mesh.h).
  - Placed by: "`mesh_plan`, the schedule stage's function" (design/aot-compilation-change-request.md:219)
- **A4 which nodes hold a function's static schedule** (design/aot-compilation-change-request.md:264-266)
  > **Which nodes hold a function's static schedule.** Each compiled function has its own static schedule (appendix
  > item 6). Every node of the map is a node of the program, since every run is launched on the whole collective, so
  > every relay holds the static schedule (A7, sender(m)).
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:219); "every node of every static
    schedule, relays among them, calls `mesh_plan` and holds the static schedule."
    (design/aot-compilation-change-request.md:380-381)
- **A4 a node to which the placement operand maps no device (mesh_plan's part)**
  (design/aot-compilation-change-request.md:267-269)
  > **A node to which the placement operand maps no device** (A2) compiles and schedules each function as every node
  > does, with no section of its own. It declares the function's transfers as every rank does, and its part of each
  > call is the relaying its routes give it.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:219); the clause's own "compiles and
    schedules each function as every node does" (design/aot-compilation-change-request.md:267-268)
- **A5 data structure and algorithm** (design/aot-compilation-change-request.md:283-291)
  > **The data structure is the page table:** names bound to frames (T. Kilburn, D. B. G. Edwards, M. J. Lanigan and
  > F. H. Sumner, One-level storage system, IRE Trans. Electronic Computers, 1962; P. J. Denning, Virtual memory, ACM
  > Comput. Surv., 1970), as design/plan-to-bridge-change-request.md R2 states it. - **The algorithm is the
  > allocator's, as R2 states it:** the u copies of appendix item 1 (Lam, 1988); the order in which two items may
  > share a page, Lamport's happens-before (L. Lamport, Time, clocks, and the ordering of events in a distributed
  > system, CACM 21(7), 1978); and the fewest pages as the fewest chains of that order, by Dilworth's theorem, found
  > by maximum bipartite matching (R. P. Dilworth, A decomposition theorem for partially ordered sets, Annals of
  > Mathematics 51(1), 1950; D. R. Fulkerson, Note on Dilworth's decomposition theorem for partially ordered sets,
  > Proc. AMS 7(4), 1956).
  - Placed by: "**Stage:** schedule. It writes the page table's bindings into the scheduled value (A9)."
    (design/aot-compilation-change-request.md:282)
- **A5 produced by, consumed by** (design/aot-compilation-change-request.md:282, 292-293)
  > **Stage:** schedule. It writes the page table's bindings into the scheduled value (A9). [...] **Produced by,
  > consumed by.** The allocator takes the values of A8's IR. The bridge and the GPU's program take the bindings from
  > the scheduled value (A9).
  - Placed by: "**Stage:** schedule. It writes the page table's bindings into the scheduled value (A9)."
    (design/aot-compilation-change-request.md:282)
- **A6 data structure and algorithm** (design/aot-compilation-change-request.md:299-305)
  > **The data structure is the task graph** G = (V, E, w, c) of design/plan-to-bridge-change-request.md R3 (O.
  > Sinnen, 2007; Y.-K. Kwok and I. Ahmad, 1999), with R4's consumers' kinds as its shape and A3's weights. - **The
  > algorithm is design/plan-to-bridge-change-request.md R5's:** R5's heap order by the completion times of one linear
  > program (C.-Y. Chen, Scheduling coflows with precedence constraints for minimizing the total weighted completion
  > time in identical parallel networks, arXiv 2205.02474, 2022; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein,
  > Scheduling to minimize average completion time: off-line and on-line approximation algorithms, Math. Oper. Res.
  > 22(3), 1997), solved by HiGHS.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:298)
- **A6 produced by, consumed by (the lists)** (design/aot-compilation-change-request.md:306-307)
  > It writes, for each adjacent node and direction, the ordered list of page instructions of A9. The bridge takes
  > those lists. Each entry is a `struct mesh_hop`.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:298); "The schedule takes A8's IR. It
    writes" (design/aot-compilation-change-request.md:306)
- **A6 one schedule per function; the schedule takes A8's IR** (design/aot-compilation-change-request.md:94-95, 296,
  306)
  > **Schedule** (schedule.py `schedule`): the placed graph and the cost operand in; each processor's operations in
  > scheduled order out. There is one schedule. [...] ### A6. One schedule per function [...] **Produced by, consumed
  > by.** The schedule takes A8's IR.
  - Placed by: "**Stage:** schedule." (design/aot-compilation-change-request.md:298)
- **A7 sender(m) (the bridge's and mesh_plan's part)** (design/aot-compilation-change-request.md:372-374): stated
  under Program, which its clause names with this stage.
- **A7 sender(m) (the launcher passes the program's nodes)** (design/aot-compilation-change-request.md:373-381):
  stated under caller, which its clause names with this stage.
- **A7 views: a static schedule is used only within its view** (design/aot-compilation-change-request.md:467-470):
  stated under Program, which its clause names with this stage.
- **A7 views: every member computes its static schedules on the view's map**
  (design/aot-compilation-change-request.md:471-478): stated under Program, which its clause names with this stage.
- **A8 data structure (the type mesh_plan takes)** (design/aot-compilation-change-request.md:502-512): stated under
  scheduled value, which its clause names with this stage.
- **A8 carried as one IR from partitioning to serialization** (design/aot-compilation-change-request.md:502-503,
  510-512): stated under scheduled value, which its clause names with this stage.
- **A9 stage and types** (design/aot-compilation-change-request.md:520-523): stated under scheduled value, which its
  clause names with this stage.
- **A9 the algorithm the bridge runs it by: the self-timed strategy** (design/aot-compilation-change-request.md:538,
  546-565): stated under bridge, which its clause names with this stage.
- **A9 every address is fixed at compile time** (design/aot-compilation-change-request.md:569-574): stated under
  scheduled value, which its clause names with this stage.
- **A10 algorithms: independence, commuting operations, the bound**
  (design/aot-compilation-change-request.md:607-617): stated under no stage named, which its clause names with this
  stage.
- **A10 its output (the receiving types)** (design/aot-compilation-change-request.md:629-634, 638-639): stated under
  scheduled value, which its clause names with this stage.
- **A10 produced by, consumed by (the schedule)** (design/aot-compilation-change-request.md:635, 637-638)
  > **Produced by, consumed by.** The schedule takes the bound as a delay of its IPC graph (appendix item 2). [...]
  > The schedule writes the IPC graph's delay edges as `struct mesh_sync` (rdma/mesh.h), which the program's
  > submission reads.
  - Placed by: The clause's own "The schedule takes the bound as a delay of its IPC graph (appendix item 2)."
    (design/aot-compilation-change-request.md:635)
- **Appendix: a chooser for edge weights** (design/aot-compilation-change-request.md:651-652, 659-664): stated under
  caller, which its clause names with this stage.
- **Appendix item 1 (unrolling)** (design/aot-compilation-change-request.md:685, 689)
  > **Produced by, consumed by.** A5's allocator takes u. Item 2's IPC graph takes it as each transfer's delay. [...]
  > **The change:** compute u as C3.3 states. D5's lag-1 condition is deleted.
  - Placed by: "**Stage:** schedule. `mesh_plan` computes u (item 2)." (design/aot-compilation-change-request.md:679)
- **Appendix item 2: data structures and algorithms (the queue, its size, the merge)**
  (design/aot-compilation-change-request.md:707-716): stated under Program, which its clause names with this stage.
- **Appendix item 2: the queue is laid out with the schedule's pages and read in place**
  (design/aot-compilation-change-request.md:760-763): stated under bridge, which its clause names with this stage.
- **Appendix item 2: which files define them** (design/aot-compilation-change-request.md:829-833): stated under graph
  module, which its clause names with this stage.
- **Appendix item 2, change: the queues replace the sequence counters, taken[p] and the one channel**
  (design/aot-compilation-change-request.md:850-854): stated under Program, which its clause names with this stage.
- **Appendix item 4 (no truncation at one end; no handler)** (design/aot-compilation-change-request.md:882-884, 892):
  stated under scheduled value, which its clause names with this stage.
- **Appendix item 5: the set** (design/aot-compilation-change-request.md:908-912, 927-929)
  > **The data structure is the set:** the pages of asynchronous, incomplete, noncontiguous consumers' operands
  > (design/plan-to-bridge-change-request.md R4: one task per page, any order) are written as sets, not as a sequence.
  > A set is one demand's pages on one of its trees, as the code numbers them (rdma/mesh-collective.c: tree j's
  > entries have `set` j + 1), so a set's pages have one producer and one route. The bridge sees sets and sequences,
  > and no kind (R4). [...] **Produced by, consumed by.** rdma/mesh-collective.c writes each entry's `set` into its
  > `struct plan_entry` and copies it from there into the entry's `struct mesh_hop` (A9). The bridge at each end of a
  > cable takes the sets from the `struct mesh_hop` lists and draws the same permutation.
  - Placed by: "**Stage:** the schedule writes sets into the scheduled value (A9). The runtime's bridge draws each
    set's order at each execution." (design/aot-compilation-change-request.md:906-907)
- **Appendix item 5: joins are sent in the schedule's order** (design/aot-compilation-change-request.md:990-990)
  > **A synchronous, complete, contiguous consumer's operands** (joins) are sent in the schedule's order.
  - Placed by: The clause's own "are sent in the schedule's order" (design/aot-compilation-change-request.md:990);
    "**Stage:** the schedule writes sets into the scheduled value (A9)."
    (design/aot-compilation-change-request.md:906)
- **Appendix item 6: the compiled function's name; alike at every rank**
  (design/aot-compilation-change-request.md:1006-1007, 1026): stated under scheduled value, which its clause names
  with this stage.
- **Appendix item 6, change: any number of compiled functions at once (mesh_plan)**
  (design/aot-compilation-change-request.md:1021-1021): stated under scheduled value, which its clause names with this
  stage.
- **Appendix item 6, change: one allocation over the whole program; deferred reclamation**
  (design/aot-compilation-change-request.md:1022-1024)
  > place each static schedule with every other of the program by the one allocation over the whole exported program
  > (A1; disjoint-pages C3.2 and C3.4). Nothing takes pages at run time. A function's pages are released when the
  > program retires it or itself, by deferred reclamation, as R8 cites;
  - Placed by: "**Stage:** schedule. It writes the page table's bindings into the scheduled value (A9)."
    (design/aot-compilation-change-request.md:282) with "**The algorithm is the allocator's, as R2 states it:**"
    (design/aot-compilation-change-request.md:286); the item's own stage line is "**Stage:** the scheduled value and
    the runtime." (design/aot-compilation-change-request.md:1004)
- **The files and functions: rdma/mesh-collective.c (fed by A2's collectives)**
  (design/aot-compilation-change-request.md:1074-1076)
  > **rdma/mesh-collective.c:** - the demands and trees are fed by A2's collectives, in place of ranges declared by
  > hand; `mesh_plan` routes every demand of one static schedule at once, by `mesh_graph_flow` (A4);
  - Placed by: "`mesh_plan`, the schedule stage's function, routes them and lowers them to sends and receives of pages
    (rdma/mesh-collective.c)" (design/aot-compilation-change-request.md:219-220); "**Produced by, consumed by.** A2's
    partitioning produces the collectives, in A8's IR. `mesh_plan` takes them"
    (design/aot-compilation-change-request.md:249-250)
- **The files and functions: rdma/mesh-collective.c (routing, u, queue sizes, sets)**
  (design/aot-compilation-change-request.md:1074-1078)
  > **rdma/mesh-collective.c:** - the demands and trees are fed by A2's collectives, in place of ranges declared by
  > hand; `mesh_plan` routes every demand of one static schedule at once, by `mesh_graph_flow` (A4); - `mesh_plan`
  > computes u as C3.3 states (item 1) and each queue's size (item 2); - tree j's entries have `set` j + 1 (item 5).
  - Placed by: "`mesh_plan`, the schedule stage's function, routes them and lowers them to sends and receives of pages
    (rdma/mesh-collective.c)" (design/aot-compilation-change-request.md:219-220); "**Stage:** schedule. `mesh_plan`
    computes u (item 2)." (design/aot-compilation-change-request.md:679); "The schedule stage (`mesh_plan`) computes
    each queue's size." (design/aot-compilation-change-request.md:705-706)

### schedule: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **The data flow (stage table and its closing sentence)** (design/compilation-composition-change-request.md:35-36):
  stated under scheduled value, which its clause names with this stage.
- **2 / Composes: one schedule orders functions and transfers together**
  (design/compilation-composition-change-request.md:132-134)
  > One schedule orders functions and transfers together and produces the scheduled value. Lowering, the runtime and
  > the archive take the scheduled value.
  - Placed by: "One schedule orders functions and transfers together and produces the scheduled value."
    design/compilation-composition-change-request.md:132-133
- **2 / Write: transformations keep information; outputs are operands; changes reflected**
  (design/compilation-composition-change-request.md:136-139): stated under scheduled value, which its clause names
  with this stage.
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **3 / Write, second bullet: callers call them, own copies deleted**
  (design/compilation-composition-change-request.md:150-157): stated under graph module, which its clause names with
  this stage.
- **3 / Write, second bullet: the four named deletions** (design/compilation-composition-change-request.md:155-157)
  > Deleted, as the existing requests name them: the second scheduling solve, copied graph walks, placement of program
  > values at run time and alternative execution paths.
  - Placed by: "**Composes:** placement and the one schedule call the graph functions; the schedule's result is in the
    scheduled value, which lowering and the runtime read." design/compilation-composition-change-request.md:150-151
- **4 / Write: placement and costs are operands** (design/compilation-composition-change-request.md:170-173): stated
  under caller, which its clause names with this stage.
- **4 / Write: compilation realizes transfers, ordering and synchronization**
  (design/compilation-composition-change-request.md:172-173)
  > **Write:** placement and costs arrive as operands; compilation realizes the graph's storage, transfers, ordering
  > and synchronization.
  - Placed by: "each processor's order of its functions and transfers, which the one schedule computes"
    design/compilation-composition-change-request.md:166
- **4 / Write: compilation realizes storage** (design/compilation-composition-change-request.md:172-173)
  > **Write:** placement and costs arrive as operands; compilation realizes the graph's storage, transfers, ordering
  > and synchronization.
  - Placed by: "| [disjoint-pages](disjoint-pages-change-request.md), C3 with later corrections | Value instances,
    lifetimes and page assignments, including modulo variable expansion. | schedule → runtime |"
    design/compilation-composition-change-request.md:96
- **5 / Replaces, site 3: strided-view traversal** (design/compilation-composition-change-request.md:212-213)
  > `mesh-collective.c plan_extents`: the strided-view traversal has one definition, and every consumer that needs it
  > calls that.
  - Placed by: "`mesh-collective.c plan_extents`" design/compilation-composition-change-request.md:212
- **5 / Replaces, site 3: signature schema and printers** (design/compilation-composition-change-request.md:213-214):
  stated under graph module, which its clause names with this stage.
- **5 / Write: one definition of each repeated behavior** (design/compilation-composition-change-request.md:215-217):
  stated under lower, which its clause names with this stage.
- **6 / Replaces 2: schedule results are operands of native preparation and of the transport**
  (design/compilation-composition-change-request.md:250-252): stated under scheduled value, which its clause names
  with this stage.
- **6 / Replaces 3** (design/compilation-composition-change-request.md:253-255): stated under lower, which its clause
  names with this stage.
- **6 / Replaces 4** (design/compilation-composition-change-request.md:256-258)
  > Typed/strided/affine accesses are present in the frontend and masked-pass analysis. Scheduling consumers instead
  > repeatedly manufacture whole-buffer conflicts. Carry the actual access/effect representation into the consumers
  > that need it; a newly introduced instruction adds its actual effects to that representation.
  - Placed by: "Scheduling consumers instead repeatedly manufacture whole-buffer conflicts."
    design/compilation-composition-change-request.md:256-257
- **Correction: generated target source / Composes: both frontends at export**
  (design/compilation-composition-change-request.md:302-303): stated under export, which its clause names with this
  stage.
- **Correction: generated target source / Write: an edge carries its cause**
  (design/compilation-composition-change-request.md:312-314): stated under scheduled value, which its clause names
  with this stage.
- **Performance correction / Replaces** (design/compilation-composition-change-request.md:357-359): stated under
  lower, which its clause names with this stage.
- **Performance correction / Write: publication and acquisition**
  (design/compilation-composition-change-request.md:363-364): stated under lower, which its clause names with this
  stage.
- **Complete replacement / operand table row 3: Values and effects**
  (design/compilation-composition-change-request.md:427-427)
  > | Values and effects | The same value identities → views, aliases, ranges/pages, reads/writes and
  > generation/lifetime facts → allocation, collective/transport scheduling, masking and publication. Bind model
  > values to kernel operands to placed storage explicitly. |
  - Placed by: "→ allocation, collective/transport scheduling, masking and publication."
    design/compilation-composition-change-request.md:427
- **Complete replacement / operand table row 4: Program relations**
  (design/compilation-composition-change-request.md:428-428): stated under scheduled value, which its clause names
  with this stage.
- **Complete replacement / replaced-implementation table row 5**
  (design/compilation-composition-change-request.md:455-455): stated under lower, which its clause names with this
  stage.
- **Source bindings / Compact effects: the mesh-graph function**
  (design/compilation-composition-change-request.md:468-473): stated under graph module, which its clause names with
  this stage.
- **Source bindings / Compact effects: producer and consumers**
  (design/compilation-composition-change-request.md:471-472)
  > **Composes:** the compiler produces the access relations; a function of `mesh-graph` computes their exact integer
  > feasibility and returns dependence causes; scheduling, fusion and lifetime consumers take that relation.
  - Placed by: "scheduling, fusion and lifetime consumers take that relation."
    design/compilation-composition-change-request.md:472
- **7 / Composes: the archive stores the scheduled value** (design/compilation-composition-change-request.md:505-507):
  stated under save and load, which its clause names with this stage.
- **7 / Write** (design/compilation-composition-change-request.md:512-516): stated under lower, which its clause names
  with this stage.
- **7 / table row 2: partitioned computation, transfers, compiled order**
  (design/compilation-composition-change-request.md:521-521): stated under lower, which its clause names with this
  stage.

### schedule: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **A3 item 1 (Items)** (design/disjoint-pages-change-request.md:448-450)
  > **Items.** One execution's graph is unrolled into u copies (C3.3). An item is one page of one copy of one value. A
  > value of several pages is several items, each placed on any page, not as a run
  > (design/plan-to-bridge-change-request.md R2).
  - Placed by: "**Which function computes it, and which read it.** `mesh_plan` calls it once per static schedule,
    before the first execution." (design/disjoint-pages-change-request.md:469-470); "- **Schedule.** `mesh_plan` (mesh
    `rdma/mesh-collective.c`) takes the program's values and its dependence graph and writes the static schedule,
    which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **A3 item 2 (The order)** (design/disjoint-pages-change-request.md:451-453)
  > **The order.** Item a precedes item b when the dependence graph orders every read of a before the first write of
  > b: happens-before, not time (L. Lamport, Time, clocks, and the ordering of events in a distributed system, CACM
  > 21(7), 1978). It is tested at the distances at which the copies' executions meet. The relation is a partial order.
  - Placed by: "**Which function computes it, and which read it.** `mesh_plan` calls it once per static schedule,
    before the first execution." (design/disjoint-pages-change-request.md:469-470); "- **Schedule.** `mesh_plan` (mesh
    `rdma/mesh-collective.c`) takes the program's values and its dependence graph and writes the static schedule,
    which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **A3 item 3 (The chains)** (design/disjoint-pages-change-request.md:454-458)
  > **The chains.** The items are split into the fewest chains, as many as the largest set of mutually unordered
  > items. Each chain is one page. Consecutive items of a chain share that page with no wait, because the dependence
  > graph already orders the earlier item's last read before the later item's first write. This is how CUDA graph
  > memory nodes reuse physical memory: only where the graph's topology guarantees the lifetimes don't overlap
  > (NVIDIA, CUDA C++ Programming Guide, "Graph Memory Nodes").
  - Placed by: "**Which function computes it, and which read it.** `mesh_plan` calls it once per static schedule,
    before the first execution." (design/disjoint-pages-change-request.md:469-470); "- **Schedule.** `mesh_plan` (mesh
    `rdma/mesh-collective.c`) takes the program's values and its dependence graph and writes the static schedule,
    which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **A3 item 4 (Operands typed as one run)** (design/disjoint-pages-change-request.md:459-460)
  > **Operands typed as one run.** A consumer typed to see one run of pages is given it as a mapping written when the
  > static schedule is made (design/plan-to-bridge-change-request.md R4).
  - Placed by: "a mapping written when the static schedule is made" (design/disjoint-pages-change-request.md:459-460);
    "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`) takes the program's values and its dependence graph
    and writes the static schedule, which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **A3 "What it takes"** (design/disjoint-pages-change-request.md:462-465)
  > **What it takes.** - Every value one execution moves or computes, with its size in pages. - The dependence graph:
  > each value's producer and its consumers, the program's own data edges, and the transport's edges (a SEND before
  > the completion of its receive, a receive's completion before its forward).
  - Placed by: "**Which function computes it, and which read it.** `mesh_plan` calls it once per static schedule,
    before the first execution." (design/disjoint-pages-change-request.md:469-470); "it takes `struct mesh_value` and
    `struct mesh_dependence`" (design/disjoint-pages-change-request.md:473); "- **Schedule.** `mesh_plan` (mesh
    `rdma/mesh-collective.c`) takes the program's values and its dependence graph and writes the static schedule,
    which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **A3 "What it writes"** (design/disjoint-pages-change-request.md:467-467)
  > **What it writes.** Every item's page, into the page table and the page lists, before the first execution.
  - Placed by: "**Which function computes it, and which read it.** `mesh_plan` calls it once per static schedule,
    before the first execution." (design/disjoint-pages-change-request.md:469-470); "and writes `struct
    mesh_page_entry` rows and the `page` of each `struct mesh_hop`" (design/disjoint-pages-change-request.md:473-474);
    "writes the static schedule, which is the scheduled value" (design/disjoint-pages-change-request.md:384)
- **A3 "Which function computes it, and which read it"** (design/disjoint-pages-change-request.md:469-474)
  > **Which function computes it, and which read it.** `mesh_plan` calls it once per static schedule, before the first
  > execution. It is the only allocator of a program's pages: nothing allocates a program page while the program runs,
  > not the bridge, not a call site, not a GPU kernel. The bridge (`rdma/mesh-flow.c`) reads the page lists, and the
  > recorded kernels (`metal_recording.m`) read the pages the page table names. The types are those of "The functions,
  > and the values between them": it takes `struct mesh_value` and `struct mesh_dependence`, and writes `struct
  > mesh_page_entry` rows and the `page` of each `struct mesh_hop`.
  - Placed by: "`mesh_plan` calls it once per static schedule, before the first execution."
    (design/disjoint-pages-change-request.md:469-470); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`)
    takes the program's values and its dependence graph and writes the static schedule, which is the scheduled value."
    (design/disjoint-pages-change-request.md:383-384)
- **A4 deleted (1)** (design/disjoint-pages-change-request.md:481-482)
  > `mesh_rows_alloc` and `mesh_arena_alloc` as call-site allocators: `mesh-call.c`'s three calls and
  > `mesh-records.c`'s call.
  - Placed by: "`mesh_plan` calls it once per static schedule, before the first execution. It is the only allocator of
    a program's pages: nothing allocates a program page while the program runs, not the bridge, not a call site, not a
    GPU kernel." (design/disjoint-pages-change-request.md:469-471); "They are called from `rdma/mesh-call.c` (a
    client's sections, lines 22, 32 and 62) and from `rdma/mesh-records.c` (the bridge's record pages, line 76)."
    (design/disjoint-pages-change-request.md:412-413)
- **A4 deleted (2)** (design/disjoint-pages-change-request.md:483-483)
  > `plan_pages`' `depth` copies and `plan_row`'s arithmetic on `slot`.
  - Placed by: "`rdma/mesh-collective.c` `plan_pages` lays each section out as `depth` copies of its pages, and
    `plan_row` gives copy `slot`'s page k as `first + slot × (pages / block) + k`."
    (design/disjoint-pages-change-request.md:414-415); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`)"
    (design/disjoint-pages-change-request.md:383)
- **A4 deleted (5)** (design/disjoint-pages-change-request.md:486-486)
  > `mesh_claim_take`'s first fit over a bitmap at each `mesh_plan` call.
  - Placed by: "at each `mesh_plan` call" (design/disjoint-pages-change-request.md:486); "- **Schedule.** `mesh_plan`
    (mesh `rdma/mesh-collective.c`) takes the program's values and its dependence graph and writes the static
    schedule, which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **C3.1 data structure** (design/disjoint-pages-change-request.md:807-809): stated under scheduled value, which its
  clause names with this stage.
- **C3.2 (a) data structure and types** (design/disjoint-pages-change-request.md:834-836)
  > **The data structure** is the page table, and **the algorithm** is A3's allocator. `mesh_plan` computes the
  > placement, and the bridge and the recorded kernels read it (A3). Each value is a `struct mesh_value`, and a value
  > placed once has `kind` `MESH_INVARIANT` (`rdma/mesh-collective.h`).
  - Placed by: "`mesh_plan` computes the placement, and the bridge and the recorded kernels read it (A3)."
    (design/disjoint-pages-change-request.md:834-835); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`)
    takes the program's values and its dependence graph and writes the static schedule, which is the scheduled value."
    (design/disjoint-pages-change-request.md:383-384)
- **C3.2 (b) what is placed** (design/disjoint-pages-change-request.md:837-838)
  > It places the weights, the KV cache, pooled intermediates, the values sent between nodes, the GPU's publication
  > records, the array of phases and the sent ranges' words (C3.7).
  - Placed by: "**C3.2 The allocator places every value.**" (design/disjoint-pages-change-request.md:833);
    "`mesh_plan` computes the placement" (design/disjoint-pages-change-request.md:834)
- **C3.2 (c) invariants placed once** (design/disjoint-pages-change-request.md:839-840)
  > A value written before the first execution and read by every execution (the weights; the KV cache as one array) is
  > placed once, for every execution, as Lam places loop invariants outside the kernel.
  - Placed by: "**C3.2 The allocator places every value.**" (design/disjoint-pages-change-request.md:833);
    "`mesh_plan` computes the placement" (design/disjoint-pages-change-request.md:834)
- **C3.2 (d) nothing laid out when created** (design/disjoint-pages-change-request.md:841-842)
  > Nothing is laid out when it is created, and no program writes a page to fault it in: the region is wired once
  > (data-model §2).
  - Placed by: "**C3.2 The allocator places every value.**" (design/disjoint-pages-change-request.md:833);
    "`mesh_plan` computes the placement" (design/disjoint-pages-change-request.md:834)
- **C3.3 algorithm 1 (L)** (design/disjoint-pages-change-request.md:847-848)
  > For each value, L is the smallest lag at which every read of it in execution t happens before its write in
  > execution t + L.
  - Placed by: "**Written by, read by.** `mesh_plan` computes u and the copies, and A3's allocator places the copies'
    pages." (design/disjoint-pages-change-request.md:852)
- **C3.3 algorithm 2 (u)** (design/disjoint-pages-change-request.md:849-850)
  > u is the largest L. Tensor-parallel decode gives u = 1; recurrent state updated in place, 2; 1F1B pipelines, up to
  > the number of stages. No cited source states what is written for a value with no such L.
  - Placed by: "**Written by, read by.** `mesh_plan` computes u and the copies, and A3's allocator places the copies'
    pages." (design/disjoint-pages-change-request.md:852)
- **C3.3 algorithm 3 (copies, t mod u)** (design/disjoint-pages-change-request.md:851-851)
  > One execution's graph is unrolled into u copies, and execution t runs copy t mod u.
  - Placed by: "**Written by, read by.** `mesh_plan` computes u and the copies, and A3's allocator places the copies'
    pages." (design/disjoint-pages-change-request.md:852); "For execution t, the bridge and the recorded kernels read
    copy t mod u's pages." (design/disjoint-pages-change-request.md:853-854)
- **C3.3 "Written by, read by"** (design/disjoint-pages-change-request.md:852-854)
  > **Written by, read by.** `mesh_plan` computes u and the copies, and A3's allocator places the copies' pages.
  > `executions` is not an input of `mesh_plan`: a static schedule runs any number of executions. For execution t, the
  > bridge and the recorded kernels read copy t mod u's pages. u is `struct mesh_state`'s `copies` (`rdma/mesh.h`).
  - Placed by: "`mesh_plan` computes u and the copies, and A3's allocator places the copies' pages."
    (design/disjoint-pages-change-request.md:852); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`) takes
    the program's values and its dependence graph and writes the static schedule, which is the scheduled value."
    (design/disjoint-pages-change-request.md:383-384)
- **C3.3 compile-time bound** (design/disjoint-pages-change-request.md:855-856)
  > It is computed at compile time, once per static schedule, over one execution's values × u, never over the run's
  > length. At run time nothing is allocated, and every address is a constant of the static schedule.
  - Placed by: "It is computed at compile time, once per static schedule"
    (design/disjoint-pages-change-request.md:855); "`mesh_plan` computes u and the copies, and A3's allocator places
    the copies' pages." (design/disjoint-pages-change-request.md:852)
- **C3.4 (a) no claim** (design/disjoint-pages-change-request.md:859-861)
  > **The data structure** is the page table of one offline allocation. A3's allocator places every value of the whole
  > exported program, at compile time (design/aot-compilation-change-request.md A1). Nothing is claimed at attach.
  > There is no claim per `mesh_plan` call, no first fit and no run-time handle lookup.
  - Placed by: "A3's allocator places every value of the whole exported program, at compile time"
    (design/disjoint-pages-change-request.md:859-860); "**Written by, read by.** `mesh_plan` writes the placement."
    (design/disjoint-pages-change-request.md:867)
- **C3.4 (b) new state in new pages, no wait** (design/disjoint-pages-change-request.md:862-866)
  > A new static schedule's state is in new pages, never the previous one's. Nothing erases a static schedule or waits
  > for one to be erased. **The algorithm for release** is deferred reclamation (RCU's `call_rcu`, P. E. McKenney and
  > J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004): the bridge releases a retired
  > static schedule's pages once none of its ports still has that static schedule loaded, and nothing waits for the
  > release.
  - Placed by: "**Written by, read by.** `mesh_plan` writes the placement. The bridge reads it, and releases its pages
    after its retirement." (design/disjoint-pages-change-request.md:867-868); "`mesh_plan`'s allocator places the
    array and the words and writes their starting values." (design/disjoint-pages-change-request.md:906)
- **C3.4 "Written by, read by"** (design/disjoint-pages-change-request.md:867-868)
  > **Written by, read by.** `mesh_plan` writes the placement. The bridge reads it, and releases its pages after its
  > retirement. The placement is the `struct mesh_page_entry` rows.
  - Placed by: "`mesh_plan` writes the placement." (design/disjoint-pages-change-request.md:867); "- **Schedule.**
    `mesh_plan` (mesh `rdma/mesh-collective.c`) takes the program's values and its dependence graph and writes the
    static schedule, which is the scheduled value." (design/disjoint-pages-change-request.md:383-384)
- **C3.5** (design/disjoint-pages-change-request.md:871-876): stated under caller, which its clause names with this
  stage.
- **C3.6 (1) late cancellation** (design/disjoint-pages-change-request.md:880-881)
  > The cancellation that arrives after a later `mesh_plan` call has zeroed the records it writes: removed by C3.4 and
  > C3.7, because no static schedule writes another's state.
  - Placed by: "a later `mesh_plan` call has zeroed the records it writes: removed by C3.4 and C3.7"
    (design/disjoint-pages-change-request.md:880-881); "`mesh_plan`'s allocator places the array and the words and
    writes their starting values." (design/disjoint-pages-change-request.md:906)
- **C3.7 data structures** (design/disjoint-pages-change-request.md:890-894): stated under scheduled value, which its
  clause names with this stage.
- **C3.7 algorithm 3** (design/disjoint-pages-change-request.md:900-901)
  > The fields start at 00 when the allocator places the array, once per static schedule, as `mbarrier.init` runs once
  > per kernel. A field is reused only with its page, so the allocator's order for the page covers it.
  - Placed by: "**Written by, read by.** `mesh_plan`'s allocator places the array and the words and writes their
    starting values." (design/disjoint-pages-change-request.md:906)
- **C3.7 algorithm 4** (design/disjoint-pages-change-request.md:902-904): stated under bridge, which its clause names
  with this stage.
- **C3.7 "Written by, read by"** (design/disjoint-pages-change-request.md:906-910)
  > **Written by, read by.** `mesh_plan`'s allocator places the array and the words and writes their starting values.
  > The bridge (`rdma/mesh-flow.c`) writes the phases and the counts and reads the words. Each range's producer writes
  > its word. The recorded kernels' first pass reads the phases (C3.1). The array, the counts and the words are at
  > `struct mesh_state`'s offsets `phases`, `counts` and `supplied`; a receive's field is its `struct mesh_hop`'s
  > `field`, and a send's word its `word` (`rdma/mesh.h`).
  - Placed by: "`mesh_plan`'s allocator places the array and the words and writes their starting values."
    (design/disjoint-pages-change-request.md:906); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`) takes
    the program's values and its dependence graph and writes the static schedule, which is the scheduled value."
    (design/disjoint-pages-change-request.md:383-384)
- **C4 mesh deleted: rdma/mesh-collective.c (executions, per-execution items)**
  (design/disjoint-pages-change-request.md:920-922)
  > rdma/mesh-collective.c: `executions` as an input of `mesh_plan`, and its per-execution items; the `memset` of
  > every declared instance's pages before `hops` (lines 749–757); the word-by-word zeroing of the supplied words and
  > landed records (lines 759–763).
  - Placed by: "rdma/mesh-collective.c: `executions` as an input of `mesh_plan`"
    (design/disjoint-pages-change-request.md:920); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`) takes
    the program's values and its dependence graph and writes the static schedule, which is the scheduled value."
    (design/disjoint-pages-change-request.md:383-384)
- **C4 mesh deleted: rdma/mesh-collective.c (memset and word zeroing)**
  (design/disjoint-pages-change-request.md:920-922)
  > rdma/mesh-collective.c: `executions` as an input of `mesh_plan`, and its per-execution items; the `memset` of
  > every declared instance's pages before `hops` (lines 749–757); the word-by-word zeroing of the supplied words and
  > landed records (lines 759–763).
  - Placed by: "rdma/mesh-collective.c: `executions` as an input of `mesh_plan`"
    (design/disjoint-pages-change-request.md:920); "- **Schedule.** `mesh_plan` (mesh `rdma/mesh-collective.c`) takes
    the program's values and its dependence graph and writes the static schedule, which is the scheduled value."
    (design/disjoint-pages-change-request.md:383-384)
- **C4 mesh deleted: mesh_plan_erased** (design/disjoint-pages-change-request.md:923-924)
  > rdma/mesh-call.h and rdma/mesh-dataflow.c: `mesh_plan_erased`, and its seq_cst fence pair with the ports' code in
  > rdma/mesh-flow.c.
  - Placed by: "**The erased flag and its Dekker fences** (`mesh_plan_erased`, mesh-dataflow.c). A new plan zeroes and
    rewrites the previous plan's supplied words and landed records in place, because they sit at fixed offsets in the
    layout. Before doing that, it stores `hops = 0`, issues a seq_cst fence, and spins for up to MESH_PLATFORM_NS (5
    s) until no port's `prepared` still holds the old generation. Each port fences on its side too."
    (design/disjoint-pages-change-request.md:629-632)
- **C4 mesh deleted: rdma/mesh-dataflow.c claim** (design/disjoint-pages-change-request.md:927-929)
  > rdma/mesh-dataflow.c: `mesh_claim`'s claim at attach of everything free, and the wait before it for the ports to
  > drop a retired client's static schedule; the claim's local run, where values were laid out when created
  > (`mesh_section_create` with `wire` 0).
  - Placed by: "A3's allocator places every value of the whole exported program, at compile time
    (design/aot-compilation-change-request.md A1). Nothing is claimed at attach."
    (design/disjoint-pages-change-request.md:859-860)
- **C4 metal-microbench written: tools/e2b_replay.m** (design/disjoint-pages-change-request.md:941-941): stated under
  caller, which its clause names with this stage.
- **D.3.2** (design/disjoint-pages-change-request.md:1386-1395)
  > **The data structure** is the page table, and **the algorithm** is A3's allocator, with invariants placed once
  > (C3.2). Every buffer the capture binds is a value `mesh_plan` places: the weights, and every delegate buffer,
  > those the recorder renames and those it doesn't. A buffer written before the first execution and read by every one
  > is an invariant. That covers the weights, the constants, and any buffer whose first read in an execution comes
  > before its first write. The rest are values of one execution, with copies (C3.3). **Written by, read by.**
  > `mesh_plan` places the pages. An invariant's producer writes it into its placed pages once, after `mesh_plan` and
  > before the first execution. For the weights, that write is one copy from the buffer the delegate loaded, at the
  > bind: the producer writing its own value (C3.7). The recorded kernels read the placed pages, and after the bind no
  > command reads a delegate buffer. A value is a `struct mesh_value`, and an invariant has `kind` `MESH_INVARIANT`
  > (`rdma/mesh-collective.h`).
  - Placed by: "Every buffer the capture binds is a value `mesh_plan` places"
    (design/disjoint-pages-change-request.md:1387); "**Written by, read by.** `mesh_plan` places the pages."
    (design/disjoint-pages-change-request.md:1391)
- **D.3.3** (design/disjoint-pages-change-request.md:1398-1405)
  > **The data structure** is the copies of each such input, and **the algorithm** is modulo variable expansion (C3.3)
  > by a declared lag. The inputs are the positions, masks, cache parameters, the schedule's per-step values and the
  > token stream. **Written by, read by.** The program declares the lag of the CPU code that writes them: with that
  > code writing execution t + 1 while the GPU runs execution t, the lag is 2. `mesh_plan`'s allocator gives the
  > inputs copies by that lag. The CPU code writes each copy, and the recorded kernels read it. The lag is `struct
  > mesh_value`'s `lag` (`rdma/mesh-collective.h`). Nothing the run binds grows with its executions.
  - Placed by: "`mesh_plan`'s allocator gives the inputs copies by that lag."
    (design/disjoint-pages-change-request.md:1402-1403); "The program declares the lag of the CPU code that writes
    them" (design/disjoint-pages-change-request.md:1401)
- **D.3.5** (design/disjoint-pages-change-request.md:1425-1431): stated under bridge, which its clause names with this
  stage.
- **D.4 metal-microbench deleted (2): per-step instances** (design/disjoint-pages-change-request.md:1461-1461)
  > The per-step instances of the inputs the CPU writes, and their bindings at t × stride.
  - Placed by: "`mesh_plan`'s allocator gives the inputs copies by that lag."
    (design/disjoint-pages-change-request.md:1402-1403); "Nothing the run binds grows with its executions."
    (design/disjoint-pages-change-request.md:1405)

### schedule: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R1 (requirement)** (design/plan-to-bridge-change-request.md:1948-1948): stated under bridge, which its clause
  names with this stage.
- **R1 data structure** (design/plan-to-bridge-change-request.md:1949-1950, 1974-1975): stated under scheduled value,
  which its clause names with this stage.
- **R1 algorithm** (design/plan-to-bridge-change-request.md:1962-1962, 1967-1968)
  > - **The algorithm is the flattening of a type map.** [...] - Each view is flattened into its list of (row, offset,
  > bytes) blocks, adjacent blocks merged, as datatype processing flattens a type map.
  - Placed by: "Produced by the schedule, `mesh_plan`." design/plan-to-bridge-change-request.md:1976
- **R1 where it composes** (design/plan-to-bridge-change-request.md:1970-1977)
  > Takes: each transfer's two views. A transfer is a send node and a receive node that placement adds to the graph.
  > Its type in the code is `struct mesh_declared`, whose `from` and `to` are each a `struct mesh_strided`
  > (rdma/mesh-collective.h, rdma/mesh-graph.h). `mesh_plan` takes them as the `declared` array of each `struct
  > mesh_function`. Writes: the blocks, as the entries of each cable's list (R5): `struct mesh_hop` `page`, `offset`
  > and `bytes` (rdma/mesh.h). These are MSCCL-IR's steps with offset and count (aot-compilation A9). Produced by the
  > schedule, `mesh_plan`. Consumed by the bridge (rdma/mesh-flow.c), which sends each block from the page that holds
  > it to the page it lands in.
  - Placed by: "Produced by the schedule, `mesh_plan`. Consumed by the bridge (rdma/mesh-flow.c), which sends each
    block from the page that holds it to the page it lands in." design/plan-to-bridge-change-request.md:1976-1977; "A
    transfer is a send node and a receive node that placement adds to the graph."
    design/plan-to-bridge-change-request.md:1970
- **R1 a value's layout is the program's own** (design/plan-to-bridge-change-request.md:1980-1981, 1983-1985)
  > - A part is sent from its producer's pages, whatever its layout, and it lands at its byte ranges in its
  > destination value's layout (R2). [...] - A change of layout exists only where a consumer's operand cannot take the
  > landed layout as its view. It is then an operator of the consumer's own, run as its out variant
  > (design/aot-compilation-change-request.md appendix item 7), never part of a transfer.
  - Placed by: "Produced by the schedule, `mesh_plan`." design/plan-to-bridge-change-request.md:1976
- **R1 Replaces: the share record** (design/plan-to-bridge-change-request.md:2007-2007)
  > **Replaces:** packing and unpacking; a layout chosen for a transfer's sake; the share record (`struct
  > mesh_share`).
  - Placed by: "**Replaces,** in `mesh_plan`'s rules (rdma/mesh-collective.h;
    design/disjoint-pages-revision-coordination.md, "The plan call"):" ... "the share record (`struct mesh_share`) and
    its rules." design/plan-to-bridge-change-request.md:2122-2128
- **R1 Replaces: packing and unpacking; a layout chosen for a transfer's sake**
  (design/plan-to-bridge-change-request.md:2007-2007)
  > **Replaces:** packing and unpacking; a layout chosen for a transfer's sake; the share record (`struct
  > mesh_share`).
  - Placed by: "Produced by the schedule, `mesh_plan`." design/plan-to-bridge-change-request.md:1976
- **R2 (requirement)** (design/plan-to-bridge-change-request.md:2013-2013)
  > The code must give every value its pages at compile time and name each page by its row in the page table.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h). The rows are part of the scheduled value."
    design/plan-to-bridge-change-request.md:2118-2119
- **R2 data structure** (design/plan-to-bridge-change-request.md:2014-2016): stated under scheduled value, which its
  clause names with this stage.
- **R2 algorithm 1: the copies (C3.3)** (design/plan-to-bridge-change-request.md:2040-2044)
  > The copies (M. Lam, Software pipelining, PLDI 1988). C3.3, verbatim: "For each value, L is the smallest lag at
  > which every read of it in execution t happens before its write in execution t + L. The unroll factor u is the
  > largest L." "The template is unrolled into u copies, and execution t runs copy t mod u." "At run time there are no
  > allocations and no bookkeeping; every address is a constant in the plan."
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2118-2119
- **R2 algorithm 2-3: the order and the chains** (design/plan-to-bridge-change-request.md:2045-2053)
  > The order. Every item is one page. Two items may share a page exactly when one's last access happens before the
  > other's first write. That is C3.3's own interference test, over Lamport's happens-before (L. Lamport, Time,
  > clocks, and the ordering of events in a distributed system, CACM 21(7), 1978). That relation is a partial order:
  > if x ends before y begins and y ends before z begins, then x ends before z begins. A page holds a chain of it, so
  > the fewest pages is the fewest chains. The chains. By Dilworth's theorem that number equals the largest set of
  > items no two of which are ordered (R. P. Dilworth, A decomposition theorem for partially ordered sets, Annals of
  > Mathematics 51(1), 1950). It is found by maximum bipartite matching (D. R. Fulkerson, Note on Dilworth's
  > decomposition theorem for partially ordered sets, Proc. AMS 7(4), 1956), exactly and in polynomial time.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2118-2119
- **R2 algorithm 4: the order over u copies** (design/plan-to-bridge-change-request.md:2054-2058)
  > Over C3.3's u copies the order has one more clause. The relation above holds over one execution. Over u copies, a
  > page is written again by its chain's first item u executions later, so a chain's last item must end before that
  > write. Copies (v, c) and (w, c′) are ordered where c ≤ c′, c′ − c is at least the least lag from v to w, and c′
  > plus w's largest least lag to a write at the node is at most u. That keeps the relation transitive.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2118-2119
- **R2 Replaces greedy by size** (design/plan-to-bridge-change-request.md:2063-2065)
  > Replaces C3.3's "The colouring is greedy by size (Y. Pisarchyk and J. Lee, arXiv 2001.03288, 2020)." No value is
  > placed as one run, so greedy by size, and the contiguous-block problem it approximates (M. R. Garey and D. S.
  > Johnson, Computers and Intractability, 1979, problem SR2), have no use here.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2118-2119
- **R2 a run is a mapping, not a placement** (design/plan-to-bridge-change-request.md:2069-2074)
  > Such an operand's run is address translation, fixed when the static schedule is made. Every page's place is a
  > constant of the static schedule (C3.3: "every address is a constant in the plan"), so the map from the operand's
  > consecutive virtual pages to its pages is written once. The GPU translates through its page tables, as it does for
  > every buffer, and a buffer is already virtual memory over frames placed anywhere. So a torch kernel reads the
  > operand as any strided buffer, and no kernel reads a table of its own.
  - Placed by: "Such an operand's run is address translation, fixed when the static schedule is made."
    design/plan-to-bridge-change-request.md:2069; "Writes: u; for each value and copy, its page-table rows; for each
    operand seen as one run, the map from its consecutive virtual pages to its pages. Each is a constant of the static
    schedule." design/plan-to-bridge-change-request.md:2115-2116; "Consumed by R5's lists, whose entries name each
    page by its row (R1), and by the GPU, which translates each run through its page tables."
    design/plan-to-bridge-change-request.md:2120-2121
- **R2 a received range is pages of its destination value** (design/plan-to-bridge-change-request.md:2082-2086)
  > **A received range is pages of its destination value.** A collective's receive lands in the pages of the value it
  > belongs to, at its place in that value: "receiving data from rank `i` at offset `i*sendcount`" (NCCL,
  > ncclAllGather); "the `recvbuff` on rank `i` will contain the i-th block of the result" (NCCL, ncclReduceScatter).
  > A range is not a value of its own. A part lands at its byte ranges in the result's own layout, whatever its
  > dimension; no layout is chosen for a transfer, and no part is padded (R1).
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h). The rows are part of the scheduled value."
    design/plan-to-bridge-change-request.md:2118-2119
- **R2 a rank's own part is shared, not copied** (design/plan-to-bridge-change-request.md:2087-2092)
  > **A rank's own part is shared, not copied.** A value is its page-table rows, so the result's rows for the rank's
  > own part name the operand's own pages. This is sharing by reference through the page table, as fbufs (P. Druschel
  > and L. L. Peterson, Fbufs: a high-bandwidth cross-domain transfer facility, SOSP 1993) and IO-Lite (V. S. Pai, P.
  > Druschel and W. Zwaenepoel, IO-Lite: a unified I/O buffering and caching system, ACM TOCS 18(1), 2000) pass
  > buffers without copying them. A value's rows naming another value's frames is the page table's own sharing, with
  > no record and no rules of its own.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h). The rows are part of the scheduled value."
    design/plan-to-bridge-change-request.md:2118-2119
- **R2 shared pages are one item of the allocation** (design/plan-to-bridge-change-request.md:2093-2096)
  > **Shared pages are one item of the allocation,** from the earlier of the two values' first writes to the later of
  > their last reads. Where that first write is a landing, the next landing into those pages follows the last read by
  > C3.3's lag test, as any value's does. Shared pages with no such lag are C3.3's case: "A value with no such L fails
  > the plan with `ENOMEM`, naming the value."
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2118-2119
- **R2 any value a transfer names is sent from its own pages** (design/plan-to-bridge-change-request.md:2097-2103)
  > **Any value a transfer names is sent from its own pages, whatever its copies.** R1: "The code must move every page
  > from the page that holds it to the page it lands in, with nothing copied on the way." An invariant, written before
  > the first execution and read by every one (C3.2: "placed once, for the plan's whole life"), is placed in the
  > registered window when a transfer names it. Every execution's send reads its one copy, and its supplied word is
  > raised once. A program's input is placed per execution, and its caller writes it there. It is sent from those
  > pages. A result whose rows name an invariant operand's pages names the operand's one copy from every copy of the
  > result.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h). The rows are part of the scheduled value."
    design/plan-to-bridge-change-request.md:2118-2119
- **R2 ranges may share a block where their bytes are disjoint** (design/plan-to-bridge-change-request.md:2104-2110)
  > **Ranges may share a block where their bytes are disjoint.** C3's rule is "Its producer or its landing writes each
  > byte once per use", each byte, not each block. Each range keeps its own landed count of each copy
  > (design/aot-compilation-change-request.md appendix item 5). A page instruction names its bytes at their offset in
  > its page, as MSCCL-IR's steps name a destination offset and a count (MSCCL-IR's step attributes are cited from
  > memory). At a source, any number of ranges read the same blocks. A halo wider than a shard, or two halos over one
  > shard, is then sent with no copy.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2118-2119
- **R2 where it composes** (design/plan-to-bridge-change-request.md:2112-2121)
  > Takes: the exported program's values with their lifetimes, and which consumers see a value as one run (R4). In the
  > code a value is a `struct mesh_value`, in the `value` array of each `struct mesh_function` that `mesh_plan` takes
  > (rdma/mesh-collective.h). Writes: u; for each value and copy, its page-table rows; for each operand seen as one
  > run, the map from its consecutive virtual pages to its pages. Each is a constant of the static schedule. This
  > item's sources name no type for the rows as written or for a run's map. Produced when the static schedule is made:
  > `mesh_plan` (rdma/mesh-collective.c) calls `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h). The rows
  > are part of the scheduled value. Consumed by R5's lists, whose entries name each page by its row (R1), and by the
  > GPU, which translates each run through its page tables.
  - Placed by: "Produced when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c) calls
    `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h). The rows are part of the scheduled value."
    design/plan-to-bridge-change-request.md:2118-2119
- **R2 Replaces (mesh_plan's rules)** (design/plan-to-bridge-change-request.md:2122-2128)
  > **Replaces,** in `mesh_plan`'s rules (rdma/mesh-collective.h; design/disjoint-pages-revision-coordination.md, "The
  > plan call"): "That value is MESH_EXECUTION and wire, its bytes the range's, and one range's alone", which cites
  > nothing; a lag-k value "named by no declared transfer"; a share's two values having "the same copies"; "no two
  > ranges at this node share a block of it"; the share record (`struct mesh_share`) and its rules.
  - Placed by: "**Replaces,** in `mesh_plan`'s rules (rdma/mesh-collective.h;
    design/disjoint-pages-revision-coordination.md, "The plan call"):"
    design/plan-to-bridge-change-request.md:2122-2123
- **R3 the task graph: V and w** (design/plan-to-bridge-change-request.md:2137-2137, 2141-2141, 2143-2144)
  > - The code must represent one execution as a task graph and nothing more. [...] - V are tasks, each one operation
  > of the captured graph (aot-compilation A1) executed at its node. [...] - w and c are the tasks' and edges'
  > weights. They are given, not computed by this code: the weighting operand of
  > design/aot-compilation-change-request.md A3.
  - Placed by: "Produced by `mesh_plan`: rdma/mesh-collective.{h,c} builds the task graph from the IR."
    design/plan-to-bridge-change-request.md:2151
- **R3 the task graph: E are pages, with weight c** (design/plan-to-bridge-change-request.md:2142-2144)
  > E are the data passed between tasks: here, pages. w and c are the tasks' and edges' weights. They are given, not
  > computed by this code: the weighting operand of design/aot-compilation-change-request.md A3.
  - Placed by: "Produced by `mesh_plan`: rdma/mesh-collective.{h,c} builds the task graph from the IR."
    design/plan-to-bridge-change-request.md:2151
- **R3 where it composes** (design/plan-to-bridge-change-request.md:2147-2155)
  > Takes: the compilation's IR (aot-compilation A8), which is the captured graph with each task at its node by the
  > placement operand (aot-compilation A2), and the weighting operand (aot-compilation A3). In the code their type is
  > the array of `struct mesh_function` that `mesh_plan` takes: each has its `declared` (`struct mesh_declared`), its
  > `weights` (`struct mesh_weight`) and its `dependence` (`struct mesh_dependence`) (rdma/mesh-collective.h).
  > Produced by `mesh_plan`: rdma/mesh-collective.{h,c} builds the task graph from the IR. No header names a type for
  > the task graph. Consumed by R5's linear program, whose variables are the graph's tasks and each hop of each page's
  > route. `mesh_graph_completion_times` (rdma/mesh-graph.h) takes them as arrays: each instruction's `length` and
  > `resource`, and the edges' `from` and `to`.
  - Placed by: "Produced by `mesh_plan`: rdma/mesh-collective.{h,c} builds the task graph from the IR."
    design/plan-to-bridge-change-request.md:2151
- **R4 (requirement)** (design/plan-to-bridge-change-request.md:2177-2177)
  > The code must schedule both kinds of function, and must not tell the bridge which is which.
  - Placed by: "Produced by the schedule, `mesh_plan`. Consumed by the bridge, which reads the lists and the supplied
    words." design/plan-to-bridge-change-request.md:2254
- **R4 behaviour at run time, 3** (design/plan-to-bridge-change-request.md:2189-2193)
  > Each cable sends its pages in the order R5's program gives. In that program the two kinds are rows, not labels:
  > the first kind's task "has an edge from every page of its operand, which with (1c) is Chen's (1a): 'the completion
  > time of coflow k is bounded by all its flows'", and each of the second kind's tasks "has one edge, from its own
  > page, so its pages are independent flows" (R5). So the order a cable sends in is computed from which pages a
  > function needs together and which it needs singly.
  - Placed by: "Each cable sends its pages in the order R5's program gives."
    design/plan-to-bridge-change-request.md:2189; "Writes: the kind as the task graph's edges (R3), from which R5's
    program orders each cable's sends." design/plan-to-bridge-change-request.md:2252; "Produced by the schedule,
    `mesh_plan`. Consumed by the bridge, which reads the lists and the supplied words."
    design/plan-to-bridge-change-request.md:2254
- **R4 behaviour at run time, 4** (design/plan-to-bridge-change-request.md:2194-2194)
  > The bridge receives that order and nothing about either kind.
  - Placed by: "The lists carry no kind, only which entries are a set's (aot-compilation, appendix item 5): `struct
    mesh_hop` `set` (rdma/mesh.h)." design/plan-to-bridge-change-request.md:2252-2253; "Produced by the schedule,
    `mesh_plan`. Consumed by the bridge, which reads the lists and the supplied words."
    design/plan-to-bridge-change-request.md:2254
- **R4 data structure** (design/plan-to-bridge-change-request.md:2200-2204)
  > **The data structure is the task graph's edges (R3).** The synchronous, complete, contiguous consumer is one task
  > with an edge from every page of its operand. The asynchronous, incomplete, noncontiguous consumer is one task per
  > page, each with an edge from its own page, and its outputs are pages that leave as each is produced. Their weights
  > come from the weighting operand.
  - Placed by: "Writes: the kind as the task graph's edges (R3), from which R5's program orders each cable's sends."
    design/plan-to-bridge-change-request.md:2252; "Produced by the schedule, `mesh_plan`."
    design/plan-to-bridge-change-request.md:2254
- **R4 contiguity is a mapping** (design/plan-to-bridge-change-request.md:2243-2248)
  > **Contiguity is a mapping (R2).** Only the synchronous, complete, contiguous consumer's operand, and an operand
  > whose consumer needs one strided progression, is seen as one run. That run is a mapping written when the static
  > schedule is made, and its pages are placed anywhere, as every value's are. Giving every value's copy one run of
  > placed pages imposes the synchronous, complete, contiguous consumer's type on the asynchronous, incomplete,
  > noncontiguous consumer's operands and on every transfer, which the operator's typing forbids: "function B requires
  > set of pages operand_b to be complete but not contiguous", "domain:Async,noncontiguous,incomplete" (above).
  - Placed by: "That run is a mapping written when the static schedule is made, and its pages are placed anywhere, as
    every value's are." design/plan-to-bridge-change-request.md:2244-2245
- **R4 where it composes** (design/plan-to-bridge-change-request.md:2250-2254)
  > Takes: each consumer's kind, which the IR carries (aot-compilation A8): `struct mesh_declared` `kind`, `MESH_JOIN`
  > or `MESH_PAGES` (rdma/mesh-collective.h). Writes: the kind as the task graph's edges (R3), from which R5's program
  > orders each cable's sends. The lists carry no kind, only which entries are a set's (aot-compilation, appendix item
  > 5): `struct mesh_hop` `set` (rdma/mesh.h). Produced by the schedule, `mesh_plan`. Consumed by the bridge, which
  > reads the lists and the supplied words.
  - Placed by: "Produced by the schedule, `mesh_plan`. Consumed by the bridge, which reads the lists and the supplied
    words." design/plan-to-bridge-change-request.md:2254
- **R5 (requirement)** (design/plan-to-bridge-change-request.md:2266-2266)
  > The code must order each cable's sends by a static schedule of one execution's task graph, minimizing its
  > makespan.
  - Placed by: "**Where it composes.** This is the one schedule of the data flow."
    design/plan-to-bridge-change-request.md:2351; "Produced by `mesh_plan` (rdma/mesh-collective.c), which calls
    `mesh_graph_completion_times` and `mesh_graph_heap_order` (rdma/mesh-graph.h)."
    design/plan-to-bridge-change-request.md:2358-2359
- **R5 data structure: one linear program over completion times** (design/plan-to-bridge-change-request.md:2267-2288)
  > - **The data structure is one linear program over completion times.** > then we need a LP solver, citing chen for
  > why, how, and what this means. - It has one variable C_j, a completion time, for each instruction j of one
  > execution's task graph (R3): each task at its node, and each hop of each page's route, on its cable
  > (design/compilation-coordination.md GO2). It has one more variable, C_max. Instruction j's length d_j is its
  > weight from the weighting operand (aot-compilation A3): w for a task, and for a hop c(e) over its cable's
  > bandwidth. - Minimize C_max, with C_max ≥ C_j for every j: Chen's (14) and (14a) (C.-Y. Chen, Scheduling coflows
  > with precedence constraints for minimizing the total weighted completion time in identical parallel networks,
  > arXiv 2205.02474, 2022, §6). Then, over (14)'s optimal solutions, minimize Σ_j C_j: Chen's (1) and Hall et al.'s
  > (1) with unit weights. The two are one lexicographic linear program (H. Isermann, Linear lexicographic
  > optimization, OR Spektrum 4, 1982). They are solved in sequence, the second with C_max held at the first's
  > optimum. The second objective fixes C̄_j off the critical path, which (14) alone leaves to whichever optimal
  > vertex the solver returns. - C_j ≥ d_j for every j: Chen's (1b), with every release time zero within one
  > execution. - C_j ≥ C_i + d_j for every edge i → j of the graph: Chen's (1c), "the precedence constraint", and Hall
  > et al.'s (3). R4's edges are among them. The synchronous, complete, contiguous consumer's task has an edge from
  > every page of its operand, which with (1c) is Chen's (1a): "the completion time of coflow k is bounded by all its
  > flows". Each of the asynchronous, incomplete, noncontiguous consumer's tasks has one edge, from its own page, so
  > its pages are independent flows. - For each resource, and each set S of the instructions on it: Σ_{j∈S} d_j C_j ≥
  > ½(d(S)² + d²(S)), where d(S) is Σ_{j∈S} d_j and d²(S) is Σ_{j∈S} d_j². A resource is a cable in one direction, or
  > a node's device. This is Hall et al.'s (5), "introduced by Queyranne (1993)". Chen's (1d) and (1e) take it for
  > each input and output port, here with one network core (m = 1).
  - Placed by: "Produced by `mesh_plan` (rdma/mesh-collective.c), which calls `mesh_graph_completion_times` and
    `mesh_graph_heap_order` (rdma/mesh-graph.h)." design/plan-to-bridge-change-request.md:2358-2359;
    "rdma/mesh-graph.{h,c}: the program's rows, the cutting-plane loop with Queyranne's separation, and the calls to
    HiGHS" design/plan-to-bridge-change-request.md:2364-2365
- **R5 algorithm steps 1-2: cutting planes, HiGHS dual simplex** (design/plan-to-bridge-change-request.md:2314-2323):
  stated under graph module, which its clause names with this stage.
- **R5 algorithm steps 3-5: priority, one global topological order, each cable's subsequence**
  (design/plan-to-bridge-change-request.md:2324-2332)
  > 3. Each instruction's priority is C̄_j, its completion time in the program's optimal solution, the least first.
  > C̄_j is a relaxation's value, a lower bound, and not a time any schedule keeps. Only its order is used. 4. One
  > global topological order of the instructions is built as MSCCLang §5.2 states in its step 3: "Sort instructions
  > into a global topological order with respect to their dependencies with a heap using the priority to order
  > instructions." Where two priorities are equal, the heap takes the lower instruction
  > (design/compilation-coordination.md GO3). That order is consistent with the precedence relation, as Hall et al.'s
  > ties require. > what algorithm is cited here, if any? no citations? use the reference algorithm instead. 5. Each
  > cable's list is the subsequence of that order made of the cable's sends.
  - Placed by: "Produced by `mesh_plan` (rdma/mesh-collective.c), which calls `mesh_graph_completion_times` and
    `mesh_graph_heap_order` (rdma/mesh-graph.h). The first writes each instruction's C̄_j (`double *completion`). The
    second takes them as `priority` and writes each instruction's place in the order (`uint32_t *place`)."
    design/plan-to-bridge-change-request.md:2358-2360
- **R5 a set's pages keep their places** (design/plan-to-bridge-change-request.md:2333-2334)
  > A set's pages (design/aot-compilation-change-request.md, appendix item 5) keep their places in that order. The
  > set's random permutation is drawn over those places.
  - Placed by: "**Where it composes.** This is the one schedule of the data flow."
    design/plan-to-bridge-change-request.md:2351; "Writes: for each copy, each cable's sends in schedule order, each
    entry with its page and its bytes (`struct mesh_hop`, rdma/mesh.h)."
    design/plan-to-bridge-change-request.md:2356-2357
- **R5 executions** (design/plan-to-bridge-change-request.md:2337-2340)
  > **Executions:** execution t runs copy t mod u, and every address is a constant of the static schedule (C3.3, R2).
  > Each cable's list holds execution t's sends in that order, then execution t + 1's. The bridge spreads each list
  > over what its devices give it, so on the cable the pages of released executions interleave (aot-compilation,
  > appendix item 2; T99).
  - Placed by: "Writes: for each copy, each cable's sends in schedule order, each entry with its page and its bytes
    (`struct mesh_hop`, rdma/mesh.h)." design/plan-to-bridge-change-request.md:2356-2357; "Consumed by the bridge.
    rdma/mesh-flow.c `link_prepare` spreads each cable's list, in list order, over what its devices give it (R7), by
    `mesh_graph_list_schedule`, and sends each entry's `bytes`." design/plan-to-bridge-change-request.md:2361-2362
- **R5 every rank computes the same schedule** (design/plan-to-bridge-change-request.md:2347-2350)
  > **Every rank computes the same schedule** from the same inputs, so no rank asks another anything at run time (F.
  > B. Schneider, Implementing fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990).
  > Every rank solves the same program with the same library, built from the same source with the same flags, on one
  > thread. So every rank finds the same solution and the same order.
  - Placed by: "**Where it composes.** This is the one schedule of the data flow."
    design/plan-to-bridge-change-request.md:2351
- **R5 where it composes** (design/plan-to-bridge-change-request.md:2351-2362)
  > **Where it composes.** This is the one schedule of the data flow. Takes: the task graph (R3), each task on its
  > node by the placement operand (aot-compilation A2), each length from the weighting operand (aot-compilation A3);
  > and each page's route, over trees packed by Garg and Könemann's rounds (N. Garg and J. Könemann, FOCS 1998;
  > rdma/mesh-graph.h `mesh_graph_pack`), each tree's pages assigned by `mesh_graph_fill` (called in
  > rdma/mesh-collective.c). Writes: for each copy, each cable's sends in schedule order, each entry with its page and
  > its bytes (`struct mesh_hop`, rdma/mesh.h). The lists are part of the scheduled value. Produced by `mesh_plan`
  > (rdma/mesh-collective.c), which calls `mesh_graph_completion_times` and `mesh_graph_heap_order`
  > (rdma/mesh-graph.h). The first writes each instruction's C̄_j (`double *completion`). The second takes them as
  > `priority` and writes each instruction's place in the order (`uint32_t *place`). Consumed by the bridge.
  > rdma/mesh-flow.c `link_prepare` spreads each cable's list, in list order, over what its devices give it (R7), by
  > `mesh_graph_list_schedule`, and sends each entry's `bytes`.
  - Placed by: "This is the one schedule of the data flow." design/plan-to-bridge-change-request.md:2351; "Produced by
    `mesh_plan` (rdma/mesh-collective.c)" design/plan-to-bridge-change-request.md:2358
- **R5 changes by file: mesh-collective.c and mesh-flow.c** (design/plan-to-bridge-change-request.md:2377-2379)
  > rdma/mesh-collective.c: writes each copy's list with each cable's sends in schedule order, each with its bytes
  > (`struct mesh_hop` `bytes`). rdma/mesh-flow.c: sends each entry's `bytes`.
  - Placed by: "rdma/mesh-collective.c: writes each copy's list with each cable's sends in schedule order"
    design/plan-to-bridge-change-request.md:2377; "Produced by `mesh_plan` (rdma/mesh-collective.c)"
    design/plan-to-bridge-change-request.md:2358; "Consumed by the bridge. rdma/mesh-flow.c `link_prepare` spreads
    each cable's list" design/plan-to-bridge-change-request.md:2361
- **R6 data structure and no run-time algorithm** (design/plan-to-bridge-change-request.md:2387-2392): stated under
  place, which its clause names with this stage.
- **R6 where it composes** (design/plan-to-bridge-change-request.md:2393-2396): stated under place, which its clause
  names with this stage.
- **R8 data structure and algorithm: one offline allocation at compile time**
  (design/plan-to-bridge-change-request.md:2429-2432)
  > **The data structure is R2's page table, and the algorithm is R2's:** one offline allocation over the whole
  > exported program (design/aot-compilation-change-request.md A1), at compile time. Its input is the program's values
  > with their lifetimes, and which consumers see one as a run, each such run a mapping. Nothing takes pages at run
  > time, and no handle is looked up at run time.
  - Placed by: "The pages are R2's rows, written when the static schedule is made and part of the scheduled value."
    design/plan-to-bridge-change-request.md:2440; "Produced when the static schedule is made: `mesh_plan`
    (rdma/mesh-collective.c) calls `mesh_graph_lags` and `mesh_graph_chains` (rdma/mesh-graph.h)."
    design/plan-to-bridge-change-request.md:2118-2119
- **R8 where it composes: placement and weights are operands** (design/plan-to-bridge-change-request.md:2438-2439):
  stated under caller, which its clause names with this stage.
- **R8 where it composes: the pages are part of the scheduled value**
  (design/plan-to-bridge-change-request.md:2440-2440): stated under scheduled value, which its clause names with this
  stage.
- **R8 Replaces: mesh_claim_take, the claim-head walk, fragmentation ENOMEM**
  (design/plan-to-bridge-change-request.md:2448-2453)
  > **Replaces:** `mesh_claim_take`, which took three contiguous runs at each `mesh_plan` call by first fit over a
  > bitmap (citing Knuth vol. 1 §2.5). That is run-time dynamic storage allocation, which the disjoint-pages request's
  > A2 names as the strategy its allocator replaces; the walk of every claim head in `mesh_transfers_start` to test
  > its handle (`mesh_plan_head`); `ENOMEM` from fragmentation, the claim's record holding one `count<<32|first` word
  > a run;
  - Placed by: "`mesh_claim_take`, which took three contiguous runs at each `mesh_plan` call by first fit over a
    bitmap" design/plan-to-bridge-change-request.md:2449; "one offline allocation over the whole exported program
    (design/aot-compilation-change-request.md A1), at compile time." design/plan-to-bridge-change-request.md:2429-2430

### schedule: [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

- **1, step 1** (design/problems-of-14b9a757-change-request.md:125-126): stated under place, which its clause names
  with this stage.
- **1, step 2** (design/problems-of-14b9a757-change-request.md:127-129): stated under place, which its clause names
  with this stage.
- **2, step 1** (design/problems-of-14b9a757-change-request.md:157-158): stated under place, which its clause names
  with this stage.
- **2, step 2** (design/problems-of-14b9a757-change-request.md:159-160): stated under place, which its clause names
  with this stage.
- **3, Change: the rule** (design/problems-of-14b9a757-change-request.md:190-194): stated under place, which its
  clause names with this stage.
- **3, Change: the result torch allocates** (design/problems-of-14b9a757-change-request.md:200-200): stated under
  place, which its clause names with this stage.
- **4, step 1** (design/problems-of-14b9a757-change-request.md:208-209): stated under place, which its clause names
  with this stage.
- **4, step 2** (design/problems-of-14b9a757-change-request.md:210-211): stated under place, which its clause names
  with this stage.
- **4, step 3** (design/problems-of-14b9a757-change-request.md:212-214): stated under place, which its clause names
  with this stage.
- **4, step 4** (design/problems-of-14b9a757-change-request.md:215-216): stated under place, which its clause names
  with this stage.
- **5, step 1** (design/problems-of-14b9a757-change-request.md:265-265): stated under place, which its clause names
  with this stage.
- **5, step 2** (design/problems-of-14b9a757-change-request.md:266-266): stated under place, which its clause names
  with this stage.
- **5, step 3** (design/problems-of-14b9a757-change-request.md:267-267): stated under place, which its clause names
  with this stage.
- **6, step 1** (design/problems-of-14b9a757-change-request.md:285-285): stated under place, which its clause names
  with this stage.
- **6, step 2** (design/problems-of-14b9a757-change-request.md:286-286): stated under place, which its clause names
  with this stage.
- **6, step 3** (design/problems-of-14b9a757-change-request.md:287-287): stated under place, which its clause names
  with this stage.
- **7 (Code)** (design/problems-of-14b9a757-change-request.md:308-310)
  > - **Code:** rdma/mesh-collective.c:820–826, 885 and 957–958 (`PLAN_ALONE`). - A named value's blocks that no range
  > covers are an item ordered with no other item. - That item therefore keeps its pages for the program's whole life.
  - Placed by: "rdma/mesh-collective.c:820–826, 885 and 957–958 (`PLAN_ALONE`)."
    design/problems-of-14b9a757-change-request.md:308; "`mesh_plan` (rdma/mesh-collective.c) is the function that
    writes the parts listed next, at each node, before execution 0"
    design/problems-of-14b9a757-change-request.md:80-81
- **7.1, step 1** (design/problems-of-14b9a757-change-request.md:317-319)
  > **A value's lifetime** is that pair, as `update_tensor_lifetime` computes it, over the graph nodes in the order
  > item 14's step 2 indexes: the node's operations in program order, which is the schedule's result (`ir.Scheduled`,
  > each node's graph in scheduled order).
  - Placed by: the reader.
- **7.1, step 2** (design/problems-of-14b9a757-change-request.md:320-322)
  > **A transfer's ends** each occupy their value from the graph node that starts the transfer to the graph node that
  > waits for its completion (MPI §3.7; the wait is 7.2's). A send's source value is occupied until its completion is
  > awaited, and a receive's destination value from the receive's start.
  - Placed by: the reader.
- **7.1, step 3** (design/problems-of-14b9a757-change-request.md:323-323)
  > **A block's lifetime** spans the lifetimes of the values whose bytes it holds.
  - Placed by: "`mesh_plan` reads them as the `begin` and the `end` of each `struct mesh_value`, as the intervals of
    item 14, and writes the memory plan and the memory schedule from them."
    design/problems-of-14b9a757-change-request.md:328-329
- **7.1, step 4** (design/problems-of-14b9a757-change-request.md:324-324)
  > **Sharing:** two blocks may share pages exactly where ExecuTorch's test finds their lifetimes disjoint.
  - Placed by: "`mesh_plan` reads them as the `begin` and the `end` of each `struct mesh_value`, as the intervals of
    item 14, and writes the memory plan and the memory schedule from them."
    design/problems-of-14b9a757-change-request.md:328-329
- **7.1, step 5** (design/problems-of-14b9a757-change-request.md:325-326)
  > **Never freed:** a value whose data is kept across executions (a copy of lag k ≥ 1, and an invariant) is never
  > freed, as ExecuTorch's mutable buffers are not. Only these have no lifetime.
  - Placed by: "`mesh_plan` reads them as the `begin` and the `end` of each `struct mesh_value`, as the intervals of
    item 14, and writes the memory plan and the memory schedule from them."
    design/problems-of-14b9a757-change-request.md:328-329
- **7.2, step 3** (design/problems-of-14b9a757-change-request.md:357-358)
  > A send's source value and a forwarded block are occupied until that graph node, which is where MPI §3.7's
  > restriction on the send buffer ends. Their lifetimes are then 7.1's.
  - Placed by: the reader.
- **7.2, Change: PLAN_SENT is deleted** (design/problems-of-14b9a757-change-request.md:371-371)
  > `PLAN_SENT` is deleted.
  - Placed by: the reader.
- **7.3, step 1** (design/problems-of-14b9a757-change-request.md:378-380)
  > There is one send-completion counter per range and copy at its source node, and one per forward at the node that
  > forwards, in the persistent request's state beside the receive counts. The counted operation is the send, which is
  > one per range: one send serves every destination of its range.
  - Placed by: "`mesh_plan` writes the counters into the persistent request's state (`struct mesh_state`)."
    design/problems-of-14b9a757-change-request.md:393
- **8, step 4** (design/problems-of-14b9a757-change-request.md:425-426): stated under place, which its clause names
  with this stage.
- **8, Change: LOADED and the execution UINT64_MAX − 1 are deleted**
  (design/problems-of-14b9a757-change-request.md:454-454)
  > LOADED and the execution UINT64_MAX − 1 are deleted.
  - Placed by: "rdma/mesh-collective.h:678 releases a range sent from a MESH_INVARIANT value once, "with `execution`
    UINT64_MAX − 1"." design/problems-of-14b9a757-change-request.md:436-437; "rdma/mesh_compile.py:2130 does the same
    (LOADED)." design/problems-of-14b9a757-change-request.md:438
- **9** (design/problems-of-14b9a757-change-request.md:460-472)
  > - **The algorithm:** none is determined. MPI says only that the use of such entries with a buffer the operation
  > updates "is erroneous". - **Composition:** `mesh_plan` writes each destination instruction as a `struct mesh_hop`
  > (rdma/mesh.h) of direction MESH_RECEIVE, its bytes named by `page`, `offset` and `bytes`. The bridge's child
  > process for a port reads them. No function reads their overlap, since no algorithm is determined. - **Code:** -
  > rdma/mesh-collective.c refuses no pair of destination instructions whose byte intervals at one node intersect
  > within one execution. - Ranges may now share a block, so such a pair is two writers of one byte. - **Source:** MPI
  > 4.1 §6.1.11 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node107.htm): "A datatype may specify overlapping
  > entries. The use of such a datatype in any communication in association with a buffer updated by the operation is
  > erroneous." - **Change:** none is determined. No source read states what `mesh_plan` does with such a pair.
  - Placed by: "`mesh_plan` writes each destination instruction as a `struct mesh_hop` (rdma/mesh.h) of direction
    MESH_RECEIVE, its bytes named by `page`, `offset` and `bytes`. The bridge's child process for a port reads them.
    No function reads their overlap, since no algorithm is determined."
    design/problems-of-14b9a757-change-request.md:462-464; "No source read states what `mesh_plan` does with such a
    pair." design/problems-of-14b9a757-change-request.md:472
- **10, Change: the representation** (design/problems-of-14b9a757-change-request.md:476-477): stated under scheduled
  value, which its clause names with this stage.
- **10, Composition: the placement stage writes it** (design/problems-of-14b9a757-change-request.md:479-480): stated
  under place, which its clause names with this stage.
- **11, step 4** (design/problems-of-14b9a757-change-request.md:499-500)
  > The persistent request's replacement is compiled on the next membership view. None is loaded partway through its
  > executions.
  - Placed by: "`mesh_plan` writes the replacement on the next membership view."
    design/problems-of-14b9a757-change-request.md:503
- **12, step 2** (design/problems-of-14b9a757-change-request.md:527-527)
  > A persistent request's `serial` is that pair, and every comparison compares the pair.
  - Placed by: "`mesh_plan` writes it as the persistent request's `serial`, in its `struct mesh_state`. The
    comparisons of rdma/mesh-flow.c read it." design/problems-of-14b9a757-change-request.md:529-530
- **14, step 1** (design/problems-of-14b9a757-change-request.md:591-592)
  > The `PLAN_LOCAL` and `PLAN_ROUTED` items of one persistent request at a node are the intervals of one cyclic
  > interval graph G.
  - Placed by: "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and `end`) from its operand, runs steps 1
    to 6, and writes each interval's colors as the memory schedule: the blocks that hold that value in each of the u
    executions." design/problems-of-14b9a757-change-request.md:607-608
- **14, step 2** (design/problems-of-14b9a757-change-request.md:593-594)
  > G's time points are the node's operations, in program order, over the u executions of the unrolled loop (Lam 1988
  > §2.3; the memo's §2.2).
  - Placed by: "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and `end`) from its operand, runs steps 1
    to 6, and writes each interval's colors as the memory schedule: the blocks that hold that value in each of the u
    executions." design/problems-of-14b9a757-change-request.md:607-608
- **14, step 3** (design/problems-of-14b9a757-change-request.md:595-598)
  > Each live range is one lifetime interval, from its write to its last read (7.1). As the memo's §2.2 has it, each
  > variable is "split into several variables depending the unrolling factor": a value of lag 0 has one interval per
  > execution, and a `PLAN_ROUTED` copy has one. A lifetime that crosses the end of the u executions is a cyclic
  > interval.
  - Placed by: "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and `end`) from its operand, runs steps 1
    to 6, and writes each interval's colors as the memory schedule: the blocks that hold that value in each of the u
    executions." design/problems-of-14b9a757-change-request.md:607-608
- **14, step 4** (design/problems-of-14b9a757-change-request.md:599-601)
  > What `mesh_plan` writes for a compiled function is a persistent request (MPI 4.1 §3.9). Each execution's start is
  > MPI_START, enqueued at its place in program order as NCCL enqueues a call. A received value's interval begins at
  > its execution's start.
  - Placed by: "What `mesh_plan` writes for a compiled function is a persistent request (MPI 4.1 §3.9)."
    design/problems-of-14b9a757-change-request.md:599; "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and
    `end`) from its operand, runs steps 1 to 6, and writes each interval's colors as the memory schedule: the blocks
    that hold that value in each of the u executions." design/problems-of-14b9a757-change-request.md:607-608
- **14, step 5** (design/problems-of-14b9a757-change-request.md:602-603)
  > An item of b blocks is one interval needing b colors (Algorithm 3.2.1's reg_num). Each color is a block of the
  > space. There is one register class.
  - Placed by: "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and `end`) from its operand, runs steps 1
    to 6, and writes each interval's colors as the memory schedule: the blocks that hold that value in each of the u
    executions." design/problems-of-14b9a757-change-request.md:607-608
- **14, step 6** (design/problems-of-14b9a757-change-request.md:604-606)
  > G is colored by Main Fatcover, Establish All Covers Of C_i and Choose A Cover For C_i as printed, with the left
  > edge algorithm (A.0.1) for the non-cyclic remainder. A cyclic interval with no fat cover gets new colors, as
  > §3.2.3's second phase states.
  - Placed by: "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and `end`) from its operand, runs steps 1
    to 6, and writes each interval's colors as the memory schedule: the blocks that hold that value in each of the u
    executions." design/problems-of-14b9a757-change-request.md:607-608
- **14, Change: replaces plan_before and mesh_graph_chains for these items**
  (design/problems-of-14b9a757-change-request.md:903-903)
  > This replaces `plan_before` and `mesh_graph_chains` for these items.
  - Placed by: "rdma/mesh-collective.c:918 (`plan_before`) returns 0 where `x->kind != y->kind`."
    design/problems-of-14b9a757-change-request.md:611; "So `mesh_graph_chains` (mesh-collective.c:1037;
    rdma/mesh-graph.h:198) never puts them on one block." design/problems-of-14b9a757-change-request.md:614;
    "`mesh_plan` reads 7.1's pairs (`struct mesh_value` `begin` and `end`) from its operand, runs steps 1 to 6"
    design/problems-of-14b9a757-change-request.md:607

### schedule: [torch-capture-change-request.md](torch-capture-change-request.md)

- **2, "It takes / It writes" (the IPC graph is a function of the static schedule)**
  (design/torch-capture-change-request.md:460-464): stated under argument_buffers, which its clause names with this
  stage.
- **3, LiteRT scope, Write, sixth bullet (task list in order of completion times)**
  (design/torch-capture-change-request.md:621-623): stated under no stage named, which its clause names with this
  stage.
- **6, Write, first bullet (offsets by Greedy by Size)** (design/torch-capture-change-request.md:727-728)
  > Each operand of a function typed `contiguous_input` or `contiguous_output` is given an offset in one memory block
  > by Greedy by Size for Offset Calculation, as TensorFlow Lite's `GreedyBySizeAssignment` states it.
  - Placed by: "The interval is over the u executions of the unrolled steady state, which is the static schedule's
    (Lam 1988), so the records are a function of `schedule.schedule`'s result. The item names no function that writes
    the offsets and none that reads them, and no type for an offset." (design/torch-capture-change-request.md:708-710)
- **6, "Where the block lies, and the usage records of the u locations", Write, first bullet (the block is its own
  allocation)** (design/torch-capture-change-request.md:753-756)
  > The memory block is its own allocation, of Greedy by Size's `total_size`, committed apart from every other
  > operand's pages as `arena_` and `persistent_arena_` are committed apart: consecutive pages, in which no page of
  > another operand lies. Nothing else constrains where it lies, the other operands being "arbitrary views of
  > arbitrarily placed pages" (2026-11-02).
  - Placed by: "The interval is over the u executions of the unrolled steady state, which is the static schedule's
    (Lam 1988), so the records are a function of `schedule.schedule`'s result. The item names no function that writes
    the offsets and none that reads them, and no type for an offset." (design/torch-capture-change-request.md:708-710)
- **6, "Where the block lies, and the usage records of the u locations", Write, second bullet (u usage records;
  do_arcs_overlap on the ring)** (design/torch-capture-change-request.md:757-761)
  > An operand's u locations are u usage records, one per location, each with the operand's size and the interval of
  > that location's lifetime over the u executions of the unrolled steady state (Memo 33 §2.2). A record whose
  > interval wraps from the last execution into the first is one interval depicted as two (Mukerji §3.1); the
  > intersection test of greedy_by_size_assignment.cc:93–94 is `do_arcs_overlap` on the ring of the u executions, and
  > nothing else of `GreedyBySizeAssignment` changes.
  - Placed by: "The interval is over the u executions of the unrolled steady state, which is the static schedule's
    (Lam 1988), so the records are a function of `schedule.schedule`'s result. The item names no function that writes
    the offsets and none that reads them, and no type for an offset." (design/torch-capture-change-request.md:708-710)
- **8. The torch scope's final lowering** (design/torch-capture-change-request.md:796-830): stated under export, which
  its clause names with this stage.

### schedule: [unequal-shards-change-request.md](unequal-shards-change-request.md)

- **2 - Stage** (design/unequal-shards-change-request.md:130-132): stated under place, which its clause names with
  this stage.

### schedule: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **5 / Write: a program with no wait** (design/synchronization-signature-change-request.md:154-155): stated under
  lower, which its clause names with this stage.

### schedule: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **1 / Write: the operator lowering produces them on every backend**
  (design/torch-source-lowering-change-request.md:166-167): stated under no stage named, which its clause names with
  this stage.
- **3 / Replaces** (design/torch-source-lowering-change-request.md:200-201)
  > **Replaces:** the order of a torch program's GPU commands, which is the order `_captured` (:1775) recorded them
  > in; and the second linear program over the GPU's commands (metal-microbench 3b719f10, `MetalOrder`).
  - Placed by: "Item 3's ordering is the schedule stage's." design/torch-source-lowering-change-request.md:132
- **3 / Write: one schedule stage** (design/torch-source-lowering-change-request.md:241-243)
  > This is the schedule stage, and there is one schedule. It takes the placed graph and the weighting operand
  > (design/aot-compilation-change-request.md A3). It produces the scheduled value: for each processor, its functions
  > in order. Lowering, the runtime and the archive each take that scheduled value.
  - Placed by: "This is the schedule stage, and there is one schedule."
    design/torch-source-lowering-change-request.md:241
- **3 / Write: the algorithm for the order** (design/torch-source-lowering-change-request.md:245-248)
  > The algorithm for the order is the completion-time linear program, which minimizes C_max and then ΣC_j over the
  > completion times C_j (C.-Y. Chen, arXiv 2205.02474, 2022; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein,
  > Math. Oper. Res. 22(3), 1997; M. Queyranne, Math. Programming 58, 1993; H. Isermann, OR Spektrum 4, 1982),
  > followed by the heap order by those completion times (M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2).
  - Placed by: "This is the schedule stage, and there is one schedule."
    design/torch-source-lowering-change-request.md:241
- **3 / Write: each task's length** (design/torch-source-lowering-change-request.md:249-249)
  > Each task's length is from the weighting operand.
  - Placed by: "This is the schedule stage, and there is one schedule. It takes the placed graph and the weighting
    operand (design/aot-compilation-change-request.md A3)." design/torch-source-lowering-change-request.md:241-242
- **3 / Write: the order is published** (design/torch-source-lowering-change-request.md:249-250)
  > The order is published with the AOT compiled static link schedule.
  - Placed by: "This is the schedule stage, and there is one schedule."
    design/torch-source-lowering-change-request.md:241
- **3 / Write: Sriram's synchronization graph** (design/torch-source-lowering-change-request.md:251-253)
  > The data structure for the orderings is Sriram's synchronization graph. Only the edges of a synchronization graph
  > that preserves the inter-processor communication graph are written: "the synchronization edges alone determine the
  > interaction between processors".
  - Placed by: "Item 3's ordering is the schedule stage's." design/torch-source-lowering-change-request.md:132
- **5 / Write: it is an operator lowering** (design/torch-source-lowering-change-request.md:295-296): stated under no
  stage named, which its clause names with this stage.
- **6 / Write: nodes of the exported graph on every processor**
  (design/torch-source-lowering-change-request.md:388-392): stated under no stage named, which its clause names with
  this stage.

### schedule: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **2 (The algorithm)** (design/integration-finishing-change-request.md:37-38)
  > **The algorithm** is list scheduling in the order of the linear program's completion times (Hall, Schulz, Shmoys
  > and Wein; design/list-scheduling-sources-2026-11-08.md, LS).
  - Placed by: "### 2. A node's replayed commands in the completion-time order"
    design/integration-finishing-change-request.md:35; "**The algorithm** is list scheduling"
    design/integration-finishing-change-request.md:37
- **2 (It takes)** (design/integration-finishing-change-request.md:39-41)
  > **It takes** a key for each of a node's replayed commands, and the commands' dependences. A command that reads a
  > received range has that range's published completion time as its key (`struct mesh_state` `times`). Every other
  > command has its dependence level as its key; no cited source states that key.
  - Placed by: "**The algorithm** is list scheduling" design/integration-finishing-change-request.md:37; "**It takes**
    a key for each of a node's replayed commands, and the commands' dependences."
    design/integration-finishing-change-request.md:39
- **2 (It writes)** (design/integration-finishing-change-request.md:42-44): stated under graph module, which its
  clause names with this stage.
- **3 (The data structure)** (design/integration-finishing-change-request.md:50-52): stated under caller, which its
  clause names with this stage.
- **3 (Consumed by)** (design/integration-finishing-change-request.md:53-54)
  > **Consumed by** the compilation's router and its scheduler, which compute no weight (A3).
  - Placed by: "**Consumed by** the compilation's router and its scheduler"
    design/integration-finishing-change-request.md:53

### schedule: [page-queue-change-request.md](page-queue-change-request.md)

- **The operator's statements: Relays** (design/page-queue-change-request.md:18-21)
  > routing within tb5 linked collectives is supported; attempting to involve nodes with no tb5 dma connection to any
  > node > within the collective is not allowed. do not expect for every node to be directly connected to all other
  > nodes > without the use of other nodes as relays; this emerges from the extremely small limit on the number of
  > inbound links > per studio or mini (6,3)
  - Placed by: "### The schedule that writes the page lists" design/page-queue-change-request.md:207; "each
    intermediate node's receive into a page of its own region, and its send of that page"
    design/page-queue-change-request.md:219; the operator: "What role does the bridge now implicitly or explicitly >
    have as a router" design/page-queue-change-request.md:74-75
- **The operator's statements: Interleaving every transfer** (design/page-queue-change-request.md:27-30)
  > this, btw, is one of the things we really ogtta do... the point of having aot schedulign and compilation of
  > schedules > is to allow us to interleave any combination of transport requirements concurrently with one another,
  > s.t. we are > never underusing the bandiwdth on any cable routing when we could counterfactually swizzle together
  > more > communications to cause pipeline-blocking operations to begin or conclude faster, no matter waht they are!
  - Placed by: "the point of having aot schedulign and compilation of schedules"
    design/page-queue-change-request.md:27
- **Claude's proposal: The page lists (the queue): the data structure** (design/page-queue-change-request.md:192-194):
  stated under scheduled value, which its clause names with this stage.
- **Claude's proposal: The page lists: Produced by** (design/page-queue-change-request.md:203-204)
  > **Produced by** the collective's function that computes a program's static schedule (the next section), in
  > `mesh-collective.{c,h}`.
  - Placed by: "**Produced by** the collective's function that computes a program's static schedule (the next
    section), in `mesh-collective.{c,h}`." design/page-queue-change-request.md:203-204
- **Claude's proposal: The schedule that writes the page lists: the algorithm**
  (design/page-queue-change-request.md:209-211)
  > **The algorithm** is the ported concurrent-flow solve (Garg–Könemann, as the statement under "Interleaving every
  > transfer" names it), run ahead of time over every transfer of a program together: every collective call and
  > pipeline hand-off, over the map's Thunderbolt-5-connected nodes, for any node count.
  - Placed by: "### The schedule that writes the page lists" design/page-queue-change-request.md:207; "the
    collective's function that computes a program's static schedule (the next section), in `mesh-collective.{c,h}`"
    design/page-queue-change-request.md:203-204
- **Claude's proposal: The schedule: blocking transfers first** (design/page-queue-change-request.md:211-212)
  > It puts the transfers that block a pipeline stage first.
  - Placed by: "### The schedule that writes the page lists" design/page-queue-change-request.md:207
- **Claude's proposal: The schedule: It takes** (design/page-queue-change-request.md:213-215)
  > **It takes** the program's declared transfers (operand ranges and destination ranks) and, from the map, node-level
  > adjacency and each node pair's capacity, the sum of that pair's cables. It takes no cable, queue pair or link
  > identity.
  - Placed by: "### The schedule that writes the page lists" design/page-queue-change-request.md:207
- **Claude's proposal: The schedule: It writes** (design/page-queue-change-request.md:217-221)
  > **It writes** page-list entries on every node a page passes through: - the source node's sends; - each
  > intermediate node's receive into a page of its own region, and its send of that page, whose `word` is the one that
  > receive's completion stores; - the destination's receive into the destination page.
  - Placed by: "### The schedule that writes the page lists" design/page-queue-change-request.md:207; "**It writes**
    page-list entries on every node a page passes through:" design/page-queue-change-request.md:217
- **Claude's proposal: The schedule: every live node runs a rank** (design/page-queue-change-request.md:222-223):
  stated under caller, which its clause names with this stage.
- **Claude's proposal: The assignment: It takes** (design/page-queue-change-request.md:231-234): stated under bridge,
  which its clause names with this stage.
- **Claude's proposal: The changes, by file: The collective: one function**
  (design/page-queue-change-request.md:251-252)
  > one function takes a program's declared transfers (operand ranges and destination ranks) and the map, and writes
  > the page lists;
  - Placed by: "**The collective** (`mesh-collective.{c,h}`, `mesh-dataflow.h`, `mesh-call.h`):"
    design/page-queue-change-request.md:250; "the collective's function that computes a program's static schedule (the
    next section), in `mesh-collective.{c,h}`" design/page-queue-change-request.md:203-204
- **Claude's proposal: The changes, by file: The collective: replaced calls**
  (design/page-queue-change-request.md:253-254)
  > `mesh_peer_channel`, `mesh_publish_bind`'s queue argument, `mesh_transfer_bind`, `mesh_link_map`,
  > `mesh_link_map_read`, `mesh_allreduce_plan` and `mesh_allreduce_bind` are replaced by that function.
  - Placed by: "**The collective** (`mesh-collective.{c,h}`, `mesh-dataflow.h`, `mesh-call.h`):"
    design/page-queue-change-request.md:250; "are replaced by that function." design/page-queue-change-request.md:254
- **Claude's proposal: Order, bullet 1** (design/page-queue-change-request.md:265-267): stated under bridge, which its
  clause names with this stage.

## argument_buffers

`argument_buffers`: the buffers on the edges into each processor's function.

### argument_buffers: [torch-capture-change-request.md](torch-capture-change-request.md)

- **2, "It takes / It writes" (the IPC graph is a function of the static schedule)**
  (design/torch-capture-change-request.md:460-464)
  > In rdma/mesh-graph.h at mesh 57d9f06e, `mesh_graph_cycle_mean` writes λ_max as a `double`, and
  > `mesh_graph_buffer_bound` writes Bff(e) as `uint64_t bound[]`; rdma/mesh-collective.c calls it. The IPC graph is a
  > function of the static schedule: in the reference program, `schedule.argument_buffers` takes `schedule.schedule`'s
  > graphs, classes each processor's arguments edge as feedback or feedforward by the same paper, and writes
  > `ir.Scheduled.feedforward`, which `runtime.Program` reads.
  - Placed by: "in the reference program, `schedule.argument_buffers` takes `schedule.schedule`'s graphs, classes each
    processor's arguments edge as feedback or feedforward by the same paper, and writes `ir.Scheduled.feedforward`,
    which `runtime.Program` reads." (design/torch-capture-change-request.md:462-464)
- **2, Write (when execution t's requests into a location start)** (design/torch-capture-change-request.md:474-475)
  > execution t's requests into one of a value's u locations start once execution t − u's use of that location has
  > ended: Lam's reuse of "the same location in non-overlapping iterations", in the form of (1).
  - Placed by: "The IPC graph is a function of the static schedule: in the reference program,
    `schedule.argument_buffers` takes `schedule.schedule`'s graphs, classes each processor's arguments edge as
    feedback or feedforward by the same paper, and writes `ir.Scheduled.feedforward`, which `runtime.Program` reads."
    (design/torch-capture-change-request.md:462-464)
- **2, "The number of locations of a value the CPU writes", Write** (design/torch-capture-change-request.md:492-497)
  > the CPU's write of a value for execution t is src(e) and the GPU's reads of it snk(e) of a feedforward edge e of
  > the node's IPC graph, the CPU and the GPU being two of its processors (each a cycle with one delay, Bhattacharyya,
  > Sriram and Lee, preprint p. 9). The value's number of locations is Bff(e): the smallest m for which adding
  > d_m(snk(e), src(e)) leaves λ_max unchanged, found by the binary search from (6)'s value, with t(v) the execution
  > time estimates. This is Lam's ⌈l/s⌉, l being the execution time along the cycle the reverse edge closes and s
  > being λ_max.
  - Placed by: "The IPC graph is a function of the static schedule: in the reference program,
    `schedule.argument_buffers` takes `schedule.schedule`'s graphs, classes each processor's arguments edge as
    feedback or feedforward by the same paper, and writes `ir.Scheduled.feedforward`, which `runtime.Program` reads."
    (design/torch-capture-change-request.md:462-464); "In rdma/mesh-graph.h at mesh 57d9f06e, `mesh_graph_cycle_mean`
    writes λ_max as a `double`, and `mesh_graph_buffer_bound` writes Bff(e) as `uint64_t bound[]`;
    rdma/mesh-collective.c calls it." (design/torch-capture-change-request.md:460-461)
- **Source for item 2's execution-time estimates, Write, third and fourth sentences (estimates stored; Bff supplies
  locations and reuse edges)** (design/torch-capture-change-request.md:874-875): stated under lower, which its clause
  names with this stage.

## scheduled value

The type `Scheduled` and the shared headers' records: what is carried between stages.

### scheduled value: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **A2 produced by, consumed by** (design/aot-compilation-change-request.md:177-181): stated under place, which its
  clause names with this stage.
- **A3 produced by, consumed by (the caller passes the operand; mesh's type)**
  (design/aot-compilation-change-request.md:202-204): stated under caller, which its clause names with this stage.
- **A4 produced by, consumed by (mesh_plan's types)** (design/aot-compilation-change-request.md:252-254): stated under
  schedule, which its clause names with this stage.
- **A5 produced by, consumed by** (design/aot-compilation-change-request.md:282, 292-293): stated under schedule,
  which its clause names with this stage.
- **A6 produced by, consumed by (the lists)** (design/aot-compilation-change-request.md:306-307): stated under
  schedule, which its clause names with this stage.
- **A6 one schedule per function; the schedule takes A8's IR** (design/aot-compilation-change-request.md:94-95, 296,
  306): stated under schedule, which its clause names with this stage.
- **A7 stage** (design/aot-compilation-change-request.md:319-319): stated under Program, which its clause names with
  this stage.
- **A8 data structure (the type mesh_plan takes)** (design/aot-compilation-change-request.md:502-512)
  > **The data structure is the task graph** of design/plan-to-bridge-change-request.md R3 (Sinnen, 2007), carried as
  > one IR from partitioning to serialization, as an operand of its own. With it: - each consumer's kind (R4: a join
  > for a synchronous, complete, contiguous consumer, a task per page for an asynchronous, incomplete, noncontiguous
  > consumer); - the placement operand (A2) and the weighting operand (A3); - A10's output. - **No algorithm of its
  > own.** MSCCLang carries its instruction DAG the same way, from lowering to scheduling (Cowan et al., 2023, §4 and
  > §5). - **Produced by, consumed by.** A2 partitions it. `mesh_plan` takes it for A4's routing and lowering, A5's
  > allocator and A6's ordering, as `struct mesh_function` with its `struct mesh_declared`, `struct mesh_weight` and
  > `struct mesh_dependence` (rdma/mesh-collective.h).
  - Placed by: "**Stage:** the type that passes from placement to the schedule."
    (design/aot-compilation-change-request.md:501)
- **A8 carried as one IR from partitioning to serialization** (design/aot-compilation-change-request.md:502-503,
  510-512)
  > **The data structure is the task graph** of design/plan-to-bridge-change-request.md R3 (Sinnen, 2007), carried as
  > one IR from partitioning to serialization, as an operand of its own. [...] **Produced by, consumed by.** A2
  > partitions it. `mesh_plan` takes it for A4's routing and lowering, A5's allocator and A6's ordering, as `struct
  > mesh_function` with its `struct mesh_declared`, `struct mesh_weight` and `struct mesh_dependence`
  > (rdma/mesh-collective.h).
  - Placed by: "**Stage:** the type that passes from placement to the schedule."
    (design/aot-compilation-change-request.md:501)
- **A9 stage and types** (design/aot-compilation-change-request.md:520-523)
  > **Stage:** the scheduled value and its archive. The schedule (A5, A6) produces it. The bridge and the GPU's
  > program take it. The reference program's type for the scheduled value is `ir.Scheduled`. Here it is `struct
  > mesh_algo`, the published table, with one `struct mesh_state` for each static schedule, which holds the `struct
  > mesh_hop` lists (rdma/mesh.h). `mesh_plan_save` and `mesh_plan_load` write and read it (rdma/mesh-collective.h).
  - Placed by: "**Stage:** the scheduled value and its archive." (design/aot-compilation-change-request.md:520)
- **A9 here, per node: the lists** (design/aot-compilation-change-request.md:528-532)
  > **Here, per node:** - for each adjacent node and direction, the ordered list of page instructions, each with its
  > page, its bytes and its dependency: the counter or phase field it waits on or writes. A send waits on its range's
  > supplied counter, or for a relay's forward on its receive's phase field; a receive writes its phase field and
  > raises its count. These are the static schedule's struct mesh_hop lists (design/plan-to-bridge-change-request.md
  > R5);
  - Placed by: "**Stage:** the scheduled value and its archive." (design/aot-compilation-change-request.md:520)
- **A9 here, per node: the bindings and the state section** (design/aot-compilation-change-request.md:533-533)
  > the page table's bindings and the static schedule's state section (C3);
  - Placed by: "**Stage:** the scheduled value and its archive." (design/aot-compilation-change-request.md:520)
- **A9 here, per node: the GPU's program** (design/aot-compilation-change-request.md:534-535)
  > the GPU's program: the function's own existing kernels, bound to their placed pages. No kernel is written by the
  > compilation. For a delegate's program these are the recorder's command buffers.
  - Placed by: "**Stage:** the scheduled value and its archive. The schedule (A5, A6) produces it. The bridge and the
    GPU's program take it." (design/aot-compilation-change-request.md:520-521)
- **A9 nothing else is serialized** (design/aot-compilation-change-request.md:536-537)
  > The bridge consumes the lists and the GPU runs the existing kernels. Nothing else is serialized but each entry's
  > `set` (appendix item 5).
  - Placed by: "**Stage:** the scheduled value and its archive." (design/aot-compilation-change-request.md:520)
- **A9 the algorithm the bridge runs it by: the self-timed strategy** (design/aot-compilation-change-request.md:538,
  546-565): stated under bridge, which its clause names with this stage.
- **A9 every address is fixed at compile time** (design/aot-compilation-change-request.md:569-574)
  > **Every address in it is fixed at compile time**, as memory planning fixes them: - "a memory planning pass is run
  > before the graph is saved to flatbuffer. This assigns a memory ID to each tensor and an offset in the buffer,
  > marking where storage for the tensor starts" (ExecuTorch, "Concepts"); - CUDA graph trees "add invariants to
  > ensure that memory is always in the same location as it were recorded" (PyTorch v2.11.0, docs, "CUDAGraph Trees").
  > So the bridge reads the serialized operand at a place its compilation fixed, and searches for nothing.
  - Placed by: "**Stage:** the scheduled value and its archive. The schedule (A5, A6) produces it."
    (design/aot-compilation-change-request.md:520)
- **A10 its output (the receiving types)** (design/aot-compilation-change-request.md:629-634, 638-639)
  > **Its output:** - for each compiled function, the most calls of it the analysis lets be outstanding at once in the
  > program that submits them, which bounds how many the program issues ahead; - the order constraints between calls,
  > which the program keeps when it submits, so the sequencer's first-in, first-out numbers keep them too (A7); - both
  > carried in A8's IR. [...] `mesh_plan` takes the order constraints as its `orders` argument, an array of
  > `uint32_t`.
  - Placed by: The clause's own "both carried in A8's IR" (design/aot-compilation-change-request.md:634); "The
    schedule and the runtime take the analysis's output, carried in A8's IR."
    (design/aot-compilation-change-request.md:591-592)
- **A10 produced by, consumed by (the schedule)** (design/aot-compilation-change-request.md:635, 637-638): stated
  under schedule, which its clause names with this stage.
- **Appendix item 4 (no truncation at one end; no handler)** (design/aot-compilation-change-request.md:882-884, 892)
  > **The algorithm is publication by a release store, read by an acquire load** (H.-J. Boehm and S. V. Adve,
  > Foundations of the C++ concurrency memory model, PLDI 2008). A static schedule is published whole, by a release
  > store after all of its writes (C3's writing order). The bridge never sees a static schedule in part. [...] **The
  > change:** the code holds no handler for a truncation in data or collective code.
  - Placed by: "**Stage:** the scheduled value's publication to the runtime."
    (design/aot-compilation-change-request.md:881); "**Produced by, consumed by.** The schedule stage writes the
    static schedule and stores last. The bridge loads it." (design/aot-compilation-change-request.md:889)
- **Appendix item 5: the set** (design/aot-compilation-change-request.md:908-912, 927-929): stated under schedule,
  which its clause names with this stage.
- **Appendix item 6: the compiled function's name; alike at every rank**
  (design/aot-compilation-change-request.md:1006-1007, 1026)
  > **The data structure is the compiled function's name:** (the program's group, the function's ordinal in the
  > program's order of compilation), the same at every rank. [...] name each compiled function alike at every rank, by
  > the name above.
  - Placed by: "**Stage:** the scheduled value and the runtime." (design/aot-compilation-change-request.md:1004); "The
    schedule stage writes one static schedule for each compiled function under its name."
    (design/aot-compilation-change-request.md:1016-1017)
- **Appendix item 6, change: any number of compiled functions at once (mesh_plan)**
  (design/aot-compilation-change-request.md:1021-1021)
  > let a program hold any number of compiled functions at once, each with its own static schedule;
  - Placed by: "**Stage:** the scheduled value and the runtime. A program holds one scheduled value for each compiled
    function, and the runtime calls each." (design/aot-compilation-change-request.md:1004-1005); "The schedule stage
    writes one static schedule for each compiled function under its name."
    (design/aot-compilation-change-request.md:1016-1017)
- **Appendix item 6, stage: a program holds one scheduled value for each compiled function (the compile chain)**
  (design/aot-compilation-change-request.md:1004-1005)
  > **Stage:** the scheduled value and the runtime. A program holds one scheduled value for each compiled function,
  > and the runtime calls each.
  - Placed by: "**Stage:** the scheduled value and the runtime." (design/aot-compilation-change-request.md:1004)
- **Appendix item 6, change: one allocation over the whole program; deferred reclamation**
  (design/aot-compilation-change-request.md:1022-1024): stated under schedule, which its clause names with this stage.

### scheduled value: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **The data flow (stage table and its closing sentence)** (design/compilation-composition-change-request.md:35-36)
  > There is one schedule. Lowering, the runtime and the archive each take the scheduled value and nothing else from
  > the stages before it.
  - Placed by: "| the scheduled value | `ir.py` `Scheduled` | the results above | one value: `exported`, `modules`,
    `devices`, `output_processor`, `channels`, `inputs`, `feedforward` |"
    design/compilation-composition-change-request.md:30
- **1 / Write: every rank, its collectives, each placed CPU or GPU function**
  (design/compilation-composition-change-request.md:115-122): stated under lower, which its clause names with this
  stage.
- **2 / data structure: what each edge carries** (design/compilation-composition-change-request.md:128-140)
  > - **The data structure** is one graph for computation and data movement together (S3's parallel computation
  > graph): its nodes are function calls and, for each value read on another processor, a `send` and a `recv`; its
  > edges are values, each with its sharding, consumer kind, accesses and lifetimes. - **The algorithm** is graph
  > substitution (S3): a transformation replaces a subgraph of that graph with another. - **Composes:** export
  > produces the graph. Placement adds the `send` and `recv` nodes to it. One schedule orders functions and transfers
  > together and produces the scheduled value. Lowering, the runtime and the archive take the scheduled value. The
  > signature is printed from what lowering emits. - **Write:** A value's sharding, consumer kind, accesses,
  > lifetimes, transfers and dependences describe the same program through lowering. Numerical decomposition, fusion
  > and communication transformations keep the information needed by subsequent scheduling, allocation and
  > synchronization. The output of each analysis or lowering is the operand its consumers actually use. Changes to the
  > graph are reflected in those dependent results before emission. The signature describes that emitted program. Each
  > row's work is completed in this composition, including its specified deletion or replacement.
  - Placed by: the reader.
- **2 / Composes: one schedule orders functions and transfers together**
  (design/compilation-composition-change-request.md:132-134): stated under schedule, which its clause names with this
  stage.
- **2 / Write: transformations keep information; outputs are operands; changes reflected**
  (design/compilation-composition-change-request.md:136-139)
  > Numerical decomposition, fusion and communication transformations keep the information needed by subsequent
  > scheduling, allocation and synchronization. The output of each analysis or lowering is the operand its consumers
  > actually use. Changes to the graph are reflected in those dependent results before emission.
  - Placed by: the reader.
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **3 / Write, second bullet: callers call them, own copies deleted**
  (design/compilation-composition-change-request.md:150-157): stated under graph module, which its clause names with
  this stage.
- **4 / Write: placement and costs are operands** (design/compilation-composition-change-request.md:170-173): stated
  under caller, which its clause names with this stage.
- **4 / Write: neither execution nor the bridge computes its own order or placement**
  (design/compilation-composition-change-request.md:175-176): stated under Program, which its clause names with this
  stage.
- **5 / data structure and Composes: one compiled native program per node**
  (design/compilation-composition-change-request.md:196-202): stated under lower, which its clause names with this
  stage.
- **5 / Write: producer and consumer agree at both ends** (design/compilation-composition-change-request.md:218-220)
  > Producer and consumer change in the same revision: page groups and sent counters, native result bindings, and
  > signature identities agree at both ends.
  - Placed by: the reader.
- **6 / Replaces 2: schedule results are operands of native preparation and of the transport**
  (design/compilation-composition-change-request.md:250-252)
  > The schedule's task identities, its dependence and order results, and the transformation that produced each are
  > compilation operands of native preparation; the transport's runtime synchronization reads the same results.
  - Placed by: "`mesh_plan` computes the calls' dependency relation and synchronization, but `MetalRecorderProgram`
    receives call handles and reconstructs its ordering inputs from buffer footprints."
    design/compilation-composition-change-request.md:248-249
- **6 / Write: connections finished; identity mapping retained**
  (design/compilation-composition-change-request.md:259-263)
  > finish each producer → representation → consumer connection before extracting auxiliary helpers. Distinguish a
  > genuinely new lower-level constraint from a repeated derivation of an existing one, and retain the identity
  > mapping through lowering. Change the producer, interface, consumer and displaced reconstruction together. An
  > unused new field, a new helper without a consumer, a second graph beside an ignored first graph, or diagnostic
  > output alone does not finish a connection.
  - Placed by: the reader.
- **Correction: generated target source / Write: an edge carries its cause**
  (design/compilation-composition-change-request.md:312-314)
  > A dependence edge of the schedule carries the value/effect/transfer relation that makes it necessary until the
  > lowering that reads it has represented that obligation in the target execution.
  - Placed by: "A dependence edge of the schedule carries the value/effect/transfer relation that makes it necessary
    until the lowering that reads it has represented that obligation in the target execution."
    design/compilation-composition-change-request.md:312-314
- **Performance correction / Emission** (design/compilation-composition-change-request.md:333-356): stated under
  lower, which its clause names with this stage.
- **Complete replacement / operand table row 4: Program relations**
  (design/compilation-composition-change-request.md:428-428)
  > | Program relations | Call, dependence and collective-schedule results → remapped relations through fusion,
  > splitting and placement → the shared graph algorithms and native program. Each edge carries the
  > value/effect/transfer that causes it. Update affected results when transformation changes their operands. |
  - Placed by: "Call, dependence and collective-schedule results → remapped relations through fusion, splitting and
    placement → the shared graph algorithms and native program." design/compilation-composition-change-request.md:428
- **7 / data structure: the scheduled value** (design/compilation-composition-change-request.md:495-501)
  > - **The data structure** is the scheduled value: typed numerical IR with its dependencies, placement, storage and
  > communication annotations, and requirement 4's static schedule. Analysis solves the abstract problem once and
  > supplies a lowerable program to its optimizing consumers. Typed numerical IR, dependencies, placement, storage and
  > communication annotations stay available until their last compiler consumer. Each may then be realized in
  > instructions, addresses, target commands and synchronization without preserving its original runtime encoding. A
  > property is preserved through the meaning of the generated program, whether its descriptive metadata is erased or
  > retained separately for loading, inspection or the required signature.
  - Placed by: "**The data structure** is the scheduled value" design/compilation-composition-change-request.md:495
- **7 / Composes: the archive stores the scheduled value** (design/compilation-composition-change-request.md:505-507):
  stated under save and load, which its clause names with this stage.
- **7 / Composes: lowering emits the native program, the runtime calls it**
  (design/compilation-composition-change-request.md:505-507): stated under lower, which its clause names with this
  stage.

### scheduled value: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **A3 "What it writes"** (design/disjoint-pages-change-request.md:467-467): stated under schedule, which its clause
  names with this stage.
- **A3 "Which function computes it, and which read it"** (design/disjoint-pages-change-request.md:469-474): stated
  under schedule, which its clause names with this stage.
- **A4 deleted (3)** (design/disjoint-pages-change-request.md:484-484): stated under bridge, which its clause names
  with this stage.
- **C3.1 data structure** (design/disjoint-pages-change-request.md:807-809)
  > **The data structure** is the array of phases (C3.7): one 2-bit field per received page, contiguous, in the state
  > section. With it are one count per received range, one threadgroup count per later pass, and one status word per
  > execution.
  - Placed by: "what it writes (`rdma/mesh.h`): ... and the state section, a contiguous section of the region apart
    from the data pages, headed by `struct mesh_state` (`copies`; the offsets `phases`, `counts`, `supplied` and
    `status`; `executions`), with the status a `struct mesh_status` `{behind, ahead, first}`"
    (design/disjoint-pages-change-request.md:391-395; the ellipsis stands for the page table and transfer-entry
    clauses of the same sentence); "`mesh_plan`'s allocator places the array and the words and writes their starting
    values." (design/disjoint-pages-change-request.md:906)
- **C3.1 "Written by, read by"** (design/disjoint-pages-change-request.md:825-829): stated under lower, which its
  clause names with this stage.
- **C3.2 (a) data structure and types** (design/disjoint-pages-change-request.md:834-836): stated under schedule,
  which its clause names with this stage.
- **C3.3 "Written by, read by"** (design/disjoint-pages-change-request.md:852-854): stated under schedule, which its
  clause names with this stage.
- **C3.4 "Written by, read by"** (design/disjoint-pages-change-request.md:867-868): stated under schedule, which its
  clause names with this stage.
- **C3.7 data structures** (design/disjoint-pages-change-request.md:890-894)
  > **The data structures**, all in the state section, apart from the data pages: - the array of phases: one 2-bit
  > field for each receive in the static schedule's list (a received page of one value's copy), Gray-coded 00 → 01 →
  > 11 → 10. It is PTX mbarrier's phase bit, widened to two; - one 64-bit word per sent range; - one count per
  > received range, and the command queue's event.
  - Placed by: "The array, the counts and the words are at `struct mesh_state`'s offsets `phases`, `counts` and
    `supplied`; a receive's field is its `struct mesh_hop`'s `field`, and a send's word its `word` (`rdma/mesh.h`)."
    (design/disjoint-pages-change-request.md:908-910); "`mesh_plan`'s allocator places the array and the words and
    writes their starting values." (design/disjoint-pages-change-request.md:906)
- **C3.7 "Written by, read by"** (design/disjoint-pages-change-request.md:906-910): stated under schedule, which its
  clause names with this stage.
- **C4 mesh deleted: rdma/mesh.h areas and constants** (design/disjoint-pages-change-request.md:925-926)
  > rdma/mesh.h: the supplied words' and landed list's fixed areas of the layout (hdr `word_off`, `landed_off`); the
  > landed record's ORed bits and its `UINT64_MAX` cancel word; the constant `MESH_SUPPLIED`.
  - Placed by: "rdma/mesh.h: the supplied words' and landed list's fixed areas of the layout"
    (design/disjoint-pages-change-request.md:925); "what it writes (`rdma/mesh.h`)"
    (design/disjoint-pages-change-request.md:391)
- **D.3.2** (design/disjoint-pages-change-request.md:1386-1395): stated under schedule, which its clause names with
  this stage.
- **D.3.5** (design/disjoint-pages-change-request.md:1425-1431): stated under bridge, which its clause names with this
  stage.
- **D.3.6** (design/disjoint-pages-change-request.md:1434-1445): stated under bridge, which its clause names with this
  stage.

### scheduled value: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R1 data structure** (design/plan-to-bridge-change-request.md:1949-1950, 1974-1975)
  > - **The data structure is a list of blocks.** A transfer is a list of byte ranges of its source's pages, landing
  > in byte ranges of its destination's pages. Each range is one block: (page-table row, offset, bytes). [...] -
  > Writes: the blocks, as the entries of each cable's list (R5): `struct mesh_hop` `page`, `offset` and `bytes`
  > (rdma/mesh.h). These are MSCCL-IR's steps with offset and count (aot-compilation A9).
  - Placed by: "Writes: for each copy, each cable's sends in schedule order, each entry with its page and its bytes
    (`struct mesh_hop`, rdma/mesh.h). The lists are part of the scheduled value."
    design/plan-to-bridge-change-request.md:2356-2357; "Produced by the schedule, `mesh_plan`. Consumed by the bridge
    (rdma/mesh-flow.c)" design/plan-to-bridge-change-request.md:1976
- **R2 (requirement)** (design/plan-to-bridge-change-request.md:2013-2013): stated under schedule, which its clause
  names with this stage.
- **R2 data structure** (design/plan-to-bridge-change-request.md:2014-2016)
  > **The data structure is the page table:** names bound to frames (T. Kilburn, D. B. G. Edwards, M. J. Lanigan and
  > F. H. Sumner, One-level storage system, IRE Trans. Electronic Computers, 1962; P. J. Denning, Virtual memory, ACM
  > Comput. Surv., 1970). A value is its ordered list of page-table rows, and its pages are placed anywhere.
  - Placed by: "The rows are part of the scheduled value." design/plan-to-bridge-change-request.md:2119; "Produced
    when the static schedule is made: `mesh_plan` (rdma/mesh-collective.c)"
    design/plan-to-bridge-change-request.md:2118
- **R2 a run is a mapping, not a placement** (design/plan-to-bridge-change-request.md:2069-2074): stated under
  schedule, which its clause names with this stage.
- **R2 where it composes** (design/plan-to-bridge-change-request.md:2112-2121): stated under schedule, which its
  clause names with this stage.
- **R4 behaviour at run time, 4** (design/plan-to-bridge-change-request.md:2194-2194): stated under schedule, which
  its clause names with this stage.
- **R5 where it composes** (design/plan-to-bridge-change-request.md:2351-2362): stated under schedule, which its
  clause names with this stage.
- **R7 where it composes** (design/plan-to-bridge-change-request.md:2410-2411): stated under bridge, which its clause
  names with this stage.
- **R8 where it composes: the pages are part of the scheduled value**
  (design/plan-to-bridge-change-request.md:2440-2440)
  > The pages are R2's rows, written when the static schedule is made and part of the scheduled value.
  - Placed by: "The pages are R2's rows, written when the static schedule is made and part of the scheduled value."
    design/plan-to-bridge-change-request.md:2440

### scheduled value: [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

- **1, step 1** (design/problems-of-14b9a757-change-request.md:125-126): stated under place, which its clause names
  with this stage.
- **3, Change: the rule** (design/problems-of-14b9a757-change-request.md:190-194): stated under place, which its
  clause names with this stage.
- **7.1, step 1** (design/problems-of-14b9a757-change-request.md:317-319): stated under schedule, which its clause
  names with this stage.
- **7.1, step 2** (design/problems-of-14b9a757-change-request.md:320-322): stated under schedule, which its clause
  names with this stage.
- **7.1, step 5** (design/problems-of-14b9a757-change-request.md:325-326): stated under schedule, which its clause
  names with this stage.
- **7.3, step 1** (design/problems-of-14b9a757-change-request.md:378-380): stated under schedule, which its clause
  names with this stage.
- **8, step 4** (design/problems-of-14b9a757-change-request.md:425-426): stated under place, which its clause names
  with this stage.
- **8, Change: LOADED and the execution UINT64_MAX − 1 are deleted**
  (design/problems-of-14b9a757-change-request.md:454-454): stated under schedule, which its clause names with this
  stage.
- **9** (design/problems-of-14b9a757-change-request.md:460-472): stated under schedule, which its clause names with
  this stage.
- **10, Change: the representation** (design/problems-of-14b9a757-change-request.md:476-477)
  > Here a tensor view is `dims` followed by `dims` (size, stride) pairs, with no fixed bound.
  - Placed by: "the placement stage writes it for each transfer's end, as `struct mesh_strided` in `mesh_plan`'s
    operand. `mesh_plan` reads it." design/problems-of-14b9a757-change-request.md:479-480; "none is run; it is a
    representation." design/problems-of-14b9a757-change-request.md:478
- **10, Composition: the placement stage writes it** (design/problems-of-14b9a757-change-request.md:479-480): stated
  under place, which its clause names with this stage.
- **10, Change: the refusal is deleted** (design/problems-of-14b9a757-change-request.md:487-487): stated under place,
  which its clause names with this stage.
- **12, step 2** (design/problems-of-14b9a757-change-request.md:527-527): stated under schedule, which its clause
  names with this stage.

### scheduled value: [torch-capture-change-request.md](torch-capture-change-request.md)

- **2, "It takes / It writes" (the IPC graph is a function of the static schedule)**
  (design/torch-capture-change-request.md:460-464): stated under argument_buffers, which its clause names with this
  stage.

### scheduled value: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **1 / Produced by** (design/synchronization-signature-change-request.md:100-101): stated under lower, which its
  clause names with this stage.

### scheduled value: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **3 / Write: one schedule stage** (design/torch-source-lowering-change-request.md:241-243): stated under schedule,
  which its clause names with this stage.

### scheduled value: [page-queue-change-request.md](page-queue-change-request.md)

- **Claude's proposal: The page lists (the queue): the data structure** (design/page-queue-change-request.md:192-194)
  > **The data structure** is a list of entries held in each node's region: the zero-copy queue of the operator's
  > statement under "The page queue". An entry is one transfer of one page between this node and an adjacent node: `{
  > page, node, direction, word, position }`.
  - Placed by: "**Produced by** the collective's function that computes a program's static schedule (the next
    section), in `mesh-collective.{c,h}`. **Consumed by** the bridge alone, in `mesh-flow.c`."
    design/page-queue-change-request.md:203-204
- **Claude's proposal: The page lists: page, node, direction** (design/page-queue-change-request.md:195-197)
  > `page`: a page of this node's region: the page sent, or the page a receive lands in. - `node`: the adjacent node,
  > by its index in the map: the node a send goes to, or the node a receive comes from. - `direction`: send or
  > receive.
  - Placed by: "**Produced by** the collective's function that computes a program's static schedule (the next
    section), in `mesh-collective.{c,h}`. **Consumed by** the bridge alone, in `mesh-flow.c`."
    design/page-queue-change-request.md:203-204
- **Claude's proposal: The page lists: word, position** (design/page-queue-change-request.md:198-200)
  > `word`, `position`: a word of the region and a value. A send is posted once its word reads its value. The page's
  > producer stores that value, or, at an intermediate node, the completion of the receive of that page stores it. A
  > receive's completion stores its value in its word.
  - Placed by: "**Produced by** the collective's function that computes a program's static schedule (the next
    section), in `mesh-collective.{c,h}`. **Consumed by** the bridge alone, in `mesh-flow.c`."
    design/page-queue-change-request.md:203-204; "post each send once its `word` > reads its `position`, and store
    each completed receive's `position` in its `word`." design/page-queue-change-request.md:244-245 (the ">" is not in
    the file; lines 244-245 read "post each send once its `word`" / "reads its `position`, and store each completed
    receive's `position` in its `word`.")
- **Claude's proposal: The page lists: no copy, not a ring** (design/page-queue-change-request.md:201-202)
  > Entries refer to pages already in the region. Nothing is copied. The lists are not a ring: they are written once
  > for a compiled program and read again by every execution of it.
  - Placed by: "**Produced by** the collective's function that computes a program's static schedule (the next
    section), in `mesh-collective.{c,h}`. **Consumed by** the bridge alone, in `mesh-flow.c`."
    design/page-queue-change-request.md:203-204
- **Claude's proposal: The page lists: Replaces** (design/page-queue-change-request.md:204-205)
  > The lists replace the per-link, per-queue transfer lists (`mesh.h` `mesh_order_length`, the geometry's `links` ×
  > `qps`).
  - Placed by: "The bridge, mesh `rdma/`:" design/page-queue-change-request.md:171 with "`mesh.h`: the per-link,
    per-queue transfer lists (`mesh_order_length`, the geometry's `links` × `qps` lists);"
    design/page-queue-change-request.md:173
- **Claude's proposal: The schedule: It writes** (design/page-queue-change-request.md:217-221): stated under schedule,
  which its clause names with this stage.

## lower

`lower`: the scheduled value to each platform's compiler input and function.

### lower: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **Appendix item 7 (torch's existing kernels on the GPU)** (design/aot-compilation-change-request.md:1030-1031)
  > design/torch-source-lowering-change-request.md item 1 replaces this item: no torch Metal kernel is submitted or
  > recorded.
  - Placed by: "`lower_metal.lower` returns them without their copies to the Metal device, whose function takes host
    pages, so it is first (design/torch-source-lowering-change-request.md, item 1: "no torch Metal kernel is ever
    submitted or recorded again.")." (design/the-compiler.md:99-101)

### lower: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **The data flow (stage table and its closing sentence)** (design/compilation-composition-change-request.md:35-36):
  stated under scheduled value, which its clause names with this stage.
- **1 / algorithm (full conversion)** (design/compilation-composition-change-request.md:113-114)
  > **The algorithm** is full conversion (S2): every operation of the input is converted to the target, and the
  > conversion succeeds only when all of them are.
  - Placed by: "Lowering takes the scheduled value and converts every function in it: for every rank, its numerical
    functions, its collectives (the `send` and `recv` nodes placement added), and each placed CPU or GPU function."
    design/compilation-composition-change-request.md:115-117
- **1 / Write: every rank, its collectives, each placed CPU or GPU function**
  (design/compilation-composition-change-request.md:115-122)
  > - **Composes:** export produces both graphs. Lowering takes the scheduled value and converts every function in it:
  > for every rank, its numerical functions, its collectives (the `send` and `recv` nodes placement added), and each
  > placed CPU or GPU function. - **Replaces:** the split in `rdma/mesh_compile.py program()` where `replayed`
  > excludes nodes whose declarations name transferred values; also its rejection of repeated calls to a compiled
  > function in the `replayed` block. - **Write:** Compile both each numerical function and the graph of calls between
  > functions. Lowering covers every rank and its collectives, and each placed CPU or GPU function. An intermediate
  > conversion is completed before the program runs. A missing lowering within the assigned domain is unfinished work,
  > not a new restriction on that domain.
  - Placed by: "Lowering takes the scheduled value and converts every function in it"
    design/compilation-composition-change-request.md:115
- **1 / Write: conversion completed before the program runs**
  (design/compilation-composition-change-request.md:121-122)
  > An intermediate conversion is completed before the program runs.
  - Placed by: "Lowering takes the scheduled value and converts every function in it"
    design/compilation-composition-change-request.md:115
- **1 / Write: the whole assigned domain** (design/compilation-composition-change-request.md:122-122)
  > A missing lowering within the assigned domain is unfinished work, not a new restriction on that domain.
  - Placed by: "A missing lowering within the assigned domain is unfinished work"
    design/compilation-composition-change-request.md:122; heading "### 1. Finish lowering the whole assigned domain"
    design/compilation-composition-change-request.md:107
- **1 / Replaces** (design/compilation-composition-change-request.md:118-119)
  > **Replaces:** the split in `rdma/mesh_compile.py program()` where `replayed` excludes nodes whose declarations
  > name transferred values; also its rejection of repeated calls to a compiled function in the `replayed` block.
  - Placed by: "Lowering takes the scheduled value and converts every function in it"
    design/compilation-composition-change-request.md:115; the file named: "`rdma/mesh_compile.py program()`"
    design/compilation-composition-change-request.md:118
- **2 / data structure: what each edge carries** (design/compilation-composition-change-request.md:128-140): stated
  under scheduled value, which its clause names with this stage.
- **2 / Composes: one schedule orders functions and transfers together**
  (design/compilation-composition-change-request.md:132-134): stated under schedule, which its clause names with this
  stage.
- **2 / the signature** (design/compilation-composition-change-request.md:132-140)
  > - **Composes:** export produces the graph. Placement adds the `send` and `recv` nodes to it. One schedule orders
  > functions and transfers together and produces the scheduled value. Lowering, the runtime and the archive take the
  > scheduled value. The signature is printed from what lowering emits. - **Write:** A value's sharding, consumer
  > kind, accesses, lifetimes, transfers and dependences describe the same program through lowering. Numerical
  > decomposition, fusion and communication transformations keep the information needed by subsequent scheduling,
  > allocation and synchronization. The output of each analysis or lowering is the operand its consumers actually use.
  > Changes to the graph are reflected in those dependent results before emission. The signature describes that
  > emitted program. Each row's work is completed in this composition, including its specified deletion or
  > replacement.
  - Placed by: "The signature is printed from what lowering emits."
    design/compilation-composition-change-request.md:134
- **2 / Write: transformations keep information; outputs are operands; changes reflected**
  (design/compilation-composition-change-request.md:136-139): stated under scheduled value, which its clause names
  with this stage.
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **3 / Write, second bullet: callers call them, own copies deleted**
  (design/compilation-composition-change-request.md:150-157): stated under graph module, which its clause names with
  this stage.
- **4 / Write: signature and external observation cover this execution**
  (design/compilation-composition-change-request.md:176-177)
  > The required signature and its external observation cover this execution.
  - Placed by: "lowering → the signature; the monitor reads it outside the program"
    design/compilation-composition-change-request.md:103
- **5 / data structure and Composes: one compiled native program per node**
  (design/compilation-composition-change-request.md:196-202)
  > - **The data structure** is one compiled native program per node, which launches all CPU, ANE and MPS work of the
  > graph (the operator's first turn above). Numerical functions and their calls are compilation inputs; placements,
  > value instances and readiness are operands. CPU, ANE and MPS are resources of the Apple Silicon target. Python
  > capture and preparation are not its CPU execution. - **The algorithm** is specialization (S6): one subject program
  > and its static operands give the residual program. - **Composes:** lowering takes the scheduled value and emits
  > the native program; the runtime calls it. The signature is printed from that program by the graph module's
  > printers.
  - Placed by: "**Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it."
    design/compilation-composition-change-request.md:201
- **5 / Composes: signature printed by the graph module's printers**
  (design/compilation-composition-change-request.md:201-202): stated under graph module, which its clause names with
  this stage.
- **5 / Replaces, site 1** (design/compilation-composition-change-request.md:205-207)
  > `mesh_compile.py _outputs` and `_natives`: both traverse the same call graph for every value-copy residue and
  > remap the same captured results. `program.make` also rebuilds the same bound-input list. Specialization computes
  > native operands, result bindings and their storage once; their consumers read its result.
  - Placed by: "**Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it."
    design/compilation-composition-change-request.md:201
- **5 / Replaces, site 2** (design/compilation-composition-change-request.md:208-210)
  > `metal_recording.m MetalReplayEncoding` and `MetalReplaySelfTimed`: both construct, encode and commit commands
  > from the same recorded commands. `tools/e2b_replay.m` separately orchestrates prepared work. Compile the required
  > ordering once and consume it through the shared native program; delete displaced orchestration.
  - Placed by: "**Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it."
    design/compilation-composition-change-request.md:201
- **5 / Write: one definition of each repeated behavior** (design/compilation-composition-change-request.md:215-217)
  > each repeated behavior has one definition; where consumers share it, its closed-over inputs are explicit operands;
  > every caller calls it; the duplicate bodies and their dead helpers are deleted. Sharing a wrapper while two
  > implementations remain does not perform this replacement.
  - Placed by: "**Replaces:** duplicated closures or traversals derived from the same producer, duplicated
    representations and scheduling/lowering decisions, and execution bodies the native program replaces."
    design/compilation-composition-change-request.md:203-204
- **5 / Write: emitted program preserves domain, collectives, page placement, lifetimes**
  (design/compilation-composition-change-request.md:217-218)
  > The emitted program preserves the assigned numerical domain, collectives, page placement and lifetime semantics.
  - Placed by: "**Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it."
    design/compilation-composition-change-request.md:201
- **6 / Replaces 1** (design/compilation-composition-change-request.md:246-247): stated under graph module, which its
  clause names with this stage.
- **6 / Replaces 2: schedule results are operands of native preparation and of the transport**
  (design/compilation-composition-change-request.md:250-252): stated under scheduled value, which its clause names
  with this stage.
- **6 / Replaces 2: signature printed from the resulting compilation**
  (design/compilation-composition-change-request.md:252-252)
  > The signature is printed from the resulting compilation.
  - Placed by: "The signature is printed from what lowering emits."
    design/compilation-composition-change-request.md:134
- **6 / Replaces 3** (design/compilation-composition-change-request.md:253-255)
  > `MetalSchedule` reduces its graph to command levels and permutation. Preparation of the recorded commands then
  > constructs another dependence graph around the same commands and later passes. The one graph is remapped through
  > command transformation, gains the newly lowered vertices/edges, and is the operand of ordering and emission.
  - Placed by: "The one graph is remapped through command transformation, gains the newly lowered vertices/edges, and
    is the operand of ordering and emission." design/compilation-composition-change-request.md:254-255
- **6 / Write: connections finished; identity mapping retained**
  (design/compilation-composition-change-request.md:259-263): stated under scheduled value, which its clause names
  with this stage.
- **6 / Write: what the emitted program preserves** (design/compilation-composition-change-request.md:263-266)
  > The emitted program preserves independent calls, interleaved computation/transport, exact copy lifetimes, placed
  > CPU/MPS work, the full assigned numerical domain and the LiteRT path. A total order, waits, whole-buffer effects
  > or another runtime do not replace them.
  - Placed by: "**Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it."
    design/compilation-composition-change-request.md:201
- **Correction: generated target source / Composes: both frontends at export**
  (design/compilation-composition-change-request.md:302-303): stated under export, which its clause names with this
  stage.
- **Correction: generated target source / Composes and Emission boundary**
  (design/compilation-composition-change-request.md:302-319)
  > - **Composes:** the Torch and LiteRT compiler interfaces produce the operations and their annotations at export.
  > Placement, the schedule and lowering take and return them. Lowering emits MSL or native target source, which the
  > target compiler takes. No function of mesh takes that source. - **Replaces:** recorder transformations whose
  > semantic operands are shader strings: scalar/quantization specialization, operand promotion, epilogue/map/row
  > fusion, reduction/packed-weight rewriting, coherent-store insertion and masked-pass construction. `MetalEntry`,
  > `MetalRowFootprint` and `MetalMaskedPass` are among them; replacing their parsers alone does not complete the
  > requirement. - **Write:** the producing Torch and LiteRT compiler interfaces supply the operation, binding, type,
  > shape, indexing, effect, alias and transfer/generation facts required by their consumers. Lowerings transform
  > those representations, preserving or updating the corresponding identities, accesses and dependencies when they
  > fuse, split, specialize or rebind operations. Consumers query the resulting annotations and relations. A
  > dependence edge of the schedule carries the value/effect/transfer relation that makes it necessary until the
  > lowering that reads it has represented that obligation in the target execution. - **Emission boundary:** generated
  > MSL or native target source is the final source emission of the mesh lowering pipeline, passed to the target
  > compiler. Subsequent mesh transformations do not infer semantics by parsing or matching that output. Source
  > spelling, generated identifier names and pretty-printing are not optimization preconditions. Retaining tokens for
  > some consumers while the other passes continue rewriting emitted strings leaves this requirement incomplete.
  - Placed by: "Lowering emits MSL or native target source, which the target compiler takes."
    design/compilation-composition-change-request.md:303-304
- **Correction: generated target source / Replaces** (design/compilation-composition-change-request.md:305-308)
  > **Replaces:** recorder transformations whose semantic operands are shader strings: scalar/quantization
  > specialization, operand promotion, epilogue/map/row fusion, reduction/packed-weight rewriting, coherent-store
  > insertion and masked-pass construction. `MetalEntry`, `MetalRowFootprint` and `MetalMaskedPass` are among them;
  > replacing their parsers alone does not complete the requirement.
  - Placed by: "Lowering emits MSL or native target source, which the target compiler takes. No function of mesh takes
    that source." design/compilation-composition-change-request.md:303-304
- **Correction: generated target source / Write: frontends supply facts; lowerings preserve identities**
  (design/compilation-composition-change-request.md:309-312): stated under export, which its clause names with this
  stage.
- **Correction: generated target source / Write: an edge carries its cause**
  (design/compilation-composition-change-request.md:312-314): stated under scheduled value, which its clause names
  with this stage.
- **Correction: generated target source / Property preserved**
  (design/compilation-composition-change-request.md:320-321)
  > **Property preserved:** the emitted program has the numerical, fusion, communication and synchronization behavior
  > required by the cited requests; the bodies that implement it through text are replaced.
  - Placed by: "Lowering emits MSL or native target source, which the target compiler takes."
    design/compilation-composition-change-request.md:303-304
- **Performance correction / Emission** (design/compilation-composition-change-request.md:333-356)
  > - **Emission.** The data structure is Tint's IR module. Dawn/Tint's [Metal shader
  > translation](https://github.com/google/dawn/blob/main/src/dawn/native/metal/ShaderModuleMTL.mm) obtains a program,
  > converts it to an IR module and invokes its MSL writer. The
  > [writer](https://github.com/google/dawn/blob/main/src/tint/lang/msl/writer/writer.cc), `Generate`, transforms the
  > module with `Raise` before `Print`. - **Fusion and storage.** MLIR's [elementwise fusion
  > implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp)
  > composes operand indexing maps and remaps operation operands/results. Its
  > [bufferization](https://mlir.llvm.org/docs/Bufferization/) uses tensor use-def, alias and equivalence relations to
  > determine storage (S8). These supply fusion and intermediate-storage elimination, rather than rediscovery of those
  > relations from buffer names or shader text. - **Constants.** MLIR's [SCCP
  > pass](https://mlir.llvm.org/docs/Passes/#-sccp), based on Wegman and Zadeck (1991), propagates known constants
  > through IR. S6 supplies the specialization equation. Static operation, dtype, shape, stride and configuration
  > operands become the residual program's instructions/constants; generic shaders do not read them again at run time
  > merely because their representation fits in a metadata buffer. - **Reductions.** MLIR's [workgroup reduction
  > lowering](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Transforms/AllReduceLowering.cpp)
  > constructs subgroup reductions and workgroup exchanges from reduction operations. Apple's [MLX
  > reduction](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal), [packed
  > arithmetic](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h) and [system-scope
  > access](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal) provide concrete target
  > mechanisms. - **Composes:** each of these is a function from operations to operations inside lowering, which takes
  > the scheduled value; `Print` comes after all of them.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Fusion and storage** (design/compilation-composition-change-request.md:338-343)
  > **Fusion and storage.** MLIR's [elementwise fusion
  > implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp)
  > composes operand indexing maps and remaps operation operands/results. Its
  > [bufferization](https://mlir.llvm.org/docs/Bufferization/) uses tensor use-def, alias and equivalence relations to
  > determine storage (S8). These supply fusion and intermediate-storage elimination, rather than rediscovery of those
  > relations from buffer names or shader text.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Constants** (design/compilation-composition-change-request.md:344-366)
  > - **Constants.** MLIR's [SCCP pass](https://mlir.llvm.org/docs/Passes/#-sccp), based on Wegman and Zadeck (1991),
  > propagates known constants through IR. S6 supplies the specialization equation. Static operation, dtype, shape,
  > stride and configuration operands become the residual program's instructions/constants; generic shaders do not
  > read them again at run time merely because their representation fits in a metadata buffer. - **Reductions.**
  > MLIR's [workgroup reduction
  > lowering](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Transforms/AllReduceLowering.cpp)
  > constructs subgroup reductions and workgroup exchanges from reduction operations. Apple's [MLX
  > reduction](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal), [packed
  > arithmetic](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h) and [system-scope
  > access](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal) provide concrete target
  > mechanisms. - **Composes:** each of these is a function from operations to operations inside lowering, which takes
  > the scheduled value; `Print` comes after all of them. - **Replaces:** the preceding recorder bodies,
  > `MetalNativeProjection`/`MetalNativeArgmax`'s source-based identification and extraction, and
  > `_pointwise`/`_reduced`/`_contract`'s premature commitment to generic dispatches where the static program still
  > requires specialization, fusion and parallel scheduling. - **Write:** preserve the numerical operations and their
  > permitted precision/rounding, express maps and epilogues as operation regions, retain values across their uses,
  > and lower reductions and packed data to the selected target operations. Preserve each actual effect and observable
  > result while removing unnecessary materialization. Compile producer publication and consumer acquisition from the
  > same range/page dependencies used by transport scheduling. Operation equivalence and applicability follow the
  > structured operands and effects, not an entry-point name, generated variable name or operator spelling. Carry
  > static facts as compiler operands and keep genuinely varying values as runtime inputs.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Reductions** (design/compilation-composition-change-request.md:348-366)
  > - **Reductions.** MLIR's [workgroup reduction
  > lowering](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Transforms/AllReduceLowering.cpp)
  > constructs subgroup reductions and workgroup exchanges from reduction operations. Apple's [MLX
  > reduction](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal), [packed
  > arithmetic](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h) and [system-scope
  > access](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal) provide concrete target
  > mechanisms. - **Composes:** each of these is a function from operations to operations inside lowering, which takes
  > the scheduled value; `Print` comes after all of them. - **Replaces:** the preceding recorder bodies,
  > `MetalNativeProjection`/`MetalNativeArgmax`'s source-based identification and extraction, and
  > `_pointwise`/`_reduced`/`_contract`'s premature commitment to generic dispatches where the static program still
  > requires specialization, fusion and parallel scheduling. - **Write:** preserve the numerical operations and their
  > permitted precision/rounding, express maps and epilogues as operation regions, retain values across their uses,
  > and lower reductions and packed data to the selected target operations. Preserve each actual effect and observable
  > result while removing unnecessary materialization. Compile producer publication and consumer acquisition from the
  > same range/page dependencies used by transport scheduling. Operation equivalence and applicability follow the
  > structured operands and effects, not an entry-point name, generated variable name or operator spelling. Carry
  > static facts as compiler operands and keep genuinely varying values as runtime inputs.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Replaces** (design/compilation-composition-change-request.md:357-359)
  > **Replaces:** the preceding recorder bodies, `MetalNativeProjection`/`MetalNativeArgmax`'s source-based
  > identification and extraction, and `_pointwise`/`_reduced`/`_contract`'s premature commitment to generic
  > dispatches where the static program still requires specialization, fusion and parallel scheduling.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Write: numerical operations, regions, values, effects**
  (design/compilation-composition-change-request.md:360-362)
  > **Write:** preserve the numerical operations and their permitted precision/rounding, express maps and epilogues as
  > operation regions, retain values across their uses, and lower reductions and packed data to the selected target
  > operations. Preserve each actual effect and observable result while removing unnecessary materialization.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Write: publication and acquisition**
  (design/compilation-composition-change-request.md:363-364)
  > Compile producer publication and consumer acquisition from the same range/page dependencies used by transport
  > scheduling.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Write: applicability from structure, not spelling**
  (design/compilation-composition-change-request.md:364-365)
  > Operation equivalence and applicability follow the structured operands and effects, not an entry-point name,
  > generated variable name or operator spelling.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Performance correction / Performance obligation** (design/compilation-composition-change-request.md:367-374)
  > **Performance obligation:** complete the execution mapping as well as operation coverage. Fusion removes launches
  > and sometimes memory traffic; promotion removes materialization; subgroup lowering removes shared exchanges;
  > publication fusion removes separate visibility/publication work. They are distinct transformations. Fusion must
  > account for repeated producer evaluation, live registers and synchronization scope. Typing alone, compiling a
  > generic shader that reads metadata at run time, or calling unchanged per-operation kernels does not accomplish
  > these transformations. The same final kernel, bindings and execution schedule can be emitted from structured
  > operations; source matching adds no execution capability. The measurement standard is metal-microbench
  > `docs/measurement.md`.
  - Placed by: "**Composes:** each of these is a function from operations to operations inside lowering, which takes
    the scheduled value; `Print` comes after all of them." design/compilation-composition-change-request.md:355-356
- **Complete replacement / all in one commit workgroup** (design/compilation-composition-change-request.md:412-414)
  > The replacement is finished all in one commit workgroup: the compiler interface is finished and the downstream
  > bodies are removed together, by other code with the same feature coverage and the same performance or better (the
  > operator's turn above).
  - Placed by: "are deleted in the commits in which code that `compile_module` and `Program` produce has all of their
    feature coverage at the same performance or better" design/the-compiler.md:104-105
- **Complete replacement / build from source** (design/compilation-composition-change-request.md:416-418)
  > Build from source the LiteRT runtime, delegate, ML Drift and Dawn/Tint dependencies that actually execute these
  > lowerings. Track the adapter, build description and upstream changes in canonical source, not hand edits under
  > `.build`.
  - Placed by: the reader.
- **Complete replacement / production E2B link** (design/compilation-composition-change-request.md:408-419): stated
  under caller, which its clause names with this stage.
- **Complete replacement / operand table row 1: Numerical operations**
  (design/compilation-composition-change-request.md:425-425): stated under export, which its clause names with this
  stage.
- **Complete replacement / operand table row 2: Kernel instructions**
  (design/compilation-composition-change-request.md:426-426)
  > | Kernel instructions | Numerical lowering → typed shader modules/regions, including explicit operand and result
  > bindings → target lowering and final emission. Construct the mesh path directly through IR builders; emitting
  > generated WGSL or MSL and reparsing it as the missing interface does not finish this connection. |
  - Placed by: "Numerical lowering → typed shader modules/regions, including explicit operand and result bindings →
    target lowering and final emission." design/compilation-composition-change-request.md:426
- **Complete replacement / operand table row 3: Values and effects**
  (design/compilation-composition-change-request.md:427-427): stated under schedule, which its clause names with this
  stage.
- **Complete replacement / operand table row 4: Program relations**
  (design/compilation-composition-change-request.md:428-428): stated under scheduled value, which its clause names
  with this stage.
- **Complete replacement / operand table row 5: Executable artifact**
  (design/compilation-composition-change-request.md:429-429)
  > | Executable artifact | Completed lowering → compiled CPU work, GPU pipelines, bindings, dispatches, transfers and
  > synchronization plus their derived signature → the native executor and bridge. CPU execution is native Apple
  > Silicon work; Python preparation is not the CPU target. |
  - Placed by: "Completed lowering → compiled CPU work, GPU pipelines, bindings, dispatches, transfers and
    synchronization plus their derived signature → the native executor and bridge."
    design/compilation-composition-change-request.md:429
- **Complete replacement / operands kept until finalization; publication before execution**
  (design/compilation-composition-change-request.md:431-434)
  > The compiler keeps these operands until finalization, including every module and constant used by later passes. A
  > callback borrowing one shader during `newLibraryWithSource` cannot supply them or fuse its dispatch with another
  > one. The compiler publishes the program and its bodies before execution; finalization occurs after the program's
  > calls and transport relations are available.
  - Placed by: the reader.
- **Complete replacement / many-to-many identities; producer-supplied meaning**
  (design/compilation-composition-change-request.md:434-436): stated under export, which its clause names with this
  stage.
- **Complete replacement / authored sources; emitted source never inspected; tuned projection/argmax**
  (design/compilation-composition-change-request.md:436-439)
  > Authored kernel sources may be compiler inputs and terminal emitter templates; emitted source is never
  > subsequently inspected to recover operands, effects or optimization decisions. Tuned projection/argmax arithmetic
  > is carried through typed parameters and computation regions, including their map/epilogue operands. A template
  > hole filled by extracting somebody else's generated source is not such a region.
  - Placed by: "Lowering emits MSL or native target source, which the target compiler takes. No function of mesh takes
    that source." design/compilation-composition-change-request.md:303-304
- **Complete replacement / Tint objects behind handles; cache hits**
  (design/compilation-composition-change-request.md:440-444)
  > Tint objects are created and destroyed in the C++ compiler; C/Objective-C/Python consumers use handles and typed
  > operand/binding records through the program boundary. They do not serialize an operation into a string to cross
  > it. Cache hits carry the same operands: Dawn's translation runs inside a compilation-cache lookup, so publishing
  > IR only from the cache-miss callback is insufficient. Cache the required representation or completed artifact with
  > its compilation inputs; a cached shader string cannot substitute for the program and its relations.
  - Placed by: "**Emission.** The data structure is Tint's IR module."
    design/compilation-composition-change-request.md:333; "each of these is a function from operations to operations
    inside lowering" design/compilation-composition-change-request.md:355
- **Complete replacement / replaced-implementation table row 1**
  (design/compilation-composition-change-request.md:451-451)
  > | `MetalSpecializeScalars`, `MetalLiteralLanes`, `MetalFoldQuantization`; Torch's reading of static
  > operator/layout metadata at run time | Constant propagation and specialization before target emission; remove
  > repeated static loads, branches and index arithmetic while preserving numerical semantics. |
  - Placed by: "Constant propagation and specialization before target emission"
    design/compilation-composition-change-request.md:451; "each of these is a function from operations to operations
    inside lowering" design/compilation-composition-change-request.md:355
- **Complete replacement / replaced-implementation table row 2**
  (design/compilation-composition-change-request.md:452-452)
  > | `MetalPromoteOperand`, `MetalFuseEpilogue`, `MetalInlineMaps`, `MetalFuseRows` and their
  > `MetalMapShape`/`MetalEntry`/`MetalRowFootprint`/`MetalRowKernel` recognizers | Compose computation regions and
  > indexing maps, remap uses and eliminate dead intermediates. Preserve observable stores and dependencies; remove
  > eligible launches and materialization without introducing repeated expensive producer work. |
  - Placed by: "each of these is a function from operations to operations inside lowering"
    design/compilation-composition-change-request.md:355
- **Complete replacement / replaced-implementation table row 3**
  (design/compilation-composition-change-request.md:453-453)
  > | `MetalNativeProjection`/`MetalNativeArgmax` source recognition and extraction, `MetalPackedWeights`,
  > `MetalProjectionReduction`, `MetalNormalizationReduction` | Select/construct the tuned algorithms from typed
  > operations, layouts and reduction domains. Preserve packed arithmetic, parallel reductions and split work;
  > propagate `MetalNativePlan` facts directly instead of rereading `MN_*` defines. |
  - Placed by: "each of these is a function from operations to operations inside lowering"
    design/compilation-composition-change-request.md:355
- **Complete replacement / replaced-implementation table row 4**
  (design/compilation-composition-change-request.md:454-454)
  > | `MetalCoherentStores`, source-spliced publication/acquisition and `MetalMaskedPass` | Lower typed memory effects
  > and page readiness to coherent access, publication, worklists and later-pass kernels. Preserve
  > computation/transport overlap and fused publication. Carry any required target extension through the IR and
  > emitter. |
  - Placed by: "Lower typed memory effects and page readiness to coherent access, publication, worklists and
    later-pass kernels." design/compilation-composition-change-request.md:454
- **Complete replacement / replaced-implementation table row 5**
  (design/compilation-composition-change-request.md:455-455)
  > | `MetalParsedSpace` and `MetalCommandAccesses` as semantic reconstruction from emitted source | Use the
  > compiler's instruction call/use-def and access relations. Remove the parser and its dead helpers once both
  > scheduling and signature consumers use those relations. |
  - Placed by: "Lowering emits MSL or native target source, which the target compiler takes. No function of mesh takes
    that source." design/compilation-composition-change-request.md:303-304
- **Complete replacement / replaced-implementation table row 6**
  (design/compilation-composition-change-request.md:456-456)
  > | Footprint reconstruction for the recorded commands, and the default (non-self-timed) preparation path | Emit the
  > shared native program's required synchronization regardless of the caller; delete that path and the orchestration
  > it replaces. |
  - Placed by: "**Composes:** lowering takes the scheduled value and emits the native program; the runtime calls it."
    design/compilation-composition-change-request.md:201
- **Complete replacement / after the replacement** (design/compilation-composition-change-request.md:458-459)
  > After the replacement the source has no source-rewrite fallback, second compiler mode or path back to the removed
  > implementation, and source matching is not an execution dependency.
  - Placed by: "Lowering emits MSL or native target source, which the target compiler takes. No function of mesh takes
    that source." design/compilation-composition-change-request.md:303-304
- **Source bindings / Software binary64** (design/compilation-composition-change-request.md:479-482)
  > They implement the scalar operations of a typed binary64 kernel when the selected target lacks native binary64
  > instructions. **Composes:** lowering takes the typed binary64 instructions and emits target instructions; it
  > preserves storage, conversions, special values and the required rounding semantics. Derived source carries the
  > upstream notices.
  - Placed by: "**Composes:** lowering takes the typed binary64 instructions and emits target instructions"
    design/compilation-composition-change-request.md:481
- **7 / Composes: lowering emits the native program, the runtime calls it**
  (design/compilation-composition-change-request.md:505-507)
  > Lowering takes it and emits the native program: the table below gives, per operand, what the emitted program
  > preserves and what it does not read. The runtime calls that program; the archive stores the scheduled value.
  - Placed by: "Lowering takes it and emits the native program" design/compilation-composition-change-request.md:505
- **7 / Replaces: run-time readers of positions, tasks, guards; static discovery**
  (design/compilation-composition-change-request.md:508-511)
  > **Replaces:** the code of `MetalProgramExecute`/`MetalNativeStep` that reads positions, tasks, guards and a loop
  > stack at run time; `MetalLoopTest` and the whole-queue waits it makes, which are deleted and not replaced; and
  > repeated static command-kind, cut, pass and binding discovery inside `MetalRowEncode`/`MetalReplayRun`. Compiling
  > an outer caller which still invokes those generic decoders does not complete their replacement.
  - Placed by: "Lowering takes it and emits the native program" design/compilation-composition-change-request.md:505
- **7 / Write** (design/compilation-composition-change-request.md:512-516)
  > pass the solved schedule and effects into native target lowering. Emit fixed target/API operations with bound
  > operands, in the scheduled order; compose and optimize CPU bodies with them, and fuse target-local work where
  > legal. CPU bodies stay lowerable until composition; they are not reduced before it to opaque function pointers.
  > The executor bodies this replaces are deleted with it.
  - Placed by: "pass the solved schedule and effects into native target lowering."
    design/compilation-composition-change-request.md:512-513
- **7 / table row 1: typed numerical operations** (design/compilation-composition-change-request.md:520-520)
  > | Typed numerical operations, shapes, views and alias/effect relations | Specified numerical semantics, permitted
  > floating-point transformations, mutations, addressing and required lifetimes | Type objects, shape inference,
  > operation names, alias-analysis graphs |
  - Placed by: "Lowering takes it and emits the native program: the table below gives, per operand, what the emitted
    program preserves and what it does not read." design/compilation-composition-change-request.md:505-506
- **7 / table row 2: partitioned computation, transfers, compiled order**
  (design/compilation-composition-change-request.md:521-521)
  > | Partitioned computation, transfer dependencies and compiled order | Required producer/consumer ordering,
  > publication and readiness; legal overlap and fused work | Original vertices, edges, partitions and an independent
  > scheduling solve |
  - Placed by: "Lowering takes it and emits the native program: the table below gives, per operand, what the emitted
    program preserves and what it does not read." design/compilation-composition-change-request.md:505-506
- **7 / table row 3: realized storage** (design/compilation-composition-change-request.md:522-522)
  > | Realized storage, copies and completion dependencies | Correct value instance, non-overlapping live storage
  > where required, reuse only after completion | Repeated lifetime analysis or parsing allocation metadata on each
  > task |
  - Placed by: "Lowering takes it and emits the native program: the table below gives, per operand, what the emitted
    program preserves and what it does not read." design/compilation-composition-change-request.md:505-506
- **7 / table row 4: target commands** (design/compilation-composition-change-request.md:523-523)
  > | Selected ICB runs, bindings, copies/fills, barriers and API boundaries | The selected target commands with their
  > required ordering and operands | Static command-kind switches, pass scans, cut discovery and generic callback
  > tables |
  - Placed by: "Lowering takes it and emits the native program: the table below gives, per operand, what the emitted
    program preserves and what it does not read." design/compilation-composition-change-request.md:505-506
- **7 / table row 5: lowered synchronization and its cause**
  (design/compilation-composition-change-request.md:524-524)
  > | Lowered synchronization and the cause recorded for each | The required externally observable synchronization
  > property of this executable | Runtime reconstruction of the compiler graph; the signature remains an associated
  > observation artifact |
  - Placed by: "Lowering takes it and emits the native program: the table below gives, per operand, what the emitted
    program preserves and what it does not read." design/compilation-composition-change-request.md:505-506
- **7 / contract paragraph** (design/compilation-composition-change-request.md:527-529)
  > The contract preserves the required observations and ordering, not a literal source graph or one runtime action
  > per vertex. Dropping an annotation before its consumer, reparsing emitted source to recover it, or silently
  > forgetting its property is an incomplete lowering.
  - Placed by: "Dropping an annotation before its consumer, reparsing emitted source to recover it, or silently
    forgetting its property is an incomplete lowering." design/compilation-composition-change-request.md:528-529
- **7 / performance paragraph** (design/compilation-composition-change-request.md:537-540)
  > Performance is an independent requirement of this replacement (the operator above: "the performance ceiling of our
  > final lowering is paramount"). Semantic equivalence does not excuse an extra run-time decoder, launch, transfer,
  > guard or wait introduced only to maintain an erased abstraction. Removing the outermost decoder while the same
  > dispatch runs underneath it does not satisfy the requirement.
  - Placed by: "the performance ceiling of our final lowering is paramount"
    design/compilation-composition-change-request.md:537-538

### lower: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **A3 "Which function computes it, and which read it"** (design/disjoint-pages-change-request.md:469-474): stated
  under schedule, which its clause names with this stage.
- **C3.1 algorithm 1 (masked first pass)** (design/disjoint-pages-change-request.md:813-816)
  > An execution's first pass loads its receives' phases, `MESH_LANE_BYTES` (a compile-time 16) bytes per GPU thread,
  > and computes in the threads whose pages have been received. The other threads are predicated off. A mask is
  > applied by select, never by multiplying by 0 or 1 (0 × NaN is NaN), and a reduction puts its identity into the
  > masked-off threads.
  - Placed by: "The recorded kernels' first pass (`mesh_read_N`, `metal_recording.m`) reads the phases and writes the
    threadgroup count" (design/disjoint-pages-change-request.md:825-826); "- **Lowering.** The kernels
    `metal_recording.m` records (`mesh_read_N`) read the pages the `struct mesh_page_entry` rows name and the state
    section's phases." (design/disjoint-pages-change-request.md:401-402)
- **C3.1 algorithm 2 (compaction)** (design/disjoint-pages-change-request.md:817-818)
  > The first pass compacts the work it left (`simd_ballot`, `simd_prefix_exclusive_sum`) and writes the later pass's
  > threadgroup count.
  - Placed by: "The recorded kernels' first pass (`mesh_read_N`, `metal_recording.m`) reads the phases and writes the
    threadgroup count" (design/disjoint-pages-change-request.md:825-826); "- **Lowering.** The kernels
    `metal_recording.m` records (`mesh_read_N`)" (design/disjoint-pages-change-request.md:401)
- **C3.1 algorithm 3 (indirect dispatch)** (design/disjoint-pages-change-request.md:819-820)
  > `dispatchThreadgroups(indirectBuffer:)` reads that count when the later pass starts, so the later pass is empty
  > when every page has been received.
  - Placed by: "The recorded kernels' first pass (`mesh_read_N`, `metal_recording.m`) reads the phases and writes the
    threadgroup count, and Metal's indirect dispatch reads that count."
    (design/disjoint-pages-change-request.md:825-827); "The threadgroup count is Metal's
    `MTLDispatchThreadgroupsIndirectArguments`." (design/disjoint-pages-change-request.md:829)
- **C3.1 algorithm 4, bridge half** (design/disjoint-pages-change-request.md:821-824): stated under bridge, which its
  clause names with this stage.
- **C3.1 algorithm 4, later-pass half** (design/disjoint-pages-change-request.md:821-823)
  > A cancelled receive does not change its phase, so the later pass finds it one phase behind, and its threads end as
  > an error that the later pass writes into the execution's status word.
  - Placed by: "The later pass writes the status word, which only the GPU writes, and `e2b_replay` reads it after the
    run." (design/disjoint-pages-change-request.md:827-828); "The recorded kernels' later pass reads the counts and
    the phases (C3.1)." (design/disjoint-pages-change-request.md:1442-1443)
- **C3.1 "Written by, read by"** (design/disjoint-pages-change-request.md:825-829)
  > **Written by, read by.** The bridge (`rdma/mesh-flow.c`) writes the phases and the counts. The recorded kernels'
  > first pass (`mesh_read_N`, `metal_recording.m`) reads the phases and writes the threadgroup count, and Metal's
  > indirect dispatch reads that count. The later pass writes the status word, which only the GPU writes, and
  > `e2b_replay` reads it after the run. The phases are at each `struct mesh_hop`'s `field`, the counts at its
  > `count`, and the status is a `struct mesh_status` (`rdma/mesh.h`). The threadgroup count is Metal's
  > `MTLDispatchThreadgroupsIndirectArguments`.
  - Placed by: "The recorded kernels' first pass (`mesh_read_N`, `metal_recording.m`) reads the phases and writes the
    threadgroup count" (design/disjoint-pages-change-request.md:825-826); "- **Lowering.** The kernels
    `metal_recording.m` records (`mesh_read_N`) read the pages the `struct mesh_page_entry` rows name and the state
    section's phases." (design/disjoint-pages-change-request.md:401-402); "- **Runtime.** The bridge (mesh
    `rdma/mesh-flow.c`) reads the `struct mesh_hop` entries and, for sends, the words at `word`, and writes the phases
    at `field` and the counts at `count`." (design/disjoint-pages-change-request.md:397-398)
- **C3.1 waits deleted** (design/disjoint-pages-change-request.md:830-831)
  > Both waits in `mesh_read_N` are deleted: the wait on the execution's mask, and the local countdown. A producer on
  > the same command queue comes before its consumer by the command buffer's order or a memory barrier.
  - Placed by: "Both waits in `mesh_read_N` are deleted" (design/disjoint-pages-change-request.md:830); "-
    **Lowering.** The kernels `metal_recording.m` records (`mesh_read_N`)"
    (design/disjoint-pages-change-request.md:401)
- **C3.2 (a) data structure and types** (design/disjoint-pages-change-request.md:834-836): stated under schedule,
  which its clause names with this stage.
- **C3.3 algorithm 3 (copies, t mod u)** (design/disjoint-pages-change-request.md:851-851): stated under schedule,
  which its clause names with this stage.
- **C3.3 "Written by, read by"** (design/disjoint-pages-change-request.md:852-854): stated under schedule, which its
  clause names with this stage.
- **C3.6 (3) local countdown** (design/disjoint-pages-change-request.md:884-884)
  > The local countdown wait that ends with zeros and no mark: removed by C3.1, with the waits in `mesh_read_N`.
  - Placed by: "removed by C3.1, with the waits in `mesh_read_N`" (design/disjoint-pages-change-request.md:884); "-
    **Lowering.** The kernels `metal_recording.m` records (`mesh_read_N`)"
    (design/disjoint-pages-change-request.md:401)
- **C3.6 (4) link loss and poll bound** (design/disjoint-pages-change-request.md:885-886)
  > A link loss and a GPU thread's poll bound that both read as `ETIMEDOUT`: the poll bound is removed by C3.1, and a
  > link loss adds to the counts as cancellations.
  - Placed by: "Both waits in `mesh_read_N` are deleted: the wait on the execution's mask, and the local countdown."
    (design/disjoint-pages-change-request.md:830); "The bridge (`rdma/mesh-flow.c`) writes the phases and the counts."
    (design/disjoint-pages-change-request.md:825)
- **C3.7 algorithm 2** (design/disjoint-pages-change-request.md:898-899)
  > A reader of execution t compares the field with Gray(⌊t / u⌋ + 1 mod 4). Equal: received. One behind: not yet.
  > Anything else: a write ran ahead of the allocator's order.
  - Placed by: "The recorded kernels' first pass reads the phases (C3.1)."
    (design/disjoint-pages-change-request.md:908)
- **C3.7 "Written by, read by"** (design/disjoint-pages-change-request.md:906-910): stated under schedule, which its
  clause names with this stage.
- **C3.7 projection kernels** (design/disjoint-pages-change-request.md:913-915)
  > The projection kernels' partial sums are not zeroed: every threadgroup writes its own before any are read. Their
  > arrival counters restore the count on the last arrival (native_projection.metal:232), as an mbarrier's phase
  > completion reloads it.
  - Placed by: "metal_recording.m: both waits in `mesh_read_N` bounded at 2^23 polls (lines 1921–1996), the remote and
    the local countdown; `mesh_seen`'s failed value; a GPU thread's `UINT64_MAX` into a step's landed record, and
    `e2b_replay`'s read of it; the zeroing of the projection kernels' partial sums (line 1546)."
    (design/disjoint-pages-change-request.md:932-934); "- **Lowering.** The kernels `metal_recording.m` records"
    (design/disjoint-pages-change-request.md:401)
- **C4 metal-microbench deleted: metal_recording.m** (design/disjoint-pages-change-request.md:932-934)
  > metal_recording.m: both waits in `mesh_read_N` bounded at 2^23 polls (lines 1921–1996), the remote and the local
  > countdown; `mesh_seen`'s failed value; a GPU thread's `UINT64_MAX` into a step's landed record, and `e2b_replay`'s
  > read of it; the zeroing of the projection kernels' partial sums (line 1546).
  - Placed by: "metal_recording.m: both waits in `mesh_read_N`" (design/disjoint-pages-change-request.md:932); "-
    **Lowering.** The kernels `metal_recording.m` records (`mesh_read_N`) read the pages the `struct mesh_page_entry`
    rows name and the state section's phases." (design/disjoint-pages-change-request.md:401-402)
- **D.3.2** (design/disjoint-pages-change-request.md:1386-1395): stated under schedule, which its clause names with
  this stage.
- **D.3.3** (design/disjoint-pages-change-request.md:1398-1405): stated under schedule, which its clause names with
  this stage.
- **D.3.4 data structure** (design/disjoint-pages-change-request.md:1408-1410): stated under Program, which its clause
  names with this stage.
- **D.3.4 "Written by, read by"** (design/disjoint-pages-change-request.md:1418-1419): stated under Program, which its
  clause names with this stage.
- **D.3.4 delegate's capture** (design/disjoint-pages-change-request.md:1420-1422)
  > The delegate's capture uses an `MTLCommandBuffer`: ml_drift encodes into one it is given, and commits its own
  > loads during the capture. The recorder's queue for the delegate exists only for the capture, and nothing the run
  > executes passes through it.
  - Placed by: "The recorder's queue for the delegate exists only for the capture"
    (design/disjoint-pages-change-request.md:1421); "- **Lowering.** The kernels `metal_recording.m` records
    (`mesh_read_N`)" (design/disjoint-pages-change-request.md:401)
- **D.3.6** (design/disjoint-pages-change-request.md:1434-1445): stated under bridge, which its clause names with this
  stage.
- **D.4 metal-microbench deleted (1): classic command queues and buffers**
  (design/disjoint-pages-change-request.md:1457-1460): stated under Program, which its clause names with this stage.
- **D.4 metal-microbench deleted (3): delegate buffers as captured**
  (design/disjoint-pages-change-request.md:1462-1462)
  > The delegate's buffers bound as captured, and `MetalNativeSubstitute`'s reads of the weights from them.
  - Placed by: "The recorded kernels read the placed pages, and after the bind no command reads a delegate buffer."
    (design/disjoint-pages-change-request.md:1393-1394); "- **Lowering.** The kernels `metal_recording.m` records
    (`mesh_read_N`) read the pages the `struct mesh_page_entry` rows name"
    (design/disjoint-pages-change-request.md:401-402)

### lower: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R4 nothing synchronizes outputs one page at a time: landings read by masks**
  (design/plan-to-bridge-change-request.md:2239-2240)
  > The receiving side reads landings by masks (C3), so a landing interrupts no backend. A backend's work and the
  > bridge's transfers overlap (aot-compilation, appendix item 2; Pathways §4.5).
  - Placed by: the reader.
- **R4 nothing synchronizes outputs: the release follows the producing work**
  (design/plan-to-bridge-change-request.md:2241-2242)
  > The release follows the producing work. The function's kernels write their pages, and the supplied store follows
  > them. The compilation writes no kernel (aot-compilation A9).
  - Placed by: the reader.

### lower: [torch-capture-change-request.md](torch-capture-change-request.md)

- **5, Write (the release's one 64-bit atomic write)** (design/torch-capture-change-request.md:696-697)
  > the fence, then one 64-bit atomic write of the word, `atomic_max_explicit` with t + 1, the 64-bit atomic write the
  > header declares for it.
  - Placed by: "The kernel source that `MetalRelease` (metal-microbench metal_recording.m) returns writes it; the
    bridge reads it (design/disjoint-pages-change-request.md C3: "The bridge posts a send of execution t once the word
    reads at least t + 1.")." (design/torch-capture-change-request.md:685-687)
- **Source for item 2's execution-time estimates, Write, first sentence (the lowering supplies the cost model's
  operands)** (design/torch-capture-change-request.md:871-872)
  > the target lowering supplies the operations, memory traffic and execution mapping of each compiled function to the
  > analytical cost model, with the machine's rates and primitive costs as operands.
  - Placed by: "**It takes** each compiled function's operations, the bytes it reads and writes and its execution
    mapping, which the target lowering supplies, and the machine's rates and primitive costs as operands."
    (design/torch-capture-change-request.md:850-851)
- **Source for item 2's execution-time estimates, Write, third and fourth sentences (estimates stored; Bff supplies
  locations and reuse edges)** (design/torch-capture-change-request.md:874-875)
  > The estimates are stored with the compiled functions and are the t(v) of item 2's λ_max and Bff(e). The resulting
  > Bff(e) supplies the value's locations and the native program's corresponding reuse edges.
  - Placed by: the reader.
- **Source for item 2's execution-time estimates, Write, last sentence (transformations update costs)**
  (design/torch-capture-change-request.md:876-876)
  > Transformations update the costs of their resulting functions along with their effects.
  - Placed by: the reader.

### lower: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **1 / The data structure** (design/synchronization-signature-change-request.md:88-92)
  > **The data structure** is the synchronization graph of Sriram and Bhattacharyya, with its fixed self-timed order
  > (Bhattacharyya, Sriram and Lee 1997; Lee and Ha: "in the order specified at compile time"). Its edges carry delays
  > across executions, from Lee and Messerschmitt's synchronous dataflow (1987). Its copies are the unrolled ones,
  > from Lam (1988). The same inputs give the same schedule on every rank, from Schneider (1990), already cited by
  > design/plan-to-bridge-change-request.md R5.
  - Placed by: "**Produced by** the AOT compilation, from the lowered IR, and nothing else writes it. Lowering takes
    the scheduled value (`ir.Scheduled`, design/torch-source-lowering-change-request.md)."
    design/synchronization-signature-change-request.md:100-101
- **1 / It holds** (design/synchronization-signature-change-request.md:93-97)
  > **It holds**, for each of the command queue's command stream, the thread and the bridge, and for each execution: -
  > the ordered list of synchronization operations: submissions, queue signals and waits, encoder fence updates, fence
  > waits and barriers, and the thread's waits on a shared event; - their counts; - the synchronization graph's edges,
  > with their delays.
  - Placed by: "**Produced by** the AOT compilation, from the lowered IR, and nothing else writes it. Lowering takes
    the scheduled value (`ir.Scheduled`, design/torch-source-lowering-change-request.md)."
    design/synchronization-signature-change-request.md:100-101
- **1 / Produced by** (design/synchronization-signature-change-request.md:100-101)
  > **Produced by** the AOT compilation, from the lowered IR, and nothing else writes it. Lowering takes the scheduled
  > value (`ir.Scheduled`, design/torch-source-lowering-change-request.md).
  - Placed by: "**Produced by** the AOT compilation, from the lowered IR, and nothing else writes it. Lowering takes
    the scheduled value (`ir.Scheduled`, design/torch-source-lowering-change-request.md)."
    design/synchronization-signature-change-request.md:100-101
- **5 / Write: a program with no wait** (design/synchronization-signature-change-request.md:154-155)
  > For a program whose static schedule has no wait, the signature holds only the method's own operations. In the 252
  > tok/s replay (metal-microbench 864a0ae4), that was one submission per run.
  - Placed by: "**Produced by** the AOT compilation, from the lowered IR, and nothing else writes it."
    design/synchronization-signature-change-request.md:100
- **6 / Write** (design/synchronization-signature-change-request.md:161-162)
  > **Write:** Barriers, fences and atomics inside kernels aren't API calls. They belong to the compiled kernels in
  > the artifact, so they are counted there, statically, as part of the same signature.
  - Placed by: "They belong to the compiled kernels in the artifact, so they are counted there, statically, as part of
    the same signature." design/synchronization-signature-change-request.md:161-162

### lower: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **1 / Write: no torch Metal kernel** (design/torch-source-lowering-change-request.md:161-161)
  > **Write:** no torch Metal kernel is ever submitted or recorded again.
  - Placed by: "`lower_metal.lower` returns them without their copies to the Metal device, whose function takes host
    pages, so it is first (design/torch-source-lowering-change-request.md, item 1: "no torch Metal kernel is ever
    submitted or recorded again.")." design/the-compiler.md:99-101; and "Items 1, 5 and 6 are operator lowerings: they
    run where the exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before
    scheduling." design/torch-source-lowering-change-request.md:129-130
- **3 / Write: one schedule stage** (design/torch-source-lowering-change-request.md:241-243): stated under schedule,
  which its clause names with this stage.
- **3 / Write: commands written in that order** (design/torch-source-lowering-change-request.md:250-250)
  > Each processor's commands are written in that order.
  - Placed by: "### 3. Self-timed lowering of dependences into Metal orderings"
    design/torch-source-lowering-change-request.md:198; "Lowering, the runtime and the archive each take that
    scheduled value." design/torch-source-lowering-change-request.md:243
- **3 / Write: same threadgroup** (design/torch-source-lowering-change-request.md:255-255)
  > same threadgroup: a threadgroup barrier;
  - Placed by: "### 3. Self-timed lowering of dependences into Metal orderings"
    design/torch-source-lowering-change-request.md:198; "Each of those dependences becomes the one Metal ordering its
    two ends' locations admit:" design/torch-source-lowering-change-request.md:254
- **3 / Write: same encoder** (design/torch-source-lowering-change-request.md:256-256)
  > same encoder: an intra-pass barrier;
  - Placed by: "### 3. Self-timed lowering of dependences into Metal orderings"
    design/torch-source-lowering-change-request.md:198; "Each of those dependences becomes the one Metal ordering its
    two ends' locations admit:" design/torch-source-lowering-change-request.md:254
- **3 / Write: same queue** (design/torch-source-lowering-change-request.md:257-257)
  > same queue: a fence;
  - Placed by: "### 3. Self-timed lowering of dependences into Metal orderings"
    design/torch-source-lowering-change-request.md:198; "Each of those dependences becomes the one Metal ordering its
    two ends' locations admit:" design/torch-source-lowering-change-request.md:254
- **3 / Write: another queue, another device or the CPU** (design/torch-source-lowering-change-request.md:258-258)
  > another queue, another device or the CPU: an event.
  - Placed by: "### 3. Self-timed lowering of dependences into Metal orderings"
    design/torch-source-lowering-change-request.md:198; "Each of those dependences becomes the one Metal ordering its
    two ends' locations admit:" design/torch-source-lowering-change-request.md:254
- **3 / Write: independent pairs** (design/torch-source-lowering-change-request.md:259-259)
  > Independent pairs get no ordering, and Metal then permits them to overlap.
  - Placed by: "### 3. Self-timed lowering of dependences into Metal orderings"
    design/torch-source-lowering-change-request.md:198

### lower: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **5** (design/integration-finishing-change-request.md:64-66)
  > **Replaces:** metal-microbench c4d41929's `dense_tensor_f16_received` and tests/gpu/received_matmul.m. -
  > **Source:** design/torch-source-lowering-change-request.md item 4, withdrawn 2026-11-08. - **Write:** both are
  > deleted.
  - Placed by: the reader.
- **7 (The data structure: one MTL4CommitOptions a commit)** (design/integration-finishing-change-request.md:75-76)
  > **The data structure** is one `MTL4CommitOptions` a commit, whose feedback handler was added for that commit
  > alone, and that commit's record: struct Feedback's semaphore, error and GPU times, and struct MetalRun's
  > `failure`.
  - Placed by: "**Produced by** every commit of the replay" design/integration-finishing-change-request.md:77; "in
    metal_recording.m, `MetalRunOf`" design/integration-finishing-change-request.md:82
- **7 (The data structure: that commit's record)** (design/integration-finishing-change-request.md:75-76)
  > **The data structure** is one `MTL4CommitOptions` a commit, whose feedback handler was added for that commit
  > alone, and that commit's record: struct Feedback's semaphore, error and GPU times, and struct MetalRun's
  > `failure`.
  - Placed by: "**Produced by** every commit of the replay: the handler writes the feedback data Metal calls it with
    to its own commit's record." design/integration-finishing-change-request.md:77-78
- **7 (Produced by)** (design/integration-finishing-change-request.md:77-78)
  > **Produced by** every commit of the replay: the handler writes the feedback data Metal calls it with to its own
  > commit's record.
  - Placed by: "**Produced by** every commit of the replay" design/integration-finishing-change-request.md:77
- **7 (Consumed by)** (design/integration-finishing-change-request.md:78-78)
  > **Consumed by** `feedback_wait`, which receives one handler's call a commit.
  - Placed by: "**Consumed by** `feedback_wait`, which receives one handler's call a commit."
    design/integration-finishing-change-request.md:78
- **7 (Replaces, e2b_replay.m feedback_start)** (design/integration-finishing-change-request.md:80-81)
  > in metal-microbench tools/e2b_replay.m, `feedback_start`: one `MTL4CommitOptions` a phase (`PREPARE`, `PREFILL`,
  > `DECODE`, `PROFILE`), which every commit of that phase carries;
  - Placed by: "in metal-microbench tools/e2b_replay.m, `feedback_start`"
    design/integration-finishing-change-request.md:80
- **7 (Replaces, metal_recording.m MetalRunOf)** (design/integration-finishing-change-request.md:82-82)
  > in metal_recording.m, `MetalRunOf`: one `MTL4CommitOptions` that every replay's commits carry.
  - Placed by: "in metal_recording.m, `MetalRunOf`" design/integration-finishing-change-request.md:82

## Program

`Program`: binds addresses and calls the functions.

### Program: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **A4 which nodes hold a function's static schedule** (design/aot-compilation-change-request.md:264-266): stated
  under schedule, which its clause names with this stage.
- **A4 a node to which the placement operand maps no device (the program's part)**
  (design/aot-compilation-change-request.md:273-274)
  > A node with no section starts its static schedules' executions, relays as its routes give it, and computes
  > nothing; it has no value for any program code to read.
  - Placed by: the reader.
- **A7 stage** (design/aot-compilation-change-request.md:319-319)
  > **Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its functions.
  - Placed by: "**Stage:** the runtime." (design/aot-compilation-change-request.md:319); "**the runtime** (runtime.py
    `Program`)" (design/aot-compilation-change-request.md:97)
- **A7 one order; total order multicast by a fixed sequencer (primitives, properties, steps of Fig. 5)**
  (design/aot-compilation-change-request.md:320-321, 339-362)
  > **What the code must do.** Order the function executions of the program on the island in one order, the same at
  > every node, and keep the program's own order of its calls within it. [...] **The algorithm is total order
  > multicast by a fixed sequencer**, as X. Défago, A. Schiper and P. Urbán define both (Total order broadcast and
  > multicast algorithms: taxonomy and survey, ACM Comput. Surv. 36(4), 2004). Nothing else orders function
  > executions. - **The primitives** (§3): "total order broadcast is defined in terms of two primitives, which are
  > called TO-broadcast(m) and TO-deliver(m)". Every message "carries the identity of its sender, denoted by sender
  > (m)". - **The properties the code must have**, verbatim: - "(VALIDITY) If a correct process TO-broadcasts a
  > message m, then it eventually TO-delivers m." (§3) - "(UNIFORM AGREEMENT) If a process TO-delivers a message m,
  > then all correct processes eventually TO-deliver m." (§3) - "(UNIFORM INTEGRITY) For any message m, every process
  > TO-delivers m at most once, and only if m was previously TO-broadcast by sender (m)." (§3) - "(UNIFORM TOTAL
  > ORDER) If processes p and q both TO-deliver messages m and m′, then p TO-delivers m before m′ if and only if q
  > TO-delivers m before m′." (§3) - "(FIFO ORDER) If a correct process TO-broadcasts a message m before it
  > TO-broadcasts a message m′, then no correct process delivers m′ unless it has previously delivered m." (§4.3.1) -
  > "(GLOBAL TOTAL ORDER) The relation < is acyclic, where < is defined as follows: m < m′ if and only if any correct
  > process delivers m and m′, in that order." (§5.2.3) - **The steps** are their Fig. 5, the "UB" variant of §7.1, as
  > written: - Sender: "procedure TO-broadcast(m): send (m) to sequencer". - Sequencer: "Initialization: seqnum := 1";
  > "when receive (m): sn(m) := seqnum; send (m, sn(m)) to all; seqnum := seqnum + 1". - Destinations:
  > "Initialization: nextdeliver := 1; pending := ∅"; "when receive (m, seqnum): pending := pending ∪ {(m, seqnum)};
  > while ∃(m′, seqnum′) ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver := nextdeliver + 1".
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A7 multicast to many groups** (design/aot-compilation-change-request.md:363-366)
  > **Multicast to many groups** is their §5.2.4, as written: "1. form a super-group with the union of all destination
  > groups; 2. whenever a message m is multicast to a group, multicast it to the super-group, and 3. processes not in
  > Dest(m) discard m." The super-group is the island's nodes. One sequence numbers every message and every node
  > delivers in its order, so no two nodes deliver two messages in opposite orders, and Global Total Order holds.
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A7 data structures: m, sn(m), Dest(m)** (design/aot-compilation-change-request.md:367-371, 385-386): stated under
  bridge, which its clause names with this stage.
- **A7 sender(m) (the bridge's and mesh_plan's part)** (design/aot-compilation-change-request.md:372-374)
  > **sender(m)** is the program's one sender: the least of the program's nodes. It TO-broadcasts the program's
  > submissions in the order the program makes them. The program's nodes are the nodes its launcher started it at, an
  > operand of the `mesh_plan` call (`program`) that the launcher passes as it passes `group` (MESH_GROUP).
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319); the clause's own "an operand of the `mesh_plan` call
    (`program`)" (design/aot-compilation-change-request.md:374)
- **A7 the sequencer** (design/aot-compilation-change-request.md:382-384): stated under bridge, which its clause names
  with this stage.
- **A7 deliver(m); produced by, consumed by** (design/aot-compilation-change-request.md:387-392)
  > **deliver(m)** at a node of Dest(m) appends m to its static schedule's queue there (appendix item 2), which
  > releases its executions at the node. The pages of delivered executions interleave on each cable (T99). -
  > **Produced by, consumed by.** The program's sender TO-broadcasts m. The sequencer numbers it and sends (m, sn(m))
  > to all. Each node of Dest(m) delivers it into the queue of appendix item 2, and the bridge at that node takes it
  > from the queue. The message is `struct mesh_order` up to its delivery. The queue's entry, which the bridge takes,
  > is `struct mesh_delivery` (rdma/mesh.h).
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node."
    (design/aot-compilation-change-request.md:705); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **A7 failures** (design/aot-compilation-change-request.md:393-396)
  > **Failures.** "This algorithm does not tolerate the failure of the sequencer" (§7.1). A static schedule is used
  > only while its map is unchanged (design/disjoint-pages-change-request.md D.3.6), so the loss of a cable or node
  > ends every static schedule that uses it, and the sequencer of a static schedule in use does not change. A static
  > schedule is also used only within its view (below).
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A7 one sender for each program** (design/aot-compilation-change-request.md:397-402)
  > **One sender for each program.** "Total Order alone does not guarantee that messages are delivered in the order in
  > which they are sent" (§4.3.1), and FIFO Order is a property of one sender's messages. Where two nodes broadcast
  > calls of one program, nothing makes one node's broadcast precede the other's (Lamport's "precedes", their
  > Definition 1), so neither FIFO Order nor Causal Order (§4.3.2) need keep the program's order. With one sender,
  > whose messages reach the sequencer first in, first out, Fig. 5's sequencer numbers them in the order they were
  > sent ("when receive (m) sn(m) := seqnum"). So FIFO Order holds, and with it A10's order constraints between the
  > program's calls.
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A7 views: the four view properties** (design/aot-compilation-change-request.md:420-429)
  > **The view properties the code must have** are G. V. Chockler, I. Keidar and R. Vitenberg's (Group communication
  > specifications: a comprehensive study, ACM Comput. Surv. 33(4), 2001), verbatim: - "(Self Inclusion) If process p
  > installs view V, then p is a member of V." (Property 3.1) - "(Local Monotonicity) If a process p installs view V
  > after installing view V′ then the identifier of V is greater than that of V′." (Property 3.2) - "(Initial View
  > Event) Every send, recv and safe_prefix event occurs within some view." (Property 3.3), with its note: "one has to
  > restrict the behavior of the application, so that no send events occur before the first view_chng event." -
  > "(Sending View Delivery) If a process p receives message m in view V, and some process q (possibly p = q) sends m
  > in view V′, then V = V′." (Property 4.3)
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A7 views: a process** (design/aot-compilation-change-request.md:430-432): stated under bridge, which its clause
  names with this stage.
- **A7 views: the membership service is the sequencer** (design/aot-compilation-change-request.md:433-441): stated
  under bridge, which its clause names with this stage.
- **A7 views: the view identifier** (design/aot-compilation-change-request.md:442-447): stated under bridge, which its
  clause names with this stage.
- **A7 views: a bridge installs a view** (design/aot-compilation-change-request.md:448-450): stated under bridge,
  which its clause names with this stage.
- **A7 views: Fig. 5 runs within each view** (design/aot-compilation-change-request.md:451-454)
  > **Fig. 5 runs within each view as written.** Its initializations, "seqnum := 1" and "nextdeliver := 1; pending :=
  > ∅", happen at the view's creation at the sequencer and at its installation at each member. Every message carries
  > its view's identifier. A member holds and delivers only the messages of the view it has installed (Sending View
  > Delivery), and the program's sender TO-broadcasts nothing before its first view (Initial View Event).
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A7 views: a change of view blocks and flushes** (design/aot-compilation-change-request.md:460-466)
  > When the sequencer's map changes, it numbers no further message of the old view. Its first message of the new view
  > carries the last sequence number it gave in the old one. - A member that installed the old view from the same
  > process delivers the old view's messages up to that number, then installs the new view. - Where the old view's
  > sequencer is gone (its bridge stopped, or the map changed to another sequencer), the old view's undelivered
  > messages are never delivered: "This algorithm does not tolerate the failure of the sequencer" (§7.1).
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319); "**The sequencer** is the bridge of the
    highest-numbered node the map reaches" (design/aot-compilation-change-request.md:382)
- **A7 views: a static schedule is used only within its view** (design/aot-compilation-change-request.md:467-470)
  > **A static schedule is used only within the view its nodes' arrivals were counted in.** Installing a view ends
  > every static schedule of the old one, as D.3.6 ends one at its map's change, and `mesh_plan` computes it again in
  > the new view. Each arrival carries its view's identifier, and the program's sender counts only the arrivals of the
  > view it has installed.
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319); the clause's own "`mesh_plan` computes it again in the
    new view" (design/aot-compilation-change-request.md:468-469)
- **A7 views: every member computes its static schedules on the view's map**
  (design/aot-compilation-change-request.md:471-478)
  > **Every member of a view computes its static schedules on the view's map.** - The map a view is created from is
  > the sequencer's, and the view's first message carries it: the records the sequencer's map holds when it creates
  > the view. Every member calls `mesh_plan` on those records. - So every member of a view computes the same static
  > schedule from the same operands. Replicas that start in the same state and execute the same inputs in the same
  > order produce the same outputs (F. B. Schneider, Implementing fault-tolerant services using the state machine
  > approach: a tutorial, ACM Comput. Surv. 22(4), 1990). An index one end's static schedule gives is the other end's
  > by construction, and nothing tests it at run time. - A static schedule is then used while its view is, and its
  > view while the sequencer's map is unchanged (D.3.6).
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319); the clause's own "Every member calls `mesh_plan` on
    those records." (design/aot-compilation-change-request.md:473)
- **A7 a static schedule's destinations exist before its first message**
  (design/aot-compilation-change-request.md:479-480, 487-494)
  > **A static schedule's destinations exist before its first message.** Within a view, Section 3's static system
  > holds for the processes the view names, and a static schedule's first message must find all of them holding it.
  > [...] Nodes finish compiling a static schedule at different times, and that is allowed. A static schedule is
  > created as an NCCL communicator is: "ncclCommInitRank implicitly synchronizes with other ranks" (NCCL 2.32.3, API,
  > Communicator Creation and Management Functions). - The program's sender counts the static schedule's nodes as each
  > holds it, as a centralized barrier counts arrivals: "each processor updates a small amount of shared state to
  > indicate its arrival and then polls that state to determine when all of the processors have arrived" (J. M.
  > Mellor-Crummey and M. L. Scott, Algorithms for scalable synchronization on shared-memory multiprocessors, ACM TOCS
  > 9(1), 1991, §3.1). It TO-broadcasts no submission of the static schedule before every node of it has arrived.
  - Placed by: "**Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its
    functions." (design/aot-compilation-change-request.md:319)
- **A10 algorithms: independence, commuting operations, the bound**
  (design/aot-compilation-change-request.md:607-617): stated under no stage named, which its clause names with this
  stage.
- **A10 its output (the receiving types)** (design/aot-compilation-change-request.md:629-634, 638-639): stated under
  scheduled value, which its clause names with this stage.
- **A10 produced by, consumed by (the schedule)** (design/aot-compilation-change-request.md:635, 637-638): stated
  under schedule, which its clause names with this stage.
- **A10 produced by, consumed by (the program issues calls ahead)** (design/aot-compilation-change-request.md:635-637)
  > At run time the program issues calls ahead, as futures, in the order the analysis allows, as Pathways dispatches
  > asynchronously (Barham et al., 2022, §4.5).
  - Placed by: "The schedule and the runtime take the analysis's output, carried in A8's IR."
    (design/aot-compilation-change-request.md:591-592); the clause's own "At run time the program issues calls ahead"
    (design/aot-compilation-change-request.md:635-636)
- **Appendix item 2: data structures and algorithms (the queue, its size, the merge)**
  (design/aot-compilation-change-request.md:707-716)
  > **The data structures:** - Fig. 5's `pending` and `nextdeliver` at each node (A7); - one first-in, first-out queue
  > for each static schedule at each node, synchronous dataflow's (Lee and Messerschmitt, 1987, below), each entry one
  > delivered message with its sequence number; - the program's IPC graph (BSL §4.1, below), from which each queue's
  > size is computed. - **The algorithms:** - delivery by Fig. 5's destinations (Défago, Schiper and Urbán, 2004); -
  > each queue's size by BSL's self-timed buffer bound, with Convert-to-SC-graph and DetermineDelays for a feedforward
  > edge; - the bridge's taking by a multiway merge on the sequence number (Knuth, vol. 3, §5.4.1).
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node.
    The schedule stage (`mesh_plan`) computes each queue's size." (design/aot-compilation-change-request.md:705-706)
- **Appendix item 2: what is ordered, and what is not; interleaving**
  (design/aot-compilation-change-request.md:720-727, 859): stated under bridge, which its clause names with this
  stage.
- **Appendix item 2: delivery at a node (Fig. 5's destinations; no wait on an earlier call)**
  (design/aot-compilation-change-request.md:744-750, 849)
  > **Each node runs the destinations' code of their Fig. 5, as written:** - "when receive (m, seqnum)": pending :=
  > pending ∪ {(m, seqnum)}; - "while ∃(m′, seqnum′) ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver :=
  > nextdeliver + 1". - Delivery does not wait for any earlier call to finish, since Fig. 5 has no such condition. A
  > call still running at a node delays no later call's delivery there. - **Delivering a call releases its execution
  > at the node.** Its pages then go as their data allows. Each waits only on its own dependences in the task graph,
  > and those include the lags C3.3 gives each value's copies. [...] deliver messages as Fig. 5's destinations do;
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node."
    (design/aot-compilation-change-request.md:705)
- **Appendix item 2: work and transfers run at once** (design/aot-compilation-change-request.md:751-759)
  > **Work and transfers run at once.** The node's GPU, CPU or ANE work and the bridge's transfers proceed
  > concurrently: - Pathways §4.5: "systems can take advantage of asynchronous APIs to overlap computation with
  > coordination"; computations "may still be interleaved with those submitted by other concurrently executing
  > programs". - Active messages integrate communication with computation, so that both proceed at once (T. von
  > Eicken, D. E. Culler, S. C. Goldstein and K. E. Schauser, Active messages: a mechanism for integrated
  > communication and computation, ISCA 1992). - The bridge serves its queues of page descriptors while the program
  > computes, as a user-level network interface does (U-Net; VIA; design/plan-to-bridge-change-request.md R1).
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node."
    (design/aot-compilation-change-request.md:705); the clause's own "The bridge serves its queues of page descriptors
    while the program computes" (design/aot-compilation-change-request.md:758)
- **Appendix item 2: the queue is laid out with the schedule's pages and read in place**
  (design/aot-compilation-change-request.md:760-763): stated under bridge, which its clause names with this stage.
- **Appendix item 2: nothing waits to write; the only waits** (design/aot-compilation-change-request.md:838-841)
  > **Nothing waits to write, and nothing checks for a full queue.** Every queue is a BBS edge with its bound as its
  > size, so no delivery reaches an entry the bridge at the node has not taken. The only waits are at the sinks of the
  > added synchronization edges, on a count that only rises. Each ends when its source moves on, or when the view ends
  > and the static schedule with it.
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node."
    (design/aot-compilation-change-request.md:705); "The count its source raises is therefore the program's starts of
    the static schedule at that node." (design/aot-compilation-change-request.md:825-826)
- **Appendix item 2, change: the queues replace the sequence counters, taken[p] and the one channel**
  (design/aot-compilation-change-request.md:850-854)
  > publish each static schedule's deliveries to the bridge at each node through its own queue above, read in place
  > and taken by the multiway merge, with no copy, no wait and no check. Each queue's size is its self-timed buffer
  > bound in the program's IPC graph, and the feedforward queues' synchronization edges come from Convert-to-SC-graph
  > and DetermineDelays, all at compile time (BSL, above). This replaces the sequence counters (`sequences`),
  > `taken[p]` and the parent's wait while they are full, and the one channel for every static schedule;
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node.
    The schedule stage (`mesh_plan`) computes each queue's size." (design/aot-compilation-change-request.md:705-706)
- **Appendix item 2, change: remove the start's wait and the late-loading port's refusal**
  (design/aot-compilation-change-request.md:855-858)
  > remove the start's wait for every port to load the static schedule, and the late-loading port's refusal of it
  > (`mesh_ports_held`). Nothing requires them. "a send that arrives first is held until its receive is posted (§3),
  > and a loss cancels the plan anyway" (Claude's turn in "The turns" that lists "Every wait, and whether anything
  > requires it", which says of this one "remove");
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node."
    (design/aot-compilation-change-request.md:705)
- **Appendix item 2, change: remove the coordination's Q3′** (design/aot-compilation-change-request.md:860-862)
  > remove the coordination's Q3′: the sequencer's number never goes back to 0, and no node drops a pending message
  > when its map changes. The number begins again at 1 only at a view's creation, by A7's views; the old view's
  > messages are flushed before the new view is installed.
  - Placed by: "**Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node."
    (design/aot-compilation-change-request.md:705); "**The sequencer** is the bridge of the highest-numbered node the
    map reaches" (design/aot-compilation-change-request.md:382)
- **Appendix item 4 (no truncation at one end; no handler)** (design/aot-compilation-change-request.md:882-884, 892):
  stated under scheduled value, which its clause names with this stage.
- **Appendix item 6: the compiled function's name; alike at every rank**
  (design/aot-compilation-change-request.md:1006-1007, 1026): stated under scheduled value, which its clause names
  with this stage.
- **Appendix item 6, change: any number of compiled functions at once (mesh_plan)**
  (design/aot-compilation-change-request.md:1021-1021): stated under scheduled value, which its clause names with this
  stage.
- **Appendix item 6, stage: a program holds one scheduled value for each compiled function (the compile chain)**
  (design/aot-compilation-change-request.md:1004-1005): stated under scheduled value, which its clause names with this
  stage.
- **Appendix item 6, change: one allocation over the whole program; deferred reclamation**
  (design/aot-compilation-change-request.md:1022-1024): stated under schedule, which its clause names with this stage.
- **Appendix item 6, change: each call launches one execution** (design/aot-compilation-change-request.md:1025-1025)
  > have each call launch one execution of its function;
  - Placed by: "**Stage:** the scheduled value and the runtime. A program holds one scheduled value for each compiled
    function, and the runtime calls each." (design/aot-compilation-change-request.md:1004-1005)
- **The files and functions: rdma/mesh-bridge.h** (design/aot-compilation-change-request.md:1079-1079): stated under
  bridge, which its clause names with this stage.
- **The files and functions: rdma/mesh-records.c** (design/aot-compilation-change-request.md:1080-1080): stated under
  bridge, which its clause names with this stage.

### Program: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **The data flow (stage table and its closing sentence)** (design/compilation-composition-change-request.md:35-36):
  stated under scheduled value, which its clause names with this stage.
- **1 / Write: conversion completed before the program runs**
  (design/compilation-composition-change-request.md:121-122): stated under lower, which its clause names with this
  stage.
- **2 / Composes: one schedule orders functions and transfers together**
  (design/compilation-composition-change-request.md:132-134): stated under schedule, which its clause names with this
  stage.
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **3 / Write, second bullet: callers call them, own copies deleted**
  (design/compilation-composition-change-request.md:150-157): stated under graph module, which its clause names with
  this stage.
- **3 / Write, second bullet: the four named deletions** (design/compilation-composition-change-request.md:155-157):
  stated under schedule, which its clause names with this stage.
- **4 / Write: placement and costs are operands** (design/compilation-composition-change-request.md:170-173): stated
  under caller, which its clause names with this stage.
- **4 / Write: runtime consumes programs and remaining inputs; self-timed calls**
  (design/compilation-composition-change-request.md:173-175)
  > Runtime consumes the resulting programs and the remaining inputs, including actual arrival and completion state
  > and permitted indexes within compiled shapes. A function is called, in the compiled order, when the data it reads
  > is available.
  - Placed by: "**The algorithm** is specialization (S6) at compilation and self-timed execution (S7) at run time."
    design/compilation-composition-change-request.md:169
- **4 / Write: neither execution nor the bridge computes its own order or placement**
  (design/compilation-composition-change-request.md:175-176)
  > Every node's execution and the bridge's use the order and placement in the scheduled value; neither computes its
  > own.
  - Placed by: "the schedule produces the scheduled value; the runtime and the bridge read it."
    design/compilation-composition-change-request.md:170-171
- **4 / Write: signature and external observation cover this execution**
  (design/compilation-composition-change-request.md:176-177): stated under lower, which its clause names with this
  stage.
- **5 / data structure and Composes: one compiled native program per node**
  (design/compilation-composition-change-request.md:196-202): stated under lower, which its clause names with this
  stage.
- **5 / Replaces, site 2** (design/compilation-composition-change-request.md:208-210): stated under lower, which its
  clause names with this stage.
- **5 / Write: one definition of each repeated behavior** (design/compilation-composition-change-request.md:215-217):
  stated under lower, which its clause names with this stage.
- **6 / Write: what the emitted program preserves** (design/compilation-composition-change-request.md:263-266): stated
  under lower, which its clause names with this stage.
- **Performance correction / Constants** (design/compilation-composition-change-request.md:344-366): stated under
  lower, which its clause names with this stage.
- **Complete replacement / all in one commit workgroup** (design/compilation-composition-change-request.md:412-414):
  stated under lower, which its clause names with this stage.
- **Complete replacement / operand table row 5: Executable artifact**
  (design/compilation-composition-change-request.md:429-429): stated under lower, which its clause names with this
  stage.
- **Complete replacement / operands kept until finalization; publication before execution**
  (design/compilation-composition-change-request.md:431-434): stated under lower, which its clause names with this
  stage.
- **7 / Composes: the archive stores the scheduled value** (design/compilation-composition-change-request.md:505-507):
  stated under save and load, which its clause names with this stage.
- **7 / Composes: lowering emits the native program, the runtime calls it**
  (design/compilation-composition-change-request.md:505-507): stated under lower, which its clause names with this
  stage.
- **7 / Replaces: MetalLoopTest and its whole-queue waits** (design/compilation-composition-change-request.md:508-510)
  > **Replaces:** the code of `MetalProgramExecute`/`MetalNativeStep` that reads positions, tasks, guards and a loop
  > stack at run time; `MetalLoopTest` and the whole-queue waits it makes, which are deleted and not replaced; and
  > repeated static command-kind, cut, pass and binding discovery inside `MetalRowEncode`/`MetalReplayRun`.
  - Placed by: "that reads positions, tasks, guards and a loop stack at run time; `MetalLoopTest` and the whole-queue
    waits it makes, which are deleted and not replaced" design/compilation-composition-change-request.md:508-509
- **7 / Replaces: run-time readers of positions, tasks, guards; static discovery**
  (design/compilation-composition-change-request.md:508-511): stated under lower, which its clause names with this
  stage.
- **7 / table row 1: typed numerical operations** (design/compilation-composition-change-request.md:520-520): stated
  under lower, which its clause names with this stage.
- **7 / table row 2: partitioned computation, transfers, compiled order**
  (design/compilation-composition-change-request.md:521-521): stated under lower, which its clause names with this
  stage.
- **7 / table row 3: realized storage** (design/compilation-composition-change-request.md:522-522): stated under
  lower, which its clause names with this stage.
- **7 / table row 4: target commands** (design/compilation-composition-change-request.md:523-523): stated under lower,
  which its clause names with this stage.
- **7 / table row 5: lowered synchronization and its cause**
  (design/compilation-composition-change-request.md:524-524): stated under lower, which its clause names with this
  stage.
- **7 / contract paragraph** (design/compilation-composition-change-request.md:527-529): stated under lower, which its
  clause names with this stage.
- **7 / residual runtime operands and waits** (design/compilation-composition-change-request.md:531-535)
  > Residual runtime operands are actual changing data: inputs, supported dynamic indexes, addresses bound at
  > preparation/load, arrival/completion and resource state. Their arithmetic and necessary API work remain compiled
  > operations. A remaining wait must follow from actual producer/consumer effects and the selected target mechanism.
  > Unrelated queued work does not justify a wait for the whole queue. Readiness is not permission to repeat placement
  > or dependency analysis.
  - Placed by: "Residual runtime operands are actual changing data: inputs, supported dynamic indexes, addresses bound
    at preparation/load, arrival/completion and resource state."
    design/compilation-composition-change-request.md:531-532; "self-timed execution of what remains (S7)"
    design/compilation-composition-change-request.md:504
- **7 / performance paragraph** (design/compilation-composition-change-request.md:537-540): stated under lower, which
  its clause names with this stage.

### Program: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **C3.1 "Written by, read by"** (design/disjoint-pages-change-request.md:825-829): stated under lower, which its
  clause names with this stage.
- **C3.2 (d) nothing laid out when created** (design/disjoint-pages-change-request.md:841-842): stated under schedule,
  which its clause names with this stage.
- **C3.6 (5) final wait bounded** (design/disjoint-pages-change-request.md:887-887)
  > `e2b_replay.m`'s final wait for completion is bounded.
  - Placed by: "- **Runtime.** ... `tools/e2b_replay.m` passes the `struct mesh_function` array to `mesh_plan`,
    commits the command buffers and reads the `struct mesh_planned` and the `struct mesh_status`."
    (design/disjoint-pages-change-request.md:397-400; the ellipsis stands for the sentence on the bridge at :397-398)
- **C3.7 algorithm 5** (design/disjoint-pages-change-request.md:905-905): stated under bridge, which its clause names
  with this stage.
- **C4 metal-microbench deleted: metal_recording.m** (design/disjoint-pages-change-request.md:932-934): stated under
  lower, which its clause names with this stage.
- **C4 metal-microbench deleted: tools/e2b_replay.m** (design/disjoint-pages-change-request.md:935-937)
  > tools/e2b_replay.m: `NULL, 0` at line 579; the refusals where steps differ from executions (24ec8cda, cfada099,
  > and 472390a8's comment); the zeroing of every local operand's whole extent (line 627) and of every pooled
  > intermediate (line 840).
  - Placed by: "- **Runtime.** ... `tools/e2b_replay.m` passes the `struct mesh_function` array to `mesh_plan`,
    commits the command buffers and reads the `struct mesh_planned` and the `struct mesh_status`."
    (design/disjoint-pages-change-request.md:397-400; the ellipsis stands for the sentence on the bridge at :397-398)
- **D.3.4 data structure** (design/disjoint-pages-change-request.md:1408-1410)
  > **The data structure** is the `MTL4CommandBuffer` from an `MTL4CommandAllocator`, committed to an
  > `MTL4CommandQueue`, with the queue's event, whose value only rises (C3.7). Every command buffer this code makes is
  > one: the decode, the prefill, the preparation and fills, the profiling, the recorder's own copies and
  > tools/native_bench.m.
  - Placed by: "**Written by, read by.** `tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and
    commit the command buffers." (design/disjoint-pages-change-request.md:1418-1419); "- **Runtime.** ...
    `tools/e2b_replay.m` passes the `struct mesh_function` array to `mesh_plan`, commits the command buffers"
    (design/disjoint-pages-change-request.md:397-399; the ellipsis stands for the sentence on the bridge at :397-398)
- **D.3.4 algorithm** (design/disjoint-pages-change-request.md:1411-1415)
  > **The algorithm** is one commit per later pass (C3.1): 1. the commands before the later pass are committed; 2.
  > `waitForEvent:value:` waits on the event at p + 1 (MTL4CommandQueue.h:274), and the wait is the GPU queue's, not a
  > CPU thread's. No source in this document's turns defines p; 3. the later pass and what follows go in the next
  > command buffer.
  - Placed by: "**Written by, read by.** `tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and
    commit the command buffers." (design/disjoint-pages-change-request.md:1418-1419); "- **Runtime.** ...
    `tools/e2b_replay.m` passes the `struct mesh_function` array to `mesh_plan`, commits the command buffers"
    (design/disjoint-pages-change-request.md:397-399; the ellipsis stands for the sentence on the bridge at :397-398)
- **D.3.4 completion** (design/disjoint-pages-change-request.md:1416-1417)
  > A completion is the queue's `signalEvent:value:` (MTL4CommandQueue.h:266) and a bounded wait on that event, or an
  > `MTL4CommitFeedback`.
  - Placed by: "**Written by, read by.** `tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and
    commit the command buffers." (design/disjoint-pages-change-request.md:1418-1419); "- **Runtime.** ...
    `tools/e2b_replay.m` passes the `struct mesh_function` array to `mesh_plan`, commits the command buffers"
    (design/disjoint-pages-change-request.md:397-399; the ellipsis stands for the sentence on the bridge at :397-398)
- **D.3.4 "Written by, read by"** (design/disjoint-pages-change-request.md:1418-1419)
  > **Written by, read by.** `tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and commit the
  > command buffers.
  - Placed by: "`tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and commit the command
    buffers." (design/disjoint-pages-change-request.md:1418-1419); "- **Runtime.** ... `tools/e2b_replay.m` passes the
    `struct mesh_function` array to `mesh_plan`, commits the command buffers"
    (design/disjoint-pages-change-request.md:397-399; the ellipsis stands for the sentence on the bridge at :397-398);
    "- **Lowering.** The kernels `metal_recording.m` records" (design/disjoint-pages-change-request.md:401)
- **D.4 metal-microbench deleted (1): classic command queues and buffers**
  (design/disjoint-pages-change-request.md:1457-1460)
  > Every MTLCommandQueue and MTLCommandBuffer this code creates outside the delegate's capture: tools/e2b_replay.m
  > (`commit_bounded`, and lines 1048, 1069, 1118 and 1277); metal_recording.m (`MetalCompletion` and `MetalCompleted`
  > outside the capture, line 492, and the `id<MTLCommandBuffer>` of `MetalReplayRecorded` and `MetalReplayEncode`);
  > tools/native_bench.m (lines 97 and 466).
  - Placed by: "**Written by, read by.** `tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and
    commit the command buffers." (design/disjoint-pages-change-request.md:1418-1419); "- **Runtime.** ...
    `tools/e2b_replay.m` passes the `struct mesh_function` array to `mesh_plan`, commits the command buffers"
    (design/disjoint-pages-change-request.md:397-399; the ellipsis stands for the sentence on the bridge at :397-398)
- **D.4 metal-microbench deleted (2): per-step instances** (design/disjoint-pages-change-request.md:1461-1461): stated
  under schedule, which its clause names with this stage.

### Program: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R4 behaviour at run time, 1** (design/plan-to-bridge-change-request.md:2180-2184)
  > A function that needs its operand whole starts when every page of that operand is present, and not before. It
  > waits for nothing else. Bhattacharyya, Sriram and Lee, §2, as design/aot-compilation-change-request.md quotes it:
  > "Each processor executes the actors assigned to it in the order specified at compile time. Before firing an actor,
  > a processor waits for the data needed by that actor to become available". Its data is every page of its operand
  > (below: "one task with an edge from every page of its operand").
  - Placed by: "**Behaviour at run time.** "Coflow" and "independent flow" here are not figures of speech. They name
    what a run does, and each statement below is something a run either does or does not do."
    design/plan-to-bridge-change-request.md:2178-2179
- **R4 behaviour at run time, 2** (design/plan-to-bridge-change-request.md:2185-2188)
  > A function that takes pages in any order has one task for each page. Each task starts when its own page is
  > present. It does not wait for another page of the same operand. Each output page is released when it is written
  > (below: I-structures, "a reader waits only for the element it reads, with no barrier over the structure").
  - Placed by: "**Behaviour at run time.** "Coflow" and "independent flow" here are not figures of speech. They name
    what a run does, and each statement below is something a run either does or does not do."
    design/plan-to-bridge-change-request.md:2178-2179
- **R8 data structure and algorithm: one offline allocation at compile time**
  (design/plan-to-bridge-change-request.md:2429-2432): stated under schedule, which its clause names with this stage.
- **R8 pages released at retirement by deferred reclamation** (design/plan-to-bridge-change-request.md:2433-2434)
  > **A program's pages are released at its retirement by deferred reclamation** (P. E. McKenney and J. D. Slingwine,
  > Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, PhD thesis, Cambridge, 2004; C3.4).
  - Placed by: the reader.

### Program: [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

- **7.2, step 2** (design/problems-of-14b9a757-change-request.md:356-356)
  > That wait completes the node's whole participation (MPI 4.1 §7.1): its sends, its forwards and its receives.
  - Placed by: "the placement stage writes the `wait_tensor` node into the exported graph. 7.1 reads its index as the
    end of the lifetimes of the transfer's ends. At run time the program's call of it reads 7.3's counters."
    design/problems-of-14b9a757-change-request.md:359-360
- **7.3, step 3** (design/problems-of-14b9a757-change-request.md:383-384)
  > The collective's `wait_tensor` at a node waits for its receive counts and its send-completion counters to reach
  > the execution's totals.
  - Placed by: "The program's `wait_tensor` reads them." design/problems-of-14b9a757-change-request.md:395
- **7.3, step 6** (design/problems-of-14b9a757-change-request.md:390-391)
  > The program's wait returns ECANCELED once the error count is nonzero, after the count and the error count together
  > reach the execution's total.
  - Placed by: "The program's wait returns ECANCELED" design/problems-of-14b9a757-change-request.md:390; "The
    program's `wait_tensor` reads them." design/problems-of-14b9a757-change-request.md:395
- **8, step 5** (design/problems-of-14b9a757-change-request.md:427-429)
  > Each is loaded once in make(), before execution 0: the node's slice is taken from the state_dict tensor, or from
  > that tensor view of it, by the partition dimension and the node's rank, and copied into the value's pages by
  > torch's `copy_`, as Megatron-LM copies the rank's pieces into the layer's weight at construction (:156–165).
  - Placed by: "make() reads the state_dict and those pages and does steps 5 and 6."
    design/problems-of-14b9a757-change-request.md:434; "`make()` is the runtime's construction at a node"
    design/problems-of-14b9a757-change-request.md:81
- **8, step 6** (design/problems-of-14b9a757-change-request.md:430-430)
  > The whole attribute is loaded only where the node reads it whole.
  - Placed by: "make() reads the state_dict and those pages and does steps 5 and 6."
    design/problems-of-14b9a757-change-request.md:434; "`make()` is the runtime's construction at a node"
    design/problems-of-14b9a757-change-request.md:81
- **11, step 5** (design/problems-of-14b9a757-change-request.md:501-501)
  > **The program:** it learns of the cancellation through the raised counts (7.3, steps 4 to 6).
  - Placed by: "**The program:** it learns of the cancellation through the raised counts (7.3, steps 4 to 6)."
    design/problems-of-14b9a757-change-request.md:501
- **13, step 3** (design/problems-of-14b9a757-change-request.md:554-555): stated under bridge, which its clause names
  with this stage.

### Program: [torch-capture-change-request.md](torch-capture-change-request.md)

- **2, "It takes / It writes" (the IPC graph is a function of the static schedule)**
  (design/torch-capture-change-request.md:460-464): stated under argument_buffers, which its clause names with this
  stage.
- **4, "It takes / It writes" (the event's value)** (design/torch-capture-change-request.md:634-635)
  > **It takes** each execution's start, which `mesh_transfers_start` (rdma/mesh-dataflow.c:734–736) writes as its
  > store of `submissions`, a field of `struct mesh_state`. **It writes** the event's value, which the command queue's
  > `waitForEvent:value:` reads.
  - Placed by: the reader.
- **4, Write, first bullet (execution t's waits; no fixed count)** (design/torch-capture-change-request.md:652-655)
  > Execution t's waits are those of the requests started for execution t, each awaited once its start is made ("The
  > request becomes active once the call is made"), for any number of executions ("zero or more repetitions"). No
  > count of executions is fixed before the first. A start is `mesh_transfers_start`'s store of `submissions`
  > (rdma/mesh-dataflow.c:734–736).
  - Placed by: the reader.
- **4, Write, second bullet (the waits end at the free)** (design/torch-capture-change-request.md:656-657)
  > The waits end at the free, once the requests started before it are complete ("deallocated only after it becomes
  > inactive").
  - Placed by: the reader.
- **4, "The value the event is set to at a failure", Write** (design/torch-capture-change-request.md:674-678)
  > at a failure the event is set to UINT64_MAX, the greatest value of its `uint64_t`, as `complete_all` sets `done`
  > to UINT_MAX: every wait on it, present and future, for any value, is satisfied (await(E, v) returns once E ≥ v),
  > with no count of executions. The CPU's waits raise the error (ULFM §16.3.1). The event is not signalled again
  > (`reinit_completion` is `complete_all`'s condition for reuse; a revoked object "can never be repaired"); the next
  > view's requests take a new event.
  - Placed by: the reader.

### Program: [unequal-shards-change-request.md](unequal-shards-change-request.md)

- **Dropped - heterogeneity.md R11** (design/unequal-shards-change-request.md:495-498)
  > Dropped as not needed to implement the requirements above (the operator: "if any of them are optional, drop
  > them"): `design/heterogeneity.md` R11, changing the tuples while a program runs; a program's shapes are fixed when
  > it is compiled (`aot-compilation-change-request.md` A1, the operator: "the use of cudagraphs style 'shapes are
  > compiled but indexes inside of shapes might not need to be'"), and a different tuple is a different compilation.
  - Placed by: the reader.

### Program: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **3 / Write: one schedule stage** (design/torch-source-lowering-change-request.md:241-243): stated under schedule,
  which its clause names with this stage.

### Program: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **2 (Consumed by)** (design/integration-finishing-change-request.md:45-46)
  > **Consumed by** the node's replay, which executes its commands in that list's order: Lee and Ha's self-timed
  > order, as design/torch-source-lowering-change-request.md item 3 quotes it.
  - Placed by: "**Consumed by** the node's replay, which executes its commands in that list's order"
    design/integration-finishing-change-request.md:45
- **7 (Consumed by)** (design/integration-finishing-change-request.md:78-78): stated under lower, which its clause
  names with this stage.

### Program: [page-queue-change-request.md](page-queue-change-request.md)

- **The operator's statements: The page queue** (design/page-queue-change-request.md:81-88): stated under bridge,
  which its clause names with this stage.

## save and load

`save` and `load`: the archive.

### save and load: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **The stages each requirement belongs to (1-6)** (design/aot-compilation-change-request.md:86-98): stated under
  export, which its clause names with this stage.
- **A9 stage and types** (design/aot-compilation-change-request.md:520-523): stated under scheduled value, which its
  clause names with this stage.
- **A9 nothing else is serialized** (design/aot-compilation-change-request.md:536-537): stated under scheduled value,
  which its clause names with this stage.

### save and load: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **The data flow (stage table and its closing sentence)** (design/compilation-composition-change-request.md:35-36):
  stated under scheduled value, which its clause names with this stage.
- **2 / Composes: one schedule orders functions and transfers together**
  (design/compilation-composition-change-request.md:132-134): stated under schedule, which its clause names with this
  stage.
- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **7 / Composes: the archive stores the scheduled value** (design/compilation-composition-change-request.md:505-507)
  > - **Composes:** the schedule produces the scheduled value. Lowering takes it and emits the native program: the
  > table below gives, per operand, what the emitted program preserves and what it does not read. The runtime calls
  > that program; the archive stores the scheduled value.
  - Placed by: "the archive stores the scheduled value." design/compilation-composition-change-request.md:507
- **7 / Composes: lowering emits the native program, the runtime calls it**
  (design/compilation-composition-change-request.md:505-507): stated under lower, which its clause names with this
  stage.
- **7 / residual runtime operands and waits** (design/compilation-composition-change-request.md:531-535): stated under
  Program, which its clause names with this stage.

### save and load: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **1 / Consumed by: SHA-256 in the artifact** (design/synchronization-signature-change-request.md:103-104)
  > **Consumed by:** it is printed in the three formats of item 2, the monitor's D program is generated from it (item
  > 4), and its SHA-256 is part of the compiled artifact.
  - Placed by: the reader.
- **6 / Write** (design/synchronization-signature-change-request.md:161-162): stated under lower, which its clause
  names with this stage.

### save and load: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **3 / Write: one schedule stage** (design/torch-source-lowering-change-request.md:241-243): stated under schedule,
  which its clause names with this stage.

## graph module

`rdma/mesh_graph.py`, `rdma/mesh-graph.h` and `rdma/mesh-graph.c`.

### graph module: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **A4 data structure and algorithm: packed spanning trees, packing**
  (design/aot-compilation-change-request.md:226-231): stated under schedule, which its clause names with this stage.
- **A10 the graph operations are the graph module's (the functions)**
  (design/aot-compilation-change-request.md:618-619, 625)
  > **The graph operations:** the order's transitive closure, the walk of commuting chains, and the width by
  > Fulkerson's matching in Hopcroft and Karp's phases. [...] The three graph operations are functions of
  > rdma/mesh_graph.py, beside `walk` and `reach`.
  - Placed by: "**Its graph operations are the graph module's.**" (design/aot-compilation-change-request.md:620); "The
    three graph operations are functions of rdma/mesh_graph.py" (design/aot-compilation-change-request.md:625)
- **A10 the graph operations are the graph module's (callers; Interpreter; wherever mesh walks a graph)**
  (design/aot-compilation-change-request.md:625-628)
  > As that module states, "A caller supplies its vertices and a neighbours function." rdma/mesh_compile.py calls them
  > and holds no copy. - A body's symbolic execution runs on torch.fx.Interpreter, not on a second loop over a graph's
  > nodes. - The same holds wherever mesh's code walks a graph.
  - Placed by: "**Its graph operations are the graph module's.**" (design/aot-compilation-change-request.md:620)
- **Appendix item 2: which files define them** (design/aot-compilation-change-request.md:829-833)
  > **Which files define them.** The strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle mean,
  > MinDelay's binary search and DetermineDelays are functions of the graph module's C half, rdma/mesh-graph.h and
  > mesh-graph.c. They take the caller's vertices, edges, delays and times as operands (A10: "The same holds wherever
  > mesh's code walks a graph"). Their caller is `mesh_plan`, which holds each static schedule's routes and u when it
  > computes it.
  - Placed by: The clause's own "are functions of the graph module's C half, rdma/mesh-graph.h and mesh-graph.c"
    (design/aot-compilation-change-request.md:830-831)
- **The files and functions: rdma/mesh_compile.py** (design/aot-compilation-change-request.md:1064-1069): stated under
  place, which its clause names with this stage.
- **The files and functions: rdma/mesh_graph.py** (design/aot-compilation-change-request.md:1070-1071)
  > **rdma/mesh_graph.py:** the order's transitive closure, the walk of commuting chains, and the width by Fulkerson's
  > matching in Hopcroft and Karp's phases (A10).
  - Placed by: The clause's own "**rdma/mesh_graph.py:**" (design/aot-compilation-change-request.md:1070); "**Its
    graph operations are the graph module's.**" (design/aot-compilation-change-request.md:620)
- **The files and functions: rdma/mesh-graph.h, mesh-graph.c** (design/aot-compilation-change-request.md:1072-1073)
  > **rdma/mesh-graph.h, mesh-graph.c:** the strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle
  > mean, MinDelay's binary search and DetermineDelays (item 2).
  - Placed by: The clause's own "**rdma/mesh-graph.h, mesh-graph.c:**"
    (design/aot-compilation-change-request.md:1072); "are functions of the graph module's C half, rdma/mesh-graph.h
    and mesh-graph.c" (design/aot-compilation-change-request.md:830-831)
- **The files and functions: rdma/mesh-collective.c (fed by A2's collectives)**
  (design/aot-compilation-change-request.md:1074-1076): stated under schedule, which its clause names with this stage.
- **The files and functions: rdma/mesh-collective.c (routing, u, queue sizes, sets)**
  (design/aot-compilation-change-request.md:1074-1078): stated under schedule, which its clause names with this stage.

### graph module: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **3 / Write, first bullet** (design/compilation-composition-change-request.md:153-154)
  > `rdma/mesh_graph.py` and `rdma/mesh-graph.{h,c}` define the shared graph functions, which take the caller's
  > operands.
  - Placed by: "`rdma/mesh_graph.py` and `rdma/mesh-graph.{h,c}` define the shared graph functions"
    design/compilation-composition-change-request.md:153
- **3 / Write, second bullet: callers call them, own copies deleted**
  (design/compilation-composition-change-request.md:150-157)
  > - **Composes:** placement and the one schedule call the graph functions; the schedule's result is in the scheduled
  > value, which lowering and the runtime read. - **Write:** - `rdma/mesh_graph.py` and `rdma/mesh-graph.{h,c}` define
  > the shared graph functions, which take the caller's operands. - Callers call those functions, and their own copies
  > are deleted. Deleted, as the existing requests name them: the second scheduling solve, copied graph walks,
  > placement of program values at run time and alternative execution paths.
  - Placed by: "placement and the one schedule call the graph functions"
    design/compilation-composition-change-request.md:150
- **3 / Write, second bullet: the four named deletions** (design/compilation-composition-change-request.md:155-157):
  stated under schedule, which its clause names with this stage.
- **5 / Composes: signature printed by the graph module's printers**
  (design/compilation-composition-change-request.md:201-202)
  > The signature is printed from that program by the graph module's printers.
  - Placed by: "The signature is printed from that program by the graph module's printers."
    design/compilation-composition-change-request.md:201-202
- **5 / Replaces, site 3: signature schema and printers** (design/compilation-composition-change-request.md:213-214)
  > Signature producers and consumers likewise call `mesh-graph`'s schema and printers; a copied Python structure or
  > second scheduling solve is deleted.
  - Placed by: "Signature producers and consumers likewise call `mesh-graph`'s schema and printers"
    design/compilation-composition-change-request.md:213
- **6 / Replaces 1** (design/compilation-composition-change-request.md:246-247)
  > `MetalProgramPrepare` calls `mesh_graph_dependences` and then `mesh_graph_synchronization`, which reconstructs the
  > same dependence edges internally. The first edge result must be the operand of synchronization reduction.
  - Placed by: "`MetalProgramPrepare` calls `mesh_graph_dependences` and then `mesh_graph_synchronization`, which
    reconstructs the same dependence edges internally." design/compilation-composition-change-request.md:246-247
- **Complete replacement / operand table row 4: Program relations**
  (design/compilation-composition-change-request.md:428-428): stated under scheduled value, which its clause names
  with this stage.
- **Source bindings / Compact effects: the mesh-graph function**
  (design/compilation-composition-change-request.md:468-473)
  > The algorithm is ISL's `isl_basic_set_is_empty`, using exact arbitrary-precision arithmetic. ISL's algorithms
  > differ from Omega's; linking ISL is not an implementation of Pugh's elimination steps. It decides whether two
  > bounded affine byte-access sets intersect. **Composes:** the compiler produces the access relations; a function of
  > `mesh-graph` computes their exact integer feasibility and returns dependence causes; scheduling, fusion and
  > lifetime consumers take that relation. Enumerating tensor elements or replacing a strided set by its bounding
  > interval is not this method.
  - Placed by: "a function of `mesh-graph` computes their exact integer feasibility and returns dependence causes"
    design/compilation-composition-change-request.md:471-472
- **Source bindings / Compact effects: producer and consumers**
  (design/compilation-composition-change-request.md:471-472): stated under schedule, which its clause names with this
  stage.

### graph module: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **A4 deleted (4)** (design/disjoint-pages-change-request.md:485-485)
  > `mesh_graph_left_edge`'s colouring, greedy by size.
  - Placed by: "`mesh_graph_left_edge`'s colouring, greedy by size." (design/disjoint-pages-change-request.md:485)

### graph module: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R2 algorithm 1: the copies (C3.3)** (design/plan-to-bridge-change-request.md:2040-2044): stated under schedule,
  which its clause names with this stage.
- **R2 algorithm 2-3: the order and the chains** (design/plan-to-bridge-change-request.md:2045-2053): stated under
  schedule, which its clause names with this stage.
- **R2 algorithm 4: the order over u copies** (design/plan-to-bridge-change-request.md:2054-2058): stated under
  schedule, which its clause names with this stage.
- **R2 where it composes** (design/plan-to-bridge-change-request.md:2112-2121): stated under schedule, which its
  clause names with this stage.
- **R3 where it composes** (design/plan-to-bridge-change-request.md:2147-2155): stated under schedule, which its
  clause names with this stage.
- **R5 data structure: one linear program over completion times** (design/plan-to-bridge-change-request.md:2267-2288):
  stated under schedule, which its clause names with this stage.
- **R5 algorithm steps 1-2: cutting planes, HiGHS dual simplex** (design/plan-to-bridge-change-request.md:2314-2323)
  > 1. Solve the program by cutting planes. The rows of the last kind are one for each set, so there are exponentially
  > many. Hall et al.: "Queyranne (1993) has shown that the linear program given by (1), (3) and (5) is solvable in
  > polynomial time via the ellipsoid algorithm; the key observation is that there is a polynomial-time separation
  > algorithm for the exponentially large class of constraints (5)." Queyranne's separation takes O(n log n) time, and
  > he proposes it for cutting-plane algorithms (M. Queyranne, Structure of a simple scheduling polyhedron, Math.
  > Programming 58, 1993). Solve the program with the rows found so far. Find each resource's violated rows by
  > Queyranne's separation, and add them. Repeat until no row is violated. 2. Each linear program is solved by HiGHS's
  > dual revised simplex method, through the HiGHS library (Q. Huangfu and J. A. J. Hall, Parallelizing the dual
  > revised simplex method, Math. Program. Comput. 10(1), 2018, the paper HiGHS asks its users to cite). The code
  > builds the rows and calls the library. It writes no solver of its own.
  - Placed by: "rdma/mesh-graph.{h,c}: the program's rows, the cutting-plane loop with Queyranne's separation, and the
    calls to HiGHS; the heap order by C̄ (`mesh_graph_heap_order`)."
    design/plan-to-bridge-change-request.md:2364-2365; "Produced by `mesh_plan` (rdma/mesh-collective.c), which calls
    `mesh_graph_completion_times` and `mesh_graph_heap_order` (rdma/mesh-graph.h)."
    design/plan-to-bridge-change-request.md:2358-2359
- **R5 algorithm steps 3-5: priority, one global topological order, each cable's subsequence**
  (design/plan-to-bridge-change-request.md:2324-2332): stated under schedule, which its clause names with this stage.
- **R5 where it composes** (design/plan-to-bridge-change-request.md:2351-2362): stated under schedule, which its
  clause names with this stage.
- **R5 changes by file: rdma/mesh-graph.{h,c}** (design/plan-to-bridge-change-request.md:2364-2365)
  > rdma/mesh-graph.{h,c}: the program's rows, the cutting-plane loop with Queyranne's separation, and the calls to
  > HiGHS; the heap order by C̄ (`mesh_graph_heap_order`).
  - Placed by: "rdma/mesh-graph.{h,c}: the program's rows, the cutting-plane loop with Queyranne's separation, and the
    calls to HiGHS; the heap order by C̄ (`mesh_graph_heap_order`)." design/plan-to-bridge-change-request.md:2364-2365

### graph module: [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

- **14, Change: replaces plan_before and mesh_graph_chains for these items**
  (design/problems-of-14b9a757-change-request.md:903-903): stated under schedule, which its clause names with this
  stage.

### graph module: [torch-capture-change-request.md](torch-capture-change-request.md)

- **2, "It takes / It writes" (the IPC graph is a function of the static schedule)**
  (design/torch-capture-change-request.md:460-464): stated under argument_buffers, which its clause names with this
  stage.
- **2, "The number of locations of a value the CPU writes", Write** (design/torch-capture-change-request.md:492-497):
  stated under argument_buffers, which its clause names with this stage.
- **Source for item 2's execution-time estimates, Write, second sentence (CPU and GPU rates)**
  (design/torch-capture-change-request.md:872-874)
  > CPU work, including the CPU's write of item 2, uses its CPU rates and traffic; a GPU function uses its GPU mapping
  > and rates.
  - Placed by: "In rdma/mesh-graph.h at mesh 57d9f06e, `mesh_graph_actor_time` takes a `struct mesh_graph_machine` and
    a `struct mesh_graph_actor` and writes the time as a `double`." (design/torch-capture-change-request.md:852-853)

### graph module: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **1 / The algorithm** (design/synchronization-signature-change-request.md:98-99)
  > **The algorithm** is the graph module's: `mesh_graph_dependences`, `mesh_graph_synchronization`,
  > `mesh_graph_redundant` and `mesh_graph_schedule`.
  - Placed by: "**The algorithm** is the graph module's" design/synchronization-signature-change-request.md:98
- **4 / Write: every edge the image of a dependence, none redundant**
  (design/synchronization-signature-change-request.md:145-147)
  > every edge has to be the image of a data dependence between two of the command queue, the thread and the bridge,
  > and must not be redundant under the 1997 paper's §7, which `mesh_graph_redundant` implements. So a wait with no
  > dependence under it shows before anything runs.
  - Placed by: "which `mesh_graph_redundant` implements" design/synchronization-signature-change-request.md:146

### graph module: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **2 (It writes)** (design/integration-finishing-change-request.md:42-44)
  > **It writes** the node's list of commands: `mesh_graph_heap_order` over those keys. In mesh rdma/mesh-graph.h that
  > function takes the keys as `const double *priority` and the dependences as `const uint32_t *from` and `const
  > uint32_t *to`, and writes `uint32_t *place`.
  - Placed by: "`mesh_graph_heap_order` over those keys. In mesh rdma/mesh-graph.h that function takes the keys"
    design/integration-finishing-change-request.md:42-43
- **4 (Replaces)** (design/integration-finishing-change-request.md:59-59)
  > **Replaces:** the walk in metal-microbench metal_recording.m `MetalMaskedPass`.
  - Placed by: "### 4. The masked pass in the graph module" design/integration-finishing-change-request.md:57; "that
    walk is a function of mesh rdma/mesh-graph.{h,c}, with caller operands."
    design/integration-finishing-change-request.md:60
- **4 (Write)** (design/integration-finishing-change-request.md:60-60)
  > **Write:** that walk is a function of mesh rdma/mesh-graph.{h,c}, with caller operands.
  - Placed by: "that walk is a function of mesh rdma/mesh-graph.{h,c}, with caller operands."
    design/integration-finishing-change-request.md:60

## bridge

The bridge process and the transport in `rdma/*.c`, which take page lists and move pages.

### bridge: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **A5 produced by, consumed by** (design/aot-compilation-change-request.md:282, 292-293): stated under schedule,
  which its clause names with this stage.
- **A6 produced by, consumed by (the lists)** (design/aot-compilation-change-request.md:306-307): stated under
  schedule, which its clause names with this stage.
- **A7 data structures: m, sn(m), Dest(m)** (design/aot-compilation-change-request.md:367-371, 385-386)
  > **The data structures are Fig. 5's:** the sequencer's `seqnum`; each destination's `nextdeliver` and `pending`;
  > and the message (m, sn(m)), which is rdma/mesh-bridge.h `struct mesh_order`. - **m** is one submission: its static
  > schedule's name (`group`, `epoch`), the identifier of the view it is sent in (below), and a range of its
  > executions [`first`, `end`). One submission is one message however many executions it carries, as Pathways sends
  > "a single message (describing the entire subgraph)" (§4.5). e2b's run is one message. [...] **sn(m)** is struct
  > mesh_order's `sequence`. - **Dest(m)** is the nodes of m's static schedule (struct mesh_order's `nodes`).
  - Placed by: "the message (m, sn(m)), which is rdma/mesh-bridge.h `struct mesh_order`"
    (design/aot-compilation-change-request.md:368); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **A7 the sequencer** (design/aot-compilation-change-request.md:382-384)
  > **The sequencer** is the bridge of the highest-numbered node the map reaches (rdma/mesh-records.c `island_set`),
  > the code's choice, which the class leaves open. "The sequencer is unique, and the responsibility is not normally
  > transfered to another processes" (§7.1). It creates the island's views (below).
  - Placed by: The clause's own "**The sequencer** is the bridge of the highest-numbered node the map reaches
    (rdma/mesh-records.c `island_set`)" (design/aot-compilation-change-request.md:382); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **A7 deliver(m); produced by, consumed by** (design/aot-compilation-change-request.md:387-392): stated under
  Program, which its clause names with this stage.
- **A7 views: a process** (design/aot-compilation-change-request.md:430-432)
  > **A process** is one start of a bridge. A bridge started again is a new process: "In Isis [RB91] a process
  > recovering after a crash is assigned a different identifier (using a new incarnation number)" (Chockler, Keidar
  > and Vitenberg, §3.1.1).
  - Placed by: The clause's own "**A process** is one start of a bridge."
    (design/aot-compilation-change-request.md:430); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **A7 views: the membership service is the sequencer** (design/aot-compilation-change-request.md:433-441)
  > **The membership service is the sequencer.** It decides each view alone, as Amoeba does: "Suspected processes are
  > excluded from the group as the result of the unilateral decision of a single process" (Défago, Schiper and Urbán,
  > §9.1.1). A view's members are the nodes the sequencer's map reaches. The sequencer creates a view at its bridge's
  > start and at each change of its map. It also creates one whenever a member's announcement names a view other than
  > the sequencer's installed one, with a counter not less than that view's identifier. Such a member installed
  > another sequencer's view after the announcement the sequencer counted, and it ignores the sequencer's view by
  > Local Monotonicity. The identifier is "computed based on the values of local counters" (below), so the sequencer
  > computes it again from the counters it now holds. To tell such a member from one that installed the sequencer's
  > own view, each announcement names the process that created its bridge's installed view as well as its counter.
  - Placed by: "**The sequencer** is the bridge of the highest-numbered node the map reaches (rdma/mesh-records.c
    `island_set`)" (design/aot-compilation-change-request.md:382); the clause's own "The sequencer creates a view at
    its bridge's start" (design/aot-compilation-change-request.md:435-436); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **A7 views: the view identifier** (design/aot-compilation-change-request.md:442-447)
  > **The view identifier is Transis's**: "a positive integer. This integer is computed based on the values of local
  > counters, maintained by all processes. This local counter is increased by a process upon each installation"
  > (Chockler, Keidar and Vitenberg, §3.1.1). A bridge's counter is the identifier of the last view it installed, 0
  > before its first. Each bridge's announcement of itself to the sequencer carries its counter, and the sequencer
  > gives a view one more than the greatest counter among its members. So Local Monotonicity holds at every member, a
  > bridge started again among them.
  - Placed by: The clause's own "A bridge's counter is the identifier of the last view it installed"
    (design/aot-compilation-change-request.md:444); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **A7 views: a bridge installs a view** (design/aot-compilation-change-request.md:448-450)
  > **A bridge installs a view** when the view's first message reaches it, if the view names it a member (Self
  > Inclusion) and the view's identifier is greater than its counter (Local Monotonicity). It ignores any other view's
  > messages.
  - Placed by: The clause's own "**A bridge installs a view**" (design/aot-compilation-change-request.md:448);
    "**Stage:** the runtime." (design/aot-compilation-change-request.md:319)
- **A7 views: a change of view blocks and flushes** (design/aot-compilation-change-request.md:460-466): stated under
  Program, which its clause names with this stage.
- **A9 stage and types** (design/aot-compilation-change-request.md:520-523): stated under scheduled value, which its
  clause names with this stage.
- **A9 nothing else is serialized** (design/aot-compilation-change-request.md:536-537): stated under scheduled value,
  which its clause names with this stage.
- **A9 the algorithm the bridge runs it by: the self-timed strategy** (design/aot-compilation-change-request.md:538,
  546-565)
  > **The algorithm the bridge runs it by is the self-timed strategy.** [...] **The strategy** is E. A. Lee and S.
  > Ha's self-timed strategy (Scheduling strategies for multiprocessor real-time DSP, Globecom 1989), as BSL states
  > it. Scheduling is "assigning actors in the DFG to processors, ordering execution of these actors on each
  > processor, and determining when each actor fires". - Self-timed keeps the first two from compile time and leaves
  > the third to run time: "it is best to discard the exact timing information from the fully static schedule, but
  > still retain the processor assignment and actor ordering specified by the fully static schedule. This results in
  > the self-timed scheduling strategy of [17]. Each processor executes the actors assigned to it in the order
  > specified at compile time. Before firing an actor, a processor waits for the data needed by that actor to become
  > available" (BSL §2; [17] is Lee and Ha). - BSL's condition for choosing it is this program's: "good estimates for
  > the execution times of actors can be obtained", which here are A3's weighting operand. - **Its terms here:** - The
  > graph is A8's task graph, the program's whole graph (A1, A10), as a CUDA graph is the whole of its captured work.
  > - Its processors are the sends of each adjacent node and direction, R5's cables. - Its assignment is R5's routing.
  > - Its order on each processor is R5's, a subsequence of MSCCLang's heap-built topological order by bottom level.
  > Item 5's sets are the exception: their order is one permutation per execution, Threefry-drawn Fisher–Yates, the
  > same at both ends. - BSL's "determining when each actor fires" is each entry's wait on its dependency, C3's
  > counted completion: BSL's "run-time synchronization when they communicate data".
  - Placed by: The clause's own "**The algorithm the bridge runs it by is the self-timed strategy.**"
    (design/aot-compilation-change-request.md:538); "**Stage:** the scheduled value and its archive."
    (design/aot-compilation-change-request.md:520)
- **A9 every address is fixed at compile time** (design/aot-compilation-change-request.md:569-574): stated under
  scheduled value, which its clause names with this stage.
- **Appendix item 2: data structures and algorithms (the queue, its size, the merge)**
  (design/aot-compilation-change-request.md:707-716): stated under Program, which its clause names with this stage.
- **Appendix item 2: what is ordered, and what is not; interleaving**
  (design/aot-compilation-change-request.md:720-727, 859)
  > **What is ordered, and what is not:** - The fixed sequencer orders function executions, Pathways' unit (P. Barham
  > et al., MLSys 2022, §4.4: "a consistent order of function executions across all programs"). It does not order
  > transactions. - Its delivery is in sequence-number order (X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv.
  > 2004), with no timeout and no skipping. - On each cable, the pages of every released execution interleave (T99).
  > Their order there is the bridge's local order: computed alike at both ends of the cable from what both hold (F. B.
  > Schneider, ACM Comput. Surv. 1990), and never global. [...] interleave released executions on each cable;
  - Placed by: The clause's own "Their order there is the bridge's local order"
    (design/aot-compilation-change-request.md:725-726); "**Stage:** the runtime takes A7's deliveries through one
    queue for each static schedule at each node." (design/aot-compilation-change-request.md:705)
- **Appendix item 2: recorded order** (design/aot-compilation-change-request.md:728-729)
  > **Recorded order:** the bridge may record the order it realized (its traces, rdma/mesh-flow.c `struct mesh_trace`
  > under `MESH_LEDGER`). No collective or data code reads it.
  - Placed by: The clause's own "the bridge may record the order it realized (its traces, rdma/mesh-flow.c `struct
    mesh_trace` under `MESH_LEDGER`)" (design/aot-compilation-change-request.md:728-729)
- **Appendix item 2: work and transfers run at once** (design/aot-compilation-change-request.md:751-759): stated under
  Program, which its clause names with this stage.
- **Appendix item 2: the queue is laid out with the schedule's pages and read in place**
  (design/aot-compilation-change-request.md:760-763)
  > **Each static schedule's delivered calls reach the bridge at a node through a queue of the static schedule's
  > own.** It is laid out with the static schedule's pages, and placed and released with them by the one allocation
  > over the program (C3.2; C3.4). The bridge reads it in place, by view, and nothing copies it. Only the driver moves
  > bytes (design/plan-to-bridge-change-request.md R1).
  - Placed by: The clause's own "The bridge reads it in place, by view, and nothing copies it."
    (design/aot-compilation-change-request.md:762); "**Stage:** the runtime takes A7's deliveries through one queue
    for each static schedule at each node. The schedule stage (`mesh_plan`) computes each queue's size."
    (design/aot-compilation-change-request.md:705-706)
- **Appendix item 2: nothing waits to write; the only waits** (design/aot-compilation-change-request.md:838-841):
  stated under Program, which its clause names with this stage.
- **Appendix item 2, change: the queues replace the sequence counters, taken[p] and the one channel**
  (design/aot-compilation-change-request.md:850-854): stated under Program, which its clause names with this stage.
- **Appendix item 2, change: remove the coordination's Q3′** (design/aot-compilation-change-request.md:860-862):
  stated under Program, which its clause names with this stage.
- **Appendix item 4 (no truncation at one end; no handler)** (design/aot-compilation-change-request.md:882-884, 892):
  stated under scheduled value, which its clause names with this stage.
- **Appendix item 5: the set** (design/aot-compilation-change-request.md:908-912, 927-929): stated under schedule,
  which its clause names with this stage.
- **Appendix item 5: the algorithm (Fisher and Yates over Threefry; Lemire)**
  (design/aot-compilation-change-request.md:913-926)
  > **The algorithm is Fisher and Yates's shuffle over a counter-based generator.** Each execution of a set on a cable
  > is sent and received in a uniformly random permutation of its pages. - The permutation is drawn by Fisher and
  > Yates's shuffle (R. A. Fisher and F. Yates, Statistical Tables, 1938; R. Durstenfeld, Algorithm 235: Random
  > permutation, CACM 7(7), 1964; D. E. Knuth, The Art of Computer Programming vol. 2, §3.4.2, Algorithm P). - Its
  > random numbers come from Threefry-4×64-20, a counter-based generator (J. K. Salmon, M. A. Moraes, R. O. Dror and
  > D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, SC 2011, §4.2; rdma/mesh-flow.c `threefry4x64_20`). Its
  > key is the static schedule's name (item 6), whose group the launcher draws anew at each launch (metal-microbench
  > tools/mesh/run_pair.py: `1 + secrets.randbelow(2**64 - 1)`, passed as `MESH_GROUP`). Its counter is (execution,
  > set). So both ends of a cable draw the same permutation without exchanging anything. - Algorithm P needs a uniform
  > integer in [1, j] at each step. The code takes the high word of j times a random word (rdma/mesh-flow.c
  > `set_draw`), which is Lemire's multiply-and-shift. With Lemire's rejection step, which `set_draw` has at mesh
  > 26701d8e, the integer is exactly uniform (D. Lemire, Fast random integer generation in an interval, ACM Trans.
  > Model. Comput. Simul. 29(1), 2019).
  - Placed by: "The runtime's bridge draws each set's order at each execution."
    (design/aot-compilation-change-request.md:906-907); the clause's own "rdma/mesh-flow.c `threefry4x64_20`"
    (design/aot-compilation-change-request.md:919)
- **Appendix item 5: one permutation of the whole set per execution** (design/aot-compilation-change-request.md:932,
  958-961, 987-988)
  > **One permutation of the whole set per execution, as one renaming on every channel the set crosses.** [...] **So
  > the bridge's assignment to cables fixes positions, not members.** Each channel has the positions the assignment
  > gives it. In each execution, the member at each position is the one σ puts there. The bridge at each node holds a
  > record for every member of every set whose positions it carries, each record a view of that member's page (R1). So
  > any member can go at any of the set's positions. [...] The counter is (execution, set), so a set is drawn once per
  > execution. A draw for each cable departs from the counter.
  - Placed by: "The runtime's bridge draws each set's order at each execution."
    (design/aot-compilation-change-request.md:906-907); the clause's own "**So the bridge's assignment to cables fixes
    positions, not members.**" (design/aot-compilation-change-request.md:958)
- **Appendix item 5: a page's round rule reads a count that only rises**
  (design/aot-compilation-change-request.md:986-986)
  > **The change:** name a page for round j by that count, in place of PM2's test of the phase field.
  - Placed by: the reader.
- **The files and functions: rdma/mesh-bridge.h** (design/aot-compilation-change-request.md:1079-1079)
  > **rdma/mesh-bridge.h:** `struct mesh_order` (A7).
  - Placed by: The clause's own "**rdma/mesh-bridge.h:**" (design/aot-compilation-change-request.md:1079); "the
    message (m, sn(m)), which is rdma/mesh-bridge.h `struct mesh_order`"
    (design/aot-compilation-change-request.md:368); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **The files and functions: rdma/mesh-records.c** (design/aot-compilation-change-request.md:1080-1080)
  > **rdma/mesh-records.c:** `island_set`, the sequencer (A7).
  - Placed by: "**The sequencer** is the bridge of the highest-numbered node the map reaches (rdma/mesh-records.c
    `island_set`)" (design/aot-compilation-change-request.md:382); "**Stage:** the runtime."
    (design/aot-compilation-change-request.md:319)
- **The files and functions: rdma/mesh-flow.c** (design/aot-compilation-change-request.md:1081-1081)
  > **rdma/mesh-flow.c:** `set_draw` and `threefry4x64_20` (item 5); `struct mesh_trace` (item 2).
  - Placed by: "The runtime's bridge draws each set's order at each execution."
    (design/aot-compilation-change-request.md:906-907); "the bridge may record the order it realized (its traces,
    rdma/mesh-flow.c `struct mesh_trace` under `MESH_LEDGER`)" (design/aot-compilation-change-request.md:728-729)

### bridge: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140): stated under no stage named, which its clause names with
  this stage.
- **4 / Write: placement and costs are operands** (design/compilation-composition-change-request.md:170-173): stated
  under caller, which its clause names with this stage.
- **4 / Write: the bridge binds physical transport as R7 specifies**
  (design/compilation-composition-change-request.md:175-175)
  > The bridge binds physical transport as R7 specifies.
  - Placed by: "The bridge binds physical transport as R7 specifies."
    design/compilation-composition-change-request.md:175
- **4 / Write: neither execution nor the bridge computes its own order or placement**
  (design/compilation-composition-change-request.md:175-176): stated under Program, which its clause names with this
  stage.
- **6 / Replaces 2: schedule results are operands of native preparation and of the transport**
  (design/compilation-composition-change-request.md:250-252): stated under scheduled value, which its clause names
  with this stage.
- **Complete replacement / operand table row 5: Executable artifact**
  (design/compilation-composition-change-request.md:429-429): stated under lower, which its clause names with this
  stage.

### bridge: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **A3 "Which function computes it, and which read it"** (design/disjoint-pages-change-request.md:469-474): stated
  under schedule, which its clause names with this stage.
- **A3 "The bridge's record pages"** (design/disjoint-pages-change-request.md:476-476)
  > The bridge's record pages are laid out once, with the region.
  - Placed by: "The bridge's record pages are laid out once, with the region."
    (design/disjoint-pages-change-request.md:476); "from `rdma/mesh-records.c` (the bridge's record pages, line 76)"
    (design/disjoint-pages-change-request.md:412-413)
- **A4 deleted (1)** (design/disjoint-pages-change-request.md:481-482): stated under schedule, which its clause names
  with this stage.
- **A4 deleted (3)** (design/disjoint-pages-change-request.md:484-484)
  > `mesh-flow.c`'s receive window and its check of forwards, and the hdr fields `depth` and `hop_stride` they read.
  - Placed by: "`mesh-flow.c`'s receive window and its check of forwards"
    (design/disjoint-pages-change-request.md:484); "- **Runtime.** The bridge (mesh `rdma/mesh-flow.c`) reads the
    `struct mesh_hop` entries" (design/disjoint-pages-change-request.md:397)
- **C3.1 algorithm 4, bridge half** (design/disjoint-pages-change-request.md:821-824)
  > A completed receive and a cancelled receive each add one to their range's count, as arrivals do on an mbarrier. A
  > cancelled receive does not change its phase, so the later pass finds it one phase behind, and its threads end as
  > an error that the later pass writes into the execution's status word. A link loss is such a cancellation. There is
  > no cancel word.
  - Placed by: "**Written by, read by.** The bridge (`rdma/mesh-flow.c`) writes the phases and the counts."
    (design/disjoint-pages-change-request.md:825)
- **C3.1 algorithm 4, later-pass half** (design/disjoint-pages-change-request.md:821-823): stated under lower, which
  its clause names with this stage.
- **C3.1 "Written by, read by"** (design/disjoint-pages-change-request.md:825-829): stated under lower, which its
  clause names with this stage.
- **C3.2 (a) data structure and types** (design/disjoint-pages-change-request.md:834-836): stated under schedule,
  which its clause names with this stage.
- **C3.3 algorithm 3 (copies, t mod u)** (design/disjoint-pages-change-request.md:851-851): stated under schedule,
  which its clause names with this stage.
- **C3.3 "Written by, read by"** (design/disjoint-pages-change-request.md:852-854): stated under schedule, which its
  clause names with this stage.
- **C3.4 (b) new state in new pages, no wait** (design/disjoint-pages-change-request.md:862-866): stated under
  schedule, which its clause names with this stage.
- **C3.4 (c) deferred reclamation** (design/disjoint-pages-change-request.md:864-866)
  > **The algorithm for release** is deferred reclamation (RCU's `call_rcu`, P. E. McKenney and J. D. Slingwine,
  > Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004): the bridge releases a retired static schedule's
  > pages once none of its ports still has that static schedule loaded, and nothing waits for the release.
  - Placed by: "the bridge releases a retired static schedule's pages once none of its ports still has that static
    schedule loaded" (design/disjoint-pages-change-request.md:865-866); "The bridge reads it, and releases its pages
    after its retirement." (design/disjoint-pages-change-request.md:867-868)
- **C3.4 "Written by, read by"** (design/disjoint-pages-change-request.md:867-868): stated under schedule, which its
  clause names with this stage.
- **C3.6 (1) late cancellation** (design/disjoint-pages-change-request.md:880-881): stated under schedule, which its
  clause names with this stage.
- **C3.6 (2) retired client's schedule** (design/disjoint-pages-change-request.md:882-883)
  > A port that keeps a retired client's static schedule past `MESH_PLATFORM_NS`: removed by C3.4, with the claim at
  > attach.
  - Placed by: "the bridge releases a retired static schedule's pages once none of its ports still has that static
    schedule loaded, and nothing waits for the release" (design/disjoint-pages-change-request.md:865-866)
- **C3.6 (4) link loss and poll bound** (design/disjoint-pages-change-request.md:885-886): stated under lower, which
  its clause names with this stage.
- **C3.7 algorithm 1** (design/disjoint-pages-change-request.md:896-897)
  > A receive's completion is one atomic XOR of one bit of its field, which never carries into the next receive's
  > field. Its one writer is the bridge code that completes the receive.
  - Placed by: "Its one writer is the bridge code that completes the receive."
    (design/disjoint-pages-change-request.md:897); "The bridge (`rdma/mesh-flow.c`) writes the phases and the counts
    and reads the words." (design/disjoint-pages-change-request.md:906-907)
- **C3.7 algorithm 4** (design/disjoint-pages-change-request.md:902-904)
  > A sent range's word has one writer, the range's producer, which stores t + 1 after its last write in execution t.
  > The bridge posts a send of execution t once the word reads at least t + 1. The word starts at 0 when placed, and
  > one word serves every execution.
  - Placed by: "The bridge (`rdma/mesh-flow.c`) writes the phases and the counts and reads the words. Each range's
    producer writes its word." (design/disjoint-pages-change-request.md:906-908); "`mesh_plan`'s allocator places the
    array and the words and writes their starting values." (design/disjoint-pages-change-request.md:906)
- **C3.7 algorithm 5** (design/disjoint-pages-change-request.md:905-905)
  > The counts and the event only rise.
  - Placed by: "The bridge (`rdma/mesh-flow.c`) writes the phases and the counts and reads the words."
    (design/disjoint-pages-change-request.md:906-907); "committed to an `MTL4CommandQueue`, with the queue's event,
    whose value only rises (C3.7)" (design/disjoint-pages-change-request.md:1408-1409)
- **C3.7 "Written by, read by"** (design/disjoint-pages-change-request.md:906-910): stated under schedule, which its
  clause names with this stage.
- **C3.7 data pages** (design/disjoint-pages-change-request.md:911-911)
  > No data page is written but by its producer or by the receive that fills it.
  - Placed by: the reader.
- **C3.7 only zeroing** (design/disjoint-pages-change-request.md:912-912)
  > The only zeroing is mesh-link.c's zeroing of the region's header, once, when the bridge creates the region.
  - Placed by: "mesh-link.c's zeroing of the region's header, once, when the bridge creates the region"
    (design/disjoint-pages-change-request.md:912)
- **C4 mesh deleted: mesh_plan_erased** (design/disjoint-pages-change-request.md:923-924): stated under schedule,
  which its clause names with this stage.
- **C4 mesh deleted: rdma/mesh-dataflow.c claim** (design/disjoint-pages-change-request.md:927-929): stated under
  schedule, which its clause names with this stage.
- **D.3.1 (and the turn's item 1: several programs, MESH_GROUP queue-pair assignment, EBUSY)**
  (design/disjoint-pages-change-request.md:1136-1138)
  > I reasoned from a queue-pair count that belongs only to the bridge and differs between macOS 26 and 27, and I
  > turned it into a limit programs would see (`EBUSY`). That is withdrawn, and I stopped the coordinator before it
  > wrote anything.
  - Placed by: "a queue-pair count that belongs only to the bridge" (design/disjoint-pages-change-request.md:1136)
- **D.3.5** (design/disjoint-pages-change-request.md:1425-1431)
  > **The data structure** is each adjacent node's receive list in the static schedule: every page of every copy, each
  > page named once, so u executions of the list at once. **The algorithm.** The bridge names a page in a new receive
  > only when the last receive that named it has completed. The allocator already orders the adjacent node's next send
  > into that page after this node's reads of the last one (C3.3). **Written by, read by.** `mesh_plan` writes the
  > receive lists. The bridge (`rdma/mesh-flow.c`) reads them and posts the receives. A receive list's entry is a
  > `struct mesh_hop` with `direction` `MESH_RECEIVE` (`rdma/mesh.h`).
  - Placed by: "The bridge names a page in a new receive only when the last receive that named it has completed."
    (design/disjoint-pages-change-request.md:1427); "The bridge (`rdma/mesh-flow.c`) reads them and posts the
    receives." (design/disjoint-pages-change-request.md:1430-1431)
- **D.3.6** (design/disjoint-pages-change-request.md:1434-1445)
  > **The data structure** is the map and its serial, a link-state database (RFC 2328). A static schedule is made
  > against one serial. **The algorithm** is link-state routing's. A change of the map floods to every bridge (RFC
  > 2328 §13; mesh-records.c `map_put`). Routes come from one version of the database, and a change recomputes them
  > (RFC 2328 §16). Where a node's map serial moves away from a static schedule's serial, its bridge cancels every
  > receive of that static schedule that has not completed: `plan_cancel` over every entry, each adding one to its
  > count and writing no phase (C3.1). Every node of the static schedule then reads ENOLINK, with no case for relays.
  > **Written by, read by.** mesh-records.c `map_put` writes the map and its serial. The bridge (`rdma/mesh-flow.c`)
  > compares the serial with the static schedule's and cancels. The recorded kernels' later pass reads the counts and
  > the phases (C3.1). The map is an array of `struct mesh_record` (`rdma/mesh-map.h`), with its serial at hdr
  > `map_serial`. At mesh db17c3ad `plan_stale` (`rdma/mesh-bridge.h`) compares the static schedule's `struct
  > mesh_view_id` (`rdma/mesh.h` `mesh_plan_view`) with hdr `view`, not the map's serial.
  - Placed by: "its bridge cancels every receive of that static schedule that has not completed"
    (design/disjoint-pages-change-request.md:1438-1439); "The bridge (`rdma/mesh-flow.c`) compares the serial with the
    static schedule's and cancels." (design/disjoint-pages-change-request.md:1441-1442)
- **D.4 mesh: rdma/mesh-flow.c deleted** (design/disjoint-pages-change-request.md:1450-1451)
  > rdma/mesh-flow.c: deleted, a receive record posted while its earlier posting is outstanding, and `link_prepare`'s
  > note that relies on the provider copying a request (D.3.5).
  - Placed by: "rdma/mesh-flow.c: deleted, a receive record posted while its earlier posting is outstanding"
    (design/disjoint-pages-change-request.md:1450); "The bridge (`rdma/mesh-flow.c`) reads them and posts the
    receives." (design/disjoint-pages-change-request.md:1430-1431)
- **D.4 mesh: rdma/mesh-flow.c written** (design/disjoint-pages-change-request.md:1451-1452)
  > Written, the cancellation of a static schedule's receives where the map serial moves (D.3.6).
  - Placed by: "rdma/mesh-flow.c:" (design/disjoint-pages-change-request.md:1450, the bullet this sentence is in);
    "The bridge (`rdma/mesh-flow.c`) compares the serial with the static schedule's and cancels."
    (design/disjoint-pages-change-request.md:1441-1442)

### bridge: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R1 (requirement)** (design/plan-to-bridge-change-request.md:1948-1948)
  > The code must move every page from the page that holds it to the page it lands in, with nothing copied on the way.
  - Placed by: "Produced by the schedule, `mesh_plan`. Consumed by the bridge (rdma/mesh-flow.c), which sends each
    block from the page that holds it to the page it lands in." design/plan-to-bridge-change-request.md:1976-1977
- **R1 data structure** (design/plan-to-bridge-change-request.md:1949-1950, 1974-1975): stated under scheduled value,
  which its clause names with this stage.
- **R1 where it composes** (design/plan-to-bridge-change-request.md:1970-1977): stated under schedule, which its
  clause names with this stage.
- **R4 (requirement)** (design/plan-to-bridge-change-request.md:2177-2177): stated under schedule, which its clause
  names with this stage.
- **R4 behaviour at run time, 3** (design/plan-to-bridge-change-request.md:2189-2193): stated under schedule, which
  its clause names with this stage.
- **R4 behaviour at run time, 4** (design/plan-to-bridge-change-request.md:2194-2194): stated under schedule, which
  its clause names with this stage.
- **R4 algorithm: presence is the supplied word** (design/plan-to-bridge-change-request.md:2213-2216)
  > Here the presence is C3's supplied word: "the producer stores it after writing its pages. One store can release a
  > whole range of pages." Its store follows its pages in C3's writing order (Boehm and Adve, 2008). A producer's
  > store releases whatever pages its work has written, any number at once and in any order. The bridge reads the
  > words in place and sends each released page (R1).
  - Placed by: "The bridge reads the words in place and sends each released page (R1)."
    design/plan-to-bridge-change-request.md:2215-2216; "Consumed by the bridge, which reads the lists and the supplied
    words." design/plan-to-bridge-change-request.md:2254
- **R4 nothing synchronizes outputs one page at a time: landings read by masks**
  (design/plan-to-bridge-change-request.md:2239-2240): stated under lower, which its clause names with this stage.
- **R4 where it composes** (design/plan-to-bridge-change-request.md:2250-2254): stated under schedule, which its
  clause names with this stage.
- **R5 executions** (design/plan-to-bridge-change-request.md:2337-2340): stated under schedule, which its clause names
  with this stage.
- **R5 where it composes** (design/plan-to-bridge-change-request.md:2351-2362): stated under schedule, which its
  clause names with this stage.
- **R5 changes by file: mesh-collective.c and mesh-flow.c** (design/plan-to-bridge-change-request.md:2377-2379):
  stated under schedule, which its clause names with this stage.
- **R6 where it composes** (design/plan-to-bridge-change-request.md:2393-2396): stated under place, which its clause
  names with this stage.
- **R7 (requirement and fact of the code)** (design/plan-to-bridge-change-request.md:2405-2409)
  > The code must keep every transport object and count inside the bridge, and the bridge takes whatever the devices
  > report on each run. **No cited source states a data structure, an algorithm or a count rule for this
  > requirement.** As a fact of the code (rdma/mesh-verbs.h `device_up`, at tag pre-return-20261103 and now), each
  > device's `max_pairs` is its `max_qp` less `MESH_PROVIDER_RESERVED_QPS`.
  - Placed by: "The bridge computes `max_pairs`, a `uint32_t`, for each device on each run, and only the bridge reads
    it." design/plan-to-bridge-change-request.md:2410-2411
- **R7 where it composes** (design/plan-to-bridge-change-request.md:2410-2411)
  > **Where it composes.** The bridge computes `max_pairs`, a `uint32_t`, for each device on each run, and only the
  > bridge reads it. The scheduled value carries no transport object and no queue-pair count.
  - Placed by: "The bridge computes `max_pairs`, a `uint32_t`, for each device on each run, and only the bridge reads
    it. The scheduled value carries no transport object and no queue-pair count."
    design/plan-to-bridge-change-request.md:2410-2411
- **R8 attaching holds nothing; no EADDRINUSE** (design/plan-to-bridge-change-request.md:2422-2428, 2441-2441)
  > - The code must hold nothing for a program but its pages. Attaching holds nothing, and there is no client word and
  > no `EADDRINUSE`. A bridge with no function execution to serve does nothing. > in this istuation is there a real
  > problem solved by this extra allocator? is this a fictitious problem with no > origin, whose error correcting code
  > nevertheless can cause real errors? is this a misplaced solution (there are two > allocation goals which are both
  > solved by the compiler at compile time and applied through a combination of > constraints (at compile time))? a
  > superflujous solution? > why do you think you need special rules about 'numbers of programs at once'? [...] -
  > rdma/mesh-dataflow.c: attaching only maps the region.
  - Placed by: "rdma/mesh-dataflow.c: attaching only maps the region." design/plan-to-bridge-change-request.md:2441
- **R8 there is no client word** (design/plan-to-bridge-change-request.md:2422-2423)
  > Attaching holds nothing, and there is no client word and no `EADDRINUSE`.
  - Placed by: "rdma/mesh-dataflow.c: attaching only maps the region." design/plan-to-bridge-change-request.md:2441
- **R8 a bridge with no function execution does nothing** (design/plan-to-bridge-change-request.md:2423-2423)
  > A bridge with no function execution to serve does nothing.
  - Placed by: "A bridge with no function execution to serve does nothing."
    design/plan-to-bridge-change-request.md:2423
- **R8 function executions ordered by a fixed sequencer** (design/plan-to-bridge-change-request.md:2435-2436)
  > **Function executions are the one program's,** ordered by a fixed sequencer:
  > design/aot-compilation-change-request.md A7 (P. Barham et al., Pathways, MLSys 2022, §4.4; Défago, Schiper and
  > Urbán, 2004).
  - Placed by: "The fixed sequencer runs in the bridge of the island's highest-numbered live node, as Défago et al.
    and Garcia-Molina describe." design/plan-to-bridge-change-request.md:962 (a quoted Claude turn in "The turns", not
    a clause of the proposal; R8 itself names no function or file for the sequencer)
- **R8 Replaces: any rule for several programs** (design/plan-to-bridge-change-request.md:2442-2448, 2454-2454)
  > - **No source states that several programs run at once.** The phrase appears only in Claude's text: this request's
  > R8 table, the disjoint-pages request's C3 item 4, D.3 item 1 and D.5, and deliverables.md's "T7. Multi-tenant
  > fabric by lease". The operator's one turn on it rejects run-time allocation: "is attaching conceptually real or do
  > programs do nothing but what the allocator lets them, with the shapes and coarse bin packing requierments of
  > allocation known aot? do you think we're supporting dynamic runtime allocations or dynamic runtime page->page
  > access patterns outside of the types and reuirements presented so far?" (the turns, above). - **Replaces:** [...]
  > - any rule for several programs.
  - Placed by: the reader.

### bridge: [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

- **7.3, step 2** (design/problems-of-14b9a757-change-request.md:381-382)
  > The port's child process raises the counter by each work completion it retrieves for that range's sends or
  > forwards. This is in rdma-core's sense: the buffer may be reused only once the completion is retrieved.
  - Placed by: "The port's child process writes their values, through each send's `struct prepared_send`."
    design/problems-of-14b9a757-change-request.md:393-395; "the bridge's child process for a port (rdma/mesh-flow.c,
    rdma/mesh-child.c) is the part of the runtime that posts sends and receives."
    design/problems-of-14b9a757-change-request.md:81-82
- **7.3, step 4** (design/problems-of-14b9a757-change-request.md:385-387)
  > When the port's child process finds a persistent request cancelled, it first retrieves the completions of the
  > sends it has posted. Each raises the count, or the error count where it completed in error, as fi_cancel has a
  > completed operation report its own status.
  - Placed by: "The port's child process writes their values, through each send's `struct prepared_send`."
    design/problems-of-14b9a757-change-request.md:393-395; "the bridge's child process for a port (rdma/mesh-flow.c,
    rdma/mesh-child.c) is the part of the runtime that posts sends and receives."
    design/problems-of-14b9a757-change-request.md:81-82
- **7.3, step 5** (design/problems-of-14b9a757-change-request.md:388-389)
  > Every send of that persistent request that the child process has not posted is then completed as cancelled: it
  > raises the error count.
  - Placed by: "The port's child process writes their values, through each send's `struct prepared_send`."
    design/problems-of-14b9a757-change-request.md:393-395; "the bridge's child process for a port (rdma/mesh-flow.c,
    rdma/mesh-child.c) is the part of the runtime that posts sends and receives."
    design/problems-of-14b9a757-change-request.md:81-82
- **7.3, step 7** (design/problems-of-14b9a757-change-request.md:392-392)
  > No buffer is reused before the completion of its posted send is retrieved.
  - Placed by: the reader.
- **9** (design/problems-of-14b9a757-change-request.md:460-472): stated under schedule, which its clause names with
  this stage.
- **11, step 1** (design/problems-of-14b9a757-change-request.md:494-495)
  > The loss of a cable that a persistent request's static link schedule names cancels that persistent request, as a
  > change of membership view does.
  - Placed by: "in the runtime, the bridge writes the removal and the load in the port's child process reads it."
    design/problems-of-14b9a757-change-request.md:502
- **11, step 2** (design/problems-of-14b9a757-change-request.md:496-497)
  > **Removal:** the cancellation removes the persistent request from the structure the load reads, stored before any
  > child process is spawned for the port.
  - Placed by: "in the runtime, the bridge writes the removal and the load in the port's child process reads it."
    design/problems-of-14b9a757-change-request.md:502
- **11, step 3** (design/problems-of-14b9a757-change-request.md:498-498)
  > **The load:** it reads the removal after its acquire and gains no reference to the persistent request.
  - Placed by: "in the runtime, the bridge writes the removal and the load in the port's child process reads it."
    design/problems-of-14b9a757-change-request.md:502
- **11, step 4** (design/problems-of-14b9a757-change-request.md:499-500): stated under schedule, which its clause
  names with this stage.
- **12, step 1** (design/problems-of-14b9a757-change-request.md:526-526)
  > A membership view's identifier is (Lamport time, creator), where the creator is (sequencer node, boot).
  - Placed by: "rdma/mesh-records.c writes the pair where it makes a membership view (`struct mesh_group_view`)."
    design/problems-of-14b9a757-change-request.md:528
- **12, step 2** (design/problems-of-14b9a757-change-request.md:527-527): stated under schedule, which its clause
  names with this stage.
- **13, step 1** (design/problems-of-14b9a757-change-request.md:552-552)
  > The queue pair is moved to the error state (`IB_QPS_ERR`).
  - Placed by: "rdma/mesh-child.c:44–75 (`lanes_flush`) moves each lane with work outstanding to the error state and
    takes the completions." design/problems-of-14b9a757-change-request.md:560-561; "in the runtime, the port's child
    process writes the error count of the send's `struct prepared_send` where it drains (`lanes_flush`)."
    design/problems-of-14b9a757-change-request.md:556-557
- **13, step 2** (design/problems-of-14b9a757-change-request.md:553-553)
  > All its send-queue completions are consumed.
  - Placed by: "rdma/mesh-child.c:44–75 (`lanes_flush`) moves each lane with work outstanding to the error state and
    takes the completions." design/problems-of-14b9a757-change-request.md:560-561; "in the runtime, the port's child
    process writes the error count of the send's `struct prepared_send` where it drains (`lanes_flush`)."
    design/problems-of-14b9a757-change-request.md:556-557
- **13, step 3** (design/problems-of-14b9a757-change-request.md:554-555)
  > Each SEND completion taken raises the error count of the record at its `wr_id`: it is one of fi_cntr(3)'s
  > "operations which have completed in error", and its `wr_id` is valid (ibv_poll_cq(3)).
  - Placed by: "in the runtime, the port's child process writes the error count of the send's `struct prepared_send`
    where it drains (`lanes_flush`). The program's `wait_tensor` reads it (7.3, step 6)."
    design/problems-of-14b9a757-change-request.md:556-558; "step 3 is written in `lanes_flush`."
    design/problems-of-14b9a757-change-request.md:583

### bridge: [torch-capture-change-request.md](torch-capture-change-request.md)

- **2, "It takes / It writes" (the IPC graph is a function of the static schedule)**
  (design/torch-capture-change-request.md:460-464): stated under argument_buffers, which its clause names with this
  stage.
- **4, "It takes / It writes" (the event's value)** (design/torch-capture-change-request.md:634-635): stated under
  Program, which its clause names with this stage.
- **4, Write, first bullet (execution t's waits; no fixed count)** (design/torch-capture-change-request.md:652-655):
  stated under Program, which its clause names with this stage.
- **5, Write (the release's one 64-bit atomic write)** (design/torch-capture-change-request.md:696-697): stated under
  lower, which its clause names with this stage.
- **6, Write, second bullet (mesh_view_compose's remapping deleted)** (design/torch-capture-change-request.md:729-729)
  > `mesh_view_compose`'s remapping of those operands is deleted.
  - Placed by: "**Replaces:** rdma/mesh-dataflow.c `mesh_view_compose` (:187–208 at mesh f55c541c), which reserves an
    address range and remaps operands' page-table rows over it with `mach_vm_remap`."
    (design/torch-capture-change-request.md:711-712)

### bridge: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **1 / It holds** (design/synchronization-signature-change-request.md:93-97): stated under lower, which its clause
  names with this stage.
- **2 / Matrix Market** (design/synchronization-signature-change-request.md:110-114): stated under no stage named,
  which its clause names with this stage.
- **2 / DOT** (design/synchronization-signature-change-request.md:115-117): stated under no stage named, which its
  clause names with this stage.
- **2 / Space-time diagram** (design/synchronization-signature-change-request.md:118-121): stated under no stage
  named, which its clause names with this stage.
- **4 / Write: the D program stops the process** (design/synchronization-signature-change-request.md:138-140): stated
  under no stage named, which its clause names with this stage.
- **4 / Write: every edge the image of a dependence, none redundant**
  (design/synchronization-signature-change-request.md:145-147): stated under graph module, which its clause names with
  this stage.

### bridge: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **1 (Write, bullet 1: diagnose and correct each machine's bridge configuration)**
  (design/integration-finishing-change-request.md:25-27): stated under machines, which its clause names with this
  stage.
- **1 (Write, bullet 3: start every bridge)** (design/integration-finishing-change-request.md:29-29): stated under
  machines, which its clause names with this stage.

### bridge: [page-queue-change-request.md](page-queue-change-request.md)

- **The operator's statements: Relays** (design/page-queue-change-request.md:18-21): stated under schedule, which its
  clause names with this stage.
- **The operator's statements: Queue pairs and links belong to the bridge**
  (design/page-queue-change-request.md:35-47)
  > find previous discussion on this topic in this repo and the basic > misidentification of whether 'queue pairs' or
  > 'streams' or any attempt to expose ortransduce the global object > representing the literal connection between two
  > physical and different-clock-domain literal computers are ever to be > exposed to collective or data processing
  > code Earlier statements found in that search: - 2026-10-19: "it is the job of hte bridge to negotiate stuff like
  > the queue pairs." - 2026-10-23: "why would any agent think they need to expose global api objects like queue pairs
  > to anything outside of the bridge service this time" - 2026-10-23: "clients don't get to compile 'orders'. only
  > pages->pages scatters." - 2026-10-23: "the bridge is the one that knows about queue pairs after all, so it might
  > only possibly be the bridge's responsibility to decide how to swizzle and tile transactions which can use multiple
  > links at a time to reach a target."
  - Placed by: "it is the job of hte bridge to negotiate stuff like the queue pairs."
    design/page-queue-change-request.md:41
- **The operator's statements: The E2B code, the bridge and two layers** (design/page-queue-change-request.md:65-68)
  > do you think you're being told to invent some totally new test, or being told to change the e2b running code so
  > that > it can interact with a bridge connected to some number of peers within a total collective of 5 machines? is
  > the bridge > being totally deleted or is the bridge also required to present a usable interface for collective
  > operations over 5 > total machines and some number of immediately tethered peers?
  - Placed by: "is the bridge > being totally deleted or is the bridge also required to present a usable interface for
    collective operations over 5 > total machines" design/page-queue-change-request.md:66-68; "change the e2b running
    code" design/page-queue-change-request.md:65
- **The operator's statements: The page queue** (design/page-queue-change-request.md:81-88)
  > collectives clearly need to submit work to a runtime. some part of the runtime needs to enqueue and dequeue pages
  > (the > unit of memory used in this repo btw) s.t. they are transmitted to their final destination within a
  > collective as > fast as possible. this means using absolutely all inbound links (and perhaps also using routing
  > across intermediate > nodes as switches in some networking solutions to a scatter operation) which connect a
  > sender and receiver, not using > 'queue pairs'. no code but the bridge can actually be responsible for
  > transmitting data, so it is clear that there > must be some kind of zero copy queue, consumption of the queue by
  > the bridge's functions for interacting with the send > and receive dma interfaces, and that this is very very very
  > boring code, which might have gone unspecified because of > how basic and simple it should have been (and how
  > implicit the requirement for its writing.).
  - Placed by: "no code but the bridge can actually be responsible for transmitting data"
    design/page-queue-change-request.md:85; "consumption of the queue by the bridge's functions"
    design/page-queue-change-request.md:86; "collectives clearly need to submit work to a runtime."
    design/page-queue-change-request.md:81
- **The operator's statements: Replacing the --link form** (design/page-queue-change-request.md:102-108)
  > this can clearly be satisfied in semantics by finishing the implementation of the replacement 'modality' (the one
  > that > isn't '--link'), ensuring it *covers all features and requirements the previous implementation covered*,
  > and only > *then* removing the now-ablatd obsolete code. attempting to front run deletion ebfore impolementation
  > is such an > obvious mistake at this point that it should not be repeated. no 'fallbacks' means that only one sort
  > of > configuration can run within one runtime, with no inversion of control or callback-decided-use of one
  > implementation > or the other, with, e.g., inlining of handlers for both in all callsites forever bc a subagent
  > was told to do > something evil like 'retain the --link form of course'.
  - Placed by: "The bridge, mesh `rdma/mesh-flow.c`, is started for each run with `-I <node>` and one `--link
    device,peer,local-address,remote-address[,service]` per cable." design/page-queue-change-request.md:145-146
- **The operator's statements: The bridge's process lifecycle** (design/page-queue-change-request.md:117-123)
  > "- Left behind: the children, the supervisor, ' why do you think you can get rid of the hcild processes and why do
  > > you think all of the work in the failport here was incorrect? code that doesn't respect sigterms shouldn't run.
  > code > that habitually breaks kernel state and requires full system restarts shouldn't be run or ablated just for
  > fun either. > "What the history actually shows: the failport fixed those failures one at a time, and its final
  > form had the clean > record." then you have to verbatim reuse that process lifecycle code since you have no
  > alternative to present or cite > which sin't odious and a waste of time to test or run.
  - Placed by: "### The bridge's process lifecycle" design/page-queue-change-request.md:115; "**The bridge**
    (`mesh-flow.c` `link_prepare`, `link_send_progress`, `link_receive_progress`, `link_run`, `link_close`):"
    design/page-queue-change-request.md:243 with "The process lifecycle is the code of tag `pre-return-20261103`,
    verbatim (`rdma/mesh-heartbeat.c`;" design/page-queue-change-request.md:246
- **The operator's statements: Failing by error, not by hardware fault** (design/page-queue-change-request.md:127-134)
  > … this is to be addresed by not writing bad code and deleting bad code / revising it s.t. launches fail-by-error >
  > rather than fail-by-hardware-fault while more detailed wrok is ongoing. > some spinwaits are allowed, as a treat,
  > if we know they conclude in bounded time even if someone messes up their > complementary async event originators >
  > "So the bridge main runs today has a machine-crash path." so delete a function header s.t. the code fials on
  > compile > instead of failing at runtime, then do the reading and background i just told you to do.
  - Placed by: "So the bridge main runs today has a machine-crash path." design/page-queue-change-request.md:133; "The
    teardown `down_pair` (mesh `rdma/mesh-verbs.h`) has its header deleted (mesh b89e6362), so `rdma/` does not
    compile." design/page-queue-change-request.md:149-150
- **Claude's proposal: The page lists (the queue): the data structure** (design/page-queue-change-request.md:192-194):
  stated under scheduled value, which its clause names with this stage.
- **Claude's proposal: The page lists: word, position** (design/page-queue-change-request.md:198-200): stated under
  scheduled value, which its clause names with this stage.
- **Claude's proposal: The page lists: Consumed by** (design/page-queue-change-request.md:204-204)
  > **Consumed by** the bridge alone, in `mesh-flow.c`.
  - Placed by: "**Consumed by** the bridge alone, in `mesh-flow.c`." design/page-queue-change-request.md:204
- **Claude's proposal: The page lists: Replaces** (design/page-queue-change-request.md:204-205): stated under
  scheduled value, which its clause names with this stage.
- **Claude's proposal: The assignment of pages to cables and queue pairs: the algorithm**
  (design/page-queue-change-request.md:227-229)
  > **The algorithm** is Graham's list scheduling (R. L. Graham, "Bounds for Certain Multiprocessing Anomalies", The
  > Bell System Technical Journal 45(9), 1966; design/list-scheduling-sources-2026-11-08.md §2) over the cables and
  > queue pairs between two adjacent nodes. Each of the two bridges computes it when the lists are loaded.
  - Placed by: "Each of the two bridges computes it when the lists are loaded."
    design/page-queue-change-request.md:229
- **Claude's proposal: The assignment: It takes** (design/page-queue-change-request.md:231-234)
  > **It takes**, for an ordered pair of adjacent nodes (a → b): - the ordered list of a's sends to b, which the
  > schedule gives b as the same ordered list of its receives from a; - the cables between a and b, from the map, and
  > each cable's bandwidth; - the queue-pair count negotiated at each cable's pairing, which is the same at both ends.
  - Placed by: "Each of the two bridges computes it when the lists are loaded."
    design/page-queue-change-request.md:229; "which the schedule gives b" design/page-queue-change-request.md:232
- **Claude's proposal: The assignment: Its step** (design/page-queue-change-request.md:235-236)
  > **Its step:** each page of the list, in the list's order, is assigned to the (cable, queue pair) whose cumulative
  > bytes over the cable's bandwidth finishes earliest, ties broken by cable id and then queue-pair index.
  - Placed by: "Each of the two bridges computes it when the lists are loaded."
    design/page-queue-change-request.md:229
- **Claude's proposal: The assignment: It writes** (design/page-queue-change-request.md:237-239)
  > **It writes** each queue pair's pages in that order. b posts each queue pair's receives in that order, and a sends
  > in that order as each page's `word` reads its `position`. Both ends compute the assignment from the same inputs,
  > so nothing is negotiated per page.
  - Placed by: "Each of the two bridges computes it when the lists are loaded."
    design/page-queue-change-request.md:229
- **Claude's proposal: The changes, by file: The bridge** (design/page-queue-change-request.md:243-245)
  > **The bridge** (`mesh-flow.c` `link_prepare`, `link_send_progress`, `link_receive_progress`, `link_run`,
  > `link_close`): load the page lists, compute the assignment above, post each queue pair's receives, post each send
  > once its `word` reads its `position`, and store each completed receive's `position` in its `word`.
  - Placed by: "**The bridge** (`mesh-flow.c` `link_prepare`, `link_send_progress`, `link_receive_progress`,
    `link_run`, `link_close`):" design/page-queue-change-request.md:243
- **Claude's proposal: The changes, by file: The bridge: process lifecycle**
  (design/page-queue-change-request.md:246-247)
  > The process lifecycle is the code of tag `pre-return-20261103`, verbatim (`rdma/mesh-heartbeat.c`;
  > `rdma/mesh-link.c` per-port children, their stop, reaping, flush and exit; the holder table).
  - Placed by: "**The bridge** (`mesh-flow.c` `link_prepare`, `link_send_progress`, `link_receive_progress`,
    `link_run`, `link_close`):" design/page-queue-change-request.md:243; "(`rdma/mesh-heartbeat.c`;"
    design/page-queue-change-request.md:246
- **Claude's proposal: The changes, by file: The bridge: queue-pair count**
  (design/page-queue-change-request.md:248-249)
  > A cable's queue-pair count is `max_qp` − `MESH_PROVIDER_RESERVED_QPS`, the smaller of the two ends', as the code
  > of tag `pre-return-20261103` has it (`rdma/mesh-verbs.h` `device_up`, `max_pairs`).
  - Placed by: "**The bridge** (`mesh-flow.c` `link_prepare`, `link_send_progress`, `link_receive_progress`,
    `link_run`, `link_close`):" design/page-queue-change-request.md:243; "(`rdma/mesh-verbs.h` `device_up`,
    `max_pairs`)" design/page-queue-change-request.md:249
- **Claude's proposal: The changes, by file: The callers: tools/mesh/run_pair.py**
  (design/page-queue-change-request.md:258-259): stated under caller, which its clause names with this stage.
- **Claude's proposal: Order, bullet 1** (design/page-queue-change-request.md:265-267)
  > The replacement above is finished until it covers everything the `--link` form and the old calls did. Then the
  > `--link` form (`-I`, `--link`, `oob`, `dial`, `listener_up`, the per-session exchange) and the replaced calls are
  > removed.
  - Placed by: "The bridge, mesh `rdma/mesh-flow.c`, is started for each run with `-I <node>` and one `--link
    device,peer,local-address,remote-address[,service]` per cable." design/page-queue-change-request.md:145-146; the
    replaced calls: "are replaced by that function." design/page-queue-change-request.md:254
- **Claude's proposal: Order, bullet 2** (design/page-queue-change-request.md:268-268)
  > One configuration runs in a runtime. No call site handles both forms.
  - Placed by: "The bridge, mesh `rdma/mesh-flow.c`, is started for each run with `-I <node>` and one `--link
    device,peer,local-address,remote-address[,service]` per cable." design/page-queue-change-request.md:145-146
- **Claude's proposal: Order, bullet 4** (design/page-queue-change-request.md:270-270)
  > Every wait is bounded, and every failure is an error value, never a hardware fault.
  - Placed by: "So the bridge main runs today has a machine-crash path." design/page-queue-change-request.md:133

## caller

What the caller of `compile_module` supplies or does.

### caller: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **A2 produced by, consumed by** (design/aot-compilation-change-request.md:177-181): stated under place, which its
  clause names with this stage.
- **A2 the chooser is separate** (design/aot-compilation-change-request.md:182-182)
  > **The chooser is a separate pass that emits the same operand, and this compilation contains none:**
  - Placed by: "The caller passes the operand." (design/aot-compilation-change-request.md:177); "It contains no
    chooser for either. A chooser that fills those operands is defined later, apart from all of this."
    (design/aot-compilation-change-request.md:13-14)
- **A3 produced by, consumed by (the caller passes the operand; mesh's type)**
  (design/aot-compilation-change-request.md:202-204)
  > **Produced by, consumed by.** The caller passes the operand. A4's routing takes the capacities, and A6's schedule
  > takes w and c. In the reference program the operand is `cost`, a function from an fx node to a float. In mesh the
  > weights are `struct mesh_weight`, a field of `struct mesh_function` (rdma/mesh-collective.h).
  - Placed by: "The caller passes the operand." (design/aot-compilation-change-request.md:202)
- **A3 the chooser of the weights** (design/aot-compilation-change-request.md:205-205)
  > **The chooser of the weights is defined later, apart, as the chooser of A2 is.**
  - Placed by: "The caller passes the operand." (design/aot-compilation-change-request.md:202); "It contains no
    chooser for either. A chooser that fills those operands is defined later, apart from all of this."
    (design/aot-compilation-change-request.md:13-14)
- **A4 which nodes hold a function's static schedule** (design/aot-compilation-change-request.md:264-266): stated
  under schedule, which its clause names with this stage.
- **A7 sender(m) (the bridge's and mesh_plan's part)** (design/aot-compilation-change-request.md:372-374): stated
  under Program, which its clause names with this stage.
- **A7 sender(m) (the launcher passes the program's nodes)** (design/aot-compilation-change-request.md:373-381)
  > The program's nodes are the nodes its launcher started it at, an operand of the `mesh_plan` call (`program`) that
  > the launcher passes as it passes `group` (MESH_GROUP). e2b's launcher, tools/mesh/run_pair.py, passes no node set:
  > it starts rank r at map index r on every node of the collective, so e2b's program's nodes are map indices 0 … R −
  > 1, which tools/e2b_replay.m passes. A torch program's launcher passes them as MESH_PROGRAM beside MESH_GROUP
  > (rdma/mesh_compile.py `_program`). Every run is launched on the whole collective ("you may never use less than the
  > entire collective under any circumstances. all is okay, 2 is not okay. all is cheaper than 2."), so the program's
  > nodes are every member of the view, every relay of A4's routes is one of them, and every node of every static
  > schedule, relays among them, calls `mesh_plan` and holds the static schedule.
  - Placed by: The clause's own "the nodes its launcher started it at, an operand of the `mesh_plan` call (`program`)
    that the launcher passes as it passes `group` (MESH_GROUP)" (design/aot-compilation-change-request.md:373-374)
- **Appendix: a chooser for edge weights** (design/aot-compilation-change-request.md:651-652, 659-664)
  > **The algorithm is the α–β model of a transfer.** A transfer of n bytes over one link takes α + n β, where α is
  > the link's start-up time and β its time per byte. [...] **The data structure it fills is A3's edge weights:** an
  > edge e of n bytes weighs α_k + n β_k on each cable k of its route, where each hop is its own instruction
  > (design/plan-to-bridge-change-request.md R5). - β_k is 1 over cable k's bandwidth, from the map (mesh-map.h). -
  > α_k is fitted from measured transfer times of at least two sizes on that cable, as Hockney fits r∞ and n½. -
  > **Produced by, consumed by:** a function from (bytes, cable) to a weight, passed to the compilation as its
  > edge-weight chooser. Any other chooser can replace it, as A3 requires.
  - Placed by: "**Stage:** it produces the cost operand that the schedule takes (A3)."
    (design/aot-compilation-change-request.md:650); "passed to the compilation as its edge-weight chooser"
    (design/aot-compilation-change-request.md:663-664)
- **Appendix item 5: the algorithm (Fisher and Yates over Threefry; Lemire)**
  (design/aot-compilation-change-request.md:913-926): stated under bridge, which its clause names with this stage.
- **Appendix item 5: every program declares each consumer's kind** (design/aot-compilation-change-request.md:991-994)
  > **Every program declares each consumer's kind.** A consumer that reduces partial sums, or otherwise takes pages in
  > any order, is an asynchronous, incomplete, noncontiguous consumer (R4). A program that declares every consumer
  > synchronous, complete and contiguous avoids demonstrating this requirement. For the program the code runs today,
  > tools/prepare_e2b.py does the declaring:
  - Placed by: The clause's own "**Every program declares each consumer's kind.**"
    (design/aot-compilation-change-request.md:991) and "For the program the code runs today, tools/prepare_e2b.py does
    the declaring:" (design/aot-compilation-change-request.md:993-994)
- **The files and functions: rdma/mesh_compile.py** (design/aot-compilation-change-request.md:1064-1069): stated under
  place, which its clause names with this stage.
- **The files and functions: metal-microbench tools/mesh/run_pair.py**
  (design/aot-compilation-change-request.md:1084-1085)
  > **tools/mesh/run_pair.py:** draws the group and passes it as `MESH_GROUP`; starts rank r at map index r on every
  > node of the collective (A7, item 5).
  - Placed by: "e2b's launcher, tools/mesh/run_pair.py, passes no node set: it starts rank r at map index r on every
    node of the collective" (design/aot-compilation-change-request.md:374-376); "whose group the launcher draws anew
    at each launch (metal-microbench tools/mesh/run_pair.py: `1 + secrets.randbelow(2**64 - 1)`, passed as
    `MESH_GROUP`)" (design/aot-compilation-change-request.md:920-921)
- **The files and functions: metal-microbench tools/e2b_replay.m**
  (design/aot-compilation-change-request.md:1086-1086)
  > **tools/e2b_replay.m:** passes the program's nodes, map indices 0 … R − 1 (A7).
  - Placed by: "The program's nodes are the nodes its launcher started it at, an operand of the `mesh_plan` call
    (`program`) that the launcher passes" (design/aot-compilation-change-request.md:373-374); "so e2b's program's
    nodes are map indices 0 … R − 1, which tools/e2b_replay.m passes" (design/aot-compilation-change-request.md:376)
- **The files and functions: metal-microbench tools/prepare_e2b.py**
  (design/aot-compilation-change-request.md:1087-1087): stated under export, which its clause names with this stage.

### caller: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **3 / Write, first bullet** (design/compilation-composition-change-request.md:153-154): stated under graph module,
  which its clause names with this stage.
- **4 / Write: placement and costs are operands** (design/compilation-composition-change-request.md:170-173)
  > **Composes:** placement takes the placement operand and the schedule takes the cost operand; the schedule produces
  > the scheduled value; the runtime and the bridge read it. **Write:** placement and costs arrive as operands;
  > compilation realizes the graph's storage, transfers, ordering and synchronization.
  - Placed by: "placement takes the placement operand and the schedule takes the cost operand"
    design/compilation-composition-change-request.md:170; "def compile_module(module, args, processors, placement,
    cost=unit_cost, **export_options):" design/data-flow.md:115
- **4 / Write: performance and numerical comparison** (design/compilation-composition-change-request.md:177-178)
  > Performance and numerical comparison follow the operator's requirements in metal-microbench `docs/measurement.md`.
  - Placed by: the reader.
- **Performance correction / Performance obligation** (design/compilation-composition-change-request.md:367-374):
  stated under lower, which its clause names with this stage.
- **Complete replacement / production E2B link** (design/compilation-composition-change-request.md:408-419)
  > - **Replaces:** metal-microbench's production E2B linkage to the wheel's `liblitert-lm`, its generated-source
  > interception as an optimization interface, the source transformations listed below, and their alternative
  > preparation paths. The replacement is finished all in one commit workgroup: the compiler interface is finished and
  > the downstream bodies are removed together, by other code with the same feature coverage and the same performance
  > or better (the operator's turn above). **Write the connected compiler interface.** Build from source the LiteRT
  > runtime, delegate, ML Drift and Dawn/Tint dependencies that actually execute these lowerings. Track the adapter,
  > build description and upstream changes in canonical source, not hand edits under `.build`. The production E2B link
  > uses those built libraries. An executable that still loads a prebuilt GPU library leaves the assignment
  > unfinished.
  - Placed by: "with the driver `tools/e2b_replay.m` that links them" design/the-compiler.md:103-104
- **Complete replacement / replaced-implementation table row 6**
  (design/compilation-composition-change-request.md:456-456): stated under lower, which its clause names with this
  stage.
- **7 / Replaces: run-time readers of positions, tasks, guards; static discovery**
  (design/compilation-composition-change-request.md:508-511): stated under lower, which its clause names with this
  stage.

### caller: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **C3.5** (design/disjoint-pages-change-request.md:871-876)
  > **The data structure** is the dependence graph A3's allocator takes: each stage's reads and writes, in the order
  > its command buffer runs them, from the captured graph's stages. It has no algorithm of its own. **Written by, read
  > by.** `tools/prepare_e2b.py` writes the dependences into the manifest. `e2b_replay` passes them to `mesh_plan` in
  > place of `NULL, 0` (e2b_replay.m:578–579). A3's allocator reads them as its order. The run's pages then don't grow
  > with its executions. Each dependence is a `struct mesh_dependence` `{from, to, lag, kinds}`
  > (`rdma/mesh-collective.h`). No source this document cites names the manifest's type.
  - Placed by: "**C3.5 The normal program passes its dependences.**" (design/disjoint-pages-change-request.md:870);
    "`tools/prepare_e2b.py` writes the dependences into the manifest. `e2b_replay` passes them to `mesh_plan` in place
    of `NULL, 0` (e2b_replay.m:578–579)." (design/disjoint-pages-change-request.md:873-874); "`tools/e2b_replay.m`
    passes the `struct mesh_function` array to `mesh_plan`" (design/disjoint-pages-change-request.md:398-399, under
    the document's heading "- **Runtime.**" at :397)
- **C4 metal-microbench deleted: tools/e2b_replay.m** (design/disjoint-pages-change-request.md:935-937): stated under
  Program, which its clause names with this stage.
- **C4 metal-microbench written: tools/prepare_e2b.py** (design/disjoint-pages-change-request.md:940-940)
  > tools/prepare_e2b.py: each stage's reads and writes, into the manifest (C3.5).
  - Placed by: "**C3.5 The normal program passes its dependences.**" (design/disjoint-pages-change-request.md:870);
    "`tools/prepare_e2b.py` writes the dependences into the manifest." (design/disjoint-pages-change-request.md:873)
- **C4 metal-microbench written: tools/e2b_replay.m** (design/disjoint-pages-change-request.md:941-941)
  > tools/e2b_replay.m: the manifest's dependences, passed to `mesh_plan` (C3.5).
  - Placed by: "`e2b_replay` passes them to `mesh_plan` in place of `NULL, 0` (e2b_replay.m:578–579)."
    (design/disjoint-pages-change-request.md:873-874); "`tools/e2b_replay.m` passes the `struct mesh_function` array
    to `mesh_plan`" (design/disjoint-pages-change-request.md:398-399, under the document's heading "- **Runtime.**" at
    :397)
- **D.3.3** (design/disjoint-pages-change-request.md:1398-1405): stated under schedule, which its clause names with
  this stage.

### caller: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R2 any value a transfer names is sent from its own pages** (design/plan-to-bridge-change-request.md:2097-2103):
  stated under schedule, which its clause names with this stage.
- **R3 the task graph: V and w** (design/plan-to-bridge-change-request.md:2137-2137, 2141-2141, 2143-2144): stated
  under schedule, which its clause names with this stage.
- **R3 the task graph: E are pages, with weight c** (design/plan-to-bridge-change-request.md:2142-2144): stated under
  schedule, which its clause names with this stage.
- **R3 where it composes** (design/plan-to-bridge-change-request.md:2147-2155): stated under schedule, which its
  clause names with this stage.
- **R4 the record of a run** (design/plan-to-bridge-change-request.md:2198-2199)
  > No quoted source says how a run is shown to have done 1 to 3: which record of a run gives, for each function, when
  > each page of its operand became present and when the function started, and each cable's sequence of sends.
  - Placed by: the reader.
- **R4 data structure** (design/plan-to-bridge-change-request.md:2200-2204): stated under schedule, which its clause
  names with this stage.
- **R5 where it composes** (design/plan-to-bridge-change-request.md:2351-2362): stated under schedule, which its
  clause names with this stage.
- **R8 where it composes: placement and weights are operands** (design/plan-to-bridge-change-request.md:2438-2439)
  > Placement and weights are operands, passed in by the program from whatever function chooses them (aot-compilation
  > A2, A3). Placement takes the placement operand, and the schedule takes the weighting operand (R3, R5).
  - Placed by: "Placement and weights are operands, passed in by the program from whatever function chooses them
    (aot-compilation A2, A3). Placement takes the placement operand, and the schedule takes the weighting operand (R3,
    R5)." design/plan-to-bridge-change-request.md:2438-2439

### caller: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **4 / Write: the digest and the review record** (design/synchronization-signature-change-request.md:143-144): stated
  under documents, which its clause names with this stage.
- **5 / Write: replay with the monitor** (design/synchronization-signature-change-request.md:153-153)
  > Replay a captured call graph through its compiled program with the monitor of items 3 and 4 attached.
  - Placed by: the reader.

### caller: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **5 / Write: the cited functions are the CPU's** (design/torch-source-lowering-change-request.md:310-313): stated
  under no stage named, which its clause names with this stage.

### caller: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **1 (Write, bullet 4: e2b on all five machines to its end)** (design/integration-finishing-change-request.md:30-33)
  > Run e2b on all five machines to its end: every rank's prefill and decode end, and the run's record states each
  > rank's decoded tokens. Where a rank stops, the code or configuration that stops it is diagnosed and corrected (the
  > operator, above: "finish the integratino work here"; "it's the job of hte agent to diagnose and correct the
  > missing or wrong configuration").
  - Placed by: "Run e2b on all five machines to its end" design/integration-finishing-change-request.md:30
- **3 (The data structure)** (design/integration-finishing-change-request.md:50-52)
  > **The data structure** is the weighting operand, as design/aot-compilation-change-request.md A3 gives it: "a
  > weight w on each task and a weight c on each edge", and the cables' capacities, from the map. The task weights are
  > the lengths of `mesh_plan`'s tasks; no cited source states that.
  - Placed by: "**Produced by** the caller of the compilation. **Consumed by** the compilation's router and its
    scheduler, which compute no weight (A3)." design/integration-finishing-change-request.md:53-54
- **3 (Produced by)** (design/integration-finishing-change-request.md:53-53)
  > **Produced by** the caller of the compilation.
  - Placed by: "**Produced by** the caller of the compilation." design/integration-finishing-change-request.md:53
- **3 (Replaces)** (design/integration-finishing-change-request.md:55-55)
  > **Replaces:** the weighting operand of one `mesh.Weight` per transfer.
  - Placed by: "**The data structure** is the weighting operand" design/integration-finishing-change-request.md:50;
    "**Produced by** the caller of the compilation." design/integration-finishing-change-request.md:53

### caller: [page-queue-change-request.md](page-queue-change-request.md)

- **The operator's statements: The E2B code, the bridge and two layers** (design/page-queue-change-request.md:65-68):
  stated under bridge, which its clause names with this stage.
- **The operator's statements: On how parity is shown** (design/page-queue-change-request.md:111-113)
  > "1. Inventory before anything else, kept in the repo. It lists everything the --link bridge does, " — No. - "2.
  > Implement, then check parity, then delete, in that order and as separate steps. " — No. - "3. Parity is shown by
  > the normal program, " — yes.
  - Placed by: "Parity is shown by the normal program, the E2B decode on the collective, in an end-to-end run."
    design/page-queue-change-request.md:269
- **Claude's proposal: The schedule: every live node runs a rank** (design/page-queue-change-request.md:222-223)
  > Every live node of the Thunderbolt-5-connected collective runs a rank, so an intermediate node's entries are
  > written on that node. No node outside the collective takes part.
  - Placed by: "**The callers** (metal-microbench):" design/page-queue-change-request.md:255 with "the driver starts
    no bridge, chooses no link, and runs one rank on every node of the collective;"
    design/page-queue-change-request.md:259; the clause stands under "### The schedule that writes the page lists"
    design/page-queue-change-request.md:207
- **Claude's proposal: The changes, by file: The callers: tools/e2b_replay.m**
  (design/page-queue-change-request.md:256-257)
  > `tools/e2b_replay.m` lines 497-500 pass the program's declared receives and sends (operands.json's names, ranks
  > and ranges) to that function instead of binding transfers;
  - Placed by: "**The callers** (metal-microbench):" design/page-queue-change-request.md:255
- **Claude's proposal: The changes, by file: The callers: tools/mesh/run_pair.py**
  (design/page-queue-change-request.md:258-259)
  > `tools/mesh/run_pair.py`'s `discover_link`, `links`, `start_bridge0`, `start_bridge1`, `bridges_up` and
  > `bridges_down` go: the driver starts no bridge, chooses no link, and runs one rank on every node of the
  > collective;
  - Placed by: "**The callers** (metal-microbench):" design/page-queue-change-request.md:255
- **Claude's proposal: The changes, by file: The callers: tools/mesh/grid.py**
  (design/page-queue-change-request.md:260-261)
  > `tools/mesh/grid.py` prepares the programs for the collective's rank count with an explicit placement operand
  > (shares per rank).
  - Placed by: "**The callers** (metal-microbench):" design/page-queue-change-request.md:255; "with an explicit
    placement operand > (shares per rank)" is in the file as "with an explicit placement operand" / "(shares per
    rank)." design/page-queue-change-request.md:260-261
- **Claude's proposal: Order, bullet 3** (design/page-queue-change-request.md:269-269)
  > Parity is shown by the normal program, the E2B decode on the collective, in an end-to-end run.
  - Placed by: "Parity is shown by the normal program, the E2B decode on the collective, in an end-to-end run."
    design/page-queue-change-request.md:269

## machines

The condition of machines, services and configuration.

### machines: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R5 changes by file: rdma/Makefile builds HiGHS** (design/plan-to-bridge-change-request.md:2366-2374): stated under
  no stage named, which its clause names with this stage.

### machines: [integration-finishing-change-request.md](integration-finishing-change-request.md)

- **1 (Write, bullet 1: diagnose and correct each machine's bridge configuration)**
  (design/integration-finishing-change-request.md:25-27)
  > Diagnose each machine's bridge configuration against the configuration the repository installs (mesh bootstrap.sh,
  > install.sh, bin/mesh-bridge.sh and the files they write): the `io.mesh.bridge` job, `bridge.conf`'s region, and
  > the block size. Correct each one so that all five machines agree.
  - Placed by: "### 1. Every mesh machine's bridge runs" design/integration-finishing-change-request.md:20; "Diagnose
    each machine's bridge configuration against the configuration the repository installs"
    design/integration-finishing-change-request.md:25
- **1 (Write, bullet 2: the mini's configuration.plist)** (design/integration-finishing-change-request.md:28-28)
  > The mini's `configuration.plist` is corrected the same way.
  - Placed by: "The mini's `configuration.plist`" design/integration-finishing-change-request.md:28
- **1 (Write, bullet 3: start every bridge)** (design/integration-finishing-change-request.md:29-29)
  > Start every bridge with `bin/mesh-bridge.sh`.
  - Placed by: "### 1. Every mesh machine's bridge runs" design/integration-finishing-change-request.md:20; "Start
    every bridge with `bin/mesh-bridge.sh`." design/integration-finishing-change-request.md:29
- **1 (Write, bullet 4: e2b on all five machines to its end)** (design/integration-finishing-change-request.md:30-33):
  stated under caller, which its clause names with this stage.

## documents

How documents and records are written.

### documents: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **How this document is written and used (silence rule)** (design/aot-compilation-change-request.md:54-55)
  > Where the document is silent, the implementer takes the minimal decision that follows from the document and the
  > code, and records it as its own choice, with its reason, apart from the document's requirements.
  - Placed by: "## How this document is written and used" (design/aot-compilation-change-request.md:29)

### documents: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **Withdrawn, 2026-11-09: clauses on control flow that depends on data ("Nested communication regions: producer and
  consumer handoff", loop stack, predicate drains)** (design/compilation-composition-change-request.md:554-554)
  > strike out anything and everything which contradicts the remarks reaffirmed (yet again) that we still aren't
  > creating fake problems for ourselves wrt 'what if there's an arbitrary python program pretended to be part of a
  > torch program and we need to play geometry dash for 12 hours inside of a graph for a larped reason gosh what are
  > we gonna do now'.
  - Placed by: "## Withdrawn, 2026-11-09" design/compilation-composition-change-request.md:542; "strike out anything
    and everything which contradicts the remarks reaffirmed" design/compilation-composition-change-request.md:554

### documents: [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

- **D.4 mesh: coordination document** (design/disjoint-pages-change-request.md:1453-1454)
  > design/disjoint-pages-revision-coordination.md, "Open at the end": the item on the provider's copy of a request,
  > deleted.
  - Placed by: "design/disjoint-pages-revision-coordination.md, "Open at the end": the item on the provider's copy of
    a request, deleted." (design/disjoint-pages-change-request.md:1453-1454)

### documents: [torch-capture-change-request.md](torch-capture-change-request.md)

- **1, Write, second bullet (a call Metal 4 declares no call for)** (design/torch-capture-change-request.md:414-415):
  stated under no stage named, which its clause names with this stage.

### documents: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **4 / Write: the digest and the review record** (design/synchronization-signature-change-request.md:143-144)
  > the digest changes for the same captured program, and each measured run's digest is appended to the review record
  > (metal-microbench output_data/change-request-reviews.jsonl);
  - Placed by: the reader.

## no stage named

The clause places itself in a flow that is none of the stages above; the words that say so follow each.

### no stage named: [aot-compilation-change-request.md](aot-compilation-change-request.md)

- **A10 algorithms: independence, commuting operations, the bound** (design/aot-compilation-change-request.md:607-617)
  > **Independence:** two calls may run in either order, or at once, where neither writes what the other reads or
  > writes. These are Bernstein's conditions (A. J. Bernstein, Analysis of programs for parallel processing, IEEE
  > Trans. Electronic Computers EC-15(5), 1966). - **Commuting operations:** calls whose effects commute, such as
  > reductions into one accumulator by an associative and commutative operation, may run in any order (M. C. Rinard
  > and P. C. Diniz, Commutativity analysis: a new analysis technique for parallelizing compilers, ACM TOPLAS 19(6),
  > 1997). Floating-point non-associativity is set aside, as the operator does. - **The bound.** The traced block is
  > finite, so each function has finitely many calls in it. Pathways sends a traced block whole: "the program sends a
  > single message (describing the entire subgraph) to the scheduler" (§4.5). So the most calls of a function
  > outstanding at once is the number of its calls in the block, or fewer where Bernstein's conditions order them.
  - Placed by: An analysis between export and the schedule: "**Stage:** export produces the calls' graph. The analysis
    below takes it. The schedule and the runtime take the analysis's output, carried in A8's IR."
    (design/aot-compilation-change-request.md:591-592); "| A10 | export (the calls' graph), then an analysis whose
    output the schedule and the runtime take |" (design/aot-compilation-change-request.md:111)
- **Appendix item 3 (a slow node, "desynchronization": no handler)**
  (design/aot-compilation-change-request.md:872-873)
  > No such event has been observed or recorded, and no requirement asks for a handler of one. - **The change:** the
  > code holds no handler for it.
  - Placed by: the reader.
- **Appendix item 8 (the heartbeat's holders)** (design/aot-compilation-change-request.md:1054-1054)
  > The operator: "settled."
  - Placed by: the reader.

### no stage named: [compilation-composition-change-request.md](compilation-composition-change-request.md)

- **2 / Write: each row's work completed with its deletion**
  (design/compilation-composition-change-request.md:140-140)
  > Each row's work is completed in this composition, including its specified deletion or replacement.
  - Placed by: The clause places itself in every row of the table of the nine requests, not in one stage: "**Applies
    to:** all nine rows of the table." design/compilation-composition-change-request.md:127; each row's stages are the
    table's third column (design/compilation-composition-change-request.md:95-103), of which the document says "The
    third column is read from the stages of `~/schedule-emit` against each clause's title. No request states it."
    design/compilation-composition-change-request.md:91

### no stage named: [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

- **R5 changes by file: rdma/Makefile builds HiGHS** (design/plan-to-bridge-change-request.md:2366-2374)
  > rdma/Makefile builds HiGHS from its source at a pinned release and links it statically. The release is tag
  > v1.15.1, commit 04024d701f79feb8e2f18bc3df0dffc04ef05088, fetched as
  > https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz and checked against SHA-256
  > a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16. metal-microbench's Makefile fetches its pinned
  > sources the same way (`E2B_REFERENCE_REVISION` with `E2B_REFERENCE_SHA256`, and Dawn's tag). It is built with cc
  > and c++. cmake is not on the nodes (m5ult2, 2026-11-05). It takes the same `-ffp-contract=off` as mesh-graph.c and
  > no fast-math. The C API is highs/interfaces/highs_c_api.h of that source. Nothing is installed by hand, and
  > nothing comes from Homebrew.
  - Placed by: The build, rdma/Makefile: "rdma/Makefile builds HiGHS from its source at a pinned release and links it
    statically." design/plan-to-bridge-change-request.md:2366

### no stage named: [torch-capture-change-request.md](torch-capture-change-request.md)

- **1, Write, first bullet (every Metal call recorded and replayed in order)**
  (design/torch-capture-change-request.md:411-413)
  > Every Metal call of the captured program is recorded, and the replay issues the recorded calls in the order they
  > were recorded (apitrace; GFXReconstruct; RenderDoc's "every API call is serialised out in order"), each by the
  > call that MTL4ComputeCommandEncoder.h or MTL4CommandQueue.h declares for it.
  - Placed by: "This item is placed only in a flow in which the program is executed and recorded before its static
    schedule exists." (design/torch-capture-change-request.md:374)
- **1, Write, second bullet (a call Metal 4 declares no call for)** (design/torch-capture-change-request.md:414-415)
  > A call for which Metal 4 declares none invalidates the capture (CUDA §4.2.2.1.2.3), and is documented in this
  > request (GFXReconstruct :89).
  - Placed by: "This item is placed only in a flow in which the program is executed and recorded before its static
    schedule exists." (design/torch-capture-change-request.md:374)
- **1, Write, third bullet (pipelines from a copy of the descriptor with supportIndirectCommandBuffers)**
  (design/torch-capture-change-request.md:416-418)
  > Each compute-pipeline creation method of MTLDevice.h (:834–864) creates its pipeline from a copy of the program's
  > descriptor, with `supportIndirectCommandBuffers` set (GFXReconstruct :102; vulkan_capture_manager.cpp:889–896;
  > MTLComputePipeline.h:70–73).
  - Placed by: "This item is placed only in a flow in which the program is executed and recorded before its static
    schedule exists." (design/torch-capture-change-request.md:374)
- **1, "A call the program makes only while a condition holds", Write, first bullet**
  (design/torch-capture-change-request.md:443-445)
  > The replay of every execution issues such a call where it was recorded: every call is serialised, and the replay
  > executes the serialised calls verbatim (RenderDoc :21, :30; CUDA's snapshot). Dawn's clear is a recorded fill,
  > replayed at every execution; its condition is not evaluated at replay.
  - Placed by: "This item is placed only in a flow in which the program is executed and recorded before its static
    schedule exists." (design/torch-capture-change-request.md:374)
- **1, "A call the program makes only while a condition holds", Write, second bullet (initial contents)**
  (design/torch-capture-change-request.md:446-449)
  > The contents at the capture's start of each buffer the captured calls read before writing are saved with the
  > capture (RenderDoc :21; GFXReconstruct's snapshot). A buffer the captured calls also write has them applied before
  > each execution's replay, so that the execution's writes are undone (RenderDoc :38; `ApplyInitialContents`;
  > vulkan_replay_frame_loop_consumer.h:182–183).
  - Placed by: "This item is placed only in a flow in which the program is executed and recorded before its static
    schedule exists." (design/torch-capture-change-request.md:374)
- **3, LiteRT scope, Write, first bullet (index space analysis)** (design/torch-capture-change-request.md:603-606)
  > A recorded kernel's source is the `source` of the recorded `newLibraryWithSource:options:error:` call (item 1
  > records every call). The index function of each of its loads and stores is computed from that source by use-def
  > chains; a kernel whose index functions are affine, a · (W0 · w0 + l0), is a contiguous kernel, and any other is
  > discontiguous (Lee §3.2.2).
  - Placed by: "Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which
    the program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:544-545)
- **3, LiteRT scope, Write, second bullet (the pages a threadgroup reads)**
  (design/torch-capture-change-request.md:607-609)
  > The pages a threadgroup of a contiguous kernel reads are the partitions its index functions' values over its
  > threads fall in ("Partitions are contiguous in memory"). A threadgroup of a discontiguous kernel reads every
  > partition, as Lee transfers "the entire input" for one.
  - Placed by: "Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which
    the program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:544-545)
- **3, LiteRT scope, Write, third bullet (first dispatch predicated by the mask)**
  (design/torch-capture-change-request.md:610-614)
  > Each recorded dispatch that reads a received operand is first executed predicated by the mask (Russell; RISC-V V).
  > Each threadgroup tests the presence bits of the pages it reads (I-structures' "presence bits"), and no other. A
  > threadgroup any of whose presence bits shows its page absent returns before any store and writes 1 to its element
  > of the mask. It passes or returns as a unit, its threads' results joined by `simd_ballot` (MSL §6.10.1 and
  > §6.10.2).
  - Placed by: "Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which
    the program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:544-545)
- **3, LiteRT scope, Write, fourth bullet (stream compaction into the worklist)**
  (design/torch-capture-change-request.md:615-618)
  > A stream compaction after that dispatch packs the indices of the threadgroups whose mask element is 1 into a dense
  > list by a scan and a scatter, and writes the list's length as the threadgroup count of the dispatch that executes
  > the list (Harris, Sengupta and Owens; Blelloch). The list is the worklist of a data-driven implementation (Nasre,
  > Burtscher and Pingali).
  - Placed by: "Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which
    the program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:544-545)
- **3, LiteRT scope, Write, fifth bullet (second dispatch executes the worklist)**
  (design/torch-capture-change-request.md:619-620)
  > A second dispatch of the kernel executes the worklist's threadgroups, its threadgroup count read from an indirect
  > buffer (MTL4ComputeCommandEncoder.h:90–112). Each of its threadgroups tests its presence bits again.
  - Placed by: "Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which
    the program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:544-545)
- **3, LiteRT scope, Write, sixth bullet (task list in order of completion times)**
  (design/torch-capture-change-request.md:621-623)
  > The dispatches that execute worklists are a task list (Graham) in the order of non-decreasing completion times of
  > the transfers they read, the times `mesh_plan`'s linear program computes, "where ties are broken by choosing an
  > order that is consistent with the precedence relation" (Hall et al.).
  - Placed by: "Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which
    the program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:544-545)
- **7, Write, first bullet (a captured buffer's interval)** (design/torch-capture-change-request.md:788-788)
  > A captured buffer's interval runs from its first write to its last read.
  - Placed by: "The positions are those of the capture's commands, so this item is placed only in a flow in which the
    program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:771-772)
- **7, Write, second bullet (a call's inputs and results)** (design/torch-capture-change-request.md:789-792)
  > A call's input's interval starts at the capture's first command, as a placeholder's `start = 0` (ExecuTorch), and
  > ends at its last read. A call's result's interval starts at its first write and ends at the capture's last
  > command, the output node being the graph's last (torch.fx) and a graph output "never ... scheduled for
  > deallocation" (TensorFlow Lite).
  - Placed by: "The positions are those of the capture's commands, so this item is placed only in a flow in which the
    program is executed and recorded before its static schedule exists."
    (design/torch-capture-change-request.md:771-772)
- **9. Every node replays** (design/torch-capture-change-request.md:841-842)
  > every node runs its captured program by replay, its collectives included. fcf5d18a's calls of torch from Python at
  > every execution are deleted.
  - Placed by: "This item is placed only in a flow in which each node's program is executed and recorded before its
    static schedule exists." (design/torch-capture-change-request.md:834-835)

### no stage named: [unequal-shards-change-request.md](unequal-shards-change-request.md)

- **6 - Data structure** (design/unequal-shards-change-request.md:238-240)
  > HAP's sharding ratios B, one B_j for each of the m devices, one B for each segment of the model (HAP §5.2); and
  > HAP §3.2's stages of the program, each with its communication time comm(i)(B) and each device's computation time
  > comp(i)_j(B_j).
  - Placed by: "the solve runs before `place`. It takes the program's HAP stages and their times"
    design/unequal-shards-change-request.md:246; "Items 6, 7 and 14 produce it before `place` runs."
    design/unequal-shards-change-request.md:55 -- the flow is a solve that runs before `place` and writes the
    placement operand; the document names no function of the data flow for it
- **6 - Algorithm** (design/unequal-shards-change-request.md:241-245)
  > HAP's linear program (3). The B_j minimise, summed over the segment's stages, the stage's communication time, a
  > linear function of the largest B_j, plus the largest over devices of device j's computation time at B_j; subject
  > to the B_j being nonnegative and summing to 1. It is solved as a linear program. The whole sizes follow by item
  > 1's rounding. A tuple is chosen for each segment of the model, not one for the whole mesh dimension.
  - Placed by: "the solve runs before `place`." design/unequal-shards-change-request.md:246; "Items 6, 7 and 14
    produce it before `place` runs." design/unequal-shards-change-request.md:55 -- the flow is a solve that runs
    before `place` and writes the placement operand; the document names no function of the data flow for it
- **6 - Stage** (design/unequal-shards-change-request.md:246-248)
  > the solve runs before `place`. It takes the program's HAP stages and their times and writes, for each segment, the
  > tuple of item 1 in the placement operand, which `place` consumes. It solves "for a fixed distributed program Q"
  > (HAP §5).
  - Placed by: "the solve runs before `place`." design/unequal-shards-change-request.md:246; "writes, for each
    segment, the tuple of item 1 in the placement operand, which `place` consumes"
    design/unequal-shards-change-request.md:246-247 -- the flow is a solve that runs before `place` and writes the
    placement operand; the document names no function of the data flow for it
- **7 - Data structure** (design/unequal-shards-change-request.md:265-267)
  > the functional performance model of Lastovetsky et al.: for each node and each kind of function, a speed s(x) = x
  > / t(x), t(x) the measured time of that function at x units. It is not one number per node.
  - Placed by: "the speeds are taken before `place`. Item 6's solve consumes them as the speeds its sharding ratios
    are chosen with." design/unequal-shards-change-request.md:269-270 -- the flow is a measurement taken before
    `place` and consumed by item 6's solve; the document names no function of the data flow for it
- **7 - Algorithm** (design/unequal-shards-change-request.md:268-268)
  > the measurement of the paper's steps 1 and 2 as quoted: the speed is first taken at x = n / p.
  - Placed by: "the speeds are taken before `place`." design/unequal-shards-change-request.md:269 -- the flow is a
    measurement taken before `place` and consumed by item 6's solve; the document names no function of the data flow
    for it
- **7 - Stage** (design/unequal-shards-change-request.md:269-270)
  > the speeds are taken before `place`. Item 6's solve consumes them as the speeds its sharding ratios are chosen
  > with.
  - Placed by: "the speeds are taken before `place`. Item 6's solve consumes them"
    design/unequal-shards-change-request.md:269 -- the flow is a measurement taken before `place` and consumed by item
    6's solve (itself before `place`); the document names no function of the data flow for either
- **11 - Data structure** (design/unequal-shards-change-request.md:342-345)
  > `tf.RaggedTensor`'s `values` and `row_splits`, which are Arrow's child array and offsets buffer. A tensor with
  > rows of differing length is two tensors: the values of all rows concatenated, and nrows+1 nondecreasing offsets,
  > row i being `values[offsets[i]:offsets[i+1]]`. A second ragged dimension is a second array of offsets over the
  > first. No row is padded.
  - Placed by: "item 13 consumes the offsets. No quoted source states an algorithm or which stage produces the two
    tensors." design/unequal-shards-change-request.md:346-347; "Item 11 states the form of a value."
    design/unequal-shards-change-request.md:56 -- the document states no producing stage; the consumer it names is
    item 13's partitioning
- **11 - Stage** (design/unequal-shards-change-request.md:346-347)
  > item 13 consumes the offsets. No quoted source states an algorithm or which stage produces the two tensors.
  - Placed by: "item 13 consumes the offsets. No quoted source states an algorithm or which stage produces the two
    tensors." design/unequal-shards-change-request.md:346-347 -- the document states no producing stage; the consumer
    it names is item 13's partitioning
- **12 - Algorithm (the all-gather's method)** (design/unequal-shards-change-request.md:379-385)
  > HAP §2.5.1 names two methods for an all-gather of unevenly partitioned tensors: padding the shards to the same
  > size, and broadcasting each shard separately. Which of them an all-gather uses is chosen by their times at the
  > sharding ratios, comm(i)(B) being "determined based on the collective operation type, the sharding ratio B" and
  > the profiled network, and is chosen again whenever the sharding ratios are solved again ("when we optimize the
  > sharding ratios, the communication methods should be updated accordingly"). Item 6's program solves the ratios
  > "for a fixed distributed program Q", so the choice is not a result of that program. No quoted source states the
  > choice for another collective of item 2, or between a padded buffer and a subarray.
  - Placed by: "No quoted source states which function chooses the method, or the type of the value that records the
    choice." design/unequal-shards-change-request.md:387-388 -- the flow is the choice of an all-gather's method by
    its times at the sharding ratios; the document states that no function is named for it
- **13 - Algorithm** (design/unequal-shards-change-request.md:444-450)
  > chain-on-chain partitioning for processors of different speeds (Pınar, Tabak and Aykanat 2008, §2 to §4). A row's
  > weight is its length; device p's load is the total weight of its rows divided by its speed; the separators
  > minimise the largest load. They are found by a probe and a search: the probe, given a candidate largest load,
  > gives each device in order the longest run of rows whose weight does not exceed the candidate times its speed, by
  > binary search in the weights' prefix sums; the search is Nicol's, over the devices in order, keeping the smallest
  > candidate that succeeded and the largest that failed. The devices' order is given; it is not searched. The
  > weights' prefix sums are item 11's offsets.
  - Placed by: "the partitioning takes item 11's offsets and the devices' speeds in their order and writes the
    separators. They are the sharding of the row dimension in the placement operand, which `place` consumes."
    design/unequal-shards-change-request.md:451-452 -- the flow is a partitioning that writes the placement operand's
    sharding, which `place` then consumes; the document names no function of the data flow for the partitioning
- **13 - Stage** (design/unequal-shards-change-request.md:451-452)
  > the partitioning takes item 11's offsets and the devices' speeds in their order and writes the separators. They
  > are the sharding of the row dimension in the placement operand, which `place` consumes.
  - Placed by: "the partitioning takes item 11's offsets and the devices' speeds in their order and writes the
    separators. They are the sharding of the row dimension in the placement operand, which `place` consumes."
    design/unequal-shards-change-request.md:451-452 -- the flow is a partitioning that writes the placement operand's
    sharding; the document names no function of the data flow for the partitioning, and names `place` as the consumer
- **14 - Data structure** (design/unequal-shards-change-request.md:467-468)
  > an upper bound u_j for each device j on its sharding ratio B_j: what the device can hold, as a fraction of the
  > dimension.
  - Placed by: "both run before `place` and write the tuple of item 1 in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:473; "Items 6, 7 and 14 produce it before `place` runs."
    design/unequal-shards-change-request.md:55 -- the flow is item 6's solve, or Whale's Algorithm 1, run before
    `place`; the document names no function of the data flow for either
- **14 - Algorithm (the bound in the linear program)** (design/unequal-shards-change-request.md:469-469)
  > in item 6's linear program the bound is the variable's upper bound, B_j ≤ u_j.
  - Placed by: "both run before `place` and write the tuple of item 1 in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:473 -- the flow is item 6's solve, run before `place`; the
    document names no function of the data flow for it
- **14 - Algorithm (Whale's Algorithm 1)** (design/unequal-shards-change-request.md:469-472)
  > Where the tuple of item 1 is given and not solved for, Whale's Algorithm 1: a device whose memory use at its part
  > of the dimension exceeds 1 gives up load to the device with free memory and the lowest computation load,
  > repeatedly, until none exceeds 1.
  - Placed by: "both run before `place` and write the tuple of item 1 in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:473 -- the flow is Whale's Algorithm 1 run before `place` on a
    given tuple; the document names no function of the data flow for it
- **14 - Stage** (design/unequal-shards-change-request.md:473-473)
  > both run before `place` and write the tuple of item 1 in the placement operand, which `place` consumes.
  - Placed by: "both run before `place` and write the tuple of item 1 in the placement operand, which `place`
    consumes." design/unequal-shards-change-request.md:473 -- the flow is the two algorithms of item 14 run before
    `place`; the document names no function of the data flow for them, and names `place` as the consumer

### no stage named: [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

- **1 / Consumed by: printed** (design/synchronization-signature-change-request.md:103-104)
  > **Consumed by:** it is printed in the three formats of item 2, the monitor's D program is generated from it (item
  > 4), and its SHA-256 is part of the compiled artifact.
  - Placed by: "Each takes the signature of item 1 and prints it."
    design/synchronization-signature-change-request.md:108
- **1 / Consumed by: the monitor's D program** (design/synchronization-signature-change-request.md:103-104)
  > **Consumed by:** it is printed in the three formats of item 2, the monitor's D program is generated from it (item
  > 4), and its SHA-256 is part of the compiled artifact.
  - Placed by: "The monitor is DTrace, in a process of its own."
    design/synchronization-signature-change-request.md:130; "The monitor's D program is generated from the signature
    of item 1." design/synchronization-signature-change-request.md:138
- **2 / Matrix Market** (design/synchronization-signature-change-request.md:110-114)
  > **Data-structural: Matrix Market coordinate format** (Boisvert, Pozo and Remington, NIST 1996). The data structure
  > is the graph as a sparse adjacency matrix, as in Kepner and Gilbert's *Graph Algorithms in the Language of Linear
  > Algebra* (SIAM 2011), with one line `i j delay` per edge. Comment lines first list, for each index, which of the
  > command queue, the thread and the bridge it is on, its IR vertex and its synchronization operation.
  > `scipy.io.mmread` and GraphBLAS read it.
  - Placed by: "Each takes the signature of item 1 and prints it."
    design/synchronization-signature-change-request.md:108
- **2 / DOT** (design/synchronization-signature-change-request.md:115-117)
  > **Algorithmic: the DOT language** (Graphviz), with one cluster for each of the command queue, the thread and the
  > bridge. `dot` lays it out by its layered method (Gansner, Koutsofios, North and Vo, IEEE TSE 1993). `tred` reduces
  > it transitively (Aho, Garey and Ullman, SIAM J. Comput. 1972), which shows which edges other paths already imply.
  - Placed by: "Each takes the signature of item 1 and prints it."
    design/synchronization-signature-change-request.md:108
- **2 / Space-time diagram** (design/synchronization-signature-change-request.md:118-121)
  > **Terminal: Lamport's space-time diagram** (1978, Fig. 1). The command queue, the thread and the bridge are each a
  > column with time running downward. An event is a ball, and a signal-to-wait edge is a stick between columns. It is
  > drawn as text the way `git log --graph` does it ("Draw a text-based graphical representation of the commit
  > history"; git's graph.c), with rows in Kahn's topological order (CACM 1962).
  - Placed by: "Each takes the signature of item 1 and prints it."
    design/synchronization-signature-change-request.md:108
- **3 / Write: the monitor is DTrace** (design/synchronization-signature-change-request.md:130-131)
  > The monitor is DTrace, in a process of its own. Its Objective-C provider catches every Metal 4 call by selector,
  > whatever class implements it. Its pid provider catches the thread's semaphore waits and shared-event waits.
  - Placed by: "The monitor is DTrace, in a process of its own."
    design/synchronization-signature-change-request.md:130
- **3 / Write: the program is not changed** (design/synchronization-signature-change-request.md:132-132)
  > The program isn't rebuilt, linked against anything or changed to be observed.
  - Placed by: "### 3. The monitor: outside the program, only observing"
    design/synchronization-signature-change-request.md:123
- **4 / Write: the D program stops the process** (design/synchronization-signature-change-request.md:138-140)
  > The monitor's D program is generated from the signature of item 1. When it sees an operation the signature doesn't
  > have at that point in the order of that command queue, thread or bridge, a destructive action stops the process at
  > that call. Its printed stack names the code that added it.
  - Placed by: "The monitor's D program is generated from the signature of item 1."
    design/synchronization-signature-change-request.md:138
- **4 / Write: at the end** (design/synchronization-signature-change-request.md:141-141)
  > At the end, any count that differs fails the run, and both signatures are printed in the three formats of item 2.
  - Placed by: "Replay a captured call graph through its compiled program with the monitor of items 3 and 4 attached."
    design/synchronization-signature-change-request.md:153
- **5 / Write: a wait the signature does not have** (design/synchronization-signature-change-request.md:156-157)
  > A wait the signature does not have stops the playback at the line that issues it. A playback that runs to
  > completion has run exactly the compiler's signature.
  - Placed by: "Replay a captured call graph through its compiled program with the monitor of items 3 and 4 attached."
    design/synchronization-signature-change-request.md:153

### no stage named: [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

- **1 / Replaces** (design/torch-source-lowering-change-request.md:138-139)
  > **Replaces,** in rdma/mesh_compile.py: `_apply` (:1521); `_captured`'s recording of torch's kernels (:1775);
  > `_kernels` (:1394).
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **1 / Write: the task graph** (design/torch-source-lowering-change-request.md:162-165)
  > The data structure is the task graph G = (V, E, w, c) of design/plan-to-bridge-change-request.md R3 (O. Sinnen,
  > Task Scheduling for Parallel Systems, Wiley, 2007), carried in the IR of design/aot-compilation-change-request.md
  > A8. A vertex is a task: a node of the graph the operator lowering produces. An edge is a page passed between
  > tasks.
  - Placed by: "A vertex is a task: a node of the graph the operator lowering produces."
    design/torch-source-lowering-change-request.md:164; "Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **1 / Write: the operator lowering produces them on every backend**
  (design/torch-source-lowering-change-request.md:166-167)
  > The operator lowering produces these vertices and edges, for each function on whichever Apple silicon backend it
  > is placed, the CPU included. Placement and the schedule (item 3) consume them.
  - Placed by: "The operator lowering produces these vertices and edges"
    design/torch-source-lowering-change-request.md:166; "Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **1 / Write: the synchronous, complete, contiguous consumer**
  (design/torch-source-lowering-change-request.md:169-170)
  > the synchronous, complete, contiguous consumer is one task with an edge from every page of its operand, and only
  > its operand, typed `contiguous_input` or `contiguous_output`, is one run of pages;
  - Placed by: "The operator lowering produces these vertices and edges, for each function on whichever Apple silicon
    backend it is placed, the CPU included. Placement and the schedule (item 3) consume them. They are typed and
    scheduled as design/plan-to-bridge-change-request.md R3 and R4 write it:"
    design/torch-source-lowering-change-request.md:166-168
- **1 / Write: the asynchronous, incomplete, noncontiguous consumer**
  (design/torch-source-lowering-change-request.md:171-172)
  > the asynchronous, incomplete, noncontiguous consumer is one task per page, each with an edge from its own page,
  > and its outputs are pages that leave as each is produced;
  - Placed by: "The operator lowering produces these vertices and edges, for each function on whichever Apple silicon
    backend it is placed, the CPU included. Placement and the schedule (item 3) consume them. They are typed and
    scheduled as design/plan-to-bridge-change-request.md R3 and R4 write it:"
    design/torch-source-lowering-change-request.md:166-168
- **1 / Write: the IR carries the kind** (design/torch-source-lowering-change-request.md:173-173)
  > the IR carries which of the two each consumer is (design/aot-compilation-change-request.md A8).
  - Placed by: "The operator lowering produces these vertices and edges, for each function on whichever Apple silicon
    backend it is placed, the CPU included. Placement and the schedule (item 3) consume them. They are typed and
    scheduled as design/plan-to-bridge-change-request.md R3 and R4 write it:"
    design/torch-source-lowering-change-request.md:166-168
- **2 / Write: the operator lowerings, then placement, consume it**
  (design/torch-source-lowering-change-request.md:191-192)
  > The operator lowerings (items 1, 5 and 6) and then placement consume it.
  - Placed by: "The operator lowerings (items 1, 5 and 6) and then placement consume it."
    design/torch-source-lowering-change-request.md:191-192; "Items 1, 5 and 6 are operator lowerings: they run where
    the exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before
    scheduling." design/torch-source-lowering-change-request.md:129-130
- **5 / Replaces** (design/torch-source-lowering-change-request.md:275-275)
  > **Replaces,** in rdma/mesh_compile.py: `_lstsq_metadata`'s restriction, "driver must be gels or gelsy".
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: one node, one LAPACK driver** (design/torch-source-lowering-change-request.md:291-294)
  > The data structure is one node of the exported graph (item 2), with fixed shapes. The algorithm is the LAPACK
  > driver for the node's problem, called through Accelerate. `torch.linalg.svd`, `eigh`, `eig` and `eigvals`, and
  > `lstsq` with the gelsd and gelss drivers, are each one such node. The iteration is inside that one function.
  > Nothing in the graph branches or repeats on tensor data.
  - Placed by: "This is an operator lowering. It produces that node in the exported graph"
    design/torch-source-lowering-change-request.md:295; "Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: it is an operator lowering** (design/torch-source-lowering-change-request.md:295-296)
  > This is an operator lowering. It produces that node in the exported graph; placement and the schedule (item 3)
  > consume it as they do any other node.
  - Placed by: "This is an operator lowering. It produces that node in the exported graph"
    design/torch-source-lowering-change-request.md:295; "Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: which routine** (design/torch-source-lowering-change-request.md:297-299)
  > For `lstsq` the quoted page names the routine: `gelsd` and `gelss`. For `svd`, `eigh`, `eig` and `eigvals` no
  > quoted source names the LAPACK routine PyTorch or JAX calls on a CPU; the header lines above show only which
  > drivers Accelerate declares.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: INFO** (design/torch-source-lowering-change-request.md:300-300)
  > The routine's `INFO` is a value the node returns; a positive `INFO` is reported.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: pinv** (design/torch-source-lowering-change-request.md:301-301)
  > `torch.linalg.pinv` is the composition PyTorch states, of the node for `svd` or for `eigh`.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: `_lstsq_metadata`** (design/torch-source-lowering-change-request.md:302-302)
  > `_lstsq_metadata` takes the four drivers PyTorch names, with the output shapes PyTorch gives each.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: matrix_exp** (design/torch-source-lowering-change-request.md:303-303)
  > `torch.linalg.matrix_exp` has no such function in LAPACK. Item 6 writes it.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: the plain message** (design/torch-source-lowering-change-request.md:304-309)
  > Every operator whose method, on the processor the placement names, needs an iteration depth that tensor data
  > decides gets, from the operator lowering, one plain message that names the operator and says that an iteration
  > depth that depends on data is not compiled (the operator, above: "as long as we emit boring ah ah errors which
  > tell the user in boring ah ah terms that data dependent iteration depth isn't compilable with meshlib"). Off the
  > CPU that includes `eigh`, what is composed from it, and the polar decomposition's Halley continuation, while item
  > 6 states no fixed depth for them.
  - Placed by: "gets, from the operator lowering, one plain message"
    design/torch-source-lowering-change-request.md:305; "Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: the cited functions are the CPU's** (design/torch-source-lowering-change-request.md:310-313)
  > The functions cited above are the CPU's. Placement is an operand (the operator, quoted in
  > design/aot-compilation-change-request.md A2: "we simply need to consume operands which choose these placements,
  > letting us define and implement an optimizing chooser function *later*"). Item 6 states what these operators are
  > on the other processors.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **5 / Write: no body repeated by tensor data** (design/torch-source-lowering-change-request.md:314-314)
  > No lowering written by mesh repeats a body a number of times that tensor data decides.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: nodes of the exported graph on every processor**
  (design/torch-source-lowering-change-request.md:388-392)
  > This is an operator lowering. Each operator below becomes nodes of the exported graph (item 2): matrix
  > multiplications, QR factorizations, Cholesky factorizations, triangular solves and selections among computed
  > values. Placement and the schedule (item 3) consume them as they do any other node. The placement operand names
  > each node's processor, on the GPU, the CPU or the Neural Engine, and its sharding across machines. No node of
  > these compositions is confined to one processor by this request.
  - Placed by: "This is an operator lowering. Each operator below becomes nodes of the exported graph (item 2)"
    design/torch-source-lowering-change-request.md:388; "Items 1, 5 and 6 are operator lowerings: they run where the
    exported graph is decomposed, before the placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: the polar decomposition** (design/torch-source-lowering-change-request.md:393-403)
  > The algorithm for the polar decomposition is QDWH as `_qdwh` states it (jax/_src/tpu/linalg/qdwh.py). The
  > coefficients of its steps are computed before any step runs, from `l = eps`, by `while l + tol_l < 1 and k <
  > max_iterations:`; `max_iterations` is an operand whose default is JAX's 10. Each of those steps is a QR
  > factorization where `c > CHOLESKY_CUTOFF`, or a Cholesky factorization and two triangular solves otherwise, with
  > matrix multiplications. Their number is known at compilation. Whether the iteration converged is a value returned
  > with the result. No quoted source says how many to compile of the steps `_qdwh` runs after its computed
  > coefficients are used up: "If l has converged but u still has not, continue with Halley's method (coef = None)
  > until convergence", under `lax.while_loop(cond_fun, body_fun, (k, u, is_not_converged))` with `return
  > jnp.logical_and(is_not_converged, k < max_iterations)`. After them the refinement `u = 1.5 * u - 0.5 * u @
  > (u.T.conj() @ u)` and the Hermitian factor `h = u.T.conj() @ x`, `h = (h + h.T.conj()) / 2` follow as `_qdwh`
  > states them.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: the Halley continuation** (design/torch-source-lowering-change-request.md:398-401)
  > No quoted source says how many to compile of the steps `_qdwh` runs after its computed coefficients are used up:
  > "If l has converged but u still has not, continue with Halley's method (coef = None) until convergence", under
  > `lax.while_loop(cond_fun, body_fun, (k, u, is_not_converged))` with `return jnp.logical_and(is_not_converged, k <
  > max_iterations)`.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: eigh** (design/torch-source-lowering-change-request.md:404-408)
  > The algorithm for `eigh` is QDWH-eig (Nakatsukasa and Higham, §3.1), in which a polar decomposition splits the
  > matrix into two blocks that share its eigenspaces below and above a split point, and the same is done to each
  > block. No quoted source gives QDWH-eig a number of steps or block sizes that tensor data does not decide: in
  > `_eigh_work` both are decided by the matrix's values, under `lax.while_loop`. No quoted source states the
  > eigendecomposition of a block at the termination size when that size is more than 1.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: svd, pinv, lstsq** (design/torch-source-lowering-change-request.md:409-411)
  > The algorithm for `svd` is Algorithm 2 (Nakatsukasa and Higham, §4): the polar decomposition, the
  > eigendecomposition of its Hermitian factor, and one product. `pinv` and `lstsq` with gelsd and gelss are the
  > compositions PyTorch states (item 5), of that `svd` or `eigh`.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: matrix_exp** (design/torch-source-lowering-change-request.md:412-417)
  > The algorithm for `matrix_exp` is scaling and squaring as `_calc_P_Q`, `_squaring` and `expm` state it. The count
  > `n_squarings` is a value computed from the L1 norm. There are `max_squarings` steps, an operand whose default is
  > JAX's 16; step `i` gives the square where `i < n_squarings` and its input otherwise. Each `lax.cond` and
  > `lax.switch` of the source is a selection among computed values, as JAX states of `cond` under `vmap`, and not a
  > branch. Where `n_squarings > max_squarings` the result is NaN, as `_nan` states, and the node reports a runtime
  > error (the operator, above: "a runtime error and some NaNs").
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: the CPU for the whole operator** (design/torch-source-lowering-change-request.md:418-418)
  > Where the placement operand names the CPU for the whole operator, the operator is item 5's node.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: no body repeated by tensor data** (design/torch-source-lowering-change-request.md:419-419)
  > No lowering here repeats a body a number of times that tensor data decides.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: eig and eigvals off the CPU** (design/torch-source-lowering-change-request.md:420-420)
  > No quoted source states `eig` and `eigvals` of a matrix that is not Hermitian on the GPU or the Neural Engine.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130
- **6 / Write: the Neural Engine** (design/torch-source-lowering-change-request.md:421-421)
  > No quoted source says which of these operations the Neural Engine executes, or at what precision.
  - Placed by: "Items 1, 5 and 6 are operator lowerings: they run where the exported graph is decomposed, before the
    placement stage adds its `send` and `recv` nodes and before scheduling."
    design/torch-source-lowering-change-request.md:129-130

### no stage named: [page-queue-change-request.md](page-queue-change-request.md)

- **The operator's statements: Collective size** (design/page-queue-change-request.md:12-13)
  > buddy pairs don't exist anymore, it's already been said that all code now must operate on collectives of arbitrary
  > > node count and there won't be any compromises here.
  - Placed by: "all code now must operate on collectives of arbitrary > node count"
    design/page-queue-change-request.md:12-13 (the clause names all code, not one stage)
