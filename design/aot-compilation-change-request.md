# Change request: the ahead-of-time compilation

This change request states what the ahead-of-time compilation does: what it takes, the functions it runs, the types
that pass between them and the static schedule it serializes. The static schedule of a compiled program (M. Lam,
Software pipelining, PLDI 1988; S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in
multiprocessor DSP systems, IEEE Trans. Signal Processing 45(6), 1997, here BSL) is what `mesh_plan` computes.
design/plan-to-bridge-change-request.md states what the static schedule gives the bridge. This document states how a
program becomes that static schedule. Each requirement quotes the operator, then names its data structure and its
algorithm, each from the literature and used as its source states it, and says which function produces the data
structure and which consumes it.

The compilation takes operands that choose what it does not choose itself: where each part of a function runs (A2)
and how much each task and transfer weighs (A3). It contains no chooser for either. A chooser that fills those operands
is defined later, apart from all of this.

## Scope

> the code is not about e2b and is about instead literally all tensor programs that can ever be written in torch and
> torch like languages and dsls for accelerated edge optimized deployment of tensor programs, so stop acting liek this
> is about a specific program called e2b

> every problem presented has a tangible and immediate application which is an example of a broad calss of problems
> which will also be supported at the very next step in integration once specific issues are resolved, this is
> inductively demonstrated everywhere, and there is no excuse for making-excuses.

Every requirement here holds for every tensor program written in torch or a torch-like language or DSL, and each is
justified by a cited algorithm or data structure, never by one program.

## How this document is written and used

> if this is a big deviation from the previous specification of any code we were using, write a change request
> document which covers all of the requirements and staes of affaires described by the user, the narrowness of scope
> of the affected data structures and functions, and write tshis as a document insdie of the docs or whatever of the
> corresponding repos, not as some wacky private note-for-subagents. the verbatim notes discusisng this with the user
> would be fine. no embellishments are needed or allowed.

(T173)

> write up yet another change request document, just like the last one, which uses block quoted user and assistant
> turns for the body of the text to make it as hard as possible to pretend there's an ambiguity or decision to make
> here, rather than an unassailable and certain difference which must be remedied by a boring and canonical path

(T212)

> append your proposal to the change request and we finally have a unit of work we can assign to a subagent; the unit
> of work *is* the change request document once the claude proposal is part of it.

(T178)

> "Citation practice:" this was the only acceptance criterion used for document format and content

(2026-11-04, after T321)

- Where the document is silent, the implementer takes the minimal decision that follows from the document and the
  code, and records it as its own choice, with its reason, apart from the document's requirements.
  > "take the minimal decision … and record it" broadly fine if you really think about it especially if agents are
  > strongly segmenting 'i made an apparnetly required or apparently free choice... and i'm *documenting* it as my
  > choice rather than preordained fate': this is more honest and makes revision and acceptance of corresponding code,
  > measurements, and artifacts simpler in expectation.

## The turns

The turns are held once, in design/plan-to-bridge-change-request.md, "The turns", after
design/disjoint-pages-change-request.md D.1. They are this document's body.

### The operator, 2026-11-10

> review the change requests one more time. here are three major kinds of defects to correct:
> 1: misleading pedantic specificity, where there are 23 numbered and lettered subsections which describe something
> about a source but don't say what algorithm is being used, what data structure is being used, and where it composes
> in a DAG of functions and the type interfaces between them.
> 2: misleading pedantic vagueness, where terms with no grounding to algorihtms, data structures, or the setting are
> introduced. `the plan that gates the flimgrundler isn't handled by the srotheclave executor, so we have to stop for
> that` would be misleading pedantic vagueness, especially with no canonical data structure, algorithm, or method
> justifying why the change request pertains to `srotheclaves`, `frimgrundlers`, `plans` or `gates`.
> 3: invented qualitative limits that do not come from the change request's plain semantics and plain use as a schema
> in the first `3` cases of change request documents used in the repo history.

## Claude's proposal

This section and the appendices are Claude's (T178). Of an earlier text of the appendix "the citations the
implementation omitted or ignored", the operator: "\"1->8\" acceptable as written here; launch revisions".

### The stages each requirement belongs to

Each stage is a function from a value to a value. The reference program ~/schedule-emit implements this data flow, and
its function and type names are given beside each stage.

1. **Export** (schedule_emit/export.py `export`): a program in, its dataflow graph out (`ir.Exported`).
2. **Placement** (placement.py `place`): the exported graph and the placement operand in; the same graph out, with
   each operation's processor and with send and recv nodes added.
3. **Devices** (devices.py `on_devices`): the placed graph and the mapping of processors onto devices in; the placed
   graph with each operation on its device out.
4. **Schedule** (schedule.py `schedule`): the placed graph and the cost operand in; each processor's operations in
   scheduled order out. There is one schedule.
5. **The scheduled value** (`ir.Scheduled`): what the four stages produce.
6. **Lowering** (lower_inductor.py `lower`), **the runtime** (runtime.py `Program`) and **the archive** (archive.py
   `save`, `load`) each take the scheduled value.

| Requirement | Stage |
| --- | --- |
| A1 | export |
| A2 | placement and devices: their operands, and the partitioning |
| A3 | the cost operand of the schedule, and the capacities of A4's routing |
| A4 | schedule: `mesh_plan` routes the collectives placement inserted and lowers them to sends and receives of pages |
| A5 | schedule: the page table's bindings it writes into the scheduled value |
| A6 | schedule |
| A7 | the runtime |
| A8 | the type that passes from placement to the schedule |
| A9 | the scheduled value and its archive; the runtime takes it |
| A10 | export (the calls' graph), then an analysis whose output the schedule and the runtime take |

### A1. The input is each program's captured dataflow graph, with shapes and types

> the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'

> Our PyTorch code can already record a whole function as a graph of steps before it runs. The graph shows which results
> each step needs.

(data-model §6)

- **Stage:** export.
- **The data structure is the exported program's dataflow graph:** a graph of operations whose tensors have their
  shapes and element types. Nothing at run time changes a shape. It is `torch.export`'s "full graph representation of a
  PyTorch model" (PyTorch v2.11.0, docs/source/user_guide/torch_compiler/export.md).
- **The algorithm is `torch.export`'s ahead-of-time trace.** `export` "produces a traced graph representing only the
  Tensor computation of the function in an Ahead-of-Time (AOT) fashion, which can subsequently be executed with
  different inputs or serialized" (torch 2.14.0, torch/export/__init__.py).
- **It is not a `torch.compile` backend.**
  > is it idiomatic claudelish trying to imply a just in time or online problem when the user sided documentation
  > emphasizes mayn constraints including saying in as many semantic frames as possible 'the bridge and mesh
  > collectives do not support dynamic or eager operations conmingled with compiled operations, and we never run
  > non-compiled operations ever, damn the consequences'?
  > "compiling each torch function at its first call (BS-3). " has anyone asked for this beidse claudes trying to
  > slowroll, procrastinate, delay the project, or otherwise tamper without benefit or cause?
  - The operator's turns: "compile the next transaction aot based on literally any relevant information instead of
    doing a lazy jit syncing stalling waiting pipeline choking 'ask'" (design/plan-to-bridge-change-request.md, the
    turns), and torch's work "should be AOT and produce serialized objects we cna resume without needing to repeatedly
    re-compile anything" (2026-10-28).
  - Torch's documentation states the difference. "`torch.compile` is a JIT compiler"; "When `torch.compile` runs
    into an untraceable part of a model, it will "graph break" and fall back to running the program in the eager
    Python runtime. In comparison, `torch.export` aims to get a full graph representation of a PyTorch model, so it
    will error out when something untraceable is reached. Since `torch.export` produces a full graph disjoint from
    any Python features or runtime, this graph can then be saved, loaded, and run in different environments and
    languages" (PyTorch v2.11.0, docs/source/user_guide/torch_compiler/export.md, "JIT vs. AOT" and "Partial vs. Full
    Graph Capture").
- **Produced by, consumed by.** Export takes the torch program and returns its graph. A2's placement takes that graph.
  The reference program's type for it is `ir.Exported`, which holds a `torch.fx.GraphModule`. No cited mesh code names
  a type for it.
  Every function of a torch program is captured and compiled before the program runs, the same at every rank, and
  nothing runs eagerly between its compiled calls. The program's calls between functions are A10's graph.
- **A program run through another framework's delegate.** Its graph is the recorder's capture of the delegate's
  commands (metal_recording.m). For the program the code runs today on LiteRT, tools/prepare_e2b.py reads the model's
  graph.
- **State of the code.** mesh called `torch.export.export(module, tuple(arguments), strict=False)` in
  rdma/torch-mesh/torch_mesh/engines.py from 2026-10-26 until 1f794b8f removed that frontend.

### A2. The placement operand: which nodes hold which sections

> which nodes are chosen to have which sections of a parallelizable task at all??? this is barely descriebd at all,
> ebcause the problem of optimizing or choosing this modeling task is *evne more deferred* than the previous topic's
> urgent deferral of how we choose operands. we simply need to consume operands which choose these placements, letting
> us define and implement an optimizing chooser function *later* in a manner which is totally decoupled from all
> implementation details addressd at this time.

- **Stage:** placement, then devices.
- **The data structure is the placement operand,** of two parts:
  - sharding specs on the function's tensors over a logical device mesh (Y. Xu et al., GSPMD: general and scalable
    parallelization for ML computation graphs, arXiv 2105.04663, 2021);
  - the mapping of the mesh's devices onto the map's nodes (rdma/mesh-map.h). The logical mesh and its mapping onto
    physical devices are Alpa's (L. Zheng et al., Alpa: automating inter- and intra-operator parallelism for
    distributed deep learning, OSDI 2022, §4.1): "For a set of physical devices, there can be multiple logical views",
    and the mapping is chosen by a pass of its own.
- **The algorithm is GSPMD's partitioning.** It consumes sharding annotations on a few tensors, propagates them to
  every operator by compiler passes over a dataflow IR, and inserts the collectives. It "produces a single program for
  all partitions" (GSPMD §1).
- **Produced by, consumed by.** The caller passes the operand. Placement takes A1's graph and the sharding specs and
  returns the partitioned graph with its collectives, which A4 turns into sends and receives of pages. Devices takes
  that graph and the mapping onto the map's nodes. A6's schedule takes the result, in A8's IR. In the reference
  program the operand is `placement`, a function from an fx node to a processor name, and `processors`, a dict from
  processor name to torch device. No cited mesh code names a type for either part.
- **The chooser is a separate pass that emits the same operand, and this compilation contains none:**
  - Alpa's own chooser works this way. "Alpa compiles each stage against its assigned device mesh, respecting the
    intra-op parallelism plan output by the ILP solver. The compilation depends on XLA and GSPMD" (Alpa, §6).
  - A later chooser of the mapping onto the cabling may be topology mapping (T. Hoefler and M. Snir, Generic topology
    mapping strategies for large-scale parallel architectures, ICS 2011).

### A3. The weighting operand: what each task and transfer costs

> 1: multiplexing and routing link scheduling must be done with respect to weighting operands which can express any
> relationship we expect to apepar later (consuming an operand of a certain type lets us defer the design or
> implementation of an optimizing chooser here).

- **Stage:** the cost operand of the schedule (A6), and the capacities of the routing (A4).
- **The data structure is the weights of the task graph:** a weight w on each task and a weight c on each edge (O.
  Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007; Y.-K. Kwok and I. Ahmad, ACM Comput. Surv., 1999), and the
  cables' capacities, from the map.
- **No algorithm computes them here.** The compilation takes every weight its router and its scheduler use as an
  operand. The task graph's weights w and c are given inputs of the scheduling problem (Sinnen; Kwok and Ahmad). The
  router's capacities and demands are the given inputs of concurrent multicommodity flow (N. Garg and J. Könemann,
  Faster and simpler algorithms for multicommodity flow and other fractional packing problems, FOCS 1998).
- **Produced by, consumed by.** The caller passes the operand. A4's routing takes the capacities, and A6's schedule
  takes w and c. In the reference program the operand is `cost`, a function from an fx node to a float. No cited mesh
  code names a type for the weights.
- **The chooser of the weights is defined later, apart, as the chooser of A2 is.**

### A4. Collectives lower to page transfers over packed trees

> the point of having aot schedulign and compilation of schedules is to allow us to interleave any combination of
> transport requirements concurrently with one another, s.t. we are never underusing the bandiwdth on any cable routing
> when we could counterfactually swizzle together more communications to cause pipeline-blocking operations to begin or
> conclude faster, no matter waht they are!

(of "route them jointly with the ported concurrent-flow solve (Garg–Könemann), rather than each in isolation")

> routing within tb5 linked collectives is supported; attempting to involve nodes with no tb5 dma connection to any node
> within the collective is not allowed.

- **Stage:** schedule. Placement inserts the collectives (A2). `mesh_plan`, the schedule stage's function, routes them
  and lowers them to sends and receives of pages (rdma/mesh-collective.c). The reference program's `place` adds its
  send and recv nodes at placement and routes nothing.
- **The data structures are two:**
  - **The instruction DAG.** MSCCLang (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS
    2023, §4 and §5) lowers operations on chunks into an instruction DAG of sends, receives and
    receive-reduce-copies. A relay's forward is its fused receive-copy-send. Here a chunk is a page.
  - **Packed spanning trees,** the routes. Blink (G. Wang et al., Blink: fast and generic collectives for distributed
    ML, MLSys 2020) builds collectives on them. §3.1: a broadcast's optimal rate is reached "by finding the maximal
    packing of a number of directed spanning trees or arborescences in the graph".
- **The algorithms:**
  - **Packing.** Blink §3.2: the packing is a multiplicative-weight update for fractional packing. `mesh_graph_pack`'s
    Garg and Könemann rounds are a multiplicative-weight method of the same kind.
  - **Joint routing.** All of one function's collectives are routed jointly over the Thunderbolt-linked nodes of the
    map, as concurrent multicommodity flow, the problem Garg and Könemann's rounds solve (N. Garg and J. Könemann,
    FOCS 1998).
  - **AllReduce.** Blink §3.3: "to do an AllReduce operation on the directed graph, we first run a reduce operation to
    a chosen root vertex using the undirected graph and then do a broadcast operation from the root vertex using the
    same tree but with links going in the reverse direction."
  - **AllGather** is one broadcast from each source over its packed trees: §3.1's broadcast, once for each source, as
    rdma/mesh_compile.py `_all_gather` declares it. That composition is this document's; Blink's own §4.1 states
    "AllGather is AllReduce without using a reduction function".
  - **ReduceScatter** "is semantically equivalent to an AllReduce followed by a DynamicSlice where each partition gets
    one slice of the fully reduced data. An efficient implementation has only half the cost of AllReduce" (GSPMD §4.2).
    Here each slice is reduced to the member that keeps it, over its packed trees: the first half of Blink §3.3's
    AllReduce, with that member as the root. That reading of GSPMD's half is this document's.
  - **CollectivePermute** "exchanges data among a list of source-destination pairs" (GSPMD §4.2): one point-to-point
    demand a pair, routed as AllToAll's are.
  - **AllToAll** is point-to-point demands, one per ordered pair of ranks, routed jointly as concurrent multicommodity
    flow.
- **Produced by, consumed by.** A2's partitioning produces the collectives, in A8's IR. `mesh_plan` takes them, the
  map and A3's capacities, and produces the instruction DAG with each instruction's route. A6's ordering, in the same
  function, takes it.
  `mesh_plan` takes `struct mesh_function`, whose `declared` are the collectives' `struct mesh_declared`
  (rdma/mesh-collective.h). It writes the instructions as `struct plan_entry` (rdma/mesh-collective.c) and from them
  the `struct mesh_hop` lists (rdma/mesh.h).
- **Where the partitioning inserts ReduceScatter and CollectivePermute:** where GSPMD's does.
  - ReduceScatter goes where a reduction's result is sharded: "gradients will be communicated with ReduceScatter
    instead of AllReduce" (GSPMD §3.2).
  - CollectivePermute goes into halo exchange, "We use the CollectivePermute operator to exchange halo data between
    partitions" (§4.3), and into a pipeline's buffer shifting, "GSPMD will turn the buffer shifting into cross-device
    communication via CollectivePermute" (§3.3).
  - The partitioning reaches them by partitioning every operator, as A2 states ("propagates them to every operator").
    GSPMD gives the rule for the operators that need CollectivePermute: "The most typical use case of halo exchange is
    for partitinoning window-based operators (e.g., Convolution, ReduceWindow)" (§4.3).
- **Which nodes hold a function's static schedule.** Each compiled function has its own static schedule (appendix item
  6). Every node of the map is a node of the program, since every run is launched on the whole collective, so every
  relay holds the static schedule (A7, sender(m)).
- **A node to which the placement operand maps no device** (A2) compiles and schedules each function as every node
  does, with no section of its own. It declares the function's transfers as every rank does, and its part of each
  call is the relaying its routes give it. NCCL states the rule for every rank of a communicator: "Collective
  operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form
  a complete collective operation. Failure to do so will result in undefined behavior, including hangs, crashes, or
  data corruption." (NCCL User Guide, "Collective Operations"). With `torch.export` (A1), every rank compiles the same
  exported program before it runs. A node with no section starts its static schedules' executions, relays as its
  routes give it, and computes nothing; it has no value for any program code to read.
- **State of the code.**
  - `mesh_plan` routes every demand of one static schedule at once (rdma/mesh-collective.c, `mesh_graph_flow`).
  - rdma/mesh_compile.py at mesh 26701d8e has `_operator`, `_convolution`, `_reduce_scatter` and
    `_collective_permute`.

### A5. Every page is placed ahead of time

- **Stage:** schedule. It writes the page table's bindings into the scheduled value (A9).
- **The data structure is the page table:** names bound to frames (T. Kilburn, D. B. G. Edwards, M. J. Lanigan and F.
  H. Sumner, One-level storage system, IRE Trans. Electronic Computers, 1962; P. J. Denning, Virtual memory, ACM
  Comput. Surv., 1970), as design/plan-to-bridge-change-request.md R2 states it.
- **The algorithm is the allocator's, as R2 states it:** the u copies of appendix item 1 (Lam, 1988); the order in
  which two items may share a page, Lamport's happens-before (L. Lamport, Time, clocks, and the ordering of events in a
  distributed system, CACM 21(7), 1978); and the fewest pages as the fewest chains of that order, by Dilworth's
  theorem, found by maximum bipartite matching (R. P. Dilworth, A decomposition theorem for partially ordered sets,
  Annals of Mathematics 51(1), 1950; D. R. Fulkerson, Note on Dilworth's decomposition theorem for partially ordered
  sets, Proc. AMS 7(4), 1956).
- **Produced by, consumed by.** The allocator takes the values of A8's IR. The bridge and the GPU's program take the
  bindings from the scheduled value (A9). rdma/mesh.h has `struct mesh_page_entry`;
  it was not checked that this is the binding's type.

### A6. One schedule per function

- **Stage:** schedule.
- **The data structure is the task graph** G = (V, E, w, c) of design/plan-to-bridge-change-request.md R3 (O. Sinnen,
  2007; Y.-K. Kwok and I. Ahmad, 1999), with R4's consumers' kinds as its shape and A3's weights.
- **The algorithm is design/plan-to-bridge-change-request.md R5's:** R5's heap order by the completion times of one
  linear program (C.-Y. Chen, Scheduling coflows with precedence constraints for minimizing the total weighted
  completion time in identical parallel networks, arXiv 2205.02474, 2022; L. A. Hall, A. S. Schulz, D. B. Shmoys and J.
  Wein, Scheduling to minimize average completion time: off-line and on-line approximation algorithms, Math. Oper.
  Res. 22(3), 1997), solved by HiGHS.
- **Produced by, consumed by.** The schedule takes A8's IR. It writes, for each adjacent node and direction, the
  ordered list of page instructions of A9. The bridge takes those lists. Each entry is a `struct mesh_hop`.

### A7. Function executions of programs that run at once are ordered by one scheduler for the island

> nothing happens when a program joins or leaves because a program is a sequence of like, function calls.

> it is rqeuired to choose nad use a scheduler. cite anyhthing do anything but don't use idiomatic citation free claude
> code anywhere here

> "The real question sits at the sequencer's level. " perhaps this must be said more definitely and more literally and
> more citationally in the change request documentation s.t. any competing interpretations are completely ablated...

- **Stage:** the runtime. It takes the scheduled value's name and nodes and orders the calls of its functions.
- **What the code must do.** Order the function executions of the program on the island in one order, the same at
  every node, and keep the program's own order of its calls within it.
  - A run is one program on the whole collective. "Across all programs running on the island" is Pathways' setting,
    many clients' programs, and no requirement here asks for it.
    > why do you think you need special rules about 'numbers of programs at once'? literature. citations. canonical
    > algorithms and datastructures. ignore and brush away idiomatic explanation, specification, etc.
    The operator's turn on attaching: "is attaching conceptually real or do programs do nothing but what the allocator
    lets them, with the shapes and coarse bin packing requierments of allocation known aot?"
    (design/plan-to-bridge-change-request.md, the turns). The scheduler below orders the one program's function
    executions, as it is written. Its one sender keeps FIFO Order (below).
- **Why there is one order.** Pathways (P. Barham et al., Pathways: asynchronous distributed dataflow for ML, MLSys
  2022):
  - §2: "the system will deadlock if communicating computations are not enqueued in a consistent order".
  - §4.4: "The PATHWAYS runtime includes a centralized scheduler per island that consistently orders all of the
    computations in the island", by "communicating with the scheduler to determine a consistent order of function
    executions across all programs running on the island". "Our current implementation simply enqueues work in FIFO
    order."
  - The island is the map's Thunderbolt-linked component (A4). Each function execution runs its own static schedule
    (A6).
- **The algorithm is total order multicast by a fixed sequencer**, as X. Défago, A. Schiper and P. Urbán define both
  (Total order broadcast and multicast algorithms: taxonomy and survey, ACM Comput. Surv. 36(4), 2004). Nothing else
  orders function executions.
  - **The primitives** (§3): "total order broadcast is defined in terms of two primitives, which are called
    TO-broadcast(m) and TO-deliver(m)". Every message "carries the identity of its sender, denoted by sender (m)".
  - **The properties the code must have**, verbatim:
    - "(VALIDITY) If a correct process TO-broadcasts a message m, then it eventually TO-delivers m." (§3)
    - "(UNIFORM AGREEMENT) If a process TO-delivers a message m, then all correct processes eventually TO-deliver m."
      (§3)
    - "(UNIFORM INTEGRITY) For any message m, every process TO-delivers m at most once, and only if m was previously
      TO-broadcast by sender (m)." (§3)
    - "(UNIFORM TOTAL ORDER) If processes p and q both TO-deliver messages m and m′, then p TO-delivers m before m′ if
      and only if q TO-delivers m before m′." (§3)
    - "(FIFO ORDER) If a correct process TO-broadcasts a message m before it TO-broadcasts a message m′, then no
      correct process delivers m′ unless it has previously delivered m." (§4.3.1)
    - "(GLOBAL TOTAL ORDER) The relation < is acyclic, where < is defined as follows: m < m′ if and only if any correct
      process delivers m and m′, in that order." (§5.2.3)
  - **The steps** are their Fig. 5, the "UB" variant of §7.1, as written:
    - Sender: "procedure TO-broadcast(m): send (m) to sequencer".
    - Sequencer: "Initialization: seqnum := 1"; "when receive (m): sn(m) := seqnum; send (m, sn(m)) to all; seqnum :=
      seqnum + 1".
    - Destinations: "Initialization: nextdeliver := 1; pending := ∅"; "when receive (m, seqnum): pending := pending ∪
      {(m, seqnum)}; while ∃(m′, seqnum′) ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver :=
      nextdeliver + 1".
  - **Multicast to many groups** is their §5.2.4, as written: "1. form a super-group with the union of all destination
    groups; 2. whenever a message m is multicast to a group, multicast it to the super-group, and 3. processes not in
    Dest(m) discard m." The super-group is the island's nodes. One sequence numbers every message and every node
    delivers in its order, so no two nodes deliver two messages in opposite orders, and Global Total Order holds.
- **The data structures are Fig. 5's:** the sequencer's `seqnum`; each destination's `nextdeliver` and `pending`; and
  the message (m, sn(m)), which is rdma/mesh-bridge.h `struct mesh_order`.
  - **m** is one submission: its static schedule's name (`group`, `epoch`), the identifier of the view it is sent in
    (below), and a range of its executions [`first`, `end`). One submission is one message however many executions it
    carries, as Pathways sends "a single message (describing the entire subgraph)" (§4.5). e2b's run is one message.
  - **sender(m)** is the program's one sender: the least of the program's nodes. It TO-broadcasts the program's
    submissions in the order the program makes them. The program's nodes are the nodes its launcher started it at,
    an operand of the `mesh_plan` call (`program`) that the launcher passes as it passes `group` (MESH_GROUP). e2b's
    launcher, tools/mesh/run_pair.py, passes no node set: it starts rank r at map index r on every node of the
    collective, so e2b's program's nodes are map indices 0 … R − 1, which tools/e2b_replay.m passes. A torch
    program's launcher passes them as MESH_PROGRAM beside MESH_GROUP (rdma/mesh_compile.py `_program`). Every run is
    launched on the whole collective ("you may never use less than the entire collective under any circumstances.
    all is okay, 2 is not okay. all is cheaper than 2."), so the program's nodes are every member of the view, every
    relay of A4's routes is one of them, and every node of every static schedule, relays among them, calls
    `mesh_plan` and holds the static schedule.
  - **The sequencer** is the bridge of the highest-numbered node the map reaches (rdma/mesh-records.c `island_set`),
    the code's choice, which the class leaves open. "The sequencer is unique, and the responsibility is not normally
    transfered to another processes" (§7.1). It creates the island's views (below).
  - **sn(m)** is struct mesh_order's `sequence`.
  - **Dest(m)** is the nodes of m's static schedule (struct mesh_order's `nodes`).
  - **deliver(m)** at a node of Dest(m) appends m to its static schedule's queue there (appendix item 2), which
    releases its executions at the node. The pages of delivered executions interleave on each cable (T99).
- **Produced by, consumed by.** The program's sender TO-broadcasts m. The sequencer numbers it and sends (m, sn(m)) to
  all. Each node of Dest(m) delivers it into the queue of appendix item 2, and the bridge at that node takes it from
  the queue. The message is `struct mesh_order` up to its delivery. The queue's entry, which the bridge takes, is
  `struct mesh_delivery` (rdma/mesh.h).
- **Failures.** "This algorithm does not tolerate the failure of the sequencer" (§7.1). A static schedule is used only
  while its map is unchanged (design/disjoint-pages-change-request.md D.3.6), so the loss of a cable or node ends
  every static schedule that uses it, and the sequencer of a static schedule in use does not change. A static schedule
  is also used only within its view (below).
- **One sender for each program.** "Total Order alone does not guarantee that messages are delivered in the order in
  which they are sent" (§4.3.1), and FIFO Order is a property of one sender's messages. Where two nodes broadcast calls
  of one program, nothing makes one node's broadcast precede the other's (Lamport's "precedes", their Definition 1), so
  neither FIFO Order nor Causal Order (§4.3.2) need keep the program's order. With one sender, whose messages reach the
  sequencer first in, first out, Fig. 5's sequencer numbers them in the order they were sent ("when receive (m) sn(m)
  := seqnum"). So FIFO Order holds, and with it A10's order constraints between the program's calls.
  > 'blah blah blah deadlock' do the works we cite in the literature have deadlocky runtimes or compiletimes or was that
  > a deviation from the work we were developing?

  > ""Holding a lane," plainly:" did you need to talk about this at all or is it an irrelevant distraction from the levle
  > ofa bstraction and the specific scopes of interfaces involved in the change request?

  > "That was the whole deadlock: my A7 gave one program several senders."
  > then this is an undercitation over-idiom issue and has a known remedy (making specification more explicit, more
  > citational, less idiomatic, removing ad libs that feel zesty and fun and exciting to write.)
- **The island is a dynamic group, and each of its views is a static system.** Section 3's specification is "the
  standard specification of total order broadcast in a static system, that is, a system in which all processes are
  created at system initialization" (§6). The island is not one: its bridges start, and start again, at different
  times. "A dynamic group is a group of processes with a membership that can change during the computation ... the
  successive memberships of the group are called the views of the group" (§6.1), and "Total order broadcast in a system
  with dynamic groups can hence be specified as view synchrony plus an additional order property" (§6.1). The
  membership is "a group membership service ... Whenever the membership changes, the service report changes to all
  group members, by providing them with the new view" (§8.2).
  - **The view properties the code must have** are G. V. Chockler, I. Keidar and R. Vitenberg's (Group communication
    specifications: a comprehensive study, ACM Comput. Surv. 33(4), 2001), verbatim:
    - "(Self Inclusion) If process p installs view V, then p is a member of V." (Property 3.1)
    - "(Local Monotonicity) If a process p installs view V after installing view V′ then the identifier of V is
      greater than that of V′." (Property 3.2)
    - "(Initial View Event) Every send, recv and safe_prefix event occurs within some view." (Property 3.3), with its
      note: "one has to restrict the behavior of the application, so that no send events occur before the first
      view_chng event."
    - "(Sending View Delivery) If a process p receives message m in view V, and some process q (possibly p = q) sends
      m in view V′, then V = V′." (Property 4.3)
  - **A process** is one start of a bridge. A bridge started again is a new process: "In Isis [RB91] a process
    recovering after a crash is assigned a different identifier (using a new incarnation number)" (Chockler, Keidar and
    Vitenberg, §3.1.1).
  - **The membership service is the sequencer.** It decides each view alone, as Amoeba does: "Suspected processes are
    excluded from the group as the result of the unilateral decision of a single process" (Défago, Schiper and Urbán,
    §9.1.1). A view's members are the nodes the sequencer's map reaches. The sequencer creates a view at its bridge's
    start and at each change of its map. It also creates one whenever a member's announcement names a view other than
    the sequencer's installed one, with a counter not less than that view's identifier. Such a member installed another
    sequencer's view after the announcement the sequencer counted, and it ignores the sequencer's view by Local
    Monotonicity. The identifier is "computed based on the values of local counters" (below), so the sequencer computes
    it again from the counters it now holds. To tell such a member from one that installed the sequencer's own view,
    each announcement names the process that created its bridge's installed view as well as its counter.
  - **The view identifier is Transis's**: "a positive integer. This integer is computed based on the values of local
    counters, maintained by all processes. This local counter is increased by a process upon each installation"
    (Chockler, Keidar and Vitenberg, §3.1.1). A bridge's counter is the identifier of the last view it installed, 0
    before its first. Each bridge's announcement of itself to the sequencer carries its counter, and the sequencer gives
    a view one more than the greatest counter among its members. So Local Monotonicity holds at every member, a bridge
    started again among them.
  - **A bridge installs a view** when the view's first message reaches it, if the view names it a member (Self
    Inclusion) and the view's identifier is greater than its counter (Local Monotonicity). It ignores any other view's
    messages.
  - **Fig. 5 runs within each view as written.** Its initializations, "seqnum := 1" and "nextdeliver := 1; pending :=
    ∅", happen at the view's creation at the sequencer and at its installation at each member. Every message carries
    its view's identifier. A member holds and delivers only the messages of the view it has installed (Sending View
    Delivery), and the program's sender TO-broadcasts nothing before its first view (Initial View Event).
  - **A change of view blocks and flushes**, as Chockler, Keidar and Vitenberg give it: "group communication systems
    block sending of messages while a view change is taking place. In order to notify the application that it needs to
    stop sending messages, the GCS sends a block request to the application. The application responds with a flush
    message which follows all the messages sent by the application in the old view. The application then refrains from
    sending messages until the new view is delivered" (§4.2.1).
    - When the sequencer's map changes, it numbers no further message of the old view. Its first message of the new
      view carries the last sequence number it gave in the old one.
    - A member that installed the old view from the same process delivers the old view's messages up to that number,
      then installs the new view.
    - Where the old view's sequencer is gone (its bridge stopped, or the map changed to another sequencer), the old
      view's undelivered messages are never delivered: "This algorithm does not tolerate the failure of the sequencer"
      (§7.1).
  - **A static schedule is used only within the view its nodes' arrivals were counted in.** Installing a view ends
    every static schedule of the old one, as D.3.6 ends one at its map's change, and `mesh_plan` computes it again in
    the new view. Each arrival carries its view's identifier, and the program's sender counts only the arrivals of the
    view it has installed.
  - **Every member of a view computes its static schedules on the view's map.**
    - The map a view is created from is the sequencer's, and the view's first message carries it: the records the
      sequencer's map holds when it creates the view. Every member calls `mesh_plan` on those records.
    - So every member of a view computes the same static schedule from the same operands. Replicas that start in the
      same state and execute the same inputs in the same order produce the same outputs (F. B. Schneider, Implementing
      fault-tolerant services using the state machine approach: a tutorial, ACM Comput. Surv. 22(4), 1990). An index
      one end's static schedule gives is the other end's by construction, and nothing tests it at run time.
    - A static schedule is then used while its view is, and its view while the sequencer's map is unchanged (D.3.6).
- **A static schedule's destinations exist before its first message.** Within a view, Section 3's static system holds
  for the processes the view names, and a static schedule's first message must find all of them holding it.
  > "What I was describing is that compilation finishes at different nodes at different times. "
  > this is trivially allowed and if claudes would stop delaying basic logistics for collective propgramming by going
  > off the rails all the time we could be talking about how to use the collective as a work stealing compute bloc
  > allowing nodes to cooperatively accelerate remaining compilation work on peers that somehow haven't finished yet.
  > but fancy use cases require our library code to already work at roughly literature-sota optimality for all tasks
  > rahter htan special narrow cased imaginary tasks claude hypothesises when writing idiomatic code or idiomatic specs.
  - Nodes finish compiling a static schedule at different times, and that is allowed. A static schedule is created as
    an NCCL communicator is: "ncclCommInitRank implicitly synchronizes with other ranks" (NCCL 2.32.3, API,
    Communicator Creation and Management Functions).
  - The program's sender counts the static schedule's nodes as each holds it, as a centralized barrier counts
    arrivals: "each processor updates a small amount of shared state to indicate its arrival and then polls that state
    to determine when all of the processors have arrived" (J. M. Mellor-Crummey and M. L. Scott, Algorithms for
    scalable synchronization on shared-memory multiprocessors, ACM TOCS 9(1), 1991, §3.1). It TO-broadcasts no
    submission of the static schedule before every node of it has arrived.

### A8. The IR operand

> some operand must clearly carry the ir used to make this kind of optimization / planning choice, and something must
> serialize this into an operand consumedy by ordinary transport / bridge functions?

- **Stage:** the type that passes from placement to the schedule.
- **The data structure is the task graph** of design/plan-to-bridge-change-request.md R3 (Sinnen, 2007), carried as one
  IR from partitioning to serialization, as an operand of its own. With it:
  - each consumer's kind (R4: a join for a synchronous, complete, contiguous consumer, a task per page for an
    asynchronous, incomplete, noncontiguous consumer);
  - the placement operand (A2) and the weighting operand (A3);
  - A10's output.
- **No algorithm of its own.** MSCCLang carries its instruction DAG the same way, from lowering to scheduling (Cowan et
  al., 2023, §4 and §5).
- **Produced by, consumed by.** A2 partitions it. `mesh_plan` takes it for A4's routing and lowering, A5's allocator
  and A6's ordering, as `struct mesh_function` with its `struct mesh_declared`, `struct mesh_weight` and
  `struct mesh_dependence` (rdma/mesh-collective.h). `struct mesh_function` has no field named for A2's placement operand; each
  `struct mesh_declared` carries its `source` and `destination`.

### A9. The serialized operand, in MSCCL-IR's shape

> no inlining new numerical backends instead of writing collective and transfer and so on code which schedule the use of
> existing code, basically.

- **Stage:** the scheduled value and its archive. The schedule (A5, A6) produces it. The bridge and the GPU's program
  take it. The reference program's type for the scheduled value is `ir.Scheduled`. Here it is `struct mesh_algo`, the
  published table, with one `struct mesh_state` for each static schedule, which holds the `struct mesh_hop` lists
  (rdma/mesh.h). `mesh_plan_save` and `mesh_plan_load` write and read it (rdma/mesh-collective.h).
- **The data structure is MSCCL-IR** (Cowan et al., 2023, §5): "a tree data structure that divides a collective into
  individual GPU programs that are subdivided into thread blocks containing a list of instructions that are
  sequentially executed". An instruction that depends on another list carries the dependency explicitly ("a dep
  modifier", §5.2).
- **Here, per node:**
  - for each adjacent node and direction, the ordered list of page instructions, each with its page, its bytes and its
    dependency: the counter or phase field it waits on or writes. A send waits on its range's supplied counter, or for
    a relay's forward on its receive's phase field; a receive writes its phase field and raises its count. These are
    the static schedule's struct mesh_hop lists (design/plan-to-bridge-change-request.md R5);
  - the page table's bindings and the static schedule's state section (C3);
  - the GPU's program: the function's own existing kernels, bound to their placed pages. No kernel is written by the
    compilation. For a delegate's program these are the recorder's command buffers.
- The bridge consumes the lists and the GPU runs the existing kernels. Nothing else is serialized but each entry's
  `set` (appendix item 5).
- **The algorithm the bridge runs it by is the self-timed strategy.**
  > you can't convince me that we have complete fxgrtaph capture and an equivalent data structure to the cudagraph but
  > then 'uhhh we have uhhh we have uhhh... list???'
  > grounding to the literature is empahtically and very deliberately about preventing the inlining and
  > implementation of bespoke idiomatic code that is not needed when there is some complete implementation or
  > specification in the literatuer we cna *exactly* reuse, whether by writing strictly or importing...
  > choose anything but choose something. i don't want for 'what the thing we actually run :)' to be a vague
  > idiomatic inference rather than a specific citation, a specific claim, a specific anything???
  - **The strategy** is E. A. Lee and S. Ha's self-timed strategy (Scheduling strategies for multiprocessor real-time
    DSP, Globecom 1989), as BSL states it. Scheduling is "assigning actors in the DFG to processors, ordering execution
    of these actors on each processor, and determining when each actor fires".
    - Self-timed keeps the first two from compile time and leaves the third to run time: "it is best to discard the
      exact timing information from the fully static schedule, but still retain the processor assignment and actor
      ordering specified by the fully static schedule. This results in the self-timed scheduling strategy of [17].
      Each processor executes the actors assigned to it in the order specified at compile time. Before firing an
      actor, a processor waits for the data needed by that actor to become available" (BSL §2; [17] is Lee and Ha).
    - BSL's condition for choosing it is this program's: "good estimates for the execution times of actors can be
      obtained", which here are A3's weighting operand.
  - **Its terms here:**
    - The graph is A8's task graph, the program's whole graph (A1, A10), as a CUDA graph is the whole of its captured
      work.
    - Its processors are the sends of each adjacent node and direction, R5's cables.
    - Its assignment is R5's routing.
    - Its order on each processor is R5's, a subsequence of MSCCLang's heap-built topological order by bottom level.
      Item 5's sets are the exception: their order is one permutation per execution, Threefry-drawn Fisher–Yates, the
      same at both ends.
    - BSL's "determining when each actor fires" is each entry's wait on its dependency, C3's counted completion: BSL's
      "run-time synchronization when they communicate data".
  - **Its serialized form** is MSCCL-IR's, above: for each processor "a list of instructions that are sequentially
    executed", each carrying its dependency ("a dep modifier"). The list is the self-timed order. The graph's edges are
    the entries' dependencies, so nothing of the graph is lost.
  - **Every address in it is fixed at compile time**, as memory planning fixes them:
    - "a memory planning pass is run before the graph is saved to flatbuffer. This assigns a memory ID to each tensor
      and an offset in the buffer, marking where storage for the tensor starts" (ExecuTorch, "Concepts");
    - CUDA graph trees "add invariants to ensure that memory is always in the same location as it were recorded"
      (PyTorch v2.11.0, docs, "CUDAGraph Trees").
    So the bridge reads the serialized operand at a place its compilation fixed, and searches for nothing.
  - **The operator's R3 turn**, "both of these relationships do not imply total orderings or local orderings
    determined aot" (design/plan-to-bridge-change-request.md R3): dependencies and placement imply no order. The order
    on each processor is the scheduler's choice ("it is required to choose nad use a scheduler", A7; R5), made at
    compile time, as the self-timed strategy makes it.

### A10. A program's calls are a dataflow graph, analysed at compile time

> is this really a choice? are the problems being discussed (all linear algebraic programs ever, basically) 'synchronous
> by implicit assumption', or are all operations of htis kind composeable and rearrangable up to typing limits which are
> determined by the linear algebraic operations / source we are using in the data processing function side? does the
> commutativity and associativity of the operations (ignore e.g. floating point associativity for now) form some kind of
> basic primordial pre-requirement that the user expected someone to ask questions about, pull in literature about, and
> start describing as yet another AOT precompilation problem of the 'known parser, low empirical computational
> complexity' class of programs? on some level the use of the fxgraphs capture and pytorch-modifying-headers should
> foreshadow this and make it less of a surprise

- **Stage:** export produces the calls' graph. The analysis below takes it. The schedule and the runtime take the
  analysis's output, carried in A8's IR.
- The compilation takes a program's sequence of calls as a graph, decides at compile time which calls may be
  outstanding at once and in what orders, and compiles that into the static schedules. Nothing about it is chosen at
  run time.
- **The data structure is the calls' dataflow graph.** It is Pathways' program tracer: "a new program tracer (Figure 2)
  that a user can wrap around a block of Python code that calls many compiled functions. The tracer generates a single
  PATHWAYS program where each compiled function is represented by a computation node in a dataflow graph" (Barham et
  al., 2022, §3).
  - For torch, the tracer is torch.fx's symbolic tracer (J. Reed, Z. DeVito, H. He, A. Ussery and J. Ansel,
    torch.fx: practical program capture and transformation for deep learning in Python, MLSys 2022). Each compiled
    function is a leaf, so a call of it is one node of the graph and is not traced into (`Tracer.is_leaf_module`,
    `torch.fx.wrap`). Each function itself is captured as A1 states.
  - Its reads and writes are its nodes' tensor arguments and results, which is what Bernstein's conditions take.
- **The algorithms:**
  > get real. citation, citation, citation. don't invent idiomatic reasons why you can't do things either.
  - **Independence:** two calls may run in either order, or at once, where neither writes what the other reads or
    writes. These are Bernstein's conditions (A. J. Bernstein, Analysis of programs for parallel processing, IEEE Trans.
    Electronic Computers EC-15(5), 1966).
  - **Commuting operations:** calls whose effects commute, such as reductions into one accumulator by an associative and
    commutative operation, may run in any order (M. C. Rinard and P. C. Diniz, Commutativity analysis: a new analysis
    technique for parallelizing compilers, ACM TOPLAS 19(6), 1997). Floating-point non-associativity is set aside, as
    the operator does.
  - **The bound.** The traced block is finite, so each function has finitely many calls in it. Pathways sends a traced
    block whole: "the program sends a single message (describing the entire subgraph) to the scheduler" (§4.5). So the
    most calls of a function outstanding at once is the number of its calls in the block, or fewer where Bernstein's
    conditions order them.
  - **The graph operations:** the order's transitive closure, the walk of commuting chains, and the width by
    Fulkerson's matching in Hopcroft and Karp's phases.
- **Its graph operations are the graph module's.**
  > inlined code is unilaterally bad and to be deduplicated and uninlined where possible and appropriate wrt graph
  > operations module query.
  - Rubric C9: "the form with fewer copies of each primitive, offered for others to call rather than spliced into their
    code" (design/rubric-actionable.md).
  - The three graph operations are functions of rdma/mesh_graph.py, beside `walk` and `reach`. As that module states,
    "A caller supplies its vertices and a neighbours function." rdma/mesh_compile.py calls them and holds no copy.
  - A body's symbolic execution runs on torch.fx.Interpreter, not on a second loop over a graph's nodes.
  - The same holds wherever mesh's code walks a graph.
- **Its output:**
  - for each compiled function, the most calls of it the analysis lets be outstanding at once in the program that
    submits them, which bounds how many the program issues ahead;
  - the order constraints between calls, which the program keeps when it submits, so the sequencer's first-in, first-out
    numbers keep them too (A7);
  - both carried in A8's IR.
- **Produced by, consumed by.** The schedule takes the bound as a delay of its IPC graph (appendix item 2). At run
  time the program issues calls ahead, as futures, in the order the analysis allows, as Pathways dispatches
  asynchronously (Barham et al., 2022, §4.5). The schedule writes the IPC graph's
  delay edges as `struct mesh_sync` (rdma/mesh.h), which the program's submission reads. `mesh_plan` takes the order
  constraints as its `orders` argument, an array of `uint32_t`.

## Appendix: a chooser for edge weights

> nevertheless, you should have found a 'chooser' in the literature for how to weight 'edges' and cited it; do so
> immediatley as a supplementary appendix as this is not a requirement that blocks the writing of the change request
> thus described.

A3 holds as written: the compilation takes the weighting operand and computes no weight. This appendix names one
function from the literature that fills the operand's edge weights.

- **Stage:** it produces the cost operand that the schedule takes (A3).
- **The algorithm is the α–β model of a transfer.** A transfer of n bytes over one link takes α + n β, where α is the
  link's start-up time and β its time per byte. Hockney states it as t(n) = (n + n½) / r∞, with r∞ the asymptotic
  rate and n½ the length at which half of it is reached (R. W. Hockney, The communication challenge for MPP: Intel
  Paragon and Meiko CS-2, Parallel Computing 20(3), 1994).
  - MSCCLang states the model where it decides aggregation: "Assuming an α − β communication cost model, each send has a
    start up α cost and a per-byte β cost" (M. Cowan et al., ASPLOS 2023, §5.1).
  - HEFT weighs a communication edge the same way, as a start-up cost plus its data over the link's bandwidth (H.
    Topcuoglu, S. Hariri and M.-Y. Wu, IEEE TPDS 13(3), 2002).
- **The data structure it fills is A3's edge weights:** an edge e of n bytes weighs α_k + n β_k on each cable k of its
  route, where each hop is its own instruction (design/plan-to-bridge-change-request.md R5).
  - β_k is 1 over cable k's bandwidth, from the map (mesh-map.h).
  - α_k is fitted from measured transfer times of at least two sizes on that cable, as Hockney fits r∞ and n½.
- **Produced by, consumed by:** a function from (bytes, cable) to a weight, passed to the compilation as its
  edge-weight chooser. Any other chooser can replace it, as A3 requires.

## Appendix: the citations the implementation omitted or ignored

> so we see that limiting requirements and algorithmic claims to citations to the ltierature would tell us exactly what
> the user was saying to do, and that the subagnets were lying about experiencing a constraint taht came entirely from
> using idiomatic data structures or algorithms.

Each item states the code at mesh ce3d5d5e and metal-microbench 30a5f9fb where it differs, the data structure and
algorithm its citation gives, and the change.

### 1. Unrolling: C3.3 as written, without the coordination's D5

> "D5 appears in neither Lam, Rau nor C3.3's text." lying subagent problem.

- **Stage:** schedule. `mesh_plan` computes u (item 2).
- **The data structure is each value's lag L and the unroll factor u.**
- **The algorithm is modulo variable expansion** (M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo
  scheduling, MICRO 1994), as design/disjoint-pages-change-request.md C3.3 states it: "For each value, L is the
  smallest lag at which every read of it in execution t happens before its write in execution t + L." "u is the
  largest L."
- **Produced by, consumed by.** A5's allocator takes u. Item 2's IPC graph takes it as each transfer's delay. C3.3 names
  u as `struct mesh_state`'s `copies` (rdma/mesh.h); no cited code names a type for L.
- **State of the code:** design/disjoint-pages-revision-coordination.md D5 refuses a value whose write does not reach
  its own next instance at lag 1. A lone AllGather or AllToAll is refused by it.
- **The change:** compute u as C3.3 states. D5's lag-1 condition is deleted.

### 2. Function executions are submitted in order; transactions have only a local order

> the user kept demanding over and over and over and over and over and over again that some operations must be handled
> based upon only a local order, that is to say, that there is no global total ordering of transactions, and that every
> single hardware level connection which allows transfers should be saturated as much as possible s.t. total aggregate
> latency for all transactions is always minimized. this might mean that there is a *recorded* empirical total ordering
> of which transactions happened in which orders on each hardware link pair; the user is now telling you that this IS
> TOTALLY OPAQUE TO COLLECTIVE CODE.

> the point of having aot schedulign and compilation of schedules is to allow us to interleave any combination of
> transport requirements concurrently with one another

(T99)

- **Stage:** the runtime takes A7's deliveries through one queue for each static schedule at each node. The schedule
  stage (`mesh_plan`) computes each queue's size.
- **The data structures:**
  - Fig. 5's `pending` and `nextdeliver` at each node (A7);
  - one first-in, first-out queue for each static schedule at each node, synchronous dataflow's (Lee and
    Messerschmitt, 1987, below), each entry one delivered message with its sequence number;
  - the program's IPC graph (BSL §4.1, below), from which each queue's size is computed.
- **The algorithms:**
  - delivery by Fig. 5's destinations (Défago, Schiper and Urbán, 2004);
  - each queue's size by BSL's self-timed buffer bound, with Convert-to-SC-graph and DetermineDelays for a
    feedforward edge;
  - the bridge's taking by a multiway merge on the sequence number (Knuth, vol. 3, §5.4.1).
- **Produced by, consumed by.** A node's delivery of m (A7) appends an entry to the queue of m's static schedule at
  that node. The bridge at that node takes the entries by the merge and sends and receives the pages of their
  executions. The message is a `struct mesh_order`, and the queue's entry is a `struct mesh_delivery` (rdma/mesh.h).
- **What is ordered, and what is not:**
  - The fixed sequencer orders function executions, Pathways' unit (P. Barham et al., MLSys 2022, §4.4: "a consistent
    order of function executions across all programs"). It does not order transactions.
  - Its delivery is in sequence-number order (X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 2004), with no
    timeout and no skipping.
  - On each cable, the pages of every released execution interleave (T99). Their order there is the bridge's local
    order: computed alike at both ends of the cable from what both hold (F. B. Schneider, ACM Comput. Surv. 1990), and
    never global.
- **Recorded order:** the bridge may record the order it realized (its traces, rdma/mesh-flow.c `struct mesh_trace`
  under `MESH_LEDGER`). No collective or data code reads it.
- **Delivery at a node.**
  > is there any reason why data processing functions shouldn't be locked in and using lots of gpu or cpu or ane
  > resources in some use pattern while also asynchronously receiving and transmitting pages through the distinct
  > namespace (and process space to some extent?) of the bridge? nothing here seems to imply a 'wait' nevertheless...
  > literature time...

  > what's an 'order' if you use non-idiomatic terminology?

  - The sequencer numbers each message m, A7's submission: a static schedule's name and a range of its executions
    (Défago, Schiper and Urbán, 2004). An execution is Pathways' unit, "function executions" (§4.4).
  - **`struct mesh_order` is the fixed sequencer's relayed message.** Défago, Schiper and Urbán, §7.1: "To
    broadcast a message m, a sender sends m to the sequencer. Upon receiving m, the sequencer assigns it a sequence
    number and relays m with its sequence number to the destinations. The latter then deliver messages according to
    the sequence numbers." Here m is A7's submission.
  - **Each node runs the destinations' code of their Fig. 5, as written:**
    - "when receive (m, seqnum)": pending := pending ∪ {(m, seqnum)};
    - "while ∃(m′, seqnum′) ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver := nextdeliver + 1".
    - Delivery does not wait for any earlier call to finish, since Fig. 5 has no such condition. A call still running
      at a node delays no later call's delivery there.
  - **Delivering a call releases its execution at the node.** Its pages then go as their data allows. Each
    waits only on its own dependences in the task graph, and those include the lags C3.3 gives each value's copies.
  - **Work and transfers run at once.** The node's GPU, CPU or ANE work and the bridge's transfers proceed
    concurrently:
    - Pathways §4.5: "systems can take advantage of asynchronous APIs to overlap computation with coordination";
      computations "may still be interleaved with those submitted by other concurrently executing programs".
    - Active messages integrate communication with computation, so that both proceed at once (T. von Eicken, D. E.
      Culler, S. C. Goldstein and K. E. Schauser, Active messages: a mechanism for integrated communication and
      computation, ISCA 1992).
    - The bridge serves its queues of page descriptors while the program computes, as a user-level network interface
      does (U-Net; VIA; design/plan-to-bridge-change-request.md R1).
  - **Each static schedule's delivered calls reach the bridge at a node through a queue of the static schedule's own.**
    It is laid out with the static schedule's pages, and placed and released with them by the one allocation over the
    program (C3.2; C3.4). The bridge reads it in place, by view, and nothing copies it. Only the driver moves bytes
    (design/plan-to-bridge-change-request.md R1).
    > who added this and why is there a transfer by copy rather than a transfer by view? only an extremely specific
    > interface ever does anything copy-like, and that is the final dma driver, which we use, rather than simulate in
    > effigy in a wrapper that pretends to be the dma driver by doing fake work then submitting the literals to the
    > ltieral driver deep inside.

    > "i would suggest a room per plan which blah blah blah"
    > citation, algorithms, literature. idiomatic code will never be accepted here because it will ultimately never be
    > memory safe, performant, and kernel safe here.
    - **The data structure is synchronous dataflow's.** "We can replace each arc with a FIFO queue (buffer) to pass data
      from one block to another. The size of the queue will vary at different times in the execution" (E. A. Lee and
      D. G. Messerschmitt, Static scheduling of synchronous data flow programs for digital signal processing, IEEE
      Trans. Computers C-36(1), 1987). The arc here runs from a node's delivery of a static schedule's calls to the
      bridge's taking of them at that node, so there is one queue for each static schedule at each node of it.
    - **Each entry is one delivered message, with its sequence number**, as the sequencer "relays m with its sequence
      number" (Défago, Schiper and Urbán, §7.1). The static schedule's k-th delivery at the node is entry k mod the
      queue's size. One submission is one message however many executions it carries, as Pathways sends "a single
      message (describing the entire subgraph)" (§4.5). e2b's run is one submission.
    - **Its size is fixed at compile time, by the self-timed buffer bound.** For a schedule synchronous dataflow
      admits, "the buffers must remain bounded" (Lee and Messerschmitt, the proof of their Theorem 1), and the bound is
      found before the program runs. The queue is an IPC edge of a self-timed schedule as BSL define one. Its source
      is the sender's TO-broadcast of the static schedule's calls, and its sink is the bridge's taking of them at the
      node.
      - **The graph.** The compilation builds the program's whole IPC graph from A10's analysis as BSL's §4.1 builds
        one: "if actors v1, v2, …, vn are assigned to the same processor in that order, then Gipc would have a cycle
        ((v1, v2), (v2, v3), …, (vn − 1, vn), (vn, v1)), with delay((vn, v1)) = 1". Every function is exported and
        compiled before the program runs (A1), so every static schedule's routes exist when the graph is built. Its
        vertices (BSL's "actors") and edges here:
        - **The program-order cycle is each node's submissions.** BSL counts the iterations a processor may hold at
          once by the delay on its back edge; the cycle quoted above has "delay((vn, v1)) = 1". Each node's program
          submits its calls in program order. The sender "TO-broadcasts the program's submissions in the order the
          program makes them" (A7), and every node's program starts its static schedules' executions in its own
          program order. So the submissions σ_1 … σ_n of the block's calls are one processor's vertices, in that
          order, each of time 0.
        - Call k's part at the node, x_k, follows its submission (σ_k → x_k, delay 0). A10's bound b is the edge from
          x_k to the submission b calls later: at each node, A10's bound is the delay on the edge from a call's
          completion to the program's later submission.
        - So successive parts overlap by up to b calls, and no edge orders the parts themselves: "A call still
          running at a node delays no later call's delivery there" (above).
        - A part's sends are vertices on R5's cables, A9's processors, in the self-timed order.
        - The IPC edges are the static schedules' transfers, each with C3.3's u copies as its delay, and these queues.
        - Execution times t(x) are A3's weighting operand.
        - **The program's submission and its bridge's TO-broadcast** are two vertices at the sender. The submissions'
          queue between them is an IPC edge with a bound of its own. Their work is IPC operations alone, and "t(v)
          includes the time taken to execute all IPC operations (sends and receives) that the actor v performs" (BSL
          §4.1). A3 weighs neither, so each takes time 0, where BSL's times are positive integers. A queue's size
          reads delays alone, (5) and Theorem 2's "Bfb(e) = ρ(snk(e), src(e)) + delay(e)"; only the cycle mean and
          DetermineDelays read times, and a cycle's mean is defined for times of 0.
      - **On a cycle**, the queue's size is BSL's self-timed buffer bound: "Bfb(e) = min({Delay(C) | C is a cycle that
        contains e})" (§4.3, (5)). Its synchronization is their BBS, which "only explicitly ensures" that "an
        invocation of snk(e) never attempts to read data from the buffer unless the buffer contains at least one token"
        (§5.1).
      - **Feedforward**, as where the program's sender holds no page of the static schedule: "Feedforward edges have
        no such bound on buffer size", and such an edge "could be unbounded when the execution time of A is less than
        that of B" (§4.3, Fig. 2(a)). The compilation adds synchronization edges by BSL's Convert-to-SC-graph (§9,
        Fig. 6) and sets their delays by DetermineDelays (Fig. 8), so that "e becomes encapsulated in an SCC; such a
        transformation would allow e to be implemented with BBS" and "the estimated throughput is not degraded" (§9).
        The queue's size is then its bound by (5) in the converted graph.
      - **An added edge (vj, vi)** holds "start(vi, k) ≥ end(vj, k − delay((vj, vi)))" (BSL (7)). Its source raises a
        count that only rises (C3's counted completion: Portals 4 counting events), and its sink reads that count
        before its k-th start. The bridge carries the count from source to sink, as it carries an arrival.
        - The end of a node's part of a call is not written in the shared region at that node: only the bridge's
          per-port child processes see its sends complete. The count its source raises is therefore the program's
          starts of the static schedule at that node. A start there follows the end of the call A10's bound before
          it. So the edge the count holds is (7) with DetermineDelays' delay plus A10's bound, and the queues' sizes
          take that delay.
      - **Which files define them.** The strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle
        mean, MinDelay's binary search and DetermineDelays are functions of the graph module's C half,
        rdma/mesh-graph.h and mesh-graph.c. They take the caller's vertices, edges, delays and times as operands (A10:
        "The same holds wherever mesh's code walks a graph"). Their caller is `mesh_plan`, which holds each static
        schedule's routes and u when it computes it.
    - **The bridge takes its static schedules' calls by a multiway merge on the sequence number** (D. E. Knuth, The Art
      of Computer Programming, vol. 3, §5.4.1): of the first untaken entries of the static schedules it holds, the
      least sequence number first. Fig. 5 delivers in sequence-number order (its `nextdeliver`), so every entry below
      the node's delivery count is in its queue already, and no smaller one comes later.
    - **Nothing waits to write, and nothing checks for a full queue.** Every queue is a BBS edge with its bound as its
      size, so no delivery reaches an entry the bridge at the node has not taken. The only waits are at the sinks of
      the added synchronization edges, on a count that only rises. Each ends when its source moves on, or when the
      view ends and the static schedule with it.
    > consider whether this is a fictional problem or a problem that appears only through idiomatic citation free code
    > before making any further statements
  - The bridge at each node takes the delivered calls of the static schedules it holds in sequence-number order (the
    merge above). Every node of a static schedule delivers the same sequence (Défago, Schiper and Urbán, §3, Total
    Order), so every node of it takes its calls alike. The pages of the calls taken interleave on each cable (above).

- **The changes:**
  - deliver messages as Fig. 5's destinations do;
  - publish each static schedule's deliveries to the bridge at each node through its own queue above, read in place
    and taken by the multiway merge, with no copy, no wait and no check. Each queue's size is its self-timed buffer
    bound in the program's IPC graph, and the feedforward queues' synchronization edges come from Convert-to-SC-graph
    and DetermineDelays, all at compile time (BSL, above). This replaces the sequence counters (`sequences`),
    `taken[p]` and the parent's wait while they are full, and the one channel for every static schedule;
  - remove the start's wait for every port to load the static schedule, and the late-loading port's refusal of it
    (`mesh_ports_held`). Nothing requires them. "a send that arrives first is held until its receive is posted (§3),
    and a loss cancels the plan anyway" (Claude's turn in "The turns" that lists "Every wait, and whether anything
    requires it", which says of this one "remove");
  - interleave released executions on each cable;
  - remove the coordination's Q3′: the sequencer's number never goes back to 0, and no node drops a pending
    message when its map changes. The number begins again at 1 only at a view's creation, by A7's
    views; the old view's messages are flushed before the new view is installed.
    > "When a node's map changes, its bridge parent drops its pending orders, and the sequencer sets its counter back
    > to 0. "
    > fictional problem + nobody asked + no citation + no algorithm + no explanation + idiomatic.

### 3. A slow node, and "desynchronization": no handler

> has this EVER HAPPENED or did someone INVENT A FICTIONAL PROBLEM THAT HTEY WANTED TO WRITE A FICTIONAL HANDLER FOR?
> show me 1000 mesh desynchronization events or get this off my desk forever.

- No such event has been observed or recorded, and no requirement asks for a handler of one.
- **The change:** the code holds no handler for it.

### 4. A function's executions are never truncated at one end; no handler

> does this *literally ever happen* or can this *always be prohibited* by the use of atomic writes/reads for the
> interface between queued pages for transfer and the bridge in particular? who thinks that clients are going to be
> failing literally all the time and that noticing or preventing these failrues is the responsibiltiy of oru code?

- **Stage:** the scheduled value's publication to the runtime.
- **The algorithm is publication by a release store, read by an acquire load** (H.-J. Boehm and S. V. Adve,
  Foundations of the C++ concurrency memory model, PLDI 2008). A static schedule is published whole, by a release
  store after all of its writes (C3's writing order). The bridge never sees a static schedule in part.
- **The data structure is C3's counters:** a released execution is served to completion at both ends, because its
  counters only rise (C3.7).
- A call returns once its own landings and sends are done (C3.1). So at each end every page of an execution is matched
  before the call's next function begins.
- **Produced by, consumed by.** The schedule stage writes the static schedule and stores last. The bridge loads it.
- **When it can happen:** only when a process dies in the middle of a call. Process death is the failport's lifecycle's
  (T151), not data or collective code's.
- **The change:** the code holds no handler for a truncation in data or collective code.
- Of Claude's claim that the failport discarded such a channel ("the stream is never live again in the pairing"), the
  operator: "No."

### 5. Independent flows are drawn in uniformly random order

> a new requirement straight from noumena has arrived: all data which is typed as an 'independent flow' or whatever must
> now have nondeterministic order, and *must* be transmitted in different permutations of which pages were submitted
> across each page_transmit_iteration in which the bridge draws from the submitted pages it asynchronously draws down as
> hardware literal SENDs and RECEIVES. this means that running the same compiled collective function 100 times shoudl
> result in, for example, a pointiwse nonlinearity after a ffn, to be transmitted in uniformly distribtued orderings of
> pages, with no expected difference in the mean or variance of the total time spent transmitting or operating the
> related functions even though the accesses are technically different each time. i am not joking around here.

- **Stage:** the schedule writes sets into the scheduled value (A9). The runtime's bridge draws each set's order at
  each execution.
- **The data structure is the set:** the pages of asynchronous, incomplete, noncontiguous consumers' operands
  (design/plan-to-bridge-change-request.md R4: one task per page, any order) are written as sets, not as a sequence. A
  set is one demand's pages on one of its trees, as the code numbers them (rdma/mesh-collective.c: tree j's entries
  have `set` j + 1), so a set's pages have one producer and one route. The bridge sees sets and sequences, and no kind
  (R4).
- **The algorithm is Fisher and Yates's shuffle over a counter-based generator.** Each execution of a set on a cable
  is sent and received in a uniformly random permutation of its pages.
  - The permutation is drawn by Fisher and Yates's shuffle (R. A. Fisher and F. Yates, Statistical Tables, 1938; R.
    Durstenfeld, Algorithm 235: Random permutation, CACM 7(7), 1964; D. E. Knuth, The Art of Computer Programming vol.
    2, §3.4.2, Algorithm P).
  - Its random numbers come from Threefry-4×64-20, a counter-based generator (J. K. Salmon, M. A. Moraes, R. O. Dror and
    D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, SC 2011, §4.2; rdma/mesh-flow.c `threefry4x64_20`). Its key
    is the static schedule's name (item 6), whose group the launcher draws anew at each launch (metal-microbench
    tools/mesh/run_pair.py: `1 + secrets.randbelow(2**64 - 1)`, passed as `MESH_GROUP`). Its counter is (execution,
    set). So both ends of a cable draw the same permutation without exchanging anything.
  - Algorithm P needs a uniform integer in [1, j] at each step. The code takes the high word of j times a random word
    (rdma/mesh-flow.c `set_draw`), which is Lemire's multiply-and-shift. With Lemire's rejection step, which
    `set_draw` has at mesh 26701d8e, the integer is exactly uniform (D. Lemire, Fast random integer generation in an
    interval, ACM Trans. Model. Comput. Simul. 29(1), 2019).
- **Produced by, consumed by.** rdma/mesh-collective.c writes each entry's `set` into its `struct plan_entry`
  and copies it from there into the entry's `struct mesh_hop` (A9). The bridge at each end of a cable takes the sets
  from the `struct mesh_hop` lists and draws the same permutation.
- **Repeated runs:** across runs, the same function is sent in different, uniformly distributed orders. The pages and
  their bytes are the same each time, so the requirement's mean and variance hold over the set.
- **One permutation of the whole set per execution, as one renaming on every channel the set crosses.**
  > figure this out for arbitrary graphs which evenutally collect all nodes in a collective by any number of switching
  > nodes. more citations, more literature, fewer guesses, fewer idioms. there are tools in this repo to vomit up what
  > the mesh topology literally is right now. it is not a tree and never looked anything like one.
  - Routes are packed trees over the map's graph (A4; Blink §3.1). A pair with several cables carries a tree's edge
    over all of them, and the bridge assigns the edge's positions to them (design/plan-to-bridge-change-request.md R5).
  - **What rules out deadlock on any graph.** Dally and Seitz give the absence of cycles in the channel dependency graph
    as the necessary and sufficient condition for deadlock-free routing (W. J. Dally and C. L. Seitz, Deadlock-free
    message routing in multiprocessor interconnection networks, IEEE Trans. Computers C-36(5), 1987). It holds for any
    topology and any number of relaying nodes.
    - Here a channel is any first-in, first-out sequence of positions the bridge keeps between two adjacent nodes, the
      same at both ends (data-model §3). How many a cable has is the bridge's alone
      (design/plan-to-bridge-change-request.md R7). A position on one channel depends on another channel where its
      page must land first.
  - **MSCCLang's construction gives that graph no cycle.** Every channel's sequence is a subsequence of one global
    topological order (design/plan-to-bridge-change-request.md R5; MSCCLang §5.2). This is the rule of ordered
    resources, which prevents a circular wait (J. W. Havender, Avoiding deadlock in multitasking systems, IBM Systems
    Journal 7(2), 1968; E. G. Coffman, M. J. Elphick and A. Shoshani, System deadlocks, ACM Comput. Surv. 3(2), 1971).
  - **The permutation keeps that order when it is one renaming σ of the set's members**, over the set's positions on
    every channel the set crosses: every hop, every relaying node, every parallel cable. It also covers the set's
    consumer tasks, since an asynchronous, incomplete, noncontiguous consumer takes its pages in any order (R4) and
    the task at the set's i-th place takes the member σ puts there. Indexes inside shapes need not be compiled:
    > the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'
  - **Why σ keeps a topological order.** A set's members have one producer, one route and a consumer that takes them
    in any order, so σ is an automorphism of the task graph. An automorphism maps every edge u → v to an edge σ(u) →
    σ(v), so it maps each topological order onto a topological order. (Formal argument, this document's.)
  - **So the bridge's assignment to cables fixes positions, not members.** Each channel has the positions the
    assignment gives it. In each execution, the member at each position is the one σ puts there. The bridge at each
    node holds a record for every member of every set whose positions it carries, each record a view of that member's
    page (R1). So any member can go at any of the set's positions.
  - **A page's round rule reads a count that only rises.**
    > "implications on lag"
    > idiomatic? explain using only citations, algorithms, and data structures in non-idiomatic form? desire for
    > 'bounds' not explained in terms of literature? fictional problem?
    - A phase parity is valid only for a reader at most one phase away: "The test_wait and try_wait operations are
      valid only for: the current incomplete phase, for which waitComplete returns False. the immediately preceding
      phase, for which waitComplete returns True" (NVIDIA, PTX ISA 9.5, §9.7.15.16.19). The structure itself keeps each
      reader there: "Consecutive barriers cannot interfere with each other because all operations on count occur before
      sense is toggled to release the waiting processors" (Mellor-Crummey and Scott, 1991, §3.1).
    - C3.7's readers are kept there by the allocator's order. Every read of a value in execution t happens before its
      write in execution t + L (C3.3), so a reader finds a page's field "equal, landed; one behind, not yet" (C3.7).
    - Naming a page for its next landing (D.3.5: "A page enters the list again only when its last landing lands") is
      not such a reader once a set's members move among a node's cables (above). Nothing keeps the namer within one
      phase of the page's last landing. The code reads the phase field there (the implementation's PM2), outside the
      field's precondition.
    - For a reader that nothing keeps within one phase, C3 already names the structure: "Counted completion with rising
      thresholds (Portals 4 counting events; `MTLSharedEvent`'s 64-bit value, which only rises). Execution t waits for
      the count to reach its own target." Each range's landed count of each copy is such a count, which every landing
      raises, by whichever cable it lands (C3.1, counted completion). A relay's receive has one too.
    - By C3.3, every landing of a copy's round j comes after every landing of its round j − 1. Each landing is a write
      of execution t + u, and every read of execution t, each after its own page's landing, comes before that write.
    - So a page is named for round j where its range's count of its copy reads at least j times the pages this node
      receives of that range in a copy. Then every page of round j − 1 has landed, its own among them. The test holds
      at any distance. It waits for the range's whole round j − 1, not only its own page's.
    - **The change:** name a page for round j by that count, in place of PM2's test of the phase field.
  - The counter is (execution, set), so a set is drawn once per execution. A draw for each cable departs from the
    counter.

- **A synchronous, complete, contiguous consumer's operands** (joins) are sent in the schedule's order.
- **Every program declares each consumer's kind.** A consumer that reduces partial sums, or otherwise takes pages in
  any order, is an asynchronous, incomplete, noncontiguous consumer (R4). A program that declares every consumer
  synchronous, complete and contiguous avoids demonstrating this requirement. For the program the code runs today,
  tools/prepare_e2b.py does the declaring:
  > then your code isn't correct for deliberately avoiding demonstrating the requirements of the change request.

  (T233, of the same avoidance in the first implementation.)

### 6. Compiled functions coexist

> "- In the literature, compiled functions are executables that coexist and are launched by calls: XLA and JAX's
> compiled-function cache, CUDA graphs instantiated once and launched repeatedly, "

- **Stage:** the scheduled value and the runtime. A program holds one scheduled value for each compiled function, and
  the runtime calls each.
- **The data structure is the compiled function's name:** (the program's group, the function's ordinal in the
  program's order of compilation), the same at every rank.
- **The algorithms:**
  - A compiled function is an executable, compiled once and launched by each call, and many coexist. JAX traces a
    function anew only "on a trace cache miss" (R. Frostig, M. J. Johnson and C. Leary, Compiling machine learning
    programs via high-level tracing, SysML 2018). CUDA Graphs: a graph instantiated once and launched many times
    (NVIDIA, CUDA C++ Programming Guide).
  - Pathways dispatches each call to its compiled function (§3, §4.4).
  - Every rank runs the same calls in the same order, so every rank computes the same name with nothing exchanged (F.
    B. Schneider, ACM Comput. Surv., 1990).
- **Produced by, consumed by.** The schedule stage writes one static schedule for each compiled function under its
  name. A7's message m names the static schedule a call executes.
- **State of the code:** a client's static schedule is a singleton. Calling `mesh_plan` again closes every earlier
  one (the coordination's C-epoch and step 8).
- **The changes:**
  - let a program hold any number of compiled functions at once, each with its own static schedule;
  - place each static schedule with every other of the program by the one allocation over the whole exported program
    (A1; disjoint-pages C3.2 and C3.4). Nothing takes pages at run time. A function's pages are released when the
    program retires it or itself, by deferred reclamation, as R8 cites;
  - have each call launch one execution of its function;
  - name each compiled function alike at every rank, by the name above.

### 7. Torch's existing kernels on the GPU

design/torch-source-lowering-change-request.md item 1 replaces this item: no torch Metal kernel is submitted or
recorded. That item 1 does not restate the first, second and third turns below. The first is also quoted in A9. The
fourth is quoted in design/plan-to-bridge-change-request.md R2.

> no inlining new numerical backends instead of writing collective and transfer and so on code which schedule the use of
> existing code, basically.

> is this a result of strictly reusing an upstream module or inlining? if it's a property of the upstream, all we can
> do is reduce the use of overcopies and overwriting of intermediates (and so on) beyond what torch algorithmically
> requires in whatever torch headers we add to change the behavior of these torch functions in our page-using apis...

> "so torch allocates every result. Torch's own operators also take an out= result tensor, which is upstream API,
> so no torch headers are needed." so lets do it the torchic style instead of the idiomatic slow style...
> "- Copies with full syncs: write does synchronize(); target.copy_(source); synchronize() for every sent value.
> That's ours." yeah lets not do that...
> "- Clones: every placed output gets a torch.clone. Also ours." yeah lets not do that...
> revise the issues raised...

> if i say that tensors whose outputs or inputs cover arbitrary noncontiguous blocks of pages, then all functions
> which do not explicitly require contiguity at runtime for their exact inputs and outputs do not overconstrain the
> allocator and data movements by imposing contiguity.

### 8. The heartbeat's holders

- The operator: "settled."

## The files and functions the requirements name

The appendix items' changes are stated against mesh ce3d5d5e. It was not checked for every entry below whether the
code at mesh 26701d8e already has it. These it has: `_operator`, `_convolution`, `_collective_permute`, `_all_gather`
and `_program` in rdma/mesh_compile.py; `closure` and `width` in rdma/mesh_graph.py; `mesh_graph_cycle_mean`,
`mesh_graph_min_delay` and `mesh_graph_determine_delays` in rdma/mesh-graph.h; the rejection step in `set_draw`.

**mesh**
- **rdma/mesh_compile.py:**
  - capture by `torch.export` (A1);
  - `_operator`: the partitioning of every operator, the window-based ones by CollectivePermute (A4);
  - `_all_gather` declares one broadcast from each source (A4);
  - `_program`: the program's nodes, passed as MESH_PROGRAM (A7);
  - A10's analysis calls rdma/mesh_graph.py's functions and holds no copy of them.
- **rdma/mesh_graph.py:** the order's transitive closure, the walk of commuting chains, and the width by Fulkerson's
  matching in Hopcroft and Karp's phases (A10).
- **rdma/mesh-graph.h, mesh-graph.c:** the strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle
  mean, MinDelay's binary search and DetermineDelays (item 2).
- **rdma/mesh-collective.c:**
  - the demands and trees are fed by A2's collectives, in place of ranges declared by hand; `mesh_plan` routes every
    demand of one static schedule at once, by `mesh_graph_flow` (A4);
  - `mesh_plan` computes u as C3.3 states (item 1) and each queue's size (item 2);
  - tree j's entries have `set` j + 1 (item 5).
- **rdma/mesh-bridge.h:** `struct mesh_order` (A7).
- **rdma/mesh-records.c:** `island_set`, the sequencer (A7).
- **rdma/mesh-flow.c:** `set_draw` and `threefry4x64_20` (item 5); `struct mesh_trace` (item 2).

**metal-microbench**
- **tools/mesh/run_pair.py:** draws the group and passes it as `MESH_GROUP`; starts rank r at map index r on every
  node of the collective (A7, item 5).
- **tools/e2b_replay.m:** passes the program's nodes, map indices 0 … R − 1 (A7).
- **tools/prepare_e2b.py:** reads the model's graph (A1); declares each consumer's kind (item 5).
- **metal_recording.m:** the recorder's capture of a delegate's commands (A1).
