# Change request: the ahead-of-time compilation

This change request states what the ahead-of-time compilation must do: what it consumes, the passes it runs, the IR it
carries and the schedule it serializes. design/plan-to-bridge-change-request.md states what the plan gives the bridge.
This document states how a program becomes that plan. Each requirement quotes the operator, then names its method: what
the method is, and where it comes from in the literature. The method is used as its source states it.

The compilation consumes operands that choose what it does not choose itself: where each part of a function runs (A2)
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

The operator set this form, and it is followed here:

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

So:
- **The body** is the turns, verbatim.
- **Claude's proposal** follows the turns, in a section labelled Claude's. Each of its clauses is one of three things:
  the operator's words, a cited source's own statement ("what is x and where does x come from in the literature"), or a
  fact of the code.
- **The document is the whole instruction to an implementer.** A brief adds nothing to it.
- **Where the document is silent, the implementer takes the minimal decision that follows from the document and the
  code, and records it as its own choice, with its reason, apart from the document's requirements.** Where no decision
  follows, it stops at that point and reports it.
  > "take the minimal decision … and record it" broadly fine if you really think about it especially if agents are
  > strongly segmenting 'i made an apparnetly required or apparently free choice... and i'm *documenting* it as my
  > choice rather than preordained fate': this is more honest and makes revision and acceptance of corresponding code,
  > measurements, and artifacts simpler in expectation.


## The turns

The turns are held once, in design/plan-to-bridge-change-request.md, "The turns", after
design/disjoint-pages-change-request.md D.1. They are this document's body.

## Claude's proposal

This section and the appendices are Claude's (T178). The operator accepted an earlier text of the appendix "the
citations the implementation omitted or ignored": "\"1->8\" acceptable as written here; launch revisions". Its items 2,
5 and 6 were revised after it, on the operator's later turns. Each clause is the operator's words, a cited source's own
statement, or a fact of the code.

### A1. The input is each program's captured dataflow graph, with shapes and types

> the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'

> Our PyTorch code can already record a whole function as a graph of steps before it runs. The graph shows which results
> each step needs.

(data-model §6)

- The compilation must take each function as a dataflow graph of operations whose tensors have their shapes and element
  types. Nothing at run time changes a shape.
- **Method:**
  - For a torch function: TorchDynamo's capture as an FX graph, handed to a `torch.compile` backend (J. Ansel et al.,
    PyTorch 2: faster machine learning through dynamic Python bytecode transformation and graph compilation, ASPLOS
    2024).
  - (Corrected 2026-11-05.) **For a torch program: `torch.export`, not a `torch.compile` backend.**
    > is it idiomatic claudelish trying to imply a just in time or online problem when the user sided documentation
    > emphasizes mayn constraints including saying in as many semantic frames as possible 'the bridge and mesh
    > collectives do not support dynamic or eager operations conmingled with compiled operations, and we never run
    > non-compiled operations ever, damn the consequences'?
    > "compiling each torch function at its first call (BS-3). " has anyone asked for this beidse claudes trying to
    > slowroll, procrastinate, delay the project, or otherwise tamper without benefit or cause?
    - Nobody asked for capture at a function's first call. The text above chose `torch.compile`'s backend with no
      requirement and no citation for the choice. The operator's turns ask the opposite: "compile the next transaction
      aot based on literally any relevant information instead of doing a lazy jit syncing stalling waiting pipeline
      choking 'ask'" (design/plan-to-bridge-change-request.md, the turns), and torch's work "should be AOT and produce
      serialized objects we cna resume without needing to repeatedly re-compile anything" (2026-10-28).
    - Torch's documentation states the difference. "`torch.compile` is a JIT compiler"; "When `torch.compile` runs
      into an untraceable part of a model, it will "graph break" and fall back to running the program in the eager
      Python runtime. In comparison, `torch.export` aims to get a full graph representation of a PyTorch model, so it
      will error out when something untraceable is reached. Since `torch.export` produces a full graph disjoint from
      any Python features or runtime, this graph can then be saved, loaded, and run in different environments and
      languages" (PyTorch v2.11.0, docs/source/user_guide/torch_compiler/export.md, "JIT vs. AOT" and "Partial vs. Full
      Graph Capture"). `export` "produces a traced graph representing only the Tensor computation of the function in
      an Ahead-of-Time (AOT) fashion, which can subsequently be executed with different inputs or serialized" (torch
      2.14.0, torch/export/__init__.py).
    - So every function of a torch program is captured and compiled before the program runs, the same at every rank,
      and nothing runs eagerly between its compiled calls. The program's calls between functions are A10's graph.
    - mesh used it before: `torch.export.export(module, tuple(arguments), strict=False)` in
      rdma/torch-mesh/torch_mesh/engines.py, from 2026-10-26 until 1f794b8f removed it with the rest of that frontend.
      Its last capture before that deletion, 1f794b8f's compiler.py, traced with `make_fx` over fake tensors.
  - For a program run through another framework's delegate: the recorder's capture of the delegate's commands
    (metal_recording.m), as now. For the program the code runs today on LiteRT, tools/prepare_e2b.py reads the model's
    graph.

### A2. The placement operand: which nodes hold which sections

> which nodes are chosen to have which sections of a parallelizable task at all??? this is barely descriebd at all,
> ebcause the problem of optimizing or choosing this modeling task is *evne more deferred* than the previous topic's
> urgent deferral of how we choose operands. we simply need to consume operands which choose these placements, letting
> us define and implement an optimizing chooser function *later* in a manner which is totally decoupled from all
> implementation details addressd at this time.

- The compilation must take placement as an operand, and partition the function by it.
- **The operand's type:**
  - sharding specs on the function's tensors over a logical device mesh;
  - the mapping of the mesh's devices onto the map's nodes.
- **Method:**
  - GSPMD consumes sharding annotations on a few tensors, propagates them to every operator by compiler passes over a
    dataflow IR, and inserts the collectives (Y. Xu et al., GSPMD: general and scalable parallelization for ML
    computation graphs, arXiv 2105.04663, 2021). It "produces a single program for all partitions" (§1).
  - The logical mesh and its mapping onto physical devices are Alpa's (L. Zheng et al., Alpa: automating inter- and
    intra-operator parallelism for distributed deep learning, OSDI 2022, §4.1): "For a set of physical devices, there
    can be multiple logical views", and the mapping is chosen by a pass of its own.
- **The chooser is a separate pass that emits the same operand, and this compilation contains none:**
  - Alpa's own chooser works this way. "Alpa compiles each stage against its assigned device mesh, respecting the
    intra-op parallelism plan output by the ILP solver. The compilation depends on XLA and GSPMD" (Alpa, §6).
  - A later chooser of the mapping onto the cabling may be topology mapping (T. Hoefler and M. Snir, Generic topology
    mapping strategies for large-scale parallel architectures, ICS 2011).

### A3. The weighting operand: what each task and transfer costs

> 1: multiplexing and routing link scheduling must be done with respect to weighting operands which can express any
> relationship we expect to apepar later (consuming an operand of a certain type lets us defer the design or
> implementation of an optimizing chooser here).

- The compilation must take every weight its router and its scheduler use as an operand. It computes none.
- **The operand's type:**
  - a weight on each task;
  - a weight on each edge;
  - the cables' capacities, from the map.
- **Method:**
  - The task graph's weights w and c are given inputs of the scheduling problem (O. Sinnen, Task Scheduling for Parallel
    Systems, Wiley, 2007; Y.-K. Kwok and I. Ahmad, ACM Comput. Surv., 1999).
  - The router's capacities and demands are the given inputs of concurrent multicommodity flow (N. Garg and J. Könemann,
    Faster and simpler algorithms for multicommodity flow and other fractional packing problems, FOCS 1998).
- **The chooser of the weights is defined later, apart, as the chooser of A2 is.**

### A4. Collectives lower to page transfers over packed trees

> the point of having aot schedulign and compilation of schedules is to allow us to interleave any combination of
> transport requirements concurrently with one another, s.t. we are never underusing the bandiwdth on any cable routing
> when we could counterfactually swizzle together more communications to cause pipeline-blocking operations to begin or
> conclude faster, no matter waht they are!

(of "route them jointly with the ported concurrent-flow solve (Garg–Könemann), rather than each in isolation")

> routing within tb5 linked collectives is supported; attempting to involve nodes with no tb5 dma connection to any node
> within the collective is not allowed.

- The compilation must lower every collective A2 inserts to sends and receives of pages. It routes all of one
  function's collectives jointly over the Thunderbolt-linked nodes of the map, as `mesh_plan` routes every demand of its
  plan at once (rdma/mesh-collective.c, `mesh_graph_flow`). Each compiled function is its own plan (appendix item 6).
  Every node of the map is a node of the program, since every run is launched on the whole collective, so every relay
  holds the plan (A7, sender(m)).
- A node of the program to which the placement operand maps no device (A2) compiles and plans each function as every
  node does, with no section of its own. It declares the function's transfers as every rank does, and its part of each
  call is the relaying its routes give it. NCCL states the rule for every rank of a communicator: "Collective
  operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form
  a complete collective operation. Failure to do so will result in undefined behavior, including hangs, crashes, or
  data corruption." (NCCL User Guide, "Collective Operations"). (Corrected 2026-11-05: rdma/mesh_compile.py's IR
  refused such a node, "map index %d, this node, is no device of the placement operand (A2)", so it held no plan and
  relayed nothing; design/compilation-coordination.md, "The revisions of 3f106747", "Where the documents are wrong".)
  - Its calls return tensors on PyTorch's meta device: "The "meta" device is an abstract device which denotes a tensor
    which records only metadata, but no actual data", and "Most operations can be performed on meta tensors, producing
    new meta tensors that describe what the result would have been if you performed the operation on a real tensor"
    (PyTorch, "Meta device"). So the program's code between calls there computes shapes alone and allocates nothing,
    and its next compiled function is traced with the shapes and types every rank's is. "Because meta tensors do not
    have real data, you cannot perform data-dependent operations like torch.nonzero() or item()" (the same page): a
    program that reads a value between calls fails at such a node, which holds no value to read. (Corrected
    2026-11-05: the implementation's ND-2 made each output on the function's device by `torch.empty_strided`, memory
    outside the region; design/compilation-coordination.md, "The revisions of b5a3c6e6".)
  - (Withdrawn 2026-11-05, later.) The meta-device text above rested on two things A1 now removes: eager code running
    between compiled calls, and Dynamo tracing each function again at that node from its inputs. Dynamo's guards
    include each input's device, so that node compiled its functions again, and its functions' ordinals drifted from
    the other ranks' (design/compilation-coordination.md, "The revisions of e4a7398e"). With `torch.export` (A1), every
    rank compiles the same exported program before it runs. A node with no section starts its plans' executions,
    relays as its routes give it, and computes nothing; it has no value for any program code to read.
- **Method:**
  - **Trees.** Blink (G. Wang et al., Blink: fast and generic collectives for distributed ML, MLSys 2020) builds
    collectives on packed spanning trees.
    - §3.1: a broadcast's optimal rate is reached "by finding the maximal packing of a number of directed spanning trees
      or arborescences in the graph".
    - §3.2: the packing is a multiplicative-weight update for fractional packing. `mesh_graph_pack`'s Garg and
      Könemann rounds are a multiplicative-weight method of the same kind.
    - §3.3: "to do an AllReduce operation on the directed graph, we first run a reduce operation to a chosen root vertex
      using the undirected graph and then do a broadcast operation from the root vertex using the same tree but with
      links going in the reverse direction."
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
    flow, the problem Garg and Könemann's rounds solve (N. Garg and J. Könemann, FOCS 1998).
  - A2's partitioner inserts these two where GSPMD's does. ReduceScatter goes where a reduction's result is sharded:
    "gradients will be communicated with ReduceScatter instead of AllReduce" (GSPMD §3.2). CollectivePermute goes into
    halo exchange, "We use the CollectivePermute operator to exchange halo data between partitions" (§4.3), and into a
    pipeline's buffer shifting, "GSPMD will turn the buffer shifting into cross-device communication via
    CollectivePermute" (§3.3). (Corrected 2026-11-05: the text said the code refused them. No partitioner inserted
    them, so the refusal never ran and the new lowerings were unreachable.)
  - The partitioner reaches them by partitioning every operator, as A2 states ("propagates them to every operator").
    GSPMD gives the rule for the operators that need CollectivePermute: "The most typical use case of halo exchange is
    for partitinoning window-based operators (e.g., Convolution, ReduceWindow)" (§4.3). The partitioner's refusal of
    an operator, "GSPMD gives no rule here for it" (rdma/mesh_compile.py `_operator`), is false for these.
    (2026-11-05: design/compilation-coordination.md, "The revisions of f3f7cf46", stop 2.)
  - **Lowering.** MSCCLang (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, §4 and
    §5) lowers operations on chunks into an instruction DAG of sends, receives and receive-reduce-copies. A relay's
    forward is its fused receive-copy-send. Here a chunk is a page.
- **Code:** rdma/mesh-collective.c's demands and trees, as now, fed by A2's collectives in place of ranges declared by
  hand.

### A5. Every page is placed ahead of time

- **Method:** design/plan-to-bridge-change-request.md R2, C3's allocator. As now.

### A6. One schedule per function

- **Method:** design/plan-to-bridge-change-request.md R3 to R5, over the weights of A3: R5's heap order by the
  completion times of one linear program (Chen, 2022; Hall, Schulz, Shmoys and Wein, 1997), solved by HiGHS.

### A7. Function executions of programs that run at once are ordered by one scheduler for the island

> nothing happens when a program joins or leaves because a program is a sequence of like, function calls.

> it is rqeuired to choose nad use a scheduler. cite anyhthing do anything but don't use idiomatic citation free claude
> code anywhere here

> "The real question sits at the sequencer's level. " perhaps this must be said more definitely and more literally and
> more citationally in the change request documentation s.t. any competing interpretations are completely ablated...

- **What the code must do.** Order the function executions of every program on the island in one order, the same at
  every node, and keep each program's own order of its calls within it.
  - (Corrected 2026-11-05.) "Every program" is the program: a run is one program on the whole collective. "Across all
    programs running on the island" is Pathways' setting, many clients' programs, and no requirement here asks for it.
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
  - The island is the map's Thunderbolt-linked component (A4). Each function execution runs its own compiled schedule
    (A6).
- **The scheduler is total order multicast by a fixed sequencer**, as X. Défago, A. Schiper and P. Urbán define both
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
  - **The algorithm** is their Fig. 5, the "UB" variant of §7.1, as written:
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
  - **What each term names here:**
    - **m** is one submission: its plan's name (`group`, `epoch`), the identifier of the view it is sent in (below),
      and a range of its executions [`first`, `end`) (rdma/mesh-bridge.h struct mesh_order). One submission is one message however many
      executions it carries, as Pathways sends "a single message (describing the entire subgraph)" (§4.5). e2b's run is
      one message.
    - **sender(m)** is the program's one sender: the least of the program's nodes. It TO-broadcasts the program's
      submissions in the order the program makes them. The program's nodes are the nodes its launcher started it at,
      an operand of the plan call (`program`) that the launcher passes as it passes `group` (MESH_GROUP). e2b's
      launcher, tools/mesh/run_pair.py, passes no node set: it starts rank r at map index r on every node of the
      collective, so e2b's program's nodes are map indices 0 … R − 1, which tools/e2b_replay.m passes. A torch
      program's launcher passes them as MESH_PROGRAM beside MESH_GROUP (rdma/mesh_compile.py `_program`). Every run is
      launched on the whole collective ("you may never use less than the entire collective under any circumstances.
      all is okay, 2 is not okay. all is cheaper than 2."), so the program's nodes are every member of the view, every
      relay of A4's routes is one of them, and every node of every plan, relays among them, calls `mesh_plan` and holds
      the plan.
    - **The sequencer** is the bridge of the highest-numbered node the map reaches (rdma/mesh-records.c `island_set`),
      the code's choice, which the class leaves open. "The sequencer is unique, and the responsibility is not normally
      transfered to another processes" (§7.1). It creates the island's views (below).
    - **sn(m)** is struct mesh_order's `sequence`.
    - **Dest(m)** is the nodes of m's plan (struct mesh_order's `nodes`).
    - **deliver(m)** at a node of Dest(m) appends m to its plan's queue there (appendix item 2), which releases its
      executions at the node. The pages of delivered executions interleave on each cable (T99).
  - **Failures.** "This algorithm does not tolerate the failure of the sequencer" (§7.1). A plan stands only while its
    map does (design/disjoint-pages-change-request.md D.3.6), so a loss ends every plan it touches, and no standing
    plan's sequencer changes. A plan also stands only within its view (below).
- **One sender for each program.** "Total Order alone does not guarantee that messages are delivered in the order in
  which they are sent" (§4.3.1), and FIFO Order is a property of one sender's messages. Where two nodes broadcast calls
  of one program, nothing makes one node's broadcast precede the other's (Lamport's "precedes", their Definition 1), so
  neither FIFO Order nor Causal Order (§4.3.2) need keep the program's order. With one sender, whose messages reach the
  sequencer first in, first out, Fig. 5's sequencer numbers them in the order they were sent ("when receive (m) sn(m)
  := seqnum"). So FIFO Order holds, and with it A10's order constraints between the program's calls.
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
    start and at each move of its map. It also creates one whenever a member's announcement names a view other than the
    sequencer's installed one, with a counter not less than that view's identifier. Such a member installed another
    sequencer's view after the announcement the sequencer counted, and it ignores the sequencer's view by Local
    Monotonicity. The identifier is "computed based on the values of local counters" (below), so the sequencer computes
    it again from the counters it now holds. To tell such a member from one that installed the sequencer's own view,
    each announcement names the process that created its bridge's installed view as well as its counter. (Corrected
    2026-11-05: a risen counter moved no map, so no view followed it; design/compilation-coordination.md, "The
    revisions of 3f106747", "Where the documents are wrong".)
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
    - When the sequencer's map moves, it numbers no further message of the old view. Its first message of the new view
      carries the last sequence number it gave in the old one.
    - A member that installed the old view from the same process delivers the old view's messages up to that number,
      then installs the new view.
    - Where the old view's sequencer is gone (its bridge stopped, or the map moved to another sequencer), the old
      view's undelivered messages are never delivered: "This algorithm does not tolerate the failure of the sequencer"
      (§7.1).
  - **A plan stands only within the view its nodes' arrivals were counted in.** Installing a view ends every plan of
    the old one with `ESTALE`, as D.3.6 ends a plan at its map's move, and the plan's replan arrives in the new view.
    Each arrival carries its view's identifier, and the program's sender counts only the arrivals of the view it has
    installed.
  - (Corrected 2026-11-06: design/compilation-coordination.md, "The revisions of 55b9cbad", "Where the documents are
    wrong".) **Every member of a view plans on the view's map.**
    - The text above let each node plan on its own map. While a map move floods, two nodes can plan one name on two
      maps within one view. An index one plan gives the other, a signal's entry, can then name an entry the other
      plan does not have: a store outside it.
    - The map a view is created from is the sequencer's, and the view's first message carries it: the records the
      sequencer's map holds when it creates the view. Every member plans on those records.
    - So every member of a view computes the same plan from the same operands. Replicas that start in the same state
      and execute the same inputs in the same order produce the same outputs (F. B. Schneider, Implementing
      fault-tolerant services using the state machine approach: a tutorial, ACM Comput. Surv. 22(4), 1990). An index
      one end's plan gives is the other end's by construction, and nothing tests it at run time.
    - A plan then stands while its view does, and its view while the sequencer's map does (D.3.6).
- **A plan's destinations exist before its first message.** Within a view, Section 3's static system holds for the
  processes the view names, and a plan's first message must find all of them holding the plan.
  > "What I was describing is that compilation finishes at different nodes at different times. "
  > this is trivially allowed and if claudes would stop delaying basic logistics for collective propgramming by going
  > off the rails all the time we could be talking about how to use the collective as a work stealing compute bloc
  > allowing nodes to cooperatively accelerate remaining compilation work on peers that somehow haven't finished yet.
  > but fancy use cases require our library code to already work at roughly literature-sota optimality for all tasks
  > rahter htan special narrow cased imaginary tasks claude hypothesises when writing idiomatic code or idiomatic specs.
  - Nodes finish compiling a plan at different times, and that is allowed. A plan is created as an NCCL communicator
    is: "ncclCommInitRank implicitly synchronizes with other ranks" (NCCL 2.32.3, API, Communicator Creation and
    Management Functions).
  - The program's sender counts the plan's nodes as each holds the plan, as a centralized barrier counts arrivals:
    "each processor updates a small amount of shared state to indicate its arrival and then polls that state to
    determine when all of the processors have arrived" (J. M. Mellor-Crummey and M. L. Scott, Algorithms for scalable
    synchronization on shared-memory multiprocessors, ACM TOCS 9(1), 1991, §3.1). It TO-broadcasts no submission of
    the plan before every node of the plan has arrived. A node lost first moves the map, and the plan fails with
    `ESTALE` (D.3.6).
- **Corrections** (2026-11-05):
  - The text before had each call submitted "by its plan's least node (`sender`, rdma/mesh.h: "the fixed sequencer's
    one sender a message")". A program's plans have different least nodes, so one program had several senders and no
    FIFO Order between them. That was the deadlock in Claude's turn. The works cited have none: Pathways' one client
    and FIFO scheduler give the consistent order its §2 requires.
  - The text before called the sequencer the bully algorithm's coordinator (H. Garcia-Molina, IEEE Trans. Comput.,
    1982). No election runs, so that algorithm is not used.
  - The text before reasoned from the bridge's ports and lanes, and from plans made "at run time". Lanes are the
    bridge's alone (design/plan-to-bridge-change-request.md R7, "Nothing outside the bridge knows the transport"), and
    a plan is the compilation's output, made before its function runs (A1, A6).
  > 'blah blah blah deadlock' do the works we cite in the literature have deadlocky runtimes or compiletimes or was that
  > a deviation from the work we were developing?

  > ""Holding a lane," plainly:" did you need to talk about this at all or is it an irrelevant distraction from the levle
  > ofa bstraction and the specific scopes of interfaces involved in the change request?

  > "That was the whole deadlock: my A7 gave one program several senders."
  > then this is an undercitation over-idiom issue and has a known remedy (making specification more explicit, more
  > citational, less idiomatic, removing ad libs that feel zesty and fun and exciting to write.)
  - (Corrected 2026-11-05, later: the text before required Fig. 5's initializations at each bridge's start and cited
    §6's static system for the island. The island's bridges start at different times, so a bridge started after the
    sequencer had numbered messages, and every node after the sequencer's bridge started again, delivered nothing
    (design/compilation-coordination.md, "The revisions of abecd389", stop 1). Défago, Schiper and Urbán leave that to
    a group membership service (§6.1, §8.2), and the text named none. The text before also had m carry "the map serial
    it was made on", which no receiver can compare: each node's serial is its own count. And it named the program's
    nodes by rdma/mesh.h, which holds none.)
  - This is not the coordination's Q3′, which appendix item 2 removes: Q3′ set the counter back to 0 and dropped
    pending orders at each node's own map move, with no citation and no algorithm. Here the numbering begins again only
    at a view's creation, which the sequencer decides for every member, and the old view's messages are flushed first
    (§4.2.1).

### A8. The IR operand

> some operand must clearly carry the ir used to make this kind of optimization / planning choice, and something must
> serialize this into an operand consumedy by ordinary transport / bridge functions?

- The compilation must carry one IR from partitioning to serialization, as an operand of its own.
- **The operand's type:**
  - the task graph of design/plan-to-bridge-change-request.md R3 (Sinnen, 2007);
  - each consumer's kind (R4: a join for a synchronous, complete, contiguous consumer, a task per page for an asynchronous, incomplete, noncontiguous consumer);
  - the placement operand (A2) and the weighting operand (A3).
- **Method:** MSCCLang carries its instruction DAG the same way, from lowering to scheduling (Cowan et al., 2023, §4 and
  §5).
- A2 partitions, A4 lowers, A5 places and A6 orders it.

### A9. The serialized operand, in MSCCL-IR's shape

> no inlining new numerical backends instead of writing collective and transfer and so on code which schedule the use of
> existing code, basically.

- **Method:** MSCCL-IR (Cowan et al., 2023, §5): "a tree data structure that divides a collective into individual GPU
  programs that are subdivided into thread blocks containing a list of instructions that are sequentially executed". An
  instruction that depends on another list carries the dependency explicitly ("a dep modifier", §5.2).
- **Here, per node:**
  - for each adjacent node and direction, the ordered list of page instructions, each with its page, its bytes and its
    dependency word. A send's word is its supplied word, or for a forward its receive's phase; a receive's are its phase
    and count. These are the plan's struct mesh_hop lists (design/plan-to-bridge-change-request.md R5);
  - the page table's bindings and the plan's state section (C3);
  - the GPU's program: the function's own existing kernels, bound to their placed pages. No kernel is written by the
    compilation. For a delegate's program these are the recorder's command buffers, as now.
- The bridge consumes the lists and the GPU runs the existing kernels. Nothing else is serialized but each entry's set
  mark (appendix item 5).
- (2026-11-05.) **What the bridge runs: the self-timed schedule.**
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
    - Its firing is each entry's wait for its dependency word, C3's counted completion: BSL's "run-time
      synchronization when they communicate data".
  - **Its serialized form** is MSCCL-IR's, above: for each processor "a list of instructions that are sequentially
    executed", each carrying its dependency ("a dep modifier"). The list is the self-timed order. The graph's edges are
    the entries' dependency words, so nothing of the graph is lost.
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
  - **What runs is written strictly to these sources:** the schedule to Lee and Ha's self-timed strategy, the
    serialization to MSCCL-IR, the sets' order to Fisher–Yates over Threefry, and the firing to C3's counted
    completion. Nothing of the bridge's execution is the code's own invention.

### A10. A program's calls are a dataflow graph, analysed at compile time

> is this really a choice? are the problems being discussed (all linear algebraic programs ever, basically) 'synchronous
> by implicit assumption', or are all operations of htis kind composeable and rearrangable up to typing limits which are
> determined by the linear algebraic operations / source we are using in the data processing function side? does the
> commutativity and associativity of the operations (ignore e.g. floating point associativity for now) form some kind of
> basic primordial pre-requirement that the user expected someone to ask questions about, pull in literature about, and
> start describing as yet another AOT precompilation problem of the 'known parser, low empirical computational
> complexity' class of programs? on some level the use of the fxgraphs capture and pytorch-modifying-headers should
> foreshadow this and make it less of a surprise

- The compilation must take a program's sequence of calls as a graph, decide at compile time which calls may be
  outstanding at once and in what orders, and compile that into the plans. Nothing about it is chosen at run time.
- **Method:**
  - **The calls' graph.** It is Pathways' program tracer: "a new program tracer (Figure 2) that a user can wrap around a
    block of Python code that calls many compiled functions. The tracer generates a single PATHWAYS program where each
    compiled function is represented by a computation node in a dataflow graph" (Barham et al., 2022, §3).
    - For torch, the tracer is torch.fx's symbolic tracer (J. Reed, Z. DeVito, H. He, A. Ussery and J. Ansel,
      torch.fx: practical program capture and transformation for deep learning in Python, MLSys 2022). Each compiled
      function is a leaf, so a call of it is one node of the graph and is not traced into (`Tracer.is_leaf_module`,
      `torch.fx.wrap`). Each function itself is captured as A1 states.
    - Its reads and writes are its nodes' tensor arguments and results, which is what Bernstein's conditions take.
  - **Its bound.** The traced block is finite, so each function has finitely many calls in it. Pathways sends a traced
    block whole: "the program sends a single message (describing the entire subgraph) to the scheduler" (§4.5). So the
    most calls of a function outstanding at once is the number of its calls in the block, or fewer where Bernstein's
    conditions order them.
  > get real. citation, citation, citation. don't invent idiomatic reasons why you can't do things either.
  - **Independence:** two calls may run in either order, or at once, where neither writes what the other reads or
    writes. These are Bernstein's conditions (A. J. Bernstein, Analysis of programs for parallel processing, IEEE Trans.
    Electronic Computers EC-15(5), 1966).
  - **Commuting operations:** calls whose effects commute, such as reductions into one accumulator by an associative and
    commutative operation, may run in any order (M. C. Rinard and P. C. Diniz, Commutativity analysis: a new analysis
    technique for parallelizing compilers, ACM TOPLAS 19(6), 1997). Floating-point non-associativity is set aside, as
    the operator does.
  - **At run time:** calls are issued ahead, as futures, in the order the analysis allows, as Pathways dispatches
    asynchronously (Barham et al., 2022, §4.5).
- **Its graph operations are the graph module's.**
  > inlined code is unilaterally bad and to be deduplicated and uninlined where possible and appropriate wrt graph
  > operations module query.
  - Rubric C9: "the form with fewer copies of each primitive, offered for others to call rather than spliced into their
    code" (design/rubric-actionable.md).
  - The order's transitive closure, the walk of commuting chains, and the width by Fulkerson's matching in Hopcroft and
    Karp's phases are functions of rdma/mesh_graph.py, beside `walk` and `reach`. As that module states, "A caller
    supplies its vertices and a neighbours function." rdma/mesh_compile.py calls them and keeps no copy.
  - A body's symbolic execution runs on torch.fx.Interpreter, as the program's run already does (`_Program`), not on a
    second loop over a graph's nodes.
  - The same holds wherever mesh's code walks a graph. (2026-11-05: the operator's ruling on the implementer's C9a,
    design/compilation-coordination.md, "The revisions of f3f7cf46". The rules' bar on rdma/mesh_graph.py does not
    stand against it.)
- **Its output:**
  - for each compiled function, the most calls of it the analysis lets be outstanding at once in the program that
    submits them, which bounds how many the program issues ahead;
  - the order constraints between calls, which the program keeps when it submits, so the sequencer's first-in, first-out
    numbers keep them too (A7);
  - both carried in A8's IR.


### What judges it

- design/plan-to-bridge-change-request.md's judges hold.
- The compilation contains no chooser of placement and no chooser of weights; both arrive as operands.
- No kernel is written by the compilation.
- The three builds are clean. Runs come only when the operator authorizes them.

## Appendix: a chooser for edge weights

> nevertheless, you should have found a 'chooser' in the literature for how to weight 'edges' and cited it; do so
> immediatley as a supplementary appendix as this is not a requirement that blocks the writing of the change request
> thus described.

This appendix is supplementary. A3 holds as written: the compilation takes the weighting operand and computes no
weight. This names one canonical function that can fill the operand's edge weights. Using it means passing it in place
of unit weights; nothing in A1 to A9 changes.

- **The model:** the α–β model of a transfer. A transfer of n bytes over one link takes α + n β, where α is the
  link's start-up time and β its time per byte. Hockney states it as t(n) = (n + n½) / r∞, with r∞ the asymptotic
  rate and n½ the length at which half of it is reached (R. W. Hockney, The communication challenge for MPP: Intel
  Paragon and Meiko CS-2, Parallel Computing 20(3), 1994).
- **Where it is already used:**
  - MSCCLang states the model where it decides aggregation: "Assuming an α − β communication cost model, each send has a
    start up α cost and a per-byte β cost" (M. Cowan et al., ASPLOS 2023, §5.1).
  - HEFT weighs a communication edge the same way, as a start-up cost plus its data over the link's bandwidth (H.
    Topcuoglu, S. Hariri and M.-Y. Wu, IEEE TPDS 13(3), 2002).
- **The edge weights it gives:** an edge e of n bytes weighs α_k + n β_k on each cable k of its route, where each hop
  is its own instruction (design/plan-to-bridge-change-request.md R5).
  - β_k is 1 over cable k's bandwidth, from the map (mesh-map.h).
  - α_k is fitted from measured transfer times of at least two sizes on that cable, as Hockney fits r∞ and n½.
  - Measuring needs runs, which wait for the operator's authorization. Until α_k is measured, it is 0, the model's
    bandwidth-only form. That default is this appendix's choice, made because it needs no run.
- **Its place in the code:** a function from (bytes, cable) to a weight, passed to the compilation as its edge-weight
  chooser. Any other chooser can replace it, as A3 requires.

## Appendix: the citations the implementation omitted or ignored

The implementation of this change request and design/plan-to-bridge-change-request.md (mesh ce3d5d5e, metal-microbench
30a5f9fb; its record is design/compilation-coordination.md) departed from them where it took decisions with no citation,
or set a cited method aside. Each item below names the decision, the citation that settles it, and what the code must
do. Where this appendix and design/compilation-coordination.md differ, this appendix holds.

> so we see that limiting requirements and algorithmic claims to citations to the ltierature would tell us exactly what
> the user was saying to do, and that the subagnets were lying about experiencing a constraint taht came entirely from
> using idiomatic data structures or algorithms.

### 1. Unrolling: C3.3 as written, without the coordination's D5

> "D5 appears in neither Lam, Rau nor C3.3's text." lying subagent problem.

- **Decision without a citation:** design/disjoint-pages-revision-coordination.md D5 refuses a value whose write does
  not reach its own next instance at lag 1. That refusal is why a lone AllGather or AllToAll is refused.
- **The citation it set aside:** C3.3 (M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo scheduling,
  MICRO 1994): "For each value, L is the smallest lag at which every read of it in execution t happens before its write
  in execution t + L. The unroll factor u is the largest L. … A value with no such L fails the plan with `ENOMEM`."
- **The code must:** compute u as C3.3 states and refuse only a value with no such L. D5's lag-1 condition goes.

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

- **Decisions without a citation:**
  - "A cable never interleaves two executions". These are Claude's words in the coordinator's brief, against T99.
  - The coordination's Q4: a node releases the next plan only once every port has served the previous one.
  - Q4′: a head not loaded within `MESH_PLATFORM_NS` is delivered "with nothing released".
- **The citations:**
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

  - The sequencer numbers each message m, A7's submission: a plan's name and a range of its executions (Défago, Schiper
    and Urbán, 2004). An execution is Pathways' unit, "function executions" (§4.4).
  - **What the code calls an ORDER** is the fixed sequencer's relayed message. Défago, Schiper and Urbán, §7.1: "To
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
  - **Each plan's delivered calls reach the bridge at a node through a queue of the plan's own.** It is laid out with
    the plan's pages when the plan is placed, and placed and released with them by the one allocation over the program
    (C3.2; C3.4 as corrected 2026-11-05). The bridge reads it in
    place, by view, and nothing copies it. Only the driver moves bytes (design/plan-to-bridge-change-request.md R1).
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
      Trans. Computers C-36(1), 1987). The arc here runs from a node's delivery of a plan's calls to the bridge's taking
      of them at that node, so there is one queue for each plan at each node of it.
    - **Each entry is one delivered message, with its sequence number**, as the sequencer "relays m with its sequence
      number" (Défago, Schiper and Urbán, §7.1). The plan's k-th delivery at the node is entry k mod the queue's room.
      One submission is one message however many executions it carries, as Pathways sends "a single message
      (describing the entire subgraph)" (§4.5). e2b's run is one submission.
    - **Its room is fixed at compile time, by the self-timed buffer bound.** For a schedule synchronous dataflow
      admits, "the buffers must remain bounded" (Lee and Messerschmitt, the proof of their Theorem 1), and the bound is
      found before the program runs. The queue is an IPC edge of a self-timed schedule as S. S. Bhattacharyya, S.
      Sriram and E. A. Lee define one (Optimizing synchronization in multiprocessor DSP systems, IEEE Trans. Signal
      Processing 45(6), 1997, here BSL). Its source is the sender's TO-broadcast of the plan's calls, and its sink is the
      bridge's taking of them at the node.
      - **The graph.** The compilation builds the program's IPC graph from A10's analysis as BSL's §4.1 builds one:
        "if actors v1, v2, …, vn are assigned to the same processor in that order, then Gipc would have a cycle ((v1,
        v2), (v2, v3), …, (vn − 1, vn), (vn, v1)), with delay((vn, v1)) = 1". The processors are the program's nodes.
        The actors are each node's parts of the program's calls, in program order. The IPC edges are the plans'
        transfers, each with C3.3's u copies as its delay, and these queues. At each node, A10's bound is the delay on
        the edge from a call's completion to the program's later submission. Execution times t(x) are A3's weighting
        operand.
        - **Each plan builds its own part of that graph**, when it is planned: its actors, its queues and its transfers,
          with the program's sender and A10's bound. A torch function is compiled at its first call (A1: TorchDynamo
          hands each captured graph to the backend), so no later function's routes exist yet. Every cycle of the part
          is a cycle of the program's graph, and "The number of tokens in any cycle of the IPC graph is always
          conserved" (BSL, Lemma 2). So (5) over the part's cycles is never less than (5) over the program's: the room
          is never too small.
        - (Withdrawn 2026-11-05, later.) The bullet above rested on capture at a function's first call, which nobody
          asked for (A1, corrected). Every function is exported and compiled before the program runs, so the
          compilation builds the program's whole IPC graph, as this item states above, and each queue's room is (5) in
          that graph.
        - (Corrected 2026-11-05: design/compilation-coordination.md, "The revisions of 4e78f6f4", IG-5.) **The
          program-order cycle is each node's submissions, not its calls' parts.**
          - BSL counts the iterations a processor may hold at once by the delay on its back edge. The cycle quoted
            above has "delay((vn, v1)) = 1".
          - Each node's program submits its calls in program order. The sender "TO-broadcasts the program's
            submissions in the order the program makes them" (A7), and every node's program starts its plans'
            executions in its own program order. So the submissions σ_1 … σ_n of the block's calls are one processor's
            actors, in that order, each of time 0, as the sender's submission is above.
          - Call k's part at the node, x_k, follows its submission (σ_k → x_k, delay 0). A10's bound b is the edge from
            x_k to the submission b calls later.
          - So successive parts overlap by up to b calls, and no edge orders the parts themselves: "A call still
            running at a node delays no later call's delivery there" (below).
          - A part's sends are actors on R5's cables, A9's processors, in the self-timed order. The two statements
            describe different processors of one IPC graph, so they do not contradict each other.
        - **The program's submission and its bridge's TO-broadcast** are two actors at the sender. The submissions'
          queue between them is an IPC edge with a bound of its own. Their work is IPC operations alone, and "t(v)
          includes the time taken to execute all IPC operations (sends and receives) that the actor v performs" (BSL
          §4.1). A3 weighs neither, so each takes time 0, where BSL's times are positive integers. A room reads delays
          alone, (5) and Theorem 2's "Bfb(e) = ρ(snk(e), src(e)) + delay(e)"; only the cycle mean and DetermineDelays
          read times, and a cycle's mean is defined for times of 0. (2026-11-05: design/compilation-coordination.md,
          "The revisions of 3f106747", BS-2 and BS-3, and "Where the documents are wrong".)
      - **On a cycle**, the queue's room is BSL's self-timed buffer bound: "Bfb(e) = min({Delay(C) | C is a cycle that
        contains e})" (§4.3, (5)). Its synchronization is their BBS, which "only explicitly ensures" that "an
        invocation of snk(e) never attempts to read data from the buffer unless the buffer contains at least one token"
        (§5.1).
      - **Feedforward**, as where the program's sender holds no page of the plan: "Feedforward edges have no such bound
        on buffer size", and such an edge "could be unbounded when the execution time of A is less than that of B"
        (§4.3, Fig. 2(a)). The compilation adds synchronization edges by BSL's Convert-to-SC-graph (§9, Fig. 6) and sets
        their delays by DetermineDelays (Fig. 8), so that "e becomes encapsulated in an SCC; such a transformation would
        allow e to be implemented with BBS" and "the estimated throughput is not degraded" (§9). The queue's room is
        then its bound by (5) in the converted graph.
      - **An added edge (vj, vi)** holds "start(vi, k) ≥ end(vj, k − delay((vj, vi)))" (BSL (7)). Its source raises a
        count that only rises (C3's counted completion: Portals 4 counting events), and its sink reads that count
        before its k-th start. The bridge carries the count from source to sink, as it carries an arrival.
        - The end of a node's part of a call is not in the region at that node: only its port children see its sends
          complete. The count its source raises is therefore the program's starts of the plan at that node. A start
          there follows the end of the call A10's bound before it, since A10's bound is the delay on the edge from a
          call's completion to the program's later submission. So the edge the count holds is (7) with
          DetermineDelays' delay plus A10's bound, and the rooms take that delay. (2026-11-05: design/compilation-
          coordination.md, "The revisions of 3f106747", BS-4.)
      - **Where they live.** The strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle mean,
        MinDelay's binary search and DetermineDelays are functions of the graph module's C half, rdma/mesh-graph.h and
        mesh-graph.c. They take the caller's vertices, edges, delays and times as operands (A10: "The same holds
        wherever mesh's code walks a graph"). Their one caller is `mesh_plan`, which alone holds each plan's routes and
        u when it plans. (Corrected 2026-11-05: the text said rdma/mesh_graph.py, where nothing calls them;
        design/compilation-coordination.md, "The revisions of 3f106747", BS-1.)
      - A plan whose bound cannot be found fails with `ENOMEM` when it is planned, as a value without a lag does in
        C3.3. That happens where A10 finds no bound, or where a cycle has no delay (BSL's Lemma 1: deadlock).
    - **The bridge takes its plans' calls by a multiway merge on the sequence number** (D. E. Knuth, The Art of Computer
      Programming, vol. 3, §5.4.1): of the first untaken entries of its standing plans, the least sequence number first.
      Fig. 5 delivers in sequence-number order (its `nextdeliver`), so every entry below the node's delivery count is in
      its queue already, and no smaller one comes later.
    - **Nothing waits to write, and nothing checks for a full queue.** Every queue is a BBS edge with its bound as its
      room, so no delivery reaches an entry the bridge at the node has not taken. The only waits are at the sinks of
      the added synchronization edges, on a count that only rises. Each ends when its source moves on, or when the view
      ends and the plan with it (`ESTALE`).
    - (Corrected 2026-11-05, later still: the text before bounded the room by "A10's bound on the plan's submissions
      outstanding at its sender, plus C3.3's u copies on each hop ... so a node's part of a call trails the sender's by
      at most u copies a hop", calling the number "this document's derivation". It holds only where the queue lies on
      a cycle through the sender. A7's program sender may hold no page of a plan, and then the queue is BSL's
      feedforward edge with no bound (design/compilation-coordination.md, "The revisions of abecd389", stop 2).)
    > consider whether this is a fictional problem or a problem that appears only through idiomatic citation free code
    > before making any further statements
    - (Corrected 2026-11-05: the implementation's M4 kept an ORDER at the channel's head while a port trailed by 4096
      entries, a wait at the writer. Its M5 copied the channel into each port's memory. Neither has a citation, and
      both go. The text before cited Kahn's unbounded channels, which a region laid out once cannot hold.)
    - (Corrected 2026-11-05, later: the text before laid out one channel for every plan, once with the region, and said
      "only a stopped process" could fill it. That is false: its bound held for one plan's calls only, and other plans'
      calls keep arriving in a shared channel while one plan's call waits at the node, as it may, for the node's program
      to start it (design/compilation-coordination.md, "The revisions of f3f7cf46", stop 1). The overflow came from the
      shared channel, which no cited work has: synchronous dataflow has one queue for each arc. Claude's later "a room
      per plan" and "each entry checked against its index" came with no citation and no algorithm.)
  - The bridge at each node takes the delivered calls of its standing plans in sequence-number order (the merge above).
    Every node of a plan delivers the same sequence (Défago, Schiper and Urbán, §3, Total Order), so every node of it
    takes its calls alike. The pages of the calls taken interleave on each cable (above).
  - (Corrected 2026-11-05: the text before this one fixed a number of delivery words a plan, from A10's bound. It then
    found no word free, at a node still serving an earlier call, for a later call's order, and marked that open. Fig.
    5's destinations have no such limit, and delivery is not completion.)

- **The code must:**
  - deliver orders as Fig. 5's destinations do;
  - publish each plan's deliveries to the bridge at each node through the plan's own queue above, read in place and
    taken by the multiway merge, with no copy, no wait and no check. Each queue's room is its self-timed buffer bound
    in its plan's part of the program's IPC graph, and the feedforward queues' synchronization edges come from Convert-to-SC-graph and
    DetermineDelays, all at compile time (BSL, above). This replaces the sequence words (`sequences`), `taken[p]` and
    the parent's wait while they are full (Lamport's buffer; the coordination's DL1), and then the one channel for
    every plan (the implementation's CH2, with M3's room). M4 and M5 stay gone;
  - remove the start's wait for every port to load the plan, and the late-loading port's refusal of it
    (`mesh_ports_held`). Nothing requires them. "a send that arrives first is held until its receive is posted (§3),
    and a loss cancels the plan anyway" (Claude's turn in "The turns" that lists "Every wait, and whether anything
    requires it", whose verdict on this one was "remove");
  - interleave released executions on each cable;
  - remove Q4's hold and Q4′'s timeout skip (done: design/compilation-coordination.md DL4, O1);
  - remove the coordination's Q3′: the sequencer's number never goes back to 0, and no node drops a pending order when
    its map changes. (2026-11-05, later: the number begins again at 1 only at a view's creation, by A7's views; the
    old view's orders are flushed before the new view is installed, and no node's own map move drops one.)
    > "When a node's map changes, its bridge parent drops its pending orders, and the sequencer sets its counter back
    > to 0. "
    > fictional problem + nobody asked + no citation + no algorithm + no explanation + idiomatic.

### 3. A slow node, and "desynchronization": no handler

> has this EVER HAPPENED or did someone INVENT A FICTIONAL PROBLEM THAT HTEY WANTED TO WRITE A FICTIONAL HANDLER FOR?
> show me 1000 mesh desynchronization events or get this off my desk forever.

- **Decision without a citation:** Q4′'s timeout, and Claude's later appeal to failure detection and group membership.
  No such event has been observed or recorded, and no requirement asks for it.
- **The code must:** hold no handler for it. Waits stay bounded, by the rule C3 already states.

### 4. A function's executions are never truncated at one end; no handler

> does this *literally ever happen* or can this *always be prohibited* by the use of atomic writes/reads for the
> interface between queued pages for transfer and the bridge in particular? who thinks that clients are going to be
> failing literally all the time and that noticing or preventing these failrues is the responsibiltiy of oru code?

- **The citation:**
  - A plan is published whole, by a release store after all of its writes, and read by an acquire load (C3's writing
    order; H.-J. Boehm and S. V. Adve, Foundations of the C++ concurrency memory model, PLDI 2008). The bridge never
    sees a plan in part.
  - A released execution is served to completion at both ends, because its words only rise (C3.7).
  - A call returns once its own landings and sends are done (C3.1). So at each end every page of an execution is matched
    before the call's next function begins.
- **When it can happen:** only when a process dies in the middle of a call. Process death is the failport's lifecycle's
  (T151), not data or collective code's.
- **The code must:** hold no handler for a truncation in data or collective code. The coordination's open item about a
  plan ending at one node while a peer still serves it is withdrawn.
- **Withdrawn:** Claude's claim that the failport discarded such a channel ("the stream is never live again in the
  pairing") is withdrawn. The operator: "No."

### 5. Independent flows are drawn in uniformly random order

> a new requirement straight from noumena has arrived: all data which is typed as an 'independent flow' or whatever must
> now have nondeterministic order, and *must* be transmitted in different permutations of which pages were submitted
> across each page_transmit_iteration in which the bridge draws from the submitted pages it asynchronously draws down as
> hardware literal SENDs and RECEIVES. this means that running the same compiled collective function 100 times shoudl
> result in, for example, a pointiwse nonlinearity after a ffn, to be transmitted in uniformly distribtued orderings of
> pages, with no expected difference in the mean or variance of the total time spent transmitting or operating the
> related functions even though the accesses are technically different each time. i am not joking around here.

- **What is serialized:** the pages of asynchronous, incomplete, noncontiguous consumers' operands (design/plan-to-bridge-change-request.md R4: one task per
  page, any order) are written as sets, not as a sequence. A set is one demand's pages on one of its trees, as the code
  numbers them (rdma/mesh-collective.c: tree j's entries have `set` j + 1), so a set's pages have one producer and one
  route. The bridge sees sets and sequences, and no kind (R4).
- **What the bridge does:** each execution of a set on a cable is sent and received in a uniformly random permutation of
  its pages.
  - The permutation is drawn by Fisher and Yates's shuffle (R. A. Fisher and F. Yates, Statistical Tables, 1938; R.
    Durstenfeld, Algorithm 235: Random permutation, CACM 7(7), 1964; D. E. Knuth, The Art of Computer Programming vol.
    2, §3.4.2, Algorithm P).
  - Its random numbers come from Threefry-4×64-20, a counter-based generator (J. K. Salmon, M. A. Moraes, R. O. Dror and
    D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, SC 2011, §4.2; rdma/mesh-flow.c `threefry4x64_20`). Its key
    is the plan's name (item 6), whose group the launcher draws anew at each launch (metal-microbench
    tools/mesh/run_pair.py: `1 + secrets.randbelow(2**64 - 1)`, passed as `MESH_GROUP`). Its counter is (execution,
    set). So both ends of a cable draw the same permutation without exchanging anything.
  - Algorithm P needs a uniform integer in [1, j] at each step. The code takes the high word of j times a random word
    (rdma/mesh-flow.c `set_draw`), which is Lemire's multiply-and-shift. With Lemire's rejection step, which the code
    must add, the integer is exactly uniform (D. Lemire, Fast random integer generation in an interval, ACM Trans.
    Model. Comput. Simul. 29(1), 2019).
- **Repeated runs:** across runs, the same function is sent in different, uniformly distributed orders. The pages and
  their bytes are the same each time, so the requirement's mean and variance hold over the set.
- **One permutation of the whole set per execution, as one renaming on every channel the set crosses.**
  > figure this out for arbitrary graphs which evenutally collect all nodes in a collective by any number of switching
  > nodes. more citations, more literature, fewer guesses, fewer idioms. there are tools in this repo to vomit up what
  > the mesh topology literally is right now. it is not a tree and never looked anything like one.
  - **The fabric now.** It was read on 2026-11-05 by bin/mesh-nodeinfo.sh on each member, matching each port's
    neighbour to the node that owns it. It has five Macs and thirteen Thunderbolt cables, and it is a multigraph with
    cycles:
    - mdots-Mac-Studio to m5ult2: three cables;
    - mdots-Mac-Studio-2 to mdots-Mac-Studio-3: four;
    - mdots-Mac-Studio to mdots-Mac-Studio-3: two;
    - one each for mdots-Mac-Studio to Ms-Mac-mini, mdots-Mac-Studio-2 to Ms-Mac-mini, mdots-Mac-Studio-2 to m5ult2,
      and Ms-Mac-mini to m5ult2.
    - One cable of each of the first two pairs is inferred from the nodes' routes, because each of its ports listed its
      own address as its neighbour.
    - No cable joins mdots-Mac-Studio and mdots-Mac-Studio-2, mdots-Mac-Studio-3 and m5ult2, or mdots-Mac-Studio-3
      and Ms-Mac-mini. Their transfers relay through other nodes.
  - Routes are packed trees over this graph (A4; Blink §3.1). A pair with several cables carries a tree's edge over
    all of them, and the bridge deals its positions among them (design/plan-to-bridge-change-request.md R5).
  - **What rules out deadlock on any graph.** Dally and Seitz give the absence of cycles in the channel dependency graph
    as the necessary and sufficient condition for deadlock-free routing (W. J. Dally and C. L. Seitz, Deadlock-free
    message routing in multiprocessor interconnection networks, IEEE Trans. Computers C-36(5), 1987). It holds for any
    topology and any number of relaying nodes.
    - Here a channel is any first-in, first-out sequence of positions the bridge keeps between two adjacent nodes, the
      same at both ends (data-model §3). How many a cable has is the bridge's alone
      (design/plan-to-bridge-change-request.md R7). A position on one channel depends on another channel where its page must land first.
  - **MSCCLang's construction gives that graph no cycle.** Every channel's sequence is a subsequence of one global
    topological order (design/plan-to-bridge-change-request.md R5; MSCCLang §5.2). This is the rule of ordered
    resources, which prevents a circular wait (J. W. Havender, Avoiding deadlock in multitasking systems, IBM Systems
    Journal 7(2), 1968; E. G. Coffman, M. J. Elphick and A. Shoshani, System deadlocks, ACM Comput. Surv. 3(2), 1971).
  - **The permutation keeps that order when it is one renaming σ of the set's members**, over the set's positions on
    every channel the set crosses: every hop, every relaying node, every parallel cable. It also covers the set's
    consumer tasks, since an asynchronous, incomplete, noncontiguous consumer takes its pages in any order (R4) and the task at the set's i-th place takes the
    member σ puts there. Indexes inside shapes need not be compiled:
    > the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'
  - **Why σ keeps a topological order.** A set's members have one producer, one route and a consumer that takes them
    in any order, so σ is an automorphism of the task graph. An automorphism maps every edge u → v to an edge σ(u) →
    σ(v), so it maps each topological order onto a topological order. (Formal argument, this document's.)
  - **So the deal fixes positions, not members.** Each channel keeps the positions the deal gives it. In each
    execution, the member at each position is the one σ puts there. The bridge at each node holds a record for every
    member of every set whose positions it carries, each record a view of that member's page (R1). So any member can go
    at any of the set's positions.
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
      phase of the page's last landing. The implementation's PM2 reads the phase field
      there, so its recorded limit ("while the page's earliest round not landed is at most three rounds before the one
      being posted") is the field used outside its precondition. It is not a bound to find or to widen. Claude's "a
      phase word wide enough for the lag, or a cited bound" was idiomatic.
    - For a reader that nothing keeps within one phase, C3 already names the structure: "Counted completion with rising
      thresholds (Portals 4 counting events; `MTLSharedEvent`'s 64-bit value, which only rises). Execution t waits for
      the count to reach its own target." Each range's landed count of each copy is such a count, which every landing
      raises, by whichever cable it lands (C3.1, counted completion). A relay's receive has one too. (Corrected
      2026-11-05: the text before said "The plan keeps such a count already" and that the test "adds no word". A
      relay's receive had no count word (rdma/mesh.h at abecd389), and the implementation added one:
      design/compilation-coordination.md, "The revisions of abecd389", RC-1.)
    - By C3.3, every landing of a copy's round j comes after every landing of its round j − 1. Each landing is a write
      of execution t + u, and every read of execution t, each after its own page's landing, comes before that write.
    - So a page is named for round j where its range's count of its copy reads at least j times the pages this node
      receives of that range in a copy. Then every page of round j − 1 has landed, its own among them. The test holds
      at any distance. It waits for the range's whole round j − 1, not only its own page's.
    - The code must name a page for round j by that count, in place of PM2's test of the phase field. The phase
      fields stay, for C3.7's readers.
  - The counter is (execution, set), so a set is drawn once per execution. A draw for each cable departs from the
    counter.
  - (Corrected 2026-11-05: the text before said each cable's sequence is the same renaming. That fails where the deal
    gives a set's positions to several cables of one pair and each cable keeps only its own members (the
    implementation's M1). The coordination's record, "The revisions of 46ae8fdf", stop 3, gives the cycle.)

- **A synchronous, complete, contiguous consumer's operands** (joins) keep the schedule's order.
- **Every program declares each consumer's kind.** A consumer that reduces partial sums, or otherwise takes pages in
  any order, is an asynchronous, incomplete, noncontiguous consumer (R4). A program that declares every consumer synchronous, complete and contiguous avoids demonstrating this
  requirement. For the program the code runs today, tools/prepare_e2b.py does the declaring:
  > then your code isn't correct for deliberately avoiding demonstrating the requirements of the change request.

  (T233, of the same avoidance in the first implementation.)

### 6. Compiled functions coexist

> "- In the literature, compiled functions are executables that coexist and are launched by calls: XLA and JAX's
> compiled-function cache, CUDA graphs instantiated once and launched repeatedly, "

- **Decision without a citation:** a client's plan is a singleton. Planning again closes every earlier plan (the
  coordination's C-epoch and step 8).
- **The citations:**
  - A compiled function is an executable, compiled once and launched by each call, and many coexist. JAX traces a
    function anew only "on a trace cache miss" (R. Frostig, M. J. Johnson and C. Leary, Compiling machine learning
    programs via high-level tracing, SysML 2018). CUDA Graphs: a graph instantiated once and launched many times
    (NVIDIA, CUDA C++ Programming Guide).
  - Pathways dispatches each call to its compiled function (§3, §4.4).
- **The code must:**
  - let a program hold any number of compiled functions at once, each its own plan with its own claim;
  - release a function's claim only when the program retires that function or itself;
  - (Corrected 2026-11-05: each plan is placed with every other plan of the program by the one allocation over the
    whole exported program (A1; disjoint-pages C3.2 and C3.4 as corrected). There is no claim per plan, and nothing
    takes pages at run time. A function's pages are released when the program retires it or itself, by deferred
    reclamation, as R8 cites.)
  - have each call launch one execution of its function;
  - name each compiled function alike at every rank: (the program's group, the function's ordinal in the program's
    order of compilation). Every rank runs the same calls in the same order, so every rank computes the same name with
    nothing exchanged (F. B. Schneider, ACM Comput. Surv., 1990).

### 7. Torch's existing kernels on the GPU

> no inlining new numerical backends instead of writing collective and transfer and so on code which schedule the use of
> existing code, basically.

- **Decision without a citation:** the torch path binds only CPU tensors (`torch.frombuffer`), so its GPU work does not
  run through torch's existing kernels.
- **The citation:** A9: "the GPU's program: the function's own existing kernels, bound to their placed pages. No kernel
  is written by the compilation."
- **The code must:** run each function's existing backend kernels (torch's MPS kernels, or any other backend's) on
  tensors over the placed pages, and write none of its own.
- (2026-11-05, later) **What the torch path adds beyond torch's own work, and removes.**
  > is this a result of strictly reusing an upstream module or inlining? if it's a property of the upstream, all we can
  > do is reduce the use of overcopies and overwriting of intermediates (and so on) beyond what torch algorithmically
  > requires in whatever torch headers we add to change the behavior of these torch functions in our page-using apis...
  - The code is not torch's; it is rdma/mesh_compile.py `_run`:
    - It runs each operator in its functional form (`local.target(*args, **kwargs)`, `torch.cat`, `torch.add`,
      `masked_fill`), so torch's allocator makes each result outside the placed pages.
    - It copies each sent value into its placed pages between two full device synchronizations (`write`:
      `synchronize()`, `target.copy_(source)`, `synchronize()`).
    - It clones each placed output (`torch.clone`).
  - Torch's own kernels take a result made beforehand. "Instead of allocating returned tensors in kernel
    implementations, an operator's out variant will take in a pre-allocated tensor to its out kwarg, and store the
    result there. This makes it easier for memory planners to perform tensor lifetime analysis. In ExecuTorch, an out
    variant pass is performed before memory planning" (ExecuTorch, "Concepts", Out variant). "In ExecuTorch, a memory
    planning pass is run before the graph is saved to flatbuffer. This assigns a memory ID to each tensor and an offset
    in the buffer, marking where storage for the tensor starts" (the same page, Memory planning). Here the memory
    planner is C3's one allocator, which places by lifetime (C3.3).
  - **The code must:**
    - place every value of the node's partitioned graph by the one allocator. A view (a narrow, a slice) is its base's
      pages, and its base lives as long as it does;
    - run each operator as its out variant into its result's placed pages;
    - have each sent value's producer make it in its sent pages, so nothing copies it there. A collective's result is
      the pages its parts land in, so no concatenation copies them (transport by view, design/plan-to-bridge-change-request.md R1);
    - publish with no wait for the device (below);
    - copy no output (below).
  - (2026-11-05, later still)
    > "so torch allocates every result. Torch's own operators also take an out= result tensor, which is upstream API,
    > so no torch headers are needed." so lets do it the torchic style instead of the idiomatic slow style...
    > "- Copies with full syncs: write does synchronize(); target.copy_(source); synchronize() for every sent value.
    > That's ours." yeah lets not do that...
    > "- Clones: every placed output gets a torch.clone. Also ours." yeah lets not do that...
    > revise the issues raised...
    - **A publish waits for nothing.** The text before allowed one device wait a publish point, which is still a
      synchronization of the whole device. Instead, a torch MPS event is recorded after the producer's out-variant
      kernel, the publish is done in that event's notification, and torch's command buffer is committed without waiting.
      This is the event form torch_mesh's allocator.mm used once it stopped calling classic Metal (mesh fba41159, "torch_mesh
      calls no classic Metal"): an action after torch's work so far, done in a torch MPSEvent's notification, by
      `at::mps::getMPSEventPool()->acquireEvent(false, stream)`, `record`, `notify`, then
      `stream->synchronize(at::mps::SyncType::COMMIT)`. Only that event form is cited here. The file's own action, the
      return of what it called a "lease", is not: a "lease" names no canonical structure (C. G. Gray and D. R. Cheriton,
      SOSP 1989, define a lease as a grant limited in time, and nothing there expired), and the coordination's rules
      forbid leases ("No hashing, receipts, credits, gates, filters, locks, leases, slots, rings",
      design/disjoint-pages-revision-coordination.md). Torch 2.14.0's headers state each call: `MPSEvent::record`,
      "records an event on the stream"; `MPSEvent::notify`, "schedules a notifyListener callback for the event";
      `SyncType::COMMIT`, "commit and flush the command buffer" (torch/include/ATen/mps/MPSEvent.h, MPSStream.h). The
      calls are torch's own API, so mesh calls no classic Metal. They need a small Objective-C++ torch extension, built
      by a versioned rule against the node's installed torch, as torch_mesh's `_stream.mm` was (mesh 116b27dc). On the
      CPU device a kernel has ended when its call returns, so the publish follows the call.
    - **No output is copied.** Each output is a view of its placed pages and holds until the plan reuses them, u
      executions later (C3.3). Torch's own compiled mode does the same with its static memory: "Error: accessing tensor
      output of CUDAGraphs that has been overwritten by a subsequent run ... If a tensor is a user-visible output that
      must remain live across generations, manually clone the tensor outside torch.compile()" (torch 2.14.0,
      torch/_inductor/cudagraph_trees.py); "the output from the first invocation will be overwritten by the second
      invocation" (docs, CUDAGraph Trees). A program that keeps an output longer clones it itself, outside the compiled
      function. This is the cudagraphs style A1 quotes.
  - **Which operands are seen as one run (2026-11-05, later).**
    > if i say that tensors whose outputs or inputs cover arbitrary noncontiguous blocks of pages, then all functions
    > which do not explicitly require contiguity at runtime for their exact inputs and outputs do not overconstrain the
    > allocator and data movements by imposing contiguity.
    - A value is a list of pages, each placed anywhere. Contiguity is a type of particular operands, as R4's synchronous, complete, contiguous consumer
      states it (design/plan-to-bridge-change-request.md R4).
    - Torch's MPS operators consume a strided view within one buffer. MPSGraph takes an `MTLBuffer` with a shape, and
      optionally "rowBytes for the fastest moving dimension" (MPSGraphTensorData.h, `initWithMTLBuffer:shape:dataType:`
      and `…rowBytes:`). Torch's MPS backend: "Use gather kernel to solve strides for macOS < 15.0 / Starting with
      macOS 15.0, MPS supports native strides directly in the kernels" (PyTorch v2.11.0,
      aten/src/ATen/native/mps/OperationUtils.mm). A torch tensor is one storage with sizes and strides
      (c10/core/TensorImpl.h, `Storage storage_`; c10/core/StorageImpl.h, one `DataPtr data_ptr_`).
    - So an operand of a torch MPS operator is seen in one strided progression of virtual addresses. Its pages are
      placed anywhere, as every value's are, and the progression is a mapping written when the plan is made, which the
      GPU translates as it does for every buffer (design/plan-to-bridge-change-request.md R2, corrected). A value read
      only by page-wise consumers (transfers, asynchronous, incomplete, noncontiguous consumers) needs no mapping.
  - **What stays, being torch's:**
    - an operator with no out variant, whose result is copied once into its placed pages where it must be there;
    - a kernel's own scratch inside torch;
    - torch's device stream.

### 8. Settled, with no change

- The heartbeat's holders. The operator: "settled."
