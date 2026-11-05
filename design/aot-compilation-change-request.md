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

> write a change request document which covers all of the requirements and staes of affaires described by the user,
> the narrowness of scope of the affected data structures and functions, and write tshis as a document insdie of the
> docs or whatever of the corresponding repos, not as some wacky private note-for-subagents. the verbatim notes
> discusisng this with the user would be fine. no embellishments are needed or allowed.

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
- **Where the document is silent, the implementer stops at that point and reports it.** It takes no decision of its own.


## The turns

The turns are held once, in design/plan-to-bridge-change-request.md, "The turns", after
design/disjoint-pages-change-request.md D.1. They are this document's body.

## Claude's proposal

This section and the appendices are Claude's (T178). The operator accepted the appendix "the citations the
implementation omitted or ignored": "\"1->8\" acceptable as written here; launch revisions". Each clause is the
operator's words, a cited source's own statement, or a fact of the code.

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
    dataflow IR, and emits one SPMD program per device with the collectives inserted (Y. Xu et al., GSPMD: general and
    scalable parallelization for ML computation graphs, arXiv 2105.04663, 2021).
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

- The compilation must lower every collective A2 inserts to sends and receives of pages. It routes all of them, of every
  function, jointly over the Thunderbolt-linked nodes of the map.
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
  - **AllGather** is one broadcast from each source over its packed trees (Blink §3.1). **AllToAll** is point-to-point
    demands, one per ordered pair of ranks, routed jointly as concurrent multicommodity flow, the problem Garg and
    Könemann's rounds solve (N. Garg and J. Könemann, FOCS 1998).
  - **Lowering.** MSCCLang (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, §4 and
    §5) lowers operations on chunks into an instruction DAG of sends, receives and receive-reduce-copies. A relay's
    forward is its fused receive-copy-send. Here a chunk is a page.
- **Code:** rdma/mesh-collective.c's demands and trees, as now, fed by A2's collectives in place of ranges declared by
  hand.

### A5. Every page is placed ahead of time

- **Method:** design/plan-to-bridge-change-request.md R2, C3's allocator. As now.

### A6. One schedule per function

- **Method:** design/plan-to-bridge-change-request.md R3 to R5. List scheduling by bottom level, with edge scheduling
  under contention, over the weights of A3.

### A7. Function executions of programs that run at once are ordered by one scheduler for the island

> nothing happens when a program joins or leaves because a program is a sequence of like, function calls.

- The code must order the function executions of every program that runs on the same nodes in one order, the same at
  every node.
- **Method:** Pathways (P. Barham et al., Pathways: asynchronous distributed dataflow for ML, MLSys 2022).
  - §2: "the system will deadlock if communicating computations are not enqueued in a consistent order".
  - §4.4: "The PATHWAYS runtime includes a centralized scheduler per island that consistently orders all of the
    computations in the island", by "communicating with the scheduler to determine a consistent order of function
    executions across all programs running on the island". "Our current implementation simply enqueues work in FIFO
    order."
  - The island is the map's Thunderbolt-linked component (A4).
  - Each function execution runs its own compiled schedule (A6).
- **The scheduler is a fixed sequencer.**
  > it is rqeuired to choose nad use a scheduler. cite anyhthing do anything but don't use idiomatic citation free claude
  > code anywhere here
  - It is the fixed-sequencer class of total-order broadcast (X. Défago, A. Schiper and P. Urbán, Total order broadcast
    and multicast algorithms: taxonomy and survey, ACM Comput. Surv. 36(4), 2004).
  - One process, the sequencer, gives every function execution submitted on the island its sequence number, in the
    order the submissions reach it. Pathways' scheduler is first-in, first-out in just this way (§4.4).
  - Every node serves function executions in sequence-number order.
  - A program submits each of its function executions to its node's bridge, and the bridge forwards it to the
    sequencer.
- **The sequencer is the bridge of the island's highest-numbered live node.** That is the bully algorithm's coordinator,
  the highest-numbered process alive (H. Garcia-Molina, Elections in a distributed computing system, IEEE Trans.
  Comput. C-31(1), 1982). Which nodes are alive is the map's, which every bridge holds (mesh-records.c).

### A8. The IR operand

> some operand must clearly carry the ir used to make this kind of optimization / planning choice, and something must
> serialize this into an operand consumedy by ordinary transport / bridge functions?

- The compilation must carry one IR from partitioning to serialization, as an operand of its own.
- **The operand's type:**
  - the task graph of design/plan-to-bridge-change-request.md R3 (Sinnen, 2007);
  - each consumer's kind (R4: a join for function A, a task per page for function B);
  - the placement operand (A2) and the weighting operand (A3).
- **Method:** MSCCLang carries its instruction DAG the same way, from lowering to scheduling (Cowan et al., 2023, §4 and
  §5).
- A2 partitions, A4 lowers, A5 places and A6 orders it.

### A9. The serialized operand, in MSCCL-IR's shape

> no inlining new numerical backends instead of writing collective and transfer and so on code which schedule the use of
> existing code, basically.

- **Method:** MSCCL-IR (Cowan et al., 2023, §5.2): "a tree data structure that divides a collective into individual GPU
  programs that are subdivided into thread blocks containing a list of instructions that are sequentially executed". An
  instruction that depends on another list carries the dependency explicitly ("a dep modifier").
- **Here, per node:**
  - for each adjacent node and direction, the ordered list of page instructions, each with its page, its bytes and its
    dependency word. A send's word is its supplied word, or for a forward its receive's phase; a receive's are its phase
    and count. These are the plan's struct mesh_hop lists (design/plan-to-bridge-change-request.md R5);
  - the page table's bindings and the plan's state section (C3);
  - the GPU's program: the function's own existing kernels, bound to their placed pages. No kernel is written by the
    compilation. For a delegate's program these are the recorder's command buffers, as now.
- The bridge consumes the lists and the GPU runs the existing kernels. Nothing else is serialized.

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
  - MSCCLang uses it for this same decision: "Assuming an α − β communication cost model, each send has a start up α
    cost and a per-byte β cost" (M. Cowan et al., ASPLOS 2023, §5.1).
  - HEFT weighs a communication edge the same way, as a start-up cost plus its data over the link's bandwidth (H.
    Topcuoglu, S. Hariri and M.-Y. Wu, IEEE TPDS 13(3), 2002).
- **The edge weights it gives:** an edge e of n bytes crossing cable k weighs α_k + n β_k.
  - β_k is 1 over cable k's bandwidth, from the map (mesh-map.h).
  - α_k is fitted from measured transfer times of at least two sizes on that cable, as Hockney fits r∞ and n½.
  - Measuring needs runs, which wait for the operator's authorization. Until α_k is measured, it is 0, the model's
    bandwidth-only form.
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
  > nothing happens when a program joins or leaves because a program is a sequence of like, function calls.
  - A program's calls are a sequence, so each plan has at most one call outstanding. The sequencer numbers each call's
    submission (Défago, Schiper and Urbán, 2004), and a call returns only once its executions are done (C3.1).
  - The bridge parent writes, for each plan, its one outstanding call's sequence number and executions into words of
    that plan's state, each with a single writer (C3's rule 2). No word is reused while its call is outstanding, so
    nothing waits to write.
  - Each port serves the outstanding calls of every standing plan in sequence-number order. Both ends of a cable read
    the same numbers, so they serve alike.
  - (Corrected 2026-11-05: the first text wrote one word per copy, a ring whose writer waits when it is full. No
    requirement asks for one, and T210 ruled modulo rings out.)
- **The code must:**
  - deliver orders in sequence order;
  - publish them, and merge them at each port, as above;
  - interleave released executions on each cable;
  - remove Q4's hold and Q4′'s timeout skip.

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

- **What is serialized:** the pages of function B's operands (design/plan-to-bridge-change-request.md R4: one task per
  page, any order) are written as a set, not a sequence. The bridge sees a set and a sequence, and no kind (R4).
- **What the bridge does:** each execution of a set on a cable is sent and received in a uniformly random permutation of
  its pages.
  - The permutation is drawn by Fisher and Yates's shuffle (R. A. Fisher and F. Yates, Statistical Tables, 1938; R.
    Durstenfeld, Algorithm 235: Random permutation, CACM 7(7), 1964; D. E. Knuth, The Art of Computer Programming vol.
    2, §3.4.2, Algorithm P).
  - Its random numbers come from a counter-based generator (J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw,
    Parallel random numbers: as easy as 1, 2, 3, SC 2011). Its key is the plan's name, a new random group each launch
    (design/plan-to-bridge-change-request.md R8), and its counter is (execution, set). So both ends of a cable draw the
    same permutation without exchanging anything.
- **Repeated runs:** across runs, the same function is sent in different, uniformly distributed orders. The pages and
  their bytes are the same each time, so the requirement's mean and variance hold over the set.
- **Function A's operands** (joins) keep the schedule's order.
- **Every program declares each consumer's kind.** A consumer that reduces partial sums, or otherwise takes pages in
  any order, is function B (R4). A program that declares every consumer function A avoids demonstrating this
  requirement. For the program the code runs today, tools/prepare_e2b.py does the declaring:
  > then your code isn't correct for deliberately avoiding demonstrating the requirements of the change request.

  (T233, of the same avoidance in the first implementation.)

### 6. Compiled functions coexist

> "- In the literature, compiled functions are executables that coexist and are launched by calls: XLA and JAX's
> compiled-function cache, CUDA graphs instantiated once and launched repeatedly, "

- **Decision without a citation:** a client's plan is a singleton. Planning again closes every earlier plan (the
  coordination's C-epoch and step 8).
- **The citations:**
  - A compiled function is an executable, compiled once and launched by each call. Many coexist (R. Frostig, M. J.
    Johnson and C. Leary, Compiling machine learning programs via high-level tracing, SysML 2018; NVIDIA, CUDA C++
    Programming Guide, "CUDA Graphs": a graph instantiated once and launched many times).
  - Pathways dispatches each call to its compiled function (§3, §4.4).
- **The code must:**
  - let a program hold any number of compiled functions at once, each its own plan with its own claim;
  - release a function's claim only when the program retires that function or itself;
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

### 8. Settled, with no change

- The heartbeat's holders. The operator: "settled."
