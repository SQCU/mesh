# Change request: the ahead-of-time compilation

This change request states what the ahead-of-time compilation must do: what it consumes, the passes it runs, the IR it
carries and the schedule it serializes. design/plan-to-bridge-change-request.md states what the plan gives the bridge.
This document states how a program becomes that plan. Each requirement quotes the operator, then names its method: what
the method is, and where it comes from in the literature. The method is used as its source states it.

The compilation consumes operands that choose what it does not choose itself: where each part of a function runs (A2)
and how much each task and transfer weighs (A3). It contains no chooser for either. A chooser that fills those operands
is defined later, apart from all of this.

## A1. The input is each program's captured dataflow graph, with shapes and types

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
  - For e2b on LiteRT: as now. tools/prepare_e2b.py reads the model's graph, and the recorder captures the delegate's
    commands (metal_recording.m).

## A2. The placement operand: which nodes hold which sections

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

## A3. The weighting operand: what each task and transfer costs

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

## A4. Collectives lower to page transfers over packed trees

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
  - **Lowering.** MSCCLang (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, §4 and
    §5) lowers operations on chunks into an instruction DAG of sends, receives and receive-reduce-copies. A relay's
    forward is its fused receive-copy-send. Here a chunk is a page.
- **Code:** rdma/mesh-collective.c's demands and trees, as now, fed by A2's collectives in place of ranges declared by
  hand.

## A5. Every page is placed ahead of time

- **Method:** design/plan-to-bridge-change-request.md R2, C3's allocator. As now.

## A6. One schedule per function

- **Method:** design/plan-to-bridge-change-request.md R3 to R5. List scheduling by bottom level, with edge scheduling
  under contention, over the weights of A3.

## A7. Function executions of programs that run at once are ordered by one scheduler for the island

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

## A8. The IR operand

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

## A9. The serialized operand, in MSCCL-IR's shape

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
    compilation. For e2b these are the recorder's command buffers, as now.
- The bridge consumes the lists and the GPU runs the existing kernels. Nothing else is serialized.

## What judges it

- design/plan-to-bridge-change-request.md's judges hold.
- The compilation contains no chooser of placement and no chooser of weights; both arrive as operands.
- No kernel is written by the compilation.
- The three builds are clean. Runs come only when the operator authorizes them.
