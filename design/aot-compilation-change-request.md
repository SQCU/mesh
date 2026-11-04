# Change request: the ahead-of-time compilation

This change request states what the ahead-of-time compilation must do: what it consumes, the passes it runs, its IR, and
the schedule it serializes. design/plan-to-bridge-change-request.md states what the plan gives the bridge. This document
states how a program becomes that plan. Each requirement quotes the operator, then names its method: what the method is,
and where it comes from in the literature. The method is used as its source states it.

## A1. The input is each program's captured dataflow graph, with shapes and types

> the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'

> Our PyTorch code can already record a whole function as a graph of steps before it runs. The graph shows which results
> each step needs.

(data-model §6)

- The compilation must take each program as a dataflow graph of operations whose tensors have their shapes and element
  types, and nothing at run time may change a shape.
- **Method:**
  - For a torch function: TorchDynamo's capture of the function as an FX graph, handed to a `torch.compile` backend (J.
    Ansel et al., PyTorch 2: faster machine learning through dynamic Python bytecode transformation and graph
    compilation, ASPLOS 2024).
  - For a program run through a delegate (e2b on LiteRT): the recorder's capture of its commands, as now
    (metal_recording.m).
- **Consumed:** each operation's operands and results, with their element types and shapes.

## A2. Sharding is an annotation on tensors; the compiler partitions the rest

> i want my arbitrary functions running by arbitrary tensor parallelism sharding approaches on arbitrary collectives as
> fast as possible

> this is one of the only important unimplemented features; we'll need to decide ahead of time how we are going to
> handle all of the different placement and scheduling reuqirements that appear at higher collective scales and link
> counts

(of "choosing tensor-parallel placement by topology")

- The compilation must take how the program's tensors are split over the nodes as annotations on a few tensors. It must
  infer every other operation's split, and insert the collectives the split requires.
- **Method:**
  - GSPMD (Y. Xu et al., GSPMD: general and scalable parallelization for ML computation graphs, arXiv 2105.04663, 2021):
    a program written for one device, with a few tensor sharding annotations. Sharding is propagated to every operator
    by compiler passes over a dataflow IR. The output is one SPMD program per device with the collectives inserted.
  - PyTorch's distributed tensor placements (Shard, Replicate, Partial; PyTorch documentation,
    `torch.distributed.tensor`) are the same annotation in torch's terms.
  - The device mesh is the map's nodes: "Placement and topology are data" (metal-microbench AGENTS.md).
- **Consumed:** the annotations, and the map (its nodes, cables and their rates; mesh-map.h).

## A3. Collectives lower to page transfers, routed jointly over the fabric

> the point of having aot schedulign and compilation of schedules is to allow us to interleave any combination of
> transport requirements concurrently with one another, s.t. we are never underusing the bandiwdth on any cable routing
> when we could counterfactually swizzle together more communications to cause pipeline-blocking operations to begin or
> conclude faster, no matter waht they are!

(of "route them jointly with the ported concurrent-flow solve (Garg–Könemann), rather than each in isolation")

> routing within tb5 linked collectives is supported; attempting to involve nodes with no tb5 dma connection to any node
> within the collective is not allowed.

- The compilation must lower every collective to sends and receives of pages, and route all of them, of every program,
  jointly over the Thunderbolt-linked nodes of the map.
- **Method:**
  - MSCCLang's lowering (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, §4 and
    §5): a collective traced as operations on chunks into a chunk DAG, then lowered to an instruction DAG of sends,
    receives and receive-reduce-copies. A relay's forward is the fused receive-copy-send. Here a chunk is a page.
  - Routes come from concurrent multicommodity flow over the map's cables (N. Garg and J. Könemann, Faster and simpler
    algorithms for multicommodity flow and other fractional packing problems, FOCS 1998). It is solved over every
    transfer of every program at once, as rdma/mesh-graph.h `mesh_graph_pack` does now.
- **Code:** rdma/mesh-collective.c's demands and trees, as now, fed by A2's collectives in place of ranges the program
  declares by hand.

## A4. Every page is placed ahead of time

- **Method:** design/plan-to-bridge-change-request.md R2, the page table bound statically by C3's allocator. As now.

## A5. One schedule in one global topological order

> uh, yes, that is the subtext and also the literal textual goal of what we're working on here with this subinterface

(of "static scheduling with costs")

- **Method:** design/plan-to-bridge-change-request.md R3 to R5. The task graph is list-scheduled by HEFT's upward rank,
  with edge scheduling under contention.
- **Its consequence, cited:** each cable's sequence is a subsequence of one global topological order, so the sequences
  cannot wait on each other in a cycle. MSCCLang states this for its thread blocks: "By assigning instructions to thread
  blocks in a topological order that respects communication and processing edges, all implicit dependencies introduced
  by thread block sequential execution cannot produce cycles so that the MSCCL-IR does not have deadlocks" (Cowan et
  al., 2023, §5.2).
- **Consumed:** each operation's cost by the roofline, from its shapes and types: the larger of its flops over the
  node's peak and its bytes over the node's memory bandwidth (S. Williams, A. Waterman and D. Patterson, CACM 2009),
  with data-model §1's measured rate.

## A6. Programs that run at once are ordered by one scheduler for the island

> do programs do nothing but what the allocator lets them, with the shapes and coarse bin packing requierments of
> allocation known aot?

- The compilation must order the transfers of every program that runs at once on the same nodes in one order, the same
  at every node.
- **Method:** Pathways (P. Barham et al., Pathways: asynchronous distributed dataflow for ML, MLSys 2022).
  - §2: "the system will deadlock if communicating computations are not enqueued in a consistent order", and "the lack
    of a centralized scheduler in single-controller systems like TF v1 makes it impossible to ensure consistent ordering
    between computations across programs."
  - §4.4: "The PATHWAYS runtime includes a centralized scheduler per island that consistently orders all of the
    computations in the island."
  - Ahead of time, the island's scheduler composes the programs' task graphs into one DAG (H. Zhao and R. Sakellariou,
    Scheduling multiple DAGs onto heterogeneous systems, IPDPS 2006) and schedules it once (A5).
  - The island is the map's Thunderbolt-linked component (A3).
- **For the operator:** which process is the island's scheduler. Pathways runs it as a service per island, and neither
  the turns nor the paper say where it runs on this fabric. This answers design/plan-to-bridge-change-request.md R8's
  open point down to that choice.

## A7. The IR is the task graph of page instructions

- The compilation must carry one IR from partitioning to serialization.
- **Method:** the instruction DAG of MSCCLang (Cowan et al., 2023, §4) is the task graph of the scheduling literature
  (O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007): computations as tasks, and page sends and receives as
  communication edges. A2 produces it, A3 lowers its collectives, A4 places its values, and A5 and A6 order it.

## A8. The output is the serialized schedule, in MSCCL-IR's shape

- **Method:** MSCCL-IR (Cowan et al., 2023, §5.2): "a tree data structure that divides a collective into individual GPU
  programs that are subdivided into thread blocks containing a list of instructions that are sequentially executed".
  Instructions that depend on another list carry that dependency explicitly ("a dep modifier").
- **Here, per node:**
  - for each adjacent node and direction, the ordered list of page instructions, each with its page, its bytes and its
    dependency: a send's supplied word or a forward's receive phase, or a receive's phase and count. These are struct
    mesh_hop's lists of the plan (design/plan-to-bridge-change-request.md R5), written into the region;
  - the page table's bindings and the plan's state section (C3);
  - the GPU's program: the captured program's own kernels, bound to their placed pages. For e2b these are the recorder's
    command buffers, as now.
- The bridge consumes the lists and the GPU runs the program. Nothing else is serialized.

## What judges it

- design/plan-to-bridge-change-request.md's judges hold.
- An annotated torch function and e2b both compile through A1 to A8.
- Every node's lists come from one schedule.
- The three builds are clean. Runs come only when the operator authorizes them.
