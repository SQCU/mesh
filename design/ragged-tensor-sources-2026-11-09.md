# Ragged tensors and unequal shards: an algorithm or data structure for each feature, quoted verbatim

2026-11-09. Each quotation below was taken from the source in this session: from the page itself for standards,
documentation and issues, and from the paper's PDF for papers. The form follows the other `-sources-` documents of
this directory. It replaces the marks **[H]** and **[S]** of
[ragged-computation-sources-2026-11-09.md](ragged-computation-sources-2026-11-09.md) for the features it covers.

## The turn

The operator, 2026-11-09:

> cite me some algorithms covering every feature which would need to be implemented to produce the ragged tensor results the requirements documents implied; you can do this immediately and you would thus have a user turn which produces a list of citations that can be verbatim quoted, wouldn't you?

## How the words were obtained

- **Pages** (MPI 4.1, TensorFlow, Apache Arrow, the two PyTorch issues, GSPMD's HTML rendering): fetched by a
  subagent that returned quotations. For MPI, Arrow, GSPMD and the issues it kept each quotation under about 125
  characters and split longer sentences into consecutive quotations; where two quotations stand side by side below,
  they are consecutive pieces of one sentence. The TensorFlow quotations are whole sentences.
- **Papers of sections 3–5 and 11** (HAP, Megatron-LM, Striped Attention, Blink, Lastovetsky et al. 2011): the arXiv PDF was saved, its
  text extracted with `pypdf`, and the passages copied from that text. Line-break hyphens were joined and ligatures
  replaced, which can alter a word that really ends in a hyphen. Check a quotation against the PDF before it goes
  into a change request.
- Claude did not open any source that is listed under "Not obtained".

## 1. Unequal shards of one dimension, one size per rank

**pytorch/pytorch#169320, "[DTensor] A new placement: RaggedShard"**, https://github.com/pytorch/pytorch/issues/169320

- "The ``RaggedShard`` placement specifies DTensor sharding based on tensor memory continuity."
- "`RaggedShard` is defined by two parameters, `dims` and `local_units`"
- `local_units`: "A tuple of integers specifying the relative allocation of elements across shards."
- "For example, ``local_units=(1, 2, 1, 1)`` assigns:" "- 20% of elements to device 0" "- 40% to device 1" "- 20%
  each to devices 2 and 3"
- "The length of the tuple must equal ``mesh.size(dim)``."
- "The existing Shard placement only supports even sharding and does not meet the needs of many modern applications"
- The signature it shows: `class RaggedShard(Placement): dims: tuple[int, ...]; local_units: tuple[int, ...]`

The issue gives no formula for the sizes and does not say how a remainder is handled.

## 2. Rows of differing length within one tensor

**TensorFlow, `tf.RaggedTensor`**, https://www.tensorflow.org/api_docs/python/tf/RaggedTensor

- "A `RaggedTensor` is a tensor with one or more *ragged dimensions* , which are dimensions whose slices may have
  different lengths. For example, the inner (column) dimension of `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is
  ragged,"
- Under "Component Tensors": "A `values` tensor, which concatenates the variable-length rows into a flattened list.
  For example, the `values` tensor for `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is `[3, 1, 4, 1, 5, 9, 2, 6]`."
- "A `row_splits` vector, which indicates how those flattened values are divided into rows. In particular, the
  values for row `rt[i]` are stored in the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`."
- Under "Alternative Row-Partitioning Schemes": `row_lengths`: "a vector with shape `[nrows]`, which specifies the
  length of each row." `value_rowids`: "a vector with shape `[nvals]`, corresponding one-to-one with `values`, which
  specifies each value's row index." `row_starts`: "a vector with shape `[nrows]`, which specifies the start offset
  of each row. Equivalent to `row_splits[:-1]`." `row_limits`: "a vector with shape `[nrows]`, which specifies the
  stop offset of each row. Equivalent to `row_splits[1:]`."
- Under "Multiple Ragged Dimensions": "`RaggedTensor`s with multiple ragged dimensions can be defined by using a
  nested `RaggedTensor` for the `values` tensor. Each nested `RaggedTensor` adds a single ragged dimension."
- "`rt.nested_row_splits` is a tuple containing the `row_splits` tensors for all ragged dimensions in `rt`, ordered
  from outermost to innermost."

**Apache Arrow, Columnar Format, "Variable-size List Layout"**, https://arrow.apache.org/docs/format/Columnar.html

- "The List layout is defined by two buffers, a validity bitmap and an offsets buffer, and a child array."
- "Rather than referencing an additional data buffer, instead these offsets reference the child array."
- From "Variable-size Binary Layout", which the List layout refers to: `slot_position = offsets[j]`,
  `slot_length = offsets[j + 1] - offsets[j]`; "Offsets must be monotonically increasing, that is
  `offsets[j+1] >= offsets[j]` for `0 <= j < length`, even for null slots."; "The offsets buffer contains
  `length + 1` signed integers"
- Its example for `List<Int8>`, values `[[12, -7, 25], null, [0, -127, 127, 50], []]`: offsets `0, 3, 3, 7, 7`;
  child values `12, -7, 25, 0, -127, 127, 50`.
- "The ListView layout is defined by three buffers: a validity bitmap, an offsets buffer, and an additional sizes
  buffer."

## 3. Choosing the shares

**Zhang et al., "HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis", EuroSys
2024, arXiv:2401.05965**

- Abstract: "We derive the optimal tensor sharding ratios by formulating it as a linear programming problem."
- §2.4, "Optimal Sharding Ratios": "In these communication operations, the devices send and receive tensor shards of
  sizes proportional to the sharding ratios. The communication time therefore depends on the size of the largest
  shard. Minimal communication time is achieved when the tensors are sharded evenly, as the size of the largest
  shard is minimized in this case."
- §2.4: "Fig. 2 shows that when the computation time dominates, CP performs better as it balances computation time on
  different devices; EV performs better when communication is the bottleneck."
- §2.4: "Further, since different layers of a model may exhibit different computation-to-communication ratios, the
  optimal sharding ratios may vary for each layer. In HAP, we formulate sharding ratio optimization as a linear
  programming problem to determine the optimal ratios for each part of the model."

**Lastovetsky, Reddy, Rychkov and Clarke, "Design and implementation of self-adaptable parallel algorithms for
scientific computing on highly heterogeneous HPC platforms", 2011, arXiv:1109.3074**

- §1: "The fundamental assumption of the conventional algorithms based on the constant performance models (CPMs) is
  that the absolute speed of the processors does not depend on the size of the computational task."
- Abstract: "New class of algorithms based on the functional performance model (FPM), representing the speed of the
  processor by a function of problem size, has been recently proposed."
- The problem statement (page 3): "A set of p (p < n) processors P1,…,Pp, whose speeds of processing x units,
  si = si(x), can be obtained by measuring the execution time, ti(x), of a computational kernel, si(x) = x/ti(x),"
  "A required relative accuracy of the solution, ε; To do: partition the set of computation units into p subsets"
  and, after its condition on the times, "Thus, the problem we study is to balance the load of heterogeneous
  processors with a given accuracy." The condition itself is an inequality whose symbols the text extraction
  scrambled; read it in the PDF.

## 4. Shares for work that is not proportional to a share's size

**Brandon et al., "Striped Attention: Faster Ring Attention for Causal Transformers", 2023, arXiv:2311.09431**

- Abstract: "identify a key workload imbalance due to triangular structure of causal attention computations."
- Abstract: "Instead of devices having contiguous subsequences, each device has a subset of tokens distributed
  uniformly throughout the sequence, which we demonstrate leads to more even workloads."
- §1: "we permute the sequence so that every device always operates on a discontiguous subset of tokens distributed
  approximately uniformly throughout the original sequence. This ensures that approximately half of the query/key
  interactions on each device will be inhibited by the causal mask."
- §1: "we exploit the permutation equivariance of the core attention computation to ensure that our decision to
  internally permute the input sequence does not affect the final output of the model."

## 5. Tensors that meet in one operation are split alike

**Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", 2019,
arXiv:1909.08053, §3**

- "Another option is to split A along its columns A = [A1, A2]. This partitioning allows the GeLU nonlinearity to be
  independently applied to the output of each partitioned GEMM"
- "Hence, we partition the first GEMM in this column parallel fashion and split the second GEMM along its rows so it
  takes the output of the GeLU layer directly without requiring any communication as shown in Figure 3a. The output
  of the second GEMM is then reduced across the GPUs before passing the output to the dropout layer."

## 6. Sending and receiving blocks of unequal size

**MPI Forum, *MPI: A Message-Passing Interface Standard*, version 4.1, chapter 7, "Collective Communication"**
(the HTML report numbers the chapter 7; `design/heterogeneity.md` cites it as chapter 6)

- §7.5 "Gather", https://mpi-forum.org/docs/mpi-4.1/mpi41-report/node122.htm: "MPI_GATHERV extends the functionality
  of MPI_GATHER by allowing a varying count of data from each MPI process". `recvcounts`: "nonnegative integer array
  (of length group size) containing the number of elements that are received from each MPI process". `displs`:
  "Entry i specifies the displacement relative to recvbuf at which to place the incoming data from MPI process i".
- §7.6 "Scatter", node124.htm: "MPI_SCATTERV is the inverse operation to MPI_GATHERV." `sendcounts`: "nonnegative
  integer array (of length group size) specifying the number of elements to send to each rank". `displs`: "Entry i
  specifies the displacement (relative to sendbuf) from which to take the outgoing data to MPI process i".
- §7.7 "Gather-to-all", node126.htm: "The block of data sent from the j-th MPI process is received by every MPI
  process" "and placed in the j-th block of the buffer recvbuf." For `MPI_ALLGATHERV`: "If comm is an
  intra-communicator, the outcome is as if all MPI processes executed calls to"
  "MPI_Gatherv(sendbuf,sendcount,sendtype,recvbuf,recvcounts,displs, recvtype,root,comm), for root = 0, ..., n-1."
- §7.8 "All-to-All Scatter/Gather", node128.htm: "MPI_ALLTOALLV adds flexibility to MPI_ALLTOALL in that the location
  of data for the send is specified by sdispls" "and the location of the placement of the data on the receive side
  is specified by rdispls." `sdispls`: "Entry j specifies the displacement (relative to sendbuf) from which to take
  the outgoing data destined for MPI process j". `rdispls`: "Entry i specifies the displacement (relative to
  recvbuf) at which to place the incoming data from MPI process i".
- §7.10.2 "MPI_REDUCE_SCATTER", node140.htm: "The resulting vector is treated as n consecutive blocks where the
  number of elements of the i-th block is recvcounts[i]." "The i-th block is sent to MPI process i and stored in the
  receive buffer defined by recvbuf, recvcounts[i] and datatype."

**pytorch/pytorch#198344, "[c10d][NCCL] Uneven all_gather_into_tensor / reduce_scatter_tensor"**,
https://github.com/pytorch/pytorch/issues/198344

- "Add split-size-aware variants of `all_gather_into_tensor` and `reduce_scatter_tensor`" "with per-rank split sizes
  along dim 0."
- The alternatives it names and their cost: "Pad every rank's shard to the largest one and use the uniform
  collectives: wastes" "memory and bandwidth proportional to the imbalance"; "Keep using the list variants with
  coalesced broadcast/reduce (what we do now):" "cannot use symmetric memory and issues `world_size` small kernels
  per" (the quotation ends where the subagent cut it).

**HAP, §2.2 "Collective Communication"**

- "For instance, NCCL [18] requires all tensors to be of the same size for All-Gather. In order to perform this
  communication operation on unevenly sharded tensors, the tensor shards must be first padded to the same size and
  subsequently trimmed upon completion of the operation, resulting in wasted bandwidth and extra memory access.
  Alternatively, All-Gather can be implemented with multiple Broadcast operations to support unevenly sliced tensor
  shards without padding, at the cost of a higher kernel launching overhead."
- §2.5.1 is titled "Padded All-Gather and Grouped Broadcast"; its text was not copied here.

## 7. Every rank holds the same counts

**MPI 4.1**, the same pages:

- §7.5: "The type signature implied by sendcount, sendtype on MPI process i must be equal to the type signature
  implied by" "recvcounts[i], recvtype at the root."
- §7.7: "The type signature associated with sendcount, sendtype, at MPI process j must be equal to the type
  signature" "associated with recvcounts[j], recvtype at any other MPI process."
- §7.8: "The type signature associated with sendcounts[j], sendtype at MPI process i must be equal to the type
  signature" "associated with recvcounts[i], recvtype at MPI process j."
- §7.10.2: "The routine is called by all group members using the same arguments for recvcounts, datatype, op and
  comm."

## 8. Fixed shapes while shards differ: pad, and mask the padding

**Xu et al., "GSPMD: General and Scalable Parallelization for ML Computation Graphs", 2021, arXiv:2105.04663,
§4.1 "Static constraints", paragraph "Static shapes."** (`design/heterogeneity.md` cites this as §3.3)

- "It's not always the case that all partitions have the same input/output shapes" "dimensions may not be evenly
  divisible by the number of partitions."
- "GSPMD rounds up the size of the shape to a multiple of partition count, and the data in that padded region can be
  arbitrary."
- "When creating certain partitioned operators, data in the padding area needs to be masked off."
- "For example, a reduce operator needs to prevent the padding data from affecting the result" "GSPMD first replaces
  them with the identity value of the reduction."
- "Masking padded data can be expressed as a Select according to a mask" "calculated by comparing Iota and an offset
  based on PartitionId."
- "The amount of padding can vary between different partitions."

## 9. Changing from one sharding to another

**GSPMD, §4.5 "Resharding"**

- "If the provided shardings are not the typical supported cases, the partitioner will perform" "resharding on the
  inputs and/or outputs."
- "It uses AllGather to replicate data along sharded dimensions, AllToAll to switch sharded dimensions,"
  "CollectivePermute to change device order, and DynamicSlice to shard replicated dimensions."

## 10. A window that crosses the boundary between shards of different sizes

**GSPMD, §4.3 "Halo exchange with dynamic bounds"**

- "We use the CollectivePermute operator to exchange halo data between partitions."
- "Non-constant halo size.": "GSPMD uses maximum halo size across partitions, then uses DynamicSlice to remove
  excessive data in the halos."

## 11. Collectives over links of different cost

**Wang et al., "Blink: Fast and Generic Collectives for Distributed ML", 2020, arXiv:1910.04940**

- Abstract: "a collective communication library that dynamically generates optimal communication primitives by
  packing spanning trees. We propose techniques to minimize the number of trees generated and extend Blink to
  leverage heterogeneous communication channels for faster data transfers."
- §3.2 "Approximate Packing": "Our specific use of MWU here follows a recently proposed algorithm to achieve
  near-linear time approximation for fractional packing problems [5]. For the case of packing spanning trees, this
  approach finds a (1− ε)-approximation in O(mlnm/ε2), where m is the number of edges."

## 12. The order of functions on processors of different speeds

**Topcuoglu, Hariri and Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing",
IEEE TPDS 13(3), 2002.** The PDF was obtained from
https://openaccess.marmara.edu.tr/bitstreams/d1584f1e-5066-47a6-8c04-15607a53960f/download by the agent that wrote
`~/schedule-emit`.

- §2, "Task-Scheduling Problem": "W is a v × q computation cost matrix in which each w_{i,j} gives the estimated
  execution time to complete task n_i on processor p_j. Before scheduling, the tasks are labeled with the average
  execution costs."
- §2, eq. (5) and (6): "EST(n_i, p_j) = max{ avail[j], max_{n_m ∈ pred(n_i)} (AFT(n_m) + c_{m,i}) }", "EFT(n_i, p_j)
  = w_{i,j} + EST(n_i, p_j)", and "The inner max block in the EST equation returns the ready time, i.e., the time
  when all data needed by n_i has arrived at processor p_j."
- §4.1, eq. (8): "The upward rank of a task n_i is recursively defined by rank_u(n_i) = w̄_i + max_{n_j ∈ succ(n_i)}
  (c̄_{i,j} + rank_u(n_j)), where succ(n_i) is the set of immediate successors of task n_i, c̄_{i,j} is the average
  communication cost of edge (i, j), and w̄_i is the average computation cost of task n_i."
- §4.2: "Task Prioritizing Phase. This phase requires the priority of each task to be set with the upward rank value,
  rank_u, which is based on mean computation and mean communication costs. The task list is generated by sorting
  the tasks by decreasing order of rank_u. Tie-breaking is done randomly."
- §4.2: "It can be easily shown that the decreasing order of rank_u values provides a topological order of tasks,
  which is a linear order that preserve the precedence constraints."
- §4.2: "the HEFT algorithm has an insertion-based policy which considers the possible insertion of a task in an
  earliest idle time slot between two already-scheduled tasks on a processor."
- Fig. 2, lines 6–8: "for each processor p_k in the processor-set (p_k ∈ Q) do" "Compute EFT(n_i, p_k) value using
  the insertion-based scheduling policy." "Assign task n_i to the processor p_j that minimizes EFT of task n_i."

## 13. Waiting, and storage, between processors under a compiled order

**Bhattacharyya, Sriram and Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", IEEE TSP 45(6), 1997.**
From the Berkeley preprint, https://ptolemy.berkeley.edu/papers/97/synchronization/synch.pdf; its wording and pages
may differ from the published article.

- §1: "Each processor executes the actors assigned to it in the order specified at compile time. Before firing an
  actor, a processor waits for the data needed by that actor to become available."
- §3: "An edge e is a feedforward edge of (V, E) if it is not contained in an SCC; an edge that is contained in an
  SCC is called a feedback edge."
- §4.1: "Thus if actors v_1, v_2, …, v_n are assigned to the same processor in that order, then G_ipc would have a
  cycle ((v_1, v_2), (v_2, v_3), …, (v_{n–1}, v_n), (v_n, v_1)), with delay((v_n, v_1)) = 1."
- §4.3, eq. (5): "B_fb(e) = min({Delay(C) | C is a cycle that contains e})" and "Feedforward edges have no such bound
  on buffer size; therefore for practical implementations we need to impose a bound on the sizes of these edges."

**Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", PLDI 1988, §2.3.** From a scan,
https://fg-kastens.cs.uni-paderborn.de/lehre/pgrekonf/seminar/Lam1988.pdf.

- "Modulo variable expansion takes advantage of the flexibility of VLIW machines in scalar computation, and reduces
  the number of locations allocated to a variable by reusing the same location in non-overlapping iterations."
- "The lifetime of a register variable is defined as the duration between the first assignment into the variable
  and its last use. If the lifetime of a variable is l, and an iteration is initiated every s cycles, then at least
  ⌈l/s⌉ number of values must be kept alive concurrently, in that many locations."

The quotations of sections 12 and 13 were transcribed by that agent from the page images it had read earlier in the
session. It rendered mathematics in plain text and joined words hyphenated across lines, and it did not re-open the
files to proofread. Check them against the PDFs before they go into a change request.

## Not obtained

No words were taken from these, which `design/heterogeneity.md` cites; each is behind a publisher's page or was not
fetched:

- Beaumont et al. (2001), Alg. 3.1: integer shares that minimise the largest finish time.
- Ibaraki and Katoh (1988): that greedy allocation is optimal for nondecreasing costs.
- Bitran and Hax (1981): variables held at their bound and the rest solved again.
- Lastovetsky and Reddy (2007): partitioning with a functional performance model.
- Träff et al. (2010): the pipelined irregular all-gather.
- LB-BSP (2020), Meta-Balancer (2012): when shares change.
- Blelloch and Sabot (1990): flattening nested parallel work.

Until one of the first two is read, section 3 has HAP's linear program and the 2011 statement of the problem, and
no quoted algorithm for rounding shares to whole grains.

## Features for which no source was found

- **Sharding a tensor that already has rows of differing length** (section 2) across ranks: which rank holds which
  rows and which part of `row_splits`. None of the sources above says.
- **Masking (section 8) together with blocks of unequal size (section 6):** GSPMD pads so that every block is the
  same size; MPI's irregular collectives send blocks at their own sizes. No source above says how the two are used
  in one program.
- **`RaggedShard`'s sizes:** the issue states proportions by example and gives no rule for a remainder.
