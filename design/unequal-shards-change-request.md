# Change request: unequal shards and ragged tensors

## The turns

The operator, in `design/heterogeneity.md`:

> "it'll take a little bit of orientation for developers (agents as well) to handle heterogenous
> links ontologically (since this is so different) and without lots of clumsy erratic
> overcorrections (this is mostly requirements in the literature, yo)."

The operator, 2026-11-09, in order:

> while youj're at it pull up all of the resources the repo already wrote / designed / solved for problems of ragged tensors, parametrically allocating compute-ragged and scheduling-ragged allocations of work across heterogenous nodes, etcetera, whihc we will ultimately need as part of this 'manifest of solved algorithmic problems whether or not their literal implementations cmopose or comport with our requirements'

> then we must judge the relevant history instead of treating this exact snapshot as authoritative in any particular way.

> instead, synthesize a positive list of definite citations and canonical algorihtms, methods, data structures, etc. which woudl form change-request-documents out of the 'requirements documents' which led to incomplete and self-inconsistent work wrt ragged computation, ragged tensors, etc.

> cite me some algorithms covering every feature which would need to be implemented to produce the ragged tensor results the requirements documents implied; you can do this immediately and you would thus have a user turn which produces a list of citations that can be verbatim quoted, wouldn't you?

> the situation, plainly stated, is that it is time to translate the previous 'requirements' document into something with the grammar and content of a 'change request document'. i expect you to do this now without devising further layers of abstraction or indirection

> find some sources, literature, computer science theory, anything really. we don't need for this to beu nderdescribed or described in idiomatic sourceless language; you can find sources right now

> are any of these requirements optional for the complete implementation of the requirements as code? if any of them are optional, drop them. if any of them are not optional, then ground them in citation

The operator, 2026-11-10:

> " no order in the chain closed." is there a vast citational gap we cannot close or do we need to revise our change request documents to be totally closed as specifications of implementations instead of leaving a bunch of 'stops' or apparent free choices up in the air?
> "which adds two whole-sequence all-gathers per attention'
> hm, stinky, and neither grounded in citation nor a requirement

> review the change requests one more time. here are three major kinds of defects to correct:
> 1: misleading pedantic specificity, where there are 23 numbered and lettered subsections which describe something about a source but don't say what algorithm is being used, what data structure is being used, and where it composes in a DAG of functions and the type interfaces between them.
> 2: misleading pedantic vagueness, where terms with no grounding to algorihtms, data structures, or the setting are introduced. `the plan that gates the flimgrundler isn't handled by the srotheclave executor, so we have to stop for that` would be misleading pedantic vagueness, especially with no canonical data structure, algorithm, or method justifying why the change request pertains to `srotheclaves`, `frimgrundlers`, `plans` or `gates`.
> 3: invented qualitative limits that do not come from the change request's plain semantics and plain use as a schema in the first `3` cases of change request documents used in the repo history.

## The stages

Each requirement composes into the stages of the reference program `~/schedule-emit` at f41c64a
(`schedule_emit/driver.py`, `compile_module`), each a function from a value to a value:

- `export(module, args)` writes an `Exported` (`schedule_emit/ir.py`), whose `graph_module` is a
  `torch.fx.GraphModule`;
- `place(graph_module, placement)` takes that `GraphModule` and the placement operand and writes the placed
  `GraphModule`, the output processor's name and the number of channels. Every node of the placed graph has
  `meta["processor"]`, and a communication is a `send` node and a `recv` node (`schedule_emit/placement.py`);
- `on_devices(graph_module, devices)` takes the placed `GraphModule` and a dict from processor name to `torch.device`
  and writes a `GraphModule` with each node's device;
- `schedule(module, cost)` takes that `GraphModule` and writes a dict from processor name to a `GraphModule` whose
  nodes are in scheduled order;
- the result is one `Scheduled` (`schedule_emit/ir.py`), which lowering, the runtime and the archive each take.

A sharding is part of the placement operand. Items 1, 5, 8, 13 and 15 state what the operand carries for a sharded
dimension. Items 6, 7 and 14 produce it before `place` runs. Items 2, 3, 4, 9, 10 and 12 state nodes that `place` adds
to the graph. Item 11 states the form of a value. The Stage line of each item rests on these functions; no quoted
source of an item names a stage.

The reference program's placement operand is a function from an fx node to a processor name, and has no field for a
sharding. Its operators are `send(value: Tensor, channel: int)` and `recv(channel: int) -> Tensor`, and have no
argument for counts, displacements or a subarray. The types of those values below are their sources': RaggedShard's
`local_units`, a tuple of integers whose length is the size of the mesh dimension; MPI's `recvcounts`, `sendcounts`
and `displs`, each an integer array of length the group size; MPI's `array_of_sizes`, `array_of_subsizes` and
`array_of_starts`, each an array of integers with one entry per dimension of the array.

## Requirements

### 1. A sharded dimension carries one integer per device of the mesh dimension

- **Source:** pytorch/pytorch#169320, "[DTensor] A new placement: RaggedShard",
  https://github.com/pytorch/pytorch/issues/169320:
  - "`RaggedShard` is defined by two parameters, `dims` and `local_units`"
  - `local_units`: "A tuple of integers specifying the relative allocation of elements across shards."
  - "For example, ``local_units=(1, 2, 1, 1)`` assigns:" "- 20% of elements to device 0" "- 40% to device 1" "- 20%
    each to devices 2 and 3"
  - "The length of the tuple must equal ``mesh.size(dim)``."
  - "The existing Shard placement only supports even sharding and does not meet the needs of many modern
    applications"
- **Replaces:** in `rdma/mesh_compile.py`, the one extent ceil(n / P) that every device of a mesh dimension is given
  (`IR.local`, :198; `_reshard`, :717 and :720).
- **Source for whole sizes:** Zhang et al., "HAP", EuroSys 2024, arXiv:2401.05965, §5.1: "After obtaining the optimal
  fractional solutions, we round the sharded sizes of each tensor to integers and ensure they add up to the total
  length of the dimension that is sharded on. We first set the sharded sizes to their nearest integers. If the sum is
  larger or smaller than the original size, we repeatedly reduce/increase the size by one for a shard that introduces
  smallest rounding errors, until the sizes of the sharded tensors sum to the original tensor."
- **Data structure:** a tuple of integers for each sharded dimension, RaggedShard's `local_units`, its length the
  size of the mesh dimension. Equal integers are the even sharding.
- **Algorithm:** HAP §5.1's rounding. It takes the tuple and the dimension's extent. Device j's size is its integer's
  fraction of the tuple's sum, of the extent, set to the nearest integer; while the sizes do not sum to the extent,
  the size whose change adds the least rounding error is reduced or increased by one. It writes one whole size per
  device, the sizes summing to the extent.
- **Stage:** the placement operand carries the tuple, of `local_units`' type. `place` consumes it with the
  dimension's extent and writes the sizes, an integer array of one entry per device, which are the `counts` of items
  2 and 3.

### 2. A collective sends each device's block at its own size

- **Source:** MPI Forum, *MPI: A Message-Passing Interface Standard*, version 4.1, chapter 7 "Collective
  Communication", https://mpi-forum.org/docs/mpi-4.1/mpi41-report/node122.htm and the pages after it:
  - §7.5: "MPI_GATHERV extends the functionality of MPI_GATHER by allowing a varying count of data from each MPI
    process". `recvcounts`: "nonnegative integer array (of length group size) containing the number of elements that
    are received from each MPI process". `displs`: "Entry i specifies the displacement relative to recvbuf at which
    to place the incoming data from MPI process i".
  - §7.6: "MPI_SCATTERV is the inverse operation to MPI_GATHERV." `sendcounts`: "nonnegative integer array (of length
    group size) specifying the number of elements to send to each rank".
  - §7.7: "The block of data sent from the j-th MPI process is received by every MPI process and placed in the j-th
    block of the buffer recvbuf." For `MPI_ALLGATHERV`: "the outcome is as if all MPI processes executed calls to
    MPI_Gatherv(sendbuf,sendcount,sendtype,recvbuf,recvcounts,displs, recvtype,root,comm), for root = 0, ..., n-1."
  - §7.8: "MPI_ALLTOALLV adds flexibility to MPI_ALLTOALL in that the location of data for the send is specified by
    sdispls and the location of the placement of the data on the receive side is specified by rdispls."
  - §7.10.2: "The resulting vector is treated as n consecutive blocks where the number of elements of the i-th block
    is recvcounts[i]. The i-th block is sent to MPI process i and stored in the receive buffer defined by recvbuf,
    recvcounts[i] and datatype."
- **Source for what it replaces:** Zhang et al., "HAP", EuroSys 2024, arXiv:2401.05965, §2.2: "the tensor shards must
  be first padded to the same size and subsequently trimmed upon completion of the operation, resulting in wasted
  bandwidth and extra memory access."
- **Replaces:** in `rdma/mesh_compile.py`, the padding of a dimension to a multiple of the mesh extent before a
  collective: `_reshard` (:715, :745) and `_partition` (:903), through `_partition_pad` (:673).
- **Data structure:** MPI's two arrays, `counts` (MPI's `recvcounts` and `sendcounts`) and `displs`, each with one
  entry per device of the mesh dimension.
- **Algorithm:** MPI 4.1's collectives of varying count, §7.5 to §7.10.2 as quoted. An all-gather, a reduce-scatter,
  an all-to-all, a gather and a scatter over a dimension sharded by item 1 each carry the two arrays. The block of
  device j is sent at `counts[j]` elements and placed at `displs[j]`. A reduce-scatter's result is n consecutive
  blocks, block i of `counts[i]` elements, sent to device i.
- **Stage:** the sends and receives of each of these collectives are `send` and `recv` nodes `place` adds to the
  `GraphModule`, with `counts` and `displs`, each an integer array of one entry per device. `on_devices` and
  `schedule` take the `GraphModule` with those nodes.

### 3. Every device holds the same counts

- **Source:** MPI 4.1, the same pages:
  - §7.5: "The type signature implied by sendcount, sendtype on MPI process i must be equal to the type signature
    implied by recvcounts[i], recvtype at the root."
  - §7.7: "The type signature associated with sendcount, sendtype, at MPI process j must be equal to the type
    signature associated with recvcounts[j], recvtype at any other MPI process."
  - §7.10.2: "The routine is called by all group members using the same arguments for recvcounts, datatype, op and
    comm."
- **Data structure:** the `counts` and `displs` of item 2.
- **Algorithm:** item 1's rounding. The counts of item 2 are computed from the tuple of item 1 and the dimension's
  extent, and every device of the collective is given the same arrays. What device j sends is `counts[j]` elements
  at every device's reading of it.
- **Stage:** `place` writes the arrays once for a collective; that collective's nodes on every device carry them.

### 4. A device's buffer keeps one shape: it is padded, and the padding is masked

- **Source:** Xu et al., "GSPMD: General and Scalable Parallelization for ML Computation Graphs", 2021,
  arXiv:2105.04663, §4.1 "Static constraints", "Static shapes.":
  - "It's not always the case that all partitions have the same input/output shapes" ... "dimensions may not be
    evenly divisible by the number of partitions."
  - "GSPMD rounds up the size of the shape to a multiple of partition count, and the data in that padded region can
    be arbitrary."
  - "When creating certain partitioned operators, data in the padding area needs to be masked off."
  - "For example, a reduce operator needs to prevent the padding data from affecting the result" ... "GSPMD first
    replaces them with the identity value of the reduction."
  - "Masking padded data can be expressed as a Select according to a mask calculated by comparing Iota and an offset
    based on PartitionId."
  - "The amount of padding can vary between different partitions."
- **Replaces:** in `rdma/mesh_compile.py`, `_partition_pad` (:673-681), which emits a pad of a dimension and an
  extent and nothing that masks it.
- **Data structure:** GSPMD §4.1's padded shape and mask. The buffer a device holds for a sharded dimension has one
  size for all devices, rounded up; what lies past a device's own shard is arbitrary. The mask compares the index
  along the dimension (GSPMD's Iota) with an offset computed from the device (GSPMD's PartitionId).
- **Algorithm:** GSPMD §4.1's masking. Before an operator that reduces over that dimension, the elements past the
  device's shard are replaced by the identity of the reduction, by a select on the mask. The padding differs by
  device.
- **Stage:** the select is a node `place` adds before the reducing operator, on each device, from that device's size
  of item 1.

### 5. Tensors that meet in one operation are split alike

- **Source:** Shoeybi et al., "Megatron-LM", 2019, arXiv:1909.08053, §3: "Hence, we partition the first GEMM in this
  column parallel fashion and split the second GEMM along its rows so it takes the output of the GeLU layer directly
  without requiring any communication as shown in Figure 3a. The output of the second GEMM is then reduced across
  the GPUs before passing the output to the dropout layer."
- **Data structure:** the tuple of item 1, one tuple for both weights.
- **Algorithm:** Megatron-LM §3's split. The first weight of a two-layer block is cut along its columns and the second
  along its rows by the same tuple, so the second takes the first's shard without a collective; the second's output
  is reduced across devices.
- **Stage:** the placement operand carries the one tuple on the first weight's column dimension and the second
  weight's row dimension. `place` consumes it and adds the nodes of the reduction.

### 6. The time of a collective is taken at the largest shard, and sharding ratios are chosen for each part of the model

- **Source:** HAP, §2.4 "Optimal Sharding Ratios":
  - "In these communication operations, the devices send and receive tensor shards of sizes proportional to the
    sharding ratios. The communication time therefore depends on the size of the largest shard. Minimal
    communication time is achieved when the tensors are sharded evenly, as the size of the largest shard is
    minimized in this case."
  - "when the computation time dominates, CP performs better as it balances computation time on different devices;
    EV performs better when communication is the bottleneck."
  - "since different layers of a model may exhibit different computation-to-communication ratios, the optimal
    sharding ratios may vary for each layer. In HAP, we formulate sharding ratio optimization as a linear
    programming problem to determine the optimal ratios for each part of the model."
- **Source for the program:** HAP, §5 "Load Balancing", §5.1 "A Base Case":
  - "As a collective communication operation involves all devices and is bottlenecked by the slowest participant, the
    communication time in the i-th stage (Sec. 3.2), comm(i)(B), is decided by the largest communication time of a
    tensor shard, i.e., comm(i)(B) is a linear function of max_{j∈[m]}(B_j)."
  - "the computation time of the i-th stage is the maximum computation time among the devices, i.e., max_j
    comp(i)_j(B_j). We then solve the following problem to obtain B: min Σ_{i∈stages(Q)} (comm(i)(B) + max_{j∈[m]}
    comp(i)_j(B_j)) (3) subject to: Σ_{j=1}^{m} B_j = 1, B_j ≥ 0, ∀j ∈ [m]"
  - "The objective function is the sum of the communication time and computation time of all stages, which is the
    per-iteration training time that we are minimizing. The constraints state that sharding ratios are non-negative
    and sum to 1."
  - "In HAP, the functions comm(i) and comp(i)_j are modeled as linear functions on bandwidth and flops (Sec. 3.2).
    Therefore, the optimization problem is a linear program and can be solved efficiently with off-the-shelf
    solvers."
  - §5.2 "Different Sharding Ratios across the Model": "If the DNN model contains many layers and the
    computation-communication ratio differs across layers, using the same sharding ratios throughout the model may
    not be ideal."
- **Source for the stages and their times** (the passages the quotations above cite as "Sec. 3.2", with line-break
  hyphens joined and the times' sub- and superscripts written inline):
  - HAP, §3.2 "Cost Modeling": "This allows us to divide the execution of a distributed program into stages, with
    each stage starting with a communication operation followed by a series of computation operations (except for
    the first stage which only contains computation), as illustrated in Fig. 6."
  - "comp(i)_j(B_j) can be calculated based on the profiled flops-per-second of the j-th device and the estimated
    number of flops of the computation operations in stage i. Specifically, common operations in DNNs have numbers
    of flops that are linear to some of the dimensions of the input tensors. If one of these dimensions are sharded,
    the number of flops of this operation on a device is proportional to the sharding ratio of this device;
    otherwise, the number of flops does not change. The computation time comp(i)_j(B_j), which is calculated for
    each operator in the i-th stage by dividing the flops with the flops-per-second of the j-th device, is therefore
    a linear function of the sharding ratio B_j."
  - "comm(i)(B) is determined based on the collective operation type, the sharding ratio B, and NCCL's profiling data
    on the cluster's network. We run each collective operation on the cluster with tensors of different sizes and
    fit the latency and bandwidth in a linear model. comm(i)(B) is then estimated using the fitted model, with the
    input of the tensor size of the largest shard."
  - HAP, §5: "the optimal sharding ratios B for a fixed distributed program Q".
  - HAP, §5.2: "The segment division can be either specified by the user (such as using the layers of the model) or
    determined using a partition algorithm such as METIS"; "each stage is entirely within a model segment. In this
    way, for each segment, we can solve the optimization problem in (3) independently to determine the optimal
    sharding ratios for tensors in that segment."
- **Replaces:** `rdma/allocate.py` `min_max`'s `shared` argument (:67), which no caller supplies; and one tuple per
  mesh dimension.
- **Data structure:** HAP's sharding ratios B, one B_j for each of the m devices, one B for each segment of the model
  (HAP §5.2); and HAP §3.2's stages of the program, each with its communication time comm(i)(B) and each device's
  computation time comp(i)_j(B_j).
- **Algorithm:** HAP's linear program (3). The B_j minimise, summed over the segment's stages, the stage's
  communication time, a linear function of the largest B_j, plus the largest over devices of device j's computation
  time at B_j; subject to the B_j being nonnegative and summing to 1. It is solved as a linear program. The whole
  sizes follow by item 1's rounding. A tuple is chosen for each segment of the model, not one for the whole mesh
  dimension.
- **Stage:** the solve runs before `place`. It takes the program's stages and their times and writes, for each
  segment, the tuple of item 1 in the placement operand, which `place` consumes.

### 7. A node's speed is a function of the size of the problem

- **Source:** Lastovetsky, Reddy, Rychkov and Clarke, "Design and implementation of self-adaptable parallel
  algorithms for scientific computing on highly heterogeneous HPC platforms", 2011, arXiv:1109.3074:
  - §1: "The fundamental assumption of the conventional algorithms based on the constant performance models (CPMs) is
    that the absolute speed of the processors does not depend on the size of the computational task."
  - Abstract: "This assumption proved inaccurate for modern and perspective highly heterogeneous HPC platforms. New
    class of algorithms based on the functional performance model (FPM), representing the speed of the processor by
    a function of problem size, has been recently proposed."
  - "A set of p (p < n) processors P1,…,Pp, whose speeds of processing x units, si = si(x), can be obtained by
    measuring the execution time, ti(x), of a computational kernel, si(x) = x/ti(x)"
- **Source for the x at which a speed is first taken** (the same paper, its algorithm's steps 1 and 2): "All the p
  processors execute n / p computation units in parallel." "processor P1
  calculates the absolute speeds of the processors, si(n / p) = (n / p) / ti(n / p) for 1 ≤ i ≤ p and builds the
  first approximation of their functional performance models in the form of constant models, si(x) = si(n / p)".
- **Data structure:** the functional performance model of Lastovetsky et al.: for each node and each kind of
  function, a speed s(x) = x / t(x), t(x) the measured time of that function at x units. It is not one number per
  node.
- **Algorithm:** the measurement of the paper's steps 1 and 2 as quoted: the speed is first taken at x = n / p.
- **Stage:** the speeds are taken before `place`. Item 6's solve consumes them as the speeds its sharding ratios are
  chosen with.

### 8. Under a causal mask, a device holds tokens spread through the sequence

- **Source:** Brandon et al., "Striped Attention: Faster Ring Attention for Causal Transformers", 2023,
  arXiv:2311.09431:
  - "identify a key workload imbalance due to triangular structure of causal attention computations."
  - "we permute the sequence so that every device always operates on a discontiguous subset of tokens distributed
    approximately uniformly throughout the original sequence. This ensures that approximately half of the query/key
    interactions on each device will be inhibited by the causal mask."
  - "we exploit the permutation equivariance of the core attention computation to ensure that our decision to
    internally permute the input sequence does not affect the final output of the model."
  - Section 2.2.1: "permuting the input tokens before the model's first embedding layer".
  - Figure 1, caption: "the tokens in the input sequence are partitioned among devices before running the first layer
    of the model", and they "remain partitioned in the same layout throughout the forward and backward passes", "with
    no extra per-layer communication required to prepare them in this state".
  - Section 2.2.1: "we partition them into sets of evenly-spaced stripes based on their residues modulo the device
    count"; "device 0 would own tokens 0,4,8,…, device 1 would own tokens 1,5,9,…, and so on".
  - Section 2.2.1: "the position ids must be permuted as well"; "we must also permute the sequence of target token
    ids used to compute the loss".
  - Section 2.2.1: the mask "is causal with respect to the order in which tokens appear in the original input
    sequence".
  - Section 3.1: "the algorithm computes the attention interactions for the tile if and only if it contains at least
    one unmasked element".
- **Data structure:** Striped Attention §2.2.1's stripes: device r of N holds the tokens whose index in the original
  sequence is r modulo N.
- **Algorithm:** Striped Attention's permutation. Where attention under a causal mask is sharded along the sequence,
  the input tokens are permuted before the first embedding into the stripes. The sequence keeps that layout through
  every function of the program, forward and backward. Position ids, and the target ids of a loss, are permuted the
  same way. Each attention's mask is causal in the tokens' original order. A block of queries and keys that the mask
  inhibits entirely is not computed.
- **Stage:** the stripes are the sharding of the sequence dimension in the placement operand, which `place` consumes.
  `place` adds no communication at an attention to bring the sequence into or out of this layout.
- **Replaces:** `rdma/mesh_compile.py` `_stripe`'s `index_select` before and `index_copy` after each attention, which
  gather the whole query and the whole output at every attention.
- The source read (the HTML text through section 5.3) does not state: a step that puts a returned per-token result
  back in the original order; the stripes where the devices' counts differ (item 1); the layout where the sequence is
  not the program's input.

### 9. Changing a tensor from one sharding to another

- **Source:** GSPMD, §4.5 "Resharding": "It uses AllGather to replicate data along sharded dimensions, AllToAll to
  switch sharded dimensions, CollectivePermute to change device order, and DynamicSlice to shard replicated
  dimensions."
- **Data structure:** the `counts` and `displs` of item 2.
- **Algorithm:** GSPMD §4.5's resharding by its four operators: AllGather, AllToAll, CollectivePermute and
  DynamicSlice. Each of the four carries the counts of item 2 where its dimension is sharded by item 1.
- **Stage:** the four are nodes `place` adds to the graph.

### 10. A window that crosses the boundary between shards

- **Applies to:** `design/problems-of-14b9a757-change-request.md` problem 4, for shards of item 1.
- **Source:** GSPMD, §4.3 "Halo exchange with dynamic bounds": "We use the CollectivePermute operator to exchange
  halo data between partitions." "GSPMD uses maximum halo size across partitions, then uses DynamicSlice to remove
  excessive data in the halos."
- **Data structure:** a halo size for each device, and their maximum.
- **Algorithm:** GSPMD §4.3's halo exchange. Halos are exchanged by a collective permute at the largest halo over the
  devices, and each device slices off what exceeds its own.
- **Stage:** the collective permute's sends and receives and the slice are nodes `place` adds to the graph.

### 11. A tensor whose rows differ in length is a values array and an array of row boundaries

- **Source:**
  - TensorFlow, `tf.RaggedTensor`, https://www.tensorflow.org/api_docs/python/tf/RaggedTensor: "A `values` tensor,
    which concatenates the variable-length rows into a flattened list." "A `row_splits` vector, which indicates how
    those flattened values are divided into rows. In particular, the values for row `rt[i]` are stored in the slice
    `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`." "`RaggedTensor`s with multiple ragged dimensions can be
    defined by using a nested `RaggedTensor` for the `values` tensor."
  - Apache Arrow, Columnar Format, https://arrow.apache.org/docs/format/Columnar.html: "The List layout is defined by
    two buffers, a validity bitmap and an offsets buffer, and a child array." "Offsets must be monotonically
    increasing, that is `offsets[j+1] >= offsets[j]` for `0 <= j < length`, even for null slots." "The offsets
    buffer contains `length + 1` signed integers"
- **Data structure:** `tf.RaggedTensor`'s `values` and `row_splits`, which are Arrow's child array and offsets
  buffer. A tensor with rows of differing length is two tensors: the values of all rows concatenated, and nrows+1
  nondecreasing offsets, row i being `values[offsets[i]:offsets[i+1]]`. A second ragged dimension is a second array
  of offsets over the first. No row is padded.
- **Stage:** item 13 consumes the offsets. No quoted source states an algorithm or which stage produces the two
  tensors.

### 12. A collective sends a device's own shard out of its padded buffer

- **Applies to:** items 2 and 4 together.
- **Source:**
  - ScaLAPACK, `TOOLS/descinit.f` (version 1.7, May 1, 1997),
    https://www.netlib.org/scalapack/scalapack-2.1.0/TOOLS/descinit.f: `LLD_A`: "The leading dimension of the local
    array. LLD_A >= MAX(1,LOCr(M_A))." "Let K be the number of rows or columns of a distributed matrix," "LOCr( K )
    denotes the number of elements of K that a process would receive if K were distributed over the p processes of
    its process column." "LOCr( M ) = NUMROC( M, MB_A, MYROW, RSRC_A, NPROW )," "LOCr( M ) <= ceil(
    ceil(M/MB_A)/NPROW )*MB_A"
  - MPI 4.1, §6.1.3 "Subarray Datatype Constructor", https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node99.htm:
    "The subarray type constructor creates an MPI datatype describing an n-dimensional subarray of an n-dimensional
    array." "The subarray may be situated anywhere within the full array, and may be of any nonzero size up to the
    size of the larger array as long as it is confined within this array." `array_of_sizes`: "number of elements of
    type oldtype in each dimension of the full array". `array_of_subsizes`: "number of elements of type oldtype in
    each dimension of the subarray". `array_of_starts`: "starting coordinates of the subarray in each dimension".
  - HAP, §2.5.1 "Padded All-Gather and Grouped Broadcast": "To perform these operations for unevenly partitioned
    tensors, we can either pad the shards to the same size before communication, or broadcast each shard seperately
    using a NCCL group call." "When the tensors are nearly evenly partitioned, the required padding is minimal, and
    the padded All-Gather method outperforms other approaches owing to optimizations in NCCL. In contrast, when a
    tensor is sharded using heavily skewed ratios among devices, the grouped Broadcast approach yields better
    performance." "As performance of the two implementations depends on the sharding ratios, when we optimize the
    sharding ratios, the communication methods should be updated accordingly to achieve the optimal overall
    performance."
- **Data structure:** MPI's subarray, three arrays for each device: `array_of_sizes`, `array_of_subsizes` and
  `array_of_starts`. A device's buffer of item 4 is the full array and its shard of item 1 is a subarray of it: the
  buffer's size along the sharded dimension is at least the number of elements the device owns, as a local array's
  leading dimension is at least LOCr.
- **Algorithm:** a collective of item 2 is given, for each device, the sizes of the full array, the sizes of the
  subarray and where it starts, and sends the subarray: `counts[j]` is the subarray's number of elements. The padding
  is not sent. HAP §2.5.1 names a second way, sending the whole padded buffer through an equal-size collective. Each
  collective of item 2 uses the one of the two that item 6's solve finds cheaper at the solved sharding ratios, its
  comm(i)(B) being "determined based on the collective operation type, the sharding ratio B" and the profiled
  network; the choice is made again whenever the sharding ratios are solved again (HAP §2.5.1: "when we optimize the
  sharding ratios, the communication methods should be updated accordingly").
- **Stage:** item 6's solve, before `place`, writes which of the two each collective uses, with the tuple of item 1.
  `place` consumes it and adds the collective's `send` and `recv` nodes (item 2); for the subarray these carry the
  three arrays, each an array of integers with one entry per dimension of the buffer.

### 13. Which device holds which rows of a tensor whose rows differ in length

- **Applies to:** item 11 with item 1.
- **Source:**
  - PETSc, `MatCreateAIJ`, https://petsc.org/main/manualpages/Mat/MatCreateAIJ/: "The parallel matrix is partitioned
    across processors such that the first m0 rows belong to process 0, the next m1 rows belong to process 1, the
    next m2 rows belong to process 2, etc." `m`: "number of local rows (or PETSC_DECIDE to have calculated if M is
    given)". `d_nnz`: "array containing the number of nonzeros in the various rows of the DIAGONAL portion of the
    local submatrix".
  - PETSc, `MatGetOwnershipRanges`, https://petsc.org/release/manualpages/Mat/MatGetOwnershipRanges/: "returns the
    range of matrix rows owned by each process." `ranges`: "start of each process's portion plus one more than the
    total length at the end, of length `size` + 1". `MatGetOwnershipRange`,
    https://petsc.org/main/manualpages/Mat/MatGetOwnershipRange/: `m`: "the global index of the first local row";
    `n`: "one more than the global index of the last local row".
  - Pınar and Aykanat, "Fast optimal load balancing algorithms for 1D partitioning", Journal of Parallel and
    Distributed Computing, 2004, §1 (preprint, https://escholarship.org/uc/item/2bh2z29x): "The objective of the CCP
    problem is to find a sequence of K − 1 separator indices to divide a chain of N tasks with associated
    computational weights into K consecutive parts such that the bottleneck value—the load of the maximally loaded
    part—is minimized." §3: "by first performing a prefix-sum operation on the workload array W, with complexity
    O(N), and then making K − 1 binary searches in the prefix-summed W-array each with complexity O(lg N)." Abstract:
    "We conclude that exact algorithms with proposed efficient implementations can effectively replace heuristics."
  - Pınar, Tabak and Aykanat, "One-dimensional partitioning for heterogeneous systems: theory and practice", Journal
    of Parallel and Distributed Computing, 2008, abstract,
    https://repository.bilkent.edu.tr/items/30ad9f3b-967e-48e9-8ecb-150649bc31db/full: "We look at two cases:
    chain-on-chain partitioning, where the order of the processors is specified and chain partitioning, where
    processor permutation is allowed. We present polynomial time algorithms to solve the chain-on-chain partitioning
    problem optimally while we prove that the chain partitioning problem is NP-complete."
  - The same 2008 paper, §2: "a chain T = ⟨t1, t2, . . . , tN⟩ of N tasks with associated positive computational
    weights W = ⟨w1, w2, . . . , wN⟩ is to be mapped onto a processor chain P = ⟨P1, P2, . . . , PP⟩ of P processors
    with associated execution speeds E = ⟨e1, e2, . . . , eP⟩. The execution time of task ti on processor Pp is
    wi/ep." "a P-way partition of a task chain with N tasks onto a processor chain with P processors is described by
    a sequence Π = ⟨s0, s1, . . . , sP⟩ of P + 1 separator indices, where s0 = 0 ≤ s1 ≤ · · · ≤ sP = N." "The cost
    C(Π) of a partition Π is determined by the maximum processor load among all processors" "This C(Π) value of a
    partition is called its bottleneck value" "The CCP problem is to find a partition Πopt that minimizes the
    bottleneck value C(Πopt)."
  - §3.3 "Parametric search": "Parametric search algorithms rely on two components: a probing operation to determine
    if a solution exists whose bottleneck value is no greater than a specified value, and a method to search the
    space of candidate values. The probe algorithm can be computed in only O(P lg N) time by using binary search on
    the prefix-summed workload array." §3.3.1: "Nicol's algorithm efficiently searches for this subchain by
    considering each processor in order as a candidate bottleneck processor. For each processor Pp, the algorithm
    does a binary search for the smallest index that will make Pp the bottleneck processor." "If the probe function
    succeeds (fails) for some B, then probe function will succeed (fail) for any B′ ≥ (≤) B. Therefore by keeping the
    smallest B that succeeded and the largest B that failed, unnecessary probing is eliminated"
  - §4: "We extend the probing operation to the heterogeneous case, as shown in Fig. 3." "These algorithms not only
    decide whether a candidate value is a feasible bottleneck value, but they also set the separator index (sp)
    values for their greedy approach. In LR-PROBE, BINSEARCH(W, w) refers to a binary search algorithm that searches
    W for the largest index m, such that W1,m ≤ w."
  - §1: "The CP problem on the other hand, is NP-complete as we prove in this paper."
  - TensorFlow, `tf.RaggedTensor`, "Alternative Row-Partitioning Schemes": `row_lengths`: "a vector with shape
    `[nrows]`, which specifies the length of each row."
- **Data structure:** one array of P+1 separator indices for P devices, s_0 = 0 ≤ s_1 ≤ ... ≤ s_P = N, the same at
  every device (PETSc's ownership ranges; the partition Π of Pınar, Tabak and Aykanat). The rows are held in
  consecutive ranges, the first range by device 0, the next by device 1, and so on. A device holds the values of its
  rows, concatenated, and the length of each of its rows (`row_lengths`).
- **Algorithm:** chain-on-chain partitioning for processors of different speeds (Pınar, Tabak and Aykanat 2008, §2 to
  §4). A row's weight is its length; device p's load is the total weight of its rows divided by its speed; the
  separators minimise the largest load. They are found by a probe and a search: the probe, given a candidate largest
  load, gives each device in order the longest run of rows whose weight does not exceed the candidate times its
  speed, by binary search in the weights' prefix sums; the search is Nicol's, over the devices in order, keeping the
  smallest candidate that succeeded and the largest that failed. The devices' order is given; it is not searched.
  The weights' prefix sums are item 11's offsets.
- **Stage:** the partitioning takes item 11's offsets and the devices' speeds in their order and writes the
  separators. They are the sharding of the row dimension in the placement operand, which `place` consumes.

### 14. A device's sharding ratio does not exceed what the device can hold

- **Applies to:** item 6.
- **Source:**
  - Jia et al., "Whale: Efficient Giant Model Training over Heterogeneous GPUs", USENIX ATC 2022,
    https://www.usenix.org/conference/atc22/presentation/jia-xianyan, Algorithm 1 "Memory-Constraint Load Balancing"
    and its text: "The main idea of the algorithm is to shift the workload from the memory-overload device to a
    memory-free device with the lowest computation load." "oom_devices records out of memory devices whose value in
    mem_utils is greater than 1" "The algorithm then iteratively shifts the load from a memory-overload device to a
    memory-available device (line 11-18). It first finds a peak_device with maximum memory utilization from
    oom_devices, then it finds a valley_device with available memory space and the lowest" computation load.
  - HiGHS 1.15.1, `README.md` (the solver mesh builds against, `rdma/.build/highs/HiGHS-1.15.1/README.md`): "linear
    optimization problems of the form min 1/2 x^T Q x + c^T x s.t. L ≤ Ax ≤ U; l ≤ x ≤ u"
- **Data structure:** an upper bound u_j for each device j on its sharding ratio B_j: what the device can hold, as a
  fraction of the dimension.
- **Algorithm:** in item 6's linear program the bound is the variable's upper bound, B_j ≤ u_j. Where the tuple of
  item 1 is given and not solved for, Whale's Algorithm 1: a device whose memory use at its part of the dimension
  exceeds 1 gives up load to the device with free memory and the lowest computation load, repeatedly, until none
  exceeds 1.
- **Stage:** both run before `place` and write the tuple of item 1 in the placement operand, which `place` consumes.

### 15. With no tuple, a dimension is sharded as `torch.chunk` splits it

- **Applies to:** item 1.
- **Source:**
  - PyTorch 2.15, `torch.distributed.tensor`, https://docs.pytorch.org/docs/2.15/distributed.tensor.html: "The
    `Shard(dim)` placement follows the `torch.chunk(dim)` semantic, where the last few shards on the DeviceMesh
    dimension might be empty when the tensor dimension is not evenly divisible on the DeviceMesh dimension."
  - PyTorch 2.15, `torch.chunk`, https://docs.pytorch.org/docs/2.15/generated/torch.chunk.html: "If the tensor size
    along the given dimension `dim` is divisible by `chunks`, all returned chunks will be the same size. If the
    tensor size along the given dimension `dim` is not divisible by `chunks`, all returned chunks will be the same
    size, except the last one. If such division is not possible, this function may return fewer than the specified
    number of chunks."
- **Algorithm:** `torch.chunk`'s sizes. A sharded dimension with no tuple of item 1 has them: every device the same
  size except the last, and trailing devices empty where the extent does not allow more. Item 1's rounding applies
  only where a tuple is given.
- **Stage:** `place` writes these sizes where the placement operand carries no tuple for a sharded dimension; they
  are the `counts` of items 2 and 3.

## Dropped

Dropped as not needed to implement the requirements above (the operator: "if any of them are optional, drop them"):
`design/heterogeneity.md` R11, changing the tuples while a program runs; a program's shapes are fixed when it is
compiled (`aot-compilation-change-request.md` A1, the operator: "the use of cudagraphs style 'shapes are compiled but
indexes inside of shapes might not need to be'"), and a different tuple is a different compilation.
