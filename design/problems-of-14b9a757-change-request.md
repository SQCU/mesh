# Change request: the problems the revisions of 14b9a757 created

Code is cited at mesh 576a3159 and metal-microbench 9823303f. The problems are those that implementation created. Each
gives:
- **Code**: the code as it is;
- **Source**: the published text, library or algorithm, quoted verbatim with where it was read;
- **Change**: what the code must be, which is the source applied and nothing more.

## The turns

> stop describing this in idiomatic terms. why does this happen in algorithmic terms. in data structural terms. in
> terms of reference literature. once this is said, what we *should* be doing will be a lot more clear, esp with al lof
> theo ther 'partial fixes', which all sound like invented problems and idiomatic sourceless code...

> none of those words are in an algorithms textbook.
> 1: nothing you wrote to yourself in the last turn is referencing an algorithm, data structure, result, paper, or text
> from the literature
> 2: if you need to 'interpret and review' something with 'multiple reviewers' you are not using strict citations to
> describe requirements in non-claude-idiomatic language or requirements
> 3: you should not be explaining or documenting code at this point in development. code is either a strict
> implementation of a reference or it isn't. inventing preposterous fictional reasons why extra uncited mechanisms,
> control flow, syncs, allocations, copies, etc. are 'explained' by subagent behavior so far is idiomatic and not
> grounded to a citation. doing this again at greater regresses is also not grounded and is also still idiomatic.
> 4: if you're saying '{letter}{number} and a user's rule say...' you are not citing published literature and
> definitely aren't citing any 'rules' the user proscribed.

> i want all of the problems created instead of solved from the previous round of implementation handled, and handled
> by the boring change request documentary format using strict citations to libraries, literature, data structures,
> and algorithms from outside of whatever claude tries to ad lib. this is not a place to express creativity and i don't
> want to hear fanciful stories about how ther ewas totally some really interesting explanation for why a bunch of
> runtime and kernel corrupting code was really satisfying to write (even if it wasn't what anyone was *told* to write).

## How it is implemented

- **Each change is the quoted source's mechanism, as the source states it.** A statement, branch, field, wait, copy or
  allocation that the quoted text does not contain is not written.
- **A commit names the source it implements.** Nothing else explains it: no decision records, no notes of reasons.
- **A source missing here is a stop.** Where this document quotes no source for something the code needs, the
  implementer stops and names the gap. It invents nothing to fill it.

## 1. An AllToAll is refused

- **Code:**
  - rdma/mesh_compile.py:1480–1484 (`_inside`) raises NotImplementedError for every `all-to-all`: "a member's own piece
    lies in its result only by a copy or a share".
- **Source:**
  - MPI 4.1 §7.8 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node128.htm): "The j-th block sent from MPI process
    i is received by MPI process j and is placed in the i-th block of recvbuf."
  - The same section, in place: "The data to be sent is taken from the recvbuf and replaced by the received data. Data
    sent and received must have the same type map as specified by recvcount and recvtype."
  - MPICH (https://raw.githubusercontent.com/pmodels/mpich/main/src/mpi/coll/alltoall/alltoall_intra_pairwise.c):

    ```
    /* Make local copy first */
    mpi_errno = MPIR_Localcopy(((char *) sendbuf + rank * sendcount * sendtype_extent), sendcount, sendtype,
                               ((char *) recvbuf + rank * recvcount * recvtype_extent), recvcount, recvtype);
    ```
- **Change:**
  - Block j of member i (j ≠ i) is a transfer from the view of block j in i's operand to the view of block i in j's
    result.
  - Block i is one local copy from the operand's view to the result's view, as MPIR_Localcopy does it. It is torch's
    `copy_` ("Copies the elements from src into self tensor and returns self", torch 2.14.0 _tensor_docs.py), run in the
    instantiated steps, with no synchronize and no clone.
  - The local copy is absent only where the operand is the result with the same type map (MPI's in-place condition).
  - The refusal is deleted.

## 2. An AllGather's own part is refused unless its producer writes it in place

- **Code:**
  - rdma/mesh_compile.py:1497–1504 (`_direct`) raises where the own part's producer is not an out variant, a
    multi-output result or a ReduceScatter.
  - rdma/mesh_compile.py:1430–1432 (`_strided`) raises for "a transfer's end in no placed value".
- **Source:**
  - MPI 4.1 §7.7 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node126.htm), in place: "the input data of each
    MPI process is assumed to be in the area where that MPI process would receive its own contribution to the receive
    buffer."
  - NCCL User Guide, In-place Operations (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html):
    "For ncclReduceScatter and ncclAllGather, in place operations are done when the per-rank pointer is located at the
    rank offset of the global buffer", with the example `ncclAllGather(data+rank*sendcount, data, sendcount, datatype,
    op, comm, stream);`.
  - MPICH ring allgather
    (https://raw.githubusercontent.com/pmodels/mpich/main/src/mpi/coll/allgather/allgather_intra_ring.c):

    ```
    /* First, load the "local" version in the recvbuf. */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf + rank * recvcount * recvtype_extent), recvcount, recvtype);
    ```
- **Change:**
  - Where the producer writes through an `out` tensor, its `out` is the result's own-part view. That is NCCL's in-place
    condition.
  - Otherwise the own part is one local copy into that view, as MPICH's `sendbuf != MPI_IN_PLACE` branch does, with the
    same form as item 1's.
  - Both refusals are deleted.

## 3. A view an operator makes has no place

- **Code:** rdma/mesh_compile.py:1430–1432 names "a view an operator makes" among the ends "in no placed value".
- **Source:** torch 2.14.0 torch/_tensor_docs.py, `view`: "Returns a new tensor with the same data as the self tensor but
  of a different shape. The returned tensor shares the same data and must have the same number of elements".
- **Change:**
  - A view node's place is its base's place, with the view's offset, sizes and strides as torch computes them.
  - Only a result torch allocates outside the placed pages has no place. That case is item 2's local copy.

## 4. A window's middle is refused; a window over a replicated operand exchanges halos

- **Code:**
  - rdma/mesh_compile.py:1490–1493 and 1497–1504 raise for a window middle not written in place. This includes a
    convolution whose operand is whole at every node, where the halos land on bytes the node already holds.
  - Two partitioned spatial dimensions stop at the same place.
- **Source:** GSPMD (Xu et al., arXiv 2105.04663, https://arxiv.org/html/2105.04663).
  - §4.5: "It uses AllGather to replicate data along sharded dimensions, AllToAll to switch sharded dimensions,
    CollectivePermute to change device order, and DynamicSlice to shard replicated dimensions."
  - §4.3: "We use the CollectivePermute operator to exchange halo data between partitions."
  - Appendix A.2:
    - "Each spatial dimension in the convolution has the following set of configurations."
    - "First, we calculate the maximum sizes of left and right halos across partitions and perform the halo exchange of
      the maximum size (Steps 1 and 2). Since some partitions may have excessive halos than needed, we use DynamicSlice
      (based on the partition ID) to get the valid region for the current partition (Step 3). Finally, some partitions
      may include garbage values (e.g., halos from out-of-range input data), so we apply masking as described in
      Section 4.1."
- **Change:**
  - **A replicated operand's window** is a DynamicSlice of the local operand: a view of its pages, with no transfer.
  - **A sharded operand's window** is A.2's sequence for each spatial dimension: the halo exchange of the maximum sizes
    by CollectivePermute, then DynamicSlice, then masking.
  - **The middle** is in place where its producer writes through the window's view (item 2's condition), and one local
    copy otherwise.
  - The refusals are deleted.
  - **Gap:** the order in which several partitioned spatial dimensions compose, so that corner halos arrive, is not in
    the quoted text. It is a stop until a source is quoted.

## 5. A CollectivePermute pair (p, p) is refused

- **Code:** rdma/mesh_compile.py:1788–1790 raises for any pair whose source is its target.
- **Source:** XLA operation semantics, CollectivePermute (https://openxla.org/xla/operation_semantics):
  - "A list of (source_replica_id, target_replica_id) pairs. For each pair, the operand is sent from source replica to
    target replica."
  - "Any two pairs should not have the same target replica id, and they should not have the same source replica id."
  - "If a replica id is not a target in any pair, then the output on that replica is a tensor consisting of 0(s) with
    the same shape as the input."
- **Change:**
  - A pair (p, p) is item 1's local copy at p.
  - A replica that is no target outputs zeros.
  - The refusal is deleted. The test of XLA's two restrictions (1785–1787) stays.

## 6. A consumer that views an in-place part fails in torch

- **Code:**
  - An own part written in place along a dimension other than the outermost is a non-contiguous view of the result.
  - rdma/mesh_compile.py inserts nothing where a consumer's `view` of it fails torch's condition, so torch raises at
    make().
- **Source:** torch 2.14.0 torch/_tensor_docs.py.
  - `view`: "each new view dimension must either be a subspace of an original dimension, or only span across original
    dimensions d, d+1, …, d+k that satisfy the following contiguity-like condition that ∀ i = d, …, d+k−1,
    stride[i] = stride[i+1] × size[i+1]. Otherwise, it will not be possible to view self tensor as shape without
    copying it (e.g., via contiguous()). When it is unclear whether a view can be performed, it is advisable to use
    reshape, which returns a view if the shapes are compatible, and copies (equivalent to calling contiguous)
    otherwise."
  - `contiguous`: "If self tensor is already in the specified memory format, this function returns the self tensor."
- **Change:**
  - Each view node is lowered as torch's `reshape` is: a view where the condition holds; otherwise `contiguous` into a
    placed value, run as an out-variant copy.
  - Nothing else is inserted.

## 7. A block holding bytes no transfer covers is unordered with every item

- **Code:** rdma/mesh-collective.c:820–826, 885 and 957–958 (PLAN_ALONE).
  - A named value's blocks that no range covers are an item ordered with no other item.
  - That item therefore keeps its pages for the program's whole life.
- **Source:**
  - ExecuTorch documentation, "Concepts", sections Out variant and Memory planning:
    - "Instead of allocating returned tensors in kernel implementations, an operator's out variant will take in a
      pre-allocated tensor to its out kwarg, and store the result there. This makes it easier for memory planners to
      perform tensor lifetime analysis."
    - "a memory planning pass ... assigns a memory ID to each tensor and an offset in the buffer".
  - The offline problem the planner solves is Garey and Johnson's DYNAMIC STORAGE ALLOCATION (Computers and
    Intractability, 1979, problem SR2; cited from memory). Each item has an arrival and a departure, and two items may
    share storage only where one departs before the other arrives.
- **Change:**
  - Every block's item arrives at the first write and departs at the last read of the values whose bytes it holds, in
    the program's order.
  - The planner reads those times from the program's graph, which it is given whole.
  - Only the program's invariants live from its start to its end.

## 8. An invariant's transfer is a per-execution publication with execution UINT64_MAX − 1

- **Code:**
  - rdma/mesh-collective.h:678 releases a range sent from a MESH_INVARIANT value once, "with `execution` UINT64_MAX −
    1".
  - rdma/mesh_compile.py:2130 does the same (LOADED).
  - The value is a sentinel in the per-execution count word.
- **Source:** loop-invariant code motion moves a computation whose operands do not change in a loop out of that loop, to
  run once before it (Aho, Lam, Sethi and Ullman, Compilers: Principles, Techniques, and Tools, 2nd ed., §9.5, "partial
  redundancy elimination" and loop-invariant computations; cited from memory).
- **Change:**
  - An invariant's transfers are moved out of the executions. They run once at make(), before execution 0, and are
    awaited once.
  - They have no per-execution count word and no sentinel value.

## 9. Two destination instructions may write the same bytes

- **Code:**
  - rdma/mesh-collective.c refuses no pair of destination instructions whose byte intervals at one node intersect within
    one execution.
  - Ranges may now share a block, so such a pair is two writers of one byte.
- **Source:** MPI 4.1 §6.1.11 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node107.htm): "A datatype may
  specify overlapping entries. The use of such a datatype in any communication in association with a buffer updated by
  the operation is erroneous."
- **Change:**
  - At the plan, the destination intervals of each node and execution are sorted by start.
  - Any interval that begins before its predecessor ends makes the plan EINVAL.
  - That is the sweep test for interval intersection, and the plan is made at compile time.

## 10. A view has at most eight dimensions

- **Code:** rdma/mesh-collective.h:430–432 (`MESH_STRIDED_DIMS 8u`, `size[8]`, `stride[8]`); rdma/mesh_compile.py:1441–1443
  refuses more.
- **Source:** MPI 4.1 §6.1.3 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node99.htm):
  "MPI_TYPE_CREATE_SUBARRAY(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype)", where
  "IN ndims: number of array dimensions (positive integer)" and "The ndims parameter specifies the number of dimensions
  in the full data array and gives the number of elements in array_of_sizes, array_of_subsizes, and array_of_starts."
- **Change:**
  - A view is `dims` followed by `dims` (size, stride) pairs, in the plan's operand, with no fixed bound.
  - The refusal is deleted.

## 11. A port child spawned after a link loss runs the view's plans from execution 0

- **Code:**
  - rdma/mesh-flow.c:1919–1926: the load's one test is the operand's `serial` against hdr `view`.
  - A child spawned for a port after a loss, before the next view is installed, passes it and loads the plans with
    their cursors at execution 0. It then sends and receives executions that have already landed.
- **Source:**
  - E. A. Lee and S. Ha's self-timed strategy (Globecom 1989), as S. S. Bhattacharyya, S. Sriram and E. A. Lee state it
    in "Optimizing synchronization in multiprocessor DSP systems", IEEE Trans. Signal Processing 45(6), 1997, §2: "retain
    the processor assignment and actor ordering specified by the fully static schedule ... Each processor executes the
    actors assigned to it in the order specified at compile time."
  - The plan's processors are its cables, so a lost cable is a lost processor of the schedule.
- **Change:**
  - The loss of a cable that a plan's lists name cancels that plan by plan_cancel, as a view change does.
  - The cancellation is done before any child is spawned for the port.
  - The plan's replacement is compiled on the next view. No plan is loaded from the middle of its executions.

## 12. A view's identifier repeats after every bridge restarts

- **Code:**
  - rdma/mesh-records.c:741 makes a view `.view=greatest+1`, which is 1 again after every bridge restarts.
  - rdma/mesh-flow.c:1922, 2089 and 2523 compare a plan's `serial` with the counter alone.
  - So a plan of an earlier view whose counter matches stands again. The view record already carries the sequencer and
    its boot (rdma/mesh-records.c:741, 1414).
- **Source:**
  - JGroups (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/ViewId.java):
    - "ViewIds are used for ordering views (each view has a ViewId and a list of members). Ordering between views is
      important for example in a virtual synchrony protocol where all views seen by a member have to be ordered."
    - Its fields are "Address of the creator of this view" and "Lamport time of the view".
    - Its order: "The comparison is done on the IDs, if they are equal, we use the creator."
  - Its address (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/util/UUID.java): "Logical
    address which is unique over space and time."
- **Change:**
  - A view's identifier is (Lamport time, creator), where the creator is (sequencer node, boot).
  - A plan's `serial` is that pair, and every comparison compares the pair.

## Sources for the stops (2026-11-06)

The implementation of the items above (mesh 496ad4ef to 6d8d18ce, metal-microbench 73fe1517) stopped at four places for
lack of a quoted source. Each source is given here verbatim, with the change it requires.

### 4. Several partitioned spatial dimensions

- **Stop:** no quoted source gave the order in which halo exchanges over several dimensions compose so that corner halos
  arrive.
- **Source:** XLA's SPMD partitioner, `PartitionedHlo::ReshardAsWindowedInput`
  (https://raw.githubusercontent.com/openxla/xla/main/xla/service/spmd/spmd_partitioner.cc, read 2026-11-06):

  ```
  // Halo exchange.
  HloInstruction* visiting_hlo = hlo_;
  ...
  // TODO(yuanzx): We are concatenating on each sharded dimension one at time,
  // and in the second dimension (and beyond) we create halos by slicing the
  // concat in the previous dimension, which is not optimal. ...
  for (int dim = 0; dim < base_shape_.dimensions().size(); ++dim) {
    int64_t shard_count = halo_exchange_target->dimension(dim);
    if (shard_count == 1 || can_leave_dimension_partitioned[dim]) {
      continue;
    }
    ...
    auto resharded = ExchangeHaloAndGetValidData(
        visiting_hlo, halo_exchange_base_shape, left_halo_size_functions,
        right_halo_size_functions, ...);
    ...
    visiting_hlo = *resharded;
  }
  ```
- **Change:**
  - The exchanges run one sharded dimension at a time, in increasing dimension order.
  - Each exchange's operand is the previous exchange's result, so the second exchange's halos carry the first's, and the
    corners arrive with them.
  - Each exchange is item 4's A.2 sequence.
  - A dimension with one shard is skipped.
  - The refusal for more than one dimension is deleted.

### 7. Blocks holding bytes no transfer covers

- **Stop:** no quoted source related a position in the program's order to the transfers. The IR has no event for the
  program's own reads and writes. Copies kept across executions had no rule.
- **Source:**
  - ExecuTorch, exir/memory_planning.py (https://raw.githubusercontent.com/pytorch/executorch/main/exir/memory_planning.py,
    read 2026-11-06), `update_tensor_lifetime`:
    - "A tensor's lifetime are represented by the index of the first and last node referring that tensor in its
      inputs/outputs."
    - "Mutable buffers and device resident constants keep their data across calls, so they are never freed."
  - The same file's test for two tensors sharing storage: `spec.lifetime[1] < alloc.spec.lifetime[0] or
    spec.lifetime[0] > alloc.spec.lifetime[1]`.
  - MPI 4.1 §3.7 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node71.htm):
    - "The sender should not modify any part of the send buffer after a nonblocking send operation is called, until the
      send completes."
    - "The receiver should not access any part of the receive buffer after a nonblocking receive operation is called,
      until the receive completes."
- **Change:**
  - **A value's lifetime** is [the index of the first node referring to it, the index of the last], over the exported
    graph's nodes in order, as `update_tensor_lifetime` computes it. The compile passes each value's pair to the plan.
  - **A transfer's ends** each occupy their value from the node that starts the transfer to the node that waits for its
    completion (MPI §3.7). A send's source is held until its completion is awaited, and a landing's destination from the
    receive's start.
  - **A block's lifetime** spans the lifetimes of the values whose bytes it holds.
  - **Sharing:** two blocks may share pages exactly where ExecuTorch's test finds their lifetimes disjoint.
  - **Never freed:** a value whose data is kept across executions (a copy of lag k ≥ 1, and an invariant) is never
    freed, as ExecuTorch's mutable buffers are not. Only these remain unordered with every item.

### 8. An invariant's transfer outside the executions

- **Stop:** no quoted source said how a computation moved out of the executions is planned and run.
- **Source:** torch 2.14.0 torch/nn/parallel/distributed.py, DistributedDataParallel's arguments: "init_sync (bool):
  Whether to sync during initialization to verify param shapes and broadcast parameters and buffers."
  - The broadcast of the invariant parameters is a collective issued once at construction, before any iteration, and not
    part of the iterations.
- **Change:**
  - An invariant's transfers are their own function, compiled and planned as any function is, called once in make()
    before execution 0, awaited, then retired.
  - The per-execution functions name the invariant's pages as a value that is never freed (item 7).
  - LOADED and the execution UINT64_MAX − 1 are deleted.

### 11. A plan loaded after its cancellation

- **Stop:** a child spawned after a loss, before the next view, still loads the cancelled plan. Nothing the load reads
  records the cancellation.
- **Source:** the Linux kernel's RCU documentation, "What is RCU?"
  (https://raw.githubusercontent.com/torvalds/linux/master/Documentation/RCU/whatisRCU.rst, read 2026-11-06):
  - "The removal phase removes references to data items within a data structure (possibly by replacing them with
    references to new versions of these data items), and can run concurrently with readers."
  - "any reader starting after the removal phase will be unable to gain a reference to the removed data items".
- **Change:**
  - **Removal:** the cancellation at a loss removes the plan from the structure the load reads, stored before any child
    is spawned for the port.
  - **One writer:** that store is the bridge parent's alone.
  - **The load:** it reads the removal after its acquire and gains no reference to the plan.
  - **The program:** it learns of the cancellation through the raised counts, as plan_cancel already gives it.

## Sources for the second stops (2026-11-06)

The implementation of the sources above (mesh 0f48775f, c840ff2f, bce1ed90, 028b27d6) stopped at three places.

### 7. The end of a send's or forward's hold

- **Stop:** no node waits for a send's or a forward's completion, so MPI §3.7's hold has no end in the program's order.
- **Source:**
  - torch 2.14.0 torch/distributed/_functional_collectives.py:
    - `wait_tensor`: "Wait on a tensor returned by the collectives ops."
    - `AsyncCollectiveTensor`: "A Tensor wrapper subclass that is used to trigger a call to wait prior to first use of
      the underlying tensor."
  - MPI 4.1 §7.1 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node113.htm):
    - "Collective operations can (but are not required to) complete as soon as the caller's participation in the
      collective communication is finished."
    - "The completion of a collective operation indicates that the caller is free to modify locations in the
      communication buffer."
- **Change:**
  - Each collective has one completion wait at each node: torch's `wait_tensor`, placed at the node before the first
    use of the collective's result there.
  - That wait completes the node's whole participation: its sends, its forwards and its landings.
  - A send's source and a forwarded block are held until that node, which ends the MPI §3.7 hold. Their lifetimes are
    then item 7's, and the stop's "unordered with every item" goes.
  - A collective whose result has no use at a node waits at the program's last node.

### 7. The u copies against one-copy items

- **Stop:** nothing orders the plan's u copies of a landing, which live across executions, against items that live
  within one execution.
- **Source:** M. Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", PLDI 1988, §2.3
  (https://fg-kastens.cs.uni-paderborn.de/lehre/pgrekonf/seminar/Lam1988.pdf, read 2026-11-06; the scan's errors are
  corrected in brackets):
  - "The lifetime of a register variable is defi[n]ed as the duration between the f[i]rst assignment into the variable
    and it[s] last use."
  - "the minimum degree of unrolling, u, to implement the same schedule is simply max_i q_i."
  - The source gives each variable locations of its own. It gives no rule for two variables sharing a location across
    the unrolled steady state.
- **Change:** none. The two kinds of item stay unordered with each other.
  - This costs pages only. It writes no byte twice.
  - It stands until a source for sharing across the steady state is quoted. Hendren, Gao, Altman and Mukerji's cyclic
    interval graphs (1992) is named as that source, not yet read.

### 8. An invariant's transfer

- **Stop:** no quoted source gave the form of a function called once outside the block. Nor did any say how the
  per-execution functions name another plan's pages.
- **Source:**
  - torch 2.14.0 torch/export/exported_program.py, `ExportedProgram`: "It contains an torch.fx.Graph that represents
    Tensor computation, a state_dict containing tensor values of all lifted parameters and buffers, and various
    metadata."
  - Every rank runs make() on the same exported program, so every rank holds every parameter and buffer.
  - GSPMD §4.5 (quoted at item 4): "DynamicSlice to shard replicated dimensions".
- **Change:**
  - No transfer has an invariant as its source.
  - Each node loads, once in make() before execution 0, the DynamicSlice of its own copy that its placement needs, into
    its own never-freed pages (item 7).
  - The plan refuses a transfer whose source is MESH_INVARIANT (EINVAL at the plan).
  - LOADED and the execution UINT64_MAX − 1 are deleted. The function called once is not needed.

## Sources for the third stops (2026-11-06)

The implementation of the second sources (mesh 2155c2b8) stopped at three places.

### 7. No word tells a node that its sends and forwards are complete

- **Stop:** send completions, of sends and forwards alike, are counted only in the port child's private `struct flow`,
  and only `plan_end` reads them. `wait_tensor` could therefore complete only the landings.
- **Source:**
  - rdma-core, libibverbs/man/ibv_post_send.3
    (https://raw.githubusercontent.com/linux-rdma/rdma-core/master/libibverbs/man/ibv_post_send.3, NOTES): "The buffers
    used by a WR can only be safely reused after WR the request is fully executed and a work completion has been
    retrieved from the corresponding completion queue (CQ)."
  - libfabric, man/fi_cntr.3.md (https://raw.githubusercontent.com/ofiwg/libfabric/main/man/fi_cntr.3.md, DESCRIPTION):
    "Counters record the number of requested operations that have completed. Counters can provide a light-weight
    completion mechanism by allowing the suppression of CQ completion entries. They are useful for applications that
    only need to know the number of requests that have completed, and not details about each request."
- **Change:**
  - Each declared transfer has a send-completion counter at its source node, and each forward at its relay, in the
    plan's state section beside the landing counts.
  - The port child raises the counter by each work completion it retrieves for that transfer's sends or forwards, as
    it raises a landing's count. This is in rdma-core's sense: the buffer may be reused only once the completion is
    retrieved.
  - The collective's `wait_tensor` at a node waits for its landing counts and its send-completion counters to reach
    the execution's totals.
  - The failport file mesh-lanes.c, which retrieves the completions, is unchanged. The raise is where `struct flow`
    already counts them, in mesh-flow.c.
  - PLAN_SENT goes once the hold has this end.

### 8. A collective over a view of an invariant

- **Stop:** where an aliasing slice, select or unsqueeze of a module attribute is an AllToAll's operand, the transfer
  sends from the invariant's pages. The plan now refuses that program. The partitioner never makes that node's whole
  value, which a DynamicSlice would need.
- **Source:** GSPMD §4.5 (quoted at item 4): "DynamicSlice to shard replicated dimensions"; torch's `ExportedProgram`
  (quoted at the second stops' item 8): every rank holds the state_dict.
- **Change:**
  - An invariant's sharding at the partitioner is replicated: every node holds it whole.
  - A value made from invariants by views alone is replicated too, since a view of a replicated value is replicated at
    every node.
  - A collective whose operand is such a value is a resharding from replicated, so it is the DynamicSlice of its
    result's sharding, at each node, with no transfer.

### 8. Loading only the slices a node reads

- **Stop:** make() loads each attribute whole, and its DynamicSlices are views of it. No source said how to load only
  the slices where one node's placement needs several of one attribute.
- **Source:** GSPMD §4.5, as above. Each resharding is its own DynamicSlice instruction, and so its own value.
- **Change:**
  - Each distinct DynamicSlice a node's program reads of an attribute is its own never-freed value at that node.
  - Each is loaded once in make(), from the host's state_dict tensor, by torch's `copy_` of that slice into the value's
    pages.
  - The whole attribute is loaded only where the node reads it whole.

## Sources for the fourth stops (2026-11-06)

The implementation of the third sources (mesh 6a8a8524, 60f43ddb) raised two points about item 7 and one about item 8.

### 7. A counter per range

- **Point:** one send serves every destination of its range, so the counter is one per range and copy, which each
  declared transfer's entry names. The text above said "each declared transfer".
- **Source:** libfabric fi_cntr(3), as quoted above: "Counters record the number of requested operations that have
  completed". The operation is the send, which is one per range.
- **Change:** the text is corrected to one counter per range and copy at its source, and one per forward at its relay.
  The code at 6a8a8524 stands.

### 7. A cancelled plan's sends

- **Stop:** nothing raises a send counter when its plan is cancelled. rdma-core forbids reusing a buffer before its
  completion is retrieved. A node that only sends therefore ends with ETIMEDOUT, not ECANCELED.
- **Source:** libfabric:
  - man/fi_endpoint.3.md (https://raw.githubusercontent.com/ofiwg/libfabric/main/man/fi_endpoint.3.md), fi_cancel:
    "Canceling an operation causes the fabric provider to search for the operation and, if it is still pending, complete
    it as having been canceled. An error queue entry will be available in the associated error queue with error code
    FI_ECANCELED. On the other hand, if the operation completed before the call to fi_cancel, then the completion status
    of that operation will be available in the associated completion queue."
  - man/fi_cntr.3.md: "fi_cntr_readerr: Reads the number of operations which have completed in error."
- **Change:**
  - Each counter has an error count beside its count.
  - When the port child finds a plan cancelled, it first retrieves the completions of the sends it has posted. Each
    raises the count, or the error count where it completed in error, as fi_cancel has a completed operation report its
    own status.
  - Every send of that plan the child has not posted is then completed as cancelled: it raises the error count.
  - The program's wait returns ECANCELED once the error count is nonzero, after the count and the error count together
    reach the execution's total.
  - No buffer is reused before the completion of its posted send is retrieved.

### 8. A slice of a view of an attribute

- **Point:** a slice of a view of an attribute (`W.t()` sliced) still loads `W` whole, since the text named slices "of an
  attribute" only.
- **Source:** GSPMD §4.5, as above. A value made from a replicated value by views alone is replicated, and its
  resharding is a DynamicSlice.
- **Change:**
  - Each distinct DynamicSlice a node reads of a value made from an attribute by views alone is its own never-freed
    value.
  - It is loaded once in make() by torch's `copy_` from that view of the state_dict tensor.

## Stops that remain (2026-11-06, at mesh de70486f)

Replaced by the appendix's items 13 and 14.

- **Sends pending when a link ends.** A send that is posted but still pending when its plan is cancelled by a link loss
  is never counted.
  - Its completion is flushed by rdma/mesh-child.c `lanes_flush`. That is a failport lifecycle file, reused verbatim,
    and it raises no count.
  - So that execution's wait ends with ETIMEDOUT after MESH_PLATFORM_NS without progress, not with ECANCELED. The wait
    is bounded.
  - **Missing source:** how those completions reach the counter without changing mesh-child.c.
- **The u copies against one-copy items** (the second stops' item 7). They stay unordered with each other, costing pages
  only.
  - **Missing source:** Hendren, Gao, Altman and Mukerji's cyclic interval graphs (1992), named and not yet read.

## Appendix: the stops of 36bd7cbd (2026-11-06)

Code is cited at mesh 36bd7cbd. The two stops above are replaced by items 13 and 14.

### The turns

> "which is a failport file I'm not allowed to change, and it doesn't raise the counter." what is the citation to a
> specific algorithm in the collective communications and async concurrent programming literature (no, this repo is
> not an example) which says 'there's something called a failrpot' and 'once the failport exists nobody is allowed to
> change it but you are allowed to use it but not to change it'?
> "The source that would allow the sharing is Hendren et al.'s cyclic interval graphs (1992), which I've named but not
> yet read."
> add this citation and the corresponding implementation goal to a change request document or appendix to an existing
> one. no further elaborationo r idiomatic computer science will be needed for a subagent to ahndle the assignment of
> strictly implementing a method on a data type...?

> i mean you can grab any ocr tool in the universe of any size as long as its under 96gb allocation plus parameters (or
> 384gb allocation plus parameters) and literally use the 4 mac studios to literally OCR-parse the thesis you want?

> but this feels like a subagent thing and i'm not surte that this exact single paper is the only source htat is needed
> to get canonical reference examples of this solverkind; other implemetnations surely exist in the apache and mit
> literature-of-programs

### 13. A send pending at a link's end is never counted

- **Code:**
  - rdma/mesh-child.c:44–75 (`lanes_flush`) moves each lane with work outstanding to the error state and takes the
    completions. Line 68 only lowers `left` for each one it takes: no counter is raised.
  - rdma/mesh-flow.c:1577–1601 (`sends_complete`): on the data path, a SEND completion with a nonzero status raises
    its record's error count (`record->count+1`, the record at `wr_id`).
- **The stop's ground:** the stop of 36bd7cbd rested on "a failport lifecycle file, reused verbatim". No published
  source says this. The operator's words it rested on (design/page-queue-change-request.md, 2026-11-03) are:
  "then you have to verbatim reuse that process lifecycle code since you have no alternative to present or cite which
  sin't odious and a waste of time to test or run." The stop is withdrawn.
- **Source:**
  - Linux v6.12, drivers/infiniband/core/verbs.c
    (https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/infiniband/core/verbs.c):
    - lines 2966–2967: "ib_drain_sq() - Block until all SQ CQEs have been consumed by the application."
    - `__ib_drain_sq`, line 2829 onward:
      ```
      struct ib_qp_attr attr = { .qp_state = IB_QPS_ERR };
      ...
      ret = ib_modify_qp(qp, &attr, IB_QP_STATE);
      ```
  - rdma-core libibverbs/enum_strs.c:110, at a70151dfec390eb5f7fc5b266f51d601da2a7778
    (https://raw.githubusercontent.com/linux-rdma/rdma-core/a70151dfec390eb5f7fc5b266f51d601da2a7778/libibverbs/enum_strs.c):
    `[IBV_WC_WR_FLUSH_ERR]		= "Work Request Flushed Error",`
  - libfabric man/fi_cntr.3.md (https://raw.githubusercontent.com/ofiwg/libfabric/main/man/fi_cntr.3.md):
    - "fi_cntr_readerr: Reads the number of operations which have completed in error."
    - "The read error call returns the number of operations that completed in error and were unable to update the
      counter."
- **Change:**
  - Each SEND completion that `lanes_flush` takes raises its record's error count, as `sends_complete` does for a SEND
    completion in error.
  - Nothing else in `lanes_flush` or `child_leave` changes.

### 14. The u copies against one-copy items

- **Code:**
  - rdma/mesh-collective.c:918 (`plan_before`) returns 0 where `x->kind != y->kind`. A PLAN_LOCAL item (a value of lag
    0, one copy, positions `begin` to `end` in each execution) and a PLAN_ROUTED item (copy c of a set, one of u) are
    unordered.
  - So `mesh_graph_chains` (mesh-collective.c:1037; rdma/mesh-graph.h:198) never puts them on one block.
  - rdma/mesh_compile.py:2036–2041: every value's positions are its lifetime. A landing's run from the execution's
    receives' start to the item that waits for it.
- **Source:**
  - C. Mukerji, "Register allocation using cyclic interval graphs", M.Sc. thesis, McGill University, July 1994, Gao,
    Guang R. (Supervisor) (https://mcgill.scholaris.ca/items/4e5e10ea-b427-4537-8a24-9a6efa7da1b6; PDF
    https://mcgill.scholaris.ca/server/api/core/bitstreams/5031fdf9-9e78-49b5-97e9-f74540dde00d/content, read from
    the page images 2026-11-06).
    - Its pages 23 and 30 carry the footnote "Parts of this section has been excerpted from [HGAM92]": L. J. Hendren,
      G. R. Gao, E. R. Altman and C. Mukerji, "A register allocation framework based on hierarchical cyclic interval
      graphs", CC 1992, LNCS 641, pp. 176–191 (https://doi.org/10.1007/3-540-55984-1_17). The paper itself was not
      read; Springer's copy is paywalled.
  - §2.1, p. 21: "Let t0, t1, ... be the starting time points of a sequence of machine operations. Without loss of
    generality, we use non-negative integers for the time points. We use [t : t'] to denote the interval between t and
    t' including both end points. The notation [t, t') denotes the same interval but with the end point t' left out."
  - §2.1, pp. 21–22: "we say that variable x is defined at time point t. The live range of x will continue to the time
    point t', (t' > t), where it is last used in a statement, e.g. u = x + v. After time t', the value in x is no longer
    live. In this paper, we define the lifetime interval of x to be [t, t')."
  - Definition 2.1.1, p. 22: "A time t is covered by an interval I1 : [t1, t1'), if (t1 ≤ t < t1'), or by an interval
    I1' : [t1, t1'] if (t1 ≤ t ≤ t1'), or by a cyclic interval I2 : ([t1, t1'), [t2, t2']), if t is covered by either
    ([t1, t1') or [t2, t2'])."
  - Definition 2.1.2, p. 22: "Two intervals I1, I2 overlap if there exists a time t that is covered by both I1 and I2."
  - §3.1, p. 29: "Cyclic intervals wrap around from one iteration to the next, it is one long interval. However, in the
    interval graphs, we choose to depict it as two intervals instead."
  - §2.2, pp. 22–23, Problem 1: "Given a set of live ranges represented as a cyclic interval graph G, find a minimum
    register (color) assignment for the intervals in G such that overlapping intervals are assigned different
    registers."
  - Definitions 2.3.1 and 2.3.2, p. 23:
    - "The width of a cyclic interval graph G at time t, written as width(G,t), is the number of intervals covering t."
    - "The maximum width of a cyclic interval graph G, written Wmax(G), is the maximum width(G,t), for all t which is
      covered by some interval in G."
    - "The minimum width of a cyclic interval graph G, written Wmin(G), is the minimum width(G,t), for all t which is
      covered by some interval in G."
  - Theorem 2.3.1 and its proof, p. 24: "Let G be an interval graph containing no cyclic intervals. Then G is optimally
    colorable with Wmax(G) colors." The proof's left edge algorithm [HS71]: "Assume G spans from time 0 to time n.
    Starting from the left end (at time, t = 0), move from left to right along the time line. For each interval, I,
    which ends at t, release its color back to the pool of free colors. For each interval I beginning at t, give I a
    free color which is not being used by any interval covering t. Initially, the pool contains k = Wmax(G) free
    colors."
  - Theorem 2.3.2 and its proof, p. 24: "Let G be an interval graph containing cyclic intervals. Then G is optimally
    colorable with Wmax(G) ≤ k ≤ Wmax(G) + Wmin(G) colors." "Cut G at the point where it has the minimum width
    Wmin(G). Take the intervals covering the cutting point out of G and call the remaining part G'. Obviously, we can
    now treat G' as a non-cyclic interval graph. Coloring G' with the left-to-right algorithm guarantees it to be
    colored with no more than k' = Wmax(G) colors (Theorem 2.3.1). Then, it is trivial to see that we can use Wmin(G)
    more colors to color the removed intervals."
  - Definitions 3.2.1 and 3.2.2 and Theorem 3.2.1, p. 32:
    - "The fat spots of a cyclic interval graph G, written fatspots(G), is the set of all times ti where width(G, ti) =
      Wmax(G)."
    - "A fat cover of a cyclic interval graph, G, relative to interval I is a subgraph F (I ∈ F) of G that obeys the
      following two properties: (1) all intervals in F are non-overlapping, and (2) ∀ti ∈ fatspots(G), there exists an
      interval in F that covers ti."
    - "If a cyclic interval graph G is colorable in k = Wmax(G) colors, then for each cyclic interval Ic of G, there
      exists a fat cover for G relative to Ic, call it F, such that G − F is k − 1 colorable."
  - §3.2.3, pp. 32–33, the fat cover algorithm: "Given a graph G with m cyclic intervals Ci1, Ci2, ..., Cim, the
    algorithm proceeds in two phases.
    1. The first phase : attempts to use m colors to find a fat cover for each of the m cyclic intervals. At the ith
       step, a traversal from left to right is performed to find a fat cover for interval Ci (call this fat cover Fi).
       If such a cover is found, a traversal from right to left is performed which assigns the same new color Ci to all
       of the intervals in Fi.
    2. The second phase : If the first phase succeeds in coloring m cyclic intervals with m colors, then the second
       phase need only consider a reduced graph G' that contains no cyclic intervals. G' has a maximum width of w =
       Wmax(G) − m. The coloring of G' is guaranteed to use only w new colors (see proof of Theorem 2.3.1). A
       straightforward left-to-right algorithm is used to color the remaining non-cyclic intervals. [...] However, if
       the first phase fails to find a fat cover at some stage, the second phase simply colors the remaining cyclic
       intervals with new colors, and applies the simple left to right algorithm to color the remaining intervals. In
       this case, the resultant coloring may or may not be optimal."
  - M. Lam, PLDI 1988, §2.3, as quoted in the second stops' item 7: "the minimum degree of unrolling, u, to implement
    the same schedule is simply max_i q_i."
- **Change:**
  - The PLAN_LOCAL and PLAN_ROUTED items of one plan at a node are the intervals of one cyclic interval graph G.
  - G's time points are the node's operations, in program order, over the u executions of the unrolled template.
  - Each live range is one lifetime interval, from its write to its last read. A value of lag 0 has one per execution;
    a routed copy has one. A lifetime that crosses the end of the u executions is a cyclic interval.
  - An item of b blocks is b intervals of the same lifetime. Each color is a block of the space.
  - G is colored by the fat cover algorithm (§3.2.3), with the left edge algorithm for the non-cyclic remainder. This
    replaces `plan_before` and `mesh_graph_chains` for these items.
  - PLAN_ALONE items stay as they are.
  - The algorithm's text in §3.2.6 and Appendix A, and the open implementations of this kind of solver, are appended
    below once read. They are the implementer's reference for the steps the quotes above leave open.
