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
