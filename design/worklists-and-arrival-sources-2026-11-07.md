Sources for the revisions of design/torch-capture-change-request.md, retrieved 2026-11-06/07 by a retrieval agent and recorded here as it wrote them. The local paths it names are the session's download directories.

# Worklists, frontiers, and partition arrival: sources and verbatim passages

Retrieved 2026-11-06/07. Sources appear in the order of the request (1–8). Each entry gives the bibliographic record, the copy that was read, and short quotations with their locations.

Conventions:
- Quotations are verbatim and deliberately short: the sentence or few sentences that state a point, or the specific lines of code. `[…]` marks an omission. Bold and italic are dropped.
- Each source was read in full around the passages quoted. Where only part of a passage is quoted, the location of the rest is given.
- Hyphens that pdftotext removed at line ends inside compound words were restored (for example "topology-driven"). These were checked against rendered page images where it mattered.
- Subscripts are written with an underscore (A_i = Aᵢ).
- Page numbers: for papers with printed page numbers, the printed number is given. Otherwise the page of the PDF that was read is given, and the entry says so.
- Local copies are in `lit3/dl-*` under `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/`. The pre-existing `lit3/mpi41/` directory, which this retrieval did not create, was not used.

---

## 1. Nasre, Burtscher, Pingali — Data-driven versus topology-driven (IPDPS 2013)

**Bibliographic.** R. Nasre, M. Burtscher, K. Pingali, "Data-Driven Versus Topology-driven Irregular Computations on GPUs," in *2013 IEEE 27th International Symposium on Parallel and Distributed Processing (IPDPS)*, Boston, MA, USA, May 2013, pp. 463–474. DOI 10.1109/IPDPS.2013.28 (from Crossref).

**Read.** Author copy at https://userweb.cs.txstate.edu/~mb92/papers/ipdps13b.pdf (IEEE PDFeXpress-certified, 12 pages), saved in `lit3/dl-nasre/`. The copy has no printed page numbers. "p. N" below is page N of that PDF.

### 1.1 Who adds items to the worklist, and how

> threads obtain work by pulling active nodes from the worklist, and they may add new active nodes to the worklist after processing an active node.

— §II-B, p. 2.

> Putting new active nodes onto the worklist can be done by atomically incrementing the length of the worklist and then writing an active node's identifier to that location in the worklist. However, this way of writing to the worklist is inefficient and does not scale on GPUs.

— §II-B, p. 2.

> The threads in a thread block can compute the total amount of work to be added by all threads in the thread block using a prefix sum (in shared memory) and then use a single atomic instruction per thread block to reserve space on the worklist.

— §II-B, p. 3.

> Based on this knowledge, the threads issue a single atomic instruction (atomicAdd) to increment the worklist size by that number of elements and then insert its elements into the obtained range.

— §III-C "Work chunking", p. 4.

> This way, all threads can push work onto the worklist in parallel without requiring atomic instructions. Note that (global) barriers are needed after the threads record their number of elements, after each step of the prefix sum, and before the first write to the worklist; in effect, fine-grain synchronization with atomics is replaced by coarse-grain synchronization with barriers.

— §III-D "Atomic-free worklist update", p. 5. The scan that produces the offsets is described on pp. 4–5.

> threads use the limited shared memory whenever there is space and push elements onto the global worklist otherwise.

— §III-B "Hierarchical worklist", p. 4.

> Work chunking does not improve performance (second bars) because the cost of batched atomics is still considerable.

— §VI-B, p. 8.

> Thus, each component gets pushed onto the worklist by all the nodes it contains.

— §VI-A (MST duplicates), p. 7.

### 1.2 In- and out-worklists between kernel launches

Fig. 1, "Pseudo code of data-driven implementation" (p. 4). The loop lines are transcribed from the rendered figure:

```
while in worklist not empty {
    process_kernel(in, out, ...)    // GPU
    transfer out worklist size      // CPU ← GPU
    clear in worklist               // GPU
    swap(in, out)                   // CPU
}
```

Fig. 2, "Pseudo code of the processing kernel" (p. 4). The push lines:

```
for each newitem in newitems
  push newitem onto out
```

> The while loop then repeatedly calls process_kernel until the input worklist is empty. The input and the output worklists are swapped (using pointers) and the accumulated work items in the output worklist become the input items for the next iteration.

— §III-A, p. 4.

> These items are added to the end of the worklist (e.g., using atomic instructions). Note that no global synchronization is necessary for extracting items from the input worklist because it is only read.

— §III-A, p. 4.

> Varying the kernel configuration is possible whenever a GPU kernel is invoked repeatedly from the host.

— §III-F, p. 5.

> This is achieved by modifying the topology-driven computation to push modified elements onto a worklist instead of (or apart from) processing them. Alternatively, an implementation may employ a separate kernel to populate the worklist.

— §V-A "Temporal hybrid", p. 7. This is about creating the input worklist when switching from topology-driven to data-driven.

> the host finds the overflow flag set, re-allocates the output worklist as well as the input worklist, does not swap the input and output worklists, and restarts the latest iteration with the overflow flag reset.

— §VI-F, p. 11.

### 1.3 How rounds end; flags and readiness

> This flag is reset at the start of each iteration and checked by the CPU after running process_kernel. If the flag is set, at least one thread modified the global data and another iteration is invoked.

— §IV-A (topology-driven `changed` flag), p. 6.

> Thus, there is a single writer per node, and a memory fence followed by the setting of a ready flag suffices to safely propagate information up the tree.

— §VI-A (Barnes-Hut), p. 7.

> In particular, all nodes are active like in a topology-driven approach but the threads wait until all children of the current node are ready akin to a data-driven approach. This way, only one kernel call is needed to complete the entire tree traversal. However, potential deadlock must be avoided [6].

— §VI-E, p. 10. The same section adds that this hybrid "performs substantially worse than the pure topology-driven approach because threads often wait rather than processing other, ready nodes."

> The idea is to invoke kernels with only as many blocks as can remain persistent in the SMs and continue operating on a worklist in a do-while loop. This avoids the need of returning to the host and the cost of repeated kernel invocations.

— §VII (persistent threads), p. 11.

---

## 2. Merrill, Garland, Grimshaw — Scalable GPU Graph Traversal (PPoPP 2012)

**Bibliographic.** D. Merrill, M. Garland, A. Grimshaw, "Scalable GPU Graph Traversal," in *PPoPP '12: Proceedings of the 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, New Orleans, LA, USA, Feb. 25–29, 2012, pp. 117–128. DOI 10.1145/2145816.2145832 (from Crossref).

**Read.** https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf (11 pages), saved in `lit3/dl-merrill/`. The copy has no printed page numbers. "p. N" below is page N of that PDF.

### 2.1 Writing the output frontier: prefix sums and scatter offsets

> Prefix sum is a bulk-synchronous algorithmic primitive that can be used to compute scatter offsets for concurrent threads given their dynamic allocation requirements.

— §1, p. 1.

> Given a list of allocation requirements for each thread, prefix sum computes the offsets for where each thread should start writing its output elements.

— §2.3, p. 3. Fig. 2 caption, p. 3: "Example of prefix sum for computing scatter offsets for run-length expansion. Input order is preserved."

> In the context of parallel BFS, parallel threads use prefix sum when assembling global edge frontiers from expanded neighbors and when outputting unique unvisited vertices into global vertex frontiers.

— §2.3, p. 3.

> uses local prefix sum in place of local atomic operations for determining enqueue offsets

— §2.2 "Our parallelization strategy", p. 3.

### 2.2 Expansion and contraction; queues across iterations

> Each iteration logically (1) expands vertices into an edge-frontier, i.e., neighbor expansion; and then (2) contracts them to a vertex-frontier, i.e., status-lookup and filtering.

— §2.2, p. 2.

> a frontier that is managed out-of-core is fully produced in off-chip memory for consumption by the next BFS iteration after a global synchronization step.

— §2.2, p. 3.

> For each iteration, tasks are mapped to unexplored vertices in the input vertex-frontier queue. Their neighbors are inspected and the unvisited ones are placed into the output vertex-frontier queue for the next iteration.

— §2.2, p. 3.

> This kernel requires 2n global storage for input and output vertex queues. The roles of these two arrays are reversed for alternating BFS iterations.

— §5.1 "Expand-contract", p. 8.

> The remaining valid neighbors are placed into the outgoing vertex queue using another local prefix sum to compute global enqueue offsets.

— §5.3 "Two-phase", p. 8. The same section says the expansion kernel "performs two overlapped local prefix sums to compute scatter offsets for the expanded neighbors into the global edge queue."

> If the edge queue for a given BFS iteration contains more vertex identifiers than resident threads, we invoke the two-phase implementation for that iteration.

— §5.4 "Hybrid", p. 8.

### 2.3 Remote queues, and the end of a round

> As adjacency lists are expanded, neighbors are enqueued to the processor that owns them. The synchronization between BFS levels occurs after the expansion phase.

— §2.2 "Distributed parallelizations", p. 3.

> 3. Barrier across all GPUs. The sorting must be completed on all GPUs before any can access their bins on remote peers. The host program uses this opportunity to terminate traversal if all bins are empty on all GPUs.

— §6.1 (multi-GPU), p. 10. Step 4 of the same list: "Kernel invocation simply uses remote pointers that reference the appropriate peer bins."

---

## 3. Wang, Davidson, Pan, Wu, Riffel, Owens — Gunrock (PPoPP 2016)

**Bibliographic.** Y. Wang, A. Davidson, Y. Pan, Y. Wu, A. Riffel, J. D. Owens, "Gunrock: A High-Performance Graph Processing Library on the GPU," in *PPoPP '16: Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, Barcelona, Spain, March 2016, pp. 1–12 (per Crossref). DOI 10.1145/2851141.2851145.

**Read.** arXiv:1501.05387v6, 22 Feb 2016 ("accepted by PPoPP'16"; the first page carries the PPoPP '16 DOI), saved in `lit3/dl-gunrock/`. "p. N" below is page N of that PDF.

### 3.1 Frontiers

> our data-centric model's key abstraction is the frontier, a subset of the edges or vertices within the graph that is currently of interest. All Gunrock operations are bulk-synchronous and manipulate this frontier, either by computing on values within it or by computing a new frontier from it.

— §1, p. 1.

### 3.2 Advance and filter

> An advance step generates a new frontier from the current frontier by visiting the neighbors of the current frontier.

— §4.1, p. 3.

> A filter step generates a new frontier from the current frontier by choosing a subset of the current frontier based on programmer-specified criteria. Though filtering is an irregular operation, using parallel scan for efficient filtering is well-understood on GPUs.

— §4.1, p. 3.

> Gunrock also supports a non-idempotent advance, which internally uses atomic operations to guarantee each element appears only once in the output frontier.

— §4.5, p. 6.

> First, SSSP enumerates the sizes of the frontier's neighbor list of edges and computes the length of the output frontier.

— §4.1, p. 4.

> Merrill et al.'s expand maps nicely to Gunrock's advance operator, and contract to Gunrock's filter operator.

— §5.1, p. 7.

### 3.3 Round structure

Figure 3 (p. 5). These are the input and output queue arguments of the advance and filter kernel launches:

```
graph_slice->ping_pong_working_queue[selector],
graph_slice->ping_pong_working_queue[selector^1],
```

Algorithm 1 (p. 4), enactor loop:

```
25: while P.frontier.Size() > 0 do
26:   ADVANCE(G, P, UpdateLabel, SetPred)
27:   FILTER(G, P, RemoveRedundant)
```

> Gunrock primitives are assembled from a sequence of these steps, which are executed sequentially: one step completes all of its operations before the next step begins. Typically, Gunrock graph primitives run to convergence, which on Gunrock usually equates to an empty frontier

— §4.1, p. 3. The paragraph continues on p. 4: "Programmers can also use other convergence criteria such as a maximum number of iterations or volatile flag values that can be set in a computation step."

---

## 4. Dong, Feng, Guessous, Liang, He — FlexAttention (arXiv:2412.05496)

**Bibliographic.** J. Dong, B. Feng, D. Guessous, Y. Liang, H. He, "Flex Attention: A Programming Model for Generating Optimized Attention Kernels," arXiv:2412.05496v1 [cs.LG], 7 Dec 2024. This is the only arXiv version. Web search results say it was presented at MLSys 2025; the proceedings entry was not read.

**Read.** https://arxiv.org/pdf/2412.05496 (13 pages), saved in `lit3/dl-flexattn/`. "p. N" below is page N of that PDF. Identifier spellings were checked against rendered pages.

### 4.1 Paper: how a BlockMask is built, and when

> FlexAttention leverages block sparsity and employs a pre-computed BlockMask. […] With the help of torch.vmap, our create_block_mask utility automatically generates BlockMask from user defined mask_mod.

— §1.1, p. 2.

> We pre-compute a block mask during compilation time to remove the runtime overhead. BlockMask first splits the score matrix into blocks along the Q_LEN and KV_LEN dimensions. Then, BlockMask specifies a block as non-computed if all score scalars in it are masked as -inf.

— §4.2, p. 5.

> Concretely, BlockMask contains two tensors, a kv_num_block of shape B × H × Num_Row and a kv_indices of shape B × H × Num_Row × Num_Col. […] kv_num_block stores the number of non-zero blocks for each row and kv_indices stores the indices of these non-zero blocks.

— §4.2, p. 5. Figure 3 (p. 5) shows `BlockMask = (full_kv_num_blocks, full_kv_indices, kv_num_blocks, kv_indices)`.

> It first adjusts the workload in each GPU block according to the kv_num_block, which specifies the number of score matrix blocks that are not masked out. Then, it uses the KV Indices to map to the next block to be processed.

— §4.2 "BlockMask Guided Indirect Memory Access", p. 6.

> Each SM iterates along the KV_LEN dimension and processes a row of blocks (Figure 4).

— §4.2, p. 6.

> During runtime, FlexAttention relies on the converted kv_index to access tokens for a specific sentence from the physical KV cache. Note that we keep the kv_num_blocks unchanged since paged attention does not change the number of unmasked blocks.

— §5.1 "Paged Attention Variants", p. 7.

### 4.2 Reference code: torch/nn/attention/flex_attention.py

**Repository.** pytorch/pytorch, commit `08d8c569422ce8058f89d6cfb6735b7c16233aad`. This is the latest commit touching the file, dated 2026-11-03.

**License.** The repository `LICENSE` at that commit is a 3-clause BSD license ("Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met"). GitHub reports the SPDX id as NOASSERTION.

**Names.** `create_block_mask`, `_convert_mask_to_block_mask` and `_dense_to_ordered` still exist under those names.

`create_block_mask`, docstring example, lines 2065 and 2069 (the mask is built, then passed to the kernel call):

```python
            block_mask = create_block_mask(causal_mask, 1, 1, 8192, 8192, device="cuda")
            output = flex_attention(query, key, value, block_mask=block_mask)
```

`create_block_mask`, body, lines 2111–2112 and 2118:

```python
    mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
    block_mask = _create_sparse_block_from_block_mask(
```

`create_mask`, lines 1920–1921 (`mask_mod` vmapped over b, h, q_idx, kv_idx):

```python
            mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
            mask = mask_mod(b, h, m, n)
```

`_convert_mask_to_block_mask`, lines 1791 and 1796–1797:

```python
    mask_block_sum = mask.sum(
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
```

`_dense_to_ordered`, lines 534 and 536:

```python
    num_blocks_in_row = dense_mask.sum(dim=-1)
        col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
```

`BlockMask` docstring, lines 927–928:

```
    `col_indices[i]` is the sequence of block positions for row i. The values of
    this row after `col_indices[i][num_blocks_in_row[i]]` are undefined.
```

The kernel template that reads these tensors is `torch/_inductor/kernel/flex/templates/flex_attention.py.jinja`, at the same commit. Lines 141 and 143–144:

```python
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
```

---

## 5. MPI 4.1 — Partitioned Communication (§4.2)

**Bibliographic.** Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard, Version 4.1*, November 2, 2023. Chapter 4, "Partitioned Point-to-Point Communication"; §4.2, "Semantics of Partitioned Point-to-Point Communication".

**Read.** https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf, downloaded fresh into `lit3/dl-mpi41/`. Page numbers below are printed page numbers; PDF page = printed + 42. All of §4.2 (pp. 114–121) was read. MPI_PARRIVED's text on pp. 120–121 was read in full; the sentences that state each rule are quoted.

### 5.1 PREADY on the sender, PARRIVED on the receiver

> A call to MPI_PREADY notifies the MPI library that a specified portion of the data buffer (a specific partition) is ready to be sent. Notification of partial completion can be done via fine-grained MPI_PARRIVED calls at the receiver before a final MPI_TEST/MPI_WAIT on the request itself

— §4.2, p. 114.

> MPI is free to choose how many transfers to do within a partitioned communication send independent of how many partitions are reported as ready to MPI through MPI_PREADY calls. Aggregation of partitions is permitted but not required. Ordering of partitions is permitted but not required.

— §4.2, Rationale, p. 115.

> For send-side operations, neither initializing nor starting the operation enables transfer of any part of the user buffer.

— §4.2.1, p. 116.

### 5.2 MPI_PREADY, MPI_PREADY_RANGE, MPI_PREADY_LIST

C bindings, pp. 118–119:

```c
int MPI_Pready(int partition, MPI_Request request)
int MPI_Pready_range(int partition_low, int partition_high, MPI_Request request)
int MPI_Pready_list(int length, const int array_of_partitions[], MPI_Request request)
```

> MPI_PREADY is a send-side call that indicates that a given partition is ready to be transferred.

— p. 119.

> After a call to MPI_START/MPI_STARTALL, all partitions associated with that operation are inactive. A call to MPI_PREADY marks the indicated partition as active. Calling MPI_PREADY on an active partition is erroneous.

— p. 119.

> A call to MPI_PREADY_RANGE has the same effect as calls to MPI_PREADY, executed for i=partition_low, . . ., partition_high, in some arbitrary order.

— p. 119.

> A call to MPI_PREADY_LIST has the same effect as calls to MPI_PREADY, executed for the partitions specified in the range array_of_partitions[0] ,. . ., array_of_partitions[count − 1] of the array_of_partitions, executed in some arbitrary order.

— p. 120.

### 5.3 MPI_PARRIVED and the receiver's access rule

```c
int MPI_Parrived(MPI_Request request, int partition, int *flag)
```

The OUT argument is defined as: "flag — true if operation completed on the specified partition, false if not (logical)" (p. 120).

> The MPI_PARRIVED function can be used to determine if the message data for the indicated partition has been received into the receive buffer. Upon success, the receiver becomes free to access the indicated partition (as well as any others that previously completed for that operation).

— §4.2.2, p. 120.

> A call to MPI_PARRIVED on an active partitioned communication request returns flag = true if the operation identified by request for the specified partition is complete. The request is not marked as complete/inactive by this procedure.

— §4.2.2, p. 121.

> Repeated calls to MPI_PARRIVED with the same request and partition arguments will eventually return flag = true if the corresponding partitioned send operation has been started and all send partitions have been marked as ready.

— §4.2.2, p. 121.

> A high quality implementation will eventually return flag = true from MPI_PARRIVED after all of the corresponding MPI_PREADY calls have been made for a receive-side partition, even if other send partitions are not yet marked as ready.

— §4.2.2, Advice to implementors, p. 121.

> The completion of a partitioned receive operation through MPI_WAIT or MPI_TEST indicates that the receive buffer contains all of the partitions.

— §4.2.2, p. 120.

> Partitions are contiguous in memory, there is no padding in between them.

— §4.2.3, p. 121.

> It uses a naive flag based method to test for multiple completed partitions per thread. Note that this means that some threads may be busy polling for completion of assigned partitions when partitions are available to work on that were not assigned to the polling threads in this example.

— §4.3.3, introducing Example 4.4, p. 124. The receive loop of Example 4.4 (p. 126) has these lines:

```c
MPI_Parrived ( request , j , & part_flag );
if ( part_flag && part1_complete == 0)
```

---

## 6. GPU partitioned communication: who writes what, and how the receiving GPU learns of arrival

### 6.1 Temuçin, Schonbein, Levy, Sojoodi, Grant, Afsahi — GPU-initiated MPI Partitioned (SC24-W / ExaMPI 2024)

**Bibliographic.** Y. H. Temuçin, W. Schonbein, S. Levy, A. Sojoodi, R. E. Grant, A. Afsahi, "Design and Implementation of MPI-Native GPU-Initiated MPI Partitioned Communication," in *SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis*, Nov. 2024, pp. 436–447. DOI 10.1109/SCW63240.2024.00065 (from Crossref).

**Read.** Author copy at https://queensu.ca/academia/afsahi/pprl/papers/ExaMPI-2024.pdf (printed pages 1–12), saved in `lit3/dl-gpupart/`.

> The receiver can, but is not required to, call MPI_Parrived in a parallel region to check if a partition has arrived.

— §II-B1, p. 2.

> Consequently, there is no guarantee a receiver is ready to receive data. One solution is to block on Pready until the receiver is initialized [10], but this will stall the kernel, requires the GPU to handle progression, and potentially increases the chance of deadlock [20].

— §II-B2, p. 2. The same section says the proposed MPIX_Pbuf_prepare is meant "to provide a guarantee to the sender that the remote buffer is ready to receive."

> During the data transfer phase, the sender will write to the partition status flags to notify the receiver that communication for that partition has completed.

— §IV-A2, p. 4.

> For progression, a set of flags is created in pinned host memory which are accessible by the device.

— §IV-A3, p. 4.

> Attached to the callback of the put operation is another ucp_put_nbx call which marks a partition as received on the receiver side. This is required so that MPI_Parrived can provide fine-grained information on partition arrival.

— §IV-A4, p. 4. The next sentence cites "IBV_WR_RDMA_WRITE_WITH_IMM [10]" as the receive-side completion that UCX lacks.

> For the Progression Engine approach, a CUDA thread updates a flag in host memory to notify the Progression Engine that an MPIX_Pready call is pending. Upon detecting this notification, the MPI Progression Engine issues an MPI_Pready on the host

— §IV-A4, pp. 4–5.

> In addition to block-level bindings, we have counters in GPU global memory that we atomically increment until a threshold is hit, before writing into host memory.

— §IV-A4, p. 5.

> For MPI_Parrived's host binding we simply poll on the receive-side completion flags. The device version polls a flag in GPU global memory as the cost of accessing global memory is much lower than host memory. However, as our receive-side completion flags are always populated in host memory, we issue a memory copy to the device in MPI_Wait as partitions arrive.

— §IV-A4, p. 5.

> the receiver counts the arrived flags until it matches the number of partitions.

— §IV-A5 (MPI_Wait), p. 5.

> For the intra-node case where we use a Kernel Copy, we are in a gray area as we conduct the data transfer within MPIX_Pready.

— §VI-A2, p. 8.

### 6.2 NVIDIA MPI-ACX prototype (device-side `MPIX_Pready` / `MPIX_Parrived`): code

**Repository.** https://github.com/NVIDIA/mpi-acx, commit `95a38d8bd8c0150de1d0b3c22367e29fc9266c10`. This is the latest commit, dated 2023-04-04. The author is Jim Dinan.

**License.** BSD-3-Clause (SPDX header; LICENSE "Copyright (c) 2022, NVIDIA Corporation & Affiliates").

**Local copy.** `lit3/dl-mpiacx/`.

> The proxy maintains a set of flags in host pinned memory that are used to coordinate with the GPU. […] a partitioned operation uses one flag per partition.

— README.md, "How does it work?", lines 109–113.

> It's also possible to create a deadlock when the same kernel performs `MPIX_Pready` operations and also polls on on an `MPIX_Parrived` operation.

— README.md, "Caveats", lines 152–153. The workaround given there is "using a cooperative kernel launch that limits grid size, or by marking ready and waiting on partitions in separate kernels."

`src/partitioned.cu`. Line 204 (device MPIX_Pready), and lines 221 and 228 (device MPIX_Parrived):

```c
    preq->flags[preq->idx[partition]] = MPIACX_OP_STATE_PENDING;
    state = preq->flags[preq->idx[partition]];
    *flag = (state == MPIACX_OP_STATE_COMPLETED);
```

`src/partitioned.cu`, line 135, in `MPIX_Start` for a receive:

```c
            mpiacx_state->flags[ireq->partitioned.flag_idx[i]] = MPIACX_OP_STATE_ISSUED;
```

`src/init.cpp`, CPU proxy thread. Lines 84 and 86 (send), and lines 106 and 110 (receive):

```c
                        MPI_Pready(op->partitioned.partition, op->partitioned.request);
                        mpiacx_state->flags[i] = MPIACX_OP_STATE_COMPLETED;
                    MPI_Parrived(op->partitioned.request, op->partitioned.partition, &flag);
                        mpiacx_state->flags[i] = MPIACX_OP_STATE_COMPLETED;
```

`src/init.cpp`, lines 220–221 (the flags are host memory mapped into the device):

```c
    CUDA_CHECK(cudaHostAlloc((void **)&mpiacx_state->flags,
                             sizeof(int) * mpiacx_state->nflags, cudaHostAllocMapped));
```

`test/src/ring-partitioned.cu`, lines 44–46 (receiving kernel):

```c
    do {
        MPIX_Parrived(preq, threadIdx.x, &flag);
    } while (!flag);
```

### 6.3 Dinan — MPI Forum Hybrid WG proposal slides (2021)

**Source.** J. Dinan, "Partitioned Communication CUDA Bindings," slides, updated 11/3/21. In the repository mpiwg-hybrid/hybrid-issues at commit `f25931e3dc52e3e00822e9caaa19efea03985611`, file `slides/2021-11-03 -- Partitioned Communicaiton CUDA Bindings.pdf` (the spelling is the original). The repository declares no license. The slides are linked from issue #4, "Accelerator Bindings for Partitioned Communication". **Local copy:** `lit3/dl-mpiwg/`.

- Slide 6: "MPI_Prequest for an MPI_Precv operation may contain: Pointer to flag in memory"
- Slide 8: "MPI_Pready operation may involve — ibverbs: Copy pre-prepared WQE to SQ and ring doorbell — libfabric: Increment event counter"
- Slide 9: "MPI_Parrived operation may involve — libibverbs: Polling a CQ, e.g. if write with immediate is used — Checking flags for completion of the given operation"
- Slide 3: `int MPI_[start,wait][_all](...)` is annotated "Keep host only", and `MPI_Pready` is annotated "Add device bindings".

### 6.4 Bridges, Skjellum, Suggs, Schafer, Bangalore — GPU triggering APIs (EuroMPI 2024)

**Bibliographic.** P. G. Bridges, A. Skjellum, E. D. Suggs, D. Schafer, P. V. Bangalore, "Understanding GPU Triggering APIs for MPI+X Communication," in *Recent Advances in the Message Passing Interface* (EuroMPI 2024), LNCS, Springer, 2024, pp. 39–55. DOI 10.1007/978-3-031-73370-3_3 (from Crossref).

**Read.** arXiv:2406.05594v3, 31 Jul 2024, saved in `lit3/dl-gpupart/`. "p. N" below is page N of that PDF.

> A specific nuance in this approach is that both MPIX_Parrived and MPIX_Pready can be called from the host and the CUDA device.

— §2.3, p. 6.

> The main exception is partitioned communication, where receives completion can be tested on the GPU using Parrived, but there is no equivalent call for testing for the completion of sends initiated from the kernel using Pready

— p. 13.

### 6.5 Temuçin, Levy, Schonbein, Grant, Afsahi — arrival over InfiniBand verbs (CLUSTER 2023; CPU receiver)

**Bibliographic.** Y. H. Temuçin, S. Levy, W. Schonbein, R. E. Grant, A. Afsahi, "A Dynamic Network-Native MPI Partitioned Aggregation Over InfiniBand Verbs," in *2023 IEEE International Conference on Cluster Computing (CLUSTER)*, pp. 259–270. DOI 10.1109/CLUSTER52292.2023.00029 (from Crossref). This is reference [10] in §6.1.

**Read.** OSTI copy https://www.osti.gov/servlets/purl/2431172 (SAND2023-08000C, 12 pages), saved in `lit3/dl-cluster23/`. "p. N" below is page N of that PDF.

> MPI_Pready executes an atomic add-and-fetch on an array of flags we have for our transport partitions.

— §IV-A, p. 4.

> This work request uses the opcode IBV_WR_RDMA_WRITE_WITH_IMM, so that we can use the immediate value to encode which user partitions are contained within the transport partition that is being posted.

— §IV-A, p. 4.

> If we obtain a successful WC, we update the receiving user partition flags. MPI_Parrived initially checks receiving user partition flag to determine if the partition has arrived.

— §IV-A, p. 4.

### 6.6 Namashivayam et al. — stream-triggered MPI (2022; not partitioned)

**Bibliographic.** N. Namashivayam, K. Kandalla, T. White, N. Radcliffe, L. Kaplan, M. Pagel, "Exploring GPU Stream-Aware Message Passing using Triggered Operations," arXiv:2208.04817v1 [cs.DC], 9 Aug 2022.

**Read.** https://arxiv.org/pdf/2208.04817, saved in `lit3/dl-gpupart/`. "p. N" below is page N of that PDF.

> The operation gets executed when the associated trigger counter object in the DWQ operation reaches a required threshold value. The GPU CP is responsible for performing this update.

— §II-C, p. 3.

> Missing DWQ features required by the MPI ST semantics, like the tagged and untagged receive operations, are emulated with an internal asynchronous progress thread per MPI process.

— §II-C, p. 3.

> the GPU CP using the enqueued waitValue operation monitors the completion counter object associated with the DWQ operation to wait for its completion.

— §II-E, p. 3.

### 6.7 Bridges et al. — CPU-free two-sided MPI (CLUSTER 2026; not partitioned)

**Bibliographic.** P. G. Bridges, D. Schafer, J. Lange, J. B. White III, A. Skjellum, E. Suggs, T. Hines, P. Bangalore, M. G. F. Dosanjh, W. Schonbein, "Co-Design and Evaluation of a CPU-Free MPI GPU Communication Abstraction and Implementation," in *2026 IEEE International Conference on Cluster Computing (CLUSTER)*, pp. 289–301. DOI 10.1109/CLUSTER68337.2026.00035 (from Crossref).

**Read.** arXiv:2602.15356v2, 5 Mar 2026, saved in `lit3/dl-bridges26/`. "p. N" below is page N of that PDF.

> As a result, GPU memory operations cannot directly read CXI counters without CPU involvement.

— §II, p. 3.

> when the fi_write associated with a send places data in this buffer, the local fi_atomic operation will update the local completion buffer associated with the persistent receive.

— §V-A, p. 6.

> MPI_Enqueue_Wait enqueues a kernel that polls for the relevant local completion buffer to be incremented to the appropriate value by an fi_atomic operation.

— §V-A, p. 6.

### 6.8 Not read (for item 6)

- **MPICH documentation** on device-side `MPIX_Pready`/`MPIX_Parrived`: none was found. A web search turned up only stream/enqueue extensions. Not read.
- **Open MPI documentation** for GPU partitioned communication: not found, not read. The Open MPI UCX component source of §6.1 was not read.
- **`MPI_Pbuf_prepare` proposal** (mpi-forum/mpi-standard PR #264; mpi-issues #302): not read.
- **Temuçin, Grant, Afsahi, ICPP 2022 micro-benchmark paper:** not read.
- **Namashivayam et al., later papers** ("Exploring Fully Offloaded GPU Stream-Aware Message Passing"; PACT 2025): not read.

---

## 7. NVSHMEM API documentation

**Source.** NVIDIA, *NVSHMEM APIs* documentation, "latest" pages (page metadata `version_match = '3.8.0'`; "Copyright © 2022-2026, NVIDIA Corporation"), retrieved into `lit3/dl-nvshmem/`:
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html

### 7.1 Put-with-signal and the data-before-signal guarantee

```c
__device__ void nvshmem_putmem_signal(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
__device__ void nvshmemx_putmem_signal_block(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
```

> The put-with-signal routines provide a method for copying data from a contiguous local data object to a data object on a specified PE and subsequently updating a remote flag to signal completion.

— signal.html, "NVSHMEM_PUT_SIGNAL", Description.

> The completion of signal update based on the sig_op signal operator using the signal flag on the remote PE indicates the delivery of its corresponding dest data words into the data object on the remote PE.

— signal.html, "NVSHMEM_PUT_SIGNAL", Description.

> Without a memory-ordering operation, there is no implied ordering between the signal update of a put-with-signal routine and another data transfer.

— signal.html, "NVSHMEM_PUT_SIGNAL", Notes.

> two successive nonblocking put-with-signal routines, or a nonblocking put-with-signal routine with another data transfer may deliver data out of order unless a call to nvshmem_fence is introduced between the two calls.

— signal.html, "NVSHMEM_PUT_SIGNAL_NBI".

> NVSHMEM_SIGNAL_SET — An update to signal data object is an atomic set operation.
> NVSHMEM_SIGNAL_ADD — An update to signal data object is an atomic add operation.

— signal.html, "Available Signal Operators".

### 7.2 nvshmemx_signal_op

```c
__device__ inline void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
```

> The nvshmemx_signal_op operation atomically updates sig_addr with signal using operation sig_op on the specified PE.

— signal.html, "NVSHMEMX_SIGNAL_OP".

### 7.3 nvshmem_signal_wait_until

```c
__device__ uint64_t nvshmem_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_val)
```

> The nvshmem_signal_wait_until operation blocks until the value contained in the signal data object, sig_addr, at the calling PE satisfies the wait condition.

— sync.html, "NVSHMEM_SIGNAL_WAIT_UNTIL".

> Implementations must ensure that nvshmem_signal_wait_until does not return before the update of the memory indicated by sig_addr is fully complete.

— sync.html, "NVSHMEM_SIGNAL_WAIT_UNTIL".

### 7.4 Counted signals (one counter, several producers)

> After those destination bytes are delivered, it atomically adds bytes to signal_addr on the same PE. Multiple counted writes may contribute to one counted signal, including writes from different producers, provided that the application accounts for every contribution in the expected byte-count epoch.

— signal.html, "NVSHMEMX_PUTMEM_SIGNAL_COUNTED_NBI_BLOCK". The Notes there say it requires "CUDA Compute Fabric Transport (CFT) counted-operation support".

> Directly polling signal_addr is not equivalent to this routine because it does not guarantee payload visibility.

— signal.html, "NVSHMEMX_SIGNAL_COUNTED_WAIT_UNTIL".

### 7.5 Fence and quiet

> nvshmem_fence guarantees order of delivery, not completion.

— ordering.html, "NVSHMEM_FENCE".

> a single thread in the CTA may call nvshmem_fence before issuing a signal to order the signal after the preceding Put operation.

— ordering.html, "NVSHMEM_FENCE", Notes.

> For example, one might use nvshmem_quiet to await delivery of a block of data before issuing another Put or nonblocking Put routine, which sets a completion flag on another PE.

— ordering.html, "NVSHMEM_QUIET", Notes.

---

## 8. Tile-level wait and notify: TileLink and Triton-distributed

### 8.1 TileLink (arXiv:2503.20313)

**Bibliographic.** S. Zheng, J. Fang, X. Zheng, Q. Hou, W. Bao, N. Zheng, Z. Jiang, D. Wang, J. Ye, H. Lin, L.-W. Chang, X. Liu, "TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives," arXiv:2503.20313v3 [cs.DC], 3 Apr 2025. Web search lists it in *Proceedings of Machine Learning and Systems 7 (MLSys 2025)*; that copy was not read.

**Read.** https://arxiv.org/pdf/2503.20313 (20 pages), saved in `lit3/dl-tilelink/`. Printed page numbers equal PDF page numbers.

Table 3, "Tile-centric primitives in TileLink" (p. 5), signal rows. The "nofity" is in the original.

| Usage | Explanation |
|---|---|
| producer_tile_nofity(tile_id, mode) | Mark producer tile done and notify its consumer tile, consumer tile is marked ready when all the producer tiles it depends on are done |
| consumer_tile_wait(tile_id) | Consumer tile is blocked until all its dependent producer tiles done |
| peer_tile_notify(tile_id, rank) | Mark current tile done and notify its peer tiles in another rank |
| peer_tile_wait(tile_id, rank) | Block current tile until its peer tile in another rank is done |

> The notify primitives carry release semantics, ensuring that any memory access occurring before producer(peer)_tile_notify and rank_notify cannot be executed after these notify primitives. Conversely, the wait primitives carry acquire semantics, ensuring that any memory access following consumer(peer)_tile_wait and rank_wait cannot be executed before these wait primitives.

— §3.2.1, p. 7.

> In the pull mode, the producer reads data from all other ranks and notifies its consumer using local barriers. In contrast, the push mode allows the producer to write local data to all other ranks while notifying its remote consumers of the data's arrival.

— §3.2.2, p. 7.

> Channel mapping assigns each tile_id to a communication barrier.

— §4.1, p. 7.

> To support dynamic mapping, TileLink transforms these mappings into lookup tables, whose values can be filled at runtime, while the access operations to these lookup tables are determined at compile time.

— §4.1, p. 8.

> the frontend primitives with memory consistency semantics are compiled to corresponding device instructions such as ld.global.acquire and red.release.

— §4.2, p. 8.

Figure 4, GEMM+RS kernel (p. 9), with the original line numbers:

```
9   producer_tile_nofity([tid_m, tid_n], "p2p")
16  consumer_tile_wait(tid_m_global, tid_n)
26  peer_tile_notify([tid_m_global, tid_n], to_rank)
```

### 8.2 Triton-distributed (arXiv:2504.19442)

**Bibliographic.** S. Zheng, W. Bao, Q. Hou, X. Zheng, J. Fang, C. Huang, T. Li, H. Duanmu, R. Chen, R. Xu, Y. Guo, N. Zheng, Z. Jiang, X. Di, D. Wang, J. Ye, H. Lin, L.-W. Chang, L. Lu, Y. Liang, J. Zhai, X. Liu, "Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler," arXiv:2504.19442v3 [cs.DC], 5 Jun 2025.

**Read.** https://arxiv.org/pdf/2504.19442 (25 pages), saved in `lit3/dl-tritondist/`. Printed page numbers equal PDF page numbers.

Table 1, "Our Communication Primitives" (p. 5), selected rows:

| Primitive | Explanation |
|---|---|
| putmem_signal | Blocking put data and write signal to remote device |
| signal_wait_until | Wait local signal until condition is meet |
| wait | Locally wait a signal until it equals to some given value |
| consume_token | used with wait primitive to create data dependency |
| notify | Notify a remote signal, similar to signal_op primitive |

> A signal is a data object that resides in symmetric memory. There is a fixed set of operations to manipulate signals, including setting the value of a signal, increasing the value of a signal, checking the value of a signal, and performing a spin-lock on a given signal.

— §2.1, p. 5.

> The first primitive wait produces a token related to a signal, while the second primitive consume_token consumes the token and creates data dependency between the token and the following data load.

— §2.3, p. 6.

Figure 4 (p. 7). Consumer GEMM lines, transcribed from the rendered figure:

```
token = wait(
  signal + pid_m, 1, "gpu", "acquire", waitValue=1)
a_ptrs = consume_token(A + offs_A, token)
```

Producer lines from the same figure:

```
signal_wait_until(signal + seg, EQ, 1)
putmem_signal(
```

> Using push mode, we can omit one synchronization operation but the data arrival order cannot be controlled; while using pull mode, we need an additional synchronization but the data arrival order can be controlled.

— §3.2, p. 9. Algorithm 1 (p. 9), line 6: `set_signal(remote_sig) // Notify the consumer`.

> To implement LL protocol, we store data and flags together into an 8 bytes data chunk and send the data directly to remote ranks, while the remote receivers use a spin-lock to check if the flag is the same as expectation to tell if the data has arrived.

— §3.4, p. 11.

**Reference code.** Repository ByteDance-Seed/Triton-distributed, commit `7908e4ea9010bb2238f9f05b28c904fd4b71c60a` (2026-09-18). License: MIT. Local copy: `lit3/dl-tritondist-code/`.

`docs/python-api/triton-dist.language.rst`:
- line 16: "Wait for distributed barriers."
- line 23: ":returns: Token for consume_token"
- line 27: "Consume a token to ensure proper ordering with wait operations."
- line 57: "Send a signal to a remote rank."

`lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp`. The `wait` lowering is a per-lane spin on an acquire load (lines 189 and 232–234):

```cpp
    const std::string ld_ptx = "ld.global."s + semantic + "."s + scope + ".b"s +
        "  "s + ld_ptx + " %tmp0, [%addr0];                             \n\t"s +
        "  setp.eq.b"s + bit_w + " %p0, %tmp0, $2;                      \n\t"s +
        "  @!%p0 bra.uni waitLoop;                                      \n\t"s +
```

`consume_token` lowers to its input (line 262):

```cpp
    rewriter.replaceOp(op, adaptor.getInput());
```

`notify`, intra-node: thread 0 emits a membar before the store (line 322). Inter-node: it calls `nvshmemx_signal_op` (line 364).

```cpp
      const std::string memBarPtx = "membar." + membarScope + ";\n\t";
          CreateNVSHMEMOp(rewriter, op, "nvshmemx_signal_op", libname, libpath,
```

---

## Summary of what was and was not read

| # | Source | Status |
|---|---|---|
| 1 | Nasre, Burtscher, Pingali, IPDPS 2013 | Read (author copy) |
| 2 | Merrill, Garland, Grimshaw, PPoPP 2012 | Read (NVIDIA copy) |
| 3 | Wang et al., Gunrock, PPoPP 2016 | Read (arXiv v6) |
| 4 | Dong et al., FlexAttention | Read (arXiv v1). Code read at pytorch@08d8c569 (BSD-3-Clause-style) |
| 5 | MPI 4.1 §4.2 | Read |
| 6 | Temuçin et al. SC24-W; MPI-ACX code; Dinan 2021 slides; Bridges et al. 2024; Temuçin et al. CLUSTER 2023; Namashivayam et al. 2022; Bridges et al. 2026 | Read. Not read: MPICH and Open MPI documentation (none found), the Pbuf_prepare PR/issue, Temuçin et al. ICPP 2022, Namashivayam et al.'s later papers |
| 7 | NVSHMEM API documentation (latest, 3.8.0) | Read |
| 8 | TileLink v3; Triton-distributed v3, plus code at 7908e4ea (MIT) | Read |

Publisher versions (ACM DL, IEEE Xplore, the MLSys 2025 proceedings) were not read. The author, arXiv or OSTI copies named above were read instead.
