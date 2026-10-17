# Collective mechanism sources

The [user's requirements](collective-goals.md) define scope. These are reference
notes about upstream mechanisms and the retained bridge, not architecture mandates.
Descriptions of protocol decisions do not require preserving those decisions.

The licences D10, D11 and D16 are restored verbatim (2026-10-17; a07d7f66 had replaced their quotes with "follow the
caller's graph" prose, which stays below each), their sources as the record listed them:

- [Thakur] Thakur, Rabenseifner and Gropp, *Optimization of Collective Communication Operations in MPICH*, IJHPCA 2005
- [JACCL] Apple, MLX `mlx/distributed/jaccl/lib/jaccl/{mesh_impl.h,rdma.h,rdma.cpp}`, https://github.com/ml-explore/mlx
- [Megatron] Shoeybi et al., *Megatron-LM*, https://arxiv.org/abs/1909.08053

## D1. The verb is SEND into a posted RECV; there is no remote write

Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
specifies the supported Thunderbolt SEND/RECV substrate. Mesh uses those verbs and
registered memory; it does not emulate a remote-write API with caller handshakes.

## D2. Receives are posted independently of numerical production

TN3205 describes receiver credits. Mesh keeps planned writable receives posted;
independent TX/RX threads supply progress. This does not turn SEND into a hardware
one-sided write. The JAX authors' [collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
illustrates distinct receive storage and forwarding alongside numerical work.

## D3. Progress drains completion queues

TN3205 specifies nonblocking queue operations with CQ polling. Mesh dedicates
hardware threads to TX and RX; numerical work runs elsewhere.

## D4. Payload lands in registered operand backing

TN3205 permits a receive buffer within registered memory. Mesh's canonical page table
names those same pages. Reassigning a logical destination changes metadata; it does
not copy payload from a transport-only staging store.

## D5. Receive consumption has per-queue FIFO order

The MLX authors' [JACCL transport](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h)
uses work-request identifiers and CQ polling. Mesh preposts physical slots;
`wr_id` identifies the completed physical chunk. A tag in the same request
contains the sequence and source-chunk row in eight bytes. Setup maps each
peer source-chunk row directly to its local buffer, page-table entry and
publication flags. Sender publication order
need not match declaration order. Chunks from different sends may interleave
within a receive run. The canonical page table preserves logical chunk order;
contiguous consumers use the [indexed placement path](pages-and-functions.md#indexed-receive-runs-and-contiguous-consumers).

## D6. Paired send and receive frame counts match

TN3205 requires matching frame counts and specifies finite queue capacity. It does
not require one request per numerical partial. Mesh uses internally sized chunks
with an eight-byte sequence/source-chunk tag. Setup prepares the same request
extent on both ends of a queue direction. Fixed framing permits preposting
independently of producer order; logical operand size determines the number of
chunks and usable bytes. Neither posting interface shortens the prepared
request. Tail padding and tag bytes are explicit transport costs in the
[byte accounting](pages-and-functions.md#prepared-native-requests), without a
caller capacity query or a requirement to reshape numerical operands.

## D7. Publication does not await delivery

Barham et al., [Pathways](https://arxiv.org/abs/2203.12533) (MLSys, 2022), separates
asynchronous dataflow execution from communication progress. Mesh publication exposes
a section to its numerical uses and configured sends without a delivery wait.

## D8. Publication follows actual writes

Papadopoulos and Culler, [Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
(ISCA, 1990), associates presence with operands. Mesh marks a received section
present after completion and page-table assignment. Missing operands constrain
only the functions that read them; they do not block a progress thread.

## D9. Work requests retain their buffers through completion

The rdma-core authors' [ibv_post_recv manual](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3)
requires storage lifetime through work completion. This is a lifetime requirement,
not an instruction to wait for old readers before posting another receive. Distinct
live sections and automatic reference release satisfy the ownership relationship.
The [ownership relation](pages-and-functions.md#what-the-page-table-is) includes
numerical uses, external ownership and device uses.

## D10. Collectives stream by chunk

> For long messages and predefined reduction operations, we use Rabenseifner's algorithm for allreduce [19], which does a reduce-scatter followed by an allgather.
> — [Thakur]
> Bandwidth optimal all reduce for large messages: a reduce scatter followed by an all gather.
> — [JACCL] `mesh_impl.h`
> Recv completed. Reduce the peer's contribution into our chunk and, if there is more data to fetch from that peer, post another recv.
> — [JACCL] `mesh_impl.h`, `sum_scatter`

Licensed use, measured: the streaming reduce tiles that took the 12B cooperative FFN from 1.209x to 1.310x the M5
alone (metal-microbench b6efa23, 2026-09-12, 512-row tiles); libnccl-mesh's chunked rings and the Python surface's
tile-by-tile reductions (design/algorithm-sources.md#collectivereduce_scatter) are its current forms.

The MPI Forum's [collective communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and Rabenseifner's *Optimization of Collective Reduction Operations* (2004) supply
distinct movement and reduction compositions. Mesh binds each verb's index relation;
reductions use existing numerical functions over partial contributions.

## D11. One all-reduce per tensor-parallel block is a data dependency, not a barrier

> This approach splits both GEMMs in the MLP block across GPUs and requires only a single all-reduce operation in the forward pass (g operator)
> — [Megatron]

Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), and MLX's
[tensor-parallel layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py)
combine local contractions with communication at their algebraic boundaries.
Their examples do not mandate all-reduce for every caller or replicate unused values.

## D12. Completion status is not a consumer acknowledgement

TN3205 distinguishes send completion from successful remote consumption. Mesh uses
local device completion for buffer ownership and receive completion for publication.
It adds no remote consumer-retirement acknowledgement.

## D13. Connection metadata is setup work

TN3205 exchanges endpoint metadata out of band. Mesh realizes endpoint and transfer
configuration before transport execution; numerical functions do not perform pairing.

Configured links own independent connections and progress threads. Links sharing
a named device reuse its protection domain and registration. A connection's setup
exchange does not wait for every other link. None of these facts selects a
collective verb: the calling program defines its scatters and gathers explicitly.
See [configured peers](async-collectives.md#configured-peers).

## D14. Teardown retains outstanding device storage

JACCL's transport owns connection resources and registered memory. Mesh realizes
send and receive use tables from setup descriptions and retains abandoned section
backing through QP teardown.
This controls resource lifetime; it is not a default collective completion barrier.

Client retirement also releases metadata rows. `mesh_rows_alloc` records their
owner even when they have no payload pages; `mesh_retire` clears their row
allocation while marking only real operand storage closed. This covers root and
return rows when a dead client is replaced or its process-exit event is observed.
The same retirement routine claims the old client identity before clearing it;
connection teardown preserves process-exit observation. It does not cancel a still-attached
failed program's unissued uses or authorize reclaiming backing before QP teardown.

## D16. A vocabulary-parallel head is reduced before it is communicated

> We parallelize the input embedding weight matrix EH×v along the vocabulary dimension E = [E1 , E2 ] (column-wise).
> — [Megatron] §3
> However, for this case, the all-gather will communicate b × s × v elements (b is the batch-size and s is the sequence length) which is huge due to vocabulary size being large. To reduce the communication size, we fuse the output of the parallel GEMM [Y1 , Y2 ] with the cross entropy loss which reduces the dimension to b × s.
> — [Megatron] §3

Megatron's vocabulary-parallel example fuses its loss to reduce communicated values.
That is application algebra, not a requirement to add a vocabulary head or model
interpreter to the collective backend.
