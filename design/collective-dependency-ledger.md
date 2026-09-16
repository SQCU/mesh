# Collective mechanism sources

The [user's requirements](collective-goals.md) define scope. These are reference
notes about upstream mechanisms and the retained bridge, not architecture mandates.
Descriptions of protocol decisions do not require preserving those decisions.

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
identifies its invocation, source head, section definition and chunk ordinal.
Setup maps peer section definitions to local bindings; the chunk ordinal selects
the page-table entry and identifies the final numerical publication. Sender publication order
need not match declaration order. Chunks from different sends may interleave
within a receive run. The canonical page table preserves logical chunk order;
contiguous consumers use the [indexed placement path](pages-and-functions.md#indexed-receive-runs-and-contiguous-consumers).

## D6. Paired send and receive frame counts match

TN3205 requires matching frame counts and specifies finite queue capacity. It does
not require one request per numerical partial. Mesh uses internally sized chunks
with a 24-byte invocation/source-head/definition/chunk tag. The fixed chunk framing permits preposting independently
of producer order; operand size determines only the number of chunks. Tail
padding and the tag's additional frame remain explicit transport costs, without
a caller capacity query or a requirement to reshape numerical operands.

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

## D10. Reduction and movement have different algebras

The MPI Forum's [collective communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and Rabenseifner's *Optimization of Collective Reduction Operations* (2004) supply
distinct movement and reduction compositions. Mesh binds each verb's index relation;
reductions use existing numerical functions over partial contributions.

## D11. Tensor-parallel boundaries follow the caller's graph

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

## D16. Application-specific output communication stays in the caller

Megatron's vocabulary-parallel example fuses its loss to reduce communicated values.
That is application algebra, not a requirement to add a vocabulary head or model
interpreter to the collective backend.
