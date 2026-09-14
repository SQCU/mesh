# Collective mechanism sources

The [async contract](async-collectives.md) is the sole specification.

- [TN3205] Apple, *TN3205: Low-latency communication with RDMA over Thunderbolt*,
  https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt
- [Pathways] Barham et al., *Pathways: Asynchronous Distributed Dataflow for ML*, MLSys 2022, https://arxiv.org/abs/2203.12533
- [JACCL] Apple, MLX `mlx/distributed/jaccl/lib/jaccl/{mesh_impl.h,rdma.h,rdma.cpp}`, https://github.com/ml-explore/mlx
- [NCCL] NVIDIA, `src/proxy.cc`, https://github.com/NVIDIA/nccl
- [Kahn] G. Kahn, *The Semantics of a Simple Language for Parallel Programming*, IFIP 1974
- [TTDA] Arvind and R. S. Nikhil, *Executing a Program on the MIT Tagged-Token Dataflow Architecture*, IEEE TC 39(3), 1990
- [Thakur] Thakur, Rabenseifner and Gropp, *Optimization of Collective Communication Operations in MPICH*, IJHPCA 2005
- [Megatron] Shoeybi et al., *Megatron-LM*, https://arxiv.org/abs/1909.08053
- [ibv_post_recv] Libibverbs Programmer's Manual, `ibv_post_recv(3)`
- [RDMAmojo] D. Barak, *ibv_post_recv()*, https://www.rdmamojo.com/2013/02/02/ibv_post_recv/ (author's reply in comments)
- [hellas] hellas.ai, *AD/FA57: Apple's RDMA-over-Thunderbolt*, https://blog.hellas.ai/blog/thunderbolt-ibverbs/6-ad-fa57/
- [MPI] MPI Forum, *MPI-4.1*, chapter 12


## D1. The verb is SEND into a posted RECV; there is no remote write

> RDMA over Thunderbolt supports: Send and receive operations only
> — [TN3205]
> RDMA over Thunderbolt only supports 2-sided operations and not hardware-initiated remote writes, thus applications should only register memory as IBV_ACCESS_LOCAL_WRITE.
> — [TN3205]
> `.opcode = IBV_WR_SEND, // Only IBV_WR_SEND is supported`
> — [TN3205]
> UC WRITE looks like it works … but Apple's wire format for UC WRITE is raw user bytes with no BTH/GRH/LRH, no rkey, no raddr. … uc_write_verify confirms the receiver MR is unchanged.
> — [hellas]

## D2. Two-sided becomes one-sided by an always-accepting receive loop

> certain RMA functions might need support for asynchronous communication agents in software (handlers, threads, etc.) in a distributed memory environment.
> — [MPI] §12.1
> The Thunderbolt controller hardware ensures this send work request isn't processed until the receiving Mac has posted a matching receive work request using a credit-based flow control system.
> — [TN3205]
> In order to prevent getting into RNR, one needs to always make sure that there are enough Receive Requests in the Receive Queue.
> — [RDMAmojo]
> If we have ops to progress, no need to block waiting for something to arrive or even wait for the lock to be available. Exit, continue progress, and come back later.
> — [NCCL] `src/proxy.cc`

## D3. Progress is polling a completion queue

> All queue pair operations are non-blocking and the caller must poll an associated completion queue for completion events.
> — [TN3205]

## D4. The receive is posted directly on the consumer's input pages

> Receive buffer can be anywhere within a registered memory region
> — [TN3205]
> When a sender sends a matching frame, the Thunderbolt controller directly copies the data into the address specified in the receive work request.
> — [TN3205]
> Host B allocates B's inputs, transmits the input buffer addresses to host A, and performs most of the preparatory work to launch node B's function. When node A completes, its outputs are sent via the accelerator interconnect directly into node B's input buffers, and then host B starts node B. The latency between one node completing and the next node starting can be made to be little more than the data transfer time.

## D5. Which send lands in which receive: per-queue FIFO order

> The order of the Receive Request consumptions in a Receive Queue is by the order that they were posted to it.
> — [RDMAmojo]
> The order of processing a Work Request is guaranteed per Work Queue according to the order the Work Requests were added to it.
> — [RDMAmojo]
> In others words, processes communicate via first-in first-out (fifo) queues.
> — [Kahn]
> … no two computing stations are allowed to send data on the same channel.
> — [Kahn]

## D6. Every message on a queue is the same number of frames

> Thunderbolt devices require a receiver and sender to post messages which are the same number of frames long. For example, if the sender sends a 16K message and the receiver posts a 4K or 32K receive buffer the receive operation will fail.
> — [TN3205]
> A maximum of 10 unreliable connection (UC) queue pairs … Message sizes of up to 16,773,120 bytes … A maximum of 4095 work requests at a time
> — [TN3205]

## D7. A producer never waits on a transfer

> (ii) enqueueing network sends to remote accelerators for the buffer futures output by function executions
> — [Pathways] §4.4
> PATHWAYS uses a sharded dataflow graph of asynchronous operators that consume and produce futures
> — [Pathways] abstract
> … but nothing can prevent a process from performing a send on a line.
> — [Kahn]

## D8. Presence is the receive completion

> … the Thunderbolt hardware will report completion of the receive to the queue pair's associated completion queue.
> — [TN3205]
> operators may fire as soon as tokens arrive at their inputs
> — [TTDA]
> The process stays blocked on a wait until something is being sent on this line by another process
> — [Kahn]

## D9. A receive's pages are reposted only after their readers finished

> The buffers used by a WR can only be safely reused after WR the request is fully executed and a work completion has been retrieved from the corresponding completion queue (CQ).
> — [ibv_post_recv]

## D10. Collectives stream by chunk

> For long messages and predefined reduction operations, we use Rabenseifner's algorithm for allreduce [19], which does a reduce-scatter followed by an allgather.
> — [Thakur]
> Bandwidth optimal all reduce for large messages: a reduce scatter followed by an all gather.
> — [JACCL] `mesh_impl.h`
> Recv completed. Reduce the peer's contribution into our chunk and, if there is more data to fetch from that peer, post another recv.
> — [JACCL] `mesh_impl.h`, `sum_scatter`

## D11. One all-reduce per tensor-parallel block is a data dependency, not a barrier

> This approach splits both GEMMs in the MLP block across GPUs and requires only a single all-reduce operation in the forward pass (g operator)
> — [Megatron]

## D12. No transport acknowledgement; errors are status, out of band

> Completion of a send operation indicates that the peer has posted sufficient receive requests to receive all sent frames and that the Thunderbolt hardware has sent the buffers to the peer. Completion does not indicate the receiver has successfully received the sent data or that data was not corrupted in flight because Thunderbolt does not perform Acks in hardware.
> — [TN3205]
> The Thunderbolt controller reports failures to send or receive in the work completion (WC) status field.
> — [TN3205]

## D13. Connection metadata is exchanged out of band once

> At this point, applications should use an out-of-band communication mechanism to share the GID and queue pair number (QPN) with a peer. Typically applications use Thunderbolt IP and a TCP socket connection for this purpose, but any reliable mechanism of exchanging this metadata works.
> — [TN3205]

## D14. A connection lives as long as the client program that uses it

> A queue pair holds RDMA Verbs connection state from a host to a particular peer, just as a socket holds connection state for a TCP connection.
> — [TN3205]

## D16. A vocabulary-parallel head is reduced before it is communicated

> We parallelize the input embedding weight matrix EH×v along the vocabulary dimension E = [E1 , E2 ] (column-wise).
> — [Megatron] §3
> However, for this case, the all-gather will communicate b × s × v elements (b is the batch-size and s is the sequence length) which is huge due to vocabulary size being large. To reduce the communication size, we fuse the output of the parallel GEMM [Y1 , Y2 ] with the cross entropy loss which reduces the dimension to b × s.
> — [Megatron] §3
