# Node-addressed transfers (wire 136, layout 155)

A client names a destination node and some bytes, and nothing else. The bridge is the only process that knows that
cables, devices, queue pairs or a per-wire order exist. This replaces the per-cable `struct mesh_stream` contract
(owner word, `wire`, per-cable lists and counts, EBUSY) and the earlier plan in
[node-addressed-crossings.md](node-addressed-crossings.md), whose receives landed in bridge pages.

## 1. What the literature implies here

Read for this document: none of the sources below was re-read; this section is from memory of them, and says so
where a detail matters.

- **Two-sided verbs with pre-posted receives and credits (IB/RoCE practice; MPI eager/rendezvous; MVAPICH; UCX).** A
  receive queue is FIFO: the n-th SEND on a queue pair consumes the n-th posted receive. Practice therefore either
  (a) posts anonymous buffers (eager: a bounce pool, a header naming the message, then a copy or an in-place read),
  sizing them with credits returned by the receiver, or (b) negotiates per message (rendezvous: RTS/CTS, then the
  payload, in MVAPICH and UCX usually as RDMA READ/WRITE). Here there is no WRITE and no immediate data, and the operator
  forbids copies and per-transaction negotiation, so neither (a) nor (b) applies. What remains of that practice is the
  credit idea as a static rule: a receive is posted only when its destination is free, and the transport (TN3205's
  credit on UC queue pairs, design/transport.md section 4) holds an early SEND until it is. No credit message is sent.
- **NCCL Simple/LL/LL128 and its proxy.** NCCL avoids negotiation by giving every (channel, peer) its own FIFO of fixed
  slots in pre-registered buffers, with a head/tail credit exchanged through memory; LL and LL128 fold a flag into the
  payload so the receiver needs no separate completion. The proxy thread is the one owner of the network queues, as
  the bridge is here. Two lessons apply: (1) a static, per-direction FIFO is enough when both ends execute the same
  program, and (2) NCCL requires operations on different communicators to be issued in the same order on every rank
  when they share resources. Here every client toward a node shares the order, so that rule is the precondition
  (section 4). LL-style flags are not available: max_sge is 1 and there is no immediate, so completion is the bridge's
  done word.
- **eRPC (NSDI 2019).** One NIC queue shared by all sessions, per-session credits, multi-packet receive queues and
  copying out of them, with the common case costing nothing extra. It shows that multiplexing many conversations over a
  few hardware queues is cheap when the per-message metadata is small; its receive side still copies, which this
  design avoids by putting the order in place of the header.
- **FaRM, HERD, FaSST.** FaRM uses one-sided WRITE into per-pair ring buffers; HERD uses WRITE for requests and UD SENDs
  for replies; FaSST uses UD SENDs with shared receive queues and a copy, and argues that connection count, not
  bandwidth, is what scales badly. With RC/UC queue pairs limited to 2 a device on macOS 27, that argument is the
  hardware here: queue pairs cannot be given to conversations, so conversations must share them.
- **Statically scheduled collectives (MSCCLang, TACCL, Blink).** The schedule (which chunk on which link at which step)
  is compiled once and executed by every rank identically, with no run-time agreement; Blink packs spanning trees over
  heterogeneous links. That is the model taken here at the bridge's level: the placement of every piece on a cable and
  queue pair is a compiled, pure function both ends evaluate.

## 2. The contract

- **Client side** (mesh-call.h). `mesh_order_open(ctx, node)` returns the order toward a node (its slot and
  generation); it takes nothing. `mesh_order_append(ctx, order, way, transfer)` appends a `struct mesh_crossing` to
  the order's ring for that direction (0: bytes this node sends, 1: bytes it receives). A transfer is an exact span
  table, a gate (a word reaching `base+s+1` for span s of a SEND; reaching `base` for a receive; or several absolute
  gates) and a done word raised to `value` once every byte crossed (UINT64_MAX: lost). Done words named by several
  transfers rise in their order, so a count a client keeps over many transfers is a prefix. Any number of clients use
  one order at once; the ring is shared and taken slot by slot by compare-and-swap.
- **The shared order** (mesh-order.h). The order of a direction is the order its transfers were appended in. The bridge
  cuts every span into pieces of at most one allocation block. Piece n of the k-th transfer goes, in the epoch holding
  n, on cable `(n - first) mod cables` and that cable's queue pair `k mod lanes`, in piece order. Consecutive pieces of a
  transfer cross every cable (its bytes take all the bandwidth); consecutive transfers take a cable's queue pairs in
  turn (a transfer held by its gate holds only the queue pair its pieces are on). Both ends compute it from what both
  declared, so there is no announcement, grant, claim or answer about any transfer.
- **Posting.** Each port child runs one walker per queue pair per direction over the ring, posting the pieces placed on
  its queue pair as soon as their gate holds: a receive as soon as its destination is free, a SEND as soon as its
  source is produced. A walker never posts past an unready piece on its own queue pair; it never waits for anything
  on another.
- **Completion.** Each completion counts a piece in its transfer's slot; the process that completes the last one
  raises the done word (in order among transfers sharing it) and any bridge process retires done transfers, moving
  the ring's anchor (the transfers and pieces before it).
- **Epochs.** The cables of an order change only by the bridges' ORDER exchange, the one message about the order: when
  the cables a bridge holds up toward a node differ from the latest epoch's (a cable came up), it freezes the order
  (nothing posted), takes each port child's frontier (pieces posted each way), and sends ORDER {epochs, cable
  identities, frontiers}. When both ends' ORDERs name the same cables, both append the same epoch, beginning each way
  at the later of the sender's and receiver's frontiers (pieces before it keep the epoch they began in), and thaw. A
  cable of the epoch lost restarts the order (generation + 1: every transfer not done is lost) and loses every cable
  toward that node, so both ends restart; the next agreement begins from piece 0. A client starting or stopping
  changes nothing: its transfers are the next ones of the order.
- **Clients that die.** Their transfers stay in the order and are carried without their gates; their rows stay held
  until their transfers retire (mesh-dataflow.c mesh_pinned). A slot taken and never filled becomes a transfer of no
  generation. A table the region does not hold is moved through the bridge's own block, so the queue pair stays
  aligned with the peer, and that transfer alone is lost.
- **Version.** MESH_VERSION 136; a bridge of another version is not paired with (its HELLO is let be).

## 3. What was deleted

The client stream API (`mesh_stream_take/give/held/view/present`, `mesh_streams_free`, `mesh_endpoints`,
`struct mesh_stream_view`, `struct mesh_entry`, `MESH_STREAM_ENTRIES`, `MESH_STREAM_GIVEN`), the bridge's holdings
(owner words, HOLD_* states, walks per stream, the CPU metadata pin, stream_end and its "stream not live for its
pairing", streams_let_go), the peer service's bridge half (mesh-peer-bridge.c, mesh-peer-plan.h, mesh-peer-cables.h,
mesh-peer-private.h, its Unix socket, OFFER/ACCEPT/ADMITTED/RECEIVED, peer-packets.c), libnccl-mesh's per-cable holds
and EBUSY, and records that named each cable (a record now names nodes). libnccl-mesh keeps one channel per peer
node; its fused crossings are one publish and one await per (peer, frame). The peer service is a client library over
the order. The quiet diagnostic names a node.

## 4. The precondition and what breaks it

The two ends' rings of a direction must hold the same transfers in the same order with spans of the same lengths:
the operator's "both ends already share the program". One client per node pair (a communicator, a peer program) meets
it by construction. Several clients meet it when their program issues their transfers toward the node in the same
order at both nodes, NCCL's rule for communicators that share resources. Two unrelated programs that start at random
times toward the same node do not meet it, and nothing in the hardware can make them: with FIFO receives, no copy and
no per-transaction message, two independent streams cannot share a queue pair without an agreed interleaving. What
happens then is a mismatch of SEND and receive lengths or contents on some queue pair; the remedy is the order's
restart (losing a cable toward the node, or the bridge's restart), after which the clients make their communicators
again. A client that dies with transfers its partner has not declared, or the reverse, likewise leaves the two rings
apart until a restart.

## 5. Testing

- `make -C rdma all` (every warning an error) and `make -C rdma check`.
- tests/order.c: the placement and epoch functions; two in-memory ends with the same declared transfers (unequal
  offsets and scatter), walked in different interleavings with gates opened in a different order and a cable added mid
  order at the two frontiers' boundary, produce the same piece sequence on every queue pair; done words rise as a
  prefix, another generation's transfer is lost, a restart loses what is not done, retirement frees a slot's lap.
- tests/sim-two-nodes.sh: two simulated nodes' regions and the bridge's own walkers over a wire that fills each queue
  pair's receives in post order, holds a SEND until a receive is posted, requires equal lengths and serves queue pairs
  in a jittered order. It runs every tests/peer-groups.c mode and a real two-rank libnccl-mesh communicator (init,
  AllReduce, AllGather and Broadcast of fp32 at 6 KB, 98 KB and 1.5 MB, every value checked, destroy), with up to four
  communicators sharing the order at once.

## 6. What may fall short on hardware

- The parent's ORDER exchange, the freeze and the restart on a cable's loss are not exercised by the simulator (it
  fixes one epoch); they are new code on the bridge's event loop.
- The per-piece bound is one allocation block (16 KiB): a 1.57 MB broadcast is 96 SENDs over 6 queue pairs, as
  libnccl-mesh already cut it. A larger piece would lower per-request cost at the price of balance across cables.
- Every walker reads every transfer of its direction (it skips pieces placed elsewhere), so the bridge's CPU work per
  transfer grows with the cables toward the node; small for 3 cables.
- Completion is published per whole transfer (a relay gated mid-transfer starts at the transfer's end). The order
  rule places a transfer's consecutive pieces on different cables, so its latency is the slowest cable's.
- `blocked` in peer-groups.c needs three queue pairs apart for three groups: it passes with 3 cables (the M5 pair) and
  fails on one cable of two queue pairs, where the third group's piece shares a queue pair with the first held one.
- A send buffer may be written again only once ncclMeshMemReusable says so (a call returns before its SENDs are done);
  the simulator's jitter exposes a program that does otherwise.
