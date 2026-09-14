# Actual outstanding frame capacity

September 13, 2026. Implements the transport-capacity portion of step 7 in
[pallas-lowering-plan.md](pallas-lowering-plan.md). No shared-memory or transfer
wire ABI change; both endpoint tuples and useful byte lengths are unchanged.

## Published rule and previous limitation

[Apple TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
specifies 4 KiB frame queue units, queried assigned capacities, matching
SEND/RECV frame lengths, and nonblocking posting with completion polling.
Previously mesh divided queried capacity by the maximum allocation block's
frame count and used that message-count cap for both payload and index QPs.
Useful lengths already saved wire bytes, but a short payload still occupied
an entire maximum-block slot in software.

At a queried capacity of 4095 frames and a 64 KiB allocation block, that old
cap is 255 messages. Actual accounting permits 1023 concurrently posted
16 KiB messages, or 4095 one-frame messages, when those messages are ready
and destination storage is available. These are capacity calculations, not
measured throughput results.

## Retained state and source proof

`mesh_queue.frames` is the sum of `mesh_posted.frames` for outstanding
requests in that exact QP direction. `link_post` records a request only after
successful verbs posting. The CQ path releases the stored cost of the
matching request. Error completions still retire their request and preserve
existing error reporting. Teardown destroys QPs before clearing these sums.

`mesh_pending.frames` on the announced-send queue reserves capacity for
already transmitted index tuples whose payload posts have not yet succeeded.
Selection subtracts both posted and reserved costs; successful payload
posting moves the cost between those two owners. A failed index post puts
selected indices back in the ready list without reserving payload capacity
or holding their source. A failed payload post retains its announced tuple,
source hold, cost and FIFO position. No software acknowledgement is added.

Before announcement, ready-list selection passes over a transfer that does
not fit the residual capacity and retains its index for a later pass. It can
therefore use residual capacity for a shorter ready message. After
announcement, both payload posting and receive posting retain hardware
message order. A receive destination already in use still requires its
existing lifetime to finish; this change does not turn live destination
reuse into a legal concurrent write.

At least one frame belongs to every valid transfer. Thus no posted or
announced queue can exceed its 4095-entry storage when its summed frame
cost stays within its assigned capacity. Unposted peer announcements cannot
outrun the sender's reserved-plus-posted payloads: those payloads cannot
complete before matching receives have been posted. Posting those receives
removes their announcement entries.

Each QP direction gets a dedicated CQ. Setup requests at most the lesser of
4095, `max_qp_wr`, and `max_cqe - 1` frames; CQ storage has one extra entry,
following TN3205's example. Assigned QP capacities are queried individually.
The bridge polls each CQ into a reusable 4095-completion array. Each CQ has
at most one completion per outstanding request, bounded by its frame count;
a maximum-block assumption no longer undersizes CQ storage. Setup prints
assigned capacities and actual CQ entry counts. A configured transfer
larger than its assigned direction capacity reports the exact required and
available frames during setup instead of remaining permanently ready but
unpostable.

Index frames retain their existing shared-memory allocation and slot count.
Their frame capacity and slot capacity are checked separately. This avoids
an ABI migration and does not prevent the payload queue from filling: each
index message carries up to 102 payload descriptors, and consumed index
messages release their slots independently of payload completion. A very
large maximum block can still leave too few index slots for peak tiny-message
rate; that is a remaining sizing decision, not a full-capacity claim for
all configured block sizes.

## Validation and limitations

`make -C rdma mesh-flow` succeeds. `cc -fblocks -O2 -Wall -Wextra
-fsyntax-only rdma/mesh-flow.c rdma/mesh-dataflow.c` succeeds without warnings.
No bridge restart or workload was run for this source increment.

Paired operational evidence is still required for queried capacities,
provider CQ availability, heterogeneous useful lengths, sustained tiny
messages, fanout and repeated storage reuse. Dedicated CQs remove aggregate
shared-CQ pressure but change polling from one shared queue to each direction;
this change makes no zero-overhead or measured bandwidth claim. Peak capacity
does not establish peak throughput or eliminate receive-buffer lifetime
constraints. The broader collective algebra and multi-participant routing
portions of step 7 are separate work.
