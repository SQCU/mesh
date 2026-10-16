# RDMA-first programming

How to get from *"RDMA is a fast socket"* to writing for what this hardware actually
is. Every number below was measured on `Mac16,11` and `Mac17,6` over one TB5 cable,
not taken from a datasheet.

## The ledger

| resource | measured | what it means |
|---|---|---|
| link | 8.9 GB/s each way (70.9 Gbit/s) | per port; three ports per mini |
| round trip | ~10 µs | **89 KB of wire time**, every time you ask a question |
| posted window | 4095 frames × 4 KB = 16.8 MB | **1.88 ms** in flight per QP — 188 round trips of slack |
| registerable | 100 MR/PD × 1 GiB = 100 GiB per domain | `max_mr_size` advertises 16.4 MB and is not the real limit — see below |
| memory bandwidth | 238 GB/s mini · 362 GB/s MacBook | shared with the workload; a copy costs 2N of it |
| scatter-gather | `max_sge: 1` | a record cannot be split across SGEs |
| queue pairs | `max_qp: 11` (10 usable), `max_srq: 0` | no shared receive queue |
| hardware threads | 12 mini (8P+4E) · 18 MacBook (6P+12E) | no SMT; a paired port's progress holds two, parked when idle |

## Expensive

**Latency.** One round trip costs 89 KB of unsent data. A handshake per 64 KB page
spends more time waiting than sending: the link runs under half rate and latency
doubles. Anything with a response per unit of work is finished before it starts.

**Duplication.** Memory bandwidth is the same budget the workload wants. Per record
of N bytes: a zero-copy relay is N writes / N reads. Add one receive-side copy and it
is 2N/2N — at three ports saturated that is 67% of the mini's entire memory system
spent moving bytes that were already in the right place.

**Asking.** Not just latency — a question implies a synchronous moment, and synchronous
moments serialize agents that had no reason to be coupled.

## Cheap

**Pages and indices.** A page is the node's (`getpagesize()`, 16 KiB), and its region entry,
16 bytes (`rdma/mesh.h` `struct mesh_page_entry`: its mapping and address), is 0.1% of it. A lease claims rows and pages
by compare-exchange on bitmap words (`rdma/mesh-call.h`), and an address is index arithmetic:
tens of nanoseconds, no data movement.

**Arithmetic on the side.** Welford's algorithm is ~5 flops per sample. At full page
turnover (the link's 70.9 Gbit/s in 16 KiB pages, about 540k pages/s/port) that is under three
megaflops per second on a machine that does billions. Running statistics are free.

**Telemetry.** This is the one people do not believe:

| rate | payload | cost |
|---|---|---|
| 1 Hz | 0.1 KB/s | 0.000001% of one link |
| 100 Hz | 12.5 KB/s | 0.000144% |
| 1 kHz | 125 KB/s | 0.001438% |

A kilohertz of node health costs one part in seventy thousand of a single link. There
is no budget argument against full observability. This is a property of the era, not
of cleverness: nobody writing for a microcontroller ever had this ratio available.

**Sparse in-channel agreement.** Control can time-share the copper. The cost is a ratio, not a
principle: agreement amortized per *program* is invisible (10–20 µs against a 112 ms gigabyte),
agreement per *page* is over 200% overhead. So a port's two bridges agree once per program over
the port's one TCP control stream (`rdma/mesh-flow.c`, design/bridge-and-ipc.md#protocol), READY,
PAIRS and POSTED as it starts and END or FAIL as it ends, and the data path never inspects bytes.

## The queue does not enforce its own depth

TN3205 says two things that only make sense together:

> "Queues are sized in units of 4 KB frames… queue depths set the maximum message
> size possible."

> "The system may adjust queue depths based on hardware capabilities so be sure to
> check final values using `ibv_query_qp`."

Queue depth is a **frame budget**, and the granted value is the one that counts. On
this hardware a request of 4095 is granted 4095, and smaller requests are rounded
*up* (256 → 1023). So query it; do not assume it.

The part the documentation does not say, measured: **`ibv_post_recv` never fails.**

```
pagesz  frames   receives accepted   frames used
4096    1        8193                8193
8192    2        8193                16386
65536   16       8193                131088     ← 32x the granted 4095
```

Posting past the budget succeeds silently and corrupts later. The frame budget is a
contract the caller must honour; the hardware will not refuse, it will misbehave.
That is the mechanism behind the `local length error` storms at depth.

So in-flight depth is not a tuning knob to be maximised. **Total posted frames must
stay within the granted depth, enforced by us.** Obeying it is also faster: at 4 KB
pages, dropping from a guessed 120 in flight to the computed budget of 60 took a hop
from 31.91 to **45.81 Gbit/s each way, zero errors, zero gaps**. Trying to keep more
outstanding than the queue allows does not pretend the link is faster; it makes it
slower.

## The frame constraint (measured)

TN3205 says sender and receiver must post the same number of frames. It does not say
that **multi-frame messages are only safe at shallow pipeline depth**, which is what
the hardware actually does:

| page | frames | in flight | result |
|---|---|---|---|
| 64 KB | 16 | 1 | clean, 7.65 Gbit/s |
| 64 KB | 16 | 2 | clean, 15.13 Gbit/s |
| 64 KB | 16 | 8 | no errors, **stalls**, 0 Gbit/s |
| 64 KB | 16 | 120 | **113k `local length error`**, receiver drops everything |
| 8 KB | 2 | 120 | 1498 errors |
| **4 KB** | **1** | **60 (granted budget)** | **0 errors, 0 gaps, 45.8 Gbit/s each way** |

`ibv_uc_pingpong` reaches 47 Gbit/s at 16 frames because it keeps exactly one message
in flight. That is not a protocol we can copy: depth ≤ 2 is a handshake in all but
name, and each round trip is 89 KB of wire.

The bridge does not follow this table. A record is one block of the node's page (`getpagesize()`,
16 KiB: four frames), and a queue pair takes another only while its posted frames,
`(length + 4095) / 4096` a record, stay within its queue's granted capacity (`rdma/mesh-flow.c`):
1023 four-frame records at 4095. That figure is calculated (design/actual-frame-capacity-2026-09-13.md),
never measured, and the 64 KB rows above stalled at depth 8 with 128 frames posted, well inside
the budget; until a run at 16 KiB and depth settles it, the measurement and the bridge's rule
disagree. A link's throughput is a program's queue pairs (its highest queue toward the node times
its slots, within `max_qp`; design/bridge-and-ipc.md#flows) over the one arena window (below).

## What follows

**Do not build a backpressure protocol.** The mechanism already exists. The hardware will not
process a send until the receiver has posted a matching receive — that is pre-granted credit,
feed-forward, free. A receiver that posts no receive stops its sender, and a sender whose queue
pair holds no more frames stops posting: backpressure propagates backwards through the graph
with **zero messages exchanged**. A credit protocol layered on top would add round trips — 89 KB
of wire each — to re-derive a signal the posted receives already carry.

**Never block; poll.** A progress thread that finds no completion and no released send polls
again. Non-blocking is achievable here because the only shared state is a word or a queue you
inspect, never a permission you await.

**Feedback has exactly two legitimate granularities**: a job completing, or data
completing one orbit of a cycle in the graph. Anything finer is a handshake wearing a
costume. A ring is a good topology precisely because an orbit *is* the synchronisation.

**Telemetry is mandatory, not optional.** Since backpressure is structural rather than
negotiated, the only way anyone knows what the graph is doing is to watch it. The bridge's
region carries each port's peer, phase, last code, bandwidth and pairings, and each client's
status toward each destination (`rdma/mesh.h`).

## The invariant core

`mesh-flow.c` and the routing it implements **do not change for a workload.** They are fixed
against the identity function, and that is not a figure of speech: routing *is* the identity
case, so a core that is correct for `id` is correct for every workload.

No workload is compiled into the transport. Were it, every workload would be its own binary,
with its own out-of-band socket, its own verbs device, its own queue pairs against a per-device
budget of ten, and its own identity to the host firewall; and the transport would call the
application, so the application could not be a thing that already existed, like a game server.
One bridge per node owns the queue pairs because the hardware permits nothing else.

The boundary is a process boundary. The bridge is the node's one resident binary
(design/bridge-and-ipc.md). Applications are separate processes, any number at once and none
aware of another, that attach to its shared region, lease pages in it, and ask the bridge to
carry them to member nodes (`rdma/mesh-call.h`). Nothing about a workload can reach the data
plane except the bytes it puts in its pages and the node it names.

A port's data path is its two progress threads (design/bridge-and-ipc.md#progress), serving every
program on the port: one posts each send its producer released while its queue pair (or its lane)
holds the frames, the other drains receive completions, publishes each landed record and posts the
ring's next receives (or a lane header's records). Neither waits on anything but a word or a
completion queue it polls while anything moves, and after 100 ms of nothing each parks on its
port's bell, which a program's group rings for the ports it rides as it starts.

## Routing is a compiled path, not a table

RDMA never routes: a queue pair joins two ports over one cable. A program names only its member
node (`rdma/mesh-call.h` `mesh_transfer_bind`). Where the two are not linked, the node's bridge
computes the route once, as the program starts, from its own map (`rdma/mesh-plan.h`
`mesh_trees_path` in the destination's heaviest tree, `mesh_trees_shortest` where none are packed;
EHOSTUNREACH where there is no path), and sends READY with it over the first link; each bridge on
the path relays the program and sends READY on with the rest of the route to the next node it
names, a member of the execution or not (design/bridge-and-ipc.md#protocol, #flows).

A next-hop table would be O(N) memory per node *and* a fleet-wide agreement problem: 2048
entries per node at 2048 nodes, a million entries and distributed consensus at a million. The
path is computed once per program, by the bridge that starts it, and after that every hop reads
the next name. The mesh can be arbitrarily large and the data plane never learns how large.

## What the transport is not

RDMA moves memory. It is pure value with no meaning: no framing, no parity, no
acknowledgement, no records. A message may span thousands of receive completions, so
there is no offset at which a header reliably lives, and a header in the first landing
page says nothing about the ten-thousandth. Deserialization happens once, at the
endpoint, over the ordered page list, in place.

Which is why a Quake match and a pipelined tensor algorithm are the same program: a stream of
opaque bytes with a fixed destination.


## There is no repair in the transport

RDMA reliability is a property of the hardware transport, not of software above it. On a
Reliable Connection the RNIC detects corruption with the invariant and variant CRCs carried
in every frame and drops what fails, and it recovers loss with an ARQ pattern driven by
packet sequence numbers, retransmitting from the sender's hardware buffer without the host
CPU. There is no forward error correction and nothing for an application to implement.

This device offers **UC only**; RC is rejected with errno 102. So the CRCs still mean a
corrupt frame is dropped rather than delivered, and the absence of ARQ means a dropped frame
is simply gone. The transport delivers what arrived. Anything missing is the application's to
notice and send again, because the application is the only layer that knows what it asked
for.

A software imitation of hardware ARQ, at the wrong layer and without the hardware buffer that
makes it work (sequence tracking, a miss bitmap, NACK generation, a retransmit window and a
sweep), generated 7.3 million phantom gaps in five seconds on a link losing nothing, drove its
own retransmit traffic from them, corrupted the page accounting, and ended in a kernel panic
inside the vendor driver. Nothing of the kind is built.

What remains is the whole job: post each program's receives, drain completions and publish what
landed, post the sends its producers release, and when a link or a peer's program is lost, fail
the program with an error naming the node, which its caller may retry at once
(design/recovery.md).


## Registration is arithmetic, not a lookup

The arena is one contiguous window registered in equal 1 GiB extents, one memory region per
extent, each registered on a port's device the first time a program names a page of it. Because
the extents are equal, an address maps to its region by division: `(addr - base) / extent`.
There is no table to search, no address that can miss, and no lookup that can fail, so no
call site needs a branch for the case where it does.

Production transfer engines register arbitrary user buffers at arbitrary addresses, so they keep
an ordered map and search it (`upper_bound` and a bounds check). Here the extent is a constant,
the registration the provider takes, not a device query (`max_mr_size` is not the real limit, the
ledger), and division answers exactly.

Setup may fail and says so. The data path may not, and so nothing in the data path asks.

## Sleeps, and where they crept in

A sleep in a data path is a latency injection with a number attached, and a sleep in a poll
loop is a wait for a condition dressed up as a wait for a clock. Neither is a schedule; both
are a missing primitive being papered over.

**The bridge's data path spins.** A `usleep(50)` when a poll finds no completions injects 50 µs into a
link whose round trip is about 10 µs: five times the link latency, added to avoid spinning. A
program's data path has two event sources: its completion queues, and the ready words of its
sends, which its producers release. Verbs offers a real blocking wait for the first —
`ibv_create_comp_channel`, `ibv_req_notify_cq`, `ibv_get_cq_event` all exist on this platform.
The second is shared memory and has no descriptor to wait on, so no primitive waits on both.
Blocking on the completion queue alone would stall a producer's first release indefinitely. The
correct resolution for a kernel-bypass data plane is to spin, which is what such transports do
by design and why they are given a core. Sleeping pretends the choice does not have to be made,
and buys a latency floor in exchange. A wait on a shared word has one form, its address wait
(`rdma/mesh.h` `mesh_notify`; the bridge's `os_sync_wait_on_address` on `control`). The bridge
spins two threads a port while anything moves, never a program's own (at six nodes a relay hub
held 50 to 70 spinning a session's life at MESH_VERSION 112), and parks them on their port's
`bell` after 100 ms of nothing: the group that releases the next ready word rings the bells of
the ports its program rides as it starts (mesh-call.h `mesh_transfers_wake`), and a relay's
landing that releases the next hop's cell rings that port's when its send thread is parked, so
the descriptor the ready words lack is the group's start or the landing, and a program that does
not ring is served within the park's 1 ms look. A port with nothing of its own to look for (no
flow listed, no lane work in flight) looks every 3 s, so a node of paired links running no
program sleeps.

**Still present.**

| where | state |
|---|---|
| `rdma/mesh-call.c` `mesh_transfers_start` | Sleeps 1 ms a turn until the bridge stores the program's `prepared` word, probing the bridge with `kill(pid, 0)` each turn. The bridge stores the word without waking it, so the client cannot wait on it; were it woken, the start would wait on the word to its deadline. |
| `rdma/nccl-mesh.c` progress thread | `usleep(20)` on each pass of a routing session that moved nothing: twice the link's round trip. |
| `rdma/nccl-mesh.c` `drain` | `usleep(50)` on each pass while the session's GPU programs finish. |
| `rdma/nccl-mesh.c` bootstrap connect | `usleep(100000)` between attempts to reach the root, to its deadline. |
| `rdma/nccl-mesh.c` `session_close` | Sleeps `LINGER_S` (1 s) before releasing the rings, on exit and before a session for other peers opens. |
| `bin/mesh-peers.sh` | Partly fixed. The browse now ends when the reply stream goes quiet instead of after a fixed three seconds, using a 50 ms poll period. The per-node `dns-sd -t` resolve and the `nc -G` info probe still run to a deadline and dominate the wall time, so the script returns no faster than before. Saying it is fixed would be false. |
| `bin/mesh-rdma-init.sh`, `install.sh` (two), `install-user.sh` (two) | Waiting on launchd to settle. Each should poll the condition it actually wants. |
| `xonotic/ipcbench/bench.c` | Same habit, game-side, inherited from the build workflow and not yet cleaned. |
