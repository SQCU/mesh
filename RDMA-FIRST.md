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
| hardware threads | 12 mini (8P+4E) · 18 MacBook (6P+12E) | no SMT; this is the agent budget |

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

**Pages and indices.** A 16-byte descriptor per 64 KB page is 0.024% overhead. Page
assignment, free lists, refcounts: index arithmetic, tens of nanoseconds, no data
movement.

**Arithmetic on the side.** Welford's algorithm is ~5 flops per sample. At full page
turnover (136k pages/s/port) that is under a megaflop per second on a machine that
does billions. Running statistics are free.

**Telemetry.** This is the one people do not believe:

| rate | payload | cost |
|---|---|---|
| 1 Hz | 0.1 KB/s | 0.000001% of one link |
| 100 Hz | 12.5 KB/s | 0.000144% |
| 1 kHz | 125 KB/s | 0.001438% |

A kilohertz of node health costs one part in seventy thousand of a single link. There
is no budget argument against full observability. This is a property of the era, not
of cleverness: nobody writing for a microcontroller ever had this ratio available.

**Sparse in-channel agreement.** Control can time-share the copper. The cost is a
ratio, not a principle: agreement amortized per *stream* is invisible (10–20 µs
against a 112 ms gigabyte), agreement per *page* is over 200% overhead. Amortize per
stream and in-channel is fine — a separate control QP keeps stream identity as the
discriminator, so the data path still never inspects bytes.

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

## The frame constraint (measured, and it forces the page size)

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

So **the page size is 4096 — one frame — and throughput scales by adding queue pairs,
not by enlarging pages.** With `max_qp: 11` (10 usable) and 12–18 hardware threads,
the scaling axis is parallel QPs on parallel threads, each owning its own page pool
shared-nothing.

This also retires an earlier misdiagnosis: a "2 frames hangs, 1 frame works" symptom
was blamed on a flapping link. It was real, and it was this.

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
status toward each destination (`rdma/mesh.h`); a program that asks for it has its trace, every
send and landing with its times (design/prepared-machine.md#M27).

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

A program's data path is its own two progress threads (design/bridge-and-ipc.md#flows): one
posts each send its producer released while its queue pair holds the frames, the other drains
receive completions, publishes each landed record and posts the ring's next receives. Neither
waits on anything but a word or a completion queue it polls, and a link with no program running
spins nothing.

## Routing is a compiled path, not a table

RDMA never routes: a queue pair joins two ports over one cable. A program bound for a node its
own node is not linked to carries its route, the path of nodes from the one to the other,
computed once from the current map by whatever sets the program up (`rdma/mesh-plan.h`
`mesh_trees_path`, in the destination's heaviest tree of trees made ahead of time; the map's
shortest-path trees where none are given). Its node's bridge sends it over the path's first
link, and each bridge on the path forwards it to the next node the path names, a member of the
execution or not; it ends at the path's last node.

A next-hop table would be O(N) memory per node *and* a fleet-wide agreement problem: 2048
entries per node at 2048 nodes, a million entries and distributed consensus at a million. The
path is computed once per program by whatever sets it up, and after that every hop reads the
next name. The mesh can be arbitrarily large and the data plane never learns how large.

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

This is worth stating because the obvious reference implementations do it differently, and
for a good reason that does not apply here. Production transfer engines register arbitrary
user buffers at arbitrary addresses, so they keep an ordered map and find the containing
region by `upper_bound` and a bounds check. Copying that shape into this bridge imported a
sorted table, a binary search, a null return and a validity test at every use, to answer a
question that division answers exactly. The extent is a constant, the registration the provider
takes, not a device query: `max_mr_size` is not the real limit (the ledger).

Setup may fail and says so. The data path may not, and so nothing in the data path asks.

## Sleeps, and where they crept in

A sleep in a data path is a latency injection with a number attached, and a sleep in a poll
loop is a wait for a condition dressed up as a wait for a clock. Neither is a schedule; both
are a missing primitive being papered over.

**The data path spins.** A `usleep(50)` when a poll finds no completions injects 50 µs into a
link whose round trip is about 10 µs: five times the link latency, added to avoid spinning. A
program's data path has two event sources: its completion queues, and the ready words of its
sends, which its producers release. Verbs offers a real blocking wait for the first —
`ibv_create_comp_channel`, `ibv_req_notify_cq`, `ibv_get_cq_event` all exist on this platform.
The second is shared memory and has no descriptor to wait on, so no primitive waits on both.
Blocking on the completion queue alone would stall a producer's first release indefinitely. The
correct resolution for a kernel-bypass data plane is to spin, which is what such transports do
by design and why they are given a core. Sleeping pretends the choice does not have to be made,
and buys a latency floor in exchange.

**Still present, with direct measurements.**

| where | measured state |
|---|---|
| `bin/mesh-peers.sh` | Partly fixed. The browse now ends when the reply stream goes quiet instead of after a fixed three seconds, using a 50 ms poll period. The per-node `dns-sd -t` resolve and the `nc -G` info probe still run to a deadline and dominate the wall time, so the script returns no faster than before. Saying it is fixed would be false. |
| `bin/mesh-rdma-init.sh`, `install.sh` (two), `install-user.sh` (two) | Waiting on launchd to settle. Each should poll the condition it actually wants. |
| `xonotic/ipcbench/bench.c` | Same habit, game-side, inherited from the build workflow and not yet cleaned. |
| `viz/serve.py`, `time.sleep(PERIOD)` | Defensible. This is a poll period — a scheduling decision about how often to sample — not a stall standing in for a wait. |
| `bench/orth.py`, `time.sleep(...)` | Defensible. Thermal duty cycling in a benchmark, deliberate. |
| `xonotic/darkplaces-work/**` | Vendored upstream. Not ours to clean. |

`pmset sleep 0` and the `displaysleep`/`disksleep` keys in `install.sh` are uniform
fleet power policy, not delays, and are unrelated.
