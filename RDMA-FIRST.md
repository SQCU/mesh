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

**Do not build a backpressure protocol.** The mechanism already exists, twice over. The
hardware will not process a send until the receiver has posted a matching receive —
that is pre-granted credit, feed-forward, free. And the pool is finite: if a yeller
cannot drain, pages do not return to the free list; if the free list is empty, no
receives are posted; if no receives are posted, the upstream peer's credits stop
replenishing and it stops sending. Backpressure propagates backwards through the graph
with **zero messages exchanged**. A credit protocol layered on top would add round
trips — 89 KB of wire each — to re-derive a signal the pool already carries.

**Never block; poll and yield.** An agent that finds an empty free list does dispatch
or transform work and checks again. Non-blocking is achievable here because the only
shared state is a queue you inspect, never a permission you await.

**Feedback has exactly two legitimate granularities**: a job completing, or data
completing one orbit of a cycle in the graph. Anything finer is a handshake wearing a
costume. A ring is a good topology precisely because an orbit *is* the synchronisation.

**Telemetry is mandatory, not optional.** Since backpressure is structural rather than
negotiated, the only way anyone knows what the graph is doing is to watch it. Stream
the page-table state space:

- `V` — pages by state: free, posted, filled, dispatched, sending
- `dV/dt` — rates, as deltas against the last sample
- `dV/V` — relative rate, dimensionless, comparable across nodes with different pools
- variance of each, via Welford

That is enough to distinguish a node that is draining steadily from one that is
hysteretic (high variance, mean near zero) from one that cannot drain (sending
monotonically increasing) from one that is crashing out (free → 0, sustained
negative). A few floats per peer per second and the whole graph knows what the whole
graph is doing.

## There was software loss repair here, and it is gone

An earlier version implemented receiver-driven NACKs over the same queue pair, a miss
bitmap, a retransmit window held in the page pool, and a sweep that re-armed requests by
sequence progress rather than by a clock. Several sections of this document used to explain
its tuning in detail.

All of it is deleted, and the reasoning that replaced it is in **There is no repair in the
transport**, below. The short version: ARQ is a property of the RC transport, implemented in
the NIC with packet sequence numbers and a sender-side hardware buffer. Reimplementing it in
userspace over UC, without that buffer, produced a mechanism that invented 7.3 million
phantom gaps in five seconds on a link losing nothing, drove its own retransmit traffic from
them, corrupted the page accounting, and ended in a kernel panic inside the vendor driver.

This section is kept as a marker rather than removed outright, because the tuning it once
described was persuasive and someone will be tempted to rebuild it.

## The invariant core

`mesh-flow.c` and the routing it implements **do not change for a workload.** They are fixed
against the identity function, and that is not a figure of speech: routing *is* the identity
case, so a core that is correct for `id` is correct for every workload.

An earlier version enforced that by compiling the workload into the transport —
`make F=f-yourthing.c` linked a `mesh_f` callback into `mesh-flow`. That is deleted, and the
reason is worth keeping. Every workload was then its own binary, which meant its own
out-of-band socket, its own verbs device, its own queue pairs against a per-device budget of
ten, and its own identity to the host firewall. It also inverted control: the transport called
the application, so the application could not be a thing that already existed, like a game
server. One bridge per node owns the queue pairs because the hardware permits nothing else.

The boundary is now a process boundary. The bridge is one binary that never changes.
Applications are separate processes that attach to its shared region and speak in pages
through three functions, documented in `README.md` under **Using the mesh**. Nothing about a
workload can reach the data plane except the bytes it puts in a page and the node it names.

The loop has five stages and no others:

```
each turn, unconditionally:
  LISSEN   every free page to ibv_post_recv, until the hardware declines
  APPREL   pages the application released return to the pool
  APPSUB   pages the application submitted go to ibv_post_send, until it declines
  POLL     drain completions; a received page is delivered or forwarded where it lies
  CENSUS   sample the page table; gates nothing
```

There is no stage that waits, no stage gated on a computed target, and no queue whose depth
is consulted before acting. What bounds the loop is the hardware refusing work.

## Addressing and routing

The wire header is 16 bytes, 0.39% of a 4096 page, leaving 4080 usable:

```c
struct wire { uint32_t magic, bytes; uint16_t src, dst; uint8_t hops, pad[3]; };
```

`src` and `dst` are node indices — *this page is from node A, for node B*. There is no
authentication and none is wanted: the topology is cables we plugged in ourselves, so this is
a routing tag, not a credential. `hops` bounds a forwarded page. `bytes` is the payload length,
clamped to the page on receipt.

It was 24 bytes, then briefly 28 when `bytes` was widened, and carried `path`, `stream`, `seq`
and a `flags` byte with
`F_FIRST`/`F_LAST`/`F_META`/`F_NACK`. Every one of those served span reassembly, sequence
tracking or the NACK protocol, and all three are gone; nothing read them. A field on the wire
that no receiver reads is not free — it is payload the application does not get, on every page
forever, plus a thing the next reader has to work out is vestigial.

An earlier revision of this section described `F_META` as "the whole of the is-this-for-us
question", distinguishing graph metadata every peer should observe from opaque payload no hop
may interpret. That distinction was never implemented, and the bit it depended on no longer
exists. The question is answered by `dst` alone.

### Routing is a compiled path, not a table

`path` holds the route as **2 bits per hop** — a node has at most three fabric ports,
so an egress port is two bits. Each hop reads the low bits, shifts right, forwards. A
`uint32` therefore carries 16 hops:

| field | hops | nodes reachable at degree 3 |
|---|---|---|
| `uint32 path` | 16 | 43,046,721 |
| `uint64 path` | 32 | 1.85 × 10^15 |

Per-hop cost is a mask, a shift and a store, inside the bytes the header already
occupies — no extra bandwidth, no lookup, no memory.

A next-hop table would be O(N) memory per node *and* a fleet-wide agreement problem:
2048 entries per node at 2048 nodes, a million entries and distributed consensus at a
million. The path is computed once per stream by whatever sets the stream up, and
after that every hop is a shift. The mesh can be arbitrarily large and the data plane
never learns how large.

Termination is `dst == node_idx`, with a hop-count ceiling so a corrupted `path` dies
rather than circulating. Measured on two nodes: a page addressed to the peer is
delivered and terminates there, at **69.45 Gbit/s one-way — 98% of the 70.9 Gbit/s
ceiling**.

## The shape

Agents that never wait, over one page table of refcounted pages:

```
free list ─▶ LISSEN ─▶ [ring] ─▶ DISPATCH ─▶ [per-peer queues] ─▶ YELLER ─▶ free list
               ▲                                                              │
               └────────────────── refcount hits 0 ───────────────────────────┘
```

Dispatch is always present — forwarding unchanged is a decision that must be made per
page, not the absence of one — and its cheapest case is a stream→destination lookup
that touches no payload. Pages are refcounted because one arrival may go to several
peers. Nothing in the data path branches on content; the transform is the one place
bytes are read, deliberately, paying the 2N.

## What the transport is not

RDMA moves memory. It is pure value with no meaning: no framing, no parity, no
acknowledgement, no records. A message may span thousands of receive completions, so
there is no offset at which a header reliably lives, and a header in the first landing
page says nothing about the ten-thousandth. Deserialization happens once, at the
endpoint, over the ordered page list, in place.

Which is why a Quake match and a pipelined tensor algorithm are the same program: a
stream of opaque bytes with a fixed destination, differing only in whether `f` at a
hop is the identity.


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

An earlier version of this bridge implemented sequence tracking, a miss bitmap, NACK
generation, a retransmit window and a sweep — a software imitation of hardware ARQ, at the
wrong layer, without the hardware buffer that makes it work. It generated 7.3 million phantom
gaps in five seconds on a link losing nothing, drove its own retransmit traffic from them,
corrupted the page accounting, and ended in a kernel panic inside the vendor driver. All of
it is deleted. The controller lost 43% of its code and gained the ability to run bidirectional
traffic without falling over.

What remains is the whole job: post receives into free pages, drain completions, hand each
arrived page to the application or forward it, send what the application submits, and take
back what it releases.


## Registration is arithmetic, not a lookup

The region is one contiguous span cut into equal chunks, one memory region per chunk. Because
the chunks are equal, an address maps to its region by division: `(addr - base) / chunk`.
There is no table to search, no address that can miss, and no lookup that can fail, so no
call site needs a branch for the case where it does.

This is worth stating because the obvious reference implementations do it differently, and
for a good reason that does not apply here. Production transfer engines register arbitrary
user buffers at arbitrary addresses, so they keep an ordered map and find the containing
region by `upper_bound` and a bounds check. Copying that shape into this bridge imported a
sorted table, a binary search, a null return and a validity test at every use, to answer a
question that division answers exactly. The chunk size is a constant, not a device query, for
the reasons under `max_mr_size` above.

Setup may fail and says so. The data path may not, and so nothing in the data path asks.

## One way to give a page back

A page belongs to exactly one holder. It is on the free list, with the hardware as a posted
receive or a send in flight, with the application because it was delivered, or in the arena,
which the application owns outright and the bridge never reclaims.

There is a single operation that returns a page to the pool and it takes no decisions. A page
the bridge does not own is never passed to it: an arena page is recognised by its index, and a
delivered page is not tracked at all until the application returns it through the release
ring. An earlier version had a `held` flag that meant two different things — delivered, and
held for retransmit — which is why freeing a page needed three branches and why 55,335 double
frees were possible in a five second run.

A page's address is arithmetic and is not stored. Its state is not stored either, because the
completion says what the page was doing.

The pool invariant is `free + posted + pool-in-flight + with-app`, and it must equal the pool
size exactly. Pool-in-flight is not the same as the send count: the send count includes arena
pages the application owns, which are not pool pages. Conflating them produced a figure that
only balanced when nothing happened to be in flight at the moment of sampling, and it was
quoted as proof of correctness four times before a viewer plotted it and the error showed.

## The census is not a queue depth

What is published once a second is a census of the page table and its variance. It is
deliberately not the depth of a queue at an instant.

A depth read at an instant is a sample of something the program should never be waiting to
look at. Receives go to lissen and sends go to yell the moment they exist; there is no point
in the loop where it is correct to stop and ask how full something is, and an implementation
that gates on a computed target rather than on the hardware refusing the work has invented a
throttle nobody asked for. Both gates existed here and are removed: LISSEN posts every free
page until `ibv_post_recv` declines, and the send path drains what the application submitted
until `ibv_post_send` declines.

The variance is per interval, not per run. A lifetime variance is dominated by the step from
idle to running and reports a number that grows for reasons unrelated to what the pool is
doing now. Each published sample describes its own second, so an evenly breathing pool reads
near zero and a starved or flooded one does not. The estimator is Welford: one pass, constant
memory, and it does not lose the variance to cancellation once the mean is large, which it is
here because these count pages and there are millions of them.

Sampling it costs something, so it is sampled, not computed on every turn of the loop. An
earlier version ran eight Welford updates and a clock read per iteration of the data path.

## A dead application does not strand its pages

An application that exits while pages sit in its delivery ring, or that took pages and never
released them, used to strand them permanently: the bridge forgets a delivered page by design,
so nothing knew they existed and the pool shrank by up to the delivery ring's capacity — 4096
pages — per death.

The bridge keeps a bitmap of delivered pages, one bit each, `npages/8` bytes. On the census
tick it checks whether the attached client still exists, and if `kill(pid, 0)` reports `ESRCH`
it returns every marked page to the pool, resets the three ring cursors so no stale descriptor
survives, and clears the client. The check is once a second and outside the data path.

Measured: an application exiting with pages outstanding leaves `held=4055`, and one tick later
`held=0` with the census closing exactly at 244140/244140.

This is why the census is published. A conservation figure nobody reads is decoration; this
one named a real leak and then confirmed the repair.

## Sleeps, and where they crept in

A sleep in a data path is a latency injection with a number attached, and a sleep in a poll
loop is a wait for a condition dressed up as a wait for a clock. Neither is a schedule; both
are a missing primitive being papered over. This is an audit of every one in code this repo
owns.

**Deleted.**

| where | what it cost |
|---|---|
| `rdma/mesh-flow.c`, `usleep(50)` when a poll found no completions | 50 us injected into the data plane on a link whose round trip is about 10 us. Five times the link latency, added by us, to avoid spinning. |
| `rdma/mesh-client.c`, `usleep(20000)` in a 500-iteration attach retry | `mesh_open` blocked for up to ten seconds waiting for a bridge that might never start, and reported nothing while doing it. It now attaches or returns NULL, and the caller decides. |
| `bin/mesh-bridge.sh`, `sleep 0.5` then `sleep 0.01` in the stop poll | Teardown measures 0.045 s. Half-second granularity reported it after 0.5 s, an eleven-fold penalty for nothing. |
| `bin/mesh-bridge.sh`, `sleep 1` in the start poll | One-second granularity to observe a process that appears in about 45 ms. |

The replacement in the shell is a bounded spin on `kill -0`, which is a shell builtin and forks
nothing. `stop` now returns in 0.042 s.

**Why the data-path one existed, since the reason is not embarrassing and the fix is not
obvious.** The loop has two event sources: the completion queue, and the application's
submission ring. Verbs offers a real blocking wait for the first — `ibv_create_comp_channel`,
`ibv_req_notify_cq`, `ibv_get_cq_event` all exist on this platform. The second is shared
memory and has no descriptor to wait on, so no primitive waits on both. Blocking on the
completion queue alone would stall an application's first submission indefinitely. The correct
resolution for a kernel-bypass data plane is to spin, which is what such transports do by
design and why they are given a core. Sleeping was a way of pretending the choice did not
have to be made, and it bought a latency floor in exchange.

**Still present, with a verdict.**

| where | verdict |
|---|---|
| `bin/mesh-peers.sh` | Partly fixed. The browse now ends when the reply stream goes quiet instead of after a fixed three seconds, using a 50 ms poll period. The per-node `dns-sd -t` resolve and the `nc -G` info probe still run to a deadline and dominate the wall time, so the script returns no faster than before. Saying it is fixed would be false. |
| `bin/mesh-rdma-init.sh`, `install.sh` (three), `install-user.sh` (two) | Waiting on launchd to settle. Each should poll the condition it actually wants. |
| `xonotic/bridge/solver/mesh_attach.h`, `xonotic/bridge/test/meshtest.c`, `xonotic/bridge/test/engine.sh`, `xonotic/ipcbench/bench.c` | Same habit, game-side, inherited from the build workflow and not yet cleaned. |
| `viz/serve.py`, `time.sleep(PERIOD)` | Defensible. This is a poll period — a scheduling decision about how often to sample — not a stall standing in for a wait. |
| `bench/orth.py`, `time.sleep(...)` | Defensible. Thermal duty cycling in a benchmark, deliberate. |
| `xonotic/darkplaces-work/**` | Vendored upstream. Not ours to clean. |

`pmset sleep 0` and the `displaysleep`/`disksleep` keys in `install.sh` and `hmi-epilogue.sh`
are power policy, not delays, and are unrelated.
