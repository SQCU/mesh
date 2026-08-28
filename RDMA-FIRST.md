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

## Software loss repair on UC (measured)

RC and UD are refused by this hardware (errno 102) and `max_qp_rd_atom` is 0, so there is
no hardware retransmit. Repair is a receiver-driven NACK over the same QP, and the
retransmit window **is** the page pool — nothing is copied, a sent page is simply held in
`txring` until ring position evicts it.

Three things were wrong, and they compounded:

**The pool was clamped to the frame budget.** `frames*npages > 4095 -> npages = 4095/frames`
conflated two unrelated limits. The granted depth bounds *posted* frames; it says nothing
about how many pages a process may own. `max_mr_size` advertises 16.4 MB, but that figure is
neither enforced nor usable: honouring it needs 1416 regions for a 23 GB pool against a quota
of 100 per protection domain. Measured under real traffic with every byte checked, single
regions of 0.258, 2.749 and 3.092 GB deliver `corrupt=0`, while 5.154 and 8.246 GB corrupt
every page. **The real cliff is 2^32, not the advertised value**, and past it registration
succeeds and silently returns wrong data. The bridge therefore chunks at a fixed 1 GiB, a
factor of four inside the observed limit, which needs 23 regions for a 23 GB pool. This is
the same failure recorded as kvcache-ai/Mooncake#2017. `max_mr_size` bounds one MR, not the
pool —
`max_mr` is 100. With the clamp, a 4096-byte page pool could never exceed 4000 pages, so the
retransmit window was 2992 pages no matter what the operator asked for. The pool is now
registered as MRs of 1 GiB; `struct page` already carried a per-page `lkey`.
`rx_target` and `tx_budget` are taken from a quarter of the *granted* frame budget, so posted
frames still honour TN3205 while the pool does not.

**Repair decremented an in-flight counter it never incremented.** The NACK path did
`pg[tp].refs++` without `sending++`, but the completion path does `sending--` when refs hits
zero. Every retransmit therefore drove `sending` one lower, permanently. Measured: `sending`
reached **-2326** inside two seconds, `while(sending<cap)` stopped bounding anything, and the
source ran with the entire window in flight (out_max 2992 against a `tx_budget` of 1000).
Fixing it cut observed page loss about threefold on its own.

**The window was consumed by latency, so retransmits were overwritten before the NIC read
them.** Repair latency is not a duration, it is a distance in sequence numbers, and it is
roughly *twice* the send-queue depth: the NACK reaches the source when the source is already
`tx_budget` pages ahead, and the retransmit then queues behind another `tx_budget` pages
before the NIC DMA-reads the buffer. Measured at `tx_budget` 1023: NACK lag 1411 mean / 1784
max, repair completing at 1865 mean. Against a 2992-page window that leaves under 200 pages
of margin, and a page evicted at that boundary is rewritten under a work request the NIC has
not yet serviced. The symptom was retransmits failing to arrive at **40-46%** while original
pages were lost at 0.05% — confirmed by marking retransmits and counting them at the far end
(5976 of 11134 arrived). It is not wire loss and not burstiness: pacing repair under the same
`sending<cap` gate as fresh data changed nothing, and a probe copy of a page that had already
been *delivered* vanished at the same rate.

So the rule is `txwin >> tx_budget`, not `txwin > 0`. Repair also became a first-class stage
of the send loop — NACK entries land in a retransmit queue drained ahead of fresh pages under
the same capacity gate — rather than an unbounded blast from inside the receive path.

### Re-arm is measured in sequence progress, never in seconds

A request can be lost, and so can its answer. Without re-arm, one lost retransmit is a
permanently lost page — which is exactly the 23-36% residue the earlier mechanism could not
explain. Two previous attempts re-armed on a clock (1 ms, then a 1024-page interval) and both
failed the same way: 99.996% of named sequences were already evicted.

The data plane already carries a monotonic quantity — `expected`, the receiver's sequence
frontier. Everything is expressed against it:

| quantity | value | meaning |
|---|---|---|
| `rearm_gap` | `tx_budget + 512` | re-request a still-missing seq once the frontier has advanced this far past its last request |
| `horizon` | `txwin - 4096`, floored at `txwin/2` | do not name a seq older than this; the sender has evicted it |
| `retire_at` | `horizon + 4096` | give up: clear the bit, count it `gone` |

`horizon` and `retire_at` must be separate. Retiring at the request horizon throws away
sequences whose retransmit is still in flight — measured, that alone dropped recovery to 14%.

The sweep only runs when `missing` is non-zero and only scans the live window, eight bitmap
words per iteration from a rolling cursor, so a healthy stream pays nothing. The gate is the
same one every other stage uses: there is work, and there is capacity.

### Measured, six seconds each, 4096-byte pages, identity workload

| pool | lost | recovered | note |
|---|---|---|---|
| 4000, before | 1923 / 4243 / 4046 | 79.7% / 61.6% / 62.3% | one request per loss, no re-arm |
| 4000, after | 1513 / 2980 / 1529 | 96.2% / 76.5% / 96.8% | window affords roughly one re-arm |
| 16000, after | 2897 / 2664 / 1976 / 3919 / 2353 / 3075 / 3973 | **100% in all seven runs** | `gone=0`, `stale=0`, `resend_fail=0` |

Throughput is unchanged to better across the change: 45-74 Gbit/s one-way, the spread being
run-to-run variance on this link, not an effect of repair. Repair traffic at full recovery is
under 0.1% of the link.

The residue at a 4000-page pool is not a protocol failure, it is the geometry: at that size
`rearm_gap` is a large fraction of `horizon`, so there is room for about one retry. Full
recovery needs a pool several times the send depth, which is now expressible.

## The invariant core

`mesh-flow.c` and the routing it implements **do not change for a workload.** They are
fixed against the identity function, and that is not a figure of speech: routing *is*
the identity case, so a core that is correct for `id` is correct for every `f`.

The build enforces it by file boundary:

| file | may change |
|---|---|
| `mesh-flow.c` | no — page table, LISSEN/DISPATCH/YELLER, budget, telemetry |
| `mesh-f.h` | no — wire header and flags |
| `f-identity.c` | **this is the workload** |

```
make F=f-yourthing.c
```

`mesh_f(payload, bytes, h, node_idx)` receives a payload pointer, a length already
clamped to the page, its header, and this node's index. It may read and write the
payload in place and may rewrite header fields to redirect a page. It may not allocate,
block, or retain the pointer past return.

If a workload seems to need a change to the core, that is a sign the workload wants the
data plane to interpret its bytes — which costs the budget in the table above and
breaks every other workload sharing the fabric.

## Addressing and routing

The wire header is 24 bytes, 0.59% of a 4096 page:

```c
struct wire { uint32_t magic, path, stream, seq; uint16_t bytes, src, dst; uint8_t flags, hops; };
```

`src` and `dst` are node indices — *this page is from node A, for node B*. There is no
authentication and none is wanted: the topology is cables we plugged in ourselves, so
this is a routing tag, not a credential.

`flags` carries `F_META`, which is the whole of the "is this for us" question. A page
either is graph metadata that every peer along a cycle should observe to maintain its
world model of the entire graph and its load, or it is opaque payload that no hop may
interpret. One bit, tested once per page, no payload dereference either way.

### Routing is a compiled path, not a table

`path` holds the route as **2 bits per hop** — a node has at most three fabric ports,
so an egress port is two bits. Each hop reads the low bits, shifts right, forwards. A
`uint32` therefore carries 16 hops:

| field | hops | nodes reachable at degree 3 |
|---|---|---|
| `uint32 path` | 16 | 43,046,721 |
| `uint64 path` | 32 | 1.85 × 10^15 |

Per-hop cost is a mask, a shift and a store, inside the 24 bytes the header already
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
