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
| registerable | 100 MR × 16.4 MB = 1.64 GB | **184 ms** of pool; `max_mr_size` caps one MR at 16.4 MB |
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
