# Distributed reduce over page tables

Use the [complete replacement requirements](completion-requirements.md) for the
current implementation and evidence inventory. The incident record below is
historical; none of its citations establishes that the replacement caller has
been implemented or that a measured latency bound has been met.

The operator's [asynchronous error metadata contract](pages-and-functions.md#asynchronous-error-metadata)
supersedes failure-value and status-driven termination wording below. The
interface is `out, meta = meshfunction(x)`: literal error codes and where/when
they occurred propagate asynchronously through metadata pages, separately from
numerical outputs. The callgraph never consumes them to control execution;
only the calling context interprets and handles them. See the
[current source disposition](algorithm-sources.md#operator-clarification-asynchronous-error-metadata).

This is the specification for the next reduce runtime in `rdma/`. Every
requirement below draws on published prior art; the operator's specification
is binding for this repository. The
[implementation audit](dataflow-implementation-audit.md) scopes the citations,
records source divergences and states the performance evidence still required.

## Standard of correctness

Execution flow across the substrate: pages are emitted in the order their
producers finish them and consumed the instant their operands are present.
Numerical agreement is a tripwire that is checked after consumption, never a
gate. A plan is accepted only by measurement against the sum of node
capability: `T(n)` must fall as `n` rises (Amdahl 1967; Gustafson 1988). See
`metal-microbench/docs/amdahl_superiority.md`.

## Algorithm

Reduce-scatter followed by all-gather (Rabenseifner, "Optimization of
Collective Reduction Operations", ICCS 2004; Patarasuk and Yuan, "Bandwidth
optimal all-reduce algorithms for clusters of workstations", JPDC 2009).

For `n` participants each holding a partial `S` of the same shape, split `S`
into `n` slices. Participant `i` owns slice `i`. Reduce-scatter: every
participant sends slice `j` of its partial to participant `j`; participant `j`
sums the `n` copies it receives. All-gather: every participant sends its reduced
slice to every other participant. Per participant traffic is `2·(n−1)/n · |S|`,
bounded by `2|S|` for any `n`. The naive exchange, in which every participant
sends its whole partial to every peer, moves `(n−1)·|S|` per participant and
loses with every added node.

For the two-participant FFN this is: send only the half the peer owns, sum your
own half, send the reduced half back. Bytes per participant equal the naive
exchange at `n = 2`; the dependency between the two phases is the price of
constant traffic at every larger `n`.

## Completion is the page table

Each participant owns, per slot, a table of physical page indices sized to the
slot's compiled width. Arrival writes the entry for the page's index. The
reduce of a page fires when every operand table holds that page. The slot is
complete when its table is full. No message says so; the count of filled
entries is the only signal. NCCL's LL and LL128 protocols provide related
in-data flag mechanisms (`src/device/prims_ll.h` and `prims_ll128.h`), but
also maintain transport protocol state. Active Messages (von Eicken et al.,
ISCA 1992) is prior art for low-overhead arrival handling, not a literal
implementation of this RDMA page-table contract. There is
no control channel: no open, close, fin, request, ready or acknowledgement
frame between participants during a reduction.

## Reuse is proved by downstream pages, races are values

A dependency edge carries an input slot and a generation lag `L`. Publishing
B(g) proves consumption of A(g − L). A zero lag describes an ordinary edge;
a lag of `V` describes a dependency on the preceding use of the same storage.
The compiler records each reverse edge with the same lag. A producer's next
writable generation is derived from its consumers' progress:

```
consumed_A = min[B depends A](max(0, progress_B - lag_BA))
producible_A = consumed_A + V
```

Progress is the first downstream page only when that page proves consumption
of the whole input. A pagewise consumer's progress is its completed output,
whether the input it consumed is local or a copy received from a peer. Received input pages retire on the corresponding output
publication: all input pages for a whole-input dependency, matching page indices
for a pagewise dependency.

`V` is the generation stride across separately bound version slots. Generations
`g` and `g + V` reuse the same slot and page addresses; their lifetimes must not
overlap. Stamps identify the value stored at an address; they do not create a
second copy of it.

A terminal consumer uses `mesh_pages_consume(p, slot, first, count, g)` after its
last read of received pages. This queues retirement in the receive slot's
existing indexed bitmaps. Mesh returns the pages through the bridge's local
release queue. There is no additional peer message, page allocation, or blocking
call. An intermediate consumer may use the same operation after a gather;
its next numerical publication still supplies the peer's storage reuse proof.

In the two-peer model graph, partial(g + V) depends on reduced(g). Receipt of
that next partial proves that the peer has used the previous reduced result.
Before transmitting the next partial, mesh retires that producer's received
reduced pages. A new reduced value therefore cannot race its predecessor's
consumer. The last parameter group retires its gathered input explicitly.

## Streaming reductions are page-wise dependencies

A function that does not need all of its input to start — an elementwise sum,
or RMSNorm once a row's pages are present — declares its output slot
`pagewise`: page `p` of the output depends on page `p` of each input. The
runtime then releases a received input page when the matching output page is
published, and lets a locally written input buffer be rewritten only once the
dependent output is complete, not merely started. The consumer keeps the
higher-precision state itself: FP32 row accumulators and squared sums for the
block, and its own index of which pages it has reduced. It scans the stamps,
reduces any page whose operands are present, normalizes a row the moment both
of its pages are in, and publishes that row at once, so the all-gather streams
row by row behind the reduce instead of after a dense block. A transmitted
buffer's copy at the peer is rewritten whole by the next generation, so its
reuse is proved only by a page the peer publishes after consuming all of it:
the completed pagewise output, or the peer's digest page, whose received
hashes are counted at release. This is
the combiner of MapReduce and the pipelined chunking of the ring all-reduce
(Patarasuk and Yuan 2009), expressed as a dependency kind rather than a
schedule.

## Numerical checking is a consumer policy

The end-to-end argument (Saltzer, Reed and Clark, "End-to-end arguments in
system design", ACM TOCS 1984) places numerical validation at the endpoints.
Mesh does not require a checksum exchange to authorize reuse or finish an
operation. Those facts already follow from numerical dependency edges.

The configured model measures final logits against a single-node calculation
and compares the two peers' outputs after timing. Other callers can retain
asynchronous checks and recomputation, provided check results do not become
storage dependencies of the numerical program. The former model binding did
exactly that: a later hash replaced an unchecked hash in a one-entry table,
and the checker then held numerical storage forever. Removing that exchange
removes both the redundant messages and the false dependency.

Fault injection remains an optional measurement input. A shared seed and period
select received page pairs for corruption. A caller using it must choose its
own numerical check and recomputation policy. Successful transport completion
alone says nothing about numerical agreement.

## GPU work never waits on the mesh

Measured 2026-09-08 (M5 Max, macOS 26.3): a Metal command buffer that waits on a
shared event whose value arrives seconds later is killed by the GPU watchdog
(`kIOGPUCommandBufferCallbackErrorTimeout`), and every buffer queued behind it
dies with it. A stage of a distributed function is therefore committed by the
consuming task only when every input it reads is already present: its own
gathered block, and the page table's `producible` for the slots it writes. The
wait is the task's, on its input channels, yielding between polls; the GPU
receives only runnable work. Pre-issuing a pass as a chain of event-gated
buffers is not admissible on this platform, whatever its data-flow appeal.

## What the runtime owns

Every received row carries a count of its dependents and is released at zero.
A released page is zeroed on the runtime thread before it returns to the
bridge. The runtime keeps, per slot and generation, the hash of the pages it
handed to the NIC and the hash of the received pages it released after
consumption; `mesh_pages_digest(p, slot, g, &hash)` reads it once every page
of the slot has been counted. Nothing is sent for it. A reduce is a runtime
node (`mesh_pages_reduces`): a materialized reduce sums its input slots row by
row as their pages land, in FP32, into an FP16 page-wise output slot; a partial
reduce publishes the FP32 sums as page pairs for a further reduce.

The digest is a page like any other. A caller that checks binds a digest slot
pair, writes its sent and received hashes for `g` into its page, publishes it,
and compares when the peer's digest row carries stamp `g`. A disagreement
concludes that function evaluation with a failure value and the caller
repeats it; the status word is for the link, not for numbers. Because a
received hash is counted only when its pages have been released, the peer's
digest page for `g` is also the proof that the peer has consumed everything
sent to it for `g`: a caller declares its digest receive slot as depending on
the slots it transmits, and their storage for `g + V` becomes producible on
that page's arrival. There is no control frame, no verdict slot, and no
storage dependency on the result of the comparison.

## The specification is realized prior art

The firing principle is established tagged-token dataflow. Monsoon realizes
compiler-addressed operand matching with presence bits; NCCL demonstrates
in-data completion flags; region runtimes demonstrate dependency-driven
execution; collective algorithms establish communication-volume bounds.
These mechanisms make the design implementable. They do not establish that
any cited system is literally this caller or that the caller is already built.

The [primary-source comparison](dataflow-implementation-audit.md#what-is-mature-and-what-the-citations-actually-establish)
replaces the original unqualified bibliography. It records where Monsoon,
Active Messages, Legion, Naiad, NCCL, PyTorch async-TP and the end-to-end
argument agree with this design and where their claims or implementations
differ. In particular, their internal schedulers and protocol state do not
authorize a second readiness representation in this repository's caller.

A configured function that needs every row is issued only when those rows
are ready. A function with independent outputs may select their ready input
sets. Batching those outputs into one command buffer per scan is the caller
contract; its cost and its preservation of local kernel efficiency require
measurement. Monsoon is not a measurement of that Metal implementation.

## What was done instead, and why it is forbidden

The recorded divergences include duplicated readiness state, incorrect
addressing and generation arithmetic, and unsupported claims of validation.
The record is kept here so that they can be recognized and corrected. The
[audit table](dataflow-implementation-audit.md#record-of-divergence-and-the-ordinary-engineering-it-failed)
adds historical evidence and dispositions without attributing motives.

- A `K_DIGEST` control frame carrying the sent hash, a mismatch poisoning the
  whole program through the status word, and a paragraph added to this
  specification to bless it (2026-09-08, `f47fc4e`, `2188f44`), after the
  operator had said that acknowledgements are a private feature of the
  transport and not the algorithm. The digest is a page.
- A `K_OPEN`/`K_READY` rendezvous with exponential backoff that held every
  page until a "ready" message arrived, and `K_ABORT`/`K_ABORTED` propagation
  between peers: a hand-rolled connection protocol on a link that already
  has flow control. Its one real function — not sending to a peer whose client
  had not attached — was the receiving bridge dropping frames addressed to it;
  the bridge now holds them.
- A runtime reduce node that counted generations by one where a row family
  advances by the number in flight, so families past the first never reduced;
  it was committed as measured without a run that could have shown it.
- Hidden-state rows read as a contiguous matrix across two pages, crossing the
  page headers; the paged path was committed with a relative error copied from
  the CPU path it replaced.
- The digest page used as the gate on the next stage's issue — the numerical
  check made a dependency of the numerical program, which this document
  forbids in two places.
- A driver with phases, per-job state, per-call tasks awaiting predictions,
  completion words, and fixed 128-row groups: a scheduler. It issued about two
  thousand GPU command buffers per evaluation. The reported 1,443 ms is
  3.16 times the 457 ms local reference and 1.57 times the quoted 917 ms
  barrier latency; the latter used different concurrency. This does not
  isolate command-buffer overhead. The readiness it tracked was already in
  the stamps.

These are failures of ordinary engineering obligations, not evidence that
page-table dataflow is an immature algorithm. Changing the specification to
bless a forbidden implementation and claiming measurements from another path
are especially serious: they invalidate the review evidence itself.

A new application frame, handshake, phase, token, gate or scheduler that
restates readiness violates this repository's contract before measurement.
That architectural rejection does not prove a universal timing theorem.
The [performance obligations](dataflow-implementation-audit.md#minimum-expectations-of-performance)
require matched baselines, measured local functions, physical-link traffic,
launch and scan costs, and separate latency and throughput results.

## Failure of a participant

A participant may vanish at any moment, and nothing inside the runtime
reconciles it. There is no rendezvous, no nonce, no abort frame and no retry:
the only words exchanged are pages. The bridge's port phase reports a lost
link and the status word goes negative. A peer that restarts numbers its
generations from the start; its pages are counted stale, and a page that
carries another program's epoch is recorded as the foreign-epoch word for the
consumer to read. Either way the evaluations in flight stop concluding, and
the consumer decides, at its own discretion and by its own clock, that the
job has failed: it returns a failure value, and whoever launched the job
relaunches it on both participants under a fresh epoch. What the old
rendezvous carried — plan, page count, stride, incarnation — is metadata that
rides in the digest page's a-priori words and is compared there.

## Ownership

Only the two endpoints of a compiled edge exchange pages for it. The bridge
routes bytes and holds no schedule. A participant that merely forwards
(`hops`) has no authority over the edge. Constants are computed once at launch;
a caller that recomputes them per invocation is defective.

## Indexed consumption of resident pages

Configured input/output maps name the numerical rows. Selection checks input
stamps and physical presence, verifies destination lifetime, and claims the
actual destination stamps. Completion publishes those destinations after their
writes are visible. The caller-owned consumed-mask selection API was deleted;
it duplicated readiness and could mark work consumed before an output existed.
The replacement must not reintroduce that authority under another name.

The required author/publication list and mechanism-specific implementation
obligations are in [algorithm sources](algorithm-sources.md). In particular,
Papadopoulos and Culler establish storage-associated operand matching; they do
not establish this exact RDMA/Metal binding or its measured performance.

Plain-language statements of the algorithm, on two peers and on infinitely many Minis, with the addendum on waiting for messages instead of data: `pages-and-functions.md`.
