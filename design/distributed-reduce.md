# Distributed reduce over page tables

The incident record below is historical; none of its citations establishes
that the replacement caller has been implemented or that a measured latency
bound has been met.

The operator, 2026-09-10, of the digest path (design/pages-and-functions.md): "hm
delete crc then bye-bye superfluous feature". Mesh does not hash payloads, exchange
or compare digests, or retain pages for checks. Device and transport error metadata
remains. The digest mechanisms in the incident record are history.

The operator, 2026-09-10 ([asynchronous error metadata](pages-and-functions.md#asynchronous-error-metadata)):
"error status returns are allowed by the specifications we use... but it can never
be synchronous or blocking or consumed by a callgraph itself... it can only be
passed monadically through a metadata channel... s.t. calling contexts can call
`out, meta = meshfunction(x)`, with literal values corresponding to error codes and
where/when they happened in a callgraph reported in the meta channel, not the
output channel, for calling consumers to interpret and handle...". The failure-value
and status-driven termination wording below predates it. See the
[current source disposition](algorithm-sources.md#operator-clarification-asynchronous-error-metadata).

This was the design of the next reduce runtime in `rdma/` (September 2026), with
the published prior art behind each part.

## Execution flow

Pages are emitted in the order their producers finish them and consumed as soon
as their operands are present. Numerical agreement is checked after consumption
(the end-to-end argument, below).

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

For the two-participant FFN each participant sends only the half the peer owns,
sums its own half, and sends the reduced half back. Bytes per participant equal
the naive exchange at `n = 2`; the dependency between the two phases is the price of
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
`g` and `g + V` reuse the same slot and page addresses; their lifetimes do not
overlap. Stamps identify the value stored at an address; they do not create a
second copy of it.

Received pages are the receiver's own rows: they land in the landing entries
it presents on its stream (design/transport.md section 14, `struct mesh_entry`),
and no page goes back to the bridge. No call retires them; the
`mesh_pages_consume` this design named is in no source. A program reuses a
landing buffer in its own order and guards that buffer in its own source
(design/transport.md section 14, "What a landing writes"). Its rows return to
the region when it returns its lease (`mesh_lease_return`, rdma/mesh-call.h:22)
or its process ends; a row that a posted receive still names stays held until
that receive lands or its port's child exits (design/transport.md section 10).
There is no peer message, page allocation or blocking call.

In the two-peer model graph, partial(g + V) depends on reduced(g). Receipt of
that next partial proves that the peer has used the previous reduced result, so
a program that sends reduced(g + V) into the buffer where reduced(g) landed
writes it only after the peer's last read. A new reduced value therefore cannot
race its predecessor's consumer. That proof is the program's own order, not a
transport call.

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
the completed numerical output. This is
the combiner of MapReduce and the pipelined chunking of the ring all-reduce
(Patarasuk and Yuan 2009), expressed as a dependency kind rather than a
schedule.

## Numerical checking is a consumer policy

The end-to-end argument (Saltzer, Reed and Clark, "End-to-end arguments in
system design", ACM TOCS 1984) places numerical validation at the endpoints.
Mesh has no checksum exchange. Reuse and completion follow from numerical
dependency edges and physical device ownership.

The configured model measures final logits against a single-node calculation
and compares the two peers' outputs after timing. The former model binding made
a check result a storage dependency of the numerical program: a later hash
replaced an unchecked hash in a one-entry table, and the checker then held
numerical storage forever. Removing that exchange
removes both the redundant messages and the false dependency.

Fault injection was an optional measurement input: a shared seed and period
selected received page pairs for corruption. Successful transport completion
alone says nothing about numerical agreement.

## A GPU wait on the mesh, measured

Measured 2026-09-08 (M5 Max, macOS 26.3): a Metal command buffer that waits on a
shared event whose value arrives seconds later is killed by the GPU watchdog
(`kIOGPUCommandBufferCallbackErrorTimeout`), and every buffer queued behind it
dies with it. In this design a stage of a distributed function is committed by
the consuming task only when every input it reads is already present: its own
gathered block, and the page table's `producible` for the slots it writes. The
wait is the task's, on its input channels, yielding between polls; the GPU
receives only runnable work.

## What the runtime owns

Every received row carries a count of its dependents and is released at zero.
A released page is zeroed on the runtime thread before it returns to the
bridge. A reduce is a configured function: it sums available input pages in
FP32 and publishes its configured output pages. Further reduction or numerical
consumption depends on those pages. Raw device and transport errors are carried
separately as asynchronous metadata. No hash state or digest dependency exists.

## The specification is realized prior art

The firing principle is established tagged-token dataflow. Monsoon realizes
compiler-addressed operand matching with presence bits; NCCL demonstrates
in-data completion flags; region runtimes demonstrate dependency-driven
execution; collective algorithms establish communication-volume bounds.
These mechanisms make the design implementable. They do not establish that
any cited system is literally this caller or that the caller is already built.

Monsoon, Active Messages, Legion, Naiad, NCCL, PyTorch async-TP and the
end-to-end argument agree with this design in places and differ in others.

A configured function that needs every row is issued only when those rows
are ready. A function with independent outputs may select their ready input
sets. The caller batches those outputs into one command buffer per scan; its cost
and its effect on local kernel efficiency are not measured here. Monsoon is not a
measurement of that Metal implementation.

## What was done instead, and why it was wrong

The record lists what went wrong: readiness state kept twice, wrong addresses
and generation counts, and claims of validation with no run behind them. It is
kept here so each one can be seen and fixed.

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
  check made a dependency of the numerical program.
- A driver with phases, per-job state, per-call tasks awaiting predictions,
  completion words, and fixed 128-row groups: a scheduler. It issued about two
  thousand GPU command buffers per evaluation. The reported 1,443 ms is
  3.16 times the 457 ms local reference and 1.57 times the quoted 917 ms
  barrier latency; the latter used different concurrency. This does not
  isolate command-buffer overhead. The readiness it tracked was already in
  the stamps.

The spec was changed to bless a form it had excluded, and numbers from one path
were claimed for another.

Readiness lives in the stamps. M2 (design/rubrics.md), the operator's pair: lower,
"if you have successfully 'asked permission' to send data"; upper, "you could have
sent the data instead" (10-19); ranked there, `READY`-style keyword states ≺ none
(10-18).

## Failure of a participant

A participant may vanish at any moment, and nothing inside the runtime
reconciles it. There is no rendezvous, no nonce, no abort frame and no retry:
the only words exchanged are pages. A lost link reads ~0 in the stream's `sent`
and `landed` words, and a page whose data does not come lands as a dud, a page of
all ones (design/transport.md sections 1 and 10). A restarted bridge sets every
stream not live with its landed words ~0, so a communicator of the old pairing
fails as a value and is made again (design/transport.md section 2). The
transport exchanges no epoch, session or stamp (design/transport.md section 3).
Either way the evaluations in flight stop concluding, and the consumer decides,
at its own discretion, that the job has failed: it returns a failure value, and
whoever launched the job relaunches it on both participants. What the old
rendezvous carried — plan, page count, stride, incarnation — is configuration,
fixed before numerical invocation, and is not exchanged.

## Ownership

Only the two endpoints of a compiled edge exchange pages for it. The bridge
routes bytes and holds no schedule. A participant that merely forwards
(`hops`) has no part in the edge. Constants are computed once at launch.

## Indexed consumption of resident pages

Configured input/output maps name the numerical rows. Selection checks input
stamps and physical presence, verifies destination lifetime, and claims the
actual destination stamps. Completion publishes those destinations after their
writes are visible. The caller-owned consumed-mask selection API was deleted;
it duplicated readiness and could mark work consumed before an output existed.

The author/publication list and the mechanism taken from each are in
[algorithm sources](algorithm-sources.md). In particular,
Papadopoulos and Culler establish storage-associated operand matching; they do
not establish this exact RDMA/Metal binding or its measured performance.

Plain-language statements of the algorithm, on two peers and on infinitely many Minis, with the addendum on waiting for messages instead of data: `pages-and-functions.md`.
