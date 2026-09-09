# Distributed reduce over page tables

This is the specification for the next reduce runtime in `rdma/`. Every
requirement below instantiates published prior art; the citation is the
authority, the sentence here is the binding form for this repository.

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
entries is the only signal. This is the in-data completion flag of NCCL's LL and
LL128 protocols (NVIDIA NCCL, `src/collectives/device/prims_ll.h`), where the
receiver polls the payload's own flag word, and the counter-completion of
one-sided RDMA writes (Active Messages, von Eicken et al., ISCA 1992). There is
no control channel: no open, close, fin, request, ready or acknowledgement
frame between participants during a reduction.

## Reuse is proved by downstream pages, races are values

If buffer B depends on buffer A, then a page of B stamped `g` from every
participant that produces B is proof that A(g) was consumed everywhere it was
needed. That is the whole rule for freeing and reusing storage:

- A transmitted or local buffer may be written for generation `g + V` once
  every buffer that depends on it holds some page stamped `≥ g`. The runtime
  keeps one word per buffer, the highest stamp present, and takes the minimum
  over dependents. Nothing is sent to establish this; with peers that DMA-read
  each other there is no local completion at all, and this rule still holds.
- A received buffer's pages for `g` are returned to the bridge when the local
  function that depended on them publishes its first output page stamped `g`.

`V` versions per buffer is a launch-time constant on every participant; a
version is a separate page set, so two live generations `g` and `g + V` never
share an entry. Every table entry carries the generation that wrote it. A
gather that finds a stamp it did not expect has gathered a wrong operand; the
value is wrong and the check in the next section finds it. No state machine
guards the entry, no acknowledgement is exchanged, no node is numbered.

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
row by row behind the reduce instead of after a dense block. Transmitted pages
are copies at the peer, so their reuse keeps the whole-buffer proof. This is
the combiner of MapReduce and the pipelined chunking of the ring all-reduce
(Patarasuk and Yuan 2009), expressed as a dependency kind rather than a
schedule.

## Integrity is checked after the fact and repaired by recomputation

The end-to-end argument (Saltzer, Reed and Clark, "End-to-end arguments in
system design", ACM TOCS 1984): integrity belongs to the endpoints that use the
data, not to the transport. Each participant computes, asynchronously and
after the reduce has consumed its operands, a hash of the operands it sent and
the operands it received for generation `g`, and publishes that hash as one
page of an ordinary one-page slot. The peer's hash page arrives into a one-page
receive slot. A compare node fires when its two-entry table is full. If the
words differ, the caller re-queues that step as a new generation. This is
re-execution from lineage (Dean and Ghemawat, "MapReduce", OSDI 2004; Zaharia
et al., "Resilient Distributed Datasets", NSDI 2012): the transport is not asked
to be lossless, the computation is cheap to repeat, and no correct step waits
for the check of any other step.

## Faults are injected on purpose

A shared seed and a period `N` select `1/N` of `(slot, generation)` attempts.
For a selected attempt, a chosen pair of received pages is XORed into each
other before the reduce reads them. The compare node then disagrees and the
step is redone. If one page of the chosen pair was already released by a
page-wise consumer, the remaining page alone is corrupted; the injector never
dereferences a released entry. The proof that the check is non-blocking is a measurement, not
a test: as `N` falls, the mean and the variance of step latency must rise,
because bad steps are recomputed while good steps continue. If they do not
rise, correct steps were being held behind the check.

## Failure of a participant

A participant may vanish at any moment. Rendezvous happens once per
incarnation with a nonce; a nonce that changes means the peer restarted. The
bridge's port phase reports a lost link. Either poisons the program; between
function evaluations the tables are cleared, a new nonce drawn, and the
schedule resumes from the last consumed step (lineage again).

## Ownership

Only the two endpoints of a compiled edge exchange pages for it. The bridge
routes bytes and holds no schedule. A participant that merely forwards
(`hops`) has no authority over the edge. Constants are computed once at launch;
a caller that recomputes them per invocation is defective.
