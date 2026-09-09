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
of the whole input. A pagewise consumer of local storage uses completed output
progress instead. Received input pages retire on the corresponding output
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
row by row behind the reduce instead of after a dense block. Transmitted pages
are copies at the peer, so their reuse keeps the whole-buffer proof. This is
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

## Indexed consumption of resident pages

`mesh_pages_select(p, slots, count, group, generation, consumed, indices)`
compacts the ready group indices into caller-owned storage. The caller binds
valid slot indices with equal page counts, a nonzero group width dividing that
count, and masks/index storage sized to the number of groups. A single consumer
owns each mask. No allocation or waiting occurs in selection.

For a group `i`, let `J_i` be its page indices and `S` its input slots:

```
ready_i = (consumed_i != g) & AND[s in S, j in J_i](stamp_sj == g & table_sj != ABSENT)
indices = compact(i, ready_i)
consumed_i = select(ready_i, g, consumed_i)
```

Readiness uses acquire loads. Receive publication stores the physical address
before releasing the generation stamp, so observing a new generation cannot
expose the previous address. `mesh_pages_data` resolves an existing selected
entry into the registered payload. Dependency ownership must remain held while
that payload is consumed. Recovery resets the consumer masks when generations
restart. The producer/consumer arithmetic and the meaning of a group remain
outside mesh.

Plain-language statements of the algorithm, on two peers and on infinitely many Minis, with the addendum on waiting for messages instead of data: `pages-and-functions.md`.
