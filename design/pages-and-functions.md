# Pages and functions

On the actual Thunderbolt substrate, payload intervals use SEND/RECV, as documented
by Apple TN3205; software publication of a received page does not require
hardware remote-write support.
Statements below about instant execution and topology-independent utilization
express the desired absence of added control dependencies, not a zero-latency
theorem or a measured performance guarantee.

Topology is discovered for each realized configuration. The 2026-10-24 morning
microprofile recorded Studio–Studio/first-Studio–Mini/second-Studio–Mini cable
counts 4/1/2 and ingress degrees 5/6/3. Read-only discovery at 12:57:27 PDT
found 4/1/1 paired cables and degrees 5/5/2, with one Studio–Mini pairing
unavailable. See [data-model.md](data-model.md#8-writes-over-many-hops-and-many-paths)
for the dated provenance and corresponding 140/120 GB/s advertised wire ceilings.
Two-peer and infinite-Mini examples below illustrate dependencies; they are not
topology constraints on either the compiler or its transport interface.

Plain statements of the algorithm, in the words of the operator's own turns:
mesh, peers, pages, page table, rows, stamps, use count, buffers as function
returns, functions, release, reduce, accumulator page, index page, NFE, status
word, metadata. The other words that have appeared in code ("stage", "chunk",
"version", "gate", "token", "group", "consumer", "scheduler") each named a
control-flow object standing in for something the page table already holds;
the algorithm below does not require those objects as extra execution handshakes.
Its logical page/range vocabulary is independent of the physical transfer quantum.

## One NFE on two peers

### Setup

One page table holds every page the runtime will use. Each function output
is a range of rows in that table. A row holds three things: which page, its
stamp, and its use count. The use count is the use-checker. This is the logical
operand/ownership model, not the bridge ABI: compiled span descriptors, local
ownership metadata and physical completion prefixes realize its dependencies.

Every function has a known number of output pages. Every function input is
the output pages of some function, on this node or on a peer. All of this is
known before the NFE starts.

The operator, 2026-09-10: "pages can be freed, pages can be consumed, but all
allocations needed within a page table for a complete compute graph should be
known a priori at the time a callgraph using these apis has been launched;
there cannot, should not, and will not be any operations which introduce data
dependent control flow, branching, eror recovery, etc. pages can be freed if we
need to 'recover' pages which are used by a hung kernel; our mesh api level
primitives should let us invalidate a page table in this manner whenever we
want to. furthermore all of our nodes will alway,s without exception, either
have nvme ssds or be rdma-peered-transitively to a system which does have nvme
ssds, allowing things like invalidated and destroyed dnn parameter pages to be
retransmitted to a node which somehow doesn't have them en-disk and also
deleted them and wants them again."

The operator, 2026-09-10: "recovery is something which is only sanctioned
through the mechanism of a totally feed forward nfe computation callgraph being
re-run by the calling process because they don't like the results they got.
there are to be no parity mechanisms or syncs, guards, waits, checks, or
inferences in the mesh computation transport and reduction layer, and functions
which use this layer are not to introduce data dependent control flow
attempting to add overhead of this sort or any sort."

### Asynchronous error metadata

The operator, 2026-09-10: "error status returns are allowed by the
specifications we use... but it can never be synchronous or blocking or
consumed by a callgraph itself... it can only be passed monadically through a
metadata channel... s.t. calling contexts can call `out, meta =
meshfunction(x)`, with literal values corresponding to error codes and
where/when they happened in a callgraph reported in the meta channel, not the
output channel, for calling consumers to interpret and handle..."

Metadata occupies page-table pages configured before launch: a return channel
beside `out`, not a private allocation, completion token or readiness gate. A
terminal device completion publishes the configured output pages whatever the
metadata holds. Their stamp says the pages are available, not that their
contents are correct: the calling context may reject them by `meta`. Without a
terminal completion the output is unavailable, and metadata published alone
carries no numerical value and does not say that outstanding work completed.

### The NFE, number k

1. A function runs on the GPU. It writes its output straight into pages. When
   a page is written, its row gets stamp k. Nothing else is said.
2. Pages the peer needs go over the mesh. On the peer they land in free pages.
   The peer's rows for that output get the page and stamp k.
3. A reduce is a function. Its inputs are two buffers: my partial and the
   peer's partial. It looks at the rows of both. When row p carries stamp k in
   both, it reads page p from each, adds them in high precision into an
   accumulator page, and marks p in its index page. When every page of a row
   of the result is in, it writes the normalized row into its output pages and
   stamps them k. It does this page by page as pages arrive. Its output pages
   go to the peer.
4. The next function has two inputs: my reduce output and the peer's reduce
   output. When all their rows carry stamp k, it runs. It reads those pages
   directly. There is no copy.
5. A function's reads depend only on the input page/range stamps they use.
   The compiled GPU stream may already contain that function with a bounded
   availability dependency. This is not a host scheduler waiting for a peer,
   and not a separate permission message. The current kernels encode GPU
   prefix observations; "the GPU never waits" was not a description of them.
   A ready page should not wait for an unrelated operand's last page.

### Freeing

6. A page is reusable after its final local arithmetic read and final outgoing
   DMA read. The compiler records those distinct uses. A naturally returning
   downstream result may prove the remote read ended, but only if the actual
   dependency graph carries that result back to the owner. The bridge's
   existing final DMA-read completion can independently release a source;
   this is local ownership bookkeeping, not a remote-consumption ACK. The
   destination's consumers have their own lifetime. No receipt message is
   introduced. If the owner has neither proof, it retains the page. Release
   does not zero the page: each next producer must define the values it exposes.
7. A page is written again only after release. Two NFEs can be in flight
   because rows for NFE k+1 point at other pages until the rows for NFE k are
   released. Reuse must be ordered before the next **first write**, not merely
   its eventual publication. A publish-after-read fact alone does not prove
   in-place overwrite safety.

### Link error

8. A negative link status is reported with its location and occurrence in
    asynchronous metadata for the affected invocation. The callgraph does not
    consume that status. The calling context may reject the result and rerun
    the complete feed-forward NFE; mesh does not perform recovery or replay.

## The same NFE on infinitely many Mac Minis

(After `metal-microbench/docs/mesh_distributed_reduction_analysis.md`: the
program installed once; weights where they live; each intermediate at its
next consumer; a finished piece enabling its local continuation with no host
decision; every Mini with at most three links; a result at depth D back in D
hops; the per-link ceiling standing for any number of workers; a partition
consumed as soon as it has arrived.)

### The mesh

There are infinitely many Mac Minis. Each one has at most three links. Each
link joins two adjacent Minis. We look at one link and the two Minis on its
ends. Everything below is what one Mini does. Every Mini does the same thing.
Nothing a Mini does depends on how many Minis there are.

### What a Mini holds

A Mini holds its share of the program's weights. They never move. It holds
one page table. The table has a row for every page the Mini will ever use. A
row holds a page, a stamp, and a use count. This describes the Mini's logical
operand state, realized by the storage and transport metadata described above.
Every function output is a range of rows in the table. Every function
input is some function's output rows, on this Mini or on an adjacent Mini.

### One NFE, number k

1. A function's required input rows carry stamp k before their first read.
   AOT dependencies express this without asking a peer or dispatching a host
   scheduling decision. Independent pages can enable independent arithmetic.
2. The function writes its output into pages. When a page is written, its row
   gets stamp k. Nothing else is said.
3. An output page that an adjacent Mini needs goes over the link to that Mini.
   It travels one hop. It lands in a free page there. The row for it on that
   Mini gets the page and stamp k. A result at depth D from here arrives after
   D hops. Their service times, finite-pipeline startup and any contention on
   shared directed links also enter the transaction latency.
4. A reduce is a function. Its inputs are my partial and the partial that
   arrived from the adjacent Mini. When row p of both carries stamp k, the
   reduce adds page p of each in high precision into an accumulator page, and
   marks p in its index page. It does this page by page as pages land. It does
   not wait for the rest. When a row of the result is complete, it writes that
   row into its output pages and stamps them k. Those pages go one hop to the
   adjacent Mini.
5. The next function reads my reduce output pages and the pages that came
   back from the adjacent Mini. When all those rows carry stamp k, it runs. It
   reads the pages where they are. There is no copy.
6. Partials are combined as they pass. A Mini that has two partials for the
   same rows may add them before the pages go on. When that associative
   reduction preserves the declared numerical semantics and its output
   extent stays fixed, it bounds bytes per result on a link independently of
   the number of contributors. This is not true for arbitrary all-gathers,
   increasing output extents, or increasing rates of independent requests.

### Freeing

7. Apply the same local-read and outgoing-DMA-read ownership proof as above.
   A dependent's stamp is sufficient only where it already arrives by a
   natural data dependency and proves **every** relevant use. Feed-forward
   topology alone does not guarantee a return path. In a one-way migration
   from rank 1 to rank 2, an unrelated later landing on rank 1 proves nothing
   about rank 2; the source bridge can still retire rank 1's source after its
   own DMA reads end, without learning when rank 2 finishes arithmetic.
8. A page is written again only after release. Two NFEs can be in flight
   because the rows for NFE k+1 point at other pages until the rows for NFE k
   are released.

### Link error

9. A negative link status is reported with its location and occurrence in
    asynchronous metadata for the affected invocation. The callgraph does not
    consume that status. The calling context may reject the result and rerun
    the complete feed-forward NFE; mesh does not perform recovery or replay.

### What does and does not generalize to more Minis

Each Mini waits only on its actual input rows. Its ownership and availability
rules do not depend on the graph's size. Performance does: a dependency-chain
service sum and the directed-link workload/capacity bounds are lower bounds,
and queue order, release times, arithmetic placement and finite buffers can
increase completion time. Adding nodes may change those bounds, contention,
the collective decomposition and the useful output extent. Express all those
possibilities in the physical graph; do not turn a two-node example into an
interface restriction or claim topology-independent utilization from AOT
knowledge alone. Compare actual whole-function execution to explicit bounds.

## Addendum: waiting for messages about data, on infinitely many Minis

Suppose a Mini does not wait on its input rows. Suppose it waits for a
message that says the data is coming, or is done, or may be used: a token, a
gate value, a "ready" flag, a scheduler's decision, a completion callback that
stands in for the pages. Then it acts on the message. This is what goes wrong.

1. **The message is a second thing on the same link.** The pages take one
   hop. The message about them takes a hop too. Every dependency on the chain
   now costs a hop for the pages and a hop for the message. On a chain of
   depth D the NFE waits for D messages it did not need. On an infinite mesh
   D is not bounded, so the extra wait is not bounded either. Waiting on the
   rows costs nothing extra: the row is stamped when the page lands.
2. **The message can be true and the data still absent.** A message says
   something about the sender's state. The pages may still be in flight, or
   lost, or land out of order, or be corrupt. Acting on the message is acting
   on a claim. To make the claim safe, the sender and receiver need
   acknowledgements, retries, and ordering. That is a protocol on top of page
   delivery. The calling context determines whether the numerical results are
   acceptable; mesh adds no correctness exchange.
3. **The message can be false and the data present.** The pages landed and
   the rows are stamped, but the message is late, or the sender concluded
   early, or the link that carried the message broke while the link that
   carried the pages did not. The Mini sits with everything it needs and does
   nothing. Waiting on the rows would have run the function at once.
4. **The wait binds this Mini to another Mini's control flow.** A message
   comes from a sender's scheduler, not from data. If that scheduler is slow,
   this Mini is slow, and every Mini downstream is slow, even though their
   pages have all arrived. On an infinite mesh there is always a slowest
   scheduler somewhere, and the chain of "waiting for whose message" has no
   end. Waiting on rows binds a Mini only to the pages in front of it.
5. **A GPU wait on an absent message can idle or fail.**
   We measured this: a command buffer that waits on a signal for seconds is
   ended by the watchdog, and every buffer behind it dies with it. A GPU given
   only useful input dependencies avoids that extra protocol wait. Actual
   utilization still depends on data availability, dispatch and arithmetic.
6. **Messages need their own storage and their own freeing.** Every message in
   flight is state: a queue entry, a counter, an event value, a callback. On an
   infinite mesh the number of messages in flight is not bounded, so this
   state is not bounded, and it is freed by other messages. Pages are freed by
   release when a typed proof covers all of their final reads. That state is
   bounded by the compiled live pages; no general feed-forward graph promises
   a returning stamp for every remote consumer.
7. **A speculative message lets a function run before its data.** "The pages
   will arrive" is not "the pages have arrived". A function that runs on the
   promise reads stale pages and writes a wrong output with a good stamp. The
   calling context may reject the result, but the work was wasted and the
   wrong output already travelled a hop.
8. **Fan-in multiplies all of the above.** A function whose inputs come from
   many Minis would need a message from each, each a hop, each a race with its
   pages. The rows give one check for all of them: are they all stamped k.

The operator, 2026-09-10: "syncs, messages, ping pongs, and gaurds or waits
are not part of a specification... remember this..."

In the algorithm above a Mini acts on its required data. A row's availability
stamp is data evidence, not a permission exchange. The calling context receives errors through
asynchronous metadata, not through numerical values or a status gate inside
the callgraph. The same dependency rules apply whether it has one neighbour
or many; latency and utilization must be measured for the actual graph.

## Executable ownership proof

`rdma/lifetimes.py` compiles `Event`, `Lease`, `Read` and `Transfer` declarations.
Each read is an obligation. Existing natural events and monotone prefixes
provide proofs; all physical outgoing fragments must be covered. It emits
incremental retirement rules and page aliases justified by existing
happens-before edges, without inserting waits, messages or copies. The
allocator/bridge can observe a typed `source_done`; data functions cannot use
it as an execution prerequisite. A partial SEND completion must not cover
unposted source pages: presentation retention lasts until **all** compiled
reads of that source range have been scheduled and completed.

`prove_recurrent_gather` proves two-slot source, landing and relay reuse only
for a recurrence where every rank consumes all source pieces before its next
publication. For independent feed-forward writes or asymmetric migration,
there is no such theorem. Unknown first-write order gets separate storage;
unknown retirement evidence retains storage. See
[whole-function.md](whole-function.md#typed-one-sided-lifetimes) and
`rdma/tests/test_lifetimes.py` for the proof and counterexamples.

The byte-extent revision resolves a logical page or affine row into exact byte
intervals. A local 16-byte `mesh_dma_span {offset, bytes}` names one payload;
`mesh_entry {spans, units, word, base}` names an immutable array of those spans.
The descriptor bytes are not payload headers. Allocation pages, 1 GiB registered
extents, 4 KiB hardware credit frames, and actual SGE payload lengths are distinct.
Short or unaligned spans transfer only their declared bytes. Exact intervals
leave adjacent destination values untouched, without padded transport copies.

[`ncclMeshWritePrepare`, `ncclMeshWriteRowsPrepare` and `ncclMeshWriteSpansPrepare`](../rdma/nccl.h) bind
placements and common packet cuts at configuration time. The rows form maps
logical byte k to `(k / row_bytes) * local_pitch + k % row_bytes`; all rows use
one span table, with no per-row host execution call or entry. Registration cuts
use the actual phase of each row. The indexed form concatenates each local
span list in logical order; endpoints may have different counts, gaps, and
physical ordering. Source intervals can repeat, while overlapping landing
intervals fail configuration. The common cuts preserve every local boundary
and actual registration phase. No numerical buffer is packed.
`ncclMeshWriteExecute` reuses the binding with no new placement exchange or
receipt. Indexed preparation has local sanitizer and protected-memory tests;
it has not yet been included in the live cluster evidence below.

This source revision is wire 135/layout 154. Its existing contiguous/affine
[live validation](transport.md#exact-byte-validation-on-2026-10-24)
passed three-rank writes from 3 bytes through 1 GiB + 19 bytes, including ragged
rows, exact byte accounting and preserved neighbors; larger transfers used all
six cables. Two-node and three-node fused runs each passed 14 cases and 94 calls
per rank. A complete three-byte numerical function took median 31.290 µs with
exact writes versus 35.569 µs with padded transfers across five trials of 128
calls per mode. These are whole-function observations, not isolated NIC timing.
The bridge deployment was temporary and the preceding bridge was restored.

Each source extent remains immutable until every declared physical reader
finishes, including reads not posted yet. Pins record physical byte intervals,
so crossing an allocation row or using an alias cannot hide a backing reader.
A destination's consumer and any forwarding read have separate lifetimes.
Serial receiver execution alone does not order an already preposted remote
write; one-way reuse still needs a natural causal proof or distinct storage.
The earlier masked frame train obeyed the same proof, but its padding is not a
requirement of the byte-span interface.

The immutable span table is itself a closure operand. SEND and LAND prefixes
retire its two directions independently, and taking its list entry does not
finish the bridge's later CPU reads of the table. A local stream give prevents
new posts; the bridge's subsequent owner transition proves its CPU reader ended.
Keep descriptor storage until completion or that local quiescence proof. None
of these ownership proofs introduces a network ACK. Affine address geometry and
disjoint final stores remain separate obligations from ownership reachability.

## What this is called elsewhere

The operator, 2026-09-10, of an agent's review of the digest path ("its
bit-at-a-time CRC loop adds substantial work even though digest results do not
gate arithmetic"): "hm delete crc then bye-bye superfluous feature". Mesh does
not hash payloads, exchange digests, compare them, or retain pages for those
operations. Literal device and transport errors travel through asynchronous
metadata, and the calling context reads them.

The firing principle is tagged-token dataflow; operand slots and presence
bits have hardware prior art in Monsoon. In-data flags, dependency-driven
execution and pipelined reductions also have published implementations.
None of them is the exact caller described here. The reduce description and
the historical incident record are in [distributed-reduce.md](distributed-reduce.md).
