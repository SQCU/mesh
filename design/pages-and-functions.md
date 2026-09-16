# Pages and functions

The [user's requirements](collective-goals.md) define scope. This note describes
operand and lifetime relationships; it prescribes no executor or scheduling method.
The caller supplies the mesh and tensor placement. Mesh realizes their storage,
indexed dependencies and numerical function bindings before invocation.

Pages back values; they do not define tensor dimensions or numerical call extents.
Configured views name the actual registered storage. Presence denotes available
values, and reader ownership retains their storage through actual use. Transport
completion and numerical availability are different facts.

Functions consume available indexed input regions and publish their output regions
after the corresponding writes are visible. A publication does not finish or pause
the enclosing operation. Other producer work and consumers with available inputs
continue independently. Missing inputs constrain only the work that reads them.

Distinct simultaneously live values have distinct storage. Source storage remains
valid until numerical readers and the transport device finish accessing it. That
lifetime does not impose a dependency on independent production.

The substrate is Thunderbolt SEND/RECV, as specified by Apple TN3205. Mesh binds
send and receive endpoints to registered storage and uses the existing completion
mechanisms. Numerical callers do not implement transport or page-table operations.

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture* (1990),
supplies operand-associated presence prior art, not a requirement for a function
scan. The JAX authors' Pallas collective matmul supplies indexed forwarding and
distinct live receive buffers. Sources are [references](algorithm-sources.md),
not additional requirements.

## What the page table is

A logical tensor section is an index range with a configured layout. Its page-table
entries name actual registered backing. A view alone does not establish contiguity:
setup decomposes the operand into the contiguous sections required by the chosen
numerical calls, preserving contraction contributions and output coordinates.

For section s, outstanding ownership is R(s) = S(s) + P(s) + C(s) + T(s): the
temporary setup owner, unfinished production, configured numerical owners and
transport uses. Setup derives known uses from the feed-forward graph and drops
S after all bindings are realized. Observations are declared consumers and count
in C. Existing completions discharge those uses. Early publication does not release an
unfinished producer. The final reference publishes a free-pool entry without
clearing payload bytes. Independent work already has its configured sections and
never awaits reclamation.

`TensorPart` contains rank, byte extent, a C section descriptor and value flags;
it has no heap-object reference or destructor. Copying or retaining the descriptor
does not add a reader or delay reclamation. `MeshMemory.sections` holds S during
configuration. `Mesh.start` releases it after binding all functions and transfers,
before starting numerical workers; failed configuration returns any setup
references still held by that array. The array is then empty. Backend views keep
the memory mapping alive; their actual reads are owned by the declared calls.

All retains occur before S is dropped, when R is necessarily positive. They need
one increment, with no resurrection check, retry or rollback. After setup, the
count only decreases. Its transition from one to zero publishes one indexed
free-pool bit; there is no sealed state, queue claim or retry. Forced teardown
preserves unfinished device ownership until the bridge closes the queue pairs.
The [reclamation events](#reclamation-events) below replace the former collector.

For a transient input, each indexed use is a numerical owner and its native
completion releases that reference. For an immutable shared input, the prepared
function holds one reference per binding across all its value indices. Destruction
of that function follows completion or cancellation of all its call records and
releases the shared reference. Thus the constant remains live throughout every
native read without per-invocation reference updates. This groups equal storage
lifetimes; it does not remove input-arrival dependencies or alter tensor values.

For the call storage itself, let O be the Mesh owner reference, W the active
worker references, and I the instances with outstanding numerical call records.
Its count is O+W+I. Startup acquires W+I before any worker starts. Each instance
counts its unissued and issued records together; issuance changes neither count.
W counts workers named by declared functions, not the configured worker limit.
A rank that only forwards received sections starts no numerical workers. A failed
worker launch releases references for the remaining unstarted worker mask;
cancellation still visits only unissued records in that index range.
Native completion retires one record. Its instance's final record releases one
program reference. Worker exit cancels its unissued records and releases its W
reference; only that worker reads and changes its records' pending counts.
This keeps callback operands and memory alive without a launch-time atomic retain,
a join in submission, or a caller completion/free protocol.

The implementation trusts caller configuration and backend completion contracts.
Its ownership records describe actual accesses; they do not police arbitrary
external code. No caller free/done call or consumer-stamp protocol is required.

## Publication notifications

ABI 50 stores tensor presence as one atomic 32-bit word per logical row, at a
fixed offset in the shared mapping. Row allocation initializes the word to zero;
the row's producer release-stores one after making the payload visible. This
replaces the packed presence bitmap and its read-modify-write. Constant binding
and dependency realization read the word during setup. Runtime numerical firing
continues through notifications and countdowns; only the explicit
`syncOnRemoteFill` implementation reads presence in a host polling loop. The X9
native stamp binding and resident-consumer demonstration remain unimplemented.

For each numerical worker or TX link, the pending notification set has one bit
per logical row. A row's one producer publishes it once for that value instance;
the reader's prepared range contains all uses of that publication. Repeated
operands in one call appear as repeated uses in the range, not repeated notices.
A constant already present at setup needs no numerical arrival notice. Its first
send binding seeds one notice per TX link before execution.

ABI 48 stores these bits in compact arrays. Level 0 has ceil(rows/64) words;
each next level has ceil(previousWords/64) words, ending at one root word.
Each level and reader bank is aligned to 64 bytes. This needs at most six levels
for the existing 32-bit row namespace. Setup fixes all offsets. Publishing sets
the row bit and then its ancestor bits with one release OR per level, always
continuing to the root. There is no CAS loop, reservation, fullness query or
linked entry. Multiple publishers use different row bits, including when their
bits share a word.

Each reader exchanges the root with zero and enumerates its set bits. Indicated
child words are acquired by the same exchange. Small reader-local arrays retain
the unprocessed bits and indices between calls. Leaf bits yield logical rows for
the existing consumer/send ranges. Empty summaries are skipped; an empty root
returns immediately. No absent row or function is scanned for readiness.

For adjacent levels, the producer sets child before parent. If a reader exchanges
the child after that set, it obtains the publication, unless it already obtained
it in an earlier exchange. If the child set occurs after the exchange, the later
parent set advertises it for a future traversal. This argument applies at every
level. Delaying a producer between the two writes can leave a stale parent bit
after a child was consumed; visiting an empty child is harmless. It cannot erase
a leaf event, duplicate a consumed leaf, or reserve a place that blocks another
publisher. Acquire exchanges make the writes preceding each leaf publication
visible to the reader.

This is a set of row events, not a counter of repeated publications to the same
live row. Reuse must follow the final declared ownership event: every numerical
use must have consumed its notice before completing, and every TX use must have
consumed its notice before posting/completing. Thus a properly recycled row has
no old leaf notice left to coalesce with its next publication. N1/X5 must preserve
that ownership rule; this queue replacement does not implement instance reuse.

For 229,376 arena rows, one link and eight worker positions, the old two-bank
notice heads and per-row links use 16,515,144 bytes before region alignment. The
new padded word arrays use 525,312 bytes. A reader's local iterator is 128 bytes.
These are layout calculations, not latency measurements. The library source
grows from 2,179 to 2,201 lines across all Swift and C/header files; the separate
reusable-instance pool still requires X5/N1 integration.

## Reusable send queue

ABI 50 gives each native TX queue a circular array of send-edge indices. For
E configured edges, its capacity is the smallest power of two at least max(1,E).
The queue uses unsigned 64-bit head/tail positions and a mask for array indexing.
Only the dedicated TX thread appends and removes entries, so these positions need
no atomic operations or producer reservations. An empty queue is a head/tail
equality observation; posting is admitted by the native verbs return value.

At most E different edges can be queued. Each publication contributes its
declared edge once, and that edge's transport reference survives until its final
native completion. ABI 51 removes the head entry after each accepted chunk and appends the same
edge at the tail when more chunks remain. This rotates among ready sections
without changing chunk order within an edge. The final chunk leaves no queued
entry; its completion resets the chunk cursor before releasing ownership. Legal reuse
cannot publish the edge again before that release. Therefore a ring sized for E
needs no fullness guard and cannot overwrite a queued edge. Power-of-two indexing
continues to work when the unsigned positions wrap. This does not select an
invocation slot by modulo: it indexes storage for already-selected queue entries.

The receive cursors, call records and native submissions still have a finite
extent. This ring removes the append-only send representation; it does not by
itself rearm a complete invocation or authorize a second publication of a live row.
N1/N1t still require those remaining changes together.

The library is 2,191 → 2,198 maintained lines for the presence and TX changes.
For 229,376 rows, the presence words use 917,504 bytes, replacing a 28,672-byte
bitmap (raw sizes before region alignment). TX entry storage is four bytes times
the sum of the per-queue capacities. These counts are not performance evidence.

## Invocation identity and storage reuse

The remaining N1 change must distinguish the invocation label k, its local
storage slot s, a logical operand row j, and the physical pages backing that row:

\[
 (k,j) \longmapsto (s(k),j) \longmapsto \operatorname{pages}(s(k),j).
\]

The current source identifies k with s throughout `mesh_call`, the status array,
prepared native objects and transfer targets. It consequently implements a finite
set of single-use invocations. Returning a slot to a free ring alone cannot fix
this: later work needs its own identity while native views continue to select the
chosen storage. Receive matching must carry that identity between peers and keep
it distinct from the physical slot that a particular rank selected.

Reuse follows final declared ownership, with every worker's call countdowns,
receive targets and native submission storage prepared for the new invocation.
It does not follow a caller's numeric index wrapping. A non-submitting rank must
perform the same transition from received work. Native completion callbacks are
not documented as a single writer for a Mesh worker's ring; assigning a function
to a worker does not establish that property. The already implemented TX ring
has one actual writer because its publisher and drainer are the same TX thread.

Result retention is a separate API lifetime: a live invocation's status must be
findable independently of the slot chosen for its operands. The pending retention
clarification concerns completed results after storage reuse, not preservation
of live work. No invocation-directory or result-retention policy has been added
to source by these prerequisites.

## Reclamation events

`82b9b98` (ABI 49) removes the reclamation stack, linked entries, duplicate-enqueue claim,
deferred list and collector thread. A section's final `mesh_buffer_release`
publishes its first logical row in `MESH_FREE`: one atomic OR after the existing
reference decrement. Its descriptor supplies the page count. TX/RX and native
callbacks do not walk the section's backing, clear allocation bits or query readers.

The free bitmap is an unordered pool of section descriptors, with at most one
entry per live section. Setup's sole allocator first uses available arena ranges.
If none fits, it drains the existing free bits once and makes one further range
pass. Draining exchanges each word, enumerates its set bits, and returns each
section's actual canonical backing to `MESH_PAGE_OWN`; `MESH_ROW_HOT` is cleared
last. Payload bytes remain untouched. These finite passes occur during
configuration, never in a numerical invocation or `submit`.

Only the decrement observing one publishes the ordinary free event. A set bit
keeps the logical row hot until its descriptor has been consumed; that row cannot
be reassigned while its event is pending. Concurrent releases of different rows
combine in the atomic word. An OR preceding the exchange is consumed in that
batch; an OR following it remains for a later allocation. No publisher reserves
a position or waits for another publisher. The allocator reads no refcounts.

Program destruction marks its sections closed and advances a retirement epoch.
The bridge processes that event outside active link execution, after joining the
link controllers and their QP teardown. It discharges still-positive abandoned
counts into the same pool. Zero counts already have their ordinary free event,
or have already been consumed. There is no repeatedly deferred buffer and no
device-ownership query on ordinary release. Partial allocation failure drops its
two known setup/producer references directly; it never acquires device ownership.

This is the existing section allocator's event pool, **not** N1's reusable
instance pool. Per-worker SPSC instance rings, the in-flight arena bound, native
launch reuse, transport rearming, and complete R2 failure cancellation remain
required. In particular, recovery of interrupted reference/event publication on
abrupt caller death is not proved by this ordinary-completion protocol.

Maintained library source is 2,201 → 2,191 lines across `swift/*.swift` and
`rdma/*.{c,h}`. The extra bitmap costs ceil(rows/64) × 8 bytes: 28,672 bytes for
229,376 rows. Removing `next` does not shrink the 32-byte buffer record because of
alignment. These are source/storage counts, not latency measurements.

## Block addressing

### Logical order and interleaved arrivals

The page table is the indirection between logical tensor coordinates and physical
registered backing. A receive relation identifies `(peer, edge, instance, chunk)`;
its completion installs the actual backing page in the corresponding logical row.
The peer qualification is realized in each link/queue's transfer table. Identical
source row integers arriving from two neighbors of a ring do not alias one another.

For a logical value X with chunks X0, X1 and X2, a mapping `[p7, p2, p8]` means
those chunks occupy those pages in that logical order. The order in which their
completions were drained is irrelevant to X's coordinates. Indexed consumers use
the mapping; forwarding reads the same registered backing. Each value's declared
ownership keeps those pages alive. No acknowledgement or global arrival order is
needed to establish this relation.

A page list does not itself make a base-pointer/length operand contiguous.
Contiguous-typed operations consume appropriate contiguous sections, a supported
contiguous alias of the same backing, or an asynchronously produced layout in
canonical operand storage. Reordering metadata or installing aliases moves no
payload. An actual scatter/gather materialization moves bytes and must be counted
as such. The layout operation publishes its own outputs; only their consumers
depend on that publication. RX and TX continue draining other work throughout.

ABI 51 permits chunk interleaving within each queue as well as independent arrival
from multiple peers. Each accepted chunk rotates its unfinished send to the tail.
A logical section therefore need not occupy consecutive physical receive blocks.
No QP per value, whole-section posting order, or receive-side reorder wait is used.
Multi-link striping of a single section remains T3 work; invocation reuse remains N1.

### Indexed receive runs and contiguous consumers

Each numerical partial has a logical head s, a byte length N, and
K = ceil(N / C) transport chunks, where C is Mesh's internal chunk capacity.
The page table contains K chunk addresses. Ownership and numerical presence
belong to s, independently of K. Value i uses `first + i * stride`; a shared
constant has stride zero. Numerical indexing continues to use its declared
shape and strides, without a transport-chunk dimension.

Local operands occupy contiguous payload pages. Setup allocates one receive run
per queue covering its finite declared transfers. These are writable posted
blocks, not a promise that consecutive positions belong to the same tensor.
A single-chunk input selects its native view directly from its received page.
Forwarding follows each logical chunk's page entry without copying it.

For a multi-chunk received input of a contiguous function, setup allocates a
contiguous canonical section per consumer invocation. Repeated uses of the same
input within that function share this section. These are actual registered arena
pages with ordinary descriptors and ownership, not a separate operand store.
C binding now distinguishes dependency inputs from their prepared operand views:
the original received rows fire the function and retain the source; the supplied
function sees the contiguous section. The placement section is an auxiliary output
owned through that invocation's native completion. It has no independent consumer,
extra publication dependency, runtime allocation, or caller-visible parameter.

The placement and supplied operation form one launch. Metal records indexed blit
copies before the supplied encoder in the same command buffer. Native views are
created at setup through the existing shared-buffer cache. CPU and Core ML paths
perform indexed `memcpy` on their numerical worker before the supplied function or
prediction. They add no GPU-to-host completion round trip and execute no layout
work on an RDMA thread. Unaffected inputs retain their existing direct bindings.
The backend choice, chunk rows, byte extents and native views are all realized
before invocation. No runtime predicate asks whether an entire receive happened
to land contiguously.

This materialization copies N bytes per distinct multi-chunk input per consumer
invocation (Metal rounds the final copy to four-byte alignment inside allocated
padding). It is not zero-copy. Its temporary arena cost is
`count * ceil(N/C) * C` for that binding, including a received shared constant;
there is no implicit shared-copy completion guard. Transport placement and
forwarding remain zero-copy. CPU bandwidth, GPU copy cost and this additional
storage must be included in subsequent performance and reusable-arena work.
The library grows from 2,198 to 2,272 maintained Swift/C/header lines (+74);
the ring configuration is 12 lines of caller data. No source generator or new
numerical implementation is introduced. Documentation changes are separate.

Before publication, receive blocks form a bijection between unfilled logical
destinations and reserved physical blocks. If the next completion fills physical
block p for destination s, let d be p's current logical owner and q = p(s).
Assignment exchanges `(s,q), (d,p)` for `(s,p), (d,q)`. Two forward-table stores
and two inverse-table stores maintain that bijection. The identity case d = s
uses the same writes. Every physical receive slot appears once in the finite
posting list, so a previously published block cannot be displaced by a later
completion. No payload is copied and no published operand changes address.

For example, let A and B each have three chunks and receive positions p0–p5:

| Completion | Updated destination | Numerical publication |
|---|---|---|
| A0 into p0 | A0 → p0 | — |
| B0 into p1 | B0 → p1 | — |
| A1 into p2 | A1 → p2 | — |
| B1 into p3 | B1 → p3 | — |
| A2 into p4 | A2 → p4 | A |
| B2 into p5 | B2 → p5 | B |

A's placement reads `[p0,p2,p4]` and B's reads `[p1,p3,p5]`. A's consumer does
not depend on B2. At another peer the same source row integers index a different
link's transfer relation and disjoint receive backing. This is the same mapping
for both directions of a ring and for an intermediate node forwarding a value.
The existing [Gram chain ring configuration](../examples/gram-chain-ring.json)
uses that path through ordinary gathers, supplied functions and reduce-scatters.
It is source usage, not a measured four-node run.

Each chunk completion performs that assignment independently, updating one
forward entry and preparing the chunk's local source tag for forwarding. The
last chunk's precomputed target also names the numerical head to publish.
FIFO completion puts that publication after all the partial's bytes are placed.
It does not publish a different tensor partition or wait for any other partial.
Numerical consumers use the direct or materialized operand bindings described above. Collection releases each chunk's actual backing through
the page table, including the displaced mappings of unfinished receives.

The registered transport address space aliases these payload pages and a
separate tag page before each chunk. Its one-entry SEND/RECV span starts at
that page's final four bytes and continues into the payload; the dense numerical address space excludes
tag pages. Both address spaces map the same shared-memory payload, not two
copies. All aliases and registrations are made before execution. The tag page
costs one OS page of storage per chunk; framing costs are recorded in
[the execution description](async-collectives.md#execution-and-ownership).

Each value has exactly one producer: setup for a constant, one numerical call,
or one receive. Its initial reference represents that write. `mesh_publish`
releases this reference once after publishing the declared uses. There is no
producer flag or duplicate-publication check. Transient numerical and transport
references were retained during realization and end through their own native completions;
shared numerical references end with their prepared function's lifetime;
external handles have ordinary automatic lifetimes. This preserves
`R = P + C + T + E` while removing a redundant atomic producer-flag operation.
The final reference publishes the section to the [free pool](#reclamation-events).
