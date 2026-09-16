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
unfinished producer. A background collector returns zero-reference backing to the
writable pool without clearing payload bytes. Independent work already has its
configured sections and never awaits that collection.

`TensorPart` contains rank, byte extent, a C section descriptor and value flags;
it has no heap-object reference or destructor. Copying or retaining the descriptor
does not add a reader or delay reclamation. `MeshMemory.sections` holds S during
configuration. `Mesh.start` releases it after binding all functions and transfers,
before starting numerical workers; failed configuration returns any setup
references still held by that array. The array is then empty. Backend views keep
the memory mapping alive; their actual reads are owned by the declared calls.

All retains occur before S is dropped, when R is necessarily positive. They need
one increment, with no resurrection check, retry or rollback. After setup, the
count only decreases. Its transition from one to zero is sufficient to enqueue
reclamation; there is no separate sealed state. Forced teardown still closes
buffers and preserves pages while the bridge owns their queue pairs. The current
reclamation queue still has a duplicate-enqueue check and CAS retry; replacing
that queue with the X5 free-list event remains required.

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
reclamation stack still requires X5 replacement.

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

The implementation below already handles interleaving between peer queues and
out-of-order complete sends within a queue. Its consecutive-chunk assumption is
a current implementation limit, not a requirement on tensor functions or the
page-table abstraction. General chunk interleaving and multi-link striping still
need the X3 layout path; no native queue per logical value is required.

### Current contiguous receive runs

Each numerical partial has a logical head s, a byte length N, and
K = ceil(N / C) transport chunks, where C is Mesh's internal chunk capacity.
The page table contains K chunk addresses. Ownership and numerical presence
belong to s, independently of K. Value i uses `first + i * stride`; a shared
constant has stride zero. Numerical indexing continues to use its declared
shape and strides, without a transport-chunk dimension.

Local operands occupy contiguous payload pages. Setup allocates one contiguous
receive page run per queue, covering the sum of its declared transfers' chunk
counts. TX posts each send's chunks consecutively in that queue. Receives consume
the posted run in that same order, so every send lands in a contiguous subrange,
even when differently sized sends publish out of declaration order. If its first
page is p(s), byte offset d has address `arena + page_size * p(s) + d`.
Native bindings are prepared for the possible starts at which the whole operand
fits. There is no remapping of a live numerical view.

Before publication, receive blocks form a bijection between unfilled logical
destinations and reserved physical blocks. If the next completion fills physical
block p for destination s, let d be p's current logical owner and q = p(s).
Assignment exchanges `(s,q), (d,p)` for `(s,p), (d,q)`. Two forward-table stores
and two inverse-table stores maintain that bijection. The identity case d = s
uses the same writes. Every physical receive slot appears once in the finite
posting list, so a previously published block cannot be displaced by a later
completion. No payload is copied and no published operand changes address.

For example, let A occupy three chunks and B one chunk. Their receive run
contains physical positions p0, p1, p2, p3. If B publishes first, the completions
produce this assignment:

| Completion | Updated destination | Numerical publication |
|---|---|---|
| B0 into p0 | B0 → p0 | B |
| A0 into p1 | A0 → p1 | — |
| A1 into p2 | A1 → p2 | — |
| A2 into p3 | A2 → p3 | A |

B's consumer can run while A is transferring. A's native operand starts at p1
and spans p1–p3 contiguously. Its setup bindings need only the possible fitting
starts p0 and p1; B's bindings cover p0–p3. Neither function sees the transport
chunk count. B's published p0 is never displaced while A's rows are assigned.

Each chunk completion performs that assignment independently, updating one
forward entry and preparing the chunk's local source tag for forwarding. The
last chunk's precomputed target also names the numerical head to publish.
FIFO completion puts that publication after all the partial's bytes are placed.
It does not publish a different tensor partition or wait for any other partial.
Numerical consumers and their prepared Metal/MPS or Core ML bindings resolve
the numerical head. Collection releases each chunk's actual backing through
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
The writable pool is still returned by the existing background collector.
