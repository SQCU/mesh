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

For section s, outstanding ownership is R(s) = P(s) + C(s) + T(s) + E(s): unfinished
production, configured numerical owners, transport uses, and external observations.
Setup derives known uses from the feed-forward graph. Existing completions and
automatic object lifetimes discharge them. Early publication does not release an
unfinished producer. A background collector returns zero-reference backing to the
writable pool without clearing payload bytes. Independent work already has its
configured sections and never awaits that collection.

For a transient input, each indexed use is a numerical owner and its native
completion releases that reference. For an immutable shared input, the prepared
function holds one reference per binding across all its value indices. Destruction
of that function follows completion or cancellation of all its call records and
releases the shared reference. Thus the constant remains live throughout every
native read without per-invocation reference updates. This groups equal storage
lifetimes; it does not remove input-arrival dependencies or alter tensor values.

For the call storage itself, let O be the Mesh owner reference, W the active
worker references, U the references reserved for unissued records, and A the
references for issued calls whose native callbacks have not finished. Its count
is O+W+U+A. Worker startup acquires its contribution to W+U in one update.
Issuing a call transfers one reference from U to A without changing the count.
Native completion removes one from A. Worker exit removes its remaining U and
its W reference; only that worker reads and changes its records' pending counts.
This keeps callback operands and memory alive without a launch-time atomic retain,
a join in submission, or a caller completion/free protocol.

The implementation trusts caller configuration and backend completion contracts.
Its ownership records describe actual accesses; they do not police arbitrary
external code. No caller free/done call or consumer-stamp protocol is required.

## Block addressing

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
