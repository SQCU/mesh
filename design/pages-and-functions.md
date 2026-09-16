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
count only decreases for that value. Its transition from one to zero publishes
one indexed return event; there is no sealed state, queue claim or retry. ABI 53
returns receive-owned rows and backing to the RX thread, which reinstalls the
declared reference count when it assigns the row to a new value. Local sections
still return to the setup allocator. Forced teardown
preserves unfinished device ownership until the bridge closes the queue pairs.
The [reclamation events](#reclamation-events) below replace the former collector.

For a transient input, each indexed use is a numerical owner and its native
completion releases that reference. For an immutable shared input, the prepared
function holds one reference per binding across all its value indices. Program
destruction follows completion or cancellation of all its call records. ABI 52
leaves shared and unissued operand references to the existing client-retirement
event, after native callbacks end and the bridge closes the queue pairs. It removes
the destructor's reconstruction of consumed rows from consumer slot numbers.
Thus the constant remains live throughout every
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
the row's producer release-stores `invocation + 1` after making the payload visible
(ABI 53; ABI 50 used the constant one). Shared constants retain stamp one. This
replaces the packed presence bitmap and its read-modify-write. Constant binding
and dependency realization read the word during setup. Runtime numerical firing
continues through notifications and countdowns; only the explicit
`syncOnRemoteFill` implementation reads presence in a host polling loop. The X9
native stamp binding and resident-consumer demonstration remain unimplemented.
The explicit blocking call searches the section's storage slots for the requested
invocation stamp; a slot number no longer identifies an invocation. This search
exists only in the opt-in synchronization path. It does not retain a value beyond
its declared readers, and can wait forever if asked for an unavailable value.

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

ABI 53 completes the companion receive-storage cycle below. Call records,
instance statuses and native submissions still have a finite extent. These rings
do not by themselves rearm a complete invocation or authorize a second
publication of a live row. N1 still requires those remaining changes together.

The library is 2,191 → 2,198 maintained lines for the presence and TX changes.
For 229,376 rows, the presence words use 917,504 bytes, replacing a 28,672-byte
bitmap (raw sizes before region alignment). TX entry storage is four bytes times
the sum of the per-queue capacities. These counts are not performance evidence.

## Receive storage return

ABI 59 removes the per-binding logical-row stacks, the definition-to-binding
lookup, and the active-source-head array. Setup expands each peer-qualified
source chunk into an aligned 32-byte receive record. ABI 60 stores the exact
destination row, its buffer address and canonical page-entry address, the declared
reference count and first/final/shared flags. RX indexes that record using the
source chunk in the received tag. It does not allocate, search for a row or
reconstruct the page-list address from a buffer descriptor.

For transfer t, resident frame f and chunk k, setup binds
`sourceFirst[t] + f * sourceStride[t] + k` to destination row
`localFirst[t] + f * localStride[t]` and its k-th canonical page entry. The
source chunk already includes f; receiving a sequence number does not require
another modulo calculation to recover it. The first chunk installs the declared
references and sequence, each chunk stores its actual received page, and the
final chunk publishes. The section's physical
pages may be interleaved with other sections or peers. The formula determines
logical coordinates; it does not make the physical pages contiguous.

A queue still owns one ring of physical receive blocks. Final-reference notices
return all of a section's actual blocks to that ring, clear its mappings, and
release the transfer's frame reference directly. RX posts available blocks until native
refusal, without a software window or a destination-reuse check. The former
logical-row stack return and first-chunk pop are deleted. Payload is not zeroed.

This is the frame-indexed representation required by N1, not yet its complete
allocation proof. In particular, modulo arithmetic cannot establish that a
previous use of the same logical frame has ended on another participant. Setup
still must establish the disjoint live intervals of the declared plan, including
receive mappings and native operands. Local SEND
completion is not remote final use. Adding an RX occupancy guard or a credit
message would not complete this requirement. Until that realization is complete,
N1 and unrestricted cross-participant frame reuse remain unfinished.

## Native slot return

Every function now has one prepared call and operand array per frame. Its
row-indexed consumer records contain the function pointer, input position,
shared-input flag and frame. A dynamic publication fills that frame's input and
decrements its pending count; zero invokes the supplied function using the same
frame's prepared output/native bindings. No hash probe, join record acquisition,
backshift deletion, or per-function native-slot ring remains.

Shared inputs update every prepared frame once and decrement each countdown.
Setup initializes all countdowns, including the root dependency for a function
with no varying inputs. Successful native completion publishes outputs, releases
that call's consumed input references, then releases its outputs' producer references.
Their final references return through the existing numerical-worker notices;
the last output return rearms the native object
when necessary, restores that frame's pending template, and releases one frame reference directly. Zero-output functions emit the same return through their metadata row.
A native error concludes status separately; complete failure cancellation is R2.

### Prepared numerical uses

The row's consumer range contains 32-byte records with direct call and optional
input-operand and canonical first-page-entry addresses. Setup records the
input's source row separately from
its view storage. This source row is also used for placement reads and reference
release when an input was already present at setup. The index and page-list
pointer are fixed then; they are not reconstructed from the function on arrival.

Local storage and prepared contiguous views keep their bindings. An unplaced
remote operand has a prepared refresh target: arrival loads its canonical first
page and sets the address. The same record points directly to the call whose
pending count is decremented. Shared-input arrival applies that binding to every
resident call and updates the reusable pending template once. No function/frame
lookup precedes the common varying-input pending decrement; the function is
read when that decrement reaches zero and launches the supplied numerical work.
Call records are allocated at setup with 64-byte size and alignment, asserted
in C; the old 40-byte array stride could split a call across cache lines. This
uses 24 additional bytes per resident call. The canonical page load for a remote
operand and the notification hierarchy remain explicit unfinished H work.

### Input lifetime ends at its own use

For a declared chain `receive(x) → f(x)=y → g(y)=z`, the numerical reference
to `x` ends at `f`'s native completion. The former implementation retained `x`
until the final reader of `y` returned it, extending ownership into downstream
execution. That extension was unnecessary: `g` reads `y`, and has its own
reference to `y`; it does not read `x`. The same argument applies to fan-out:
every direct numerical reader releases its own reference at its completion,
and every transport reader releases its reference at native SEND completion.
The last of those events returns the backing. No caller reports that it is done.

The consumed-input index list and initial reference counts are already compiled
from bindings. CPU, Metal and Core ML use the same completion entry point,
including native failure. Output publication comes first; input reference
decrements follow. Publication no longer drops the output's producer reference.
Within publication, TX notices precede the presence store and local-use mask
load/notifications. Received sections use the same ordering for onward sends.
For each output y, that reference keeps R(y) ≥ 1 while completion reads its input
metadata, even if every consumer finishes immediately. After input cleanup,
completion releases each producer reference. Therefore final output return
implies that the callback has finished reading its operands; the numerical worker
can rearm its native slot. No additional reference, notice, queue, allocation,
reader query or wait is required. Zero-output and failed calls retain their
existing terminal return row.

The C completion entry point takes the native error directly; the success-only
forwarding wrapper is deleted and all three Swift backend bindings supply zero
on success. The existing failure entry point normalizes a zero error code before
calling that implementation. The public Swift interface is unchanged.

Native output storage and command-buffer rearming retain their separate existing
lifetimes. This local lifetime fact does not establish cross-participant frame
reuse; that remains N1.

The numerical worker drains ready publications before processing output/terminal
return notices. Cleanup and command-buffer rearming therefore follow launches
already represented in its publication queue, rather than preceding them.

RX now drains return notices within `link_receive` itself. It posts available
pages first, consumes a returned section, clears its old page entries, appends
the pages to the receive ring, and immediately repeats posting. Native capacity
refusal leaves the unposted pages in that ring but does not prevent draining
other returned sections. An empty return set ends the pass; a fatal native error
concludes the link. No tensor-readiness or destination-occupancy condition is
introduced. This removes the separate one-return-per-pass `link_returns` helper.
An RX completion is published before this return/repost pass: draining accumulated
returns must not delay delivery of a completion already polled. TX retains its
post-before-completion-cleanup order. Full hot/cold separation, native request
preparation and the notice-ring replacement remain group H work; no H row is
completed by this change.

These changes reduce maintained Swift/C/header source from 2,415 to 2,413 lines.
They add no runtime storage or completion notices. Documentation changes are
explanatory additions, not source migration.

Core ML's feature provider and output options use the frame's prepared bindings.
Indexed CPU/Metal operands keep the canonical page-list address. A contiguous
backend still uses the already implemented asynchronous placement path when its
received physical pages are not contiguous. No copy is removed or called
zero-copy merely because receive rows have a compiled logical order.

Metal command-buffer replacement remains row 19s: the current rearm callback
creates a new single-use command buffer on the numerical worker. It is not solved
by these frame-array changes, and completed command buffers are not recommitted.

## Invocation identity and storage reuse

The ABI-59 runtime sequence is 32 bits. The public integer is converted to that
sequence; the resident frame is `sequence % inFlight`. Every function's call,
operands, countdown and native index use the same frame. The sequence in
`mesh_operand.invocation` remains available to the supplied function. The
section row determines its canonical page list. Different producer completion
orders therefore need no assignment-order matching table.

Frame ownership is an array indexed by frame. Setup counts numerical and
varying-transfer references per frame, plus the initial shared-transfer references.
A function's final output return or transfer completion/storage return directly
decrements that frame's atomic count. Zero restores its recurring count,
concludes status, and marks it available. Shared-transfer completion releases one
initial reference from every frame. The last release is not enqueued elsewhere.
ABI 60's send record contains the exact frame-refcount address; received sections
have their frame index installed in `buffer.binding` during allocation. Transfer
return therefore does not derive frame ownership from the invocation label.

The lifecycle thread, producer event rings and their arena storage are deleted,
along with the keyed status directory, tombstones, frame-to-label array and
submission-association event. This also removes the separate event-ring capacity
obligation. `result(index)` loads that resident frame's status once; it is not a
historical result dictionary. Completed result values can be retained by the caller.

Submission currently reads the selected frame's availability word, immediately
returns busy if it is unavailable, and otherwise resets status and publishes the
roots. It does not allocate, wait, query readers or touch a receive queue. This
is **not** the requested free-frame-ring admission: it can return busy while a
different frame is free. The missing admission mechanism and whole-plan lifetime
realization remain N1 work. ABI 61 addresses passive result identity below;
cancellation remains unfinished, so N2 is still partial.

These edits reduce maintained Swift/C/header source from 2,612 to 2,415 lines,
including `Bounds.swift` and `Topology.swift` and excluding the separately pending
control-event work. Documentation replacement is separate. The source compiles
with existing callers; no runtime, deployment, safe-unbounded-reuse or latency
claim accompanies this intermediate replacement. ABI 60's prepared transport
addresses preserve that source count. Send records grow from 20 to 32 bytes
per resident send edge and use a 32-byte-aligned allocation; receive records
remain 32 bytes. The ABI changes because the receive buffer's `binding` field
now supplies its setup-assigned frame index to the bridge.

### Versioned frame results

For invocation i and resident capacity V, result reads the status word at i mod V.
Each new traversal uses a new 32-bit invocation index, as in `submit(inFlight + k)`;
the index is not a reusable frame handle. A dynamic success stores i in its
otherwise unused 32-bit code field. A success
for j != i decodes as busy, so an earlier traversal's success cannot complete a
new traversal on a passive rank. A static success (ordinal one) covers all indices
when the program has no recurring local work. Errors retain their native code
and function/link identity. The lookup remains one atomic 64-bit acquire load.

Submission supplies the frame's invocation before publishing roots. For received
work, the prepared receive record supplies the frame index. After publishing the
completed section, RX stores the invocation and then releases the section's
producer reference. That reference prevents the section's final return, and
hence its frame-reference decrement, from preceding the store. The final frame
reference publishes that invocation's success. Frame and receive records both
remain 32 bytes; their spare space holds the new fields. ABI 61 is required on
both sides because result encoding and the frame layout change.

Conclusion loads the old word and makes one strong compare-exchange when it is
success or pending. It never retries or replaces an error. For a native failure
and a final success racing on the same observed word, only one wins; success
cannot clear an already published error. A new traversal can replace an earlier
success without first writing busy at receipt. Link failure invalidates resident
slot results, including earlier successes not retained by the caller. Stored
`Result` values are unaffected. These are reusable slots, not a result archive.

The Core ML chain's existing driver keeps submitted and completed indices within
its configured window and calls `result` on both submitting and passive ranks.
Numerical callbacks still only produce/consume their tensor operands. The library
adds no synchronization call or transport permission check. This fixes stale
successes; it does not prove disjoint cross-participant frame lifetimes or close
N1 and failure cancellation.

### Prepared native requests

The exact-tail SEND change in ABI 60 was incorrect: a shortened SEND may use
fewer native frames than its preposted RECV. Apple
[TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
requires matching frame counts. The tail-length field and runtime length choice
are removed. Setup chooses the matching request extents, and posting submits
those extents unchanged. No truncation occurs in SEND or RECV.

For transport payload capacity C and a queue's declared logical transfer lengths
N_t, setup chooses L_q = min(C, max_t N_t) + 8 bytes for both ends of that
direction. A section of N bytes occupies K = ceil(N/C) requests, each of L_q
bytes. The posted SGE byte total is K L_q: N logical bytes, 8K tag bytes, and
K(L_q - 8) - N padding bytes. The receiver's prepared request has the same
extent for every arrival, independent of which producer finishes first. This
corrects the former N + 8K claim; it is API byte accounting, not measured wire
time. Reducing padding would require selecting compatible requests before
posting, without imposing producer order or changing the caller's tensor.

Each physical receive block has a 64-byte aligned native WR/SGE record. The RX
pool ring carries indices into this array; returning a page computes its index
once, and reposting passes the prepared WR directly. Each resident send edge
has a 256-byte record aligned to 128 bytes, containing its native WR, SGE and
existing ownership/cursor fields. Its common SEND header, SGE and application
fields fit the first 128 bytes; the full native SEND WR is itself 128 bytes.
TX still selects the backing address/key from the canonical page table, but
constructs no WR and changes no extent at posting. All signalled chunk
completions identify the edge; its thread-local countdown releases the buffer
and frame references after the final completion. WR/SGE metadata is reusable
after `ibv_post_*` returns; payload backing stays owned through completion.

Receive requests add 64 bytes per pool block; send records grow from 32 to 256
bytes per resident edge. There is no shared-memory ABI change. This correction
adds three maintained source lines to the 2,413-line library, excluding the
separate unfinished control-event work. Shapes, partial publication and consumer
layouts remain unchanged; transport fragmentation and numerical partials remain
separate. The remaining TX page lookup, wire tag and RX mapping are still H work.

## Reclamation events

`82b9b98` (ABI 49) removes the reclamation stack, linked entries, duplicate-enqueue claim,
deferred list and collector thread. A section's final `mesh_buffer_release`
publishes its first logical row in `MESH_FREE`: one atomic OR after the existing
reference decrement. Its descriptor supplies the page count. Publication and
native callbacks do not walk backing or query readers. ABI 53's RX return handler
performs the indexed backing walk required to repost its freed blocks.

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
counts into the same pool. ABI 53 also returns zero-count receive-pool rows here;
their posted backing stayed owned by RX throughout active execution. Local zero
counts already have their ordinary free event, or have been consumed. There is no repeatedly deferred buffer and no
device-ownership query on ordinary release. Partial allocation failure drops its
two known setup/producer references directly; it never acquires device ownership.

This is the existing section allocator's event pool, **not** N1's reusable
instance pool. Per-worker SPSC instance rings, the in-flight arena bound,
unbounded invocation matching, and complete R2 failure cancellation remain
required. In particular, recovery of interrupted reference/event publication on
abrupt caller death is not proved by this ordinary-completion protocol.

Maintained library source is 2,201 → 2,191 lines across `swift/*.swift` and
`rdma/*.{c,h}`. The extra bitmap costs ceil(rows/64) × 8 bytes: 28,672 bytes for
229,376 rows. Removing `next` does not shrink the 32-byte buffer record because of
alignment. These are source/storage counts, not latency measurements.

## Block addressing

### Prepared compact page lists

ABI 58 separates a buffer descriptor's identity, metadata extent, payload extent
and page-list address. For an N-byte value, page size P and transport capacity
C = BP, setup computes K = ceil(N/C). Each resident slot owns K logical entries
and KB physical pages. Its immutable `mapping` offset names those K entries in
canonical shared memory; its `rows` count governs metadata reclamation, while
`pages` governs backing reclamation. Metadata ownership was previously VKB rows
for V slots; it is now VK. The bridge still reserves its existing metadata arena,
so this is not a claim that process RSS falls by B.

TX and RX address entry j through that prepared list. An indexed operand caches
the list pointer; a byte offset b uses entry floor(b/C) and relative offset b mod C.
The contiguous CPU/Core ML and Metal placement paths read the dependency's same
list. No allocator, packing solver, mapping syscall or strategy selection runs on
receipt or scalar consumption. Retain/release addresses one buffer directly;
setup and destruction enumerate section slots explicitly. The reference-count
range walks and their assumption that payload extent determines descriptor
positions are removed.

ABI 59 removes the section-definition field and narrows the runtime sequence;
the buffer descriptor and operand are each 48 bytes. Compact page-list addressing
still adds a descriptor dependency on TX/RX; X10 must finish the event records.

### Explicit section identity on the wire

ABI 59 carries one atomic 64-bit word `(sequence << 32) | sourceChunkRow`.
`sourceChunkRow = sourceHead + chunk` uses the compact logical row namespace,
not OS-page spacing. Every sending link for that backing writes the same tag.
The tag and payload share one SEND request. Pairing rejects another wire ABI
before posting data.

At setup, each receiving queue prepares a source-chunk-indexed table. Each entry
is 32 bytes and aligned to 32 bytes; compilation asserts both properties. Source
rows are qualified by their peer/queue, so equal row integers on two peers do not
alias. RX loads one entry containing the destination row, buffer address and exact
canonical page-entry address. ABI 60 moves that address arithmetic to setup.
No definition lookup, binding pointer, active-head array or free-row pop intervenes.
The table uses 32 times the peer's advertised row count per receiving queue.
This is a setup-memory cost; the eight-byte wire tag replaces 24 bytes.

The compiled row is independent of which registered page received that chunk.
RX associates the actual page through the canonical list. Arbitrary interleaving
is preserved; physical contiguity and N1's reuse proof do not follow from this
metadata reduction.

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

Contiguity is requested by `TensorFunction(inputViews:outputViews:)` and
`TensorFunction.prediction`. Raw CPU/Metal functions do not allocate these
placement sections. A raw host function reads logical scalar indices with
`operand.load(at:as:)`; a native view factory receives its requested contiguous
`MeshSpan`. For an indexed operand, the prepared descriptor holds the canonical
page-list pointer and geometry, and the actual input binding selects that list
alongside the first-page pointer. A placed operand instead retains the mapping
of its placement section. Thus the original dependency row and the presented
operand mapping remain distinct without reconstructing either on each read.
The scalar address calculation is one mapping load and index arithmetic; it has
no readiness read or loop. It is host indexing, not an implementation of X9's
resident GPU path. Aligned scalar reads use the same indices at every internal
transport extent. The helper does not promise contiguity for a whole tensor.

The placement and supplied operation form one launch. Metal records indexed blit
copies before the supplied encoder in the same command buffer. Native views are
created at setup through the existing shared-buffer cache. CPU and Core ML paths
perform indexed `memcpy` on their numerical worker before the supplied function or
prediction. They add no GPU-to-host completion round trip and execute no layout
work on an RDMA thread. Unaffected inputs retain their existing direct bindings.
The backend choice, relative chunk offsets, byte extents and native views are all realized
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

ABI 53 keeps unfilled logical rows unmapped. A completion assigns its actual
physical page directly to the selected row and chunk offset. Return removes that
mapping before reposting the page; the logical head returns to its binding only
after all its chunks are detached. There is no inverse-table permutation and no
payload copy. The [return proof](#receive-storage-return) covers the physical
page-ring bound and preservation of live operands; it identifies the separate
unresolved logical-row bound under repeated admissions.

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

Each chunk completion performs that assignment independently. TX prepares the
forwarded chunk's local head, section definition, invocation and ordinal when it
posts that chunk. The binding's chunk count identifies the publication boundary.
FIFO completion puts that publication after all the partial's bytes are placed.
It does not publish a different tensor partition or wait for any other partial.
Numerical consumers use the direct or materialized operand bindings described above. Collection releases each chunk's actual backing through
the page table.

The registered transport address space aliases these payload pages and a
separate tag page before each chunk. Its one-entry SEND/RECV span starts at
that page's final 24 bytes and continues into the payload; the dense numerical address space excludes
tag pages. Both address spaces map the same shared-memory payload, not two
copies. All aliases and registrations are made before execution. The tag page
costs one OS page of storage per chunk; framing costs are recorded in
[the execution description](async-collectives.md#execution-and-ownership).

Each value has exactly one producer: setup for a constant, one numerical call,
or one receive. Its initial reference represents that write. `mesh_publish`
releases this reference once after publishing the declared uses. There is no
producer flag or duplicate-publication check. Transient numerical and transport
references were retained during realization and end through their own native completions;
shared numerical references end with client retirement after native users finish;
external handles have ordinary automatic lifetimes. This preserves
`R = P + C + T + E` while removing a redundant atomic producer-flag operation.
The final reference publishes the section to the [free pool](#reclamation-events).
