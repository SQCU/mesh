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

ABI 53 connects the final reference of a received value directly to its RX
thread. The buffer head names its receive channel and binding. Refzero publishes
one row bit to that queue's return-notice bank; local sections continue to use the
general free bitmap. Callbacks neither inspect readers nor walk payload backing.
Only that link's RX thread consumes the return banks. Concurrent native and TX
publishers use the existing atomic notification mechanism.

Each queue has a fixed physical-page ring, initially containing all its registered
receive backing. Each binding has a stack of its logical section rows. RX alone
changes these structures. Unfilled rows have no page mapping. The receive pool
owns the posted pages independently of those rows; its canonical descriptor
records the physical extent and client owner. `mesh_receive_range` reads that
descriptor through the transfer's pool index, replacing the private range array.

Every queue progress step polls its CQ, consumes one available return notice,
and attempts one available RECV before handling the completion. The working ABI 55
return path appends all K backing indices of that returned section to the physical
ring and clears their mappings. It returns the logical row before posting from
that ring. This removes the partially detached return queue: replenishing only
part of a section could admit another value before its logical row returned.
Reclamation costs K metadata loads and stores in one RX pass, without payload
copying. This cost grows with section size; it is not a constant-time latency
claim. Native refusal leaves the page queued; no software outstanding-frame limit
or retry wait exists.

After its final chunk is detached, the logical row returns to its binding's
stack. The first chunk of a new value pops a row, installs the binding's realized
reference count, clears its old presence and sets its invocation. It neither
queries occupancy nor waits. Completion places each received page with one
indexed store; later chunks use the same head and prepared relative offsets.
The inverse page table, placeholder assignments and four-store permutation are
removed. A returned row is already unmapped, and a posted page has no live reader,
so there is no displaced mapping to repair or payload to move.

Protocol 57 replaces the source-chunk occurrence cursors with explicit definition,
source-head and chunk fields. The first chunk records the actual local head for
that source head; later chunks directly select the same head and their offset.
SEND ordering and final ownership keep a sender slot's next value behind its
previous value on that queue. The final chunk publishes the selected head.
This also serves forwarding: a received row can carry k=0, be forwarded, return
at refzero, and carry k=1. The existing send edge resets, and downstream matching
accepts the same source row with the new invocation. No extra local submit,
acknowledgement, epoch barrier or inferred collective is involved.

For P physical blocks, every block is in one of: the post ring, a posted
receive/completion, a live value, or a pending return. A block moves to the ring
only after its value's final reference. Hence queued blocks never exceed P.
Power-of-two capacity covers this bound without fullness checks. A live reader's
reference prevents its physical page from being reposted. There is no return-cursor
allocation or queue in the working ABI 55 implementation.

For a binding with V configured rows and L assigned values whose rows have not
returned, its free-row count is V-L. The original finite API admitted at most V
values in total, which established this bound. ABI 55 permits more than V labels
over time, so that historical argument does not establish its receive capacity.
Physical backing and logical rows currently have different allocation domains:
backing is pooled per queue, while rows are reserved per binding. Other bindings'
unused backing can receive another value even when this binding has L=V. The
unchecked first-chunk row pop then has no valid row. Returning a complete section
before reposting its pages does not resolve this cross-binding case.

This is N1's unfinished allocation contract, independent of arrival order or peer
count. Realization must provide logical descriptors and backing for the declared
live dataflow together. Local SEND completion alone does not establish remote
last use. A receive-side occupancy check, acknowledgement or wait would withhold
work without repairing the allocation model and is not the proposed remedy.

Numerical consumer ranges are now indexed by the buffer's immutable section
`definition`, rather than duplicated for each resident row. All root rows name
the same root definition; all slots of a tensor section name its section
definition. A publication still supplies its actual row and invocation, so input
binding, page mapping and ownership continue to refer to the actual value. This
is the prerequisite for assigning a received descriptor independently of its
section's old private row range. The current RX allocator and TX forwarding
arrays have not yet made that transition, and no receive-capacity completion is
claimed from this dependency-table change.

Setup captures the binding's declared reference count and removes those counts
from its initially unused rows. The pool owns their storage while no value occupies
them. Assignment adds the count for a new value. Atomic add/subtract preserves
the client-close flag if teardown overlaps. Posted backing never enters the
ordinary allocator while a QP may name it. After QP teardown, client retirement
clears received row mappings and returns their logical slots, then releases the
inactive client's physical pool extents. New-client pool ownership is preserved.
Payload is never zeroed.

Payload allocation is unchanged. Buffer heads grow from 32 to 40 bytes for the
channel and binding. Each receive queue adds a return-notice bank per client bank,
a four-byte physical-ring entry per rounded-up block capacity, and an eight-byte
return cursor per rounded-up section capacity. Logical free rows use four bytes,
source values eight, receive bindings 24, and the additional source-row offsets
four each. Targets remain 12 bytes. Sixteen-byte pool descriptors per arena block
replace the inverse table's four bytes per OS page. The private receive-range
array, `mesh_queue` and its indexing wrapper are removed. The library is
2,272 → 2,349 maintained Swift/C/header lines. This implements N1t's transport
storage cycle in source; call/status namespaces, native rearm, per-worker
instance pools, and the unbounded N1 API remain unfinished at ABI 53. Native slot
return is implemented by ABI 54 below. No run is claimed.

## Native slot return

The native return path returns operand storage to the function's numerical worker on
ordinary completion. A slot owns its output sections, including canonical input
placement storage, and one native invocation. Setup captures each output's
reference-count template, removes the unused references, and records the owning
slot and return channel in its existing buffer head. Unassigned local backing
stays allocated to the realized function. The existing bitmap allocator remains
responsible for its eventual program retirement.

The numerical worker owns a compact free-index ring for each of its functions.
A first operand or root event acquires an indexed join record, separate from the
native slots. It records the invocation and arriving logical input rows. The
operand countdown reaching zero pops a native slot, transfers those row indices
to its prepared operands, and returns the join record. Assignment restores output
reference counts and invocation stamps. No slot is chosen by invocation modulo
and no occupied slot is queried.

In the working ABI 55 source, a slot with Q outputs has Q return events. Native
completion publishes outputs directly, so an output's final ownership also
proves its native execution has finished. The slot retains its input references
through final output ownership. Its numerical worker releases those references
when returning the slot, including on the failure notification. A successful call
with no outputs publishes one synthetic return event. The final event replenishes
the native launch object, if required, and pushes the index back into the ring.
Only the numerical worker decrements this return count or changes the ring. Core ML and Metal
callbacks and TX completions only publish the existing atomic row notices; they
are not incorrectly treated as a single SPSC writer.

Each output emits once at refzero. No event from a previous use can remain when
its slot returns: every event was consumed to make the count zero. The worker
drains available returns before the next publication dequeue; it never waits for
a missing return. Callbacks publish completed outputs directly, with no extra
completion hop. A per-worker active-callback count keeps program metadata alive
through the callback's final action. It does not authorize publication or transfer.

Metal's private queue has V command-buffer positions. Initially V objects are
prepared. A return event proves this slot's GPU execution has finished, so at
most V-1 other objects are unfinished when its replacement is made. Creation
therefore requires no completion from another slot. The numerical worker alone
reads or changes the command array, and the completed object is never recommitted.
This uses Apple's existing command-buffer API; object creation and encoding still
have their native costs. No operand allocation, view factory, pipeline compilation
or queue-capacity polling is added to numerical invocation. CPU functions need no
rearm callback. Core ML reuses its prepared feature provider and output options
only when that slot returns.

The same mechanism applies on a rank whose calls are all driven by received
operands. Neither assignment nor output/native retirement depends on a local
`submit`. For a function with a varying input, choose any one of its input
sections with V rows. Every occupied native slot retains a distinct row of that
section. A newly complete input set owns another distinct row, so at most V-1
native slots are occupied. Returning input references and the native index on the
same numerical worker precedes its next publication dequeue. Thus native admission
needs no availability guard. Functions with only shared inputs consume local
root admissions, also bounded by V.

For E varying operands, at most E*V pending joins can exist: each owns at least
one input occurrence, and no occurrence belongs to two invocations. Shared-only
functions instead have at most V pending root admissions. The match table has
power-of-two capacity at least twice this bound, preserving an empty probe entry.
These proofs depend on valid input-row ownership. The receive-binding bound across
unbounded submissions and shared physical pools remains unfinished N1 work.
Local send completion alone does not establish that remote bound.
Failure cancellation remains R2 work: a failed native call does not publish its
outputs, and abandoned references are reclaimed during program/device retirement.

The free-index ring has power-of-two capacity at least 2*max(E,1)*V, shared with the
invocation-match table's mask, and four bytes per entry. It uses the preallocated
ring representation cited under [Program.copy](algorithm-sources.md#programcopy),
with one numerical worker owning both positions and no fullness or reader query.
Output reference templates use four bytes per output; each native slot has one
metadata-only return-notification row. Working ABI 55 also changes invocation tags,
matching and result/event storage; its complete capacity accounting and caller
integration remain N1 work. No runtime or speedup claim follows from these edits.

## Invocation identity and storage reuse

ABI 52 distinguishes the invocation label k, each function's native storage slot
s_f(k), a logical operand row j, and the physical pages backing that row:

\[
 (k,j) \longmapsto (s_f(k),j) \longmapsto \operatorname{pages}(s_f(k),j).
\]

The row's publication carries k. A transport chunk contains `(k, sourceChunkRow)`
in its eight-byte tag; the queue's source-row relation determines its destination,
and the received head retains k for numerical consumers and forwarding. The tag
does not carry a raw pointer or select the receiver's native storage. TX stores
the tag immediately before posting. Concurrent links sending the same backing
write the same atomic word; their declared references keep that backing live.
RX does not rewrite the received tag. The existing pairing version rejects a
different wire ABI before posting data.

A prepared consumer range contains `(function, inputPosition)` entries. Its
single numerical worker indexes that function's call record directly by k;
there is no hash search, cross-worker slot claim or admission coordinator.
The first dynamic input or root publication assigns the function's next prepared
native slot. All inputs for k populate that record, independently of their own
producer slots. Shared inputs update their prepared views once across the finite
extent and discharge their dependency for every record. They do not allocate
native slots ahead of dynamic inputs.

For example, P can produce k=1 in slot 0 and k=0 in slot 1, while Q produces k=0
in slot 0 and k=1 in slot 1. If P(1) reaches C first, C assigns its slot 0 to k=1.
Q(0) can independently assign C's slot 1 to k=0. Q(1) then fills C(1)'s second
operand using Q's slot 1; P(0) fills C(0) using P's slot 1. Neither C call mixes
invocations or requires the producers to agree on slot order.

`mesh_operand.invocation` is k; `.index` selects that operand's native storage;
`.row` names its original logical value; `.page` and `.data` select its native
view. A materialized input retains its original row while its data/index name
the consuming function's contiguous storage. Placement resolves each source
chunk from `.row`. Completion releases the original rows actually consumed,
publishes output rows already held in the operand array, and retires status k.
Core ML feature inputs and Metal matrix inputs therefore select their own views;
native command buffers and prediction output options use the consumer slot.

This is still a finite extent: call/status arrays are indexed by k in `0..<count`,
ABI 54 recycles each function's prepared storage slots, and ABI 53 recycles receive
targets and backing within that namespace. It does not implement
`submit(inFlight + k)`. The wire identity and
independent storage selection remove the prior coupling. ABI 54 supplies native
slot return; unbounded invocation matching and instance admission remain required.

The maintained Swift/C/header total remains 2,272 lines. The operand is now 32
bytes instead of 24; the invocation field fills existing padding in the 32-byte
buffer head. Consumer entries remain eight bytes; shared-input entries shrink
from `count` entries to one per binding. Send edges shrink from 24 to 20 bytes,
receive targets from 16 to 12. A function pointer array adds eight bytes per
function. The duplicate output-section array, consumed-section copies,
remote-refresh index list, completed flags and submission/retirement forwarding
helpers are removed. These are representation costs, not measured latency gains.

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
of live work. The current finite status array adds no completed-result eviction
policy and supplies no unbounded result-retention claim.

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

### Explicit section identity on the wire

Protocol 57 carries `(invocation, sourceHead, definition, chunk)` in a 24-byte
tag, replacing the 16-byte invocation/source-chunk tag. `definition` names the
sender's logical section; `sourceHead` names its current live descriptor; `chunk`
is an ordinal independent of either peer's OS-page units. A queue's setup table
maps the peer's definition to its local binding. The first chunk assigns the
local head, and an array indexed by source head names it for subsequent chunks.
Publication occurs at the binding's final chunk. No arrival cursor reconstructs
the section identity, and no particular value can hold up another queue.

The receive target records, source-value records, prefix offsets, cyclic cursors
and transport `chunk_stride` field are removed. The two new lookup arrays use
eight bytes per row in the peer's advertised row namespace per receiving queue;
queues with no receives allocate only empty placeholders. They may exceed the
old prefix arrays when the old source-row high-water mark was small. Setup no
longer generates target records for every chunk of every source slot.

All TX links forwarding the same backing write identical tag fields, as they
did for the old tag; no destination-specific field is stored in shared payload
backing. The same-QP ordering contract ensures a source head's old final chunk is
drained before its next first chunk replaces the active-head entry. Other source
heads and other peer queues may interleave freely. The versioned pairing rejects
the old wire layout before data posts. The extra eight tag bytes do not add a
4096-byte transport frame for page-multiple payloads. No latency result is claimed.

This removes receive matching's dependency on pre-enumerated sender storage
slots. Per-binding destination row stacks, TX edge storage and native/lifecycle
capacity still need the remaining N1 allocation work; this change does not
establish an unlimited-admission bound.

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
