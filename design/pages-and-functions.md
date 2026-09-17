# Pages and functions

The [user's requirements](collective-goals.md) define scope. This note describes
operand and lifetime relationships; it prescribes no executor or scheduling method.
The caller supplies the mesh and tensor placement. Mesh realizes their storage,
indexed dependencies and numerical function bindings before invocation.

Pages back values; they do not define tensor dimensions or numerical call extents.
Configured views name the actual registered storage. Presence denotes available
values, and declared reads retain their storage through native completion. Transport
completion and numerical availability are different facts.

The page table exists to minimize address-resolution latency. Setup resolves
fixed destinations, offsets, fan-out and invocation targets. Runtime page indices
represent incoming storage; they do not authorize walking descriptors to discover
an already-known destination or interpreting the declared call graph again.

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
ABI 72 stores the physical page and its prepared native-view index in one
64-bit entry. The latter is also the receive-pool return index for remote
backing. [Native view selection](#prepared-native-view-indices) reads this
terminal value instead of reconstructing it through an indexing closure.

There is one owning reference: an unfinished use keeps its allocation alive.
For section s, R(s) is the number of those uses, including setup's temporary
ownership. Producers, numerical readers, observations and transport operations
all use the same `mesh_buffer.references` and `mesh_buffer_release`; their names
describe where use ends, not separate reference mechanisms. Setup derives the
uses from the declared feed-forward graph and drops its reference after all
bindings are realized. Existing completions discharge those uses. Early publication does not release an
unfinished producer. The final reference publishes a free-pool entry without
clearing payload bytes. Independent work already has its configured sections and
never awaits reclamation.

`TensorPart` contains rank, byte extent, a C section descriptor and value flags;
it has no heap-object reference or destructor. Copying or retaining the descriptor
does not add a reader or delay reclamation. `MeshMemory.sections` holds setup ownership during
configuration. `Mesh.start` releases it after binding all functions and transfers,
before starting numerical workers; failed configuration returns any setup
references still held by that array. The array is then empty. Backend views keep
the memory mapping alive; their actual reads are owned by the declared calls.

All retains occur before setup ownership is dropped, when R is necessarily positive. They need
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
storage remains alive through outstanding native callbacks. ABI 52
leaves shared and unissued operand references to the existing client-retirement
event, after native callbacks end and the bridge closes the queue pairs. It removes
the destructor's reconstruction of consumed rows from consumer slot numbers.
Thus the constant remains live throughout every
native read without per-invocation reference updates. This groups equal storage
lifetimes; it does not remove input-arrival dependencies or alter tensor values.

Call storage has one Mesh owner reference and one reference per started numerical
worker. Startup acquires the worker references before creating the threads;
failed creation releases the references for threads not started. Each worker's
local submitted count and completion-only atomic count keep that worker alive
through native callbacks after destroy clears `running`. Launch performs no
shared counter increment; the completed count is read only during shutdown. Worker exit drops its program reference. There is no additional
per-instance program reference, and worker exit does not walk or cancel unissued
operand uses. Those references end at client retirement after native callbacks
and QP teardown. A forwarding-only rank starts no numerical workers.

The implementation trusts caller configuration and backend completion contracts.
Its ownership records describe actual accesses; they do not police arbitrary
external code. No caller free/done call or consumer-stamp protocol is required.

An address, page-table entry or copied section descriptor identifies storage; it
does not itself own a use. A call's pending-input count expresses a numerical
dependency, not allocation ownership. A frame's completion count records when
its work ends. Private call storage has its own allocation count, governed by the
same lifetime rule. None of those distinctions licenses another buffer refcounter
or a caller-managed release protocol.

### Client retirement has one owner

Detach, replacement of a dead client, and the controller's process-exit event use
the same `mesh_retire`. It claims the exact old client identity with a fresh
generation and the retiring process's live PID, retaining the old notice bank.
The claim precedes clearing configuration, transfer lengths and row ownership.
A competing controller or replacement cannot clear a newer client's state;
attachment sees a live owner until retirement publishes the vacant client slot.
No reference category, extra thread, or data-path ownership check is introduced.

The link controller registers `NOTE_EXIT` before pairing. It closes a failed
connection through `link_close`, but continues observing client exit until detach
or bridge shutdown. A socket/CQ failure therefore cannot discard a later process
exit. The controller also handles `ESRCH` during process-watch registration through
retirement. Closing the socket removes its descriptor events while preserving
the process watch, as specified by the cited kqueue interface.

Retirement marks operand storage closed; it does not immediately free pages.
The bridge's existing outer loop discharges abandoned references and returns
backing only after every old link controller has joined and its QPs have closed.
Client disappearance now triggers that path without needing a new attachment.
Death during pairing is observed after the existing bounded pairing call returns.
This does not implement cancellation of a still-attached failed program or
same-program recovery; R2/R4 and N1 remain open.

ABI 66 also makes the existing buffer owner atomic. Setup reclamation and the
bridge's discharge of abandoned uses each claim that field before changing the
descriptor. The bridge publishes the free-pool bit before restoring the owner as
its last write. Setup leaves the bit pending until it claims the descriptor,
then removes the bit and returns backing and row capacity. Only setup makes the
descriptor reusable. Initialization publishes owner last and writes its atomic
fields individually. This prevents cleanup's stale closed/count observations
from modifying a replacement allocation.

The claims occur in setup and teardown, with no retry or wait. Publication and
transport retain their existing reference-release and notification paths; owner
reads there are relaxed loads. The buffer remains 48 bytes and receives no new
field. The allocator still has one configuration owner per region; this change
does not implement T7's multiple concurrent clients or N1's remote-value reuse.
Maintained library source is 2,454 → 2,481 lines across the same eleven tracked
Swift/C/header files. Documentation changes are counted separately; this repair
is not a structural source reduction.
Strict C compilation, the existing four Mesh callers and the engine's Mesh
library build pass. No runtime or latency result is claimed. The local bridge
build uses the Makefile's ad-hoc signing fallback because the named signing
identity is unavailable; no bridge deployment was performed.

## Publication notifications

Tensor presence is one atomic 64-bit word per logical row at a fixed offset in
the shared mapping; ABI 50 introduced per-row words and ABI 53 widened the stamp. Row allocation initializes the word to zero;
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

ABI 70 publishes the destination's already prepared 32-bit record index and
the 32-bit invocation label together in one atomic 64-bit event. Setup assigns each row/destination a 16-byte target
`{stream_offset, record_index, count}`; no process address crosses the mapping.
The count is setup data. Publication reads the stream and final index from that
same target, writes the TX events first, stores presence, then writes local
events. It then stores the invocation used by later receive-return bookkeeping.
TX and numerical consumers take their label from the event, so neither rereads
the publishing buffer. The publisher's existing producer reference protects
that final store until publication returns and releases the reference. No new
reference or completion handshake is required. Destination reconstruction from
the tensor row remains deleted.

For TX, ABI 71 transfer realization counts each row's native requests on each
link and assigns a contiguous send-record range. Bridge preparation fills that exact range. The
first send record contains its end. The TX event therefore goes directly to
`send_edges[index]`; `link_publications` no longer reads a row-indexed offset
table. The event's invocation is stored with the queued range instead of
retaining an address of the buffer's label. The remaining loop visits actual declared sends, including distinct
queues on the same link. Its setup cursor array is freed before progress starts.

For numerical work, setup expands every input use, including a shared input's
uses across all resident frames, into contiguous records. The event names the
first use record, which contains the range end. The worker no longer reads
`offsets[row]` and `offsets[row+1]`. Those arrays exist only during construction
and are freed before worker launch. Use and call records remain 64 bytes;
range end fits inside the use record after removing duplicate remote-address
fields. Native SEND wrappers remain 256 bytes and receive
records remain 32 bytes. No tensor function or collective is specialized by this
change. A row with three sends or five consumers still visits all of them.
Each use now carries a setup-defined invocation mask instead of a label pointer.
For a varying input the all-ones mask selects the event label; a shared input's
zero mask preserves the call's label, including when the shared value arrives
after varying inputs. The expression is `old ^ ((old ^ incoming) & mask)`;
it adds no readiness predicate. Native dispatch loads only the use and call
records for these fields; the label no longer requires another buffer record.

### Direct publication operands

ABI 76 moves the send/use counts out of `mesh_buffer` and into a prefix of
the existing target array. The prefix and targets form one immutable publication
record, aligned to 64 bytes. Counts occupy bytes 0–7, the cold ownership row
bytes 8–11, and the first target bytes 16–31. Publication reads no lifecycle
buffer before its handoffs. This is the existing target storage with its counts
adjacent, not a descriptor pointing to another target allocation.

Each numerical operand stores the terminal publication pointer prepared at
binding; its copied row number is deleted. Completion loads that pointer and
calls `mesh_publish` directly. Each receive record likewise stores its final
publication pointer, replacing its row and separate publish flag while remaining
32 bytes. Intermediate receive entries have no publication operand, as previously
expressed by their false publish flag. This is unchanged declared message
structure, not a readiness or reuse test. Root admission still addresses its
prepared record by frame; this change does not complete N1.

Generated `mesh_publish` is frameless. It writes all TX events before reading
presence storage, then writes local events, and only then resolves the lifecycle
buffer for its existing invocation store. The numerical output loop performs
no row-to-target arithmetic or arena-header target lookup. Stream cursor/slot
loads, the output-pointer load and the publication-record load remain counted;
this does not establish the full one-record H1–H7 budget.

For D = links + compute-worker capacity and R arena rows, target storage changes
from `16DR` bytes to `round_up(16 + 16D, 64) * R`. At one link and eight workers,
the stride is 192 rather than 144 bytes: an extra 11,010,048 bytes at R = 229,376.
The lifecycle buffer remains 64 bytes despite losing its two counts. Operands
grow from 40 to 48 bytes to hold the terminal pointer instead of the row; there
is no per-invocation allocation. Maintained source grows from 2,805 to 2,830
lines across the same eleven files. Native library, existing callers and engine
integration builds pass; no runtime workload or latency measurement is claimed.

### Independent publication streams

ABI 68's independent slots removed the unpublished-reservation stall of ABI 67,
but required O(E) probes over E subscribed row/destination pairs. ABI 69 groups
bindings into preallocated single-writer streams during setup:

- All receive publications from one link use its one RX writer.
- One native completion publishes its outputs in sequence.
- Native writers may share a stream when an explicit recurring input dependency
  orders their writes to that destination. For local notifications, setup also
  checks that the predecessor's destination store precedes its notification of
  the successor's worker. This uses declared edges and target order, not an
  inspection of the supplied tensor function.
- Immutable constants have a single setup writer. Final-reference returns stay
  independent because their releasing thread is not known from the producer.

`mesh_events_prepare` uses temporary parent/group arrays, sorts the binding
indices, assigns ring capacities and rewrites every publication and return
handle to the final stream offset. These temporary maps are then freed. The
parent walk, sort and remapping do not execute on receipt, publication or launch.
The chain grouping is greedy, not an optimal path cover. Forks and unproved
ordering remain separate; arbitrary residual graphs are not claimed to collapse
to one stream. A straight single-output dependency chain can share a stream per
resident frame where the notification-order condition holds, independent of
whether it has eight or one hundred functions.

Each stream contains 64-bit words encoded as `(invocation << 32) | (index + 1)`,
with zero meaning empty. The index retains its existing reserved absent value.
Return events use invocation zero. Its producer advances its own position and
release-stores the word. A compile-time assertion requires an eight-byte,
lock-free slot. There is no shared reservation, CAS, sequence admission or occupancy
read. The consumer's position and payload pointer are private, allocated before
progress begins. It acquire-loads the next slot, clears a consumed slot and
advances; an empty stream advances the polling cursor to another stream.
No unpublished ticket in one stream blocks a different stream.

A stream's capacity is the next power of two at least as large as its number
of bound publication locations. Reuse of a location follows its declared
operand lifetime; consumption clears the event before those uses execute and
release their references. This argument still requires N1's complete lifetime
realization. It does not prove cross-participant frame reuse by assuming it.
Constants are seeded once after record indices and stream offsets are final,
before the transport is configured. No stream handle changes during execution.

ABI 73 places each stream's slots immediately after its cursor and mask. The
separate slot-offset field is deleted. A compiled stream starts on a 128-byte
boundary and occupies `round_up(8 + 8 * capacity, 128)` bytes; no other stream
shares that range. Its slot address is `stream_base + 8 + 8 * index`, without
loading a second address. Rings with capacity at most eight fit entirely in
one 128-byte region. Larger rings retain the same address calculation but their
slots span more regions. This makes no claim that an entire larger ring is one
cache line or remains resident.

Provisional bindings also have aligned locations. Their first slot temporarily
holds the setup writer identity. For each queue, setup copies all those
identities into its sorting scratch before writing any final streams over the
provisional storage. It zeros every compiled slot, rewrites the handles, then
initializes readers and seeds constants. The reader's variable-size layout walk
runs only at setup; its existing private input records hold final slot pointers
and masks. Attachment no longer clears the entire reserved slot capacity.

Reader-input allocations now begin and end on 128-byte boundaries, and each
numerical worker's cursor/active state occupies its own asserted 128-byte record.
The encompassing call object uses an aligned allocation too. These boundaries
prevent unrelated owners from sharing those mutable lines at that granularity;
they add no thread, synchronization, publication store or runtime layout walk.
Producer/consumer sharing of the event slots themselves remains the intended
handoff.

### Costs and remaining work

For P compiled streams an empty pass performs P acquire probes, rather than E
individual-location probes. P is not guaranteed independent of graph size.
A probe still reads private input state and then its shared slot. An enqueue
reads its cursor/mask pair, updates its producer position and stores its event
directly into the inline slots. Generated ARM64 code uses `ldp` for the two
32-bit fields, followed by indexed address arithmetic and `stlr`; the stored
slot-address load is absent. Cross-stream descriptor sharing is removed, but
private-reader/slot probes and polling latency remain. P is unchanged by this
layout correction; grouping is not free progress.

After a successful dequeue, TX and numerical dispatch directly address the
terminal record array. Optimized ARM64 output shows `index << 8` followed by the
SEND range-end load at byte 40, and `index << 6` followed by the use range-end
load at byte 40. The former offset-table loads are absent from those paths.
This is source/assembly evidence of one removed dependent table access per
dispatch, not a measurement of cache misses or end-to-end latency. ABI 71 TX
uses the prepared backing address/key and writes the wire tag. Numerical launch still accesses the declared dependency count and sequence
value and records its active lifetime. ABI 74 removes cold return processing's
row-to-call lookup through [direct return indices](#direct-native-return-indices). The operand changes below remove address refresh and both
operand loops from launch; they do not remove stream polling or TX binding.
H1/H3/H5/H6/H7 and N1 therefore remain incomplete.

The allocation tradeoff is explicit. At 229,376 rows, one link and eight native
queues, the two banks of 25 event arrays reserve 367,008,000 bytes before region
alignment in ABI 70, versus ABI 69's 275,257,600 and ABI 68's 45,881,600.
Carrying labels adds 91,750,400 reserved bytes over ABI 69. Targets remain
33,030,144 bytes; the buffer remains 64 bytes. The index/label pair shares one
atomic load/store, but event density decreases from eight to four per 32 bytes.
This is a storage trade for removing dependent buffer-label reads, not free
capacity or proof of cache residency.
Only compiled streams are polled. Each private stream input is 16 bytes and each
reader is 16 bytes. Numerical offset tables no longer occupy 1,835,016 bytes per
worker during execution; the TX setup cursor likewise is not retained as a live
allocation. This trades reserved arena capacity for fewer probes and removes
two runtime lookup paths; it is not a source-size reduction. ABI 70 changes
maintained library source from 2,713 to 2,717 lines across the same eleven
Swift/C/header files. Documentation changes are separate. Strict C diagnostics,
the four existing Mesh callers and the engine Mesh library build pass with ABI 70.
Optimized ARM64 dispatch loads the use and call records with no buffer-label
load; publication writes the persisted buffer label after its event stores.
In ABI 70 SEND preparation decreased from 23 to 22 instructions and 12 to 11
load instructions in the previously defined interval. ABI 71's [prepared native
operands](#prepared-send-operands) subsequently remove the page/registration
chain from posting. The bridge uses
the existing ad-hoc signing fallback because the named identity is unavailable;
no bridge deployment, runtime test or latency result is claimed.

ABI 73 reserves `128 + 128R` bytes per event array. For a group of n bindings,
its next-power-of-two capacity c gives a final extent
`round_up(8 + 8c, 128) <= 128n`. The n = 1 case occupies exactly 128 bytes;
for n >= 2, `c < 2n` and rounding adds less than 128 bytes. Summing over the
queue's groups proves that its aligned rings fit the reserved R binding slots.
Thus the two banks of 25 arrays at R = 229,376 reserve 1,468,012,800 bytes,
an increase of 1,101,004,800 over ABI 72. The reservation is deliberately
reported separately from the compiled stream extents and from resident memory.
Setup touches provisional bindings and final rings; it does not clear every
reserved event slot. Neither source accounting nor that deletion proves RSS.
The coarse per-queue reservation and graph-dependent stream count remain open.

Each private reader allocates `round_up(16 * max(1, P), 128)` bytes for its P
inputs, instead of `16 * max(1, P)`. Numerical worker fields still occupy 56
bytes, but their aligned records use 128 rather than 56 bytes each. The containing
call allocation is 1,280 bytes in the native build, including padding, versus
536 bytes. These are setup allocations; the publication and polling loops add
no allocator, fullness predicate, shared ticket or progress thread.

The eleven maintained library files grow from 2,791 to 2,799 lines. Static
assertions fix the stream alignment, slot offset and worker stride. Strict
native compilation, generated C/Swift assembly, the bridge and existing caller
builds pass. The engine Mesh integration build also passes. No workload,
deployment or measured latency result accompanies this change. The removed
slot-address load and owner separation are regression boundaries, not full H1.

### Operand addresses and sequence values

Address binding is not a separate runtime phase. Let A be the numerical client's
mapped data base, s its page size, Q the realized byte extent of one entry, and
p[k] its physical page. ABI 75 stores the terminal address a[k] in that same
canonical entry when backing is established:

```
a[k] = A + s * p[k]                 // setup or receive placement
address(offset) = a[floor(offset / Q)] + (offset mod Q)
```

The current 48-byte operand carries the entry pointer, Q, extent, prepared
publication pointer, frame index and a pointer to its call's sequence value. It no longer carries A or
separate page geometry. Swift `data` loads a[0] directly; indexed `load` selects
the entry and adds the within-entry offset. The `page` and `invocation`
accessors retain their meanings. There is no dispatch refresh loop.
Native `MeshBindings` select the already constructed view using the index now
stored beside the physical page; Core ML objects, output backing and model
invocation remain prepared. The ABI 72 change below deletes their indexing closure.
The canonical read remains, including for local raw `data`. Memoizing the
address removes reconstruction, not the entry read or every dependent load.
The [storage and placement account](#prepared-operand-addresses) below therefore
does not claim H6 completion.

Every operand of a call references the same existing sequence scalar. Launch
therefore does not stamp each operand. ABI 70 carries the invocation beside
the record index in each publication event. Numerical dispatch selects that
label with the use's prepared mask; TX retains it with its queued range. Neither
follows a label pointer back to the source buffer. The single persisted buffer
label is written after TX and local notifications for receive-return bookkeeping.
The existing producer reference keeps that bookkeeping live through the store.
The remaining publication and lifetime work still counts against H3/H5.

The remaining pending count has a narrower meaning than a call-state machine:
for a function with n declared varying input uses, each arriving use decrements
its prepared scalar, and the final use invokes the already stored function
address with its already prepared operand array. Shared input uses are expanded
at setup and counted for their actual first use. Reference and completion
counts govern buffer/callback lifetime. No address-binding problem licenses
additional discovery, interpreter state, backend selection or default waits.
These statements describe the implementation changes and the remaining actual
data dependencies; they are not an argument to defer the rest of the hot-path
rewrite.

### Prepared native view indices

ABI 72 memoizes the native-view selection result in the canonical page entry:

```
entry = (uint64(view_index) << 32) | physical_page
```

For local storage, `mesh_backing_bind` writes the section's resident slot index
during allocation. All chunks of that slot share its native contiguous view;
shared constants have slot zero. For received storage, the completion writes
the physical block's index within that queue's prepared pool. Native views for
a directly consumed remote partial are already constructed in that same pool
order. A contiguous materialized input uses its local placement entry, whose
index was prepared during allocation. These are configuration and placement
facts; the numerical reader does not classify the operand.

`MeshBindings.index` now loads that prepared index, and subscripting uses it
directly. The stored Swift closure, its captured range geometry, indirect call,
remote page subtraction/division and local stride multiplication are deleted.
The method keeps the existing `bindings.index(operand)` calling syntax used by
the engine's supplied encoders. No numerical function or backend is replaced.
`mesh_operand_view` accepts the already prepared entry pointer and returns a
native-sized integer. Generated subscripting code inlines the load and shift;
the public index method uses a direct tail branch to those same two operations,
with no stack frame, captured closure or operand copy at that boundary. Array
access and native-object access still have their normal costs.

At ABI 72 the 48-byte operand retained one pointer to this canonical array. Indexed tensor
reads and the `page` accessor use the low word; view selection uses the high
word. The per-call `operand.index` continues to mean the resident invocation
slot and is unchanged. There is no second mapping table, mutable cache per
consumer, dispatch refresh loop, receive-readiness predicate or new handoff.
For local views the read now visits the canonical entry instead of using the
operand's inline slot through a closure. That one data read remains explicit;
the removal is the closure/context traversal and recomputation, not every load.

RX already derives the block coordinate `b = region * region_blocks + slot`.
Setup supplies `pool_offset = first_region_block - first_pool_block`, stored
modulo 2^32. The view index is `b + pool_offset` modulo 2^32, which is exactly
the nonnegative pool index for every posted receive. RX packs this and its
physical page into one aligned lock-free 64-bit store before publication.
Recycling loads the same word and appends its high half directly to the prepared
RECV ring. It no longer subtracts a pool base and divides for each returned
block. Its row traversal uses consecutive entry indices; only the section's
chunk-count division remains, once before that loop.

The page array costs eight bytes per arena row instead of four: an extra 4R
raw metadata bytes for R rows, or 917,504 bytes at R = 229,376. Operand storage
and payload storage are unchanged. RX geometry grows from a 32-byte header to
a 40-byte extent aligned to 64; the complete receive state stays 128 bytes.
Static assertions fix that extent and alignment. No additional allocation or
event occurs during execution. Bridge and clients must agree on ABI 72.

The eleven maintained library files total 2,791 lines, up from 2,790. The existing
`native-audit` target also emits optimized Swift assembly from the library
source, alongside the C assembly; it adds no evaluator or runtime harness.
Strict C compilation, both library builds, all four existing callers and the
engine's Mesh library build pass. No runtime workload or latency measurement
is claimed. Canonical indexed reads, contiguous-input materialization,
publication-stream probing and full H1–H7 closure remain unfinished.

### Prepared operand addresses

ABI 75 extends the existing canonical entry from eight to sixteen bytes:
the physical-page/native-view word and the final numerical-client address.
Both are individually atomic; the entry is aligned to sixteen bytes. Local
allocation computes the address once. Attach records the numerical client's
mapped data base in its existing ownership bank; link configuration caches
that base in the receive geometry. Receive placement computes the same address
for its actual incoming page. The bridge never dereferences a client address.
The mappings need not share a virtual base, and there is no remapping step.

The address store precedes the existing release publication. Consumers acquire
that publication before using the entry; the existing reference lifetime keeps
the backing live during use. These are the same ownership requirements as the
page/view word, not an additional readiness check. Reclamation need not clear
the address. Cross-participant reuse remains the unfinished N1 obligation.

The existing contiguous-input path now consumes setup-prepared copy records.
CPU/Core ML records contain the source entry, terminal destination pointer and
byte count. Metal records contain the source entry, prepared native-view array,
destination object, offset and byte count. They are flat per-frame arrays;
runtime no longer calls `mesh_row_page` or reconstructs destination addresses,
pool coordinates or copy extents. Shared sources keep their zero stride;
each destination uses its own resident frame. This changes neither copied bytes
nor the supplied function. These copies remain an explicit H6 defect.

Native views use `ContiguousArray`, including `MeshBindings.values`. This fixes
their storage representation during setup, removing the bridged-array fallback
from native view selection. Existing subscripting and index methods remain.

For R arena rows, canonical storage grows by 8R bytes, or 1,835,008 bytes at
R = 229,376. Each operand shrinks from 48 to 40 bytes. The header stores two
eight-byte client bases; receive geometry grows from 40 to 48 bytes within its
existing 64-byte alignment and 128-byte state. RX adds one address store and
its fixed arithmetic per received entry. For F frames and C copied entries,
CPU copy records reserve 24FC bytes and Metal records reserve 40FC bytes,
excluding Swift array headers and the existing native objects. This explicitly
trades setup storage and one producer-side calculation for repeated consumer
resolution. It is not free work or a whole-path latency result.

The native `data` getter is two loads and a return: descriptor to entry, entry
to terminal address. CPU placement's inner loop loads its 24-byte copy record,
loads the cached source address, and calls `memcpy`; Metal uses 40-byte records.
Native library, existing caller and engine integration builds pass. No runtime
workload was run. Maintained source is 2,790 → 2,805 lines across the same eleven
files. H1–H7, N1 and R2 remain open; no completion status is upgraded here.

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
destination row, its buffer address and canonical page-entry address. ABI 64
removes its duplicate reference count and first-chunk flag; ABI 65 replaces the
shared flag and frame index with a prepared completion count, retaining only
the final-chunk marker. RX indexes that record using the source chunk in the
received tag. It does not allocate, search for a row or
reconstruct the page-list address from a buffer descriptor.

For transfer t, resident frame f and chunk k, setup binds
`sourceFirst[t] + f * sourceStride[t] + k` to destination row
`localFirst[t] + f * localStride[t]` and its k-th canonical page entry. The
source chunk already includes f; receiving a sequence number does not require
another modulo calculation to recover it. Realization preserves the declared
references; each chunk stores its actual received page. The final chunk stores
the sequence and publishes. The section's physical
pages may be interleaved with other sections or peers. The formula determines
logical coordinates; it does not make the physical pages contiguous.

A queue still owns one ring of physical receive blocks. Final-reference notices
return all of a section's actual blocks to that ring, clear its mappings, restore
its prepared ownership and clear presence through `mesh_buffer_reset`, and
release its declared completion range. A long-lived buffer has an empty range;
its return releases no frame reference. RX posts available blocks until native
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

Every function has one prepared call state and operand array per frame. Its
row-indexed consumer records contain the launch target and arguments.
A dynamic publication fills that frame's input and
decrements its pending count; zero invokes the supplied function using the same
frame's prepared output/native bindings. No hash probe, join record acquisition,
backshift deletion, or per-function native-slot ring remains.

Shared inputs update every prepared frame once and decrement each countdown.
Setup initializes all countdowns, including the root dependency for a function
with no varying inputs. Successful native completion publishes outputs, releases
that call's consumed input references, then releases its outputs' producer references.
Their final references return through the existing numerical-worker notices;
the last output return rearms the native object
when necessary, restores that frame's pending template, and releases one frame reference directly. Zero-output functions now emit their final call index directly.
A native error concludes status separately; complete failure cancellation is R2.

### Direct native return indices

ABI 74 removes the return path `row -> buffer.binding -> calls.slots -> call`.
Call records occupy one stable, aligned array. Setup assigns function f and
resident frame v the index `f * inFlight + v`; operand sequence pointers and
prepared uses reference those records from their initial construction. Nothing
is relocated or rebound at launch. The numerical worker retains the array base
when it starts and addresses `values[event_index]` directly on a return.

A buffer's existing 32-bit binding field is now its prepared `return_index`.
Received buffers retain their logical row as that index; numerical outputs carry
their owning call index. Their already distinct destination streams determine
which array consumes the index. `mesh_buffer_release` emits that value when the
one ownership count reaches zero. It adds a field read from the same 64-byte
buffer record, while removing the buffer-record read and pointer-table read at
the numerical consumer. There is no runtime return-kind classification.

Each native completion also has its prepared call index for zero-output and
failed calls. Zero-output functions therefore need no dummy row, buffer binding,
row allocation or matching lookup. The `return_first`, `return_count` and
`return_row` bookkeeping and the `calls.slots` pointer array are deleted.
All three supplied backend kinds retain the same completion path and owning
worker for native rearming. A call with several outputs still counts their
actual final-reference events, and rearms after the last one; the reference
protocol and event count are unchanged.

Receive setup still binds a frame number for its instance-reference release.
That field shares the eight bytes formerly needed only for the setup publisher
identity. The publisher identity is consumed and discarded when streams are
compiled, before the client publishes `configured`; the bridge installs the
receive frame afterward, before receive progress. No runtime union test or
additional buffer storage is introduced. The buffer and call records remain
64 bytes with their existing assertions.

For S = function_count * inFlight and R arena rows, the old implementation used
64S bytes in separate call allocations plus an 8S-byte pointer table. The stable
array reserves 64R bytes when the first numerical function is bound; only the S
bound records are initialized. A transport-only participant allocates none.
At R = 229,376 the array reserves 14,680,064 bytes. The existing S <= R setup
bound moves to binding, where it prevents writing beyond that array. There is
no invocation-time allocation or new admission check.

Deleting dummy rows also deletes their implicit accounting for return-event
capacity. With at most R distinct output rows and at most R call records,
numerical returns need at most 2R provisional bindings, including zero-output
completions. Publication queues still need at most R. The current uniform event
reservation uses the 2R bound for every queue, and setup scratch uses the same
bound. No stream-capacity check is added to execution. This coarse allocation
remains an explicit cost: `128 + 256R` bytes per event array, or 2,936,019,200
bytes for the documented two banks of 25 arrays at R = 229,376, versus ABI 73's
1,468,012,800. These are reserved bytes, not a resident-memory measurement.

Maintained library source decreases from 2,799 to 2,790 lines across the same
eleven files. Generated numerical return code uses the event index shifted by
six to address its call record; no row binding or call-pointer table intervenes.
Strict C/Swift compilation, the bridge, existing callers and engine Mesh build
pass. No runtime workload or latency result is claimed. This removes address
discovery from local return processing; it does not close N1's cross-participant
reuse, R2 cancellation, or the complete H1–H7 path requirements.

### Prepared numerical uses

The row's consumer range contains aligned 64-byte records. Each contains the
submit target, argument, completion handle, operand array/counts, invocation
source and optional remote-input binding addresses. Setup copies these from the
declaration. Dispatch calls that target directly; it does not load a call or
function descriptor to discover the target or arguments. The completion handle
still addresses mutable pending/invocation state and the frame index.

Setup expands a shared input into one record per resident call in the same
contiguous range. Arrival no longer classifies the input, loads its function,
finds the call array or reconstructs fan-out. Setup computes initially missing
inputs separately from the varying/root count used on reuse. Shared arrival
does not mutate the recurring template. Its invocation source points to the
call's current invocation, preserving the index established by a varying input
or root regardless of arrival order.

Local inputs and contiguous views retain their prepared bindings. An unplaced
remote input has its operand and canonical first-page-entry addresses in the
event record. Its source row remains distinct from the view's storage, including
for inputs already present at setup. No row, view index or page-list address is
reconstructed from a function on arrival.

For V frames, E varying/root bindings and S missing shared bindings, use storage
changes from 32(VE+S) to 64V(E+S) bytes. This spends setup memory to remove runtime
discovery. Both use records and call state have asserted 64-byte size/alignment.
Optimized Arm assembly loads the target and argument together with `ldp` from
the event record, then invokes that target with `blr`. No queue, callback,
runtime allocation or guard is added.

This removes descriptor dependencies, not every critical-path load. Notification
traversal, row ranges, mutable dependency state, optional remote-page binding and
invocation propagation still precede execution. The whole one-cache-line target
remains open; source length and successful builds are not latency evidence.

### Output ownership is prepared before launch

For each output y, setup determines R(y) = 1 + C(y) + T(y), where one reference
owns production and the others own declared numerical and transport reads.
Startup preserves those references. It no longer subtracts R merely to add it
back when the function becomes ready. Setup also initializes the call's number
of outstanding output returns, using one terminal return for a zero-output call.

Successful completion publishes outputs and releases its consumed inputs and
producer references. Every reader releases its reference through its existing
completion. Only after every output has returned does the owning numerical
worker rearm native storage, restore each output's R through `mesh_buffer_reset`,
clear its old presence, and restore the return and pending counts. All old uses
have ended. The reset uses an atomic store; the retirement flag is a separate word.
The worker releases the call's frame reference after this preparation. There
is no occupancy query, reader scan, additional return event or caller free.

Dispatch decrements pending, propagates invocation indices, increments the
existing active-call count and calls the target packed in its use record. It
does no output reference arithmetic or presence clearing. `mesh_call_ready` and
the call-record launch-field lookup are deleted. The function's submit pointer
is read at setup, when preparing the use records. Error cancellation and safe
cross-participant frame reuse remain R2/N1 work; this ordering proves local
rearming only.

### One buffer ownership count

ABI 64 stores the prepared reference count alongside the live 32-bit counter in
`mesh_buffer`. The packed 64-bit count/flag word is deleted; retirement has its
own 32-bit flag, so resetting the count cannot clear that flag. The prepared
count replaces the redundant logical-row extent (`pages / block`), so the record
remains 40 bytes. Realization snapshots the count once after all
bindings and setup releases. The function's separately allocated count array and
every receive record's duplicate count are deleted. Numerical return and receive
return call the same reset; publication and native completion release references
through the same existing `mesh_buffer_release`.

Producer, numerical reader and transport reader are uses of that one mechanism.
The prepared count is immutable setup data, not a second live reference count.
Long-lived binding references end with the program's existing retirement; they
do not require a special shared-input destructor. Retirement also discharges
unissued uses after native users and the QPs have ended.

This change does not make the buffer's logical row a safe reusable identity.
The row and operand page list can still be rebound while an earlier remote use
is live (N1). The separate receive/frame accounting defect discovered here is
corrected by ABI 65's declared completion ranges below. Cross-participant
allocation identity and cancellation remain unfinished.

Source effects: first-chunk receipt loses one atomic ownership addition and one
presence store; reset uses stores on final return. Sequence is stored at
publication, so the first-chunk flag and its per-chunk conditional are deleted.
Retain/release use 32-bit atomics without packed flag extraction. There is no extra event,
payload copy, allocation on launch, or measured latency claim. Maintained source
changes from 2,456 to 2,451 lines across the same eleven Swift/C/header files;
documentation changes are separate.

### Completion references are declared at setup

ABI 65 removes the assumption that every returned buffer owns the frame named
by its default binding index. Allocation now assigns no completion binding.
Numerical and transport realization assign the actual owners and release ranges.
The ranges are immutable program metadata; runtime neither chooses a reference
category nor asks whether a destination is occupied.

| Event | First frame | References released |
|---|---|---|
| Numerical return | Call's frame | 1 |
| Per-invocation SEND completion | Transfer's frame | 1 |
| Long-lived SEND completion | 0 | `inFlight` |
| Per-invocation receive publication | 0 | 0 |
| Long-lived receive publication | 0 | `inFlight` |
| Per-invocation receive storage return | Transfer's frame | 1 |
| Long-lived receive storage return | 0 | 0 |

Every range uses `mesh_instance_release`; the separate `mesh_shared_release`
helper and the TX reference-category branch are deleted. An empty range changes
no counter. Thus receiving and later returning a constant cannot decrement
frame 0 twice. The receive record contains a completion count instead of a
shared flag; buffer return has its own declared count. Neither controls posting.

Received invocation identity is recorded in the buffer at publication and in
the owning frame when its receive storage returns, before releasing that frame
reference. The retained reference prevents successful conclusion before this
store. Buffer mappings, counts and presence are reset before frame availability,
preserving the previous local reuse order. This does not establish remote reuse
safety or conclude failure cancellation.

The buffer record grows from 40 to 48 bytes (8 bytes per arena row). RX records
remain aligned 32-byte records; the SEND count occupies existing padding in the
256-byte native request record. No queue, notification, payload copy or live
counter is added. Maintained source changes from 2,451 to 2,454 lines across the
same eleven files. Source and build checks establish this accounting change;
there is no new runtime or latency result.

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
ABI 65's send record contains the exact start and length of its frame-reference
range. Receive setup assigns the buffer's completion binding and count; allocation
does not infer a frame relationship. Transfer return therefore does not derive
frame ownership from the invocation label.

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

For invocation i and resident capacity V, result reads the status at i mod V.
Each new traversal uses a new 32-bit invocation index; the index is not a reusable
frame handle. ABI 62 stores `{value, completed}` as one lock-free atomic 16-byte
snapshot. `completed` is i + 1 for the latest successful traversal, zero before
success, and `UINT64_MAX` for completed static-only work. `value` holds the
success/pending/fault code. The query returns success when `completed` matches;
otherwise it returns the fault or busy. It does not query a second shared value.
The former ABI-61 code word let a fault erase a completed result, contrary to F10.

Submission supplies the invocation before publishing roots. RX uses its prepared
frame index after publishing the completed section, stores the invocation, then
releases the section's producer reference. That reference prevents the section's
return and frame-reference decrement from preceding the store. The final frame
reference publishes that invocation's success. Frame and receive records remain
32 bytes; the larger status fits the frame's former spare space. Both participants
need ABI 62 because the header and status layout change.

A successful conclusion makes one strong compare-exchange and cannot overwrite
an error. A fault preserves the completed marker. If it races with a success,
it retries using the snapshot returned by the failed compare-exchange, retaining
the success while recording the fault for pending work. Another fault terminates
that retry with the first error preserved. These are local error-record updates,
not transport retries or waits for a peer. A completed resident invocation remains
successful across link loss. Explicitly submitting another invocation reassigns
that slot, so callers retain any longer-lived result history themselves.

The Core ML chain's driver observes its configured window on submitting and
passive ranks. A passive explicit `submit` checks the program fault before
returning; received numerical work still requires no local submit. The compiled
ARM64 query uses `ldp` and an acquire fence, then register selection. The static
assertion requires that snapshot to be lock-free. This is code-generation evidence,
not a timing claim. N1's cross-participant lifetime bound and cancellation remain
unfinished.

### Direct SEND record operands

In `01c1774`, configuration places the page-entry address, registered
span-array address, sequence-word address, queue pair and block geometry in the
existing SEND record. These values were already fixed for the record's lifetime.
The TX loop no longer follows the mapping header to reconstruct page and buffer
addresses, follows the device object to find registration spans, or indexes the
provider's queue-pair array. The selected span's address is already the shared
tag-word address, so it also replaces the separately derived tag mapping.

The runtime data flow is now: ready index → SEND record → canonical page value
→ registered span; the sequence value is read through its prepared address;
the tag and request address/key are written and the native post is called.
The page/registration and sequence reads remain explicit. This removes object
traversal; it does not establish the complete one-record contract.
The 256-byte SEND allocation and 128-byte alignment are unchanged. Added fields
use existing space; the asserted common native header still fits the first
128 bytes. The range-end field is now at byte 80. No interface or shared-region
ABI changes in this step.

With the same `cc -fblocks -O2 -S` invocation, compare the generated
`link_send_ready` instructions immediately after calculating the SEND-record
address through, but excluding, the native indirect call. The previous source
has 39 instructions, including 20 load instructions and two unsigned divides;
the replacement has 25 instructions, including 14 loads and one divide. Paired
loads count as one instruction, not one scalar access or cache-line fetch.
The native provider dispatch itself is included in both preparation intervals.
The nonempty-path stack frame decreases from 80 to 64 bytes; the empty path
now returns before allocating a stack frame. These are generated-code counts,
not cache-miss or latency measurements. Full H2/H3/H7 completion remains open.

Maintained library source changes from 2,680 to 2,686 lines across the same
eleven Swift/C/header files. Documentation cleanup is reported separately.
Strict C diagnostics and the existing bridge build pass; no runtime test or
bridge deployment is performed.

### Memoized native dispatch

The SDK's inline post/poll functions traverse handle → context → provider
function on every invocation. Those functions and handles are fixed for the
native queue's lifetime. Setup now resolves the function targets once; progress
reads them beside their arguments. Queue creation, pairing and teardown remain
the existing native operations. Re-pairing replaces the prepared targets before
starting progress, and teardown still joins progress threads before destroying
the handles.

The former separate queue-pair and completion-queue pointer arrays are replaced
by one 64-byte aligned record per queue pair: pair, two completion queues, two
poll targets, send target and receive target. Its size and alignment are asserted.
The enclosing link allocation is aligned accordingly. This costs 64 bytes per
queue pair instead of 24, or 320 additional bytes for eight pairs, plus enclosing
structure padding. The SEND record memoizes its post target in existing space;
its allocation remains 256 bytes aligned to 128. Queue/range indices, address
inputs, target and common native request header fit the first 120 bytes. The
range end is at byte 72. Completion-only reference fields follow the native
request. There is no additional runtime descriptor or forwarding function.

These are the regression boundaries at the native call site:

| Operation | Function-target and handle retrieval | Forbidden reintroduction |
|---|---|---|
| SEND | Pair and post target from one prepared SEND header | Pair → context → operations lookup |
| RECV | Pair and receive target from one prepared queue record | Pair → context → operations lookup |
| POLL | Completion queue and poll target from one prepared queue record | CQ → context → operations lookup |

With `cc -fblocks -O2 -S` on the same ARM64 compiler, SEND now loads its pair
and function together with `ldp` and calls the target with `blr`. Polling loads
its handle and target from the same queue record using independent addresses.
Receive posting likewise reads its handle and target directly. The previous
25-instruction SEND preparation interval is now 23 instructions, with 12 load
instructions instead of 14, using the same interval defined above. The 64-byte
nonempty-path stack frame is unchanged. These counts describe generated code,
not elapsed time, cache misses or the complete native provider body.

`make -C rdma native-audit` compiles the three existing C implementations to
optimized assembly under `.build`, including their compile-time layout checks.
Review must follow actual load dependencies; neither a symbol scan nor the
total instruction count certifies the one-record property. No runtime check,
timer or alternate implementation is introduced. The native-target lookup is
closed; SEND still reads the canonical page, registered span and sequence,
and RECV still resolves its returned-page index and prepared request. Those
remaining accesses are not hidden by this regression boundary.

Maintained library source is 2,704 lines across the same eleven Swift/C/header
files, up from 2,686. The Makefile gains four build-rule lines and one phony target.
Documentation changes are separate. Strict compilation, assembly generation
and the existing bridge/C/Swift library builds pass; no workload or deployment
is performed.

### Direct Swift dispatch

The `MeshInvocation` heap object, its C callback trampoline and its separate
per-function disposer are deleted. The setup value (`MeshLaunch`) begins with
the 32-byte pair of ordinary launch and optional rearm functions. The existing memory owner
retains these values until native workers and callbacks finish, then releases
the array once. Dispatch never indexes or reads that ownership array.

Each 32-byte consumer record contains code, context, call address, invocation
mask and the end of its contiguous use range. The duplicate operand-array
address and input count were removed from that record; they already occupy the
same 64-byte call record as the required dependency count. The generated ARM64
worker loads code and context together, puts the context in `x20`, and passes
the call address in `x0`. After updating the dependency count it executes `blr`
without reading operand pointers, input counts or frame indices. The selected
native function reads only the fields it uses; resident launch reads the frame
index and commits its prepared command. There is no intermediate invocation
object, block conversion, launch retain/release or shared atomic increment.
The supplied function and its backend still have their own captures and work;
this does not claim to delete those native operations.

A local submitted count replaces the pre-launch atomic increment. Native
completion still increments one completion counter after output publication
and ownership cleanup. Only shutdown compares the counts. This deletes one
contended atomic RMW per invocation; it does not relocate it to another thread.
The duplicate `running` acquire/branch around arrival drainage is also deleted.

Before this change the maintained library had 2,833 lines across
`swift/*.swift`, `rdma/*.c` and `rdma/*.h` (eleven files). The new total is 2,830.
The material reduction is the removed runtime object access, callback adapter,
atomic operation and 64-to-32-byte consumer record, not a halving of source.
Documentation is counted separately. Shared wire layout remains ABI 76;
the native library callback ABI changed and requires dependent libraries to
rebuild. No workload or latency measurement was performed.

This closes the invocation-object lookup and launch-lifetime increment only.
Stream probing, indexed receive placement, contiguous materialization, backend
command-buffer creation, inline transport reclamation and N1/R2 remain open.
The [audit follow-up](h-audit-2026-09-16.md#follow-up--native-function-dispatch)
records these limits instead of marking the entire H group complete.

### Memoized numerical completion

The existing 64-byte call record now stores the arena pointer and input/output
counts in its former 16 bytes of padding. Setup already knows all three values.
Completion reads that record, traverses its actual output range and publishes
the rows. It no longer follows call → function → calls → context to discover
the arena before the first output publication. Function/worker ownership is
read afterward for the existing reference releases; no extra reference or
completion event is added. Record size, alignment and array stride stay 64.

ABI 76 also binds the [terminal publication operands](#direct-publication-operands).
The optimized output loop loads each prepared pointer and calls `mesh_publish`,
without a row-to-target lookup. A compiler-enforced tail call transfers to
`mesh_call_cleanup` after all publications. Cleanup preserves input releases,
output producer releases, error/zero-output returns and active-reference release
in their existing order, on the same thread. It introduces no queue or scheduler.

Separating cleanup reduces the generated pre-publication stack frame from 112
to 64 bytes. The cleanup frame is 112 bytes and starts after the publication
frame has been removed; maximum nested depth does not increase at that boundary.
The extra prologue/epilogue work is explicit: this moves lifetime-related spills
off the handoff path, not a claim that total stack traffic falls. Publication's
own generated code remains frameless. Preserve both the tail-call boundary and
the first lifecycle load's position in native-code review. Operand, publication
and stream reads still count against the full budget.

### Direct receive completion addresses

The ABI 69 change used each native RECV's local canonical tag address as `wr_id`.
The [prepared forwarding change](#prepared-receive-forwarding) below now supplies
the registered alias itself, preserving the direct tag read.
The completion reads that address directly. The identifier is local to the
native queue; it is not transmitted to a peer. Both the tag alias used by the
registered receive and the canonical tag address name the same shared backing.
Setup stores the first tag address, page-size shift and block size beside the
receive queue. For returned tag address t, first tag address t0, page size 2^s
and block size B pages, the canonical physical page is `((t - t0) >> s) * B`.
The handler stores that page through its already prepared destination pointer.
No runtime arena-header lookup is needed to locate the arriving tag, and no
new binding table, wrapper, allocation or receive-data predicate is introduced.
Queue-local constants add 16 bytes per queue; native receive requests and
destination records retain their 64-byte and 32-byte layouts. ABI 69 is unchanged.

The native posting loop is shared by initial posting and returned-page posting
and forced inline. On a return it attempts to drain the native request queue
before resetting the logical row or releasing instance references. Native
capacity refusal still leaves the head request in place and permits draining
other returned pages; it introduces no software credit or waiting condition.
The final-reference event already ended the page's prior uses. Reposting does
not require its logical row to have finished bookkeeping, and the same receive
thread processes later native completions after that bookkeeping.

`native-audit` shows the completion identifier followed immediately by the tag
load, without the old arena-header loads or division. It also shows the native
repost before reset/refcount instructions and no out-of-line posting helper.
The bridge build and strict assembly generation pass. These are source and
code-generation results, with no workload execution or latency claim. The
source total is 2,713 lines over the same eleven maintained library files,
up from 2,704. The full receive path still includes publication, native request
selection and return bookkeeping; these changes do not close H3/H5/H7 or N1.

### Prepared SEND operands

ABI 71 removes `pages`, `spans`, `offset` and `block` from the SEND record.
Each native request has its own 256-byte record with its SGE, queue pair,
post target, tag row and native WR. The SGE and common native header occupy
the first 88 bytes, within the asserted 128-byte alignment. Native requests
are complete before their index reaches the posting loop. Posting reads the
prepared address, writes the supplied invocation/tag row and calls the native
target. It performs no page lookup, registration lookup, address binding or
request reconstruction, including when retrying after native refusal.

For a declared send e with K_e native requests, setup assigns an interval
`[I_e, I_e + K_e)`. One 16-byte queued range contains its current index, end
and invocation. A successful post advances that index and rotates unfinished
ranges; native refusal leaves the request in place. This preserves immediate
drainage and per-transfer chunk order without expanding each publication into
K_e separate pending entries. The compiler inlines the posting loop into its
callers; no additional posting call frame remains in the optimized ARM64 build.

The local binding cases converge on that same record format:

- Locally allocated backing is known during configuration. Its registered
  address/key are written once into the native record there.
- A received backing becomes known from its native completion. Its prepared
  32-byte destination record also contains a range of 16-byte scatter operands.
  Each names an outgoing SGE and its already resolved registration key.
  Completion selects the prepared range for the received region and stores
  its own registered address plus the inline key directly into that SGE. It publishes
  the partial after binding its blocks. No canonical-page reread, peer lookup,
  queue selection or tensor-function dispatch occurs in this scatter.

Every configured outgoing link and queue is represented, including links using
different registration keys. Required local devices are registered during
configuration. All links' native-record allocations and immutable range fields
are prepared before any controller/progress thread starts. Later queue setup
writes the pair, post target and length; receive binding writes address/key.
These are separate fields. Publication's release/acquire handoff precedes the
TX read. The existing buffer references protect the backing and binding until
all declared sends and numerical uses finish. No extra reference protocol or
runtime configuration-readiness flag is added.

All K_e native requests carry I_e as their completion identifier. Their
completions decrement the existing group's remaining count in its first record.
The last completion releases the same one buffer reference and one declared
transfer contribution as before. Per-instance completion counts and numerical
caller interfaces are unchanged. Private record indices now count native
requests, which requires the ABI 71 agreement between client and bridge.

The memory cost is explicit: native records use `256 * sum(K_e)` bytes instead
of `256 * number_of_sends`; each receive-to-send binding uses another 16 bytes
per receive-pool region, as accounted for below.
Each queue reserves 16 bytes times its existing power-of-two range capacity,
instead of four bytes per queued transfer. These are metadata allocations;
operand backing is unchanged and no tensor payload is copied. Maintained
library source grows from 2,717 to 2,779 lines across the same eleven files.

Strict C compilation, optimized assembly generation, the bridge, the four
existing callers and the engine Mesh library build pass. Generated posting code
loads its address, tag row and native target from the same prepared header and
its invocation from the queued range. It contains neither the former page load
and divide nor the registration-span load and SGE rewrite. Receive forwarding
reads its scatter operands once per binding; its registration-span lookup is
removed by the following change. Publication streams, native
operand view selection, full H1–H7 latency evidence and N1 realization remain
unfinished. No runtime workload or deployment accompanies this change.

### Direct TX publication

The TX event already identifies prepared native requests. Publication attempts
every request directly until one is refused by the native provider. It writes
the request's existing tag and calls its prepared native target. A successful
post advances directly to the next prepared request. It performs no pending-ring
enqueue, head lookup, range reload or rotation. This applies to every declared
send, link and queue, including a single-request FFN partial.

Only `ENOMEM`/`EAGAIN` enqueues the unaccepted suffix, immediately followed by CQ
progress and retry drainage. The existing CQ progress body owns that drainage;
the separate `link_send_ready` function and its unconditional invocation after
every publication are deleted. Queued intervals retain their rotation and
refusal behavior. One forced-inline `link_send_request` body serves both paths.
It adds no call frame, request construction or capacity predicate.
Transient refusal can cost an additional unsuccessful native attempt for a new
publication; this is accounted for rather than hidden behind a software gate.

There is at most one unfinished interval per declared live send. A fresh
publication has none queued, and direct posting either finishes it or enqueues
its sole remaining suffix. Enqueue therefore preserves the existing E-range
capacity bound, with no new reservation or allocation. Requests within an
interval retain their order. Different intervals can reach posting in a
different interleaving; indexed receive matching already supports that order.
This does not establish N1's cross-participant lifetime bound.

Setup binds each of the existing two dedicated threads directly to its send
or receive entry point. The shared direction-dispatch function, runtime direction
field and worker-to-link wrapper are deleted. Poll/post targets have constant
direction indices. The send thread retains the native request-array base for
its lifetime, including completion processing and retry posting. Controller
teardown still joins both threads before changing that storage.

Each poll requests at most one completion, so its consumer uses that one output
record directly rather than a completion-index loop. The verbs `bad_wr` arguments
are output storage whose values Mesh never reads; their per-post null stores
are removed. Native error handling and completion-driven ownership are preserved.

Optimized ARM64 holds the SEND-array base in a callee-saved register. From the
event index to the first post, it reads the prepared native header, writes the
tag, and calls the stored function. Pending-ring loads/stores follow that call.
The helper has no emitted call frame. Stream polling and the later range queue
still have their recorded costs; no whole-path latency result is claimed.

This is private bridge code with the same ABI 76 wire/shared layout and queue
reservations. It removes the worker wrapper storage and adds no allocation.
This transport change alone adds 27 maintained source lines. The accompanying
deletion of contribution typing removes 24 Swift lines, giving 2,830 → 2,833 lines
across the same eleven files; documentation changes are excluded. The
[dependency recount](h-audit-2026-09-16.md#follow-up--direct-first-send-and-deletion-of-contribution-typing)
records the net first-post depth reduction separately from that source count.
Strict C diagnostics, the bridge and native assembly generation pass. No workload
or fleet deployment was performed.

### Prepared receive forwarding

The forwarding descriptor holds `{destination SGE, registration key}` in 16
aligned bytes. The former registration-span pointer and its dependent read are
deleted. All local devices register the same wire aliases; keys differ by device
and registration region. Configuration therefore emits one contiguous binding
range per region intersecting the receive queue's physical pool. It resolves
each key through the destination device during setup. Every configured outgoing
link and queue participates; no single-peer or single-region case is assumed.

The native RECV identifier is its registered SGE address. Completion reads the
tag there directly and can store that same address into outgoing SGEs. It does
not reconstruct an address, look up a registration, follow a native request, or
read the canonical page table to bind forwarding. The selected binding range
contains the terminal destination and key together. The stores still precede
publication, using the existing release/acquire handoff and buffer references.

The registered alias map already places each region in a 4 GiB bank and each
physical block in `B + 1` pages, one tag page followed by B payload pages. Let
`s = log2(page_bytes)`, `d = B + 1`, and `a` be the returned registered tag address.
The tag occupies the last eight bytes of its first page. Consequently
`n = low32(a) >> s = k d`, where k is its block index within the region.
Setup stores `mu = ceil(2^32 / d)`. Then

```
k = (uint64(n) * mu) >> 32
r = high32(a) - first_alias_bank
b = r * region_blocks + k
page = first_region_page + b * B
bindings = record.sends + r * record.send_count
```

This quotient needs no division or corrective branch: writing
`d mu = 2^32 + e`, with `0 <= e < d`, gives
`n mu / 2^32 = k + k e / 2^32`, and `k e < k d = n < 2^32`.
The first region may start before the queue's pool and the last may be partial;
the stored bases and the pool's actual region interval cover both cases.
No address or identifier is exchanged with peers by this local convention.

The destination-record pointer and placement geometry occupy a 40-byte extent
aligned to 64 bytes after ABI 72's addition of the prepared pool index.
Static assertions retain that boundary,
the 32-byte destination record and the 16-byte forwarding descriptor. Optimized
ARM64 code reads `wr_id` then the tag, uses shifts/multiplies for the page, and
loads each destination/key pair before its two direct stores. It performs no
registration-table load, runtime division, helper call or added readiness test
in this binding interval. The explicit scatter loop remains proportional to
the declared forwarding destinations.

For queue q with D_q declared forwarding requests and R_q registration regions
intersecting its pool, binding storage is `16 * max(1, D_q R_q)` bytes rather
than `16 * max(1, D_q)`. This replicates keys by region, not physical page.
Receive state is 128 bytes, now aligned to 64, versus the original 104 bytes; no tensor data
is copied or additional event posted. Native requests and public/shared layouts
were unchanged by this forwarding change, which retained ABI 71; ABI 72 extends
the canonical entries as described above. This forwarding step brought the
eleven maintained library files to 2,790 source lines, up from 2,779. Strict native compilation, bridge compilation
and generated-code inspection pass; no runtime workload or latency measurement
was performed. Publication stream probing, numerical operand views, lifetime
realization and the complete H1–H7 paths remain separate unfinished work.

### Prepared native requests

Setup prepares matching native request extents under Apple's
[TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
frame-count contract. Posting submits those extents unchanged.

For transport payload capacity C and a queue's declared logical transfer lengths
N_t, setup chooses L_q = min(C, max_t N_t) + 8 bytes for both ends of that
direction. A section of N bytes occupies K = ceil(N/C) requests, each of L_q
bytes. The posted SGE byte total is K L_q: N logical bytes, 8K tag bytes, and
K(L_q - 8) - N padding bytes. The receiver's prepared request has the same
extent for every arrival, independent of which producer finishes first. This is API byte accounting, not measured wire time.

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

For an N-byte value, page size P and transport capacity C = BP, setup computes
K = ceil(N/C). Each resident slot owns K consecutive logical entries and KB
physical pages. Slot v starts at row r = first + vK; chunk j always occupies
`mesh_page(m)[r + j]`. The ranges for distinct slots are disjoint. The buffer's
`rows` count governs metadata reclamation, while `pages` governs backing
reclamation. ABI 58 reduced metadata ownership from VKB rows to VK for V slots.
The bridge still reserves its metadata arena, so this is not a claim that
process RSS falls by B.

TX and RX address entry j through that prepared list. An indexed operand caches
the list pointer; a byte offset b uses entry floor(b/C) and relative offset b mod C.
The contiguous CPU/Core ML and Metal placement paths read the dependency's same
list. No allocator, packing solver, mapping syscall or strategy selection runs on
receipt or scalar consumption. Retain/release addresses one buffer directly;
setup and destruction enumerate section slots explicitly. The reference-count
range walks and their assumption that payload extent determines descriptor
positions are removed.

ABI 63 deletes `buffer.mapping` and `mesh_buffer_pages`: both initializers stored
exactly `page_off + row * sizeof(uint32_t)`, with no alternate mapping or mutation.
Direct row indexing therefore selects the same canonical entry for every chunk.
It removes the descriptor load from TX address resolution, receive-page return,
reclamation and `mesh_row_page`. RX completion and indexed numerical operands
already retain prepared entry addresses; they gain no additional runtime-load
reduction from this deletion. No payload moves and no page-table entries disappear.

Buffer metadata shrinks from 48 to 40 bytes, saving eight bytes per arena row
before region alignment. Operands remain 48 bytes. The shared-memory layout
changes, so both participants and their clients require ABI 63. Runtime page
loads, wire tags, hierarchical notices and contiguous-input copies remain;
this deletion does not establish H1–H7 or a measured latency result.

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
