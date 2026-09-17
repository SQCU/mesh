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
active-call count keeps that worker alive through native callbacks after destroy
clears `running`. Worker exit drops its program reference. There is no additional
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

### Costs and remaining work

For P compiled streams an empty pass performs P acquire probes, rather than E
individual-location probes. P is not guaranteed independent of graph size.
A probe still reads private input state and then its shared slot. An enqueue
still reads a prepared stream descriptor, updates its producer position and
stores its index. These costs, possible cache-line sharing between packed
stream descriptors, and polling latency remain; grouping is not free progress.

After a successful dequeue, TX and numerical dispatch directly address the
terminal record array. Optimized ARM64 output shows `index << 8` followed by the
SEND range-end load at byte 40, and `index << 6` followed by the use range-end
load at byte 40. The former offset-table loads are absent from those paths.
This is source/assembly evidence of one removed dependent table access per
dispatch, not a measurement of cache misses or end-to-end latency. ABI 71 TX
uses the prepared backing address/key and writes the wire tag. Numerical launch still accesses the declared dependency count and sequence
value and records its active lifetime. Cold return processing retains its
row-to-call lookup. The operand changes below remove address refresh and both
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

### Operand addresses and sequence values

Address binding is not a separate runtime phase. Let A be the registered arena
base, s its page size, B the transport block size in pages, Q = sB, and p the
canonical page-index array prepared for an operand. Its byte address is

```
k = floor(offset / Q)
address(offset) = A + s * p[k] + (offset - k * Q)
```

The receive completion stores its actual page at the prepared p[k] location.
The 48-byte operand carries A, p, s, B, extent, logical row, frame index and a
pointer to its call's sequence value. There is no cached first-page identity or
cached materialized first-page pointer. `mesh_operand_address` performs the
indexed gather and affine arithmetic directly. The Swift `data`, `page` and
`invocation` accessors preserve their caller spelling while using those values.
Native `MeshBindings` select the already constructed view using the actual page
index; Core ML objects, output backing and model invocation remain prepared.
The dispatch-time remote-input classification, page load and two copied fields
are deleted. Indexed reads previously loaded p again after that refresh; now
there is one canonical lookup at the actual access. Local raw `data` access also
reads its canonical first entry, where the old cached pointer needed no such
read. This is an explicit cost of the uniform descriptor, not hidden zero-cost
addressing or completion of H6.

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
when necessary, restores that frame's pending template, and releases one frame reference directly. Zero-output functions emit the same return through their metadata row.
A native error concludes status separately; complete failure cancellation is R2.

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

### Memoized numerical completion

The existing 64-byte call record now stores the arena pointer and input/output
counts in its former 16 bytes of padding. Setup already knows all three values.
Completion reads that record, traverses its actual output range and publishes
the rows. It no longer follows call → function → calls → context to discover
the arena before the first output publication. Function/worker ownership is
read afterward for the existing reference releases; no extra reference or
completion event is added. Record size, alignment and array stride stay 64.

The optimized ARM64 code reads the arena at byte 48 and both counts at byte 56,
then calls `mesh_publish` for the output rows. Its first function-record load
appears after the publication loop. Preserve this ordering in the native-code
regression review. The output descriptor reads and `mesh_publish`'s own loads
remain part of the total publication budget; this change does not certify them
or hide them behind the callback boundary.

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
page = first_region_page + r * region_pages + k * B
bindings = record.sends + r * record.send_count
```

This quotient needs no division or corrective branch: writing
`d mu = 2^32 + e`, with `0 <= e < d`, gives
`n mu / 2^32 = k + k e / 2^32`, and `k e < k d = n < 2^32`.
The first region may start before the queue's pool and the last may be partial;
the stored bases and the pool's actual region interval cover both cases.
No address or identifier is exchanged with peers by this local convention.

The destination-record pointer and all six geometry constants occupy the first
32 aligned bytes of the receive state. Static assertions retain that boundary,
the 32-byte destination record and the 16-byte forwarding descriptor. Optimized
ARM64 code reads `wr_id` then the tag, uses shifts/multiplies for the page, and
loads each destination/key pair before its two direct stores. It performs no
registration-table load, runtime division, helper call or added readiness test
in this binding interval. The explicit scatter loop remains proportional to
the declared forwarding destinations.

For queue q with D_q declared forwarding requests and R_q registration regions
intersecting its pool, binding storage is `16 * max(1, D_q R_q)` bytes rather
than `16 * max(1, D_q)`. This replicates keys by region, not physical page.
Receive state is now 128 bytes, aligned to 32, versus 104 bytes; no tensor data
is copied or additional event posted. Native requests and public/shared layouts
are unchanged, so ABI 71 remains. The eleven maintained library files total
2,790 source lines, up from 2,779. Strict native compilation, bridge compilation
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
