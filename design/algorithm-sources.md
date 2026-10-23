# Algorithm references

The task is [GOAL.md](../GOAL.md). Section names are the anchors the source cites.

The operator, 2026-10-22: "connection to the prior literature is important for understanding what are choices and
what are strictly dominating superior methods we must use".

The operator, 2026-10-17, of deleted implementations: "everything which is an additive rewrite (we write another
caller which adheres to interfaces compatible with the service and other headers... then we reimplement the previous
algorithms to match the adherent calller... then we're done?) should be recoverable subagent-parallel"; "very little
is 'dead' or 'needs to be completely rewritten' based on an early estimate of what keeps getting 'deleted as dead'."

## Program

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
reference for higher-order numerical calls over indexed tensor operands.
The implementation realizes a finite storage capacity once. A section's
`first + slot * stride` selects its logical row; shared constants have zero
stride. Working ABI 55 carries invocation labels separately from reusable slots.
Submission publishes root indices into existing numerical-worker queues,
while consumers are indexed by operand publication. There is no function scan or repeated
realization. The [execution description](transport.md)
separates shared function metadata from each value's operands and uses.

**The Python surface** (`python/mesh`, restored from mesh `57628982^:python/mesh/` onto the current library; program
type B, the library, its kernels and layers `examples/kernels.py` and `examples/nn.py` type C, design/deliverables.md
section 5): a program surface spelled as Pallas spells it (the Swift module of the same contract, swift/Mesh.swift,
was deleted in a8952de9). A `Program(backend, *, functions,
group, trace)` is SPMD: every member records the same calls, and `peer=` (a kernel's), a copy's two ends and a
collective's members place the work. Recording builds the graph; `realize()` (also `start`, python/mesh/__init__.py:795) prepares it;
the host then writes its inputs (`write`) and reads its results (`export`, `Result.ready`, `consume`). The firing
rule is the old executor's (design/deliverables.md L3): an edge per (writer, reader) pair of overlapping regions on a
member, a token on it when the writer finishes, a call firing when every input edge holds a token and every output
edge is empty (a region is reused only once its readers finished: storage of capacity one, Kahn and MacQueen 1977's
bounded channels), a host write a token on the edges of the block it fills, a call whose inputs are all constants
firing once. Kernels run on one compute thread, the ready call earliest by row, then record. Operands are torch
tensors (`Ref.tensor`; `Ref.array` and `write` give NumPy views of CPU ones), each backing allocated where a member's
call touches it, in the node's bridge's region where a crossing of that member reads or writes it
(`torch_mesh.backend.blank`, over `mesh.lease`). Every wait ends at an event (a call completing, a failure, `close`), never at a time;
a failure is a value (`Program.report`, `MeshError`) and the caller makes the program again. The package's import
name is `mesh`, the name `rdma/mesh.py` also takes: importing it loads that module from the same checkout and carries
its names (`ffi`, `lib`, `lease`, `current`, ...), so `torch_mesh` and `mesh_mpi` find what they import; a program
puts the checkout's `python/` ahead of `rdma/` on its path before it imports torch (the examples do).

## Program.tensor

The JAX authors, [BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html),
and Apple [mmap](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html):
references for indexed sections and virtual mappings of actual shared backing.
The Swift accessor `mesh_operand.load(at:as:)` and the C operand ABI under it (ABI 63-75: `mesh_operand_address`,
`mesh_operand_page`, `mesh_backing_bind`, `mesh_operand_view`, `MeshBindings.index`, `mesh_operand_data`) are in no
source; swift/ was deleted in a8952de9. The Python surface's views and blocks are below.

George E. Collins, [A method for overlapping and erasure of lists](https://doi.org/10.1145/367487.367501)
(1960): reference counting. The operator, 2026-09-15 (UTC): "have a subagent literally write a refcounter which
doesn't rely on users of buffers specifically and literally inlining some kind of 'please say im done using the
resource' explicit free like we're in c."; 2026-09-14: "remember that garbage collection in this case simply means 'releasing' pages into
the 'pool of pages you can write to if you wanna lol' without even needing to zero them lol".
The counted ownership that cited it, the Swift module's `TensorPart` and `MeshMemory.sections` and libmesh's
reference counts, free bitmaps and receive pools (ABI 49-66), is in no source. A page's lifetime is now its lease: a
client's rows are its own until it returns them (`mesh_lease_return`, rdma/mesh-call.h:22) or its process ends, and a
program that reuses a buffer guards it in its own source (design/transport.md section 14;
[extents-and-liveness.md](extents-and-liveness.md)).

The Berkeley/Apple [mmap specification](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/mmap.2)
provides shared mappings of file-backed pages. Apple's
[pshm_mmap implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/posix_shm.c)
maps the selected object offsets with copy-on-write disabled. `wire_map` uses those mappings to
give the device a payload-plus-tag address sequence and the numerical function
a dense address sequence over the same payload pages. Aliases are fixed at bridge
setup. Runtime receive assignment changes page indices, not virtual mappings.
Metal buffers cover each operand's page-aligned window, using Apple's
[no-copy buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)).
The whole-arena Metal buffer and transport-capacity query have been removed.

In the Python surface a `Ref` is the old native `View` over a backing: (offset, rows, columns, row stride, column
stride) and `origin`, its corner in the tensor it was cut from. `slice`, `T` and `broadcast` (stride 0) make new
views of the same backing; two views overlap where their element intervals do (the old row-interval sweep). A
`Tensor` is its blocks (`block_shape`, `grid`), each its own backing, or one `contiguous` span whose blocks are views of
it; `region` cuts within a block, `broadcast_to` and `T` view every block. A whole block is the unit of host
publication (`write`) and of `constant`.

## MeshError

Saltzer, Reed and Clark's end-to-end argument, cited below, places recovery with the caller. The Python surface's
`MeshError` (python/mesh/__init__.py:65) is an exception whose `kind` is `link`, with the member across it, or
`function`, with the call's record index, each with an errno (python/mesh/__init__.py:808); a wait on a failed program
raises it. Its L5 refusal is a TypeError at recording (`Partial`, [Program.kernel_call](#programkernel_call),
python/mesh/__init__.py:514), so the [restored L5](deliverables.md#l--library-core) holds there. The Swift
`MeshError` it was restored from (swift/Mesh.swift, 71bced9e) was deleted with swift/ in a8952de9; its setup error
`partialOperand` had gone at 3bd1eaa.

## Mesh.result

Saltzer, Reed and Clark, [End-to-End Arguments in System Design](https://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)
(1984), motivates keeping recovery with the caller. Mesh records native errors;
it does not retransmit an operation or make its consumers wait for recovery.
ABI 62's status snapshot and `mesh_calls_result`, which the Swift decoder read, are in no source
(`mesh_calls_result` deleted in f531113f, the decoder with swift/ in a8952de9).

Submission records the invocation before publishing roots. RX records it in the
buffer before publication, then copies it to the owning frame on storage return,
before releasing the frame reference declared at setup. That ownership event
cannot release the frame's final reference before the invocation store.
The last reference publishes success with that invocation. No passive-arrival
busy store, extra notification or transport permission check is added.

`mesh_result_conclude` never overwrites an error. Normal success makes one strong
compare-exchange. Fault publication preserves `completed`; if its compare-exchange
loses to a concurrent successful completion, it retries with that observed
snapshot so the completion is retained and the fault is not lost. This retry is
confined to error reporting; it neither retries a transfer nor waits for remote
work. An existing error ends it. Consequently a link error can report failure
for pending work while the last completed invocation still returns success.
A new explicit submission reassigns its resident slot, so the library keeps no
older result. This does not establish N1's cross-participant ownership
bound. A passive rank's explicit `submit` now reads the program fault before
returning success for its lack of local roots; reception still needs no submit.

The existing Core ML chain now keeps a bounded number of predictions in flight and observes each
invocation's result before reusing its slot among them, including on ranks whose
work arrives solely from peers. It reports native failure through the same
public `Result` path. This adds no library scheduling path or numerical kernel;
it exercises the status API in the existing arbitrary-stage producer/consumer
composition. No runtime or killed-peer result is claimed.

Collins's reference counting, cited above, supplies the ownership mechanism.
Setup counts numerical and transfer references per frame. Final function/transfer
release decrements the frame's atomic count directly; zero restores its recurring
template, concludes status and marks it available. ABI 65 represents completion
ownership as a prepared start and count. The same `mesh_instance_release` walks
every range; the separate shared-release helper is deleted. A long-lived
transfer releases its initial frame references at completion and has no frame
reference to release when its buffer is eventually returned. No receiving-data
predicate selects the reference category. The lifecycle thread and its
event rings/arena storage are deleted: reference release needs no handoff to a
second poller. The native SEND record contains its frame-reference pointer and
count; receive setup assigns the publication and storage-return ranges. These are
applications of the same direct-address mechanism: completion does not reload
an invocation label and divide it to reconstruct ownership. There is no hash,
tombstone, label-to-frame search or
submission-association event. [Completion ranges](transport.md)
record the exact events, storage cost and remaining limitations. The current selected-frame admission and missing
whole-plan reuse proof are specified explicitly in
[frame identity and reuse](transport.md).
Neither the one-load result nor deleting a stack establishes safe unbounded reuse.

Native post errors other than capacity refusal, CQ poll errors and unsuccessful
work completions end that link's invocation after publishing its error result.
The TX/RX loops return on that native error; the controller joins them and closes
the QPs. It does not re-pair and replay the same invocation. The bridge remains
available for a newly realized client. The receive-side `failed` latch and its
publication guard are deleted; successful completions do not consult a software
health flag. Outstanding references remain owned until normal completion or
teardown, so publishing an error does not release device storage early.

Owner and worker references keep program metadata alive through native callbacks.
A forwarding-only rank needs no numerical worker. The frame-indexed representation
replaces the old assignment-order storage identities. N1 admission/capacity,
R2 cancellation and driver recovery remain unfinished; ABI 62 implements the
status snapshot but does not complete N2.

Jonathan Lemon and Apple's [kqueue interface](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/kqueue.2)
provide socket EOF and process-exit events. The link controller retains
the pairing socket for the connection's lifetime. The existing link controller
observes its EOF/error and the local client's `NOTE_EXIT`; either event records
a link failure and stops that connection. A native CQ/post error uses `link_stop`
to wake the controller through its preallocated user event. `link_close` shuts
down the socket, joins TX/RX, and closes QPs/CQs without deleting the process watch.
Registration precedes pairing; after connection teardown the controller keeps
observing the client until detach or bridge shutdown. `NOTE_EXIT`, or `ESRCH` at
registration, invokes the same `mesh_retire` used by detach and dead-client
replacement. The kqueue interface removes descriptor events on close and defines
`ESRCH` as a nonexistent target process. Its event queue remains open
until bridge shutdown, after every controller has joined, so a stop notification
cannot target a recycled descriptor. Event registration and notification use
`KEVENT_FLAG_IMMEDIATE`; only the separate controller's event observation blocks.
No heartbeat, elapsed-time failure inference, or healthy-path TX/RX socket query
is added. The first program and instance errors and each frame's last completed
invocation are preserved. The listener closes the moment it accepts, so a peer
dialing for its next realization is refused, and dials again, until this node's next
realization listens; nothing lands in a backlog that no pairing accepts. A connection
reset before the peers' first exchange is an abandoned attempt, and the pairing
connects again within its deadline.
This implements event observation in source, not complete R2 cancellation or
R3–R7 recovery. No timer detects an unreported silent partition.
No caller-death or cable-loss run accompanies this change.

`mesh_rows_alloc` now records the client owner for metadata rows as well as
operand rows. `mesh_retire` clears their row ownership even when `pages == 0`;
only actual operand buffers receive the closed-storage flag. Previously a dead
client's root and return rows had owner zero and survived replacement, leaking
row capacity across realizations. Normal client retirement or replacement of a
dead client now covers both. Retirement claims the exact old client identity
before clearing its state, uses a fresh generation and the retiring process's live PID during cleanup,
and publishes vacancy last. This prevents competing exit observers/replacements
from clearing a newer client's state. The mechanism and limits are documented in
[client retirement](transport.md).
Actual backing is retired at device close, as before.
Link error alone still does not cancel unissued numerical uses
or rearm the same program: those R2/R4 steps remain unfinished.

## Index hand-off

Papadopoulos and Culler's
[Monsoon: an Explicit Token-Store Architecture](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-F03/handouts/papers/p398-papadopoulos.pdf)
(ISCA, 1990), section 2, assigns graph arcs to explicit frame locations rather
than discovering matches dynamically. Mesh applies that principle to the
already realized use/send indices: `mesh_transfers_prepare` prepares each
program's destination records from its SEND lists (its targets, a section of
its slot sorted by row, `mesh_row_targets`; MESH_VERSION 111, where they had
been a 1,088-byte table on every row of the region), publication sends the
final index, and dispatch directly indexes the terminal record array. Numerical offset tables and TX cursor arrays are setup
scratch. Mesh's ring software and compiler are not Monsoon hardware.

ABI 76's [publication operands](transport.md)
apply the same explicit-address principle at the producer: the output and
receive record hold the terminal publication pointer. A row's publications are
a run of its program's targets, sorted by row and found by bisection at
preparation (`mesh_row_targets`); no lifecycle buffer is read, and stream access
still has its separately accounted cost. Until MESH_VERSION 111 the counts were
inline before a per-row target array, the first target and counts in one aligned
32-byte prefix.

The [Disruptor paper](https://lmax-exchange.github.io/disruptor/disruptor.html)
(Thompson, Farley, Barker, Gee and Stewart, 2011), sections 2.2 and 3.1–3.3,
explains single-writer ownership and preallocated circular storage. Mesh uses
that principle without importing a consumer gating strategy. Independent
writers get independent streams; a writer increments its own position and
release-stores its prepared event. The consumer clears slots after acquiring their values.
Declared lifetimes provide the reuse order; the producer does not read fullness.

ABI 73 puts each stream's cursor, mask and slots in one aligned extent, applying
the same preallocated circular-storage mechanism without a separate slot-address
field. `mesh_stream_bytes` supplies the setup layout; reader initialization
walks those extents once and records their terminal slot pointers. Shared
producer streams, private reader arrays and numerical-worker mutable state are
separated at 128-byte boundaries. The [allocation analysis](transport.md)
records the substantial reservation increase, actual layout formulas and the
unchanged O(P) probe count. This is a layout specialization, not a new queue
algorithm or a claim that padding alone solves handoff latency.

ABI 70 carries the publication's invocation value alongside its final record
index in the same lock-free eight-byte slot. TX copies that value into the
prepared SEND record. A numerical use selects it with its setup-defined mask,
preserving the current call label for a shared input. No destination reads the
publishing buffer merely to recover this already supplied value. The persisted
receive-return label is written after publication; the producer's existing
reference protects that write. This extends the event operand, not the queue's
admission or synchronization behavior.

`mesh_events_prepare`, `mesh_event_compare`, `mesh_event_root` and
`mesh_event_remap` implement Mesh's setup-only assignment. They group known
writers and dependency-ordered native completions, sort binding indices,
allocate power-of-two ring spans and replace provisional handles with final
shared-memory offsets. The grouping is greedy and makes no optimality or
union-find complexity claim. Scratch parents and mappings are discarded.
`mesh_event_reader_init` prepares private reader addresses before progress;
`mesh_event_take` polls these independent streams, not the tensor graph.

TX and numerical workers keep their prepared reader headers as thread-local
values. Each reader has exactly one consumer; its input-array base and count
are immutable for the thread's lifetime, and its cursor has no outside reader.
The numerical worker likewise retains its immutable target-array base. This
removes repeated owner-record loads without allocating another table or changing
slot publication, stream order or lifetimes. The generated code keeps the
polling header fields in registers, with spills across native calls. The
[assembly account](h-audit-2026-09-16.md#thread-local-reader-headers) reports both
the removed reads and the increased stack frames; independent stream probing
and its O(P) cost remain.

The [handoff analysis](transport.md) gives
the complete source flow, capacity assumption and remaining polling/metadata
costs. ABI 67's shared-ticket FIFO is deleted: an unpublished reservation could
delay unrelated published entries. ABI 68's O(E) individual-slot polling is
replaced by O(P) stream polling, but P can still grow with the declared graph.
Neither this change nor these citations establish the latency target.

## Program.kernel_call

The Swift invocation ABI that this section cited (`meshInvocation`, `MeshLaunch`, a C call into Swift's context
register) was deleted with swift/ in a8952de9.

The engine's `MatrixOperations.multiply` now binds concrete operands before
returning the numerical function, using Apple's
[`setBuffers:offsets:withRange:`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbuffers(_:offsets:range:))
for contiguous native binding arrays. Fixed calls no longer invoke the indexed
adapter or reread `MatrixView` descriptors. Indexed callers select from a setup
array of prepared functions. MPS uses its existing matrix multiplication with
concrete operands. The source path and compiler comparison are recorded in
[resolved matrix dispatch](../../../metal-microbench/docs/async_collectives.md#resolved-matrix-dispatch).
The engine now also records its complete native decode step; see
[Native Metal program](#native-metal-program).

The [direct return indices](transport.md)
at ABI 74 apply Monsoon's explicit frame addressing to native-call reclamation.
`mesh_call_bind` assigns each resident call a stable array index. Final-reference
and native terminal events carry that index to the owning numerical worker;
`mesh_call_progress` addresses the call directly. This deletes the row binding,
pointer-table discovery and zero-output routing rows. The existing Collins
reference count still supplies the final-use event; this is not a new ownership
protocol. Array reservation, event bounds and the setup-only publisher/frame
storage reuse are accounted for in the linked analysis.

The [prepared completion record](transport.md)
memoizes the arena and operand counts at binding. Output publication consumes
those fields directly; shared function/worker ownership is followed only for
the subsequent reference releases. This applies the same explicit-record
mechanism cited below without adding a scheduler or a completion protocol.
ABI 76 puts the existing reference-release body in `mesh_call_cleanup`, reached
by a compiler-enforced tail call after publication. It retains Collins's same
ownership events; the purpose is to move their register spills and metadata
loads after the handoffs, with the added stack traffic explicitly accounted for.

ABI 59 implements Monsoon's frame-indexed activation directly: the prepared
consumer record includes the frame, its input updates that call's operand, and
its countdown reaching zero invokes the supplied function in the same frame.
The invocation hash, join/free tables and independent native-slot ring are deleted.
The [current lifetime contract](transport.md) supersedes
the older slot-assignment descriptions below. N1 remains incomplete.

Gregory M. Papadopoulos and David E. Culler,
[Monsoon: an Explicit Token-Store Architecture](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-F03/handouts/papers/p398-papadopoulos.pdf)
(ISCA, 1990), supplies the operand-associated state-bit mechanism and statically
assigned token locations. Mesh uses prepared frame/operand locations and declared
consumer ranges with countdown firing. ABI 68 retains the deletion of the software notification
summary hierarchy; publication now uses the
[index hand-off](#index-hand-off) directly. The page table retains canonical
operand mappings, independently of notification storage.
The historical ABI 55 implementation kept invocation-keyed join records separate
from native call slots; ABI 59 deleted these join and slot structures. Linear probing matches an arriving logical row; countdown
zero binds the accumulated row indices to a native slot and removes the match by
backward shifting indices. Join and native indices have separate owner-managed
rings. This is Mesh's software representation of operand matching, not an
algorithm claimed to have been copied from Monsoon. Input ownership follows
Collins's counted-lifetime mechanism through final output ownership. The
[native-slot derivation](transport.md) proves the
join and native bounds and identifies the still-unfinished receive bound.
ABI 50 separates lasting presence from the notification set: `mesh_presence`
addresses one 32-bit word per logical row; `mesh_publish` release-stores one.
The packed presence bitmap and its read-modify-write are deleted. Setup consumes
constant presence while realizing dependencies; the host runtime still fires
through the existing notification ranges and countdowns.
ABI 53 writes the logical invocation plus one as the presence stamp. This lets
the explicitly blocking synchronization call find a moved invocation through
its storage slots without a racy read of mutable buffer identity. Default firing
does not read these stamps.

Robert A. van de Geijn and Jerrell Watts,
[SUMMA: Scalable Universal Matrix Multiplication Algorithm](https://www.cs.utexas.edu/~rvdg/abstracts/SUMMA.html)
(1997), supplied the panel algebra of `examples/gram-chain.swift`'s `contract`, deleted with swift/ in a8952de9.

Apple's [Core ML Tools MIL builder](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html)
and compiler create the supplied model artifacts in `examples/coreml-models.py`.
The exporter composes existing `matmul`, `gelu` and `add` operations for the
[documented FFN residual chain](transport.md), with caller-supplied dimensions
and owners. It does not generate numerical kernels or add model semantics to Mesh.

Yousef Saad, [Iterative Methods for Sparse Linear Systems, second edition](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf),
SIAM (2003), section 3.4: compressed row storage uses row offsets to index contiguous
entries. Mesh applies that representation to declared consumer references, not
numerical sparse-matrix computation. `mesh_calls_start` counts references, forms
prefix offsets and scatters function/input identities into those ranges before
transport activation. Publication traverses only its row's range. ABI 52 applies
Monsoon's explicit activation-address idea to the finite invocation namespace:
the function worker indexes its call record by the published invocation, then
selects its prepared native storage independently. This is direct array indexing,
not associative token matching or Monsoon hardware. Each input publication supplies
its actual source row. Shared inputs have one consumer entry per binding. Completion
releases those actual rows; the obsolete slot-derived cleanup and remote-refresh
paths are removed. The original shared and unissued references end at the existing
client/device retirement event after native callbacks, without caller release.
`mesh_row_page` is the direct canonical row lookup used by native placement;
the redundant section-to-page wrapper is removed. The [identity derivation and
costs](transport.md) separate this
finite implementation from unfinished N1 reuse.
Row 19v realizes the consumer relation for every resident row. Dispatch indexes
its CSR range directly by the event's row; the section-definition lookup and the
function-index-to-pointer `program` array are removed. Each consumer record holds
the prepared function pointer, input position and shared-input classification in
32 bytes with 32-byte alignment, asserted at compilation. Records use an aligned
setup allocation. For E varying/root uses, S shared uses, V resident slots and F
functions, target storage changes from 8(E+S) to 32(VE+S) bytes and the separate
8F-byte function-pointer array disappears. Row-offset storage is unchanged.
This spends setup memory to remove dependent runtime reads. The supplied stamp
argument also removes `mesh_publish`'s read of `buffer.invocation`: native completion
passes its call's stamp, RX passes the tag's stamp, and constants pass one.
The current use record packs the submit target, argument, operands/counts,
completion handle, invocation source and optional remote binding into 64 aligned
bytes. Setup expands shared-input fan-out into this same contiguous range and
computes initial and recurring pending counts separately. Arrival has no
shared-input dispatch branch or function/call-array lookup. Call records occupy
64 aligned bytes with an asserted layout;
the former 40-byte array elements could cross cache lines. Setup writes the
source row, logical index and view's page-list address into every operand.
`mesh_call_input` and the function's separate placement-flag array are deleted.
A local input or preallocated contiguous view requires no arrival-time rebinding.
An unplaced remote input refreshes only its current first page and address;
that choice is represented by the prepared operand pointer. Its numerical use
retains the canonical page entries through native completion. A shared remote
input refreshes each resident call through its prepared use record and decrements
its dependency count. The recurring template is immutable after setup.
Source identity remains distinct from a
contiguous view's storage, including when the input was already present at setup.
The event record supplies the actual C submit target and arguments without a
call-record lookup for those fields. Mutable call state and operand binding
remain separate data accesses; the full one-line path is not claimed.
Collins's counted ownership applies to the complete declared use set:
startup preserves the compiled output counts, and final output return restores
them through the buffer's common reset before releasing the frame. This removes
refcount increments and presence clearing from launch, as derived in
[output ownership](transport.md).
The former `mesh_call_ready` and shared-input fan-out interpreter are deleted.
These changes apply Monsoon's prepared activation
addresses and Saad's contiguous row ranges; they do not complete H1's handoffs,
H2's remaining records or N1's frame replacement.
ABI 51 separates dependency rows from contiguous operand
views. A single-chunk input selects a view by its received page offset; multi-chunk
inputs use the indexed placement described under [transport](#programcopy).
Contiguous placement is now requested by the existing `inputViews` factory or
`prediction` constructor. Raw CPU/Metal callbacks request no such placement.
This is a setup property of the supplied operand interface, not an inference from
the numerical operation or a runtime layout branch. The BLAS callers and engine
host-input parsers explicitly bind native contiguous views. This change exposes
host indexed scalar reads; device-resident indexed consumption is still X9 work.

Apple [Metal command buffers](https://developer.apple.com/documentation/metal/mtlcommandbuffer)
and [Core ML prediction](https://developer.apple.com/documentation/coreml/mlmodel):
existing numerical submission and completion interfaces.
Apple [pointer-backed MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray/init(datapointer:shape:datatype:strides:deallocator:))
and [outputBackings](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings):
Core ML operand interfaces. Their contracts govern the selected backend's actual
operands; they do not require a mesh-owned executor or function scan.
Apple's `MLPredictionOptions.h`, available with the macOS SDK, describes
`outputBackings` as proposed storage: a model lacking support can return separately
allocated output. The supplied function's actual output-storage contract governs
Core ML use. The current path publishes the declared Mesh output after native
completion; it does not turn separately allocated model output into a registered
operand.
Apple's [MLFeatureProvider](https://developer.apple.com/documentation/coreml/mlfeatureprovider)
protocol supplies named feature lookup. `TensorFunction.prediction` prepares named
input/output views, feature values, providers and output-backing dictionaries once.
Each provider reads its call's realized operand array and selects each input view
independently. The protocol returns nil for an unknown feature name; that lookup
is native operand access, not a readiness or synchronization mechanism. Native
completion ends the provider's access to the call's operands. The example uses
this path for supplied calls with earlier-value inputs, including skip connections,
and scatters their outputs according to the caller's explicit destinations.
Apple [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
accepts independently prepared input and result matrices. The SDK's `MPSKernel.h`
describes reuse after encoding and separate kernel instances for concurrent host
encoders. Each mesh call already has one numerical worker. The existing matrix
implementation now binds input/output arrays separately and encodes their indices
through one operation; Core ML selects its prepared feature provider and output
options independently. `MeshBindings` contains the corresponding operand views
and index function. This removes the Cartesian product of address-bound calls,
without changing the numerical algorithm or requiring additional synchronization.
`TensorFunction` owns a preparation function with one view factory per operand.
Each input and output is prepared independently; no unary function assumption
or product of operand address combinations is introduced. Calls, maps and
reductions use this same function value, including when the reduction allocates
its intermediate outputs. The former native-only `Mesh.map` overload is removed.
Shared constants are inputs with zero index stride. All preparation completes
in `start()`; runtime submission receives the realized operand arrays through
the resolved `MeshInvocation.submit` closure. It does not invoke preparation
factories or choose a backend during numerical execution.
Apple's [makeCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandqueue/makecommandbuffer())
documents waiting only when the queue has no free command buffers; its
[completion handler](https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler(_:))
runs after GPU execution. The installed `MTLCommandQueueDescriptor` describes
`maxCommandBufferCount` as the bound on uncompleted command buffers. Mesh owns a
private queue of V positions for each declared Metal function, prepares V objects,
and replenishes a slot on its numerical worker after native completion and final
output ownership. At replenishment, at most V-1 other objects are unfinished.
No other caller can occupy that queue, no command buffer is recommitted, and no
callback mutates the command array. This is a capacity argument for the native
factory, not a claim that command creation or encoding costs nothing.
CPU functions have no rearm callback; Core ML retains and reuses its prepared
feature providers and output options. The [native slot-return derivation](transport.md)
accounts for return events, output lifetimes, non-submitting ranks, metadata cost,
and the still-finite call/status namespace. Collins's counted ownership and the
existing row-notification mechanism supply the return events; one numerical
worker owns each free-index array. This supplies ordinary native storage reuse,
not X5's unfinished instance-admission rings or R2 failure cancellation.

`Program.kernel_call(kernel, *, out_shape, grid, in_specs, out_specs, peer=None)` in the Python surface records one
call a grid coordinate: each `BlockSpec(block_shape, index_map, region_map)` resolves an operand's block at the
coordinate (`index_map` its block indices, `region_map(ref, *coordinate)` a region within it; `block_shape` None the
whole operand), outputs are fresh tensors of the out specs' blocks, and the call runs on member `peer` (every member
where None). A kernel is any callable `kernel(program, inputs, outputs, coordinate)` over Refs and its grid coordinate
(a program's `functions` maps a kernel to another binding); the library calls it and computes nothing itself. An input
carrying a `Partial` (an unfinished sum: some of its contribution set) is refused (TypeError) by a kernel that does not
declare `combine` (an attribute true of the sum's combine, examples/kernels.py `add`): a nonlinear function of a sum
waits for the sum (design/deliverables.md, Definition, consequence 3); `Partial.merge` refuses a contribution counted
twice.

## Program.copy

The [direct TX publication path](transport.md)
uses the same Apple/rdma-core native posting interface cited below. The prepared
requests reach that interface directly until the provider refuses one; only that
unaccepted suffix enters the pending ring. One inlined posting body handles both
direct posts and retries. No pending record, head/tail access or range rotation
intervenes between successful direct posts. The existing pending-interval capacity
and counted ownership remain. Dedicated thread roles are selected during setup,
and their poll target indices are constants. These are partial evaluation and
removal of intermediate storage, not a replacement collective or transport.

ABI 71's [prepared SEND operands](transport.md)
apply the native WR/SGE mechanism cited below to every native request at setup.
`link_prepare` assigns contiguous request intervals before progress starts.
Local address/key bindings are realized in configuration; receive bindings use
a compiled scatter of `{destination SGE, resolved outgoing registration key}`. The
posting loop consumes these resolved records directly. Native completion IDs
name the existing transfer contribution, so preparing separate requests adds
no new buffer-ownership or completion protocol. This is Mesh's realization of
the registered SEND/RECV interface, independent of the supplied tensor function.

The [prepared receive forwarding](transport.md)
step specializes those keys by the finite registration regions intersecting
each queue's pool. Its native completion identifier carries the registered alias
address itself. The alias map's fixed page arithmetic gives the canonical page
and region without another lookup; the cited section derives the integer
multiply/shift quotient for these exact multiples. This is setup specialization
of the same Apple/rdma-core registered-memory mechanism, with no replacement
provider, dynamic address cache or new ownership protocol.

The subsequent [prepared physical receive values](transport.md)
remove that physical-page/address reconstruction from completion. The posted
buffer's known pool index, physical page and client address are stored beside its
existing local GPU address and received tag. Apple's shared `mmap` mechanism
cited under [Program.tensor](#programtensor) makes the registered tag alias and
canonical local metadata refer to the same backing. Completion loads terminal
values from that one 32-byte span and scatters them into the tag-selected logical
entry. No extra allocation, lookup, transmitted metadata or readiness condition
is added. The existing region selector for forwarding keys remains.

Apple's installed SDK `infiniband/verbs.h` implements `ibv_post_send`,
`ibv_post_recv` and `ibv_poll_cq` as calls through the handle's
`context->ops` field. Mesh resolves those exact function pointers when the
native handles are created, storing handles and functions together in its
prepared queue record and the send target in its existing SEND record.
The [memoized dispatch implementation](transport.md)
calls those provider functions directly with the same arguments and return
handling; it does not implement a replacement provider or completion protocol.
Re-pairing fills these records again before progress starts. This is partial
evaluation of upstream dispatch, not a new transport algorithm.

The [direct completion address](transport.md)
uses the native `wr_id` return mechanism cited below to carry the prepared local
tag address. Affine tag-address inversion recovers the physical page without
reconstructing that address from the arena header. `link_receive_post` submits
the same prepared native requests for initial posting and returned pages; it
is inlined, and the return path posts before logical-row and instance cleanup.

The [prepared receive publication](transport.md)
uses the same direct-index mechanism to bind the exact chunk-entry address,
completed section row and publication counts in the existing 32-byte receive
record. Publication is one inline primitive with explicit counts; the separate
native publisher call is removed. A receive with no notification targets, such
as the engine's resident FFN input, writes its canonical stamp without reading a
publication header or notification table. Nonempty ranges still read their
canonical targets and streams. This specializes fixed graph metadata; it does
not infer a collective or introduce a receive mode. Native matching, registered
pages, onward-SGE writes and reference-counted retirement remain as before.

ABI 59 realizes a peer-qualified, source-chunk-indexed table of aligned 32-byte
receive records. The eight-byte tag names the 32-bit sequence and source chunk.
ABI 60's record supplies the destination row, buffer address, exact canonical
page-entry address, reference count and chunk flags. This is direct addressing
of the known transfer relation, applying
the existing compressed-row/direct-index mechanisms; it introduces no collective
algorithm or credit protocol. The [wire layout](transport.md)
accounts for table storage, shared-tag writers and the unfinished N1 lifetime proof.

Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
MLX authors' [JACCL transport](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h),
and rdma-core's [ibv_post_recv](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3):
registered SEND/RECV and work completion. The [hardware notes](transport.md)
distinguish substrate facts from the retained bridge's protocol decisions.
TN3205 explicitly limits this transport to `IBV_WR_SEND`; its SDK enum for
`IBV_WR_SEND_WITH_IMM` does not establish hardware support. Both TN3205 and JACCL
show local `wr_id` values returned with completions. Mesh uses these identifiers
for buffer lifetime. ABI 59 carries one `(sequence, sourceChunkRow)` word. ABI 60 resolves the
chunk's local row and page-entry address at setup, alongside its publication flag.
TN3205 requires paired SEND and RECV requests to occupy the same number of
4 KiB frames. Setup fixes their extents; posting submits the prepared requests.
[Byte accounting](transport.md) records their
cost independently of numerical partial boundaries.

`link_configure` prepares a native RECV WR and SGE per pool block and a native
SEND WR and SGE per resident send edge. RX posts the prepared request selected
by its pool index. TX selects the current backing address and registration key,
then posts its prepared request without changing the length. `link_post`, the separate posting wrapper, is deleted. The local Apple SDK defines a 128-byte `ibv_send_wr`,
32-byte `ibv_recv_wr` and 16-byte SGE: a complete SEND WR plus SGE cannot fit the
former H2 limit of 64 bytes. The receive request occupies 64 aligned bytes;
the send record occupies 256 aligned bytes, with its SGE, application fields
and common SEND header in the first 128 bytes. Compile-time assertions check
these layouts; they do not prove one-line provider access or the whole H2 path.
Following ABI 69, `link_configure` also binds the canonical page-entry pointer,
registered span-array pointer, sequence-word pointer and queue pair directly in
the existing SEND record. `link_send_ready` reads those prepared operands. The
selected registered span already starts at the shared tag word, so that same
address supplies the CPU tag store; no second tag mapping is derived. This uses
the existing shared alias, not a new mapping or transport operation. The
[load accounting](transport.md) records the
remaining indexed inputs and generated-code comparison.
The send edge counts its signalled chunk completions before releasing its
existing buffer reference. This is thread-local completion bookkeeping, with
no new completion event or admission check. TX address resolution and wire-tag
stores still precede the post; those remaining H3/H6 costs are not hidden by
request preparation.
ABI 63 also posts a newly enqueued edge on its selected queue immediately.
The preceding all-queue CQ scan per publication is deleted. Native capacity
refusal returns to the existing CQ progress path; other errors retain the
existing link-failure path. The outer loop still services every configured
queue. This changes posting order relative to bookkeeping, not collective
semantics, request extents or source lifetimes.
The tag arrives in the payload's own work request. There is no index QP or
cross-QP identity join. On September 15, `ibv_devinfo -v` reported `max_sge: 1`
on the local Thunderbolt devices. The alias representation uses one SGE and
does not infer two-entry support from the general verbs API. The
[record layout and execution path](transport.md)
describe this mesh-specific representation. Receive storage is preallocated
and reused by the owning RX thread at final reference release. The existing
notification mechanism carries the return event; only RX writes its page and
return ring. Each queue poll returns one completed section's backing,
rotates unfinished sections and attempts an available post. It reposts actual pages without clearing them and
indexes the prepared source-chunk record. The [receive-storage proof](transport.md)
accounts for receive ownership, bounded metadata, teardown, repeated forwarding,
and the finite caller namespace still awaiting N1. `link_receive_destroy`
disposes this setup-owned metadata after its link worker stops. This is an
application of the cited SEND/RECV and reference-count mechanisms, not a new
algorithm attributed to JACCL. Posting always uses available pool backing;
the RX thread does not wait for a particular consumer or return event.
Apple's [Metal buffer copy](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:to:destinationoffset:size:))
encodes a copy between existing buffers. ABI 51 uses it to place logical chunks
into canonical contiguous storage before the supplied encoder in that same
command buffer. The documented macOS four-byte alignment is satisfied using
reserved tail padding. CPU and Core ML inputs use the system `memcpy` on their
numerical worker. Setup resolves both dependency operands and their native input
views, allocates the canonical placement outputs, and prepares chunk rows and
views. This changes representation only, performs no tensor arithmetic, and
adds no queue wait or native completion hop. The [data-flow and cost account](transport.md)
covers ownership, finite-instance storage, and copies per consumer invocation.

The dedicated TX worker consumes publication events and advances the associated
send edges, rotating unfinished edges after each accepted chunk. RX independently refills receives before publishing each received
section. Neither constructs a runtime list of all chunks before posting the first.

The rdma-core authors' [ibv_post_send contract](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)
returns acceptance or a native error for each posted request. X4 removes Mesh's
software outstanding-request count and its derived request-capacity gates.
Each TX/RX progress step polls a completion and posts available requests until
the prepared ring is empty or the native provider refuses, then continues across queues. A refused request does not advance the cursor.
`ENOMEM` and `EAGAIN` are deferred to the next step; other post errors conclude
the link's invocation through [its Result](#meshresult). RX now publishes the
completion before its return/repost pass. A zero-completion poll still permits posting. Initial receive setup
posts until the declared list ends or the native queue refuses. Actual QP capacity is
used only at setup to check that an individual framed request fits. This does
not claim the remaining receive mapping or publication queues satisfy W2/W3.

Receive posting and returned-page draining now share `link_receive`. After each
returned section's entries are detached, its backing is immediately offered to
the NIC. A capacity refusal retains the unposted cursor and still allows the
return set to drain; it is not a condition on reclaiming other pages. The old
single-return `link_returns` stage is deleted. This changes when existing
references and ready pages are processed, not their counts or the wire protocol.
The receive completion publishes before that drain, so accumulated return work
cannot precede publication of the already polled completion. TX still offers
ready sends before processing its completion cleanup.

A section crosses intermediate nodes along the map's route ([placement](#placement)):
at each, the landed section itself is the onward SEND's. No additional function,
operand copy or runtime route lookup is introduced.

Dotan Barak's rdma-core [work-completion contract](https://man7.org/linux/man-pages/man3/ibv_poll_cq.3.html)
defines the valid fields of an unsuccessful completion as its work-request id,
status, QP number and vendor error. Mesh returns on that native status before
reading the received tag or resolving a destination. It does not treat an error
completion as a filled partial tensor.

The compressed-row representation cited under [numerical submission](#programkernel_call)
also stores each published row's send edges contiguously. Thompson, Farley,
Barker, Gee and Stewart's [Disruptor technical paper](https://lmax-exchange.github.io/disruptor/disruptor.html)
(2011), sections 3.1–3.3, describes preallocated ring storage and sequence indexing.
ABI 50 uses that storage representation for each TX queue, with power-of-two
capacity covering its configured edge count. One TX thread owns both positions;
the [edge-lifetime bound](transport.md) replaces any
software fullness gate. No Disruptor wait strategy or sequence barrier is imported.
Receive targets are grouped by source-chunk row with one advancing cursor per
group, retaining a target for each declared copy. The contiguous receive page run
is addressed arithmetically. These replace linked send/target records and the
receive-address array without changing SEND/RECV ordering or collective semantics.

At the higher-order call interface, each part has an identity assigned at setup.
Repeated deliveries with the same identity, source rank, destination and queue share one
receive operand. This is Mesh's setup-level sharing of a declared immutable value;
numerical input positions and their ownership remain distinct.
Different source ranks, destinations or queues are not coalesced. The delivery map is discarded
before execution, so its lookup is not transport or numerical work.

The rdma-core authors' [queue-pair creation](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_qp.3)
and [memory registration](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
associate QPs and registered memory with a protection domain. Each port's child of
the bridge holds that port's device context, protection domain, registration and
completion queues; the port children are the only processes that hold a device's
objects or post on them (design/transport.md section 2). A child makes its link's
queue pairs once a pairing, as many as the link's plan names
([HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md) section 5), and every client
shares them (design/transport.md section 2). A client holds a stream of the link
(`mesh_stream_take`, rdma/mesh-call.h:30) and presents its own rows on it, and holds
no queue pair; the port's send and receive threads serve every stream of the link
(rdma/mesh-link.c `port_progress`). Several ports register the same region through
their own devices; they do not create operand copies.

A link pairs once per port child: HELLO and LINK on the control socket carry the
queue-pair numbers, packet sequence numbers and GIDs the hardware assigns at each
pairing (design/transport.md section 3). The parent dials the control connection
nonblocking to the link-local address its peer's hello came from
(rdma/mesh-link.c:922-923), so pairing performs no host or service lookup. Apple's BSD [connect](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/connect.2.html)
and [accept](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/accept.2.html)
contracts supply the nonblocking socket mechanism. A program pairs nothing: its
communicator takes streams of links already paired (design/transport.md section 14).
Since 2026-10-22 no clock decides a transaction; the
platform deadline bounds the bridge's own lifecycle (design/transport.md section 5).
As [design/transport.md](transport.md) explains, userspace deadlines cannot unwind
a driver call already blocked in kernel sleep.

In the Python surface `Program.copy(source.on(sender), destination.on(receiver), *, queue=0)` (also `send`, `recv`)
cuts tensors at both block grids into regions; a region whose two ends are one member is a local copy call, else a
crossing: libnccl-mesh's send and receive (`torch.distributed` on the "mesh" backend: rdma/torch-mesh/README.md),
the source read on the sender, the destination defined on the receiver. A communicator's positions pair in the order
both ends issue them (rdma/NCCL.md "Ranks, lists and streams"), so every member computes one order from the graph all
members recorded: a topological order of the calls, the earliest row first, then the earliest record. Each queue is
its own communicator, its crossings in that order, round after round, on a thread of its own (CPU: a host-path call
completes where it returns) or, on MPS, issued by the compute thread as they become ready (encoded on torch's stream;
a receive's landing is waited where its first reader runs). A crossing waits only for its own inputs, so a section
withheld on one queue holds only that queue's later crossings: progress independent of a withheld section, or of the
other direction, is a queue of its own. `replicate(source, peer)` sends each backing once and views it on the peer.

## Independent native queues

Dotan Barak's rdma-core
[completion API](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)
and Barak, Majd Dibbiny and Yishai Hadas's
[posting API](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)
describe independently posted requests and retrieved native completions. The
rdma-core authors' [rping example](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/examples/rping.c)
also separates CQ handling from its posting caller. These are references for
native queue ownership and completion lifetime. Mesh retains the registered
SEND/RECV operations described under [transport](#programcopy); it adds no native
transport implementation.

`link_send_completions` owns all SEND CQs for its link and directly publishes
completed indices to the existing retirement ring. `link_send_progress` owns
SEND posting and its native-refusal cursors. It discovers a new publication,
posts its requests directly, and visits each nonempty refusal queue for one
native attempt before inspecting publications again. Successful attempts advance
the existing cursor; a native refusal preserves it. The next transfer boundary
is prepared in each SEND record, so successful posting walks native records
without loading/reconstructing a separate chunk loop. The retirement chunk count
remains beside the other cold fields. Continuous publication cannot suppress CQ
polling, and a CQ sweep cannot precede discovery in the posting thread.

`link_receive_progress` owns all RECV CQs and publishes received operands.
`link_receive_posting` owns the existing returned-page ring's head and posts its
prepared requests. It attempts one available receive per QP on each continuous
sweep, so a batch returned to one QP does not have to finish before another QP is
visited. Initial configuration still fills native receive queues before pairing
completes. Runtime reposting no longer executes in the CQ-consumption path.

Each CQ has one reader, each native send/receive queue one posting thread, and
each software ring retains its existing single writer and reader. Completion
scratch is a native record on its CQ reader's stack. The shared heap allocation
and its provider pointer are deleted. This adds two threads per link, from three
to five including retirement, and zero per-event handoffs, queues, counters,
locks or payload copies. All progress loops spin; there is no timer, task queue
or wakeup protocol. The extra threads consume CPU resources; no claim of free
scheduling or measured latency follows from their separation.

The startup order is retirement, SEND CQ, RECV CQ, receive posting, SEND posting.
Teardown stops and joins all four native-queue workers before stopping retirement
and destroying native queues. A partially completed startup uses the same join
order. Operand references, frame reuse, page return and native-error reporting
retain their existing mechanisms. The remaining metadata loads and measured
native bodies are recorded in the
[structural follow-up](e2b-structural-latency-2026-09-17.md#follow-up-independent-native-queue-progress).

## Placement

Pitch Patarasuk and Xin Yuan,
[Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
(2009), motivate sparse connected topologies. A route is the map's: `mesh_map` (mesh-plan.h) links members the
links leave apart along the routing table's route between them ([Topology](#topology)), each node between them a
member of the communicator with no contribution: the bridges forward nothing (design/transport.md). No
route is chosen by a caller or guessed from rank count.

For a section s on `v0 → v1 → … → vk`, each intermediate node's program receives on the rows it sends: a record
landed from `v(i-1)` releases the SEND cell that carries it to `v(i+1)` (transport.md), with no numerical
function, consumer stamp or payload copy. For the distinct realized legs E and a cut C, payload traffic is
`B(C) = sum(bytes(s) * count(s) for (u,v,s) in E crossing C)`.

## Topology

`struct mesh_link_map` (mesh-plan.h): MLX authors' [JACCL hostfile](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/distributed.rst#defining-a-mesh) supplies the full-mesh special case; Patarasuk–Yuan, [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) (2009), supplies tree-connectivity sufficiency.
A map is an operand: its kind (mesh, ring, tree, graph), its nodes, its links and each link's cost (α, β; Hockney
1994). Trees, rings with spurs, tori and full meshes are graphs of one representation, and `mesh_link_map_check`
admits any of them, joined or not: the bridges' records while a cable is out are a map with a node no link joins.
Whether a map joins its nodes is `mesh_link_map_joined`, and the trees, collectives and sessions run only on one that
joins its members (the tree-connectivity the paper's sufficiency needs). Neither source establishes that this runtime
achieves the paper's bandwidth bound.

A members' map (`mesh_map_members`, which `mesh_map` reads the bridges' records by and a planner prices a record's
members by) keeps every node of the whole map: the members first, the others relays that hold no rank, every link,
each node's mark (its bridge's node number) and the routing table, an operand: each node's in-tree of shortest routes
(fewest links, then the least sum of betas). Two members are linked where their route crosses relays alone, that hop
costed as the relayed path it is: its links' α summed and the greatest β, a node between handing each slot on as it
lands (mesh-program.c receive_input, the index hand-off of a member with no contribution), not the betas summed; and
its bytes charged to every link it crosses in the tree packing (a port's capacity shared by the hops whose routes cross
it), so hops that share a cable share its time. The bridges forward nothing: a node between two members takes part as
a member of their communicator with no contribution (design/transport.md). Several cables between one pair
are links of their own, as the bridges carry them (design/transport.md): a program between the two is
striped over the cables its communicator leased, position t on cable t mod n, so the hop runs at n times the slowest
cable's bandwidth (round robin waits on the slowest), not at their bandwidths summed. The bridges' map lists a pair's
cables as such, each at the slowest port's bandwidth at either end (mesh-dataflow.c mesh_joined).

The table (`mesh_routes`): Dijkstra [1959] toward each node in its dense form, every length an exact integer (a
pair's cables striped) so every bridge and member that holds the same records makes
the same table, each node's parent among its equals the least hash of its mark, the parent's and the root's
[splitmix64's finalizer: Steele, Lea & Flood, OOPSLA 2014], equal-cost multipath by a hash as ECMP spreads flows
[Hopps, RFC 2992]. A node's parent reads its own candidates alone, so the route between two nodes is a function of the
shortest routes between them: it moves only where those change (a link or node one of them crosses lost or its cost
changed, a link joining or a cost falling that makes another as short), never for a link lost elsewhere or a node
joining by its own cables. The route between two nodes is the path up the higher-marked one's in-tree, so a route and
its reverse are one path. It replaces the lowest-numbered parent among equals, which sent the ring of six's three antipodal routes
through nodes 0 and 1, and the round's first pass's global search on link loads, which balanced the table better (the
ring of six's busiest link 5 routes in every numbering, 5 or 6 by the hash) but moved a membership's routes for a link
lost elsewhere in up to 35 % of cases and for a node joining by one cable in up to 45 % (metal-microbench
output_data/library-20261015/pass2/sims/routes_after.json: none by the hash). The balance the collectives need is
their tree sets' own: `mesh_trees_shortest` builds each member's in-tree over the members' hops, a parent among equals
chosen by a local search on the congestion game's potential, the sum of the squares of the physical links' loads,
each the tree edges whose hops cross it child to parent [Rosenthal, Int. J. Game Theory 1973], which every move
lowers, ties to the hash; a communicator's tree sets are its own (its members' digest carries them), so balancing
them moves no route of the bridges'.

The collectives (mesh-collective.c): along the packed trees [Garg & Konemann, FOCS 1998; Blink, Wang et al. MLSys
2020], the shortest-path trees, and the single-phase all-reduce [Trivance, SIGCOMM 2026]; each tree's part cut into
chunks, chunk j of every move at its round + j, so a node forwards a chunk as it arrives (the pipelined trees of
NCCL's tree algorithm and of MSCCLang's chunks [Cowan et al., ASPLOS 2023]); and the program a call runs chosen by its
time as libnccl-mesh runs it (`mesh_choose`: every collective, root and size class, over both tree sets and every chunk
count 1 to 64) [Thakur, Rabenseifner & Gropp 2005; NCCL's tuner]. That time (`mesh_program_time` on `struct
mesh_wire`) is the runtime's own timing, one implementation with it: every rank's steps laid on its channels by the
layout libnccl-mesh lays them by (`mesh_lay`), each position a whole slot of its class on the wire (a filler's too), a
position published once its sender has consumed the peer's position lag back (NCCL's Simple protocol's credit) and,
a piece's first, once the receive before its step is consumed (`mesh_steps_before`), consumed in position order; ports
as the alpha-beta model's [Hockney 1994], a relay cut through. An alpha-beta step model of the same programs priced
chunks of less than a slot at their bytes, so it chose them where each costs a whole slot (metal-microbench
output_data/libreview-20261015: up to 8.8 times off); this time equals the review's position-level simulator of the host
path to the last bit on 1,520 candidates (output_data/library-20261015/pass2/sims/model_vs_review.json).

Swift's `Topology` (swift/Topology.swift, restored 71bced9e), with `Topology.Pair`, `Topology.links` and
`Topology.merging`, was deleted with swift/ in a8952de9.

### Link

`port_bandwidth` (mesh-link.c): Dotan Barak, rdma-core's [ibv_query_port manual](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_query_port.3), supplies observed port speed and width.
Bandwidth is bits per second: speed codes 1/2/4/8/16/32/64/128 mean
2.5/5/10/10/14/25/50/100 Gb/s per lane; width codes 1/2/4/8 mean 1/4/8/12 lanes.
Unrecognized codes yield zero (unavailable), without withholding the link.
The bridge publishes bandwidth before the paired phase; atomic scalar access also
covers re-pairing. A link is named by its port's index; no device name leaves the
bridge. Port bandwidth is an attribute, not measured payload throughput, and it is the only fact of a link that
routing reads (`mesh_map`; `MESH_ALPHA_US` and `MESH_BETA_NS` were deleted in cc8a6f8e).

### Mesh.observe

`mesh_reached`, `mesh_region`, `mesh_map`, `mesh_map_wait` and `mesh_nodes_health`, the bridge's map readers (mesh-plan.h, rdma/mesh-dataflow.c:26-150): Apple's [shared mmap mechanism](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/mmap.2) supplies a read-only view of the node's bridge's region, without attaching a client or changing the bridge's state.
`mesh_reached` returns this node and the nodes its streams reach, each with its bandwidth, and `mesh_region` the
region's bytes and those not leased. `mesh_map` copies the bridge's map (design/transport.md), every node's newest
record, as the seqlock's reader (Boehm, MSPC 2012): the serial read with acquire once it is even, the records copied,
the copy kept where the serial is unchanged (`mesh_map_whole`), else copied again. From that copy it makes the whole
map, two nodes linked where each one's record reaches the other, at the lesser bandwidth (`mesh_joined`), its table the
shortest routes (`mesh_routes`), and over the members the members' map of it (#topology). `mesh_nodes_health` reads
the same records for each node's region, memory, model and health. `mesh_map_wait` waits on the serial, within its
bound, so a loop of executions takes the next map. The bridges keep their links paired with or without a client, so
the map is current whether or not anything runs (design/transport.md). `mesh_observe`, which copied the port table,
became `mesh_reached` in f4b363ad, and Swift's `Mesh.observe` was deleted with swift/ in a8952de9.

## Program.write

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
operand production.
Apple's [vDSP_vramp](https://developer.apple.com/documentation/accelerate/vdsp_vramp)
is the existing numerical generator called by the configured Core ML chain's
initial producers. It writes directly into each registered operand.

`Program.write(ref)` (a context manager) yields the block to fill (a NumPy view of a CPU block; a staged array copied
in on MPS) and on exit publishes it: a token on each edge from the block to its readers. A block whose last
publication still has readers refuses (BlockingIOError); `Ref.writable` asks. A write before `realize()` is published
at it.

## Program.export

MLX authors' [Metal evaluation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/eval.cpp):
reference for retaining operands through actual operation completion.

`Program.export(ref)` is a host reader of a region on the exporting member alone (it changes no member's order):
`Result.ready` once its writers' tokens arrived (on MPS once an event recorded after them completed), `Result.array`
its read-only values, `Result.consume()` releases it, so its writers may run again.

## kernels.expression

The JAX authors' [Pallas indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs):
values, indices and masks.

The Python surface's examples' expressions (`examples/kernels.py`, program type C: `arguments`, `indices`,
`program_id`, `select`, `.at(row, column, mask, other)`, `.astype`, `.sum(axis)`, `.T`, `.exp`, `.rsqrt`, the
arithmetic and comparison operators) are kernels evaluated by torch over each output block: no source is generated.
`row` and `column` are the output block's own indices, `program_id(axis)` the grid coordinate; floating values are
computed in FP32 and stored in the output's type.

## kernels.dot

Dongarra, Du Croz, Hammarling and Duff, *A Set of Level 3 Basic Linear Algebra
Subprograms* (1990), Apple [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication),
[BNNS matmul](https://developer.apple.com/documentation/accelerate/bnnsfiltercreatelayerbroadcastmatmul(_:_:))
and [Core ML MIL operations](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html):
existing contractions. The [reduction algebra](transport.md) describes
contributions without prescribing intermediate-buffer or launch counts.

In the Python surface's examples `kernels.matmul` (a kernel over a (rows, K) and a (K, columns) Ref) and an
expression's `dot` contract by torch's own product: the contraction a supplied function, the library importing no
engine.

## kernels.add

The JAX authors' [Pallas accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation):
reference for combining numerical contributions.
The engine's existing `encAdd` encoder and `add_inplace` shader now live in its
shared matrix source files, retaining their existing arithmetic and launch
geometry. Callers prepare their pipeline and dimensions and supply `encAdd` with
the actual operand buffers and offsets. Mesh implements no addition kernel;
the unused `MatrixOperations.add` convenience wrapper has been removed.
Apple's [vDSP_vadd](https://developer.apple.com/documentation/accelerate/vdsp_vadd)
is the existing float32 vector sum supplied to reduce-scatter by the configured
Core ML caller. It receives resolved input/output pointers and introduces no
payload allocation, copy, or replacement sum implementation in Mesh.

The examples' `kernels.add`, `maximum` and `minimum` (NaN-ignoring), `swish` and `identity` are expressions; `add`
declares `combine`, so it takes `Partial` operands and is a supplied combine a reduction takes. The Python surface's
own reductions are libnccl-mesh's operators, `collective.SUM`, `MAX` and `MIN` (`ncclSum`, `ncclMax`, `ncclMin`), and
its partials merge, `PARTIALS`.

## collective.reduce_scatter

Rabenseifner, *Optimization of Collective Reduction Operations* (2004), and the
MPI Forum's [collectives](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm):
reduction semantics and decompositions. A source does not mandate one decomposition
for every collective or caller placement.

The Python surface's `reduce_scatter(program, value, *, peers, owners, op=SUM, queue=0)` (also `sum_scatter`),
`reduce(..., root, peers)` (`reduce_scatter` with every block owned by `root`) and `all_reduce(..., peers,
owners=None)` (`all_sum`, `all_max`, `all_min`) reduce a block row in one libnccl-mesh call over its members where
`op` is `SUM`, `MAX` or `MIN`: `reduce_scatter` MPI_Reduce_scatter (`ncclMeshReduceScatterV`, member m's segment the
row's blocks m owns, every member's terms packed in member order; an owner off the peers contributes the operator's
identity), `all_reduce` the row's all-reduce; an update of each block, its readers taking the reduced value. `PARTIALS`
merges each block by its log-sum-exp (nccl.h `ncclMeshRedOpCreatePartials`'s operator, a reduce toward its owner or an
all-reduce toward every peer, through torch_mesh's ProcessGroupMesh.merge_partials on either path; the block read as
rows of [m, l, O], a member off the peers an empty partial). Any other associative combine is a supplied
kernel declaring `combine`: each peer's term moved to the owner by one all-to-all a row
and combined there by a tree of the combine's kernel (`kernel_call(op, peer=owner)`), into fresh blocks; `all_reduce`
of such a combine is that `reduce_scatter`, then `all_gather`.

Swift's `reduce(_:to:using:worker:)`, `reduceScatter` and `allReduce` (swift/Mesh.swift), which ran
mesh-collective.c's compiled programs, were deleted with swift/ in a8952de9 (2026-10-19).

## Collective movement

The MPI Forum's [collectives](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and MLX's [distributed operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp):
distinct communication relations, one for each verb.

In the Python surface each verb is one libnccl-mesh call a block row over the communicator of its members (a row
crosses once its own blocks are ready): `all_gather(program, value, *, peers, owners)` is MPI_Allgatherv
(`ncclMeshAllGatherV`, member m's segment the row's blocks m owns, every member, the peers and the owners, receiving
every block) and `broadcast(..., root, peers)` the same with every block `root`'s; `all_to_all(..., owners,
destinations)` is MPI_Alltoallv (`torch.distributed.all_to_all_single` on the "mesh" backend), and `scatter` and
`gather` the same with every block's sender or receiver `root`. Each defines the block on its receivers in place, its
blocks packed into one region tensor and back.

Swift's `broadcast`, `allGather`, `gather`, `scatter`, `allScatter` and `allToAll` were deleted with swift/ in
a8952de9.

## Vocabulary projection

Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), and the MLX
authors' [tensor-parallel linear layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py)
provide the output-column decomposition. For replicated h and disjoint column
ranges C_i, each participant computes `logits[C_i] = softcap(h W[:, C_i])`.
These are disjoint output coordinates; they are placed, not summed. The engine
reuses its existing packed, transposed GEMV, and matrix launch paths with a
weight offset/stride or a matrix subview. The selected backend is unchanged.

The [Pallas operand representation](#programtensor) supplies the contiguous
selection scopes. Setup prepares each vocabulary tile's canonical page array,
element offset, presence address and invocation address, and the selection's
kernels (argmax.metal) resolve a contiguous pointer once per tile. A record kept
for divergence is the retained set (`mesh_retain`: a head of the greatest z and a
tail of the greatest z + g outside it, (id, z) each, beside the row's (m, l)),
never candidate tokens with their logprobs. This is ordinary indexed operand
access, not a second selection or a tensor allocation.
A Metal function constant selects indexed versus ordinary inputs at pipeline
creation. The ordinary specialization does not poll Mesh or load page metadata.
The indexed partials pass uses the existing [resident presence mechanism](#resident-metal);
subsequent passes use their existing command-buffer dependencies.

The engine binding's selection is each member's draw over its own vocabulary
columns at temperature 1 (argmax.metal `mesh_draw`: the greatest z + g, g the Gumbel
noise of the row's seed, position and token) and the select collective, the take of
the members' (z + g, token) pairs (`mesh_take`). Min-p's cutoff is m + log min_p, m
the greatest z from the members' value-only maxima (`mesh_normaliser`), so no member
computes or keeps a token other than the draw's. The engine account is
[here](../../../metal-microbench/docs/async_collectives.md#distributed-vocabulary-projection).

The same column decomposition now applies over explicit input-row scopes. Setup
partitions each vocabulary shard into packed rectangles of at most 120 KiB, on
selection-tile boundaries, and binds row offsets in all three existing
projection backends. Larger batches use multiple row scopes. Each completed
projection is published from its existing resident command buffer before the
next projection; the sampler's precomputed operand records name each tile's own
presence word and local offset. There is no full-shard completion prerequisite
or dense logit assembly. The [bounded-publication account](../../../metal-microbench/docs/async_collectives.md#bounded-vocabulary-publication)
records added publication/dispatch work and metadata capacity. The full 35+1-push
contract and the floor latency remain open; per-tile payload bounds do not
establish them.

The engine now records the entire supplied projection/publication list using
Apple's [native Metal program](#native-metal-program). Each output scope carries
its setup-only continuation commands; the numerical binding remains independent
of Mesh. Those continuations reuse `mesh_coherent` and `mesh_signal`, with their
existing payload visibility and publication mechanism. No shader arithmetic,
payload backing, dispatch count or collective route changes. Native projection
backends replay one recorded range instead of encoding every tile on the host;
MPS retains its own encoders and prepared following native segments. The engine's
[encoding account](../../../metal-microbench/docs/async_collectives.md#recorded-vocabulary-projection-and-publication)
states the removed host calls and the remaining device/driver costs.

## nn.ffn

Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), and MLX's
[tensor-parallel layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py):
local numerical functions composed with collectives. Hendrycks and Gimpel,
[GELU](https://arxiv.org/abs/1606.08415): the activation already supplied by the engine.
The configured function caller uses `Cij = Fij(Xi)` and `Yj = sum_i Cij`, then
passes each `Yj` to its supplied consumer. For linear T, Fij is its input-i,
output-j block. Reduce-scatter places each sum at its caller-selected owner;
there is no assembled whole-tensor intermediate or global stage barrier.
The [caller description](transport.md) distinguishes this source composition
from model artifacts and runtime performance evidence.

The engine's E1d binding uses the same FFN column decomposition with a direct
send of each codomain partial to every consumer, followed by the supplied local
sum and layer finishing functions. `bindMeshDecodeLayer` declares this graph;
`bindDecodeAttention` and `bindLayerFinish` reuse the standalone numerical
functions over explicit buffers. The
[source account](../../../metal-microbench/docs/async_collectives.md#e2b-decode-direct-partial-exchange)
records the remaining command-buffer, publication and KV-initialization gaps.

For a resident GPU handoff, the Apple MLX authors' actual
[fence implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/fence.cpp)
and [Metal kernels](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal)
use system-coherent accesses and a system-scope fence. Cross-device publication
also dispatches `input_coherent` over the payload before updating the timestamp.
Its fast wait is a submitted single-thread GPU kernel. This is the reference
mechanism; a timestamp alone is not evidence that payload stores
are visible. Mesh has not yet integrated this mechanism, and its latency is not
established by the engine's direct-send graph.

The Python surface's examples' `nn.linear(program, x, w, *, tile_rows, tile_k, tile_columns, peer, output_dtype)`
(`examples/nn.py`, program type C) cuts K into panels at `tile_k` (and at the operands' block edges): each panel a
grid of `kernels.matmul` calls whose products are `Partial`s of one contribution set, summed by a tree of
`kernels.add` calls tile by tile, so a tile's sum is published as its panels finish (SUMMA's K-panel rank-k updates, van de Geijn and Watts 1997). `nn.ffn(program, inputs,
up_weights, down_weights, *, tile_rows, tile_k, tile_columns, exchange=None, peer=None)` sums each hidden section's
input partitions, applies swish to the completed hidden tiles, hands each to `exchange` (a caller's crossing) and
contracts it with the section's down projection, the sections' products summed.

## nn.rmsnorm

Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
(2019): normalization used by the existing numerical implementation.

The Python surface's examples' `nn.rmsnorm(program, x, gamma, *, tile_rows, epsilon=1e-6, peer=None)`: each column
panel's sum of squares a `Partial` of the row statistic, the panels summed, then each tile normalized by the completed
statistic.

## nn.embedding

The JAX authors' Pallas indexing and the llama.cpp authors'
[GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):
references for indexed operands and the existing model loader.

The engine's embedding gather uses grid coordinates `(batch row, section)`:
`output[(row * sections + section) * width + component] = table[token[row] * stride + section * width + component]`.
This is the same indexed gather for ordinary embeddings and E2B per-layer inputs.
One grid replaces the layer-by-layer host dispatch loop. Row/section/component
storage also lets the engine use one existing matrix contraction for the full
model projection, by concatenating the per-layer weight columns. Layer consumers
bind their component slice using an offset and row stride; the GELU (`streamed_gelu_rows`)
reads the second operand at its own row stride. The
[projection algebra and consumer addresses](../../../metal-microbench/docs/compute_execution.md#per-layer-inputs)
account for all row counts and the removed 34 projection launches on E2B.
Mesh gains no arithmetic kernel or transport behavior from this engine change.

The Python surface's examples' `nn.embedding(program, table, indices, *, tile_rows, tile_columns, peer)` gathers each
(row, column) tile of the table's rows by `kernels.gather` (a negative index counts from the end); `nn.summed_embedding`
adds tables' embeddings to an input, tile by tile.

## collective.sync_on_remote_fill

The MPI Forum's [communication completion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm):
explicit completion operations. The operator, 2026-09-14: "if we want to supply a 'sync_on_remote_fill' barrier,
this is library code that needs to be used EXPLICITLY AND DIEGETICALLY IN CALLING CONTEXTS, it can NEVER be a default
behavior"; "we must now implement an explicit 'sync on remote fill' call that our library users can use, and a peice of
demonstration code which shows it is slower and can deadlock forever".

In the Python surface `collective.sync_on_remote_fill(*results, timeout=None)` waits until every result is filled,
within its program's `timeout` (TimeoutError past it; MeshError where the program failed). No verb calls it.

## Streaming overlap measurement

The Python surface's reproduction harnesses (restored from mesh `57628982^:examples/streaming-chain.py`,
`82be91f:examples/streaming-algebra.py` and `4540397d^:examples/streaming-overlap.py`). Each runs paired (two ranks,
launched as torchrun launches them: `RANK`, `WORLD_SIZE=2`, `MASTER_ADDR`, `MASTER_PORT`, one a node, each node's
bridge resident), on `--backend cpu` or `metal`, and states the comparison it reproduces; the frontier records stay
where they are. None runs on one rank. The operator, 2026-10-21 (design/rubrics.md, Quantities): "if no work is split
across devices (tuple len == 1) for the tp ratio, then the run is not qualified for any sort of comparison for the mesh
lib at all".

- `examples/streaming-chain.py --chain literature`: `FFN → RMSNorm → summed learned embedding → FFN → RMSNorm` on
  256 x 128 FP32 inputs in 64-row, 64-K, 64-column tiles (`nn.ffn`, `nn.rmsnorm`, `nn.summed_embedding`), the first
  FFN's hidden tiles crossing from rank 0 to rank 1 and the second's back, each tile as it finishes (a queue per row
  section and direction); one warm instance, then eight supplied together, row section 0 of the first withheld until
  its output's section 1 is seen. Frontier (measurements/literature-pipeline-2026-09-13, mesh 82be91f, the deleted
  native executor): paired CPU **675.0** invocations/s, first section 4.9 ms; paired Metal **172.6**.
- `examples/streaming-chain.py --chain tensor-parallel --input X.npy --up-weight U.npy --down-weight D.npy
  --consumer-weight C.npy --split S`: the 09-14 chain, each member's FFN over its hidden columns, `collective.reduce`
  to the root, optional `--normalize` (`--residual`), swish, the consumer's contraction
  (measurements/tensor-parallel-chain-2026-09-14.txt records its outputs, no timing).
- `examples/streaming-overlap.py streamed|whole`: `Y = (X @ W) @ V`, rank 0 producing `X @ W` tile by tile, rank 1
  consuming each tile as it lands and returning `Y`'s tiles (a queue per input slot and direction); streamed in
  64-row tiles against whole (one tile), the same arithmetic. Frontier (measurements/streaming-overlap-2026-09-13.json,
  mesh fb7828b: `--rows 4096 --depth 2`, 10 warm then 3,000 measured): **1,429.2** results/s streamed against
  **962.9** whole (+48.6% paired), 1.326 ms against 2.007 ms a result.
- `examples/sync-on-remote-fill.py --mode stream|sync|deadlock|independent`: the explicit wait and its counterexample
  (`deadlock` ends in the program's bound, TimeoutError, rather than waiting forever); section 0 rides queues of its
  own, so `independent` (section 0 never written) completes every other.

`--trace FILE` keeps each call's (kind, record, row, begin and end ns, thread); the literature summary counts the
crossings that overlap a kernel on the same member.

## report statistics

B. P. Welford, [Note on a Method for Calculating Corrected Sums of Squares and
Products](https://doi.org/10.1080/00401706.1962.10490022), *Technometrics* 4(3),
419–420 (1962). The existing engine public-path report uses the online recurrence
`n += 1; delta = x - mean; mean += delta/n; M2 += delta*(x - mean)` and reports
sample variance `M2/(n-1)`. The inputs are observed request completion periods and
TTFT, outside every tensor function and mesh progress thread. Completion periods
include the interval from batch submission to the first completion, so their mean
is total batch time divided by request count. The [report contract](../../../metal-microbench/docs/amdahl_superiority.md#public-prefill-report)
distinguishes concurrent throughput, request latency, and the units of mesh bounds.

## Device operands

The JAX authors' [Pallas reference and indexing model](https://docs.jax.dev/en/latest/pallas/design/design.html),
Apple's [Metal buffer GPU addresses](https://developer.apple.com/documentation/metal/mtlbuffer/gpuaddress)
and [queue residency sets](https://developer.apple.com/documentation/metal/mtlcommandqueue/addresidencyset(_:)),
and the MLX authors' [system-coherent Metal accesses](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal)
provide the mechanisms used here. This is operand representation; numerical
functions remain supplied by the caller.

The page-table entry is 16 bytes, `{ mapping, address }`, 16-byte aligned, its size and offset checked by a static
assertion (rdma/mesh.h:47-52). ABI 77's 32-byte entry `{ mapping, hostAddress, deviceAddress, stamp }`, whose C type
kept 16-byte alignment for the Swift module's import, is in no source.

`mesh_device_bind` resolves Metal addresses during setup. For a local operand it
writes its canonical entries. For a receive pool it writes the address immediately
before each existing wire tag, in receiver-local metadata outside the receive SGE.
RX loads that address beside the received tag and writes it beside the CPU address
before release-publishing the stamp. These are the actual registered payload pages,
not placement copies. The configured Metal device's addresses are used; CPU and GPU
virtual addresses are never presumed equal. The extra RX address load is in the
same cache line as the tag, and the stamp store now shares the canonical address
record's cache line. This is a source-level locality account, not measured latency.

`TensorPart.stamp` is the CPU address of the first instance's canonical word;
nonlocal parts have no local address. TensorPart remains a value of primitive
fields and pointers without an owning object. Per-instance Metal bindings below
provide the corresponding GPU table addresses, including the stamp at byte 24.
`MeshSpan.metalOffset` is the offset inside its page-aligned `bytesNoCopy` buffer;
metadata users bind both buffer and offset. Existing page-aligned tensor spans
retain offset zero.

`TensorFunction.metal(device, operands:)` realizes per-instance `MeshMetalOperand`
arrays before execution. A Metal operand contains its canonical table binding,
resource lifetimes and an optional contiguous local buffer. Each received chunk is
wrapped independently; the whole receive arena need not fit one Metal buffer.
The native launch owns the backing buffers, and its queue has a residency set
installed at setup. The shader receives resources through their GPU addresses;
there is no per-launch walk declaring every possible receive buffer.

`MeshMetalOperand.source` supplies `mesh::contiguous<T>`: it resolves one address
and length for a contiguous section of the logical operand. `pipeline` specializes
its indexing with Metal function constant 37. That constant is reserved by this
shader ABI and does not change the numerical function. Address resolution happens
per contiguous section, not per scalar. Allocation, Metal wrapping and pipeline
compilation are setup work. The engine's `add_inplace` uses the same `matrixAdd`
body for ordinary pointers and for these sections; its arithmetic is not duplicated
inside Mesh.

This integrates device-side address resolution into the E1d decode sum. The
[resident Metal path](#resident-metal) now uses those bindings in a command buffer
submitted before the layer's remote inputs arrive. No floor latency or end-to-end
improvement follows from the ABI or compilation alone.

## Resident Metal

Apple's MLX authors supply the system-coherent payload accesses and system-scope
fence mechanism in [fence.metal](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal)
and [fence.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/fence.cpp).
The shader adaptation (rdma/mesh.metal's waits, payload loads and publication) is attributed to Apple (2024),
under the [MLX MIT license](../licenses/mlx.txt).

## Native Metal program

Apple, [Encoding indirect command buffers on the CPU](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu),
supplies the native command reuse mechanism. Apple's
[`MTLIndirectComputeCommand`](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand),
[`setBarrier`](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setbarrier()),
and [`supportIndirectCommandBuffers`](https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/supportindirectcommandbuffers)
define compute recording, dependencies and pipeline compilation. This is native
Metal replay, not recommitting an ordinary command buffer.

The engine's `MetalProgram.bind` consumes setup-only dispatch descriptions.
The earlier `bindMetalProgram` entry point is replaced by setup composition. For
multiple commands or indirect reads, it records the supplied pipelines, buffer
bindings, scalar constants and grids into one `MTLIndirectCommandBuffer`.
Constants are packed UInt32 metadata in shared
storage; tensor payloads retain their supplied storage. Setup deduplicates read
and write resource declarations, captures pipeline lifetimes, and discards the
dispatch descriptions. Invocation creates one compute encoder, declares two
prepared resource arrays through Metal's native bulk API, executes the recorded
command range and ends encoding. There is no runtime dispatch-description walk,
pipeline selection, scalar packing, buffer rebinding or Swift resource walk.
The native resource declarations still incur Metal's own work and are not free.

A single command without indirect reads instead captures direct pipeline, grid
and bulk buffer bindings at setup. Invocation creates an encoder, sets the
pipeline and prepared buffers, dispatches and ends encoding. There is no native
indirect command or separate resource-declaration call in this lowering. Ordinary
matrix products use the same direct implementation, replacing their duplicate
closure. This setup choice adds no runtime branch or descriptor traversal.

The actual five-stage engine sampler uses this program in ordinary decode,
prefill and the resident Mesh decode function. `sampleTokenCommands` and
`logprobCommands` describe the existing numerical functions; `bindSampling`
records them together. The four explicit device command barriers preserve their
data dependencies: local statistics precede mass computation and sampling, and
candidate/logprob reads follow remote-input acquisition. These replace the
prior separate serial compute passes. No host wait, remote-fill library guard,
or transport synchronization is introduced. Indexed sampling additionally
declares its local projection output as a read resource; a metadata buffer alone
would hide that dependency from Metal. Existing Mesh frame residency covers the
remote payload, and the supplied shader retains its remote visibility fences.

`bindDecodeOutput` binds final normalization, projection and the prepared sampler
during setup. The ordinary output encoders use the same prepared sampler. The
existing numerical sampler entry point also uses this construction instead of
retaining a second imperative implementation.

The same construction records each complete FFN sequence in
`DenseFFNWeights.bind`: normalization, the existing fused gate/up matrix operation
and down projection, or the existing three/five-kernel batched GEMV sequence.
`gateUpCommands`, `denseGemvPipeline` and `rmsNormCommand` realize native pipeline,
operand, offset, grid and scalar bindings once. `MatrixOperations.dispatch`
shares its preparation with ordinary `multiply`; the latter still captures direct
native calls, without an indirect program around an individual kernel. The old
split `bindFFN` wrapper and paired up/down invocation are deleted. CPU encoding
shrinks from three/five encoders to one per replayed FFN, with five native calls
per invocation. Native device barriers preserve the existing normalization,
activation and down-projection dependencies. The MPS and ordinary matrix-prefill
branch retains its existing native encoders.

The [matrix product](#kernelsdot) sums over exactly the operand's inner
dimension. The pre-existing
FP16 GEMV implementation read complete 32-element groups even for a shorter final
group; it now bounds that loop by the actual input width. Output launches cover
`ceil(columns/32)` groups, and inactive lanes use a valid weight address before
the common threadgroup reduction and masked store. `BatchPipelines` supplies the
actual occupancy as a Metal function constant; `ENTRY_DENSE_GEMV` specializes its
existing bounded implementation for non-power-of-two batches. This preserves
caller placement and prevents reads/writes of absent rows without adding a
readiness guard, transport condition or runtime pipeline choice.

The engine's [FFN source account](../../../metal-microbench/docs/async_collectives.md#prepared-ffn-programs)
records the before/after encoding counts and limitations. Both engine libraries
build; 126 affected native pipelines compile, and Metal/tensor FFN projection
programs record with 1, 8 and 128 rows and nonzero offsets without GPU submission.
The native Metal branch of `bindLayerFinish` records the six existing E2B
post-sum operations together, using the same matrix setup factory and existing
normalization, activation and scaling shaders. MPS retains its native path;
non-PLE models retain their existing single fused encoder. Fixed PLE offsets,
strides and launch grids are recorded once. The five local dependency boundaries
replace the former separate serial passes. The engine's
[finishing account](../../../metal-microbench/docs/async_collectives.md#prepared-layer-finishing)
counts six encoders/45 native calls becoming one encoder/five calls per layer.
Resource-array lengths are captured at setup too: optimized invocation SIL
contains neither of the former two array-header count loads. This does not remove
native resource tracking or captured-value loads. The three additional pipeline
configurations compile with indirect support, and both engine libraries build.
`qkvCommands` realizes the existing staged/batched/separate projection sequence;
`bindDecodeAttention` records that sequence together with the existing Q/K
normalization and RoPE and V normalization. Matrix projection configurations
retain their numerical encoders and record the following transforms. KV-sharing
layers record only their query preparation. Current positions remain buffer data,
and QKV capture still follows preparation. Native recording neither changes
attention arithmetic nor expands the declared KV resource set.

`PerLayerInputs.begin` records the seven-operation prefix on native Metal matrix
backends, using the existing projection views, embedding layout and numerical
functions. MPS retains its encoder sequence. There is no new tensor storage or
copy. The [source account](../../../metal-microbench/docs/async_collectives.md#prepared-qkv-and-ple-prefix)
states the removed native encoding calls and the unchanged argument/launch
contracts. Both engine libraries build; 35 affected pipeline configurations
compile with indirect support without GPU submission.

Apple's [residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)
and [`MTLResidencySet`](https://developer.apple.com/documentation/metal/mtlresidencyset)
supply persistent allocation declarations and setup residency requests. The
engine now fills its existing KV set once with the unique configured cache
buffers, attaches it to the ordinary queue and captures it for one attachment
per Mesh step. It deletes the decode page scan, per-layer KV buffer walk,
visited-chunk set and pin-on-growth callback. Full configured capacity is now
requested at setup; the engine's [cache account](../../../metal-microbench/docs/async_collectives.md#prepared-cache-and-attention)
reports that resident-memory scope explicitly. Residency does not track hazards:
`attentionCommands` records the existing attention/reduction pair, and the
producer prepends the existing KV scatter with a device dependency before
attention. Reduction retains its numerical dependency too. No kernel arithmetic,
tensor backing or remote-fill condition is added. The bound row count determines the
KV grid. Vocabulary projection and publication are now recorded together;
token embedding and streaming sums now join the complete native step recording.
Replicated sampling and the E1d/E8 floor remain open.


Engine `fb3bab4` uses `MetalProgram` to compose its existing command
descriptions before native recording. `append` resolves relative command and external-encoder positions and
preserves each numerical dependency. `bind` stores native commands, packed scalar
constants and deduplicated resources once. Its inline native encoder body makes
five Metal calls without a command-description walk. Ordinary single commands
retain direct bulk binding. Apple's
[`setThreadgroupMemoryLength`](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setthreadgroupmemorylength(_:index:))
records the dense-prefill kernels' existing dynamic threadgroup storage too.

`decodeAttentionProgram`, `DenseFFNWeights.program`, `layerFinishProgram`,
`unembedProgram` and `samplingCommands` supply the existing numerical sequences.
`denseMmProgram` and `realizedMatrixProgram` share native matrix preparation with
the ordinary engine; MPS remains an actual native external encoder where selected.
`meshDecodeLayerProgram` includes the unchanged supplied streaming sum. The full
`bindMeshDecodeStep` now binds one concatenated sequence: embedding, PLE prefix,
all layers, output normalization, vocabulary publications and rank-0 sampling.
Prepared diagnostic ranges refer to the same recording. The one runtime
inspection predicate preserves changing QKV capture/ablation requests; its
ordinary branch enters the native body directly. Only explicit instrumentation
or real external MPS calls execute a segmented callback sequence.

The [whole-step source account](../../../metal-microbench/docs/async_collectives.md#whole-step-native-recording)
records the 251/250-to-one encoder reduction, corresponding native call counts,
source growth, exact scope and remaining limitations. This deletes per-layer
rearm invocation and repeated binding; it does not change the collective routes,
shader arithmetic, operand backing or device visibility dependencies. Source and
compilation evidence are not a measured floor-latency pass.

## Prefill KV

Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
(2017), supplies the projected query/key/value algebra. Dao et al.,
[FlashAttention](https://arxiv.org/abs/2205.14135) (2022), supplies the tiled
attention reference. This change reuses the engine's existing projections,
normalization, rotary transform, KV scatter and attention kernel; it adds no
numerical operation to Mesh.

E1d's replicated decode attention consumes all KV heads on every rank. The
engine's head-partitioned prefill formerly wrote only its assigned heads, with
that smaller head count as the physical cache stride. A rank assigned no heads
wrote nothing. Those caches could not be read as full-head caches at decode.

`bindPrefillKV` now binds the existing full K/V projection and write sequence,
shared by standalone prefill and Mesh. A rank computing attention executes it
inside its existing attention command buffer. A rank with no attention heads
executes it before its FFN stage, using the original hidden inputs already bound
to that stage; `missingKV` prepares that input normalization and the existing KV
function. The Metal and Core ML FFN compositions both use this preparation.
The existing prefill matrix views request half-precision conversion explicitly
from `realizeDenseMatrix` at setup; the packed weight buffers and decode functions
remain unchanged. Prefill can therefore be constructed without switching the
engine-wide backend just to satisfy its weight-view representation.
The FFN partial is published only after that command buffer completes, so the
existing FFN reduction dependency also carries the completed KV write. No extra
collective, host callback or command buffer is introduced.

Prefill readers keep their local query/KV grouping while specialization constants
bind the full physical KV head stride and the first selected head. For page p,
position t, head h and component d, the address is
`((p * PAGE + t) * fullKVHeads + firstHead + h) * headDimension + d`.
The same reader remains the standalone kernel when those constants are absent.
Shared-KV layers reuse their configured source layer and do not recompute KV.

The existing caller placement accepts an explicit `prefillRows` partition of
`B * MAX_Q_LEN`; `placement.py --prefill-rows` writes that partition from the
same caller-supplied rates. Decode's replicated batch and prefill's token-row
partition are distinct dimensions, so B=1 no longer forces an empty prefill
operand for the second rank. The old batch-row interpretation remains the
fallback for existing placements. Setup checks the complete token-row partition.

Distributed decode now always constructs its associated prefill graph. The
prefill region defaults to the decode region plus `-prefill`; its instance count
defaults to the decode count. The existing explicit prefill region/count settings
still override those defaults. Both regions come from the bridge setup.
The switch that left all prompt computation on the leader is removed.

The source path covers ordinary and continued distributed prefill, not imported
leader-only prefix-cache snapshots. Numerical KV equivalence, public decode,
latency and speedup are still unmeasured. It implements E1d's replicated cache
preparation, not F11's separate sequence partition; F11 remains open.
