# Algorithm references

The [user's requirements](collective-goals.md) define scope. These sources supply
mechanisms, not additional features, architecture, tests or prerequisites. Section
names retain existing source citation anchors; they do not prescribe public APIs.
The deleted implementation is not an implementation template.

## Program

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
reference for higher-order numerical calls over indexed tensor operands.
The implementation realizes a finite storage capacity once. A section's
`first + slot * stride` selects its logical row; shared constants have zero
stride. Working ABI 55 carries invocation labels separately from reusable slots.
Submission publishes root indices into existing numerical-worker queues,
while consumers are indexed by operand publication. No function scan or repeated
realization is required. The [execution description](async-collectives.md#execution-and-ownership)
separates shared function metadata from each value's operands and uses.

## Program.tensor

The JAX authors, [BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html),
and Apple [mmap](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html):
references for indexed sections and virtual mappings of actual shared backing.
`mesh_operand.load(at:as:)` applies that logical indexing to naturally aligned,
fixed-width tensor scalars. Its C address calculation uses the operand's canonical
mapping and realized page geometry. For byte offset b and transport
extent C, it reads entry floor(b/C), then addresses b mod C within that backing.
It does not allocate, check presence, scan pages or materialize the operand. The
helper is inlined into supplied Swift numerical functions. Indexed operands carry
one mapping pointer and two geometry integers; the current operand is 48 bytes.
The first-page
`data` pointer alone does not describe a discontiguous input of `bytes` length.
Local outputs remain contiguous and retain their direct pointer interface.
The indexed scalar API describes logical elements; neither its indices nor the
caller function changes when the bridge's transport extent changes.
ABI 63 removes ABI 58's redundant page-list offset and `mesh_buffer_pages`.
Every list starts at its buffer's logical row, so chunk j is directly
`mesh_page(m)[row + j]`. `mesh_row_page` selects that entry for existing native
placement paths. Logical
metadata extent is independent of payload-page extent, and retain/release acts
on one buffer identity. The [compact-list derivation](pages-and-functions.md#prepared-compact-page-lists)
gives the address algebra, allocation counts and metadata cost. Neither the
indexed scalar helper nor receive placement performs packing or allocation.
George E. Collins, [A method for overlapping and erasure of lists](https://doi.org/10.1145/367487.367501)
(1960): reference counting. The user explicitly requested automatic ownership
release and background pool return, without caller free/done calls.
The implementation groups shared-input ownership by prepared function binding,
while transient inputs retain one reference per indexed use. Native completion
releases those input references after publishing outputs; downstream readers own
the outputs separately. Each prepared call holds its frame's input countdown
and output storage, selected by the realized consumer records.
The [lifetime derivation](pages-and-functions.md#what-the-page-table-is) describes
completion, cancellation and destruction. This is an application of counted
ownership to known lifetimes, not a new collection algorithm attributed to Collins.
`TensorPart` contains primitive values and an optional C `mesh_section`, with no
`MeshSection` object. The descriptor carries its receive channel, with
`MESH_ABSENT` for local storage; the separate Swift receive-channel field and C
receive boolean are removed. Setup owns one reference per allocated section in
`MeshMemory.sections`. `releaseSections` drops those references after binding and
empties the array; the same method returns outstanding setup ownership during
destruction. There is no per-part ARC release on the numerical path and no
descriptor-retention reference outside the declared dataflow. This changes
ownership representation, not the numerical operand ABI or transport framing.
Every retain occurs while setup still owns the section, so it is one relaxed
increment; there is no attempt to resurrect a zero-reference value. Execution
only releases declared uses. The retain CAS loop, rollback path, sealed flag and
seal operation are deleted. ABI 49 also removes the reclamation claim, retry stack
and collector thread. Refzero sets one row bit in the section free pool; setup
consumes those entries without reading refcounts or zeroing payload. The bridge
discharges abandoned positive counts at the device-close retirement event.
ABI 53 keeps received backing in its RX-owned pool, including when no live value
occupies a row. Refzero notifies that RX thread; it returns both backing and the
logical row. Setup captures the binding's reference-count template, and a new
value reinstalls it on a popped row. Client/device retirement returns the entire
receive pool, including zero-count posted rows, only after QP teardown.
The [event derivation and limits](pages-and-functions.md#reclamation-events)
distinguish this bitmap pool from the still-required X5/N1 per-worker instance
rings. Counted ownership follows Collins; the bitmap and teardown epoch are Mesh's
representation, not a new algorithm attributed to that paper.
Contiguous sections use chunk-indexed backing and relative numerical byte offsets,
with one presence bit and ownership record per numerical value. The
[address and ownership derivation](pages-and-functions.md#block-addressing)
shows the receive permutation and the producer's single initial reference.
These are mesh's representation choices, not new algorithms attributed to Pallas
or Collins. Transport chunk counts do not define tensor partitions.
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

## TensorPart.partial

The PyTorch authors, [DTensor Partial](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.placement_types.Partial): pending reduction is a declared value property. Mesh marks contribution and intermediate views and clears the completed reduction view. A completed reduction receives a new logical identity over its existing storage, including when its contribution list has length one. There is no copy or extra numerical call for that identity change.

Movement preserves the logical identity, so a contribution's setup restriction follows sends declared before or after its reduction. The delivery key includes source rank as well as logical identity, destination and queue; explicitly different transport edges remain different. The original value-type snapshots are not mutated. The setup contribution set recognizes those earlier snapshots during validation and is discarded before numerical workers start. No runtime function reads `partial` or the logical identity. See the [L5 source derivation](distributed-reduce.md#l5-contribution-typing).

## MeshError

The Legion authors, [reduction privileges](https://legion.stanford.edu/tutorial/privileges.html): reduction operands have restricted uses. Mesh's public `call` records one validation closure, executed by `start()` after all reductions have been declared. Its sole throw reports `partialOperand` with a partial view of the offending operand. The reduction's supplied combine uses the same private binding path without that validation. This is a declaration-time restriction, independent of backend and numerical completion order.

## Mesh.result

Saltzer, Reed and Clark, [End-to-End Arguments in System Design](https://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)
(1984), motivates keeping recovery with the caller. Mesh records native errors;
it does not retransmit an operation or make its consumers wait for recovery.
ABI 62 uses one atomic 16-byte status snapshot inside the existing aligned
32-byte frame record: `{value, completed}`. `value` retains the two kind bits,
thirty function/link ordinal bits and thirty-two native-code bits. `completed`
is the last successful invocation plus one, zero before success, or `UINT64_MAX`
for completed static-only work. `mesh_calls_result` returns success for that
completed invocation and otherwise returns the fault or busy. The Swift decoder
is unchanged. The compile-time assertion requires a lock-free 16-byte snapshot;
on this ARM64 compiler the acquire read is `ldp` plus `dmb ishld`, with register
comparisons and no polling loop. This replaces ABI 61's single 64-bit code word,
which could lose the prior success on link failure and therefore violated F10.

Submission records the invocation before publishing roots. RX records it after
`mesh_publish` and before dropping the buffer's existing producer reference;
the 32-byte receive record supplies the frame index from setup. That ownership
event cannot release the frame's final reference before the invocation store.
The last reference publishes success with that invocation. No passive-arrival
busy store, extra notification or transport permission check is added.

`mesh_result_conclude` never overwrites an error. Normal success makes one strong
compare-exchange. Fault publication preserves `completed`; if its compare-exchange
loses to a concurrent successful completion, it retries with that observed
snapshot so the completion is retained and the fault is not lost. This retry is
confined to error reporting; it neither retries a transfer nor waits for remote
work. An existing error ends it. Consequently a link error can report failure
for pending work while the last completed invocation still returns success.
A new explicit submission reassigns its resident slot; callers retain older
results themselves. This does not establish N1's cross-participant ownership
bound. A passive rank's explicit `submit` now reads the program fault before
returning success for its lack of local roots; reception still needs no submit.

The existing Core ML chain now submits a bounded window and observes each
invocation's result before reusing its window position, including on ranks whose
work arrives solely from peers. It reports native failure through the same
public `Result` path. This adds no library scheduling path or numerical kernel;
it exercises the status API in the existing arbitrary-stage producer/consumer
composition. No runtime or killed-peer result is claimed.

Collins's reference counting, cited above, supplies the ownership mechanism.
Setup counts numerical and transfer references per frame. Final function/transfer
release decrements the frame's atomic count directly; zero restores its recurring
template, concludes status and marks it available. Shared-transfer completion
releases one initial reference from every frame. The lifecycle thread and its
event rings/arena storage are deleted: reference release needs no handoff to a
second poller. ABI 60 prepares the frame-reference pointer in each aligned
32-byte send record and the frame index in each receive descriptor. These are
applications of the same direct-address mechanism: completion does not reload
an invocation label and divide it to reconstruct ownership. There is no hash,
tombstone, label-to-frame search or
submission-association event. The current selected-frame admission and missing
whole-plan reuse proof are specified explicitly in
[frame identity and reuse](pages-and-functions.md#invocation-identity-and-storage-reuse).
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
to wake the controller through its preallocated user event. The controller shuts
down the socket, joins TX/RX, and closes QPs/CQs. Its event queue remains open
until bridge shutdown, after every controller has joined, so a stop notification
cannot target a recycled descriptor. Event registration and notification use
`KEVENT_FLAG_IMMEDIATE`; only the separate controller's event observation blocks.
No heartbeat, elapsed-time failure inference, or healthy-path TX/RX socket query
is added. The first program and instance errors and each frame's last completed
invocation are preserved. The old listener is closed after QP teardown, so a new
realization opens a fresh listener without the prior connection's backlog.
This implements event observation in source, not complete R2 cancellation or
R3–R7 recovery. An unreported silent partition is not detected by inventing a timer.
No caller-death or cable-loss run accompanies this change.

`mesh_rows_alloc` now records the client owner for metadata rows as well as
operand rows. `mesh_retire` clears their row ownership even when `pages == 0`;
only actual operand buffers receive the closed-storage flag. Previously a dead
client's root and return rows had owner zero and survived replacement, leaking
row capacity across realizations. Normal client retirement or replacement of a
dead client now covers both. Actual backing retains the existing device-close
retirement rule. Link error alone still does not cancel unissued numerical uses
or rearm the same program: those R2/R4 steps remain unfinished.

## Program.kernel_call

ABI 59 implements Monsoon's frame-indexed activation directly: the prepared
consumer record includes the frame, its input updates that call's operand, and
its countdown reaching zero invokes the supplied function in the same frame.
The invocation hash, join/free tables and independent native-slot ring are deleted.
The [current lifetime contract](pages-and-functions.md#native-slot-return) supersedes
the older slot-assignment descriptions below. N1 remains incomplete.

Collins's reference counting releases a numerical input at the completion of its
own declared use. `mesh_call_complete` performs those decrements directly for
CPU, Metal and Core ML, after publishing outputs. Publication retains each
output's existing producer reference through this cleanup, then completion
releases those references. Final output return therefore implies that the
callback has finished using its operand metadata, without an additional native
completion notice or counter. The success-only completion wrapper is deleted;
Swift passes zero to the implementation on success. Constants and RX publication
also release their producer reference explicitly after publication. The worker
drains publication notices before return cleanup and rearming; both use their
existing indexed notification queues.
Within publication, all declared TX notices precede the presence store and local
consumer-mask load/notifications. Forwarded receives follow the same ordering;
the wire-facing notification does not depend on local-consumer bookkeeping.
The [chain and fan-out derivation](pages-and-functions.md#input-lifetime-ends-at-its-own-use)
distinguishes this local storage fact from unfinished global frame reuse.

Gregory M. Papadopoulos and David E. Culler,
[Monsoon: an Explicit Token-Store Architecture](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-F03/handouts/papers/p398-papadopoulos.pdf)
(ISCA, 1990), supplies the operand-associated state-bit mechanism and statically
assigned token locations. Mesh applies those ideas to notifications: each reader
has a bit for each logical row, with compact 64-way summary words to enumerate
pending rows. The software summary layout is Mesh's implementation, not code or
a queue algorithm copied from Monsoon. `mesh_notice_push` sets the row bit then
its summaries using release operations. `mesh_notice_take` exchanges indicated
words with acquire semantics and enumerates their bits. There is no reservation
cursor, CAS retry, linked entry or wait for another publisher. `mesh_layout`
realizes the word offsets; `mesh_notice_reader_init` binds one reader's pointers.
The [publication proof](pages-and-functions.md#publication-notifications) covers
concurrent writers, delayed summaries and reuse. This is a pending-event set,
not a scan of tensor presence or function readiness. The declared consumer ranges
and countdown continue to implement firing.
The working ABI 55 implementation keeps invocation-keyed join records separate
from native call slots. Linear probing matches an arriving logical row; countdown
zero binds the accumulated row indices to a native slot and removes the match by
backward shifting indices. Join and native indices have separate owner-managed
rings. This is Mesh's software representation of operand matching, not an
algorithm claimed to have been copied from Monsoon. Input ownership follows
Collins's counted-lifetime mechanism through final output ownership. The
[native-slot derivation](pages-and-functions.md#native-slot-return) proves the
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
(1997), supplies the contraction decomposition into products of operand panels.
The caller's `contract` in `examples/gram-chain.swift` declares each two-input
product independently and combines K-panel contributions with `reduceScatter`.
This uses SUMMA's panel algebra; it does not copy its MPI broadcast schedule.
The supplied numerical function calls Accelerate's Level-3 BLAS `cblas_sgemm`;
the supplied combine calls `vDSP_vadd`. The same composition implements the
rectangular projections and both products of `R(RᵀH)`, with dimensions and
transpose flags captured at declaration. Mesh contains none of that arithmetic.
The [G1 derivation](function-chain.md#g1-gram-and-projection-chain) gives the
operand identities, placements and repeated residual composition.

Apple's [Core ML Tools MIL builder](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html)
and compiler create the supplied model artifacts in `examples/coreml-models.py`.
The exporter composes existing `matmul`, `gelu` and `add` operations for the
[documented FFN residual chain](function-chain.md), with caller-supplied dimensions
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
costs](pages-and-functions.md#invocation-identity-and-storage-reuse) separate this
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
The current use record names the prepared `mesh_call` and, when required, the
input `mesh_operand` and canonical first-page entry directly, retaining its
32-byte size. Call records occupy 64 aligned bytes with an asserted layout;
the former 40-byte array elements could cross cache lines. Setup writes the
source row, logical index and view's page-list address into every operand.
`mesh_call_input` and the function's separate placement-flag array are deleted.
A local input or preallocated contiguous view requires no arrival-time rebinding.
An unplaced remote input refreshes only its current first page and address;
that choice is represented by the prepared operand pointer. Its numerical use
still owns the canonical page entries through native completion. A shared remote
input refreshes the corresponding view in every resident call and decrements
each call's dependency count; the function's shared-input template is updated
once, before any of those calls fire. Source identity remains distinct from a
contiguous view's storage, including when the input was already present at setup.
The call's prepared submit function, argument and operand counts now occupy the
unused space in its existing 64-byte record. Launch reads no function/program
object. Collins's counted ownership applies to the complete declared use set:
startup preserves the compiled output counts, and final output return restores
them before releasing the frame. This removes refcount increments and presence
clearing from launch, as derived in
[output ownership](pages-and-functions.md#output-ownership-is-prepared-before-launch).
The readiness function takes the call directly and reads its function only when
its pending count reaches zero. These changes apply Monsoon's prepared activation
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
feature providers and output options. The [native slot-return derivation](pages-and-functions.md#native-slot-return)
accounts for return events, output lifetimes, non-submitting ranks, metadata cost,
and the still-finite call/status namespace. Collins's counted ownership and the
existing row-notification mechanism supply the return events; one numerical
worker owns each free-index array. This supplies ordinary native storage reuse,
not X5's unfinished instance-admission rings or R2 failure cancellation.

## Program.map

The JAX authors' [Pallas scalar prefetch](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)
and [jax.lax.gather](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html),
and Gale et al., [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
(2022), supply references for block indices as data, indexed reads, and block-sparse routed experts.
Indices are ordinary operands of `Mesh.map`; the supplied function performs the indexed
read and any expert selection or neighbourhood sum inside the caller.
Mesh implements no indexed gather/scatter or application routing.
The existing `indexed-gather` caller uses `load(at:as:)` for both table values and
index operands. Owners, consumers, routes and dimensions come from its JSON plan.
`examples/indexed-gather-ring.json` sends values from rank 0 and indices from
rank 2 to consumers on ranks 1 and 3. Each numerical table is 8192 × 513 × 4 =
16,809,984 bytes, larger than the substrate's maximum single request. Every
consumer uses the same supplied functions without a transport-size argument,
contiguous placement section or one-peer branch. This is source usage, not a run
or a performance claim. With the ring links configured, build with
`make -C rdma indexed-gather`, then invoke on each rank:

```
rdma/indexed-gather RANK 4 /mesh0 examples/indexed-gather-ring.json
```

## Program.copy

ABI 59 realizes a peer-qualified, source-chunk-indexed table of aligned 32-byte
receive records. The eight-byte tag names the 32-bit sequence and source chunk.
ABI 60's record supplies the destination row, buffer address, exact canonical
page-entry address, reference count and chunk flags. This is direct addressing
of the known transfer relation, applying
the existing compressed-row/direct-index mechanisms; it introduces no collective
algorithm or credit protocol. The [wire layout](pages-and-functions.md#explicit-section-identity-on-the-wire)
accounts for table storage, shared-tag writers and the unfinished N1 lifetime proof.

Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
MLX authors' [JACCL transport](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h),
and rdma-core's [ibv_post_recv](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3):
registered SEND/RECV and work completion. The [hardware notes](collective-dependency-ledger.md)
distinguish substrate facts from the retained bridge's protocol decisions.
TN3205 explicitly limits this transport to `IBV_WR_SEND`; its SDK enum for
`IBV_WR_SEND_WITH_IMM` does not establish hardware support. Both TN3205 and JACCL
show local `wr_id` values returned with completions. Mesh uses these identifiers
for buffer lifetime. ABI 59 carries one `(sequence, sourceChunkRow)` word. ABI 60 resolves the
chunk's local row and page-entry address at setup, alongside its publication flag.
Request extents belong to setup, before a native request is submitted. TN3205
requires paired SEND and RECV requests to occupy the same number of 4 KiB
frames. ABI 60's exact-tail SEND change violated that requirement whenever the
shortened SEND crossed fewer frames than the preposted RECV. That change and
its claim of padding-free wire traffic are withdrawn. Setup chooses one matched
extent per queue direction from its declared transfers; native posting neither
truncates nor infers a length. Logical operand lengths remain exact. The
[framing accounting](pages-and-functions.md#prepared-native-requests) includes
padding explicitly, independently of numerical partial boundaries.

`link_configure` prepares a native RECV WR and SGE per pool block and a native
SEND WR and SGE per resident send edge. RX posts the prepared request selected
by its pool index. TX selects the current backing address and registration key,
then posts its prepared request without changing the length. `link_post` and
`last_bytes` are deleted. The local Apple SDK defines a 128-byte `ibv_send_wr`,
32-byte `ibv_recv_wr` and 16-byte SGE: a complete SEND WR plus SGE cannot fit the
former H2 limit of 64 bytes. The receive request occupies 64 aligned bytes;
the send record occupies 256 aligned bytes, with its SGE, application fields
and common SEND header in the first 128 bytes. Compile-time assertions check
these layouts; they do not prove one-line provider access or the whole H2 path.
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
semantics, request extents or the source-lifetime completion rule.
The tag arrives in the payload's own work request. There is no index QP or
cross-QP identity join. On September 15, `ibv_devinfo -v` reported `max_sge: 1`
on the local Thunderbolt devices. The alias representation uses one SGE and
does not infer two-entry support from the general verbs API. The
[record layout and execution path](async-collectives.md#execution-and-ownership)
describe this mesh-specific representation. Receive storage is preallocated
and reused by the owning RX thread at final reference release. The existing
notification mechanism carries the return event; only RX writes its page and
return ring. Each queue poll returns one completed section's backing,
rotates unfinished sections and attempts an available post. It reposts actual pages without clearing them and
indexes the prepared source-chunk record. The [receive-storage proof](pages-and-functions.md#receive-storage-return)
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
adds no queue wait or native completion hop. The [data-flow and cost account](pages-and-functions.md#indexed-receive-runs-and-contiguous-consumers)
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

`Mesh.send` expands [explicit placement routes](#placement) before binding these
transfers. At an intermediate rank the received section itself is retained by
the onward SEND. No additional function, operand copy or runtime route lookup
is introduced.

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
the [edge-lifetime bound](pages-and-functions.md#reusable-send-queue) replaces any
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
associate QPs and registered memory with a protection domain. Mesh now keeps
those device resources once per named device while realizing a separate connection
and TX/RX progress pair for every configured link. Configured peer identities,
channels and operand row ranges are resolved at setup. Publication scatters its
row index to those links' queues through a precomputed membership bitset, independent
of collective verb and peer count. Multiple links register the same canonical
payload backing through their selected devices; they do not create operand copies.
The [configured-peer description](async-collectives.md#configured-peers) records
setup, per-link preposting, shared registration, retirement and configuration.

One pairing attempt has a single 30-second monotonic deadline, created in
`verbs_up` after device registration. `pairing_active` checks that deadline and
the existing cancellation state. The nonblocking `accept`, `connect`, `read` and
`write` paths share it across endpoint metadata, every queue's descriptors and
the initial receive-posting exchange; receiving a byte does not renew it.
Apple's BSD [connect](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/connect.2.html)
and [accept](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/accept.2.html)
contracts supply the nonblocking socket mechanism. Every socket's `O_NONBLOCK`
assignment must succeed before using it. Expiry returns `ETIMEDOUT` through the
existing link error and QP cleanup owner; it does not terminate the bridge or
reduce its configured capacity.

The configured control endpoints are numeric IPv4/IPv6 addresses and TCP ports.
`getaddrinfo` uses `AI_NUMERICHOST | AI_NUMERICSERV`, so pairing performs no host or
service lookup. Apple's [getaddrinfo](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/getaddrinfo.3.html)
contract specifies numeric host parsing; the macOS SDK's `netdb.h` specifies the
corresponding numeric-service flag. Scope-qualified IPv6 addresses remain valid.
The deadline is setup state only: TX, RX, numerical publication and consumer
completion contain no clock check. As [RDMA-RULES.md](../RDMA-RULES.md) explains,
userspace deadlines cannot unwind a driver call already blocked in kernel sleep.

## Placement

Pitch Patarasuk and Xin Yuan,
[Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
(2009), motivate supporting sparse connected topologies. `Placement` and its
directed `Edge` keys represent caller-selected paths; the library does not run
the paper's route-selection or all-reduce algorithm. A route lists the subsequent
ranks through its destination, so `Edge(0, 2): [1, 2]` means `0 → 1 → 2`.
The reverse direction has its own key. `owners` is the caller's section-owner
list; callers still pass ownership explicitly to tensor and function declarations.
The same value retains the existing work, traffic, cut and path inputs to `bounds`.

`Mesh.send` walks the supplied path during declaration and registers each directed
leg through the existing send binding. With no path entry, it requests the direct
edge. A missing configured physical edge reports the existing setup error; no
route is guessed from rank count or topology. The queue argument applies to each
leg's selected pair. Existing identical-leg deduplication remains in effect.
`Mesh.start` discards its route dictionary before workers run.

For a section s on `v0 → v1 → … → vk`, each intermediate receive section is the
actual source of the next SEND. Its receive completion publishes that section;
the existing send-use metadata queues onward transport. The intermediate owns
one producer reference until receive publication and one transport reference
until onward SEND completion. It has no extra numerical function or consumer
stamp. Setup starts compute threads only for workers with declared functions.
For the distinct realized legs E and a cut C, payload traffic is
`B(C) = sum(bytes(s) * count(s) for (u,v,s) in E crossing C)`.
Transport framing remains the separate S3 mechanism. This source construction
does not establish measured forwarding latency or repair W1–W6.

## Topology

`Topology`: MLX authors' [JACCL hostfile](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/distributed.rst#defining-a-mesh) supplies the full-mesh special case; Patarasuk–Yuan, [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) (2009), supplies tree-connectivity sufficiency.
The value stores a node set and lists of parallel links on unordered pairs. Trees,
rings with spurs, tori and full meshes use the same representation. The paper's
tree algorithm motivates admitting sparse connected graphs; source forwarding
is implemented through explicit `Placement` routes. Neither establishes that
this runtime achieves the paper's bandwidth bound.

### Topology.Pair

`Topology.Pair`: the [graph representation above](#topology), with the smaller endpoint first, gives both endpoint orders the same dictionary key.

### Topology.links

`Topology.links(between:_:)`: the [graph representation above](#topology) represents a missing edge by an empty list; dictionary subscripts remain optional.

### Topology.merging

`Topology.merging`: the [graph representation above](#topology) uses node-set union and per-pair list concatenation; the caller supplies observations and mesh exchanges none.
Concatenation preserves repeated observations, including both endpoint descriptions
of the same physical link; this operation does not deduplicate them.

### Link

`Link`: Dotan Barak, rdma-core's [ibv_query_port manual](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_query_port.3), supplies observed port speed and width.
Bandwidth is bits per second: speed codes 1/2/4/8/16/32/64/128 mean
2.5/5/10/10/14/25/50/100 Gb/s per lane; width codes 1/2/4/8 mean 1/4/8/12 lanes.
Unrecognized codes yield zero (unavailable), without withholding the link.
The bridge publishes bandwidth before the paired phase; atomic scalar access also
covers re-pairing. Device names come from the configured link. Latency is optional
seconds and remains nil in observations until the bridge records a measurement.
Port bandwidth is an attribute, not measured payload throughput.

### Mesh.observe

`Mesh.observe` and `mesh_observe`: Apple's [shared mmap mechanism](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/mmap.2) supplies a read-only view of bridge pairing metadata.
The C function returns the link count or negative errno, copies up to capacity into
plain `mesh_link_view` records, and closes/unmaps without attaching a client.
Swift resizes if a replacement region has more links, then includes only paired
links and their peers alongside the local node. Header/layout validation prevents
invalid reads; it changes no bridge state. Per-link phase reads are acquired
individually, not a simultaneous fleet snapshot. Before a client configures
transfers the current bridge has no paired links. Explicit forwarding is described
under `Placement`; idle pairing and latency measurement remain separate deliverables.

## Program.write

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
operand production. The user's contract requires publication of completed sections.
Apple's [vDSP_vramp](https://developer.apple.com/documentation/accelerate/vdsp_vramp)
is the existing numerical generator called by the configured Core ML chain's
initial producers. It writes directly into each registered operand.

## Program.export

MLX authors' [Metal evaluation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/eval.cpp):
reference for retaining operands through actual operation completion.

## kernels.expression

The JAX authors' [Pallas indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs):
values, indices and masks. This is not a requirement for a general expression compiler.

## kernels.dot

Dongarra, Du Croz, Hammarling and Duff, *A Set of Level 3 Basic Linear Algebra
Subprograms* (1990), Apple [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication),
[BNNS matmul](https://developer.apple.com/documentation/accelerate/bnnsfiltercreatelayerbroadcastmatmul(_:_:))
and [Core ML MIL operations](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html):
existing contractions. The [reduction algebra](distributed-reduce.md) describes
contributions without prescribing intermediate-buffer or launch counts.

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

## collective.reduce_scatter

Rabenseifner, *Optimization of Collective Reduction Operations* (2004), and the
MPI Forum's [collectives](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm):
reduction semantics and decompositions. A source does not mandate one decomposition
for every collective or caller placement.

## Collective movement

The MPI Forum's [collectives](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and MLX's [distributed operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp):
[distinct communication relations](collective-verbs.md).

## nn.ffn

Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), and MLX's
[tensor-parallel layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py):
local numerical functions composed with collectives. Hendrycks and Gimpel,
[GELU](https://arxiv.org/abs/1606.08415): the activation already supplied by the engine.
The configured function caller uses `Cij = Fij(Xi)` and `Yj = sum_i Cij`, then
passes each `Yj` to its supplied consumer. For linear T, Fij is its input-i,
output-j block. Reduce-scatter places each sum at its caller-selected owner;
there is no assembled whole-tensor intermediate or global stage barrier.
The [caller description](function-chain.md) distinguishes this source composition
from model artifacts and runtime performance evidence.

## nn.rmsnorm

Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
(2019): normalization used by the existing numerical implementation.

## nn.embedding

The JAX authors' Pallas indexing and the llama.cpp authors'
[GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):
references for indexed operands and the existing model loader.

## collective.sync_on_remote_fill

The MPI Forum's [communication completion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm):
explicit completion operations. The user requested an explicit counterexample;
no default collective may invoke it.

## bounds

`bounds` and its plain value types (`Topology`, `Program`, `Placement`, `Bounds`) follow
Hockney (1994, `t0 + n/r∞`), Alexandrov et al., [LogGP](https://doi.org/10.1145/215399.215427)
(1995, startup and per-byte gap), Kwasniewski et al., [COSMA](https://arxiv.org/abs/1908.09606)
(2019, I/O lower bound), and the four-lower-bounds paragraph of
`metal-microbench/docs/amdahl_superiority.md`: aggregate and per-node work over
sustained rate, aggregate and per-node traffic over bandwidth, bytes across each cut
over that cut's summed bandwidth, and the dependency path's summed startup and
transfer time; "their maximum is a lower bound, not an exact execution-time formula".
The caller supplies the placement; `bounds` evaluates it and never chooses one
(`examples/bounds-table.swift` supplies the proportional placement of the ten-minute table).
Zero work or traffic takes zero time; positive work or traffic at zero capacity
yields infinity. Missing nodes and links contribute zero capacity. These rules
come from the existing `seconds` function; its prose comments now live here.

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
