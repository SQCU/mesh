# Algorithm references

The [user's requirements](collective-goals.md) define scope. These sources supply
mechanisms, not additional features, architecture, tests or prerequisites. Section
names retain existing source citation anchors; they do not prescribe public APIs.
The deleted implementation is not an implementation template.

## Program

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
reference for higher-order numerical calls over indexed tensor operands.
The implementation realizes a finite value-index extent once. A section's
`first + index * stride` selects its logical row; shared constants have zero
stride. Submission publishes root indices into existing numerical-worker queues,
while consumers are indexed by operand publication. No function scan or repeated
realization is required. The [execution description](async-collectives.md#execution-and-ownership)
separates shared function metadata from each value's operands and uses.

## Program.tensor

The JAX authors, [BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html),
and Apple [mmap](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html):
references for indexed sections and virtual mappings of actual shared backing.
George E. Collins, [A method for overlapping and erasure of lists](https://doi.org/10.1145/367487.367501)
(1960): reference counting. The user explicitly requested automatic ownership
release and background pool return, without caller free/done calls.
The implementation groups shared-input ownership by prepared function binding,
while transient inputs retain one reference per indexed use. Native call-record
references are counted per instance before workers start; dispatch transfers
ownership from an unissued record to its native call without changing the total.
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
`mesh_calls_result` performs one acquire load. Swift decodes success, busy,
`link(peer:code:)` or `function(call:code:)` from that word. Identities are local
function or link ordinals; a setup-captured peer array maps link ordinals to the
configured ranks. The encoding uses two kind bits, thirty ordinal bits and
thirty-two code bits.

The counted ownership mechanism is Collins's reference counting cited above.
Each instance has a separate reference word: low thirty-two bits count numerical
calls, high thirty-two bits count transfers. Setup
counts all declared uses, including shared-constant transfers for every instance.
Native numerical completion decrements the low count; the final ordered SEND or
RECV chunk decrements the high count. Transport chunk count introduces no extra
result references. A zero total attempts one strong compare-exchange from busy to
success. Native failure attempts one strong compare-exchange from busy to its
error. Neither operation retries, and neither replaces an already concluded result.

Native post errors other than capacity refusal, CQ poll errors and unsuccessful
work completions end that link's invocation after publishing its error result.
The TX/RX loops return on that native error; the controller joins them and closes
the QPs. It does not re-pair and replay the same invocation. The bridge remains
available for a newly realized client. The receive-side `failed` latch and its
publication guard are deleted; successful completions do not consult a software
health flag. Outstanding references remain owned until normal completion or
teardown, so publishing an error does not release device storage early.

There is no submission reference. An unsubmitted local root already has its
numerical-call reference; a receive-driven rank has its declared receive and call
references. Their existing completions account for all work without requiring a
local `submit`. A rank with no declared work has a successful result at setup.

`mesh_call_retire` replaces the old per-call whole-program reference update.
Only the last numerical call of an instance drops that instance's program
reference. `mesh_call_finish` shares input retirement between success and failure.
`mesh_calls_cancel` retires unissued records after their workers stop; it does not
invoke numerical completion or publish their outputs. Setup reserves owner,
worker and instance references before launching any worker. Operand ownership and
the existing page collector remain separate: a result is not a free-page claim.
The worker mask is derived from declared function bindings during setup. Only
those workers acquire references and start threads; unused indices retain no
worker reference. Startup failure cancels the unstarted function range and
releases the remaining mask's worker references. A rank with only forwarding
edges starts no numerical thread, while its transfers still count toward Result.

This implements status publication for the current finite extent. N1 reuse,
detecting a remote caller's death, driver recovery and the existing W failures
remain open. The bridge currently owns the QPs beyond a caller's death and closes
its pairing socket after setup. This status word alone cannot detect that death.

## Program.kernel_call

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
ABI 50 separates lasting presence from the notification set: `mesh_presence`
addresses one 32-bit word per logical row; `mesh_publish` release-stores one.
The packed presence bitmap and its read-modify-write are deleted. Setup consumes
constant presence while realizing dependencies; the host runtime still fires
through the existing notification ranges and countdowns.

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
prefix offsets and scatters call pointers into those ranges before transport
activation. Publication traverses only its row's range. Setup also prepares local
operand addresses and records direct received-input positions; invocation resolves
only those positions. ABI 51 separates dependency rows from contiguous operand
views. A single-chunk input selects a view by its received page offset; multi-chunk
inputs use the indexed placement described under [transport](#programcopy).

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
documents blocking when a queue has no free command buffers. The SDK's
`MTLDevice.h` exposes `newCommandQueueWithMaxCommandBufferCount` and specifies
64 as the ordinary queue's default capacity. `TensorFunction.metal` takes a
device and encoder; `MeshInvocation` owns a private queue for that declared call,
with capacity for its `count` invocation records. Both direct calls and native
view factories use this constructor. Setup creates all `count` command buffers;
submission indexes that array using the existing C call index. At the k-th
construction, k-1 buffers occupy a pool of count slots, with k <= count. No
completion is needed to provide a free slot, and unrelated callers cannot occupy
this pool. Creation does not enqueue a buffer; encoding and commit occur on
submission, so an earlier unused index does not hold a queue position.
The former caller-provided queue and example-specific `3 * count` workaround are
removed. This establishes the bound for the current declared extent, not a
completed reusable-stream lifecycle or a claim about all native launch costs.
Apple's [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
documents reusable command-buffer objects and separate command allocators.
The installed SDK's `MTL4CommandAllocator.h` requires the allocator's recorded
commands to have completed before resetting its storage. This provides a native
reuse mechanism for N1, but `MTL4CommandBuffer` and its encoders are distinct
interfaces from the `MTLCommandBuffer` used by current supplied encoders. No
Metal 4 adapter or unsupported replay of the current command buffers is present.

## Program.map

The JAX authors' [Pallas scalar prefetch](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)
and [jax.lax.gather](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html),
and Gale et al., [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
(2022), supply references for block indices as data, indexed reads, and block-sparse routed experts.
Indices are ordinary operands of `Mesh.map`; the supplied function performs the indexed
read and any expert selection or neighbourhood sum inside the caller.
Mesh implements no indexed gather/scatter or application routing.

## Program.copy

Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
MLX authors' [JACCL transport](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h),
and rdma-core's [ibv_post_recv](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3):
registered SEND/RECV and work completion. The [hardware notes](collective-dependency-ledger.md)
distinguish substrate facts from the retained bridge's protocol decisions.
TN3205 explicitly limits this transport to `IBV_WR_SEND`; its SDK enum for
`IBV_WR_SEND_WITH_IMM` does not establish hardware support. Both TN3205 and JACCL
show local `wr_id` values returned with completions. Mesh uses these identifiers
for buffer lifetime. Every chunk carries a source-chunk row tag; setup maps that
row to a destination chunk and, for the final chunk, the numerical publication.
The tag arrives in the payload's own work request. There is no index QP or
cross-QP identity join. On September 15, `ibv_devinfo -v` reported `max_sge: 1`
on the local Thunderbolt devices. The alias representation uses one SGE and
does not infer two-entry support from the general verbs API. The
[record layout and execution path](async-collectives.md#execution-and-ownership)
describe this mesh-specific representation. Receive storage is preallocated
across the finite extent; refill does not depend on consumer completion or page
reclamation.
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
Each TX/RX progress step polls one completion and attempts one available request,
then continues immediately. A refused request does not advance the cursor.
`ENOMEM` and `EAGAIN` are deferred to the next step; other post errors conclude
the link's invocation through [its Result](#meshresult). RX attempts its post before publishing the
completion. A zero-completion poll still permits posting. Initial receive setup
posts until the declared list ends or the native queue refuses. Actual QP capacity is
used only at setup to check that an individual framed request fits. This does
not claim the remaining receive mapping or publication queues satisfy W2/W3.

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
