# Asynchronous partial tensor calls

The [user's requirements](collective-goals.md) define scope. This document
explains the source; it does not add requirements.

The previous claim of full completion is withdrawn. The source demonstrates
finite producer/collective/consumer chains, but a single-use value extent and
example-only integration do not establish the reusable deployment
interface. The disposition table below records implemented mechanisms;
it is not proof that those narrower mechanisms satisfy the deployment target.

`swift/Mesh.swift` accepts tensor functions. It contains no implementations of
matmul, activation, attention, normalization, sum, maximum or minimum. The
numerical operation is a `TensorFunction` supplied to `call`, `map` or `reduce`.
`TensorPart` names a contiguous section and its caller-selected rank. A tensor
is a list of these sections. Mesh allocates their actual shared, registered
backing and binds their uses before `start()`. `count` is the AOT extent of the
value index; `submit(index)` starts that value through the realized functions.

## Algebra and callers

For a linear function, `X = sum(P[i])` gives `T(X) = sum(T(P[i]))`. No inverse
is needed when the input partials are available. Splitting an already computed
`T(X)` using `T Q[i] inverse(T)` is a valid construction for invertible T, but
is not a transport operation and is not implemented in mesh. A coordinate
slice of `T(X)` is not generally `T` applied to a coordinate slice of `X`.

The scalar `linear-chain` program and the earlier fixed matrix executable have
been deleted. Neither establishes the requested large composed use.

[`examples/coreml-chain.swift`](../examples/coreml-chain.swift) now accepts a
caller configuration of tensor shapes, owners and supplied compiled functions.
At stage d it declares the block-function relation

`C[d,i,j] = F[d,i,j](X[d,i])`,
`Y[d,j] = sum_i C[d,i,j]`,
`X[d+1] = G(selected earlier partials, selected Y[d])` when finishing calls are supplied, otherwise Y.

F and G are existing Core ML functions supplied by the caller. The sum invokes
Accelerate's existing `vDSP_vadd`; initial producers invoke `vDSP_vramp`. Mesh
implements none of that arithmetic. The caller's `functions` array gives the
input-to-output block relation, and `reduceScatter` places each sum at its declared
owner. Finishing calls name their inputs, outputs, owner and worker. They can
read an earlier partial for a skip connection, take multiple reduced inputs,
or return several sections. Their input gather and output scatter are explicit
caller operations. Shapes can differ between input sections, output sections and stages.
The stage list can contain the requested 8–100 blocks; there is no depth limit or
separate block/layer API. The [caller configuration](function-chain.md) describes
this actual source path and its inputs.

For two input sections and output owners 0 and 1, node 0 computes F00 and F01;
node 1 computes F10 and F11. F01 goes to node 1 and F10 to node 0. Node 0 combines
F00 with F10; node 1 combines F01 with F11. Each Yj feeds its own finishing
function and next-stage uses. Completion of Y0 does not await Y1. Both nodes
can compute at every depth, and adjacent depths can overlap wherever their
actual operand relationships allow it. There is no all-gather of the inputs or
layer-wide completion barrier in this calling context.

Configuration parsing, model loading, native views, routes and calls are realized
before submission. Repeated model paths reuse the loaded model locally. Each
index in the caller's configured count traverses the declared chain using distinct
backing; the model list and graph are not interpreted during numerical execution.
The caller discards its construction history before starting; declared input
references retain earlier values through their later readers.
This establishes a source composition, not measured throughput or a completed
unbounded-stream lifecycle.

## Execution and ownership

A call declares its input and output sections and a numerical worker. Its
function object stores the supplied function and section descriptors once. Each
value index has an operand array, pending-operand count and completion record
allocated during realization. For descriptor `(first, stride)`, its logical row
is `first + index * stride`. A shared constant has stride zero. Setup retains
each transient indexed input use and each shared input binding. Shared inputs
remain retained until the prepared function is destroyed, rather than being
retained and released separately for every value index. Before activating transport, setup stores consumer
references in contiguous row ranges, indexed by an offset array. Constants already
published at setup have no pending arrival edge. A runtime publication visits
only that section's range, without following linked use records. A consumer with multiple operands becomes
callable when those specific operands exist; unrelated sections and collective
participants are not a barrier. The runtime does not scan all functions.

`start()` realizes receive storage, completes bindings and consumer ranges, starts
the numerical workers, then activates transport. An early receive therefore cannot
race construction of its consumers. Each
`submit(index)` publishes the index's root row to its configured root workers
using their existing notice queues. The caller supplies each index in
`0..<count` once; there is no occupancy check, replay guard, configuration change
or operand allocation in submission. A call depending only on constants also
consumes that root publication. A shared received constant contributes its actual
arrival dependency; it does not bypass submission. Other consumers have only
their declared operand dependencies, with no per-index global completion barrier.

There are two independent streaming levels. Numerical streaming publishes the
partials declared by supplied tensor functions. Transport streaming carries each
send through internally sized requests. Transport fragmentation does not change
the tensor partition, numerical function, shape, or send/receive API.

Setup prepares local input and output addresses once. At invocation, the numerical
worker resolves only received inputs through the canonical page table and invokes
the supplied function. Input references keep local backing fixed through completion;
received addresses reflect the actual placement of incoming bytes. CPU completion is its return; Metal and Core ML
use their native asynchronous completion. Successful completion publishes the
outputs and releases transient input references. Completion traverses a contiguous
list of transient input descriptors prepared at setup; it does not test each
input's storage kind or touch shared-constant reference counts. Failures are recorded and do not publish
failed output as valid data.

The lifetime references for native call records are also known at setup. Before
starting a numerical worker, `mesh_calls_start` acquires its worker reference and
all of its declared call references together. Launch performs no reference-count
increment. Each native completion releases its existing call reference. On exit,
that worker cancels the references for its unissued records and releases its own
reference in one update. It identifies those records from the pending counts it
alone owns; this traversal occurs after numerical progress stops, not during
dispatch. A failed thread creation returns that worker's entire reserved count.
The ordinary Mesh owner reference keeps startup alive throughout these steps.

For N operand bytes and internal payload capacity C, setup represents
K = ceil(N / C) transport chunks. A value has K page-table entries and one
numerical presence bit and ownership record. A 10,648-element float32 operand
has 42,592 bytes: with C = 16,384, it uses three transport chunks but remains
one operand with the same 10,648 elements. Changing C changes none of the
caller's numerical declarations.

`mesh_section_create` allocates local operands according to N. Received operands
reserve logical rows; `mesh_transfers_prepare` allocates their actual backing as
one contiguous page run per receive queue before native views are constructed.
There is no temporary operand allocation replaced during setup. All simultaneously
live receives have writable backing without consumer credits or reclamation waits.

The bridge prepares two virtual representations of the same shared-memory
payload pages. Numerical functions see dense operands. Registered transport
spans alias a tag page followed by each C-byte chunk. A request starts at
the tag page's final four bytes and continues into the payload. This uses one SGE, as exposed
by the local device. It adds no payload copy, runtime mmap, or second identity
channel. Registration spans are whole alias slots, placed in separate virtual
address banks; no request crosses a memory-registration boundary. Numerical
operands may cross those boundaries transparently.

TX receives a numerical publication, indexes its contiguous range of configured
send edges, and starts posting chunks immediately. Each queue appends edge indices
to its preallocated FIFO range. A send edge occurs once in that range, so its
capacity is exactly the declared send count; there is no runtime queue allocation
or linked-list append. One cursor per send advances through its
realized page indices; there is no runtime construction or traversal of the
whole chunk list before the first post. Chunks of one send remain consecutive
on their queue. Sends can publish in any order and have different byte lengths.
Each transport step polls one completion and attempts one available request,
including when the poll yields no completion. The next step proceeds immediately;
there is no software outstanding-request count or request-capacity gate. Initial
receive setup fills the native queue before traffic starts. RX runs on its own
hardware thread and attempts its replacement receive before publishing the
completion. Numerical workers and the collector have separate threads.

Receive posting computes `firstPage + chunkIndex * blockPages` over the contiguous
run allocated at setup; the former table of every receive address is removed.
Every RX completion identifies its physical chunk through `wr_id`. Its tag indexes
a cursor in the contiguous destination table. Repeated sends of the same source
have one target per declared copy, in the same order for every chunk; the cursor
advances through those targets without following a linked record. Assignment exchanges forward and inverse
page-table entries, preserving ownership of the displaced unfilled backing.
The page-index list therefore fills incrementally. The final chunk's target
also names the numerical head to publish; earlier chunks do not publish a
replacement numerical partial. The queue's FIFO order establishes that this
partial's earlier chunks are already placed. No thread waits for another
partial or a whole operation to finish. A failed receive suppresses subsequent
numerical publication on that queue and reports the provider error.

Because each send's chunks consume consecutive positions in a preallocated
receive run, the complete operand is also contiguous in the numerical address
space. Native bindings cover possible start positions at which that operand
fits. Publication never remaps a native view. The page-table permutation and
lifetime relationship are derived in [pages and functions](pages-and-functions.md#block-addressing).

One transport reference retains the source through all its chunks. Only its
final ordered send completion releases that reference; there is no fragment
completion counter. Numerical completion, publication and ordinary object
lifetime discharge the other known references. The collector returns each
chunk's actual backing without clearing it. The client notice banks keep old
device publications distinct during handoff.

Setup chooses L = min(C, max operand bytes on that queue direction). The wire
request carries L + 4 bytes and occupies ceil((L + 4) / 4096) native frames.
Queue capacity counts those frames. This preserves short requests when every
operand is smaller than C; a large operand never enlarges a request beyond C.
Tail chunks and smaller operands on a mixed queue include padding up to L. Each physical
chunk additionally has one OS page for its tag in the registered alias layout.
Those framing, padding and metadata costs are real; zero-copy does not mean
zero cost. The obsolete rule that every operand fit one message, the public
capacity query, and unbounded largest-operand message sizing are deleted.

The mechanisms use [Apple TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
[JACCL's SEND/RECV interfaces](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h),
and [shared page mappings](algorithm-sources.md#programtensor). The alias layout
and chunk-to-publication relation are Mesh's implementation, not upstream code.

## Configured peers

One bridge owns one canonical page arena and a configured list of links. There is
no two-rank bound in `Mesh`, no global verbs provider or listener, and no inferred
other rank. Each link has its own connection, send queue notices, work-request
state, receive assignment metadata, TX thread and RX thread. Links using the same
named device share its context, protection domain and registered memory. The wire
alias map is common to all devices; each device contributes its own registration
keys. Device initialization is serialized only at setup, before that device's
links start; no numerical publication or work-request completion takes that lock.

During realization, `send(part, to: rank)` resolves the configured peer and queue
to a channel. Its two endpoints bind the transfer; other participants retain only
the placement descriptor. The receive allocator and native views use that exact
channel. Publication scatters notices to the predeclared send links and numerical
workers. No peer lookup, collective inference or whole-mesh completion check runs
between publication and posting. Each link posts receives and completes its own
connection exchange before enabling its sends; a different link can already be
progressing. Setup does not join every peer before allowing one link to run.

The collective declarations still determine the communication relation. For
example, `scatter([a, b], to: [2, 0])` sends a to rank 2 and b to rank 0;
`gather([c, d], to: 1)` places those two sections at rank 1 in that order. Neither
becomes a broadcast or all-gather because a third rank exists. Broadcast and
all-collectives remain explicit calls. The source's reduction tree combines only
the contributions named for each output; it does not wait for unrelated outputs.

Transfer metadata, like notices, uses separate banks for the active and retiring
client. A new client's setup cannot overwrite the transfer descriptors still read
by an old link controller. The existing device-ownership record is cleared only
after every old connection has destroyed its QPs. This protects teardown storage;
it is not a completion barrier between tensor functions.

`bin/mesh-bridge.sh` reads `links` from the bridge configuration. Each entry is
`device,peer-rank,local-control-address,remote-control-address[,service]`; the
optional numeric TCP port defaults to 18519. Numeric IPv4/IPv6 addresses and
device names are supplied by the operator's mesh configuration. Distinct links sharing a local control address
use distinct services. For example, a rank's configuration can contain:

```sh
node=0
links=(
  'rdma_en4,1,fe80::a%en4,fe80::b%en4'
  'rdma_en5,2,fe80::c%en5,fe80::d%en5'
)
```

These addresses illustrate the fields, not observed hardware. Keep the configured
arena, block and queue geometry alongside this list. Each peer supplies the
matching endpoint description with its own local interface names. The lower
configured node number dials that link's control endpoint. No node number is
reserved as the single server. The transport uses the explicitly configured RDMA
links; it does not invent routes or placement. `rdma/peers.py` now emits the
supported link arguments without the obsolete detour and hop-limit flags.
The old `peer=` setting and machine-specific one-peer example profiles are gone.
Status reports each link's phase and error separately.

Pairing uses one 30-second monotonic deadline across socket connection, endpoint
metadata, queue descriptors and the initial receive-posting exchange. All socket
I/O is nonblocking; numeric endpoints require no DNS or service lookup. A timeout
reports `ETIMEDOUT` and returns through the link's existing cleanup and setup path
at the configured capacity. The deadline is never consulted by TX/RX progress or
numerical execution. It bounds socket waiting, not a native driver call stalled
inside the kernel; see [the socket mechanism](algorithm-sources.md#programcopy).

## Native contiguous operands

`TensorFunction` owns its preparation function. Its native constructor takes
input and output view factories and a function over the resulting `MeshBindings`.
Each view factory receives a `MeshSpan` during realization. `MeshBindings` holds
the prepared values and their operand-to-index function. CPU, Metal and prediction
constructors supply the actual numerical submission. Multiple varying inputs,
constants and multiple outputs use the same preparation contract.

`call`, `map`, `reduce`, `reduceScatter` and `allReduce` accept this same function
value. All calls finish preparation in `start()`, after receive storage is assigned
and before numerical workers start. The former native-only `Mesh.map` overload
is deleted. A reduction can prepare its native views using its actual operands,
without exposing intermediate pages or building bindings during invocation.
The internal `MeshSubmission` enum selects the backend during realization;
`MeshInvocation.submit` remains the resolved runtime closure. Function factories
are not traversed during numerical execution.

A local operand selects a binding by value index; a shared constant selects
index zero. Each receive channel has a contiguous preallocated page range.
Its operand selects a prepared binding by `(page - firstPage) / blockPages`.
The range is recorded once when allocated; neither a per-operand slot table nor
a rescan of transfer descriptors is needed. The numerical caller can select a prepared value by operand, or pass its
index to an existing indexed numerical function. This keeps the selection in
ordinary operand indexing. The former table of input/output pairs and its
additional `MeshInvocation` dispatch layer have been deleted.

The existing `MatrixOperations.multiply` accepts independent input and output
view arrays. One specialized operation accepts their indices at encoding.
MPS matrices are prepared independently; its multiplication object is reused by
that call's numerical worker. Metal/tensor paths select their existing pipeline
at setup and pass the indexed buffers and offsets to the existing encoder. The
ordinary fixed-operand `multiply` is the index-zero partial application of that
same implementation. Mesh does not provide a competing numerical implementation.
`TensorFunction.prediction` takes a model and named input/output view factories.
It prepares a feature provider and output options for each value index. The
provider points to that call's realized operand array and selects each named
input's prepared feature value independently. Output options bind the local
results at setup. Submission assigns the operand-array pointer and calls native
asynchronous prediction directly. The previous provider-building callback is
removed; multiple inputs do not require a product of possible addresses.
No virtual address is remapped while a native call uses it. No matrix binding,
MLMultiArray, feature provider or operand allocation is constructed during the
numerical invocation.

`MeshSpan.metal` presents each registered operand window as a no-copy Metal
buffer. It does not wrap the whole arena in one Metal allocation. `MeshSpan.multiArray` presents it through the pointer-backed Core ML
API. The Core ML example supplies its actual output section through
`MLPredictionOptions.outputBackings`. Core ML's model feature and backing
contracts still apply; unknown output names are ignored by Core ML, so the
example requires the supplied model's actual feature names. Apple's SDK also
states that models which do not support supplied output buffers may return their
own storage. A valid supplied prediction function must write its declared Mesh
outputs, just as a valid CPU or Metal function must. The current prediction path
does not adopt foreign output storage or copy it into the registered operands;
successful native completion alone does not prove that an arbitrary model honored
those bindings. Model backing support remains part of selecting that function.
These mechanisms
come from the [Apple operand and completion APIs](algorithm-sources.md#programkernel_call).
They are operand representations, not numerical backends implemented by mesh.

## Collective relations

`send(part,to:)` declares both endpoints and returns the receive operand.
At setup, repeated requests for the same immutable part, destination and queue
reuse that receive operand. A private part identity distinguishes values without
reading payload or inferring a collective. The delivery map is discarded after
binding; it is absent from publication and transport execution. Different
destinations or queues remain distinct, and numerical input uses are still counted
separately: combining a part with itself still combines two contributions.
Broadcast replicates sections. Scatter assigns sections to destinations.
Gather returns the source sections at its destination. All-gather does that for
each participant. All-scatter distributes each source's sections to the supplied
destinations; all-to-all uses the participant list as that destination list.
None of these movements performs arithmetic.

Reduce builds a binary combination tree from the supplied function. The final
owner is explicit. Reduce-scatter does this independently for each section and
its supplied owner. All-reduce replicates those independently reduced sections.
It does not gather every input before starting combinations. Sum, max and min
are meanings of supplied combining functions, not transport modes. A combining
function used with the tree must satisfy the intended reduction algebra; floating
point addition has its usual reassociation differences. The communication
relations follow [MPI and MLX](algorithm-sources.md#collective-movement).

## Explicit synchronization counterexample

[`examples/sync-on-remote-fill.swift`](../examples/sync-on-remote-fill.swift) has
three explicit modes, all with the same four producer/consumer chains:

- `parallel`: each producer runs independently.
- `serial`: producer i explicitly calls `syncOnRemoteFill` for the returned
  result of chain i-1 before producing its own input. Four independent chains
  become one ordered chain.
- `deadlock`: each producer explicitly waits for the returned result that
  requires its own unpublished output. The dependency is a cycle, and the wait
  has no timeout. No queue-draining strategy can create the missing value.

For one chain of producer cost P, forward and return communication cost L each,
and consumer cost C, the independent-chain critical path is P+2L+C. The inserted
serial dependencies make it 4(P+2L+C). This compares the causal paths with enough
workers, not measured wall times. The default API never invokes this operation.
`dispatchMain` in the standalone examples only keeps process-owned storage and
workers alive; it does not gate tensor issuance.

## Current extent and remaining work

The one-peer restriction has been removed. World size and operand owners come
from the caller. The bridge realizes a list of explicitly configured peer links,
up to 8 numerical workers, and configured transport queues per link. Numerical
sections can span arbitrarily many transport chunks within the
configured arena. The transport count is internal; larger tensors do not require
caller-side repartitioning.
The programs are finite AOT data flows. One `start()` realizes the configuration;
`submit(index)` uses it for successive values, without making separate chain
declarations. The configured count reserves all value backing ahead of execution.
Shared function and route metadata are reused, while value storage remains
distinct. Freed backing returns to the pool through the existing collector;
this change does not feed it back into an unbounded receive cycle or reuse an
already submitted index. The deleted matrix executable no longer constitutes
integration evidence.
Static graph structure does not imply a single-use execution budget. Neither this
finite extent nor the absence of a serving caller is an authorized scope decision.
The source has not integrated returned pages into continued use of the configured
stream, or integrated collective invocation into the serving path.

Transport chunk padding, source-tag handling, publication and reference-count
atomics remain actual costs. The [P1 Core ML chain](function-chain.md#p1-paired-core-ml-run)
exercised the alias mechanism with four/eight-chunk sections on the RDMA pair.
For N indices and R possible receive positions, native input/output preparation
constructs R input bindings and N output bindings. Received-view selection keeps
the channel's first page and block size, with no slot map over arena blocks.
It creates one numerical submission function per declared
call. The previous N*R product represented paired addresses, which was unnecessary
for APIs that take independently bound input and output operands. Local inputs
also need only N bindings; shared local constants need one. These are source
allocation counts, not performance measurements.
Shared function metadata is constant in N; per-value operands, consumer references,
pending counts and backing grow with the finite extent. For W numerical workers,
L logical rows and E declared arrival dependencies, adjacency contains W*(L+1)
offsets and E call pointers. This replaces list heads and two-pointer use
records, including records previously allocated for inputs already present at setup.
Repeated uses of one operand retain repeated references and dependencies;
compaction does not deduplicate the dataflow itself. Submission touches the root
workers only; publication visits the published row's uses, not all functions.
These are explicit remaining costs; no claim of zero total overhead, JACCL cost parity, or speedup
over world size 1 follows from this source change.

There is no runtime testing gate here. The bridge, C and Swift libraries, configured Core ML chain and synchronization
counterexample are build targets.
P1 deployed ABI 43 with explicit links and ran the Core ML chain on both nodes,
including a fresh Mini clone and an external caller target. Its returned outputs
are recorded in the [chain documentation](function-chain.md#p1-paired-core-ml-run).
Public serving-path measurements remain separate deliverables.

Client attachment and bridge startup no longer run a process-memory ranking scan.
The unrelated `mesh-memory.h`, its `--memory-check` command and launch-script hook
were deleted. Configured arena geometry and the caller's explicit memory cap
still determine storage realization.

Build the native libraries and examples with
`make -C rdma all coreml-chain sync-on-remote-fill`.
The Core ML caller takes `rank world-size region configuration.json`.
The counterexample takes `rank region parallel|serial|deadlock`.

## Source inventory

These counts include the implementation just added, not only the deletion.
The mesh baseline is `1ed126d`, before deletion commit `5762898`.

| Counted set | Before | Current |
|---|---:|---:|
| All mesh repository source files with the extensions below | 665,701 lines / 1,056 files | 651,848 lines / 992 files |
| Replaced paths, including new Swift code and old root setup.py | 16,378 lines / 77 files | 2,524 lines / 13 files |
| Build metadata in those paths, including pyproject.toml | 47 lines | 25 lines |
| Engine's deleted mesh_matrix.swift, tools/mesh/sync.sh and replacement matrix example | 203 lines | 0 lines |

The replacement-path set is `rdma/`, `python/`, `swift/`, `examples/`,
`xonotic/solver/`, `xonotic/planner/`, and the old root `setup.py`. Source extensions
are `.c .h .m .mm .swift .py .metal .sh .zsh .js .ts .jsx .tsx .qc`. The whole-repository
row includes the large unchanged Xonotic sources. It has not been halved.
Makefiles, module maps and pyproject.toml are reported separately above. Shared
engine dependencies `parameter_configuration.swift`, `matrix_shaders.swift`, and
`matrix_operations.swift` now total 532 maintained lines in the working tree; the
deleted matrix caller used them. That includes the existing addition encoder and shader
moved from `bootstrap.swift` and `kernels.swift`, where their 26 lines were
removed. This change does not attribute pre-existing working-tree edits to the
mesh refactor.

Markdown is excluded from these source counts. Earlier documentary deletion and
scope correction were committed separately as mesh `a07d7f6` and engine `14d1057`.
This implementation explanation is documentation, not structural source reduction.
The block-addressing refactor reduces maintained source by 51 lines; source-comment
migration removes two further lines. Removing the process-memory scan and its
callers removes another 65 source lines, including its launch-script hook. There
is no replacement source generator.

## Source disposition against the objective

The objective is the five requirements in the user-selected attachment named in
[collective-goals.md](collective-goals.md). The following evidence describes the
current source and example callers. It does not close the deployment gaps above.

| Requirement | Source evidence |
|---|---|
| Higher-order partial tensor functions using existing numerics | `TensorFunction` owns raw or native operand preparation. Calls, maps and reductions accept the same function value. Backend selection and view construction finish before invocation. The configured caller supplies existing Core ML functions. |
| Distinct collective semantics | `Mesh.swift` defines send/receive endpoints, broadcast, scatter, gather, all-scatter, all-gather, all-to-all, reduce, reduce-scatter and all-reduce. Movement returns indexed sections; only the supplied combining function performs reduction arithmetic. |
| AOT bindings, zero-copy asynchronous use | `mesh_call_bind` prepares operands and retains inputs. `mesh_calls_start` realizes contiguous consumer ranges before transfer activation. Swift prepares native views once; only received input addresses resolve at invocation. `link_configure` realizes routes and posts receives before `verbs_up` enables sends. `mesh_receive_assign` places each chunk through page-index assignment over registered aliases without copying. Dedicated TX/RX threads post and drain; numerical completion publishes only the corresponding value's uses. |
| Delete incompatible implementation and callers | The former executor/frontend and engine adapters are absent from the current tree. The source inventory includes their replacements. The index transport channel, paired native-binding Cartesian product, per-page receive metadata and process-memory ranking scan are also absent. |
| Actual producer/collective/numerical-consumer integration | `coreml-chain.swift` composes caller-supplied block functions, reduce-scatter and supplied consumers across a configurable stage list. P1 ran four FFN residual blocks and four indices on the RDMA pair, reaching all eight final consumers. Accelerate performs the supplied float32 sum. This establishes operation, without a throughput claim. |
| Automatic lifetime; explicit synchronization only | `mesh_buffer_retain` accounts for declared uses. `mesh_publish`, native numerical completion, TX completion and ordinary object destruction discharge their references; `mesh_collect` returns backing without clearing payload. Runtime presence polling occurs only in the explicitly called `mesh_sync_on_remote_fill`; its source counterexample includes a self-dependent permanent wait. |

Configuration loops, native post-result handling, indexed
operand dependencies, reference updates and native launch operations remain.
There is no whole-function readiness scan, caller release protocol, remote
consumer acknowledgement or default remote-fill wait. The finite storage extent,
and native API contracts above remain explicit limits. Builds
establish integration consistency; no runtime measurements or speedup claims are
used as evidence for these source properties.

The generic Metal invocation selects command buffers prepared during realization,
with pool capacity owned by Mesh for every declared call. Native view
preparation and raw callbacks both support multiple input and output operands.
Each configured link has independent TX and RX workers. Each runtime post attempt
is preceded by a CQ poll; draining never waits for an entire section or publication
list to be posted.
The remaining transport and lifetime restrictions need disposition against the
user's requirements; accepting a convenient example does not settle them.
The user's four-residual-FFN-layer case further requires useful numerical work on
both participants at each depth; added global guards, waits or synchronization
that make this chain slower than local execution invalidate the implementation.
The deleted matrix example did not establish this case. The user's 8–100-block
workloads must use the same tensor-function-invariant interface; they do not
justify an architecture-specific library path.

The transport-size replacement changes maintained Mesh source by +7 lines net,
including the operand allocation and native-view changes; removing the unused
engine addition wrapper removes 8 more source lines. Eight existing prose comments were replaced with documentation citations;
those equal-line replacements are not counted as structural reduction. Documentation changes are
reported separately from that structural count. The bridge, native libraries
and retained callers now compile at ABI 43. Compilation does not establish RDMA
execution or end-to-end integration.

The shared function-preparation path and configured Core ML caller add 28 source
lines net relative to `98742c8`, including deletion of the fixed scalar caller.
This is implementation growth, not documentation migration or claimed reduction.

The multi-peer replacement adds 59 maintained source lines net relative to
`bd0b779`: 57 in the replacement paths and 2 in the launch script. That net
count includes replacing a two-line source comment with one documentation
citation; the implementation change excluding that migration is +60 lines. Deleting the
15 lines of obsolete machine-specific configuration is reported separately;
configuration files and these documentation edits are outside the source count.

The contiguous consumer ranges, prepared local operands and arithmetic receive-view
selection change maintained source by +3 lines net relative to `c9d9e18`.
This removes runtime pointer chasing and repeated local-address resolution;
the source count is not a claim of measured latency reduction. Documentation is
reported separately, and no numerical function or collective verb was added.

The transport tables remove 5 maintained source lines relative to `7397d58`.
For E sends, edge records and FIFO indices occupy 20E bytes instead of 24E bytes
of linked records. For F received chunks, target records occupy 8F bytes instead
of 12F bytes, and the separate 4F-byte receive-address table is removed. Offset
arrays add one 32-bit sentinel each. These are representation sizes, not timing
measurements. The changes preserve distinct copies of a source and the declared
peer/queue mapping; they neither infer a collective verb nor change the finite
invocation extent. Documentation changes are separate from the source reduction.

The scoped-reference change adds 10 maintained source lines relative to `e798fe7`.
For N value indices, S shared input bindings and V transient input bindings, input
retains change from N*(S+V) to S+N*V. Numerical completion releases only the N*V
transient references; function destruction releases S shared references. The
per-call global retain is replaced by one acquisition per worker at startup.
One additional descriptor slot per input is reserved within the existing function
metadata allocation for the contiguous transient-input list. There is no extra
per-value allocation or runtime input-kind branch. These are source operation and
storage counts, not measured latency gains; the finite invocation extent remains.

Named Core ML operand preparation and the caller's explicit earlier-value inputs
add 52 maintained source lines relative to `8a42213`. They replace the raw
prediction-provider callback and unary finishing-call format. The construction
history and named-call configuration belong only to the example, outside Mesh.
The removed competing deliverables checklist is a separate documentation deletion,
committed as `0a8eb26`; it is not part of the source reduction.
The delivery map and consolidated part construction add another 14 source lines,
bringing this implementation change to +66. The map removes repeated sends and
receive allocations for the same value/destination/queue; it does not merge
distinct numerical uses or add a runtime lookup.
