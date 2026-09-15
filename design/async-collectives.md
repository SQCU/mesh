# Asynchronous partial tensor calls

The [user's requirements](collective-goals.md) define scope. This document
explains the source; it does not add requirements.

The previous claim of full completion is withdrawn. The source demonstrates
finite producer/collective/consumer chains, but a single-use value extent, one-peer
transport and an example-local command-pool bound do not establish the reusable
deployment interface. The disposition table below records implemented mechanisms;
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

[`examples/linear-chain.swift`](../examples/linear-chain.swift) supplies the
literal `T(x,y,z)=(x+y,y+z,z+x)` from the conversation. Its data flow is:

1. Three independent producers write `(x,0,0)`, `(0,y,0)`, `(0,0,z)`.
2. Scatter places these partials on the supplied mesh; each invokes the same T.
3. Each published `T(P[i])` also goes directly to a consumer applying the same T.
   Consumer i does not depend on either of the other producer contributions.
4. A supplied add function reconstructs both `sum(T(P[i]))` and
   `sum(T(T(P[i])))`. Each binary combination consumes just its two operands.
   Linearity makes the second result `T(T(X))`. For `(1,2,3)`, the results are
   `(3,5,4)` and `(8,9,7)`.

The program declares the chain once with `count: 4`, then submits indices
`3, 0, 2, 1` through that configuration. The producer reads its coordinate data
using `outputs[0].index`. The function objects, collective routes and operand
bindings are not rebuilt between submissions. Each index has distinct value
storage; all four can be in flight without an edge between their computations.

[`metal-microbench/examples/mesh-matrix.swift`](../../../metal-microbench/examples/mesh-matrix.swift)
uses the engine's existing `MatrixOperations.multiply` for both stages and its
existing addition encoder for all-reduce. Each row section j is split across
contraction coordinates K_r. Scatter places `X[j,r]` beside `W[K_r,:]`;
the supplied matrix function produces contributions to
`Y[j] = sum_r X[j,r] W[K_r,:]`. All-reduce supplies Y[j] to each consumer,
which computes distinct output columns `Z[j,r] = Y[j] W[:,K_r]`.
Gather returns their indexed sections, reconstructing `(X[j] W) W` without
assembling a copied dense tensor. The consumer uses the complete reduced input;
there is no redundant second reduction. The algebra and actual operand layouts
are detailed in [the caller documentation](../../../metal-microbench/docs/async_collectives.md).
Row sizes 1, 8, 16 and 4 use separate numerical workers and have no cross-row
dependency. Their payloads fit one wire frame and share transport queue zero.
The caller supplies the backend and weight ownership. Four submitted
indices reuse the same chain and canonical shared weight shards.
The caller sizes each Metal queue for `3 * count` command buffers, covering its
entire finite set of local multiplication and combining calls. Thus native
command-buffer creation cannot exhaust that queue's configured pool. Generic
supplied numerical functions remain responsible for their native API contracts;
mesh adds no completion wait to their launch paths.

[`examples/coreml-chain.swift`](../examples/coreml-chain.swift) takes an existing
compiled model, input/output feature names and width as arguments. It sends three
sections through that supplied model and then through the same model on the
receiving participant, for four submitted indices through the same configuration.
It creates no model, operation implementation or compiler.

## Execution and ownership

A call declares its input and output sections and a numerical worker. Its
function object stores the supplied function and section descriptors once. Each
value index has an operand array, pending-operand count and completion record
allocated during realization. For descriptor `(first, stride)`, its logical row
is `first + index * stride`. A shared constant has stride zero. Setup retains
each indexed input use and installs section-to-use adjacency. Constants already
published at setup have no pending arrival edge. A runtime publication indexes
only the uses of that section. A consumer with multiple operands becomes
callable when those specific operands exist; unrelated sections and collective
participants are not a barrier. The runtime does not scan all functions.

`start()` completes bindings and starts the numerical workers once. Each
`submit(index)` publishes the index's root row to its configured root workers
using their existing notice queues. The caller supplies each index in
`0..<count` once; there is no occupancy check, replay guard, configuration change
or operand allocation in submission. A call depending only on constants also
consumes that root publication. A shared received constant contributes its actual
arrival dependency; it does not bypass submission. Other consumers have only
their declared operand dependencies, with no per-index global completion barrier.

The numerical worker resolves operand addresses through the canonical page
table and invokes the supplied function. Each contiguous section has one
block-head entry; interior addresses follow from the configured relative offsets.
CPU completion is its return. Metal
completion is the command buffer's native completion handler. Core ML uses its
native asynchronous prediction completion. Successful completion publishes each
output and releases the input references. Provider failures are recorded in the
shared port error fields and do not publish failed output as valid data.

Send completion releases the transport's input reference. Receive completion
publishes the new section and ends producer ownership. When its last declared
use and external value handle are gone, the existing collector returns backing
to the page pool without clearing it. Callers do not publish consumer stamps,
free pages, or signal completion. The library retains its context through actual
native completion, including when its Swift owner leaves scope.
The configuring client uses a different notice bank from the client's still-open
device work. Old publications cannot be consumed as new values during handoff;
this is an address distinction, without a handoff wait in numerical execution.

TX, RX and collection have separate threads. Numerical workers never poll an
RDMA completion queue. Setup preposts the receive window while queue pairs are
in RTR and exchanges setup completion before enabling sends. Runtime refill
remains on the dedicated RX thread, before delivering the completed section.
Publication directly indexes configured sends. Its buffer use mask names
numerical workers and transport queues, replacing the separate send-source plane.
One transfer descriptor covers its extent and row stride. Local source tags are
written during realization. Receive forwarding sets its local source tags before
publication; each tag stays immutable through all sends of that value.

A message contains the operand bytes and a four-byte source-row tag in the same
registered block. Setup maps peer source rows to local receive uses. Repeated
sends of one source have identical tags and payloads; a precomputed per-source
use list supplies their distinct destination rows. These lists follow the
sender's configured per-source edge order. Other sources can arrive in any order.
The receiver reads the tag from the completed block and indexes its local use
list. No separate identity arrival, data copy or identity/payload join remains.

Verbs' `wr_id` carries the send's retained logical row or receive's physical page.
The native completion identifies that storage directly, replacing the software
posted-request FIFOs. The dedicated RX worker refills before assigning and
publishing the completed partial. Receive posting traverses all preallocated
values until the configured extent is exhausted. It never requires a numerical
consumer to return a credit before it can post the next planned buffer.

Assignment exchanges two block-head entries and their inverse mappings, using
four stores independent of the configured number of pages per block. Publication
writes one presence bit for the complete section and releases the producer's
initial reference. The unused constant bitmap, interior-page mapping accessors,
per-page alias metadata and duplicate producer-flag operation are removed. The
[address and lifetime derivation](pages-and-functions.md#block-addressing)
explains why published native operands remain fixed through out-of-order arrivals.

For queue direction q, setup chooses
`wire_bytes[q] = 4096 * ceil((max_payload_bytes[q] + 4) / 4096)`.
All requests in that direction use this known length, so receivers prepost the
correct frame count before knowing which source will publish next. A queue
mixing large and small partials pads to its largest configured size. The caller
can assign separate queues to different partial sizes. Storage still reserves a
whole block per live value; `sectionCapacity` is block bytes minus four. Operand
starts remain page-aligned, and only declared tensor bytes belong to numerical
functions. Tags for different outgoing queue lengths all lie beyond those bytes.
Queue lengths use the same client banks as publications, so an old device's
forwarding reads its own realized lengths during client handoff.

These mechanisms use the plain SEND/RECV and local work-request identifiers
specified by [Apple TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
and used by [JACCL](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h).
The record layout and source-to-use relation are mesh's implementation. Compared
with the previous source, the runtime removes the index queue pair, its two CQs,
index-frame storage, index copies, and the three identity/payload join counters.
It posts one SEND and one matching RECV per communicated partial.

## Native contiguous operands

The direct `TensorFunction` form takes already resolved operand spans. The
higher-order `map` overload takes independent `inputView` and `outputView`
functions, plus the function that prepares numerical submission. Each view
function receives a `MeshSpan` during realization. Its `MeshBindings` holds the
prepared values and their operand-to-index function. The supplied submission
function is constructed once from the two binding collections.

A local operand selects a binding by value index; a shared constant selects
index zero. A received operand selects its binding through a physical-page slot
map. The numerical caller can select a prepared value by operand, or pass its
index to an existing indexed numerical function. This keeps the selection in
ordinary operand indexing. The former table of input/output pairs and its
additional `MeshInvocation` dispatch layer have been deleted.

The matrix caller uses `MatrixOperations.multiply` with independent input and
output view arrays. One specialized operation accepts their indices at encoding.
MPS matrices are prepared independently; its multiplication object is reused by
that call's numerical worker. Metal/tensor paths select their existing pipeline
at setup and pass the indexed buffers and offsets to the existing encoder. The
ordinary fixed-operand `multiply` is the index-zero partial application of that
same implementation, so local serving and mesh use one numerical implementation.
Core ML prepares feature providers and output options separately, then passes
the selected pair to native asynchronous prediction.
No virtual address is remapped while a native call uses it. No matrix binding,
MLMultiArray, feature provider or operand allocation is constructed during the
numerical invocation.

`MeshSpan.metal` presents registered storage as a no-copy Metal buffer and
byte offset. `MeshSpan.multiArray` presents it through the pointer-backed Core ML
API. The Core ML example supplies its actual output section through
`MLPredictionOptions.outputBackings`. Core ML's model feature and backing
contracts still apply; unknown output names are ignored by Core ML, so the
example requires the supplied model's actual feature names. These mechanisms
come from the [Apple operand and completion APIs](algorithm-sources.md#programkernel_call).
They are operand representations, not numerical backends implemented by mesh.

## Collective relations

`send(part,to:)` declares both endpoints and returns the receive operand.
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

The native bridge has one peer. This source exposes world sizes 1 and 2, up to
8 numerical workers, and the configured transport queues. Each contiguous
section fits one configured transport block with four bytes reserved for its tag;
larger tensors use several sections.
The programs are finite AOT data flows. One `start()` realizes the configuration;
`submit(index)` uses it for successive values, without making separate chain
declarations. The configured count reserves all value backing ahead of execution.
Shared function and route metadata are reused, while value storage remains
distinct. Freed backing returns to the pool through the existing collector;
this change does not feed it back into an unbounded receive cycle or reuse an
already submitted index. The matrix executable integrates existing engine
numerical functions into a contraction-partitioned producer, all-reduce and
column-partitioned consumer. It is not a language-model serving implementation.
Static graph structure does not imply a single-use execution budget. Neither this
finite extent nor the absence of a serving caller is an authorized scope decision.
The source has not integrated returned pages into continued use of the configured
stream, or integrated collective invocation into the serving path.

Transport sends each queue direction's realized frame length. Frame rounding,
padding between unequal partials sharing a queue, source-tag handling, publication
and reference-count atomics remain actual costs.
For N indices and R possible receive positions, native input/output preparation
now constructs R input bindings and N output bindings, plus one input slot map
over arena blocks. It creates one numerical submission function per declared
call. The previous N*R product represented paired addresses, which was unnecessary
for APIs that take independently bound input and output operands. Local inputs
also need only N bindings; shared local constants need one. These are source
allocation counts, not performance measurements.
Shared function metadata is constant in N; per-value operands, uses, pending
counts and backing grow with the finite extent. Submission touches the root
workers only; publication visits the published row's uses, not all functions.
These are explicit remaining costs; no claim of zero total overhead, JACCL cost parity, or speedup
over world size 1 follows from this source change.

There is no runtime testing gate here. The bridge, C and Swift libraries, literal
chain, Core ML chain, synchronization counterexample, and existing-matrix chain
were built, along with the existing serving library after the matrix refactor.
They have not been run or deployed by this change. Both participants
need the source's ABI 40 bridge before these callers can attach.

Client attachment and bridge startup no longer run a process-memory ranking scan.
The unrelated `mesh-memory.h`, its `--memory-check` command and launch-script hook
were deleted. Configured arena geometry and the caller's explicit memory cap
still determine storage realization.

Build the native libraries and examples with
`make -C rdma all linear-chain coreml-chain sync-on-remote-fill`.
Build the existing-matrix caller with `make mesh-matrix` in metal-microbench.
The ordinary examples take `rank world-size region`; the matrix caller adds its
backend argument. The Core ML caller adds `model.mlmodelc width input-name
output-name`. The counterexample takes `rank region parallel|serial|deadlock`.

## Source inventory

These counts include the implementation just added, not only the deletion.
The mesh baseline is `1ed126d`, before deletion commit `5762898`.

| Counted set | Before | Current |
|---|---:|---:|
| All mesh repository source files with the extensions below | 665,701 lines / 1,056 files | 651,669 lines / 993 files |
| Replaced paths, including new Swift code and old root setup.py | 16,378 lines / 77 files | 2,347 lines / 14 files |
| Build metadata in those paths, including pyproject.toml | 47 lines | 27 lines |
| Engine's deleted mesh_matrix.swift and tools/mesh/sync.sh; replacement matrix example | 203 lines | 96 lines |

The replacement-path set is `rdma/`, `python/`, `swift/`, `examples/`,
`xonotic/solver/`, `xonotic/planner/`, and the old root `setup.py`. Source extensions
are `.c .h .m .mm .swift .py .metal .sh .zsh .js .ts .jsx .tsx .qc`. The whole-repository
row includes the large unchanged Xonotic sources. It has not been halved.
Makefiles, module maps and pyproject.toml are reported separately above. Shared
engine dependencies `parameter_configuration.swift`, `matrix_shaders.swift`, and
`matrix_operations.swift` now total 540 maintained lines in the working tree; the
new caller reuses them. That includes the existing addition encoder and shader
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
| Higher-order partial tensor functions using existing numerics | `Mesh.call` and both `Mesh.map` forms accept supplied functions. `MeshInvocation` selects CPU, Metal or Core ML submission during realization. The matrix caller passes existing multiplication and addition implementations; the Core ML caller passes an existing compiled model. |
| Distinct collective semantics | `Mesh.swift` defines send/receive endpoints, broadcast, scatter, gather, all-scatter, all-gather, all-to-all, reduce, reduce-scatter and all-reduce. Movement returns indexed sections; only the supplied combining function performs reduction arithmetic. |
| AOT bindings, zero-copy asynchronous use | `mesh_call_bind` realizes indexed uses and operand storage. Swift prepares native views before `mesh_calls_start`. `link_configure` realizes routes and posts receives before `verbs_up` enables sends. `mesh_receive_complete` changes block-head mappings over registered operands without copying. Dedicated TX/RX threads post and drain; numerical completion publishes only the corresponding value's uses. |
| Delete incompatible implementation and callers | The former executor/frontend and engine adapters are absent from the current tree. The source inventory includes their replacements. The index transport channel, paired native-binding Cartesian product, per-page receive metadata and process-memory ranking scan are also absent. |
| Actual producer/collective/numerical-consumer integration | `linear-chain.swift` applies the supplied T to produced and transported partials before reconstruction. `mesh-matrix.swift` composes contraction-partitioned multiplication, supplied addition through all-reduce, column-partitioned multiplication and gather. `coreml-chain.swift` composes two native predictions through transferred sections. Each declares once and submits four distinct indices. |
| Automatic lifetime; explicit synchronization only | `mesh_buffer_retain` accounts for declared uses. `mesh_publish`, native numerical completion, TX completion and ordinary object destruction discharge their references; `mesh_collect` returns backing without clearing payload. Runtime presence polling occurs only in the explicitly called `mesh_sync_on_remote_fill`; its source counterexample includes a self-dependent permanent wait. |

Configuration loops, capacity checks while posting native work requests, indexed
operand dependencies, reference updates and native launch operations remain.
There is no whole-function readiness scan, caller release protocol, remote
consumer acknowledgement or default remote-fill wait. The finite storage extent,
one-peer transport and native API contracts above remain explicit limits. Builds
establish integration consistency; no runtime measurements or speedup claims are
used as evidence for these source properties.

In particular, the generic Metal invocation still creates command buffers at
submission; only the matrix example supplies the derived pool capacity. Native
view preparation supports one varying input and one output, while the raw callback
form supports multiple operands. The transport supports one peer, and its TX
worker walks the whole detached publication list before returning to CQ polling.
Those concrete restrictions and scheduling costs need disposition against the
user's requirements; accepting a convenient example does not settle them.
