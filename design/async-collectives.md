# Asynchronous partial tensor calls

The [user's requirements](collective-goals.md) define scope. This document
explains the source; it does not add requirements.

The previous claim of full completion is withdrawn. The source demonstrates
finite producer/collective/consumer chains, but a single-use value extent, one-peer
transport and example-only integration do not establish the reusable deployment
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

The small matrix executable, generated weights and fixed shapes were deleted.
They are not evidence for the deployment target. Mesh accepts functions as
arguments; it does not implement or recognize a matrix algorithm, model, block,
or layer. Caller composition supplies the repeated numerical dataflow.

For Metal functions, Mesh realizes a private command queue and all command
buffers during setup. Submission selects a prepared command buffer by invocation
index, encodes and commits it. Direct calls and native-view factories use this
same path.

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

There are two independent streaming levels. Numerical streaming publishes the
partials declared by supplied tensor functions. Transport streaming carries each
send through internally sized requests. Transport fragmentation does not change
the tensor partition, numerical function, shape, or send/receive API.

The numerical worker resolves operands through the canonical page table and
invokes the supplied function. CPU completion is its return; Metal and Core ML
use their native asynchronous completion. Successful completion publishes the
outputs and releases input references. Failures are recorded and do not publish
failed output as valid data.

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

TX receives a numerical publication, directly indexes its configured send edges,
and starts posting chunks immediately. One cursor per send advances through its
realized page indices; there is no runtime construction or traversal of the
whole chunk list before the first post. Chunks of one send remain consecutive
on their queue. Sends can publish in any order and have different byte lengths.
The sender polls completions after each edge's available posts. RX runs on its
own hardware thread, refilling before delivering each completion. Numerical
workers and the collector have separate threads.

Every RX completion identifies its physical chunk through `wr_id`. Its tag indexes
the precomputed destination chunk. Assignment exchanges forward and inverse
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

## Native contiguous operands

The direct `TensorFunction` form takes already resolved operand spans. The
higher-order `map` overload takes input and output partial arrays, a view factory
for each operand, and a function that prepares numerical submission. Each view
factory receives a `MeshSpan` during realization. Its `MeshBindings` holds the
prepared values and their operand-to-index function. The supplied submission
function is constructed once from the two arrays of binding collections.
The native form supports multiple varying inputs, shared constants, and multiple
outputs. Constants are ordinary inputs; no separate constant-operand API is needed.

A local operand selects a binding by value index; a shared constant selects
index zero. A received operand selects its binding through a physical-page slot
map. The numerical caller can select a prepared value by operand, or pass its
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
Core ML prepares feature providers and output options separately, then passes
the selected pair to native asynchronous prediction.
No virtual address is remapped while a native call uses it. No matrix binding,
MLMultiArray, feature provider or operand allocation is constructed during the
numerical invocation.

`MeshSpan.metal` presents each registered operand window as a no-copy Metal
buffer. It does not wrap the whole arena in one Metal allocation. `MeshSpan.multiArray` presents it through the pointer-backed Core ML
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
8 numerical workers, and the configured transport queues. Numerical sections can span arbitrarily many transport chunks within the
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
atomics remain actual costs. The alias mechanism has not been deployed or exercised
on the RDMA link by this change.
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
chain, Core ML chain and synchronization counterexample are build targets.
They have not been run or deployed by this change. Both participants
need the source's ABI 41 bridge before these callers can attach.

Client attachment and bridge startup no longer run a process-memory ranking scan.
The unrelated `mesh-memory.h`, its `--memory-check` command and launch-script hook
were deleted. Configured arena geometry and the caller's explicit memory cap
still determine storage realization.

Build the native libraries and examples with
`make -C rdma all linear-chain coreml-chain sync-on-remote-fill`.
The ordinary examples take `rank world-size region`. The Core ML caller adds
`model.mlmodelc width input-name
output-name`. The counterexample takes `rank region parallel|serial|deadlock`.

## Source inventory

These counts include the implementation just added, not only the deletion.
The mesh baseline is `1ed126d`, before deletion commit `5762898`.

| Counted set | Before | Current |
|---|---:|---:|
| All mesh repository source files with the extensions below | 665,701 lines / 1,056 files | 651,687 lines / 993 files |
| Replaced paths, including new Swift code and old root setup.py | 16,378 lines / 77 files | 2,365 lines / 14 files |
| Build metadata in those paths, including pyproject.toml | 47 lines | 27 lines |
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
| Higher-order partial tensor functions using existing numerics | `Mesh.call` and both `Mesh.map` forms accept supplied functions. `MeshInvocation` selects CPU, Metal or Core ML submission during realization. The native form prepares multiple input/output bindings independently. The Core ML caller passes an existing compiled model. |
| Distinct collective semantics | `Mesh.swift` defines send/receive endpoints, broadcast, scatter, gather, all-scatter, all-gather, all-to-all, reduce, reduce-scatter and all-reduce. Movement returns indexed sections; only the supplied combining function performs reduction arithmetic. |
| AOT bindings, zero-copy asynchronous use | `mesh_call_bind` realizes indexed uses and operand storage. Swift prepares native views before `mesh_calls_start`. `link_configure` realizes routes and posts receives before `verbs_up` enables sends. `mesh_receive_assign` places each chunk through page-index assignment over registered aliases without copying. Dedicated TX/RX threads post and drain; numerical completion publishes only the corresponding value's uses. |
| Delete incompatible implementation and callers | The former executor/frontend and engine adapters are absent from the current tree. The source inventory includes their replacements. The index transport channel, paired native-binding Cartesian product, per-page receive metadata and process-memory ranking scan are also absent. |
| Actual producer/collective/numerical-consumer integration | `linear-chain.swift` applies the supplied T to produced and transported partials before reconstruction. `coreml-chain.swift` composes two native predictions through transferred sections. Each declares once and submits four distinct indices. |
| Automatic lifetime; explicit synchronization only | `mesh_buffer_retain` accounts for declared uses. `mesh_publish`, native numerical completion, TX completion and ordinary object destruction discharge their references; `mesh_collect` returns backing without clearing payload. Runtime presence polling occurs only in the explicitly called `mesh_sync_on_remote_fill`; its source counterexample includes a self-dependent permanent wait. |

Configuration loops, capacity checks while posting native work requests, indexed
operand dependencies, reference updates and native launch operations remain.
There is no whole-function readiness scan, caller release protocol, remote
consumer acknowledgement or default remote-fill wait. The finite storage extent,
one-peer transport and native API contracts above remain explicit limits. Builds
establish integration consistency; no runtime measurements or speedup claims are
used as evidence for these source properties.

The generic Metal invocation selects command buffers prepared during realization,
with pool capacity owned by Mesh for every declared call. Native view
preparation and raw callbacks both support multiple input and output operands.
The transport supports one peer. TX polls completions after each edge's available chunk posts instead of delaying
polling until the entire publication list is processed.
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
and retained callers compile at ABI 41. Compilation does not establish RDMA
execution or end-to-end integration.
