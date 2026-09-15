# Asynchronous partial tensor calls

The [user's requirements](collective-goals.md) define scope. This document
explains the source; it does not add requirements.

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
uses the engine's existing `MatrixOperations.multiply` for both stages. Row
sections of sizes 1, 8, 16 and 4 go through scatter, `X_i W`, gather, and a
consumer computing `(X_i W) W`. The gather returns indexed sections; it does
not assemble a copied dense tensor. The backend argument goes directly to the
existing operation constructor. The fixture weights are canonical shared
operands declared as constant inputs to the calls. Four submitted indices reuse
the same chain and weights; index-dependent input data reaches both matrix stages.

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
table and invokes the supplied function. CPU completion is its return. Metal
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
higher-order `map` overload takes a function from two `MeshSpan` values to a
native call. During setup it binds the actual possible receive positions and
each indexed output position. A local-input call selects its binding directly
by value index. A received-input call uses one physical-page-to-slot table and a
dense array of the configured input-position/output-index bindings.
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
already submitted index. Serving integration for continued prefill and speculative verifier
invocations remains unfinished. The matrix executable uses existing engine
numerical code; it is not a completed language-model serving integration.

Transport sends each queue direction's realized frame length. Frame rounding,
padding between unequal partials sharing a queue, source-tag handling, publication
and reference-count atomics remain actual costs.
For N indices and R possible receive positions, a native received-input factory
constructs N*R fixed native bindings at setup, plus one slot map over arena blocks.
Local-input factories construct N bindings with no page lookup map. This avoids
N sparse arena-sized tables, but the N*R native combinations remain setup work.
Shared function metadata is constant in N; per-value operands, uses, pending
counts and backing grow with the finite extent. Submission touches the root
workers only; publication visits the published row's uses, not all functions.
These are explicit remaining costs; no claim of zero total overhead, JACCL cost parity, or speedup
over world size 1 follows from this source change.

There is no runtime testing gate here. The bridge, C and Swift libraries, literal
chain, Core ML chain, synchronization counterexample, and existing-matrix chain
were built. They have not been run or deployed by this change. Both participants
need the source's ABI 39 bridge before these callers can attach.

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
| All mesh repository source files with the extensions below | 665,701 lines / 1,056 files | 651,787 lines / 994 files |
| Replaced paths, including new Swift code and old root setup.py | 16,378 lines / 77 files | 2,464 lines / 15 files |
| Build metadata in those paths, including pyproject.toml | 47 lines | 27 lines |
| Engine's deleted mesh_matrix.swift and tools/mesh/sync.sh; replacement matrix example | 203 lines | 60 lines |

The replacement-path set is `rdma/`, `python/`, `swift/`, `examples/`,
`xonotic/solver/`, `xonotic/planner/`, and the old root `setup.py`. Source extensions
are `.c .h .m .mm .swift .py .metal .sh .zsh .js .ts .jsx .tsx .qc`. The whole-repository
row includes the large unchanged Xonotic sources. It has not been halved.
Makefiles, module maps and pyproject.toml are reported separately above. Shared
engine dependencies `parameter_configuration.swift`, `matrix_shaders.swift`, and
`matrix_operations.swift` remain 480 maintained lines in the working tree; the
new caller reuses them. This change does not attribute their existing working-tree
edits to the mesh refactor.

Markdown is excluded from these source counts. Earlier documentary deletion and
scope correction were committed separately as mesh `a07d7f6` and engine `14d1057`.
This implementation explanation is documentation, not structural source reduction.
