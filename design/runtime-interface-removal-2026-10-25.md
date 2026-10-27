# Runtime interface removal manifest

Recorded: **2026-10-25T16:50:37-07:00** (America/Los_Angeles).

Source reviewed: mesh `c42333365618a777c920572a67ce6aae1aa379db`; metal-microbench
`8d508ea59b99c0be17b8b234688ff601011eedb7`, both on `main`.

Status: **See each item below. R12–R14 now use the common prepared native submission path; R18 was removed in `18be9473`.**

The assignment is to remove execution interfaces which redo work that the library can already settle before
a call. Keep the numerical operations. Use the existing declarations, page plans, partial tensors and command
records to express them. Do not replace each deleted executor with another executor of the same kind.

The governing rules are [data-model.md](data-model.md), especially sections 5–6, and the operator's current
instructions: allocation and routing happen before a data call; runtime uses prepared indices; feedback changes
operands between whole calls; measurements do not own or alter execution. The older
[partial-tensor requirements](collective-goals-2026-09-14.md) also require early use of partial inputs.

This is a review of the library's major compute, graph, transport and measurement interfaces and their callers.
It is not a claim that every line in both repositories has been reviewed. An existing capability is named for
each removal. Where it lacks a complete replacement, that gap is explicit. A matching name or an older date
does not establish that an implementation is faster or meets the model.

## Integration status — 2026-10-26T07:38:01-07:00

The native recorder, access ordering, replay compiler and submission scheduler live in
`rdma/metal-recording.h` and `.m`. Native command descriptions, captured LiteRT commands,
prepared Torch transport and native CPU/ANE shares use that implementation. Native ANE's
predictor and marker threads, private semaphores, per-layer timers and copied-output substitute
are removed. Goal A now declares its whole tiled pipeline through `MetalProgram` and
`MeshRankRecord`; its separate two-queue loop, submission throttle and tile timers are removed.
Ragged reduce-scatter is a configured operation of the same rank crossing interface.

Python list gathers bind final output buffers. Native column gathers bind final row views.
KV handoff now declares each sender's and receiver's own row pitch and head offset to the same
write planner; its staging buffer, `placeColumnsCommand` and `place_columns` kernel are removed.
The coded drafter now produces natural columns and binds their indexed final views to that same
planner. Its staging tensor and `mesh_draft_place` permutation kernel are deleted. FX ragged
gathers also bind source and final recipient views; their `gather_land` kernel and fused-copy
rewrite are deleted. The LSTM produces its local columns in the final state tensor. A separately
supplied source still requires its ordinary local placement into a distinct local result; there
is no remote receive-and-copy-back substitute. Fused reductions retain their numerical fold.

TP, EP and CP declare whole functions. The CP example supplies compiler crossing indices.
Hourly workload bodies live in `mesh/examples`; the hourly supplies configurations. The former
A/B runner is now a reader of canonical hourly records, with no installer, script substitution,
private timer or numerical loop. `mesh-statistics.h` supplies ordinary host time and paired
latency/useful-rate arithmetic; Python and native callers import it. LM serving reuses the same latency/rate record
for display and history; speculative submissions also enter the shared scheduler.
Prepared Torch crossings use one native queue per framework stream. Native crossing
bindings now advance through the declared invocation slots with the same offset
calculation as recorded compute; strided WRITE views realize the peers' agreed slot period.

Collective repetitions, recurrent chain commands and DNN row blocks are prepared before enqueue.
FX native bindings and fused crossings share `Preparations` and its ordered operands; the LSTM
service supplies its complete call order to it. Span plans are bound after routing is assigned.
Recurrent inputs name the frame each step consumes. Invocation storage uses assigned ranges in
region pages; the free-buffer scan and `ncclMeshMemReusable` API are removed. Payload ranges keep
page alignment. The preceding 64-byte payload alignment caused a registration-boundary failure
in the canonical hourly and has been corrected. Descriptor storage keeps its separate alignment.

Complete forward/backward graphs use functionalization for mutating custom operators before
kernel lowering. Dead framework coordinate queries are removed from the graph; they previously
introduced a dynamic `nonzero` into TP lowering. These are preparation changes, not measured
performance claims.

Open work includes native producer-kernel publication, generic
Torch numerical spans' framework encoding and temporary storage, and the eager CPU compatibility
executor. Shared code and successful builds do not establish maximum performance. Canonical
hourlies have recorded failures; completion of all seven assigned items is not claimed here.
Nonleading FX reduce-scatter axes use original and final view fragments. Partial tensors and
FX use the shared `views.alias_runs`/`cut_runs` calculation and the same configuration-time
cut agreement, then the native reduction compiler. The duplicated partial-layout walker is removed.
The constrained data-model document is unchanged.

Updated 2026-10-26: FX numerical records now use the existing logical view calculation to declare exact
access ranges in the shared recorder. Strided holes, broadcast repeats and empty views no longer become
whole enclosing accesses. Binding reaches name argument indices and follow their prepared invocation slots.
Thread-count replay retains reaches when recorded into another program. Native streamed panel output rows
use the same interface. See [compiler-pipeline.md](../docs/compiler-pipeline.md#tensor-views-in-command-dependencies).
This closes that metadata loss; it does not implement early producer publication or remove the remaining
native commands' coarse buffer declarations.

Further update 2026-10-26: `MetalViewReaches` now performs the access-union calculation for both native
and FX descriptions. Native normalization, ordinary packed matrix products, stacked products, pointwise
row operations and head outputs supply their layouts. Fused epilogues preserve those declarations when
capturing addresses. [The publication review](../docs/native-access-publication.md) distinguishes these
completed changes from the still-open binding of native tile completion to physical WRITE prefixes.

## Integration — 2026-10-26T21:13:39-07:00

The seven current interventions are recorded in
[compiler-pipeline.md](../docs/compiler-pipeline.md) and
[native-access-publication.md](../docs/native-access-publication.md): prepared
cache/view accesses; row/head publication; graph declarations for the existing
selection, rotation and coded kernels; configured LiteRT roles and one capture;
direct CPU/ANE views and general exported graph binding; joint assignment and
compiler operands; public page-mask WRITE declarations.

Removed execution interfaces: eager coded/sampling/rotation dispatches, the CPU
Torch operation loop, native ANE input publication copy, share concatenation,
LiteRT's pre-capture payload warm run, and the public raw-range WRITE preparers.
The common recorder, scheduler and WRITE executor remain the execution path.
No hourly, timing, measurement or unit-test implementation was added.

## Reviewed interfaces

The bullets below preserve the removal rationale. The integration update above
records the current status; unmarked entries remain outstanding.

### R01. Per-operation engine orchestration — replaced with prepared bindings

- **Remove:** the execution contract of `Operation.__call__`, `_decision`, `_shard` and `_separate` in
  [engines.py](../rdma/torch-mesh/torch_mesh/engines.py). It chooses shares, prepares or loads shards, allocates
  outputs, starts host futures, joins them and combines outputs inside an operation. `_shard` can reach
  `Ane.load`; the standalone `prepare(engines)` does not prevent this path.
- **Use:** that module's existing `Engine.prepare`/`run` numerical adapters, `neural_engine_share`, whole-function
  [assignment operands](../rdma/assignment.py), and [declarations.Preparations](../rdma/torch-mesh/torch_mesh/declarations.py).
  Native [NeuralEngineShares](../../../metal-microbench/neural_engine.swift) also already binds shared input,
  result and progress storage before execution.
- **Integration still required:** declare all operation shapes, allocate their results, load their models and
  bind CPU/GPU/ANE work before the call. CPU/ANE records now declare input/output ranges in the shared schedule;
  the start/consume token interface and Python runtime callback have been removed. Preserve `Linear`/`FFN`
  mathematics and the engine capabilities while deleting the hidden execution policy.

### R02. SparseLinear's private optimizer and timer — removed

- **Remove:** `SparseLinear.route`, `_settle`, and the timing/selection state and branches in `forward` in
  [sparse.py](../rdma/torch-mesh/torch_mesh/sparse.py). They alternate implementations, create timing events,
  synchronize pending work and apply a statistical acceptance threshold within operator use.
- **Use:** `Closure`, the existing dense and tile numerical kernels, declared output storage, and whole-call
  observations through [rollout.py](../rdma/rollout.py) and `assignment.assign`.
- **Integration still required:** make the selected lowering a configuration operand of the whole function.
  Keep both numerical capabilities where useful. Delete their independent runtime selection interface.
  This path is exposed as a library module; this review does not establish that the current hourly calls it.

### R03. The Python partial-tensor executor's second scheduling system — removed

The worker threads, condition variables, token edges, ready heap, runtime guards, per-edge Work waits,
result-consume interface and blocking result helpers have been deleted. `Partial`, `Tensor`, `Ref`,
`BlockSpec`, `kernel_call` and the collective declarations remain. Realization resolves their dependencies,
compiles numerical spans with `declarations.Function`, and generates a fixed FX call sequence over those
spans and native prepared bindings. It executes no initialization payload. Input access exposes the existing
tensor without NumPy staging or implicit publication. Distributed CPU partial graphs fail preparation;
the synchronous CPU transport is not used as a fallback. Shared native stream concerns remain R17.

### R04. Attention's runtime workspace pool — removed

Updated 2026-10-26T20:42:55Z. `_lease`, `_returned`, `_FREE`, `_MASKS` and the custom backward's storage
management are deleted from [attention.py](../rdma/torch-mesh/torch_mesh/attention.py). Attention declares
its math; autograd includes all three input gradients in the complete function graph. The causal mask is
an index relation in that graph, with no lazy cache.

[native.py](../rdma/torch-mesh/torch_mesh/native.py) now binds every compiler-created intermediate and
linear output to region storage during preparation. This replaces the oversized MPS heaps that motivated
the old pool in `00d2b32c`; it does not return score allocation to that allocator. Storage reuse is assigned
by the preparation program before native command recording. Numerical execution does not call that program.

`BoundGraph` also binds subsequent spans and collectives directly to those native outputs. Its separate
destination allocations and inserted output `copy_` operations are removed; output aliases remain views.
Preparation compiled a causal training graph at the TP example's `[2, 5, 1024, 128]` shape with a collective
boundary and an aliased result, as well as rectangular query/key/value shapes. No numerical work was
submitted, so this records integration and compilation evidence, not attained latency or bandwidth.

### R05. Coded-matrix calls that choose and allocate while executing

- **Remove:** runtime realization from `Matrix.dispatch`, `coded_input`, `product` and their `_pipeline`/`_encode`
  route in [streamed.py](../rdma/torch-mesh/torch_mesh/streamed.py): repeated shape choices, output/scratch
  creation and first-use pipeline construction must leave the numerical call.
- **Use:** the same `streamed.metal` templates and tensor representation; native `StreamedMatrix`,
  `StreamedDerived`, `streamedInstance` and `MetalProgram` in
  [streamed_weights.swift](../../../metal-microbench/streamed_weights.swift) and
  [matrix_operations.swift](../../../metal-microbench/matrix_operations.swift) already separate command
  preparation from submission.
- **Integration still required:** retain PyTorch dispatch, slicing, coded weights and gradients as declarations
  lowered before execution. A first-use cache is not a guarantee that a call is prepared.

### R06. Small-kernel allocation and lazy compilation wrappers

- **Remove:** allocation and first-use compilation from `argmax.draw`, `retain`, `_rows`, `_kernel`, and
  `structured.Rotation.rows`/`rows_inverse` when they execute a declared numerical function.
- **Use:** their existing [argmax.metal](../rdma/torch-mesh/torch_mesh/argmax.metal) and
  [structured.metal](../rdma/torch-mesh/torch_mesh/structured.metal) bodies, prebound operands and the native
  [draw/retain command builders](../../../metal-microbench/decode_crossings.swift).
- **Integration still required:** result and temporary pages become explicit prepared outputs. The Python
  tensor-facing operations remain available as frontend declarations, without a second storage policy.

### R07. Chain-specific reconstruction of known commands and block readiness

- **Remove:** rebuilding known dispatch bindings in `chain_run` in
  [_stream.mm](../rdma/torch-mesh/torch_mesh/_stream.mm), and the whole input-block readiness condition in
  `chain_input` in [chain.metal](../rdma/torch-mesh/torch_mesh/chain.metal).
- **Use:** `chain.operands`/`prepare`, the fused numerical kernel, partial-input dependency descriptions,
  `MetalOperand`'s invocation slots and recorded command bindings. Preparation must retain the current
  physical multi-link write capabilities.
- **Integration still required:** compile each actual input dependency and publication at the granularity its
  consumer needs. The block wait was restored by **our `9f80ce00`**; it is not justified merely because an older
  implementation had it. The historical fast `9a167a57` also used `dispatch_sync` for host encoding, so that
  queue call alone is not a bisection explanation for the later performance loss.

### R08. DNN's whole-output join and handwritten dispatch loop — replaced with recorded row blocks

- **Remove:** `DNN.run`'s repeated binding/list construction and its `land_ready` whole-operand join in
  [dnn.py](../rdma/torch-mesh/torch_mesh/dnn.py).
- **Use:** its existing numerical product/fold bodies and prepared writes, with `BlockSpec`/partial declarations
  and recorded execution. Historical `da21ce68:python/mesh/nn.py` records K-panel products and combines by output
  tile; that capability must not be replaced by waiting for every unrelated tile.
- **Integration still required:** publish and consume useful partial ranges through the common transport.
  Reusing the transport call alone has not supplied this dependency granularity.

### R09. Per-issue collective schedule construction

- **Remove:** execution-time `steps_copy`, `group_bound` allocation/copying and unplanned compilation,
  `place_group` descriptor construction, `enqueue_counts` segment allocation, and corresponding per-launch
  metadata allocation in [nccl-mesh.c](../rdma/nccl-mesh.c).
- **Use:** `ncclMeshCommPlan`, `mesh_routed_compile`, and the existing prepared write machinery. The common
  [mesh-span-plan.h](../rdma/mesh-span-plan.h) and [mesh-row-span-plan.h](../rdma/mesh-row-span-plan.h) already
  compute matching cuts across registration boundaries before execution.
- **Integration still required:** all collective forms must consume those cuts and immutable schedules.
  `ncclMeshWriteExecute` still enters the generic enqueue/launch machinery. Naming a path “prepared” has not
  removed that machinery. `place_group` can still reject its own generated span at a registration boundary.

### R10. Runtime scratch acquisition and reuse queries

- **Remove:** `spare_take` and `spare_where` as runtime scratch suppliers, and numerical execution's dependence
  on `ncclMeshMemReusable`/lease-retirement queries to choose storage.
- **Use:** preallocated page slots, [lifetimes.py](../rdma/lifetimes.py)'s static dependency relations,
  `mesh_routed_program` retirement relations and native invocation-slot bindings.
- **Integration still required:** compile causal reuse into the page plan. Keep the bridge's hardware queue
  credit accounting and actual data publication relations. Those are distinct from a numerical allocator
  searching for a reusable buffer. `LifetimePlan.tracker` is not the desired replacement: it is another
  runtime observer; no external caller was found in the searched library/example source.

### R11. Raw-pointer/extent operands at the program boundary

- **Remove:** the program-facing contract that permits arbitrary pointers, byte lengths and spans through
  `ncclMeshWritePrepare`, `ncclMeshWriteRowsPrepare`, `ncclMeshWriteSpansPrepare` and `ncclMeshSpan` in
  [nccl.h](../rdma/nccl.h).
- **Use:** page identity plus a mask limited to that page, lowered through the existing affine and common-cut
  planners into local bridge descriptors. `migration.compile_rows` and `views.byte_spans` contain useful
  layout mathematics for that lowering.
- **Integration still required:** a constrained public page/mask type is not supplied merely by those byte-span
  helpers. Local device bindings may contain addresses and lengths; application write operands must not.
  NCCL/PyTorch compatibility is a frontend lowering responsibility, not permission for a competing transport.

### R12. Native submission throttling and allocation — removed

Updated 2026-10-26 03:06 UTC; implemented in metal-microbench `e539114e`. `MeshRankPrepare` encodes the complete declared call, including relay-only
ranks and speculative epilogues, before execution. `MeshRankRun` commits that stream. `MetalSchedule`
holds its command buffer, argument table, residency and completion observer. There is no submission array,
per-batch semaphore, two-batch cap, host relay executor or callback that releases the next batch.
Native, DSV41 and LiteRT callers use this interface. DSV41 prefill writes decode's input operand directly.

Provenance: `92f8de55` (2026-09-25) introduced the two-window loop in `tools/e2b_replay.m`; `a542178b`
(2026-09-26) moved it into this shared library. Both name zyzzyva as author and Claude Opus 5.5 as co-author.
`22589255` (2026-10-16) carried it into encoded collective crossings; `caa90712` (2026-10-21) renamed
windows to submissions. `b1633730` removed the argument-ring bound on October 22 but kept the two-batch cap.
The original rationale was streamed token reporting and stopping at window boundaries. These commits give
no hardware requirement or performance result establishing that the cap was necessary.

Whole-function completion supplies elapsed time and useful-output rates. The per-phase host timing loop,
`runs_ms`, `submission_steps` launch option and native GPU interval counter are removed. Build validation
alone does not establish a performance gain.

### R13. LiteRT's implicit late preparation and blocking stage executor — removed

Updated 2026-10-26 02:01 UTC. `LiteRtProgramDeclare` now owns binding, stage warmup and command recording.
The phase declarations and speculative closing step are prepared by `MeshRankPrepare` and submitted by `MeshRankRun`.
The unrecorded stage executor, late recording fallback, WebGPU host executor and `MeshRankDeclareHost` are gone.
`MeshRankCross` declares a crossing during recording; it no longer selects eager execution outside a recording.
The call explicitly configures Metal recorded execution; an unsupported provider fails preparation.

The importer consumes the shared phase parts and KV handoff declarations. Model configuration states transfer
type, width and axis order. Existing graph tensors supply the source and destination head views, and the
receiver reuses the export's cache-update fragment. Whole-phase placement and imported tensor metadata do not
select an execution backend at runtime. R12 and R14 describe their common execution and access contract.

### R14. Whole-buffer ordering in place of known partial accesses — removed

LiteRT receives the shared recording device and queue. Pipelines retain their reflected read/write modes;
`MetalRecordingAdopt` and its blanket serialization are deleted. The LiteRT binding stage supplies
`MetalBindingView` records with each external tensor's buffer, offset and byte extent. `MetalRecordViews`
applies those declarations during capture. Internal buffers use their binding extent; pointer reaches still
require `MetalDispatchReaching` declarations. An unreflected pipeline fails preparation.

One access-order calculation serves individual records, repeated slots and the complete function. It uses
byte overlap and the actual invocation's slot, including output offsets. It does not equate sharing an
allocation with a dependency. Known native subranges continue through `MetalReach` and `MetalReachBinding`.

### R15. Measurement modes that change the program they observe

- **Remove:** `StepProfile.pass` and `forceDirect` as execution/timing modes in
  [matrix_operations.swift](../../../metal-microbench/matrix_operations.swift), with their switches in
  `lm_engine.swift` and `dense_execution.swift`; `_stream.timed` and the private operation-timing loops in
  `sparse.py` and `streamed.py` as alternate performance measurement mechanisms.
- **Use:** passive whole-function observations, imported library workloads and the canonical hourly record
  of elapsed time together with useful GB/s (and rows/s for the chain).
- **Integration still required:** measure the opaque function that the application runs. Reading an existing
  completion callback differs from inserting counter samples, forcing direct execution or draining for a
  precise timestamp. This item does not classify every existing completion counter as a synchronization.

### R16. Runtime FX crossing lookup alongside prepared indices — removed

`eager_index`, `_EAGER_CROSSINGS`, their occurrence cursor, and `prepare_gather` have been deleted.
`Preparations` requires compiler-assigned indices. `Function` always uses the shared mesh compiler;
its optional eager/compiler construction contract is removed. Unlowered multi-rank gather and fused-sum
operations fail instead of running a metadata lookup or a generic gather substitute. The LSTM caller uses
the compulsory compiler interface. Single-rank copies and products remain their ordinary numerical forms.

### R17. Implicit stream drains and a synchronous host collective executor

- **Remove:** `submit`'s whole-stream completion/drain before a host collective in `_stream.mm`, and the
  synchronous numerical scheduling contract of `launch`/`run` in `nccl-mesh.c`. `run` inspects step readiness
  and retirement state while carrying out a collective; the host adapter blocks until that execution finishes.
- **Use:** the same prepared physical write plan and exact producer/consumer relationships used by the GPU
  path, with prebound CPU work and publication operands. Keep `mesh_mpi.Comm`'s mathematical operations as
  declarations rather than a second execution policy.
- **Integration still required:** preserve real CPU dependencies without draining unrelated GPU work. The
  current common compiler is reusable; a complete asynchronous prepared CPU executor is not established here.
  Historical `3d32355e` is a reference for prebound CPU operands, not permission to restore a deleted scheduler.

### R18. First-issue compilation and its peer barrier — removed

Added 2026-10-25T17:22:35-07:00. Removed `torch_mesh.record`, `Recording` and their first-issue graph cache,
TCPStore compilation rendezvous, warm execution and state reset in the LSTM service. The service now prepares
each configured chunk through the existing `declarations.Function` interface. It binds the complete declaration
and calls the compiled functions directly. The numerical tick body and common collective compiler are retained.
This closes the duplicate graph-preparation path; it does not close the native submission and crossing issues
listed above. Whole-function performance is recorded by the canonical hourly.

## Existing capabilities to preserve and connect

“Aligned” below describes the named mechanism, not a blanket performance certification of its callers.
The [graph history](aot-graphs-and-pytorch.md) explains the connections and prior deletions.

| Existing source / main types | Problem already addressed | Adherence and integration finding |
|---|---|---|
| [mesh.h](../rdma/mesh.h), `mesh_entry`, `mesh_stream`; [mesh-call.h](../rdma/mesh-call.h), `mesh_endpoint`, `mesh_stream_view` | Shared storage, indexed stream records, physical wire identity | Core page/stream machinery exists; the exposed extent and per-issue contracts still need R09–R11. |
| [mesh-dataflow.h](../rdma/mesh-dataflow.h), `mesh_ctx`, `mesh_section` | Attach clients and bind already allocated shared regions | Use allocation and region ownership at setup. Do not import client lifecycle machinery into each numerical step. |
| [mesh-bridge.h](../rdma/mesh-bridge.h), [mesh-verbs.h](../rdma/mesh-verbs.h), `mesh-link.c` | Persistent device queues, posted receives, DMA forwarding, frame credits | Preserve the single device writer and credit accounting. They do not justify numerical allocation or ACK protocols. |
| [mesh-observe.h](../rdma/mesh-observe.h), `mesh_link_info`; [mesh-diagnostics.h](../rdma/mesh-diagnostics.h) | Shared topology/health records and explicit administration | Keep observation and administration separate from numerical execution and its performance measurement. These are not alternative collective backends. |
| [mesh-plan.h](../rdma/mesh-plan.h), `mesh_fabric`, `mesh_cable`, `mesh_paths`, `mesh_routed_program` | Many physical cables, paths, relays, collective lowering | Routing expressiveness exists. Do not replace it with a one-cable peer adapter. Compiler output still reaches R09. |
| `mesh-collective.c`, `mesh-fabric.c`, `mesh-routed.c`; `mesh_trees`, `mesh_program`, `mesh_step` | Logical collective forms and physical transfer plans | Keep alternative mathematical forms as compiler inputs to one physical executor. A chosen tree is not proof of optimality. |
| [mesh-write-routes.h](../rdma/mesh-write-routes.h), [mesh-write-order.h](../rdma/mesh-write-order.h) | Path construction and AOT relay-fragment ordering | Useful compile-time mechanisms; they are not an empirical guarantee about every whole call. |
| Span/row planners, [migration.py](../rdma/migration.py), [views.py](../rdma/views.py) | Ragged layout, affine rows and common DMA cuts | Reuse the layout mathematics; finish page-mask constraints and generic-collective integration. |
| [mesh.metal](../rdma/mesh.metal), [nccl-mesh-kernels.h](../rdma/nccl-mesh-kernels.h), `ncclMeshFused` | Coherent payload access, fused publication and consumption | Useful shared primitives. Current polling granularity and timeout/guard paths are not evidence of minimum overhead. |
| [nccl-mesh-metal.h](../rdma/nccl-mesh-metal.h), `metal_program`, `metal_wait`, `metal_release` | Bind collective numerical operations into Metal encoders | Shared backend machinery, with current allocation and coarse dependency gaps. Callers should supply the compiled relation, not an independent wait policy. |
| [routing.py](../rdma/routing.py), `Fabric`; [assignment.py](../rdma/assignment.py); [allocate.py](../rdma/allocate.py) | All-simple-path capacity model, shares, finite-difference feedback and projections | Configuration-side operands fit the model. Active measured hop mean/spread integration was not found; `completion_model` had no external caller in the searched scope. |
| [lifetimes.py](../rdma/lifetimes.py), `Event`, `Read`, `Transfer`, `LifetimePlan` | Compile causal page-reuse relations | Preserve static analysis; remove runtime storage choice under R10. |
| [python/mesh](../python/mesh/__init__.py), `Partial`, `Tensor`, `Ref`, `BlockSpec`, `Program` | Partial sums, slices and recorded tile calls | Strong existing declaration vocabulary; its former runtime scheduler is replaced by compiled spans and native bindings (R03). |
| [collective.py](../python/mesh/collective.py) | Gather, scatter, reduce, partials and prepared strided moves | Preserve semantic operations and common transport. Whole-fill synchronization is explicit, not a default. |
| [partition.py](../rdma/torch-mesh/torch_mesh/partition.py), `RaggedShard`; [declarations.py](../rdma/torch-mesh/torch_mesh/declarations.py) | Ragged DTensor integration; fake tracing; full forward/backward declarations; indexed crossings | Valuable PyTorch frontend work. It does not yet make all eager callers or all backend launches AOT (R09); indexed crossing lookup is now mandatory (R16). |
| [mesh.py](../rdma/mesh.py), [mesh_mpi.py](../rdma/mesh_mpi.py), `WritePlan`, `Comm`; [backend.py](../rdma/torch-mesh/torch_mesh/backend.py), `ProcessGroupMesh` | Python, NumPy/MPI-style and PyTorch access to the same native collective library | Preserve frontend compatibility; raw extent contracts and synchronous host execution remain R11/R17. |
| `_stream.mm`, `PreparedWrites`, `FusedForm`, `PreparedChain`; [fused.py](../rdma/torch-mesh/torch_mesh/fused.py) | Shared storage and fused crossings from PyTorch | Keep the bindings and numerical fusion; remove per-operation realization and excess joins (R07–R09). |
| `linear.metal`, `streamed.metal`, `sparse.metal`, `structured.metal` | Dense/grouped products, coded-weight streaming, tile masks and rotations | Reusable numerical bodies, including native callers. Their Python wrappers still have R02/R05/R06 work. |
| `attention.metal`, `argmax.metal`, `drafter.metal`, `speculate.metal` | Online attention partials, selection and speculative state | Keep numerical semantics and indexed state. Decode attention does not supply training backward automatically. |
| [metal_recording.h](../rdma/metal-recording.h), `MetalReplay`, `MetalOperand`, `MetalReach` | Record commands, bind invocation slots, derive access order and residency | Major AOT foundation; R12/R14 removed. ICB, encoded and serial forms share this representation. |
| [matrix_operations.swift](../../../metal-microbench/matrix_operations.swift), `MatrixView`, `MetalDispatch`, `MetalProgram` | Strides, partial access descriptions and prepared numerical commands | Reuse the preparation interface; remove intrusive timing modes (R15). |
| [metal_native.h](../../../metal-microbench/metal_native.h), [metal_runtime.swift](../../../metal-microbench/metal_runtime.swift), `Generation`, `Placement` | Shared kernel source, pipeline preparation, model dimensions and ragged placement | Configuration work already exists. Callers must complete it before the numerical function. |
| [instance_fit.swift](../../../metal-microbench/instance_fit.swift), `InstanceSpec`, `SliceUnits`, `InstanceFit`, `AttentionPlan` | Represent kernel work and choose prepared product/attention geometry | Useful configuration metadata. A fitted choice must feed prepared operands and whole-call feedback; it is not authority for another operation timer. |
| [engine_recording.swift](../../../metal-microbench/engine_recording.swift), `RankCall`, `RankRecording`, `EngineRecording`, `EngineProgram` | Whole native prefill/decode phases over shared operand slots | Uses the recorder already; uses the prepared submission in R12. |
| [dense_execution.swift](../../../metal-microbench/dense_execution.swift), `Network`, `StepProgram`; [streamed_weights.swift](../../../metal-microbench/streamed_weights.swift) | Compose native model steps and coded/dense matrix work | Existing composition layer to reuse, not a reason to introduce PyTorch into the native engine. |
| [decode_crossings.swift](../../../metal-microbench/decode_crossings.swift), `CrossingPoint`, `StepCrossings`, `SpecOperands` | Collective operands, shared-prefix descriptors and GPU speculative state | Useful prepared data. Crossings still inherit common collective executor gaps. |
| [litert_program.h](../../../metal-microbench/litert_program.h), `LiteRtProgram`, `LiteRtPositions` | Upstream model compiler plus native command capture and shared tensors | Separate frontend using the same recorder; R12–R14 removed. |
| [neural_engine.swift](../../../metal-microbench/neural_engine.swift), `NeuralEngineShares`, `NeuralEnginePartner` | Prebound CPU/GPU/ANE shared work | Storage, numerical adapters and declared host dependencies use the shared recorded schedule (R01). |
| [mesh-coreml.m](../rdma/mesh-coreml.m), `mesh_coreml_predict`; `engines.Ane` | CoreML model execution over shared tensor storage | Retain the engine capability. Model preparation must be outside the call; the current wrapper is not certified as allocation-free native execution. |
| Native `PageManager`, prefix trie; [dsv41.swift](../../../metal-microbench/dsv41.swift) model composition | Paged model state, shared prefixes, another model's native phases | Capability identified. Session management is not certified as a fixed numerical schedule; it is not a replacement transport. |
| [rollout.py](../rdma/rollout.py), [store.py](../rdma/store.py), `configuration.bind`/`record` | Asynchronous whole-call observations and in-memory feedback | Correct separation of observations from numerical execution. Observation history currently has process lifetime; late records do not drive a new per-op controller. |
| [mesh_hourly.py](../rdma/mesh_hourly.py), microbench's [hourly entry](../../../metal-microbench/tools/mesh/hourly/hourly.py) | Run committed library workloads and record time/useful rate | Entry imports the driver. Do not add numerical code, a second measurement implementation or a different runtime here. |

## Removal boundaries

The deletions remove runtime policy and duplicated realization, not useful mathematical operations or hardware
support. Model loading, validation of a new configuration, graph construction and compiler allocation belong
before a call. A semantic choice in the numerical problem can be expressed as prepared data and masks; it does
not grant permission to discover routes or storage while a transfer is running.

The three `MetalForm` values are not three collective implementations. They consume common records, and the
native history documents a reason for direct recorded dispatches: an ICB's barrier can cost more than a Metal 4
stage barrier. Some pipelines also lack ICB support. Keep the shared representation and select the form before
execution. Do not claim that using an ICB proves maximum performance.

Closure of these entries requires the affected callers to use the shared prepared capability, with their
numerical features intact. Performance evidence belongs to committed-main hourlies: whole-call time paired
with useful bandwidth across the configured shapes. This manifest reports source findings, not an unmeasured
claim that every proposed substitution already improves that curve.


## FX integration update — 2026-10-26T17:30:31Z

`TorchSchedule`, its framework/native start/join handoff, indexed crossing execution, `Preparations`, and
`_CAPTURE` are removed. `native.compile_graph` uses numerical IR and Metal source generation during
preparation; its wrapper binds `_stream.Commands` once. Those commands and collective bindings append to
`MetalSchedule` before execution. LSTM, TP/EP/CP, partial tensor programs and engine-share composition
use that same native submission. The Dynamo backend entry point and eager allocating linear executors
are removed. `matrix.metal` is the DNN matrix body, now shared with larger FX products. The existing
small-row linear body remains in use.

CPU/ANE engine shares use the native recorder's access declarations and bound native operations. Generic CPU FX emission remains
unsupported; missing lowering errors do not select a framework executor. This update does not close every
entry in this manifest. Native and extension builds passed; whole-call latency and useful bandwidth still
require the canonical committed-main run. No new test suite or alternate measurement path was added.
