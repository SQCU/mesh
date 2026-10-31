# Compiler and runtime interface inventory

Recorded 2026-10-31, 13:21 PDT. Source review only; no numerical execution or performance claim.

The initial checkout was mesh `f82f0b15`. During review, concurrent changes reached `99b05c3c`; the collective findings below were rechecked against that revision. The other repo is metal-microbench `c6f87fc2`. Source references describe these revisions, not an unspecified future main.

The compiled numerical path is connected, but the complete dependency-lowering requirement is not met. A prepared function does not rerun its Python lowering on each invocation. Its emitted program still contains broad encoder barriers, whole-group landing joins, runtime readiness searches, and host submission gates. Moving the construction of these instructions to preparation does not remove their execution cost.

## 1. The pool was an incomplete replacement, not a formal requirement

The specific fallback discussed in this session was `zabbadoo` → `metal_tracks`/`TRACK_KERNEL`. It no longer exists at `99b05c3c`:

| Commit | Change |
|---|---|
| `4530374d`, October 30 10:59 PDT | Replaced per-track workers with a common pool claiming ready chunks. |
| `a77e5b74`, October 31 00:42 | Added direct grouped reductions and landing waits alongside the pool. |
| `1fa09f2c`, October 31 06:46 | Moved direct reduction plan construction to placement; retained the pool for unsupported cases. |
| `c6fec3723`, October 31 06:50 | Sent reductions following COPY landings to the pool. Its source comment records starvation when waiting groups occupied the GPU. Other comments record overlapping sums losing contributions. These are implementation defects, not evidence that a second executor is necessary. |
| `a9bd4e7a`, `53c1fe72` | Deleted the pool from collective construction, Metal emission and kernels. |
| `99b05c3c`, October 31 13:19 | Deleted the host mirror of in-flight groups, their progress counters and ordered retirement. |

Deletion did **not** finish the mandatory domain. [zabbadoo_plan](../rdma/nccl-mesh.c#L4810) now rejects reductions without landing runs, reductions in repeated frames, reductions following COPY landings, differing input landing runs, and cross-call overlapping sums or unorderable grouped sums. Some malformed dependency graphs should be diagnosed during preparation, but repeated frames, different partitions and COPY→reduction dependencies are ordinary program structures. Their refusal is a lowering gap, not a hardware capability restriction.

The remaining reduction also calls [mesh_graph_take](../rdma/nccl-mesh-kernels.h#L219). That helper scans readiness, claims work with atomics, and repeats until work becomes ready or a stop condition holds. Deleting the generic pool did not eliminate runtime scheduling.

No project requirement found calls for the deleted pool. Correctly representing the cases it covered is required. The replacement must cover those cases through the shared dependency and page operands; it cannot declare completion by replacing fallback with rejection. The source comments explain why the duplicate survived, without establishing permission for it or proving either implementation fastest across all domains.

## 2. Current API pairs and conditional substitutions

Here, **fallback** means missing support or a condition redirects a request into a different behavior. **Competing interface** means two entry points retain different execution contracts; this does not assert automatic delegation between them. Backend selection from explicit capability/cost operands and a genuinely required AOT layout conversion are separately identified, rather than being mislabeled as fallback.

Torch paths in this table are under `rdma/torch-mesh/torch_mesh`. Microbench paths name the sibling repository.

| ID | Pair and exact trigger | Current effect |
|---|---|---|
| F1 | Prepared `Function` / direct `RaggedTensor.__torch_dispatch__`: [declarations.py:836](../rdma/torch-mesh/torch_mesh/declarations.py#L836), [partition.py:1067](../rdma/torch-mesh/torch_mesh/partition.py#L1067) | Competing public contracts. The latter performs placement work and invokes local operations. It is not established as a fallback from prepared `Function`. |
| F2 | Direct repartition / replicate then repartition: [partition.py:876](../rdma/torch-mesh/torch_mesh/partition.py#L876) | Same shard axis with different partition identities introduces a `Replicate` intermediate. Default synchronous redistribution also waits after each async transform at 902. |
| F3 | Ragged reshape/view / replicated reshape: [partition.py:1947](../rdma/torch-mesh/torch_mesh/partition.py#L1947) | If no candidate single-axis output shard preserves `_view_region`, replicate the source dimension and retry. The general page relation is not consumed here. |
| F4 | Partial distributed operation / replicated operation: [partition.py:2023](../rdma/torch-mesh/torch_mesh/partition.py#L2023) | Partial placements outside exact `Partial(sum/max/min)`, and scalar-shape-adjuster operators, go through `_whole` and retry after gathering. |
| F5 | Distributed equality / replicated equality: [partition.py:1108](../rdma/torch-mesh/torch_mesh/partition.py#L1108), applied at 2009 | `aten.equal` belongs to `_WHOLE`; ragged dimensions are replicated. |
| F6 | Custom all-reduce / upstream functional collective: [backend.py:459](../rdma/torch-mesh/torch_mesh/backend.py#L459) | Mesh/MPS sum with fp16/fp32 takes the compilation-only branch, except singleton cloning. Other cases delegate to `_c10d_functional.all_reduce`. Current mesh lacks the old numerical process-group implementation, so delegation may fail for mesh; a successfully running slower mesh fallback is not established. Non-mesh groups retain upstream execution. |
| F7 | Declared communicator / imperative NumPy MPI adapter: [mesh_mpi.py:161](../rdma/mesh_mpi.py#L161), methods at 186 | The adapter still calls NCCL with a null stream, copies noncontiguous Allgatherv input, allocates optional Reduce output and scalar Barrier/allreduce operands. [launch_metal](../rdma/nccl-mesh.c#L5318) now requires a prepared encoder/table/residency set. This is an incompatible residual interface, not a working host fallback. |
| F8 | Collective direct operand reads / staged operand copies: [plan_direct](../rdma/nccl-mesh.c#L5133), [operand_copies](../rdma/nccl-mesh.c#L5245) | Direct reads are disabled for counted/fused/premultiplied calls, multi-element reduction units, missing steps, some aliases and partial-range conflicts. Metadata allocation failure can also leave `direct=0`. The result is copied/premultiplied into the working operand. Each case needs a declared relation or operation; a false direct flag is not evidence that every resulting copy is necessary. |
| F9 | Fused crossing records / standalone collective dispatches: [launch_metal](../rdma/nccl-mesh.c#L5353), [fused_records](../rdma/nccl-mesh.c#L4992) | `list[0].fused && remote` chooses producer/consumer records; otherwise grouped sums and whole-channel landing waits are emitted. This is a declaration-driven split, not a runtime failure retry, but its dependency granularity differs. The standalone form remains unresolved. |
| F10 | Whole-rank prepared replay / service per-step planning: [rank.swift:670](../../../metal-microbench/rank.swift#L670), [lm_engine.swift:1851](../../../metal-microbench/lm_engine.swift#L1851) | Rank prepares invocation recordings. Service chooses a chain implementation and constructs plans at each admitted step, with one batch in flight. Same numerical leaves do not make these equivalent execution contracts. |
| F11 | Configured speculative work / ordinary autoregressive plan: [lm_engine.swift:1861](../../../metal-microbench/lm_engine.swift#L1861) | `speculative().flatMap(planVerify) ?? plan(...)`. Conditions at 1458–1472 include multiple ready decode sessions, controls/logits/CoT/bias modes, insufficient context, differing token position, and failed page availability. This is runtime compatibility fallback, not proof speculation is faster in every case. |
| F12 | Cached speculative lead / backup drafter or empty lead: [lm_engine.swift:1506](../../../metal-microbench/lm_engine.swift#L1506) | Runtime collection and hit/miss choose `.cached`, `.backup`, or no lead. This is not represented as one precompiled function consuming ordinary masks. |

Additional alternative lowerings and filters must be reconciled without conflating different semantics:

| ID | Source | Finding |
|---|---|---|
| L1 | [partition.py:2015](../rdma/torch-mesh/torch_mesh/partition.py#L2015) | Missing sharding propagation selects CompositeImplicitAutograd decomposition when available. Decomposition can be legitimate source rewriting; the old direct dispatcher makes it further runtime-dispatched work. |
| L2 | [partition.py:2209](../rdma/torch-mesh/torch_mesh/partition.py#L2209) | The special ragged-attention dispatcher rejects FakeTensor capture, masks, dropout and GQA; other placements use `_propagated`. Its newer `_ring_attention` at 2464 is traced source. The old real-only/graph-break restriction remains inconsistent with that implementation. |
| L3 | [engines.py:118](../rdma/torch-mesh/torch_mesh/engines.py#L118) | A coded-weight fused FFN is used only if provided and input gradients are unnecessary. Otherwise products, GELU and multiplication are expanded. `Prepared` captures that source in `Function`; this is not an eager numerical fallback inside replay. Gradient-domain integration remains a concrete distinction. |
| L4 | [native.py:141](../rdma/torch-mesh/torch_mesh/native.py#L141) | Small affine fp16/unit-stride inputs select `linear_commands`; other inputs select general `matrix.product`. Both emit prepared commands. The crossover is configured; other capability predicates are inline. Their relative performance has not been established here. |
| L5 | [native.py:1117](../rdma/torch-mesh/torch_mesh/native.py#L1117) | `page_pieces` on either node unconditionally vetoes fusion. This is not an actual pairwise conflict analysis. Distinct selected implementation IDs also prevent fusion. A layout annotation must not indefinitely exempt work from the common fusion pass. |
| L6 | [mesh-coreml.m:201](../rdma/mesh-coreml.m#L201) | CPU product binding chooses BLAS-compatible matrix strides or a prepared BNNS strided descriptor. Both are preparation choices; no exception-driven substitute was found. Their constraints need to remain compiler operands. |
| L7 | [litert_program.m:739](../../../metal-microbench/litert_program.m#L739) | Direct binding or prepared conversion/copy follows encoding, alignment, layout and output-slot constraints. Required AOT materialization for an actually contiguous consumer was explicitly authorized. This is not permission for generic staging when a page view suffices. |
| L8 | [model_file.swift:79](../../../metal-microbench/model_file.swift#L79) | Missing global KV-head metadata is filled from ordinary KV-head metadata. This is import-time implicit configuration, not a numerical runtime fallback. |
| L9 | [metal-recording.m:1564](../rdma/metal-recording.m#L1564), [1631](../rdma/metal-recording.m#L1631) | Encoded/indirect replay are configured forms. GPU-only ranges and mixed host/GPU work receive different preparation treatment. They share the recorder but still must preserve one dependency contract; neither distinction proves a slower fallback by itself. |

Unknown external native kernels currently fail preparation at `native.py:960/972`; no upstream ATen execution fallback was found there. LiteRT `record_stages` fails for an absent compiled declaration at `litert_program.m:3124`; no substitute interpreter was found. `host.py:9` rejects direct numerical host execution. `ncclMeshWritePagesPrepare`, `ncclMeshWriteLayoutPrepare`, `ncclMeshWriteExecute`, the old `_stream.encode` executor and the blocking C host collective executor are absent from the current inspected interfaces. Do not keep counting removed implementations as live ones.

## 3. Collective and bridge ordering still emitted or performed

This inventory includes every located mechanism, even where a particular data relationship may warrant ordering. Listing a mechanism does not prove every instance unnecessary. Its justification must be an actual page dependency or explicit backend constraint, not merely the fact that an earlier version used it.

| ID | Source | Runtime effect |
|---|---|---|
| C1 | [nccl-mesh-metal.m:211](../rdma/nccl-mesh-metal.m#L211) | `metal_barrier` emits a Dispatch/Blit → Dispatch/Blit device-visibility barrier. `metal_order` calls it for every ordinary dispatch; only subsequent publication dispatches and the explicitly marked spin batch bypass it. Call sites: `dispatch4` at 226 and grouped reduction at 441. This is not a pairwise dependency solve. |
| C2 | [nccl-mesh.c:4948](../rdma/nccl-mesh.c#L4948) | Every standalone group emits waits through each channel's final landing, at 4981. Spin dispatches can overlap one another; subsequent ordered work still observes a whole-group join. |
| C3 | [nccl-mesh.c:5302](../rdma/nccl-mesh.c#L5302), [5415](../rdma/nccl-mesh.c#L5415) | Standalone non-WRITE groups declare whole send/receive operand boundaries. WRITE landing ranges are declared after constructing their wait group. The common graph lacks the full inner command graph. |
| C4 | [metal-recording.m:1495](../rdma/metal-recording.m#L1495) | `MetalScheduleCrossWork` represents a collective as one work item, then flushes during preparation to obtain publication operands. This limits whole-function scheduling visibility even though the flush itself is not runtime planning. |
| C5 | [nccl-mesh-kernels.h:219](../rdma/nccl-mesh-kernels.h#L219), [mesh-graph.h:143](../rdma/mesh-graph.h#L143) | The surviving reduction repeatedly scans runs, observes prerequisites, atomically claims a ready block and advances done prefixes. Constant run/sum/cell descriptors are also dereferenced during execution. This is a runtime work selector, despite prepared tables. |
| C6 | [mesh-graph.h:165](../rdma/mesh-graph.h#L165) | Completion is exposed as a contiguous prefix: later finished blocks can remain unavailable until earlier blocks finish. Prefix advancement scans states and uses compare-exchange. |
| C7 | [nccl-mesh-kernels.h:103](../rdma/nccl-mesh-kernels.h#L103), [mesh.metal:27](../rdma/mesh.metal#L27) | Landing/source readiness helpers poll counts, check failure/bounds, fence and broadcast readiness within a threadgroup. A fused caller inherits these operations when it uses the helpers. Fused does not mean wait-free. |
| C8 | [nccl-mesh-metal.m:447](../rdma/nccl-mesh-metal.m#L447) | Reduction worker groups are limited to `min(blocks, cores)` and dynamically claim work. This is an execution strategy, not a mathematical necessity for page operations. The concrete resident-wait design motivates its occupancy cap. |
| C9 | [mesh-link.c:114](../rdma/mesh-link.c#L114), [135](../rdma/mesh-link.c#L135) | Bridge posting walks linked plan entries and checks their `after` words at runtime. Static entry identity/dependency structure and live completion values are mixed in the posting loop. |
| C10 | [mesh-link.c:225](../rdma/mesh-link.c#L225) | `MESH_PASS_UNITS=64` sets a literal per-pass posting budget. SEND and receive loops choose a lane, inspect one ordered cursor, then break on queue fullness, missing entry, unmet predecessor, producer prefix or credit capacity. Later independent work on that cursor cannot pass it. Hardware queue capacity does not justify the hardcoded software budget. |
| C11 | [mesh-link.c:145](../rdma/mesh-link.c#L145), [293](../rdma/mesh-link.c#L293) | Lane completion exposes ordered prefixes; one-by-one posting and completion processing impose an explicit transport order. The driver requires matching SEND/RECV order on a QP, but cross-QP independence must not be discarded. |
| C12 | [mesh-link.c:436](../rdma/mesh-link.c#L436), [475](../rdma/mesh-link.c#L475) | Progress threads park while administratively guarded or without served lanes; port changes wait for the thread's acknowledgement. These are bridge lifecycle gates, not per-tensor numerical barriers. They are included rather than concealed as non-runtime code. |
| C13 | [nccl-mesh.c:216](../rdma/nccl-mesh.c#L216) | Collective construction uses global group/attach locks; communicator initialization/configuration uses guard/prepare locks. Ordinary prepared replay does not reenter `launch_metal`. The background watchdog separately holds the live registry lock while reading communicator failures. These phases must not be conflated. |

Additional exposed administrative/observation waits: `mesh-dataflow.c:88/212` block while map/health snapshots are being written; `:249` implements explicit map-change waiting. `mesh-link.c:1540` parks a client-holder process on bridge control; `:1510` waits for its child, and shutdown at 1596/2004/2022 waits for quiet windows or child exits. These do not execute inside the prepared tensor graph, but remain blocking APIs in the surrounding system. The map/health interfaces must not be invoked as numerical runtime inputs that hide these waits. ISL allocator locks at `mesh-isl-memory.c:51/90` belong to preparation only.

### Literal collective, graph and bridge fence sites

`T` = threadgroup barrier; `F` = explicit atomic fence. Macro/string sites emit GPU instructions, including multiple instances and loop iterations from one source line.

| Source | All located literal sites | Scope |
|---|---|---|
| [nccl-mesh-kernels.h](../rdma/nccl-mesh-kernels.h#L99) | T: 106,109,112,123,223 (twice),232,237. F: 99,107,124,225,233,238. | Error publication, landing acquisition, source release and surviving grouped reduction. |
| [mesh-graph.h](../rdma/mesh-graph.h#L161) | F: 161,174. | Every readiness-search iteration; repeated prefix publication. |
| [mesh.metal](../rdma/mesh.metal#L32) | F: 32,67. | Polling/acquisition and publication, system scope. |
| [mesh-link.c](../rdma/mesh-link.c#L365) | F: 365,672,718; `__sync_synchronize`: 1906. | Completion acquisition, link/control publication and initial bridge-header publication. |
| [mesh.h](../rdma/mesh.h#L230) | F: 230,239. | Notification release and map snapshot acquisition. |
| [mesh-stat.c](../rdma/mesh-stat.c#L24) | F: 24. | Statistics snapshot acquisition. |

The old pool's barriers, `track_op_of`, `metal_after`, `metal_tracks` and ordered `retire` loop were present in the first snapshot and are deleted in the final reviewed revision. They are not current findings. `metal_wait` remains an exported host event-wait helper at `nccl-mesh-metal.m:139`; no current caller was located.

## 4. Shared recorder, backend and application ordering

The following source inventory was verified at `f82f0b15`; its listed files did not change in the four collective-removal commits through `99b05c3c`.

### Shared recorder: preparation emits these runtime barriers

All paths below are in [metal-recording.m](/Users/mdot/dox/mesh/rdma/metal-recording.m).

| Site | Runtime effect and placement condition |
|---|---|
| [951](../rdma/metal-recording.m#L951) | `setBarrier` on an indirect compute command when preparation's `ordered[t*count+i]` bit is set. These are compiled barriers, despite the API call itself happening during preparation. |
| [1462](../rdma/metal-recording.m#L1462) | `MetalScheduleComputeNow`: Dispatch/Blit → Dispatch/Blit encoder barrier, device visibility, when `MetalEncoderAccesses` reports a prior conflict. |
| [1574](../rdma/metal-recording.m#L1574) | Encoded `MetalScheduleRangeNow` replay: the same stage-wide barrier for a conflicting command. |
| [1580](../rdma/metal-recording.m#L1580) | Indirect replay: flush pending ICB span, then broad encoder barrier when an external conflict is not already covered by the command's recorded barrier, or occurs at the pending span start. |
| [1610](../rdma/metal-recording.m#L1610) | `MetalScheduleCopy`: broad encoder barrier when `MetalEncoderOrder` reports a source/destination access conflict. |
| [1786](../rdma/metal-recording.m#L1786) | `MetalScheduleVisible`: Dispatch/Blit queue-stage → Dispatch/Blit encoder-stage barrier with device visibility. At most once per part; called when a preceding part or earlier function has a conflicting access. This orders more work than an individual page store. |

The frontier solver at [787](../rdma/metal-recording.m#L787)–804 gives each command a phase; a prior conflict can advance the whole phase and erase its accumulated access frontier. Thus dependency analysis is already performed during preparation, but its output includes broad phase barriers. This is different from a runtime range scan, and different again from a barrier-free output program.

### Shared recorder: runtime gates and other ordering

| Site | Mechanism and scope |
|---|---|
| [metal-recording.m:2127](../rdma/metal-recording.m#L2127) | `MetalScheduleEnqueue` emits every compiled `queue waitForEvent:value:` before committing the part. Queue-side runtime waits, not CPU waits. It commits one command buffer per part, in the compiled linked order, then signals its part event. |
| [metal-recording.m:1735](../rdma/metal-recording.m#L1735) | `MetalSchedulePublication` closes the producer's whole GPU command-buffer part to provide a host-consumable event. Host dependency publication is therefore a part boundary, not necessarily the individual needed page. |
| [metal-recording.m:1764](../rdma/metal-recording.m#L1764) | Each host task stores a remaining-event count. Shared-event listeners at 1772 decrement it; only the final prerequisite callback calls `task.submit` at 1776. Each actual GPU predecessor contributes the whole-part event above; an otherwise independent host node waits on the function start event. Explicit `after` events are also added. |
| [metal-recording.m:1882](../rdma/metal-recording.m#L1882) | Host event listeners use the global user-initiated dispatch queue. This is not a one-thread worker queue, but host execution still begins through framework notification and the final-predecessor branch. |
| [metal-recording.m:1714](../rdma/metal-recording.m#L1714) | Preparation imposes a predecessor on the use `capacity` positions earlier for each declared shared host resource. This is a resource-capacity serialization edge, even without a page conflict. Capacity comes from provider resources. |
| [metal-recording.m:1388](../rdma/metal-recording.m#L1388) | `MetalScheduleAfter` records an external event/value in the subsequent scope. It can make broad framework/whole-function completion a prerequisite where a caller supplies that event. |
| [metal-recording.m:2017](../rdma/metal-recording.m#L2017) | Preparation creates one flat linked submission order, preferring ready parts with fewer new waits. Runtime traverses that order on one supplied queue. Once a queue wait is inserted, later independent submissions on that queue are also behind it. |
| [metal-recording.m:2077](../rdma/metal-recording.m#L2077) | `MetalScheduleEnd` directly blocks the host until the root arrival flag: `os_sync_wait_on_address_with_timeout` at 2083, in a loop with a one-second error-observation timeout. Public `MetalScheduleStatus` separately provides nonblocking observation. |
| [_stream.mm:1999](../rdma/torch-mesh/torch_mesh/_stream.mm#L1999) | `completion_end` releases the Python GIL and calls the blocking `MetalScheduleEnd` for each schedule in the observed tuple. It is not intrinsically a nonblocking read of an already-available timestamp. |

`MetalScheduleFlush` and `MetalScheduleOrder` themselves run during preparation. They should not be described as a runtime JIT scheduler. The emitted event waits, completion counters and barriers remain runtime behavior. GPU-only replay and mixed host/provider execution both use this recorder.

### Framework and provider execution

| Site | Mechanism and classification |
|---|---|
| [_stream.mm:205](../rdma/torch-mesh/torch_mesh/_stream.mm#L205) | `host_wait` synchronously enters PyTorch's MPS serial queue. Its helper at 174 can choose `COMMIT_AND_WAIT` or `COMMIT`. Current callers found pass `false`, so the live call is host serialization/command-buffer commit, not a demonstrated GPU completion wait. |
| [_stream.mm:214](../rdma/torch-mesh/torch_mesh/_stream.mm#L214) | `commit` visits every registered MPS stream and performs that synchronous queue entry. Called at preparation/compile boundaries. |
| [_stream.mm:224](../rdma/torch-mesh/torch_mesh/_stream.mm#L224) | `publish_boundary` synchronously enters the current MPS queue and commits its prior work; configuration/migration boundary. |
| [_stream.mm:450](../rdma/torch-mesh/torch_mesh/_stream.mm#L450) | `schedule_after_inputs` synchronously enters the current MPS queue, commits the stream, and at 451 supplies its progress event to `MetalScheduleAfter`. Preparation blocks on CPU queue dispatch and emits a runtime prerequisite for the entire prior MPS stream, rather than just the declared input pages. |
| [_stream.mm:170](../rdma/torch-mesh/torch_mesh/_stream.mm#L170) | Framework event notification uses the per-stream `rising` mutex before publishing monotonically increasing progress. This lock executes in completion callbacks. Stream registry locks at 185/195 serialize registry creation/enumeration. |
| [allocator.mm:195](../rdma/torch-mesh/torch_mesh/allocator.mm#L195) | `waitForEvents` synchronizes each owned buffer's MPS event; delegates non-owned buffers to Torch's `waitForEvents` at 197. Host-read/lifetime API, not the prepared enqueue loop. |
| [allocator.mm:70](../rdma/torch-mesh/torch_mesh/allocator.mm#L70) | Buffer release records an MPS event, arranges freeing on completion and calls MPS `COMMIT` when off its queue. Tensor destruction can therefore perform framework-queue work. |
| [allocator.mm:39](../rdma/torch-mesh/torch_mesh/allocator.mm#L39) | Lease-map mutex in `with_lease`; also acquisition/release/accounting lock sites at 58,85,93,108,221. These are storage/lifetime and host-access serialization, not barriers inside a numerical kernel. |
| [partition.py:902](../rdma/torch-mesh/torch_mesh/partition.py#L902) | The redistribution adapter calls `local.wait()` after each placement-transform step when `async_op` is false and the result is `AsyncCollectiveTensor`. The preceding private DTensor helpers at 885–900 are framework collective calls. |
| [mesh-coreml.m:325](../rdma/mesh-coreml.m#L325) | Runtime `cpu_submit` invokes every bound CPU instruction serially in array order, then completes the entire host task. The common recorder cannot overlap independent instructions hidden inside that one task. |
| [mesh-coreml.m:669](../rdma/mesh-coreml.m#L669) | Direct ANE submission invokes the configured `evaluateWithModel:options:request:qos:error:` entry. Completion is the prepared request callback installed at 685. There is no wrapper semaphore around inference in this file. The call's internal blocking/queue behavior is private-provider implementation and cannot be proven absent from this wrapper. |
| [mesh-coreml.m:530](../rdma/mesh-coreml.m#L530) | ANE request capacity is appended to the host-resource operands; it becomes the capacity predecessor relation at recorder 1714. |

Public Core ML is loaded for metadata/placement but `mesh_coreml_bind` at 739 rejects advisory output storage; current prepared page execution reaches direct ANE binding. Do not invent a second live public-Core-ML inference wait here. `_stream`'s `Drained` guard at 2014 only pushes/pops an Objective-C autorelease pool; it does not drain GPU work despite its name.

### Microbench: service and engine runtime ordering

| Site | Mechanism and scope |
|---|---|
| [metal_runtime.swift:409](../../../metal-microbench/metal_runtime.swift#L409) | One shared engine Metal 4 command queue. Every engine schedule is submitted to it in prepared/submitted order. |
| [metal_runtime.swift:426](../../../metal-microbench/metal_runtime.swift#L426) | `submitAndWait` directly waits on a fresh completion semaphore. Current located call is weight `formatRows` at weights.swift:1026, a preparation/conversion helper. |
| [metal_runtime.swift:434](../../../metal-microbench/metal_runtime.swift#L434) | `drainQueue` signals an event behind all previously enqueued GPU work and waits with `UInt64.max`. Current callers are EngineProgram retirement (engine_recording.swift:328) and KV-pool resizing/retirement (weights.swift:929). |
| [lm_engine.swift:1852](../../../metal-microbench/lm_engine.swift#L1852) | `submitStep` refuses another batch while `inFlight`. Set at 1627 before enqueue and cleared only by whole-batch completion at 1645. One batch in flight, independent of partial-page readiness. |
| [lm_engine.swift:1622](../../../metal-microbench/lm_engine.swift#L1622) | `whenIdle` defers host actions until whole-batch completion. `finishBatch` at 1646–1649 drains them serially. |
| [ffi_batch.swift:437](../../../metal-microbench/ffi_batch.swift#L437) | `gemma_poll` holds the global `gEngineLock` across host planning/submission and result preparation. `completeEngineBatch` at 42–45 uses the same lock for completion/session updates. |
| [ffi_batch.swift:497](../../../metal-microbench/ffi_batch.swift#L497) | Host poll waits on `gIntakeCond` when no work/update can be issued. Engine lock is released during this condition wait, then reacquired. Admissions and session closure are also held until `!engine.inFlight` at 452/479. |
| [rank.swift:455](../../../metal-microbench/rank.swift#L455) | `Rank.take` is a FIFO whole-call admission gate. Its condition wait at 459 continues until no current call and this caller is first. `release` happens after the call/output lifecycle. One admitted collective call per rank service. |
| [rank.swift:748](../../../metal-microbench/rank.swift#L748) | Call teardown waits until all revoke operations have finished before detaching the mesh. |
| [engine_recording.swift:632](../../../metal-microbench/engine_recording.swift#L632) | Every recording receives `MeshRankAfter(gVisionEvent, tickets.max())`: all vision encodes are prerequisites of the recording, even before an individual first consumer would need its particular image pages. MeshRank preparation forwards this to `MetalScheduleAfter`. |
| [engine_recording.swift:303](../../../metal-microbench/engine_recording.swift#L303) | `EncodeEnd.error` blocks on its completion semaphore and signals it again for subsequent readers. `EngineProgram.end` at 552 evaluates every encode's `.error`. It occurs at call completion, not merely inside image preparation. |
| [ffi.swift:254](../../../metal-microbench/ffi.swift#L254) | Vision tower completion callback constructs and submits a separate padding/copy schedule. Hence copies cannot be submitted before the whole tower completes, even if individual outputs are available. |
| [ffi.swift:278](../../../metal-microbench/ffi.swift#L278) | Only inside that tower completion callback does `_kickVisionDispatch` start the next queued vision batch. `gVisionBatchInFlight` admission at 361–366 permits one tower batch in flight. Padding may overlap next tower subject to the shared queue's dependencies. |
| [ffi.swift:199](../../../metal-microbench/ffi.swift#L199) | `gemma_vision_fetch_softs_by_key` creates and waits on a semaphore if the cached soft-token event ticket has not arrived, then CPU-copies the result out. This is a public synchronous host fetch API. |
| [ffi.swift:48](../../../metal-microbench/ffi.swift#L48) | Vision event notifications run on one explicitly serial `vision.listener` dispatch queue. |
| [drafter.swift:865](../../../metal-microbench/drafter.swift#L865) | `SaguaroSpeculator` has one serial dispatch worker for all fan-out work; inside each job its slots are processed in a loop. |
| [drafter.swift:901](../../../metal-microbench/drafter.swift#L901) | `collect(round:)` releases its map mutex, blocks on a semaphore, reacquires the mutex and repeats until that exact round exists. It is an execution-time host dependency wait. |
| [vision_tower.swift:377](../../../metal-microbench/vision_tower.swift#L377) | A shared scratch lock covers schedule/command construction using shared tower scratch. This serializes preparations; it is released before asynchronous GPU completion. |
| [harness.swift:114](../../../metal-microbench/harness.swift#L114) | The current draft-data generation harness's local `run` builds a schedule, enqueues and waits on a semaphore on every call inside its generation loop. This is not the hourly runner; it is an owned active data-generation program. |

The ordinary recorded rank entry [mesh_rank.m:1369](../../../metal-microbench/mesh_rank.m#L1369) sets a submitted field and calls `MetalScheduleEnqueue`; it has no additional per-step semaphore there. LiteRT records native/provider commands into the same shared recorder (`litert_program.m:2802` onward). No direct LiteRT inference wait was found in the owned wrapper. That is not evidence that vendor compilation/inference internals contain no waits.

### Preparation-only blocking and registry locks

These still serialize work, but should not be misreported as unconditional per-transfer execution waits.

* [metal-recording.m:1152](../rdma/metal-recording.m#L1152): ICB optimization commits a preparation GPU job and waits on its event until completion (`UINT64_MAX`).
* [_stream.mm:539](../rdma/torch-mesh/torch_mesh/_stream.mm#L539), [613](../rdma/torch-mesh/torch_mesh/_stream.mm#L613): semaphore waits for Metal library/pipeline compilation, bounded by `MESH_PREPARATION_NS`. Pipeline/cache locks at 505,517,577,589,607; fused preparation cache locks at 1367,1429,1499. The fused maker releases its mutex while acquiring the Python GIL/calling Python at 1345 and then reacquires it.
* [mesh-coreml.m:67](../rdma/mesh-coreml.m#L67): one class monitor encloses CPU compilation/cache lookup, compiler spawn and `waitpid`; serializes CPU source compilation. Resource registries also use monitors at 124/159.
* [mesh-coreml.m:379](../rdma/mesh-coreml.m#L379): bounded semaphore wait for `MLComputePlan.loadContentsOfURL`. [585](../rdma/mesh-coreml.m#L585): model class monitor encloses loading/placement; includes synchronous `MLModel modelWithContentsOfURL`. Model retain/release/contract registry monitors at 629,729,735,749,758. `mesh_coreml_compile` at 765 invokes synchronous public Core ML compilation.
* [mesh_rank.m:269](../../../metal-microbench/mesh_rank.m#L269), [295](../../../metal-microbench/mesh_rank.m#L295), [1109](../../../metal-microbench/mesh_rank.m#L1109): readiness polling with `usleep(1000)` while communicator initialization is in progress. Planning then performs configuration gathers around 1117. Group monitors: 107,248,276,286,306. Kept-communicator registry mutex: 191–227.
* [weights.swift:969](../../../metal-microbench/weights.swift#L969): staging-slot conversion semaphore, before reusing each of one/two conversion slots and at final completion. `formatRows` submits each band at 983 and signals completion at 985. At 978 `DispatchQueue.concurrentPerform` synchronously joins CPU staging copies. This is model preparation, not the compiled numerical graph; a caller that invokes conversion during processing inherits its blocking behavior.
* [common.swift:301](../../../metal-microbench/common.swift#L301): `concurrentPerform` joins all specialized pipeline compilations. Kernel source/pipeline/archive registry locks appear at 24,29,31,42,51,57,78,81,87,94,119,134,159,175,177,189,224,228,257,272. These belong to preparation/cache interfaces.
* Additional preparation registries: [metal_native.m:19](../../../metal-microbench/metal_native.m#L19) capability cache; [model_file.swift:26](../../../metal-microbench/model_file.swift#L26) model parsing; [instance_fit.swift:192](../../../metal-microbench/instance_fit.swift#L192) instance loading; [bootstrap.swift:551](../../../metal-microbench/bootstrap.swift#L551) configuration loading; [neural_engine.swift:144](../../../metal-microbench/neural_engine.swift#L144) model registry; [streamed_weights.swift:307](../../../metal-microbench/streamed_weights.swift#L307),415,432,439,446 selection/compilation cache. `Generation` installation/current/cache operations in metal_runtime.swift:439–530 are mutex-protected.
* [rank.swift:448](../../../metal-microbench/rank.swift#L448): administrative child process `waitUntilExit`. Per-preparation locks are created at 525–530. The server's administrative child wait is not a numerical-kernel dependency.
* [litert_program.m:2075](../../../metal-microbench/litert_program.m#L2075): synchronous `LiteRtCreateCompiledModel` call during preparation. Owned wrapper exports captured provider commands to the common recorder; vendor internals were not scanned.

Other runtime registry/state mutexes include FFI failed-ticket/cache/residency/batch state (ffi.swift:29,34,39,221,228,239,288,341,361), the vision ticket counter (bootstrap.swift:12), and Saguaro slot/tally state (drafter.swift:889 onward). These locks do not make a CPU wait for the GPU by themselves; they serialize their critical sections.

### Literal owned shader barrier and fence emission sites

`T` means `threadgroup_barrier`; `S` means `simdgroup_barrier`; `F` means `atomic_thread_fence`. These execute on the GPU. The source emitter may have run during preparation. All sites are included regardless of whether they protect a real reduction, shared-memory tile reuse, publication, or a broader ordering choice. Shared helper instantiations can introduce fences into files with no literal fence token.

Mesh paths below are under `/Users/mdot/dox/mesh/rdma/torch-mesh/torch_mesh/`.

| File | Literal sites | Scope/conditions |
|---|---|---|
| [chain.metal](../rdma/torch-mesh/torch_mesh/chain.metal#L55) | T: 55,63,141,171. F: 142. | Stage/reduction cooperation; phase-zero input/previous-result boundary; publication/last-arrival boundary. F142 is system seq_cst. |
| [drafter.metal](../rdma/torch-mesh/torch_mesh/drafter.metal#L83) | T: 83,94,102. | Cooperative draft kernel stages. |
| [topk.metal](../rdma/torch-mesh/torch_mesh/topk.metal#L36) | T: 36. | Cooperative selection stages. |
| [structured.metal](../rdma/torch-mesh/torch_mesh/structured.metal#L31) | T: 31,39. | Shared-memory reduction/stage loop. |
| [speculate.metal](../rdma/torch-mesh/torch_mesh/speculate.metal#L66) | T: 66,81,86. | Device/threadgroup data publication and cooperative scan stages. |
| [attention.metal](../rdma/torch-mesh/torch_mesh/attention.metal#L42) | T: 42,53,70,73,176,204,208,311,335,360,464. | Tile staging and reductions; 70 conditional on `!Fresh || r0>0`; 204 on `Shared && end<J`; others follow their kernel/loop specialization. |
| [argmax.metal](../rdma/torch-mesh/torch_mesh/argmax.metal#L62) | F: 62,63. T: 70,74,97,103,105,169,176,193,198,210,232. | F62 system or F63 device selected by `System`; helper `mesh_last_arrival` fences before counting and after winning. T74 only if a shared `last` word is supplied. Other sites are reduction/bin/selection cooperation. |
| [streamed.metal](../rdma/torch-mesh/torch_mesh/streamed.metal#L230) | T: 230,232,571,654,736,949,956,1034,1038,1051,1236,1259,1288,1389,1391,1451,1453. | Shared tile staging/reduction in streamed matrix/attention functions; 230 conditional on `!Fresh`; loop sites repeat at their staged tile cadence. |
| [linear.metal](../rdma/torch-mesh/torch_mesh/linear.metal#L77) | T: 77,84,147. | Cooperative multiply staging and fused-publication boundary. |
| [partition.py](../rdma/torch-mesh/torch_mesh/partition.py#L245) (generated Metal) | T: 245,263. F: 264. | Fused producer publication and landing-consumption shader; F264 system seq_cst. |
| [dnn.py](../rdma/torch-mesh/torch_mesh/dnn.py#L43) (generated Metal) | T: 43. F: 44. | Input landing acquisition before generated addition, system seq_cst. |

Microbench paths below are under `/Users/mdot/metal-microbench/`.

| File | Literal sites | Scope/conditions |
|---|---|---|
| [litert_native.metal](../../../metal-microbench/litert_native.metal#L222) | T: 222. | Device-memory barrier in native provider/speculative helper. |
| [metal/rows.metal](../../../metal-microbench/metal/rows.metal#L45) | T: 45,124,291,294,319. | Row cooperation/merging; 45 conditional on `n`, 124 conditional on `k`; latter three use device memory scope. |
| [vision_tower.swift](../../../metal-microbench/vision_tower.swift#L491) (generated Metal) | T: 491,500,510,543,556. | Shared vision-attention tile/reduction stages. |
| [metal/kernels.metal](../../../metal-microbench/metal/kernels.metal#L137) | T: 137,155,244,246,573,582. | Cooperative tile/reduction stages. |
| [metal/layers.metal](../../../metal-microbench/metal/layers.metal#L85) | T: 85. F: 86. | Device barrier then system seq_cst publication/consumption boundary. |
| [metal/numerical.metal](../../../metal-microbench/metal/numerical.metal#L349) | T: 349. | Threadgroup-memory cooperation inside the numerical kernel loop. |
| [metal/attention.metal](../../../metal-microbench/metal/attention.metal#L81) | S: 81. F: 82,439,521. T: 210,295,438,520. | S81/F82 in access acquisition; T295 only before another channel batch. T438/F439 and T520/F521 device/system publication/consumption boundaries. |

### Shader waits and synchronization inherited through shared helpers

These calls inherit the synchronization in section 3; counting only literal barriers in the call-site files misses it:

* `chain.metal`: `mesh_step_enter` 103,131,137,170; `mesh_step_counts` 132,138; `mesh_last_arrival<true>` 172; releases at 104/174. Phase zero acquires all named input channels, and (unless first) all named prior-result channels, before the phase's data loop.
* `linear.metal`: `mesh_step_enter` 143, `mesh_last_arrival<true>` 148, release 150.
* Generated `partition.py`: enter 232/260; last arrival 246; release 248; counts 261.
* Generated `dnn.py`: enter 40 and counts 41.
* `matrix.metal:49`: `mesh_last_arrival<true>`; `argmax.metal:111`: `mesh_last_arrival`. The helper itself has a fetch-add last-arriver election and conditional return, even though it does not poll/spin waiting for other groups.
* Micro `metal/kernels.metal:76,77`: `mesh_landing_await` for key and value pages.
* Micro `metal/attention.metal:140`: `access.await`; 329: `mesh_landing_await` when `RECEIVED`; 288/451: `mesh_last_arrival` conditional on nonzero partials.
* Micro `metal/rows.metal:43`: `mesh_last_arrival` for holder/share contributions; non-final groups continue without doing the final row work.
* Shared `rdma/mesh-graph.h:161,174` contains system/device seq_cst fences; listed in section 3.

The source does contain both compiled ordering and runtime completion/data-publication mechanisms. Removing a runtime access scan does not by itself remove the broader barriers, event gates, whole-batch limits, or callback-controlled submission boundaries listed here. Conversely, cooperative threadgroup synchronization and event-based visibility should not be called equivalent to a whole-collective barrier without examining their actual scope.

## 5. What can be proved about the required runtime

There is no need for a second scheduler object model to express these bounded tensor programs. Page operands, masked operations and compiled loops can express their numerical work and dependency state. That expressiveness claim is distinct from claiming that every runtime observation can be erased, or that any legal schedule is the fastest schedule.

### A. Compile the known relations once

Let `V` be the operation instances of one declared execution, including iteration coordinates. Let `R(v,a)` and `W(v,a)` describe reads and writes, where `a` names a page version and its selected lanes. Using only page identity would incorrectly order disjoint masks; using only an address without a version would incorrectly merge successive values.

For uniquely defined values, the producer→consumer relation is:

```
E_RAW = { (u,v) : exists a, W(u,a) and R(v,a) } = W ; inverse(R)
```

Add required reuse edges for physical pages, the declared ordering of noncommuting reductions/writes, and the constraints of the chosen backend/resource assignment. For a legal acyclic instance graph, choose a topological order or several command streams satisfying these edges. For loop domains, derive a legal loop schedule over those relations rather than discovering the graph on each iteration.

**Proof of required-order and result equivalence.** Start with the declared input page versions. Induct over any legal execution of `E`. Each operation's required predecessors have produced exactly its input versions; the bound operation therefore produces the same output versions. Two incomparable operations with commuting effects may exchange order. Thus any legal execution preserves the required dependent-access order and result. Floating-point reassociation requires permission in the numerical contract; it is not silently covered by the commuting assumption.

This construction needs no runtime overlap scan, signature lookup, implementation choice, dependency reconstruction or range rejection. It does not call for a global layer barrier. Resource assignment is compiler input/output; physical readiness and memory visibility must implement the emitted dependencies.

Computed page indices stay ordinary page operands with their producing edges. A top-k result does not require another kind of executor or pointer resolver. Where computed indices may collide, the program must declare the collision/reduction semantics; fixed index-array width alone is not a proof of disjointness. Quasi-affine relations can be solved algebraically; nonaffine index *values* are handled by the declared indexed operations, not guessed by the compiler.

### B. Masked loops are sufficient to express a bounded scheduler too

For a deterministic scheduler with finite state and bounded work, write its transition as:

```
(state[t+1], selected[t]) = F(state[t], readiness[t])
```

`state`, `readiness` and `selected` can all occupy ordinary pages. Finite-state transitions are Boolean functions and can be implemented by comparisons, arithmetic and masks. Sparse dependency incidence and eligibility can likewise be expressed as masked operations over declared arrays. A compiled loop over these transitions can reproduce the scheduler's selected payload operations.

**Proof.** Give both executions the same initial state, readiness observations and tie-breaking inputs. If states agree at iteration `t`, evaluation of the same `F` gives identical selections and next states. Induction gives identical selected payload-access order. A false mask must suppress the access itself, not load unavailable data and mask the result afterward. If the original scheduler has nondeterministic atomic winners, those choices must be fixed or included in the observations to require an identical trace.

This proves that a separate runtime scheduling API is not necessary for expressiveness. It does **not** prove that translating the old polling loop verbatim removes its cost. Eligibility recomputation and repeated readiness reads still run if `F` contains them. Compiler-known parts must be evaluated during preparation; only genuine operand values remain. Arbitrarily delayed arrivals also mean that bounded numerical work does not imply a finite fixed bound on polling rounds.

### C. Exact adaptive trace without its inputs is impossible

Suppose independent operations A and B share one execution unit. A consumes remote page P; B consumes Q. In execution X, P arrives first and an arrival-driven scheduler runs A then B. In execution Y, Q arrives first and it runs B then A. The compiled declarations and eventual payloads are the same.

A fixed program observing neither arrival order nor any operand encoding it must choose the same first access in both executions. It cannot match both traces. Likewise, with unbounded arrival delay, a fixed unconditional load time cannot guarantee the page has arrived.

This is not permission for the existing polling pool or whole-group joins. The target is the **required partial order**, not the accidental total order of independent work in a superseded scheduler. A compiled dependency can be enacted by the backend's available causal mechanism without re-solving the graph. If exact adaptive trace equality is required, its observations must remain inputs, as in B.

An actual RAW dependency also requires visibility of the producer's stores. ISL's relation proof alone does not prove that a particular Metal fence is redundant. Each fence or barrier in the inventory requires a narrower dependency implementation or a backend visibility argument before erasure. Being fixed at compile time gives an ordering no authority beyond that relationship.

### D. What ISL currently does in this source

[mesh_relation.py](../rdma/mesh_relation.py#L1) is preparation-only relation algebra: composition, inverse, intersection, projection and related operations. [mesh-index-relation.m:508](../rdma/mesh-index-relation.m#L508) uses ISL AST construction to emit index expressions. A search of first-party RDMA C, Objective-C and Python found no calls to ISL scheduling/flow-analysis entry points (`isl_schedule*`, `isl_union_access_info*`, `compute_flow`). The owned command scheduler is still `MetalWorkDependencies`/`MetalScheduleOrder` plus the separate collective construction.

ISL is compatible because these static tensor/loop access relations fit its integer-set algebra. Its presence does not mean inter-command dependency scheduling has been integrated. Its scheduler consumes supplied validity and other relations; it cannot supply semantic edges that the lowering omitted or predict remote arrival order. See the [ISL scheduling manual](https://libisl.sourceforge.io/user.html#Scheduling).

## 6. Remaining compiler responsibilities exposed by this review

1. Export the collective's publications, copies, reductions and exact consumer accesses into the same graph as numerical commands before any encoder flush. A whole collective callback is too coarse.
2. Complete the ordinary domains now rejected by `zabbadoo_plan`: frame coordinates, unequal landing partitions, COPY→reduction edges and cross-call value versions. General relation composition should replace specialized admission lists. Invalid graphs and missing declared capability operands remain preparation errors; supported tensor structures must not be refused because their lowering is unfinished.
3. Resolve static work identity, dependencies, resource assignment and loop structure during preparation. The surviving reduction's scan/claim/prefix machinery must not be mistaken for completion of this requirement merely because the older pool was deleted.
4. Replace broad encoder/whole-group/whole-part ordering with the actual producer→consumer relations. Carry backend visibility and contiguity requirements as operands through the same lowering. No duplicate direct/fused dependency implementation is entitled to weaker coverage.
5. Remove competing eager/imperative frontend and service execution contracts by integrating their ordinary source into the prepared function interface. Complete required derivative, layout and placement domains rather than keeping replication or rejection as a substitute.
6. Make batch admission, vision output consumption, speculative operations and CPU instruction dependencies visible at their true granularity. Their broad host gates currently hide parallel work from an otherwise prepared compiler.

No fallback's existence grants permission to keep it. No deletion establishes coverage by itself. This review identifies implementation obligations; it neither adds another executor nor substitutes a counted test suite for whole-function performance.

## 7. Coverage and unresolved evidence

Reviewed first-party mesh `rdma` compiler, collective, bridge and Torch sources; microbench native/LiteRT/provider integration, engine/service execution and owned numerical shaders, including embedded generated shader text. Literal barrier/fence sites and their shared helper calls are listed above. Preparation-only locks/waits are distinguished from emitted runtime operations and from completion observation. Administrative bridge gates are included with that role stated. The pre-existing NVMe KV offload subsystem was not reviewed, as instructed.

Provider/driver binaries, vendor-generated kernels and compiler-generated machine instructions are opaque to this source inventory. Historical probes, unrelated tools and administrative server I/O are outside the compile→numerical-runtime scope. The inventory therefore does not claim to prove absence of all hidden serialization inside external implementations. Function-internal numerical recurrences and reduction arithmetic also have their own required order; the serialization findings concern imposed execution/dependency boundaries, not the mere existence of a mathematical loop.

The quoted cadence defect—requesting 32 updates reacquires sources 32 times—was not located in either repository's inspected compiler/runtime source. No function or commit is attributed to that quotation here. It remains an explicit evidence gap, rather than being silently equated with `realize(repetitions)` or another superficially similar loop.
