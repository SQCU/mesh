# Existing graphs, native replay and the PyTorch frontend

Recorded: **2026-10-25T16:50:37-07:00** (America/Los_Angeles).

Source reviewed: mesh `c4233336`; metal-microbench `8d508ea5`, both on `main`.
This is implementation history and a guide to code reuse. The short hardware and data contract belongs in
[data-model.md](data-model.md). Outstanding removals are recorded in the
[runtime interface manifest](runtime-interface-removal-2026-10-25.md).

## Why the graph comes before the call

The library already has several ways to describe a whole function before running its data. A Python caller can
record tile operations. A PyTorch caller can supply a function whose graph is traced without running the real
payload. A native caller can record the Metal commands its program emits. These are valuable existing pieces.
They tell us which results each step reads, what it writes, and which parts can be used early.

That information lets the compiler assign pages, split work among unequal nodes, choose paths through the
physical cables, and bind each write to its destination before execution. A relay can send from the page where
the data lands. A receiver need not answer a request, pick a target or copy an incoming payload into another
place. Repeated calls can use the same commands and page indices with new values.

“Ahead of time” here means **before the numerical call that uses the plan**. A new shape or new work shares may
need a new plan. The compiler may run in the same process as the application. The distinction is when its work
happens: tracing, loading, allocation, route search and binding must finish before data execution begins.
Compiling a shader on its first data use is still late preparation.

Our target nodes have substantial shared memory and useful interconnect capacity, while arithmetic capacity,
memory bandwidth and ports differ across nodes. Cheap DMA relays can use capacity on a node with a small compute
share. These properties reward exact partial-input dependencies and joint page placement. A host loop that
rediscovers a tensor's placement or joins a whole operand can waste the advantage even when its kernel is fast.
The hardware comparisons are in data-model sections 1–4; they are not a license to assume every operation is
limited by only one resource.

## What the existing pieces do

| Layer | Existing code | Role before execution | Remaining gap |
|---|---|---|---|
| Partial tensor declarations | [python/mesh](../python/mesh/__init__.py): `Partial`, `Tensor`, `Ref`, `BlockSpec`, `Program.kernel_call`, `realize` | Name tile regions, contributions, kernel calls and their dependencies | Realization now generates a fixed FX sequence of compiled numerical spans and bound native crossings. The worker executor is removed. |
| PyTorch graph declarations | [declarations.py](../rdma/torch-mesh/torch_mesh/declarations.py): `trace`, `lower`, `compiled_graph`, `Function`, `Preparations` | Trace a complete function with fake tensors; describe collective calls; bind indexed crossings | Function compilation is mandatory and crossings have prepared indices. FX still does not lower to the native Metal recorder. |
| Ragged placement | [partition.py](../rdma/torch-mesh/torch_mesh/partition.py), `RaggedShard`, `Preparations`; [configuration.py](../rdma/torch-mesh/torch_mesh/configuration.py) | Map uneven tensor shares to a device mesh and declare their communication | Generic collective launch still reconstructs metadata; some examples impose whole-operand joins. |
| Physical write compiler | [mesh-plan.h](../rdma/mesh-plan.h), `mesh_routed_compile`, span/row planners, [routing.py](../rdma/routing.py) | Turn logical data needs into many-cable routes, relay slots, packet cuts and ordering | Raw pointer/extent write contracts and generic registration cuts are not fully reconciled with the page/mask model. |
| Native numerical composition | [matrix_operations.swift](../../../metal-microbench/matrix_operations.swift): `MetalProgram`, `MetalDispatch`, `MatrixView`, `Reach` | Compose kernels, bind constants and describe accessed ranges | Some callers and profiling modes still change work during execution. |
| Native graph record/replay | [metal_recording.h](../../../metal-microbench/metal_recording.h): `MetalRecord`, `MetalReplay`, `MetalOperand`, `MetalReach` | Capture command bindings, prepare repeated slots, derive access order, retain residency | Shared prepared submission and byte-range access ordering replace the former host batch loop. |
| Native model adapters | [engine_recording.swift](../../../metal-microbench/engine_recording.swift), [litert_program.h](../../../metal-microbench/litert_program.h), [mesh_rank.h](../../../metal-microbench/mesh_rank.h) | Declare prefill/decode phases and their collective crossings using shared records | LiteRT now consumes phase parts and explicit handoff formats, then declares native replay. `MeshRankRun` still waits and allocates. |
| Feedback as data | [assignment.py](../rdma/assignment.py), [rollout.py](../rdma/rollout.py), [lifetimes.py](../rdma/lifetimes.py) | Use whole-call observations to emit the next shares, routes and storage relations | Existing capability is not proof of convergence or optimality. Some relevant metadata is not yet connected to the active solver. |

These layers are not interchangeable. FX describes operators and tensor metadata. The Metal recorder sees
commands and bound storage. The physical write compiler sees cables and page movements. The integration task
is to carry the useful information between them without creating a new runtime decision loop at each boundary.

## PyTorch can be the language without owning the schedule

The current `trace` uses `make_fx` with fake tensors. It records the supplied function and the backward work
that function includes. `Function.declare` retains parameter gradients as graph outputs, so an explicit backward
cannot be discarded merely because the normal forward result does not use it. It does not invent training for
a forward-only function.

`lower` reads operator, shape and group metadata. `compiled_graph` rewrites supported collective occurrences
to `indexed_step` or `indexed_step_out`. `Preparations` allocates occurrence slots and binds their operands
before execution. Thus the useful extension is already more than a Python call to an ordinary eager collective:
the frontend can supply enough information to remove repeated crossing discovery.

`Function.declare` invokes the shared compiler during configuration with the explicit FX graph. The compiler
publishes its final collective graph there, so later transformations cannot hide or reorder communication after
the transport plan has been made. The interface has no eager execution option.

PyTorch tensor notation, autograd and DTensor placement can therefore serve as an authoring interface for an
alternative compiler. They do not require a CUDA execution stack or a particular default `torch.compile`
backend. This repository already extends that interface with ragged shares, coded matrices, partial operations,
fused publication and indexed crossings. The compiler should lower those declarations to the same page and
command machinery used by native programs.

This describes an existing frontend and a required integration, not a claim that the whole FX graph already
lowers into `MetalReplay`. That complete connection was not found. Replacing a fused kernel with many eager
tensor operations is not that connection either: it can discard useful algorithm and publication structure.

## What the native recorder actually reuses

`MetalRecordingDevice` records commands emitted through its device and queues. `MetalOperand` describes a
buffer's repeated slots; invocation `t` binds the prepared slot. `MetalReach` supplies exact accesses that are
not fully described by the ordinary buffer arguments. `MetalReplayBuild` uses those records to retain storage
and construct the chosen command form before running the call.

The indirect form prebinds pipelines, buffers, offsets, threadgroup memory and dispatch geometry in ICBs.
The encoded form uses the same records for direct Metal 4 dispatches. The serial form handles a local program
without crossings. These forms represent different command-processor costs and capabilities, rather than
different collective implementations. [Kernel design](../../../metal-microbench/docs/kernels.md#edges-the-dependency-edge-between-commands)
documents why direct recorded dispatches were introduced: stage barriers can be cheaper than ICB drains.
Some captured pipelines do not support ICBs at all.

`MeshRankPrepare` now encodes the complete declared function. `MeshRankRun` enqueues it without a batch
admission loop. The completion observer reports results asynchronously. LiteRT uses the recording device's
pipeline reflection and supplies external tensor byte views during capture. Repeated slots are ordered by
actual byte overlap. These changes remove manifest R12–R14; performance remains an empirical property of
the completed function, paired with its useful-output rate.

The PyTorch extension is a separate adapter. Its `dispatch_sync` hands a host encoding closure to the MPS
stream queue and waits for that closure. Encoding sets pipeline/buffer bindings and dispatch geometry; it does
not mean that tensor values are shuttled back and forth, nor necessarily that GPU work has completed. The
historical fast chain also used this queue call. Repeating known bindings remains removable overhead, but
that fact alone does not identify the later regression's cause.

## Roles in project history

Commit identifiers below belong to the named repository. Historical files can be read with
`git show COMMIT:path`; their removal is not a reason to lose the algorithmic information they contain.

| Date | Repository and evidence | Role given to the machinery |
|---|---|---|
| 2026-04-19 | microbench `86298f18`, `e0367349`, `lm_engine.swift` | The Swift/Metal Gemma engine already had multi-session and batched decode work. This native backend predates the later PyTorch extension work. |
| 2026-09-13–14 | [operator requirements](collective-goals-2026-09-14.md) | Partial tensor producers and consumers must stream without waiting for a large unrelated operand. A Pallas-like interface describes the graph AOT; transport is zero-copy and async. |
| 2026-09-14 | mesh `57628982` | Deleted the then-current collective executor/compiler and callers. The later restoration identifies the deleted Python partial API explicitly; this was not a capability first invented in October. |
| 2026-09-16–18 | mesh `929abb80`, `f531113f`, `3d32355e` | Compiled publication indices, prebound invocation operands and inline publication programs removed forms of runtime reconstruction and slot admission. These are useful precedents for the present removal scope. |
| 2026-09-18 | microbench `33ce26ce`, `144ad848`, `e8a8b876` | Recorded upstream tensor kernels once, kept a prepared residency set, and prepared boundary operands for the complete token chain. `litert_program.m`'s followed history passes through this earlier upstream adapter. |
| 2026-09-18 | mesh `2b49f3c6` | Removed a numerical executor and worker queues. Reuse its prepared-operand ideas without restoring that execution framework. |
| 2026-09-19 | microbench `57ca2399`, `0f83cf36` | Derived native barriers from buffer accesses and compiled RDMA completion operands into numerical consumers. Data dependencies were compiler inputs, not an excuse for blanket host waits. |
| 2026-09-26 | microbench `2211303f`, `08052664`, `1f6aa208` | Applied the generic recorder to the native engine, declared decode crossings through it and added tensor-parallel prefill. |
| 2026-09-27 | microbench `32b24185` | `litert_program.m` replaced `e2b_replay` under the common rank driver. LiteRT was a distinct native frontend, not a PyTorch backend. |
| 2026-09-30 | mesh `b0f24903` | Deleted the prepared-program transport in favor of communicator sessions. Current `mesh-call.c` is not the old numerical graph executor described in some historical documents. |
| 2026-10-17 | mesh `da21ce68`, `b08b1a7b` | Restored the Python partial surface; moved numerical kernels and layers to examples. Historical `python/mesh/nn.py:linear` records K-panel products using `BlockSpec`, then combines by output tile. |
| 2026-10-17 | microbench `49a32932` | Removed mandatory precise GPU timestamps around every crossed window by making the timing mode explicit. Current instructions go further: measurement must not alter execution. |
| 2026-10-21 | mesh `9a167a57` | The fused chain was an existing fast numerical implementation using shared fused transport. A later unification must preserve that algorithm rather than infer that abstraction alone improves it. |
| 2026-10-21 | microbench `28ea2afe`, `3ae9e2bc` | A resident-kernel attempt was added and reverted. “Fewer dispatches” did not alone establish better complete execution. Recorded forms still need whole-function performance evidence. |
| 2026-10-22 | mesh `910214c3` | Removed numerical examples including the K-panel layer examples during the test-code deletion. The partial declaration types remain; their intended numerical use must be recovered from history. |
| 2026-10-23 | mesh `1c0afa53` | Removed the fused chain implementation in favor of DTensor/Inductor operations. The commit subject concerns an error formatter, so the subject does not describe the full source change. Shared library reuse did not require losing the fused algorithm. |
| 2026-10-23 | microbench `04f25200` | Optimized prepared ICB segments on their submission queue and removed preparation-time empty blit waits for forms with no ICB. This was preparation work, not proof that every runtime wait was gone. |
| 2026-10-24 | mesh `4bb01da0`, `92060dc5`, `1c5b65ee` | Added the current FX declaration module, declarations for functionalized destination writes and whole-function observation/routing integration. |
| 2026-10-25 | mesh `86928ab3`, `0fe2374f`, `7f498de7` | Lowered crossings to prepared indices, extended sum lowering and corrected FX operand replacement. These are real AOT frontend capabilities. |
| 2026-10-25 | mesh `079c057f`, `0cd9e915`, `9f80ce00` | Removed an execution controller, restored the fused chain, then batched host encoding and restored a block wait. The block wait still conflicts with the partial-input requirement; this review does not excuse our own addition. |

## MPSGraph and what was actually found

The current project source and the inspected source history did not yield a project-owned MPSGraph graph
builder. The concrete historical reference found is mesh
`c97d1745:rdma/torch-mesh/REVIEW-commit-rule.md`, discussing what happens when MPSGraph calls
`commitAndContinue` inside the framework's MPS command-buffer handling. It is an integration note, not this
project's graph representation.

That finding does not say PyTorch or a system framework never uses MPSGraph internally. It identifies the
machinery this project owns and can reuse now: partial declarations, the FX declaration path, the native Metal
recorder, the physical write compiler and the two native model adapters. None needs a new per-step controller
to explain its purpose.

## The integration rule

Take graphs and placement as compiler inputs. Produce prepared page assignments, masks, routes and command
operands. Let numerical code use those operands. Feed observations from completed whole calls into the next
configuration, even when several old records arrive together. The solver changes the plan; it does not acquire
authority to pause, time or wrap each operation.

The existing code supplies much of this structure. The removal manifest records where a caller still bypasses
it or where the common implementation remains incomplete. The remaining work is to connect and simplify these
pieces while preserving the best numerical algorithms and all physical routing capabilities.

The [partial tensor and FX reference](../docs/partial-tensors-and-fx.md) documents the current interfaces.
It includes the `18be9473` consolidation of LSTM chunk preparation into `declarations.Function`, replacing
the separate first-issue compilation and peer-compilation barrier.
