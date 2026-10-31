# Partial tensors and the FX declaration path

This is the interface reference for the ragged PyTorch frontend and the native headers it
uses. python/mesh, a Pallas-spelled block DSL over torch_mesh, was deleted on 2026-10-30: nothing imported it. It describes the source on `main` through 2026-10-27. The [data model](../design/data-model.md)
explains why the project uses pages, compilation and one-sided writes. This document names the existing types,
their operands, and the work each interface performs. The [graph history](../design/aot-graphs-and-pytorch.md)
records how the native and PyTorch paths acquired their roles.

## How the pieces connect

| Input | Preparation | Execution |
|---|---|---|
| A PyTorch function, including backward when the function calls it | `declarations.Function` fake-traces the function; the compiler binds numerical commands and collective operands | Submit the same native schedule as the native frontends. |
| Native `MatrixView`, `MetalProgram`, or recorded Metal commands | Bind dispatches, operand slots, accesses and residency | Native command submission/replay, also used by the FX frontend. |

FX declarations meet the library in `backend.Call`, `ProcessGroupMesh`, and `libnccl-mesh`.
The communicator's physical graph retains parallel cables and relay nodes. A logical participant set is an
operand describing who contributes or receives values; it is not a replacement physical topology.

Two meanings of “partial” must stay distinct:

- PyTorch's `Partial` placement describes an unreduced distributed value. `RaggedShard` describes unequal shards.
- The `PARTIALS` collective combines attention records `[m, l, O...]` with the native log-sum-exp operator.

A tensor block is a numerical view. It is not necessarily one bridge page. Program-facing WRITEs select local
pages with masks confined to each page. Tensor adapters resolve local storage into that page domain during
preparation. Affine rows stay compact until the common compiler emits physical DMA descriptors. A descriptor
can cover a sub-page selection or merge adjacent selections across page boundaries.

## The FX path

Sources: [declarations.py](../rdma/torch-mesh/torch_mesh/declarations.py),
[compiler](../rdma/torch-mesh/torch_mesh/__init__.py), and
[the current LSTM caller](../examples/lstm-service.py).

### Whole-function declaration

`Function(function, *arguments, parameters=())` holds a numerical callable, its bound operands
and its preparation results. The tracer receives those operands explicitly as fake inputs; `run` binds them
once for execution. The callable can also close over selected weights and layout metadata. If training is intended, it must include the desired
backward operations; supplying parameters alone does not invent a backward graph.

| Member | Work performed |
|---|---|
| `declare(group)` | Fake-traces the whole callable, retains parameter gradients as graph outputs, lowers communication, and invokes the shared mesh compiler now. Returns this group's ordered `Call` groups. |
| `graph`, `groups` | The FX graph and declarations keyed by process-group name. |
| `run` | Native submission with bound operands, returning the prebound flattened outputs and gradients. There is no eager execution option. |
| `output_spec` | The output tree description needed to reconstruct the user's structured result. |
| `prepare(repetitions=1)` | Append numerical commands and transport to one native schedule per invocation, after the communicator plan is installed. |
| `__call__()` | Call `run`, restore parameter gradients and reconstruct structured outputs. Call `run` directly when those adaptations are unnecessary. |

Preparation order is: select placement and allocate operands; declare the functions; install their
communication plan; append their numerical and transport records; finish the native schedule; submit it.
`configuration.bind` supplies the plan/install/preparation order. A function used on several groups must
have every group's declaration planned.

`execution_sequence(steps, output)` composes objects with `append(schedule)` during preparation.
`realize(repetitions)` binds invocation slots. `forward` submits the next slot; `completion` returns its
ordinary native completion record. Neither method enters PyTorch's encoding queue. The LSTM service
appends its input broadcast, starting ticks and all chunks into one complete native function.

### Tracing and lowering interfaces

The complete transformation inventory and native/LiteRT correspondence are in
[the compiler pipeline reference](compiler-pipeline.md).

| Interface | Inputs and outputs |
|---|---|
| `trace(function, *arguments, parameters=())` | Fake-traces the function and explicit backward, saves/restores real gradient references, returns the graph and communication declarations. |
| `operation(node)` | Reads operators and named operands, including functionalization wrappers. |
| `lower(graph)` | Produces ordered common `Call` groups; unknown communication fails preparation. |
| `BoundGraph` | Splits the declaration at communication and binds consumers directly to prepared numerical outputs and their aliases. It adds no publication copy or separate numerical output allocation. |
| `compiler(graph, operands)` | Functionalizes the declaration, preserves the existing small-row linear fusions, and calls the native emitter. |
| `native.prepare_graph` | Retains tensor IR, typed access relations and the fusion scheduler. Ordinary and owned functions select and bind this same graph through `BoundGraph`; generated Python runs during preparation to bind native commands. |
| `native.empty_strided(shape, stride, dtype, device)` | Assigns a compiler-declared view to bridge region storage during preparation. Both ordinary intermediates and custom linear outputs use it. |
| `_stream.Commands` | Converts pipelines, tensor views, constants, access modes and dispatch dimensions into `MetalCommandDescription` records. |
| `Execution` | Appends numerical and transport records to the common schedule before execution. |

PyTorch supplies tensor descriptions, graph capture and differentiation. Our compiler owns placement,
storage, kernel choice, collective plans and submission. Compiler passes can be reused or extended without
adopting PyTorch's runtime or its device choices. This is a general tensor interface; model names do not
select another executor.

The current generic FX emitter still borrows Inductor's numerical IR and Metal source generator.
Products already lower to our shared native matrix bodies. Other numerical lowerings can replace upstream
source generation while retaining the same graph metadata and command interface. Generic CPU FX lowering
remains missing; CPU/ANE shares currently enter through `engines.Prepared` and the common native recorder.
Joint graph placement across those capabilities remains integration work.

Training attention now declares its forward math and lets autograd include the backward. Its runtime
workspace pool, saved-score ownership machinery and lazy mask cache are removed. The shared preparation
allocator supplies its buffers, so numerical execution needs neither that pool nor PyTorch's MPS heap
allocator. The same storage change applies to other FX numerical spans and their collective outputs.

`compiled_graph`, `Preparations`, `_CAPTURE`, crossing-key lookup and indexed crossing runtime operators
are removed. Mathematical kernel choice remains preparation work. `linear.metal` serves the existing
small-row products. `matrix.metal` contains the matrix body extracted from the DNN program; the DNN
and larger FX matrix products now import that same implementation. Segmented inputs remain separate views:
each segment contributes to the same accumulator, including ragged boundaries and batched views. Wider LSTM
batches therefore need no concatenation copy or separate executor. Publication counts include every batch.
This removes a runtime distinction;
whole-function hourlies must establish the resulting latency and useful bandwidth.

### Operators at the Python/native boundary

| Operator family | Role |
|---|---|
| `torch_mesh::allgatherv`, `allgatherv_out` | Gather unequal rank segments; the output form writes supplied storage. |
| `torch_mesh::reduce_scatterv` | Reduce and deliver unequal segments. |
| `torch_mesh::all_reduce`, `linear_sum` | Sum, or a local product followed by a sum. |
| `torch_mesh::merge_partials` | Merge native attention records. Public `torch_mesh.merge_partials(out, lse, group=None, root=-1)` converts to/from that representation. |
| `torch_mesh::linear`, `linears` | Existing small-row dense/grouped numerical kernels. |

Fake implementations describe output metadata. `_stream.linear_commands` binds the existing linear
kernel into command records with supplied output storage. No eager linear implementation allocates an
output or enters a framework queue during a compiled call. `FusedForm` remains preparation metadata for
the native chain; it is not a separate FX executor.

`ViewWrites` takes bindings `(tensor, (row_bytes, rows, pitch), spans, source, destination)` and provides
`append(schedule)` for composition and `realize()` / `enqueue()` for a complete declared transfer. `ProcessGroupMesh.prepare_writes` constructs these from local tensor views.
Affine views have an empty span list; irregular views have zero row operands. There is one prepared executor.
`Collectives` binds reduction `Call` operands through the same native preparation and encoder as generic
collectives. Its `enqueue()` includes the GPU consumer dependencies without creating a host Work handle.
`mesh_view_layout(shape, byte_strides, rank, itemsize, ...)` (`rdma/mesh-write-compiler.h`; `mesh.view_layout` from
Python) returns a base displacement, compact row operands and any irregular spans. It reads metadata, not tensor
values. Reversed and broadcast dimensions retain their mappings. Affine geometry resolves in time bounded by
dimensions, with no span.

`MeshWork.wait`, `isCompleted` and `getFuture` concern host encoding completion in this adapter. They are not
a passive GPU completion clock. `_stream.completion()` and `_stream.completed(token)` expose ordinary command
completion state to the asynchronous observer. The canonical measurement path uses that observation, not a
new numerical executor or a timing drain.

## Ragged tensors and parallel operations

Source: [partition.py](../rdma/torch-mesh/torch_mesh/partition.py). This layer extends PyTorch's DTensor
placement and parallel styles. It is not the `mesh.Tensor` type above.

| Type or interface | Contract |
|---|---|
| `_Operand` | One immutable partition identity: key, integer capacity tuple, total units, CPU parts tensor and cached derived masks/indices. |
| `RaggedShard(dim, key)` | A `Shard` whose sizes come from that partition identity. Derives local sizes, offsets and redistribution behavior. |
| `RaggedTensor` | A DTensor carrying ragged placement. Provides `to_local`, `redistribute`, `full_tensor` and `from_local`. Autograd adapters preserve placement information. |
| `attach(mesh, **dims)` | Creates initial equal shares for named dimensions. |
| `operand(mesh, name)`, `split(mesh, name, N)` | Retrieve partition metadata or integer sizes for an extent that is a multiple of its units. |
| `_attach(mesh, name, parts)` | Configuration-level installation of a new partition identity. Old tensors retain their previous identity until moved. |
| `empty(mesh, name, shape, dim, dtype=None)` | Allocate a region-backed ragged tensor for that dimension. |
| `distribute_tensor(tensor, device_mesh, placements, *, src_data_rank=0)` | Distribute a global tensor according to the ragged placements. |
| `from_local(local_tensor, device_mesh, placements, *, run_check=False, shape=None, stride=None, grad_placements=None)` | Wrap local storage with global layout metadata. |
| `redistribute(tensor, placements, *, async_op=False)` | Convert the distributed layout. |
| `follow(*tensors)` | Move tensors from their old partition identities to the current configuration. This is configuration work. |
| `calls(tensor)` | Declare the gathers that this ragged tensor’s `full_tensor()` would issue. |
| `plan(mesh, name, groups)` | Install ordered collective groups, or a callable producing those groups from the current split. `_replan` realizes a changed configuration. |
| `gathered(x, dim, sizes, group)`, `summed(x, group)`, `scattered(x, dim, sizes, op='sum')` | Produce common collective declarations for the corresponding operations. |
| `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel` | PyTorch parallel styles with ragged parameter and activation placement. |
| `context_parallel_buffers(mesh, name, buffers, seq_dims, load_balance=True)` | Select each rank's sequence pieces. |
| `context_parallel_unshard(mesh, name, buffers, seq_dims, load_balance=True, *, sequence_length=None)` | Reconstruct those sequences. |
| `ring_attention(mesh, name, query, key, value, is_causal=False, scale=None, load_balance=True, *, sequence_length=None)` | Compute distributed attention through query gathering and partial-result reduce-scatter. The name does not require a physical ring. |
| `ring_calls(mesh, name, query_shape, dtype, value_width, *, sequence_length=None)` | Declare that attention's communication. |
| `unshard_calls(mesh, name, shapes, dims, dtype, *, sequence_length=None)` | Declare sequence reconstruction. |

Zero-share ranks need explicit global shape/stride or sequence length where local shape cannot determine it.
They remain available to the physical routing compiler. No compute share means neither “remove this node” nor
“discard its cables.” Allocation, migration and derivation of cached masks belong before the numerical call;
current eager wrappers are not all free of such work.

## Shared call and native header contracts

### Ordered calls

[backend.py](../rdma/torch-mesh/torch_mesh/backend.py) defines:

```python
Call(kind, count, dtype, root=0, op=ReduceOp.SUM, counts=None,
     destination=0, fused=False, contributors=None, matrix=None, recipients=None, tile_bytes=0)
```

The operation constants are `ALLREDUCE`, `REDUCE`, `BROADCAST`, `ALLGATHER`, `REDUCE_SCATTER`, `ALLTOALL`, and
`WRITE`. `counts` carries variable rank counts; all-to-all can carry send and receive counts and a source-major
rank matrix. `contributors` and `recipients` independently select logical sources and results. For `WRITE`,
`root` is the source and `destination` the receiver. `tile_bytes` gives the logical publication
width; zero means one publication for the whole operand. A declared group is a list of calls, optionally paired with
its repetition count. This is different from the six-field execution tuple
`(kind, send_tensors, receive_tensors, counts, operation, root)` used by `_issue`.

`ProcessGroupMesh.plan` translates declarations to `ncclMeshCall`. Its explicit `order` operand
accepts `ncclMeshPlanRun` dictionaries (`first`, `count`, `repeat`) to repeat contiguous blocks
of group definitions without duplicating calls. `configuration` temporarily installs a
configuration plan; `plan_demands` reports preparation demands. The
process group implements PyTorch collectives, coalesced operations, variable-count gathering/scattering,
attention partial merging, and group lifecycle (`shrink`, `grow`, `revoke`, `shutdown`). `partials_operator(width)`
prepares the native reduction operator. Some generic list-buffer paths still use `packed` storage; do not
infer zero-copy behavior merely from their sharing the communicator.

### Headers and their data types

| Source | Types and entry points | Role |
|---|---|---|
| [nccl.h](../rdma/nccl.h) | `ncclMeshCall`, `ncclMeshPlanRun`, `ncclMeshPlan`; `ncclMeshCommPlan`, `PlanDemands`, `PlanRouteUnits` | Common ordered declaration, compiled publication/landing entries and routing demand. |
| [nccl.h](../rdma/nccl.h) | `ncclMeshBuffer`, `ncclMeshFused`, `ncclMeshStream` | Local Metal buffer/offset, physical crossing bindings, and caller-owned encoding context. |
| [nccl.h](../rdma/nccl.h) | `ncclMeshPageTable_t`, `ncclMeshLayout`, `ncclMeshBufferSlots`; Call `views`/`bindings`, `ncclMeshCommEncode` | Declare exact local page views in the complete communicator plan and record its next group. No standalone WRITE lifecycle. |
| [mesh-write-compiler.h](../rdma/mesh-write-compiler.h) | `ncclMeshSpan`, `mesh_write_layout`; `mesh_compile_write`, `mesh_compile_rows`, `mesh_compile_spans` | Compiler-local tensor bindings into the same page domain; compact affine rows and typed repeats. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_operand`, `mesh_step`, `mesh_program`, `mesh_move` | Element metadata, selected numerical/transport steps and collective programs. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_cable`, `mesh_fabric`, `mesh_paths`, `mesh_transfer`, `mesh_routed_rank`, `mesh_routed_program`; `mesh_write_compile`, `mesh_routed_compile` | All physical cables, path weights and routing output, including relays. Path weights use fixed-point units. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_link_map`, `mesh_trees`, `mesh_group`, `mesh_group_call`, `mesh_place`, `mesh_group_place` | Logical connectivity, collective trees and placement of ordered calls on transport streams. |
| [mesh-span-plan.h](../rdma/mesh-span-plan.h), [nccl-mesh.c](../rdma/nccl-mesh.c) | Modular repeat and common view cut compilers | Reconcile logical cuts, publication tiles and each occurrence's physical registration boundaries during preparation. |
| [mesh-write-routes.h](../rdma/mesh-write-routes.h), [mesh-write-order.h](../rdma/mesh-write-order.h) | Routed write and ordered publication helpers | Common path and dependency descriptions used by prepared writes. |
| [mesh.h](../rdma/mesh.h) | `mesh_page_entry`, `mesh_dma_span`, `mesh_entry`, `mesh_units`, `mesh_pages`, `mesh_row` | Bridge page mappings, DMA spans, indexed stream entries, transport-unit counts, allocation-page counts and row identities. These counts are distinct. |
| [mesh-call.h](../rdma/mesh-call.h) | `mesh_section`, `mesh_endpoint`, `mesh_stream_view`; section and stream binding functions | Preallocated sections, physical endpoint discovery and selected stream bindings. |
| [mesh-dataflow.h](../rdma/mesh-dataflow.h) | `mesh_ctx`; attach, backing/row allocation and binding, retirement | Bridge setup and storage ownership. These are not per-transfer allocation APIs. |
| [nccl-mesh-metal.h](../rdma/nccl-mesh-metal.h), [nccl-mesh-kernels.h](../rdma/nccl-mesh-kernels.h) | Metal adapters and common numerical/collective kernels | Encode the common compiled plan into the caller's Metal context. |

`ncclMeshCall` carries operation, peer, count/counts, dtype, reduction, group, repetition, destination, fused
flag, all-to-all matrix, contributors and recipients. `ncclMeshPlan` provides group count, issue count, first
issue, publication/landing entry counts and each declared all-to-all's complete count matrix. `ncclMeshFused` stores arrays of six-word publication and landing
records, their counts, repeat count and local frame stride. These arrays are sized by physical channels,
not by peer count.

[fused.py](../rdma/torch-mesh/torch_mesh/fused.py) exposes `Crossing(publish, land)`, `steps()`, `writes`,
`write_rows`, `encode` and `encode_many`. `steps()` supplies the shared Metal prelude. `writes` binds routed
source/landing storage and repeated frames; `write_rows` binds useful row spans. `encode` supplies constants
and crossing bindings to one numerical kernel; `encode_many` batches the same arguments into one host queue
entry. It still builds encoding metadata when called.

The prelude's `mesh_step_binding` and `mesh_step_enter`, `mesh_step_await`, `mesh_step_release`,
`mesh_step_write`, `mesh_step_read` and `mesh_last_arrival` connect a kernel's reads and stores to its compiled
publication/landing relations. Their dependency operations do not authorize an arbitrary whole-buffer wait,
a peer handshake, or another runtime route selector.

## Native partial views and command preparation

The LiteRT frontend joins the same native preparation interface. Microbench's `tools/prepare_e2b.py` reads a
FlatBuffer graph, cuts its dependency graph, applies the shared planner's phase parts, and emits numerical stages
and collective declarations. `tools/mesh/programs.py:phase_ranks` resolves both disjoint phase membership and
per-phase layer overrides. A PyTorch/FX description is another frontend; importing LiteRT does not require
translating its operators into PyTorch or Inductor.

`handoff(prefill, decode, S, bs, rows, layout)` intersects KV head ranges. The model configuration supplies
`layout = {type, element, axes}` explicitly. `crossed_heads` checks those declarations against the graph and emits
a `view = {name, shape, heads, axis, element}` over an existing tensor. Source and destination consume that same
format declaration. The receiver uses the export's cache-update fragment; no packing graph is inserted.

`LiteRtProgramCreate` compiles the stages and names their buffers. `LiteRtProgramDeclare` binds those buffers,
records native commands and crossing marks, and declares them through `MeshRankDeclareReplay`. `MeshRankPrepare`
encodes all phases and the speculative epilogue into one command stream. `MeshRankRun` enqueues it; completion
reports the whole function. Relay-only ranks use this same path. The host-stage executor, late recording,
eager crossing path, host relay executor and per-batch submission waits are removed (manifest R12–R14).

The earlier Swift surface `swift/Mesh.swift` was deleted in `a8952de9`; it is historical source, not an API
available to import today. The Python surface preserves its block-oriented contract. The native numerical
composition types that remain are in microbench's
[matrix_operations.swift](../../../metal-microbench/matrix_operations.swift):

| Type or function | Meaning |
|---|---|
| `MatrixElement` | `float16` or `float32`. |
| `MatrixView` | Buffer, rows, columns, element strides, byte offset and element type. `byteOffset(row:column:)` resolves a local element address. |
| `Reach` | Buffer, byte offset, length and access flags; `.declared` creates a `MetalReach`. |
| `MetalDispatch` | Prepared pipeline, buffers, offsets, constants, grid, threadgroup memory, access reaches and optional collective crossing. |
| `MetalProgram` | Ordered dispatch list. `append` composes it; `bind` prepares a `MatrixStep`, selecting direct or indirect encoding during binding; `described` hands a recording its command descriptions, host operations and crossings. |
| `MatrixStep` | `(OpaquePointer) -> Void`, a bound numerical step appended to a Metal 4 schedule. |

[metal-recording.h](../rdma/metal-recording.h) provides `MetalCommandDescription`, `MetalHostDescription`,
`MetalReplayDescribed`, `MetalReplayCommands`, `MetalSchedule` and `MetalReplayFree`. A program's commands are
described, never captured from an encoder; a crossing is a mark its describing caller places at a command index
(`MeshRankCross`). `MetalOperand` supplies slots, stride, instance count and output offset. `MetalReach` supplies
subranges and access flags. LiteRT partial-row stages carry a `MetalTensorLayout` per binding. The layout
supplies both shader indices and the ranges in its command's description;
its `threadGrid` operand distinguishes thread counts from group counts during
preparation. The former groups-only `MetalDispatchReaching` wrapper is removed.
`MetalReplay` retains commands and residency. `MetalForm` selects indirect or encoded preparation;
`MetalCrossing` binds a collective at a recorded mark during preparation.

`MetalScheduleCreate`, `After`, `Append` and `Finish` prepare one function. `MetalScheduleEnqueue` commits it.
`MetalScheduleFree` retires it after completion. The configured replay form and state operands are bound before execution.

These types already carry information that later wrappers must reuse. They do not establish that all current
callers use fine-grained reaches or that every submission is allocation-free. The native GPU interval counter was removed.
`StepProfile` remains a removal target under manifest R15, not the measurement
contract for this integration.

## Current integration boundary

The shared declarations preserve routes, unequal work shares, partial views and physical channel counts.
The LSTM uses common AOT function preparation. The partial API compiles its execution order; eager crossing
lookup is removed. Remaining work includes the constrained page/mask boundary and connecting
prepared numerical commands without per-operation allocation or encoding waits. Those scopes are recorded
in the [dated removal manifest](../design/runtime-interface-removal-2026-10-25.md).

Neither an interface name nor compilation alone establishes a faster implementation. Performance records
come from the canonical committed-main hourlies: complete-function elapsed time and useful output bandwidth,
with rows/s for chain and LSTM workloads. This reference adds no alternative runner or measurement path.

Tiled writes declare `ncclMeshCall.production`: one predicted readiness time
and producer-order ordinal per logical tile. `torch_mesh.backend.Call` and the
MPI dictionary interface carry those same records. Native head and attention
lowerings report their local producer geometry before the shared graph is
compiled. The DNN's matrix row tiles now form one WRITE family per pair, and
`fused.writes(tile_bytes=...)` returns tile-major physical-channel bindings.
Several contributions to one consumer tile share each cable's last required
prefix. [Producer ordering](native-access-publication.md#producer-order-in-the-common-write-compiler--2026-10-27-0647-utc)
records the current scheduling prior and remaining empirical-fitting work.
