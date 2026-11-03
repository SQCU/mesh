# Partial tensors and the FX declaration path

The [data model](../design/data-model.md)
explains why the project uses pages, compilation and one-sided writes. This document names the existing types,
their operands, and the work each interface performs. The [graph history](../design/aot-graphs-and-pytorch.md)
records how the native and PyTorch paths acquired their roles.

## How the pieces connect

| Input | Preparation | Execution |
|---|---|---|
| Native `MatrixView`, `MetalProgram`, or recorded Metal commands | Bind dispatches, operand slots, accesses and residency | Native command submission/replay. |

The communicator's physical graph retains parallel cables and relay nodes. A logical participant set is an
operand describing who contributes or receives values; it is not a replacement physical topology.

A tensor block is a numerical view. It is not necessarily one bridge page. A descriptor
can cover a sub-page selection or merge adjacent selections across page boundaries.

## The FX path

Sources: [compiler](../rdma/torch-mesh/torch_mesh/__init__.py), and
[the current LSTM caller](../examples/lstm-service.py).

### Tracing and lowering interfaces

The complete transformation inventory and native/LiteRT correspondence are in
[the compiler pipeline reference](compiler-pipeline.md).

PyTorch supplies tensor descriptions, graph capture and differentiation. Our compiler owns placement,
storage, kernel choice, collective plans and submission. Compiler passes can be reused or extended without
adopting PyTorch's runtime or its device choices. This is a general tensor interface; model names do not
select another executor.

Generic CPU FX lowering
remains missing.

Mathematical kernel choice remains preparation work. Segmented inputs remain separate views:
each segment contributes to the same accumulator, including ragged boundaries and batched views. Wider LSTM
batches therefore need no concatenation copy or separate executor.

### Operators at the Python/native boundary

| Operator family | Role |
|---|---|
| `torch_mesh::allgatherv_out` | Gather unequal rank segments; the output form writes supplied storage. |

Fake implementations describe output metadata.

Prepared functions return their existing completion through `completion()`.
Applications observe results at their host data dependencies and yield at final shutdown until pending reports arrive.

## Ragged tensors and parallel operations

Source: [partition.py](../rdma/torch-mesh/torch_mesh/partition.py).

| Type or interface | Contract |
|---|---|
| `attach(mesh, **dims)` | Creates initial equal shares for named dimensions. |
| `split(mesh, name, N)` | Retrieve integer sizes for an extent. |
| `distribute_tensor(tensor, device_mesh, placements)` | Distribute a global tensor according to the placements. |

Zero-share ranks remain available to the physical routing compiler. No compute share means neither “remove this node” nor
“discard its cables.”

## Shared call and native header contracts

### Headers and their data types

| Source | Types and entry points | Role |
|---|---|---|
| [nccl.h](../rdma/nccl.h) | `ncclMeshCall`, `ncclMeshPlanRun`, `ncclMeshPlan` | Common ordered declaration, compiled publication/landing entries and routing demand. |
| [nccl.h](../rdma/nccl.h) | `ncclMeshBuffer`, `ncclMeshStream` | Local Metal buffer/offset and caller-owned encoding context. |
| [nccl.h](../rdma/nccl.h) | `ncclMeshPageTable_t`, `ncclMeshLayout`; Call `views`/`bindings` | Declare exact local page views in the complete communicator plan. No standalone WRITE lifecycle. |
| [mesh-write-compiler.h](../rdma/mesh-write-compiler.h) | `ncclMeshSpan`, `mesh_write_layout` | Compiler-local tensor bindings into the same page domain; typed repeats. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_operand`, `mesh_step` | Element metadata, selected numerical/transport steps. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_cable`, `mesh_fabric`, `mesh_paths`, `mesh_transfer`, `mesh_routed_rank`, `mesh_routed_program`; `mesh_write_compile`, `mesh_routed_compile` | All physical cables, path weights and routing output, including relays. Path weights use fixed-point units. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_link_map`, `mesh_trees` | Logical connectivity, collective trees. |
| [mesh-span-plan.h](../rdma/mesh-span-plan.h), [nccl-mesh.c](../rdma/nccl-mesh.c) | Modular repeat and common view cut compilers | Reconcile logical cuts, publication tiles and each occurrence's physical registration boundaries during preparation. |
| [mesh-graph.h](../rdma/mesh-graph.h) `mesh_graph_schedule` | Routed write and ordered publication helpers | Common path and dependency descriptions. |
| [mesh.h](../rdma/mesh.h) | `mesh_page_entry`, `mesh_dma_span`, `mesh_units`, `mesh_pages`, `mesh_row` | Bridge page mappings, DMA spans, transport-unit counts, allocation-page counts and row identities. These counts are distinct. |
| [mesh-call.h](../rdma/mesh-call.h) | `mesh_section`, `mesh_endpoint`, `mesh_stream_view`; section and stream binding functions | Preallocated sections, physical endpoint discovery and selected stream bindings. |
| [mesh-dataflow.h](../rdma/mesh-dataflow.h) | `mesh_ctx`; attach, backing/row allocation and binding, retirement | Bridge setup and storage ownership. These are not per-transfer allocation APIs. |

`ncclMeshCall` carries operation, peer, count/counts, dtype, reduction, group, destination,
all-to-all matrix, contributors and recipients. `ncclMeshPlan` provides group count, issue count, first
issue, publication/landing entry counts and each declared all-to-all's complete count matrix.

## Native partial views and command preparation

Microbench's `tools/prepare_e2b.py` reads a
FlatBuffer graph, cuts its dependency graph, applies the shared planner's phase parts, and emits numerical stages
and collective declarations. `tools/mesh/programs.py:phase_ranks` resolves both disjoint phase membership and
per-phase layer overrides. A PyTorch/FX description is another frontend; importing LiteRT does not require
translating its operators into PyTorch or Inductor.

`handoff(prefill, decode, S, bs, rows, layout, source_scope)` intersects KV head ranges. The model configuration supplies
`layout = {type, element, axes}` explicitly. `crossed_heads` checks those declarations against the graph and emits
a `view = {name, shape, heads, axis, element}` over an existing tensor. Source and destination consume that same
format declaration. The receiver uses the export's cache-update fragment; no packing graph is inserted.

`LiteRtProgramCreate` compiles the stages and names their buffers. `LiteRtProgramDeclare` binds those buffers,
records native commands and crossing marks, and declares them through `MeshRankDeclareReplay`. `MeshRankPrepare`
encodes all phases and the speculative epilogue into one command stream. `MeshRankRun` enqueues it; completion
reports the whole function. Relay-only ranks use this same path.

The native numerical
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

These types already carry information that later wrappers must reuse. They do not establish that all current
callers use fine-grained reaches or that every submission is allocation-free.

## Current integration boundary

The shared declarations preserve routes, unequal work shares, partial views and physical channel counts.
The LSTM uses common AOT function preparation. Remaining work includes the constrained page/mask boundary and connecting
prepared numerical commands without per-operation allocation or encoding waits. Those scopes are recorded
in the [dated removal manifest](../design/runtime-interface-removal-2026-10-25.md).

Neither an interface name nor compilation alone establishes a faster implementation. Performance records
come from the canonical committed-main hourlies: complete-function elapsed time and useful output bandwidth,
with rows/s for chain and LSTM workloads. This reference adds no alternative runner or measurement path.
