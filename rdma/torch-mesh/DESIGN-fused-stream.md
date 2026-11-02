# Numerical source and one-sided writes

Current contract, 2026-11-01. See [the data model](../../design/data-model.md),
[whole-function dependencies](../../design/whole-function.md), and [nccl.h](../nccl.h).

The numerical compiler and collective compiler share one preparation pass. Frontends retain numerical source,
its page operands, and its work-to-page relations. Collective declarations retain their source and destination
page selections. The pass resolves their dependencies, composes production and consumption effects, then binds
backend commands. Runtime submits the prepared function.

## The preparation boundary

`declarations.Function` captures tensor source, including its derivative graph when requested. Ownership and
layout assignment produce ordinary page views. `LoweredGraph` retains the numerical sections and collective
boundaries. A subsequent declaration can consume those same prepared sections with explicit new page operands
and occurrence coordinates; it does not trace the source again or restore an executable snapshot.

Native kernels enter through the same retained-source contract. Each declaration supplies its source, specialization,
resource geometry, operands, and exact dispatch-group-to-scalar relations. Those relations become page accesses
through the common layout algebra. They are not replaced by an inferred whole-buffer access.

`Execution` flattens these declarations into one source/effect solve. The solver attaches incoming-data observations
and outgoing publications only where the declared accesses need them. The shared recorder binds the resulting
commands. The source body does not carry a separate per-channel wait loop or publication wrapper.

## The examples

`chain.py` retains the historical SIMD matrix phases from `chain.metal`. Resource capabilities determine their
geometry during preparation. The recurrence uses two declared storage generations. Input broadcasts and recurrent
contributions are ordinary selected-page WRITE declarations, including masks that leave padded rows and columns
out of DMA. A batch reuses one framed declaration for its shape and storage generation. Input windows use explicit
linear binding slots; the plan's order repeats these definitions. Frame positions identify where each producer and
consumer enters the common dependency solve, including any required source-preservation copy.
Native leaves are recorded at those same positions through explicit command selections on the sole
`ncclMeshCommEncode` API. Descriptor-publication leaves remain at occurrence entry. Recording does not move
all native leaves ahead of the external arithmetic or replace exact generations with a forward-read guess.

`dnn.py` captures products, tanh, and all-reduce in an ordinary `Function`. Each distinct row count is captured and
lowered once. Other tiles bind their input and result pages to the retained sections and give their source calls an
explicit occurrence. Every tile's commands join the same whole-function preparation pass.

`lstm-service.py`, the tensor-parallel examples, and these native declarations use the same `Execution`. A frontend
may describe source through FX or through a retained native kernel; it cannot substitute another runtime scheduler.

The former `torch_mesh.fused`, `FusedForm`, `PreparedChain`, `chain_record`, `chain_run`, and manual
`linear_publish`/`gather_ready` machinery is removed. Source declarations and page relations replace those APIs.
The repository history retains the earlier implementation and its measurements.

## What has been checked

The native library, CoreML provider, CFFI module, and extension build after removal of the obsolete interfaces. Offline Metal compilation covers the retained
chain's initial, recurrent, and final arithmetic phases, including a ragged logical row/column extent. Python source
also parses. The current relation library needs bridge-backed preparation pages; no bridge region was attached for
this review. This migration has not yet established a complete region-backed preparation or an end-to-end
latency/throughput result.
