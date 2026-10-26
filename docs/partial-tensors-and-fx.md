# Partial tensors and the FX declaration path

This is the interface reference for `python/mesh`, the ragged PyTorch frontend, and the native headers they
use. It describes the source on `main` on 2026-10-26. The [data model](../design/data-model.md)
explains why the project uses pages, compilation and one-sided writes. This document names the existing types,
their operands, and the work each interface performs. The [graph history](../design/aot-graphs-and-pytorch.md)
records how the native and PyTorch paths acquired their roles.

## How the pieces connect

| Input | Preparation | Execution |
|---|---|---|
| `mesh.Program`, tensor blocks and kernel declarations | Build read/write relationships; allocate local backings; declare collective groups and bind native writes | One native schedule contains the numerical spans and prebound crossings. |
| A PyTorch function, including backward when the function calls it | `declarations.Function` fake-traces the function; the compiler binds numerical commands and collective operands | Submit the same native schedule as the native frontends. |
| Native `MatrixView`, `MetalProgram`, or recorded Metal commands | Bind dispatches, operand slots, accesses and residency | Native command submission/replay, also used by the FX frontend. |

Partial movements and FX declarations meet in `backend.Call`, `ProcessGroupMesh`, and `libnccl-mesh`.
The communicator's physical graph retains parallel cables and relay nodes. A logical participant set is an
operand describing who contributes or receives values; it is not a replacement physical topology.

Three meanings of “partial” must stay distinct:

- `mesh.Partial` records which mathematical terms a value contains and which it still needs.
- PyTorch's `Partial` placement describes an unreduced distributed value. `RaggedShard` describes unequal shards.
- The `PARTIALS` collective combines attention records `[m, l, O...]` with the native log-sum-exp operator.

A tensor block is a numerical view. It is not necessarily one bridge page. A transport unit can cover less than
one allocation page. The current byte-span interfaces also remain broader than the page-plus-within-page-mask
contract required by the data model; the existence of those interfaces does not resolve that gap.

## Python partial tensor API

Source: [python/mesh/__init__.py](../python/mesh/__init__.py). The package exports `Program`, `Tensor`, `Ref`,
`Partial`, `BlockSpec`, `ShapeDtypeStruct`, and `Result`. It also carries the bridge module's
names from [rdma/mesh.py](../rdma/mesh.py), so existing users of `mesh.ffi`, `mesh.lib`, and region storage use the
same substrate. Put the checkout's `python/` before `rdma/` on the import path when using this surface.

### Storage and view types

| Type or member | Operands and meaning |
|---|---|
| `Tensor` | A two-dimensional shape, block shape, dtype and map from block coordinates to `Ref`. Edge blocks can be smaller. Construct through `Program.tensor`. |
| `Tensor.blocks`, `Tensor[i, j]` | The block map and one block reference. |
| `Tensor.region(row, column, rows, columns)` | A view within one backing block, or across blocks when the tensor has a contiguous backing. |
| `Tensor.T`, `Tensor.broadcast_to(shape)` | Transposed or broadcast view metadata. |
| `Tensor.on(peer)` | The pair `(tensor, peer)` used to place a copy. |
| `Tensor.dense()` | The backing view when contiguous; otherwise concatenates blocks into a new tensor. It is not always a zero-copy accessor. |
| `Ref` | A program, backing, element offset, rows, columns, row and column strides, and logical origin. `shape`, `dtype` and `partial` describe the value. |
| `Ref.key` | `(backing.index, offset, rows, columns, row_stride, column_stride)`, identifying a view. |
| `Ref.tensor` | A `torch.as_strided` view over the backing. |
| `Ref.array` | A NumPy view for CPU storage; converting an MPS value can copy it to CPU. |
| `Ref.slice(row, column, rows, columns)` | A smaller view within the reference. |
| `Ref.T`, `Ref.broadcast(rows, columns)` | View transformations; broadcast uses zero strides. |
| `Ref.on(peer)` | The pair `(ref, peer)` used to place a copy. |
| `Ref.whole()` | Requires a contiguous view covering its entire backing. This helper's restriction is not the rule for all current remote movement. |
| `_Backing` | Internal storage record: program, element count, scalar type, transferability, index, storage and region membership. `flat()` creates storage if absent; realization forces the local backings it needs. |
| `ShapeDtypeStruct(shape, dtype=np.float32)` | Describes a kernel's output before its storage exists. |
| `Partial(required, terms)` | Frozen sets of required and present term labels. `merge(refs)` unions terms, rejects duplicate terms, and returns `None` when the result is complete. |

Offsets and strides in `Ref` are in elements. Native byte views multiply them by element width during
preparation. A `Ref` does not expose a remote process pointer. The lower native APIs nevertheless accept local
buffer/extent descriptions, so this alone does not establish the final constrained page interface.

### Block selection and numerical kernels

`BlockSpec(block_shape, index_map=None, region_map=None)` connects an operand to a kernel's grid. Its
`index_map(*coordinate)`, required for a non-`None` block shape, selects a block origin, multiplied by `block_shape`; edge extents are clipped to the
tensor. `region_map(ref, *coordinate)` can further select the reference passed to the kernel. A `None` block
shape passes the operand through. `_bind(tensor)` and `resolve(coordinate)` bind and resolve that metadata while
recording calls.

`Program.kernel_call(kernel, *, out_shape, grid, in_specs, out_specs, peer=None)` returns a configuration
function. Calling that function with input operands creates the output `Tensor` objects and records one kernel
call for each grid coordinate. A single `ShapeDtypeStruct` returns one tensor; a sequence returns a tuple.
There must be one block specification for each input and output. `peer=None` places the kernel on all ranks;
a rank number places it on that rank.

The numerical implementation is called as:

```python
implementation(program, sources, targets, coordinate)
```

Here `sources` and `targets` are the references resolved by the block specifications. The implementation is
`program.functions.get(kernel, kernel)`. A kernel accepting incomplete mathematical terms must declare
`kernel.combine = True`; ordinary kernels cannot consume those terms as complete values. Recording merges
input term metadata into output metadata. This mechanism does not automatically infer missing terms from an
arbitrary matrix multiply.

For example, a matrix product can name four output tiles. The input references include each tile's full
inner dimension, so the following declaration needs no reduction of partial sums:

```python
from mesh import Program, BlockSpec, ShapeDtypeStruct

def product(program, sources, targets, coordinate):
    targets[0].tensor.copy_(sources[0].tensor @ sources[1].tensor)

p = Program('metal')
a = p.tensor((64, 64), block_shape=(32, 64))
b = p.tensor((64, 64), block_shape=(64, 32))
c = p.kernel_call(
    product, out_shape=ShapeDtypeStruct((64, 64)), grid=(2, 2),
    in_specs=(BlockSpec((32, 64), lambda i, j: (i, 0)),
              BlockSpec((64, 32), lambda i, j: (0, j))),
    out_specs=BlockSpec((32, 32), lambda i, j: (i, j)),
)(a, b)
# Fill the input views before preparation.
p.realize()
```

This compiles the numerical span without executing it. Call `p.run()` to submit
the prepared function. There is no implicit launch when an input write ends. The distributed form also
records the collective declarations and rank ownership before `realize`.

### Program lifecycle and host access

| Interface | Behavior |
|---|---|
| `Program(backend='cpu', *, functions=None, group=None)` | Creates the SPMD declaration surface. Backends are `cpu`, `metal` and its alias `mps`. Every rank records the same program; ownership operands place work. |
| `tensor(shape, block_shape=None, dtype=np.float32, transferable=True, *, contiguous=False)` | Declares blocked storage. `contiguous=True` gives blocks one shared backing. |
| `copy(source.on(a), destination.on(b), *, queue=0)` | Records a local copy or native remote movement. Tensor copies intersect source and destination block cuts. Remote preparation binds original source and final destination views. |
| `send`, `recv` | Aliases of `copy`. They are two names for the same graph declaration, not a runtime request/response pair. |
| `replicate(source, peer)` | Declares and caches a copy of the source backing at another peer, retaining the source view's layout. |
| `constant(ref, value=None)`, `is_constant(ref)` | Mark storage as constant, optionally fill it, and query that declaration. |
| `write(ref)` | Yields the input's existing torch tensor. It allocates no staging buffer and starts no execution. Fill inputs before preparing their call. |
| `export(ref)` | Records a local result use and returns a `Result`. Declare it before realization. |
| `realize()`, alias `start()` | Resolves read/write order, allocates storage, binds native crossings, and compiles numerical spans through `Function`. It executes no payload and starts no threads. |
| `run` | The native submission function, bound by `realize`. Its commands and storage are already fixed. |
| `close()` | Ends owned group lifecycle. Context-manager exit calls it. |
| `group`, `rank`, `size`, `graph` | The communicator, physical rank information and prepared sequence metadata. |

`Result.tensor` is the declared output view. Completion is observed for the whole submitted function,
using the same external completion mechanism as the other compiled examples. There is no per-result
readiness guard, consume acknowledgment, reader token, failure controller or execution trace.

`_Task` retains declaration metadata only. `realize` computes overlapping read/write dependencies and a
single order across all ranks. Maximal numerical spans use `Function`; native crossings retain the original
and final storage. Both append to one native schedule before execution. No Python scheduler walks tasks
at runtime. Compilation failure propagates without running the source callable. The native FX emitter
currently supports Metal; a CPU numerical graph fails preparation instead of using a framework executor.
CPU/ANE shares explicitly bound by `engines.Prepared` use the common recorder's host records.

A numerical kernel uses tensor operations that the FX compiler can trace. NumPy conversion or opaque host
work is not an alternate execution path. Constants are input declarations; preparation does not execute a
separate constant-only numerical graph. Repeated complete calls follow the same compiled graph.

### Collective declarations over blocks

Source: [collective.py](../python/mesh/collective.py). Every function below takes `(program, value, *, ...)`
and has `queue=0`. `owners` maps block coordinates to physical ranks. `peers` names logical participants.
Movement functions return the original tensor with its declared movement; a user-supplied combine can create
new result blocks.

| Function | Other keyword operands | Result placement |
|---|---|---|
| `all_gather` | `peers`, `owners` | Owned blocks replicated among participants. |
| `broadcast` | `root`, `peers` | Root's blocks replicated to participants. |
| `all_to_all` | `owners`, `destinations` | Each block moved from its owner to its specified destination. |
| `scatter` | `root`, `owners` | Root's blocks moved to owners. |
| `gather` | `root`, `owners` | Owners' blocks moved to root. |
| `reduce_scatter` | `peers`, `owners`, `op=SUM` | Combined blocks held by their owners. |
| `reduce` | `root`, `peers`, `op=SUM` | Combined blocks held by root. |
| `all_reduce` | `peers`, `owners=None`, `op=SUM` | Combined blocks replicated among participants. |

`sum_scatter` aliases `reduce_scatter`; `all_sum` aliases `all_reduce`; `all_max` and `all_min` select `MAX`
and `MIN`. `send` and `recv` alias `Program.copy`; `recv_like` aliases `Program.replicate`.

`SUM`, `MAX` and `MIN` bind native reductions. `PARTIALS` requires complete contiguous float32 records within
each row, with at least `[m, l, O]`; gaps between rows are allowed. An associative callable `op` moves terms to
an owner and combines them with a tree of kernel calls. It must declare `combine`, and custom `all_reduce`
requires an owner map.

The movement preparation hooks build `SpanWrites` from original/final views and compact relay storage.
Reduction preparation computes common logical cuts, then binds contiguous aliases of the original operands.
Each preparation returns its native bindings and common `Call` group. The full physical communicator remains present
when some ranks only relay. [The `_native` adapter](../python/mesh/_native.py) supplies:

| Interface | Role |
|---|---|
| `World(device, group)` | Own or reuse a mesh process group; establish rank and size. |
| `World.union_cuts(task, cuts)` | Exchange layout cuts during preparation. |
| `World.prepare(keys)`, `World.of(queue, members)` | Bind logical queue/member keys to the physical communicator. |
| `World.declare(tasks)` | Plan the prepared task declarations through the common library. |
| `World.close()` | Group lifecycle. |
| `dtype`, `numpy_dtype` | Normalize scalar types; NumPy conversion does not provide a bfloat16 dtype here. |
| `allocate(numel, scalar, device, in_region)` | Prepare CPU/MPS storage, using region storage when required. |
| `call(...)`, `prepare_writes(deliveries, group)`, `prepare_fragments(bindings, group)` | Adapt metadata, prepare native writes, and prepare native reduction fragments. |

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

| Interface | Inputs and outputs |
|---|---|
| `trace(function, *arguments, parameters=())` | Fake-traces the function and explicit backward, saves/restores real gradient references, returns the graph and communication declarations. |
| `operation(node)` | Reads operators and named operands, including functionalization wrappers. |
| `lower(graph)` | Produces ordered common `Call` groups; unknown communication fails preparation. |
| `BoundGraph` | Splits the declaration at communication and binds consumers directly to prepared numerical outputs and their aliases. It adds no publication copy or separate numerical output allocation. |
| `compiler(graph, operands)` | Functionalizes the declaration, preserves the existing small-row linear fusions, and calls the native emitter. |
| `native.compile_graph` | Uses Inductor's tensor IR, fusion and Metal source generator. Its generated Python runs once to allocate and bind commands, not to execute data. Unknown external kernels fail preparation. |
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

`SpanWrites` takes bindings `(tensor, [(offset, bytes), ...], source, destination)` and provides
`append(schedule)` for composition and `realize()` / `enqueue()` for a complete declared transfer. `ProcessGroupMesh.prepare_writes` constructs these from local tensor views.
`Collectives` binds reduction `Call` operands through the same native preparation and encoder as generic
collectives. Its `enqueue()` includes the GPU consumer dependencies without creating a host Work handle.
`views.byte_spans(shape, byte_strides, itemsize)` returns a base displacement and logical-order byte spans;
it reads metadata, not tensor values. Reversed and broadcast dimensions retain their mappings.

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
     destination=0, fused=False, contributors=None, matrix=None, recipients=None)
```

The operation constants are `ALLREDUCE`, `REDUCE`, `BROADCAST`, `ALLGATHER`, `REDUCE_SCATTER`, `ALLTOALL`, and
`WRITE`. `counts` carries variable rank counts; all-to-all can carry send and receive counts and a source-major
rank matrix. `contributors` and `recipients` independently select logical sources and results. For `WRITE`,
`root` is the source and `destination` the receiver. A declared group is a list of calls, optionally paired with
its repetition count. This is different from the six-field execution tuple
`(kind, send_tensors, receive_tensors, counts, operation, root)` used by `_issue`.

`ProcessGroupMesh.plan` translates declarations to `ncclMeshCall`. `configuration` temporarily installs a
configuration plan; `plan_demands`, `reset_fused_configuration` and `plan_active` support preparation. The
process group implements PyTorch collectives, coalesced operations, variable-count gathering/scattering,
attention partial merging, and group lifecycle (`shrink`, `grow`, `revoke`, `shutdown`). `partials_operator(width)`
prepares the native reduction operator. Some generic list-buffer paths still use `packed` storage; do not
infer zero-copy behavior merely from their sharing the communicator.

### Headers and their data types

| Source | Types and entry points | Role |
|---|---|---|
| [nccl.h](../rdma/nccl.h) | `ncclMeshCall`, `ncclMeshPlan`; `ncclMeshCommPlan`, `PlanDemands`, `PlanRouteUnits`, `PlanActive` | Common ordered declaration, compiled publication/landing entries and routing demand. |
| [nccl.h](../rdma/nccl.h) | `ncclMeshBuffer`, `ncclMeshIssue`, `ncclMeshFused`, `ncclMeshStream` | Local Metal buffer/offset, ticket/publication state, physical crossing bindings, and caller-owned encoding context. |
| [nccl.h](../rdma/nccl.h) | `ncclMeshWritePlan_t`, `ncclMeshSpan`; `ncclMeshWritePrepare`, `WriteRowsPrepare`, `WriteSpansPrepare`, `WriteExecute`, `WritePlanDestroy` | Prepare native one-sided writes from contiguous, affine-row or span views; execute the prepared plan. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_operand`, `mesh_step`, `mesh_program`, `mesh_move` | Element metadata, selected numerical/transport steps and collective programs. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_cable`, `mesh_fabric`, `mesh_paths`, `mesh_transfer`, `mesh_routed_rank`, `mesh_routed_program`; `mesh_write_compile`, `mesh_routed_compile` | All physical cables, path weights and routing output, including relays. Path weights use fixed-point units. |
| [mesh-plan.h](../rdma/mesh-plan.h) | `mesh_link_map`, `mesh_trees`, `mesh_group`, `mesh_group_call`, `mesh_place`, `mesh_group_place` | Logical connectivity, collective trees and placement of ordered calls on transport streams. |
| [mesh-span-plan.h](../rdma/mesh-span-plan.h), [mesh-row-span-plan.h](../rdma/mesh-row-span-plan.h) | Span and row-span compilation helpers | Reconcile logical cuts with physical registration boundaries during preparation. |
| [mesh-write-routes.h](../rdma/mesh-write-routes.h), [mesh-write-order.h](../rdma/mesh-write-order.h) | Routed write and ordered publication helpers | Common path and dependency descriptions used by prepared writes. |
| [mesh.h](../rdma/mesh.h) | `mesh_page_entry`, `mesh_dma_span`, `mesh_entry`, `mesh_units`, `mesh_pages`, `mesh_row` | Bridge page mappings, DMA spans, indexed stream entries, transport-unit counts, allocation-page counts and row identities. These counts are distinct. |
| [mesh-call.h](../rdma/mesh-call.h) | `mesh_section`, `mesh_endpoint`, `mesh_stream_view`; section and stream binding functions | Preallocated sections, physical endpoint discovery and selected stream bindings. |
| [mesh-dataflow.h](../rdma/mesh-dataflow.h) | `mesh_ctx`; attach, backing/row allocation and binding, retirement | Bridge setup and storage ownership. These are not per-transfer allocation APIs. |
| [nccl-mesh-metal.h](../rdma/nccl-mesh-metal.h), [nccl-mesh-kernels.h](../rdma/nccl-mesh-kernels.h) | Metal adapters and common numerical/collective kernels | Encode the common compiled plan into the caller's Metal context. |

`ncclMeshCall` carries operation, peer, count/counts, dtype, reduction, group, repetition, destination, fused
flag, all-to-all matrix, contributors and recipients. `ncclMeshPlan` provides group count, issue count, first
issue and publication/landing entry counts. `ncclMeshFused` stores arrays of six-word publication and landing
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
| `MetalProgram` | Ordered dispatch list. `append` composes it; `bind` prepares a `MatrixStep`, selecting direct or indirect encoding during binding. |
| `MatrixStep` | `(MTLCommandBuffer) -> Void`, a bound numerical encoding function. |

[metal_recording.h](../../../metal-microbench/metal_recording.h) provides `MetalRecordingDevice`,
`MetalDispatchReaching`, `MetalRecord`, `MetalSchedule` and `MetalReplayFree`. LiteRT receives the recording
device and queue, so its pipelines retain reflected read/write modes. `MetalBindingView` and `MetalRecordViews`
carry its declared external tensor extents into capture. `MetalRecordMark` marks crossings. `MetalOperand`
supplies slots, stride, instance count and output offset. `MetalReach` supplies subranges and access flags.
`MetalReplay` retains commands and residency. `MetalForm` selects indirect, encoded or serial preparation;
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
