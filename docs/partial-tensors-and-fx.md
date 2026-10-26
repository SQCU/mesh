# Partial tensors and the FX declaration path

This is the interface reference for `python/mesh`, the ragged PyTorch frontend, and the native headers they
use. It describes the source on `main` after `18be9473` (2026-10-25). The [data model](../design/data-model.md)
explains why the project uses pages, compilation and one-sided writes. This document names the existing types,
their operands, and the work each interface performs. The [graph history](../design/aot-graphs-and-pytorch.md)
records how the native and PyTorch paths acquired their roles.

## How the pieces connect

| Input | Preparation | Execution |
|---|---|---|
| `mesh.Program`, tensor blocks and kernel declarations | Build read/write relationships; allocate local backings; declare collective groups and bind native writes | The current Python task executor runs kernels and prepared crossings. It still has dynamic scheduling work. |
| A PyTorch function, including backward when the function calls it | `declarations.Function` fake-traces the function; the `mesh` compiler lowers its communication and gives crossings fixed indices | Call the prepared numerical function. Its custom operators reach `_stream.mm` and the same collective library. |
| Native `MatrixView`, `MetalProgram`, or recorded Metal commands | Bind dispatches, operand slots, accesses and residency | Native command submission/replay. FX does not currently lower into this recorder. |

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
`Partial`, `BlockSpec`, `ShapeDtypeStruct`, `Result`, `MeshError`, and `Report`. It also carries the bridge module's
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
| `Ref.writable` | Queries the current executor's reader state. This is runtime bookkeeping, not a page-table lookup. |
| `_Backing` | Internal storage record: program, element count, scalar type, transferability, index, storage and region membership. `flat()` creates storage if absent; realization forces the local backings it needs. |
| `ShapeDtypeStruct(shape, dtype=np.float32)` | Describes a kernel's output before its storage exists. |
| `Partial(required, terms)` | Frozen sets of required and present term labels. `merge(refs)` unions terms, rejects duplicate terms, and returns `None` when the result is complete. |

Offsets and strides in `Ref` are in elements. Native byte views multiply them by element width during
preparation. A `Ref` does not expose a remote process pointer. The lower native APIs nevertheless accept local
buffer/extent descriptions, so this alone does not establish the final constrained page interface.

### Block selection and numerical kernels

`BlockSpec(block_shape, index_map=None, region_map=None)` connects an operand to a kernel's grid. Its
`index_map(*coordinate)` selects a block origin, multiplied by `block_shape`; edge extents are clipped to the
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

### Program lifecycle and host access

| Interface | Behavior |
|---|---|
| `Program(backend='cpu', *, functions=None, group=None, trace=False)` | Creates the SPMD declaration surface. Backends are `cpu`, `metal` and its alias `mps`. Every rank records the same program; ownership operands place work. |
| `tensor(shape, block_shape=None, dtype=np.float32, transferable=True, *, contiguous=False)` | Declares blocked storage. `contiguous=True` gives blocks one shared backing. |
| `copy(source.on(a), destination.on(b), *, queue=0)` | Records a local copy or native remote movement. Tensor copies intersect source and destination block cuts. Remote preparation binds original source and final destination views. |
| `send`, `recv` | Aliases of `copy`. They are two names for the same graph declaration, not a runtime request/response pair. |
| `replicate(source, peer)` | Declares and caches a copy of the source backing at another peer, retaining the source view's layout. |
| `constant(ref, value=None)`, `is_constant(ref)` | Mark storage as constant, optionally fill it, and query that declaration. |
| `write(ref)` | Host write context for a registered block. It checks current readers; CPU gives a NumPy view, while MPS currently stages through a NumPy buffer and copies on exit. Exit publishes the host input to the executor. |
| `export(ref)` | Records a local result use and returns a `Result`. Declare it before realization. |
| `realize()`, alias `start()` | Builds dependencies, allocates local storage, prepares crossings, declares their order and starts worker threads. It also executes constant-only initial work. It is not a preparation-only function. |
| `close()` | Ends the executor and releases its owned resources; unfinished activity may cause communicator revocation. Context-manager exit calls it. |
| `group`, `rank`, `size`, `report`, `trace` | The communication group, physical rank information, failure report and optional execution trace. |

`Result.ready` observes its task and event. `Result.tensor` returns the view when ready and raises `EAGAIN`
otherwise. `Result.array` returns a read-only CPU view or conversion. `Result.consume()` releases the local
export use in the executor. It is not a remote receipt acknowledgment.

`MeshError(kind, code, *, peer=None, call=None, detail='')` reports a failed call. `Report` contains `code`,
`kind`, `peer`, `call`, `detail`, `kernels` and `crossings`. The last two describe executor progress, not a
performance measurement or an enumerated correctness suite.

The internal `_Task` stores ownership, roles, input/output edges, guards, execution state, event and prepared
bindings. `_Edge` stores its source, target, tokens, sticky state and outstanding work. `_Order` maintains a
cyclic crossing order. The executor still uses a ready heap, conditions and reader tokens. These are the
specific runtime interfaces listed for replacement in [manifest R03](../design/runtime-interface-removal-2026-10-25.md#r03-the-python-partial-tensor-executors-second-scheduling-system).
The useful block and access descriptions do not require retaining that executor.

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

`sync_on_remote_fill(*results)` explicitly waits for every result. `sync_on_some_remote_fill(*results)` waits
for any result from one program. These are existing blocking host APIs, not preparation steps or requirements
of one-sided transport. They are not the model for a numerical runtime.

The movement preparation hooks build `SpanWrites` from original/final views and compact relay storage.
Reduction preparation computes common logical cuts, then binds contiguous aliases of the original operands.
`_declaration(task)` exports the ordered common `Call` groups. The full physical communicator remains present
when some ranks only relay. [The `_native` adapter](../python/mesh/_native.py) supplies:

| Interface | Role |
|---|---|
| `World(device, group)` | Own or reuse a mesh process group; establish rank and size. |
| `World.union_cuts(task, cuts)` | Exchange layout cuts during preparation. |
| `World.prepare(keys)`, `World.of(queue, members)` | Bind logical queue/member keys to the physical communicator. |
| `World.declare(tasks)` | Plan the prepared task declarations through the common library. |
| `World.revoke()`, `World.close()` | Group lifecycle. |
| `dtype`, `numpy_dtype` | Normalize scalar types; NumPy conversion does not provide a bfloat16 dtype here. |
| `allocate(numel, scalar, device, in_region)` | Prepare CPU/MPS storage, using region storage when required. |
| `call(...)`, `prepare_writes(deliveries, group)`, `fragments(bindings, group)` | Adapt metadata, prepare native writes, and issue prepared reduction fragments. |

## The FX path

Sources: [declarations.py](../rdma/torch-mesh/torch_mesh/declarations.py),
[compiler](../rdma/torch-mesh/torch_mesh/__init__.py), and
[the current LSTM caller](../examples/lstm-service.py).

### Whole-function declaration

`Function(function, *arguments, parameters=(), compiler=None)` holds a numerical callable, its bound operands
and its preparation results. The tracer receives those operands explicitly as fake inputs; `run` binds them
once for execution. The callable can also close over selected weights and layout metadata. If training is intended, it must include the desired
backward operations; supplying parameters alone does not invent a backward graph.

| Member | Work performed |
|---|---|
| `declare(group)` | Fake-traces the whole callable, retains parameter gradients as graph outputs, lowers communication, and invokes the chosen compiler now. Returns this group's ordered `Call` groups. |
| `graph`, `groups` | The FX graph and declarations keyed by process-group name. |
| `forms`, `preparations` | Crossing metadata and the storage-binding preparation built from it. |
| `run` | The prepared numerical callable. With no compiler it is the original Python function with its arguments bound. With a compiler it is the compiled FX callable returning flattened outputs and gradients. |
| `output_spec` | The output tree description needed to reconstruct the user's structured result. |
| `prepare(repetitions=1)` | Bind the crossing storage and repeated forms before numerical execution, after the communicator's plan is installed. |
| `__call__()` | Call `run`, restore parameter gradients and reconstruct structured outputs. Call `run` directly when those adaptations are unnecessary. |

Preparation order is: select placement and allocate operands; declare/compile the functions; concatenate their
communication declarations in invocation order; install the common communicator plan; bind crossing forms;
then invoke the prepared functions. `configuration.bind` already supplies the plan/install/preparation order
for the examples. A function used on several groups must have every group's declaration planned.

The LSTM service prepares a function for each distinct chunk length, including its remainder. It constructs
its tuple of compiled callables before execution. The former `torch_mesh.record` / `Recording` interface was
removed in `18be9473`: there is no first-issue compilation, store rendezvous for peer compilation, throwaway
numerical replay, or warmup reset in that service. Its starting ticks use the same declaration and compilation path. The compiler prepares the native
linear pipelines from graph metadata, so this path does not need to execute a product to compile it. This
change does not establish native command-buffer replay for FX.

### Tracing and lowering interfaces

| Interface | Inputs and outputs |
|---|---|
| `trace(function, *arguments, parameters=())` | Uses `make_fx` with fake tensors, saves/restores real gradient references, and returns `(graph, group_declarations)`. It does not run a payload to discover its communication. |
| `operation(node)` | Reads the operator and named operands from direct nodes or PyTorch functionalization wrappers, using graph metadata. |
| `lower(graph, *, forms=None)` | Produces `group_name -> ordered Call groups`; optionally appends crossing forms. Unknown communication fails during configuration. |
| `compiled_graph(graph)` | Lowers the final compiled graph, allocates crossing indices, and replaces fused crossing nodes with indexed operators. Publishes declarations to `_CAPTURE` when a caller is collecting them. |
| `tensor_metadata(x)` | `None` or `(shape, stride, dtype)`. |
| `crossing_key(form)` | The tensor metadata, dimension, rank sizes, group name, addition flag and repeat count that identify a crossing form. |
| `crossing_slot(form, destination)` | Allocates an indexed native slot; records output shape; returns `(index, destination)`. |
| `Preparations(forms)` | Collects repeated forms, eager slots and compiler-assigned bindings. Calling it prepares storage and native crossings; `repetitions` expands their count. |
| `eager_index(...)` | Looks up a metadata key and advances an eager slot cursor. This remains a runtime alternative identified for removal in manifest R16. Compiled indexed nodes do not need this lookup. |

A form's first seven entries are `(x, w, dim, sizes, group_name, adds, repeats)`. Compiled forms also carry a
sequence of `(index, destination)` bindings. `w=None` means no fused product; `destination=True` denotes the
operator writing a supplied output. `_INDEXED_SHAPES` supplies fake output metadata. `_CAPTURE` is a context
variable used during compilation, not an execution controller.

`lower` recognizes mesh all-gather, sum, linear-sum, fused crossings, reduce-scatter and attention merging. It
also recognizes functional c10d all-reduce, broadcast, all-gather, reduce-scatter, all-to-all and supported
coalesced forms. `wait_tensor` contributes no new transport call. That does not mean every downstream
implementation of a dependency wait has been removed.

The `mesh` backend uses Inductor's numerical compilation and rewrites suitable small-row products to the
existing `linear` / `linears` kernels. It folds supported activation and concatenation patterns and combines
repeated compatible crossings. Its final graph, including those rewrites, supplies the transport declaration.
A cached numerical binary therefore does not substitute for declaring the communication of the current plan.

### Operators at the Python/native boundary

| Operator family | Role |
|---|---|
| `torch_mesh::allgatherv`, `allgatherv_out` | Gather unequal rank segments; the output form writes supplied storage. |
| `torch_mesh::reduce_scatterv` | Reduce and deliver unequal segments. |
| `torch_mesh::all_reduce`, `linear_sum` | Sum, or a local product followed by a sum. |
| `torch_mesh::merge_partials` | Merge native attention records. Public `torch_mesh.merge_partials(out, lse, group=None, root=-1)` converts to/from that representation. |
| `torch_mesh::linear`, `linears` | Existing small-row dense/grouped numerical kernels. |
| `torch_mesh::fused_step`, `fused_step_out` | Metadata-rich forms consumed during preparation/lowering. |
| `torch_mesh::indexed_step`, `indexed_step_out` | Execution forms naming the prepared crossing index; the latter writes supplied output storage. |

`_stream.prepare_linear(xs, ws, x1s, act)` prepares the native row/group specialization from tensor metadata
(`act=-1` selects the grouped kernel). The compiler invokes it for the final product nodes. Unprepared eager
callers still have the older lazy pipeline lookup, and output allocation remains part of manifest R06.

Fake implementations describe output metadata for tracing. Native implementations and slot storage are in
[_stream.mm](../rdma/torch-mesh/torch_mesh/_stream.mm). `PreparedWrites`, `FusedForm` and `IndexedCrossing`
hold prepared native state. `prepare_crossing` prepares an indexed dispatch and its output where needed;
`run_crossing` still updates bindings and enters the MPS encoding queue. Indexed transport preparation is not
yet the same as a fully prerecorded native numerical command stream.

`SpanWrites` takes bindings `(tensor, [(offset, bytes), ...], source, destination)` and provides
`execute(async_op=False)`. `ProcessGroupMesh.prepare_writes` constructs these from local tensor views.
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
| `prepare_gather(x, dim, sizes, group, *, adds=False, w=None, repeats=1, batches=1)` | Prepare fused gathering/summing forms before execution. |
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
`MetalDispatchReaching`, `MetalRecord`, `MetalReplaySubmit` and `MetalReplayFree`. `MetalRecordingAdopt` admits
commands from a library using the supplied queue; its current accesses are conservative. `MetalRecordMark`
marks crossing positions. `MetalOperand` supplies buffer slots, stride, instance count and output offset.
`MetalReach` supplies subranges and read/write/binding/residency flags. `MetalReplay` retains commands,
residency and their owner. `MetalForm` selects indirect, encoded or serial preparation; `MetalCrossing` is the
callback binding a collective at a recorded mark.

These types already carry information that later wrappers must reuse. They do not establish that all current
callers use fine-grained reaches or that every submission is allocation-free. Profiling helpers such as
`StepProfile` and `MetalReplayGPUNanoseconds` remain removal targets under manifest R15, not the measurement
contract for this integration.

## Current integration boundary

The shared declarations preserve routes, unequal work shares, partial views and physical channel counts.
The LSTM now uses the common AOT function preparation. The remaining work is concrete: replace the partial
API's dynamic executor, remove eager crossing lookup, finish the constrained page/mask boundary, and connect
prepared numerical commands without per-operation allocation or encoding waits. Those scopes are recorded
in the [dated removal manifest](../design/runtime-interface-removal-2026-10-25.md).

Neither an interface name nor compilation alone establishes a faster implementation. Performance records
come from the canonical committed-main hourlies: complete-function elapsed time and useful output bandwidth,
with rows/s for chain and LSTM workloads. This reference adds no alternative runner or measurement path.
