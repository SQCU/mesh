# torch-mesh

torch.distributed's `mesh` backend over [libnccl-mesh](../NCCL.md). MPS tensors run on the library's Metal path:
encoded on torch's MPS stream, with region operands passed as their storage's `MTLBuffer` and offset. The transport
reads and lands those bytes in place; higher-level packing and layout operations can still make copies, as described
below. CPU tensors run on the host path.

Ordinary backend collectives are groups issued by `_stream.mm`'s `issue` (the backend's methods, torch's functional
collectives and the library's own calls), each call one kind of the library's set. Fused producer/consumer calls
bind the same physical compiler through their native lowering, described below. An all-gather and a
reduce-scatter are always the V calls (`ncclMeshAllGatherV`, `ncclMeshReduceScatterV`; the library takes equal
segments as its equal program). `all_to_all`, `all_to_all_single`, `gather` and `scatter` are `ncclMeshAlltoAllV`,
MPI_Alltoallv with its counts on the host (a gather or scatter of any sizes is the rooted V, MPI's `Gatherv` and
`Scatterv`: every count 0 but toward or from the root, as NCCL's gather and scatter are its forms).
List operands bind their original buffers through the same fragment compiler; no packed receive or copy-back path remains. `send` and `recv` are the library's point to point; the
partials merge is a reduction by the library's partials operator.

## Install

A mesh checkout holds the library and this package, built in place (rdma/NCCL.md "Build and run": `make install`,
`rdma/mesh_c_build.py`, `setup.py build_ext --inplace` here), and a program run with the user's torch environment,
`~/.venv-mesh-uv` (`bin/mesh-user.sh`: `rdma/requirements.txt`, torch 2.14.0 on CPython 3.12.14), imports `mesh`,
`torch_mesh` and the mesh's modules from it by path (`PYTHONPATH=<checkout>/rdma:<checkout>/rdma/torch-mesh`).

## Porting a CUDA/NCCL program

```python
dist.init_process_group(backend="mesh", device_id=torch.device("mps", 0))   # or no backend: MPS's default is mesh
```

- `"nccl"` becomes `"mesh"` (or omit it); `"cuda"` becomes `"mps"`; drop `torch.cuda.set_device`. `device_id`
  needs its index (`mps:0`).
- One rank a node, launched as torchrun launches one (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`); each node's
  bridge is its resident service, so a launch starts no bridge.
- A group is made by NCCL's init (`ncclCommInitRank`, its unique id passed through the group's store): the ranks
  exchange their nodes and the nodes their bridges reach, and the group keeps the map its init made of them.
- A bidirectional exchange with one peer goes in one batch: `dist.batch_isend_irecv`, or a coalescing manager.
  Separate `isend`/`irecv` calls in the same order on both sides do not pair, and the library does not detect it
  ([NCCL.md](../NCCL.md#rules-a-program-keeps)).
- A subgroup (`new_group`) is a communicator of its own, made in any order, before or after the world group; no two
  communicators hold one stream. Each communicator acquires a stream on every participating physical cable. A link
  has up to four, subject to its endpoints' actual lane capacity; a required busy link refuses another communicator
  (EBUSY) until a stream becomes free
  ([NCCL.md](../NCCL.md#rules-a-program-keeps)).
- Every wait carries a deadline set when it is made (design/data-model.md section 6): a work's `wait(timeout)` its
  timeout, 5 s where none is given; each turn of torch's queue and each command buffer `_stream.mm` waits for, 5 s;
  the library's own waits theirs ([NCCL.md](../NCCL.md#bounded-waits)). Past it the process ends, naming the wait and
  its deadline.
- A group's collectives run where a path between its ranks joins every two of them, each rank's bridge reaching the
  next one's node. Routes cross communicator members; explicit WRITE can use a member solely to relay pages, and
  generic gathers support zero-share ranks. A group
  whose ranks no such path joins fails at its first collective that crosses a link
  ([NCCL.md](../NCCL.md#routes-between-unlinked-ranks)).
- Memory ([NCCL.md](../NCCL.md#memory)): the backend's own CPU tensors (a
  barrier's tensor and explicitly declared outputs) are leases of the node's
  bridge's region (`torch_mesh.empty`). A program's CPU tensor in the region is
  `torch_mesh.empty(shape, dtype=..., device="cpu")`.
- Region MPS tensors: every MPS tensor is a lease's Metal buffer (`ncclMeshMemBuffer`), `torch_mesh.empty(shape,
  dtype=..., device="mps")` and torch's own (its factories', eager temporaries', Inductor's, `.to("mps")`'s) alike:
  `_stream` makes torch's MPS allocator the region's as it loads (`torch_mesh/allocator.mm`), in place over the fixed
  instance torch 2.14's factories call (`at::mps::getIMPSAllocator`, no registered or pluggable allocator for MPS), and
  its import fails where that instance is not of torch's class `at::mps::MPSAllocator`. A lease the region cannot meet
  is torch's out-of-memory error. A tensor's release returns its lease once the command buffer open on torch's stream
  at the release, which the release commits, has completed; `ncclMemFree` then defers reuse until
  both its local GPU users and all registered local DMA reads have completed, including reads the bridge has not yet
  posted (`ncclMeshMemReusable`). GPU completion alone does not establish source retirement, and no receipt collective
  is needed to establish local DMA-read completion. Every collective passes it as its `MTLBuffer` and offset, so the library reaches it where it lies wherever
  it reads or writes region operands in place ([NCCL.md](../NCCL.md#memory)). `torch_mesh.records()` lists the region
  tensors' live leases and `torch_mesh.address(t)` where a tensor's bytes lie in the region (0: outside it).
  `torch_mesh.check_memory()` warns where the process's MPS tensors outside the region come to 1 GiB or more
  ([docs/writing-a-service.md](../../docs/writing-a-service.md#memory-outside-the-region)).
  Reproduction target (metal-microbench docs/measurement.md 2026-09-30, part A): a 16.8 MB all-reduce on region
  tensors copies 0 bytes (16.78 MB otherwise); part B: tp 271.5-272.2 ms a step. Only torch's pinned CPU memory and MPS
  tensors made before `_stream` loaded lie in torch's own heaps, and a collective that sends from one, or lands in one,
  across a link fails as a value (`ncclInvalidUsage`: the library copies nothing into the region). Raw transport operands must satisfy their
  declared contiguous region layout. The generic fused gather separately supports strided source views by compiling
  their strides into its publisher, without making a contiguous input tensor; it still writes a collective segment
  and assembles the logical output.
  metal-microbench's callers: `crossing.py` `REGION_TENSORS=1` all-reduces one in place.

## What works

| works | notes |
|---|---|
| all_reduce | sum, prod, max, min, avg on int8, uint8, int32, int64, float16, bfloat16 and float32 |
| all_gather_object, all_to_all of lists, blocking send/recv | |
| broadcast, reduce | |
| all_gather (list and into a tensor) | a list of unequal sizes: one MPI_Allgatherv (Uneven nodes) |
| reduce_scatter (list and tensor) | a list of unequal sizes: one MPI_Reduce_scatter (Uneven nodes) |
| gather, scatter | any per-rank sizes: the rooted V (`ncclMeshAlltoAllV`) |
| all_to_all, all_to_all_single | uneven splits, empty inputs (`ncclMeshAlltoAllV`) |
| send, recv, isend, irecv, batch_isend_irecv | |
| barrier, async_op works, coalescing | |
| functional collectives in `declarations.Function` | AOT numerical and transport commands share the native recorder (Performance). |
| DDP | |
| ring attention | `partition.ring_attention` on capacity blocks (equal parts: torch's equal shards), and SDPA on a `RaggedShard` of the sequence, forward only, in place of `context_parallel`: split-KV partials merged by the library (below) |
| partials merge | `torch_mesh.merge_partials(out, lse, group)`: each member's attention over its own keys merged into the attention over all of them (the operator `torch_mesh::merge_partials`, below) |
| training attention | `torch_mesh.attention.attention`: SDPA's math (no mask, dropout or GQA), including autograd's backward in the whole-function declaration. Native preparation assigns region storage to scores, gradients and other intermediates; no runtime workspace pool. |
| pipeline parallelism | torch.distributed.pipelining, GPipe and 1F1B, gradients checked |
| expert parallelism | all_to_all_single (below) |

An indexed scatter into the blocks of `all_to_all_single` at a program's maximum extents, written in the program's own
code, is any dispatch by index (an expert dispatch among them); how much of a block a step fills is data in its pages.

### Physical fused writes

Every communicator retains its physical cables, including parallel cables to the same peer. Explicit WRITE and
fused collective lowering share the same physical fragment compiler, relay dependencies, and channel bindings.
`torch_mesh.fused.writes` declares immutable source/destination pairs; `fused.encode` takes a physical `channels`
extent, independently of rank counts. The removed `fused.exchange` interface is not supported. Unsupported fused
forms fail rather than executing an ordinary source read before the producer publishes it.

Generic gather and sum declare one page-padded ALLGATHER. A single `gather_ready` dispatch observes their incoming
prefixes before a parallel consumer; a one-threadgroup consumer inlines that observation. The LSTM example uses
`torch_mesh::allgatherv_out` to write its persistent hidden-state destination directly, eliminating the additional
result-to-state copy. Generic gather still uses its common segment layout; not every remaining layout copy has been
removed. Positive-contribution recurrent gathers have a typed two-slot reuse proof; zero-share cases use fresh
generations and deferred retirement. See [DESIGN-fused-stream.md](DESIGN-fused-stream.md) for the current API,
ownership requirements, and dated two-/three-rank validation.

### Partials merge

Attention whose keys are split among members (split-KV, Flash-Decoding) leaves each member a partial: each query row's
logsumexp and its attention over the member's keys. `torch_mesh.merge_partials(out, lse, group, root=-1)` merges them
by the rows' logsumexp into the attention over every member's keys (and its logsumexp), on every member or at `root`:
the operator `torch_mesh::merge_partials`, an all-reduce or reduce by libnccl-mesh's partials operator
(`ncclMeshRedOpCreatePartials`, [NCCL.md](../NCCL.md)) over float32 rows of `[m, l, O]` (`backend.partial` and
`backend.attention` make and read them). The backend's `merge_partials(out, partials, rows=None, root=-1)` also takes
`rows`, each rank's rows of the operand: a reduce-scatter by the operator (`ncclMeshReduceScatterV`), each rank's rows
merged at it. Like every other call it is one group of the one path, so it coalesces. `partition`'s ring attention is
its caller: every rank's query block gathered, each block's attention over the rank's own keys written as its owner's
rows (a half of a block that sees none of the rank's keys left at none), and one reduce-scatter, so each owner holds its
rows' attention. Its reproduction target is `cp.py`'s (Uneven nodes, below).

### Prepare and call a whole function

`declarations.Function(function, *arguments, parameters=())` fake-traces the complete numerical function.
`declare(group)` binds its numerical commands and communication declarations before execution. Install the
communicator plan, then call `prepare()` to finish the native schedule. `run()` submits it and returns
prebound output views. `completion()` exposes ordinary native completion. `Function.__call__` additionally
restores structured outputs and parameter gradients. Tracing includes backward work when the supplied
function contains it.

`execution_sequence(steps, output)` appends declared functions and transport into one native function.
The LSTM service composes the input broadcast, starting ticks and every recurrent chunk this way.
TP, EP and CP likewise compose their complete configured workload before submission.

Inductor supplies tensor IR, fusion and Metal source during preparation. Its generated wrapper only binds
buffers and command records once. It is not a runtime executor. `_stream.Commands` passes the result to
`MetalReplayCommands` and `MetalScheduleAppend`, the same interfaces used by native Metal and LiteRT.
The old `TorchSchedule` start/join handoff and indexed crossing launcher are removed. Unknown numerical
lowerings fail preparation; there is no eager or ATen execution substitute. The current native emitter
supports Metal; CPU/ANE graph declarations lower to the common recorder's host records.

The existing small-row linear kernel is preserved. Larger matrix products and the streamed DNN import
one shared matrix body. Historical interposition-based record/replay is not restored. A statistics thread
observes ordinary completion with a host clock; it does not submit work or drain a framework stream.
See the [partial tensor and FX reference](../../docs/partial-tensors-and-fx.md).

## What does not

- FSDP2 (`fully_shard`): torch's MPS has no streams or `current_device`, and its copy-in ops have no MPS kernel.
  The project does not shard parameters (FSDP is a last resort among the parallelisms): port to DDP, tensor or
  expert parallelism, or shard optimizer state by hand over `reduce_scatter_tensor` and `all_gather_into_tensor`.
- Streams: torch's MPS has one queue and no `torch.mps.Stream`, so CUDA-stream overlap (DeepSpeed `overlap_comm`,
  Megatron's overlapped gradient reduction, side streams) does not port as written. What overlaps here: an
  `async_op=True` collective waited after the compute it may overlap, or a compiled graph of functional
  collectives.
- `init_device_mesh("mps", ...)` and `DeviceMesh("mps", ...)`: torch 2.14's DeviceMesh setup calls
  `torch.mps.is_initialized` and `set_device`, which torch's MPS module does not have. Form an MPS mesh from the
  groups instead, which skips that setup: `DeviceMesh.from_group(dist.group.WORLD, "mps", mesh_dim_names=("tp",))`
  (a list of groups and `mesh=` for more dimensions); DTensor, tensor and context parallelism then run on it.
  DTensor's own sharding has no rule for MPS's `linear_backward` or SDPA: on a mesh dimension `partition` attaches,
  its RaggedTensor's dispatch carries both (Uneven nodes, below). torch-mesh patches nothing of torch.
- ReduceOp PREMUL_SUM and the bitwise ops; `monitored_barrier` (it needs a CPU device type the group does not
  list); `split_group`. One call mixing MPS and CPU tensors.
- More than one rank a node.

## Uneven nodes

Stock PyTorch splits every sharded extent equally. `torch_mesh.partition` is a partition operand on raw tensors: a
mesh dimension's extents split among its ranks by parts the backend sets and moves by the measured residual, the
program holding its share and calling the list collectives with unequal sizes (`all_gather(list)`, `reduce_scatter(out, list)`,
`all_to_all_single` with splits: MPI's v-collectives). An unequal list is one program over unequal segments, the
library's MPI_Allgatherv and MPI_Reduce_scatter (`ncclMeshAllGatherV`, `ncclMeshReduceScatterV`, [NCCL.md](../NCCL.md)),
not a broadcast or reduce rooted at each rank; where the list is the pieces of one contiguous tensor (`out.split(sizes)`,
as `partition`'s collectives pass it) the call reads or writes that tensor in place. Its callers are metal-microbench
`tools/torch_parallel/tp.py`, `cp.py` and `ep.py` and this repository's examples/chain* and lstm*. On DTensor the partition is a placement, `RaggedShard`, carried
by `RaggedTensor`, a DTensor subclass whose own dispatch keeps it capacity-shaped through every op (padding re-masked
before contractions, gathered where an op needs the dimension whole); nothing of torch is patched.

```python
from torch_mesh import partition
partition.attach(mesh, tp=16)                    # the units of every extent split along "tp"; the backend's parts
blocks = partition.split(mesh, "tp", N)          # each rank's block of an extent N, in rank order
w = partition.distribute_tensor(weight, mesh, [Shard(0)])            # a RaggedTensor: capacity blocks
parallelize_module(block, mesh, {"wq": partition.ColwiseParallel(), "wo": partition.RowwiseParallel()})
```

- `partition.distribute_tensor`, `from_local` and `redistribute` are torch's with a `Shard` on an attached dimension
  its `RaggedShard`; `ColwiseParallel`, `RowwiseParallel` and `SequenceParallel` are torch's styles over them.
- Context parallelism over capacity blocks: `context_parallel_buffers`, `ring_attention` (padded keys masked, the
  blocks merged by the partials merge) and `context_parallel_unshard`, called in place of torch's `context_parallel`.
- A partitioned dimension may be any of its mesh's: a 2-D mesh's second dimension partitioned with its first
  replicated or splitting another tensor dimension (SDPA's ring runs along the partitioned dimension's group, the
  heads or batch split by the other).

- The program declares the units and no parts. The backend splits them evenly at first, each part its rank's
  capacity, and the parts move on the run's own evidence (below); see [design/heterogeneity.md](../../design/heterogeneity.md).
  Parts are indexed by rank, not by node.

### Parts from the run's own evidence

The program declares units. `configuration.placement` computes the next work split before tensors are built.
`configuration.bind` compiles the declared calls and installs their routes. The function then runs directly.
Observations are data for the next compilation, not a saved controller to restore.

`partition.follow(*ts)` compiles exact migration ranges. Page-aligned remote ranges in declared contiguous region
layouts write from original source views to final destination views through the common WRITE interface, with
distinct relay leases and no receipt all-gather. Unchanged contiguous local intervals can alias their sources;
other retained pieces copy locally once. Contiguous sub-page layouts use compact affine
masks and repeated physical WRITE frames: each frame streams selected original bytes into owned ingress pages, and
its consumer stores valid bytes directly into the final positions. Padding cannot overwrite neighboring live
destination bytes. Source frames are produced before incoming waits, and incoming frames are consumed round-robin;
`fused.encode_many` batches the encoded sequences and submits any remaining source-producer command buffer at the
batch boundary without waiting, so later descriptor allocation cannot strand its publications. Each train is bound once, with distinct immutable storage per
frame and no receipt collective. Padding remains a wire cost, and the current implementation holds the complete
train rather than recycling frames without a lifetime proof. Unsupported non-contiguous layouts still use the
older whole-operand staging fallback.
- A placement has two throughputs, the aggregate and each stream's, so the calls trace a frontier rather than a best:
  `allocate.pareto(rows, coordinate=("ctx_in", "ctx_out"))` keeps, per workload coordinate, the call records (each a
  call's own `aggregate_tok_s` and `per_sequence_tok_s`, its `program`, `backend` and batch `bs`) that no other beats in
  both, and which backends dominate others over the points both ran; from calls alone (no solo run, no fit). The
  sampling of batch and context that feeds it is metal-microbench's `tools/mesh/frontier.py` (every point a call
  through `grid.py`'s call path). Reproduction target (metal-microbench
  output_data/confirm-20260924/summary.json): at 16384/1024, bs 4, 564.6 aggregate and 141.1 per-sequence tok/s.

## Uneven engines within a node

A node can split an operation among engines with different work rates.
`torch_mesh/engines.py` declares shares across the
node's configured engines (`mps`: native Metal programs; `cpu`: fp32 products through Accelerate, the CPU's matrix units;
`ane`: the Neural Engine, Core ML graphs of 1x1 convolutions with constant weights through `rdma/mesh-coreml.m`, which
binds the bridge's shared pages), each holding its shard of the weights resident, the shares
running at once on unified memory. Required work-rate operands seed the shares for each declared operation and shape.
`assignment.assign` adjusts them from whole-function observations, using the same numerical interface as routing.
CPU core classes and GPU counts come from `mesh_compute.hardware`; counts are not interchangeable work rates.

```python
from torch_mesh import engines
from torch_mesh.declarations import execution_sequence
pool = engines.Engines(["mps", "cpu", "ane"], ane=configuration["ane"])
ffn = engines.FFN(gate, up, down, engines=pool, name="ffn")
key = ffn.declare(x.shape[0], x.dtype, capacities=configuration["work_rates"])
assigned = engines.placement(pool, observations, step=episode)
call = ffn.prepare(x, parts=assigned["parts"][key])
program = execution_sequence([call] * invocations, call.output)
program.realize()
program.forward()
y = call.output
```

- Operations: `Linear` shares its output features (they concatenate), `FFN` its intermediate neurons (each engine a
  partial of the whole output; they sum: Megatron's split), the Neural Engine's FFN share one fused graph. An `FFN`'s
  operands may be any tensors, coded matrices among them ("Streamed coded weights"): a share is a slice, the GPU's
  run by `engines.ffn` (an operand type's fused couple where no gradient is due), another engine's decoded by its copy. A decision
  is an (operation name, row count): a stack's layers of one name share it and pool their evidence, measured where
  the program runs them, between its other operations.
- The Neural Engine's outputs carry an absolute error floor (about 3e-4 RMS for a 3840-wide contraction): accurate on
  normalized activations (an FFN 1.5e-3), not on small ones. An operation's `tolerance`, which checked each
  allocation's first call against the GPU's share (`Engines.errors`), was deleted 2026-10-22 with the other test code.
- The Neural Engine's `block` option is the units one convolution of a share's graph takes, the configuration's: its
  compiler refuses a convolution past a weight size (on an M4 Pro 22.5 to 24 MiB) and Core ML then runs the share on
  the CPU, which the share's placement (`devices`) shows; the search that halved the block and built again until it
  placed, and its store of a node's largest block, were deleted 2026-10-22 (no retry).
- Engines never time-share a physical unit: each states what it occupies (`Engine.occupies`: `mps` the GPU, `cpu` the
  CPU's cores, `ane` the Neural Engine and whatever units Core ML's compute plan places a share's operations on, read
  at the share's load through `mesh_coreml_placement`; the engine's `occupies` option, `ane={"occupies": "ane+gpu"}`,
  states it in place of the engine's own), and two engines occupying one unit never hold units of a decision together
  (`allocate.Coupled.forbid`), applied when the decision is made and when a load reads its placement, before any of its
  calls is measured; the search then also tries the whole share moved between the two. Reproduction target
  (metal-microbench docs/measurement.md:496-497): an M4 Pro four-layer FFN stack on GPU and Neural Engine 59.2 ms a step
  against 81.5 (1.38x); the engine's prefill chunk with the Neural Engine 90.5 ms against 104.3.
- Nothing is implicit: an operation without an `Engines` operand is torch's own; an engine the configuration names and
  the node lacks is an error; an engine's limits (dtypes, constant weights) are data. A share's preparation
  and first run at a row count are no evidence; once an allocation stands, one call in 16 is measured.

## Streamed coded weights

`torch_mesh/streamed.metal` is the decompression of a coded matrix (metal-microbench `tools/model_code.py` exports,
format fragment-1, metal-microbench docs/kernels.md#fragment-tiles: 32 x 32 tiles, each its own width and step, in
bands of 32 rows; a tile's codes are word planes in the lanes of matmul2d's right-input cooperative tensor, a tile word
its offset, width and fp16 step, a band word its base; row and column orders and scales; for some a basis of the input:
an int8 rotation, or a randomized Hadamard transform's signs and H_K, `.signs` and `.hadamard`) as a producer whose
consumer is an operand. One decode (`fragment_values`: a lane's 16 pairs of a tile as odd integers) serves every
instance, each with literal sizes and the matrix's widths as template arguments (a caller appends the explicit
instantiations of its shapes to the source): `streamed_alu` (1 to 4 rows on the ALUs: BANDS bands a threadgroup, SK
simdgroups a band, J shares of K, F tiles a simdgroup in flight, 8 independent sums a lane, 1-bit tiles by tables of
x's sums where LUT), `streamed_couple` (gate and up with the GELU between), `streamed_coop` (8 to 16 rows: each tile
decoded into the right-input cooperative tensor of matmul2d<M, 32, 32> at the node's slots, read once by
`streamed_coopmap`; RING 1 the paired producer-consumer ring; RING r >= 8, 32 to 256 rows, the role split by rows: SK
producer simdgroups a band decoding each tile once into the threadgroup ring, M / r consumers each multiplying every
producer's tile by r rows of their own), `streamed_panel` and `streamed_panel_t` (prefill and the transposed product: eight tiles decoded
into threadgroup memory, one operation a 256-wide panel); and the finish, the input gather, rotation (writing the 1-bit
tables) and Hadamard butterfly (`streamed_hadamard`, a threadgroup a row: H_N by structured.metal's stages on every
block at once, H_K mixing the blocks as each coordinate is put, the RMS norm foldable), the GELU of gate and up, the
dense decode. `streamed_value` and `streamed_rows` read a product's output (its
K shares summed and scaled, or the plain values a crossing summed), and `streamed_gather` writes a product's input in its
column order and scale; the constants are one `streamed_dims` block at buffer 15. Which instance a slice takes at its
rows on a node (its kernel, simdgroups, bands, K shares, tiles in flight, tables; the couple or the products apart) is
the instance fit's choice (`streamed.py`, metal-microbench docs/kernels.md#the-instance-fit): the argmin of a model of
the node's times over each candidate's units, its coefficients fitted to the node's observations (its evidence file,
MESH_INSTANCES or `~/.cache/mesh/instances.json`; `python -m torch_mesh.streamed fit`), read by `Matrix.dispatch` and,
as a table of a member's slices (`python -m torch_mesh.streamed resolve`), by metal-microbench's engine, which compiles
the same source. `streamed_value` and `streamed_rows` take the partials as any pointer type, so a finish run inside its
product reads them `coherent(device)`.

A product's decomposition (its instance: kernel, template arguments, grid, threads, shares) is also an operand, in
place of the fit's choice: `streamed.spec_of('coop/M=64,SK=2,BANDS=1,J=1,LOADS=0,RING=32')` makes one, `Matrix.dispatch`,
`product`, `apply` and `ffn` take it as `decomposition=` (admitted where the matrix's bands are a multiple of BANDS and
the rows at most M), and `decomposition_of` gives metal-microbench's engine its decompositions entry (kernel, args,
grid, threads, shares; metal-microbench docs/kernels.md#decompositions). The role split by rows (RING r >= 8, mesh
99a9e70a, restored from metal-microbench a9d17c7^ tools/streamed_gemv.metal `streamed_gemm`) is reached this way only:
no candidate of the fit lists it and nothing prices it. Reproduction target (the E2B FFN gate on the M5, cold): at 64
rows at least 1.41-1.49x torch's fp16 product (the study's 752 us against 1128.6), at 128 rows at or under torch's, at
256 rows against the 128-row panel's 0.123 ms and torch's 0.086 (metal-microbench docs/kernels.md#fragment-tiles). The
`decompose` command that read named instances on random inputs beside torch's product, and the `probe` command that
read every candidate so, each checked against float64, were deleted 2026-10-22 with the other test code.

`torch_mesh/attention.metal` is decode attention as one algorithm over a KV type (`mesh_attention<A>`): a task's chunks
(the KV type's scores of each lane's positions, the chunk's online softmax of R rows, the KV type's value product), the
position groups' sum and the KV type's finish (its parts' merge and store). metal-microbench's engine instantiates it over
its fp16 pages (`decode_paged`) and its recorder over LiteRT's int8 window (`k7_window`, K7); a new KV layout is a new type
(metal-microbench docs/kernels.md#decode-attention). A prefix several streams share (their first positions [0, end) in
the same physical pages of every one's block table) is an operand of the Tile plan (mesh 2bc084cf, restored from
metal-microbench 845838e^ `paged_attn_{slide,full}_ar_shared`): `mesh_shared {end, splits}` the range and its part
count, `mesh_shared_part` a `mesh_tile` task of 8 sharing rows of any streams and query heads of one KV head over the
range (its `splits` parts at slots [0, splits)), `mesh_shared_tail` a sharing stream's own task cut to the positions past
the range (its parts at slots [splits, splits + J), returning splits + J), and `mesh_merge_row` over a row's parts after
both dispatches. A task holds 8 rows, so it reads the range less than the per-row path only where a stream holds fewer
than 8 rows of a KV head. The file still defines no kernel: metal-microbench's `decode_attention_shared<D>` instantiates it
(metal-microbench ffa946a, selected by its instance table's attention plan "shared"; docs/kernels.md#decode-attention, its
reproduction target). `torch_mesh/argmax.metal` is the library's selection: the draw at temperature 1
(`mesh_draw`: the greatest of each candidate's logit plus its Gumbel noise, keyed by the row's seed, key and token), the
normaliser (`mesh_normaliser`: a row's greatest logit and its sum of exponentials, log Z) and the retained set
(`mesh_retain`: a head of the 2^k greatest logits and a tail of 2^m of the rest drawn as the draw is, with log Z), each a
row's split reduction finished by the last threadgroup to arrive (`mesh_argmax`'s merge). metal-microbench's engine and
its LiteRT program draw by it; torch programs reach it through `torch_mesh/argmax.py`: `draw(z, seed, keys)` each row's
token, `retain(z, seed, positions, head, tail)` each row's retained set.

`torch_mesh/streamed.py` makes a coded matrix a torch tensor (a wrapper subclass whose aten operations are those
kernels), so a program is written once in torch and queried for what it needs: evaluated, or differentiated by torch's
own autograd, which derives the backward from the forward the program ran.

```python
from torch_mesh import engines, streamed
s = streamed.Streamed("gemma-4-E2B-it-streamed.safetensors")
layer.mlp.down_proj.weight = torch.nn.Parameter(s["model.language_model.layers.3.mlp.down_proj"], requires_grad=False)
y = F.linear(x, s["lm_head"])                     # forward; y.backward(...) runs the transposed consumer
gate, up, down = (s[f"model.language_model.layers.3.mlp.{p}_proj"] for p in ("gate", "up", "down"))
h = engines.ffn(x, gate, up, down)                # the fused couple where no gradient is due, else torch's products
ffn = engines.FFN(gate, up, down, engines=pool)   # neurons shared among engines: gate[a:b], down[:, a:b]
w = s["lm_head"].to("cpu")                        # decoded
```

- `x @ m` is one pipeline whichever way `m` is transposed: the input side's order and scale (and rotation), the
  product contracting the coded rows (`streamed_alu`, `streamed_coop`, `streamed_panel`) or the coded columns
  (`streamed_panel_t`: the same decoded tiles consumed the other way, its operand band-major because a tensor
  operation's strides past 2^16 elements corrupt its rows past the first), the output side's order and scale. So
  `F.linear` and `mm` and their backward (`linear_backward`, `mm` against the untransposed matrix) need nothing else.
- Slicing takes a range of bands (rows) or of each band's tile words (columns, `streamed_dims.pitch`), in multiples of
  32, each order keeping its range; the codes stay where they are. A copy off the GPU decodes. `Streamed` holds an FFN's hidden in its code order (one permutation of its neurons, the FFN unchanged), so
  neurons [a, b) are code rows [a, b): `engines.FFN`'s shares and the Neural Engine share builder slice it as they slice
  a checkpoint. An operation the tensor lacks is an error naming the ones it has.
- Operands are fp16. Another dtype's rows are scaled by powers of two into fp16's range (`streamed_scale_rows`, which
  reads the caller's dtype) and the products finished over the scale in that dtype (`streamed_finish<T>`): exact, and
  two dispatches fewer than a cast each way; a backward's gradients, far below fp16's normal range, keep their
  precision by it. The scaling and the caller's dtype take one dispatch before the product and none after (the finish
  writes that dtype).
- The codes are constants: no gradient reaches them, only through them.

## Performance

- The MPS path is the measured path; async ops hide the crossing behind compute.
- Two-rank direct-exchange all-reduces stripe their crossing over the physical parallel cables according to the
  compiled form. Rank count does not select one cable or establish that a particular collective form is optimal.
- Prepared transport and numerical calls use `rdma/metal-recording.h`, the same
  command representation and submission path as native Metal and LiteRT. A
  prepared invocation owns its immutable bindings through completion. It
  submits the finished command stream without reconstructing its routes.
- FX numerical kernels are recorded before execution. The runtime does not commit a framework
  stream between numerical spans and collective crossings. Initial framework input creation is ordered
  at preparation; it does not create a per-operation handoff.
- `mesh-statistics.h` and `mesh_statistics.py` calculate elapsed time and useful
  rate from ordinary completion observations. Statistics do not drain queues
  or choose when work runs.

- `Function` is the whole-function compiler interface. The ordinary Dynamo backend registration is
  removed: a runtime framework wrapper cannot substitute for a complete declaration.
- Readings and their records: metal-microbench `docs/measurement.md`.

## Prepared engine shares

`engines.Linear` and `engines.FFN` are declarations. Call `declare(rows, dtype, capacities=work_rates)`
before choosing the allocation, then `prepare(input, parts=...)`.
Preparation captures casts, GPU work, host calls and combinations in one FX graph.
GPU and host declarations share the operand schema in `operand.py`; each use carries
its layout requirements into placement before storage and commands are bound.
The prepared share owns its fixed outputs and host views. Append it to `execution_sequence` with the
other function parts. CPU/ANE work uses `MetalScheduleHost` with a native entry point
and declared input/output views. The compiler derives dependencies from overlapping
accesses, using the same records as native engine capture. Consumers need no token
or explicit join. The operation-call allocator, shard loader, Python callback,
thread pool and future joins are removed.
`examples/engines.py` uses the same interface as an application.
Its required `ENGINE_CAPACITIES` JSON object supplies one work rate per named engine for the configured shape.
Missing rates fail before tensors or models are prepared; no equal-rate fallback is used. Zero explicitly excludes
an engine. Accelerate remains one CPU backend; the inventory's core classes do not promise per-class CPU affinity.

`mesh-statistics.h` is the host clock and time/useful-rate arithmetic for native
users; `mesh_statistics.py` binds the same C source for Python and hourly reports.
Neither interface submits numerical work or drains an engine.
