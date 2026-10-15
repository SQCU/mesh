# torch-mesh

torch.distributed's `mesh` backend over [libnccl-mesh](../NCCL.md). MPS tensors run on the library's Metal path:
encoded on torch's MPS stream, each tensor passed as its storage's `MTLBuffer` and offset, no copies. CPU tensors
run on the host path.

## Install

The user's mesh checkout holds the library and its Python bindings (`bin/mesh-user.sh`, rdma/NCCL.md "Build and run"),
and the user's torch environment, `~/.venv-mesh-uv` (`bin/mesh-user.sh`): `rdma/requirements.txt` (torch 2.14.0 on CPython 3.12.14), this package built against the
installed library whenever either changes, and the library's directory on the environment's path (`mesh.pth`), so a
program imports `mesh`, `torch_mesh` and the mesh's modules with nothing of its own.

## Porting a CUDA/NCCL program

```python
dist.init_process_group(backend="mesh", device_id=torch.device("mps", 0))   # or no backend: MPS's default is mesh
```

- `"nccl"` becomes `"mesh"` (or omit it); `"cuda"` becomes `"mps"`; drop `torch.cuda.set_device`. `device_id`
  needs its index (`mps:0`).
- One rank a node, launched as torchrun launches one (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`); each node's
  bridge is its resident service, so a launch starts no bridge.
- A group is made by NCCL's init (`ncclCommInitRank`, its unique id passed through the group's store): the ranks
  exchange their nodes alone, and each session of the group reads the map its bridge keeps when the session opens.
- A bidirectional exchange with one peer goes in one batch: `dist.batch_isend_irecv`, or a coalescing manager.
  Separate `isend`/`irecv` calls in the same order on both sides do not pair; the library reports it, and on MPS
  tensors the next call raises.
- A subgroup (`new_group`) is a communicator of its own, made in any order, before or after the world group.
- A link that lands nothing for the library's one bound (`mesh.h` `MESH_SILENT_NS`, 10 s) is cancelled, and the
  collective waiting on it fails; the group's `timeout` is not the library's to read.
- Everything runs on any connected cable graph. Messages between ranks the map does not link (p2p, `all_to_all`,
  `gather`, `scatter`, a collective's crossings) are forwarded by the bridges on their route
  ([NCCL.md](../NCCL.md#routes-between-unlinked-ranks)).
- Memory ([NCCL.md](../NCCL.md#memory)): the backend's own CPU tensors (a non-contiguous CPU tensor's contiguous copy, a
  barrier's tensor, the flattened operands of the list `all_gather` and `reduce_scatter`) are leases of the node's
  bridge's window (`backend.blank`, over `mesh.lease`). A program's CPU tensor in the window is
  `torch.frombuffer(mesh.lease(nbytes), dtype=...)`. MPS tensors are torch's MPS allocator's; the Metal path copies
  them into the window's rings, and placing them in the window is not done here.

## What works

| works | notes |
|---|---|
| all_reduce | sum, prod, max, min, avg on int8, uint8, int32, int64, float16, bfloat16 and float32; a non-contiguous view |
| all_gather_object, all_to_all of lists, blocking send/recv | |
| broadcast, reduce | |
| all_gather (list and into a tensor) | |
| reduce_scatter (list and tensor) | |
| gather, scatter | any per-rank sizes |
| all_to_all, all_to_all_single | uneven splits, empty inputs |
| send, recv, isend, irecv, batch_isend_irecv | |
| barrier, async_op works, coalescing | |
| functional collectives under torch.compile | one graph |
| DDP | |
| ring attention | `torch_mesh._mps.ring(group, q, k, v)` on the sequence shards, forward only, in place of `context_parallel` |
| pipeline parallelism | torch.distributed.pipelining, GPipe and 1F1B, gradients checked |
| expert parallelism | all_to_all_single |

## What does not

- FSDP2 (`fully_shard`): torch's MPS has no streams or `current_device`, and its copy-in ops have no MPS kernel.
  The project does not shard parameters (FSDP is a last resort among the parallelisms): port to DDP, tensor or
  expert parallelism, or shard optimizer state by hand over `reduce_scatter_tensor` and `all_gather_into_tensor`.
- Streams: torch's MPS has one queue and no `torch.mps.Stream`, so CUDA-stream overlap (DeepSpeed `overlap_comm`,
  Megatron's overlapped gradient reduction, side streams) does not port as written. What overlaps here: an
  `async_op=True` collective waited after the compute it may overlap, or a compiled graph of functional
  collectives.
- `DeviceMesh("mps")`, so DTensor, tensor and context parallelism and 2-D meshes on MPS: torch 2.14's DeviceMesh
  calls `torch.mps.is_initialized` and `set_device`, which torch's MPS module does not have, and DTensor has no
  sharding of MPS's `linear_backward` or SDPA. torch-mesh patches none of them.
- ReduceOp PREMUL_SUM and the bitwise ops; `monitored_barrier` (it needs a CPU device type the group does not
  list); `split_group`. One call mixing MPS and CPU tensors.
- More than one rank a node.

## Uneven nodes

Stock PyTorch splits every sharded extent equally. `torch_mesh.partition` is a partition operand on raw tensors: a
mesh dimension's extents split among its ranks by parts proportional to each node's speed, the program holding its
share and calling the list collectives with unequal sizes (`all_gather(list)`, `reduce_scatter(out, list)`,
`all_to_all_single` with splits: MPI's v-collectives). DTensor shards as stock torch does.

```python
from torch_mesh import partition
partition.attach(mesh, tp=((5, 11), (5, 11)))    # (capacities, parts) of each coordinate along "tp"
blocks = partition.split(mesh, "tp", N)          # each rank's block of an extent N, in rank order
```

- A coordinate's capacity is the most it holds; a rank that keeps its share in a buffer of its capacity, its valid
  elements first, keeps its shapes while the parts move within the capacities. `partition.write(mesh, tp=parts)`
  changes the parts between calls, the same on every rank.
- The first parts can come from each node's rates (`rdma/allocate.py` `equal_finish`), later ones from the run itself
  (below); see [design/heterogeneity.md](../../design/heterogeneity.md). Parts are indexed by rank, not by node.

### Parts from the run's own evidence

Nobody needs to know the parts in advance. Call `partition.rebalance` once a step, on every rank alike:

```python
parts, written = partition.rebalance(mesh, "tp", every=8)   # written: moved within the capacities
```

- The first call watches the dimension's group. Every collective on it is then marked as it is issued and once its
  wait has run: on MPS by timing events on torch's stream (in `_stream.mm`, functional collectives included), on CPU
  by the host clock (`torch_mesh/evidence.py`). A rank's work in a step is the time between its collectives, never
  inside them, so a slow peer never reads as a slow rank.
- Every `every` steps the ranks' stretches (one a collective) are gathered to all of them, and each runs the same
  step of `rdma/allocate.py` `Balancer`, so all get the same parts. It minimises the steps' time, the sum over the
  stretches of the slowest rank's, not the ranks' totals, which can agree while every collective still waits. That
  step is DFPA: the ranks' measured points, the grains given to whoever finishes earliest with them, and a stop once
  the measurements promise no gain past their own resolution. The parts then stand, so jitter does not reshard.
- Parts that move past a capacity are the program's to attach again (capacities and parts both), at the cost of its
  buffers' new shapes.
- `partition.balancing(mesh, "tp")` returns what the last window saw.

## Uneven engines within a node

A node's engines are uneven too, which is fastest depends on the operation's shape, and engines running at once slow
each other. `torch_mesh/engines.py` resolves an operation's backend as one joint decision: its units shared among the
node's configured engines (`mps`: torch's GPU queue; `cpu`: fp32 products through Accelerate, the CPU's matrix units;
`ane`: the Neural Engine, Core ML graphs of 1x1 convolutions with constant weights through `rdma/mesh-coreml.m`, which
predicts on torch's shared MPS memory with no CPU copy), each holding its shard of the weights resident, the shares
running at once on unified memory, and `allocate.Coupled` choosing the allocation: each engine's time a fitted
function of every engine's share (its own work, slowed multiplicatively by each co-runner's), the call's the slowest
engine's plus its set's measured overhead, minimised over the whole lattice of allocations (no per-engine moves to
conflict), a design about the start until the model is determined, a measured allocation outranking the model.

```python
from torch_mesh import engines
pool = engines.Engines.configured()                       # MESH_ENGINES="mps,cpu,ane"; None where none is named
ffn = engines.FFN(gate, up, down, engines=pool, name="ffn", tolerance=1e-2)   # without engines: torch's own
y = ffn(x)
engines.rebalance(pool)                                    # every few steps
```

- Operations: `Linear` shares its output features (they concatenate), `FFN` its intermediate neurons (each engine a
  partial of the whole output; they sum: Megatron's split), the Neural Engine's FFN share one fused graph. An `FFN`'s
  operands may be any tensors, coded matrices among them ("Streamed coded weights"): a share is a slice, the GPU's
  run by `engines.ffn` (an operand type's fused couple where no gradient is due), another engine's decoded by its copy. A decision
  is an (operation name, row count): a stack's layers of one name share it and pool their evidence, measured where
  the program runs them, between its other operations.
- Engines never time-share a physical unit: each states what it occupies (the GPU, the CPU's cores, the Neural
  Engine, and whatever units Core ML's compute plan places a Neural Engine share's operations on), and two engines
  occupying one unit never hold units of a decision together (`Coupled.forbid`).
- An operation's `tolerance` is checked on each allocation's first call (every non-GPU share against the same share on
  the GPU); an engine that misses it holds none of the decision. The Neural Engine's outputs carry an absolute error
  floor (about 3e-4 RMS for a 3840-wide contraction): accurate on normalized activations (an FFN 1.5e-3), not on small
  ones.
- Nothing is implicit: an operation without an `Engines` operand is torch's own; an engine the configuration names and
  the node lacks is an error; an engine's limits (dtypes, constant weights, row tiles) are data. A share's preparation
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
`streamed_coopmap`), `streamed_panel` and `streamed_panel_t` (prefill and the transposed product: eight tiles decoded
into threadgroup memory, one operation a 256-wide panel); and the finish, the input gather, rotation (writing the 1-bit
tables) and Hadamard butterfly (`streamed_hadamard`, a threadgroup a row: H_N by structured.metal's stages on every
block at once, H_K mixing the blocks as each coordinate is put, the RMS norm foldable), the GELU of gate and up, the
dense decode and a range prefetch (`streamed_touch`). `streamed_value` and `streamed_rows` read a product's output (its
K shares summed and scaled, or the plain values a crossing summed), and `streamed_gather` writes a product's input in its
column order and scale; the constants are one `streamed_dims` block at buffer 15. Which instance a slice takes at its
rows on a node (its kernel, simdgroups, bands, K shares, tiles in flight, tables; the couple or the products apart) is
the instance fit's choice (`streamed.py`, metal-microbench docs/kernels.md#the-instance-fit): the argmin of a model of
the node's times over each candidate's units, its coefficients fitted to the node's observations (its evidence file,
MESH_INSTANCES or `~/.cache/mesh/instances.json`; `python -m torch_mesh.streamed fit`), read by `Matrix.dispatch` and,
as a table of a member's slices (`python -m torch_mesh.streamed resolve`), by metal-microbench's engine, which compiles
the same source. `streamed_value` and `streamed_rows` take the partials as any pointer type, so a finish run inside its
product reads them `coherent(device)`.

`torch_mesh/attention.metal` is decode attention as one algorithm over a KV type (`mesh_attention<A>`): a task's chunks
(the KV type's scores of each lane's positions, the chunk's online softmax of R rows, the KV type's value product), the
position groups' sum and the KV type's finish (its parts' merge and store). metal-microbench's engine instantiates it over
its fp16 pages (`decode_paged`) and its recorder over LiteRT's int8 window (`k7_window`, K7); a new KV layout is a new type
(metal-microbench docs/kernels.md#decode-attention). `torch_mesh/argmax.metal` is the library's selection: the draw at temperature 1
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
- On two ranks every all-reduce is the direct exchange: one crossing.
- Readings and their records: metal-microbench `docs/measurement.md`.
