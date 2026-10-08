# torch-mesh

torch.distributed's `mesh` backend over [libnccl-mesh](../NCCL.md). MPS tensors run on the library's Metal path:
encoded on torch's MPS stream, each tensor passed as its storage's `MTLBuffer` and offset, no copies. CPU tensors
run on the host path.

## Install

On a node, metal-microbench's `tools/mesh/node-setup.sh` does it. By hand:

```
make -C ~/mesh/rdma all mesh-c PYTHON=~/.venv-mesh-uv/bin/python
uv pip install --python ~/.venv-mesh-uv/bin/python -r ~/mesh/rdma/requirements.txt
uv pip install --python ~/.venv-mesh-uv/bin/python --no-build-isolation -e ~/mesh/rdma/torch-mesh
```

The working set is torch 2.14.0 on CPython 3.12.14 (`rdma/requirements.txt`). Rebuild the extension (the last line)
after every pull.

## Porting a CUDA/NCCL program

```python
dist.init_process_group(backend="mesh", device_id=torch.device("mps", 0))   # or no backend: MPS's default is mesh
```

- `"nccl"` becomes `"mesh"` (or omit it); `"cuda"` becomes `"mps"`; drop `torch.cuda.set_device`. `device_id`
  needs its index (`mps:0`).
- One rank a node. Launch with metal-microbench's `~/.venv-mesh-uv/bin/python tools/mesh/grid.py run --members ...
  -- '{python} prog.py'` (without it: [NCCL.md](../NCCL.md#starting-bridges-by-hand)). The program runs from each
  node's `~/metal-microbench`, so commit and pull it first (metal-microbench docs/nodes.md "Running").
  - It gets torchrun's environment with the bridges up.
  - Rank 0 is a member other than the driver, and the rank-to-node table is printed.
  - Each rank reads its node from its bridge, and the links and their costs from `MESH_LINK_MAP`.
- A bidirectional exchange with one peer goes in one batch: `dist.batch_isend_irecv`, or a coalescing manager.
  Separate `isend`/`irecv` calls in the same order on both sides do not pair; the library reports it, and on MPS
  tensors the next call raises.
- Make subgroups (`new_group`, `DeviceMesh`) after the world group, as torch does: they run on its session, and
  members the cables do not join are joined through the world's other nodes.
- The group's `timeout` is how long a link may stay silent before the library cancels it (`MESH_REMOTE_BOUND`
  overrides it).
- Everything runs on any connected cable graph. Messages between ranks the map does not link (p2p, `all_to_all`,
  `gather`, `scatter`) are forwarded by the nodes between them
  ([NCCL.md](../NCCL.md#routes-between-unlinked-ranks)); the launcher gives the bridges the queue pairs for it.

## What works

Run on the pair's two GPUs, each row checked by one of metal-microbench's `tools/torch_parallel/{collectives,tp,dp,ep,cp}.py`
against one device (2026-10-04); 3 to 6 ranks with CPU tensors on loopback:

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
| functional collectives under torch.compile | one graph: TP, EP |
| DTensor redistribution, DDP, tensor parallelism | |
| context parallelism | ring attention, forward only |
| pipeline parallelism | torch.distributed.pipelining, GPipe and 1F1B, gradients checked |
| 2-D device meshes | pipeline times context (ring attention), expert or tensor parallelism (`mesh2d.py`) |
| expert parallelism | all_to_all_single |

Examples, each checked against one device: `tools/torch_parallel/{tp,dp,ep,cp,collectives}.py` in
metal-microbench. `pair.py --programs tp,dp,ep,cp,collectives` runs them on a link map's nodes.

## What does not

- FSDP2 (`fully_shard`): torch's MPS has no streams or `current_device`, and its copy-in ops have no MPS kernel.
  The project does not shard parameters (FSDP is a last resort among the parallelisms): port to DDP, tensor or
  expert parallelism, or shard optimizer state by hand over `reduce_scatter_tensor` and `all_gather_into_tensor`.
- Streams: torch's MPS has one queue and no `torch.mps.Stream`, so CUDA-stream overlap (DeepSpeed `overlap_comm`,
  Megatron's overlapped gradient reduction, side streams) does not port as written. What overlaps here: an
  `async_op=True` collective waited after the compute it may overlap, or a compiled graph of functional
  collectives (metal-microbench `tools/torch_parallel/crossing.py OVERLAP=1` reads it).
- ReduceOp PREMUL_SUM and the bitwise ops; `monitored_barrier` (it needs a CPU device type the group does not
  list); `split_group`. One call mixing MPS and CPU tensors.
- More than one rank a node.

## Uneven nodes

Stock PyTorch splits every sharded extent equally. `torch_mesh.partition` splits a mesh dimension by parts
proportional to each node's speed, in buffers whose shapes stay fixed:

```python
from torch_mesh import partition
partition.attach(mesh, tp=((5, 11), (5, 11)))    # (capacities, parts) of each coordinate along "tp"
```

A coordinate's capacity is its local buffer's size, and stock kernels compute over all of it, padding included. So
a capacity above the part costs compute: give each coordinate the largest part it will hold. `partition.write(mesh,
tp=parts)` changes the parts between calls within the capacities (no reshape, recompile or reallocation).

A program on raw tensors, not DTensor, splits by its own counts and uses the list collectives with unequal sizes
(`all_gather(list)`, `reduce_scatter(out, list)`, `all_to_all_single` with splits).

The first parts can come from each node's rates (`rdma/allocate.py` `equal_finish`); later ones from the run itself
(below); see
[design/heterogeneity.md](../../design/heterogeneity.md). Parts are indexed by rank, not by node: the group's nodes
in rank order are `dist.distributed_c10d._get_default_group().node`.

On the pair, `tp.py` takes them from the environment: `pair.py --programs tp SP=0 PARTITION=5,11` (sequence
parallelism off with uneven parts) gives rank 0 five sixteenths and rank 1 eleven. Rank 0 is the Mini, so the M5
holds the larger part.

### Parts from the run's own evidence

Nobody needs to know the parts in advance. Call `partition.rebalance` once a step, on every rank alike:

```python
parts, written = partition.rebalance(mesh, "tp", every=8)
if list(parts) != partition.operand(mesh, "tp").parts.tolist():   # moved past the capacities
    mesh = DeviceMesh(device, list(range(world)), mesh_dim_names=("tp",))
    partition.attach(mesh, tp=(parts, parts))                      # attach again at them, and shard again
elif written:
    ...                                                            # moved within them: partition.relay the shards
```

- The first call watches the dimension's group. Every collective on it is then marked as it is issued and once its
  wait has run: on MPS by timing events on torch's stream (in `_stream.mm`, so DTensor's functional collectives are
  included), on CPU by the host clock (`torch_mesh/evidence.py`). A rank's work in a step is the time between its
  collectives, never inside them, so a slow peer never reads as a slow rank.
- Every `every` steps the ranks' stretches (one a collective) are gathered to all of them, and each runs the same
  step of `rdma/allocate.py` `Balancer`, so all get the same parts. It minimises the steps' time, the sum over the
  stretches of the slowest rank's, not the ranks' totals, which can agree while every collective still waits. That step is DFPA: the ranks' measured points, the
  grains given to whoever finishes earliest with them, and a stop once the measurements promise no gain past their
  own resolution. The parts then stand, so jitter does not reshard.
- A rank computes its whole buffer, so its work follows its capacity. Attach with the capacities equal to the
  parts, and attach again when the parts move; that costs one step.
- `partition.balancing(mesh, "tp")` returns what the last window saw.

On the pair (`tools/torch_parallel/tp.py REBALANCE=4`, 16 heads from 8/8), the parts moved to 1/15, 3/13 and then
4/12, and stood. Steps went from 365 to 243-245 ms, and the output and gradients still matched one device.

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
  partial of the whole output; they sum: Megatron's split), the Neural Engine's FFN share one fused graph. A decision
  is an (operation name, row count): a stack's layers of one name share it and pool their evidence, measured where
  the program runs them, between its other operations.
- Engines never time-share a physical unit: each states what it occupies (the GPU, the CPU's cores, the Neural
  Engine, and whatever units Core ML's compute plan places a Neural Engine share's operations on), and two engines
  occupying one unit never hold units of a decision together (`Coupled.forbid`). On both nodes Core ML places every
  computational operation of the shares on the Neural Engine.
- An operation's `tolerance` is checked on each allocation's first call (every non-GPU share against the same share on
  the GPU); an engine that misses it holds none of the decision. The Neural Engine's outputs carry an absolute error
  floor (about 3e-4 RMS for a 3840-wide contraction): accurate on normalized activations (an FFN 1.5e-3), not on small
  ones.
- Nothing is implicit: an operation without an `Engines` operand is torch's own; an engine the configuration names and
  the node lacks is an error; an engine's limits (dtypes, constant weights, row tiles) are data. A share's preparation
  and first run at a row count are no evidence; once an allocation stands, one call in 16 is measured.
- On the pair (metal-microbench `tools/torch_parallel/engines.py`: four layers of a GPU projection, an RMS norm and the
  shared operation, 1024 rows, hidden 3840, 3456 neurons; steady steps compared in alternating blocks): the M5 resolves
  to the GPU alone (its steps within 1-4 % of all-GPU); the M4 Pro's FFN stack stands at GPU and Neural Engine
  [1024, 0, 2432] neurons, a step 59.2 ms against 81.5 all-GPU (1.38x; the FFN 5.3 ms against about 14.7), its linear
  stack at [2048, 0, 1792], 41.0 against 44.2 ms.
- What took the most finding: through coremltools each Neural Engine prediction copied its input and output on the
  CPU, and beside a busy GPU the CPU's performance cores clocked down to 1.3-2.8 GHz (powermetrics), so a 3 ms
  prediction became 9 after a few seconds; no allocation could repair it. The native binding holds 1.44 ms steadily.

## Streamed coded weights

`torch_mesh/streamed.metal` is the decompression of a coded matrix (metal-microbench `tools/model_code.py` exports:
32 x 32 tiles, each its own width and step, row and column orders and scales, a rotation for some) as a producer whose
consumer is an operand: `streamed_pairs` hands each decoded pair of a lane's tile row to a function, `streamed_panels`
decodes a 32 x 256 panel in threadgroup memory and then runs a function on the threadgroup, `streamed_value` and
`streamed_rows` read a product's output (its K shares summed and scaled, or the plain values a crossing summed), and
`streamed_put` writes a product's input in its column order and scale. Every kernel there is an instantiation (the
one-row product, the 16- and 128-row panels on the matrix units, the finish, the input gather and rotation, the GELU of
gate and up, the dense decode), its constants one `streamed_dims` block at buffer 15; metal-microbench's engine
compiles the same source.

```python
from torch_mesh import streamed
s = streamed.Streamed("gemma-4-E2B-it-streamed.safetensors")
y = s["model.language_model.layers.3.mlp.down_proj"].linear(x)            # x [n, 6144] fp16, any n
h = streamed.ffn(x, *(s[f"model.language_model.layers.3.mlp.{p}_proj"] for p in ("gate", "up", "down")))
w = s["lm_head"].dense()                                                  # decoded, in the matrix's coordinates
```

`python -m torch_mesh.streamed check EXPORT [NAME ...]` checks the decode against `model_code.py`'s reference decode
and the products and the FFN against the dense ones (E2B and E4B: decode 3e-6 to 3e-4, products at 1, 8 and 300 rows
3e-4 to 5e-4, the FFN 5e-4 to 8e-4 relative).

## Performance

- The MPS path is the measured path. An eager collective costs about 17 us on the pair; collectives compiled into a
  graph cost about 13 us; async ops hide the crossing behind compute.
- On two ranks every all-reduce is the direct exchange: one crossing.
- Readings and their records: metal-microbench `docs/measurement.md`.
