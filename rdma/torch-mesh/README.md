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
- Every `every` steps the ranks' work is gathered to all of them, and each runs the same step of
  `rdma/allocate.py` `Balancer`, so all get the same parts. That step is DFPA: the ranks' measured points, the
  grains given to whoever finishes earliest with them, and a stop once the measurements promise no gain past their
  own resolution. The parts then stand, so jitter does not reshard.
- A rank computes its whole buffer, so its work follows its capacity. Attach with the capacities equal to the
  parts, and attach again when the parts move; that costs one step.
- `partition.balancing(mesh, "tp")` returns what the last window saw.

On the pair (`tools/torch_parallel/tp.py REBALANCE=4`, 16 heads from 8/8), the parts moved to 3/13 and then 4/12,
and stood after two windows. Steps went from 365 to about 248 ms, and the output and gradients still matched one
device.

## Performance

- The MPS path is the measured path. An eager collective costs about 17 us on the pair; collectives compiled into a
  graph cost about 13 us; async ops hide the crossing behind compute.
- On two ranks every all-reduce is the direct exchange: one crossing.
- Readings and their records: metal-microbench `docs/measurement.md`.
