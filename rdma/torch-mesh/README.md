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
  -- '{python} prog.py'`. The program runs from each node's `~/metal-microbench`, so commit and pull it first.
  - It gets torchrun's environment with the bridges up.
  - Rank 0 is a member other than the driver, and the rank-to-node table is printed.
  - Each rank reads its node from its bridge, and the links and their costs from `MESH_LINK_MAP`.
- A bidirectional exchange with one peer goes in one batch: `dist.batch_isend_irecv`, or a coalescing manager.
  Separate `isend`/`irecv` calls in the same order on both sides do not pair.
- Make subgroups (`new_group`, `DeviceMesh`) after the world group, as torch does: they run on its session.
- The group's `timeout` is how long a link may stay silent before the library cancels it (`MESH_REMOTE_BOUND`
  overrides it).
- Collectives run on any cable graph. Messages between ranks the map does not link (p2p, `all_to_all`, `gather`,
  `scatter`) need routes the bridges' queue pairs carry ([NCCL.md](../NCCL.md#routes-between-unlinked-ranks));
  otherwise they are refused with the numbers.

## What works

Run on two GPUs (`tools/torch_parallel/collectives.py` checks each):

| works | notes |
|---|---|
| all_reduce | sum, prod, max, min, avg |
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
| expert parallelism | all_to_all_single |

Examples, each checked against one device: `tools/torch_parallel/{tp,dp,ep,cp,collectives}.py` in
metal-microbench. `pair.py --programs tp,dp,ep,cp,collectives` runs them on a link map's nodes.

## What does not

- FSDP2 (`fully_shard`): torch's MPS has no streams or `current_device`, and its copy-in ops have no MPS kernel.
  The project does not shard parameters (FSDP is a last resort among the parallelisms).
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

The parts come from each node's rates (`rdma/allocate.py` `equal_finish`); see
[design/heterogeneity.md](../../design/heterogeneity.md). Parts are indexed by rank, not by node: the group's nodes
in rank order are `dist.distributed_c10d._get_default_group().node`.

On the pair, `tp.py` takes them from the environment: `pair.py --programs tp SP=0 PARTITION=5,11` (sequence
parallelism off with uneven parts) gives rank 0 five sixteenths and rank 1 eleven. Rank 0 is the Mini, so the M5
holds the larger part.

## Performance

- The MPS path is the measured path. An eager collective costs about 17 us on the pair; collectives compiled into a
  graph cost about 13 us; async ops hide the crossing behind compute.
- On two ranks every all-reduce is the direct exchange: one crossing.
- Readings and their records: metal-microbench `docs/measurement.md`.
