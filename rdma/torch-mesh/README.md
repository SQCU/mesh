# torch-mesh

torch.distributed's `mesh` backend over [libnccl-mesh](../NCCL.md).

## Install

A mesh checkout holds the library and this package, built in place (rdma/NCCL.md "Build and run": `make install`, then
`make -C rdma modules`, which runs `rdma/mesh_c_build.py` and `setup.py build_ext --inplace` here where their inputs changed), and a program run with the user's torch environment,
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
  exchange their nodes and the nodes their bridges reach, and the group keeps the map its init made of them. That id
  is the store's one use.
- A subgroup (`new_group`) is a communicator of its own, made in any order, before or after the world group; no two
  communicators hold one stream. Each communicator acquires a stream on every participating physical cable. A link
  has up to four, subject to its endpoints' actual lane capacity; a required busy link refuses another communicator
  (EBUSY) until a stream becomes free
  ([NCCL.md](../NCCL.md#rules-a-program-keeps)).
- Every wait carries a deadline set when it is made (design/eager-execution.md, assumption A4):
  the library's own waits theirs ([NCCL.md](../NCCL.md#bounded-waits)). Past it the process ends, naming the wait and
  its deadline.
- A group's collectives run where a path between its ranks joins every two of them, each rank's bridge reaching the
  next one's node. Routes cross communicator members. A group
  whose ranks no such path joins fails at its first collective that crosses a link
  ([NCCL.md](../NCCL.md#routes-between-unlinked-ranks)).
- Memory ([NCCL.md](../NCCL.md#memory)): A program's CPU tensor in the region is
  `torch_mesh.empty(shape, dtype=..., device="cpu")`.
- Region CPU tensors: every CPU tensor's bytes are a lease's pages, `torch_mesh.empty(shape, dtype=..., device="cpu")`
  and torch's own alike. `_stream` makes torch's CPU allocator the region's as it loads (`c10::SetCPUAllocator`, above
  torch's default). A lease the region cannot meet is torch's out-of-memory error, and a tensor's release returns its
  lease (`ncclMemFree`). Memory torch did not allocate stays where it is (`torch.from_numpy`'s, made before the load).
- Region MPS tensors: every MPS tensor is a lease's Metal buffer (`ncclMeshMemBuffer`), `torch_mesh.empty(shape,
  dtype=..., device="mps")` and torch's own (its factories', eager temporaries', Inductor's, `.to("mps")`'s) alike:
  `_stream` makes torch's MPS allocator the region's as it loads (`torch_mesh/allocator.mm`), in place over the fixed
  instance torch 2.14's factories call (`at::mps::getIMPSAllocator`, no registered or pluggable allocator for MPS), and
  its import fails where that instance is not of torch's class `at::mps::MPSAllocator`. A lease the region cannot meet
  is torch's out-of-memory error. A tensor's release returns its lease once the command buffer open on torch's stream
  at the release, which the release commits, has completed; `ncclMemFree` then defers reuse until
  both its local GPU users and all registered local DMA reads have completed, including reads the bridge has not yet
  posted. GPU completion alone does not establish source retirement, and no receipt collective
  is needed to establish local DMA-read completion. `torch_mesh.address(t)` is where a
  tensor's bytes lie in the region (0: outside it).
  `torch_mesh.check_memory()` warns where the process's MPS tensors outside the region come to 1 GiB or more
  ([docs/writing-a-service.md](../../docs/writing-a-service.md#memory-outside-the-region)).
  Only torch's pinned CPU memory and MPS
  tensors made before `_stream` loaded lie in torch's own heaps. Raw transport operands must satisfy their
  declared contiguous region layout.

## What works

### Numerical work and physical writes

Every communicator retains its physical cables, including parallel cables to the same peer.

### Prepare and call a whole function

Unknown numerical
lowerings fail preparation; there is no eager or ATen execution substitute. The current native emitter
supports Metal.

A statistics thread
observes ordinary completion with a host clock; it does not submit work or drain a framework stream.
See the [partial tensor and FX reference](../../docs/partial-tensors-and-fx.md).

## What does not

- FSDP2 (`fully_shard`): torch's MPS has no streams or `current_device`, and its copy-in ops have no MPS kernel.
  The project does not shard parameters (FSDP is a last resort among the parallelisms).
- Streams: torch's MPS has one queue and no `torch.mps.Stream`, so CUDA-stream overlap (DeepSpeed `overlap_comm`,
  Megatron's overlapped gradient reduction, side streams) does not port as written.
- `init_device_mesh("mps", ...)` and `DeviceMesh("mps", ...)`: torch 2.14's DeviceMesh setup calls
  `torch.mps.is_initialized` and `set_device`, which torch's MPS module does not have. Form an MPS mesh from the
  groups instead, which skips that setup: `DeviceMesh.from_group(dist.group.WORLD, "mps", mesh_dim_names=("tp",))`
  (a list of groups and `mesh=` for more dimensions). torch-mesh patches nothing of torch.
- ReduceOp PREMUL_SUM and the bitwise ops; `monitored_barrier`; `split_group`. One call mixing MPS and CPU tensors.
- More than one rank a node.

## Uneven nodes

Stock PyTorch splits every sharded extent equally. `torch_mesh.partition` is a partition operand on raw tensors: a
mesh dimension's extents split among its ranks. Its callers are this repository's examples/chain* and lstm*.

```python
from torch_mesh import partition
partition.attach(mesh, tp=16)                    # the units of every extent split along "tp"; the backend's parts
blocks = partition.split(mesh, "tp", N)          # each rank's block of an extent N, in rank order
w = partition.distribute_tensor(weight, mesh, [Shard(0)])
```

- A partitioned dimension may be any of its mesh's: a 2-D mesh's second dimension partitioned with its first
  replicated or splitting another tensor dimension.

- The program declares the units and no parts. The backend splits them evenly at first, each part its rank's
  capacity, and the parts move on the run's own evidence; see [design/heterogeneity.md](../../design/heterogeneity.md).
  Parts are indexed by rank, not by node.

### Parts solved on the collective as it is

- A placement has two throughputs, the aggregate and each stream's, so the calls trace a frontier rather than a best:
  `allocate.pareto(rows, coordinate=("ctx_in", "ctx_out"))` keeps, per workload coordinate, the call records (each a
  call's own `aggregate_tok_s` and `per_sequence_tok_s`, its `program`, `backend` and batch `bs`) that no other beats in
  both, and which backends dominate others over the points both ran; from calls alone (no solo run, no fit).
  Reproduction target (metal-microbench
  output_data/confirm-20260924/summary.json): at 16384/1024, bs 4, 564.6 aggregate and 141.1 per-sequence tok/s.

## Performance

- The MPS path is the measured path.
- Rank count does not select one cable or establish that a particular collective form is optimal.
- A
  prepared invocation owns its immutable bindings through completion. It
  submits the finished command stream without reconstructing its routes.
- FX numerical kernels are recorded before execution. The runtime does not commit a framework
  stream between numerical spans and collective crossings. Initial framework input creation is ordered
  at preparation; it does not create a per-operation handoff.
- `mesh-statistics.h` and `mesh_statistics.py` calculate elapsed time and useful
  rate from ordinary completion observations. Statistics do not drain queues
  or choose when work runs.

- Readings and their records: metal-microbench `docs/measurement.md`.

`mesh-statistics.h` is the host clock and time/useful-rate arithmetic for native
users; `mesh_statistics.py` binds the same C source for Python and hourly reports.
Neither interface submits numerical work or drains an engine.
