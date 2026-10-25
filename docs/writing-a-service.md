# Writing a service on the mesh

The mesh is not a language model server. Its library is for any linear algebra, or any SIMD-style function you give
it. Transformers are one user, and they must be fast on it because all code must be fast on it. Its bridge moves
bytes between Macs, and it does not care what the bytes mean.
This page shows a program that uses the mesh and has nothing to do with language. The program is
[examples/chain-service.py](../examples/chain-service.py). It names three kinds of program. For each one, it says what
the program does and what it never does.

## The function

The service runs one rule over a stream of inputs. Call the inputs e₀, e₁, e₂ and so on, where each input is a block
of rows, one row for each independent stream. At each step t:

```
y_t     = f0(x_t)
z_t     = f1(y_t)
x_{t+1} = e_{t+1} + z_t        (x_0 = e_0)
```

Each step needs the step before it, and inside a step, f1 needs what f0 made. This is not an LSTM, since it has no
gates and no cell. It is the plain shape of a filter, a control loop, a simulation or a scan. The example uses
f0(x) = tanh(rot₀(x) Aᵀ) and f1(y) = tanh(rot₁(y) B). Any f0 and f1 with the same needs fit the same plan.

## Rotations

rot₀ and rot₁ are normalized Hadamard rotations of an operand's columns, in blocks: `rotation_blocks` gives each one's
block, a power of two that divides the width. A block of 1 is no rotation, and a block of the whole width is the full
Hadamard. A Hadamard H is linear, symmetric and orthogonal, so rot₀(x) Aᵀ = x (A H)ᵀ and rot₁(y) B = y (H B). Rank 0
holds the whole weights before it distributes them, so it folds each rotation into them there, once, in region tensors
(`rotated` in examples/chain-service.py). rot₁ spans every rank's columns of y, and it folds all the same: each rank's
rows of H B are the rows its block of y meets. The step never rotates anything, and a rotation costs no time a step.
An operand would have to be rotated as it is made only where its parameters are stored rotated and quantized, so that
H cannot be folded into them; this example's parameters are float16.

## What can run at once

Step t+1 cannot start before step t ends, but the rows can run at the same time, and so can the parts of the width. Give each
Mac the same block of A's rows and of B's rows, a block the backend sizes. Then each Mac makes its own part of y with
no talk, and its own part of the sum for z. One all-reduce adds the parts, and every Mac then holds the whole z. Each Mac makes x_{t+1}
on its own, from the same z and the same input. This is Megatron's split of two layers (Shoeybi et al., 2019). The
first layer is split by columns and the second by rows. It costs one sum a step, and that is the only crossing on
the chain.

## Program one: the bridge

The bridge is `rdma/mesh-flow`. Each node runs one, kept up by the `io.mesh.bridge` job
([design/transport.md](../design/transport.md) section 2). It owns the Thunderbolt RDMA devices, the links and their
queue pairs. It also owns the region, the shared memory that the devices may read and write. A program marks bytes in
the region as ready to send, and the bridge sends them. When bytes arrive, the bridge marks them as landed. That is
all it does with data.

The bridge does not know what a model, a token, a layer or a step is. It runs no program and takes no request, and it
starts no rank and picks no split. It runs whether or not a program is using it. To the bridge, a
language model and this chain look the same: rows of bytes, sent and landed.

## Program two: the mesh collective client

The collective client is a library inside your process. It is `libnccl-mesh` ([rdma/NCCL.md](../rdma/NCCL.md)), and
it is reached through torch-mesh ([rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md)), `mesh_mpi.py` or
`python/mesh`. It rents rows of the region for your tensors and opens groups of ranks. It turns calls like all-reduce,
broadcast and send into work on the bridge's streams. Its waits end when data lands, or at their bound: a GPU wait's
loads, or a host wait's platform deadline, 5 s in which nothing moves. Then the call fails as a value.

The client does not own the devices, and it does not know your function or where your inputs come from. It does not
pick how the work is split, and it is not a process of its own. It lives and dies with the service that loads it, and nothing
it holds outlives that service.

## Program three: the data processing service

The service is your program. One copy runs on each node, and every copy runs the same code (SPMD). It holds the
function and its own slice of the weights. It takes input tensors, runs the chain and fills the output tensors. At each
crossing it calls the collective client. Everything about the data is the service's job.

The service never opens a device, and it names no address, port or region. It does not ssh to a peer or start a
bridge.
The rank service in metal-microbench is a service of this kind for a language model, and
`examples/chain-service.py` is one for this chain. Both can be clients of the same bridge at the same time.

## How the example moves its data

Rank 0 draws the inputs and weights in region tensors from the configuration's seed. It folds A and B's
rotations before the communicator exists, so no peer's device wait spans that host work. The same configuration
gives every rank the steps and rows; learning the shape needs no message.

Rank 0 distributes A and B as ragged shards of their rows (`partition.distribute_tensor` with `Shard(0)` on the mesh
dimension `tp`: the backend's scatter, MPI_Scatterv), so each rank holds the block of rows the backend's partition gives
it ([The step](#the-step)). The inputs are one operand, [steps, rows, width], broadcast whole before the first step.
The outputs are one region tensor of the same shape on every rank. Each step writes its tanh(z) there. The service
returns that tensor when the whole call ends. No input, weight or output tensor is written to a file.
Within a step only the step's sum crosses between ranks.

The example puts its shared tensors in the region with `torch_mesh.empty`. Collectives then read and write them in
place, with no copy. They also use memory the bridge already holds, so no second copy of anything is made.

## Memory outside the region

A program builds its buffers from pages of the region. In C it leases them with `ncclMemAlloc`. In torch it makes
region tensors with `torch_mesh.empty`, which lease the same pages. The bridge wired these pages for the devices once,
when it came up. So a collective sends from the pages a tensor lies in and lands bytes in the pages they belong in.

A program should not take large blocks of memory of its own outside the region. They gain it nothing. No peer can be
passed them zero copy, so the program must copy them into the region to send them, and it moves the bytes twice. When
the GPU uses them, macOS wires them in RAM beside the region, where the node has little room. On 2026-10-23 at 03:08,
the chain's process wired 3.64 GiB beside an 86 GiB region on a 96 GiB Studio, and both Studios reset
([etc/bridge/README.md](../etc/bridge/README.md)).

So put weights, KV caches, inputs, outputs and large scratch in the region. Only a kernel's few rows of scratch belong
outside it. `torch_mesh.check_memory()` warns when the process's MPS tensors outside the region come to 1 GiB or more.
Both examples call it once, after they make their tensors.

## The step

The step is a program on torch-mesh's tensor-parallel backend
([rdma/torch-mesh/torch_mesh/partition.py](../rdma/torch-mesh/torch_mesh/partition.py)) over the mesh backend of
torch.distributed ([examples/chain.py](../examples/chain.py), `step`). It holds the math. Everything that crosses ranks
is the library's.

- **Shares.** The rows of A and B are the mesh dimension `tp`, attached with the backend
  (`partition.attach(mesh, tp=width)`, a unit a row). Each rank's block is the backend's (`partition.split`), never
  the configuration's. The program makes no call about them: the collective's balancer sets them.
- **y = tanh(x Aᵀ), column-parallel.** x is replicated and Aᵀ is a `RaggedShard(1)`, so the backend's dispatch gives
  each rank its own columns of y, with no crossing.
- **z = y B, row-parallel.** y's columns and B's rows are the same ragged shards, so each rank's product is a partial
  sum of z (`Partial`). `redistribute` to `Replicate` adds the partials on every rank: the library's all-reduce, the one
  crossing a step.
- **x_{t+1} = e_{t+1} + tanh(z).** Every rank writes tanh(z), the step's output, and makes the next x from it on its
  own.
- **One graph.** The step is recorded once (`torch_mesh.record(step, k)`: torch.compile by the "mesh" backend of k
  steps unrolled in one graph, and of the fewer that k does not divide in a graph of their own) and replayed
  (`Recording.issue`). The backend's dispatch runs when a graph is compiled, never between its ops, and a replay's host
  cost is paid once for k steps. The recorded step (`chain.stepper`) reads e_t from the input operand and writes
  tanh(z) into the output operand at a step counter it holds on the GPU, so every replay is the same graph.
- **The products.** The compile backend runs both products as `torch_mesh::linear`, torch-mesh's small-rows linear
  kernel, one dispatch each at a host cost of about 2 µs where MPS's matmul costs about 12, with y's tanh folded in. It
  reads a weight laid out as nn.Linear's, so the chain holds B transposed, its rows padded to whole 4-element loads.
- **The crossing.** The all-reduce is partition's fused crossing (`partition._fused`), declared once for the run
  (`torch_mesh.plan` with `partition.summed`): z's product publishes itself (`torch_mesh::linear_sum`: linear's
  `linear_publish` writes each rank's part into its send segment through the collective's `mesh_step_write` and
  releases it from its last threadgroup), and `sum_land` adds every rank's part, in rank order, once they land.

## The configuration

The service takes one operand: a configuration file. It gives `steps`, `rows`, `width`, `seed`, the two
`rotation_blocks`, and `steps_per_replay`, the k steps one replay runs. `observations` names the small record of
completed calls used for the next work assignment. It names no tensor files, shares or rank: each rank's block is the
backend's. [examples/chain-service.json](../examples/chain-service.json) is a sample.

`chain.inputs(configuration, steps, rows)` returns input and weight arrays in region memory: standard normal inputs,
and A and B standard normal over the square root of the width. The draw order and float16 values are the same as
in the earlier file-backed runs. They are drawn a chunk at a time through a region scratch. The seed is 0 in the
shipped configurations. Neither drawing these arrays nor consuming the outputs needs a file write.

## Running it

Start one rank on each node from that node's own checkout, and launch it as torchrun would, with `RANK`, `WORLD_SIZE`,
`MASTER_ADDR` and `MASTER_PORT`. `MASTER_ADDR` is rank 0's address on the cable. Nothing starts a process on another
node. A person or an agent starts each run, and nothing keeps the service resident.

```
RANK=0 WORLD_SIZE=2 MASTER_ADDR=<rank 0's cable address> MASTER_PORT=29500 \
  ~/.venv-mesh-uv/bin/python examples/chain-service.py examples/chain-service.json
```

Each rank prints one record at the end. It gives the backend's blocks, the rows, the steps and the whole call's
seconds together with rows a second and useful output bytes a second. The hourly record retains these numbers;
it does not retain tensor contents.

The step's graphs are compiled, every rank's compilation met, and each run once before the timed run, which replays
from the first step again. The whole-run times include the input's broadcast and the first steps' other one-time
costs. The caller runs the function directly. A separate observer uses the host clock and ordinary command-buffer
completion state. It does not add GPU timing events or wait for measurements inside the function.

## The LSTM

The LSTM (examples/lstm.py, lstm-service.py) is a program on torch-mesh's tensor-parallel backend
([rdma/torch-mesh/torch_mesh/partition.py](../rdma/torch-mesh/torch_mesh/partition.py)) over the mesh backend of
torch.distributed. It holds the math. Everything that crosses ranks is the library's. Each step, layer l takes
u = [v, h_l(t - 1)], where v is the step's input for layer 0 and h_{l-1}(t) above it. It makes the gates z = u W_lᵀ + b_l,
split as (i, f, g, o), then c_l = σ(f) c_l + σ(i) tanh(g) and h_l = σ(o) tanh(c_l). The output is the top layer's h.

**Wavefront.** At tick s, layer l works on step s - l. Everything it reads was made at tick s - 1: h_{l-1}(s - l) by
the layer below and h_l(s - l - 1) by itself. So one gather a tick carries every layer, and an output leaves L ticks
after its input.

**Split.** The hidden units are the mesh dimension `tp`, attached with the backend (`partition.attach(mesh, tp=H)`),
which splits them evenly; each rank's block of them is the backend's (`partition.split`), never the configuration's.
Each layer's weights are [4H, K] (w_ih and w_hh side by side), held as [4, H, K]: each gate's units in
turn. The backend's scatter (`partition.distribute_tensor`, `Shard(1)`) gives each rank its units of every gate, and
the rank holds them from load as [4 x its units, K]. Its gates are then torch's linear of u by its own rows: a column
split, so nothing crosses for it (in the recorded graph, torch_mesh::linear). The layers' gates are stacked and their
cells run at once (`Tick.layers`), one kernel for every layer's c and h. Its units' cell state never leaves the rank. The new h slices of every layer are one
ragged tensor along the units, and the backend's allgatherv (`torch_mesh::allgatherv`, partition's op) gathers them to
every rank along that dimension: one fused crossing a tick. This is Megatron's split of a cell by hidden unit. The h slices (two, a tick's and the
last), c and rank 0's output are region tensors made at load (`torch_mesh.empty`), and the blocks hold for the whole
run.

**The recorded tick.** Every tick after the first L - 1 runs every layer, so it is one function of no arguments
(`Tick.stepper`): it reads its input at a step counter held on the GPU, runs each layer and the gather, writes the top
layer's h into the output at the counter less L - 1 (rank 0), and advances the counter. `torch_mesh.record(step, k)`
records it as the chain's step is, k ticks unrolled in one graph and the fewer that k does not divide in a graph of
their own, and the run issues every later tick through it (`Recording.issue`); the first L - 1 ticks, while the
layers above have not begun, run as they are (`Tick.starting`). The service replays it before the run (k ticks and
the rest, so each graph compiles on every rank) and runs the starting ticks once, so every path the run takes is
compiled before it begins, then zeroes the state and the counter. The compiler declares
the complete broadcast, starting gathers and recorded gathers before execution. It
reads destination-writing operators through PyTorch's functionalization metadata,
and constructs the declaration before looking up cached numerical code. A replay's k
gathers are one crossing issued once, each tick its repeat.

**Configuration.** `steps`, `rows`, `seed`, `input_width`, `width` (H), `layers`, `steps_per_replay` and
`observations`. `lstm.inputs(configuration, steps, rows)` returns the input and weights in region memory, using
Torch's initialization: `w_ih_<l>` [4H, its input], `w_hh_<l>` [4H, H] and `b_<l>` [4H] in Torch's gate order.
Rank 0 arranges them before the communicator exists, so no peer's wait spans that host work; every other rank
passes the global shape alone to the scatter. The input is broadcast once. The top layer's full output sequence
stays in its region tensor, returned by rank 0 after the call. Neither example writes tensors to disk.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and the rules a program keeps.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
