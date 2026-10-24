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
function and its own slice of the weights. It reads the inputs, runs the chain and writes the outputs. At each
crossing it calls the collective client. Everything about the data is the service's job.

The service never opens a device, and it names no address, port or region. It does not ssh to a peer or start a
bridge.
The rank service in metal-microbench is a service of this kind for a language model, and
`examples/chain-service.py` is one for this chain. Both can be clients of the same bridge at the same time.

## How the example moves its data

Rank 0 holds the files. Before the communicator exists it reads A and B memory-mapped and folds their rotations in, into
region tensors, in float16: seconds of host work at a large width, which no peer's wait may span (a wait on the GPU ends
at its stream's bound, about the platform's 5 s), while the other ranks wait in torch's rendezvous. Then it reads the
input's shape and broadcasts two numbers, the steps and the rows, so every rank knows them.

Rank 0 distributes A and B as ragged shards of their rows (`partition.distribute_tensor` with `Shard(0)` on the mesh
dimension `tp`: the backend's scatter, MPI_Scatterv), so each rank holds the block of rows the backend's partition gives
it ([The step](#the-step)). The inputs are one operand, [steps, rows, width]. Rank 0 copies them from its file into a
region tensor and broadcasts it whole to every rank before the first step. The outputs are one operand of the same shape
on every rank. Each step writes its tanh(z) there, and after the last step rank 0 copies it into the output file's
memory. Within a step only the step's sum crosses between ranks.

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
  the configuration's: the blocks the last evaluation on these nodes left, else an even split. Nothing about them is
  chosen within an evaluation. Once the run has ended, every rank calls `partition.evaluated(mesh, 'tp')`, and the
  backend's balancer sets the next evaluation's blocks from this one's evidence.
- **y = tanh(x Aᵀ), column-parallel.** x is replicated and Aᵀ is a `RaggedShard(1)`, so the backend's dispatch gives
  each rank its own columns of y, with no crossing.
- **z = y B, row-parallel.** y's columns and B's rows are the same ragged shards, so each rank's product is a partial
  sum of z (`Partial`). `redistribute` to `Replicate` adds the partials on every rank: the library's all-reduce, the one
  crossing a step.
- **x_{t+1} = e_{t+1} + tanh(z).** Every rank writes tanh(z), the step's output, and makes the next x from it on its
  own.

## The configuration

The service takes one operand: a configuration file. It names the input, the weights (a directory holding `a.npy` and
`b.npy`, each [width, width]) and the output. It gives the width and the two rotations' blocks. Its paths are read from
the directory the service runs in, the checkout's root in the sample. The input file says how many steps and rows there
are, so the configuration does not. It names no shares and no rank: each rank's block is the backend's. Every value must
be there. The service has no defaults, and a missing value is an error.
[examples/chain-service.json](../examples/chain-service.json) is a sample.

`chain.inputs(configuration, steps, rows)` makes sample inputs and weights for it from a fixed seed: standard normal
inputs, and A and B standard normal over the square root of the width. Each is composed in a CPU region tensor
(`chain.region`), drawn into it a chunk at a time through a region scratch, and written to its file from there.

## Running it

Start one rank on each node from that node's own checkout, and launch it as torchrun would, with `RANK`, `WORLD_SIZE`,
`MASTER_ADDR` and `MASTER_PORT`. `MASTER_ADDR` is rank 0's address on the cable. Nothing starts a process on another
node. A person or an agent starts each run, and nothing keeps the service resident.

```
RANK=0 WORLD_SIZE=2 MASTER_ADDR=<rank 0's cable address> MASTER_PORT=29500 \
  ~/.venv-mesh-uv/bin/python examples/chain-service.py examples/chain-service.json
```

Each rank prints one line at the end. It gives the backend's blocks, the rows, the steps and the seconds. It gives the
rows a second of all streams together and the steps a second of each stream. It gives three times a step: the whole
step, the host's own time to issue it, and a plain all-reduce of the step's sum alone (the library's), the step's
crossing without its math.

The whole-run times include the input's broadcast and the first steps' one-time costs, such as the library compiling
its kernels. The line also gives the steady step and the rows a second it reaches: the GPU's time from the middle step
to the last, read from two timing events on torch's stream, over the steps between them. No wait is added to take it.

## The LSTM

The LSTM (examples/lstm.py, lstm-service.py) is a program on torch-mesh's tensor-parallel backend
([rdma/torch-mesh/torch_mesh/partition.py](../rdma/torch-mesh/torch_mesh/partition.py)) over the mesh backend of
torch.distributed. It holds the math. Everything that crosses ranks is the library's. Each step, layer l takes
u = [v, h_l(t - 1)], where v is the step's input for layer 0 and h_{l-1}(t) above it. It makes the gates z = u W_lᵀ + b_l,
split as (i, f, g, o), then c_l = σ(f) c_l + σ(i) tanh(g) and h_l = σ(o) tanh(c_l). The output is the top layer's h.

**Wavefront.** At tick s, layer l works on step s - l. Everything it reads was made at tick s - 1: h_{l-1}(s - l) by
the layer below and h_l(s - l - 1) by itself. So one gather a tick carries every layer, and an output leaves L ticks
after its input.

**Split.** The hidden units are the mesh dimension `tp`, attached with the backend (`partition.attach`), and each
rank's block of them is the backend's (`partition.split`), never the configuration's. A rank holds its units' gate rows
of W_l and b_l, unit-major, as a ragged shard (`RaggedShard(0)`: torch's column-parallel split of a linear layer), and
its units' cell state, which never leaves the rank. The gate product is torch's linear on the replicated u and the
sharded weights, so the backend's dispatch gives each rank its own gate columns. The new h slices of every layer are
then one ragged tensor (`RaggedShard(2)`), and `full_tensor()` gathers them to every rank along that dimension: the
backend's allgatherv, one crossing a tick.
This is Megatron's split of a cell by hidden unit. The h slices (two, a tick's and the last), c and rank 0's
output are region tensors made at load (`torch_mesh.empty`). The blocks hold for the whole run, one evaluation:
once it has ended, `partition.evaluated` gives the backend's Balancer the run's own evidence, and the next run's
attach takes the blocks it sets (design/data-model.md section 9).

**Blocks.** The program declares the units, `partition.attach(mesh, tp=H)`, and the backend splits them evenly at
first. After each tick it calls `partition.rebalance(mesh, 'tp')`. That measures each rank's own work between its
collectives, and every 8 ticks the Balancer moves the blocks by the residual between the ranks and attaches `tp`
again. The tick then follows (`Tick.follow`, `partition.follow`): the gate rows and cell state that change rank
cross once, by one all-to-all. The program never chooses a block.

**Configuration.** `input`, `weights`, `output`, `input_width`, `width` (H) and `layers`. The weights directory holds
`w_ih_<l>` [4H, its input], `w_hh_<l>` [4H, H] and `b_<l>` [4H] in torch's gate order, one .npy file an array.
`lstm.inputs(configuration, steps, rows)` makes sample ones with torch's initialization. Rank 0 reads them
memory-mapped and sends each rank its gate rows, into MPS region tensors; the input is broadcast once into a region
tensor.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and the rules a program keeps.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
