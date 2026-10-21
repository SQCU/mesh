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
f0(x) = tanh(x Aᵀ) and f1(y) = tanh(y B). Any f0 and f1 with the same needs fit the same plan.

## What can run at once

Step t+1 cannot start before step t ends, but the rows can run at the same time, and so can the parts of the width. Give each
Mac the same slice of A's rows and of B's rows. Then each Mac makes its own part of y with no talk, and its own part of
the sum for z. One all-reduce adds the parts, and every Mac then holds the whole z. Each Mac makes x_{t+1}
on its own, from the same z and the same input. This is Megatron's split of two layers (Shoeybi et al., 2019). The
first layer is split by columns and the second by rows. It costs one sum a step, and that is the only crossing on
the chain.

## Program one: the bridge

The bridge is `rdma/mesh-flow`. Each node runs one, kept up by the `io.mesh.bridge` job
([design/transport.md](../design/transport.md) section 2). It owns the Thunderbolt RDMA devices, the links and their
queue pairs. It also owns the region, the shared memory that the devices may read and write. A program marks bytes in
the region as ready to send, and the bridge sends them. When bytes arrive, the bridge marks them as landed. That is
all it does with data.

The bridge does not know what a model, a token, a layer or a step is. It runs no program and takes no request, it
forwards nothing, and it starts no rank and picks no split. It runs whether or not a program is using it. To the bridge, a
language model and this chain look the same: rows of bytes, sent and landed.

## Program two: the mesh collective client

The collective client is a library inside your process. It is `libnccl-mesh` ([rdma/NCCL.md](../rdma/NCCL.md)), and
it is reached through torch-mesh ([rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md)), `mesh_mpi.py` or
`python/mesh`. It rents rows of the region for your tensors and opens groups of ranks. It turns calls like all-reduce,
broadcast and send into work on the bridge's streams. Its waits end when data lands, never when a clock runs out.

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

Rank 0 holds the files, and at the start it sends each rank that rank's rows of A and B. Before each chunk of steps, it
broadcasts that chunk's inputs to every rank. Each step then makes one all-reduce, and nothing else crosses. Rank 0
copies one chunk of outputs to the host at the end of the chunk, not once a step.

The example puts its shared tensors in the region with `torch_mesh.empty`. Collectives then read and write them in
place, with no copy. They also use memory the bridge already holds, so no second copy of anything is made.

## The step kernel

Each step on a rank is one kernel, run through torch-mesh's own calls. `streamed.instance` names an instance of the
template in [examples/chain.metal](../examples/chain.metal), `_stream.pipeline` compiles it, and `_stream.encode` puts
it on torch's stream, so no host waits for it ([examples/chain.py](../examples/chain.py)). The rank's GPU cores run it
as G workers that are all on the GPU at once. They hand work to each other through stamp words in memory. Every wait
has a limit, the patience, so a stuck kernel ends instead of hanging the GPU.

The kernel does three things. First, each worker writes its slice of x = e_t + tanh(z), where z is the last step's sum.
Second, the workers make y = tanh(x Aᵀ) for the rank's rows of A, each worker some of the columns. Third, they make the
rank's part of z from y and its rows of B. The all-reduce then adds the parts on the same stream, and the next step's
kernel starts once that sum has landed.

The number of workers is the largest power of two that fits the GPU's cores and divides the widths. A product is split
along K where a worker has fewer items than simdgroups. How many rows a simdgroup takes at once is chosen by timing each
choice on the rank itself, at the start. Each run prints two times a step: the whole step, and the all-reduce alone. The
gap between them is what the kernel and its hand-offs cost.

## The configuration

The service takes one operand: a configuration file. It names the input, the weights and the output. It gives the rows,
the width, the steps, the chunk and the parts. It also says how many first steps to check, and gives the kernel's
threads, its patience and how many steps each timing takes. A part is how many rows of A and B a rank holds, in rank order. The service never
guesses a part. A balancer that measures the calls writes the parts. Both Macs are always loaded fully.
[examples/chain-service.json](../examples/chain-service.json) is a sample. Its parts are equal only because it is a
sample.

These lines make sample inputs and weights for it:

```python
import numpy as np
g = np.random.default_rng(0)
np.save('chain-input.npy', g.standard_normal((4096, 8, 1024)).astype(np.float16))
np.savez('chain-weights.npz', a=(g.standard_normal((1024, 1024)) / 32).astype(np.float16),
         b=(g.standard_normal((1024, 1024)) / 32).astype(np.float16))
```

## Running it

Start one rank on each node from that node's own checkout, and launch it as torchrun would, with `RANK`, `WORLD_SIZE`,
`MASTER_ADDR` and `MASTER_PORT`. `MASTER_ADDR` is rank 0's address on the cable. Nothing starts a process on another
node.

```
RANK=0 WORLD_SIZE=2 MASTER_ADDR=<rank 0's cable address> MASTER_PORT=29500 \
  ~/.venv-mesh-uv/bin/python examples/chain-service.py examples/chain-service.json
```

Each rank prints one line at the end. It gives the rows, the steps and the seconds. It gives the throughput of all
streams together and of each stream, and the time per step. Rank 0 also prints how far its first steps are from a
plain float64 run of the same rule. To keep the service up, make it a launchd job from
[templates/](../templates/), as the README's "Adding a workload" says.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and the rules a program keeps.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
