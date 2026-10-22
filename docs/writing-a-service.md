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

Rank 0 holds the files. At the start it reads the input's shape and broadcasts two numbers, the steps and the rows, so
every rank knows them. Then it sends each rank that rank's rows of A and B.

The inputs are one operand: a block of rows for each step. They move through a ring of slots in the region, one slot a
step. Each rank has two rings, one for the inputs and one for the sums.

The inputs cross a block at a time, a block early. They depend on nothing in the chain, so nothing makes them wait.
Before the first step of each block, rank 0 copies the next block's inputs from its file into its ring, and broadcasts
them into every rank's ring. Both go on the GPU's own queue, and the host does not wait for them. The host waits for
that broadcast just before the next block's first step, the first thing that reads it. A step then has two host calls:
its kernel and one all-reduce of its part of z. Nothing else crosses during a block.

The library lands each block's broadcast sooner than that wait, though. A call finishes every call issued before it, so
the block's first all-reduce lands the next block's inputs too. That all-reduce waits for the whole block's bytes to
arrive, once a block.

At the end of each block, rank 0 puts tanh of that block's sums, and a copy of them, on the GPU's queue as well. The
copy writes straight into the output file's memory. No step waits for the host, and the next section says when the
host waits for the GPU.

The example puts its shared tensors in the region with `torch_mesh.empty`. Collectives then read and write them in
place, with no copy. They also use memory the bridge already holds, so no second copy of anything is made.

## The rings

The input ring is two blocks deep. While the GPU works through one block, the next block's inputs arrive in the
other. The sum ring is one block deep: a block's sums are copied out at the block's end, before the next block writes
over them.

How long is a block? The configuration gives `ring_bytes`, the bytes of the region the two rings may hold on each node.
If each ring can hold the whole stream, a block is the whole stream, and no slot is used twice. If not, a block is a third
of the slots that fit: two blocks for the inputs and one for the sums. If `ring_bytes` cannot hold three slots, the
service stops with an error. It never picks a depth of its own.

The host may run ahead of the GPU, but never by more than the input ring. At the end of each block it puts a mark on
the GPU's queue. At the start of a block, it waits for the mark two blocks back, so one block is still queued.

There is a second bound, and today it is the tighter one. Metal's queue holds at most 64 command buffers not yet done,
and torch makes the host wait when it is full. Today each publication and each copy ends a command buffer, so the host
gets about 32 steps ahead and the ring's mark never waits. The host's time a step then includes that wait. Once the
library ends a command buffer only when the GPU needs one, the ring's mark is the bound.

## The step kernel

Each step on a rank is one kernel, run through torch-mesh's own calls. `streamed.instance` names an instance of the
template in [examples/chain.metal](../examples/chain.metal), `_stream.pipeline` compiles it, and `_stream.encode` puts
it on torch's stream, so no host waits for it ([examples/chain.py](../examples/chain.py)). The rank's GPU cores run it
as G workers that are all on the GPU at once. They hand work to each other through stamp words in memory. Every wait
has a limit, so a stuck kernel ends instead of hanging the GPU. Once a wait has failed, each later step's kernel
reads that at its start and ends at once, so queued steps never spin again. The limit is the library's own, `NCCL_MESH_POLLS` loads
(`_stream.polls`), which holds a GPU at most the platform's 5 seconds. It is not a setting.

The kernel does three things. First, each worker writes its slice of x = e_t + tanh(z), where z is the last step's sum.
Second, the workers make y = tanh(x Aᵀ) for the rank's rows of A, each worker some of the columns. Third, they make the
rank's part of z from y and its rows of B. The all-reduce then adds the parts on the same stream, and the next step's
kernel starts once that sum has landed. A step's all-reduce follows torch-mesh's one commit rule, the same for every
caller (torch-mesh README, Performance). While the GPU still holds enough of the stream's buffers, the all-reduce joins
the open command buffer instead of paying for a commit of its own. The next step's all-reduce, or the host's next wait,
commits that buffer.

G is the number of the GPU's cores, so the whole GPU works: one worker a core. The columns come in tiles of 8, and
each worker takes a fair share of them. If there are n tiles, each worker takes n/G of them, rounded down, and the first
few workers take one more, so the shares differ by one tile at most. y's columns over the share are split this way, and
so are z's columns over the width. The width and the rank's share must be multiples of 8, with at least one tile for
every core. If one is not, the service stops with an error that names the nearest values that would run.

The rows are not built into the kernel. The row count is a word the host passes with each step. The kernel's work
items are pairs: a group of rows and a tile of columns. Only the size of a row group, RG, is built in, since it sets
how many registers a simdgroup uses. So one compiled kernel serves any row count that is a multiple of 8 times RG, up
to the capacity the configuration gives. A product is split along K where a worker has fewer items than simdgroups.
That split is a word too, worked out once for each row count.

Each rank builds one instance of the kernel. Its RG is 1, so a work item is one tile of 8 rows by one tile of 8
columns. Its threads are the most the GPU builds this kernel for. The rank compiles the kernel at 32 threads, one
simdgroup, and doubles the count until the GPU refuses to build the pipeline. It keeps the last count that built. That
compile reads what the GPU can run. Nothing is timed, and no choice is made by speed.

## The configuration

The service takes one operand: a configuration file. It names the input, the weights and the output. It gives the
width and the parts. It gives `row_capacity`, the most rows a step may take, and `ring_bytes`, the region bytes for the
two rings. It also says how many first steps to check. The kernel's threads are the most the GPU builds, its row
group is one tile, and its wait limit is the library's, so the configuration names none of them. The input file says how many steps and rows there are, so the configuration does not. Every value
must be there. The service has no defaults, and a missing value is an error.

A part is how many rows of A and B a rank holds, in rank order. The service never guesses a part. A balancer that
measures the calls writes the parts. Both Macs are always loaded fully.
[examples/chain-service.json](../examples/chain-service.json) is a sample. Its parts are equal only because it is a
sample. Its `ring_bytes` holds three blocks of 512 steps, so the sample's rings wrap many times.

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
node. A person or an agent starts each run, and nothing keeps the service resident.

```
RANK=0 WORLD_SIZE=2 MASTER_ADDR=<rank 0's cable address> MASTER_PORT=29500 \
  ~/.venv-mesh-uv/bin/python examples/chain-service.py examples/chain-service.json
```

Each rank prints one line at the end. It gives the parts, the rows, the steps, the block, the two rings' depths and
the seconds. It gives the rows a second of all streams together and the steps a second of each stream. It gives four
times a step: the whole step, the host's own time (with any wait inside torch), the host's wait on the ring's mark, and
the all-reduce alone. It gives the bytes a step's two collectives move, and those bytes a second. Rank 0 also checks
two windows of steps against a plain float64 run of the same rule: the first steps, and the first steps of the last
block, which start from rank 0's own output. The second window reads slots the ring has used before.

The line also splits the run in two. The first block carries costs a run pays once, such as the library compiling
its kernels the first time a collective runs. So the line gives the host's time to issue the first step, and the
first block's time. It then gives the time a step of the later blocks takes, and the rows a second they reach. A block's
end is the moment the host's wait on its mark returns: the host issues ahead of the GPU, so it is already waiting
when the block ends, and no wait is added to take the time. The whole-run times include the first block; these do
not.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and the rules a program keeps.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
