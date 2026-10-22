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
the fused all-reduce of its part of z, which only writes the crossing's records, and its kernel. Nothing else crosses
during a block.

The library lands each block's broadcast sooner than that wait, though. A call finishes every call issued before it, so
the block's first fused all-reduce lands the next block's inputs too. That landing waits for the whole block's bytes
to arrive, once a block.

Each step's kernel lands the step before's sum, and on rank 0 it also writes tanh of that sum, the step before's output,
into a staging buffer one block deep. So a block's last output is written by the next block's first step, and one step
after each block ends, rank 0 puts a copy of the block's outputs on the GPU's queue. After the last step, one more
dispatch of the kernel lands the last sum, and the last block is copied the same way. The copy writes straight into the
output file's memory. No step waits for the host, and the next section says when the host waits for the GPU.

The example puts its shared tensors in the region with `torch_mesh.empty`. Collectives then read and write them in
place, with no copy. They also use memory the bridge already holds, so no second copy of anything is made.

## The rings

The input ring is two blocks deep. While the GPU works through one block, the next block's inputs arrive in the
other. The sum ring is one block deep. It holds each step's part of z, which the next step's kernel reads as its own
part when it lands the sum, before a later step writes over it.

How long is a block? The configuration gives `ring_bytes`, the bytes of the region the two rings may hold on each node.
If each ring can hold the whole stream, a block is the whole stream, and no slot is used twice. If not, a block is a third
of the slots that fit: two blocks for the inputs and one for the sums. If `ring_bytes` cannot hold three slots, the
service stops with an error. It never picks a depth of its own.

The host may run ahead of the GPU, but never by more than the input ring. After each block's outputs are copied, one
step into the next block, it puts a mark on the GPU's queue. At the start of a block, it waits for the newest mark, the
one two blocks back, so one block is still queued.

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

The kernel does three things, and it runs the step's crossing itself (torch-mesh's `fused`: rdma/nccl.h
`ncclMeshFusableRooted`). First, each worker lands its slice of the last step's sum. It waits for the peer's part of z
to land, adds its own part, and writes its slice of x = e_t + tanh(z) (on rank 0, its slice of tanh(z) too). Second, the
workers make y = tanh(x Aᵀ) for the rank's rows of A, each worker some of the columns. Third, they make the rank's part
of z from y and its rows of B, and publish it. Each worker stores its slice of the part into the crossing's slot. The
last worker to finish releases the slot, so the bridge sends it. Within a block, no kernel of the library's runs between
two steps. The wait and the release are the library's own steps (`ncclMeshMetalSteps`), which the kernel calls. Each
wait is bounded, and the release is one fence and the slot's ready words. A step's publication follows torch-mesh's one
commit rule, the same for every caller (torch-mesh README, Performance). While the GPU still holds enough of the
stream's buffers, it joins the open command buffer instead of paying for a commit of its own. The next step's
publication, or the host's next wait, commits that buffer.

The kernel is compiled for its shape: the width K, the rank's share S and the row count. Each row count the service
plans gets its own instance (`chain.form`). Every choice in it follows from three inputs: the shape, the capability
file (below), and what Metal reports of the device and of the compiled kernel. Nothing is timed, and no choice is made
by speed. A measurement of the sampled shapes checks the rules; it is never read back by the code.

- **Tiles.** Columns and rows come in tiles of the matrix unit's edge (8 on Apple7 and later), so the width, the share
  and the row count must be positive multiples of it.
- **Workers.** G is the GPU's cores, or the larger stage's column tiles where there are fewer. Each worker takes a fair
  share of a stage's tiles: n/G rounded down, the first few one more. y's columns over the share are split this way,
  and so are z's columns over the width. Where a stage has fewer tiles than workers, the workers without one wait at its
  stamps.
- **Simdgroups.** A work item is a group of RG row tiles by one column tile, over K's tiles. A worker runs as many
  simdgroups as the busier stage can keep working, each on one K tile at least, up to the device's limit, rounded
  down to a power of two. The SIMD width is the compiled pipeline's (`threadExecutionWidth`), since no document states
  it.
- **Row groups.** RG is the largest power of two that divides the row tiles and keeps that many simdgroups working. A
  larger RG loads each weight tile once for more rows. If the compiled pipeline then holds fewer threads than the
  simdgroups need (`maxTotalThreadsPerThreadgroup`, which its registers set), RG halves.
- **The K split.** Where a worker has fewer items than simdgroups, a product is split along K, ks ways, a power of two.
  The ks partials of an item are summed as a tree in threadgroup memory: log2(ks) rounds, one barrier each, a batch of
  RB row tiles' partials a round, RB as many as the threadgroup memory holds. The split is a word the host passes,
  worked out once for each row count.

## The capability

A capability file states what Apple documents about a GPU family: its threads and threadgroup memory a threadgroup,
the alignment of that memory, and the edge of its matrix unit. [etc/compute/apple9.json](../etc/compute/apple9.json)
covers the M4-series, and [etc/compute/apple10.json](../etc/compute/apple10.json) the M5-series. Each names its sources,
Apple's Metal Feature Set Tables and the Metal Shading Language Specification. The service is given the file for the
node it runs on. It checks the file against what Metal reports of the device and stops where they differ. What no
document states, the SIMD width and how many threads a compiled kernel holds, is read from the compiled pipeline.

## The configuration

The service takes one operand: a configuration file. It names the input, the weights and the output. It gives the
width and the parts. It gives `row_capacity`, the most rows a step may take, and `ring_bytes`, the region bytes for the
two rings. It also says how many first steps to check, and names the node's capability file. The kernel's form
follows from the shape and that file, and its wait limit is the library's, so the configuration names none of them. The input file says how many steps and rows there are, so the configuration does not. Every value
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

Each rank prints one line at the end. It gives the parts, the rows, the steps, the block, the two rings' depths and the
seconds. It gives the rows a second of all streams together and the steps a second of each stream. It gives four times a
step: the whole step, the host's own time (with any wait inside torch), the host's wait on the ring's mark, and a plain
all-reduce alone (the library's own kernels, not fused), the crossing the fused step no longer pays. It gives the bytes
a step's two collectives move, and those bytes a second. Rank 0 also checks two windows of steps against a plain float64
run of the same rule: the first steps, and the first steps of the last block, which start from rank 0's own output. The
second window reads slots the ring has used before.

The line also splits the run in two. The first block carries costs a run pays once, such as the library compiling its
kernels the first time a collective runs. So the line gives the host's time to issue the first step, and the first
block's time. It then gives the time a step of the later blocks takes, and the rows a second they reach. The first
block's end is the moment the host's wait on its mark returns, and that mark comes one step into the next block, so the
later steps counted are the ones after it: the host issues ahead of the GPU, so it is already waiting when the block
ends, and no wait is added to take the time. The whole-run times include the first block; these do not.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and the rules a program keeps.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
