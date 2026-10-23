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
block, a power of two that divides the operand's width (x's width for rot₀, the sum of the parts for rot₁). A block of 1
is no rotation, and a block of the whole width is the full Hadamard. A Hadamard H is linear, symmetric and orthogonal,
so rot₀(x) Aᵀ = x (A H)ᵀ and rot₁(y) B = y (H B). Rank 0 holds the whole weights before it sends each rank its part, so
it folds each rotation into them there, once (`rotated` in examples/chain-service.py). rot₁ spans every rank's columns
of y, and it folds all the same: each rank's rows of H B are the rows its part of y meets. The step kernel never rotates
anything, and a rotation costs no time a step.
An operand would have to be rotated as it is made only where its parameters are stored rotated and quantized, so that
H cannot be folded into them; this example's parameters are float16.

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

Rank 0 holds the files. Before the communicator exists it loads A and B and folds their rotations in, in float16:
seconds of host work at a large width, which no peer's wait may span (a wait on the GPU ends at its stream's bound,
about the platform's 5 s), while the other ranks wait in torch's rendezvous. Then it reads the input's shape and
broadcasts two numbers, the steps and the rows, so every rank knows them, and sends each rank that rank's rows of A
and B.

The inputs are one operand: a block of rows for each step. They move through a ring of slots in the region, one slot a
step. Each rank has two rings, one for the inputs and one for the sums.

The inputs cross a block at a time, a block early. They depend on nothing in the chain, so nothing makes them wait.
Before the first step of each block, rank 0 copies the next block's inputs from its file into its ring, and broadcasts
them into every rank's ring. The copy's command buffer ends before the broadcast is issued: the NIC reads a GPU's plain
stores only once their command buffer has ended (metal-microbench docs/measurement.md, 2026-09-29), and the broadcast
sends the ring's own pages. The broadcast goes on the GPU's own queue, and the host does not wait for it. The host waits for
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
`ncclMeshFusableRooted`). First, each worker lands its slice of the last step's sum. It waits for every peer's part of
z to land, adds them to its own part, and writes its slice of x = e_t + tanh(z) (on rank 0, its slice of tanh(z) too). Second, the
workers make y = tanh(x Aᵀ) for the rank's rows of A, each worker some of the columns. Workers read each other's x and y
within one dispatch, which Metal makes visible only through `coherent(device)` memory, so those loads and stores carry
it: each lane moves the two elements its 8 x 8 matrix fragment holds (MLX's coordinates, steel/gemm/mma.h), since
`simdgroup_load` and `simdgroup_store` take no coherent pointer. Third, they make the rank's part
of z from y and its rows of B, and publish it. Each worker writes its slice of the part straight into the step's send
tensor, each element once through the library's `mesh_step_write` (a system-coherent store: the bridge sends as soon as
`ready` passes, while the kernel's command buffer still runs), and the last worker to finish releases it to every peer: one fence and one store of each holding's `ready`,
after which the bridge sends the tensor's own pages down each link. Nothing is copied. Within a block, no kernel of the library's runs between two
steps. The wait and the release are the library's own steps (`ncclMeshMetalPrelude`), which the kernel calls. Each wait
is bounded. A step's publication follows torch-mesh's one
commit rule, the same for every caller (torch-mesh README, Performance). While the GPU still holds enough of the
stream's buffers, it joins the open command buffer instead of paying for a commit of its own. The next step's
publication, or the host's next wait, commits that buffer.

The kernel is compiled for its shape: the width K, the rank's share S and the row count. Each row count the service
plans gets its own instance (`chain.form`). Every choice in it follows from three inputs: the shape, the capability
file (below), and what Metal reports of the device and of the compiled kernel. Nothing is timed, and no choice is made
by speed. A measurement of the sampled shapes checks the rules; it is never read back by the code.

- **Tiles.** Columns and rows come in tiles of the matrix unit's edge (8 on Apple7 and later), so the width, the share
  and the row count must be positive multiples of it.
- **Workers.** G is half the GPU's cores, or the larger stage's column tiles where there are fewer. The workers wait
  for each other at their stamps, so a step runs only while all G are on the GPU. Other processes' GPU work shares
  the GPU with it. Measured 2026-10-23 (metal-microbench output_data/rank0-20261023):
  - The GPUs come in partitions (ioreg `GPUConfigurationVariable`: `num_mgpus` and `core_mask_list`): of 8 cores on
    the Mini and the Studios, of 10 on the MacBook's M5 Max. The Mini's 16 cores are 2 partitions, a Studio's 64 are
    8, and the M5 Max's 40 are 4. Its half, 20, leaves two partitions free but splits the 128 tiles 6 or 7 each.
  - With nothing of ours running, other GPU work comes in bursts. On the Mini, a burst lasts up to 90 ms, several
    times a second. On m5ult1, each burst is under 10 ms, ten times a second.
  - With G at 16, 15 or 12 on the Mini, so on both of its partitions, a step stopped for 15 to 90 ms several times a
    second. With G at 8, so on one partition, it never did.
  - The same showed when other kernels spun on the Mini's GPU (2026-10-23 14:37-14:43): at G 16 a step took about
    580 us, and at G 8 it took 31.
  - On m5ult1, a second process's small kernels beside 64 spinning workers waited at most 1.5 ms. On the Mini, beside
    12 or 16, a few waited 0.1 to 2.5 s.

  So on the Mini a step on every partition waits whenever other work takes the GPU, while a step that leaves one
  partition free runs beside that work. Half the cores leaves one of the Mini's two partitions free. On a Studio,
  32 is the largest power of two that leaves a partition free, so the width's 128 tiles divide evenly. In 15
  interleaved pairs, the two-Studio chain averaged 28.8 us at 32 workers and 29.1 at 64. In 4 pairs after the Mini's
  restart, the three-node chain read 28.9 to 29.6 us at half the cores and 30.0 to 31.3 at all of them. The LSTM's
  workers do not wait for each other (below), so its G stays at the cores.

  Each worker takes a fair share of a stage's tiles: n/G rounded down, the first few one more. y's columns over the
  share are split this way, and so are z's columns over the width. Where a stage has fewer tiles than workers, the
  workers without one wait at its stamps.
- **Simdgroups.** A work item is a group of RG row tiles by one column tile, over K's tiles. A worker runs as many
  simdgroups as the busier stage can keep working, each on one K tile at least, up to the device's limit, rounded
  down to a power of two. The SIMD width is the compiled pipeline's (`threadExecutionWidth`), since no document states
  it.
- **Row groups.** RG is the largest power of two that divides the row tiles, keeps that many simdgroups working, and
  leaves each stage's K split as it is at RG = 1. A larger RG loads each weight tile once for more rows. But with the
  simdgroups fixed, it halves the work items, so the K split doubles: each simdgroup's chain of multiply-adds stays as
  long, and the sum gains a tree round. Where the split is already 1 (weights streamed from memory, wide layers), the
  reuse is all gain; where it would deepen the tree, RG stays. If the compiled pipeline holds fewer threads than the
  simdgroups need (`maxTotalThreadsPerThreadgroup`, which its registers set), RG halves.
- **The K split.** Where a worker has fewer items than simdgroups, a product is split along K, ks ways, a power of two.
  The ks partials of an item are summed as a tree in threadgroup memory: log2(ks) rounds, one barrier each, a batch of
  RB row tiles' partials a round, RB as many as the threadgroup memory holds. The split is a word the host passes,
  worked out once for each row count.

## The crossing

A crossing is one send to each peer and one landing from each (torch-mesh's `fused.exchange(send, landings,
polls)`): the step's part of z goes from the rank's send tensor to each of the n - 1 other ranks, and each peer's part
lands in its own slot of the rank's landing tensor; the next step's kernel reads the n - 1 slots and adds them to its
own part. This is the one-shot all-reduce (each rank reads every peer's whole part and sums locally: one crossing a step,
(n - 1) parts each way, over n - 1 links at once). It is the form for a part this small; the two-shot form
(reduce-scatter, then all-gather: two crossings, each 1/n of a part) is the form for large parts. Every tensor that crosses is a region tensor (`torch_mesh.empty`): a range of rows of the node's page table, each
row a page. The bridge sends each page from the page its row names and lands each page in the page the receiver's row
names, one page a SEND (design/pages-and-functions.md; design/transport.md section 14). The weights rank 0 sends each
rank are region tensors too. The service lays its ring of sums on pages (`chain.paged`: each slot whole pages, its rows
first), and its input ring's blocks start on pages. One landing tensor serves every step: the library posts a step's
receive only once the rank's own publication of that step has stored `ready`, after the kernel that read the landing
before. A crossing of any size fuses; the 61,440-byte limit is gone with the bridge's rings.

## The capability

A capability file states what Apple documents about a GPU family: its threads and threadgroup memory a threadgroup,
the alignment of that memory, and the edge of its matrix unit. [etc/compute/apple7.json](../etc/compute/apple7.json)
covers the M1-series, [apple8.json](../etc/compute/apple8.json) the M2-series, [apple9.json](../etc/compute/apple9.json)
the M3- and M4-series, and [apple10.json](../etc/compute/apple10.json) the M5-series: every family with the matrix unit
the step kernel multiplies on, Apple7 and later. Each names its sources, Apple's Metal Feature Set Tables and the Metal
Shading Language Specification, and its `family` as a number: n of Metal's `MTLGPUFamilyApple<n>`.

The configuration names the directory that holds the files, and each rank takes from it the file of its own GPU,
`apple<n>.json`, where n is the newest Apple family its Metal device supports (`chain.capability`). torch-mesh's
`_stream.device_facts` finds n by asking Metal for each family from Apple1 up, since a GPU of a family supports every
earlier one's. So every node runs the same configuration, and the ranks of one run may hold different GPUs, each with
its own file. Where Metal reports a limit of the device (its threads and threadgroup memory a threadgroup), the rank
takes the device's figure. What no document states, the SIMD width and how many threads a compiled kernel holds, is
read from the compiled pipeline.

## The configuration

The service takes one operand: a configuration file. It names the input, the weights and the output. It gives the
width and the parts. It gives `row_capacity`, the most rows a step may take, and `ring_bytes`, the region bytes for the
two rings. It also gives the two rotations' blocks, and names `capabilities`, the
directory of capability files, from which each rank takes its own GPU's. Its paths are read from the directory the
service runs in, the checkout's root in the sample. The kernel's form follows from the shape and the capability file,
and its wait limit is the library's, so the configuration names none of them. The input file says how many steps and
rows there are, so the configuration does not. Every value must be there. The service has no defaults, and a missing
value is an error.

A part is how many rows of A and B a rank holds, in rank order. The service never guesses a part. A balancer that
measures the calls writes the parts. Every Mac is always loaded fully.
[examples/chain-service.json](../examples/chain-service.json) is a sample. Its parts are equal only because it is a
sample. Its `ring_bytes` holds three blocks of 512 steps, so the sample's rings wrap many times.

`chain.inputs(configuration, steps, rows)` makes sample inputs and weights for it from a fixed seed: standard normal
inputs, and A and B standard normal over the square root of the width, drawn a chunk at a time, so no float64 copy of a
matrix is held. `chain.reference(e, a, b, carried)` is the function itself, in its operands' precision: from the inputs
of a run of steps, the weights as rank 0 folds them, and the sum carried into the first step, it gives those steps'
outputs.

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
a rank's crossing moves a step (its part out to each of the n - 1 peers, and each peer's part in), and those bytes a
second.

The line also splits the run in two. The first block carries costs a run pays once, such as the library compiling its
kernels the first time a collective runs. So the line gives the host's time to issue the first step, and the first
block's time. It then gives the time a step of the later blocks takes, and the rows a second they reach. The first
block's end is the moment the host's wait on its mark returns, and that mark comes one step into the next block, so the
later steps counted are the ones after it: the host issues ahead of the GPU, so it is already waiting when the block
ends, and no wait is added to take the time. The whole-run times include the first block; these do not.

## The LSTM

The same template runs an L-layer LSTM (examples/lstm.metal, lstm.py, lstm-service.py). Each step, layer l takes
u = [v, h_l(t - 1)], where v is the step's input for layer 0 and h_{l-1}(t) above it. It makes the gates z = u W_l + b_l,
split as (i, f, g, o), then c_l = σ(f) c_l + σ(i) tanh(g) and h_l = σ(o) tanh(c_l). The output is the top layer's h.

**Wavefront.** At tick s, layer l works on step s - l. Everything it reads was made at tick s - 1: h_{l-1}(s - l) by
the layer below and h_l(s - l - 1) by itself. So one crossing a tick carries every layer, a tick depends only on the
tick before, and the kernel needs no barrier between its workers. An output leaves L ticks after its input.

**Split.** Each rank owns a share of every layer's hidden units: their 4 gate columns of W_l, their bias, and their
cell state, which never leaves the rank. The shares are configuration (`shares`) and may be ragged. A tick's crossing
sends the rank's new h slices, every layer's, to each peer, and lands each peer's slices: an all-gather, through the
same `fused.exchange` the chain sums over. This is Megatron's split of a cell by hidden unit. Splitting by input rows
instead, with an all-reduce of gate partials, would send about 4n times as many bytes and repeat the cell update on
every rank.

**Reads in place.** A layer's u is never assembled. Its columns are segments read where they are: the input slot, this
rank's previous send slot, and each peer's landing slot. They are in rank order, each a whole number of 8-column tiles.
The matmul's loader (`Segments`) finds a tile's segment and loads each lane's two fragment elements system-coherent
(the NIC wrote the landings). The weights are tiled once at load, and their gate columns are laid out unit-major, so an
8-column tile holds two units' (i, f, g, o).

**Epilogue.** A lane holds (i, f) or (g, o) of one unit and row. One `simd_shuffle_xor` pairs them, and the (i, f) lane
updates c and stores h through `mesh_step_write` into the tick's send slot. A layer that has not started (s < l) stores
zeros there, its h(-1). Rank 0 writes the top layer's landed h to the output staging ring, as the chain does.

**Configuration.** `input`, `output` and `ring_bytes` as the chain's. Also `input_width`, `layers`,
`shares` (one count of hidden units a rank, multiples of 8, their sum the width) and `capabilities`. The weights file
holds `w_ih_<l>` [4H, its input], `w_hh_<l>` [4H, H] and `b_<l>` [4H] in torch's gate order, so `torch.nn.LSTM` is
the reference. `lstm.inputs(configuration, steps, rows)` makes sample ones from a fixed seed, with torch's
initialization. `weights` is a directory of one .npy file an array. Before the communicator exists, rank 0 reads them
memory-mapped and writes each rank's gate rows, already tiled, into CPU region tensors, 64 tiles at a time. It then
sends each rank its tiles, and each rank receives them into an MPS region tensor the kernel reads. Nothing the size of
a weight matrix lives outside the region: a Studio's region wires 82 of its 96 GiB, and the rest is all that other
memory has.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and the rules a program keeps.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
