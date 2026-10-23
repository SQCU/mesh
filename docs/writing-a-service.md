# Writing a service on the mesh

The mesh is not a language model server. Its library is for any linear algebra, or any SIMD-style function you give
it, and transformers are one user. The operator, 2026-10-17 (design/rubrics.md D1): "the universal comes before the
particular and we use the particular ot show off that the universal is correct"; 2026-10-19 (D10): "no part of any model
or eval we study may ever be slow". Its bridge moves bytes between Macs, and it does not care what the bytes mean.
This page shows a program that uses the mesh and has nothing to do with language. The program is
[examples/chain-service.py](../examples/chain-service.py). It names three kinds of program. For each one, it says what
the program does and what it never does.

## The function

The service runs one recurrence over a stream of inputs. Call the inputs e₀, e₁, e₂ and so on, where each input is a block
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
broadcast and send into work on the bridge's streams. A host wait ends when its data lands, or when the link is lost
(its count reads ~0) and the call then fails as a value; no clock ends it. A GPU wait ends when its data lands or at its bound of loads, and its kernel goes on with the bytes it finds.

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

Rank 0 holds the files. Before the communicator exists it loads A and B in float16, while the other ranks wait in
torch's rendezvous. Then it reads the input's shape and broadcasts two numbers, the steps and the rows, so every rank
knows them, and sends each rank that rank's rows of A and B.

The inputs are one operand, a block of rows for each step: one region tensor of every step's rows
(`chain.landed`), the pages of the whole stream held before the first step, as the page table holds every page a run
will use (design/pages-and-functions.md, Setup). Rank 0 writes its file's rows into it through a CPU view of the same
pages (`torch_mesh.cpu_view`) and broadcasts it on the host path, once, after every rank has compiled its kernels: the
inputs land before the chain starts, so no step waits on them, and each rank's last wait before the chain is on these
bytes, not on a barrier. A step then has two host calls: the start of its crossing (`exchange.start`), which presents
the step's entries and writes the slot's table, and its kernel's three dispatches. Nothing else crosses during the
chain.

Each step's kernel lands the step before's sum, and on rank 0 it also writes tanh of that sum, the step before's output,
into its row of the output, a region tensor of every step's rows. After the last step, one more dispatch, the kernel's
first stage alone, lands the last sum and writes the last output. After the run rank 0 copies the outputs into the
output file's memory, once. No step waits for the host, and the next section says when the host waits for the GPU.

The example puts its shared tensors in the region with `torch_mesh.empty`. Collectives then read and write them in
place, with no copy. They also use memory the bridge already holds, so no second copy of anything is made. The pages
a step's crossing sends from and lands in are the exchange's own, which the library allocates
([rdma/NCCL.md](../rdma/NCCL.md#persistent-exchanges)).

## The pages

Every operand is held whole: the inputs and the outputs of every step, the weights' shares and, for the plain
all-reduce timing after the run, one tensor of a step's rows. No slot is used twice and no index wraps. The step's own
parts of z are the exchange's slots: a step's part is read as its own part by the next step's kernel, and the library
gives the next start other pages while that kernel has not run, releasing a slot by its use count
([rdma/NCCL.md](../rdma/NCCL.md#persistent-exchanges)).

The host may run ahead of the GPU. Metal's queue holds at most 64 command buffers not yet done, and torch makes the host
wait when it is full. A publication ends a command buffer under the commit rule, so the host gets as many steps ahead
as the queue holds the library's buffers, and the exchange holds a slot for each start the GPU has not finished. The host's time a step includes that wait. After the first step
the service records a timing event, which it reads after the run (below).

## Memory outside the region

The region is the node's memory for mesh programs. The bridge holds it and the devices read and write it, so every
lease of it (`torch_mesh.empty`, `ncclMemAlloc`) is wired already, and a collective reads and writes a tensor there
where it lies.

Memory a program allocates for itself outside the region gains it nothing:

- A collective given such a tensor stages it through the library's own pages: one copy in and one copy out, with those
  pages leased while its plan lives ([rdma/NCCL.md](../rdma/NCCL.md), Memory). Zero copy is lost and the bytes moved
  double.
- When the GPU runs on it, macOS wires it beside the region. On an M5, one kernel writing a 4 GiB MPS tensor moved
  4.2 GiB from free to wired (2026-10-23).
- A node whose region leaves little beside it has nowhere to put it. On 2026-10-23 at 03:08, both Studios ran the
  chain with an 86 GiB region of their 96 GiB, and the chain's process wired 3.64 GiB outside it. Wired memory reached
  92.6 GiB with 915 pages free. watchdogd, launchd and the chain's threads waited on the file system for 90 seconds,
  and the watchdog reset both nodes. The region is now sized from those readings (82 GiB,
  [etc/bridge/m5ultra-headless.toml](../etc/bridge/m5ultra-headless.toml)), which still leaves a rank about 3.6 GiB.

So allocate every large tensor with `torch_mesh.empty`: operands, outputs, shares of weights, and any scratch of more
than a few MiB. Only a kernel's few rows of scratch belong outside. Read inputs from files into region tensors through
`torch_mesh.cpu_view`, as the example does.

`torch_mesh.memory()` gives this process's GPU memory outside the region (`outside`), the region's leases (`region`)
and what the node can give without pressure (`available`: the kernel's `kern.memorystatus_level`, the figure each bridge
publishes for its node, `mesh.nodes()`). `torch_mesh.check_memory()` warns, with an `OutsideRegionWarning`, where
`outside` is more than `available`: the program will probably take its node down with it. torch-mesh checks this
itself when an exchange is made and when a collective is first given a tensor outside the region, which it also warns
about as a copy.

## The step kernel

Each step on a rank is three dispatches, one for each stage, of one kernel template, run through torch-mesh's own
calls. `streamed.instance` names the template's instance for each stage in [examples/chain.metal](../examples/chain.metal),
`fused.pipeline` compiles it after the library's device header (`fused.header()`), and `exchange.encode` puts the three
back to back in torch's open compute encoder, so no host waits for them ([examples/chain.py](../examples/chain.py)).
That encoder is serial (Metal's `MTLDispatchTypeSerial`): each dispatch ends before the next begins, and its stores are
seen by the next. So a dispatch boundary orders the stages, and no worker waits for another. The kernel spins on
nothing but the landing wait. The operator, 2026-10-21: "keep only the landing wait, one acquire fence, a poll bound for
the node". That wait and its bound are the library's: the kernel names no bound, keeps no error word and no patience of
its own. The operator, 2026-10-22: "error states can be made unreachable and proven to be unreachable".

The step's crossing is composed into its kernel ([rdma/NCCL.md](../rdma/NCCL.md#persistent-exchanges)). The first
stage, bound to land the step before's start, calls `mesh_enter`: each worker waits for every peer's part of z to land.
It adds them to its own part (`mesh_own`, `mesh_landing`), writes its slice of x = e_t + tanh(z) (on rank 0, its slice
of tanh(z) too), and calls `mesh_leave`, which marks the landed pages read once the last worker is done. The second
makes y = tanh(x Aᵀ) for the rank's rows of A, each worker some of the columns. The third, bound to publish the step's
start, makes the rank's part of z from y and its rows of B. x and y cross a dispatch boundary, so they are plain
memory, and `simdgroup_load` reads them as the products' operands. Each worker of the third stage writes its slice of
the part through `mesh_send`, each element once (a system-coherent store: the bridge sends as soon as the slot's
`published` passes, while the kernel's command buffer still runs), into the start's own pages. Then `mesh_leave`: each
worker fences its stores and counts its arrival, and the last to arrive releases the part to every peer, one system
fence and then a store of the slot's `published`, after which the bridge sends those pages down each link. That is the order the release needs
and no more: each worker's stores, its arrival, one release fence, the stores. Nothing is copied. No kernel of the
library's runs between two steps. A step's publication is committed as torch-mesh commits every caller's
(torch-mesh README, Performance). While the GPU still holds enough of the stream's buffers, it joins the open command
buffer instead of paying for a commit of its own. The next step's publication, or the host's next wait, commits that
buffer.

The kernel guards no buffer: no start writes pages another start still holds. A start's send part is written again
only once the bridge has sent it, and its landing areas are landed in again only once the kernel that read them has
ended; until then each start takes another slot.

The kernel is compiled for its shape: the width K, the rank's share S and the row count. Each row count the service
plans gets its own three instances (`chain.form`). Every choice in it follows from three inputs: the shape, the capability
file (below), and what Metal reports of the device and of the compiled kernel. Nothing is timed, and no choice is made
by speed. A measurement of the sampled shapes checks the choices; it is never read back by the code.

- **Tiles.** Columns and rows come in tiles of the matrix unit's edge (8 on Apple7 and later), so the width, the share
  and the row count are positive multiples of it; `chain.form` stops with an error on any other.
- **Workers.** G is the GPU's cores, or the larger stage's column tiles where there are fewer. Each worker takes a fair
  share of a stage's tiles: n/G rounded down, the first few one more. y's columns over the share are split this way,
  and so are z's columns over the width. Where a stage has fewer tiles than workers, the workers without one have nothing
  to do in its dispatch.
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

A crossing is one send to each peer and one landing from each (torch-mesh's `fused.Exchange`, a persistent exchange:
made once, started each step): the step's part of z goes from the start's send part to each of the n - 1 other ranks,
and each peer's part lands in its own landing area of the start; the next step's kernel reads the n - 1 parts and adds
them to its own. This is the one-shot all-reduce (each rank reads every peer's whole part and sums locally: one crossing
a step, (n - 1) parts each way, over n - 1 links at once). It is the form for a part this small; the two-shot form
(reduce-scatter, then all-gather: two crossings, each 1/n of a part) is the form for large parts. The exchange is
declared at the row capacity, and each start moves the step's rows. Every page that crosses is a row of the node's page
table: the exchange's slots are leases, and the weights rank 0 sends each rank are region tensors (`torch_mesh.empty`).
The bridge sends each page from the page its row names and lands each page in the page the receiver's row names, one
page a SEND (design/pages-and-functions.md; design/transport.md section 14).

## The capability

A capability file states what Apple documents about a GPU family: its threads and threadgroup memory a threadgroup,
the alignment of that memory, and the edge of its matrix unit. [etc/compute/apple7.json](../etc/compute/apple7.json)
covers the M1-series, [apple8.json](../etc/compute/apple8.json) the M2-series, [apple9.json](../etc/compute/apple9.json)
the M3- and M4-series, and [apple10.json](../etc/compute/apple10.json) the M5-series: every family with the matrix unit
the step kernel multiplies on, Apple7 and later. Each names its sources, Apple's Metal Feature Set Tables and the Metal
Shading Language Specification, and its `family` as a number: n of Metal's `MTLGPUFamilyApple<n>`.

The configuration names the directory that holds the files, and each rank takes from it the file of its own GPU,
`apple<n>.json`, where n is the newest Apple family its Metal device supports (`chain.capability`). torch-mesh's
`fused.device()` finds n by asking Metal for each family from Apple1 up, since a GPU of a family supports every
earlier one's. So every node runs the same configuration, and the ranks of one run may hold different GPUs, each with
its own file. Where Metal reports a limit of the device (its threads and threadgroup memory a threadgroup), the rank
takes the device's figure. What no document states, the SIMD width and how many threads a compiled kernel holds, is
read from the compiled pipeline.

## The configuration

The service takes one operand: a configuration file. It names the input, the weights and the output. It gives the
width and the parts. It gives `row_capacity`, the most rows a step may take, and names `capabilities`, the
directory of capability files, from which each rank takes its own GPU's. Its paths are read from the directory the
service runs in, the checkout's root in the sample. The kernel's form follows from the shape and the capability file,
and its wait limit is the library's, so the configuration names none of them. The input file says how many steps and
rows there are, so the configuration does not. The service has no defaults, and a missing value is an
error.

A part is how many rows of A and B a rank holds, in rank order. The service never guesses a part. A balancer that
measures the calls writes the parts. The operator's pair, 2026-10-17 (design/rubrics.md D5): lower, "refusing to use it
at all is never correct"; upper, "the *second entire computer* is a resource".
[examples/chain-service.json](../examples/chain-service.json) is a sample. Its parts are equal only because it is a
sample. Its input's every step is held in the region (4,096 steps of 8 rows of 1,024 halves, 64 MiB a rank).

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

Each rank prints one line at the end. It gives the parts, the rows, the steps and the seconds. It gives the rows a second of all streams together and the steps a second of each stream. It gives three times a
step: the whole step, the host's own time (with any wait inside torch), and a plain all-reduce
alone (the library's own kernels, not composed), the crossing the composed step no longer pays. It gives the bytes
a rank's crossing moves a step (its part out to each of the n - 1 peers, and each peer's part in), and those bytes a
second.

The line also splits the run in two. The first step carries costs a run pays once, such as the library compiling its
kernels the first time a collective runs. So the line gives the first step's time. It then gives the time a later step
takes, and the rows a second they reach. The first step's end is the GPU's time at its timing event, so the later steps
counted are the ones after it. The event is on the stream (`torch.mps.Event`), and the service reads the events only
after the run, so the step loop reads no clock and no wait is added to take the time. The whole-run times include the
first step; these do not.

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
same kind of exchange the chain sums over (`fused.Exchange`, each rank's part its share's rows). This is Megatron's split of a cell by hidden unit. Splitting by input rows
instead, with an all-reduce of gate partials, would send about 4n times as many bytes and repeat the cell update on
every rank.

**Reads in place.** A layer's u is never assembled in device memory. Its columns are segments read where they are: the
step's input, this rank's own part of the landed start (`mesh_own`), and each peer's part (`mesh_landing`). They are in rank order,
each a whole number of 8-column tiles. Each worker stages u once a layer, as GEMM operand tiling does (CUTLASS; Volkov
and Demmel, "Benchmarking GPUs to tune dense linear algebra", SC 2008). After the landing wait, one parallel pass over
the segments (`Segments::fill`, the segment outermost, so no segment lookup is left in the k-loop) loads a K-chunk of
the R-row band system-coherent (the NIC wrote the landings) into threadgroup memory. Then `simdgroup_load` reads it
there for every column tile the worker owns, where each tile once loaded its own copy system-coherent. The chunk is a
fit, not a cost model: `lstm.form` gives the band the threadgroup memory left beside the K split's partials (`red`),
each row padded by one tile (MLX steel/gemm's 16 bytes), in as few equal chunks as fit. Where a layer's K is more than one
chunk and a worker's items take more than one pass of its simdgroups (ks = 1, wide layers), each pass stages every
chunk again. This is a known cost: the system-coherent band reads repeat, R x K halves a pass against the pass's
weights, 32 tiles of K x 8 at least. The results are the same. The weights are tiled once at load, and their gate columns are laid out
unit-major, so an 8-column tile holds two units' (i, f, g, o).

**Epilogue.** A lane holds (i, f) or (g, o) of one unit and row. One `simd_shuffle_xor` pairs them, and the (i, f) lane
updates c and stores h through the tick's send view (`mesh_send`). A layer that has not started (s < l) stores
zeros there, its h(-1). The tick's kernel lands the tick before and publishes its own, so it calls `mesh_enter` at its
start and `mesh_leave` at its end, and the last worker releases the tick's publication in the chain's order. Rank 0 writes the top layer's landed h to its step's row of the output, as the chain does.

**Configuration.** `input` and `output` as the chain's. Also `input_width`, `layers`,
`shares` (one count of hidden units a rank, multiples of 8, their sum the width) and `capabilities`. The weights file
holds `w_ih_<l>` [4H, its input], `w_hh_<l>` [4H, H] and `b_<l>` [4H] in torch's gate order. `weights` is a directory of one .npy file an array. Before the communicator exists, rank 0 reads them
memory-mapped and writes each rank's gate rows, already tiled, into CPU region tensors, 64 tiles at a time. It then
sends each rank its tiles, and each rank receives them into an MPS region tensor the kernel reads. Nothing the size of
a weight matrix lives outside the region: a Studio's region wires 86 of its 96 GiB, and the rest is all that other
memory has.

## What to read next

- [design/transport.md](../design/transport.md): the bridge, its region, its links and how they fail.
- [rdma/NCCL.md](../rdma/NCCL.md): the collective client, its groups and a program's part in them.
- [rdma/torch-mesh/README.md](../rdma/torch-mesh/README.md): the same client from PyTorch, and region tensors.
- [design/heterogeneity.md](../design/heterogeneity.md): nodes of different speeds and links of different costs.
