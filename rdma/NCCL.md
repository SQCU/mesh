# libnccl-mesh

NCCL's C API over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each. Code
written against NCCL's `nccl.h` compiles against `rdma/nccl.h`. Its header comment is the contract; this page is
how to run it. PyTorch programs use it through [torch-mesh](torch-mesh/README.md). NumPy programs use it through
`mesh_mpi.py` (below). Uneven nodes and links: [design/heterogeneity.md](../design/heterogeneity.md).

## Build

```
make -C rdma libmesh.dylib libnccl-mesh.dylib mesh-flow   # the library and the bridge
make -C rdma mesh-c PYTHON=~/.venv-mesh-uv/bin/python      # _mesh_c: the headers for Python (mesh.py)
cc app.c -I rdma -L rdma -lnccl-mesh -Wl,-rpath,<rdma>
```

On a node, metal-microbench's `tools/mesh/node-setup.sh` does all of it (docs/nodes.md there).

## Run

A rank is a process on a node whose bridge is up. Bridges are started per call by a driver, never by launchd.
metal-microbench's launcher (the operator's repository; without it, [by hand](#starting-bridges-by-hand)) starts them
on a membership and runs any command on every node:

```
~/.venv-mesh-uv/bin/python tools/mesh/grid.py run --members 0,2,3 -- '{python} my_program.py'
```

Drivers run under `~/.venv-mesh-uv` (the bindings are built for it). Each node runs the command from its own
`~/metal-microbench`, so a program must be committed and pulled there first. Each rank's output is in
`output_data/grid-run/<tag>/run-rank<r>.log`. In the command, `{python}` is the node's venv, `{node}` the node's
place in the membership's map, and `{jsonl}` a file of the rank's own.

Validated: two nodes on TB5 with GPUs, and three to eight on one host's loopback (metal-microbench docs/nodes.md).

It reads the link map (default `configs/links/pair-ring.txt`, or `MESH_LINK_MAP`; `--links` another). It keeps the
links the members hold and probes only those links. It starts one bridge per node, with the queue pairs the routes
need (below: 4 where any pair is unlinked, else 2). Each rank gets `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `MESH_REGION` and `MESH_LINK_MAP`
(the membership's map, its nodes numbered from 0), as torchrun sets them. `RANK` is the launcher's: rank 0 is a member
other than the driver (the store listens there), and the launcher prints the rank-to-node table. A program that
makes its own communicator passes its node as its rank (`{node}`, or `ncclMeshCommInitRank` with a `node` table). The
bridges stop by SIGTERM afterwards.
Its functions are `grid.ranks_up`, `ranks_run` and `ranks_down`, for drivers of their own.

## Starting bridges by hand

What the launcher does, for a driver of your own. Number the membership's nodes 0..N-1 in a link map (below) and on
each node start one bridge:

```
MESH_QPS=4 ~/mesh/rdma/mesh-flow -I 2 -A 131072 -B 1 -R 524288 -O 4096 -s /ranks \
  --link 'rdma_en3,1,fe80::a%en3,fe80::b%en3,18612' --link 'rdma_en5,3,fe80::c%en5,fe80::d%en5,18614'
```

- `-I` this node's number in the map; `-s` the region the ranks attach to (their `MESH_REGION`). `-A` arena pages,
  `-B` pages a block, `-R` page-table rows, `-O` transfer-list entries: the values above are the launcher's (a 2 GiB
  arena of 16 KiB pages).
- One `--link` a cable: the RDMA device of the port (`ibv_devinfo`: `rdma_en<k>`), the peer's node number, this end's
  and the peer's IPv6 link-local addresses on that port (`ifconfig en<k>`, the `fe80::` line) with its scope, and a
  TCP port both ends give alike (one a cable; the higher-numbered node listens). Which port reaches which peer:
  `ping6` the peer's link-local over each port.
- `MESH_QPS`: 4 where the map leaves any pair unlinked, else 2. `MESH_PAIR_SECONDS`: how long a link may take to pair
  (default 30).
- A bridge prints `bridge node <n>:` to its stderr once it is up, and `pair up:` per link. Then start a rank on each
  node with `MESH_REGION=/ranks`, `MESH_LINK_MAP` the map file, and torchrun's `RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
  `MASTER_PORT` for torch (RANK may be any order: each rank reads its node from its bridge).
- Afterwards SIGTERM each bridge's pid, never SIGKILL.

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's. Rank r runs on node r, the region is `MESH_REGION`, and the links
  are `MESH_LINK_MAP`'s (every pair linked without it).
- `ncclMeshCommInitRank(comm, rank, topology, paths, programs, count, node, region)`: the topology is an operand.
  - `topology`: a link map (`mesh-plan.h`: kind `mesh`, `ring`, `tree` or `graph`; links; each link's alpha in
    microseconds and beta in nanoseconds a byte). `mesh_link_map_read` reads the text form.
  - `paths`: the trees messages between unlinked ranks take. NULL means shortest paths.
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the declared default.
  - `node`: each rank's bridge node. NULL means rank r runs on node r.
  - Every rank passes the same values.
- `ncclCommSplit`: keeps the links among the ranks it takes.
- A subgroup whose members the cables do not join (a split, or `ncclMeshCommInitRank` with a graph that leaves its
  ranks apart, as torch's `new_group` makes) is joined along the paths of the widest communicator alive over its
  members: the cheapest path between two parts first, until one part remains. Its programs cross those paths through
  the other members' nodes.

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR). The default table:

- All-reduce: single-phase along shortest-path trees (the map's diameter in dependent crossings) below the
  alpha-beta crossover, then reduce-scatter and all-gather along packed spanning trees.
- Broadcast, reduce, reduce-scatter and all-gather: along the packed trees, which pack by each link's own cost.

To choose otherwise, pass your own table (`mesh.compile` in Python). Nothing about routing is decided where a call
runs.

## Two paths

- **Host path.** A NULL stream: host pointers, and each call is synchronous. It works for any array on any node
  (NumPy, CPU tensors). It is functional, not the measured path.
- **Metal path.** An `ncclMeshStream`: the group is encoded into a Metal command buffer, and every buffer is an
  `ncclMeshBuffer` (an `MTLBuffer` and a byte offset). Zero copies into the bridge's registered memory.
  - torch-mesh runs MPS tensors here.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`).

Types: the Metal path moves the integer types, float16, float32 and bfloat16; the host path adds float64 and fp8.
Ops: sum, prod, max, min, avg; premul-sum takes a host immediate.

## Routes between unlinked ranks

Collectives run on linked ranks alone, so they need no routes. A message between ranks the map does not link
(send/recv, all-to-all, gather and scatter, which are grouped sends) takes the shortest path, and each node between
forwards it (NCCL's PXN proxy) on the link's relay rings:

- One relay ring a class (large and small) each way, on queue pairs 2 and 3. Every routed pair crossing the link
  shares it, so a map of any size and shape needs 4 queue pairs a link; a TB5 device has 10.
- Each message carries its ends and position. A node delivers what is addressed to it into its channel, and queues
  the rest for the next link, so a ring never waits on another.
- A ring's slots are credited back by its receiver, on the messages going the other way or alone.

The launcher starts the bridges at 4 queue pairs where any pair of members is unlinked, else 2. With 2, the library
refuses those messages by name; every collective still runs.

## Rules a program keeps

- One rank a node.
- A bidirectional exchange with one peer is one group (`ncclGroupStart`/`End`; torch's `batch_isend_irecv`).
  Separate send then receive calls in the same order on both sides do not pair: a channel has one position axis for
  both directions. The library reports it as `ncclInvalidUsage` ("calls that do not pair"), as it does unequal counts
  and different collectives on two ranks: a position where a rank sends nothing carries a stamp, and the receiver
  checks every position. The host path fails the call. The Metal path finds it on the GPU and fails the session: the
  communicator's asynchronous error (`ncclCommGetAsyncError`), and the next call.
- Make the world communicator first. A process's session opens over the widest communicator alive among whose ranks
  a group's are, so subgroups made after it never reopen it.
- One process holds one session for its life. `NCCL_BUFFSIZE` and `MESH_POSITIONS` are read when it opens.

## Failures

A link with positions pending that delivers nothing for `MESH_REMOTE_BOUND` seconds is cancelled (default 10;
torch-mesh sets it to the group's timeout, at most 60 s: a wait on the Metal path is a kernel polling on the GPU).
Every wait on that link ends, and a host-path group with nothing landing for as long fails. The failure is returned:

- by the group, or
- by `ncclCommGetAsyncError` for work already enqueued.

Recover across calls: make the communicators again, on the nodes that remain.

## Environment

| variable | read by | meaning |
|---|---|---|
| `MESH_REGION` | library, torch-mesh, mesh_mpi | this rank's bridge region (default `/mesh0`) |
| `MESH_LINK_MAP` | library (ncclCommInitRank), torch-mesh, mesh_mpi | the link map file (the launcher's: the membership's) |
| `MESH_REMOTE_BOUND` | library | seconds a link may stay silent with positions pending |
| `NCCL_BUFFSIZE` | library | bytes of a large ring's slots together (default 4 MiB) |
| `MESH_POSITIONS` | library | the session's cyclic invocations (default 8192) |
| `MESH_QPS` | bridge | queue pairs a link (the launcher's: 4 where any pair is unlinked, else 2) |
| `MESH_LEDGER` | bridge | each link's crossing times written to its log at teardown |
| `MESH_PAIR_SECONDS` | bridge | how long a link may take to pair from its start (default 30; the launcher passes it) |
| `MESH_HOST_TRACE` | host executor | a timed trace of the walk in the rank's log |
| `MESH_CHECK_TRACE` | library | each group's scheduled messages and each position check (the pairing report's inputs) in the rank's log |
| `MESH_LOOP_FABRIC` | loopback bridges | the shared-memory fabric of one host's bridges |
| `MESH_RDMA` | torch-mesh | the rdma directory (default the package's own) |

## NumPy

`rdma/mesh_mpi.py` spells the host path as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import:

```python
from mesh_mpi import Comm, SUM, MAX     # sys.path: ~/mesh/rdma
comm = Comm()                           # the world over MESH_LINK_MAP; its rank is its bridge's node
comm.Allreduce(local, total, op=SUM)    # Allgather, Bcast, Reduce, Reduce_scatter_block, Alltoall, Send, Recv,
best = comm.allreduce(x.max(), op=MAX)  # Barrier; IN_PLACE as a send buffer
```

metal-microbench `tools/mesh/numpy_power.py` is a ported power iteration (`grid.py run -- '{python}
tools/mesh/numpy_power.py'`).

## On one host

- `tools/nccl_demo.py plans`: every compiled program on data, over every family of maps n = 2..8. CPU only, about
  2 s.
- `tools/nccl_demo.py loop --nodes N --family F [--device] [--groups RE]`: N bridges over the shared-memory fabric
  (`loopverbs.c`, which refuses a device's eleventh queue pair as TB5 does), N ranks, and the NCCL matrix. Run it
  where no firewall prompts for listeners (the Mini).
  - `--device` runs the Metal path's walk on the host executor (`make -C rdma host`). Several ranks' GPU kernels
    on one GPU have no forward-progress guarantee.
  - `MESH_POSITIONS=256` before the command makes every channel and relay ring cross the session's cycle of positions
    within seconds, which a default session reaches only after 8192 positions on one channel.

## Not implemented

ncclCommRevoke, ncclCommShrink, ncclCommGrow, ncclCommInitRankScalable, ncclCommSuspend/Resume, the window and
signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd, ncclSetEncryption, the ncclParam
functions (`nccl.h`).
