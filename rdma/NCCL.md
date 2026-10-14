# libnccl-mesh

NCCL's C API over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each. Code
written against NCCL's `nccl.h` compiles against `rdma/nccl.h`. Its header comment is the contract; this page is
how to run it. PyTorch programs use it through [torch-mesh](torch-mesh/README.md). NumPy programs use it through
`mesh_mpi.py` (below). Uneven nodes and links: [design/heterogeneity.md](../design/heterogeneity.md).

## Build and run

A node's mesh installation (`install.sh`, kept current by `io.mesh.update`) builds the bridge, the libraries and the
Python bindings, installs the library (its headers, libraries and modules) under `/usr/local/mesh/rdma`, keeps the
node's torch environment with torch-mesh and the library on its path (`~/.venv-mesh-uv`, `bin/mesh-venv.sh`), and runs
the bridge as the node's service (`io.mesh.bridge`, design/bridge-and-ipc.md). A program links or imports the
installed library:

```
cc app.c -I /usr/local/mesh/rdma -L /usr/local/mesh/rdma -lnccl-mesh -Wl,-rpath,/usr/local/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`); the communicator's
map is what the ranks' bridges record when it is made.

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's, with NCCL's bootstrap: the id is the root's `host:port`
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens); each rank's node and its bridge's ports meet there, and
  the map is their join (`mesh_link_map_join`): a link where both ranks' bridges hold a port paired with the other,
  priced by the widest such port's bandwidth.
- `ncclMeshCommInitRank(comm, rank, topology, paths, programs, count, node, name)`: the topology is an operand.
  - `topology`: a link map (`mesh-plan.h`: kind `mesh`, `ring`, `tree` or `graph`; links; each link's alpha in
    microseconds and beta in nanoseconds a byte).
  - `paths`: the trees messages between unlinked ranks take. NULL means shortest paths.
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the declared default.
  - `node`: each rank's bridge node. NULL means rank r runs on node r.
  - `name`: the communicator's, which its sessions pair under on every member.
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
  `ncclMeshBuffer` (an `MTLBuffer` and a byte offset). Zero copies into the bridge's region.
  - torch-mesh runs MPS tensors here.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`).

Types: the Metal path moves the integer types, float16, float32, bfloat16 and fp8 (e4m3fn, e5m2: decoded, combined in
float32 and encoded to nearest even as the host does, overflow to NaN or infinity; metal-microbench docs/kernels.md
Float8); the host path adds float64.
Ops: sum, prod, max, min, avg; premul-sum takes a host immediate.

## Routes between unlinked ranks

Collectives run on linked ranks alone, so they need no routes. A message between ranks the map does not link
(send/recv, all-to-all, gather and scatter, which are grouped sends) takes the shortest path, and each node between
forwards it (NCCL's PXN proxy) on the link's relay rings:

- One relay ring a class (large and small) each way, on queue pairs 2 and 3. Every routed pair crossing the link
  shares it, so a map of any size and shape needs 4 queue pairs a link; a TB5 device has 10, which the bridge leases to
  its node's programs and refuses past (ENOSPC).
- Each message carries its ends and position. A node delivers what is addressed to it into its channel, and queues
  the rest for the next link, so a ring never waits on another.
- A ring's slots are credited back by its receiver, on the messages going the other way or alone.

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

A link lost under a group fails the group (its program on that link ends; nothing resumes it), and the next group
opens a session again on the links as they are then. Recover across calls: make the communicators again, on the
nodes that remain.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | library (ncclGetUniqueId) | the bootstrap root's `host:port` |
| `MESH_REMOTE_BOUND` | library | seconds a link may stay silent with positions pending |
| `NCCL_BUFFSIZE` | library | bytes of a large ring's slots together (default 4 MiB) |
| `MESH_POSITIONS` | library | the session's cyclic invocations (default 8192) |
| `MESH_CHECK_TRACE` | library | each group's scheduled messages and each position check (the pairing report's inputs) in the rank's log |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against (default its parent; the mesh's install passes `/usr/local/mesh/rdma`) |

## NumPy

`rdma/mesh_mpi.py` spells the host path as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import:

```python
from mesh_mpi import Comm, SUM, MAX     # sys.path: /usr/local/mesh/rdma
comm = Comm()                           # RANK of WORLD_SIZE, NCCL's bootstrap at MASTER_ADDR:MASTER_PORT
comm.Allreduce(local, total, op=SUM)    # Allgather, Bcast, Reduce, Reduce_scatter_block, Alltoall, Send, Recv,
best = comm.allreduce(x.max(), op=MAX)  # Barrier; IN_PLACE as a send buffer
```

Subcommunicators are `Split(color, key)` and a Cartesian grid: `grid = comm.Create_cart([2, 2])`, `grid.Sub([False,
True])` its row, `grid.Shift(0, 1)` the neighbours, as mpi4py spells them.

metal-microbench `tools/mesh/numpy_power.py` is a ported power iteration, and `tools/mesh/numpy_summa.py` SUMMA on a
2-D grid (row and column broadcasts).

## On one host

`tools/nccl_demo.py plans` (metal-microbench): every compiled program on data, over every family of maps n = 2..8.
CPU only, about 2 s.

## Not implemented

ncclCommRevoke, ncclCommShrink, ncclCommGrow, ncclCommInitRankScalable, ncclCommSuspend/Resume, the window and
signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd, ncclSetEncryption, the ncclParam
functions (`nccl.h`).
