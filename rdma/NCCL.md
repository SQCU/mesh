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
metal-microbench's launcher starts them on a membership and runs any command on every node:

```
tools/mesh/grid.py run --members 0,2,3 -- '{python} my_program.py'
```

It reads the link map (default `configs/links/pair-ring.txt`, or `MESH_LINK_MAP`; `--links` another). It keeps the
links the members hold and probes only those links. It starts one bridge per node, with the queue pairs the routes
need (below). Each rank gets `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `MESH_REGION` and `MESH_LINK_MAP`
(the membership's map, its nodes numbered from 0), as torchrun sets them. The bridges stop by SIGTERM afterwards.
Its functions are `grid.ranks_up`, `ranks_run` and `ranks_down`, for drivers of their own.

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's. Every pair is linked, rank r runs on node r, and the region is
  `MESH_REGION`. Use it only where every pair of nodes is cabled.
- `ncclMeshCommInitRank(comm, rank, topology, paths, programs, count, node, region)`: the topology is an operand.
  - `topology`: a link map (`mesh-plan.h`: kind `mesh`, `ring`, `tree` or `graph`; links; each link's alpha in
    microseconds and beta in nanoseconds a byte). `mesh_link_map_read` reads the text form.
  - `paths`: the trees messages between unlinked ranks take. NULL means shortest paths.
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the declared default.
  - `node`: each rank's bridge node. NULL means rank r runs on node r.
  - Every rank passes the same values.
- `ncclCommSplit`: keeps the links among the ranks it takes.

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
(send/recv, all-to-all, gather and scatter, which are grouped sends) is forwarded by the nodes between them on
queue pairs of its own: `2 + 2 x` the routed pairs crossing the busiest link.

A TB5 device has 10 usable queue pairs a link. Where the routes need more, the launcher starts the bridges at 2
queue pairs, and the library refuses those messages with the numbers. Every collective still runs. Cable the pairs
that exchange messages, or run those calls on linked ranks.

- On a complete graph, no routes are needed.
- A ring or star of 5 fits the budget.
- A line of 5 or more, or a ring of 6 or more, does not.

## Rules a program keeps

- One rank a node.
- A bidirectional exchange with one peer is one group (`ncclGroupStart`/`End`; torch's `batch_isend_irecv`).
  Separate send then receive calls on both sides do not pair: a channel has one position axis for both directions.
- Make the world communicator first. A process's session opens over the widest communicator alive among whose ranks
  a group's are, so subgroups made after it never reopen it.
- One process holds one session for its life. `NCCL_BUFFSIZE` and `MESH_POSITIONS` are read when it opens.

## Failures

A link with positions pending that delivers nothing for `MESH_REMOTE_BOUND` seconds is cancelled (default 10;
torch-mesh sets it to the group's timeout). Every wait on that link ends. The failure is returned:

- by the group, or
- by `ncclCommGetAsyncError` for work already enqueued.

Recover across calls: make the communicators again, on the nodes that remain.

## Environment

| variable | read by | meaning |
|---|---|---|
| `MESH_REGION` | library, torch-mesh, mesh_mpi | this rank's bridge region (default `/mesh0`) |
| `MESH_LINK_MAP` | torch-mesh, mesh_mpi | the link map file (the launcher's: the membership's) |
| `MESH_REMOTE_BOUND` | library | seconds a link may stay silent with positions pending |
| `NCCL_BUFFSIZE` | library | bytes of a large ring's slots together (default 4 MiB) |
| `MESH_POSITIONS` | library | the session's cyclic invocations (default 8192) |
| `MESH_QPS` | bridge | queue pairs a link (the launcher's: 2, or what the routes need) |
| `MESH_LEDGER` | bridge | each link's crossing times written to its log at teardown |
| `MESH_HOST_TRACE` | host executor | a timed trace of the walk in the rank's log |
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

## Not implemented

ncclCommRevoke, ncclCommShrink, ncclCommGrow, ncclCommInitRankScalable, ncclCommSuspend/Resume, the window and
signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd, ncclSetEncryption, the ncclParam
functions (`nccl.h`).
