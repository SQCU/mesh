# libnccl-mesh

NCCL's C API over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each. Code
written against NCCL's `nccl.h` compiles against `rdma/nccl.h`. Its header comment is the contract; this page is
how to run it. PyTorch programs use it through [torch-mesh](torch-mesh/README.md). NumPy programs use it through
`mesh_mpi.py` (below). Uneven nodes and links: [design/heterogeneity.md](../design/heterogeneity.md).

## Build and run

The node's user keeps the library in the user's mesh checkout (`~/mesh`): `bin/mesh-user.sh` builds the bridge, the
libraries and the Python bindings there, keeps the user's torch environment with torch-mesh and the checkout's library on
its path (`~/.venv-mesh-uv`), and runs the bridge as the user's agent (`io.mesh.bridge`, design/bridge-and-ipc.md) with
the node's bridge configuration, its one operand (`bin/mesh-user.sh CONFIGURATION`, etc/bridge/README.md; there is no
default); no part of it needs root. A program links or imports the checkout's library:

```
cc app.c -I ~/mesh/rdma -L ~/mesh/rdma -lnccl-mesh -Wl,-rpath,$HOME/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's, with NCCL's bootstrap: the id is the root's `host:port`
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens), where each rank's node meets the others'.
- `ncclMeshCommInitRank(comm, nranks, rank, paths, programs, count, node, name)`: the collectives are operands; the
  map is the bridges' (below).
  - `paths`: the trees the default single-phase all-reduce is compiled along, checked against the map each session
    reads. NULL means the map's shortest-path trees.
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the default (below).
  - `node`: each rank's bridge node.
  - `name`: the communicator's, which its sessions pair under on every member.
  - Every rank passes the same values.
- `ncclCommSplit` takes the ranks of a color; `ncclCommShrink` (NCCL's signature) leaves out the ranks named, those
  that departed, so a loop goes on with the ranks that remain and no new bootstrap.
- Making a communicator, a lone rank's too, attaches the process's client of its node's bridge ([Memory](#memory)),
  so the node's bridge runs; it fails where `ncclMemAlloc` would.

Each session takes its node's bridge's map when it opens (`mesh-plan.h` `mesh_map` over the communicator's nodes): two
ranks are linked where their bridges hold a port paired with each other, and ranks the cables leave apart are linked
along a route through other nodes, members or not, which the bridges forward along. A later session reads the map
again, so a recabling, a link that paired again or a node gone reaches the next session with no edit and no restart.

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR). The default table:

- All-reduce: single-phase along shortest-path trees (the map's diameter in dependent crossings) below the
  alpha-beta crossover, then reduce-scatter and all-gather along packed spanning trees.
- Broadcast, reduce, reduce-scatter and all-gather: along the packed trees, which pack by each link's own cost.

To choose otherwise, pass your own table (`mesh.compile` in Python). Nothing about routing is decided where a call
runs.

## Two paths

- **Host path.** A NULL stream: host pointers, copied through the session's rings, and each call is synchronous. It
  works for any array on any node whose bridge runs (NumPy, CPU tensors). It is functional, not the measured path.
- **Metal path.** An `ncclMeshStream`: the group is encoded into a Metal command buffer, and every buffer is an
  `ncclMeshBuffer` (an `MTLBuffer` and a byte offset). The kernels copy each operand into the session's rings in the
  bridge's window and land what arrives from them, with no staging on the host.
  - torch-mesh runs MPS tensors here.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`).

Types: the Metal path moves the integer types, float16, float32, bfloat16 and fp8 (e4m3fn, e5m2: decoded, combined in
float32 and encoded to nearest even as the host does, overflow to NaN or infinity; metal-microbench docs/kernels.md
Float8); the host path adds float64.
Ops: sum, prod, max, min, avg; premul-sum takes a host immediate.

## Memory

The node's bridge registers its window, its configuration's share of the node's memory (etc/bridge/README.md), on
every device when the device comes up, before any queue pair exists, and never again while the device is up
(design/prepared-machine.md#M09). The operator, 2026-10-15: "the bridge's job is to allocate essentially all of the
memory on the entire device so that there are large already 'pinned' buffers which are ready to receive surprise
transfers in whatever order they happen from whatever peers and links are operational ... ontologically it must be
the responsibility of client programs to know how much allocation they need ... the bridge is there to behave like a
system call".

- The process is one client of its node's bridge: one attach of the node's region, made at its first `ncclMemAlloc` or
  communicator. Its slot owns the process's leases, and each communicator's session takes its program's own slot on
  it, under the communicator's name (a slot is one execution's program).
- `ncclMemAlloc(ptr, size)` is the lease, the system call: whole pages of the window covering `size` (one page for 0),
  already registered; a lease registers nothing. Its bookkeeping is the region's (the lease's rows are owned by the
  client's slot), so when the process exits or dies its bridge returns the pages as it returns every retired client's
  rows.
- A lease the window cannot meet is `ncclSystemError` naming the bytes asked and the pages free. While every client
  slot of the bridge is held it is `ncclSystemError` naming `mesh.h` `MESH_CLIENTS`. Where the process attaches no
  region of a bridge that runs, it names the requirement: the node's bridge runs with a configuration whose window
  holds the client's footprint (`ncclRemoteError` where the bridge stopped, else `ncclSystemError`). Nothing falls back
  to other memory.
- `ncclMemFree(ptr)` returns the lease's pages at once.
- `ncclCommRegister` takes a buffer in the window (its handle is the buffer itself) and refuses any other with
  `ncclInvalidArgument` (lease it with `ncclMemAlloc`). `ncclCommDeregister` releases nothing.
- A lease is an operand of the window it came from, and a bridge that stops takes its region with it: the lease's
  pages stay readable but are no longer registered memory. While the process holds such a lease, its next
  `ncclMemAlloc`, communicator, session (a communicator's next group that crosses a link) and `ncclCommRegister` fail
  with `ncclRemoteError` naming it, and `ncclMeshMemLive()` says so for a process that makes no other call (a server
  between calls). Once the process has freed every such lease (`ncclMemFree`), its next call attaches to the bridge
  that runs then and leases from its window, so a process never holds leases of two regions. This is the recovery
  doctrine (design/recovery.md, mandates 12 and 13, the operator, 2026-10-14): "functions can fail from any one operand
  becoming invalidatd, servers should not fail when any operand is invalidated (ideally)"; "within call failures
  cannot be resume and instead must return an error (which a caller can presumably retry against immediately if that's
  what they want)".

The client sizes the window on each node it runs on, and on the nodes it peers or streams with: weights, KV and every
buffer a peer reads are leases. The operator, 2026-10-15: "why would weights or kv be outside of the memory pinned
pool? being outside of it means that they cannot be zero copy passed to peers. ... pipeline parallelism, context
parallelism, and megatron column parallelism are all required. ... you can't defend against this; it's *your code*
you're pretending you need to defend against; you can write it right all along."

Porting:

- An NCCL program takes its collectives' buffers from `ncclMemAlloc` (NCCL's allocator for the buffers it registers), and
  its node's bridge configuration's window holds them.
- A NumPy program takes them from `mesh_mpi.empty(shape, dtype)`, the role mpi4py's `MPI.Alloc_mem` plays; `mesh_mpi`'s
  own temporaries are such arrays.
- Python's lease is `mesh.lease(nbytes)`: a writable buffer for `numpy.frombuffer` or `torch.frombuffer`, returned to
  the window when it and everything over it are gone.
- torch-mesh's own CPU staging and temporaries are leases ([torch-mesh](torch-mesh/README.md)).
- Both paths copy every operand, a lease or not, through the session's rings; no libnccl-mesh collective crosses a
  caller's buffer without a copy yet. The window is where a program that binds its own section (`mesh-call.h`
  `mesh_transfer_bind`) crosses without one; `ncclCommRegister` refuses a buffer outside it.

## Routes between unlinked ranks

A message between ranks the map does not link (send/recv, all-to-all, gather and scatter, which are grouped sends, and
a collective's crossing between ranks the map joins by a route) is one program toward its destination node. The
sending node's bridge carries it over the first link of the route, and each bridge on the route forwards it to the
next node the route names, landing each record in rows it leases and sending it on from there, zero copy
(design/bridge-and-ipc.md#flows). The session keeps no relay rings and takes no queue pairs a link beyond its own
programs'.

## Rules a program keeps

- One rank a node.
- A bidirectional exchange with one peer is one group (`ncclGroupStart`/`End`; torch's `batch_isend_irecv`).
  Separate send then receive calls in the same order on both sides do not pair: a channel has one position axis for
  both directions. The library reports it as `ncclInvalidUsage` ("calls that do not pair"), as it does unequal counts
  and different collectives on two ranks: a position where a rank sends nothing carries a stamp, and the receiver
  checks every position. The host path fails the call. The Metal path finds it on the GPU and fails the session: the
  communicator's asynchronous error (`ncclCommGetAsyncError`), and the next call.
- Each communicator's session is its own program on the process's client of its node's bridge, under its own name;
  communicators in one process know nothing of one another.

## Failures

A destination with positions pending that lands nothing for `mesh.h` `MESH_SILENT_NS` (10 s, the library's one silent
bound, with no setting) is cancelled: every wait on it ends, and a host-path group with nothing landing for as long
fails. The failure is returned:

- by the group, or
- by `ncclCommGetAsyncError` for work already enqueued,

and `ncclGetLastError` names the node where it failed: the destination's, or a relay's on its route. A link or a relay
lost under a group fails the group (its program toward that node ends; nothing resumes it), and the next group opens a
session again on the map as it is then; a session also opens again at its next group once the bridge's map serial has
moved. A rank that departed is left out with `ncclCommShrink`, a joining one taken in with `ncclCommGetUniqueId` and
`ncclCommGrow`, and `ncclCommRevoke` ends a communicator's groups in flight.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | library (ncclGetUniqueId) | the bootstrap root's `host:port` |
| `NCCL_BUFFSIZE` | library | bytes of a large ring's slots together (default 4 MiB), read when a session opens |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against (default its parent; `bin/mesh-user.sh` passes the checkout's) |

## NumPy

`rdma/mesh_mpi.py` spells the host path as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import:

```python
from mesh_mpi import Comm, SUM, MAX     # sys.path: ~/mesh/rdma
comm = Comm()                           # RANK of WORLD_SIZE, NCCL's bootstrap at MASTER_ADDR:MASTER_PORT
comm.Allreduce(local, total, op=SUM)    # Allgather, Bcast, Reduce, Reduce_scatter_block, Alltoall, Send, Recv,
best = comm.allreduce(x.max(), op=MAX)  # Barrier; IN_PLACE as a send buffer
```

Subcommunicators are `Split(color, key)` and a Cartesian grid: `grid = comm.Create_cart([2, 2])`, `grid.Sub([False,
True])` its row, `grid.Shift(0, 1)` the neighbours, as mpi4py spells them.

metal-microbench `tools/mesh/numpy_power.py` is a ported power iteration, and `tools/mesh/numpy_summa.py` SUMMA on a
2-D grid (row and column broadcasts).

## On one host

`tools/nccl_demo.py plans` (metal-microbench): every compiled program on data, over rings and stars of n = 3..8, priced
at the mean cost of the links of this node's bridge's map.

## Not implemented

ncclCommInitRankScalable, ncclCommSuspend/Resume,
ncclCommMemStats, the window and signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd,
ncclSetEncryption, the ncclParam functions (`nccl.h`).
