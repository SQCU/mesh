# libnccl-mesh

NCCL's collective calls over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each.
An application builds against `rdma/nccl.h`, whose header comment is the contract; this page is how to build and link
it and what it does underneath. PyTorch code uses it through [torch-mesh](torch-mesh/README.md), NumPy code through
`mesh_mpi.py` (below). The transport under it: [design/PAGE-TABLE-SPEC-20261122.md](../design/PAGE-TABLE-SPEC-20261122.md)
(section 6 is this library).

## Building and linking

The node's user has the library in the user's mesh checkout (`~/mesh`): `bin/mesh-user.sh` builds the bridge, the
libraries and the Python bindings there, sets up the user's torch environment with torch-mesh and the checkout's
library on its path (`~/.venv-mesh-uv`), and starts the bridge as the user's launchd agent (`io.mesh.bridge`,
design/transport.md) with the node's bridge configuration (`bin/mesh-user.sh CONFIGURATION`, etc/bridge/README.md); no
part of it needs root. `make -C rdma .build/libnccl-mesh.dylib` builds the library alone, from `nccl-mesh.c` over
`libmesh.dylib`. An application links the checkout's library:

```
cc app.c -I ~/mesh/rdma -L ~/mesh/rdma -lnccl-mesh -Wl,-rpath,$HOME/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), or hand every rank
the root's `ncclGetUniqueId`, or set `NCCL_COMM_ID` to the root's `host:port`.

## Calls

Every call is blocking and moves host memory. It returns once this rank's part of it is done: its pages written, and
every page due to it landed and taken into `recvbuff`. Its stream is NULL; any other stream is refused
(`ncclInvalidArgument`). `sendbuff` and `recvbuff` are ordinary memory. `ncclMemAlloc` allocates memory aligned to the
system's page, and `ncclMemFree` frees it; any other memory does as well.

Every rank makes the same calls on a communicator, in the same sequence, with the same count, type, operator and root,
as NCCL requires. Nothing is compared at call time: calls that do not pair are the caller's error, and each rank takes
the bytes due to it as they landed.

With B a rank's count of bytes, the calls move pages so:

| call | pages |
|---|---|
| `ncclAllGather` | each rank's B to every other rank, one write a page naming them all, landing at `recvbuff + rank B` |
| `ncclBroadcast` | the root's B to every other rank |
| `ncclReduce` | every other rank's B to the root, which reduces them with its own |
| `ncclReduceScatter` | segment d of every rank's sendbuff to rank d, which reduces them with its own |
| `ncclAllReduce`, B one page | each rank's B to every other rank, each reducing them all: one round of pages |
| `ncclAllReduce`, more | a reduce-scatter of the count, split as evenly as the ranks allow, then an all-gather of the reduced segments |
| `ncclAlltoAll` | segment d of each rank's sendbuff to rank d, landing at `recvbuff + rank B` |
| `ncclGather` | every other rank's B to the root, landing at its `recvbuff + rank B` |
| `ncclScatter` | the root's segment d to rank d |

"One page" is at most `MESH_LONG_PAYLOAD` bytes (mesh-call.h). The single-phase all-reduce is not a timing rule: where
the data is one page, both forms send one page to each peer, and the single phase saves a round.

A rank's own part is copied in place, and not at all where it is there already (NCCL's in-place forms). The all-to-all
has no in-place form, as in NCCL.

### Reductions

A reduction is computed on the host, rank by rank: each page of the result is rank 0's contribution, then rank 1's
combined into it, and so on. Every rank therefore computes the same bits, call after call. All twelve NCCL types
reduce, with `ncclSum`, `ncclProd`, `ncclMax`, `ncclMin` and `ncclAvg`. The arithmetic is numpy's and ml_dtypes': the
16- and 8-bit floating types are computed in float32 and rounded to nearest even, and integers wrap. A floating average
premultiplies every contribution by 1 / nranks before the sum, and an integer one divides the sum by nranks (NCCL's
`FuncPreMulSum` and `FuncSumPostDiv`). In a two-phase all-reduce, a segment's owner divides it before the all-gather.

### Groups and threads

`ncclGroupStart`/`ncclGroupEnd` bracket a group; a call outside one is a group of one. `ncclGroupEnd` makes the
group's calls one after another, as they were made, each to its end, on whatever communicators they name. Every call
is made whatever another's result, and the first failure is returned. An application that groups calls on several
communicators makes them in the same sequence on every rank they share.

A communicator's calls are made by one thread at a time. Calls on different communicators may come from different
threads, and are moved one after the other: the process moves one call at a time, on its one pool of threads (below).
A call that waits for the one before it to end ends at its own communicator's revoke, as any wait of a call does.
So, as NCCL asks of communicators used at once, the ranks that several communicators share make their calls on them in
the same sequence. Where one process calls on two communicators from two threads, their sequence is whichever call
takes the pool first, so such an application sequences those calls itself, or makes them from one thread. A process
is one rank of a communicator: two ranks of one communicator in one process would each wait for the other's pages.

## Pages and indices

A page is one SEND: `MESH_HEAD` bytes of header, which the bridge writes, and then its payload. A short page is
`MESH_CLASS_SHORT` frames (payload `MESH_SHORT_PAYLOAD`, 7936 bytes); a long one is `MESH_CLASS_LONG` frames (payload
`MESH_LONG_PAYLOAD`, 36608 bytes). A message of at most 7936 bytes is one short page. A longer one is long pages, each
full but its last. Every rank cuts a message alike. Both payloads are multiples of 8 bytes, so a page always has whole
elements.

**Channels and threads.** Every rank cuts a call's pages into `CHANNELS` (8) channels alike: page j of a message, and
a reduction's position j, belong to channel j mod 8. A rank moves its channels on its process's pool of T threads,
channel c on thread c mod T. The first thread is the calling thread. At its first communicator the process takes one
client slot of its node's bridge for each thread, so each thread has a write ring of its own, and starts a worker for
each slot past the first. T is `NCCL_MESH_THREADS` where set (1 to 8), else 4 (the fastest measured on the fleet), at
most half the node's top-level cores, and fewer where no more of the bridge's client slots are vacant. The workers
spin between calls and never sleep, from then until the process lets its attach go (once its bridge's region has
expired and it has destroyed every communicator of it) or exits. A worker joins a call only where the call is moved on
more threads than its number. A call whose messages are one page each moves on the calling thread alone. Each thread
copies its own channels' pages in and out, and makes its channels' part of the call's local copy and of a reduction.
So a large call's copies are made on all the threads at once, and the index layout is the same at every rank whatever
its thread count.

Each rank of a communicator takes, for each channel, on its node's page table (mesh-call.h `mesh_pages`, in the slot
of the channel's thread):

- **source pages**: `SOURCES` (128) long pages and 128 short pages, before the init's first round;
- **receive rings**: one ring of `depth` indices for every rank, `nranks` × `depth` indices, naming no page until one
  lands, after the first round has given every rank the depth.

Every rank computes the same **depth**, the receive ring's depth, from the first round's reports: at most `RECEIVES`
(512), and the most for which, at every node v of the communicator, its c_v ranks' source pages and a page at each
index of their channels' rings fit in half of the node's pool:

    c_v CHANNELS (SOURCES + (nranks - 1) depth) <= P_v / 2

where P_v is the fewer pages of the node's two classes. The other half is left for its bridge's lanes' receives, its
relays' pages and its other clients'. A depth of 0 fails the init (`ENOSPC`). An index names one page at a time, so
this is the most of either class the communicator ever has at v, whatever its peers write ahead. The bound covers one
communicator: communicators and clients that share a node share its pool.

The k-th page of channel c that rank s sends rank d on the communicator is written to d's index
`base[d][c] + s depth + k mod depth`. Rank s's pages of the channel k mod depth, then that plus depth, and so on, land
there in turn, so page k is the index's (k / depth + 1)-th landing (its `landed` count, which `mesh_entry` returns).
Each rank counts, for every channel and every peer, the pages it has sent it and the pages it has taken from it. Both
ends count each pair's pages alike, because every rank makes the same calls in the same sequence and cuts each message
alike. So the init's two rounds (each rank's node, pool and depth; then each channel's base) are the only exchange
that ever names an index.

**Sending.** The rank copies a page's bytes into a source page and calls `mesh_write` with the page's destinations.
A page to several ranks is one write to each `MESH_DESTS` of them by rank, each destination the index its own pair's
page count names, each write copied into a source page of its own. A source page is copied into again only once its
`uses` are back to 1: the table word's own use, with no write, SEND or landing of it still under way (`mesh_entry`).
Where no source page is free, or the write ring is full, the rank takes what has landed and tries again. A slot's
write ring has one producer at a time: its own thread of the pool, which moves one call at a time.

**Receiving.** Every page is independent of the others, and the rank takes every page it finds landed in the pass
that finds it. Each pass reads, for each peer, the indices of its next `depth` pages due, each at an index of its own,
so no index is read for a later page while an earlier page there is due. The rank takes page k once its index's
`landed` count reaches k / depth + 1. Where the index names a page, the rank copies the bytes due from it to the
page's place in `recvbuff`, whatever the page's bytes are, then invalidates the index (`mesh_invalidate`): the page is
dropped, and the index and its count stay. Where the index names no page (its page was taken under an earlier count),
nothing is copied. A page written `depth` pages or more ahead of the page its receiver takes there replaces that page
at the index, and is taken as it. This is the defragment of
[design/DEFRAGMENT-pseudocode-20261122.md](../design/DEFRAGMENT-pseudocode-20261122.md): one local copy a page,
charged to this consumer.

A reduction combines each page position's contributions rank by rank, each straight from its page. A page whose
rank's turn has come is combined at once, and its index is invalidated. A page that lands before its turn is left at
its index, which still names it, and the rank reads that index for no later page until the page's turn; then it is
combined from there and the index is invalidated. So no landed page is copied anywhere but its place in `recvbuff`.
Reduction pages left for their turn, and pages its peers write for a later phase or call, stay at their indices, one
an index, which is what the depth bounds within half of each node's pool. The reductions are this library's own code
over the page table's calls: the bridge carries pages and knows nothing of what they contain.

The waits are the call's own. A call's wait ends once its pages have landed, or at the communicator's revoke, which
each pass reads; nothing else bounds it. Nothing in the transport waits.

## Communicators

`ncclCommInitRank` is NCCL's bootstrap. The unique id is the root's `host:port`: `NCCL_COMM_ID`, or else the
listener `ncclGetUniqueId` opens on the node's fabric address (its first address under `MESH_ULA_PREFIX`), or on its
host name where it has none. The root listens on that host alone where it is an IPv6 address. The init is two rounds
over the bootstrap's sockets: in each, rank 0 takes every rank's report and sends each rank the whole table. A rank's
first-round report is its status, its node, its node's pool (the depth's P_v) and its standing listener's address; its
second-round report is its status and the first receive index of each of its channels. The communicator has the
sockets for a split's or a grow's later exchange.

Making a communicator does the following, and a rank that cannot go on fails every rank's init alike:

1. It attaches the process's client of its node's bridge where it has none (`mesh_attach`), with its pool of threads.
   This fails where every client slot is taken, or where no region of a bridge that is up exists.
2. It takes the communicator's source pages.
3. It opens the rank's standing listener.
4. The first round gathers every rank's report, and every rank computes the depth.
5. It takes the communicator's receive rings, and the second round gathers every rank's channels' first receive
   indices.

The bootstrap's waits are plain blocking calls on its sockets. An init's waits end when its peers answer: a rank whose
root does not listen yet connects again every 100 ms, and no revoke reaches an init, since no other thread holds the
communicator until its init returns it. A split's or a grow's exchange over a communicator also ends at that
communicator's revoke, which shuts its sockets down. Nothing else bounds them. Two ranks of a communicator may share
a node; a page between them is set at the destination index on that node without a SEND.

**Splits, shrinks and grows**:

- `ncclCommSplit` gathers every rank's (color, key) over the parent's bootstrap. Each color's communicator, its ranks
  sorted by key and then rank, meets at its first rank's standing listener.
- `ncclCommShrink` makes the remaining ranks' communicator at the first remaining rank's standing listener. It
  exchanges nothing over the parent, so an excluded rank, gone or not, takes no part. With `NCCL_SHRINK_ABORT` the
  parent is revoked first.
- A rank of no color, or an excluded one, gets NULL.
- `ncclCommGrow` meets at the id `ncclCommGetUniqueId` gives the parent's root, which the root gives its ranks over the
  parent's bootstrap. A joining process passes a NULL comm and the root's id.
- `config` is not read.

**Revoke and destroy**:

- `ncclCommRevoke` ends comm's calls under way as a value at their next look. Every later call on it is refused
  (`ncclInvalidUsage`), and `ncclCommGetAsyncError` reads `ncclSuccess` after it.
- `ncclCommDestroy` (and `ncclCommAbort`) revokes comm and returns at once, waiting for nothing. comm is freed (its
  indices, whose pages go back to the pool as their last uses end, its sockets and comm itself) by whoever lets it go
  last: the destroy, where nothing else is under way on comm, else the last of its calls (queued or under way), splits,
  shrinks and grows to end, each at its next look. comm is not named again after its destroy.
- A communicator's life is one state word: live, failed (its first recorded failure kept with it) or revoked. A call
  reads it with one load, and takes no lock.
- The process detaches from its region once the region has expired and it has no communicator of it; its pool's
  workers end then.

A region its bridge's stop released, or that a bridge of another layout replaced, is expired. While the process has a
communicator of it, a new one fails (`ncclRemoteError`). Once it has destroyed them all, its next one attaches the
bridge's next region.

## Failures

Failures are `ncclResult_t` values, and `ncclGetLastError` names what failed. A call records two failures on its
communicator: a write its node's table refuses and an index it cannot read. `ncclCommGetAsyncError` reads the first
one recorded, and a communicator that recorded one fails its later calls alike, until it is destroyed and made again.
Every other failure (an invalid argument or usage, other memory, the bootstrap) is the result of its call alone. A
revoked communicator's calls return `ncclInvalidUsage`, and `ncclCommGetAsyncError` reads `ncclSuccess` after the
revoke.

A call never fails on what lands. The library checks nothing that lands, and a page's bytes are taken as they landed:
a write may land wrong, or not at all (mesh-call.h). No timer judges a peer lost: a page that never lands leaves its
call waiting until the application revokes or aborts that communicator (`ncclCommRevoke`, `ncclCommAbort`), as with
NCCL's own.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | `ncclGetUniqueId` | the bootstrap root's `host:port` |
| `NCCL_MESH_THREADS` | the process's first communicator | the threads of the process's pool, which move a call's channels, 1 to 8 (default 4, at most half the top-level cores) |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against |

## NumPy

`rdma/mesh_mpi.py` spells these calls as mpi4py spells a communicator's uppercase (array) methods, so an mpi4py
application ports by its import (its docstring lists them):

```python
from mesh_mpi import Comm, SUM          # sys.path: ~/mesh/rdma
comm = Comm()                           # RANK of WORLD_SIZE, NCCL's bootstrap at MASTER_ADDR:MASTER_PORT
comm.Allreduce(local, total, op=SUM)
```

## Not implemented

These are not exported:

- the non-blocking init: `ncclCommInitRankConfig`;
- `ncclCommInitAll`, `ncclCommInitRankScalable`, `ncclCommFinalize`;
- `ncclCommSuspend`/`Resume`, `ncclCommMemStats`, `ncclCommCuDevice`;
- `ncclCommRegister`/`Deregister`, `ncclCommWindowRegister`/`Deregister`, `ncclWinGetUserPtr` and the signal functions;
- `ncclRedOpCreatePreMulSum`/`ncclRedOpDestroy`;
- `ncclBcast`, `ncclSend`/`ncclRecv` and the `nccl*Config` collectives;
- `ncclGroupSimulateEnd`, `ncclSetEncryption` and the `ncclParam` functions.

A standard call that nothing in the repository calls is left out rather than left as a stub.
