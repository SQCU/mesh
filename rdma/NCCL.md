# libnccl-mesh

NCCL's collective calls over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each.
A program compiles against `rdma/nccl.h`, whose header comment is the contract; this page is how to run it and what
it does underneath. PyTorch programs use it through [torch-mesh](torch-mesh/README.md), NumPy programs through
`mesh_mpi.py` (below). The transport it runs on: [design/PAGE-TABLE-SPEC-20261122.md](../design/PAGE-TABLE-SPEC-20261122.md)
(section 6 is this library).

## Build and run

The node's user keeps the library in the user's mesh checkout (`~/mesh`): `bin/mesh-user.sh` builds the bridge, the
libraries and the Python bindings there, keeps the user's torch environment with torch-mesh and the checkout's library on
its path (`~/.venv-mesh-uv`), and runs the bridge as the user's launchd agent (`io.mesh.bridge`, design/transport.md) with
the node's bridge configuration (`bin/mesh-user.sh CONFIGURATION`, etc/bridge/README.md); no part of it needs root.
`make -C rdma .build/libnccl-mesh.dylib` builds the library alone, from `nccl-mesh.c` over `libmesh.dylib`. A program
links the checkout's library:

```
cc app.c -I ~/mesh/rdma -L ~/mesh/rdma -lnccl-mesh -Wl,-rpath,$HOME/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), or hand every rank
the root's `ncclGetUniqueId`, or set `NCCL_COMM_ID` to the root's `host:port`.

## Calls

Every call is blocking and moves host memory. It returns once this rank's part of it is done: its pages written, and
every page due to it landed and taken into its buffers. Its stream is NULL; any other stream is refused
(`ncclInvalidArgument`). Buffers are ordinary memory. `ncclMemAlloc` allocates memory aligned to the system's page,
and `ncclMemFree` frees it; any other buffer does as well.

Every rank makes the same calls on a communicator, in the same order, with the same count, type, operator and root, as
NCCL requires. Nothing is compared at run time: calls that do not pair are the caller's error, and each rank takes the
bytes due to it as they landed.

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
the buffer is one page, both forms send one page to each peer, and the single phase saves a round.

A rank's own part is copied in place, and not at all where it is there already (NCCL's in-place forms). The all-to-all
has no in-place form, as in NCCL.

### Reductions

A reduction runs on the host, in rank order: each page of the result is rank 0's contribution, then rank 1's combined
into it, and so on. Every rank therefore computes the same bits, run after run. All twelve NCCL types reduce, with
`ncclSum`, `ncclProd`, `ncclMax`, `ncclMin` and `ncclAvg`. The arithmetic is numpy's and ml_dtypes': the 16- and
8-bit floating types are computed in float32 and rounded to nearest even, and integers wrap. A floating average
premultiplies every contribution by 1 / nranks before the sum, and an integer one divides the sum by nranks
(NCCL's `FuncPreMulSum` and `FuncSumPostDiv`). In a two-phase all-reduce, a segment's owner divides it before the
all-gather.

### Groups

`ncclGroupStart`/`ncclGroupEnd` bracket a group; a call outside one is a group of one. `ncclGroupEnd` runs the
group's calls in the order they were made, each to its end, on whatever communicators they name. Every call runs
whatever another's result, and the first failure is returned. A program that groups calls on several communicators
makes them in the same order on every rank they share. A communicator's calls are made by one thread at a time; calls
on different communicators may be made by different threads at once.

## Pages and indices

A page is one SEND: `MESH_HEAD` bytes of header, which the bridge writes, and then its payload. A short page is
`MESH_CLASS_SHORT` frames (payload `MESH_SHORT_PAYLOAD`, 7936 bytes); a long one is `MESH_CLASS_LONG` frames (payload
`MESH_LONG_PAYLOAD`, 36608 bytes). A message of at most 7936 bytes is one short page. A longer one is long pages, each
full but its last. Every rank cuts a message alike. Both payloads are multiples of 8 bytes, so a page always holds
whole elements.

**Shares and threads.** Every rank cuts a call's pages into `SHARES` (8) shares alike: page j of a message, and a
reduction's position j, belong to share j mod 8. A rank moves its shares on its threads, share s on thread s mod its
thread count. The first thread is the calling thread's. The process takes one client slot of its node's bridge for
each thread, so each thread has a write ring of its own. The thread count is `NCCL_MESH_THREADS` where set (1 to 8),
else half the node's performance cores, at most 8, and fewer where the bridge's client slots run out. A call whose
messages are one page each moves on the calling thread alone. Between calls a thread looks for the next one for
2 ms, then sleeps until a call wakes it. Each thread copies its own shares' pages in and out, and makes its shares'
part of the call's local copy and of a reduction. So a large call's copies run on all the threads at once, and the
index layout is the same at every rank whatever its thread count.

At its init each rank of a communicator takes, for each share, on its node's page table (mesh-call.h `mesh_pages`, in
the slot of the share's thread):

- **receive indices**: `RECEIVES` (512) for every rank, naming no page until one lands;
- **source pages**: `SOURCES` (128) long pages and 128 short pages.

Each pair of a share uses W of its `RECEIVES` indices, the communicator's **window**. Every rank computes the same W
from the init's reports: at most `RECEIVES`, and the most for which, at every node v of the communicator, its c_v
ranks' source pages and a page at each index of their shares' windows fit in half of the node's pool:

    c_v SHARES (SOURCES + (nranks - 1) W) <= P_v / 2

where P_v is the fewer pages of the node's two classes. The other half is left for its bridge's lanes' receives, its
relays' pages and its other clients'. A W of 0 fails the init. An index holds one page, so this is the most of either
class the communicator ever holds at v, whatever its peers write ahead. The bound covers one communicator:
communicators and clients that share a node share its pool.

The k-th page of share x that rank s sends rank d on the communicator is written to d's index
`base[d][x] + s RECEIVES + k mod W`. Rank s's pages of the share k mod W, then that plus W, and so on, land there in
turn, so page k is the index's (k / W + 1)-th landing (its `landed` count, which `mesh_entry` returns). Each rank
keeps, for every share and every peer, the pages it has sent it and the pages it has taken from it. Both ends count each pair's pages alike, because every rank makes the same calls
in the same order and cuts each message alike. So each rank's node, each share's base and its node's pool, gathered at the
init, are the only exchange that ever names an index.

**Sending.** The rank copies a page's bytes into a source page and calls `mesh_write` with the page's destinations.
A page to several ranks is one write to each `MESH_DESTS` of them in rank order, each destination the index its own
pair's page count names, each write copied into a source page of its own. A source page is copied into again only
once its `uses` are back to 1: the table word's own use, with no write, SEND or landing of it still under way
(`mesh_entry`). Where no source page is free, or the write ring is full, the rank takes what has landed and tries
again.

**Receiving.** Every page is independent of the others, and the rank takes every page it finds landed in the pass
that finds it. Each pass reads, for each peer, the indices of its next W pages due, each at an index of its own, so
no index is read for a later page while an earlier page there is due. The rank takes page k once its index's
`landed` count reaches k / W + 1. Where the index names a page, the rank copies the bytes due from it to the page's
place in its buffer, whatever the page holds, then invalidates the index (`mesh_invalidate`): the page is dropped,
and the index and its count stay. Where the index names no page (its page was taken under an earlier count),
nothing is copied. A page written W pages or more ahead of the page its receiver takes there replaces that page at
the index, and is taken as it. This is the defragment of
[design/DEFRAGMENT-pseudocode-20261122.md](../design/DEFRAGMENT-pseudocode-20261122.md): one local copy a page,
charged to this consumer.

A reduction combines each page position's contributions in rank order. A page whose rank's turn has come is combined
straight from the page. A page that lands before its turn is copied at once into the call's scratch (sized from the
phase, allocated at the first such page), and its index is invalidated; it is combined from the scratch in its turn.
So every landed page of the phase a rank is in leaves the pool as soon as the rank finds it. Pages its peers write
for a later phase or call stay at their indices until it gets there, one an index, which is what the window keeps
within half of each node's pool. The reductions are this library's own code over the page table's calls: the bridge
carries pages and knows nothing of what they hold.

The waits are the call's own. A call's wait ends once its pages have landed, or at the communicator's revoke, which
each pass reads; nothing else bounds it. Nothing in the transport waits.

## Communicators

`ncclCommInitRank` is NCCL's bootstrap. The unique id is the root's `host:port`: `NCCL_COMM_ID`, or else the
listener `ncclGetUniqueId` opens on the node's fabric address (its first address under `MESH_ULA_PREFIX`), or on its
host name where it has none. The root listens on that host alone where it is an IPv6 address. Rank 0 takes every rank's
report and sends each rank the whole table. A rank's report is its status, its node, the first receive index of each of its shares,
its node's pool (the window's P_v) and its standing listener's address. The communicator keeps the sockets, for a
split's or a grow's exchange.

Making a communicator does the following, and a rank that cannot go on fails every rank's init alike:

1. It attaches the process's client of its node's bridge where it has none (`mesh_attach`). This fails where every
   client slot is held, or where no region of a running bridge exists.
2. It takes the communicator's indices and pages.
3. It opens the rank's standing listener.

The bootstrap's waits are 100 ms slices that read the revoke; nothing else bounds them. Two ranks of a communicator
may share a node; a page between them is set at the destination index on that node without a SEND.

**Splits, shrinks and grows**:

- `ncclCommSplit` gathers every rank's (color, key) over the parent's bootstrap. Each color's communicator, its ranks
  ordered by key and then rank, meets at its first rank's standing listener.
- `ncclCommShrink` makes the kept ranks' communicator at the first kept rank's standing listener. It exchanges nothing
  over the parent, so an excluded rank, gone or not, takes no part. With `NCCL_SHRINK_ABORT` the parent is revoked
  first.
- A rank of no color, or an excluded one, gets NULL.
- `ncclCommGrow` meets at the id `ncclCommGetUniqueId` gives the parent's root, which the root gives its ranks over the
  parent's bootstrap. A joining process passes a NULL comm and the root's id.
- `config` is not read.

**Revoke and destroy**:

- `ncclCommRevoke` ends comm's calls under way as a value at their next look. Every later call on it is refused
  (`ncclInvalidUsage`), and `ncclCommGetAsyncError` reads `ncclSuccess` after it.
- `ncclCommDestroy` (and `ncclCommAbort`) revokes comm and waits for its calls under way to return. It then frees its
  indices, whose pages go back to the pool as their last uses end, closes its sockets and frees comm.
- The process detaches from its region once the region has expired and it holds no communicator of it.

A region its bridge's stop released, or that a bridge of another layout replaced, is expired. While the process holds
a communicator of it, a new one fails (`ncclRemoteError`). Once it has destroyed them all, its next one attaches the
bridge's next region.

## Failures

Failures are `ncclResult_t` values, and `ncclGetLastError` names what failed. A call records three failures on its
communicator: a write its node's table refuses, an index it cannot read, and no memory for the pages that land before
their turn. `ncclCommGetAsyncError` reads the first one recorded, and a communicator that recorded one fails its later
calls alike, until it is destroyed and made again. Every other failure (an invalid argument or usage, other memory,
the bootstrap) is the result of its call alone. A revoked communicator's calls return `ncclInvalidUsage`, and
`ncclCommGetAsyncError` reads `ncclSuccess` after the revoke.

A call never fails on what lands. The library checks nothing that lands, and a page's bytes are taken as they landed:
a write may land wrong, or not at all (mesh-call.h). No timer declares a peer lost: a page that never lands leaves its
call waiting until the application revokes or aborts that communicator (`ncclCommRevoke`, `ncclCommAbort`), as with
NCCL's own.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | `ncclGetUniqueId` | the bootstrap root's `host:port` |
| `NCCL_MESH_THREADS` | the process's first communicator | the threads a call's shares move on, 1 to 8 (default: half the performance cores, at most 8) |
| `MESH_REGION` | libmesh | another region's name on the same machine (mesh.h `mesh_name`) |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against |

## NumPy

`rdma/mesh_mpi.py` spells these calls as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import (its docstring lists them):

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
- `ncclCommRegister`/`Deregister` and the window and signal functions;
- `ncclRedOpCreatePreMulSum`/`ncclRedOpDestroy`;
- `ncclBcast`, `ncclSend`/`ncclRecv` and the `nccl*Config` collectives;
- `ncclGroupSimulateEnd`, `ncclSetEncryption` and the `ncclParam` functions.

A standard call that nothing in the repository calls is left out rather than kept as a stub.
