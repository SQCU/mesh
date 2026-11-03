# libnccl-mesh

NCCL-style signatures over the mesh: Macs joined by Thunderbolt 5 RDMA,
a bridge process (`mesh-flow`) on each. Supported collective callers compile
against `rdma/nccl.h`.
Its header comment is the contract; this page is
how to run it. Tensor programs use the whole-function declaration in [torch-mesh](torch-mesh/README.md).
NumPy may view their shared input and output pages. Uneven nodes and links: [design/heterogeneity.md](../design/heterogeneity.md).

## Build and run

The node's user keeps the library in the user's mesh checkout (`~/mesh`): `bin/mesh-user.sh` builds the bridge and the
libraries there, keeps the user's torch environment's base requirements (`~/.venv-mesh-uv`; a checkout's Python bindings,
`_mesh_c` and torch-mesh, are built in place by `make -C rdma modules` (`rdma/mesh_c_build.py` and torch-mesh's
`setup.py build_ext --inplace`, each only where its inputs changed) and loaded by path, as `node prepare` does), and runs the bridge as the user's launchd agent (`io.mesh.bridge`, design/transport.md) with
the node's bridge configuration, its one operand (`bin/mesh-user.sh CONFIGURATION`, etc/bridge/README.md; there is no
default); no part of it needs root. A program links or imports the checkout's library:

```
cc app.c -I ~/mesh/rdma -L ~/mesh/rdma -lnccl-mesh -Wl,-rpath,$HOME/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's, with NCCL's bootstrap: the id is the root's `host:port`
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens), where
  each rank's node meets the others'. The root listens on that host alone where it is an IPv6 address.
- `ncclMeshCommInitRank(comm, nranks, id, rank, paths, name, config)`: the same init and
  bootstrap (`id` the root's `host:port`, which the caller hands every rank; each rank's node is its bridge's); the
  map is its members' (below).
  - `name`: the communicator's, which its failures carry (NULL: the id).
  - `config`: NULL, or `blocking` not 0, is a blocking init; `blocking` 0 is NCCL's non-blocking init (below).
  - Every rank passes the same values. `ncclCommInitRank` is this call with no paths, name or config,
    and `ncclCommInitRankConfig` the same with its config.
- Both inits hold the communicator's streams at the bootstrap's two gathers. The first carries each rank's node and the
  nodes its bridge reaches; each rank then holds a stream on every physical cable joining it to another member, as
  it holds rows (the stream's owner word taken from 0 by one compare-and-swap). The second carries each rank's
  holding's status. A communicator names nodes and its own pages alone. An init fails as a value where two ranks are on
  one node (`ncclInvalidUsage`: its holdings are kept a node, so two ranks of one node would share one slot), where a physical cable has no stream free at both ends (EBUSY: a link has up to four, each
  held by one communicator, and one whose last holding its bridge is still ending is not free) or the link is gone
  (ENOLINK)
  (`ncclSystemError`); what it held is let go, and the next init reads the streams as they then stand. A bootstrap
  phase past its deadline ends the program (below).
- A connection to the root whose report fails (its peer closed it), or names no rank still to meet (a rank out of
  range, or one already met), is closed alone, and the root meets the next; one that sends nothing holds the root's
  read until the phase's deadline, which ends the program.
- Every wait of a bootstrap, blocking init or not, is a 100 ms slice after which the revoke and the phase's deadline
  are read again (design/rubrics.md M3, C5). A phase's deadline is `MESH_PREPARATION_NS` after it began: a non-root's
  connect, then its first gather; the root's accepts, every rank's report and its first gather under one deadline made
  as its accepts begin; the second gather has its own `MESH_PREPARATION_NS`. Past a deadline the program ends, the wait and its
  deadline named on stderr (design/eager-execution.md, assumption A4: the operator, 2026-10-23, "no wait may ever run longer than
  a deadline operand presented at the time the wait was constructed"); nothing waits on a peer that never comes.
- A non-blocking init sets `comm` at once, runs on a thread of its own and returns `ncclInProgress`;
  `ncclCommGetAsyncError` gives `ncclInProgress` until it ends, then its result. `ncclCommRevoke`, `ncclCommAbort` and
  `ncclCommDestroy` end it as a value with what it held let go: the revoke shuts its bootstrap's sockets down, and its
  connect, its root's accept and its gathers read the revoke at each 100 ms slice. `ncclCommAbort` or `ncclCommDestroy` then frees the
  communicator, failed or not; no call but these four takes it before its init has ended.
- A rank's connect and the root's accept hold nothing of the node's bridge: a rank attaches once its bootstrap has
  connected, or accepted every rank, and its report (its node and the nodes its bridge reaches) follows that attach. A
  blocking init's root waits in its accept, at most its phase's deadline.
- A rank's connect to its root has the deadline `MESH_PREPARATION_NS` after its first try: an init toward a root that does
  not listen ends the program then, never retried for good.
- Making a communicator, a lone rank's too, attaches the process's client of its node's bridge ([Memory](#memory)),
  so the node's bridge runs; it fails where `ncclMemAlloc` would.

Open: who names cables. NCCL.md:124-126 at a6617e1a: "Every physical cable has a stable wire identity and its own
channel; parallel cables remain separate. Bootstrap acquires a mutually free stream on every cable between members."
zorvek.md:285-286 and :332-333 at a6617e1a: "The defect: the collective names a cable (a stream per wire, routes as
cable lists, a transfer's cable index)"; "Links, cables, queue pairs and lanes, and the order in which a link's queue
pairs carry positions, are the bridge's alone".

The bridges' records are a map whatever reaches what: a node that still has a record may be reached by no bridge,
and the whole map (`mesh.current()` with no members, `mesh_map` over none, mesh-stat's `map`) holds it as it is.
`mesh_link_map_check` refuses only what is no map; whether a map joins every two of its nodes is
`mesh_link_map_joined`, which names the first two it leaves apart. The trees and the
collectives run only on a map that joins their members.

Every member makes the map from the same gathered reports, so the members' maps are one, and every rank passes the same
paths; nothing is compared at run time.

**The partials merge.** `ncclMeshRedOpCreatePartials(&op, dim, ncclFloat32, comm)` makes a reduction operator, as
`ncclRedOpCreatePreMulSum` makes one (freed by `ncclRedOpDestroy`): split-KV attention partials merged by their
log-sum-exp [Dao et al. 2023, Flash-Decoding; FlashInfer's state merge]. The operand is rows of `2 + dim` float32,
each row `[m, l, O: dim]` (a row's maximum score, its sum of exponentials, its unnormalized output); two rows merge to
the greater m, each one's l and O rescaled by e^(its m - that) and summed, a row with no positions (l not above 0)
counting for none. The result is the merged partial, its O / l the attention over every member's positions.

## What each program owns

The crossing path's mechanisms, each judged by the contract of the program that owns it (metal-microbench
output_data/transport-buffers-20261021/LIBRARY.md ranks the changes of 2026-10-21):

- **The bridge** (design/transport.md): its devices, queue pairs, lanes, the receives it
  posts, the region's registration. It reads `ready` and each SEND's word, and writes `sent`
  and `landed`.
- **The library**, by NCCL's contract: the caller touches no buffer of a collective in flight and issues the same
  collectives in the same order on every rank; the library delivers correct bytes.
- **The data function**: its own buffers and their order.

## Memory

The node's bridge registers its region, its configuration's share of the node's memory (etc/bridge/README.md), on
every device when the device comes up, and never again while the device is up
(design/transport.md). The operator, 2026-10-15: "the bridge's job is to allocate essentially all of the
memory on the entire device so that there are large already 'pinned' buffers which are ready to receive surprise
transfers in whatever order they happen from whatever peers and links are operational ... ontologically it must be
the responsibility of client programs to know how much allocation they need ... the bridge is there to behave like a
system call".

- The process is one client of its node's bridge: one attach of the node's region, made at its first `ncclMemAlloc` or
  communicator. Its slot owns the process's leases and holds the streams its communicators use.
- `ncclMemAlloc(ptr, size)` is the lease, the system call: whole pages of the region covering `size` (one page for 0),
  already registered; a lease registers nothing. Its bookkeeping is the region's (the lease's rows are owned by the
  client's slot), so when the process exits or dies its bridge returns the pages as it returns every retired client's
  rows.
- A lease the region cannot meet is `ncclSystemError` naming the bytes asked and the pages free. While every client
  slot of the bridge is held it is `ncclSystemError` naming `mesh.h` `MESH_CLIENTS`. Where the process attaches no
  region of a bridge that runs, it names the requirement: the node's bridge runs with a configuration whose region
  holds the client's footprint (`ncclRemoteError` where the bridge stopped, else `ncclSystemError`). Nothing falls back
  to other memory.
- `ncclMemFree(ptr)` logically frees the lease without waiting. Registered local DMA reads, including descriptors
  not yet taken by the bridge, keep its pages alive. Once no holding can post a future read, already-posted reads can
  remain protected by the bridge's pins until completion. This handoff permits asynchronous retirement, not an
  application overwrite. No application receipt or remote ACK is added.
- `ncclCommRegister` takes a buffer in the region (its handle is the buffer itself) and refuses any other with
  `ncclInvalidArgument` (lease it with `ncclMemAlloc`). `ncclCommDeregister` releases nothing.
- A lease is an operand of the node's region, which outlives a restart of its bridge: the next bridge adopts it
  (design/transport.md section 2), so a lease, its pages and every buffer over them stand across a bridge's restart,
  and nothing is reloaded. A region a release (or a bridge of another layout) expired is returned as a value: while the
  process holds a lease or communicator of it, its next `ncclMemAlloc` and communicator fail with `ncclRemoteError`
  naming the expiry, and `ncclMeshRegionExpired` reads 1; once the process has freed every one (`ncclMemFree`,
  `ncclCommDestroy`), or ended, the library lets that region go and its next lease attaches the bridge's next region;
  a process that held a communicator or a lease's `ncclMeshMemBuffer` ends instead, since the library made Metal
  buffers over the region for them, which its GPU work and residency set may hold; a caller's own Metal buffers over a
  lease are released, with their residency, before `ncclMemFree`, or the caller ends. A process never
  holds two regions. This is the recovery
  doctrine (the operator, 2026-10-14): "functions can fail from any one operand
  becoming invalidatd, servers should not fail when any operand is invalidated (ideally)"; "within call failures
  cannot be resume and instead must return an error (which a caller can presumably retry against immediately if that's
  what they want)".

The client sizes the region on each node it runs on, and on the nodes it peers or streams with: weights, KV and every
buffer a peer reads are leases. The operator, 2026-10-15: "why would weights or kv be outside of the memory pinned
pool? being outside of it means that they cannot be zero copy passed to peers. ... pipeline parallelism, context
parallelism, and megatron column parallelism are all required. ... you can't defend against this; it's *your code*
you're pretending you need to defend against; you can write it right all along."

Porting:

- An NCCL program takes its collectives' buffers from `ncclMemAlloc` (NCCL's allocator for the buffers it registers), and
  its node's bridge configuration's region holds them.
- Python's lease is `mesh.lease(nbytes)`: a writable buffer for `numpy.frombuffer` or `torch.frombuffer`, returned to
  the region when it and everything over it are gone.
- torch-mesh's own CPU staging and temporaries are leases ([torch-mesh](torch-mesh/README.md)).
- `ncclMeshMemBuffer(ptr, &buffer, &offset)` is the Metal buffer over the whole lease holding `ptr` (any byte of it)
  and ptr's offset in it, made once and released by `ncclMemFree`: the buffer a tensor allocator makes MPS tensors over
  (torch-mesh's region tensors).
- **Direct transport.** The operator, 2026-10-21: "zero copy methods must always be used over access by copy because
  memory bandwidth and flops are too scarce on this hardware api"; "good performance cannot be opt-in ... code which
  can't perform well can fail instead of being run". The bridge sends a crossing's bytes from the pages they lie in and
  lands them in the pages they belong in: a COPY lands in its destination, and a combine lands in a landing area of the
  communicator's and is combined from there. The 2026-10-24 operator amendment
  permits explicit masked streaming data functions; those functions are
  numerical producers/consumers, not hidden native transport repair. Current
  native transport moves one whole page a unit (the page table's quantum, block x pgsz bytes: `mesh.h`
  `struct mesh_zwoggle_entry`) and one SGE per unit
  (`max_sge` is 1). The bridge registers the region in `MESH_EXTENT` (1 GiB)
  extents inside the 4 GiB banks no registration may cross (`tbt_post_recv`
  keeps only an address's low 32 bits: `mesh-verbs.h`).
  Therefore:
  - every crossing call's send buffer, and every buffer it sends from or lands in, is a lease's (in this node's region);
  - each resulting SGE stays inside one registered extent;
  - hardware credits round span service to 4 KiB frames without adding bytes
    to its SGE payload.
- The NIC reads a SEND's pages until it finishes. A local group's end does not establish
  that fact. Source and relay leases must remain live through their downstream uses;
  `ncclMemFree` follows the data program's last use, not merely publication.
- `ncclCommRegister` refuses a buffer outside the region.

## Routes between unlinked ranks

A communicator retains every paired physical cable between its admitted ranks.
Every node used as a relay participates in that physical communicator.

## Rules a program keeps

- One rank a node.
- All ranks declare the same complete function and physical participants.
- Every buffer a crossing reads or writes is a lease.
- Communicators know nothing of one another, and no two hold one stream: a link has up to four (`MESH_LINK_STREAMS`;
  fewer where its plan has fewer lane pairs, design/HETEROGENEOUS-LINKS.md), so that many communicators use a node
  pair at once, and one more is refused (EBUSY) until one of them is destroyed.

## Failures

A group's entries wait for no room: its tables are linked to the bridge whole (design/zorvek.md zinth). A member
that fails, is revoked or ends sends nothing more: its bridge posts no page of a holding that ended, and nothing of
its own. A holding given with units still outstanding takes its stream out of its link's pairing, and a link none
of whose streams is live is lost and paired again from 0 (design/transport.md section 14).

A communicator keeps the map its init made, so a change that leaves its links working (a link lost elsewhere, a
node that joins) reaches the next communicator (metal-microbench makes each call's communicators at the call's
start). `ncclCommRevoke` is NCCL's, from any thread and under the communicator's own lock alone: its operations in
flight are revoked and its resources kept (its streams stay held, and its bridge sends what its groups published
and nothing else); an init still running (a non-blocking one's) ends as a value; `ncclCommFinalize` after it
returns `ncclSuccess`, and its `revokeFlags` is not read. The caller must complete its recorded function before
calling `ncclCommDestroy`, which returns its streams.
The library has no private queue or submission to wait for; work encoded in the caller's recorder is the caller's
to finish first, as NCCL's
contract has it: a kernel of it still queued when its streams are given back would write a stream the next holder
may take. Likewise a lease a group of a revoked or failed communicator named is the caller's to keep until
`ncclCommDestroy`: the communicator still holds its streams, its bridge may still post receives into that lease for
entries already presented, and only the give ends that (design/transport.md section 14, "A holding's end").

The process holds one of its bridge's client slots (`mesh.h` `MESH_CLIENTS`) whatever its communicators.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | library (ncclGetUniqueId) | the bootstrap root's `host:port` |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against (default its parent; `node prepare` passes its checkout's) |

## Not implemented

ncclCommInitRankScalable, ncclCommSuspend/Resume,
ncclCommMemStats, the window and signal functions, ncclGroupSimulateEnd,
ncclSetEncryption, the ncclParam functions (`nccl.h`).
