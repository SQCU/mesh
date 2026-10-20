# libnccl-mesh

NCCL's C API over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each. Code
written against NCCL's `nccl.h` compiles against `rdma/nccl.h`. Its header comment is the contract; this page is
how to run it. PyTorch programs use it through [torch-mesh](torch-mesh/README.md). NumPy programs use it through
`mesh_mpi.py` (below). Uneven nodes and links: [design/heterogeneity.md](../design/heterogeneity.md).

## Build and run

The node's user keeps the library in the user's mesh checkout (`~/mesh`): `bin/mesh-user.sh` builds the bridge, the
libraries and the Python bindings there, keeps the user's torch environment with torch-mesh and the checkout's library on
its path (`~/.venv-mesh-uv`), and runs the bridge as the user's agent (`io.mesh.bridge`, design/transport.md) with
the node's bridge configuration, its one operand (`bin/mesh-user.sh CONFIGURATION`, etc/bridge/README.md; there is no
default); no part of it needs root. A program links or imports the checkout's library:

```
cc app.c -I ~/mesh/rdma -L ~/mesh/rdma -lnccl-mesh -Wl,-rpath,$HOME/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's, with NCCL's bootstrap: the id is the root's `host:port`
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens, on the node's fabric address where it holds one), where
  each rank's node meets the others'. The root listens on that host alone where it is an IPv6 address.
- `ncclMeshCommInitRank(comm, nranks, id, rank, paths, programs, count, small, name, config)`: the same init and
  bootstrap (`id` the root's `host:port`, which the caller hands every rank; each rank's node is its bridge's), with
  the collectives as operands; the map is its members' (below).
  - `paths`: the routing tables every collective of the structural table (below) is compiled along, each member's
    trees, checked against the communicator's map. NULL means the map's shortest-path trees.
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the structural table
    (below).
  - `small`: not read (the wire's classes are its streams').
  - `name`: the communicator's, which its failures and its splits' names carry (NULL: the id).
  - `config`: NULL, or `blocking` not 0, is a blocking init; `blocking` 0 is NCCL's non-blocking init (below).
  - Every rank passes the same values. `ncclCommInitRank` is this call with no paths, programs, small, name or config,
    and `ncclCommInitRankConfig` the same with its config.
- Both inits hold the communicator's streams at the bootstrap's two gathers. The first carries each rank's node and the
  nodes its bridge reaches; each rank then holds its bridge's stream toward each other member's node it is joined to, as
  it holds rows (the stream's owner word taken while the stream is free). The second carries each rank's holding's
  status and where its streams' units start. A communicator names
  nodes alone. An init fails as a value where two ranks are on one node (`ncclInvalidUsage`), where no stream
  toward a member's node is free at both ends (EBUSY: a link has four, each held by one communicator) or the link is
  gone (ENOLINK)
  (`ncclSystemError`), or where its bootstrap's phase passes its bound (`ncclRemoteError`, ETIMEDOUT); what it held is
  let go, and the next init reads the streams as they then stand.
- Every wait of a bootstrap, blocking init or not, is a 100 ms slice after which the revoke and the phase's end are read
  again (design/rubrics.md M3, C5). A phase ends `MESH_PLATFORM_NS` after the init began where no rank was met, and
  `MESH_PLATFORM_NS` after its first rank met (the root's first accept, a non-root's connect) for the rest of it; the
  second gather has its own `MESH_PLATFORM_NS`. A root that no rank meets therefore fails as a value and is made again
  by its caller; nothing waits on a peer that never comes.
- While the process holds a lease of a bridge that stopped, its next lease (`ncclRemoteError`), communicator
  (`ncclInvalidUsage`: the process's own leases, which only it can free) and `ncclCommRegister` are refused naming it,
  until it has freed every such lease: a process never holds leases of two regions. A node's window is most of its
  memory, and GPU work on a stopped region's leases beside the new window's registration wired the Mini's whole memory
  (10-20, metal-microbench output_data/bridge-20261019/REVIEW-131.md).
- A non-blocking init sets `comm` at once, runs on a thread of its own and returns `ncclInProgress`;
  `ncclCommGetAsyncError` gives `ncclInProgress` until it ends, then its result. `ncclCommRevoke`, `ncclCommAbort` and
  `ncclCommDestroy` end it as a value with what it held let go: the revoke shuts its bootstrap's sockets down, and its
  connect, its root's accept and its gathers read the revoke at each 100 ms slice. `ncclCommAbort` or `ncclCommDestroy` then frees the
  communicator, failed or not; no call but these four takes it before its init has ended.
- A rank's connect and the root's accept hold nothing of the node's bridge: a rank attaches once its bootstrap has
  connected, or accepted every rank, and its report (its node and the nodes its bridge reaches) follows that attach. A
  blocking init's root waits in its accept, at most its phase's bound.
- A rank's connect to its root gives up `MESH_PLATFORM_NS` after its first try: an init toward a root that does not
  listen fails as a value (`ncclRemoteError`, ETIMEDOUT), never retried for good.
- `ncclCommSplit` takes the ranks of a color and holds the child's streams over comm (NCCL's bootstrapSplit, with the
  init's two gathers), streams of its own, as every init takes. `ncclCommShrink` (NCCL's signature) leaves out the ranks named, those that departed, with no
  exchange among the rest, so it holds no stream: a shrink to one rank is made, and one to two ranks or more is refused
  (`ncclInvalidUsage`, ENOLCK).
- Making a communicator, a lone rank's too, attaches the process's client of its node's bridge ([Memory](#memory)),
  so the node's bridge runs; it fails where `ncclMemAlloc` would.

Each communicator makes its map at its init from its members' reports (`nccl-mesh.c` `map_of`), alike on every member,
and keeps it for its life: its members its nodes, each marked by its bridge's node number, two joined by one link
where each one's bridge reaches the other's node, at the lesser of the two ends' bandwidths, and the routing table
(`mesh_routes`: each node's in-tree of shortest routes, by fewest links and then over the faster hops, a parent among
equals the least hash of its mark, its parent's and the root's, so routes of equal length spread rather than pile onto
the lowest-numbered node, and a route moves only where the shortest routes between its ends change, never for a link
lost elsewhere or a node joining; a route between two nodes the higher-marked one's in-tree path, one path both ways).
No node outside the communicator is on it: the bridges forward nothing. Two ranks are linked where a link joins them; a
communicator has a channel of each class toward each member it is joined to, and none toward another ([Routes between
unlinked ranks](#routes-between-unlinked-ranks)). A later communicator reads the reports again, so a node that joins,
a node gone or a link lost (down until its ports' next agents pair it again, with nobody acting: design/transport.md
section 14) reaches the next communicator with no edit and no restart. libmesh's whole map of every node's record (`mesh_map` over no members, Python `mesh.current()`) joins two
nodes by the same rule.

The bridges' records are a map whatever reaches what: a node that still has a record may be reached by no bridge,
and the whole map (`mesh.current()` with no members, `mesh_map` over none, mesh-stat's `map`) holds it as it is.
`mesh_link_map_check` refuses only what is no map; whether a map joins every two of its nodes is
`mesh_link_map_joined` (Python `mesh.joined`, which names the first two it leaves apart). The trees and the
collectives run only on a map that joins their members: a communicator's collective on a map that does not fails as a
value naming two members no path between members joins (`ncclInvalidUsage`).

Every member makes the map from the same gathered reports, so the members' maps are one, and every rank passes the same
paths, programs and forms; nothing is compared at run time.

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR), the communicator's table of them. The caller passes its own table (`ncclMeshCommInitRank`'s `programs`;
`mesh.compile` in Python, `chunks` its cut, `whole` the single-phase all-reduce); with none, the communicator compiles
the structural table at its first call that needs it, along the caller's `paths` or else the map's shortest-path trees
(each member's in-tree of the map's routes, its parents among equals spread over the links):

- the all-reduce single-phase (Trivance: each member's sum up its own tree, the map's diameter in dependent crossings;
  at two members, and wherever every two members are linked, the direct exchange);
- the reduce-scatter and the all-gather, segment s up and down s's trees;
- each root's reduce and broadcast, up and down the root's trees.

Each is uncut: a step's piece is cut on the wire into slots of its class, each carrying whole elements of the operand
(`mesh_wire_pieces`: no element is split, so a combine of a slot's piece combines whole ones). A call runs its table's
first program of its collective (and root) whose size bound its operand is under. Nothing about a program is decided
where a call runs.

**Counted all-to-all.** `ncclMeshAlltoAllCounted(send, sendrows, counts, sendsegs, recv, capacity, received, landed,
recvsegs, arrived, row, type, comm, stream)` is MPI_Alltoallv whose counts the GPU wrote (int64, a segment a destination
rank: an MoE router's per-expert counts grouped by the rank holding each expert): each rank's segment for a peer moves
with its rows in one exchange, the receiver's rows are packed in rank order, and the landed counts reach `received`
and `landed` with `*arrived` set to 1 while the rows may still be on the wire (2: the call failed). On the Metal path
no host reads a count: every row's place is computed on the GPU from the counts (`nccl-mesh-metal.m` `counted_*`), its
waits are in-stream waits on the landed words its bridge raises, and `received` and `arrived` are host memory
in leases, which the GPU writes. The positions are laid without the counts: each pair carries room for
`min(sendrows, capacity)` rows (every rank passes the same `sendrows`, `capacity`, `row` and type), and counts that pass
that layout fail as a value with `*arrived` 2. Reproduction target (metal-microbench docs/measurement.md 2026-09-30,
part B): ep at 27.5-27.7 ms a step (the host-split `all_to_all_single` form read 39.3-40.1), the counted call's overhead
473-480 us, the host idle 7.4-7.9 ms. A crossing counted call had failed since its restoration (mesh 598cb6cf: a NULL
program read where the uneven collectives check their segments) and, on a reused window, could read a pair's position
with no rows as a filler; with both fixed, on the loopback fabric (a line of 4) every rank's counts and rows land,
host path and walk.

**All-to-all with host counts.** `ncclMeshAlltoAllV(send, sendcounts, recv, recvcounts, type, comm, stream)` is
MPI_Alltoallv with its counts on the host: rank r sends `sendcounts[q]` elements to rank q and receives
`recvcounts[q]` from it, both packed in rank order (rank r's `sendcounts[q]` is rank q's `recvcounts[r]`). It is the
counted all-to-all's call with its counts known to both ends: no count crosses, each pair's elements move as one
message of their own size, one call a group. NCCL's `ncclAlltoAll` is it with every count `count`, and `ncclGather` and
`ncclScatter` its rooted forms (every count 0 but toward or from the root), where they were each rank's sends and
receives in a group (NCCL's own lowering, `classifyCollToP2pTasks`). On the loopback fabric (a line of 4) every rank's
elements land, host path and walk, the all-to-all, every root's gather and scatter and an uneven all-to-all with
zero counts among its pairs.

**The partials merge.** `ncclMeshRedOpCreatePartials(&op, dim, ncclFloat32, comm)` makes a reduction operator, as
`ncclRedOpCreatePreMulSum` makes one (freed by `ncclRedOpDestroy`): split-KV attention partials merged by their
log-sum-exp [Dao et al. 2023, Flash-Decoding; FlashInfer's state merge]. The operand is rows of `2 + dim` float32,
each row `[m, l, O: dim]` (a row's maximum score, its sum of exponentials, its unnormalized output); two rows merge to
the greater m, each one's l and O rescaled by e^(its m - that) and summed, a row with no positions (l not above 0)
counting for none. The result is the merged partial, its O / l the attention over every member's positions. Every
reducing collective takes it (`ncclAllReduce`, `ncclReduce`, `ncclReduceScatter`, `ncclMeshReduceScatterV`, counts in
whole rows), inside a group or not: one group, the communicator's program of that collective with a row for an
element, each row merged up its tree where a sum adds an element, the wire's positions carrying whole rows. This is the
planner's `partials` lowering of a split-KV attention (metal-microbench docs/placement.md Crossings): the engine
(metal-microbench decode_attention_partial, mesh_rank.m issue) and torch-mesh (ProcessGroupMesh.merge_partials) each
lay their partials as these rows and call `ncclAllReduce`, or `ncclReduce` to the one member reading them, with it.
Reproduction target: never measured; the planner's pricing of the three-group form it replaces at 3, 4 and 6 members
of a full mesh was 331, 331 and 226 us against the gather's 534 a chunk's layer (metal-microbench
output_data/planner-20261015/pass2 engine2-after.json); on the loopback fabric (a line of 4) every rank's merge exact
to 1e-5 against double arithmetic, host path and walk, 3 to 2,100 rows of 64, one and two positions a message on
either channel.

**Forms as configuration.** `ncclMeshCommForms(comm, forms, count)` is the communicator's configuration operand: a
list of `ncclMeshForm` entries, each a collective (and root, -1 every root) below a size bound, compiled ahead of the
structural table at each map read, so a call takes the first form that fits it and the default otherwise:

| field | form |
|---|---|
| `trees` | `ncclMeshTreesShortest` (the communicator's paths, else the shortest-path trees) or `ncclMeshTreesPacked` (`mesh_trees_pack` at the entry's `eps` and `most` trees a root: Blink's packed spanning trees, every link a share) |
| `shares` | a packed form's cut of each segment among its root's trees (root r's k-th tree, heaviest first, `shares[r x most + k]`; NULL: equally): the configuration's allocation, never the flows the packing infers from the map's links |
| `whole` | an all-reduce's: 1 single-phase, 0 two-phase (the reduce-scatter and then the all-gather along the same trees) |
| `chunks` | each tree's part cut into chunks, pipelined down and up the trees (1: uncut) |
| `below` | the bytes a call is under to take it (0: any size): the size classes |

These are the forms mesh cc8a6f8e^'s default table chose among by a price (its `sets[1]`, single-phase or not, 1 to 64
chunks, by size class); the price is gone and the caller's configuration document names the form (metal-microbench
docs/principles.md section 6: a decision is structure, a measurement the calls make, or an algorithm restored from
history). With no forms nothing changes: the structural table as above. Every rank passes the same forms, set before
the communicator's first group that crosses a link; splits and shrinks keep them; a communicator over the caller's own
programs refuses them. A form that cuts the operand (two-phase, chunked) is not fusable (`ncclMeshFusableRooted`
answers 0 for both steps). Reproduction target: simulated 0.51x the default's time at 1 GiB on six-node maps and 0.58x at 16 MiB, the ring4
16 MiB all-reduce 2.18 to 1.71 ms, a six-node full mesh's 1 GiB all-gather 158.1 to 20.5 ms (metal-microbench
output_data/library-20261015/pass2/sims), and the pair's 256 and 512 KiB calls 57.4 to 31.2-36.2 us
(pair_programs.json): never measured on more than two nodes.

**Uneven segments.** `ncclMeshAllGatherV(send, recv, counts, type, comm, stream)` and
`ncclMeshReduceScatterV(send, recv, counts, type, op, comm, stream)` are MPI_Allgatherv and MPI_Reduce_scatter: rank
r's segment is `counts[r]` elements (any of them 0), packed in rank order. Each is one program, the communicator's
all-gather or reduce-scatter, its operand cut into those segments (`mesh-plan.h` `mesh_program_steps_cut`: a part
within segment s takes its tree's share of `counts[s]`: a configured form's `shares`, else equal), so a capacity-shaped division of the work (metal-microbench
docs/placement.md) is one tree program over unequal segments rather than a rooted broadcast or reduce a rank. Every
rank passes the same counts; equal counts are NCCL's own calls. Reproduction target (metal-microbench docs/measurement.md
2026-09-30): tp SP=1 at capacity-shaped parts 255-256 ms a step; per call, the SP=1 all-gather 214-229 us and
reduce-scatter 196-212 us; one uneven all-gather against the same segments as N broadcasts.

## Two paths

- **Host path.** A NULL stream: host pointers, copied through its streams' rings, and each call is synchronous. It
  works for any array on any node whose bridge runs (NumPy, CPU tensors). It is functional, not the measured path.
- **Metal path.** An `ncclMeshStream`: the group is encoded into a Metal command buffer, and every buffer is an
  `ncclMeshBuffer` (an `MTLBuffer` and a byte offset). The kernels copy each operand into its streams' rings, the
  bridge's, and land what arrives from them, with no staging on the host; a send buffer that is a lease's Metal
  buffer is read in place ([Memory](#memory)).
  - torch-mesh runs MPS tensors here.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`).
  - **Metal 4.** A stream whose `argumentTable` is set encodes into the caller's `MTL4ComputeCommandEncoder`
    (`commandEncoder`) between the commands before and after it, every binding through that argument table (at least
    six buffer bindings; each dispatch's arguments a record of the library's ring, never `setBytes`), on pipelines of
    the caller's device, the library's buffers added to the caller's `residencySet`; nothing is committed or
    signalled. Every collective and group takes the stream. Its waits are bounded by the stream's `polls`
    as the classic form's are, 0, or more than `NCCL_MESH_POLLS`, being `NCCL_MESH_POLLS` ([Bounded
    waits](#bounded-waits)). Without a workspace its scratch is the communicator's, so such a group completes in its
    encoder (no `issue`). A recorded program's replay calls it at each crossing it marks
    (metal-microbench docs/kernels.md#any-program).
  - A wait polls the landed word of its run's last unit, which the bridge sets `~0` where the link is lost, and its
    communicator's stop word, which a revoke sets ([Failures](#failures)); the stream's `polls` bounds it ([Bounded
    waits](#bounded-waits)).

### Per-channel progress

A group's completion runs every channel on its own (`nccl-mesh-metal.h` `metal_record`, `metal_tracks`). This is
NCCL's per-peer progress, chosen by the operator on 2026-10-20 in place of one serial order of the whole group.

- **Tracks.** Each channel's remaining positions are a track with two cursors: its publications and its consumptions,
  each in position order.
  - A publication runs once its track has consumed the position LAG back (the ring's credit).
  - A consumption runs once its landing came, and it is combined or copied where it belongs.
  - A step's first piece that has a receive before it (`mesh_group_before`) waits on that receive's channel having
    consumed it.
  - Neither cursor waits on the other beyond the credit.
- **On the GPU.** The tracks are one dispatch, one thread group per track. Thread 0 picks the next item whose
  conditions hold, every thread runs it through the same kernel bodies the single dispatches use, and a consumption
  raises its track's count. A group that can pick nothing for `polls` passes records its blocked items' failures, as
  a wait does.
- **On the host executor.** The same tracks run in one cooperative loop.
- **What ranks share.** No rank orders anything another rank must share. Each channel's two ends agree on its
  positions (`mesh_group_lay`) and nothing else, so groups of any calls, including counted all-to-alls of per-rank
  shapes, among any number of ranks, complete wherever their steps' dependencies do.
- **Not completion.** Issuing a group (`issue`) encodes no wait and is unchanged.
- **Residency.** Thread groups of one dispatch wait on each other only through a step's receive before it. Metal does
  not promise they are all resident; a wait on one that is not reaches its bound and fails as a value.

### Bounded waits

Every GPU wait is bounded (design/transport.md; the operator, 2026-10-18: "you cannot be using a low level
kernel like a dma or networking send/rcv channel for more than 5 seconds without something happening/concluding").
`ncclMeshStream.polls` is the most loads of a landed word one wait of the group makes (`rdma/mesh.metal`
`mesh_wait_bounded`): the caller's bound for the group's crossings, carried per crossing. 0, or more than
`NCCL_MESH_POLLS` (nccl.h), is `NCCL_MESH_POLLS`, in either form: torch-mesh's waits and the engine's encoded crossings
pass 0. A wait that reaches its bound records its position, its channel and the bound in its communicator's failure
words, every later dispatch of the communicator does nothing, and the group fails as a value after its command buffer
(`ncclCommGetAsyncError`, or the communicator's next group: "landed nothing within N polls").
`ncclMeshComplete` bounds the completion's waits by the stream it is given.

The caller derives the bound from its plan: the lag it expects at the crossing (the record's wait there, times the
margin its plan names) over the period of one load on the node's GPU. That period, read on the idle Mini (M4 Pro,
2026-10-16; metal-microbench output_data/library-20261015/round3 `poll_period.m`, a word nothing writes, waited on in
both of the library's shapes, one thread and thread 0 of a 1,024-thread group, 256 to 65,536 polls a dispatch):

| GPU | clock | ns a load |
|---|---|---|
| M4 Pro (Mini) | working (after ~4 ms of work; 65,536 polls in 16.4 to 16.8 ms) | 251 to 265 (fit 253 to 255) |
| M4 Pro (Mini) | rising (an idle GPU's first ~4 ms of work) | 1,004 to 1,097 |
| M5 Max | not read (its GPU runs the operator's work) | |

So a bound stands for at least polls x 251 ns on the Mini (the time past which a late peer fails the call) and holds
the GPU at most about polls x 1.1 us (a wait that never lands). `NCCL_MESH_POLLS` is `MESH_PLATFORM_NS` over the
slowest period read, 4.5 x 10^6 loads: at most 4.95 s of the GPU, and about 1.2 s at the Mini's working clock, past
which a peer that has not yet published fails the call. The M5's period is not read; if it is slower than 1.1 us, the
bound passes 5 s there.

Types: the Metal path moves the integer types, float16, float32, bfloat16 and fp8 (e4m3fn, e5m2: decoded, combined in
float32 and encoded to nearest even as the host does, overflow to NaN or infinity; metal-microbench docs/kernels.md
Float8); the host path adds float64.
Ops: sum, prod, max, min, avg; premul-sum takes a host immediate.

## Memory

The node's bridge registers its window, its configuration's share of the node's memory (etc/bridge/README.md), on
every device when the device comes up, and never again while the device is up
(design/transport.md). The operator, 2026-10-15: "the bridge's job is to allocate essentially all of the
memory on the entire device so that there are large already 'pinned' buffers which are ready to receive surprise
transfers in whatever order they happen from whatever peers and links are operational ... ontologically it must be
the responsibility of client programs to know how much allocation they need ... the bridge is there to behave like a
system call".

- The process is one client of its node's bridge: one attach of the node's region, made at its first `ncclMemAlloc` or
  communicator. Its slot owns the process's leases and holds the streams its communicators use; a communicator takes
  no bytes of the window (its rings are its streams', the bridge's).
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
- A lease is an operand of the node's window, which outlives a restart of its bridge: the next bridge adopts it
  (design/transport.md section 2), so a lease, its pages and every buffer over them stand across a bridge's restart,
  and nothing is reloaded. A window a release (or a bridge of another layout) expired is returned as a value: while the
  process holds a lease or communicator of it, its next `ncclMemAlloc` and communicator fail with `ncclRemoteError`
  naming the expiry, and `ncclMeshWindowExpired` reads 1; once the process has freed every one (`ncclMemFree`,
  `ncclCommDestroy`), or ended, the library lets that window go and its next lease attaches the bridge's next window;
  a process that held a communicator or a lease's `ncclMeshMemBuffer` ends instead, since the library made Metal
  buffers over the window for them, which its GPU work and residency set may hold; a caller's own Metal buffers over a
  lease are released, with their residency, before `ncclMemFree`, or the caller ends. A process never
  holds two windows. A communicator whose
  streams a restart or a new pairing opened again reports its failure through `ncclCommGetAsyncError` before any group
  runs on it, and is made again. This is the recovery
  doctrine (design/transport.md, mandates 12 and 13, the operator, 2026-10-14): "functions can fail from any one operand
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
- `ncclMeshMemBuffer(ptr, &buffer, &offset)` is the Metal buffer over the whole lease holding `ptr` (any byte of it)
  and ptr's offset in it, made once and released by `ncclMemFree`: the buffer a tensor allocator makes MPS tensors over
  (torch-mesh's window tensors).
- A Metal-path collective whose send buffer is a lease's Metal buffer reads it in place where its program allows: a
  SEND of a range no step of the rank wrote reads the send buffer, the first combine of such a range writes send op
  piece (`combine3_*`), the result's ranges no step writes are copied from it after the publications, and no staging
  or scratch copy of the operand is made (a scratch buffer no step writes is never made). The two-rank all-reduce out of
  place, for one, makes no copy of its operand where it made two (the stage into the result and the scratch its SEND
  read). Every such read is in the group's own command buffer or encoder (an issued group stages its operand as
  before), and the bridges read only the streams' rings, so once that command buffer completes nothing reads the
  lease and `ncclMemFree` may return its pages. The bytes on the wire still pass through the streams' rings, the
  bridge's; in-place, an operand's own copies go, not the ring's.
  Reproduction target (metal-microbench docs/measurement.md 2026-09-30 part A, 95967e6..42a16ed): per 16.8 MB
  all-reduce, the library's copies of the operand 16.78 MB to 0; tp 271.5-272.2 ms a step.
- A host-path group waits for its own communicator's GPU groups alone and fails while one of them is issued and not
  complete; it spins on its landings, never parks, and holds no lock another communicator's groups take, so a rank
  waiting in a host-path collective (an `ncclBroadcast` from a peer) does not stop its other communicators.
- The host path copies every operand through its streams' rings, a lease or not; `ncclMeshMemUse(ptr, bytes, event,
  value, write)` declares the caller's GPU work on a lease (an `MTLSharedEvent` and the value it reaches), which a
  host-path group reading or writing that lease waits for (until the GPU's work reaches the value). `ncclCommRegister`
  refuses a buffer outside the window.

## Routes between unlinked ranks

The bridges forward nothing (design/transport.md). A communicator's init holds a stream only toward the members whose
nodes its bridge reaches and whose bridges reach its node, and its channels go only toward those members. A collective
whose members no path between members joins fails at its first call that needs its programs, naming two, and a call
whose program or point-to-point step moves bytes between two members not joined fails at that call
(`ncclInvalidUsage`). A node a route would cross between two members takes part only as a member of the
communicator; a program may have it contribute nothing (`mesh-plan.h` `mesh_compile`'s `contributors`).

## Rules a program keeps

- One rank a node.
- A bidirectional exchange with one peer is one group (`ncclGroupStart`/`End`; torch's `batch_isend_irecv`).
  Separate send then receive calls in the same order on both sides do not pair: a channel has one position axis for
  both directions. The library does not detect it, nor unequal counts or different collectives on two ranks: no landed
  payload is checked, so these are the caller's error.
- Communicators know nothing of one another, and no two hold one stream: a link has four, so four communicators use
  a node pair at once, and a fifth is refused (EBUSY) until one of them is destroyed.

## Failures

The library keeps no clock that fails a group. A host-path group's waits end at their landings, at a landed word's
`~0` (its bridge's, where the link is lost or the bridge stops), or at a revoke; a GPU wait also ends at its bound
([Bounded waits](#bounded-waits)). A group declares its units to its bridge as it is issued, with no deadline while its
member lives. A member that fails, is revoked or ends has the units it declared and did not supply sent at once as
duds: all ones, placed where a send must happen and its producer is gone (design/transport.md section 10). Its peers
land them as bytes like any other. A dud is no signal and fails nothing at the consumer; the failure is the
producer's own (design/rubrics.md M4: "nothing else changes"; M10). A peer that has not yet issued its group is waited
for, a host-path group's wait on it until a landing or a revoke. The failure is returned:

- by the group, or
- by `ncclCommGetAsyncError` for work already enqueued, and for a communicator whose bridge stopped or one of whose
  streams is no longer its own (its link lost and paired again opens every stream new: design/transport.md section 14),
  before any group runs on it,

and `ncclGetLastError` names what failed: the node of the channel where a landing failed. A link lost under a group
fails the group (nothing resumes it), and a member whose group failed is replaced at its peers by duds, which they
land and complete with: the failure stays with the member that failed. A communicator whose group failed fails its later groups: the caller makes it again (`ncclCommDestroy`, then an
init) on the map as it is then (the operator, 2026-10-14: "it's okay for a function call ... to fail instead of
returning if the link topo changed mid function execution"). A communicator keeps the map its init made, so a change
that leaves its links working (a link lost elsewhere, a node that joins) reaches the next communicator
(metal-microbench makes each call's communicators at the call's start). A rank that departed is left out with
`ncclCommShrink` (to one rank: [Communicators](#communicators)), and a joining one taken in with `ncclCommGetUniqueId`
and `ncclCommGrow`, an init at the id comm's root broadcasts on comm, holding streams of its own as every init does. `ncclCommRevoke` is NCCL's, from any thread and under the communicator's own lock
alone: its operations in flight are revoked and its resources kept (its streams stay held, and the units its groups
declared and supplied nothing go at once as duds, so its peers' groups fail rather than wait); an init
still running (a non-blocking one's) ends as a value, its host-path groups return at their next look and its GPU waits
end at their next look at its stop word, and every later group on it is refused (`ncclInvalidUsage`, "revoked") until
`ncclCommDestroy` gives its streams back.

A communicator takes no bytes of its node's window: its rings are its streams', the bridge's (`mesh.h`
`struct mesh_stream`). The process holds one of its bridge's client slots (`mesh.h` `MESH_CLIENTS`) whatever its
communicators; each communicator of two ranks or more holds a stream toward each member's node it is joined to from
its init to its destruction, one communicator a stream (each channel's head credit sizes the stream's ring for that
channel alone), of the four a link has.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | library (ncclGetUniqueId) | the bootstrap root's `host:port` |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against (default its parent; `bin/mesh-user.sh` passes the checkout's) |

## NumPy

`rdma/mesh_mpi.py` spells the host path as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import:

```python
from mesh_mpi import Comm, SUM, MAX     # sys.path: ~/mesh/rdma
comm = Comm()                           # RANK of WORLD_SIZE, NCCL's bootstrap at MASTER_ADDR:MASTER_PORT
comm.Allreduce(local, total, op=SUM)    # Allgather, Bcast, Reduce, Reduce_scatter_block, Alltoall, Send, Recv,
best = comm.allreduce(x.max(), op=MAX)  # Barrier; IN_PLACE as a send buffer
comm.Allgatherv(part, [whole, counts])  # a count a rank: ncclMeshAllGatherV
comm.Reduce_scatter(full, mine, counts) # ncclMeshReduceScatterV
```

Subcommunicators are `Split(color, key)` and a Cartesian grid: `grid = comm.Create_cart([2, 2])`, `grid.Sub([False,
True])` its row, `grid.Shift(0, 1)` the neighbours, as mpi4py spells them.

metal-microbench `tools/mesh/numpy_power.py` is a ported power iteration, and `tools/mesh/numpy_summa.py` SUMMA on a
2-D grid (row and column broadcasts).

## On one host

`tools/nccl_demo.py plans` (metal-microbench): every compiled program on data, over rings and stars of n = 3..8.

## Not implemented

ncclCommInitRankScalable, ncclCommSuspend/Resume,
ncclCommMemStats, the window and signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd,
ncclSetEncryption, the ncclParam functions (`nccl.h`).
