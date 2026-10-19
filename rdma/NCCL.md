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
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens, on the node's fabric address where it holds one), where
  each rank's node meets the others'. The root listens on that host alone where it is an IPv6 address.
- `ncclMeshCommInitRank(comm, nranks, id, rank, paths, programs, count, small, name, config)`: the same init and
  bootstrap (`id` the root's `host:port`, which the caller hands every rank; each rank's node is its bridge's), with
  the collectives as operands; the map is the bridges' (below).
  - `paths`: the routing tables every collective of the structural table (below) is compiled along, each member's
    trees, checked against the map each session reads. NULL means the map's shortest-path trees.
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the structural table
    (below).
  - `small`: the bytes of the small class's slot (0: the bridges' block).
  - `name`: the communicator's, which its sessions pair under on every member and its members' digest holds (NULL:
    the id).
  - `config`: NULL, or `blocking` not 0, is a blocking init; `blocking` 0 is NCCL's non-blocking init (below).
  - Every rank passes the same values. `ncclCommInitRank` is this call with no paths, programs, small, name or config,
    and `ncclCommInitRankConfig` the same with its config.
- Both inits lease the communicator's lane pair toward each peer at the bootstrap's two gathers, the second bounded by
  `MESH_PLATFORM_NS` a send or receive; an init fails as a value where a link has no pair free at both ends.
- A non-blocking init sets `comm` at once, runs on a thread of its own and returns `ncclInProgress`;
  `ncclCommGetAsyncError` gives `ncclInProgress` until it ends, then its result. `ncclCommRevoke`, `ncclCommAbort` and
  `ncclCommDestroy` end it as a value with its claims released: the revoke shuts its bootstrap's sockets down, and its
  connect and its root's accept read the revoke at each 100 ms look. `ncclCommAbort` or `ncclCommDestroy` then frees the
  communicator, failed or not; no call but these four takes it before its init has ended.
- A rank's connect and the root's accept hold nothing of the node's bridge: a rank attaches once its bootstrap has
  connected, or accepted every rank, and its report (the lane pairs free at its bridge) follows that attach. A blocking
  init's root waits in its accept.
- A rank's connect to its root gives up `MESH_PLATFORM_NS` after its first try: an init toward a root that does not
  listen fails as a value (`ncclRemoteError`, ETIMEDOUT), never retried for good.
- `ncclCommSplit` takes the ranks of a color; `ncclCommShrink` (NCCL's signature) leaves out the ranks named, those
  that departed, so a loop goes on with the ranks that remain and no new bootstrap.
- Making a communicator, a lone rank's too, attaches the process's client of its node's bridge ([Memory](#memory)),
  so the node's bridge runs; it fails where `ncclMemAlloc` would.

Each session takes its node's bridge's map when it opens (`mesh-plan.h` `mesh_map` over the communicator's nodes):
every node the bridges' records hold, the members first and the others the map's relays, every cable two bridges pair
(a link each, at the slowest of its ports' bandwidths: the bridge stripes a program between two nodes over the cables
its communicator leased, position t on cable t mod n, so the pair carries it at n times the slowest cable), and the
routing table (`mesh_routes`: each node's in-tree of shortest routes, by fewest links and then over the faster hops, a
parent among equals the least hash of its mark, its parent's and the root's, so routes of equal length spread rather
than pile onto the lowest-numbered node, and a route moves only where the shortest routes between its ends change,
never for a link lost elsewhere or a node joining; a route between two nodes the higher-marked one's in-tree path, one
path both ways). Two ranks are linked where their route crosses no other node; a session binds a channel to every other
member, which needs a cable between the two ([Routes between unlinked ranks](#routes-between-unlinked-ranks)). A later
communicator reads the map again, so a recabling, a link that paired again or a node gone reaches the next session
with no edit and no restart. The members' map of a record is the same function's (`mesh_map_members`, Python `mesh.members`;
`mesh.current().map` is the members' map itself).

The bridges' records are a map whatever their cabling: while a cable is out, a node that still has a record may have
no link, and the whole map (`mesh.current()` with no members, `mesh_map` over none, mesh-stat's `map`) holds it as it
is. `mesh_link_map_check` refuses only what is no map; whether a map joins every two of its nodes is
`mesh_link_map_joined` (Python `mesh.joined`, which names the first two it leaves apart). The trees, the collectives and
a session run only on a map that joins its members: `mesh.current(members)` and `mesh.members` raise ValueError naming
two members no route joins, and a session on such a map fails as a value naming them (`ncclRemoteError`, "has no
route").

Before any group runs, the members agree: each publishes its digest of what its programs are made from (the route
between each two members and the bandwidth of the links it crosses, the planner's version `MESH_PLAN_VERSION`, the
session's wire, the paths and trees, the programs) and of the communicator's name on every channel, at the session's
first position, and compares its peers'. Members that read the map on either side of a change fail the session as a
value naming the disagreement (`ncclRemoteError`, "the members' maps differ") and open the next on the map then; no
group runs on two maps, and two builds of the library that make other trees of one map disagree there too, not as a
send/receive mismatch later, as do ranks of two communicators that one bootstrap port paired.

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR), the communicator's table of them. The caller passes its own table (`ncclMeshCommInitRank`'s `programs`;
`mesh.compile` in Python, `chunks` its cut, `whole` the single-phase all-reduce); with none, the communicator compiles
the structural table at each session, along the caller's `paths` or else the map's shortest-path trees (each member's
in-tree of the bridges' routes, its parents among equals spread over the links):

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
waits are the session's in-stream waits on the bridge's completion words, and `received` and `arrived` are host memory
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
planner's `reduce` form for partials (metal-microbench docs/placement.md Crossings), until now only the engine's own
kernels and torch-mesh's single-GPU merge. `ncclMeshAllReducePartials(send, recv, rows, dim, comm, stream)` and
`ncclMeshReducePartials(send, recv, rows, dim, root, comm, stream)` take the engine's own layout,
`[m: rows][l: rows][O: rows x dim]`: the piece laid as rows into a scratch, merged there with the operator and laid
back, so not inside a group (until their caller lays rows; a member off a reduce's root gets its own partial back).
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
programs refuses them. A form that cuts the operand (two-phase, chunked) is not fusable (`ncclMeshFusable` answers
0). Reproduction target: simulated 0.51x the default's time at 1 GiB on six-node maps and 0.58x at 16 MiB, the ring4
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

- **Host path.** A NULL stream: host pointers, copied through the session's rings, and each call is synchronous. It
  works for any array on any node whose bridge runs (NumPy, CPU tensors). It is functional, not the measured path.
- **Metal path.** An `ncclMeshStream`: the group is encoded into a Metal command buffer, and every buffer is an
  `ncclMeshBuffer` (an `MTLBuffer` and a byte offset). The kernels copy each operand into the session's rings in the
  bridge's window and land what arrives from them, with no staging on the host; a send buffer that is a lease's Metal
  buffer is read in place ([Memory](#memory)).
  - torch-mesh runs MPS tensors here.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`).
  - **Metal 4.** A stream whose `argumentTable` is set encodes into the caller's `MTL4ComputeCommandEncoder`
    (`commandEncoder`) between the commands before and after it, every binding through that argument table (at least
    six buffer bindings; each dispatch's arguments a record of the library's ring, never `setBytes`), on pipelines of
    the caller's device, the library's buffers added to the caller's `residencySet`; nothing is committed or
    signalled. `ncclMeshEncodeMetal4(comm, op, send, recv, count, type, encoder, table, residency, polls)` is the
    all-reduce spelled so; every collective and group takes the stream. Its waits are bounded by the stream's `polls`
    as the classic form's are, 0, or more than `NCCL_MESH_POLLS`, being `NCCL_MESH_POLLS` ([Bounded
    waits](#bounded-waits)). Without a workspace its scratch is the communicator's, so such a group completes in its
    encoder (no `issue`). A recorded program's replay calls it at each crossing it marks
    (metal-microbench docs/kernels.md#any-program).
  - A wait is a kernel polling its position's completion word, a word of the session's that its canceller ends (a
    lost link, or one silent for `MESH_PLATFORM_NS`, which its bridge concludes; a revoke: [Failures](#failures)), and
    the stream's `polls` bounds it ([Bounded waits](#bounded-waits)).

### Bounded waits

Every GPU wait is bounded (design/recovery.md#five-seconds; the operator, 2026-10-18: "you cannot be using a low level
kernel like a dma or networking send/rcv channel for more than 5 seconds without something happening/concluding").
`ncclMeshStream.polls` is the most loads of a completion word one wait of the group makes (`rdma/mesh.metal`
`mesh_remote_wait_bounded`): the caller's bound for the group's crossings, carried per crossing. 0, or more than
`NCCL_MESH_POLLS` (nccl.h), is `NCCL_MESH_POLLS`, in either form: torch-mesh's waits and the engine's encoded crossings
pass 0. A wait that reaches its bound records its position, its channel and the bound in the session's violation
words, every later dispatch of the session does nothing, and the group fails as a value after its command buffer
(`ncclCommGetAsyncError`, or the communicator's next group: "landed nothing within N polls"), every wait of the session
ended (its words `~0`). `ncclMeshComplete` bounds the completion's waits by the stream it is given.

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
- `ncclMeshMemBuffer(ptr, &buffer, &offset)` is the Metal buffer over the whole lease holding `ptr` (any byte of it)
  and ptr's offset in it, made once and released by `ncclMemFree`: the buffer a tensor allocator makes MPS tensors over
  (torch-mesh's window tensors).
- A Metal-path collective whose send buffer is a lease's Metal buffer reads it in place where its program allows: a
  SEND of a range no step of the rank wrote reads the send buffer, the first combine of such a range writes send op
  piece (`combine3_*`), the result's ranges no step writes are copied from it after the publications, and no staging
  or scratch copy of the operand is made (a scratch buffer no step writes is never made). The two-rank all-reduce out of
  place, for one, makes no copy of its operand where it made two (the stage into the result and the scratch its SEND
  read). Every such read is in the group's own command buffer or encoder (an issued group stages its operand as
  before), and the bridges read only the session's rings, so once that command buffer completes nothing reads the
  lease and `ncclMemFree` may return its pages. The bytes on the wire still pass through the session's rings (the
  bridges' pre-posted flows, bound when the session opened); in-place, an operand's own copies go, not the ring's.
  Reproduction target (metal-microbench docs/measurement.md 2026-09-30 part A, 95967e6..42a16ed): per 16.8 MB
  all-reduce, the library's copies of the operand 16.78 MB to 0; tp 271.5-272.2 ms a step.
- A host-path group waits for its own communicator's GPU groups alone and fails while one of them is issued and not
  complete; while no landing of its own has come it parks (its session's watcher wakes it within a millisecond of one)
  and holds no lock another communicator's groups take, so a rank may wait in `ncclRecv` on a peer for as long as the
  peer takes while its other communicators run (metal-microbench docs/elastic.md#call-delivery).
- The host path copies every operand through the session's rings, a lease or not; `ncclMeshMemUse(ptr, bytes, event,
  value, write)` declares the caller's GPU work on a lease (an `MTLSharedEvent` and the value it reaches), which a
  host-path group reading or writing that lease waits for (until the GPU's work reaches the value). The window is where a program
  that binds its own section (`mesh-call.h` `mesh_transfer_bind`) crosses without the rings; `ncclCommRegister`
  refuses a buffer outside it.

## Routes between unlinked ranks

The bridges forward nothing (design/bridge-and-ipc.md#routes). A communicator's init leases its lane pair toward each
peer only on a cable of its bridge, and its session binds a channel to every other member, so a member no cable of this
node reaches fails the session at its first group that crosses a link (`ncclInvalidUsage`, ENOLCK), and a program toward
it is refused by the bridge (EHOSTUNREACH). Nodes a route crosses between members a cable does not join take part as
members of the communicator that contribute nothing (`mesh-plan.h` `mesh_compile`'s `contributors`): such a member
receives on the rows it sends, so each landing is handed to its SEND as it lands, with no copy
(design/prepared-machine.md M08).

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

The library keeps no clock that fails a session. A session's waits end at their landings or at its words' `~0`: its
bridge's where a link is lost, where a link is silent for `mesh.h` `MESH_PLATFORM_NS` or a SEND is held that long with no
receive for it (design/recovery.md#five-seconds), and where the bridge stops; the progress thread's where the bridge is
gone or refused or failed the session's program toward a peer (that destination's status); and a revoke's. A GPU wait
also ends at its bound ([Bounded waits](#bounded-waits)). A peer that is alive and has not yet published is waited for,
a host-path group's wait on it until it publishes or the caller revokes the communicator. The failure is returned:

- by the group, or
- by `ncclCommGetAsyncError` for work already enqueued,

and `ncclGetLastError` names the node where it failed: the destination's, or this node's where its own bridge refused
the program. A link lost under a group fails the group (its program toward that node ends; nothing resumes it) on every
member. A session that fails closes with its communicator's program slot and lanes, so the communicator's next group
fails too: the caller makes it again (`ncclCommDestroy`, then an init), and its session opens on the map as it is then
(the operator, 2026-10-14: "it's okay for a function call ... to fail instead of returning if the link topo changed mid
function execution"). A change of the map never closes a session: it runs on the map its members agreed on until a
link or node its routes cross is lost, so no member reopens at a group of its own and fails a group a peer had in
flight on the old session, and a change that leaves its routes working (a link that flaps elsewhere, a node that joins,
a better route that appears) reaches the next communicator (metal-microbench makes each call's communicators at the
call's start). A rank that
departed is left out with `ncclCommShrink`, a joining one taken in with `ncclCommGetUniqueId` and `ncclCommGrow`, and
`ncclCommRevoke` revokes a communicator for good, from any thread and under the communicator's own lock alone: an init
still running (a non-blocking one's) ends as a value, its session's waits end and its groups in flight fail, a session opening ends (a member waiting in its session's start for
a peer that never comes, a member that failed to load, is ended at once) and none opens after it, and every later
group on it fails at once (`ncclRemoteError`, "revoked") until `ncclCommDestroy` closes its session.

A session's footprint in its node's window is `ncclMeshSessionBytes(peers, wire, &bytes)` (Python
`mesh.session_bytes(peers)`): toward each other member its four rings (16 slots each way of each class) and its transfer
lists, then its SEND cells, completion words, cancellation ranges and violation page. A session binds every other
member of its communicator (a point-to-point call may name any, and its members' agreement meets on those channels
whatever map each read); a split or shrink is a communicator of its own, with its own session. Each communicator of
two ranks or more holds one of its bridge's client slots (`mesh.h` `MESH_CLIENTS`) and a lane pair of each link toward
each peer (four communicators a link at once) from its init to its destruction.

## Environment

| variable | read by | meaning |
|---|---|---|
| `NCCL_COMM_ID` | library (ncclGetUniqueId) | the bootstrap root's `host:port` |
| `NCCL_BUFFSIZE` | library | bytes of a large ring's slots together (default 4 MiB, `mesh-plan.h` `MESH_BUFFSIZE`), read when a session opens; `mesh_wire_of` makes the wire of it |
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
