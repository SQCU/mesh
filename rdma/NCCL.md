# libnccl-mesh

NCCL's C API over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each. Code
written against NCCL's `nccl.h` compiles against `rdma/nccl.h`: NVIDIA/nccl 12df1a1 `src/nccl.h.in` (NCCL 2.32.3) with
its CUDA includes removed, its version filled in, and only the functions `libnccl-mesh.dylib` exports declared, each
also as its `pnccl` profiling name, then the mesh's own calls (`ncclMesh*`). The sources carry no prose; this page is
their contract. PyTorch programs use it through [torch-mesh](torch-mesh/README.md). NumPy programs use it through
`mesh_mpi.py` (below). Uneven nodes and links: [design/heterogeneity.md](../design/heterogeneity.md).

## Build and run

The node's user keeps the library in the user's mesh checkout (`~/mesh`): `bin/mesh-user.sh` builds the bridge, the
libraries and the Python bindings there, keeps the user's torch environment with torch-mesh and the checkout's library on
its path (`~/.venv-mesh-uv`), and runs the bridge as the user's launchd agent (`io.mesh.bridge`, design/transport.md) with
the node's bridge configuration, its one operand (`bin/mesh-user.sh CONFIGURATION`, etc/bridge/README.md; there is no
default); no part of it needs root. A program links or imports the checkout's library:

```
cc app.c -I ~/mesh/rdma -L ~/mesh/rdma -lnccl-mesh -Wl,-rpath,$HOME/mesh/rdma
```

A rank is a process on a node, a client of that node's bridge; it starts no bridge and names no device, address, port
or region. Ranks are launched as torchrun launches them (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

### The library's kernels

libnccl-mesh's own Metal kernels (`nccl-mesh-kernels.h`, after `mesh.metal`'s primitives) are compiled when the
library is built, never in a process that runs it. They are named by the kernels' SHA-256: the hash of their text and
of the line that compiles it. That hash is compiled into `libnccl-mesh.dylib`, and `make -C rdma install` puts two
files beside the dylib:

- `libnccl-mesh.<SHA-256>.metallib`: the kernels as AIR, which every Apple GPU's driver takes. It is the committed
  `vendor/libnccl-mesh.metallib` (vendor/PROVENANCE.md).
- `libnccl-mesh.<SHA-256>.binarchive`: this node's binary archive, the GPU code of every kernel's pipeline for this
  node's GPU. The build tool (`nccl-mesh-kernels.m`) makes it from the metallib at every install, in about 0.9 s, so
  an archive is never copied between nodes.

A process loads both at its first Metal collective and looks up every pipeline in the archive with
`MTLPipelineOptionFailOnBinaryArchiveMiss`, so it compiles no kernel. On the M5 this takes about 3 ms (the library
0.2 ms, the archive 1 ms, 72 pipelines 1.7 ms), where compiling took about 100 ms, and about 250 ms after any change to
the text. Where a file is missing, or the archive lacks a kernel (the system's GPU compiler changed: an OS update since
the install), the collective fails as a value. Its message starts with `make -C rdma install`, which makes both anew,
and names the file; after an OS update that changed the compiler, the archive lacks its kernels until that install
runs. A failed load keeps nothing, so the same process's next collective loads the files the install made.

`install` installs every other image first, then the kernels' files and `libnccl-mesh.dylib` (`install-nccl`). So a
failure of the kernels leaves the bridge's images installed and the node's earlier libnccl-mesh in place. Earlier
kernels' files stay, for a process that mapped an earlier library but has not loaded its kernels yet; `make -C rdma
clean-kernels` removes them, and no other target runs it.

Only a machine with the Metal toolchain compiles the AIR: an Xcode with its Metal toolchain component, named by
`DEVELOPER_DIR` (on this mesh, the M5 alone). `make -C rdma kernels` (which `install` runs) checks the committed record,
`vendor/libnccl-mesh.sha256`: the kernels' SHA-256 the AIR was compiled from, then the AIR's own. Where both match, the
committed AIR is used. Where either does not, the kernels are compiled into `vendor/`, and the two files are committed
with the change to the kernels (`nccl-mesh-kernels.h`, `mesh.metal`, or the compile line). On a machine without the
toolchain that compile fails, and `make` names the build to run: it never installs AIR of other kernels.

`ncclMeshMetalPrelude()` hands the device header's text to a caller that compiles its own kernels after it
(metal-microbench, torch-mesh's `_stream`), which compiles them in its own process: `rdma/mesh.metal` as the library
was built with it, then `nccl-mesh-kernels.h`'s `STEPS` ([Persistent exchanges](#persistent-exchanges)). It comes from
the library the process loaded, so a caller never reads a file that may be another build's. Only the library's own
kernels moved to build time.

The build tool `nccl-mesh-kernels TEXT` writes the kernels' text (`mesh.metal`'s primitives, then `STEPS`, `KERNELS`,
`TRACKS` and `TRACK_KERNEL`), which the Makefile compiles to the metallib; `nccl-mesh-kernels METALLIB ARCHIVE` makes
every kind's pipeline (`KINDS_OF`, the track kernel's among them) from the metallib into one serialized binary archive.
Each fails naming what failed, exit status 1. `KINDS_OF` is the one list of kernel kinds: the host's numbering of its
steps, a device's pipelines' index and the track kernel's switch are all made from it by the preprocessor, so neither
names a kind the other lacks. Each operator's arithmetic is written once in the kernels (`op_int`, `op_float`, `op_f8`,
`merge_row`), and is the host's (`combine_into`, `merge_rows`, `premultiply`, `truncdiv`): the small floating types in
float32 rounded to nearest even, float8 decoded and encoded by `mesh.metal`'s `f8_to` and `f8_from` (OCP 8-bit floating
point, 2023: E4M3 with no infinity and NaN S.1111.111, E5M2 IEEE-like, saturating to NaN and infinity), integers
wrapping. A landing's combine loads both its operand and the landed bytes system-coherent: the operand may itself have
landed in place (a COPY earlier in the group), and the landed bytes are the NIC's.

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's, with NCCL's bootstrap: the id is the root's `host:port`
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens, on the node's fabric address where it holds one), where
  each rank's node meets the others'. The root listens on that host alone where it is an IPv6 address.
- `ncclMeshCommInitRank(comm, nranks, id, rank, programs, count, small, name, config)`: the same init and
  bootstrap (`id` the root's `host:port`, which the caller hands every rank; each rank's node is its bridge's), with
  the collectives as operands; the map is its members' (below).
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means each plan compiles its own
    along the solver's trees (below).
  - `small`: not read.
  - `name`: the communicator's, which its failures and its splits' names carry (NULL: the id).
  - `config`: NULL, or `blocking` not 0, is a blocking init; `blocking` 0 is NCCL's non-blocking init (below).
  - The library compares none of these values across ranks. `ncclCommInitRank` is this call with no programs, small, name or config,
    and `ncclCommInitRankConfig` the same with its config.
- Both inits hold the communicator's streams at the bootstrap's two gathers. The first carries each rank's node and the
  cables its bridge reaches them by (one reach entry a cable: its node, its `cable` and its bandwidth); each rank then holds
  one stream on every cable that joins it to another member, the least `wire` free at both ends on that cable, so both
  ends take the same stream, as it holds rows (the stream's owner word taken from 0 by one compare-and-swap). The second carries each rank's
  holding's status. A communicator names nodes and its own pages alone. Two ranks on one node have no cable between
  them, so the map joins them by no edge and a call that crosses between them fails as a value where it is planned (no
  route, or no channel). An init fails as a value where no stream toward a member's node is free at both ends (EBUSY: a link has up to four, each
  held by one communicator, and one whose last holding its bridge is still ending is not free) or the link is gone
  (ENOLINK)
  (`ncclSystemError`), or where its bootstrap's phase passes its bound (`ncclRemoteError`, ETIMEDOUT); what it held is
  let go, and the next init reads the streams as they then stand.
- A connection to the root whose report fails (its peer closed it), or names no rank still to meet (a rank out of
  range, or one already met), ends the init as a value (`ncclRemoteError`, EPROTO); one that sends nothing holds the
  root's read until the phase ends.
- Every wait of a bootstrap, blocking init or not, is a 100 ms slice after which the revoke and the phase's end are read
  again (design/rubrics.md M3, C5). A phase ends `MESH_PLATFORM_NS` after the init began where no rank was met, and
  `MESH_PLATFORM_NS` after its first rank met (the root's first accept, a non-root's connect) for the rest of it; the
  second gather has its own `MESH_PLATFORM_NS`. A root that no rank meets therefore fails as a value, for its caller to
  run again (C6); nothing in the library or its callers retries it.
- While the process holds a lease of a bridge that stopped, its next lease (`ncclRemoteError`) and communicator
  (`ncclInvalidUsage`: the process's own leases, which only it can free) are refused naming it,
  until it has freed every such lease: a process never holds leases of two regions. A node's region is most of its
  memory, and GPU work on a stopped region's leases beside the new region's registration wired the Mini's whole memory
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
  exchange among the rest, so it holds no stream: a shrink to one rank is a lone rank's communicator, and one to two
  ranks or more is made with no channel and no map, its first call that crosses a link failing as a value (no map of
  its members, or no link toward the peer). A rank among those it excludes gets `NULL`, as `NCCL_SPLIT_NOCOLOR`'s.
- Inside a group, `ncclCommSplit` and `ncclCommGrow` run their own collectives at once, the group's calls set aside
  until they return; `ncclCommShrink` makes none. An existing member's `rank` to `ncclCommGrow` is not read: it keeps
  its rank.
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
communicator has one channel toward each member it is joined to, and none toward another ([Routes between
unlinked ranks](#routes-between-unlinked-ranks)). A later communicator reads the reports again, so a node that joins,
a node gone or a link lost (down until its ports' next children pair it again, with nobody acting: design/transport.md
section 14) reaches the next communicator with no edit and no restart. libmesh's whole map of every node's record (`mesh_map` over no members, Python `mesh.current()`) joins two
nodes the same way.

The bridges' records are a map whatever reaches what: a node that still has a record may be reached by no bridge,
and the whole map (`mesh.current()` with no members, `mesh_map` over none, mesh-stat's `map`) holds it as it is. The
trees and the collectives run only on a map that joins their members: a communicator's collective on a map that does not fails as a
value naming two members no path between members joins (`ncclInvalidUsage`).

Every member makes the map from the same gathered reports, so the members' maps are one. Programs, forms and the solver
are each rank's caller's, and nothing compares them at run time.

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR). The caller passes its own table (`ncclMeshCommInitRank`'s `programs`; `mesh_compile`, `chunks` its cut,
`whole` the single-phase all-reduce); with none, a plan compiles each of its collectives when it is made, along the
trees the solver gives the plan's traffic ([Routing](#routing)):

- the all-reduce single-phase (each member's sum up its own trees, the map's diameter in dependent crossings; at two
  members, and wherever every two members are linked, the direct exchange) or two-phase;
- the reduce-scatter and the all-gather, segment s up and down s's trees;
- each root's reduce and broadcast, up and down the root's trees.

Each is uncut: a step's piece is one entry of its plan's table, any size, which the bridge alone cuts into SENDs. A
program's own cuts (a two-phase all-reduce's segments, a tree's shares, chunks) lie on the region's pages
(`mesh_program_steps_cut`), so every piece starts on a page wherever its buffer does ([Memory](#memory)). A call runs
the caller's first program of its collective (and root) whose size bound its operand is under, else its plan's. Nothing
about a program is decided where a plan restarts.

**All-to-all with host counts.** `ncclMeshAlltoAllV(send, counts, recv, type, comm, stream)` is MPI_Alltoallv with
its counts on the host, every member passing the same matrix: `counts[a x nranks + b]` elements go from rank a to rank
b, packed in rank order in a's send buffer and in b's receive buffer (a rank's own block, `counts[r x nranks + r]`, is
copied). MPI_Alltoallv takes each rank's own `sendcounts` and `recvcounts`, a row and a column of this matrix; here
every member passes the whole n x n matrix, so that every member plans every block (design/rubrics.md C7, at a C1
cost). A program ported from MPI holds its own row and column; an SPMD program holds the matrix; torch-mesh, which is
given torch's per-rank splits, fails such a call as a value and takes the matrix in `torch_mesh.all_to_all`, from the
compiled program (its configuration's shapes and shares, a program's maximum extents, how much of a block a step
fills being data in its pages). Each segment starts on a page of its buffer (else the call fails as a value). NCCL's `ncclAlltoAll` is it with
every count `count`, and `ncclGather` and `ncclScatter` its rooted forms (every count 0 but toward or from the root),
where they were each rank's sends and receives in a group (NCCL's own lowering, `classifyCollToP2pTasks`). Every member
knows every block of the four, so they run along the communicator's routing table, each block cut among its pair's
routes, through the members between two that no link joins ([Routes between unlinked
ranks](#routes-between-unlinked-ranks)). torch-mesh passes the matrix torch's call gives (an all-to-all of equal splits,
a gather, a scatter) or the caller's (`torch_mesh.all_to_all`).

**The partials merge.** `ncclMeshRedOpCreatePartials(&op, dim, ncclFloat32, comm)` makes a reduction operator, as
`ncclRedOpCreatePreMulSum` makes one (freed by `ncclRedOpDestroy`): split-KV attention partials merged by their
log-sum-exp [Dao et al. 2023, Flash-Decoding; FlashInfer's state merge]. The operand is rows of `2 + dim` float32,
each row `[m, l, O: dim]` (a row's maximum score, its sum of exponentials, its unnormalized output); two rows merge to
the greater m, each one's l and O rescaled by e^(its m - that) and summed, a row with no positions (l not above 0)
counting for none. The result is the merged partial, its O / l the attention over every member's positions. Every
reducing collective takes it (`ncclAllReduce`, `ncclReduce`, `ncclReduceScatter`, `ncclMeshReduceScatterV`, counts in
whole rows), inside a group or not: one group, the communicator's program of that collective with a row for an
element, each row merged up its tree where a sum adds an element, every piece whole rows of any width. This is the
planner's `partials` lowering of a split-KV attention (metal-microbench docs/placement.md Crossings): the engine
(metal-microbench decode_attention_partial, mesh_rank.m issue) and torch-mesh (ProcessGroupMesh.merge_partials) each
lay their partials as these rows and call `ncclAllReduce`, or `ncclReduce` to the one member reading them, with it.
Reproduction target: never measured; the planner's pricing of the three-group form it replaces at 3, 4 and 6 members
of a full mesh was 331, 331 and 226 us against the gather's 534 a chunk's layer (metal-microbench
output_data/planner-20261015/pass2 engine2-after.json).

**Forms as configuration.** `ncclMeshCommForms(comm, forms, count)` is the communicator's configuration operand: a
list of `ncclMeshForm` entries, each a collective (and root, -1 every root) below a size bound, so a plan compiles a
call by the first form that fits it and else by the default (the all-reduce single-phase, every collective uncut):

| field | form |
|---|---|
| `whole` | an all-reduce's: 1 single-phase, 0 two-phase (the reduce-scatter and then the all-gather along the same trees) |
| `chunks` | each tree's part cut into chunks, pipelined down and up the trees (1: uncut) |
| `below` | the bytes a call is under to take it (0: any size): the size classes |

These are the forms mesh cc8a6f8e^'s default table chose among by a price (its `sets[1]`, single-phase or not, 1 to 64
chunks, by size class); the price is gone and the caller's configuration document names the form (the operator,
2026-10-17: "if there is an overriding operand that stops our code from running beacuse someone wanted ot put in a shim
1 month ago and tried to justify it by calling the hardocde a 'model', we need to rip out all of the rleated tech debt,
includnig notatinoal debt"). A rank's forms are taken at the communicator's next plan, and nothing compares them across
ranks; splits and shrinks keep them. A form's cuts lie on pages. Reproduction target: simulated 0.51x the default's time at 1 GiB on six-node maps and 0.58x at 16 MiB, the ring4
16 MiB all-reduce 2.18 to 1.71 ms, a six-node full mesh's 1 GiB all-gather 158.1 to 20.5 ms (metal-microbench
output_data/library-20261015/pass2/sims), and the pair's 256 and 512 KiB calls 57.4 to 31.2-36.2 us
(pair_programs.json): never measured on more than two nodes.

**Uneven segments.** `ncclMeshAllGatherV(send, recv, counts, type, comm, stream)` and
`ncclMeshReduceScatterV(send, recv, counts, type, op, comm, stream)` are MPI_Allgatherv and MPI_Reduce_scatter: rank
r's segment is `counts[r]` elements (any of them 0), packed in rank order. Each is one program, the communicator's
all-gather or reduce-scatter, its operand cut into those segments (`mesh-plan.h` `mesh_program_steps_cut`: a part
within segment s takes its tree's share of `counts[s]` in pages, the solver's; each
segment starts on a page of its buffer, else the call fails as a value), so a capacity-shaped division of the work (metal-microbench
docs/placement.md) is one tree program over unequal segments rather than a rooted broadcast or reduce a rank. MPI
specifies the counts alike on every rank [MPI 4.1 §6.7, §6.10.2], and the library compares none across ranks; equal
counts are NCCL's own calls. Reproduction target (metal-microbench docs/measurement.md
2026-09-30): tp SP=1 at capacity-shaped parts 255-256 ms a step; per call, the SP=1 all-gather 214-229 us and
reduce-scatter 196-212 us; one uneven all-gather against the same segments as N broadcasts.

## Two paths

Both paths move a crossing's bytes the same way: zero copy, through plans the bridge serves (design/transport.md
section 14, "The plans"), in MPI's persistent form: a request made once (`MPI_Send_init`, `MPI_Recv_init`) and started
at each call (`MPI_Start`) [MPI 4.1 sections 3.9 and 6.13].
- **A table a channel, presented once.** A group's plan ([Plans](#plans)), and each slot of a persistent exchange
  ([Persistent exchanges](#persistent-exchanges)), holds in its own pages one table for each channel it crosses
  (`mesh.h` `struct mesh_plan`): its send entries and then its landing entries, each step one entry (`struct
  mesh_entry {row, pages, word, need, done, count}`: a range of page-table rows, each row a page), in the order the
  group's program lays them at both ends (`mesh_group_lay`). The table is presented on the channel's stream once, when
  the plan or slot is made (`mesh_stream_present`), and unlinked once, between calls, when the plan or slot will not
  be started again (`mesh_stream_unlink`, [Plans](#plans)).
- **A call is a stamp.** The communicator numbers its calls on each channel 1, 2, 3 and on, in the order it issues
  them, counting the calls with an entry on the channel (`channel.calls`); both ends number alike, because both lay the
  same calls and a send at one end is a landing at the other. A call's start writes its entries' rows for this call
  (the caller's buffers and the plan's own pages, `place_group`) and each send entry's `need`, then stores the table's
  `start` = k. Nothing else is presented per call, nothing is numbered modulo anything, and nothing waits for room.
- **The stamps.** The bridge sends each page of a send entry once the word the entry names (`word`) reaches its `need`,
  posts each receive into the page its row gives for k, and stamps the entry `done` = k once its pages have completed,
  and the table `sent` and `landed` = k once every entry of each way has; ~0 is the link lost. A wait reads its own
  entry's stamp, not a running count of the channel's pages. The library never learns a lane, a frame or a queue pair.
- **Types.** A page count is a `mesh_pages` (`nccl-mesh-metal.h`), a type of its own, so a byte count, an entry or an
  element count is never added to it, taken from it or compared with it but through a named conversion (`pages_of`,
  the one conversion from bytes). A stamp is a call's number, and `stamp_lost` reads the bridge's `~0`.

- **Host path.** A NULL stream: host pointers into the region (leases), sent from and landed in in place, combined on
  the host. Each call is synchronous and returns once its SENDs are done. It works for any lease on any node whose
  bridge runs (NumPy, CPU region tensors). It is functional, not the measured path.
- **Metal path.** An `ncclMeshStream` (`commandBuffer`, `commandEncoder`, `issue`, `argumentTable`, `residencySet`):
  the group is encoded into a Metal command buffer, and every buffer is an `ncclMeshBuffer` (an `MTLBuffer` and a byte
  offset into it: an MPS tensor's storage and offset), of any type. float64 moves as its bytes; a float64 reduction there
  fails as a value (Apple GPUs have no float64 arithmetic). A publication is one kernel's system fence and one store of
  the table's `ready`: `(k << 32) + n` once the call's first n send entries there are written, send entry i's `need`
  being `(k << 32) + i + 1`. A landing is a kernel's bounded wait on its entry's `done` reaching k, then its combine,
  every threadgroup of a wide landing kernel awaiting on its own. A COPY lands in place, so its consumption is the wait
  alone.
  - torch-mesh runs MPS region tensors here.
  - With `commandEncoder`, the kernels go into that open compute encoder, left open with nothing signalled at the
    group's end; without it the library opens one in `commandBuffer`, ends it, and signals the library's event at the
    program's end, which a host-path group and a destroy await.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`): issued, its operands are made in place and
    every publication that needs no wait is encoded, with no wait for a peer; `issue->ticket` names it (0: it completed
    at once), `issue->published` says whether anything was published (the caller then commits, so the peers' waits
    progress), and the issue pointer stays valid until `ncclGroupEnd` returns. `ncclMeshComplete(ticket, stream,
    &published)` completes it, with every group issued before it, on a later stream of the same queue; its operands are
    not touched until then. `ncclMeshRetired()` is the highest ticket through which every issued group is complete (or
    lost with a failed communicator). A group without issue completes at once, with every group before it.
  - **Landings.** The transport writes landings as the plan presents them. A landing entry names no word, so the
    bridge posts its receives once its call has started, and a peer's bytes may land before the group's kernels reach
    them. The operator, 2026-10-22: "data procesisng code gets to write their own guards in their own source. if
    source says to write continuously into ab uffer, the nic must write continuously into that buffer even if this
    means that dat aprocessing code can race or lose parts of a stream." The pages a group lands in that are the
    library's own are its plan's ([Plans](#plans)), and a plan is started again only once the program that read them
    has ended, so they are never landed in while read. A send entry's word stays: the table's `ready`, the producer's
    own declaration that its pages are written.
  - **Group end.** The NIC reads a SEND's pages until its entry is stamped. So a group's end waits for its SENDs: folded
    into its last wait where it can be, else one wait on each table's `sent` reaching k, after which the NIC reads
    nothing of the send buffer.
  - **Metal 4.** A stream whose `argumentTable` is set encodes into the caller's `MTL4ComputeCommandEncoder`
    (`commandEncoder`) between the commands before and after it (a barrier before the first dispatch and after each),
    every binding through that argument table (at least five buffer bindings; each dispatch's arguments a record in rows
    of a plan's lease, never `setBytes`), on pipelines of the residency set's device, the library's buffers (its
    holdings' counts, its plans' pages) added to the caller's `residencySet` and the set committed where one was added;
    nothing is committed or signalled. Every collective and group takes the stream. A recorded program's replay calls
    it at each crossing it marks (metal-microbench docs/kernels.md#any-program).
  - **Argument records.** A program's argument records (each dispatch's on Metal 4, and the tracks' table on either
    form) are rows of a plan's lease (`struct metal_rows`: its lease's pages, then further leases taken as its records
    need them), the plan of a group the program encodes for. A plan's records are written again at each of its starts
    from the first row (`plan_take`), and the plan is taken again only once idle: every program that encoded while it
    was in flight has run its stamp, and a program stamps the plan whose rows hold its records last
    (`program_end`), after every dispatch that reads them. So nothing accumulates across submissions, and an encode is
    never refused for records the GPU has not yet read.
  - The calls of one group name one command buffer or encoder: the group's dispatches are one program.
  - A group on several communicators is a group of each communicator's calls, in the order the group first names them,
    progressing together: each is issued, and where the stream asks no issue all are completed through the greatest
    ticket on the last one's stream, their channels interleaved as tracks (a stream that asks issue gets that ticket);
    on the host path each runs on a thread of its own.
    Every one is launched whatever another's result; the first failure is returned.
  - **Persistent exchanges** ([below](#persistent-exchanges)) run their crossings in the caller's own kernels: the
    library composes the waits and the publication around the kernel's function and runs no step of its own.
  - A wait polls a stamp and nothing else: each poll one `volatile coherent(system)` load after a system-scope
    fence (`mesh.metal` `mesh_wait_bounded`; without it a running kernel saw the host's store milliseconds late:
    metal-microbench tools/poll_probe.m, 2026-10-21), and one more fence after the observed count before the landed
    bytes are read (`mesh_step_await`), each byte loaded coherent (`mesh_payload`). The bridge sets the stamps `~0`
    where the link is lost; the library's bound ends the wait ([Bounded waits](#bounded-waits)). No kernel halts on a
    failure: each wait ends at its count or at its bound ([Bounded
    waits](#bounded-waits), [Failures](#failures)).
  - Metal orders a compute pass after the passes that write (or, for a write, read) a tracked buffer only where both
    name it. So a publication declares the operands its SEND reads (`metal_uses`, `useResource`) on the classic
    encoder, and a persistent exchange's binding declares its slots and the holdings' counts; otherwise a publish could
    run before its operand's bytes were written. The Metal 4 form orders its passes by its barriers.

### What each program owns

The crossing path's mechanisms, each judged by the contract of the program that owns it (metal-microbench
output_data/transport-buffers-20261021/LIBRARY.md ranks the changes of 2026-10-21):

- **The bridge** (design/transport.md): its devices, queue pairs, lanes, the cut of entries into SENDs, the receives it
  posts, the region's registration. It reads `ready`, and writes `sent` and `landed`.
- **The library**, by NCCL's contract: the caller touches no buffer of a collective in flight and issues the same
  collectives in the same order on every rank; the library delivers correct bytes. So the order of every read and
  write of a group is known from the programs every rank compiled, and the library encodes it in the stream's order.
  It names its own pages and its tables' words, credits nothing (TN3205's credit holds a SEND until its receive is posted),
  publishes with one fence, waits for `sent` only where a write meets bytes a SEND reads, and bounds each wait by a
  count of polls, since a GPU spin with no end halts the node.
- **The data function**: its own buffers and their order.

### One track

A completion whose groups all lie on one channel of one communicator (a two-rank collective, as each crossing of a
two-rank decode is) is plain dispatches in the GPU command stream, and the stream's own order is the dependency
(`nccl-mesh.c` `complete_track`). Whether a group lies on one channel is read from its schedule where the group is
planned (`struct inflight` `lone`), the entries both ends lay alike; nothing at run time chooses the form. A
persistent exchange's start is a composed group in flight: its sends count as published and its landings as
consumed when it is started (the caller's kernels do both, encoded after the start), so a completion passes its
entries (`consumed_past`).

- The next send entries are published as soon as their steps' receives before them are consumed: a run of them is one
  `publish` dispatch, one store of the table's `ready` past the last.
- Each consumption is its own dispatch after that. A combine is one wide `land_*`: every threadgroup's thread 0 waits on
  `landed`, then the threadgroup combines its share. A COPY is one `spin`. A first combine of a call reading its send
  buffer in place is a `spin`, then `combine3_*`.
- The two-rank all-reduce's SEND reads the operand its REDUCE combines into. The combine is encoded after the SEND's
  publication, and its wait also awaits the SEND's entry stamp (`war`), so the SEND's bytes are read before the sum is made
  and no scratch copy is made. At a group's last landing on the channel, with every SEND of the group there already
  published, the wait awaits the table's `sent` too, so the group's end needs no wait of its own.
- No track is recorded, no thread picks and no `tracks` dispatch runs: on one track the stream's order is the one
  serial order every rank builds alike, and a wait on it holds no other channel.

### Per-channel progress

A completion on two or more tracks runs every channel on its own (`nccl-mesh-metal.h` `metal_record`, `metal_tracks`).
This is NCCL's per-peer progress, chosen by the operator on 2026-10-20 in place of one serial order of the whole group.

- **Tracks.** Each channel's remaining entries are a track with two cursors: its publications (send entries) and its
  consumptions (landing entries), each in list order.
  - Nothing credits a publication.
  - A consumption runs once its landing came, and it is combined where it belongs; where it combines into bytes a SEND
    of its channel reads, it runs once that publication has run (`need`), and its wait awaits that SEND's stamp.
  - A step's first event that has a receive before it (`mesh_group_before`) waits on that receive's channel having
    consumed it.
  - Neither cursor waits on the other beyond these.
- **On the GPU.** The tracks are one dispatch, one thread group per track. Thread 0 picks the next item whose
  conditions hold, every thread runs it through the same kernel bodies the single dispatches use, and a consumption
  raises its track's count. A group that can pick nothing for `polls` passes ends there, its later items not run; the
  entries it left stay unpublished, and the channel's later calls stay behind them in the bridge's order.
- **What ranks share.** No rank orders anything for another rank to share. The two ends of each list agree on its
  entries (`mesh_group_lay`) and nothing else, so groups of any calls among any number of ranks complete wherever
  their steps' dependencies do.
- **Not completion.** Issuing a group (`issue`) encodes no wait and is unchanged.
- **Residency.** Thread groups of one dispatch wait on each other only through a step's receive before it. Metal does
  not promise they are all resident; a wait on one that is not reaches its bound.

### Bounded waits

Every GPU wait is bounded (design/transport.md; the operator, 2026-10-18: "you cannot be using a low level
kernel like a dma or networking send/rcv channel for more than 5 seconds without something happening/concluding").
The most loads of a count one wait makes (`rdma/mesh.metal` `mesh_wait_bounded`) is the library's own bound,
`MESH_POLLS` (`nccl-mesh-metal.h`, 4.5 x 10^6), a constant of the library's build: no wait the library composes takes
a bound from data code. A data kernel's own wait on a word its partner writes (the Neural Engine's, metal-microbench
neural_engine.swift) is the prelude's `mesh_wait(word, target)`, `mesh_wait_bounded` at the same bound. A wait that
reaches its bound ends there, and its kernel goes on with the bytes the area holds: nothing is recorded, and no other wait
changes (the operator, 2026-10-22: "the collective code does not detect mend, heal, or prevent any kind of error. it is
not stateful with respect to errors"). Each wait of a dispatch reaches its bound alone: a wide landing's threadgroups
run in waves where the grid passes the GPU's co-residency, and each wave's threadgroups wait on their own. The bound is
a count of polls and nothing else: it exists because a GPU spin with no end halts the node (the Mini's watchdog reset,
2026-10-21), not to guard any buffer. A wait's poll is `mesh_wait_bounded`: the word once it holds at least the target, or as it is
after the bound, each load after a system-scope fence (without it a running kernel sees a store of the NIC's or the
host's late: on the M4 Pro about 3.2 ms in nearly every wait, on the M5 Max 2.3 ms at the 99th percentile, while the
fence costs at most 0.13 us a wait: metal-microbench tools/poll_probe.m, 2026-10-21, unfenced 3198 us median on the Mini
and 0.50 us with a 2308 us p99 on the M5, fenced 0.58 us on both; a poll 0.23-0.26 us either way). A reader orders its later loads after the observation with a
system-scope fence of its own, after the barrier that gives the observation to its threads (`mesh_step_await`).
`mesh_payload` loads a landed value through `volatile coherent(system)` in the widest words that divide it;
`mesh_publish` stores a count after a system-scope fence, so every system-coherent store before it (the bytes a SEND
reads, stored system-coherent: a plain store reaches the NIC only once its command buffer has ended, metal-microbench
docs/measurement.md, 2026-09-29) is visible before the count the bridge acts on.

A bound in polls is the lag at the crossing it covers (the record's wait there, times the margin its plan names) over
the period of one load on the node's GPU. That period, read on the idle Mini (M4 Pro,
2026-10-16; metal-microbench output_data/library-20261015/round3 `poll_period.m`, a word nothing writes, waited on in
both of the library's shapes, one thread and thread 0 of a 1,024-thread group, 256 to 65,536 polls a dispatch):

| GPU | clock | ns a load |
|---|---|---|
| M4 Pro (Mini) | working (after ~4 ms of work; 65,536 polls in 16.4 to 16.8 ms) | 251 to 265 (fit 253 to 255) |
| M4 Pro (Mini) | rising (an idle GPU's first ~4 ms of work) | 1,004 to 1,097 |
| M5 Max | not read (its GPU runs the operator's work) | |

So a bound stands for at least polls x 251 ns on the Mini (the time past which a late peer fails the call) and holds
the GPU at most about polls x 1.1 us (a wait that never lands). `MESH_POLLS` is `MESH_PLATFORM_NS` over the
slowest period read, 4.5 x 10^6 loads: at most 4.95 s of the GPU, and about 1.2 s at the Mini's working clock, past
which a peer that has not yet published fails the call. The M5's period is not read; if it is slower than 1.1 us, the
bound passes 5 s there.

Types: the Metal path moves and combines the integer types, float16, float32, bfloat16 and fp8 (e4m3fn, e5m2: decoded,
combined in float32 and encoded to nearest even as the host does, overflow to NaN or infinity; metal-microbench
docs/kernels.md Float8), and moves float64 (Apple GPUs have no float64 arithmetic, so a float64 reduction there fails as
a value); the host path combines float64 too.
Ops: sum, prod, max, min, avg; premul-sum's scalar read when it is made (a host immediate) or, in device memory, at
each host-path call (a Metal-path call by it fails as a value: the kernels take the scalar when the group is encoded);
an operator reduces the datatype it was made for alone; a communicator makes as many operators as its handles hold.

## Plans

A group's plan is made once and started at each call: MPI-4's persistent collectives (`MPI_Allreduce_init` and its
kind, then `MPI_Start` at each step [MPI 4.1 section 6.13]) behind NCCL's entry points, found again by the calls as
NCCL's user-buffer registration finds a registered buffer (`ncclMemAlloc`). `nccl-mesh.c`
`struct plan`:

- **Made** (`plan_make`) the first time a communicator's group has these calls: each call's kind, root or peer,
  counts (and an uneven call's segments), type and operator, whether it reads its send buffer in place, and whether it
  is on a stream (`call_same`). Its steps are planned (`plan_calls`: a collective's program, cut on the
  communicator's quantum by `mesh_program_steps_cut`; the hosted family's blocks along the routing table,
  `mesh_exchange_steps`) and laid on each channel relative to the
  channel's counts (`schedule_group`).
- **Its pages.** A call's extra pages, f(n) pages for its n: a not-in-place collective's operand (a reduce-scatter's, a
  non-root reduce's, a one-rank group's), each scratch buffer its program names, each combine's landing area, and a
  hosted call's relay area. They
  are rows of one lease the plan takes when it is made (`lease_take`), at offsets fixed then (`call.at`, `steps.at`,
  `message.area`). Peers land into them by scatter, and the reduce reads them in place.
- **Presented** when it is made: each channel's table, in the plan's pages after its operand and landing areas, linked
  on the channel's stream once (`mesh_stream_present`). A presentation the bridge refuses (the stream lost or not the
  communicator's) stamps the table ~0, and every call of the plan returns the refusal as its value while its program
  runs as compiled.
- **Started** (`plan_start`) at each call: its entries rebased onto the channels' positions, its pages and the caller's
  buffers placed as rows (`place_group`), each channel with entries taking its next call number k, its table written
  for k and its `start` stored.
- **Released by use count.** A plan is started again only once it is not in flight and every program that named its
  pages has ended. Its lease holds one 32-bit word past its pages. Each program that touches the plan's communicator
  while the plan is in flight names it (`reader`, one more) and adds one to that word at its end (`program_end`,
  `metal_stamp`: one `stamp` dispatch after the program's other dispatches, on either encoder: the page's stamp, written
  by the GPU, with no completion handler standing in for it), so the word equals `reader` once each of those programs has ended, in whatever order the caller
  commits them and on whatever queues, and once each of its tables' `sent` and `landed` has reached its last call
  (`plan_quiet`: the bridge has done with its pages and its table). Where every plan of these calls is held, a new one is made
  (design/pages-and-functions.md, Freeing 6 and 7: the use count goes down when the reading program's stamp lands, and
  a page is written again only after release). The `stamp` dispatch is a system-scope fence, then the atomic add, so
  the pages' stores are visible before the count the host reads with acquire (design/hardware-and-discipline.md
  section 1); a stamp that cannot be encoded fails its program's call as a value.
- A plan made before `ncclMeshCommForms` or `ncclMeshCommSetSolver` changed the communicator's forms or solver is
  never started again. A plan whose group failed is started again as any other: the failed run retired its positions
  (`run`), and nothing of the failure stays with it. A plan never started again is released between calls (`release`,
  at each group's plan and each exchange's start): once it is idle and its tables' last stamps are in, each table is
  unlinked from its stream (`mesh_stream_unlink`, the link's one-word edit undone), and once a call started on each of
  those streams after the unlink has its `sent` and `landed`, the plan's lease is returned (design/transport.md
  section 14, "The plans": every walk of the bridge's that could hold the table has then ended). A table whose stream
  was not live at its unlink, or whose stamps are ~0, is returned with the communicator's leases (`ncclCommDestroy`,
  after its streams are given back). A lease the region cannot meet fails the call as a value.
- The host path takes plans as the Metal path does; its group has ended when it returns, so its plan is idle at once.

**Page lengths.** A link pairs only two nodes of one page length (`block x pgsz`): the bridge ends a pairing whose
HELLO names another (mesh-link.c, EPROTO). Channels go only over links, so the two ends of every channel lay the same
pages, and the program's own cuts lie on this node's page (`mesh_program_steps_cut`). No length is compared at run
time. `mesh.h` `struct mesh_entry` names its rows and a count of pages; an entry's length in bytes, which would let the
bridge post a run's last unit at its own length, is not a field of it.

## Persistent exchanges

`ncclMeshExchangeInit(parts, lands, type, comm, &request)` is MPI-4's `MPI_Neighbor_allgatherv_init` [MPI 4.1 section
8.8] over the adjacency the caller names (as `MPI_Dist_graph_create_adjacent` names one), every member passing the same
exchange: rank a's part is `parts[a]` elements, and it lands at each rank b with `lands[a x nranks + b]` set. Each member
compiles the whole exchange into its own legs at the init, with no message: each part is a demand of the solver (its
owner the root, the ranks that land it the leaves, its bytes down: [Routing](#routing)), and each of the part's trees
carries a piece of it, the part cut by the trees' shares in pages (`mesh_cut_shares`; a start's count cut the same
way). A member's landing legs are the pieces it receives from its parent in each tree, on the channel toward that
parent, in order of the part's owner and then the tree; its send legs the pieces it sends to its children, in the same
order: from its own part, or from the rows where its landing leg of that piece landed. Both ends of a channel compile
the same legs in the same order, so its entries pair. The library allocates the buffers, as MPI's `MPI_Alloc_mem` and
NCCL's `ncclMemAlloc` do for buffers a library registers: a slot is one lease holding this rank's part, an area for each
part it receives any piece of (each piece at its offset), and the slot's table (`struct mesh_table`, then a `struct
mesh_peer` for each part this rank lands, in rank order, then a `struct mesh_wait` for each landing leg of those parts,
then a `struct mesh_relay` for each piece it sends on from its rows). A request makes its slots as its starts need
them.

- `ncclMeshStart(request, counts, stream, &start)` is `MPI_Start`: the next start, numbered from 1, takes a free slot,
  else the request makes one. A slot is released by use count: every kernel bound to it (`ncclMeshBind`, landing its
  start or publishing it) counts its end in the slot's table (`ended`, the last threadgroup of `mesh_leave`), and the slot
  is free once that count reaches the binds made of it (`bound`) and the bridge has stamped its tables (below),
  whatever kernels the caller binds or none. So no start reuses pages
  another start still holds, and a kernel writes no guard of its own (the operator, 2026-10-22: "programs overwrite
  each other in data processing by default with no extraneous lines of code written"; NCCL-style primitives "use
  explicit allocation per buffer or pageset that they operate against, rather than an attempt to make timed single
  landings"). `counts` (nranks entries, each rank's part, every member passing the same) moves fewer elements than
  declared (partitioned communication's `MPI_Pready` [MPI 4.1 chapter 4]; design/extents-and-liveness.md, b); NULL moves
  the declared counts. A slot holds a bridge table for each channel its legs cross, presented when the slot is made;
  the start writes each leg's entry for this start (a piece's rows in the slot, its pages from `counts`) and starts
  each table as the channel's next call. A send entry of this rank's part names the slot's `published` and one of a
  relayed part its relay record's (`struct mesh_entry` `word`, its region offset), each with `need` the start's number,
  which a kernel bound to the start stores there (`mesh_leave`). So a group or another start encoded before this
  start's publishing kernel sends nothing of the slot before it is written. A landing leg's wait reads its entry's
  `done`, and a relay's its landing leg's. A channel's calls are posted in their order, so that group's SENDs on the
  channel wait behind the start's until its
  publishing kernel has run: a GPU wait that needs them reaches its bound, and a host-path group
  fails as a value at its call while a start is not yet bound to the kernel that publishes it (`ncclMeshBind`
  `publishes`), as it does while the communicator's GPU groups are issued and not complete (MPI's persistent request:
  its buffer is the request's between `MPI_Start` and its completion). The start first completes any earlier group of
  the communicator with a send not yet published on a channel where it sends. It is then in flight as a composed group,
  whose sends count as published and whose landings as consumed: the caller's kernels do both, encoded after the
  start. Where its bridge refuses an entry (its link lost), the start still lays every other leg and returns the
  refusal as its value; the kernels bound to it run and publish as they would (design/pages-and-functions.md, Link
  error: the callgraph does not consume the status).
- A start takes the first of the request's slots whose use count is zero, else makes one, and `ncclMeshBind` finds a
  start's slot by its number (`slot_of`). A slot's use count is zero once its bound kernels have ended (above) and
  each of its tables' `sent` and `landed` has reached the slot's last call on that channel (~0 reads reached): until
  then the bridge may post a receive into the slot. A start that only publishes is free once its
  publishing kernel has ended and its SENDs are done; one whose landing no kernel reads, once it has landed. A start
  never takes the slot of the start before it, whose landing a kernel bound with the new start's number reads.
- `ncclMeshBind(request, start, lands, publishes, stream, index, groups)` binds the crossing record (`struct
  mesh_crossing`) at buffer `index` of the caller's next dispatch on the stream's encoder: with `lands`, start − 1's
  landings and this rank's part of it; with `publishes`, start's send part and publication; `groups` the dispatch's
  threadgroups. On the classic form the record is `setBytes`, and its slots are declared (`useResource`); on Metal 4 it is a record in rows of the slot it publishes, else the slot it lands (released by
  that slot's use count; a binding that names neither binds the library's one record of zeros), at the table's
  `index`, and its buffers join the residency set.
- `ncclMeshRequestFree(request)` is `MPI_Request_free`: the request is ended, and each slot is released as a plan
  never started again is ([Plans](#plans)): unlinked once its last start's stamps are in and every kernel bound to it
  has ended, its lease returned once a later call on each of its streams has its stamps. A request not freed when its
  communicator is destroyed keeps its slots' leases until the process
  ends, and a later free of it reads the destroyed communicator (MPI frees a request before its communicator,
  `MPI_Request_free` before `MPI_Comm_free`).

**The composed form.** The crossing's waits and its publication are the library's, compiled into the caller's kernel
as a prologue and an epilogue around its function (Triton-distributed's and TileLink's tile primitives, 2025; Flux's
communication fused into the GEMM's prologue and epilogue, 2024). The device header (`ncclMeshMetalPrelude`) gives the
kernel:

- `mesh_enter(crossing, tid)`, called by every thread: thread 0 of each threadgroup awaits each of the landed start's
  waits (each piece's `landed` past its entry, bounded), then a barrier and one system-scope fence;
- the views: `mesh_landing<T>(crossing, q)` (the q-th peer's part, in rank order, q below `mesh_landings(crossing)`),
  `mesh_own<T>(crossing)` (this rank's part of the landed start) and `mesh_send<T>(crossing)` (this start's part), each
  a `mesh_view<T>` whose `[k]` loads system-coherent and whose `store(k, v)` stores system-coherent; `view + k` is the
  view from its k-th element and `view.as<U>()` the same bytes as `U`; any kernel may make a `mesh_view` over a buffer
  of its own that the NIC writes. A view's address is private, and the crossing record's fields are private to the
  library's functions (`mesh_enter`, `mesh_leave`, the views), so a kernel names no holding's word and no slot's
  address (design/rubrics.md, What data code reaches; the operator: "nobody ever gives them a dereferenced anything");
- `mesh_leave(crossing, tid)`, called by every thread of every threadgroup: each threadgroup fences its stores and
  counts its arrival in the table's word, and the last to arrive, after one system-scope fence, stores the published
  slot's `published` (the pages of this start's part), then relays: it reads each relay record's landing count and
  stores its `published` once that leg has landed, looking at all of them in turn up to the library's bound (so no
  relay waits behind another); a relay whose leg has not landed by the bound is left unpublished, its entry and the
  channel's later calls behind it in the bridge's order. It counts the
  kernel's end in the table of each slot it names and zeroes the arrival word for the slot's next use. The relays read
  are those of the slot the kernel publishes, else of the slot it lands: a member that relays a part binds a kernel to
  each start, as every member of a sum, a reduce or a broadcast does.

The kernel's math is its own. A dispatch bound with neither `lands` nor `publishes` takes an empty record, and enter and
leave do nothing in it. The compiled form was chosen over dispatches of the library's own: the in-kernel crossings
already were this form (the step kernels awaited and released inside their own threadgroups), and a dispatch for each
wait and each publication adds a dispatch boundary a crossing (1-2 us each: design/hardware-and-discipline.md section 1)
and a publication that waits for a whole pass rather than for the last threadgroup's stores. examples/chain.metal and
lstm.metal are written this way ([docs/writing-a-service.md](../docs/writing-a-service.md)); torch-mesh's
`torch_mesh.fused.Exchange` makes the calls on torch's stream ([torch-mesh](torch-mesh/README.md)).

## Memory

The node's bridge registers its region, its configuration's share of the node's memory (etc/bridge/README.md), on
every device when the device comes up, and never again while the device is up
(design/transport.md). The operator, 2026-10-15: "the bridge's job is to allocate essentially all of the
memory on the entire device so that there are large already 'pinned' buffers which are ready to receive surprise
transfers in whatever order they happen from whatever peers and links are operational ... ontologically it must be
the responsibility of client programs to know how much allocation they need ... the bridge is there to behave like a
system call".

- The process is one client of its node's bridge: one attach of the node's region, made at its first `ncclMemAlloc` or
  communicator. Its slot owns the process's leases and holds the streams its communicators use. The pages a group sends
  from and lands in that are not the caller's are its plan's ([Plans](#plans)), and an exchange's are its slots'
  ([Persistent exchanges](#persistent-exchanges)): rows of the region, released by use count. The operator,
  2026-10-22, of such pages: "if we have to provide dynamically chosen expandedb uffers for reductions for our code, we
  can still allocate or choose f(n) pages for every n pages that need a collective-api-provided
  not-in-place-collective-provided-reduction. choosing pages for each accumulater or buffer or wahtever does not need
  any api different from the use of pages and indexes and scatter-gathers."
- `ncclMemAlloc(ptr, size)` is the lease, the system call: whole pages of the region covering `size` (one page for 0),
  already registered; a lease registers nothing. Its bookkeeping is the region's (the lease's rows are owned by the
  client's slot), so when the process exits or dies its bridge returns the pages as it returns every retired client's
  rows.
- A lease the region cannot meet is `ncclSystemError` naming the bytes asked and the pages free. While every client
  slot of the bridge is held it is `ncclSystemError` naming `mesh.h` `MESH_CLIENTS`. Where the process attaches no
  region of a bridge that runs, it names the requirement: the node's bridge runs with a configuration whose region
  holds the client's footprint (`ncclRemoteError` where the bridge stopped, else `ncclSystemError`). Nothing falls back
  to other memory.
- `ncclMemFree(ptr)` returns the lease's pages at once. The library registers no buffer of the caller's
  (`ncclCommRegister` is not provided): the region's leases are the registered memory.
- A lease is an operand of the node's region, which outlives a restart of its bridge: the next bridge adopts it
  (design/transport.md section 2), so a lease, its pages and every buffer over them stand across a bridge's restart,
  and nothing is reloaded. A region a release (or a bridge of another layout) expired is returned as a value: while the
  process holds a lease or communicator of it, its next `ncclMemAlloc` and communicator fail with `ncclRemoteError`
  naming the expiry, and `ncclMeshRegionExpired` reads 1; once the process has freed every one (`ncclMemFree`,
  `ncclCommDestroy`), or ended, the library lets that region go and its next lease attaches the bridge's next region;
  a process that held a communicator or a lease's `ncclMeshMemBuffer` ends instead, since the library made Metal
  buffers over the region for them, which its GPU work and residency set may hold; the library releases none of a
  caller's own Metal buffers over a lease, nor their residency. A process never holds two regions. The operator, 2026-10-14: "functions can fail from any one operand
  becoming invalidatd, servers should not fail when any operand is invalidated (ideally)"; "within call failures
  cannot be resume and instead must return an error (which a caller can presumably retry against immediately if that's
  what they want)".

On weights, KV and every buffer a peer reads, the operator, 2026-10-15: "why would weights or kv be outside of the memory pinned
pool? being outside of it means that they cannot be zero copy passed to peers. ... pipeline parallelism, context
parallelism, and megatron column parallelism are all required. ... you can't defend against this; it's *your code*
you're pretending you need to defend against; you can write it right all along."

Porting:

- An NCCL program takes its collectives' buffers from `ncclMemAlloc` (NCCL's allocator for the buffers it registers), and
  its node's bridge configuration's region holds them.
- A NumPy program takes them from `mesh_mpi.empty(shape, dtype)`, the role mpi4py's `MPI.Alloc_mem` plays; `mesh_mpi`'s
  own temporaries are such arrays.
- Python's lease is `mesh.lease(nbytes)`: a writable buffer for `numpy.frombuffer` or `torch.frombuffer`, returned to
  the region when it and everything over it are gone.
- torch-mesh's own CPU staging and temporaries are leases ([torch-mesh](torch-mesh/README.md)).
- `ncclMeshMemBuffer(ptr, &buffer, &offset)` is the Metal buffer over the whole lease holding `ptr` (any byte of it)
  and ptr's offset in it, made once and released by `ncclMemFree`: the buffer a tensor allocator makes MPS tensors over
  (torch-mesh's region tensors).
- `ncclMeshMemFreeAfter(ptr, word, stream)` returns ptr's lease once the work the stream holds before the call has
  ended: it encodes one `stamp` dispatch on the stream (its command buffer and encoder), which adds one to the 32-bit
  word at byte `word` of the lease (the caller's, zero when the lease was taken), and the lease is returned at the
  library's next lease taken or deferred after the word reads one (design/pages-and-functions.md, Freeing 6: the use
  count goes down when the reading program's stamp lands; no completion handler stands in for it).
- **Zero copy.** The operator, 2026-10-21: "zero copy methods must always be used over access by copy because
  memory bandwidth and flops are too scarce on this hardware api"; "good performance cannot be opt-in ... code which
  can't perform well can fail instead of being run". The bridge sends a crossing's bytes from the pages they lie in and
  lands them in the pages they belong in: a COPY lands in its destination, and a combine lands in a landing area of the
  communicator's and is combined from there. Each step is one entry, a range of page-table rows (`struct mesh_entry`),
  so it starts on a page by construction: nothing in the transport is a byte offset. Each SEND is one whole page with
  one SGE (`max_sge` is 1). The bridge registers the region in extents of 2^k bytes, the least power of two at which
  the region's extents number no more than the device's `max_mr` (1 GiB on the Studios' 86 GiB under max_mr 100;
  design/transport.md, The registered region), each inside one of the 4 GiB banks no registration may cross
  (`tbt_post_recv` keeps only an address's low 32 bits: `rdma/mesh-verbs.h`), so no page crosses an extent. So:
  - every crossing call's send buffer, and every buffer it sends from or lands in, is a lease's (in this node's region);
  - every piece starts on a page;
  - the program's own cuts lie on pages (`mesh_program_steps_cut`).
  A cut NCCL's call fixes (a reduce-scatter's and an all-gather's segment r at r x count, an all-to-all's segments, a
  send's or receive's buffer) that is not on a page, or an operand outside the region, fails the call as a value
  (`ncclInvalidUsage`, naming the piece and its buffer) before anything is presented; nothing is copied to make it
  fit.
- A collective whose send buffer lies apart from its result reads it in place where its program allows: a SEND of a
  range no step of the rank wrote reads the send buffer, the first combine of such a range writes send op piece
  (`combine3_*`), the result's ranges no step writes are copied from it after the publications, and no scratch buffer
  no step writes is made. A call in place (its send buffer its result, or an all-gather's own segment of it) sends from,
  combines into and lands in that buffer, with no copy; a combine into a range a SEND of it reads waits for that SEND's
  bytes to be sent ([One track](#one-track)). Otherwise its operand is made in the region: a premultiply (the
  scaled operand is a new value), or a copy within the region where the send buffer and the result overlap apart from
  in place. Reproduction target (metal-microbench
  docs/measurement.md 2026-09-30 part A, 95967e6..42a16ed): per 16.8 MB all-reduce, the library's copies of the
  operand 16.78 MB to 0; tp 271.5-272.2 ms a step. The slot copy into the rings, which that left, is gone too.
- The NIC reads a SEND's pages until its entry is stamped, so a group completes only once its SENDs are
  done. Once it has completed in the stream's order, nothing reads or writes its leases for it, and `ncclMemFree` may
  return their pages.
- A host-path group waits for its own communicator's GPU groups alone and fails while one of them is issued and not
  complete; it spins on its landings, never parks, and holds no lock another communicator's groups take, so a rank
  waiting in a host-path collective (an `ncclBroadcast` from a peer) does not stop its other communicators.

## Routing

The routes of every crossing come from one solver, run when a plan is made and when an exchange is initialized, on
data every member holds alike, so nothing is exchanged (design/rubrics.md C7). The operator, 2026-10-22: "congestion,
using all links which would reduce the aggregate latency of the complete compiled function instead of using arbitrary
links wihtout any feedback from the effect of the previous routing? an iterative solver that converges on all graphs
including unseen ones in fast finite time is preferred".

- **Input.** The traffic as demands (`mesh-plan.h` `struct mesh_demand`): a root, the leaves it joins (a bit a rank;
  NULL every rank), and the bytes each edge of its tree carries toward the root (`up`) and away from it (`down`).
  - A plan's collective: a demand a segment's owner, every rank a leaf: a reduce-scatter's segment `up`, an
    all-gather's `down`, a two-phase all-reduce's both, a single-phase all-reduce's whole operand `up` toward every
    rank; a reduce's operand `up` toward its root, a broadcast's `down` from it.
  - A plan's all-to-all (the hosted family and `ncclMeshAlltoAllV`): a demand a block from a to b, root b, leaf a, `up`
    its bytes.
  - An exchange: a demand a rank's part, root its owner, the ranks that land it the leaves, `down` its bytes.
  - The capacities are the communicator's map, one link a cable: each cable's bandwidth in each direction (a cable
    without a figure one unit), every cable an edge of its own; and a per-hop latency (`hop`, seconds: 0, the bridges'
    records carrying no latency). A tree's edge names its cable (`way`, the cable's place among the pair's cables in
    cable order), so the load spreads over a pair's cables by their capacities, and every step carries the cable it
    crosses (`struct mesh_step` `way`): the channel of that cable's stream. A point-to-point call between linked
    members is cut among the pair's cables by their bandwidths in page quanta.
- **Multiplicative weights** [Garg and Könemann, FOCS 1998; Fleischer, SIAM J. Discrete Math. 2000; the framework of
  Arora, Hazan and Kale, Theory of Computing 2012]: T = floor(ln m / ε²) + 1 rounds (m the directed links, ε
  `MESH_SOLVE_EPS`, 0.1). In each round a link's length is exp(ε (x_e − x_max) / ρ) / c_e, x_e its load so far over its
  capacity c_e and ρ the most load a round can put on the narrowest link; each demand takes the tree of least length
  from its leaves to its root (Dijkstra toward the root, an edge's length its `up` bytes times the length toward the
  root plus its `down` bytes times the length away, plus `hop`; ties to the lower rank), and after the round each
  tree's edges add the demand's bytes / T to their loads. A reduction's leaves meet in one in-tree, so its partials
  combine where their paths meet (design/pages-and-functions.md item 6). The average over the rounds is within (1 + ε)
  of the least worst-link congestion of trees, in O(ε⁻² m log m) rounds on any graph (collectives as multicommodity
  flow: TE-CCL [Liu et al., SIGCOMM 2024]; packed collective forests: ForestColl [Zhao et al., 2024]).
- **Output.** For each demand its distinct trees, each with its share (the rounds that took it over T), in three flat
  arrays: `demand`, `parent` (n entries a tree, UINT32_MAX off it) and `share` (`mesh_solve`; room from
  `mesh_solve_room`). A collective's trees become its program's (`mesh_flow_trees`, `mesh_compile`), an all-to-all's
  its table's routes (`mesh_flow_routing`, `mesh_block_parts`, `mesh_exchange_steps`), an exchange's its legs; every
  cut by shares is in page quanta (`mesh_cut_shares`).
- **When.** Once when a plan is made (`plan_calls`) and once when an exchange is initialized; a plan's restarts read its
  programs. The solver is deterministic: every member computes the same trees from the same map and demands (its exp
  and log IEEE-exact, the planner compiled with `FP_CONTRACT OFF`). A map that leaves two members of a demand apart
  fails the call as a value.
- **The operand.** `ncclMeshCommSetSolver(comm, solver, context)` sets a function of the solver's signature
  (`ncclMeshSolver`) in its place, every member passing the same between its groups (NULL: `mesh_solve`); the
  communicator's plans are made again. A learned optimizer passes its own (design/rubrics.md D6).

## Routes between unlinked ranks

A bridge posts only its own holders' entries (design/transport.md); members relay. A communicator's init holds a stream
only toward the members whose nodes its bridge reaches and whose bridges reach its node, so a channel joins two linked
members, and the bytes between two members no link joins land in the rows of a member between them, which sends those
rows on.
Bytes between two members no link joins cross the members between them, along the communicator's routing table, in the
calls where every member knows them: the operator, 10-21, "forwarding is REQUIRED"; "cabling is extremely dynamic from
this point onwards and therefore all topologies must be supported forever without justification by current environment
state"; "the collective api can contain a solver or optimizer".

**Routes** of every crossing are the solver's ([Routing](#routing)). An all-to-all's table (`mesh-plan.h` `struct
mesh_routing`, the solver's paths for its blocks, `mesh_flow_routing`): for each ordered pair
(a, b) of the communicator's ranks, the routes a block from a to b takes, each a list of ranks from a to b whose every
two consecutive ones are linked, and each route's share of the pair's bytes. A pair's bytes cross several routes at
once: a Thunderbolt link moves far fewer bytes a second than a node's memory, through which a member between two others
relays (the operator, 10-21, of single path against multipath: "a choice related to whether link bandwidth is higher or
lower than SHM<->CPU bandwidth").

**Collectives** run along their plan's trees ([Communicators](#communicators)): a member between two others on a
tree receives, combines or copies, and sends on, in its own program; nothing about it differs where two members are not
linked.

**The hosted family** (`ncclAlltoAll`, `ncclGather`, `ncclScatter`): every member knows every block (its two ranks, its
count, where it is read and where it lands), so every member lays the same steps along the table (`mesh_exchange_steps`,
MPI_Alltoall's exchange as a multi-commodity flow over routes). A block is cut among its pair's routes by their shares,
in quanta of the least multiple of the page and an element, so every part starts on a page (`mesh_block_parts`). A part
on a route of two ranks is one SEND and one COPY; a longer one is a hop a round: its first rank sends it from its send
buffer, each rank between receives it into pages of its plan (the relay area, one more of a plan's f(n) pages) and sends
it on, and its last rank receives it into its receive buffer. The relay's SEND is a send entry over the very rows its
receive landed in, nothing copied, published once that receive is consumed (the step's receive before it: a host-path
look, or a GPU wait and then the publication), so the relay sends no page before it has landed. Where the incoming link
is lost, the relay's call fails as a value at that receive. On the Metal path its later dispatches run as encoded
([Bounded waits](#bounded-waits)): the SEND is published once the wait has ended, and the next hop lands the bytes the
relay's rows hold, its own positions being on a link that was not lost. On the host path the relay's group returns at
the loss and retires its remaining positions. A relay's group ends once its SENDs are done, so its plan's pages are started again only
after the bytes it relays have gone on. Gating the relay's SEND on the incoming landing entry's stamp itself (its send
entry's `word` the landing entry's `done`, its `need` k) would send the piece once it has landed, with no wait or
publication of the relay's; the plan's send entries name their table's `ready` instead, which the relay's program
stores in its order.

**Calls every member plans.** An `ncclMeshAlltoAllV` and a persistent exchange are passed alike at every member (the
count matrix; each rank's part and where it lands), as the hosted family's blocks are known to every member, so each
member compiles from the solver's routes the pairs that route through it, with no message (MSCCL's and TACCL's relay
schedules, compiled at every rank [Cowan et al., ASPLOS 2023; Shah et al., NSDI 2023]). For a pair x to b through a:
x's plan presents its pages as a send to a, a's plan presents those rows as a landing and then as a send to b, and b's
plan lands them. An all-to-all's relay is the hosted family's (above: a send gated on the step's receive). An
exchange's relay is a leg of a's slot: a's send of the rows its landing leg of a piece landed in, gated on the relay record's
`published`, which a's kernel bound to the start stores once the landing has landed ([Persistent
exchanges](#persistent-exchanges)); the slot's rows return by its use count once its entries are done.

**Pairwise calls** (`ncclSend` and `ncclRecv`). `ncclMeshGroupPairs(from, to, bytes, count, comm, stream)`, called in a
group by every member with the same list, names every message of the group's point-to-point calls: message i goes from
rank `from[i]` to rank `to[i]`, `bytes[i]` bytes, the k-th message of a pair in the list being the sender's k-th
`ncclSend` to that peer and the receiver's k-th `ncclRecv` from it. Every member then plans every message, as it plans
an all-to-all's blocks: each message a demand of the solver (root its receiver, leaf its sender), cut among its routes
by their shares, a member between two that no link joins landing it in its plan's relay rows and sending those rows on
(the hosted family's relay, a send gated on the step's receive). The group's `ncclSend` and `ncclRecv` calls then give
only their buffers. This departs from NCCL's point-to-point semantics, where only the two ranks of a message call
anything (design/rubrics.md C1): the members between two unlinked ones must know the message before its bytes arrive,
and C2 (every cabling served without a branch) and C7 (every position known a priori) rank that over C1. Without the
list, a message is its two ranks' alone, on the cables joining them, and one between two ranks no cable joins fails as a
value where it is planned. torch-mesh gives the list a caller passes from its
compiled program (`torch_mesh.batch_isend_irecv(ops, pairs)`); torch's own `batch_isend_irecv` gives none.

## Ranks, lists and streams

- Two ranks on one node: no cable joins them, so a call that crosses between them fails as a value where it is planned
  ([Communicators](#communicators)).
- Each direction between two ranks is its own list, so a rank's k-th send to a peer pairs with the peer's k-th receive
  from it, as NCCL's do. A blocking exchange with one peer completes as one group (`ncclGroupStart`/`End`; torch's
  `batch_isend_irecv`); as two blocking sends before the receives, each waits on the other's group. The library does
  not detect unequal counts or different collectives on two ranks: no landed payload is checked.
- A crossing whose buffer is not a lease's, or whose piece does not start on a page, fails as a value
  ([Memory](#memory)).
- Communicators know nothing of one another, and no two hold one stream: a link has up to four (`MESH_LINK_STREAMS`;
  fewer where its plan has fewer lane pairs, design/HETEROGENEOUS-LINKS.md), so that many communicators use a node
  pair at once, and one more is refused (EBUSY) until one of them is destroyed.

## Failures

No clock ends a group's wait (the operator, 2026-10-22: "we don't introduce race conditions or use timing anywhere ...
because using timing is always slower than not using timing"). A group presents nothing per call and waits for no
room (design/transport.md section 14, "The plans"): its plan's tables were presented once, a start is one store a
channel, and a table is started again only once its last call's stamps are in; a call whose every plan is still in
flight makes another plan. A host-path group
waits for its own communicator's GPU work until its command buffer completes, an error status (which the platform's
watchdog gives a command buffer that overruns) failing the group, and one not yet committed failing it at once
(`ncclInvalidUsage`). Each wait reads the revoke between looks, so a revoke ends it (design/rubrics.md M3, 10-18), and holds no lock: the groups' lock is
released for the wait and taken again after, so other communicators' groups and revokes go on (M4). A host-path group's waits end at
their counts, at a count's `~0` (its bridge's, where the link is lost or the bridge stops), or at a revoke; a GPU wait at its
count, at a count's `~0` or at its bound ([Bounded waits](#bounded-waits)), the operator's "poll bound for the node"
(10-21). A group presents its entries to its bridge as it is issued. A member that fails, is revoked or ends sends
nothing more: its bridge posts no page of a holding that ended, and nothing of its own. A peer that has not yet issued
its group, or that failed, is waited for, a host-path group's wait on it until a landing or a revoke; whether to revoke
and run again is the caller's (the operator, 2026-09-10, design/pages-and-functions.md: "recovery is something which is
only sanctioned through the mechanism of a totally feed forward nfe computation callgraph being re-run by the calling
process because they don't like the results they got"). A holding given with units still outstanding takes its stream out of its link's pairing, and a
link none of whose streams is live is lost and paired again from 0 (design/transport.md section 14). A failure is
the value of the call it happened in, and `ncclGetLastError` names it (the node of the channel); nothing of it is kept,
and no later call reads it (the operator, 2026-10-22: "the collective code does not detect mend, heal, or prevent any
kind of error. it is not stateful with respect to errors"). A host-path group that fails counts its remaining entries
done (`run`) and writes no `ready` for them. An entry its bridge refuses fails its call while the group's other entries
are laid and the group runs as compiled (design/pages-and-functions.md, Link error: "The callgraph does not consume that
status"), so every channel's later entries stay in step with its peer's. A GPU wait that ends at
a count's `~0` or at its bound records nothing; its kernel goes on with the bytes the area holds
([Bounded waits](#bounded-waits)). `ncclCommGetAsyncError` gives a non-blocking init's state and result (the operator,
2026-10-14: "it's okay for a function call ... to fail instead of returning if the link topo changed mid function
execution"). A communicator keeps the map its init made, so a change
that leaves its links working (a link lost elsewhere, a node that joins) reaches the next communicator
(metal-microbench makes each call's communicators at the call's start). A rank that departed is left out with
`ncclCommShrink` ([Communicators](#communicators)), and a joining one taken in with `ncclCommGetUniqueId`
and `ncclCommGrow`, an init at the id comm's root broadcasts on comm, holding streams of its own as every init does. `ncclCommRevoke` is NCCL's, from any thread and under the communicator's own lock
alone: its operations in flight are revoked and its resources kept (its streams stay held, and its bridge sends what
its groups published and nothing else); an init
still running (a non-blocking one's) ends as a value, its host-path groups return at their next look, its GPU waits
end at their landings or their bounds (no kernel stops at the revoke), and every later group on it is refused (`ncclInvalidUsage`, "revoked") until `ncclCommDestroy` gives its
streams back; `ncclCommFinalize` after it returns `ncclSuccess`, and its `revokeFlags` is not read. `ncclCommDestroy` awaits the command buffer of the last GPU program the library committed for the communicator until its
status is Completed or Error (the platform's watchdog ends one that overruns), with no clock and the groups' lock
released meanwhile; one not committed is not awaited; work encoded in a caller's encoder (torch-mesh's) is the caller's to finish first, as NCCL's
contract has it: a kernel of it still queued when its streams are given back would write a stream the next holder
may take. Likewise, while a revoked communicator still holds its streams, its bridge may still post receives
into a lease its groups named, for entries already presented, until `ncclCommDestroy` gives the streams back
(design/transport.md section 14, "A holding's end"): the library keeps its own such leases (plans, slots) until then
([Plans](#plans)), and the bridge may post into a caller's lease that a lost group named until then too.

A communicator's bytes of its node's region are its plans' and its exchanges' leases ([Plans](#plans),
[Persistent exchanges](#persistent-exchanges)). The process
holds one of its bridge's client slots (`mesh.h` `MESH_CLIENTS`) whatever its communicators; each communicator of two
ranks or more holds a stream on every cable that joins it to another member from its init to its destruction, one
communicator a stream (a holding's plans are its holder's alone), of the up to four a cable has; a channel
is one held stream.

## Devices, operators and errors

- One device: `ncclCommInitAll` takes one device and `ncclCommCuDevice` returns 0 (a node has one Metal device).
- Operators: `ncclRedOpCreatePreMulSum`'s scalar is read when the operator is made (`ncclScalarHostImmediate`), else
  where it lies at each host-path call that reduces by it (`ncclScalarDevice`; a Metal-path call by it fails as a value,
  its kernels taking the scalar as an argument written when the group is encoded). An operator reduces the datatype it
  was made for alone (NCCL's `hostToDevRedOp` alike), and a communicator makes as many operators as it holds handles
  for. `ncclAvg` of an integer type is the wrapped sum's magnitude over n, its sign kept (NCCL's `FuncSumPostDiv`);
  of a floating type, a premultiply by 1/n.
- Errors: `ncclGetLastError` names the calling thread's last failure; a failure kept with its communicator (or a
  non-blocking init) is kept once with an identity, and reading it again copies nothing it already holds
  (design/rubrics.md, Strings are messages).

## The planner (`mesh-plan.h`)

libmesh's read side and the collective planner (`mesh-dataflow.c`, `mesh-collective.c`): one header every consumer uses
as written (C, Objective-C, Swift's module map, torch-mesh's extension, and Python through `mesh_c_build.py`'s cffi),
plain C declarations and integer constants. Every execution reads its node's bridge here when it starts; no consumer
gathers, parses or joins bridge records itself.

- `mesh_reached(out, capacity, &node)`: this node's bridge's node, and the cables it reaches other nodes by, each entry a
  node, its `cable` (both bridges compute it alike from the cable's first stream) and the cable's bandwidth in bits a
  second; their count (0: the node alone), or -errno where the bridge is unreachable. A
  node is all a client names of the mesh (design/rubrics.md, Three kinds of source). `mesh_region(&bytes, &unleased)`:
  the region's arena and the bytes no lease holds. `mesh_nodes_health`: every node's record the bridge holds (its
  bridge's node, its region's bytes, the node's memory and `hw.model`, its boot and the memory available as its bridge
  last sampled it).
- **A link map** (`struct mesh_link_map`), an operand: `nodes` nodes and an edge list, one edge a cable (`struct
  mesh_edge {a, b, cable, capacity}`, the capacity in bytes a second), built from the bridges' records of each cable
  (`mesh_map`, a communicator's `map_of`: each cable's bandwidth the lesser of its two ends', over 8). The capacity is a
  hardware fact: the solver's capacity of a pair is its parallel cables' summed, and `mesh_routes` takes of routes of
  equally few edges the one over the faster hops. A construction given an edge with no capacity fails (EINVAL; the
  communicator's init, `mesh_map`, `mesh_map_members`, Python's `LinkMap`): no capacity is defaulted. There are no map
  kinds (the MESH, RING, TREE and GRAPH of 761223a8 to this revision kept a second store of which pairs are joined,
  beside the edges, which no parallel cable fit, and a complete map's unlisted pairs were edges of capacity 1 byte a
  second).
  `relays` are nodes numbered from `nodes` that a route may cross and that hold no rank (a communicator's map has none:
  the bridges forward nothing). `mark` is each node's and relay's mark (its bridge's node number), which every map
  carries. `next` is the routing table, of N = nodes + relays: each node b's in-tree, `next[v x N + b]` v's parent in
  it. The route between a and b is the path up the in-tree of the one of the higher mark from the other, so a route
  and its reverse are one path. `mesh_link_between` is whether the map links two of its nodes (their route crossing
  relays alone); `mesh_route_path` writes a route up the map's own table (none where the map has no table); no table
  is derived where a route is read, and `mesh_map_members` takes the whole map's table and marks (EINVAL without
  them). The table-less forms (a route as the direct link alone, a join check by search, a members' map whose
  table it derived, default marks) served synthetic maps alone and are deleted.
- `mesh_routes(map, next)`: each node's in-tree of shortest routes toward it, by fewest links, then the least sum of its
  hops' inverse bandwidths in whole fs a byte (Dijkstra [1959] in its dense form, every length an exact integer), each
  node's parent among its equals the one of the least hash of its mark, the parent's and the root's (48 bits of
  SplitMix64's finalizer [Steele, Lea and Flood, OOPSLA 2014]: equal-cost multipath by a hash, as ECMP spreads flows
  [RFC 2992]). So routes of equal length spread over the links, every bridge and member that holds the same records
  makes the same table, and a route moves only where the shortest routes between its ends change.
- `mesh_map_members(whole, members, n, whole_node, map)`: the members' map of a whole map, its other nodes relays,
  every edge of the whole map kept with its cable and capacity, two members linked where their route crosses no other
  member; `mesh_map` is it over the bridge's records as they
  are when it is read.
- **Trees** (`struct mesh_trees`): tree t rooted at `root[t]`, node v's parent `parent[t x nodes + v]`, `log_weight[t]`
  its unnormalised log share of what its root gathers and sends; a root's trees split its elements by exp(weight), each
  the floor of its share and the first ones one element more. A reduction toward r runs up r's trees, a broadcast from
  r down them. `mesh_flow_trees` makes them from the solver's trees of a range of demands.
- **The solver** (`mesh_solve(map, eps, hop, demands, count, room, demand, parent, share)`, `mesh_solve_room`):
  [Routing](#routing). Its output holds `mesh_solve_room` trees, which no solve passes: each of its T rounds adds at
  most one tree a demand. `mesh_flow_routing` makes an all-to-all's routing table from its demands' trees (each a path
  from its leaf to its root); `mesh_cut_shares(elements, element_bytes, page, share, count, first, length)` cuts
  elements by shares in quanta of the least multiple of the page and an element.
- **Programs** (`struct mesh_program`, as MSCCL-IR is the program the MSCCL runtime interprets [Cowan et al., ASPLOS
  2023]): every rank's moves, its operand cut into parts, part p within segment `segment[p]` at its log weight.
  `mesh_compile(trees, what, root, whole, chunks, contributors, ...)` (room from `mesh_compile_room`): a reduce up the
  root's trees, a broadcast down them, a reduce-scatter's segment s up s's trees and an all-gather's down them; an
  all-reduce two-phase (`whole` 0: the reduce-scatter and then the all-gather [Patarasuk and Yuan 2009; Blink]) or
  single-phase (`whole` 1: the whole operand reduced toward every node up its heaviest tree, a forwarding node's partial
  from scratch of its own; at two nodes the direct exchange, whose one SEND and one REDUCE meet in the operand, the
  combine after the SEND's bytes are sent [Trivance 2026]). Each tree's part is cut into `chunks`, chunk j of every move
  at its round + j, so a node forwards a chunk as it arrives (Blink; NCCL's chunked trees; MSCCLang's chunks). A
  round's SENDs come before its receives; a message coalesces with the next between the same two nodes in the same
  round where their parts adjoin. `contributors` (a bit a node) restricts whose operands an all-reduce combines and whose
  segments an all-gather gathers.
- `mesh_program_steps_cut(program, rank, operand, segments, quantum, steps)`: a rank's steps for an operand, its
  segments the given sizes (MPI_Allgatherv's and MPI_Reduce_scatter's counts) or the program's own cut, every cut the
  program makes on a multiple of the least common multiple of the quantum (the communicator's page) and an element, a
  part within a segment the floor of its share of the segment's quanta, the remainder one quantum each to the first
  parts of weight above 0.
- **Lists**: `mesh_lay` and `mesh_group_lay` place a rank's steps on its channel toward a peer, its SENDs on the send
  list and its receives on the landing list, by round, the collectives in call order, then the hosted all-to-alls,
  then the point-to-point calls, so a rank's k-th send to a peer is the peer's k-th receive from it.
  `mesh_steps_before` and `mesh_group_before` give each step the receive before it (a step's first event waits until
  that receive is consumed; a hosted all-to-all's receives each progress alone).
- **Routing tables** (`struct mesh_routing`): `nodes` nodes and `routes` routes; pair (a, b)'s routes are
  `[pair[a x nodes + b], pair[a x nodes + b + 1])` (`pair`: nodes x nodes + 1 entries, nondecreasing; none where a is
  b), route k the nodes `node[start[k] .. start[k + 1])` from a to b, every two consecutive ones linked and none twice,
  and `share[k]`, above 0, its share.
- **Blocks along a table** (`struct mesh_block`: `elements` of `element_bytes` from rank `from`'s buffer `source` at
  element `first` to rank `to`'s buffer `target` at element `at`). `mesh_block_parts(table, block, shortest, page,
  parts)`: the block cut among its pair's routes (`shortest`: those of the fewest nodes) by their shares, in quanta of
  the least multiple of the page and an element, so every part starts on a page wherever the block does; both ends
  cut it alike from the block and the table. `mesh_exchange_steps(table, rank, blocks, count, pairwise, relay, page,
  steps, block, &relayed)`: a rank's steps of an exchange of blocks: a part on a route of two nodes one SEND and one
  COPY of round 0; a longer one (not `pairwise`) hop j in round j, a rank between its ends receiving it into buffer
  `relay` (at the next of the bytes it relays, each part from a multiple of the page, `relayed` their total) and
  sending it on from there; `pairwise`, the parts on longer routes are no rank's steps. A round's SENDs, then its
  receives, each in block, then route order, so a hop's SENDs at one end and its receives at the other are the same
  parts in the same order (`mesh_lay`). `mesh_exchange_room` the most steps it makes at a rank. The table is the
  solver's for these blocks (`mesh_flow_routing`), so every block's pair has a route.

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
best = comm.allreduce(x.max(), op=MAX)  # IN_PLACE as a send buffer
comm.Allgatherv(part, [whole, counts])  # a count a rank: ncclMeshAllGatherV
comm.Reduce_scatter(full, mine, counts) # ncclMeshReduceScatterV
```

Subcommunicators are `Split(color, key)` and a Cartesian grid: `grid = comm.Create_cart([2, 2])`, `grid.Sub([False,
True])` its row, `grid.Shift(0, 1)` the neighbours, as mpi4py spells them.

`Comm()` is the world: this process is rank `RANK` of `WORLD_SIZE`, its communicator made by NCCL's bootstrap at
`NCCL_COMM_ID`, else `MASTER_ADDR:MASTER_PORT`; `COMM_WORLD` is `Comm()`, made at first use. Every call is the host path:
C-contiguous arrays at host pointers, a NULL stream, complete when it returns. The methods are `Get_rank`, `Get_size`,
`Allreduce`, `Reduce`, `Bcast`, `Allgather`, `Allgatherv`, `Reduce_scatter_block`, `Reduce_scatter`, `Alltoall`,
`Gather`, `Scatter`, `Send`, `Recv`, `Sendrecv` (one group, so both directions pair), `allreduce` (a Python
number) and `Free`; `IN_PLACE` as a send buffer, and `None` for a buffer a rank does not use (`Gather`'s off the root,
`Scatter`'s send off it). An array of a dtype the library has no type of moves as its bytes; a reduction takes the
library's types (with `ml_dtypes`, bfloat16 and the two float8 types too). `Allgatherv` and `Reduce_scatter` take a count
a rank as mpi4py spells them (a receive buffer `[array, counts]` or `[array, (counts, displacements)]`, the
displacements the counts' running sums), one program over the segments; `Allgather` and `Reduce_scatter_block` are the
same calls with equal counts. `Shrink(exclude)`, `Grow(size)` with `Comm.joining`, and `Revoke()` are NCCL's.
`Create_cart(dims)` holds a grid of fewer ranks than the communicator over its first ones (a Split of them), the others
`None`, as `MPI_Cart_create` gives them `MPI_COMM_NULL`. Not here: nonblocking calls, `Barrier` (a sync with no data), `Alltoallv`, `Gatherv`,
`Scatterv`, and object (pickled) collectives. `empty(shape, dtype)` is an array in the region (`mesh.lease`), the memory a program
hands the collectives, the role mpi4py's `MPI.Alloc_mem` plays; the module's own temporaries are such arrays.

metal-microbench `tools/mesh/numpy_power.py` is a ported power iteration, and `tools/mesh/numpy_summa.py` SUMMA on a
2-D grid (row and column broadcasts).

## Not implemented

ncclCommInitRankScalable, ncclCommSuspend/Resume, ncclCommRegister/Deregister (the region's leases are the registered memory),
ncclCommMemStats, the window and signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd,
ncclSetEncryption, the ncclParam functions (`nccl.h`).
