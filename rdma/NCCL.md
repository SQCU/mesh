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
or region. Ranks are launched as torchrun launches them (`RANK`, `WORLD_SIZE`); the communicator's id reaches every rank
by the caller's means (torch's store, `NCCL_COMM_ID`), and the library opens no socket ([Communicators](#communicators)).

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
`merge_row`), and is the host's (`combine_from`, `merge_rows`, `premultiply`, `truncdiv`): the small floating types in
float32 rounded to nearest even, float8 decoded and encoded by `mesh.metal`'s `f8_to` and `f8_from` (OCP 8-bit floating
point, 2023: E4M3 with no infinity and NaN S.1111.111, E5M2 IEEE-like, saturating to NaN and infinity), integers
wrapping. A landing's combine loads both its operand and the landed bytes system-coherent: the operand may itself have
landed in place (a COPY earlier in the group), and the landed bytes are the NIC's.

## Communicators

- `ncclGetUniqueId(id)`: the id names the communicator and its incarnation: 128 bytes drawn at random
  (`arc4random_buf`) at each call, so no two runs of a program share one. The caller hands every rank the same id
  (torch's store, a file); the library sends it nowhere. A communicator's id is the FNV-1a hash of its 128 bytes, 64
  bits (1 where it is 0); a split's or a shrink's is made from its parent's, so it carries the parent's incarnation.
- `ncclCommInitRank(comm, n, id, rank)`: NCCL's signature and meaning, with no bootstrap of NCCL's: no root, socket,
  listener, address or gather. The rank attaches its node's bridge, registers (its communicator's id, its rank) there,
  and reads from the map which node holds each rank's registration.
- `ncclMeshCommInitRank(comm, nranks, id, rank, name, config)`: the same init; the map is its members' (below).
  - `name`: the communicator's, which its failures and its splits' names carry (NULL: its id's hash, in hex).
  - `config`: NULL, or `blocking` not 0, is a blocking init; `blocking` 0 is NCCL's non-blocking init (below).
  - The library compares neither across ranks. `ncclCommInitRank` is this call with no name or config, and
    `ncclCommInitRankConfig` the same with its config.

**Registration and the members.** The client's words, and who reads them (design/transport.md section 14, "The queue"):

| Word | Written by | Read by |
|---|---|---|
| the communicator's name table: a table of no tuples whose `name` is (id, rank), linked into the process's queue | the init, once; the destroy takes it out | its node's bridge, which carries the name in its record as soon as a port's child walks it |
| a record's `names` and `name` | the bridge | every member, through the map (`mesh_names_find`) |
| `members[r]`: the node that holds (id, r) | the init, from the map | the communicator's map and channels |

- The init links its name table, then reads the map until every rank's name is in it, a look each millisecond. The
  rank's node is the node whose record holds its name: "ranks aren't real ... they do not exist until some peering has
  already happened" (the operator, 2026-10-22). The init ends as a value at a revoke, or `MESH_PLATFORM_NS` after the
  last look that found a rank not found before (`ncclRemoteError`, ETIMEDOUT): a rank no bridge has heard of fails its
  peers' inits, for their callers to run again (C6); nothing retries it.
- A tuple toward a rank whose name no record yet holds stays in the sender's queue until one does: a split's members
  send to each other before every member's name has flooded, and nothing waits for it.
- `ncclCommSplit` gathers each rank's color and key over comm (NCCL's bootstrapSplit, one `ncclAllGather`); the child's
  id is FNV-1a over (comm's id, comm's split count, color), its members comm's members of its color, in key order. Each
  member links its name table and reads no map for its members: they are comm's. `ncclCommShrink` (NCCL's signature)
  is the same over the ranks it keeps (color −1), with no exchange. A rank among those it excludes gets `NULL`, as
  `NCCL_SPLIT_NOCOLOR`'s.
- `ncclCommGrow` broadcasts the id over comm (an existing member's `rank` is not read: it keeps its rank), then inits.
  Inside a group, `ncclCommSplit` and `ncclCommGrow` run their own collectives at once, the group's calls set aside until
  they return; `ncclCommShrink` makes none.
- While the process holds a lease of a bridge that stopped, its next lease (`ncclRemoteError`) and communicator
  (`ncclInvalidUsage`: the process's own leases, which only it can free) are refused naming it,
  until it has freed every such lease: a process never holds leases of two regions. A node's region is most of its
  memory, and GPU work on a stopped region's leases beside the new region's registration wired the Mini's whole memory
  (10-20, metal-microbench output_data/bridge-20261019/REVIEW-131.md).
- A non-blocking init sets `comm` at once, runs on a thread of its own and returns `ncclInProgress`;
  `ncclCommGetAsyncError` gives `ncclInProgress` until it ends, then its result. `ncclCommRevoke`, `ncclCommAbort` and
  `ncclCommDestroy` end it as a value, its look at the map reading the revoke each millisecond. `ncclCommAbort` or
  `ncclCommDestroy` then frees the communicator, failed or not; no call but these four takes it before its init has
  ended.
- Making a communicator, a lone rank's too, attaches the process's client of its node's bridge ([Memory](#memory)),
  so the node's bridge runs; it fails where `ncclMemAlloc` would.
- The TCP bootstrap of versions to 135 is deleted: the root's listener, `NCCL_COMM_ID` as an address, the node's fabric
  address, the two gathers and their reports, their phases and 100 ms slices, and the streams the second gather held.

Each communicator makes its map at its init (`nccl-mesh.c` `map_of`) and keeps it for its life: its members its nodes,
each marked by its bridge's node number, two joined by one link wherever the whole map's route (`mesh_routes`) joins
their nodes, and with it the cables that link uses (`struct mesh_uses`, the solver's graph: [Routing](#routing)).
Every bridge relays,
so two members are joined whatever nodes lie between them (design/transport.md section 14, "Relays"); two members on
one node are joined by no link, and a call that crosses between them fails as a value where it is planned. A member reads
its own bridge's map; members whose bridges hold the same records make the same map. A later communicator reads the map
again, so a node that joins, a node gone or a link lost (down until its ports' next children pair it again, with nobody
acting: design/transport.md section 14) reaches the next communicator with no edit and no restart.

The bridges' records are a map whatever reaches what: a node that still has a record may be reached by no bridge,
and the whole map (`mesh.current()` with no members, `mesh_map` over none, mesh-stat's `map`) holds it as it is. The
trees and the collectives run only on a map that joins their members: a communicator's collective on a map that does
not fails as a value naming two members no route joins (`ncclInvalidUsage`).

**Keys.** A communicator has a channel toward each member on another node: `peer`, `node`, and the counts of its two
lists (`sends`, `lands`: the entries laid on each since the init, so a rank's k-th send to a peer is the peer's k-th
receive from it), `published` and `consumed`. A send entry on the channel toward rank q is a tuple whose key is (id, q,
me, 0, `sends` + i), and a landing entry from q one whose key is (id, me, q, 0, `lands` + i), i the entry's place in
this call's list: the two ends compose the same key for the same pages. A persistent exchange's keys are in a space of
its own, its number on the communicator (counted from 1 at every member's `ncclMeshExchangeInit`, a whole word): (id,
the receiving rank, the sending rank, the exchange's number, start × the channel's legs + the leg's place), so no group
between two starts moves them, and no two exchanges share a key. The sender's registration is the key's first and
third words, so its bridge can tell when the sender's route moves. The bridge compares keys and names as words; a run
whose key no started landing holds, while a started landing of the same first four words has a later position by their
signed difference, is of a call already over (design/transport.md section 14, "The index").

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR). A plan compiles each of its collectives when it is made, along the trees the solver gives the plan's
traffic ([Routing](#routing)):

- the all-reduce single-phase (each member's sum up its own trees, the map's diameter in dependent crossings; at two
  members, and wherever every two members are linked, the direct exchange);
- the reduce-scatter and the all-gather, segment s up and down s's trees;
- each root's reduce and broadcast, up and down the root's trees.

Each is uncut: a step's piece is one entry of its plan's table, any size, which the bridge alone cuts into SENDs. A
program's own cuts (a tree's shares, a segment) lie on the region's pages (`mesh_program_steps_cut`), so every piece
starts on a page wherever its buffer does ([Memory](#memory)). Nothing about a program is decided where a plan
restarts. The caller's program table (`ncclMeshProgram`), the forms operand (`ncclMeshCommForms`: two-phase all-reduce
and chunked trees by size class) and the solver operand (`ncclMeshCommSetSolver`) had no caller in either repository
and were deleted 2026-10-22 with the two-phase and chunked forms only they selected.

**All-to-all with host counts.** `ncclMeshAlltoAllV(send, counts, recv, type, comm, stream)` is MPI_Alltoallv with
its counts on the host, every member passing the same matrix: `counts[a x nranks + b]` elements go from rank a to rank
b, packed in rank order in a's send buffer and in b's receive buffer (a rank's own block, `counts[r x nranks + r]`, is
copied). MPI_Alltoallv takes each rank's own `sendcounts` and `recvcounts`, a row and a column of this matrix; here
every member passes the whole n x n matrix, a deviation argued from the hardware's interface (design/rubrics.md C1's
upper pole: a member between two that no link joins relays a block, so it must know the block before its bytes
arrive, and every member plans every block, C7). A porter passes the matrix its program's sizes give: a program ported
from MPI holds its own row and column and gathers the rest once; an SPMD program holds the matrix; torch-mesh, which is
given torch's per-rank splits, fails such a call as a value and takes the matrix in `torch_mesh.all_to_all`, from the
compiled program (its configuration's shapes and shares, a program's maximum extents, how much of a block a step
fills being data in its pages). A segment may start anywhere in its buffer (an entry's byte offset). NCCL's `ncclAlltoAll` is it with
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

**Uneven segments.** `ncclMeshAllGatherV(send, recv, counts, type, comm, stream)` and
`ncclMeshReduceScatterV(send, recv, counts, type, op, comm, stream)` are MPI_Allgatherv and MPI_Reduce_scatter: rank
r's segment is `counts[r]` elements (any of them 0), packed in rank order. Each is one program, the communicator's
all-gather or reduce-scatter, its operand cut into those segments (`mesh-plan.h` `mesh_program_steps_cut`: a part
within segment s takes its tree's share of `counts[s]` in pages, the solver's; a segment may start anywhere in its
buffer), so a capacity-shaped division of the work (metal-microbench
docs/placement.md) is one tree program over unequal segments rather than a rooted broadcast or reduce a rank. MPI
specifies the counts alike on every rank [MPI 4.1 §6.7, §6.10.2], and the library compares none across ranks; equal
counts are NCCL's own calls. Reproduction target (metal-microbench docs/measurement.md
2026-09-30): tp SP=1 at capacity-shaped parts 255-256 ms a step; per call, the SP=1 all-gather 214-229 us and
reduce-scatter 196-212 us; one uneven all-gather against the same segments as N broadcasts.

## Two paths

Both paths move a crossing's bytes the same way: zero copy, through tables in the process's queue the bridge serves
(design/transport.md section 14, "The queue"), in MPI's persistent form: a request made once (`MPI_Send_init`, `MPI_Recv_init`) and started
at each call (`MPI_Start`) [MPI 4.1 sections 3.9 and 6.13].
- **A table a channel, linked once.** A group's plan ([Plans](#plans)), and each slot of a persistent exchange
  ([Persistent exchanges](#persistent-exchanges)), holds in its own pages one table for each channel it crosses
  (`mesh.h` `struct mesh_plan`): its send entries and then its landing entries, each step one tuple (`struct
  mesh_entry {row, pages, word, need, done, count, posted, key}`: a range of page-table rows, each row a page, and its
  key), in the order the group's program lays them at both ends (`mesh_group_lay`). The table is linked into the
  process's queue once, when the plan or slot is made (`mesh_queue_link`), and taken out once, when the plan or slot
  will not be started again (`mesh_queue_unlink`, [Plans](#plans)).
- **A call is a stamp.** The process numbers its table starts 1, 2, 3 and on, across every communicator, in the order it
  issues them (the attach's `calls`), so its bridge sends a process's tables toward a node in that order. A call's start
  writes its entries' rows (the caller's buffers and the plan's own pages, `place_group`), keys ([Keys](#communicators))
  and each send entry's `need`, sets each entry's `count` and `posted` to k << 32 and the table's `completed`, stamps
  the entries of no pages and a way with none (`table_begin`), and then stores the table's `start` = k. Nothing else is
  linked per call, nothing is numbered modulo anything, and nothing waits for room.
- **The stamps.** The bridge sends each page of a send entry once the word the entry names (`word`) reaches its `need`,
  lands each page of a landing entry in the page its row gives, by the key of the run it came in, and stamps the entry
  `done` = k once its pages have completed, and the table `sent` and `landed` = k once every entry of each way has; ~0
  is a call lost. A wait reads its own entry's stamp. The library never learns a lane, a frame, a cable or a queue
  pair.
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

### Composed starts

A persistent exchange's start is a composed group in flight: its sends count as published and its landings as consumed
when it is started (the caller's kernels do both, encoded after the start), so a completion passes its entries
(`consumed_past`). The one-track form (a completion on one channel of one communicator as plain dispatches in the
stream's order, `complete_track`, chosen by the schedule's `lone`) was deleted 2026-10-22 (mesh 91e72125) as a second
form of completion: every completion runs through the tracks below, a two-rank collective's on its one track.

### Per-channel progress

A completion on two or more tracks runs every channel on its own (`nccl-mesh-metal.h` `metal_record`, `metal_tracks`).
This is NCCL's per-peer progress, chosen by the operator on 2026-10-20 in place of one serial order of the whole group.

- **Tracks.** Each channel's remaining entries are a track with two cursors: its publications (send entries) and its
  consumptions (landing entries), each in list order.
  - Nothing credits a publication.
  - A consumption runs once its landing came, and it is combined where it belongs; where it combines into bytes a SEND
    of its channel reads (the two-rank all-reduce's SEND reads the operand its REDUCE combines into), it runs once that
    publication has run (`need`), and its wait awaits that SEND's stamp (`war`), so the SEND's bytes are read before
    the sum is made and no scratch copy is made.
  - A first combine of a call reading its send buffer in place is a `spin`, then `combine3_*`; a later one a `land_*`;
    a COPY a `spin`. The group's end waits for each channel's `sent` (`retire`).
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
- **Linked** when it is made: each channel's table, in the plan's pages after its operand and landing areas, linked
  into the process's queue once (`mesh_queue_link`, under the attach's lock).
- **Started** (`plan_start`) at each call: its entries rebased onto the channels' positions, its pages and the caller's
  buffers placed as rows (`place_group`), each channel with entries taking the process's next call number k, its table
  written for k (`tables_start`, `table_begin`) and its `start` stored.
- **Released by use count.** A plan is started again only once it is not in flight and every program that named its
  pages has ended. Its lease holds one 32-bit word past its pages. Each program that touches the plan's communicator
  while the plan is in flight names it (`reader`, one more) and adds one to that word at its end (`program_end`,
  `metal_stamp`: one `stamp` dispatch after the program's other dispatches, on either encoder: the page's stamp, written
  by the GPU, with no completion handler standing in for it), so the word equals `reader` once each of those programs has ended, in whatever order the caller
  commits them and on whatever queues, and once each of its tables' `sent` and `landed` has reached its last call
  (`plan_quiet`: the bridge has done with its pages and its table, ~0 stamps included once every tuple's `posted`
  equals its `count`). Where every plan of these calls is held, a new one is made (design/pages-and-functions.md, Freeing 6 and 7: the use count goes down when the reading program's stamp lands, and
  a page is written again only after release). The `stamp` dispatch is a system-scope fence, then the atomic add, so
  the pages' stores are visible before the count the host reads with acquire (design/hardware-and-discipline.md
  section 1); a stamp that cannot be encoded fails its program's call as a value.
- A plan whose group failed is started again as any other: the failed run retired its positions (`run`), and nothing of
  the failure stays with it. A freed persistent request's slot is released between calls (`release`, at each group's
  plan and each exchange's start): once it is idle and its tables' last stamps are in, each table is
  taken out of the queue (`mesh_queue_unlink`) and the lease kept on the attach's retired list (`retire`), with the
  bridge's `walking` words read at the unlink; it is returned (`retired_return`, at the next lease taken and at a
  destroy) once each word read even or has moved, each of its tuples' `posted` equals its `count`, and each word read
  then is even or has moved again (design/transport.md section 14, "Walks"). A plan's lease, and a slot's whose kernels have ended, are retired so at
  `ncclCommDestroy`; a slot whose bound kernels have not ended is taken out of the queue and its lease kept. A lease
  the region cannot meet fails the call as a value.
- The host path takes plans as the Metal path does; its group has ended when it returns, so its plan is idle at once.

**Page lengths.** A link pairs only two nodes of one page length (`block x pgsz`): the bridge ends a pairing whose
HELLO names another (mesh-link.c, EPROTO), and a relaying bridge lands and sends whole pages of that length, so the
two ends of every channel lay the same pages, and the program's own cuts lie on this node's page
(`mesh_program_steps_cut`). No length is compared at run
time. `mesh.h` `struct mesh_entry` names its first row, its byte `offset` in that row's page and its `bytes` (layout
150): unit i of an entry is its bytes from i pages on, at most a page, read or landed at the row's address plus the
offset plus i pages, and a run's key unit carries its tuple's `bytes`, so a last unit goes at its own length.

## Persistent exchanges

`ncclMeshExchangeInit(parts, lands, type, comm, &request)` is MPI-4's `MPI_Neighbor_allgatherv_init` [MPI 4.1 section
8.8] over the adjacency the caller names (as `MPI_Dist_graph_create_adjacent` names one), every member passing the same
exchange: rank a's part is `parts[a]` elements, and it lands at each rank b with `lands[a x nranks + b]` set. Each member
compiles the whole exchange into its own legs at the init, with no message: each part is a demand of the solver (its
owner the root, the ranks that land it the leaves, its bytes down: [Routing](#routing)), and each of the part's trees
carries a piece of it, the part cut by the trees' shares in pages at the init (`mesh_cut_shares`); a start's count
fills the pieces in order, each piece at its offset and its count the start's elements within it. A member's landing legs are the pieces it receives from its parent in each tree, on the channel toward that
parent, in order of the part's owner and then the tree; its send legs the pieces it sends to its children, in the same
order: from its own part, or from the rows where its landing leg of that piece landed. Both ends of a channel compile
the same legs in the same order, so its entries pair. The library allocates the buffers, as MPI's `MPI_Alloc_mem` and
NCCL's `ncclMemAlloc` do for buffers a library registers: a slot holds this rank's part, an area for each part it
receives any piece of (each piece at its offset), and the slot's table (`struct mesh_table`, then a `struct mesh_peer`
for each part this rank lands, in rank order, then a `struct mesh_wait` for each landing leg of those parts, then a
`struct mesh_relay` for each piece it sends on from its rows), then a table of its sends and a table of its landings
for each channel its legs cross. The init makes the request's three slots in one lease with one Metal buffer over it,
links their tables into the queue and takes their rows (a Metal 4 binding's records), once: start k uses slot k mod 3,
two calls in flight and one posted ahead (design/pages-and-functions.md, Freeing 7: "rows for NFE k+1 point at other
pages until the rows for NFE k are released"). No start or binding allocates.

Receives are posted ahead. Start k posts start k+1's landings into slot k+1 mod 3, once the call that last used it
(k−2) is released: each landing at the piece's whole room, its count not yet known (`need` ~0), its key, its wait and
its peers' views, its landing table started as the process's next call. So no page of start k+1 arrives before its
landing exists; a run that comes early lands in its rows. Start k then writes its own sends into slot k mod 3, and
stores the count of each of its landings (posted at start k−1) from `counts`, unless the bridge has from the first
run's key unit; a landing of no pages it stamps done. An exchange's keys are its own: the communicator's id, the
receiving rank, and the sending rank with the request's number on the communicator (seven bits, beside the bit that
marks an exchange) and its position start x legs + leg (rdma/NCCL.md, Keys), so no group between two starts moves
them, and a start takes no position of the channels' and no record in the groups' order. `ncclMeshRequestFree` stamps
~0 the landings posted ahead for a start never made; the bridge posts no receive for a landing whose count it has not
heard of, so none is left for them.

- `ncclMeshStart(request, counts, stream, &start)` is `MPI_Start`: the next start, numbered from 1, posts into the slot
  of its number plus one modulo 3 once that slot is released, waiting outside the groups' lock for the command buffer
  of the kernels last bound to it to complete, in slices of MESH_PAUSE_NS, then for the bridge's stamps, the revoke read
  between slices (`ncclCommAbort` ends it); where the kernels bound to that slot's last start are in a command buffer not yet committed, or one that ended
  in an error, it fails as a value naming it (torch-mesh commits torch's open buffer before each start). A slot is
  released by use count: every kernel bound to it (`ncclMeshBind`, landing its
  start or publishing it) counts its end in the slot's table (`ended`, the last threadgroup of `mesh_leave`), and the slot
  is free once that count reaches the binds made of it (`bound`) and the bridge has stamped its tables (below),
  whatever kernels the caller binds or none. So no start reuses pages
  another start still holds, and a kernel writes no guard of its own (the operator, 2026-10-22: "programs overwrite
  each other in data processing by default with no extraneous lines of code written"; NCCL-style primitives "use
  explicit allocation per buffer or pageset that they operate against, rather than an attempt to make timed single
  landings"). `counts` (nranks entries, each rank's part, every member passing the same) moves fewer elements than
  declared (partitioned communication's `MPI_Pready` [MPI 4.1 chapter 4]; design/extents-and-liveness.md, b); NULL moves
  the declared counts. The start writes each send leg's entry for this start (a piece's rows in the slot, its pages from
  `counts`, its key, a leg of no pages counted at none) and starts each send table as the process's next call
  (`table_begin`). A send entry of this rank's part names the slot's `published` and one of a
  relayed part its relay record's (`struct mesh_entry` `word`, its region offset), each with `need` the start's number,
  which a kernel bound to the start stores there (`mesh_leave`). So a group or another start encoded before this
  start's publishing kernel sends nothing of the slot before it is written. A landing leg's wait reads its entry's
  `done`, and a relay's its landing leg's. The bridge sends a process's tables toward a node in the order of their
  starts, so that group's SENDs toward the node wait behind the start's until its
  publishing kernel has run: a GPU wait that needs them reaches its bound, and a host-path group
  fails as a value at its call while a start is not yet bound to the kernel that publishes it (`ncclMeshBind`
  `publishes`), as it does while the communicator's GPU groups are issued and not complete (MPI's persistent request:
  its buffer is the request's between `MPI_Start` and its completion). The start first completes any earlier group of
  the communicator with a send not yet published on a channel where it sends; it takes no record of its own in the
  groups' order, since its keys are not the channels'. The caller's kernels publish and consume it, encoded after the
  start. Where a link is lost under it, its entries read ~0; the kernels bound to it run and publish as they would
  (design/pages-and-functions.md, Link error: the callgraph does not consume the status).
- `ncclMeshBind` finds a start's slot by its number (`slot_of`, the slot of its number modulo 3 while that is its
  start). A slot's use count is zero once its bound kernels have ended (above) and each of its tables' `sent` and
  `landed` has reached the slot's last call or ~0 with every tuple's `posted` equal to its `count` (`slot_held`):
  until then the bridge may post a receive into the slot. A start that only publishes is free once its
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
  has ended, its lease returned once the bridge's walks have passed its tables. A request not freed when its
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
  communicator. Its slot owns the process's leases and heads its queue of tables. The pages a group sends
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
  communicator's and is combined from there. Each step is one entry, its first page-table row and a byte offset in it
  (`struct mesh_entry`), NCCL's displacements as they come. Each SEND is one unit, a page or a last part of one, with
  one SGE (`max_sge` is 1); a unit at an offset spans two pages of its lease, which are adjacent in the region. The bridge registers the region in extents of 2^k bytes, the least power of two at which
  the region's extents number no more than the device's `max_mr` (1 GiB on the Studios' 86 GiB under max_mr 100;
  design/transport.md, The registered region), each inside one of the 4 GiB banks no registration may cross
  (`tbt_post_recv` keeps only an address's low 32 bits: `rdma/mesh-verbs.h`), so no page crosses an extent. So:
  - every piece the bridge sends from or lands in is a lease's rows (in this node's region): the caller's own, or its
    plan's stage where the caller's memory is in no lease (below);
  - the program's own cuts lie on pages (`mesh_program_steps_cut`), and a cut NCCL's call fixes lies where it does.
  `max_sge` is 1 on every device measured (design/HETEROGENEOUS-LINKS.md section 1), so a unit is one range, named by
  one memory region. Each port registers two families: extents of 2^k bytes from the region's start, and the same
  extents shifted by half of one, each shifted one that lies within a 4 GiB bank (verbs allow overlapping
  registrations), k the least at which both number no more than the device's `max_mr`: on a Studio's 86 GiB, max_mr
  100, 43 extents of 2 GiB and 22 shifted ones, 65 regions. `max_mr_size` reads 16384000 but is not enforced (the
  bridge registered 86 one-GiB extents). A unit, at most a page, lies within one extent or within the shifted one
  across its line, and `span_of` names that one. No registration and no unit crosses a 4 GiB bank's line
  (`tbt_post_recv` keeps an address's low 32 bits): the arena lays a lease of at most a bank inside one, and a larger
  one from a bank's line (`mesh_arena_alloc`), and a call whose page would cross a bank's line inside a lease larger
  than a bank fails as a value (`ncclInvalidUsage`, `bank_crossed`) before anything is presented. A piece whose buffer lies in no lease of
  this process (memory not from `ncclMemAlloc`, a private MPS tensor) is staged through the communicator's own
  registered pages by one copy, as NCCL stages an unregistered user buffer through its own: each plan holds a stage of
  every SEND's and every copy landing's bytes (a reduce lands in the plan's pages already), taken from the region the
  first time a call of it needs one and kept with the plan; a SEND's bytes are copied into its stage just before its
  publication (the host's `memcpy` in `run_group`, a copy dispatch before the publication on the Metal path), and a
  landing's out of its stage just after its wait (`consume`). Memory from `ncclMemAlloc` and the region tensors stay
  zero copy.
- Every crossing collective but a premultiplied one reads its send buffer and writes its result where they lie (the
  direct form): a SEND of a range no step of the rank wrote reads the send buffer; the first combine of such a range
  writes send op piece (`combine3_*`, a partials merge's `combine3_merge` too; the host's `combine_from`, three
  operands); a reduce-scatter's own segment is combined in its receive buffer; a broadcast's and an all-gather's landings
  land in theirs, a non-root's too; the result's ranges no step writes are copied from the send buffer after the
  publications (the call's out-of-place result); and no scratch buffer no step writes is made. A call in place (its
  send buffer its result, or an all-gather's own segment of it) sends from, combines into and lands in that buffer; a
  combine into a range a SEND of it reads waits for that SEND's bytes to be sent ([Per-channel
  progress](#per-channel-progress)). A program's messages are not coalesced across parts (`mesh_compile`), so a step's
  range is wholly written or wholly unwritten before it and the direct form holds for every program; a send buffer that
  overlaps the result apart from in place is NCCL's undefined case and is not copied. A premultiplied call
  (`ncclRedOpCreatePreMulSum`, a floating `ncclAvg`) still makes its scaled operand in its plan's pages, as does a call
  that crosses nothing. Reproduction target (metal-microbench
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
    all-gather's `down`, an all-reduce's whole operand `up` toward every rank; a reduce's operand `up` toward its
    root, a broadcast's `down` from it.
  - A plan's all-to-all (the hosted family and `ncclMeshAlltoAllV`): a demand a block from a to b, root b, leaf a, `up`
    its bytes.
  - An exchange: a demand a rank's part, root its owner, the ranks that land it the leaves, `down` its bytes.
  - The capacities are the physical cables of the whole map, one edge a cable with its capacity (`struct mesh_uses`):
    a link of the communicator's map, a pair of members, is a weighted use of the cables along the bridges' route
    between their nodes, every cable of each hop by its share of that hop's capacities, as the bridges place a key's
    runs (design/transport.md section 14, "The wire"). A pair's load is added to the cables it uses, so a cable that several
    pairs' routes share is counted once; and a per-hop latency (`hop`, seconds: 0, the bridges' records carrying no
    latency). A step names its peer alone. The per-cable `way` of trees, steps, routes and the solver's output, and the
    point-to-point cut among a pair's cables, are deleted.
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

## Routes between unlinked ranks

The bridges relay: a tuple for a node no cable reaches goes to the next node of the map's route, whose bridge lands its
pages and sends the same pages on (design/transport.md section 14, "Relays"). So a communicator has a channel toward
every member on another node, and its map joins every two such members ([Communicators](#communicators)): the
operator, 10-21, "forwarding is REQUIRED"; "cabling is extremely dynamic from this point onwards and therefore all
topologies must be supported forever without justification by current environment state"; 2026-10-22, "streaming
pinned memory by telling a complementary progarm on another node to copy-once-arrives is so simple ... without adding
any overhead". Members still relay where the solver's routes cross a member, in the calls where every member knows the
bytes: "the collective api can contain a solver or optimizer".

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
only their buffers. This is a deviation from NCCL's point-to-point semantics, where only the two ranks of a message
call anything, argued from the hardware's interface (design/rubrics.md C1's upper pole): a member between two that no
link joins relays the message, so it must know the message before its bytes arrive. A porter passes the list its
program's sends and receives make, every member the same list. Without the list, a message is its two ranks' alone,
on their channel, and the bridges carry it whatever nodes lie between them. torch-mesh gives the list a caller passes from its
compiled program (`torch_mesh.batch_isend_irecv(ops, pairs)`); torch's own `batch_isend_irecv` gives none.

## Ranks and lists

- Two ranks on one node: no queue pair joins them, so a call that crosses between them fails as a value where it is
  planned ([Communicators](#communicators)).
- Each direction between two ranks is its own list, so a rank's k-th send to a peer pairs with the peer's k-th receive
  from it, as NCCL's do. A blocking exchange with one peer completes as one group (`ncclGroupStart`/`End`; torch's
  `batch_isend_irecv`); as two blocking sends before the receives, each waits on the other's group. The library does
  not detect unequal counts or different collectives on two ranks: no landed payload is checked.
- Communicators know nothing of one another, and share every queue pair of every link: a key names its communicator,
  and no init is refused for the links others use. The bridge sends a process's tables toward a node in the order the
  process started them, so communicators whose calls the ranks issue in the same order progress as NCCL's do on one
  stream.

## Failures

No clock ends a group's wait (the operator, 2026-10-22: "we don't introduce race conditions or use timing anywhere ...
because using timing is always slower than not using timing"). A group links nothing per call and waits for no
room (design/transport.md section 14, "The queue"): its plan's tables were linked once, a start is one store a
channel, and a table is started again only once its last call's stamps are in; a call whose every plan is still in
flight makes another plan. A host-path group
waits for its own communicator's GPU work until its command buffer completes, an error status (which the platform's
watchdog gives a command buffer that overruns) failing the group, and one not yet committed failing it at once
(`ncclInvalidUsage`). Each wait reads the revoke between looks, so a revoke ends it (design/rubrics.md M3, 10-18), and holds no lock: the groups' lock is
released for the wait and taken again after, so other communicators' groups and revokes go on (M4). A host-path group's waits end at
their counts, at a count's `~0` (its bridge's, where the link is lost or the bridge stops), or at a revoke; a GPU wait at its
count, at a count's `~0` or at its bound ([Bounded waits](#bounded-waits)), the operator's "poll bound for the node"
(10-21). A group starts its tables as it is issued. A member that ends sends nothing more: its retirement stamps its
started tuples ~0, and its bridge takes no run of a ~0 tuple and sends nothing of its own. A member's tuple whose
receiver has not yet started the call waits in the receiving lane, which carries nothing past it until it does
(design/transport.md section 14, "The index"). A peer that has not yet issued
its group, or that failed, is waited for, a host-path group's wait on it until a landing or a revoke; whether to revoke
and run again is the caller's (the operator, 2026-09-10, design/pages-and-functions.md: "recovery is something which is
only sanctioned through the mechanism of a totally feed forward nfe computation callgraph being re-run by the calling
process because they don't like the results they got"). A link lost stamps ~0 the sends with requests out on it and
the landings whose sender's route comes over it, and the link is paired again from 0; a call whose member's name
leaves the map, or whose route to it moves, is stamped at that map (design/transport.md section 14, "Lost"). A
failure is
the value of the call it happened in, and `ncclGetLastError` names it (the node of the channel); nothing of it is kept,
and no later call reads it (the operator, 2026-10-22: "the collective code does not detect mend, heal, or prevent any
kind of error. it is not stateful with respect to errors"). A host-path group that fails counts its remaining entries
done (`run`) and writes no `ready` for them, so every channel's later entries stay in step with its peer's
(design/pages-and-functions.md, Link error: "The callgraph does not consume that status"). A GPU wait that ends at
a count's `~0` or at its bound records nothing; its kernel goes on with the bytes the area holds
([Bounded waits](#bounded-waits)). `ncclCommGetAsyncError` gives a non-blocking init's state and result (the operator,
2026-10-14: "it's okay for a function call ... to fail instead of returning if the link topo changed mid function
execution"). A communicator keeps the map its init made, so a change
that leaves its links working (a link lost elsewhere, a node that joins) reaches the next communicator
(metal-microbench makes each call's communicators at the call's start). A rank that departed is left out with
`ncclCommShrink` ([Communicators](#communicators)), and a joining one taken in with `ncclCommGetUniqueId`
and `ncclCommGrow`, an init at the id comm's root broadcasts on comm, registering as every init does. `ncclCommRevoke` is NCCL's, from any thread and under the communicator's own lock
alone: its operations in flight are revoked and its resources kept (its tables stay linked, and its bridge sends what
its groups published and nothing else); an init
still running (a non-blocking one's) ends as a value, its host-path groups return at their next look, its GPU waits
end at their landings or their bounds (no kernel stops at the revoke), and every later group on it is refused (`ncclInvalidUsage`, "revoked") until `ncclCommDestroy` takes its
tables out of the queue; `ncclCommFinalize` after it returns `ncclSuccess`, and its `revokeFlags` is not read. `ncclCommDestroy` awaits the command buffer of the last GPU program the library committed for the communicator until its
status is Completed or Error (the platform's watchdog ends one that overruns), with no clock and the groups' lock
released meanwhile; one not committed is not awaited; work encoded in a caller's encoder (torch-mesh's) is the caller's to finish first, as NCCL's
contract has it. While a revoked communicator's tables are linked, its bridge may still land a run into a lease its
groups named, for tuples already started, until `ncclCommDestroy` takes them out; the library returns its own such
leases (plans, slots) once the bridge's walks have passed them and their tuples' `posted` equals their `count`
([Plans](#plans); design/transport.md section 14, "Rows"), and a caller's lease that a lost group named is held by
the same words.

A communicator's bytes of its node's region are its plans' and its exchanges' leases ([Plans](#plans),
[Persistent exchanges](#persistent-exchanges)). The process
holds one of its bridge's client slots (`mesh.h` `MESH_CLIENTS`) whatever its communicators; each communicator of two
ranks or more holds its name table (one page) from its init to its destruction, and a channel is a peer's two lists:
counts in the communicator, entries in its tables. A node's records carry at most `MESH_NAMES` registrations.

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
  node, its `cable` (both bridges compute it alike from the link's lane 0) and the cable's bandwidth in bits a
  second, read from this node's own record in the map; their count (0: the node alone), or -errno where the bridge is
  unreachable. A
  node is all a client names of the mesh (design/rubrics.md, Three kinds of source). `mesh_region(&bytes, &unleased)`:
  the region's arena and the bytes no lease holds. `mesh_nodes_health`: every node's record the bridge holds (its
  bridge's node, its region's bytes, the node's memory and `hw.model`, its boot and the memory available as its bridge
  last sampled it).
- **A link map** (`struct mesh_link_map`), an operand: `nodes` nodes and an edge list, one edge a cable (`struct
  mesh_edge {a, b, cable, capacity}`, the capacity in bytes a second), built from the bridges' records of each cable
  (`mesh_map`, a communicator's `map_of`: each cable's bandwidth the lesser of its two ends', over 8). The capacity is a
  hardware fact: the solver's capacity of a pair is its parallel cables' summed, and `mesh_routes` takes of routes of
  equally few edges the one over the faster hops. No capacity is defaulted: a cable whose port reports a width or speed code
  verbs does not define, or XDR's, which Apple's 8-bit `active_speed` cannot hold, is an edge of capacity 0. It costs
  infinity in the routes (its hop's inverse, 10^15 / 0) and in the solver (its length), so routes of equally few hops
  avoid it, a hop's share of it is none, and nothing refuses it. There are no map
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
  hops' inverse bandwidths in whole fs a byte (Dijkstra [1959] in its dense form, every finite length an exact integer), each
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
- **The solver** (`mesh_solve(map, uses, eps, hop, demands, count, room, demand, parent, share)`, `mesh_solve_room`; `uses`
  NULL where each link is a cable of its own):
  [Routing](#routing). Its output holds `mesh_solve_room` trees, which no solve passes: each of its T rounds adds at
  most one tree a demand. `mesh_flow_routing` makes an all-to-all's routing table from its demands' trees (each a path
  from its leaf to its root); `mesh_cut_shares(elements, element_bytes, page, share, count, first, length)` cuts
  elements by shares in quanta of the least multiple of the page and an element.
- **Programs** (`struct mesh_program`, as MSCCL-IR is the program the MSCCL runtime interprets [Cowan et al., ASPLOS
  2023]): every rank's moves, its operand cut into parts, part p within segment `segment[p]` at its log weight.
  `mesh_compile(trees, what, root, ...)` (room from `mesh_compile_room`): a reduce up the root's trees, a broadcast
  down them, a reduce-scatter's segment s up s's trees and an all-gather's down them; an all-reduce single-phase (the
  whole operand reduced toward every node up its heaviest tree, a forwarding node's partial from scratch of its own; at
  two nodes the direct exchange, whose one SEND and one REDUCE meet in the operand, the combine after the SEND's bytes
  are sent [Trivance 2026]). A round's SENDs come before its receives; each part's message is its own (no message
  coalesces adjoining parts, so a step's range has one history). The two-phase all-reduce, chunked trees and the
  contributors set had no caller once the forms operand went, and were deleted with it.
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
| `NCCL_COMM_ID` | `mesh_mpi.py` | the text its world's id is made of (the library reads no environment for an id) |
| `MESH_RDMA` | torch-mesh's build | the rdma directory its extension links against (default its parent; `bin/mesh-user.sh` passes the checkout's) |

## NumPy

`rdma/mesh_mpi.py` spells the host path as mpi4py spells a communicator's buffer methods, so an mpi4py program ports
by its import:

```python
from mesh_mpi import Comm, SUM, MAX     # sys.path: ~/mesh/rdma
comm = Comm()                           # RANK of WORLD_SIZE, its id the text of MASTER_ADDR:MASTER_PORT
comm.Allreduce(local, total, op=SUM)    # Allgather, Bcast, Reduce, Reduce_scatter_block, Alltoall, Send, Recv,
best = comm.allreduce(x.max(), op=MAX)  # IN_PLACE as a send buffer
comm.Allgatherv(part, [whole, counts])  # a count a rank: ncclMeshAllGatherV
comm.Reduce_scatter(full, mine, counts) # ncclMeshReduceScatterV
```

Subcommunicators are `Split(color, key)` and a Cartesian grid: `grid = comm.Create_cart([2, 2])`, `grid.Sub([False,
True])` its row, `grid.Shift(0, 1)` the neighbours, as mpi4py spells them.

`Comm()` is the world: this process is rank `RANK` of `WORLD_SIZE`, its communicator's id the text of
`NCCL_COMM_ID`, else of `MASTER_ADDR:MASTER_PORT`, alike at every rank ([Communicators](#communicators)); `COMM_WORLD` is `Comm()`, made at first use. Every call is the host path:
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
