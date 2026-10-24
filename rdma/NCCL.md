# libnccl-mesh

NCCL's C API over the mesh: Macs joined by Thunderbolt 5 RDMA, a bridge process (`mesh-flow`) on each. Code
written against NCCL's `nccl.h` compiles against `rdma/nccl.h`. Its header comment is the contract; this page is
how to run it. PyTorch programs use it through [torch-mesh](torch-mesh/README.md). NumPy programs use it through
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
or region. Launch the ranks as torchrun does (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

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
and names the file. Run that install after every OS update. A failed load keeps nothing, so the same process's next
collective loads the files the install made.

`install` installs every other image first, then the kernels' files and `libnccl-mesh.dylib` (`install-nccl`). So a
failure of the kernels leaves the bridge's images installed and the node's earlier libnccl-mesh in place. Earlier
kernels' files stay, for a process that mapped an earlier library but has not loaded its kernels yet; `make -C rdma
clean-kernels` removes them, run by hand where no such process runs.

Only a machine with the Metal toolchain compiles the AIR: an Xcode with its Metal toolchain component, named by
`DEVELOPER_DIR` (on this mesh, the M5 alone). `make -C rdma kernels` (which `install` runs) checks the committed record,
`vendor/libnccl-mesh.sha256`: the kernels' SHA-256 the AIR was compiled from, then the AIR's own. Where both match, the
committed AIR is used. Where either does not, the kernels are compiled into `vendor/`, and the two files are committed
with the change to the kernels (`nccl-mesh-kernels.h`, `mesh.metal`, or the compile line). On a machine without the
toolchain that compile fails, and `make` names the build to run: it never installs AIR of other kernels.

`ncclMeshMetalPrelude()` hands the steps' text to a caller that compiles its own kernels after them (metal-microbench,
torch-mesh's `_stream`), which compiles them in its own process. Only the library's own kernels moved to build time.

## Communicators

- `ncclCommInitRank(comm, n, id, rank)`: NCCL's, with NCCL's bootstrap: the id is the root's `host:port`
  (`NCCL_COMM_ID`, else the listener `ncclGetUniqueId` opens, on the node's fabric address where it holds one), where
  each rank's node meets the others'. The root listens on that host alone where it is an IPv6 address.
- `ncclMeshCommInitRank(comm, nranks, id, rank, paths, programs, count, small, name, config)`: the same init and
  bootstrap (`id` the root's `host:port`, which the caller hands every rank; each rank's node is its bridge's), with
  the collectives as operands; the map is its members' (below).
  - `paths`: the routing tables every collective of the structural table (below) is compiled along, each member's
    trees, checked against the communicator's map. NULL means the map's packed trees (below).
  - `programs`: the compiled collectives, a table of `ncclMeshProgram` entries. NULL means the structural table
    (below).
  - `small`: not read.
  - `name`: the communicator's, which its failures and its splits' names carry (NULL: the id).
  - `config`: NULL, or `blocking` not 0, is a blocking init; `blocking` 0 is NCCL's non-blocking init (below).
  - Every rank passes the same values. `ncclCommInitRank` is this call with no paths, programs, small, name or config,
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
  are read again (design/rubrics.md M3, C5). A phase's deadline is `MESH_PLATFORM_NS` after it began: a non-root's
  connect, then its first gather; the root's accepts, every rank's report and its first gather under one deadline made
  as its accepts begin; the second gather has its own `MESH_PLATFORM_NS`. Past a deadline the program ends, the wait and its
  deadline named on stderr (design/data-model.md section 6: the operator, 2026-10-23, "no wait may ever run longer than
  a deadline operand presented at the time the wait was constructed"); nothing waits on a peer that never comes.
- While the process holds a lease of a bridge that stopped, its next lease (`ncclRemoteError`), communicator
  (`ncclInvalidUsage`: the process's own leases, which only it can free) and `ncclCommRegister` are refused naming it,
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
  blocking init's root waits in its accept, at most its phase's deadline.
- A rank's connect to its root has the deadline `MESH_PLATFORM_NS` after its first try: an init toward a root that does
  not listen ends the program then, never retried for good.
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

Each communicator builds an immutable physical multigraph from reciprocal endpoint reports
(`nccl-mesh.c` `fabric_discover`). Every physical cable has a stable wire identity and its own
channel; parallel cables remain separate. Bootstrap acquires a mutually free stream on
every cable between members. A busy or nonreciprocal cable fails initialization instead
of silently reducing the fabric to one channel per peer. Later communicators discover
the then-current fabric.

`ncclMeshCommFabric` exposes that table. Logical collective routes aggregate parallel
capacities, and `mesh_fabric_stripe` lowers each crossing to explicit cable/range records.
Their positive weights and transport quanta determine the split before execution. A
crossing uses every parallel cable when it has enough quanta; there is no size threshold
choosing a separate single-cable executor. Multihop `ncclMeshWrite` paths name physical
cables, including their parallel alternatives. A relay forwards received pages directly
from its landing rows. See [whole-function routing](../design/whole-function.md).

The bridges' records are a map whatever reaches what: a node that still has a record may be reached by no bridge,
and the whole map (`mesh.current()` with no members, `mesh_map` over none, mesh-stat's `map`) holds it as it is.
`mesh_link_map_check` refuses only what is no map; whether a map joins every two of its nodes is
`mesh_link_map_joined`, which names the first two it leaves apart. The trees and the
collectives run only on a map that joins their members: a communicator's collective on a map that does not fails as a
value naming two members no path between members joins (`ncclInvalidUsage`).

Every member makes the map from the same gathered reports, so the members' maps are one, and every rank passes the same
paths, programs and forms; nothing is compared at run time.

Collectives are compiled programs: every rank's moves, made ahead of time (`mesh-plan.h` `struct mesh_program`, as
MSCCL-IR), the communicator's table of them. The caller passes its own table (`ncclMeshCommInitRank`'s `programs`;
`mesh_compile`, `whole` the single-phase all-reduce); with none, the communicator compiles
the structural table at its first call that needs it, along the caller's `paths` or else the map's links packed
(`mesh_trees_pack`, Garg and Konemann's maximum concurrent flow: every member's trees at once, and each root's alone
for its reduce and broadcast; every link in use, each tree's part its flow's share, so the routes end together):

- the all-reduce two-phase: the reduce-scatter and then the all-gather over every pair;
- the reduce-scatter and the all-gather, segment s up and down s's trees;
- each root's reduce and broadcast, up and down the root's own packed trees, toward its min cut.

Each is uncut: a step's piece is one entry of its holding's list, any size, which the bridge alone cuts into SENDs. A
relay sends on what it received as it lands, page by page: the SEND of a piece a COPY landed is presented with that
landing's holding's `landed` as its supplied count (`struct mesh_entry` `word` and `base`), so no count of chunks
exists (4 chunks ran the all-gather, the reduce-scatter and the all-reduce 15 to 50% slower from 256 KiB to 4 MiB on
three nodes). A program's own cuts (a two-phase all-reduce's segments, a tree's shares) lie on the region's pages
(`mesh_program_steps_cut`), so every piece starts on a page wherever its buffer does ([Memory](#memory)). A call runs
its table's first program of its collective (and root) whose size bound its operand is under. Nothing about a program
is decided where a call runs.

**All-to-all with host counts.** `ncclMeshAlltoAllV(send, sendcounts, recv, recvcounts, type, comm, stream)` is
MPI_Alltoallv with its counts on the host: rank r sends `sendcounts[q]` elements to rank q and receives
`recvcounts[q]` from it, both packed in rank order (rank r's `sendcounts[q]` is rank q's `recvcounts[r]`; its own
segment copied, the lesser of `sendcounts[rank]` and `recvcounts[rank]` elements). Its counts
known to both ends, no count crosses, and each pair's elements move as one entry of their own size, one call a group,
each segment starting on a page of its buffer (else the call fails as a value). NCCL's `ncclAlltoAll` is it with every count `count`, and `ncclGather` and
`ncclScatter` its rooted forms (every count 0 but toward or from the root), where they were each rank's sends and
receives in a group (NCCL's own lowering, `classifyCollToP2pTasks`).

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
list of `ncclMeshForm` entries, each a collective (and root, -1 every root) below a size bound, compiled ahead of the
structural table at each map read, so a call takes the first form that fits it and the default otherwise:

| field | form |
|---|---|
| `trees` | `ncclMeshTreesOwn` (the communicator's paths, else its packed trees) or `ncclMeshTreesPacked` (`mesh_trees_pack` at the entry's `eps` and `most` trees a root: Blink's packed spanning trees, every link a share) |
| `shares` | a form's cut of each segment among its root's trees (root r's k-th tree, the packing's heaviest first or the communicator's trees' order, `shares[r x most + k]`, a tree of share 0 or past `most` carrying none; NULL: the packing's flows) |
| `whole` | an all-reduce's: 1 single-phase, 0 two-phase (the reduce-scatter and then the all-gather along the same trees) |
| `below` | the bytes a call is under to take it (0: any size): the size classes |

These are the forms mesh cc8a6f8e^'s default table chose among by a price (its `sets[1]`, single-phase or not, 1 to 64
chunks, by size class); the price is gone and the caller's configuration document names the form (metal-microbench
docs/principles.md section 6: a decision is structure, a measurement the calls make, or an algorithm restored from
history). With no forms nothing changes: the structural table as above. Every rank passes the same forms at the same
point between its groups, taken at the communicator's next group; splits and shrinks keep them; a communicator over the
caller's own programs compiles them ahead of its programs. A share is 0 or more: a tree of share 0 carries none of its
root's segment, and a form whose root's trees all take 0 fails as a value where it is compiled. A form that cuts a broadcast's or an all-gather's operand (chunked) does not fuse
(both its counts 0); its cuts lie on pages. Reproduction target: simulated 0.51x the default's time at 1 GiB on six-node maps and 0.58x at 16 MiB, the ring4
16 MiB all-reduce 2.18 to 1.71 ms, a six-node full mesh's 1 GiB all-gather 158.1 to 20.5 ms (metal-microbench
output_data/library-20261015/pass2/sims), and the pair's 256 and 512 KiB calls 57.4 to 31.2-36.2 us
(pair_programs.json): never measured on more than two nodes.

**Uneven segments.** `ncclMeshAllGatherV(send, recv, counts, type, comm, stream)` and
`ncclMeshReduceScatterV(send, recv, counts, type, op, comm, stream)` are MPI_Allgatherv and MPI_Reduce_scatter: rank
r's segment is `counts[r]` elements (any of them 0), packed in rank order. Each is one program, the communicator's
all-gather or reduce-scatter, its operand cut into those segments (`mesh-plan.h` `mesh_program_steps_cut`: a part
within segment s takes its tree's share of `counts[s]` in pages: a configured form's `shares`, else equal; each
segment starts on a page of its buffer, else the call fails as a value), so a capacity-shaped division of the work (metal-microbench
docs/placement.md) is one tree program over unequal segments rather than a rooted broadcast or reduce a rank. Every
rank passes the same counts; equal counts are NCCL's own calls. Reproduction target (metal-microbench docs/measurement.md
2026-09-30): tp SP=1 at capacity-shaped parts 255-256 ms a step; per call, the SP=1 all-gather 214-229 us and
reduce-scatter 196-212 us; one uneven all-gather against the same segments as N broadcasts.

## Two paths

Both paths move a crossing's bytes the same way: zero copy. Each step of a program is one entry of its holding's send
list (the rows of the pages its bytes lie in) or landing list (the rows of the pages they land in): `mesh.h` `struct
mesh_entry` `{row, pages, word, base}`, a range of page-table rows, each row a page, and every count in pages. The
bridge sends each page as one SEND from the page its row names, and posts each receive into the page its row names; the
library never learns a lane, a frame or a queue pair (design/transport.md section 14).

- **Host path.** A NULL stream: host pointers into the region (leases), sent from and landed in in place, combined on
  the host. Each call is synchronous and returns once its landings are consumed, its SENDs perhaps not yet done
  ([Failures](#failures)). It works for any lease on any node whose
  bridge runs (NumPy, CPU region tensors). It is functional, not the measured path.
- **Metal path.** An `ncclMeshStream`: the group is encoded into a Metal command buffer, and every buffer is an
  `ncclMeshBuffer` (an `MTLBuffer` and a byte offset into it). A publication is one kernel's system fence and one store
  of the holding's `ready` (a count of pages). A landing is a kernel's bounded wait on its `landed`, then its combine,
  every threadgroup of a wide landing kernel awaiting on its own. A COPY lands in place, so its consumption is the wait alone.
  - torch-mesh runs MPS region tensors here.
  - A group can be issued and completed later (`issue`, `ncclMeshComplete`).
  - **Gates.** A Metal-path group's landing entries are gated on the stream reaching the group. Each names a word and a
    value before which the bridge posts no receive into its pages: the holding's own `ready` past the group's first
    publication on that channel (where that publication waits on no receive), else the communicator's gate word, which
    the group's first dispatch stores. So no receive lands in pages the stream's earlier work still reads or writes.
    Meanwhile the peer's SEND waits in the hardware's credit. A fused crossing's channel that only lands (a reduce's
    root, a broadcast's receiver) is gated on the gate word, which the call stores: the caller's kernel that reads the
    previous landing in the same buffer is encoded before the call. A channel that also publishes is gated on its
    publication, which the caller's publishing kernel stores from its last threadgroup, so there the reading kernel may
    be the publishing kernel itself, encoded after the call (torch_mesh/fused.py's step kernel).
  - **Group end.** Publication is not send completion. There is no `sent` wait; the
    data program retains source and relay buffers until their downstream consumers
    finish, and establishes reuse through its dependencies (data-model.md section 6).
  - **Metal 4.** A stream whose `argumentTable` is set encodes into the caller's `MTL4ComputeCommandEncoder`
    (`commandEncoder`) between the commands before and after it, every binding through that argument table (at least
    six buffer bindings; each dispatch's arguments a record of the library's ring, never `setBytes`), on pipelines of
    the caller's device, the library's buffers (its holdings' counts, its pool's leases) added to the caller's
    `residencySet`; nothing is committed or signalled. Every collective and group takes the stream. Its waits are
    bounded by the stream's `polls` as the classic form's are, 0, or more than `NCCL_MESH_POLLS`, being
    `NCCL_MESH_POLLS`, the waits a group encodes at its call bounded by the greatest of its calls', those
    `ncclMeshComplete` encodes by its stream's ([Bounded waits](#bounded-waits)). A recorded program's
    replay calls it at each crossing it marks (metal-microbench docs/kernels.md#any-program).
  - The calls of one group name one command buffer or encoder: the group's dispatches are one program. A classic
    stream with an encoder and no `workspace` takes the communicator's scratch for a call that crosses no link, reused
    in the encoder's order, as the Metal 4 form does.
  - A group on several communicators is a group of each communicator's calls, in the order the group first names them,
    progressing together: each is issued, and where the stream asks no issue all are completed through the greatest
    ticket on the last one's stream, their channels interleaved as tracks (a stream that asks issue gets that ticket);
    on the host path each runs on a thread of its own.
    Every one is launched whatever another's result; the first failure is returned.
  - **Fused crossings.** A group whose steps that cross are at most one SEND and one landing in place toward each peer,
    each of a whole buffer (`ncclMeshFused`'s records, a publication and a landing a peer), runs its steps in the
    caller's own kernels. That is `ncclSend` and `ncclRecv` toward each peer in one group (an all-reduce's or a
    reduce's one-shot form on n ranks: each rank sends its part to every peer and receives every peer's into a landing
    buffer of its own, and the caller sums what lands with its own part), any of them alone, an in-place broadcast, or
    an in-place all-gather. On a stream whose `fused` is set, the call writes the steps' argument records into the
    caller's lists and returns their bindings (`ncclMeshFused`: room given, counts written, the records in its peers'
    order, its channels'); it dispatches no step. It
    completes at once, issued or not, the groups in flight before it completed first, each of its channels in the
    stream's order. A declared group's entries (`ncclMeshCommPlan`'s `ncclMeshPlan`) give the counts: an all-reduce's n - 1 of each, a reduce's root n - 1
    landings and every other rank one publication, a broadcast's and an all-gather's this rank's moves where the
    communicator's program moves whole operands (a forwarding rank, a chain's middle or a broadcast tree's inner rank,
    does not fuse), each where this rank has a channel to every peer the form uses (the communicator's map); else 0 and
    0, and the caller issues the unfused call. A group on a fused stream that is no fused crossing is not refused: it
    runs as an ordinary group on that stream, its steps the library's, and sets both counts to 0, so the caller's
    kernels publish and await nothing.
    - The caller's kernel writes its send buffer, and its last threadgroup to arrive calls `mesh_step_release` on each
      publication: one fence and one store of that holding's `ready`.
    - The kernel after it calls `mesh_step_await` on each landing (the landing, and its own SEND to that peer done) and
      reads each landing buffer with `mesh_step_read`.
    - Any type, any size; the send buffer and every landing buffer leases, each from a page.
    - Either form: a Metal 4 kernel binds them through its argument table; a classic one takes them as one
      `mesh_step_binding` record, and its stream names a `residencySet`, which the library fills with the buffers the
      bindings name and the caller makes resident for its queue (torch-mesh's `torch_mesh/fused.py` on torch's
      stream).
  - A wait polls a holding's count and nothing else: each poll one `volatile coherent(system)` load after a system-scope
    fence (`mesh.metal` `mesh_wait_bounded`; without it a running kernel saw the host's store milliseconds late:
    metal-microbench tools/poll_probe.m, 2026-10-21), and one more fence after the observed count before the landed
    bytes are read (`mesh_step_await`), each byte loaded coherent (`mesh_payload`). The bridge sets the counts `~0`
    where the link is lost; the stream's `polls` bounds the wait ([Bounded waits](#bounded-waits)). No kernel stops at a
    stop word, and no kernel halts on a failure; a wait ends early only once another wait on the same failure words has
    failed ([Bounded waits](#bounded-waits), [Failures](#failures)).

### What each program owns

The crossing path's mechanisms, each judged by the contract of the program that owns it (metal-microbench
output_data/transport-buffers-20261021/LIBRARY.md ranks the changes of 2026-10-21):

- **The bridge** (design/transport.md): its devices, queue pairs, lanes, the cut of entries into SENDs, the receives it
  posts, the region's registration. It reads `ready` and each landing's gate word, and writes `sent` and `landed`.
- **The library**, by NCCL's contract: the caller touches no buffer of a collective in flight and issues the same
  collectives in the same order on every rank; the library delivers correct bytes. So the order of every read and
  write of a group is known from the programs every rank compiled, and the library encodes it in the stream's order.
  It names its own pages and four counts, credits nothing (TN3205's credit holds a SEND until its receive is posted),
  publishes with one fence, waits on required `landed` dependencies, and bounds each wait by a
  count of polls, since a GPU spin with no end halts the node.
- **The data function**: its own buffers and their order.

### One track

A completion whose groups all lie on one physical channel of one communicator is plain dispatches in the GPU command stream, and the stream's own order is the dependency
(`nccl-mesh.c` `complete_track`). Whether a group lies on one channel is read from its schedule where the group is
planned (`struct inflight` `lone`), the entries both ends lay alike; nothing at run time chooses the form. A fused
group ([Fused crossings](#two-paths)) on any number of channels completes each of them this way in turn, in the
stream's order, its steps records the caller's kernels run and no dispatch (`nccl-mesh.c` `complete`).

- The next send entries are published as soon as their steps' receives before them are consumed: a run of them is one
  `publish` dispatch, one store of `ready` past the last.
- Each consumption is its own dispatch after that. A combine is one wide `land_*`: every threadgroup's thread 0 waits on
  `landed`, then the threadgroup combines its share. A COPY is one `spin`. A first combine of a call reading its send
  buffer in place is a `spin`, then `combine3_*`.
- The in-place form reads the operand it combines into. Publication alone does not
  establish that a downstream reader has finished. Buffer ownership belongs to the
  data program; the executor does not insert `sent` waits to repair an aliasing schedule.
- No track is recorded, no thread picks and no `tracks` dispatch runs: on one track the stream's order is the one
  serial order every rank builds alike, and a wait on it holds no other channel.

### Per-channel progress

A completion on two or more tracks runs every channel on its own (`nccl-mesh-metal.h` `metal_record`, `metal_tracks`).
This is NCCL's per-peer progress, chosen by the operator on 2026-10-20 in place of one serial order of the whole group.

- **Tracks.** Each channel's remaining entries are a track with two cursors: its publications (send entries) and its
  consumptions (landing entries), each in list order.
  - Nothing credits a publication.
  - A consumption runs once its landing came, and it is combined where it belongs; where it combines into bytes a SEND
    of its channel reads, it runs once that publication has run (`need`). This is publication ordering, not DMA completion.
  - A step depends on every earlier receive whose buffer range overlaps its range.
    Disjoint receives on different cables need no dependency edge.
  - Neither cursor waits on the other beyond these.
- **On the GPU.** The tracks are one dispatch, one thread group per track. Thread 0 picks the next item whose
  conditions hold, every thread runs it through the same kernel bodies the single dispatches use, and a consumption
  raises its track's count. A group that can pick nothing for `polls` passes records its blocked items' failures, as
  a wait does, and ends; a track reads no other track's failure.
- **What ranks share.** No rank orders anything another rank must share. The two ends of each list agree on its
  entries (`mesh_group_lay`) and nothing else, so groups of any calls among any number of ranks complete wherever
  their steps' dependencies do.
- **Not completion.** Issuing a group (`issue`) encodes no wait and is unchanged.
- **Residency.** Thread groups of one dispatch wait on each other only through a step's receive before it. Metal does
  not promise they are all resident; a wait on one that is not reaches its bound and fails as a value.

### Bounded waits

Every GPU wait is bounded (design/transport.md; the operator, 2026-10-18: "you cannot be using a low level
kernel like a dma or networking send/rcv channel for more than 5 seconds without something happening/concluding").
`ncclMeshStream.polls` is the most loads of a count one wait of the group makes (`rdma/mesh.metal`
`mesh_wait_bounded`): the caller's bound for the group's crossings, carried per crossing. 0, or more than
`NCCL_MESH_POLLS` (nccl.h), is `NCCL_MESH_POLLS`, in either form: torch-mesh's waits and the engine's encoded crossings
pass 0. The bound is the wait's deadline (design/data-model.md section 6; a kernel has no clock, so its deadline is a
count of loads). A wait that reaches its bound records its entry, its channel and the bound in its communicator's
failure words, and the program ends where the host first reads them, after the command buffer
(`ncclCommGetAsyncError`, or the communicator's next group), naming the wait and its bound on stderr; a bound reached
after a revoke, and a lost link, fail the group as a value. No kernel halts on the record: every later dispatch runs as encoded. But once
one wait has recorded a failure, every other wait on the same failure words ends: a later one at once
(`mesh_step_failed`, read before each wait), and one under way within 2^10 loads (`mesh_wait_bounded`'s stop word, the
failure claim). So a dispatch's waits hold the GPU about one bound in all, whatever its width: a wide landing's
threadgroups run in waves where the grid passes the GPU's co-residency, and each wave's threadgroups wait on their own.
The bound is a count of polls and nothing else: it exists because a GPU spin with no end halts the node (the Mini's
watchdog reset, 2026-10-21), not to guard any buffer.
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

Types: the Metal path moves and combines the integer types, float16, float32, bfloat16 and fp8 (e4m3fn, e5m2: decoded,
combined in float32 and encoded to nearest even as the host does, overflow to NaN or infinity; metal-microbench
docs/kernels.md Float8), and moves float64 (Apple GPUs have no float64 arithmetic, so a float64 reduction there fails as
a value); the host path combines float64 too.
Ops: sum, prod, max, min, avg; premul-sum's scalar read when it is made (a host immediate) or, in device memory, at
each host-path call (a Metal-path call by it fails as a value: the kernels take the scalar when the group is encoded);
an operator reduces the datatype it was made for alone; a communicator makes as many operators as its handles hold.

## Memory

The node's bridge registers its region, its configuration's share of the node's memory (etc/bridge/README.md), on
every device when the device comes up, and never again while the device is up
(design/transport.md). The operator, 2026-10-15: "the bridge's job is to allocate essentially all of the
memory on the entire device so that there are large already 'pinned' buffers which are ready to receive surprise
transfers in whatever order they happen from whatever peers and links are operational ... ontologically it must be
the responsibility of client programs to know how much allocation they need ... the bridge is there to behave like a
system call".

- The process is one client of its node's bridge: one attach of the node's region, made at its first `ncclMemAlloc` or
  communicator. Its slot owns the process's leases and holds the streams its communicators use. A communicator leases
  a page for its gate, and a pool of leases for the pages its groups send from and land in that are not the caller's
  (a reduce-scatter's operand, a tree's scratch, a combine's landing area). A lease of the pool is held by a group until
  it retires, then reused by a later group (gated on that group's start), and returned at `ncclCommDestroy`. It goes
  back at the retire while the group's SENDs may still read it: a later group may write it before a late peer's receive
  takes the earlier bytes, and that peer then receives the later ones. That is accepted rather than a wait or a kept
  lease ([Failures](#failures)).
- `ncclMemAlloc(ptr, size)` is the lease, the system call: whole pages of the region covering `size` (one page for 0),
  already registered; a lease registers nothing. Its bookkeeping is the region's (the lease's rows are owned by the
  client's slot), so when the process exits or dies its bridge returns the pages as it returns every retired client's
  rows.
- A lease the region cannot meet is `ncclSystemError` naming the bytes asked and the pages free. While every client
  slot of the bridge is held it is `ncclSystemError` naming `mesh.h` `MESH_CLIENTS`. Where the process attaches no
  region of a bridge that runs, it names the requirement: the node's bridge runs with a configuration whose region
  holds the client's footprint (`ncclRemoteError` where the bridge stopped, else `ncclSystemError`). Nothing falls back
  to other memory.
- `ncclMemFree(ptr)` returns the lease's pages at once.
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
  holds two regions. A communicator whose
  streams a restart or a new pairing opened again reports its failure through `ncclCommGetAsyncError` before any group
  runs on it, and is made again. This is the recovery
  doctrine (design/transport.md, mandates 12 and 13, the operator, 2026-10-14): "functions can fail from any one operand
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
- A NumPy program takes them from `mesh_mpi.empty(shape, dtype)`, the role mpi4py's `MPI.Alloc_mem` plays; `mesh_mpi`'s
  own temporaries are such arrays.
- Python's lease is `mesh.lease(nbytes)`: a writable buffer for `numpy.frombuffer` or `torch.frombuffer`, returned to
  the region when it and everything over it are gone.
- torch-mesh's own CPU staging and temporaries are leases ([torch-mesh](torch-mesh/README.md)).
- `ncclMeshMemBuffer(ptr, &buffer, &offset)` is the Metal buffer over the whole lease holding `ptr` (any byte of it)
  and ptr's offset in it, made once and released by `ncclMemFree`: the buffer a tensor allocator makes MPS tensors over
  (torch-mesh's region tensors).
- **Zero copy, always.** The operator, 2026-10-21: "zero copy methods must always be used over access by copy because
  memory bandwidth and flops are too scarce on this hardware api"; "good performance cannot be opt-in ... code which
  can't perform well can fail instead of being run". The bridge sends a crossing's bytes from the pages they lie in and
  lands them in the pages they belong in: a COPY lands in its destination, and a combine lands in a landing area of the
  communicator's and is combined from there. Each step is one entry, a range of page-table rows (`struct mesh_entry`),
  so it starts on a page by construction: nothing in the transport is a byte offset. Each SEND is one whole page with
  one SGE (`max_sge` is 1). The bridge registers the region in extents of `MESH_EXTENT` (1 GiB, its own choice) inside
  the 4 GiB banks no registration may cross (`tbt_post_recv` keeps only an address's low 32 bits:
  `rdma/mesh-verbs.h`), so no page crosses an extent. So:
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
  no step writes is made. Otherwise its operand is made in place in the region: a premultiply, or a copy within the
  region. The two-rank all-reduce in place makes no copy: its program sends the operand from the operand itself, and
  buffer-reuse schedule must account for outgoing readers ([One track](#one-track)). Historical reproduction (metal-microbench
  docs/measurement.md 2026-09-30 part A, 95967e6..42a16ed): per 16.8 MB all-reduce, the library's copies of the
  operand 16.78 MB to 0; tp 271.5-272.2 ms a step. The slot copy into the rings, which that left, is gone too.
- The NIC reads a SEND's pages until it finishes. A local group's end does not establish
  that fact. Source and relay leases must remain live through their downstream uses;
  `ncclMemFree` follows the data program's last use, not merely publication.
- A host-path group waits for its own communicator's GPU groups alone and fails while one of them is issued and not
  complete; it spins on its landings, never parks, and holds no lock another communicator's groups take, so a rank
  waiting in a host-path collective (an `ncclBroadcast` from a peer) does not stop its other communicators.
- `ncclCommRegister` refuses a buffer outside the region.

## Routes between unlinked ranks

A communicator's init holds a stream only toward the members whose nodes its bridge reaches and whose bridges reach its
node, and its channels go only toward those members. A collective
whose members no path between members joins fails at its first call that needs its programs, naming two, and a call
whose program or point-to-point step moves bytes between two members not joined fails at that call
(`ncclInvalidUsage`). A node a route would cross between two members takes part as a member of the communicator. In the
communicator's own table it contributes its operand like every member (`compile_entry` compiles with no
`contributors`), so it combines its own operand into what it sends on. A program a C caller passes to
`ncclMeshCommInitRank`, compiled with `mesh-plan.h` `mesh_compile`'s `contributors`, may have it contribute nothing; it
then runs the identity: it copies its first child's partial as it comes, combines the rest, and sends the result on. No
NCCL call and no binding declares such a member, where design/data-model.md section 9 ("A relay") has every relay be
the identity at a rank.

## Rules a program keeps

- One rank a node.
- Each direction between two ranks is its own list, so a rank's k-th send to a peer pairs with the peer's k-th receive
  from it, as NCCL's do. A blocking exchange with one peer is still one group (`ncclGroupStart`/`End`; torch's
  `batch_isend_irecv`): two blocking sends before the receives would wait on each other's group. The library does not
  detect unequal counts or different collectives on two ranks: no landed payload is checked, so these are the
  caller's error.
- Every buffer a crossing reads or writes is a lease, and every piece starts on a page ([Memory](#memory)).
- Communicators know nothing of one another, and no two hold one stream: a link has up to four (`MESH_LINK_STREAMS`;
  fewer where its plan has fewer lane pairs, design/HETEROGENEOUS-LINKS.md), so that many communicators use a node
  pair at once, and one more is refused (EBUSY) until one of them is destroyed.

## Failures

Every wait of a group has a deadline, `MESH_PLATFORM_NS` from the wait's start, past which the program ends, the wait
and its deadline named on stderr: no retry and no recovery (design/data-model.md section 6; the operator, 2026-10-23:
"no wait may ever run longer than a deadline operand presented at the time the wait was constructed; the presumed
default is 5 seconds"). A group whose entries do not find room in its bridge's
lists (`MESH_STREAM_ENTRIES` presented and not taken) waits for the bridge's `taken` to advance; a host-path group
waits for its own communicator's GPU work until its command buffer completes, an error status (which the platform's
watchdog gives a command buffer that overruns) failing the group, and one not yet committed failing it at once
(`ncclInvalidUsage`). Each wait reads the revoke and its holding between
looks, so a revoke or the holding's end ends it (design/rubrics.md M3, 10-18), and holds no lock: the groups' lock is
released for the wait and taken again after, so other communicators' groups and revokes go on (M4). A host-path group's waits end at
their counts, at a count's `~0` (its bridge's, where the link is lost or the bridge stops), at its holding's end (its stream
no longer the communicator's own: a pairing lost and made again, a bridge restarted), or at a revoke; a GPU wait at its
count, at a count's `~0` or at its bound ([Bounded waits](#bounded-waits)), the operator's "poll bound for the node"
(10-21). A group presents its entries to its bridge as it is issued. A member that fails, is revoked or ends sends
nothing more: its bridge posts no page of a holding that ended, and nothing of its own. A peer that has not yet issued
its group, or that failed, is waited for, a host-path group's wait on it until a landing or a revoke: the caller decides
whether to revoke and run again (design/pages-and-functions.md: "The caller decides whether to rerun the complete
feed-forward NFE"). A holding given with units still outstanding takes its stream out of its link's pairing, and a
link none of whose streams is live is lost and paired again from 0 (design/transport.md section 14). The failure is
returned:

- by the group, or
- by `ncclCommGetAsyncError` for work already enqueued, and for a communicator whose bridge stopped or one of whose
  streams is no longer its own (its link lost and paired again opens every stream new: design/transport.md section 14),
  before any group runs on it,

and `ncclGetLastError` names what failed: the node of the channel where a landing failed. A link lost under a group
fails the group (nothing resumes it), and a member whose group failed sends nothing more: the failure stays with the
member that failed, and its peers' groups end at their caller's revoke. A communicator whose group failed fails its later groups: the caller makes it again (`ncclCommDestroy`, then an
init) on the map as it is then (the operator, 2026-10-14: "it's okay for a function call ... to fail instead of
returning if the link topo changed mid function execution"). A communicator keeps the map its init made, so a change
that leaves its links working (a link lost elsewhere, a node that joins) reaches the next communicator
(metal-microbench makes each call's communicators at the call's start). A rank that departed is left out with
`ncclCommShrink` ([Communicators](#communicators)), and a joining one taken in with `ncclCommGetUniqueId`
and `ncclCommGrow`, an init at the id comm's root broadcasts on comm, holding streams of its own as every init does. `ncclCommRevoke` is NCCL's, from any thread and under the communicator's own lock
alone: its operations in flight are revoked and its resources kept (its streams stay held, and its bridge sends what
its groups published and nothing else); an init
still running (a non-blocking one's) ends as a value, its host-path groups return at their next look, its GPU waits
end at their landings or their bounds, or once another wait on the communicator's failure words has failed (no kernel
stops at the revoke; a wait that reaches its bound after it names it as its cause), and every later group on it is refused (`ncclInvalidUsage`, "revoked") until `ncclCommDestroy` gives its
streams back; `ncclCommFinalize` after it returns `ncclSuccess`, and its `revokeFlags` is not read. `ncclCommDestroy` awaits the command buffer of the last GPU program the library committed for the communicator until its
status is Completed or Error (the platform's watchdog ends one that overruns), within its deadline,
`MESH_PLATFORM_NS`, past which the program ends, the groups' lock released meanwhile; one not committed is not awaited; work encoded in a caller's encoder (torch-mesh's) is the caller's to finish first, as NCCL's
contract has it: a kernel of it still queued when its streams are given back would write a stream the next holder
may take. Likewise a lease a group of a revoked or failed communicator named is the caller's to keep until
`ncclCommDestroy`: the communicator still holds its streams, its bridge may still post receives into that lease for
entries already presented, and only the give ends that (design/transport.md section 14, "A holding's end").

A communicator's bytes of its node's region are its gate's page and its pool's leases ([Memory](#memory)). The process
holds one of its bridge's client slots (`mesh.h` `MESH_CLIENTS`) whatever its communicators; each communicator of two
ranks or more holds a stream toward each member's node it is joined to from its init to its destruction, one
communicator a stream (a holding's lists and counts are its holder's alone), of the up to four a link has.

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

## Not implemented

ncclCommInitRankScalable, ncclCommSuspend/Resume,
ncclCommMemStats, the window and signal functions (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd,
ncclSetEncryption, the ncclParam functions (`nccl.h`).
