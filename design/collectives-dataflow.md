# Collectives data flow (libnccl-mesh)

What happens when a rank calls a collective: which thread, process and GPU does the work, and which words and
buffers it reads and writes. It describes the code at `MESH_VERSION` 125 and metal-microbench's call delivery
(metal-microbench docs/elastic.md#call-delivery). Citations are `file:function`; a bare function name is in
`rdma/nccl-mesh.c`.

- Usage, build, rules a program keeps: [rdma/NCCL.md](../rdma/NCCL.md). The API contract: `rdma/nccl.h`'s header
  comment. Neither is repeated here.
- Everything inside the bridge process (taking a program onto its lanes, posting a released SEND cell, writing a
  landing's completion word, cancelling words, the five-second conclusions, retiring a dead process's slot):
  `design/bridge-dataflow.md`. This page describes only the library's side of the region and its words.

## 1. Layers and threads

`libnccl-mesh.dylib` is `nccl-mesh.c` + `nccl-mesh-metal.m` + `mesh-metal.m` (rdma/Makefile, target
`libnccl-mesh.dylib`), linked against `libmesh.dylib` (`mesh-dataflow.c`: attach, slots, map, leases; `mesh-call.c`:
sections, transfer binding, start, wake, the lanes' leases; `mesh-collective.c`: trees, compile, wire, host
publish/arrive). The GPU code is `rdma/mesh.metal` (embedded as `.build/mesh_metal.h`) followed by `nccl-mesh-metal.m`'s
`STEPS` and `KERNELS` text, compiled once per device at first use (`nccl-mesh-metal.m:kernels_of`, `MTLMathModeSafe`).

```
 ┌──────────────────────────── client process: one rank (rank r of comm, on node N) ──────────────────────────────┐
 │ caller thread (torch: its MPS queue, via dispatch_sync; metal-microbench: the window's encoding thread)        │
 │   ncclAllReduce ─► enqueue ─► ncclGroupEnd ─┬─ stream NULL ──────► launch ─► session_run ─► run() (host memcpy,│
 │                                             │                                          combine_into, spins)    │
 │                                             └─ ncclMeshStream ───► launch_metal ─► walk() ─► metal_* (encode)  │
 │   session_lock (one, process-wide) held for the whole group; released only inside session_open's start+agree   │
 │                                                                                                                │
 │ init thread, one per non-blocking init (ncclMeshCommInitRank with blocking 0: init_run ─► comm_init)           │
 │ progress thread, one per open session (session_open: pthread_create ─► progress_run)                           │
 │   scans completion words [signaled, target); ~0, its bridge gone or a destination refused → release_all        │
 │   (cancel_all: every word of the session ~0); reads violation words (violated); sleeps when nothing is pending │
 │                                                                                                                │
 │ GPU: the caller's MTLCommandBuffer / compute encoder, or MTL4ComputeCommandEncoder                             │
 │   library kernels (copy_*, send_small, land_*, spin, publish, publish_filler, combine_*, premul_*, truncdiv_*) │
 │   or the caller's own kernels calling mesh_step_* (a fused crossing); every wait bounded (NCCL_MESH_POLLS)     │
 └─────────────┬───────────────────────────────────────────────────────────────────────────────▲──────────────────┘
               │ shared memory: region "/mesh0" (shm_open, mmap MAP_SHARED; mesh_open)         │
               │ ready words, slots, lane leases ──►    ◄── completion words (1+cycle, or ~0)  │
 ┌─────────────▼───────────────────────────────────────────────────────────────────────────────┴──────────────────┐
 │ node N's bridge, mesh-flow (design/bridge-dataflow.md): takes the session's program onto its leased lanes,     │
 │ posts each released slot over TB5 RDMA, writes the receiver's completion word on landing, cancels, concludes   │
 │ what is silent five seconds, retires dead clients' slots                                                        │
 └──────────────────────────────────────────── RDMA ──► peer node's bridge ──► peer rank's ring slot + word ──────┘
```

| Thread | Made by | Runs | Locks |
|---|---|---|---|
| Caller | the program | every API call; a blocking init's bootstrap; map reads; a host group's `run()` spin; a Metal group's encode | `session_lock` for each group (`launch`, `launch_metal`, `ncclMeshComplete`, `ncclMeshFusableRooted`, `ncclCommDestroy`, `ncclCommGetAsyncError`); `attach_lock` for leases and attach; `comm->guard` for opening, revoke and the bootstrap's sockets |
| Init thread | `ncclMeshCommInitRank` with `config->blocking == 0` (`init_run`), joined by `ncclCommDestroy` | `comm_init`: NCCL's bootstrap and the lanes' lease | `comm->guard` for its sockets |
| Progress thread | `session_open`, joined by `session_close` | `progress_run`: the session's watcher; it is on no landing's path (the GPU and the host wait on the words themselves) | `s->lock` for its waits only |
| Advisor thread | `comm_advise` at an init or a split, detached | `mesh_program_advise` over the communicator's table, warnings only | none |
| GPU | the caller's commit | kernels the library encoded, or the caller's kernels running the library's steps | none |
| `atexit` | `session_open` registers `session_exit` once | closes every session | `session_lock` |

Client-side threads that drive the library: metal-microbench's rank (`mesh_rank.m`): its calls' threads and
`MetalReplaySubmit`'s completion handlers (Metal's threads; they touch no library state). A member's group thread
(`MeshRankJoin`) waits in a host-path `ncclBroadcast` from the head on its two-rank communicator for as long as no call
comes, and the head's (`MeshRankCall`) in the end-of-call `ncclGather`: `session_run` releases `session_lock` around `run`, which parks on its session's `landed` until the progress
thread's scan finds a landing (`landed_tell`, also on a failure or a revoke), and a host-path launch waits for, and is
refused beside, its own session's GPU groups alone (`session_issued`, `drain`); process exit leaves a session a
host-path group waits in (`runs`) to the exit.
torch-mesh calls the library on torch's MPS stream queue inside `dispatch_sync` (`torch_mesh/_stream.mm:on_stream`,
`begin`, `complete`).

**What the library reads and writes in the region:**

- **Header** (`mesh.h` `struct hdr`). The library reads `bridge_pid` and `boot` (`attach_live`, `bridge_gone`),
  `node` (`session_open` checks it is the rank's member), `pgsz` and `block` (the wire), `data_off` and `length`
  (`ncclCommRegister`'s window bounds), and the map serial `map` (`mesh_map_wait`). It reads the lane lease words
  (`mesh_lanes_free`) and claims and releases its own (`mesh_lanes_claim`, `mesh_lanes_release`). It rings `control`
  (`mesh_transfers_start`, `mesh_retire`) and `bell[p]` (`mesh_transfers_wake`).
- **The process's client slot** (`mesh_attach`, key 0). Its rows own the leases (`ncclMemAlloc`).
- **One program slot per communicator** of two ranks or more (`mesh_program_attach`, key = FNV-1a of the
  communicator's name, `mesh-dataflow.c:mesh_key`), taken at its init and moved into its session as that opens; its
  owner word is the owner of the lane leases the init claimed.
  - The library writes: `owner`, `configured`, `depth`, `cyclic`, `digest`, `tx[d]` (the transfer lists, cells, cancel
    offset and `lane`), `targets`.
  - It reads: `status[d]` (`prepared`, `phase`, `code`, `failed`) and `ports`.
- **The session's own leases** (`mesh_section_create`, whole blocks of the window): four rings toward each peer, the
  SEND cells and targets, the completion words, the cancellation ranges and a violation page.

## 2. Communicator lifecycle

**The init (`ncclMeshCommInitRank` → `comm_init`), NCCL's bootstrap** (NVIDIA/nccl src/bootstrap.cc). Every init is
this path: `ncclCommInitRank` and `ncclCommInitRankConfig` are it with no paths, programs, small or name.

- The id is the root's `host:port`, given by `NCCL_COMM_ID`. Without it, `ncclGetUniqueId` opens a listener on an
  ephemeral port of the first non-loopback IPv6 address with prefix `MESH_ULA_PREFIX` (`fd6d:6573:68:`, the node's
  fabric identity), else of `gethostname`. The root listens on the id's host alone where it is an IPv6 address
  (`bootstrap_listen`, NCCL's `bootstrapCreateRoot`).
- `bootstrap_open`: a non-root connects without blocking, looked at every 100 ms and tried again 100 ms after a
  failure, reading the revoke at each look; the root accepts each rank (polled every 100 ms for a non-blocking init,
  else a blocking accept) and reads its first report. The connect and the accept hold nothing of the bridge's.
- The rank then attaches the process's client (`comm_attach`; it fails like `ncclMemAlloc`, section 6) and reports
  `{rank, status, node, the lane pairs free toward each peer}` (`mesh_lanes_free`).
- **The first gather**: every report to the root and the whole table to every rank (`gather`). No deadline: it carries
  no DMA and enters no driver, and a revoke or the caller's process ends it. `members[r]` is rank r's node.
- **The second gather**: each rank takes its session's program slot under the communicator's name (`comm_slot`),
  applies `mesh_lanes_choose` to the reports and claims its own row (`lease_claim`: a CAS of each lease word); each
  send and receive is bounded by `MESH_PLATFORM_NS` (`bootstrap_bound`, design/recovery.md#five-seconds), since the
  claims are words other clients read. Every rank's status is gathered, and any rank's failure (a slot refused, a
  link with no pair free at both ends `ENOSPC`, a pair taken first `EAGAIN`, a peer silent past the bound
  `ETIMEDOUT`, a revoke) fails the init on every rank as a value, its claims released (`comm_leave`) and its sockets
  closed.
- Made: the leased pairs kept (`lease_keep`: `leased`, `chosen`), the bootstrap's sockets kept for the communicator's
  life (`peers`, each bounded by `MESH_PLATFORM_NS`; a revoke shuts them down), and the advisor started. No map is read and no session opens.
- A lone rank makes no bootstrap and takes no slot.
- The name is the caller's, else the id, and enters the session's digest (`map_digest`), so ranks of two communicators
  that one bootstrap port paired fail the agreement as a value.

**The non-blocking init** (`config->blocking == 0`, NCCL's): `*comm` is set at once, `comm_init` runs on its own thread
(`init_run`), and `async` is `ncclInProgress` until it ends, then its result (`ncclCommGetAsyncError`). `ncclCommRevoke`
sets `revoked` and shuts every bootstrap socket down, so the connect, the accept and the gathers end and the init fails
as a value with its claims released; `ncclCommDestroy` and `ncclCommAbort` revoke, join the init's thread, then free.

**The other constructors:**

- `ncclCommSplit`: a host-path `ncclAllGather` on the parent of each rank's `(color, key)` and lane report; the
  child (`comm_sub`, named `"<parent>/<split>.<color>"`) takes its slot, chooses and claims its lanes, and a second
  all-gather confirms every claim (NCCL's `bootstrapSplit`).
- `ncclCommShrink`: no exchange, so no lanes: a shrink to two ranks or more is refused (`ncclInvalidUsage`, ENOLCK);
  `NCCL_SHRINK_ABORT` revokes the parent first.
- `ncclCommGrow`: rank 0 broadcasts the new id on the parent (host path), then every rank runs `ncclCommInitRank`.
- `ncclCommInitAll`: `ndev` must be 1.

**Fields (`struct ncclComm`).**

| Field | Meaning | Written by |
|---|---|---|
| `members[nranks]`, `nranks`, `rank` | bridge node of each rank; this rank | `comm_init` (the first gather) |
| `name`, `id` | execution name, the session's slot key on every member; the bootstrap's id | `ncclMeshCommInitRank`, `comm_sub` |
| `peers` | the bootstrap's sockets | `bootstrap_open` |
| `init`, `initing`, `async`, `why` | a non-blocking init's thread, its state and its failure's text; `async` also the last failed group's result, never cleared | `ncclMeshCommInitRank`, `init_run`, `launch_metal`, `ncclGroupEnd` |
| `attach`, `slot`, `leased`, `chosen` | the process's attach; the session's program slot (its lease words' owner); its bridge's report of its links and the pair chosen on each | `comm_attach`, `comm_slot`, `lease_keep` |
| `map`, `next`, `mark` | the members' map as last read: nodes, links, costs, routing table | `comm_read` |
| `wire`, `small` | `{slot, small, lag}` of the rings; the small class's operand | `session_open`, or `ncclMeshFusableRooted` before any session |
| `digest` | FNV-1a over everything the programs are made from and the name (`map_digest`); 0 until the first session opens | `session_open` |
| `paths` | the caller's trees, every structural program's (`ncclMeshCommInitRank`; none: `trees`) | `ncclMeshCommInitRank` |
| `trees` | the map's shortest-path trees (`mesh_trees_shortest`), where the caller gave no paths | `comm_routes` |
| `forms`, `form_count` | the configured forms (`ncclMeshForm`) with their shares, compiled ahead of the structural table; kept by splits and shrinks | `ncclMeshCommForms`, `comm_sub` |
| `entries`, `programs`, `listed` | the table of programs: the caller's (`listed`, kept for its life), else the forms' and then the structural one, compiled at each `comm_read` | `ncclMeshCommInitRank`, `comm_routes` |
| `session`, `opening` | the open session; a session between its start and its agreement | `session_open` |
| `revoked` | set by `ncclCommRevoke`, kept until destroy | `ncclCommRevoke` |
| `splits` | splits and shrinks made of it, which name theirs | `ncclCommSplit`, `ncclCommShrink` |
| `ops[64]` | operators made on it: PreMulSum's scalar, or the partials merge's row (`unit`) | `ncclRedOpCreatePreMulSum`, `ncclMeshRedOpCreatePartials` |
| `scratch`, `spent`, `merge` | the Metal 4 form's scratch when the caller gives no workspace; the partials' layout scratch | `workspace`, `partials` |

**`comm_read`** runs at every `session_open`, once in `ncclMeshFusableRooted` if no session has read the map yet, and
in the advisor on its own copy.

1. `routes_free(comm, 0)` drops the previous map, trees and (unless listed) table.
2. `map_read` calls `mesh_map(members)` (`mesh-dataflow.c`). It opens the region, reads the bridges' records under one
   serial and builds the whole map. The members become nodes 0..n-1 and every other node a relay of the map's
   routes. Each link is a cable, at the bandwidth of the slowest port at either end. The table is `mesh_routes`'.
   `map_read` retries with more room on `ENOSPC`.
3. `mesh_link_map_joined`. `ENOTCONN` fails the session with `ncclRemoteError`, naming the two members that have no
   route.
4. `comm_routes`:
   - a listed communicator does nothing more;
   - `paths` are checked against the map (`mesh_trees_check`) and are the communicator's trees; otherwise `trees` is
     `mesh_trees_shortest`;
   - each configured form (`ncclMeshCommForms`), in order, is compiled first (`compile_form`): along those trees, or,
     a form naming `ncclMeshTreesPacked`, along `mesh_trees_pack(map, the form's eps, the form's most)` made for it,
     each segment cut among its root's packed trees by the form's `shares` (root r's k-th tree, the packing's order,
     `shares[r x most + k]`), else equally: the trees' weights `mesh_compile` copies into the program's parts, which
     `mesh_program_ranges` cuts by, are the configuration's, never the packing's flows; single-phase or two-phase, at
     its chunks, below its bound, a rooted form at its root or every root;
   - then the structural table: the single-phase all-reduce, the reduce-scatter, the all-gather and each root's reduce
     and broadcast, along the trees, uncut, of any size.

**The rank's communicators (metal-microbench `mesh_rank.m`, docs/elastic.md#call-delivery).** A call's communicators
are its crossings' member sets. The driver's record names each set of two members or more once (`flows`: its nodes,
its root's `host:port`, its name, its `small`).

- **`MeshRankFlows`**, at a call's start: every set this rank's node is in begun at once, non-blocking
  (`ncclMeshCommInitRank` with `blocking` 0), entered in the call's groups under its call ranks, so a cancel reaches
  it during its bootstrap (`MeshRankCancel` revokes every communicator of the call).
- **`MeshRankGroup`**, at a crossing's first naming of a set: its communicator, waited for while
  `ncclCommGetAsyncError` gives `ncclInProgress` and the call is not cancelled (1 ms pacing); a set of one is made
  alone, with no bootstrap.
- **`MeshRankDetach`** destroys every communicator of the call. No communicator outlives its call.

**`ncclCommRevoke`** runs under `comm->guard` alone, never `session_lock`, so it reaches a member blocked in
another thread's group.

- It sets `revoked` and shuts the bootstrap's sockets down (an init in progress fails).
- A session that is opening (`comm->opening`) is retired: `session_retire` → `mesh_retire` clears the slot's `owner`,
  so the bridge ends its program and its peers' programs toward it end at their own ends.
- An open session is released: `release_all` cancels every word of the session, so every GPU and host wait ends, and
  sets `signaled = target` on every channel.
- Every later group fails at `launch`/`launch_metal`'s `revoked()` check.

**`ncclCommDestroy`** (and `ncclCommAbort`):

1. revokes the communicator;
2. joins a non-blocking init's thread;
3. takes `session_lock`, so it waits behind any group of any communicator;
4. waits until no opening is in progress (`opened` cond);
5. runs `session_close`;
6. frees the communicator: its slot, where no session took it, retired and its lanes' lease released (`slot_end`), its
   sockets closed, its hold of the attach let go.

**`session_close`**, in order:

1. Loses the groups in flight (`inflight_lose` records each as a `losses` entry for the next completion).
2. Waits on `finished` for the library-signalled GPU programs that touched this session (`last`, kept by
   `program_end`), no other session's, so a destroy under `session_lock` never waits on another communicator's work.
3. Stops and joins the progress thread.
4. Releases the ring and violation `MTLBuffer`s.
5. Destroys the transport.
6. `slot_end`: `mesh_detach` (`mesh_retire` gives the slot's rows to the bridge), then `mesh_lanes_release` gives the
   lanes back (each word its owner's to 0, or to DIRTY alone where the bridge set it).

Its caller must have completed every window that waits on the session's words. The Metal 4 form signals nothing to the
library (`program_end` with a borrowed encoder), so `session_close` cannot wait for those windows.
`mesh_rank.m:run_windows` waits for every submitted window before the detach.

**`ncclCommGetAsyncError`** returns `async` where set: `ncclInProgress` while a non-blocking init runs, its failure, or
the last failed launch. Otherwise it calls `session_async`, which takes `session_lock` (waiting behind another thread's
group), turns violation words into the session's failure (`violated`), and, where nothing is outstanding, reads each
channel's next position: a `~0` there (the peer's program ended, or its bridge cancelled the session's words, while the
session was idle) fails the session. It returns `s->failed`.

## 3. Sessions

A session is a communicator's program on its node's bridge. It opens at the communicator's first group that crosses a
link (`crosses`: elements, more than one rank, a point-to-point peer other than itself). It runs on the slot and the
lanes the init took: a session that fails closes, its slot and lanes' lease with it, and the communicator's next group
fails, so the communicator is made again (`ncclCommDestroy`, then an init).

**`session_open`** is called and returns with `session_lock` held:

1. Revoked → fail. A slot gone with an earlier session, or an attach whose bridge stopped → fail.
2. `mesh_map_wait(&serial, 0)` reads the map serial, then `comm_read`.
3. The communicator's slot becomes the session's.
4. Checks `hdr->node == members[rank]`.
5. The wire: `mesh_wire_of(pgsz x block, NCCL_BUFFSIZE or MESH_BUFFSIZE, comm->small)`. Then `digest = map_digest`,
   which covers `MESH_PLAN_VERSION`, the wire, the members, every member pair's route marks and its links' costs, the
   tree sets, the caller's paths and programs, and the communicator's name.
6. Two channels toward every other member, whatever the map. `ring_make` leases `DEPTH x slot` bytes each way.
   `ring_bind` binds them:

   | Channel | Queue | Transfer identity |
   |---|---|---|
   | large | 0 | `1 + 2 x sender_rank + 1` |
   | small | 1 | `1 + 2 x sender_rank` |

   Each class has its own queue because one queue's receives are posted invocation-major across its transfers.
7. Each destination's `tx.lane` is its lane pair (`lane_pair`): the large class's queue 0 on the pair's first lane, the
   small class's queue 1 on its second. A member toward which the init leased no pair (no cable of this node reaches
   it) fails the session before anything is handed to the bridge (`ncclInvalidUsage`, ENOLCK).
8. `mesh_transfers_prepare(1 slot, POSITIONS invocations, DEPTH)` lays out the SEND cells (each cell's request
   `wr_id = 1`, the "argument") and the targets. Then:
   - `mesh_transfers_digest(digest)`;
   - `mesh_transfers_cyclic`: invocation T is cell/word T mod 32, cycle T / 32;
   - `mesh_metal_transport_create`: the `publication` `MTLBuffer` wraps the SEND-cell lease, `mesh_host_inputs` leases
     the completion words and cancellation ranges, and `inputs` wraps the words;
   - `channel_metal` per channel: `ring_out`/`ring_in` `MTLBuffer`s, `cell` (the offset of the invocation-0 SEND cell
     in `publication`), `argument`, and `word`/`word_stride` (the in ring's last chunk's completion word,
     `mesh_metal_receive_prepare`);
   - a one-page violation lease (`violation`, wrapped as `violation_buffer`);
   - the process-wide `finished` `MTLSharedEvent`;
   - the progress thread, under the bridge the session opens under (`pid`, `boot`).
9. Sets `comm->opening = s` and **unlocks `session_lock`**.
10. `mesh_transfers_start` (`mesh-call.c`) compare-exchanges `configured`, rings `control` and returns: it waits on
    nothing (design/recovery.md#five-seconds C1). `ESRCH` (bridge gone) or `ECANCELED` (the slot retired: a revoke)
    fail it.
11. `session_agree`; the wait time is added to `session_waits` (`ncclMeshSessionWaits`).
12. Relocks, clears `opening`, broadcasts `opened`, links `s` into `sessions` and sets `comm->session`.

**The agreement (`session_agree`)** is position 0 of every channel.

- Each member writes `{digest, serial, MESH_PLAN_VERSION, node, cables}` into slot 0 of its out ring (`cables`: the
  cables its init leased its pair on toward that node) and publishes position 0 (`mesh_host_publish`); position 0 is
  the progress thread's target.
- It then polls each in ring's position-0 word (`mesh_host_arrived`, `usleep(20)` between polls) until it lands, reads
  `~0`, the session fails (the watcher's: its bridge gone, a destination refused) or the communicator is revoked. A
  peer that never starts its program leaves this end's position-0 SEND held, which the bridge fails at five seconds
  (design/recovery.md#five-seconds B4), so its words read `~0`.
- It compares version and digest, then the cable counts (W3d). `serial` and `node` only name the disagreement.
- Every channel's cursors then become 1.

**A channel (`struct channel`)** is the rings toward one peer, of one class.

- `sending` and `receiving` are the host addresses of the out and in rings; `ring_out` and `ring_in` the same as
  `MTLBuffer`s.
- Its cursors are positions on that one channel. One position axis serves both directions, so calls must pair.

| Cursor | Meaning |
|---|---|
| `end` | positions scheduled so far: the next group's first position (`schedule_group`) |
| `sent` | the next position to publish (host: published; GPU: encoded) |
| `consumed` | the next position to consume (host: combined or copied; GPU: its consumption encoded) |
| `waited` | positions whose landing an encoded GPU wait already covers |
| `checked` | positions checked against their roles (the filler protocol) |
| `target` (atomic) | how far the progress thread watches: `end` at each group's schedule (`progress_targets`) |
| `signaled` (atomic) | positions the progress thread has seen land |

**The wire (`mesh-collective.c:mesh_wire_of`, `mesh_wire_large`, `mesh_wire_pieces`).**

- `block` is `pgsz x block`, the bridge's row, and a ring's pitch: slot `t` of a ring starts at `t x
  mesh_wire_pitch(slot)`, its bytes rounded up to blocks (the transfer's `invocation_pages`), whatever the slot's bytes.
- `small` is the communicator's operand (`ncclMeshCommInitRank`'s `small`: metal-microbench's record names each set's,
  its decode crossings' largest), at least `MESH_SMALL_LEAST` (32, the agreement position 0 carries); 0 gives one
  block.
- `slot` is `ceil(NCCL_BUFFSIZE / LAG / block) x block`, at least a block.
- `lag = MESH_LAG = 8` (NCCL_STEPS).
- A piece takes the large channel if its bytes exceed `lag x small`, or its element is wider than a small slot. It is
  cut into positions of whole elements: `ceil(bytes / carried)`, `carried` its class's slot less the remainder over its
  element (`mesh_wire_pieces`; `struct message` `carried`, `piece_at`), so a combine of a position's piece combines
  whole elements: a typed element of 1 to 8 bytes on slots of whole blocks (or the pair's 3,072-byte small slot)
  carries the whole slot; a counted all-to-all's rows message is laid in its type's elements; the partials merge's
  element is its row.
- Every position is its slot's bytes on the wire: TN3205 pairs a SEND with a receive of the same 4 KiB frames, and a
  ring's receives are posted ahead at the slot's bytes, so a position's frames are the slot's (bridge-and-ipc.md,
  lanes). A small slot of 3,072 bytes (the pair's one-row decode crossing) is one frame; one block is four.
- On a 16 KiB block with the default 4 MiB and no small given: small 16 KiB, slot 512 KiB, large above 128 KiB.

**A position t on a channel:**

| Item | Where | Value |
|---|---|---|
| Out ring slot | `sending + (t % DEPTH) x slot`, `DEPTH = 2 LAG = 16` | the piece, or for an empty position `METAL_STAMP(t)` in its first 8 bytes |
| SEND cell | `publication + cell + (t % POSITIONS) x 256`, `POSITIONS = 2 DEPTH = 32` | ready word `:= argument + t / 32` (= `1 + cycle`); the bridge reads it |
| In ring slot | `receiving + (t % DEPTH) x slot` | written by RDMA |
| Completion word | `inputs + word + (t % POSITIONS) x word_stride` | written by the bridge: `1 + t / 32` once landed, `~0` once cancelled. Landed iff `word != ~0 && word >= 1 + t / 32` (`landed`) |

**Credit.** An end publishes t only once it has consumed the peer's t − LAG. The peer then has consumed this end's
t − DEPTH, the slot t reuses (`run`: `sent < consumed + LAG`; `publish`: `arrive(need = t − LAG + 1)`). Every position
is published by both ends, a filler where an end has nothing to send. The receiver checks every slot's stamp against
its own role: host in `run()`, GPU in `mesh_step_check`. A mismatch is `MISMATCH` / `ncclInvalidUsage`.

**What a session costs.**

- Window bytes: `ncclMeshSessionBytes` → `session_bytes`: for each peer, four rings and its lists; plus the SEND
  cells, completion words, cancellation ranges and a violation page.
- Slots: one of the bridge's `MESH_CLIENTS` (64) client slots per communicator, from its init.
- Lanes: one pair of each link toward each peer, from its init; four communicators a link at once.
- Threads: one progress thread.
- Time: the start (no wait) and the agreement; `ncclMeshSessionWaits` sums them.

**`session_ensure`** decides what a group runs on:

- wait for another thread's opening of it;
- open a session if there is none;
- if `s->failed` (after `violated`): return the failure and close the session. This group fails, and so does the next
  (its slot is gone): the communicator is made again;
- if no group is in flight and the attach's bridge is gone: close the session, and the reopening fails alike.

A change of the map never closes a session.

## 4. Programs

**A collective is a compiled program** (`mesh-plan.h` `struct mesh_program`): every rank's moves over parts of buffer 0
(the operand, made in place) and scratch buffers b ≥ 1.

- `mesh_program_steps(program, rank, operand)` makes a rank's steps (SEND, REDUCE, COPY; peer, round, buffer, first,
  piece).
- `mesh_program_steps_cut(program, rank, operand, segments)` makes them over unequal segments: segment s the
  `segments[s]` elements after those before it, each part its weight's share of its segment (mesh 430cf7b2^ `mesh_cut`;
  `mesh_program_ranges`: the program's weights, the caller's trees' or a configured form's shares, else equal).
  An uneven call (`ncclMeshAllGatherV`, `ncclMeshReduceScatterV`) carries its own copy of the counts (`struct call`
  `segments`, `total` their sum; `elements` and `segment_at` read them), `plan_calls` cuts its steps by them, and
  `order_group` cuts every other rank's alike, so one program serves any division; its program's segments must be the
  ranks (`ncclInvalidUsage` else). Staging (`stage`, `launch_metal`) and the result (`finish`, `epilogue`) take rank r's
  segment at `segment_at`.
- A point-to-point call is one step (`plan_calls`): a SEND, or a COPY into its receive buffer.
- Routes are data: the map's routing table in `map.next`, and the tree sets.

**Choosing a program (`program_of`):** every communicator takes the first entry of its table with the call's `what`
(and `root` for a rooted call) and `!below || bytes < below`. With no match, the call fails: `ncclInvalidUsage`, "no
program". The table is the caller's programs, or the configured forms' entries followed by the structural table
(`comm_routes`), so a call no form fits runs the default. No price, timing or cache chooses; the forms are entries a
configuration names (`ncclMeshCommForms`). `MESH_PLAN_VERSION` (5: positions carry whole elements) enters both the
digest and the agreement, so members whose planning rules differ fail at the session instead of mismatching later; the
digest covers every entry's program, so members configured with different forms fail there too.

**Places where a formula or constant, not a table or evidence, chooses:**

| Place | Choice made by |
|---|---|
| `program_of` | the table's order: the configured forms (the caller's configuration), then the structural defaults |
| `mesh_map` link bandwidths | the port's nominal speed, read by the routes (ties among routes of equally few links) and by `mesh_trees_pack` |
| `compile_form` | packing `eps` and `most`, the trees' `shares`, and every other form field, are the configuration's (`ncclMeshForm`) |
| wire (`mesh_wire_of`, `mesh_wire_large`) | `lag = 8`; the large/small split at `lag x block` (or an element wider than a small slot); slot = `NCCL_BUFFSIZE / 8` rounded to a block, carrying whole elements |
| `mesh_lanes_choose` | on each link the lowest lane pair free at both ends |
| `nccl-mesh-metal.m:dispatch` | 256 threads a pointwise kernel, at most 1,024 threads in one threadgroup for a block's piece, at most 2^20 units a grid |
| `ncclMeshFusableRooted` | fusable only at 2 ranks, one small slot, float16 sum (or a broadcast of any type), at most 1 scratch |

**Scratch.** `program->scratch` is the most scratch buffers a rank uses.

- Host path: `malloc` copies of the staged operand (`launch`).
- Metal path, each a `metal_copy` of `own` (`launch_metal`; none where the send buffer is read in place, below) into
  either:
  - the stream's `workspace` callback (classic form; required with a borrowed encoder);
  - the communicator's `scratch[k]` (Metal 4 form without a workspace; grown, never shrunk);
  - a private buffer the program keeps until it ends (classic form without an encoder).
- **The two-rank all-reduce** (the direct exchange, `whole`) has exactly one scratch: the rank's copy of its operand,
  which only its SEND reads while the REDUCE sums into buffer 0. The fused form makes no copy and aliases the scratch to
  `own`, because the caller's kernel publishes from its own store.

**Reading a lease in place** (`plan_direct`, mesh 95967e6 ported onto the session's rings). On the Metal path a
collective whose send buffer is a lease's Metal buffer (`lease_buffer`, `ncclMeshMemBuffer`; `lease_of_buffer` tells it
by the buffer) is planned to read it in place where the program allows: not issued, fused or premultiplied, the send
buffer apart from its result, and every step's range in its buffer wholly written or wholly unwritten by the rank's
steps before it (`spans_meet`, `spans_add`).

- A SEND of an unwritten range reads the send buffer (`struct steps` `read`); `publish` takes it as the piece's source.
- A REDUCE of an unwritten range is its first combine (`fresh`): `consume` encodes `arrive` and `combine3_<type>`
  (buffer = send op the landed piece) in place of `land_<type>`/`combine_<type>`.
- No stage copy of the operand and no scratch copy; a scratch buffer no step writes (`needed`) is never made.
- The result's ranges no step writes (an all-gather's own segment, a broadcast root's operand) are `complement`,
  copied from the send buffer after the walk's publications (`complement_copy`, before `retire`).
- Every read is encoded in the group's own command buffer or encoder, and the bridges read only the rings, so a lease
  is free to return once that command buffer completes; an issued group stages as before.
- The host path copies; `ncclMeshMemUse` points on a lease (`struct lease` `points`) are waited for by a host-path
  group that reads or writes it (`lease_wait`, before `session_lock`).

**A counted all-to-all** (`ncclMeshAlltoAllCounted`, mesh 3981ca80^ `run_counted` and `counted_copy` ported onto the
session's positions) is a group call of its own kind (`WHAT_COUNTED`) whose steps `counted_steps` makes, all of round 0:
to each peer q its counts segment (buffer 0, the call's counts) and then its rows (buffer 1, room for the pair's bound
`min(sendrows, capacity)` rows), from each its counts (buffer 2, the landed counts `got`, rank order) and then, after
every counts receive (`mesh_steps_before`), its rows (buffer 3, the receive buffer).

- `lay` places counted calls after the collectives and before the point-to-point pairs, so `order_group`, which knows
  no other rank's counted steps, still places every rank's collectives where they are.
- Host path: `counted_got` makes the landed counts with this rank's own segment and a tail (the rows before its own
  segment, the rows of all its counts, an overflow word); `run` copies a rows piece by `counted_place` (its place from
  the counts, its valid bytes within them); `counted_host_end` copies its own rows, writes `received`, `landed` and the
  word.
- Metal path: `counted_stage` (scratch `got`, `METAL_COUNTED_START`); `publish` writes a rows piece with
  `METAL_COUNTED_SEND` (the place summed from the counts on the GPU) and releases its cell; `consume` lands one with
  `arrive` (the session's spin on the bridge's completion word) and `METAL_COUNTED_LAND`; each counts receive consumed
  checks its pair's bound (`counted_landed`, `METAL_COUNTED_CHECK`), and the last encodes `counted_end` (own rows,
  `received` with system-coherent stores, `landed`, then `METAL_COUNTED_ARRIVED`: the word 1, or 2 and violation word
  [5], which `violated` makes the session's failure).
- A group lost or failed sets each counted word still 0 to 2 (`counted_failed`, from `launch`, `launch_metal` and
  `inflight_lose`).
- `received` and `arrived` on the Metal path are host memory in leases; `lease_buffer` gives the lease's Metal buffer,
  made at its first use and released by `ncclMemFree`.
- A pair whose counts carry no rows still takes its room's positions; its sender writes none of the slot, so every ring
  slot's first word is cleared when the session makes its rings (`ring_make`): a filler's stamp an earlier session left
  at the same position is none of this one's.

**An all-to-all with host counts** (`ncclMeshAlltoAllV`; NCCL's `ncclAlltoAll`, `ncclGather`, `ncclScatter`: `routed`)
is the counted call with no `counts` (`hosted`): its `segments` the elements to each rank and then from each, the
same on both ends of every pair, so `counted_steps` makes to each other rank q one SEND of its elements for q (buffer
0, the send buffer at q's segment) and from each one COPY (buffer 1, the receive buffer at q's segment), sized by the
counts, and every piece runs the plain paths (`run`, `publish`, `consume`); its own segment is copied (`own_segment`:
`counted_host_end` on the host, `counted_stage` on the GPU, none where it is in place). A gather's counts are 0 but
toward the root and a scatter's but from it, so their unused buffers are never read (NULL off the root).

**The partials merge** (`ncclMeshRedOpCreatePartials`) is an operator of the communicator's table (`struct redop`
`unit` = 2 + dim), as PreMulSum's is. `reduction` gives the call `combine = COMBINE_MERGE` and its `unit`, and from
there the call is its collective with a row for an element:

- `admit` refuses counts that are not whole rows; `plan_calls` makes the operand `{float32, 4 unit bytes, elements /
  unit}` and cuts an uneven call's steps by its counts over the unit (`struct steps` `cut`), so every program, cut,
  chunk and tree moves whole rows, and the wire's positions carry whole rows (`mesh_wire_pieces`).
- A piece's combine is `combine_into` / `metal_combine` / `metal_land` with type `METAL_MERGE` and the row's floats for
  the operator (`ktype`, `kop`, `kcount`): `merge_rows` on the host, `merge_row` in `combine_merge` and `land_merge` on
  the GPU, a thread a row.
- It reads no send buffer in place (`plan_direct`): the operand is staged.

`ncclMeshAllReducePartials` and `ncclMeshReducePartials` (`partials`) take the engine's `[m: rows][l: rows][O: rows x
dim]`: the piece laid as rows into a scratch (`partials_layout` on the host, `partials_rows` on the GPU, a row with no
positions written as the merge's none), the all-reduce or reduce with the merge on it (completed in the stream: its
stream copy has no issue), then laid back into `recv` (`partials_columns`). The layouts are the library's own programs
on the stream (`layout_on`: `metal_begin`/`metal_begin4`, `program_end`, under `session_lock`); the scratch is the
stream's workspace, else the communicator's `merge` (grown, the replaced one kept on `spent`).

**Roots.**

| Collective | Root's role |
|---|---|
| reduce | sums up the root's trees; only the root's result is written (`finish`/`epilogue`) |
| broadcast | runs down the root's trees |
| reduce-scatter, all-gather | segment s up or down s's trees; non-roots of a reduce and every reduce-scatter work in scratch (`launch_metal`'s `in_place`) |

### Advisories

`mesh_program_advise(program, wire, operand, warn, arg)` (mesh-plan.h, mesh-collective.c) reads a compiled program's
steps for an operand on a wire and hands each finding to `warn` as a line `check: ...`. It is advisory: it refuses
nothing and changes nothing; a communicator's init runs it on a thread of its own over its table (`comm_advise`, each
warning logged as NCCL logs one), on no call's path. `make mesh-advise` builds `.build/mesh-advise`, which runs it over
every collective `mesh_compile` makes on the loopback fabric's maps (the pair, line 4, ring 5, star 8, torus 6 and 8;
shortest-path and packed trees; chunks 1 to 4; every root; both all-reduce forms; all or the even nodes contributing;
four operands) and prints each program's findings and a summary.

| Check | What it reads | Source |
|---|---|---|
| projection | rank r's i-th SEND to p against p's i-th receive from r, piece for piece (first, elements, element type and bytes); both ends' layout of each channel (`mesh_lay`), position for position, and its length: TN3205's equal frames at both ends | session types' projection [Honda, Yoshida & Carbone 2008]; synchronous dataflow's balance [Lee & Messerschmitt 1987] |
| priorities | each rank's steps in (round, a round's SENDs first) order, and each SEND and its receive in one round: with projection clean, the program is deadlock-free by the priority argument | priorities [Dardha & Gay 2018], after [Caires & Pfenning 2010] |
| depth | an element wider than its channel's slot (a position splits it); a step whose first position waits on its own channel's receive ending at or past that position (the publication rule leaves `lag` positions of slack) | n-synchronous bounded buffers [Cohen et al. 2006] |
| causality | every rank's publications and consumptions of every channel's positions under libnccl-mesh's rule (`run`: position t published once the peer's t − lag is consumed; consumed in order once landed; a step's first position after the receive before it, `mesh_steps_before`, is consumed), a marked graph, sorted with no timing: an event that can never occur is a deadlock, reported with a cycle | an admissible schedule [Lee & Messerschmitt 1987]; Lustre's instantaneous-cycle rejection [Halbwachs et al. 1991] |
| hazard | a receive after a SEND of the same rank into elements the SEND reads, with no order in the graph from the SEND's last position to the receive's first (a write after read); a SEND after a receive of what it reads, with none from the receive's last position to the SEND's first (a read after write) | — |

The bound on a channel's outstanding positions (2 lag, the ring's slots) holds by the publication rule itself, which
the causality graph encodes; it is not a property of a program.

## 5. The life of one collective

### (a) Library form, host path: `ncclAllReduce(send, recv, n, type, op, comm, NULL)`

1. **`enqueue`** validates and appends a `struct call` to the thread-local `calls`. The implicit
   `ncclGroupStart`/`End` around it runs the group immediately when the call is outside a group. Inside
   `ncclGroupStart`/`End`, calls accumulate until the outermost `ncclGroupEnd`.
2. **`launch`**:
   - `reduction` sets `combine`, `premultiplied`, `post`;
   - revoked → fail;
   - `lease_wait` for any declared GPU work on its leases;
   - **locks `session_lock`**; `session_ensure` (it may open the session, section 3);
   - fails if any GPU group is issued and not complete (process-wide `inflight`);
   - `drain` waits for library-signalled GPU programs (section 6);
   - `plan_calls`.
3. **Staging.**
   - Each collective gets a host `result` buffer from `calloc`. `stage` copies or premultiplies `send` into it.
   - Each scratch b ≥ 1 is a `malloc` copy of it.
   - A SEND's buffer 0 is `send` itself, a receive's is `recv`.
4. **`session_run`**:
   - `schedule_group`: `lay` gives each channel's positions in call order (`mesh_lay`: round by round, SENDs and
     receives side by side), then point-to-point calls with the j-th send beside the j-th receive.
     `mesh_steps_before` gives each step's prior receive. `ch->end` advances.
   - `progress_targets`: `target = end`, waking the watcher.
   - `mesh_transfers_wake` rings the bells of the bridge's ports.
5. **`run()`** spins on the caller's thread, still holding `session_lock`, until every channel has
   `sent == consumed == end`. Per channel:
   - **Consume.** While `mesh_host_arrived(in, consumed)` has landed:
     - check the stamp against the role;
     - `combine_into` (REDUCE) or `memcpy` (COPY) from the in slot into `host[buffer] + offset`;
     - a piece's first position waits until its prior receive is `done`.
   - **Publish.** While `sent < consumed + LAG`:
     - `memcpy` the piece from `host[buffer]` into the out slot, or write the filler stamp;
     - `mesh_host_publish(out, sent)` stores `argument + cycle` into the position's SEND cell.
   - On `~0` or `s->failed`: fail, and `session_close` the session.
6. Unlock. `finish` copies `result` into `recv`, or truncates and divides for an integer `ncclAvg`.
   `target = signaled = waited = end`.
7. On failure, `ncclGroupEnd` sets `comm->async`.

### (b) Library form, Metal path (classic or Metal 4), not issued

The call is the same with an `ncclMeshStream` whose buffers are `ncclMeshBuffer`s. `launch_metal`:

1. **Checks.** One communicator, one command buffer or encoder, one `polls`; no float64. A Metal 4 stream must have
   its encoder and residency set, and a workspace if it is issued. A classic stream with an encoder must have a
   workspace. Then `reduction`, the revoke check, lock, `session_ensure`, `plan_calls`, `metal_collect(finished)`.
2. **Program.**
   - `metal_begin4` (Metal 4): the caller's encoder, argument table and residency set, the device's argument ring and
     a leading barrier.
   - `metal_begin` (classic): the caller's command buffer; the caller's encoder (borrowed) or one made lazily.
   - Either binds the session's `violation_buffer` and the stream's `polls` (0, or more than `NCCL_MESH_POLLS`:
     `NCCL_MESH_POLLS`). The walk gates each group's dispatches on its own session's words (`metal_gate`).
   - Metal 4 makes every ring, `publication` and `inputs` resident.
3. **Staging.**
   - `own` is `recv` for in-place work (all-reduce, broadcast, all-gather, the root's reduce), else a workspace
     scratch.
   - `metal_premultiply` or `metal_copy(PLAIN)` of `send` into `own` (or the rank's all-gather segment), unless they
     are the same buffer and offset, or the send buffer is read in place.
   - Each scratch b ≥ 1 is a `metal_copy` of `own`.
4. **In flight.**
   - An `inflight` group `g`: its positions `[start, end)` per channel, a fresh `ticket`, appended to the
     process-wide `inflight` list.
   - `order_group` builds every rank's layout on every channel and orders it with Kahn's algorithm (an MSCCLang-style
     topological order). Each end's publication and consumption of each position is a node. A publication's level is
     1 + its predecessors' maximum; a consumption's is that maximum. This rank's publications are sorted by `(level,
     channel, position)`, which every rank's walk keeps.
   - `progress_targets`, `mesh_transfers_wake`.
5. **`walk(program, NULL, g)`.** `complete_group` runs for every group up to `g`; for each publication in order:
   - `consume` up to t − LAG + 1, and the SEND's prior receive;
   - `publish`: `arrive(need)` encodes `spin` dispatches for positions not yet `waited`, then:

   | Position | Small channel | Large channel |
   |---|---|---|
   | publish a piece | `send_small`: one threadgroup copies the piece into the slot, `threadgroup_barrier`, thread 0 `mesh_publish`es the cell (system-scope fence, then store `1 + cycle`) | `copy_send` into the slot, then `publish` (one thread) |
   | publish nothing | `publish_filler`: stamp + release | `publish_filler` |
   | consume a piece | `land_<type>` (REDUCE) or `land_copy` (COPY): thread 0 waits on the completion word, at most the program's polls, and checks the window of positions, then barrier + system fence, then all threads combine or copy from the slot | `spin`, then `combine_<type>` or `copy_land` |
   | consume nothing | no dispatch (checked by the next wait that covers it) | no dispatch |

   After each dispatch: Metal 4 `dispatch4` encodes a barrier; the classic encoder is serial.
6. **Retire.** `retire` runs `epilogue` (results into `recv`: a copy if `sum != recv`, `truncdiv_*` for an integer
   `ncclAvg`) for every group whose positions are all consumed, and sets `retired`.
7. **End.** `program_end`: a borrowed encoder signals nothing; otherwise `encodeSignalEvent(finished, ++issued)` for
   `drain`. Metal 4 commits the residency set if a buffer was added.
8. On an encode failure, `walk_failed` fails every session through the ticket. Unlock; on failure set `comm->async`.
9. **The caller commits.** The bridge then carries each released cell (design/bridge-dataflow.md), the peer's kernels'
   waits end on their words, and the progress thread advances `signaled`. Completion is the caller's command buffer
   completing. The library learns nothing of it except through `finished` (non-borrowed encoders only).

**Issue and complete.**

- With `stream->issue`, `walk(program, g, NULL)` runs `issue_group`: only publications that need no wait
  (`need <= waited`, prior receives already consumed). `issue->ticket` is the ticket, or 0 if the group retired at
  once.
- `ncclMeshComplete(ticket, stream)`:
  1. reports losses through the ticket (`losses_through`);
  2. begins a program on the given stream;
  3. `walk(program, NULL, through)`, then `retire`.
- torch-mesh's `MeshWork::finish` → `complete` uses this; `_stream.mm:release` frees the workspaces through
  `ncclMeshRetired()`.

### (c) Fused form recorded into a Metal program (`ncclMeshFusable`, `ncclMeshFusableRooted`)

**Recording time** (metal-microbench).

- `decode_crossings.swift:StepCrossings.init` calls `MeshRankGroup` for each crossing, then `MeshRankFusable`
  (`mesh_rank.m`) → `ncclMeshFusableRooted`.
- That function:
  - requires 2 ranks, `count > 0`, root 0 or 1 (0 for an all-reduce), and a float16 sum for anything but a broadcast;
  - under `session_lock`, `comm_read`s the map if no session has (`!chosen`), and sets the wire if `wire.small == 0`
    (`attach_current` + `mesh_wire_of`);
  - `program_of`, then fusable iff `program->scratch <= (all-reduce ? 1 : 0)` and `bytes <= wire.small`;
  - returns `publishes = all-reduce || (broadcast == rooted)` and `lands = all-reduce || (reduce == rooted)`.
- The crossing's record carries `fused = publishes | lands << 1`.
- The engine picks kernels that run the steps:
  - the producer's finish publishes: `metal/layers.metal:streamed_finish_publish`, or `streamed_finish_norm_add` with
    flags bit 128;
  - the next finish lands: `streamed_finish_norm_add` with flags bit 64 (`dense_execution.swift`).
- `MeshRankCross` inside a recording only marks the crossing (`MetalRecordMark`); no library call is made.
- The engine compiles the library's step text into its prelude (`metal_runtime.swift`: `ncclMeshMetalSteps()` after
  `mesh.metal`).

**Replay time** (`mesh_rank.m:run_windows` → `window_submit` → `metal_recording.m:MetalReplaySubmit`), per invocation t
and mark k:

1. Encode the recorded commands up to the mark, minus the last one if it publishes.
2. **`cross()`** calls `issue()` → `ncclAllReduce` (or `ncclReduce`, `ncclBroadcast`) with
   `ncclMeshStream{commandEncoder, argumentTable, residencySet, fused = &steps}`, `polls 0`. In `launch_metal`:
   - It refuses a fused group that is not exactly one in-place all-reduce, broadcast or reduce on a Metal 4 stream,
     not issued. It also refuses scratch buffers other than `1 + (all-reduce at 2 ranks)`.
   - Staging: `send == recv`, so no copy; the scratch is aliased, not copied.
   - `fusing` marks the call. In the walk:
     - `publish` writes `metal_send_deferred` instead of `send_small`: `record4` writes the argument record
       (`dst = slot offset`, `n = bytes`, `aux = cell offset`, `scalar = 1 + cycle`, `sequence`) and returns its six
       GPU addresses `{d = ring_out, s = d, record, a = publication, v, progress}`;
     - `consume` writes `metal_land_deferred` instead of `land_*`:
       `{d = own buffer, s = ring_in, record (src = slot offset, aux = word offset, scalar = cycle, the check window), a = inputs, v = violation, progress}`.
   - Nothing is dispatched for the crossing except `metal_begin4`'s leading barrier, and the spins of any position not
     yet `waited` (none in steady state: the previous crossing's landing advanced `waited`).
   - A one-sided crossing's other role is still the library's. The member that does not publish (a reduce's root, a
     broadcast's non-root) still publishes its filler with a library `publish_filler` dispatch at the position. The
     member that does not land (a reduce's non-root, a broadcast's root) consumes nothing there, so no dispatch.
   - It checks that exactly the fused steps were written; otherwise `ncclInvalidUsage`.
3. Encode the publishing command (the producer kernel) with `steps.publish` bound at argument-table indices 7..12
   (`MetalFusedBinding`), then the landing command with `steps.land` at 7..12.

**GPU time, both ranks symmetric for the all-reduce, position p:**

1. The producer kernel (`streamed_finish_publish`):
   - `mesh_step_enter` (thread 0 stores the record's `sequence` into the argument ring's progress word; the kernel
     returns early if its session's violation[2] is set);
   - each element goes to the engine's output and, with `mesh_step_store`, into `ring_out + (p % 16) x small`;
   - `mesh_step_release`: device barrier, then thread 0 `mesh_publish`es SEND cell `p % 32` with `1 + p / 32`.
2. The bridge sees the ready word and posts the slot. The peer's bridge writes the peer's completion word `p % 32` to
   `1 + p / 32` (design/bridge-dataflow.md).
3. The landing kernel (`streamed_finish_norm_add`, flags 64), each threadgroup:
   - `mesh_step_enter`; `mesh_step_await`: thread 0 waits with `mesh_remote_wait_bounded` (`mesh_step_wait`, at most
     the program's `polls` loads), then `mesh_step_check` checks the window's stamps (violation [0], [1]) or records a
     reached bound ([2..4]);
   - device barrier + system-scope fence;
   - adds the peer's row from `ring_in + (p % 16) x small` (`mesh_step_read`) to its own before the norm.
   - Nothing reads the operand buffer as the sum.

The crossing therefore costs the landing's wait inside the consumer kernel and no dispatch of its own. The library
form costs `copy_plain` (scratch), `send_small` and `land_f16`. The wire behaviour (positions, slot bytes, cells,
stamps) is the same in both forms, so two members may encode the same crossing differently without disagreeing.

## 6. Failure data flow

**Who writes `~0` into a completion word:**

- the bridge, when a link is lost or concluded, a flow fails (a SEND held five seconds among them), the program is
  refused at its take, a port is concluded around a stuck call, or the bridge stops (design/bridge-dataflow.md);
- the library's `cancel_all` (`mesh.h:mesh_cancel` over every destination's range), called from `release_all`:
  - by the progress thread, on a landing that reads `~0`, on its bridge gone (`bridge_gone`) or on a destination's
    program refused or failed (`start_refused`), while positions are pending;
  - by `violated`;
  - by `ncclCommRevoke`;
  - by `walk_failed`;
  - by `session_async`, on an idle session's cancelled next position.

A cancelled word stays `~0` while the session stands, as a landing never lowers a word.

| Event | Detected by | State written | What the caller sees |
|---|---|---|---|
| Link lost or concluded, a flow failed, the peer's program refused | bridge → words `~0` and the status; then whichever reads first: `progress_run` (positions pending), `run()`, `session_agree`, a GPU wait | `s->failed = ncclRemoteError` ("the link to node N was lost", or the status's node and code) | host: the group fails. GPU: waits end, kernels proceed on stale slots, the failure after the window or at the next group |
| Bridge gone (stopped, or ended by a fault, which stores no `~0`) | `progress_run`'s `bridge_gone`; `session_ensure`'s `attach_live` | `release_all` → `~0`; or the session closed | `ncclRemoteError`; the communicator is made again |
| GPU wait reached `polls` | `mesh_step_check` → violation [2] = position + 1, [3] = channel, [4] = polls; later dispatches of the session skip (`mesh_step_enter`) | `violated` → `release_all` | async error / next group: "landed nothing within N polls" |
| Calls that do not pair | host `run()` stamp check; GPU `mesh_step_check` → violation [0], [1] | host: `ncclInvalidUsage` at once; GPU: `s->failed` via `violated` | `MISMATCH` text |
| Revoke | `ncclCommRevoke` | `revoked`; the bootstrap's sockets shut down; the opening slot retired; `release_all` | `ncclRemoteError` "revoked", every later group |
| Map disagreement, or the two ends leased different cables | `session_agree` | agreement failure; `session_close` | `ncclRemoteError` naming nodes, serials, digests or cable counts; the communicator is made again on the map then |
| A peer that never starts its program | the bridge: this end's position-0 SEND held five seconds (B4) | the flow failed, its words `~0` | `ncclRemoteError` at the agreement |
| No route between members | `comm_read` (`mesh_link_map_joined`) | none | `ncclRemoteError` "has no route" |
| No lane pair toward a member (no cable of this node reaches it) | `session_open` (`lane_pair`) | none | `ncclInvalidUsage` (ENOLCK) |
| An init that cannot lease (`ENOSPC`, `EAGAIN`), a peer silent past the second gather's bound (`ETIMEDOUT`), a revoke | `comm_init` | every claim released (`comm_leave`) | the init's result, or `ncclCommGetAsyncError` after a non-blocking init |
| Stale leases: this process holds a lease of a stopped bridge | `attach_current` → `attaches_stale` | none | `ncclMemAlloc`, communicator, session, `ncclCommRegister`, `ncclMeshMemLive` fail with `ncclRemoteError` until every such lease is freed |
| Encode failure mid-walk (e.g. `EOVERFLOW` argument ring) | `launch_metal` / `ncclMeshComplete` | `walk_failed`: `release_all` and `inflight_lose` on each session through the ticket | the group's result; the later completion's loss |

**`drain`** applies before a host-path group; `session_close` waits for its own session's programs only. It waits on
`finished` for every library-signalled GPU program (classic form, library-made encoder, process-wide), looking again
every `MESH_PLATFORM_NS`, which decides nothing: every GPU wait is bounded, so each program ends.

## 7. Configuration operands

| Operand | Read by | Effect on the data flow |
|---|---|---|
| `NCCL_COMM_ID` (env) | `ncclGetUniqueId` | the root's `host:port`; bypasses opening a listener |
| `NCCL_BUFFSIZE` (env, bytes; ≤ 0 or unset → `MESH_BUFFSIZE` = 4 MiB) | `setting` in `session_open`, `ncclMeshFusableRooted` and the advisor | the large slot = `NCCL_BUFFSIZE / 8` rounded up to a block: ring bytes, pieces per message, and the digest (members must agree) |
| `ncclMeshStream.polls` (per call) | `metal_begin`/`metal_begin4` → `p.polls` | bounds each GPU wait to `polls` loads (0, or more than `NCCL_MESH_POLLS`: `NCCL_MESH_POLLS`) |
| `ncclMeshStream.workspace`, `issue`, `fused`, `argumentTable`, `residencySet`, `commandEncoder` | `enqueue`, `launch_metal` | form (classic or Metal 4), scratch source, issue/complete, deferred steps |
| `ncclMeshCommInitRank`'s `paths`, `programs`, `small`, `name`, `config` | `comm_routes`, `program_of`, `mesh_wire_of`, `comm_new`, `mesh_key`; `config->blocking` | trees, programs and the small slot as operands; slot key = FNV-1a(name); a non-blocking init |
| `ncclMeshCommForms`'s forms | `comm_routes` (`compile_form`), `program_of` | the collective forms ahead of the defaults: packed trees, two-phase all-reduce, chunks, size bounds |
| Bridge configuration (window, `pgsz`, `block`) | read from the region | the window for leases; a ring's pitch; `small` where the caller gives none |
| `ncclConfig_t` beyond `blocking`, `ncclCollConfig_t` | ignored | none (`algSelection` not read) |
| `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | `mesh_mpi.py` (`NCCL_COMM_ID` first) and torch's store | bootstrap only; the library reads none of them |
| `MESH_RDMA` | torch-mesh `setup.py` | build only |

Compile-time constants that shape the flow, none of them settable:

| Constant | Value | Meaning |
|---|---|---|
| `MESH_PLATFORM_NS` | 5 s | the second gather's and every later bootstrap send's or receive's bound; `drain`'s re-look |
| `NCCL_MESH_POLLS` | 4,500,000 | the most loads of any GPU wait |
| `MESH_LAG` | 8 | so `DEPTH` = 16, `POSITIONS` = 32 |
| `WATCH_NS` | 1 ms | the watcher's scan period while positions are pending |
| the bootstrap's pacing | 100 ms | a non-root's connect looked at and retried; a non-blocking root's accept looked for |
| `MESH_PLAN_VERSION` | 5 | planning rules, in digest and agreement |
| `OPS` | 64 | operators made on a communicator (PreMulSum, the partials merge) |
| `RECORDS` x `RECORD` | 65,536 x 128 B | the Metal 4 argument ring; overflow is `EOVERFLOW` |
| `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` / `MESH_LANE_PAIRS` | 64 / 64 / 4 / 4 | region limits; communicators a link at once |

Not read by the library (client-side, metal-microbench): `RECORD_ONLY` (`run_windows` skips submission),
`MESH_PROFILE` (`metal_recording.m`).

## 8. Where to look

| Question | Function |
|---|---|
| How does a rank find its node, and the root? | `ncclGetUniqueId`, `bootstrap_listen`, `bootstrap_open`, `comm_init` (`mesh-call.c:mesh_lanes_free`) |
| How are a communicator's lanes leased? | `comm_init`, `gather`, `bootstrap_bound`, `lease_claim`, `lease_keep`; `mesh-call.c:mesh_lanes_choose`, `mesh_lanes_claim`, `mesh_lanes_release` |
| A non-blocking init? | `ncclMeshCommInitRank`, `init_run`, `ncclCommGetAsyncError`, `ncclCommRevoke`, `ncclCommDestroy` |
| When does the process attach to its bridge; why `ESTALE` or `EBUSY`? | `attach_current`, `attaches_stale`, `attach_fail`; `mesh-dataflow.c:mesh_attach`, `mesh_slot_take` |
| What is a communicator's state? | `struct ncclComm`, `comm_new`, `comm_attach`, `comm_slot`, `ncclMeshCommInitRank` |
| When and how is the map read? | `session_open` → `comm_read` → `map_read` (`mesh-dataflow.c:mesh_map`) |
| Which trees? | `comm_routes`, `compile_form` (`mesh_trees_shortest`, `mesh_trees_pack`, `mesh_trees_check`) |
| Which program runs a call? | `program_of`, `comm_routes`, `ncclMeshCommForms` |
| What do members compare before running? | `map_digest`, `session_agree`, `lane_pair` |
| When is a session opened or closed? | `session_ensure`, `session_open`, `session_close` |
| What does a session lease, and how much? | `ring_make`, `ring_bind`, `mesh-call.c:mesh_transfers_prepare`, `mesh-collective.c:mesh_host_inputs`, `session_bytes` |
| Where is position t's slot, cell, word? | `publish`, `consume`, `channel_metal`, `cycle_of`, `landed` |
| Which positions does a group take? | `schedule_group`, `lay`, `mesh-collective.c:mesh_lay` |
| What order does the GPU encoding follow? | `order_group`, `walk`, `complete_group`, `issue_group` |
| Which kernel does what? | `nccl-mesh-metal.m` `STEPS`/`KERNELS`, `dispatch`, `dispatch4`, `record4` |
| The host path's loop? | `launch`, `session_run`, `run` |
| Fused crossings? | `ncclMeshFusableRooted`, `launch_metal` (`fusing`), `fused_piece`, `metal_send_deferred`, `metal_land_deferred`; mb `mesh_rank.m:window_submit`, `metal_recording.m:MetalReplaySubmit`, `metal/layers.metal:streamed_finish_publish`, `streamed_finish_norm_add` |
| Who watches the landings? | `progress_run`, `progress_targets`, `bridge_gone`, `start_refused`, `cancelled`, `release_all`, `cancel_all` |
| How does a GPU-side failure reach the host? | `mesh_step_check` → violation page → `violated` → `session_async` / `session_ensure` |
| What does async-error report? | `ncclCommGetAsyncError`, `session_async` |
| Revoke, destroy, exit? | `ncclCommRevoke`, `session_retire`, `ncclCommDestroy`, `session_close`, `slot_end`, `drain`, `session_exit` |
| Issue / complete / tickets? | `launch_metal` (`issue`), `ncclMeshComplete`, `retire`, `losses_through`, `ncclMeshRetired` |
| The rank's communicators? | mb `mesh_rank.m`: `MeshRankFlows`, `MeshRankGroup`, `MeshRankCancel`, `MeshRankDetach`; call delivery: `MeshRankDoor`, `MeshRankSend` (docs/elastic.md#call-delivery) |
| torch's use? | `torch_mesh/backend.py:ProcessGroupMesh`, `_stream.mm:on_stream`, `complete`, `MeshWork`, `functional` |

## Ambiguities and hazards in the code

1. **A bridge ended by a fault stores no `~0`.** An idle session learns of it at its next group (`session_ensure`'s
   `attach_live`), and a session with positions pending from its watcher (`bridge_gone`); a GPU wait already encoded
   ends at its bound.
2. **`ncclCommGetAsyncError` takes `session_lock`**, so it waits behind another thread's group, a host-path group
   holding it until its positions land or fail.
3. **`comm->async` is sticky.** Every later `ncclCommGetAsyncError` reports the first failed launch.
4. **The bounds are also bounds on peer skew.** A GPU wait ends at its polls (about 1.2 s at the Mini's working clock,
   at most about 5 s: rdma/NCCL.md "Bounded waits"), and the bridge fails a flow whose SENDs none retire for five seconds
   (B4): a member whose matching group, or whose program, starts that much later than its peer's fails the call, a
   live peer among them.
5. **A fused answer before the session.** `ncclMeshFusableRooted` reads the map and compiles the table before any
   session (`comm_read`); the session reads them again, so a map that moved between the two can make a fused answer
   disagree with the session's program, which the issue then refuses. A configured form that cuts the operand is never
   fusable (`whole_moves`).
6. **torch-mesh never calls `ncclCommDestroy`**, and calls `ncclCommGetAsyncError` only while it waits on a counted
   all-to-all's counts. A Metal-path failure found after a group returned surfaces at that communicator's next group;
   sessions live until `session_exit`.
7. **`run()` busy-spins with `session_lock` held.** A slow peer costs a core and blocks every other communicator's
   group, `ncclCommDestroy`, `ncclCommGetAsyncError` and `ncclMeshFusableRooted` in the process until its positions
   land, its words are cancelled (a SEND of its held five seconds: B4) or the communicator is revoked.
