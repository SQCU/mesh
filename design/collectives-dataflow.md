# Collectives data flow (libnccl-mesh)

What happens when a rank calls a collective: which thread, process and GPU does the work, and which words and
buffers it reads and writes. This is accurate to mesh `725a3dfb` and metal-microbench `6b1ea2b`; the bridge's stop, the
cancel (`ncclMeshCommCancel`) and the held flows to mesh `0fb8eb54` and metal-microbench `b181d45`. Citations are
`file:function` (line numbers at that HEAD); a bare function name is in `rdma/nccl-mesh.c`.

- Usage, build, rules a program keeps: [rdma/NCCL.md](../rdma/NCCL.md). The API contract: `rdma/nccl.h`'s header
  comment. Neither is repeated here.
- Everything inside the bridge process (pairing a program, posting a released SEND cell, writing a landing's
  completion word, cancelling words, relaying, retiring a dead process's slot, its stop): `design/bridge-dataflow.md`.
  This page describes only the library's side of the region and its words.

## 1. Layers and threads

`libnccl-mesh.dylib` is `nccl-mesh.c` + `nccl-mesh-metal.m` + `mesh-metal.m` (rdma/Makefile, target
`libnccl-mesh.dylib`), linked against `libmesh.dylib` (`mesh-dataflow.c`: attach, slots, map, leases; `mesh-call.c`:
sections, transfer binding, start, wake; `mesh-collective.c`: trees, compile, choose, wire, host publish/arrive). The
GPU code is `rdma/mesh.metal` (embedded as `.build/mesh_metal.h`) followed by `nccl-mesh-metal.m`'s `STEPS` and
`KERNELS` text, compiled once per device at first use (`nccl-mesh-metal.m:kernels_of`, `MTLMathModeSafe`).

```
 ┌──────────────────────────── client process: one rank (rank r of comm, on node N) ──────────────────────────────┐
 │ caller thread (torch: its MPS queue, via dispatch_sync; metal-microbench: the window's encoding thread)        │
 │   ncclAllReduce ─► enqueue ─► ncclGroupEnd ─┬─ stream NULL ──────► launch ─► session_run ─► run() (host memcpy,│
 │                                             │                                          combine_into, spins)    │
 │                                             └─ ncclMeshStream ───► launch_metal ─► walk() ─► metal_* (encode)  │
 │   session_lock (one, process-wide) held for the whole group; released only inside session_open's start+agree   │
 │                                                                                                                │
 │ GPU: the caller's MTLCommandBuffer / compute encoder, or MTL4ComputeCommandEncoder                             │
 │   library kernels (copy_*, send_small, land_*, spin, publish, publish_filler, combine_*, premul_*, truncdiv_*) │
 │   or the caller's own kernels calling mesh_step_* (a fused crossing)                                           │
 └─────────────┬───────────────────────────────────────────────────────────────────────────────▲──────────────────┘
               │ shared memory: region "/mesh0" (shm_open, mmap MAP_SHARED; mesh_open)         │
               │ ready words, slots ──►    ◄── completion words (1+cycle, or ~0)               │
 ┌─────────────▼───────────────────────────────────────────────────────────────────────────────┴──────────────────┐
 │ node N's bridge, mesh-flow (design/bridge-dataflow.md): pairs the session's program, posts each released slot  │
 │ over TB5 RDMA, writes the receiver's completion word on landing, relays, cancels, retires dead clients' slots  │
 └──────────────────────────────────────────── RDMA ──► peer node's bridge ──► peer rank's ring slot + word ──────┘
```

| Thread | Made by | Runs | Locks |
|---|---|---|---|
| Caller | the program | every API call; bootstrap (synchronous); map reads (synchronous); a host group's `run()` spin; a Metal group's encode; `session_watch` (a session's landings read where a call asks for its state) | `session_lock` for each group (`launch`, `launch_metal`, `ncclMeshComplete`, `ncclMeshFusableRooted`, `ncclCommDestroy`); `attach_lock` for leases and attach; `comm->guard` for opening/revoke |
| GPU | the caller's commit | kernels the library encoded, or the caller's kernels running the library's steps | none |
| `atexit` | `session_open` registers `session_exit` once | closes every session | `session_lock` |

libmesh's watcher of the bridge (`mesh-dataflow.c` `mesh_watch`: one thread a process, blocked in `kevent` on the
bridge's `NOTE_EXIT`) is the one other thread: at the bridge's exit it cancels the process's own slots' words and fails
their statuses (`ESRCH`), so no wait of the library outlives its bridge. Client-side threads that drive the library: metal-microbench `MetalReplaySubmit`'s completion handlers
(Metal's threads; they touch no library state). torch-mesh calls the library on torch's MPS stream queue inside
`dispatch_sync` (`torch_mesh/_stream.mm:on_stream`, `begin`, `complete`).

**What the library reads and writes in the region:**

- **Header** (`mesh.h` `struct hdr`). The library reads `bridge_pid` (`attach_live`), `node` (`session_open` checks it
  is the rank's member), `pgsz` and `block` (the wire), `data_off` and `length` (`ncclCommRegister`'s window bounds) and the
  map serial `map` (`mesh_map_serial`). It rings `control`
  (`mesh_transfers_start`, `mesh_retire`) and `bell[p]` (`mesh_transfers_wake`).
- **The process's client slot** (`mesh_attach`, key 0). Its rows own the leases (`ncclMemAlloc`).
- **One program slot per open session** (`mesh_program_attach`, key = FNV-1a of the communicator's name,
  `mesh-dataflow.c:mesh_key`).
  - The library writes: `owner`, `configured`, `depth`, `cyclic`, `digest`, `tx[d]` (the transfer lists, cells and
    cancel offset), `targets`.
  - It reads: `status[d]` (`prepared`, `phase`, `code`, `failed`) and `ports`.
- **The session's own leases** (`mesh_section_create`, whole blocks of the window): four rings toward each peer, the
  SEND cells and targets, the completion words, the cancellation ranges and a violation page.

## 2. Communicator lifecycle

**Bootstrap (`ncclGetUniqueId`, `ncclCommInitRank` → `bootstrap`, lines 605-735).**

- The id is the root's `host:port`, given by `NCCL_COMM_ID`. Without it, `ncclGetUniqueId` opens an IPv6 listener on
  an ephemeral port and holds it until the root's `ncclCommInitRank`, so it listens before any rank holds the id.
- The host part is the first non-loopback IPv6 address with prefix `MESH_ULA_PREFIX` (`fd6d:6573:68:`, the node's
  fabric identity), else `gethostname`.
- Each rank reports `{rank, node}`, where `node` is its bridge's node number from `mesh_observe` (a read-only map of the
  region).
- Rank 0 accepts until every other rank has reported and sends every rank the whole table. A non-root rank connects
  once: a root not listening refuses it, and the init fails as a value (`ncclRemoteError` naming the id). Nothing in
  the bootstrap is bounded by a clock.
- `members[r]` is rank r's node. `comm_make` attaches the process's client if it has none (`attach_current`; it fails
  like `ncclMemAlloc`, see section 6), allocates the communicator and names it after the id.
- Nothing crosses the fabric, no map is read and no session opens.

**The other constructors:**

- `ncclMeshCommInitRank` (line 752): no bootstrap. `members = node[]`, the name is the caller's (default `"nccl"`),
  `paths` and `programs` are copied, and `listed = count > 0`. Every rank must pass the same values; nothing checks
  this before the session's agreement (section 3).
- `ncclCommSplit`: a host-path `ncclAllGather` of `(color, key)` on the parent, then `comm_sub`, named
  `"<parent>/<split>.<color>"`.
- `ncclCommShrink`: no exchange. `NCCL_SHRINK_ABORT` revokes the parent first.
- `ncclCommGrow`: rank 0 broadcasts the new id on the parent (host path), then every rank runs `ncclCommInitRank`.
- `ncclCommInitAll`: `ndev` must be 1.

**Fields (`struct ncclComm`, line 213).**

| Field | Meaning | Written by |
|---|---|---|
| `members[nranks]`, `nranks`, `rank` | bridge node of each rank; this rank | `comm_make` |
| `name` | execution name; the session's slot key on every member | `comm_make` |
| `map`, `next`, `mark` | the members' map as last read: nodes, relays, links, costs, routing table | `comm_read` |
| `wire` | `{slot, small, lag}` of the rings | `session_open`, or `ncclMeshFusableRooted` before any session |
| `digest` | FNV-1a over everything the programs are made from (`map_digest`, line 403); 0 until the first session opens | `session_open` |
| `paths` | the caller's trees, every structural program's (`ncclMeshCommInitRank`; none: `trees`) | `ncclMeshCommInitRank` |
| `trees` | the map's shortest-path trees (`mesh_trees_shortest`), where the caller gave no paths | `comm_routes` |
| `forms`, `form_count` | the configured forms (`ncclMeshForm`) with their shares (one allocation, `forms_copy`), compiled ahead of the structural table; kept by splits and shrinks | `ncclMeshCommForms`, `comm_sub` |
| `entries`, `programs`, `listed` | the table of programs: the caller's (`listed`, kept for its life), else the forms' and then the structural one, compiled at each `comm_read` | `ncclMeshCommInitRank`, `comm_routes` |
| `session`, `opening` | the open session; a session between its start and its agreement | `session_open` |
| `revoked` | set by `ncclCommRevoke`, kept until destroy | `ncclCommRevoke` |
| `async` | the last failed group's result; never cleared | `launch_metal`, `ncclGroupEnd` |
| `ops[64]` | operators made on it: PreMulSum's scalar, or the partials merge's row (`unit`) | `ncclRedOpCreatePreMulSum`, `ncclMeshRedOpCreatePartials` |
| `scratch`, `spent` | the Metal 4 form's scratch when the caller gives no workspace | `workspace` |

**`comm_read` (line 567)** runs at every `session_open` and once in `ncclMeshFusableRooted` if no session has read
the map yet.

1. `routes_free(comm, 0)` drops the previous map, trees and (unless listed) table.
2. `map_read` calls `mesh_map(members)` (`mesh-dataflow.c`). It opens the region, reads the bridges' records under one
   serial and builds the whole map. The members become nodes 0..n-1 and every other node a relay. Each link carries
   the bandwidth of its slowest port at either end (0 where none reports one). The table is `mesh_routes`'. `map_read`
   retries with more room on `ENOSPC`.
3. `mesh_link_map_joined`. `ENOTCONN` fails the session with `ncclRemoteError`, naming the two members that have no
   route.
4. `comm_routes`:
   - a listed communicator does nothing more;
   - `paths` are checked against the map (`mesh_trees_check`) and are the communicator's trees; otherwise `trees` is
     `mesh_trees_shortest`;
   - each configured form (`ncclMeshCommForms`), in order, is compiled first (`compile_form`): along those trees, or,
     a form naming `ncclMeshTreesPacked`, along `mesh_trees_pack(map, the form's eps, the form's most)` made for it
     (mesh cc8a6f8e^'s `sets[1]`, whose `eps 0.1, most 4` were constants of the library; now the configuration's
     values), each segment cut among its root's packed trees by the form's `shares` (root r's k-th tree, the packing's
     order, `shares[r x most + k]`), else equally: the trees' weights `mesh_compile` copies into the program's parts,
     which `mesh_program_ranges` cuts by, are the configuration's, never the packing's flows (inferred from the map's
     link bandwidths); single-phase or two-phase, at its chunks, below its bound, a rooted form at its root or every
     root;
   - then the structural table: the single-phase all-reduce, the reduce-scatter, the all-gather and each root's reduce
     and broadcast, along the trees, uncut, of any size.

**Held and adopted communicators (metal-microbench `mesh_rank.m`).** A call's communicators are its crossings' member
sets (`MeshRankGroup`, line 207).

- **Held map.** `held_flows()` is a process-wide name → `ncclComm_t` map. It survives across calls in one serving
  process.
- **`MeshRankFlows`** (line 149), at a call's start. It reads the record's `flows` (each `{nodes, name, small, hold}`)
  and destroys every held communicator the record does not list.
- **`MeshRankGroup`** (line 207), at the first crossing over a set. A set the record names takes the flow's name, else
  `"<call>/<ranks>"`. Then `adopted(name, count, rank)` (line 176):
  - takes the held communicator out of the map;
  - returns it only if `healthy(comm)` (line 61: `ncclCommGetAsyncError` is `ncclSuccess` and the async result is
    `ncclSuccess`) and `ncclCommCount` and `ncclCommUserRank` match;
  - otherwise destroys it.
  - An adopted communicator is taken up only where every member takes it up for this call (`agreed`, line 195:
    `ncclMeshCommAgree` on a ticket of the call's name), else destroyed.
  - Without an adopted communicator, `ncclMeshCommInitRank` makes one.
- **What adoption keeps.** An adopted communicator keeps its open session: its program stays paired, its rings posted,
  its positions continue from the last call's.
- **`MeshRankDetach`** (line 664) holds a communicator only if:
  - its flow has `hold`;
  - the call was not cancelled;
  - `healthy(comm)` holds.

  Every other communicator is destroyed (`ncclCommDestroy`). Nothing in the rank reads the bridge's stop: a held
  communicator whose bridge stopped has its next position cancelled (`~0`), so `healthy` refuses it at the next
  adoption and it is destroyed.
- **Adopting or making is decided by each member locally** (section 6); taking up an adopted one is agreed with
  the peers (`agreed`).

**`ncclCommRevoke` (line 797)** runs under `comm->guard` alone, never `session_lock`, so it reaches a member blocked in
another thread's group.

- It sets `revoked`.
- A session that is opening (`comm->opening`) is retired: `session_retire` → `mesh_retire` clears the slot's `owner`
  and wakes `status[d].prepared`. The start's wait returns `ECANCELED`, and the bridge fails the peers' programs toward
  it.
- An open session's failure is kept (`release_all`: `failed`, `signaled = target` on every channel) and its program
  retired as an abort (`session_retire`): its bridge mutes the program's flows, tells each peer FAIL and cancels its
  words past their landed counts, so every GPU and host wait ends (design/bridge-and-ipc.md#control).
- Every later group fails at `launch`/`launch_metal`'s `revoked()` check.

**`ncclMeshCommCancel`** (line 931) does what the revoke does to a call in flight, under `comm->guard` alone, and
keeps the communicator: for a caller whose requester went away (its request's channel closed) and that holds the
communicator for the next request.

- A session that is opening is retired (`session_retire`): its start returns `ECANCELED` and the group fails.
- An open session sets `cancelled`, keeps the failure (`release_all`, "its call on communicator ... was cancelled") and
  is retired as an abort (`session_retire`): its bridge tells each peer FAIL and cancels its words past their landed
  counts, so every GPU and host wait of the call ends and the call fails as a value.
- `revoked` is not set. The next group's `session_ensure` reads `cancelled` (and clears it), closes the cancelled
  session as an abort without failing that group, and opens a new one; the peers' sessions fail at the FAIL and open
  again at their next groups.

**The close** (NCCL's Finalize, Destroy and Abort): nothing of it goes on the wire.

- **`ncclCommFinalize`** takes `session_lock`, waits for any opening, `drain`s the GPU programs and returns the
  failure the session holds (`session_kept`: `session_watch`, then `s->failed`), if any. `finalized` is set: no group
  runs on comm after it.
- **`ncclCommDestroy`** takes `session_lock`, waits for any opening, runs `session_close(s, orderly = !revoked)` and
  frees the communicator. It waits on no peer: the bridge posts every cell the session released, takes those SENDs'
  completions, then releases the session's flows and rows; no peer is told.
- **`ncclCommAbort`** is `ncclCommRevoke` then `ncclCommDestroy`.

**`session_close(s, orderly)`**, in order:

1. Where not `orderly`, or the session failed: `session_retire` (an abort, before anything else, so the bridge mutes
   and fails its flows and every wait on its words ends).
2. Loses the groups in flight (`inflight_lose` records each as a `losses` entry for the next completion).
3. `drain`: waits for library-made GPU programs; see section 6.
4. Releases the ring and violation `MTLBuffer`s.
5. Destroys the transport.
6. `mesh_detach`: the slot retired in order (`MESH_ORDERLY`), its rows the bridge's until it has released its flows
   (a no-op where the abort retired it).
7. Decrements and prunes the attach.

Its caller must have completed every window that waits on the session's words. The Metal 4 form signals nothing to the
library (`program_end` with a borrowed encoder), so `drain` cannot wait for those windows. `mesh_rank.m:run_windows`
waits for every submitted window before the detach.

**`ncclCommGetAsyncError`** returns `comm->async` if set. Otherwise it calls `session_async` under `session_lock`:

- `session_watch(s)`: `violated(s)`, which turns violation words [0..3] and [5] into the session's failure, then each
  channel's completion words from `signaled` up to `target` (a `~0` the session's failure; `signaled` moved past those
  landed);
- for an idle session (nothing outstanding), each channel's word at `end`: `~0` is the peer's program ended while the
  session was idle;
- returns `s->failed`.

What it detects: a launch of this communicator that failed (`comm->async`; never cleared); a cancelled GPU wait or a
position whose filler stamp disagrees with its role (violation page); a landing cancelled while positions were pending
or at an idle session's next position; a revoke or a cancel. This node's bridge's stop is found as the words it
cancels: its own teardown cancels them past their landed counts as its ports are lost, and at its exit libmesh's
watch cancels the rest, so the next read finds `~0`; the next group's `session_ensure` and `ncclMeshMemLive` then see
the bridge gone.

**An idle held communicator learns that its peer process exited** at `ncclCommGetAsyncError` (its next position's word
reads `~0`) or at its next group:

- Host path: `run()` reads `~0` (the bridge cancelled the positions past the landed count: the peer's FAIL, its link
  or its bridge lost) and the group fails.
- Metal path: the GPU wait that reads `~0` records it in the session's violation words ([2] and [3]) and every later
  wait of the program ends; the land kernels after it combine whatever their slots hold, and none of the program's
  cells is posted after the cancel (the bridge muted its flows first). The session fails as a value (`violated`), and
  the caller sees the failure after the window (`ncclCommGetAsyncError`) or at the next group.

## 3. Sessions

A session is a communicator's program on its node's bridge. It opens at the first group that crosses a link
(`crosses`: elements, more than one rank, a point-to-point peer other than itself), and again after a failure.

**`session_open` (line 1386)** is called and returns with `session_lock` held:

1. Revoked → fail.
2. `mesh_map_serial()` reads the map serial, then `comm_read`.
3. `attach_current` + `mesh_program_attach(&s->context, &a->context, comm->name)`: a new slot on the process's
   attach, keyed by the name.
4. Checks `hdr->node == members[rank]`.
5. The wire: `mesh_wire_of(pgsz x block, NCCL_BUFFSIZE or MESH_BUFFSIZE, comm->small)`. Then `digest = map_digest`, which covers
   `MESH_PLAN_VERSION`, the wire, the members, every member pair's route marks and its links' costs, the tree sets, and
   the caller's paths and programs.
6. Two channels toward every other member, whatever the map. `ring_make` leases `DEPTH x slot` bytes each way.
   `ring_bind` binds them:

   | Channel | Queue | Transfer identity |
   |---|---|---|
   | large | 0 | `1 + 2 x sender_rank + 1` |
   | small | 1 | `1 + 2 x sender_rank` |

   Each class has its own queue because one queue's receives are posted invocation-major across its transfers.
7. `mesh_transfers_prepare(1 slot, POSITIONS invocations, DEPTH)` lays out the SEND cells (each cell's request
   `wr_id = 1`, the "argument") and the targets. Then:
   - `mesh_transfers_digest(digest)`: the bridge compares digests on READY; a mismatch is `ESTALE`;
   - `mesh_transfers_cyclic`: invocation T is cell/word T mod 32, cycle T / 32;
   - `mesh_metal_transport_create`: the `publication` `MTLBuffer` wraps the SEND-cell lease, `mesh_host_inputs` leases
     the completion words and cancellation ranges, and `inputs` wraps the words;
   - `channel_metal` per channel: `ring_out`/`ring_in` `MTLBuffer`s, `cell` (the offset of the invocation-0 SEND cell
     in `publication`), `argument`, and `word`/`word_stride` (the in ring's last chunk's completion word,
     `mesh_metal_receive_prepare`);
   - a one-page violation lease (`violation`, wrapped as `violation_buffer`);
   - the process-wide `finished` `MTLSharedEvent`.
8. Sets `comm->opening = s` and **unlocks `session_lock`**.
9. `mesh_transfers_start` (`mesh-call.c` line 199) compare-exchanges `configured`, rings `control`, and waits on each
   destination's `status[d].prepared` until the bridge has paired the program toward it, directly or along its route.
   The wait has no bound: it ends at the pairing, a failure the bridge tells, `ESRCH` (the bridge gone: libmesh's
   watch fails the statuses and wakes it) or `ECANCELED` (the slot was retired: a cancel or a revoke). A peer whose
   program never starts is waited for until the caller's cancel (`ncclMeshCommCancel`) or revoke. A failure goes through `start_failure` and names the destination,
   the node where it failed and `strerror(code)`.
10. `session_agree`, then the wait time is added to `session_waits` (`ncclMeshSessionWaits`, a measure).
11. Relocks, clears `opening`, broadcasts `opened`, links `s` into `sessions` and sets `comm->session`.

**The agreement (`session_agree`, line 1290)** is position 0 of every channel.

- Each member writes `{MESH_PLAN_VERSION, digest, serial, node}` into slot 0 of its out ring and publishes position 0
  (`mesh_host_publish`).
- It then polls each in ring's position-0 word (`mesh_host_arrived`, `sched_yield` between polls) until it lands, is
  `~0` (its bridge's cancel: the peer's program or link lost; or the process's own where the bridge exits), or the
  communicator is revoked; no clock ends it.
- It compares version and digest. `serial` and `node` only name the disagreement.
- Every channel's cursors then become 1.

**A channel (`struct channel`, line 984)** is the rings toward one peer, of one class.

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
| `target` | how far `session_watch` reads: `end` at each group's schedule (`session_targets`) |
| `signaled` | positions `session_watch` has found landed |

**The wire (`mesh-collective.c:mesh_wire_of`, `mesh_wire_large`, `mesh_wire_pieces`).**

- `block` is `pgsz x block`, the bridge's row, and a ring's pitch: slot `t` of a ring starts at `t x
  mesh_wire_pitch(slot)`, its bytes rounded up to blocks (the transfer's `invocation_pages`), whatever the slot's bytes.
- `small` is the communicator's operand (`ncclMeshCommInitRank`'s `small`: mb's record names each flow's, its decode
  crossings' largest, `tools/mesh/flows.py smalls`), at least `MESH_SMALL_LEAST` (32, the agreement position 0
  carries); 0 gives one block.
- `slot` is `ceil(NCCL_BUFFSIZE / LAG / block) x block`, at least a block.
- `lag = MESH_LAG = 8` (NCCL_STEPS).
- A piece takes the large channel if its bytes exceed `lag x small`, or its element is wider than a small slot. It is
  cut into positions of whole elements: `ceil(bytes / carried)`, `carried` its class's slot less the remainder over its
  element (`mesh_wire_pieces`; `struct message` `carried`, `piece_at`), so a combine of a position's piece combines
  whole elements: a typed element of 1 to 8 bytes on slots of whole blocks (or the pair's 3,072-byte small slot)
  carries the whole slot, as before; a counted all-to-all's rows message is laid in its type's elements, as before;
  the partials merge's element is its row.
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
| Completion word | `inputs + word + (t % POSITIONS) x word_stride` | written by the bridge: `1 + t / 32` once landed, `~0` once cancelled. Landed iff `word != ~0 && word >= 1 + t / 32` (`landed`, line 1026) |

**Credit.** An end publishes t only once it has consumed the peer's t − LAG. The peer then has consumed this end's
t − DEPTH, the slot t reuses (`run`: `sent < consumed + LAG`; `publish`: `arrive(need = t − LAG + 1)`). Every position
is published by both ends, a filler where an end has nothing to send. The receiver checks every slot's stamp against
its own role: host in `run()`, GPU in `mesh_step_check`. A mismatch is `MISMATCH` / `ncclInvalidUsage`.

**What a session costs.**

- Window bytes: `ncclMeshSessionBytes` → `session_bytes` (line 1350): for each peer, four rings and its lists; plus the
  SEND cells, completion words, cancellation ranges and a violation page.
  - A relay's bridge leases `ncclMeshRelayBytes` per routed direction.
- Slots: one of the bridge's `MESH_CLIENTS` (64) client slots, plus one on each relay per direction.
- Threads: none of its own.
- Time: the start (the peer's program's start plus pairing) and the agreement; `ncclMeshSessionWaits` sums them.

**`session_ensure` (line 1744)** decides when a session is reopened:

- wait for another thread's opening of it;
- open a session if there is none;
- if `s->failed` (after `session_watch`): return the failure and close the session. This group fails; the next one
  opens a new session;
- if no group is in flight and the attach's bridge is gone: close and reopen now.

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
- A point-to-point call is one step (`plan_calls`, line 1544): a SEND, or a COPY into its receive buffer.
- Routes are data: the bridges' table in `map.next`, and the tree sets.

**Choosing a program (`program_of`):** every communicator takes the first entry of its table with the call's `what`
(and `root` for a rooted call) and `!below || bytes < below`. With no match, the call fails: `ncclInvalidUsage`, "no
program". The table is the caller's programs, or the configured forms' entries followed by the structural table
(`comm_routes`), so a call no form fits runs the default. No price, timing or cache chooses: mesh cc8a6f8e deleted
`mesh_choose`, `mesh_program_time`, the size classes and `kept`, and the forms they chose among are entries a
configuration names (`ncclMeshCommForms`). `MESH_PLAN_VERSION` (5: positions carry whole elements) enters both the digest and the agreement, so members
whose planning rules differ fail at the session instead of mismatching later; the digest covers every entry's program,
so members configured with different forms fail there too.

**Places where a formula or constant, not a table or evidence, chooses:**

| Place | Choice made by |
|---|---|
| `program_of` | the table's order: the configured forms (the caller's configuration), then the structural defaults |
| `mesh_map` link bandwidths | the port's nominal speed, read by the routes (ties among routes of equally few links) and by `mesh_trees_pack` |
| `compile_form` | packing `eps` and `most`, the trees' `shares`, and every other form field, are the configuration's (`ncclMeshForm`) |
| wire (`mesh_wire_of`, `mesh_wire_large`) | `lag = 8`; the large/small split at `lag x block` (or an element wider than a small slot); slot = `NCCL_BUFFSIZE / 8` rounded to a block, carrying whole elements |
| `nccl-mesh-metal.m:dispatch` | 256 threads a pointwise kernel, at most 1,024 threads in one threadgroup for a block's piece, at most 2^20 units a grid |
| `ncclMeshFusableRooted` | fusable only at 2 ranks, one small slot, float16 sum (or a broadcast of any type), at most 1 scratch |

**Scratch.** `program->scratch` is the most scratch buffers a rank uses.

- Host path: `malloc` copies of the staged operand (`launch` 1865-1867).
- Metal path, each a `metal_copy` of `own` (`launch_metal`; none where the send buffer is read in place, below) into
  either:
  - the stream's `workspace` callback (classic form; required with a borrowed encoder);
  - the communicator's `scratch[k]` (Metal 4 form without a workspace; grown, never shrunk);
  - a private buffer the program keeps until it ends (classic form without an encoder).
- **The two-rank all-reduce** (the direct exchange, `whole`) has exactly one scratch: the rank's copy of its operand,
  which only its SEND reads while the REDUCE sums into buffer 0. The fused form makes no copy and aliases the scratch to
  `own` (line 2530), because the caller's kernel publishes from its own store.

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
- The host path copies as before; the caller's own GPU work on a lease is complete before a host-path group reads or
  writes it.

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

## 5. The life of one collective

### (a) Library form, host path: `ncclAllReduce(send, recv, n, type, op, comm, NULL)`

1. **`enqueue`** (line 2694) validates and appends a `struct call` to the thread-local `calls`. The implicit
   `ncclGroupStart`/`End` around it runs the group immediately when the call is outside a group. Inside
   `ncclGroupStart`/`End`, calls accumulate until the outermost `ncclGroupEnd`.
2. **`launch`** (line 1835):
   - `reduction` sets `combine`, `premultiplied`, `post`;
   - revoked → fail;
   - **locks `session_lock`**; `session_ensure` (it may open the session, section 3);
   - fails if any GPU group is issued and not complete (process-wide `inflight`);
   - `drain` waits for library-signalled GPU programs (section 6);
   - `plan_calls`.
3. **Staging.**
   - Each collective gets a host `result` buffer from `calloc`. `stage` copies or premultiplies `send` into it.
   - Each scratch b ≥ 1 is a `malloc` copy of it.
   - A SEND's buffer 0 is `send` itself, a receive's is `recv`.
4. **`session_run`** (line 1772):
   - `schedule_group`: `lay` gives each channel's positions in call order (`mesh_lay`: round by round, SENDs and
     receives side by side), then point-to-point calls with the j-th send beside the j-th receive.
     `mesh_steps_before` gives each step's prior receive. `ch->end` advances.
   - `session_targets`: `target = end`.
   - `mesh_transfers_wake` rings the bells of the bridge's ports.
5. **`run()`** (line 1680) spins on the caller's thread, still holding `session_lock`, until every channel has
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

The call is the same with an `ncclMeshStream` whose buffers are `ncclMeshBuffer`s. `launch_metal` (line 2453):

1. **Checks.** One communicator, one command buffer or encoder; no float64. A Metal 4 stream must have
   its encoder and residency set, and a workspace if it is issued. A classic stream with an encoder must have a
   workspace. Then `reduction`, the revoke check, lock, `session_ensure`, `plan_calls`, `metal_collect(finished)`.
2. **Program.**
   - `metal_begin4` (Metal 4, line 231): the caller's encoder, argument table and residency set, the device's argument
     ring and a leading barrier.
   - `metal_begin` (classic, line 220): the caller's command buffer; the caller's encoder (borrowed) or one made lazily.
   - Either binds the session's `violation_buffer` to every dispatch that names no check's words (`p.halts`).
   - Metal 4 makes every ring, `publication` and `inputs` resident.
3. **Staging.**
   - `own` is `recv` for in-place work (all-reduce, broadcast, all-gather, the root's reduce), else a workspace
     scratch.
   - `metal_premultiply` or `metal_copy(PLAIN)` of `send` into `own` (or the rank's all-gather segment), unless they
     are the same buffer and offset.
   - Each scratch b ≥ 1 is a `metal_copy` of `own`.
4. **In flight.**
   - An `inflight` group `g`: its positions `[start, end)` per channel, a fresh `ticket`, appended to the
     process-wide `inflight` list.
   - `order_group` (line 2022) builds every rank's layout on every channel and orders it with Kahn's algorithm (an
     MSCCLang-style topological order). Each end's publication and consumption of each position is a node. A
     publication's level is 1 + its predecessors' maximum; a consumption's is that maximum. This rank's publications
     are sorted by `(level, channel, position)`, which every rank's walk keeps.
   - `session_targets`, `mesh_transfers_wake`.
5. **`walk(program, NULL, g)`.** `complete_group` (line 2263) runs for every group up to `g`; for each publication in
   order:
   - `consume` up to t − LAG + 1, and the SEND's prior receive;
   - `publish` (line 2219): `arrive(need)` encodes `spin` dispatches for positions not yet `waited`, then:

   | Position | Small channel | Large channel |
   |---|---|---|
   | publish a piece | `send_small`: one threadgroup copies the piece into the slot, `threadgroup_barrier`, thread 0 `mesh_publish`es the cell (system-scope fence, then store `1 + cycle`) | `copy_send` into the slot, then `publish` (one thread) |
   | publish nothing | `publish_filler`: stamp + release | `publish_filler` |
   | consume a piece | `land_<type>` (REDUCE) or `land_copy` (COPY): thread 0 waits on the completion word and checks the window of positions, then barrier + system fence, then all threads combine or copy from the slot | `spin`, then `combine_<type>` or `copy_land` |
   | consume nothing | no dispatch (checked by the next wait that covers it) | no dispatch |

   After each dispatch: Metal 4 `dispatch4` encodes a barrier; the classic encoder is serial.
6. **Retire.** `retire` (line 2401) runs `epilogue` (results into `recv`: a copy if `sum != recv`, `truncdiv_*` for an
   integer `ncclAvg`) for every group whose positions are all consumed, and sets `retired`.
7. **End.** `program_end`: a borrowed encoder signals nothing; otherwise `encodeSignalEvent(finished, ++issued)` for
   `drain`. Metal 4 commits the residency set if a buffer was added.
8. On an encode failure, `walk_failed` fails every session through the ticket. Unlock; on failure set `comm->async`.
9. **The caller commits.** The bridge then carries each released cell (design/bridge-dataflow.md), the peer's kernels'
   waits end on their words, and `session_watch` advances `signaled` where a call reads the session. Completion is the caller's command buffer
   completing. The library learns nothing of it except through `finished` (non-borrowed encoders only).

**Issue and complete.**

- With `stream->issue`, `walk(program, g, NULL)` runs `issue_group` (line 2286): only publications that need no wait
  (`need <= waited`, prior receives already consumed). `issue->ticket` is the ticket, or 0 if the group retired at
  once.
- `ncclMeshComplete(ticket, stream)` (line 2581):
  1. reports losses through the ticket (`losses_through`);
  2. begins a program on the given stream;
  3. `walk(program, NULL, through)`, then `retire`.
- torch-mesh's `MeshWork::finish` → `complete` uses this; `_stream.mm:release` frees the workspaces through
  `ncclMeshRetired()`.

### (c) Fused form recorded into a Metal program (`ncclMeshFusable`, `ncclMeshFusableRooted`)

**Recording time** (metal-microbench).

- `decode_crossings.swift:StepCrossings.init` calls `MeshRankGroup` for each crossing, then `MeshRankFusable`
  (`mesh_rank.m` line 320) → `ncclMeshFusableRooted` (line 2623).
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
  - the next finish lands: `streamed_finish_norm_add` with flags bit 64 (`dense_execution.swift` 309-321, 428-440).
- `MeshRankCross` inside a recording only marks the crossing (`MetalRecordMark`); no library call is made.
- The engine compiles the library's step text into its prelude (`metal_runtime.swift:447`: `ncclMeshMetalSteps()`
  after `mesh.metal`).

**Replay time** (`mesh_rank.m:run_windows` → `window_submit` → `metal_recording.m:MetalReplaySubmit` 865-887), per
invocation t and mark k:

1. Encode the recorded commands up to the mark, minus the last one if it publishes.
2. **`cross()`** calls `issue()` → `ncclAllReduce` (or `ncclReduce`, `ncclBroadcast`) with
   `ncclMeshStream{commandEncoder, argumentTable, residencySet, fused = &steps}`. In `launch_metal`:
   - It refuses a fused group that is not exactly one in-place all-reduce, broadcast or reduce on a Metal 4 stream,
     not issued. It also refuses scratch buffers other than `1 + (all-reduce at 2 ranks)` (2476-2486).
   - Staging: `send == recv`, so no copy; the scratch is aliased, not copied (2530).
   - `fusing` (2557-2561) marks the call. In the walk:
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
   - It checks that exactly the fused steps were written (2563); otherwise `ncclInvalidUsage`.
3. Encode the publishing command (the producer kernel) with `steps.publish` bound at argument-table indices 7..12
   (`MetalFusedBinding`), then the landing command with `steps.land` at 7..12.

**GPU time, both ranks symmetric for the all-reduce, position p:**

1. The producer kernel (`streamed_finish_publish`):
   - `mesh_step_enter` (thread 0 stores the record's `sequence` into the argument ring's progress word);
   - each element goes to the engine's output and, with `mesh_step_store`, into `ring_out + (p % 16) x small`;
   - `mesh_step_release`: device barrier, then thread 0 `mesh_publish`es SEND cell `p % 32` with `1 + p / 32`.
2. The bridge sees the ready word and posts the slot. The peer's bridge writes the peer's completion word `p % 32` to
   `1 + p / 32` (design/bridge-dataflow.md).
3. The landing kernel (`streamed_finish_norm_add`, flags 64), each threadgroup:
   - `mesh_step_enter`; `mesh_step_await`: thread 0 waits with `mesh_remote_wait`'s spin (ended by the landing, a
     cancel, or violation [2] read once in 64 looks), then `mesh_step_check` checks the window's stamps (violation [0],
     [1]) or records a cancel ([2], [3]);
   - device barrier + system-scope fence;
   - adds the peer's row from `ring_in + (p % 16) x small` (`mesh_step_read`) to its own before the norm.
   - Nothing reads the operand buffer as the sum.

The crossing therefore costs the landing's wait inside the consumer kernel and no dispatch of its own. The library
form costs `copy_plain` (scratch), `send_small` and `land_f16`. The wire behaviour (positions, slot bytes, cells,
stamps) is the same in both forms, so two members may encode the same crossing differently without disagreeing.

## 6. Failure data flow

**Who writes `~0` into a completion word:**

- the bridge, past each channel's landed count, after the last landing it stores (design/bridge-and-ipc.md#control):
  when a link, a relay or the peer's bridge is lost (link off, its control socket's end, its FAREWELL, an error
  completion), the peer's program fails or aborts (FAIL), this session's program is retired as an abort (a revoke, a
  cancel, a failed session's close), or the bridge stops (its ports lost: a signal or a change of its binary). A GPU wait that reads a cancel halts the program's later waits.
- the process itself where its bridge exits (libmesh's watch, `mesh-dataflow.c` `mesh_watch`): every word of its own
  slots' programs not yet landed (every one of a cyclic program's), its destinations' statuses failed `ESRCH`.

| Event | Detected by | State written | What the caller sees |
|---|---|---|---|
| Link or relay lost, peer's bridge lost, peer program failed or aborted, this node's bridge exited | bridge (or libmesh's watch) → words past landed `~0`; then whichever reads first: `run()`, `session_agree`, a GPU wait, `session_watch` (`ncclCommGetAsyncError`, the next group, Finalize) | `s->failed = ncclRemoteError` ("link to node N was lost") | host: the group fails. GPU: the wait records the cancel (violation [2], [3]) and later waits end; failure after the window or at the next group |
| GPU wait read a cancel | `mesh_step_check` → violation [2], [3]; later waits end at [2] (read once in 64 looks while spinning) | `violated` → `release_all` | async error / next group: "was cancelled under a GPU wait" |
| Calls that do not pair | host `run()` stamp check; GPU `mesh_step_check` → violation [0], [1] | host: `ncclInvalidUsage` at once; GPU: `s->failed` via `violated` | `MISMATCH` text |
| Cancel | `ncclMeshCommCancel` | `cancelled`; retire the opening slot, or the open session's program (an abort); `release_all`; the next group's `session_ensure` closes it and opens a new session | `ncclRemoteError` "was cancelled", the call in flight alone |
| Revoke | `ncclCommRevoke` | `revoked`; retire the opening slot, or the open session's program (an abort); `release_all` | `ncclRemoteError` "revoked", every later group |
| Map disagreement | bridge compares the READY digests (`mesh_transfers_digest`) → start fails with `ESTALE`; else `session_agree` compares | `start_failure` / agreement failure; `session_close` | `ncclRemoteError` naming nodes, serials, digests; the next group reopens on the map then |
| No route between members | `comm_read` (`mesh_link_map_joined`) | none | `ncclRemoteError` "has no route" |
| Stale leases (`ESTALE`): this process holds a lease of a stopped bridge | `attach_current` → `attaches_stale` | none | `ncclMemAlloc`, communicator, session, `ncclCommRegister`, `ncclMeshMemLive` fail with `ncclRemoteError` until every such lease is freed |
| Local bridge gone | libmesh's watch (`NOTE_EXIT`): the process's words cancelled, statuses `ESRCH`; `attach_live` | `session_ensure` closes and reopens (no group in flight); `mesh_transfers_start` `ESRCH` | reopen, or `ncclRemoteError` |
| Peer never starts its program | none (no event says so) | none | the start waits until the caller's cancel (`ncclMeshCommCancel`) or revoke (`ECANCELED`) |
| Encode failure mid-walk (e.g. `EOVERFLOW` argument ring) | `launch_metal` / `ncclMeshComplete` | `walk_failed`: `release_all` and `inflight_lose` on each session through the ticket | the group's result; the later completion's loss |

**The bridge's stop.** The library reads no stop word. A bridge stops at once at a signal or a change of its binary
(design/bridge-dataflow.md#5-failure-cancellation-and-release), and every session on it fails as a value: the peers'
at its FAREWELL, this process's as its own teardown cancels the words past their landed counts and, at its exit, as
libmesh's watch cancels the rest (statuses `ESRCH`). The next group's `session_ensure` closes the session and opens
one on the bridge started again (`attach_live`, `attach_current`); libmesh's `mesh_exit_wait` and `mesh_bridge_wait`
tell a caller the old bridge's exit and the new one's start. metal-microbench and torch-mesh read no stop either.

**`drain`** applies before a host-path group, in `ncclCommFinalize` and in `session_close`.

- It waits on `finished` for every library-signalled GPU program (classic form, library-made encoder, process-wide),
  with no bound: `metal_wait(finished, issued)`.
- Its end is an event: each program's completion (a wait in it ends at its landing, its bridge's cancel, or, where
  the bridge exits, the process's own cancel by libmesh's watch). A program whose peer's client never runs its side
  waits until the caller cancels its call (`ncclMeshCommCancel`) or revokes the communicator (`ncclCommRevoke`): its
  bridge cancels the words.

## 7. Configuration operands

| Operand | Read by | Effect on the data flow |
|---|---|---|
| `NCCL_COMM_ID` (env) | `ncclGetUniqueId` | the root's `host:port`; bypasses opening a listener |
| `NCCL_BUFFSIZE` (env, bytes; ≤ 0 or unset → `MESH_BUFFSIZE` = 4 MiB) | `setting` in `session_open` and `ncclMeshFusableRooted` | the large slot = `NCCL_BUFFSIZE / 8` rounded up to a block: ring bytes, pieces per message, and the digest (members must agree) |
| `ncclMeshStream.workspace`, `issue`, `fused`, `argumentTable`, `residencySet`, `commandEncoder` | `enqueue`, `launch_metal` | form (classic or Metal 4), scratch source, issue/complete, deferred steps |
| `ncclMeshCommInitRank`'s `paths`, `programs`, `small`, `node`, `name` | `comm_routes`, `program_of`, `mesh_wire_of`, `comm_make`, `mesh_key` | trees, programs and the small slot as operands; slot key = FNV-1a(name) |
| `ncclMeshCommForms`'s forms | `comm_routes` (`compile_form`), `program_of` | the collective forms ahead of the defaults: packed trees, two-phase all-reduce, chunks, size bounds |
| Bridge configuration (window, `pgsz`, `block`) | read from the region | the window for leases; a ring's pitch; `small` where the caller gives none |
| `ncclConfig_t`, `ncclCollConfig_t` | ignored | none (`algSelection` not read) |
| `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | `mesh_mpi.py` (`NCCL_COMM_ID` first) and torch's store | bootstrap only; the library reads none of them |
| `MESH_RDMA` | torch-mesh `setup.py` | build only |

Compile-time constants that shape the flow, none of them settable:

| Constant | Value | Meaning |
|---|---|---|
| `MESH_LAG` | 8 | so `DEPTH` = 16, `POSITIONS` = 32 |
| `MESH_PLAN_VERSION` | 5 | planning rules, in digest and agreement |
| `OPS` | 64 | operators made on a communicator (PreMulSum, the partials merge) |
| `RECORDS` x `RECORD` | 65,536 x 128 B | the Metal 4 argument ring; overflow is `EOVERFLOW` |
| `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` | 64 / 64 / 4 | region limits |

Not read by the library (client-side, metal-microbench): `RECORD_ONLY` (`run_windows` skips submission),
`MESH_PROFILE` (`metal_recording.m`).

## 8. Where to look

| Question | Function |
|---|---|
| How does a rank find its node, and the root? | `ncclGetUniqueId`, `bootstrap` (`mesh-dataflow.c:mesh_observe`) |
| When does the process attach to its bridge; why `ESTALE` or `EBUSY`? | `attach_current`, `attaches_stale`, `attach_fail`; `mesh-dataflow.c:mesh_attach`, `mesh_slot_take` |
| What is a communicator's state? | `struct ncclComm` (213), `comm_make`, `ncclMeshCommInitRank` |
| When and how is the map read? | `session_open` → `comm_read` → `map_read` (`mesh-dataflow.c:mesh_map`) |
| Which trees? | `comm_routes`, `compile_form` (`mesh_trees_shortest`, `mesh_trees_pack`, `mesh_trees_check`) |
| Which program runs a call? | `program_of`, `comm_routes`, `ncclMeshCommForms` |
| What do members compare before running? | `map_digest`, `session_agree`, `mesh-call.c:mesh_transfers_digest` |
| When is a session opened or reopened? | `session_ensure`, `session_open` |
| What does a session lease, and how much? | `ring_make`, `ring_bind`, `mesh-call.c:mesh_transfers_prepare`, `mesh-collective.c:mesh_host_inputs`, `session_bytes` |
| Where is position t's slot, cell, word? | `publish`, `consume`, `channel_metal`, `cycle_of`, `landed` |
| Which positions does a group take? | `schedule_group`, `lay`, `mesh-collective.c:mesh_lay` |
| What order does the GPU encoding follow? | `order_group`, `walk`, `complete_group`, `issue_group` |
| Which kernel does what? | `nccl-mesh-metal.m` `STEPS`/`KERNELS`, `dispatch`, `dispatch4`, `record4` |
| The host path's loop? | `launch`, `session_run`, `run` |
| Fused crossings? | `ncclMeshFusableRooted`, `launch_metal` (`fusing`), `fused_piece`, `metal_send_deferred`, `metal_land_deferred`; mb `mesh_rank.m:window_submit`, `metal_recording.m:MetalReplaySubmit`, `metal/layers.metal:streamed_finish_publish`, `streamed_finish_norm_add` |
| Who reads the landings, and who finds a loss? | `session_watch`, `session_targets`, `release_all`; the bridge's events (design/bridge-and-ipc.md#control) and libmesh's watch of the bridge (`mesh-dataflow.c:mesh_watch`) |
| How does a GPU-side failure reach the host? | `mesh_step_check` → violation page → `violated` → `session_async` / `session_ensure` |
| What does async-error report? | `ncclCommGetAsyncError`, `session_async` |
| Revoke, cancel, finalize, destroy, abort, exit? | `ncclCommRevoke`, `ncclMeshCommCancel`, `session_retire`, `ncclCommFinalize`, `session_kept`, `ncclCommDestroy`, `ncclCommAbort`, `session_close`, `drain`, `session_exit` |
| Issue / complete / tickets? | `launch_metal` (`issue`), `ncclMeshComplete`, `retire`, `losses_through`, `ncclMeshRetired` |
| Held flows across calls? | mb `mesh_rank.m`: `MeshRankFlows`, `adopted`, `healthy`, `MeshRankGroup`, `agreed`, `MeshRankDetach`; `tools/mesh/flows.py:plan`, `observed` |
| Bridge upgrade? | its stop (design/bridge-dataflow.md#5-failure-cancellation-and-release); `session_ensure`, `attach_live`; `mesh-dataflow.c mesh_exit_wait`, `mesh_bridge_wait` |
| torch's use? | `torch_mesh/backend.py:ProcessGroupMesh`, `_stream.mm:on_stream`, `complete`, `MeshWork`, `functional` |

## Ambiguities and hazards in the code at this HEAD

1. **An idle session finds its loss only where a call reads it.** `session_async` (`ncclCommGetAsyncError`) and
   `session_ensure` run `session_watch` and read an idle session's next position; nothing reads it between calls.
   - `healthy()`/`adopted()` (mb `mesh_rank.m`) adopt a held communicator on `ncclCommGetAsyncError`'s answer at that
     moment; a cancel after it is found by the first window's GPU landings (`land_*` AWAIT ends on `~0`, the cancel
     recorded) and the call fails after that window. The same holds for the engine's fused landing (`layers.metal`).
2. **`ncclCommGetAsyncError` waits for `session_lock`**, so it answers only once another thread's group or completion
   in the process has returned. Both `healthy()` and `window_result` (mb `mesh_rank.m`) call it.
3. **`comm->async` is sticky** (2577, 2686). Every later `ncclCommGetAsyncError` reports the first failure, though the
   library lets the communicator's next group reopen and run.
4. **Members decide adopt or make alone.**
   - Each member decides by local readings: `healthy` and cancellation.
   - `tools/mesh/flows.py:plan` keeps a name across committed calls without consulting each member's reported
     `held` (131-138, 145-155).
   - So one member can adopt a session at position N while its peer makes a new communicator of the same name (same
     slot key), opens at position 0 and waits for an agreement the adopter published long ago (`session_agree`, until a
     cancel or a revoke).
     Or the maker waits in `mesh_transfers_start` (with no bound, until a cancel or a revoke), depending on how the bridge pairs a
     new program against a running one of the same key (design/bridge-dataflow.md).
5. **A peer's skew is no failure.** A slow peer is waited for; a dead one is found by events
   (design/bridge-and-ipc.md#control). A caller that encodes and never commits holds `drain` (ncclCommDestroy,
   ncclCommFinalize, a host-path group) until it commits: its own program's fault, no transport event.
6. **A fused answer before the session.** `ncclMeshFusableRooted` reads the map and compiles the table before any
   session (`comm_read`); the session reads them again, so a map that moved between the two can make a fused answer
   disagree with the session's program, which the issue then refuses. (The digest-0 `kept` cache of this hazard's
   first form went with cc8a6f8e.) A configured form that cuts the operand is never fusable (`whole_moves`).
7. **One violation buffer for several sessions.** A completion or launch spanning groups of several sessions binds the
   first group's session's violation buffer as every non-check dispatch's gate (`launch_metal`, `ncclMeshComplete`;
   `nccl-mesh-metal.m` `record4`, `dispatch_as`), so one session's cancelled wait ends another session's later waits.
8. **Host-path cycles wrap at 2^32 positions.** `mesh_host_publish` takes `(uint32_t)at` (1295, 1722;
   `mesh-collective.c` 1259-1267), so its cycle wraps after 2^32 positions on one session; the GPU path and `landed`
   use 64 bits.
9. **torch-mesh never calls `ncclCommGetAsyncError` or `ncclCommDestroy`.** A Metal-path failure found after a group
    returned surfaces only at that communicator's next group. Sessions live until `session_exit`.
10. **`run()` busy-spins with `session_lock` held** (1680-1729). A slow peer costs a core and blocks every other
    communicator's group, `ncclCommDestroy` and `ncclMeshFusableRooted` in the process, until the peer publishes or is
    lost.
