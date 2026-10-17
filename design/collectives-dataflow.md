# Collectives data flow (libnccl-mesh)

What happens when a rank calls a collective: which thread, process and GPU does the work, and which words and
buffers it reads and writes. This is accurate to mesh `725a3dfb` and metal-microbench `6b1ea2b`. Citations are
`file:function` (line numbers at that HEAD); a bare function name is in `rdma/nccl-mesh.c`.

- Usage, build, rules a program keeps: [rdma/NCCL.md](../rdma/NCCL.md). The API contract: `rdma/nccl.h`'s header
  comment. Neither is repeated here.
- Everything inside the bridge process (pairing a program, posting a released SEND cell, writing a landing's
  completion word, cancelling words, relaying, retiring a dead process's slot, `stopping`): `design/bridge-dataflow.md`.
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
 │ progress thread, one per open session (session_open: pthread_create ─► progress_run)                           │
 │   scans completion words [signaled, target); ~0 → release_all; MESH_SILENT_NS with none landing → release_all  │
 │   (cancel_all: every word of the session ~0); reads violation words (violated); sleeps when nothing is pending │
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
| Caller | the program | every API call; bootstrap (synchronous); map reads (synchronous); a host group's `run()` spin; a Metal group's encode | `session_lock` for each group (`launch`, `launch_metal`, `ncclMeshComplete`, `ncclMeshFusableRooted`, `ncclCommDestroy`); `attach_lock` for leases and attach; `comm->guard` for opening/revoke |
| Progress thread | `session_open` (line 1465), joined by `session_close` | `progress_run`: the session's one silent-link watcher; it is on no landing's path (the GPU and the host wait on the words themselves) | `s->lock` for its waits only |
| GPU | the caller's commit | kernels the library encoded, or the caller's kernels running the library's steps | none |
| `atexit` | `session_open` registers `session_exit` once | closes every session | `session_lock` |

There are no other library threads. Client-side threads that drive the library: metal-microbench `mesh_rank.m:watch_held`
(a GCD utility-queue loop every 250 ms while communicators are held), and `MetalReplaySubmit`'s completion handlers
(Metal's threads; they touch no library state). torch-mesh calls the library on torch's MPS stream queue inside
`dispatch_sync` (`torch_mesh/_stream.mm:on_stream`, `begin`, `complete`).

**What the library reads and writes in the region:**

- **Header** (`mesh.h` `struct hdr`). The library reads `bridge_pid` (`attach_live`), `node` (`session_open` checks it
  is the rank's member), `pgsz` and `block` (the wire), `data_off` and `length` (`ncclCommRegister`'s window bounds), the
  map serial `map` (`mesh_map_wait`) and `stopping` (`ncclMeshBridgeStopping`). It rings `control`
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
  an ephemeral port.
- The host part is the first non-loopback IPv6 address with prefix `MESH_ULA_PREFIX` (`fd6d:6573:68:`, the node's
  fabric identity), else `gethostname`.
- Each rank reports `{rank, node}`, where `node` is its bridge's node number from `mesh_observe` (a read-only map of the
  region).
- Rank 0 accepts the other ranks within `DEADLINE_S` (120 s) and sends every rank the whole table. A non-root rank
  retries `connect` every 100 ms until the same deadline.
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
| `paths` | the caller's trees for the single-phase all-reduce | `ncclMeshCommInitRank` |
| `sets[2]` | `[0]` shortest-path trees (or `paths`), `[1]` packed trees | `comm_routes` |
| `chosen[(3+2n) x CLASSES]` | the default programs compiled so far, one per (collective, root, size class); freed at each `comm_read` | `program_of` |
| `entries`, `programs`, `listed` | the caller's program table | `ncclMeshCommInitRank` |
| `session`, `opening` | the open session; a session between its start and its agreement | `session_open` |
| `revoked` | set by `ncclCommRevoke`, kept until destroy | `ncclCommRevoke` |
| `async` | the last failed group's result; never cleared | `launch_metal`, `ncclGroupEnd` |
| `ops[64]` | PreMulSum scalars | `ncclRedOpCreatePreMulSum` |
| `scratch`, `spent` | the Metal 4 form's scratch when the caller gives no workspace | `workspace` |

**`comm_read` (line 567)** runs at every `session_open` and once in `ncclMeshFusableRooted` if no session has read
the map yet.

1. `routes_free(comm, 0)` drops the previous map, tree sets and `chosen`.
2. `map_read` calls `mesh_map(members)` (`mesh-dataflow.c`). It opens the region, reads the bridges' records under one
   serial and builds the whole map. The members become nodes 0..n-1 and every other node a relay. Each link's cost is
   alpha `MESH_ALPHA_US` (5 us a message) and beta `8e9 / bandwidth` ns a byte at its slowest port (`MESH_BETA_NS`
   where the bandwidth is unknown). The table is `mesh_routes`'. `map_read` retries with more room on `ENOSPC`.
3. `mesh_link_map_joined`. `ENOTCONN` fails the session with `ncclRemoteError`, naming the two members that have no
   route.
4. `comm_routes` (line 306):
   - a listed communicator does nothing more;
   - `paths` are checked against the map (`mesh_trees_check`) and become `sets[0]`; otherwise `sets[0]` is
     `mesh_trees_shortest`;
   - `sets[1]` is `mesh_trees_pack(map, eps 0.1, most 4)`.

**Held and adopted communicators (metal-microbench `mesh_rank.m`).** A call's communicators are its crossings' member
sets (`MeshRankGroup`, line 207).

- **Held map.** `held_flows()` is a process-wide name → `ncclComm_t` map. It survives across calls in one serving
  process.
- **`MeshRankFlows`** (line 164), at a call's start. It reads the record's `flows` (each `{nodes, name, hold}`). It
  destroys every held communicator if `ncclMeshBridgeStopping()`, and every held one the record does not list.
- **`MeshRankGroup`**, at the first crossing over a set. A set the record names takes the flow's name, else
  `"<call>/<ranks>"`. Then `adopted(name, count, rank)` (line 190):
  - takes the held communicator out of the map;
  - returns it only if `healthy(comm)` (line 55: `ncclCommGetAsyncError` is `ncclSuccess` and the async result is
    `ncclSuccess`) and `ncclCommCount` and `ncclCommUserRank` match;
  - otherwise destroys it.
  - Without an adopted communicator, `ncclMeshCommInitRank(count, at, NULL, NULL, 0, nodes, name)` makes one.
- **What adoption keeps.** An adopted communicator keeps its open session: its program stays paired, its rings posted,
  its positions continue from the last call's and its progress thread stays asleep.
- **`MeshRankDetach`** (line 547) holds a communicator only if:
  - its flow has `hold`;
  - its bridge is not stopping (read once at the detach);
  - the call was not cancelled;
  - `healthy(comm)` holds.

  Every other communicator is destroyed (`ncclCommDestroy`). Holding starts `watch_held`: every 250 ms it destroys every
  held communicator once `ncclMeshBridgeStopping()` is set, and exits when nothing is held.
- **No member checks with its peers.** Adopting or making is decided by each member locally (section 6).

**`ncclCommRevoke` (line 797)** runs under `comm->guard` alone, never `session_lock`, so it reaches a member blocked in
another thread's group.

- It sets `revoked`.
- A session that is opening (`comm->opening`) is retired: `session_retire` → `mesh_retire` clears the slot's `owner`
  and wakes `status[d].prepared`. The start's wait returns `ECANCELED`, and the bridge fails the peers' programs toward
  it.
- An open session is released: `release_all` cancels every word of the session, so every GPU and host wait ends, and
  sets `signaled = target` on every channel.
- Every later group fails at `launch`/`launch_metal`'s `revoked()` check.

**`ncclCommDestroy` (line 815)**:

1. revokes the communicator;
2. takes `session_lock`, so it waits behind any group of any communicator;
3. waits until no opening is in progress (`opened` cond);
4. runs `session_close`;
5. frees the communicator.

**`session_close` (line 1194)**, in order:

1. Loses the groups in flight (`inflight_lose` records each as a `losses` entry for the next completion).
2. `drain`: waits for library-made GPU programs; see section 6.
3. Stops and joins the progress thread.
4. Releases the ring and violation `MTLBuffer`s.
5. Destroys the transport.
6. `mesh_detach`: `mesh_retire` gives the slot's rows to the bridge.
7. Decrements and prunes the attach.

Its caller must have completed every window that waits on the session's words. The Metal 4 form signals nothing to the
library (`program_end` with a borrowed encoder), so `drain` cannot wait for those windows. `mesh_rank.m:run_windows`
waits for every submitted window before the detach.

**`ncclCommGetAsyncError` (line 831)** returns `comm->async` if set. Otherwise it calls `session_async` (line 1058):

- `pthread_mutex_trylock(&session_lock)`; if another thread holds it, it returns `ncclSuccess`;
- `violated(s)`, which turns violation words [0..4] into the session's failure;
- returns `s->failed`.

What it does and does not detect:

| Detected | Not detected |
|---|---|
| A launch of this communicator that failed (`comm->async`; never cleared, so it is still reported after a later session opened fine) | Anything while any thread of the process holds `session_lock` (a false `ncclSuccess`) |
| A GPU wait that reached its `polls` bound, or a position whose filler stamp disagrees with its role (violation page) | **An idle session's peer exiting, or the bridge cancelling its words.** The progress thread sleeps once nothing is outstanding (`progress_run` 1141-1144). No code scans an idle session's words or its `client->status[d]`. |
| A failure the progress thread or a revoke made: silence or `~0` while positions were pending, `walk_failed` | This node's bridge stopping. Only the next group's `session_ensure`, `ncclMeshMemLive` and `ncclMeshBridgeStopping` see that. |

**An idle held communicator does not learn that its peer process exited.** It learns at its next group:

- Host path: `run()` reads `~0`, or the watcher sees `MESH_SILENT_NS` of silence, and the group fails.
- Metal path: the GPU waits read `~0` and fall through. `mesh_step_check` skips a `~0` word, so the land kernels
  combine whatever the slot holds. The progress thread then fails the session, and the caller sees the failure only
  after the window (`ncclCommGetAsyncError`) or at the next group.

## 3. Sessions

A session is a communicator's program on its node's bridge. It opens at the first group that crosses a link
(`crosses`: elements, more than one rank, a point-to-point peer other than itself), and again after a failure.

**`session_open` (line 1386)** is called and returns with `session_lock` held:

1. Revoked → fail.
2. `mesh_map_wait(&serial, 0)` reads the map serial, then `comm_read`.
3. `attach_current` + `mesh_program_attach(&s->context, &a->context, comm->name)`: a new slot on the process's
   attach, keyed by the name.
4. Checks `hdr->node == members[rank]`.
5. The wire: `mesh_wire_of(pgsz x block, NCCL_BUFFSIZE or MESH_BUFFSIZE)`. Then `digest = map_digest`, which covers
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
   The wait is bounded by `MESH_PEER_NS + MESH_PAIR_NS + MESH_PAUSE_NS` (633 s) and ends early on `ESRCH` (bridge gone)
   or `ECANCELED` (the slot was retired: a revoke). A failure goes through `start_failure` and names the destination,
   the node where it failed and `strerror(code)`.
10. `session_agree`, then the wait time is added to `session_waits` (`ncclMeshSessionWaits`).
11. Relocks, clears `opening`, broadcasts `opened`, starts the progress thread, links `s` into `sessions` and sets
    `comm->session`.

**The agreement (`session_agree`, line 1290)** is position 0 of every channel.

- Each member writes `{MESH_PLAN_VERSION, digest, serial, node}` into slot 0 of its out ring and publishes position 0
  (`mesh_host_publish`).
- It then polls each in ring's position-0 word (`mesh_host_arrived`, `usleep(20)` between polls) until it lands, is
  `~0`, the communicator is revoked, or `MESH_SILENT_NS` passes since its own start returned.
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
| `target` (atomic) | how far the progress thread watches: `end` at each group's schedule (`progress_targets`) |
| `signaled` (atomic) | positions the progress thread has seen land |

**The wire (`mesh-collective.c:mesh_wire_of`, `mesh_wire_large`, `mesh_wire_pieces`).**

- `small` is one block: `pgsz x block`, the bridge's row.
- `slot` is `ceil(NCCL_BUFFSIZE / LAG / small) x small`, at least `small`.
- `lag = MESH_LAG = 8` (NCCL_STEPS).
- A piece takes the large channel if its bytes exceed `lag x small`. It is cut into `ceil(bytes / slot of its class)`
  positions.
- On a 16 KiB block with the default 4 MiB: small 16 KiB, slot 512 KiB, large above 128 KiB.

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
- Threads: one progress thread.
- Time: the start (the peer's program's start plus pairing) and the agreement; `ncclMeshSessionWaits` sums them.

**`session_ensure` (line 1744)** decides when a session is reopened:

- wait for another thread's opening of it;
- open a session if there is none;
- if `s->failed` (after `violated`): return the failure and close the session. This group fails; the next one opens a
  new session;
- if no group is in flight and the attach's bridge is gone: close and reopen now.

A change of the map never closes a session.

## 4. Programs

**A collective is a compiled program** (`mesh-plan.h` `struct mesh_program`): every rank's moves over parts of buffer 0
(the operand, made in place) and scratch buffers b ≥ 1.

- `mesh_program_steps(program, rank, operand)` makes a rank's steps (SEND, REDUCE, COPY; peer, round, buffer, first,
  piece).
- A point-to-point call is one step (`plan_calls`, line 1544): a SEND, or a COPY into its receive buffer.
- Routes are data: the bridges' table in `map.next`, and the tree sets.

**Choosing a program (`program_of`, line 347):**

- **A listed communicator** takes the first entry of its table with the call's `what` (and `root` for a rooted call)
  and `!below || bytes < below`. With no match, the call fails: `ncclInvalidUsage`, "no program".
- **The default:**
  1. The size class k is the least `2^k >= bytes`, k from `MESH_CLASS_LEAST` (10) to `MESH_CLASS_MOST` (34). The
     program is cached in `chosen[slot_of(what, root)][k]`.
  2. On a miss it looks in the process-wide `kept[]`, keyed by `(digest, slot, k)`, under `session_lock`.
  3. On a miss there, `mesh_choose(map, sets, 2, what, root, operand {0, 4, 2^k / 4}, wire, MESH_ALPHA_US,
     MESH_BETA_NS)` times every candidate with `mesh_program_time` and picks the fastest: both tree sets; for an
     all-reduce also `whole` (single-phase); chunks 1, 2, 4 … `MESH_CHUNKS` (64). Ties go to the first; a later
     candidate wins only if faster by more than 2^-30 of the time.
  4. `program_compile` compiles the choice along `sets[choice.set]`.
- `MESH_PLAN_VERSION` (2) enters both the digest and the agreement, so members whose planning rules differ fail at the
  session instead of mismatching later.

**Places where a formula or constant, not a table or evidence, chooses:**

| Place | Choice made by |
|---|---|
| `program_of` 357-358, 366 | rounding to a power-of-two class, and pricing the class at `2^k / 4` elements of **4 bytes** whatever the call's type and exact size |
| `mesh_map` link costs | **constant alpha** `MESH_ALPHA_US` = 5 us on every link; beta from the port's nominal speed (`8e9 / bandwidth`), or the constant `MESH_BETA_NS` = 0.1. No measured rate enters a choice. |
| `comm_routes` 307, 329 | packing constants `eps = 0.1`, `most = 4` trees a root |
| `mesh_choose` | `MESH_CHUNKS` = 64 cap; the 2^-30 tie margin |
| wire (`mesh_wire_of`, `mesh_wire_large`) | `lag = 8`; the large/small split at `lag x block`; slot = `NCCL_BUFFSIZE / 8` rounded to a block |
| `nccl-mesh-metal.m:dispatch` | 256 threads a pointwise kernel, at most 1,024 threads in one threadgroup for a block's piece, at most 2^20 units a grid |
| `ncclMeshFusableRooted` | fusable only at 2 ranks, one small slot, float16 sum (or a broadcast of any type), at most 1 scratch |

**Scratch.** `program->scratch` is the most scratch buffers a rank uses.

- Host path: `malloc` copies of the staged operand (`launch` 1865-1867).
- Metal path, each a `metal_copy` of `own` (`launch_metal` 2527-2535) into either:
  - the stream's `workspace` callback (classic form; required with a borrowed encoder);
  - the communicator's `scratch[k]` (Metal 4 form without a workspace; grown, never shrunk);
  - a private buffer the program keeps until it ends (classic form without an encoder).
- **The two-rank all-reduce** (the direct exchange, `whole`) has exactly one scratch: the rank's copy of its operand,
  which only its SEND reads while the REDUCE sums into buffer 0. The fused form makes no copy and aliases the scratch to
  `own` (line 2530), because the caller's kernel publishes from its own store.

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
   - `progress_targets`: `target = end`, waking the watcher.
   - `mesh_transfers_wake` rings the bells of the bridge's ports.
5. **`run()`** (line 1680) spins on the caller's thread, still holding `session_lock`, until every channel has
   `sent == consumed == end`. Per channel:
   - **Consume.** While `mesh_host_arrived(in, consumed)` has landed:
     - check the stamp against the role;
     - `combine_into` (REDUCE) or `memcpy` (COPY) from the in slot into `host[buffer] + offset`;
     - a piece's first position waits until its prior receive is `done`.
   - **Publish.** While `sent < consumed + LAG`:
     - `memcpy` the piece from `host[buffer]` into the out slot, or write the filler stamp;
     - `mesh_host_publish(out, sent, first_send(...))` stores the position's first SEND length, then `argument + cycle`,
       into the position's SEND cell (bridge-dataflow.md#send-length).
   - On `~0` or `s->failed`: fail, and `session_close` the session.
6. Unlock. `finish` copies `result` into `recv`, or truncates and divides for an integer `ncclAvg`.
   `target = signaled = waited = end`.
7. On failure, `ncclGroupEnd` sets `comm->async`.

### (b) Library form, Metal path (classic or Metal 4), not issued

The call is the same with an `ncclMeshStream` whose buffers are `ncclMeshBuffer`s. `launch_metal` (line 2453):

1. **Checks.** One communicator, one command buffer or encoder, one `polls`; no float64. A Metal 4 stream must have
   its encoder and residency set, and a workspace if it is issued. A classic stream with an encoder must have a
   workspace. Then `reduction`, the revoke check, lock, `session_ensure`, `plan_calls`, `metal_collect(finished)`.
2. **Program.**
   - `metal_begin4` (Metal 4, line 231): the caller's encoder, argument table and residency set, the device's argument
     ring and a leading barrier.
   - `metal_begin` (classic, line 220): the caller's command buffer; the caller's encoder (borrowed) or one made lazily.
   - Either binds the session's `violation_buffer` and the stream's `polls`.
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
   - `progress_targets`, `mesh_transfers_wake`.
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
   waits end on their words, and the progress thread advances `signaled`. Completion is the caller's command buffer
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
   `ncclMeshStream{commandEncoder, argumentTable, residencySet, fused = &steps}`, `polls 0`. In `launch_metal`:
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
   - `mesh_step_enter` (thread 0 stores the record's `sequence` into the argument ring's progress word; a bounded
     program returns early if violation[2] is set);
   - each element goes to the engine's output and, with `mesh_step_store`, into `ring_out + (p % 16) x small`;
   - `mesh_step_release`: device barrier, then thread 0 `mesh_publish`es SEND cell `p % 32` with `1 + p / 32`.
2. The bridge sees the ready word and posts the slot. The peer's bridge writes the peer's completion word `p % 32` to
   `1 + p / 32` (design/bridge-dataflow.md).
3. The landing kernel (`streamed_finish_norm_add`, flags 64), each threadgroup:
   - `mesh_step_enter`; `mesh_step_await`: thread 0 waits with `mesh_remote_wait` (or the bounded form if `polls`),
     then `mesh_step_check` checks the window's stamps (violation [0], [1]) or records a reached bound ([2..4]);
   - device barrier + system-scope fence;
   - adds the peer's row from `ring_in + (p % 16) x small` (`mesh_step_read`) to its own before the norm.
   - Nothing reads the operand buffer as the sum.

The crossing therefore costs the landing's wait inside the consumer kernel and no dispatch of its own. The library
form costs `copy_plain` (scratch), `send_small` and `land_f16`. The wire behaviour (positions, slot bytes, cells,
stamps) is the same in both forms, so two members may encode the same crossing differently without disagreeing.

## 6. Failure data flow

**Who writes `~0` into a completion word:**

- the bridge, when a link or a relay is lost, or the peer's program is lost or retired (design/bridge-dataflow.md);
- the library's `cancel_all` (`mesh.h:mesh_cancel` over every destination's range), called from `release_all`:
  - by the progress thread on silence;
  - by `violated`;
  - by `ncclCommRevoke`;
  - by `walk_failed`;
  - by `drain`'s first timeout.
- A failed session's progress thread re-cancels every `RECANCEL_NS` (10 ms) while the session stands, so a late
  landing cannot un-cancel a word a committed window spins on (`progress_run` 1141-1155).

| Event | Detected by | State written | What the caller sees |
|---|---|---|---|
| Link or relay lost, peer program failed | bridge → words `~0`; then whichever reads first: `progress_run` (if positions pending), `run()`, `session_agree`, a GPU wait | `s->failed = ncclRemoteError` ("link to node N was lost") | host: the group fails. GPU: waits fall through, kernels proceed on stale slots, failure after the window or at the next group |
| Peer silent: positions pending, none landing for `MESH_SILENT_NS` (10 s) | `progress_run` 1127 | `release_all` → `~0` | `ncclRemoteError` "landed nothing in 10 s" |
| GPU wait reached `polls` | `mesh_step_check` → violation [2] = position + 1, [3] = channel, [4] = polls; later dispatches of the session skip (`mesh_step_enter`) | `violated` → `release_all` | async error / next group: "landed nothing within N polls" |
| Calls that do not pair | host `run()` stamp check; GPU `mesh_step_check` → violation [0], [1] | host: `ncclInvalidUsage` at once; GPU: `s->failed` via `violated` | `MISMATCH` text |
| Revoke | `ncclCommRevoke` | `revoked`; retire the opening slot; `release_all` | `ncclRemoteError` "revoked", every later group |
| Map disagreement | bridge compares the READY digests (`mesh_transfers_digest`) → start fails with `ESTALE`; else `session_agree` compares | `start_failure` / agreement failure; `session_close` | `ncclRemoteError` naming nodes, serials, digests; the next group reopens on the map then |
| No route between members | `comm_read` (`mesh_link_map_joined`) | none | `ncclRemoteError` "has no route" |
| Stale leases (`ESTALE`): this process holds a lease of a stopped bridge | `attach_current` → `attaches_stale` | none | `ncclMemAlloc`, communicator, session, `ncclCommRegister`, `ncclMeshMemLive` fail with `ncclRemoteError` until every such lease is freed |
| Local bridge gone | `attach_live` (`kill(pid, 0)`) | `session_ensure` closes and reopens (no group in flight); `mesh_transfers_start` `ESRCH` | reopen, or `ncclRemoteError` |
| Peer never starts its program | `mesh_transfers_start` | none | `ETIMEDOUT` after 633 s, or a revoke |
| Encode failure mid-walk (e.g. `EOVERFLOW` argument ring) | `launch_metal` / `ncclMeshComplete` | `walk_failed`: `release_all` and `inflight_lose` on each session through the ticket | the group's result; the later completion's loss |

**The bridge's `stopping` word.**

- The library only reads it: `ncclMeshBridgeStopping` returns `hdr->stopping != 0` of the newest live attach under
  `attach_lock`. No other library path reads it.
- metal-microbench acts on it so held flows do not hold up the bridge's upgrade:
  - `MeshRankFlows` releases everything held when it is set;
  - `MeshRankDetach` holds nothing when it is set;
  - `watch_held` destroys everything held within 250 ms of it being set.
- torch-mesh never reads it.

**`drain` (line 1179)** applies before a host-path group and in `session_close`.

- It waits on `finished` for every library-signalled GPU program: classic form, library-made encoder, process-wide.
- `MESH_SILENT_NS` with nothing finishing and nothing landing → `release_all(ncclTimeout)`. A second such interval →
  `ncclInternalError` "did not finish".

## 7. Configuration operands

| Operand | Read by | Effect on the data flow |
|---|---|---|
| `NCCL_COMM_ID` (env) | `ncclGetUniqueId` | the root's `host:port`; bypasses opening a listener |
| `NCCL_BUFFSIZE` (env, bytes; ≤ 0 or unset → `MESH_BUFFSIZE` = 4 MiB) | `setting` in `session_open` and `ncclMeshFusableRooted` | the large slot = `NCCL_BUFFSIZE / 8` rounded up to a block: ring bytes, pieces per message, and the digest (members must agree) |
| `ncclMeshStream.polls` (per call) | `metal_begin`/`metal_begin4` → `p.polls` | bounds each GPU wait to `polls` loads (0: until it lands or is cancelled) |
| `ncclMeshStream.workspace`, `issue`, `fused`, `argumentTable`, `residencySet`, `commandEncoder` | `enqueue`, `launch_metal` | form (classic or Metal 4), scratch source, issue/complete, deferred steps |
| `ncclMeshCommInitRank`'s `paths`, `programs`, `node`, `name` | `comm_routes`, `program_of`, `comm_make`, `mesh_key` | trees and programs as operands; slot key = FNV-1a(name) |
| Bridge configuration (window, `pgsz`, `block`) | read from the region | the window for leases; `small` = one block |
| `ncclConfig_t`, `ncclCollConfig_t` | ignored | none (`algSelection` not read) |
| `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | `mesh_mpi.py` (`NCCL_COMM_ID` first) and torch's store | bootstrap only; the library reads none of them |
| `MESH_RDMA` | torch-mesh `setup.py` | build only |

Compile-time constants that shape the flow, none of them settable:

| Constant | Value | Meaning |
|---|---|---|
| `MESH_SILENT_NS` | 10 s | every silence bound |
| `MESH_LAG` | 8 | so `DEPTH` = 16, `POSITIONS` = 32 |
| `WATCH_NS` / `RECANCEL_NS` | 1 ms / 10 ms | watcher scan and re-cancel periods |
| `DEADLINE_S` | 120 s | bootstrap |
| `MESH_PEER_NS` + `MESH_PAIR_NS` + `MESH_PAUSE_NS` | 600 + 30 + 3 s | the start's bound |
| `MESH_PLAN_VERSION` | 2 | planning rules, in digest and agreement |
| `MESH_CLASS_LEAST` .. `MOST` | 10 .. 34 | size classes |
| `MESH_ALPHA_US` / `MESH_BETA_NS` | 5.0 / 0.1 | link costs where nothing measured them |
| `MESH_CHUNKS` | 64 | chunk-count cap in `mesh_choose` |
| `OPS` | 64 | PreMulSum operators a communicator |
| `MAX_RANKS` | 64 | the Metal lowering of all-to-all, gather, scatter |
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
| Which trees? | `comm_routes` (`mesh_trees_shortest`, `mesh_trees_pack`, `mesh_trees_check`) |
| Which program runs a call? | `program_of`, `kept`, `mesh-collective.c:mesh_choose`, `mesh_program_time` |
| What do members compare before running? | `map_digest`, `session_agree`, `mesh-call.c:mesh_transfers_digest` |
| When is a session opened or reopened? | `session_ensure`, `session_open` |
| What does a session lease, and how much? | `ring_make`, `ring_bind`, `mesh-call.c:mesh_transfers_prepare`, `mesh-collective.c:mesh_host_inputs`, `session_bytes` |
| Where is position t's slot, cell, word? | `publish`, `consume`, `channel_metal`, `cycle_of`, `landed` |
| Which positions does a group take? | `schedule_group`, `lay`, `mesh-collective.c:mesh_lay` |
| What order does the GPU encoding follow? | `order_group`, `walk`, `complete_group`, `issue_group` |
| Which kernel does what? | `nccl-mesh-metal.m` `STEPS`/`KERNELS`, `dispatch`, `dispatch4`, `record4` |
| The host path's loop? | `launch`, `session_run`, `run` |
| Fused crossings? | `ncclMeshFusableRooted`, `launch_metal` (`fusing`), `fused_piece`, `metal_send_deferred`, `metal_land_deferred`; mb `mesh_rank.m:window_submit`, `metal_recording.m:MetalReplaySubmit`, `metal/layers.metal:streamed_finish_publish`, `streamed_finish_norm_add` |
| Who watches for silence? | `progress_run`, `progress_targets`, `release_all`, `cancel_all` |
| How does a GPU-side failure reach the host? | `mesh_step_check` → violation page → `violated` → `session_async` / `session_ensure` |
| What does async-error report? | `ncclCommGetAsyncError`, `session_async` |
| Revoke, destroy, exit? | `ncclCommRevoke`, `session_retire`, `ncclCommDestroy`, `session_close`, `drain`, `session_exit` |
| Issue / complete / tickets? | `launch_metal` (`issue`), `ncclMeshComplete`, `retire`, `losses_through`, `ncclMeshRetired` |
| Held flows across calls? | mb `mesh_rank.m`: `MeshRankFlows`, `adopted`, `healthy`, `MeshRankGroup`, `MeshRankDetach`, `watch_held`, `release_held`; `tools/mesh/flows.py:plan`, `observed` |
| Bridge upgrade? | `ncclMeshBridgeStopping`; mb `watch_held` |
| torch's use? | `torch_mesh/backend.py:ProcessGroupMesh`, `_stream.mm:on_stream`, `complete`, `MeshWork`, `functional` |

## Ambiguities and hazards in the code at this HEAD

1. **An idle session hides its loss** (`progress_run` 1141-1144; `session_async` 1058-1066; `session_ensure`
   1747-1756).
   - `healthy()`/`adopted()` (mb `mesh_rank.m` 55-58, 190-204) adopt a held communicator whose peer exited or whose
     words the bridge cancelled.
   - The first window on it then runs GPU landings over cancelled words: `land_*` AWAIT ends on `~0`, and
     `mesh_step_check` skips (`nccl-mesh-metal.m` 40-48). The same holds for the engine's fused landing
     (`layers.metal` 371-373). The call fails only after that window.
2. **`ncclCommGetAsyncError` reports `ncclSuccess` whenever another thread holds `session_lock`** (line 1059). Both
   `healthy()` and `window_result` (mb `mesh_rank.m` 437-444) rely on it.
3. **`comm->async` is sticky** (2577, 2686). Every later `ncclCommGetAsyncError` reports the first failure, though the
   library lets the communicator's next group reopen and run.
4. **Members decide adopt or make alone.**
   - Each member decides by local readings: `stopping` (mb 178, 552, 79), `healthy` and cancellation.
   - `tools/mesh/flows.py:plan` keeps a name across committed calls without consulting each member's reported
     `held` (131-138, 145-155).
   - So one member can adopt a session at position N while its peer makes a new communicator of the same name (same
     slot key), opens at position 0 and waits for an agreement the adopter published long ago (`session_agree`, 10 s).
     Or the maker waits in `mesh_transfers_start` (up to 633 s), depending on how the bridge pairs a new program
     against a running one of the same key (design/bridge-dataflow.md).
5. **`MESH_SILENT_NS` is also a bound on peer skew, not only on dead links:**
   - `progress_targets` sets targets when a group is launched or encoded (1776, 2555). A peer that enters the matching
     group, or whose GPU reaches it, more than 10 s later fails the group. This applies to host groups, encoded
     windows and issued (async) groups alike (`progress_run` 1127).
   - `session_agree` (1297-1306) gives a peer 10 s from this rank's own start returning, though the peer publishes its
     agreement only after its own start returns, which waits on all of its destinations (`mesh-call.c` 208-223) and may
     take `MESH_PAIR_NS` (30 s) a pairing. With three or more members or relayed routes, a live peer can trip it.
   - `drain` (1182-1184) times out after 10 s on a slow peer, or on a caller that encoded but did not commit.
6. **Pre-session `kept` key.** `ncclMeshFusableRooted` runs `program_of` before any session has set `comm->digest`
   (2636-2646 → 363). Choices made before any session are cached under digest 0 and shared across communicators. The
   session discards them (`comm_read` 569), so a fused answer can disagree with the session's program, which the issue
   then refuses (2484-2486, 2563).
7. **One violation buffer for several sessions.** A completion or launch spanning groups of several sessions binds the
   first group's session's violation buffer as every non-check dispatch's gate (2491-2492, 2593-2595;
   `nccl-mesh-metal.m` 278, 319). With `polls > 0`, one session's reached bound stops another session's publications.
8. **Host-path cycles wrap at 2^32 positions.** `mesh_host_publish` takes `(uint32_t)at` (1295, 1722;
   `mesh-collective.c` 1259-1267), so its cycle wraps after 2^32 positions on one session; the GPU path and `landed`
   use 64 bits.
9. **Stale comment.** `nccl-mesh-metal.h` 43 says "nccl-mesh.c gives it at least 1 here". `launch_metal` passes
   `polls` unchanged (2491), and 0 is accepted (rdma/NCCL.md "Bounded waits").
10. **torch-mesh never calls `ncclCommGetAsyncError` or `ncclCommDestroy`.** A Metal-path failure found after a group
    returned surfaces only at that communicator's next group. Sessions live until `session_exit`.
11. **`run()` busy-spins with `session_lock` held** (1680-1729). A slow peer costs a core and blocks every other
    communicator's group, `ncclCommDestroy` and `ncclMeshFusableRooted` in the process, for up to 10 s.
