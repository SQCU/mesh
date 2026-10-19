# The bridge's data flow

Where each byte and word of the mesh bridge goes: who writes it, who waits on it, and what happens when it fails. The
reasons for the design are in [bridge-and-ipc.md](bridge-and-ipc.md) (the bridge), [recovery.md](recovery.md) (loss,
recovery and the five-second rule) and [heterogeneity.md](heterogeneity.md) (link costs and shares); this page repeats
none of them.

Conventions. The page describes the code at `MESH_VERSION` 125. A file name with no directory is in `rdma/`; the
bridge's functions are in `mesh-link.c` and `mesh-program.c` (`mesh-bridge.h` declares what the two share). A citation
is `file function`. "Word" means an 8-byte atomic in the shared region. "Block" is `hdr.block * hdr.pgsz`. The bridge
always starts with `block` = 1 (`main`: `mesh_layout(&geometry,pg,1,...)`), so a block is one 16 KiB page on Apple
silicon, and a row is one block. "The bound" is mesh.h `MESH_PLATFORM_NS`, five seconds (recovery.md#five-seconds,
whose labels B1-B16 and P4 this page cites).

## 1. The processes and memory

### Processes and threads

| process / thread | started by | does | waits on |
|---|---|---|---|
| `rdma/mesh-flow`, the bridge (one per node) | the launchd agent `io.mesh.bridge` (`bin/mesh-user.sh` writes it with `ProgramArguments` = mesh-flow and the canonical configuration plist, `KeepAlive`, `RunAtLoad`, stderr to `~/Library/Logs/io.mesh.bridge.log`); the keeper loads it at boot | everything below | — |
| the bridge's event loop (its main thread) | `main` | ports, control sockets, the map, flows' state, client slots, the record's available memory, and every five-second conclusion (§2, "The event loop's turn") | `kevent64` with a 250 ms timeout; `EVFILT_USER` (`wake`), sockets, `NOTE_EXIT` on client pids, signals, kernel network events |
| `control_watch` thread | `main` | turns a change of `hdr.control` into a `wake()` of the event loop | `os_sync_wait_on_address(&M->control)` |
| port p's work queue `mesh.port` (serial dispatch queue) | `ports_found` | every verbs call except post and poll: open the device, register the window, make, connect, move to the error state and (at a pairing) destroy the link's lanes; prepare a flow's records and cells (no verbs call); let go of a flow's lanes | its jobs (`port_open`, `link_job`, `flow_job`, `flow_stop`) |
| port p's two progress threads `mesh.rdma.receive` (side 0) and `mesh.rdma.send` (side 1), QoS user-interactive, each on an alternate signal stack | `port_threads`, at the port's first listed flow or its link's connect; never joined (they end with the process) | side 1 polls SEND cells' `ready` words, posts SENDs on their lanes, takes every lane's SEND completions and posts the keepalive's frame; side 0 posts a listed flow's first receives, takes every lane's receive completions, stores landing words (only raising them) and reposts receives | spin while anything moves; parked on `hdr.bell[p]`, or on `port.guard` while it is raised (`port_progress`) |
| finder queue `mesh.ports` | `main` | reads the device list (`ports_find`) and the host name (`host_read`) | — |
| terminations queue `mesh.terminations` | `terminations_watch`, at the start | IOKit's `kIOTerminatedNotification` for `AppleThunderboltRDMAPeerInterface`: every window held the bound, the port halted (`peer_terminated`, B9b) | — |
| a client process (a rank, anything linked to libnccl-mesh or libmesh) | its user | attaches to the region, leases, binds, prepares, starts programs, publishes and consumes | its own landing words |
| libnccl-mesh session progress thread | `nccl-mesh.c session_open` | the session's watcher: a landing that reads `~0`, its bridge gone (pid or boot), or a destination's program refused or failed cancels the session's words (`release_all`) | `pthread_cond_timedwait`, 1 ms (`WATCH_NS`) between scans while positions are pending; else a condition wait |
| the client's GPU work | the client's command buffers | publishes into SEND cells; waits on landing words | `mesh.metal mesh_remote_wait_bounded`, at most the stream's polls (`NCCL_MESH_POLLS` at most) |

### The shared region `/mesh0`

The bridge creates the region at start: `shm_unlink`, `shm_open(MESH_NAME, O_CREAT, 0666)`, `ftruncate`, `mmap`
(`main`). It lays the region out with `mesh.h mesh_layout`. Every section is page-aligned and the arena is
block-aligned:

```
offset 0      struct hdr
planes_off    3 row bitmaps (MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE), one bit per row
arena_off     arena occupancy bitmap, one bit per arena page
page_off      struct mesh_page_entry[rows]   mapping (slot index << 32 | arena page), address (region offset)
buffer_off    struct mesh_buffer[rows]       owner (client owner word), closed, pages (lease extent, head row)
link_off      struct mesh_link_info[8]       the port table
client_off    struct mesh_client[64]         client slots
delivery_off  struct mesh_delivery[rows]     device_input (region offset of a landing word), device_stride
data_off      ARENA = the window: arena pages, registered on every port's device
```

Every time a client or reader maps the region, it checks the region's `magic` and `version` (`MESH_VERSION`) and checks
that `bridge_pid` is alive (`mesh-dataflow.c mesh_open`). A region of another version is refused with `EINVAL`; a dead
bridge gives `ESRCH`.

**Header words** (`mesh.h struct hdr`):

| word | written by | read / waited by | meaning |
|---|---|---|---|
| `magic`, `version`, `pgsz`, `block`, `rows`, `node`, offsets, `length`, `boot` | bridge at start (`main`); `magic` last | clients at attach | geometry; `node` is a 31-bit fold of `gethostuuid`; `boot` is the bridge's start time on the wall clock |
| `bridge_pid` | bridge (start; 0 at stop) | every client wait (`mesh_gone`); libnccl-mesh's watcher (`bridge_gone`) | liveness of the bridge |
| `serial` | `mesh_slot_take` (clients and the bridge's own leases) | — | slot generation counter |
| `control` | clients (`mesh-call.c mesh_transfers_start`, `mesh-dataflow.c mesh_retire`); bridge (at its stop) | the bridge's `control_watch` | "look at the client slots now" |
| `map`, `map_off`, `map_nodes` | bridge (`map_put`, `map_grow`) | `mesh-dataflow.c mesh_records_read` (seqlock), `mesh_map_wait` | the map: an odd `map` value means a write is in progress |
| `health` | bridge (`health_put`) | `mesh_nodes_health` | a second seqlock for record changes that leave the map as it is |
| `busy`, `idle_since` | bridge (`busy_tell`) | `mesh_busy`, `mesh-stat` | flows; wall-clock ns at which the count last reached 0 |
| `stopping` | nothing | nothing | — |
| `lanes[8][10]` | clients claim a lane's word (CAS 0 to the owner word of the slot whose program rides it: `mesh_lanes_claim`) and release it (CAS on the word observed: the owner word to 0, owner\|DIRTY to DIRTY: `mesh_lanes_release`); bridge (`lanes_take` makes it owner\|DIRTY as a flow takes it, `lanes_release` clears DIRTY once it has reset the lane, `lanes_reap` lets go of a gone owner's with its DIRTY kept, `link_make` clears DIRTY) | `lanes_leased` (`program_take`), the clients' init (`mesh_lanes_free`) | a lane's lease (§3) |
| `bell[8]` | clients (`mesh_transfers_wake`); bridge (`port_bell`, `send_wake`) | port p's progress threads, when parked | a futex-like word; this is not a hardware doorbell |

**A client slot** (`mesh.h struct mesh_client`, 64 slots):

| field | written by | meaning |
|---|---|---|
| `owner` | the client takes the slot with a CAS (`mesh_slot_take`); `mesh_retire` clears it | `(generation << 38) \| (slot << 32) \| pid`; 0 means vacant |
| `configured` | the client sets it with a CAS 0→owner (`mesh_transfers_start`); `mesh_retire` sets 0 | the program is started |
| `key` | the client (`mesh_slot_take`: FNV-1a of the execution's name, `mesh_key`) | the execution; it is the same on every member |
| `depth`, `cyclic`, `digest`, `targets`, `target_count` | the client (`mesh_transfers_prepare`, `_cyclic`, `_digest`) | ring depth; whether the program is cyclic (M30); program digest (0 = none); the program's publications sorted by row |
| `ports` | bridge (cleared at acceptance in `clients_turn`; a bit set per flow's port in `program_take`) | the bells `mesh_transfers_wake` rings |
| `tx[d]` (64), `struct mesh_tx` | the client (`mesh_transfer_bind`, `_prepare`, `mesh_host_inputs`, and its init's lanes) | the destination `node`; the lists region offset `orders` with `length[4 queues x 2 directions]` and `room`; `cells` (region offset of this destination's first SEND cell); `cancel` (region offset of the cancellation record); `slots`, `invocations`, `count`, `once`; `lane[2]`, the link lanes its queues 0 and 1 ride |
| `status[d]`, `struct mesh_port_info` | bridge (`status_prepared`, `status_failed`, `program_paired`, `program_take`, `programs_stop`); the client zeroes it at take | `phase` (`MESH_PAIRING`, `MESH_PAIRED`, `MESH_STOPPED`), `code`, `domain`, `failed` (the node where the failure happened), `prepared` (set to the owner once the program is paired or failed; libnccl-mesh's watcher reads it, `start_refused`) |

**Inside a client's leases** (arena pages owned through rows). The bridge writes these as well (bridge-and-ipc.md#region):

- **SEND cells** (`mesh.h struct mesh_send`, 256 B, 128-aligned). One column per slot of each SEND transfer: one
  cell for each active invocation, plus one terminal cell. All of them sit in one lease from `mesh_transfers_prepare`,
  followed by the program's `struct mesh_target`s. The client writes `request.wr_id = 1` (the producer argument)
  at prepare time and `ready` at each publication. The bridge writes `pair`, `span`, `request` (the chunk-0 work
  request), `successor`, `chunks`, `queue` (the lane it rides) and `port` (`sends_configure`). Where a program sends
  the rows it receives (a member that carries on what lands), the bridge's receive thread also writes `ready` (§4).
- **Landing (completion) words and the cancellation record** (`mesh-collective.c mesh_host_inputs`). There is one
  `uint64_t` for each received chunk of each invocation, per destination, each destination starting on a 16-word
  boundary. The words of destination d form `cancel->ranges[d]` (`mesh.h struct mesh_cancellation`); every
  `tx[d].cancel` names that one record. `mesh_host_inputs` writes each received row's `mesh_delivery` (the word of
  invocation 0 and the per-invocation stride `8*count*chunks`). The bridge's receive thread stores into the words (§4).
  The words are cancelled with `~0` (§5).
- **Rings and payload**: the client's own sections (`mesh-call.c mesh_section_create`). The NIC writes received
  payload into them by DMA. Senders write payload into them (`memcpy` on the host, Metal kernels on the GPU).
- **Transfer lists** (`struct mesh_transfer`, 48 B), leased per destination (`mesh-call.c mesh_lists_grow`).

**The window and the bridge's two mappings.** The arena is mapped twice in the bridge. `M` maps the whole region.
`wire.data` is an alias of the arena alone, at a 4 GiB-aligned base (`mesh-verbs.h wire_map`), because
`tbt_post_recv` loads only the low 32 bits of an SGE address. Every SGE address is in `wire.data` (`wire_span`). A
client maps the region once, at its own address; its leases are region offsets. A GPU sees the client's SEND-cell lease
and its landing words as `MTLBuffer`s created with `newBufferWithBytesNoCopy` over the client's mapping (`mesh-metal.m
mesh_metal_transport_create`).

**The bridge's own arena leases**: the map's records (`map_grow`), and every port's link block (`link_reserve`: one
block a port, the keepalive's frames' source and sink, for all `MESH_PORTS`, taken at start).

### Verbs objects and who owns them

| object | per | made | ended | owner |
|---|---|---|---|---|
| context, PD | device (port) | `mesh-verbs.h device_up` on the port's queue (`port_open`) | never by a call: `down_device` forgets the handles where the device failed with its link active; the kernel releases them at exit | `struct port.device` |
| MRs | device: the window in 1 GiB extents (`MESH_REGION`), `IBV_ACCESS_LOCAL_WRITE` only | `device_up`, before any QP exists; checked against `max_mr` (`EFBIG`) | the kernel, at exit | `device.regions[]` |
| the link's lanes: `MESH_LINK_LANES` (10) UC QPs on 2 CQs (receive `completion`, send `sent`), the CQs made once a context | port | `link_job(LINK_JOB_MAKE)` on hearing the peer's hello: `pair_up` (INIT); `LINK_JOB_CONNECT`: `queue_receive` (RTR), `queue_send` (RTS) for every lane | reset in place (`pairs_reset`: RESET, INIT, RTR, RTS) at a flow's stop (B6 (a)); moved to the error state (`pairs_error`) at a concluded link's flush (`LINK_JOB_FLUSH`, B6 (b)), and at the next pairing's `link_make` before `pairs_down` destroys each (B6 (c); one whose destruction fails kept for the next, B16; the CQs kept and drained), or forgotten (`pair_forget`) where the context is gone | `port.link.provider` |
| a flow | — | no verbs object: its rings and streams post on its lanes' QPs | — | — |

All QPs are `IBV_QPT_UC` (`mesh-verbs.h pair_make`), with MTU 4096, a GRH with hop limit 1 and the port's GID
(`device_gid` takes the first non-empty entry). Each QP's first PSN is `psn + q`, with `psn` random. Capacity per QP is
`min(max_qp_wr, QD=4095)`, which must be less than `max_cqe`. The usable QPs of a device are `max_pairs = max_qp - 1`.
Only `IBV_WR_SEND` and `post_recv` are ever posted, each with one SGE and each SEND signaled. The code has no RDMA WRITE
or READ, because TB5 offers only SEND and RECEIVE (bridge-and-ipc.md#lanes, TN3205).

### Diagram

```
 node A                                                                          node B
 ┌─ client process ───────────────┐   ┌─ mesh-flow (io.mesh.bridge) ───────────────────┐       ┌─ mesh-flow ─┐
 │ host: memcpy payload -> ring   │   │ event loop: kqueue (sockets, NOTE_EXIT,        │       │   (same)    │
 │       mesh_host_publish ->     │   │   signals, EVFILT_USER <- control_watch,       │       │             │
 │       cell.ready = 1+cycle     │   │   the terminations queue); 5 s conclusions     │       │             │
 │ GPU:  mesh_publish -> ready    │   │ port p: work queue (verbs objects, guarded)    │       │             │
 │       mesh_wait_bounded <- word│   │   send thread:  poll cell.ready -> post SEND ──┼─UC────┼─> RECV ─┐   │
 │ mesh_transfers_wake -> bell[p] │   │   recv thread:  poll CQ -> store word, repost  │ TB5   │ recv thread │
 │ mesh_transfers_start -> control│   │   lane 0: keepalive frames both ways           │       │ stores word │
 │                                │   │ control socket TCP [fe80::%enN]:18519 ─────────┼─TCP───┼─ control    │
 └──────────┬─────────────────────┘   └──────────┬─────────────────────────────────────┘       └──────┬──────┘
            │ mmap                                │ mmap (M) + wire alias of the arena (4 GiB-aligned)  │
            ▼                                     ▼                                                     ▼
 ┌──────────────────────────────── /mesh0 on node A ───────────────────────────────┐     /mesh0 on node B
 │ hdr: magic version node boot bridge_pid control map health busy lanes bell[8]    │
 │ planes | arena bits | page entries | buffers | link table[8] | client slots[64]  │
 │   slot: owner configured key depth cyclic ports digest tx[64] status[64]          │
 │ delivery[rows] (row -> landing word, stride)                                      │
 │ ARENA (window, MR-registered whole on every device):                              │
 │   client leases: rings, SEND cells+targets, landing words+cancel record, lists    │
 │   bridge leases: map records, link blocks                                         │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

## 2. The control plane

### Ports and peers

1. **Ports are found.** `ports_find` lists `rdma_*` devices on the finder queue, and `ports_found` gives each a port
   index in the order found, its quiet window that of any termination heard meanwhile. This runs at start, once the
   terminations are watched and IOKit's registry is quiet (`IOKitWaitQuiet`, at most the bound), and again where
   `network_heard` sees an `en<N>` interface attach or its link come on that no port has, once every window has ended
   and no call is stuck (`find_wanted`, B11). The port table entry is `link_set(port, 0, MESH_STOPPED, 0)`.
2. **The device opened** (`port_open` on the port's queue, guarded), where the port is `opening`: at the start, at
   `KEV_DL_LINK_ON` / `KEV_DL_IF_ATTACHED`, at a hello heard while it is not open (B16; not while its open's job is out
   or its registration was refused), and as the call it was concluded around returns (B12); never while its quiet
   window stands (B9), while any port's call is stuck (`stalled`), or while it has a control socket. It joins the
   hellos' group on its interface, then `device_up` (context, port active, PD, window MRs) and `device_gid`. A context
   that fails with the port active is forgotten with every object made on it (no verbs call) and the device opened
   afresh. In `port_opened`: a registration refusal leaves the port down until its link next comes on; another failure
   until its link comes on or a hello is heard on it; on success the port records `max_pairs`, LID, GID and bandwidth,
   sets the phase to `MESH_PAIRING` and says hello at once.
3. **Hello** (`port_hello`): an open port with no control socket and no job sends a UDP `HELLO` (node, boot, block
   bytes, port index) to `[ff02::1%enN]:18519`, every `MESH_PAUSE_NS` (3 s), a paced emission that decides nothing; a
   port whose paired link was lost says it at once.
4. **Hello heard** (`hello_heard`). A hello of another magic, of this node, or on a port with a control socket is
   dropped. The higher node answers with its own HELLO. The lower node dials the higher's TCP listener at the address
   it heard from, its `heard_at` stamped at the dial; the listener's accept is the higher's port's control socket where
   the port is open and has none, its `heard_at` stamped there. Both ends `port_greet` (non-blocking, `SO_NOSIGPIPE`)
   and send a `HELLO` over the socket. The first message on a control socket must be a `HELLO` with an equal block size
   (and the dialed node's number), otherwise `EPROTO` (`port_heard`). Once greeted, the bridge sets
   `link_set(..., MESH_PAIRED)` and `pairings++`, sends every record in its map to the new peer, and makes its link's
   lanes (`link_turn`).
5. **A connection to the listener with no port's scope** (loopback, a LAN address) is closed.

### The link's lanes

`link_turn` runs each event-loop turn on every port. Each job runs on the port's queue, the guard raised (the progress
threads park until it is lowered; a raise not held within the bound halts the port and the job makes no change), with
the peer's numbers copied at its dispatch and its pairing compared as it starts and between its steps (B10):

| state | event | job | next |
|---|---|---|---|
| `LINK_DOWN` (greeted, no job, past its quiet window, no call stuck) | its link block leased (`ENOMEM` concludes the link where the window has none) | `LINK_JOB_MAKE`: the last pairing's lanes to the error state and destroyed, or forgotten with the context; every DIRTY cleared, the lanes' `generation` moved; 10 QPs in INIT | `LINK_MADE`, LINK sent |
| `LINK_MADE` | the peer's LINK heard (`link_heard`: 10 lanes, once, else `EPROTO`) | `LINK_JOB_CONNECT`: every lane RTR, then every lane RTS, each gate's capacity set, the keepalive's receives posted, the progress threads started | `LINK_UP`: every lane served, the port table's `lanes` 10, a new record |
| `LINK_UP` | lanes found failed, or not reset at a flow's stop (`broken`) | none: `port_lost(EIO)`, the control socket closed | both ends pair afresh |
| any | `flush_at` reached (its conclusion's window ended), no termination window, no call stuck | `LINK_JOB_FLUSH`: every lane to the error state; the `generation` moved where every QP moved (B6 (b), B7) | — |
| any | a job's failure | none | `port_lost(code)`, paired again on hearing the peer (B16); a refused registration's port down until its link comes on |

A job whose pairing was lost (`job_pairing` no longer `pairing`) makes no call and is dropped as it ends.

### The map

- Each node has one record (`mesh.h struct mesh_record`): node, name, boot, sequence, window, memory, model, the port
  table (`mesh_link_info[8]`: peer, phase, code, bandwidth, pairings, pairs, max_pairs, lanes) and health (`booted`,
  `available`).
- `record_changed` builds this node's record with `sequence+1` and sends `RECORD` on every greeted port. It runs on
  any `link_set` that changes peer or phase, as a link comes up, on a rename (`host_taken`), and from `health_turn`,
  which samples the available memory (`kern.memorystatus_level` of `hw.memsize`) every `MESH_HEALTH_NS` (10 s).
- `record_retell` sends this node's held record, at its current sequence, on every greeted port each liveness turn
  (`MESH_PAUSE_NS`, 3 s): a receiver drops it as no newer than its newest, and the control socket is heard within the
  bound (B3).
- `record_heard` keeps the newest record per node in `known[]`, apart from the map (RFC 2328 §13). It takes a record
  only if the record is newer by (boot, sequence), then forwards it on every other greeted port. A copy of a pruned
  node is put back only where a path reaches that node again; a node's later boot is always newer.
- Writing the record:
  - If the links (each port's peer, phase and bandwidth), name, boot, window, memory and model are unchanged
    (`record_links_same`), `health_put` writes it under the `health` seqlock, and the map serial does not move.
  - Otherwise `map_put` writes it under the `map` seqlock (odd while writing). `map_put` prunes nodes that no paired
    path reaches (BFS over `mesh_joined`), logs each pruned node, and writes only when the map differs.
- Routes: a member reads the records' whole map (a link a cable at the slowest of its ports' bandwidths,
  `mesh-dataflow.c mesh_join`) and takes `mesh-collective.c mesh_routes`'s path. Every member holding the same records
  computes the same route.
- Digests: the **program digest** (`client.digest`, set by `mesh_transfers_digest`) is the client's; the bridge reads
  none of it. libnccl-mesh's `map_digest` (FNV-1a over the planner version, wire, members, routes, link costs, trees,
  programs and the communicator's name) travels in the session's position 0, whose agreement compares it with the
  cables each end leased (`session_agree`).

### Messages on a control socket

`struct message` carries `magic` (`XMAGIC` = `0x4d595048 + MESH_VERSION + (1 << 12)`, the last the wire's revision, so
bridges of different versions or revisions drop each other: `control_heard` raises `EPROTO`), `kind`, `node`, `count`,
`boot`, `code`, `block`, `port`, `bytes`, `lid`, `gid` and `pair[10][2]`. A payload of `bytes` (at most 64 MiB, else
`EPROTO`) follows the message. Sends queue in the port's buffer, and more than `MESH_UNSENT` (64 MiB) unsent loses the
link with `ENOBUFS` (`port_say`). Each whole message stamps the port's `heard_at` (B2). No message names a program or a
flow.

| kind | sender, when | receiver's effect (`port_heard` → `message_heard`) |
|---|---|---|
| HELLO | UDP: `port_hello` (every 3 s while unpaired), `hello_heard` (higher node's answer); TCP: `port_greet` | greets the port (peer, peer port index, LID, GID) |
| RECORD | `record_changed`, `record_retell` (every 3 s), `record_heard` (forwarding), a newly greeted port (every held record) | `record_heard` |
| LINK | `link_turn` once the link's lanes are made (`count` 10: each lane's QP and PSN) | `link_heard`: every lane, once a pairing, before the connect (`LINK_JOB_CONNECT`); anything else `EPROTO` |
| FAREWELL | `main` at stop, with `code` = the signal, or -1 for a binary change | `port.parted = 1`, the code kept, `port_lost(ECONNRESET)` at once: nothing posted toward the peer again, logged "parted" |
| any other | — | `port_lost(EPROTO)` |

### The event loop's turn

Each turn (`main`), at most 250 ms apart:

1. `ports_found` where the finder read the devices.
2. `terminations_turn`: each new termination logged; each port a termination halted concluded (`port_lost(ETIMEDOUT)`,
   no flush: B9b); each port a provider's fault was absorbed on lost (`open = 0`, `port_lost(ENETDOWN)`, no flush: P4).
3. `ports_stalled` (B12): a port whose open's job, link job or a flow stop's move is out past the bound, or whose
   progress threads did not hold for a settle or guard within it (`unguarded`), is concluded once: `open = 0`,
   `port_lost(ETIMEDOUT)`, no flush, `programs_stop(ETIMEDOUT, port)` (the words of its flows whose stops have not run
   `~0`, from this thread); `stalled` set while any call is so out; a port stays `stuck` until the call returns and
   its threads tell the conclusion's epoch, then opens again.
4. The devices read again where `find_wanted`, no read is out, no call is stuck and every window has ended.
5. Each port: `port_silent` (B2: nothing heard for the bound, no keepalive frame heard for the bound since this
   pairing's first, or a frame in flight as long → `port_lost(ETIMEDOUT)`); `gates_turn` (B4: a served program lane
   whose gate retired nothing for the bound with frames posted → its holder `flow_lost(ETIMEDOUT, 2)`); an open's end
   (`port_opened`); `port_open`; `port_hello`; `link_turn`.
6. `flows_turn`, `health_turn`, `host_taken`.
7. Every `MESH_PAUSE_NS` (the liveness turn): `host_read`, `record_retell`, and the binary's inode and mtime read
   (a change stops the bridge).
8. `clients_turn`, then `kevent64` for at most 250 ms.

### Client slots: `clients_turn`

`clients_turn(liveness)` runs on every event-loop iteration. A liveness turn (`alive`) comes every 3 s.

1. A flow whose client owner is no longer owned, or whose slot's `configured` is not that owner, gets
   `flow_end(f)`: `flow_stop(code 0)`, nothing told.
2. For each slot: on a liveness turn, an owner whose pid is gone (`mesh_gone`) is retired (`mesh_retire`). A new
   owner gets a kqueue `NOTE_EXIT` watch; if the pid is already gone (`ESRCH`), it is retired. In the event loop,
   `NOTE_EXIT` retires every slot of that pid.
3. A slot with `configured == owner`, not yet in `accepted[s]`, and with no flow of an earlier owner of the same slot
   running or stopping (one held for lanes its stop could not reset, its stop run, touches no word of the slot), is accepted:
   `ports = 0`, and `program_take(client, owner, d)` runs for each destination with transfers.
4. Once no flow references a non-owned owner (`zombie == 0`, a flow held for lanes not reset among them: B7),
   `mesh_retired_release` returns closed rows to the window, and `mesh_reclaim` frees their arena pages.

**Retiring** (`mesh-dataflow.c mesh_retire`, called by the client's `mesh_detach`, by `nccl-mesh.c session_retire` for
a revoke, or by the bridge):

1. CAS `owner` to `owner | MESH_RETIRING`.
2. Set `configured = 0` and every list `length` to 0.
3. On every row the client owns, set `closed = 1` if it heads a lease, and clear its `ROW_OWN` bit.
4. Set `owner = 0`: the slot is vacant at once.
5. `mesh_notify(control)`.

## 3. Flows

A flow is one program's traffic toward one far node on one port, for one stripe of the program's invocations
(`struct flow`): its client's slot (`owner`) and destination (`destination`), its queues (at most two, of one slot
each), queue k on lane `lane[k]` of the port's link, and the stripe j of n cables it carries. It holds no queue pair.

**Leases.** A program's lanes toward a node are a pair its client leased at its communicator's init, the same pair
index at both ends of each link (mesh-call.h `mesh_lanes_choose`, `mesh_lanes_claim`), named in its slot's
`tx[d].lane`. `program_take` (event loop) finds, among the cables paired with the node (`cables`, in `cable_rank`
order), those on which `lanes_leased` holds: the link up, each lane's lease word exactly the program's owner word (no
DIRTY: no flow holds it) and the lane served. The flows are stripe j of n on the n such cables. It refuses the
destination as a value (`program_refused`: its status, then `~0` into its cancel words) where the node is this one or
none (EINVAL naming it), the program is not a lane pair's shape (`tx.slots` other than 1, no queue or more than two,
two queues on one lane: EINVAL naming this node), no cable reaches the node (EHOSTUNREACH naming it), or no cable
carries its lease (ENOLCK naming this node); a flow that cannot be made fails the program (`program_fail`: a port's
table full, E2BIG; a lease let go meanwhile, ENOLCK).

**Making.**

| step | where | what |
|---|---|---|
| `flow_make` | event loop (`program_take`) | the flow, its table index, its lanes' lease words owner→owner\|DIRTY by CAS (`lanes_take`, B5) and their `holder` set to it |
| `flow_job` | port's queue | `receives_configure` (the records, no posting); `sends_configure` (the cells' chains, each cell's `pair` its lane's QP and `queue` the lane); `port_list` (in the port's table, `progressing`) |
| `flow_arm` | the receive thread's first look at the listed flow | each ring's first records posted (`ring_advance`): every record of a lane is posted by the thread that takes its completions |
| `flow_done` | event loop | a job's error fails the flow (domain 1); else `running`, and `program_paired` once every flow of the program and destination runs |

**Client status.** `program_take` sets `status[d].phase = MESH_PAIRING`. `program_paired` sets `MESH_PAIRED`, `code =
0`, then `prepared = owner` and wakes any waiter. `status_failed` sets `MESH_STOPPED`, `code`, `domain` and `failed`,
then `prepared = owner`. The client's start, `mesh_transfers_start`, waits on none of it (C1): libnccl-mesh's watcher
reads a `MESH_STOPPED` status with a code (`start_refused`), and a refusal's `~0` ends a wait on its words.

**The QP budget.** Per device `max_pairs = max_qp - 1` (10 on the TB5 devices); `port.committed` counts the link's 10
lanes, from their first making; no flow takes one. Four communicators ride a link at once (lanes 2 to 9 in pairs).

**`flow_free`** (event loop, once no job runs and `flow_stop` reported `stopped == 1`, or `stopped == 2` and the link's
lanes' `generation` moved past the flow's `lane_epoch`): unlinks the flow, `port.flows--`, frees the table index (the
next `flow_make` at that index bumps its generation), `flow_count--`, then `busy_tell`.

### Progress threads (`port_progress`, `port_pass`)

**Passes.**
- One pass reads the link's `served` once and serves only the listed, progressing flows whose every lane is in it
  (B8).
- One pass of side 1 runs `flow_send` on each such flow (its streams' SENDs on their lanes, each through its lane's
  gate); then, while the link serves any lane, `link_send`: the link's SEND completions, each naming its lane by its
  QP's number (`link_lane`; a lane not served, or a QP of an earlier pairing, drops its own): a program's lane's frames
  retired through the lane's gate, lane 0's keepalive frame no longer in flight; an error status or a failed poll stops
  the whole link (`link_failed`); then lane 0's next frame where it is due (`link_alive`, B1).
- One pass of side 0 arms every such flow not yet armed (`flow_arm`), then `link_receive` while the link serves any
  lane: a program's record landed (`record_landed`, to the lane's holder; a lane with none drops what comes), a
  keepalive frame counted (`alive_heard`) and its receive posted again; an error stops the whole link.
- `port_pass`, `flow_send`, `flow_arm`, `record_landed`, `link_send` and `link_receive` are one translation unit
  (`mesh-link.c`), inlined into `port_pass` as in the measured build (its only calls are the verbs' function pointers
  and cold failure paths).

**Parking.**
- A thread with something to look for (`any`) that moves nothing for `MESH_IDLE_NS` (100 ms; counted in
  `MESH_IDLE_PASSES` = 4096-pass batches) parks on `bell[p]` with a 1 ms timeout.
- With nothing to look for, it parks at once, with a 3 s timeout.
- A bell ring wakes it immediately.
- While the port's guard is raised (odd), it tells the guard it holds at and parks on the guard word until the guard
  moves (B13), telling the epoch as it wakes.

**Table changes.** Flows enter the port's table with `port_list` (release store, then a bell ring). They leave with
`port_unlist` and `port_settle`: `epoch++`, then the raiser parks until both threads tell `seen >= epoch`. After that,
no thread reads the flow. Past the bound the port is halted (`unguarded`, concluded by `ports_stalled`) and the flow
stays listed, kept whole, no lane of it reset (the link concluded). A thread stores `seen` only when the epoch it read differs from the one
it last told, so the line of `high`, `epoch` and `seen` stays clean in both cores between table changes.

## 4. The life of one transfer

This section follows one libnccl-mesh large-class position `at` from rank A (node a) to rank B (node b) over a flow on
their communicator's lanes. Each step names the word written, its writer and its reader. The constants:
- `LAG = MESH_LAG = 8`, `DEPTH = 16` (ring slots), `POSITIONS = 32` (invocations of the cyclic program;
  `nccl-mesh.c`).
- Position `at` is invocation `t = at % 32` in cycle `c = at / 32`, and uses ring slot `at % 16`.
- The large slot is `NCCL_BUFFSIZE / 8` rounded to a block (512 KiB, `mesh-collective.c mesh_wire_of`), which is 32
  chunks of 16 KiB and 128 frames of 4 KiB. The small slot is the communicator's `small` (one block where none is
  given; the pair's decode passes 3,072 bytes, one frame), on a ring whose slots start a block apart (the transfer's
  `invocation_pages`, its `bytes` the slot's). The large class rides queue 0 and the small class queue 1
  (`ring_bind(..., !large, ...)`). One slot per queue means ring index = queue, on lane `tx.lane[queue]`.

**Setup, once per session** (`nccl-mesh.c session_open`):
1. The communicator's init leased a lane pair toward b on every cable both bridges pair (NCCL's bootstrap; rdma/NCCL.md
   "Communicators"); the session names it in `tx.lane`.
2. The client leases its rings (`ring_make`: `DEPTH * slot` bytes each way).
3. It binds the out ring as a SEND and the in ring as a receive toward b (`mesh_transfer_bind`, binding identity
   `1 + 2*rank + large`, `invocation_pages = slot / pgsz`, depth 16).
4. It prepares (`mesh_transfers_prepare(slots=1, invocations=32, depth=16)`): cells, targets, and each transfer's
   `first`.
5. It sets the digest, makes the program cyclic, and creates the Metal transport, which calls `mesh_host_inputs`
   (landing words, the cancellation record, the deliveries).
6. It starts the program (`mesh_transfers_start`); the bridge runs §3 with no message to b.
7. On both ends, `sends_configure` turns each SEND cell into a chain of chunk work requests over its invocation's ring
   slot. The chunk address is `transfer_row` = `row + ((t - begin) % depth) * invocation_pages / block + chunk`, and
   each chunk's SGE comes from `chunk_span`, in `wire.data`. Each stream is linked as t = 0..31, and the last cell's
   successor is the first, because the program is cyclic.
8. `receives_configure` builds the receive ring: one record per chunk per invocation. Each record's word comes from
   `receive_input`.
9. Position 0 of every channel is the members' agreement (`session_agree`). Data positions start at 1.

**Sending (node a):**

1. **Payload.** The host copies the piece into `ch->sending + (at % 16) * slot` (`nccl-mesh.c run`). The GPU form
   copies into `ring_out` (`metal_copy(METAL_SEND)`), or, for a small piece, copies and releases in one kernel
   (`metal_send_small`). A position with nothing to send carries `METAL_STAMP(at)` in its first 8 bytes (a filler).
   It still costs a whole slot on the wire.
2. **Client credit.** The host publishes position `at` only once `at < consumed + LAG` (`run`). A rank's consumption
   of the peer's position `at - LAG` is what lets it publish `at`, so the reverse direction's positions carry the
   head credit.
3. **Release.**
   - Host: `mesh-collective.c mesh_host_publish(ctx, ch->out, at)` finds the row's targets (`mesh_publication_prepare`
     → `mesh_row_targets`). It stores `argument + c` = `1 + c` (release order) into `ready` of the cell at
     `targets[i].cells + 256 * t`.
   - GPU: `mesh.metal mesh_publish(cell, ch->argument + cycle_of(at))`, after a system-scope fence. The cell is at
     publication-buffer offset `ch->cell + (at % 32) * 256` (`nccl-mesh.c`, the walk that issues `metal_publish`).
4. **The bridge notices.** There is no interrupt and no kqueue event on the data path. Bridge a's send thread polls
   `ready` in `flow_send`: with the stream at `cell` and `stream->cycle`, a cell is released once
   `ready > stream->cycle`. A parked thread looks again within 1 ms while the flow is listed, or at once when the
   group's start rang the bell (`mesh_transfers_wake`, called by `nccl-mesh.c session_run`, by the Metal issue path,
   and by `ncclMeshComplete`). `mesh_host_publish` itself rings no bell.
5. **Posting.** For each chunk: if the lane's gate has room (frames in flight `posted - retired` plus this request's
   `ceil(len/4096)` frames, at most `gate.capacity` = `min(send_capacity, the lane's share of the sent CQ)`), post a
   copy of the prepared request with `IBV_SEND_SIGNALED` and `wr_id = lane << 48 | posted frames` on the lane's QP
   (`cell->pair`, `cell->queue` the lane), one request per chunk; a post refused fails the flow (domain 1). The cell's
   own `request.wr_id` stays 1, the producer's argument. When the chain is done, `stream->cell = successor`; on
   reaching the first cell again, `stream->cycle++`.
6. **SEND completion.** `link_send` polls the link's `sent` CQ, names the lane by the QP's number and sets its gate's
   `retired` from the `wr_id` (a completion whose `wr_id` top byte is not its lane's reset count is of a SEND posted
   before the lane's reset, dropped: `lane_stale`). An error status stops the whole link (`link_failed`): every lane
   served no more, each lane's holder failed (`flow_lost(status, 2)`), the link concluded. The device completes every SEND (`mesh-verbs.h struct
   mesh_queue`).
7. **Hardware credit.** TN3205's credit flow control holds each SEND until the peer has posted its matching receive
   (`sends_configure`). The bridge has no software credit protocol on a lane. A gate that retires nothing for the
   bound, with frames posted, fails its flow (`gates_turn`, B4).

**Receiving (node b):**

8. **Receives posted ahead.** The receive thread's first look at the listed flow posts each ring's first records
   (`flow_arm`); then `ring_advance` posts record n once record `n - behind` has landed. `behind` spans the
   transfer's ring depth less one, divided by the stripes; for this session on one cable that is 15 invocations.
   Posting also stays within `capacity` = `min(receive_capacity, the lane's share of the completion CQ)` frames.
   Gating uses landings and capacity alone. Nothing in the bridge waits for the local reader to consume a slot, or for
   an onward SEND to complete, before posting into that slot again. Reuse is safe only under the clients'
   `LAG`/`DEPTH` credit (step 2).
9. **DMA.** The NIC writes the chunk into the posted SGE: b's ring slot `(at % 16)`, chunk k, inside b's window.
10. **Landing.** `link_receive` polls the link's `completion` CQ, names the lane by the QP's number and skips non-RECV
    opcodes; a completion of a lane with a holder goes to it (`record_landed`): `record = wr_id`, then
    `ring->landed++`, then it raises `*record->input` to `1 + record->cycle` (a compare-exchange that never lowers a
    word, so a cancellation's `~0` stands), and then calls `ring_advance`. `receive_input` chose the word:
    - **(a) a client row with a delivery inside this program's cancellation range for that destination**: the
      chunk's own word, at `device_input + device_stride * u + 8 * index * chunks` (index = slot where all slots share
      the rows);
    - **(b) no such word, at the record's last chunk, where the receiving program publishes that row** (a member that
      carries on what lands, a node between two members on a route: metal-microbench
      output_data/structural-20261018/DESIGN.md decision 1): the SEND cell `ready` of the row's publication, at
      `targets[index].cells + 256 * u`, with `release = 1`. Then `send_wake` rings the cell's port's bell if that send
      thread is parked. This is the index hand-off: the member copies nothing, and its send thread posts the same rows
      onward (steps 4-7 on it).
    - **(c) otherwise**: the record's private `sink`.
11. **The client observes the landing.** A reader waits on the last chunk's word, because chunks of one receive land in
    order on one lane (`mesh_host_arrived`, `mesh_metal_receive_prepare`).
    - Host: `mesh_host_arrived(ctx, ch->in, at)`, an acquire load. Landed means `word != ~0 && word >= 1 + c`
      (`nccl-mesh.c landed`).
    - GPU: `mesh_remote_wait_bounded(inputs + ch->word + (at % 32) * ch->word_stride, c, polls)` spins with a
      system-scope fence until `word >= 1 + c` (`mesh.metal`), for at most `polls` loads (W3a); past them the position
      is written into violation words [2..4]. `~0` also satisfies the wait, and the caller tells the three apart.
12. **Consumption.** The host copies or combines from `ch->receiving + (at % 16) * slot`; `METAL_STAMP` mismatches
    are role violations. This frees the client to publish `at + LAG` (step 2).

## 5. Failure and cancellation data flow

### The `~0` landing word

`mesh.h mesh_cancel(m, cancel, d)` stores `UINT64_MAX` into every word of `cancel->ranges[d]` that is still 0. In a
cyclic program it stores into every word that is not already `~0`, because a reader of a later cycle would wait past a
landed `1 + c`. It also sets `cancel->requested = 1`. Every spin in `mesh.metal` and every host check treats `~0` as
"cancelled". Two parties cancel:
- **The bridge**: `flow_stop(code != 0)` for the failed flow's destination, while the owner is still owned;
  `program_refused` for a destination it refused, after its status; `programs_stop` for every program at the stop, and
  for a port's programs whose stops wait behind a stuck call (B12).
- **libnccl-mesh**: `cancel_all`, for every destination of the session (`release_all`), on a landing that reads `~0`,
  on a bounded wait's violation, on its bridge gone (`bridge_gone`, W3c), on a destination's program refused or failed
  (`start_refused`), on a revoke, and on a walk's failure. A cancelled word stays `~0`, since a landing never lowers a
  word.

### `flow_fail` and `flow_stop`

`flow_fail(f, code, domain, where, lost)` (event loop): `ended = 1`; `flow_stop`; every other flow of the same owner
and destination (its other cables) failed alike. Nothing is told the peer: its program ends by what ends it at its own
end.

`flow_stop`:

1. On the event loop: if `code != 0` and the owner is still owned, `mesh_cancel` that destination's words, then
   `status_failed` (phase `MESH_STOPPED`, `code`, `domain`, `failed = where`, `prepared` stored).
2. On the port's queue, `lanes_release`: the flow's lanes' `holder` cleared, then `port_unlist` (once neither progress
   thread reads it, no completion of its lanes is taken as its own). Unless the stop is its link's loss (`lost`), and
   while every lane of it is served, no call of any port is stuck and no termination's window stands, each of its
   lanes is reset in place under the guard (`pairs_reset`: RESET drops the receives its ring posted ahead and any SEND
   not completed, with no completion; INIT, RTR, RTS toward the same peer queue pair with the pairing's numbers; the
   reset stamped `moving_at` for B12), its reset count moved (`lane_stale` drops a completion of an earlier request),
   its gate emptied (`retired = posted`) and its `MESH_LANE_DIRTY` cleared. A lane not reset (the link lost or not
   served, a window, the guard not held, a step refused) is `broken`: the event loop concludes the link.
3. The census line of each queue (its lane's gate and its ring, `reset` or `not reset`); `stopped` 2 where a lane was not
   reset (its rows held until the lanes' `generation` moves: B7), else 1.

Domains: 1 = a post, ring or job error (`flow_lost(...,1)`, `flow_done`); 2 = an error completion status, a failed
SEND poll, or a SEND held the bound (B4); 3 = a failed receive poll; 4 = a program-level failure or a link's loss.

### How each failure reaches the clients

| event | bridge path | what the local client sees | what the peer side sees |
|---|---|---|---|
| a post refused | progress thread `flow_lost` (CAS `failed`, `progressing = 0`, `wake`), then `flows_turn` → `flow_fail` | its words for that destination become `~0`; status `MESH_STOPPED`, code and domain, `failed` = this node | nothing told: its SENDs toward this end are held, and fail at the bound (B4) |
| a failed completion or poll, on any lane | `link_failed`: `served = 0`, every lane `broken`, each holder `flow_lost`; `link_turn` concludes the link (`port_lost(EIO)`), and both ends pair afresh on hearing each other | every holder's words `~0` | the link lost, then paired again |
| a SEND held the bound (B4) | `gates_turn` → `flow_lost(ETIMEDOUT, 2)` → `flow_fail` | its words `~0`, ETIMEDOUT | nothing told |
| the peer's program ended (`mesh_detach`, exit, revoke) | nothing here: the peer's flow stops at its end and resets its lanes there | this end's SENDs toward it are held and fail at the bound (B4); its other waits end by that or its caller's revoke | — |
| a program not a lane pair's shape, no lease on any cable, no cable, a table full | `program_take`: `program_refused` (EINVAL, ENOLCK, EHOSTUNREACH: its status, then `~0`) or `program_fail` (E2BIG, ENOLCK) | the code, as its status and its words' `~0` | — |
| a link lost: its socket's end or error, a send's failure, `EPROTO`, `KEV_DL_LINK_OFF` / `IF_DETACHED` (`ENETDOWN`, the port down until its link comes on), a FAREWELL (`ECONNRESET`), five seconds unheard (`ETIMEDOUT`, B2) | `port_lost`: its quiet window (B9) and, where the link was connected, its flush at the window's end (B6 (b)); `link_lost` (`served = 0` before the socket closes, the pairing moved, no verbs call); the socket closed; `flow_fail(code, 4, where = peer, lost)` on every flow of the port; `link_set(PAIRING` or `STOPPED)` → RECORD → map serial; a paired port says hello at once | words `~0`; status naming the peer; libnccl-mesh fails `ncclRemoteError` ("the link to node N was lost") and cancels all its destinations; the next session reads the new map | the same, on the other end (its socket's end, or its own five seconds) |
| a program's stop with receives posted ahead or SENDs not completed | its lanes reset in place at its stop (`pairs_reset`, B6 (a)); DIRTY cleared once reset; earlier completions dropped by reset count; where no reset can be made, the link concluded | — | the same |
| a link job fails (making or connecting) | `link_turn`: `port_lost(code)`, paired again on hearing the peer (B16); a refused registration's port down until its link next comes on | as a link lost | the link lost |
| a verbs call out the bound, or a guard not held within it (B12) | `ports_stalled`: `open = 0`, `port_lost(ETIMEDOUT)`, no flush, `programs_stop(ETIMEDOUT, port)`; no port dispatches a driver call while any is out; the port opens again once it returns | words `~0`, ETIMEDOUT | the link lost |
| the peer interface terminating (B9b) | `peer_terminated`: every port's and the exit's window held the bound, the port halted; `terminations_turn`: `port_lost(ETIMEDOUT)`, no flush | words `~0`, ETIMEDOUT | — |
| the provider's doorbell faults (P4) | `fault_heard`: one page mapped at the faulting page, the port halted; `terminations_turn`: `open = 0`, `port_lost(ENETDOWN)`, no flush; the bridge does not exit | words `~0`, ENETDOWN | the link lost |
| the peer's bridge stops | peer: FAREWELL, then close; here: `parted = 1`, `port_lost(ECONNRESET)` at once | `ECONNRESET`, logged "parted" | — |
| this bridge stops (a signal, or its binary changed: at once) | `main`: per port FAREWELL queued, the socket closed, `closing = 1`; `programs_stop(ESHUTDOWN)`: every program's words `~0` and its status, stored from the main thread; `bridge_pid = 0`; `mesh_notify(control)`; the exit held to the latest window, at most the bound past the stop's start, link events read meanwhile (B9, B9b); `atexit` `shm_unlink`, the kernel releasing every verbs object and thread. No queue, thread or verbs call is waited on | every program's words `~0`, status `ESHUTDOWN` naming this node. Later waits see `bridge_pid` 0 (`ESRCH`); libnccl-mesh's watcher cancels a session that waits (`bridge_gone`). A lease of the old region fails the next libnccl-mesh lease or session (`ncclMeshMemLive`, `attaches_stale`) | FAREWELL → `ECONNRESET` |
| a client process dies | `NOTE_EXIT`, or the liveness `mesh_gone` → `mesh_retire` → next turn `flow_end`; its leases let go at the next `link_turn` (`lanes_reap`, DIRTY kept) | — | nothing told |

**A changed binary.** On a liveness turn where the binary's inode or mtime changed (`stat(argv[0])`), the bridge stops
at once, as at a signal: its programs fail as values and launchd starts the new binary. `make install` and
`bin/mesh-user.sh` rename a new binary into place, and `mesh-user.sh` gives the binary a new file when only the
configuration changed, so a configuration change also fires this; a build alone (`make all`) installs nothing
(recovery.md#install).

## 6. Configuration operands

**The bridge's operands.** The bridge has one operand: the canonical plist that `bin/mesh-user.sh` writes from the
consumer's JSON, TOML or YAML document through `bridge_configuration.py load`, to
`~/Library/Application Support/io.mesh.bridge/configuration.plist`. Nothing has a default. `configuration_read` reads
up to 64 KiB of it.

| key (document) | canonical | effect on the data flow |
|---|---|---|
| `purpose` | `purpose` | none (required, for people) |
| `window_gib` / `window_percent` (exactly one) | `window` (bytes; a percentage is resolved against the node's memory on the node that installs it) | the arena is `window / pgsz` pages = rows (`mesh_layout`); every lease, ring, cell, landing word, map record and link block comes from it; a window outside [1 page, 2^32 pages] is refused at start; every device registers it whole, and a device with fewer than `ceil(window / 1 GiB)` MRs refuses it (`EFBIG`), its port down until its link next comes on |
| `keepalive_us` (optional) | `keepalive` (ns) | lane 0's frame pace while a flow of the port is listed (B1); none: the idle beat alone |
| (the document's stem) | `name` | logged at start only |

Changing the plist takes effect only when the bridge restarts. `mesh-user.sh` gives the binary a new file on a
configuration change, so the binary-change stop (§5) applies.

**Fixed in the source, not operands** (each changes the data flow):

| constant | value | where | what it changes |
|---|---|---|---|
| `MESH_PLATFORM_NS` | 5 s | `mesh.h` | every five-second conclusion (B2, B4, B12, the settle and guard waits), each quiet window (B9, B9b), the stop's hold, IOKit's quiet wait at start; a fifth of it the idle keepalive beat (B1) |
| `MESH_PAUSE_NS` | 3 s | `mesh.h` | a port's hellos while unpaired, liveness turns (the record retold, the binary read), the 3 s timeout of a progress thread with nothing to look for |
| `MESH_IDLE_NS` / `MESH_IDLE_POLL_NS` | 100 ms / 1 ms | `mesh-bridge.h` | spin time before parking; poll period while parked |
| `MESH_HEALTH_NS` | 10 s | `mesh-bridge.h` | the available memory's sample and its record |
| `MESH_LINK_LANES` / `MESH_FLOWS` / `LINK_ALIVE_RECEIVES` | 10 / 256 / 256 | `mesh.h`, `mesh-bridge.h` | a link's lanes; flows a port; lane 0's standing receives |
| `MESH_PORTS` / `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` / `MESH_ORDERS` | 8 / 64 / 64 / 4 / 4096 | `mesh.h` | region tables |
| `MESH_UNSENT` | 64 MiB | `mesh-bridge.h` | unsent control bytes before a link is dropped |
| `QD`, `MESH_REGION`, `MESH_BANK` | 4095, 1 GiB, 4 GiB | `mesh-verbs.h` | QP depth cap, MR extent, wire alias alignment |

**Operands of the clients that shape the bridge's traffic:**
- `NCCL_BUFFSIZE` (environment, default `MESH_BUFFSIZE` 4 MiB; `nccl-mesh.c setting`) sets the large slot through
  `mesh_wire_of`.
- `ncclMeshStream.polls` bounds GPU waits (0, or more than `NCCL_MESH_POLLS`: `NCCL_MESH_POLLS`).
- `ncclMeshCommInitRank`'s `small` sets the small slot; metal-microbench's call record names it with each crossing
  set's bootstrap id and name (`flows`: docs/elastic.md#call-delivery).

## 7. Where to look

| question | function |
|---|---|
| Who creates the region, and what is in it? | `main`, `mesh.h mesh_layout` |
| How does a client attach or take a slot? | `mesh-dataflow.c mesh_attach`, `mesh_slot_take`, `mesh_program_attach` |
| Where do leases come from? | `mesh-call.c mesh_section_create`, `mesh-dataflow.c mesh_rows_alloc`, `mesh_arena_alloc`, `mesh_lease_return` |
| How does a communicator lease its lanes? | `mesh-call.c mesh_lanes_free`, `mesh_lanes_choose`, `mesh_lanes_claim`, `mesh_lanes_release`; `nccl-mesh.c comm_init` |
| How is a transfer declared? | `mesh-call.c mesh_transfer_bind`, `mesh_lists_grow` |
| Where are SEND cells and targets laid out? | `mesh-call.c mesh_transfers_prepare` |
| Where are landing words and the cancel record laid out? | `mesh-collective.c mesh_host_inputs` |
| How does a start reach the bridge? | `mesh-call.c mesh_transfers_start`; `control_watch`, `clients_turn`, `program_take` |
| How is a peer found and paired? | `ports_find`, `ports_found`, `port_open`, `port_opened`, `port_hello`, `hello_heard`, `port_greet`, `port_heard`, `link_turn` |
| How is the map written and read? | `record_changed`, `record_retell`, `record_heard`, `map_put`, `health_put`; `mesh-dataflow.c mesh_records_read` |
| What does each message do? | `port_heard`, `message_heard`, `link_heard` |
| How does a program get its lanes and cables? | `program_take`, `lanes_leased`, `lanes_take`, `cables` |
| How does a publication become a SEND? | `mesh-collective.c mesh_host_publish`; `mesh.metal mesh_publish`; `flow_send`, `sends_configure` |
| When are receives posted? | `receives_configure`, `ring_advance`, `flow_arm` |
| Which word does a landing store? | `receive_input`; `record_landed` |
| How does a member hand a landing on? | `receive_input` (`release`), `send_wake` |
| How does the client see a landing? | `mesh-collective.c mesh_host_arrived`; `mesh.metal mesh_remote_wait_bounded`; `nccl-mesh.c landed`, `run` |
| How do the link's lanes work? | `link_job`, `link_make`, `link_connect`, `link_turn`, `link_heard`, `link_send`, `link_receive`, `link_alive`, `link_failed`, `lane_lost` |
| When do progress threads sleep or wake? | `port_progress`, `port_pass`, `port_list`, `port_unlist`, `port_wait`, `guard_raise`, `mesh-call.c mesh_transfers_wake` |
| Which five-second conclusion fires where? | `port_silent` (B2), `gates_turn` (B4), `ports_stalled` (B12), `port_lost` (B9, B6 (b)), `peer_terminated` and `terminations_turn` (B9b, P4), `fault_heard` (P4) |
| What happens on failure? | `flow_lost`, `link_failed`, `lane_lost`, `flow_fail`, `flow_stop`, `lanes_release`, `flow_free`, `port_lost`, `link_turn` (the idle cycle) |
| Cancellation? | `mesh.h mesh_cancel`; `program_refused`, `programs_stop`; `nccl-mesh.c cancel_all`, `release_all`, `progress_run` |
| A client exits? | `clients_turn`, `flow_end`, `lanes_reap`, `mesh-dataflow.c mesh_retire`, `mesh_retired_release` |
| Upgrade or stop? | `main` (the liveness block and the stop) |
| The QP budget? | `link_turn`, `mesh-verbs.h pair_up`, `device_up` (`max_pairs`) |
| The configuration? | `bridge_configuration.py load`, `configuration_read`, `bin/mesh-user.sh` |

## The loopback fabric

The operator: "our tensor parallelism code should be fast both on days when the 5 computers are a ring and also on days
when the 5 computers are a star" (metal-microbench docs/principles.md section 4), and "i cannot and will not precommit to
a specific ordering or routing!" (2026-10-03). Before the Studios arrive, the only way to run the bridge and the
library's trees on 3 to 8 nodes is N bridges on one host. The fabric that did that was restored from mesh 800fe6bc^
(`rdma/loopverbs.c`, the host executor `rdma/nccl-mesh-host.c`, the `mesh-flow-loop` and `host` targets), whose last
readings were the NCCL matrix on an 8-node line 2,346 of 2,346 and line 8, shuffled ring 8, star 8 and torus 2x4 104
of 104 each, and which found the 4-node ring+tree deadlock (c65445c) and the cycle-0 defect (c5a4674), which "nothing
at two nodes can show". They are separate build targets of the resident bridge's source; the deployed `mesh-flow` is
the same source without `MESH_LOOP_FABRIC`.

**loopverbs** (`rdma/loopverbs.c`): the ibverbs calls the bridge makes, over a shared segment (its name the loop
configuration's `fabric`): a byte ring a queue pair, its number the ring's, held while the queue pair lives (each making
moves the ring's generation). A SEND is copied into its destination's ring and completes as it is delivered (one to a
queue pair gone or made again is dropped and completes, as UC drops it); the owner's polls drain its ring into its
receives in the order posted, a SEND waiting for its receive as TB5's credit flow control holds it, `IBV_WC_LOC_LEN_ERR`
for a SEND longer than its receive; a move to `IBV_QPS_ERR` flushes the posted receives and SENDs
(`IBV_WC_WR_FLUSH_ERR`), as `pairs_error` expects; 10 queue pairs a device (TB5's); a SEND at most half a ring, 4 MiB,
the port's `max_msg_sz`; one link-local GID a port. Deviations from 800fe6bc^: the segment's name from the bridge's
configuration (`loop_fabric`), not an environment variable; device names `rdma_loop0` to `rdma_loop63`; a queue pair's
ring taken by compare-exchange of its `used` word, so numbers are reused only once free (800fe6bc^ took the next of
1,024 modulo, clobbering a live ring past 1,024 makings); the dropped SEND; the flush; rings of 8 MiB (1 MiB).

**A loop bridge** (`make -C rdma mesh-flow-loop`, `rdma/.build/loop/mesh-flow-loop CONFIGURATION`): mesh-link.c and
mesh-program.c built with `MESH_LOOP_FABRIC` and loopverbs in place of `-lrdma`. Its configuration (the canonical
property list, written by `mesh-loop up`, not by `bridge_configuration.py`) adds `loop`: `region` (its shared memory
object, in place of `/mesh0`), `node` (its number, in place of the host UUID's fold), `fabric`, and `ports`, each a
loopverbs `device` and its cable's two loopback ports (`listen`, this end's; `peer`, the other end's).
`configuration_read` refuses one without them (`loop_read`). Its ports are those devices alone, in that order
(`ports_find`); a port's turn opens its device and the cable's lower port dials the higher's listener, which accepts it
as the port's control socket (the event loop's `TAG_LOOP`), in place of a link's hello and the lower node's dial; the
HELLO, records, map, lanes, flows and five-second conclusions are the deployed bridge's, without IOKit's terminations
and quiet wait; its listener and hello socket take any port. Its name is its configuration's (`host_read` reads no host
name).

**A loop client** (`make -C rdma host`): `rdma/.build/host/libmesh.dylib` (libmesh with `MESH_LOOP_FABRIC`:
`mesh_loop_bind(region)` names the bridge a process attaches to, before its first attach) and
`rdma/.build/host/libnccl-mesh.dylib` (nccl-mesh.c with `MESH_HOST_EXECUTOR` and nccl-mesh-host.c in place of
nccl-mesh-metal.m and mesh-metal.m: every function of nccl-mesh-metal.h run on the CPU in the walk's order, transcribed
from the kernels, its waits polls of the words the bridge stores, every wait bounded (the program's polls, and at most
`MESH_PLATFORM_NS`), so the Metal path's walk runs with no GPU kernel; the Metal 4 form's own steps are refused as the
classic form refuses them). Deviation from 800fe6bc^: HEAD's Metal half (bounded waits, the violation words, combine3,
counted all-to-all, split-KV partials, retain and contents) transcribed; the arithmetic is nccl-mesh.c's own
(`nccl_mesh_combine`, `nccl_mesh_premultiply`, `nccl_mesh_truncdiv`, hidden, where `MESH_HOST_EXECUTOR` is defined).

**The driver** (`rdma/.build/host/mesh-loop`, rdma/mesh-loop.c; metal-microbench b9dbc43^ tools/nccl_demo.py `loop`'s
part that needs no Python):
- `mesh-loop up FAMILY N WINDOW_MIB DIRECTORY`: a line, ring, star, complete map or a torus of two rows, of N (2 to 8)
  nodes, every cable two loopverbs devices and two free loopback ports; each node's configuration written, its bridge
  started (posix_spawn, its log in DIRECTORY), and the call returns once every bridge's map holds N nodes and its own
  cables are paired (`mesh_nodes_health`, `mesh_observe`, within 60 s).
- `mesh-loop relay DIRECTORY BYTES`: every ordered pair, one after another, a program of one transfer of BYTES from
  the one to the other, every word checked at the receiver, each end's wait at most the bound.
- `mesh-loop nccl DIRECTORY COUNT host|walk`: N ranks, all-reduce, all-gather, reduce-scatter, broadcast and reduce
  from every root, and a send/receive around the ring, COUNT float32 each, every value exact, on the host path or the
  Metal path's walk on the host executor.
- `mesh-loop down DIRECTORY`: SIGTERM to every bridge, the fabric unlinked.

**Reproduction targets** (800fe6bc^, its readings with the GPU walk on the host executor): the NCCL matrix on an 8-node
line 2,346 of 2,346; line 8, shuffled ring 8, star 8 and torus 2x4 104 of 104 each; ring 5 with routes 46 of 46. The
matrix itself (every datatype, op and root, metal-microbench b9dbc43^ tools/nccl_demo.py) is the consumer's driver's to
restore on these targets; `mesh-loop nccl` is its float32 core. The integration check: `mesh-loop up line 8 512 D`,
then `nccl D 65536 host` and `nccl D 65536 walk`, then the same on star 8, ring 5 and torus 8.

## Routed readiness

This is the numerical executor's (`mesh-call.h`), not the bridge's. Scatter-add and an expert layer's dispatch and
combine route each update to a destination known only at run time (metal-microbench docs/principles.md section 4,
"Routed experts and engrams are one partition": "unrouted slices contribute exact zeros, so they are neither computed
nor, where placement allows, loaded"). mesh 45902d8e (2026-09-14, "Delete sparse routing and segmented indexed-add
machinery") removed the readiness half of that from the library with its caller; it is restored here on the numerical
executor, where every function fires as its arrivals come (`mesh_call_progress`): the routing directory, so an owner
waits on the producers routed to it alone, and active segments, so a slot past a count produced at run time is omitted
(Megablox's bounded grid; mesh e1acaf7e^ design/active-segments.md). The sort that writes the directory and the
segmented sums that read it stay the caller's functions (the operator's separation of 2026-09-14); the executor owns
only when each may run.

**The calls** (`mesh-call.h`, each before `mesh_calls_prepare`):
- `mesh_route_create(calls, worker, owners, ordinals, offsets, candidates, n, consumers)`: the directory is three
  `uint32` vectors (`struct mesh_route_vector`: a section and a byte offset) a function writes each generation (its
  sections a slot an index): `owners[k]` candidate k's consumer (`MESH_ABSENT` or past `consumers`: none),
  `ordinals[offsets[c] .. offsets[c + 1])` consumer c's candidates; the candidates are n sections (each a producer's
  partial, or received), a slot an index. Its events run on `worker`.
- `mesh_route_attach(function, route, c)`: the function is consumer c, on the route's worker. The route is one of its
  arrivals a generation, beside its own waited inputs; it binds the candidates it reads as inputs past its dependency
  count (operands, unwaited).
- `mesh_route_producers(route, producers, n)`: candidate k's producer function (NULL: none), which must be active and
  write candidate k (checked at prepare).
- `mesh_call_active(function, count, slot)`: the function's call of a generation runs where `slot < *count` (a `uint32`
  of a section among its waited inputs); past it, it is omitted.

**At prepare** (`mesh_calls_prepare`, `mesh_routes_layout`): every route's directory sections (`meta`, the distinct
sections of its three vectors) and candidates become arrivals on the route's worker, a cell or a landing a slot as any
waited input's (`struct mesh_use`), each naming a route event (its call index `route_base` plus the event's number,
`route_base` the program's call records' count; `UINT32_MAX` where the program has no route, so HEAD's call path is one
compare more). A consumer's `pending` counts one arrival a route. An active function's call records hold the address of
their count's word and their slot (`struct mesh_call` `count`, `slot`), and every publication of theirs must be a
route's arrival cell (else `EINVAL`).

**As it runs** (`mesh_call_progress`, on the route's worker):

| event | where | what |
|---|---|---|
| an active call's arrivals all came, its slot past the count | `mesh_call_fire` → `mesh_call_omit` | nothing submitted; each of its publications stored `argument + cycle` with bit 63 set (`MESH_OMITTED`); its generation's event completed (`mesh_instance_complete`) |
| a directory section arrives | `mesh_route_arrival` | its bit in the index's state of the cycle's parity (two states an index, as a cycle's events may come while the one before finishes); once every section came, `mesh_route_prepare` |
| the directory read | `mesh_route_prepare` | checked: each consumer's span within the candidates, each of its ordinals owned by it, and every owned candidate in its owner's span; each consumer's routed candidates not yet come counted (`remaining`), a consumer with none fired (`mesh_route_deliver`: its call's `pending - 1`, fired at 0) |
| a candidate arrives | `mesh_route_arrival` | marked (1 produced, 2 omitted); once the directory was read, its owner's `remaining - 1`, fired at 0; an unowned one is discharged as it comes, never waited for |
| the index's cycle done | `mesh_route_reset` | the directory came, every candidate came and every consumer fired: its state laid out afresh |
| a directory that breaks the rules, or a routed candidate omitted | `mesh_route_refuse` | the generation fails as a value (`MESH_RESULT_FUNCTION`, the first consumer's identity, `EPROTO`), and every consumer fires so the call records stay in step (45902d8e waited for good) |

**Deviations from 45902d8e^**, each for the executor's interfaces:
1. The old routes ran on the presence-driven executor (`mesh_execution_route`, rows' PRESENT and READ bits, a
   dispatch queue); here a route's events are the executor's own arrivals, its state a count a consumer and a mark a
   candidate.
2. The directory is three contiguous vectors in sections; `mesh_route_vector`'s strided views (`row_stride`,
   `column_stride`) and `mesh_algebra_route_table` (a GPU table of the candidates' addresses for the caller's kernel)
   are not restored: a consumer reads its candidates as its own operands.
3. An omitted producer's disposition is the omitted mark on its arrival (one word), not a disposition row; an active
   function's outputs may feed routes alone (45902d8e^ asked the same: "count and every indexed selector/range map ...
   within one actual produced output region").
4. Every route section is a slot an index (each generation writes its own directory); a constant directory is refused.

**Reproduction target** (45902d8e^ measurements/lowering-2026-09-13/active-omission-provenance.json and
expert-provenance.json; "no ... speedup claim"): 1,025 updates to 17 destinations in chunks of 513 and 512 (four
features): 34 partial slots where an unrouted grid takes 1,025; over the four occurrences (normal, changed routing,
all invalid, normal again) 50 partial launches and 86 omissions of the 136 configured slot occurrences, every
destination exact; the expert test (forward, dX, dW, ragged tiles, an empty expert) at 206.2 invocations/s paired
(653.8 CPU, 222.7 Metal). The caller (the sort and the segmented sums as functions of a program, and the expert
program on the pair) is to be written against these calls; metal-microbench tools/torch_parallel/ep.py
(`all_to_all_single` with host-read splits) is what it is measured against.
